"""Native replayed independent-observation likelihoods.

Every evaluation reconstructs existing analytic row objectives on bounded
batches and sums values, gradients and Hessians. It does not average fitted
batch models. The global sample/design contract lives in ``replay_sample``.
"""
from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares
from openecon.engines.execution import qr_factor
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.models import ModelSpec, ResultBundle
from . import registry
from .core import ModelFrame, build_result, information_criteria, wald_test
from .replay_sample import ReplayBatch, ReplaySample

_SUPPORTED = frozenset({"glm", "poisson", "cloglog", "fracreg", "nbreg", "betareg",
                       "hetprobit", "biprobit", "ologit", "oprobit", "mlogit",
                       "tobit", "intreg", "truncreg", "heckman", "heckprobit",
                       "cpoisson", "cnbreg", "tpoisson", "tnbreg", "zip", "zinb", "gnbreg",
                       "hurdle", "churdle", "ivprobit", "ivtobit", "frontier", "streg"})


def supports(spec: ModelSpec) -> bool:
    return isinstance(spec, ModelSpec) and spec.estimator in _SUPPORTED


def _option(spec: ModelSpec, name: str, fallback=None):
    entry = registry.get(spec.estimator).option(name)
    return spec.options.get(name, entry.default if entry is not None else fallback)


def _role(spec: ModelSpec, name: str) -> list[str]:
    return registry.role_columns(spec, name)


def _offset(batch: ReplayBatch, spec: ModelSpec) -> Tensor | None:
    offset = batch.numeric(_role(spec, "offset")[0]) if _role(spec, "offset") else None
    if _role(spec, "exposure"):
        exposure = batch.numeric(_role(spec, "exposure")[0])
        if bool((exposure <= 0).any()):
            raise AnalysisError("invalid_exposure", "Exposure must be positive.")
        offset = exposure.log() if offset is None else offset+exposure.log()
    return offset


class ReplayObjective:
    """A sum of existing native analytic batch objectives, with O(k²) totals."""

    def __init__(self, sample: ReplaySample, builder: Callable[[ReplayBatch], Any], size: int):
        self.sample, self.builder, self.size = sample, builder, size
        self.precision_rejected = False

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value, gradient, hessian = (_CompensatedSum(()), _CompensatedSum((self.size,)),
                                    _CompensatedSum((self.size, self.size)))
        for batch in self.sample.batches():
            objective = _precision_trial(self.builder(batch))
            v, g, h = objective(theta)
            self.precision_rejected |= bool(getattr(objective, "precision_rejected", False))
            if not bool(torch.isfinite(v)):
                return torch.tensor(-math.inf, dtype=torch.float64), torch.zeros_like(theta), -torch.eye(self.size, dtype=torch.float64)
            if g.shape != (self.size,) or h.shape != (self.size, self.size):
                raise AnalysisError("invalid_design", "A replay row objective changed parameter dimensions.")
            value.add(v)
            gradient.add(g)
            hessian.add(h)
        return value.value, gradient.value, (hessian.value+hessian.value.T)/2

    def value(self, theta: Tensor) -> Tensor:
        total = _CompensatedSum(())
        for batch in self.sample.batches():
            objective = _precision_trial(self.builder(batch))
            value = objective.value(theta)
            self.precision_rejected |= bool(getattr(objective, "precision_rejected", False))
            if not bool(torch.isfinite(value)):
                return torch.tensor(-math.inf, dtype=torch.float64)
            total.add(value)
        return total.value


class _GlmRows:
    """Adapt the native GLM state's score interface to a parameter interface."""

    def __init__(self, objective):
        self.objective = objective

    def __call__(self, theta):
        return self.objective(theta)

    def value(self, theta):
        return self.objective.value(theta)

    def score_rows(self, theta):
        state = self.objective.state(theta)
        if state is None:
            raise AnalysisError("numerical_failure", "GLM state is outside its link domain.")
        return self.objective.score_rows(state)


class _PartitionedRows:
    """Independent selection/body likelihood in outcome, selection, ancillary order."""

    def __init__(self, body, selection, chosen, k, q, ancillary):
        self.body, self.selection, self.chosen = body, selection, chosen
        self.k, self.q, self.size = k, q, k+q+ancillary
        self.first = torch.tensor([*range(k), *range(k+q, self.size)], dtype=torch.int64)
        self.second = torch.arange(k, k+q)

    def __call__(self, theta):
        value, gradient, hessian = self.selection(theta[self.second])
        g = torch.zeros(self.size, dtype=torch.float64)
        h = torch.zeros((self.size, self.size), dtype=torch.float64)
        g[self.second], h[self.second[:, None], self.second] = gradient, hessian
        if self.body is not None:
            v, gradient, hessian = self.body(theta[self.first])
            value = value+v
            g[self.first], h[self.first[:, None], self.first] = gradient, hessian
        self.precision_rejected = bool(getattr(self.selection, "precision_rejected", False) or getattr(self.body, "precision_rejected", False))
        return value, g, h

    def value(self, theta):
        value = self.selection.value(theta[self.second])+(self.body.value(theta[self.first]) if self.body is not None else 0.)
        self.precision_rejected = bool(getattr(self.selection, "precision_rejected", False) or getattr(self.body, "precision_rejected", False))
        return value

    def score_rows(self, theta):
        rows = torch.zeros((len(self.chosen), self.size), dtype=torch.float64)
        rows[:, self.second] = self.selection.score_rows(theta[self.second])
        if self.body is not None:
            indices = self.chosen.nonzero().flatten()
            rows[indices[:, None], self.first] = self.body.score_rows(theta[self.first])
        return rows


def _linear_start(sample: ReplaySample, name: str, target: Callable[[ReplayBatch], Tensor],
                  selector: Callable[[ReplayBatch], Tensor] | None = None,
                  matrix: Callable[[ReplayBatch], Tensor] | None = None) -> tuple[Tensor, float]:
    """Global weighted TSQR start and residual variance, never a sampled fit."""
    tree = _TSQRTree()
    mass = _CompensatedSum(())
    width = None
    for batch in sample.batches():
        y = target(batch)
        chosen = selector(batch) if selector else torch.ones(len(y), dtype=torch.bool)
        if not bool(chosen.any()):
            continue
        if not bool(torch.isfinite(y[chosen]).all()):
            raise AnalysisError("invalid_start", "The global starting target is not finite.")
        root = batch.weights[chosen].sqrt()
        design = matrix(batch) if matrix is not None else batch.designs[name]
        width = design.shape[1]
        augmented = torch.cat((design[chosen], y[chosen, None]), 1)*root[:, None]
        factor = qr_factor(augmented)
        tree.add(factor)
        mass.add(batch.weights[chosen].sum())
    if width is None:
        raise AnalysisError("empty_equation_sample", "No observations identify the starting equation.")
    factor = tree.finish()
    run = least_squares(factor[:, :width], factor[:, width], drop_collinear=False)
    variance = float(run.ssr)/float(mass.value)
    return run.beta, variance


def _require_variance(sample, variance, target, selector=None):
    total, reference = _CompensatedSum(()), _CompensatedSum(())
    for batch in sample.batches():
        y = target(batch)
        chosen = selector(batch) if selector else torch.ones(len(y), dtype=torch.bool)
        total.add(batch.weights[chosen].sum())
        reference.add((batch.weights[chosen]*y[chosen].square()).sum())
    threshold = 1e-24*float(reference.value/total.value)
    if not math.isfinite(variance) or not variance > threshold:
        raise AnalysisError("perfect_fit", "The global starting regression has no identifiable positive error variance.")


def _binary_y(batch: ReplayBatch, name: str) -> Tensor:
    y = batch.numeric(name)
    if bool(((y != 0) & (y != 1)).any()):
        raise AnalysisError("invalid_binary_outcome", f"'{name}' must be coded0/1.")
    return y


def _limits(spec: ModelSpec) -> tuple[float | None, float | None]:
    ll, ul = _option(spec, "ll"), _option(spec, "ul")
    if _role(spec, "left_limit") or _role(spec, "right_limit"):
        raise AnalysisError("streaming_options_unsupported", "This replay normal likelihood currently needs scalar censoring/truncation limits.")
    if ll is not None and ul is not None and not ll < ul:
        raise AnalysisError("invalid_limits", "The lower limit must be below the upper limit.")
    return ll, ul


def _censored_count_bounds(batch, spec):
    y = batch.numeric(spec.outcome)
    limits = []
    for option, role in (("ll", "left_limit"), ("ul", "right_limit")):
        columns = _role(spec, role)
        if columns and option in spec.options:
            raise AnalysisError("invalid_spec", "Give each count censoring limit as a scalar or a column, not both.")
        bound = batch.numeric(columns[0], allow_missing=True) if columns else torch.full_like(y, math.nan if _option(spec, option) is None else _option(spec, option))
        finite = ~torch.isnan(bound)
        if bool((~torch.isfinite(bound[finite])|(bound[finite] < 0)|(bound[finite] > 2**53)|(bound[finite] != bound[finite].round())).any()):
            raise AnalysisError("invalid_censoring_limit", "Count censoring limits must be exact nonnegative integers up to2^53, or missing unbounded values.")
        limits.append(bound)
    ll, ul = limits
    both = torch.isfinite(ll)&torch.isfinite(ul)
    if _role(spec, "censoring"):
        statuses = batch.frame[_role(spec, "censoring")[0]]
        mapping = {"exact": 0, "left": 1, "right": 2, "interval": 3}
        if not bool(statuses.isin(mapping).all()):
            raise AnalysisError("invalid_censoring_status", "Censoring events must be exact, left, right or interval.")
        status = torch.tensor(statuses.map(mapping).tolist(), dtype=torch.int64)
        left, right = (status == 1)|(status == 3), (status == 2)|(status == 3)
        if bool((left&~torch.isfinite(ll)).any()) or bool((right&~torch.isfinite(ul)).any()):
            raise AnalysisError("missing_censoring_limit", "Censored events need their applicable finite limits.")
        if bool((both&((ll > ul)|((ll == ul)&(status != 3)))).any()):
            raise AnalysisError("invalid_censoring_limits", "Count limits need ll<ul, or ll<=ul for explicit intervals.")
        if bool((((status == 1)&(y > ll))|((status == 2)&(y < ul))|((status == 3)&((y < ll)|(y > ul)))).any()):
            raise AnalysisError("inconsistent_censoring", "The recorded count contradicts its declared censoring event.")
    else:
        if bool((both&(ll >= ul)).any()):
            raise AnalysisError("invalid_censoring_limits", "Automatic count censoring needs ll<ul.")
        status = torch.zeros(len(y), dtype=torch.int64)
        status[torch.isfinite(ll)&(y <= ll)] = 1
        status[torch.isfinite(ul)&(y >= ul)] = 2
    low, high = y.clone(), y.clone()
    low[status == 1], high[status == 1] = 0., ll[status == 1]
    low[status == 2], high[status == 2] = ul[status == 2], math.inf
    low[status == 3], high[status == 3] = ll[status == 3], ul[status == 3]
    return low, high


def _censored_bounds(batch: ReplayBatch, spec: ModelSpec) -> tuple[Tensor, Tensor]:
    if spec.estimator == "intreg":
        low = batch.numeric(spec.outcome, allow_missing=True)
        high = batch.numeric(_role(spec, "upper")[0], allow_missing=True)
        if bool((low > high).any()) or bool(torch.isinf(low).any()) or bool(torch.isinf(high).any()):
            raise AnalysisError("invalid_interval", "Interval bounds must be ordered finite values or missing open endpoints.")
        return torch.nan_to_num(low, nan=-math.inf), torch.nan_to_num(high, nan=math.inf)
    y = batch.numeric(spec.outcome)
    ll, ul = _limits(spec)
    low, high = y.clone(), y.clone()
    if ll is not None:
        chosen = y <= ll
        low[chosen], high[chosen] = -math.inf, ll
    if ul is not None:
        chosen = y >= ul
        low[chosen], high[chosen] = ul, math.inf
    return low, high


def _create(spec: ModelSpec, source: Dataset, *, batch_rows: int | None = None) -> ReplaySample:
    requested_spec = spec
    from .streaming_likelihood_options import resolve_sample_limits
    spec, limit_baseline = resolve_sample_limits(spec, source, batch_rows=batch_rows)
    name = spec.estimator
    if spec.weight_type == "pweight" and spec.covariance in {"nonrobust", "opg"}:
        raise AnalysisError("unsupported_covariance", "pweights require robust or cluster covariance.")
    if len(registry.cluster_columns(spec)) > 2:
        raise AnalysisError("unsupported_cluster_dimensions", "Replay likelihood supports one or two cluster dimensions.")
    allow_missing, row_filter = [], None
    if name in {"heckman", "heckprobit"}:
        allow_missing = [spec.outcome]

        def row_filter(frame):
            selected = torch.as_tensor((frame[_role(spec, "select")[0]] == 1).to_numpy(), dtype=torch.bool)
            missing = torch.as_tensor(frame[spec.outcome].isna().to_numpy(), dtype=torch.bool)&selected
            if bool(missing.any()) and spec.missing == "raise":
                raise AnalysisError("missing_values", "Selected observations need observed outcomes.")
            return ~missing
    elif name == "intreg":
        upper = _role(spec, "upper")[0]
        allow_missing = [spec.outcome, upper]

        def row_filter(frame):
            empty = frame[spec.outcome].isna()&frame[upper].isna()
            if bool(empty.any()) and spec.missing == "raise":
                raise AnalysisError("missing_values", "At least one interval endpoint must be observed.")
            return torch.as_tensor((~empty).to_numpy(), dtype=torch.bool)
    elif name == "streg":
        from .survival.data import check_times
        def row_filter(frame):
            time = torch.as_tensor(frame[spec.outcome].to_numpy(dtype="float64"), dtype=torch.float64)
            entry_name = (_role(spec, "entry") or [None])[0]
            entry = torch.as_tensor(frame[entry_name].to_numpy(dtype="float64"), dtype=torch.float64) if entry_name else None
            return check_times(time, entry, spec.outcome, entry_name)
    elif name == "truncreg":
        ll, ul = _limits(spec)

        def row_filter(frame):
            y = torch.as_tensor(frame[spec.outcome].to_numpy(dtype="float64"), dtype=torch.float64)
            return (y > ll if ll is not None else torch.ones_like(y, dtype=torch.bool)) & (y < ul if ul is not None else torch.ones_like(y, dtype=torch.bool))
    if name in {"cpoisson", "cnbreg"}:
        allow_missing = [*_role(spec, "left_limit"), *_role(spec, "right_limit")]
    if name in {"tobit", "truncreg", "ivtobit"}:
        if name in {"tobit", "ivtobit"} and _option(spec, "ll") is None and _option(spec, "ul") is None:
            raise AnalysisError("invalid_limits", "Tobit needs at least one censoring limit.")
    sample = ReplaySample(spec, source, allow_missing=allow_missing, row_filter=row_filter,
                          batch_rows=batch_rows, outcome_categories=name in {"ologit", "oprobit", "mlogit"},
                          ordered_outcome=name in {"ologit", "oprobit"})
    if name == "streg":
        # Strata are labels even if numeric, and are discovered globally using
        # the same bounded dictionary and deterministic treatment coding.
        for column in _role(spec, "strata"):
            sample._levels.setdefault(column, {})
    sample.add_design("mean", [*spec.predictors, *_role(spec, "strata")] if name == "streg" else None, intercept=False if name in {"ologit", "oprobit"} else None,
                      implicit_constant=name in {"ologit", "oprobit"},
                      prefix=f"{spec.outcome}:" if name == "biprobit" else "")
    if name == "streg":
        from .survival.parametric import ANCILLARY
        dist = _option(spec, "distribution")
        anc = ANCILLARY.get(dist)
        if anc is None and _role(spec, "ancillary"):
            raise AnalysisError("invalid_spec", "The exponential survival model has no ancillary equation.")
        if anc is not None:
            sample.add_design("ancillary", [*_role(spec, "ancillary"), *_role(spec, "strata")],
                              intercept=True, prefix=f"{anc}:")
    elif name == "hetprobit":
        sample.add_design("variance", _role(spec, "het"), intercept=False, implicit_constant=True, prefix="lnsigma:")
    elif name == "betareg":
        sample.add_design("precision", _role(spec, "scale"), intercept=True, prefix="scale:")
    elif name == "biprobit":
        sample.add_design("second", _role(spec, "predictors2") or spec.predictors, prefix=f"{_role(spec, 'outcome2')[0]}:")
    elif name in {"heckman", "heckprobit"}:
        sample.designs["mean"].selector = lambda frame: torch.as_tensor((frame[_role(spec, "select")[0]] == 1).to_numpy(), dtype=torch.bool)
        sample.add_design("selection", _role(spec, "select_x"), intercept=True, prefix="select:")
    elif name in {"zip", "zinb"}:
        sample.add_design("inflate", _role(spec, "inflate"), intercept=not _option(spec, "noconstant", False), prefix="inflate:")
    elif name == "gnbreg":
        sample.add_design("dispersion", _role(spec, "lnalpha"), intercept=True, prefix="lnalpha:")
    elif name in {"hurdle", "churdle"}:
        limit = _option(spec, "ll", 0.) if name == "churdle" else 0.
        sample.designs["mean"].selector = lambda frame: torch.as_tensor((frame[spec.outcome] > limit).to_numpy(), dtype=torch.bool)
        sample.add_design("selection", _role(spec, "select_x") or (spec.predictors if name == "hurdle" else []),
                          intercept=True, prefix="select:")
    elif name in {"ivprobit", "ivtobit"}:
        sample.add_design("instruments", list(dict.fromkeys([*spec.predictors, *_role(spec, "instruments")])))
        sample.add_design("endogenous", _role(spec, "endogenous"), intercept=False)
    sample.prepare()
    sample.requested_spec = requested_spec
    if limit_baseline is not None:
        if (sample.baseline["data_hash"] != limit_baseline["data_hash"]
                or sample.baseline["original"] != limit_baseline["original"]):
            raise AnalysisError("source_changed", "The source changed after global censoring-limit discovery.")
        sample.notes["limit_discovery"] = {"scope": "full positive-weight complete-case sample before truncation",
                                           "lower": _option(spec, "ll"), "upper": _option(spec, "ul")}
        if name == "truncreg":
            pre_count = limit_baseline["frequency"] if spec.weight_type == "fweight" else limit_baseline["used"]
            sample.notes["n_truncated"] = pre_count-sample.nobs
        sample.passes += 1
    if name == "streg":
        from .streaming_likelihood_options import validate_survival_subjects
        sample.notes["survival"] = validate_survival_subjects(sample)
    return sample


def _adapter(sample: ReplaySample):
    """Return builder, global start, report map/metadata for existing row kernels."""
    from .glm.families import CANONICAL_LINK, make_family, make_link
    from .glm.kernels import GlmObjective, BetaObjective, ScaleLink
    from .discrete.kernels import HetprobitObjective, OrderedObjective, MultinomialObjective
    from .discrete.bivariate import BiprobitObjective
    from .limited.kernels import CensoredObjective, TruncatedObjective
    from .limited.selection_kernels import HeckmanObjective, HeckprobitObjective
    from .limited.iv_kernels import IVObjective, ProbitRows
    from .limited.kernels import CensoredRows
    from .count.kernels import IndexObjective, CountPieces, NegBinDensity, PoissonDensity, TruncatedPieces, ZeroInflatedPieces, BinaryPieces, TruncatedNormalPieces
    from .count.censored_kernels import CensoredPieces

    spec, name = sample.spec, sample.spec.estimator
    mean, k = sample.designs["mean"], len(sample.designs["mean"].terms)
    if not k and name not in {"ologit", "oprobit"}:
        raise AnalysisError("no_parameters", "No identified mean-equation regressor remains.")
    terms, equations = list(mean.terms), [spec.outcome]*k
    transform = mean.transform
    extra: dict[str, Any] = {}
    report_map = None
    if name in {"glm", "poisson", "cloglog", "fracreg"}:
        if name == "glm":
            from .glm.glm import _settings
            _settings(ModelFrame(spec, sample.sample))
        family_name = _option(spec, "family", "gaussian") if name == "glm" else "poisson" if name == "poisson" else "binomial"
        link_name = _option(spec, "link") if name in {"glm", "fracreg"} else "log" if name == "poisson" else "cloglog"
        link_name = link_name or CANONICAL_LINK[family_name]
        if _role(spec, "trials") and family_name != "binomial":
            raise AnalysisError("invalid_spec", "Trials are a binomial denominator and require family='binomial'.")
        dispersion = _option(spec, "dispersion", 1.)
        link = make_link(link_name, power=_option(spec, "power"), k=dispersion)

        def builder(batch):
            y = batch.numeric(spec.outcome)
            trials = batch.numeric(_role(spec, "trials")[0]) if _role(spec, "trials") else None
            if trials is not None:
                if bool(((trials <= 0)|(trials != trials.round())|(y < 0)|(y > trials)|(y != y.round())).any()):
                    raise AnalysisError("invalid_binomial_outcome", "Trials/counts must be valid binomial integers.")
                y = y/trials
            elif name == "glm" and family_name == "binomial":
                _binary_y(batch, spec.outcome)
            if family_name in {"poisson", "nbinomial"} and bool((y < 0).any()):
                raise AnalysisError("invalid_count_outcome", "Counts must be nonnegative.")
            if family_name in {"gamma", "inverse_gaussian"} and bool((y <= 0).any()):
                raise AnalysisError("invalid_positive_outcome", "This GLM family requires positive outcomes.")
            if family_name == "binomial" and bool(((y < 0)|(y > 1)).any()):
                raise AnalysisError("invalid_fractional_outcome", "Binomial/fractional responses must be in[0,1].")
            if name == "cloglog":
                _binary_y(batch, spec.outcome)
            family = make_family(family_name, k=dispersion, trials=trials, combinatorial=name != "fracreg")
            return _GlmRows(GlmObjective(batch.designs["mean"], y, batch.weights if trials is None else batch.weights*trials,
                                        _offset(batch, spec), family, link, trials))

        def target(batch):
            obj = builder(batch).objective
            initial = obj.family.start_mu(obj.y)
            return link.link(initial)-(_offset(batch, spec) if _offset(batch, spec) is not None else 0.)

        if name == "glm" and _option(spec, "optimizer") == "irls":
            from .streaming_likelihood_options import glm_start
            start = glm_start(sample, builder)
        else:
            start = _linear_start(sample, "mean", target)[0]
        extra.update({"family": family_name, "link": link_name, "quasi_likelihood": name == "fracreg"})
    elif name in {"ologit", "oprobit", "mlogit"}:
        categories = len(sample.labels)
        totals = _CompensatedSum((categories,))
        for batch in sample.batches():
            totals.add(torch.bincount(sample.codes(batch.frame), weights=batch.weights, minlength=categories))
        link = "logit" if name == "ologit" else "probit"
        if name == "mlogit":
            base_value = _option(spec, "base")
            if base_value is not None and base_value not in sample.labels:
                raise AnalysisError("invalid_base_category", "Requested base category is absent.")
            base = int(totals.value.argmax()) if base_value is None else sample.labels.index(base_value)
            def builder(batch):
                return MultinomialObjective(batch.designs["mean"], sample.codes(batch.frame), categories, base, batch.weights)
            start = torch.zeros(k*(categories-1), dtype=torch.float64)
            transform = torch.kron(torch.eye(categories-1, dtype=torch.float64), mean.transform)
            from .discrete.common import label_text
            labels = [label_text(label) for i, label in enumerate(sample.labels) if i != base]
            terms = [f"{label}:{term}" for label in labels for term in mean.terms]
            equations = [label for label in labels for _ in mean.terms]
            extra.update({"base": sample.labels[base], "equations": labels})
        else:
            def builder(batch):
                return OrderedObjective(batch.designs["mean"], sample.codes(batch.frame), categories,
                                                                   batch.weights, _offset(batch, spec), link)
            cumulative = totals.value.cumsum(0)[:-1]/totals.value.sum()
            cuts = torch.logit(cumulative) if link == "logit" else torch.special.ndtri(cumulative)
            start = torch.cat((torch.zeros(k, dtype=torch.float64), cuts))
            transform = torch.block_diag(mean.transform, torch.eye(categories-1, dtype=torch.float64))
            transform[k:, :k] = mean.means[mean.kept]/mean.scales[mean.kept]
            terms.extend(f"/cut{i}" for i in range(1, categories))
            equations.extend([None]*(categories-1))
            extra["link"] = link
            import pandas as pd
            extra["category_order"] = ("ordered Categorical" if isinstance(sample.sample[spec.outcome].dtype, pd.CategoricalDtype) else "sorted numeric values")
        extra.update({"categories": sample.labels, "category_counts": totals.value.tolist()})
    elif name == "hetprobit":
        variance = sample.designs["variance"]
        q = len(variance.terms)
        if not q:
            raise AnalysisError("no_variance_regressors", "No identified variance regressor remains.")
        def builder(batch):
            return HetprobitObjective(batch.designs["mean"], batch.designs["variance"], _binary_y(batch, spec.outcome), batch.weights)
        beta = _linear_start(sample, "mean", lambda batch: _binary_y(batch, spec.outcome)*2-1)[0]
        start = torch.cat((beta, torch.zeros(q, dtype=torch.float64)))
        terms.extend(variance.terms)
        equations.extend(["lnsigma"]*q)
        positives, total = _CompensatedSum(()), _CompensatedSum(())
        for batch in sample.batches():
            positives.add(batch.weights@_binary_y(batch, spec.outcome))
            total.add(batch.weights.sum())
        extra.update({"zero_outcomes": float(total.value-positives.value),
                      "nonzero_outcomes": float(positives.value),
                      "variance_function": "sigma = exp(z'g), no constant",
                      "variance_terms": variance.terms,
                      "variance_regressor_means": variance.means[variance.kept].tolist()})

        def report_map(theta, covariance):
            from .discrete.hetprobit import _uncentre
            jacobian = torch.block_diag(mean.transform, variance.transform)
            reported, cov = jacobian@theta, jacobian@covariance@jacobian.T
            return _uncentre(reported, cov, variance.means[variance.kept], k, name)
    elif name in {"biprobit", "heckman", "heckprobit"}:
        second = sample.designs["second" if name == "biprobit" else "selection"]
        q = len(second.terms)
        second_y = _role(spec, "outcome2")[0] if name == "biprobit" else _role(spec, "select")[0]
        def selected(batch):
            return _binary_y(batch, second_y) == 1
        if name == "biprobit":
            def builder(batch):
                return BiprobitObjective(batch.designs["mean"], batch.designs["second"],
                                                                    _binary_y(batch, spec.outcome), _binary_y(batch, second_y), batch.weights)
            beta, variance = _linear_start(sample, "mean", lambda batch: _binary_y(batch, spec.outcome)*2-1)
        else:
            def builder(batch):
                chosen = selected(batch)
                y = batch.numeric(spec.outcome, allow_missing=True)
                y = torch.where(chosen, y, torch.zeros_like(y))
                if name == "heckprobit" and bool(((y[chosen] != 0)&(y[chosen] != 1)).any()):
                    raise AnalysisError("invalid_binary_outcome", "Selected binary outcomes must be0/1.")
                kind = HeckmanObjective if name == "heckman" else HeckprobitObjective
                return kind(batch.designs["mean"], batch.designs["selection"], y, chosen, batch.weights)
            beta, variance = _linear_start(sample, "mean", lambda batch: batch.numeric(spec.outcome, allow_missing=True), selected)
        if name == "heckman":
            _require_variance(sample, variance, lambda batch: batch.numeric(spec.outcome, allow_missing=True), selected)
        gamma = _linear_start(sample, second.name, lambda batch: _binary_y(batch, second_y)*2-1)[0]
        ancillary = 2 if name == "heckman" else 1
        tail = [0., .5*math.log(variance)] if name == "heckman" else [0.]
        start = torch.cat((beta, gamma, torch.tensor(tail, dtype=torch.float64)))
        transform = torch.block_diag(mean.transform, second.transform, torch.eye(ancillary, dtype=torch.float64))
        terms.extend([*second.terms, "/athrho", *(["/lnsigma"] if name == "heckman" else [])])
        equations.extend([second_y]*q+[None]*ancillary)
    elif name == "betareg":
        precision = sample.designs["precision"]
        link, scale_link = make_link(_option(spec, "link")), ScaleLink(_option(spec, "scale_link"))

        def builder(batch):
            y = batch.numeric(spec.outcome)
            if bool(((y <= 0)|(y >= 1)).any()):
                raise AnalysisError("invalid_fractional_outcome", "Beta regression needs outcomes strictly between0 and1.")
            return BetaObjective(batch.designs["mean"], batch.designs["precision"], y, batch.weights, link, scale_link)

        beta = _linear_start(sample, "mean", lambda batch: link.link(batch.numeric(spec.outcome)))[0]
        gamma = torch.zeros(len(precision.terms), dtype=torch.float64)
        gamma[0] = scale_link.link(torch.tensor(10., dtype=torch.float64))
        start = torch.cat((beta, gamma))
        transform = torch.block_diag(mean.transform, precision.transform)
        terms.extend(precision.terms)
        equations.extend(["scale"]*len(precision.terms))
        extra.update({"link": link.name, "scale_link": scale_link.name})
    elif name in {"tobit", "intreg", "truncreg"}:
        if name == "truncreg":
            ll, ul = _limits(spec)
            def builder(batch):
                return TruncatedObjective(batch.designs["mean"], batch.numeric(spec.outcome), batch.weights, ll, ul, _offset(batch, spec))
            def target(batch):
                return batch.numeric(spec.outcome)
        else:
            def builder(batch):
                low, high = _censored_bounds(batch, spec)
                return CensoredObjective(batch.designs["mean"], low, high, batch.weights, _offset(batch, spec))

            def target(batch):
                low, high = _censored_bounds(batch, spec)
                return torch.where(torch.isneginf(low), high, torch.where(torch.isposinf(high), low, (low+high)/2))
        beta, variance = _linear_start(sample, "mean", lambda batch: target(batch)-(_offset(batch, spec) if _offset(batch, spec) is not None else 0.))
        _require_variance(sample, variance, lambda batch: target(batch)-(_offset(batch, spec) if _offset(batch, spec) is not None else 0.))
        start = torch.cat((beta, torch.tensor([.5*math.log(variance)], dtype=torch.float64)))
        transform = torch.block_diag(mean.transform, torch.ones((1, 1), dtype=torch.float64))
        terms.append("/lnsigma" if name == "intreg" else "/sigma")
        equations.append(None)
        linear_transform = transform

        def report_map(theta, covariance):
            theta, covariance = linear_transform@theta, linear_transform@covariance@linear_transform.T
            if name == "intreg":
                return theta, covariance
            jacobian = torch.eye(len(theta), dtype=torch.float64)
            sigma = torch.exp(theta[-1])
            theta[-1], jacobian[-1, -1] = sigma, sigma
            return theta, jacobian@covariance@jacobian.T
        extra["limits"] = {"lower": _option(spec, "ll"), "upper": _option(spec, "ul")}
    elif name in {"ivprobit", "ivtobit"}:
        instrument, endogenous = sample.designs["instruments"], sample.designs["endogenous"]
        kz, p = len(instrument.terms), len(endogenous.terms)
        if p != len(_role(spec, "endogenous")) or kz < k+p:
            raise AnalysisError("underidentified", "The retained instruments must identify every endogenous regressor.")
        if instrument.terms[:k] != mean.terms:
            raise AnalysisError("underidentified", "Included exogenous columns must survive first in the instrument design.")

        scales = endogenous.scales[endogenous.kept]
        centres = endogenous.means[endogenous.kept] if mean.intercept else torch.zeros(p, dtype=torch.float64)

        def endog_values(batch):
            # Match the native recursive ML working coordinates. In particular,
            # two-way score PSD repair belongs to this centred parameter basis.
            return batch.designs["endogenous"]-centres/scales

        def builder(batch):
            if name == "ivprobit":
                rows = ProbitRows(_binary_y(batch, spec.outcome))
            else:
                low, high = _censored_bounds(batch, spec)
                rows = CensoredRows(low, high)
            return IVObjective(batch.designs["instruments"], endog_values(batch), k, rows, batch.weights)

        # A bounded prototype supplies only the kernel's coordinate offsets;
        # all starting regressions and fitted likelihoods replay the full data.
        iterator = iter(sample.batches())
        try:
            prototype = builder(next(iterator))
        finally:
            iterator.close()
        start = torch.zeros(prototype.size, dtype=torch.float64)
        reduced = []
        for j in range(p):
            def reduced_matrix(batch, j=j):
                return torch.cat((batch.designs["instruments"], endog_values(batch)[:, :j]), 1)
            beta, variance = _linear_start(sample, "instruments", lambda batch, j=j: endog_values(batch)[:, j], matrix=reduced_matrix)
            _require_variance(sample, variance, lambda batch, j=j: endog_values(batch)[:, j])
            start[prototype.o_phi[j]:prototype.o_phi[j]+len(beta)] = beta
            start[prototype.o_tau+j] = .5*math.log(variance)
            reduced.append(beta)

        def control_matrix(batch):
            z, endog = batch.designs["instruments"], endog_values(batch)
            residuals = torch.stack([endog[:, j]-torch.cat((z, endog[:, :j]), 1)@reduced[j] for j in range(p)], 1)
            return torch.cat((z[:, :k], endog, residuals), 1)

        def target(batch):
            return _binary_y(batch, spec.outcome)*2-1 if name == "ivprobit" else batch.numeric(spec.outcome)

        beta, variance = _linear_start(sample, "instruments", target, matrix=control_matrix)
        start[:k+2*p] = beta
        if prototype.scale:
            _require_variance(sample, variance, lambda batch: batch.numeric(spec.outcome))
            start[-1] = .5*math.log(variance)
        out_transform = torch.block_diag(mean.transform, endogenous.transform)
        if mean.intercept:
            out_transform[0, k:k+p] = -centres/scales
        reduced_transforms = [instrument.transform*scale for scale in scales]
        names = prototype.ancillary_names()
        transformation = torch.block_diag(out_transform, *reduced_transforms, torch.eye(len(names), dtype=torch.float64))
        terms = [*mean.terms, *endogenous.terms,
                 *(f"{endog}:{term}" for endog in _role(spec, "endogenous") for term in instrument.terms), *names]
        equations = [spec.outcome]*(k+p)+[endog for endog in _role(spec, "endogenous") for _ in instrument.terms]+[None]*len(names)

        def report_map(theta, covariance):
            structural, jacobian = prototype.structural(theta)
            if not bool(torch.isfinite(structural).all()) or not bool(torch.isfinite(jacobian).all()):
                raise AnalysisError("boundary_solution", "The endogenous likelihood has no finite interior correlation/scale.")
            reported = transformation@structural
            covariance = transformation@jacobian@covariance@jacobian.T@transformation.T
            for j in range(p):
                if mean.intercept:
                    reported[k+p+j*kz] += centres[j]
                reported[terms.index(f"/lnsigma{j+2}")] += torch.log(scales[j])
            return reported, covariance
        extra.update({"method": "ml", "endogenous": _role(spec, "endogenous"), "instruments": _role(spec, "instruments")})
        if name == "ivtobit":
            extra["limits"] = {"lower": _option(spec, "ll"), "upper": _option(spec, "ul")}
        units = torch.ones(prototype.size, dtype=torch.float64)
        units[:k] = 1/mean.scales[mean.kept]
        units[k:k+2*p] = torch.cat((1/scales, 1/scales))
        for j in range(p):
            pos = prototype.o_phi[j]
            units[pos:pos+kz] = scales[j]/instrument.scales[instrument.kept]
            units[pos+kz:pos+kz+j] = scales[j]/scales[:j]
        extra["_canonical_covariance_units"] = units
    elif name == "frontier":
        from .systems.frontier_kernels import FrontierObjective, ANCILLARIES
        from .systems.frontier import _ancillary_start
        distribution, cost = _option(spec, "distribution"), bool(_option(spec, "cost"))
        sign = -1. if cost else 1.
        beta, variance = _linear_start(sample, "mean", lambda batch: batch.numeric(spec.outcome))
        total, centre = _CompensatedSum(()), _CompensatedSum(())
        for batch in sample.batches():
            residual = batch.numeric(spec.outcome)-batch.designs["mean"]@beta
            total.add(batch.weights.sum())
            centre.add(batch.weights@residual)
        center = centre.value/total.value
        second, third, reference = _CompensatedSum(()), _CompensatedSum(()), _CompensatedSum(())
        for batch in sample.batches():
            residual = batch.numeric(spec.outcome)-batch.designs["mean"]@beta-center
            second.add(batch.weights@residual.square())
            third.add(batch.weights@residual.pow(3))
            reference.add(batch.weights@batch.numeric(spec.outcome).square())
        m2, m3 = float(second.value/total.value), float(third.value/total.value)
        if variance <= 1e-24*float(reference.value/total.value):
            raise AnalysisError("perfect_fit", "The frontier starting regression has no identifiable error scale.")
        skew = -sign*m3
        if skew <= 0 and distribution != "tnormal":
            raise AnalysisError("boundary_solution", "The global residual skewness puts frontier inefficiency at zero.")
        skew = max(skew, 1e-6*m2**1.5)
        su = (skew/(2 if distribution == "exponential" else math.sqrt(2/math.pi)*(4/math.pi-1)))**(1/3)
        vu = su**2*(1 if distribution == "exponential" else 1-2/math.pi)
        mean_u = su*(1 if distribution == "exponential" else math.sqrt(2/math.pi))
        if vu >= .95*m2:
            scale = .5*m2/vu
            su, mean_u, vu = su*math.sqrt(scale), mean_u*math.sqrt(scale), .5*m2
        if mean.intercept:
            beta[0] += sign*mean_u
        start = torch.cat((beta, _ancillary_start(distribution, m2-vu, su**2)))
        tail = ANCILLARIES[distribution]
        transform = torch.block_diag(mean.transform, torch.eye(len(tail), dtype=torch.float64))
        terms.extend(tail)
        equations.extend([None]*len(tail))
        def builder(batch):
            return FrontierObjective(batch.designs["mean"], batch.numeric(spec.outcome), batch.weights, distribution, cost)
        extra.update({"distribution": distribution, "cost": cost, "ols_residual_variance": variance})
    elif name == "streg":
        from .survival.parametric import ParametricObjective, SurvivalData, ANCILLARY
        from .survival.streg import resolve_metric
        distribution = _option(spec, "distribution")
        metric = resolve_metric(distribution, _option(spec, "metric"))
        anc = ANCILLARY.get(distribution)
        ancillary = sample.designs.get("ancillary")
        q = len(ancillary.terms) if ancillary is not None else 0
        def survival(batch):
            time = batch.numeric(spec.outcome)
            entry = batch.numeric(_role(spec, "entry")[0]) if _role(spec, "entry") else None
            failure = _binary_y(batch, _role(spec, "failure")[0]) if _role(spec, "failure") else torch.ones_like(time)
            return SurvivalData(time, entry, failure)
        def builder(batch):
            return ParametricObjective(distribution, metric, batch.designs["mean"],
                    batch.designs.get("ancillary"), survival(batch), batch.weights, _offset(batch, spec))
        moments = _CompensatedSum((4,))
        for batch in sample.batches():
            data = survival(batch)
            moments.add(torch.stack(((batch.weights*data.delta).sum(),
                       (batch.weights*(data.t-data.t0)).sum(), (batch.weights*data.log_t).sum(), batch.weights.sum())))
        events, exposure, log_mean, mass = moments.value.tolist()
        if events <= 0:
            raise AnalysisError("no_failures", "The retained parametric survival sample has no failures.")
        rate, mu = math.log(events/exposure), log_mean/mass
        variance = _CompensatedSum(())
        for batch in sample.batches():
            data = survival(batch)
            variance.add((batch.weights*(data.log_t-mu).square()).sum())
        spread = math.sqrt(max(float(variance.value)/mass, 1e-4))
        if distribution == "loglogistic":
            spread *= math.sqrt(3)/math.pi
        const = rate if metric == "ph" else -rate
        ancconst = 0.
        if distribution not in {"exponential", "weibull", "gompertz"}:
            const, ancconst = mu, math.log(spread)
        start = torch.zeros(k+q+int(distribution == "ggamma"), dtype=torch.float64)
        start[0] = const
        if q:
            start[k] = ancconst
            transform = torch.block_diag(mean.transform, ancillary.transform)
            terms.extend(ancillary.terms if _role(spec, "ancillary") or _role(spec, "strata") else [f"/{anc}"])
            equations.extend([anc]*q if _role(spec, "ancillary") or _role(spec, "strata") else [None])
        if distribution == "ggamma":
            start[-1] = .3
            transform = torch.block_diag(transform, torch.ones((1, 1), dtype=torch.float64))
            terms.append("/kappa")
            equations.append(None)
        extra.update({"distribution": distribution, "metric": metric, "ancillary": anc,
                      "ancillary_predictors": _role(spec, "ancillary"),
                      "log_likelihood_scale": "log survival time: time-scale likelihood + sum w failure log(time)",
                      "derivatives": "numerical in kappa; analytic in indices" if distribution == "ggamma" else "analytic"})
    elif name in {"hurdle", "churdle"}:
        selection = sample.designs["selection"]
        q = len(selection.terms)
        limit = _option(spec, "ll", 0.) if name == "churdle" else 0.
        link = _option(spec, "select_link") if name == "churdle" else _option(spec, "zero_link")
        distribution = _option(spec, "dist", "poisson")
        kind = _option(spec, "model", "exponential")
        ancillary = 1 if name == "churdle" or distribution == "nbinomial" else 0
        if name == "churdle" and kind == "exponential" and limit < 0:
            raise AnalysisError("invalid_option", "Exponential hurdle limits must be nonnegative.")

        def chosen(batch):
            return batch.numeric(spec.outcome) > limit

        selected_count = 0
        for batch in sample.batches():
            selected_count += int(chosen(batch).sum())
        if not selected_count or selected_count == sample.nrows:
            raise AnalysisError("no_selection_variation", "A hurdle likelihood needs both bounded and above-limit outcomes.")
        if selected_count <= k+ancillary:
            raise AnalysisError("insufficient_observations", "The above-limit equation needs more observations than parameters.")

        def builder(batch):
            y = batch.numeric(spec.outcome)
            on = y > limit
            participation = IndexObjective(BinaryPieces(on, link), [batch.designs["selection"]], batch.weights)
            body = None
            if bool(on.any()):
                if name == "churdle":
                    observed = y[on]
                    t = observed if kind == "linear" else observed.log()
                    cut = limit if kind == "linear" else math.log(limit) if limit > 0 else None
                    pieces = TruncatedNormalPieces(t, cut, None if kind == "linear" else -t)
                else:
                    if bool(((y < 0)|(y != y.round())|(y >2**53)).any()):
                        raise AnalysisError("invalid_count_outcome", "Hurdle counts must be nonnegative exact integers.")
                    pieces = TruncatedPieces(NegBinDensity() if ancillary else PoissonDensity(), y[on], 0)
                designs = [batch.designs["mean"][on], *([None] if ancillary else [])]
                offset = _offset(batch, spec)
                body = IndexObjective(pieces, designs, batch.weights[on],
                                      [offset[on] if offset is not None else None, *([None] if ancillary else [])])
                body.reject_precision_trials = True
            return _PartitionedRows(body, participation, on, k, q, ancillary)

        def target(batch):
            y = batch.numeric(spec.outcome)
            if name == "churdle":
                return y if kind == "linear" else y.clamp_min(torch.finfo(torch.float64).tiny).log()
            return (y+.1).log()-(_offset(batch, spec) if _offset(batch, spec) is not None else 0.)

        beta, variance = _linear_start(sample, "mean", target, chosen)
        gamma = _linear_start(sample, "selection", lambda batch: chosen(batch).to(torch.float64)*2-1)[0]
        tail = [.5*math.log(variance) if name == "churdle" else math.log(.5)] if ancillary else []
        start = torch.cat((beta, gamma, torch.tensor(tail, dtype=torch.float64)))
        transform = torch.block_diag(mean.transform, selection.transform, torch.eye(ancillary, dtype=torch.float64))
        terms.extend([*selection.terms, *(["/lnsigma" if name == "churdle" else "/lnalpha"] if ancillary else [])])
        equations.extend(["select"]*q+[None]*ancillary)
        extra.update({"select_link": link, "model": kind, "ll": limit} if name == "churdle" else {"zero_link": link, "dist": distribution})
    else:
        dispersion = _option(spec, "dispersion", "mean")
        negbin = name in {"nbreg", "cnbreg", "tnbreg", "zinb", "gnbreg"}
        density = NegBinDensity(dispersion if name != "gnbreg" else "mean") if negbin else PoissonDensity()

        def builder(batch):
            y = batch.numeric(spec.outcome)
            if bool((y < 0).any()) or bool((y > 2**53).any()):
                raise AnalysisError("invalid_count_outcome", "Counts must be nonnegative and within exact float64 count precision.")
            if name in {"tpoisson", "tnbreg", "cpoisson", "cnbreg", "zip", "zinb"} and bool((y != y.round()).any()):
                raise AnalysisError("invalid_count_outcome", "This count likelihood requires integer outcomes.")
            designs, offsets = [batch.designs["mean"]], [_offset(batch, spec)]
            if name in {"tpoisson", "tnbreg"}:
                limit = _option(spec, "ll", 0)
                if _role(spec, "truncation"):
                    limit = batch.numeric(_role(spec, "truncation")[0])
                if bool((y <= limit).any()):
                    raise AnalysisError("outcome_not_truncated", "Every retained count must exceed its truncation limit.")
                pieces = TruncatedPieces(density, y, limit)
            elif name in {"cpoisson", "cnbreg"}:
                low, high = _censored_count_bounds(batch, spec)
                pieces = CensoredPieces(density, low, high)
            elif name in {"zip", "zinb"}:
                pieces = ZeroInflatedPieces(density, y, _option(spec, "inflate_link"))
                designs.append(batch.designs["inflate"])
                offsets.append(None)
            else:
                pieces = CountPieces(density, y)
            if negbin:
                designs.append(batch.designs["dispersion"] if name == "gnbreg" else None)
                offsets.append(None)
            result = IndexObjective(pieces, designs, batch.weights, offsets)
            result.reject_precision_trials = True
            return result

        beta, _ = _linear_start(sample, "mean", lambda batch: torch.log(batch.numeric(spec.outcome)+.1)-(_offset(batch, spec) if _offset(batch, spec) is not None else 0.))
        start, transforms = [beta], [mean.transform]
        if name in {"zip", "zinb"}:
            inflate = sample.designs["inflate"]
            infl_start = torch.zeros(len(inflate.terms), dtype=torch.float64)
            if inflate.intercept:
                infl_start[0] = -1.
            start.append(infl_start)
            transforms.append(inflate.transform)
            terms.extend(inflate.terms)
            equations.extend(["inflate"]*len(inflate.terms))
            extra["inflate_link"] = _option(spec, "inflate_link")
        if negbin:
            if name == "gnbreg":
                shape = sample.designs["dispersion"]
                start.append(torch.zeros(len(shape.terms), dtype=torch.float64))
                transforms.append(shape.transform)
                terms.extend(shape.terms)
                equations.extend(["lnalpha"]*len(shape.terms))
            else:
                start.append(torch.tensor([math.log(.5)], dtype=torch.float64))
                transforms.append(torch.ones((1, 1), dtype=torch.float64))
                terms.append("/lnalpha" if dispersion == "mean" else "/lndelta")
                equations.append(None)
            extra["dispersion"] = dispersion
        start, transform = torch.cat(start), torch.block_diag(*transforms)
    if len(start) >384 or sample.nobs <= len(start):
        raise AnalysisError("insufficient_observations", "The global likelihood sample needs more observations than parameters (maximum384 parameters).")
    if "_canonical_covariance_units" not in extra:
        extra["_canonical_covariance_units"] = (
            torch.cat((1/mean.scales[mean.kept], 1/sample.designs["variance"].scales[sample.designs["variance"].kept]))
            if name == "hetprobit" else transform.diagonal().clone())
    if report_map is None:
        def report_map(theta, covariance):
            return (transform@theta, transform@covariance@transform.T)
    return builder, start, terms, equations, extra, report_map


def _precision_trial(objective):
    """Native count kernels may reject trial points but never a final result."""
    if hasattr(objective, "reject_precision_trials"):
        objective.reject_precision_trials = True
    if isinstance(objective, _PartitionedRows):
        _precision_trial(objective.selection)
        if objective.body is not None:
            _precision_trial(objective.body)
    return objective


def _binary_certificate(sample, design_name, response, selector=None):
    from openecon.engines.separation import certify_separation
    width = len(sample.designs[design_name].terms)
    if not width:
        return

    def replay():
        for batch in sample.batches():
            y, x = response(batch), batch.designs[design_name]
            if selector:
                chosen = selector(batch)
                y, x = y[chosen], x[chosen]
            if not len(y):
                continue
            sign = torch.where(y == 1, 1., -1.)
            interior = (y > 0)&(y < 1)
            # Interior fractional/binomial observations constrain x'd=0.
            yield x*sign[:, None]
            if bool(interior.any()):
                yield x[interior]
    total, count = _CompensatedSum((width,)), 0
    for rows in replay():
        total.add(rows.sum(0))
        count += len(rows)
    if count:
        record = certify_separation(replay, total.value/count, width)
        sample.notes.setdefault("separation_certificates", {})[design_name] = record


def _validate_before(sample):
    spec, name = sample.spec, sample.spec.estimator
    if spec.weight_type == "pweight" and spec.covariance not in {"robust", "cluster"}:
        raise AnalysisError("invalid_pweight_covariance", "Sampling weights need robust or cluster covariance.")
    low, high, selected = math.inf, -math.inf, 0
    all_zero_lower = all_infinite_upper = True
    censoring_counts = _CompensatedSum((3,))
    for batch in sample.batches():
        if name not in {"ologit", "oprobit", "mlogit", "intreg"}:
            y = batch.numeric(spec.outcome, allow_missing=name in {"heckman", "heckprobit"})
            finite = y[torch.isfinite(y)]
            if len(finite):
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
            if name == "hurdle" and bool(((y < 0)|(y != y.round())|(y > 2**53)).any()):
                raise AnalysisError("invalid_count_outcome", "Hurdle outcomes must be nonnegative exact float64 integers.")
        if name in {"cpoisson", "cnbreg"}:
            lower, upper = _censored_count_bounds(batch, spec)
            all_zero_lower &= bool((lower == 0).all())
            all_infinite_upper &= bool(torch.isposinf(upper).all())
        if name in {"heckman", "heckprobit"}:
            on = _binary_y(batch, _role(spec, "select")[0]) == 1
            selected += int(on.sum())
        if name in {"tobit", "ivtobit"}:
            lower, upper = _censored_bounds(batch, spec)
            frequency = batch.weights if spec.weight_type == "fweight" else torch.ones_like(lower)
            left, right = torch.isneginf(lower), torch.isposinf(upper)
            censoring_counts.add(torch.stack((frequency[left].sum(), frequency[~left&~right].sum(), frequency[right].sum())))
    if name in {"tobit", "ivtobit"}:
        if censoring_counts.value[1] == 0:
            raise AnalysisError("no_uncensored_observations", "The complete Tobit sample has no uncensored observations.")
        sample.notes["censoring_counts"] = censoring_counts.value.tolist()
    if name in {"cpoisson", "cnbreg"}:
        if all_zero_lower and all_infinite_upper:
            raise AnalysisError("unidentified_censoring", "Every count event covers the entire support and contains no information.")
        if spec.intercept and (all_zero_lower or all_infinite_upper):
            raise AnalysisError("separation_detected", "Every count event includes zero or is right-unbounded; no finite intercept maximum exists.")
    if name in {"poisson", "nbreg", "gnbreg", "cpoisson", "cnbreg", "tpoisson", "tnbreg", "zip", "zinb"} and high <= 0:
        raise AnalysisError("constant_outcome", "The retained count outcome is zero throughout.")
    if name in {"cloglog", "hetprobit", "biprobit", "fracreg", "ivprobit", "heckprobit", "frontier"} and low == high:
        raise AnalysisError("constant_outcome", "The retained response does not vary.")
    if name in {"heckman", "heckprobit"} and (not selected or selected == sample.nrows):
        raise AnalysisError("no_selection_variation", "A selection likelihood needs selected and unselected observations.")
    if name in {"cloglog", "fracreg", "hetprobit", "biprobit"}:
        _binary_certificate(sample, "mean", lambda batch: batch.numeric(spec.outcome))
    if name == "glm" and _option(spec, "family") == "binomial":
        def response(batch):
            y = batch.numeric(spec.outcome)
            return y/batch.numeric(_role(spec, "trials")[0]) if _role(spec, "trials") else y
        _binary_certificate(sample, "mean", response)
    if name == "biprobit":
        _binary_certificate(sample, "second", lambda batch: _binary_y(batch, _role(spec, "outcome2")[0]))
    if name in {"heckman", "heckprobit"}:
        _binary_certificate(sample, "selection", lambda batch: _binary_y(batch, _role(spec, "select")[0]))
        if name == "heckprobit":
            _binary_certificate(sample, "mean", lambda batch: batch.numeric(spec.outcome, allow_missing=True),
                                lambda batch: batch.numeric(_role(spec, "select")[0]) == 1)
    if name in {"hurdle", "churdle"}:
        limit = _option(spec, "ll", 0.) if name == "churdle" else 0.
        _binary_certificate(sample, "selection", lambda batch: (batch.numeric(spec.outcome) > limit).to(torch.float64))


def _check_final(sample, builder, theta, hessian, *, converged=True):
    """Retain the native finite-interior/precision checks on the global fit."""
    from .discrete.common import EXTREME, perfectly_predicted
    from .glm.glm import _check_edge
    from .count.common import FLAT_INFORMATION
    name, spec = sample.spec.estimator, sample.spec
    mass = sample.nrows if spec.weight_type == "aweight" else sample.nrows*sample.weight_mean
    if name in {"nbreg", "cnbreg", "tnbreg", "zinb"} or (name == "hurdle" and _option(spec, "dist") == "nbinomial"):
        information = -float(hessian[-1, -1])*sample.nobs/mass
        if float(theta[-1]) < 0 and information < FLAT_INFORMATION:
            raise AnalysisError("boundary_solution", "The negative-binomial dispersion is weakly identified at the Poisson boundary.")
    if name in {"biprobit", "heckprobit", "heckman"}:
        position = -2 if name == "heckman" else -1
        if 1-float(torch.tanh(theta[position]))**2 < 1e-10:
            raise AnalysisError("boundary_solution", "The estimated correlation reaches the boundary of its parameter space.")
    if name == "frontier":
        from .systems.frontier import _check_boundary
        _check_boundary(_option(spec, "distribution"), theta[len(sample.designs["mean"].terms):])
    for batch in sample.batches():
        native = builder(batch)
        if hasattr(native, "check_precision"):
            native.check_precision(theta)
        if isinstance(native, _PartitionedRows) and native.body is not None:
            native.body.check_precision(theta[native.first])
        if name in {"glm", "poisson", "cloglog", "fracreg"}:
            obj = native.objective
            state = obj.state(theta)
            _check_edge(obj, state, name)
            if state is None:
                raise AnalysisError("numerical_failure", "The final GLM response is outside its link domain.")
            if obj.family.binomial and not converged:
                own = torch.where(obj.y == 1, state.mu, torch.where(obj.y == 0, state.comp, torch.zeros_like(obj.y)))
                if bool((own > 1-EXTREME).any()):
                    raise AnalysisError("separation_detected", "A final binomial response is perfectly predicted.")
        elif name in {"ologit", "oprobit", "mlogit", "hetprobit"} and not converged:
            probabilities = native.probabilities(theta) if name != "hetprobit" else native.outcome_probabilities(theta)
            if perfectly_predicted(probabilities, EXTREME):
                raise AnalysisError("separation_detected", "A final category or binary response is perfectly predicted.")
            if name in {"ologit", "oprobit"} and native.determined(theta, EXTREME):
                raise AnalysisError("separation_detected", "A final category direction is determined with numerical certainty.")
        elif name in {"zip", "zinb"} and not converged:
            from .count.kernels import link_pieces
            probability = torch.exp(link_pieces(_option(spec, "inflate_link"), native.indices(theta)[1], False)[0])
            if bool(((probability < EXTREME)|(probability > 1-EXTREME)).any()):
                raise AnalysisError("boundary_solution", "A final inflation probability reaches its boundary; no rounded interior fit is reported.")
        elif name == "gnbreg" and float(native.indices(theta)[1].min()) < math.log(1e-9):
            raise AnalysisError("precision_unsupported", "A final varying dispersion exceeds the validated count precision region.")


def _chart_sample(sample, builder, theta):
    """One fully verified pass; store only400 actual response/latent predictions."""
    name, spec = sample.spec.estimator, sample.spec
    if name in {"ologit", "oprobit", "mlogit", "hurdle", "churdle", "streg"}:
        return None, None, None
    fitted, observed, remaining = [], [], 400
    k = len(sample.designs["mean"].terms)
    definition = "fitted conditional response"
    for batch in sample.batches():
        if not remaining:
            continue
        obj = builder(batch)
        y = batch.numeric(spec.outcome, allow_missing=name in {"heckman", "heckprobit", "intreg"})
        if name in {"glm", "poisson", "cloglog", "fracreg"}:
            pred = obj.objective.state(theta).mu
            y = obj.objective.y
        elif name == "hetprobit":
            pred = obj.probabilities(theta)
        elif name == "biprobit":
            pred = obj.marginal_probabilities(theta)[0]
            definition = "first-equation marginal probability"
        elif name in {"heckman", "heckprobit"}:
            q = len(sample.designs["selection"].terms)
            pred = torch.special.ndtr(batch.designs["selection"]@theta[k:k+q])
            y = batch.numeric(_role(spec, "select")[0])
            definition = "fitted selection probability against selection status"
        elif name == "betareg":
            pred = obj.fitted(theta)[0]
        elif name in {"tobit", "intreg", "truncreg", "frontier"}:
            pred = batch.designs["mean"]@theta[:k]
            off = _offset(batch, spec)
            if off is not None:
                pred += off
            definition = "linear prediction of the latent outcome"
            if name == "intreg":
                low, high = _censored_bounds(batch, spec)
                y = torch.where(torch.isneginf(low), high, torch.where(torch.isposinf(high), low, (low+high)/2))
                definition += "; observed interval midpoint or finite endpoint"
        elif name in {"ivprobit", "ivtobit"}:
            pred = obj.index(theta)
            if name == "ivprobit":
                pred = torch.special.ndtr(pred)
            definition = "conditional outcome index including reduced-form residuals"
        else:
            from .count.kernels import link_pieces
            indices = obj.indices(theta)
            pred = torch.exp(indices[0])
            if name in {"zip", "zinb"}:
                pred *= torch.exp(link_pieces(_option(spec, "inflate_link"), indices[1], False)[1])
            elif name in {"tpoisson", "tnbreg"}:
                pred = obj.pieces.conditional_mean(indices)
        take = min(remaining, len(batch.frame))
        fitted.append(pred[:take].clone())
        observed.append(y[:take].clone())
        remaining -= take
    return torch.cat(fitted), torch.cat(observed), definition


def fit_streaming(spec: ModelSpec, source: Dataset, *, batch_rows: int | None = None) -> ResultBundle:
    """Fit supported native likelihoods from a replayable Dataset without collect."""
    if not supports(spec):
        raise AnalysisError("streaming_unsupported", f"{spec.estimator} has no replay likelihood adapter.")
    try:
        with torch.no_grad(), torch.device("cpu"):
            sample = _create(spec, source, batch_rows=batch_rows)
            _validate_before(sample)
            if spec.estimator in {"heckman", "ivprobit", "ivtobit"} and _option(spec, "method") == "twostep":
                from .streaming_twostep import fit_twostep
                return fit_twostep(sample)
            builder, start, terms, equations, extra, report_map = _adapter(sample)
            widths = sum(len(design.terms) for design in sample.designs.values())
            per_row = 128*(widths+len(start)+4)
            if spec.estimator in {"biprobit", "heckprobit"}:
                per_row += 128*96
            sample.plan_rows("replay analytic likelihood", {"optimizer_information_score_buffers": 128*len(start)**2}, per_row)
            objective = ReplayObjective(sample, builder, len(start))
            irls = spec.estimator == "glm" and _option(spec, "optimizer") == "irls"
            if irls:
                from .streaming_likelihood_options import fit_irls
                run = fit_irls(sample, builder, start)
            else:
                run = optimize.maximize_newton(
                    lambda theta: tuple(value*sample.optimization_scale for value in objective(theta)), start,
                    value_fn=lambda theta: objective.value(theta)*sample.optimization_scale,
                    max_iter=_option(spec, "max_iterations", 200), step_tol=_option(spec, "tolerance", 1e-10),
                    scaled_gradient_tol=_option(spec, "tolerance", 1e-10),
                    raise_on_failure=False)
            _check_final(sample, builder, run.theta, run.hessian/sample.optimization_scale, converged=run.converged)
            if not run.converged:
                if objective.precision_rejected:
                    raise AnalysisError("precision_unsupported", "The replay likelihood reached an unvalidated precision domain.")
                raise AnalysisError("nonconvergence", f"The global replay likelihood did not identify a finite interior maximum: {run.diagnostics.get('message')}")
            hessian = run.hessian/sample.optimization_scale
            from .streaming_likelihood_options import likelihood_covariance
            covariance, inference = likelihood_covariance(sample, builder, run.theta, hessian,
                                                           canonical_units=extra.pop("_canonical_covariance_units"))
            if irls:
                inference["correction"] = inference["correction"].replace("observed information", "expected information")
                extra.update({"optimizer": "irls", "information": "expected"})
            params, covariance = report_map(run.theta.clone(), covariance)
            if spec.estimator in {"ologit", "oprobit"}:
                extra["cutpoints"] = params[len(sample.designs["mean"].terms):].tolist()
            log_likelihood = run.value/sample.optimization_scale
            if spec.estimator in {"ivprobit", "ivtobit"}:
                # Standardized endogenous responses change their density by
                # a constant Jacobian. Derivatives and the ML point stay exact.
                mass = sample.nrows if spec.weight_type == "aweight" else sample.weight_mean*sample.nrows
                log_likelihood -= mass*float(torch.log(sample.designs["endogenous"].scales[
                    sample.designs["endogenous"].kept]).sum())
            if spec.estimator == "streg":
                log_time = _CompensatedSum(())
                for batch in sample.batches():
                    failure = _binary_y(batch, _role(spec, "failure")[0]) if _role(spec, "failure") else torch.ones(len(batch.frame), dtype=torch.float64)
                    log_time.add((batch.weights*failure*batch.numeric(spec.outcome).log()).sum())
                log_likelihood += float(log_time.value)
            metrics = information_criteria(log_likelihood, len(start), sample.nobs)
            if spec.estimator in {"ologit", "oprobit", "mlogit"}:
                metrics["n_categories"] = len(sample.labels)
            # GLM Newton maximizes -deviance/2. The reported likelihood is the
            # family's actual likelihood at the global fitted response.
            if spec.estimator in {"glm", "poisson", "cloglog", "fracreg"}:
                from .glm.glm import dispersion, _statistic_scale
                deviance, pearson, ll = _CompensatedSum(()), _CompensatedSum(()), _CompensatedSum(())
                family = None
                reference = _CompensatedSum(())
                for batch in sample.batches():
                    native = builder(batch).objective
                    state = native.state(run.theta)
                    family = native.family
                    reference.add(torch.tensor(_statistic_scale(family, native.y, native.prior), dtype=torch.float64))
                    deviance.add(torch.tensor(state.deviance, dtype=torch.float64))
                    pearson.add(torch.tensor(native.pearson(state), dtype=torch.float64))
                df_resid = sample.nobs-len(start)
                phi, rule = dispersion(_option(spec, "scale"), family, float(deviance.value), float(pearson.value), df_resid, float(reference.value))
                phi_ll = float(deviance.value)/sample.nobs if family.name == "gaussian" else 1.
                for batch in sample.batches():
                    native = builder(batch).objective
                    ll.add(torch.tensor(native.log_likelihood(native.state(run.theta), phi_ll, batch.weights), dtype=torch.float64))
                log_likelihood = float(ll.value)
                metrics = information_criteria(log_likelihood, len(start), sample.nobs)
                metrics.update({"deviance": float(deviance.value), "pearson": float(pearson.value), "scale": phi})
                if spec.covariance == "nonrobust":
                    covariance *= phi
                elif spec.covariance == "opg":
                    covariance *= phi**2
                inference.update({"dispersion": phi, "scale_rule": rule})
            if spec.estimator == "frontier":
                from .systems.frontier import _transforms
                distribution = _option(spec, "distribution")
                k = len(sample.designs["mean"].terms)
                extra["ancillary"] = {}
                from openecon.engines.distributions import normal_ppf
                z = normal_ppf(1-spec.alpha/2)
                for field, (value, gradient) in _transforms(distribution, params[k:]).items():
                    se = math.sqrt(max(0., float(gradient@covariance[k:, k:]@gradient)))
                    metrics[field] = value
                    extra["ancillary"][field] = {"estimate": value, "std_error": se, "ci_low": value-z*se, "ci_high": value+z*se}
            frame = ModelFrame(spec, sample.sample, allow_missing=sample.allow_missing)
            fitted, observed, definition = _chart_sample(sample, builder, run.theta)
            k = len(sample.designs["mean"].terms)
            mean_count = k+len(sample.designs["endogenous"].terms) if spec.estimator in {"ivprobit", "ivtobit"} else k
            slopes = [i for i, term in enumerate(terms[:mean_count]) if not term.endswith("Intercept")]
            if spec.estimator == "mlogit":
                slopes = [i for i, term in enumerate(terms) if not term.endswith("Intercept")]
            use_t = spec.estimator == "tobit"
            df_resid = sample.nobs-len(slopes) if use_t else None
            if use_t:
                inference["df_inference"] = df_resid
                metrics.update({"df_model": len(slopes), "df_resid": df_resid})
            tests = {"model": wald_test(params, covariance, slopes, df_resid=df_resid,
                                          label="Wald F of replay model slopes" if use_t else "Wald chi2 of replay model slopes")}
            from .streaming_likelihood_diagnostics import augment
            augment(sample, builder, run.theta, terms, params, covariance, metrics, tests, extra)
            result = build_result(frame, terms=terms, params=params, covariance=covariance, equations=equations,
                                  nobs=sample.nobs, metrics=metrics, tests=tests, extra=extra, inference=inference,
                                  use_t=use_t, df_inference=df_resid, df_resid=df_resid, fitted=fitted, observed=observed, categories=sample.categories,
                                  solver="torch_replayed_fisher_scoring_tsqr" if irls else "torch_replayed_analytic_newton",
                                  solver_diagnostics={**run.diagnostics, "converged": True, "iterations": run.iterations,
                                                      "dense_observation_matrix": False},
                                  provenance={**sample.provenance(), "prediction_sample": "first400 retained observations" if fitted is not None else None,
                                              "prediction_definition": definition,
                                              "separation_certificates": sample.notes.get("separation_certificates", {})})
            for row in result.predictions:
                row["row"] = sample.sample_positions[int(row["row"])]
            result.nobs_original = sample.original_count
            result.dropped_rows = sample.original_count-sample.nrows
            result.sample_positions = []
            return result
    except AnalysisError:
        raise
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
