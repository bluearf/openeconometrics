"""Saved count-mixture and left-truncated means in reporting coordinates.

The count index includes its offset; the auxiliary index never does. ``response``
is the overall mixture mean, and ``conditional`` conditions on a positive count
(or on the fitted truncation limit). All equation and dispersion covariance
cross-terms remain on the saved parameter axes. No outcome is used to reconstruct
these indexes. Tail series are positive recurrences, not subtraction of nearly
equal cumulative probabilities; unsupported precision/work domains fail openly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

import torch
from torch.nn import functional as F

from openecon.analysis_contracts import AnalysisError, _MAX_DESIGN_BYTES
from openecon.econometrics.glm.families import make_link
from openecon.econometrics.postest.inference import _numeric
from openecon.engines.count_numeric import poisson_logmass, validate_nb_precision
from openecon.engines.contracts import KernelError

ESTIMATORS = frozenset({"zip", "zinb", "hurdle", "tpoisson", "tnbreg"})
_MAX_SERIES = 512
_MAX_LOWER_TERMS = 10000


def _error(code, message):
    raise AnalysisError(code, message)


def _workspace(rows, parameters, terms=0):
    # Common design plus two indexes, two changed-category designs, Jacobian and
    # the retained first/second derivative tape. Never construct a row x row jac.
    if 8 * rows * (32 * max(1, parameters) + 32 * terms + 64) > _MAX_DESIGN_BYTES:
        _error("prediction_memory_limit", "Count prediction derivative buffers exceed 256 MiB; use smaller evaluation batches.")


def _finite(value, message="Count predictions exceed finite float64 range."):
    if not bool(torch.isfinite(value).all()):
        _error("prediction_precision", message)
    return value


def _positive_exp(logarithm):
    value = _finite(logarithm.exp())
    if bool((value < torch.finfo(torch.float64).tiny).any()):
        _error("prediction_precision", "A positive count-distribution parameter enters the unvalidated subnormal float64 region.")
    return value


def _log_probability(z, link, complement=False):
    if link == "logit":
        return -F.softplus(z if complement else -z)
    if link == "probit":
        # For |z|>1e6 the squared normal tail and any subsequent finite mean
        # cancellation are outside this adapter's derivative validation domain.
        if bool((z.abs() > 1e6).any()):
            _error("prediction_precision", "Normal mixture indexes above 1e6 in magnitude are not supported.")
        return torch.special.log_ndtr(-z if complement else z)
    if complement:
        return -z.clamp(max=7).exp()
    t = z.clamp(max=7).exp()
    # Keep log Pr(participation) finite at z=-800, where exp(z) underflows,
    # and do not let an unused branch introduce 0/0 backward products.
    small = z < -15
    direct = torch.log(-torch.expm1(-torch.where(small, 1., t)))
    u = torch.where(small, t, 0.)
    expansion = z + torch.log1p(-u / 2 + u.square() / 6 - u.pow(3) / 24)
    return torch.where(small, expansion, direct)


def _ratio_log1p(t):
    small = t < .001
    safe = torch.where(small, 1., t)
    direct = safe / torch.log1p(safe)
    u = torch.where(small, t, 0.)
    polynomial = 1 + u / 2 - u.square() / 12 + u.pow(3) / 24 - 19 * u.pow(4) / 720 + 3 * u.pow(5) / 160
    return torch.where(small, polynomial, direct)


def _ratio_expm1(t):
    small = t < .01
    safe = torch.where(small, 1., t)
    direct = safe / -torch.expm1(-safe)
    u = torch.where(small, t, 0.)
    polynomial = 1 + u / 2 + u.square() / 12 - u.pow(4) / 720 + u.pow(6) / 30240 - u.pow(8) / 1209600
    return torch.where(small, polynomial, direct)


def _tail_series(mu, shape, limit, *, parameters, probability=None, reserved_terms=0):
    """E[Y|Y>=m] from positive tail weights normalized at m=ll+1.

    ``shape=None`` is Poisson. NB weights have ratio q*(shape+m+j)/(m+j+1).
    A relative second-moment remainder check protects derivative convergence.
    """
    m = limit + 1
    if shape is not None and probability is None:
        probability = mu / (shape + mu)
    total, excess, term = torch.ones_like(mu), torch.zeros_like(mu), torch.ones_like(mu)
    for j in range(1, _MAX_SERIES + 1):
        _workspace(len(mu), parameters, reserved_terms + j)
        ratio = mu / (m + j) if shape is None else probability * (shape + m + j - 1) / (m + j)
        term = term * ratio
        total, excess = total + term, excess + j * term
        # The next ratios are bounded by max(q,current ratio) for NB and
        # decrease for Poisson. Bound the remaining weighted second moment.
        bound = ratio if probability is None else torch.maximum(ratio, probability)
        if bool((bound < 1).all()):
            remainder = term * (j + 1) ** 2 / (1 - bound).pow(3)
            if bool((remainder <= 2e-17 * total).all()):
                return m + excess / total
    _error("prediction_precision", "The conditional count tail did not converge within 512 positive terms; use a less extreme evaluation point.")


def _poisson_conditional(eta, limit, parameters):
    mu = _positive_exp(eta)
    value = torch.empty_like(mu)
    zero = limit == 0
    if bool(zero.any()):
        value[zero] = _ratio_expm1(mu[zero])
    other = ~zero
    if bool(other.any()):
        u, ll = mu[other], limit[other]
        m = ll + 1
        # A rare upper tail can underflow even when its conditional mean and
        # derivatives are ordinary finite numbers. Compute those rows directly.
        series = u <= .9 * m
        part = torch.empty_like(u)
        if bool(series.any()):
            part[series] = _tail_series(u[series], None, ll[series], parameters=parameters)
        if bool((~series).any()):
            s = ~series
            lower = torch.special.gammainc(m[s], u[s])
            upper = torch.special.gammaincc(m[s], u[s])
            log_s = torch.where(upper < .5, torch.log1p(-upper), lower.log())
            if not bool(torch.isfinite(log_s).all()):
                _error("prediction_precision", "The Poisson truncation tail is not representable in the validated gamma region.")
            part[s] = u[s] + (eta[other][s] + poisson_logmass(ll[s], u[s], eta=eta[other][s]) - log_s).exp()
        value[other] = part
    return _finite(value)


def _nb_conditional(eta, tau, limit, form, parameters):
    mu = _positive_exp(eta)
    shape = _positive_exp(-tau if form == "mean" else eta - tau)
    try:
        validate_nb_precision(limit, shape)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    t = _positive_exp(eta + tau if form == "mean" else tau)
    value = torch.empty_like(mu)
    zero = limit == 0
    if bool(zero.any()):
        length = shape[zero] * torch.log1p(t[zero])
        value[zero] = _ratio_log1p(t[zero]) * _ratio_expm1(length)
    other = ~zero
    if bool(other.any()):
        u, r, ll = mu[other], shape[other], limit[other]
        a = eta[other] + tau[other] if form == "mean" else tau[other]
        log_p = -F.softplus(a)
        log_q = -F.softplus(-a)
        log_mass = r * log_p
        cdf, first = log_mass.exp(), torch.zeros_like(u)
        top = int(ll.max())
        if top > _MAX_LOWER_TERMS:
            _error("prediction_work_limit", "NB conditional means require at most 10000 lower-tail terms per evaluation row.")
        _workspace(len(u), parameters, top + 1)
        for j in range(1, top + 1):
            log_mass = log_mass + torch.log(r + j - 1) - math.log(j) + log_q
            mass = log_mass.exp() * (ll >= j)
            cdf, first = cdf + mass, first + j * mass
        survival = 1 - cdf
        stable = survival >= 1e-5
        part = torch.empty_like(u)
        if bool(stable.any()):
            part[stable] = (u[stable] - first[stable]) / survival[stable]
        if bool((~stable).any()):
            s = ~stable
            # NB1 q=delta/(1+delta) is independent of eta. Reconstructing it
            # as mu/(mu+shape) invents a cancellation-sized eta derivative,
            # swamping the genuine tiny shape effect in the log-series limit.
            part[s] = _tail_series(u[s], r[s], ll[s], parameters=parameters,
                                   probability=torch.sigmoid(a[s]), reserved_terms=top + 1)
        value[other] = part
    return _finite(value)


def _log_positive_zero(eta, tau, form):
    """Log positive-count mean without an exp(index)->log chain.

    Tiny participation can offset a huge, still finite count mean. Taking log
    of that mean after materializing it creates an underflowed reverse-mode
    derivative before the count exponential can restore its scale.
    """
    mu = _positive_exp(eta)
    if form is None:
        small = mu < .01
        safe = torch.where(small, 1., mu)
        direct = eta - torch.log(-torch.expm1(-safe))
        local = _ratio_expm1(torch.where(small, mu, 1.)).log()
        return torch.where(small, local, direct)
    logr = -tau if form == "mean" else eta - tau
    r = _positive_exp(logr)
    try:
        validate_nb_precision(torch.zeros_like(mu), r)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    logt = eta + tau if form == "mean" else tau
    t = _positive_exp(logt)
    length = r * F.softplus(logt)
    small = length < .01
    safe = torch.where(small, 1., length)
    direct = eta - torch.log(-torch.expm1(-safe))
    tiny_t = t < .001
    ratio_log = torch.where(tiny_t,
                            _ratio_log1p(torch.where(tiny_t, t, 1.)).log(),
                            logt - torch.log(F.softplus(logt)))
    local = ratio_log + _ratio_expm1(torch.where(small, length, 1.)).log()
    return torch.where(small, local, direct)


@dataclass(frozen=True)
class CountMixture:
    estimator: str
    count_indices: tuple[int, ...]
    auxiliary_indices: tuple[int, ...]
    count_slopes: dict[str, int]
    auxiliary_slopes: dict[str, int]
    link: str | None
    dispersion: str | None
    dispersion_index: int | None
    truncation_column: str | None
    limit: int
    accepts_outcome: bool = False
    parameter_count: int = 0

    @property
    def kinds(self):
        return {"response", "xb", "conditional", "stdp", "derivative"}

    def required(self):
        return [] if self.truncation_column is None else [self.truncation_column]

    def workspace_row_bytes(self, frame=None, kind=None):
        """Conservative row-local tape budget for projected Dataset scheduling.

        A column cutoff without fetched rows uses the supported worst-case lower
        sum. A fetched frame can refine it before numerical coding. Poisson's
        rare-tail series and NB's lower sum plus series are reserved together.
        """
        k = self.parameter_count or max((*self.count_indices, *self.auxiliary_indices,
                                         *(() if self.dispersion_index is None else (self.dispersion_index,))), default=0) + 1
        limit = self.limit
        if self.truncation_column is not None:
            if frame is None:
                limit = _MAX_LOWER_TERMS
            elif len(frame):
                raw = _numeric(frame[self.truncation_column].dropna().tolist(), code="invalid_truncation", ndim=1)
                limit = int(raw.max()) if len(raw) else 0
        terms = 0
        if kind not in {"xb", "stdp"} and limit:
            terms = _MAX_SERIES if self.dispersion is None else min(limit, _MAX_LOWER_TERMS) + 1 + _MAX_SERIES
        return 8 * (32 * k + 32 * terms + 64)

    def encode(self, frame, design):
        _workspace(len(frame), design.x.shape[1])
        if self.truncation_column is None:
            limit = torch.full((len(frame),), float(self.limit), dtype=torch.float64)
        else:
            limit = _numeric(frame[self.truncation_column].tolist(), code="invalid_truncation", ndim=1)
        maximum = 1e12 if self.dispersion is None else 1e7 - 1
        if bool(((limit < 0) | (limit != limit.round()) | (limit > maximum)).any()):
            _error("invalid_truncation", "Prediction truncation points must be nonnegative exact integers inside the count special-function precision region.")
        return replace(design, scale=limit)

    def indexes(self, design, beta):
        _workspace(len(design.x), len(beta))
        eta = design.x[:, list(self.count_indices)] @ beta[list(self.count_indices)] + design.deterministic
        z = (design.x[:, list(self.auxiliary_indices)] @ beta[list(self.auxiliary_indices)]
             if self.auxiliary_indices else eta * 0)
        tau = beta[self.dispersion_index].expand_as(eta) if self.dispersion_index is not None else eta * 0
        return tuple(_finite(v, "A saved count equation index is nonfinite.") for v in (eta, z, tau))

    def _values(self, eta, z, tau, limit, kind, parameters):
        if kind == "xb":
            return eta
        if kind not in {"response", "conditional"}:
            _error("unsupported_prediction_kind", "Saved count mixtures support response, xb, and conditional means.")
        if self.estimator in {"zip", "zinb"} and kind == "response":
            return _positive_exp(eta + _log_probability(z, self.link, True))
        if self.estimator == "hurdle" and kind == "response":
            return _positive_exp(_log_positive_zero(eta, tau, self.dispersion)
                                 + _log_probability(z, self.link))
        rounded = limit.round()
        tolerance = 8 * torch.finfo(torch.float64).eps * rounded.abs().clamp_min(1)
        if bool(((limit < 0) | ((limit - rounded).abs() > tolerance)).any()):
            _error("unsupported_margins_transform", "An encoded mean of varying truncation cutoffs must remain a nonnegative integer; set one valid cutoff before MEM evaluation.")
        # Global convex MEM reduction can put an exactly constant integer cutoff
        # one ULP away from itself. Raw input cutoffs were checked exactly in
        # encode; permit only this numerical reduction tolerance here.
        limit = rounded
        conditional = (_poisson_conditional(eta, limit, parameters) if self.dispersion is None
                       else _nb_conditional(eta, tau, limit, self.dispersion, parameters))
        return conditional

    def values(self, design, beta, kind):
        return self._values(*self.indexes(design, beta), design.scale,
                            "xb" if kind == "stdp" else kind, len(beta))

    def _partials(self, design, beta, kind, variable=None):
        with torch.enable_grad():
            indexes = tuple(v if v.requires_grad else v.detach().requires_grad_(True) for v in self.indexes(design, beta))
            eta, z, tau = indexes
            value = self._values(eta, z, tau, design.scale, kind, len(beta))
            slopes = torch.autograd.grad(value.sum(), indexes, create_graph=True, allow_unused=True)
            slopes = tuple(v * 0 if g is None else g for v, g in zip(indexes, slopes, strict=True))
            if variable is not None:
                b = beta[self.count_slopes[variable]] if variable in self.count_slopes else beta.sum() * 0
                g = beta[self.auxiliary_slopes[variable]] if variable in self.auxiliary_slopes else beta.sum() * 0
                value = slopes[0] * b + slopes[1] * g
            return _finite(value), indexes, slopes

    def effects(self, design, beta, variable, kind):
        return self._partials(design, beta, kind, variable)[0]

    def jacobian(self, design, beta, kind, variable=None):
        # Three vector indexes give row-local partials. Chain them to parameter
        # axes, rather than asking autograd for an N by N vector Jacobian.
        with torch.enable_grad():
            requested = "response" if kind == "derivative" else "xb" if kind == "stdp" else kind
            value, indexes, slopes = self._partials(design, beta, requested, variable)
            if variable is None:
                partials = slopes
            else:
                partials = torch.autograd.grad(value.sum(), indexes, create_graph=False, allow_unused=True)
                partials = tuple(v * 0 if g is None else g for v, g in zip(indexes, partials, strict=True))
            jac = torch.zeros_like(design.x)
            for partial, indices in zip(partials[:2], (self.count_indices, self.auxiliary_indices), strict=True):
                if indices:
                    jac[:, list(indices)] += partial[:, None] * design.x[:, list(indices)]
            if self.dispersion_index is not None:
                jac[:, self.dispersion_index] += partials[2]
            if variable is not None:
                if variable in self.count_slopes:
                    jac[:, self.count_slopes[variable]] += slopes[0]
                if variable in self.auxiliary_slopes:
                    jac[:, self.auxiliary_slopes[variable]] += slopes[1]
            return _finite(jac, "Count parameter derivatives are not finite.").detach()

    def definition(self, kind):
        if kind == "xb":
            return "count-equation linear index including its offset"
        if kind == "stdp":
            return "delta-method standard error of the count-equation linear index including its offset"
        if self.estimator in {"tpoisson", "tnbreg"}:
            return "conditional E[Y | Y > fitted truncation limit, X]"
        if kind == "conditional":
            return "conditional positive-count E[Y | Y > 0, X]; auxiliary probability cancels"
        return "overall E[Y | X, Z], including the fitted zero-inflation/participation equation"


def _names(value):
    if (not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value)
            or len(set(value)) != len(value)):
        _error("invalid_result", "Saved equation roles require distinct original column names.")
    return list(value)


def _column(value):
    if value is None or value == []:
        return None
    if not isinstance(value, str) or not value:
        _error("invalid_result", "Saved count offset/exposure/truncation must identify one original column.")
    return value


def configure(result, state):
    """Return the strict saved-design contract consumed by prediction dispatch."""
    spec = result.spec
    if spec.estimator not in ESTIMATORS:
        return None
    if (state.df is not None or spec.panel or spec.time or not isinstance(spec.intercept, bool)
            or not isinstance(spec.columns, Mapping) or not isinstance(spec.options, Mapping)
            or not isinstance(result.extra, Mapping) or result.extra.get("constrained_terms")):
        _error("invalid_result", "Saved count mixtures require their native independent-row normal-inference specification.")
    names, categorical = _names(spec.predictors), _names(spec.categorical)
    second, prefix, link, dispersion = [], "", None, None
    permitted_columns = {"offset", "exposure"}
    permitted_options = {"workspace_budget_bytes"}
    if spec.estimator in {"zip", "zinb"}:
        permitted_columns.add("inflate")
        permitted_options.add("inflate_link")
        second, prefix = _names(spec.columns.get("inflate", [])), "inflate:"
        link = spec.options.get("inflate_link", "logit")
        if link not in {"logit", "probit"} or result.extra.get("inflate_link") != link:
            _error("invalid_result", "Saved zero-inflation link is missing, invalid, or inconsistent.")
        dispersion = "mean" if spec.estimator == "zinb" else None
    elif spec.estimator == "hurdle":
        permitted_columns.add("select_x")
        permitted_options.update({"dist", "zero_link"})
        second, prefix = _names(spec.columns.get("select_x", [])) or names, "select:"
        link, dist = spec.options.get("zero_link", "logit"), spec.options.get("dist", "poisson")
        if (link not in {"logit", "probit", "cloglog"} or dist not in {"poisson", "nbinomial"}
                or result.extra.get("zero_link") != link or result.extra.get("dist") != dist):
            _error("invalid_result", "Saved hurdle link/distribution is inconsistent with its fitted specification.")
        dispersion = "mean" if dist == "nbinomial" else None
    else:
        permitted_columns.add("truncation")
        permitted_options.add("ll")
        if spec.estimator == "tnbreg":
            permitted_options.add("dispersion")
            dispersion = spec.options.get("dispersion", "mean")
            if dispersion not in {"mean", "constant"} or result.extra.get("dispersion") != dispersion:
                _error("invalid_result", "Saved truncated NB dispersion does not match its fitted parameterization.")
    if set(spec.columns) - permitted_columns or set(spec.options) - permitted_options:
        _error("invalid_result", "Saved count prediction roles/options are outside the fitted native contract.")
    predictors = list(dict.fromkeys([*names, *second]))
    if spec.outcome in predictors or set(categorical) - set(predictors):
        _error("invalid_result", "Saved original predictor/categorical roles are inconsistent.")
    coding = result.provenance.get("categorical_encoding", {})
    if not isinstance(coding, Mapping) or set(coding) != set(categorical):
        _error("invalid_result", "Saved category encodings do not match the fitted equation roles.")
    categories = {}
    for name in categorical:
        record = coding[name]
        if (not isinstance(record, Mapping) or record.get("coding") != "treatment_drop_first"
                or not isinstance(record.get("levels"), list) or len(record["levels"]) < 2
                or record.get("reference") != record["levels"][0]):
            _error("invalid_result", "Saved treatment category encoding is incomplete or inconsistent.")
        levels = record["levels"]
        if any(isinstance(v, (list, dict)) or v is None or isinstance(v, float) and not math.isfinite(v) for v in levels):
            _error("invalid_result", "Saved category levels must be finite scalar labels.")
        if len({(type(v).__name__, str(v)) for v in levels}) != len(levels):
            _error("invalid_result", "Saved category labels must be distinct.")
        categories[name] = levels

    def block(original, constant, namespace):
        features = {namespace + "Intercept": ("constant", None)} if constant else {}
        for name in original:
            if name in categories:
                features.update({f"{namespace}{name}[{level}]": ("category", (name, level)) for level in categories[name][1:]})
            else:
                features[namespace + name] = ("numeric", name)
        return features

    count = block(names, spec.intercept, "")
    auxiliary = block(second, True, prefix) if prefix else {}
    features = {**count, **auxiliary}
    if len(features) != len(count) + len(auxiliary):
        _error("invalid_result", "Saved equation reporting names collide.")
    omitted = result.provenance.get("omitted_terms", [])
    if (not isinstance(omitted, list) or any(not isinstance(v, str) for v in omitted)
            or len(set(omitted)) != len(omitted) or set(omitted) - set(features)):
        _error("invalid_result", "Saved count/auxiliary omitted terms are invalid.")
    ancillary = "/lndelta" if dispersion == "constant" else "/lnalpha"
    expected = set(features) - set(omitted)
    if dispersion is not None:
        expected.add(ancillary)
    if set(state.terms) != expected or not (set(count) & expected) or (prefix and not (set(auxiliary) & expected)):
        _error("invalid_result", "Saved parameters do not account for all fitted equations and omissions.")
    if dispersion is not None:
        name = "delta" if dispersion == "constant" else "alpha"
        recorded = result.extra.get("alpha")
        try:
            estimate = math.exp(float(state.beta[state.terms.index(ancillary)]))
        except OverflowError as exc:
            raise AnalysisError("invalid_result", "Saved dispersion is outside finite float64 range.") from exc
        if estimate == 0:
            _error("invalid_result", "Saved dispersion cannot equal zero.")
        if (not isinstance(recorded, Mapping) or recorded.get("parameter") != name
                or not isinstance(recorded.get("estimate"), (int, float))
                or isinstance(recorded.get("estimate"), bool)
                or not math.isclose(recorded["estimate"], estimate, rel_tol=1e-10, abs_tol=0)
                or not isinstance(result.metrics.get(name), (int, float))
                or not math.isclose(result.metrics[name], estimate, rel_tol=1e-10, abs_tol=0)):
            _error("invalid_result", "Saved log-dispersion coefficient and reported dispersion metadata disagree.")
    for coefficient in result.coefficients:
        equation = None if coefficient.term == ancillary and dispersion is not None else prefix[:-1] if coefficient.term in auxiliary else spec.outcome
        # Native tpoisson reports its single equation with no explicit label.
        accepted = {None, spec.outcome} if spec.estimator == "tpoisson" else {equation}
        if coefficient.equation not in accepted:
            _error("invalid_result", "Saved count coefficient equation labels are inconsistent.")
    offset, exposure = None, None
    for role in ("offset", "exposure", "truncation"):
        value = _column(spec.columns.get(role))
        if role == "offset":
            offset = value
        elif role == "exposure":
            exposure = value
    if offset and exposure:
        _error("invalid_result", "Saved count offset and exposure are mutually exclusive.")
    truncation = _column(spec.columns.get("truncation"))
    limit = 0
    if spec.estimator in {"tpoisson", "tnbreg"}:
        if truncation:
            if "ll" in spec.options or result.extra.get("truncation_column") != truncation:
                _error("invalid_result", "Saved row-specific truncation metadata disagree.")
        else:
            limit = spec.options.get("ll", 0)
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0 or limit != int(limit) or limit > 2 ** 53:
                _error("invalid_result", "Saved scalar truncation must be an exact nonnegative integer.")
            limit = int(limit)
            if result.extra.get("truncation_point") != limit:
                _error("invalid_result", "Saved scalar truncation metadata disagree.")
    def slopes(original, namespace):
        return {name: state.terms.index(namespace + name) for name in original
                if namespace + name in expected and name not in categories}
    adapter = CountMixture(spec.estimator,
                           tuple(i for i, t in enumerate(state.terms) if t in count),
                           tuple(i for i, t in enumerate(state.terms) if t in auxiliary),
                           slopes(names, ""), slopes(second, prefix), link, dispersion,
                           state.terms.index(ancillary) if dispersion is not None else None,
                           truncation, limit, parameter_count=len(state.terms))
    return dict(link=make_link("log"), family="poisson", predictors=predictors,
                categories=categories, features=features, fixed={},
                mean_terms=set(features) & expected, offset=offset, exposure=exposure,
                trials=None, adapter=adapter)
