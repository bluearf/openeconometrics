"""Bootstrap and jackknife covariances for any registry estimator.

Both procedures refit ``result.spec`` on modified copies of the estimation data
and keep only the parameter vectors (memory is O(replicates x parameters)).

Population. The rows resampled are the estimation sample of the original fit
(``result.sample_positions``), as Stata's prefixes drop observations outside
``e(sample)``. For estimators with a panel variable the units are whole panels
and the population is every row of the estimation panels, so that periods an
estimator uses only as lags (dynamic panels) travel with their panel; the refit
re-applies its own sample rules (missing values, singletons, lags).

The dataset passed must be the one the model was fitted on: its model columns
are hashed and compared with ``provenance['data_hash']``.
"""

from __future__ import annotations

import secrets
from collections import Counter
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import kernel_call
from openecon.econometrics.postest.common import (
    coefficient_vector, input_columns, is_suest, matched_frame, model_wald, rebuild,
    refit_parameters, refit_spec, refitter, require_result, result_covariance,
)
from openecon.econometrics.postest.draws import (
    Clusters, Strata, WeightedStrata, bootstrap_covariance, bootstrap_intervals, draw_units,
    draw_weighted, jackknife_covariance,
)
from openecon.models import ResultBundle

FAILURE_SHARE = 0.10          # more failed replicates than this raises (bootstrap_failed)
JACKKNIFE_MAX_REFITS = 2000   # default bound on the number of delete-one refits
_CI_KINDS = ("normal", "percentile", "bc")
_GROUP_ROLES = ("group", "id")   # grouping roles relabelled with the resampled clusters


class _Population:
    """The resampling population: data rows, units, strata and relabelling rules."""

    def __init__(self, result: ResultBundle, data: Any, cluster: str | None,
                 strata: str | None, procedure: str):
        spec = result.spec
        self.spec = spec
        self.notes: list[str] = []
        frame, rows = matched_frame(result, data)
        for name, role in ((cluster, "cluster"), (strata, "strata")):
            if name is not None and (not isinstance(name, str) or name not in frame.columns):
                raise AnalysisError("missing_columns", f"The {role} column {name!r} is not in the "
                                    "data.")
        self.cluster = cluster if cluster is not None else spec.panel
        self.strata = strata
        positions = torch.tensor(rows, dtype=torch.int64)
        if spec.panel is not None:
            # Whole panels: every data row of the panels in the estimation sample.
            labels = frame[spec.panel]
            used = labels.iloc[positions.numpy()].unique()
            keep = labels.isin(used).to_numpy()
            for name in (self.cluster, strata):
                if name is not None:
                    keep &= frame[name].notna().to_numpy()
            positions = torch.as_tensor(keep.nonzero()[0], dtype=torch.int64)
            self.population = "all rows of the panels in the estimation sample"
        else:
            self.population = "estimation sample"
        columns = list(dict.fromkeys([*input_columns(result), *registry.spec_columns(spec),
                                      *(name for name in (self.cluster, strata) if name)]))
        self.base = frame.iloc[positions.numpy()].loc[:, columns].reset_index(drop=True)
        self.n = len(self.base)
        if self.n < 2:
            raise AnalysisError("insufficient_observations", f"{procedure} needs at least two "
                                "observations.")
        self.cluster_codes, self.n_clusters = self._codes(self.cluster, "cluster")
        self.strata_codes, self.n_strata = self._codes(strata, "strata")
        self.frequency = None
        if spec.weight_type == "fweight" and spec.weights is not None:
            self.frequency = torch.as_tensor(self.base[spec.weights].to_numpy(dtype="float64"))
        # Grouping columns relabelled per drawn cluster copy (Stata's idcluster): the panel
        # variable (must nest in the clusters) and group/id roles that nest in them.
        self.grouping: list[tuple[str, Tensor, int]] = []
        action = "resample" if procedure == "bootstrap" else "delete"
        if self.cluster_codes is not None:
            for name in self._grouping_columns():
                codes, count = self._codes(name, "grouping")
                nested = self._nested(codes, count, self.cluster_codes)
                if name == spec.panel and not nested:
                    raise AnalysisError("invalid_groups", "Each panel must lie within one cluster "
                                        "for resampling.")
                if nested and name != self.cluster:
                    self.grouping.append((name, codes, count))
        elif self._grouping_columns():
            self.notes.append(
                f"Single observations were {action}d although the model groups them by "
                f"{', '.join(map(repr, self._grouping_columns()))}; pass cluster= to {action} "
                "whole groups.")
        if self.cluster_codes is not None and self.strata_codes is not None:
            if not self._nested(self.cluster_codes, self.n_clusters, self.strata_codes):
                raise AnalysisError("invalid_groups", "Each cluster must lie within one stratum "
                                    "for resampling.")
        if self.cluster_codes is not None and self.n_clusters < 2:
            raise AnalysisError("insufficient_clusters", f"{procedure} needs at least two "
                                f"clusters of {self.cluster!r} in the estimation sample.")
        fitted_clusters = [name for name in registry.cluster_columns(spec)
                           if name != self.cluster]
        if fitted_clusters and self.cluster is None:
            self.notes.append(
                f"The fit clusters on {', '.join(map(repr, fitted_clusters))} but single "
                f"observations were {action}d; pass cluster={fitted_clusters[0]!r} to {action} "
                "whole clusters.")

    def _grouping_columns(self) -> list[str]:
        spec = self.spec
        names = [spec.panel] if spec.panel is not None else []
        names += [name for role in _GROUP_ROLES for name in registry.role_columns(spec, role)]
        return list(dict.fromkeys(names))

    def _codes(self, name: str | None, role: str) -> tuple[Tensor | None, int]:
        if name is None:
            return None, 0
        column = self.base[name]
        if bool(column.isna().any()):
            raise AnalysisError("missing_values", f"The {role} column {name!r} has missing values "
                                "in the estimation sample.")
        try:
            codes, levels = pd.factorize(column, sort=False)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_groups", f"Labels in {name!r} must be scalar values.") \
                from exc
        return torch.from_numpy(codes.astype("int64")), len(levels)

    @staticmethod
    def _nested(inner: Tensor, n_inner: int, outer: Tensor) -> bool:
        """Whether every inner group lies within one outer group."""
        low = torch.full((n_inner,), torch.iinfo(torch.int64).max, dtype=torch.int64)
        high = torch.full((n_inner,), -1, dtype=torch.int64)
        low = low.scatter_reduce(0, inner, outer, "amin")
        high = high.scatter_reduce(0, inner, outer, "amax")
        return bool((low == high).all())

    def unit_strata(self) -> tuple[Tensor, int]:
        """Stratum code of every resampling unit (row or cluster); one stratum by default."""
        units = self.n_clusters if self.cluster_codes is not None else self.n
        if self.strata_codes is None:
            return torch.zeros(units, dtype=torch.int64), 1
        if self.cluster_codes is None:
            return self.strata_codes, self.n_strata
        codes = torch.zeros(units, dtype=torch.int64)
        codes[self.cluster_codes] = self.strata_codes
        return codes, self.n_strata

    def relabel(self, frame: pd.DataFrame, rows: Tensor, slot: Tensor) -> None:
        """Stata's idcluster: every drawn copy of a cluster becomes a distinct cluster/panel."""
        uses_cluster = self.cluster in {self.spec.panel, *registry.cluster_columns(self.spec),
                                        *self._grouping_columns()}
        if uses_cluster:
            frame[self.cluster] = slot.numpy()
        for name, codes, count in self.grouping:
            frame[name] = (slot * count + codes[rows]).numpy()


def _refuse_unsupported(result: ResultBundle, procedure: str) -> None:
    spec, info = result.spec, registry.get(result.spec.estimator)
    if is_suest(result):
        raise AnalysisError(f"{procedure}_unsupported", f"A combined suest result cannot be "
                            f"refitted; apply {procedure} to the individual models.")
    if spec.panel is None and (spec.time is not None or (info.time != "none"
                                                         and info.panel == "none")):
        raise AnalysisError(
            f"{procedure}_unsupported",
            f"{spec.estimator} is a time-series model: resampling or deleting single "
            "observations destroys the serial dependence the model describes, so the "
            f"{procedure} covariance would be invalid. Use the estimator's own robust "
            "(HAC/OPG) covariance instead.")
    if result_covariance(result) in {"bootstrap", "jackknife"} or \
            spec.covariance in {"bootstrap", "jackknife"}:
        raise AnalysisError(f"{procedure}_unsupported", "The fit already uses a resampling "
                            "covariance; refit it with its conventional covariance first.")


_MAX_SEED = 2**64 - 1   # torch.Generator.manual_seed accepts seeds up to this value


def _seed(seed: Any) -> int:
    if seed is None:
        return secrets.randbits(62)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise AnalysisError("invalid_spec", "seed must be an integer between 0 and 2**64 - 1.")
    return seed


def _alpha_spec(result: ResultBundle, alpha: Any):
    if alpha is None:
        return result.spec
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_spec", "alpha must lie strictly between zero and one.")
    return result.spec.model_copy(update={"alpha": float(alpha)})


def _failure_note(failed: int, total: int, reasons: Counter, what: str) -> str:
    detail = ", ".join(f"{code}: {count}" for code, count in reasons.most_common())
    return f"{failed} of {total} {what} failed and were excluded ({detail})."


# Replicates whose spread is below this share of the estimates' magnitude differ only by
# floating-point rounding (an exact fit): their "covariance" is noise, not sampling variation.
_ROUNDING_SPREAD = 1e-12


def _require_variation(replicates: Tensor, observed: Tensor, terms: list[str]) -> None:
    """Refuse estimates that do not vary across replicates beyond rounding."""
    spread = (replicates - observed).abs().amax(dim=0)
    scale = torch.maximum(replicates.abs().amax(dim=0), observed.abs())
    flat = (spread <= _ROUNDING_SPREAD * scale).nonzero().flatten().tolist()
    if flat:
        raise AnalysisError(
            "invalid_covariance", f"The estimates of {', '.join(terms[i] for i in flat)} are "
            "identical in every replicate up to rounding (the model fits these data exactly), "
            "so a resampling covariance is not defined.")


def bootstrap(result: ResultBundle, data: Any, *, reps: int = 200, seed: int | None = None,
              cluster: str | None = None, strata: str | None = None, size: int | None = None,
              alpha: float | None = None, ci: str = "normal", scheme: str = "iid",
              block_length: int | None = None, time: str | None = None,
              null: dict[str, float] | None = None, wild: str = "rademacher",
              max_refits: int = 10000) -> ResultBundle:
    """Nonparametric bootstrap covariance of any fitted model (Stata's ``bootstrap`` prefix).

    The model ``result.spec`` is refitted on ``reps`` samples drawn with
    replacement from the estimation sample; the bootstrap covariance of the
    estimates is

        V = 1/(R-1) sum_r (b_r - b_bar)(b_r - b_bar)',   b_bar = mean of the b_r,

    over the R successful replicates (Stata's divisor). Coefficient tests use
    z = b / sqrt(V_jj) against the standard normal and the confidence
    intervals in the coefficient table are normal based, b -+ z_(1-alpha/2) se,
    as Stata reports them by default.

    Resampling units
    ----------------
    - observations (default): n rows drawn with replacement;
    - ``cluster='col'``: whole clusters drawn with replacement (Stata's
      ``cluster()``); every drawn copy is relabelled as a distinct cluster
      (Stata's ``idcluster()``), and so is the panel variable, so panel fixed
      effects of repeated panels stay distinct;
    - panel estimators (spec with a panel column) resample whole panels by
      default, using every data row of the estimation panels;
    - ``strata='col'``: draws are made independently within strata (and
      clusters must nest in strata);
    - frequency-weighted fits draw N = sum f units with probability
      proportional to f, the bootstrap of the expanded data; analytic and
      sampling weights travel with their rows.
    ``size`` sets the number of units drawn per stratum (Stata's ``size()``;
    default: all of them); it may not exceed the units in any stratum.

    Parameters
    ----------
    result : ResultBundle
        Any registry fit except time-series models and fits that already use
        a resampling covariance.
    data : DataFrame
        Exactly the dataset ``result`` was fitted on (verified by hash).
    reps : int
        Number of bootstrap replications R (at least 2; 200 by default, use
        1000 or more for percentile/bc intervals).
    seed : int, optional
        Seed (0 to 2**64 - 1) of the ``torch.Generator`` that draws every
        replicate; the same seed reproduces the result exactly. A random seed
        is drawn and recorded when omitted.
    cluster, strata : str, optional
        Columns defining resampling clusters and strata.
    size : int, optional
        Units drawn per stratum.
    alpha : float, optional
        Level of the intervals (default: the fit's alpha).
    ci : {'normal', 'percentile', 'bc', 'bca', 'studentized'}
        Interval type stored in ``extra['bootstrap_ci']``: normal based,
        percentile (Stata's ``_pctile`` definition) or bias-corrected
        percentile (z0 = Phi^-1(#{b_r <= b}/R)), BCa delete-unit acceleration
        or bootstrap-t using each refit's standard error. Advanced intervals
        currently require unweighted, unstratified full-size draws.
    scheme : {'iid', 'moving_block', 'residual', 'residual_block', 'wild_cluster'}
        Dependent schemes validate explicit-column OLS. Blocks require a regular
        time column and block_length; wild_cluster requires cluster=.
    null : dict, optional
        Coefficient restrictions for restricted wild cluster-t tests; requires
        the original covariance clustered on the same column and ci='normal'.
        Confidence-set inversion is not supplied.
    wild : {'rademacher', 'webb'}
        Wild cluster multiplier distribution.
    max_refits : int
        Advanced draw plus BCa acceleration budget (default 10,000).

    Every replicate refits the original specification; an ``mlogit`` fit
    without ``base=`` is refitted with the base category of the original fit,
    so that a resample with another most frequent outcome estimates the same
    parameters.

    Failed replicates
    -----------------
    A replicate fails when the estimator raises (e.g. a collinear or
    separated resample), when a numerical exception escapes it on a degenerate
    resample, or when it reports different terms (a regressor omitted for
    collinearity, a category absent). Failures are counted by reason and
    reported; more than 10% of ``reps`` raises
    ``AnalysisError('bootstrap_failed')``. Estimates that are identical in
    every replicate up to rounding (an exact fit) raise ``invalid_covariance``;
    a cluster column with one cluster raises ``insufficient_clusters``. When the
    fit clusters but single observations are resampled, a warning suggests
    ``cluster=``.

    Returns
    -------
    A new ResultBundle with the original estimates, the bootstrap covariance,
    z inference and a model Wald chi2 test (``tests['model']``). The original
    specification tests and ``extra`` are not carried over. ``inference``
    records ``covariance='bootstrap'``, ``reps``, ``reps_completed``,
    ``failed_replicates``, ``seed`` and the resampling unit;
    ``extra['bootstrap']`` holds per-term ``bias`` (mean of the replicates
    minus the estimate) and ``replicate_mean``; ``extra['bootstrap_ci']`` the
    requested interval type. ``result.spec`` still describes the point
    estimates (its ``covariance`` field is the original one).

    Stata: ``bootstrap, reps(200) seed(1) [cluster() idcluster() strata()
    size()]: command`` or ``command, vce(bootstrap)``; ``estat bootstrap,
    percentile bc``. SPSS: the BOOTSTRAP command. EViews: no general
    equivalent.

    Example::

        fit = oe.probit(data=df, y="inlf", x=["educ", "age"])
        boot = oe.bootstrap(fit, df, reps=500, seed=12, ci="percentile")
        boot.extra["bootstrap_ci"]
    """
    result = require_result(result, "result")
    if scheme != "iid" or ci in {"bca", "studentized"}:
        from openecon.econometrics.postest.dependent_resampling import advanced_bootstrap

        return advanced_bootstrap(result, data, reps=reps, seed=seed, cluster=cluster,
                                  strata=strata, size=size, alpha=alpha, ci=ci, scheme=scheme,
                                  block_length=block_length, time=time, null=null, wild=wild,
                                  max_refits=max_refits)
    if block_length is not None or time is not None or null is not None or wild != "rademacher":
        raise AnalysisError("invalid_spec", "Dependence options require an explicit bootstrap scheme.")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 2:
        raise AnalysisError("invalid_spec", "reps must be an integer of at least 2.")
    if ci not in _CI_KINDS:
        raise AnalysisError("invalid_spec", f"ci must be one of: {', '.join(_CI_KINDS)}.")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
        raise AnalysisError("invalid_spec", "size must be a positive integer.")
    _refuse_unsupported(result, "bootstrap")
    spec = _alpha_spec(result, alpha)
    seed_value = _seed(seed)
    population = _Population(result, data, cluster, strata, "bootstrap")
    unit_strata, n_strata = population.unit_strata()
    weighted = population.frequency is not None and population.cluster_codes is None
    if weighted:
        frequency = population.frequency
        assert frequency is not None
        totals = torch.zeros(n_strata, dtype=torch.float64).index_add_(0, unit_strata, frequency)
        available = totals.round().to(torch.int64)
        draw_table = WeightedStrata.build(unit_strata, n_strata, frequency)
    else:
        draw_table = Strata.build(unit_strata, n_strata)
        available = draw_table.count
    if size is not None and bool((available < size).any()):
        raise AnalysisError("invalid_spec", f"size={size} exceeds the {int(available.min())} "
                            "units available in some stratum.")
    sizes = available if size is None else torch.full_like(available, size)
    clusters = (Clusters.build(population.cluster_codes, population.n_clusters)
                if population.cluster_codes is not None else None)
    generator = torch.Generator().manual_seed(seed_value)
    refit = refitter(refit_spec(result))
    terms = [c.term for c in result.coefficients]
    replicates: list[Tensor] = []
    reasons: Counter = Counter()
    weight_column = result.spec.weights
    for _ in range(reps):
        if weighted:
            counts = draw_weighted(draw_table, sizes, population.n, generator)
            rows = counts.nonzero().flatten()
            frame = population.base.iloc[rows.numpy()].reset_index(drop=True)
            frame[weight_column] = counts[rows].numpy()
        elif clusters is not None:
            drawn = draw_units(draw_table, sizes, generator)
            rows, slot = clusters.expand(drawn)
            frame = population.base.iloc[rows.numpy()].reset_index(drop=True)
            population.relabel(frame, rows, slot)
        else:
            rows = draw_units(draw_table, sizes, generator)
            frame = population.base.iloc[rows.numpy()].reset_index(drop=True)
        values, reason = refit_parameters(refit, frame, terms)
        if values is None:
            reasons[reason] += 1
        else:
            replicates.append(values)
    failed = reps - len(replicates)
    warnings = list(population.notes)
    if failed:
        note = _failure_note(failed, reps, reasons, "bootstrap replications")
        if failed > FAILURE_SHARE * reps or len(replicates) < 2:
            raise AnalysisError("bootstrap_failed", f"{note} More than 10% of the replications "
                                "failed, so the bootstrap distribution is not reliable. Resample "
                                "clusters, merge rare categories or use a robust covariance.")
        warnings.append(note)
    stacked = torch.stack(replicates)
    observed = coefficient_vector(result)
    _require_variation(stacked, observed, terms)
    covariance = kernel_call(bootstrap_covariance, stacked)
    level = 1 - spec.alpha
    intervals: dict[str, Any] = {}
    if ci == "normal":
        from openecon.engines.distributions import normal_ppf

        z = normal_ppf(1 - spec.alpha / 2)
        errors = covariance.diagonal().clamp_min(0).sqrt()
        pairs = list(zip((observed - z * errors).tolist(), (observed + z * errors).tolist(),
                         strict=True))
    else:
        pairs = bootstrap_intervals(stacked, observed, spec.alpha, ci)
    undefined = []
    for term, (low, high) in zip(terms, pairs, strict=True):
        intervals[term] = {"ci_low": low, "ci_high": high}
        if low is None:
            undefined.append(term)
    if undefined:
        warnings.append("The bias-corrected interval is undefined for "
                        f"{', '.join(undefined)} (all replicates on one side of the estimate).")
    unit = (f"clusters of '{population.cluster}'" if clusters is not None
            else "frequency-weighted observations" if weighted else "observations")
    record = {
        "reps": reps, "reps_completed": len(replicates), "failed_replicates": failed,
        "seed": seed_value, "resampling_unit": unit, "strata": strata, "size": size,
        "cluster_column": population.cluster if clusters is not None else None,
        "cluster_count": population.n_clusters if clusters is not None else None,
    }
    mean = stacked.mean(dim=0)
    extra = {
        "bootstrap": {**record, "population": population.population,
                      "bias": dict(zip(terms, (mean - observed).tolist(), strict=True)),
                      "replicate_mean": dict(zip(terms, mean.tolist(), strict=True)),
                      "failure_reasons": dict(reasons)},
        "bootstrap_ci": {"type": ci, "confidence_level": level, "intervals": intervals},
    }
    provenance = {"postestimation": {
        "method": "bootstrap", **record, "population": population.population,
        "failure_reasons": dict(reasons), "generator": "torch.Generator (CPU) manual_seed",
        "idcluster": clusters is not None, "original_result_id": result.id,
        "original_covariance": result_covariance(result),
    }}
    title = f"{result.title or result.spec.estimator} (bootstrap standard errors)"
    tests = model_wald(result, covariance, df_resid=None, label_text="Wald chi2 test of the "
                       "coefficients (bootstrap covariance)")
    return rebuild(result, covariance, method="bootstrap", use_t=False, df_inference=None,
                   correction=f"bootstrap: covariance of {len(replicates)} replicates "
                              "(divisor R-1)", inference=record, provenance=provenance,
                   extra=extra, tests=tests, warnings=warnings, title=title, spec=spec)


def jackknife(result: ResultBundle, data: Any, *, cluster: str | None = None,
              max_refits: int = JACKKNIFE_MAX_REFITS) -> ResultBundle:
    """Delete-one (or delete-one-cluster) jackknife covariance (Stata's ``jackknife`` prefix).

    ``result.spec`` is refitted once per unit with that unit removed, giving
    the replicates b_(1), ..., b_(N). The jackknife covariance is

        V = (N-1)/N sum_i (b_(i) - b_bar)(b_(i) - b_bar)',   b_bar = mean of the b_(i),

    and coefficient tests use Student t with N - 1 degrees of freedom (Stata).
    Units are observations, or clusters with ``cluster='col'`` (N = G, df =
    G - 1); panel estimators delete one whole panel at a time by default.
    With frequency weights one replicated observation is deleted at a time
    (its weight is lowered by one); all copies of a row give the same
    replicate, so each row's replicate counts f_i times and N = sum f_i.
    Analytic and sampling weights stay attached to their rows.

    Parameters
    ----------
    result : ResultBundle
        Any registry fit except time-series models and fits that already use
        a resampling covariance.
    data : DataFrame
        Exactly the dataset ``result`` was fitted on (verified by hash).
    cluster : str, optional
        Delete one cluster at a time.
    max_refits : int
        Bound on the number of refits (default 2000). Larger problems raise
        ``AnalysisError('jackknife_too_large')``: use ``cluster=`` or
        ``oe.bootstrap`` instead.

    Failed replicates (estimator errors, numerical exceptions or omitted
    terms) are excluded and reported, N and the degrees of freedom count the
    successful ones; more than 10% failures raises ``jackknife_failed``.
    ``mlogit`` refits keep the original base category; a single cluster raises
    ``insufficient_clusters`` and estimates that do not vary beyond rounding
    raise ``invalid_covariance``.

    Returns
    -------
    A new ResultBundle with the original estimates, the jackknife covariance,
    t inference with N - 1 df and a model Wald F test. ``extra['jackknife']``
    holds ``refits``, ``failed_replicates``, the per-term Quenouille bias
    estimate ``(N-1)(b_bar - b)`` and ``replicate_mean``; ``inference``
    records ``covariance='jackknife'`` and the unit.

    Stata: ``jackknife [, cluster()]: command`` or ``command, vce(jackknife)``.

    Example::

        fit = oe.poisson(data=df, y="docvis", x=["age", "income"])
        jk = oe.jackknife(fit, df)
        jk.coefficients[1].std_error
    """
    result = require_result(result, "result")
    if isinstance(max_refits, bool) or not isinstance(max_refits, int) or max_refits < 2:
        raise AnalysisError("invalid_spec", "max_refits must be an integer of at least 2.")
    _refuse_unsupported(result, "jackknife")
    population = _Population(result, data, cluster, None, "jackknife")
    clustered = population.cluster_codes is not None
    units = population.n_clusters if clustered else population.n
    if units > max_refits:
        raise AnalysisError(
            "jackknife_too_large",
            f"The jackknife would need {units} refits (more than max_refits={max_refits}). Use "
            "cluster= to delete groups, oe.bootstrap, or raise max_refits.")
    if units < 2:
        raise AnalysisError("insufficient_observations", "The jackknife needs at least two units.")
    refit = refitter(refit_spec(result))
    terms = [c.term for c in result.coefficients]
    frequency = population.frequency if not clustered else None
    weight_column = result.spec.weights
    replicates: list[Tensor] = []
    multiplicity: list[float] = []
    reasons: Counter = Counter()
    codes = population.cluster_codes
    base = population.base
    for unit in range(units):
        if clustered:
            assert codes is not None
            frame = base.loc[(codes != unit).numpy()].reset_index(drop=True)
            count = 1.0
        elif frequency is not None and float(frequency[unit]) > 1:
            frame = base.copy()
            frame[weight_column] = frame[weight_column].astype("float64")
            frame.iloc[unit, frame.columns.get_loc(weight_column)] = float(frequency[unit]) - 1
            count = float(frequency[unit])
        else:
            keep = torch.ones(population.n, dtype=torch.bool)
            keep[unit] = False
            frame = base.loc[keep.numpy()].reset_index(drop=True)
            count = 1.0 if frequency is None else float(frequency[unit])
        values, reason = refit_parameters(refit, frame, terms)
        if values is None:
            reasons[reason] += 1
        else:
            replicates.append(values)
            multiplicity.append(count)
    failed = units - len(replicates)
    warnings = list(population.notes)
    if failed:
        note = _failure_note(failed, units, reasons, "jackknife replications")
        if failed > FAILURE_SHARE * units or len(replicates) < 2:
            raise AnalysisError("jackknife_failed", f"{note} More than 10% of the delete-one "
                                "refits failed; use cluster= or oe.bootstrap.")
        warnings.append(note)
    stacked = torch.stack(replicates)
    _require_variation(stacked, coefficient_vector(result), terms)
    weights = torch.tensor(multiplicity, dtype=torch.float64)
    covariance, mean, total = kernel_call(jackknife_covariance, stacked, weights)
    df = total - 1
    observed = coefficient_vector(result)
    unit_name = (f"clusters of '{population.cluster}'" if clustered
                 else "frequency-weighted observations" if frequency is not None
                 else "observations")
    record = {
        "refits": units, "reps_completed": len(replicates), "failed_replicates": failed,
        "resampling_unit": unit_name, "n_units": total,
        "cluster_column": population.cluster if clustered else None,
        "cluster_count": population.n_clusters if clustered else None,
        "cluster_df": population.n_clusters - 1 if clustered else None,
    }
    extra = {"jackknife": {
        **record, "population": population.population,
        "bias": dict(zip(terms, ((total - 1) * (mean - observed)).tolist(), strict=True)),
        "replicate_mean": dict(zip(terms, mean.tolist(), strict=True)),
        "failure_reasons": dict(reasons)}}
    provenance = {"postestimation": {
        "method": "jackknife", **record, "population": population.population,
        "failure_reasons": dict(reasons), "original_result_id": result.id,
        "original_covariance": result_covariance(result),
    }}
    title = f"{result.title or result.spec.estimator} (jackknife standard errors)"
    tests = model_wald(result, covariance, df_resid=df, label_text="Wald F test of the "
                       "coefficients (jackknife covariance)")
    return rebuild(result, covariance, method="jackknife", use_t=True, df_inference=df,
                   correction="jackknife: (N-1)/N sum (b_(i) - b_bar)(b_(i) - b_bar)', t(N-1)",
                   inference=record, provenance=provenance, extra=extra, tests=tests,
                   warnings=warnings, title=title)
