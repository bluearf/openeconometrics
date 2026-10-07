"""Cox proportional hazards regression: Stata's ``stcox``, SPSS ``COXREG``.

Model ``h(t | x) = h0_s(t) exp(x'b + offset)`` with an unspecified baseline hazard
``h0_s`` per stratum s. b maximizes the partial likelihood (``cox_kernels``) by
Newton-Raphson from b = 0 with the analytic score and Hessian; there is no
constant (a covariate that is constant within every stratum is not identified
and is omitted with a warning).

Data: ``time`` (exit), ``failure`` (0/1), ``entry`` (delayed entry and
``(start, stop]`` records), ``id`` (subject of multiple records), ``strata``.
Time-varying covariates come either from multiple records per subject or from
``tvc``: Stata's ``tvc(z) texp(_t)`` adds ``z * g(t)`` evaluated at each failure
time, which is implemented exactly as Stata describes it, by splitting every
record at the failure times inside its interval (one row per subject and risk
set; at most ``TVC_LIMIT`` rows).

Covariance (Stata [ST] stcox, Methods and formulas; [R] vce_option)
---------------------------------------------------------------------
* ``nonrobust``: inverse of the negative Hessian of the partial likelihood.
* ``robust``: Lin and Wei (1989): the sandwich with the efficient score
  residuals as scores, times ``N/(N-1)`` (Stata applies N/(N-1) unless
  ``noadjust``). When an ``id`` is declared, ``robust`` clusters on it, as stcox
  does ("stcox knew to specify vce(cluster id) for us").
* ``cluster``: residuals summed within clusters, times ``G/(G-1)``.

Weights (breslow only: "Weights are not supported with efron and exactp",
[ST] stcox): ``fweight`` (replication), ``iweight`` (as given) and ``pweight``
(robust covariance required). pweights are normalized to ``w N / sum w`` in the
log pseudolikelihood; coefficients and the robust covariance do not depend on
that scale, only the reported log pseudolikelihood (and AIC/BIC) does, and
whether Stata normalizes there is not documented (an uncertain convention).
``exactp`` takes neither ``tvc`` nor a robust or cluster covariance ("exactm and
exactp may not be specified with tvc(), vce(robust), or vce(cluster)").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call,
)
from openecon.econometrics.discrete.common import (
    Weights, build_spec, check_pweights, likelihood_covariance, maximize, model_test,
    optimizer_record, resolve_covariance,
)
from openecon.econometrics.survival import diagnostics
from openecon.econometrics.survival.cox_kernels import CoxObjective, ExactObjective
from openecon.econometrics.survival.data import (
    FLOAT, RiskSets, check_overlap, check_times, end_before_entry_note, failure_indicator,
    safe_exp,
)
from openecon.engines.absorb import demean
from openecon.engines.distributions import chi2_sf
from openecon.engines.inference import critical_value
from openecon.engines.optimize import information_inverse
from openecon.models import ModelSpec, ResultBundle

TVC_LIMIT = 5_000_000


@dataclass
class CoxSample:
    """Everything the Cox kernels need, built once from a spec and data."""

    frame: ModelFrame
    design: Design
    terms: list[str]
    x: Tensor                 # [rows, k] centred design (tvc rows after splitting)
    means: Tensor             # [k] centring of the main-equation columns (tvc: 0)
    risk: RiskSets
    failure: Tensor
    time: Tensor
    entry: Tensor | None
    weights: Weights
    offset: Tensor | None
    strata: Tensor | None
    strata_labels: list[Any]
    record: Tensor | None     # tvc: original record of every split row
    n_main: int
    ids: tuple[Tensor, int] | None


def _survival_columns(frame: ModelFrame) -> tuple[Tensor, Tensor | None, Tensor]:
    spec = frame.spec
    failure_name = (frame.role("failure") or [None])[0]
    entry_name = (frame.role("entry") or [None])[0]
    t = frame.numeric(spec.outcome)
    t0 = frame.numeric(entry_name) if entry_name else None
    delta = failure_indicator(frame.numeric(failure_name) if failure_name else None, frame.n,
                              failure_name)
    return t, t0, delta


def _weights(frame: ModelFrame) -> Weights:
    raw = frame.weights()
    if raw is None:
        return Weights(torch.ones(frame.n, dtype=FLOAT), None, frame.n, False)
    kind = frame.spec.weight_type
    if kind == "fweight":
        return Weights(raw, raw, int(round(float(raw.sum()))), True)
    if bool((raw < 0).any()):
        raise AnalysisError("negative_weights", "stcox needs nonnegative iweights.")
    if kind == "pweight":
        raw = raw * (frame.n / float(raw.sum()))     # w N / sum(w): see the module notes
    return Weights(raw, None, frame.n, True)


def prepare(spec: ModelSpec, data: Any, command: str = "stcox") -> CoxSample:
    """Sample, design, weights and risk sets of a Cox specification."""
    frame = ModelFrame(spec, data)
    ties = frame.option("ties")
    if ties in {"efron", "exactp"} and spec.weights is not None:
        raise AnalysisError("unsupported_weights", f"{command}: weights are not supported with "
                            f"ties='{ties}' (Stata's rule); use ties='breslow'.")
    tvc_names = frame.role("tvc")
    if ties == "exactp" and (tvc_names or spec.covariance != "nonrobust"):
        raise AnalysisError("invalid_spec", f"{command}: ties='exactp' may not be combined with "
                            "tvc or a robust/cluster covariance (Stata's rule).")
    check_pweights(frame)
    t, t0, delta = _survival_columns(frame)
    usable = check_times(t, t0, spec.outcome, (frame.role("entry") or [None])[0])
    if not bool(usable.all()):
        frame.restrict(usable, end_before_entry_note(int((~usable).sum())))
        t, t0, delta = _survival_columns(frame)
    weights = _weights(frame)
    strata_names = frame.role("strata")
    strata, labels = None, [None]
    if strata_names:
        strata, count = frame.codes(strata_names)
        labels = (frame.levels(strata_names[0]) if len(strata_names) == 1
                  else list(range(count)))
    ids = None
    if frame.role("id"):
        ids = frame.codes(frame.role("id")[0])
        check_overlap(t, t0 if t0 is not None else torch.zeros_like(t), ids[0])
    offset_names = frame.role("offset")
    offset = frame.numeric(offset_names[0]) if offset_names else None
    design = frame.design(intercept=False)
    screen = weights.for_screen()
    codes = strata if strata is not None else torch.zeros(frame.n, dtype=torch.int64)
    groups = int(codes.max()) + 1
    within = kernel_call(demean, design.x, [(codes, groups)], screen).values
    design, _ = frame.drop_absorbed(design, within, screen)
    n_main = len(design.terms)
    x = design.x
    if weights.weighted:
        means = (x * weights.user[:, None]).sum(0) / weights.user.sum()
    else:
        means = x.mean(0)
    x = x - means
    terms = list(design.terms)
    record = None
    risk = RiskSets.build(t, t0, delta, strata)
    if tvc_names:
        z = frame.matrix(tvc_names)
        x, t, t0, delta, weights, offset, strata, record, risk = _split(
            frame, x, z, t, t0, delta, weights, offset, strata, risk)
        terms += [f"tvc:{name}" for name in tvc_names]
        means = torch.cat([means, torch.zeros(len(tvc_names), dtype=FLOAT)])
    if not terms:
        raise AnalysisError("no_covariates", f"{command}: no covariate remains after omitting "
                            "those that are constant within strata.")
    if risk.n_events == 0:
        raise AnalysisError("no_failures", f"{command}: the sample has no failures, so the "
                            "partial likelihood carries no information.")
    return CoxSample(frame, design, terms, x, means, risk, delta, t, t0, weights, offset, strata,
                     labels, record, n_main, ids)


def _split(frame: ModelFrame, x: Tensor, z: Tensor, t: Tensor, t0: Tensor | None, delta: Tensor,
           weights: Weights, offset: Tensor | None, strata: Tensor | None, risk: RiskSets):
    """Split records at the failure times inside (entry, exit] for ``tvc`` (stsplit)."""
    texp = frame.option("texp")
    sizes = risk.exit_count - risk.entry_count
    total = int(sizes.sum())
    if total > TVC_LIMIT:
        raise AnalysisError("tvc_too_large", f"tvc needs one row per record and risk set: "
                            f"{total:,} rows (limit {TVC_LIMIT:,}). Coarsen the time scale "
                            "(fewer distinct failure times) or split the data yourself.")
    record = torch.repeat_interleave(torch.arange(t.numel()), sizes)
    starts = torch.cumsum(sizes, 0) - sizes
    local = torch.arange(total) - starts[record]
    event = risk.entry_count[record] + local
    exit_ = risk.event_time[event]
    previous = risk.event_time[(event - 1).clamp_min(0)]
    begin = torch.zeros_like(t) if t0 is None else t0
    entry = torch.where(local > 0, previous, begin[record])
    failed = (delta[record] > 0) & (risk.event_of[record] == event)
    g = exit_ if texp == "identity" else torch.log(exit_)
    rows_x = torch.cat([x[record], z[record] * g[:, None]], dim=1)
    w = Weights(weights.user[record], None if weights.frequency is None
                else weights.frequency[record], weights.nobs, weights.weighted)
    rows_strata = None if strata is None else strata[record]
    failure = failed.to(FLOAT)
    split_risk = RiskSets.build(exit_, entry, failure, rows_strata)
    frame.notes["tvc_rows"] = total
    return (rows_x, exit_, entry, failure, w, None if offset is None else offset[record],
            rows_strata, record, split_risk)


def _objective(sample: CoxSample, ties: str):
    if ties == "exactp":
        return kernel_call(ExactObjective, sample.x, sample.risk, sample.failure, sample.offset)
    return CoxObjective(sample.x, sample.risk, sample.weights.user if sample.weights.weighted
                        else None, sample.offset, ties)


def _score_rows(sample: CoxSample, objective: CoxObjective, beta: Tensor) -> Tensor:
    rows = objective.residuals(beta)[0]
    if sample.record is not None:
        rows = torch.zeros((sample.frame.n, rows.shape[1]), dtype=FLOAT).index_add_(
            0, sample.record, rows)
    return rows


def _covariance(sample: CoxSample, objective, beta: Tensor, hessian: Tensor):
    frame, weights = sample.frame, sample.weights
    kind = frame.spec.covariance
    base = weights
    if sample.record is not None:   # covariance units are the original records
        user = sample.frame.weights()
        base = _weights(frame) if user is not None else Weights(
            torch.ones(frame.n, dtype=FLOAT), None, frame.n, False)
    overrides: dict[str, Any] = {}
    if kind == "robust" and sample.ids is not None:
        overrides = {"kind": "cluster", "clusters": [sample.ids],
                     "cluster_names": frame.role("id")}
    covariance, info = likelihood_covariance(frame, hessian,
                                             lambda: _score_rows(sample, objective, beta),
                                             base, **overrides)
    if overrides:
        info["covariance"] = "robust"
        info["correction"] = "Lin-Wei sandwich clustered on the id variable: G/(G-1)"
    elif kind == "robust":
        info["correction"] = "Lin-Wei sandwich (efficient score residuals): N/(N-1)"
    elif kind == "cluster":
        info["correction"] = "Lin-Wei residuals summed within clusters: G/(G-1)"
    return covariance, info


def _score_test(objective, k: int) -> dict[str, Any]:
    _, gradient, hessian = objective(torch.zeros(k, dtype=FLOAT))
    try:
        inverse = kernel_call(information_inverse, -hessian)
    except AnalysisError:
        return {"statistic": None, "df": k, "p_value": None, "distribution": "chi2",
                "label": "Score (log-rank) test of b = 0: information singular at b = 0"}
    statistic = float(gradient @ inverse @ gradient)
    return {"statistic": statistic, "df": k, "p_value": chi2_sf(statistic, k),
            "distribution": "chi2", "label": "Score test of b = 0"}


def _subjects(sample: CoxSample) -> float:
    frame, weights = sample.frame, sample.frame.weights()
    if sample.ids is None:
        return float(weights.sum()) if frame.spec.weight_type == "fweight" else float(frame.n)
    codes, count = sample.ids
    if frame.spec.weight_type == "fweight":
        replicas = torch.zeros(count, dtype=FLOAT).scatter_reduce(0, codes, weights, "amax",
                                                                  include_self=False)
        return float(replicas.sum())
    return float(count)


def _hazard_ratios(terms: list[str], beta: Tensor, covariance: Tensor, alpha: float):
    z = critical_value(alpha, None)
    se = covariance.diagonal().sqrt()
    return {term: {"hazard_ratio": safe_exp(float(b)), "std_error": safe_exp(float(b)) * float(s),
                   "ci_low": safe_exp(float(b - z * s)), "ci_high": safe_exp(float(b + z * s))}
            for term, b, s in zip(terms, beta, se, strict=True)}


def fit_stcox(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``stcox``: ``(spec, data) -> ResultBundle`` (see the module notes)."""
    command = "stcox"
    sample = prepare(spec, data, command)
    frame, weights = sample.frame, sample.weights
    ties = frame.option("ties")
    k = len(sample.terms)
    objective = _objective(sample, ties)
    start = torch.zeros(k, dtype=FLOAT)
    result = maximize(objective, start, what=command, scale=weights.scale)
    beta = result.theta
    if not result.converged:
        spread = sample.x.std(0) if sample.x.shape[0] > 1 else torch.ones(k, dtype=FLOAT)
        # the index moves by more than 10 across one SD of a covariate: monotone likelihood
        if bool(((beta * spread).abs() > 10).any()):
            raise AnalysisError(
                "separation_detected",
                f"{command}: the partial likelihood keeps increasing as a coefficient diverges "
                "(monotone likelihood: a covariate separates failures from survivors, e.g. a "
                "group without failures). Drop or recode that covariate.")
        raise AnalysisError("nonconvergence", f"{command} did not converge: "
                            f"{result.diagnostics.get('message')} Check the scale of the "
                            "covariates.")
    covariance, info = _covariance(sample, objective, beta, result.hessian)
    log_likelihood = result.value
    null = float(objective.value(start))
    nobs = weights.nobs
    criteria = information_criteria(log_likelihood, k, nobs)
    frequency = frame.weights() if spec.weight_type == "fweight" else None
    base = frame.numeric(spec.outcome)
    entry_names = frame.role("entry")
    begin = frame.numeric(entry_names[0]) if entry_names else torch.zeros_like(base)
    original_failure = _survival_columns(frame)[2]
    counts = original_failure if frequency is None else original_failure * frequency
    at_risk = (base - begin) if frequency is None else (base - begin) * frequency
    tests = {"model": model_test(frame, beta, covariance, range(k), log_likelihood, null,
                                 null_model="b = 0", what="the coefficients"),
             "score": _score_test(objective, k)}
    extra: dict[str, Any] = {
        "ties": ties, "hazard_ratios": _hazard_ratios(sample.terms, beta, covariance, spec.alpha),
        "covariate_means": dict(zip(sample.terms[:sample.n_main],
                                    sample.means[:sample.n_main].tolist(), strict=True)),
        "strata": frame.role("strata") or None, "strata_levels": sample.strata_labels
        if frame.role("strata") else None,
        "partial_likelihood": "Breslow" if ties == "breslow" else
        ("Efron" if ties == "efron" else "exact partial (conditional logit for tied times)"),
    }
    concordance = None
    notes = []
    # Residuals, PH tests and baseline functions after exactp use the Peto-Breslow formulas
    # at the exact estimates ([ST] stcox postestimation: "If you specified breslow ...,
    # exactm, or exactp, all predictions are carried out using the Peto-Breslow method").
    residual_objective = objective
    if ties == "exactp":
        residual_objective = CoxObjective(sample.x, sample.risk, None, sample.offset, "breslow")
        notes.append("After ties='exactp' the PH tests and baseline functions use the "
                     "Peto-Breslow formulas at the exact estimates, as Stata does.")
    tests.update(_ph_tests(sample, residual_objective, beta, covariance, notes))
    if sample.record is None:
        shift = float(sample.means @ beta)
        increments, log_alpha = residual_objective.baseline(beta, shift)
        extra["baseline"] = diagnostics.baseline_table(sample.risk, increments, log_alpha,
                                                       labels=sample.strata_labels
                                                       if sample.strata is not None else None)
    else:
        notes.append("Baseline functions are not reported with tvc (they depend on the "
                     "time-varying covariate path).")
    single_record = sample.ids is None or sample.ids[1] == frame.n
    if (frame.option("concordance") and spec.weights is None and sample.record is None
            and (sample.entry is None or not bool((sample.entry > 0).any())) and single_record):
        eta = sample.x @ beta if sample.offset is None else sample.x @ beta + sample.offset
        harrell = diagnostics.harrell_c(eta, sample.time, sample.failure, sample.strata)
        concordance = harrell.get("concordance")
        extra["concordance"] = harrell
    elif frame.option("concordance"):
        notes.append("Harrell's C is not computed with weights, delayed entry, multiple records "
                     "per subject or tvc (Stata's restrictions).")
    if notes:
        extra["notes"] = notes
    metrics = {"log_likelihood": log_likelihood, "log_likelihood_null": null,
               "aic": criteria["aic"], "bic": criteria["bic"], "df_model": k,
               "n_subjects": _subjects(sample), "n_failures": float(counts.sum()),
               "time_at_risk": float(at_risk.sum()), "concordance": concordance}
    if spec.covariance == "nonrobust":
        info.setdefault("correction", "observed information (inverse negative Hessian)")
    equations = None
    if frame.role("tvc"):
        equations = ["main"] * sample.n_main + ["tvc"] * (k - sample.n_main)
    provenance = {"ties": ties, "bic_n": "number of observations (records)",
                  "tvc_rows": frame.notes.get("tvc_rows")}
    return build_result(
        frame, terms=sample.terms, params=beta, covariance=covariance, use_t=False,
        equations=equations, metrics=metrics, nobs=nobs, inference=info, tests=tests,
        extra=extra, categories=sample.design.categories, provenance=provenance,
        solver="newton_partial_likelihood",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def _ph_tests(sample: CoxSample, objective: CoxObjective, beta: Tensor, covariance: Tensor,
              notes: list[str]) -> dict[str, Any]:
    if sample.record is not None:
        notes.append("Schoenfeld-residual PH tests are not computed after tvc (Stata's estat "
                     "phtest is not allowed after tvc()).")
        return {}
    transform = sample.frame.option("phtest")
    _, means = objective.residuals(beta)
    fail = objective.failing
    schoenfeld = sample.x[fail] - means[objective.event_of[fail]]
    times = sample.time[fail]
    weights = sample.weights.user[fail]
    g = diagnostics.time_function(transform, times, sample.time, sample.entry, sample.failure,
                                  sample.weights.user)
    return diagnostics.ph_tests(schoenfeld, g, weights, covariance, beta, sample.terms, transform)


def stcox(*, data: Any, time: str, x: Sequence[str], failure: str | None = None,
          entry: str | None = None, id: str | None = None,
          strata: str | Sequence[str] | None = None, ties: str = "breslow",
          covariance: str | None = None, cluster: str | None = None,
          weights: str | None = None, weight_type: str | None = None, offset: str | None = None,
          tvc: Sequence[str] | None = None, texp: str = "identity",
          phtest: str = "identity", concordance: bool = True,
          categorical: Sequence[str] | None = None, missing: str = "raise",
          alpha: float = 0.05) -> ResultBundle:
    """Cox proportional hazards regression (Stata ``stcox``; SPSS ``COXREG``).

    Model
        ``h(t | x) = h0_s(t) exp(x'b + offset)``: the hazard of failure at analysis time t
        is an unspecified baseline hazard ``h0_s`` of the record's stratum times the
        relative hazard ``exp(x'b)``. ``exp(b_j)`` is the hazard ratio of a unit increase
        in ``x_j``. There is no constant.

    Estimator
        Newton-Raphson on the partial likelihood with analytic score and Hessian built
        from risk-set sums (one sort, then O(n k^2) per iteration). ``ties``:
        ``'breslow'`` (Peto-Breslow, Stata's default),
        ``ln L = sum_k [sum_{D_k} w (x'b) - d_k ln sum_{R_k} w exp(x'b)]``; ``'efron'``
        (``sum_r ln(S0 - r/d_k sum_{D_k} exp(x'b))`` over the d_k tied failures);
        ``'exactp'`` (exact partial likelihood: the conditional-logit probability of the
        tied failure set within its risk set, by the elementary-symmetric-function
        recursion; at most 5,000,000 expanded risk-set rows).

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        time: exit time (analysis time of failure or censoring), >= 0.
        x: covariates. Covariates constant within every stratum are omitted.
        failure: 0/1 failure indicator (default: every record fails).
        entry: entry time: delayed entry (left truncation) or the start of ``(start,
            stop]`` records; a record is at risk at s when ``entry < s <= time``.
        id: subject identifier for multiple records per subject (time-varying covariates
            by episode splitting). Records of one id may not overlap. With ``id``,
            ``covariance='robust'`` clusters on it (Stata's rule).
        strata: column(s) with separate baseline hazards (``strata(varnames)``).
        covariance: ``'nonrobust'`` (default; inverse negative Hessian), ``'robust'``
            (Lin-Wei sandwich with efficient score residuals times N/(N-1)), ``'cluster'``
            (residuals summed within ``cluster``, times G/(G-1)).
        cluster: cluster column for ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'iweight'`` or ``'pweight'`` (pweights imply
            ``'robust'``; they are normalized to ``w N / sum w`` in the reported log
            pseudolikelihood); breslow only.
        offset: column entering ``x'b`` with coefficient 1.
        tvc: covariates interacted with ``g(t)`` (``texp``: ``'identity'`` = t, Stata's
            default ``texp(_t)``, or ``'log'``); reported as ``tvc:<name>`` terms.
        phtest: time function of the Schoenfeld-residual PH tests: ``'identity'``
            (Stata's default, analysis time), ``'log'``, ``'km'`` (1 - Kaplan-Meier),
            ``'rank'``.
        concordance: compute Harrell's C (single-record unweighted data without delayed
            entry or tvc).
        categorical: covariates to expand into treatment-coded indicators.
        missing: ``'raise'`` (default) or ``'drop'``.
        alpha: significance level of the intervals.

    Result
        Coefficients are log hazard ratios with z tests. ``metrics``: log_likelihood
        (partial), log_likelihood_null (b = 0), aic, bic (N = observations), df_model,
        n_subjects, n_failures, time_at_risk, concordance. ``tests``: ``model`` (LR chi2(k)
        for nonrobust, Wald otherwise), ``score`` (score test of b = 0), ``ph_global`` and
        ``ph_<term>`` (Grambsch-Therneau; with ``rho``). ``extra``: hazard_ratios (with
        delta-method std_error and exp-transformed CIs), baseline (at covariates and
        offset zero, at most 400 times; ``oe.stcurve`` gives the full curve): the Breslow
        cumulative hazard H0 (Efron's averaged risk sets after ties='efron'; Stata's
        ``predict basechazard``), ``survivor`` = exp(-H0) and ``survivor_kp``, the
        Kalbfleisch-Prentice product-limit estimator of Stata's ``predict basesurv``;
        covariate_means, concordance details, ties.

    Stata
        ``stset t, failure(died) id(id)`` + ``stcox age drug, efron vce(robust)`` is
        ``oe.stcox(data=df, time='t', failure='died', id='id', x=['age', 'drug'],
        ties='efron', covariance='robust')``; ``estat phtest, detail`` is in
        ``result.tests``; ``estat concordance`` is ``result.metrics['concordance']``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"age": rng.normal(60, 8, 500), "drug": rng.integers(0, 2, 500)})
        >>> t = rng.exponential(1 / np.exp(0.03 * (df.age - 60) - 0.7 * df.drug))
        >>> df["t"], df["died"] = np.minimum(t, 2.0), (t < 2.0).astype(int)
        >>> print(oe.stcox(data=df, time="t", failure="died", x=["age", "drug"]).summary())
    """
    spec = build_spec(
        "stcox", outcome=time, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=False,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"failure": failure, "entry": entry, "id": id,
                 "strata": column_list([strata] if isinstance(strata, str) else strata,
                                       "strata"),
                 "offset": offset, "tvc": column_list(tvc, "tvc")},
        options={"ties": ties, "texp": texp, "phtest": phtest, "concordance": concordance})
    return fit(spec, data=data)
