"""Recurrent-event Cox models: Andersen-Gill, Prentice-Williams-Peterson and Wei-Lin-Weissfeld.

Input: counting-process records, one row per ``(entry, time]`` interval of a
subject (``id``), with ``failure`` = 1 when the interval ends in an event.
Records of one subject must not overlap. With the records of a subject ordered
by entry, ``prior_r`` is the number of events of the subject in earlier
records and ``last_r`` the time of the most recent of them (0 if none).

* ``model='ag'`` (Andersen and Gill 1982): one risk set on the total-time scale,
  every record ``(entry, time]``.
* ``model='pwp_total'`` (Prentice, Williams and Peterson 1981, total time):
  records stratified by event number ``prior_r + 1``; a subject is at risk for
  its k-th event only after its (k-1)-th.
* ``model='pwp_gap'`` (PWP gap time): the same strata on the time since the
  previous event, ``(entry - last_r, time - last_r]``.
* ``model='wlw'`` (Wei, Lin and Weissfeld 1989, marginal): strata k = 1..K, K the
  largest number of events of a subject (or ``max_events``). Stratum k holds every
  record of a subject that starts before its k-th event (``prior_r < k``) on the
  total-time scale, with an event only on the record that ends at the k-th event;
  subjects with fewer events are censored at the end of follow-up. The data are
  expanded to one row per record and stratum (at most ``EXPANSION_LIMIT`` rows).

``max_events`` caps the strata: PWP pools event numbers above it into the last
stratum; WLW models the first ``max_events`` events only.

Estimation is the Cox partial likelihood (``cox_kernels``, Breslow or Efron ties)
by Newton-Raphson. Covariance (``robust``, the default) is the Lin-Wei sandwich
with the efficient score residuals of every (expanded) row summed within subject,
times G/(G-1) (Stata's ``vce(cluster id)``; R ``coxph(... + cluster(id))`` applies
no G/(G-1)); ``nonrobust`` is the inverse information, which ignores the
dependence of a subject's events (reported for comparison only).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call,
)
from openecon.econometrics.discrete.common import (
    Weights, build_spec, likelihood_covariance, maximize, model_test, optimizer_record,
)
from openecon.econometrics.survival.cox import _hazard_ratios, _survival_columns
from openecon.econometrics.survival.cox_kernels import CoxObjective
from openecon.econometrics.survival.data import (
    FLOAT, RiskSets, check_overlap, check_times, end_before_entry_note,
)
from openecon.engines.absorb import demean
from openecon.models import ModelSpec, ResultBundle

EXPANSION_LIMIT = 5_000_000
TIE_TOLERANCE = 1.4901161193847656e-08      # sqrt(float64 eps), R survival's aeqSurv default
MODELS = ("ag", "pwp_total", "pwp_gap", "wlw")
_TITLES = {"ag": "Andersen-Gill", "pwp_total": "Prentice-Williams-Peterson (total time)",
           "pwp_gap": "Prentice-Williams-Peterson (gap time)", "wlw": "Wei-Lin-Weissfeld marginal"}


def event_history(time: Tensor, entry: Tensor, failure: Tensor, ids: Tensor
                  ) -> tuple[Tensor, Tensor]:
    """Prior events and the time of the last prior event of every record (exact, loop free)."""
    n = time.numel()
    order = torch.argsort(entry, stable=True)
    order = order[torch.argsort(ids[order], stable=True)]
    sorted_ids, sorted_fail = ids[order], failure[order]
    position = torch.arange(n)
    first = torch.ones(n, dtype=torch.bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    start = torch.cummax(torch.where(first, position, torch.zeros_like(position)), 0).values
    counts = torch.cumsum(sorted_fail, 0)
    before = torch.where(start > 0, counts[(start - 1).clamp_min(0)], torch.zeros_like(counts))
    prior_sorted = counts - sorted_fail - before
    marks = torch.where(sorted_fail > 0, position, torch.full_like(position, -1))
    last = torch.cummax(marks, 0).values
    exclusive = torch.full_like(last, -1)
    exclusive[1:] = last[:-1]
    has = exclusive >= start
    sorted_time = time[order]
    last_time = torch.where(has, sorted_time[exclusive.clamp_min(0)], torch.zeros_like(sorted_time))
    prior = torch.empty_like(prior_sorted)
    prior[order] = prior_sorted
    previous = torch.empty_like(last_time)
    previous[order] = last_time
    return prior.round().to(torch.int64), previous


def snap_times(entry: Tensor, exit_: Tensor, tolerance: float = TIE_TOLERANCE
               ) -> tuple[Tensor, Tensor]:
    """Merge derived times closer than ``tolerance`` (absolute, or relative to the mean |time|).

    The rule of R survival's ``aeqSurv`` (``coxph`` timefix): sorted distinct values
    whose consecutive differences are within the tolerance form one tied run, which
    takes its smallest value. Applied to gap times only, whose subtraction
    ``time - last`` leaves rounding differences between equal decimal gaps.
    """
    values = torch.cat([entry, exit_])
    unique = torch.unique(values, sorted=True)
    if unique.numel() < 2:
        return entry, exit_
    scale = float(unique.abs().mean())
    steps = unique[1:] - unique[:-1]
    tied = (steps <= tolerance) | (steps / scale <= tolerance)
    if not bool(tied.any()):
        return entry, exit_
    run = torch.zeros(unique.numel(), dtype=torch.int64)
    run[1:] = torch.cumsum((~tied).to(torch.int64), 0)
    first = torch.zeros(int(run[-1]) + 1, dtype=torch.int64).scatter_reduce(
        0, run, torch.arange(unique.numel()), "amin", include_self=False)
    representative = unique[first[run]]
    snapped = representative[torch.searchsorted(unique, values)]
    n = entry.numel()
    new_entry, new_exit = snapped[:n], snapped[n:]
    if bool(((new_exit <= new_entry) & (exit_ > entry)).any()):
        raise AnalysisError("invalid_survival_time", "A gap-time record shorter than the tie "
                            "tolerance collapses to length zero; remove or merge it.")
    return new_entry, new_exit


def fit_stcox_recurrent(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``stcox_recurrent``: ``(spec, data) -> ResultBundle`` (module notes)."""
    command = "stcox_recurrent"
    frame = ModelFrame(spec, data)
    model, ties, cap = frame.option("model"), frame.option("ties"), frame.option("max_events")
    entry_name = (frame.role("entry") or [None])[0]
    t, t0, delta = _survival_columns(frame)
    usable = check_times(t, t0, spec.outcome, entry_name)
    if not bool(usable.all()):
        frame.restrict(usable, end_before_entry_note(int((~usable).sum())))
        t, t0, delta = _survival_columns(frame)
    begin = t0 if t0 is not None else torch.zeros_like(t)
    id_name = frame.role("id")[0]
    ids, subjects = frame.codes(id_name)
    check_overlap(t, begin, ids)
    if subjects < 2:
        raise AnalysisError("insufficient_clusters", f"{command} needs at least two subjects.")
    prior, last = event_history(t, begin, delta, ids)
    events_total = prior + delta.round().to(torch.int64)
    largest = int(events_total.max())
    if largest == 0:
        raise AnalysisError("no_failures", f"{command}: the sample has no events.")
    design = frame.design(intercept=False)
    zeros = torch.zeros(frame.n, dtype=torch.int64)
    within = kernel_call(demean, design.x, [(zeros, 1)], None).values
    design, _ = frame.drop_absorbed(design, within, None)
    if not design.terms:
        raise AnalysisError("no_covariates", f"{command}: no covariate varies in the sample.")
    x = design.x - design.x.mean(0)
    record = torch.arange(frame.n)
    exit_, start, failure, strata = t, begin, delta, None
    strata_count = 1
    if model in {"pwp_total", "pwp_gap"}:
        # Records after a subject's last event form the (event-free) next stratum, as in
        # R's strata(enum); max_events pools every event number above the cap.
        strata = prior if cap is None else prior.clamp_max(cap - 1)
        strata_count = int(strata.max()) + 1
        if model == "pwp_gap":
            start, exit_ = snap_times(begin - last, t - last)
    elif model == "wlw":
        top = largest if cap is None else min(cap, largest)
        copies = (top - prior).clamp_min(0)
        total = int(copies.sum())
        if total > EXPANSION_LIMIT:
            raise AnalysisError("resource_limit", f"{command}: the marginal (WLW) data need "
                                f"{total:,} rows (limit {EXPANSION_LIMIT:,}); set max_events.")
        record = torch.repeat_interleave(torch.arange(frame.n), copies)
        offsets = torch.cumsum(copies, 0) - copies
        local = torch.arange(total) - offsets[record]
        strata = prior[record] + local
        failure = delta[record] * (local == 0).to(FLOAT)
        exit_, start = t[record], begin[record]
        strata_count = top
    frame.workspace_plan("recurrent-event rows and risk sets", {
        "expanded_rows": 8 * record.numel() * (x.shape[1] + 8)})
    rows = x[record]
    has_entry = bool((start > 0).any())
    risk = RiskSets.build(exit_, start if has_entry else None, failure, strata)
    if risk.n_events == 0:
        raise AnalysisError("no_failures", f"{command}: the sample has no events.")
    objective = CoxObjective(rows, risk, None, None, ties)
    k = len(design.terms)
    result = maximize(objective, torch.zeros(k, dtype=FLOAT), what=command)
    if not result.converged:
        spread = rows.std(0) if rows.shape[0] > 1 else torch.ones(k, dtype=FLOAT)
        if bool(((result.theta * spread).abs() > 10).any()):
            raise AnalysisError("separation_detected", f"{command}: the partial likelihood keeps "
                                "increasing as a coefficient diverges (a covariate separates "
                                "events from survivors). Drop or recode it.")
        raise AnalysisError("nonconvergence", f"{command} did not converge: "
                            f"{result.diagnostics.get('message')}")
    beta = result.theta
    unweighted = Weights(torch.ones(frame.n, dtype=FLOAT), None, frame.n, False)

    def scores() -> Tensor:
        residuals = objective.residuals(beta)[0]
        return torch.zeros((frame.n, k), dtype=FLOAT).index_add_(0, record, residuals)

    overrides: dict[str, Any] = {}
    if spec.covariance == "robust":
        overrides = {"kind": "cluster", "clusters": [(ids, subjects)], "cluster_names": [id_name]}
    covariance, info = likelihood_covariance(frame, result.hessian, scores, unweighted,
                                             **overrides)
    if overrides:
        info.update({"covariance": "robust", "correction": "Lin-Wei sandwich of the score "
                     "residuals summed within subject: G/(G-1)"})
    else:
        info["correction"] = "inverse information (ignores within-subject dependence)"
    log_likelihood = float(result.value)
    null = float(objective.value(torch.zeros(k, dtype=FLOAT)))
    criteria = information_criteria(log_likelihood, k, frame.n)
    tests = {"model": model_test(frame, beta, covariance, range(k), log_likelihood, null,
                                 null_model="b = 0", what="the coefficients")}
    layout = []
    if strata is not None:
        row_counts = torch.bincount(strata, minlength=strata_count)
        event_counts = torch.zeros(strata_count, dtype=FLOAT).index_add_(0, strata, failure)
        layout = [{"stratum": i + 1, "rows": int(r), "events": float(e)}
                  for i, (r, e) in enumerate(zip(row_counts.tolist(), event_counts.tolist(),
                                                 strict=True))]
    metrics = {"log_likelihood": log_likelihood, "log_likelihood_null": null,
               "aic": criteria["aic"], "bic": criteria["bic"], "df_model": k,
               "n_subjects": subjects, "n_records": frame.n, "n_rows": int(record.numel()),
               "n_failures": float(delta.sum()), "n_strata": strata_count,
               "max_events": largest, "time_at_risk": float((t - begin).sum())}
    extra = {"model": model, "ties": ties, "time_scale": "gap" if model == "pwp_gap" else "total",
             "hazard_ratios": _hazard_ratios(design.terms, beta, covariance, spec.alpha),
             "strata": layout, "max_events_option": cap, "id": id_name,
             "stratification": {"ag": "none", "pwp_total": "event number",
                                "pwp_gap": "event number", "wlw": "event number (marginal)"}[model]}
    return build_result(
        frame, terms=design.terms, params=beta, covariance=covariance, use_t=False,
        metrics=metrics, nobs=frame.n, inference=info, tests=tests, extra=extra,
        categories=design.categories, title=f"{_TITLES[model]} recurrent-event Cox regression",
        provenance={"ties": ties, "model": model, "bic_n": "number of records"},
        solver="newton_partial_likelihood",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def stcox_recurrent(*, data: Any, time: str, x: Sequence[str], id: str, failure: str,
                    entry: str | None = None, model: str = "ag", max_events: int | None = None,
                    ties: str = "breslow", covariance: str | None = None,
                    categorical: Sequence[str] | None = None, missing: str = "raise",
                    alpha: float = 0.05) -> ResultBundle:
    """Recurrent-event Cox regression: Andersen-Gill, PWP total/gap time or WLW marginal.

    Model
        Counting-process records ``(entry, time]`` per subject ``id`` with ``failure`` = 1
        when a record ends in an event. ``'ag'``: one total-time risk set; ``'pwp_total'``
        and ``'pwp_gap'``: strata by event number on total or gap time (time since the
        previous event); ``'wlw'``: marginal strata k = 1..K holding each subject's
        records up to its k-th event.

    Estimator
        Cox partial likelihood (Breslow or Efron ties) by Newton-Raphson on the derived
        layout; robust covariance clusters the efficient score residuals on ``id``.

    Parameters
        data: DataFrame. time, entry: record exit and entry (entry 0 when omitted).
        x: covariates (may vary between records). id: subject column (required).
        failure: 0/1 event indicator at the record's exit.
        model: ``'ag'`` (default), ``'pwp_total'``, ``'pwp_gap'`` or ``'wlw'``.
        max_events: cap on strata (PWP pools higher event numbers; WLW models the first
            ``max_events`` events).
        ties: ``'breslow'`` (default) or ``'efron'``.
        covariance: ``'robust'`` (default; Lin-Wei clustered on id, G/(G-1)) or
            ``'nonrobust'``.
        categorical, missing, alpha: as in ``oe.stcox``.

    Result
        Log hazard ratios with z tests. ``metrics``: log_likelihood, n_subjects, n_records,
        n_rows (expanded WLW rows), n_failures, n_strata, max_events. ``tests['model']``:
        Wald chi2 (robust) or LR chi2. ``extra``: hazard_ratios, strata (rows and events).

    Stata / R
        AG: ``stset stop, id(id) failure(event) time0(start) exit(time .)`` then
        ``stcox x, vce(robust)``; R ``coxph(Surv(start, stop, event) ~ x + cluster(id))``.
        PWP: add ``strata(enum)``; gap time uses ``Surv(stop - last, ...)``.

    Example
        >>> import pandas as pd, openecon as oe
        >>> df = pd.DataFrame({"id": [1, 1, 2, 2, 3, 4], "start": [0, 1, 0, 2, 0, 0],
        ...                    "stop": [1, 3, 2, 4, 3, 2.5], "event": [1, 0, 1, 1, 0, 1],
        ...                    "x": [0.5, 0.5, -1.0, -1.0, 0.2, 1.4]})
        >>> fit = oe.stcox_recurrent(data=df, time="stop", entry="start", id="id",
        ...                          failure="event", x=["x"], model="pwp_gap")
    """
    spec = build_spec(
        "stcox_recurrent", outcome=time, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=False,
        covariance=covariance, missing=missing, alpha=alpha,
        columns={"failure": failure, "entry": entry, "id": id},
        options={"model": model, "max_events": max_events, "ties": ties})
    return fit(spec, data=data)
