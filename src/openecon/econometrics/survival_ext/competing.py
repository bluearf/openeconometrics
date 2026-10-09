"""Aalen--Johansen/Cause-specific Nelson--Aalen curves and fixed-time contrasts.

Infinitesimal-jackknife derivatives are propagated through the exact tied
risk-set recursion. The reported covariance is the Gram matrix of the subject
case-weight derivatives, including every requested cause/time cross term.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.nonparametric.common import procedure
from openecon.engines.distributions import chi2_sf, normal_sf
from .common import confidence, event_data, output, positional_list, prediction_times, workspace

REFERENCES = [
    "https://stat.ethz.ch/R-manual/R-devel/RHOME/library/survival/doc/compete.pdf",
    "https://doi.org/10.1007/s10985-023-09597-5",
]


def _causes(causes, event):
    if causes is None:
        result = sorted(set(event[event > 0].tolist()))
        if not result:
            raise AnalysisError("invalid_cause", "With no observed failures, supply the positive cause codes to estimate.")
    else:
        result = positional_list(causes, "causes", limit=256)
    if any(isinstance(v, bool) or not isinstance(v, Real) or v <= 0 or v > 2**31-1
           or not math.isfinite(float(v)) or int(v) != v for v in result):
        raise AnalysisError("invalid_cause", "causes require distinct positive integer codes (<=2^31-1).")
    result = [int(v) for v in result]
    if len(set(result)) != len(result):
        raise AnalysisError("invalid_cause", "causes must be distinct; caller order is retained.")
    return result


def _grid(times, t):
    grid = prediction_times(times, default=torch.unique(t, sorted=True).tolist())
    if float(grid[-1]) > float(t.max()):
        raise AnalysisError("outside_support", "Requested times cannot exceed the last observed follow-up time; no tail extrapolation.")
    return grid


@torch.no_grad()
def _recursion(t, event, causes, grid, *, hazard=False):
    """Differentiate weighted event/risk totals at unit case weights.

    A derivative is d estimate / d w_i at weights w_i=1, rather than its n
    times scaled influence function. Scale invariance makes its row sum zero;
    D D' is the unweighted iid IJ plug-in covariance, without a df correction.
    Only causes actually jumping at a time need a vector update.
    """
    n, k = len(t), len(causes)
    estimate = torch.zeros(k, dtype=torch.float64, device="cpu")
    derivative = torch.zeros((k, n), dtype=torch.float64, device="cpu")
    survival = 1.
    survival_derivative = torch.zeros(n, dtype=torch.float64, device="cpu")
    observed_times = torch.unique(t, sorted=True).tolist()
    observed_causes = torch.unique(event[event > 0]).tolist()
    positions = {cause: j for j, cause in enumerate(causes)}
    records, at_times, gradients, survival_at_times = [], [], [], []
    steps = []
    for u in observed_times:
        at = t == u
        risk = t >= u
        failures = at & (event > 0)
        counts = {int(code): int(((event == code) & at).sum())
                  for code in torch.unique(event[failures]).tolist()}
        at_risk, n_failures = int(risk.sum()), int(failures.sum())
        steps.append((u, at_risk, n_failures, counts))
        records.append(dict(time=u, n_risk=at_risk, events_total=n_failures,
                            censored=int((at & (event == 0)).sum()),
                            cause_counts=json.dumps(counts, sort_keys=True)))
    cursor = 0
    for horizon in grid.tolist():
        while cursor < len(steps) and steps[cursor][0] <= horizon:
            u, at_risk, n_failures, counts = steps[cursor]
            risk_mask = t >= u
            failures = (t == u) & (event > 0)
            risk = risk_mask.to(torch.float64)
            total_jump = n_failures/at_risk
            for cause, count in counts.items():
                if cause not in positions:
                    continue
                j = positions[cause]
                jump = count/at_risk
                individual = (failures & (event == cause)).to(torch.float64)
                jump_derivative = (individual-jump*risk)/at_risk
                if hazard:
                    estimate[j] += jump
                    derivative[j] += jump_derivative
                else:
                    estimate[j] += survival*jump
                    derivative[j] += survival_derivative*jump+survival*jump_derivative
            if not hazard:
                total_derivative = (failures.to(torch.float64)-total_jump*risk)/at_risk
                survival_derivative = survival_derivative*(1-total_jump)-survival*total_derivative
                survival *= 1-total_jump
            cursor += 1
        if not hazard and len(observed_causes) == 1 and observed_causes[0] in positions:
            # The sole observed cause CIF is exactly 1-S. Using that identity
            # preserves its exactly zero IJ at an empirical terminal boundary,
            # instead of treating accumulated roundoff as positive variance.
            j = positions[observed_causes[0]]
            estimate[j] = 1-survival
            derivative[j] = -survival_derivative
        at_times.append(estimate.clone())
        gradients.append(derivative.clone())
        survival_at_times.append(survival if not hazard else None)
    values = torch.stack(at_times).reshape(-1)
    d = torch.cat(gradients, dim=0)
    covariance = d@d.T
    covariance = (covariance+covariance.T)/2
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(covariance).all()):
        raise AnalysisError("numerical_failure", "Competing-risk recursion or complete covariance is nonfinite.")
    zero_sum_error = float(d.sum(dim=1).abs().max())
    if zero_sum_error > 2e-10 * max(float(d.abs().sum(dim=1).max()), 1e-300):
        raise AnalysisError("numerical_failure", "Case-weight scale-invariance derivative gate failed.")
    return values, covariance, records, survival_at_times, zero_sum_error


def _curves(method, time, event, *, causes, times, level, device, weights, hazard):
    data = event_data(time, event, device=device, weights=weights)
    selected = _causes(causes, data.event)
    grid = _grid(times, data.t)
    budget = workspace(data.n, len(grid)*len(selected))
    level, critical = confidence(level)
    values, covariance, risksets, survival, derivative_error = _recursion(data.t, data.event, selected, grid, hazard=hazard)
    standard_errors = covariance.diagonal().clamp_min(0).sqrt()
    rows, components, labels = [], [], []
    index = 0
    for horizon in grid.tolist():
        for cause in selected:
            estimate, se = float(values[index]), float(standard_errors[index])
            lo, hi = estimate-critical*se, estimate+critical*se
            if not hazard:
                lo, hi = max(0., lo), min(1., hi)
            else:
                lo = max(0., lo)
            if se == 0:
                lo, hi = None, None
            label = f"{index}:time={horizon:g},cause={cause}"
            labels.append(label)
            rows.append(dict(time=horizon, cause=cause, estimate=estimate, standard_error=se,
                             ci_lower=lo, ci_upper=hi, n_risk=int((data.t >= horizon).sum()),
                             inference_status="unavailable: zero plug-in variance" if se == 0 else "pointwise asymptotic normal"))
            components.append(dict(component=index, label=label, time=horizon, cause=cause))
            index += 1
    settings = dict(data.settings, **budget, causes=selected, times=grid.tolist(), level=level,
                    observed_causes=sorted(set(data.event[data.event > 0].tolist())),
                    estimand="cause-specific cumulative Nelson-Aalen hazard" if hazard else "Aalen-Johansen cumulative incidence of mutually exclusive cause",
                    covariance="unweighted iid infinitesimal jackknife: Gram matrix of recursive case-weight derivatives; full cross-time/cause covariance; no n/(n-1) correction",
                    component_order="time-major, caller cause order; components table labels the full square covariance",
                    tie_policy="exact tied failures share Y(t-), including tied censors; no jitter or independent cause-specific KM shortcut",
                    confidence_interval="pointwise asymptotic normal, projected to nonnegative hazard" if hazard else "pointwise asymptotic normal interval intersected with [0,1]; no simultaneous or finite-sample coverage claim",
                    derivatives_zero_sum_error=derivative_error, normal_reference=True,
                    support_policy="no requested horizon beyond last observed follow-up; times=None saves every observed event/censor time, no thinning",
                    survival_at_times=survival if not hazard else None,
                    sources=REFERENCES, optimization="none; exact tied risk-set recursion")
    notes = ["Independent noninformative right censoring and iid subjects are assumptions; competing causes need not be independent.",
             "Zero plug-in variance has no normal CI; an empirical boundary does not establish certainty or exact finite-sample confidence coverage.",
             "No delayed entry, repeated events, covariates, Gray test, Fine-Gray model or sampling weights."]
    if hazard:
        notes.append("This estimates cumulative cause-specific hazard, not instantaneous hazard, cumulative incidence or a subdistribution hazard.")
    frames = dict(curve=rows, covariance=table(covariance.tolist(), columns=labels, index=labels),
                  components=components, risksets=risksets)
    return output(method, frames, settings, notes)


@procedure
def cumulative_incidence(time, event, *, causes=None, times=None, level=.95, device="cpu", weights=None):
    """Aalen--Johansen CIF with full iid IJ covariance for exact tied risk sets."""
    return _curves("cumulative_incidence", time, event, causes=causes, times=times,
                   level=level, device=device, weights=weights, hazard=False)


@procedure
def cause_specific_hazard(time, event, *, causes=None, times=None, level=.95, device="cpu", weights=None):
    """Cause-specific cumulative Nelson--Aalen hazard with full iid IJ covariance."""
    return _curves("cause_specific_hazard", time, event, causes=causes, times=times,
                   level=level, device=device, weights=weights, hazard=True)


def _groups(group, n):
    supplied = positional_list(group, "group")
    if len(supplied) != n:
        raise AnalysisError("invalid_input", "group must match the complete time/event vectors.")
    canonical, keys = [], []
    for value in supplied:
        if isinstance(value, str):
            native, key = value, (2, value)
        elif isinstance(value, bool):
            native, key = value, (0, int(value))
        elif isinstance(value, Integral):
            # Group identifiers need exact integer equality; converting to
            # float would silently merge adjacent labels above 2**53.
            native, key = int(value), (1, int(value))
        elif isinstance(value, Real) and math.isfinite(float(value)):
            native = float(value)
            key = (1, native)
        else:
            raise AnalysisError("invalid_group", "Groups require complete finite numeric, boolean or string labels.")
        canonical.append(native)
        keys.append(key)
    unique = sorted(set(keys))
    if len(unique) != 2:
        raise AnalysisError("invalid_group", "Exactly two independent groups are required.")
    labels = [next(canonical[i] for i, key in enumerate(keys) if key == wanted) for wanted in unique]
    masks = [torch.tensor([key == wanted for key in keys], dtype=torch.bool, device="cpu") for wanted in unique]
    if min(int(mask.sum()) for mask in masks) < 2:
        raise AnalysisError("insufficient_observations", "Each group needs at least two independent observations.")
    return canonical, labels, masks


@procedure
def cif_compare(time, event, group, *, cause, times, level=.95, device="cpu", weights=None):
    """Prespecified-time CIF group-2 minus group-1 normal contrasts and full-rank Wald."""
    data = event_data(time, event, device=device, weights=weights)
    selected = _causes([cause], data.event)
    canonical, labels, masks = _groups(group, data.n)
    grid = prediction_times(times)
    if grid is None:
        raise AnalysisError("invalid_input", "cif_compare requires an explicit prespecified times vector.")
    if float(grid[-1]) > min(float(data.t[mask].max()) for mask in masks):
        raise AnalysisError("outside_support", "Comparison horizons must lie within both groups' observed follow-up.")
    budget = workspace(data.n, len(grid))
    level, critical = confidence(level)
    estimates, covariances = [], []
    group_risksets = []
    for label, mask in zip(labels, masks):
        value, covariance, risksets, _, _ = _recursion(data.t[mask], data.event[mask], selected, grid)
        estimates.append(value)
        covariances.append(covariance)
        group_risksets.extend(dict(group=label, **row) for row in risksets)
    difference = estimates[1]-estimates[0]
    covariance = covariances[0]+covariances[1]
    se = covariance.diagonal().clamp_min(0).sqrt()
    rows, component_labels = [], []
    for j, horizon in enumerate(grid.tolist()):
        scale, delta = float(se[j]), float(difference[j])
        statistic = delta/scale if scale > 0 else None
        rows.append(dict(time=horizon, cause=selected[0], group_1=labels[0], group_2=labels[1],
                         cif_1=float(estimates[0][j]), cif_2=float(estimates[1][j]),
                         difference=delta, standard_error=scale, z=statistic,
                         p_value=2*normal_sf(abs(statistic)) if statistic is not None else None,
                         ci_lower=delta-critical*scale if scale > 0 else None,
                         ci_upper=delta+critical*scale if scale > 0 else None,
                         inference_status="pointwise asymptotic normal" if scale > 0 else "unavailable: zero plug-in variance"))
        component_labels.append(f"{j}:time={horizon:g},cause={selected[0]}")
    positive = se > 0
    correlation = covariance[positive][:, positive]/(se[positive][:, None]*se[positive][None, :])
    try:
        eigenvalues = torch.linalg.eigvalsh(correlation) if bool(positive.any()) else torch.empty(0, dtype=torch.float64, device="cpu")
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("numerical_failure", "Cannot identify the joint CIF covariance rank.") from error
    rank = int((eigenvalues > float(eigenvalues[-1])*1e-10).sum()) if len(eigenvalues) else 0
    identified = rank == len(grid)
    joint = dict(identified=identified, numerical_rank=rank, n_restrictions=len(grid),
                 rank_tolerance="eigenvalue > 1e-10 * maximum eigenvalue of nonzero-variance correlation matrix",
                 df=len(grid) if identified else None, statistic=None, p_value=None,
                 status="asymptotic chi-square Wald" if identified else "unavailable: singular joint covariance; no generalized-inverse test")
    if identified:
        standardized = difference/se
        try:
            statistic = float(standardized@torch.linalg.solve(correlation, standardized))
        except torch.linalg.LinAlgError as error:
            raise AnalysisError("numerical_failure", "Cannot solve the identified joint CIF covariance.") from error
        if not math.isfinite(statistic) or statistic < -1e-10:
            raise AnalysisError("numerical_failure", "The identified joint CIF contrast Wald statistic is invalid.")
        joint.update(statistic=max(0., statistic), p_value=chi2_sf(max(0., statistic), len(grid)))
    settings = dict(data.settings, **budget, group=canonical, group_labels=labels,
                    group_sizes=[int(mask.sum()) for mask in masks], cause=selected[0], times=grid.tolist(), level=level,
                    contrast="group_2 minus group_1; deterministic label ordering boolean/numeric/string, numeric order within family",
                    covariance="independent group AJ IJ covariance sum; full cross-time contrasts",
                    inference="prespecified-time normal individual contrasts; joint chi-square Wald only when complete covariance has full numerical rank",
                    sources=REFERENCES, component_order=component_labels,
                    sample="independent iid subjects within each of two independent fixed groups; independent noninformative censoring within group",
                    no_claim="Gray equality-of-CIF test, Fine-Gray model, simultaneous confidence bands or exact finite-sample inference")
    return output("cif_compare", dict(contrast=rows, covariance=table(covariance.tolist(), columns=component_labels, index=component_labels),
                                      joint=[joint], risksets=group_risksets), settings,
                  ["Group labels determine contrast direction; no automatic category recoding or merging.",
                   "A zero-variance component has no normal test or CI. A singular joint covariance has no joint Wald test.",
                   "Fixed-time contrasts are not Gray's omnibus competing-risk test; censoring must be independent within each group."])
