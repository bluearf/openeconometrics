"""Randomized two-arm restricted mean survival time with integrated Greenwood variance.

The fixed horizon is caller-declared, common to the two arms, and admitted only
within each arm's observed follow-up. Kaplan-Meier risk sets process failures
before censoring at the same time. No proportional-hazards model is fitted.

Reference: Royston and Parmar (2013), doi:10.1186/1471-2288-13-152.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate.common import check_number
from openecon.engines.inference import critical_value
from openecon.resources import plan_workspace

from . import common as m

TERMS = ("rmst_control", "rmst_treated", "difference")
RISK_COLUMNS = (
    "time",
    "n_risk",
    "n_event",
    "n_censored",
    "survival_before",
    "survival",
    "tail_area_to_tau",
    "variance_contribution",
)
EFFECT_COLUMNS = (
    "term",
    "estimate",
    "std_error",
    "z",
    "p_value",
    "ci_low",
    "ci_high",
    "df",
    "degenerate_variance",
)
ARM_COLUMNS = (
    "arm",
    "n",
    "n_event",
    "n_censored",
    "max_followup",
    "tau",
    "rmst",
    "variance",
)
ASSUMPTIONS = (
    "Independent subjects and independently randomized treatment assignment; "
    "consistency/SUTVA; right censoring independent of potential event time "
    "within each arm; a prespecified common horizon with follow-up support "
    "in both arms. These identification assumptions are declared, not verified "
    "by the observed sample."
)


def _arm(times, events, tau):
    """One sort and linear risk/area scans; no quadratic survival covariance."""
    order = torch.argsort(times, stable=True)
    unique, inverse, counts = torch.unique_consecutive(
        times[order], return_inverse=True, return_counts=True
    )
    failures = torch.zeros(len(unique), dtype=torch.float64, device="cpu")
    failures.scatter_add_(0, inverse, events[order])
    counts_float = counts.to(torch.float64)
    censored = counts_float - failures
    risk = len(times) - torch.cumsum(counts_float, 0) + counts_float
    survival = torch.cumprod(1 - failures / risk, 0)
    before = torch.cat((torch.ones(1, dtype=torch.float64, device="cpu"), survival[:-1]))
    # Clipping interval endpoints at tau includes the event at tau in the
    # saved curve, but that event has no remaining area or variance contribution.
    ends = torch.cat((unique[1:], unique[-1:])).clamp(max=tau)
    widths = ends - unique.clamp(max=tau)
    areas = survival * widths
    tail = torch.flip(torch.cumsum(torch.flip(areas, (0,)), 0), (0,))
    estimate = unique[0].clamp(max=tau) + areas.sum()
    alive = risk - failures
    increment = torch.zeros_like(risk)
    ordinary = alive > 0
    increment[ordinary] = failures[ordinary] / (risk[ordinary] * alive[ordinary])
    # When all remaining subjects fail, S and its remaining area are exactly
    # zero. Its limiting integrated contribution is zero, not 0/0 or epsilon.
    contribution = (torch.sqrt(increment) * tail).square()
    variance = contribution.sum()
    if not bool(torch.isfinite(torch.stack((estimate, variance))).all()):
        raise AnalysisError(
            "numerical_failure",
            "RMST or its integrated variance exceeds finite float64 arithmetic.",
        )
    if bool(((tail > 0) & (increment > 0) & (contribution == 0)).any()):
        raise AnalysisError(
            "numerical_failure",
            "A positive RMST variance contribution underflowed float64 arithmetic.",
        )
    curve = torch.stack((unique, risk, failures, censored, before, survival, tail, contribution), 1)
    return float(estimate), float(variance), curve.tolist()


@m.procedure
def treatment_rmst(
    data,
    time,
    event,
    treatment,
    *,
    design,
    tau,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Compare fixed-horizon RMST in independently randomized binary arms.

    ``design='randomized'`` and positive finite scalar ``tau`` are required. Each
    arm needs at least three complete observations and maximum follow-up >= tau.
    All risk-set rows, the full covariance of control/treated/difference, sample
    positions and assumptions are retained in a checksummed causal-design result.
    Inference is asymptotic normal using integrated Greenwood plug-in variance;
    zero variance retains point intervals and reports undefined z/p explicitly.
    """
    m.options(device, weights, max_work, level)
    if not isinstance(design, str) or design != "randomized":
        raise AnalysisError(
            "unsupported_design",
            "treatment_rmst requires design='randomized' and independent randomized arms.",
        )
    tau = check_number(tau, "tau", minimum=0, exclusive=True)
    names = [time, event, treatment]
    selected, metadata = m.sample(
        data, names, numeric=names, missing=missing, max_work=max_work, cost=82
    )
    n = len(selected)
    m.work(
        n * (math.ceil(math.log2(max(n, 2))) + 64),
        max_work,
        "two-arm KM sorting, complete risk tables and integrated inference",
    )
    plan = plan_workspace(
        "randomized RMST computation and complete output",
        {
            "selected_sample_and_provenance": 544 * n,
            "sort_risk_and_area_tensors": 192 * n,
            "complete_risk_tables_and_saved_state": 1024 * n,
            "joint_inference": 4096,
        },
    )
    metadata["rmst_resource_plan"] = plan.record()
    times, events = m.tensor(selected, time), m.tensor(selected, event)
    treatment_values = m.binary(selected, treatment)
    if bool((times < 0).any()):
        raise AnalysisError(
            "invalid_time", "Survival time must be finite and nonnegative, including time zero."
        )
    if not bool(((events == 0) | (events == 1)).all()):
        raise AnalysisError(
            "invalid_event", "event must contain numeric 0 (right censoring) or 1 (failure)."
        )
    arms = []
    for arm in (0, 1):
        members = treatment_values == arm
        count = int(members.sum())
        if count < 3:
            raise AnalysisError(
                "insufficient_observations",
                "Each randomized arm requires at least three complete observations.",
            )
        max_followup = float(times[members].max())
        if tau > max_followup:
            raise AnalysisError(
                "unsupported_horizon",
                "tau must not exceed the maximum observed follow-up in either arm; extrapolation is unavailable.",
            )
        arms.append((members, count, max_followup))
    estimates, variances, risks, statistics = [], [], [], []
    for arm, (members, count, max_followup) in enumerate(arms):
        estimate, variance, rows = _arm(times[members], events[members], tau)
        estimates.append(estimate)
        variances.append(variance)
        risks.append(m.frame(rows, columns=RISK_COLUMNS))
        statistics.append(
            dict(
                arm=arm,
                n=count,
                n_event=int(events[members].sum()),
                n_censored=count - int(events[members].sum()),
                max_followup=max_followup,
                tau=tau,
                rmst=estimate,
                variance=variance,
            )
        )
    values = [*estimates, estimates[1] - estimates[0]]
    v0, v1 = variances
    covariance = [[v0, 0.0, -v0], [0.0, v1, v1], [-v0, v1, v0 + v1]]
    with torch.device("cpu"):
        zcrit = critical_value(1 - float(level))
    effects = []
    for term, estimate, variance in zip(TERMS, values, (v0, v1, v0 + v1), strict=True):
        se = math.sqrt(variance)
        if se == 0:
            z, p = None, None
        else:
            z = estimate / se
            if not math.isfinite(z):
                raise AnalysisError(
                    "numerical_failure",
                    "The RMST normal statistic exceeds finite float64 arithmetic.",
                )
            p = math.erfc(abs(z) / math.sqrt(2))
        low, high = estimate - zcrit * se, estimate + zcrit * se
        if not all(math.isfinite(value) for value in (estimate, variance, se, low, high)):
            raise AnalysisError(
                "numerical_failure", "RMST inference exceeds finite float64 arithmetic."
            )
        effects.append(
            dict(
                term=term,
                estimate=estimate,
                std_error=se,
                z=z,
                p_value=p,
                ci_low=low,
                ci_high=high,
                df=None,
                degenerate_variance=se == 0,
            )
        )
    settings = dict(
        time=time,
        event=event,
        treatment=treatment,
        design=design,
        tau=tau,
        level=float(level),
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )
    state = dict(
        target="superpopulation E[min(T(1),tau)]-E[min(T(0),tau)]",
        assumptions=ASSUMPTIONS,
        arm_statistics=statistics,
        covariance_terms=list(TERMS),
        covariance_matrix=covariance,
        risk_table_names=["risk_control", "risk_treated"],
        variance_method="Integrated Greenwood Kaplan-Meier plug-in; no finite-sample n/(n-1) multiplier",
        inference="asymptotic normal; df=None; zero plug-in variance gives a point interval and undefined z/p",
        time_ties="risk immediately before failures; tied right censoring is removed after failures",
        horizon_role="caller-prespecified common scalar; no extrapolation",
        individual_effect_inference=False,
    )
    notes = [
        ASSUMPTIONS,
        "RMST targets a superpopulation average of restricted potential survival; individual counterfactual effects are not identified.",
        "Normal intervals are asymptotic and are not clipped to the parameter domain.",
    ]
    if any(row["degenerate_variance"] for row in effects):
        notes.append(
            "At least one integrated Greenwood variance is zero; its point interval is retained and z/p are undefined."
        )
    return m.result(
        "treatment_rmst",
        {
            "effects": m.frame(effects, columns=EFFECT_COLUMNS),
            "covariance": m.frame(covariance, columns=TERMS, index=TERMS),
            "arms": m.frame(statistics, columns=ARM_COLUMNS),
            "risk_control": risks[0],
            "risk_treated": risks[1],
        },
        metadata,
        settings,
        state,
        notes,
    )
