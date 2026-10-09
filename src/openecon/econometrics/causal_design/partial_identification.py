"""Direct-RR E-values, consistency bounds and monotone-selection Lee bounds.

References are recorded in docs/econometrics/causal-identification-bounds.md.
All returned bounds describe sensitivity or identification, not sampling
confidence intervals. No observational ignorability or assignment mechanism is
inferred from the supplied rows.
"""

from __future__ import annotations

import hashlib
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

from . import common as m


def _roles(values):
    names = [c.check_name(value, role) for role, value in values]
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_roles", "Every supplied identification role must name a different column.")
    return names


def _finite(value, message):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", message)


def _evalue(ratio):
    # Multiplying two square roots avoids the premature overflow of r*(r-1).
    transformed = torch.where(ratio < 1, ratio.reciprocal(), ratio)
    value = transformed + transformed.sqrt() * (transformed - 1).sqrt()
    _finite(transformed, "Inverting a protective risk ratio overflowed float64.")
    _finite(value, "The E-value exceeds finite float64 arithmetic.")
    return transformed, value


@m.procedure
def evalue(data, rr: str, *, lower: str | None = None, upper: str | None = None,
           missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Direct-risk-ratio E-values for one or more supplied summary rows.

    Optional lower/upper confidence-limit columns must both be supplied and
    contain the point RR. Protective estimates are inverted. A confidence
    interval containing the null RR=1 has confidence-limit E-value=1. Odds
    ratios, hazard ratios and log estimates are not automatically converted.
    These deterministic sensitivity summaries do not have estimated covariance.
    """
    m.options(device, weights, max_work)
    if (lower is None) != (upper is None):
        raise AnalysisError("invalid_confidence_limits", "Supply both lower and upper RR limit columns, or neither.")
    names = _roles([("rr", rr)] + ([] if lower is None else [("lower", lower), ("upper", upper)]))
    rows, metadata = m.sample(data, names, numeric=names, missing=missing,
                              max_work=max_work, cost=36, minimum=1)
    n = len(rows)
    m.work(n * 36, max_work, "all direct RR summaries and optional confidence limits")
    plan = plan_workspace("direct RR E-values and complete sensitivity state", {
        "source_positions_and_selected_rows": 544 * n,
        "ratios_transforms_and_evalues": 256 * n,
    })
    if any(pd.api.types.is_bool_dtype(rows[name].dtype) for name in names):
        raise AnalysisError("invalid_risk_ratio", "Direct risk-ratio estimates and limits must be numeric, not Boolean labels.")
    ratio = m.tensor(rows, rr)
    if bool((ratio <= 0).any()):
        raise AnalysisError("invalid_risk_ratio", "Risk ratios must be strictly positive finite numbers.")
    transformed, values = _evalue(ratio)
    ci_low = ci_high = closest = closest_transformed = ci_values = None
    if lower is not None:
        ci_low, ci_high = m.tensor(rows, lower), m.tensor(rows, upper)
        if bool(((ci_low <= 0) | (ci_high <= 0) | (ci_low > ratio) | (ci_high < ratio)).any()):
            raise AnalysisError("invalid_confidence_limits", "Positive RR confidence limits must satisfy lower <= RR <= upper.")
        null_included = (ci_low <= 1) & (ci_high >= 1)
        closest = torch.where(null_included, torch.ones_like(ratio), torch.where(ratio < 1, ci_high, ci_low))
        closest_transformed, ci_values = _evalue(closest)
    summaries = []
    for i, position in enumerate(metadata["positions"]):
        direction = "protective_inverse" if ratio[i] < 1 else "null" if ratio[i] == 1 else "increased_risk"
        summaries.append([
            position, float(ratio[i]), None if ci_low is None else float(ci_low[i]),
            None if ci_high is None else float(ci_high[i]), direction, float(transformed[i]),
            float(values[i]), None if closest is None else float(closest[i]),
            None if closest_transformed is None else float(closest_transformed[i]),
            None if ci_values is None else float(ci_values[i]),
        ])
    state = dict(
        target="unmeasured-confounding sensitivity of supplied direct risk ratios toward RR=1",
        risk_ratios=ratio.tolist(), risk_ratios_away_from_null=transformed.tolist(),
        evalues=values.tolist(), lower_limits=None if ci_low is None else ci_low.tolist(),
        upper_limits=None if ci_high is None else ci_high.tolist(),
        closest_null_limits=None if closest is None else closest.tolist(),
        confidence_limit_evalues=None if ci_values is None else ci_values.tolist(),
        formula="r + sqrt(r)*sqrt(r-1), r=RR if RR>=1 else 1/RR",
        confidence_limit_rule="RR interval includes 1: E=1; otherwise use lower if RR>1, upper if RR<1",
        covariance=None, standard_error=None, confidence_interval=None,
        inference="deterministic transformation of supplied estimates/limits; no new sampling inference",
        assumptions="direct positive risk ratios conditional on measured covariates; the confounding association strengths use the RR scale; this summary does not measure actual confounding, selection bias or measurement error",
        automatic_measure_conversion=False, computation_resource_plan=plan.record(),
    )
    return m.result("evalue", {"evalues": m.frame(summaries, columns=[
        "position", "risk_ratio", "ci_lower", "ci_upper", "direction", "rr_away_from_null",
        "e_value", "ci_limit_closest_to_null", "ci_rr_away_from_null", "e_value_ci",
    ])}, metadata, dict(rr=rr, lower=lower, upper=upper, missing=missing, device=device,
                       weights=None, max_work=max_work), state,
        notes=[state["inference"], "Direct RR inputs only: OR/HR/log-effect conversion is not implemented.", state["assumptions"]])


@m.procedure
def manski_ate(data, y, treatment, *, lower, upper, missing="raise", device="cpu",
               weights=None, max_work=100_000_000):
    """Sharp empirical ATE bounds under consistency and declared outcome support.

    The support bounds apply to both potential outcomes. Observed means are
    joint, arm-probability-weighted moments of the analyzed sample; treatment
    randomization or ignorability is not assumed. These plug-in identification
    bounds do not supply standard errors or population confidence intervals.
    """
    m.options(device, weights, max_work)
    support_low = c.check_number(lower, "lower")
    support_high = c.check_number(upper, "upper")
    if support_low > support_high:
        raise AnalysisError("invalid_support", "Declared lower outcome support must not exceed upper.")
    names = _roles([("y", y), ("treatment", treatment)])
    rows, metadata = m.sample(data, names, numeric=names, missing=missing,
                              max_work=max_work, cost=40)
    n = len(rows)
    m.work(n * 40, max_work, "joint observed moments and all sharp support extremizers")
    plan = plan_workspace("Manski empirical ATE bounds and complete extremizers", {
        "source_positions_and_selected_rows": 544 * n,
        "outcomes_assignment_and_four_potential_extremizers": 96 * n,
    })
    outcome, assignment = m.tensor(rows, y), m.binary(rows, treatment)
    if bool(((outcome < support_low) | (outcome > support_high)).any()):
        raise AnalysisError("support_violation", "All analyzed outcomes must lie inside the caller-declared support.")
    counts = [int((assignment == arm).sum()) for arm in (0, 1)]
    probabilities = [count / n for count in counts]
    observed_joint = [float((outcome * (assignment == arm)).mean()) for arm in (0, 1)]
    means = [float(outcome[assignment == arm].mean()) for arm in (0, 1)]
    for arm in (0, 1):
        values = outcome[assignment == arm]
        one_sign_nonzero = bool((values != 0).any()) and (bool((values >= 0).all()) or bool((values <= 0).all()))
        if one_sign_nonzero and (observed_joint[arm] == 0 or means[arm] == 0):
            raise AnalysisError("numerical_failure", "A nonzero observed outcome mean underflowed float64.")
    products = [probabilities[1-arm] * endpoint for arm in (0, 1)
                for endpoint in (support_low, support_high)]
    for arm in (0, 1):
        for j, endpoint in enumerate((support_low, support_high)):
            if endpoint != 0 and products[2 * arm + j] == 0:
                raise AnalysisError("numerical_failure", "A nonzero support contribution underflowed float64.")
    potential_bounds = [[observed_joint[arm] + products[2 * arm],
                         observed_joint[arm] + products[2 * arm + 1]] for arm in (0, 1)]
    if support_low == support_high:
        # A singleton potential-outcome support identifies ATE=0 algebraically.
        potential_bounds = [[support_low, support_high], [support_low, support_high]]
    bound_low = potential_bounds[1][0] - potential_bounds[0][1]
    bound_high = potential_bounds[1][1] - potential_bounds[0][0]
    bound_width = bound_high - bound_low
    if not all(math.isfinite(value) for row in potential_bounds for value in row) or not all(
        math.isfinite(value) for value in (bound_low, bound_high, bound_width)
    ):
        raise AnalysisError("numerical_failure", "The requested support bounds exceed finite float64 arithmetic.")
    if bound_high < bound_low:
        raise AnalysisError("numerical_failure", "Floating-point cancellation reversed the identification bounds.")
    if support_high > support_low and bound_width == 0:
        raise AnalysisError("numerical_failure", "A positive identification interval vanished in float64 arithmetic.")
    extremizers = {
        f"y{arm}_{kind}": torch.where(assignment == arm, outcome, torch.full_like(outcome, endpoint)).tolist()
        for arm in (0, 1) for kind, endpoint in (("lower", support_low), ("upper", support_high))
    }
    state = dict(
        target="consistency-only empirical ATE of the analyzed rows",
        outcomes=outcome.tolist(), assignment=assignment.tolist(), arm_counts=counts,
        arm_probabilities=probabilities, observed_arm_means=means,
        observed_joint_moments=observed_joint, support=[support_low, support_high],
        potential_mean_bounds=potential_bounds, ate_bounds=[bound_low, bound_high],
        extremizers=extremizers, bound_width=bound_width,
        covariance=None, standard_error=None, confidence_interval=None,
        inference="sharp empirical identification interval; not a sampling confidence interval",
        assumptions="consistency/SUTVA and both potential outcomes inside prespecified common support; no random assignment, ignorability or outcome monotonicity assumed",
        missing_target="analyzed complete rows; dropping missing inputs changes the empirical target population",
        computation_resource_plan=plan.record(),
    )
    return m.result("manski_ate", {
        "bounds": m.frame([["ate", bound_low, bound_high, bound_width]], columns=["target", "lower", "upper", "width"]),
        "potential_means": m.frame([[arm, counts[arm], probabilities[arm], means[arm],
                                      observed_joint[arm], *potential_bounds[arm]] for arm in (0, 1)],
                                    columns=["arm", "n", "probability", "observed_mean", "observed_joint_moment", "mean_lower", "mean_upper"]),
        "extremizers": m.frame([[metadata["positions"][i], int(assignment[i]), float(outcome[i]),
                                  *[extremizers[key][i] for key in ("y0_lower", "y0_upper", "y1_lower", "y1_upper")]]
                                 for i in range(n)], columns=["position", "treatment", "observed_y", "y0_lower", "y0_upper", "y1_lower", "y1_upper"]),
    }, metadata, dict(y=y, treatment=treatment, lower=support_low, upper=support_high,
                      missing=missing, device=device, weights=None, max_work=max_work), state,
        notes=[state["inference"], state["assumptions"], state["missing_target"]])


def _trim(values, numerator, denominator, *, largest):
    """Fractional empirical tail mean; a boundary tie shares retention equally."""
    order = torch.argsort(values, descending=largest, stable=True)
    sorted_values = values[order]
    unique, inverse, counts = torch.unique_consecutive(sorted_values, return_inverse=True, return_counts=True)
    starts = counts.cumsum(0) - counts
    # Rational counts avoid ambiguous cancellation when retained mass is an
    # integer, including equal selection rates and the no-trimming boundary.
    retained_numerator = (numerator - starts * denominator).clamp_min(0)
    capped = torch.minimum(retained_numerator, counts * denominator)
    group_weights = capped.to(torch.float64) / (counts * denominator).to(torch.float64)
    sorted_weights = group_weights[inverse]
    retention = torch.empty_like(values)
    retention[order] = sorted_weights
    mass = numerator / denominator
    mean_weights = retention / mass
    products = mean_weights * values
    if bool(((mean_weights > 0) & (values != 0) & (products == 0)).any()):
        raise AnalysisError("numerical_failure", "A nonzero trimmed-mean contribution underflowed float64.")
    mean = float(products.sum())
    if not math.isfinite(mean):
        raise AnalysisError("numerical_failure", "A trimmed empirical mean exceeds finite float64 arithmetic.")
    ties = [[float(value), int(count), float(weight)] for value, count, weight in zip(unique, counts, group_weights)]
    return mean, retention, mean_weights, ties


@m.procedure
def lee_bounds(data, y, treatment, selection, *, design, monotonicity, missing="raise",
               device="cpu", weights=None, max_work=100_000_000):
    """Always-selected ATE Lee bounds under randomized monotone selection.

    Both design='randomized' and monotonicity='increasing'/'decreasing' must be
    declared explicitly. Increasing means S(1)>=S(0), decreasing S(1)<=S(0).
    All original assignment/selection rows determine the rates. Missing Y is
    allowed only when selection=0. No listwise rate-changing deletion is allowed.
    Empirical rates inconsistent with the declared direction are refused.
    """
    m.options(device, weights, max_work)
    c.check_choice(design, "design", ("randomized",))
    c.check_choice(monotonicity, "monotonicity", ("increasing", "decreasing"))
    if missing != "raise":
        raise AnalysisError("unsupported_missing_policy", "Lee bounds preserve the original selection rates; missing='drop' is unavailable.")
    names = _roles([("y", y), ("treatment", treatment), ("selection", selection)])
    # Outcomes are intentionally outside this full original-topology sample:
    # unselected missing outcomes never delete assignment/selection records.
    rows, metadata = m.sample(data, [treatment, selection], numeric=[treatment, selection],
                              missing="raise", max_work=max_work, cost=48)
    n = len(rows)
    sorting = max(1, math.ceil(math.log2(n)))
    m.work(n * (48 + 4 * sorting), max_work, "complete selection topology and both fractional tail trims")
    plan = plan_workspace("Lee bounds complete selection sample and trimming state", {
        "source_positions_and_selected_rows": 640 * n,
        "sorts_ties_retention_and_mean_weights": 384 * n,
    })
    source = c.source(data)
    c.require_numeric(source, [y])
    assignment = m.binary(rows, treatment)
    if pd.api.types.is_bool_dtype(rows[selection].dtype):
        raise AnalysisError("invalid_selection", "Selection must use numeric 0/1, not Boolean labels.")
    observed = m.tensor(rows, selection)
    if not bool(((observed == 0) | (observed == 1)).all()):
        raise AnalysisError("invalid_selection", "Selection must contain only numeric 0 and 1.")
    selected_positions = torch.nonzero(observed == 1).flatten().tolist()
    selected_rows = source.iloc[selected_positions].loc[:, [y]].copy()
    if selected_rows[y].isna().any():
        raise AnalysisError("missing_selected_outcome", "Every selection=1 row requires a finite observed outcome.")
    selected_outcomes = m.tensor(selected_rows, y)
    counts = [int((assignment == arm).sum()) for arm in (0, 1)]
    selected_counts = [int(((assignment == arm) & (observed == 1)).sum()) for arm in (0, 1)]
    if min(selected_counts) == 0:
        raise AnalysisError("empty_always_selected_target", "Both arms require selected outcomes to define the always-selected mean target.")
    cross_difference = selected_counts[1] * counts[0] - selected_counts[0] * counts[1]
    if (monotonicity == "increasing" and cross_difference < 0) or (
        monotonicity == "decreasing" and cross_difference > 0
    ):
        raise AnalysisError("incompatible_empirical_selection", "Empirical selection rates contradict the declared monotonicity direction; no reversed or clipped bounds are returned.")
    high_arm = 1 if monotonicity == "increasing" else 0
    low_arm = 1 - high_arm
    numerator = selected_counts[low_arm] * counts[high_arm]
    denominator = counts[low_arm]
    retained_mass = numerator / denominator
    rates = [selected_counts[arm] / counts[arm] for arm in (0, 1)]
    selected_assignment = assignment[torch.tensor(selected_positions, dtype=torch.int64, device="cpu")]
    arm_outcomes = [selected_outcomes[selected_assignment == arm] for arm in (0, 1)]
    means = [float(values.mean()) for values in arm_outcomes]
    for values, mean in zip(arm_outcomes, means):
        if mean == 0 and bool((values != 0).any()) and (bool((values >= 0).all()) or bool((values <= 0).all())):
            raise AnalysisError("numerical_failure", "A nonzero selected outcome mean underflowed float64.")
    tail_low, retention_low, normalized_low, tie_low = _trim(arm_outcomes[high_arm], numerator, denominator, largest=False)
    tail_high, retention_high, normalized_high, tie_high = _trim(arm_outcomes[high_arm], numerator, denominator, largest=True)
    if high_arm == 1:
        bounds = [tail_low - means[0], tail_high - means[0]]
        lower_retention, upper_retention = retention_low, retention_high
        lower_normalized, upper_normalized = normalized_low, normalized_high
    else:
        bounds = [means[1] - tail_high, means[1] - tail_low]
        lower_retention, upper_retention = retention_high, retention_low
        lower_normalized, upper_normalized = normalized_high, normalized_low
    if not all(math.isfinite(value) for value in [*bounds, bounds[1] - bounds[0]]) or bounds[0] > bounds[1]:
        raise AnalysisError("numerical_failure", "The fractional Lee bounds are nonfinite or numerically reversed.")
    raw_lower = torch.zeros(n, dtype=torch.float64, device="cpu")
    raw_upper = torch.zeros_like(raw_lower)
    mean_lower, mean_upper = torch.zeros_like(raw_lower), torch.zeros_like(raw_lower)
    high_positions = torch.nonzero((assignment == high_arm) & (observed == 1)).flatten()
    low_positions = torch.nonzero((assignment == low_arm) & (observed == 1)).flatten()
    raw_lower[low_positions] = raw_upper[low_positions] = 1.0
    mean_lower[low_positions] = mean_upper[low_positions] = 1.0 / selected_counts[low_arm]
    raw_lower[high_positions], raw_upper[high_positions] = lower_retention, upper_retention
    mean_lower[high_positions], mean_upper[high_positions] = lower_normalized, upper_normalized
    observed_y = [None] * n
    for i, position in enumerate(selected_positions):
        observed_y[position] = float(selected_outcomes[i])
    # Scientific input identity includes every original design row and each
    # actually observed Y at its original position. Ignored unselected values
    # are neither imputed nor allowed to change the scientific identity.
    outcome_identity = dict(y=y, dtype=str(source[y].dtype), positions=selected_positions,
                            values=selected_outcomes.tolist(), design_sha256=metadata["sample_sha256"])
    full_hash = hashlib.sha256(m.canonical(outcome_identity).encode()).hexdigest()
    metadata.update(columns=names, outcome_observation_sha256=full_hash,
                    n_selected=sum(selected_counts), n_unselected=n-sum(selected_counts),
                    selected_positions=selected_positions,
                    missing_outcome_policy="unselected Y may be missing; selected Y and all assignment/selection rows are required")
    state = dict(
        target="E[Y(1)-Y(0) | S(1)=S(0)=1], always-selected principal stratum",
        assignment=assignment.tolist(), selection=observed.tolist(), observed_y=observed_y,
        arm_counts=counts, selected_counts=selected_counts, selection_rates=rates,
        observed_selected_means=means, always_selected_fraction=min(rates), trimmed_arm=high_arm,
        retention_fraction=retained_mass/selected_counts[high_arm],
        retained_selected_mass=dict(numerator=numerator, denominator=denominator, value=retained_mass),
        trimmed_tail_means=[tail_low, tail_high], ate_bounds=bounds, bound_width=bounds[1]-bounds[0],
        retention_lower_bound=raw_lower.tolist(), retention_upper_bound=raw_upper.tolist(),
        mean_weights_lower_bound=mean_lower.tolist(), mean_weights_upper_bound=mean_upper.tolist(),
        lower_tail_tie_groups=tie_low, upper_tail_tie_groups=tie_high,
        tie_rule="fractional boundary mass distributed equally within each equal-outcome group",
        empirical_direction_check="exact integer comparison of cross-multiplied arm rates",
        unselected_outcomes="not used or imputed; selected outcomes and original topology retained",
        covariance=None, standard_error=None, confidence_interval=None,
        inference="plug-in sharp Lee identification bounds; no sampling confidence interval or population monotonicity test",
        assumptions="random assignment independent of potential outcomes and selection; consistency/SUTVA; declared unit-level selection monotonicity; nonempty always-selected population",
        computation_resource_plan=plan.record(),
    )
    return m.result("lee_bounds", {
        "bounds": m.frame([["always_selected_ate", *bounds, bounds[1]-bounds[0]]], columns=["target", "lower", "upper", "width"]),
        "selection_rates": m.frame([[arm, counts[arm], selected_counts[arm], rates[arm], means[arm]] for arm in (0, 1)],
                                   columns=["arm", "original_n", "selected_n", "selection_rate", "selected_mean"]),
        "trimming_weights": m.frame([[i, int(assignment[i]), int(observed[i]), observed_y[i],
                                       float(raw_lower[i]), float(raw_upper[i]), float(mean_lower[i]), float(mean_upper[i])]
                                      for i in range(n)], columns=["position", "treatment", "selection", "observed_y", "retention_lower_bound", "retention_upper_bound", "mean_weight_lower_bound", "mean_weight_upper_bound"]),
    }, metadata, dict(y=y, treatment=treatment, selection=selection, design=design,
                      monotonicity=monotonicity, missing=missing, device=device, weights=None,
                      max_work=max_work), state, notes=[state["inference"], state["assumptions"],
            "The target is always selected under both assignments, not all randomized subjects.",
            "An incompatible empirical selection-rate direction can arise from sampling noise; refusal is not a test of population monotonicity."])
