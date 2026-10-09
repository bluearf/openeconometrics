"""Predeclared marginal distribution targets in independent randomized arms.

These are differences between potential-outcome marginal distributions. Quantile
differences do not identify quantiles of individual treatment effects. Population
causal interpretation requires a randomized two-arm design, consistency/SUTVA,
independent observations and an appropriate population sample. Quantile bootstrap
intervals additionally require locally continuous distributions with positive
density at the declared probabilities; ties and degenerate empirical distributions
can invalidate that approximation and are reported without artificial variance.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from statistics import NormalDist

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace

from .common import binary, frame, options, procedure, result, sample, tensor, work


_ASSUMPTIONS = (
    "Randomized assignment to numeric treatment arms 0 and 1, consistency/SUTVA, "
    "independent sampled rows, finite outcomes, and fixed targets declared before "
    "inspecting the analysis outcomes. Inference conditions on the observed arm "
    "counts. Randomization and predeclaration are user declarations, not verified "
    "by the estimator. No observational adjustment, paired/clustered design, "
    "survey weights or simultaneous confidence bands are implemented."
)


def _targets(values, name, *, probabilities=False):
    if not isinstance(values, (list, tuple)) or not values:
        raise AnalysisError("invalid_spec", f"{name} must be a nonempty predeclared list.")
    selected = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise AnalysisError("invalid_spec", f"{name} must contain finite numeric targets.")
        number = float(value)
        if probabilities and not 0 < number < 1:
            raise AnalysisError("invalid_spec", "quantiles must be strictly between 0 and 1.")
        selected.append(number)
    if len(set(selected)) != len(selected):
        raise AnalysisError("invalid_spec", f"{name} must contain unique targets.")
    return selected


def _design(design):
    if design != "randomized":
        raise AnalysisError(
            "unsupported_design", "Declare design='randomized' for two independent arms."
        )


def _arms(data, y, treatment, missing, max_work, cost):
    if not isinstance(y, str) or not isinstance(treatment, str) or y == treatment:
        raise AnalysisError("invalid_spec", "y and treatment must be different column names.")
    rows, metadata = sample(
        data,
        [y, treatment],
        numeric=[y, treatment],
        missing=missing,
        max_work=max_work,
        cost=cost,
    )
    assignment = binary(rows, treatment)
    outcome = tensor(rows, y)
    positions = [torch.nonzero(assignment == arm).flatten() for arm in (0, 1)]
    if min(len(pos) for pos in positions) < 3:
        raise AnalysisError(
            "insufficient_sample", "Each randomized arm requires at least three rows."
        )
    return outcome, assignment, positions, metadata


def _finite(value, label):
    if not torch.isfinite(value).all():
        raise AnalysisError("numerical_failure", f"{label} overflowed the finite float64 domain.")


@procedure
def treatment_cdf(
    data,
    y,
    treatment,
    *,
    design,
    thresholds,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Estimate F1(t)-F0(t) at a fixed finite grid with joint sample covariance.

    Covariance is the sum of each arm's unbiased sample covariance of indicator
    vectors divided by its arm count. Pointwise normal intervals are untruncated
    Wald intervals, not simultaneous bands. A zero empirical variance is exposed
    with a point interval and undefined z/p; it is not proof of population certainty.
    """
    options(device, weights, max_work, level=level)
    _design(design)
    grid = _targets(thresholds, "thresholds")
    k = len(grid)
    outcome, assignment, positions, metadata = _arms(
        data,
        y,
        treatment,
        missing,
        max_work,
        cost=1 + k + k * k,
    )
    n = len(outcome)
    work(n * (1 + k + k * k) + 3 * k * k, max_work, "CDF indicators and joint covariance")
    resource_plan = plan_workspace(
        "treatment CDF full fixed grid",
        {
            "source_sample_and_positions": 512 * n,
            "indicators_centered_python_state_and_json": 192 * n * k,
            "arm_joint_covariance_and_complete_table_state": 512 * k * k,
            "thresholds_means_effects_and_inference": 96 * k,
        },
    )
    threshold = torch.tensor(grid, dtype=torch.float64, device="cpu")
    indicators = [(outcome[pos, None] <= threshold[None, :]).to(torch.float64) for pos in positions]
    cdfs = [values.mean(0) for values in indicators]
    arm_covariances = []
    for values, cdf in zip(indicators, cdfs):
        centered = values - cdf
        count = len(values)
        arm_covariances.append(centered.T @ centered / (count * (count - 1)))
    covariance = arm_covariances[0] + arm_covariances[1]
    effect = cdfs[1] - cdfs[0]
    se = covariance.diag().clamp_min(0).sqrt()
    critical = -NormalDist().inv_cdf((1 - float(level)) / 2)
    table_rows = []
    degeneracy = []
    for j, threshold_value in enumerate(grid):
        estimate, error = float(effect[j]), float(se[j])
        is_zero = error == 0.0
        z = None if is_zero else estimate / error
        p = None if is_zero else math.erfc(abs(z) / math.sqrt(2))
        status = "zero_empirical_variance" if is_zero else "pointwise_normal"
        if is_zero:
            degeneracy.append(j)
        table_rows.append(
            [
                threshold_value,
                float(cdfs[0][j]),
                float(cdfs[1][j]),
                estimate,
                error,
                z,
                p,
                estimate - critical * error,
                estimate + critical * error,
                status,
            ]
        )
    labels = [str(value) for value in grid]
    tables = {
        "effects": frame(
            table_rows,
            columns=[
                "threshold",
                "cdf0",
                "cdf1",
                "estimate",
                "std_error",
                "z",
                "p_value",
                "ci_low",
                "ci_high",
                "inference_status",
            ],
        ),
        "covariance": frame(covariance.tolist(), columns=labels, index=labels),
    }
    notes = [_ASSUMPTIONS]
    if degeneracy:
        notes.append(
            "Zero empirical variance at grid indices "
            + str(degeneracy)
            + ": point intervals are uninformative about unsampled tail probability; z and p are undefined."
        )
    state = {
        "outcomes": outcome.tolist(),
        "assignment": assignment.tolist(),
        "arm_sample_positions": [pos.tolist() for pos in positions],
        "arm_counts": [len(pos) for pos in positions],
        "thresholds": grid,
        "cdf0": cdfs[0].tolist(),
        "cdf1": cdfs[1].tolist(),
        "indicator0": indicators[0].tolist(),
        "indicator1": indicators[1].tolist(),
        "arm_covariances": [value.tolist() for value in arm_covariances],
        "covariance": covariance.tolist(),
        "effect": effect.tolist(),
        "std_error": se.tolist(),
        "degenerate_grid_indices": degeneracy,
        "target": "marginal_cdf1_minus_cdf0",
        "conditioning": "observed arm counts",
        "covariance_method": "unbiased arm indicator sample covariance divided by arm count",
        "inference": "pointwise_normal",
        "critical_value": critical,
        "assumptions": _ASSUMPTIONS,
        "computation_resource_plan": resource_plan.record(),
    }
    settings = {
        "y": y,
        "treatment": treatment,
        "design": design,
        "thresholds": grid,
        "level": float(level),
        "missing": missing,
        "device": "cpu",
        "weights": None,
        "max_work": max_work,
    }
    return result("treatment_cdf", tables, metadata, settings, state, notes=notes)


def _inverse(sorted_values, probabilities):
    count = sorted_values.shape[-1]
    ranks = torch.tensor(
        [math.ceil(count * q) - 1 for q in probabilities], dtype=torch.int64, device="cpu"
    )
    return sorted_values[..., ranks]


@procedure
def treatment_quantile(
    data,
    y,
    treatment,
    *,
    design,
    quantiles,
    reps=999,
    seed=1729,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Marginal Q1(q)-Q0(q) using empirical inverse CDF and arm bootstrap.

    The empirical quantile is sorted_y[ceil(n*q)-1]. Each arm is independently
    resampled at its observed count using a private CPU Torch generator. Saves
    every replicate quantile, effect and their joint covariance. Percentile
    intervals use linear interpolation across replicate effects. P-values are
    not defined for this bootstrap interval procedure.
    """
    options(device, weights, max_work, level=level)
    _design(design)
    probabilities = _targets(quantiles, "quantiles", probabilities=True)
    if isinstance(reps, bool) or not isinstance(reps, Integral) or reps < 2:
        raise AnalysisError("invalid_spec", "reps must be an integer of at least two.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise AnalysisError("invalid_spec", "seed must be an integer in [0, 2**63).")
    reps, seed = int(reps), int(seed)
    k = len(probabilities)
    outcome, assignment, positions, metadata = _arms(
        data,
        y,
        treatment,
        missing,
        max_work,
        cost=1 + reps,
    )
    n = len(outcome)
    # Conservatively account for full draw/value/sort matrices, source rows,
    # all retained bootstrap targets, covariance, and sorting comparisons.
    sorting = sum(len(pos) * max(1, math.ceil(math.log2(len(pos)))) for pos in positions)
    work(
        n + reps * (3 * n + sorting + 3 * k + 9 * k * k) + 9 * k * k,
        max_work,
        "Quantile source sample, independent arm draws and joint covariance",
    )
    resource_plan = plan_workspace(
        "treatment quantile full bootstrap",
        {
            "source_sample_and_positions": 512 * n,
            "full_arm_draw_indices_values_and_sorted_values": 64 * reps * n,
            "retained_replicates_joint_python_state_and_json": 640 * reps * k,
            "joint_covariance_complete_table_and_state": 768 * k * k,
            "quantiles_and_intervals": 96 * k,
        },
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    arm_samples = [outcome[pos] for pos in positions]
    sorted_arms = [values.sort().values for values in arm_samples]
    quantile_values = [_inverse(values, probabilities) for values in sorted_arms]
    arm_replicates = []
    for values in arm_samples:
        draw = torch.randint(len(values), (reps, len(values)), generator=generator, device="cpu")
        sorted_draws = values[draw].sort(dim=1).values
        arm_replicates.append(_inverse(sorted_draws, probabilities))
    effect = quantile_values[1] - quantile_values[0]
    replicate_effect = arm_replicates[1] - arm_replicates[0]
    full_replicates = torch.cat([arm_replicates[0], arm_replicates[1], replicate_effect], dim=1)
    center = full_replicates - full_replicates.mean(0)
    joint_covariance = center.T @ center / (reps - 1)
    if bool(((joint_covariance.diag() == 0) & (center != 0).any(0)).any()):
        raise AnalysisError(
            "numerical_failure",
            "A nonconstant bootstrap target variance underflowed float64; zero empirical variance is not asserted.",
        )
    covariance = joint_covariance[2 * k :, 2 * k :]
    se = covariance.diag().clamp_min(0).sqrt()
    alpha = (1 - float(level)) / 2
    intervals = torch.quantile(
        replicate_effect, torch.tensor([alpha, 1 - alpha], dtype=torch.float64, device="cpu"), dim=0
    )
    for value, label in [
        (effect, "Quantile contrast"),
        (joint_covariance, "Bootstrap covariance"),
        (intervals, "Bootstrap percentile interval"),
    ]:
        _finite(value, label)
    table_rows = [
        [
            q,
            float(quantile_values[0][j]),
            float(quantile_values[1][j]),
            float(effect[j]),
            float(se[j]),
            None,
            float(intervals[0, j]),
            float(intervals[1, j]),
            "percentile_bootstrap",
        ]
        for j, q in enumerate(probabilities)
    ]
    labels = [str(value) for value in probabilities]
    joint_labels = [f"{arm}:{label}" for arm in ("q0", "q1", "effect") for label in labels]
    tables = {
        "effects": frame(
            table_rows,
            columns=[
                "quantile",
                "q0",
                "q1",
                "estimate",
                "std_error",
                "p_value",
                "ci_low",
                "ci_high",
                "inference_status",
            ],
        ),
        "covariance": frame(covariance.tolist(), columns=labels, index=labels),
        "joint_covariance": frame(
            joint_covariance.tolist(), columns=joint_labels, index=joint_labels
        ),
    }
    ties = [len(torch.unique(values)) < len(values) for values in arm_samples]
    notes = [
        _ASSUMPTIONS,
        "Quantile effects compare marginal potential-outcome quantiles, not individual-effect quantiles.",
        "Percentile bootstrap coverage requires continuity and positive density near each population quantile; "
        "finite-sample validity is not guaranteed. No bootstrap p-values or simultaneous bands are claimed.",
    ]
    if any(ties):
        notes.append(
            "Tied outcomes occur in the analysis sample; bootstrap quantile regularity is not verified."
        )
    if (se == 0).any():
        notes.append(
            "Some bootstrap contrasts have zero empirical variance; no artificial positive variance is added."
        )
    state = {
        "outcomes": outcome.tolist(),
        "assignment": assignment.tolist(),
        "arm_sample_positions": [pos.tolist() for pos in positions],
        "arm_counts": [len(pos) for pos in positions],
        "sorted_arm_outcomes": [values.tolist() for values in sorted_arms],
        "quantiles": probabilities,
        "q0": quantile_values[0].tolist(),
        "q1": quantile_values[1].tolist(),
        "effect": effect.tolist(),
        "std_error": se.tolist(),
        "bootstrap_q0": arm_replicates[0].tolist(),
        "bootstrap_q1": arm_replicates[1].tolist(),
        "bootstrap_effects": replicate_effect.tolist(),
        "covariance": covariance.tolist(),
        "joint_covariance": joint_covariance.tolist(),
        "joint_target_order": joint_labels,
        "ci_low": intervals[0].tolist(),
        "ci_high": intervals[1].tolist(),
        "target": "marginal_q1_minus_q0",
        "quantile_method": "empirical inverse CDF: ceil(n*q)-1",
        "bootstrap_method": "independent arm resampling conditional on observed counts",
        "bootstrap_draw_order": "arm0 full reps-by-n0 indices, then arm1 full reps-by-n1 indices",
        "interval_method": "pointwise percentile; linear interpolation of bootstrap effects",
        "rng": {
            "engine": "torch.Generator(cpu)",
            "seed": seed,
            "torch_version": str(torch.__version__),
        },
        "reps": reps,
        "sample_contains_ties": ties,
        "p_value_applicability": "not applicable",
        "computation_resource_plan": resource_plan.record(),
        "assumptions": _ASSUMPTIONS
        + " Quantile inference additionally requires continuous distributions "
        "with positive density near the declared population quantiles.",
    }
    settings = {
        "y": y,
        "treatment": treatment,
        "design": design,
        "quantiles": probabilities,
        "reps": reps,
        "seed": seed,
        "level": float(level),
        "missing": missing,
        "device": "cpu",
        "weights": None,
        "max_work": max_work,
    }
    return result("treatment_quantile", tables, metadata, settings, state, notes=notes)
