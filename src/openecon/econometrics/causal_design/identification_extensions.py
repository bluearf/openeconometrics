"""IID population Manski regions and discrete conditional Lee bounds.

Primary mathematics and the exact supported contracts are documented in
docs/econometrics/causal-population-bounds.md. Sampling uncertainty and
identification uncertainty remain different quantities throughout.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

from . import common as m

_CAPACITY = 64


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{label} exceeds finite float64 arithmetic.")
    return value


def _sum(values):
    """Bounded native FastTwoSum expansions preserve nested cancellation."""
    partials = []
    for x in values:
        keep = 0
        for y in partials:
            larger = x.abs() >= y.abs()
            a, b = torch.where(larger, x, y), torch.where(larger, y, x)
            high = _finite(a + b, "floating expansion")
            low = b - (high - a)
            if bool((low != 0).any()):
                partials[keep] = low
                keep += 1
            x = high
        partials[keep:] = [x]
        if len(partials) > _CAPACITY:
            raise AnalysisError(
                "numerical_failure", "The declared 64-part summation capacity is exceeded."
            )
    total = torch.zeros(values.shape[1:], dtype=torch.float64, device="cpu")
    for value in reversed(partials):
        total = total + value
    return _finite(total, "expanded sum")


def _divide(value, denominator, label):
    out = _finite(value / denominator, label)
    if bool(((value != 0) & (out == 0)).any()):
        raise AnalysisError("numerical_failure", f"A nonzero {label} underflows float64.")
    return out


def _multiply(left, right, label):
    out = _finite(left * right, label)
    if bool(((left != 0) & (right != 0) & (out == 0)).any()):
        raise AnalysisError("numerical_failure", f"A nonzero {label} underflows float64.")
    return out


def _mean(values):
    return _divide(_sum(values), len(values), "mean")


def _roles(*names):
    names = [c.check_name(name, "input role") for name in names]
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_roles", "Every input role must name a distinct column.")
    return names


def _preflight(data, names, max_work, *, cells=0, label_bytes=0):
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset",
            "These bounds require a resident table; Dataset replay is unsupported.",
        )
    source = c.source(data)
    n = len(source)
    if n > 10000:
        raise AnalysisError(
            "resource_limit", "Population-bound inputs are limited to 10000 original rows."
        )
    cost = (
        512 + (256 if cells else 64) * _CAPACITY + 32 * cells + 8 * max(1, math.ceil(math.log2(n)))
    )
    cost += 16 * label_bytes
    work = n * cost + 2048 * max(1, cells)
    m.work(
        work, max_work, "complete samples, covariance, strata, trims, bounded sums and artifacts"
    )
    # common.sample repeats these labels in metadata/scientific state. Charge
    # their actual escaped serialization before that sample is constructed.
    index_bytes = 0
    for value in source.index:
        saved = m._label(value)
        if isinstance(saved, str) and len(saved) > 8192:
            raise AnalysisError(
                "resource_limit", "Each saved original index label is limited to 8192 JSON bytes."
            )
        encoded = m.canonical(saved).encode()
        if len(encoded) > 8192:
            raise AnalysisError(
                "resource_limit", "Each saved original index label is limited to 8192 JSON bytes."
            )
        index_bytes += len(encoded)
    work += 16 * index_bytes
    m.work(work, max_work, "complete computation and escaped original index identities")
    plan = plan_workspace(
        "complete population identification extensions",
        {
            "original_selected_columns_typed_labels_and_source": 1024 * n * len(names),
            "scores_extremizers_trim_weights_and_full_json": 4096 * n,
            "full_cells_tables_covariance_and_json": 16384 * max(1, cells),
            "bounded_sum_expansions_and_live_vectors": 128 * _CAPACITY,
            "midpoint_expansion_and_subtraction_vectors": 256 * n,
            "escaped_typed_strata_source_state_hashes_and_JSON_copies": 16
            * (n + cells)
            * label_bytes,
            "escaped_original_index_identity_and_JSON_copies": 16 * index_bytes,
        },
    )
    absent = [name for name in names if name not in source]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {absent}.")
    return source, work, cost, plan.record()


def _complete(missing):
    if missing != "raise":
        raise AnalysisError(
            "unsupported_missing",
            "The declared population/topology requires missing='raise'; row deletion is unsupported.",
        )


def _two_sum(left, right):
    high = _finite(left + right, "midpoint expansion")
    removed = high - left
    low = (left - (high - removed)) + (right - removed)
    return high, low


def _center_scores(outcome, assignment, a, b):
    if a == b:
        return torch.zeros_like(outcome), [a, 0.0]
    left = _divide(
        torch.tensor(a, dtype=torch.float64, device="cpu"), 2, "lower midpoint component"
    )
    right = _divide(
        torch.tensor(b, dtype=torch.float64, device="cpu"), 2, "upper midpoint component"
    )
    high, low = _two_sum(left, right)
    difference, remainder = _two_sum(outcome, -high)
    # A rounded midpoint alone can erase a small difference. Preserve its
    # low component and the subtraction remainder until the final rounding.
    centered = _sum(torch.stack((difference, remainder, -low.expand_as(outcome))))
    return (2 * assignment - 1) * centered, [float(high), float(low)]


@m.procedure
def manski_ate_inference(
    data,
    y,
    treatment,
    *,
    lower,
    upper,
    sampling,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """IID population endpoint covariance and an entire-identified-set region.

    Common potential-outcome support is fixed by the caller. The confidence
    region uses two Bonferroni one-sided asymptotic normal endpoint limits;
    it is not a finite-sample guarantee or an optimized point-parameter CI.
    """
    m.options(device, weights, max_work, level)
    c.check_choice(sampling, "sampling", ("iid",))
    _complete(missing)
    a, b = c.check_number(lower, "lower"), c.check_number(upper, "upper")
    if a > b or max(abs(a), abs(b)) > 1e150:
        raise AnalysisError(
            "invalid_support", "Common support must satisfy lower<=upper and abs(endpoints)<=1e150."
        )
    names = _roles(y, treatment)
    source, work, cost, plan = _preflight(data, names, max_work)
    rows, metadata = m.sample(
        source, names, numeric=names, missing="raise", max_work=max_work, cost=cost, minimum=3
    )
    if pd.api.types.is_bool_dtype(rows[y].dtype):
        raise AnalysisError("invalid_outcome", "Outcome must be numeric, not Boolean.")
    outcome, d = m.tensor(rows, y), m.binary(rows, treatment)
    if bool(((outcome < a) | (outcome > b)).any()):
        raise AnalysisError(
            "support_violation",
            "Observed outcomes must lie in the prespecified common potential-outcome support.",
        )
    n, width = len(rows), b - a
    half = width / 2
    if width > 0 and half == 0:
        raise AnalysisError(
            "numerical_failure",
            "The positive support width cannot be represented at the required scale.",
        )
    # Both endpoint scores equal this center score plus a fixed offset. This
    # avoids cancelling a large common outcome location and preserves the
    # exact rank-one endpoint covariance instead of estimating it twice.
    center_score, midpoint = _center_scores(outcome, d, a, b)
    _finite(center_score, "center endpoint score")
    center = _mean(center_score)
    centered = center_score - center
    if bool((centered != 0).any()):
        square = _multiply(centered, centered, "endpoint score square")
        variance = _divide(_sum(square), n * (n - 1), "endpoint estimator variance")
        if variance <= 0:
            raise AnalysisError(
                "numerical_failure", "Nonconstant endpoint uncertainty vanished in float64."
            )
    else:
        variance = torch.zeros((), dtype=torch.float64, device="cpu")
    se = variance.sqrt()
    critical = math.sqrt(2) * torch.erfinv(torch.tensor(level, dtype=torch.float64, device="cpu"))
    _finite(critical, "normal critical value")
    bounds = torch.stack((center - half, center + half))
    radius = critical * se
    region = torch.stack((bounds[0] - radius, bounds[1] + radius))
    _finite(torch.cat((bounds, region, radius.reshape(1))), "population bounds and region")
    if width > 0 and bounds[0] == bounds[1]:
        raise AnalysisError(
            "numerical_failure", "A positive identification interval vanished in float64."
        )
    if se > 0 and (radius <= 0 or region[0] == bounds[0] or region[1] == bounds[1]):
        raise AnalysisError(
            "numerical_failure", "Nonzero confidence expansion is not representable in float64."
        )
    covariance = torch.ones((2, 2), dtype=torch.float64, device="cpu") * variance
    endpoint_scores = torch.stack((center_score - half, center_score + half), dim=1)
    contributions = torch.stack((centered, centered), dim=1) / math.sqrt(n * (n - 1))
    extremizers = {
        f"y{arm}_{kind}": torch.where(
            d == arm, outcome, torch.full_like(outcome, endpoint)
        ).tolist()
        for arm in (0, 1)
        for kind, endpoint in (("lower", a), ("upper", b))
    }
    counts = [int((d == arm).sum()) for arm in (0, 1)]
    metadata.update(computation_resource_plan=plan, planned_work=work)
    state = dict(
        target="population E[Y(1)-Y(0)] identified set under consistency and fixed common support",
        outcomes=outcome.tolist(),
        assignment=d.tolist(),
        support=[a, b],
        arm_counts=counts,
        center_scores=center_score.tolist(),
        support_midpoint_expansion=midpoint,
        center_estimate=float(center),
        endpoint_scores=endpoint_scores.tolist(),
        endpoint_centered_scores=torch.stack((centered, centered), dim=1).tolist(),
        normalized_endpoint_contributions=contributions.tolist(),
        endpoint_covariance=covariance.tolist(),
        endpoint_standard_errors=[float(se), float(se)],
        ate_bounds=bounds.tolist(),
        bound_width=width,
        confidence_region=region.tolist(),
        critical_value=float(critical),
        tail_error=(1 - level) / 2,
        covariance_type="HC1 unbiased IID mean covariance, denominator N(N-1)",
        endpoint_dependence="U_i-L_i=upper-lower; identical centered endpoint scores; covariance rank at most one",
        covariance_rank=0 if variance == 0 else 1,
        extremizers=extremizers,
        zero_variance_status="singleton_support_identifies_zero"
        if width == 0
        else "zero_empirical_endpoint_variance"
        if variance == 0
        else None,
        inference="asymptotic Bonferroni one-sided endpoint limits covering the entire population identified set; not a finite-sample guarantee or Imbens-Manski optimized point-parameter CI",
        assumptions="IID complete observations; consistency/SUTVA; both potential outcomes in prespecified common support; nondegenerate endpoint-score CLT or deterministic-score limit; no treatment ignorability assumed",
        computation_resource_plan=plan,
        planned_work=work,
    )
    tables = {
        "bounds": m.frame(
            [["population_ate", *bounds.tolist(), width, float(se), *region.tolist(), level]],
            columns=[
                "target",
                "lower",
                "upper",
                "width",
                "endpoint_se",
                "region_lower",
                "region_upper",
                "level",
            ],
        ),
        "endpoint_covariance": m.frame(
            covariance.tolist(), columns=["lower", "upper"], index=["lower", "upper"]
        ),
        "subject_scores": m.frame(
            [
                [
                    i,
                    int(d[i]),
                    float(outcome[i]),
                    float(center_score[i]),
                    *endpoint_scores[i].tolist(),
                    *contributions[i].tolist(),
                ]
                for i in range(n)
            ],
            columns=[
                "position",
                "treatment",
                "observed_y",
                "center_score",
                "lower_score",
                "upper_score",
                "lower_contribution",
                "upper_contribution",
            ],
        ),
        "extremizers": m.frame(
            [
                [
                    i,
                    *[
                        extremizers[key][i]
                        for key in ("y0_lower", "y0_upper", "y1_lower", "y1_upper")
                    ],
                ]
                for i in range(n)
            ],
            columns=["position", "y0_lower", "y0_upper", "y1_lower", "y1_upper"],
        ),
    }
    return m.result(
        "manski_ate_inference",
        tables,
        metadata,
        dict(
            y=y,
            treatment=treatment,
            lower=a,
            upper=b,
            sampling=sampling,
            level=level,
            missing=missing,
            device=device,
            weights=None,
            max_work=max_work,
        ),
        state,
        notes=[
            state["inference"],
            state["assumptions"],
            "Zero empirical variance does not establish certainty about unsampled outcomes.",
        ],
    )


def _level(value):
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, Integral):
        kind, value = "int", int(value)
        if abs(value) > 2**63 - 1:
            raise AnalysisError(
                "invalid_strata", "Integer stratum labels must fit the signed int64 domain."
            )
    elif isinstance(value, Real) and math.isfinite(float(value)):
        kind, value = "float", float(value)
    elif isinstance(value, str) and len(value.encode("utf-8")) <= 1024:
        kind = "str"
    else:
        raise AnalysisError(
            "invalid_strata",
            "Stratum labels must be finite scalar numbers, Boolean values or strings.",
        )
    return dict(type=kind, value=value)


def _levels(levels):
    if not isinstance(levels, (list, tuple)) or not 1 <= len(levels) <= 32:
        raise AnalysisError(
            "invalid_strata",
            "Declare 1..32 fixed discrete levels; automatic binning is unavailable.",
        )
    labels = [_level(value) for value in levels]
    keys = [m.canonical(value) for value in labels]
    if len(set(keys)) != len(keys):
        raise AnalysisError("invalid_strata", "Declared typed levels must be unique.")
    return labels, {key: i for i, key in enumerate(keys)}


def _trim(values, numerator, denominator, *, largest):
    order = torch.argsort(values, descending=largest, stable=True)
    ordered = values[order]
    unique, inverse, counts = torch.unique_consecutive(
        ordered, return_inverse=True, return_counts=True
    )
    starts = counts.cumsum(0) - counts
    retained = torch.minimum((numerator - starts * denominator).clamp_min(0), counts * denominator)
    group_weights = retained.to(torch.float64) / (counts * denominator).to(torch.float64)
    retention = torch.empty_like(values)
    retention[order] = group_weights[inverse]
    mass = numerator / denominator
    weights = retention / mass
    value = _sum(_multiply(values, weights, "fractional trimmed outcome"))
    ties = [
        [float(v), int(count), float(weight)]
        for v, count, weight in zip(unique, counts, group_weights)
    ]
    return value, retention, weights, ties


@m.procedure
def stratified_lee_bounds(
    data,
    y,
    treatment,
    selection,
    *,
    strata,
    levels,
    design,
    monotonicity,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Discrete conditional-randomization Lee identification bounds; no SE/CI.

    Each prespecified cell is trimmed separately. Overall weights are its full
    cohort proportion times its lower selection rate, normalized across cells;
    treatment allocation probabilities may differ across cells.
    """
    m.options(device, weights, max_work)
    c.check_choice(design, "design", ("randomized_within_strata",))
    c.check_choice(monotonicity, "monotonicity", ("increasing", "decreasing"))
    _complete(missing)
    names = _roles(y, treatment, selection, strata)
    labels, lookup = _levels(levels)
    escaped_bytes = max(len(m.canonical(label).encode()) for label in labels)
    source, work, cost, plan = _preflight(
        data, names, max_work, cells=len(labels), label_bytes=escaped_bytes
    )
    # Validate each scalar before allocating repeated typed row labels. Once
    # this passes, actual saved row labels are bounded by admitted declarations.
    for value in source[strata]:
        if m.canonical(_level(value)) not in lookup:
            raise AnalysisError(
                "undeclared_stratum", "Every original row must belong to a declared typed stratum."
            )
    rows, metadata = m.sample(
        source,
        [treatment, selection, strata],
        numeric=[treatment, selection],
        missing="raise",
        max_work=max_work,
        cost=cost,
    )
    d = m.binary(rows, treatment)
    if pd.api.types.is_bool_dtype(rows[selection].dtype):
        raise AnalysisError(
            "invalid_selection", "Selection must use numeric 0/1, not Boolean labels."
        )
    s = m.tensor(rows, selection)
    if not bool(((s == 0) | (s == 1)).all()):
        raise AnalysisError("invalid_selection", "Selection must contain numeric 0/1 only.")
    raw_levels = [_level(value) for value in rows[strata].tolist()]
    if any(m.canonical(value) not in lookup for value in raw_levels):
        raise AnalysisError(
            "undeclared_stratum", "Every original row must belong to a declared typed stratum."
        )
    codes = torch.tensor(
        [lookup[m.canonical(value)] for value in raw_levels], dtype=torch.int64, device="cpu"
    )
    c.require_numeric(source, [y])
    if pd.api.types.is_bool_dtype(source[y].dtype):
        raise AnalysisError(
            "invalid_outcome", "Selected outcomes must be numeric, not Boolean labels."
        )
    observed_positions = torch.nonzero(s == 1).flatten().tolist()
    observed_rows = source.iloc[observed_positions].loc[:, [y]].copy()
    if observed_rows[y].isna().any():
        raise AnalysisError(
            "missing_selected_outcome", "Every selected row requires a finite observed outcome."
        )
    outcomes = m.tensor(observed_rows, y)
    n = len(rows)
    selected_y = torch.zeros(n, dtype=torch.float64, device="cpu")
    selected_y[torch.tensor(observed_positions, dtype=torch.int64, device="cpu")] = outcomes
    high = 1 if monotonicity == "increasing" else 0
    low = 1 - high
    retention = torch.zeros((n, 2), dtype=torch.float64, device="cpu")
    local_weights = torch.zeros_like(retention)
    centered_y = torch.zeros_like(selected_y)
    cell_records = []
    cell_bounds = []
    cell_mass = []
    for cell, label in enumerate(labels):
        member = codes == cell
        counts = [int((member & (d == arm)).sum()) for arm in (0, 1)]
        selected = [int((member & (d == arm) & (s == 1)).sum()) for arm in (0, 1)]
        if min(counts) == 0 or min(selected) == 0:
            raise AnalysisError(
                "empty_stratum_support",
                "Every declared stratum needs both assignment arms and selected outcomes in both arms.",
            )
        difference = selected[1] * counts[0] - selected[0] * counts[1]
        if (high == 1 and difference < 0) or (high == 0 and difference > 0):
            raise AnalysisError(
                "incompatible_empirical_selection",
                "A stratum's empirical rates contradict the declared common monotonicity; the entire calculation is refused.",
            )
        positions = [torch.nonzero(member & (d == arm) & (s == 1)).flatten() for arm in (0, 1)]
        anchor = selected_y[positions[low][0]]
        cell_selected = member & (s == 1)
        centered_y[cell_selected] = selected_y[cell_selected] - anchor
        means = [float(_mean(selected_y[p])) for p in positions]
        numerator = selected[low] * counts[high]
        denominator = counts[low]
        lower_tail, rlo, wlo, tlo = _trim(
            selected_y[positions[high]], numerator, denominator, largest=False
        )
        upper_tail, rhi, whi, thi = _trim(
            selected_y[positions[high]], numerator, denominator, largest=True
        )
        retention[positions[low]] = 1
        local_weights[positions[low]] = 1 / selected[low]
        retention[positions[high], 0] = rlo if high == 1 else rhi
        retention[positions[high], 1] = rhi if high == 1 else rlo
        local_weights[positions[high], 0] = wlo if high == 1 else whi
        local_weights[positions[high], 1] = whi if high == 1 else wlo
        # Sum the two signed arms together: separately rounding two large
        # means can otherwise invent a zero effect after their cancellation.
        cell_contributions = _multiply(
            _multiply(
                local_weights[member], (2 * d[member] - 1)[:, None], "within-cell signed weight"
            ),
            centered_y[member, None],
            "within-cell outcome",
        )
        bounds = _sum(cell_contributions)
        if bounds[0] > bounds[1]:
            raise AnalysisError(
                "numerical_failure", "Signed stratum bounds are numerically reversed."
            )
        rates = [selected[arm] / counts[arm] for arm in (0, 1)]
        mass = sum(counts) / n * rates[low]
        cell_mass.append(mass)
        cell_bounds.append(bounds)
        cell_records.append(
            dict(
                stratum_id=cell,
                level=label,
                positions=torch.nonzero(member).flatten().tolist(),
                arm_counts=counts,
                selected_counts=selected,
                selection_rates=rates,
                selected_means=means,
                outcome_centering_anchor=float(anchor),
                trimmed_tail_means=[float(lower_tail), float(upper_tail)],
                cohort_fraction=sum(counts) / n,
                always_selected_mass=mass,
                trimmed_arm=high,
                retained_selected_mass=dict(
                    numerator=numerator, denominator=denominator, value=numerator / denominator
                ),
                retention_fraction=(numerator / denominator) / selected[high],
                lower_tail_tie_groups=tlo,
                upper_tail_tie_groups=thi,
                ate_bounds=bounds.tolist(),
            )
        )
    masses = torch.tensor(cell_mass, dtype=torch.float64, device="cpu")
    total_mass = _sum(masses)
    population_weights = masses / total_mass
    for cell, record in enumerate(cell_records):
        record["population_weight"] = float(population_weights[cell])
    all_bounds = torch.stack(cell_bounds)
    bounds = _sum(_multiply(all_bounds, population_weights[:, None], "stratum-weighted bound"))
    _finite(bounds, "overall Lee bounds")
    if bounds[0] > bounds[1]:
        raise AnalysisError(
            "numerical_failure", "Overall identification bounds are numerically reversed."
        )
    weights = local_weights * population_weights[codes, None]
    signs = (2 * d - 1)[:, None]
    contributions = _multiply(
        _multiply(weights, signs, "signed row weight"), centered_y[:, None], "selected row outcome"
    )
    observed_y = [None] * n
    for i, position in enumerate(observed_positions):
        observed_y[position] = float(outcomes[i])
    identity = dict(
        y=y,
        dtype=str(source[y].dtype),
        positions=observed_positions,
        values=outcomes.tolist(),
        topology_sha256=metadata["sample_sha256"],
        typed_levels=labels,
        typed_row_levels=raw_levels,
    )
    metadata.update(
        columns=names,
        n_selected=len(observed_positions),
        n_unselected=n - len(observed_positions),
        selected_positions=observed_positions,
        outcome_observation_sha256=m.digest(identity),
        full_scientific_sample_sha256=m.digest(identity),
        computation_resource_plan=plan,
        planned_work=work,
        missing_outcome_policy="All original treatment/selection/strata rows required; unselected Y ignored, selected Y finite",
    )
    state = dict(
        target="E[Y(1)-Y(0) | S(1)=S(0)=1], always-selected population, discrete conditional randomization",
        assignment=d.tolist(),
        selection=s.tolist(),
        observed_y=observed_y,
        typed_stratum_levels=labels,
        typed_row_strata=raw_levels,
        stratum_codes=codes.tolist(),
        strata=cell_records,
        always_selected_fraction=float(total_mass),
        stratum_population_weights=population_weights.tolist(),
        cell_ate_bounds=all_bounds.tolist(),
        ate_bounds=bounds.tolist(),
        bound_width=float(bounds[1] - bounds[0]),
        retention_lower_bound=retention[:, 0].tolist(),
        retention_upper_bound=retention[:, 1].tolist(),
        within_stratum_mean_weights=local_weights.tolist(),
        population_mean_weights=weights.tolist(),
        row_signed_bound_contributions=contributions.tolist(),
        row_outcome_centering_anchors=[
            cell_records[int(code)]["outcome_centering_anchor"] for code in codes
        ],
        contribution_centering_rule="Within each cell subtract a common observed low-arm anchor; each arm's conditional mean weights sum to one algebraically, so the anchor cancels in its treatment contrast.",
        row_contribution_bound_reconstruction=_sum(contributions).tolist(),
        outcome_identity=identity,
        tie_rule="Fractional boundary mass shared equally across equal-outcome rows",
        population_weight_rule="P_hat(X=x)*min(s_hat_0(x),s_hat_1(x)) / sum_x P_hat(X=x)*min(s_hat_0(x),s_hat_1(x))",
        covariance=None,
        standard_error=None,
        confidence_interval=None,
        inference="plug-in sharp conditional-law identification bounds; no sampling confidence interval or randomization test",
        assumptions="Pretreatment fixed discrete partition; conditional random assignment independent of both potential outcomes and selection within each stratum; consistency/SUTVA; common declared unit-level monotone selection; positive treatment and selected support in every declared cell",
        empirical_direction_check="Exact integer comparison of cross-multiplied arm rates in every cell; sampling noise incompatibility is a refusal, not a population monotonicity test",
        computation_resource_plan=plan,
        planned_work=work,
    )
    tables = {
        "bounds": m.frame(
            [["always_selected_ate", *bounds.tolist(), float(bounds[1] - bounds[0])]],
            columns=["target", "lower", "upper", "width"],
        ),
        "strata": m.frame(
            [
                [
                    r["stratum_id"],
                    m.canonical(r["level"]),
                    *r["arm_counts"],
                    *r["selected_counts"],
                    *r["selection_rates"],
                    *r["selected_means"],
                    r["cohort_fraction"],
                    r["always_selected_mass"],
                    r["population_weight"],
                    r["retention_fraction"],
                    *r["ate_bounds"],
                ]
                for r in cell_records
            ],
            columns=[
                "stratum_id",
                "typed_level_json",
                "n0",
                "n1",
                "selected0",
                "selected1",
                "rate0",
                "rate1",
                "mean0",
                "mean1",
                "cohort_fraction",
                "always_selected_mass",
                "population_weight",
                "retention_fraction",
                "lower",
                "upper",
            ],
        ),
        "trimming_weights": m.frame(
            [
                [
                    i,
                    int(codes[i]),
                    int(d[i]),
                    int(s[i]),
                    observed_y[i],
                    *retention[i].tolist(),
                    *local_weights[i].tolist(),
                    *weights[i].tolist(),
                    *contributions[i].tolist(),
                ]
                for i in range(n)
            ],
            columns=[
                "position",
                "stratum_id",
                "treatment",
                "selection",
                "observed_y",
                "retention_lower",
                "retention_upper",
                "local_mean_weight_lower",
                "local_mean_weight_upper",
                "population_mean_weight_lower",
                "population_mean_weight_upper",
                "signed_lower_contribution",
                "signed_upper_contribution",
            ],
        ),
    }
    return m.result(
        "stratified_lee_bounds",
        tables,
        metadata,
        dict(
            y=y,
            treatment=treatment,
            selection=selection,
            strata=strata,
            levels=labels,
            design=design,
            monotonicity=monotonicity,
            missing=missing,
            device=device,
            weights=None,
            max_work=max_work,
        ),
        state,
        notes=[
            state["inference"],
            state["assumptions"],
            "Stratum weights target always-selected subjects, not the full cohort or raw selected-arm cell shares.",
        ],
    )
