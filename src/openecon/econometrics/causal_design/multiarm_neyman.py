"""Known Bernoulli and fixed-count multi-arm finite-population inference.

All covariance outputs are estimated conservative design bounds, not the
unidentified true randomization covariance. Normal inference is asymptotic.
"""

from __future__ import annotations

from datetime import date, time, timedelta
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import common as c
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.resources import plan_workspace

from . import common as m
from .neyman import _add, _difference, _label_bytes, _moments
from .observational_survival import _SUM_CAPACITY, _divide, _finite, _multiply, _sum_rows
from .paired import _pair_key


def _declaration(design, required, missing):
    if not isinstance(design, str) or design != required:
        raise AnalysisError("unsupported_design", f"Explicitly declare design='{required}'.")
    if missing != "raise":
        raise AnalysisError(
            "unsupported_missing",
            "Design inference requires missing='raise'; deleting outcomes changes the fixed population and assignment topology.",
        )


def _typed(value):
    try:
        key, label = _pair_key(value)
    except AnalysisError as exc:
        raise AnalysisError(
            "invalid_design_identifier", "Arm/stratum labels must be finite typed scalar labels."
        ) from exc
    _label_bytes(label)
    return key, label


def _arm_labels(arms):
    if not isinstance(arms, (list, tuple)) or not 3 <= len(arms) <= 8:
        raise AnalysisError(
            "invalid_arms", "arms must explicitly list 3..8 ordered treatment labels."
        )
    keys, labels, kinds = [], [], []
    for value in arms:
        key, label = _typed(value)
        if key in keys:
            raise AnalysisError("invalid_arms", "Declared typed arm labels must be distinct.")
        keys.append(key)
        labels.append(label)
        kinds.append(key[0])
    return keys, labels, kinds


def _exact_number(value, label, *, defer_nonfinite=False):
    """Reject numeric admission that silently changes the supplied value."""
    if isinstance(value, Real) and not isinstance(value, bool):
        kind = "integer" if isinstance(value, Integral) else "numeric"
        try:
            represented = float(value)
        except (OverflowError, ValueError) as exc:
            raise AnalysisError(
                "numerical_failure", f"The original {kind} {label} does not fit exact float64."
            ) from exc
        if defer_nonfinite and not math.isfinite(represented):
            # Missing/nonfinite outcomes are diagnosed after the original
            # assignment topology, rather than changing validation order.
            return
        if not math.isfinite(represented) or represented != value:
            raise AnalysisError(
                "numerical_failure", f"The original {kind} {label} does not fit exact float64."
            )


def _gradual_underflow():
    normal = torch.tensor(torch.finfo(torch.float64).tiny, dtype=torch.float64, device="cpu")
    half = normal * 0.5
    if bool(half == 0) or bool(half * 2 != normal):
        raise AnalysisError(
            "numerical_failure",
            "Native expansion arithmetic requires gradual float64 underflow; caller FTZ/DAZ mode is unsupported and is not changed.",
        )


def _index_bytes(value):
    # Generic metadata labeling falls back to str(value). Refuse compound/custom
    # objects before that conversion can allocate an unbounded caller repr.
    if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "item"):
        value = value.item()
    if not (
        value is None
        or value is pd.NA
        or isinstance(value, (str, bool, int, float, date, time, timedelta))
    ):
        raise AnalysisError(
            "invalid_index_label",
            "Original indices require bounded scalar numeric, text or calendar labels; compound/custom labels are unsupported.",
        )
    return _label_bytes(value)


def _contrasts(contrasts, null_values, k):
    if contrasts is None:
        contrasts = {
            f"difference[{a},{b}]": [int(j == b) - int(j == a) for j in range(k)]
            for a in range(k)
            for b in range(a + 1, k)
        }
    if not isinstance(contrasts, dict) or not 1 <= len(contrasts) <= 32:
        raise AnalysisError(
            "invalid_contrasts", "contrasts must be an ordered mapping with 1..32 rows."
        )
    names, rows, certificates = [], [], []
    for name, coefficients in contrasts.items():
        c.check_name(name, "contrast name")
        _label_bytes(name)
        if name in {f"mean[{a}]" for a in range(k)}:
            raise AnalysisError("invalid_contrasts", "A contrast name collides with an arm mean.")
        if not isinstance(coefficients, (list, tuple)) or len(coefficients) != k:
            raise AnalysisError(
                "invalid_contrasts", "Every contrast must have one coefficient per ordered arm."
            )
        for value in coefficients:
            _exact_number(value, "contrast coefficient")
        row = [c.check_number(value, "contrast coefficient") for value in coefficients]
        if not any(row):
            raise AnalysisError("invalid_contrasts", "A contrast must have a nonzero coefficient.")
        # Binary64 inputs have exact dyadic ratios. This is an exact admission
        # certificate, not a floating tolerance pretending a row sums to zero.
        ratios = [value.as_integer_ratio() for value in row]
        denominator = max(den for _, den in ratios)
        integers = [num * (denominator // den) for num, den in ratios]
        if sum(integers) != 0:
            raise AnalysisError(
                "invalid_contrasts",
                "The represented contrast coefficients must sum exactly to zero.",
            )
        names.append(name)
        rows.append(row)
        certificates.append(
            dict(denominator=str(denominator), numerators=[str(v) for v in integers])
        )
    if null_values is None:
        nulls = [0.0] * len(names)
    elif not isinstance(null_values, dict) or set(null_values) != set(names):
        raise AnalysisError(
            "invalid_null_values", "null_values must map every contrast name exactly once."
        )
    else:
        for value in null_values.values():
            _exact_number(value, "contrast null value")
        nulls = [c.check_number(null_values[name], "contrast null value") for name in names]
    transform = torch.cat(
        (
            torch.eye(k, dtype=torch.float64, device="cpu"),
            torch.tensor(rows, dtype=torch.float64, device="cpu"),
        )
    )
    return names, rows, nulls, certificates, transform


def _preflight(data, names, dimension, k, max_work, *, labels=(), typed_columns=()):
    _gradual_underflow()
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset", "Design inference requires a resident table, not Dataset replay."
        )
    names = [c.check_name(name, "selected role") for name in names]
    if len(set(names)) != len(names):
        raise AnalysisError(
            "invalid_spec", "Selected outcome, assignment and design roles must be distinct."
        )
    source = c.source(data)
    n = len(source)
    if n > 100000:
        raise AnalysisError("resource_limit", "Design inference is limited to100000 original rows.")
    absent = [name for name in names if name not in source]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {absent}.")
    for name in names:
        if (
            pd.api.types.is_float_dtype(source[name].dtype)
            and getattr(source[name].dtype, "itemsize", 8) > 8
        ):
            raise AnalysisError(
                "unsupported_dtype", "Floating input dtypes wider than binary64 are unsupported."
            )
    for value in source[names[0]]:
        _exact_number(value, "outcome", defer_nonfinite=True)
    label_bytes = sum(_index_bytes(value) for value in source.index)
    label_bytes += sum(_label_bytes(value) for value in (*names, *labels))
    for column in typed_columns:
        label_bytes += sum(_label_bytes(_typed(value)[1]) for value in source[column])
    # Worst-case 64-part expansions for all original/global/group coordinates,
    # group covariance construction and every retained table/state/JSON copy.
    cost = 128 + 48 * _SUM_CAPACITY * dimension + 16 * k * dimension * dimension
    work = n * cost + 64 * _SUM_CAPACITY * k * dimension * dimension + 32 * label_bytes
    m.work(
        work,
        max_work,
        "complete design, bounded point sums, full covariance bound and scientific state",
    )
    plan = plan_workspace(
        "known Bernoulli/multi-arm complete inference",
        {
            "selected_original_topology_numeric_copies_and_provenance": 2048 * n * len(names),
            "original_group_scores_and_complete_tables_state": 3072 * n * dimension,
            "complete_covariance_and_all_group_bounds": 2048 * n * dimension * dimension,
            "bounded_expansion_buffers_and_products": 128 * _SUM_CAPACITY * dimension * dimension,
            "escaped_labels_full_metadata_tables_and_json": 32 * label_bytes,
        },
    )
    return source, cost, work, plan.record(), label_bytes


def _covariance(transform, variances):
    products = _multiply(
        transform[:, None, :], transform[None, :, :], "contrast covariance coefficient"
    )
    weighted = _multiply(products, variances, "shared covariance bound contribution")
    bound = _sum_rows(weighted.permute(2, 0, 1).reshape(len(variances), -1)).reshape(
        len(transform), len(transform)
    )
    _finite(bound, "complete covariance bound")
    if not bool((bound == bound.T).all()) or not bool((bound.diag() >= 0).all()):
        raise AnalysisError(
            "numerical_failure", "The covariance bound is not symmetric with nonnegative variances."
        )
    scale = float(bound.abs().max())
    if scale:
        normalized = _divide(bound, scale, "covariance bound PSD normalization")
        if (
            float(torch.linalg.eigvalsh(normalized).min())
            < -256 * len(bound) * torch.finfo(torch.float64).eps
        ):
            raise AnalysisError(
                "numerical_failure",
                "The unrepaired covariance bound exceeds its PSD roundoff domain.",
            )
    return bound


def _product_parts(left, right, label):
    """Exact two-part products within the declared finite scaling domain."""
    product = _multiply(left, right, label)
    a, ae = torch.frexp(left)
    b, be = torch.frexp(right)
    splitter = 134217729.0
    ac, bc = a * splitter, b * splitter
    ah, bh = ac - (ac - a), bc - (bc - b)
    al, bl = a - ah, b - bh
    high = a * b
    low = ((ah * bh - high) + ah * bl + al * bh) + al * bl
    exponent = ae + be
    scaled_high = _finite(torch.ldexp(high, exponent), f"{label} high part")
    scaled_low = _finite(torch.ldexp(low, exponent), f"{label} low part")
    if (
        not bool((scaled_high == product).all())
        or not bool((torch.ldexp(scaled_high, -exponent) == high).all())
        or not bool((torch.ldexp(scaled_low, -exponent) == low).all())
    ):
        raise AnalysisError(
            "numerical_failure", f"The exact {label} expansion is not representable."
        )
    return scaled_high, scaled_low


def _point_anchored(values):
    anchor = values[values.abs().argmin()]
    negative = -anchor
    residuals = _finite(values + negative, "common-location residuals")
    z = residuals - values
    low = _finite((values - (residuals - z)) + (negative - z), "common-location remainder")
    # Merely checking whether subtraction swallowed an entire operand misses
    # lost low bits, e.g. 1e16-2.5. Any nonzero exact TwoSum remainder means the
    # original outcomes must stay uncentered in the native expansion instead.
    if bool((low != 0).any()):
        return torch.tensor(0.0, dtype=torch.float64, device="cpu"), values.clone()
    return anchor, residuals


def _count_denominator(groups, codes, k, n):
    # A common integer denominator removes artificial residuals from separately
    # rounded per-row division, including an exactly zero arm/contrast numerator.
    # Bound it before multiplication: every count factor is an exact binary64
    # integer, and Python integer operations certify counts, not estimate effects.
    common = 1
    for group in groups:
        for arm in range(k):
            count = sum(int(codes[i]) == arm for i in group["positions"])
            common = common // math.gcd(common, count) * count
            if common * n > 2**53:
                raise AnalysisError(
                    "numerical_failure",
                    "The fixed-count common denominator exceeds the exact2^53 integer domain.",
                )
    return common, n * common


def _fixed_points(values, codes, groups, transform):
    k = transform.shape[1]
    anchor, residuals = _point_anchored(values)
    common, denominator = _count_denominator(groups, codes, k, len(values))
    numerators = torch.zeros((len(values), 4, len(transform)), dtype=torch.float64, device="cpu")
    for group in groups:
        for arm in range(k):
            positions = torch.tensor(
                [i for i in group["positions"] if int(codes[i]) == arm],
                dtype=torch.int64,
                device="cpu",
            )
            factor = group["n"] * (common // len(positions))
            high, low = _product_parts(
                residuals[positions],
                torch.tensor(float(factor), dtype=torch.float64, device="cpu"),
                "original-row integer-count numerator",
            )
            high_high, high_low = _product_parts(
                high[:, None], transform[:, arm], "original-row contrast numerator"
            )
            low_high, low_low = _product_parts(
                low[:, None], transform[:, arm], "original-row contrast remainder"
            )
            numerators[positions] = torch.stack((high_high, high_low, low_high, low_low), dim=1)
    # Mean rows have coefficient one; contrast rows have exact certified zero.
    coefficients = torch.cat(
        (torch.ones(k, dtype=torch.float64), torch.zeros(len(transform) - k, dtype=torch.float64))
    )
    location = _multiply(anchor, coefficients, "fixed-count point anchor contribution")
    parts = numerators.reshape(-1, len(transform))
    numerator = _sum_rows(parts)
    quotient = _divide(numerator, denominator, "fixed-count point denominator")
    # Correct double rounding from first collapsing the numerator expansion.
    # Compute the original numerator minus quotient*denominator as an exact
    # expanded product/sum, rather than subtracting two rounded large numbers.
    high, low = _product_parts(
        quotient,
        torch.tensor(float(denominator), dtype=torch.float64, device="cpu"),
        "fixed-count quotient denominator product",
    )
    remainder = _sum_rows(torch.cat((parts, -high[None], -low[None])))
    correction = _divide(remainder, denominator, "fixed-count division remainder")
    centered_points = _sum_rows(torch.stack((quotient, correction)))
    points = _add(centered_points, location, "fixed-count point location restoration")
    return points, dict(
        original_row_point_numerator_parts=numerators.tolist(),
        point_count_denominator=denominator,
        point_count_common_multiple=common,
        point_residual_numerator=numerator.tolist(),
        point_division_remainder=remainder.tolist(),
        point_division_correction=correction.tolist(),
        point_residual_estimates=centered_points.tolist(),
        point_anchor=float(anchor),
        point_anchor_coefficients=coefficients.tolist(),
        point_anchor_contributions=location.tolist(),
    )


def _inference(points, bound, terms, k, nulls, level, alternative):
    critical = normal_isf((1 - level) / 2)
    if not math.isfinite(critical) or critical <= 0:
        raise AnalysisError("numerical_failure", "The normal critical value is not representable.")
    rows = []
    for i, (term, estimate, error) in enumerate(
        zip(terms, points.tolist(), bound.diag().sqrt().tolist(), strict=True)
    ):
        z = p = low = high = None
        null = None if i < k else nulls[i - k]
        if error:
            radius = critical * error
            low, high = estimate - radius, estimate + radius
            if (
                not all(math.isfinite(value) for value in (radius, low, high))
                or radius == 0
                or low == estimate
                or high == estimate
            ):
                raise AnalysisError(
                    "numerical_failure", "A nonzero normal confidence shift is not representable."
                )
            if i >= k:
                difference = float(
                    _difference(
                        torch.tensor(estimate, dtype=torch.float64),
                        torch.tensor(null, dtype=torch.float64),
                        "weak-null contrast",
                    )
                )
                z = difference / error
                if not math.isfinite(z) or (difference != 0 and z == 0):
                    raise AnalysisError(
                        "numerical_failure", "The normal weak-null statistic is not representable."
                    )
                p = (
                    2 * normal_sf(abs(z))
                    if alternative == "two-sided"
                    else normal_sf(z if alternative == "greater" else -z)
                )
        rows.append([term, estimate, error, null, z, p, low, high, None, error == 0])
    return m.frame(
        rows,
        columns=[
            "term",
            "estimate",
            "std_error",
            "null_value",
            "z",
            "p_value",
            "ci_low",
            "ci_high",
            "df",
            "degenerate_variance",
        ],
    ), critical


def _finish(
    name, points, bound, terms, k, nulls, metadata, settings, tables, state, alternative="two-sided"
):
    effects, critical = _inference(points, bound, terms, k, nulls, settings["level"], alternative)
    state.update(
        target="finite-population potential-outcome arm means and declared average-effect contrasts over all original supplied individuals",
        potential_outcomes_fixed=True,
        estimates=points.tolist(),
        estimated_covariance_bound=bound.tolist(),
        coordinate_order=terms,
        true_randomization_covariance=None,
        covariance_identified=False,
        covariance_method="estimated conservative design covariance bound; expectation dominates the unidentified true covariance in PSD order, not a realized-sample guarantee",
        inference="pointwise asymptotic normal under the stated unconditional design CLT and bound-consistency conditions; no exact Student-t, Fisher weak-null or finite-sample coverage guarantee",
        critical_value=critical,
        df=None,
        sum_method="native bounded64-part FastTwoSum expansion, final float64 rounding",
        sum_capacity=_SUM_CAPACITY,
        gradual_float64_underflow_required=True,
        missing_policy="raise only; original fixed population retained",
        assumptions=[
            "consistency and no interference between assignment units",
            "original population/groups fixed before assignment",
            "declared known assignment law is the actual experiment",
            "appropriate nondegenerate design CLT and covariance-bound consistency",
        ],
    )
    return m.result(
        name,
        {
            "effects": effects,
            "covariance_bound": m.frame(bound.tolist(), columns=terms, index=terms),
            **tables,
        },
        metadata,
        settings,
        state,
        notes=[
            "The covariance table is an estimated conservative design bound, not the unidentified actual randomization covariance. Conservatism is in expectation/asymptotically.",
            "All normal intervals are marginal and asymptotic; no simultaneous, finite-sample or Student-t coverage is asserted.",
            "Zero estimated variance retains the point and bound but gives no z, p or interval. It does not establish precision or population certainty.",
        ],
    )


@m.procedure
def bernoulli_neyman_ate(
    data,
    y,
    treatment,
    probability,
    *,
    design,
    null_effect=0.0,
    alternative="two-sided",
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Raw fixed-N HT arm means/ATE under independent known Bernoulli assignment.

    Declare design='bernoulli_randomized'. probability names the prespecified
    true assignment-probability column, not fitted observational propensities.
    All-zero/all-one observed assignments remain admitted. No HC1 factor or
    normalization by a realized arm count is used.
    """
    m.options(device, weights, max_work, level)
    _declaration(design, "bernoulli_randomized", missing)
    _exact_number(null_effect, "null effect")
    null_effect = c.check_number(null_effect, "null_effect")
    c.check_choice(alternative, "alternative", ("two-sided", "greater", "less"))
    source, cost, work, plan, label_bytes = _preflight(
        data, [y, treatment, probability], 3, 2, max_work
    )
    topology, topology_meta = m.sample(
        source,
        [treatment, probability],
        numeric=[treatment, probability],
        cost=cost,
        max_work=max_work,
    )
    assignment, p = m.tensor(topology, treatment), m.tensor(topology, probability)
    if pd.api.types.is_bool_dtype(topology[treatment].dtype) or not bool(
        ((assignment == 0) | (assignment == 1)).all()
    ):
        raise AnalysisError(
            "invalid_treatment",
            "Treatment must be numeric0/1; both observed arms are not required.",
        )
    if not bool(((p > 0) & (p < 1)).all()):
        raise AnalysisError(
            "invalid_probability",
            "Known preassigned Bernoulli probabilities must be strictly between0 and1; no clipping is performed.",
        )
    selected, metadata = m.sample(
        source,
        [y, treatment, probability],
        numeric=[y, treatment, probability],
        cost=cost,
        max_work=max_work,
    )
    outcome = m.tensor(selected, y)
    n = len(outcome)
    metadata.update(
        design_sample_sha256=topology_meta["sample_sha256"],
        computation_resource_plan=plan,
        planned_work=work,
        original_escaped_label_bytes=label_bytes,
    )
    # Original uncentered HT scores define the second-moment bound. Both these
    # scores and the actual inverse-probability anchor coefficients must fit.
    arm_scores = torch.zeros((n, 2), dtype=torch.float64, device="cpu")
    coefficient_rows = torch.zeros_like(arm_scores)
    for arm in (0, 1):
        used = assignment == arm
        denominator = p[used] if arm else 1 - p[used]
        arm_scores[used, arm] = _divide(
            _divide(outcome[used], n, "HT fixed population normalization"),
            denominator,
            "original-row HT score",
        )
        coefficient_rows[used, arm] = _divide(
            torch.ones_like(denominator) / n, denominator, "HT anchor coefficient"
        )
    transform = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]], dtype=torch.float64, device="cpu"
    )
    variances = _sum_rows(
        _multiply(arm_scores, arm_scores, "HT second-moment variance contribution")
    )
    bound = _covariance(transform, variances)
    anchor, residuals = _point_anchored(outcome)
    residual_scores = torch.zeros((n, 3), dtype=torch.float64, device="cpu")
    for arm in (0, 1):
        used = assignment == arm
        denominator = p[used] if arm else 1 - p[used]
        residual_scores[used, arm] = _divide(
            _divide(residuals[used], n, "HT centered population normalization"),
            denominator,
            "HT original-row centered score",
        )
    residual_scores[:, 2] = _difference(
        residual_scores[:, 1], residual_scores[:, 0], "HT residual contrast"
    )
    coefficient_scores = torch.cat(
        (coefficient_rows, (coefficient_rows[:, 1] - coefficient_rows[:, 0])[:, None]), dim=1
    )
    coefficients = _sum_rows(coefficient_scores)
    location = _multiply(anchor, coefficients, "HT observed anchor contribution")
    points = _add(_sum_rows(residual_scores), location, "HT point location restoration")
    terms = ["mean_control", "mean_treated", "difference"]
    settings = dict(
        y=y,
        treatment=treatment,
        probability=probability,
        design=design,
        null_effect=null_effect,
        alternative=alternative,
        level=level,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    tables = {
        "subjects": m.frame(
            list(
                zip(
                    metadata["positions"],
                    assignment.tolist(),
                    outcome.tolist(),
                    p.tolist(),
                    strict=True,
                )
            ),
            columns=["position", "treatment", "outcome", "assignment_probability"],
        )
    }
    state = dict(
        original_outcomes=outcome.tolist(),
        original_assignment=assignment.tolist(),
        known_assignment_probabilities=p.tolist(),
        assignment_law="independent Bernoulli assignments with original prespecified known per-unit probabilities; unconditional on realized arm counts",
        probability_provenance="caller-declared true preassigned randomization law; learned propensities are unsupported and not authenticated by a column",
        original_row_ht_scores=arm_scores.tolist(),
        original_row_mean_contributions=residual_scores.tolist(),
        point_anchor=float(anchor),
        point_anchor_coefficients=coefficients.tolist(),
        point_anchor_contributions=location.tolist(),
        original_row_anchor_coefficients=coefficient_scores.tolist(),
        point_estimate_method="raw fixed-N HT; original-row common-location residuals plus actual observed inverse-probability coefficient sums, not location-invariant Hájek means",
        observed_arm_counts=[int((assignment == arm).sum()) for arm in (0, 1)],
        arm_variance_bounds=variances.tolist(),
        hc1_factor=None,
        variance_bound_formula="diag(sum_i I(A_i=a)*Yobs_i^2/p_ia^2)/N^2, transformed to both raw HT means and their difference; no empirical centering or HC1 factor",
        expectation_psd_gap="N^-2 sum_i [Y_i(0),Y_i(1)] [Y_i(0),Y_i(1)]'; transformed by the same coordinate matrix",
        asymptotic_conditions="independent assignments; along the fixed-potential population sequence probabilities stay away from0/1, Lindeberg/no-dominating-score conditions and second-moment-bound consistency hold; realized empty arms do not identify missing potential means",
        all_zero_all_one_allowed=True,
    )
    return _finish(
        "bernoulli_neyman_ate",
        points,
        bound,
        terms,
        2,
        [null_effect],
        metadata,
        settings,
        tables,
        state,
        alternative,
    )


def _multiarm(
    data,
    y,
    treatment,
    strata,
    *,
    arms,
    design,
    contrasts,
    null_values,
    level,
    missing,
    device,
    weights,
    max_work,
):
    m.options(device, weights, max_work, level)
    required = "complete_randomized" if strata is None else "stratified_randomized"
    _declaration(design, required, missing)
    keys, labels, kinds = _arm_labels(arms)
    k = len(keys)
    contrast_names, coefficients, nulls, certificates, transform = _contrasts(
        contrasts, null_values, k
    )
    terms = [f"mean[{a}]" for a in range(k)] + contrast_names
    names = [y, treatment] + ([] if strata is None else [strata])
    typed_columns = [treatment] + ([] if strata is None else [strata])
    source, cost, work, plan, label_bytes = _preflight(
        data,
        names,
        len(terms),
        k,
        max_work,
        labels=[*labels, *contrast_names],
        typed_columns=typed_columns,
    )
    topology, topology_meta = m.sample(
        source, names[1:], cost=cost, max_work=max_work, minimum=2 * k
    )
    lookup = {key: arm for arm, key in enumerate(keys)}
    assignments = []
    groups, found = [], {}
    for i, value in enumerate(topology[treatment]):
        key, _ = _typed(value)
        if key not in lookup:
            raise AnalysisError(
                "invalid_assignment",
                "An original treatment label is outside the declared ordered arms.",
            )
        assignments.append(lookup[key])
        group_key, group_label = (
            (None, None) if strata is None else _typed(topology[strata].iloc[i])
        )
        if group_key not in found:
            found[group_key] = len(groups)
            groups.append(
                dict(
                    kind=None if group_key is None else group_key[0],
                    label=group_label,
                    positions=[],
                )
            )
        groups[found[group_key]]["positions"].append(i)
    for group in groups:
        counts = [sum(assignments[i] == arm for i in group["positions"]) for arm in range(k)]
        if min(counts) < 2:
            raise AnalysisError(
                "insufficient_observations",
                "Every original group needs at least two observations in every declared arm.",
            )
        group.update(n=len(group["positions"]), arm_counts=counts)
    _count_denominator(groups, assignments, k, len(assignments))
    selected, metadata = m.sample(
        source, names, numeric=[y], cost=cost, max_work=max_work, minimum=2 * k
    )
    outcome = m.tensor(selected, y)
    codes = torch.tensor(assignments, dtype=torch.int64, device="cpu")
    metadata.update(
        design_sample_sha256=topology_meta["sample_sha256"],
        computation_resource_plan=plan,
        planned_work=work,
        original_escaped_label_bytes=label_bytes,
    )
    summaries, group_rows, matrices, group_labels = [], [], [], []
    for gi, group in enumerate(groups):
        positions = group["positions"]
        values, local_codes = outcome[positions], codes[positions]
        variances, centered = [], []
        for arm in range(k):
            used = local_codes == arm
            _, variance, residuals = _moments(values[used, None])
            variances.append(
                _divide(variance[0, 0], int(used.sum()), "fixed-count arm variance bound")
            )
            centered.append(residuals[:, 0].tolist())
        local_bound = _covariance(transform, torch.stack(variances))
        local_points, local_state = _fixed_points(
            values,
            local_codes,
            [dict(positions=list(range(len(values))), n=len(values))],
            transform,
        )
        weight = group["n"] / len(outcome)
        matrices.append(
            _multiply(
                _multiply(local_bound, weight, "group covariance weight"),
                weight,
                "group covariance contribution",
            )
        )
        summaries.append(
            dict(
                weight=weight,
                estimates=local_points.tolist(),
                covariance_bound=local_bound.tolist(),
                arm_centered_outcomes=centered,
                arm_variance_bounds=[float(v) for v in variances],
                **local_state,
            )
        )
        for term, point, variance in zip(
            terms, local_points.tolist(), local_bound.diag().tolist(), strict=True
        ):
            group_rows.append(
                [
                    gi,
                    group["kind"],
                    None,  # Installed below as object dtype, before numeric inference.
                    group["n"],
                    m.canonical(group["arm_counts"]),
                    weight,
                    term,
                    point,
                    variance,
                ]
            )
            group_labels.append(group["label"])
    bound = _sum_rows(torch.stack(matrices).reshape(len(groups), -1)).reshape(
        len(terms), len(terms)
    )
    points, point_state = _fixed_points(outcome, codes, groups, transform)
    group_table = m.frame(
        group_rows,
        columns=[
            "group_index",
            "identifier_type",
            "identifier",
            "n",
            "arm_counts",
            "weight",
            "term",
            "estimate",
            "variance_bound",
        ],
    )
    group_table["identifier"] = pd.Series(group_labels, dtype=object)
    arm_counts = [int((codes == arm).sum()) for arm in range(k)]
    arm_table = m.frame(
        [[a, kinds[a], None, arm_counts[a]] for a in range(k)],
        columns=["arm_index", "identifier_type", "identifier", "n_assigned"],
    )
    arm_table["identifier"] = pd.Series(labels, dtype=object)
    tables = dict(
        subjects=m.frame(
            list(zip(metadata["positions"], codes.tolist(), outcome.tolist(), strict=True)),
            columns=["position", "arm_index", "outcome"],
        ),
        arms=arm_table,
        groups=group_table,
        contrast_matrix=m.frame(
            coefficients, columns=[f"arm[{a}]" for a in range(k)], index=contrast_names
        ),
    )
    settings = dict(
        y=y,
        treatment=treatment,
        design=design,
        arms=labels,
        arm_types=kinds,
        contrasts=dict(zip(contrast_names, coefficients, strict=True)),
        null_values=dict(zip(contrast_names, nulls, strict=True)),
        level=level,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    if strata is not None:
        settings["strata"] = strata
    state = dict(
        original_outcomes=outcome.tolist(),
        original_arm_codes=assignments,
        arm_order=labels,
        arm_types=kinds,
        design_groups=groups,
        group_summaries=summaries,
        contrast_names=contrast_names,
        contrast_coefficients=coefficients,
        contrast_zero_sum_certificates=certificates,
        coordinate_transform=transform.tolist(),
        assignment_law="uniform assignments with each original fixed arm count, independently across predeclared strata"
        if strata is not None
        else "uniform complete assignments with each original fixed arm count",
        variance_bound_formula="sum_h (N_h/N)^2 diag(s_ha^2/n_ha), transformed jointly to all arm means and declared contrasts; unidentified potential-outcome covariance correction omitted",
        expectation_psd_gap="sum_h (N_h/N)^2 * L S_potential,h L' /N_h, positive semidefinite; complete design has one group",
        point_estimate_method="bounded original-row common-location residual contributions plus analytically certified [one for means,zero for contrast] anchor coefficients; rounded arm/group means are not aggregated",
        asymptotic_conditions="fixed number of arms/strata, positive limiting arm fractions in nonnegligible groups, no dominating centered potential outcome, covariance-bound consistency and nondegenerate contrast CLT",
        joint_tests=False,
        **point_state,
    )
    name = "multiarm_neyman_ate" if strata is None else "stratified_multiarm_neyman_ate"
    return _finish(name, points, bound, terms, k, nulls, metadata, settings, tables, state)


@m.procedure
def multiarm_neyman_ate(
    data,
    y,
    treatment,
    *,
    arms,
    design,
    contrasts=None,
    null_values=None,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Complete fixed-count 3..8-arm means/contrasts and shared Neyman bound."""
    return _multiarm(
        data,
        y,
        treatment,
        None,
        arms=arms,
        design=design,
        contrasts=contrasts,
        null_values=null_values,
        level=level,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )


@m.procedure
def stratified_multiarm_neyman_ate(
    data,
    y,
    treatment,
    strata,
    *,
    arms,
    design,
    contrasts=None,
    null_values=None,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Independent fixed-count strata, fixed-size weighted multi-arm effects."""
    c.check_name(strata, "strata")
    return _multiarm(
        data,
        y,
        treatment,
        strata,
        arms=arms,
        design=design,
        contrasts=contrasts,
        null_values=null_values,
        level=level,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )
