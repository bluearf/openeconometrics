"""Bounded joint mean inference under explicit finite transformation models.

The normalizers are invariant on each coordinate's transformation orbit. No
resampled variance estimate or Gaussian population approximation is used.
Exactness requires invariance of the *joint true-null subvector*, which is
stronger than zero means or equal marginal means. Numerical work uses CPU
float64 Torch; pandas handles the resident input and positional sample only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_integer_dtype, is_numeric_dtype

from openecon.analysis import _frame_hasher, _json_scalar
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.multiple import stepdown
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace

_MAX_ROWS = 10000
_MAX_COLUMNS = 32
_MAX_TRANSFORMS = 100000
_MAX_STATISTIC_CELLS = 500000
_MAX_WORK = 100000000
_CHUNK = 128
_SOURCE = "https://doi.org/10.1198/016214504000000539"


def _count(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise AnalysisError("invalid_option", f"{name} must be an integer in {low}..{high}.")
    return int(value)


def _options(calibration, draws, seed, alpha, tail, model):
    if not isinstance(calibration, str) or calibration not in {"enumerated", "monte_carlo"}:
        raise AnalysisError("invalid_option", "calibration must be enumerated or monte_carlo.")
    draws = _count(draws, "draws", 2, _MAX_TRANSFORMS)
    seed = _count(seed, "seed", 0, 2**63 - 1)
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise AnalysisError(
            "invalid_alpha", "alpha must be finite and strictly between zero and one."
        )
    if not isinstance(tail, str) or tail not in {"two-sided", "greater", "less"}:
        raise AnalysisError("invalid_option", "tail must be two-sided, greater or less.")
    if not isinstance(model, str) or not model.strip() or len(model) > 2000:
        raise AnalysisError(
            "missing_randomization_model",
            "Declare the joint true-null invariance model (1..2000 characters).",
        )
    return draws, seed, float(alpha), model.strip()


def _sample(data, columns, missing, group=None):
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError("unsupported_data", "Supply a resident pandas DataFrame.")
    if len(data) > _MAX_ROWS:
        raise AnalysisError(
            "work_budget_exceeded", "Randomization input requires at most 10000 rows."
        )
    if data.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    if (
        not isinstance(columns, (list, tuple))
        or not 1 <= len(columns) <= _MAX_COLUMNS
        or any(not isinstance(c, str) or not c for c in columns)
        or len(set(columns)) != len(columns)
    ):
        raise AnalysisError("invalid_columns", "Declare 1..32 unique outcome column names.")
    columns = list(columns)
    if not isinstance(missing, str) or missing not in {"raise", "drop"}:
        raise AnalysisError("invalid_option", "missing must be raise or drop.")
    if group is not None and (not isinstance(group, str) or not group or group in columns):
        raise AnalysisError("invalid_group", "Declare a separate group column name.")
    selected = columns + ([] if group is None else [group])
    if any(c not in data.columns for c in selected):
        raise AnalysisError("missing_columns", "Every outcome and group column must exist.")
    for name in columns:
        dtype = data[name].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise AnalysisError(
                "invalid_values", "Outcomes must contain real numeric values, not booleans."
            )
    # Admission precedes projected copies/conversion; it also covers nullable
    # pandas bridges, the mask and hash buffers. Input objects are caller-owned.
    input_plan = plan_workspace(
        "randomization resident input",
        {"projected_input_bridges_and_masks": len(data) * (len(selected) * 32 + 32)},
    )
    # Validate raw numeric identity before *any* float64 bridge. Otherwise a
    # large integer can change its null-relative sign, and extended-precision
    # differences can disappear before the orbit is constructed. Validate all
    # available observations, including rows later omitted for another input.
    for name in columns:
        series = data[name]
        if is_integer_dtype(series.dtype):
            minimum, maximum = series.min(skipna=True), series.max(skipna=True)
            if not pd.isna(minimum) and (int(minimum) < -(2**53) or int(maximum) > 2**53):
                raise AnalysisError(
                    "numerical_range",
                    "Integer observations must lie within the exact float64 domain [-2**53,2**53].",
                )
        elif getattr(series.dtype, "itemsize", 8) > 8:
            raise AnalysisError(
                "numerical_range",
                "Use float64-or-narrower numeric columns; wider observations require explicit user conversion.",
            )
    projected = data.loc[:, selected]
    keep = projected.notna().all(axis=1)
    if missing == "raise" and not bool(keep.all()):
        raise AnalysisError(
            "missing_values", "All tested outcomes and group identities must be complete."
        )
    positions = keep.to_numpy().nonzero()[0].tolist()
    if len(positions) < 2:
        raise AnalysisError(
            "insufficient_sample", "At least two jointly complete rows are required."
        )
    # Inf is invalid even when another outcome would exclude that row.
    for name in columns:
        values = projected[name].to_numpy(dtype="float64", na_value=math.nan)
        if any(math.isinf(float(v)) for v in values):
            raise AnalysisError("non_finite_values", "Outcomes may not contain infinite values.")
    labels, codes = [], None
    if group is not None:
        identities = projected[group].iloc[positions]
        for value in identities:
            if (
                not isinstance(value, (str, Real, Integral))
                or (isinstance(value, str) and not value)
                or (isinstance(value, Real) and not math.isfinite(float(value)))
            ):
                raise AnalysisError(
                    "invalid_group",
                    "Group identities must be finite scalar numbers or nonempty strings.",
                )
        raw_codes, raw_labels = pd.factorize(identities, sort=False)
        if len(raw_labels) != 2:
            raise AnalysisError(
                "invalid_group",
                "Exactly two observed groups are required after joint row deletion.",
            )
        labels = [_json_scalar(v) for v in raw_labels]
        codes = raw_codes.tolist()
    return {
        "frame": projected.iloc[positions],
        "columns": columns,
        "positions": positions,
        "original_rows": len(data),
        "missing_policy": missing,
        "input_hash": _frame_hasher(projected).hexdigest(),
        "sample_hash": _frame_hasher(projected.iloc[positions]).hexdigest(),
        "labels": labels,
        "codes": codes,
        "input_resource_plan": input_plan.record(),
    }


def _bounded_combinations(n, k):
    """Count combinations, stopping before an unadmitted huge integer exists."""
    value = 1
    for j in range(1, min(k, n - k) + 1):
        value = value * (n - j + 1) // j
        if value > _MAX_TRANSFORMS:
            return None
    return value


def _admit(n, m, calibration, draws, kind, n1=None):
    if kind == "sign":
        orbit_count = 2**n if n <= 16 else None
    else:
        orbit_count = _bounded_combinations(n, n1)
    if calibration == "enumerated" and orbit_count is None:
        raise AnalysisError(
            "enumeration_limit", "The complete transformation orbit exceeds 100000 assignments."
        )
    count = orbit_count if calibration == "enumerated" else draws
    work = count * n * m
    comparisons = count * m * m
    if count * m > _MAX_STATISTIC_CELLS or max(work, comparisons) > _MAX_WORK:
        raise AnalysisError(
            "work_budget_exceeded",
            "Transformations exceed 500000 stored statistics or 100000000 arithmetic/comparison operations.",
        )
    chunk = min(count, _CHUNK)
    plan = plan_workspace(
        "joint randomization transformations and maxT",
        {
            "input_centering_and_scales": n * m * 64,
            "assignment_chunk_and_rng_workspace": chunk * n * 32,
            "observed_and_joint_statistics": (count + 1) * m * 48,
            "stepdown_ordering_and_maxima": count * m * 16 + count * 24,
            "sample_positions_and_group_codes": n * 32,
        },
    )
    return count, orbit_count, work, comparisons, plan.record()


def _numeric(sample):
    value = torch.stack(
        [
            torch.as_tensor(
                sample["frame"][c].to_numpy(dtype="float64", copy=True), dtype=torch.float64
            )
            for c in sample["columns"]
        ],
        dim=1,
    )
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError(
            "non_finite_values", "The jointly selected sample must contain finite real values."
        )
    return value


def _targets(null_values, columns):
    if null_values is None:
        return [0.0] * len(columns)
    if not isinstance(null_values, Mapping) or set(null_values) != set(columns):
        raise AnalysisError(
            "invalid_null", "null_values must map every tested column to its null center."
        )
    values = []
    for column in columns:
        original = null_values[column]
        if isinstance(original, bool) or not isinstance(original, Real):
            raise AnalysisError("invalid_null", "Null centers must be finite real numbers.")
        if getattr(getattr(original, "dtype", None), "itemsize", 8) > 8:
            raise AnalysisError(
                "numerical_range",
                "Use float64-or-narrower null centers; wider scalars require explicit user conversion.",
            )
        try:
            value = float(original)
        except (OverflowError, ValueError, TypeError) as exc:
            raise AnalysisError(
                "invalid_null", "Null centers must be representable finite real numbers."
            ) from exc
        if not math.isfinite(value):
            raise AnalysisError("invalid_null", "Null centers must be finite real numbers.")
        if isinstance(original, Integral) and value != int(original):
            raise AnalysisError(
                "numerical_range", "Integer null centers must be exactly representable in float64."
            )
        values.append(value)
    return values


def _safe_scale(value):
    """Rescale columns before sums/squares to preserve finite-scale invariance."""
    maximum = value.abs().amax(0)
    if bool((maximum == 0).any()) or not bool(torch.isfinite(maximum).all()):
        raise AnalysisError(
            "zero_scale", "Every tested coordinate needs a nonzero finite orbit scale."
        )
    return value / maximum


def _inclusive(joint, n, tail):
    # All coordinates are bounded by sqrt(N), after stable maximum rescaling.
    # This conservative guard includes equality despite differing reduction
    # order. It can increase p-values; it never discards a computed exceedance.
    bound = 64 * torch.finfo(torch.float64).eps * n * math.sqrt(n)
    if tail == "two-sided":
        return joint.abs() + bound, bound
    return joint + bound if tail == "greater" else joint - bound, bound


def _output(
    sample,
    observed,
    joint,
    *,
    kind,
    model,
    calibration,
    seed,
    alpha,
    tail,
    orbit_count,
    work,
    comparisons,
    resource_plan,
    null,
    group=None,
    group_sizes=None,
    null_hash=None,
    transformation_hash=None,
):
    guarded, tolerance = _inclusive(joint, len(sample["positions"]), tail)
    tests = stepdown(
        observed,
        guarded,
        method="romano_wolf",
        tail=tail,
        calibration=calibration,
        alpha=alpha,
        labels=sample["columns"],
        null_description=model,
    )
    ordered_observed = (
        observed.abs() if tail == "two-sided" else (-observed if tail == "less" else observed)
    )
    ordered_joint = (
        guarded.abs() if tail == "two-sided" else (-guarded if tail == "less" else guarded)
    )
    marginal_counts = (ordered_joint >= ordered_observed).sum(0).tolist()
    ordering = tests.attrs["ordering_positions"]
    stages = []
    offset = 0
    for tied in tests.attrs["tie_groups"]:
        remaining = ordering[offset:]
        count = int((ordered_joint[:, remaining].amax(1) >= ordered_observed[tied[0]]).sum())
        stages.append({"removed_together": tied, "remaining": remaining, "exceedances": count})
        offset += len(tied)
    if kind == "sign":
        assumptions = (
            "Independent rows; the joint subvector of true-null coordinates is centrally symmetric "
            "about its declared null centers in each row. Joint complete-row selection must preserve "
            "this invariance. Zero means or marginal symmetry alone are insufficient."
        )
        statistic = "sum(X_j-null_j)/sqrt(sum((X_j-null_j)^2)); fixed orbit-invariant norm"
    else:
        assumptions = (
            "Independent pooled rows; fixed group counts; the joint true-null subvector has the "
            "same distribution in both groups and is exchangeable across labels. Joint complete-row "
            "selection must preserve this invariance. Equality of means or separate marginal "
            "distributions alone is insufficient."
        )
        statistic = "(mean(group0)-mean(group1))/(pooled centered RMS*sqrt(1/N0+1/N1)); fixed orbit-invariant scale"
    tests.attrs.update(
        assumptions=assumptions,
        statistic_kind=statistic,
        null_design_verified=False,
        null_design_scope="Transformations are implemented and checked; population invariance is caller-declared.",
        missing_policy="joint complete-case row deletion"
        if len(sample["positions"]) < sample["original_rows"]
        else "joint complete sample",
    )
    sample_rows = [
        {"position": position, **({"group": sample["labels"][sample["codes"][i]]} if group else {})}
        for i, position in enumerate(sample["positions"])
    ]
    return saved_summary(
        TableSet(
            {"tests": tests, "sample": table(sample_rows)},
            title="Joint row-sign mean stepdown"
            if kind == "sign"
            else "Joint two-group permutation mean stepdown",
            method="mean_sign_stepdown" if kind == "sign" else "mean_permutation_stepdown",
            columns=sample["columns"],
            family_size=len(observed),
            alpha=alpha,
            tail=tail,
            calibration=calibration,
            draws=len(joint),
            orbit_count=orbit_count,
            orbit_size_description=f"2**{len(sample['positions'])}"
            if kind == "sign"
            else f"choose({len(sample['positions'])},{group_sizes[0]})",
            orbit_count_exceeds_budget=orbit_count is None,
            enumerated=calibration == "enumerated",
            seed=seed,
            random_generator="local CPU torch.Generator; independent uniform transformations with replacement"
            if calibration == "monte_carlo"
            else "deterministic complete equally weighted orbit",
            p_convention="inclusive exceedances/count"
            if calibration == "enumerated"
            else "(inclusive exceedances+1)/(draws+1)",
            statistic_definition=statistic,
            observed_statistics=observed.tolist(),
            joint_null_statistics=joint.tolist(),
            marginal_exceedances=marginal_counts,
            stepdown_stage_counts=stages,
            joint_null_statistic_hash=hashlib.sha256(
                joint.contiguous().numpy().astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            transformation_hash=transformation_hash,
            input_hash=sample["input_hash"],
            sample_hash=sample["sample_hash"],
            null_hash=null_hash,
            null_values=null,
            group=group,
            group_labels=sample["labels"],
            group_sizes=group_sizes,
            group_difference="first observed group minus second observed group" if group else None,
            nobs=len(sample["positions"]),
            nobs_original=sample["original_rows"],
            sample_positions=sample["positions"],
            excluded_rows=sample["original_rows"] - len(sample["positions"]),
            missing_policy=sample["missing_policy"],
            assumptions=assumptions,
            declared_model=model,
            finite_sample_control="Strong FWER under the declared joint true-null invariance; nonrandomized conservative ties",
            numerical_tail_guard=tolerance,
            numerical_tail_rule="Move each transformed tail statistic outward by the recorded absolute float64 bound; inclusive/conservative equality handling",
            work=work,
            comparison_work=comparisons,
            resource_plan=resource_plan,
            input_resource_plan=sample["input_resource_plan"],
            precision="float64",
            input_conversion="Float64-or-narrower numeric columns; integer observations restricted to exact [-2**53,2**53] domain; integer null centers must convert exactly",
            device="cpu",
            resident=True,
            weights="unsupported",
            dataset_support=False,
            degrees_of_freedom=None,
            fitted_model=False,
            source=_SOURCE,
            persistence="oe.summary_state(output) stores every sample row and joint statistic; console preview may truncate",
        )
    )


@resident_cpu
@torch.no_grad()
def mean_sign_stepdown(
    data,
    columns,
    *,
    symmetry_model,
    null_values=None,
    calibration="enumerated",
    draws=9999,
    seed=1729,
    alpha=0.05,
    tail="two-sided",
    missing="raise",
):
    """Joint maxT stepdown using independent row sign flips about declared null centers.

    Independent rows must have a centrally symmetric joint true-null subvector;
    marginal zero means do not suffice. Enumeration includes all 2**N flips;
    MC samples iid uniform flips and uses plus-one p-values. Fixed column norms
    avoid degenerate orbit-specific studentization. Resident N<=10000, K<=32;
    orbit, arithmetic, stored-statistic and workspace admission precedes draws.
    """
    draws, seed, alpha, model = _options(calibration, draws, seed, alpha, tail, symmetry_model)
    sample = _sample(data, columns, missing)
    target = _targets(null_values, sample["columns"])
    n, m = len(sample["positions"]), len(sample["columns"])
    count, orbit, work, comparisons, plan = _admit(n, m, calibration, draws, "sign")
    x = _numeric(sample) - torch.tensor(target, dtype=torch.float64)
    if not bool(torch.isfinite(x).all()):
        raise AnalysisError(
            "non_finite_values", "Subtracting null centers exceeded finite float64 range."
        )
    x = _safe_scale(x)
    norm = torch.linalg.vector_norm(x, dim=0)
    normalized = x / norm
    observed = normalized.sum(0)
    joint = torch.empty((count, m), dtype=torch.float64)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    digest = hashlib.sha256()
    for start in range(0, count, _CHUNK):
        size = min(_CHUNK, count - start)
        if calibration == "enumerated":
            bits = torch.arange(start, start + size, dtype=torch.int64)[:, None]
            signs = (((bits >> torch.arange(n, dtype=torch.int64)) & 1) * 2 - 1).to(torch.float64)
        else:
            signs = (torch.randint(2, (size, n), generator=rng, device="cpu") * 2 - 1).to(
                torch.float64
            )
        digest.update(signs.to(torch.int8).numpy().tobytes())
        joint[start : start + size] = signs @ normalized
    # Exact identity appears as the all-positive signs. Use the same reduction
    # for it and for the observation; the guard also includes signed mirrors.
    if calibration == "enumerated":
        joint[-1] = observed
    null = dict(zip(sample["columns"], target))
    null_hash = hashlib.sha256(
        json.dumps(null, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return _output(
        sample,
        observed,
        joint,
        kind="sign",
        model=model,
        calibration=calibration,
        seed=seed,
        alpha=alpha,
        tail=tail,
        orbit_count=orbit,
        work=work,
        comparisons=comparisons,
        resource_plan=plan,
        null=null,
        null_hash=null_hash,
        transformation_hash=digest.hexdigest(),
    )


@resident_cpu
@torch.no_grad()
def mean_permutation_stepdown(
    data,
    columns,
    group,
    *,
    exchangeability_model,
    calibration="enumerated",
    draws=9999,
    seed=1729,
    alpha=0.05,
    tail="two-sided",
    missing="raise",
):
    """Joint maxT stepdown from fixed-size two-group label assignments.

    Joint true-null coordinates must be label-exchangeable in independent rows,
    a stronger condition than equal means. Group0 is the first observed label.
    Enumerates every choose(N,N0) assignment or samples independent uniform
    assignments with a local RNG. RMS normalization is label invariant.
    """
    draws, seed, alpha, model = _options(
        calibration, draws, seed, alpha, tail, exchangeability_model
    )
    sample = _sample(data, columns, missing, group)
    n, m = len(sample["positions"]), len(sample["columns"])
    n1 = sample["codes"].count(0)
    n2 = n - n1
    count, orbit, work, comparisons, plan = _admit(n, m, calibration, draws, "permutation", n1)
    x = _numeric(sample)
    # An invariant half-range anchor avoids loss of tiny differences in a
    # nearly constant column. Computing halves before adding avoids overflow
    # even when the raw range cannot be represented in float64. Subtracting
    # this midpoint has at most half the raw range; close same-sign inputs
    # benefit from exact (Sterbenz) subtraction before maximum rescaling.
    anchor = x.amin(0) / 2 + x.amax(0) / 2
    x = x - anchor
    if not bool(torch.isfinite(x).all()):
        raise AnalysisError("non_finite_values", "Pooled centering exceeded finite float64 range.")
    if bool((x.abs().amax(0) == 0).any()):
        raise AnalysisError(
            "zero_scale", "Every tested coordinate needs positive pooled centered RMS."
        )
    x = _safe_scale(x)
    x = x - x.mean(0)
    rms = torch.sqrt(x.square().mean(0))
    if bool((rms == 0).any()) or not bool(torch.isfinite(rms).all()):
        raise AnalysisError(
            "zero_scale", "Every tested coordinate needs positive pooled centered RMS."
        )
    normalized = x / rms / math.sqrt(1 / n1 + 1 / n2)
    total = normalized.sum(0)
    codes = torch.tensor(sample["codes"], dtype=torch.int64)
    observed = normalized[codes == 0].sum(0) / n1 - normalized[codes == 1].sum(0) / n2
    joint = torch.empty((count, m), dtype=torch.float64)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    digest = hashlib.sha256()
    observed_assignment = [i for i, code in enumerate(sample["codes"]) if code == 0]
    assignments = itertools.combinations(range(n), n1) if calibration == "enumerated" else None
    for start in range(0, count, _CHUNK):
        size = min(_CHUNK, count - start)
        indicators = torch.zeros((size, n), dtype=torch.float64)
        identity_rows = []
        for row in range(size):
            # randperm is discrete-uniform without floating-key tie bias; each
            # new call is independent and only this bounded row is allocated.
            indices = (
                next(assignments)
                if assignments is not None
                else torch.randperm(n, generator=rng, device="cpu")[:n1].tolist()
            )
            indicators[row, list(indices)] = 1
            digest.update(bytes(int(v) for v in indicators[row].tolist()))
            if calibration == "enumerated" and list(indices) == observed_assignment:
                identity_rows.append(row)
        selected_sum = indicators @ normalized
        joint[start : start + size] = selected_sum / n1 - (total - selected_sum) / n2
        for row in identity_rows:
            joint[start + row] = observed
    null_hash = hashlib.sha256(
        json.dumps(
            {"equal_joint_distribution": sample["columns"], "group_counts": [n1, n2]},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return _output(
        sample,
        observed,
        joint,
        kind="permutation",
        model=model,
        calibration=calibration,
        seed=seed,
        alpha=alpha,
        tail=tail,
        orbit_count=orbit,
        work=work,
        comparisons=comparisons,
        resource_plan=plan,
        null=None,
        null_hash=null_hash,
        group=group,
        group_sizes=[n1, n2],
        transformation_hash=digest.hexdigest(),
    )
