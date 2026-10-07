"""Global Moran's I with fixed-W normality/randomization moments and permutation."""

from __future__ import annotations

import math

import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.resources import plan_workspace

from .weights import SpatialWeights, _keys


def moran(
    data,
    outcome,
    *,
    key,
    spatial_weights,
    missing="raise",
    null="randomization",
    alternative="two-sided",
    permutations=0,
    seed=0,
    max_permutation_work=50_000_000,
):
    """Global Moran's I, not network centrality or a regression residual test.

    N includes explicitly retained zero-row isolates. Under randomization all
    observed values are exchangeable across all retained unit keys. The analytic
    p-value is a normal approximation; optional Monte Carlo uses inclusive tails
    and (extreme+1)/(B+1), centered on E[I] for two-sided alternatives.
    """
    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported",
            "Moran's I has no Dataset route; no implicit materialization is performed.",
        )
    if (
        missing not in {"raise", "drop"}
        or null not in {"randomization", "normality"}
        or alternative not in {"greater", "less", "two-sided"}
    ):
        raise AnalysisError(
            "invalid_spec",
            "Choose missing='raise'/'drop', null='randomization'/'normality', and greater/less/two-sided alternative.",
        )
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, int)
        or not 0 <= permutations <= 100_000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
        or isinstance(max_permutation_work, bool)
        or not isinstance(max_permutation_work, int)
        or max_permutation_work < 1
    ):
        raise AnalysisError(
            "invalid_spec",
            "permutations must be an integer 0..100000, seed 0..2^63-1 and max_permutation_work a positive integer.",
        )
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    if key == outcome or any(column not in frame.columns for column in (outcome, key)):
        raise AnalysisError(
            "missing_columns", "Provide distinct existing outcome and spatial key columns."
        )
    weights = SpatialWeights.from_payload(spatial_weights)
    plan = plan_workspace(
        "sparse Moran moments and streaming permutations",
        {
            "selected_input_and_sample": 2
            * sum(int(frame[name].memory_usage(index=False, deep=True)) for name in (outcome, key))
            + 32 * len(frame),
            "COO_vectors_permutation_and_moments": 8 * (16 * len(frame) + 8 * weights.nnz),
        },
    )
    weights.align(_keys(frame[key].tolist()))
    missing_rows = frame[outcome].isna()
    if bool(missing_rows.any()) and missing == "raise":
        raise AnalysisError(
            "missing_values",
            "Moran's inputs contain missing values; choose missing='drop' explicitly.",
        )
    positions = (~missing_rows).to_numpy().nonzero()[0].tolist()
    selected = frame.loc[~missing_rows, [outcome, key]].reset_index(drop=True)
    weights = weights.align(_keys(selected[key].tolist()), subset=True)
    n, nnz = weights.n, weights.nnz
    if n < 4:
        raise AnalysisError(
            "insufficient_observations",
            "Moran inference requires at least four complete spatial units.",
        )
    if permutations * (n + nnz) > max_permutation_work:
        raise AnalysisError(
            "spatial_permutation_work_limit",
            "Requested permutations exceed max_permutation_work=B*(N+nnz); increase the explicit budget or reduce permutations.",
        )
    x = _numeric(selected[outcome], outcome)
    # Moran is invariant to positive scaling. Bound moments before centering so
    # finite extreme inputs cannot overflow fourth moments or squared deviations.
    scale = float(x.abs().max())
    x = x / scale if scale else x
    z = x - x.mean()
    denominator = z.square().sum()
    if float(denominator) <= 0:
        raise AnalysisError(
            "moran_constant_outcome", "Moran's I is undefined for a constant outcome."
        )
    rows, cols = (
        torch.tensor(weights.rows, dtype=torch.int64),
        torch.tensor(weights.cols, dtype=torch.int64),
    )
    values = torch.tensor(weights.values, dtype=torch.float64)
    s0 = float(values.sum())
    if not math.isfinite(s0) or s0 <= 0:
        raise AnalysisError(
            "invalid_spatial_weights", "Moran's I requires positive finite total spatial weight."
        )
    # The moment formula squares S0. Reject an unsafe scale explicitly; the
    # user can choose row normalization without changing Moran under global rescaling.
    if s0 > 1e150 or s0 < 1e-150:
        raise AnalysisError(
            "moran_weight_scale",
            "Moran inference requires total weight between 1e-150 and 1e150; rescale W explicitly.",
        )
    edge_map = dict(zip(zip(weights.rows, weights.cols, strict=True), weights.values, strict=True))
    s1 = sum(
        value * value + value * edge_map.get((j, i), 0.0) for (i, j), value in edge_map.items()
    )
    row_sums, col_sums = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
    row_sums.index_add_(0, rows, values)
    col_sums.index_add_(0, cols, values)
    s2 = float((row_sums + col_sums).square().sum())
    expected = -1 / (n - 1)
    kurtosis = float(n * z.pow(4).sum() / denominator.square())
    if null == "normality":
        variance = (n * n * s1 - n * s2 + 3 * s0**2) / ((n * n - 1) * s0**2) - expected**2
    else:
        numerator = n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0**2) - kurtosis * (
            (n * n - n) * s1 - 2 * n * s2 + 6 * s0**2
        )
        variance = numerator / ((n - 1) * (n - 2) * (n - 3) * s0**2) - expected**2
    if not math.isfinite(variance) or variance <= 1e-15:
        raise AnalysisError(
            "moran_degenerate_variance",
            "Moran's null variance is zero, negative or numerically undefined for these weights/values.",
        )

    def statistic(vector):
        return float(n / s0 * (values * vector[rows] * vector[cols]).sum() / denominator)

    observed = statistic(z)
    zscore = (observed - expected) / math.sqrt(variance)
    p_value = (
        math.erfc(abs(zscore) / math.sqrt(2))
        if alternative == "two-sided"
        else 0.5 * math.erfc((zscore if alternative == "greater" else -zscore) / math.sqrt(2))
    )
    permutation = None
    if permutations:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        count, total, squares = 0, 0.0, 0.0
        threshold = abs(observed - expected) if alternative == "two-sided" else observed
        for _ in range(permutations):
            draw = statistic(z[torch.randperm(n, generator=generator)])
            extreme = (
                abs(draw - expected) >= threshold - 1e-12
                if alternative == "two-sided"
                else draw >= threshold - 1e-12
                if alternative == "greater"
                else draw <= threshold + 1e-12
            )
            count += int(extreme)
            total += draw
            squares += draw * draw
        permutation = {
            "p_value": (count + 1) / (permutations + 1),
            "extreme_count": count,
            "permutations": permutations,
            "seed": seed,
            "inclusive_ties": True,
            "two_sided_center": expected,
            "null": "random reassignment of all observed values to fixed unit keys",
            "mean": total / permutations,
            "variance_population": max(0.0, squares / permutations - (total / permutations) ** 2),
            "work": permutations * (n + nnz),
            "max_permutation_work": max_permutation_work,
        }
    return {
        "statistic": observed,
        "expected": expected,
        "variance": variance,
        "z": zscore,
        "p_value": p_value,
        "p_value_method": "normal approximation using specified null moments",
        "null": null,
        "alternative": alternative,
        "nobs": n,
        "nobs_original": len(frame),
        "dropped_rows": int(missing_rows.sum()),
        "sample_positions": positions,
        "sample_keys": list(weights.keys),
        "isolate_n_policy": "retained zero rows count in N; no adjust.n reduction",
        "moments": {"S0": s0, "S1": s1, "S2": s2, "kurtosis": kurtosis},
        "permutation": permutation,
        "spatial_summary": weights.summary(),
        "provenance": {
            "backend": "openecon.torch",
            "precision": "float64",
            "resource_plan": plan.record(),
            "sample_hash": _frame_hasher(selected).hexdigest(),
            "weights_hash": weights.summary()["sha256"],
            "regression_residual_test": False,
        },
    }
