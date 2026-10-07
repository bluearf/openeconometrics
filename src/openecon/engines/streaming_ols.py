"""Repeatable, bounded-memory float64 OLS using tall-skinny QR.

Batches are read in three passes: normalization, factorization, and
residual/covariance accumulation. An optional numerical factory serves the first
two passes. Each factory must yield the same finite numerical rows in the same
order. The public analysis layer checks the source digest; this kernel checks
dimensions and observation counts on every pass.

Only one observation block and small factors survive between iterations. No
normal-equations matrix, full Q, prediction vector, or observation index is
materialized. The reduced-factor QR reduction follows a binary TSQR tree:
Demmel et al., https://doi.org/10.1137/080731992.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .contracts import KernelError


Batch = tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Sequence[bytes]]
BatchFactory = Callable[[], Iterable[Batch]]
_MAX_COLUMNS = 1024
_MAX_SAMPLES = 400
_FACTOR_LEVELS = 64


@dataclass
class StreamingOLSResult:
    parameters: Tensor
    covariance: Tensor
    nobs: int
    nparams: int
    residual_ss: float
    total_ss: float
    log_likelihood: float
    condition_number: float
    observed: Tensor
    fitted: Tensor
    residuals: Tensor
    sample_positions: Tensor
    group_count: int | None = None
    solver: str = "torch_tsqr"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class _CompensatedSum:
    """Kahan addition for the fixed-size scalar and covariance accumulators."""

    def __init__(self, shape: tuple[int, ...]):
        self.value = torch.zeros(shape, dtype=torch.float64)
        self.correction = torch.zeros_like(self.value)

    def add(self, increment: Tensor) -> None:
        adjusted = increment - self.correction
        updated = self.value + adjusted
        self.correction = (updated - self.value) - adjusted
        self.value = updated

    def scale(self, multiplier: Tensor) -> None:
        self.value *= multiplier
        self.correction *= multiplier


class _ScaledMoments:
    """Chan-merged moments in maximum-magnitude units, without x-squared overflow."""

    def __init__(self, width: int):
        self.n = 0
        self.magnitude = torch.zeros(width, dtype=torch.float64)
        self.minimum = torch.full((width,), math.inf, dtype=torch.float64)
        self.maximum = torch.full((width,), -math.inf, dtype=torch.float64)
        self.mean = _CompensatedSum((width,))
        self.m2 = _CompensatedSum((width,))

    def add(self, values: Tensor) -> None:
        size = len(values)
        self.minimum = torch.minimum(self.minimum, values.amin(dim=0))
        self.maximum = torch.maximum(self.maximum, values.amax(dim=0))
        magnitude = torch.maximum(self.magnitude, values.abs().amax(dim=0))
        divisor = torch.where(magnitude == 0, torch.ones_like(magnitude), magnitude)
        ratio = self.magnitude / divisor
        self.mean.scale(ratio)
        self.m2.scale(ratio.square())
        normalized = values / divisor
        block_mean = normalized.mean(dim=0)
        block_m2 = (normalized - block_mean).square().sum(dim=0)
        updated_n = self.n + size
        delta = block_mean - self.mean.value
        self.m2.add(block_m2 + delta.square() * ((self.n / updated_n) * size))
        self.mean.add(delta * (size / updated_n))
        self.n = updated_n
        self.magnitude = magnitude


class _TSQRTree:
    """At most 64 small factors; balanced reduction avoids a long error chain."""

    def __init__(self):
        self.levels: list[Tensor | None] = [None] * _FACTOR_LEVELS
        self.depth = self.peak_factors = 0

    def add(self, factor: Tensor) -> None:
        for level, previous in enumerate(self.levels):
            if previous is None:
                self.levels[level] = factor
                self.depth = max(self.depth, level)
                self.peak_factors = max(self.peak_factors,
                                       sum(item is not None for item in self.levels))
                return
            _, factor = torch.linalg.qr(torch.cat((previous, factor), dim=0), mode="r")
            self.levels[level] = None
        raise KernelError("factor_capacity", "The TSQR reduction exceeded its 64-level factor capacity.")

    def finish(self) -> Tensor:
        result: Tensor | None = None
        final_merges = 0
        # Higher levels represent earlier rows. Final reduction retains their
        # positional order; sign choices of R do not affect the OLS solution.
        for factor in reversed(self.levels):
            if factor is not None:
                if result is None:
                    result = factor
                else:
                    _, result = torch.linalg.qr(torch.cat((result, factor), dim=0), mode="r")
                    final_merges += 1
        assert result is not None
        self.depth += final_merges
        return result


def _batches(
    factory: BatchFactory, *, width: int | None = None, intercept: bool, cluster: bool = False
) -> Iterator[Batch]:
    for batch in factory():
        if not isinstance(batch, (tuple, list)) or len(batch) != (3 if cluster else 2):
            if cluster:
                raise KernelError("invalid_clusters", "Cluster streaming batches need (design, outcome, byte_keys).")
            raise KernelError("invalid_design", "Streaming batches must be (design, outcome) pairs.")
        x, y = batch[:2]
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise KernelError("invalid_design", "Streaming batches must contain PyTorch tensors.")
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise KernelError("unsupported_device", "Streaming OLS currently requires CPU float64 tensors.")
        if x.dtype != torch.float64 or y.dtype != torch.float64:
            raise KernelError("invalid_precision", "Streaming OLS requires float64 tensors.")
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or x.shape[1] < 1:
            raise KernelError("invalid_design", "Expected an n-by-k design and an n-element outcome.")
        if x.shape[1] > _MAX_COLUMNS:
            raise KernelError("design_too_wide", "Streaming OLS supports at most 1024 design columns.")
        if width is not None and x.shape[1] != width:
            raise KernelError("source_changed", "The streaming design width changed between batches or passes.")
        if cluster:
            keys = batch[2]
            if (not isinstance(keys, Sequence) or isinstance(keys, (str, bytes))
                    or len(keys) != len(y) or any(not isinstance(key, bytes) for key in keys)):
                raise KernelError("invalid_clusters", "Cluster covariance needs one canonical byte key per observation.")
        if len(x) == 0:
            continue
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise KernelError("non_finite_values", "Streaming kernel inputs must be finite.")
        if intercept and not bool(torch.all(x[:, 0] == 1)):
            raise KernelError("invalid_design", "An intercept design must have a leading column of ones.")
        yield (x, y, keys) if cluster else (x, y)


def _normalize(values: Tensor, magnitude: Tensor, centers: Tensor, rms: Tensor) -> Tensor:
    # Subtract in original units when representable, retaining precision around
    # large offsets. The scaled difference handles finite inputs whose direct
    # subtraction overflows (e.g. mixed-sign values close to float64 limits).
    difference = values - centers * magnitude
    scaled = torch.where(torch.isfinite(difference), difference / magnitude,
                         values / magnitude - centers)
    normalized = scaled / rms
    if not bool(torch.isfinite(normalized).all()):
        raise KernelError("numerical_failure", "Normalization exceeded finite float64 precision.")
    return normalized


def _same_count(count: int, expected: int) -> None:
    if count != expected:
        raise KernelError("source_changed", "The streaming observation count changed between passes.")


def solve_ols(
    batch_factory: BatchFactory,
    covariance: str = "HC3",
    *,
    intercept: bool = True,
    sample_limit: int = _MAX_SAMPLES,
    scratch_directory: Path | None = None,
    numeric_batch_factory: BatchFactory | None = None,
) -> StreamingOLSResult:
    """Fit repeatable numerical batches without retaining all observations.

    ``x`` includes a leading column of ones when ``intercept=True``. Prediction
    samples contain the first at most 400 fitted observations, independently of
    source batch size. Missing values must be handled before this boundary.
    Cluster covariance requires canonical byte keys in each batch and spills
    group scores to owned temporary storage. CR1 uses the observed group count.
    An optional numeric factory supplies (design, outcome) pairs for the first
    two passes, so cluster labels need encoding only for covariance accumulation.
    Its rows must match the main factory; the caller verifies source digests.
    """
    if covariance not in {"nonrobust", "HC1", "HC3", "cluster"}:
        raise KernelError("unsupported_covariance", "Streaming OLS supports nonrobust, HC1, HC3, and one-way cluster covariance.")
    if not callable(batch_factory):
        raise KernelError("invalid_design", "Streaming OLS requires a repeatable batch factory.")
    if numeric_batch_factory is not None and not callable(numeric_batch_factory):
        raise KernelError("invalid_design", "The numeric batch factory must be callable.")
    if not isinstance(intercept, bool):
        raise KernelError("invalid_design", "intercept must be a boolean.")
    if (not isinstance(sample_limit, int) or isinstance(sample_limit, bool)
            or not 0 <= sample_limit <= _MAX_SAMPLES):
        raise KernelError("invalid_solver_options", "sample_limit must be an integer from 0 to 400.")
    clustered = covariance == "cluster"
    numerical_factory = batch_factory if numeric_batch_factory is None else numeric_batch_factory
    numerical_clusters = clustered and numeric_batch_factory is None
    with torch.inference_mode(), ExitStack() as resources:
        moments: _ScaledMoments | None = None
        outcome_moments = _ScaledMoments(1)
        width = None
        max_batch_rows = max_batch_bytes = 0
        for batch in _batches(numerical_factory, intercept=intercept, cluster=numerical_clusters):
            x, y = batch[:2]
            if moments is None:
                width = x.shape[1]
                moments = _ScaledMoments(width)
            if x.shape[1] != width:
                raise KernelError("source_changed", "The streaming design width changed between batches.")
            moments.add(x)
            outcome_moments.add(y[:, None])
            max_batch_rows = max(max_batch_rows, len(x))
            max_batch_bytes = max(max_batch_bytes, (x.numel() + y.numel()) * 8)
        if moments is None or width is None or moments.n <= width:
            raise KernelError("insufficient_observations", "Estimation requires more observations than design columns.")
        n, k = moments.n, width
        magnitude = moments.magnitude
        predictor_start = int(intercept)
        if bool(torch.any(moments.minimum[predictor_start:] == moments.maximum[predictor_start:])):
            raise KernelError("constant_predictor", "A numeric predictor is constant; use the intercept option instead.")
        if bool(torch.any(magnitude == 0)):
            raise KernelError("singular_design", "The design contains an all-zero column.")
        centers = moments.mean.value.clone() if intercept else torch.zeros_like(magnitude)
        if intercept:
            centers[0] = 0
        variance = moments.m2.value / n
        rms = variance.sqrt() if intercept else (variance + moments.mean.value.square()).sqrt()
        if intercept:
            rms[0] = 1
        if bool(torch.any(rms <= 0)) or not bool(torch.isfinite(rms).all()):
            raise KernelError("singular_design", "The design contains an all-zero or constant predictor.")
        if float(outcome_moments.m2.value[0]) <= 0:
            raise KernelError("constant_outcome", "The outcome must contain at least two distinct values.")
        outcome_magnitude = outcome_moments.magnitude
        outcome_center = outcome_moments.mean.value if intercept else torch.zeros(1, dtype=torch.float64)
        unit = torch.ones(1, dtype=torch.float64)
        transform = torch.diag(1 / magnitude / rms)
        if intercept:
            transform[0, 1:] = -centers[1:] / rms[1:]
        if not bool(torch.isfinite(transform).all()):
            raise KernelError("numerical_failure", "Predictor units exceed float64 coefficient precision; rescale the input.")

        tree = _TSQRTree()
        count = block_count = 0
        for batch in _batches(numerical_factory, width=k, intercept=intercept, cluster=numerical_clusters):
            x, y = batch[:2]
            z = _normalize(x, magnitude, centers, rms)
            normalized_y = _normalize(y[:, None], outcome_magnitude, outcome_center, unit)[:, 0]
            augmented = torch.cat((z, normalized_y[:, None]), dim=1)
            _, block_r = torch.linalg.qr(augmented, mode="r")
            tree.add(block_r)
            count += len(x)
            block_count += 1
        _same_count(count, n)
        augmented_r = tree.finish()
        r = augmented_r[:k, :k]
        singular = torch.linalg.svdvals(r)
        # Rank is diagnosed on the small normalized factor, independently of N.
        # Scaling the cutoff by observation count would reject legitimate
        # moderately conditioned designs simply because their source is large.
        rank_limit = torch.finfo(torch.float64).eps * k * max(1, tree.depth) * singular[0]
        if not bool(torch.isfinite(singular).all()) or bool(singular[-1] <= rank_limit):
            raise KernelError("singular_design", "The normalized design matrix is rank deficient.")
        theta_scaled = torch.linalg.solve_triangular(r, augmented_r[:k, k, None], upper=True)[:, 0]
        theta = theta_scaled * outcome_magnitude[0]
        if intercept:
            theta[0] += outcome_center[0] * outcome_magnitude[0]
        parameters = transform @ theta
        inverse_r = torch.linalg.solve_triangular(r, torch.eye(k, dtype=torch.float64), upper=True)

        rss = _CompensatedSum(())
        total = _CompensatedSum(())
        meat = _CompensatedSum((k, k))
        cluster_accumulator = None
        if clustered:
            from .streaming_groups import ClusterAccumulator
            cluster_accumulator = resources.enter_context(ClusterAccumulator(k, scratch_directory))
        sample_count = min(n, sample_limit)
        observed = torch.empty(sample_count, dtype=torch.float64)
        fitted_samples = torch.empty_like(observed)
        residual_samples = torch.empty_like(observed)
        count = sampled = 0
        max_leverage = 0.0
        for batch in _batches(batch_factory, width=k, intercept=intercept, cluster=clustered):
            x, y = batch[:2]
            z = _normalize(x, magnitude, centers, rms)
            centered_y = y - outcome_center[0] * outcome_magnitude[0]
            residual = centered_y - z @ (theta_scaled * outcome_magnitude[0])
            if not bool(torch.isfinite(residual).all()):
                raise KernelError("numerical_failure", "Residuals exceeded finite float64 precision.")
            rss.add(torch.dot(residual, residual))
            total.add(torch.dot(centered_y, centered_y))
            if covariance != "nonrobust":
                q = torch.linalg.solve_triangular(r.T, z.T, upper=False).T
                adjusted_residual = residual
                if covariance == "HC3":
                    leverage = q.square().sum(dim=1)
                    complement = 1 - leverage
                    if bool(torch.any(complement <= 100 * torch.finfo(torch.float64).eps)):
                        raise KernelError("undefined_hc3", "HC3 is undefined for observations with unit leverage.")
                    adjusted_residual = residual / complement
                    max_leverage = max(max_leverage, float(leverage.max()))
                scores = q * adjusted_residual[:, None]
                if clustered:
                    cluster_accumulator.add(batch[2], scores)
                else:
                    meat.add(scores.T @ scores)
            take = min(sample_count - sampled, len(y))
            if take:
                observed[sampled:sampled + take] = y[:take]
                residual_samples[sampled:sampled + take] = residual[:take]
                fitted_samples[sampled:sampled + take] = y[:take] - residual[:take]
                sampled += take
            count += len(x)
        _same_count(count, n)
        residual_ss = float(rss.value)
        total_ss = float(total.value)
        if not math.isfinite(residual_ss) or residual_ss <= 0 or not math.isfinite(total_ss):
            raise KernelError("numerical_failure", "Residual variance or outcome variance is outside finite float64 range.")
        group_count = None
        cluster_diagnostics = {}
        if clustered:
            cluster_meat, group_count = cluster_accumulator.finish()
            meat.value = cluster_meat
            cluster_diagnostics = cluster_accumulator.diagnostics
        if covariance == "nonrobust":
            normalized_covariance = (inverse_r @ inverse_r.T) * (residual_ss / (n - k))
        else:
            normalized_covariance = inverse_r @ meat.value @ inverse_r.T
            if covariance == "HC1":
                normalized_covariance *= n / (n - k)
            elif clustered:
                normalized_covariance *= group_count / (group_count - 1) * (n - 1) / (n - k)
        variance = transform @ normalized_covariance @ transform.T
        variance = (variance + variance.T) / 2
        if not bool(torch.isfinite(parameters).all()) or not bool(torch.isfinite(variance).all()):
            raise KernelError("numerical_failure", "Estimation produced non-finite coefficients or covariance.")
        log_likelihood = -0.5 * n * (math.log(2 * math.pi) + 1 + math.log(residual_ss / n))
        diagnostics = {
            "engine": "torch", "device": "cpu", "dtype": "float64", "autograd": False,
            "torch_version": torch.__version__, "passes": 3, "factorization_blocks": block_count,
            "numeric_batch_factory_used": numeric_batch_factory is not None,
            "condition_number_basis": "centered_and_scaled_design" if intercept else "scaled_design",
            "column_centers": (centers * magnitude).tolist(), "column_scales": (magnitude * rms).tolist(),
            "rank_diagnostic": "SVD of normalized TSQR factor",
            "rank_tolerance": "eps * design_columns * max(1, reduction_depth) * largest_singular_value",
            "rank_tolerance_value": float(rank_limit),
            "rank_tolerance_interpretation": "conservative small-factor numerical-rank heuristic",
            "tsqr_tree_depth": tree.depth, "tsqr_peak_retained_factors": tree.peak_factors,
            "tsqr_factor_capacity": _FACTOR_LEVELS,
            "dense_observation_matrix": False, "retains_full_q": False,
            "retained_factor_shape": list(augmented_r.shape),
            "maximum_batch_rows": max_batch_rows, "maximum_input_batch_bytes": max_batch_bytes,
            "accumulation": "scaled Chan moments; Kahan residual, total and covariance sums",
            "prediction_sample": "first positional fitted observations", "prediction_limit": sample_limit,
            "memory_complexity": "O(batch_rows * design_columns + design_columns ** 2)",
        }
        if clustered:
            diagnostics.update(cluster_diagnostics)
            diagnostics["cluster_count"] = group_count
            diagnostics["small_sample_correction"] = group_count / (group_count - 1) * (n - 1) / (n - k)
            diagnostics["memory_complexity"] = "O(batch_rows * design_columns + design_columns ** 2 + fixed cluster cache)"
        if covariance == "HC3":
            diagnostics["max_leverage"] = max_leverage
        return StreamingOLSResult(
            parameters=parameters, covariance=variance, nobs=n, nparams=k,
            residual_ss=residual_ss, total_ss=total_ss, log_likelihood=log_likelihood,
            condition_number=float(singular[0] / singular[-1]), observed=observed,
            fitted=fitted_samples, residuals=residual_samples,
            sample_positions=torch.arange(sample_count, dtype=torch.int64), group_count=group_count, diagnostics=diagnostics,
        )
