"""Replayable, bounded-memory float64 binary maximum likelihood.

Likelihood, observed information and score use the same tensor formulas as the
dense engine. Every Newton and line-search pass reads one observation block at a
time. Separation uses globally verified directions and a bounded native Torch cutting
plane diagnostic; an inconclusive check fails instead of accepting a fit.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .contracts import KernelError
from .streaming_ols import _CompensatedSum, _ScaledMoments, _TSQRTree, _normalize, _same_count
from .torch_engine import _binary_log_likelihood, _binary_terms

BatchFactory = Callable[[], Iterable[tuple]]
_MAX_COLUMNS = 1024
_MAX_SAMPLES = 400


@dataclass
class StreamingBinaryResult:
    parameters: Tensor
    covariance: Tensor
    nobs: int
    nparams: int
    residual_ss: float
    total_ss: float
    log_likelihood: float
    null_log_likelihood: float
    condition_number: float
    observed: Tensor
    fitted: Tensor
    residuals: Tensor
    sample_positions: Tensor
    iterations: int
    solver: str = "torch_streaming_newton_cholesky_backtracking"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    group_count: int | None = None


def _batches(factory: BatchFactory, *, width: int | None, intercept: bool,
             cluster: bool) -> Iterator[tuple[Tensor, Tensor, Sequence[bytes] | None]]:
    for batch in factory():
        if not isinstance(batch, (tuple, list)) or len(batch) not in {2, 3}:
            raise KernelError("invalid_design", "Streaming batches must contain design, outcome, and optional cluster keys.")
        x, y = batch[:2]
        keys = batch[2] if len(batch) == 3 else None
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise KernelError("invalid_design", "Streaming batches must contain PyTorch tensors.")
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise KernelError("unsupported_device", "Streaming binary models require CPU float64 tensors.")
        if x.dtype != torch.float64 or y.dtype != torch.float64:
            raise KernelError("invalid_precision", "Streaming binary models require float64 tensors.")
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or x.shape[1] < 1:
            raise KernelError("invalid_design", "Expected an n-by-k design and an n-element outcome.")
        if x.shape[1] > _MAX_COLUMNS:
            raise KernelError("design_too_wide", "Streaming binary models support at most 1024 design columns.")
        if width is not None and x.shape[1] != width:
            raise KernelError("source_changed", "The streaming design width changed between batches or passes.")
        if keys is not None or cluster:
            if (not isinstance(keys, Sequence) or isinstance(keys, (bytes, str))
                    or len(keys) != len(x) or any(not isinstance(key, bytes) for key in keys)):
                raise KernelError("invalid_clusters", "Cluster covariance requires one encoded bytes key per observation.")
        if not len(x):
            continue
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise KernelError("non_finite_values", "Streaming kernel inputs must be finite.")
        if not bool(((y == 0) | (y == 1)).all()):
            raise KernelError("invalid_binary_outcome", "Binary models require 0 and 1 outcomes.")
        if intercept and not bool(torch.all(x[:, 0] == 1)):
            raise KernelError("invalid_design", "An intercept design must have a leading column of ones.")
        yield x, y, keys


def _check_separation(replay: Callable, objective: Tensor, width: int) -> dict[str, Any]:
    """Native box LP with independently checked bounds and full-source witnesses."""
    from .separation import certify_separation

    def signed_rows():
        for z, y, _ in replay():
            yield (2 * y - 1)[:, None] * z

    return certify_separation(signed_rows, objective, width)


def solve_binary(estimator: str, batch_factory: BatchFactory, covariance: str = "nonrobust", *,
                 intercept: bool = True, sample_limit: int = _MAX_SAMPLES,
                 max_iter: int = 100, tolerance: float = 1e-10,
                 scratch_directory: str | Path | None = None,
                 numeric_batch_factory: BatchFactory | None = None) -> StreamingBinaryResult:
    """Fit logit/probit from repeatable CPU float64 blocks; retain at most 400 rows.

    ``x`` already contains the intercept and encoded categorical columns. Cluster
    labels cross this boundary as stable bytes keys, with external spill storage.
    The caller owns source-content hashing; this kernel verifies shapes/counts on
    every replay. No full observation matrix, row index or prediction is retained.
    ``numeric_batch_factory`` may omit cluster-key encoding before the final
    covariance pass. It must replay exactly the same projected observations and
    source hashes; the original factory supplies the final keys and scores.
    """
    if estimator not in {"logit", "probit"}:
        raise KernelError("unsupported_estimator", "Streaming binary models support logit and probit.")
    if covariance not in {"nonrobust", "cluster"}:
        raise KernelError("unsupported_covariance", f"{estimator} supports nonrobust and cluster covariance.")
    if (not callable(batch_factory) or not isinstance(intercept, bool)
            or (numeric_batch_factory is not None and not callable(numeric_batch_factory))):
        raise KernelError("invalid_design", "A repeatable batch factory and boolean intercept are required.")
    if (not isinstance(sample_limit, int) or isinstance(sample_limit, bool)
            or not 0 <= sample_limit <= _MAX_SAMPLES):
        raise KernelError("invalid_solver_options", "sample_limit must be an integer from 0 to 400.")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise KernelError("invalid_solver_options", "max_iter must be a positive integer.")
    if (not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool)
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise KernelError("invalid_solver_options", "tolerance must be positive and finite.")
    try:
        with torch.inference_mode():
            return _solve(estimator, batch_factory, covariance, intercept, sample_limit, max_iter, tolerance,
                          scratch_directory, numeric_batch_factory)
    except KernelError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        # Replay callbacks carry source/schema errors using the shared code
        # contract. Keep their identity without importing client-layer classes.
        if isinstance(getattr(error, "code", None), str):
            raise
        raise KernelError("numerical_failure", f"PyTorch could not complete streaming float64 estimation: {error}") from error


def _solve(estimator: str, factory: BatchFactory, covariance: str, intercept: bool,
           sample_limit: int, max_iter: int, tolerance: float,
           scratch_directory: str | Path | None, numeric_batch_factory: BatchFactory | None) -> StreamingBinaryResult:
    cluster = covariance == "cluster"
    numeric_factory = factory if numeric_batch_factory is None else numeric_batch_factory
    numeric_cluster = cluster and numeric_batch_factory is None
    moments = None
    width = None
    positives = 0
    maximum_rows = maximum_bytes = 0
    for x, y, _ in _batches(numeric_factory, width=None, intercept=intercept, cluster=numeric_cluster):
        if moments is None:
            width = x.shape[1]
            moments = _ScaledMoments(width)
        if width != x.shape[1]:
            raise KernelError("source_changed", "The streaming design width changed between batches.")
        moments.add(x)
        positives += int(torch.count_nonzero(y))
        maximum_rows = max(maximum_rows, len(x))
        maximum_bytes = max(maximum_bytes, 8 * (x.numel() + y.numel()))
    if moments is None or width is None or moments.n <= width:
        raise KernelError("insufficient_observations", "Estimation requires more observations than design columns.")
    n, k = moments.n, width
    negatives = n - positives
    if not positives or not negatives:
        raise KernelError("invalid_binary_outcome", "Binary models require both 0 and 1 outcomes.")
    magnitude = moments.magnitude
    start = int(intercept)
    if bool(torch.any(moments.minimum[start:] == moments.maximum[start:])):
        raise KernelError("constant_predictor", "A predictor is constant; use the intercept option instead.")
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
        raise KernelError("singular_design", "The normalized design contains an all-zero or constant column.")
    transform = torch.diag(1 / magnitude / rms)
    if intercept:
        transform[0, 1:] = -centers[1:] / rms[1:]
    if not bool(torch.isfinite(transform).all()):
        raise KernelError("numerical_failure", "Predictor units exceed float64 coefficient precision; rescale the input.")
    passes = 1

    def replay(*, covariance_pass: bool = False):
        nonlocal passes, maximum_rows, maximum_bytes
        passes += 1
        count = 0
        current_factory = factory if covariance_pass else numeric_factory
        current_cluster = cluster if covariance_pass else numeric_cluster
        for x, y, keys in _batches(current_factory, width=k, intercept=intercept, cluster=current_cluster):
            maximum_rows = max(maximum_rows, len(x))
            maximum_bytes = max(maximum_bytes, 8 * (x.numel() + y.numel()))
            yield _normalize(x, magnitude, centers, rms), y, keys
            count += len(y)
        _same_count(count, n)

    tree = _TSQRTree()
    objective_sum = _CompensatedSum((k,))
    blocks = 0
    for z, y, _ in replay():
        _, factor = torch.linalg.qr(z, mode="r")
        tree.add(factor)
        objective_sum.add(z.T @ (2 * y - 1))
        blocks += 1
    r = tree.finish()
    singular = torch.linalg.svdvals(r)
    rank_limit = torch.finfo(torch.float64).eps * k * max(1, tree.depth) * singular[0]
    if not bool(torch.isfinite(singular).all()) or bool(singular[-1] <= rank_limit):
        raise KernelError("singular_design", "The normalized design matrix is rank deficient.")
    objective = objective_sum.value / n

    def evaluate(theta: Tensor, *, direction: Tensor | None = None):
        likelihood = _CompensatedSum(())
        separating = direction is not None and bool(torch.any(direction != 0))
        positive_margin = False
        if separating:
            direction = direction / direction.abs().max()
        for z, y, _ in replay():
            signed = 2 * y - 1
            likelihood.add(_binary_log_likelihood(torch, estimator, signed, z, theta))
            if separating:
                margin = signed * (z @ direction)
                separating = bool(torch.all(margin >= 0))
                positive_margin = positive_margin or bool(torch.any(margin > 1e-8))
        if separating and positive_margin and float(torch.dot(objective, direction)) > 1e-8:
            raise KernelError("separation_detected", "Complete or quasi-complete separation was detected; no finite unpenalized estimate exists.")
        return float(likelihood.value)

    theta = torch.zeros(k, dtype=torch.float64)
    if intercept:
        if estimator == "logit":
            theta[0] = math.log(positives) - math.log(negatives)
        else:
            tail = torch.tensor(min(positives, negatives) / n, dtype=torch.float64)
            theta[0] = torch.special.ndtri(tail) * (1 if positives <= negatives else -1)
    converged = False
    backtracks = 0
    eps = torch.finfo(torch.float64).eps
    for iteration in range(1, max_iter + 1):
        likelihood = _CompensatedSum(())
        gradient_sum = _CompensatedSum((k,))
        information_sum = _CompensatedSum((k, k))
        for z, y, _ in replay():
            ll, score, weight, _ = _binary_terms(torch, estimator, z, y, theta)
            if not all(bool(torch.isfinite(value).all()) for value in (ll, score, weight)) or bool(torch.any(weight < 0)):
                raise KernelError("numerical_failure", "The binary likelihood produced invalid information or scores.")
            likelihood.add(ll)
            gradient_sum.add(z.T @ score)
            information_sum.add(z.T @ (z * weight[:, None]))
        information = (information_sum.value + information_sum.value.T) / 2
        chol, status = torch.linalg.cholesky_ex(information, check_errors=False)
        if int(status) != 0 or not bool(torch.isfinite(chol).all()):
            _check_separation(replay, objective, k)
            raise KernelError("singular_information", "Observed information is not positive definite; no finite stable estimate is available.")
        gradient = gradient_sum.value
        step = torch.cholesky_solve(gradient[:, None], chol)[:, 0]
        if not bool(torch.isfinite(step).all()):
            raise KernelError("numerical_failure", "The binary Newton step exceeded float64 precision.")
        gradient_norm = float(gradient.abs().max()) / n
        step_norm = float(step.abs().max())
        if gradient_norm <= tolerance and step_norm <= tolerance * (1 + float(theta.abs().max())):
            converged = True
            break
        ll_value = float(likelihood.value)
        directional = float(torch.dot(gradient, step))
        allowance = 8 * eps * max(1.0, abs(ll_value))
        fraction = 1.0
        for _ in range(50):
            candidate = theta + fraction * step
            next_ll = evaluate(candidate, direction=step)
            if math.isfinite(next_ll) and next_ll >= ll_value + 1e-4 * fraction * directional - allowance:
                theta = candidate
                break
            fraction *= 0.5
            backtracks += 1
        else:
            _check_separation(replay, objective, k)
            raise KernelError("nonconvergence", "The binary likelihood line search could not find a stable improving step.")
    if not converged:
        _check_separation(replay, objective, k)
        raise KernelError("nonconvergence", f"The {estimator} solver did not converge in {max_iter} iterations.")
    separation = _check_separation(replay, objective, k)
    bread = torch.cholesky_solve(torch.eye(k, dtype=torch.float64), chol)
    sample_count = min(n, sample_limit)
    observed = torch.empty(sample_count, dtype=torch.float64)
    fitted = torch.empty_like(observed)
    residuals = torch.empty_like(observed)
    rss = _CompensatedSum(())
    sampled = 0
    group_count = None
    if cluster:
        from .streaming_groups import ClusterAccumulator
        accumulator = ClusterAccumulator(k, scratch_directory=scratch_directory)
    else:
        accumulator = None
    with accumulator if accumulator is not None else nullcontext():
        for z, y, keys in replay(covariance_pass=True):
            _, score, _, probability = _binary_terms(torch, estimator, z, y, theta)
            residual = y - probability
            rss.add(torch.dot(residual, residual))
            if accumulator is not None:
                accumulator.add(keys, z * score[:, None])
            take = min(sample_count - sampled, len(y))
            if take:
                observed[sampled:sampled + take] = y[:take]
                fitted[sampled:sampled + take] = probability[:take]
                residuals[sampled:sampled + take] = residual[:take]
                sampled += take
        normalized_covariance = bread
        cluster_diagnostics = {}
        if accumulator is not None:
            meat, group_count = accumulator.finish()
            correction = group_count / (group_count - 1) * (n - 1) / (n - k)
            normalized_covariance = bread @ meat @ bread * correction
            cluster_diagnostics = dict(accumulator.diagnostics)
            cluster_diagnostics.update({"cluster_count": group_count, "small_sample_correction": correction})
    parameters = transform @ theta
    covariance_matrix = transform @ normalized_covariance @ transform.T
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2
    if not all(bool(torch.isfinite(value).all()) for value in (parameters, covariance_matrix, fitted, residuals)):
        raise KernelError("numerical_failure", "Estimation produced non-finite coefficients, covariance or predictions.")
    minority = min(positives, negatives)
    minority_probability = minority / n
    null_ll = minority * math.log(minority_probability) + (n - minority) * math.log1p(-minority_probability)
    diagnostics = {
        "engine": "torch", "device": "cpu", "dtype": "float64", "autograd": False,
        "torch_version": torch.__version__, "passes": passes, "factorization_blocks": blocks,
        "condition_number_basis": "centered_and_scaled_design" if intercept else "scaled_design",
        "column_centers": (centers * magnitude).tolist(), "column_scales": (magnitude * rms).tolist(),
        "rank_diagnostic": "SVD of normalized TSQR factor",
        "rank_tolerance": "eps * design_columns * max(1, reduction_depth) * largest_singular_value",
        "rank_tolerance_value": float(rank_limit), "tsqr_tree_depth": tree.depth,
        "tsqr_peak_retained_factors": tree.peak_factors, "tsqr_factor_capacity": 64,
        "dense_observation_matrix": False, "retains_full_q": False, "retained_factor_shape": list(r.shape),
        "maximum_batch_rows": maximum_rows, "maximum_input_batch_bytes": maximum_bytes,
        "accumulation": "scaled Chan moments; Kahan likelihood, score, information and residual sums",
        "prediction_sample": "first positional fitted observations", "prediction_limit": sample_limit,
        "memory_complexity": "O(batch_rows * design_columns + design_columns ** 2)",
        "converged": True, "gradient_max_per_observation": gradient_norm,
        "newton_step_max": step_norm, "line_search_backtracks": backtracks,
        "convergence_rule": "normalized_score_and_relative_newton_step",
        **separation, **cluster_diagnostics,
    }
    return StreamingBinaryResult(
        parameters=parameters, covariance=covariance_matrix, nobs=n, nparams=k,
        residual_ss=float(rss.value), total_ss=positives * negatives / n,
        log_likelihood=float(likelihood.value), null_log_likelihood=null_ll,
        condition_number=float(singular[0] / singular[-1]), observed=observed,
        fitted=fitted, residuals=residuals, sample_positions=torch.arange(sample_count, dtype=torch.int64),
        iterations=iteration, diagnostics=diagnostics, group_count=group_count,
    )
