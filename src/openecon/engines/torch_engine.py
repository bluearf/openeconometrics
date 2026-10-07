"""OpenEconometrics's single float64 tensor estimation core.

The statistical algorithms are implemented here; PyTorch supplies tensor kernels.
Inputs cross to the selected device at entry. Iteration checks synchronize only
scalar diagnostics; internal results retain tensors for the analysis layer.
No autograd graph, compilation warm-up, or reduced-precision fallback is used.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch

from .contracts import KernelError, KernelResult


def _resolve_device(torch: Any, device: str) -> Any:
    if device.startswith("mps"):
        raise KernelError(
            "unsupported_device",
            "MPS does not support the float64 precision required by OpenEconometrics; select CPU or CUDA.",
        )
    if device != "cpu" and device != "cuda" and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise KernelError("unsupported_device", "PyTorch devices must be cpu, cuda, or cuda:<index>.")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise KernelError("device_unavailable", "CUDA was requested but no CUDA device is available.")
        if ":" in device and int(device.split(":")[1]) >= torch.cuda.device_count():
            raise KernelError("device_unavailable", "The requested CUDA device index does not exist.")
    return torch.device(device)


def _basis(torch: Any, x: Any, intercept: bool, *, need_q: bool) -> tuple[Any, Any, float, Any, dict[str, Any]]:
    """Center/scale the solve, then retain its exact coefficient transformation."""
    n, k = x.shape
    magnitude = x.abs().amax(dim=0)
    if bool(torch.any(magnitude == 0)):
        raise KernelError("singular_design", "The design contains an all-zero column.")
    z = x / magnitude
    means = torch.zeros(k, dtype=x.dtype, device=x.device)
    if intercept:
        if not bool(torch.all(x[:, 0] == 1)):
            raise KernelError("invalid_design", "An intercept design must have a leading column of ones.")
        means[1:] = z[:, 1:].mean(dim=0)
        z[:, 1:] -= means[1:]
    rms = z.square().mean(dim=0).sqrt()
    if bool(torch.any(rms == 0)):
        raise KernelError("singular_design", "The design contains an all-zero or constant predictor.")
    scales = magnitude * rms
    z /= rms
    # The singular values of R equal those of the n-by-k normalized design,
    # avoiding a second decomposition over the full observation matrix.
    q, r = torch.linalg.qr(z, mode="reduced" if need_q else "r")
    singular = torch.linalg.svdvals(r)
    rank_limit = torch.finfo(x.dtype).eps * max(n, k) * singular[0]
    if bool(singular[-1] <= rank_limit):
        raise KernelError("singular_design", "The normalized design matrix is rank deficient.")
    transform = torch.diag(1 / scales)
    if intercept:
        transform[0, 1:] = -means[1:] / rms[1:]
    if not bool(torch.isfinite(transform).all()):
        raise KernelError("numerical_failure", "Predictor units exceed float64 coefficient precision; rescale the input.")
    diagnostics = {
        "condition_number_basis": "centered_and_scaled_design" if intercept else "scaled_design",
        "column_centers": (means * magnitude).cpu().tolist(),
        "column_scales": scales.cpu().tolist(),
        "rank_diagnostic": "SVD of scaled reduced QR factor",
        "rank_tolerance": "eps * max(n, k) * largest_singular_value",
        "dense_observation_matrix": False,
    }
    return z, transform, float((singular[0] / singular[-1]).item()), (q, r), diagnostics


def _cluster_meat(torch: Any, scores: Any, groups: Any | None) -> tuple[Any, int]:
    n, k = scores.shape
    if groups is None:
        raise KernelError("invalid_clusters", "Cluster covariance requires one group code per observation.")
    try:
        if isinstance(groups, torch.Tensor):
            if groups.ndim != 1 or len(groups) != n or not bool(torch.isfinite(groups).all()):
                raise ValueError("Invalid cluster shape or values")
            _, index = torch.unique(groups.to(device=scores.device), return_inverse=True)
        else:
            if len(groups) != n:
                raise ValueError("Invalid cluster shape")
            codes, _ = pd.factorize(groups, sort=False)
            if (codes < 0).any():
                raise ValueError("Missing cluster labels")
            index = torch.as_tensor(codes, dtype=torch.int64, device=scores.device)
    except (TypeError, ValueError) as exc:
        raise KernelError("invalid_clusters", "Cluster labels must be nonmissing scalar values, one per row.") from exc
    count = int(index.max().item()) + 1
    if count < 2:
        raise KernelError("insufficient_clusters", "Cluster covariance requires at least two groups.")
    correction = count / (count - 1) * (n - 1) / (n - k)
    if count == n:
        return (scores.T @ scores) * correction, count
    totals = torch.zeros((count, k), dtype=scores.dtype, device=scores.device)
    totals.index_add_(0, index, scores)
    return (totals.T @ totals) * correction, count


def _binary_terms(torch: Any, estimator: str, z: Any, y: Any, theta: Any) -> tuple[Any, Any, Any, Any]:
    eta = z @ theta
    signed = 2 * y - 1
    t = signed * eta
    if estimator == "logit":
        ll = torch.nn.functional.logsigmoid(t).sum()
        score = signed * torch.sigmoid(-t)
        weight = torch.sigmoid(eta) * torch.sigmoid(-eta)
        fitted = torch.sigmoid(eta)
    else:
        log_cdf = torch.special.log_ndtr(t)
        ll = log_cdf.sum()
        # erfcx is stable in the negative tail where exp(log_pdf-log_cdf)
        # eventually loses precision by subtracting two very large numbers.
        negative_t = torch.minimum(t, torch.zeros_like(t))
        negative_ratio = math.sqrt(2 / math.pi) / torch.special.erfcx(-negative_t / math.sqrt(2))
        positive_ratio = torch.exp(-0.5 * t.square() - 0.5 * math.log(2 * math.pi) - log_cdf)
        ratio = torch.where(t < 0, negative_ratio, positive_ratio)
        score = signed * ratio
        weight = ratio * (ratio + t)
        inverse_square = torch.minimum(t, torch.full_like(t, -1.0)).reciprocal().square()
        asymptotic = 1 - inverse_square + 6 * inverse_square.square() - 50 * inverse_square.pow(3)
        weight = torch.where(t < -1e4, asymptotic, weight)
        fitted = torch.special.ndtr(eta)
    return ll, score, weight, fitted


def _binary_log_likelihood(torch: Any, estimator: str, signed: Any, z: Any, theta: Any) -> Any:
    margin = signed * (z @ theta)
    if estimator == "logit":
        return torch.nn.functional.logsigmoid(margin).sum()
    return torch.special.log_ndtr(margin).sum()


def _observed_information(torch: Any, z: Any, weights: Any) -> tuple[Any, Any]:
    information = z.T @ (z * weights[:, None])
    chol, info = torch.linalg.cholesky_ex(information, check_errors=False)
    if int(info.item()) != 0 or not bool(torch.isfinite(chol).all()):
        raise KernelError("singular_information", "The observed information is not positive definite; a finite stable estimate is unavailable.")
    return information, chol


def _binary_fit(
    torch: Any, estimator: str, z: Any, y: Any, max_iter: int, tolerance: float, intercept: bool
) -> tuple[Any, Any, Any, Any, Any, int, dict[str, Any]]:
    n, k = z.shape
    theta = torch.zeros(k, dtype=z.dtype, device=z.device)
    if intercept:
        prevalence = y.mean()
        theta[0] = torch.logit(prevalence) if estimator == "logit" else torch.special.ndtri(prevalence)
    signed = 2 * y - 1
    converged = False
    backtracks = 0
    eps = torch.finfo(z.dtype).eps
    for iteration in range(1, max_iter + 1):
        ll, score, weight, fitted = _binary_terms(torch, estimator, z, y, theta)
        if not bool(torch.isfinite(ll)) or not bool(torch.isfinite(weight).all()):
            raise KernelError("numerical_failure", "The binary likelihood produced non-finite values.")
        _, chol = _observed_information(torch, z, weight)
        gradient = z.T @ score
        step = torch.cholesky_solve(gradient[:, None], chol)[:, 0]
        gradient_norm = float(gradient.abs().max().item()) / n
        step_norm = float(step.abs().max().item())
        theta_norm = float(theta.abs().max().item())
        if gradient_norm <= tolerance and step_norm <= tolerance * (1 + theta_norm):
            converged = True
            break
        ll_value = float(ll.item())
        directional = float(torch.dot(gradient, step).item())
        allowance = 8 * eps * max(1.0, abs(ll_value))
        fraction = 1.0
        for _ in range(50):
            candidate = theta + fraction * step
            next_ll = _binary_log_likelihood(torch, estimator, signed, z, candidate)
            if bool(torch.isfinite(next_ll)) and float(next_ll.item()) >= ll_value + 1e-4 * fraction * directional - allowance:
                theta = candidate
                break
            fraction *= 0.5
            backtracks += 1
        else:
            raise KernelError("nonconvergence", "The binary likelihood line search could not find a stable improving step.")
    if not converged:
        raise KernelError("nonconvergence", f"The {estimator} solver did not converge in {max_iter} iterations.")
    bread = torch.cholesky_solve(torch.eye(k, dtype=z.dtype, device=z.device), chol)
    diagnostics = {
        "converged": True,
        "gradient_max_per_observation": gradient_norm,
        "newton_step_max": step_norm,
        "line_search_backtracks": backtracks,
        "convergence_rule": "normalized_score_and_relative_newton_step",
    }
    return theta, bread, fitted, ll, score, iteration, diagnostics


def solve(
    estimator: str,
    x: Any,
    y: Any,
    covariance: str,
    groups: Any | None = None,
    *,
    intercept: bool = True,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    device: str = "cpu",
) -> KernelResult:
    """Estimate with native tensor algebra and an explicitly selected device.

    Inputs must already have passed sample/category/separation validation in the
    analysis layer. Direct kernel calls still validate shapes, rank, and solver
    controls. CUDA scalar convergence checks synchronize each Newton iteration.
    """
    if estimator not in {"ols", "logit", "probit"}:
        raise KernelError("unsupported_estimator", f"Unsupported estimator: {estimator}.")
    supported = {"nonrobust", "HC1", "HC3", "cluster"} if estimator == "ols" else {"nonrobust", "cluster"}
    if covariance not in supported:
        raise KernelError("unsupported_covariance", f"{estimator} does not support {covariance} covariance.")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise KernelError("invalid_solver_options", "max_iter must be a positive integer.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise KernelError("invalid_solver_options", "tolerance must be positive and finite.")
    selected = _resolve_device(torch, device)
    try:
        with torch.inference_mode():
            tx = torch.as_tensor(x, dtype=torch.float64, device=selected)
            ty = torch.as_tensor(y, dtype=torch.float64, device=selected)
            if tx.ndim != 2 or ty.ndim != 1 or len(ty) != len(tx) or tx.shape[1] < 1:
                raise KernelError("invalid_design", "Expected an n-by-k design and an n-element outcome.")
            n, k = tx.shape
            if n <= k:
                raise KernelError("insufficient_observations", "Estimation requires more observations than design columns.")
            if not bool(torch.isfinite(tx).all()) or not bool(torch.isfinite(ty).all()):
                raise KernelError("non_finite_values", "Kernel inputs must be finite.")
            if estimator != "ols" and (not bool(((ty == 0) | (ty == 1)).all()) or torch.unique(ty).numel() != 2):
                raise KernelError("invalid_binary_outcome", "Binary models require both 0 and 1 outcomes.")
            z, transform, condition, (q, r), diagnostics = _basis(torch, tx, intercept, need_q=estimator == "ols")
            diagnostics.update({
                "engine": "torch", "device": str(selected), "dtype": "float64",
                "transfer_policy": "input boundary only; no per-iteration array transfers",
                "iteration_scalar_synchronization": selected.type == "cuda" and estimator != "ols",
                "autograd": False, "torch_version": torch.__version__,
            })
            group_count = None
            if estimator == "ols":
                theta = torch.linalg.solve_triangular(r, (q.T @ ty)[:, None], upper=True)[:, 0]
                fitted = z @ theta
                residual = ty - fitted
                ssr = torch.dot(residual, residual)
                if not bool(torch.isfinite(ssr)) or float(ssr.item()) <= 0:
                    raise KernelError("numerical_failure", "Residual variance is zero or outside finite float64 range.")
                inverse_r = torch.linalg.solve_triangular(r, torch.eye(k, dtype=z.dtype, device=selected), upper=True)
                if covariance == "nonrobust":
                    variance = (inverse_r @ inverse_r.T) * ssr / (n - k)
                else:
                    score_residual = residual
                    if covariance == "HC3":
                        complement = 1 - q.square().sum(dim=1)
                        if bool(torch.any(complement <= 100 * torch.finfo(z.dtype).eps)):
                            raise KernelError("undefined_hc3", "HC3 is undefined for observations with unit leverage.")
                        score_residual = residual / complement
                        diagnostics["max_leverage"] = float((1 - complement).max().item())
                    # Use the orthogonal QR basis for robust covariance to
                    # avoid forming a squared-condition X'X sandwich.
                    scores = q * score_residual[:, None]
                    if covariance == "cluster":
                        meat, group_count = _cluster_meat(torch, scores, groups)
                    else:
                        meat = scores.T @ scores
                        if covariance == "HC1":
                            meat = meat * n / (n - k)
                    variance = inverse_r @ meat @ inverse_r.T
                ll = -0.5 * n * (math.log(2 * math.pi) + 1 + torch.log(ssr / n))
                iterations = 0
                solver = "torch_qr"
            else:
                # QR's Q is not needed by Newton or its sandwich covariance.
                del q, r
                theta, bread, fitted, ll, scores_scalar, iterations, convergence = _binary_fit(
                    torch, estimator, z, ty, max_iter, tolerance, intercept
                )
                diagnostics.update(convergence)
                if covariance == "cluster":
                    meat, group_count = _cluster_meat(torch, z * scores_scalar[:, None], groups)
                    variance = bread @ meat @ bread
                else:
                    variance = bread
                solver = "torch_newton_cholesky_backtracking"
            parameters = transform @ theta
            variance = transform @ variance @ transform.T
            variance = (variance + variance.T) / 2
            if not bool(torch.isfinite(parameters).all()) or not bool(torch.isfinite(variance).all()):
                raise KernelError("numerical_failure", "Estimation produced non-finite coefficients or covariance.")
            if group_count is not None:
                diagnostics["cluster_count"] = group_count
                diagnostics["small_sample_correction"] = group_count / (group_count - 1) * (n - 1) / (n - k)
            return KernelResult(
                parameters=parameters, covariance=variance,
                fitted=fitted, log_likelihood=float(ll.item()),
                condition_number=condition, iterations=iterations, solver=solver, diagnostics=diagnostics,
            )
    except KernelError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise KernelError("numerical_failure", f"PyTorch could not complete float64 estimation on {selected}: {exc}") from exc
