"""Float64 Gaussian SAR/SEM/SAC/SDM exact likelihood and multiplier kernels.

No scipy, statsmodels, graph layout, stochastic logdet or unreported fallback.
Derivatives are native PyTorch autograd of the explicit scalar likelihood.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

from openecon.engines.contracts import KernelError
from openecon.engines.optimize import information_inverse, maximize_newton


@dataclass
class SpatialFit:
    params: torch.Tensor
    covariance: torch.Tensor
    log_likelihood: float
    gradient: torch.Tensor
    hessian: torch.Tensor
    fitted: torch.Tensor
    innovations: torch.Tensor
    structural_residuals: torch.Tensor
    diagnostics: dict


def spatial_names(model):
    return ("lambda",) if model == "sem" else ("rho", "lambda") if model == "sac" else ("rho",)


def stable_bound(w):
    """Conservative Neumann stability interval; valid for asymmetric W too."""
    norm = float(w.abs().sum(dim=1).max())
    if not math.isfinite(norm) or norm <= 0:
        raise KernelError(
            "unidentified_spatial_parameter", "A spatial model requires nonzero finite weights."
        )
    return (1 - 1e-6) / norm


def _validate(y, x, w, w_error, model):
    if model not in {"sar", "sem", "sac", "sdm"}:
        raise KernelError("invalid_spatial_model", "Choose SAR, SEM, SAC or SDM.")
    n = len(y)
    if (
        y.ndim != 1
        or x.ndim != 2
        or x.shape[0] != n
        or w.shape != (n, n)
        or w_error.shape != (n, n)
    ):
        raise KernelError(
            "invalid_spatial_dimensions", "Spatial y, X and weight matrices must align."
        )
    if any(
        value.dtype != torch.float64
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
        for value in (y, x, w, w_error)
    ):
        raise KernelError(
            "invalid_spatial_inputs", "Spatial kernels require finite float64 CPU tensors."
        )
    if any(
        bool((matrix < 0).any()) or bool((matrix.diagonal() != 0).any()) for matrix in (w, w_error)
    ):
        raise KernelError(
            "invalid_spatial_weights", "Spatial kernels require nonnegative zero-diagonal W."
        )
    p = x.shape[1] + len(spatial_names(model)) + 1
    if n <= p or x.shape[1] == 0:
        raise KernelError(
            "insufficient_observations",
            "Spatial ML needs more observations than all regression, spatial and variance parameters.",
        )
    if int(torch.linalg.matrix_rank(x)) != x.shape[1] or float(torch.linalg.cond(x)) > 1e7:
        raise KernelError(
            "spatial_rank_deficient",
            "The spatial design is rank deficient or poorly conditioned; rescale or remove regressors explicitly.",
        )


def _filters(spatial, w, w_error, model):
    identity = torch.eye(w.shape[0], dtype=torch.float64)
    if model == "sem":
        return identity, identity - spatial[0] * w
    a = identity - spatial[0] * w
    b = identity - spatial[1] * w_error if model == "sac" else identity
    return a, b


def _logdet(matrix):
    sign, value = torch.linalg.slogdet(matrix)
    # The supported connected-to-zero stable domain has positive determinant.
    if float(sign.detach()) != 1 or not bool(torch.isfinite(value.detach())):
        raise KernelError(
            "spatial_singular_filter",
            "The spatial filter is singular or outside the positive determinant domain.",
        )
    return value


def log_likelihood(params, y, x, w, *, model="sar", w_error=None):
    """Full unprofiled Gaussian likelihood in [beta, spatial, ln(sigma^2)] units.

    e = (I-lambda W_error) [(I-rho W)y-X beta]; logdet of both filters
    plus the Gaussian innovation density. SDM passes [X, W X_slopes] as x.
    This low-level function is differentiable; fit validates its inputs first.
    """
    w_error = w if w_error is None else w_error
    k = x.shape[1]
    a, b = _filters(params[k:-1], w, w_error, model)
    innovations = b @ (a @ y - x @ params[:k])
    return (
        _logdet(a)
        + _logdet(b)
        - 0.5 * len(y) * (math.log(2 * math.pi) + params[-1])
        - 0.5 * innovations.square().sum() * torch.exp(-params[-1])
    )


def profile_likelihood(spatial, y, x, w, *, model="sar", w_error=None, components=False):
    w_error = w if w_error is None else w_error
    a, b = _filters(spatial, w, w_error, model)
    transformed_y, transformed_x = b @ (a @ y), b @ x
    # QR solves GLS without squaring X's condition number. QR/triangular solve
    # support native first and second derivatives in the supported full-rank domain.
    q, r = torch.linalg.qr(transformed_x, mode="reduced")
    beta = torch.linalg.solve_triangular(r, (q.T @ transformed_y)[:, None], upper=True)[:, 0]
    innovation = transformed_y - transformed_x @ beta
    sigma2 = innovation.square().sum() / len(y)
    if float(sigma2.detach()) <= torch.finfo(torch.float64).eps * max(
        1.0, float(y.square().mean())
    ):
        raise KernelError(
            "spatial_degenerate_variance", "The innovation variance is zero at working precision."
        )
    value = _logdet(a) + _logdet(b) - 0.5 * len(y) * (math.log(2 * math.pi) + 1 + torch.log(sigma2))
    return (value, beta, sigma2, innovation) if components else value


def derivatives(function, params):
    point = params.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        value = function(point)
        gradient = torch.autograd.grad(value, point, create_graph=True)[0]
        hessian = torch.stack(
            [
                torch.autograd.grad(gradient[i], point, retain_graph=True)[0]
                for i in range(len(point))
            ]
        )
    return value.detach(), gradient.detach(), hessian.detach()


def fit_ml(y, x, w, *, model="sar", w_error=None, max_iterations=100, tolerance=1e-9):
    w_error = w if w_error is None else w_error
    _validate(y, x, w, w_error, model)
    names = spatial_names(model)
    bounds = torch.tensor(
        [stable_bound(w if name == "rho" or model == "sem" else w_error) for name in names],
        dtype=torch.float64,
    )

    def value_fn(point):
        if not bool(torch.isfinite(point).all()) or bool((point.abs() >= bounds).any()):
            return float("-inf")
        return profile_likelihood(point, y, x, w, model=model, w_error=w_error)

    def objective(point):
        if bool((point.abs() >= bounds).any()):
            return (
                torch.tensor(float("-inf"), dtype=torch.float64),
                torch.zeros_like(point),
                -torch.eye(len(point), dtype=torch.float64),
            )
        return derivatives(
            lambda p: profile_likelihood(p, y, x, w, model=model, w_error=w_error), point
        )

    # A reported finite grid plus multiple starts addresses the SAC likelihood's
    # known ridges/local maxima. This is not a proof of a unique global maximum.
    fractions = (-0.8, -0.4, 0.0, 0.4, 0.8)
    starts = [
        bounds * torch.tensor(v, dtype=torch.float64)
        for v in itertools.product(fractions, repeat=len(names))
    ]
    ranked = sorted(
        ((float(value_fn(start)), start) for start in starts),
        key=lambda item: item[0],
        reverse=True,
    )
    attempts = []
    candidates = []
    for _, start in ranked[: 5 if model == "sac" else 2]:
        try:
            fit = maximize_newton(
                objective,
                start,
                max_iter=max_iterations,
                gradient_tol=tolerance,
                step_tol=tolerance,
                scaled_gradient_tol=tolerance,
                value_fn=value_fn,
                raise_on_failure=False,
            )
            attempts.append(
                {
                    "start": start.tolist(),
                    "converged": fit.converged,
                    "iterations": fit.iterations,
                    "log_likelihood": fit.value,
                }
            )
            if fit.converged and bool((fit.theta.abs() < bounds * (1 - 1e-5)).all()):
                candidates.append(fit)
        except KernelError as exc:
            attempts.append({"start": start.tolist(), "converged": False, "error": exc.code})
    if not candidates:
        error = KernelError(
            "spatial_nonconvergence",
            "No interior spatial likelihood maximum converged; increase the iteration budget or inspect identification/boundary solutions.",
        )
        error.diagnostics = {"attempts": attempts, "bounds": bounds.tolist()}
        raise error
    fit = max(candidates, key=lambda candidate: candidate.value)
    if fit.value < ranked[0][0] - tolerance * max(1.0, abs(fit.value)):
        raise KernelError(
            "spatial_nonconvergence",
            "The converged spatial maximum is below the evaluated grid; a better/boundary solution remains unresolved.",
        )
    value, beta, sigma2, innovations = profile_likelihood(
        fit.theta, y, x, w, model=model, w_error=w_error, components=True
    )
    params = torch.cat((beta, fit.theta, sigma2.log().reshape(1)))
    full_value, gradient, hessian = derivatives(
        lambda p: log_likelihood(p, y, x, w, model=model, w_error=w_error), params
    )
    covariance = information_inverse(-hessian)
    a, b = _filters(fit.theta, w, w_error, model)
    fitted = torch.linalg.solve(a, x @ beta)  # unconditional reduced-form mean
    structural = a @ y - x @ beta
    diagnostics = {
        "converged": True,
        "iterations": fit.iterations,
        "attempts": attempts,
        "grid_points": len(starts),
        "grid_best_log_likelihood": ranked[0][0],
        "gradient_max": float(gradient.abs().max()),
        "profile_gradient": fit.gradient.tolist(),
        "profile_hessian": fit.hessian.tolist(),
        "parameter_bounds": dict(zip(names, bounds.tolist(), strict=True)),
        "domain": "abs(parameter) < (1-1e-6)/max_abs_row_sum(W); interior maximum required",
        "global_maximum_guaranteed": False,
        "logdet": "exact dense LU slogdet",
        "covariance": "inverse full observed information in reporting units",
        "derivatives": "float64 PyTorch autograd",
        "starts": "five-point grid per spatial parameter",
    }
    return SpatialFit(
        params.detach(),
        covariance.detach(),
        float(full_value),
        gradient,
        hessian,
        fitted.detach(),
        innovations.detach(),
        structural.detach(),
        diagnostics,
    )


def multiplier_impacts(params, covariance, w, *, terms, predictors, model, alpha=0.05):
    """Exact LeSage-Pace averages and full-covariance delta-method uncertainty."""
    from openecon.engines.inference import critical_value

    if model not in {"sar", "sdm", "sac"}:
        raise KernelError(
            "unsupported_spatial_impacts", "Multiplier impacts require SAR, SDM or SAC."
        )
    rho_index = terms.index("rho")
    rho = params[rho_index]
    identity = torch.eye(w.shape[0], dtype=torch.float64)
    if abs(float(rho)) >= stable_bound(w):
        raise KernelError(
            "spatial_domain", "rho is outside the supported stable multiplier domain."
        )
    multiplier = torch.linalg.solve(identity - rho * w, identity)
    derivative_rho = multiplier @ w @ multiplier
    critical = critical_value(alpha, None)
    rows = []
    covariance_by_predictor = {}
    for predictor in predictors:
        index = terms.index(predictor)
        lag_index = terms.index("W:" + predictor) if model == "sdm" else None
        beta, theta = params[index], params[lag_index] if lag_index is not None else 0.0
        base = beta * identity + theta * w
        effect = multiplier @ base
        drho = derivative_rho @ base
        dbeta = multiplier
        dtheta = multiplier @ w

        def summaries(m):
            return torch.stack(
                (m.diagonal().mean(), m.sum() / len(w) - m.diagonal().mean(), m.sum() / len(w))
            )

        estimates = summaries(effect)
        jacobian = torch.zeros((3, len(params)), dtype=torch.float64)
        jacobian[:, index], jacobian[:, rho_index] = summaries(dbeta), summaries(drho)
        if lag_index is not None:
            jacobian[:, lag_index] = summaries(dtheta)
        impact_covariance = jacobian @ covariance @ jacobian.T
        covariance_by_predictor[predictor] = impact_covariance.tolist()
        if not bool(torch.isfinite(impact_covariance).all()) or bool(
            (impact_covariance.diagonal() < -1e-10).any()
        ):
            raise KernelError("invalid_covariance", "Impact delta-method variance is invalid.")
        se = impact_covariance.diagonal().clamp_min(0).sqrt()
        for i, kind in enumerate(("direct", "indirect", "total")):
            rows.append(
                {
                    "predictor": predictor,
                    "impact": kind,
                    "estimate": float(estimates[i]),
                    "std_error": float(se[i]),
                    "ci_low": float(estimates[i] - critical * se[i]),
                    "ci_high": float(estimates[i] + critical * se[i]),
                    "gradient": jacobian[i].tolist(),
                }
            )
    return {
        "rows": rows,
        "covariance_by_predictor": covariance_by_predictor,
        "covariance_order": ["direct", "indirect", "total"],
        "method": "full observed-information delta method",
        "alpha": alpha,
        "definition": "direct=trace(S)/N; total=sum(S)/N; indirect=total-direct; S=(I-rho W)^-1(beta I+theta W)",
        "conditioning": "W fixed and exogenous; uncertainty of W is excluded",
        "sample_n": len(w),
    }
