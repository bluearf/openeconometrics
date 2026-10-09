"""Bounded CF and binary partial-target rotations on CPU float64 tensors.

The Crawford--Ferguson criterion is a convex combination of squared-loading
row and column complexity. Partial targets minimize half the squared error of
the specified cells. Both use analytic loading gradients and monotone gradient
projection, starting at identity. A small projected gradient establishes local
stationarity, never global optimality or rotational uniqueness.

References: Crawford and Ferguson (1970), doi:10.1007/BF02310792;
Jennrich (2001, 2002), doi:10.1007/BF02294840 and doi:10.1007/BF02294706;
https://www.stata.com/manuals/mvrotate.pdf (binary partial targets).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from .rotation import ACCEPTABLE, Rotation, _row_scale

MAX_VARIABLES = 256
MAX_FACTORS = 16
MAX_CONDITION = 1e10


@dataclass
class RotationExtension:
    rotation: Rotation
    diagnostics: dict
    target: Tensor | None = None
    mask: Tensor | None = None


def cf_value(pattern: Tensor, kappa: float) -> tuple[float, Tensor]:
    """CF objective and exact derivative with respect to the pattern matrix."""
    squares = pattern.square()
    row = squares.sum(1, keepdim=True) - squares
    column = squares.sum(0, keepdim=True) - squares
    complexity = (1 - kappa) * row + kappa * column
    return float((squares * complexity).sum() / 4), pattern * complexity


def partial_value(pattern: Tensor, target: Tensor, mask: Tensor) -> tuple[float, Tensor]:
    """Half specified-cell residual sum of squares; free cells contribute zero."""
    residual = torch.where(mask, pattern - target, 0.)
    return float(residual.square().sum() / 2), residual


def _condition(matrix: Tensor, what: str) -> float:
    singular = torch.linalg.svdvals(matrix)
    condition = float(singular.max() / singular.min().clamp_min(1e-300))
    if not math.isfinite(condition) or condition > MAX_CONDITION \
            or float(singular.min()) <= 1e-12 * max(float(singular.max()), 1e-300):
        raise KernelError("singular_matrix", f"The {what} is singular or numerically ill-conditioned.")
    return condition


def _inputs(loadings: Tensor, *, oblique: bool, kaiser: bool, max_iter: int, tol: float):
    if not isinstance(loadings, Tensor) or loadings.device.type != "cpu" \
            or loadings.dtype != torch.float64 or loadings.ndim != 2:
        raise KernelError("invalid_spec", "Rotation extensions require a CPU float64 loading matrix.")
    p, m = loadings.shape
    if p < m or m < 2 or p > MAX_VARIABLES or m > MAX_FACTORS:
        raise KernelError("workspace_limit", "Rotation extensions require 2--16 factors and at most 256 variables.")
    if not bool(torch.isfinite(loadings).all()) or float(loadings.abs().max()) > 1e50:
        raise KernelError("non_finite_values", "Loading values are non-finite or too large for float64 rotation.")
    if not isinstance(oblique, bool) or not isinstance(kaiser, bool):
        raise KernelError("invalid_option", "oblique and kaiser must be boolean.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or not 1 <= max_iter <= 10000:
        raise KernelError("invalid_option", "max_iter must be an integer from 1 through 10000.")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) \
            or not math.isfinite(tol) or not 1e-12 <= tol <= ACCEPTABLE:
        raise KernelError("invalid_option", "tol must be finite and from 1e-12 through 1e-5.")
    _condition(loadings, "loading matrix")
    scale = _row_scale(loadings, kaiser)
    return loadings / scale[:, None], scale


def _projected(t: Tensor, gradient: Tensor, oblique: bool) -> Tensor:
    if oblique:
        return gradient - t * (t * gradient).sum(0)
    inner = t.T @ gradient
    return gradient - t @ ((inner + inner.T) / 2)


def _fit(a: Tensor, criterion: Callable, *, oblique: bool, max_iter: int,
         tol: float, what: str) -> tuple[Tensor, int, bool, float, float]:
    """Identity-start projected descent on O(m) or unit-column transformations."""
    m = a.shape[1]
    t = torch.eye(m, dtype=a.dtype)

    def evaluate(trial: Tensor):
        if oblique:
            _condition(trial, "oblique transformation")
            inverse = torch.linalg.inv(trial)
            pattern = a @ inverse.T
        else:
            pattern = a @ trial
        value, slope = criterion(pattern)
        gradient = -(pattern.T @ slope @ inverse).T if oblique else a.T @ slope
        if not math.isfinite(value) or not bool(torch.isfinite(gradient).all()):
            raise KernelError("numerical_failure", "Rotation criterion or gradient is non-finite.")
        return value, gradient

    value, gradient = evaluate(t)
    step = 1.
    noise = 64 * torch.finfo(a.dtype).eps
    for iteration in range(max_iter + 1):
        projected = _projected(t, gradient, oblique)
        size = float(torch.linalg.matrix_norm(projected))
        if size <= tol or (iteration == max_iter and size <= ACCEPTABLE):
            matrix = torch.linalg.inv(t).T if oblique else t
            return matrix, iteration, size <= tol, value, size
        if iteration == max_iter:
            raise KernelError("no_convergence", f"The {what} rotation did not converge in {max_iter} iterations "
                              f"(projected gradient {size:.2e}).")
        step *= 2
        for _ in range(60):
            trial = t - step * projected
            if oblique:
                norms = trial.square().sum(0).sqrt()
                if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-300).any()):
                    step /= 2
                    continue
                trial = trial / norms
            else:
                u, _, vh = torch.linalg.svd(trial)
                trial = u @ vh
            try:
                new_value, new_gradient = evaluate(trial)
            except (KernelError, torch.linalg.LinAlgError):
                step /= 2
                continue
            required = .5 * step * size * size
            if value - new_value > required:
                break
            if required <= noise * max(abs(value), 1.):
                reduced = float(torch.linalg.matrix_norm(_projected(trial, new_gradient, oblique)))
                if reduced < size and new_value <= value + noise * max(abs(value), 1.):
                    break
            step /= 2
        else:
            if size <= ACCEPTABLE:
                matrix = torch.linalg.inv(t).T if oblique else t
                return matrix, iteration, False, value, size
            raise KernelError("no_convergence", f"The {what} rotation line search could not reduce the criterion.")
        t, value, gradient = trial, new_value, new_gradient
    raise AssertionError("unreachable")


def tangent_jacobian(pattern: Tensor, matrix: Tensor, mask: Tensor, *, oblique: bool) -> Tensor:
    """Specified-loading Jacobian in an orthonormal tangent basis at the result."""
    m = matrix.shape[0]
    columns = []
    if not oblique:
        for j in range(m - 1):
            for k in range(j + 1, m):
                skew = torch.zeros_like(matrix)
                skew[j, k], skew[k, j] = 1 / math.sqrt(2), -1 / math.sqrt(2)
                columns.append((pattern @ skew)[mask])
    else:
        t = torch.linalg.inv(matrix).T
        for j in range(m):
            # The right singular vectors other than t_j span its orthogonal complement.
            _, _, vh = torch.linalg.svd(t[:, j][None, :], full_matrices=True)
            for direction in vh[1:]:
                delta = torch.zeros_like(t)
                delta[:, j] = direction
                columns.append((-pattern @ delta.T @ matrix)[mask])
    return torch.stack(columns, dim=1)


def _result(loadings: Tensor, matrix: Tensor, iterations: int, converged: bool,
            *, oblique: bool, canonical: bool) -> tuple[Rotation, float]:
    condition = _condition(matrix, "rotation matrix")
    pattern = loadings @ matrix
    if canonical:
        order = torch.argsort(pattern.square().sum(0), descending=True, stable=True)
        pattern, matrix = pattern[:, order], matrix[:, order]
        sign = torch.sign(pattern[pattern.abs().argmax(0), torch.arange(matrix.shape[0])])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        pattern, matrix = pattern * sign, matrix * sign
    phi = torch.linalg.inv(matrix.T @ matrix) if oblique else torch.eye(matrix.shape[0], dtype=matrix.dtype)
    phi = (phi + phi.T) / 2
    if not bool(torch.isfinite(pattern).all()) or not bool(torch.isfinite(phi).all()):
        raise KernelError("numerical_failure", "Rotation produced non-finite output.")
    if oblique and not torch.allclose(phi.diagonal(), torch.ones(matrix.shape[0], dtype=matrix.dtype),
                                      atol=1e-9, rtol=1e-9):
        raise KernelError("numerical_failure", "Oblique factor variances are not unit; no repair is applied.")
    common = pattern @ phi @ pattern.T
    expected = loadings @ loadings.T
    if not torch.allclose(common, expected, atol=1e-9 * max(float(expected.abs().max()), 1e-300), rtol=1e-9):
        raise KernelError("numerical_failure", "Rotation failed to preserve the common covariance.")
    return Rotation(pattern, matrix, phi, iterations, converged, oblique), condition


def _diagnostics(value: float, gradient: float, condition: float, rotation: Rotation, tol: float) -> dict:
    return {"rotation_criterion": value, "rotation_projected_gradient": gradient,
            "rotation_condition": condition, "rotation_converged": rotation.converged,
            "rotation_oblique": rotation.oblique, "rotation_start": "identity",
            "rotation_optimum": "local stationary point; minimum/global uniqueness not established",
            "rotation_tolerance": tol, "rotation_acceptable_tolerance": ACCEPTABLE,
            "rotation_inference": "descriptive; loading SE/CI not provided"}


def cf(loadings: Tensor, *, kappa: float = 0., oblique: bool = False,
       kaiser: bool = True, max_iter: int = 10000, tol: float = 1e-10) -> RotationExtension:
    """Crawford--Ferguson rotation, with kappa weighting column complexity."""
    a, _ = _inputs(loadings, oblique=oblique, kaiser=kaiser, max_iter=max_iter, tol=tol)
    if isinstance(kappa, bool) or not isinstance(kappa, (int, float)) \
            or not math.isfinite(kappa) or not 0 <= kappa <= 1:
        raise KernelError("invalid_option", "CF kappa must be finite and from zero through one.")
    matrix, iterations, converged, value, gradient = _fit(a, lambda z: cf_value(z, kappa),
        oblique=oblique, max_iter=max_iter, tol=tol, what="Crawford--Ferguson")
    rotation, condition = _result(loadings, matrix, iterations, converged, oblique=oblique, canonical=True)
    diagnostics = _diagnostics(value, gradient, condition, rotation, tol)
    diagnostics.update({"cf_kappa": kappa, "rotation_objective": "Crawford-Ferguson quarter squared-loading complexity",
                        "rotation_orientation": "descending pattern SS; largest loading positive"})
    return RotationExtension(rotation, diagnostics)


def _target_inputs(target, mask, shape) -> tuple[Tensor, Tensor]:
    for value, name in ((target, "target"), (mask, "target mask")):
        try:
            dimensions = tuple(value.shape) if hasattr(value, "shape") else (len(value), len(value[0]))
        except (TypeError, IndexError) as exc:
            raise KernelError("invalid_spec", f"The {name} must match the loading dimensions.") from exc
        if dimensions != tuple(shape):
            raise KernelError("invalid_spec", "Target and mask dimensions must match the loadings.")
        if isinstance(value, Tensor) and (value.device.type != "cpu" or value.is_complex()) \
                or getattr(getattr(value, "dtype", None), "kind", None) == "c":
            raise KernelError("invalid_spec", f"The {name} must be a real CPU matrix.")
    try:
        target = torch.as_tensor(target, dtype=torch.float64)
        numeric_mask = torch.as_tensor(mask, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise KernelError("invalid_spec", "Supply a numeric target and a binary target mask.") from exc
    if target.shape != shape or numeric_mask.shape != shape:
        raise KernelError("invalid_spec", "Target and mask dimensions must match the loadings.")
    if not bool(torch.isfinite(numeric_mask).all()) or not bool(((numeric_mask == 0) | (numeric_mask == 1)).all()):
        raise KernelError("invalid_spec", "Target mask must contain only zero/one or boolean values.")
    mask = numeric_mask.bool()
    if not bool(mask.any()) or not bool(torch.isfinite(target[mask]).all()):
        raise KernelError("invalid_spec", "Specify at least one target; every specified target must be finite.")
    if not bool((mask & (target != 0)).any(0).all()):
        raise KernelError("singular_matrix", "Each target factor needs a specified nonzero orientation anchor.")
    target = torch.where(mask, target, 0.)
    if float(target.abs().max()) > 1e100:
        raise KernelError("non_finite_values", "Target values are too large for float64 rotation.")
    return target, mask


def partial_target(loadings: Tensor, *, target, mask, oblique: bool = False,
                   kaiser: bool = True, max_iter: int = 10000, tol: float = 1e-10) -> RotationExtension:
    """Binary partial target, preserving caller axes and nonzero anchor orientation.

    The mask is fixed and never estimated. Specified targets use the loading
    metric; under Kaiser normalization both loadings and targets are row-scaled.
    Rank of the final masked tangent Jacobian establishes local sensitivity,
    not globally unique rotation. Zero-only factor targets are outside this
    bounded contract because they do not anchor the factor sign.
    """
    a, scale = _inputs(loadings, oblique=oblique, kaiser=kaiser, max_iter=max_iter, tol=tol)
    target, mask = _target_inputs(target, mask, loadings.shape)
    m = loadings.shape[1]
    dimension = m * (m - 1) if oblique else m * (m - 1) // 2
    specified = int(mask.sum())
    if specified < dimension:
        raise KernelError("singular_matrix", "The specified targets are fewer than the rotation's tangent dimensions.")
    scaled_target = target / scale[:, None]
    matrix, iterations, converged, value, gradient = _fit(a,
        lambda z: partial_value(z, scaled_target, mask), oblique=oblique,
        max_iter=max_iter, tol=tol, what="partial-target")
    jacobian = tangent_jacobian(a @ matrix, matrix, mask, oblique=oblique)
    singular = torch.linalg.svdvals(jacobian)
    rank = int((singular > 1e-10 * singular.max().clamp_min(1.)).sum())
    if rank != dimension:
        raise KernelError("singular_matrix", "The specified targets do not locally identify the factor rotation.")
    jacobian_condition = _condition(jacobian, "specified-target tangent Jacobian")
    rotation, condition = _result(loadings, matrix, iterations, converged, oblique=oblique, canonical=False)
    diagnostics = _diagnostics(value, gradient, condition, rotation, tol)
    diagnostics.update({"rotation_objective": "half binary specified-target residual SS",
                        "rotation_orientation": "caller target column order/sign; local identification only",
                        "target_specified": specified, "target_tangent_dimension": dimension,
                        "target_jacobian_rank": rank, "target_jacobian_condition": jacobian_condition,
                        "target_mask_contract": "fixed binary mask; unspecified cells excluded",
                        "target_normalization": "target and loadings divided by loading row norm" if kaiser else "original loading metric"})
    # Persist ignored cells as NaN, making the specified/free distinction explicit.
    persisted = torch.where(mask, target, float("nan"))
    return RotationExtension(rotation, diagnostics, persisted, mask)
