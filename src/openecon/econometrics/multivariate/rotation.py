"""Factor rotation kernels on float64 tensors.

Notation. A is the [p, m] unrotated loading matrix. Every rotation returns a
matrix M with

    pattern  L = A M,          factor correlations  Phi = (M'M)^{-1},

so that the reproduced common part is unchanged: L Phi L' = A A'. For an
orthogonal rotation M is orthogonal and Phi = I; for an oblique rotation the
structure matrix (correlations between variables and factors) is L Phi.

Kaiser normalization (``normalize``) rotates the rows of A scaled to unit length
(divided by the square root of the communality) and scales the rotated rows
back; M is unaffected by the rescaling.

Orthomax family (varimax, quartimax, equamax). The criterion

    Q(L) = sum_j [ sum_i l_ij^4 - (gamma / p) (sum_i l_ij^2)^2 ]

is maximized over orthogonal M (gamma = 1 varimax, 0 quartimax, m/2 equamax) by
the SVD iteration: with G = dQ/dL / 4 = L^3 - (gamma/p) L diag(colsum(L^2)),
M <- U V' from the SVD A'G = U S V' (for gamma > 1, where that iteration can
cycle, by sweeps of Kaiser's pairwise planar rotations with their closed-form
angles). At a maximum L'G is symmetric; the iteration stops when the relative
asymmetry ||L'G - G'L|| / ||L'G|| falls below the tolerance.

Oblimin (direct oblimin; gamma = 0 is direct quartimin). The criterion

    f(L) = (1/4) sum_{s != t} [ sum_i l_is^2 l_it^2 - (gamma/p) sum_i l_is^2 sum_i l_it^2 ]

is minimized over oblique transformations (unit-length columns of T, L = A T'^{-1},
Phi = T'T) by Jennrich's (2002) gradient projection algorithm; it stops when
the norm of the projected gradient falls below the tolerance.

Promax (Hendrickson and White 1964). A varimax rotation V is followed by the
least-squares (Procrustes) fit of the power target P, p_ij = sign(v_ij) |v_ij|^kappa:
U = (V'V)^{-1} V'P with columns rescaled so that Phi = (U'U)^{-1} has a unit
diagonal. With Kaiser normalization the target is built from the row-normalized
varimax loadings, p_ij = sign(v_ij) |v_ij / h_i|^kappa, as documented for SPSS.

The tolerance used here (1e-10) is far tighter than the 1e-5 of SPSS and of
Jennrich's reference code, so rotated loadings are converged to all printed
digits; a rotation that reaches only 1e-5 within the iteration limit is
accepted and flagged, anything looser is a ``no_convergence`` error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError

TOLERANCE = 1e-10
ACCEPTABLE = 1e-5
ORTHOGONAL = ("varimax", "quartimax", "equamax")
OBLIQUE = ("promax", "oblimin")


@dataclass
class Rotation:
    pattern: Tensor            # [p, m] rotated loadings L = A M
    matrix: Tensor             # [m, m] M
    phi: Tensor                # [m, m] factor correlations (identity when orthogonal)
    iterations: int
    converged: bool            # reached TOLERANCE (otherwise only ACCEPTABLE)
    oblique: bool


def _row_scale(loadings: Tensor, normalize: bool) -> Tensor:
    if not normalize:
        return torch.ones(loadings.shape[0], dtype=loadings.dtype)
    scale = loadings.square().sum(1).sqrt()
    if bool((scale <= 1e-12 * scale.max().clamp_min(1e-300)).any()):
        raise KernelError("zero_communality", "Kaiser normalization divides each variable's "
                          "loadings by the square root of its communality, but a variable has "
                          "communality zero. Pass kaiser=False or drop the variable.")
    return scale


def _stopped(measure: float, iteration: int, max_iter: int, what: str) -> bool | None:
    """True/False = converged tightly / acceptably; None = keep iterating."""
    if measure <= TOLERANCE:
        return True
    if iteration < max_iter:
        return None
    if measure <= ACCEPTABLE:
        return False
    raise KernelError("no_convergence", f"The {what} rotation did not converge in {max_iter} "
                      f"iterations (criterion gradient {measure:.2e}). Raise max_iterations, "
                      "extract fewer factors or choose another rotation.")


def _asymmetry(z: Tensor, gamma: float) -> tuple[float, Tensor]:
    """Relative asymmetry of L'G (zero at a stationary point) and G = dQ/dL / 4."""
    g = z.pow(3) - (gamma / z.shape[0]) * z * z.square().sum(0)
    inner = z.T @ g
    return float(torch.linalg.matrix_norm(inner - inner.T)
                 / torch.linalg.matrix_norm(inner).clamp_min(1e-300)), g


def _planar_sweep(z: Tensor, rotation: Tensor, gamma: float) -> None:
    """One sweep of Kaiser's pairwise rotations, each maximizing the criterion exactly.

    For the pair of columns (x, y) with u = x^2 - y^2 and v = 2xy the optimal angle is
    phi = atan2(D - 2 gamma A B / p, C - gamma (A^2 - B^2) / p) / 4, where A = sum u,
    B = sum v, C = sum (u^2 - v^2) and D = 2 sum u v.
    """
    p, m = z.shape
    for j in range(m - 1):
        for k in range(j + 1, m):
            x, y = z[:, j].clone(), z[:, k].clone()
            u, v = x * x - y * y, 2.0 * x * y
            a_, b_ = float(u.sum()), float(v.sum())
            c_, d_ = float((u * u - v * v).sum()), 2.0 * float((u * v).sum())
            phi = 0.25 * math.atan2(d_ - 2.0 * gamma * a_ * b_ / p,
                                    c_ - gamma * (a_ * a_ - b_ * b_) / p)
            if abs(phi) < 1e-16:
                continue
            cos, sin = math.cos(phi), math.sin(phi)
            z[:, j], z[:, k] = cos * x + sin * y, cos * y - sin * x
            rj, rk = rotation[:, j].clone(), rotation[:, k].clone()
            rotation[:, j], rotation[:, k] = cos * rj + sin * rk, cos * rk - sin * rj


def orthomax_matrix(a: Tensor, gamma: float, max_iter: int, what: str
                    ) -> tuple[Tensor, int, bool]:
    """Orthogonal M maximizing the orthomax criterion of A M (A already row-scaled).

    The SVD iteration increases the criterion monotonically for gamma <= 1 (varimax,
    quartimax); for larger gamma (equamax with three or more factors) it can cycle,
    so Kaiser's pairwise planar rotations are used there.
    """
    m = a.shape[1]
    rotation = torch.eye(m, dtype=a.dtype)
    iteration = 0
    z = a.clone()
    while True:
        measure, g = _asymmetry(z, gamma)
        done = _stopped(measure, iteration, max_iter, what)
        if done is not None:
            return rotation, iteration, done
        if gamma <= 1.0:
            u, _, vh = torch.linalg.svd(a.T @ g)
            rotation = u @ vh
            z = a @ rotation
        else:
            _planar_sweep(z, rotation, gamma)
        iteration += 1


def _oblimin_value(pattern: Tensor, gamma: float) -> tuple[float, Tensor]:
    """Oblimin criterion f and its gradient with respect to the pattern matrix."""
    squares = pattern.square()
    other = squares.sum(1, keepdim=True) - squares         # L^2 (11' - I)
    if gamma != 0.0:
        other = other - (gamma / pattern.shape[0]) * other.sum(0, keepdim=True)
    return float((squares * other).sum()) / 4.0, pattern * other


def oblimin_matrix(a: Tensor, gamma: float, max_iter: int) -> tuple[Tensor, int, bool]:
    """M = T'^{-1} minimizing the oblimin criterion by gradient projection (Jennrich 2002)."""
    m = a.shape[1]
    t = torch.eye(m, dtype=a.dtype)

    def evaluate(trial: Tensor) -> tuple[Tensor, float, Tensor]:
        inverse = torch.linalg.inv(trial)
        pattern = a @ inverse.T
        value, gq = _oblimin_value(pattern, gamma)
        return pattern, value, -(pattern.T @ gq @ inverse).T

    _, value, gradient = evaluate(t)
    step, iteration = 1.0, 0
    noise = 64 * torch.finfo(a.dtype).eps
    while True:
        projected = gradient - t * (t * gradient).sum(0)
        size = float(torch.linalg.matrix_norm(projected))
        done = _stopped(size, iteration, max_iter, "oblimin")
        if done is not None:
            return torch.linalg.inv(t).T, iteration, done
        step *= 2.0
        for _ in range(60):
            trial = t - step * projected
            trial = trial / trial.square().sum(0).sqrt()
            try:
                _, new_value, new_gradient = evaluate(trial)
            except torch.linalg.LinAlgError:
                new_value = float("inf")
            required = 0.5 * size * size * step
            if required > noise * abs(value):
                if value - new_value > required:
                    break
            else:
                # The criterion no longer resolves a step of this size (its rounding
                # error exceeds the required decrease): the gradient decides instead.
                reduced = new_gradient - trial * (trial * new_gradient).sum(0)
                if float(torch.linalg.matrix_norm(reduced)) < size:
                    break
            step /= 2.0
        else:
            # No decrease at any step length: the projected gradient is at the
            # rounding level of the criterion.
            _stopped(size, max_iter, max_iter, "oblimin")
            return torch.linalg.inv(t).T, iteration, False
        t, value, gradient = trial, new_value, new_gradient
        iteration += 1


def promax_matrix(varimax: Tensor, target_base: Tensor, power: float) -> Tensor:
    """U with pattern = V U: least-squares fit of the power target, columns rescaled."""
    target = torch.sign(target_base) * target_base.abs().pow(power)
    u = torch.linalg.lstsq(varimax, target).solution
    gram = u.T @ u
    scale = torch.linalg.inv(gram).diagonal()
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise KernelError("singular_matrix", "The promax transformation is singular; "
                          "extract fewer factors or choose another rotation.")
    return u * scale.sqrt()


def rotate(loadings: Tensor, method: str, *, normalize: bool = True, power: float = 4.0,
           gamma: float = 0.0, max_iter: int = 1000) -> Rotation:
    """Rotate a [p, m] loading matrix; factors are then ordered and signed.

    The rotated factors are ordered by decreasing sum of squared pattern loadings
    and each is signed so that its loading of largest absolute value is positive.
    """
    p, m = loadings.shape
    identity = torch.eye(m, dtype=loadings.dtype)
    if m < 2:
        return Rotation(loadings.clone(), identity, identity, 0, True, method in OBLIQUE)
    scale = _row_scale(loadings, normalize)
    a = loadings / scale[:, None]
    if method in ORTHOGONAL or method == "promax":
        criterion = {"varimax": 1.0, "quartimax": 0.0, "equamax": m / 2.0}.get(method, 1.0)
        matrix, iterations, converged = orthomax_matrix(
            a, criterion, max_iter, "varimax" if method == "promax" else method)
        if method == "promax":
            # The regressors are the varimax loadings in their original metric; the
            # target comes from the row-normalized loadings under Kaiser normalization
            # (SPSS) and from the loadings themselves otherwise (R's promax).
            rotated = loadings @ matrix
            matrix = matrix @ promax_matrix(rotated, a @ matrix, power)
    elif method == "oblimin":
        matrix, iterations, converged = oblimin_matrix(a, gamma, max_iter)
    else:
        raise KernelError("invalid_option", f"Unknown rotation '{method}'.")
    pattern = loadings @ matrix
    order = torch.argsort(pattern.square().sum(0), descending=True, stable=True)
    pattern, matrix = pattern[:, order], matrix[:, order]
    pivot = pattern.abs().argmax(0)
    sign = torch.sign(pattern[pivot, torch.arange(m)])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    pattern, matrix = pattern * sign, matrix * sign
    if method in ORTHOGONAL:
        phi = identity
    else:
        phi = torch.linalg.inv(matrix.T @ matrix)
        phi = (phi + phi.T) / 2
        phi.diagonal().fill_(1.0)
    if not bool(torch.isfinite(pattern).all()) or not bool(torch.isfinite(phi).all()):
        raise KernelError("numerical_failure", "The rotation produced non-finite loadings.")
    return Rotation(pattern, matrix, phi, iterations, converged, method in OBLIQUE)
