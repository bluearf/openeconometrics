"""Tensor kernels of the linear system estimators on the compressed data.

Every equation enters as ``(X_i, y_i)`` in compressed form: ``p``-row blocks of
the factor ``R`` of ``sqrt(w) W`` (see ``common``), so that inner products of
the compressed columns equal the weighted inner products of the data. With
``h`` leading rows (``h = L``, the number of instrument columns) the same
inner products are those of the projections ``P_Z X_i`` and ``P_Z y_i``.

Equation by equation (OLS with ``h = p``, 2SLS with ``h = L``)
    ``b_i = argmin ||y_i[:h] - X_i[:h] b||^2`` by Householder QR, with
    ``(X_i'X_i)^-1`` (or ``(X_i'P_Z X_i)^-1``) from the same factor.

Stacked GLS (SUR with ``h = p``, 3SLS with ``h = L``)
    For the residual covariance ``Sigma = C C'`` (Cholesky) and ``T = C^-1``,
    ``sum_ij s^ij e_i'e_j = tr(Sigma^-1 E'E) = ||E T'||_F^2`` and column ``k`` of
    ``E T'`` is ``sum_{i<=k} T_ki e_i``. The GLS estimator therefore solves ONE
    least-squares problem with ``M`` row blocks: block ``k`` holds
    ``T_ki X_i[:h]`` in the columns of equation ``i`` and the target
    ``sum_i T_ki y_i[:h]``. Its QR gives

        b = (X'(Sigma^-1 (x) A) X)^-1 X'(Sigma^-1 (x) A) y,   V = (X'(Sigma^-1 (x) A) X)^-1,

    with ``A = I`` (SUR) or ``A = P_Z`` (3SLS), without ever forming a
    Kronecker product or an ``N``-row matrix. Linear constraints ``R b = r``
    enter as the exact reparameterization ``b = b_p + Q_2 g`` (``cnsreg``).

Cross-equation covariance of equation-by-equation OLS (``mvreg``)
    ``Cov(b_i, b_j) = s_ij (X_i'X_i)^-1 X_i'X_j (X_j'X_j)^-1 = s_ij A_i'A_j`` with
    ``A_i = X_i (X_i'X_i)^-1`` in compressed form.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.linalg import LeastSquares, least_squares


@dataclass
class EquationData:
    design: Tensor           # [p, k_i] compressed regressors (centred if the equation has a constant)
    target: Tensor           # [p] compressed outcome


@dataclass
class Restriction:
    """``b = particular + basis @ g`` in the parameterization the kernels estimate."""

    basis: Tensor            # [K, K - q]
    particular: Tensor       # [K]


@dataclass
class SystemFit:
    beta: Tensor             # [K] stacked over equations
    covariance: Tensor       # [K, K]
    sigma: Tensor            # [M, M] residual covariance used by the final estimation step
    resid: Tensor            # [p, M] compressed residuals at the final estimates
    iterations: int
    converged: bool


def _solve(design: Tensor, target: Tensor, what: str) -> LeastSquares:
    fit = least_squares(design, target, drop_collinear=True, tol=0.0)
    if fit.omitted:
        raise KernelError("singular_design", f"The {what} is rank deficient; remove redundant "
                          "regressors or constraints.")
    return fit


def cholesky_factor(sigma: Tensor) -> Tensor:
    """Lower Cholesky factor of a residual covariance; a rounding-level pivot is singular.

    LAPACK accepts many exactly singular matrices whose rounded pivot happens to
    be positive; a pivot ``L_jj^2 <= 8 m eps S_jj`` is treated as zero (the rule
    of ``engines.linalg``).
    """
    factor, info = torch.linalg.cholesky_ex(sigma)
    m = sigma.shape[0]
    eps = torch.finfo(torch.float64).eps
    if int(info) != 0 or not bool(torch.isfinite(factor).all()) \
            or bool((factor.diagonal().square() <= 8 * m * eps * sigma.diagonal()).any()):
        raise KernelError("singular_sigma", "The residual covariance matrix is singular: the "
                          "residuals of some equation are an exact combination of the others "
                          "(for example two identical equations). Remove the redundant "
                          "equation.")
    return factor


def offsets(equations: Sequence[EquationData]) -> list[int]:
    edges = [0]
    for equation in equations:
        edges.append(edges[-1] + equation.design.shape[1])
    return edges


def equation_fits(equations: Sequence[EquationData], rows: int | None) -> list[LeastSquares]:
    """OLS (``rows=None``) or 2SLS (``rows=L``) of every equation."""
    return [_solve(eq.design[:rows], eq.target[:rows], "equation design") for eq in equations]


def residuals(equations: Sequence[EquationData], beta: Tensor) -> Tensor:
    edges = offsets(equations)
    return torch.stack([eq.target - eq.design @ beta[edges[i]:edges[i + 1]]
                        for i, eq in enumerate(equations)], dim=1)


def stacked_gls(equations: Sequence[EquationData], sigma: Tensor, rows: int | None,
                restriction: Restriction | None = None) -> tuple[Tensor, Tensor]:
    """System GLS (see the module docstring); returns ``(beta, covariance)``."""
    factor = cholesky_factor(sigma)
    m = len(equations)
    transform = torch.linalg.solve_triangular(
        factor, torch.eye(m, dtype=torch.float64), upper=False)
    h = equations[0].design[:rows].shape[0]
    edges = offsets(equations)
    design = torch.zeros((m * h, edges[-1]), dtype=torch.float64)
    target = torch.zeros(m * h, dtype=torch.float64)
    for k in range(m):
        block = slice(k * h, (k + 1) * h)
        for i in range(k + 1):
            scale = float(transform[k, i])
            if scale == 0.0:
                continue
            design[block, edges[i]:edges[i + 1]] = scale * equations[i].design[:rows]
            target[block] += scale * equations[i].target[:rows]
    if restriction is None:
        fit = _solve(design, target, "stacked system")
        return fit.beta, fit.xtx_inv
    fit = _solve(design @ restriction.basis, target - design @ restriction.particular,
                 "constrained system")
    basis = restriction.basis
    return restriction.particular + basis @ fit.beta, basis @ fit.xtx_inv @ basis.T


def block_diagonal(fits: Sequence[LeastSquares], scales: Sequence[float]) -> Tensor:
    return torch.block_diag(*(fit.xtx_inv * scale for fit, scale in zip(fits, scales,
                                                                         strict=True)))


def cross_covariance(equations: Sequence[EquationData], fits: Sequence[LeastSquares],
                     sigma: Tensor) -> Tensor:
    """``s_ij (X_i'X_i)^-1 X_i'X_j (X_j'X_j)^-1`` for every pair (equation-by-equation OLS)."""
    loadings = [eq.design @ fit.xtx_inv for eq, fit in zip(equations, fits, strict=True)]
    rows = []
    for i, left in enumerate(loadings):
        rows.append(torch.cat([float(sigma[i, j]) * (left.T @ right)
                               for j, right in enumerate(loadings)], dim=1))
    covariance = torch.cat(rows, dim=0)
    return (covariance + covariance.T) / 2


def iterate_gls(equations: Sequence[EquationData], start: Tensor, sigma_of: Callable[[Tensor],
                Tensor], rows: int | None, restriction: Restriction | None, *, iterate: bool,
                tolerance: float, max_iterations: int) -> SystemFit:
    """FGLS from the first-step residuals, optionally iterated to convergence.

    ``sigma_of(resid)`` maps compressed residuals ``[p, M]`` to the residual
    covariance (it applies the divisor convention). Convergence of the
    iterated estimator: ``max_j |b_j - b_j_old| / (|b_j_old| + 1) <= tolerance``.
    """
    sigma = sigma_of(residuals(equations, start))
    beta, covariance = stacked_gls(equations, sigma, rows, restriction)
    iterations, converged = 1, not iterate
    while iterate:
        new_sigma = sigma_of(residuals(equations, beta))
        try:
            new_beta, new_covariance = stacked_gls(equations, new_sigma, rows, restriction)
        except KernelError as exc:
            if exc.code != "singular_sigma":
                raise
            raise KernelError("singular_sigma", f"The iterated estimator drove the residual "
                              f"covariance to singularity after {iterations} iteration(s): the "
                              "criterion it maximizes is unbounded for this system (typically "
                              "an outcome that also appears as a regressor of another equation "
                              "while every variable is treated as exogenous). Use the two-step "
                              "estimator, or 3SLS for simultaneous equations.") from exc
        iterations += 1
        change = float(((new_beta - beta).abs() / (beta.abs() + 1)).max())
        beta, covariance, sigma = new_beta, new_covariance, new_sigma
        if change <= tolerance:
            converged = True
            break
        if iterations >= max_iterations:
            break
    return SystemFit(beta, covariance, sigma, residuals(equations, beta), iterations, converged)
