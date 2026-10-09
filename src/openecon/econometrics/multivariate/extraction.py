"""Factor extraction kernels on float64 tensors (correlation-matrix input).

All methods factor the p-by-p correlation matrix R as R ~ L L' + Psi with an
[p, m] loading matrix L and diagonal uniquenesses Psi.

Principal factors (pf). The diagonal of R is replaced by prior communalities,
the squared multiple correlations h_j^2 = 1 - 1 / r^jj (r^jj the diagonal of
R^{-1}); the reduced matrix is decomposed once and L = V_m sqrt(D_m).

Iterated principal factors (ipf, SPSS PAF). Starting from the squared multiple
correlations, the communalities are replaced by the row sums of squares of L
and the reduced matrix is decomposed again, until the largest change of a
communality is below the tolerance.

Principal-component factors (pcf). Communalities of one: L = V_m sqrt(D_m) of R.

Maximum likelihood (ml), Joreskog (1967). For given uniquenesses psi let
g_1 >= ... >= g_p and w_1, ..., w_p be the eigenvalues and eigenvectors of
Psi^{-1/2} R Psi^{-1/2}. The loadings that minimize the ML discrepancy

    F(L, Psi) = ln|Sigma| + tr(R Sigma^{-1}) - ln|R| - p,   Sigma = L L' + Psi,

for that Psi are L = Psi^{1/2} W_m (G_m - I)^{1/2}, and the concentrated function is

    F(psi) = sum_{j > m} (g_j - ln g_j - 1),
    dF/dpsi_i = -(1 / psi_i) sum_{j > m} (g_j - 1) w_ij^2.

F is minimized over theta_i = ln(psi_i - PSI_FLOOR) by BFGS with this analytic
gradient (``engines.optimize.maximize_bfgs`` applied to -F). PSI_FLOOR = 0.005 is
the lower bound on a uniqueness used by Joreskog's program and by R's factanal;
a uniqueness that ends at the bound is a Heywood case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from openecon.engines import optimize
from openecon.engines.contracts import KernelError

PSI_FLOOR = 0.005
_HEYWOOD = 1e-6          # psi - PSI_FLOOR below this counts as "at the bound"


@dataclass
class Extraction:
    loadings: Tensor               # [p, m]
    factored: Tensor               # eigenvalues of the matrix that was factored (descending)
    uniqueness: Tensor             # [p]
    iterations: int = 0
    converged: bool = True
    discrepancy: float | None = None
    heywood: list[int] = field(default_factory=list)


def _descending(a: Tensor) -> tuple[Tensor, Tensor]:
    values, vectors = torch.linalg.eigh((a + a.T) / 2)
    values, vectors = values.flip(0), vectors.flip(1)
    pivot = vectors.abs().argmax(0)
    sign = torch.sign(vectors[pivot, torch.arange(vectors.shape[1])])
    return values, vectors * torch.where(sign == 0, torch.ones_like(sign), sign)


def squared_multiple_correlations(r: Tensor) -> Tensor:
    """h_j^2 = 1 - 1 / r^jj; ``singular_matrix`` when R is not positive definite."""
    factor, info = torch.linalg.cholesky_ex(r, check_errors=False)
    p = r.shape[0]
    lost = factor.diagonal().square() <= 8 * p * torch.finfo(r.dtype).eps
    if int(info) != 0 or bool(lost.any()):
        raise KernelError("singular_matrix", "The correlation matrix is singular (a variable is "
                          "a linear combination of the others), so squared multiple "
                          "correlations are undefined. Remove the redundant variable, or use "
                          "method='pcf'.")
    inverse = torch.cholesky_inverse(factor)
    return (1.0 - 1.0 / inverse.diagonal()).clamp(0.0, 1.0)


def reduced_eigen(r: Tensor, communalities: Tensor) -> tuple[Tensor, Tensor]:
    reduced = r.clone()
    reduced.diagonal().copy_(communalities)
    return _descending(reduced)


def _loadings(values: Tensor, vectors: Tensor, m: int) -> Tensor:
    return vectors[:, :m] * values[:m].clamp_min(0.0).sqrt()


def principal_factors(r: Tensor, communalities: Tensor, m: int) -> Extraction:
    values, vectors = reduced_eigen(r, communalities)
    loadings = _loadings(values, vectors, m)
    return Extraction(loadings, values, 1.0 - loadings.square().sum(1))


def principal_component_factors(r: Tensor, m: int) -> Extraction:
    values, vectors = _descending(r)
    loadings = _loadings(values, vectors, m)
    return Extraction(loadings, values.clamp_min(0.0), 1.0 - loadings.square().sum(1))


def iterated_principal_factors(r: Tensor, communalities: Tensor, m: int, *, max_iter: int,
                               tol: float) -> Extraction:
    """Iterate communalities until max |change| < tol (SPSS PAF, Stata ipf)."""
    current = communalities
    for iteration in range(1, max_iter + 1):
        values, vectors = reduced_eigen(r, current)
        loadings = _loadings(values, vectors, m)
        updated = loadings.square().sum(1)
        change = float((updated - current).abs().max())
        current = updated
        if change < tol:
            # The reported eigenvalues belong to the matrix reduced by the final communalities.
            values, vectors = reduced_eigen(r, current)
            loadings = _loadings(values, vectors, m)
            heywood = torch.nonzero(current >= 1.0).flatten().tolist()
            return Extraction(loadings, values, 1.0 - loadings.square().sum(1), iteration, True,
                              heywood=heywood)
    raise KernelError("no_convergence", f"Iterated principal factors did not converge in "
                      f"{max_iter} iterations (last communality change {change:.2e}, tolerance "
                      f"{tol:g}). Raise max_iterations (the default 25 is SPSS's limit; slowly "
                      "converging solutions may need several hundred), extract fewer factors, "
                      "or use method='pf' or 'ml'.")


def ml_discrepancy(theta: Tensor, r: Tensor, m: int) -> tuple[Tensor, Tensor]:
    """(F, dF/dtheta) of the concentrated ML discrepancy at psi = PSI_FLOOR + exp(theta)."""
    free = torch.exp(theta)
    psi = PSI_FLOOR + free
    scale = psi.rsqrt()
    values, vectors = torch.linalg.eigh(r * torch.outer(scale, scale))
    tail, tail_vectors = values[:r.shape[0] - m], vectors[:, :r.shape[0] - m]
    if not bool((tail > 0).all()):
        nan = torch.full_like(theta, float("nan"))
        return torch.tensor(float("inf"), dtype=theta.dtype), nan
    value = (tail - torch.log(tail) - 1.0).sum()
    slope = -(tail_vectors.square() @ (tail - 1.0)) / psi
    return value, free * slope


def ml_loadings(psi: Tensor, r: Tensor, m: int) -> Tensor:
    """L = Psi^{1/2} W_m (G_m - I)^{1/2}, the conditional minimizer for given uniquenesses."""
    scale = psi.rsqrt()
    values, vectors = _descending(r * torch.outer(scale, scale))
    return psi.sqrt()[:, None] * vectors[:, :m] * (values[:m] - 1.0).clamp_min(0.0).sqrt()


def maximum_likelihood(r: Tensor, smc: Tensor, m: int, *, max_iter: int) -> Extraction:
    p = r.shape[0]
    if (p - m) ** 2 < p + m:
        raise KernelError("too_many_factors", f"A maximum-likelihood model with {m} factor(s) "
                          f"for {p} variables has negative degrees of freedom; extract fewer "
                          "factors.")
    # Joreskog's start: psi_i = (1 - m / (2p)) / r^ii, with 1 / r^ii = 1 - SMC_i.
    start = ((1.0 - 0.5 * m / p) * (1.0 - smc)).clamp_min(2.0 * PSI_FLOOR)
    theta0 = torch.log(start - PSI_FLOOR)

    def objective(theta: Tensor) -> tuple[Tensor, Tensor]:
        value, gradient = ml_discrepancy(theta, r, m)
        return -value, -gradient

    result = optimize.maximize_bfgs(objective, theta0, max_iter=max_iter,
                                    raise_on_failure=False)
    free = torch.exp(result.theta)
    bound = free < _HEYWOOD
    gradient_max = float(result.diagnostics["gradient_max"])
    # At a boundary (Heywood) solution the Hessian is singular in the bound direction,
    # so the optimizer cannot certify a maximum; the gradient test decides instead.
    if not result.converged and not (bool(bound.any()) and gradient_max <= 1e-6):
        raise KernelError("no_convergence", "Maximum-likelihood factor extraction did not "
                          f"converge ({result.diagnostics['message']}). Raise max_iterations, "
                          "extract fewer factors, or use method='ipf'.")
    psi = PSI_FLOOR + free
    psi = torch.where(bound, torch.full_like(psi, PSI_FLOOR), psi)
    loadings = ml_loadings(psi, r, m)
    discrepancy = max(float(-result.value), 0.0)
    if not math.isfinite(discrepancy) or not bool(torch.isfinite(loadings).all()):
        raise KernelError("numerical_failure", "Maximum-likelihood factor extraction produced "
                          "non-finite estimates.")
    order = torch.argsort(loadings.square().sum(0), descending=True, stable=True)
    loadings = loadings[:, order]
    return Extraction(loadings, loadings.square().sum(0), psi, result.iterations, True,
                      discrepancy, torch.nonzero(bound).flatten().tolist())


def minres_objective(loadings: Tensor, r: Tensor) -> tuple[Tensor, Tensor]:
    """Half the off-diagonal squared residual sum and its loading gradient."""
    residual = loadings @ loadings.T - r
    residual.diagonal().zero_()
    return residual.square().sum() / 2, 2 * residual @ loadings


def minimum_residual(r: Tensor, smc: Tensor, m: int, *, max_iter: int,
                     tolerance: float = 1e-7) -> Extraction:
    """Unweighted least-squares EFA; no ML fit test or hidden uniqueness bound."""
    p = r.shape[0]
    if (p - m) ** 2 < p + m:
        raise KernelError("too_many_factors", "Minimum-residual EFA needs nonnegative model degrees of freedom.")
    start = principal_factors(r, smc, m).loadings

    def objective(theta: Tensor):
        value, gradient = minres_objective(theta.reshape(p, m), r)
        return -value, -gradient.flatten()

    result = optimize.maximize_bfgs(objective, start.flatten(), max_iter=max_iter,
                                    raise_on_failure=False)
    loadings = result.theta.reshape(p, m)
    value, gradient = minres_objective(loadings, r)
    size = float(gradient.abs().max())
    if not bool(torch.isfinite(loadings).all()) or not math.isfinite(float(value)):
        raise KernelError("numerical_failure", "Minimum-residual EFA produced non-finite loadings.")
    if size > tolerance:
        raise KernelError("no_convergence", f"Minimum-residual EFA gradient {size:.2e} exceeds {tolerance:g}; raise max_iterations.")
    if float(torch.linalg.eigvalsh(result.hessian).max()) > 1e-6:
        raise KernelError("no_convergence", "Minimum-residual EFA reached a stationary saddle rather than a local minimum.")
    # Principal axes identify the rotation-equivalent minimizer for reporting.
    u, singular, _ = torch.linalg.svd(loadings, full_matrices=False)
    loadings = u * singular
    uniqueness = 1 - loadings.square().sum(1)
    values = reduced_eigen(r, 1 - uniqueness)[0]
    return Extraction(loadings, values, uniqueness, result.iterations, True, float(value),
                      torch.nonzero(uniqueness <= 0).flatten().tolist())
