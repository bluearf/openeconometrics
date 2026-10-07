"""Tensor kernels of the VAR family: system least squares, lag selection, diagnostics.

Notation. ``y`` is the [n, K] matrix of levels in time order and p the lag order.
The T = n - p usable rows are regressed on the same m regressors in every
equation, so one Householder QR of the [T, m] design solves all K equations:

    Y = Z B' + U,   B [K, m],   Sigma_ml = U'U / T,
    ln L = -(T/2) {ln|Sigma_ml| + K ln(2 pi) + K}.

Lagged values enter ``lag_block`` variable by variable (column v*p + (j-1) is lag
j of variable v), which is the order Stata reports them in.

Everything here works on float64 tensors, has no Python loop over observations
and never forms a T-by-T matrix. Failures raise ``KernelError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.linalg import (
    cholesky_solve, least_squares, symmetrize, weighted_crossprod,
)

_EPS = torch.finfo(torch.float64).eps
_LOG_2PI = math.log(2.0 * math.pi)
# An equation whose residual sum of squares is below this fraction of its outcome's sum of
# squares (residual norm below 1e-10 of the outcome norm) is fitted exactly up to rounding.
_EXACT_FIT = 1e-20


@dataclass
class SystemFit:
    coef: Tensor               # [K, m]; row i holds the coefficients of equation i
    xtx_inv: Tensor            # [m, m] (Z'Z)^-1
    resid: Tensor              # [T, K]
    fitted: Tensor             # [T, K]
    sigma_ml: Tensor           # [K, K] U'U / T
    logdet_ml: float           # ln |Sigma_ml|
    condition_number: float


@torch.no_grad()
def lag_block(y: Tensor, p: int) -> Tensor:
    """[n - p, K p] lagged levels; column v*p + (j-1) is lag j of variable v."""
    n, k = y.shape
    if p == 0:
        return y.new_empty((n, 0))
    return torch.stack([y[p - j:n - j] for j in range(1, p + 1)], dim=2).reshape(n - p, k * p)


@torch.no_grad()
def spd_factor(a: Tensor, *, code: str = "singular_residual_covariance",
               what: str = "residual covariance matrix") -> Tensor:
    """Lower Cholesky factor of a symmetric positive definite matrix.

    A pivot at the rounding level (L_jj^2 <= 8 k eps A_jj) is treated as zero,
    the rule of ``engines.linalg``: a numerically singular matrix raises
    instead of yielding a log determinant of order ln(eps).
    """
    k = a.shape[0]
    matrix = symmetrize(a)
    factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    lost = factor.diagonal().square() <= 8 * k * _EPS * matrix.diagonal()
    if int(info.item()) != 0 or not bool(torch.isfinite(factor).all()) or bool(lost.any()):
        raise KernelError(code, f"The {what} is singular: the endogenous variables (or their "
                          "residuals) are linearly dependent, or the model has as many parameters "
                          "per equation as observations. Remove the redundant variable or reduce "
                          "the number of lags.")
    return factor


def logdet(factor: Tensor) -> float:
    """ln |A| from the Cholesky factor of A."""
    return float(2.0 * factor.diagonal().log().sum())


def log_likelihood(logdet_ml: float, t: int, k: int) -> float:
    """Gaussian log likelihood at the ML covariance: -(T/2){ln|Sigma| + K ln(2 pi) + K}."""
    return -0.5 * t * (logdet_ml + k * _LOG_2PI + k)


@torch.no_grad()
def fit_system(z: Tensor, y: Tensor, names: list[str] | None = None) -> SystemFit:
    """Equation-by-equation OLS of the K columns of ``y`` on ``z`` from one QR.

    ``names`` label the equations in error messages. An equation that is
    fitted exactly (an endogenous variable that is a deterministic function of
    the regressors, such as a time trend or an accounting identity) is an
    error: its residual variance is rounding noise, and standard errors, the
    likelihood and every test would be meaningless.
    """
    t, k = y.shape
    if t <= z.shape[1]:
        raise KernelError("insufficient_observations", f"Each equation has {z.shape[1]} "
                          f"parameters but only {t} observations are usable. Reduce the number "
                          "of lags or variables.")
    # tol=0 turns off the kernel's uncentered collinearity screen: the caller has already
    # screened the design on mean-deviated columns, which is the right test for series
    # with a large level. An exactly singular design is still caught (omitted or by the
    # kernel's condition-number test).
    fit = least_squares(z, y, drop_collinear=True, tol=0.0)
    if fit.omitted:
        raise KernelError("collinear_system", "The regressors of the system are exactly "
                          "linearly dependent. Remove the redundant variable or reduce the "
                          "number of lags.")
    exact = fit.resid.square().sum(dim=0) <= _EXACT_FIT * y.square().sum(dim=0)
    if bool(exact.any()):
        labels = [names[i] if names else f"number {i + 1}" for i in exact.nonzero().flatten()
                  .tolist()]
        raise KernelError("perfect_fit", f"The equation(s) of {', '.join(labels)} are fitted "
                          "exactly: the variable is a deterministic function of the lagged "
                          "variables and the other regressors (for example a time trend or an "
                          "identity), so its disturbance has no variance. Remove it from y, or "
                          "pass it as an exogenous regressor.")
    sigma = symmetrize(fit.resid.T @ fit.resid / t)
    factor = spd_factor(sigma)
    return SystemFit(fit.beta.T.contiguous(), fit.xtx_inv, fit.resid, fit.fitted, sigma,
                     logdet(factor), fit.condition_number)


@torch.no_grad()
def robust_covariance(z: Tensor, resid: Tensor, xtx_inv: Tensor) -> Tensor:
    """Huber-White covariance of the stacked coefficients, without a small-sample factor.

    With s_t = u_t (x) z_t the score of observation t, the equation-major
    sandwich is (I (x) A) [sum_t s_t s_t'] (I (x) A), A = (Z'Z)^-1; block (i, j)
    is A [Z' diag(u_i u_j) Z] A. Only K (K + 1) / 2 weighted cross products of
    the design are formed, never the [T, K m] score matrix.
    """
    k, m = resid.shape[1], z.shape[1]
    out = z.new_zeros((k * m, k * m))
    for i in range(k):
        for j in range(i, k):
            meat = weighted_crossprod(z, resid[:, i] * resid[:, j])
            block = xtx_inv @ meat @ xtx_inv
            out[i * m:(i + 1) * m, j * m:(j + 1) * m] = block
            if j != i:
                out[j * m:(j + 1) * m, i * m:(i + 1) * m] = block.T
    return symmetrize(out)


@torch.no_grad()
def nested_logdets(z: Tensor, y: Tensor, sizes: list[int]) -> list[float]:
    """ln |Sigma_ml| of the regressions of ``y`` on the first c columns of ``z``, c in sizes.

    One QR of [Z, Y] gives every nested fit: with R its triangular factor, the
    residual cross product of the model on the first c columns is
    R[c:, Y]' R[c:, Y] (the part of Y outside the span of those columns).
    """
    t, m = z.shape
    k = y.shape[1]
    if t < m + k:
        raise KernelError("insufficient_observations", "Too few observations for the lag-order "
                          "statistics: reduce maxlag.")
    r = torch.linalg.qr(torch.cat([z, y], dim=1), mode="r").R
    total = r[:, m:].square().sum(dim=0)                  # y'y of every equation
    out = []
    for c in sizes:
        block = r[c:, m:]
        if bool((block.square().sum(dim=0) <= _EXACT_FIT * total).any()):
            raise KernelError("perfect_fit", "An equation is fitted exactly at one of the lag "
                              "orders: an endogenous variable is a deterministic function of "
                              "its regressors. Remove it from y or pass it as an exogenous "
                              "regressor.")
        out.append(logdet(spd_factor(block.T @ block / t)))
    return out


@torch.no_grad()
def lm_autocorrelation(z: Tensor, resid: Tensor, xtx_inv: Tensor, logdet_ml: float,
                       lags: int) -> list[float | None]:
    """Johansen's LM statistics for residual autocorrelation at lags 1..``lags`` (varlmar).

    For lag s the model is augmented with the residuals lagged s times (missing
    initial values set to zero, Davidson and MacKinnon 1993). With Sigma~_s the
    ML residual covariance of the augmented model,

        LM_s = (T - d - 0.5) ln(|Sigma^| / |Sigma~_s|),   d = m + K,

    d being the number of coefficients per equation of the augmented model.
    By the Frisch-Waugh theorem the augmented residual cross product is
    U'U - U'E (E'M_Z E)^-1 E'U, which needs only cross products with E.
    A statistic that cannot be formed (T - d - 0.5 <= 0) is ``None``.
    """
    t, k = resid.shape
    d = z.shape[1] + k
    total = resid.T @ resid
    out: list[float | None] = []
    for s in range(1, lags + 1):
        if s >= t or t - d - 0.5 <= 0:
            out.append(None)
            continue
        lagged = torch.zeros_like(resid)
        lagged[s:] = resid[:-s]
        cross = z.T @ lagged                                   # [m, K]
        inner = symmetrize(lagged.T @ lagged - cross.T @ xtx_inv @ cross)
        moment = lagged.T @ resid                              # E'U (U is orthogonal to Z)
        try:
            explained = moment.T @ cholesky_solve(inner, moment)
            augmented = logdet(spd_factor(symmetrize(total - explained) / t))
        except KernelError:
            out.append(None)
            continue
        out.append((t - d - 0.5) * (logdet_ml - augmented))
    return out


@torch.no_grad()
def normality_moments(resid: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Third and fourth moments of the Cholesky-orthogonalized residuals (varnorm).

    w_t = P^-1 u_t with P the lower Cholesky factor of ``sigma``; returns
    b1_k = mean(w_k^3) and b2_k = mean(w_k^4).
    """
    factor = spd_factor(sigma)
    w = torch.linalg.solve_triangular(factor, resid.T, upper=False).T
    return w.pow(3).mean(dim=0), w.pow(4).mean(dim=0)


@torch.no_grad()
def lag_matrices(coef: Tensor, k: int, p: int) -> Tensor:
    """[p, K, K] autoregressive matrices A_1..A_p from the coefficient rows.

    ``coef`` [K, >= K p] holds, for equation i, the coefficient of lag j of
    variable v in column v*p + (j-1).
    """
    return coef[:, :k * p].reshape(k, k, p).permute(2, 0, 1).contiguous()


@torch.no_grad()
def companion(a: Tensor) -> Tensor:
    """[K p, K p] companion matrix of A_1..A_p (first block row [A_1 ... A_p])."""
    p, k, _ = a.shape
    out = a.new_zeros((k * p, k * p))
    out[:k] = a.permute(1, 0, 2).reshape(k, k * p)
    if p > 1:
        out[k:, :k * (p - 1)] = torch.eye(k * (p - 1), dtype=a.dtype)
    return out


@torch.no_grad()
def companion_eigenvalues(a: Tensor) -> list[dict[str, float]]:
    """Eigenvalues of the companion matrix, largest modulus first."""
    if a.shape[0] == 0:
        return []
    try:
        values = torch.linalg.eigvals(companion(a))
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The companion eigenvalues failed: {exc}") from exc
    modulus = values.abs()
    order = torch.argsort(modulus, descending=True, stable=True)
    return [{"real": float(values[i].real), "imag": float(values[i].imag),
             "modulus": float(modulus[i])} for i in order.tolist()]


@torch.no_grad()
def ma_matrices(a: Tensor, steps: int) -> Tensor:
    """[steps + 1, K, K] moving-average matrices: Phi_0 = I, Phi_i = sum_j Phi_(i-j) A_j."""
    p, k, _ = a.shape
    out = a.new_zeros((steps + 1, k, k))
    out[0] = torch.eye(k, dtype=a.dtype)
    for i in range(1, steps + 1):
        width = min(i, p)
        # Phi_(i-1), ..., Phi_(i-width) against A_1, ..., A_width.
        out[i] = torch.einsum("jab,jbc->ac", out[i - width:i].flip(0), a[:width])
    if not bool(torch.isfinite(out).all()):
        raise KernelError("numerical_failure", "The moving-average coefficients overflow: the "
                          "model is explosive over this horizon. Reduce the number of steps.")
    return out
