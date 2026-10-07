"""Impulse responses, forecast-error variance decompositions and their standard errors.

For a VAR with autoregressive matrices A_1..A_p and innovation covariance
Sigma = P P' (P lower triangular, the Cholesky factor in the model's variable
order) the responses at horizon i are

    simple            Phi_i,    Phi_0 = I,  Phi_i = sum_(j<=min(i,p)) Phi_(i-j) A_j
    orthogonalized    Theta_i = Phi_i P
    generalized       Psi_i   = Phi_i Sigma diag(Sigma)^(-1/2)      (Pesaran and Shin 1998)

Element [i][r][c] is the response of variable r, i periods after a shock to
variable c (a unit shock for the simple responses, a one-standard-deviation
shock for the other two). The forecast-error variance decomposition is

    omega_(rc,h) = sum_(i<h) Theta_i[r,c]^2 / MSE_r(h),
    MSE_r(h) = sum_(i<h) (Phi_i Sigma Phi_i')[r,r],

the share of the h-step forecast-error variance of r due to orthogonalized
shocks in c; it is stored with omega_(.,0) = 0, as Stata's ``irf`` tables do.

Standard errors use the delta method (Lutkepohl 2005, section 3.7). With
alpha = vec(A_1, ..., A_p), sigma = vech(Sigma), M the companion matrix and
J = [I_K 0 ... 0],

    G_i   = d vec(Phi_i) / d alpha' = sum_(m<i) J (M')^(i-1-m) (x) Phi_m
    C_i   = (P' (x) I_K) G_i,   Cbar_i = (I_K (x) Phi_i) H,
    H     = d vec(P) / d sigma' = L_K' {L_K (I + K_KK) (P (x) I_K) L_K'}^-1,
    V(vec Phi_i)   = G_i V_alpha G_i'
    V(vec Theta_i) = C_i V_alpha C_i' + Cbar_i V_sigma Cbar_i',
    V_sigma        = (2 / T) D_K^+ (Sigma (x) Sigma) D_K^+'

(L_K elimination, D_K duplication, K_KK commutation matrix). The generalized
responses use the same G_i with Q = Sigma diag(Sigma)^(-1/2) in place of P and
the Jacobian of vec(Q) with respect to sigma. The variance decomposition is a
smooth function of Theta_0..Theta_(h-1), so its Jacobian is accumulated from
C_i and Cbar_i by the chain rule; this equals Lutkepohl's d and dbar vectors.

G_i is obtained from the recursion Ghat_(i+1) = (M' (x) I_K) Ghat_i + I (x) Phi_i
(G_i are its first K^2 rows), which costs O(steps) small products instead of
O(steps^2) Kronecker products.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.econometrics.var.kernels import ma_matrices, spd_factor
from openecon.engines.contracts import KernelError

KINDS = ("simple", "orthogonalized", "generalized")
# Standard errors need K^2 p by K^2 p work arrays; beyond this size they are skipped.
MAX_SE_PARAMETERS = 1500


@dataclass
class Responses:
    simple: Tensor                    # [steps + 1, K, K]
    orthogonalized: Tensor
    generalized: Tensor
    fevd: Tensor                      # [steps + 1, K, K], row 0 is zero
    se: dict[str, Tensor] | None      # the same keys (and "fevd"), or None when not computed


@torch.no_grad()
def elimination_matrix(k: int) -> Tensor:
    """L_K with vech(F) = L_K vec(F) (vech stacks the lower triangle column by column)."""
    rows = [(r, c) for c in range(k) for r in range(c, k)]
    out = torch.zeros((len(rows), k * k), dtype=torch.float64)
    for index, (r, c) in enumerate(rows):
        out[index, c * k + r] = 1.0
    return out


@torch.no_grad()
def duplication_matrix(k: int) -> Tensor:
    """D_K with vec(F) = D_K vech(F) for symmetric F."""
    rows = [(r, c) for c in range(k) for r in range(c, k)]
    out = torch.zeros((k * k, len(rows)), dtype=torch.float64)
    for index, (r, c) in enumerate(rows):
        out[c * k + r, index] = 1.0
        out[r * k + c, index] = 1.0
    return out


@torch.no_grad()
def commutation_matrix(k: int) -> Tensor:
    """K_KK with K_KK vec(G) = vec(G')."""
    out = torch.zeros((k * k, k * k), dtype=torch.float64)
    index = torch.arange(k * k)
    out[index, (index % k) * k + index // k] = 1.0
    return out


@torch.no_grad()
def alpha_positions(k: int, p: int, m: int) -> Tensor:
    """Positions of alpha = vec(A_1, ..., A_p) inside the equation-major coefficient vector.

    Element c*K + i of alpha (c = (j-1) K + v) is the coefficient of lag j of
    variable v in equation i, stored at i*m + v*p + (j-1).
    """
    c = torch.arange(k * p)
    lag, variable = c // k, c % k
    column = variable * p + lag                       # regressor index within an equation
    equation = torch.arange(k)
    return (equation[None, :] * m + column[:, None]).reshape(-1)


def _quadratic_diagonal(jacobian: Tensor, covariance: Tensor) -> Tensor:
    return ((jacobian @ covariance) * jacobian).sum(dim=1)


def _as_matrix(variance: Tensor, k: int) -> Tensor:
    """vec-ordered variances (column c, row r at c*K + r) as standard errors [r, c]."""
    return variance.clamp_min(0.0).sqrt().reshape(k, k).T


@torch.no_grad()
def impulse_responses(a: Tensor, sigma: Tensor, steps: int, *, cov_alpha: Tensor | None = None,
                      n_obs: int | None = None) -> Responses:
    """Responses and variance decomposition up to ``steps``; SEs when ``cov_alpha`` is given.

    ``a`` is [p, K, K]; ``cov_alpha`` the covariance of vec(A_1..A_p) and
    ``n_obs`` the T of the covariance of vech(Sigma).

    The computation runs in standardized units (each variable divided by its
    innovation standard deviation s), an exact reparameterization: A_j[i, v]
    becomes A_j[i, v] s_v / s_i and Sigma a correlation matrix. Simple
    responses map back by s_r / s_c, orthogonalized and generalized ones by
    s_r, and the variance decomposition is unit free. Variables measured on
    very different scales therefore do not affect the accuracy of the
    Jacobians.
    """
    p, k, _ = a.shape
    spd_factor(sigma)
    s = sigma.diagonal().sqrt()
    ratio = s[None, :] / s[:, None]
    if cov_alpha is not None and p:
        d = (s[:, None] / s[None, :]).expand(p, k, k).reshape(-1)     # index (j K + v) K + i
        cov_alpha = cov_alpha * d[:, None] * d[None, :]
    out = _standardized_responses(a * ratio, sigma / s[:, None] / s[None, :], steps,
                                  cov_alpha, n_obs)
    back = {"simple": 1.0 / ratio, "orthogonalized": s[:, None], "generalized": s[:, None]}
    for name, scale in back.items():
        setattr(out, name, getattr(out, name) * scale)
        if out.se is not None:
            out.se[name] = out.se[name] * scale
    return out


def _standardized_responses(a: Tensor, sigma: Tensor, steps: int, cov_alpha: Tensor | None,
                            n_obs: int | None) -> Responses:
    p, k, _ = a.shape
    phi = ma_matrices(a, steps)
    sigma = sigma.contiguous()
    factor = spd_factor(sigma).contiguous()
    scale = sigma.diagonal().rsqrt()
    q = sigma * scale[None, :]
    theta = phi @ factor
    psi = phi @ q
    squares = theta.square()
    mse = squares.sum(dim=2).cumsum(dim=0)                       # [steps + 1, K]: horizons 1..
    fevd = torch.zeros_like(theta)
    fevd[1:] = (squares.cumsum(dim=0) / mse[:, :, None])[:-1]
    if not all(bool(torch.isfinite(x).all()) for x in (theta, psi, fevd)):
        raise KernelError("numerical_failure", "The impulse responses overflow: the model is "
                          "explosive over this horizon. Reduce the number of steps.")
    if cov_alpha is None or n_obs is None or p == 0 or k * k * p > MAX_SE_PARAMETERS:
        return Responses(phi, theta, psi, fevd, None)

    c_size, s_size = k * k * p, k * (k + 1) // 2
    eye = torch.eye(k * k, dtype=torch.float64)
    elimination, duplication = elimination_matrix(k), duplication_matrix(k)
    d_plus = torch.linalg.solve(duplication.T @ duplication, duplication.T)
    cov_sigma = 2.0 / n_obs * d_plus @ torch.kron(sigma, sigma) @ d_plus.T
    inner = elimination @ (eye + commutation_matrix(k)) @ torch.kron(factor, torch.eye(
        k, dtype=torch.float64)) @ elimination.T
    h = torch.linalg.solve(inner.T, elimination).T                 # [K^2, S]
    # d vec(Q) / d vec(Sigma)' for Q = Sigma diag(Sigma)^(-1/2), then onto vech(Sigma).
    f = torch.zeros((k * k, k * k), dtype=torch.float64)
    index = torch.arange(k * k)
    column, row = index // k, index % k
    f[index, index] = scale[column]
    f[index, column * k + column] -= 0.5 * sigma[row, column] * scale[column].pow(3)
    q_sigma = f @ duplication                                      # [K^2, S]

    top = a.permute(1, 0, 2).reshape(k, k * p)                     # first block row of M
    state = torch.zeros((k * p, k, c_size), dtype=torch.float64)   # Ghat_i
    blocks = torch.arange(k * p)
    se = {name: torch.zeros_like(phi) for name in (*KINDS, "fevd")}
    sum_alpha = torch.zeros((k, k, c_size), dtype=torch.float64)   # sum_i Theta_i[r,c] dTheta_i
    sum_sigma = torch.zeros((k, k, s_size), dtype=torch.float64)
    for i in range(steps + 1):
        g = state[:k].reshape(k * k, c_size)                       # G_i (zero for i = 0)
        g3 = g.reshape(k, k, c_size)                               # [column, row, alpha]
        se["simple"][i] = _as_matrix(_quadratic_diagonal(g, cov_alpha), k)
        c_orth = torch.einsum("ac,arx->crx", factor, g3).reshape(k * k, c_size)
        cbar_orth = torch.einsum("rb,cbs->crs", phi[i], h.reshape(k, k, s_size)) \
            .reshape(k * k, s_size)
        se["orthogonalized"][i] = _as_matrix(
            _quadratic_diagonal(c_orth, cov_alpha) + _quadratic_diagonal(cbar_orth, cov_sigma), k)
        c_gen = torch.einsum("ac,arx->crx", q, g3).reshape(k * k, c_size)
        cbar_gen = torch.einsum("rb,cbs->crs", phi[i], q_sigma.reshape(k, k, s_size)) \
            .reshape(k * k, s_size)
        se["generalized"][i] = _as_matrix(
            _quadratic_diagonal(c_gen, cov_alpha) + _quadratic_diagonal(cbar_gen, cov_sigma), k)
        if i < steps:
            # Horizon h = i + 1 of the variance decomposition uses Theta_0..Theta_i.
            sum_alpha += theta[i][:, :, None] * c_orth.reshape(k, k, c_size).permute(1, 0, 2)
            sum_sigma += theta[i][:, :, None] * cbar_orth.reshape(k, k, s_size).permute(1, 0, 2)
            share = fevd[i + 1]
            weight = 2.0 / mse[i]
            jac_alpha = weight[:, None, None] * (
                sum_alpha - share[:, :, None] * sum_alpha.sum(dim=1, keepdim=True))
            jac_sigma = weight[:, None, None] * (
                sum_sigma - share[:, :, None] * sum_sigma.sum(dim=1, keepdim=True))
            variance = _quadratic_diagonal(jac_alpha.reshape(k * k, c_size), cov_alpha) \
                + _quadratic_diagonal(jac_sigma.reshape(k * k, s_size), cov_sigma)
            se["fevd"][i + 1] = variance.clamp_min(0.0).sqrt().reshape(k, k)
            # Ghat_(i+1) = (M' (x) I) Ghat_i + I (x) Phi_i, using the sparsity of M.
            new = torch.einsum("ar,asx->rsx", top, state[:k])
            if p > 1:
                new[:k * (p - 1)] += state[k:]
            view = new.reshape(k * p, k, k * p, k)
            view[blocks, :, blocks, :] += phi[i]
            state = new
    if not all(bool(torch.isfinite(x).all()) for x in se.values()):
        return Responses(phi, theta, psi, fevd, None)
    return Responses(phi, theta, psi, fevd, se)


@torch.no_grad()
def forecast_mse(phi: Tensor, sigma: Tensor) -> Tensor:
    """[steps, K, K] innovation part of the forecast MSE: sum_(i<h) Phi_i Sigma Phi_i'."""
    return (phi @ sigma @ phi.transpose(1, 2)).cumsum(dim=0)


@torch.no_grad()
def parameter_mse(transition: Tensor, gamma: Tensor, covariance: Tensor, phi: Tensor,
                  k: int) -> Tensor:
    """[steps, K, K] estimation-uncertainty part of the forecast MSE (Lutkepohl 2005, 3.5).

    With Z_t the regressor vector of period t+1, Z_(t+1) = B Z_t + ..., the h-step
    forecast has d y_t(h) / d vec(B)' = sum_(i<h) Z_t' (B')^(h-1-i) (x) Phi_i, and
    Stata's fcast compute averages the implied variance over the sample:

        (1/T) Omega(h) = (1/T) sum_t [sum_i Z_t'(B')^(h-1-i) (x) Phi_i] V [...]'
                       = sum_(l,l') sum_(i,j) (B^(h-1-i) Gamma B'^(h-1-j))[l,l']
                         Phi_i V[l,l'] Phi_j',        Gamma = Z'Z / T,

    where V[l,l'] is the K-by-K covariance between the coefficients of
    regressors l and l'. With Gamma = L L' the sum over i is carried by
    E_h[c, l] = sum_i (B^(h-1-i) L)[l, c] Phi_i, which obeys
    E_(h+1)[c, l] = sum_l'' B[l, l''] E_h[c, l''] + L[l, c] Phi_h.

    ``transition`` is B [m, m], ``covariance`` the equation-major [K m, K m]
    coefficient covariance and ``phi`` holds Phi_0..Phi_(steps-1).
    """
    m = transition.shape[0]
    steps = phi.shape[0]
    lower = spd_factor(gamma, code="singular_design", what="regressor moment matrix")
    blocks = covariance.reshape(k, m, k, m).permute(1, 3, 0, 2).contiguous()   # [l, l', i, j]
    state = lower.T[:, :, None, None] * phi[0]                                 # E_1[c, l]
    out = phi.new_zeros((steps, k, k))
    for h in range(steps):
        half = torch.einsum("clab,lmbd->cmad", state, blocks)
        out[h] = torch.einsum("cmad,cmed->ae", half, state)
        if h + 1 < steps:
            state = torch.einsum("lq,cqab->clab", transition, state) \
                + lower.T[:, :, None, None] * phi[h + 1]
    return (out + out.transpose(1, 2)) / 2
