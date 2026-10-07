"""Random-effects GLM likelihoods by adaptive Gauss-Hermite quadrature (tensors only).

Model
-----
For group j with rows i, a binary (logit, probit) or count (Poisson) outcome

    eta_ij = x_ij'b + offset_ij + sum_d z_ijd sigma_d v_jd,    v_j ~ N(0, I_q),

with q = 1 (random intercept) or q = 2 (an independent random slope as well)
and sigma_d = exp(s_d). The random effects u_jd = sigma_d v_jd are integrated
out on the STANDARDIZED scale v, so the quadrature nodes do not move when the
variances change and a vanishing variance shows as a flat likelihood.

Quadrature
----------
With Gauss-Hermite nodes a_k and weights W_k (product rule for q = 2) and the
group's adaptive location mu_j and Cholesky scale tau_j, v_jk = mu_j + sqrt(2)
tau_j a_k and

    L_j ~= sum_k exp(l_jk),
    l_jk = (q/2) ln 2 + ln|tau_j| + ln W_k + a_k'a_k - (q/2) ln 2 pi - v_jk'v_jk/2
           + sum_i ln f(y_ij | eta_ijk).

mean-variance adaptive (Stata's default ``mvaghermite``): mu_j and tau_j tau_j'
are the posterior mean and covariance of v_j, obtained by fixed-point
iteration started from the posterior mode and curvature (``mcaghermite``,
the mode-curvature rule, stops there); ``ghermite`` is the non-adaptive rule
(mu = 0, tau = I). The adaptive parameters are held fixed while the
likelihood is maximized and recomputed between Newton runs until the
estimates stop moving, as Stata does; value, gradient and Hessian are then
exact derivatives of the quadrature sum.

Derivatives
-----------
With p_jk = exp(l_jk)/L_j the posterior node weights, s_ik and h_ik the first
and second derivatives of ln f with respect to eta, g_jk = sum_i s_ik x_i and
c_jkd = u_jkd sum_i s_ik z_id,

    d ln L_j / db   = sum_k p_jk g_jk,          d ln L_j / ds_d = sum_k p_jk c_jkd
    d2 ln L_j       = sum_k p_jk [d2 l_jk + d l_jk d l_jk'] - (d ln L_j)(d ln L_j)'

where d2 l_jk has blocks sum_i h_ik x_i x_i', sum_i h_ik x_i z_id u_jkd and
sum_i h_ik z_id z_ie u_jkd u_jke + [d = e] c_jkd. Every per-observation sum is
an ``index_add_`` over the [N, K] node arrays; the node loop (K = 7 to 49
iterations) only builds the [G, K, p] group scores.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid

from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_sums
from openecon.engines.distributions import gauss_hermite

FAMILIES = ("logit", "probit", "poisson")
METHODS = ("mvaghermite", "mcaghermite", "ghermite")
# Largest number of (observation, node) pairs: each [N, K] float64 array is then at most
# 200 MB, and the derivatives hold a handful of them.
NODE_LIMIT = 25_000_000
_LOG_2PI = math.log(2 * math.pi)


def family_pieces(family: str, y: Tensor, eta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ln f(y | eta), d ln f/d eta and d2 ln f/d eta2 (elementwise, any shape of eta)."""
    if family == "logit":
        if eta.ndim > y.ndim:
            y = y[..., None]
        log_f = y * logsigmoid(eta) + (1 - y) * logsigmoid(-eta)
        mu = torch.sigmoid(eta)
        return log_f, y - mu, -mu * torch.sigmoid(-eta)
    if family == "probit":
        sign = 2 * y - 1
        if eta.ndim > y.ndim:
            sign = sign[..., None]
        log_cdf, ratio, curvature = mills_ratio(sign * eta)
        return log_cdf, sign * ratio, -curvature
    if family == "poisson":
        if eta.ndim > y.ndim:
            y = y[..., None]
        mu = torch.exp(eta)
        return y * eta - mu - torch.lgamma(y + 1), y - mu, -mu
    raise KernelError("invalid_option", f"Unknown family '{family}'.")


class PooledObjective:
    """Log likelihood of the pooled (no random effect) model, for starts and LR tests."""

    def __init__(self, x: Tensor, y: Tensor, family: str, offset: Tensor | None = None):
        self.x, self.y, self.family = x, y, family
        self.offset = torch.zeros_like(y) if offset is None else offset

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        log_f, score, curvature = family_pieces(self.family, self.y,
                                                self.x @ theta + self.offset)
        value = log_f.sum()
        if not bool(torch.isfinite(value)):
            k = len(theta)
            return (torch.tensor(-math.inf, dtype=torch.float64), torch.zeros(k, dtype=torch.float64),
                    -torch.eye(k, dtype=torch.float64))
        return value, self.x.T @ score, (self.x * curvature[:, None]).T @ self.x

    def value(self, theta: Tensor) -> Tensor:
        value = family_pieces(self.family, self.y, self.x @ theta + self.offset)[0].sum()
        return value if bool(torch.isfinite(value)) else torch.tensor(-math.inf,
                                                                      dtype=torch.float64)

    def mean(self, theta: Tensor) -> Tensor:
        eta = self.x @ theta + self.offset
        if self.family == "logit":
            return torch.sigmoid(eta)
        if self.family == "probit":
            return torch.special.ndtr(eta)
        return torch.exp(eta)


class RandomEffectsGLMM:
    """Quadrature log likelihood in theta = (b, s_1..s_q), s_d = ln sigma_d."""

    def __init__(self, x: Tensor, y: Tensor, z: Tensor, codes: Tensor, n_groups: int,
                 family: str, points: int, *, offset: Tensor | None = None,
                 method: str = "mvaghermite"):
        if family not in FAMILIES or method not in METHODS:
            raise KernelError("invalid_option", "Unknown family or integration method.")
        n, p = x.shape
        q = z.shape[1]
        if q not in (1, 2) or z.shape[0] != n or y.shape != (n,) or codes.shape != (n,):
            raise KernelError("invalid_design", "The random-effects inputs do not conform.")
        if n * points ** q > NODE_LIMIT:
            raise KernelError("design_too_large", f"{n} observations times {points ** q} "
                              "quadrature nodes exceed the working-memory limit of "
                              f"{NODE_LIMIT:.0e} node evaluations; lower intpoints.")
        self.x, self.y, self.z, self.codes, self.groups = x, y, z, codes, n_groups
        self.family, self.method, self.p, self.q = family, method, p, q
        self.offset = torch.zeros(n, dtype=torch.float64) if offset is None else offset
        self.size = p + q
        nodes, weights = gauss_hermite(points)
        if q == 1:
            grid = nodes[:, None]
            log_w = torch.log(weights)
        else:
            grid = torch.cartesian_prod(nodes, nodes)
            log_w = torch.log(weights)[:, None] + torch.log(weights)[None, :]
            log_w = log_w.reshape(-1)
        self.grid = grid                                                      # [K, q]
        self.node_const = (log_w + grid.square().sum(dim=1)
                           + 0.5 * q * (math.log(2) - _LOG_2PI))              # [K]
        self.xz = torch.cat([x, z], dim=1)
        self.mu = torch.zeros((n_groups, q), dtype=torch.float64)
        self.tau = torch.eye(q, dtype=torch.float64).expand(n_groups, q, q).clone()
        self.adaptations = 0

    # ---- nodes ------------------------------------------------------------------

    def _nodes(self) -> tuple[Tensor, Tensor]:
        """v_jk [G, K, q] and the node part of l_jk [G, K]."""
        v = self.mu[:, None, :] + math.sqrt(2) * torch.einsum("gab,kb->gka", self.tau, self.grid)
        log_det = torch.log(torch.diagonal(self.tau, dim1=1, dim2=2)).sum(dim=1)
        base = self.node_const[None, :] + log_det[:, None] - 0.5 * v.square().sum(dim=2)
        return v, base

    def _eta(self, theta: Tensor, u: Tensor) -> Tensor:
        """eta_ik [N, K] for random effects u [G, K, q] (already multiplied by sigma)."""
        eta = (self.x @ theta[:self.p] + self.offset)[:, None]
        for d in range(self.q):
            eta = eta + self.z[:, d:d + 1] * u[:, :, d][self.codes]
        return eta

    def _posterior(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Node weights p_jk, v_jk and the group log likelihoods."""
        v, base = self._nodes()
        sigma = torch.exp(theta[self.p:])
        log_f = family_pieces(self.family, self.y, self._eta(theta, v * sigma))[0]
        ell = base + group_sums(log_f, self.codes, self.groups)
        log_lik = torch.logsumexp(ell, dim=1)
        return torch.exp(ell - log_lik[:, None]), v, log_lik

    # ---- adaptive parameters -------------------------------------------------------

    def _mode(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """Posterior mode of v_j and the Cholesky factor of the inverse curvature there."""
        sigma = torch.exp(theta[self.p:])
        zs = self.z * sigma                                                   # [N, q]
        xb = self.x @ theta[:self.p] + self.offset
        v = torch.zeros((self.groups, self.q), dtype=torch.float64)
        eye = torch.eye(self.q, dtype=torch.float64)
        for _ in range(100):
            eta = xb + (zs * v[self.codes]).sum(dim=1)
            _, score, curvature = family_pieces(self.family, self.y, eta)
            gradient = group_sums(zs * score[:, None], self.codes, self.groups) - v
            outer = (zs[:, :, None] * zs[:, None, :] * curvature[:, None, None]).reshape(-1,
                                                                                    self.q ** 2)
            hessian = group_sums(outer, self.codes, self.groups).reshape(-1, self.q, self.q) - eye
            chol, info = torch.linalg.cholesky_ex(-hessian)
            if bool((info != 0).any()) or not bool(torch.isfinite(gradient).all()):
                raise KernelError("numerical_failure", "The posterior mode of a group's "
                                  "random effect could not be computed (non-finite curvature); "
                                  "check the scale of the regressors.")
            step = torch.cholesky_solve(gradient[:, :, None], chol)[:, :, 0]
            largest = step.abs().amax(dim=1, keepdim=True)
            step = step * torch.clamp(2.0 / largest.clamp_min(1e-300), max=1.0)
            v = v + step
            if float(largest.max()) < 1e-10:
                break
        eta = xb + (zs * v[self.codes]).sum(dim=1)
        curvature = family_pieces(self.family, self.y, eta)[2]
        outer = (zs[:, :, None] * zs[:, None, :] * curvature[:, None, None]).reshape(-1, self.q ** 2)
        information = eye - group_sums(outer, self.codes, self.groups).reshape(-1, self.q, self.q)
        # Cholesky factor of the posterior covariance (inverse curvature) from the factor of
        # the information: Sigma = (L L')^-1.
        factor, info = torch.linalg.cholesky_ex(information)
        if not bool((info == 0).all()):
            raise KernelError("numerical_failure", "The curvature of a group's posterior is "
                              "not positive definite.")
        scale, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(factor))
        if not bool((info == 0).all()):
            raise KernelError("numerical_failure", "The posterior covariance of a group's "
                              "random effect is not positive definite.")
        return v, scale

    def adapt(self, theta: Tensor, *, max_iter: int = 50, tol: float = 1e-9) -> None:
        """Set the adaptive locations and scales at theta (see the module docstring)."""
        if self.method == "ghermite":
            return
        if not bool(torch.isfinite(theta).all()):
            raise KernelError("invalid_parameters", "The parameters are not finite.")
        self.mu, self.tau = self._mode(theta)
        self.adaptations += 1
        if self.method == "mcaghermite":
            return
        for _ in range(max_iter):
            weights, v, _ = self._posterior(theta)
            mean = (weights[:, :, None] * v).sum(dim=1)
            centred = v - mean[:, None, :]
            cov = torch.einsum("gk,gka,gkb->gab", weights, centred, centred)
            chol, info = torch.linalg.cholesky_ex(cov)
            if bool((info != 0).any()):
                break
            change = max(float((mean - self.mu).abs().max()),
                         float((chol - self.tau).abs().max()))
            self.mu, self.tau = mean, chol
            if change < tol:
                break

    # ---- likelihood and derivatives -------------------------------------------------

    def value(self, theta: Tensor) -> Tensor:
        value = self._posterior(theta)[2].sum()
        return value if bool(torch.isfinite(value)) else torch.tensor(-math.inf,
                                                                      dtype=torch.float64)

    def group_log_likelihood(self, theta: Tensor) -> Tensor:
        return self._posterior(theta)[2]

    def _derivatives(self, theta: Tensor, hessian: bool):
        p, q = self.p, self.q
        v, base = self._nodes()
        sigma = torch.exp(theta[p:])
        u = v * sigma                                                         # [G, K, q]
        log_f, score, curvature = family_pieces(self.family, self.y, self._eta(theta, u))
        ell = base + group_sums(log_f, self.codes, self.groups)
        log_lik = torch.logsumexp(ell, dim=1)
        weights = torch.exp(ell - log_lik[:, None])                          # [G, K]
        nodes = weights.shape[1]
        # Group scores per node: T_jk = sum_i s_ik [x_i, z_i]  ->  g_jk and c_jk.
        blocks = torch.stack([group_sums(score[:, k:k + 1] * self.xz, self.codes, self.groups)
                              for k in range(nodes)], dim=1)                 # [G, K, p + q]
        g = blocks[:, :, :p]
        c = blocks[:, :, p:] * u
        d_ell = torch.cat([g, c], dim=2)                                      # [G, K, p + q]
        group_score = (weights[:, :, None] * d_ell).sum(dim=1)                # [G, p + q]
        value = log_lik.sum()
        gradient = group_score.sum(dim=0)
        if not hessian:
            return value, gradient, None, group_score
        w_obs = weights[self.codes]                                           # [N, K]
        hw = w_obs * curvature
        hbar = hw.sum(dim=1)
        k = p + q
        out = torch.zeros((k, k), dtype=torch.float64)
        out[:p, :p] = (self.x * hbar[:, None]).T @ self.x
        u_obs = [u[:, :, d][self.codes] for d in range(q)]                    # q x [N, K]
        for d in range(q):
            hu = (hw * u_obs[d]).sum(dim=1)
            out[:p, p + d] = self.x.T @ (self.z[:, d] * hu)
            out[p + d, :p] = out[:p, p + d]
            for e in range(d, q):
                huu = (hw * u_obs[d] * u_obs[e]).sum(dim=1)
                out[p + d, p + e] = (self.z[:, d] * self.z[:, e] * huu).sum()
                out[p + e, p + d] = out[p + d, p + e]
            out[p + d, p + d] = out[p + d, p + d] + gradient[p + d]
        root = (weights.sqrt()[:, :, None] * d_ell).reshape(-1, k)
        out = out + root.T @ root - group_score.T @ group_score
        return value, gradient, (out + out.T) / 2, group_score

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value, gradient, hessian, _ = self._derivatives(theta, True)
        if not (bool(torch.isfinite(value)) and bool(torch.isfinite(gradient).all())
                and bool(torch.isfinite(hessian).all())):
            k = self.size
            return (torch.tensor(-math.inf, dtype=torch.float64),
                    torch.zeros(k, dtype=torch.float64), -torch.eye(k, dtype=torch.float64))
        return value, gradient, hessian

    def group_scores(self, theta: Tensor) -> Tensor:
        return self._derivatives(theta, False)[3]

    def posterior_means(self, theta: Tensor) -> Tensor:
        """Empirical Bayes means of the random effects u_j (scale of the data), [G, q]."""
        weights, v, _ = self._posterior(theta)
        return (weights[:, :, None] * v).sum(dim=1) * torch.exp(theta[self.p:])
