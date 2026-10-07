"""Tensor kernels of the panel-data linear family.

All group work is O(N) through ``index_add_`` (``absorb.group_means``,
``covariance.group_sums``); no n-by-n matrix and no loop over observations
is ever formed. Panels are identified by dense int64 codes 0..n-1.

Variance components (Stata's xtreg, re; Swamy and Arora 1972)

    sigma_e^2 = RSS_within / (N - n - k_w)
    sigma_u^2 = max(0, RSS_between / (n - K_b) - sigma_e^2 / T_bar),   T_bar harmonic mean

with RSS_within from the within (fixed-effects) regression of the k_w
regressors that vary within panels and RSS_between from the unweighted OLS
of the panel means ybar_i on xbar_i and a constant (K_b = its rank). The
Baltagi-Chang (1994) unbalanced estimator (Stata's ``sa`` option) replaces
the harmonic-mean approximation by the exact expectation of the T_i-weighted
between sum of squares:

    sigma_u^2 = max(0, [RSS_between,T - (n - K_b) sigma_e^2] /
                       [N - tr((Z'PZ)^-1 Z'P~Z)]),
    Z'PZ = sum_i T_i zbar_i zbar_i',   Z'P~Z = sum_i T_i^2 zbar_i zbar_i'.

The GLS transform is y_it - theta_i ybar_i with
theta_i = 1 - sqrt(sigma_e^2 / (T_i sigma_u^2 + sigma_e^2)).

Random-effects Gaussian likelihood (Stata's xtreg, mle), per panel with
V_i = sigma_e^2 I + sigma_u^2 11', a_i = sigma_e^2 + T_i sigma_u^2,
S_i = sum_t r_it, W_i = sum_t (r_it - rbar_i)^2:

    ll_i = -(T_i/2) ln 2pi - ((T_i - 1)/2) ln sigma_e^2 - (1/2) ln a_i
           - W_i / (2 sigma_e^2) - S_i^2 / (2 T_i a_i)

because V_i^-1 = (1/sigma_e^2)[I - sigma_u^2/a_i 11'] and
|V_i| = sigma_e^(2(T_i-1)) a_i. The parameters are (beta, ln sigma_u, ln sigma_e);
gradient and Hessian are analytic (see ``RandomEffectsLikelihood``).

Breusch-Pagan LM test for random effects (Baltagi and Li 1990, Stata's xttest0),
on pooled OLS residuals e:

    LM = N^2 / (2 (sum_i T_i^2 - N)) * (sum_i (sum_t e_it)^2 / sum e_it^2 - 1)^2,

one-sided: chibar2(01) with the chi2(1) tail halved.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.absorb import group_means
from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_sums
from openecon.engines.linalg import cholesky_solve, symmetrize

_LOG_2PI = math.log(2 * math.pi)


def _codes(codes: Tensor, n: int, rows: int) -> None:
    if not isinstance(codes, Tensor) or codes.dtype != torch.int64 or codes.shape != (rows,):
        raise KernelError("invalid_codes", "Panel codes must be an int64 vector, one per row.")
    if rows and (int(codes.min()) < 0 or int(codes.max()) >= n):
        raise KernelError("invalid_codes", "Panel codes must lie in 0..n-1.")


@torch.no_grad()
def panel_sizes(codes: Tensor, n: int, multiplicity: Tensor | None = None) -> Tensor:
    """T_i: rows per panel as float64 [n], or the sum of ``multiplicity`` (frequency weights)."""
    _codes(codes, n, codes.numel())
    if multiplicity is None:
        return torch.bincount(codes, minlength=n).to(torch.float64)
    return group_sums(multiplicity, codes, n)


@torch.no_grad()
def constant_within_panel(values: Tensor, codes: Tensor, n: int) -> bool:
    """True when ``values`` takes a single value inside every panel (weights must)."""
    _codes(codes, n, values.numel())
    low = torch.full((n,), math.inf, dtype=torch.float64)
    high = torch.full((n,), -math.inf, dtype=torch.float64)
    low.scatter_reduce_(0, codes, values, "amin")
    high.scatter_reduce_(0, codes, values, "amax")
    present = low <= high
    return bool((high[present] == low[present]).all())


@torch.no_grad()
def correlation(a: Tensor, b: Tensor, weights: Tensor | None = None) -> float | None:
    """Weighted Pearson correlation, invariant to each vector's finite units.

    Normalize before centering so that finite large values cannot overflow a
    mean, then normalize the centered vectors separately. This also avoids
    squaring tiny values and comparing variances measured in unrelated units.
    """
    if a.numel() < 2:
        return None
    scale_a, scale_b = float(a.abs().max()), float(b.abs().max())
    if not math.isfinite(scale_a) or not math.isfinite(scale_b):
        raise KernelError("non_finite_values", "Correlation inputs must be finite.")
    if scale_a == 0 or scale_b == 0:
        return None
    a, b = a / scale_a, b / scale_b
    if weights is not None:
        scale_w = float(weights.abs().max())
        if not math.isfinite(scale_w) or bool((weights < 0).any()) or scale_w == 0:
            raise KernelError("invalid_weights", "Correlation weights must be finite, nonnegative and have a positive sum.")
        weights = weights / scale_w
        positive = weights > 0
        # Zero-weight observations do not create variation in the weighted
        # population; test this before a rounding residue in its mean.
        if bool((a[positive] == a[positive][0]).all()) or bool((b[positive] == b[positive][0]).all()):
            return None
    if weights is None:
        da, db = a - a.mean(), b - b.mean()
    else:
        total = float(weights.sum())
        da, db = a - float(weights @ a) / total, b - float(weights @ b) / total
    scale_da, scale_db = float(da.abs().max()), float(db.abs().max())
    if scale_da == 0 or scale_db == 0:
        return None
    da, db = da / scale_da, db / scale_db
    if weights is None:
        saa, sbb, sab = float(da @ da), float(db @ db), float(da @ db)
    else:
        saa, sbb = float(weights @ da.square()), float(weights @ db.square())
        sab = float(weights @ (da * db))
    if saa <= 0 or sbb <= 0:
        return None
    return max(-1.0, min(1.0, sab / math.sqrt(saa) / math.sqrt(sbb)))


def squared_correlation(a: Tensor, b: Tensor, weights: Tensor | None = None) -> float | None:
    """Squared correlation: Stata's R-squared within/between/overall."""
    rho = correlation(a, b, weights)
    return None if rho is None else rho * rho


@torch.no_grad()
def swamy_arora(ssr_within: float, df_within: float, ssr_between: float, df_between: float,
                sizes: Tensor) -> tuple[float, float, float]:
    """Harmonic-mean Swamy-Arora variance components (Stata's xtreg, re default).

    Returns (sigma_e^2, sigma_u^2, T_bar) with sigma_u^2 truncated at zero.
    """
    if df_within <= 0 or df_between <= 0:
        raise KernelError("insufficient_observations",
                          "The variance components need positive within and between degrees "
                          "of freedom (more observations than panels plus regressors, and more "
                          "panels than regressors plus one).")
    t_bar = sizes.numel() / float((1 / sizes).sum())
    sigma_e2 = ssr_within / df_within
    sigma_u2 = max(0.0, ssr_between / df_between - sigma_e2 / t_bar)
    return sigma_e2, sigma_u2, t_bar


@torch.no_grad()
def baltagi_chang_sigma_u(ssr_between_weighted: float, df_between: float, sigma_e2: float,
                          zbar: Tensor, sizes: Tensor) -> float:
    """Baltagi-Chang (1994) unbalanced Swamy-Arora sigma_u^2 (Stata's ``sa`` option).

    ``ssr_between_weighted`` is the residual sum of squares of the between
    regression weighted by T_i (equivalently OLS of P y on P Z over all N rows),
    ``zbar`` [n, K_b] the panel means of the between regressors kept in that
    regression (constant included) and ``sizes`` the T_i.
    """
    ztpz = (zbar * sizes[:, None]).T @ zbar
    ztppz = (zbar * sizes.square()[:, None]).T @ zbar
    trace = float(torch.trace(cholesky_solve(symmetrize(ztpz), ztppz, code="singular_design",
                                             what="between cross-product")))
    denominator = float(sizes.sum()) - trace
    if denominator <= 0:
        raise KernelError("insufficient_observations",
                          "The Baltagi-Chang denominator N - tr((Z'PZ)^-1 Z'P~Z) is not positive.")
    return max(0.0, (ssr_between_weighted - df_between * sigma_e2) / denominator)


@torch.no_grad()
def gls_theta(sizes: Tensor, sigma_u2: float, sigma_e2: float) -> Tensor:
    """theta_i = 1 - sqrt(sigma_e^2 / (T_i sigma_u^2 + sigma_e^2)), [n]."""
    if sigma_e2 <= 0:
        raise KernelError("numerical_failure", "sigma_e^2 must be positive for the GLS transform.")
    return 1 - torch.sqrt(sigma_e2 / (sizes * sigma_u2 + sigma_e2))


@torch.no_grad()
def breusch_pagan_lm(resid: Tensor, codes: Tensor, n: int,
                     multiplicity: Tensor | None = None) -> tuple[float, float]:
    """Baltagi-Li (1990) Breusch-Pagan LM statistic and its one-sided chibar2(01) p-value.

    ``multiplicity`` replicates rows (frequency weights). Returns (LM, p) with
    p = P(chi2(1) > LM) / 2, as Stata's xttest0 reports.
    """
    from openecon.engines.distributions import chi2_sf

    _codes(codes, n, resid.numel())
    m = torch.ones_like(resid) if multiplicity is None else multiplicity
    total_squares = float(m @ resid.square())
    if total_squares <= 0:
        raise KernelError("numerical_failure", "The pooled residuals are all zero.")
    sums = group_sums(resid * m, codes, n)
    sizes = group_sums(m, codes, n)
    count = float(m.sum())
    spread = float(sizes.square().sum()) - count
    if spread <= 0:
        raise KernelError("insufficient_observations",
                          "The Breusch-Pagan test needs panels with more than one observation.")
    lm = count * count / (2 * spread) * (float(sums.square().sum()) / total_squares - 1) ** 2
    return lm, 0.5 * chi2_sf(lm, 1)


@torch.no_grad()
def ols_log_likelihood(ssr: float, nobs: float) -> float:
    """Gaussian log likelihood of a least-squares fit: -N/2 (ln 2pi + ln(RSS/N) + 1)."""
    if ssr <= 0 or nobs <= 0:
        raise KernelError("numerical_failure",
                          "The log likelihood needs a positive residual sum of squares.")
    return -0.5 * nobs * (_LOG_2PI + math.log(ssr / nobs) + 1)


class RandomEffectsLikelihood:
    """Gaussian random-effects log likelihood in theta = (beta, ln sigma_u, ln sigma_e).

    Precomputes X'X, the panel means X-bar [n, K], T_i and the within cross
    product X~'X~ = X'X - sum_i T_i xbar_i xbar_i'. Each evaluation costs one
    pass over the N rows (residuals, their panel sums and sums of squares) and
    O(n K^2) for the mean part of the Hessian. No autograd: the derivatives in
    the module docstring are coded by hand and verified against numerical
    differentiation in the tests.
    """

    def __init__(self, x: Tensor, y: Tensor, codes: Tensor, n: int):
        rows, _ = x.shape
        _codes(codes, n, rows)
        self.x, self.y, self.codes, self.n = x, y, codes, n
        self.sizes = torch.bincount(codes, minlength=n).to(torch.float64)
        if bool((self.sizes <= 0).any()):
            raise KernelError("invalid_codes", "Every panel code must have at least one row.")
        self.means = group_means(x, codes, n)
        self.xtx = symmetrize(x.T @ x)
        self.within_xtx = symmetrize(self.xtx - (self.means * self.sizes[:, None]).T @ self.means)

    def _pieces(self, theta: Tensor):
        beta = theta[:-2]
        s_u, s_e = torch.exp(2 * theta[-2]), torch.exp(2 * theta[-1])
        r = self.y - self.x @ beta
        sums = group_sums(r, self.codes, self.n)
        squares = group_sums(r.square(), self.codes, self.n)
        sizes = self.sizes
        a = s_e + sizes * s_u
        within = (squares - sums.square() / sizes).clamp_min(0.0)
        value = (-0.5 * _LOG_2PI * sizes - 0.5 * (sizes - 1) * torch.log(s_e) - 0.5 * torch.log(a)
                 - within / (2 * s_e) - sums.square() / (2 * sizes * a)).sum()
        return r, sums, within, sizes, a, s_u, s_e, value

    def value(self, theta: Tensor) -> Tensor:
        """Log likelihood only: the cheap evaluation for rejected line-search trials."""
        return self._pieces(theta)[-1]

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(log likelihood, gradient [K+2], Hessian [K+2, K+2]) at theta.

        theta = (beta [K], ln sigma_u, ln sigma_e); the derivatives are with
        respect to these parameters (chain rule through sigma^2 = exp(2 ln sigma)).
        """
        r, sums, within, sizes, a, s_u, s_e, value = self._pieces(theta)
        xtr = self.x.T @ r
        xbar_s = self.means.T @ sums                      # sum_i T_i xbar_i rbar_i
        inv_a, inv_a2 = 1 / a, 1 / a.square()
        g_beta = xtr / s_e + self.means.T @ (sums * (inv_a - 1 / s_e))
        s2 = sums.square()
        g_e = (-(sizes - 1) - s_e * inv_a + within / s_e + s_e * s2 / sizes * inv_a2).sum()
        g_u = s_u * (-sizes * inv_a + s2 * inv_a2).sum()
        gradient = torch.cat([g_beta, g_u.reshape(1), g_e.reshape(1)])
        h_bb = -(self.within_xtx / s_e + (self.means * (sizes * inv_a)[:, None]).T @ self.means)
        h_be = -2 * (xtr - xbar_s) / s_e - 2 * s_e * (self.means.T @ (sums * inv_a2))
        h_bu = -2 * s_u * (self.means.T @ (sizes * sums * inv_a2))
        inv_a3 = inv_a2 * inv_a
        h_ee = (-2 * s_e * inv_a + 2 * s_e.square() * inv_a2 - 2 * within / s_e
                + 2 * s_e * s2 / sizes * inv_a2 - 4 * s_e.square() * s2 / sizes * inv_a3).sum()
        h_eu = 2 * s_u * (s_e * sizes * inv_a2 - 2 * s_e * s2 * inv_a3).sum()
        h_uu = 2 * g_u + 2 * s_u.square() * (
            sizes.square() * inv_a2 - 2 * sizes * s2 * inv_a3).sum()
        k = g_beta.numel()
        hessian = torch.empty((k + 2, k + 2), dtype=torch.float64)
        hessian[:k, :k] = h_bb
        hessian[:k, k], hessian[k, :k] = h_bu, h_bu
        hessian[:k, k + 1], hessian[k + 1, :k] = h_be, h_be
        hessian[k, k], hessian[k + 1, k + 1] = h_uu, h_ee
        hessian[k, k + 1] = hessian[k + 1, k] = h_eu
        return value, gradient, hessian
