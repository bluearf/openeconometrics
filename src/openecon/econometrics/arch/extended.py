"""Variance models outside Stata's ``arch``: component GARCH and FIGARCH.

These models are linear in the squared innovations s_t = e_t^2 at given parameters, so
the conditional variance and all of its derivative paths are obtained without any loop
over time. The mean equation (regressors and ARMA disturbances) and its residuals, the
presample variance h_0 and the innovation densities are those of ``kernels`` and
``densities``; the presample convention is the family's (Stata's ``arch0(xb)``): every
presample variance, squared innovation and variance component equals
h_0 = (1/T) sum_t e~_t^2, the mean squared residual at the current parameters, which is
differentiated with the mean parameters.

Component GARCH (Engle, R. F. and Lee, G. G. J. (1999), "A long-run and short-run
component model of stock return volatility", in Engle and White (eds.), Cointegration,
Causality and Forecasting, Oxford University Press, 475-497), in the parameterization of
rugarch's ``csGARCH`` (filters.c, ``csgarchfilter``):

    q_t = omega + rho q_(t-1) + phi (e_(t-1)^2 - h_(t-1))                  permanent
    h_t = q_t + alpha (e_(t-1)^2 - q_(t-1)) + beta (h_(t-1) - q_(t-1))     total variance

(Engle and Lee write the permanent equation as q_t = w + rho (q_(t-1) - w) + ..., i.e.
omega = w (1 - rho), w = omega / (1 - rho) the long-run variance.) With x_t = (h_t, q_t)
this is x_t = c + A x_(t-1) + B s_(t-1) with A = [[beta - phi, rho - alpha - beta],
[-phi, rho]], B = (phi + alpha, phi), c = (omega, omega). Eliminating q gives, exactly,

    h_t = omega (1 - alpha - beta) + (phi + alpha) s_(t-1) - c2 s_(t-2)
          + d1 h_(t-1) + d2 h_(t-2),
    d1 = beta - phi + rho,   d2 = phi (alpha + beta) - beta rho,
    c2 = alpha rho + phi (alpha + beta),

valid from t = 2 on; with presample x_0 = (h_0, h_0), s_0 = h_0 the first variance is
h_1 = omega + rho h_0 (= q_1). The kernel evaluates this second-order recursion with
``arima.filters.inverse_filter`` (cyclic reduction) and every derivative path with the
same filter; q follows from its own first-order filter. Tests check the path against the
two-state recursion written out in NumPy and rugarch's filter started from the same
state.

FIGARCH(1,d,1) (Baillie, R. T., Bollerslev, T. and Mikkelsen, H. O. (1996), "Fractionally
integrated generalized autoregressive conditional heteroskedasticity", Journal of
Econometrics 74, 3-30, eq. (8)-(9)): (1 - phi L)(1 - L)^d e_t^2 = omega + (1 - beta L) nu_t,
i.e. the ARCH(infinity) form

    h_t = omega / (1 - beta) + lambda(L) e_t^2,
    lambda(L) = 1 - (1 - phi L)(1 - L)^d / (1 - beta L) = sum_(i>=1) lambda_i L^i,

truncated at K lags (default 1000), in the parameterization and with the weight
recursion of Python ``arch``'s FIGARCH (``figarch_weights``):

    delta_1 = d,   delta_i = (i - 1 - d) / i delta_(i-1),
    lambda_1 = phi - beta + d,   lambda_i = beta lambda_(i-1) + delta_i - phi delta_(i-1).

Presample squared innovations equal h_0 (Python arch's ``backcast``), so

    h_t = omega / (1 - beta) + sum_(i<t) lambda_i e_(t-i)^2 + h_0 sum_(i=t..K) lambda_i.

The convolution with the K weights is a real FFT (no loop over time); the weights and
their derivatives with respect to (phi, d, beta) come from a cumulative product and a
first-order filter, so every derivative path is one more convolution.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.arch import densities
from openecon.econometrics.arch.kernels import (
    ArchLikelihood, Evaluation, _finite, distribution_parameters,
)
from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.arch.recursions import lagged
from openecon.econometrics.arima.filters import inverse_filter


class ExtendedLikelihood(ArchLikelihood):
    """Likelihood of a model linear in s_t = e_t^2 (see the module docstring).

    Subclasses implement ``_variance``; the mean equation, the density and the score
    assembly are shared. Per-observation derivative paths are always carried forward
    (the models have a handful of parameters).
    """

    def __init__(self, y: Tensor, x: Tensor, z: Tensor, layout: Layout):
        super().__init__(y, x, z, layout)
        self.linear = True

    def scan_lags(self) -> tuple[int, int]:          # no transition maps are stored
        return 0, 0

    def _variance(self, theta: Tensor, s: Tensor, h0: float, ds: Tensor | None,
                  dh0: Tensor | None, derivatives: bool):
        """``(h, d h / d variance parameters, d h / d mean parameters, state)`` or ``None``.

        ``ds`` is ``d s / d mean`` ([k_mean, T]) and ``dh0`` ``d h_0 / d mean``; the
        variance-parameter block is ordered as ``self.variance_index``.
        """
        raise NotImplementedError

    @property
    def variance_index(self) -> list[int]:
        ix = self.layout.index
        return [*ix.a, *ix.b, *ix.r, ix.c]

    @torch.no_grad()
    def evaluate(self, theta: Tensor, *, scores: bool = False, derivatives: bool = True,
                 engine: str | None = None, signs: Tensor | None = None,
                 active: list[int] | None = None, flat: Tensor | None = None,
                 ) -> Evaluation | None:
        layout, ix, n = self.layout, self.layout.index, self.n
        if not _finite(theta):
            return None
        tau = distribution_parameters(theta, ix)
        if not densities.feasible(layout.dist, tau):
            return None
        base = self._mean(theta, derivatives)
        if base is None:
            return None
        m, e, d0, h0, dh0 = base
        s = e.square()
        ds = 2.0 * e[None, :] * d0 if derivatives else None
        out = self._variance(theta, s, h0, ds, dh0, derivatives)
        if out is None:
            return None
        h, dh_var, dh_mean, state = out
        if not _finite(h) or not bool((h > 0.0).all()):
            return None
        loglik, d_e, d_log_h, d_tau = densities.log_density(layout.dist, e, h, tau, derivatives)
        value = float(loglik.sum())
        if not math.isfinite(value):
            return None
        evaluation = Evaluation(value, None, None, loglik, e, m, h, h0)
        evaluation.extra_state = state
        if not derivatives:
            return evaluation
        weight = d_log_h / h
        score = torch.zeros((ix.k, n), dtype=torch.float64)
        score[self.variance_index] = dh_var * weight
        mean_ix = ix.mean
        if mean_ix:
            score[mean_ix] = dh_mean * weight + d0 * d_e
        if ix.d:
            score[ix.d] = d_tau.reshape(len(ix.d), n)
        if not _finite(score):
            return None
        evaluation.gradient = score.sum(dim=1)
        evaluation.scores = score.T.contiguous() if scores else None
        return evaluation


class ComponentLikelihood(ExtendedLikelihood):
    """Engle-Lee component GARCH(1,1) (``model="cgarch"``)."""

    @property
    def variance_index(self) -> list[int]:
        ix = self.layout.index
        return [ix.a[0], ix.b[0], ix.r[0], ix.r[1], ix.c]

    def _variance(self, theta: Tensor, s: Tensor, h0: float, ds: Tensor | None,
                  dh0: Tensor | None, derivatives: bool):
        ix, n = self.layout.index, self.n
        alpha, beta = float(theta[ix.a[0]]), float(theta[ix.b[0]])
        rho, phi = float(theta[ix.r[0]]), float(theta[ix.r[1]])
        omega = float(theta[ix.c])
        d1 = beta - phi + rho
        d2 = phi * (alpha + beta) - beta * rho
        c2 = alpha * rho + phi * (alpha + beta)
        s1, s2 = lagged(s, 1, h0), lagged(s, 2, h0)
        forcing = omega * (1.0 - alpha - beta) + (phi + alpha) * s1 - c2 * s2
        forcing[0] = omega + rho * h0
        if n > 1:
            forcing[1] += d2 * h0
        poles = [-d1, -d2]                  # inverse_filter divides by 1 + c1 L + c2 L^2
        h = inverse_filter(forcing, poles)
        if not _finite(h) or not bool((h > 0.0).all()):
            return None
        # Permanent component: q_t = omega + rho q_(t-1) + phi (s_(t-1) - h_(t-1)).
        q_forcing = omega + phi * (s1 - lagged(h, 1, h0))
        q_forcing[0] += rho * h0
        q = inverse_filter(q_forcing, [-rho])
        state = {"permanent": q[-1:].tolist()}
        if not derivatives:
            return h, None, None, state
        h1, h2 = lagged(h, 1, 0.0), lagged(h, 2, 0.0)
        first = torch.zeros(n, dtype=torch.float64)
        first[0] = 1.0
        second = torch.zeros(n, dtype=torch.float64)
        if n > 1:
            second[1] = h0
        later = 1.0 - first                           # 1 from t = 1 on
        # Direct derivatives of the forcing plus d(d1) h_(t-1) + d(d2) h_(t-2).
        rows = [
            # alpha: c2' = rho + phi, d2' = phi
            later * (-omega + s1 - (rho + phi) * s2) + phi * second + phi * h2,
            # beta: c2' = phi, d1' = 1, d2' = phi - rho
            later * (-omega - phi * s2) + (phi - rho) * second + h1 + (phi - rho) * h2,
            # rho: c2' = alpha, d1' = 1, d2' = -beta; h_1 = omega + rho h_0
            first * h0 - later * alpha * s2 - beta * second + h1 - beta * h2,
            # phi: c2' = alpha + beta, d1' = -1, d2' = alpha + beta
            later * (s1 - (alpha + beta) * s2) + (alpha + beta) * second - h1
            + (alpha + beta) * h2,
            # omega
            first + later * (1.0 - alpha - beta),
        ]
        dh_var = inverse_filter(torch.stack(rows), poles)
        dh_mean = None
        if ds is not None and ds.shape[0]:
            dh0_column = dh0[:, None]
            g = (phi + alpha) * lagged(ds, 1, dh0_column) - c2 * lagged(ds, 2, dh0_column)
            g[:, 0] = rho * dh0
            if n > 1:
                g[:, 1] += d2 * dh0
            dh_mean = inverse_filter(g, poles)
        return h, dh_var, dh_mean, state


def figarch_weights(phi: float, d: float, beta: float, count: int,
                    derivatives: bool = False) -> tuple[Tensor, Tensor | None]:
    """lambda_1..lambda_K of FIGARCH(1,d,1) and, on request, d lambda / d (phi, d, beta).

    Returns ``(lam [K], dlam [3, K])`` with time index 0 for lambda_1.
    """
    i = torch.arange(1, count, dtype=torch.float64)            # 1 .. K-1
    ratios = (i - d) / (i + 1.0)
    product = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(ratios, 0)])
    delta = d * product                                          # delta_1 .. delta_K
    forcing = torch.empty(count, dtype=torch.float64)
    forcing[0] = phi - beta + d
    forcing[1:] = delta[1:] - phi * delta[:-1]
    lam = inverse_filter(forcing, [-beta]) if beta != 0.0 else forcing
    if not derivatives:
        return lam, None
    # d product / d d = product * sum_(j<=i) -1 / (j - d).
    slope = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(-1.0 / (i - d), 0)])
    d_delta = product + d * product * slope
    rows = torch.zeros((3, count), dtype=torch.float64)
    rows[0, 0], rows[0, 1:] = 1.0, -delta[:-1]                   # phi
    rows[1, 0], rows[1, 1:] = 1.0, d_delta[1:] - phi * d_delta[:-1]   # d
    rows[2, 0] = -1.0                                            # beta: direct part
    rows[2] += lagged(lam, 1, 0.0)                               # ... and lambda_(i-1)
    dlam = inverse_filter(rows, [-beta]) if beta != 0.0 else rows
    return lam, dlam


def _tail_sums(weights: Tensor, n: int) -> Tensor:
    """R_t = sum_(k >= t) weights_k (0-based) for t = 0..n-1: the presample share of h_t."""
    count = weights.shape[-1]
    tail = weights.flip(-1).cumsum(-1).flip(-1)
    out = torch.zeros((*weights.shape[:-1], n), dtype=torch.float64)
    reach = min(n, count)
    out[..., :reach] = tail[..., :reach]
    return out


def _lagged_convolution(weights: Tensor, signal: Tensor) -> Tensor:
    """c_t = sum_(k=0..K-1) weights_k signal_(t-1-k) (signal before 0 counts as zero).

    ``weights`` is ``[a, K]`` or ``[K]`` and ``signal`` ``[b, n]`` or ``[n]``; leading
    dimensions broadcast. Real FFT of length >= n + K.
    """
    n, count = signal.shape[-1], weights.shape[-1]
    size = 1 << (n + count).bit_length()
    product = torch.fft.rfft(weights, n=size) * torch.fft.rfft(signal, n=size)
    full = torch.fft.irfft(product, n=size)[..., :n]
    return lagged(full, 1, 0.0)


class FigarchLikelihood(ExtendedLikelihood):
    """FIGARCH(p,d,q), p, q in {0, 1} (``model="figarch"``)."""

    @property
    def variance_index(self) -> list[int]:
        ix = self.layout.index
        return [*ix.a, ix.r[0], *ix.b, ix.c]

    def parameters(self, theta: Tensor) -> tuple[float, float, float, float]:
        ix = self.layout.index
        phi = float(theta[ix.a[0]]) if ix.a else 0.0
        beta = float(theta[ix.b[0]]) if ix.b else 0.0
        return float(theta[ix.c]), phi, float(theta[ix.r[0]]), beta

    def _variance(self, theta: Tensor, s: Tensor, h0: float, ds: Tensor | None,
                  dh0: Tensor | None, derivatives: bool):
        ix, n = self.layout.index, self.n
        omega, phi, d, beta = self.parameters(theta)
        if not (0.0 < d < 1.0) or not beta < 1.0:
            return None
        lam, dlam = figarch_weights(phi, d, beta, self.layout.truncation, derivatives)
        if not _finite(lam):
            return None
        level = omega / (1.0 - beta)
        h = level + h0 * _tail_sums(lam, n) + _lagged_convolution(lam, s)
        keep = min(self.layout.truncation, n)
        state = {"squared_residuals": s[-keep:].tolist()}
        if not derivatives:
            return h, None, None, state
        through = h0 * _tail_sums(dlam, n) + _lagged_convolution(dlam, s)   # phi, d, beta
        rows = [through[0]] if ix.a else []
        rows.append(through[1])
        if ix.b:
            rows.append(through[2] + omega / (1.0 - beta) ** 2)
        rows.append(torch.full((n,), 1.0 / (1.0 - beta), dtype=torch.float64))
        dh_var = torch.stack(rows)
        dh_mean = None
        if ds is not None and ds.shape[0]:
            dh_mean = _lagged_convolution(lam, ds) + dh0[:, None] * _tail_sums(lam, n)
        return h, dh_var, dh_mean, state


def likelihood(y: Tensor, x: Tensor, z: Tensor, layout: Layout) -> ArchLikelihood:
    """The likelihood object of a layout: the extended models or the Stata family."""
    if layout.kind == "cgarch":
        return ComponentLikelihood(y, x, z, layout)
    if layout.kind == "figarch":
        return FigarchLikelihood(y, x, z, layout)
    return ArchLikelihood(y, x, z, layout)
