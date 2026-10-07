"""Parametric survival likelihoods of Stata's ``streg`` on float64 tensors.

Every model has a main linear index ``mu_i = x_i'b + offset_i`` and (except the
exponential) an ancillary index ``a_i = z_i'g`` (a constant unless ``ancillary``
columns are given); the generalized gamma adds the scalar ``kappa``. With exit
time t, entry t0 (left truncation) and failure indicator delta,

    l_i = delta_i ln h(t_i) + ln S(t_i) - ln S(t0_i)          (ln S(0) = 0)

on the time scale; it is maximized as written. Stata reports the log likelihood of
``ln t`` instead, which adds the parameter-free ``sum_i w_i delta_i ln t_i``
(``streg`` applies it to the reported values). Parameterizations (Stata [ST] streg):

==============  ======  =================================================  ==========
distribution    metric  survivor function                                  ancillary
==============  ======  =================================================  ==========
exponential     PH      exp(-exp(mu) t)                                    --
exponential     AFT     exp(-exp(-mu) t)                                   --
weibull         PH      exp(-exp(mu) t^p),  p = exp(a)                     /ln_p
weibull         AFT     exp(-(exp(-mu) t)^p)                               /ln_p
gompertz        PH      exp(-exp(mu) (exp(gamma t) - 1) / gamma), gamma=a  /gamma
lognormal       AFT     1 - Phi((ln t - mu) / sigma), sigma = exp(a)       /lnsigma
loglogistic     AFT     1 / (1 + exp((ln t - mu) / gamma)), gamma = exp(a) /lngamma
ggamma          AFT     1 - I(g, u) (kappa > 0), I(g, u) (kappa < 0),      /lnsigma,
                        lognormal at kappa = 0; g = |kappa|^-2,            /kappa
                        z = sign(kappa)(ln t - mu)/sigma, u = g exp(|kappa| z)
==============  ======  =================================================  ==========

Derivatives: for every closed-form model the first and second derivatives of
``l_i`` with respect to ``(mu_i, a_i)`` are written out below and assembled into
the score ``X' (w l_mu)`` and the Hessian blocks ``X' diag(w l_mu mu) X`` ...;
O(n k^2) per iteration and no autograd. The generalized gamma, whose survivor
function is the regularized incomplete gamma function, has analytic derivatives in
``(mu_i, a_i)`` and five-point central differences in ``kappa`` (the shape of the
incomplete gamma function; see ``ggamma``; recorded in the result's provenance).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.survival.data import FLOAT
from openecon.engines.contracts import KernelError

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
DISTRIBUTIONS = ("exponential", "weibull", "gompertz", "lognormal", "loglogistic", "ggamma")
ANCILLARY = {"weibull": "ln_p", "gompertz": "gamma", "lognormal": "lnsigma",
             "loglogistic": "lngamma", "ggamma": "lnsigma"}
# |gamma t| below which the Gompertz G-functions use their power series.
_GOMPERTZ_SERIES = 0.5


def _gompertz_g(gamma: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """G = (exp(g t) - 1)/g and its first two derivatives in g (series near g t = 0)."""
    x = gamma * t
    small = x.abs() < _GOMPERTZ_SERIES
    safe = torch.where(small, torch.ones_like(gamma), gamma)
    e = torch.exp(x)
    g0 = torch.expm1(x) / safe
    g1 = (t * e - g0) / safe
    g2 = (t * t * e - 2 * g1) / safe
    # series in x = g t: G = t sum_j x^j/(j+1)!, G' = t^2 sum_j j x^(j-1)/(j+1)!,
    # G'' = t^3 sum_j j (j-1) x^(j-2)/(j+1)!
    s0, s1, s2 = torch.zeros_like(t), torch.zeros_like(t), torch.zeros_like(t)
    power = torch.ones_like(t)
    for j in range(25):
        s0 = s0 + power / float(math.factorial(j + 1))
        s1 = s1 + (j + 1) * power / float(math.factorial(j + 2))
        s2 = s2 + (j + 2) * (j + 1) * power / float(math.factorial(j + 3))
        power = power * x
    s0, s1, s2 = t * s0, t * t * s1, t * t * t * s2
    return (torch.where(small, s0, g0), torch.where(small, s1, g1), torch.where(small, s2, g2))


def _standard(kind: str, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """ln f, its z-derivatives, ln S and its z-derivatives of the standardized error."""
    if kind == "normal":
        log_s = torch.special.log_ndtr(-z)
        mills = torch.exp(-0.5 * z * z - LOG_SQRT_2PI - log_s)
        return (-0.5 * z * z - LOG_SQRT_2PI, -z, -torch.ones_like(z), log_s, -mills,
                -mills * (mills - z))
    p = torch.sigmoid(z)
    soft = torch.nn.functional.softplus(z)
    return (z - 2 * soft, 1 - 2 * p, -2 * p * (1 - p), -soft, -p, -p * (1 - p))


class Pieces:
    """Per-observation log likelihood and derivatives w.r.t. the per-observation indices."""

    def __init__(self, value: Tensor, grad: list[Tensor], hess: list[list[Tensor]]):
        self.value, self.grad, self.hess = value, grad, hess


class SurvivalData:
    """Times of the parametric likelihood (log times precomputed)."""

    def __init__(self, t: Tensor, t0: Tensor | None, delta: Tensor):
        self.t, self.delta = t, delta
        self.log_t = torch.log(t)
        self.truncated = None if t0 is None or not bool((t0 > 0).any()) else t0 > 0
        self.t0 = torch.zeros_like(t) if t0 is None else t0
        self.log_t0 = torch.log(torch.where(self.t0 > 0, self.t0, torch.ones_like(self.t0)))


def closed_form(dist: str, metric: str, data: SurvivalData, mu: Tensor,
                a: Tensor | None) -> Pieces:
    """``l_i`` and its derivatives in (mu, a) for the closed-form distributions."""
    d, lt, lt0 = data.delta, data.log_t, data.log_t0
    trunc = data.truncated
    zero = torch.zeros_like(mu)

    def masked(values: Tensor) -> Tensor:
        return zero if trunc is None else torch.where(trunc, values, zero)

    if dist in {"exponential", "weibull"}:
        p = torch.ones_like(mu) if a is None else torch.exp(a)
        log_p = zero if a is None else a
        if metric == "ph":
            big = torch.exp(mu + p * lt)
            big0 = masked(torch.exp(mu + p * lt0))
            value = d * (log_p + (p - 1) * lt + mu) - big + big0
            l_mu = d - big + big0
            l_mm = -big + big0
            if a is None:
                return Pieces(value, [l_mu], [[l_mm]])
            plt, plt0 = p * lt, masked(p * lt0)
            l_a = d * (1 + plt) - plt * big + plt0 * big0
            l_ma = -plt * big + plt0 * big0
            l_aa = d * plt - plt * big * (1 + plt) + plt0 * big0 * (1 + plt0)
            return Pieces(value, [l_mu, l_a], [[l_mm, l_ma], [l_ma, l_aa]])
        u = p * (lt - mu)
        u0 = masked(p * (lt0 - mu))
        e, e0 = torch.exp(u), masked(torch.exp(u0))
        value = d * (log_p - lt + u) - e + e0
        l_mu = -p * (d - e + e0)
        l_mm = -p * p * (e - e0)
        if a is None:
            return Pieces(value, [l_mu], [[l_mm]])
        l_a = d * (1 + u) - u * e + u0 * e0
        l_ma = -p * (d - e + e0 - u * e + u0 * e0)
        l_aa = d * u - u * e * (1 + u) + u0 * e0 * (1 + u0)
        return Pieces(value, [l_mu, l_a], [[l_mm, l_ma], [l_ma, l_aa]])
    if dist == "gompertz":
        gamma = a
        scale = torch.exp(mu)
        g0, g1, g2 = _gompertz_g(gamma, data.t)
        h0, h1, h2 = _gompertz_g(gamma, data.t0) if trunc is not None else (zero, zero, zero)
        value = d * (mu + gamma * data.t) - scale * (g0 - h0)
        l_mu = d - scale * (g0 - h0)
        l_a = d * data.t - scale * (g1 - h1)
        return Pieces(value, [l_mu, l_a], [[-scale * (g0 - h0), -scale * (g1 - h1)],
                                           [-scale * (g1 - h1), -scale * (g2 - h2)]])
    kind = "normal" if dist == "lognormal" else "logistic"
    sigma = torch.exp(a)
    z = (lt - mu) / sigma
    lf, f1, f2, ls, s1, s2 = _standard(kind, z)
    first = d * f1 + (1 - d) * s1
    second = d * f2 + (1 - d) * s2
    value = d * (lf - a - lt) + (1 - d) * ls
    l_mu, l_mm = -first / sigma, second / sigma ** 2
    l_a = -first * z - d
    l_ma = second * z / sigma + first / sigma
    l_aa = second * z * z + first * z
    if trunc is not None:
        z0 = torch.where(trunc, (lt0 - mu) / sigma, torch.full_like(z, -40.0))
        _, _, _, ls0, g1, g2 = _standard(kind, z0)
        value = value - masked(ls0)
        g1, g2 = masked(g1), masked(g2)
        l_mu = l_mu + g1 / sigma
        l_mm = l_mm - g2 / sigma ** 2
        l_a = l_a + g1 * z0
        l_ma = l_ma - g2 * z0 / sigma - g1 / sigma
        l_aa = l_aa - g2 * z0 * z0 - g1 * z0
    return Pieces(value, [l_mu, l_a], [[l_mm, l_ma], [l_ma, l_aa]])


class ParametricObjective:
    """``sum_i w_i l_i`` with analytic gradient and Hessian over (b, g[, kappa]).

    ``x`` is the main design (with its constant), ``z`` the ancillary design or
    None (exponential). ``pieces(mu, a, kappa)`` supplies the per-observation
    derivatives; the generalized gamma passes its numerical version.
    """

    def __init__(self, dist: str, metric: str, x: Tensor, z: Tensor | None, data: SurvivalData,
                 weights: Tensor, offset: Tensor | None):
        if dist not in DISTRIBUTIONS:
            raise KernelError("invalid_option", f"Unknown distribution {dist!r}.")
        self.dist, self.metric, self.x, self.z, self.data = dist, metric, x, z, data
        self.w, self.offset = weights, offset
        self.k = x.shape[1]
        self.q = 0 if z is None else z.shape[1]
        self.extra = 1 if dist == "ggamma" else 0
        self.size = self.k + self.q + self.extra

    def indices(self, theta: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        mu = self.x @ theta[:self.k]
        if self.offset is not None:
            mu = mu + self.offset
        a = None if self.z is None else self.z @ theta[self.k:self.k + self.q]
        kappa = theta[-1] if self.extra else None
        return mu, a, kappa

    def pieces(self, theta: Tensor) -> Pieces:
        mu, a, kappa = self.indices(theta)
        if self.dist == "ggamma":
            from openecon.econometrics.survival.ggamma import ggamma_pieces

            return ggamma_pieces(self.data, mu, a, kappa)
        return closed_form(self.dist, self.metric, self.data, mu, a)

    def observation_values(self, theta: Tensor) -> Tensor:
        mu, a, kappa = self.indices(theta)
        if self.dist == "ggamma":
            from openecon.econometrics.survival.ggamma import ggamma_loglik

            return ggamma_loglik(self.data, mu, a, kappa)
        return closed_form(self.dist, self.metric, self.data, mu, a).value

    def value(self, theta: Tensor) -> Tensor:
        return (self.w * self.observation_values(theta)).sum()

    def _blocks(self) -> list[Tensor | None]:
        blocks: list[Tensor | None] = [self.x]
        if self.z is not None:
            blocks.append(self.z)
        if self.extra:
            blocks.append(None)                     # kappa: a column of ones
        return blocks

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        piece = self.pieces(theta)
        blocks = self._blocks()
        w = self.w
        gradient = torch.cat([(w * g).sum().reshape(1) if block is None else block.T @ (w * g)
                              for block, g in zip(blocks, piece.grad, strict=True)])
        rows = []
        for i, left in enumerate(blocks):
            row = []
            for j, right in enumerate(blocks):
                weight = w * piece.hess[i][j]
                if left is None and right is None:
                    row.append(weight.sum().reshape(1, 1))
                elif left is None:
                    row.append((right.T @ weight).reshape(1, -1))
                elif right is None:
                    row.append((left.T @ weight).reshape(-1, 1))
                else:
                    row.append((left * weight[:, None]).T @ right)
            rows.append(torch.cat(row, dim=1))
        hessian = torch.cat(rows, dim=0)
        return (w * piece.value).sum(), gradient, (hessian + hessian.T) / 2

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, size] before weights."""
        piece = self.pieces(theta)
        parts = []
        for block, g in zip(self._blocks(), piece.grad, strict=True):
            parts.append(g[:, None] if block is None else block * g[:, None])
        return torch.cat(parts, dim=1)


def check_finite(value: Tensor, what: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise KernelError("numerical_failure", f"The {what} likelihood is not finite.")


def as_float(values: list[float]) -> Tensor:
    return torch.tensor(values, dtype=FLOAT)
