"""Skewed innovation densities of the ARCH family, standardized to mean 0 and variance 1.

Two skewed distributions are offered. They use DIFFERENT skewing mechanisms, each
the one of the software it is validated against; the parameterizations are not
interchangeable.

``skewt``: Hansen's (1994) skewed Student t (as in Python ``arch``'s ``SkewStudent``)
    With eta > 2 degrees of freedom and asymmetry -1 < lambda < 1,

        c = G((eta+1)/2) / (sqrt(pi (eta-2)) G(eta/2)),   a = 4 lambda c (eta-2)/(eta-1),
        b^2 = 1 + 3 lambda^2 - a^2,
        g(z) = b c (1 + ((b z + a) / (1 + s lambda))^2 / (eta - 2))^(-(eta+1)/2),

    s = sign(b z + a). Hansen, B. E. (1994), "Autoregressive conditional density
    estimation", International Economic Review 35(3), 705-730, eq. (10)-(11). Reported
    parameters: ``/lndfm2 = ln(eta - 2)`` (as for ``t``) and ``/atlambda = atanh(lambda)``.
    lambda > 0 skews to the right. lambda = 0 is the unit-variance Student t.

``sged``: the Fernandez-Steel (1998) skewed generalized error distribution, standardized
    (as fGarch's ``dsged`` and rugarch's ``"sged"``). With f the unit-variance GED of
    shape nu and skewness xi > 0,

        m1 = 2^(1/nu) lam G(2/nu) / G(1/nu),         lam = (2^(-2/nu) G(1/nu) / G(3/nu))^(1/2),
        mu = m1 (xi - 1/xi),    sigma^2 = (1 - m1^2)(xi^2 + 1/xi^2) + 2 m1^2 - 1,
        y = z sigma + mu,       g(z) = sigma 2 / (xi + 1/xi) f(y / xi^sign(y)).

    Fernandez, C. and Steel, M. F. J. (1998), "On Bayesian modeling of fat tails and
    skewness", JASA 93(441), 359-371; standardization as in Wuertz et al., fGarch
    (``.dsged``). Reported parameters: ``/lnshape = ln(nu)`` (as for ``ged``) and
    ``/lnxi = ln(xi)``. xi > 1 skews to the right; xi = 1 is the GED.

Everything returns analytic derivatives with respect to e_t, ln h_t and both reported
parameters. Moments that have no closed form here (E|z|^p, E[z^2 1(z < 0)]) are computed
by double-exponential quadrature of the density split at its kinks (z = 0 and the switch
point of the two pieces); quantiles are closed forms in the Student t quantile
(``engines.distributions.t_ppf``) and the chi-square quantile of the GED kernel.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor

from openecon.engines.distributions import chi2_ppf, t_ppf

SKEWED = ("skewt", "sged")
_LOG2 = math.log(2.0)
# Admissible ranges of the reported parameters (first: ln(eta - 2) or ln(nu); second:
# atanh(lambda) or ln(xi)). |lambda| < tanh(8) and 1/e^4 < xi < e^4 are far beyond any
# skewness a sample can identify; the bounds keep the constants finite.
BOUNDS = {"skewt": ((-12.0, math.log(1000.0)), (-8.0, 8.0)),
          "sged": ((-2.5, 4.0), (-4.0, 4.0))}


def _digamma(value: float) -> float:
    return float(torch.digamma(torch.tensor(value, dtype=torch.float64)))


def feasible(dist: str, tau: tuple[float, float]) -> bool:
    return all(low < value < high for value, (low, high) in zip(tau, BOUNDS[dist], strict=True))


def natural(dist: str, tau: tuple[float, float]) -> tuple[float, float]:
    """(degrees of freedom, lambda) for ``skewt``; (shape, xi) for ``sged``."""
    if dist == "skewt":
        return 2.0 + math.exp(tau[0]), math.tanh(tau[1])
    return math.exp(tau[0]), math.exp(tau[1])


# ---- Hansen skewed t ------------------------------------------------------------------------


def _hansen_constants(eta: float, lam: float) -> tuple[float, float, float, float]:
    """(ln c, a, b, c) of Hansen's skewed t."""
    log_c = math.lgamma(0.5 * (eta + 1.0)) - math.lgamma(0.5 * eta) \
        - 0.5 * math.log(math.pi * (eta - 2.0))
    c = math.exp(log_c)
    a = 4.0 * lam * c * (eta - 2.0) / (eta - 1.0)
    b = math.sqrt(1.0 + 3.0 * lam * lam - a * a)
    return log_c, a, b, c


def _skewt(e: Tensor, h: Tensor, tau: tuple[float, float], derivatives: bool):
    scale = math.exp(tau[0])                    # eta - 2
    eta, lam = 2.0 + scale, math.tanh(tau[1])
    log_c, a, b, c = _hansen_constants(eta, lam)
    root = h.sqrt()
    z = e / root
    m = b * z + a
    side = torch.where(m < 0.0, -torch.ones_like(m), torch.ones_like(m))
    denominator = 1.0 + side * lam
    u = m / denominator
    w = 1.0 + u.square() / scale
    log_w = torch.log1p(u.square() / scale)
    value = math.log(b) + log_c - 0.5 * h.log() - 0.5 * (eta + 1.0) * log_w
    if not derivatives:
        return value, None, None, None
    d_z = -(eta + 1.0) * u * b / (scale * w * denominator)
    d_e = d_z / root
    d_log_h = -0.5 - 0.5 * z * d_z
    # lambda: a, b and the piece denominators move.
    a_lam = 4.0 * c * (eta - 2.0) / (eta - 1.0)
    b_lam = (3.0 * lam - a * a_lam) / b
    u_lam = (b_lam * z + a_lam) / denominator - u * side / denominator
    d_lam = b_lam / b - (eta + 1.0) * u * u_lam / (scale * w)
    # eta: c, a, b and the kernel.
    dlog_c = 0.5 * (_digamma(0.5 * (eta + 1.0)) - _digamma(0.5 * eta)) - 0.5 / scale
    c_eta = c * dlog_c
    a_eta = 4.0 * lam * (c_eta * (eta - 2.0) / (eta - 1.0) + c / (eta - 1.0) ** 2)
    b_eta = -a * a_eta / b
    u_eta = (b_eta * z + a_eta) / denominator
    dlog_w = (2.0 * u * u_eta / scale - u.square() / scale ** 2) / w
    d_eta = b_eta / b + dlog_c - 0.5 * log_w - 0.5 * (eta + 1.0) * dlog_w
    d_tau = torch.stack([scale * d_eta, (1.0 - lam * lam) * d_lam])
    return value, d_e, d_log_h, d_tau


def _skewt_quantile(probability: float, tau: tuple[float, float]) -> float:
    eta, lam = natural("skewt", tau)
    _, a, b, _ = _hansen_constants(eta, lam)
    split = 0.5 * (1.0 - lam)
    if probability < split:
        base = t_ppf(probability / (1.0 - lam), eta) * (1.0 - lam)
    else:
        base = t_ppf(0.5 + (probability - split) / (1.0 + lam), eta) * (1.0 + lam)
    return (base * math.sqrt(1.0 - 2.0 / eta) - a) / b


def _skewt_switch(tau: tuple[float, float]) -> float:
    eta, lam = natural("skewt", tau)
    _, a, b, _ = _hansen_constants(eta, lam)
    return -a / b


# ---- Fernandez-Steel skewed GED -------------------------------------------------------------


def _ged_log_scale(shape: float) -> float:
    return 0.5 * (math.lgamma(1.0 / shape) - math.lgamma(3.0 / shape)) - _LOG2 / shape


def _sged_constants(shape: float, xi: float) -> tuple[float, float, float, float]:
    """(ln lam, m1, mu, sigma) of the standardized Fernandez-Steel skewed GED."""
    log_scale = _ged_log_scale(shape)
    m1 = math.exp(_LOG2 / shape + log_scale + math.lgamma(2.0 / shape) - math.lgamma(1.0 / shape))
    mu = m1 * (xi - 1.0 / xi)
    sigma = math.sqrt((1.0 - m1 * m1) * (xi * xi + 1.0 / (xi * xi)) + 2.0 * m1 * m1 - 1.0)
    return log_scale, m1, mu, sigma


def _sged(e: Tensor, h: Tensor, tau: tuple[float, float], derivatives: bool):
    shape, xi = math.exp(tau[0]), math.exp(tau[1])
    log_scale, m1, mu, sigma = _sged_constants(shape, xi)
    root = h.sqrt()
    z = e / root
    y = z * sigma + mu
    negative = y < 0.0
    side = torch.where(negative, -torch.ones_like(y), torch.ones_like(y))
    stretch = torch.where(negative, torch.full_like(y, xi), torch.full_like(y, 1.0 / xi))
    w = y * stretch
    zero = w == 0.0
    log_abs = torch.where(zero, torch.zeros_like(w), w.abs().log()) - log_scale
    q = torch.where(zero, torch.zeros_like(w), (shape * log_abs).exp())    # |w / lam|^nu
    kernel = math.log(shape) - log_scale - (1.0 + 1.0 / shape) * _LOG2 - math.lgamma(1.0 / shape)
    constant = math.log(sigma) + _LOG2 - math.log(xi + 1.0 / xi) + kernel
    value = constant - 0.5 * h.log() - 0.5 * q
    if not derivatives:
        return value, None, None, None
    d_w = torch.where(zero, torch.zeros_like(w), -0.5 * shape * q / w)
    d_z = d_w * sigma * stretch
    d_e = d_z / root
    d_log_h = -0.5 - 0.5 * z * d_z
    # xi
    mu_xi = m1 * (1.0 + 1.0 / (xi * xi))
    sigma_xi = (1.0 - m1 * m1) * (2.0 * xi - 2.0 / xi ** 3) / (2.0 * sigma)
    y_xi = z * sigma_xi + mu_xi
    w_xi = stretch * (y_xi - side * y / xi)
    d_xi = sigma_xi / sigma - (1.0 - 1.0 / (xi * xi)) / (xi + 1.0 / xi) + d_w * w_xi
    # nu
    psi1, psi2, psi3 = _digamma(1.0 / shape), _digamma(2.0 / shape), _digamma(3.0 / shape)
    log_scale_nu = (_LOG2 - 0.5 * psi1 + 1.5 * psi3) / shape ** 2
    log_m1_nu = -_LOG2 / shape ** 2 + log_scale_nu - 2.0 * psi2 / shape ** 2 + psi1 / shape ** 2
    m1_nu = m1 * log_m1_nu
    mu_nu = m1_nu * (xi - 1.0 / xi)
    sigma_nu = 2.0 * m1 * m1_nu * (2.0 - xi * xi - 1.0 / (xi * xi)) / (2.0 * sigma)
    w_nu = stretch * (z * sigma_nu + mu_nu)
    kernel_nu = 1.0 / shape - log_scale_nu + _LOG2 / shape ** 2 + psi1 / shape ** 2
    d_nu = sigma_nu / sigma + kernel_nu \
        - 0.5 * q * (log_abs - shape * log_scale_nu) + d_w * w_nu
    d_tau = torch.stack([shape * d_nu, xi * d_xi])
    return value, d_e, d_log_h, d_tau


def _ged_quantile(probability: float, shape: float) -> float:
    """Quantile of the unit-variance GED: |x / lam|^nu is chi-squared with 2/nu df."""
    if probability == 0.5:
        return 0.0
    upper = probability > 0.5
    tail = 2.0 * probability - 1.0 if upper else 1.0 - 2.0 * probability
    size = math.exp(_ged_log_scale(shape)) * chi2_ppf(tail, 2.0 / shape) ** (1.0 / shape)
    return size if upper else -size


def _sged_quantile(probability: float, tau: tuple[float, float]) -> float:
    shape, xi = natural("sged", tau)
    _, _, mu, sigma = _sged_constants(shape, xi)
    split = 1.0 / (1.0 + xi * xi)                 # P(y < 0)
    if probability < split:
        y = _ged_quantile(probability * (1.0 + xi * xi) / 2.0, shape) / xi
    else:
        y = xi * _ged_quantile(0.5 + (probability - split) * (1.0 + 1.0 / (xi * xi)) / 2.0, shape)
    return (y - mu) / sigma


def _sged_switch(tau: tuple[float, float]) -> float:
    shape, xi = natural("sged", tau)
    _, _, mu, sigma = _sged_constants(shape, xi)
    return -mu / sigma


# ---- public interface -----------------------------------------------------------------------


def log_density(dist: str, e: Tensor, h: Tensor, tau: tuple[float, float],
                derivatives: bool = True):
    """``(l, dl/de, dl/dln h, dl/dtau)``; ``dl/dtau`` is ``[2, T]`` (both parameters)."""
    if dist == "skewt":
        return _skewt(e, h, tau, derivatives)
    if dist == "sged":
        return _sged(e, h, tau, derivatives)
    raise ValueError(f"Unknown skewed distribution {dist!r}.")


def quantile(dist: str, probability: float, tau: tuple[float, float]) -> float:
    """Quantile of the standardized skewed innovation."""
    if dist == "skewt":
        return _skewt_quantile(probability, tau)
    return _sged_quantile(probability, tau)


@lru_cache(maxsize=4)
def _double_exponential(step: float, limit: float) -> tuple[Tensor, Tensor]:
    t = torch.arange(-limit, limit + 0.5 * step, step, dtype=torch.float64)
    return t, torch.full_like(t, step)


def _integrate(function, low: float, high: float) -> float:
    """``int_low^high function(z) dz`` by tanh-sinh / exp-sinh quadrature (infinite ends ok).

    The nodes never touch the end points, so integrable algebraic singularities and
    kinks at the ends (the segments are split at the kinks of the integrand) cost
    nothing; on a half line the substitution z = end + exp(pi/2 sinh t) turns algebraic
    tails into double-exponential decay. Accuracy is close to machine precision for the
    smooth, unimodal integrands used here.
    """
    if low == high:
        return 0.0
    t, w = _double_exponential(1.0 / 48.0, 4.5)
    half_pi = 0.5 * math.pi
    if math.isinf(low) and math.isinf(high):
        raise ValueError("split the real line at a finite point")
    if math.isinf(high) or math.isinf(low):
        start, sign = (low, 1.0) if math.isinf(high) else (high, -1.0)
        grow = torch.exp(half_pi * torch.sinh(t))
        z = start + sign * grow
        jacobian = half_pi * torch.cosh(t) * grow
    else:
        middle, radius = 0.5 * (low + high), 0.5 * (high - low)
        inner = half_pi * torch.sinh(t)
        z = middle + radius * torch.tanh(inner)
        jacobian = radius * half_pi * torch.cosh(t) / torch.cosh(inner).square()
    values = function(z) * jacobian
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return float((values * w).sum())


def expectation(dist: str, tau: tuple[float, float], function) -> float:
    """E[function(z)] of the standardized skewed innovation (a function smooth off 0)."""
    switch = _skewt_switch(tau) if dist == "skewt" else _sged_switch(tau)
    points = sorted({0.0, switch})
    one = torch.ones(1, dtype=torch.float64)

    def integrand(z: Tensor) -> Tensor:
        return function(z) * log_density(dist, z, one.expand_as(z), tau, False)[0].exp()

    edges = [-math.inf, *points, math.inf]
    return sum(_integrate(integrand, low, high) for low, high in zip(edges[:-1], edges[1:]))


def abs_moment(dist: str, power: float, tau: tuple[float, float]) -> float:
    """E|z|^power (``inf`` when it does not exist: Student t with power >= eta)."""
    if dist == "skewt" and power >= natural(dist, tau)[0]:
        return math.inf
    return expectation(dist, tau, lambda z: z.abs().pow(power))


def negative_share(dist: str, tau: tuple[float, float]) -> float:
    """E[z^2 1(z < 0)]: the expected GJR threshold news per unit of variance."""
    return expectation(dist, tau, lambda z: torch.where(z < 0, z.square(), torch.zeros_like(z)))
