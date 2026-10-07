"""Regularized incomplete gamma function on tensors: ``ln P(a, x)`` and ``ln Q(a, x)``.

For a scalar shape ``a > 0`` and a tensor ``x >= 0`` (the generalized gamma
likelihood evaluates one shape at many points), both tails are returned on the
log scale, each computed directly where it is the small one:

* ``a <= LARGE_SHAPE``: the power series of ``P`` for ``x < a + 1`` and the
  modified-Lentz continued fraction of ``Q`` for ``x >= a + 1`` (Numerical
  Recipes ``gser``/``gcf``), with prefactor ``x^a e^-x / Gamma(a)`` formed as
  ``exp(a ln x - x - lgamma(a))``; iterations run on all elements at once until
  every element has converged to machine precision.
* ``a > LARGE_SHAPE``: Temme's uniform asymptotic expansion (DLMF 8.12.8-8.12.10)

      Q(a, x) = erfc(eta sqrt(a/2)) / 2 + R_a(eta),
      R_a(eta) ~ exp(-a eta^2 / 2) / sqrt(2 pi a) (c_0(eta) + c_1(eta) / a),
      eta^2 / 2 = lambda - 1 - ln lambda,  lambda = x / a,  sign(eta) = sign(lambda - 1),
      c_0 = 1/(lambda-1) - 1/eta,
      c_1 = 1/eta^3 - 1/(lambda-1)^3 - 1/(lambda-1)^2 - 1/(12 (lambda-1)),

  with the Taylor values ``c_0 = -1/3 + eta/12 - 2 eta^2/135 + eta^3/864``,
  ``c_1 = -1/540 - eta/288`` near ``eta = 0`` (DLMF 8.12.12). The truncation
  error is of order ``a^-2`` relative to ``R``. The small tail is written as
  ``exp(-a eta^2 / 2) [erfcx(+-y) / 2 +- (c_0 + c_1/a) / sqrt(2 pi a)]``,
  ``y = eta sqrt(a/2)``, so it keeps full relative accuracy far into the tail.

The callers pass ``lambda`` through ``s = ln lambda`` so that the shape can grow
without bound (the lognormal limit of the generalized gamma).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError

LARGE_SHAPE = 1e4
_EPS = 2.220446049250313e-16
_TINY = 1e-300
_MAX_ITER = 5000


def expm1_minus(s: Tensor) -> Tensor:
    """``exp(s) - 1 - s`` without cancellation (series for |s| < 0.1)."""
    series = s * s * (0.5 + s * (1 / 6 + s * (1 / 24 + s * (1 / 120 + s * (1 / 720 + s * (
        1 / 5040 + s * (1 / 40320 + s / 362880)))))))
    return torch.where(s.abs() < 0.1, series, torch.expm1(s) - s)


def stirling_delta(a: float) -> float:
    """``lgamma(a) - (a - 1/2) ln a + a - ln(2 pi) / 2`` (Stirling remainder)."""
    if a >= 10:
        inv = 1 / a
        inv2 = inv * inv
        return inv * (1 / 12 - inv2 * (1 / 360 - inv2 * (1 / 1260 - inv2 / 1680)))
    return math.lgamma(a) - (a - 0.5) * math.log(a) + a - 0.5 * math.log(2 * math.pi)


def _series_cf(a: float, x: Tensor) -> tuple[Tensor, Tensor]:
    log_p = torch.full_like(x, -math.inf)
    log_q = torch.zeros_like(x)
    # x = +inf: P = 1, Q = 0; NaN propagates (the optimizer then backtracks). Only finite
    # positive arguments enter the iterations, which otherwise never meet their tolerance.
    log_p = torch.where(x == math.inf, torch.zeros_like(x), log_p)
    log_q = torch.where(x == math.inf, torch.full_like(x, -math.inf), log_q)
    nan = torch.isnan(x)
    log_p = torch.where(nan, x, log_p)
    log_q = torch.where(nan, x, log_q)
    positive = (x > 0) & torch.isfinite(x)
    if not bool(positive.any()):
        return log_p, log_q
    lgam = math.lgamma(a)
    series = positive & (x < a + 1)
    if bool(series.any()):
        xs = x[series]
        term = torch.full_like(xs, 1 / a)
        total = term.clone()
        n = 0
        while True:
            n += 1
            term = term * xs / (a + n)
            total = total + term
            if n % 8 == 0 and bool((term <= total * _EPS).all()):
                break
            if n > _MAX_ITER:
                raise KernelError("numerical_failure", "The incomplete gamma series did not "
                                  "converge.")
        lp = a * torch.log(xs) - xs - lgam + torch.log(total)
        log_p[series] = lp
        log_q[series] = torch.log1p(-torch.exp(lp).clamp(max=1.0))
    fraction = positive & ~series
    if bool(fraction.any()):
        xs = x[fraction]
        b = xs + 1 - a
        c = torch.full_like(xs, 1 / _TINY)
        d = 1 / b
        h = d.clone()
        n = 0
        while True:
            n += 1
            an = -n * (n - a)
            b = b + 2
            d = an * d + b
            d = torch.where(d.abs() < _TINY, torch.full_like(d, _TINY), d)
            c = b + an / c
            c = torch.where(c.abs() < _TINY, torch.full_like(c, _TINY), c)
            d = 1 / d
            delta = d * c
            h = h * delta
            if n % 4 == 0 and bool(((delta - 1).abs() <= _EPS).all()):
                break
            if n > _MAX_ITER:
                raise KernelError("numerical_failure", "The incomplete gamma continued fraction "
                                  "did not converge.")
        lq = a * torch.log(xs) - xs - lgam + torch.log(h)
        log_q[fraction] = lq
        log_p[fraction] = torch.log1p(-torch.exp(lq).clamp(max=1.0))
    return log_p, log_q


def _temme(a: float, s: Tensor) -> tuple[Tensor, Tensor]:
    """Both log tails from ``s = ln(x / a)`` for a large shape (see the module notes)."""
    lam_m1 = torch.expm1(s)                              # lambda - 1
    eta = torch.sign(s) * torch.sqrt(2 * expm1_minus(s))   # eta^2/2 = expm1(s) - s
    small = eta.abs() < 1e-2
    safe_eta = torch.where(small, torch.ones_like(eta), eta)
    safe_lam = torch.where(small, torch.ones_like(lam_m1), lam_m1)
    c0 = torch.where(small, -1 / 3 + eta * (1 / 12 + eta * (-2 / 135 + eta / 864)),
                     1 / safe_lam - 1 / safe_eta)
    c1 = torch.where(small, -1 / 540 - eta / 288,
                     1 / safe_eta ** 3 - 1 / safe_lam ** 3 - 1 / safe_lam ** 2
                     - 1 / (12 * safe_lam))
    correction = (c0 + c1 / a) / math.sqrt(2 * math.pi * a)
    y = eta * math.sqrt(a / 2)
    exponent = -0.5 * a * eta * eta
    upper = y >= 0
    small_tail = torch.where(upper, 0.5 * torch.special.erfcx(y.abs()) + correction,
                             0.5 * torch.special.erfcx(y.abs()) - correction)
    log_small = exponent + torch.log(small_tail.clamp_min(_TINY))
    log_large = torch.log1p(-torch.exp(log_small).clamp(max=1.0))
    return torch.where(upper, log_large, log_small), torch.where(upper, log_small, log_large)


def log_gamma_tails(a: float, s: Tensor) -> tuple[Tensor, Tensor]:
    """``(ln P(a, x), ln Q(a, x))`` at ``x = a exp(s)`` (``s = -inf`` means x = 0)."""
    if not math.isfinite(a) or a <= 0:
        raise KernelError("numerical_failure", "The incomplete gamma shape must be positive.")
    if a > LARGE_SHAPE:
        finite = torch.isfinite(s)
        log_p, log_q = _temme(a, torch.where(finite, s, torch.zeros_like(s)))
        low = s == -math.inf
        log_p = torch.where(low, torch.full_like(s, -math.inf), log_p)
        log_q = torch.where(low, torch.zeros_like(s), log_q)
        high = s == math.inf
        nan = torch.isnan(s)
        return (torch.where(nan, s, torch.where(high, torch.zeros_like(s), log_p)),
                torch.where(nan, s, torch.where(high, torch.full_like(s, -math.inf), log_q)))
    return _series_cf(a, a * torch.exp(s))
