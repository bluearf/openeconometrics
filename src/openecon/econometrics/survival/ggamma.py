"""Generalized gamma survival likelihood (Stata ``streg, distribution(ggamma)``).

With ``sigma = exp(a)``, ``gamma = |kappa|^-2``, ``z = sign(kappa) (ln t - mu) / sigma``
and ``s = |kappa| z`` (so that ``u = gamma exp(s)``), Stata's density and survivor
function are

    f(t) = gamma^gamma / (sigma t sqrt(gamma) Gamma(gamma)) exp(z sqrt(gamma) - u)
    S(t) = 1 - I(gamma, u)  (kappa > 0),   I(gamma, u)  (kappa < 0),

and the lognormal at ``kappa = 0``. The density is evaluated in the
cancellation-free form

    ln f = -a - ln t - ln(2 pi)/2 - delta(gamma) - gamma (exp(s) - 1 - s)

(``delta`` the Stirling remainder of ``ln Gamma``), which tends to the lognormal
density as kappa -> 0, and the survivor function by ``special.log_gamma_tails``
(series / continued fraction, Temme's uniform expansion for gamma > 1e4).

Derivatives (hybrid analytic / numerical path, recorded in the result's provenance):

* in ``(mu, a)`` they are analytic. With ``w = (ln t - mu) / sigma`` the index is
  ``s = kappa w``, so ``ds/dmu = -kappa / sigma`` and ``ds/da = -s``. For ``ln f``,
  ``d ln f / ds = -gamma (e^s - 1)`` and ``d2 ln f / ds2 = -gamma e^s`` (plus the
  explicit ``-a``). For ``ln S`` (``u = gamma e^s``, ``du/ds = u``) with
  ``m = u^gamma e^-u / (Gamma(gamma) T)``, ``T`` the tail that forms S (Q for
  kappa > 0, P for kappa < 0) and ``c = -1`` (kappa > 0) or ``+1`` (kappa < 0):
  ``d ln S / ds = c m`` and ``d2 ln S / ds2 = c m (gamma - u - c m)``; ``ln m`` is
  evaluated as ``ln(gamma)/2 - ln(2 pi)/2 - delta(gamma) - gamma (e^s - 1 - s) - ln T``,
  which stays finite in the lognormal limit;
* in ``kappa`` (the shape of the incomplete gamma function, whose derivative has no
  closed form) they are five-point central differences, step 1e-3, of the value and
  of the analytic ``(mu, a)`` derivatives: four extra evaluations per iteration
  (fourth-order accurate).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.survival.parametric import LOG_SQRT_2PI, Pieces, SurvivalData
from openecon.econometrics.survival.special import expm1_minus, log_gamma_tails, stirling_delta

# |kappa| below which the lognormal limit is used exactly.
KAPPA_ZERO = 1e-10
STEP = 1e-3


def _log_terms(log_t: Tensor, mu: Tensor, a: Tensor, kappa: float) -> tuple[Tensor, Tensor]:
    """(ln f, ln S) at the given times."""
    sigma = torch.exp(a)
    if abs(kappa) < KAPPA_ZERO:
        z = (log_t - mu) / sigma
        return -a - log_t - LOG_SQRT_2PI - 0.5 * z * z, torch.special.log_ndtr(-z)
    gamma = kappa ** -2
    z = math.copysign(1.0, kappa) * (log_t - mu) / sigma
    s = abs(kappa) * z
    log_f = -a - log_t - LOG_SQRT_2PI - stirling_delta(gamma) - gamma * expm1_minus(s)
    log_p, log_q = log_gamma_tails(gamma, s)
    return log_f, (log_q if kappa > 0 else log_p)


def ggamma_loglik(data: SurvivalData, mu: Tensor, a: Tensor, kappa: Tensor | float) -> Tensor:
    """Per-observation ``delta ln f + (1 - delta) ln S(t) - ln S(t0)``."""
    kappa = float(kappa)
    log_f, log_s = _log_terms(data.log_t, mu, a, kappa)
    value = data.delta * log_f + (1 - data.delta) * log_s
    if data.truncated is not None:
        _, log_s0 = _log_terms(data.log_t0, mu, a, kappa)
        value = value - torch.where(data.truncated, log_s0, torch.zeros_like(log_s0))
    return value


def _analytic(log_t: Tensor, delta: Tensor, mu: Tensor, a: Tensor, kappa: float,
              truncated: Tensor | None, log_t0: Tensor | None) -> Pieces:
    """Value and analytic derivatives in (mu, a) of ``delta ln f + (1-delta) ln S - ln S0``."""
    sigma = torch.exp(a)

    def tail_terms(lt: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """ln S, d ln S / dmu, d ln S / da and second derivatives packed as (g1, g2, s, w)."""
        w = (lt - mu) / sigma
        if abs(kappa) < KAPPA_ZERO:                       # lognormal limit
            log_s = torch.special.log_ndtr(-w)
            mills = torch.exp(-0.5 * w * w - LOG_SQRT_2PI - log_s)
            # d ln S / dw = -mills, d2 = -mills (mills - w); in s-units with s = w
            return log_s, -mills, -mills * (mills - w), w
        gamma = kappa ** -2
        s = kappa * w
        log_p, log_q = log_gamma_tails(gamma, s)
        log_tail, sign = (log_q, -1.0) if kappa > 0 else (log_p, 1.0)
        log_m = (0.5 * math.log(gamma) - LOG_SQRT_2PI - stirling_delta(gamma)
                 - gamma * expm1_minus(s) - log_tail)
        m = torch.exp(log_m)
        u = gamma * torch.exp(s)
        return log_tail, sign * m, sign * m * (gamma - u - sign * m), s

    def chain(g1: Tensor, g2: Tensor, s: Tensor) -> tuple[Tensor, ...]:
        """(d/dmu, d/da, d2/dmu2, d2/dmu da, d2/da2) of a function of s alone."""
        k = kappa if abs(kappa) >= KAPPA_ZERO else 1.0   # lognormal: s = w, ds/dmu = -1/sigma
        return (-g1 * k / sigma, -g1 * s, g2 * (k / sigma) ** 2, k / sigma * (g2 * s + g1),
                s * g1 + s * s * g2)

    w = (log_t - mu) / sigma
    if abs(kappa) < KAPPA_ZERO:
        s = w
        f1, f2 = -w, -torch.ones_like(w)                   # d ln phi / dw, d2
        log_f = -a - log_t - LOG_SQRT_2PI - 0.5 * w * w
    else:
        gamma = kappa ** -2
        s = kappa * w
        log_f = -a - log_t - LOG_SQRT_2PI - stirling_delta(gamma) - gamma * expm1_minus(s)
        f1 = -torch.expm1(s) / kappa ** 2                  # -gamma (e^s - 1)
        f2 = -torch.exp(s) / kappa ** 2                    # -gamma e^s
    fm, fa, fmm, fma, faa = chain(f1, f2, s)
    fa = fa - 1.0                                          # the explicit -a of ln f
    log_s, g1, g2, s_tail = tail_terms(log_t)
    sm, sa, smm, sma, saa = chain(g1, g2, s_tail)
    d, c = delta, 1 - delta
    value = d * log_f + c * log_s
    grad = [d * fm + c * sm, d * fa + c * sa]
    hess = [[d * fmm + c * smm, d * fma + c * sma], [None, d * faa + c * saa]]
    if truncated is not None:
        log_s0, h1, h2, s0 = tail_terms(log_t0)
        zero = torch.zeros_like(value)
        keep = truncated

        def cut(values: Tensor) -> Tensor:
            return torch.where(keep, values, zero)

        tm, ta, tmm, tma, taa = chain(h1, h2, s0)
        value = value - cut(log_s0)
        grad = [grad[0] - cut(tm), grad[1] - cut(ta)]
        hess = [[hess[0][0] - cut(tmm), hess[0][1] - cut(tma)], [None, hess[1][1] - cut(taa)]]
    hess[1][0] = hess[0][1]
    return Pieces(value, grad, hess)


def ggamma_pieces(data: SurvivalData, mu: Tensor, a: Tensor, kappa: Tensor) -> Pieces:
    """Value and per-observation derivatives in (mu, a, kappa): analytic in (mu, a),
    five-point central differences in kappa (see the module notes)."""
    kappa = float(kappa)
    h = STEP

    def at(k: float) -> Pieces:
        return _analytic(data.log_t, data.delta, mu, a, k, data.truncated, data.log_t0)

    centre, plus, minus, plus2, minus2 = at(kappa), at(kappa + h), at(kappa - h), \
        at(kappa + 2 * h), at(kappa - 2 * h)

    def first(values: list[Tensor]) -> Tensor:
        f2, f1, f_1, f_2 = values
        return (-f2 + 8 * f1 - 8 * f_1 + f_2) / (12 * h)

    values = [plus2.value, plus.value, minus.value, minus2.value]
    d_kappa = first(values)
    d_kk = (-plus2.value + 16 * plus.value - 30 * centre.value + 16 * minus.value
            - minus2.value) / (12 * h * h)
    d_mk = first([plus2.grad[0], plus.grad[0], minus.grad[0], minus2.grad[0]])
    d_ak = first([plus2.grad[1], plus.grad[1], minus.grad[1], minus2.grad[1]])
    hess = [[centre.hess[0][0], centre.hess[0][1], d_mk],
            [centre.hess[1][0], centre.hess[1][1], d_ak],
            [d_mk, d_ak, d_kk]]
    return Pieces(centre.value, [centre.grad[0], centre.grad[1], d_kappa], hess)
