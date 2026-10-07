"""Innovation densities of the ARCH family, standardized to unit variance.

The disturbance is e_t = sqrt(h_t) z_t with E z = 0 and Var z = 1, so the log density of
e_t given the past is ``ln f(e_t / sqrt(h_t)) - ln(h_t) / 2``:

normal
    ln f(z) = -ln(2 pi)/2 - z^2/2.

Student t with nu > 2 degrees of freedom, parameter tau = ln(nu - 2) (Stata's /lndfm2)
    ln f(z) = lnG((nu+1)/2) - lnG(nu/2) - ln(pi (nu - 2))/2 - (nu+1)/2 ln(1 + z^2/(nu - 2)).

generalized error distribution with shape s > 0, parameter tau = ln(s) (Stata's /lnshape)
    ln f(z) = ln s - ln lam - (1 + 1/s) ln 2 - lnG(1/s) - |z / lam|^s / 2,
    lam = [2^(-2/s) G(1/s) / G(3/s)]^(1/2);    s = 2 is the normal, s = 1 the Laplace.

``log_density`` returns the per-observation log density and its analytic derivatives with
respect to e_t, ln h_t and tau. ``abs_moment`` is E|z|^p (persistence of power ARCH,
EGARCH forecasts) and ``critical_value`` the two-sided quantile used for forecast
intervals.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.distributions import chi2_isf, normal_isf, t_isf

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG2 = math.log(2.0)
ABS_NORMAL = math.sqrt(2.0 / math.pi)        # E|z| of the standard normal
# Admissible range of the distribution parameter tau (ln(df - 2) or ln(shape)). A Student
# t with more than 1000 degrees of freedom cannot be told from the normal by any sample
# (the likelihood is flat in tau there and the difference of log-gamma functions starts
# to lose digits), so the optimizer is stopped at that bound and the fit reports that the
# degrees of freedom diverge instead of a spurious maximum on the flat ridge.
TAU_BOUNDS = {"t": (-12.0, math.log(1000.0)), "ged": (-2.5, 4.0)}


def _digamma(value: float) -> float:
    return float(torch.digamma(torch.tensor(value, dtype=torch.float64)))


def feasible(dist: str, tau: float) -> bool:
    if dist == "normal":
        return True
    low, high = TAU_BOUNDS[dist]
    return low < tau < high


def shape_value(dist: str, tau: float) -> float | None:
    """Degrees of freedom (t) or shape (GED) implied by the reported parameter."""
    if dist == "t":
        return 2.0 + math.exp(tau)
    if dist == "ged":
        return math.exp(tau)
    return None


def _ged_log_scale(shape: float) -> float:
    """ln lam of the unit-variance GED."""
    return 0.5 * (math.lgamma(1.0 / shape) - math.lgamma(3.0 / shape)) - _LOG2 / shape


@torch.no_grad()
def log_density(dist: str, e: Tensor, h: Tensor, tau: float = 0.0, derivatives: bool = True,
                flat: Tensor | None = None) -> tuple[Tensor, Tensor | None, Tensor | None,
                                                     Tensor | None]:
    """``(l, dl/de, dl/dln h, dl/dtau)`` per observation; derivatives ``None`` on request.

    ``dl/dtau`` is ``None`` for the normal density. ``flat`` (GED only) is a boolean mask
    of observations whose kernel ``|z / lam|^s`` is taken as zero whatever e_t: the
    observations held at z_t = 0 by a kink of the EGARCH likelihood, where the kernel
    vanishes but its second derivative is unbounded for s < 2 (see ``kinks``).
    """
    log_h = h.log()
    if dist == "normal":
        z2 = e.square() / h
        value = -_HALF_LOG_2PI - 0.5 * log_h - 0.5 * z2
        if not derivatives:
            return value, None, None, None
        return value, -e / h, 0.5 * (z2 - 1.0), None
    if dist == "t":
        scale = math.exp(tau)                     # nu - 2
        nu = 2.0 + scale
        w = e.square() / (h * scale)
        log1p = torch.log1p(w)
        constant = math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) \
            - 0.5 * math.log(math.pi * scale)
        value = constant - 0.5 * log_h - 0.5 * (nu + 1.0) * log1p
        if not derivatives:
            return value, None, None, None
        ratio = w / (1.0 + w)
        d_e = -(nu + 1.0) * e / (h * scale * (1.0 + w))
        d_log_h = -0.5 + 0.5 * (nu + 1.0) * ratio
        d_nu = 0.5 * (_digamma(0.5 * (nu + 1.0)) - _digamma(0.5 * nu)) - 0.5 / scale \
            - 0.5 * log1p + 0.5 * (nu + 1.0) * ratio / scale
        return value, d_e, d_log_h, scale * d_nu
    if dist == "ged":
        shape = math.exp(tau)
        log_scale = _ged_log_scale(shape)
        zero = e == 0.0 if flat is None else (e == 0.0) | flat
        log_abs = torch.where(zero, torch.zeros_like(e), e.abs().log()) - 0.5 * log_h - log_scale
        q = torch.where(zero, torch.zeros_like(e), (shape * log_abs).exp())
        constant = tau - log_scale - (1.0 + 1.0 / shape) * _LOG2 - math.lgamma(1.0 / shape)
        value = constant - 0.5 * log_h - 0.5 * q
        if not derivatives:
            return value, None, None, None
        d_e = torch.where(zero, torch.zeros_like(e), -0.5 * shape * q / e)
        d_log_h = -0.5 + 0.25 * shape * q
        psi1, psi3 = _digamma(1.0 / shape), _digamma(3.0 / shape)
        d_log_scale = (_LOG2 - 0.5 * psi1 + 1.5 * psi3) / shape ** 2
        d_shape = 1.0 / shape - d_log_scale + (_LOG2 + psi1) / shape ** 2 \
            - 0.5 * q * (log_abs - shape * d_log_scale)
        return value, d_e, d_log_h, shape * d_shape
    raise ValueError(f"Unknown distribution {dist!r}.")


def abs_moment(dist: str, power: float, tau: float = 0.0) -> float:
    """E|z|^power of the unit-variance innovation (``inf`` when it does not exist)."""
    if dist == "normal":
        return 2.0 ** (0.5 * power) * math.gamma(0.5 * (power + 1.0)) / math.sqrt(math.pi)
    if dist == "t":
        scale = math.exp(tau)
        nu = 2.0 + scale
        if power >= nu:
            return math.inf
        log_value = 0.5 * power * math.log(scale) + math.lgamma(0.5 * (power + 1.0)) \
            + math.lgamma(0.5 * (nu - power)) - math.lgamma(0.5 * nu) - 0.5 * math.log(math.pi)
        return math.exp(log_value)
    shape = math.exp(tau)
    log_value = power * _ged_log_scale(shape) + power / shape * _LOG2 \
        + math.lgamma((power + 1.0) / shape) - math.lgamma(1.0 / shape)
    return math.exp(log_value)


def critical_value(dist: str, alpha: float, tau: float = 0.0) -> float:
    """c with P(|z| > c) = alpha for the unit-variance innovation."""
    if dist == "normal":
        return normal_isf(0.5 * alpha)
    if dist == "t":
        scale = math.exp(tau)
        nu = 2.0 + scale
        return t_isf(0.5 * alpha, nu) * math.sqrt(scale / nu)
    shape = math.exp(tau)
    # |z / lam|^s is chi-squared with 2/s degrees of freedom.
    return math.exp(_ged_log_scale(shape)) * chi2_isf(alpha, 2.0 / shape) ** (1.0 / shape)
