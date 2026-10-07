"""Float64 count likelihood primitives shared by the native estimators.

Count magnitudes and special-function shapes are distinct from row counts.
The conservative NB region prevents cancellation in unverified gamma algebra;
it does not impose an observation-count limit.
"""
from __future__ import annotations

import math

import torch

from openecon.engines.contracts import KernelError

NB_MAX_GAMMA_ARGUMENT = 1e7


def validate_nb_precision(y, shape=None):
    """Reject NB gamma arguments beyond the independently checked region."""
    limit = y + 1 if shape is None else y + shape + 1
    outside = limit >= NB_MAX_GAMMA_ARGUMENT if shape is None else limit > NB_MAX_GAMMA_ARGUMENT
    if bool(torch.as_tensor(outside).any()):
        raise KernelError("precision_unsupported", "Negative binomial count plus shape exceeds the validated gamma-function region (1e7); this is a count/shape precision limit, not an observation-count limit.")


def _poisson_half_deviance(y, mu, eta):
    if eta is None:
        eta = torch.log(mu)
    positive = y > 0
    safe_y = torch.where(positive, y, 1.)
    log_y = torch.log(safe_y)
    delta = (mu - safe_y) / safe_y
    tiny = positive & (delta.abs() <= .001)
    small = torch.where(tiny, delta, 0.)
    # Six terms give a next-term relative bound below3e-16 at1e-3.
    series = small.square() * (.5 + small * (-1 / 3 + small * (1 / 4
        + small * (-1 / 5 + small / 6))))
    moderate = (delta.abs() <= .5) & ~tiny
    safe_delta = torch.where(moderate, delta, 0.)
    direct = torch.where(moderate, safe_y * (safe_delta - torch.log1p(safe_delta)),
                         mu - safe_y + safe_y * (log_y - eta))
    deviance = torch.where(tiny, safe_y * series, direct)
    # Only genuinely large near-mean counts need the longer series. Ordinary
    # count batches avoid twenty-five full-vector Torch iterations per step.
    high = positive & (safe_y > 1e6) & (delta.abs() <= .125) & ~tiny
    if bool(high.any()):
        small = delta[high]
        power = small.square()
        series = power / 2
        for exponent in range(3, 26):
            power = -power * small
            series = series + power / exponent
        deviance = deviance.clone()
        deviance[high] = safe_y[high] * series
    return torch.where(positive, deviance, mu)


def poisson_deviance(y, mu, *, eta=None):
    """Stable Poisson unit deviance, including fractional quasi-ML outcomes."""
    return 2 * _poisson_half_deviance(y, mu, eta)


def poisson_logmass(y, mu, *, eta=None, log_factorial=None):
    """Poisson log mass from the deviance and a Stirling remainder.

    ``y*log(mu)-mu-lgamma(y+1)`` can lose every digit of a near-mean
    log mass at large counts. The deviance is evaluated as a local series
    instead of subtracting large log-factorial terms. Noninteger positive
    outcomes keep the same gamma extension used by the Poisson quasi-ML API.
    """
    positive = y > 0
    safe_y = torch.where(positive, y, 1.)
    log_y = torch.log(safe_y)
    deviance = _poisson_half_deviance(y, mu, eta)
    inverse = 1 / safe_y
    inverse2 = inverse.square()
    remainder = inverse * (1 / 12 + inverse2 * (-1 / 360 + inverse2 * (1 / 1260
        + inverse2 * (-1 / 1680 + inverse2 * (1 / 1188
        + inverse2 * (-691 / 360360 + inverse2 / 156))))))
    factorial = torch.lgamma(safe_y + 1) if log_factorial is None else log_factorial
    small_remainder = factorial - (safe_y + .5) * log_y + safe_y - .5 * math.log(2 * math.pi)
    remainder = torch.where(safe_y >= 15, remainder, small_remainder)
    value = -deviance - .5 * (math.log(2 * math.pi) + log_y) - remainder
    return torch.where(positive, value, -mu)
