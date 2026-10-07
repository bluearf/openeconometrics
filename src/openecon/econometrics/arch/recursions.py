"""Recursions of the ARCH family: lag operators, the state path and the derivative scan.

Tensors are float64 with time as the LAST dimension unless stated otherwise.

Three tools cover every model of the family:

``lagged``, ``lead`` and ``dense`` shift series and build lag polynomials for
``arima.filters.inverse_filter``, which evaluates a recursion with CONSTANT
coefficients (``v_t = f_t + sum_j b_j v_(t-j)``: the GARCH, GJR, IGARCH and power-ARCH
variance equations, their derivatives, and the moving-average filter of the mean
equation) for all periods at once by cyclic reduction, without a time loop.
``adjoint_filter`` is its transpose, used for gradients in reverse mode.

``run_path`` is the one loop over time in the family. EGARCH is nonlinear in its own
lagged state (the standardized shock z = e / sqrt(h) divides by the lagged variance) and
ARCH-in-mean makes the residual depend on the current variance, so their state path
(e_t, h_t) cannot be written as a linear filter. The loop runs on Python floats and does
nothing but the model's own recursion: no derivative is carried along it. Models with
one ARCH lag, at most one GARCH lag and no ARMA terms (EGARCH(1,1), GARCH(1,1)-in-mean,
...) take a loop on scalar state, about three times cheaper per period than the
general one.

``solve_recurrence`` then obtains ALL derivative paths of such a model without a time
loop. Given the state path, the derivatives obey a linear recursion with time-varying
coefficients,

    s_t = M_t s_(t-1) + f_t,        s_(-1) = 0,

where s_t stacks d v_t, d v_(t-1), ..., d e_t, d e_(t-1), ... (one column per parameter).
The affine maps (M_t, f_t) are composed by a parallel prefix scan: after the round with
stride 2^r every (M_t, f_t) spans 2^(r+1) periods,

    f_t <- M_t f_(t-s) + f_t,       M_t <- M_t M_(t-s),            s = 2^r,

so ceil(log2 T) rounds of batched small matrix products give every s_t exactly (products
of contractions underflow to zero harmlessly; nothing is divided). The rounds stop early
once every remaining product is below 1e-20: a stable recursion forgets its past at a
geometric rate, so about log2(50 / (1 - persistence)) rounds suffice whatever T.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from openecon.econometrics.arima.filters import inverse_filter

_NEGLIGIBLE = 1e-20        # products of transition maps below this no longer matter


def lagged(x: Tensor, lag: int, fill: float | Tensor = 0.0) -> Tensor:
    """``L^lag x`` along the last dimension; the first ``lag`` entries are ``fill``.

    ``fill`` is the presample value: a number, or a tensor that broadcasts against the
    leading ``lag`` entries (for example a ``[k, 1]`` column for a ``[k, T]`` block).
    """
    n = x.shape[-1]
    out = torch.empty_like(x)
    head = min(lag, n)
    out[..., :head] = fill
    if lag < n:
        out[..., lag:] = x[..., :n - lag]
    return out


def lead(x: Tensor, lag: int) -> Tensor:
    """``F^lag x`` along the last dimension: entry t is x_(t+lag); the last ``lag`` are zero."""
    n = x.shape[-1]
    out = torch.zeros_like(x)
    if lag < n:
        out[..., :n - lag] = x[..., lag:]
    return out


def adjoint_filter(x: Tensor, coefficients: Sequence[float]) -> Tensor:
    """The transpose of ``inverse_filter``: ``y_t = x_t - sum_i c_i y_(t+i)``, run backwards.

    For any series, ``sum_t x_t (f / c(L))_t = sum_t (adjoint_filter(x))_t f_t``: one
    anti-causal filter of the weights replaces a causal filter of every derivative path.
    """
    if not coefficients:
        return x
    return inverse_filter(x.flip(-1), coefficients).flip(-1)


def dense(values: Sequence[float], lags: Sequence[int], sign: float = 1.0) -> list[float]:
    """Dense coefficients ``c_1..c_m`` with ``c_lag = sign * value`` (zeros elsewhere)."""
    coefficients = [0.0] * (max(lags) if lags else 0)
    for value, lag in zip(values, lags, strict=True):
        coefficients[lag - 1] = sign * float(value)
    return coefficients


@torch.no_grad()
def solve_recurrence(transition: Tensor, forcing: Tensor) -> Tensor:
    """All states of ``s_t = M_t s_(t-1) + f_t`` with ``s_(-1) = 0`` (parallel prefix scan).

    ``transition`` is ``[T, m, m]`` and ``forcing`` ``[T, m, K]`` (time FIRST here, so that
    the batched products are contiguous); returns ``[T, m, K]``. The inputs are not
    modified. An explosive recursion yields non-finite values, which callers treat as an
    infeasible point.
    """
    n = forcing.shape[0]
    state, maps = forcing.clone(), transition.clone()
    stride = 1
    while stride < n:
        state[stride:] += maps[stride:] @ state[:n - stride]
        if 2 * stride < n:
            # Only maps that are still incomplete (t >= 2 stride) are needed again.
            maps[2 * stride:] = maps[2 * stride:] @ maps[stride:n - stride]
            if not float(maps[2 * stride:].abs().max()) > _NEGLIGIBLE:
                break       # a stable recursion has forgotten everything older than 2 stride
        stride *= 2
    return state


def run_path(
    mean_residual: list[float], omega: float | list[float], *, kind: str,
    arch: Sequence[tuple[float, int]], asymmetry: Sequence[tuple[float, int]],
    garch: Sequence[tuple[float, int]], presample: tuple[float, float, float],
    power: float = 2.0, psi: float = 0.0, archm: str | None = None,
    ar: Sequence[tuple[float, int]] = (), ma: Sequence[tuple[float, int]] = (),
    abs_mean: float = 0.0, signs: list[float] | None = None,
) -> tuple[list[float], list[float], list[float]] | None:
    """The state path of one model by its own recursion (the family's only time loop).

    ``mean_residual`` is ``y_t - x_t b``; ``omega`` the variance constant (a list when
    variance regressors make it time-varying). ``arch``, ``asymmetry``, ``garch``, ``ar``
    and ``ma`` are ``(coefficient, lag)`` pairs. ``presample`` holds the presample values of
    the two news terms and of the variance state v. Per period:

        v_t   = omega_t + sum a_i n1_(t-i) + sum g_i n2_(t-i) + sum b_j v_(t-j)
        h_t   = v_t (garch, gjr) | exp(v_t) (egarch) | v_t^(2/power) (parch)
        u_t   = (y_t - x_t b) - psi g(h_t)                       g: h, sqrt(h) or ln h
        e_t   = u_t - sum rho_j u_(t-j) - sum theta_k e_(t-k)    presample u, e = 0
        news  = e^2 [, e^2 1(e<0)] | z, |z| - E|z| | |e|^power   with z = e / sqrt(h)

    ``signs`` (EGARCH only) replaces |z_t| by ``signs[t] * z_t``: with the signs of a
    reference point the path is a smooth function of the parameters across z_t = 0 (the
    piece of the likelihood on which that point lies; see ``kinks``).

    Returns ``(e, u, v)`` as lists (the variance is a transformation of the state v, left
    to the caller), or ``None`` when the variance leaves its domain (non-positive,
    non-finite or beyond the float64 range of exp).
    """
    if not ar and not ma and all(lag == 1 for _, lag in (*arch, *asymmetry, *garch)) \
            and len(arch) == 1 and len(garch) <= 1:
        return _run_first_order(
            mean_residual, omega, kind, arch[0][0], asymmetry[0][0] if asymmetry else 0.0,
            garch[0][0] if garch else 0.0, presample, power, psi, archm, abs_mean, signs)
    pre1, pre2, pre_v = presample
    longest = max([1] + [lag for _, lag in (*arch, *asymmetry, *garch, *ar, *ma)])
    news1, news2 = [pre1] * longest, [pre2] * longest
    state = [pre_v] * longest
    structural, residual = [0.0] * longest, [0.0] * longest
    constant = None if isinstance(omega, list) else omega
    egarch, parch, gjr = kind == "egarch", kind == "parch", kind == "gjr"
    exponent = 2.0 / power if parch else 1.0
    in_mean = {"variance": 1, "sd": 2, "log": 3}.get(archm or "", 0)
    exp, sqrt, log = math.exp, math.sqrt, math.log
    try:
        for t, value in enumerate(mean_residual):
            v = constant if constant is not None else omega[t]
            for coefficient, lag in arch:
                v += coefficient * news1[-lag]
            for coefficient, lag in asymmetry:
                v += coefficient * news2[-lag]
            for coefficient, lag in garch:
                v += coefficient * state[-lag]
            if egarch:
                if not -700.0 < v < 700.0:
                    return None
                h = exp(v)
            else:
                if not 0.0 < v < 1e300:
                    return None
                h = v ** exponent if parch else v
                if not 0.0 < h < 1e300:
                    return None
            if in_mean == 1:
                value -= psi * h
            elif in_mean == 2:
                value -= psi * sqrt(h)
            elif in_mean == 3:
                value -= psi * log(h)
            e = value
            for coefficient, lag in ar:
                e -= coefficient * structural[-lag]
            for coefficient, lag in ma:
                e -= coefficient * residual[-lag]
            if egarch:
                z = e / sqrt(h)
                news1.append(z)
                news2.append((abs(z) if signs is None else signs[t] * z) - abs_mean)
            elif parch:
                news1.append(abs(e) ** power)
            else:
                square = e * e
                news1.append(square)
                if gjr:
                    news2.append(square if e < 0.0 else 0.0)
            state.append(v)
            structural.append(value)
            residual.append(e)
    except (OverflowError, ZeroDivisionError, ValueError):
        return None                     # a float power or product left the float64 range
    return residual[longest:], structural[longest:], state[longest:]


def _run_first_order(
    mean_residual: list[float], omega: float | list[float], kind: str, a: float, g: float,
    b: float, presample: tuple[float, float, float], power: float, psi: float,
    archm: str | None, abs_mean: float, signs: list[float] | None,
) -> tuple[list[float], list[float], list[float]] | None:
    """``run_path`` for one ARCH lag, at most one GARCH lag and no ARMA terms (scalar state).

    Without ARMA terms the innovation is the structural disturbance, so the same list is
    returned for both (the input itself when there is no ARCH-in-mean term).
    """
    news1, news2, v = presample
    constant = None if isinstance(omega, list) else omega
    egarch, parch, gjr = kind == "egarch", kind == "parch", kind == "gjr"
    exponent = 2.0 / power
    in_mean = {"variance": 1, "sd": 2, "log": 3}.get(archm or "", 0)
    exp, sqrt, log = math.exp, math.sqrt, math.log
    state: list[float] = []
    keep_v = state.append
    try:
        if egarch and not in_mean and constant is not None:
            # EGARCH(1,1): z = e exp(-ln h / 2) needs one exponential per period.
            for t, e in enumerate(mean_residual):
                v = constant + a * news1 + g * news2 + b * v
                if not -700.0 < v < 700.0:
                    return None
                news1 = e * exp(-0.5 * v)
                news2 = (abs(news1) if signs is None else signs[t] * news1) - abs_mean
                keep_v(v)
            return mean_residual, mean_residual, state
        residual: list[float] = []
        keep_e = residual.append
        for t, e in enumerate(mean_residual):
            v = (constant if constant is not None else omega[t]) + a * news1 + g * news2 + b * v
            if egarch:
                if not -700.0 < v < 700.0:
                    return None
                h = exp(v)
            else:
                if not 0.0 < v < 1e300:
                    return None
                h = v ** exponent if parch else v
                if not 0.0 < h < 1e300:
                    return None
            if in_mean:
                e -= psi * (h if in_mean == 1 else sqrt(h) if in_mean == 2 else log(h))
            if egarch:
                news1 = e / sqrt(h)
                news2 = (abs(news1) if signs is None else signs[t] * news1) - abs_mean
            elif parch:
                news1 = abs(e) ** power
            else:
                news1 = e * e
                if gjr:
                    news2 = news1 if e < 0.0 else 0.0
            keep_e(e)
            keep_v(v)
    except (OverflowError, ZeroDivisionError, ValueError):
        return None
    return residual, residual, state
