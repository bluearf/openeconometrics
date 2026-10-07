"""Lag-polynomial filters and small state-space helpers for the ARIMA family.

Everything here works on float64 tensors whose LAST dimension is time and never
loops over observations.

Lag polynomials are dense coefficient lists ``[c_0, c_1, ..., c_m]`` standing for
``c(L) = c_0 + c_1 L + ... + c_m L^m``. Values before the first observation are
zero, which is exactly the "presample values at their expectation" convention of
conditional ARMA estimation.

``apply_polynomial`` computes ``c(L) x`` (a finite impulse response: m shifted
additions). ``inverse_filter`` computes ``x / c(L)`` for ``c(L) = 1 + c_1 L^u +
... + c_q L^(qu)`` (the recursion ``z_t = x_t - c_1 z_(t-u) - ...``) WITHOUT a time
loop, by cyclic reduction: since ``c(L) c(-L)`` is a polynomial in ``L^2``,

    1 / c(L) = c(-L) / d(L^2),        d(L^2) = c(L) c(-L),

the numerator is applied as a finite filter and the same identity is applied to
``1 / d(L^2)`` with the lag unit doubled. The roots of ``d`` are the squares of
the roots of ``c``, so for an invertible polynomial the coefficients of the
remaining denominator vanish doubly exponentially; the iteration stops when they
are below 1e-18 or when the lag unit exceeds the sample (then the remaining
denominator cannot touch the sample and the result is exact for ANY polynomial,
including unit roots). A filter of order q on n points costs about
``q * log2(min(n, decay length))`` vector additions. Polynomials whose roots meet
under squaring (rho and -rho) would lose accuracy in the division; ``inverse_filter``
folds exactly even polynomials into the doubled lag unit, verifies every result by
its residual, refines it when needed and falls back on the companion form.

``transition_powers`` and ``lyapunov`` solve the discrete Lyapunov equation
``P = T P T' + Q`` by the doubling algorithm ``X <- X + A X A', A <- A A`` (the
partial sums of ``sum_j T^j Q T'^j``), which also serves a batch of right-hand
sides for the derivative equations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

NEGLIGIBLE = 1e-18
_LYAPUNOV_STAGES = 48
_RESIDUAL_TOL = 1e-13       # relative residual above which an inverse filter is refined
_REFINEMENTS = 3
_CHECK_POINTS = 2048        # periods in each half of the residual sample


def lag_polynomial(coefficients: Sequence[float], *, unit: int = 1,
                   sign: float = 1.0) -> list[float]:
    """Dense coefficients of ``1 + sign * sum_i coefficients[i-1] L^(i*unit)``."""
    polynomial = [0.0] * (len(coefficients) * unit + 1)
    polynomial[0] = 1.0
    for i, value in enumerate(coefficients, start=1):
        polynomial[i * unit] = sign * float(value)
    return polynomial


def multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Product of two lag polynomials."""
    product = [0.0] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        if left != 0.0:
            for j, right in enumerate(b):
                product[i + j] += left * right
    return product


def apply_polynomial(x: Tensor, polynomial: Sequence[float]) -> Tensor:
    """``c(L) x`` along the last dimension, with zeros before the sample."""
    n = x.shape[-1]
    out = x.clone() if polynomial[0] == 1.0 else x * polynomial[0]
    for lag in range(1, min(len(polynomial), n)):
        weight = polynomial[lag]
        if weight != 0.0:
            out[..., lag:].add_(x[..., :n - lag], alpha=weight)
    return out


def _fold(c: list[float], step: int) -> tuple[list[float], int]:
    """Rewrite a polynomial whose odd coefficients are all zero in the doubled lag unit."""
    while len(c) > 2 and not any(c[1::2]):
        c, step = c[::2], 2 * step
    while len(c) > 1 and c[-1] == 0.0:
        c = c[:-1]
    return c, step


def _reduce(x: Tensor, c: list[float], step: int) -> Tensor:
    """One cyclic-reduction pass: ``x / c(L^step)`` for ``c = [1, c_1, ..., c_q]``."""
    n = x.shape[-1]
    c, step = _fold(c, step)
    while len(c) > 1 and step < n:
        q = len(c) - 1
        largest = max(abs(value) for value in c[1:])
        if not math.isfinite(largest):
            return torch.full_like(x, math.nan)
        if largest <= NEGLIGIBLE:
            break
        out = x.clone()
        for i in range(1, q + 1):
            lag = i * step
            if lag >= n:
                break
            if c[i] != 0.0:
                out[..., lag:].add_(x[..., :n - lag], alpha=-c[i] if i % 2 else c[i])
        x = out
        # d_j = sum_i (-1)^i c_i c_(2j-i): the coefficients of c(L) c(-L) in powers of L^2.
        c = [1.0] + [
            sum((-c[i] if i % 2 else c[i]) * c[2 * j - i]
                for i in range(max(0, 2 * j - q), min(q, 2 * j) + 1))
            for j in range(1, q + 1)
        ]
        c, step = _fold(c, 2 * step)
    return x


def _companion_filter(x: Tensor, c: list[float], step: int) -> Tensor:
    """``x / c(L^step)`` by doubling on the companion form (the fallback of inverse_filter).

    With z_t = C z_(t-step) + e_1 x_t (C the companion matrix of c) the output is the first
    state. The partial sums sum_(i < 2^k) C^i e_1 x_(t - i step) are built by doubling with
    the matrix powers C^(2^k); no polynomial is divided, so repeated roots do no harm.
    The cost is q^2 instead of q operations per observation and stage.
    """
    n, q = x.shape[-1], len(c) - 1
    companion = torch.zeros((q, q), dtype=torch.float64)
    companion[0] = -torch.tensor(c[1:], dtype=torch.float64)
    if q > 1:
        companion[1:, :-1] = torch.eye(q - 1, dtype=torch.float64)
    state = torch.zeros((*x.shape, q), dtype=torch.float64)
    state[..., 0] = x
    power = companion.T.contiguous()            # row convention: z' <- z' C'
    lag = step
    while lag < n:
        size = float(power.abs().max())
        if not math.isfinite(size):
            return torch.full_like(x, math.nan)
        if size <= NEGLIGIBLE:
            break
        state[..., lag:, :] += state[..., :n - lag, :] @ power
        power = power @ power
        lag *= 2
    return state[..., 0].contiguous()


def inverse_filter(x: Tensor, coefficients: Sequence[float], unit: int = 1) -> Tensor:
    """``x / (1 + sum_i coefficients[i-1] L^(i*unit))`` along the last dimension.

    Cyclic reduction (see the module docstring); no loop over time. The input
    is not modified. A polynomial whose inverse explodes beyond the float64
    range yields non-finite values, which callers treat as an infeasible point.

    Accuracy. The reduction divides by c(L) c(-L); when c has roots rho and -rho (or
    roots that meet after repeated squaring, as the seasonal factors of a dense
    polynomial near (1 - L)(1 - L^s) do) the squared polynomial has repeated roots and
    rounding in its coefficients is amplified. Exactly even polynomials are therefore
    rewritten in the doubled lag unit, and the result of every polynomial of order two
    or more is verified by its residual r = x - c(L) out (``_verified``): while the test
    fails the result is refined (out += r / c(L), the classical iterative refinement, at
    most three times), and a result that still fails is recomputed on the companion form.
    """
    n = x.shape[-1]
    c = [1.0, *(float(value) for value in coefficients)]
    step = max(1, int(unit))
    c, step = _fold(c, step)
    if len(c) == 1 or step >= n:
        return x.clone()
    if not all(math.isfinite(value) for value in c):
        return torch.full_like(x, math.nan)
    out = _reduce(x, c, step)
    if len(c) == 2 or not bool(torch.isfinite(out).all()):
        return out                      # a first-order polynomial has no roots that can meet
    for attempt in range(_REFINEMENTS + 1):
        if _verified(x, out, c, step):
            return out
        if attempt < _REFINEMENTS:
            sparse = [0.0] * ((len(c) - 1) * step + 1)
            sparse[::step] = c
            out = out + _reduce(x - apply_polynomial(out, sparse), c, step)
            if not bool(torch.isfinite(out).all()):
                break
    return _companion_filter(x, c, step)


def _verified(x: Tensor, out: Tensor, c: list[float], step: int) -> bool:
    """Residual test of ``out = x / c(L^step)`` on a sample of periods.

    r_t = x_t - sum_i c_i out_(t - i step) is evaluated at the last periods and at evenly
    spaced ones (at most about 4000 in all; inaccuracy of the reduction is systematic and
    grows with t), and compared with 1e-13 (max|x| + sum|c_i| max|out|) over the sample.
    """
    n = x.shape[-1]
    weight = sum(abs(value) for value in c)
    if n <= 2 * _CHECK_POINTS:                  # short series: the whole residual
        residual = x - out
        for i in range(1, len(c)):
            lag = i * step
            if c[i] != 0.0 and lag < n:
                residual[..., lag:].add_(out[..., :n - lag], alpha=-c[i])
        scale = x.abs().amax(dim=-1, keepdim=True) + weight * out.abs().amax(dim=-1, keepdim=True)
        return bool((residual.abs() <= _RESIDUAL_TOL * scale).all())
    index = torch.cat([torch.arange(0, n - _CHECK_POINTS, (n - _CHECK_POINTS) // _CHECK_POINTS),
                       torch.arange(n - _CHECK_POINTS, n)])
    sample = out[..., index]
    residual = x[..., index] - sample
    for i in range(1, len(c)):
        lag = i * step
        if c[i] != 0.0 and lag < n:
            reach = index >= lag
            residual[..., reach] -= c[i] * out[..., index[reach] - lag]
    scale = x[..., index].abs().amax(dim=-1, keepdim=True) \
        + weight * sample.abs().amax(dim=-1, keepdim=True)
    return bool((residual.abs() <= _RESIDUAL_TOL * scale).all())


def shift(x: Tensor, lag: int) -> Tensor:
    """``L^lag x`` along the last dimension (zeros enter at the start)."""
    out = torch.zeros_like(x)
    n = x.shape[-1]
    if lag < n:
        out[..., lag:] = x[..., :n - lag]
    return out


def difference(x: Tensor, d: int, seasonal_d: int = 0, period: int = 0) -> Tensor:
    """``(1 - L)^d (1 - L^s)^D x`` along the FIRST dimension; drops d + D*s leading rows."""
    for _ in range(seasonal_d):
        x = x[period:] - x[:-period]
    for _ in range(d):
        x = x[1:] - x[:-1]
    return x


def difference_polynomial(d: int, seasonal_d: int = 0, period: int = 0) -> list[float]:
    """Dense coefficients of ``(1 - L)^d (1 - L^s)^D``."""
    polynomial = [1.0]
    for _ in range(d):
        polynomial = multiply(polynomial, [1.0, -1.0])
    for _ in range(seasonal_d):
        polynomial = multiply(polynomial, lag_polynomial([1.0], unit=period, sign=-1.0))
    return polynomial


def inverse_roots(coefficients: Sequence[float]) -> Tensor:
    """Inverse roots of ``1 - a_1 z - ... - a_m z^m`` (eigenvalues of its companion matrix).

    The polynomial is stationary (an AR polynomial) or invertible (an MA
    polynomial ``1 + theta_1 z + ...``, passed as ``a_i = -theta_i``) when every
    inverse root lies strictly inside the unit circle. Returns a complex128
    vector; trailing zero coefficients are harmless (they add zero roots).
    """
    m = len(coefficients)
    if m == 0:
        return torch.zeros(0, dtype=torch.complex128)
    companion = torch.zeros((m, m), dtype=torch.float64)
    companion[0] = torch.as_tensor(list(coefficients), dtype=torch.float64)
    if m > 1:
        companion[1:, :-1] = torch.eye(m - 1, dtype=torch.float64)
    return torch.linalg.eigvals(companion)


def spectral_radius(coefficients: Sequence[float]) -> float:
    """Largest modulus of the inverse roots of ``1 - a_1 z - ... - a_m z^m``."""
    m = len(coefficients)
    if m == 0:
        return 0.0
    if m == 1:
        return abs(float(coefficients[0]))
    if not all(math.isfinite(float(value)) for value in coefficients):
        return math.inf
    return float(inverse_roots(coefficients).abs().max())


def transition_powers(transition: Tensor) -> list[Tensor] | None:
    """``[T, T^2, T^4, ...]`` until the power is negligible; None when T is not stable."""
    powers = []
    power = transition
    for _ in range(_LYAPUNOV_STAGES):
        size = float(power.abs().max())
        if not math.isfinite(size) or size > 1e100:
            return None
        powers.append(power)
        if size <= NEGLIGIBLE:
            return powers
        power = power @ power
    return None


def lyapunov(powers: Sequence[Tensor], q: Tensor) -> Tensor:
    """Solution ``X = sum_j T^j Q T'^j`` of ``X = T X T' + Q`` for one or a batch of Q."""
    x = q
    for power in powers:
        x = x + power @ x @ power.T
    return x


def levinson(autocovariances: Sequence[float], order: int) -> tuple[list[float], list[float]]:
    """Durbin-Levinson recursion on autocovariances (or autocorrelations) g_0..g_order.

    Returns the Yule-Walker AR coefficients of order ``order`` and the partial
    autocorrelations phi_11, ..., phi_mm. Stops early (zeros afterwards) if the
    prediction variance vanishes.
    """
    g = [float(value) for value in autocovariances]
    coefficients: list[float] = []
    partial: list[float] = []
    variance = g[0]
    for m in range(1, order + 1):
        if not variance > 0.0:
            partial.append(0.0)
            coefficients.append(0.0)
            continue
        reflection = (g[m] - sum(coefficients[j] * g[m - 1 - j] for j in range(m - 1))) / variance
        coefficients = [coefficients[j] - reflection * coefficients[m - 2 - j]
                        for j in range(m - 1)] + [reflection]
        partial.append(reflection)
        variance *= 1.0 - reflection * reflection
    return coefficients, partial
