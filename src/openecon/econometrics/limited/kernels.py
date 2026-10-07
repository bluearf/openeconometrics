"""Censored and truncated normal likelihood kernels on float64 tensors.

No autograd: every objective returns its analytic gradient and Hessian, which
the tests compare with numerical derivatives. ``w_i`` is the weight of
observation i in every likelihood sum (ones when unweighted). All work is
O(n k^2); nothing n-by-n is formed and there is no loop over observations.

Parameters are ``theta = (b, tau)`` with ``tau = ln sigma``; the linear
predictor is ``m_i = x_i'b + offset_i`` and ``s = exp(tau)``.

Row types
---------
Every observation contributes one of three kinds of terms. With ``phi`` /
``Phi`` the standard normal density / distribution function:

* point (an observed value ``y``), ``z = (y - m) / s``:

      l = -tau - z^2 / 2 - ln sqrt(2 pi),
      l_m = z / s,            l_tau = z^2 - 1,
      l_mm = -1 / s^2,        l_m,tau = -2 z / s,      l_tau,tau = -2 z^2;

* one-sided tail ``l = ln Phi(t)``, ``t = d (c - m) / s`` with ``d = +1`` for
  ``y* <= c`` (left-censored) and ``d = -1`` for ``y* >= c`` (right-censored).
  With the inverse Mills ratio ``r = phi(t) / Phi(t)`` and ``v = r (r + t)``:

      l_m = -d r / s,         l_tau = -r t,
      l_mm = -v / s^2,        l_m,tau = d (r - v t) / s,   l_tau,tau = r t - v t^2;

* interval ``l = ln[Phi(b) - Phi(a)]``, ``a = (lo - m) / s < b = (hi - m) / s``.
  With ``A = phi(a) / P``, ``B = phi(b) / P`` and ``P = Phi(b) - Phi(a)``:

      l_m = (A - B) / s,      l_tau = a A - b B,
      l_mm = [a A - b B - (A - B)^2] / s^2,
      l_m,tau = [(B - A) + a^2 A - b^2 B - (A - B)(a A - b B)] / s,
      l_tau,tau = (b B - a A) + a^3 A - b^3 B - (a A - b B)^2.

``ln Phi`` uses ``log_ndtr`` and the Mills ratio the scaled complementary
error function, so both tails are stable; ``P`` is formed in the tail where
both distribution functions are small (``Phi(b) - Phi(a) = Phi(-a) - Phi(-b)``).

Objectives
----------
``CensoredObjective``: tobit and interval regression. Each observation is a
point, a left- or right-censored value or an interval; the log likelihood is
the sum of the corresponding terms.

``TruncatedObjective``: truncated regression. Each observation is a point whose
density is divided by ``Pr(ll < y* < ul)``; the log likelihood is the point
term minus a tail (one limit) or interval (two limits) term.

The gradient and Hessian in ``(b, tau)`` follow from the row derivatives:

    g = [X'(w l_m), sum w l_tau],
    H = [[X' diag(w l_mm) X, X'(w l_m,tau)], [., sum w l_tau,tau]].
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import weighted_crossprod

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)

Rows = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def log_interval_probability(a: Tensor, b: Tensor) -> Tensor:
    """``ln[Phi(b) - Phi(a)]`` for ``a < b``, formed where both tails are small."""
    reflect = (a + b) > 0
    low = torch.where(reflect, -b, a)
    high = torch.where(reflect, -a, b)
    log_high = torch.special.log_ndtr(high)
    return log_high + torch.log(-torch.expm1(torch.special.log_ndtr(low) - log_high))


def point_rows(z: Tensor, inverse: Tensor) -> Rows:
    """Log density (without ``-tau``) of a point and its derivatives in ``(m, tau)``."""
    square = z.square()
    return (-0.5 * square - _LOG_SQRT_2PI, z * inverse, square - 1,
            (-inverse.square()).expand_as(z), -2 * z * inverse, -2 * square)


def tail_rows(t: Tensor, sign: Tensor, inverse: Tensor) -> Rows:
    """``ln Phi(t)``, ``t = sign (c - m) / s``, and its derivatives in ``(m, tau)``."""
    log_cdf, ratio, curvature = mills_ratio(t)
    return (log_cdf, -sign * ratio * inverse, -ratio * t, -curvature * inverse.square(),
            sign * (ratio - curvature * t) * inverse, ratio * t - curvature * t.square())


def interval_rows(a: Tensor, b: Tensor, inverse: Tensor) -> Rows:
    """``ln[Phi(b) - Phi(a)]`` for finite ``a < b`` and its derivatives in ``(m, tau)``."""
    log_p = log_interval_probability(a, b)
    ratio_a = torch.exp(-0.5 * a.square() - _LOG_SQRT_2PI - log_p)
    ratio_b = torch.exp(-0.5 * b.square() - _LOG_SQRT_2PI - log_p)
    slope_a, slope_b = a * ratio_a, b * ratio_b
    difference, moment = ratio_a - ratio_b, slope_a - slope_b
    return (log_p, difference * inverse, moment,
            (moment - difference.square()) * inverse.square(),
            (-difference + a * slope_a - b * slope_b - difference * moment) * inverse,
            -moment + a.square() * slope_a - b.square() * slope_b - moment.square())


def _rejected(size: int) -> tuple[Tensor, Tensor, Tensor]:
    """A non-finite objective for a trial point outside the float64 range."""
    return (torch.tensor(-math.inf, dtype=torch.float64), torch.zeros(size, dtype=torch.float64),
            -torch.eye(size, dtype=torch.float64))


class _NormalObjective:
    """Assembly of value, gradient, Hessian and score rows from row derivatives."""

    def __init__(self, x: Tensor, weights: Tensor, offset: Tensor | None):
        if x.ndim != 2 or weights.shape != (x.shape[0],):
            raise KernelError("invalid_design", "The design and the weights do not conform.")
        self.x, self.w, self.offset = x, weights, offset
        self.n, self.k = x.shape
        self.size = self.k + 1

    def index(self, theta: Tensor) -> Tensor:
        """Linear predictor ``x'b + offset``."""
        index = self.x @ theta[:self.k]
        return index if self.offset is None else index + self.offset

    def _rows(self, theta: Tensor) -> Rows:           # pragma: no cover - abstract
        raise NotImplementedError

    def _log_rows(self, theta: Tensor) -> Tensor:     # pragma: no cover - abstract
        raise NotImplementedError

    def value(self, theta: Tensor) -> Tensor:
        if abs(float(theta[self.k])) > 300:
            return torch.tensor(-math.inf, dtype=torch.float64)
        return (self.w * self._log_rows(theta)).sum()

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if abs(float(theta[self.k])) > 300:
            return _rejected(self.size)
        log_l, g_m, g_t, h_mm, h_mt, h_tt = self._rows(theta)
        k, w, x = self.k, self.w, self.x
        gradient = torch.empty(self.size, dtype=torch.float64)
        gradient[:k] = x.T @ (w * g_m)
        gradient[k] = (w * g_t).sum()
        hessian = torch.empty((self.size, self.size), dtype=torch.float64)
        hessian[:k, :k] = weighted_crossprod(x, w * h_mm)
        cross = x.T @ (w * h_mt)
        hessian[:k, k] = cross
        hessian[k, :k] = cross
        hessian[k, k] = (w * h_tt).sum()
        return (w * log_l).sum(), gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, k + 1], before weights."""
        _, g_m, g_t, _, _, _ = self._rows(theta)
        return torch.cat([self.x * g_m[:, None], g_t[:, None]], dim=1)


class CensoredRows:
    """Row terms of point, censored and interval observations given the index and ``ln sigma``.

    ``lower`` and ``upper`` bound the latent outcome of each observation:
    equal finite bounds are a point, ``lower = -inf`` a left-censored value
    (``y* <= upper``), ``upper = +inf`` a right-censored value
    (``y* >= lower``) and two different finite bounds an interval. The object
    is shared by the tobit / interval objectives and the outcome equation of
    ``ivtobit``.
    """

    has_scale = True

    def __init__(self, lower: Tensor, upper: Tensor):
        if lower.ndim != 1 or lower.shape != upper.shape:
            raise KernelError("invalid_design", "One pair of bounds is needed per observation.")
        low_open, high_open = torch.isinf(lower), torch.isinf(upper)
        if bool((lower > upper).any()) or bool((low_open & high_open).any()) \
                or bool(torch.isnan(lower).any()) or bool(torch.isnan(upper).any()):
            raise KernelError("invalid_interval", "Every observation needs lower <= upper with "
                              "at least one finite bound.")
        self.n = len(lower)
        point = lower == upper
        tail = low_open | high_open
        self._point = point.nonzero().flatten()
        self._tail = tail.nonzero().flatten()
        self._interval = (~point & ~tail).nonzero().flatten()
        self._y = lower[self._point]
        self._limit = torch.where(low_open, upper, lower)[self._tail]
        self._sign = torch.where(low_open, 1.0, -1.0).to(torch.float64)[self._tail]
        self._low, self._high = lower[self._interval], upper[self._interval]
        self.uniform = bool(point.all())

    def log_rows(self, index: Tensor, tau: Tensor) -> Tensor:
        """Log likelihood of every observation at linear predictor ``index``."""
        inverse = torch.exp(-tau)
        if self.uniform:
            return -0.5 * ((self._y - index) * inverse).square() - _LOG_SQRT_2PI - tau
        out = torch.empty(self.n, dtype=torch.float64)
        if len(self._point):
            z = (self._y - index[self._point]) * inverse
            out[self._point] = -0.5 * z.square() - _LOG_SQRT_2PI - tau
        if len(self._tail):
            t = self._sign * (self._limit - index[self._tail]) * inverse
            out[self._tail] = torch.special.log_ndtr(t)
        if len(self._interval):
            center = index[self._interval]
            out[self._interval] = log_interval_probability(
                (self._low - center) * inverse, (self._high - center) * inverse)
        return out

    def rows(self, index: Tensor, tau: Tensor) -> Rows:
        """Log likelihood and its derivatives in ``(index, tau)`` for every observation."""
        inverse = torch.exp(-tau)
        if self.uniform:
            log_l, *rest = point_rows((self._y - index) * inverse, inverse)
            return (log_l - tau, *rest)
        out = [torch.empty(self.n, dtype=torch.float64) for _ in range(6)]

        def assign(rows: Tensor, values: Rows) -> None:
            for target, value in zip(out, values, strict=True):
                target[rows] = value

        if len(self._point):
            log_l, *rest = point_rows((self._y - index[self._point]) * inverse, inverse)
            assign(self._point, (log_l - tau, *rest))
        if len(self._tail):
            t = self._sign * (self._limit - index[self._tail]) * inverse
            assign(self._tail, tail_rows(t, self._sign, inverse))
        if len(self._interval):
            center = index[self._interval]
            assign(self._interval, interval_rows(
                (self._low - center) * inverse, (self._high - center) * inverse, inverse))
        return tuple(out)


class CensoredObjective(_NormalObjective):
    """Log likelihood of point, censored and interval observations in ``(b, ln sigma)``.

    ``lower`` / ``upper`` are the bounds of ``CensoredRows``.
    """

    def __init__(self, x: Tensor, lower: Tensor, upper: Tensor, weights: Tensor,
                 offset: Tensor | None = None):
        super().__init__(x, weights, offset)
        if lower.shape != (self.n,) or upper.shape != (self.n,):
            raise KernelError("invalid_design", "One pair of bounds is needed per observation.")
        self.terms = CensoredRows(lower, upper)

    def _log_rows(self, theta: Tensor) -> Tensor:
        return self.terms.log_rows(self.index(theta), theta[self.k])

    def _rows(self, theta: Tensor) -> Rows:
        return self.terms.rows(self.index(theta), theta[self.k])


class TruncatedObjective(_NormalObjective):
    """Log likelihood of the normal regression truncated to ``ll < y < ul`` in ``(b, ln sigma)``.

    ``ll`` / ``ul`` are scalars or ``None`` (no truncation on that side):

        l_i = -tau + ln phi(z_i) - ln[Phi((ul - m_i) / s) - Phi((ll - m_i) / s)].
    """

    def __init__(self, x: Tensor, y: Tensor, weights: Tensor, ll: float | None,
                 ul: float | None, offset: Tensor | None = None):
        super().__init__(x, weights, offset)
        if y.shape != (self.n,):
            raise KernelError("invalid_design", "One outcome is needed per observation.")
        if ll is not None and ul is not None and not ll < ul:
            raise KernelError("invalid_limits", "The lower limit must be below the upper limit.")
        self.y, self.ll, self.ul = y, ll, ul

    def _mass(self, index: Tensor, inverse: Tensor) -> Rows | None:
        """Rows of ``ln Pr(ll < y* < ul)``; None without truncation."""
        if self.ll is None and self.ul is None:
            return None
        if self.ll is None:
            positive = torch.ones((), dtype=torch.float64)
            return tail_rows((self.ul - index) * inverse, positive, inverse)
        if self.ul is None:
            negative = -torch.ones((), dtype=torch.float64)
            return tail_rows((index - self.ll) * inverse, negative, inverse)
        return interval_rows((self.ll - index) * inverse, (self.ul - index) * inverse, inverse)

    def _log_rows(self, theta: Tensor) -> Tensor:
        tau = theta[self.k]
        inverse = torch.exp(-tau)
        index = self.index(theta)
        out = -0.5 * ((self.y - index) * inverse).square() - _LOG_SQRT_2PI - tau
        if self.ll is None and self.ul is None:
            return out
        if self.ll is None:
            return out - torch.special.log_ndtr((self.ul - index) * inverse)
        if self.ul is None:
            return out - torch.special.log_ndtr((index - self.ll) * inverse)
        return out - log_interval_probability((self.ll - index) * inverse,
                                              (self.ul - index) * inverse)

    def _rows(self, theta: Tensor) -> Rows:
        tau = theta[self.k]
        inverse = torch.exp(-tau)
        index = self.index(theta)
        log_l, *rest = point_rows((self.y - index) * inverse, inverse)
        rows = (log_l - tau, *rest)
        mass = self._mass(index, inverse)
        if mass is None:
            return rows
        return tuple(density - probability for density, probability in zip(rows, mass,
                                                                           strict=True))
