"""Bivariate normal distribution function and the bivariate probit likelihood.

Bivariate normal probabilities
------------------------------
``bvn_upper(h, k, r) = Pr(X > h, Y > k)`` for standard normal ``X, Y`` with
correlation ``r``, vectorized over observations, by the algorithm of Genz
(2004, "Numerical computation of rectangular bivariate and trivariate normal
and t probabilities", Statistics and Computing 14), a refinement of Drezner and
Wesolowsky (1990):

* ``|r| < 0.925``: Plackett's identity ``dL/dr = phi2(h, k; r)`` integrated
  from 0 to r in the variable ``t = asin(r)``,

      L(h, k, r) = Phi(-h) Phi(-k)
                   + (1 / 2 pi) int_0^{asin r} exp(-(h^2 + k^2 - 2 h k sin t) / (2 cos^2 t)) dt,

  with a 20-point Gauss-Legendre rule.
* ``|r| >= 0.925``: the integral is taken from ``|r|`` to 1 instead (where
  ``L`` is a univariate probability), after subtracting the leading terms of
  the integrand's expansion at ``r = +-1``, which are integrated in closed
  form; the smooth remainder uses the same 20-point rule.

The absolute error is about 1e-15 (the tests compare with adaptive quadrature
of an independent one-dimensional representation). ``bvn_cdf(a, b, r)`` is
``Pr(X <= a, Y <= b) = bvn_upper(-a, -b, r)``. Arguments are clamped to +-40,
beyond which every probability involved is exactly 0 or 1 in float64.

Logarithm in the tails
----------------------
A likelihood needs ``ln Phi2`` with RELATIVE accuracy, which an absolute error
of 1e-15 does not give for a small probability (an outlying observation).
``log_bvn_cdf`` therefore recomputes every probability below 1e-5 from the
representation with a positive integrand,

    Phi2(a, b; r) = int_{-inf}^{a} phi(x) Phi((b - r x) / sqrt(1 - r^2)) dx,   a <= b,

whose logarithm ``g(x)`` is concave with ``g'' <= -1``. The mode of ``g`` on
``(-inf, a]`` (the endpoint ``a`` unless ``g'(a) < 0``) and the points where
``g`` has fallen by 40 from its maximum are bracketed by bisection, and the
integral over that interval is a 96-point Gauss-Legendre sum accumulated in
logarithms (``logsumexp``). The result has no cancellation and no underflow:
it is accurate to about 1e-12 in relative terms for probabilities as small as
``exp(-1e5)``, like ``log_ndtr`` in one dimension.

Bivariate probit (``BiprobitObjective``)
----------------------------------------
With ``q_j = 2 y_j - 1``, ``w_j = q_j x_j'b_j`` and ``r* = q_1 q_2 rho``, the
log likelihood of an observation is ``ln Phi2(w_1, w_2; r*)`` and ``rho`` is
parameterized as ``rho = tanh(a)`` (Stata's ``/athrho``). With
``d = (1 - r*^2)^(-1/2)``, ``v_1 = d (w_2 - r* w_1)``, ``v_2 = d (w_1 - r* w_2)``,
``g_1 = phi(w_1) Phi(v_1)``, ``g_2 = phi(w_2) Phi(v_2)``, the bivariate density
``f = phi2(w_1, w_2; r*)`` and the ratios ``G_j = g_j / Phi2``, ``D = f / Phi2``:

    dl/dw_j = G_j,                        dl/dr* = D,
    d2l/dw_1^2    = -w_1 G_1 - r* D - G_1^2      (and symmetrically for w_2),
    d2l/dw_1 dw_2 = D - G_1 G_2,
    d2l/dw_1 dr*  = -D (d v_2 + G_1),     d2l/dw_2 dr* = -D (d v_1 + G_2),
    d2l/dr*^2     = D [d^2 r* (1 - Q) + d^2 w_1 w_2 - D],
    Q = d^2 (w_1^2 + w_2^2 - 2 r* w_1 w_2),

and the chain rule with ``dr*/da = q_1 q_2 (1 - rho^2)`` and
``d2 rho/da2 = -2 rho (1 - rho^2)`` gives the analytic score and Hessian in
``(b_1, b_2, a)``. ``1 - rho^2`` is computed as ``1 / cosh(a)^2`` so that it
keeps full relative accuracy when ``rho`` is close to one.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.distributions import gauss_legendre

_TWO_PI = 2 * math.pi
_SPLIT = 0.925          # Genz's switch between the two integral representations
_LIMIT = 40.0           # |argument| beyond which Phi is exactly 0 or 1 in float64
_TAIL = 1e-5            # probabilities below this are recomputed with relative accuracy
_DROP = 40.0            # the tail integrand is negligible exp(-40) below its maximum
_TAIL_NODES = 96
_LOG_SQRT_2PI = 0.5 * math.log(_TWO_PI)
_rule: tuple[Tensor, Tensor] | None = None
_tail_rule: tuple[Tensor, Tensor] | None = None


def _nodes() -> tuple[Tensor, Tensor]:
    """20-point Gauss-Legendre rule mapped to [0, 2]: abscissae ``1 + x`` and weights."""
    global _rule
    if _rule is None:
        x, w = gauss_legendre(20)
        _rule = (1 + x, w)
    return _rule


def _moderate(h: Tensor, k: Tensor, r: Tensor) -> Tensor:
    x, w = _nodes()
    half = torch.asin(r) / 2
    sine = torch.sin(half[:, None] * x)
    exponent = (sine * (h * k)[:, None] - ((h.square() + k.square()) / 2)[:, None]) \
        / (1 - sine.square())
    integral = (torch.exp(exponent) * w).sum(dim=1) * half / _TWO_PI
    return integral + torch.special.ndtr(-h) * torch.special.ndtr(-k)


def _extreme(h: Tensor, k: Tensor, r: Tensor, complement: Tensor) -> Tensor:
    x, w = _nodes()
    negative = r < 0
    k = torch.where(negative, -k, k)
    hk = h * k
    interior = complement > 0                           # |r| < 1
    safe = torch.where(interior, complement, torch.ones_like(complement))
    a = safe.sqrt()
    bs = (h - k).square()
    c = (4 - hk) / 8
    d = (12 - hk) / 16
    exponent = -(bs / safe + hk) / 2
    total = torch.where(
        exponent > -100,
        a * torch.exp(exponent)
        * (1 - c * (bs - safe) * (1 - d * bs / 5) / 3 + c * d * safe.square() / 5),
        torch.zeros_like(h))
    b = bs.sqrt()
    total = total - torch.where(
        hk > -100,
        torch.exp(-hk / 2) * math.sqrt(_TWO_PI) * torch.special.ndtr(-b / a) * b
        * (1 - c * bs * (1 - d * bs / 5) / 3),
        torch.zeros_like(h))
    half = a / 2
    xs = (half[:, None] * x).square()
    rs = (1 - xs).sqrt()
    inner = -(bs[:, None] / xs + hk[:, None]) / 2
    tilt = -hk[:, None] * xs / (2 * (1 + rs).square())
    terms = torch.exp(inner + tilt) / rs \
        - torch.exp(inner) * (1 + c[:, None] * xs * (1 + d[:, None] * xs))
    terms = torch.where(inner > -100, terms, torch.zeros_like(terms))
    total = total + half * (terms * w).sum(dim=1)
    total = torch.where(interior, -total / _TWO_PI, torch.zeros_like(total))
    upper = torch.special.ndtr(-torch.maximum(h, k))
    between = torch.where(h < 0, torch.special.ndtr(k) - torch.special.ndtr(h),
                          torch.special.ndtr(-h) - torch.special.ndtr(-k))
    lower = -total + torch.where(k > h, between, torch.zeros_like(between))
    return torch.where(negative, lower, total + upper)


def bvn_upper(h: Tensor, k: Tensor, r: Tensor | float, complement: Tensor | float | None = None
              ) -> Tensor:
    """``Pr(X > h, Y > k)`` for a standard bivariate normal with correlation ``r``.

    ``h``, ``k`` and ``r`` broadcast against each other; ``complement`` optionally
    supplies ``1 - r^2`` when the caller knows it more accurately than
    ``(1 - r)(1 + r)`` (correlations within 1e-8 of one).
    """
    h = torch.as_tensor(h, dtype=torch.float64)
    k = torch.as_tensor(k, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64)
    if complement is None:
        complement = (1 - r) * (1 + r)
    complement = torch.as_tensor(complement, dtype=torch.float64)
    h, k, r, complement = torch.broadcast_tensors(h, k, r, complement)
    shape = h.shape
    h = h.reshape(-1).clamp(-_LIMIT, _LIMIT)
    k = k.reshape(-1).clamp(-_LIMIT, _LIMIT)
    r, complement = r.reshape(-1), complement.reshape(-1)
    moderate = r.abs() < _SPLIT
    if bool(moderate.all()):
        out = _moderate(h, k, r)
    elif not bool(moderate.any()):
        out = _extreme(h, k, r, complement)
    else:
        out = torch.empty_like(h)
        rest = ~moderate
        out[moderate] = _moderate(h[moderate], k[moderate], r[moderate])
        out[rest] = _extreme(h[rest], k[rest], r[rest], complement[rest])
    return out.clamp(0.0, 1.0).reshape(shape)


def bvn_cdf(a: Tensor, b: Tensor, r: Tensor | float, complement: Tensor | float | None = None
            ) -> Tensor:
    """``Phi2(a, b; r) = Pr(X <= a, Y <= b)``, Stata's ``binormal(a, b, r)``."""
    return bvn_upper(-torch.as_tensor(a, dtype=torch.float64),
                     -torch.as_tensor(b, dtype=torch.float64), r, complement)


def _bisect(left_side, left: Tensor, right: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    """Shrink ``[left, right]`` around the point where ``left_side(x)`` turns false."""
    for _ in range(steps):
        middle = (left + right) / 2
        flag = left_side(middle)
        left, right = torch.where(flag, middle, left), torch.where(flag, right, middle)
    return left, right


def _tail_log_integrand(x: Tensor, other: Tensor, r: Tensor, scale: Tensor) -> Tensor:
    """``g(x) = ln[phi(x) Phi((other - r x) / scale)]``."""
    return -0.5 * x.square() - _LOG_SQRT_2PI + torch.special.log_ndtr((other - r * x) / scale)


def _tail_rising(x: Tensor, other: Tensor, r: Tensor, scale: Tensor) -> Tensor:
    """``g'(x) > 0``: ``x`` lies to the left of the mode of the tail integrand."""
    u = (other - r * x) / scale
    lower = math.sqrt(2 / math.pi) / torch.special.erfcx(-torch.clamp(u, max=0.0) / math.sqrt(2))
    upper = torch.exp(-0.5 * u.square() - _LOG_SQRT_2PI - torch.special.log_ndtr(u))
    return -x - (r / scale) * torch.where(u < 0, lower, upper) > 0


def _log_tail(a: Tensor, b: Tensor, r: Tensor, complement: Tensor) -> Tensor:
    """``ln Phi2(a, b; r)`` from the positive one-dimensional integrand (see module notes)."""
    global _tail_rule
    if _tail_rule is None:
        x, w = gauss_legendre(_TAIL_NODES)
        _tail_rule = ((1 + x) / 2, torch.log(w / 2))
    limit, other = torch.minimum(a, b), torch.maximum(a, b)
    scale = complement.sqrt()
    # Mode of the concave log integrand on (-inf, limit]: the endpoint unless g'(limit) < 0.
    mode = limit.clone()
    interior = ~_tail_rising(limit, other, r, scale)
    if bool(interior.any()):
        pick = interior.nonzero().flatten()
        shape = (other[pick], r[pick], scale[pick])
        right, width = limit[pick], torch.ones(len(pick), dtype=torch.float64)
        left = right - width
        for _ in range(64):
            short = ~_tail_rising(left, *shape)
            if not bool(short.any()):
                break
            width = torch.where(short, 2 * width, width)
            right = torch.where(short, left, right)
            left = torch.where(short, left - width, left)
        mode[pick] = _bisect(lambda x: _tail_rising(x, *shape), left, right, 40)[1]
    top = _tail_log_integrand(mode, other, r, scale)
    floor = top - _DROP
    reach = math.sqrt(2 * _DROP)                        # g'' <= -1 bounds both cuts
    low = _bisect(lambda x: _tail_log_integrand(x, other, r, scale) < floor,
                  mode - reach, mode, 30)[0]
    high = _bisect(lambda x: _tail_log_integrand(x, other, r, scale) >= floor,
                   mode, torch.minimum(limit, mode + reach), 30)[1]
    nodes, log_weights = _tail_rule
    length = high - low
    values = _tail_log_integrand(low[:, None] + length[:, None] * nodes, other[:, None],
                                 r[:, None], scale[:, None]) + log_weights
    result = torch.logsumexp(values, dim=1) + torch.log(length)
    return torch.where(torch.isfinite(top), result, torch.full_like(top, -math.inf))


def log_bvn_cdf(a: Tensor, b: Tensor, r: Tensor | float,
                complement: Tensor | float | None = None) -> Tensor:
    """``ln Phi2(a, b; r)`` with relative accuracy, also for vanishing probabilities."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64)
    if complement is None:
        complement = (1 - r) * (1 + r)
    complement = torch.as_tensor(complement, dtype=torch.float64)
    a, b, r, complement = torch.broadcast_tensors(a, b, r, complement)
    shape = a.shape
    a, b, r, complement = (value.reshape(-1) for value in (a, b, r, complement))
    probability = bvn_cdf(a, b, r, complement)
    small = ~(probability > _TAIL) & (complement > 0)
    out = torch.log(probability)
    if bool(small.any()):
        out = out.clone()
        out[small] = _log_tail(a[small], b[small], r[small], complement[small])
    return out.reshape(shape)


class BiprobitObjective:
    """Log likelihood of the bivariate probit in ``(b_1, b_2, atanh rho)``."""

    def __init__(self, x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, weights: Tensor):
        self.x1, self.x2, self.w = x1, x2, weights
        self.q1, self.q2 = 2 * y1 - 1, 2 * y2 - 1
        self.sign = self.q1 * self.q2
        self.k1, self.k2 = x1.shape[1], x2.shape[1]
        self.size = self.k1 + self.k2 + 1

    def _rejected(self) -> tuple[Tensor, Tensor, Tensor]:
        return (torch.tensor(-math.inf, dtype=torch.float64),
                torch.zeros(self.size, dtype=torch.float64),
                -torch.eye(self.size, dtype=torch.float64))

    def _probability(self, theta: Tensor) -> tuple[Tensor, ...] | None:
        alpha = float(theta[-1])
        if not math.isfinite(alpha) or abs(alpha) > 300:
            return None
        rho, complement = math.tanh(alpha), 1 / math.cosh(alpha) ** 2
        w1 = self.q1 * (self.x1 @ theta[:self.k1])
        w2 = self.q2 * (self.x2 @ theta[self.k1:self.k1 + self.k2])
        r = self.sign * rho
        log_p = log_bvn_cdf(w1, w2, r, complement)
        if not bool(torch.isfinite(log_p).all()):
            return None
        return w1, w2, r, log_p, rho, complement

    def value(self, theta: Tensor) -> Tensor:
        pieces = self._probability(theta)
        if pieces is None:
            return torch.tensor(-math.inf, dtype=torch.float64)
        return (self.w * pieces[3]).sum()

    def marginal_probabilities(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """Fitted ``Pr(y_1 = 1)`` and ``Pr(y_2 = 1)``."""
        return (torch.special.ndtr(self.x1 @ theta[:self.k1]),
                torch.special.ndtr(self.x2 @ theta[self.k1:self.k1 + self.k2]))

    def outcome_probabilities(self, theta: Tensor) -> Tensor:
        """Fitted joint probability of each observation's own pair of outcomes."""
        pieces = self._probability(theta)
        if pieces is None:
            return torch.zeros(len(self.w), dtype=torch.float64)
        return torch.exp(pieces[3])

    def _ratios(self, theta: Tensor):
        pieces = self._probability(theta)
        if pieces is None:
            return None
        w1, w2, r, log_p, rho, complement = pieces
        scale = math.cosh(float(theta[-1]))                              # d = (1 - rho^2)^-1/2
        v1, v2 = scale * (w2 - r * w1), scale * (w1 - r * w2)
        log_phi = -_LOG_SQRT_2PI
        g1 = torch.exp(log_phi - 0.5 * w1.square() + torch.special.log_ndtr(v1) - log_p)
        g2 = torch.exp(log_phi - 0.5 * w2.square() + torch.special.log_ndtr(v2) - log_p)
        form = (w1.square() + w2.square() - 2 * r * w1 * w2) / complement  # Q
        density = torch.exp(-0.5 * form + math.log(scale) - math.log(_TWO_PI) - log_p)
        return w1, w2, r, log_p, g1, g2, density, v1, v2, form, rho, complement, scale

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pieces = self._ratios(theta)
        if pieces is None:
            return self._rejected()
        w1, w2, r, log_p, g1, g2, density, v1, v2, form, rho, complement, scale = pieces
        k1, k2, w = self.k1, self.k2, self.w
        last = k1 + k2
        value = (w * log_p).sum()
        gradient = torch.empty(self.size, dtype=torch.float64)
        gradient[:k1] = self.x1.T @ (w * self.q1 * g1)
        gradient[k1:last] = self.x2.T @ (w * self.q2 * g2)
        gradient[last] = (w * self.sign * density).sum() * complement
        h11 = -w1 * g1 - r * density - g1.square()
        h22 = -w2 * g2 - r * density - g2.square()
        h12 = density - g1 * g2
        h1r = -density * (scale * v2 + g1)
        h2r = -density * (scale * v1 + g2)
        hrr = density * (r * (1 - form) / complement + w1 * w2 / complement - density)
        hessian = torch.empty((self.size, self.size), dtype=torch.float64)
        hessian[:k1, :k1] = self.x1.T @ (self.x1 * (w * h11)[:, None])
        hessian[k1:last, k1:last] = self.x2.T @ (self.x2 * (w * h22)[:, None])
        cross = self.x1.T @ (self.x2 * (w * self.sign * h12)[:, None])
        hessian[:k1, k1:last] = cross
        hessian[k1:last, :k1] = cross.T
        first = self.x1.T @ (w * self.q2 * h1r) * complement
        second = self.x2.T @ (w * self.q1 * h2r) * complement
        hessian[:k1, last] = first
        hessian[last, :k1] = first
        hessian[k1:last, last] = second
        hessian[last, k1:last] = second
        hessian[last, last] = (w * hrr).sum() * complement ** 2 \
            - 2 * rho * complement * (w * self.sign * density).sum()
        return value, gradient, hessian

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores [n, k_1 + k_2 + 1], before weights."""
        pieces = self._ratios(theta)
        if pieces is None:
            raise KernelError("numerical_failure", "The bivariate probit likelihood is not "
                              "finite at the estimates.")
        _, _, _, _, g1, g2, density, _, _, _, _, complement, _ = pieces
        return torch.cat([self.x1 * (self.q1 * g1)[:, None], self.x2 * (self.q2 * g2)[:, None],
                          (self.sign * density * complement)[:, None]], dim=1)
