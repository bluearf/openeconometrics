"""Reference distributions for test statistics on float64 scalars; no SciPy at runtime.

Everything reduces to two kernels that return BOTH tails, each computed directly when it
is the small one, so neither ``sf`` nor ``cdf`` is ever formed as ``1 - (number near 1)``:

Incomplete gamma ``P(a, x), Q(a, x)`` (chi-square):
  * prefactor ``x^a e^-x / Gamma(a) = sqrt(a / 2pi) exp(a (ln(1+u) - u) - D(a))`` with
    ``u = (x - a) / a`` and ``D`` the Stirling remainder of ``ln Gamma``; ``ln(1+u) - u`` is
    summed without cancellation, so the prefactor keeps full relative accuracy for any a;
  * ``x < a + 1``: power series for P; otherwise modified-Lentz continued fraction for Q;
  * ``a < 1`` and small x: a series for Q built on ``Gamma(1+a) - 1`` (no ``1 - P``);
  * ``a >= 1000``: the series/fraction would need O(sqrt(a)) terms, so the smaller tail of
    the density is integrated by composite 16-point Gauss-Legendre panels whose edges
    follow the local quadratic model of the log density (about 6 nats lost per panel).
    The integrand is analytic and nearly Gaussian there, which gives ~1e-13 relative
    accuracy at a cost that does not grow with a (this replaces a Temme expansion and
    its coefficient tables).

Incomplete beta ``I_x(a, b)`` (Student t, F), always called with x and y = 1 - x:
  * prefactor ``x^a y^b / B(a, b)`` in the same cancellation-free Stirling form;
  * ``max(a, b) < 1000``: modified-Lentz continued fraction on the side below the mean;
  * one large and one small parameter (large-df t, F with a large denominator df): with
    ``t = e^-u`` the integral becomes ``int e^(-g u) u^(b-1) (sinh(u/2) / (u/2))^(b-1) du``,
    ``g = a + (b-1)/2``; expanding the last factor in powers of u^2 gives a rapidly
    convergent sum of incomplete gamma functions that tends to chi-square as a -> inf;
  * both parameters large: the same Gauss-Legendre tail quadrature as for the gamma.

Quantiles solve ``ln tail(x) = ln target`` on the smaller tail by Newton in ``ln x`` (the
tails of chi-square, t and F are close to linear on that scale) inside a monotone
bracket with geometric bisection as the safeguard. The normal quantile refines Acklam's
rational approximation with Halley steps on the erfc form.

Gauss-Hermite and Gauss-Legendre rules use Golub-Welsch eigenvalues of the symmetric
Jacobi matrix, polished by Newton on the orthonormal three-term recurrence (rescaled so
it cannot overflow); weights come from the recurrence, not from eigenvectors.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from functools import lru_cache

import torch

from .contracts import KernelError

_EPS = 2.220446049250313e-16
_TINY = 1e-300
_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_EULER = 0.5772156649015329
# zeta(k) - 1 for k = 2..21: Taylor coefficients of ln Gamma(1 + a) about a = 0.
_ZETA_M1 = (
    0.6449340668482264, 0.2020569031595943, 0.08232323371113819, 0.03692775514336993,
    0.01734306198444914, 0.008349277381922827, 0.004077356197944339, 0.002008392826082214,
    0.0009945751278180853, 0.0004941886041194646, 0.0002460865533080483,
    0.0001227133475784891, 6.124813505870483e-05, 3.058823630702049e-05,
    1.528225940865187e-05, 7.637197637899762e-06, 3.817293264999840e-06,
    1.908212716553939e-06, 9.539620338727961e-07, 4.769329867878065e-07,
)
_GAMMA_LARGE = 1000.0       # a above which P/Q use the tail quadrature
_BETA_LARGE = 1000.0        # max(a, b) above which the continued fraction is not used
_BETA_BOTH_LARGE = 60.0     # min(a, b) above which (with a large max) quadrature is used
_PANELS = 8
_PANEL_DROP = 6.0
_PANEL_NODES = 16
_MAX_TERMS = 100000


def _invalid(message: str) -> KernelError:
    return KernelError("invalid_inference", message)


def _scalar(value: object, name: str) -> float:
    if isinstance(value, (str, bytes, bytearray)):      # float("1") would silently parse
        raise _invalid(f"{name} must be a real scalar, not text.")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, RuntimeError) as error:
        raise _invalid(f"{name} must be a real scalar.") from error
    if math.isnan(number):
        raise _invalid(f"{name} must not be NaN.")
    return number


def _positive(value: object, name: str, *, allow_inf: bool = False) -> float:
    number = _scalar(value, name)
    if number <= 0 or (math.isinf(number) and not allow_inf):
        raise _invalid(f"{name} must be positive{'' if allow_inf else ' and finite'}.")
    return number


def _probability(value: object) -> float:
    number = _scalar(value, "The probability")
    if not 0.0 < number < 1.0:
        raise _invalid("The probability must lie strictly between zero and one.")
    return number


# --------------------------------------------------------------------------------------
# Elementary pieces
# --------------------------------------------------------------------------------------

def _log1pmx(u: float, one_plus_u: float | None = None) -> float:
    """ln(1 + u) - u.  Near zero it uses r = u / (2 + u), ln(1 + u) = 2 atanh(r):

        ln(1 + u) - u = r (2 r^2 (1/3 + r^2/5 + r^4/7 + ...) - u),

    which has no cancellation. ``one_plus_u`` lets a caller pass 1 + u when it knows that
    ratio more accurately than the sum (u close to -1).
    """
    if -0.5 <= u <= 1.0:
        r = u / (2.0 + u)
        r2 = r * r
        total, power, k = 1.0 / 3.0, r2, 5
        while power > 1e-17:
            total += power / k
            power *= r2
            k += 2
        return r * (2.0 * r2 * total - u)
    return math.log(1.0 + u if one_plus_u is None else one_plus_u) - u


def _log1pmx_tensor(u: torch.Tensor) -> torch.Tensor:
    """Vectorized ln(1 + u) - u; the atanh series is used for |u| <= 0.4."""
    r = u / (2.0 + u)
    r2 = r * r
    total = torch.full_like(u, 1.0 / 33.0)
    for k in range(31, 1, -2):
        total = total * r2 + 1.0 / k
    series = r * (2.0 * r2 * total - u)
    return torch.where(u.abs() <= 0.4, series, torch.log1p(u) - u)


def _stirling_delta(z: float) -> float:
    """D(z) = ln Gamma(z) - [(z - 1/2) ln z - z + ln sqrt(2 pi)]  (Stirling remainder)."""
    if z >= 10.0:
        w = 1.0 / (z * z)
        return (1 / 12 - w * (1 / 360 - w * (1 / 1260 - w * (1 / 1680 - w * (
            1 / 1188 - w * (691 / 360360 - w * (1 / 156 - w * 3617 / 122400))))))) / z
    return math.lgamma(z) - ((z - 0.5) * math.log(z) - z + _LOG_SQRT_2PI)


def _lgamma1p(a: float) -> float:
    """ln Gamma(1 + a) with full relative accuracy as a -> 0.

    For a < 1/4:  -ln(1 + a) + a (1 - gamma) + sum_{k>=2} (-1)^k (zeta(k) - 1) a^k / k.
    """
    if a >= 0.25:
        return math.lgamma(1.0 + a)
    total, power = 0.0, -a
    for k, zeta_m1 in enumerate(_ZETA_M1, start=2):
        power *= -a
        total += zeta_m1 * power / k
    return a * (1.0 - _EULER) - math.log1p(a) + total


def _log_ratio(value: float, target: float) -> float:
    """ln(value / target), exact to first order when the two nearly coincide."""
    if 0.5 * target <= value <= 2.0 * target:
        return math.log1p((value - target) / target)
    return math.log(value) - math.log(target)


@torch.no_grad()
def _tail_quadrature(slope_curvature: Callable[[float], tuple[float, float]],
                     log_ratio: Callable[[torch.Tensor], torch.Tensor], limit: float) -> float:
    """integral_0^limit exp(log_ratio(v)) dv for a log-concave density ratio (1 at v = 0).

    Panel edges solve the local model  g1 d + g2 d^2 / 2 = -_PANEL_DROP  of the exponent, so
    every panel loses about the same number of nats whether the tail is Gaussian (near the
    mode) or exponential (far out); 8 panels reach e^-48 of the starting density.
    """
    edges = [0.0]
    for _ in range(_PANELS):
        g1, g2 = slope_curvature(edges[-1])
        root = math.sqrt(g1 * g1 + 2.0 * _PANEL_DROP * abs(g2))
        step = (g1 + root) / abs(g2) if g1 > 0 else 2.0 * _PANEL_DROP / (root - g1)
        if not edges[-1] + step < limit:
            edges.append(limit)
            break
        edges.append(edges[-1] + step)
    nodes, weights = _gauss_legendre(_PANEL_NODES)
    left = torch.tensor(edges[:-1], dtype=torch.float64)
    half = 0.5 * (torch.tensor(edges[1:], dtype=torch.float64) - left)
    points = left[:, None] + half[:, None] * (nodes + 1.0)
    return float((torch.exp(log_ratio(points)) @ weights) @ half)


# --------------------------------------------------------------------------------------
# Incomplete gamma
# --------------------------------------------------------------------------------------

def _gamma_front(a: float, x: float) -> float:
    """x^a e^-x / Gamma(a) = sqrt(a / 2 pi) exp(a [ln(x/a) - (x - a)/a] - D(a))."""
    if 0.5 * a <= x <= 2.0 * a:
        exponent = a * _log1pmx((x - a) / a)
    else:
        ratio = x / a
        log_ratio = (math.log(ratio) if 0.0 < ratio < math.inf
                     else math.log(x) - math.log(a))
        exponent = a * log_ratio - (x - a)
    return math.sqrt(a / (2.0 * math.pi)) * math.exp(exponent - _stirling_delta(a))


def _gamma_series(a: float, x: float) -> float:
    """sum_{n>=0} x^n / ((a+1)...(a+n)), so that P(a, x) = front * sum / a."""
    term = total = 1.0
    denominator = a
    for _ in range(_MAX_TERMS):
        denominator += 1.0
        term *= x / denominator
        total += term
        if term <= 0.25 * _EPS * total:
            return total
    raise KernelError("inference_nonconvergence", "The incomplete gamma series did not converge.")


def _gamma_fraction(a: float, x: float) -> float:
    """Legendre continued fraction 1/(x+1-a- 1(1-a)/(x+3-a- 2(2-a)/(x+5-a- ...))) by
    modified Lentz, so that Q(a, x) = front * fraction."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    result = d
    for i in range(1, _MAX_TERMS):
        numerator = -i * (i - a)
        b += 2.0
        d = numerator * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= _EPS:
            return result
    raise KernelError("inference_nonconvergence",
                      "The incomplete gamma continued fraction did not converge.")


def _gamma_q_small(a: float, x: float) -> tuple[float, float]:
    """(Q, front) for a < 1 and small x without forming 1 - P.

    Q = [(Gamma(1+a) - 1) - (x^a - 1)] / Gamma(1+a) - x^a / Gamma(a) * S,
    S = sum_{n>=1} (-x)^n / (n! (a + n));  both brackets are evaluated with expm1.
    """
    log_gamma = _lgamma1p(a)
    log_power = a * math.log(x)
    term, total = 1.0, 0.0
    for n in range(1, 200):
        term *= -x / n
        total += term / (a + n)
        if abs(term) <= _EPS * abs(total) * (a + n):
            break
    scaled_power = a * math.exp(log_power - log_gamma)           # x^a / Gamma(a)
    head = (math.expm1(log_gamma) - math.expm1(log_power)) * math.exp(-log_gamma)
    return head - scaled_power * total, scaled_power * math.exp(-x)


def _gamma_tail_quadrature(a: float, x: float, front: float, upper: bool) -> float:
    """Mass of the Gamma(a) density t^(a-1) e^-t / Gamma(a) beyond x on the chosen side."""
    sign = 1.0 if upper else -1.0
    am1 = a - 1.0
    slope = sign * (am1 - x) / x

    def slope_curvature(v: float) -> tuple[float, float]:
        t = x + sign * v
        return sign * (am1 / t - 1.0), -am1 / (t * t)

    def log_ratio(v: torch.Tensor) -> torch.Tensor:
        # (a-1) ln(t/x) - (t - x) with the linear term of the logarithm taken out exactly.
        return am1 * _log1pmx_tensor(sign * v / x) + slope * v

    return front / x * _tail_quadrature(slope_curvature, log_ratio, math.inf if upper else x)


def _gamma_core(a: float, x: float) -> tuple[float, float, float]:
    """(P(a, x), Q(a, x), x^a e^-x / Gamma(a)) for a > 0 and x >= 0."""
    if x <= 0.0:
        return 0.0, 1.0, 0.0
    if math.isinf(x):
        return 1.0, 0.0, 0.0
    if a < 1.0 and x < 1.1 and a <= (0.75 * x if x >= 0.5 else -0.4 / math.log(x)):
        q, front = _gamma_q_small(a, x)
        q = min(1.0, max(0.0, q))
        return 1.0 - q, q, front
    front = _gamma_front(a, x)
    if front == 0.0:
        return (0.0, 1.0, 0.0) if x < a else (1.0, 0.0, 0.0)
    if a >= _GAMMA_LARGE:
        upper = x > a
        tail = min(1.0, _gamma_tail_quadrature(a, x, front, upper))
        return (1.0 - tail, tail, front) if upper else (tail, 1.0 - tail, front)
    if x < a + 1.0:
        p = min(1.0, front * _gamma_series(a, x) / a)
        return p, 1.0 - p, front
    q = min(1.0, front * _gamma_fraction(a, x))
    return 1.0 - q, q, front


def gamma_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma  P(a, x) = (1 / Gamma(a)) int_0^x t^(a-1) e^-t dt."""
    a, x = _positive(a, "The shape"), _scalar(x, "The argument")
    if x < 0:
        raise _invalid("The incomplete gamma argument must be non-negative.")
    return _gamma_core(a, x)[0]


def gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma  Q(a, x) = (1 / Gamma(a)) int_x^inf t^(a-1) e^-t dt.

    Evaluated directly (continued fraction, small-a series or tail quadrature) whenever it
    is the small tail, so it stays accurate down to the underflow threshold.
    """
    a, x = _positive(a, "The shape"), _scalar(x, "The argument")
    if x < 0:
        raise _invalid("The incomplete gamma argument must be non-negative.")
    return _gamma_core(a, x)[1]


# --------------------------------------------------------------------------------------
# Incomplete beta
# --------------------------------------------------------------------------------------

def _beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction of DLMF 8.17.22 by modified Lentz: I_x(a,b) = front * cf / a."""
    total, above, below = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - total * x / above
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    result = d
    for m in range(1, _MAX_TERMS):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((below + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        result *= d * c
        numerator = -(a + m) * (total + m) * x / ((a + m2) * (above + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= _EPS:
            return result
    raise KernelError("inference_nonconvergence",
                      "The incomplete beta continued fraction did not converge.")


def _beta_large_first(a: float, b: float, x: float, y: float) -> float:
    """I_x(a, b) for a large and b << a (x + y = 1), accurate in its lower tail.

    With t = e^-u, g = a + (b - 1)/2, xi = -ln x and (sinh(u/2)/(u/2))^(b-1) = sum c_k u^2k,

        I_x(a, b) = Gamma(a+b) / (Gamma(a) g^b) * sum_k c_k Gamma(b + 2k, g xi) / (Gamma(b) g^2k).

    c_k follows from the power-of-a-series recurrence; Gamma(b+m, z) / Gamma(b) from the
    stable upward recurrence T_{m+1} = (b + m) T_m + z^(b+m) e^-z / Gamma(b).
    """
    nu = b - 1.0
    g = a + 0.5 * nu
    xi = -math.log1p(-y) if y < 0.5 else -math.log(x)
    _, tail, edge = _gamma_core(b, g * xi)
    if tail == 0.0 and edge == 0.0:
        return 0.0
    log_scale = (b * math.log1p((b + 1.0) / (2.0 * g)) + (a - 0.5) * _log1pmx(b / a)
                 - b / (2.0 * a) + _stirling_delta(a + b) - _stirling_delta(a))
    sigma = [1.0]                      # coefficients of sinh(u/2)/(u/2) in powers of u^2
    coefficients = [1.0]
    total = tail
    edge /= g
    power = 1.0                        # xi^m
    m = 0
    for k in range(1, 400):
        sigma.append(sigma[-1] / (4.0 * (2 * k) * (2 * k + 1)))
        for _ in range(2):
            tail = (b + m) / g * tail + edge * power
            power *= xi
            m += 1
        c = sum((j * (nu + 1.0) - k) * sigma[j] * coefficients[k - j]
                for j in range(1, k + 1)) / k
        coefficients.append(c)
        term = c * tail
        total += term
        if abs(term) <= 0.25 * _EPS * abs(total):
            return math.exp(log_scale) * total
    raise KernelError("inference_nonconvergence",
                      "The large-parameter incomplete beta expansion did not converge.")


def _beta_tail_quadrature(a: float, b: float, x: float, y: float, delta: float,
                          front: float, upper: bool) -> float:
    """Mass of the Beta(a, b) density beyond x on the chosen side (delta = x - a/(a+b))."""
    sign = 1.0 if upper else -1.0
    am1, bm1 = a - 1.0, b - 1.0
    # (a-1)/x - (b-1)/y written through delta so it survives x close to the mean.
    slope = -sign * ((a + b) * delta + (y - x)) / (x * y)

    def slope_curvature(v: float) -> tuple[float, float]:
        t, s = x + sign * v, y - sign * v
        d = delta + sign * v
        return -sign * ((a + b) * d + (s - t)) / (t * s), -(am1 / (t * t) + bm1 / (s * s))

    def log_ratio(v: torch.Tensor) -> torch.Tensor:
        return (am1 * _log1pmx_tensor(sign * v / x) + bm1 * _log1pmx_tensor(-sign * v / y)
                + slope * v)

    return front / (x * y) * _tail_quadrature(slope_curvature, log_ratio, y if upper else x)


def _beta_core(a: float, b: float, x: float, y: float) -> tuple[float, float, float]:
    """(I_x(a, b), I_y(b, a), x^a y^b / B(a, b)) for x + y = 1, both supplied accurately.

    The prefactor is  sqrt(a b / (2 pi (a+b))) exp(a f(u) + b f(w) + D(a+b) - D(a) - D(b)),
    f(u) = ln(1+u) - u, u = (x - x0)/x0, w = (y - y0)/y0, x0 = a/(a+b): both f terms are
    non-positive, so no cancellation occurs however large a and b are.
    """
    if x <= 0.0:
        return 0.0, 1.0, 0.0
    if y <= 0.0:
        return 1.0, 0.0, 0.0
    total = a + b
    x0, y0 = a / total, b / total
    delta = x - x0 if x <= y else y0 - y
    exponent = (a * _log1pmx(delta / x0, x / x0) + b * _log1pmx(-delta / y0, y / y0)
                + _stirling_delta(total) - _stirling_delta(a) - _stirling_delta(b))
    front = math.sqrt(a / (2.0 * math.pi) * (b / total)) * math.exp(exponent)
    # The continued fraction converges below (a+1)/(a+b+2) only; decide the side from the
    # accurately known argument (x rounded to 1 would otherwise put a tiny y on the wrong
    # side of its own threshold (b+1)/(a+b+2)).
    lower_side = (x < (a + 1.0) / (total + 2.0) if x <= y
                  else y > (b + 1.0) / (total + 2.0))
    if max(a, b) < _BETA_LARGE:
        if lower_side:
            p = min(1.0, front * _beta_fraction(a, b, x) / a)
            return p, 1.0 - p, front
        q = min(1.0, front * _beta_fraction(b, a, y) / b)
        return 1.0 - q, q, front
    if min(a, b) >= _BETA_BOTH_LARGE:
        if front == 0.0:
            return (0.0, 1.0, 0.0) if delta < 0 else (1.0, 0.0, 0.0)
        upper = delta > 0
        tail = min(1.0, _beta_tail_quadrature(a, b, x, y, delta, front, upper))
        return (1.0 - tail, tail, front) if upper else (tail, 1.0 - tail, front)
    # One large, one small parameter: the tail on the small-parameter side converges fast
    # as a continued fraction; the other tail uses the incomplete-gamma expansion.
    if (a < b) == lower_side:
        if lower_side:
            p = min(1.0, front * _beta_fraction(a, b, x) / a)
            return p, 1.0 - p, front
        q = min(1.0, front * _beta_fraction(b, a, y) / b)
        return 1.0 - q, q, front
    if a >= b:
        p = min(1.0, _beta_large_first(a, b, x, y))
        return p, 1.0 - p, front
    q = min(1.0, _beta_large_first(b, a, y, x))
    return 1.0 - q, q, front


def _log_beta(a: float, b: float) -> float:
    """ln B(a, b) = a ln x0 + b ln y0 - ln sqrt(a b / (2 pi (a+b))) + D(a) + D(b) - D(a+b),
    x0 = a/(a+b) = 1 - y0: the Stirling form has no large cancelling log-gamma values."""
    total = a + b
    x0, y0 = a / total, b / total
    log_x0 = math.log(x0) if x0 <= 0.5 else math.log1p(-y0)
    log_y0 = math.log(y0) if y0 < 0.5 else math.log1p(-x0)
    return (a * log_x0 + b * log_y0 - 0.5 * math.log(a / (2.0 * math.pi) * y0)
            + _stirling_delta(a) + _stirling_delta(b) - _stirling_delta(total))


def _beta_corner(a: float, b: float, log_x: float) -> tuple[float, float, float]:
    """(I_x, 1 - I_x, front) for x below ~1e-290 (x itself may have underflowed), from ln x.

    For t <= x the factor (1 - t)^(b-1) equals e^(-(b-1) t) to double precision, so

        I_x(a, b) = exp(s) P(a, z),   z = (b-1) x,   s = ln[Gamma(a+b) / (Gamma(b) (b-1)^a)],

    and the same approximation gives the upper tail as exp(s) Q(a, z): each tail is then
    relatively exact, whereas 1 - exp(s) P would carry the O(b x^2) absolute error of the
    exponential approximation into a far upper tail. s is O(a^2 / b) and is formed from
    the same cancellation-free Stirling pieces as the other prefactors. When z < 1e-17
    (every case with b below ~1e273) this reduces to x^a / (a B(a, b)). For tiny a the
    result is not small (x^a -> 1), which is why the complement is still formed
    explicitly.
    """
    log_beta = _log_beta(a, b)
    log_z = log_x + math.log(b - 1.0) if b > 1.0 else -math.inf
    if log_z < -40.0:
        front = math.exp(a * log_x - log_beta)
        p = min(1.0, front / a)
        return p, 1.0 - p, front
    z = math.exp(log_z)
    lower, upper, _ = _gamma_core(a, z)
    scale = math.exp(a * math.log1p((a + 1.0) / (b - 1.0)) + (b - 0.5) * _log1pmx(a / b)
                     - a / (2.0 * b) + _stirling_delta(a + b) - _stirling_delta(b))
    return min(1.0, scale * lower), min(1.0, scale * upper), math.exp(a * log_x - log_beta - z)


_CORNER = 1e-290


def beta_inc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta  I_x(a, b) = (1 / B(a, b)) int_0^x t^(a-1) (1-t)^(b-1) dt."""
    a, b = _positive(a, "The first shape"), _positive(b, "The second shape")
    x = _scalar(x, "The argument")
    if not 0.0 <= x <= 1.0:
        raise _invalid("The incomplete beta argument must lie in [0, 1].")
    return _beta_core(a, b, x, 1.0 - x)[0]


# --------------------------------------------------------------------------------------
# Monotone quantile solver
# --------------------------------------------------------------------------------------

def _solve_log_newton(residual: Callable[[float], tuple[float, float]], start: float,
                      increasing: bool) -> float:
    """Positive root of a monotone residual by bracketed Newton in ln x.

    ``residual(x)`` returns ``(r, dr / d ln x)`` with ``r = ln(tail(x) / target)``. A Newton
    step that leaves the bracket (or an underflowed tail) is replaced by geometric
    bisection, or by a squared-factor expansion while one side is still unbounded.
    """
    x = start
    low, high = 0.0, math.inf
    growth = 2.0
    previous = math.inf
    for _ in range(200):
        r, slope = residual(x)
        if r == 0.0:
            return x
        if (r > 0.0) == increasing:
            high = x
        else:
            low = x
        candidate = math.nan
        if math.isfinite(r) and math.isfinite(slope) and slope != 0.0:
            step = -r / slope
            if abs(step) < 700.0:
                candidate = x * math.exp(step)
                # Converged, or the step has reached the rounding floor of the residual.
                if abs(step) <= 2.0 * _EPS or (previous < 1e-7 and abs(step) >= previous):
                    return candidate if low < candidate < high else x
                previous = abs(step)
        if not low < candidate < high:
            previous = math.inf
            if math.isinf(high):
                candidate = x * growth
                growth *= growth
            elif low == 0.0:
                candidate = x / growth
                growth *= growth
            else:
                candidate = math.sqrt(low) * math.sqrt(high)
                if not low < candidate < high:
                    return candidate
            if candidate == 0.0 or math.isinf(candidate):
                return candidate
        x = candidate
    return x


# --------------------------------------------------------------------------------------
# Normal
# --------------------------------------------------------------------------------------

def normal_pdf(x: float) -> float:
    """Standard normal density  phi(x) = exp(-x^2 / 2) / sqrt(2 pi)."""
    x = _scalar(x, "The argument")
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def normal_cdf(x: float) -> float:
    """Standard normal distribution function  Phi(x) = erfc(-x / sqrt 2) / 2."""
    return 0.5 * math.erfc(-_scalar(x, "The argument") / _SQRT2)


def normal_sf(x: float) -> float:
    """Upper tail  1 - Phi(x) = erfc(x / sqrt 2) / 2, accurate until it underflows."""
    return 0.5 * math.erfc(_scalar(x, "The argument") / _SQRT2)


_ACKLAM_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACKLAM_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
_ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)


def _normal_lower_quantile(p: float) -> float:
    """x <= 0 with Phi(x) = p for 0 < p <= 1/2.

    Acklam's rational approximation (relative error ~1e-9) followed by Halley steps on
    e = Phi(x) - p:  x <- x - h / (1 + x h / 2),  h = e / phi(x); the ratio p / phi(x) is
    formed in logs so the far tail (p down to the subnormals) neither over- nor underflows.
    """
    if p < 0.02425:
        s = math.sqrt(-2.0 * math.log(p))
        c, d = _ACKLAM_C, _ACKLAM_D
        x = ((((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5])
             / ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1.0))
    else:
        s = p - 0.5
        r = s * s
        a, b = _ACKLAM_A, _ACKLAM_B
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * s
             / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    log_p = math.log(p)
    for _ in range(4):
        if p > 0.25:
            # Phi(x) - p = erf(x / sqrt 2) / 2 + (1/2 - p): both pieces are exact to
            # rounding, where erfc(-x / sqrt 2) / 2 - p would cancel to nothing near 1/2.
            h = (0.5 * math.erf(x / _SQRT2) + (0.5 - p)) * _SQRT_2PI * math.exp(0.5 * x * x)
        else:
            cdf = 0.5 * math.erfc(-x / _SQRT2)
            if cdf == 0.0:
                break
            h = (cdf - p) / p * math.exp(log_p + 0.5 * x * x + _LOG_SQRT_2PI)
        step = h / (1.0 + 0.5 * x * h)
        x -= step
        if abs(step) <= 2.0 * _EPS * max(1.0, abs(x)):
            break
    return min(x, 0.0)


def normal_ppf(p: float) -> float:
    """Standard normal quantile  Phi^-1(p),  0 < p < 1  (Stata ``invnormal``)."""
    p = _probability(p)
    return _normal_lower_quantile(p) if p <= 0.5 else -_normal_lower_quantile(1.0 - p)


def normal_isf(q: float) -> float:
    """Upper-tail normal quantile: x with 1 - Phi(x) = q; equals -normal_ppf(q) exactly."""
    return -normal_ppf(q)


# --------------------------------------------------------------------------------------
# Chi-square
# --------------------------------------------------------------------------------------

def chi2_cdf(x: float, df: float) -> float:
    """Chi-square distribution function  P(df/2, x/2)  (Stata ``chi2(df, x)``)."""
    x, df = _scalar(x, "The statistic"), _positive(df, "Degrees of freedom")
    return _gamma_core(0.5 * df, 0.5 * x)[0]


def chi2_sf(x: float, df: float) -> float:
    """Chi-square upper tail  Q(df/2, x/2)  (Stata ``chi2tail(df, x)``)."""
    x, df = _scalar(x, "The statistic"), _positive(df, "Degrees of freedom")
    return _gamma_core(0.5 * df, 0.5 * x)[1]


def _gamma_quantile(a: float, probability: float, lower: bool) -> float:
    """x with P(a, x) = probability (lower) or Q(a, x) = probability (upper)."""
    p, q = (probability, 1.0 - probability) if lower else (1.0 - probability, probability)
    use_lower = p <= q
    target = p if use_lower else q
    log_lower = (math.log(p) + math.lgamma(a + 1.0)) / a if p > 0.0 else -math.inf
    start = None
    if use_lower and log_lower < math.log(0.2 * (a + 1.0)):
        # P(a, x) ~ x^a / Gamma(a + 1) as x -> 0.
        if log_lower < -745.0:
            return 0.0
        start = math.exp(log_lower)
    elif a >= 0.5:
        # Wilson-Hilferty: (X / a)^(1/3) is nearly normal(1 - 1/(9a), 1/(9a)).
        z = _normal_lower_quantile(p) if use_lower else -_normal_lower_quantile(q)
        cube = 1.0 - 1.0 / (9.0 * a) + z / (3.0 * math.sqrt(a))
        if cube > 0.0:
            start = a * cube ** 3
    if start is None:
        # Q(a, x) ~ x^(a-1) e^-x / Gamma(a) as x -> inf.
        far = -(math.log(q) + math.lgamma(a))
        start = far + (a - 1.0) * math.log(far) if far > 2.0 else math.exp(max(log_lower, -700.0))
        start = max(start, 1e-300)

    def residual(x: float) -> tuple[float, float]:
        lower_tail, upper_tail, front = _gamma_core(a, x)
        tail = lower_tail if use_lower else upper_tail
        if tail <= 0.0:
            return -math.inf, 0.0
        return _log_ratio(tail, target), (front if use_lower else -front) / tail

    return _solve_log_newton(residual, start, use_lower)


def chi2_ppf(p: float, df: float) -> float:
    """Chi-square quantile: x with P(df/2, x/2) = p  (Stata ``invchi2(df, p)``)."""
    p, df = _probability(p), _positive(df, "Degrees of freedom")
    return 2.0 * _gamma_quantile(0.5 * df, p, True)


def chi2_isf(q: float, df: float) -> float:
    """Chi-square upper-tail quantile: x with Q(df/2, x/2) = q  (Stata ``invchi2tail``)."""
    q, df = _probability(q), _positive(df, "Degrees of freedom")
    return 2.0 * _gamma_quantile(0.5 * df, q, False)


# --------------------------------------------------------------------------------------
# F
# --------------------------------------------------------------------------------------

def _f_core(x: float, df1: float, df2: float) -> tuple[float, float, float]:
    """(cdf, sf, d cdf / d ln x) of F(df1, df2) at 0 < x < inf, both df finite.

    cdf = I_w(df1/2, df2/2) with w = df1 x / (df1 x + df2); w and 1 - w are both formed
    from the odds df1 x / df2, so neither loses digits near 0 or 1.
    """
    odds = x * (df1 / df2)
    if odds <= 1.0:
        if odds < _CORNER:                    # ln(odds) from the logs: the ratio may underflow
            return _beta_corner(0.5 * df1, 0.5 * df2,
                                math.log(x) + math.log(df1) - math.log(df2))
        w, v = odds / (1.0 + odds), 1.0 / (1.0 + odds)
    else:
        inverse = (df2 / df1) / x
        if inverse < _CORNER:
            upper, lower, front = _beta_corner(0.5 * df2, 0.5 * df1,
                                               math.log(df2) - math.log(df1) - math.log(x))
            return lower, upper, front
        w, v = 1.0 / (1.0 + inverse), inverse / (1.0 + inverse)
    return _beta_core(0.5 * df1, 0.5 * df2, w, v)


def _f_tails(x: float, df1: float, df2: float) -> tuple[float, float]:
    if math.isinf(df1) and math.isinf(df2):
        raise _invalid("At most one F degrees-of-freedom argument may be infinite.")
    if x <= 0.0:
        return 0.0, 1.0
    if math.isinf(x):
        return 1.0, 0.0
    if math.isinf(df2):                       # df1 F -> chi2(df1)
        return _gamma_core(0.5 * df1, 0.5 * df1 * x)[:2]
    if math.isinf(df1):                       # df2 / F -> chi2(df2)
        upper, lower, _ = _gamma_core(0.5 * df2, 0.5 * df2 / x)
        return lower, upper
    return _f_core(x, df1, df2)[:2]


def f_cdf(x: float, df1: float, df2: float) -> float:
    """F distribution function  I_w(df1/2, df2/2),  w = df1 x / (df1 x + df2)  (Stata ``F``)."""
    x = _scalar(x, "The statistic")
    df1 = _positive(df1, "Numerator degrees of freedom", allow_inf=True)
    df2 = _positive(df2, "Denominator degrees of freedom", allow_inf=True)
    return _f_tails(x, df1, df2)[0]


def f_sf(x: float, df1: float, df2: float) -> float:
    """F upper tail  I_v(df2/2, df1/2),  v = df2 / (df2 + df1 x)  (Stata ``Ftail``).

    Uses the complementary incomplete-beta argument, never 1 - cdf.
    """
    x = _scalar(x, "The statistic")
    df1 = _positive(df1, "Numerator degrees of freedom", allow_inf=True)
    df2 = _positive(df2, "Denominator degrees of freedom", allow_inf=True)
    return _f_tails(x, df1, df2)[1]


def _f_quantile(probability: float, df1: float, df2: float, lower: bool) -> float:
    if math.isinf(df1) and math.isinf(df2):
        raise _invalid("At most one F degrees-of-freedom argument may be infinite.")
    if math.isinf(df2):
        return 2.0 * _gamma_quantile(0.5 * df1, probability, lower) / df1
    if math.isinf(df1):
        reciprocal = 2.0 * _gamma_quantile(0.5 * df2, probability, not lower)
        return df2 / reciprocal if reciprocal > 0.0 else math.inf
    p, q = (probability, 1.0 - probability) if lower else (1.0 - probability, probability)
    use_lower = p <= q
    target = p if use_lower else q
    a, b = 0.5 * df1, 0.5 * df2
    if a >= 1.0 and b >= 1.0:
        # Abramowitz-Stegun 26.5.22 (Paulson): F ~ exp(-2 w) with z the upper deviate.
        z = -_normal_lower_quantile(p) if use_lower else _normal_lower_quantile(q)
        lam = (z * z - 3.0) / 6.0
        ia, ib = 1.0 / (2.0 * a - 1.0), 1.0 / (2.0 * b - 1.0)
        h = 2.0 / (ia + ib)
        w = z * math.sqrt(h + lam) / h - (ib - ia) * (lam + 5.0 / 6.0 - 2.0 / (3.0 * h))
        start = math.exp(max(-340.0, min(340.0, -2.0 * w)))
    else:
        # Power-law tails: I_w(a, b) ~ w^a / (a B) and 1 - I_w ~ v^b / (b B).
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        scale = math.log(df2 / df1)
        if use_lower:
            log_w = min(-1e-3, (math.log(p * a) + log_beta) / a)
            log_start = log_w - math.log(-math.expm1(log_w)) + scale
        else:
            log_v = min(-1e-3, (math.log(q * b) + log_beta) / b)
            log_start = math.log(-math.expm1(log_v)) - log_v + scale
        if abs(log_start) > 745.0:          # the quantile itself leaves the float64 range
            return 0.0 if log_start < 0 else math.inf
        start = math.exp(min(log_start, 709.0))

    def residual(x: float) -> tuple[float, float]:
        lower_tail, upper_tail, front = _f_core(x, df1, df2)
        tail = lower_tail if use_lower else upper_tail
        if tail <= 0.0:
            return -math.inf, 0.0
        return _log_ratio(tail, target), (front if use_lower else -front) / tail

    return _solve_log_newton(residual, start, use_lower)


def f_ppf(p: float, df1: float, df2: float) -> float:
    """F quantile: x with f_cdf(x, df1, df2) = p  (Stata ``invF(df1, df2, p)``)."""
    p = _probability(p)
    df1 = _positive(df1, "Numerator degrees of freedom", allow_inf=True)
    df2 = _positive(df2, "Denominator degrees of freedom", allow_inf=True)
    return _f_quantile(p, df1, df2, True)


def f_isf(q: float, df1: float, df2: float) -> float:
    """F upper-tail quantile: x with f_sf(x, df1, df2) = q  (Stata ``invFtail``)."""
    q = _probability(q)
    df1 = _positive(df1, "Numerator degrees of freedom", allow_inf=True)
    df2 = _positive(df2, "Denominator degrees of freedom", allow_inf=True)
    return _f_quantile(q, df1, df2, False)


# --------------------------------------------------------------------------------------
# Student t
# --------------------------------------------------------------------------------------

def _t_core(t: float, df: float) -> tuple[float, float, float]:
    """(P(|T| <= t), P(|T| > t), front) for finite t > 0 via T^2 ~ F(1, df):

    P(|T| > t) = I_v(df/2, 1/2),  v = df / (df + t^2).
    """
    odds = t * (t / df)
    if odds <= 1.0:
        if odds < _CORNER:
            return _beta_corner(0.5, 0.5 * df, 2.0 * math.log(t) - math.log(df))
        w, v = odds / (1.0 + odds), 1.0 / (1.0 + odds)
    else:
        inverse = (df / t) / t
        if inverse < _CORNER:
            upper, lower, front = _beta_corner(0.5 * df, 0.5, math.log(df) - 2.0 * math.log(t))
            return lower, upper, front
        w, v = 1.0 / (1.0 + inverse), inverse / (1.0 + inverse)
    return _beta_core(0.5, 0.5 * df, w, v)


def _t_tails(x: float, df: float) -> tuple[float, float]:
    if math.isinf(df):
        return 0.5 * math.erfc(-x / _SQRT2), 0.5 * math.erfc(x / _SQRT2)
    if x == 0.0:
        return 0.5, 0.5
    if math.isinf(x):
        return (1.0, 0.0) if x > 0 else (0.0, 1.0)
    inside, outside, _ = _t_core(abs(x), df)
    body, tail = 0.5 + 0.5 * inside, 0.5 * outside
    return (body, tail) if x > 0 else (tail, body)


def t_cdf(x: float, df: float) -> float:
    """Student t distribution function; for x < 0 it is I_v(df/2, 1/2) / 2 with
    v = df / (df + x^2), evaluated directly so the lower tail keeps relative accuracy."""
    x = _scalar(x, "The statistic")
    return _t_tails(x, _positive(df, "Degrees of freedom", allow_inf=True))[0]


def t_sf(x: float, df: float) -> float:
    """Student t upper tail  P(T > x) = I_v(df/2, 1/2) / 2 for x > 0  (Stata ``ttail``)."""
    x = _scalar(x, "The statistic")
    return _t_tails(x, _positive(df, "Degrees of freedom", allow_inf=True))[1]


def _t_upper_quantile(q: float, df: float) -> float:
    """t >= 0 with P(T > t) = q for 0 < q <= 1/2."""
    if math.isinf(df):
        return -_normal_lower_quantile(q)
    if q == 0.5:
        return 0.0
    # Far tail: P(T > t) ~ v^(df/2) / (df B(df/2, 1/2)) with v = df / (df + t^2).
    log_v = 2.0 * (math.log(q * df) + _log_beta(0.5 * df, 0.5)) / df
    if log_v < math.log(0.5):
        if 0.5 * (math.log(df) - log_v) > 709.0:
            return math.inf
        start = math.sqrt(df) * math.exp(-0.5 * log_v)
    else:
        # Cornish-Fisher correction of the normal deviate.
        z = -_normal_lower_quantile(q)
        start = z + (z ** 3 + z) / (4.0 * df) if z * z < df else z * math.sqrt(1.0 + z * z / df)
        start = max(start, 1e-300)

    def residual(t: float) -> tuple[float, float]:
        _, outside, front = _t_core(t, df)
        if outside <= 0.0:
            return -math.inf, 0.0
        return _log_ratio(0.5 * outside, q), -2.0 * front / outside

    return _solve_log_newton(residual, start, False)


def t_ppf(p: float, df: float) -> float:
    """Student t quantile: x with t_cdf(x, df) = p  (Stata ``invt(df, p)``)."""
    p, df = _probability(p), _positive(df, "Degrees of freedom", allow_inf=True)
    return -_t_upper_quantile(p, df) if p <= 0.5 else _t_upper_quantile(1.0 - p, df)


def t_isf(q: float, df: float) -> float:
    """Student t upper-tail quantile: x with t_sf(x, df) = q  (Stata ``invttail(df, q)``)."""
    return -t_ppf(q, df)


# --------------------------------------------------------------------------------------
# Test-level conveniences
# --------------------------------------------------------------------------------------

def _family(distribution: str, df: float | None, df2: float | None) -> str:
    name = distribution.lower() if isinstance(distribution, str) else ""
    if name not in ("normal", "t", "chi2", "f"):
        raise _invalid("The reference distribution must be one of normal, t, chi2 or F.")
    if (name != "normal" and df is None) or (name == "f" and df2 is None):
        raise _invalid(f"The {distribution} distribution needs its degrees of freedom.")
    return name


def p_value(statistic: float, distribution: str, df: float | None = None,
            df2: float | None = None, *, two_sided: bool = True) -> float:
    """p-value of a test statistic against its reference distribution.

    normal / t: 2 * sf(|statistic|) when ``two_sided`` (Stata's z and t tests), otherwise
    the upper tail sf(statistic). chi2 / F: always the upper tail (Wald, LR, Hausman, J).
    """
    name = _family(distribution, df, df2)
    value = _scalar(statistic, "The test statistic")
    if name == "chi2":
        return chi2_sf(value, df)
    if name == "f":
        return f_sf(value, df, df2)
    if two_sided:
        value = abs(value)
    tail = normal_sf(value) if name == "normal" else t_sf(value, df)
    return min(1.0, 2.0 * tail) if two_sided else tail


def critical(alpha: float, distribution: str, df: float | None = None,
             df2: float | None = None, *, two_sided: bool = True) -> float:
    """Critical value c with p_value(c, ...) = alpha.

    normal / t: the upper alpha/2 point when ``two_sided`` (confidence-interval multiplier),
    otherwise the upper alpha point. chi2 / F: the upper alpha point.
    """
    name = _family(distribution, df, df2)
    alpha = _probability(alpha)
    if name == "chi2":
        return chi2_isf(alpha, df)
    if name == "f":
        return f_isf(alpha, df, df2)
    level = 0.5 * alpha if two_sided else alpha
    return normal_isf(level) if name == "normal" else t_isf(level, df)


# --------------------------------------------------------------------------------------
# Gaussian quadrature
# --------------------------------------------------------------------------------------

def _order(n: object) -> int:
    try:
        order = operator.index(n)         # ints, NumPy integers, 0-dim integer tensors
    except TypeError:
        order = 0
    if isinstance(n, bool) or order < 1:
        raise KernelError("invalid_quadrature", "The quadrature order must be a positive integer.")
    return order


def _hermite_pair(x: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h_n, h_{n-1}) e^-s and the log scale s for the orthonormal Hermite polynomials

        h_0 = pi^(-1/4),  h_{j+1} = x sqrt(2/(j+1)) h_j - sqrt(j/(j+1)) h_{j-1}.

    The pair is renormalized every 16 steps, so large n cannot overflow.
    """
    previous = torch.zeros_like(x)
    current = torch.full_like(x, math.pi ** -0.25)
    log_scale = torch.zeros_like(x)
    for j in range(n):
        previous, current = current, (x * math.sqrt(2.0 / (j + 1)) * current
                                      - math.sqrt(j / (j + 1)) * previous)
        if j % 16 == 15:
            scale = torch.maximum(current.abs(), previous.abs())
            current, previous = current / scale, previous / scale
            log_scale += torch.log(scale)
    return current, previous, log_scale


@lru_cache(maxsize=64)
@torch.no_grad()
def _gauss_hermite(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(1, n, dtype=torch.float64)
    off = torch.sqrt(0.5 * k)
    x = torch.linalg.eigvalsh(torch.diag(off, 1) + torch.diag(off, -1))
    x = 0.5 * (x - x.flip(0))
    for _ in range(2):
        h, h_below, _ = _hermite_pair(x, n)
        x = x - h / (math.sqrt(2.0 * n) * h_below)      # h_n' = sqrt(2n) h_{n-1}
        x = 0.5 * (x - x.flip(0))
    _, h_below, log_scale = _hermite_pair(x, n)
    w = torch.exp(-math.log(n) - 2.0 * (torch.log(h_below.abs()) + log_scale))
    w = 0.5 * (w + w.flip(0))
    return x, w * (math.sqrt(math.pi) / w.sum())


@lru_cache(maxsize=64)
@torch.no_grad()
def _gauss_legendre(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.arange(1, n, dtype=torch.float64)
    off = k / torch.sqrt(4.0 * k * k - 1.0)
    x = torch.linalg.eigvalsh(torch.diag(off, 1) + torch.diag(off, -1))
    x = 0.5 * (x - x.flip(0))
    for _ in range(3):
        below, current = torch.ones_like(x), x.clone()
        for j in range(1, n):
            below, current = current, ((2 * j + 1) * x * current - j * below) / (j + 1)
        derivative = n * (x * current - below) / (x * x - 1.0)
        x = x - current / derivative
        x = 0.5 * (x - x.flip(0))
    w = 2.0 / ((1.0 - x * x) * derivative * derivative)
    w = 0.5 * (w + w.flip(0))
    return x, w * (2.0 / w.sum())


def gauss_hermite(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Hermite rule:  int e^(-x^2) f(x) dx ~= sum_i w_i f(x_i), nodes ascending.

    Nodes are the eigenvalues of the Jacobi matrix with off-diagonal sqrt(k/2), polished
    by Newton; weights are w_i = 1 / (n h_{n-1}(x_i)^2) with orthonormal h. For a
    N(mu, s^2) expectation use x = mu + sqrt(2) s x_i and weights w_i / sqrt(pi).
    Results are cached; each call returns fresh float64 CPU copies.
    """
    x, w = _gauss_hermite(_order(n))
    return x.clone(), w.clone()


def gauss_legendre(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre rule on [-1, 1]:  int_-1^1 f(x) dx ~= sum_i w_i f(x_i), nodes ascending.

    Nodes are the eigenvalues of the Jacobi matrix with off-diagonal k / sqrt(4k^2 - 1),
    polished by Newton on P_n; weights are w_i = 2 / ((1 - x_i^2) P_n'(x_i)^2).
    Results are cached; each call returns fresh float64 CPU copies.
    """
    x, w = _gauss_legendre(_order(n))
    return x.clone(), w.clone()
