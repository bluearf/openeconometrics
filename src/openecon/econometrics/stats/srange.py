"""Studentized range (Tukey) and Dunnett many-one distributions; no SciPy at run time.

Both are scale mixtures: with s = sqrt(chi2_nu / nu) independent of the normal part,

    P(Q > q) = int_0^inf f_nu(s) T(q s) ds,

where T(w) is the upper tail of the corresponding infinite-df statistic:

* studentized range of k standard normals (Tukey 1949; Hartley 1942):

      T_k(w) = 1 - k int phi(z) [Phi(z) - Phi(z - w)]^(k-1) dz
             = k int phi(z) Phi(z)^(k-1) {1 - [1 - Phi(z - w)/Phi(z)]^(k-1)} dz,

  the second form having no cancellation, so small tails keep relative accuracy;

* two-sided Dunnett statistic max_i |Z_i| with the product correlation
  corr(Z_i, Z_j) = lambda_i lambda_j, lambda_i = sqrt(n_i / (n_i + n_0)) (Dunnett 1955):

      T(w) = int phi(u) {1 - prod_i [Phi((lambda_i u + w)/c_i) - Phi((lambda_i u - w)/c_i)]} du,
      c_i = sqrt(1 - lambda_i^2).

Numerical method. log T is tabulated once per k (or per set of lambdas) at the
nodes of 16-point Gauss-Legendre panels of width 0.5 in w, each value by composite
Gauss-Legendre quadrature of the smooth inner integrand over the region where it
is not negligible, and is evaluated anywhere by barycentric interpolation inside
its panel (degree 15, relative error near 1e-13). The outer integral then costs
one interpolation per node: 70 panels of 16 nodes between the points where the
chi density has fallen by exp(-37), graded geometrically towards s = 0 because
s^(nu-1) is not analytic there for fractional nu (Games-Howell degrees of
freedom). The chi density is evaluated in a Stirling form without cancellation,
so nu = 1e6 is as accurate as nu = 2. Everything is vectorized over (q, nu)
pairs; quantiles use a bracketed Illinois iteration on log P(Q > q).

Accuracy (tests): about 1e-11 absolute against scipy.stats.studentized_range for
k = 2..100 and nu >= 2, and 1e-10 relative against the closed forms (k = 2 and a
single Dunnett comparison are Student's t) down to tail probabilities of 1e-250.
The approach follows the Gauss-Legendre treatment of Hartley's form by Copenhaver
and Holland (1988), which R's ptukey also uses, but tabulates and interpolates the
inner integral instead of nesting two fixed low-order rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.distributions import gauss_legendre

_ORDER = 16
_STEP = 0.5                 # width of a tabulation panel in w
_RANGE_MAX = 50.0           # T_k(w) is treated as 0 beyond (below 1e-270)
_DUNNETT_MAX = 37.0
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_CHUNK = 256
_FLOOR = 1e-320


def _invalid(message: str) -> KernelError:
    return KernelError("invalid_distribution_argument", message)


@lru_cache(maxsize=1)
def _rule() -> tuple[Tensor, Tensor, Tensor]:
    """Gauss-Legendre nodes, weights and barycentric interpolation weights."""
    x, w = gauss_legendre(_ORDER)
    bary = torch.sqrt((1.0 - x * x) * w)
    bary[1::2] *= -1.0
    return x, w, bary


def _panel_nodes(low: Tensor, high: Tensor, panels: int) -> tuple[Tensor, Tensor]:
    """Composite rule on [low, high] (one interval per row): nodes and weights [M, panels*16]."""
    x, w, _ = _rule()
    step = (high - low) / panels
    left = low[:, None] + step[:, None] * torch.arange(panels, dtype=torch.float64)
    nodes = left[:, :, None] + 0.5 * step[:, None, None] * (x + 1.0)
    weights = (0.5 * step)[:, None, None] * w.expand(low.shape[0], panels, _ORDER)
    return nodes.reshape(low.shape[0], -1), weights.reshape(low.shape[0], -1)


class _TailTable:
    """log T(w) on [0, w_max] as piecewise degree-15 interpolants on panels of width 0.5."""

    def __init__(self, function: Callable[[Tensor], Tensor], w_max: float, rate: float):
        x, _, _ = _rule()
        self.w_max = w_max
        self.rate = rate            # far tail: ln T(w) ~ -rate * w^2 / 2
        self.panels = int(round(w_max / _STEP))
        edges = _STEP * torch.arange(self.panels, dtype=torch.float64)
        nodes = edges[:, None] + 0.5 * _STEP * (x + 1.0)
        self.values = function(nodes.reshape(-1)).reshape(self.panels, _ORDER)

    def __call__(self, w: Tensor) -> Tensor:
        x, _, bary = _rule()
        flat = w.reshape(-1)
        index = torch.clamp(torch.floor(flat / _STEP), 0, self.panels - 1).to(torch.int64)
        local = 2.0 * (flat - index * _STEP) / _STEP - 1.0
        diff = local[:, None] - x[None, :]
        diff = torch.where(diff == 0.0, torch.full_like(diff, 1e-300), diff)
        terms = bary / diff
        out = (terms * self.values[index]).sum(1) / terms.sum(1)
        out = torch.where(flat > self.w_max, torch.full_like(out, -float("inf")), out)
        return out.reshape(w.shape)


@lru_cache(maxsize=16)
@torch.no_grad()
def _range_table(k: int) -> _TailTable:
    def log_tail(w: Tensor) -> Tensor:
        # The integrand lives where the density of the maximum does, and for large w
        # around z = w/2 (exp(-(z - w/2)^2 - w^2/4)).
        low = torch.clamp(0.5 * w - 7.5, min=-8.6)
        high = torch.clamp(0.5 * w + 7.5, min=9.2)
        z, weights = _panel_nodes(low, high, 24)
        log_cdf = torch.special.log_ndtr(z)
        share = torch.exp(torch.special.log_ndtr(z - w[:, None]) - log_cdf).clamp(max=1.0)
        bracket = -torch.expm1((k - 1) * torch.log1p(-share))
        density = torch.exp(math.log(k) - 0.5 * z * z - _LOG_SQRT_2PI + (k - 1) * log_cdf)
        return torch.log((weights * density * bracket).sum(1).clamp_min(_FLOOR))

    return _TailTable(log_tail, _RANGE_MAX, 0.5)


@lru_cache(maxsize=16)
@torch.no_grad()
def _dunnett_table(lambdas: tuple[float, ...]) -> _TailTable:
    lam = torch.tensor(lambdas, dtype=torch.float64)
    scale = torch.sqrt(1.0 - lam * lam)
    top = float(lam.max())
    width = min(1.0, 2.0 * float((scale / lam).min()))
    panels = math.ceil((top * _DUNNETT_MAX + 8.5) / width)
    if panels > 600:
        raise KernelError("dunnett_unbalanced", "Dunnett's distribution is not evaluated when a "
                          "group is several hundred times larger than the control group.")

    def log_tail(w: Tensor) -> Tensor:
        u, weights = _panel_nodes(torch.zeros_like(w), top * w + 8.5, panels)
        total = torch.zeros_like(u)
        for l_i, c_i in zip(lam.tolist(), scale.tolist(), strict=True):
            # Phi(x) = erfc(-x / sqrt 2) / 2 keeps relative accuracy in the lower tail.
            outside = 0.5 * (torch.erfc((l_i * u + w[:, None]) / (c_i * _SQRT2))
                             + torch.erfc((w[:, None] - l_i * u) / (c_i * _SQRT2)))
            total += torch.log1p(-outside.clamp(max=1.0))
        density = 2.0 * torch.exp(-0.5 * u * u - _LOG_SQRT_2PI)     # even integrand, u >= 0
        return torch.log((weights * density * (-torch.expm1(total))).sum(1).clamp_min(_FLOOR))

    return _TailTable(log_tail, _DUNNETT_MAX, 1.0)


def _log1pmx(delta: Tensor) -> Tensor:
    """ln(1 + delta) - delta without cancellation for small delta."""
    r = delta / (2.0 + delta)
    r2 = r * r
    series = torch.zeros_like(r)
    for j in range(23, 1, -2):
        series = r2 * (series + 1.0 / j)
    series = -r * delta + 2.0 * r * series
    return torch.where(delta.abs() < 0.3, series, torch.log1p(delta.clamp_min(-1.0)) - delta)


def _log_chi_density(s: Tensor, nu: Tensor) -> Tensor:
    """log density of s = sqrt(chi2_nu / nu).

    f(s) = 2 (nu/2)^(nu/2) s^(nu-1) exp(-nu s^2 / 2) / Gamma(nu/2); for nu >= 30 the
    constant uses Stirling's series and the exponent is written around s = 1.
    """
    a = 0.5 * nu[:, None]
    log_s = torch.log(s)
    direct = math.log(2.0) + a * torch.log(a) - torch.lgamma(a) + (2.0 * a - 1.0) * log_s \
        - a * s * s
    big = a.clamp_min(15.0)
    inv = 1.0 / big
    inv2 = inv * inv
    remainder = inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))
    delta = s - 1.0
    stirling = math.log(2.0) + 0.5 * torch.log(big / (2.0 * math.pi)) - remainder - log_s \
        + 2.0 * big * _log1pmx(delta) - big * delta * delta
    return torch.where(a < 15.0, direct, stirling)


@lru_cache(maxsize=1)
def _outer_rule() -> tuple[Tensor, Tensor]:
    """Relative nodes and weights on [0, 1]: 64 uniform panels, the first graded towards 0."""
    x, w, _ = _rule()
    edges = [0.0] + [8.0 ** -j / 64.0 for j in range(6, 0, -1)] + [i / 64.0 for i in range(1, 65)]
    left = torch.tensor(edges[:-1], dtype=torch.float64)
    width = torch.tensor(edges[1:], dtype=torch.float64) - left
    nodes = left[:, None] + 0.5 * width[:, None] * (x + 1.0)
    weights = 0.5 * width[:, None] * w
    return nodes.reshape(-1), weights.reshape(-1)


def _mixture_chunk(table: _TailTable, q: Tensor, nu: Tensor) -> Tensor:
    finite = torch.isfinite(nu)
    df = torch.where(finite, nu, torch.ones_like(nu))
    mode = torch.sqrt((df - 1.0).clamp_min(0.0) / df)
    half = 8.6 / torch.sqrt(df)
    # The integrand f(s) T(q s) is log-concave with curvature <= -nu; for large q its
    # peak moves from the mode of f down to about sqrt((nu - 2) / (nu + rate q^2)).
    peak = torch.sqrt((df - 2.0).clamp_min(0.0) / (df + table.rate * q * q))
    low = (peak - half).clamp_min(0.0)
    high = torch.minimum(mode + half, table.w_max / q.clamp_min(1e-300))
    length = (high - low).clamp_min(0.0)
    rel_nodes, rel_weights = _outer_rule()
    s = low[:, None] + length[:, None] * rel_nodes
    value = (length[:, None] * rel_weights
             * torch.exp(_log_chi_density(s, df) + table(q[:, None] * s))).sum(1)
    value = torch.where(finite, value, torch.exp(table(q)))
    return value.clamp(0.0, 1.0)


@torch.no_grad()
def _mixture_sf(table: _TailTable, q: Tensor, nu: Tensor) -> Tensor:
    out = torch.empty_like(q)
    for start in range(0, q.shape[0], _CHUNK):
        stop = start + _CHUNK
        out[start:stop] = _mixture_chunk(table, q[start:stop], nu[start:stop])
    return torch.where(q <= 0.0, torch.ones_like(out), out)


@torch.no_grad()
def _mixture_isf(table: _TailTable, p: Tensor, nu: Tensor) -> Tensor:
    """q with P(Q > q) = p, by a bracketed Illinois iteration on log P(Q > q) - log p.

    Elements are retired as soon as their residual is below 5e-13 (a relative error
    of that size in the tail probability) or their bracket has collapsed.
    """
    target = torch.log(p)

    def residual(q: Tensor, rows: Tensor) -> Tensor:
        return torch.log(_mixture_sf(table, q, nu[rows]).clamp_min(_FLOOR)) - target[rows]

    everything = torch.arange(p.shape[0])
    low = torch.zeros_like(p)
    f_low = -target
    high = torch.full_like(p, 4.0)
    f_high = residual(high, everything)
    for _ in range(80):
        rows = torch.nonzero(f_high > 0.0).reshape(-1)
        if not rows.numel():
            break
        low[rows], f_low[rows] = high[rows], f_high[rows]
        high[rows] = 2.0 * high[rows]
        f_high[rows] = residual(high[rows], rows)
    else:
        raise KernelError("numerical_failure", "The quantile is not finite for these degrees "
                          "of freedom and probability.")
    root = 0.5 * (low + high)
    side = torch.zeros_like(p)
    rows = everything
    for _ in range(200):
        a, b, fa, fb = low[rows], high[rows], f_low[rows], f_high[rows]
        x = (a * fb - b * fa) / (fb - fa)
        x = torch.where(torch.isfinite(x) & (x > a) & (x < b), x, 0.5 * (a + b))
        f_x = residual(x, rows)
        right = f_x < 0.0
        last = side[rows]
        fa = torch.where(right & (last > 0), 0.5 * fa, fa)
        fb = torch.where(~right & (last < 0), 0.5 * fb, fb)
        side[rows] = torch.where(right, torch.ones_like(last), -torch.ones_like(last))
        high[rows], f_high[rows] = torch.where(right, x, b), torch.where(right, f_x, fb)
        low[rows], f_low[rows] = torch.where(right, a, x), torch.where(right, fa, f_x)
        root[rows] = x
        done = (f_x.abs() <= 5e-13) | ((high[rows] - low[rows]) <= 1e-14 * x.clamp_min(1.0))
        rows = rows[~done]
        if not rows.numel():
            break
    return root


def _vectors(first: object, df: object, what: str) -> tuple[Tensor, Tensor]:
    a = torch.as_tensor(first, dtype=torch.float64).reshape(-1)
    nu = torch.as_tensor(df, dtype=torch.float64).reshape(-1)
    if a.numel() != nu.numel():
        if a.numel() != 1 and nu.numel() != 1:
            raise _invalid(f"{what} and the degrees of freedom must have matching lengths.")
        size = max(a.numel(), nu.numel())
        a, nu = a.expand(size).clone(), nu.expand(size).clone()
    if bool(torch.isnan(a).any()) or bool(torch.isnan(nu).any()) or bool((nu < 1.0).any()):
        raise _invalid("The degrees of freedom must be at least 1 and every argument must be "
                       "a number.")
    return a, nu


def _groups(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= 1000:
        raise _invalid("The number of groups must be an integer between 2 and 1000.")
    return k


def ptukey_sf(q: object, k: int, df: object) -> Tensor:
    """Upper tail P(Q > q) of the studentized range of ``k`` means with ``df`` error df.

    ``q`` and ``df`` are scalars or equally long vectors (``df`` may be infinite
    and fractional, >= 1); the result is a 1-D float64 tensor.
    """
    table = _range_table(_groups(k))
    values, nu = _vectors(q, df, "q")
    if bool(torch.isinf(values).any()):
        raise _invalid("q must be finite.")
    return _mixture_sf(table, values.clamp_min(0.0), nu)


def ptukey(q: object, k: int, df: object) -> Tensor:
    """Distribution function P(Q <= q) of the studentized range (R's ptukey)."""
    return 1.0 - ptukey_sf(q, k, df)


def qtukey(p: object, k: int, df: object) -> Tensor:
    """Quantile of the studentized range: q with P(Q <= q) = p (R's qtukey), 0 < p < 1."""
    table = _range_table(_groups(k))
    values, nu = _vectors(p, df, "p")
    if bool(((values <= 0.0) | (values >= 1.0)).any()):
        raise _invalid("p must be strictly between 0 and 1.")
    return _mixture_isf(table, 1.0 - values, nu)


def qtukey_upper(p: object, k: int, df: object) -> Tensor:
    """Upper-tail quantile of the studentized range: q with P(Q > q) = p, 0 < p < 1.

    Critical values are computed from the tail probability itself, so a very small
    significance level is not rounded away as it would be in ``qtukey(1 - p)``.
    """
    table = _range_table(_groups(k))
    values, nu = _vectors(p, df, "p")
    if bool(((values <= 0.0) | (values >= 1.0)).any()):
        raise _invalid("p must be strictly between 0 and 1.")
    return _mixture_isf(table, values, nu)


def _lambdas(lambdas: object) -> tuple[float, ...]:
    values = tuple(float(value) for value in torch.as_tensor(lambdas, dtype=torch.float64)
                   .reshape(-1).tolist())
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise _invalid("Dunnett's correlation factors must lie strictly between 0 and 1.")
    return values


def pdunnett_sf(d: object, lambdas: object, df: object) -> Tensor:
    """P(max_i |T_i| > d) for Dunnett's two-sided many-one t statistics.

    ``lambdas[i] = sqrt(n_i / (n_i + n_0))`` for treatment group i against the
    control group 0, so that corr(T_i, T_j) = lambdas[i] * lambdas[j].
    """
    table = _dunnett_table(_lambdas(lambdas))
    values, nu = _vectors(d, df, "d")
    if bool(torch.isinf(values).any()):
        raise _invalid("d must be finite.")
    return _mixture_sf(table, values.clamp_min(0.0), nu)


def qdunnett(p: object, lambdas: object, df: object) -> Tensor:
    """Two-sided Dunnett critical value d with P(max_i |T_i| <= d) = p."""
    table = _dunnett_table(_lambdas(lambdas))
    values, nu = _vectors(p, df, "p")
    if bool(((values <= 0.0) | (values >= 1.0)).any()):
        raise _invalid("p must be strictly between 0 and 1.")
    return _mixture_isf(table, 1.0 - values, nu)


def qdunnett_upper(p: object, lambdas: object, df: object) -> Tensor:
    """Two-sided Dunnett critical value from the tail: d with P(max_i |T_i| > d) = p."""
    table = _dunnett_table(_lambdas(lambdas))
    values, nu = _vectors(p, df, "p")
    if bool(((values <= 0.0) | (values >= 1.0)).any()):
        raise _invalid("p must be strictly between 0 and 1.")
    return _mixture_isf(table, values, nu)
