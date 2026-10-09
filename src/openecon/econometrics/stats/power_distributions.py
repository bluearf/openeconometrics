"""Bounded noncentral probability laws for prospective designs, CPU float64.

Noncentral t integrates P(Z > c sqrt(V/nu) - delta) over independent
V~chi-square(nu). F and chi-square use the Poisson mixture of central laws.
Neither SciPy nor an asymptotic replacement is used. These private kernels
have explicit work limits; they are not a general unbounded distribution API.
"""
from __future__ import annotations

from functools import lru_cache
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import (
    _gauss_legendre, _log1pmx_tensor, _stirling_delta,
    beta_inc, chi2_sf, f_sf, t_isf,
)

MAX_NONCENTRALITY = 4096.
MAX_DF = 40000


def _limit(nc: float, *dfs: int) -> None:
    if not math.isfinite(nc) or not 0 <= nc <= MAX_NONCENTRALITY:
        raise AnalysisError("resource_limit", "Noncentrality exceeds the supported [0,4096] budget.")
    if any(not 1 <= df <= MAX_DF for df in dfs):
        raise AnalysisError("resource_limit", "Degrees of freedom exceed the [1,40000] budget.")


@lru_cache(maxsize=1024)
def t_cut(alpha: float, alternative: str, df: int) -> float:
    with torch.device("cpu"):
        return t_isf(alpha / (2 if alternative == "two-sided" else 1), df)


def _t_integral(cut: float, delta: float, df: int, order: int) -> float:
    root = math.sqrt(df)
    lower, upper = max(0., root - 12.), root + 12.
    panels = math.ceil((upper - lower) / .5)
    edges = [lower + (upper - lower) * k / panels for k in range(panels + 1)]
    # Resolve even a very sharp conditional-normal transition at tiny df/alpha.
    center, scale = delta * root / cut, root / abs(cut)
    edges += [center + k * scale for k in range(-12, 13, 2)
              if lower < center + k * scale < upper]
    edges = sorted(set(edges))
    with torch.device("cpu"), torch.no_grad():
        nodes, weights = _gauss_legendre(order)
        left = torch.tensor(edges[:-1], dtype=torch.float64, device="cpu")
        half = (torch.tensor(edges[1:], dtype=torch.float64, device="cpu") - left) / 2
        r = left[:, None] + half[:, None] * (nodes + 1)
        a = df / 2
        u = (r - root) * (r + root) / df
        # Stable chi density: gamma-front(df/2,r^2/2) * 2/r.
        log_ratio = torch.where(u.abs() <= .4, _log1pmx_tensor(u),
                                2 * torch.log(r / root) - u)
        density = torch.exp(.5 * math.log(a / (2 * math.pi))
                            + a * log_ratio - _stirling_delta(a)) * (2 / r)
        tail = .5 * torch.erfc((cut * r / root - delta) / math.sqrt(2))
        return float(((density * tail) @ weights) @ half)


def t_power(delta: float, df: int, alpha: float, alternative: str) -> float:
    _limit(delta * delta, df)
    cut = t_cut(alpha, alternative, df)
    sign = -1 if alternative == "lower" else 1
    def evaluate(order: int) -> float:
        upper = _t_integral(cut, sign * delta, df, order)
        return upper + (_t_integral(cut, -delta, df, order)
                        if alternative == "two-sided" else 0.)
    first, refined = evaluate(16), evaluate(32)
    if not math.isfinite(refined) or abs(first - refined) > 2e-11:
        raise AnalysisError("numerical_failure", "Noncentral-t quadrature did not meet its error gate.")
    if not -2e-12 <= refined <= 1 + 2e-12:
        raise AnalysisError("numerical_failure", "Invalid noncentral-t probability.")
    return min(1., max(0., refined))


@lru_cache(maxsize=256)
def _poisson(nc: float) -> tuple[tuple[int, float], ...]:
    """Centered recurrence, <= 1150 terms, omitted mass <= 2e-14.

Weights are normalized after the independently checked tail-mass bound;
normalization changes the answer by at most that omitted mass plus rounding.
"""
    mu = nc / 2
    if mu == 0:
        return ((0, 1.),)
    radius = math.ceil(12 * math.sqrt(mu) + 30)
    low, high = max(0, math.floor(mu) - radius), math.ceil(mu) + radius
    # Poisson lower/upper tail = complementary gamma tails.
    lost = (chi2_sf(2 * mu, 2 * low) if low else 0.)
    # P(J>high) = gamma P(high+1,mu); evaluate directly, not 1-Q.
    from openecon.engines.distributions import chi2_cdf
    lost += chi2_cdf(2 * mu, 2 * (high + 1))
    if lost > 2e-14:
        raise AnalysisError("numerical_failure", "Poisson truncation mass exceeds its error budget.")
    center = math.floor(mu)
    weights = {center: 1.}
    for j in range(center + 1, high + 1):
        weights[j] = weights[j - 1] * mu / j
    for j in range(center - 1, low - 1, -1):
        weights[j] = weights[j + 1] * (j + 1) / mu
    total = math.fsum(weights.values())
    return tuple((j, weights[j] / total) for j in range(low, high + 1))


def f_power(cut: float, df1: int, df2: int, nc: float) -> float:
    _limit(nc, df1, df2)
    with torch.device("cpu"):
        if nc == 0:
            return f_sf(cut, df1, df2)
        x = df2 / (df2 + df1 * cut)
        return math.fsum(weight * beta_inc(df2 / 2, df1 / 2 + j, x)
                         for j, weight in _poisson(nc))


def chi_power(cut: float, df: int, nc: float) -> float:
    _limit(nc, df)
    with torch.device("cpu"):
        return math.fsum(weight * chi2_sf(cut, df + 2 * j)
                         for j, weight in _poisson(nc))
