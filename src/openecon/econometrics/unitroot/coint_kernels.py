"""Batched residual-test kernels for cointegration tests with break searches.

Every candidate break date (or pair of dates) defines one cointegrating
regression. The kernels here fit a whole batch of such regressions at once
(``[B, n, p]`` designs, QR least squares on float64 tensors) and evaluate the
residual statistics of Gregory and Hansen's (1996) published programs for
every member of the batch:

* ``adf_scan``: the Dickey-Fuller t ratio of the residual regression

      De_t = rho e_(t-1) + sum_(j=1..k) c_j De_(t-j) + v_t,   t = k+2..n,

  (no deterministic terms) for every lag order k = 0..kmax, with the lag
  rules of the authors' code (fixed, AIC, BIC, downward t);
* ``phillips_scan``: the Phillips (1987) Z_alpha and Z_t statistics with the
  quadratic-spectral correction of Gregory and Hansen's ``PHILLIPS``
  procedure (AR(1) plug-in bandwidth of Andrews 1991 computed on the
  first-stage innovations, weights for lags 1..floor(M)).

No loop runs over observations: the loops run over lag orders and over
chunks of candidate dates, each step being one batched tensor operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

_TINY = 1e-300


@dataclass
class BatchFit:
    """Batched OLS: coefficients, residuals, SSR, t ratios and a rank flag."""

    beta: Tensor        # [B, p]
    resid: Tensor       # [B, N]
    ssr: Tensor         # [B]
    tstat: Tensor       # [B, p]
    ok: Tensor          # [B] bool: full column rank and positive residual variance


@torch.no_grad()
def batched_ols(design: Tensor, target: Tensor, *, rank_tol: float = 1e-9) -> BatchFit:
    """OLS of ``target`` ([B, N] or [N]) on ``design`` ([B, N, p]) by batched QR.

    A member whose design loses rank (a column within ``rank_tol`` of the span
    of the preceding ones, relative to its norm) or whose residual sum of
    squares is zero is flagged ``ok = False``; its numbers are not used.
    """
    b, n, p = design.shape
    if target.dim() == 1:
        target = target.expand(b, n)
    q, r = torch.linalg.qr(design)
    diag = r.diagonal(dim1=-2, dim2=-1).abs()
    norms = design.square().sum(dim=1).sqrt()
    ok = (diag > rank_tol * norms.clamp_min(_TINY)).all(dim=-1)
    eye = torch.eye(p, dtype=design.dtype)
    safe_r = torch.where(ok[:, None, None], r, eye.expand(b, p, p))
    qty = (q.transpose(-1, -2) @ target[..., None])[..., 0]
    beta = torch.linalg.solve_triangular(safe_r, qty[..., None], upper=True)[..., 0]
    resid = target - (q @ qty[..., None])[..., 0]
    ssr = resid.square().sum(dim=-1)
    df = n - p
    s2 = ssr / df if df > 0 else torch.full_like(ssr, float("nan"))
    rinv = torch.linalg.solve_triangular(safe_r, eye.expand(b, p, p), upper=True)
    variance = s2[:, None] * rinv.square().sum(dim=-1)
    ok = ok & (ssr > 1e-28 * target.square().sum(dim=-1)) & (variance > 0).all(dim=-1)
    tstat = beta / variance.clamp_min(_TINY).sqrt()
    return BatchFit(beta=beta, resid=resid, ssr=ssr, tstat=tstat, ok=ok)


@dataclass
class AdfScan:
    statistic: Tensor   # [B] t ratio of e_(t-1) at the selected order
    lags: Tensor        # [B] int64 selected order
    ssr: Tensor         # [B] SSR of the selected Dickey-Fuller regression
    ok: Tensor          # [B]


@torch.no_grad()
def adf_scan(e: Tensor, *, rule: str, kmax: int, critical: float = 1.96,
             strict: bool = False) -> AdfScan:
    """Residual Dickey-Fuller t ratios for a batch ``e`` [B, n] with a lag rule.

    ``rule``: ``"fixed"`` (k = kmax), ``"aic"`` / ``"bic"`` (Gregory-Hansen
    program: ln(SSR/n1) + c (k+2)/n1 with c = 2 or ln n1 on each order's own
    sample n1 = n-1-k; the first minimum in k = 0..kmax wins), ``"t"``
    (downward from kmax: the largest k >= 1 whose last lagged difference has
    |t| >= ``critical`` (``> critical`` with ``strict``), else k = 0).
    """
    bsize, n = e.shape
    de = e[:, 1:] - e[:, :-1]
    orders = [kmax] if rule == "fixed" else list(range(kmax + 1))
    first, last, ssrs, oks, criteria = [], [], [], [], []
    for k in orders:
        columns = [e[:, k:n - 1]] + [de[:, k - j:n - 1 - j] for j in range(1, k + 1)]
        fit = batched_ols(torch.stack(columns, dim=-1), de[:, k:])
        rows = n - 1 - k
        first.append(fit.tstat[:, 0])
        last.append(fit.tstat[:, -1])
        ssrs.append(fit.ssr)
        oks.append(fit.ok)
        if rule in {"aic", "bic"}:
            penalty = 2.0 if rule == "aic" else math.log(rows)
            criteria.append(torch.log(fit.ssr.clamp_min(_TINY) / rows) + penalty * (k + 2) / rows)
    first_t, last_t = torch.stack(first, dim=1), torch.stack(last, dim=1)
    ssr_all, ok_all = torch.stack(ssrs, dim=1), torch.stack(oks, dim=1)
    if rule == "fixed":
        index = torch.zeros(bsize, dtype=torch.int64)
        chosen = torch.full((bsize,), kmax, dtype=torch.int64)
    elif rule in {"aic", "bic"}:
        values = torch.stack(criteria, dim=1)
        values = torch.where(ok_all, values, torch.full_like(values, float("inf")))
        index = torch.argmin(values, dim=1)
        chosen = index
    else:
        magnitude = last_t.abs()
        significant = magnitude > critical if strict else magnitude >= critical
        significant = significant & ok_all
        significant[:, 0] = True                               # k = 0 always stops
        positions = torch.arange(kmax + 1, dtype=torch.int64).expand(bsize, kmax + 1)
        index = torch.where(significant, positions, torch.full_like(positions, -1)).amax(dim=1)
        chosen = index
    gather = index[:, None]
    statistic = first_t.gather(1, gather)[:, 0]
    ok = ok_all.gather(1, gather)[:, 0]
    ssr = ssr_all.gather(1, gather)[:, 0]
    return AdfScan(statistic=statistic, lags=chosen, ssr=ssr, ok=ok)


@dataclass
class PhillipsScan:
    za: Tensor          # [B]
    zt: Tensor          # [B]
    bandwidth: Tensor   # [B]
    ok: Tensor          # [B]


def _qs_weight(c: Tensor) -> Tensor:
    z = 1.2 * math.pi * c
    return 75.0 / (6.0 * math.pi * c).square() * (torch.sin(z) / z - torch.cos(z))


@torch.no_grad()
def phillips_scan(e: Tensor) -> PhillipsScan:
    """Z_alpha and Z_t of Gregory and Hansen's ``PHILLIPS`` procedure for a batch [B, n].

    rho = sum e_t e_(t+1) / sum e_t^2 (t = 1..n-1); v_t = e_t - rho e_(t-1);
    the AR(1) coefficient b of v gives a2 = 4 b^2 / (1 - b)^4 and the
    bandwidth M = 1.3221 (a2 (n-1))^(1/5); lambda = sum_(j=1..floor(M)) w(j/M)
    gamma_j with quadratic-spectral weights and gamma_j = sum v_t v_(t+j)/(n-1);
    rho* = (sum e_t e_(t+1) - (n-1) lambda) / sum e_t^2,
    Z_alpha = n (rho* - 1), Z_t = (rho* - 1) / sqrt((gamma_0 + 2 lambda) / sum e_t^2).
    The lag sum is truncated at n-2 (the published program fails beyond it).
    """
    bsize, n = e.shape
    lead, lagged = e[:, 1:], e[:, :-1]
    sxx = lagged.square().sum(dim=1)
    sxy = (lagged * lead).sum(dim=1)
    rho = sxy / sxx.clamp_min(_TINY)
    v = lead - rho[:, None] * lagged                           # [B, nu]
    nu = n - 1
    vv = v[:, :-1].square().sum(dim=1)
    bu = (v[:, :-1] * v[:, 1:]).sum(dim=1) / vv.clamp_min(_TINY)
    su = (v[:, 1:] - bu[:, None] * v[:, :-1]).square().mean(dim=1)
    a2 = (4.0 * bu.square() * su / (1.0 - bu) ** 8) / (su / (1.0 - bu) ** 4)
    bandwidth = 1.3221 * (a2 * nu) ** 0.2
    finite = torch.isfinite(bandwidth)
    usable = torch.where(finite, bandwidth.floor(), torch.zeros_like(bandwidth))
    usable = usable.clamp(max=nu - 1)
    top = int(usable.max()) if bsize else 0
    lam = torch.zeros(bsize, dtype=e.dtype)
    if top > 0:
        lags = torch.arange(1, top + 1, dtype=e.dtype)
        if top <= 64:
            gamma = torch.stack([(v[:, :-j] * v[:, j:]).sum(dim=1) for j in range(1, top + 1)],
                                dim=1) / nu
        else:
            size = 1 << (2 * nu - 1).bit_length()
            spectrum = torch.fft.rfft(v, n=size, dim=1)
            gamma = torch.fft.irfft(spectrum.abs().square(), n=size, dim=1)[:, 1:top + 1] / nu
        safe_m = torch.where(finite & (bandwidth > 0), bandwidth, torch.ones_like(bandwidth))
        weights = _qs_weight(lags[None, :] / safe_m[:, None])
        weights = torch.where(lags[None, :] <= usable[:, None], weights, torch.zeros_like(weights))
        lam = (weights * gamma).sum(dim=1)
    rho_star = (sxy - (n - 1) * lam) / sxx.clamp_min(_TINY)
    za = n * (rho_star - 1.0)
    sigma2 = 2.0 * lam + v.square().sum(dim=1) / nu
    zt = (rho_star - 1.0) / (sigma2 / sxx.clamp_min(_TINY)).clamp_min(_TINY).sqrt()
    ok = finite & (sigma2 > 0) & (sxx > 0) & torch.isfinite(za) & torch.isfinite(zt)
    return PhillipsScan(za=za, zt=zt, bandwidth=bandwidth, ok=ok)


def gauss_round(value: float) -> int:
    """GAUSS/MATLAB ``round`` (half away from zero) of a nonnegative number."""
    return int(math.floor(value + 0.5))

