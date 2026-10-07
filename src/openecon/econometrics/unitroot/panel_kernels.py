"""Batched per-panel regressions for the panel unit-root and cointegration tests.

Panels of equal length are stacked as an [N, T] tensor and every per-panel
least-squares problem is solved in one batched Householder QR ([N, n, K]), so
there is no Python loop over panels or observations. Unbalanced panels are
processed as one batch per distinct length (and lag order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot import tables
from openecon.econometrics.unitroot.common import Panel
from openecon.engines.covariance import kernel_weights

_EPS = torch.finfo(torch.float64).eps


@dataclass
class BatchFit:
    beta: Tensor            # [N, K]
    se: Tensor              # [N, K] classical, SSR / (n - K)
    resid: Tensor           # [N, n]
    ssr: Tensor             # [N]
    nobs: int
    k: int


@torch.no_grad()
def batch_ols(x: Tensor, y: Tensor, what: str) -> BatchFit:
    """OLS of y [N, n] on x [N, n, K] for all panels at once (QR, no normal equations)."""
    count, n, k = x.shape
    if n <= k:
        raise AnalysisError("insufficient_observations", f"{what} has {n} observation(s) per "
                            f"panel for {k} parameter(s); use fewer lags or longer panels.")
    if k == 0:
        return BatchFit(torch.zeros((count, 0), dtype=torch.float64),
                        torch.zeros((count, 0), dtype=torch.float64), y,
                        y.square().sum(dim=1), n, 0)
    q, r = torch.linalg.qr(x)
    pivots = r.diagonal(dim1=1, dim2=2).abs()
    norms = torch.linalg.vector_norm(x, dim=1)
    if bool((pivots <= 1e-10 * norms).any()):
        raise AnalysisError("collinear_regressors", f"{what} is rank deficient in at least one "
                            "panel: a panel's series does not vary enough to identify every "
                            "term (constant series, exact trend or too many lags).")
    theta = (q.transpose(1, 2) @ y[:, :, None])
    resid = y - (q @ theta)[:, :, 0]
    beta = torch.linalg.solve_triangular(r, theta, upper=True)[:, :, 0]
    inverse = torch.linalg.solve_triangular(
        r, torch.eye(k, dtype=torch.float64).expand(count, k, k), upper=True)
    ssr = resid.square().sum(dim=1)
    se = (ssr / (n - k))[:, None].sqrt() * inverse.square().sum(dim=2).sqrt()
    return BatchFit(beta, se, resid, ssr, n, k)


def _deterministics(count: int, n: int, trend: str) -> list[Tensor]:
    columns = []
    if trend in {"constant", "trend"}:
        columns.append(torch.ones((count, n), dtype=torch.float64))
    if trend == "trend":
        columns.append(torch.arange(1, n + 1, dtype=torch.float64).expand(count, n))
    return columns


def adf_blocks(y: Tensor, lags: int, trend: str, start: int | None = None):
    """(Delta y_t [N, n], y_{t-1} [N, n], other regressors [N, n, lags + d]) for t = start+2..T."""
    start = lags if start is None else start
    count, total = y.shape
    n = total - 1 - start
    parameters = 1 + lags + {"none": 0, "constant": 1, "trend": 2}[trend]
    if n <= parameters:
        raise AnalysisError("insufficient_observations", f"The Dickey-Fuller regression has "
                            f"{max(n, 0)} observation(s) per panel for {parameters} "
                            "parameter(s); use fewer lags or longer panels.")
    dy = y[:, 1:] - y[:, :-1]
    columns = [dy[:, start - j:total - 1 - j] for j in range(1, lags + 1)]
    columns += _deterministics(count, n, trend)
    others = torch.stack(columns, dim=2) if columns \
        else torch.empty((count, n, 0), dtype=torch.float64)
    return dy[:, start:], y[:, start:total - 1], others


@torch.no_grad()
def adf_batch(y: Tensor, lags: int, trend: str, start: int | None = None) -> BatchFit:
    """Per-panel ADF regressions; coefficient 0 is that of y_{t-1}."""
    target, level, others = adf_blocks(y, lags, trend, start)
    return batch_ols(torch.cat([level[:, :, None], others], dim=2), target,
                     "The Dickey-Fuller regression")


def groups(panel: Panel) -> list[tuple[Tensor, Tensor]]:
    """[(panel indices, series [N_g, T_g])] with one group per distinct panel length."""
    series = panel.values[:, 0]
    starts = panel.counts.cumsum(dim=0) - panel.counts
    result = []
    for length in torch.unique(panel.counts).tolist():
        members = (panel.counts == length).nonzero().flatten()
        rows = starts[members][:, None] + torch.arange(length)[None, :]
        result.append((members, series[rows]))
    return result


def demean_cross_section(panel: Panel) -> None:
    """Subtract the cross-sectional mean of each period from the series (in place)."""
    periods, index = torch.unique(panel.period, return_inverse=True)
    sums = torch.zeros(len(periods), dtype=torch.float64).index_add_(0, index, panel.values[:, 0])
    sizes = torch.zeros(len(periods), dtype=torch.float64).index_add_(
        0, index, torch.ones_like(panel.values[:, 0]))
    panel.values[:, 0] -= (sums / sizes)[index]


@torch.no_grad()
def choose_lags(y: Tensor, trend: str, method: str, maxlag: int) -> Tensor:
    """Per-panel ADF lag order (int64 [N]) by AIC, BIC or HQIC on the common sample t > maxlag+1.

    The candidates are 1..maxlag, as Stata's ``lags(aic #)`` fits them (0 only
    when maxlag is 0). With M observations and K parameters the criterion is
    ``M ln(SSR/M) + K c``, c = 2 (AIC), ln M (BIC) or 2 ln ln M (HQIC): the
    Gaussian -2 ln L plus the penalty, up to a constant. Ties go to the
    smaller order.
    """
    count = y.shape[0]
    best = torch.zeros(count, dtype=torch.int64)
    value = torch.full((count,), math.inf, dtype=torch.float64)
    for order in range(maxlag, 0 if maxlag else -1, -1):
        fit = adf_batch(y, order, trend, start=maxlag)
        penalty = {"aic": 2.0, "bic": math.log(fit.nobs),
                   "hqic": 2.0 * math.log(math.log(fit.nobs))}[method]
        criterion = fit.nobs * torch.log((fit.ssr / fit.nobs).clamp_min(1e-300)) + fit.k * penalty
        better = criterion <= value
        best[better], value[better] = order, criterion[better]
    return best


@torch.no_grad()
def long_run_variances(d: Tensor, lags: int, kernel: str) -> Tensor:
    """Per-panel long-run variance of d [N, n] (no centring): g0 + 2 sum_j w_j g_j, divisor n."""
    n = d.shape[1]
    variance = d.square().sum(dim=1) / n
    reach = min(n - 1, lags if kernel != "quadratic_spectral" else n - 1)
    if lags == 0 or reach < 1:
        return variance
    weights = kernel_weights(lags, kernel, count=reach).tolist()
    for j, weight in enumerate(weights, start=1):
        if weight != 0.0:
            variance = variance + 2.0 * weight * (d[:, j:] * d[:, :n - j]).sum(dim=1) / n
    return variance


@torch.no_grad()
def detrend(y: Tensor, trend: str) -> Tensor:
    """Residuals of each row of y [N, n] on nothing, a constant, or a constant and trend."""
    if trend == "none":
        return y
    n = y.shape[1]
    columns = [torch.ones(n, dtype=torch.float64)]
    if trend == "trend":
        columns.append(torch.arange(1, n + 1, dtype=torch.float64))
    q = torch.linalg.qr(torch.stack(columns, dim=1))[0]
    return y - (y @ q) @ q.T


def mackinnon_p_batch(statistics: Tensor, case: str) -> Tensor:
    """MacKinnon (1994) p-values of Dickey-Fuller t statistics (N = 1), vectorized."""
    star, low, high = tables.TAU_STAR[case][0], tables.TAU_MIN[case][0], tables.TAU_MAX[case][0]
    small = [v * s for v, s in zip(tables.TAU_SMALL_P[case][0], tables.TAU_SMALL_SCALE,
                                   strict=True)]
    large = [v * s for v, s in zip(tables.TAU_LARGE_P[case][0], tables.TAU_LARGE_SCALE,
                                   strict=True)]
    t = statistics
    left = small[0] + small[1] * t + small[2] * t * t
    right = large[0] + large[1] * t + large[2] * t * t + large[3] * t * t * t
    p = torch.special.ndtr(torch.where(t <= star, left, right))
    p = torch.where(t > high, torch.ones_like(p), p)
    return torch.where(t < low, torch.zeros_like(p), p)
