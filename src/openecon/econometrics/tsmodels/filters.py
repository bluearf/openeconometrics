"""Trend-cycle filters (Stata's ``tsfilter hp | bk | cf``; EViews: HP and band-pass filters;
Hamilton's 2018 regression filter).

Each filter splits one series into a trend and a cycle, ``y_t = trend_t + cycle_t``:

``hp``  Hodrick-Prescott: the trend minimizes
        sum_t (y_t - tau_t)^2 + lambda sum_t (tau_(t+1) - 2 tau_t + tau_(t-1))^2,
        i.e. solves (I + lambda K'K) tau = y with K the (T-2) x T second-difference
        matrix. The pentadiagonal system is solved by a banded LDL' factorization in
        O(T) (a loop over periods on Python floats; no T x T matrix is formed).
``bk``  Baxter-King (1999): symmetric moving average of order K (``smaorder``) with the
        ideal band-pass weights b_0 = (w2 - w1)/pi, b_j = (sin(j w2) - sin(j w1))/(pi j),
        w1 = 2 pi / maxperiod, w2 = 2 pi / minperiod, shifted by a constant so that they
        sum to zero; the first and last K cycle values are not defined (missing).
``cf``  Christiano-Fitzgerald (2003) asymmetric full-sample filter for a random walk:
        c_t = B_0 y_t + sum_(j=1..T-t-1) B_j y_(t+j) + B~_(T-t) y_T + sum_(j=1..t-2) B_j y_(t-j)
              + B~_(t-1) y_1  (1-based), with the end weights making each row sum to zero.
        ``drift=True`` first removes the line through the first and last observation.
        The interior sums are one convolution, computed by FFT (O(T log T)).
``hamilton`` Hamilton (2018): OLS of y_(t+h) on a constant and y_t, ..., y_(t-p+1); the
        cycle is the residual and the trend the fitted value (dated t + h); the first
        h + p - 1 periods are not defined.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import load_series
from openecon.econometrics.core import kernel_call, table
from openecon.econometrics.tsmodels.common import check_positive
from openecon.engines.linalg import least_squares

METHODS = ("hp", "bk", "cf", "hamilton")


def hp_trend(y: Tensor, smooth: float) -> Tensor:
    """Hodrick-Prescott trend by banded LDL' of I + lambda K'K (O(T), no dense matrix)."""
    n = y.shape[0]
    if n < 3:
        return y.clone()
    lam = float(smooth)
    a = [1.0 + lam * ((i <= n - 3) + 4.0 * (1 <= i <= n - 2) + (i >= 2)) for i in range(n)]
    b = [-2.0 * lam * ((i <= n - 3) + (1 <= i <= n - 2)) for i in range(n - 1)]
    c = lam                                            # A[i, i+2] = lambda for every i
    d = [0.0] * n
    l1 = [0.0] * n
    l2 = [0.0] * n
    d[0] = a[0]
    l1[1] = b[0] / d[0]
    d[1] = a[1] - l1[1] * l1[1] * d[0]
    for i in range(2, n):
        l2i = c / d[i - 2]
        l1i = (b[i - 1] - l2i * d[i - 2] * l1[i - 1]) / d[i - 1]
        l2[i], l1[i] = l2i, l1i
        d[i] = a[i] - l1i * l1i * d[i - 1] - l2i * l2i * d[i - 2]
    values = y.tolist()
    z = [0.0] * n
    z[0] = values[0]
    z[1] = values[1] - l1[1] * z[0]
    for i in range(2, n):
        z[i] = values[i] - l1[i] * z[i - 1] - l2[i] * z[i - 2]
    x = [0.0] * n
    x[n - 1] = z[n - 1] / d[n - 1]
    x[n - 2] = z[n - 2] / d[n - 2] - l1[n - 1] * x[n - 1]
    for i in range(n - 3, -1, -1):
        x[i] = z[i] / d[i] - l1[i + 1] * x[i + 1] - l2[i + 2] * x[i + 2]
    return torch.tensor(x, dtype=torch.float64)


def bk_weights(minperiod: float, maxperiod: float, order: int) -> Tensor:
    """Baxter-King weights a_-K..a_K (symmetric, summing to zero)."""
    w1, w2 = 2.0 * math.pi / maxperiod, 2.0 * math.pi / minperiod
    j = torch.arange(1, order + 1, dtype=torch.float64)
    side = (torch.sin(w2 * j) - torch.sin(w1 * j)) / (math.pi * j)
    weights = torch.cat([side.flip(0), torch.tensor([(w2 - w1) / math.pi], dtype=torch.float64), side])
    return weights - weights.mean()


def bk_cycle(y: Tensor, minperiod: float, maxperiod: float, order: int) -> Tensor:
    """Baxter-King cycle with NaN in the first and last ``order`` periods."""
    n = y.shape[0]
    weights = bk_weights(minperiod, maxperiod, order)
    windows = y.unfold(0, 2 * order + 1, 1)              # [n - 2K, 2K + 1], a view
    out = torch.full((n,), math.nan, dtype=torch.float64)
    out[order:n - order] = windows @ weights
    return out


def _convolve(signal: Tensor, kernel: Tensor) -> Tensor:
    """Full linear convolution by FFT."""
    size = signal.shape[0] + kernel.shape[0] - 1
    length = 1 << max(1, (size - 1).bit_length())
    product = torch.fft.rfft(signal, length) * torch.fft.rfft(kernel, length)
    return torch.fft.irfft(product, length)[:size]


def cf_cycle(y: Tensor, minperiod: float, maxperiod: float, drift: bool) -> tuple[Tensor, Tensor]:
    """Christiano-Fitzgerald random-walk filter; returns (cycle, the filtered input)."""
    n = y.shape[0]
    x = y.clone()
    if drift:
        x = x - torch.arange(n, dtype=torch.float64) * (x[-1] - x[0]) / (n - 1)
    a, b = 2.0 * math.pi / maxperiod, 2.0 * math.pi / minperiod
    j = torch.arange(1, n + 1, dtype=torch.float64)
    bj = torch.cat([torch.tensor([(b - a) / math.pi], dtype=torch.float64),
                    (torch.sin(b * j) - torch.sin(a * j)) / (math.pi * j)])
    b0 = float(bj[0])
    cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), bj[1:].cumsum(0)])  # S_m
    interior = x.clone()
    interior[0] = 0.0
    interior[-1] = 0.0
    kernel = torch.cat([bj[1:n].flip(0), bj[:n]])        # B_|d| for d = -(n-1)..(n-1)
    conv = _convolve(interior, kernel)[n - 1:2 * n - 1]
    i = torch.arange(n)
    forward = cumulative[(n - i - 2).clamp_min(0)]
    backward = cumulative[(i - 1).clamp_min(0)]
    end_weight = -0.5 * b0 - forward
    start_weight = -b0 - forward - backward - end_weight
    edges = torch.zeros(n, dtype=torch.float64)
    edges[0] = b0 * x[0]
    edges[-1] = edges[-1] + b0 * x[-1]
    cycle = conv + edges + end_weight * x[-1] + start_weight * x[0]
    return cycle, x


def hamilton_filter(y: Tensor, h: int, p: int) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Hamilton (2018) regression filter: (trend, cycle) with NaN in the first h + p - 1 periods."""
    n = y.shape[0]
    lost = h + p - 1
    rows = n - lost
    if rows <= p + 2:
        raise AnalysisError("insufficient_observations", f"The Hamilton filter with h={h} and p={p} "
                            f"needs more than {lost + p + 2} observations.")
    columns = [torch.ones(rows, dtype=torch.float64)]
    columns += [y[p - 1 - j:p - 1 - j + rows] for j in range(p)]
    target = y[lost:]
    fit = kernel_call(least_squares, torch.stack(columns, dim=1), target.contiguous())
    trend = torch.full((n,), math.nan, dtype=torch.float64)
    cycle = torch.full((n,), math.nan, dtype=torch.float64)
    trend[lost:], cycle[lost:] = fit.fitted, fit.resid
    coefficients = {"Intercept": float(fit.beta[fit.kept.index(0)]) if 0 in fit.kept else 0.0}
    for j in range(p):
        coefficients[f"L{j}"] = float(fit.beta[fit.kept.index(j + 1)]) if j + 1 in fit.kept else 0.0
    return trend, cycle, coefficients


def tsfilter(data: Any, y: str, *, method: str = "hp", time: str | None = None,
             smooth: float = 1600.0, minperiod: float = 6.0, maxperiod: float = 32.0,
             smaorder: int = 12, drift: bool = True, hamilton_h: int = 8,
             hamilton_p: int = 4) -> Any:
    """Split a series into trend and cycle (Stata's ``tsfilter``; EViews HP / band-pass filters).

    Methods
    -------
    ``"hp"`` Hodrick-Prescott: trend tau minimizes ``sum (y - tau)^2 + smooth * sum
    (D^2 tau)^2`` (default ``smooth`` = 1600, quarterly data; 6.25 or 100 annual,
    129600 monthly). Exact solution of the pentadiagonal system by an O(T) banded
    factorization; the cycle is ``y - trend``.

    ``"bk"`` Baxter-King symmetric band-pass filter keeping periods between
    ``minperiod`` and ``maxperiod`` (defaults 6 and 32, quarterly business cycles)
    with moving-average order ``smaorder`` = K (default 12, Stata's default); the
    first and last K cycle values are missing (NaN).

    ``"cf"`` Christiano-Fitzgerald asymmetric full-sample band-pass filter (random-walk
    assumption) with the same period band; ``drift=True`` (default) first removes the
    straight line through the first and last observations.

    ``"hamilton"`` Hamilton (2018): OLS of ``y_(t+h)`` on a constant and ``y_t, ...,
    y_(t-p+1)`` with ``h = hamilton_h`` (default 8) and ``p = hamilton_p`` (default 4)
    for quarterly data (24 and 12 for monthly); cycle = residual, trend = fitted value,
    the first h + p - 1 periods missing.

    Parameters
    ----------
    data : table; y : the series (no missing values; consecutive periods);
    time : optional time column (rows are sorted by it; integer gaps are refused).

    Returns
    -------
    A table with one row per period: ``period`` (the time value, or the row number
    0..T-1), ``observed``, ``trend``, ``cycle`` (NaN where the filter is undefined).
    ``attrs``: ``method``, the filter settings, ``lost_start``/``lost_end`` and for
    Hamilton the regression coefficients.

    Stata: ``tsfilter hp c = y, smooth(1600) trend(t)``; ``tsfilter bk c = y,
    minperiod(6) maxperiod(32) smaorder(12)``; ``tsfilter cf c = y, drift``. EViews:
    Proc > Hodrick-Prescott Filter, Frequency Filter (BK, CF).

    Example
    -------
    >>> parts = oe.tsfilter(df, "lgdp", method="hp", time="quarter", smooth=1600)
    >>> parts[["period", "trend", "cycle"]].tail()
    """
    if method not in METHODS:
        raise AnalysisError("invalid_option", f"method must be one of: {', '.join(METHODS)}.")
    series, periods = load_series(data, y, time)
    n = series.shape[0]
    if n < 4:
        raise AnalysisError("insufficient_observations", "Filtering needs at least four observations.")
    settings: dict[str, Any] = {"method": method, "series": y, "nobs": n}
    if method in ("bk", "cf"):
        check_positive(minperiod, "minperiod", minimum=2.0, strict=False)
        check_positive(maxperiod, "maxperiod")
        if not maxperiod > minperiod:
            raise AnalysisError("invalid_option", "maxperiod must exceed minperiod.")
        settings.update(minperiod=minperiod, maxperiod=maxperiod)
    lost_start = lost_end = 0
    with torch.no_grad():
        if method == "hp":
            check_positive(smooth, "smooth")
            trend = hp_trend(series, smooth)
            cycle = series - trend
            settings["smooth"] = smooth
        elif method == "bk":
            check_positive(smaorder, "smaorder", integer=True)
            if 2 * smaorder + 1 > n:
                raise AnalysisError("insufficient_observations", f"smaorder={smaorder} needs at least "
                                    f"{2 * smaorder + 1} observations.")
            cycle = bk_cycle(series, minperiod, maxperiod, smaorder)
            trend = series - cycle
            lost_start = lost_end = smaorder
            settings["smaorder"] = smaorder
        elif method == "cf":
            if not isinstance(drift, bool):
                raise AnalysisError("invalid_option", "drift must be True or False.")
            cycle, _ = cf_cycle(series, minperiod, maxperiod, drift)
            trend = series - cycle
            settings["drift"] = drift
        else:
            check_positive(hamilton_h, "hamilton_h", integer=True)
            check_positive(hamilton_p, "hamilton_p", integer=True)
            trend, cycle, coefficients = hamilton_filter(series, hamilton_h, hamilton_p)
            lost_start = hamilton_h + hamilton_p - 1
            settings.update(hamilton_h=hamilton_h, hamilton_p=hamilton_p,
                            coefficients=coefficients)
    if not bool(torch.isfinite(cycle[lost_start:n - lost_end]).all()):
        raise AnalysisError("non_finite_result", "The filter produced non-finite values; rescale "
                            "the series.")
    labels = {"hp": "Hodrick-Prescott", "bk": "Baxter-King", "cf": "Christiano-Fitzgerald",
              "hamilton": "Hamilton regression"}
    return table({"period": periods.tolist(), "observed": series.tolist(), "trend": trend.tolist(),
                  "cycle": cycle.tolist()},
                 title=f"{labels[method]} filter of {y}", lost_start=lost_start,
                 lost_end=lost_end, **settings)
