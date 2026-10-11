"""STL: seasonal-trend decomposition by loess (Cleveland, Cleveland, McRae and Terpenning 1990).

Native float64 Torch translation of the published algorithm, following the
reference Fortran (netlib ``stl``; the copy shipped as R ``stats::stl`` and
translated by statsmodels ``STL``) step by step:

Inner loop (``inner`` passes), with the current trend T and robustness weights rho:

1. detrend ``y - T``;
2. cycle-subseries smoothing: every subseries (all observations at the same
   position in the cycle) is smoothed by loess with window ``seasonal`` and degree
   ``seasonal_degree`` and extended by one value at each end (loess evaluated at
   positions 0 and k+1), giving C of length n + 2*period;
3. low-pass filter of C: moving averages of length period, period and 3, then loess
   with window ``low_pass`` and degree ``low_pass_degree``, giving L;
4. seasonal ``S = C[period:period+n] - L``;
5. deseasonalize ``y - S``;
6. trend: loess of ``y - S`` with window ``trend`` and degree ``trend_degree``.

Outer loop (``outer`` passes): robustness weights rho = B(|R| / (6 median|R|)) with
the bisquare B, R = y - T - S, used in steps 2 and 6 of the next inner loop. With
``outer=0`` the weights are 1.

Loess at x uses the q = window nearest points, tricube weights
(1 - (d/h)^3)^3 with h the largest distance in the neighbourhood (increased by
(window - n)//2 when the window exceeds n), weights 1 for d <= 0.001h and 0 for
d > 0.999h, and a local constant (degree 0) or local line (degree 1, applied when
the weighted spread of positions exceeds 0.001 (n - 1)). With ``*_jump`` = J > 1
loess is evaluated at every J-th point (and at the last point) and linearly
interpolated in between, exactly as in the reference code.

Conventions
-----------
``convention="r"`` (default) uses the defaults of R ``stats::stl``: ``seasonal``
(s.window) must be given (odd integer or ``"periodic"``), seasonal degree 0, trend
and low-pass degree 1, jumps ceil(window/10), inner 2 (robust: 1) and outer 0
(robust: 15); ``seasonal="periodic"`` sets the window to 10n+1 with degree 0 and
replaces the seasonal component by its mean at each cycle position.
``convention="statsmodels"`` uses the defaults of statsmodels ``STL``: seasonal
window 7, all degrees 1, jumps 1, inner 5 (robust: 2), outer 0 (robust: 15).
Both: trend window = ceil(1.5 period / (1 - 1.5 / seasonal)) rounded up to an odd
integer, low-pass window = period rounded up to an odd integer. Windows must be odd
integers >= 3 (the reference code silently rounds even windows up; here they are
refused).

Robustness weights use the exact median of |R| (as statsmodels, which corrected the
netlib partial sort). R's ``stats::stl`` still uses the netlib ``psort``, which for
some residual vectors returns a non-median order statistic; robust R fits that hit
this defect differ (the test suite reproduces R exactly with a port of that sort).

Every loess call is a batched gather over evaluation points (chunked to a fixed
element budget); there is no loop over observations, only over iterations, cycle
subseries groups and evaluation chunks.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.tsworkflows.seasonal_common import (
    SeasonalResult, _integer, check_replay, future_labels, load_seasonal_series, load_state,
    sample_contract, state_record, strength,
)
from openecon.resources import plan_workspace

CONVENTIONS = ("r", "statsmodels")
CHUNK_ELEMENTS = 2_000_000
DEFAULT_MAX_WORK = 5_000_000_000
MAX_WORK = 50_000_000_000


# ---- loess kernel ------------------------------------------------------------------


def _loess_at(y: Tensor, rw: Tensor | None, length: int, degree: int, xs: Tensor,
              nleft: Tensor, width: int) -> tuple[Tensor, Tensor]:
    """Loess values at positions ``xs`` (1-based) for a batch of equal-length series.

    y, rw: [B, n]; xs: [E] float; nleft: [E] int64 (1-based) and every window has
    ``width`` points. Returns (values [B, E], ok [B, E]) as in the reference ``est``.
    """
    batch, n = y.shape
    count = xs.shape[0]
    values = torch.empty((batch, count), dtype=torch.float64)
    ok = torch.empty((batch, count), dtype=torch.bool)
    step = max(1, CHUNK_ELEMENTS // max(1, batch * width))
    offsets = torch.arange(width, dtype=torch.int64)
    for start in range(0, count, step):
        stop = min(count, start + step)
        x = xs[start:stop]
        left = nleft[start:stop]
        positions = left[:, None] + offsets[None, :]                     # [E, W], 1-based
        where = positions.to(torch.float64)
        right = (left + width - 1).to(torch.float64)
        h = torch.maximum(x - left.to(torch.float64), right - x)
        if length > n:
            h = h + float((length - n) // 2)
        distance = (where - x[:, None]).abs()
        ratio = distance / torch.where(h > 0, h, torch.ones_like(h))[:, None]
        weight = (1.0 - ratio.pow(3)).pow(3)
        weight = torch.where(distance <= 0.001 * h[:, None], torch.ones_like(weight), weight)
        weight = torch.where(distance <= 0.999 * h[:, None], weight, torch.zeros_like(weight))
        weight = weight.unsqueeze(0).expand(batch, -1, -1)
        if rw is not None:
            weight = weight * rw[:, positions - 1]
        total = weight.sum(dim=-1)
        good = total > 0
        weight = weight / torch.where(good, total, torch.ones_like(total)).unsqueeze(-1)
        if degree > 0:
            mean = (weight * where).sum(dim=-1)
            slope = x[None, :] - mean
            spread = (weight * (where - mean.unsqueeze(-1)).square()).sum(dim=-1)
            local = (h[None, :] > 0) & (spread.sqrt() > 0.001 * (n - 1))
            factor = slope / torch.where(local, spread, torch.ones_like(spread))
            adjusted = weight * (factor.unsqueeze(-1) * (where - mean.unsqueeze(-1)) + 1.0)
            weight = torch.where(local.unsqueeze(-1), adjusted, weight)
        values[:, start:stop] = (weight * y[:, positions - 1]).sum(dim=-1)
        ok[:, start:stop] = good
    return values, ok


def _windows(points: Tensor, n: int, length: int) -> tuple[Tensor, int]:
    """Left ends (1-based) of the loess neighbourhoods at integer ``points``."""
    if length >= n:
        return torch.ones_like(points), n
    half = (length + 1) // 2
    return (points - half).clamp(min=0).clamp(max=n - length) + 1, length


def loess_smooth(y: Tensor, length: int, degree: int, jump: int, rw: Tensor | None) -> Tensor:
    """The reference ``ess``: loess smoothing of every row of y [B, n] with jumps."""
    batch, n = y.shape
    if n < 2:
        return y.clone()
    jump = min(jump, n - 1)
    points = torch.arange(1, n + 1, jump, dtype=torch.int64)
    left, width = _windows(points, n, length)
    values, ok = _loess_at(y, rw, length, degree, points.to(torch.float64), left, width)
    values = torch.where(ok, values, y[:, points - 1])
    if jump == 1:
        return values
    last = int(points[-1])
    anchors = points
    if last != n:
        # The reference code evaluates the last point with the last neighbourhood used.
        end, end_ok = _loess_at(y, rw, length, degree, torch.tensor([float(n)], dtype=torch.float64),
                                left[-1:], width)
        end = torch.where(end_ok, end, y[:, n - 1:n])
        values = torch.cat([values, end], dim=1)
        anchors = torch.cat([points, torch.tensor([n], dtype=torch.int64)])
    position = torch.arange(1, n + 1, dtype=torch.int64)
    segment = (torch.searchsorted(anchors, position, right=True) - 1).clamp(max=anchors.shape[0] - 2)
    lower, upper = anchors[segment], anchors[segment + 1]
    fraction = ((position - lower).to(torch.float64) / (upper - lower).to(torch.float64))
    smooth = values[:, segment] + (values[:, segment + 1] - values[:, segment]) * fraction
    smooth[:, anchors - 1] = values
    return smooth


def _moving_average(x: Tensor, length: int) -> Tensor:
    return x.unfold(-1, length, 1).mean(dim=-1)


def _cycle_subseries(y: Tensor, rw: Tensor | None, period: int, length: int, degree: int,
                     jump: int) -> Tensor:
    """The reference ``ss``: smoothed, end-extended cycle subseries of length n + 2 period."""
    n = y.shape[0]
    rows = -(-n // period)
    pad = rows * period - n
    nan = torch.full((pad,), math.nan, dtype=torch.float64)
    grid = torch.cat([y, nan]).reshape(rows, period).T                  # [period, rows]
    weights = None if rw is None else torch.cat([rw, nan]).reshape(rows, period).T
    counts = torch.tensor([(n - 1 - j) // period + 1 for j in range(period)])
    out = torch.full((period, rows + 2), math.nan, dtype=torch.float64)
    for k in sorted(set(counts.tolist())):
        members = torch.nonzero(counts == k).flatten()
        block = grid[members, :k]
        block_rw = None if weights is None else weights[members, :k]
        smooth = loess_smooth(block, length, degree, jump, block_rw)
        width = min(length, k)
        first, first_ok = _loess_at(block, block_rw, length, degree,
                                    torch.tensor([0.0], dtype=torch.float64),
                                    torch.tensor([1], dtype=torch.int64), width)
        tail, tail_ok = _loess_at(block, block_rw, length, degree,
                                  torch.tensor([float(k + 1)], dtype=torch.float64),
                                  torch.tensor([max(1, k - length + 1)], dtype=torch.int64), width)
        first = torch.where(first_ok, first, smooth[:, :1])
        tail = torch.where(tail_ok, tail, smooth[:, -1:])
        out[members, :k + 2] = torch.cat([first, smooth, tail], dim=1)
    return out.T.reshape(-1)[:n + 2 * period]


def _robustness_weights(y: Tensor, fit: Tensor) -> Tensor:
    r = (y - fit).abs()
    n = r.shape[0]
    ordered = torch.sort(r).values
    mid1 = n // 2 + 1
    mid2 = n - mid1 + 1
    cmad = 3.0 * float(ordered[mid1 - 1] + ordered[mid2 - 1])
    weights = (1.0 - (r / cmad).square()).square() if cmad > 0 else torch.zeros_like(r)
    weights = torch.where(r <= 0.999 * cmad, weights, torch.zeros_like(r))
    return torch.where(r <= 0.001 * cmad, torch.ones_like(r), weights)


def stl_kernel(y: Tensor, period: int, settings: dict[str, Any]) -> dict[str, Tensor]:
    """Seasonal, trend, remainder and robustness weights of the STL iterations."""
    n = y.shape[0]
    ns, nt, nl = settings["seasonal"], settings["trend"], settings["low_pass"]
    sd, td, ld = settings["seasonal_degree"], settings["trend_degree"], settings["low_pass_degree"]
    sj, tj, lj = settings["seasonal_jump"], settings["trend_jump"], settings["low_pass_jump"]
    inner, outer = settings["inner"], settings["outer"]
    trend = torch.zeros(n, dtype=torch.float64)
    season = torch.zeros(n, dtype=torch.float64)
    rw: Tensor | None = None
    passes = 0
    with torch.no_grad():
        while True:
            for _ in range(inner):
                cycle = _cycle_subseries(y - trend, rw, period, ns, sd, sj)
                low = _moving_average(_moving_average(_moving_average(cycle, period), period), 3)
                low = loess_smooth(low.unsqueeze(0), nl, ld, lj, None)[0]
                season = cycle[period:period + n] - low
                trend = loess_smooth((y - season).unsqueeze(0), nt, td, tj,
                                     None if rw is None else rw.unsqueeze(0))[0]
            passes += 1
            if passes > outer:
                break
            rw = _robustness_weights(y, trend + season)
        weights = rw if outer > 0 and rw is not None else torch.ones(n, dtype=torch.float64)
        if settings["periodic"]:
            position = torch.arange(n) % period
            sums = torch.zeros(period, dtype=torch.float64).index_add_(0, position, season)
            counts = torch.zeros(period, dtype=torch.float64).index_add_(
                0, position, torch.ones(n, dtype=torch.float64))
            season = (sums / counts)[position]
    if not (bool(torch.isfinite(season).all()) and bool(torch.isfinite(trend).all())):
        raise AnalysisError("non_finite_result", "STL produced non-finite components; rescale the series.")
    return {"seasonal": season, "trend": trend, "remainder": y - season - trend,
            "robustness_weight": weights}


# ---- settings ----------------------------------------------------------------------


def _odd_window(value: Any, name: str) -> int:
    value = _integer(value, name, 3, 10 * 10**6)
    if value % 2 == 0:
        raise AnalysisError("invalid_option", f"{name} must be an odd integer >= 3 (the reference "
                            f"code would silently use {value + 1}).")
    return value


def _degree(value: Any, name: str) -> int:
    if isinstance(value, bool) or value not in (0, 1):
        raise AnalysisError("invalid_option", f"{name} must be 0 or 1.")
    return int(value)


def _next_odd(value: float) -> int:
    value = int(round(value))
    return value + 1 if value % 2 == 0 else value


def stl_settings(n: int, period: int, *, convention: str, seasonal: Any, seasonal_degree: Any,
                 trend: Any, trend_degree: Any, low_pass: Any, low_pass_degree: Any,
                 seasonal_jump: Any, trend_jump: Any, low_pass_jump: Any, robust: Any,
                 inner: Any, outer: Any) -> dict[str, Any]:
    """Effective STL settings after the convention's defaults and validation."""
    if convention not in CONVENTIONS:
        raise AnalysisError("invalid_option", "convention must be 'r' (stats::stl defaults) or "
                            "'statsmodels' (statsmodels STL defaults).")
    if not isinstance(robust, bool):
        raise AnalysisError("invalid_option", "robust must be True or False.")
    if n <= 2 * period:
        raise AnalysisError("insufficient_observations", f"STL needs more than two complete cycles "
                            f"(n > 2 * period = {2 * period}); got n = {n}.")
    periodic = False
    if isinstance(seasonal, str):
        if seasonal != "periodic":
            raise AnalysisError("invalid_option", "seasonal must be an odd integer >= 3 or 'periodic'.")
        periodic = True
        if seasonal_degree not in (None, 0):
            raise AnalysisError("invalid_option", "seasonal='periodic' fixes seasonal_degree at 0.")
        seasonal, seasonal_degree = 10 * n + 1, 0
    elif seasonal is None:
        if convention == "r":
            raise AnalysisError("invalid_option", "convention='r' follows stats::stl, which has no "
                                "default seasonal window: pass seasonal='periodic' or an odd integer.")
        seasonal = 7
    seasonal = _odd_window(seasonal, "seasonal")
    r_style = convention == "r"
    seasonal_degree = _degree((0 if r_style else 1) if seasonal_degree is None else seasonal_degree,
                              "seasonal_degree")
    trend_degree = _degree(1 if trend_degree is None else trend_degree, "trend_degree")
    low_pass_degree = _degree((trend_degree if r_style else 1) if low_pass_degree is None
                              else low_pass_degree, "low_pass_degree")
    if trend is None:
        trend = max(3, _next_odd(math.ceil(1.5 * period / (1.0 - 1.5 / seasonal))))
    trend = _odd_window(trend, "trend")
    if low_pass is None:
        low_pass = max(3, _next_odd(period))
    low_pass = _odd_window(low_pass, "low_pass")

    def jump(value: Any, window: int, name: str) -> int:
        if value is None:
            return max(1, math.ceil(window / 10)) if r_style else 1
        return _integer(value, name, 1, 10 * 10**6)

    # As in both references, ``robust`` only selects the default iteration counts;
    # an explicit outer > 0 runs robustness iterations.
    if inner is None:
        inner = (1 if robust else 2) if r_style else (2 if robust else 5)
    if outer is None:
        outer = 15 if robust else 0
    inner = _integer(inner, "inner", 1, 100)
    outer = _integer(outer, "outer", 0, 100)
    return {
        "convention": convention, "period": period, "periodic": periodic,
        "seasonal": seasonal, "trend": trend, "low_pass": low_pass,
        "seasonal_degree": seasonal_degree, "trend_degree": trend_degree,
        "low_pass_degree": low_pass_degree,
        "seasonal_jump": jump(seasonal_jump, seasonal, "seasonal_jump"),
        "trend_jump": jump(trend_jump, trend, "trend_jump"),
        "low_pass_jump": jump(low_pass_jump, low_pass, "low_pass_jump"),
        "robust": robust, "inner": inner, "outer": outer,
    }


def _admit(n: int, period: int, settings: dict[str, Any], max_work: Any) -> int:
    max_work = _integer(max_work, "max_work", 1, MAX_WORK)
    rows = -(-n // period)
    per_pass = (n * min(settings["seasonal"], rows) // settings["seasonal_jump"]
                + n * min(settings["trend"], n) // settings["trend_jump"]
                + n * min(settings["low_pass"], n) // settings["low_pass_jump"] + 4 * n)
    work = (settings["outer"] + 1) * settings["inner"] * per_pass
    if work > max_work:
        raise AnalysisError("search_budget", f"STL needs about {work} loess kernel evaluations, "
                            f"exceeding max_work={max_work}.")
    plan_workspace("STL loess neighbourhood chunks", {
        "series_and_components": n * 8 * 12,
        "neighbourhood_weights": CHUNK_ELEMENTS * 8 * 6,
    })
    return work


# ---- public API --------------------------------------------------------------------


def _build(series: Tensor, index, calendar, settings: dict[str, Any], components: dict[str, Tensor],
           source: dict[str, Any], work: int) -> SeasonalResult:
    seasonal, trend, remainder = (components[k] for k in ("seasonal", "trend", "remainder"))
    adjusted = series - seasonal
    labels = list(index)
    rows = {"period": labels, "observed": series.tolist(), "trend": trend.tolist(),
            "seasonal": seasonal.tolist(), "remainder": remainder.tolist(),
            "seasonally_adjusted": adjusted.tolist(),
            "robustness_weight": components["robustness_weight"].tolist()}
    record = state_record("stl", settings, series, index, calendar, components, source)
    reconstruction = float((trend + seasonal + remainder - series).abs().max())
    title = f"STL decomposition of {source['y']} (period {settings['period']})"
    return SeasonalResult(
        {"components": table(rows, title=title)},
        title=title,
        method="STL (Cleveland et al. 1990), native float64",
        mode="additive",
        convention=settings["convention"],
        settings=settings,
        nobs=int(series.shape[0]),
        sample=sample_contract(int(series.shape[0])),
        inference="descriptive decomposition: no standard errors, p-values or intervals",
        calendar=calendar,
        strength_seasonal=strength(seasonal, remainder),
        strength_trend=strength(trend, remainder),
        reconstruction_max_abs_error=reconstruction,
        work_units=work,
        seasonal_state=record,
        notes=[
            "observed = trend + seasonal + remainder; seasonally_adjusted = observed - seasonal.",
            f"Defaults follow {'R stats::stl' if settings['convention'] == 'r' else 'statsmodels STL'}; "
            "effective windows, degrees, jumps and iterations are listed in settings.",
            "Descriptive decomposition: no sampling distribution or standard errors are reported.",
        ],
    )


@resident_cpu
def stl(data: Any, y: str, *, period: int | None = None, time: str | None = None,
        seasonal: int | str | None = None, trend: int | None = None, low_pass: int | None = None,
        seasonal_degree: int | None = None, trend_degree: int | None = None,
        low_pass_degree: int | None = None, seasonal_jump: int | None = None,
        trend_jump: int | None = None, low_pass_jump: int | None = None, robust: bool = False,
        inner: int | None = None, outer: int | None = None, convention: str = "r",
        max_work: int = DEFAULT_MAX_WORK) -> SeasonalResult:
    """Seasonal-trend decomposition by loess (STL; Cleveland et al. 1990).

    Additive decomposition ``y = trend + seasonal + remainder`` of one complete,
    regularly spaced series. For a multiplicative decomposition decompose the
    logarithm of a positive series.

    Parameters
    ----------
    data, y : table and the numeric series (no missing values).
    period : observations per seasonal cycle; implied by a monthly (12) or quarterly
        (4) datetime ``time`` column, required otherwise; 2..366.
    time : optional time column (integer consecutive periods or a regular datetime
        calendar); rows are sorted by it.
    seasonal : seasonal (cycle-subseries) loess window, odd >= 3, or ``"periodic"``.
        Required with ``convention="r"`` (R's s.window); default 7 with
        ``convention="statsmodels"``.
    trend, low_pass : trend and low-pass loess windows (odd >= 3); defaults as in R
        and statsmodels (see the module documentation).
    seasonal_degree, trend_degree, low_pass_degree : local degree 0 or 1.
    seasonal_jump, trend_jump, low_pass_jump : loess evaluation step (linear
        interpolation in between); R default ceil(window / 10), statsmodels 1.
    robust : robustness iterations with bisquare weights.
    inner, outer : iteration counts (R: 2/0, robust 1/15; statsmodels: 5/0, robust 2/15).
    convention : ``"r"`` (stats::stl defaults) or ``"statsmodels"`` (STL defaults).
    max_work : bound on loess kernel evaluations (checked before computing).

    Returns
    -------
    ``SeasonalResult``: table ``components`` (period, observed, trend, seasonal,
    remainder, seasonally_adjusted, robustness_weight) and attrs with the effective
    settings, seasonal/trend strength and the sealed saved state (``to_json()``;
    ``oe.seasonal_restore``). Forecast with ``oe.stl_forecast``.

    R: ``stl(ts(y, frequency = 12), s.window = 7)``. statsmodels: ``STL(y, period=12).fit()``.

    Example
    -------
    >>> parts = oe.stl(df, "sales", time="month", seasonal=13, robust=True)
    >>> parts["components"][["period", "trend", "seasonal"]].head()
    """
    series, index, calendar, period = load_seasonal_series(data, y, time, period, what="STL")
    settings = stl_settings(int(series.shape[0]), period, convention=convention, seasonal=seasonal,
                            seasonal_degree=seasonal_degree, trend=trend, trend_degree=trend_degree,
                            low_pass=low_pass, low_pass_degree=low_pass_degree,
                            seasonal_jump=seasonal_jump, trend_jump=trend_jump,
                            low_pass_jump=low_pass_jump, robust=robust, inner=inner, outer=outer)
    work = _admit(int(series.shape[0]), period, settings, max_work)
    components = stl_kernel(series, period, settings)
    source = {"y": y, "time": time, "max_work": int(max_work)}
    return _build(series, index, calendar, settings, components, source, work)


def restore_stl(record: dict[str, Any], series: Tensor, index) -> SeasonalResult:
    settings = dict(record["settings"])
    source = record["source"]
    if (set(source) != {"y", "time", "max_work"} or not isinstance(source["y"], str)
            or (source["time"] is not None and not isinstance(source["time"], str))):
        raise AnalysisError("invalid_state", "Saved STL source metadata is invalid.")
    n = int(series.shape[0])
    expected = stl_settings(n, _integer(settings.get("period"), "period", 2, 366),
                            convention=settings.get("convention"),
                            seasonal="periodic" if settings.get("periodic") else settings.get("seasonal"),
                            seasonal_degree=settings.get("seasonal_degree"),
                            trend=settings.get("trend"), trend_degree=settings.get("trend_degree"),
                            low_pass=settings.get("low_pass"),
                            low_pass_degree=settings.get("low_pass_degree"),
                            seasonal_jump=settings.get("seasonal_jump"),
                            trend_jump=settings.get("trend_jump"),
                            low_pass_jump=settings.get("low_pass_jump"),
                            robust=settings.get("robust"), inner=settings.get("inner"),
                            outer=settings.get("outer"))
    if expected != settings:
        raise AnalysisError("invalid_state", "Saved STL settings are not a valid effective setting record.")
    work = _admit(n, settings["period"], settings, source["max_work"])
    components = stl_kernel(series, settings["period"], settings)
    check_replay(record, components)
    return _build(series, index, record["calendar"], settings, components, source, work)


@resident_cpu
def seasonal_restore(value: Any) -> SeasonalResult:
    """Rebuild an ``oe.stl`` or ``oe.x11`` result from its saved state (``to_json()``).

    The record's SHA-256 digest, schema, input series, period labels and calendar are
    checked, the deterministic decomposition is replayed from the stored input and
    settings, and the stored components must reproduce; otherwise ``invalid_state``.
    """
    record, series, index = load_state(value)
    if record["method"] == "stl":
        return restore_stl(record, series, index)
    if record["method"] == "x11":
        from openecon.econometrics.tsworkflows.x11 import restore_x11

        return restore_x11(record, series, index)
    raise AnalysisError("invalid_state", "Saved seasonal state names an unknown method.")


# ---- forecasting -------------------------------------------------------------------

FORECAST_METHODS = ("naive", "drift", "ets", "arima")


def _random_walk(adjusted: Tensor, steps: int, drift: bool) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """forecast::rwf on the adjusted series (drift: OLS mean of differences)."""
    diffs = adjusted[1:] - adjusted[:-1]
    m = diffs.shape[0]
    h = torch.arange(1, steps + 1, dtype=torch.float64)
    if drift:
        if m < 2:
            raise AnalysisError("insufficient_observations", "The drift forecast needs at least three periods.")
        slope = float(diffs.mean())
        residuals = diffs - slope
        mse = float(residuals.square().sum()) / (m - 1)
        slope_se = math.sqrt(mse / m)
        point = adjusted[-1] + h * slope
        variance = mse * h + (h * slope_se).square()
        info = {"drift": slope, "drift_std_error": slope_se, "sigma": math.sqrt(mse)}
    else:
        mse = float(diffs.square().sum()) / m
        point = adjusted[-1] + 0.0 * h
        variance = mse * h
        info = {"sigma": math.sqrt(mse)}
    return point, variance.sqrt(), info


@resident_cpu
def stl_forecast(result: SeasonalResult, steps: int, *, method: str = "naive", alpha: float = 0.05,
                 ets_model: str = "ANN", arima_order: Any = (0, 1, 1),
                 arima_constant: bool = True) -> Any:
    """Forecast from an STL decomposition (Hyndman and Athanasopoulos, ``forecast::forecast.stl``).

    The seasonally adjusted series ``y - seasonal`` is forecast by a non-seasonal
    method and the seasonal naive continuation of the STL seasonal component (the
    last cycle repeated) is added; the seasonal component is treated as known.

    method :
        ``"naive"`` random walk (sigma^2 = mean squared first difference;
        se_h = sigma sqrt(h)), ``"drift"`` random walk with drift (drift = mean
        difference; se_h^2 = s^2 h + h^2 se(drift)^2, s^2 with n - 2 df), both exactly
        as ``forecast::rwf``; ``"ets"`` the native additive innovations ETS
        (``oe.ets``) with non-seasonal ``ets_model`` ANN, AAN or AdN; ``"arima"`` the
        native ``oe.arima`` with ``arima_order`` (p, d, q) and ``arima_constant``.
        ETS/ARIMA parameters are estimated on the adjusted series and their native
        forecast functions supply the intervals.

    Intervals exclude decomposition, seasonal and (except the drift term) parameter
    uncertainty. Returns a table: period, forecast, std_error, ci_low, ci_high,
    seasonal, adjusted_forecast.
    """
    if not isinstance(result, SeasonalResult) or result.attrs.get("seasonal_state", {}).get("method") != "stl":
        raise AnalysisError("invalid_result", "stl_forecast needs a result returned by oe.stl.")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10_000:
        raise AnalysisError("invalid_steps", "steps must be an integer in 1..10000.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must lie strictly between 0 and 1.")
    if method not in FORECAST_METHODS:
        raise AnalysisError("invalid_option", f"method must be one of: {', '.join(FORECAST_METHODS)}.")
    record, series, index = load_state(result)
    components = {name: torch.tensor(values, dtype=torch.float64)
                  for name, values in record["components"].items()}
    period = record["settings"]["period"]
    seasonal = components["seasonal"]
    adjusted = series - seasonal
    n = int(series.shape[0])
    repeat = seasonal[n - period:].repeat(-(-steps // period))[:steps]
    details: dict[str, Any] = {}
    if method in ("naive", "drift"):
        point, se, details = _random_walk(adjusted, steps, method == "drift")
    else:
        import pandas as pd

        frame = pd.DataFrame({"adjusted": adjusted.tolist()})
        if method == "ets":
            if ets_model not in ("ANN", "AAN", "AdN"):
                raise AnalysisError("invalid_option", "ets_model must be a non-seasonal additive "
                                    "model: ANN, AAN or AdN.")
            from openecon.econometrics.tsworkflows.ets import ets, forecast as ets_forecast

            fit = ets(data=frame, y="adjusted", model=ets_model)
            out = ets_forecast(fit, steps, alpha=alpha)
            details = {"ets_model": ets_model, "parameters": {c.term: c.estimate for c in fit.coefficients}}
        else:
            from openecon.econometrics.arima.estimators import arima
            from openecon.econometrics.arima.forecast import forecast as arima_forecast

            if not isinstance(arima_constant, bool):
                raise AnalysisError("invalid_option", "arima_constant must be True or False.")
            fit = arima(data=frame, y="adjusted", order=tuple(arima_order), constant=arima_constant)
            out = arima_forecast(fit, steps, alpha=alpha)
            details = {"arima_order": list(fit.spec.options["order"]),
                       "parameters": {c.term: c.estimate for c in fit.coefficients}}
        point = torch.tensor(out["forecast"].tolist(), dtype=torch.float64)
        se = torch.tensor(out["std_error"].tolist(), dtype=torch.float64)
    from openecon.engines.distributions import normal_isf

    critical = normal_isf(alpha / 2.0)
    mean = point + repeat
    if not (bool(torch.isfinite(mean).all()) and bool(torch.isfinite(se).all())):
        raise AnalysisError("non_finite_forecast", "The STL forecast is not finite.")
    labels = future_labels(index, record["calendar"], steps)
    return table(
        {"period": labels, "forecast": mean.tolist(), "std_error": se.tolist(),
         "ci_low": (mean - critical * se).tolist(), "ci_high": (mean + critical * se).tolist(),
         "seasonal": repeat.tolist(), "adjusted_forecast": point.tolist()},
        title=f"STL + {method} forecasts of {record['source']['y']}",
        method=f"STL + {method}", steps=steps, alpha=alpha, confidence_level=1.0 - alpha,
        origin="end of the decomposed sample", details=details,
        uncertainty="adjusted-series forecast uncertainty only; the seasonal naive component is "
                    "treated as known and decomposition uncertainty is excluded",
    )
