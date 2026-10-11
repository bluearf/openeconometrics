"""X-11 method (native): classical X-11 seasonal adjustment on float64 Torch tensors.

This is the X-11 *method* (Shiskin, Young and Musgrave 1967; Ladiray and Quenneville
2001, "Seasonal Adjustment with the X-11 Method", Springer LNS 158), implemented in
OpenEconometrics. It is NOT X-13ARIMA-SEATS: there is no regARIMA pre-adjustment,
no forecast/backcast extension, no trading-day/holiday regression inside the
iterations, no MSR seasonal-filter selection and no SEATS. No external binary runs.

Operator conventions (``mode``): multiplicative ``a (-) b = a / b`` with irregular
mean 1, additive ``a (-) b = a - b`` with mean 0. ``M`` is the centred 2 x period
moving average (2x12 monthly, 2x4 quarterly), ``S33``/``S35`` the 3x3 / 3x5 seasonal
moving averages applied to each cycle position separately with the classical
asymmetric end weights (3x3: (5,11,11)/27, (3,7,10,7)/27; 3x5: (9,17,17,17)/60,
(4,11,15,15,15)/60, (4,8,13,13,13,9)/60), ``H`` a Henderson moving average with
Musgrave asymmetric end weights.

Tables (B: preliminary, C: extreme-value corrected, D: final):

    B1 = y;  B2 = M(B1);  B3 = B1 (-) B2 (centre span);  B4: replacements of extreme B3
    from the irregular B3 (-) normalized S33(B3);  B5 = normalized initial filter of
    B3 with B4 applied, extended periodically to the ends;  B6 = B1 (-) B5;
    B7 = H(B6);  B8 = B1 (-) B7;  B9: replacements of extreme B8;  B10 = normalized
    final filter of B8 with B9 applied;  B11 = B1 (-) B10;  B13 = B11 (-) B7;
    B17 weights and B20 extreme-value factors of B13.
    C1 = B1 (-) B20;  C2, C4, C5, C6, C7, C9, C10, C11 = B1 (-) C10, C13 as in B
    (no SI replacement);  C17 weights, C20 factors of C13.
    D1 = B1 (-) C20;  D2, D4, D5, D6, D7 as in C;  D8 = B1 (-) D7;  D9 = D1 (-) D7 where it
    differs from D8 (replacement values);  D10 = normalized final filter of D1 (-) D7;
    D11 = B1 (-) D10;  D12 = H(D1 (-) D10);  D13 = D11 (-) D12.

Normalization divides (subtracts) the centred 2 x period moving average of the
seasonal estimates, whose first/last period/2 values copy the nearest computed one.

Henderson length: fixed (5/7 quarterly, 9/13/23 monthly) or automatic from the
I/C ratio (mean absolute period-to-period change of the irregular over that of the
trend-cycle, from a H(period+1) pre-smoothing; quarterly ratios times 3): monthly
9 (I/C < 1), 13 (1 <= I/C < 3.5), 23 (>= 3.5); quarterly 5 (< 1), 7 (>= 3.5) and 5
otherwise; in B7 any ratio >= 1 uses H(period + 1). Musgrave end weights use the
X-11 ratios R = 1 (H9), 3.5 (H13), 4.5 (H23), 0.001 (H5 quarterly), 4.5 (H7
quarterly) with D = 4 / (pi R^2). Multiplicative trend-cycles replace nonpositive
values by the mean of the nearest positive neighbours.

Extreme values: moving five-year root mean squared deviation of the irregular
from its mean (0 or 1), computed by calendar years (the first/last two years use the
first/last five complete years, together with an incomplete first/last year), a
second pass without the zero-weight values; weights 1 below ``sigma_lower``, 0 above
``sigma_upper`` and linear in between (a year whose sigma is below 1e-5 is not
screened). An SI value with weight w < 1 is replaced by (w SI + the four nearest
full-weight values of the same cycle position) / (4 + w), two on each side when
available; the factors are I / (1 + w (I - 1)) and (1 - w) I. The procedure follows
the published X-11 tables as implemented in JDemetra+ 3 (jdplus-main
``jdplus.x13.base.core.x11``), which was read as the algorithm description.

All filters are tensor convolutions/gathers; the only Python loops run over the
fixed table steps, filter end positions and cycle-position length groups.
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
    SeasonalResult, _integer, check_replay, load_seasonal_series, sample_contract, state_record,
)
from openecon.resources import plan_workspace

LABEL = "X-11 method (native)"
MODES = ("multiplicative", "additive")
SEASONAL_FILTERS = ("3x3", "3x5")
HENDERSON = {12: (9, 13, 23), 4: (5, 7)}
_SEASONAL_WEIGHTS = {
    "3x3": ([1, 2, 3, 2, 1], 9.0, [[3, 7, 10, 7], [5, 11, 11]], 27.0),
    "3x5": ([1, 2, 3, 3, 3, 2, 1], 15.0, [[4, 8, 13, 13, 13, 9], [4, 11, 15, 15, 15], [9, 17, 17, 17]], 60.0),
}
_EPS = 1e-15
_NULL_SIGMA = 1e-5
NCYCLES = 5


# ---- filters -----------------------------------------------------------------------


def henderson_weights(length: int) -> Tensor:
    """Symmetric Henderson weights of odd ``length`` (lags -m..m)."""
    m = length // 2
    p = m + 2
    j = torch.arange(-m, m + 1, dtype=torch.float64)
    j2 = j.square()
    up = 315.0 * ((p - 1) ** 2 - j2) * (p ** 2 - j2) * ((p + 1) ** 2 - j2) * (3 * p ** 2 - 16 - 11 * j2)
    down = 8.0 * p * (p ** 2 - 1) * (4 * p ** 2 - 1) * (4 * p ** 2 - 9) * (4 * p ** 2 - 25)
    return up / down


def musgrave_ratio(length: int, period: int) -> float:
    """X-11 I/C ratio of the Musgrave end filters of a Henderson length."""
    if period == 4:
        return 0.001 if length <= 5 else 4.5
    return 1.0 if length <= 9 else 3.5 if length <= 13 else 4.5


def musgrave_filters(weights: Tensor, ratio: float) -> list[Tensor]:
    """Musgrave asymmetric filters: entry k has 2m - k weights on lags -m..m-1-k."""
    n = weights.shape[0]
    m = n // 2
    d = 4.0 / (math.pi * ratio * ratio)
    out = []
    for k in range(m):
        size = n - 1 - k                                    # available points
        kept, lost = weights[:size], weights[size:]
        index = torch.arange(1, n + 1, dtype=torch.float64)
        centre = (size + 1) / 2.0
        p1 = float(lost.sum()) / size
        p2 = float((lost * (index[size:] - centre)).sum())
        scale = d / (1.0 + size * (size - 1) * (size + 1) * d / 12.0)
        out.append(kept + p1 + (index[:size] - centre) * scale * p2)
    return out


def seasonal_filters(kind: str) -> tuple[Tensor, list[Tensor]]:
    symmetric, total, ends, end_total = _SEASONAL_WEIGHTS[kind]
    return (torch.tensor(symmetric, dtype=torch.float64) / total,
            [torch.tensor(w, dtype=torch.float64) / end_total for w in ends])


def trend_ma_weights(period: int) -> Tensor:
    """Centred 2 x period moving average (period even) or simple period average."""
    if period % 2:
        return torch.full((period,), 1.0 / period, dtype=torch.float64)
    weights = torch.full((period + 1,), 1.0 / period, dtype=torch.float64)
    weights[0] = weights[-1] = 0.5 / period
    return weights


def _valid(x: Tensor, weights: Tensor) -> Tensor:
    """Valid-mode symmetric moving average along the last dimension."""
    return x.unfold(-1, weights.shape[0], 1) @ weights


def filter_with_ends(x: Tensor, symmetric: Tensor, ends: list[Tensor]) -> Tensor:
    """Symmetric filter in the interior and mirrored asymmetric end filters (rows of x)."""
    length = x.shape[-1]
    m = symmetric.shape[0] // 2
    out = torch.empty_like(x)
    out[..., m:length - m] = _valid(x, symmetric)
    for k, weights in enumerate(ends):
        width = weights.shape[0]
        out[..., length - m + k] = x[..., length - width:] @ weights
        out[..., m - 1 - k] = x[..., :width].flip(-1) @ weights
    return out


def _subseries_filter(x: Tensor, period: int, kind: str) -> Tensor:
    """Seasonal moving average applied to each cycle position of x (by index mod period)."""
    symmetric, ends = seasonal_filters(kind)
    length = x.shape[0]
    rows = -(-length // period)
    padded = torch.cat([x, torch.full((rows * period - length,), math.nan, dtype=torch.float64)])
    grid = padded.reshape(rows, period).T
    counts = [(length - 1 - j) // period + 1 for j in range(period)]
    out = torch.full_like(grid, math.nan)
    for k in sorted(set(counts)):
        members = torch.tensor([j for j in range(period) if counts[j] == k])
        out[members, :k] = filter_with_ends(grid[members, :k], symmetric, ends)
    return out.T.reshape(-1)[:length]


def _remove(a: Tensor, b: Tensor, mul: bool) -> Tensor:
    return a / b if mul else a - b


def _normalize(s: Tensor, extend: int, period: int, mul: bool) -> Tensor:
    weights = trend_ma_weights(period)
    drop = weights.shape[0] // 2
    centre = torch.empty_like(s)
    centre[drop:s.shape[0] - drop] = _valid(s, weights)
    centre[:drop] = centre[drop]
    centre[s.shape[0] - drop:] = centre[s.shape[0] - drop - 1]
    normalized = _remove(s, centre, mul)
    return _extend_periodic(normalized, extend, period) if extend else normalized


def _extend_periodic(x: Tensor, extend: int, period: int) -> Tensor:
    """Extend by ``extend`` values at each end, copying the values one period inside."""
    head = x[period - extend:period]
    tail = x[x.shape[0] - period:x.shape[0] - period + extend]
    return torch.cat([head, x, tail])


def _positive(x: Tensor) -> Tensor:
    """Replace nonpositive values by the mean of the nearest positive neighbours."""
    good = x > 0
    if bool(good.all()):
        return x
    if not bool(good.any()):
        raise AnalysisError("non_positive_trend", "The multiplicative trend-cycle is nonpositive "
                            "everywhere; use mode='additive'.")
    n = x.shape[0]
    index = torch.arange(n)
    before = torch.where(good, index, torch.full_like(index, -1)).cummax(0).values
    after = torch.where(good, index, torch.full_like(index, n)).flip(0).cummin(0).values.flip(0)
    before = torch.cat([torch.tensor([-1]), before[:-1]])
    after = torch.cat([after[1:], torch.tensor([n])])
    left = x[before.clamp(min=0)]
    right = x[after.clamp(max=n - 1)]
    both = (before >= 0) & (after < n)
    fill = torch.where(both, (left + right) / 2, torch.where(before >= 0, left, right))
    return torch.where(good, x, fill)


def _henderson(x: Tensor, length: int, period: int, mul: bool) -> Tensor:
    weights = henderson_weights(length)
    out = filter_with_ends(x, weights, musgrave_filters(weights, musgrave_ratio(length, period)))
    return _positive(out) if mul else out


def _abs_mean_variation(x: Tensor, mul: bool) -> float:
    change = (x[1:] - x[:-1]).abs()
    if mul:
        change = change / x[:-1]
    return float(change.mean())


def ic_ratio(s: Tensor, period: int, mul: bool) -> float:
    """I/C ratio from a symmetric H(period+1) pre-smoothing (quarterly ratio x 3)."""
    weights = henderson_weights(period + 1)
    drop = weights.shape[0] // 2
    trend = _valid(s, weights)
    irregular = _remove(s[drop:drop + trend.shape[0]], trend, mul)
    ratio = _abs_mean_variation(irregular, mul) / _abs_mean_variation(trend, mul)
    return ratio * 3.0 if period == 4 else ratio


def select_henderson(ratio: float, period: int) -> int:
    if 1.0 <= ratio < 3.5:
        return period + 1
    if ratio < 1.0:
        return 9 if period == 12 else 5
    return 23 if period == 12 else 7


# ---- extreme values -----------------------------------------------------------------


def _rms(values: Tensor, mean: float) -> Tensor:
    """Root mean square deviation from ``mean`` over the non-missing entries of each row."""
    present = ~torch.isnan(values)
    squared = torch.where(present, (values - mean).square(), torch.zeros_like(values))
    return (squared.sum(-1) / present.sum(-1)).sqrt()


def _sigmas(s: Tensor, start: int, period: int, mean: float) -> Tensor:
    """Moving five-year sigmas per calendar year of the series."""
    n = s.shape[0]
    years = 1 + (start + n - 1) // period
    complete = 1 + (start + n - 1) // period
    lead = 0
    if start:
        lead = period - start
        complete -= 1
    tail = (start + n) % period
    if tail:
        complete -= 1
    if complete < NCYCLES:
        return _rms(s.unsqueeze(0), mean)
    first = NCYCLES // 2 + (1 if lead else 0)
    last = first + complete - NCYCLES
    windows = s[lead:lead + (complete - 1) * period + period].unfold(0, NCYCLES * period, period)
    centre = _rms(windows, mean)[:last - first + 1]
    sigma = torch.empty(years, dtype=torch.float64)
    sigma[first:last + 1] = centre
    sigma[:first] = _rms(s[:lead + NCYCLES * period].unsqueeze(0), mean)[0] if lead else centre[0]
    if tail:
        position = lead + (last - first) * period
        sigma[last + 1:] = _rms(s[position:].unsqueeze(0), mean)[0]
    else:
        sigma[last + 1:] = centre[-1]
    return sigma


def _weights(s: Tensor, sigma: Tensor, start: int, period: int, mean: float,
             lower: float, upper: float) -> Tensor:
    year = ((torch.arange(s.shape[0]) + start) // period).clamp(max=sigma.shape[0] - 1)
    sd = sigma[year]
    lv, uv = sd * lower, sd * upper
    distance = (s - mean).abs()
    weights = torch.where(distance - lv > _EPS, (uv - distance) / (uv - lv), torch.ones_like(s))
    weights = torch.where(distance - uv > _EPS, torch.zeros_like(s), weights)
    return torch.where(sd.abs() < _NULL_SIGMA, torch.ones_like(s), weights)


def extreme_weights(s: Tensor, start: int, period: int, mul: bool, lower: float,
                    upper: float) -> Tensor:
    """Weights of the irregular ``s`` (two passes of moving five-year sigma limits)."""
    mean = 1.0 if mul else 0.0
    weights = _weights(s, _sigmas(s, start, period, mean), start, period, mean, lower, upper)
    if bool((weights < 1).any()):
        trimmed = torch.where(weights == 0, torch.full_like(s, math.nan), s)
        weights = _weights(s, _sigmas(trimmed, start, period, mean), start, period, mean, lower, upper)
    return weights


def replacements(values: Tensor, weights: Tensor, period: int) -> Tensor:
    """Replacement values for weights < 1 (NaN where the value is kept)."""
    n = values.shape[0]
    rows = -(-n // period)
    pad = rows * period - n
    grid = torch.cat([values, torch.full((pad,), math.nan, dtype=torch.float64)]).reshape(rows, period).T
    w = torch.cat([weights, torch.full((pad,), math.nan, dtype=torch.float64)]).reshape(rows, period).T
    full = w == 1.0
    count = full.sum(dim=1, keepdim=True)                                   # [period, 1]
    before = torch.cumsum(full.to(torch.int64), dim=1) - full.to(torch.int64)  # full entries strictly before
    # Full-weight values of each row in order, padded.
    order = torch.argsort((~full).to(torch.int64), dim=1, stable=True)
    ordered = torch.gather(grid, 1, order)
    nb, na = before, count - before
    lp, up = nb.clamp(max=2), na.clamp(max=2)
    take_before = torch.where(lp < 2, lp, torch.where(up < 2, torch.minimum(nb, 4 - up), lp))
    take_after = torch.where(lp < 2, torch.minimum(na, 4 - lp), up)
    total = torch.zeros_like(grid)
    for offset in range(4):
        prior = before - 1 - offset
        after = before + offset
        total = total + torch.where(offset < take_before, torch.gather(ordered, 1, prior.clamp(min=0)),
                                    torch.zeros_like(grid))
        total = total + torch.where(offset < take_after,
                                    torch.gather(ordered, 1, after.clamp(max=rows - 1)),
                                    torch.zeros_like(grid))
    found = (take_before + take_after) >= 4
    valid = ~torch.isnan(grid)
    averages = torch.where(valid, grid, torch.zeros_like(grid)).sum(1, keepdim=True) / valid.sum(1, keepdim=True)
    replaced = torch.where(found, (w * grid + total) / (4.0 + w), averages.expand_as(grid))
    replaced = torch.where(full | ~valid, torch.full_like(grid, math.nan), replaced)
    return replaced.T.reshape(-1)[:n]


def _apply(values: Tensor, corrections: Tensor) -> Tensor:
    return torch.where(torch.isnan(corrections), values, corrections)


def correction_factors(s: Tensor, weights: Tensor, mul: bool) -> Tensor:
    if mul:
        factor = s / (1.0 + weights * (s - 1.0))
        return torch.where(weights < 1, factor, torch.ones_like(s))
    return torch.where(weights < 1, s * (1.0 - weights), torch.zeros_like(s))


# ---- the B/C/D iteration --------------------------------------------------------------


def x11_kernel(y: Tensor, period: int, settings: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """All B, C and D tables (aligned to the full sample; NaN where undefined)."""
    mul = settings["mode"] == "multiplicative"
    start = settings["first_season"] - 1
    initial, final = settings["initial_filter"], settings["final_filter"]
    lower, upper = settings["sigma_lower"], settings["sigma_upper"]
    fixed = settings["henderson"]
    n = y.shape[0]
    ma = trend_ma_weights(period)
    drop = ma.shape[0] // 2
    tables: dict[str, Tensor] = {}
    record: dict[str, Any] = {}

    def full(values: Tensor, offset: int = 0) -> Tensor:
        out = torch.full((n,), math.nan, dtype=torch.float64)
        out[offset:offset + values.shape[0]] = values
        return out

    def trend(step: str, series: Tensor, first_pass: bool = False) -> Tensor:
        if fixed is not None:
            length, ratio = fixed, None
        else:
            ratio = ic_ratio(series, period, mul)
            length = (period + 1) if first_pass and ratio >= 1.0 else select_henderson(ratio, period)
        record[step] = {"henderson": length, "ic_ratio": ratio}
        return _henderson(series, length, period, mul)

    def screen(irregular: Tensor, position: int) -> Tensor:
        return extreme_weights(irregular, (start + position) % period, period, mul, lower, upper)

    with torch.no_grad():
        b1 = y
        b2 = _valid(b1, ma)
        b3 = _remove(b1[drop:n - drop], b2, mul)
        b4a = _normalize(_subseries_filter(b3, period, initial), 0, period, mul)
        weights = screen(_remove(b3, b4a, mul), drop)
        b4 = replacements(b3, weights, period)
        b5 = _normalize(_subseries_filter(_apply(b3, b4), period, initial), drop, period, mul)
        b6 = _remove(b1, b5, mul)
        b7 = trend("B7", b6, first_pass=True)
        b8 = _remove(b1, b7, mul)
        b9c = _normalize(_subseries_filter(b8, period, final), 0, period, mul)
        weights = screen(_remove(b8, b9c, mul), 0)
        b9 = replacements(b8, weights, period)
        b10 = _normalize(_subseries_filter(_apply(b8, b9), period, final), 0, period, mul)
        b11 = _remove(b1, b10, mul)
        b13 = _remove(b11, b7, mul)
        b17 = screen(b13, 0)
        b20 = correction_factors(b13, b17, mul)

        c1 = _remove(b1, b20, mul)
        c2 = _valid(c1, ma)
        c4 = _remove(c1[drop:n - drop], c2, mul)
        c5 = _normalize(_subseries_filter(c4, period, initial), drop, period, mul)
        c6 = _remove(c1, c5, mul)
        c7 = trend("C7", c6)
        c9 = _remove(c1, c7, mul)
        c10 = _normalize(_subseries_filter(c9, period, final), 0, period, mul)
        c11 = _remove(b1, c10, mul)
        c13 = _remove(c11, c7, mul)
        c17 = screen(c13, 0)
        c20 = correction_factors(c13, c17, mul)

        d1 = _remove(b1, c20, mul)
        d2 = _valid(d1, ma)
        d4 = _remove(d1[drop:n - drop], d2, mul)
        d5 = _normalize(_subseries_filter(d4, period, initial), drop, period, mul)
        d6 = _remove(d1, d5, mul)
        d7 = trend("D7", d6)
        d8 = _remove(b1, d7, mul)
        d9bis = _remove(d1, d7, mul)
        d9 = torch.where((d9bis - d8).abs() < 1e-9, torch.full_like(d8, math.nan), d9bis)
        d10 = _normalize(_subseries_filter(d9bis, period, final), 0, period, mul)
        d11bis = _remove(d1, d10, mul)
        final_ratio = ic_ratio(d11bis, period, mul)
        d12 = trend("D12", d11bis)
        record["D12"]["ic_ratio"] = final_ratio
        d11 = _remove(b1, d10, mul)
        d13 = _remove(d11, d12, mul)
    tables.update({
        "b1": b1, "b2": full(b2, drop), "b3": full(b3, drop), "b4": full(b4, drop), "b5": b5,
        "b6": b6, "b7": b7, "b8": b8, "b9": b9, "b10": b10, "b11": b11, "b13": b13, "b17": b17,
        "b20": b20, "c1": c1, "c2": full(c2, drop), "c4": full(c4, drop), "c5": c5, "c6": c6,
        "c7": c7, "c9": c9, "c10": c10, "c11": c11, "c13": c13, "c17": c17, "c20": c20,
        "d1": d1, "d2": full(d2, drop), "d4": full(d4, drop), "d5": d5, "d6": d6, "d7": d7,
        "d8": d8, "d9": d9, "d10": d10, "d11": d11, "d12": d12, "d13": d13,
    })
    finals = [tables[k] for k in ("d10", "d11", "d12", "d13")]
    if not all(bool(torch.isfinite(t).all()) for t in finals):
        raise AnalysisError("non_finite_result", "The X-11 iteration produced non-finite values.")
    return tables, record


# ---- public API --------------------------------------------------------------------


def x11_settings(n: int, period: int, *, mode: Any, seasonal_filter: Any,
                 initial_seasonal_filter: Any, henderson: Any, sigma_limits: Any,
                 first_season: Any) -> dict[str, Any]:
    if mode not in MODES:
        raise AnalysisError("invalid_option", "mode must be 'multiplicative' or 'additive'.")
    for name, value in (("seasonal_filter", seasonal_filter),
                        ("initial_seasonal_filter", initial_seasonal_filter)):
        if value not in SEASONAL_FILTERS:
            raise AnalysisError("invalid_option", f"{name} must be '3x3' or '3x5'.")
    if henderson is not None:
        henderson = _integer(henderson, "henderson", 5, 23)
        if henderson not in HENDERSON[period]:
            allowed = ", ".join(map(str, HENDERSON[period]))
            raise AnalysisError("invalid_option", f"henderson must be None (automatic) or one of "
                                f"{allowed} for period {period}.")
    try:
        lower, upper = (float(v) for v in sigma_limits)
    except (TypeError, ValueError):
        raise AnalysisError("invalid_option", "sigma_limits must be two numbers (lower, upper).") from None
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.5 < lower < upper):
        raise AnalysisError("invalid_option", "sigma_limits need 0.5 < lower < upper.")
    first_season = _integer(1 if first_season is None else first_season, "first_season", 1, period)
    # Every cycle position needs a full symmetric seasonal window: the SI span of the
    # initial filter is one year shorter than the series.
    span = {"3x3": 5, "3x5": 7}
    years = max(span[initial_seasonal_filter] + 1, span[seasonal_filter])
    if n < years * period:
        raise AnalysisError("insufficient_observations", f"The X-11 method with these seasonal "
                            f"filters needs at least {years} complete years ({years * period} "
                            "observations); shorter series would need the stable-filter fallbacks "
                            "of X-13, which are not implemented.")
    return {"period": period, "mode": mode, "initial_filter": initial_seasonal_filter,
            "final_filter": seasonal_filter, "henderson": henderson, "sigma_lower": lower,
            "sigma_upper": upper, "first_season": first_season}


_COMPONENTS = {"seasonal": "d10", "seasonally_adjusted": "d11", "trend": "d12",
               "irregular": "d13", "extreme_weight": "c17"}


def _build(series: Tensor, index, calendar, settings: dict[str, Any], tables: dict[str, Tensor],
           steps: dict[str, Any], source: dict[str, Any]) -> SeasonalResult:
    labels = list(index)
    components = {name: tables[key] for name, key in _COMPONENTS.items()}
    record = state_record("x11", settings, series, index, calendar, components, source)
    rows = {"period": labels, "observed": series.tolist(),
            **{name: value.tolist() for name, value in components.items()},
            "extreme_factor": tables["c20"].tolist(),
            "modified_si_replacement": tables["d9"].tolist()}
    mul = settings["mode"] == "multiplicative"
    combined = tables["d10"] * tables["d12"] * tables["d13"] if mul else (
        tables["d10"] + tables["d12"] + tables["d13"])
    title = f"{LABEL} seasonal adjustment of {source['y']} ({settings['mode']}, period {settings['period']})"
    return SeasonalResult(
        {"components": table(rows, title=title),
         "tables": table({"period": labels, **{k: v.tolist() for k, v in tables.items()}},
                         title="X-11 B, C and D tables")},
        title=title,
        method=LABEL,
        not_x13=("Native X-11 method: no X-13ARIMA-SEATS, regARIMA pre-adjustment, forecast "
                 "extension, trading-day/holiday estimation, MSR filter choice or SEATS."),
        mode=settings["mode"],
        settings=settings,
        nobs=int(series.shape[0]),
        sample=sample_contract(int(series.shape[0])),
        inference="descriptive filter output: no standard errors, p-values or intervals",
        calendar=calendar,
        henderson_steps=steps,
        final_ic_ratio=steps["D12"]["ic_ratio"],
        extreme_values_c17=int((tables["c17"] < 1).sum()),
        reconstruction_max_abs_error=float((combined - series).abs().max()),
        seasonal_state=record,
        notes=[
            "observed = seasonal x trend x irregular (multiplicative) or the sum (additive); "
            "seasonally_adjusted = D11, trend = D12, irregular = D13, seasonal = D10.",
            "extreme_weight is C17; extreme_factor is C20; modified_si_replacement is D9.",
            "Descriptive filter output: no sampling distribution or standard errors are reported.",
        ],
    )


@resident_cpu
def x11(data: Any, y: str, *, period: int | None = None, time: str | None = None,
        mode: str = "multiplicative", seasonal_filter: str = "3x5",
        initial_seasonal_filter: str = "3x3", henderson: int | None = None,
        sigma_limits: tuple[float, float] = (1.5, 2.5),
        first_season: int | None = None) -> SeasonalResult:
    """Classical X-11 seasonal adjustment, native implementation ("X-11 method (native)").

    Not X-13ARIMA-SEATS: no regARIMA pre-adjustment, forecast extension, calendar
    regression or SEATS (see ``oe.calendar_regressors`` and ``oe.arima`` for a
    separate regARIMA step). Monthly (period 12) or quarterly (period 4) series.

    Parameters
    ----------
    data, y : table and the series (complete; positive in multiplicative mode).
    period : 12 or 4; implied by a monthly/quarterly datetime ``time`` column.
    time : optional time column (integer consecutive periods or datetimes).
    mode : ``"multiplicative"`` (default) or ``"additive"``.
    seasonal_filter : final seasonal moving average (B9/B10, C10, D10): "3x5"
        (default) or "3x3".
    initial_seasonal_filter : first seasonal moving average (B4/B5, C5, D5): "3x3"
        (default, classical X-11) or "3x5".
    henderson : None (automatic by I/C ratio) or a fixed length (9, 13, 23 monthly;
        5, 7 quarterly).
    sigma_limits : lower and upper sigma limits of the extreme-value weights.
    first_season : cycle position (1..period) of the first observation; taken from a
        datetime calendar (month or quarter), default 1 otherwise. It positions the
        calendar years of the five-year sigma windows.

    Returns
    -------
    ``SeasonalResult`` with tables ``components`` (period, observed, seasonal D10,
    seasonally_adjusted D11, trend D12, irregular D13, extreme_weight C17,
    extreme_factor C20, modified_si_replacement D9) and ``tables`` (every B, C and D
    table aligned to the sample, NaN where undefined), attrs with the Henderson
    lengths and I/C ratios per step and the sealed saved state.

    Example
    -------
    >>> sa = oe.x11(df, "sales", time="month")
    >>> sa["components"][["period", "seasonally_adjusted"]].tail()
    """
    series, index, calendar, period = load_seasonal_series(data, y, time, period, what="X-11",
                                                          allowed_periods=(4, 12),
                                                          calendar_periods=True)
    if first_season is None and calendar is not None and calendar["kind"] == "datetime":
        stamp = index[0]
        first_season = (stamp.month - 1) % 12 + 1 if period == 12 else (stamp.month - 1) // 3 + 1
    elif first_season is not None and calendar is not None and calendar["kind"] == "datetime":
        raise AnalysisError("invalid_option", "first_season is taken from the datetime calendar; "
                            "do not pass it with a datetime time column.")
    settings = x11_settings(int(series.shape[0]), period, mode=mode, seasonal_filter=seasonal_filter,
                            initial_seasonal_filter=initial_seasonal_filter, henderson=henderson,
                            sigma_limits=sigma_limits, first_season=first_season)
    if mode == "multiplicative" and not bool((series > 0).all()):
        raise AnalysisError("nonpositive_values", "Multiplicative X-11 needs strictly positive "
                            "values; use mode='additive'.")
    plan_workspace("X-11 tables", {"tables": int(series.shape[0]) * 8 * 64})
    tables, steps = x11_kernel(series, period, settings)
    source = {"y": y, "time": time}
    return _build(series, index, calendar, settings, tables, steps, source)


def restore_x11(record: dict[str, Any], series: Tensor, index) -> SeasonalResult:
    settings = dict(record["settings"])
    source = record["source"]
    if (set(source) != {"y", "time"} or not isinstance(source["y"], str)
            or (source["time"] is not None and not isinstance(source["time"], str))):
        raise AnalysisError("invalid_state", "Saved X-11 source metadata is invalid.")
    period = settings.get("period")
    if period not in (4, 12):
        raise AnalysisError("invalid_state", "Saved X-11 settings have an invalid period.")
    expected = x11_settings(int(series.shape[0]), period, mode=settings.get("mode"),
                            seasonal_filter=settings.get("final_filter"),
                            initial_seasonal_filter=settings.get("initial_filter"),
                            henderson=settings.get("henderson"),
                            sigma_limits=(settings.get("sigma_lower"), settings.get("sigma_upper")),
                            first_season=settings.get("first_season"))
    if expected != settings:
        raise AnalysisError("invalid_state", "Saved X-11 settings are not a valid effective setting record.")
    if settings["mode"] == "multiplicative" and not bool((series > 0).all()):
        raise AnalysisError("invalid_state", "Saved multiplicative X-11 input is not positive.")
    tables, steps = x11_kernel(series, period, settings)
    components = {name: tables[key] for name, key in _COMPONENTS.items()}
    check_replay(record, components)
    return _build(series, index, record["calendar"], settings, tables, steps, source)
