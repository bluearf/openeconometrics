"""Deterministic calendar and intervention regressors for regARIMA-style models.

``oe.calendar_regressors`` builds, for a monthly or quarterly calendar (optionally
extended by future periods), the trading-day, leap-year, length-of-period and
user-holiday regressors in the X-13ARIMA-SEATS definitions (U.S. Census Bureau,
X-13ARIMA-SEATS Reference Manual, ``regression`` spec: ``td``, ``td1coef``,
``lpyear``, ``lom``/``loq``):

- six trading-day contrasts ``td_mon .. td_sat``: number of Mondays .. Saturdays in
  the period minus the number of Sundays;
- one weekday contrast ``td_weekday``: weekdays - (5/2) weekend days;
- ``leap_year``: 0.75 in a period containing a leap-year February, -0.25 in a period
  containing a non-leap February, 0 otherwise;
- ``length_of_period``: days in the period minus the mean length (365.25 / 12 or
  365.25 / 4);
- ``holiday_<name>``: the number of user-listed holiday dates in each period.

The regressors are exogenous columns for the regression part of ``oe.arima``
(regression with ARIMA errors) and its forecasts; no holiday calendar (e.g. Easter)
is built in and no regressor is estimated here. Day counts use integer day numbers
and closed-form weekday counts on tensors (no loop over days or periods).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.tsworkflows.seasonal_common import MAX_PERIODS, _integer, natural_period
from openecon.econometrics.tsworkflows.ssdiffuse import _calendar

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
TRADING_DAY = ("six", "one", None)
_EPOCH = pd.Timestamp("1970-01-01")


def _dates(dates: Any, start: Any, periods: Any, frequency: Any, horizon: Any,
           what: str) -> tuple[pd.DatetimeIndex, int, int, str]:
    """All period stamps (sample then future), the sample size, seasons per year, frequency."""
    horizon = _integer(0 if horizon is None else horizon, "horizon", 0, 10_000)
    if dates is not None:
        if start is not None or periods is not None:
            raise AnalysisError("invalid_option", "Pass either dates or start/periods/frequency, not both.")
        try:
            index = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="raise"))
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_time", "dates must be valid datetimes.") from exc
        if frequency is not None:
            raise AnalysisError("invalid_option", "frequency is inferred from dates; do not pass it.")
    else:
        if start is None or periods is None or frequency is None:
            raise AnalysisError("invalid_option", "Pass dates, or start, periods and frequency.")
        periods = _integer(periods, "periods", 3, MAX_PERIODS)
        try:
            index = pd.date_range(start=pd.Timestamp(start), periods=periods, freq=frequency)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_time", "start/frequency do not define a calendar.") from exc
    if len(index) > MAX_PERIODS:
        raise AnalysisError("system_budget", f"{what} admits at most {MAX_PERIODS} periods.")
    if index.tz is not None:
        index = index.tz_localize(None)
    calendar = _calendar(index)
    per_year = natural_period(calendar["frequency"])
    if per_year not in (4, 12):
        raise AnalysisError("unsupported_calendar", f"{what} needs a monthly or quarterly calendar; "
                            f"got frequency '{calendar['frequency']}'.")
    n = len(index)
    if horizon:
        future = pd.date_range(start=index[-1], periods=horizon + 1, freq=calendar["frequency"])[1:]
        index = index.append(future)
    return index, n, per_year, calendar["frequency"]


def _bounds(index: pd.DatetimeIndex, per_year: int, frequency: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Day numbers [start, end) of each period and the first month/year of the period."""
    months = 12 // per_year
    offset = pd.tseries.frequencies.to_offset(frequency)
    ends_at_stamp = "End" in type(offset).__name__
    year = torch.tensor(index.year.tolist(), dtype=torch.int64)
    month = torch.tensor(index.month.tolist(), dtype=torch.int64)
    first = (year * 12 + month - 1) - (months - 1 if ends_at_stamp else 0)
    starts = first
    stops = first + months

    def days(month_count: Tensor) -> Tensor:
        stamps = pd.to_datetime({"year": (month_count // 12).tolist(),
                                 "month": (month_count % 12 + 1).tolist(), "day": 1})
        return torch.tensor((stamps - _EPOCH).dt.days.tolist(), dtype=torch.int64)

    return days(starts), days(stops), starts % 12 + 1, starts // 12


def _weekday_counts(start: Tensor, stop: Tensor) -> Tensor:
    """[periods, 7] counts of Mondays..Sundays in [start, stop) (day numbers since 1970-01-01)."""
    first = torch.tensor([(k - 3) % 7 for k in range(7)], dtype=torch.int64)   # 1970-01-01 = Thursday
    upper = torch.div(stop[:, None] - 1 - first[None, :], 7, rounding_mode="floor")
    lower = torch.div(start[:, None] - 1 - first[None, :], 7, rounding_mode="floor")
    return upper - lower


def _leap_year(first_month: Tensor, first_year: Tensor, months: int) -> Tensor:
    value = torch.zeros(first_month.shape[0], dtype=torch.float64)
    for j in range(months):
        absolute = first_year * 12 + first_month - 1 + j
        month, year = absolute % 12 + 1, absolute // 12
        leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
        value = value + torch.where(month == 2, torch.where(leap, 0.75, -0.25),
                                    torch.zeros_like(value))
    return value


@resident_cpu
def calendar_regressors(dates: Any = None, *, start: Any = None, periods: int | None = None,
                        frequency: str | None = None, horizon: int = 0,
                        trading_day: str | None = "six", leap_year: bool = True,
                        length_of_period: bool = False, holidays: dict[str, Any] | None = None) -> Any:
    """Trading-day, leap-year, length-of-period and holiday regressors (X-13 definitions).

    Parameters
    ----------
    dates : the sample's monthly or quarterly datetime stamps (regular, increasing);
        or give ``start``, ``periods`` and ``frequency`` (e.g. "MS", "ME", "QS").
    horizon : number of future periods appended after the sample (``future`` = True),
        for forecasting with ``oe.forecast(fit, horizon, exog=...)``.
    trading_day : ``"six"`` (td_mon..td_sat = count of each weekday minus Sundays),
        ``"one"`` (td_weekday = weekdays - 2.5 weekend days) or None.
    leap_year : add ``leap_year`` (0.75 / -0.25 in leap / non-leap February periods).
    length_of_period : add days in the period minus 365.25/12 (or /4).
    holidays : {name: list of dates}; adds ``holiday_<name>`` = number of listed dates
        in each period. Dates outside the calendar are counted in attrs, not used.

    Returns
    -------
    Table with ``period`` (the stamp), ``future`` and the regressor columns; attrs
    document the definitions. Merge it with the data on the time column and pass
    the regressor names as ``x`` of ``oe.arima``; pass the future rows as ``exog`` of
    ``oe.forecast``.

    Example
    -------
    >>> reg = oe.calendar_regressors(df["month"], horizon=12)
    >>> data = df.merge(reg, left_on="month", right_on="period")
    >>> fit = oe.arima(data=data, y="sales", x=[c for c in reg.columns if c.startswith("td_")],
    ...                time="month", order=(0, 1, 1), seasonal=(0, 1, 1), period=12)
    """
    what = "calendar_regressors"
    if trading_day not in TRADING_DAY:
        raise AnalysisError("invalid_option", "trading_day must be 'six', 'one' or None.")
    for name, flag in (("leap_year", leap_year), ("length_of_period", length_of_period)):
        if not isinstance(flag, bool):
            raise AnalysisError("invalid_option", f"{name} must be True or False.")
    index, n, per_year, frequency = _dates(dates, start, periods, frequency, horizon, what)
    start_day, stop_day, first_month, first_year = _bounds(index, per_year, frequency)
    columns: dict[str, Any] = {"period": list(index), "future": [i >= n for i in range(len(index))]}
    definitions: dict[str, str] = {}
    counts = _weekday_counts(start_day, stop_day).to(torch.float64)
    if trading_day == "six":
        for k in range(6):
            columns[f"td_{WEEKDAYS[k]}"] = (counts[:, k] - counts[:, 6]).tolist()
        definitions["td_<day>"] = "number of <day>s minus number of Sundays in the period"
    elif trading_day == "one":
        columns["td_weekday"] = (counts[:, :5].sum(1) - 2.5 * counts[:, 5:].sum(1)).tolist()
        definitions["td_weekday"] = "weekdays minus 5/2 times weekend days in the period"
    if leap_year:
        columns["leap_year"] = _leap_year(first_month, first_year, 12 // per_year).tolist()
        definitions["leap_year"] = "0.75 in leap-year February periods, -0.25 in other February periods, 0 otherwise"
    if length_of_period:
        mean = 365.25 / per_year
        columns["length_of_period"] = ((stop_day - start_day).to(torch.float64) - mean).tolist()
        definitions["length_of_period"] = f"days in the period minus {mean}"
    usage: dict[str, Any] = {}
    if holidays is not None:
        if not isinstance(holidays, dict) or not holidays:
            raise AnalysisError("invalid_option", "holidays must be a non-empty {name: dates} mapping.")
        for name, listed in holidays.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise AnalysisError("invalid_option", "Holiday names must be identifiers.")
            try:
                stamps = pd.DatetimeIndex(pd.to_datetime(pd.Index(list(listed)), errors="raise"))
            except (TypeError, ValueError) as exc:
                raise AnalysisError("invalid_option", f"holidays['{name}'] must list valid dates.") from exc
            if stamps.tz is not None:
                stamps = stamps.tz_localize(None)
            if stamps.hasnans or stamps.has_duplicates:
                raise AnalysisError("invalid_option", f"holidays['{name}'] has missing or repeated dates.")
            day = torch.tensor(((stamps.normalize() - _EPOCH).days).tolist(), dtype=torch.int64)
            inside = (day >= start_day[0]) & (day < stop_day[-1])
            slot = torch.searchsorted(start_day, day[inside], right=True) - 1
            value = torch.zeros(len(index), dtype=torch.float64).index_add_(
                0, slot, torch.ones(slot.shape[0], dtype=torch.float64))
            columns[f"holiday_{name}"] = value.tolist()
            usage[name] = {"listed": int(day.shape[0]), "used": int(inside.sum()),
                           "outside_calendar": int((~inside).sum())}
        definitions["holiday_<name>"] = "number of listed holiday dates falling in the period"
    regressors = [c for c in columns if c not in ("period", "future")]
    if not regressors:
        raise AnalysisError("invalid_option", "Request at least one regressor.")
    return table(columns, title="Calendar regressors", frequency=frequency, seasons_per_year=per_year,
                 sample_periods=n, future_periods=len(index) - n, regressors=regressors,
                 definitions=definitions, holidays=usage,
                 convention="X-13ARIMA-SEATS regression-spec definitions (td, td1coef, lpyear, lom/loq)",
                 inference="deterministic regressors; coefficients and covariance come from oe.arima")



# ---- intervention regressors ---------------------------------------------------------

EVENT_TYPES = ("pulse", "step", "ramp")


def _event_index(periods: Any, horizon: int) -> tuple[pd.Index, int, dict[str, Any] | None]:
    """Labels of the sample and future periods, the sample size and the calendar."""
    if isinstance(periods, int) and not isinstance(periods, bool):
        n = _integer(periods, "periods", 2, MAX_PERIODS)
        return pd.RangeIndex(n + horizon), n, None
    try:
        index = pd.Index(list(periods))
    except TypeError as exc:
        raise AnalysisError("invalid_time", "periods must be a row count or a sequence of period "
                            "labels.") from exc
    if not 2 <= len(index) <= MAX_PERIODS:
        raise AnalysisError("invalid_time", f"periods must list 2..{MAX_PERIODS} labels.")
    if pd.api.types.is_datetime64_any_dtype(index.dtype) or index.inferred_type in ("datetime64", "date"):
        index = pd.DatetimeIndex(pd.to_datetime(index))
        if index.tz is not None:
            index = index.tz_localize(None)
    elif pd.api.types.is_bool_dtype(index.dtype) or not pd.api.types.is_numeric_dtype(index.dtype):
        raise AnalysisError("invalid_time", "Period labels must be numeric periods or datetimes.")
    calendar = _calendar(index)
    n = len(index)
    if horizon:
        if calendar["kind"] == "datetime":
            future = pd.date_range(start=index[-1], periods=horizon + 1, freq=calendar["frequency"])[1:]
        else:
            future = pd.Index([index[-1] + calendar["step"] * h for h in range(1, horizon + 1)])
        index = index.append(future)
    return index, n, calendar


def _locate(index: pd.Index, label: Any, name: str, field: str) -> int:
    if isinstance(index, pd.RangeIndex) and (isinstance(label, bool) or not isinstance(label, int)):
        raise AnalysisError("invalid_option", f"Event '{name}': {field} must be a row position.")
    try:
        key = pd.Timestamp(label) if isinstance(index, pd.DatetimeIndex) else label
        position = index.get_loc(key)
    except (KeyError, TypeError, ValueError):
        raise AnalysisError("invalid_option", f"Event '{name}': {field}={label!r} is not a period of "
                            "the sample or of the requested horizon.") from None
    if not isinstance(position, int):
        raise AnalysisError("invalid_option", f"Event '{name}': {field} must identify one period.")
    return position


@resident_cpu
def intervention_regressors(periods: Any, events: list[dict[str, Any]], *, horizon: int = 0) -> Any:
    """Pulse, step and ramp intervention regressors, optionally with a fixed decay.

    Parameters
    ----------
    periods : the sample's period labels (consecutive integers or a regular datetime
        calendar), or the number of rows (positions 0..n-1).
    events : list of {"name", "type", "at", optional "end", optional "decay"}:

        - ``pulse``: 1 at ``at`` (X-13 additive outlier ``ao``); with ``decay`` d in
          [0, 1): d^(t - at) from ``at`` on (temporary change ``tc``);
        - ``step``: 1 from ``at`` on (level shift ``ls``); with ``decay`` d:
          (1 - d^(t - at + 1)) / (1 - d) from ``at`` on (a step through the bounded
          first-order transfer function 1 / (1 - d L), long-run gain 1 / (1 - d));
        - ``ramp``: 0 up to ``at``, (t - at) / (end - at) between, 1 from ``end`` on
          (X-13 ``rp`` plus one); without ``end`` the open ramp max(0, t - at).
          Ramps take no decay (the transfer would be unbounded).

        ``decay`` is a fixed, declared constant, not estimated: the regressor is linear
        in its coefficient, which ``oe.arima`` estimates with full covariance.
    horizon : future periods appended (``future`` = True) for ``oe.forecast(exog=...)``.

    Returns
    -------
    Table with ``period``, ``future`` and one column per event name.
    """
    horizon = _integer(0 if horizon is None else horizon, "horizon", 0, 10_000)
    if not isinstance(events, (list, tuple)) or not events:
        raise AnalysisError("invalid_option", "events must be a non-empty list of event records.")
    index, n, calendar = _event_index(periods, horizon)
    total = len(index)
    t = torch.arange(total, dtype=torch.float64)
    columns: dict[str, Any] = {"period": list(index), "future": [i >= n for i in range(total)]}
    records = []
    for event in events:
        if not isinstance(event, dict) or not {"name", "type", "at"} <= set(event) or \
                set(event) - {"name", "type", "at", "end", "decay"}:
            raise AnalysisError("invalid_option", "Each event needs name, type and at (optional end, decay).")
        name, kind = event["name"], event["type"]
        if not isinstance(name, str) or not name.isidentifier() or name in columns:
            raise AnalysisError("invalid_option", f"Event name {name!r} must be a new identifier "
                                "other than period/future.")
        if kind not in EVENT_TYPES:
            raise AnalysisError("invalid_option", f"Event '{name}': type must be pulse, step or ramp.")
        at = _locate(index, event["at"], name, "at")
        decay = event.get("decay")
        if decay is not None:
            if kind == "ramp":
                raise AnalysisError("unbounded_transfer", f"Event '{name}': a ramp through a "
                                    "first-order transfer function is unbounded; no decay is allowed.")
            if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0.0 <= decay < 1.0:
                raise AnalysisError("unbounded_transfer", f"Event '{name}': decay must lie in [0, 1) "
                                    "for a bounded first-order transfer effect.")
        if "end" in event and kind != "ramp":
            raise AnalysisError("invalid_option", f"Event '{name}': only ramps take an end.")
        lag = t - at
        after = lag >= 0
        rate = torch.tensor(0.0 if decay is None else float(decay), dtype=torch.float64)
        zero = torch.zeros_like(t)
        if kind == "pulse":                                 # d^(t - at); d = 0 is the pulse
            value = torch.where(after, rate.pow(lag.clamp(min=0)), zero)
        elif kind == "step":                                # sum_(j <= t - at) d^j
            value = torch.where(after, (1.0 - rate.pow(lag.clamp(min=0) + 1)) / (1.0 - rate), zero)
        else:
            if "end" in event:
                end = _locate(index, event["end"], name, "end")
                if end <= at:
                    raise AnalysisError("invalid_option", f"Event '{name}': end must follow at.")
                value = (lag / (end - at)).clamp(min=0.0, max=1.0)
            else:
                value = lag.clamp(min=0.0)
        columns[name] = value.tolist()
        records.append({"name": name, "type": kind, "at": index[at], "position": at,
                        "end": index[_locate(index, event["end"], name, "end")] if "end" in event else None,
                        "decay": decay, "in_sample": at < n})
    return table(columns, title="Intervention regressors", sample_periods=n, future_periods=total - n,
                 calendar=calendar, events=records, regressors=[r["name"] for r in records],
                 convention="pulse = X-13 ao (tc with decay), step = ls, ramp = X-13 rp + 1; "
                            "decays are fixed, bounded first-order transfer constants",
                 inference="deterministic regressors; coefficients and covariance come from oe.arima")
