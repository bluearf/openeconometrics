"""Shared calendar, period and saved-state contracts of the native seasonal routines.

The seasonal decompositions (``oe.stl``, ``oe.x11``) work on one complete,
regularly spaced numeric series. This module

- loads the series in time order (``arima.diagnostics.load_series``: sorted by the
  time column, integer periods consecutive, no missing values, no silent drops);
- validates the calendar: datetimes need an inferable regular frequency, and a
  monthly or quarterly calendar fixes the seasonal period (12 or 4; a declared
  period that contradicts it is refused); other calendars need an explicit period;
- builds future period labels for forecasts;
- defines the portable saved state: the complete input series, its period labels
  and calendar, the effective settings and the components, sealed with a SHA-256
  digest. Restoring replays the deterministic decomposition from the stored input
  and settings and refuses a record whose components do not reproduce.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import load_series
from openecon.econometrics.core import TableSet
from openecon.econometrics.tsworkflows.ssdiffuse import (
    _calendar, _digest, _index_record, _json_admission, _load_record, _restore_index,
)
from openecon.resources import plan_workspace

STATE_SCHEMA = "openecon.seasonal-adjustment.v1"
MAX_PERIODS = 50_000
MAX_PERIOD = 366
# Relative tolerance of the replay check on restore (pure float64 recomputation).
REPLAY_TOLERANCE = 1e-9


class SeasonalResult(TableSet):
    """Tables of a native seasonal decomposition with its portable saved state.

    ``to_json()`` returns the sealed state; ``oe.seasonal_restore`` rebuilds the
    identical result from it after replaying the decomposition.
    """

    def to_json(self) -> str:
        return json.dumps(self.attrs["seasonal_state"], sort_keys=True, allow_nan=False)


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            import operator

            value = operator.index(value)
        except TypeError:
            raise AnalysisError("invalid_option", f"{name} must be an integer in {low}..{high}.") from None
    if not low <= value <= high:
        raise AnalysisError("invalid_option", f"{name} must be an integer in {low}..{high}.")
    return int(value)


def natural_period(frequency: str) -> int | None:
    """Seasonal period implied by a pandas calendar frequency (None if not fixed)."""
    offset = pd.tseries.frequencies.to_offset(frequency)
    multiple = int(getattr(offset, "n", 1) or 1)
    name = type(offset).__name__
    per_year = None
    if name in {"MonthBegin", "MonthEnd", "BusinessMonthBegin", "BusinessMonthEnd"}:
        per_year = 12
    elif name in {"QuarterBegin", "QuarterEnd", "BQuarterBegin", "BQuarterEnd"}:
        per_year = 4
    elif name in {"YearBegin", "YearEnd", "BYearBegin", "BYearEnd"}:
        return 1
    if per_year is None:
        return None
    if multiple <= 0 or per_year % multiple:
        raise AnalysisError("invalid_time", f"Calendar frequency '{frequency}' does not divide a year "
                            "into a whole number of seasons.")
    return per_year // multiple


def load_seasonal_series(data: Any, y: str, time: str | None, period: Any, *,
                         what: str, allowed_periods: tuple[int, ...] | None = None,
                         calendar_periods: bool = False
                         ) -> tuple[Tensor, pd.Index, dict[str, Any] | None, int]:
    """(series, period labels, calendar record, seasonal period) after full validation."""
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", f"{what} needs a resident table; no Dataset "
                            "collection or streaming decomposition is performed.")
    series, labels = load_series(data, y, time)
    n = int(series.shape[0])
    if n > MAX_PERIODS:
        raise AnalysisError("system_budget", f"{what} admits at most {MAX_PERIODS} periods; got {n}.")
    index = pd.Index(labels.to_numpy() if time is not None else range(n))
    calendar = None
    implied = None
    if time is not None:
        calendar = _calendar(index if not pd.api.types.is_datetime64_any_dtype(index.dtype)
                             else pd.DatetimeIndex(index))
        if calendar["kind"] == "datetime":
            index = pd.DatetimeIndex(index)
            implied = natural_period(calendar["frequency"])
            if implied == 1:
                raise AnalysisError("not_seasonal", f"{what}: an annual calendar has no seasonal "
                                    "cycle within the year.")
            if implied is None and calendar_periods:
                raise AnalysisError("unsupported_calendar", f"{what} needs a monthly or quarterly "
                                    f"calendar; the {calendar['frequency']} calendar has no fixed "
                                    "number of seasons per year.")
    if period is None:
        if implied is None:
            raise AnalysisError("missing_period", f"{what} needs period (the number of observations "
                                "per seasonal cycle) unless the time column is a monthly or "
                                "quarterly datetime calendar.")
        period = implied
    period = _integer(period, "period", 2, MAX_PERIOD)
    if implied is not None and period != implied:
        raise AnalysisError("period_mismatch", f"period={period} contradicts the "
                            f"{calendar['frequency']} calendar, which has {implied} seasons per year.")
    if allowed_periods is not None and period not in allowed_periods:
        allowed = " or ".join(str(value) for value in allowed_periods)
        raise AnalysisError("unsupported_period", f"{what} supports period {allowed} only.")
    return series, index, calendar, period


def future_labels(index: pd.Index, calendar: dict[str, Any] | None, steps: int) -> list[Any]:
    """Period labels of the next ``steps`` periods after the sample."""
    if calendar is None:
        start = len(index)
        return list(range(start, start + steps))
    if calendar["kind"] == "datetime":
        future = pd.date_range(start=index[-1], periods=steps + 1, freq=calendar["frequency"])[1:]
        return list(future)
    step = calendar["step"]
    return [index[-1] + step * h for h in range(1, steps + 1)]


def state_record(method: str, settings: dict[str, Any], series: Tensor, index: pd.Index,
                 calendar: dict[str, Any] | None, components: dict[str, Tensor],
                 source: dict[str, Any]) -> dict[str, Any]:
    """The sealed portable state of a decomposition."""
    record = {
        "schema": STATE_SCHEMA,
        "method": method,
        "settings": settings,
        "source": source,
        "y": series.tolist(),
        "index": _index_record(index),
        "calendar": calendar,
        "components": {name: value.tolist() for name, value in components.items()},
    }
    _json_admission(record)
    record["sha256"] = _digest(record)
    return record


def load_state(value: Any) -> tuple[dict[str, Any], Tensor, pd.Index]:
    """Validated record, its series and period labels (no replay yet)."""
    try:
        if isinstance(value, SeasonalResult):
            value = value.attrs["seasonal_state"]
        record = _load_record(value)
        keys = {"schema", "method", "settings", "source", "y", "index", "calendar", "components",
                "sha256"}
        if set(record) != keys or record["schema"] != STATE_SCHEMA:
            raise ValueError("Unknown seasonal state schema")
        if record["sha256"] != _digest({k: v for k, v in record.items() if k != "sha256"}):
            raise ValueError("Seasonal state checksum mismatch")
        values = record["y"]
        if (not isinstance(values, list) or not 2 <= len(values) <= MAX_PERIODS
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError("Invalid stored series")
        n = len(values)
        plan_workspace("seasonal saved-state replay", {"series_and_components": n * 8 * 16})
        series = torch.tensor(values, dtype=torch.float64)
        index = _restore_index(record["index"], n)
        calendar = record["calendar"]
        if calendar is not None:
            if isinstance(index, pd.DatetimeIndex) or calendar.get("kind") == "datetime":
                index = pd.DatetimeIndex(index)
            if _calendar(index) != calendar:
                raise ValueError("Calendar does not reproduce the stored period labels")
        if (not isinstance(record["components"], dict) or not isinstance(record["settings"], dict)
                or not isinstance(record["source"], dict)):
            raise ValueError("Invalid seasonal state containers")
        return record, series, index
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        raise AnalysisError("invalid_state", "Saved seasonal decomposition state is not a valid, "
                            "unmodified record.") from error


def check_replay(record: dict[str, Any], components: dict[str, Tensor]) -> None:
    """The stored components must equal a fresh deterministic recomputation."""
    stored = record["components"]
    if set(stored) != set(components):
        raise AnalysisError("invalid_state", "Saved seasonal components do not match the method.")
    for name, value in components.items():
        saved = stored[name]
        if not isinstance(saved, list) or len(saved) != value.shape[0]:
            raise AnalysisError("invalid_state", f"Saved component '{name}' has the wrong length.")
        reference = torch.tensor([math.nan if v is None else v for v in saved], dtype=torch.float64)
        both_missing = torch.isnan(reference) & torch.isnan(value)
        scale = max(1.0, float(value[torch.isfinite(value)].abs().max()) if bool(torch.isfinite(value).any()) else 1.0)
        difference = (reference - value).abs()
        bad = ~both_missing & ~(difference <= REPLAY_TOLERANCE * scale)
        if bool(bad.any()):
            raise AnalysisError("invalid_state", f"Saved component '{name}' does not reproduce on "
                                "replay of the stored input and settings.")


def sample_contract(n: int) -> dict[str, Any]:
    """The declared sample/missing/weight/device domain of the resident seasonal routines."""
    return {"nobs": n, "missing": "refused; no observation is dropped or interpolated",
            "weights": "not supported", "categories": "not applicable",
            "device": "CPU float64 (resident table; no Dataset streaming)"}


def strength(component: Tensor, remainder: Tensor) -> float:
    """Wang, Smith and Hyndman (2006) strength max(0, 1 - Var(R) / Var(C + R))."""
    total = (component + remainder).var()
    if not bool(total > 0):
        return 0.0
    return max(0.0, 1.0 - float(remainder.var() / total))
