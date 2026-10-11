"""Rolling-origin backtests of the decomposition and intermittent-demand forecasters.

For every origin o (the first ``minimum`` observations, then every ``step`` periods)
each declared model is refitted on y_1..y_o only — the STL decomposition (and the
ETS/ARIMA model of its adjusted series) or the Croston-type smoothing, including any
estimated constants — and forecasts ``steps`` periods ahead. Rows are kept for the
targets with an observed actual. Observations after an origin never enter that
origin's fit. A model failure at an origin is recorded in ``attrs["failures"]`` with
its error code and leaves that forecast missing; nothing is refitted on other data.

The long table (origin, horizon, period, actual and one column per model) filtered to
one horizon is directly the ``history`` argument of ``oe.forecast_combination``.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import load_series
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.tsworkflows.seasonal_common import MAX_PERIODS, _integer, natural_period
from openecon.econometrics.tsworkflows.ssdiffuse import _calendar
from openecon.resources import plan_workspace

STL_KEYS = {"model", "forecast", "seasonal", "trend", "low_pass", "seasonal_degree", "trend_degree",
            "low_pass_degree", "seasonal_jump", "trend_jump", "low_pass_jump", "robust", "inner",
            "outer", "convention", "ets_model", "arima_order", "arima_constant"}
CROSTON_KEYS = {"model", "variant", "alpha", "beta", "loss"}
MAX_ORIGINS = 2000


def _check_models(models: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(models, dict) or not 1 <= len(models) <= 16:
        raise AnalysisError("invalid_option", "models must map 1..16 names to model declarations.")
    for name, spec in models.items():
        if not isinstance(name, str) or not name.isidentifier() or name in {"origin", "horizon", "period", "actual"}:
            raise AnalysisError("invalid_option", f"Model name {name!r} must be an identifier other "
                                "than origin/horizon/period/actual.")
        if not isinstance(spec, dict) or spec.get("model") not in ("stl", "croston"):
            raise AnalysisError("invalid_option", f"Model '{name}' needs model='stl' or 'croston'.")
        allowed = STL_KEYS if spec["model"] == "stl" else CROSTON_KEYS
        unknown = set(spec) - allowed
        if unknown:
            raise AnalysisError("invalid_option", f"Model '{name}' has unknown settings: "
                                f"{', '.join(sorted(unknown))}.")
    return models


def _forecast(spec: dict[str, Any], train: list[float], period: int | None, steps: int) -> list[float]:
    frame = {"y": train}
    if spec["model"] == "stl":
        from openecon.econometrics.tsworkflows.stl import stl, stl_forecast

        options = {k: v for k, v in spec.items() if k not in {"model", "forecast", "ets_model",
                                                               "arima_order", "arima_constant"}}
        fit = stl(frame, "y", period=period, **options)
        extra = {k: spec[k] for k in ("ets_model", "arima_order", "arima_constant") if k in spec}
        out = stl_forecast(fit, steps, method=spec.get("forecast", "naive"), **extra)
    else:
        from openecon.econometrics.tsworkflows.intermittent import croston

        options = {k: v for k, v in spec.items() if k not in {"model", "variant"}}
        out = croston(frame, "y", steps=steps, method=spec.get("variant", "croston"), **options)["forecast"]
    return [float(v) for v in out["forecast"]]


@resident_cpu
def forecast_backtest(data: Any, y: str, *, models: dict[str, dict[str, Any]], steps: int,
                      minimum: int, step: int = 1, time: str | None = None,
                      period: int | None = None, max_origins: int = 200) -> Any:
    """Rolling-origin backtest refitting every model within each origin.

    Parameters
    ----------
    data, y : table and the complete, regularly spaced series.
    models : {name: declaration}. ``{"model": "stl", "forecast": "naive" | "drift" | "ets"
        | "arima", <oe.stl options>, "ets_model", "arima_order", "arima_constant"}`` or
        ``{"model": "croston", "variant": "croston" | "sba" | "sbj" | "tsb", "alpha",
        "beta", "loss"}`` (constants may be "estimate": they are re-estimated per origin).
    steps : forecast horizon 1..120.
    minimum : size of the first training sample; origins are minimum, minimum + step, ...
        while an actual for horizon 1 exists.
    period : seasonal period for STL models (implied by a monthly/quarterly calendar).
    max_origins : bound on the number of origins (checked before fitting).

    Returns
    -------
    Long table: ``origin`` (label of the last training period), ``horizon``, ``period``
    (target label), ``actual`` and one forecast column per model. attrs: origins,
    per-model mean squared and absolute errors by horizon, failures (origin, model,
    error code), and the no-look-ahead contract.
    """
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "forecast_backtest needs a resident table.")
    models = _check_models(models)
    steps = _integer(steps, "steps", 1, 120)
    step = _integer(step, "step", 1, MAX_PERIODS)
    max_origins = _integer(max_origins, "max_origins", 1, MAX_ORIGINS)
    series, labels = load_series(data, y, time)
    n = int(series.shape[0])
    if n > MAX_PERIODS:
        raise AnalysisError("system_budget", f"forecast_backtest admits at most {MAX_PERIODS} periods.")
    minimum = _integer(minimum, "minimum", 2, max(2, n - 1))
    index = pd.Index(labels.to_numpy() if time is not None else range(n))
    if time is not None:
        if pd.api.types.is_datetime64_any_dtype(index.dtype):
            index = pd.DatetimeIndex(index)
        calendar = _calendar(index)
        implied = natural_period(calendar["frequency"]) if calendar["kind"] == "datetime" else None
        if period is None and implied not in (None, 1):
            period = implied
    if any(spec["model"] == "stl" for spec in models.values()) and period is None:
        raise AnalysisError("missing_period", "STL models need period (or a monthly/quarterly calendar).")
    origins = list(range(minimum, n, step))
    if len(origins) > max_origins:
        raise AnalysisError("search_budget", f"The backtest has {len(origins)} origins, exceeding "
                            f"max_origins={max_origins}.")
    plan_workspace("forecast backtest rows", {"rows": len(origins) * steps * (len(models) + 4) * 16})
    values = series.tolist()
    rows: list[list[Any]] = []
    failures: list[dict[str, Any]] = []
    for origin in origins:
        horizon = min(steps, n - origin)
        train = values[:origin]                         # no observation after the origin
        predictions = {}
        for name, spec in models.items():
            try:
                predictions[name] = _forecast(spec, train, period, horizon)
            except AnalysisError as exc:
                failures.append({"origin": index[origin - 1], "model": name, "code": exc.code,
                                 "message": str(exc)})
                predictions[name] = [math.nan] * horizon
        for h in range(1, horizon + 1):
            rows.append([index[origin - 1], h, index[origin + h - 1], values[origin + h - 1],
                         *[predictions[name][h - 1] for name in models]])
    columns = ["origin", "horizon", "period", "actual", *models]
    accuracy: dict[str, Any] = {}
    horizon_index = torch.tensor([row[1] for row in rows])
    actual = torch.tensor([row[3] for row in rows], dtype=torch.float64)
    for j, name in enumerate(models):
        error = actual - torch.tensor([row[4 + j] for row in rows], dtype=torch.float64)
        per = {}
        for h in range(1, steps + 1):
            chosen = (horizon_index == h) & torch.isfinite(error)
            if bool(chosen.any()):
                per[h] = {"mse": float(error[chosen].square().mean()),
                          "mae": float(error[chosen].abs().mean()), "count": int(chosen.sum())}
        accuracy[name] = per
    return table(rows, columns=columns, title=f"Rolling-origin backtest of {y}",
                 origins=len(origins), first_origin=index[minimum - 1], step=step, steps=steps,
                 models=models, period=period, accuracy=accuracy, failures=failures,
                 contract="each origin refits every decomposition, model and estimated constant "
                          "on the observations up to the origin only")
