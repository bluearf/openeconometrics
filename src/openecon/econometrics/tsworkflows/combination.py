"""Forecast combination over supplied forecasts with identical origins and horizons.

Weights (Bates and Granger 1969, "The combination of forecasts", Operational Research
Quarterly 20; Timmermann 2006, Handbook of Economic Forecasting, ch. 4):

- ``"equal"``: w_k = 1 / K;
- ``"inverse_mse"``: w_k proportional to 1 / MSE_k, the mean squared past
  out-of-sample error of model k (Bates-Granger weights ignoring error correlation);
- ``"bates_granger"``: w = S^-1 1 / (1' S^-1 1) with S = E'E / T the uncentred
  second-moment matrix of the past errors E [T x K] (the variance-minimizing linear
  combination with weights summing to one; weights can be negative).

Past errors come from ``history``: a table with an ``actual`` column and one column
per model with that model's past forecasts of the same targets. The combined point
forecast is sum_k w_k f_k. No combined interval is produced: the joint distribution of
the component forecast errors is not available from saved forecast tables; the
in-history combined MSE is reported as a diagnostic.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace

METHODS = ("equal", "inverse_mse", "bates_granger")
MAX_MODELS = 64
MAX_HISTORY = 100_000


def _frame(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid_forecasts", f"Forecast '{name}' must be a table with period and "
                            "forecast columns.") from exc


def _finite(frame: pd.DataFrame, column: str, what: str) -> torch.Tensor:
    series = frame[column]
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise AnalysisError("invalid_forecasts", f"{what} must be numeric.")
    values = torch.tensor(series.to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("missing_values", f"{what} contains missing or non-finite values.")
    return values


@resident_cpu
def forecast_combination(forecasts: dict[str, Any], *, method: str = "equal", history: Any = None,
                         origins: dict[str, Any] | None = None) -> Any:
    """Combine point forecasts of several models for the same origin and periods.

    Parameters
    ----------
    forecasts : {name: table}, each with ``period`` and ``forecast`` columns (for
        example the tables of ``oe.forecast``, ``oe.stl_forecast`` or the ``forecast``
        table of ``oe.croston``). Every table must list exactly the same periods in the
        same order (``mismatched_forecasts`` otherwise); relabel step-numbered
        forecasts before combining forecasts from different calendars.
    method : ``"equal"``, ``"inverse_mse"`` or ``"bates_granger"``.
    history : for the estimated weights, a table with ``actual`` and one column per
        model name holding past out-of-sample forecasts of the same targets
        (rows with any missing value are refused, not dropped).
    origins : optional {name: forecast origin label}; all labels must be equal
        (``mismatched_origins`` otherwise). Supply it whenever the forecasts come from
        separately fitted models so that the common origin is checked explicitly.

    Returns
    -------
    Table with ``period``, ``forecast`` (combined) and ``forecast_<name>`` for every
    component; attrs: weights, method, past MSE per model and of the combination, the
    history size and notes.
    """
    if method not in METHODS:
        raise AnalysisError("invalid_option", f"method must be one of: {', '.join(METHODS)}.")
    if not isinstance(forecasts, dict) or not 2 <= len(forecasts) <= MAX_MODELS:
        raise AnalysisError("invalid_forecasts", f"forecasts must map 2..{MAX_MODELS} model names to tables.")
    names = list(forecasts)
    if any(not isinstance(name, str) or not name for name in names):
        raise AnalysisError("invalid_forecasts", "Model names must be nonempty strings.")
    if origins is not None:
        if not isinstance(origins, dict) or set(origins) != set(names):
            raise AnalysisError("mismatched_origins", "origins must give one origin label per model.")
        labels = {repr(origins[name]) for name in names}
        if len(labels) != 1:
            raise AnalysisError("mismatched_origins", "The forecasts have different origins: "
                                + ", ".join(f"{n}={origins[n]!r}" for n in names) + ".")
    frames = {name: _frame(forecasts[name], name) for name in names}
    periods = None
    columns = {}
    for name, frame in frames.items():
        if "period" not in frame.columns or "forecast" not in frame.columns:
            raise AnalysisError("invalid_forecasts", f"Forecast '{name}' lacks period/forecast columns.")
        if len(frame) == 0:
            raise AnalysisError("invalid_forecasts", f"Forecast '{name}' is empty.")
        labels = list(frame["period"])
        if periods is None:
            periods = labels
        elif labels != periods:
            raise AnalysisError("mismatched_forecasts", f"Forecast '{name}' covers different periods "
                                f"or horizons than '{names[0]}'; combine forecasts of the same "
                                "origin and periods only.")
        columns[name] = _finite(frame, "forecast", f"Forecast '{name}'")
    if len(set(map(repr, periods))) != len(periods):
        raise AnalysisError("mismatched_forecasts", "Forecast periods are repeated.")
    stacked = torch.stack([columns[name] for name in names], dim=1)               # [H, K]
    k = len(names)
    attrs: dict[str, Any] = {"method": method, "models": names}
    if method == "equal":
        if history is not None:
            raise AnalysisError("invalid_option", "Equal weights use no history; do not pass it.")
        weights = torch.full((k,), 1.0 / k, dtype=torch.float64)
    else:
        if history is None:
            raise AnalysisError("missing_history", f"method='{method}' needs history with actual "
                                "values and each model's past forecasts.")
        past = _frame(history, "history")
        missing = [c for c in ["actual", *names] if c not in past.columns]
        if missing:
            raise AnalysisError("missing_columns", f"history lacks columns: {', '.join(missing)}.")
        rows = len(past)
        if rows > MAX_HISTORY:
            raise AnalysisError("system_budget", f"history admits at most {MAX_HISTORY} rows.")
        plan_workspace("forecast combination history", {"errors": rows * (k + 2) * 8 * 4})
        actual = _finite(past, "actual", "history column 'actual'")
        errors = torch.stack([actual - _finite(past, name, f"history column '{name}'")
                              for name in names], dim=1)                          # [T, K]
        mse = errors.square().mean(0)
        if not bool((mse > 0).all()):
            raise AnalysisError("degenerate_history", "A model has zero past error; weights are undefined.")
        if method == "inverse_mse":
            if rows < 1:
                raise AnalysisError("insufficient_history", "history needs at least one row.")
            weights = (1.0 / mse) / (1.0 / mse).sum()
        else:
            if rows <= k:
                raise AnalysisError("insufficient_history", f"Bates-Granger weights need more history "
                                    f"rows than models ({k}); got {rows}.")
            second = errors.T @ errors / rows
            factor, info = torch.linalg.cholesky_ex(second)
            if int(info):
                raise AnalysisError("singular_error_covariance", "The past error second-moment "
                                    "matrix is singular (collinear forecast errors); drop a model "
                                    "or use inverse_mse.")
            ones = torch.ones((k, 1), dtype=torch.float64)
            solved = torch.cholesky_solve(ones, factor).flatten()
            weights = solved / solved.sum()
            attrs["condition_number"] = float(torch.linalg.cond(second))
        combined_errors = errors @ weights
        attrs.update(history_rows=rows, history_mse={n: float(v) for n, v in zip(names, mse, strict=True)},
                     history_combined_mse=float(combined_errors.square().mean()))
    combined = stacked @ weights
    data = {"period": periods, "forecast": combined.tolist()}
    for i, name in enumerate(names):
        data[f"forecast_{name}"] = stacked[:, i].tolist()
    return table(data, title=f"Forecast combination ({method})",
                 inference="point combination only: no combined interval or weight uncertainty",
                 weights={n: float(w) for n, w in zip(names, weights, strict=True)},
                 steps=len(periods), **attrs,
                 notes=["Point combination only: no combined interval is implied by the saved "
                        "component forecasts.",
                        "Weights are estimated from the supplied history and treated as fixed."
                        if method != "equal" else "Equal weights."])
