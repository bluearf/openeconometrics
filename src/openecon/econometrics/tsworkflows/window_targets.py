"""Calendar panel/reverse analysis and isolated forward predictive evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from openecon.analysis import fit, _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, table
from openecon.models import ModelSpec
from openecon.resources import plan_workspace


def _timeline(spec, data, time, panel, calendar):
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported",
            "Window workflows need in-memory data; Dataset is not collected.",
        )
    raw = _coerce_frame(data)
    if raw.columns.has_duplicates or not len(raw):
        raise AnalysisError("invalid_data", "Window inputs must be nonempty with unique columns.")
    plan_workspace(
        "window timeline",
        {"input_copy_and_order": int(raw.memory_usage(deep=True).sum()) * 3 + len(raw) * 64},
    )
    if panel and (not time or not calendar):
        raise AnalysisError(
            "invalid_time",
            "Pooled panel windows require time and calendar=True integer-period spans.",
        )
    keys = [key for key in (time, panel) if key]
    if keys:
        if any(key not in raw for key in keys) or raw[keys].isna().any().any():
            raise AnalysisError(
                "invalid_time",
                "Time/panel columns must be present and complete, including excluded model rows.",
            )
        if raw.duplicated(keys).any():
            raise AnalysisError(
                "repeated_time_values", "Time is repeated within a panel or single series."
            )
    if calendar and not time:
        raise AnalysisError("invalid_time", "Calendar windows need an explicit time column.")
    if time:
        values = raw[time]
        if pd.api.types.is_bool_dtype(values.dtype) or pd.api.types.is_complex_dtype(values.dtype):
            raise AnalysisError("invalid_time", "Time ordering cannot use boolean or complex values.")
        if calendar:
            if (
                not pd.api.types.is_numeric_dtype(values.dtype)
                or pd.api.types.is_bool_dtype(values.dtype)
                or pd.api.types.is_complex_dtype(values.dtype)
            ):
                raise AnalysisError(
                    "invalid_time",
                    "Calendar windows require integer periods; convert dates explicitly without filling gaps.",
                )
            if not np.isfinite(values.to_numpy()).all() or not (values == values.round()).all():
                raise AnalysisError("invalid_time", "Calendar periods must be finite integers.")
            if values.min() < np.iinfo(np.int64).min or values.max() >= np.iinfo(np.int64).max:
                raise AnalysisError(
                    "invalid_time", "Calendar periods exceed the supported int64 range."
                )
        elif not pd.api.types.is_numeric_dtype(
            values.dtype
        ) and not pd.api.types.is_datetime64_any_dtype(values.dtype):
            raise AnalysisError(
                "invalid_time", "Physical windows require numeric or datetime time ordering."
            )
        elif (
            pd.api.types.is_numeric_dtype(values.dtype) and not np.isfinite(values.to_numpy()).all()
        ):
            raise AnalysisError("invalid_time", "Time must be finite.")
        # Index labels need not be unique: order original physical positions instead.
        ordering = (
            pd.DataFrame({key: raw[key].to_numpy() for key in keys})
            .sort_values(keys, kind="stable")
            .index.to_numpy()
        )
    else:
        ordering = np.arange(len(raw))
    rows = raw.iloc[ordering].copy().reset_index(drop=True)
    times = rows[time].to_numpy(dtype=np.int64) if calendar else np.arange(len(rows))
    return rows, ordering.tolist(), times


def _windows(times, window, step, expanding, reverse, max_fits, horizon=0):
    from .workflows import _integer

    first, last = int(times.min()), int(times.max()) - horizon
    span = last - first + 1
    window = _integer(window, "window", 2, span)
    step = _integer(step, "step", 1, span)
    max_fits = _integer(max_fits, "max_fits", 1, 1000)
    if reverse and expanding:
        raise AnalysisError(
            "invalid_option",
            "reverse already selects reverse recursive windows; expanding must be False.",
        )
    count = (span - window) // step + 1
    if count > max_fits:
        raise AnalysisError(
            "search_budget", f"Window plan requires {count} fits; increase max_fits explicitly."
        )
    if reverse:
        return [(start, last) for start in range(first, last - window + 2, step)]
    return [
        (first if expanding else stop - window + 1, stop)
        for stop in range(first + window - 1, last + 1, step)
    ]


def advanced_rolling(
    spec,
    *,
    data,
    window,
    step,
    expanding,
    reverse,
    calendar,
    forecast_steps,
    max_fits,
    on_error,
    exog,
):
    from .workflows import _integer

    _integer(forecast_steps, "forecast_steps", 0, 10000)
    if forecast_steps or exog is not None:
        raise AnalysisError(
            "unsupported_forecast",
            "Panel/calendar/reverse analysis does not imply causal forecasting; use the forward predictive workflow or original scalar rolling forecasts.",
        )
    if on_error not in {"raise", "record"}:
        raise AnalysisError("invalid_option", "on_error must be raise or record.")
    if spec.estimator in {"midas", "umidas"}:
        raise AnalysisError(
            "unsupported_alignment",
            "Calendar/reverse MIDAS alignment is not supported; use recorded forward MIDAS windows.",
        )
    calendar = bool(spec.panel) if calendar is None else calendar
    rows, positions, times = _timeline(spec, data, spec.time, spec.panel, calendar)
    windows = _windows(times, window, step, expanding, reverse, max_fits)
    plan_workspace(
        "window result states",
        {"results": len(windows) * (16384 + len(rows) * (128 + len(spec.predictors) * 32))},
    )
    estimates, history = [], []
    for start, stop in windows:
        mask = (times >= start) & (times <= stop)
        selected = np.flatnonzero(mask).tolist()
        input_positions = [positions[i] for i in selected]
        record = {
            "origin": stop,
            "start": start,
            "stop_exclusive": stop + 1,
            "input_positions": input_positions,
            "status": "ok",
        }
        try:
            result = fit(spec, data=rows.iloc[selected].copy())
            if result.extra.get("target") == "prediction":
                raise AnalysisError(
                    "unsupported_target", "Prediction-only fits require rolling_predict."
                )
            record.update(
                result_id=result.id,
                nobs=result.nobs,
                sample_positions=[input_positions[i] for i in result.sample_positions],
                coefficients=[c.model_dump(mode="json") for c in result.coefficients],
                covariance_matrix=result.covariance_matrix,
                inference=result.inference,
                metrics=result.metrics,
                spec=result.spec.model_dump(mode="json"),
                result=result.model_dump(mode="json"),
                convergence=result.provenance.get("optimizer"),
            )
            for coef in result.coefficients:
                estimates.append(
                    {
                        "origin": stop,
                        "start": start,
                        "term": coef.term,
                        "estimate": coef.estimate,
                        "std_error": coef.std_error,
                        "ci_low": coef.ci_low,
                        "ci_high": coef.ci_high,
                    }
                )
        except AnalysisError as exc:
            if on_error == "raise":
                raise
            record.update(status="failed", error_code=exc.code, error=str(exc))
        history.append(record)
    return TableSet(
        {"coefficients": table(estimates), "forecasts": table([])},
        title="Reverse recursive retrospective refits"
        if reverse
        else "Calendar panel refits"
        if spec.panel
        else "Calendar refits",
        spec=spec.model_dump(mode="json"),
        window=window,
        step=step,
        expanding=expanding,
        reverse=reverse,
        origins=history,
        sample_positions=positions,
        window_units="integer calendar periods" if calendar else "physical ordered rows",
        missing="screened independently within each window; calendar gaps never compressed",
        panel_scope="pooled entities with declared estimator panel roles" if spec.panel else None,
        lookahead="retrospective fixed final cutoff; no causal forecast claim"
        if reverse
        else "fit rows at or before cutoff",
    )


def rolling_predict(
    spec,
    *,
    data,
    time,
    window,
    step=1,
    horizon=1,
    expanding=False,
    panel=None,
    calendar=False,
    max_fits=100,
    on_error="raise",
):
    """Fit/tune on each past window, then evaluate the next disjoint time span.

    Numeric calendar periods preserve gaps; physical mode orders rows by time.
    Evaluation labels never enter fitting, tuning or preprocessing. Overlapping
    evaluations are recorded individually rather than called independent tests.
    """
    from .workflows import _integer
    from openecon.econometrics.regularized.prediction import regularized_predict

    if not isinstance(spec, ModelSpec) or spec.estimator not in {
        "ridge",
        "lasso",
        "elasticnet",
        "pls",
        "kernelreg",
        "localreg",
    }:
        raise AnalysisError(
            "unsupported_target",
            "rolling_predict supports declared scalar regularized/PLS/local prediction targets.",
        )
    if (
        not isinstance(expanding, bool)
        or not isinstance(calendar, bool)
        or on_error not in {"raise", "record"}
    ):
        raise AnalysisError("invalid_option", "Invalid expanding/calendar/on_error option.")
    if spec.options.get("query") is not None:
        raise AnalysisError(
            "invalid_option",
            "Window prediction determines evaluation queries; an external query option is not permitted.",
        )
    rows, positions, times = _timeline(spec, data, time, panel, calendar)
    horizon = _integer(horizon, "horizon", 1, 10000)
    windows = _windows(times, window, step, expanding, False, max_fits, horizon=horizon)
    if not windows:
        raise AnalysisError(
            "insufficient_observations", "No complete training/evaluation window exists."
        )
    width = len(spec.predictors)
    plan_workspace(
        "saved predictive windows",
        {"states_and_outputs": len(windows) * (32768 + len(rows) * (256 + width * 64))},
    )
    predictions, history = [], []
    for start, stop in windows:
        training = np.flatnonzero((times >= start) & (times <= stop)).tolist()
        evaluation = np.flatnonzero((times > stop) & (times <= stop + horizon)).tolist()
        record = {
            "origin": stop,
            "start": start,
            "evaluation_stop": stop + horizon,
            "training_input_positions": [positions[i] for i in training],
            "evaluation_input_positions": [positions[i] for i in evaluation],
            "status": "ok",
        }
        try:
            result = fit(spec, data=rows.iloc[training].copy())
            frame = ModelFrame(spec, rows.iloc[evaluation].copy())
            prediction = regularized_predict(result, frame.sample)
            observed = frame.numeric(spec.outcome).numpy()
            values = prediction.to_numpy()
            mse = float(np.mean((observed - values) ** 2))
            if not np.isfinite(mse):
                raise AnalysisError("numerical_failure", "Held-out error is not finite.")
            evaluated = [positions[evaluation[i]] for i in frame.positions]
            for pos, actual, predicted in zip(evaluated, observed, values, strict=True):
                predictions.append(
                    {
                        "origin": stop,
                        "row": pos,
                        "observed": float(actual),
                        "predicted": float(predicted),
                        "error": float(actual - predicted),
                    }
                )
            record.update(
                result_id=result.id,
                result=result.model_dump(mode="json"),
                training_sample_positions=[positions[training[i]] for i in result.sample_positions],
                evaluation_sample_positions=evaluated,
                n_evaluated=len(evaluated),
                mse=mse,
                evaluation_warnings=frame.warnings,
            )
        except AnalysisError as exc:
            if on_error == "raise":
                raise
            record.update(status="failed", error_code=exc.code, error=str(exc))
        history.append(record)
    return TableSet(
        {"predictions": table(predictions)},
        title="Forward predictive window evaluation",
        spec=spec.model_dump(mode="json"),
        origins=history,
        time=time,
        panel=panel,
        window=window,
        step=step,
        horizon=horizon,
        expanding=expanding,
        window_units="integer calendar periods" if calendar else "physical ordered rows",
        lookahead="training fit/scaling/tuning sees only rows through origin; next-span labels used only for scoring",
        inference="prediction target; no coefficient/pointwise CI or independent-error claim",
        sample_positions=positions,
    )
