"""Auditable bounded ARIMA selection and generic rolling/recursive refits."""

from __future__ import annotations
import itertools
import math
from typing import Any
import torch
from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame,
    make_spec,
    TableSet,
    table,
    forecast,
    column_list,
    _frame_hasher,
    _position_bytes,
)
from openecon.econometrics.arima.estimators import arima, prepare
from openecon.econometrics.unitroot.series import kpss
from openecon.econometrics.tsworkflows.common import ordered
from openecon.models import ModelSpec


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise AnalysisError("invalid_option", f"{name} must be an integer in {low}..{high}.")
    return value


def auto_arima(
    *,
    data: Any,
    y: str,
    x=None,
    time=None,
    d=None,
    seasonal_d=0,
    period=None,
    max_p=2,
    max_q=2,
    max_P=0,
    max_Q=0,
    max_d=2,
    criterion="aicc",
    constant="auto",
    max_candidates=81,
    covariance="nonrobust",
    missing="raise",
    max_iterations=200,
    tolerance=1e-8,
    alpha=0.05,
):
    """Exact-ML exhaustive bounded order search, retaining the selected ARIMA result.

    Differencing is fixed before likelihood comparisons: explicit ``d`` or repeated
    level-KPSS at 5% (no exogenous regressors in automatic-d mode). Seasonal D is
    explicit. All candidates have exactly the same retained response positions.
    No CSS approximation, stepwise/global-optimum claim, or comparison across d.
    Every candidate failure and criterion/parameter count is recorded in ``extra``.
    """
    max_candidates = _integer(max_candidates, "max_candidates", 1, 256)
    bounds = [
        _integer(v, name, 0, 8)
        for v, name in zip(
            (max_p, max_q, max_P, max_Q), ("max_p", "max_q", "max_P", "max_Q"), strict=True
        )
    ]
    max_d = _integer(max_d, "max_d", 0, 2)
    seasonal_d = _integer(seasonal_d, "seasonal_d", 0, 2)
    if criterion not in {"aic", "aicc", "bic"} or not (
        isinstance(constant, bool) or constant == "auto"
    ):
        raise AnalysisError(
            "invalid_option", "criterion must be aic/aicc/bic and constant True/False/'auto'."
        )
    if period is None and (seasonal_d or max_P or max_Q):
        raise AnalysisError("invalid_option", "Seasonal search needs period.")
    if period is not None:
        _integer(period, "period", 2, 365)
    x = column_list(x, "x")
    base = make_spec(
        "arima",
        outcome=y,
        predictors=x,
        time=time,
        missing=missing,
        covariance=covariance,
        options={"order": [0, 0, 0]},
    )
    common = ModelFrame(base, data)
    ordered(common)
    common.workspace_plan(
        "auto ARIMA retained data and selection records",
        {"candidate_records": max_candidates * 4096, "response": common.n * 8},
    )
    sample = common.sample.copy()
    checks = []
    if d is None:
        if x:
            raise AnalysisError(
                "invalid_option",
                "With regressors choose d explicitly; marginal outcome KPSS is not a regression residual test.",
            )
        series = common.numeric(y)
        d = 0
        while True:
            test = kpss({y: series.tolist()}, y, auto=True)
            statistic = float(test.attrs["statistic"])
            reject = statistic > 0.463
            checks.append(
                {
                    "d": d,
                    "statistic": statistic,
                    "critical_5pct": 0.463,
                    "lags": test.attrs["lags"],
                    "reject_stationarity": reject,
                }
            )
            if not reject:
                break
            if d == max_d:
                raise AnalysisError(
                    "differencing_limit",
                    "KPSS still rejects stationarity at max_d; choose a defensible d explicitly.",
                )
            series = torch.diff(series)
            d += 1
    else:
        d = _integer(d, "d", 0, 3)
    constants = (
        [False, True]
        if constant == "auto" and d <= 1
        else [False if constant == "auto" else constant]
    )
    count = math.prod(v + 1 for v in bounds) * len(constants)
    if count > max_candidates:
        raise AnalysisError(
            "search_budget",
            f"Search needs {count} fits, exceeding max_candidates={max_candidates}.",
        )
    records, best, best_score, positions = [], None, math.inf, None
    for p, q, P, Q, include in itertools.product(*(range(v + 1) for v in bounds), constants):
        row = {
            "order": [p, d, q],
            "seasonal": [P, seasonal_d, Q] if period else None,
            "constant": include,
            "status": "failed",
        }
        try:
            result = arima(
                data=sample,
                y=y,
                x=x,
                time=time,
                order=(p, d, q),
                seasonal=(P, seasonal_d, Q) if period else None,
                period=period,
                constant=include,
                covariance=covariance,
                max_iterations=max_iterations,
                tolerance=tolerance,
                alpha=alpha,
                missing=missing,
            )
            if positions is None:
                positions = result.sample_positions
            if result.sample_positions != positions:
                raise AnalysisError(
                    "incomparable_sample", "Candidates returned different response samples."
                )
            k, n = len(result.coefficients), result.nobs
            aic = float(result.metrics["aic"])
            aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else None
            score = {"aic": aic, "aicc": aicc, "bic": result.metrics["bic"]}[criterion]
            if score is None:
                raise AnalysisError("insufficient_aicc_sample", "AICc is undefined when N <= K+1.")
            row.update(
                status="ok",
                aic=aic,
                aicc=aicc,
                bic=result.metrics["bic"],
                log_likelihood=result.metrics["log_likelihood"],
                nobs=n,
                parameters=k,
                criterion=float(score),
            )
            if float(score) < best_score:
                best, best_score = result, float(score)
        except AnalysisError as exc:
            if exc.code == "incomparable_sample":
                raise
            row.update(error_code=exc.code, error=str(exc))
        records.append(row)
    if best is None:
        error = AnalysisError(
            "no_valid_model",
            f"All {count} ARIMA candidates failed; inspect error.candidates for the complete search record.",
        )
        error.candidates = records
        raise error
    best.extra["auto_selection"] = {
        "search": "exhaustive bounded exact ML",
        "criterion": criterion,
        "criterion_value": best_score,
        "candidate_count": count,
        "candidates": records,
        "differencing": {
            "d": d,
            "D": seasonal_d,
            "method": "KPSS level 5%" if checks else "explicit",
            "tests": checks,
        },
        "common_input_positions": common.positions,
        "common_response_positions": [common.positions[i] for i in positions],
        "selection_uncertainty": "coefficient inference is conditional on selected order; selection uncertainty excluded",
    }
    best.sample_positions = [common.positions[i] for i in best.sample_positions]
    for row in best.predictions:
        row["row"] = common.positions[row["row"]]
    best.nobs_original = len(common.original)
    best.dropped_rows = best.nobs_original - best.nobs
    lineage = prepare(best.spec, common.original).frame
    input_hasher = _frame_hasher(lineage.original)
    sample_hasher = _frame_hasher(lineage.sample)
    sample_hasher.update(_position_bytes(lineage.positions))
    best.provenance.update(
        data_hash=input_hasher.hexdigest(),
        sample_hash=sample_hasher.hexdigest(),
        input_columns=list(lineage.original.columns),
    )
    best.provenance["workflow"] = "auto_arima; selected underlying ARIMA specification retained"
    best.warnings.append("Inference and forecast intervals condition on the selected ARIMA order.")
    return best


def rolling(
    spec: ModelSpec,
    *,
    data: Any,
    window: int,
    step=1,
    expanding=False,
    forecast_steps=0,
    max_fits=100,
    on_error="raise",
    exog=None,
    reverse=False,
    calendar=None,
    reverse_recursive=False,
    selection=None,
    evaluate=False,
):
    """Refit an existing ModelSpec over explicit regular-calendar windows.

    ``window`` is minimum training size; expanding=True retains the first row.
    Optional forecasts call the existing family handler. Future regressors must
    be supplied separately in ``exog`` keyed by integer origin (never read from
    the future estimation table). Failures are either raised or explicit records.
    Reverse windows traverse fixed windows backward or expand anchored suffixes;
    these are retrospective and reject forward forecasting/evaluation. Bounded
    ARIMA selection is rerun on each training window. evaluate=True records
    observed future outcomes only after fitting and forecasting, with physical
    target positions and per-horizon errors. Panel calendars use rolling_panel.
    """
    if not isinstance(spec, ModelSpec) or any(
        not isinstance(flag, bool) for flag in (expanding, reverse, evaluate)
    ):
        raise AnalysisError("invalid_spec", "rolling needs ModelSpec and a boolean expanding flag.")
    if reverse and (forecast_steps or evaluate):
        raise AnalysisError(
            "unsupported_target",
            "Reverse windows are retrospective; forward forecast/evaluation is not available.",
        )
    if evaluate and (not forecast_steps or spec.estimator not in {"arima", "ets", "ucm", "sspace"}):
        raise AnalysisError(
            "unsupported_target",
            "Evaluation needs forward forecasts of one arima/ets/ucm/sspace response.",
        )
    if selection is not None:
        allowed = {
            "d",
            "seasonal_d",
            "period",
            "max_p",
            "max_q",
            "max_P",
            "max_Q",
            "max_d",
            "criterion",
            "constant",
            "max_candidates",
            "max_iterations",
            "tolerance",
        }
        if spec.estimator == "ets":
            allowed = {"models", "period", "criterion", "candidate_options", "max_candidates", "max_work", "max_iterations", "tolerance"}
        if spec.estimator not in {"arima", "ets"} or not isinstance(selection, dict) or set(selection) - allowed:
            raise AnalysisError(
                "invalid_option",
                "selection must contain bounded auto_arima or auto_ets options for the matching ARIMA/ETS spec.",
            )
    if not isinstance(reverse_recursive, bool) or calendar is not None and not isinstance(calendar, bool):
        raise AnalysisError("invalid_option", "calendar and reverse_recursive need boolean values.")
    if isinstance(spec, ModelSpec) and (spec.panel or calendar is not None or reverse_recursive):
        if selection is not None or evaluate:
            raise AnalysisError("unsupported_target", "Calendar/panel/reverse-recursive analysis does not accept ARIMA selection or forecast evaluation; use the original scalar workflow.")
        from .window_targets import advanced_rolling
        return advanced_rolling(
            spec, data=data, window=window, step=step, expanding=expanding,
            reverse=reverse or reverse_recursive,
            calendar=bool(spec.panel) if calendar is None else calendar,
            forecast_steps=forecast_steps, max_fits=max_fits, on_error=on_error, exog=exog,
        )
    extra_columns = []
    common_spec = spec
    if spec.estimator in {"midas", "umidas"}:
        from openecon.econometrics.tsworkflows.midas import _alignment
        from openecon.analysis import _coerce_frame
        from openecon.econometrics.registry import role_columns

        aligned = _coerce_frame(data)
        if spec.estimator == "umidas":
            from openecon.econometrics.tsworkflows.umidas import bound_alignment
            bound_alignment(spec.options["alignment"], aligned, role_columns(spec, "lags"))
        else:
            _alignment(spec.options["alignment"], aligned, role_columns(spec, "lags"))
        extra_columns = spec.options["alignment"]["binding_columns"]
        if spec.time is None:
            common_spec = ModelSpec.model_validate(
                {**spec.model_dump(mode="json"), "time": spec.options["alignment"]["low_time"]}
            )
    common = ModelFrame(common_spec, data, extra_columns=extra_columns)
    if spec.panel:
        raise AnalysisError(
            "unsupported_panel",
            "Rolling origins are for one ordered series; split panel entities explicitly.",
        )
    ordered(common)
    window = _integer(window, "window", 2, common.n)
    step = _integer(step, "step", 1, common.n)
    forecast_steps = _integer(forecast_steps, "forecast_steps", 0, 10000)
    max_fits = _integer(max_fits, "max_fits", 1, 1000)
    if on_error not in {"raise", "record"}:
        raise AnalysisError("invalid_option", "on_error must be raise or record.")
    windows = (
        [
            (start, common.n if expanding else start + window)
            for start in range(common.n - window, -1, -step)
        ]
        if reverse
        else [
            (0 if expanding else stop - window, stop) for stop in range(window, common.n + 1, step)
        ]
    )
    if len(windows) > max_fits:
        raise AnalysisError(
            "search_budget", f"Rolling requires {len(windows)} fits; increase max_fits explicitly."
        )
    common.workspace_plan(
        "rolling summaries", {"results": len(windows) * (16384 + forecast_steps * 128)}
    )
    estimates, forecasts, history, evaluation = [], [], [], []
    for start, stop in windows:
        origin = start if reverse else stop - 1
        rows = common.sample.iloc[start:stop].copy()
        window_spec = spec
        if spec.estimator in {"midas", "umidas"}:
            from openecon.econometrics.tsworkflows.midas import _hash

            positions = common.positions[start:stop]
            alignment = spec.options["alignment"]
            selected = {
                **alignment,
                "rows": [alignment["rows"][i] for i in positions],
                "original_positions": [alignment["original_positions"][i] for i in positions],
                "window_input_positions": positions,
                "parent_data_hash": alignment["data_hash"],
                "data_hash": _hash(rows, alignment["binding_columns"]),
            }
            window_spec = ModelSpec.model_validate(
                {**spec.model_dump(mode="json"), "options": {**spec.options, "alignment": selected}}
            )
        record = {
            "origin": origin,
            "start": start,
            "stop_exclusive": stop,
            "input_positions": common.positions[start:stop],
            "status": "ok",
        }
        try:
            if selection is not None and spec.estimator == "ets":
                from openecon.econometrics.tsworkflows.selection import auto_ets

                ets_selection = {"period": spec.options.get("period", 2), **selection}
                result = auto_ets(data=rows, y=spec.outcome, time=spec.time,
                                  missing="raise", alpha=spec.alpha, **ets_selection)
            else:
                result = (
                    fit(window_spec, data=rows)
                    if selection is None
                    else auto_arima(
                        data=rows,
                        y=spec.outcome,
                        x=spec.predictors,
                        time=spec.time,
                        covariance=spec.covariance,
                        missing="raise",
                        alpha=spec.alpha,
                        **selection,
                    )
                )
            record.update(
                nobs=result.nobs,
                sample_positions=[common.positions[start + i] for i in result.sample_positions],
                result_id=result.id,
                convergence=result.provenance.get("optimizer"),
                coefficients=[coef.model_dump(mode="json") for coef in result.coefficients],
                covariance_matrix=result.covariance_matrix,
                inference=result.inference,
                metrics=result.metrics,
                spec=result.spec.model_dump(mode="json"),
                auto_selection=result.extra.get("auto_selection"),
            )
            if result.extra.get("target") == "prediction":
                raise AnalysisError(
                    "unsupported_target",
                    "Rolling coefficient inference does not synthesize parameters for prediction-only models; use an explicit predictive validation workflow.",
                )
            for coef in result.coefficients:
                estimates.append(
                    {
                        "origin": origin,
                        "term": coef.term,
                        "estimate": coef.estimate,
                        "std_error": coef.std_error,
                        "ci_low": coef.ci_low,
                        "ci_high": coef.ci_high,
                    }
                )
            if forecast_steps:
                options = {} if exog is None or origin not in exog else {"exog": exog[origin]}
                future = forecast(result, forecast_steps, **options)
                future_rows = future.to_dict("records")
                if evaluate and len(future_rows) != forecast_steps:
                    raise AnalysisError(
                        "unsupported_target",
                        "Evaluation requires exactly one forecast per horizon.",
                    )
                for horizon, row in enumerate(future_rows, 1):
                    forecasts.append({"origin": origin, **row})
                    if evaluate and stop + horizon - 1 < common.n:
                        predicted = row.get("mean", row.get("forecast"))
                        if predicted is None:
                            raise AnalysisError(
                                "unsupported_target",
                                "Forecast output needs a mean or forecast column.",
                            )
                        observed = float(common.sample[spec.outcome].iloc[stop + horizon - 1])
                        error = observed - float(predicted)
                        evaluation.append(
                            {
                                "origin": origin,
                                "horizon": horizon,
                                "target_position": common.positions[stop + horizon - 1],
                                "observed": observed,
                                "forecast": float(predicted),
                                "error": error,
                                "squared_error": error**2,
                                "absolute_error": abs(error),
                            }
                        )
        except AnalysisError as exc:
            if on_error == "raise":
                raise
            record.update(status="failed", error_code=exc.code, error=str(exc))
        history.append(record)
    return TableSet(
        {
            "coefficients": table(estimates),
            "forecasts": table(forecasts),
            "evaluation": table(evaluation),
        },
        title="Recursive refits" if expanding else "Rolling refits",
        spec=spec.model_dump(mode="json"),
        window=window,
        step=step,
        expanding=expanding,
        reverse=reverse,
        selection=selection,
        origins=history,
        lookahead=(
            "retrospective reverse windows; no forward evaluation"
            if reverse
            else "estimation and differencing/order selection end at origin; future outcomes only enter scoring; supplied future exog is explicitly conditional"
        ),
        sample_positions=common.positions,
    )


def recursive(spec, *, data, minimum, **options):
    """Expanding-window alias of rolling; minimum is the first training length."""
    if "expanding" in options:
        raise AnalysisError("invalid_option", "recursive already sets expanding=True.")
    return rolling(spec, data=data, window=minimum, expanding=True, **options)
