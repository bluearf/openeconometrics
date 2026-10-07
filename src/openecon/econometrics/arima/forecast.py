"""Dynamic forecasts from a fitted ARIMA model (Stata: ``predict, dynamic()``; EViews: Forecast).

A fitted model is, for the differenced and regression-adjusted series
w_t = (1-L)^d (1-L^s)^D (y_t - x_t'b) - b_0,

    phi*(L) w_t = theta*(L) e_t        (expanded polynomials of degrees p*, q*).

Forecasts from the last observation n use the recursions

    w^_(n+h) = sum_i phi*_i w^_(n+h-i) + sum_(j>=h) theta*_j e^_(n+h-j),

with w^_t = w_t for t <= n and e^_t = E[e_t | y_1..y_n] for the last q* periods (the
stored end-of-sample state), are integrated back to levels with the last d + D*s
adjusted levels, and the regression part x_(n+h)'b of the supplied future
regressors is added. The forecast-error variance is

    MSE_h = sigma^2 [ sum_(j<h) Psi_j^2 + c_h' Omega c_h ],

where Psi(L) = theta*(L) / (phi*(L) (1-L)^d (1-L^s)^D) are the moving-average weights
of the integrated process and the second term carries the uncertainty Omega about the
last q* disturbances (zero for conditional estimation and negligible for an invertible
model on a long sample). Parameter uncertainty is not included, as in Stata's
``predict, mse`` and EViews.
"""

from __future__ import annotations

import operator
from importlib import import_module
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.estimators import (
    Orders, _likelihood, prepare, read_orders, state_record,
)
from openecon.econometrics.arima.filters import (
    difference_polynomial, inverse_filter, lag_polynomial, multiply,
)
from openecon.econometrics import registry
from openecon.econometrics.core import kernel_call, table
from openecon.engines.distributions import normal_isf
from openecon.models import ResultBundle

MAX_STEPS = 10_000


def _future_regressors(result: ResultBundle, terms: list[str], exog: Any, steps: int) -> Tensor:
    """[steps, len(terms)] design of the future regressors, coded as in the fit."""
    if exog is None:
        raise AnalysisError("missing_exog", "The model has regressors "
                            f"({', '.join(terms)}); pass their future values as exog, a table "
                            f"with {steps} row(s).")
    frame = _coerce_frame(exog)
    if len(frame) != steps:
        raise AnalysisError("invalid_exog", f"exog must have exactly one row per forecast step "
                            f"({steps}), got {len(frame)}.")
    encoding = result.provenance.get("categorical_encoding") or {}
    indicators = {f"{name}[{level}]": (name, level)
                  for name, record in encoding.items() for level in record["levels"][1:]}
    columns = []
    for term in terms:
        if term in indicators:
            name, level = indicators[term]
            if name not in frame.columns:
                raise AnalysisError("missing_columns", f"exog lacks the column '{name}'.")
            known = set(encoding[name]["levels"])
            if frame[name].isna().any() or not set(frame[name].tolist()) <= known:
                raise AnalysisError("invalid_exog", f"exog column '{name}' contains values that "
                                    "were not categories of the fitted model.")
            columns.append(torch.as_tensor((frame[name] == level).to_numpy(dtype="float64")))
        elif term in frame.columns:
            if frame[term].isna().any():
                raise AnalysisError("missing_values", f"exog column '{term}' contains missing "
                                    "values.")
            columns.append(_numeric(frame[term], term))
        else:
            raise AnalysisError("missing_columns", f"exog lacks the column '{term}'.")
    return torch.stack(columns, dim=1)


def _state_from_data(result: ResultBundle, orders: Orders, terms: list[str], theta: Tensor,
                     data: Any) -> dict[str, Any]:
    """Run the fitted model over new data and return its end-of-sample state."""
    prepared = prepare(result.spec, data)
    slopes = [term for term in terms if term != "Intercept"]
    fitted = result.provenance.get("categorical_encoding") or {}
    for name, record in prepared.design.categories.items():
        known = (fitted.get(name) or {}).get("levels") or []
        unseen = [level for level in record["levels"] if level not in known]
        if unseen:
            # An unknown level has no coefficient; it must not pass as the reference category.
            raise AnalysisError("invalid_data", f"Column '{name}' of the supplied data contains "
                                f"categories that were not in the fitted model: "
                                f"{', '.join(map(str, unseen[:5]))}.")
    absent = [term for term in slopes if term not in prepared.design.terms]
    if absent:
        raise AnalysisError("missing_columns", "The supplied data do not produce the regression "
                            f"terms of the fitted model: {', '.join(absent)}. Categorical "
                            "regressors need the reference category and every fitted category "
                            "present in the supplied data.")
    n = prepared.y.shape[0]
    if n < orders.state + 2:
        raise AnalysisError("insufficient_observations", "The supplied data are too short to "
                            "rebuild the model's state.")
    columns = [prepared.design.terms.index(term) for term in slopes]
    x = prepared.x[:, columns]
    if "Intercept" in terms:
        x = torch.cat([torch.ones((n, 1), dtype=torch.float64), x], dim=1)
    like = _likelihood(prepared, x.contiguous(), result.extra["method"])
    decomposition = kernel_call(like.decompose, theta, scores=False)
    return state_record(prepared, terms, x, theta, decomposition)


def _other_family_forecast(result: ResultBundle):
    """The forecast function another family registered for ``result`` (its ``FORECAST``)."""
    target = registry.forecasters().get(result.spec.estimator)
    if target is None or target == ("openecon.econometrics.arima.forecast", "forecast"):
        return None
    module, name = target
    try:
        return getattr(import_module(module), name, None)
    except ImportError:
        return None


def forecast(result: ResultBundle, steps: int, *, data: Any = None, exog: Any = None,
             alpha: float = 0.05) -> pd.DataFrame:
    """Dynamic multi-step forecasts from a fitted ``oe.arima`` model.

    ``oe.forecast(result, steps, ...)`` dispatches here for ``arima`` results; the
    same function is exported as ``oe.arima_forecast``. A result of another
    time-series family that registered a forecast function is passed on to it.

    Parameters
    ----------
    result : the ResultBundle returned by ``oe.arima``.
    steps : number of periods to forecast beyond the last observation (1..10000).
    data : optional table. By default the forecast starts at the end of the
        estimation sample, from the state stored in ``result.extra['last_state']``.
        With ``data`` the fitted model (parameters fixed) is run over that table,
        which must contain the model's columns, and the forecast starts at its
        end; use this to forecast from a sample updated with new observations.
    exog : future values of the regressors, one row per step, when the model has
        regressors (same columns and categories as in the fit).
    alpha : the intervals have coverage ``1 - alpha``.

    Returns
    -------
    Table with one row per step: ``period`` (the integer time value continued
    from the sample, or the step number 1..steps when the time column is absent or
    a datetime), ``forecast`` (the outcome in levels: differences are integrated
    and the regression part added), ``std_error`` (root mean squared forecast
    error, disturbance uncertainty only), ``ci_low``, ``ci_high`` (normal
    intervals). ``attrs`` records the model, ``sigma``, ``alpha`` and the
    forecast origin.

    The forecasts are the minimum mean-squared-error predictions of the Gaussian
    ARIMA model: the AR recursion on past values and on the estimated last
    disturbances, with future disturbances at zero. Parameter uncertainty is
    ignored, as in Stata's ``predict, dynamic() mse`` and EViews.

    Stata: ``predict yhat, y dynamic(tq(1990q4))`` after ``tsappend``. EViews:
    Forecast (dynamic).

    Example
    -------
    >>> fit = oe.arima(data=df, y="sales", order=(0, 1, 1), seasonal=(0, 1, 1), period=12)
    >>> oe.forecast(fit, steps=24)
    """
    if not isinstance(result, ResultBundle):
        raise AnalysisError("invalid_result", "forecast needs a result returned by oe.arima.")
    if result.spec.estimator != "arima":
        other = _other_family_forecast(result)
        if other is None:
            raise AnalysisError("invalid_result", "forecast needs a result returned by oe.arima; "
                                f"this is a {result.spec.estimator} result.")
        options = {name: value for name, value in (("data", data), ("exog", exog))
                   if value is not None}
        return other(result, steps, alpha=alpha, **options)
    if not isinstance(steps, (bool, float, str, bytes)):
        try:
            steps = operator.index(steps)          # NumPy integers are integers too
        except TypeError:
            pass
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= MAX_STEPS:
        raise AnalysisError("invalid_steps", f"steps must be an integer between 1 and {MAX_STEPS}.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise AnalysisError("invalid_option", "alpha must be a number strictly between 0 and 1.")
    orders = read_orders(result.spec)
    names = [c.term for c in result.coefficients]
    theta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    k = len(names) - orders.n_arma - 1
    terms = names[:k]
    sigma = float(theta[-1])
    extra = result.extra
    if data is None:
        state, origin = extra["last_state"], "end of the estimation sample"
    else:
        state = _state_from_data(result, orders, terms, theta, data)
        origin = "end of the supplied data"

    s = max(orders.period, 1)
    ar_full = multiply(lag_polynomial(extra["ar"], sign=-1.0),
                       lag_polynomial(extra["seasonal"]["ar"], unit=s, sign=-1.0))
    ma_full = multiply(lag_polynomial(extra["ma"]),
                       lag_polynomial(extra["seasonal"]["ma"], unit=s))
    delta = difference_polynomial(orders.d, orders.seasonal_d, orders.period)
    p_full, q_full = len(ar_full) - 1, len(ma_full) - 1

    # Point forecasts of the stationary ARMA disturbance u (a loop over forecast steps only).
    process = [float(value) for value in state["process"]]
    disturbances = [float(value) for value in state["disturbances"]]
    for h in range(1, steps + 1):
        value = -sum(ar_full[i] * process[-i] for i in range(1, min(p_full, len(process)) + 1))
        value += sum(ma_full[j] * disturbances[q_full - 1 - (j - h)]
                     for j in range(h, q_full + 1))
        process.append(value)
    drift = float(theta[terms.index("Intercept")]) if "Intercept" in terms else 0.0
    levels = [float(value) for value in state["levels"]]
    base = len(levels)
    for h in range(steps):
        value = drift + process[len(process) - steps + h]
        value -= sum(delta[j] * levels[-j] for j in range(1, min(len(delta) - 1, len(levels)) + 1))
        levels.append(value)
    point = torch.tensor(levels[base:], dtype=torch.float64)
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    if slopes:
        future = _future_regressors(result, [terms[i] for i in slopes], exog, steps)
        point = point + future @ theta[slopes]
    elif exog is not None:
        raise AnalysisError("invalid_exog", "The model has no regressors; do not pass exog.")

    # Forecast-error variance: MA(infinity) weights of the integrated process.
    def integrate(numerator: Tensor) -> Tensor:
        """numerator / (phi(L) Phi(L^s) (1-L)^d (1-L^s)^D), one factor at a time."""
        out = inverse_filter(numerator, [-value for value in extra["ar"]])
        out = inverse_filter(out, [-value for value in extra["seasonal"]["ar"]], s)
        for _ in range(orders.d):
            out = inverse_filter(out, [-1.0])
        for _ in range(orders.seasonal_d):
            out = inverse_filter(out, [-1.0], s)
        return out

    with torch.no_grad():
        impulse = torch.zeros((1, steps), dtype=torch.float64)
        impulse[0, :min(steps, q_full + 1)] = torch.tensor(ma_full[:steps], dtype=torch.float64)
        weights = integrate(impulse)[0]
        mse = weights.square().cumsum(dim=0)
        omega = state.get("disturbance_covariance")
        if omega is not None and q_full:
            omega = torch.tensor(omega, dtype=torch.float64)
            load = torch.zeros((q_full, steps), dtype=torch.float64)
            for index in range(q_full):
                for h in range(1, min(index + 1, steps) + 1):
                    load[index, h - 1] = ma_full[h + q_full - 1 - index]
            load = integrate(load)
            mse = mse + (load * (omega @ load)).sum(dim=0)
        std_error = sigma * mse.clamp_min(0.0).sqrt()
    if not bool(torch.isfinite(point).all()) or not bool(torch.isfinite(std_error).all()):
        raise AnalysisError("non_finite_result", "The forecasts are not finite; the fitted model "
                            "is explosive over this horizon.")
    critical = normal_isf(alpha / 2.0)
    last = state.get("last_period")
    periods = list(range(1, steps + 1)) if last is None else [last + h for h in range(1, steps + 1)]
    return table(
        {"period": periods, "forecast": point.tolist(), "std_error": std_error.tolist(),
         "ci_low": (point - critical * std_error).tolist(),
         "ci_high": (point + critical * std_error).tolist()},
        title=f"Dynamic forecasts of {result.spec.outcome}: {orders.label()}",
        model=orders.label(), outcome=result.spec.outcome, steps=steps, alpha=alpha,
        confidence_level=1.0 - alpha, origin=origin,
        period_unit="time value" if last is not None else "steps ahead",
        std_error_definition="disturbance uncertainty only (no parameter uncertainty)",
        sigma=sigma)
