"""Multi-step mean and conditional-variance forecasts from a fitted ARCH-family model.

From the last observation n the recursions of the fitted model are continued with
future innovations replaced by their expectations given the sample:

    variance state   v^_(n+k) = w_(n+k) + sum a_i N1_(n+k-i) + sum g_i N2_(n+k-i)
                                + sum b_j v^_(n+k-j)
    mean             u^_(n+k) = sum rho_j u^_(n+k-j) + sum theta_k e_(n+k-k')      (future e = 0)
                     y^_(n+k) = x_(n+k)'b + psi g(h^_(n+k)) + u^_(n+k)

where observed periods use the actual innovations and variances and future periods use
E[e^2] = h^, E[e^2 1(e<0)] = h^/2 (GARCH, GJR, IGARCH: the recursion then yields the
exact conditional expectation E_n[h_(n+k)]), E[z] = 0 and E|z| - sqrt(2/pi) (EGARCH) or
E|e|^phi = E|z|^phi s^ (power ARCH). For EGARCH and power ARCH the reported variance
is the transformation of the forecast state, exp(E_n[ln h]) and (E_n[s])^(2/phi): exact
one step ahead, and beyond that the usual plug-in forecast (Stata's dynamic ``predict,
variance``), which is not the conditional expectation of h itself.

The root mean squared error of the mean forecast uses the moving-average weights psi_j
of the ARMA disturbance, MSE_k = sum_(j<k) psi_j^2 h^_(n+k-j); parameter uncertainty and
the feedback of variance uncertainty through an ARCH-in-mean term are ignored.
"""

from __future__ import annotations

import math
import operator
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arch import densities
from openecon.econometrics.arch.estimators import prepare
from openecon.econometrics.arch.kernels import ArchLikelihood
from openecon.econometrics.arch.layout import Layout, read_layout
from openecon.econometrics.arch.postfit import state_record
from openecon.econometrics.core import table
from openecon.models import ResultBundle

MAX_STEPS = 10_000


def _layout(result: ResultBundle) -> Layout:
    model = result.extra.get("model") or {}
    return read_layout(result.spec, int(model.get("mean_terms", 0)),
                       int(model.get("variance_regressors", 0)))


def _future_design(result: ResultBundle, terms: list[str], prefix: str, exog: Any,
                   steps: int) -> Tensor:
    """``[steps, len(terms)]`` design of future regressors, coded as in the fit."""
    if not terms:
        return torch.empty((steps, 0), dtype=torch.float64)
    names = [term.removeprefix(prefix) for term in terms]
    if exog is None:
        raise AnalysisError("missing_exog", "The model has regressors "
                            f"({', '.join(names)}); pass their future values as exog, a table "
                            f"with {steps} row(s).")
    frame = _coerce_frame(exog)
    if len(frame) != steps:
        raise AnalysisError("invalid_exog", "exog must have exactly one row per forecast step "
                            f"({steps}), got {len(frame)}.")
    encoding = result.provenance.get("categorical_encoding") or {}
    indicators = {f"{name}[{level}]": (name, level)
                  for name, record in encoding.items() for level in record["levels"][1:]}
    columns = []
    for name in names:
        if name in indicators:
            column, level = indicators[name]
            if column not in frame.columns:
                raise AnalysisError("missing_columns", f"exog lacks the column '{column}'.")
            known = set(encoding[column]["levels"])
            if frame[column].isna().any() or not set(frame[column].tolist()) <= known:
                raise AnalysisError("invalid_exog", f"exog column '{column}' contains values "
                                    "that were not categories of the fitted model.")
            columns.append(torch.as_tensor((frame[column] == level).to_numpy(dtype="float64")))
        elif name in frame.columns:
            if frame[name].isna().any():
                raise AnalysisError("missing_values", f"exog column '{name}' contains missing "
                                    "values.")
            columns.append(_numeric(frame[name], name))
        else:
            raise AnalysisError("missing_columns", f"exog lacks the column '{name}'.")
    return torch.stack(columns, dim=1)


def _state_from_data(result: ResultBundle, layout: Layout, terms: list[str], theta: Tensor,
                     data: Any) -> dict[str, Any]:
    """Run the fitted model (parameters fixed) over new data; return its end-of-sample state."""
    prepared = prepare(result.spec, data, screen=False)
    ix = layout.index
    x_terms = [terms[i] for i in ix.x]
    z_terms = [terms[i] for i in ix.z]
    absent = [term for term in x_terms if term not in prepared.design.terms] \
        + [term for term in z_terms if term not in prepared.z_terms]
    if absent:
        raise AnalysisError("missing_columns", "The supplied data do not produce the terms of "
                            f"the fitted model: {', '.join(absent)}. Categorical regressors need "
                            "the reference category and every fitted category present.")
    if prepared.frame.n < layout.max_lag + 1:
        raise AnalysisError("insufficient_observations", "The supplied data are too short to "
                            "rebuild the model's state.")
    x = prepared.design.x[:, [prepared.design.terms.index(term) for term in x_terms]]
    z = prepared.z[:, [prepared.z_terms.index(term) for term in z_terms]]
    like = ArchLikelihood(prepared.y, x.contiguous(), z.contiguous(), layout)
    evaluation = like.evaluate(theta, derivatives=False)
    if evaluation is None:
        raise AnalysisError("invalid_data", "The fitted model does not define a positive, finite "
                            "conditional variance on the supplied data.")
    return state_record(layout, evaluation, prepared.last_period)


def _check_state(state: Any, layout: Layout) -> None:
    """The end-of-sample state must hold ``max_lag`` finite innovations and variances."""
    problem = AnalysisError(
        "invalid_result", "The result does not carry the end-of-sample state of the fitted "
        "model (extra['state']); fit the model again with oe.arch, or pass data= to rebuild "
        "the state.")
    if not isinstance(state, dict):
        raise problem
    series = [state.get(name) for name in ("residuals", "disturbances", "variances")]
    if any(not isinstance(values, list) or len(values) < layout.max_lag
           or any(isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) for value in values) for values in series):
        raise problem
    if any(not value > 0.0 for value in series[2]):
        raise problem


def arch_forecast(result: ResultBundle, steps: int, *, data: Any = None, exog: Any = None,
                  alpha: float = 0.05) -> pd.DataFrame:
    """Multi-step forecasts of the mean and the conditional variance after ``oe.arch``.

    ``oe.forecast(result, steps, ...)`` dispatches here for ``arch`` results.

    Parameters
    ----------
    result : the ResultBundle returned by ``oe.arch``.
    steps : number of periods beyond the last observation (1..10000).
    data : optional table. By default the forecast starts at the end of the estimation
        sample, from ``result.extra['state']``. With ``data`` the fitted model (parameters
        fixed) is run over that table, which must contain the model's columns, and the
        forecast starts at its end: use it to forecast from a sample extended with new
        observations.
    exog : future values of the regressors of the mean and of the variance equation, one
        row per step, when the model has any (same columns and categories as in the fit).
    alpha : the intervals of the mean forecast have coverage ``1 - alpha``.

    Returns
    -------
    Table with one row per step: ``period`` (the integer time value continued from the
    sample, or the step number when the time column is absent or a datetime),
    ``mean_forecast``, ``variance_forecast`` (the conditional variance h), ``std_error``
    (root mean squared error of the mean forecast), ``ci_low`` and ``ci_high``. ``attrs``
    records the model, the origin and the conventions.

    Variance forecasts continue the fitted recursion with future squared innovations at
    their expectation: for GARCH, GJR and IGARCH models
    ``h^_(n+k) = omega + sum (a_i + g_i/2) h^_(n+k-i) + sum b_j h^_(n+k-j)`` once all lags
    are in the future, which converges to the unconditional variance at the rate of the
    persistence (and grows linearly for IGARCH with a positive constant). EGARCH and
    power-ARCH forecasts beyond one step are the plug-in forecasts exp(E[ln h]) and
    (E[h^(phi/2)])^(2/phi). The mean forecast adds the regression part, the ARCH-in-mean
    term at the forecast variance and the ARMA forecast of the disturbance. Its
    ``std_error`` is ``sqrt(sum_(j<k) psi_j^2 h^_(n+k-j))`` with the MA(infinity) weights of
    the ARMA part (simply ``sqrt(h^_(n+k))`` without ARMA terms); the interval uses the
    quantile of the fitted innovation distribution and is exact one step ahead.
    Parameter uncertainty is ignored, as in Stata's ``predict`` and EViews.

    Stata: ``tsappend, add(k)`` then ``predict v, variance dynamic(...)`` and
    ``predict yhat, y dynamic(...)``. EViews: Forecast.

    Example
    -------
    >>> fit = oe.arch(data=df, y="ret", arch=1, garch=1)
    >>> oe.arch_forecast(fit, steps=20)[["period", "variance_forecast"]]
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "arch":
        raise AnalysisError("invalid_result", "arch_forecast needs a result returned by oe.arch.")
    if not isinstance(steps, (bool, float, str, bytes)):
        try:
            steps = operator.index(steps)
        except TypeError:
            pass
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= MAX_STEPS:
        raise AnalysisError("invalid_steps", f"steps must be an integer between 1 and {MAX_STEPS}.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise AnalysisError("invalid_option", "alpha must be a number strictly between 0 and 1.")
    layout = _layout(result)
    ix, kind = layout.index, layout.kind
    terms = [c.term for c in result.coefficients]
    theta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    if len(terms) != ix.k:
        raise AnalysisError("invalid_result", "The result does not match its specification.")
    if data is None:
        state, origin = result.extra.get("state"), "end of the estimation sample"
    else:
        state, origin = _state_from_data(result, layout, terms, theta, data), \
            "end of the supplied data"
    _check_state(state, layout)

    slopes = [i for i in ix.x if terms[i] != "Intercept"]
    future_x = _future_design(result, [terms[i] for i in slopes], "", exog, steps)
    future_z = _future_design(result, [terms[i] for i in ix.z], "HET:", exog, steps)
    if exog is not None and not slopes and not ix.z:
        raise AnalysisError("invalid_exog", "The model has no regressors; do not pass exog.")
    regression = future_x @ theta[slopes] if slopes else torch.zeros(steps, dtype=torch.float64)
    constant = [i for i in ix.x if terms[i] == "Intercept"]
    if constant:
        regression = regression + theta[constant[0]]
    if ix.z:
        linear = float(theta[ix.c]) + future_z @ theta[ix.z]
        omega = (linear if kind == "egarch" else linear.exp()).tolist()
    else:
        omega = [float(theta[ix.c])] * steps

    tau = float(theta[ix.d[0]]) if ix.d else 0.0
    phi = float(theta[ix.p[0]]) if ix.p else 2.0
    psi = float(theta[ix.m[0]]) if ix.m else 0.0
    a, b = theta[ix.a].tolist(), theta[ix.b].tolist()
    g = theta[ix.g].tolist() if ix.g else [0.0] * len(a)
    rho, tma = theta[ix.ar].tolist(), theta[ix.ma].tolist()
    residuals = [float(value) for value in state["residuals"]]
    disturbances = [float(value) for value in state["disturbances"]]
    variances = [float(value) for value in state["variances"]]
    if kind == "egarch":
        standardized = [e / math.sqrt(h) for e, h in zip(residuals, variances)]
        news1 = standardized
        news2 = [abs(z) - densities.ABS_NORMAL for z in standardized]
        states = [math.log(h) for h in variances]
        expected2 = densities.abs_moment(layout.dist, 1.0, tau) - densities.ABS_NORMAL
    elif kind == "parch":
        news1, news2 = [abs(e) ** phi for e in residuals], []
        states = [h ** (0.5 * phi) for h in variances]
        moment = densities.abs_moment(layout.dist, phi, tau)
        if not math.isfinite(moment):
            raise AnalysisError("non_finite_result", "E|z|^power does not exist for the fitted "
                                "Student t distribution (power >= degrees of freedom), so "
                                "multi-step variance forecasts are undefined.")
    else:
        news1 = [e * e for e in residuals]
        news2 = [e * e if e < 0.0 else 0.0 for e in residuals]
        states = list(variances)

    variance, mean = [], []
    try:
        for step in range(steps):
            v = omega[step]
            for value, lag in zip(a, layout.arch_lags):
                v += value * news1[-lag]
            if layout.asymmetric:
                for value, lag in zip(g, layout.arch_lags):
                    v += value * news2[-lag]
            for value, lag in zip(b, layout.garch_lags):
                v += value * states[-lag]
            if kind == "egarch":
                h = math.exp(v)
                news1.append(0.0)
                news2.append(expected2)
            elif kind == "parch":
                if not v > 0.0:
                    raise ValueError
                h = v ** (2.0 / phi)
                news1.append(moment * v)
            else:
                if not v > 0.0:
                    raise ValueError
                h = v
                news1.append(h)
                news2.append(0.5 * h)
            states.append(v)
            u = sum(value * disturbances[-lag] for value, lag in zip(rho, layout.ar_lags)) \
                + sum(value * residuals[-lag] for value, lag in zip(tma, layout.ma_lags))
            disturbances.append(u)
            residuals.append(0.0)
            in_mean = 0.0
            if layout.archm is not None:
                in_mean = psi * {"variance": h, "sd": math.sqrt(h), "log": math.log(h)}[
                    layout.archm]
            variance.append(h)
            mean.append(u + in_mean)
    except (ValueError, OverflowError):
        raise AnalysisError("non_finite_result", "The variance forecasts are not positive and "
                            "finite; the fitted variance equation is explosive or leaves its "
                            "domain over this horizon.") from None
    variance_t = torch.tensor(variance, dtype=torch.float64)
    point = regression + torch.tensor(mean, dtype=torch.float64)

    # Root MSE of the mean forecast: MA(infinity) weights of the ARMA disturbance.
    if layout.ar_lags or layout.ma_lags:
        weights = [1.0]
        ma_at, ar_at = dict(zip(layout.ma_lags, tma)), dict(zip(layout.ar_lags, rho))
        for j in range(1, steps):
            weights.append(ma_at.get(j, 0.0)
                           + sum(value * weights[j - lag] for lag, value in ar_at.items()
                                 if lag <= j))
        squared = torch.tensor(weights, dtype=torch.float64).square()
        kernel = squared.flip(0)[None, None, :]
        padded = torch.nn.functional.pad(variance_t[None, None, :], (steps - 1, 0))
        mse = torch.nn.functional.conv1d(padded, kernel)[0, 0]
    else:
        mse = variance_t
    std_error = mse.clamp_min(0.0).sqrt()
    if not all(bool(torch.isfinite(values).all()) for values in (point, variance_t, std_error)):
        raise AnalysisError("non_finite_result", "The forecasts are not finite; the fitted model "
                            "is explosive over this horizon.")
    critical = densities.critical_value(layout.dist, alpha, tau)
    last = state.get("last_period")
    periods = list(range(1, steps + 1)) if last is None else [last + k for k in range(1, steps + 1)]
    plug_in = kind in {"egarch", "parch"}
    return table(
        {"period": periods, "mean_forecast": point.tolist(),
         "variance_forecast": variance_t.tolist(), "std_error": std_error.tolist(),
         "ci_low": (point - critical * std_error).tolist(),
         "ci_high": (point + critical * std_error).tolist()},
        title=f"Forecasts of {result.spec.outcome}: {layout.label()}",
        model=layout.label(), outcome=result.spec.outcome, steps=steps, alpha=alpha,
        confidence_level=1.0 - alpha, origin=origin,
        period_unit="time value" if last is not None else "steps ahead",
        variance_forecast_definition=(
            "transformation of the forecast variance state (exp(E[ln h]) or E[s]^(2/power)); "
            "exact one step ahead" if plug_in else
            "conditional expectation of h given the sample"),
        std_error_definition="innovation uncertainty only (no parameter uncertainty)",
        interval_distribution=layout.dist, critical_value=critical)
