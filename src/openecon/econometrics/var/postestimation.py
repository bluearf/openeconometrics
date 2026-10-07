"""Post-estimation for ``oe.var`` and ``oe.vec`` results: impulse responses and forecasts.

``irf`` rebuilds the autoregressive matrices, the innovation covariance and the
coefficient covariance from the stored result and returns impulse responses and
the forecast-error variance decomposition as a long table (Stata: ``irf create``
followed by ``irf table``; EViews: impulse responses / variance decomposition).

``var_forecast`` and ``vec_forecast`` compute dynamic forecasts of the levels
(Stata: ``fcast compute``). ``oe.forecast(result, steps)`` dispatches to them.
The formulas are in ``impulse.py`` and in the docstrings below.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.econometrics.var import impulse, kernels
from openecon.econometrics.var.common import (
    check_alpha, check_steps, load_sample, require_result,
)
from openecon.engines.distributions import normal_isf
from openecon.engines.linalg import cholesky_inverse
from openecon.models import ResultBundle

MAX_IRF_STEPS = 500
MAX_FORECAST_STEPS = 5000


def _tensor(value: Any) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _reject_unknown(function: str, unknown: dict[str, Any], allowed: str) -> None:
    """``oe.forecast`` forwards any keyword: an unknown one is an AnalysisError, not a TypeError."""
    if unknown:
        raise AnalysisError("invalid_option", f"{function} does not take the option(s) "
                            f"{', '.join(sorted(unknown))}; it accepts {allowed}.")


def _system(result: ResultBundle) -> dict[str, Any]:
    """Autoregressive matrices, innovation covariance and bookkeeping of a fitted model."""
    if result.spec.estimator == "vec":
        record = result.extra["var_representation"]
        a = _tensor(record["A"])
        return {"names": result.extra["layout"]["variables"], "a": a,
                "sigma": _tensor(result.extra["omega"]), "cov_alpha": None, "n_obs": None}
    layout = result.extra["layout"]
    names, p, m = layout["variables"], layout["lags"], layout["n_regressors"]
    k = len(names)
    coef = _tensor([c.estimate for c in result.coefficients]).reshape(k, m)
    positions = impulse.alpha_positions(k, p, m)
    covariance = _tensor(result.covariance_matrix)
    return {"names": names, "a": kernels.lag_matrices(coef, k, p), "coef": coef,
            "sigma": _tensor(result.extra["sigma"]), "covariance": covariance,
            "cov_alpha": covariance[positions][:, positions], "n_obs": int(result.metrics["T"])}


def irf(result: ResultBundle, steps: int = 8, kind: str = "orthogonalized", *,
        alpha: float = 0.05) -> pd.DataFrame:
    """Impulse responses and variance decomposition of a fitted VAR or VEC model.

    With Phi_i the moving-average matrices of the model (Phi_0 = I,
    Phi_i = sum_j Phi_(i-j) A_j) and Sigma = P P' the Cholesky factorization of
    the innovation covariance in the order of the variables:

    - ``kind="simple"``: Phi_i, the response to a unit shock in one innovation.
    - ``kind="orthogonalized"`` (default): Theta_i = Phi_i P, the response to a
      one-standard-deviation orthogonalized shock; the table also carries the
      forecast-error variance decomposition ``fevd``, the share of the
      step-ahead forecast-error variance of the response variable due to the
      impulse variable (zero at step 0, as in Stata's ``irf table``).
    - ``kind="generalized"``: Psi_i = Phi_i Sigma diag(Sigma)^(-1/2) (Pesaran and
      Shin 1998), which does not depend on the ordering.

    Standard errors (``std_error``, ``fevd_std_error``) are asymptotic,
    by the delta method of Lutkepohl (2005, section 3.7), from the coefficient
    covariance of the fit (whatever ``covariance=`` was used) and
    ``(2/T) D+ (Sigma (x) Sigma) D+'`` for vech(Sigma). The intervals are
    normal-based at level ``alpha``. For ``oe.vec`` results the responses come
    from the VAR representation of the VEC model and have no standard errors
    (as in Stata).

    Parameters
    ----------
    result : the result of ``oe.var`` or ``oe.vec``.
    steps : horizon (responses for steps 0..steps); at most 500.
    kind : ``"orthogonalized"``, ``"simple"`` or ``"generalized"``.
    alpha : level of the confidence intervals.

    Returns a table with one row per (impulse, response, step): ``impulse``,
    ``response``, ``step``, ``irf``, ``std_error``, ``ci_low``, ``ci_high`` and,
    for the orthogonalized kind, ``fevd`` and ``fevd_std_error``.

    Stata: ``irf create m1, step(8) set(myirf)`` then ``irf table oirf fevd``.

    Example
    -------
    >>> result = oe.var(data=frame, y=["income", "consumption"], lags=2)
    >>> oe.irf(result, steps=10, kind="orthogonalized")
    """
    result = require_result(result, ("var", "vec"), "irf")
    steps = check_steps(steps, MAX_IRF_STEPS)
    alpha = check_alpha(alpha)
    if kind not in impulse.KINDS:
        raise AnalysisError("invalid_option", f"kind must be one of {', '.join(impulse.KINDS)}.")
    system = _system(result)
    names = system["names"]
    k = len(names)
    responses = kernel_call(impulse.impulse_responses, system["a"], system["sigma"], steps,
                            cov_alpha=system["cov_alpha"], n_obs=system["n_obs"])
    values = getattr(responses, kind)
    errors = None if responses.se is None else responses.se[kind]
    critical = normal_isf(alpha / 2.0)
    rows = []
    for c in range(k):
        for r in range(k):
            for step in range(steps + 1):
                value = float(values[step, r, c])
                error = None if errors is None else float(errors[step, r, c])
                row = [names[c], names[r], step, value, error,
                       None if error is None else value - critical * error,
                       None if error is None else value + critical * error]
                if kind == "orthogonalized":
                    row += [float(responses.fevd[step, r, c]),
                            None if responses.se is None
                            else float(responses.se["fevd"][step, r, c])]
                rows.append(row)
    columns = ["impulse", "response", "step", "irf", "std_error", "ci_low", "ci_high"]
    if kind == "orthogonalized":
        columns += ["fevd", "fevd_std_error"]
    note = None
    if responses.se is None:
        note = ("standard errors are not available for vec results"
                if result.spec.estimator == "vec" else
                "standard errors were not computed for a system of this size")
    return table(rows, columns=columns, kind=kind, steps=steps, variables=names, alpha=alpha,
                 estimator=result.spec.estimator, standard_errors=note or "delta method",
                 ordering="Cholesky order = order of the variables" if kind == "orthogonalized"
                 else None)


def _future_exogenous(result: ResultBundle, terms: list[str], exog: Any, steps: int) -> Tensor:
    """[steps, len(terms)] future values of the exogenous terms, coded as in the fit."""
    if not terms:
        if exog is not None:
            raise AnalysisError("invalid_exog", "The model has no exogenous regressors; do not "
                                "pass exog.")
        return torch.empty((steps, 0), dtype=torch.float64)
    if exog is None:
        raise AnalysisError("missing_exog", "The model has exogenous regressors "
                            f"({', '.join(terms)}); pass their future values as exog, a table "
                            f"with {steps} row(s).")
    frame = _coerce_frame(exog)
    if len(frame) != steps:
        raise AnalysisError("invalid_exog", "exog must have exactly one row per forecast step "
                            f"({steps}), got {len(frame)}.")
    encoding = result.provenance.get("categorical_encoding") or {}
    indicators = {f"{name}[{level}]": (name, level)
                  for name, record in encoding.items() for level in record["levels"][1:]}
    columns = []
    for term in terms:
        name, level = indicators.get(term, (term, None))
        if name not in frame.columns:
            raise AnalysisError("missing_columns", f"exog lacks the column '{name}'.")
        if frame[name].isna().any():
            raise AnalysisError("missing_values", f"exog column '{name}' contains missing values.")
        if term in indicators:
            if not set(frame[name].tolist()) <= set(encoding[name]["levels"]):
                raise AnalysisError("invalid_exog", f"exog column '{name}' contains values that "
                                    "were not categories of the fitted model.")
            columns.append(torch.as_tensor((frame[name] == level).to_numpy(dtype="float64")))
        else:
            columns.append(_numeric(frame[name], name))
    return torch.stack(columns, dim=1)


def _state(result: ResultBundle, data: Any, p: int) -> tuple[Tensor, int | None, int]:
    """Last p levels (chronological), the last period and the trend index of the last row."""
    record = result.extra["forecast"]
    if data is None:
        return (_tensor(record["last_values"]).reshape(p, -1), record["last_period"],
                int(record["last_trend"]))
    sample = load_sample(result.spec, data)
    n = sample.levels.shape[0]
    if n < p:
        raise AnalysisError("insufficient_observations", f"The supplied data must hold at least "
                            f"{p} observation(s) to start the forecast.")
    trend = n
    if sample.last_period is not None and record["last_period"] is not None:
        trend = int(record["last_trend"]) + sample.last_period - int(record["last_period"])
    return sample.levels[n - p:], sample.last_period, trend


def _forecast_table(names: list[str], mean: Tensor, mse: Tensor, last_period: int | None,
                    alpha: float, **attrs: Any) -> pd.DataFrame:
    steps = mean.shape[0]
    errors = mse.diagonal(dim1=1, dim2=2).clamp_min(0.0).sqrt()
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(errors).all()):
        raise AnalysisError("non_finite_result", "The forecasts overflow: the model is explosive "
                            "over this horizon. Reduce the number of steps.")
    critical = normal_isf(alpha / 2.0)
    rows = []
    for v, name in enumerate(names):
        for h in range(steps):
            value, error = float(mean[h, v]), float(errors[h, v])
            period = [] if last_period is None else [last_period + h + 1]
            rows.append([name, h + 1, *period, value, error, value - critical * error,
                         value + critical * error])
    columns = ["variable", "step", *([] if last_period is None else ["period"]), "forecast",
               "std_error", "ci_low", "ci_high"]
    return table(rows, columns=columns, steps=steps, alpha=alpha, variables=names, **attrs)


def _parameter_term(result: ResultBundle, system: dict[str, Any], moment: Tensor,
                    phi: Tensor) -> Tensor | None:
    """Coefficient-uncertainty part of the forecast MSE, or None when it cannot be formed.

    The computation runs in the mean-deviated regressor coordinates of the fit
    (Zc = Z C, C = [[I, 0], [-zbar', 1]]), in which the stored moment matrix is
    well conditioned: coefficients b_c = C^-1 b, transition B_c = C' B C'^-1. The
    conventional covariance is rebuilt exactly as Sigma (x) (Zc'Zc)^-1; a robust
    covariance is rotated from the stored one, which is only reliable while the
    level of the series does not swamp float64 (checked; otherwise None).
    """
    layout = result.extra["layout"]
    names, p, m = system["names"], layout["lags"], layout["n_regressors"]
    k = len(names)
    t = system["n_obs"]
    change = torch.eye(m, dtype=torch.float64)
    back = torch.eye(m, dtype=torch.float64)              # C^-1, known in closed form
    means = result.extra["forecast"].get("means")
    if means is not None:
        change[-1, :-1] = -_tensor(means)
        back[-1, :-1] = _tensor(means)
    transition = torch.zeros((m, m), dtype=torch.float64)
    slots = torch.arange(k) * p
    transition[slots] = system["coef"]
    for j in range(1, p):
        transition[slots + j, slots + j - 1] = 1.0
    if layout["constant"]:
        transition[m - 1, m - 1] = 1.0
    transition = change.T @ transition @ back.T
    try:
        if result.spec.covariance == "nonrobust":
            scale = t / (t - m) if layout["small"] or layout["dfk"] else 1.0
            inverse = kernel_call(cholesky_inverse, moment) / t
            covariance = torch.kron(_tensor(result.extra["sigma_ml"]) * scale,
                                    inverse.contiguous())
        else:
            stored = system["covariance"]
            covariance = torch.einsum("la,iajb,mb->iljm", back, stored.reshape(k, m, k, m),
                                      back).reshape(k * m, k * m)
            lost = torch.finfo(torch.float64).eps * float(stored.abs().max())
            if lost > 1e-6 * float(covariance.diagonal().abs().min()):
                return None
        return kernel_call(impulse.parameter_mse, transition, moment, covariance, phi, k)
    except AnalysisError:
        return None


def var_forecast(result: ResultBundle, steps: int, *, data: Any = None, exog: Any = None,
                 alpha: float = 0.05, **unknown: Any) -> pd.DataFrame:
    """Dynamic forecasts of the levels from a fitted ``oe.var`` model (Stata ``fcast compute``).

    From the last observation T the h-step forecast is

        y_T(h) = v + d (T + h) + A_1 y_T(h-1) + ... + A_p y_T(h-p) + B x_(T+h),

    with y_T(j) = y_(T+j) for j <= 0. The forecast-error covariance is

        Sigma_y^(h) = sum_(i<h) Phi_i Sigma Phi_i'  +  (1/T) Omega(h),

    the innovation part plus, for models without exogenous regressors or
    trend, Lutkepohl's (2005, 3.5) estimation-uncertainty term

        (1/T) Omega(h) = (1/T) sum_t G_t V G_t',
        G_t = sum_(i<h) Z_t' (B')^(h-1-i) (x) Phi_i,

    averaged over the sample regressor vectors Z_t with V the coefficient
    covariance (this is Stata's default). ``Sigma`` is the model's innovation
    covariance (``extra["sigma"]``). With exogenous regressors or a trend only
    the innovation part is reported (Stata reports no standard errors then);
    ``attrs["mse_formula"]`` says which formula was used. Intervals are normal-based.

    Parameters
    ----------
    result : the result of ``oe.var``.
    steps : number of periods to forecast.
    data : optional table holding the series up to a new forecast origin (same
        columns as the fit). The forecast then starts after its last row. For a
        model with a trend the table must either have an integer time column or
        start at the first observation of the estimation data.
    exog : future values of the exogenous regressors, one row per step (required
        when the model has any).
    alpha : level of the forecast intervals.

    Any other keyword raises ``AnalysisError("invalid_option")`` (``oe.forecast``
    forwards its keywords unchanged).

    Returns a table with one row per variable and step: ``variable``, ``step``,
    ``period`` (when the time column holds integer periods), ``forecast``,
    ``std_error``, ``ci_low``, ``ci_high``.

    Example
    -------
    >>> result = oe.var(data=frame, y=["income", "consumption"], lags=2)
    >>> oe.forecast(result, 8)          # the same as oe.var_forecast(result, 8)
    """
    result = require_result(result, ("var",), "var_forecast")
    _reject_unknown("var_forecast", unknown, "data, exog and alpha")
    steps = check_steps(steps, MAX_FORECAST_STEPS)
    alpha = check_alpha(alpha)
    system = _system(result)
    layout = result.extra["layout"]
    names, p = system["names"], layout["lags"]
    k = len(names)
    coef = system["coef"]
    history, last_period, trend = _state(result, data, p)
    future = _future_exogenous(result, layout["exogenous"], exog, steps)
    recent = history.flip(0).clone()                      # most recent observation first
    mean = torch.zeros((steps, k), dtype=torch.float64)
    for h in range(steps):
        pieces = [recent.T.reshape(-1), future[h]]
        if layout["trend"]:
            pieces.append(torch.tensor([float(trend + h + 1)], dtype=torch.float64))
        if layout["constant"]:
            pieces.append(torch.ones(1, dtype=torch.float64))
        mean[h] = coef @ torch.cat(pieces)
        recent = torch.cat([mean[h][None], recent[:-1]], dim=0)
    phi = kernel_call(kernels.ma_matrices, system["a"], steps - 1)
    mse = impulse.forecast_mse(phi, system["sigma"])
    formula = "innovations only: sum_(i<h) Phi_i Sigma Phi_i'"
    moment = result.extra["forecast"].get("moment")
    if moment is not None and not layout["exogenous"] and not layout["trend"]:
        extra = _parameter_term(result, system, _tensor(moment), phi)
        if extra is not None:
            mse = mse + extra
            formula = ("innovations plus coefficient uncertainty (Lutkepohl 2005, 3.5; "
                       "fcast compute)")
    return _forecast_table(names, mean, mse, last_period, alpha, estimator="var",
                           mse_formula=formula)


def vec_forecast(result: ResultBundle, steps: int, *, data: Any = None,
                 alpha: float = 0.05, **unknown: Any) -> pd.DataFrame:
    """Dynamic forecasts of the levels from a fitted ``oe.vec`` model (Stata ``fcast compute``).

    The VEC model is rewritten as the VAR in levels
    ``y_t = A_1 y_(t-1) + ... + A_p y_(t-p) + c_0 + c_1 t + e_t`` with
    ``A_1 = alpha beta' + Gamma_1 + I``, ``A_i = Gamma_i - Gamma_(i-1)``,
    ``A_p = -Gamma_(p-1)`` and forecasts are iterated from the last p
    observations. The forecast-error covariance is Stata's

        Sigma_y^(h) = T / (T - d) * sum_(i<h) Phi_i Omega Phi_i',

    with Omega the ML innovation covariance and d the model's degrees of
    freedom per equation; parameter uncertainty is not included.

    Parameters: ``result`` (from ``oe.vec``), ``steps``, optional ``data`` with a
    new forecast origin (see ``oe.var_forecast``) and ``alpha``; any other
    keyword raises ``AnalysisError("invalid_option")``. Returns the same table
    as ``oe.var_forecast``.

    Example
    -------
    >>> result = oe.vec(data=frame, y=["y1", "y2"], lags=2, rank=1)
    >>> oe.forecast(result, 12)
    """
    result = require_result(result, ("vec",), "vec_forecast")
    _reject_unknown("vec_forecast", unknown, "data and alpha (a VEC model has no exogenous "
                    "regressors)")
    steps = check_steps(steps, MAX_FORECAST_STEPS)
    alpha = check_alpha(alpha)
    record = result.extra["var_representation"]
    names = result.extra["layout"]["variables"]
    a = _tensor(record["A"])
    p, k, _ = a.shape
    constant, slope = _tensor(record["constant"]), _tensor(record["trend"])
    history, last_period, trend = _state(result, data, p)
    recent = history.flip(0).clone()
    mean = torch.zeros((steps, k), dtype=torch.float64)
    for h in range(steps):
        mean[h] = torch.einsum("jab,jb->a", a, recent) + constant + slope * (trend + h + 1)
        recent = torch.cat([mean[h][None], recent[:-1]], dim=0)
    phi = kernel_call(kernels.ma_matrices, a, steps - 1)
    t, d = result.metrics["T"], result.metrics["df_eq"]
    mse = impulse.forecast_mse(phi, _tensor(result.extra["omega"])) * (t / (t - d))
    return _forecast_table(names, mean, mse, last_period, alpha, estimator="vec",
                           mse_formula="T/(T-d) * sum_(i<h) Phi_i Omega Phi_i' (no parameter "
                                       "uncertainty; fcast compute)")
