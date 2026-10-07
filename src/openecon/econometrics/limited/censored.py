"""Tobit and interval regression by full maximum likelihood: Stata's ``tobit`` and ``intreg``.

Model
-----
A latent normal regression ``y* = x'b + e``, ``e ~ N(0, sigma^2)``, of which
only partial information is observed:

* ``tobit``: ``y = max(ll, min(y*, ul))``. Outcomes at or below the lower
  limit are left-censored, outcomes at or above the upper limit right-censored
  and the others observed exactly.
* ``intreg``: each observation is a point ``y* = a``, an interval
  ``a <= y* <= b``, a left-censored value ``y* <= b`` or a right-censored value
  ``y* >= a``, coded by the pair (lower bound, upper bound) with a missing
  value for an open side.

Both are the same likelihood (``kernels.CensoredObjective``):

    ln L = sum_i w_i * { -ln sigma + ln phi((y_i - m_i) / sigma)            point
                         ln Phi((b_i - m_i) / sigma)                        left-censored
                         ln Phi(-(a_i - m_i) / sigma)                       right-censored
                         ln[Phi((b_i - m_i)/sigma) - Phi((a_i - m_i)/sigma)]  interval }

with ``m_i = x_i'b + offset_i``.

Estimation
----------
Newton-Raphson on ``(b, ln sigma)`` with the analytic score and Hessian,
started at the weighted least-squares fit of the observed values (interval
midpoints, censoring limits) and its ML residual scale. With a constant the
regressors are centred at their means during the iteration and the constant
is mapped back exactly (``common`` module notes). The constant-only model is
fitted the same way for the likelihood-ratio test and the pseudo R-squared.

Reported (Stata's conventions)
------------------------------
``tobit``: Student-t coefficient tests with ``N - df_model`` degrees of
freedom (``df_model`` = number of slope coefficients), the ancillary
``/sigma`` with its delta-method standard error, the LR chi2 model test under
a likelihood-based covariance and the Wald F test otherwise, McFadden's pseudo
R-squared. ``intreg``: z tests, the ancillary ``/lnsigma`` (``sigma`` in
``metrics`` and ``extra``), the LR chi2 or Wald chi2 model test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, lr_test,
    wald_test,
)
from openecon.econometrics.limited.common import (
    LIKELIHOOD_KINDS, Weights, build_spec, centre, check_pweights, constant_shift, count,
    likelihood_covariance, likelihood_weights, limit_options, maximize, offset_column,
    ols_start, require_observations, resolve_covariance, resolve_limits, run_newton,
    scale_to_sigma, sigma_record, slope_indices, solver_fields, uncentre, weight_scale,
)
from openecon.econometrics.limited.kernels import CensoredObjective
from openecon.econometrics.registry import role_columns
from openecon.models import ModelSpec, ResultBundle

_ADVICE = ("Check that the uncensored outcomes vary, that the limits match the data and the "
           "scale of the regressors.")


def _null_log_likelihood(frame: ModelFrame, lower: Tensor, upper: Tensor, start: Tensor,
                         weights: Weights, offset: Tensor | None, command: str) -> float | None:
    """Log likelihood of the constant-only model (None, with a warning, if it fails)."""
    ones = torch.ones((len(start), 1), dtype=torch.float64)
    objective = kernel_call(CensoredObjective, ones, lower, upper, weights.user, offset)
    beta, tau = ols_start(ones, start, weights.user, offset, command=command)
    result = maximize(objective, torch.cat([beta, torch.tensor([tau], dtype=torch.float64)]),
                      what=f"{command} (constant-only model)", scale=weight_scale(weights))
    if not result.converged:
        frame.warn("The constant-only model did not converge; the model test is a Wald test "
                   "and no pseudo R-squared is reported.")
        return None
    return result.value


def _fit_censored(frame: ModelFrame, lower: Tensor, upper: Tensor, start: Tensor,
                  weights: Weights, command: str):
    """Shared estimation of the censored-normal likelihood; returns the pieces of a result.

    ``(design, fitted, result, theta, covariance, info, null)``: ``theta`` =
    ``(b, ln sigma)`` and its covariance refer to the regressors as given (the
    iteration itself runs on centred regressors), ``fitted`` is ``x'b +
    offset`` and ``null`` the log likelihood of the constant-only model.
    """
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    offset = offset_column(frame)
    k = len(design.terms)
    if k == 0:
        raise AnalysisError("no_parameters", f"{command}: no regressor is left; include a "
                            "constant or a regressor that varies.")
    require_observations(weights.nobs, k + 1)
    means = centre(design, weights.for_screen())
    objective = kernel_call(CensoredObjective, design.x, lower, upper, weights.user, offset)
    beta, tau = ols_start(design.x, start, weights.user, offset, command=command)
    result = run_newton(objective, torch.cat([beta, torch.tensor([tau], dtype=torch.float64)]),
                        command=command, advice=_ADVICE, scale=weight_scale(weights))
    centred = result.theta
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    theta, covariance = uncentre(centred, covariance, constant_shift([means], extra=1))
    null = None
    if design.intercept:
        null = result.value if k == 1 else _null_log_likelihood(
            frame, lower, upper, start, weights, offset, command)
    return design, objective.index(centred), result, theta, covariance, info, null


def _limits_mask(y: Tensor, limit: float | None, below: bool) -> Tensor:
    if limit is None:
        return torch.zeros(len(y), dtype=torch.bool)
    return y <= limit if below else y >= limit


def censoring_bounds(y: Tensor, ll: float | None, ul: float | None, weights: Weights,
                     command: str) -> tuple[Tensor, Tensor, int, int, int]:
    """Bounds of the latent outcome of a tobit model and the censoring counts.

    Returns ``(lower, upper, n_left, n_uncensored, n_right)``: an outcome at or
    below ``ll`` is known to be ``<= ll``, one at or above ``ul`` to be
    ``>= ul`` and the others are observed exactly.
    """
    left, right = _limits_mask(y, ll, True), _limits_mask(y, ul, False)
    n_left, n_right = count(left, weights), count(right, weights)
    n_uncensored = weights.nobs - n_left - n_right
    if n_uncensored == 0:
        # Even with observations censored on both sides the likelihood increases
        # monotonically as sigma grows: there is no interior maximum.
        raise AnalysisError(
            "no_uncensored_observations",
            f"{command}: every observation is censored (at or beyond a limit), so the scale of "
            "the latent regression is not identified. Check the limits ll / ul against the "
            "outcome.")
    infinity = torch.full_like(y, math.inf)
    lower = torch.where(left, -infinity, torch.where(right, torch.full_like(y, ul or 0.0), y))
    upper = torch.where(right, infinity, torch.where(left, torch.full_like(y, ll or 0.0), y))
    return lower, upper, n_left, n_uncensored, n_right


def fit_tobit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``tobit``: ``(spec, data) -> ResultBundle``."""
    command = "tobit"
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    y = frame.numeric(spec.outcome)
    ll, ul = resolve_limits(frame, y, command)
    weights = likelihood_weights(frame)
    lower, upper, n_left, n_uncensored, n_right = censoring_bounds(y, ll, ul, weights, command)
    design, fitted, result, theta, covariance, info, null = _fit_censored(
        frame, lower, upper, y, weights, command)
    k = len(design.terms)
    log_likelihood = result.value
    params, reported = scale_to_sigma(theta, covariance, k)
    sigma = float(params[k])
    slopes = slope_indices(design)
    df_model = len(slopes)
    df_resid = weights.nobs - df_model
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations", f"{command}: no residual degrees of "
                            "freedom remain.")
    tests: dict[str, Any] = {}
    if df_model:
        if spec.covariance in LIKELIHOOD_KINDS and null is not None:
            tests["model"] = lr_test(log_likelihood, null, df_model,
                                     label="LR chi2 test against the constant-only model")
        else:
            tests["model"] = wald_test(params, reported, slopes, df_resid=df_resid,
                                       label="Wald F test of the slopes")
    criteria = information_criteria(log_likelihood, k + 1, weights.nobs)
    metrics = {
        "log_likelihood": log_likelihood,
        "pseudo_r_squared": None if not null else 1 - log_likelihood / null,
        "aic": criteria["aic"], "bic": criteria["bic"], "sigma": sigma,
        "n_left_censored": n_left, "n_uncensored": n_uncensored, "n_right_censored": n_right,
        "df_model": df_model, "df_resid": df_resid,
    }
    sigma_se = math.sqrt(max(float(reported[k, k]), 0.0))
    extra = {
        "limits": {"lower": ll, "upper": ul},
        "null_log_likelihood": null,
        "lnsigma": {"estimate": float(theta[k]),
                    "std_error": math.sqrt(max(float(covariance[k, k]), 0.0))},
        "variance": {"estimate": sigma ** 2, "std_error": 2 * sigma * sigma_se,
                     "method": "var(e) = sigma^2; delta-method standard error 2 sigma se(sigma)"},
        "parameterization": "estimated in (b, ln sigma); /sigma by the delta method",
        "starting_values": "least squares of the observed outcome and its ML residual scale",
    }
    info["df_inference"] = df_resid
    info["correction"] = (f"{info['correction']}; /sigma by the delta method; Student t with "
                          f"N - df_model = {df_resid} degrees of freedom (Stata's tobit)")
    return build_result(
        frame, terms=[*design.terms, "/sigma"], params=params, covariance=reported,
        equations=[spec.outcome] * k + [None], use_t=True, df_inference=df_resid,
        df_resid=df_resid, metrics=metrics, fitted=fitted, nobs=weights.nobs,
        inference=info, tests=tests, extra=extra, categories=design.categories,
        provenance={"prediction_definition": "linear prediction x'b of the latent outcome"},
        **solver_fields(result))


def _bound(frame: ModelFrame, name: str) -> Tensor:
    """An interval bound of the current sample: float64 with NaN for a missing (open) side."""
    series = frame.series(name)
    if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
        raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric.")
    values = torch.as_tensor(series.to_numpy(dtype="float64", na_value=float("nan")).copy(),
                             dtype=torch.float64)
    if bool(torch.isinf(values).any()):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains infinite values; "
                            "use a missing value for an open interval bound.")
    return values


def fit_intreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``intreg``: ``(spec, data) -> ResultBundle``."""
    command = "intreg"
    high_name = role_columns(spec, "upper")[0]
    if high_name == spec.outcome:
        raise AnalysisError("invalid_spec", f"{command}: y_low and y_high must be two different "
                            "columns (give equal values for point data).")
    if high_name in spec.predictors:
        raise AnalysisError("invalid_spec", f"{command}: the upper bound cannot be a regressor.")
    frame = ModelFrame(spec, data, allow_missing=[spec.outcome, high_name])
    check_pweights(frame)
    empty = torch.isnan(_bound(frame, spec.outcome)) & torch.isnan(_bound(frame, high_name))
    if bool(empty.any()):
        number = int(empty.sum())
        if spec.missing == "raise":
            raise AnalysisError(
                "missing_values",
                f"{number} observation(s) have both interval bounds missing and carry no "
                "information. Choose missing='drop' explicitly to exclude them.")
        frame.restrict(~empty, f"Excluded {number} observation(s) with both interval bounds "
                               "missing.")
    low, high = _bound(frame, spec.outcome), _bound(frame, high_name)
    open_low, open_high = torch.isnan(low), torch.isnan(high)
    if bool((low > high).any()):
        raise AnalysisError("invalid_interval", f"{command}: {int((low > high).sum())} "
                            f"observation(s) have '{spec.outcome}' above '{high_name}'. The "
                            "lower bound must not exceed the upper bound.")
    weights = likelihood_weights(frame)
    point = ~open_low & ~open_high & (low == high)
    interval = ~open_low & ~open_high & (low < high)
    n_point, n_interval = count(point, weights), count(interval, weights)
    n_left, n_right = count(open_low, weights), count(open_high, weights)
    start = torch.where(open_low, high, torch.where(open_high, low, (low + high) / 2))
    if n_point == 0 and n_interval == 0 and (
            n_left == 0 or n_right == 0 or bool((start == start[0]).all())):
        # One-sided censoring only, or a single common threshold (a probit): the
        # scale of the latent regression cannot be estimated.
        raise AnalysisError(
            "no_uncensored_observations",
            f"{command}: every observation is censored on the same side or at the same value, "
            "so the regression and its scale are not identified. Provide point or interval "
            "observations.")
    infinity = torch.full_like(low, math.inf)
    lower = torch.where(open_low, -infinity, low)
    upper = torch.where(open_high, infinity, high)
    design, fitted, result, theta, covariance, info, null = _fit_censored(
        frame, lower, upper, start, weights, command)
    k = len(design.terms)
    log_likelihood = result.value
    slopes = slope_indices(design)
    tests: dict[str, Any] = {}
    if slopes:
        if spec.covariance in LIKELIHOOD_KINDS and null is not None:
            tests["model"] = lr_test(log_likelihood, null, len(slopes),
                                     label="LR chi2 test against the constant-only model")
        else:
            tests["model"] = wald_test(theta, covariance, slopes,
                                       label="Wald chi2 test of the slopes")
    criteria = information_criteria(log_likelihood, k + 1, weights.nobs)
    sigma = sigma_record(float(theta[k]), float(covariance[k, k]), spec.alpha)
    metrics = {
        "log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
        "sigma": sigma["estimate"], "n_left_censored": n_left, "n_uncensored": n_point,
        "n_right_censored": n_right, "n_interval": n_interval, "n_point": n_point,
        "df_model": len(slopes),
    }
    extra = {
        "sigma": sigma, "null_log_likelihood": null, "bounds": [spec.outcome, high_name],
        "starting_values": "least squares of the interval midpoints (censoring limits) and its "
                           "ML residual scale",
    }
    return build_result(
        frame, terms=[*design.terms, "/lnsigma"], params=theta, covariance=covariance,
        equations=[spec.outcome] * k + [None], use_t=False, metrics=metrics,
        fitted=fitted, observed=start, nobs=weights.nobs, inference=info,
        tests=tests, extra=extra, categories=design.categories,
        provenance={"prediction_definition": "linear prediction x'b; the observed value is the "
                                             "interval midpoint or the finite bound"},
        **solver_fields(result))


def tobit(*, data: Any, y: str, x: Sequence[str], ll: float | str | None = None,
          ul: float | str | None = None, covariance: str | None = None,
          cluster: str | Sequence[str] | None = None, weights: str | None = None,
          weight_type: str | None = None, offset: str | None = None,
          categorical: Sequence[str] | None = None, intercept: bool = True,
          missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Tobit (censored normal) regression, Stata's ``tobit``.

    Model
        ``y* = x'b + e`` with ``e ~ N(0, sigma^2)`` and ``y = max(ll, min(y*, ul))``:
        the outcome is observed exactly between the limits and only known to be at or
        beyond a limit otherwise. The coefficients are the effects on the latent ``y*``.

    Estimator
        Full maximum likelihood,

            ``ln L = sum_unc w [-ln sigma + ln phi((y - x'b)/sigma)]
                     + sum_left w ln Phi((ll - x'b)/sigma)
                     + sum_right w ln Phi((x'b - ul)/sigma)``,

        by Newton-Raphson on ``(b, ln sigma)`` with the analytic score and Hessian (stable
        ``ln Phi`` and inverse Mills ratio in both tails), started at least squares.
        ``/sigma`` and its standard error follow from the delta method. With a constant
        the regressors are centred at their means during the iteration and the constant
        is mapped back exactly, so a regressor with a large level (a date) costs no
        precision.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome; values ``<= ll`` are left-censored, values ``>= ul`` right-censored.
        x: regressors; collinear columns are omitted with a warning.
        ll, ul: censoring limits. A number, ``'min'`` / ``'max'`` for the smallest /
            largest outcome in the sample (Stata's ``ll`` / ``ul`` without a value), or
            None for no censoring on that side. With neither limit the fit is the normal
            regression by ML (OLS coefficients, ``sigma^2 = SSR / N``).
        covariance: ``'nonrobust'`` (default; inverse observed information, ``vce(oim)``),
            ``'opg'`` (``(sum w s s')^-1``), ``'robust'`` (``N/(N-1)`` sandwich),
            ``'cluster'`` (``G/(G-1)``; implied by ``cluster``; two columns give two-way
            clustering).
        cluster: cluster column, or a list of two columns.
        weights, weight_type: ``'fweight'`` (replicated rows, ``N = sum w``), ``'aweight'``
            (rescaled to sum to N), ``'pweight'`` (robust by default), ``'iweight'``.
        offset: column added to ``x'b`` with coefficient one.
        categorical: columns of ``x`` to expand into treatment-coded indicators.
        intercept: include a constant (default True).
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Errors
        ``no_uncensored_observations`` (every outcome is at or beyond a limit: the scale
        is not identified), ``perfect_fit`` (the regressors fit the outcome exactly),
        ``invalid_limits`` (``ll >= ul``), ``nonconvergence`` (for example a regressor
        that separates censored from uncensored outcomes).

    Result
        Student-t statistics with ``N - df_model`` degrees of freedom (``df_model`` = the
        number of slopes), as Stata's tobit. Terms: the regressors (equation = the outcome)
        and the ancillary ``/sigma``. ``metrics``: log_likelihood, pseudo_r_squared
        (McFadden, against the constant-only tobit), aic, bic (``K + 1`` parameters),
        sigma, n_left_censored, n_uncensored, n_right_censored, df_model, df_resid.
        ``tests['model']``: LR chi2(df_model) against the constant-only model under
        ``nonrobust`` / ``opg`` (with a constant), else the Wald F(df_model, df_resid).
        ``extra``: limits, null_log_likelihood, lnsigma, variance (``var(e) = sigma^2``
        with its delta-method standard error, the quantity recent Stata versions print).
        The chart sample holds the linear prediction ``x'b``.

    Stata / EViews
        ``tobit y x1 x2, ll(0) vce(robust)`` is
        ``oe.tobit(data=df, y='y', x=['x1', 'x2'], ll=0, covariance='robust')``;
        ``tobit y x1, ll ul(10)`` is ``oe.tobit(..., ll='min', ul=10)``. EViews: censored
        regression with normal errors (``censored(d=n) y c x1 x2``).

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=1000)})
        >>> df["y"] = np.maximum(0.0, 0.5 + df.x + rng.normal(size=1000))
        >>> print(oe.tobit(data=df, y="y", x=["x"], ll=0).summary())
    """
    spec = build_spec(
        "tobit", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"offset": offset}, options=limit_options(ll, ul, "tobit"))
    return fit(spec, data=data)


def intreg(*, data: Any, y_low: str, y_high: str, x: Sequence[str],
           covariance: str | None = None, cluster: str | Sequence[str] | None = None,
           weights: str | None = None, weight_type: str | None = None,
           offset: str | None = None, categorical: Sequence[str] | None = None,
           intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Interval regression, Stata's ``intreg``.

    Model
        ``y* = x'b + e`` with ``e ~ N(0, sigma^2)``, where ``y*`` is known only through
        the pair ``(y_low, y_high)``:

        ===================  ===========================  ================================
        ``y_low``            ``y_high``                   observation
        ===================  ===========================  ================================
        ``a``                ``a``                        point data ``y* = a``
        ``a``                ``b > a``                    interval ``a <= y* <= b``
        missing              ``b``                        left-censored ``y* <= b``
        ``a``                missing                      right-censored ``y* >= a``
        ===================  ===========================  ================================

        A row with both bounds missing carries no information: it raises
        ``missing_values`` (or is excluded with ``missing='drop'``). ``y_low > y_high``
        raises ``invalid_interval``; a sample of censored observations on one side only,
        or at a single common value, raises ``no_uncensored_observations``.

    Estimator
        Full maximum likelihood by Newton-Raphson on ``(b, ln sigma)`` with the analytic
        score and Hessian; an interval contributes
        ``ln[Phi((b - x'b)/sigma) - Phi((a - x'b)/sigma)]``, computed in the tail where
        both terms are small. Starting values: least squares of the interval midpoints
        (the finite bound of censored observations). With a constant the regressors are
        centred during the iteration and the constant is mapped back exactly.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y_low, y_high: lower and upper bound columns (missing = open side).
        x: regressors; collinear columns are omitted with a warning.
        covariance: ``'nonrobust'`` (default; inverse observed information), ``'opg'``,
            ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``; implied by
            ``cluster``; two columns give two-way clustering).
        cluster: cluster column, or a list of two columns.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        offset: column added to ``x'b`` with coefficient one.
        categorical: columns of ``x`` to expand into treatment-coded indicators.
        intercept: include a constant (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing regressors (or with
            both bounds missing).
        alpha: significance level of the confidence intervals.

    Result
        z statistics. Terms: the regressors (equation = ``y_low``) and the ancillary
        ``/lnsigma``. ``metrics``: log_likelihood, aic, bic, sigma, n_left_censored,
        n_uncensored (point data; also as n_point), n_right_censored, n_interval, df_model.
        ``tests['model']``: LR chi2 against the constant-only model under ``nonrobust`` /
        ``opg`` (with a constant), else the Wald chi2 of the slopes. ``extra['sigma']``:
        ``sigma = exp(lnsigma)`` with its delta-method standard error and the interval
        obtained by exponentiating the ``lnsigma`` interval, as Stata prints. The chart
        sample compares ``x'b`` with the interval midpoint (or the finite bound).

    Stata
        ``intreg wage1 wage2 age educ, vce(robust)`` is
        ``oe.intreg(data=df, y_low='wage1', y_high='wage2', x=['age', 'educ'],
        covariance='robust')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=1000)})
        >>> latent = 1.0 + 0.8 * df.x + rng.normal(size=1000)
        >>> df["lo"] = np.floor(latent)            # the outcome is known up to its unit
        >>> df["hi"] = df.lo + 1.0
        >>> print(oe.intreg(data=df, y_low="lo", y_high="hi", x=["x"]).summary())
    """
    spec = build_spec(
        "intreg", outcome=y_low, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"upper": y_high, "offset": offset})
    return fit(spec, data=data)
