"""Prais-Winsten and Cochrane-Orcutt regression (Stata's ``prais``; EViews: LS with AR(1)).

The model is a linear regression with first-order autocorrelated errors,

    y_t = x_t'b + u_t,      u_t = rho u_(t-1) + e_t,      e_t iid (0, sigma^2),   |rho| < 1.

For a given rho the generalized least-squares estimator is OLS on the
quasi-differenced data

    y*_t = y_t - rho y_(t-1),        x*_t = x_t - rho x_(t-1),            t = 2..N,

and, for Prais-Winsten, the first observation is kept with the transformation
``y*_1 = sqrt(1 - rho^2) y_1`` (``x*_1`` likewise); Cochrane-Orcutt drops it.
rho is estimated from the residuals ``u_t = y_t - x_t'b`` by one of the
formulas of Stata's ``rhotype()`` (Stata [TS] prais, Methods and formulas):

    regress  rho = sum_(t>=2) u_t u_(t-1) / sum_(t>=2) u_(t-1)^2      (the default)
    freg     rho = sum_(t>=2) u_t u_(t-1) / sum_(t>=2) u_t^2
    tscorr   rho = sum_(t>=2) u_t u_(t-1) / sum_(t>=1) u_t^2
    dw       rho = 1 - DW / 2
    theil    rho = rho_tscorr (N - K) / N
    nagar    rho = (rho_dw N^2 + K^2) / (N^2 - K^2)

The iteration starts from OLS (rho = 0), computes rho from the residuals,
re-estimates b on the transformed data, recomputes rho from the untransformed
residuals, and stops when rho changes by less than ``tolerance``; the reported
coefficients are those of the transformed regression at the final rho.
``twostep`` stops after the first transformed regression.

Inference is that of OLS on the transformed data with N* = N (Prais-Winsten)
or N - 1 (Cochrane-Orcutt) observations: ``nonrobust`` s^2 (X*'X*)^-1 with
s^2 = SSR*/(N* - K); ``robust``/``HC1`` the White sandwich times N*/(N* - K)
(Stata's ``vce(robust)``), ``HC2``/``HC3`` the leverage-adjusted versions;
Student t with N* - K degrees of freedom. rho is treated as known, as in Stata.
The model test is Stata's ANOVA F of the transformed regression with the
nonrobust covariance, ``((TSS* - SSR*)/(K - 1)) / (SSR*/(N* - K))`` with TSS*
centered at the mean of y* (``regress ..., hascons``; for Prais-Winsten it is not
the Wald F of the slopes because the transformed constant is not a constant
column), and the Wald F of the slopes with a robust covariance.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    build_result, column_list, kernel_call, linear_covariance, wald_test,
)
from openecon.econometrics.tsmodels.common import (
    as_int, build_spec, durbin_watson, ordered_frame, perfect_fit, require_variation,
)
from openecon.engines.distributions import f_sf
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

RHOTYPES = ("regress", "freg", "tscorr", "dw", "theil", "nagar")


def estimate_rho(u: Tensor, rhotype: str, k: int) -> float:
    """rho from the untransformed residuals by Stata's rhotype formula."""
    n = u.shape[0]
    cross = float(torch.dot(u[1:], u[:-1]))
    if rhotype == "regress":
        return cross / float(u[:-1].square().sum())
    if rhotype == "freg":
        return cross / float(u[1:].square().sum())
    tscorr = cross / float(u.square().sum())
    if rhotype == "tscorr":
        return tscorr
    if rhotype == "theil":
        return tscorr * (n - k) / n
    rho_dw = 1.0 - durbin_watson(u) / 2.0
    if rhotype == "dw":
        return rho_dw
    return (rho_dw * n * n + k * k) / (n * n - k * k)          # nagar


def transform(values: Tensor, rho: float, prais: bool) -> Tensor:
    """Quasi-differences along the first dimension (first row kept or dropped)."""
    rest = values[1:] - rho * values[:-1]
    if not prais:
        return rest
    return torch.cat([math.sqrt(1.0 - rho * rho) * values[:1], rest])


def _transformed_fit(x: Tensor, y: Tensor, rho: float, prais: bool):
    xs, ys = transform(x, rho, prais), transform(y, rho, prais)
    try:
        fit = kernel_call(least_squares, xs, ys, drop_collinear=False)
    except AnalysisError as exc:
        if exc.code != "singular_design":
            raise
        raise AnalysisError("singular_design", f"The regressors are collinear after the transformation "
                            f"with rho = {rho:.6g}; remove regressors that are (nearly) "
                            "proportional to the transformed constant or to each other.") from exc
    return fit, xs, ys


def fit_prais(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``prais`` estimator (see the module docstring and ``prais``)."""
    frame, _ = ordered_frame(spec, data, what="Prais-Winsten regression")
    method, rhotype = frame.option("method"), frame.option("rhotype")
    twostep = bool(frame.option("twostep"))
    max_iterations, tolerance = int(frame.option("max_iterations")), float(frame.option("tolerance"))
    if not tolerance > 0.0:
        raise AnalysisError("invalid_option", "tolerance must be positive.")
    prais = method == "prais"
    design = frame.drop_collinear(frame.design())
    x, y = design.x, frame.numeric(spec.outcome)
    n, k = x.shape
    n_used = n if prais else n - 1
    if k == 0:
        raise AnalysisError("empty_design", "No regressors remain after removing collinear terms.")
    if n_used - k < 1 or n < 3:
        raise AnalysisError("insufficient_observations", f"{n} observations are too few for "
                            f"{k} coefficients after the {'Prais-Winsten' if prais else 'Cochrane-Orcutt'} "
                            "transformation.")
    require_variation(y, spec.outcome)
    ols = kernel_call(least_squares, x, y, drop_collinear=False)
    if perfect_fit(float(ols.ssr), y):
        raise AnalysisError("perfect_fit", "The regression fits the outcome exactly; rho is undefined.")
    dw_original = durbin_watson(ols.resid)
    rho = estimate_rho(ols.resid, rhotype, k)
    path = [0.0, rho]
    converged = twostep
    iterations = 1

    def admissible(value: float) -> float:
        if not math.isfinite(value):
            raise AnalysisError("numerical_failure", "rho could not be computed from the residuals.")
        if prais and abs(value) >= 1.0:
            raise AnalysisError("rho_out_of_range", f"The estimated rho ({value:.6g}) is not inside "
                                "(-1, 1), so the Prais-Winsten transformation of the first "
                                "observation is undefined; the errors look nonstationary. Use "
                                "method='corc', difference the data, or try another rhotype.")
        return value

    admissible(rho)
    fit, xs, ys = _transformed_fit(x, y, rho, prais)
    while not twostep and iterations < max_iterations:
        new = admissible(estimate_rho(y - x @ fit.beta, rhotype, k))
        iterations += 1
        path.append(new)
        change, rho = abs(new - rho), new
        fit, xs, ys = _transformed_fit(x, y, rho, prais)
        if change < tolerance:
            converged = True
            break
    if not converged:
        raise AnalysisError("nonconvergence", f"rho did not converge within {max_iterations} "
                            "iterations; raise max_iterations, loosen tolerance, or use "
                            "twostep=True.")
    if abs(rho) >= 1.0:
        frame.warn(f"The estimated rho ({rho:.6g}) is outside (-1, 1); the errors look nonstationary.")

    beta, resid_t = fit.beta, fit.resid
    ssr = float(fit.ssr)
    df_resid = n_used - k
    covariance, info = linear_covariance(frame, x=xs, resid=resid_t, bread=fit.xtx_inv, n=n_used,
                                         k=k, df_resid=df_resid, ssr=ssr)
    if design.intercept:
        total = float((ys - ys.mean()).square().sum())
    else:
        total = float(ys.square().sum())
    r_squared = 1.0 - ssr / total if total > 0.0 else math.nan
    df_model = k - int(design.intercept)
    adjusted = 1.0 - (1.0 - r_squared) * (n_used - int(design.intercept)) / df_resid
    structural = x @ beta
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests = {}
    if slopes and design.intercept and spec.covariance == "nonrobust" and total > 0.0:
        # Stata's prais reports the ANOVA F of the transformed regression (regress, hascons):
        # (TSS* - SSR*) / (K - 1) over SSR* / (N* - K). For Prais-Winsten the transformed
        # constant is not a constant column, so this differs from the Wald F of the slopes.
        statistic = ((total - ssr) / df_model) / (ssr / df_resid)
        tests["model"] = {"statistic": statistic, "df": df_model, "df2": df_resid,
                          "p_value": f_sf(max(statistic, 0.0), df_model, df_resid),
                          "distribution": "F",
                          "label": "ANOVA F test of the transformed regression (model sum of "
                                   "squares about the mean of y*), as Stata's prais"}
    elif slopes:
        tests["model"] = wald_test(beta, covariance, slopes, df_resid=df_resid,
                                   label="F test that all coefficients except the constant are zero")
    metrics = {
        "r_squared": r_squared, "adjusted_r_squared": adjusted, "rmse": math.sqrt(ssr / df_resid),
        "rho": rho, "durbin_watson_original": dw_original,
        "durbin_watson_transformed": durbin_watson(resid_t), "iterations": iterations,
        "df_model": df_model, "df_resid": df_resid, "ssr": ssr,
    }
    name = "Prais-Winsten" if prais else "Cochrane-Orcutt"
    info["correction"] = f"{info.get('correction')} on the {name}-transformed data; rho treated as known"
    observed = y
    if not prais:
        frame.restrict(torch.arange(n) >= 1, "Cochrane-Orcutt drops the first observation (it enters "
                       "only as the lag of the second).")
        observed, structural = y[1:], structural[1:]
    extra = {
        "method": method, "rhotype": rhotype, "twostep": twostep, "rho": rho,
        "rho_path": path, "converged": converged, "tolerance": tolerance,
        "transformed_r_squared_definition": "1 - SSR*/TSS* of the transformed regression; TSS* "
                                            "centered at the mean of y* when the model has a "
                                            "constant, uncentered otherwise",
    }
    return build_result(
        frame, terms=design.terms, params=beta, covariance=covariance,
        title=f"{name} AR(1) regression" + (" (two-step)" if twostep else " (iterated)"),
        df_inference=df_resid, df_resid=df_resid, metrics=metrics, fitted=structural,
        observed=observed, solver="iterated_feasible_gls_qr" if not twostep else "two_step_gls_qr",
        solver_diagnostics={"converged": converged, "iterations": iterations,
                            "condition_number": fit.condition_number},
        inference=info, tests=tests, extra=extra, categories=design.categories,
        provenance={"method": method, "rhotype": rhotype,
                    "fitted_values": "structural prediction x_t'b; residuals are u_t = y_t - x_t'b",
                    "time_order": f"sorted by {spec.time}" if spec.time else "input row order"},
    )


def prais(*, data: Any, y: str, x: Any, time: str | None = None, method: str = "prais",
          rhotype: str = "regress", twostep: bool = False, covariance: str | None = None,
          categorical: Any = None, intercept: bool = True, max_iterations: int = 100,
          tolerance: float = 1e-6, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Prais-Winsten and Cochrane-Orcutt regression with AR(1) errors (Stata's ``prais``).

    Model
    -----
    ``y_t = x_t'b + u_t`` with ``u_t = rho u_(t-1) + e_t``. Feasible GLS: OLS on the
    quasi-differenced data ``y_t - rho y_(t-1)`` and ``x_t - rho x_(t-1)``. Prais-Winsten
    (``method="prais"``, default) keeps the first observation transformed by
    ``sqrt(1 - rho^2)``; Cochrane-Orcutt (``method="corc"``) drops it.

    Estimator
    ---------
    Iterated as in Stata: OLS, rho from the residuals, transformed regression, new rho
    from the untransformed residuals ``y_t - x_t'b``, ... until rho changes by less than
    ``tolerance`` (default 1e-6, at most ``max_iterations`` updates). ``twostep=True``
    stops after the first transformed regression. ``rhotype`` chooses the formula for
    rho (Stata's ``rhotype()``): ``"regress"`` (default; u_t on u_(t-1) without a
    constant), ``"freg"`` (u_t on u_(t+1)), ``"tscorr"`` (u'u_(-1) / u'u), ``"dw"``
    (1 - DW/2), ``"theil"`` (tscorr * (N-K)/N), ``"nagar"`` ((rho_dw N^2 + K^2) /
    (N^2 - K^2)).

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of row records.
    y : outcome column.
    x : list of regressor columns.
    time : optional time column (integer periods without gaps, or datetimes). Without
        it the row order is the time order.
    method : ``"prais"`` or ``"corc"``.
    rhotype, twostep, max_iterations, tolerance : see Estimator.
    covariance : ``"nonrobust"`` (default; s^2 (X*'X*)^-1, s^2 = SSR*/(N*-K)), ``"robust"``
        or ``"HC1"`` (White sandwich on the transformed data times N*/(N*-K), Stata's
        ``vce(robust)``), ``"HC2"``, ``"HC3"`` (Stata's ``vce(hc2)``, ``vce(hc3)``).
    categorical : regressors to expand into treatment-coded indicators.
    intercept : include a constant (transformed like every other regressor).
    missing : ``"raise"`` or ``"drop"``; dropped rows may only lie at the start or end.
    alpha : significance level of the confidence intervals.

    Result
    ------
    Coefficients b with t tests on N* - K degrees of freedom (N* = N for Prais-Winsten,
    N - 1 for Cochrane-Orcutt). ``metrics``: ``r_squared``, ``adjusted_r_squared``,
    ``rmse`` (of the transformed regression, as Stata), ``rho``,
    ``durbin_watson_original`` (OLS residuals), ``durbin_watson_transformed``,
    ``iterations``, ``df_model``, ``df_resid``, ``ssr``. ``tests["model"]``: with the
    nonrobust covariance Stata's ANOVA F ``((TSS* - SSR*)/(K-1)) / (SSR*/(N*-K))`` of the
    transformed regression; with a robust covariance the Wald F that all coefficients
    except the constant are zero. ``extra``: ``rho_path`` (rho at
    every iteration, starting with 0 for OLS), ``method``, ``rhotype``, ``converged``.
    Predictions plot the structural fit x_t'b against y_t.

    Errors (``AnalysisError``): ``rho_out_of_range`` (Prais-Winsten needs |rho| < 1),
    ``nonconvergence``, ``time_gaps``, ``insufficient_observations``,
    ``singular_design``, ``perfect_fit``, ``invalid_spec``.

    Stata: ``prais y x1 x2`` / ``prais y x1 x2, corc`` / ``..., twostep rhotype(dw)
    vce(robust)``. EViews: ``ls y c x1 x2 ar(1)`` estimates the same model by nonlinear
    least squares (different first-observation treatment).

    Example
    -------
    >>> import openecon as oe
    >>> fit = oe.prais(data=df, y="usr", x=["idle"], time="t")
    >>> fit.metrics["rho"], fit.metrics["durbin_watson_transformed"]
    """
    spec = build_spec(
        "prais", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept, time=time,
        covariance=covariance, missing=missing, alpha=alpha,
        options={"method": method, "rhotype": rhotype, "twostep": twostep,
                 "max_iterations": as_int(max_iterations), "tolerance": tolerance},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)
