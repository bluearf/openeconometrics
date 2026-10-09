"""Stata's xtreg: fixed, random, between, first-difference, ML and pooled panel regressions.

Notation: N observations in n panels of T_i periods, k slopes (K = k + 1 with
the constant), panel means ybar_i, xbar_i and grand means ybar, xbar. All
least-squares steps are Householder QR (``linalg.least_squares``); group
means are one ``index_add_`` pass. Conventions follow the Methods and
formulas of ``xtreg`` (see ``docs/econometrics/panel.md``):

fe      Within estimator: OLS of y_it - ybar_i + ybar on x_it - xbar_i + xbar and a
        constant, so the reported Intercept is Stata's _cons = ybar - xbar'b.
        Regressors without within variation are omitted. sigma_e^2 = RSS/(N - n - k),
        nonrobust V = sigma_e^2 (X*'X*)^-1; u_i = ybar_i - xbar_i'b, sigma_u = sd of
        the u_i over the n panels (n - 1 divisor), rho = sigma_u^2/(sigma_u^2 +
        sigma_e^2), corr_u_xb = corr(u_i, x_it'b) over the N observations (Stata's
        e(corr)). R-squared within, between and overall are corr((x_it - xbar_i)'b,
        y_it - ybar_i)^2, corr(xbar_i'b, ybar_i)^2 and corr(x_it'b, y_it)^2. Tests:
        F(k, N-n-k) for the slopes (F(k, G-1) under cluster, F(k, T-1) under
        Driscoll-Kraay) and, nonrobust only, the F(n-1, N-n-k) test that all u_i = 0
        against pooled OLS on the same regressors. An outcome without within
        variation, or a within regression whose residuals are rounding noise, is
        refused (no_within_variation) instead of reported with noise standard errors.
re      Swamy-Arora feasible GLS: components from ``common.random_effects_gls``,
        y_it - theta_i ybar_i on x_it - theta_i xbar_i and 1 - theta_i by OLS; z
        inference; nonrobust V = rmse^2 (X*'X*)^-1 with rmse^2 = RSS*/(N - K) of that
        transformed regression (Stata's e(rmse), reported as metrics['rmse']);
        Wald chi2(k) model test and the Breusch-Pagan LM test for sigma_u = 0.
be      OLS of ybar_i on xbar_i and a constant over the n panels (``wls`` weights by
        T_i); t inference with n - K degrees of freedom; robust is HC1 on this
        regression, cluster needs a cluster column constant within panel.
fd      OLS of y_it - y_i,t-1 on x_it - x_i,t-1 and a constant (the drift), using
        consecutive periods only (gaps and first observations are dropped and
        recorded); t inference with N_d - K degrees of freedom.
pooled  Pooled OLS with the panel covariances (robust = cluster on the panel,
        Driscoll-Kraay available).
mle     ``panel.mle``: Gaussian random-effects maximum likelihood.

Cluster covariances for fe and re require every cluster column to nest the
panels, as Stata does (cluster_not_nested). Weights: fe (constant within
panel), fd and pooled take aweight/fweight/pweight; re, be and mle take none.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test,
)
from openecon.econometrics.panel import kernels
from openecon.econometrics.panel.common import (
    EXACT_FIT, PanelSample, center_slopes, covariance, panel_sample, random_effects_gls,
    require_residual_variation, uncenter_coefficients,
)
from openecon.engines import linalg
from openecon.engines.absorb import group_means
from openecon.engines.distributions import f_sf
from openecon.models import ModelSpec, ResultBundle

_LINEAR = {"nonrobust", "robust", "cluster"}
_COVARIANCES = {"cre": _LINEAR, "fe": _LINEAR | {"driscoll_kraay"}, "pooled": _LINEAR | {"driscoll_kraay"},
                "re": _LINEAR, "be": _LINEAR, "fd": _LINEAR, "mle": {"nonrobust"}}
_TITLES = {"fe": "Fixed-effects (within) regression", "re": "Random-effects GLS regression",
           "be": "Between regression", "fd": "First-difference regression",
           "pooled": "Pooled OLS regression", "mle": "Random-effects ML regression"}
_NO_WEIGHTS = {"cre": "none", "re": "none", "be": "none",
               "mle": "iweights only, which OpenEconometrics does not support"}


def fit_xtreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point: dispatch on ``options['model']`` after the shared validation."""
    frame = ModelFrame(spec, data)
    model = frame.option("model")
    _validate(frame, model)
    frame.sort_panel()
    if model == "fd":
        return _fit_fd(frame)
    sample = panel_sample(frame, constant_weights=model == "fe")
    if model == "cre":
        from openecon.econometrics.panel.mundlak import fit_cre
        return fit_cre(sample)
    if model == "mle":
        from openecon.econometrics.panel.mle import fit_mle

        return fit_mle(sample)
    return {"fe": _fit_fe, "re": _fit_re, "be": _fit_be, "pooled": _fit_pooled}[model](sample)


def _validate(frame: ModelFrame, model: str) -> None:
    spec = frame.spec
    if spec.covariance not in _COVARIANCES[model]:
        allowed = ", ".join(sorted(_COVARIANCES[model]))
        raise AnalysisError("unsupported_covariance",
                            f"model='{model}' supports only these covariances: {allowed}.")
    if spec.covariance != "driscoll_kraay" and ("lags" in spec.options or "kernel" in spec.options):
        raise AnalysisError("invalid_spec",
                            "lags and kernel apply to covariance='driscoll_kraay' only.")
    if spec.covariance == "driscoll_kraay" and spec.time is None:
        raise AnalysisError("invalid_spec", "Driscoll-Kraay covariance needs a time column.")
    if model == "fd" and spec.time is None:
        raise AnalysisError("invalid_spec", "model='fd' needs a time column to difference "
                            "consecutive periods.")
    for option, owner in (("sa", "re"), ("wls", "be")):
        if frame.option(option) and model != owner and not (option == "sa" and model == "cre"):
            raise AnalysisError("invalid_spec", f"{option}=True applies to model='{owner}' only.")
    if spec.weights is not None and model in _NO_WEIGHTS:
        raise AnalysisError("unsupported_weights",
                            f"model='{model}' does not accept weights: Stata's xtreg, {model} "
                            f"allows {_NO_WEIGHTS[model]}. Weights are available for "
                            "model='fe' (constant within panel), 'fd' and 'pooled'.")


def _least_squares(x: Tensor, y: Tensor, weights: Tensor | None = None, **options: Any):
    return kernel_call(linalg.least_squares, x, y, weights, **options)


def _mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values.mean(dim=0)
    if values.ndim == 1:
        return (weights @ values) / weights.sum()
    return (weights[:, None] * values).sum(dim=0) / weights.sum()


def _sum_squares(values: Tensor, weights: Tensor | None) -> float:
    return float(values.square().sum() if weights is None else weights @ values.square())


def _ones(n: int) -> Tensor:
    return torch.ones((n, 1), dtype=torch.float64)


def _f_test(numerator: float, denominator: float, df1: float, df2: float, label: str) -> dict:
    if denominator <= 0 or df1 <= 0 or df2 <= 0:
        return {"statistic": None, "df": df1, "df2": df2, "p_value": None, "distribution": "F",
                "label": label}
    statistic = max(0.0, numerator / denominator)
    return {"statistic": statistic, "df": df1, "df2": df2, "p_value": f_sf(statistic, df1, df2),
            "distribution": "F", "label": label}


def _require_df(df_resid: float, what: str) -> None:
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations",
                            f"{what} leaves no residual degrees of freedom ({df_resid:g}).")


def _ols_metrics(y: Tensor, ssr: float, weights: Tensor | None, nobs: int, kk: int,
                 outcome: str, what: str, scale: float | None = None) -> dict:
    """regress-style fit statistics; ``scale`` is ``sum w y^2`` of the outcome's level."""
    tss = _sum_squares(y - _mean(y, weights), weights)
    require_residual_variation(ssr, tss, _sum_squares(y, weights) if scale is None else scale,
                               outcome, what)
    r2 = 1 - ssr / tss
    return {"r_squared": r2, "adjusted_r_squared": 1 - (nobs - 1) / (nobs - kk) * (1 - r2),
            "rmse": math.sqrt(ssr / (nobs - kk)), "df_model": kk - 1, "df_resid": nobs - kk}


def _slope_columns(design: Design, terms: Sequence[str]) -> list[int]:
    """Column positions in ``design`` of the named (non-constant) terms."""
    index = {term: i for i, term in enumerate(design.terms)}
    return [index[term] for term in terms]


# ---- fixed effects ------------------------------------------------------------------


def _check_constant_variance(frame: ModelFrame, robust: float, classical: float) -> None:
    """Flag a cluster-robust variance of the fe constant that is zero up to rounding.

    Within residuals sum to zero in every panel, hence in every cluster that nests
    panels, so the sandwich variance of Stata's _cons is exactly xbar'V_b xbar. With
    regressors of (near-)zero grand mean that is rounding noise: Stata prints it, and so
    do we, but with a warning; an exact zero cannot be reported at all.
    """
    if robust <= 0:
        raise AnalysisError("zero_variance_constant",
                            "The cluster-robust variance of the Intercept is exactly zero: within "
                            "residuals sum to zero in every panel and the regressors have zero "
                            "grand means. Use covariance='nonrobust' or shift a regressor; the "
                            "slopes are unaffected.")
    if robust <= 1e-20 * classical:
        frame.warn("The cluster-robust variance of the Intercept is zero up to rounding: it equals "
                   "xbar'V xbar (within residuals sum to zero in every panel) and the regressors "
                   "have near-zero grand means. Its standard error, t statistic and p-value are "
                   "not meaningful; the slopes are unaffected.")


def _fit_fe(sample: PanelSample) -> ResultBundle:
    frame, spec, w = sample.frame, sample.spec, sample.weights
    codes, n, nobs, rows = sample.codes, sample.n, sample.nobs, sample.frame.n
    design = frame.design(intercept=False)
    y = frame.numeric(spec.outcome)
    ybar = group_means(y, codes, n, w)
    within_y = y - ybar[codes]
    # Rounding floor of a sum of squares of deviations of y (see common.EXACT_FIT).
    floor = EXACT_FIT * _sum_squares(y, w)
    if _sum_squares(within_y, w) <= floor:
        raise AnalysisError("no_within_variation",
                            f"The outcome '{spec.outcome}' has no within-panel variation (it is "
                            "constant inside every panel), so the fixed-effects model cannot be "
                            "estimated. Use model='be' or model='pooled'.")
    # One N x (k+1) buffer holds [1, x_it - xbar_i] and is reused for the pooled F test.
    # Stata's transformed design [1, x_it - xbar_i + xbar] is this one with the grand means
    # added back, an exact reparameterization (Intercept = Intercept_c - xbar'b) that is
    # undone below; here the constant stays orthogonal to the slopes, so the regression
    # and its sandwich covariances are well conditioned whatever the level of the regressors.
    k = design.x.shape[1]
    xs = torch.empty((rows, k + 1), dtype=torch.float64)
    xs[:, 0] = 1.0
    xs[:, 1:] = design.x
    xs[:, 1:] -= group_means(design.x, codes, n, w)[codes]
    design, within_x = frame.drop_absorbed(design, xs[:, 1:], w)
    if within_x.shape[1] != k:
        xs = torch.cat([xs[:, :1], within_x], dim=1)
        k = within_x.shape[1]
    del within_x
    df_resid = nobs - n - k
    _require_df(df_resid, "The within regression")
    xbar = group_means(design.x, codes, n, w)
    mean_x, mean_y = _mean(design.x, w), _mean(y, w)
    fit = _least_squares(xs, within_y + mean_y, w, drop_collinear=False)
    ssr = float(fit.ssr)
    if ssr <= floor:
        raise AnalysisError("no_within_variation",
                            f"The regressors explain the within-panel variation of "
                            f"'{spec.outcome}' exactly (the residual sum of squares is rounding "
                            "noise), so standard errors are undefined. Remove the regressor "
                            "that reproduces the outcome.")
    v, info, df_inference = covariance(sample, x=xs, resid=fit.resid, bread=fit.xtx_inv,
                                       nobs=nobs, k=k + 1, df_resid=df_resid, weights=w,
                                       means=mean_x, nested=True)
    beta = uncenter_coefficients(fit.beta, mean_x)
    slopes = beta[1:]
    xb = design.x @ slopes
    xb_bar = xbar @ slopes
    u = ybar - xb_bar
    sigma_e = math.sqrt(ssr / df_resid)
    if spec.covariance in {"robust", "cluster"}:
        _check_constant_variance(frame, float(v[0, 0]), sigma_e ** 2 * float(
            fit.xtx_inv[0, 0] + mean_x @ fit.xtx_inv[1:, 1:] @ mean_x))
    sigma_u = float(u.std()) if n > 1 else None
    rho = None if sigma_u is None else sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2)
    metrics = {
        "r_squared_within": kernels.squared_correlation(xb - xb_bar[codes], within_y, w),
        "r_squared_between": kernels.squared_correlation(xb_bar, ybar),
        "r_squared_overall": kernels.squared_correlation(xb, y, w),
        "sigma_u": sigma_u, "sigma_e": sigma_e, "rho": rho,
        "corr_u_xb": kernels.correlation(u[codes], xb, w), "rmse": sigma_e,
        "df_resid": df_resid, **sample.structure(),
    }
    tests = {"model": wald_test(beta, v, range(1, k + 1), df_resid=df_inference,
                                label="F test of the slopes")}
    if spec.covariance == "nonrobust" and n > 1:
        xs[:, 1:] = design.x                      # pooled OLS of y on [1, X - xbar], same buffer
        xs[:, 1:] -= mean_x
        pooled = _least_squares(xs, y, w)
        tests["fixed_effects"] = _f_test((float(pooled.ssr) - ssr) / (n - 1), ssr / df_resid,
                                         n - 1, df_resid, "F test that all u_i = 0")
    extra = {"model": "fe", "ssr": ssr, "absorbed_effects": n,
             "k_small_sample": info.get("k_small_sample", k + 1)}
    return build_result(
        frame, terms=["Intercept", *design.terms], params=beta, covariance=v,
        title=_TITLES["fe"], use_t=True, df_inference=df_inference, df_resid=df_resid,
        metrics=metrics, fitted=beta[0] + xb, solver="qr_within", inference=info,
        tests=tests, extra=extra, categories=design.categories, nobs=nobs,
        provenance={"model": "fe"},
    )


# ---- pooled OLS ---------------------------------------------------------------------


def _fit_pooled(sample: PanelSample) -> ResultBundle:
    frame, spec, w = sample.frame, sample.spec, sample.weights
    design = frame.drop_collinear(frame.design(), w)
    y = frame.numeric(spec.outcome)
    means = center_slopes(design.x, w)
    fit = _least_squares(design.x, y, w, drop_collinear=False)
    nobs, kk = sample.nobs, design.x.shape[1]
    _require_df(nobs - kk, "The pooled regression")
    metrics = {**_ols_metrics(y, float(fit.ssr), w, nobs, kk, spec.outcome, "the pooled sample"),
               **sample.structure()}
    v, info, df_inference = covariance(sample, x=design.x, resid=fit.resid, bread=fit.xtx_inv,
                                       nobs=nobs, k=kk, df_resid=nobs - kk, weights=w,
                                       means=means)
    beta = uncenter_coefficients(fit.beta, means)
    tests = {"model": wald_test(beta, v, range(1, kk), df_resid=df_inference,
                                label="F test of the slopes")}
    return build_result(
        frame, terms=design.terms, params=beta, covariance=v, title=_TITLES["pooled"],
        use_t=True, df_inference=df_inference, df_resid=nobs - kk, metrics=metrics,
        fitted=fit.fitted, solver="qr", inference=info, tests=tests,
        extra={"model": "pooled", "ssr": float(fit.ssr)}, categories=design.categories,
        nobs=nobs, provenance={"model": "pooled"},
    )


# ---- between ------------------------------------------------------------------------


def _fit_be(sample: PanelSample) -> ResultBundle:
    frame, spec, codes, n = sample.frame, sample.spec, sample.codes, sample.n
    design = frame.design(intercept=False)
    y = frame.numeric(spec.outcome)
    xbar, ybar = group_means(design.x, codes, n), group_means(y, codes, n)
    between = Design(torch.cat([_ones(n), xbar], dim=1), ["Intercept", *design.terms],
                     design.categories, True)
    sizes = sample.sizes if frame.option("wls") else None
    between = frame.drop_collinear(between, sizes)
    means = center_slopes(between.x, sizes)
    fit = _least_squares(between.x, ybar, sizes, drop_collinear=False)
    kk = between.x.shape[1]
    df_resid = n - kk
    _require_df(df_resid, "The between regression")
    ssr = float(fit.ssr)
    tss = _sum_squares(ybar - _mean(ybar, sizes), sizes)
    require_residual_variation(ssr, tss, _sum_squares(ybar, sizes), spec.outcome,
                               "the panel means")
    v, info, df_inference = covariance(sample, x=between.x, resid=fit.resid, bread=fit.xtx_inv,
                                       nobs=n, k=kk, df_resid=df_resid, weights=sizes,
                                       means=means, robust="HC1", panel_level=True)
    beta = uncenter_coefficients(fit.beta, means)
    kept = _slope_columns(design, between.terms[1:])
    x, slopes = design.x[:, kept], beta[1:]
    xb = x @ slopes
    metrics = {
        "r_squared_between": 1 - ssr / tss,
        "r_squared_within": kernels.squared_correlation(
            (x - xbar[:, kept][codes]) @ slopes, y - ybar[codes]),
        "r_squared_overall": kernels.squared_correlation(xb, y),
        "rmse": math.sqrt(ssr / df_resid), "df_resid": df_resid, **sample.structure(),
    }
    tests = {"model": wald_test(beta, v, range(1, kk), df_resid=df_inference,
                                label="F test of the slopes")}
    extra = {"model": "be", "ssr": ssr, "weighted_by_panel_length": sizes is not None}
    return build_result(
        frame, terms=between.terms, params=beta, covariance=v, title=_TITLES["be"],
        use_t=True, df_inference=df_inference, df_resid=df_resid, metrics=metrics,
        fitted=beta[0] + xb, solver="qr_between", inference=info, tests=tests, extra=extra,
        categories=between.categories, nobs=sample.nobs, provenance={"model": "be"},
    )


# ---- first differences --------------------------------------------------------------


def _fit_fd(frame: ModelFrame) -> ResultBundle:
    spec = frame.spec
    codes, _ = frame.codes(spec.panel)
    time = frame.time_index()
    design = frame.design(intercept=False)
    y = frame.numeric(spec.outcome)
    consecutive = (codes[1:] == codes[:-1]) & (time[1:] - time[:-1] == 1)
    valid = torch.cat([torch.zeros(1, dtype=torch.bool), consecutive])
    kept_rows = int(valid.sum())
    if kept_rows == 0:
        raise AnalysisError("empty_sample", "No panel has two consecutive periods; first "
                            "differences cannot be formed.")
    dx = (design.x[1:] - design.x[:-1])[consecutive]
    dy = (y[1:] - y[:-1])[consecutive]
    levels = y[1:][consecutive]                    # rounding of dy scales with the level of y
    frame.restrict(valid, f"Dropped {frame.n - kept_rows} observation(s) without an observation "
                          "in the previous period: first differences use consecutive periods.")
    sample = panel_sample(frame, constant_weights=False)
    w = sample.weights
    differenced = frame.drop_collinear(
        Design(torch.cat([_ones(kept_rows), dx], dim=1), ["Intercept", *design.terms],
               design.categories, True), w)
    means = center_slopes(differenced.x, w)
    fit = _least_squares(differenced.x, dy, w, drop_collinear=False)
    nobs, kk = sample.nobs, differenced.x.shape[1]
    _require_df(nobs - kk, "The first-difference regression")
    metrics = {**_ols_metrics(dy, float(fit.ssr), w, nobs, kk, spec.outcome,
                              "first differences", _sum_squares(levels, w)),
               **sample.structure()}
    v, info, df_inference = covariance(sample, x=differenced.x, resid=fit.resid,
                                       bread=fit.xtx_inv, nobs=nobs, k=kk, df_resid=nobs - kk,
                                       weights=w, means=means)
    beta = uncenter_coefficients(fit.beta, means)
    tests = {"model": wald_test(beta, v, range(1, kk), df_resid=df_inference,
                                label="F test of the slopes")}
    extra = {"model": "fd", "ssr": float(fit.ssr), "differenced_observations": kept_rows,
             "dropped_for_differencing": len(valid) - kept_rows}
    return build_result(
        frame, terms=differenced.terms, params=beta, covariance=v, title=_TITLES["fd"],
        use_t=True, df_inference=df_inference, df_resid=nobs - kk, metrics=metrics,
        fitted=fit.fitted, observed=dy, solver="qr_differences", inference=info, tests=tests,
        extra=extra, categories=differenced.categories, nobs=nobs,
        provenance={"model": "fd", "residual_definition": "differenced outcome minus fitted"},
    )


# ---- random effects (GLS) -----------------------------------------------------------


def _theta_summary(theta: Tensor) -> dict[str, float]:
    quantiles = torch.quantile(theta, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return {"min": float(theta.min()), "p5": float(quantiles[0]), "median": float(quantiles[1]),
            "p95": float(quantiles[2]), "max": float(theta.max())}


def _fit_re(sample: PanelSample, design: Design | None = None) -> ResultBundle:
    frame, spec, codes, nobs = sample.frame, sample.spec, sample.codes, sample.nobs
    design = frame.drop_collinear(frame.design() if design is None else design)
    y = frame.numeric(spec.outcome)
    # Centered slopes: the GLS transform of [1, X - m] spans the same space as that of
    # [1, X] (the offset m maps onto the transformed constant 1 - theta_i), so this is an
    # exact reparameterization, undone below.
    means = center_slopes(design.x)
    gls = random_effects_gls(sample, design, y, sa=bool(frame.option("sa")))
    kk = design.x.shape[1]
    _require_df(nobs - kk, "The random-effects regression")
    v, info, _ = covariance(sample, x=gls.xs, resid=gls.resid, bread=gls.xtx_inv, nobs=nobs,
                            k=kk, df_resid=nobs - kk, weights=None, means=means, nested=True)
    beta = uncenter_coefficients(gls.beta, means)
    info["df_inference"] = None
    if spec.covariance == "nonrobust":
        info["correction"] = ("GLS: rmse^2 (X*'X*)^-1 with rmse^2 = RSS*/(N-K) of the "
                              "transformed regression (Stata's e(rmse))")
    x, slopes = design.x[:, 1:], beta[1:]
    balanced = bool((sample.sizes == sample.sizes[0]).all())
    sigma_u, sigma_e = math.sqrt(gls.sigma_u2), math.sqrt(gls.sigma_e2)
    metrics = {
        "r_squared_within": kernels.squared_correlation((x - gls.means[codes]) @ slopes,
                                                        y - gls.ybar[codes]),
        "r_squared_between": kernels.squared_correlation(gls.means @ slopes, gls.ybar),
        "r_squared_overall": kernels.squared_correlation(x @ slopes, y),
        "sigma_u": sigma_u, "sigma_e": sigma_e, "rho": gls.sigma_u2 / (gls.sigma_u2 + gls.sigma_e2),
        "theta": float(gls.theta[0]) if balanced else None,
        "rmse": math.sqrt(gls.ssr / (nobs - kk)), **sample.structure(),
    }
    lm, p_value = kernel_call(kernels.breusch_pagan_lm, gls.pooled_resid, codes, sample.n)
    tests = {
        "model": wald_test(beta, v, range(1, kk), label="Wald chi2 test of the slopes"),
        "breusch_pagan": {"statistic": lm, "df": 1, "p_value": p_value,
                          "distribution": "chibar2",
                          "label": "Breusch-Pagan LM test of sigma_u = 0 (chibar2(01): chi2(1) "
                                   "tail halved)"},
    }
    extra = {
        "model": "re",
        "variance_components": {
            "method": gls.method, "sigma_u_squared": gls.sigma_u2, "sigma_e_squared": gls.sigma_e2,
            "ssr_within": gls.ssr_within, "df_within": gls.df_within,
            "within_regressors": gls.k_within, "ssr_between": gls.ssr_between,
            "df_between": gls.df_between, "t_bar_harmonic": gls.t_bar,
        },
        "theta": _theta_summary(gls.theta), "ssr_transformed": gls.ssr,
    }
    return build_result(
        frame, terms=design.terms, params=beta, covariance=v, title=_TITLES["re"],
        use_t=False, df_resid=nobs - kk, metrics=metrics, fitted=design.x @ gls.beta,
        solver="qr_gls", inference=info, tests=tests, extra=extra, categories=design.categories,
        nobs=nobs, provenance={"model": "re"},
    )


# ---- convenience ---------------------------------------------------------------------


def xtreg(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None = None,
          model: str = "fe", covariance: str | None = None,
          cluster: str | Sequence[str] | None = None, weights: str | None = None,
          weight_type: str | None = None,
          categorical: Sequence[str] | None = None, lags: int | None = None,
          kernel: str | None = None, sa: bool = False, wls: bool = False, missing: str = "raise",
          alpha: float = 0.05) -> ResultBundle:
    """Panel-data linear regression (Stata's ``xtreg``; ``xtscc`` for Driscoll-Kraay).

    The model is y_it = x_it'b + u_i + e_it for panels i = 1..n observed in
    T_i periods. ``model`` selects the estimator:

    - ``'fe'`` (default): the within estimator (``xtreg, fe``). Panel effects u_i
      are removed by subtracting panel means; regressors that do not vary
      within panels are omitted (recorded in ``warnings`` and
      ``provenance['omitted_terms']``). The ``Intercept`` is Stata's ``_cons``
      (grand mean added back). sigma_e^2 = RSS/(N - n - k), sigma_u is the
      standard deviation of u_i = ybar_i - xbar_i'b over the n panels,
      rho = sigma_u^2/(sigma_u^2 + sigma_e^2) and corr_u_xb = corr(u_i, x_it'b)
      over the observations (Stata's ``e(corr)``). An outcome without
      within-panel variation raises ``no_within_variation``.
    - ``'re'``: Swamy-Arora feasible GLS (``xtreg, re``): sigma_e^2 from the
      within regression, sigma_u^2 = max(0, RSS_b/(n - K) - sigma_e^2/T_bar)
      from the between regression with the harmonic mean T_bar; ``sa=True``
      uses the Baltagi-Chang unbalanced formula instead. The conventional
      covariance is rmse^2 (X*'X*)^-1 from the transformed regression (Stata's
      ``e(rmse)``, also ``metrics['rmse']``). Coefficients use z inference;
      ``tests['breusch_pagan']`` is Stata's ``xttest0``.
    - ``'be'``: between regression of panel means (``xtreg, be``); ``wls=True``
      weights panels by T_i.
    - ``'fd'``: first differences over consecutive periods with a drift term
      (``regress D.y D.x``); needs ``time``.
    - ``'mle'``: Gaussian random-effects maximum likelihood (``xtreg, mle``)
      with ancillary terms ``/sigma_u`` and ``/sigma_e`` (confidence intervals
      from the ln sigma parameterization, as Stata prints), LR tests of the
      slopes and of sigma_u = 0 (chibar2(01)).
    - ``'pooled'``: pooled OLS with the panel covariances below (the base of
      ``xtscc``).

    Parameters: ``data`` a DataFrame (or columns/records); ``y`` the outcome;
    ``x`` the list of regressors; ``panel`` the panel identifier; ``time`` the
    integer (or datetime) period, required for ``'fd'`` and Driscoll-Kraay;
    ``categorical`` regressors to treatment-code; ``weights``/``weight_type``
    an ``aweight``, ``fweight`` or ``pweight`` column for ``'fe'`` (constant
    within panel), ``'fd'`` and ``'pooled'`` (pweights need a robust or cluster
    covariance); ``'re'``, ``'be'`` and ``'mle'`` take no weights, as in Stata
    (``unsupported_weights``); ``missing`` ``'raise'`` or ``'drop'``; ``alpha``
    the test size.

    Covariance (``covariance``): ``'nonrobust'`` (default) is the classical
    one with Stata's degrees of freedom (N - n - k for fe); ``'robust'`` IS
    Stata's ``vce(robust)`` for xtreg, which clusters on the panel variable
    (CR1: G/(G-1) (N-1)/(N-K) with K = k + 1, t tests with G - 1 df) — for
    ``'be'`` it is HC1 on the panel-mean regression; ``'cluster'`` with
    ``cluster`` (one or two columns) is the same sandwich on those clusters.
    For ``'fe'`` and ``'re'`` every cluster column must nest the panels, as
    Stata requires (``cluster_not_nested`` otherwise); ``'pooled'`` and
    ``'fd'`` accept any cluster column. ``'driscoll_kraay'`` (fe and pooled,
    needs ``time``) is Hoechle's xtscc: a Bartlett-kernel HAC of the
    cross-sectional score sums with ``lags`` (default floor(4 (T/100)^(2/9)))
    and ``kernel``, times T/(T-1) (N-1)/(N-K), inference with T - 1 df.

    The result's ``metrics`` hold the model's R-squared definitions, sigma_u,
    sigma_e, rho, n_groups and t_min/t_avg/t_max; ``tests`` hold the model
    test and the fixed-effects, Breusch-Pagan or LR tests; ``extra`` the
    variance components and small bookkeeping.

    Example::

        import openecon as oe
        robust = oe.xtreg(data=df, y="lwage", x=["exp", "exp2", "union"], panel="id",
                          time="year", covariance="robust")
        print(robust.summary())
        fe = oe.xtreg(data=df, y="lwage", x=["exp", "exp2", "union"], panel="id")
        re = oe.xtreg(data=df, y="lwage", x=["exp", "exp2", "union"], panel="id",
                      model="re")
        print(oe.hausman(fe, re, sigmamore=True))   # needs the conventional covariances
    """
    from openecon.analysis import fit

    spec = make_spec(
        "xtreg", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
        covariance=covariance, cluster=cluster, categorical=column_list(categorical, "categorical"),
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        options={"model": model, "lags": lags, "kernel": kernel, "sa": True if sa else None,
                 "wls": True if wls else None},
    )
    return fit(spec, data=data)
