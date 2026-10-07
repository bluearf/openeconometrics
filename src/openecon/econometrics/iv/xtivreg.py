"""Panel-data instrumental-variables regression: Stata's ``xtivreg``.

Model: ``y_it = x_it'b + u_i + e_it`` with some regressors endogenous and
excluded instruments ``z_it``; N observations in n panels of T_i periods, k
slopes and K = k + 1 coefficients with the constant. Every estimator is 2SLS
(``kernels``: two QR projections, no projection matrix) on transformed data:

fe   Within transformation of y, regressors and instruments (panel means
     removed, grand means added back so that the reported ``Intercept`` is
     Stata's ``_cons = ybar - xbar'b``). ``sigma_e^2 = RSS/(N - n - k)``;
     ``sigma_u`` is the standard deviation of ``u_i = ybar_i - xbar_i'b`` over
     panels; regressors and instruments without within variation are omitted.
fd   First differences over consecutive periods (needs ``time``), 2SLS with a
     constant; ``sigma^2 = RSS/(N_d - K)``.
be   2SLS on the panel means with a constant; ``sigma^2 = RSS/(n - K)``.
re   G2SLS (Balestra and Varadharajan-Krishnakumar 1987). Variance components
     from the within and between 2SLS residuals,

         sigma_e^2 = RSS_w / (N - n - k_w),
         sigma_u^2 = max(0, {SSR*_b - (n - K_b) sigma_e^2} / (N - r)),
         SSR*_b = sum_i T_i ubar_i^2,
         r = trace{(sum_i T_i z_i z_i')^-1 sum_i T_i^2 z_i z_i'},

     the Swamy-Arora estimators adapted to unbalanced panels (Baltagi and Chang
     2000; xtivreg's default): ``ubar_i`` are the residuals of the between 2SLS
     in which each panel mean appears ``T_i`` times and ``z_i`` the panel means
     of its regressors; with balanced panels
     ``sigma_u^2 = RSS_b/(n - K_b) - sigma_e^2/T``. k_w and K_b count the
     columns with within / between variation.
     ``theta_i = 1 - sqrt(sigma_e^2 / (T_i sigma_u^2 + sigma_e^2))``; then 2SLS of
     ``y - theta_i ybar_i`` on ``x - theta_i xbar_i`` (constant: ``1 - theta_i``)
     with instruments ``z - theta_i zbar_i``. ``ec2sls=True`` uses Baltagi's
     instrument set instead: the within-transformed instruments and their panel
     means separately. Conventional covariance: that of the 2SLS regression on
     the transformed data, ``RSS*/(N - K) (X*'P X*)^-1`` (as ``xtreg, re`` takes
     the OLS covariance of its transformed regression).

Covariance: ``nonrobust`` as above; ``robust`` IS clustering on the panel (the
between model: HC1 on the panel-mean regression); ``cluster`` on the given
column. Sandwiches carry ``G/(G-1) (N-1)/(N-K)`` (K = k + 1; the absorbed
panel effects are counted only when they are not nested in the clusters).

Statistics: z and Wald chi2 by default for every model, as Stata's xtivreg
prints them; ``small=True`` gives t and F with the residual degrees of freedom
(``G - 1`` with a cluster covariance). The covariance is the same either way.
R-squared: the model's own transformation reports the R-squared of its 2SLS
regression, ``1 - RSS/TSS`` (within for fe, between for be, the differenced
equation for fd; it can be negative, where Stata prints a missing value); the
other two are squared correlations between ``x'b`` and ``y`` (levels, panel
means or within deviations), as for ``xtreg``. ``corr_u_xb`` (fe) is the
correlation of ``u_i`` with ``x_it'b`` over the N observations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test,
)
from openecon.econometrics.iv import kernels
from openecon.econometrics.iv.common import (
    Blocks, Setting, check_fit, covariance, estimate, role_designs, screen, sum_of_squares,
)
from openecon.engines.absorb import absorbed_degrees_of_freedom, demean, group_means
from openecon.engines.distributions import f_sf
from openecon.engines.linalg import cholesky_solve, collinear_columns, weighted_crossprod
from openecon.models import ModelSpec, ResultBundle

TITLES = {"fe": "Fixed-effects (within) IV regression",
          "fd": "First-difference IV regression",
          "be": "Between-effects IV regression",
          "re": "G2SLS random-effects IV regression"}


def _ones(n: int) -> Tensor:
    return torch.ones((n, 1), dtype=torch.float64)


def _squared_correlation(a: Tensor, b: Tensor) -> float | None:
    """Squared Pearson correlation; None when either vector is constant."""
    if a.numel() < 2:
        return None
    da, db = a - a.mean(), b - b.mean()
    saa, sbb, sab = float(da @ da), float(db @ db), float(da @ db)
    scale = max(saa, sbb)
    if scale <= 0 or min(saa, sbb) <= 1e-28 * scale:
        return None
    return min(1.0, sab * sab / (saa * sbb))


def _structure(sizes: Tensor, nobs: int) -> dict[str, float]:
    return {"n_groups": sizes.numel(), "t_min": float(sizes.min()),
            "t_avg": nobs / sizes.numel(), "t_max": float(sizes.max())}


def _first_rows(codes: Tensor) -> Tensor:
    starts = (codes[1:] != codes[:-1]).nonzero().flatten() + 1
    return torch.cat([torch.zeros(1, dtype=torch.int64), starts])


def _setting(frame: ModelFrame, nobs: int, codes: Tensor, n: int, *, absorbed: int = 0,
             panel_level: bool = False) -> Setting:
    """Covariance setting; ``robust`` clusters on the panel (HC1 for the between model).

    ``absorbed`` is the number of panel effects partialled out beyond the
    constant (fe); they are dropped from K when nested in the clusters.
    """
    spec = frame.spec
    if spec.covariance == "nonrobust" or (spec.covariance == "robust" and panel_level):
        return Setting(nobs, None, spec.covariance, True, absorbed=absorbed)
    if spec.covariance == "robust":
        if n < 2:
            raise AnalysisError("insufficient_clusters", "The robust covariance clusters on "
                                "the panel variable and needs at least two panels.")
        if n < 30:
            frame.warn(f"Only {n} panels; cluster-robust inference on the panel variable may "
                       "be unreliable.")
        return Setting(nobs, None, "cluster", True, clusters=[(codes, n)],
                       cluster_names=[spec.panel])
    names = registry.cluster_columns(spec)
    clusters = frame.cluster_dimensions()
    if panel_level:
        level = []
        for (cluster, count), name in zip(clusters, names, strict=True):
            low = torch.full((n,), count, dtype=torch.int64).scatter_reduce_(
                0, codes, cluster, "amin")
            high = torch.full((n,), -1, dtype=torch.int64).scatter_reduce_(
                0, codes, cluster, "amax")
            if bool((low != high).any()):
                raise AnalysisError("cluster_varies_within_panel", f"Cluster column '{name}' "
                                    "must be constant within each panel for the between "
                                    "estimator.")
            level.append((low, count))
        return Setting(nobs, None, "cluster", True, clusters=level, cluster_names=names)
    if absorbed:
        nested = any(kernel_call(absorbed_degrees_of_freedom, [(codes, n)], cluster).nested[0]
                     for cluster in clusters)
        if nested:
            absorbed = 0
        else:
            frame.warn("The panel effects are not nested within the cluster variable: the "
                       f"{n} absorbed effects are counted in the small-sample factor.")
    return Setting(nobs, None, "cluster", True, absorbed=absorbed, clusters=clusters,
                   cluster_names=names)


def _relabel(frame: ModelFrame, info: dict[str, Any], small: bool) -> dict[str, Any]:
    info = dict(info)
    if frame.spec.covariance == "robust" and info.get("covariance") == "cluster":
        info["covariance"] = "robust"
        info["correction"] = "vce(robust) = vce(cluster panel); " + str(info["correction"])
    info["small"] = small
    info["df_inference"] = info.get("df_inference") if small else None
    return info


def _model_test(beta: Tensor, v: Tensor, terms: list[str], df: float | None,
                small: bool) -> dict[str, Any]:
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    label = "F test of the slopes" if small else "Wald chi2 test of the slopes"
    return wald_test(beta, v, slopes, df_resid=df if small else None, label=label)


def _flag_constant(frame: ModelFrame, robust: float, classical: float) -> None:
    """Flag a cluster-robust variance of the fe constant that is zero up to rounding.

    Within residuals sum to zero in every panel, hence in every cluster that nests the
    panels, so the sandwich variance of Stata's _cons is exactly ``xbar'V_b xbar``. With
    regressors of (near-)zero grand mean that is rounding noise: it is reported, with a
    warning; an exact zero cannot be reported at all (the rule of ``xtreg, fe``).
    """
    if robust <= 0:
        raise AnalysisError("zero_variance_constant", "The cluster-robust variance of the "
                            "Intercept is exactly zero: within residuals sum to zero in every "
                            "panel and the regressors have zero grand means. Use "
                            "covariance='nonrobust' or shift a regressor; the slopes are "
                            "unaffected.")
    if robust <= 1e-20 * classical:
        frame.warn("The cluster-robust variance of the Intercept is zero up to rounding: it "
                   "equals xbar'V xbar (within residuals sum to zero in every panel) and the "
                   "regressors have near-zero grand means. Its standard error, test statistic "
                   "and p-value are not meaningful; the slopes are unaffected.")


def _require_df(df_resid: float, what: str) -> None:
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations",
                            f"{what} leaves no residual degrees of freedom ({df_resid:g}).")


def fit_xtivreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``xtivreg``: dispatch on ``options['model']``."""
    frame = ModelFrame(spec, data)
    model = frame.option("model")
    if frame.option("ec2sls") and model != "re":
        raise AnalysisError("invalid_spec", "ec2sls=True applies to model='re' only.")
    if model == "fd" and spec.time is None:
        raise AnalysisError("invalid_spec", "model='fd' needs a time column to difference "
                            "consecutive periods.")
    small = bool(frame.option("small"))
    frame.sort_panel()
    return {"fe": _fit_fe, "fd": _fit_fd, "be": _fit_be, "re": _fit_re}[model](frame, small)


# ---- fixed effects ------------------------------------------------------------------


def _fit_fe(frame: ModelFrame, small: bool) -> ResultBundle:
    spec, nobs = frame.spec, frame.n
    codes, n = frame.codes(spec.panel)
    sizes = torch.bincount(codes, minlength=n).to(torch.float64)
    y = frame.numeric(spec.outcome)
    if nobs <= n:
        raise AnalysisError("insufficient_observations", "Every panel has a single observation: "
                            "the within transformation leaves no variation. The fixed-effects "
                            "estimator needs panels observed at least twice.")
    exog, endog, instr = role_designs(frame, intercept=False)
    k1, q = exog.x.shape[1], endog.x.shape[1]
    within = kernel_call(demean, torch.cat([y[:, None], exog.x, endog.x, instr.x], dim=1),
                         [(codes, n)]).values
    blocks = screen(frame, exog, endog, instr, None,
                    transformed=(within[:, 1:1 + k1], within[:, 1 + k1:1 + k1 + q],
                                 within[:, 1 + k1 + q:]))
    terms = ["Intercept", *blocks.terms]
    k = len(terms) - 1
    df_resid = nobs - n - k
    _require_df(df_resid, "The within IV regression")
    # Grand means added back: 2SLS with a constant then reports Stata's _cons.
    x1 = torch.cat([_ones(nobs), blocks.x1 + blocks.exog.x.mean(dim=0)], dim=1)
    system = Blocks(Design(x1, terms[:1 + len(blocks.exog.terms)], blocks.exog.categories, True),
                    blocks.endog, blocks.instr, x1, blocks.x2 + blocks.endog.x.mean(dim=0),
                    blocks.z2 + blocks.instr.x.mean(dim=0), blocks.omitted_instruments)
    setting = _setting(frame, nobs, codes, n, absorbed=n - 1)
    est = estimate(frame, setting, within[:, 0] + y.mean(), system, method="2sls")
    tss_within = sum_of_squares(within[:, 0], None, centered=False)
    check_fit(est.rss, tss_within, df_resid, sum_of_squares(y, None, centered=False))
    if setting.kind == "cluster":
        _flag_constant(frame, float(est.covariance[0, 0]),
                       float(est.tsls.bread[0, 0]) * est.rss / df_resid)
    info = _relabel(frame, est.info, small)
    slopes = est.beta[1:]
    x = torch.cat([blocks.exog.x, blocks.endog.x], dim=1)
    ybar, xb_bar = group_means(y, codes, n), group_means(x, codes, n) @ slopes
    u = ybar - xb_bar
    sigma_e = math.sqrt(est.rss / df_resid)
    sigma_u = float(u.std()) if n > 1 else None
    xb = x @ slopes
    metrics = {
        # The within R-squared is that of the transformed IV regression (it can be negative;
        # Stata then prints a missing value); between and overall are squared correlations.
        "r_squared_within": 1 - est.rss / tss_within,
        "r_squared_between": _squared_correlation(xb_bar, ybar),
        "r_squared_overall": _squared_correlation(xb, y),
        "sigma_u": sigma_u, "sigma_e": sigma_e,
        "rho": None if sigma_u is None else sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2),
        # Stata's e(corr): correlation of u_i and x_it'b over the N observations.
        "corr_u_xb": None if n < 2 else _signed_correlation(u[codes], xb), "rmse": sigma_e,
        "df_resid": df_resid, **_structure(sizes, nobs),
        "n_instruments": len(blocks.instr.terms), "n_endogenous": len(blocks.endog.terms),
    }
    tests = {"model": _model_test(est.beta, est.covariance, terms, est.info.get("df_inference"),
                                  small)}
    if spec.covariance == "nonrobust" and n > 1:
        tests["fixed_effects"] = _fixed_effects_test(frame, y, blocks, est.rss, n, df_resid)
    extra = {"model": "fe", "endogenous": blocks.endog.terms, "instruments": blocks.instr.terms,
             "omitted_instruments": blocks.omitted_instruments, "ssr": est.rss,
             "absorbed_effects": n}
    return build_result(
        frame, terms=terms, params=est.beta, covariance=est.covariance, title=TITLES["fe"],
        use_t=small, df_inference=est.info.get("df_inference") if small else None,
        df_resid=df_resid, metrics=metrics, fitted=est.beta[0] + xb, solver="qr_within_2sls",
        inference=info, tests=tests, extra=extra, categories=blocks.exog.categories,
        provenance={"model": "fe"})


def _signed_correlation(a: Tensor, b: Tensor) -> float | None:
    squared = _squared_correlation(a, b)
    if squared is None:
        return None
    sign = float(((a - a.mean()) * (b - b.mean())).sum())
    return math.copysign(math.sqrt(squared), sign)


def _fixed_effects_test(frame: ModelFrame, y: Tensor, blocks: Blocks, rss: float, n: int,
                        df_resid: float) -> dict[str, Any]:
    """F(n-1, N-n-k) that all u_i = 0: pooled 2SLS against the within 2SLS residuals."""
    label = "F test that all u_i = 0"
    try:
        x1 = torch.cat([_ones(frame.n), blocks.exog.x], dim=1)
        pooled = kernel_call(kernels.k_class, kernel_call(
            kernels.project, y, x1, blocks.endog.x, blocks.instr.x), None, 1.0)
    except AnalysisError:
        return {"statistic": None, "df": n - 1, "df2": df_resid, "p_value": None,
                "distribution": "F", "label": label,
                "note": "the pooled IV regression is not identified"}
    statistic = max(0.0, (pooled.rss - rss) / (n - 1)) / (rss / df_resid)
    return {"statistic": statistic, "df": n - 1, "df2": df_resid,
            "p_value": kernel_call(f_sf, statistic, n - 1, df_resid), "distribution": "F",
            "label": label}


# ---- first differences ----------------------------------------------------------------


def _fit_fd(frame: ModelFrame, small: bool) -> ResultBundle:
    spec = frame.spec
    codes, _ = frame.codes(spec.panel)
    time = frame.time_index()
    y = frame.numeric(spec.outcome)
    exog, endog, instr = role_designs(frame, intercept=False)
    consecutive = (codes[1:] == codes[:-1]) & (time[1:] - time[:-1] == 1)
    valid = torch.cat([torch.zeros(1, dtype=torch.bool), consecutive])
    rows = int(valid.sum())
    if rows == 0:
        raise AnalysisError("empty_sample", "No panel has two consecutive periods; first "
                            "differences cannot be formed.")

    def difference(values: Tensor) -> Tensor:
        return (values[1:] - values[:-1])[consecutive]

    dy = difference(y)
    exog_d = Design(torch.cat([_ones(rows), difference(exog.x)], dim=1),
                    ["Intercept", *exog.terms], exog.categories, True)
    endog_d = Design(difference(endog.x), endog.terms, {}, False)
    instr_d = Design(difference(instr.x), instr.terms, {}, False)
    frame.restrict(valid, f"Dropped {frame.n - rows} observation(s) without an observation in "
                          "the previous period: first differences use consecutive periods.")
    codes, n = frame.codes(spec.panel)
    sizes = torch.bincount(codes, minlength=n).to(torch.float64)
    blocks = screen(frame, exog_d, endog_d, instr_d, None)
    terms = blocks.terms
    kk = len(terms)
    df_resid = rows - kk
    _require_df(df_resid, "The first-difference IV regression")
    setting = _setting(frame, rows, codes, n)
    est = estimate(frame, setting, dy, blocks, method="2sls")
    tss = sum_of_squares(dy, None, centered=True)
    check_fit(est.rss, tss, df_resid, sum_of_squares(dy, None, centered=False))
    info = _relabel(frame, est.info, small)
    r_squared = 1 - est.rss / tss
    metrics = {
        "r_squared": r_squared,
        "adjusted_r_squared": 1 - (1 - r_squared) * (rows - 1) / df_resid,
        "rmse": math.sqrt(est.rss / df_resid), "df_model": kk - 1, "df_resid": df_resid,
        **_structure(sizes, rows), "n_instruments": len(blocks.instr.terms),
        "n_endogenous": len(blocks.endog.terms),
    }
    tests = {"model": _model_test(est.beta, est.covariance, terms, est.info.get("df_inference"),
                                  small)}
    extra = {"model": "fd", "endogenous": blocks.endog.terms, "instruments": blocks.instr.terms,
             "omitted_instruments": blocks.omitted_instruments, "ssr": est.rss,
             "differenced_observations": rows, "dropped_for_differencing": len(valid) - rows}
    return build_result(
        frame, terms=terms, params=est.beta, covariance=est.covariance, title=TITLES["fd"],
        use_t=small, df_inference=est.info.get("df_inference") if small else None,
        df_resid=df_resid, metrics=metrics, fitted=dy - est.resid, observed=dy,
        solver="qr_differences_2sls", inference=info, tests=tests, extra=extra,
        categories=blocks.exog.categories, nobs=rows,
        provenance={"model": "fd", "residual_definition": "differenced outcome minus fitted"})


# ---- between ------------------------------------------------------------------------


def _means(values: Tensor, codes: Tensor, n: int) -> Tensor:
    if values.ndim == 2 and values.shape[1] == 0:
        return torch.empty((n, 0), dtype=torch.float64)
    return group_means(values, codes, n)


def _fit_be(frame: ModelFrame, small: bool) -> ResultBundle:
    spec, nobs = frame.spec, frame.n
    codes, n = frame.codes(spec.panel)
    sizes = torch.bincount(codes, minlength=n).to(torch.float64)
    y = frame.numeric(spec.outcome)
    exog, endog, instr = role_designs(frame, intercept=False)
    ybar = group_means(y, codes, n)
    blocks = screen(
        frame,
        Design(torch.cat([_ones(n), _means(exog.x, codes, n)], dim=1),
               ["Intercept", *exog.terms], exog.categories, True),
        Design(_means(endog.x, codes, n), endog.terms, {}, False),
        Design(_means(instr.x, codes, n), instr.terms, {}, False), None)
    terms = blocks.terms
    kk = len(terms)
    df_resid = n - kk
    _require_df(df_resid, "The between IV regression")
    setting = _setting(frame, n, codes, n, panel_level=True)
    est = estimate(frame, setting, ybar, blocks, method="2sls")
    tss_between = sum_of_squares(ybar, None, centered=True)
    check_fit(est.rss, tss_between, df_resid, sum_of_squares(ybar, None, centered=False))
    info = _relabel(frame, est.info, small)
    # Level columns of the kept terms, in coefficient order.
    level = torch.cat([exog.x[:, [exog.terms.index(t) for t in blocks.exog.terms[1:]]],
                       endog.x[:, [endog.terms.index(t) for t in blocks.endog.terms]]], dim=1)
    slopes = est.beta[1:]
    xb, xb_bar = level @ slopes, torch.cat([blocks.x1[:, 1:], blocks.x2], dim=1) @ slopes
    metrics = {
        "r_squared_within": _squared_correlation(xb - xb_bar[codes], y - ybar[codes]),
        # The between R-squared is that of the IV regression on the panel means.
        "r_squared_between": 1 - est.rss / tss_between,
        "r_squared_overall": _squared_correlation(xb, y),
        "rmse": math.sqrt(est.rss / df_resid), "df_resid": df_resid, **_structure(sizes, nobs),
        "n_instruments": len(blocks.instr.terms), "n_endogenous": len(blocks.endog.terms),
    }
    tests = {"model": _model_test(est.beta, est.covariance, terms, est.info.get("df_inference"),
                                  small)}
    extra = {"model": "be", "endogenous": blocks.endog.terms, "instruments": blocks.instr.terms,
             "omitted_instruments": blocks.omitted_instruments, "ssr": est.rss}
    return build_result(
        frame, terms=terms, params=est.beta, covariance=est.covariance, title=TITLES["be"],
        use_t=small, df_inference=est.info.get("df_inference") if small else None,
        df_resid=df_resid, metrics=metrics, fitted=est.beta[0] + xb, solver="qr_between_2sls",
        inference=info, tests=tests, extra=extra, categories=blocks.exog.categories,
        provenance={"model": "be"})


# ---- random effects (G2SLS / EC2SLS) --------------------------------------------------


def _added(base: Tensor, block: Tensor, reference: Tensor | None = None) -> list[int]:
    """Columns of ``block`` that add to ``base``: nonzero and not collinear (silent screen).

    ``reference`` holds the columns before a transformation; a transformed
    column whose sum of squares fell below 1e-13 of its original centered one
    has no variation left.
    """
    if block.shape[1] == 0:
        return []
    alive = block.square().sum(dim=0) > 0
    if reference is not None:
        centered = (reference - reference.mean(dim=0)).square().sum(dim=0)
        alive &= block.square().sum(dim=0) > 1e-13 * centered
    index = alive.nonzero().flatten().tolist()
    if not index:
        return []
    kept, _ = kernel_call(collinear_columns, torch.cat([base, block[:, index]], dim=1))
    return [index[i - base.shape[1]] for i in kept if i >= base.shape[1]]


def _within_variance(y: Tensor, blocks: Blocks, codes: Tensor, n: int) -> tuple[float, int, float]:
    """(sigma_e^2, k_w, RSS_w) from the within 2SLS of the time-varying columns."""
    x1, x2, z2 = blocks.x1[:, 1:], blocks.x2, blocks.z2
    k1, q = x1.shape[1], x2.shape[1]
    within = kernel_call(demean, torch.cat([y[:, None], x1, x2, z2], dim=1), [(codes, n)]).values
    yw, x1w = within[:, 0], within[:, 1:1 + k1]
    x1w = x1w[:, _added(x1w[:, :0], x1w, x1)]
    x2w = within[:, 1 + k1:1 + k1 + q]
    x2w = x2w[:, _added(x1w, x2w, x2)]
    z2w = within[:, 1 + k1 + q:]
    z2w = z2w[:, _added(x1w, z2w, z2)]
    k_w = x1w.shape[1] + x2w.shape[1]
    df = y.numel() - n - k_w
    _require_df(df, "The within IV regression behind sigma_e")
    if x2w.shape[1] == 0:
        rss = float(kernel_call(kernels.solve, x1w, yw, None, "within regressors").ssr)
    elif z2w.shape[1] < x2w.shape[1]:
        raise AnalysisError(
            "underidentified", "The within IV regression that estimates sigma_e is not "
            "identified: the instruments have too little variation within panels. Add "
            "time-varying instruments or use model='be'.")
    else:
        rss = kernel_call(kernels.k_class, kernel_call(kernels.project, yw, x1w, x2w, z2w),
                          None, 1.0).rss
    return rss / df, k_w, rss


def _between_fit(ybar: Tensor, xbar: Tensor, zbar: Tensor, x: Tensor, z2: Tensor, k1: int,
                 sizes: Tensor) -> tuple[float, int, float]:
    """(SSR*_b, K_b, r) of the between 2SLS regression behind sigma_u.

    The between estimator here is 2SLS on the panel means "in which each average
    appears T_i times" (the between transform P applied to the N observations, i.e.
    ``T_i``-weighted 2SLS on the n means). ``SSR*_b = sum_i T_i ubar_i^2`` is its
    residual sum of squares and
    ``r = trace{(Zb'Zb)^-1 Zb'Z_mu Z_mu'Zb} = trace{(sum_i T_i z_i z_i')^-1 sum_i T_i^2 z_i z_i'}``
    with ``z_i`` the panel means of the K_b regressors, so that
    ``E[SSR*_b] = (n - K_b) sigma_e^2 + (N - r) sigma_u^2`` when the regressors are
    exogenous (Swamy-Arora for unbalanced panels, Baltagi and Chang; ``r = T K_b``
    when balanced).

    Columns without between variation (a time trend or period dummies in a
    balanced panel) are collinear with the constant at the panel level and are
    left out of this auxiliary regression, silently, as the within regression
    leaves out time-invariant columns; ``K_b`` counts the columns kept.
    """
    n = ybar.numel()
    base = _ones(n)

    def centered(block: Tensor) -> Tensor:
        return block - block.mean(dim=0)

    x1b = centered(xbar[:, 1:k1])
    x1b = torch.cat([base, x1b[:, _added(base, x1b, x[:, 1:k1])]], dim=1)
    x2b = centered(xbar[:, k1:])
    x2b = x2b[:, _added(x1b, x2b, x[:, k1:])]
    zb = centered(zbar)
    zb = zb[:, _added(x1b, zb, z2)]
    k_b = x1b.shape[1] + x2b.shape[1]
    _require_df(n - k_b, "The between IV regression behind sigma_u")
    if x2b.shape[1] == 0:
        ssr = float(kernel_call(kernels.solve, x1b, ybar, sizes, "between regressors").ssr)
    elif zb.shape[1] < x2b.shape[1]:
        raise AnalysisError(
            "underidentified", "The between IV regression that estimates sigma_u is not "
            "identified: too few instruments vary between panels. Add instruments with "
            "between-panel variation or use model='fe'.")
    else:
        try:
            ssr = kernel_call(kernels.k_class, kernel_call(
                kernels.project, ybar, x1b, x2b, zb, sizes), sizes, 1.0).rss
        except AnalysisError as exc:
            raise AnalysisError("underidentified", "The between IV regression that estimates "
                                f"sigma_u could not be computed: {exc}") from exc
    regressors = torch.cat([x1b, x2b], dim=1)
    gram = weighted_crossprod(regressors, sizes)
    trace = float(kernel_call(cholesky_solve, gram, weighted_crossprod(regressors, sizes * sizes),
                              code="singular_design", what="between regressors").trace())
    return ssr, k_b, trace


def _theta_summary(theta: Tensor) -> dict[str, float]:
    quantiles = torch.quantile(theta, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return {"min": float(theta.min()), "p5": float(quantiles[0]), "median": float(quantiles[1]),
            "p95": float(quantiles[2]), "max": float(theta.max())}


def _fit_re(frame: ModelFrame, small: bool) -> ResultBundle:
    spec, nobs = frame.spec, frame.n
    codes, n = frame.codes(spec.panel)
    sizes = torch.bincount(codes, minlength=n).to(torch.float64)
    y = frame.numeric(spec.outcome)
    tss = sum_of_squares(y, None, centered=True)
    if tss <= 0:
        raise AnalysisError("constant_outcome", "The outcome has no variation in the estimation "
                            "sample; nothing can be explained.")
    blocks = screen(frame, *role_designs(frame, intercept=True), None)
    terms = blocks.terms
    kk, k1 = len(terms), blocks.x1.shape[1]
    _require_df(nobs - kk, "The random-effects IV regression")
    ec2sls = bool(frame.option("ec2sls"))
    x = torch.cat([blocks.x1, blocks.x2], dim=1)
    xbar, zbar, ybar = group_means(x, codes, n), group_means(blocks.z2, codes, n), \
        group_means(y, codes, n)
    sigma_e2, k_within, rss_within = _within_variance(y, blocks, codes, n)
    if rss_within <= 1e-24 * sum_of_squares(y, None, centered=False):
        raise AnalysisError("perfect_fit", "The within IV regression fits the outcome exactly "
                            "(sigma_e = 0), so the random-effects transformation is undefined. "
                            "The outcome has no variation within panels beyond the regressors; "
                            "use model='be'.")
    ssr_between, k_between, trace = _between_fit(ybar, xbar, zbar, x, blocks.z2, k1, sizes)
    if nobs - trace <= 0:
        raise AnalysisError("insufficient_observations", "The panels are too few or too short "
                            "to estimate sigma_u (N - r <= 0 in the Swamy-Arora formula).")
    sigma_u2 = max(0.0, (ssr_between - (n - k_between) * sigma_e2) / (nobs - trace))
    if sigma_u2 == 0:
        frame.warn("The estimated sigma_u is zero: the random-effects IV estimator reduces to "
                   "pooled 2SLS.")
    theta = 1 - torch.sqrt(sigma_e2 / (sizes * sigma_u2 + sigma_e2))
    shrink = theta[codes][:, None]
    ys, xs = y - shrink[:, 0] * ybar[codes], x - shrink * xbar[codes]
    if ec2sls:
        # Baltagi: within-transformed and between-transformed instruments, separately.
        levels = torch.cat([blocks.x1[:, 1:], blocks.z2], dim=1)
        means = torch.cat([xbar[:, 1:k1], zbar], dim=1)[codes]
        candidates = torch.cat([levels - means, means], dim=1)
        base = _ones(nobs)
        # A within-transformed column counts only if variation is left relative to its level.
        kept = _added(base, candidates, torch.cat([levels, means], dim=1))
        used = torch.cat([base, candidates[:, kept]], dim=1)
    else:
        used = torch.cat([xs[:, :k1], blocks.z2 - shrink * zbar[codes]], dim=1)
    empty = xs[:, :0]
    proj = kernel_call(kernels.project, ys, empty, xs, used)
    fit = kernel_call(kernels.k_class, proj, None, 1.0)
    if spec.covariance == "nonrobust":
        # The conventional VCE of the 2SLS regression on the transformed data, as xtreg, re
        # takes the OLS VCE of its transformed regression.
        v = fit.bread * (fit.rss / (nobs - kk))
        info = {"covariance": "nonrobust", "df_inference": nobs - kk,
                "correction": "2SLS on the GLS-transformed data: RSS*/(N-K) (X*'P X*)^-1"}
    else:
        setting = _setting(frame, nobs, codes, n)
        if spec.covariance == "cluster" and not kernel_call(
                absorbed_degrees_of_freedom, [(codes, n)], setting.clusters[0]).nested[0]:
            frame.warn("The panels are not nested within the cluster variable: the random-"
                       "effects transformation leaves within-panel correlation that this "
                       "cluster covariance ignores.")
        v, info = covariance(frame, setting, score_x=kernels.score_regressors(empty, proj),
                             resid=fit.resid, bread=fit.bread, k=kk)
    df = info.get("df_inference")
    info = _relabel(frame, info, small)
    slopes, level = fit.beta[1:], x[:, 1:]
    xb, xb_bar = level @ slopes, xbar[:, 1:] @ slopes
    balanced = bool((sizes == sizes[0]).all())
    metrics = {
        "r_squared_within": _squared_correlation(xb - xb_bar[codes], y - ybar[codes]),
        "r_squared_between": _squared_correlation(xb_bar, ybar),
        "r_squared_overall": _squared_correlation(xb, y),
        "sigma_u": math.sqrt(sigma_u2), "sigma_e": math.sqrt(sigma_e2),
        "rho": sigma_u2 / (sigma_u2 + sigma_e2),
        "theta": float(theta[0]) if balanced else None,
        "rmse": math.sqrt(fit.rss / (nobs - kk)), **_structure(sizes, nobs),
        "n_instruments": len(blocks.instr.terms), "n_endogenous": len(blocks.endog.terms),
    }
    extra = {
        "model": "re", "estimator": "ec2sls" if ec2sls else "g2sls",
        "endogenous": blocks.endog.terms, "instruments": blocks.instr.terms,
        "omitted_instruments": blocks.omitted_instruments,
        "instrument_columns_used": used.shape[1],
        "variance_components": {
            "sigma_u_squared": sigma_u2, "sigma_e_squared": sigma_e2, "ssr_within": rss_within,
            "df_within": nobs - n - k_within, "within_regressors": k_within,
            "ssr_between": ssr_between, "df_between": n - k_between,
            "between_regressors": k_between, "trace": trace,
            "method": "swamy_arora"},
        "theta": _theta_summary(theta), "ssr_transformed": fit.rss,
    }
    return build_result(
        frame, terms=terms, params=fit.beta, covariance=v,
        title="EC2SLS random-effects IV regression" if ec2sls else TITLES["re"], use_t=small,
        df_inference=df if small else None, df_resid=nobs - kk, metrics=metrics,
        fitted=x @ fit.beta, solver="qr_gls_2sls", inference=info,
        tests={"model": _model_test(fit.beta, v, terms, df, small)}, extra=extra,
        categories=blocks.exog.categories, provenance={"model": "re"})


# ---- convenience ---------------------------------------------------------------------


def xtivreg(*, data: Any, y: str, x: Sequence[str] | None = None, endog: Sequence[str],
            instruments: Sequence[str], panel: str, time: str | None = None, model: str = "fe",
            covariance: str | None = None, cluster: str | None = None, ec2sls: bool = False,
            small: bool = False, categorical: Sequence[str] | None = None,
            missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Panel-data instrumental-variables regression (Stata's ``xtivreg``).

    Model
        ``y_it = x_it'b1 + endog_it'b2 + u_i + e_it``: panel effects ``u_i``, regressors
        in ``endog`` correlated with ``e_it``, excluded ``instruments``. ``model`` selects
        the transformation; each estimator is two-stage least squares on the
        transformed outcome, regressors and instruments:

        - ``'fe'`` (default): within transformation. Columns without within variation
          are omitted. ``Intercept`` is Stata's ``_cons`` (grand means added back);
          ``sigma_e^2 = RSS/(N - n - k)``, ``sigma_u`` the standard deviation of
          ``u_i = ybar_i - xbar_i'b``, ``rho = sigma_u^2/(sigma_u^2 + sigma_e^2)``.
        - ``'fd'``: first differences over consecutive periods (needs ``time``), with a
          constant. Observations without a predecessor are dropped and recorded.
        - ``'be'``: panel means with a constant (``n - K`` degrees of freedom).
        - ``'re'``: G2SLS random effects. ``sigma_e^2 = RSS_w/(N - n - k_w)`` from the
          within 2SLS, ``sigma_u^2 = max(0, {SSR*_b - (n - K_b) sigma_e^2}/(N - r))``
          from the between 2SLS in which each panel mean appears ``T_i`` times
          (Swamy-Arora adapted to unbalanced panels: ``SSR*_b = sum_i T_i ubar_i^2``,
          ``r = trace{(Zb'Zb)^-1 Zb'Z_mu Z_mu'Zb}``; with balanced panels
          ``RSS_b/(n - K_b) - sigma_e^2/T``), quasi-demeaning by
          ``theta_i = 1 - sqrt(sigma_e^2/(T_i sigma_u^2 + sigma_e^2))`` of outcome,
          regressors and instruments. ``ec2sls=True`` uses Baltagi's EC2SLS instrument
          set (within-transformed instruments and their panel means). ``K_b`` and
          ``k_w`` count the columns that vary between / within panels.

    Parameters
        data: DataFrame (or columns/records). y: outcome. x: exogenous regressors (may be
            omitted). endog: endogenous regressors. instruments: excluded instruments.
        panel: panel identifier. time: integer or datetime period (required for
            ``'fd'``; duplicate periods within a panel are an error).
        covariance: ``'nonrobust'`` (default; ``'unadjusted'`` accepted):
            ``s^2 (X~'P X~)^-1`` on the transformed data with ``s^2 = RSS/(N - n - k)``
            (fe), ``RSS/(N_d - K)`` (fd), ``RSS/(n - K)`` (be), ``RSS*/(N - K)`` (re);
            ``'robust'``: clustered on the panel (HC1 on the panel means for ``'be'``);
            ``'cluster'`` (implied by ``cluster``). Sandwiches use
            ``G/(G-1) (N-1)/(N-K)`` with ``K = k + 1``.
        cluster: one cluster column (constant within panel for ``'be'``). For ``'fe'`` a
            cluster column that does not nest the panels counts the panel effects in
            ``K`` (with a warning).
        small: ``False`` (default, as Stata's xtivreg): z statistics and a Wald chi2
            model test. ``True``: t and F statistics with the residual degrees of
            freedom (``G - 1`` with a cluster covariance). The covariance matrix is the
            same in both cases.
        categorical: exogenous regressors expanded as dummies. missing: ``'raise'`` or
            ``'drop'``. alpha: significance level of the confidence intervals.

    Result
        ``metrics``: fe/re/be: r_squared_within, r_squared_between, r_squared_overall
        (the model's own transformation: ``1 - RSS/TSS`` of its 2SLS regression, which
        can be negative; the others: squared correlations of ``x'b`` and ``y``),
        sigma_u, sigma_e, rho (fe, re), corr_u_xb (fe: correlation of ``u_i`` and
        ``x_it'b`` over observations), theta (re, balanced panels), rmse, n_groups,
        t_min, t_avg, t_max, n_instruments, n_endogenous; fd: r_squared and rmse of the
        differenced equation. ``tests``: ``model`` (Wald chi2 of the slopes, F with
        ``small``) and, for fe with the nonrobust covariance, ``fixed_effects`` (F test
        that all u_i = 0: pooled against within 2SLS residual sums of squares).
        ``extra``: the kept instruments, omitted instruments and, for re, the variance
        components and the distribution of ``theta_i``.

    Errors
        ``underidentified`` (too few usable instruments, also within or between panels),
        ``no_endogenous_regressors``, ``insufficient_observations``, ``constant_outcome``,
        ``perfect_fit``, ``repeated_time_values``, ``empty_sample`` (fd without
        consecutive periods), ``insufficient_clusters``, ``cluster_varies_within_panel``
        (be), ``missing_values``, ``invalid_spec``.

    Stata
        ``xtset id year`` then ``xtivreg y x1 (p = z1 z2), fe`` is
        ``oe.xtivreg(data=df, y='y', x=['x1'], endog=['p'], instruments=['z1', 'z2'],
        panel='id', time='year')``; ``, re`` is ``model='re'``; ``, re ec2sls`` adds
        ``ec2sls=True``; ``, fd`` and ``, be`` likewise.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"id": np.repeat(range(100), 5), "year": np.tile(range(5), 100)})
        >>> a = np.repeat(rng.normal(size=100), 5); v = rng.normal(size=500)
        >>> df["z"] = rng.normal(size=500); df["p"] = df.z + a + v
        >>> df["y"] = 1 + 2 * df.p + a + 0.6 * v + rng.normal(size=500)
        >>> print(oe.xtivreg(data=df, y="y", endog=["p"], instruments=["z"], panel="id",
        ...                  time="year").summary())
    """
    from openecon.analysis import fit

    spec = make_spec(
        "xtivreg", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
        covariance="nonrobust" if covariance == "unadjusted" else covariance, cluster=cluster,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        columns={"endogenous": column_list(endog, "endog"),
                 "instruments": column_list(instruments, "instruments")},
        options={"model": model, "ec2sls": True if ec2sls else None,
                 "small": True if small else None})
    return fit(spec, data=data)
