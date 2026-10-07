"""Single-equation instrumental-variables regression: Stata's ``ivregress``.

Model: ``y = X1 b1 + X2 b2 + u`` with exogenous ``X1`` (the constant included),
endogenous ``X2`` (``E[X2'u] != 0``) and excluded instruments ``Z2``;
``Z = [X1 Z2]`` satisfies ``E[Z'u] = 0``. K = K1 + q regressors, L = K1 + L2
instruments, order condition ``L2 >= q``.

Estimators (``kernels``; no projection matrix is formed)
    2sls   b = (X'P_Z X)^-1 X'P_Z y, from the first-stage QR and a QR of the
           instrument-driven part of the endogenous regressors.
    liml   k-class with kappa the smallest eigenvalue of (Y'M_Z Y)^-1 Y'M_1 Y,
           Y = [y X2]; kappa = 1 (2SLS) when exactly identified.
    gmm    two-step efficient GMM b = (X'Z S^-1 Z'X)^-1 X'Z S^-1 Z'y: first step
           2SLS, S from its residuals by ``wmatrix`` (robust, cluster, hac,
           unadjusted), optionally iterated (``igmm``) and centered (``center``).

Stata conventions (ivregress Methods and formulas)
    Default (``small=False``): z statistics, Wald chi2 model test, error
    variance RSS/N, no N/(N-K) factor on robust covariances; cluster covariances
    carry G/(G-1). ``small=True``: t and F with N-K degrees of freedom (G-1 with
    clusters), RSS/(N-K), N/(N-K) on robust and HAC and (N-1)/(N-K) G/(G-1) on
    cluster covariances. R-squared is 1 - RSS/TSS with the IV residuals (it can be
    negative); TSS is centered with a constant and uncentered without.

Diagnostics are described in ``diagnostics``; the result layout in
:func:`ivregress`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, make_spec, wald_test,
)
from openecon.econometrics.iv.common import (
    Blocks, Estimate, Setting, check_fit, check_pweights, estimate, estimation_weights,
    role_designs, screen, sum_of_squares,
)
from openecon.econometrics.iv.diagnostics import Diagnostics
from openecon.models import ModelSpec, ResultBundle

TITLES = {"2sls": "Instrumental-variables (2SLS) regression",
          "fuller": "Instrumental-variables (Fuller) regression", "kclass": "Instrumental-variables (k-class) regression",
          "liml": "Instrumental-variables (LIML) regression",
          "gmm": "Instrumental-variables (GMM) regression"}
SOLVERS = {"2sls": "householder_qr_two_stage", "liml": "householder_qr_k_class",
           "fuller": "householder_qr_k_class", "kclass": "householder_qr_k_class",
           "gmm": "householder_qr_two_stage_cholesky_gmm"}


def gmm_options(frame: ModelFrame, method: str) -> tuple[str, bool, bool]:
    """Resolved (wmatrix, igmm, center); rejects GMM settings on other methods."""
    spec = frame.spec
    given = [name for name in ("wmatrix", "igmm", "center") if spec.options.get(name)]
    if method != "gmm":
        if given:
            raise AnalysisError("invalid_spec", f"Option(s) {', '.join(given)} apply to "
                                "method='gmm' only.")
        return "robust", False, False
    wmatrix = spec.options.get("wmatrix") or (
        spec.covariance if spec.covariance in {"cluster", "hac"} else "robust")
    if wmatrix == "cluster" and spec.covariance != "cluster":
        raise AnalysisError("invalid_spec", "wmatrix='cluster' needs the cluster column(s): "
                            "pass cluster=... (the covariance is then cluster as well).")
    return wmatrix, bool(spec.options.get("igmm")), bool(spec.options.get("center"))


def check_hac(frame: ModelFrame, wmatrix: str) -> None:
    spec = frame.spec
    hac = spec.covariance == "hac" or wmatrix == "hac"
    given = [name for name in ("lags", "kernel") if name in spec.options]
    if hac:
        if "lags" not in spec.options:
            raise AnalysisError("invalid_spec", "A HAC covariance or weight matrix needs the "
                                "option lags (the Newey-West lag length; 0 gives White's "
                                "estimator).")
        if spec.panel is not None and spec.time is None:
            raise AnalysisError("invalid_spec", "HAC autocovariances within panels need a time "
                                "column alongside the panel column.")
    elif given:
        raise AnalysisError("invalid_spec", f"Option(s) {', '.join(given)} apply only with a "
                            "HAC covariance or weight matrix.")


def model_test(est: Estimate, terms: list[str], small: bool) -> dict[str, Any]:
    """Wald test of every coefficient but the constant: chi2, or F with ``small``."""
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    df = est.info.get("df_inference") if small else None
    label = "Model F test (slopes)" if small else "Wald chi2 test of the slopes"
    return wald_test(est.beta, est.covariance, slopes, df_resid=df, label=label)


def describe(blocks: Blocks, est: Estimate, diagnostics: Diagnostics) -> dict[str, Any]:
    """The ``extra`` record common to ivregress and ivreghdfe."""
    extra: dict[str, Any] = {
        "method": est.method, "exogenous": blocks.exog.terms, "endogenous": blocks.endog.terms,
        "instruments": blocks.instr.terms, "omitted_instruments": blocks.omitted_instruments,
        "first_stage": diagnostics.first_stage,
    }
    if est.kappa is not None:
        extra["kappa"] = est.kappa
    if est.gmm is not None:
        extra["gmm"] = dict(est.gmm)
    return extra


def fit_ivregress(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``ivregress``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    method, small = frame.option("method"), bool(frame.option("small"))
    wmatrix, igmm, center = gmm_options(frame, method)
    check_hac(frame, wmatrix)
    weights, nobs = estimation_weights(frame)
    blocks = screen(frame, *role_designs(frame, intercept=spec.intercept), weights)
    y = frame.numeric(spec.outcome)
    clusters = frame.cluster_dimensions() if spec.covariance == "cluster" else None
    setting = Setting(nobs, weights, spec.covariance, small, clusters=clusters,
                      cluster_names=registry.cluster_columns(spec) or None)
    est = estimate(frame, setting, y, blocks, method=method, wmatrix=wmatrix, igmm=igmm,
                   center=center)
    terms = blocks.terms
    k, constant = len(terms), int(blocks.exog.intercept)
    df_resid = nobs - k
    tss = sum_of_squares(y, weights, centered=bool(constant))
    check_fit(est.rss, tss, df_resid, sum_of_squares(y, weights, centered=False))
    diagnostics = Diagnostics(frame, setting, y, blocks, est).run()
    r_squared = 1 - est.rss / tss
    metrics: dict[str, Any] = {
        "r_squared": r_squared,
        "adjusted_r_squared": 1 - (1 - r_squared) * (nobs - constant) / df_resid,
        "rmse": (est.rss / (df_resid if small else nobs)) ** 0.5,
    }
    if est.kappa is not None:
        metrics["kappa"] = est.kappa
    metrics.update({"df_model": k - constant, "df_resid": df_resid,
                    "n_instruments": len(blocks.instr.terms),
                    "n_endogenous": len(blocks.endog.terms)})
    if est.gmm is not None:
        metrics.update({"j": est.gmm["j"], "gmm_iterations": est.gmm["iterations"]})
    tests = {"model": model_test(est, terms, small), **diagnostics.tests}
    info = dict(est.info)
    info.update({
        "df_inference": info.get("df_inference") if small else None, "nobs": nobs,
        "method": method, "small": small,
        "error_variance": "RSS/(N-K)" if small else "RSS/N",
        "r_squared_definition": "centered" if constant else "uncentered",
        "residual_definition": "outcome minus X b with the observed (not fitted) endogenous "
                               "regressors",
    })
    return build_result(
        frame, terms=terms, params=est.beta, covariance=est.covariance, title=TITLES[method],
        use_t=small, df_inference=est.info.get("df_inference") if small else None,
        df_resid=df_resid, metrics=metrics, fitted=y - est.resid, nobs=nobs, inference=info,
        tests=tests, extra=describe(blocks, est, diagnostics),
        categories=blocks.exog.categories, solver=SOLVERS[method],
        solver_diagnostics={
            "first_stage_condition_number": est.proj.first.condition_number,
            "second_stage_condition_number": est.condition_number,
            "condition_number_basis": "unit-norm column-scaled weighted blocks"},
        optimizer=None if est.gmm is None else {
            "method": "iterated GMM" if est.gmm["igmm"] else "two-step GMM",
            "iterations": est.gmm["iterations"], "converged": est.gmm["converged"]})


def resolve_covariance(covariance: str | None, cluster: Any, weight_type: str | None,
                       method: str = "2sls", wmatrix: str | None = None) -> str | None:
    """Convenience-function covariance: Stata's names and defaults."""
    if covariance == "unadjusted":
        return "nonrobust"
    if covariance is not None or cluster:
        return covariance
    if method == "gmm":
        # Stata: the GMM VCE defaults to the type of the weight matrix (robust by default).
        return {"unadjusted": "nonrobust", "cluster": None}.get(wmatrix or "robust",
                                                                wmatrix or "robust")
    return "robust" if weight_type == "pweight" else None


def ivregress(*, data: Any, y: str, x: Sequence[str] | None = None, endog: Sequence[str],
              instruments: Sequence[str], method: str = "2sls", covariance: str | None = None,
              kappa: float | None = None, fuller_alpha: float | None = None,
              cluster: str | Sequence[str] | None = None, small: bool = False,
              weights: str | None = None, weight_type: str | None = None,
              wmatrix: str | None = None, igmm: bool = False, lags: int | None = None,
              kernel: str | None = None, time: str | None = None, panel: str | None = None,
              center: bool = False, categorical: Sequence[str] | None = None,
              intercept: bool = True, missing: str = "raise",
              alpha: float = 0.05) -> ResultBundle:
    """Instrumental-variables regression by 2SLS, LIML, Fuller, k-class or GMM.

    Model
        ``y = x'b1 + endog'b2 + u`` where the regressors in ``endog`` are correlated
        with ``u``. ``instruments`` are the EXCLUDED instruments; the exogenous
        regressors ``x`` (and the constant) instrument themselves. With ``Z`` all
        instruments and ``P_Z`` the projection on them:

        - ``method='2sls'``: ``b = (X'P_Z X)^-1 X'P_Z y``.
        - ``method='liml'``: ``b = {X'(I - k M_Z)X}^-1 X'(I - k M_Z)y`` with ``k`` the
          smallest eigenvalue of ``(Y'M_Z Y)^-1 Y'M_x Y``, ``Y = [y endog]`` (reported
          as ``kappa``; equal to 2SLS when exactly identified).
        - ``method='gmm'``: ``b = (X'Z S^-1 Z'X)^-1 X'Z S^-1 Z'y``, ``S`` the covariance
          of the moments ``z_i u_i`` from the 2SLS residuals (two-step), or iterated
          to convergence with ``igmm=True``.
        - ``method='fuller'``: LIML kappa minus ``fuller_alpha/(N-L)``;
          positive ``fuller_alpha`` defaults to 1 and L includes exogenous instruments.
        - ``method='kclass'``: explicit ``kappa`` in [0,1], where 0 is OLS
          and 1 is 2SLS. Conventional covariance requires the model assumptions;
          use ``iv_weak_test``/``iv_ar_confidence_set`` for weak-IV inference.

        Collinear regressors and instruments are omitted left to right with a warning;
        at least as many usable instruments as endogenous regressors are required.

    Parameters
        data: DataFrame, mapping of columns or list of row records.
        y: outcome. x: exogenous regressors (may be omitted). endog: endogenous
            regressors. instruments: excluded instruments. All numeric.
        covariance: ``'nonrobust'`` (default for 2sls/liml; alias ``'unadjusted'``):
            ``s^2 (X'P_Z X)^-1`` with ``s^2 = RSS/N``; ``'robust'``: the Huber-White
            sandwich on the projected regressors, without ``N/(N-K)``; ``'cluster'``
            (implied by ``cluster``): cluster sandwich times ``G/(G-1)``; ``'hac'``:
            Newey-West with ``lags``/``kernel``. For GMM the default is the type of
            ``wmatrix`` and the efficient ``(X'Z S^-1 Z'X)^-1`` is reported; a different
            type gives the GMM sandwich. pweights default to ``'robust'``.
        cluster: one or two cluster columns (two: Cameron-Gelbach-Miller, ``G_min``).
        small: report t and F statistics with ``N-K`` degrees of freedom (``G-1`` with
            clusters), ``s^2 = RSS/(N-K)``, ``N/(N-K)`` on robust/HAC and
            ``(N-1)/(N-K)`` on cluster covariances (Stata's ``small``).
        weights, weight_type: ``'aweight'``, ``'fweight'`` (N = sum of weights) or
            ``'pweight'`` (needs a robust/cluster/hac covariance).
        wmatrix: GMM weight matrix: ``'robust'`` (default), ``'cluster'`` (uses
            ``cluster``), ``'hac'`` (uses ``lags``/``kernel``) or ``'unadjusted'``
            (reproduces 2SLS). igmm: iterate GMM until the coefficients converge.
            center: center the moments in the weight matrix.
        lags, kernel, time, panel: HAC lag length ``L`` (bandwidth ``L+1``), kernel
            (``'bartlett'``, ``'parzen'``, ``'quadratic_spectral'``, ``'truncated'``),
            integer period column and, for panels, the unit column.
        categorical: exogenous regressors expanded as ``name[level]`` dummies.
        intercept: include the constant ``Intercept``. missing: ``'raise'`` or ``'drop'``.
        alpha: significance level of the confidence intervals.

    Result
        ``coefficients`` (exogenous terms first, then endogenous) with z tests (t with
        ``small``). ``metrics``: r_squared (``1 - RSS/TSS``, may be negative),
        adjusted_r_squared, rmse, kappa (liml), df_model, df_resid, n_instruments,
        n_endogenous, and j / gmm_iterations (gmm). ``tests``: ``model`` (Wald chi2 or
        F of the slopes); overidentification ``overid_sargan`` and ``overid_basmann``
        (2sls, nonrobust), ``overid_score`` (2sls, robust: Wooldridge),
        ``anderson_rubin`` and ``basmann_f`` (liml), ``hansen_j`` (gmm); endogeneity
        ``endog_durbin`` and ``endog_wu_hausman`` (2sls, nonrobust),
        ``endog_robust_score`` and ``endog_robust_regression`` (2sls, robust),
        ``endog_c`` (gmm); weak identification ``cragg_donald`` (minimum eigenvalue
        statistic) and, with a robust covariance, ``kleibergen_paap_rk_f``.
        ``extra['first_stage']``: per endogenous regressor the F test of the excluded
        instruments, R-squared, adjusted, partial and Shea's partial R-squared;
        ``extra['instruments']``, ``extra['omitted_instruments']``, ``extra['gmm']``.

    Errors
        ``underidentified`` (order or rank condition), ``no_endogenous_regressors``,
        ``insufficient_observations``, ``constant_outcome``, ``perfect_fit``,
        ``perfect_first_stage`` (overidentified LIML whose instruments reproduce a
        regressor), ``singular_weight_matrix`` (GMM: fewer clusters than instruments),
        ``unsupported_covariance`` (pweights with the conventional covariance),
        ``insufficient_clusters``, ``repeated_time_values``, ``missing_values``,
        ``invalid_spec`` (overlapping roles, GMM or HAC options without their method).

    Stata
        ``ivregress 2sls y x1 (p = z1 z2), vce(robust)`` is
        ``oe.ivregress(data=df, y='y', x=['x1'], endog=['p'], instruments=['z1', 'z2'],
        covariance='robust')``; ``estat firststage``, ``estat overid`` and
        ``estat endogenous`` are in ``result.extra['first_stage']`` and ``result.tests``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> z = rng.normal(size=(500, 2)); v = rng.normal(size=500)
        >>> df = pd.DataFrame({"z1": z[:, 0], "z2": z[:, 1], "x1": rng.normal(size=500)})
        >>> df["p"] = df.z1 + 0.5 * df.z2 + v
        >>> df["y"] = 1 + 2 * df.p - df.x1 + 0.8 * v + rng.normal(size=500)
        >>> result = oe.ivregress(data=df, y="y", x=["x1"], endog=["p"],
        ...                       instruments=["z1", "z2"])
        >>> print(result.summary())
    """
    spec = make_spec(
        "ivregress", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type, method, wmatrix),
        cluster=cluster, weights=weights, weight_type=weight_type, panel=panel, time=time,
        missing=missing, alpha=alpha,
        columns={"endogenous": column_list(endog, "endog"),
                 "instruments": column_list(instruments, "instruments")},
        options={"method": method, "small": True if small else None, "wmatrix": wmatrix,
                 "igmm": True if igmm else None, "center": True if center else None,
                 "lags": lags, "kernel": kernel, "kappa": kappa, "fuller_alpha": fuller_alpha})
    return fit(spec, data=data)
