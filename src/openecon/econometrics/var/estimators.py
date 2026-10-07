"""Vector autoregression (Stata: var, varsoc, varstable, vargranger, varwle, varnorm, varlmar).

The VAR(p) model for the K endogenous variables y_t with exogenous regressors
x_t is

    y_t = A_1 y_(t-1) + ... + A_p y_(t-p) + B x_t + v + d t + u_t,   E[u_t u_t'] = Sigma.

Every equation has the same regressors, so equation-by-equation OLS is the
Gaussian maximum-likelihood (and the SUR/GLS) estimator; all K equations are
solved from one Householder QR of the lagged design. The first p observations
supply the lags; T = n - p observations are used.

Conventions follow the Methods and formulas of Stata's ``var`` and of its
postestimation commands; they are stated in ``var`` below and in
``docs/econometrics/var.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, build_result, column_list, kernel_call, table, wald_test,
)
from openecon.econometrics.var import impulse, kernels
from openecon.econometrics.var.common import (
    Sample, build_spec, check_alpha, check_design_size, integer, load_sample, variable_list,
)
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import collinear_columns, weighted_crossprod
from openecon.models import ModelSpec, ResultBundle

# Arrays kept in result.extra are bounded: impulse responses are stored only while
# (irf_steps + 1) K^2 stays below this, the regressor moment matrix only for small designs.
MAX_STORED_RESPONSES = 5000
MAX_STORED_MOMENT = 60
# The result stores the full coefficient covariance, (K m)^2 numbers.
MAX_COEFFICIENTS = 1000


def check_model_size(coefficients: int) -> None:
    if coefficients > MAX_COEFFICIENTS:
        raise AnalysisError("model_too_large", f"The system has {coefficients} coefficients "
                            f"(equations times regressors per equation); the limit is "
                            f"{MAX_COEFFICIENTS} because their full covariance matrix is stored "
                            "with the result. Reduce the number of lags or variables.")


def exogenous_design(sample: Sample, *, constant: bool, trend: bool,
                     lags: int) -> tuple[Design, Tensor]:
    """The exogenous block (all n rows) and the deterministic block [n, trend + constant].

    Collinear exogenous columns are omitted Stata style (recorded in the
    result). The screen runs on the rows that enter the regression, with the
    constant and the trend first, so a constant exogenous regressor loses
    against the model's own constant.
    """
    frame = sample.frame
    n = frame.n
    design = frame.design(intercept=False)
    labels = (["Intercept"] if constant else []) + (["trend"] if trend else [])
    clash = [term for term in design.terms if term in set(labels)]
    if clash:
        raise AnalysisError("duplicate_terms", f"The exogenous regressor(s) {', '.join(clash)} "
                            "collide with the model's own constant or trend. Rename the columns.")
    columns = []
    if trend:
        columns.append(torch.arange(1, n + 1, dtype=torch.float64)[:, None])
    if constant:
        columns.append(torch.ones((n, 1), dtype=torch.float64))
    block = torch.cat(columns, dim=1) if columns else torch.empty((n, 0), dtype=torch.float64)
    if design.x.shape[1]:
        rows = slice(min(lags, n - 1), n)
        screen = Design(torch.cat([block[rows].flip(1), design.x[rows]], dim=1),
                        labels + design.terms, design.categories, constant)
        kept = set(frame.drop_collinear(screen).terms[len(labels):])
        if len(kept) < len(design.terms):
            design = design.select([i for i, term in enumerate(design.terms) if term in kept])
    return design, block


def _screen_system(z: Tensor, labels: list[str], constant: bool) -> None:
    """Refuse a design whose lagged endogenous block is collinear.

    With a constant (the last column) the other columns of ``z`` are already
    mean-deviated; the constant is screened first, as Stata sweeps it first.
    """
    m = z.shape[1]
    _, omitted = kernel_call(collinear_columns, z.flip(1) if constant else z)
    if omitted:
        names = [labels[m - 1 - i] if constant else labels[i] for i in omitted]
        raise AnalysisError(
            "collinear_system", "The lagged endogenous variables are linearly dependent "
            f"(redundant: {', '.join(names[:6])}). Remove the variable that is an exact "
            "combination of the others, or reduce the number of lags.")


def _lag_order(levels: Tensor, exog: Tensor, block: Tensor, maxlag: int, alpha: float,
               constant: bool) -> dict[str, Any]:
    """Lag-order selection statistics for lags 0..maxlag on their common sample (varsoc).

    ``block`` holds the deterministic columns (trend, then the constant when
    ``constant``). With a constant every other column and the outcomes are
    mean-deviated first, an exact reparameterization that keeps series with a
    large level accurate.
    """
    n, k = levels.shape
    t = n - maxlag
    d0 = block.shape[1] + exog.shape[1]
    if t < d0 + k * maxlag + k + 1:
        raise AnalysisError("insufficient_observations", f"maxlag={maxlag} leaves {t} "
                            "observations, too few for the lag-order statistics. Reduce maxlag.")
    lagged = [levels[maxlag - j:n - j] for j in range(1, maxlag + 1)]
    z = torch.cat([block[maxlag:], exog[maxlag:], *lagged], dim=1)
    y = levels[maxlag:]
    if constant:
        others = [i for i in range(z.shape[1]) if i != block.shape[1] - 1]
        z[:, others] -= z[:, others].mean(dim=0)
        y = y - y.mean(dim=0)
    sizes = [d0 + k * j for j in range(maxlag + 1)]
    logdets = kernel_call(kernels.nested_logdets, z, y, sizes)
    rows = []
    previous = None
    for j, (m, logdet) in enumerate(zip(sizes, logdets, strict=True)):
        ll = kernels.log_likelihood(logdet, t, k)
        parameters = k * m
        row: dict[str, Any] = {"lag": j, "ll": ll, "lr": None, "df": None, "p_value": None}
        if previous is not None:
            row["lr"] = max(0.0, 2.0 * (ll - previous))
            row["df"] = k * k
            row["p_value"] = chi2_sf(row["lr"], k * k)
        row["fpe"] = math.exp(logdet + k * (math.log(t + m) - math.log(t - m))) if t > m else None
        row["aic"] = (-2.0 * ll + 2.0 * parameters) / t
        row["hqic"] = (-2.0 * ll + 2.0 * math.log(math.log(t)) * parameters) / t
        row["sbic"] = (-2.0 * ll + math.log(t) * parameters) / t
        rows.append(row)
        previous = ll
    selected: dict[str, int | None] = {}
    for name in ("fpe", "aic", "hqic", "sbic"):
        values = [(row[name], row["lag"]) for row in rows if row[name] is not None]
        selected[name] = min(values)[1] if values else None
    rejected = [row["lag"] for row in rows[1:] if row["p_value"] < alpha]
    selected["lr"] = max(rejected) if rejected else 0
    return {"maxlag": maxlag, "n_obs": t, "rows": rows, "selected": selected, "alpha": alpha,
            "convention": "Stata varsoc: all lags on the sample of the longest lag; AIC, HQIC and "
                          "SBIC are -2 ln L / T plus the penalty per observation; FPE = "
                          "|Sigma_ml| ((T + m) / (T - m))^K; the LR rule selects the highest lag "
                          "whose LR test against one lag fewer rejects at level alpha"}


def _wald_rows(params: Tensor, covariance: Tensor, groups: list[tuple[dict[str, Any], list[int]]],
               df_resid: float | None) -> list[dict[str, Any]]:
    rows = []
    for record, indices in groups:
        test = wald_test(params, covariance, indices, df_resid=df_resid)
        rows.append({**record, **{key: test.get(key) for key in
                                  ("statistic", "df", "df2", "p_value", "distribution")
                                  if key in test}})
    return rows


def residual_tests(names: list[str], z: Tensor, fit: kernels.SystemFit, sigma: Tensor,
                   lm_lags: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normality (varnorm) and LM autocorrelation (varlmar) tests of a fitted system."""
    t, k = fit.resid.shape
    tests: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    b1, b2 = kernel_call(kernels.normality_moments, fit.resid, sigma)
    skew, kurt = t * b1.square() / 6.0, t * (b2 - 3.0).square() / 24.0
    rows = []
    for i, name in enumerate([*names, "ALL"]):
        if i < k:
            s, q, df = float(skew[i]), float(kurt[i]), 1
            record = {"equation": name, "skewness": float(b1[i]), "kurtosis": float(b2[i])}
        else:
            s, q, df = float(skew.sum()), float(kurt.sum()), k
            record = {"equation": name, "skewness": None, "kurtosis": None}
        rows.append({**record, "skewness_chi2": s, "skewness_p": chi2_sf(s, df),
                     "kurtosis_chi2": q, "kurtosis_p": chi2_sf(q, df),
                     "jarque_bera": s + q, "jarque_bera_df": 2 * df,
                     "jarque_bera_p": chi2_sf(s + q, 2 * df)})
    extra["normality"] = rows
    tests["normality"] = {
        "statistic": rows[-1]["jarque_bera"], "df": 2 * k, "p_value": rows[-1]["jarque_bera_p"],
        "distribution": "chi2",
        "label": "Jarque-Bera test that the disturbances are jointly normal"}
    statistics = kernel_call(kernels.lm_autocorrelation, z, fit.resid, fit.xtx_inv,
                             fit.logdet_ml, lm_lags)
    for s, value in enumerate(statistics, start=1):
        if value is None:
            continue
        value = max(0.0, value)
        tests[f"lm_autocorrelation_L{s}"] = {
            "statistic": value, "df": k * k, "p_value": chi2_sf(value, k * k),
            "distribution": "chi2", "label": f"LM test of no residual autocorrelation at lag {s}"}
    return tests, extra


def fit_var(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``var`` estimator (see ``var`` for the model and conventions)."""
    sample = load_sample(spec, data)
    frame, names, levels = sample.frame, sample.names, sample.levels
    n, k = levels.shape
    p = int(frame.option("lags"))
    constant = bool(spec.intercept) and bool(frame.option("constant"))
    trend, small, dfk = (bool(frame.option(name)) for name in ("trend", "small", "dfk"))
    maxlag = frame.option("maxlag")
    maxlag = p if maxlag is None else int(maxlag)
    irf_steps, lm_lags = int(frame.option("irf_steps")), int(frame.option("lm_lags"))
    kinds = frame.option("irf_kinds")
    kinds = list(impulse.KINDS) if kinds is None else list(dict.fromkeys(kinds))
    unknown = [kind for kind in kinds if kind not in impulse.KINDS]
    if unknown:
        raise AnalysisError("invalid_option", f"irf_kinds must be chosen from "
                            f"{', '.join(impulse.KINDS)}; got {', '.join(map(str, unknown))}.")
    t = n - p
    if t < 2:
        raise AnalysisError("insufficient_observations", f"{p} lag(s) leave {max(t, 0)} "
                            "observation(s); the model needs a longer series.")
    design, block = exogenous_design(sample, constant=constant, trend=trend, lags=p)
    m = k * p + design.x.shape[1] + block.shape[1]
    check_design_size(t, m + k)
    if t <= m:
        raise AnalysisError("insufficient_observations", f"Each equation has {m} parameters but "
                            f"only {t} observations remain after {p} lag(s). Reduce the number "
                            "of lags or variables.")
    check_model_size(k * m)
    regressors = [f"L{j}.{name}" for name in names for j in range(1, p + 1)] + design.terms \
        + (["trend"] if trend else []) + (["Intercept"] if constant else [])
    z = torch.cat([kernels.lag_block(levels, p), design.x[p:], block[p:]], dim=1)
    y = levels[p:]
    # With a constant (the last column) the system is estimated on mean-deviated regressors
    # and outcomes: Zc = Z C with C = [[I, 0], [-zbar', 1]]. The slopes are unchanged, the
    # constant and (Z'Z)^-1 are mapped back exactly, and series with a large level keep
    # full accuracy in every statistic computed from cross products.
    work = y
    change = torch.eye(m, dtype=torch.float64)
    if constant:
        mean_z, mean_y = z[:, :-1].mean(dim=0), y.mean(dim=0)
        z[:, :-1] -= mean_z
        work = y - mean_y
        change[-1, :-1] = -mean_z
    _screen_system(z, regressors, constant)
    fit = kernel_call(kernels.fit_system, z, work, names)
    coef = fit.coef @ change.T
    if constant:
        coef[:, -1] += mean_y
    xtx_inv = change @ fit.xtx_inv @ change.T
    if p:
        frame.restrict(torch.arange(n) >= p, f"The first {p} observation(s) supply the lagged "
                       f"values; the estimation sample has {t} observations.")

    # Covariance of the stacked coefficients (equation-major).
    df_resid = t - m
    adjusted = small or dfk
    sigma = fit.sigma_ml * (t / df_resid) if dfk else fit.sigma_ml
    info: dict[str, Any] = {"covariance": spec.covariance, "small": small, "dfk": dfk}
    if spec.covariance == "robust":
        factor = t / df_resid if adjusted else t / (t - 1)
        covariance = kernel_call(kernels.robust_covariance, z, fit.resid, fit.xtx_inv) * factor
        covariance = torch.einsum("la,iajb,mb->iljm", change, covariance.reshape(k, m, k, m),
                                  change).reshape(k * m, k * m)
        info.update({"correction": "Huber-White system sandwich: "
                                   + ("T/(T-m)" if adjusted else "T/(T-1)"),
                     "small_sample_correction": factor})
    else:
        scale = t / df_resid if adjusted else 1.0
        covariance = torch.kron(fit.sigma_ml * scale, xtx_inv)
        info["correction"] = ("Sigma (x) (Z'Z)^-1 with Sigma = U'U/(T-m)" if adjusted else
                              "Sigma (x) (Z'Z)^-1 with the maximum-likelihood Sigma = U'U/T")
    info["reference"] = ("Student t and F with T - m degrees of freedom (small)" if small
                         else "standard normal and chi-squared")
    params = coef.reshape(-1)
    reference_df = float(df_resid) if small else None

    # Fit statistics.
    ll = kernels.log_likelihood(fit.logdet_ml, t, k)
    parameters = k * m
    metrics = {
        "log_likelihood": ll, "aic": -2.0 * ll + 2.0 * parameters,
        "bic": -2.0 * ll + math.log(t) * parameters,
        "hqic": -2.0 * ll + 2.0 * math.log(math.log(t)) * parameters,
        "aic_per_obs": (-2.0 * ll + 2.0 * parameters) / t,
        "hqic_per_obs": (-2.0 * ll + 2.0 * math.log(math.log(t)) * parameters) / t,
        "sbic_per_obs": (-2.0 * ll + math.log(t) * parameters) / t,
        "fpe": math.exp(fit.logdet_ml + k * (math.log(t + m) - math.log(t - m))),
        "det_sigma_ml": math.exp(fit.logdet_ml), "n_equations": k, "n_lags": p, "df_eq": m,
        "df_model": parameters, "df_resid": df_resid, "T": t,
    }
    ssr = fit.resid.square().sum(dim=0)
    tss = work.square().sum(dim=0)            # centered when the model has a constant
    slopes = [i for i, label in enumerate(regressors) if label != "Intercept"]
    equations = []
    for i, name in enumerate(names):
        record: dict[str, Any] = {
            "equation": name, "parms": m, "rmse": math.sqrt(float(ssr[i]) / df_resid),
            "r_squared": 1.0 - float(ssr[i] / tss[i]) if float(tss[i]) > 0 else None}
        if slopes:
            test = wald_test(params, covariance, [i * m + c for c in slopes],
                             df_resid=reference_df)
            record.update({key: test.get(key) for key in
                           ("statistic", "df", "df2", "p_value", "distribution") if key in test})
        equations.append(record)

    # Granger causality (vargranger) and lag exclusion (varwle).
    tests: dict[str, Any] = {}
    granger_groups, exclusion_groups = [], []
    for i, name in enumerate(names):
        others = [v for v in range(k) if v != i]
        for v in others:
            granger_groups.append(({"equation": name, "excluded": names[v]},
                                   [i * m + v * p + j for j in range(p)]))
        if others and p:
            granger_groups.append(({"equation": name, "excluded": "ALL"},
                                   [i * m + v * p + j for v in others for j in range(p)]))
    for j in range(p):
        for i, name in enumerate(names):
            exclusion_groups.append(({"lag": j + 1, "equation": name},
                                     [i * m + v * p + j for v in range(k)]))
        exclusion_groups.append(({"lag": j + 1, "equation": "ALL"},
                                 [i * m + v * p + j for i in range(k) for v in range(k)]))
    granger = _wald_rows(params, covariance, granger_groups, reference_df) if p else []
    exclusion = _wald_rows(params, covariance, exclusion_groups, reference_df)
    for row in granger:
        if row["excluded"] == "ALL":
            tests[f"granger_all_{row['equation']}"] = {
                **{key: row[key] for key in row if key not in ("equation", "excluded")},
                "label": f"Granger causality: all other variables excluded from "
                         f"{row['equation']}"}
    for row in exclusion:
        if row["equation"] == "ALL":
            tests[f"lag_exclusion_L{row['lag']}"] = {
                **{key: row[key] for key in row if key not in ("equation", "lag")},
                "label": f"Wald test that all endogenous variables at lag {row['lag']} are "
                         "jointly zero"}
    residual, residual_extra = residual_tests(names, z, fit, sigma, lm_lags)
    tests.update(residual)

    # Stability, lag-order selection and impulse responses.
    a = kernels.lag_matrices(coef, k, p)
    roots = kernel_call(kernels.companion_eigenvalues, a)
    stable = all(root["modulus"] < 1.0 for root in roots)
    if roots and not stable:
        frame.warn("The VAR does not satisfy the stability condition: at least one eigenvalue of "
                   f"the companion matrix has modulus {roots[0]['modulus']:.4f} >= 1. Impulse "
                   "responses and forecasts do not die out.")
    extra: dict[str, Any] = {
        "layout": {"variables": names, "lags": p, "exogenous": design.terms, "trend": trend,
                   "constant": constant, "n_regressors": m, "regressors": regressors,
                   "small": small, "dfk": dfk},
        "sigma": sigma, "sigma_ml": fit.sigma_ml, "equations": equations,
        "stability": {"eigenvalues": roots, "stable": stable,
                      "convention": "eigenvalues of the companion matrix; stable when every "
                                    "modulus is below 1"},
        "granger": granger, "lag_exclusion": exclusion, **residual_extra,
    }
    try:
        extra["lag_order_selection"] = _lag_order(levels, design.x, block, maxlag, spec.alpha,
                                                  constant)
    except AnalysisError as exc:
        if exc.code not in {"insufficient_observations", "singular_residual_covariance",
                            "perfect_fit"}:
            raise
        frame.warn(f"Lag-order selection statistics were not computed: {exc}")
    if p and (irf_steps + 1) * k * k <= MAX_STORED_RESPONSES:
        positions = impulse.alpha_positions(k, p, m)
        responses = kernel_call(impulse.impulse_responses, a, sigma, irf_steps,
                                cov_alpha=covariance[positions][:, positions], n_obs=t)
        record: dict[str, Any] = {
            "steps": irf_steps, "variables": names, "kinds": kinds,
            "convention": "array[step][response][impulse]; orthogonalized responses use the "
                          "Cholesky factor of sigma in the order of the variables",
        }
        for kind in kinds:
            record[kind] = getattr(responses, kind)
        if "orthogonalized" in kinds:
            record["fevd"] = responses.fevd
        if responses.se is not None:
            record["se"] = {name: value for name, value in responses.se.items()
                            if name in kinds or (name == "fevd" and "orthogonalized" in kinds)}
        else:
            record["se"] = None
            record["se_note"] = "standard errors were not computed for a system of this size"
        extra["irf"] = record
    elif p:
        frame.warn("Impulse responses were not stored in the result (too many variables and "
                   "steps); compute them with oe.irf(result, steps=..., kind=...).")
    extra["forecast"] = {
        "last_values": levels[n - p:], "last_period": sample.last_period, "last_trend": n,
        # Zc'Zc / T of the mean-deviated regressors and their means (for the forecast MSE).
        "moment": weighted_crossprod(z, None) / t if m <= MAX_STORED_MOMENT else None,
        "means": mean_z if constant else None,
    }

    return build_result(
        frame, terms=[f"{name}:{label}" for name in names for label in regressors],
        params=params, covariance=covariance, title="Vector autoregression",
        equations=[name for name in names for _ in regressors], use_t=small,
        df_inference=reference_df, df_resid=float(df_resid), metrics=metrics,
        fitted=y[:, 0] - fit.resid[:, 0], observed=y[:, 0], solver="qr_least_squares",
        solver_diagnostics={"condition_number": fit.condition_number, "equations": k,
                            "regressors_per_equation": m},
        inference=info, tests=tests, extra=extra, categories=design.categories,
        provenance={
            "method": "equation-by-equation OLS (Gaussian ML) from one QR of the lagged design",
            "time_order": f"sorted by {spec.time}" if spec.time else "input row order",
            "sample": {"levels": n, "lags_lost": p, "used": t},
            "system": names,
            "fitted_values": f"one-step-ahead predictions of {names[0]} (the first equation)",
            "trend": "t = 1, 2, ... counted from the first observation of the ordered sample"
                     if trend else None,
        },
    )


def var(*, data: Any, y: Any, x: Any = None, time: str | None = None, lags: int = 2,
        constant: bool = True, trend: bool = False, small: bool = False, dfk: bool = False,
        covariance: str | None = None, maxlag: int | None = None, irf_steps: int = 8,
        irf_kinds: Any = None, lm_lags: int = 2, categorical: Any = None,
        missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Vector autoregression (Stata ``var``; EViews VAR), estimated by OLS equation by equation.

    Model: ``y_t = A_1 y_(t-1) + ... + A_p y_(t-p) + B x_t + v + d t + u_t`` for the
    K variables in ``y``, with ``E[u_t u_t'] = Sigma``. All equations share the same
    regressors, so OLS per equation is the Gaussian ML estimator; one QR of the
    lagged design solves the system. The first ``lags`` observations supply the
    lagged values (T = n - lags observations are used).

    Parameters
    ----------
    data : table (pandas/OpenEconometrics DataFrame, dict of columns, records).
    y : list of the endogenous variables. Their order is the Cholesky order of the
        orthogonalized impulse responses. (In the ``ModelSpec`` the first one is
        ``outcome`` and the whole list is the column role ``system``.)
    x : optional list of exogenous regressors (Stata's ``exog()``); ``categorical``
        names those to expand into indicators. Collinear exogenous terms are omitted.
    time : optional time column (integer periods or datetimes). Rows are sorted by it
        and must be consecutive; without it the row order is the time order.
    lags : p; lags 1..p of every endogenous variable enter each equation (default 2).
    constant, trend : include a constant (default) and a linear trend t = 1, 2, ...
    small : report t and F statistics with T - m degrees of freedom and compute the
        standard errors with the equation's degrees of freedom (Stata's ``small``).
    dfk : estimate Sigma with the divisor T - m instead of T (Stata's ``dfk``); m is
        the number of parameters per equation.
    covariance : ``"nonrobust"`` (default) is ``Sigma (x) (Z'Z)^-1``; ``"robust"`` is the
        Huber-White system sandwich (factor T/(T-1), or T/(T-m) with small/dfk).
    maxlag : highest lag of the lag-order selection table (default: ``lags``).
    irf_steps, irf_kinds : horizon (default 8) and kinds (``simple``, ``orthogonalized``,
        ``generalized``; default all) of the impulse responses stored in the result.
    lm_lags : lags of the residual autocorrelation LM test (default 2).
    missing : ``"raise"`` (default) or ``"drop"``; dropped rows may only lie at the ends
        of the series. alpha : level of the confidence intervals and of the LR lag rule.

    Result
    ------
    Coefficients are named ``<equation>:L<j>.<variable>``, ``<equation>:<exogenous>``,
    ``<equation>:trend`` and ``<equation>:Intercept`` and grouped by equation; inference
    is z (t with ``small``).

    ``metrics``: log_likelihood ``-(T/2){ln|Sigma_ml| + K ln(2 pi) + K}``; aic, bic, hqic
    (``-2 ln L`` plus 2, ln T and 2 ln ln T times the K m parameters) and Stata's
    per-observation versions aic_per_obs, hqic_per_obs, sbic_per_obs (the former divided
    by T); fpe ``|Sigma_ml| ((T+m)/(T-m))^K``; det_sigma_ml, n_equations, n_lags, df_eq
    (m), df_model, df_resid (T - m), T.

    ``tests``: ``granger_all_<eq>`` (all other variables jointly excluded from an
    equation), ``lag_exclusion_L<j>`` (all coefficients at lag j, all equations),
    ``normality`` (joint Jarque-Bera on Cholesky-orthogonalized residuals),
    ``lm_autocorrelation_L<s>`` (Johansen's LM statistic, chi2(K^2)). Wald tests are
    chi2, or F with T - m denominator degrees of freedom under ``small``.

    ``extra``: ``sigma`` (divisor T, or T - m with dfk) and ``sigma_ml``; ``equations``
    (parms, RMSE with divisor T - m, R-squared, Wald test of all slopes);
    ``lag_order_selection`` (LL, LR, FPE, AIC, HQIC, SBIC for lags 0..maxlag and the
    selected lags); ``stability`` (companion eigenvalues and the ``stable`` flag);
    ``granger`` and ``lag_exclusion`` (every pairwise / per-equation Wald test);
    ``normality`` (skewness, kurtosis and Jarque-Bera per equation and jointly);
    ``irf`` (arrays ``[step][response][impulse]`` of simple, orthogonalized and
    generalized responses, the variance decomposition ``fevd`` and delta-method
    standard errors ``se``); ``forecast`` (the last p observations for ``oe.forecast``).

    Post-estimation: ``oe.irf(result, steps=..., kind=...)``, ``oe.forecast(result,
    steps)`` (= ``oe.var_forecast``), and ``oe.varsoc`` before estimation.

    Stata: ``var y1 y2, lags(1/2) exog(x)`` then ``varsoc``, ``varstable``,
    ``vargranger``, ``varwle``, ``varnorm``, ``varlmar``, ``irf create``, ``fcast compute``.

    Example
    -------
    >>> import numpy as np, pandas as pd, openecon as oe
    >>> rng = np.random.default_rng(1)
    >>> e = rng.normal(size=(300, 2))
    >>> data = np.zeros((300, 2))
    >>> for t in range(1, 300):
    ...     data[t] = [0.5 * data[t-1, 0] + 0.2 * data[t-1, 1], 0.4 * data[t-1, 1]] + e[t]
    >>> frame = pd.DataFrame(data, columns=["income", "consumption"])
    >>> result = oe.var(data=frame, y=["income", "consumption"], lags=1)
    >>> print(result.summary())
    >>> oe.irf(result, steps=4, kind="orthogonalized")
    """
    names = variable_list(y)
    spec = build_spec(
        "var", outcome=names[0], predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), time=time, intercept=constant,
        covariance=covariance, missing=missing, alpha=alpha, columns={"system": names},
        options={"lags": integer(lags), "trend": trend, "small": small, "dfk": dfk,
                 "maxlag": integer(maxlag), "irf_steps": integer(irf_steps),
                 # A value that is not iterable is left for the registry to reject.
                 "irf_kinds": [irf_kinds] if isinstance(irf_kinds, str) else
                 list(irf_kinds) if hasattr(irf_kinds, "__iter__") else irf_kinds,
                 "lm_lags": integer(lm_lags)},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)


def varsoc(*, data: Any, y: Any, x: Any = None, time: str | None = None, maxlag: int = 4,
           constant: bool = True, trend: bool = False, categorical: Any = None,
           missing: str = "raise", alpha: float = 0.05) -> Any:
    """Lag-order selection statistics for a VAR (Stata ``varsoc``; EViews lag length criteria).

    VARs with 0, 1, ..., ``maxlag`` lags are fitted on the SAME sample (the
    observations available to the longest model, T = n - maxlag). For each
    lag order j with m_j parameters per equation the table reports

        ll    = -(T/2) {ln|Sigma_ml| + K ln(2 pi) + K}
        lr    = 2 {ll(j) - ll(j-1)}  ~  chi2(K^2), with its p-value
        fpe   = |Sigma_ml| ((T + m_j) / (T - m_j))^K
        aic   = -2 ll / T + 2 K m_j / T
        hqic  = -2 ll / T + 2 ln(ln T) K m_j / T
        sbic  = -2 ll / T + ln(T) K m_j / T

    and ``attrs["selected"]`` gives the lag chosen by each criterion (the minimum)
    and by the LR rule (the highest lag whose LR test rejects at level ``alpha``).
    All nested fits come from one QR of the design ordered lag by lag.

    Parameters are those of ``oe.var`` (``y`` the endogenous variables, ``x``
    exogenous regressors, ``time``, ``constant``, ``trend``, ``missing``).
    Returns a table with one row per lag; ``attrs`` hold ``selected``, ``n_obs``
    and ``maxlag``.

    Stata: ``varsoc y1 y2, maxlag(4)``.

    Example
    -------
    >>> oe.varsoc(data=frame, y=["income", "consumption"], maxlag=4)
    """
    names = variable_list(y)
    maxlag = integer(maxlag)
    if not isinstance(maxlag, int) or isinstance(maxlag, bool) or maxlag < 0:
        raise AnalysisError("invalid_lags", "maxlag must be a non-negative integer.")
    alpha = check_alpha(alpha)
    spec = build_spec(
        "var", outcome=names[0], predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), time=time, intercept=constant,
        missing=missing, alpha=alpha, columns={"system": names},
        options={"lags": max(1, maxlag), "trend": trend},
    )
    sample = load_sample(spec, data)
    if sample.frame.n - maxlag < 2:
        raise AnalysisError("insufficient_observations", f"maxlag={maxlag} leaves fewer than two "
                            "observations.")
    design, block = exogenous_design(sample, constant=constant, trend=trend, lags=maxlag)
    record = _lag_order(sample.levels, design.x, block, maxlag, alpha, constant)
    columns = ["lag", "ll", "lr", "df", "p_value", "fpe", "aic", "hqic", "sbic"]
    return table([[row[name] for name in columns] for row in record["rows"]], columns=columns,
                 selected=record["selected"], n_obs=record["n_obs"], maxlag=maxlag, alpha=alpha,
                 variables=names, exogenous=design.terms, convention=record["convention"],
                 warnings=list(sample.frame.warnings))


__all__ = ["fit_var", "var", "varsoc"]
