"""Vector error-correction model by Johansen's maximum likelihood (Stata: vec).

Model, for K variables with a VAR(p) in levels and r cointegrating equations:

    D y_t = alpha (beta' y_(t-1) + mu + rho t) + sum_(i<p) Gamma_i D y_(t-i) + gamma + tau t + e_t,
    E[e_t e_t'] = Omega.

Estimation follows the Methods and formulas of Stata's ``vec`` (``johansen.py``
has the reduced-rank regression):

1. beta~ = the r eigenvectors of the largest eigenvalues, normalized as
   beta~' = (I_r, beta-breve') on the first r variables (Johansen's normalization).
2. alpha = S01 beta~ (beta~' S11 beta~)^-1 and Omega = S00 - alpha beta~' S10 (ML).
3. Constants and trends that are not estimated inside the cointegrating
   equations are backed out of the unrestricted coefficients v and tau of the
   regression of D y_t on (beta~' Z1_t, Z2_t):
       mu = (alpha'alpha)^-1 alpha' v,   rho = (alpha'alpha)^-1 alpha' tau.
4. The short-run parameters (alpha, Gamma_i, gamma, tau) are the OLS
   coefficients of D y_t on the demeaned cointegrating equations
   E_t = beta' y_(t-1) + mu + rho t, the lagged differences and the
   unrestricted deterministic terms. Their covariance is the VAR formula
   conditional on beta (which is superconsistent), with the small-sample
   divisor T - d:  V = T/(T-d) Omega (x) (W'W)^-1,
   d = floor({K m2 + (K + m1 - r) r} / K).
5. The free elements of beta have covariance
   (1/(T-d)) (alpha' Omega^-1 alpha)^-1 (x) (H' S11 H)^-1, H selecting the rows
   of Z1 below the normalized block; backed-out mu and rho have no standard errors.

Inference is z based. The log likelihood is
``-(T/2)[K{ln(2 pi) + 1} + ln|S00| + sum_(i<=r) ln(1 - l_i)]``.

The header statistics follow the output of Stata's ``vec``: per equation the
RMSE is ``sqrt(SSR / (T - d))``, R-squared is UNCENTERED (``1 - SSR / sum y^2``
of the differenced outcome, whether or not the equation has a constant) and
chi2 is the Wald test that ALL coefficients of the equation, the constant
included, are zero, ``chi2(parms) = (T - d) R2 / (1 - R2)``. Each cointegrating
equation carries the Wald test that its K - r free coefficients on the
variables are jointly zero.

The lagged levels are used in deviations from their sample means when the
model has a constant (see ``johansen.py``); the constants reported here are
mapped back to the original levels exactly.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import build_result, kernel_call, wald_test
from openecon.econometrics.var import johansen as jo
from openecon.econometrics.var import kernels
from openecon.econometrics.var.common import (
    build_spec, check_design_size, integer, load_sample, variable_list,
)
from openecon.econometrics.var.estimators import check_model_size, residual_tests
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.engines.linalg import cholesky_inverse, cholesky_solve, least_squares
from openecon.models import ModelSpec, ResultBundle


def _project(alpha: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(alpha'alpha)^-1 alpha'v by QR (the Gram matrix would square the condition number)."""
    try:
        return kernel_call(least_squares, alpha.contiguous(), v.contiguous(),
                           drop_collinear=False, tol=1e-20).beta
    except AnalysisError as exc:
        if exc.code != "singular_design":
            raise
        raise AnalysisError(
            "singular_adjustment", "The constant or trend of the cointegrating equations "
            "cannot be backed out: the adjustment coefficients alpha are numerically rank "
            "deficient in the units of the data (mu = (alpha'alpha)^-1 alpha'v depends on "
            "those units). Rescale the variables to comparable magnitudes, lower the rank, or "
            "use trend='rconstant' / 'rtrend', which estimate these terms directly.") from exc


def fit_vec(spec: ModelSpec, data: Any, *, _replay=None) -> ResultBundle:
    """Entry point of the ``vec`` estimator (see ``vec`` for the model and conventions)."""
    sample = load_sample(spec, data) if _replay is None else _replay.sample()
    frame, names, levels = sample.frame, sample.names, sample.levels
    n, k = levels.shape if _replay is None else (_replay.ordered.count, len(names))
    p, r, trend = int(frame.option("lags")), int(frame.option("rank")), frame.option("trend")
    lm_lags = int(frame.option("lm_lags"))
    if k < 2:
        raise AnalysisError("invalid_spec", "A vector error-correction model needs at least two "
                            "endogenous variables.")
    if r >= k:
        raise AnalysisError("invalid_rank", f"rank must be smaller than the number of variables "
                            f"({k}): with rank = {k} the levels are stationary; fit oe.var.")
    if not spec.intercept and trend in {"constant", "rtrend", "trend"}:
        # Only reachable with a hand-built ModelSpec; oe.vec has no intercept argument.
        frame.warn(f"intercept=False is ignored: the deterministic terms of a VEC model are "
                   f"chosen by the trend option, and trend='{trend}' includes an unrestricted "
                   "constant. Use trend='none' or trend='rconstant' for a model without one.")
    if _replay is None:
        check_design_size(n, k * p + 2 + 2 * k)
    result = kernel_call(jo.johansen, levels, names, p, trend) if _replay is None else _replay.johansen()
    t, m1, m2 = result.t, result.m1, result.m2
    check_model_size(k * (r + m2))
    if _replay is None:
        frame.restrict(torch.arange(n) >= p, f"The first {p} observation(s) supply the lagged "
                       f"levels and differences; the estimation sample has {t} observations.")
    else:
        frame.warn(f"The first {p} observation(s) supply the lagged levels and differences; the estimation sample has {t} observations.")
    d = result.parameters(r) // k
    if t - d < 1:
        raise AnalysisError("insufficient_observations", f"The model has {d} parameters per "
                            f"equation but only {t} observations. Reduce the number of lags.")

    # Johansen normalization: beta~' = (I_r, beta-breve'). The eigenvectors satisfy
    # V'S11 V = I, so each row scaled by the root mean square of its Z1 column is of order
    # one whatever the units of the variables; the leading block is inverted in that scale.
    failure = AnalysisError(
        "normalization_failed", "The cointegrating vectors cannot be normalized on the first "
        f"{r} variable(s): they do not enter the estimated cointegrating equations "
        "independently. Reorder y so that the variables that enter the cointegrating "
        "equations come first.")
    scale = result.s11.diagonal().sqrt()
    standardized = result.vectors[:, :r] * scale[:, None]
    lead = standardized[:r]
    try:
        singular = torch.linalg.svdvals(lead)
        if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= 1e-12 * float(
                singular[0]):
            raise failure
        beta = torch.linalg.solve(lead.T, standardized.T).T
    except RuntimeError as exc:
        raise failure from exc
    beta = beta * scale[None, :r] / scale[:, None]            # coordinates of the centered Z1
    if not bool(torch.isfinite(beta).all()):
        raise failure
    beta[:r] = torch.eye(r, dtype=torch.float64)
    if _replay is not None:
        _replay.set_beta(beta)
    ce = result.z1 @ beta                                              # beta~' Z1_t (centered)
    labels_d = [f"D_{name}" for name in names]
    system_fit = kernels.fit_system if _replay is None else _replay.fit_system
    first = kernel_call(system_fit, torch.cat([ce, result.z2], dim=1), result.z0,
                        labels_d)
    alpha = first.coef[:, :r]
    omega = first.sigma_ml
    # beta~' ybar: what the centering of the levels moved into the constants.
    shift = beta[:k].T @ result.shift                                  # [r]

    # Back out mu and rho where they are not estimated inside the cointegrating equations.
    labels = result.z2_labels
    mu = rho = unrestricted_constant = None
    centered_mu = None
    if "Intercept" in labels:
        centered = first.coef[:, r + labels.index("Intercept")]
        centered_mu = _project(alpha, centered)
        mu = centered_mu - shift
        unrestricted_constant = centered - alpha @ shift               # v of the original levels
    if "trend" in labels:
        rho = _project(alpha, first.coef[:, r + labels.index("trend")])
    demeaned = ce.clone()
    if centered_mu is not None:
        demeaned += centered_mu if _replay is None else _replay.ones[:, None] * centered_mu
    if rho is not None:
        demeaned += result.trend_index[:, None] * rho
    w = torch.cat([demeaned, result.z2], dim=1)
    fit = kernel_call(system_fit, w, result.z0, labels_d) \
        if (mu is not None or rho is not None) else first
    if trend == "rconstant":
        beta[k] = beta[k] - shift                                      # constant of the levels
    m = r + m2
    factor = t / (t - d)
    covariance = torch.kron(omega * factor, fit.xtx_inv)
    params = fit.coef.reshape(-1)
    regressors = [f"L._ce{i + 1}" for i in range(r)] + labels
    equations = labels_d

    # Cointegrating equations with the Johansen standard errors of the free elements:
    # V(vec beta-breve) = (alpha' Omega^-1 alpha)^-1 (x) (H'S11 H)^-1 / (T - d).
    critical = normal_isf(spec.alpha / 2.0)
    beta_se = torch.full_like(beta, float("nan"))
    left = right = None
    if m1 > r:
        left = kernel_call(cholesky_inverse, alpha.T @ kernel_call(cholesky_solve, omega, alpha))
        right = kernel_call(cholesky_inverse, result.s11[r:, r:])
        if trend == "rconstant":
            # Constant of the original levels: c = c_centered - sum_j beta_j ybar_j.
            jacobian = torch.eye(m1 - r, dtype=torch.float64)
            jacobian[-1, :-1] = -result.shift[r:]
            right = jacobian @ right @ jacobian.T
        beta_se[r:] = (right.diagonal()[:, None] * left.diagonal()[None, :] / (t - d)).sqrt()
    full_labels = list(result.z1_labels)
    extra_rows: list[tuple[str, torch.Tensor]] = []
    if rho is not None:
        extra_rows.append(("_trend", rho))
    if mu is not None:
        extra_rows.append(("_cons", mu))
    cointegrating = []
    for i in range(r):
        entries = []
        for j, label in enumerate(full_labels):
            estimate, error = float(beta[j, i]), float(beta_se[j, i])
            entry: dict[str, Any] = {"variable": label, "estimate": estimate, "std_error": None,
                                     "statistic": None, "p_value": None, "ci_low": None,
                                     "ci_high": None,
                                     "status": "normalized" if j < r else "estimated"}
            if j >= r and math.isfinite(error) and error > 0:
                z = estimate / error
                entry.update({"std_error": error, "statistic": z, "p_value": 2 * normal_sf(abs(z)),
                              "ci_low": estimate - critical * error,
                              "ci_high": estimate + critical * error})
            entries.append(entry)
        for label, values in extra_rows:
            entries.append({"variable": label, "estimate": float(values[i]), "std_error": None,
                            "statistic": None, "p_value": None, "ci_low": None, "ci_high": None,
                            "status": "backed out of the unrestricted estimates"})
        record: dict[str, Any] = {"equation": f"_ce{i + 1}", "parms": k - r,
                                  "coefficients": entries}
        if right is not None and k > r:
            # Stata's "Cointegrating equations" table: the free coefficients on the variables.
            block = right[:k - r, :k - r] * (left[i, i] / (t - d))
            test = wald_test(beta[r:k, i], block, list(range(k - r)))
            record.update({key: test.get(key) for key in
                           ("statistic", "df", "p_value", "distribution")})
        cointegrating.append(record)

    # VAR representation in levels: y_t = sum A_i y_(t-i) + c0 + c1 t + e_t.
    gamma = fit.coef[:, r:r + k * (p - 1)].reshape(k, k, p - 1).permute(2, 0, 1)   # [p-1, K, K]
    pi = alpha @ beta[:k].T
    a = torch.zeros((p, k, k), dtype=torch.float64)
    a[0] = pi + torch.eye(k, dtype=torch.float64)
    for i in range(p - 1):
        a[i] += gamma[i]
        a[i + 1] -= gamma[i]
    c0 = torch.zeros(k, dtype=torch.float64)
    c1 = torch.zeros(k, dtype=torch.float64)
    if trend == "rconstant":
        c0 = alpha @ beta[k]
    elif trend == "rtrend":
        c1 = alpha @ beta[k]
    if unrestricted_constant is not None:
        c0 = unrestricted_constant.clone()
    if "trend" in labels:
        c1 = first.coef[:, r + labels.index("trend")].clone()
    roots = kernel_call(kernels.companion_eigenvalues, a)
    remaining = roots[k - r:]
    stable = all(root["modulus"] < 1.0 for root in remaining)
    if not stable:
        frame.warn("Beyond the unit moduli imposed by the model, an eigenvalue of the companion "
                   f"matrix has modulus {remaining[0]['modulus']:.4f} >= 1: the specified rank "
                   "may be too high or the process explosive.")

    # Fit statistics.
    ll = result.log_likelihood(r)
    count = result.parameters(r)
    metrics = {
        "log_likelihood": ll, "aic": -2.0 * ll + 2.0 * count,
        "bic": -2.0 * ll + math.log(t) * count,
        "hqic": -2.0 * ll + 2.0 * math.log(math.log(t)) * count,
        "aic_per_obs": (-2.0 * ll + 2.0 * count) / t,
        "hqic_per_obs": (-2.0 * ll + 2.0 * math.log(math.log(t)) * count) / t,
        "sbic_per_obs": (-2.0 * ll + math.log(t) * count) / t,
        "det_sigma_ml": math.exp(fit.logdet_ml), "rank": r, "n_equations": k, "n_lags": p,
        "df_model": count, "df_eq": d, "T": t,
    }
    y = result.z0
    ssr = fit.resid.square().sum(dim=0)
    has_constant = "Intercept" in labels
    # Stata's vec header: uncentered R-squared and the Wald test of ALL coefficients of the
    # equation (the constant is one of the regressors of the VAR in differences).
    tss = y.square().sum(dim=0)
    centered_tss = ((y - y.mean(dim=0)).square().sum(dim=0) if _replay is None
                    else _replay.centered_tss)
    equation_rows = []
    for i, name in enumerate(equations):
        test = wald_test(params, covariance, [i * m + c for c in range(m)])
        row: dict[str, Any] = {
            "equation": name, "parms": m, "rmse": math.sqrt(float(ssr[i]) / (t - d)),
            "r_squared": 1.0 - float(ssr[i] / tss[i]) if float(tss[i]) > 0 else None,
            **{key: test.get(key) for key in ("statistic", "df", "p_value", "distribution")}}
        if has_constant:
            row["r_squared_centered"] = (1.0 - float(ssr[i] / centered_tss[i])
                                         if float(centered_tss[i]) > 0 else None)
        equation_rows.append(row)
    tests: dict[str, Any] = {}
    for j in range(p - 1):
        indices = [i * m + r + v * (p - 1) + j for i in range(k) for v in range(k)]
        tests[f"lag_exclusion_L{j + 1}D"] = wald_test(
            params, covariance, indices,
            label=f"Wald test that all lagged differences at lag {j + 1} are jointly zero")
    residual, residual_extra = (residual_tests(equations, w, fit, omega, lm_lags) if _replay is None
                               else _replay.diagnostics(beta, centered_mu, rho, fit, omega, lm_lags))
    tests.update(residual)

    rows = jo.rank_rows(result)
    extra: dict[str, Any] = {
        "layout": {"variables": names, "lags": p, "rank": r, "trend": trend,
                   "n_regressors": m, "regressors": regressors, "exogenous": [],
                   "constant": has_constant},
        "eigenvalues": result.eigenvalues, "rank_test": rows,
        "selected_rank": jo.select_rank(rows, "trace", 5, k),
        "critical_values_flag": jo.critical_values.provenance_note(trend, k),
        "beta": cointegrating, "beta_matrix": beta[:k], "alpha": alpha,
        "ce_constant": mu if mu is not None else (beta[k] if trend == "rconstant" else None),
        "ce_trend": rho if rho is not None else (beta[k] if trend == "rtrend" else None),
        "pi": pi, "omega": omega, "equations": equation_rows,
        "stability": {"eigenvalues": roots, "unit_moduli_imposed": k - r, "stable": stable,
                      "convention": "eigenvalues of the companion matrix of the VAR in levels; "
                                    "the model imposes K - r unit moduli, the others must lie "
                                    "inside the unit circle"},
        "var_representation": {"A": a, "constant": c0, "trend": c1,
                               "convention": "y_t = sum_i A_i y_(t-i) + constant + trend * t + "
                                             "e_t, t the 1-based position in the ordered sample"},
        "forecast": {"last_values": levels[n - p:] if _replay is None else levels, "last_period": sample.last_period,
                     "last_trend": n},
        **residual_extra,
    }
    fitted, observed = (fit.fitted[:, 0], y[:, 0]) if _replay is None else _replay.preview(beta, centered_mu, rho, fit)
    bundle = build_result(
        frame, terms=[f"{name}:{label}" for name in equations for label in regressors],
        params=params, covariance=covariance, title="Vector error-correction model",
        equations=[name for name in equations for _ in regressors], use_t=False,
        nobs=t, df_resid=float(t - d), metrics=metrics, fitted=fitted, observed=observed,
        solver="johansen_reduced_rank_regression",
        solver_diagnostics={"condition_number": fit.condition_number, "equations": k,
                            "regressors_per_equation": m},
        inference={"covariance": spec.covariance, "intercept": has_constant,
                   "correction": "Omega (x) (W'W)^-1 conditional on beta, divisor T - d "
                                 "(Stata vec); d = floor(free parameters / K)",
                   "small_sample_correction": factor, "degrees_of_freedom_d": d,
                   "beta": "Johansen asymptotic covariance of the free elements of beta with "
                           "divisor T - d"},
        tests=tests, extra=extra,
        provenance={
            "method": "Johansen maximum likelihood (reduced-rank regression), Johansen "
                      "normalization on the first r variables",
            "time_order": f"sorted by {spec.time}" if spec.time else "input row order",
            "sample": {"levels": n, "lags_lost": p, "used": t},
            "system": names, "trend_specification": trend,
            "fitted_values": f"one-step-ahead predictions of D.{names[0]} (the first equation)",
            "trend": "t = 1, 2, ... is the position of the observation in the ordered sample",
        },
    )
    return bundle if _replay is None else _replay.finalize(bundle)


def vec(*, data: Any, y: Any, time: str | None = None, lags: int = 2, rank: int = 1,
        trend: str = "constant", covariance: str | None = None, lm_lags: int = 2,
        missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Vector error-correction model by Johansen's maximum likelihood (Stata ``vec``; EViews VEC).

    Model: ``D y_t = alpha (beta' y_(t-1) + mu + rho t) + sum_(i<lags) Gamma_i D y_(t-i)
    + gamma + tau t + e_t`` for the K variables in ``y`` (levels), with ``rank``
    cointegrating equations beta (K x r) and adjustment coefficients alpha (K x r).
    beta is estimated by reduced-rank regression and normalized so that its first r
    rows are the identity (Johansen's normalization: order ``y`` accordingly); the
    short-run parameters are OLS given beta.

    Parameters
    ----------
    data : table; y : list of the K >= 2 endogenous variables in levels.
    time : optional time column (rows are sorted by it; gaps are an error).
    lags : lags of the underlying VAR in levels (default 2); the VEC model has
        ``lags - 1`` lagged differences.
    rank : number of cointegrating equations, 1 <= rank < K (see ``oe.vecrank``).
    trend : ``"none"``, ``"rconstant"``, ``"constant"`` (default), ``"rtrend"`` or
        ``"trend"`` (Stata's trend()): which of mu, rho, gamma, tau are in the model.
    covariance : only ``"nonrobust"`` (the default): Stata's conditional covariance
        described below. Any other name is rejected.
    lm_lags : lags of the residual autocorrelation LM test (default 2).
    missing : ``"raise"`` or ``"drop"`` (rows may only be dropped at the ends).
    alpha : level of the confidence intervals.

    Result
    ------
    Coefficients (grouped by equation ``D_<variable>``): ``D_<v>:L._ce<i>`` (the
    adjustment coefficients alpha), ``D_<v>:LD.<w>``, ``D_<v>:L2D.<w>``, ... (Gamma_i),
    ``D_<v>:trend`` and ``D_<v>:Intercept``. Their covariance is
    ``T/(T-d) Omega (x) (W'W)^-1`` (Stata's small-sample divisor T - d) and inference
    is z based.

    ``metrics``: log_likelihood, aic, bic, hqic (and Stata's per-observation
    aic_per_obs, hqic_per_obs, sbic_per_obs), det_sigma_ml, rank, n_equations, n_lags,
    df_model (free parameters), df_eq (d), T.

    ``tests``: ``lag_exclusion_L<j>D``, ``normality`` (Jarque-Bera on
    Cholesky-orthogonalized residuals), ``lm_autocorrelation_L<s>``.

    ``extra``: ``beta`` (each cointegrating equation: its coefficients with the
    standard errors of the free elements, and Stata's Wald test ``chi2(K - r)`` that
    the free coefficients on the variables are zero), ``beta_matrix``, ``alpha``,
    ``ce_constant`` (mu), ``ce_trend`` (rho), ``pi`` = alpha beta', ``omega`` (ML),
    ``eigenvalues``, ``rank_test`` (the ``oe.vecrank`` table), ``selected_rank``,
    ``critical_values_flag``, ``equations`` (Stata's header: ``parms``, ``rmse`` =
    sqrt(SSR/(T-d)), the UNCENTERED ``r_squared`` and the Wald test of all coefficients
    of the equation, constant included, ``chi2(parms) = (T-d) R2/(1-R2)``; the centered
    R-squared is added as ``r_squared_centered`` when the equation has a constant),
    ``stability`` (companion eigenvalues; K - r unit moduli are imposed),
    ``var_representation`` (A_1..A_p and deterministic terms of the VAR in levels),
    ``normality``.

    Post-estimation: ``oe.forecast(result, steps)`` (= ``oe.vec_forecast``) and
    ``oe.irf(result, ...)``.

    Stata: ``vec y1 y2 y3, lags(2) rank(1) trend(constant)``.

    Example
    -------
    >>> import numpy as np, pandas as pd, openecon as oe
    >>> rng = np.random.default_rng(3)
    >>> walk = rng.normal(size=400).cumsum()
    >>> frame = pd.DataFrame({"y1": walk + rng.normal(size=400),
    ...                       "y2": 0.5 * walk + rng.normal(size=400)})
    >>> oe.vecrank(data=frame, y=["y1", "y2"], lags=2)
    >>> result = oe.vec(data=frame, y=["y1", "y2"], lags=2, rank=1)
    >>> print(result.summary()); result.extra["beta"]
    """
    names = variable_list(y)
    spec = build_spec(
        "vec", outcome=names[0], time=time, covariance=covariance, missing=missing,
        alpha=alpha, columns={"system": names},
        options={"lags": integer(lags), "rank": integer(rank), "trend": trend,
                 "lm_lags": integer(lm_lags)},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)
