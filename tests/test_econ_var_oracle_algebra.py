"""Independent algebraic oracles for the var family (part 2 of the verification suite).

Every reference value is derived here from first principles, with formulas that
differ from the implementation's:

* ``oe.var``: generalized least squares on the stacked Kronecker system, a
  numerically differentiated Gaussian likelihood, restricted-versus-unrestricted
  regressions for every Wald test, roots of the characteristic polynomial for
  the stability check, simulated shock propagation for the impulse responses,
  the companion form and Lutkepohl's trace formula for the forecast covariance.
* ``oe.vecrank`` / ``oe.vec``: canonical correlations from QR factors, the
  nesting of the VEC model between a VAR in differences and a VAR in levels,
  numerically differentiated likelihoods written in the original (uncentered)
  levels for both covariance formulas, a brute-force maximization, and the
  reconstruction of residuals and forecasts from the reported coefficients.
"""

import warnings
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize

import openecon as oe

TRENDS = ["none", "rconstant", "constant", "rtrend", "trend"]


# ---- helpers --------------------------------------------------------------------------------

def var_frame(seed=7, n=90):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.0], [0.6, 0.8]]).T
    x = rng.normal(size=n)
    y = np.zeros((n, 2))
    a1 = np.array([[0.5, 0.2], [-0.1, 0.4]])
    a2 = np.array([[-0.2, 0.1], [0.1, 0.1]])
    for t in range(2, n):
        y[t] = np.array([0.4, -0.2]) + a1 @ y[t - 1] + a2 @ y[t - 2] + 0.5 * x[t] + 0.01 * t + e[t]
    frame = pd.DataFrame(y, columns=["u", "v"])
    frame["x"] = x
    frame["when"] = np.arange(200, 200 + n)
    return frame


def columns(frame, names, p, exog=(), trend=False, constant=True):
    """Regressor columns keyed by OpenEcon's term labels, and the outcome rows."""
    y = frame[names].to_numpy(dtype=float)
    n = len(y)
    out = {f"L{j}.{name}": y[p - j:n - j, v] for v, name in enumerate(names)
           for j in range(1, p + 1)}
    for name in exog:
        out[name] = frame[name].to_numpy(dtype=float)[p:]
    if trend:
        out["trend"] = np.arange(p + 1, n + 1, dtype=float)
    if constant:
        out["Intercept"] = np.ones(n - p)
    return out, y[p:]


def labels_of(result):
    first = result.coefficients[0].equation
    return [c.term.split(":", 1)[1] for c in result.coefficients if c.equation == first]


def design_of(result, cols):
    """The design in the order of the reported terms (no ordering formula is assumed)."""
    return np.column_stack([cols[label] for label in labels_of(result)])


def estimates(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


def lag_matrices(result, names, p):
    """A_1..A_p read from the coefficient table by term name."""
    rows = {c.term: c.estimate for c in result.coefficients}
    return np.array([[[rows[f"{i}:L{j}.{v}"] for v in names] for i in names]
                     for j in range(1, p + 1)])


# ---- oe.var ---------------------------------------------------------------------------------

@pytest.mark.parametrize("options", [{}, {"dfk": True}, {"small": True},
                                     {"small": True, "dfk": True}])
def test_var_is_generalized_least_squares_on_the_stacked_system(options):
    frame = var_frame()
    names = ["u", "v"]
    result = oe.var(data=frame, y=names, x=["x"], time="when", lags=2, trend=True, **options)
    cols, y = columns(frame, names, 2, exog=["x"], trend=True)
    z = design_of(result, cols)
    t, m = z.shape
    k = 2
    assert (t, m) == (88, 7) and result.nobs == t
    big = np.kron(np.eye(k), z)                               # equation-major stacking
    stacked = y.T.reshape(-1)
    ols = np.linalg.lstsq(big, stacked, rcond=None)[0]
    u = (stacked - big @ ols).reshape(k, t).T
    sigma_ml = u.T @ u / t
    sigma_v = sigma_ml * (t / (t - m) if options else 1.0)    # small and dfk both rescale V
    weight = np.kron(np.linalg.inv(sigma_v), np.eye(t))
    information = big.T @ weight @ big
    gls = np.linalg.solve(information, big.T @ weight @ stacked)
    covariance = np.linalg.inv(information)
    est, se = estimates(result)
    assert_allclose(est, gls, rtol=1e-7, atol=1e-9)
    assert_allclose(est, ols, rtol=1e-7, atol=1e-9)           # GLS = OLS: identical regressors
    assert_allclose(np.array(result.covariance_matrix), covariance, rtol=1e-7, atol=1e-13)
    assert_allclose(se, np.sqrt(np.diag(covariance)), rtol=1e-8)
    reported_sigma = sigma_ml * (t / (t - m) if options.get("dfk") else 1.0)
    assert_allclose(result.extra["sigma"], reported_sigma, rtol=1e-9)
    assert_allclose(result.extra["sigma_ml"], sigma_ml, rtol=1e-9)
    statistic = gls / np.sqrt(np.diag(covariance))
    if options.get("small"):
        p_values = 2 * stats.t.sf(np.abs(statistic), t - m)
        critical = stats.t.ppf(0.975, t - m)
        assert result.inference["df_inference"] == t - m and result.inference["use_t"]
    else:
        p_values = 2 * stats.norm.sf(np.abs(statistic))
        critical = stats.norm.ppf(0.975)
        assert result.inference["df_inference"] is None and not result.inference["use_t"]
    assert_allclose([c.statistic for c in result.coefficients], statistic, rtol=1e-7)
    assert_allclose([c.p_value for c in result.coefficients], p_values, rtol=1e-6, atol=1e-300)
    assert_allclose([c.ci_low for c in result.coefficients],
                    gls - critical * np.sqrt(np.diag(covariance)), rtol=1e-6, atol=1e-9)
    assert_allclose([c.ci_high for c in result.coefficients],
                    gls + critical * np.sqrt(np.diag(covariance)), rtol=1e-6, atol=1e-9)
    # Likelihood and criteria never depend on small / dfk.
    ll = -0.5 * t * (np.log(np.linalg.det(sigma_ml)) + k * np.log(2 * np.pi) + k)
    metrics = result.metrics
    assert metrics["log_likelihood"] == pytest.approx(ll, rel=1e-11)
    assert metrics["aic"] == pytest.approx(-2 * ll + 2 * k * m, rel=1e-11)
    assert metrics["bic"] == pytest.approx(-2 * ll + np.log(t) * k * m, rel=1e-11)
    assert metrics["hqic"] == pytest.approx(-2 * ll + 2 * np.log(np.log(t)) * k * m, rel=1e-11)
    assert metrics["sbic_per_obs"] == pytest.approx(metrics["bic"] / t, rel=1e-12)
    assert (metrics["df_eq"], metrics["df_model"], metrics["df_resid"], metrics["T"]) == (
        m, k * m, t - m, t)
    assert metrics["fpe"] == pytest.approx(
        np.linalg.det(sigma_ml) * ((t + m) / (t - m)) ** k, rel=1e-9)


def test_robust_covariance_by_an_explicit_sum_over_observations():
    frame = var_frame(seed=11, n=60)
    names = ["u", "v"]
    result = oe.var(data=frame, y=names, x=["x"], lags=1, covariance="robust")
    cols, y = columns(frame, names, 1, exog=["x"])
    z = design_of(result, cols)
    t, m = z.shape
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    u = y - z @ beta
    meat = np.zeros((2 * m, 2 * m))
    for row in range(t):                                      # s_t = u_t (x) z_t
        score = np.kron(u[row], z[row])
        meat += np.outer(score, score)
    bread = np.kron(np.eye(2), np.linalg.inv(z.T @ z))
    covariance = bread @ meat @ bread * t / (t - 1)
    assert_allclose(np.array(result.covariance_matrix), covariance, rtol=1e-8, atol=1e-14)
    assert result.inference["covariance"] == "robust"
    assert result.inference["small_sample_correction"] == pytest.approx(t / (t - 1))
    adjusted = oe.var(data=frame, y=names, x=["x"], lags=1, covariance="robust", small=True)
    assert_allclose(np.array(adjusted.covariance_matrix), covariance * (t - 1) / (t - m),
                    rtol=1e-8, atol=1e-14)
    # The point estimates and Sigma do not depend on the covariance estimator.
    plain = oe.var(data=frame, y=names, x=["x"], lags=1)
    assert_allclose(estimates(result)[0], estimates(plain)[0], rtol=1e-12)
    assert_allclose(result.extra["sigma"], plain.extra["sigma"], rtol=1e-12)
    assert result.metrics["log_likelihood"] == pytest.approx(plain.metrics["log_likelihood"])


def numerical_hessian(function, point, steps):
    size = len(point)
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            total = 0.0
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                moved = point.copy()
                moved[i] += si * steps[i]
                moved[j] += sj * steps[j]
                total += si * sj * function(moved)
            out[i, j] = out[j, i] = total / (4 * steps[i] * steps[j])
    return out


def test_covariance_is_the_inverse_hessian_of_an_independent_likelihood():
    frame = var_frame(seed=3, n=70)
    names = ["u", "v"]
    result = oe.var(data=frame, y=names, x=["x"], lags=1)
    cols, y = columns(frame, names, 1, exog=["x"])
    z = design_of(result, cols)
    t, m = z.shape
    k = 2

    def loglik(flat, sigma):
        u = y - z @ flat.reshape(k, m).T
        inverse = np.linalg.inv(sigma)
        return (-0.5 * t * np.log(np.linalg.det(sigma)) - 0.5 * np.einsum("ti,ij,tj->", u,
                                                                            inverse, u)
                - 0.5 * t * k * np.log(2 * np.pi))

    def negative(theta):                                      # full likelihood, Sigma = L L'
        lower = np.array([[np.exp(theta[-3]), 0.0], [theta[-2], np.exp(theta[-1])]])
        return -loglik(theta[:k * m], lower @ lower.T)

    best = minimize(negative, np.zeros(k * m + 3), method="BFGS",
                    options={"gtol": 1e-8, "maxiter": 5000})
    est, _ = estimates(result)
    assert -best.fun == pytest.approx(result.metrics["log_likelihood"], rel=1e-9)
    assert_allclose(best.x[:k * m], est, atol=3e-5)
    lower = np.array([[np.exp(best.x[-3]), 0.0], [best.x[-2], np.exp(best.x[-1])]])
    sigma = np.array(result.extra["sigma"])
    assert_allclose(lower @ lower.T, sigma, atol=3e-5)
    hessian = numerical_hessian(lambda b: loglik(b, sigma), est, np.full(k * m, 1e-3))
    assert_allclose(np.array(result.covariance_matrix), np.linalg.inv(-hessian), rtol=2e-6,
                    atol=1e-12)


def restricted_ssr(z, y, drop):
    keep = [i for i in range(z.shape[1]) if i not in set(drop)]
    resid = y - z[:, keep] @ np.linalg.lstsq(z[:, keep], y, rcond=None)[0]
    return resid.T @ resid


@pytest.mark.parametrize("options", [{}, {"dfk": True}, {"small": True}])
def test_wald_tests_equal_restricted_regression_statistics(options):
    rng = np.random.default_rng(21)
    n, k, p = 140, 3, 2
    names = ["a", "b", "c"]
    y = np.zeros((n, k))
    a1 = np.array([[0.4, 0.2, 0.0], [0.1, 0.3, 0.2], [0.0, -0.3, 0.4]])
    for t in range(1, n):
        y[t] = 0.2 + a1 @ y[t - 1] + rng.normal(size=k) @ np.array(
            [[1.0, 0.3, 0.0], [0.0, 1.0, -0.2], [0.0, 0.0, 1.0]])
    frame = pd.DataFrame(y, columns=names)
    result = oe.var(data=frame, y=names, lags=p, **options)
    cols, outcome = columns(frame, names, p)
    labels = labels_of(result)
    z = np.column_stack([cols[label] for label in labels])
    t, m = z.shape
    full = restricted_ssr(z, outcome, [])
    scale = (t - m) if options else t                         # divisor of the Wald statistics
    small = bool(options.get("small"))

    def check(row, drop, equations):
        difference = restricted_ssr(z, outcome, drop) - full
        index = [names.index(e) for e in equations]
        block = np.ix_(index, index)
        wald = scale * np.trace(np.linalg.solve(full[block], difference[block]))
        q = len(drop) * len(index)
        assert row["df"] == q
        if small:
            assert row["distribution"] == "F" and row["df2"] == t - m
            assert row["statistic"] == pytest.approx(wald / q, rel=1e-8)
            assert row["p_value"] == pytest.approx(stats.f.sf(wald / q, q, t - m), rel=1e-7)
        else:
            assert row["distribution"] == "chi2" and "df2" not in row
            assert row["statistic"] == pytest.approx(wald, rel=1e-8)
            assert row["p_value"] == pytest.approx(stats.chi2.sf(wald, q), rel=1e-7)

    assert len(result.extra["granger"]) == k * k
    for row in result.extra["granger"]:
        others = [n_ for n_ in names if n_ != row["equation"]]
        excluded = others if row["excluded"] == "ALL" else [row["excluded"]]
        drop = [labels.index(f"L{j}.{v}") for v in excluded for j in range(1, p + 1)]
        check(row, drop, [row["equation"]])
    assert len(result.extra["lag_exclusion"]) == p * (k + 1)
    for row in result.extra["lag_exclusion"]:
        drop = [labels.index(f"L{row['lag']}.{v}") for v in names]
        check(row, drop, names if row["equation"] == "ALL" else [row["equation"]])
    for row in result.extra["equations"]:                     # all slopes of one equation
        drop = [i for i, label in enumerate(labels) if label != "Intercept"]
        check(row, drop, [row["equation"]])
        i = names.index(row["equation"])
        centered = outcome[:, i] - outcome[:, i].mean()
        assert row["r_squared"] == pytest.approx(1 - full[i, i] / (centered @ centered), rel=1e-10)
        assert row["rmse"] == pytest.approx(np.sqrt(full[i, i] / (t - m)), rel=1e-10)
        assert row["parms"] == m
    for name in names:
        summary = result.tests[f"granger_all_{name}"]
        row = next(r for r in result.extra["granger"]
                   if r["equation"] == name and r["excluded"] == "ALL")
        assert summary["statistic"] == row["statistic"] and summary["df"] == row["df"]


def test_lag_order_statistics_equal_separately_fitted_models():
    """varsoc row j = the VAR(j) fitted on the common sample; LR = 2 (ll_j - ll_(j-1))."""
    frame = var_frame(seed=5, n=120)
    names = ["u", "v"]
    maxlag = 4
    table = oe.varsoc(data=frame, y=names, x=["x"], maxlag=maxlag)
    n = len(frame)
    t = n - maxlag
    assert table.attrs["n_obs"] == t and list(table["lag"]) == list(range(maxlag + 1))
    ll = []
    for j in range(1, maxlag + 1):
        trimmed = frame.iloc[maxlag - j:].reset_index(drop=True)      # the same T observations
        fit = oe.var(data=trimmed, y=names, x=["x"], lags=j)
        assert fit.nobs == t
        ll.append(fit.metrics["log_likelihood"])
        for name in ("aic", "hqic", "sbic"):
            assert table[name][j] == pytest.approx(fit.metrics[f"{name}_per_obs"], rel=1e-10)
        assert table["fpe"][j] == pytest.approx(fit.metrics["fpe"], rel=1e-9)
        own = fit.extra["lag_order_selection"]["rows"][j]             # post-estimation table
        assert own["ll"] == pytest.approx(fit.metrics["log_likelihood"], rel=1e-11)
    assert_allclose(table["ll"].to_numpy()[1:], ll, rtol=1e-10)
    # Lag 0: the regression on the exogenous variable and the constant only.
    y = frame[names].to_numpy()[maxlag:]
    z0 = np.column_stack([frame.x.to_numpy()[maxlag:], np.ones(t)])
    u = y - z0 @ np.linalg.lstsq(z0, y, rcond=None)[0]
    sigma0 = u.T @ u / t
    ll0 = -0.5 * t * (np.log(np.linalg.det(sigma0)) + 2 * np.log(2 * np.pi) + 2)
    assert table["ll"][0] == pytest.approx(ll0, rel=1e-10)
    assert table["fpe"][0] == pytest.approx(np.linalg.det(sigma0) * ((t + 2) / (t - 2)) ** 2,
                                            rel=1e-9)
    assert table["aic"][0] == pytest.approx((-2 * ll0 + 2 * 2 * 2) / t, rel=1e-10)
    lr = 2 * np.diff([ll0, *ll])
    assert_allclose(table["lr"].to_numpy()[1:], lr, rtol=1e-8)
    assert list(table["df"].to_numpy()[1:]) == [4.0] * maxlag
    assert_allclose(table["p_value"].to_numpy()[1:], stats.chi2.sf(lr, 4), rtol=1e-7, atol=1e-300)
    selected = table.attrs["selected"]
    for name in ("fpe", "aic", "hqic", "sbic"):
        assert selected[name] == int(np.argmin(table[name].to_numpy()))
    rejected = [j + 1 for j in range(maxlag) if stats.chi2.sf(lr[j], 4) < 0.05]
    assert selected["lr"] == (max(rejected) if rejected else 0)
    strict = oe.varsoc(data=frame, y=names, x=["x"], maxlag=maxlag, alpha=1e-12)
    assert strict.attrs["selected"]["lr"] == max(
        [j + 1 for j in range(maxlag) if stats.chi2.sf(lr[j], 4) < 1e-12], default=0)


def test_stability_roots_solve_the_characteristic_polynomial():
    """det(I - A_1 z - A_2 z^2) = 0: the companion eigenvalues are the reciprocal roots."""
    frame = var_frame(seed=2, n=150)
    names = ["u", "v"]
    result = oe.var(data=frame, y=names, lags=2)
    a = lag_matrices(result, names, 2)
    poly = np.polynomial.polynomial

    def entry(i, j):                                          # 1{i=j} - a1 z - a2 z^2
        return np.array([1.0 if i == j else 0.0, -a[0][i, j], -a[1][i, j]])

    determinant = poly.polysub(poly.polymul(entry(0, 0), entry(1, 1)),
                               poly.polymul(entry(0, 1), entry(1, 0)))
    roots = poly.polyroots(determinant)
    expected = np.sort(1 / np.abs(roots))[::-1]
    stability = result.extra["stability"]
    assert_allclose([r["modulus"] for r in stability["eigenvalues"]], expected, rtol=1e-8)
    assert stability["stable"] is bool(expected[0] < 1)
    for root in stability["eigenvalues"]:
        assert np.hypot(root["real"], root["imag"]) == pytest.approx(root["modulus"], rel=1e-12)
        value = complex(root["real"], root["imag"])           # lambda^2 I - lambda A_1 - A_2
        matrix = value ** 2 * np.eye(2) - value * a[0] - a[1]
        assert abs(np.linalg.det(matrix)) < 1e-10


def propagate(a, shock, steps):
    """Path of the system y_t = sum_j A_j y_(t-j) after y_0 = shock (zero history)."""
    p, k, _ = a.shape
    path = [np.asarray(shock, dtype=float)]
    for step in range(1, steps + 1):
        path.append(sum(a[j] @ path[step - 1 - j] for j in range(min(p, step))))
    return np.array(path)                                     # [steps + 1, K]


@pytest.mark.parametrize("dfk", [False, True])
def test_impulse_responses_equal_simulated_shock_propagation(dfk):
    rng = np.random.default_rng(8)
    n, names, steps = 200, ["a", "b", "c"], 7
    y = np.zeros((n, 3))
    for t in range(2, n):
        y[t] = (0.1 + np.array([[0.5, 0.1, 0.0], [0.2, 0.2, 0.1], [0.0, -0.2, 0.3]]) @ y[t - 1]
                + np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.2]]) @ y[t - 2]
                + rng.normal(size=3) @ np.array([[1.0, 0.5, 0.2], [0.0, 1.0, -0.4],
                                                 [0.0, 0.0, 0.7]]))
    frame = pd.DataFrame(y, columns=names)
    result = oe.var(data=frame, y=names, lags=2, dfk=dfk, irf_steps=steps)
    a = lag_matrices(result, names, 2)
    sigma = np.array(result.extra["sigma"])
    factor = np.linalg.cholesky(sigma)
    stored = result.extra["irf"]
    assert stored["steps"] == steps and stored["variables"] == names
    simple = np.stack([propagate(a, np.eye(3)[c], steps) for c in range(3)], axis=2)
    orthogonal = np.stack([propagate(a, factor[:, c], steps) for c in range(3)], axis=2)
    general = np.stack([propagate(a, sigma[:, c] / np.sqrt(sigma[c, c]), steps)
                        for c in range(3)], axis=2)
    assert_allclose(stored["simple"], simple, atol=1e-10)                 # [step][response][impulse]
    assert_allclose(stored["orthogonalized"], orthogonal, atol=1e-10)
    assert_allclose(stored["generalized"], general, atol=1e-10)
    squares = np.cumsum(orthogonal ** 2, axis=0)
    share = squares / squares.sum(axis=2, keepdims=True)
    assert_allclose(np.array(stored["fevd"])[1:], share[:-1], atol=1e-10)
    assert_allclose(np.array(stored["fevd"])[0], 0.0)
    # The h-step forecast-error variance of each variable is the sum of the squared responses.
    mse_one = sigma
    assert_allclose(squares[0].sum(axis=1), np.diag(mse_one), rtol=1e-10)
    # The long table of oe.irf holds the same numbers, for any horizon.
    table = oe.irf(result, steps=12, kind="generalized")
    longer = np.stack([propagate(a, sigma[:, c] / np.sqrt(sigma[c, c]), 12) for c in range(3)],
                      axis=2)
    for c, impulse in enumerate(names):
        for r, response in enumerate(names):
            block = table[(table.impulse == impulse) & (table.response == response)]
            assert_allclose(block.irf, longer[:, r, c], atol=1e-10)
    assert (table.std_error.to_numpy()[np.asarray(table.step) > 0] > 0).all()


@pytest.mark.parametrize("options", [{}, {"dfk": True}, {"small": True}])
def test_forecast_covariance_by_companion_form_and_lutkepohl_trace_formula(options):
    """Sigma_yhat(h) = Sigma_y(h) + Omega(h) / T (Lutkepohl 2005, eq. 3.5.12 and 3.5.16).

    Regressors in Lutkepohl's order Z_t = (1, y_t', y_(t-1)')', Gamma = Z Z' / T and
    Omega(h) = sum_ij tr{(B')^(h-1-i) Gamma^-1 B^(h-1-j) Gamma} Phi_i Sigma Phi_j'.
    """
    frame = var_frame(seed=13, n=110)
    names, p, steps = ["u", "v"], 2, 6
    result = oe.var(data=frame, y=names, lags=p, **options)
    table = oe.forecast(result, steps)
    y = frame[names].to_numpy()
    n, k = y.shape
    t = n - p
    a = lag_matrices(result, names, p)
    rows = {c.term: c.estimate for c in result.coefficients}
    nu = np.array([rows[f"{name}:Intercept"] for name in names])
    # Point forecasts by iterating the system.
    history = [y[-1], y[-2]]
    path = []
    for _ in range(steps):
        value = nu + a[0] @ history[0] + a[1] @ history[1]
        path.append(value)
        history = [value, history[0]]
    assert_allclose(table.forecast.to_numpy().reshape(k, steps).T, np.array(path), atol=1e-10)
    assert list(table.variable[:steps]) == ["u"] * steps and list(table.step[:steps]) == list(
        range(1, steps + 1))
    # Innovation part from the companion form: J (sum_i F^i Q F'^i) J'.
    sigma = np.array(result.extra["sigma"])
    companion = np.zeros((k * p, k * p))
    companion[:k, :k], companion[:k, k:] = a[0], a[1]
    companion[k:, :k] = np.eye(k)
    q = np.zeros((k * p, k * p))
    q[:k, :k] = sigma
    power, total, innovation = np.eye(k * p), np.zeros((k * p, k * p)), []
    for _ in range(steps):
        total = total + power @ q @ power.T
        innovation.append(total[:k, :k].copy())
        power = companion @ power
    # Estimation uncertainty: the coefficient covariance is Sigma_v (x) (Z Z')^-1.
    m = k * p + 1
    sigma_v = np.array(result.extra["sigma_ml"]) * (t / (t - m) if options else 1.0)
    regress = np.column_stack([np.ones(t), y[p - 1:n - 1], y[p - 2:n - 2]])      # Z_t rows
    gamma = regress.T @ regress / t
    b = np.zeros((m, m))
    b[0, 0] = 1.0
    b[1:1 + k, 0], b[1:1 + k, 1:1 + k], b[1:1 + k, 1 + k:] = nu, a[0], a[1]
    b[1 + k:, 1:1 + k] = np.eye(k)
    phi = [np.eye(k)]
    for i in range(1, steps):
        phi.append(sum(phi[i - j] @ a[j - 1] for j in range(1, min(i, p) + 1)))
    gamma_inverse = np.linalg.inv(gamma)
    powers = [np.linalg.matrix_power(b, i) for i in range(steps)]
    expected = []
    for h in range(1, steps + 1):
        omega = np.zeros((k, k))
        for i in range(h):
            for j in range(h):
                weight = np.trace(powers[h - 1 - i].T @ gamma_inverse @ powers[h - 1 - j] @ gamma)
                omega += weight * phi[i] @ sigma_v @ phi[j].T
        expected.append(np.sqrt(np.diag(innovation[h - 1] + omega / t)))
    assert_allclose(table.std_error.to_numpy().reshape(k, steps).T, np.array(expected), rtol=1e-8)
    assert "coefficient uncertainty" in table.attrs["mse_formula"]
    critical = stats.norm.ppf(0.95)
    wide = oe.forecast(result, steps, alpha=0.10)
    assert_allclose(wide.ci_high, wide.forecast + critical * wide.std_error, rtol=1e-10)
    assert_allclose(wide.ci_low, wide.forecast - critical * wide.std_error, rtol=1e-10)


def test_equivalent_specifications_give_the_same_fit():
    frame = var_frame(seed=17, n=100)
    names = ["u", "v"]
    base = oe.var(data=frame, y=names, x=["x"], lags=2, trend=True)
    rows = {c.term: c for c in base.coefficients}
    # (1) The built-in trend equals an exogenous trend column counted the same way.
    manual = oe.var(data=frame.assign(tt=np.arange(1.0, len(frame) + 1)), y=names, x=["x", "tt"],
                    lags=2)
    other = {c.term.replace(":tt", ":trend"): c for c in manual.coefficients}
    for term, row in rows.items():
        assert other[term].estimate == pytest.approx(row.estimate, rel=1e-8, abs=1e-10)
        assert other[term].std_error == pytest.approx(row.std_error, rel=1e-8)
    assert manual.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"])
    # (2) A constant supplied as an exogenous column equals the built-in constant.
    plain = oe.var(data=frame, y=names, lags=2)
    as_exog = oe.var(data=frame.assign(one=1.0), y=names, x=["one"], lags=2, constant=False)
    other = {c.term.replace(":one", ":Intercept"): c for c in as_exog.coefficients}
    for c in plain.coefficients:
        assert other[c.term].estimate == pytest.approx(c.estimate, rel=1e-8, abs=1e-10)
        assert other[c.term].std_error == pytest.approx(c.std_error, rel=1e-8)
    assert as_exog.metrics["log_likelihood"] == pytest.approx(plain.metrics["log_likelihood"])
    assert_allclose(as_exog.extra["sigma"], plain.extra["sigma"], rtol=1e-9)
    # (3) Rescaling an exogenous regressor rescales its coefficient only.
    scaled = oe.var(data=frame.assign(x=frame.x * 1000.0), y=names, x=["x"], lags=2, trend=True)
    for c in scaled.coefficients:
        factor = 1000.0 if c.term.endswith(":x") else 1.0
        assert c.estimate * factor == pytest.approx(rows[c.term].estimate, rel=1e-8, abs=1e-10)
        assert c.std_error * factor == pytest.approx(rows[c.term].std_error, rel=1e-8)
        assert c.p_value == pytest.approx(rows[c.term].p_value, rel=1e-6, abs=1e-12)
    # (4) Rescaling an endogenous variable: A[i, v] -> A[i, v] s_i / s_v, ll shifts by -T ln s.
    units = oe.var(data=frame.assign(v=frame.v * 250.0), y=names, x=["x"], lags=2, trend=True)
    for c in units.coefficients:
        equation, label = c.term.split(":", 1)
        factor = (250.0 if equation == "v" else 1.0) / (250.0 if label.endswith(".v") else 1.0)
        assert c.estimate / factor == pytest.approx(rows[c.term].estimate, rel=1e-7, abs=1e-10)
        assert c.statistic == pytest.approx(rows[c.term].statistic, rel=1e-6)
    assert units.metrics["log_likelihood"] == pytest.approx(
        base.metrics["log_likelihood"] - base.nobs * np.log(250.0), rel=1e-10)
    for name in ("normality", "lm_autocorrelation_L1", "lag_exclusion_L2", "granger_all_u"):
        assert units.tests[name]["statistic"] == pytest.approx(base.tests[name]["statistic"],
                                                               rel=1e-6)
    # (5) Rows in any order with a time column, integer or datetime, and missing ends dropped.
    shuffled = frame.sample(frac=1.0, random_state=5)
    by_time = oe.var(data=shuffled, y=names, x=["x"], time="when", lags=2, trend=True)
    assert_allclose(estimates(by_time)[0], estimates(base)[0], rtol=1e-10, atol=1e-12)
    dated = shuffled.assign(day=pd.Timestamp("2020-01-01") + pd.to_timedelta(shuffled.when, "D"))
    by_date = oe.var(data=dated, y=names, x=["x"], time="day", lags=2, trend=True)
    assert_allclose(estimates(by_date)[1], estimates(base)[1], rtol=1e-10)
    padded = pd.concat([frame.iloc[:3].assign(u=np.nan), frame,
                        frame.iloc[:2].assign(v=np.nan)], ignore_index=True)
    padded["when"] = np.arange(len(padded))
    dropped = oe.var(data=padded, y=names, x=["x"], time="when", lags=2, trend=True,
                     missing="drop")
    assert dropped.nobs == base.nobs and dropped.dropped_rows == 5 + 2
    slopes = ["Intercept" not in c.term for c in base.coefficients]
    assert_allclose(estimates(dropped)[0][slopes], estimates(base)[0][slopes], rtol=1e-8,
                    atol=1e-10)
    # (6) A categorical exogenous regressor equals hand-made indicators.
    levels = np.array(["hi", "lo", "mid"])[np.arange(len(frame)) % 3]
    coded = oe.var(data=frame.assign(g=levels), y=names, x=["g"], categorical=["g"], lags=1)
    dummies = oe.var(data=frame.assign(lo=(levels == "lo") * 1.0, mid=(levels == "mid") * 1.0),
                     y=names, x=["lo", "mid"], lags=1)
    assert [c.term for c in coded.coefficients if c.equation == "u"] == [
        "u:L1.u", "u:L1.v", "u:g[lo]", "u:g[mid]", "u:Intercept"]
    assert_allclose(estimates(coded)[0], estimates(dummies)[0], rtol=1e-10)
    assert_allclose(estimates(coded)[1], estimates(dummies)[1], rtol=1e-10)


def test_tiny_sample_matches_explicit_algebra():
    """Seven observations, two variables, one lag: T = 6, m = 3."""
    frame = pd.DataFrame({"u": [1.0, 2.5, 1.5, 3.0, 2.0, 3.5, 2.2],
                          "v": [0.5, 0.1, 0.9, 0.2, 1.1, 0.4, 0.8]})
    result = oe.var(data=frame, y=["u", "v"], lags=1, small=True, lm_lags=1)
    cols, y = columns(frame, ["u", "v"], 1)
    z = design_of(result, cols)
    t, m = z.shape
    assert (t, m) == (6, 3)
    beta = np.linalg.solve(z.T @ z, z.T @ y)
    u = y - z @ beta
    sigma = u.T @ u / t
    est, se = estimates(result)
    assert_allclose(est, beta.T.ravel(), rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(np.kron(sigma * t / (t - m), np.linalg.inv(z.T @ z)))),
                    rtol=1e-9)
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.t.sf(np.abs(est / se), t - m), rtol=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(
        -0.5 * t * (np.log(np.linalg.det(sigma)) + 2 * np.log(2 * np.pi) + 2), rel=1e-10)


# ---- oe.vecrank and oe.vec --------------------------------------------------------------------

def vec_frame(seed=31, n=220, k=3, level=0.0):
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n).cumsum(), rng.normal(size=n).cumsum()
    noise = rng.normal(size=(n, 4))
    noise[1:] += 0.4 * noise[:-1]
    data = {"y1": w1 + noise[:, 0], "y2": 0.6 * w1 + 2.0 + noise[:, 1],
            "y3": w2 - 0.5 * w1 + noise[:, 2], "y4": w2 + 0.05 * np.arange(n) + noise[:, 3]}
    frame = pd.DataFrame(data)[[f"y{i + 1}" for i in range(k)]] + level
    frame["period"] = np.arange(1, n + 1)
    return frame


def blocks(y, p, trend):
    """Z0, Z1, Z2 of the VEC regression in the ORIGINAL levels; t is the 1-based position."""
    n, k = y.shape
    dy = y[1:] - y[:-1]
    t = n - p
    index = np.arange(p + 1, n + 1, dtype=float)
    z0, z1 = dy[p - 1:], y[p - 1:n - 1]
    lagged = [dy[p - 1 - i:n - 1 - i, v] for v in range(k) for i in range(1, p)]
    z2 = np.column_stack(lagged) if lagged else np.empty((t, 0))
    z2_labels = [("LD." if i == 1 else f"L{i}D.") + f"y{v + 1}" for v in range(k)
                 for i in range(1, p)]
    if trend == "rconstant":
        z1 = np.column_stack([z1, np.ones(t)])
    elif trend == "rtrend":
        z1 = np.column_stack([z1, index])
    if trend == "trend":
        z2, z2_labels = np.column_stack([z2, index]), [*z2_labels, "trend"]
    if trend in ("constant", "rtrend", "trend"):
        z2, z2_labels = np.column_stack([z2, np.ones(t)]), [*z2_labels, "Intercept"]
    return z0, z1, z2, index, z2_labels


def residualize(a, z2):
    return a if z2.shape[1] == 0 else a - z2 @ np.linalg.lstsq(z2, a, rcond=None)[0]


def canonical(y, p, trend):
    """Eigenvalues as squared canonical correlations of the residual blocks (QR + SVD)."""
    z0, z1, z2, _, _ = blocks(y, p, trend)
    r0, r1 = residualize(z0, z2), residualize(z1, z2)
    q0, q1 = np.linalg.qr(r0)[0], np.linalg.qr(r1)[0]
    correlations = np.linalg.svd(q0.T @ q1, compute_uv=False)
    t, k = z0.shape
    return correlations[:k] ** 2, np.linalg.slogdet(r0.T @ r0 / t)[1], t, z1.shape[1], z2.shape[1]


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("k, p", [(4, 3), (3, 1), (2, 2)])
def test_vecrank_by_canonical_correlations(trend, k, p):
    frame = vec_frame(k=k)
    names = [f"y{i + 1}" for i in range(k)]
    table = oe.vecrank(data=frame, y=names, time="period", lags=p, trend=trend)
    values, logdet_s00, t, m1, m2 = canonical(frame[names].to_numpy(), p, trend)
    assert table.attrs["n_obs"] == t == len(frame) - p
    assert_allclose(table.attrs["eigenvalues"], values, rtol=1e-8, atol=1e-12)
    assert_allclose(table["eigenvalue"].to_numpy()[1:], values, rtol=1e-8, atol=1e-12)
    logs = np.log1p(-values)
    assert_allclose(table["trace"].to_numpy()[:k], [-t * logs[r:].sum() for r in range(k)],
                    rtol=1e-8)
    assert_allclose(table["max"].to_numpy()[:k], -t * logs, rtol=1e-8)
    ll = np.array([-0.5 * t * (k * (np.log(2 * np.pi) + 1) + logdet_s00 + logs[:r].sum())
                   for r in range(k + 1)])
    assert_allclose(table["ll"], ll, rtol=1e-10)
    # Likelihood-ratio form of both statistics.
    assert_allclose(table["trace"].to_numpy()[:k], 2 * (ll[k] - ll[:k]), rtol=1e-7, atol=1e-7)
    assert_allclose(table["max"].to_numpy()[:k], 2 * np.diff(ll), rtol=1e-7, atol=1e-7)
    parms = [k * m2 + (k + m1 - r) * r for r in range(k + 1)]
    assert list(table["parms"]) == parms
    assert_allclose(table["aic"], (-2 * ll + 2 * np.array(parms)) / t, rtol=1e-10)
    assert_allclose(table["sbic"], (-2 * ll + np.log(t) * np.array(parms)) / t, rtol=1e-10)
    assert_allclose(table["hqic"], (-2 * ll + 2 * np.log(np.log(t)) * np.array(parms)) / t,
                    rtol=1e-10)
    rule = next((r for r in range(k) if table["trace"][r] <= table["trace_cv5"][r]), k)
    assert table.attrs["selected_rank"] == rule
    rule = next((r for r in range(k) if table["trace"][r] <= table["trace_cv1"][r]), k)
    assert table.attrs["selected_rank_1pct"] == rule
    rule = next((r for r in range(k) if table["max"][r] <= table["max_cv5"][r]), k)
    assert table.attrs["selected_rank_max"] == rule
    assert np.isnan(table["trace"][k]) and np.isnan(table["eigenvalue"][0])


@pytest.mark.parametrize("p", [1, 2, 3])
def test_vec_model_is_nested_between_two_vars(p):
    """Rank K is the VAR(p) in levels, rank 0 the VAR(p - 1) in first differences."""
    frame = vec_frame(k=3, seed=4)
    names = ["y1", "y2", "y3"]
    tables = {trend: oe.vecrank(data=frame, y=names, lags=p, trend=trend) for trend in TRENDS}
    k = 3
    levels = {"none": dict(constant=False), "constant": {}, "trend": dict(trend=True)}
    for trend, options in levels.items():
        full = oe.var(data=frame, y=names, lags=p, **options)
        assert tables[trend]["ll"][k] == pytest.approx(full.metrics["log_likelihood"], rel=1e-9)
        assert tables[trend]["parms"][k] == full.metrics["df_model"]
        assert tables[trend]["aic"][k] == pytest.approx(full.metrics["aic_per_obs"], rel=1e-9)
    # A restricted term stops being a restriction at full rank.
    assert tables["rconstant"]["ll"][k] == pytest.approx(tables["constant"]["ll"][k], rel=1e-10)
    assert tables["rtrend"]["ll"][k] == pytest.approx(tables["trend"]["ll"][k], rel=1e-10)
    assert tables["rconstant"]["parms"][k] == tables["constant"]["parms"][k]
    # ... and the models are nested: none < rconstant < constant < rtrend < trend at every rank.
    for r in range(k + 1):
        chain = [tables[trend]["ll"][r] for trend in TRENDS]
        assert all(b >= a - 1e-8 for a, b in pairwise(chain))
        assert all(tables[trend]["ll"][r] <= tables[trend]["ll"][k] + 1e-9 for trend in TRENDS)
    if p > 1:
        differences = frame[names].diff().iloc[1:].reset_index(drop=True)
        for trend, options in levels.items():
            short = oe.var(data=differences, y=names, lags=p - 1, **options)
            assert short.nobs == tables[trend].attrs["n_obs"]
            assert tables[trend]["ll"][0] == pytest.approx(short.metrics["log_likelihood"],
                                                           rel=1e-9)
            assert tables[trend]["parms"][0] == short.metrics["df_model"]


def reported(result, names, p, trend):
    """alpha, beta, constants, Gamma and deterministic terms read from a vec result by name."""
    k, r = len(names), int(result.metrics["rank"])
    rows = {c.term: c.estimate for c in result.coefficients}
    alpha = np.array([[rows[f"D_{n}:L._ce{i + 1}"] for i in range(r)] for n in names])
    gamma = np.array([[[rows[f"D_{n}:{'LD' if i == 1 else f'L{i}D'}.{v}"] for v in names]
                       for n in names] for i in range(1, p)]).reshape(p - 1, k, k)
    intercept = np.array([rows.get(f"D_{n}:Intercept", 0.0) for n in names])
    slope = np.array([rows.get(f"D_{n}:trend", 0.0) for n in names])
    beta = np.zeros((k, r))
    constant, drift = np.zeros(r), np.zeros(r)
    for i, equation in enumerate(result.extra["beta"]):
        entries = {e["variable"]: e["estimate"] for e in equation["coefficients"]}
        beta[:, i] = [entries[n] for n in names]
        constant[i], drift[i] = entries.get("_cons", 0.0), entries.get("_trend", 0.0)
    return dict(alpha=alpha, beta=beta, constant=constant, drift=drift, gamma=gamma,
                intercept=intercept, slope=slope)


def one_step(pieces, history, position):
    """D y_t from the reported pieces; history[0] = y_(t-1), position = 1-based index of t."""
    ce = pieces["beta"].T @ history[0] + pieces["constant"] + pieces["drift"] * position
    out = pieces["alpha"] @ ce + pieces["intercept"] + pieces["slope"] * position
    for i, gamma in enumerate(pieces["gamma"]):
        out = out + gamma @ (history[i] - history[i + 1])
    return out


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("rank, p, level", [(1, 2, 0.0), (2, 3, 0.0), (1, 1, 0.0),
                                            (2, 2, 5000.0)])
def test_vec_reported_pieces_reproduce_residuals_and_forecasts(trend, rank, p, level):
    """Everything a user reads from the result is mutually consistent, in the original levels."""
    frame = vec_frame(k=3, seed=12, level=level)
    names = ["y1", "y2", "y3"]
    result = oe.vec(data=frame, y=names, lags=p, rank=rank, trend=trend)
    y = frame[names].to_numpy()
    n, k = y.shape
    t = n - p
    pieces = reported(result, names, p, trend)
    assert_allclose(pieces["beta"][:rank], np.eye(rank), atol=0)
    assert_allclose(result.extra["beta_matrix"], pieces["beta"], rtol=1e-12)
    assert_allclose(result.extra["alpha"], pieces["alpha"], rtol=1e-12)
    resid = np.array([y[row] - y[row - 1] - one_step(
        pieces, [y[row - 1 - i] for i in range(p)], row + 1) for row in range(p, n)])
    omega = resid.T @ resid / t
    assert_allclose(result.extra["omega"], omega, rtol=1e-6, atol=1e-9)
    assert abs(resid.mean(axis=0)).max() < (1e-6 if trend not in ("none", "rconstant") else 1.0)
    # The likelihood of these residuals is the reported one.
    ll = -0.5 * t * (np.linalg.slogdet(omega)[1] + k * np.log(2 * np.pi) + k)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-8)
    assert result.metrics["det_sigma_ml"] == pytest.approx(np.linalg.det(omega), rel=1e-6)
    # The eigenvalues give the same likelihood through Johansen's formula.
    values, logdet_s00, *_ = canonical(y, p, trend)
    assert result.metrics["log_likelihood"] == pytest.approx(
        -0.5 * t * (k * (np.log(2 * np.pi) + 1) + logdet_s00 + np.log1p(-values[:rank]).sum()),
        rel=1e-9)
    # Dynamic forecasts by iterating the error-correction form.
    steps = 5
    history = [y[n - 1 - i] for i in range(p)]
    path = []
    for h in range(steps):
        value = history[0] + one_step(pieces, history, n + h + 1)
        path.append(value)
        history = [value, *history[:-1]]
    table = oe.forecast(result, steps)
    assert_allclose(table.forecast.to_numpy().reshape(k, steps).T, np.array(path),
                    rtol=1e-8, atol=1e-6)
    # The VAR representation carries the same dynamics: I - sum A_i = -alpha beta'.
    a = np.array(result.extra["var_representation"]["A"])
    assert_allclose(np.eye(k) - a.sum(axis=0), -pieces["alpha"] @ pieces["beta"].T, atol=1e-9)
    assert_allclose(result.extra["pi"], pieces["alpha"] @ pieces["beta"].T, atol=1e-10)
    moduli = [e["modulus"] for e in result.extra["stability"]["eigenvalues"]]
    assert_allclose(moduli[:k - rank], 1.0, atol=1e-7)
    assert result.extra["stability"]["unit_moduli_imposed"] == k - rank
    # Forecast standard errors: T / (T - d) sum_i Phi_i Omega Phi_i' from the companion form.
    d = int(result.metrics["df_eq"])
    companion = np.zeros((k * p, k * p))
    companion[:k] = np.hstack(list(a))
    companion[k:, :k * (p - 1)] = np.eye(k * (p - 1))
    q = np.zeros((k * p, k * p))
    q[:k, :k] = omega * t / (t - d)
    power, total, errors = np.eye(k * p), np.zeros((k * p, k * p)), []
    for _ in range(steps):
        total = total + power @ q @ power.T
        errors.append(np.sqrt(np.diag(total[:k, :k])))
        power = companion @ power
    assert_allclose(table.std_error.to_numpy().reshape(k, steps).T, np.array(errors), rtol=1e-6)


def vec_loglik(y, p, trend, pieces, omega, *, concentrate):
    """Gaussian log likelihood of a VEC model written in the original levels.

    ``concentrate=True`` maximizes over the coefficients of Z2 (lagged
    differences and unrestricted deterministic terms) for the given alpha,
    beta and restricted terms, by least squares.
    """
    z0, z1, z2, index, _ = blocks(y, p, trend)
    t, k = z0.shape
    ce = z1[:, :k] @ pieces["beta"] + pieces["constant"] + np.outer(index, pieces["drift"])
    error = z0 - ce @ pieces["alpha"].T
    if concentrate:
        error = residualize(error, z2)
    else:
        n = len(y)
        dy = y[1:] - y[:-1]
        for i, gamma in enumerate(pieces["gamma"], start=1):
            error = error - dy[p - 1 - i:n - 1 - i] @ gamma.T
        error = error - pieces["intercept"] - np.outer(index, pieces["slope"])
    inverse = np.linalg.inv(omega)
    return (-0.5 * t * np.linalg.slogdet(omega)[1] - 0.5 * np.einsum("ti,ij,tj->", error,
                                                                      inverse, error)
            - 0.5 * t * k * np.log(2 * np.pi))


@pytest.mark.parametrize("trend", TRENDS)
@pytest.mark.parametrize("rank, level", [(1, 0.0), (2, 0.0), (2, 300.0)])
def test_vec_covariances_are_inverse_hessians_of_an_independent_likelihood(trend, rank, level):
    """Both covariance formulas of Stata's vec, from numerically differentiated likelihoods.

    Short-run parameters: Hessian with beta, mu, rho and Omega held fixed
    (the VAR in differences given the cointegrating equations), divisor T - d.
    Free elements of beta: Hessian of the likelihood concentrated over the
    Z2 coefficients with alpha and Omega held fixed, divisor T - d.
    """
    frame = vec_frame(k=3, seed=44, level=level)
    names, p = ["y1", "y2", "y3"], 2
    result = oe.vec(data=frame, y=names, lags=p, rank=rank, trend=trend)
    y = frame[names].to_numpy()
    k = 3
    t, d = int(result.metrics["T"]), int(result.metrics["df_eq"])
    pieces = reported(result, names, p, trend)
    omega = np.array(result.extra["omega"])
    labels = labels_of(result)
    m = len(labels)
    _, z1, z2, _, z2_labels = blocks(y, p, trend)
    assert labels == [f"L._ce{i + 1}" for i in range(rank)] + z2_labels
    assert d == (k * z2.shape[1] + (k + z1.shape[1] - rank) * rank) // k
    # (a) short-run parameters, in the order of the coefficient table.
    est, se = estimates(result)

    def short_run(theta):
        coef = theta.reshape(k, m)
        moved = dict(pieces, alpha=coef[:, :rank])
        gammas = [np.column_stack([coef[:, rank + v * (p - 1) + i] for v in range(k)])
                  for i in range(p - 1)]
        moved["gamma"] = np.array(gammas).reshape(p - 1, k, k)
        if "trend" in labels:
            moved["slope"] = coef[:, labels.index("trend")]
        if "Intercept" in labels:
            moved["intercept"] = coef[:, labels.index("Intercept")]
        return vec_loglik(y, p, trend, moved, omega, concentrate=False)

    assert short_run(est) == pytest.approx(result.metrics["log_likelihood"], rel=1e-9)
    steps = np.maximum(np.abs(se), 1e-8) * 0.5
    hessian = numerical_hessian(short_run, est, steps)
    covariance = np.linalg.inv(-hessian) * t / (t - d)
    assert_allclose(np.array(result.covariance_matrix), covariance, rtol=2e-5, atol=1e-13)
    assert result.inference["small_sample_correction"] == pytest.approx(t / (t - d))
    # (b) free elements of beta (and the restricted constant / trend), by equation.
    free = [(i, e["variable"], e["estimate"], e["std_error"])
            for i, equation in enumerate(result.extra["beta"])
            for e in equation["coefficients"] if e["status"] == "estimated"]
    extra = {"rconstant": ["_cons"], "rtrend": ["_trend"]}.get(trend, [])
    assert len(free) == rank * (k - rank + len(extra))

    def long_run(theta):
        moved = dict(pieces, beta=pieces["beta"].copy(), constant=pieces["constant"].copy(),
                     drift=pieces["drift"].copy())
        for value, (i, variable, _, _) in zip(theta, free, strict=True):
            if variable == "_cons":
                moved["constant"][i] = value
            elif variable == "_trend":
                moved["drift"][i] = value
            else:
                moved["beta"][names.index(variable), i] = value
        return vec_loglik(y, p, trend, moved, omega, concentrate=True)

    point = np.array([value for _, _, value, _ in free])
    errors = np.array([error for _, _, _, error in free])
    assert long_run(point) == pytest.approx(result.metrics["log_likelihood"], rel=1e-9)
    hessian = numerical_hessian(long_run, point, errors * 0.5)
    covariance = np.linalg.inv(-hessian) * t / (t - d)
    assert_allclose(errors, np.sqrt(np.diag(covariance)), rtol=5e-5)
    # Stata's "Cointegrating equations" table: Wald test of the K - r coefficients on variables.
    for i, equation in enumerate(result.extra["beta"]):
        where = [j for j, (ce, variable, _, _) in enumerate(free)
                 if ce == i and not variable.startswith("_")]
        b = point[where]
        wald = b @ np.linalg.solve(covariance[np.ix_(where, where)], b)
        assert equation["parms"] == equation["df"] == k - rank
        assert equation["statistic"] == pytest.approx(wald, rel=1e-4)
        assert equation["p_value"] == pytest.approx(stats.chi2.sf(wald, k - rank), rel=1e-3,
                                                    abs=1e-300)
        if k - rank == 1:
            z = next(e["statistic"] for e in equation["coefficients"]
                     if e["status"] == "estimated" and not e["variable"].startswith("_"))
            assert equation["statistic"] == pytest.approx(z ** 2, rel=1e-9)
    # Backed-out and normalized elements carry no standard error.
    for equation in result.extra["beta"]:
        for e in equation["coefficients"]:
            assert (e["std_error"] is None) == (e["status"] != "estimated")


@pytest.mark.parametrize("trend", ["none", "rconstant", "constant"])
def test_johansen_estimates_maximize_an_independent_likelihood(trend):
    """Brute-force maximization over alpha, the free beta and the Z2 coefficients (K = 2, r = 1)."""
    frame = vec_frame(k=2, seed=9, n=160)
    names, p = ["y1", "y2"], 2
    y = frame[names].to_numpy()
    result = oe.vec(data=frame, y=names, lags=p, rank=1, trend=trend)
    z0, z1, z2, _, _ = blocks(y, p, trend)
    t, k = z0.shape
    free = z1.shape[1] - 1

    def negative(theta):
        alpha, beta = theta[:k], np.concatenate([[1.0], theta[k:k + free]])
        error = residualize(z0 - np.outer(z1 @ beta, alpha), z2)
        return 0.5 * t * np.linalg.slogdet(error.T @ error / t)[1]

    start = np.concatenate([[-0.1, 0.1], [-1.0], np.zeros(free - 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = minimize(negative, start, method="Nelder-Mead",
                        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000,
                                 "maxfev": 40000})
        best = minimize(negative, best.x, method="BFGS", options={"gtol": 1e-9})
    ll = -best.fun - 0.5 * t * k * (np.log(2 * np.pi) + 1)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-9)
    pieces = reported(result, names, p, trend)
    assert_allclose(pieces["alpha"][:, 0], best.x[:k], rtol=2e-4, atol=1e-6)
    assert pieces["beta"][1, 0] == pytest.approx(best.x[k], rel=2e-4)
    if trend == "rconstant":
        assert pieces["constant"][0] == pytest.approx(best.x[k + 1], rel=5e-4, abs=1e-5)
    # No other rank-one decomposition does better: perturbing the optimum lowers the likelihood.
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert negative(best.x + rng.normal(scale=1e-3, size=len(best.x))) >= best.fun - 1e-9


@pytest.mark.parametrize("trend", TRENDS)
def test_vec_header_follows_stata(trend):
    """Uncentered R-squared, chi2 over all coefficients with T - d, RMSE with T - d."""
    frame = vec_frame(k=3, seed=23)
    names, p, rank = ["y1", "y2", "y3"], 2, 1
    result = oe.vec(data=frame, y=names, lags=p, rank=rank, trend=trend)
    y = frame[names].to_numpy()
    n, k = y.shape
    t, d = n - p, int(result.metrics["df_eq"])
    pieces = reported(result, names, p, trend)
    resid = np.array([y[row] - y[row - 1] - one_step(
        pieces, [y[row - 1 - i] for i in range(p)], row + 1) for row in range(p, n)])
    dy = (y[1:] - y[:-1])[p - 1:]
    m = len(labels_of(result))
    count = int(result.metrics["df_model"])
    assert d == count // k and result.metrics["rank"] == rank
    for i, row in enumerate(result.extra["equations"]):
        ssr = resid[:, i] @ resid[:, i]
        uncentered = 1 - ssr / (dy[:, i] @ dy[:, i])
        assert row["equation"] == f"D_{names[i]}" and row["parms"] == m == row["df"]
        assert row["r_squared"] == pytest.approx(uncentered, rel=1e-8)
        assert row["rmse"] == pytest.approx(np.sqrt(ssr / (t - d)), rel=1e-8)
        assert row["statistic"] == pytest.approx((t - d) * uncentered / (1 - uncentered), rel=1e-6)
        assert row["p_value"] == pytest.approx(stats.chi2.sf(row["statistic"], m), rel=1e-6,
                                               abs=1e-300)
        assert row["distribution"] == "chi2"
        if trend in ("constant", "rtrend", "trend"):
            centered = dy[:, i] - dy[:, i].mean()
            assert row["r_squared_centered"] == pytest.approx(1 - ssr / (centered @ centered),
                                                              rel=1e-8)
            assert row["r_squared_centered"] < row["r_squared"]
        else:
            assert "r_squared_centered" not in row
    ll = result.metrics["log_likelihood"]
    assert result.metrics["aic_per_obs"] == pytest.approx((-2 * ll + 2 * count) / t, rel=1e-12)
    assert result.metrics["sbic_per_obs"] == pytest.approx((-2 * ll + np.log(t) * count) / t,
                                                           rel=1e-12)
    assert result.metrics["hqic_per_obs"] == pytest.approx(
        (-2 * ll + 2 * np.log(np.log(t)) * count) / t, rel=1e-12)
    assert result.metrics["bic"] == pytest.approx(-2 * ll + np.log(t) * count, rel=1e-12)
    # e(V_alpha) = Omega (x) (beta' S11 beta)^-1 / (T - d) is the alpha block of the covariance.
    _, z1, z2, index, _ = blocks(y, p, trend)
    full_beta = np.vstack([pieces["beta"]] + ([pieces["constant"][None]] if trend == "rconstant"
                                              else [pieces["drift"][None]] if trend == "rtrend"
                                              else []))
    r1 = residualize(z1, z2)
    middle = full_beta.T @ (r1.T @ r1 / t) @ full_beta
    v_alpha = np.kron(np.array(result.extra["omega"]), np.linalg.inv(middle)) / (t - d)
    where = [i * m + j for i in range(k) for j in range(rank)]
    assert_allclose(np.array(result.covariance_matrix)[np.ix_(where, where)], v_alpha,
                    rtol=1e-6)


def test_vec_is_equivariant_to_units_and_exactly_invariant_to_levels():
    frame = vec_frame(k=3, seed=6)
    names = ["y1", "y2", "y3"]
    scale = np.array([1e6, 1.0, 1e-5])
    for trend in TRENDS:
        base = oe.vec(data=frame, y=names, lags=2, rank=2, trend=trend)
        scaled = frame.copy()
        scaled[names] = frame[names].to_numpy() * scale
        other = oe.vec(data=scaled, y=names, lags=2, rank=2, trend=trend)
        b0, b1 = np.array(base.extra["beta_matrix"]), np.array(other.extra["beta_matrix"])
        a0, a1 = np.array(base.extra["alpha"]), np.array(other.extra["alpha"])
        assert_allclose(b1[2] * scale[2] / scale[:2], b0[2], rtol=1e-8)   # beta_ji s_j / s_i
        assert_allclose(a1 / scale[:, None] * scale[None, :2], a0, rtol=1e-8)
        se0 = [e["std_error"] for eq in base.extra["beta"] for e in eq["coefficients"]
               if e["variable"] == "y3"]
        se1 = [e["std_error"] for eq in other.extra["beta"] for e in eq["coefficients"]
               if e["variable"] == "y3"]
        assert_allclose(np.array(se1) * scale[2] / scale[:2], se0, rtol=1e-7)
        assert other.metrics["log_likelihood"] == pytest.approx(
            base.metrics["log_likelihood"] - base.nobs * np.log(scale).sum(), rel=1e-10)
        for name in ("normality", "lm_autocorrelation_L1"):
            assert other.tests[name]["statistic"] == pytest.approx(
                base.tests[name]["statistic"], rel=1e-6)
        assert_allclose([r["statistic"] for r in other.extra["equations"]],
                        [r["statistic"] for r in base.extra["equations"]], rtol=1e-7)
        rank_a = oe.vecrank(data=frame, y=names, lags=2, trend=trend)
        rank_b = oe.vecrank(data=scaled, y=names, lags=2, trend=trend)
        assert_allclose(rank_b["trace"].to_numpy()[:3], rank_a["trace"].to_numpy()[:3], rtol=1e-8)
        if trend == "none":
            continue                       # without a constant the level is part of the model
        shifted = frame.copy()
        shifted[names] = frame[names] + 1e6
        moved = oe.vec(data=shifted, y=names, lags=2, rank=2, trend=trend)
        assert moved.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"],
                                                                abs=1e-6)
        assert_allclose(moved.extra["beta_matrix"], b0, rtol=1e-8, atol=1e-10)
        assert_allclose(moved.extra["alpha"], a0, rtol=1e-7, atol=1e-10)
        # The constant of each cointegrating equation moves by -beta' 1e6, exactly.
        assert_allclose(moved.extra["ce_constant"],
                        np.array(base.extra["ce_constant"]) - b0.sum(axis=0) * 1e6, rtol=1e-9)
        slopes = ["Intercept" not in c.term for c in base.coefficients]
        assert_allclose(estimates(moved)[1][slopes], estimates(base)[1][slopes], rtol=1e-7)
        assert_allclose(oe.forecast(moved, 4).forecast - 1e6, oe.forecast(base, 4).forecast,
                        atol=1e-5)
        shifted_rank = oe.vecrank(data=shifted, y=names, lags=2, trend=trend)
        assert_allclose(shifted_rank["trace"].to_numpy()[:3], rank_a["trace"].to_numpy()[:3],
                        rtol=1e-8)
