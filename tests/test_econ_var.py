"""Independent oracles for oe.var and oe.varsoc.

Coefficients, covariance and fit statistics are checked against explicit NumPy
algebra, the statsmodels VAR and equation-by-equation statsmodels OLS; the
Stata conventions (ML Sigma, dfk, small, information criteria, RMSE, the LM
autocorrelation statistic, varsoc) against the output printed in the Stata
manual for the Lutkepohl data, which ships with statsmodels' test-suite.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import pytest
import statsmodels
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ModelSpec, ResultBundle

NAMES = ["a", "b", "c"]


def make_data(seed=1, n=240, exog=False):
    rng = np.random.default_rng(seed)
    k = 3
    mix = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.2, -0.3, 1.0]])
    e = rng.normal(size=(n, k)) @ mix.T
    a1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.3, 0.1], [0.0, -0.2, 0.4]])
    a2 = np.array([[-0.1, 0.0, 0.1], [0.0, 0.1, 0.0], [0.1, 0.0, -0.2]])
    x = rng.normal(size=n)
    y = np.zeros((n, k))
    for t in range(2, n):
        y[t] = 0.3 + a1 @ y[t - 1] + a2 @ y[t - 2] + e[t] + (0.4 * x[t] if exog else 0.0)
    frame = pd.DataFrame(y, columns=NAMES)
    frame["x"] = x
    frame["g"] = pd.Categorical(np.where(rng.uniform(size=n) > 0.5, "hi", "lo"))
    frame["period"] = np.arange(1001, 1001 + n)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def fitted(data):
    return oe.var(data=data, y=NAMES, lags=2)


def design(frame, names, p, extra=(), constant=True, trend=False):
    """Regressors in OpenEcon's order: lags by variable, exogenous, trend, constant."""
    y = frame[names].to_numpy()
    n = len(y)
    columns = [y[p - j:n - j, v] for v in range(len(names)) for j in range(1, p + 1)]
    columns += [np.asarray(col, dtype=float)[p:] for col in extra]
    if trend:
        columns.append(np.arange(p + 1, n + 1, dtype=float))
    if constant:
        columns.append(np.ones(n - p))
    return np.column_stack(columns), y[p:]


def ols_system(z, y):
    b = np.linalg.lstsq(z, y, rcond=None)[0].T          # [K, m]
    u = y - z @ b.T
    return b, u, u.T @ u / len(y), np.linalg.inv(z.T @ z)


def estimates(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


def loglik(sigma, t):
    k = len(sigma)
    return -0.5 * t * (np.log(np.linalg.det(sigma)) + k * np.log(2 * np.pi) + k)


def test_coefficients_covariance_and_metrics_match_numpy(data, fitted):
    z, y = design(data, NAMES, 2)
    b, u, sigma, zzinv = ols_system(z, y)
    t, m, k = len(y), z.shape[1], 3
    est, se = estimates(fitted)
    assert_allclose(est, b.ravel(), rtol=1e-10, atol=1e-12)
    cov = np.kron(sigma, zzinv)
    assert_allclose(np.array(fitted.covariance_matrix), cov, rtol=1e-9, atol=1e-14)
    assert_allclose(se, np.sqrt(np.diag(cov)), rtol=1e-9)
    ll = loglik(sigma, t)
    metrics = fitted.metrics
    assert_allclose(metrics["log_likelihood"], ll, rtol=1e-12)
    assert_allclose(metrics["aic"], -2 * ll + 2 * k * m, rtol=1e-12)
    assert_allclose(metrics["bic"], -2 * ll + np.log(t) * k * m, rtol=1e-12)
    assert_allclose(metrics["hqic"], -2 * ll + 2 * np.log(np.log(t)) * k * m, rtol=1e-12)
    assert_allclose(metrics["aic_per_obs"], metrics["aic"] / t, rtol=1e-12)
    assert_allclose(metrics["sbic_per_obs"], metrics["bic"] / t, rtol=1e-12)
    assert_allclose(metrics["hqic_per_obs"], metrics["hqic"] / t, rtol=1e-12)
    assert_allclose(metrics["det_sigma_ml"], np.linalg.det(sigma), rtol=1e-10)
    assert_allclose(metrics["fpe"], np.linalg.det(sigma) * ((t + m) / (t - m)) ** k, rtol=1e-10)
    assert (metrics["n_equations"], metrics["n_lags"], metrics["df_eq"]) == (3, 2, 7)
    assert metrics["T"] == t
    assert fitted.nobs == t and fitted.dropped_rows == 2
    assert fitted.sample_positions == list(range(2, len(data)))
    assert fitted.inference["distribution"] == "normal" and not fitted.inference["use_t"]
    z_stat = est / se
    assert_allclose([c.p_value for c in fitted.coefficients], 2 * stats.norm.sf(np.abs(z_stat)),
                    rtol=1e-8, atol=1e-300)
    assert_allclose(np.array(fitted.extra["sigma"]), sigma, rtol=1e-10)
    # Terms: Stata's order, grouped by equation.
    assert [c.term for c in fitted.coefficients[:7]] == [
        "a:L1.a", "a:L2.a", "a:L1.b", "a:L2.b", "a:L1.c", "a:L2.c", "a:Intercept"]
    assert {c.equation for c in fitted.coefficients} == set(NAMES)
    # Chart sample: first equation.
    first = fitted.predictions[0]
    assert first["row"] == 2
    assert_allclose(first["fitted"], (z @ b[0])[0], rtol=1e-10)


def test_matches_statsmodels_var(data, fitted):
    reference = VAR(data[NAMES]).fit(2, trend="c")
    ours = np.array([c.estimate for c in fitted.coefficients]).reshape(3, 7)
    theirs = reference.params.to_numpy().T              # [K, const + lags by lag then variable]
    assert_allclose(ours[:, 6], theirs[:, 0], rtol=1e-9)
    for v in range(3):
        for j in range(2):
            assert_allclose(ours[:, v * 2 + j], theirs[:, 1 + j * 3 + v], rtol=1e-8, atol=1e-12)
    assert_allclose(fitted.metrics["log_likelihood"], reference.llf, rtol=1e-11)
    # statsmodels drops the constant K (1 + ln 2 pi) of the likelihood from its criteria.
    shift = 3 * (1 + np.log(2 * np.pi))
    assert_allclose(fitted.metrics["aic_per_obs"], reference.aic + shift, rtol=1e-10)
    assert_allclose(fitted.metrics["sbic_per_obs"], reference.bic + shift, rtol=1e-10)
    assert_allclose(fitted.metrics["hqic_per_obs"], reference.hqic + shift, rtol=1e-10)
    assert_allclose(fitted.metrics["fpe"], reference.fpe, rtol=1e-10)
    assert_allclose(fitted.metrics["det_sigma_ml"], np.linalg.det(reference.sigma_u_mle), rtol=1e-9)
    # statsmodels always uses the degrees-of-freedom adjusted covariance: our dfk.
    adjusted = oe.var(data=data, y=NAMES, lags=2, dfk=True)
    assert_allclose(np.array(adjusted.extra["sigma"]), reference.sigma_u.to_numpy(), rtol=1e-10)
    ours_se = np.array([c.std_error for c in adjusted.coefficients]).reshape(3, 7)
    theirs_se = reference.stderr.to_numpy().T
    assert_allclose(ours_se[:, 6], theirs_se[:, 0], rtol=1e-8)
    assert_allclose(ours_se[:, 0], theirs_se[:, 1], rtol=1e-8)
    # dfk leaves the estimates, the likelihood and the criteria unchanged.
    assert_allclose(estimates(adjusted)[0], estimates(fitted)[0], rtol=1e-12)
    assert adjusted.metrics["log_likelihood"] == pytest.approx(fitted.metrics["log_likelihood"])
    assert_allclose(np.array(adjusted.extra["sigma_ml"]), np.array(fitted.extra["sigma"]),
                    rtol=1e-12)


def test_small_matches_equation_by_equation_ols(data):
    result = oe.var(data=data, y=NAMES, lags=2, small=True)
    z, y = design(data, NAMES, 2)
    t, m = z.shape
    assert result.inference["use_t"] and result.inference["df_inference"] == t - m
    for i, name in enumerate(NAMES):
        ols = sm.OLS(y[:, i], z).fit()
        rows = [c for c in result.coefficients if c.equation == name]
        assert_allclose([c.estimate for c in rows], ols.params, rtol=1e-9, atol=1e-12)
        assert_allclose([c.std_error for c in rows], ols.bse, rtol=1e-9)
        assert_allclose([c.p_value for c in rows], ols.pvalues, rtol=1e-7, atol=1e-300)
        assert_allclose([c.ci_low for c in rows], ols.conf_int()[:, 0], rtol=1e-7, atol=1e-10)
        record = result.extra["equations"][i]
        assert record["distribution"] == "F" and record["df"] == m - 1 and record["df2"] == t - m
        assert_allclose(record["statistic"], ols.fvalue, rtol=1e-8)
        assert_allclose(record["p_value"], ols.f_pvalue, rtol=1e-7)
        assert_allclose(record["r_squared"], ols.rsquared, rtol=1e-10)
        assert_allclose(record["rmse"], np.sqrt(ols.mse_resid), rtol=1e-10)
    # small changes the standard errors but not Sigma; dfk changes Sigma.
    plain = oe.var(data=data, y=NAMES, lags=2)
    assert_allclose(np.array(result.extra["sigma"]), np.array(plain.extra["sigma"]), rtol=1e-12)
    both = oe.var(data=data, y=NAMES, lags=2, small=True, dfk=True)
    assert_allclose(estimates(both)[1], estimates(result)[1], rtol=1e-12)
    assert_allclose(np.array(both.extra["sigma"]), np.array(plain.extra["sigma"]) * t / (t - m),
                    rtol=1e-12)


def test_equation_table_default_is_chi2(data, fitted):
    z, y = design(data, NAMES, 2)
    t, m = z.shape
    for i in range(3):
        ols = sm.OLS(y[:, i], z).fit()
        record = fitted.extra["equations"][i]
        assert record["parms"] == m and record["distribution"] == "chi2" and record["df"] == m - 1
        r2 = ols.rsquared
        assert_allclose(record["statistic"], t * r2 / (1 - r2), rtol=1e-8)
        assert_allclose(record["p_value"], stats.chi2.sf(record["statistic"], m - 1), rtol=1e-8)
        assert_allclose(record["rmse"], np.sqrt(ols.ssr / (t - m)), rtol=1e-10)


@pytest.mark.parametrize("options, factor", [({}, "t1"), ({"small": True}, "tm"),
                                             ({"dfk": True}, "tm")])
def test_robust_covariance(data, options, factor):
    result = oe.var(data=data, y=NAMES, lags=2, covariance="robust", **options)
    z, y = design(data, NAMES, 2)
    b, u, _, zzinv = ols_system(z, y)
    t, m = z.shape
    scores = np.einsum("ti,tm->tim", u, z).reshape(t, -1)          # u_t (x) z_t, equation-major
    bread = np.kron(np.eye(3), zzinv)
    scale = t / (t - 1) if factor == "t1" else t / (t - m)
    cov = bread @ (scores.T @ scores) @ bread * scale
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-8, atol=1e-14)
    assert result.inference["small_sample_correction"] == pytest.approx(scale)
    # Diagonal blocks are the White covariances of the single equations.
    for i in range(3):
        ols = sm.OLS(y[:, i], z).fit(cov_type="HC0")
        rows = [c.std_error for c in result.coefficients if c.equation == NAMES[i]]
        assert_allclose(rows, ols.bse * np.sqrt(scale), rtol=1e-8)
    if options.get("small"):
        hc1 = sm.OLS(y[:, 0], z).fit(cov_type="HC1")
        assert_allclose([c.std_error for c in result.coefficients if c.equation == "a"], hc1.bse,
                        rtol=1e-8)
    assert result.spec.covariance == "robust"
    assert "sandwich" in result.inference["correction"]


def test_exogenous_trend_and_categorical(data):
    frame = make_data(seed=3, exog=True)
    result = oe.var(data=frame, y=NAMES, x=["x", "g"], categorical=["g"], lags=1, trend=True)
    z, y = design(frame, NAMES, 1, extra=[frame.x, (frame.g == "lo").astype(float)], trend=True)
    b, u, sigma, zzinv = ols_system(z, y)
    est, se = estimates(result)
    assert_allclose(est, b.ravel(), rtol=1e-8, atol=1e-11)
    assert_allclose(se, np.sqrt(np.diag(np.kron(sigma, zzinv))), rtol=1e-8)
    assert [c.term for c in result.coefficients[:7]] == [
        "a:L1.a", "a:L1.b", "a:L1.c", "a:x", "a:g[lo]", "a:trend", "a:Intercept"]
    assert result.extra["layout"]["exogenous"] == ["x", "g[lo]"]
    # statsmodels with the same regressors (its trend is counted from the estimation sample).
    reference = VAR(frame[NAMES], exog=np.column_stack([frame.x, frame.g == "lo"]).astype(float)
                    ).fit(1, trend="ct")
    assert_allclose(result.metrics["log_likelihood"], reference.llf, rtol=1e-10)
    # No constant: uncentered R-squared and no Intercept terms.
    bare = oe.var(data=frame, y=NAMES, lags=1, constant=False)
    z0, y0 = design(frame, NAMES, 1, constant=False)
    b0, u0, *_ = ols_system(z0, y0)
    assert_allclose(estimates(bare)[0], b0.ravel(), rtol=1e-9, atol=1e-12)
    assert not any("Intercept" in c.term for c in bare.coefficients)
    assert_allclose(bare.extra["equations"][0]["r_squared"],
                    1 - (u0[:, 0] ** 2).sum() / (y0[:, 0] ** 2).sum(), rtol=1e-10)


def test_granger_and_lag_exclusion_wald(data):
    result = oe.var(data=data, y=NAMES, lags=2, dfk=True)
    reference = VAR(data[NAMES]).fit(2, trend="c")
    est = estimates(result)[0]
    cov = np.array(result.covariance_matrix)
    m = 7
    for row in result.extra["granger"]:
        i = NAMES.index(row["equation"])
        causing = [n for n in NAMES if n != row["equation"]] if row["excluded"] == "ALL" \
            else [row["excluded"]]
        index = [i * m + NAMES.index(v) * 2 + j for v in causing for j in range(2)]
        wald = est[index] @ np.linalg.solve(cov[np.ix_(index, index)], est[index])
        assert_allclose(row["statistic"], wald, rtol=1e-8)
        assert row["df"] == len(index) and row["distribution"] == "chi2"
        assert_allclose(row["p_value"], stats.chi2.sf(wald, len(index)), rtol=1e-8)
        theirs = reference.test_causality(row["equation"], causing, kind="wald")
        assert_allclose(row["statistic"], theirs.test_statistic, rtol=1e-7)
    assert len(result.extra["granger"]) == 9
    assert result.tests["granger_all_a"]["statistic"] == pytest.approx(
        next(r for r in result.extra["granger"] if r["equation"] == "a"
             and r["excluded"] == "ALL")["statistic"])
    for row in result.extra["lag_exclusion"]:
        equations = range(3) if row["equation"] == "ALL" else [NAMES.index(row["equation"])]
        index = [i * m + v * 2 + row["lag"] - 1 for i in equations for v in range(3)]
        wald = est[index] @ np.linalg.solve(cov[np.ix_(index, index)], est[index])
        assert_allclose(row["statistic"], wald, rtol=1e-8)
        assert row["df"] == len(index)
    assert result.tests["lag_exclusion_L2"]["df"] == 9
    # small: F = chi2 / q with T - m denominator degrees of freedom.
    small = oe.var(data=data, y=NAMES, lags=2, small=True)
    row, chi = small.extra["granger"][0], result.extra["granger"][0]
    t = small.nobs
    assert row["distribution"] == "F" and row["df2"] == t - m
    assert_allclose(row["statistic"], chi["statistic"] / chi["df"], rtol=1e-8)
    assert_allclose(row["p_value"], stats.f.sf(row["statistic"], row["df"], t - m), rtol=1e-8)


def test_normality_and_lm_autocorrelation(data, fitted):
    z, y = design(data, NAMES, 2)
    b, u, sigma, zzinv = ols_system(z, y)
    t, m = z.shape
    w = np.linalg.solve(np.linalg.cholesky(sigma), u.T).T
    b1, b2 = (w ** 3).mean(0), (w ** 4).mean(0)
    rows = fitted.extra["normality"]
    assert [r["equation"] for r in rows] == [*NAMES, "ALL"]
    assert_allclose([r["skewness"] for r in rows[:3]], b1, rtol=1e-9)
    assert_allclose([r["kurtosis"] for r in rows[:3]], b2, rtol=1e-9)
    assert_allclose([r["jarque_bera"] for r in rows[:3]], t * b1 ** 2 / 6 + t * (b2 - 3) ** 2 / 24,
                    rtol=1e-9)
    joint = t * (b1 ** 2).sum() / 6 + t * ((b2 - 3) ** 2).sum() / 24
    assert_allclose(fitted.tests["normality"]["statistic"], joint, rtol=1e-9)
    assert fitted.tests["normality"]["df"] == 6
    assert_allclose(fitted.tests["normality"]["p_value"], stats.chi2.sf(joint, 6), rtol=1e-8)
    assert_allclose(rows[3]["skewness_p"], stats.chi2.sf(t * (b1 ** 2).sum() / 6, 3), rtol=1e-8)
    reference = VAR(data[NAMES]).fit(2, trend="c").test_normality()
    assert_allclose(joint, reference.test_statistic, rtol=1e-8)
    # LM autocorrelation: explicit augmented regressions.
    for s in (1, 2):
        lagged = np.zeros_like(u)
        lagged[s:] = u[:-s]
        _, _, sigma_s, _ = ols_system(np.column_stack([z, lagged]), y)
        lm = (t - (m + 3) - 0.5) * np.log(np.linalg.det(sigma) / np.linalg.det(sigma_s))
        test = fitted.tests[f"lm_autocorrelation_L{s}"]
        assert_allclose(test["statistic"], lm, rtol=1e-7)
        assert test["df"] == 9
        assert_allclose(test["p_value"], stats.chi2.sf(lm, 9), rtol=1e-7)
    more = oe.var(data=data, y=NAMES, lags=2, lm_lags=4)
    assert "lm_autocorrelation_L4" in more.tests
    none = oe.var(data=data, y=NAMES, lags=2, lm_lags=0)
    assert not any(name.startswith("lm_") for name in none.tests)


def test_stability(data, fitted):
    z, y = design(data, NAMES, 2)
    b = ols_system(z, y)[0]
    a1, a2 = b[:, [0, 2, 4]], b[:, [1, 3, 5]]
    companion = np.block([[a1, a2], [np.eye(3), np.zeros((3, 3))]])
    moduli = np.sort(np.abs(np.linalg.eigvals(companion)))[::-1]
    record = fitted.extra["stability"]
    assert_allclose([e["modulus"] for e in record["eigenvalues"]], moduli, rtol=1e-8)
    assert record["stable"] is True
    reference = VAR(data[NAMES]).fit(2, trend="c")
    assert_allclose(np.sort(1 / np.abs(reference.roots))[::-1], moduli, rtol=1e-7)
    # A random walk is flagged with a warning.
    rng = np.random.default_rng(0)
    walk = pd.DataFrame(rng.normal(size=(400, 2)).cumsum(axis=0) + np.arange(400)[:, None],
                        columns=["u", "v"])
    unstable = oe.var(data=walk, y=["u", "v"], lags=1)
    if not unstable.extra["stability"]["stable"]:
        assert any("stability" in w for w in unstable.warnings)
    assert unstable.extra["stability"]["eigenvalues"][0]["modulus"] > 0.97


def test_lag_order_selection_matches_numpy_and_statsmodels(data):
    table = oe.varsoc(data=data, y=NAMES, maxlag=4)
    n, k, maxlag = len(data), 3, 4
    y_all = data[NAMES].to_numpy()
    t = n - maxlag
    rows = []
    for j in range(maxlag + 1):
        cols = [y_all[maxlag - i:n - i] for i in range(1, j + 1)] + [np.ones((t, 1))]
        z = np.column_stack(cols)
        _, _, sigma, _ = ols_system(z, y_all[maxlag:])
        m = z.shape[1]
        ll = loglik(sigma, t)
        rows.append((ll, np.linalg.det(sigma) * ((t + m) / (t - m)) ** k,
                     (-2 * ll + 2 * k * m) / t, (-2 * ll + 2 * np.log(np.log(t)) * k * m) / t,
                     (-2 * ll + np.log(t) * k * m) / t))
    rows = np.array(rows)
    assert_allclose(table["ll"], rows[:, 0], rtol=1e-10)
    assert_allclose(table["fpe"], rows[:, 1], rtol=1e-9)
    assert_allclose(table["aic"], rows[:, 2], rtol=1e-10)
    assert_allclose(table["hqic"], rows[:, 3], rtol=1e-10)
    assert_allclose(table["sbic"], rows[:, 4], rtol=1e-10)
    lr = 2 * np.diff(rows[:, 0])
    assert_allclose(table["lr"].to_numpy()[1:], lr, rtol=1e-8)
    assert_allclose(table["p_value"].to_numpy()[1:], stats.chi2.sf(lr, 9), rtol=1e-7)
    assert np.isnan(table["lr"].to_numpy()[0]) or table["lr"].to_numpy()[0] is None
    selected = table.attrs["selected"]
    assert selected["aic"] == int(np.argmin(rows[:, 2]))
    assert selected["sbic"] == int(np.argmin(rows[:, 4]))
    assert selected["lr"] == max(j + 1 for j in range(maxlag) if stats.chi2.sf(lr[j], 9) < 0.05)
    assert table.attrs["n_obs"] == t
    reference = VAR(data[NAMES]).select_order(4, trend="c")
    assert selected["aic"] == reference.aic and selected["sbic"] == reference.bic
    assert selected["hqic"] == reference.hqic and selected["fpe"] == reference.fpe
    # The post-estimation table uses the model's own sample by default.
    result = oe.var(data=data, y=NAMES, lags=2)
    record = result.extra["lag_order_selection"]
    assert record["maxlag"] == 2 and record["n_obs"] == n - 2
    assert_allclose(record["rows"][2]["ll"], result.metrics["log_likelihood"], rtol=1e-11)
    assert_allclose(record["rows"][2]["aic"], result.metrics["aic_per_obs"], rtol=1e-11)
    wider = oe.var(data=data, y=NAMES, lags=2, maxlag=4)
    assert_allclose([r["ll"] for r in wider.extra["lag_order_selection"]["rows"]], rows[:, 0],
                    rtol=1e-10)
    # Exogenous regressors and no constant enter every lag order, including lag 0.
    bare = oe.varsoc(data=data, y=NAMES, maxlag=2, constant=False)
    assert_allclose(bare["ll"][0], loglik(y_all[2:].T @ y_all[2:] / (n - 2), n - 2), rtol=1e-10)
    assert bare["fpe"][0] == pytest.approx(np.linalg.det(y_all[2:].T @ y_all[2:] / (n - 2)))


def lutkepohl():
    root = os.path.dirname(statsmodels.__file__)
    paths = glob.glob(os.path.join(root, "tsa", "tests", "results", "lutkepohl2.dta"))
    if not paths:
        pytest.skip("statsmodels does not ship the Lutkepohl data")
    frame = pd.read_stata(paths[0])
    frame = frame[frame.qtr <= "1978-12-31"].iloc[1:].reset_index(drop=True)
    return frame[["dln_inv", "dln_inc", "dln_consump", "qtr"]]


def test_published_stata_output():
    """[TS] var, varbasic, varsoc and varlmar examples of the Stata manual (Lutkepohl data)."""
    frame = lutkepohl()
    names = ["dln_inv", "dln_inc", "dln_consump"]
    result = oe.var(data=frame, y=names, time="qtr", lags=2, dfk=True, lm_lags=2)
    assert result.nobs == 73
    assert result.metrics["log_likelihood"] == pytest.approx(606.307, abs=5e-4)
    assert result.metrics["fpe"] == pytest.approx(2.18e-11, rel=5e-3)
    assert result.metrics["det_sigma_ml"] == pytest.approx(1.23e-11, rel=5e-3)
    stata = {  # var dln_inv dln_inc dln_consump if qtr<=tq(1978q4), lutstats dfk
        "dln_inv:L1.dln_inv": (-.3196318, .1254564), "dln_inv:L2.dln_inv": (-.1605508, .1249066),
        "dln_inv:L1.dln_inc": (.1459851, .5456664), "dln_inv:L2.dln_inc": (.1146009, .5345709),
        "dln_inv:L1.dln_consump": (.9612288, .6643086),
        "dln_inv:L2.dln_consump": (.9344001, .6650949), "dln_inv:Intercept": (-.0167221, .0172264),
        "dln_inc:L1.dln_inv": (.0439309, .0318592), "dln_inc:Intercept": (.0157672, .0043746),
        "dln_consump:L2.dln_inc": (.3549135, .1094069),
        "dln_consump:Intercept": (.0129258, .0035256),
    }
    rows = {c.term: c for c in result.coefficients}
    for term, (estimate, error) in stata.items():
        assert rows[term].estimate == pytest.approx(estimate, abs=6e-8)
        assert rows[term].std_error == pytest.approx(error, abs=6e-8)
    header = [(.046148, 0.1286, 9.736909, 0.1362), (.011719, 0.1142, 8.508289, 0.2032),
              (.009445, 0.2513, 22.15096, 0.0011)]
    for record, (rmse, r2, chi2, p) in zip(result.extra["equations"], header, strict=True):
        assert record["parms"] == 7
        assert record["rmse"] == pytest.approx(rmse, abs=6e-7)
        assert record["r_squared"] == pytest.approx(r2, abs=6e-5)
        assert record["statistic"] == pytest.approx(chi2, rel=2e-6)
        assert record["p_value"] == pytest.approx(p, abs=6e-5)
    # varlmar
    assert result.tests["lm_autocorrelation_L1"]["statistic"] == pytest.approx(5.5871, abs=6e-5)
    assert result.tests["lm_autocorrelation_L1"]["p_value"] == pytest.approx(0.78043, abs=6e-6)
    assert result.tests["lm_autocorrelation_L2"]["statistic"] == pytest.approx(6.3189, abs=6e-5)
    assert result.tests["lm_autocorrelation_L2"]["p_value"] == pytest.approx(0.70763, abs=6e-6)
    # varbasic (no dfk): the same RMSE, larger chi2, the default information criteria.
    plain = oe.var(data=frame, y=names, time="qtr", lags=2)
    assert plain.metrics["aic_per_obs"] == pytest.approx(-16.03581, abs=6e-6)
    assert plain.metrics["hqic_per_obs"] == pytest.approx(-15.77323, abs=6e-6)
    assert plain.metrics["sbic_per_obs"] == pytest.approx(-15.37691, abs=6e-6)
    for record, chi2 in zip(plain.extra["equations"], (10.76961, 9.410683, 24.50031), strict=True):
        assert record["statistic"] == pytest.approx(chi2, rel=2e-6)
    assert plain.extra["equations"][0]["rmse"] == pytest.approx(.046148, abs=6e-7)
    # varsoc, maxlag(4): log likelihoods, LR statistics and the starred lags.
    table = oe.varsoc(data=frame, y=names, time="qtr", maxlag=4)
    assert_allclose(table["ll"], [564.784, 576.409, 588.859, 591.237, 598.457], atol=6e-4)
    assert_allclose(table["lr"].to_numpy()[1:], [23.249, 24.901, 4.7566, 14.438], atol=6e-4)
    assert_allclose(table["p_value"].to_numpy()[1:], [0.006, 0.003, 0.855, 0.108], atol=6e-4)
    assert_allclose(table["fpe"], [2.7e-11, 2.5e-11, 2.3e-11, 2.7e-11, 2.9e-11], rtol=3e-2)
    assert table.attrs["selected"] == {"fpe": 2, "aic": 2, "hqic": 0, "sbic": 0, "lr": 2}
    assert table.attrs["n_obs"] == 71


def test_collinear_exogenous_is_omitted_and_collinear_system_is_an_error(data):
    frame = data.assign(x2=2 * data.x, one=1.0)
    result = oe.var(data=frame, y=NAMES, x=["x", "x2", "one"], lags=1)
    assert result.provenance["omitted_terms"] == ["x2", "one"]
    assert any("Omitted because of collinearity" in w for w in result.warnings)
    reference = oe.var(data=frame, y=NAMES, x=["x"], lags=1)
    assert_allclose(estimates(result)[0], estimates(reference)[0], rtol=1e-9, atol=1e-12)
    assert [c.term for c in result.coefficients] == [c.term for c in reference.coefficients]
    twin = data.assign(d=data.a * 2.0)
    with pytest.raises(AnalysisError) as error:
        oe.var(data=twin, y=["a", "b", "d"], lags=1)
    assert error.value.code == "collinear_system"
    clash = data.assign(trend=np.arange(len(data), dtype=float) ** 2)
    with pytest.raises(AnalysisError) as error:
        oe.var(data=clash, y=NAMES, x=["trend"], lags=1, trend=True)
    assert error.value.code == "duplicate_terms"


def test_time_column_orders_the_sample(data, fitted):
    shuffled = data.sample(frac=1.0, random_state=4).reset_index(drop=True)
    result = oe.var(data=shuffled, y=NAMES, time="period", lags=2)
    assert_allclose(estimates(result)[0], estimates(fitted)[0], rtol=1e-10, atol=1e-13)
    assert result.provenance["sample_order"] == "sorted by period"
    assert result.extra["forecast"]["last_period"] == int(data.period.iloc[-1])
    dated = data.assign(date=pd.date_range("2001-01-01", periods=len(data), freq="MS"))
    by_date = oe.var(data=dated.sample(frac=1.0, random_state=1), y=NAMES, time="date", lags=2)
    assert_allclose(estimates(by_date)[0], estimates(fitted)[0], rtol=1e-10, atol=1e-13)
    assert any("consecutive periods" in w for w in by_date.warnings)
    with pytest.raises(AnalysisError) as error:
        oe.var(data=data.drop(index=50), y=NAMES, time="period", lags=2)
    assert error.value.code == "time_gaps"
    repeated = data.assign(period=np.r_[data.period.iloc[:-1], data.period.iloc[-2]])
    with pytest.raises(AnalysisError) as error:
        oe.var(data=repeated, y=NAMES, time="period", lags=2)
    assert error.value.code == "repeated_time_values"
    with pytest.raises(AnalysisError) as error:
        oe.var(data=data.assign(label="q"), y=NAMES, time="label", lags=2)
    assert error.value.code == "invalid_time"


def test_missing_data_policy(data, fitted):
    holes = data.copy()
    holes.loc[[0, 1, len(data) - 1], "b"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.var(data=holes, y=NAMES, lags=2)
    assert error.value.code == "missing_values"
    dropped = oe.var(data=holes, y=NAMES, lags=2, missing="drop")
    trimmed = oe.var(data=data.iloc[2:-1].reset_index(drop=True), y=NAMES, lags=2)
    assert_allclose(estimates(dropped)[0], estimates(trimmed)[0], rtol=1e-11)
    assert dropped.nobs == len(data) - 5 and dropped.sample_positions[0] == 4
    assert any("Excluded 3 observation(s)" in w for w in dropped.warnings)
    interior = data.copy()
    interior.loc[40, "a"] = np.nan
    for options in ({}, {"time": "period"}):
        with pytest.raises(AnalysisError) as error:
            oe.var(data=interior, y=NAMES, lags=2, missing="drop", **options)
        assert error.value.code == "time_gaps"


def test_error_codes(data):
    cases = [
        (dict(y="a"), "invalid_spec"),
        (dict(y=[]), "invalid_spec"),
        (dict(y=["a", "a"]), "invalid_spec"),
        (dict(y=NAMES, x=["a"]), "invalid_spec"),
        (dict(y=NAMES, x="x"), "invalid_spec"),
        (dict(y=NAMES, lags=0), "invalid_spec"),
        (dict(y=NAMES, lags=1.5), "invalid_spec"),
        (dict(y=NAMES, covariance="cluster"), "invalid_spec"),
        (dict(y=NAMES, covariance="hac"), "invalid_spec"),
        (dict(y=NAMES, irf_kinds=["structural"]), "invalid_option"),
        (dict(y=NAMES, irf_steps=0), "invalid_spec"),
        (dict(y=NAMES, missing="ignore"), "invalid_spec"),
        (dict(y=NAMES, alpha=1.5), "invalid_spec"),
        (dict(y=["a", "zz"]), "missing_columns"),
        (dict(y=NAMES, lags=100), "insufficient_observations"),
        (dict(y=NAMES, time="a"), "invalid_spec"),
    ]
    for options, code in cases:
        with pytest.raises(AnalysisError) as error:
            oe.var(data=data, **options)
        assert error.value.code == code, options
    with pytest.raises(AnalysisError) as error:
        oe.var(data=data.iloc[:8], y=NAMES, lags=2)
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.var(data=data.assign(a=1.0), y=NAMES, lags=1)
    assert error.value.code in {"collinear_system", "singular_residual_covariance"}
    wide = pd.DataFrame(np.random.default_rng(0).normal(size=(400, 10)),
                        columns=[f"w{i}" for i in range(10)])
    with pytest.raises(AnalysisError) as error:
        oe.var(data=wide, y=list(wide.columns), lags=10)
    assert error.value.code == "model_too_large"
    with pytest.raises(AnalysisError) as error:
        oe.varsoc(data=data, y=NAMES, maxlag=-1)
    assert error.value.code == "invalid_lags"
    with pytest.raises(AnalysisError) as error:
        oe.varsoc(data=data.iloc[:20], y=NAMES, maxlag=10)
    assert error.value.code == "insufficient_observations"
    # A hand-built spec keeps pydantic's error; the outcome must lead the system.
    with pytest.raises(ValidationError):
        ModelSpec(estimator="var", outcome="a", options={"lags": 2})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="var", outcome="a", columns={"system": NAMES}, weights="x",
                  weight_type="aweight")
    spec = ModelSpec(estimator="var", outcome="b", columns={"system": NAMES})
    with pytest.raises(AnalysisError) as error:
        oe.fit(spec, data=data)
    assert error.value.code == "invalid_spec"


def test_spec_round_trip_summary_latex_and_registry(data, fitted):
    again = type(fitted).model_validate_json(fitted.model_dump_json())
    assert again == fitted and isinstance(again, ResultBundle)
    json.loads(fitted.model_dump_json())
    spec = ModelSpec(estimator="var", outcome="a", columns={"system": NAMES},
                     options={"lags": 2})
    direct = oe.fit(spec, data=data)
    assert_allclose(estimates(direct)[0], estimates(fitted)[0], rtol=1e-12)
    assert direct.spec.covariance == "nonrobust"
    text = fitted.summary()
    assert "Vector autoregression" in text and "[b]" in text and "b:L2.c" in text
    assert "log_likelihood" in text and "Granger causality" in text and "LM test" in text
    latex = str(fitted.to_latex())
    assert ("tabular" in latex or "longtable" in latex) and "L1.a" in latex
    assert fitted.provenance["stata_parity_validated"] is False
    assert fitted.provenance["family"] == "var"
    assert "varsoc" in fitted.provenance["stata_equivalent"]
    capability = oe.capabilities()["estimators"]["var"]
    assert capability["covariances"] == ["nonrobust", "robust"]
    assert capability["columns"]["system"]["required"] is True
    for name in ("var", "varsoc", "irf", "var_forecast", "vec", "vecrank", "vec_forecast"):
        assert callable(getattr(oe, name))
    # Stored arrays are bounded.
    assert len(fitted.extra["irf"]["simple"]) == 9
    big = oe.var(data=data, y=NAMES, lags=1, irf_steps=3, irf_kinds=["simple"])
    assert set(big.extra["irf"]) >= {"simple", "se"} and "orthogonalized" not in big.extra["irf"]


def test_series_with_a_large_level_keep_full_accuracy(data, fitted):
    """Estimation on mean-deviated data: adding 1e9 to every series changes only the constants."""
    shifted = data.copy()
    shifted[NAMES] = data[NAMES] + 1e9
    for options in ({}, {"covariance": "robust"}):
        base = oe.var(data=data, y=NAMES, lags=2, **options)
        result = oe.var(data=shifted, y=NAMES, lags=2, **options)
        slope = np.array(["Intercept" not in c.term for c in base.coefficients])
        assert_allclose(estimates(result)[0][slope], estimates(base)[0][slope], rtol=1e-6,
                        atol=1e-8)
        assert_allclose(estimates(result)[1][slope], estimates(base)[1][slope], rtol=1e-6)
        assert result.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"],
                                                                 abs=1e-5)
        for name in ("lm_autocorrelation_L1", "lm_autocorrelation_L2", "normality",
                     "granger_all_c"):
            assert result.tests[name]["statistic"] == pytest.approx(
                base.tests[name]["statistic"], rel=1e-5)
        assert_allclose([r["ll"] for r in result.extra["lag_order_selection"]["rows"]],
                        [r["ll"] for r in base.extra["lag_order_selection"]["rows"]], atol=1e-5)
    # The constant and its standard error follow from the exact back-mapping.
    z, y = design(shifted, NAMES, 2)
    centered = z.copy()
    centered[:, :-1] -= centered[:, :-1].mean(0)
    slopes = np.linalg.lstsq(centered, y - y.mean(0), rcond=None)[0][:-1]
    intercept = y.mean(0) - z[:, :-1].mean(0) @ slopes
    result = oe.var(data=shifted, y=NAMES, lags=2)
    rows = {c.term: c for c in result.coefficients}
    assert_allclose([rows[f"{n}:Intercept"].estimate for n in NAMES], intercept, rtol=1e-9)
    forecast, reference = oe.forecast(result, 3), oe.forecast(fitted, 3)
    assert_allclose(forecast.forecast - 1e9, reference.forecast, atol=1e-5)
    assert_allclose(forecast.std_error, reference.std_error, rtol=1e-5)


def test_univariate_var_is_an_autoregression(data):
    result = oe.var(data=data, y=["a"], lags=3)
    z, y = design(data, ["a"], 3)
    ols = sm.OLS(y[:, 0], z).fit()
    assert_allclose(estimates(result)[0], ols.params, rtol=1e-9)
    assert_allclose(estimates(result)[1], ols.bse * np.sqrt(ols.df_resid / ols.nobs), rtol=1e-9)
    assert result.extra["granger"] == [] and result.metrics["n_equations"] == 1


def test_ols_is_the_gaussian_maximum_likelihood_estimator():
    """Brute-force maximization of an independently written likelihood (K = 2, p = 1)."""
    frame = make_data(seed=11, n=120)[["a", "b"]]
    result = oe.var(data=frame, y=["a", "b"], lags=1)
    y = frame.to_numpy()
    y1, y0 = y[1:], y[:-1]
    t = len(y1)

    def negative(theta):
        a, c = theta[:4].reshape(2, 2), theta[4:6]
        lower = np.array([[np.exp(theta[6]), 0.0], [theta[7], np.exp(theta[8])]])
        u = y1 - y0 @ a.T - c
        w = np.linalg.solve(lower, u.T)
        return 0.5 * (w ** 2).sum() + t * (theta[6] + theta[8]) + t * np.log(2 * np.pi)

    best = minimize(negative, np.zeros(9), method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
    assert_allclose(-best.fun, result.metrics["log_likelihood"], rtol=1e-9)
    rows = {c.term: c.estimate for c in result.coefficients}
    a = best.x[:4].reshape(2, 2)
    assert_allclose([rows["a:L1.a"], rows["a:L1.b"], rows["b:L1.a"], rows["b:L1.b"]], a.ravel(),
                    atol=2e-5)
    assert_allclose([rows["a:Intercept"], rows["b:Intercept"]], best.x[4:6], atol=2e-5)
