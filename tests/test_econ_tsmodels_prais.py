"""Independent oracles for oe.prais (Prais-Winsten / Cochrane-Orcutt).

The iteration is re-implemented in NumPy from Stata's Methods and formulas
(every rhotype), the transformed-regression covariances are checked against
statsmodels OLS on the transformed data, and the Cochrane-Orcutt regression at
a fixed rho against statsmodels GLSAR.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def make_data(seed=3, n=150, rho=0.6):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n).cumsum() * 0.2 + rng.normal(size=n)
    x2 = rng.normal(size=n)
    u = np.zeros(n)
    e = rng.normal(size=n) * (1 + 0.5 * (x2 > 0))
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + u
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "t": np.arange(1, n + 1) + 1950})


@pytest.fixture(scope="module")
def df():
    return make_data()


def np_rho(u, rhotype, k):
    n = len(u)
    cross = u[1:] @ u[:-1]
    dw = np.sum(np.diff(u) ** 2) / (u @ u)
    if rhotype == "regress":
        return cross / (u[:-1] @ u[:-1])
    if rhotype == "freg":
        return cross / (u[1:] @ u[1:])
    if rhotype == "tscorr":
        return cross / (u @ u)
    if rhotype == "theil":
        return cross / (u @ u) * (n - k) / n
    if rhotype == "dw":
        return 1 - dw / 2
    return ((1 - dw / 2) * n * n + k * k) / (n * n - k * k)


def np_transform(a, rho, prais):
    rest = a[1:] - rho * a[:-1]
    return np.concatenate([np.sqrt(1 - rho * rho) * a[:1], rest]) if prais else rest


def np_prais(y, X, method="prais", rhotype="regress", twostep=False, tol=1e-6, maxit=100):
    prais = method == "prais"
    k = X.shape[1]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rho = np_rho(y - X @ beta, rhotype, k)
    iterations = 1
    while True:
        ys, xs = np_transform(y, rho, prais), np_transform(X, rho, prais)
        beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
        if twostep:
            break
        new = np_rho(y - X @ beta, rhotype, k)
        iterations += 1
        change, rho = abs(new - rho), new
        if change < tol:
            ys, xs = np_transform(y, rho, prais), np_transform(X, rho, prais)
            beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
            break
        assert iterations < maxit
    e = ys - xs @ beta
    n = len(ys)
    s2 = e @ e / (n - k)
    cov = s2 * np.linalg.inv(xs.T @ xs)
    r2 = 1 - (e @ e) / np.sum((ys - ys.mean()) ** 2)
    dw_t = np.sum(np.diff(e) ** 2) / (e @ e)
    return dict(beta=beta, cov=cov, rho=rho, r2=r2, dw=dw_t, xs=xs, ys=ys, iterations=iterations)


@pytest.mark.parametrize("method", ["prais", "corc"])
@pytest.mark.parametrize("rhotype", ["regress", "freg", "tscorr", "dw", "theil", "nagar"])
def test_iterated_matches_stata_formulas(df, method, rhotype):
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    ref = np_prais(df.y.to_numpy(), X, method, rhotype)
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], time="t", method=method, rhotype=rhotype)
    assert_allclose([c.estimate for c in fit.coefficients], ref["beta"], rtol=1e-9)
    assert_allclose([c.std_error for c in fit.coefficients], np.sqrt(np.diag(ref["cov"])), rtol=1e-8)
    assert_allclose(fit.metrics["rho"], ref["rho"], rtol=1e-10)
    assert_allclose(fit.metrics["r_squared"], ref["r2"], rtol=1e-9)
    assert_allclose(fit.metrics["durbin_watson_transformed"], ref["dw"], rtol=1e-9)
    assert fit.metrics["iterations"] == ref["iterations"]
    assert fit.nobs == (len(df) if method == "prais" else len(df) - 1)
    assert fit.inference["df_inference"] == fit.nobs - 3
    assert fit.extra["rho_path"][0] == 0.0 and fit.extra["converged"]


def test_twostep_and_original_dw(df):
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    ref = np_prais(df.y.to_numpy(), X, "prais", "regress", twostep=True)
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], twostep=True)
    assert_allclose([c.estimate for c in fit.coefficients], ref["beta"], rtol=1e-10)
    assert fit.metrics["iterations"] == 1
    ols = sm.OLS(df.y, sm.add_constant(df[["x1", "x2"]])).fit()
    assert_allclose(fit.metrics["durbin_watson_original"],
                    sm.stats.durbin_watson(ols.resid), rtol=1e-10)
    assert "two-step" in fit.title


@pytest.mark.parametrize("cov,sm_cov", [("nonrobust", "nonrobust"), ("robust", "HC1"),
                                        ("HC1", "HC1"), ("HC2", "HC2"), ("HC3", "HC3")])
@pytest.mark.parametrize("method", ["prais", "corc"])
def test_covariances_on_transformed_data(df, cov, sm_cov, method):
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], method=method, covariance=cov)
    rho = fit.metrics["rho"]
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    prais = method == "prais"
    ys, xs = np_transform(df.y.to_numpy(), rho, prais), np_transform(X, rho, prais)
    ref = sm.OLS(ys, xs).fit(cov_type=sm_cov)
    assert_allclose([c.estimate for c in fit.coefficients], ref.params, rtol=1e-10)
    assert_allclose([c.std_error for c in fit.coefficients], ref.bse, rtol=1e-8)
    assert_allclose([c.p_value for c in fit.coefficients], 2 * (1 - __import__("scipy").stats.t.cdf(
        np.abs(ref.params / ref.bse), len(ys) - 3)), rtol=1e-6, atol=1e-14)
    model = fit.tests["model"]
    assert model["df"] == 2 and model["df2"] == len(ys) - 3


def test_corc_matches_glsar_at_fixed_rho(df):
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], method="corc", rhotype="tscorr")
    rho = fit.metrics["rho"]
    X = sm.add_constant(df[["x1", "x2"]].to_numpy())
    glsar = sm.GLSAR(df.y.to_numpy(), X, rho=np.array([rho])).fit()
    assert_allclose([c.estimate for c in fit.coefficients], glsar.params, rtol=1e-10)
    assert_allclose([c.std_error for c in fit.coefficients], glsar.bse, rtol=1e-8)


def test_no_intercept_uncentered_r2(df):
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], intercept=False)
    rho = fit.metrics["rho"]
    X = df[["x1", "x2"]].to_numpy()
    ys, xs = np_transform(df.y.to_numpy(), rho, True), np_transform(X, rho, True)
    ref = sm.OLS(ys, xs).fit()
    assert_allclose(fit.metrics["r_squared"], ref.rsquared, rtol=1e-9)   # uncentered without a constant
    assert "model" in fit.tests and fit.tests["model"]["df"] == 2


def test_time_order_collinearity_and_categorical(df):
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    a = oe.prais(data=df, y="y", x=["x1", "x2"], time="t")
    b = oe.prais(data=shuffled, y="y", x=["x1", "x2"], time="t")
    assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients],
                    rtol=1e-12)
    dup = df.assign(x3=2 * df.x1, g=np.where(df.x2 > 0, "hi", "lo"))
    fit = oe.prais(data=dup, y="y", x=["x1", "x3", "g"], categorical=["g"], time="t")
    assert [c.term for c in fit.coefficients] == ["Intercept", "x1", "g[lo]"]
    assert fit.provenance["omitted_terms"] == ["x3"]


def test_missing_policy(df):
    holed = df.copy()
    holed.loc[0, "x1"] = np.nan
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=holed, y="y", x=["x1", "x2"], time="t")
    assert err.value.code == "missing_values"
    fit = oe.prais(data=holed, y="y", x=["x1", "x2"], time="t", missing="drop")
    assert fit.nobs == len(df) - 1
    ref = oe.prais(data=df.iloc[1:], y="y", x=["x1", "x2"], time="t")
    assert_allclose([c.estimate for c in fit.coefficients], [c.estimate for c in ref.coefficients])
    holed.loc[70, "x2"] = np.nan
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=holed, y="y", x=["x1", "x2"], time="t", missing="drop")
    assert err.value.code == "time_gaps"


def test_errors(df):
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=df, y="y", x=["x1"], max_iterations=1)
    assert err.value.code == "nonconvergence"
    n = 60
    explosive = pd.DataFrame({"y": 1.08 ** np.arange(n), "x": np.random.default_rng(1).normal(size=n)})
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=explosive, y="y", x=["x"])
    assert err.value.code == "rho_out_of_range"
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=df, y="y", x=["x1"], rhotype="bogus")
    assert err.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=df, y="y", x="x1")
    assert err.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=df.assign(t=df.t * 2), y="y", x=["x1"], time="t")
    assert err.value.code == "time_gaps"
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=df.iloc[:3], y="y", x=["x1", "x2"])
    assert err.value.code == "insufficient_observations"


def test_round_trip_and_rendering(df):
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], time="t", covariance="robust")
    assert type(fit).model_validate_json(fit.model_dump_json()) == fit
    text = fit.summary()
    assert "Prais-Winsten" in text and "rho" in text
    assert "x1" in fit.to_latex()
    assert isinstance(fit, ResultBundle) and fit.spec.estimator == "prais"
    assert fit.provenance["stata_parity_validated"] is False
