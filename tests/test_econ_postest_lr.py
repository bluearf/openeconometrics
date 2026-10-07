"""Independent oracles for oe.lrtest and oe.estat_ic (post-estimation family).

The likelihood-ratio statistic is checked against 2 (ll_1 - ll_0) computed
from statsmodels fits of the same nested Poisson and logit models (and from an
explicit tobit likelihood maximized by SciPy), with SciPy chi2 tails; the
information criteria against statsmodels' AIC/BIC and null log likelihoods.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize

import openecon as oe
from openecon.analysis import AnalysisError


def make_data(seed=7, n=400):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    frame["count"] = rng.poisson(np.exp(0.3 + 0.4 * x1 - 0.2 * x2))
    frame["nb"] = rng.negative_binomial(2, 1 / (1 + np.exp(0.3 + 0.4 * x1) / 2))
    frame["binary"] = (0.2 + x1 - 0.5 * x2 + rng.logistic(size=n) > 0).astype(int)
    latent = 0.5 + x1 + 0.3 * x2 + rng.normal(size=n)
    frame["censored"] = np.maximum(latent, 0.0)
    frame["group"] = rng.integers(0, 25, size=n)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def sm_poisson(frame, columns, y="count"):
    return sm.Poisson(frame[y], sm.add_constant(frame[columns])).fit(disp=0)


def sm_logit(frame, columns, y="binary"):
    return sm.Logit(frame[y], sm.add_constant(frame[columns])).fit(disp=0)


def test_lrtest_poisson_matches_statsmodels(data):
    full = oe.poisson(data=data, y="count", x=["x1", "x2", "x3"])
    small = oe.poisson(data=data, y="count", x=["x1"])
    result = oe.lrtest(full, small)
    lr = 2 * (sm_poisson(data, ["x1", "x2", "x3"]).llf - sm_poisson(data, ["x1"]).llf)
    row = result.iloc[0]
    assert_allclose(row["statistic"], lr, rtol=1e-7)
    assert row["df"] == 2
    assert_allclose(row["p_value"], stats.chi2.sf(lr, 2), rtol=1e-7)
    assert_allclose(row["ll_full"], full.metrics["log_likelihood"])
    assert_allclose(row["ll_restricted"], small.metrics["log_likelihood"])
    assert result.attrs["distribution"] == "chi2"
    assert result.attrs["k_full"] == 4 and result.attrs["k_restricted"] == 2
    assert "nested" in result.attrs["notes"][0]
    assert result.attrs["forced"] is False


def test_lrtest_logit_core_estimator(data):
    full = oe.logit(data=data, y="binary", x=["x1", "x2"])
    small = oe.logit(data=data, y="binary", x=["x1"])
    result = oe.lrtest(full, small)
    lr = 2 * (sm_logit(data, ["x1", "x2"]).llf - sm_logit(data, ["x1"]).llf)
    assert_allclose(result.iloc[0]["statistic"], lr, rtol=1e-6)
    assert_allclose(result.iloc[0]["p_value"], stats.chi2.sf(lr, 1), rtol=1e-6)


def tobit_loglik(frame, columns):
    y = frame["censored"].to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy()])
    left = y <= 0

    def negative(theta):
        beta, sigma = theta[:-1], np.exp(theta[-1])
        xb = x @ beta
        ll = np.where(left, stats.norm.logcdf(-xb / sigma),
                      stats.norm.logpdf((y - xb) / sigma) - np.log(sigma))
        return -ll.sum()

    start = np.r_[np.linalg.lstsq(x, y, rcond=None)[0], 0.0]
    fitted = minimize(negative, start, method="BFGS", options={"gtol": 1e-9})
    return -fitted.fun


def test_lrtest_tobit_against_brute_force_likelihood(data):
    full = oe.tobit(data=data, y="censored", x=["x1", "x2", "x3"], ll=0.0)
    small = oe.tobit(data=data, y="censored", x=["x1"], ll=0.0)
    result = oe.lrtest(full, small)
    lr = 2 * (tobit_loglik(data, ["x1", "x2", "x3"]) - tobit_loglik(data, ["x1"]))
    assert_allclose(result.iloc[0]["statistic"], lr, rtol=1e-5)
    assert result.iloc[0]["df"] == 2
    # The tobit sigma is not an extra parameter of the full model: no boundary note.
    assert result.attrs["boundary_terms"] == []


def test_lrtest_ols_gaussian_likelihood(data):
    full = oe.ols(data=data, y="censored", x=["x1", "x2"])
    small = oe.ols(data=data, y="censored", x=["x1"])
    result = oe.lrtest(full, small)
    llf = sm.OLS(data["censored"], sm.add_constant(data[["x1", "x2"]])).fit().llf
    llr = sm.OLS(data["censored"], sm.add_constant(data[["x1"]])).fit().llf
    assert_allclose(result.iloc[0]["statistic"], 2 * (llf - llr), rtol=1e-8)


def test_lrtest_user_df_and_note(data):
    full = oe.poisson(data=data, y="count", x=["x1", "x2", "x3"])
    small = oe.poisson(data=data, y="count", x=["x1"])
    result = oe.lrtest(full, small, df=3)
    assert result.iloc[0]["df"] == 3
    assert result.attrs["df_source"] == "given"
    assert any("parameter counts differ by 2" in note for note in result.attrs["notes"])
    assert_allclose(result.iloc[0]["p_value"], stats.chi2.sf(result.iloc[0]["statistic"], 3))
    with pytest.raises(AnalysisError) as error:
        oe.lrtest(full, small, df=0)
    assert error.value.code == "invalid_spec"


def test_lrtest_boundary_nbreg_against_poisson(data):
    nb = oe.nbreg(data=data, y="nb", x=["x1"])
    pois = oe.poisson(data=data, y="nb", x=["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.lrtest(nb, pois)
    assert error.value.code == "incompatible_models"
    assert "different estimators" in str(error.value)
    result = oe.lrtest(nb, pois, force=True)
    lr = 2 * (nb.metrics["log_likelihood"] - pois.metrics["log_likelihood"])
    assert_allclose(result.iloc[0]["statistic"], lr)
    assert result.attrs["boundary_terms"] == ["/lnalpha"]
    assert_allclose(result.attrs["p_value_chibar2"], 0.5 * stats.chi2.sf(lr, 1))
    assert result.attrs["forced"] is True


@pytest.mark.parametrize("case", ["samples", "outcome", "robust", "order", "missing_ll",
                                  "larger_restricted", "not_result", "pweight"])
def test_lrtest_refusals(data, case):
    full = oe.poisson(data=data, y="count", x=["x1", "x2"])
    small = oe.poisson(data=data, y="count", x=["x1"])
    if case == "samples":
        holes = data.copy()
        holes.loc[:9, "x2"] = np.nan
        args = (oe.poisson(data=holes, y="count", x=["x1", "x2"], missing="drop"),
                oe.poisson(data=holes, y="count", x=["x1"]))
    elif case == "outcome":
        args = (full, oe.poisson(data=data, y="nb", x=["x1"]))
    elif case == "robust":
        args = (oe.poisson(data=data, y="count", x=["x1", "x2"], covariance="robust"), small)
    elif case == "order":
        args = (small, full)
    elif case == "missing_ll":
        args = (oe.qreg(data=data, y="censored", x=["x1", "x2"]),
                oe.qreg(data=data, y="censored", x=["x1"]))
    elif case == "larger_restricted":
        args = (small, full)
    elif case == "pweight":
        weighted = data.assign(w=1.0 + (data["x3"] > 0))
        args = (oe.poisson(data=weighted, y="count", x=["x1", "x2"], weights="w",
                           weight_type="pweight"),
                oe.poisson(data=weighted, y="count", x=["x1"], weights="w",
                           weight_type="pweight"))
    else:
        args = ({"not": "a result"}, small)
    kwargs = {"df": 1} if case == "larger_restricted" else {}
    with pytest.raises(AnalysisError) as error:
        oe.lrtest(*args, **kwargs)
    expected = "invalid_result" if case == "not_result" else "incompatible_models"
    assert error.value.code == expected
    if case == "robust":
        forced = oe.lrtest(*args, force=True)
        assert forced.attrs["forced"] is True
        assert any("pseudo" in note or "forced" in note for note in forced.attrs["notes"])
    if case == "larger_restricted":
        assert "larger log likelihood" in str(error.value)


def test_lrtest_samples_cannot_be_forced(data):
    holes = data.copy()
    holes.loc[:9, "x2"] = np.nan
    full = oe.poisson(data=holes, y="count", x=["x1", "x2"], missing="drop")
    small = oe.poisson(data=holes, y="count", x=["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.lrtest(full, small, force=True)
    assert error.value.code == "incompatible_models"
    assert "estimation samples" in str(error.value)


def test_estat_ic_matches_statsmodels(data):
    pois = oe.poisson(data=data, y="count", x=["x1", "x2"])
    logit = oe.logit(data=data, y="binary", x=["x1", "x2"])
    nb = oe.nbreg(data=data, y="count", x=["x1", "x2"])
    table = oe.estat_ic(pois, logit, nb, names=["poisson", "logit", "nbreg"])
    assert list(table.columns) == ["model", "nobs", "ll_null", "ll", "df", "aic", "bic"]
    assert list(table["model"]) == ["poisson", "logit", "nbreg"]
    sp, sl = sm_poisson(data, ["x1", "x2"]), sm_logit(data, ["x1", "x2"])
    assert_allclose(table["ll"].iloc[:2], [sp.llf, sl.llf], rtol=1e-7)
    assert_allclose(table["aic"].iloc[:2], [sp.aic, sl.aic], rtol=1e-7)
    assert_allclose(table["bic"].iloc[:2], [sp.bic, sl.bic], rtol=1e-7)
    assert_allclose(table["ll_null"].iloc[:2], [sp.llnull, sl.llnull], rtol=1e-6)
    assert list(table["df"]) == [3, 3, 4]
    n = len(data)
    ll = nb.metrics["log_likelihood"]
    assert_allclose(table["aic"].iloc[2], -2 * ll + 8)
    assert_allclose(table["bic"].iloc[2], -2 * ll + 4 * np.log(n))
    # Each result's own AIC/BIC metrics follow the same convention.
    assert_allclose(table["aic"].iloc[0], pois.metrics["aic"])
    assert_allclose(table["bic"].iloc[1], logit.metrics["bic"])
    assert any("different outcomes" in note for note in table.attrs["notes"])


def test_estat_ic_default_names_and_sample_note(data):
    holes = data.copy()
    holes.loc[:4, "x2"] = np.nan
    a = oe.poisson(data=holes, y="count", x=["x1", "x2"], missing="drop")
    b = oe.poisson(data=holes, y="count", x=["x1"])
    table = oe.estat_ic(a, b)
    assert list(table["model"]) == ["1: poisson", "2: poisson"]
    assert list(table["nobs"]) == [len(data) - 5, len(data)]
    assert any("different estimation samples" in note for note in table.attrs["notes"])


def test_estat_ic_refusals(data):
    q = oe.qreg(data=data, y="censored", x=["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.estat_ic(q)
    assert error.value.code == "missing_log_likelihood"
    pois = oe.poisson(data=data, y="count", x=["x1"])
    for kwargs in ({"names": ["a", "b"]}, {"names": "a"}):
        with pytest.raises(AnalysisError) as error:
            oe.estat_ic(pois, **kwargs)
        assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.estat_ic()
    assert error.value.code == "invalid_spec"


def test_tables_render_and_export(data):
    full = oe.poisson(data=data, y="count", x=["x1", "x2"])
    small = oe.poisson(data=data, y="count", x=["x1"])
    lr = oe.lrtest(full, small)
    ic = oe.estat_ic(full, small)
    assert "statistic" in lr.to_string()
    assert "tabular" in str(lr.to_latex())
    assert "aic" in str(ic.to_latex())
    assert callable(oe.lrtest) and callable(oe.estat_ic)
    assert "lrtest" in dir(oe) and "estat_ic" in dir(oe)
