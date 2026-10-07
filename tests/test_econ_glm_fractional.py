"""cloglog, fracreg and betareg against independent oracles.

Oracles: statsmodels GLM (binomial family) and BetaModel, brute-force
maximization with scipy.optimize of independently written (quasi-)likelihoods
built on scipy.stats / scipy.special, and numerical scores and Hessians of
those likelihoods for the covariance estimators.
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import optimize, special, stats
from statsmodels.othermod.betareg import BetaModel

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.glm.families import make_link
from openecon.econometrics.glm.kernels import BetaObjective, ScaleLink
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec, ResultBundle

L = sm.families.links
X = ["x1", "x2"]
INVERSE = {
    "logit": special.expit, "probit": stats.norm.cdf,
    "cloglog": lambda eta: -np.expm1(-np.exp(eta)), "loglog": lambda eta: np.exp(-np.exp(-eta)),
}
SCALE_INVERSE = {"log": np.exp, "identity": lambda eta: eta, "sqrt": np.square}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(555)
    n = 500
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n), "z1": rng.normal(size=n),
        "firm": np.repeat(np.arange(50), 10), "state": rng.choice(list("abcdefgh"), size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "off": rng.normal(scale=0.3, size=n),
    })
    eta = -0.3 + 0.6 * frame.x1 - 0.4 * frame.x2
    frame["event"] = rng.binomial(1, -np.expm1(-np.exp(eta + frame.off))).astype(float)
    mean = special.expit(0.2 + 0.5 * frame.x1 - 0.3 * frame.x2)
    frame["share"] = np.clip(mean + rng.normal(scale=0.25, size=n), 0.0, 1.0)      # zeros and ones occur
    precision = np.exp(1.8 + 0.4 * frame.z1)
    frame["rate"] = rng.beta(mean * precision, (1 - mean) * precision).clip(1e-6, 1 - 1e-6)
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def design(frame, columns=X):
    return sm.add_constant(frame[list(columns)])


def numeric_scores(per_observation, theta, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = h * max(1.0, abs(theta[j]))
        columns.append((per_observation(theta + step) - per_observation(theta - step)) / (2 * step[j]))
    return np.column_stack(columns)


def numeric_hessian(total, theta, h=1e-4):
    theta = np.asarray(theta, dtype=float)
    k = len(theta)
    steps = np.diag([h * max(1.0, abs(value)) for value in theta])
    hessian = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            value = (total(theta + steps[i] + steps[j]) - total(theta + steps[i] - steps[j])
                     - total(theta - steps[i] + steps[j]) + total(theta - steps[i] - steps[j]))
            hessian[i, j] = hessian[j, i] = value / (4 * steps[i, i] * steps[j, j])
    return hessian


def cluster_meat(scores, labels):
    sums = pd.DataFrame(scores).groupby(np.asarray(labels), sort=False).sum().to_numpy()
    return sums.T @ sums


def bernoulli_quasi_likelihood(theta, y, x, link, offset=0.0):
    """Independent Bernoulli (quasi-)log-likelihood per observation."""
    eta = x @ theta + offset
    if link == "cloglog":
        log_p, log_q = np.log(-np.expm1(-np.exp(eta))), -np.exp(eta)
    elif link == "probit":
        log_p, log_q = stats.norm.logcdf(eta), stats.norm.logcdf(-eta)
    else:
        log_p, log_q = -np.logaddexp(0, -eta), -np.logaddexp(0, eta)
    return y * log_p + (1 - y) * log_q


# ---- cloglog -------------------------------------------------------------------------------


def test_cloglog_matches_statsmodels_brute_force_and_stata_reporting(data):
    result = oe.cloglog(data=data, y="event", x=X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.GLM(data.event, design(data), family=sm.families.Binomial(L.CLogLog())).fit(
            tol=1e-13, tol_criterion="params", maxiter=300)
    n, k = 500, 3
    assert_allclose(estimates(result), ref.params.values, rtol=1e-8)
    # Stata's cloglog reports the OBSERVED information (statsmodels' default is the expected one).
    observed = np.linalg.inv(-ref.model.hessian(ref.params.values, scale=1, observed=True))
    assert_allclose(np.asarray(result.covariance_matrix), observed, rtol=1e-7)
    assert not np.allclose(errors(result), ref.bse.values, rtol=1e-4)
    assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-11)
    y, x = data.event.to_numpy(), design(data).to_numpy()
    brute = optimize.minimize(lambda t: -bernoulli_quasi_likelihood(t, y, x, "cloglog").sum(), np.zeros(3),
                              method="BFGS", options={"gtol": 1e-9})
    assert_allclose(estimates(result), brute.x, atol=1e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(-brute.fun, abs=1e-8)
    share = y.mean()
    null = n * (share * np.log(share) + (1 - share) * np.log(1 - share))
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-11)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - ref.llf / null, rel=1e-9)
    model = result.tests["model"]
    assert model["statistic"] == pytest.approx(2 * (ref.llf - null), rel=1e-9) and model["df"] == 2
    assert model["p_value"] == pytest.approx(stats.chi2.sf(2 * (ref.llf - null), 2), rel=1e-7)
    assert result.metrics["aic"] == pytest.approx(-2 * ref.llf + 2 * k)
    assert result.metrics["bic"] == pytest.approx(-2 * ref.llf + k * np.log(n))
    assert list(result.metrics) == ["log_likelihood", "pseudo_r_squared", "aic", "bic", "df_resid"]
    assert result.extra["zero_outcomes"] == (y == 0).sum() and result.extra["nonzero_outcomes"] == y.sum()
    assert result.title == "Complementary log-log regression" and result.inference["use_t"] is False
    through_glm = oe.glm(data=data, y="event", x=X, family="binomial", link="cloglog")
    assert_allclose(estimates(result), estimates(through_glm), rtol=1e-12)
    assert_allclose(errors(result), errors(through_glm), rtol=1e-12)


def test_cloglog_covariances_offset_and_weights(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.GLM(data.event, design(data), family=sm.families.Binomial(L.CLogLog()), offset=data.off).fit(
            tol=1e-13, tol_criterion="params", maxiter=300)
    params = ref.params.values
    scores = ref.model.score_obs(params, scale=1)
    bread = np.linalg.inv(-ref.model.hessian(params, scale=1, observed=True))
    n = len(data)
    common = {"data": data, "y": "event", "x": X, "offset": "off"}
    plain = oe.cloglog(**common)
    assert_allclose(estimates(plain), params, rtol=1e-8)
    assert_allclose(np.asarray(plain.covariance_matrix), bread, rtol=1e-7)
    # The null model keeps the offset: a one-parameter brute-force fit.
    y, off = data.event.to_numpy(), data.off.to_numpy()
    null = optimize.minimize_scalar(
        lambda c: -bernoulli_quasi_likelihood(np.array([c]), y, np.ones((n, 1)), "cloglog", off).sum(),
        bounds=(-5, 5), method="bounded", options={"xatol": 1e-12})
    assert plain.extra["null_log_likelihood"] == pytest.approx(-null.fun, rel=1e-10)
    robust = oe.cloglog(**common, covariance="robust")
    assert_allclose(np.asarray(robust.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=1e-7)
    assert robust.tests["model"]["label"].startswith("Wald")
    cluster = oe.cloglog(**common, cluster="firm")
    assert_allclose(np.asarray(cluster.covariance_matrix),
                    50 / 49 * bread @ cluster_meat(scores, data.firm) @ bread, rtol=1e-7)
    opg = oe.cloglog(**common, covariance="opg")
    assert_allclose(np.asarray(opg.covariance_matrix), np.linalg.inv(scores.T @ scores), rtol=1e-7)
    duplicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for covariance in ("nonrobust", "robust"):
        weighted = oe.cloglog(data=data, y="event", x=X, weights="fw", weight_type="fweight",
                              covariance=covariance)
        copies = oe.cloglog(data=duplicated, y="event", x=X, covariance=covariance)
        assert weighted.nobs == copies.nobs == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(copies), rtol=1e-9)
        assert_allclose(errors(weighted), errors(copies), rtol=1e-8)
        assert weighted.extra["zero_outcomes"] == copies.extra["zero_outcomes"]
        for name in weighted.metrics:
            assert weighted.metrics[name] == pytest.approx(copies.metrics[name], rel=1e-9), name
    x, w = design(data).to_numpy(), data.aw.to_numpy()
    brute = optimize.minimize(lambda t: -(w * bernoulli_quasi_likelihood(t, y, x, "cloglog")).sum(),
                              np.zeros(3), method="BFGS", options={"gtol": 1e-9})
    importance = oe.cloglog(data=data, y="event", x=X, weights="aw", weight_type="iweight")
    assert_allclose(estimates(importance), brute.x, atol=1e-5)
    sampling = oe.cloglog(data=data, y="event", x=X, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), estimates(importance), rtol=1e-10)
    # Stata's cloglog takes no aweights.
    with pytest.raises(AnalysisError) as caught:
        oe.cloglog(data=data, y="event", x=X, weights="aw", weight_type="aweight")
    assert caught.value.code == "invalid_spec" and "aweight" in str(caught.value)


def test_cloglog_separation_errors_and_round_trip(data):
    separated = data.assign(perfect=(data.x1 > 0).astype(float))
    with pytest.raises(AnalysisError) as caught:
        oe.cloglog(data=separated, y="perfect", x=X)
    assert caught.value.code == "separation_detected"
    for arguments, code in [
        ({"y": "share"}, "invalid_binary_outcome"), ({"x": "x1"}, "invalid_spec"),
        ({"covariance": "HC1"}, "invalid_spec"), ({"x": ["nope"]}, "missing_columns"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.cloglog(**{"data": data, "y": "event", "x": X, **arguments})
        assert caught.value.code == code, arguments
    with pytest.raises(AnalysisError) as caught:
        oe.cloglog(data=data.assign(one=1.0), y="one", x=X)
    assert caught.value.code == "constant_outcome"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="cloglog", outcome="event", predictors=X, columns={"exposure": "aw"})
    holes = data.copy()
    holes.loc[[0, 9], "x1"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.cloglog(data=holes, y="event", x=X)
    assert caught.value.code == "missing_values"
    assert oe.cloglog(data=holes, y="event", x=X, missing="drop").nobs == 498
    result = oe.cloglog(data=data.assign(twice=2 * data.x1), y="event", x=["x1", "sector", "twice"],
                        categorical=["sector"], cluster="firm")
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "sector[b]", "sector[c]"]
    assert result.provenance["omitted_terms"] == ["twice"]
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Complementary log-log regression — event" in text and "Wald chi2 test of the slopes: chi2(3)" in text
    assert "sector[b]" in str(result.to_latex()).replace(r"\_", "_")


# ---- fracreg -------------------------------------------------------------------------------


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_fracreg_matches_brute_force_and_statsmodels(data, link):
    result = oe.fracreg(data=data, y="share", x=X, link=link)
    y, x = data.share.to_numpy(), design(data).to_numpy()
    assert (y == 0).any() and (y == 1).any()
    brute = optimize.minimize(lambda t: -bernoulli_quasi_likelihood(t, y, x, link).sum(), np.zeros(3),
                              method="BFGS", options={"gtol": 1e-10})
    assert_allclose(estimates(result), brute.x, atol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(-brute.fun, abs=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        family = sm.families.Binomial(L.Logit() if link == "logit" else L.Probit())
        ref = sm.GLM(data.share, design(data), family=family).fit(tol=1e-13, tol_criterion="params", maxiter=300)
    params = ref.params.values
    assert_allclose(estimates(result), params, rtol=1e-8)
    # Default covariance (Stata): robust sandwich with N/(N-1) and the observed information.
    scores = ref.model.score_obs(params, scale=1)
    bread = np.linalg.inv(-ref.model.hessian(params, scale=1, observed=True))
    n, k = 500, 3
    assert result.spec.covariance == "robust"
    assert_allclose(np.asarray(result.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=1e-7)
    share = y.mean()
    null = n * (share * np.log(share) + (1 - share) * np.log(1 - share))
    log_likelihood = bernoulli_quasi_likelihood(params, y, x, link).sum()
    assert result.metrics["log_likelihood"] == pytest.approx(log_likelihood, rel=1e-11)
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-11)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - log_likelihood / null, rel=1e-9)
    assert result.metrics["aic"] == pytest.approx(-2 * log_likelihood + 2 * k)
    assert list(result.metrics) == ["log_likelihood", "pseudo_r_squared", "aic", "bic", "df_resid"]
    wald = result.tests["model"]
    b, v = estimates(result)[1:], np.asarray(result.covariance_matrix)[1:, 1:]
    assert wald["statistic"] == pytest.approx(b @ np.linalg.solve(v, b), rel=1e-9) and wald["df"] == 2
    assert result.extra["quasi_likelihood"] is True and result.extra["link"] == link
    cluster = oe.fracreg(data=data, y="share", x=X, link=link, cluster="firm")
    assert_allclose(np.asarray(cluster.covariance_matrix),
                    50 / 49 * bread @ cluster_meat(scores, data.firm) @ bread, rtol=1e-7)
    nonrobust = oe.fracreg(data=data, y="share", x=X, link=link, covariance="nonrobust")
    assert_allclose(np.asarray(nonrobust.covariance_matrix), bread, rtol=1e-7)
    assert nonrobust.tests["model"]["label"].startswith("Wald")          # fracreg always reports Wald
    opg = oe.fracreg(data=data, y="share", x=X, link=link, covariance="opg")
    assert_allclose(np.asarray(opg.covariance_matrix), np.linalg.inv(scores.T @ scores), rtol=1e-7)


def test_fracreg_weights_binary_limit_and_errors(data):
    duplicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for options in ({}, {"cluster": "firm"}, {"covariance": "nonrobust"}):
        weighted = oe.fracreg(data=data, y="share", x=X, weights="fw", weight_type="fweight", **options)
        copies = oe.fracreg(data=duplicated, y="share", x=X, **options)
        assert weighted.nobs == copies.nobs
        assert_allclose(estimates(weighted), estimates(copies), rtol=1e-9)
        assert_allclose(errors(weighted), errors(copies), rtol=1e-8)
    sampling = oe.fracreg(data=data, y="share", x=X, weights="aw", weight_type="pweight")
    analytic = oe.fracreg(data=data, y="share", x=X, weights="aw", weight_type="aweight")
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-9)
    assert_allclose(errors(sampling), errors(analytic), rtol=1e-8)      # the sandwich is scale invariant
    # A 0/1 outcome reduces to the logit model.
    binary = oe.fracreg(data=data, y="event", x=X, covariance="nonrobust")
    logit = oe.glm(data=data, y="event", x=X, family="binomial")
    assert_allclose(estimates(binary), estimates(logit), rtol=1e-12)
    assert binary.metrics["log_likelihood"] == pytest.approx(logit.metrics["log_likelihood"], rel=1e-12)
    for frame, code in [
        (data.assign(share=data.share * 100), "invalid_fractional_outcome"),
        (data.assign(share=data.share - 0.5), "invalid_fractional_outcome"),
        (data.assign(share=0.4), "constant_outcome"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.fracreg(data=frame, y="share", x=X)
        assert caught.value.code == code
    with pytest.raises(AnalysisError) as caught:
        oe.fracreg(data=data, y="share", x=X, link="cloglog")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.fracreg(data=data.assign(perfect=(data.x1 > 0).astype(float)), y="perfect", x=X)
    assert caught.value.code == "separation_detected"
    result = oe.fracreg(data=data, y="share", x=["x1", "sector"], categorical=["sector"], link="probit")
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Fractional response regression — share" in text and "Covariance: robust" in text
    assert "Wald chi2 test of the slopes: chi2(3)" in text and "Pseudo" in str(result.to_latex())


# ---- betareg -------------------------------------------------------------------------------


def beta_log_likelihood(theta, y, x, z, link="logit", scale_link="log"):
    """Independent beta regression log likelihood per observation (scipy.stats.beta)."""
    k = x.shape[1]
    mu = INVERSE[link](x @ theta[:k])
    phi = SCALE_INVERSE[scale_link](z @ theta[k:])
    return stats.beta.logpdf(y, mu * phi, (1 - mu) * phi)


def test_betareg_matches_statsmodels_beta_model(data):
    result = oe.betareg(data=data, y="rate", x=X, scale=["z1"])
    z = sm.add_constant(data[["z1"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = BetaModel(data.rate, design(data), exog_precision=z).fit(disp=0, method="bfgs", maxiter=2000,
                                                                      gtol=1e-9)
    n, k, q = 500, 3, 2
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2", "scale:Intercept", "scale:z1"]
    assert [c.equation for c in result.coefficients] == ["rate"] * 3 + ["scale"] * 2
    assert_allclose(estimates(result), ref.params.values, rtol=1e-6, atol=1e-8)
    assert_allclose(errors(result), ref.bse.values, rtol=1e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-10)
    assert result.metrics["aic"] == pytest.approx(-2 * ref.llf + 2 * (k + q), rel=1e-10)
    assert result.metrics["bic"] == pytest.approx(-2 * ref.llf + (k + q) * np.log(n), rel=1e-10)
    assert result.metrics["df_resid"] == n - k - q
    assert list(result.metrics) == ["log_likelihood", "aic", "bic", "pseudo_r_squared", "df_resid"]
    # Pseudo R-squared: squared correlation between the linear predictor and logit(y).
    eta = design(data).to_numpy() @ estimates(result)[:3]
    correlation = np.corrcoef(eta, special.logit(data.rate.to_numpy()))[0, 1]
    assert result.metrics["pseudo_r_squared"] == pytest.approx(correlation ** 2, rel=1e-9)
    wald = result.tests["model"]
    b, v = estimates(result)[1:3], np.asarray(result.covariance_matrix)[1:3, 1:3]
    assert wald["statistic"] == pytest.approx(b @ np.linalg.solve(v, b), rel=1e-9) and wald["df"] == 2
    assert "precision" not in result.extra and len(result.extra["precision_range"]) == 2
    assert result.title == "Beta regression" and result.inference["correction"] == "observed information"
    assert_allclose([p["fitted"] for p in result.predictions[:4]], special.expit(eta[:4]), rtol=1e-9)
    # Constant precision: phi itself with a delta-method interval.
    constant = oe.betareg(data=data, y="rate", x=X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = BetaModel(data.rate, design(data)).fit(disp=0, method="bfgs", maxiter=2000, gtol=1e-9)
    assert_allclose(estimates(constant), ref.params.values, rtol=1e-6, atol=1e-8)
    record = constant.extra["precision"]
    log_phi, se = estimates(constant)[3], errors(constant)[3]
    critical = stats.norm.ppf(0.975)
    assert record["estimate"] == pytest.approx(math.exp(log_phi)) and constant.spec.columns == {}
    assert record["std_error"] == pytest.approx(math.exp(log_phi) * se, rel=1e-10)
    assert record["ci_low"] == pytest.approx(math.exp(log_phi - critical * se), rel=1e-9)
    assert record["ci_high"] == pytest.approx(math.exp(log_phi + critical * se), rel=1e-9)


LINK_CASES = [("probit", "log"), ("cloglog", "log"), ("loglog", "log"), ("logit", "identity"),
              ("logit", "sqrt"), ("probit", "sqrt")]


@pytest.mark.parametrize("link,scale_link", LINK_CASES, ids=[f"{a}-{b}" for a, b in LINK_CASES])
def test_betareg_links_maximize_an_independent_likelihood(data, link, scale_link):
    result = oe.betareg(data=data, y="rate", x=X, link=link, scale_link=scale_link)
    theta = estimates(result)
    y, x, z = data.rate.to_numpy(), design(data).to_numpy(), np.ones((len(data), 1))

    def total(t):
        return beta_log_likelihood(t, y, x, z, link, scale_link).sum()

    assert result.metrics["log_likelihood"] == pytest.approx(total(theta), rel=1e-11)
    # The estimate is a stationary point of the independent likelihood ...
    gradient = numeric_scores(lambda t: beta_log_likelihood(t, y, x, z, link, scale_link), theta).sum(axis=0)
    assert np.abs(gradient).max() < 1e-4
    # ... and a derivative-free search started nearby does not find a better one.
    brute = optimize.minimize(lambda t: -total(t), theta + 0.02, method="Nelder-Mead",
                              options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000, "maxfev": 20000})
    assert -brute.fun <= total(theta) + 1e-8
    assert_allclose(brute.x, theta, atol=5e-5)
    hessian = numeric_hessian(total, theta)
    assert_allclose(np.asarray(result.covariance_matrix), np.linalg.inv(-hessian), rtol=5e-4, atol=1e-9)
    assert result.extra["link"] == link and result.extra["scale_link"] == scale_link
    statsmodels_link = {"probit": L.Probit(), "cloglog": L.CLogLog(), "loglog": L.LogLog(), "logit": L.Logit()}
    precision_link = {"log": L.Log(), "identity": L.Identity(), "sqrt": L.Sqrt()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = BetaModel(data.rate, design(data), link=statsmodels_link[link],
                        link_precision=precision_link[scale_link]).fit(
            disp=0, method="bfgs", maxiter=3000, gtol=1e-9, start_params=theta + 0.01)
    assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-9)
    assert_allclose(theta, ref.params.values, rtol=1e-4, atol=1e-6)
    # The reported precision is on its natural scale whatever the link.
    assert result.extra["precision"]["estimate"] == pytest.approx(float(SCALE_INVERSE[scale_link](theta[3])))


def test_betareg_robust_cluster_and_opg_from_numerical_scores(data):
    y, x = data.rate.to_numpy(), design(data).to_numpy()
    z = sm.add_constant(data[["z1"]]).to_numpy()
    common = {"data": data, "y": "rate", "x": X, "scale": ["z1"]}
    robust = oe.betareg(**common, covariance="robust")
    theta = estimates(robust)
    scores = numeric_scores(lambda t: beta_log_likelihood(t, y, x, z), theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: beta_log_likelihood(t, y, x, z).sum(), theta))
    n = len(data)
    assert_allclose(np.asarray(robust.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=5e-4, atol=1e-9)
    cluster = oe.betareg(**common, cluster="firm")
    assert_allclose(np.asarray(cluster.covariance_matrix),
                    50 / 49 * bread @ cluster_meat(scores, data.firm) @ bread, rtol=5e-4, atol=1e-9)
    twoway = oe.betareg(**common, cluster=["firm", "state"])
    both = data.firm.astype(str) + "/" + data.state
    # The likelihood is maximized on the centered designs (the constant of each equation absorbs
    # the shift), and the covariance is mapped back: theta = T theta_c, V = T V_c T'. The
    # positive-semidefinite repair below depends on the coordinates, so it is done there too.
    transform = np.eye(5)
    transform[0, 1:3] = -x[:, 1:].mean(axis=0)
    transform[3, 4] = -z[:, 1].mean()
    centered = scores @ transform
    meat = (cluster_meat(centered, data.firm) + cluster_meat(centered, data.state)
            - cluster_meat(centered, both))
    # Inclusion-exclusion need not be positive semidefinite (8 state clusters, 5 parameters):
    # negative eigenvalues are then set to zero (Cameron, Gelbach and Miller 2011) and recorded.
    values, vectors = np.linalg.eigh(meat)
    indefinite = bool(values.min() < -1e-10 * values.max())
    assert twoway.inference["psd_adjusted"] is indefinite
    assert any("not positive semidefinite" in warning for warning in twoway.warnings) is indefinite
    meat = (vectors * values.clip(min=0)) @ vectors.T
    bread_centered = np.linalg.inv(transform) @ bread @ np.linalg.inv(transform).T
    expected = transform @ (8 / 7 * bread_centered @ meat @ bread_centered) @ transform.T
    assert_allclose(np.asarray(twoway.covariance_matrix), expected, rtol=2e-3, atol=1e-8)
    assert twoway.inference["cluster_counts"][:2] == [50, 8]
    opg = oe.betareg(**common, covariance="opg")
    assert_allclose(np.asarray(opg.covariance_matrix), np.linalg.inv(scores.T @ scores), rtol=5e-4, atol=1e-9)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "loglog"])
@pytest.mark.parametrize("scale_link", ["log", "identity", "sqrt"])
def test_betareg_analytic_derivatives_match_numerical_ones(link, scale_link):
    rng = np.random.default_rng(17)
    n = 150
    x = torch.as_tensor(np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, size=n)]))
    z = torch.as_tensor(np.column_stack([np.ones(n), rng.normal(size=n)]))
    y = torch.as_tensor(rng.beta(2.0, 3.0, size=n))
    prior = torch.as_tensor(rng.uniform(0.5, 2.0, size=n))
    objective = BetaObjective(x, z, y, prior, make_link(link), ScaleLink(scale_link))
    theta = torch.tensor([-0.3, 0.2, 0.1, 4.0 if scale_link == "identity" else 1.4, 0.1], dtype=torch.float64)
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-7, report
    assert report["hessian_max_rel_error"] < 1e-6, report
    expected = (prior.numpy() * beta_log_likelihood(theta.numpy(), y.numpy(), x.numpy(), z.numpy(), link,
                                                    scale_link)).sum()
    assert float(objective.value(theta)) == pytest.approx(expected, rel=1e-11)
    rows = objective.score_rows(theta)
    assert_allclose((rows * prior[:, None]).sum(dim=0).numpy(), objective(theta)[1].numpy(), rtol=1e-10)


def test_betareg_weights_inputs_and_errors(data):
    duplicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for options in ({}, {"covariance": "robust"}, {"cluster": "firm"}):
        weighted = oe.betareg(data=data, y="rate", x=X, scale=["z1"], weights="fw", weight_type="fweight",
                              **options)
        copies = oe.betareg(data=duplicated, y="rate", x=X, scale=["z1"], **options)
        assert weighted.nobs == copies.nobs == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(copies), rtol=1e-7, atol=1e-9)
        assert_allclose(errors(weighted), errors(copies), rtol=1e-6)
        for name in weighted.metrics:
            assert weighted.metrics[name] == pytest.approx(copies.metrics[name], rel=1e-7), name
    y, x, w = data.rate.to_numpy(), design(data).to_numpy(), data.aw.to_numpy()
    z = np.ones((len(data), 1))
    importance = oe.betareg(data=data, y="rate", x=X, weights="aw", weight_type="iweight")
    theta = estimates(importance)
    gradient = (w[:, None] * numeric_scores(lambda t: beta_log_likelihood(t, y, x, z), theta)).sum(axis=0)
    assert np.abs(gradient).max() < 1e-4
    assert importance.metrics["log_likelihood"] == pytest.approx((w * beta_log_likelihood(theta, y, x, z)).sum())
    sampling = oe.betareg(data=data, y="rate", x=X, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), theta, rtol=1e-10)
    # Categorical and collinear terms in both equations.
    frame = data.assign(twice=2 * data.z1)
    result = oe.betareg(data=frame, y="rate", x=["x1", "sector"], scale=["z1", "sector", "twice"],
                        categorical=["sector"])
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "sector[b]", "sector[c]", "scale:Intercept", "scale:z1", "scale:sector[b]",
        "scale:sector[c]"]
    assert result.provenance["omitted_terms"] == ["scale:twice"]
    assert result.tests["model"]["df"] == 3
    no_constant = oe.betareg(data=data, y="rate", x=X, intercept=False)
    assert [c.term for c in no_constant.coefficients] == ["x1", "x2", "scale:Intercept"]
    for frame, code in [
        (data.assign(rate=data.share), "invalid_fractional_outcome"),          # contains 0 and 1
        (data.assign(rate=data.rate + 1), "invalid_fractional_outcome"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.betareg(data=frame, y="rate", x=X)
        assert caught.value.code == code and "fracreg" in str(caught.value)
    for arguments, code in [
        ({"scale": "z1"}, "invalid_spec"), ({"link": "log"}, "invalid_spec"),
        ({"scale_link": "logit"}, "invalid_spec"), ({"weights": "aw", "weight_type": "aweight"}, "invalid_spec"),
        ({"weights": "aw", "weight_type": "pweight", "covariance": "nonrobust"}, "unsupported_covariance"),
        ({"scale": ["nope"]}, "missing_columns"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.betareg(**{"data": data, "y": "rate", "x": X, **arguments})
        assert caught.value.code == code, arguments
    holes = data.copy()
    holes.loc[[4, 8], "z1"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.betareg(data=holes, y="rate", x=X, scale=["z1"])
    assert caught.value.code == "missing_values"
    assert oe.betareg(data=holes, y="rate", x=X, scale=["z1"], missing="drop").nobs == 498
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored == result
    assert json.loads(result.model_dump_json())["coefficients"][4]["equation"] == "scale"
    text = result.summary()
    assert "Beta regression — rate" in text and "[scale]" in text and "scale:z1" in text
    assert "Wald chi2 test of the mean-equation slopes: chi2(3)" in text
    assert "scale:z1" in str(result.to_latex()).replace(r"\_", "_")


def test_fractional_models_handle_large_samples_in_linear_time():
    rng = np.random.default_rng(21)
    n = 200_000
    frame = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    columns = list(frame.columns)
    mean = special.expit(0.3 + 0.4 * frame.x0 - 0.2 * frame.x1)
    frame["rate"] = rng.beta(mean * 6, (1 - mean) * 6).clip(1e-6, 1 - 1e-6)
    frame["event"] = rng.binomial(1, mean)
    frame["g"] = rng.integers(0, 2000, size=n)
    beta = oe.betareg(data=frame, y="rate", x=columns)
    assert abs(beta.coefficients[1].estimate - 0.4) < 0.01
    assert abs(beta.extra["precision"]["estimate"] - 6) < 0.1
    fractional = oe.fracreg(data=frame, y="rate", x=columns, cluster="g")
    assert abs(fractional.coefficients[1].estimate - 0.4) < 0.01
    binary = oe.cloglog(data=frame, y="event", x=columns)
    assert binary.nobs == n and len(binary.predictions) == 400
