"""poisson, nbreg and ppmlhdfe against independent oracles.

Oracles: statsmodels Poisson / NegativeBinomial / GLM, scipy.stats.nbinom with
scipy.optimize (brute-force likelihood maximization), numerical scores and
Hessians of independently written log likelihoods, and the explicit
dummy-variable Poisson regression for ppmlhdfe.
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
from scipy import optimize, stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.glm.kernels import NegativeBinomialObjective, ppml_irls
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec, ResultBundle

X = ["x1", "x2"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(77123)
    n = 600
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "firm": np.repeat(np.arange(60), 10), "state": rng.choice(list("abcdefgh"), size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "expo": rng.uniform(0.5, 3.0, size=n),
    })
    mu = frame.expo * np.exp(0.4 + 0.35 * frame.x1 - 0.25 * frame.x2)
    frame["count"] = rng.poisson(mu).astype(float)
    frame["negative"] = frame.x1 - 1.0
    frame["over"] = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)       # alpha = 0.5
    frame["f1"] = rng.integers(0, 25, size=n)
    frame["f2"] = rng.integers(0, 10, size=n)
    a1, a2 = rng.normal(scale=0.4, size=25), rng.normal(scale=0.4, size=10)
    frame["trade"] = rng.poisson(np.exp(a1[frame.f1] + a2[frame.f2] + 0.3 * frame.x1 - 0.2 * frame.x2)
                                 * frame.expo).astype(float)
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


# ---- poisson -------------------------------------------------------------------------------


def test_poisson_matches_statsmodels_and_stata_reporting(data):
    result = oe.poisson(data=data, y="count", x=X)
    ref = sm.Poisson(data["count"], design(data)).fit(disp=0, method="newton", tol=1e-13)
    glm = sm.GLM(data["count"], design(data), family=sm.families.Poisson()).fit(tol=1e-13)
    n, k = 600, 3
    assert_allclose(estimates(result), ref.params.values, rtol=1e-9)
    assert_allclose(errors(result), ref.bse.values, rtol=1e-8)
    assert_allclose([c.p_value for c in result.coefficients], ref.pvalues.values, rtol=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-11)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(ref.prsquared, rel=1e-9)
    assert result.metrics["aic"] == pytest.approx(ref.aic, rel=1e-11)
    assert result.metrics["bic"] == pytest.approx(ref.bic, rel=1e-11)
    assert result.metrics["deviance"] == pytest.approx(glm.deviance, rel=1e-10)
    assert result.metrics["pearson"] == pytest.approx(glm.pearson_chi2, rel=1e-9)
    assert result.metrics["df_resid"] == n - k
    assert list(result.metrics) == ["log_likelihood", "pseudo_r_squared", "aic", "bic", "deviance",
                                    "pearson", "df_resid"]
    model = result.tests["model"]
    assert model["statistic"] == pytest.approx(ref.llr, rel=1e-9) and model["df"] == 2
    assert model["p_value"] == pytest.approx(ref.llr_pvalue, rel=1e-7) and "LR chi2" in model["label"]
    assert result.extra["null_log_likelihood"] == pytest.approx(ref.llnull, rel=1e-11)
    for name, statistic in (("gof_deviance", glm.deviance), ("gof_pearson", glm.pearson_chi2)):
        test = result.tests[name]
        assert test["statistic"] == pytest.approx(statistic, rel=1e-9) and test["df"] == n - k
        assert test["p_value"] == pytest.approx(stats.chi2.sf(statistic, n - k), rel=1e-8)
    assert result.title == "Poisson regression" and result.inference["use_t"] is False
    assert result.inference["correction"] == "observed information"
    assert result.provenance["stata_equivalent"] == ["poisson", "estat gof"]
    # The same model through glm.
    through_glm = oe.glm(data=data, y="count", x=X, family="poisson")
    assert_allclose(estimates(result), estimates(through_glm), rtol=1e-12)
    assert_allclose(errors(result), errors(through_glm), rtol=1e-12)


def test_poisson_robust_cluster_and_opg_covariances(data):
    ref = sm.Poisson(data["count"], design(data)).fit(disp=0, method="newton", tol=1e-13)
    x = design(data).to_numpy()
    mu = np.exp(x @ ref.params.values)
    scores = x * (data["count"].to_numpy() - mu)[:, None]
    bread = np.linalg.inv(x.T @ (x * mu[:, None]))
    n = len(data)
    robust = oe.poisson(data=data, y="count", x=X, covariance="robust")
    assert_allclose(np.asarray(robust.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=1e-8)
    hc0 = sm.Poisson(data["count"], design(data)).fit(disp=0, method="newton", tol=1e-13, cov_type="HC0")
    assert_allclose(errors(robust), hc0.bse.values * math.sqrt(n / (n - 1)), rtol=1e-8)
    wald = robust.tests["model"]
    b, v = estimates(robust)[1:], np.asarray(robust.covariance_matrix)[1:, 1:]
    assert wald["statistic"] == pytest.approx(b @ np.linalg.solve(v, b), rel=1e-9) and "Wald" in wald["label"]
    cluster = oe.poisson(data=data, y="count", x=X, cluster="firm")
    assert_allclose(np.asarray(cluster.covariance_matrix),
                    60 / 59 * bread @ cluster_meat(scores, data.firm) @ bread, rtol=1e-8)
    assert cluster.inference["cluster_count"] == 60 and cluster.inference["correction"].startswith(
        "cluster sandwich: G/(G-1)")
    twoway = oe.poisson(data=data, y="count", x=X, cluster=["firm", "state"])
    both = data.firm.astype(str) + "/" + data.state
    meat = cluster_meat(scores, data.firm) + cluster_meat(scores, data.state) - cluster_meat(scores, both)
    assert_allclose(np.asarray(twoway.covariance_matrix), 8 / 7 * bread @ meat @ bread, rtol=1e-8)
    assert any("Only 8 clusters" in warning for warning in twoway.warnings)
    opg = oe.poisson(data=data, y="count", x=X, covariance="opg")
    assert_allclose(np.asarray(opg.covariance_matrix), np.linalg.inv(scores.T @ scores), rtol=1e-8)
    assert opg.tests["model"]["label"].startswith("Wald")


def test_poisson_exposure_offset_and_no_constant(data):
    ref = sm.Poisson(data["count"], design(data), exposure=data.expo).fit(disp=0, method="newton", tol=1e-13)
    result = oe.poisson(data=data, y="count", x=X, exposure="expo")
    assert_allclose(estimates(result), ref.params.values, rtol=1e-9)
    assert_allclose(errors(result), ref.bse.values, rtol=1e-8)
    assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-11)
    # The null model keeps the exposure: mu_0 = exposure * sum(y) / sum(exposure).
    y = data["count"].to_numpy()
    null_mu = data.expo.to_numpy() * y.sum() / data.expo.sum()
    null = stats.poisson.logpmf(y, null_mu).sum()
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-11)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - ref.llf / null, rel=1e-9)
    offset = oe.poisson(data=data.assign(lnexpo=np.log(data.expo)), y="count", x=X, offset="lnexpo")
    assert_allclose(estimates(offset), estimates(result), rtol=1e-12)
    # Without a constant the comparison model is b = 0.
    plain = oe.poisson(data=data, y="count", x=X, intercept=False)
    ref = sm.Poisson(data["count"], data[X]).fit(disp=0, method="newton", tol=1e-13)
    assert_allclose(estimates(plain), ref.params.values, rtol=1e-9)
    assert plain.extra["null_log_likelihood"] == pytest.approx(stats.poisson.logpmf(y, 1.0).sum(), rel=1e-11)
    assert plain.tests["model"]["df"] == 2 and "no constant" in plain.extra["null_model"]


def test_poisson_weights(data):
    duplicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for covariance in ("nonrobust", "opg", "robust", "cluster"):
        options = {"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance}
        weighted = oe.poisson(data=data, y="count", x=X, exposure="expo", weights="fw", weight_type="fweight",
                              **options)
        plain = oe.poisson(data=duplicated, y="count", x=X, exposure="expo", **options)
        assert weighted.nobs == plain.nobs == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(plain), rtol=1e-10)
        assert_allclose(errors(weighted), errors(plain), rtol=1e-9)
        for name in weighted.metrics:
            assert weighted.metrics[name] == pytest.approx(plain.metrics[name], rel=1e-9), name
        for name in weighted.tests:
            assert weighted.tests[name]["statistic"] == pytest.approx(plain.tests[name]["statistic"], rel=1e-8)
    n = len(data)
    normalized = data.aw * n / data.aw.sum()
    ref = sm.GLM(data["count"], design(data), family=sm.families.Poisson(), freq_weights=normalized).fit(tol=1e-13)
    analytic = oe.poisson(data=data, y="count", x=X, weights="aw", weight_type="aweight")
    assert_allclose(estimates(analytic), ref.params.values, rtol=1e-9)
    assert_allclose(errors(analytic), ref.bse.values, rtol=1e-8)
    assert analytic.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-10) and analytic.nobs == n
    raw = sm.GLM(data["count"], design(data), family=sm.families.Poisson(), freq_weights=data.aw).fit(tol=1e-13)
    importance = oe.poisson(data=data, y="count", x=X, weights="aw", weight_type="iweight")
    assert_allclose(errors(importance), raw.bse.values, rtol=1e-8)
    sampling = oe.poisson(data=data, y="count", x=X, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    assert sampling.metrics["log_likelihood"] == pytest.approx(raw.llf, rel=1e-10)
    x = design(data).to_numpy()
    mu = np.exp(x @ raw.params.values)
    scores = x * (data.aw.to_numpy() * (data["count"].to_numpy() - mu))[:, None]
    bread = np.linalg.inv(x.T @ (x * (data.aw.to_numpy() * mu)[:, None]))
    assert_allclose(np.asarray(sampling.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=1e-8)
    with pytest.raises(AnalysisError) as caught:
        oe.poisson(data=data, y="count", x=X, weights="aw", weight_type="pweight", covariance="nonrobust")
    assert caught.value.code == "unsupported_covariance"
    zero = data.assign(fw=np.where(np.arange(n) < 5, 0.0, data.fw))
    assert oe.poisson(data=zero, y="count", x=X, weights="fw", weight_type="fweight").dropped_rows == 5


def test_poisson_inputs_and_error_codes(data):
    result = oe.poisson(data=data.assign(twice=2 * data.x1), y="count", x=["x1", "sector", "twice", "x2"],
                        categorical=["sector"])
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "sector[b]", "sector[c]", "x2"]
    assert result.provenance["omitted_terms"] == ["twice"]
    fractional = oe.poisson(data=data.assign(half=data["count"] + 0.5), y="half", x=X, covariance="robust")
    assert any("non-integer" in warning for warning in fractional.warnings)
    holes = data.copy()
    holes.loc[[1, 2], "x2"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.poisson(data=holes, y="count", x=X)
    assert caught.value.code == "missing_values"
    assert oe.poisson(data=holes, y="count", x=X, missing="drop").nobs == 598
    for arguments, code in [
        ({"y": "negative"}, "invalid_count_outcome"), ({"x": "x1"}, "invalid_spec"),
        ({"covariance": "HC1"}, "invalid_spec"), ({"exposure": "x1"}, "invalid_exposure"),
        ({"exposure": "expo", "offset": "x2"}, "invalid_spec"), ({"x": ["missing"]}, "missing_columns"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.poisson(**{"data": data, "y": "count", "x": X, **arguments})
        assert caught.value.code == code, arguments
    with pytest.raises(AnalysisError) as caught:
        oe.poisson(data=data.assign(zero=0.0), y="zero", x=X)
    assert caught.value.code == "constant_outcome"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="poisson", outcome="count", predictors=X, options={"family": "poisson"})
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored == result
    text = result.summary()
    assert "Poisson regression — count" in text and "LR chi2 test against the constant-only model: chi2(4)" in text
    assert "Deviance goodness of fit: chi2(595)" in text and "pseudo_r_squared:" in text
    assert "Pseudo" in str(result.to_latex())


# ---- nbreg ---------------------------------------------------------------------------------


def nb_log_likelihood(theta, y, x, offset, form):
    """Independent negative binomial log likelihood per observation (scipy.stats.nbinom)."""
    mu = np.exp(x @ theta[:-1] + offset)
    dispersion = np.exp(theta[-1])
    if form == "mean":
        size = 1 / dispersion
        return stats.nbinom.logpmf(y, size, size / (size + mu))
    return stats.nbinom.logpmf(y, mu / dispersion, 1 / (1 + dispersion))


@pytest.mark.parametrize("form,method,name", [("mean", "nb2", "alpha"), ("constant", "nb1", "delta")])
def test_nbreg_matches_statsmodels_and_a_brute_force_optimum(data, form, method, name):
    result = oe.nbreg(data=data, y="over", x=X, exposure="expo", dispersion=form)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.NegativeBinomial(data.over, design(data), exposure=data.expo, loglike_method=method).fit(
            disp=0, method="bfgs", maxiter=2000, gtol=1e-9)
    n, k = 600, 3
    value = ref.params.values[-1]
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2", f"/ln{name}"]
    assert [c.equation for c in result.coefficients] == ["over"] * 3 + [None]
    assert_allclose(estimates(result)[:3], ref.params.values[:3], rtol=1e-6, atol=1e-8)
    assert math.exp(estimates(result)[3]) == pytest.approx(value, rel=1e-6)
    assert result.metrics[name] == pytest.approx(value, rel=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-10)
    assert result.metrics["aic"] == pytest.approx(-2 * ref.llf + 2 * (k + 1), rel=1e-10)
    assert result.metrics["bic"] == pytest.approx(-2 * ref.llf + (k + 1) * np.log(n), rel=1e-10)
    assert result.metrics["df_resid"] == n - k - 1
    assert list(result.metrics) == ["log_likelihood", "pseudo_r_squared", "aic", "bic", name, "df_resid"]
    # Brute-force maximization of an independently written likelihood.
    y, x, offset = data.over.to_numpy(), design(data).to_numpy(), np.log(data.expo.to_numpy())
    brute = optimize.minimize(lambda t: -nb_log_likelihood(t, y, x, offset, form).sum(),
                              np.array([0.0, 0.0, 0.0, 0.0]), method="BFGS", options={"gtol": 1e-8})
    assert_allclose(estimates(result), brute.x, atol=2e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(-brute.fun, abs=1e-7)
    # Observed-information standard errors from a numerical Hessian of that likelihood.
    theta = estimates(result)
    hessian = numeric_hessian(lambda t: nb_log_likelihood(t, y, x, offset, form).sum(), theta)
    assert_allclose(np.asarray(result.covariance_matrix), np.linalg.inv(-hessian), rtol=2e-4, atol=1e-9)
    # At the maximum the information transforms exactly: se(ln a) = se(a) / a.
    assert errors(result)[3] == pytest.approx(ref.bse.values[-1] / value, rel=2e-4)
    assert_allclose(errors(result)[:3], ref.bse.values[:3], rtol=2e-4)
    record = result.extra["alpha"]
    critical = stats.norm.ppf(0.975)
    assert record["parameter"] == name and record["estimate"] == pytest.approx(value, rel=1e-6)
    assert record["std_error"] == pytest.approx(math.exp(theta[3]) * errors(result)[3], rel=1e-12)
    assert record["ci_low"] == pytest.approx(math.exp(theta[3] - critical * errors(result)[3]), rel=1e-9)
    assert record["ci_high"] == pytest.approx(math.exp(theta[3] + critical * errors(result)[3]), rel=1e-9)
    # LR test of the dispersion against Poisson: chibar2(01).
    poisson = oe.poisson(data=data, y="over", x=X, exposure="expo")
    statistic = 2 * (result.metrics["log_likelihood"] - poisson.metrics["log_likelihood"])
    test = result.tests["alpha"]
    assert test["statistic"] == pytest.approx(statistic, rel=1e-10) and test["distribution"] == "chibar2"
    assert test["p_value"] == pytest.approx(0.5 * stats.chi2.sf(statistic, 1), rel=1e-7)
    assert result.extra["poisson_log_likelihood"] == pytest.approx(poisson.metrics["log_likelihood"])
    # Model LR test against the constant-only negative binomial model.
    assert result.extra["null_log_likelihood"] == pytest.approx(ref.llnull, rel=1e-8)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(ref.prsquared, rel=1e-6)
    model = result.tests["model"]
    assert model["statistic"] == pytest.approx(ref.llr, rel=1e-6) and model["df"] == 2
    assert result.extra["dispersion"] == form and result.title == "Negative binomial regression"


@pytest.mark.parametrize("form", ["mean", "constant"])
def test_nbreg_robust_cluster_and_opg_from_numerical_scores(data, form):
    y, x, offset = data.over.to_numpy(), design(data).to_numpy(), np.log(data.expo.to_numpy())
    common = {"data": data, "y": "over", "x": X, "exposure": "expo", "dispersion": form}
    robust = oe.nbreg(**common, covariance="robust")
    theta = estimates(robust)
    scores = numeric_scores(lambda t: nb_log_likelihood(t, y, x, offset, form), theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: nb_log_likelihood(t, y, x, offset, form).sum(), theta))
    n = len(data)
    assert_allclose(np.asarray(robust.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=5e-4, atol=1e-9)
    assert "alpha" not in robust.tests and "alpha_test_note" in robust.extra
    assert robust.tests["model"]["label"].startswith("Wald") and robust.tests["model"]["df"] == 2
    cluster = oe.nbreg(**common, cluster="firm")
    assert_allclose(np.asarray(cluster.covariance_matrix),
                    60 / 59 * bread @ cluster_meat(scores, data.firm) @ bread, rtol=5e-4, atol=1e-9)
    opg = oe.nbreg(**common, covariance="opg")
    assert_allclose(np.asarray(opg.covariance_matrix), np.linalg.inv(scores.T @ scores), rtol=5e-4, atol=1e-9)
    assert "alpha" in opg.tests


@pytest.mark.parametrize("form", ["mean", "constant"])
def test_nbreg_analytic_derivatives_match_numerical_ones(form):
    rng = np.random.default_rng(31)
    n = 150
    x = torch.as_tensor(np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, size=n)]))
    mu = np.exp(0.5 + 0.3 * x[:, 1].numpy())
    y = torch.as_tensor(rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float))
    prior = torch.as_tensor(rng.uniform(0.5, 2.0, size=n))
    offset = torch.as_tensor(rng.normal(scale=0.1, size=n))
    objective = NegativeBinomialObjective(x, y, prior, offset, form)
    for log_dispersion in (math.log(0.05), math.log(0.6), math.log(4.0)):
        theta = torch.tensor([0.4, 0.3, -0.1, log_dispersion], dtype=torch.float64)
        report = check_derivatives(objective, theta)
        assert report["gradient_max_rel_error"] < 1e-7, (form, log_dispersion, report)
        assert report["hessian_max_rel_error"] < 1e-6, (form, log_dispersion, report)
        # The objective is the weighted sum of the independent per-observation likelihood.
        expected = (prior.numpy() * nb_log_likelihood(theta.numpy(), y.numpy(), x.numpy(), offset.numpy(),
                                                      form)).sum()
        assert float(objective.value(theta)) == pytest.approx(expected, rel=1e-11)
        # Score rows are finite and sum (with the weights) to the gradient.
        rows = objective.score_rows(theta)
        assert rows.shape == (n, 4) and bool(torch.isfinite(rows).all())
        assert_allclose((rows * prior[:, None]).sum(dim=0).numpy(), objective(theta)[1].numpy(), rtol=1e-10)


def test_nbreg_weights_and_offsets(data):
    duplicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for form in ("mean", "constant"):
        for covariance in ("nonrobust", "robust", "cluster"):
            options = {"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance}
            weighted = oe.nbreg(data=data, y="over", x=X, dispersion=form, weights="fw", weight_type="fweight",
                                **options)
            plain = oe.nbreg(data=duplicated, y="over", x=X, dispersion=form, **options)
            assert weighted.nobs == plain.nobs == int(data.fw.sum())
            assert_allclose(estimates(weighted), estimates(plain), rtol=1e-7, atol=1e-9)
            assert_allclose(errors(weighted), errors(plain), rtol=1e-6)
            for name in weighted.metrics:
                assert weighted.metrics[name] == pytest.approx(plain.metrics[name], rel=1e-7), name
    # iweights against a weighted brute-force optimum; aweights are the same weights rescaled to N.
    y, x, w = data.over.to_numpy(), design(data).to_numpy(), data.aw.to_numpy()
    zero = np.zeros(len(data))
    brute = optimize.minimize(lambda t: -(w * nb_log_likelihood(t, y, x, zero, "mean")).sum(),
                              np.zeros(4), method="BFGS", options={"gtol": 1e-8})
    importance = oe.nbreg(data=data, y="over", x=X, weights="aw", weight_type="iweight")
    assert_allclose(estimates(importance), brute.x, atol=2e-5)
    assert importance.metrics["log_likelihood"] == pytest.approx(-brute.fun, abs=1e-6)
    analytic = oe.nbreg(data=data, y="over", x=X, weights="aw", weight_type="aweight")
    assert_allclose(estimates(analytic), estimates(importance), rtol=1e-8)
    scale = len(data) / w.sum()
    assert analytic.metrics["log_likelihood"] == pytest.approx(importance.metrics["log_likelihood"] * scale,
                                                               rel=1e-9)
    assert_allclose(errors(analytic), errors(importance) / math.sqrt(scale), rtol=1e-7)
    sampling = oe.nbreg(data=data, y="over", x=X, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust" and "alpha" not in sampling.tests
    assert_allclose(estimates(sampling), estimates(importance), rtol=1e-10)
    # offset and exposure are the same model.
    a = oe.nbreg(data=data, y="over", x=X, exposure="expo")
    b = oe.nbreg(data=data.assign(lnexpo=np.log(data.expo)), y="over", x=X, offset="lnexpo")
    assert_allclose(estimates(a), estimates(b), rtol=1e-10)
    no_constant = oe.nbreg(data=data, y="over", x=X, intercept=False)
    assert [c.term for c in no_constant.coefficients] == ["x1", "x2", "/lnalpha"]
    assert "pseudo_r_squared" in no_constant.metrics


def test_nbreg_boundary_errors_and_round_trip(data):
    rng = np.random.default_rng(4)
    under = data.assign(under=rng.binomial(8, 0.4, size=len(data)).astype(float))    # variance < mean
    for form in ("mean", "constant"):
        with pytest.raises(AnalysisError) as caught:
            oe.nbreg(data=under, y="under", x=X, dispersion=form)
        assert caught.value.code == "boundary_solution" and "oe.poisson" in str(caught.value)
    for arguments, code in [
        ({"y": "negative"}, "invalid_count_outcome"), ({"dispersion": "nb3"}, "invalid_spec"),
        ({"x": "x1"}, "invalid_spec"), ({"exposure": "x2"}, "invalid_exposure"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.nbreg(**{"data": data, "y": "over", "x": X, **arguments})
        assert caught.value.code == code, arguments
    with pytest.raises(AnalysisError) as caught:
        oe.nbreg(data=data.assign(zero=0.0), y="zero", x=X)
    assert caught.value.code == "constant_outcome"
    result = oe.nbreg(data=data, y="over", x=["x1", "sector"], categorical=["sector"], cluster="firm")
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Negative binomial regression — over" in text and "/lnalpha" in text and "[over]" in text
    assert "alpha:" in text and "Wald chi2 test of the slopes: chi2(3)" in text
    assert "/lnalpha" in str(result.to_latex())
    assert json.loads(result.model_dump_json())["extra"]["alpha"]["parameter"] == "alpha"


# ---- ppmlhdfe ------------------------------------------------------------------------------


def dummy_poisson(frame, y, x, absorb, weights=None, exposure=None):
    """Poisson regression on explicit fixed-effect dummies (full column rank when connected)."""
    blocks = []
    for position, name in enumerate(absorb):
        block = pd.get_dummies(frame[name].astype(str), prefix=name, dtype=float)
        blocks.append(block if position == 0 else block.iloc[:, 1:])
    z = pd.concat([frame[list(x)], *blocks], axis=1)
    options = {}
    if weights is not None:
        options["freq_weights"] = np.asarray(weights)
    if exposure is not None:
        options["exposure"] = np.asarray(exposure)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = sm.GLM(frame[y], z, family=sm.families.Poisson(), **options).fit(
            tol=1e-13, tol_criterion="params", maxiter=500)
    return fitted, z.to_numpy()


def dummy_sandwich(fitted, z, y, weights, labels=None):
    """(bread, meat) of the full dummy model: scores z_i w_i (y_i - mu_i)."""
    mu = np.asarray(fitted.mu)
    bread = np.linalg.inv(z.T @ (z * (weights * mu)[:, None]))
    scores = z * (weights * (np.asarray(y) - mu))[:, None]
    meat = scores.T @ scores if labels is None else cluster_meat(scores, labels)
    return bread, meat


@pytest.mark.parametrize("absorb", [["f1"], ["f1", "f2"]])
def test_ppmlhdfe_equals_the_dummy_variable_poisson(data, absorb):
    result = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=absorb)
    assert result.nobs == 600 and result.metrics["n_separated_dropped"] == 0
    fitted, z = dummy_poisson(data, "trade", X, absorb)
    n, k, columns = 600, 2, z.shape[1]
    assert [c.term for c in result.coefficients] == X
    assert_allclose(estimates(result), fitted.params.values[:k], rtol=1e-8)
    # The linear predictor is recovered without estimating the fixed effects.
    rows = [p["row"] for p in result.predictions]
    assert_allclose([p["fitted"] for p in result.predictions], np.asarray(fitted.mu)[rows], rtol=1e-7)
    assert result.metrics["deviance"] == pytest.approx(fitted.deviance, rel=1e-9)
    assert result.metrics["log_likelihood"] == pytest.approx(fitted.llf, rel=1e-10)
    assert result.metrics["df_absorbed"] == columns - k and result.metrics["df_resid"] == n - columns
    y = data.trade.to_numpy()
    null = stats.poisson.logpmf(y, y.mean()).sum()
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - fitted.llf / null, rel=1e-9)
    assert list(result.metrics) == ["pseudo_r_squared", "log_likelihood", "deviance", "df_resid", "df_absorbed",
                                    "iterations", "n_separated_dropped", "n_singletons_dropped"]
    # Default: robust sandwich on the partialled-out regressors times N/(N-K).
    bread, meat = dummy_sandwich(fitted, z, y, np.ones(n))
    assert result.spec.covariance == "robust"
    assert_allclose(np.asarray(result.covariance_matrix), (n / (n - columns) * bread @ meat @ bread)[:k, :k],
                    rtol=1e-7)
    assert result.inference["small_sample_correction"] == pytest.approx(n / (n - columns))
    assert result.inference["use_t"] is False and result.inference["df_inference"] is None
    nonrobust = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=absorb, covariance="nonrobust")
    assert_allclose(np.asarray(nonrobust.covariance_matrix), fitted.cov_params().values[:k, :k], rtol=1e-7)
    wald = result.tests["model"]
    b, v = estimates(result), np.asarray(result.covariance_matrix)
    assert wald["statistic"] == pytest.approx(b @ np.linalg.solve(v, b), rel=1e-9) and wald["df"] == 2
    assert [entry["column"] for entry in result.extra["absorbed"]] == absorb
    assert result.extra["absorbed"][0]["levels"] == 25 and result.extra["absorbed"][0]["redundant"] == 0
    if len(absorb) == 2:
        assert result.extra["absorbed"][1] == {"column": "f2", "levels": 10, "redundant": 1, "nested": False}


def test_ppmlhdfe_cluster_covariances_follow_reghdfe_degrees_of_freedom(data):
    y, n, k = data.trade.to_numpy(), 600, 2
    fitted, z = dummy_poisson(data, "trade", X, ["f1", "f2"])
    # Cluster variable unrelated to the fixed effects: K = k + (25 + 10 - 1).
    result = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1", "f2"], cluster="firm")
    bread, meat = dummy_sandwich(fitted, z, y, np.ones(n), data.firm)
    total = k + 34
    factor = 60 / 59 * (n - 1) / (n - total)
    assert_allclose(np.asarray(result.covariance_matrix), (factor * bread @ meat @ bread)[:k, :k], rtol=1e-7)
    assert result.metrics["df_absorbed"] == 34 and result.inference["cluster_count"] == 60
    # A fixed effect nested in the cluster variable costs no degrees of freedom; when all
    # are nested one is added back for the constant.
    nested = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1", "f2"], cluster="f1")
    bread, meat = dummy_sandwich(fitted, z, y, np.ones(n), data.f1)
    factor = 25 / 24 * (n - 1) / (n - (k + 10))
    assert_allclose(np.asarray(nested.covariance_matrix), (factor * bread @ meat @ bread)[:k, :k], rtol=1e-7)
    assert nested.metrics["df_absorbed"] == 10 and nested.extra["absorbed"][0]["nested"] is True
    one_way, z1 = dummy_poisson(data, "trade", X, ["f1"])
    all_nested = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1"], cluster="f1")
    bread, meat = dummy_sandwich(one_way, z1, y, np.ones(n), data.f1)
    factor = 25 / 24 * (n - 1) / (n - (k + 1))
    assert_allclose(np.asarray(all_nested.covariance_matrix), (factor * bread @ meat @ bread)[:k, :k], rtol=1e-7)
    assert all_nested.metrics["df_absorbed"] == 1 and all_nested.extra["constant_degree_of_freedom_added"]
    # Two-way clustering: inclusion-exclusion with the smallest G.
    twoway = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1", "f2"], cluster=["firm", "state"])
    bread, _ = dummy_sandwich(fitted, z, y, np.ones(n))
    scores = z * (y - np.asarray(fitted.mu))[:, None]
    both = data.firm.astype(str) + "/" + data.state
    meat = cluster_meat(scores, data.firm) + cluster_meat(scores, data.state) - cluster_meat(scores, both)
    factor = 8 / 7 * (n - 1) / (n - total)
    assert_allclose(np.asarray(twoway.covariance_matrix), (factor * bread @ meat @ bread)[:k, :k], rtol=1e-7)


def test_ppmlhdfe_drops_separated_levels_and_singletons(data):
    frame = data.copy()
    frame.loc[frame.f1 == 3, "trade"] = 0.0              # a level whose outcomes are all zero
    separated = int((frame.f1 == 3).sum())
    extra = frame.iloc[[0]].assign(f1=999, trade=4.0)    # an observation alone in its level
    frame = pd.concat([frame, extra], ignore_index=True)
    result = oe.ppmlhdfe(data=frame, y="trade", x=X, absorb=["f1", "f2"])
    assert result.metrics["n_separated_dropped"] == separated
    assert result.metrics["n_singletons_dropped"] == 1
    assert result.nobs == len(frame) - separated - 1 and result.dropped_rows == separated + 1
    assert any("separated by a fixed effect" in warning for warning in result.warnings)
    assert any("singleton" in warning for warning in result.warnings)
    kept = frame[(frame.f1 != 3) & (frame.f1 != 999)].reset_index(drop=True)
    clean = oe.ppmlhdfe(data=kept, y="trade", x=X, absorb=["f1", "f2"])
    assert_allclose(estimates(result), estimates(clean), rtol=1e-10)
    assert_allclose(errors(result), errors(clean), rtol=1e-10)
    assert clean.metrics["n_separated_dropped"] == 0 and clean.warnings == []
    # A kept singleton is fitted exactly by its own effect: same slopes, one more observation.
    keep = oe.ppmlhdfe(data=frame, y="trade", x=X, absorb=["f1", "f2"], drop_singletons=False)
    assert keep.nobs == result.nobs + 1 and keep.metrics["n_singletons_dropped"] == 0
    assert_allclose(estimates(keep), estimates(result), rtol=1e-8)
    # Dropping can cascade: removing a zero level leaves a singleton behind in another dimension.
    chain = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "b": [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 9, 9],
                          "x": [0.1, 0.5, -0.3, 0.8, 0.2, -0.6, 0.9, -0.1, 0.4, -0.7, 0.3, 0.6],
                          "y": [1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 1.0, 3.0, 2.0, 5.0, 0.0, 0.0]})
    chained = oe.ppmlhdfe(data=chain, y="y", x=["x"], absorb=["a", "b"], covariance="nonrobust")
    assert chained.metrics["n_separated_dropped"] == 2 and chained.nobs == 10


def test_ppmlhdfe_weights_exposure_and_absorbed_regressors(data):
    duplicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for options in ({}, {"covariance": "nonrobust"}, {"cluster": "firm"}):
        weighted = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1", "f2"], weights="fw",
                               weight_type="fweight", **options)
        plain = oe.ppmlhdfe(data=duplicated, y="trade", x=X, absorb=["f1", "f2"], **options)
        assert weighted.nobs == plain.nobs == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(plain), rtol=1e-8)
        assert_allclose(errors(weighted), errors(plain), rtol=1e-7)
        for name in ("pseudo_r_squared", "log_likelihood", "deviance", "df_resid", "df_absorbed"):
            assert weighted.metrics[name] == pytest.approx(plain.metrics[name], rel=1e-8), name
    # With frequency weights a level is a singleton only if its total frequency is one.
    lone = pd.concat([data, data.iloc[[0]].assign(f1=999, fw=2.0)], ignore_index=True)
    copies = lone.loc[lone.index.repeat(lone.fw.astype(int))].reset_index(drop=True)
    a = oe.ppmlhdfe(data=lone, y="trade", x=X, absorb=["f1"], weights="fw", weight_type="fweight")
    b = oe.ppmlhdfe(data=copies, y="trade", x=X, absorb=["f1"])
    assert a.nobs == b.nobs and a.metrics["n_singletons_dropped"] == 0
    assert_allclose(errors(a), errors(b), rtol=1e-7)
    n, k = 600, 2
    w = (data.aw * n / data.aw.sum()).to_numpy()
    fitted, z = dummy_poisson(data, "trade", X, ["f1", "f2"], weights=w)
    analytic = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1", "f2"], weights="aw", weight_type="aweight")
    assert_allclose(estimates(analytic), fitted.params.values[:k], rtol=1e-8)
    bread, meat = dummy_sandwich(fitted, z, data.trade, w)
    expected = (n / (n - z.shape[1]) * bread @ meat @ bread)[:k, :k]
    assert_allclose(np.asarray(analytic.covariance_matrix), expected, rtol=1e-7)
    sampling = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1", "f2"], weights="aw", weight_type="pweight")
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-9)
    assert_allclose(errors(sampling), errors(analytic), rtol=1e-8)
    with pytest.raises(AnalysisError) as caught:
        oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1"], weights="aw", weight_type="pweight",
                    covariance="nonrobust")
    assert caught.value.code == "unsupported_covariance"
    # Exposure enters as an offset; the null model of the pseudo R-squared keeps it.
    fitted, _ = dummy_poisson(data, "trade", X, ["f1", "f2"], exposure=data.expo)
    exposed = oe.ppmlhdfe(data=data, y="trade", x=X, absorb=["f1", "f2"], exposure="expo")
    assert_allclose(estimates(exposed), fitted.params.values[:k], rtol=1e-8)
    y = data.trade.to_numpy()
    null = stats.poisson.logpmf(y, data.expo.to_numpy() * y.sum() / data.expo.sum()).sum()
    assert exposed.metrics["pseudo_r_squared"] == pytest.approx(1 - fitted.llf / null, rel=1e-8)
    # Regressors without within variation are omitted and recorded.
    level = data.assign(level=data.f1 * 2.0, twice=2 * data.x1)
    result = oe.ppmlhdfe(data=level, y="trade", x=["x1", "level", "twice", "x2"], absorb=["f1", "f2"])
    assert [c.term for c in result.coefficients] == X
    assert result.provenance["omitted_terms"] == ["level", "twice"]
    with pytest.raises(AnalysisError) as caught:
        oe.ppmlhdfe(data=level, y="trade", x=["level"], absorb=["f1"])
    assert caught.value.code == "empty_design"


def test_ppmlhdfe_kernel_error_codes_and_round_trip(data):
    for arguments, code in [
        ({"y": "negative"}, "invalid_count_outcome"), ({"x": "x1"}, "invalid_spec"),
        ({"absorb": "f1"}, "invalid_spec"), ({"absorb": []}, "invalid_spec"),
        ({"covariance": "opg"}, "invalid_spec"), ({"tolerance": 5.0}, "invalid_option"),
        ({"max_iterations": 1}, "nonconvergence"), ({"exposure": "x1"}, "invalid_exposure"),
        ({"weights": "aw", "weight_type": "iweight"}, "invalid_spec"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.ppmlhdfe(**{"data": data, "y": "trade", "x": X, "absorb": ["f1"], **arguments})
        assert caught.value.code == code, arguments
    with pytest.raises(AnalysisError) as caught:
        oe.ppmlhdfe(data=data.assign(zero=0.0), y="zero", x=X, absorb=["f1"])
    assert caught.value.code == "constant_outcome"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="ppmlhdfe", outcome="trade", predictors=X, columns={"absorb": ["f1"]})  # intercept
    # The kernel on its own: converged, and the returned working design is at the final weights.
    x = torch.as_tensor(data[X].to_numpy())
    y = torch.as_tensor(data.trade.to_numpy())
    codes = torch.as_tensor(data.f1.to_numpy())
    kernel = ppml_irls(x, y, torch.ones(600, dtype=torch.float64), None, [(codes, 25)], tol=1e-10,
                       max_iter=50, demean_tol=1e-12, demean_max_iter=1000)
    assert kernel.converged and kernel.iterations < 15
    weights = kernel.mu
    assert_allclose((kernel.x_within.T @ (weights[:, None] * kernel.x_within)).numpy(),
                    np.linalg.inv(kernel.xtx_inv.numpy()), rtol=1e-8)
    score = kernel.x_within.T @ (y - kernel.mu)
    assert float(score.abs().max()) < 1e-6
    result = oe.ppmlhdfe(data=data, y="trade", x=["x1", "sector"], categorical=["sector"], absorb=["f1", "f2"],
                         cluster="firm")
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Poisson pseudo-likelihood regression with fixed effects — trade" in text
    assert "Covariance: cluster" in text and "df_absorbed: 34" in text and "sector[b]" in text
    assert "Intercept" not in text and "x1" in str(result.to_latex())
    assert result.provenance["solver"] == "irls_partialled_out_qr"
    assert result.spec.intercept is False and result.spec.columns["absorb"] == ["f1", "f2"]


def test_count_models_handle_large_samples_in_linear_time():
    rng = np.random.default_rng(9)
    n = 200_000
    frame = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    columns = list(frame.columns)
    mu = np.exp(0.3 + 0.2 * frame.x0 - 0.1 * frame.x1)
    frame["count"] = rng.poisson(mu)
    frame["over"] = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    frame["g"] = rng.integers(0, 2000, size=n)
    frame["h"] = rng.integers(0, 500, size=n)
    poisson = oe.poisson(data=frame, y="count", x=columns, cluster="g")
    assert abs(poisson.coefficients[1].estimate - 0.2) < 0.01 and len(poisson.predictions) == 400
    negative = oe.nbreg(data=frame, y="over", x=columns)
    assert abs(negative.metrics["alpha"] - 0.5) < 0.02
    absorbed = oe.ppmlhdfe(data=frame, y="count", x=columns, absorb=["g", "h"], cluster="g")
    assert abs(absorbed.coefficients[0].estimate - 0.2) < 0.01 and absorbed.nobs == n
