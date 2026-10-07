"""glm against independent oracles: statsmodels GLM, explicit NumPy algebra, numerical derivatives."""

import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics import registry
from openecon.econometrics.glm import ESTIMATORS, EXPORTS
from openecon.econometrics.glm import families as glm_families
from openecon.econometrics.glm.kernels import GlmObjective
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec, ResultBundle

L = sm.families.links
X = ["x1", "x2"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261)
    n = 500
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "firm": np.repeat(np.arange(50), 10), "state": rng.choice(list("abcdefgh"), size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "expo": rng.uniform(0.5, 3.0, size=n), "trials": rng.integers(1, 9, size=n).astype(float),
    })
    eta = 0.4 + 0.35 * frame.x1 - 0.25 * frame.x2
    mu = np.exp(eta)
    frame["normal"] = 1 + 0.5 * frame.x1 - 0.3 * frame.x2 + rng.normal(size=n)
    frame["count"] = rng.poisson(mu).astype(float)
    frame["rate"] = rng.poisson(mu * frame.expo).astype(float)
    frame["gamma"] = rng.gamma(2.5, mu / 2.5)
    frame["wald"] = rng.wald(1 / np.sqrt(1.0 + 0.2 * frame.x1 + 0.1 * frame.x2), 3.0)
    frame["over"] = rng.negative_binomial(1 / 0.7, 1 / (1 + 0.7 * mu)).astype(float)
    p = 1 / (1 + np.exp(-(0.2 + 0.7 * frame.x1 - 0.4 * frame.x2)))
    frame["binary"] = rng.binomial(1, p).astype(float)
    frame["successes"] = rng.binomial(frame.trials.astype(int), p).astype(float)
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def design(frame, columns=X):
    return sm.add_constant(frame[list(columns)])


def reference(frame, y, family, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.GLM(frame[y], design(frame), family=family, **kwargs).fit(
            tol=1e-13, tol_criterion="params", maxiter=500)


CASES = [
    ("gaussian", "identity", "normal", lambda: sm.families.Gaussian()),
    ("gaussian", "log", "gamma", lambda: sm.families.Gaussian(L.Log())),
    ("binomial", "logit", "binary", lambda: sm.families.Binomial()),
    ("binomial", "probit", "binary", lambda: sm.families.Binomial(L.Probit())),
    ("binomial", "cloglog", "binary", lambda: sm.families.Binomial(L.CLogLog())),
    ("binomial", "loglog", "binary", lambda: sm.families.Binomial(L.LogLog())),
    ("poisson", "log", "count", lambda: sm.families.Poisson()),
    ("poisson", "identity", "gamma", lambda: sm.families.Poisson(L.Identity())),
    ("gamma", "reciprocal", "gamma", lambda: sm.families.Gamma()),
    ("gamma", "log", "gamma", lambda: sm.families.Gamma(L.Log())),
    ("gamma", "identity", "gamma", lambda: sm.families.Gamma(L.Identity())),
    ("inverse_gaussian", "inverse_squared", "wald", lambda: sm.families.InverseGaussian()),
    ("inverse_gaussian", "log", "wald", lambda: sm.families.InverseGaussian(L.Log())),
    ("nbinomial", "log", "over", lambda: sm.families.NegativeBinomial(alpha=1.0)),
]


def test_manifest_registers_the_family_and_public_functions():
    names = ["glm", "poisson", "nbreg", "cloglog", "fracreg", "betareg", "ppmlhdfe"]
    assert [info.name for info in ESTIMATORS] == names and EXPORTS == {}
    assert all(info.family == "glm" and info.inference == "z" for info in ESTIMATORS)
    for name in names:
        assert callable(getattr(oe, name)) and name in dir(oe)
        assert registry.get(name).function == name
    assert registry.get("fracreg").default_covariance == "robust"
    assert registry.get("ppmlhdfe").default_covariance == "robust"
    assert registry.get("ppmlhdfe").role("absorb").required
    record = oe.capabilities()["estimators"]["glm"]
    assert record["covariances"] == ["nonrobust", "opg", "robust", "cluster"]
    assert record["options"]["family"]["choices"] == list(glm_families.FAMILY_NAMES)
    assert record["options"]["link"]["choices"] == list(glm_families.LINK_NAMES)
    assert set(record["columns"]) == {"offset", "exposure", "trials"}
    assert record["weights"] == ["fweight", "aweight", "pweight", "iweight"]
    assert oe.capabilities()["estimators"]["cloglog"]["binary_outcome"] is True
    # The manifest must stay importable without the tensor runtime.
    code = ("import sys, openecon.econometrics.glm as g; "
            "assert 'torch' not in sys.modules and 'pandas' not in sys.modules; "
            "print(len(g.ESTIMATORS))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "7"


@pytest.mark.parametrize("family,link,outcome,make", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_glm_matches_statsmodels_for_every_family_and_link(data, family, link, outcome, make):
    ref = reference(data, outcome, make())
    n, k = len(data), 3
    for optimizer in ("ml", "irls"):
        result = oe.glm(data=data, y=outcome, x=X, family=family, link=link, optimizer=optimizer)
        assert_allclose(estimates(result), ref.params.values, rtol=1e-7, atol=1e-9)
        assert result.metrics["deviance"] == pytest.approx(ref.deviance, rel=1e-9)
        assert result.metrics["pearson"] == pytest.approx(ref.pearson_chi2, rel=1e-8)
        assert result.metrics["scale"] == pytest.approx(ref.scale, rel=1e-8)
        assert result.metrics["dispersion_deviance"] == pytest.approx(ref.deviance / (n - k), rel=1e-9)
        assert result.metrics["dispersion_pearson"] == pytest.approx(ref.pearson_chi2 / (n - k), rel=1e-8)
        assert result.metrics["df_resid"] == n - k and result.nobs == n
        # nonrobust: scale * inverse information, observed for ml and expected for irls.
        hessian = ref.model.hessian(ref.params.values, scale=1, observed=optimizer == "ml")
        expected = np.linalg.inv(-hessian) * ref.scale
        assert_allclose(np.asarray(result.covariance_matrix), expected, rtol=2e-5, atol=1e-12)
        assert result.extra["information"] == ("observed" if optimizer == "ml" else "expected")
        assert result.inference["use_t"] is False and result.inference["dispersion"] == result.metrics["scale"]
        assert_allclose([c.p_value for c in result.coefficients],
                        2 * stats.norm.sf(np.abs(estimates(result) / errors(result))), rtol=1e-9)
    if family == "gaussian" and link != "identity":
        # statsmodels evaluates this case at the Pearson scale; OpenEcon concentrates phi out.
        expected_ll = -0.5 * n * (1 + np.log(2 * np.pi) + np.log(ref.deviance / n))
    elif family in {"gamma", "inverse_gaussian"}:
        # Stata's glm evaluates these likelihoods at phi = 1 (statsmodels' own Stata fixtures;
        # see tests/test_econ_glm_oracle.py), not at the Pearson scale statsmodels uses.
        expected_ll = ref.family.loglike(data[outcome].to_numpy(), ref.mu, scale=1)
    else:
        expected_ll = ref.llf
    assert result.metrics["log_likelihood"] == pytest.approx(expected_ll, rel=1e-8)
    assert result.metrics["aic"] == pytest.approx(-2 * expected_ll + 2 * k, rel=1e-8)
    assert result.metrics["bic"] == pytest.approx(-2 * expected_ll + k * np.log(n), rel=1e-8)
    assert result.metrics["aic_glm"] == pytest.approx((-2 * expected_ll + 2 * k) / n, rel=1e-8)
    assert result.metrics["bic_glm"] == pytest.approx(ref.deviance - (n - k) * np.log(n), rel=1e-8)
    assert list(result.metrics) == ["deviance", "pearson", "dispersion_deviance", "dispersion_pearson",
                                    "scale", "log_likelihood", "aic", "bic", "aic_glm", "bic_glm",
                                    "df_resid", "iterations"]
    assert result.extra["family"] == family and result.extra["canonical_link"] == glm_families.CANONICAL_LINK[family]
    assert result.provenance["stata_parity_validated"] is False
    assert len(result.predictions) == 400
    assert_allclose([p["fitted"] for p in result.predictions[:3]],
                    ref.fittedvalues.values[[p["row"] for p in result.predictions[:3]]], rtol=1e-6)


def test_gaussian_identity_reproduces_regress(data):
    result = oe.glm(data=data, y="normal", x=X)
    ols = oe.regress(data=data, y="normal", x=X)
    assert result.spec.options["family"] == "gaussian" and result.extra["link"] == "identity"
    assert_allclose(estimates(result), estimates(ols), rtol=1e-12)
    assert_allclose(errors(result), errors(ols), rtol=1e-11)
    assert result.metrics["log_likelihood"] == pytest.approx(ols.metrics["log_likelihood"], rel=1e-12)
    assert result.metrics["scale"] == pytest.approx(ols.metrics["rmse"] ** 2, rel=1e-12)
    assert result.extra["log_likelihood_dispersion_rule"] == "deviance / N (concentrated)"
    assert result.extra["log_likelihood_dispersion"] == pytest.approx(result.metrics["deviance"] / 500)
    wald = result.tests["model"]
    b, v = estimates(result)[1:], np.asarray(result.covariance_matrix)[1:, 1:]
    assert wald["statistic"] == pytest.approx(b @ np.linalg.solve(v, b), rel=1e-10)
    assert wald["df"] == 2 and wald["distribution"] == "chi2"
    assert wald["p_value"] == pytest.approx(stats.chi2.sf(wald["statistic"], 2), rel=1e-8)


def test_default_links_follow_stata(data):
    expected = {"gaussian": ("normal", "identity"), "binomial": ("binary", "logit"),
                "poisson": ("count", "log"), "gamma": ("gamma", "reciprocal"),
                "inverse_gaussian": ("wald", "inverse_squared"), "nbinomial": ("over", "log")}
    for family, (outcome, link) in expected.items():
        result = oe.glm(data=data, y=outcome, x=X, family=family)
        assert result.extra["link"] == link and "link" not in result.spec.options
        assert result.extra["scale_rule"] == ("fixed" if family in {"binomial", "poisson", "nbinomial"}
                                              else "Pearson chi2 / df")


def test_power_and_negative_binomial_links(data):
    ref = reference(data, "count", sm.families.Poisson(L.Sqrt()))
    result = oe.glm(data=data, y="count", x=X, family="poisson", link="power", power=0.5, optimizer="irls")
    assert_allclose(estimates(result), ref.params.values, rtol=1e-7)
    assert_allclose(errors(result), ref.bse.values, rtol=1e-7)
    assert result.extra["link"] == "power(0.5)" and result.spec.options["power"] == 0.5
    # power 0 is the log link and power 1 the identity (Stata).
    for power, link in ((0, "log"), (1, "identity"), (-1, "reciprocal"), (-2, "inverse_squared")):
        a = oe.glm(data=data, y="gamma", x=X, family="gamma", link="power", power=power)
        b = oe.glm(data=data, y="gamma", x=X, family="gamma", link=link)
        assert_allclose(estimates(a), estimates(b), rtol=1e-9)
        assert_allclose(errors(a), errors(b), rtol=1e-8)
    # Fixed overdispersion k with the log link and with the canonical nbinomial link.
    ref = reference(data, "over", sm.families.NegativeBinomial(alpha=0.7))
    result = oe.glm(data=data, y="over", x=X, family="nbinomial", dispersion=0.7)
    assert_allclose(estimates(result), ref.params.values, rtol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-9)
    assert result.metrics["deviance"] == pytest.approx(ref.deviance, rel=1e-9)
    assert result.extra["nbinomial_dispersion"] == 0.7 and "0.7 mu^2" in result.extra["variance_function"]
    ref = reference(data, "over", sm.families.NegativeBinomial(link=L.NegativeBinomial(alpha=0.7), alpha=0.7))
    for optimizer in ("ml", "irls"):
        canonical = oe.glm(data=data, y="over", x=X, family="nbinomial", link="nbinomial", dispersion=0.7,
                           optimizer=optimizer)
        assert_allclose(estimates(canonical), ref.params.values, rtol=1e-6)
        assert_allclose(errors(canonical), ref.bse.values, rtol=1e-6)      # canonical: OIM == EIM


def test_binomial_with_trials_matches_the_two_column_binomial(data):
    endog = np.column_stack([data.successes, data.trials - data.successes])
    for link, sm_link in (("logit", L.Logit()), ("probit", L.Probit()), ("cloglog", L.CLogLog())):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sm.GLM(endog, design(data), family=sm.families.Binomial(sm_link)).fit(
                tol=1e-13, tol_criterion="params", maxiter=500)
        result = oe.glm(data=data, y="successes", x=X, family="binomial", link=link, trials="trials",
                        optimizer="irls")
        assert_allclose(estimates(result), ref.params.values, rtol=1e-6)
        assert_allclose(errors(result), ref.bse.values, rtol=1e-6)
        assert result.metrics["deviance"] == pytest.approx(ref.deviance, rel=1e-9)
        assert result.metrics["pearson"] == pytest.approx(ref.pearson_chi2, rel=1e-8)
        assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-9)   # with ln C(n, y)
        # Fitted values are on the count scale, like the outcome.
        first = result.predictions[0]
        assert first["observed"] == data.successes[0]
        assert first["fitted"] == pytest.approx(ref.fittedvalues[0] * data.trials[0], rel=1e-6)
    # Trials of one reproduce the Bernoulli model exactly.
    ones = oe.glm(data=data.assign(one=1.0), y="binary", x=X, family="binomial", trials="one")
    bernoulli = oe.glm(data=data, y="binary", x=X, family="binomial")
    assert_allclose(estimates(ones), estimates(bernoulli), rtol=1e-12)
    assert ones.metrics["log_likelihood"] == pytest.approx(bernoulli.metrics["log_likelihood"], rel=1e-12)
    # The log link of the binomial family (risk ratios) against a brute-force optimum.
    risk = data.assign(rare=np.random.default_rng(3).binomial(1, np.exp(-1.5 + 0.2 * data.x1.clip(-2, 2))))
    result = oe.glm(data=risk, y="rare", x=["x1"], family="binomial", link="log")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.GLM(risk.rare, sm.add_constant(risk[["x1"]]), family=sm.families.Binomial(L.Log())).fit(
            tol=1e-13, tol_criterion="params", maxiter=500)
    assert_allclose(estimates(result), ref.params.values, rtol=1e-6)


@pytest.mark.parametrize("family,outcome,make", [
    ("poisson", "count", lambda: sm.families.Poisson()),
    ("gamma", "gamma", lambda: sm.families.Gamma(L.Log())),
    ("binomial", "binary", lambda: sm.families.Binomial(L.Probit())),
])
def test_robust_cluster_and_opg_covariances_follow_stata_ml_conventions(data, family, outcome, make):
    link = {"poisson": "log", "gamma": "log", "binomial": "probit"}[family]
    ref = reference(data, outcome, make())
    params = ref.params.values
    scores = ref.model.score_obs(params, scale=1)
    bread = np.linalg.inv(-ref.model.hessian(params, scale=1, observed=True))
    n = len(data)

    def meat(labels):
        sums = pd.DataFrame(scores).groupby(np.asarray(labels), sort=False).sum().to_numpy()
        return sums.T @ sums

    common = {"data": data, "y": outcome, "x": X, "family": family, "link": link}
    robust = oe.glm(**common, covariance="robust")
    assert_allclose(np.asarray(robust.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=1e-6)
    assert robust.inference["small_sample_correction"] == pytest.approx(n / (n - 1))
    assert robust.inference["correction"].startswith("Huber-White sandwich: N/(N-1)")
    cluster = oe.glm(**common, cluster="firm")
    assert cluster.spec.covariance == "cluster"
    assert_allclose(np.asarray(cluster.covariance_matrix), 50 / 49 * bread @ meat(data.firm) @ bread, rtol=1e-6)
    assert cluster.inference["cluster_count"] == 50 and cluster.inference["use_t"] is False
    twoway = oe.glm(**common, cluster=["firm", "state"])
    both = data.firm.astype(str) + "/" + data.state
    combined = meat(data.firm) + meat(data.state) - meat(both)
    assert_allclose(np.asarray(twoway.covariance_matrix), 8 / 7 * bread @ combined @ bread, rtol=1e-6)
    assert twoway.inference["cluster_counts"][:2] == [50, 8]
    opg = oe.glm(**common, covariance="opg")
    # The scores of the dispersion-phi likelihood are S/phi, so the OPG is phi^2 (S'S)^-1;
    # phi (S'S)^-1 is not consistent (see tests/test_econ_glm_oracle.py).
    assert_allclose(np.asarray(opg.covariance_matrix), ref.scale ** 2 * np.linalg.inv(scores.T @ scores),
                    rtol=1e-6)
    # Robust and cluster covariances are not multiplied by the dispersion.
    scaled = oe.glm(**common, covariance="robust", scale=7.5)
    assert_allclose(errors(scaled), errors(robust), rtol=1e-12)
    # With IRLS the bread is the expected information.
    expected = np.linalg.inv(-ref.model.hessian(params, scale=1, observed=False))
    irls = oe.glm(**common, covariance="robust", optimizer="irls")
    assert_allclose(np.asarray(irls.covariance_matrix), n / (n - 1) * expected @ scores.T @ scores @ expected,
                    rtol=2e-5)


def test_scale_options(data):
    base = {"data": data, "y": "count", "x": X, "family": "poisson"}
    unit = oe.glm(**base)
    n, k = 500, 3
    pearson, deviance = unit.metrics["pearson"], unit.metrics["deviance"]
    for scale, phi in (("x2", pearson / (n - k)), ("dev", deviance / (n - k)), (2.5, 2.5), (None, 1.0)):
        result = oe.glm(**base, scale=scale)
        assert result.metrics["scale"] == pytest.approx(phi, rel=1e-12)
        assert_allclose(errors(result), errors(unit) * np.sqrt(phi), rtol=1e-10)
        assert_allclose(estimates(result), estimates(unit), rtol=1e-12)
        assert result.metrics["log_likelihood"] == pytest.approx(unit.metrics["log_likelihood"])
    quasi = sm.GLM(data["count"], design(data), family=sm.families.Poisson()).fit(scale="X2")
    assert_allclose(errors(oe.glm(**base, scale="x2")), quasi.bse.values, rtol=1e-7)
    # Gamma: the scale multiplies the covariance only; the log likelihood stays at phi = 1 (Stata).
    for scale in ("x2", "dev", 0.3):
        ours = oe.glm(data=data, y="gamma", x=X, family="gamma", link="log", scale=scale)
        ref = sm.GLM(data.gamma, design(data), family=sm.families.Gamma(L.Log())).fit(
            scale={"x2": "X2", "dev": "dev"}.get(scale, scale), tol=1e-13, tol_criterion="params", maxiter=300)
        assert ours.metrics["scale"] == pytest.approx(ref.scale, rel=1e-7)
        assert ours.metrics["log_likelihood"] == pytest.approx(
            ref.family.loglike(data.gamma.to_numpy(), ref.mu, scale=1), rel=1e-9)
        assert_allclose(errors(ours), np.sqrt(np.diag(
            np.linalg.inv(-ref.model.hessian(ref.params.values, scale=1, observed=True)) * ref.scale)), rtol=1e-6)
    # Gaussian: the likelihood stays concentrated (regress's) whatever scale the covariance uses.
    fixed = oe.glm(data=data, y="normal", x=X, scale=2.0)
    rss = fixed.metrics["deviance"]
    assert fixed.metrics["scale"] == 2.0
    assert fixed.metrics["log_likelihood"] == pytest.approx(-0.5 * n * (1 + np.log(2 * np.pi * rss / n)))


def test_offset_and_exposure(data):
    ref = sm.GLM(data.rate, design(data), family=sm.families.Poisson(), exposure=data.expo).fit(tol=1e-13)
    by_exposure = oe.glm(data=data, y="rate", x=X, family="poisson", exposure="expo")
    by_offset = oe.glm(data=data.assign(lnexpo=np.log(data.expo)), y="rate", x=X, family="poisson",
                       offset="lnexpo")
    for result in (by_exposure, by_offset):
        assert_allclose(estimates(result), ref.params.values, rtol=1e-8)
        assert_allclose(errors(result), ref.bse.values, rtol=1e-8)
        assert result.metrics["deviance"] == pytest.approx(ref.deviance, rel=1e-10)
    assert by_exposure.spec.columns == {"exposure": "expo"}
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=data, y="rate", x=X, family="poisson", exposure="expo", offset="x2")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=data.assign(bad=data.expo - 1.0), y="rate", x=X, family="poisson", exposure="bad")
    assert caught.value.code == "invalid_exposure"


def test_frequency_weights_equal_duplicated_rows_for_every_covariance(data):
    duplicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for family, outcome, extra in (("poisson", "count", {}), ("gamma", "gamma", {"link": "log"}),
                                   ("binomial", "successes", {"trials": "trials", "link": "probit"})):
        for covariance in ("nonrobust", "opg", "robust", "cluster"):
            options = {"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance}
            weighted = oe.glm(data=data, y=outcome, x=X, family=family, weights="fw", weight_type="fweight",
                              **extra, **options)
            plain = oe.glm(data=duplicated, y=outcome, x=X, family=family, **extra, **options)
            assert weighted.nobs == plain.nobs == int(data.fw.sum())
            assert_allclose(estimates(weighted), estimates(plain), rtol=1e-9)
            assert_allclose(errors(weighted), errors(plain), rtol=1e-8)
            for name in ("deviance", "pearson", "scale", "log_likelihood", "aic", "bic", "df_resid"):
                assert weighted.metrics[name] == pytest.approx(plain.metrics[name], rel=1e-9), name
            assert weighted.tests["model"]["statistic"] == pytest.approx(plain.tests["model"]["statistic"],
                                                                         rel=1e-7)


def test_analytic_importance_and_sampling_weights(data):
    n = len(data)
    normalized = data.aw * n / data.aw.sum()
    for family, outcome, make, link in (("poisson", "count", sm.families.Poisson, "log"),
                                        ("gamma", "gamma", lambda: sm.families.Gamma(L.Log()), "log")):
        # aweights: rescaled to sum to N, then ordinary likelihood weights.
        ref = sm.GLM(data[outcome], design(data), family=make(), freq_weights=normalized).fit(
            tol=1e-13, tol_criterion="params", maxiter=300)
        result = oe.glm(data=data, y=outcome, x=X, family=family, link=link, weights="aw",
                        weight_type="aweight")
        assert_allclose(estimates(result), ref.params.values, rtol=1e-7)
        assert result.metrics["deviance"] == pytest.approx(ref.deviance, rel=1e-9)
        expected_ll = ref.llf if family == "poisson" else ref.family.loglike(   # gamma: phi = 1 (Stata)
            data[outcome].to_numpy(), ref.mu, freq_weights=normalized.to_numpy(), scale=1)
        assert result.metrics["log_likelihood"] == pytest.approx(expected_ll, rel=1e-8)
        assert result.metrics["scale"] == pytest.approx(ref.scale, rel=1e-7)
        hessian = ref.model.hessian(ref.params.values, scale=1, observed=True)
        assert_allclose(np.asarray(result.covariance_matrix), np.linalg.inv(-hessian) * ref.scale, rtol=1e-6)
        assert result.nobs == n
    # iweights are used as given: the information scales with them.
    raw = sm.GLM(data["count"], design(data), family=sm.families.Poisson(), freq_weights=data.aw).fit(tol=1e-13)
    importance = oe.glm(data=data, y="count", x=X, family="poisson", weights="aw", weight_type="iweight")
    assert_allclose(estimates(importance), raw.params.values, rtol=1e-8)
    assert_allclose(errors(importance), raw.bse.values, rtol=1e-8)
    assert importance.metrics["log_likelihood"] == pytest.approx(raw.llf, rel=1e-10) and importance.nobs == n
    # pweights: iweight point estimates and pseudolikelihood with the robust sandwich by default.
    sampling = oe.glm(data=data, y="count", x=X, family="poisson", weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), raw.params.values, rtol=1e-8)
    assert sampling.metrics["log_likelihood"] == pytest.approx(raw.llf, rel=1e-10)
    scores = raw.model.score_obs(raw.params.values, scale=1)        # already weighted
    bread = np.linalg.inv(-raw.model.hessian(raw.params.values, scale=1, observed=True))
    assert_allclose(np.asarray(sampling.covariance_matrix), n / (n - 1) * bread @ scores.T @ scores @ bread,
                    rtol=1e-6)
    rescaled = oe.glm(data=data.assign(aw=data.aw * 1000), y="count", x=X, family="poisson", weights="aw",
                      weight_type="pweight")
    assert_allclose(errors(rescaled), errors(sampling), rtol=1e-9)
    clustered = oe.glm(data=data, y="count", x=X, family="poisson", weights="aw", weight_type="pweight",
                       cluster="firm")
    sums = pd.DataFrame(scores).groupby(data.firm.to_numpy()).sum().to_numpy()
    assert_allclose(np.asarray(clustered.covariance_matrix), 50 / 49 * bread @ sums.T @ sums @ bread, rtol=1e-6)
    for covariance in ("nonrobust", "opg"):
        with pytest.raises(AnalysisError) as caught:
            oe.glm(data=data, y="count", x=X, family="poisson", weights="aw", weight_type="pweight",
                   covariance=covariance)
        assert caught.value.code == "unsupported_covariance"
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=data.assign(aw=data.aw - 1.0), y="count", x=X, family="poisson", weights="aw",
               weight_type="iweight")
    assert caught.value.code == "negative_weights"


DERIVATIVE_CASES = [(family, link) for family, links in glm_families.FAMILY_LINKS.items() for link in links]


@pytest.mark.parametrize("family,link", DERIVATIVE_CASES, ids=[f"{name}-{link}" for name, link in DERIVATIVE_CASES])
def test_analytic_derivatives_match_numerical_ones_for_every_family_and_link(family, link):
    rng = np.random.default_rng(11)
    n = 120
    x = torch.as_tensor(np.column_stack([np.ones(n), rng.normal(size=n), rng.uniform(-1, 1, size=n)]))
    prior = torch.as_tensor(rng.uniform(0.5, 2.0, size=n))
    offset = torch.as_tensor(rng.normal(scale=0.05, size=n))
    link_object = glm_families.make_link(link, power=0.5, k=0.7)
    trials = None
    if family == "binomial":
        trials = torch.as_tensor(rng.integers(1, 6, size=n).astype(float))
        y = torch.as_tensor(rng.binomial(trials.numpy().astype(int), 0.4) / trials.numpy())
        prior = prior * trials
        beta = {"log": [-1.0, 0.1, -0.1], "identity": [0.5, 0.05, 0.05]}.get(link, [-0.5, 0.4, -0.3])
    else:
        positive = np.exp(0.5 + 0.3 * x[:, 1].numpy())
        if family == "gaussian":
            y = torch.as_tensor(rng.normal(size=n) + 2.0)
        elif family in {"gamma", "inverse_gaussian"}:
            y = torch.as_tensor(rng.gamma(2.0, positive / 2.0))
        else:
            y = torch.as_tensor(rng.poisson(positive).astype(float))
        beta = {"identity": [3.0, 0.2, 0.1], "log": [0.5, 0.2, 0.1], "nbinomial": [-1.0, 0.1, -0.1]}.get(
            link, [-0.4, 0.2, 0.1] if link in {"logit", "probit", "cloglog", "loglog"} else [1.5, 0.1, 0.05])
    objective = GlmObjective(x, y, prior, offset, glm_families.make_family(family, k=0.7, trials=trials),
                             link_object, trials)
    theta = torch.tensor(beta, dtype=torch.float64)
    assert objective.state(theta) is not None
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-7, report
    assert report["hessian_max_rel_error"] < 1e-6, report
    assert report["hessian_asymmetry"] < 1e-9
    # The expected information equals minus the Hessian exactly for canonical links.
    state = objective.state(theta)
    if link == glm_families.CANONICAL_LINK[family] and family != "nbinomial":
        assert_allclose(objective.fisher_information(state).numpy(), -objective.hessian(state).numpy(),
                        rtol=1e-10)


def test_links_stay_accurate_in_the_tails():
    eta = torch.tensor([-40.0, -8.0, 0.3, 8.0, 40.0], dtype=torch.float64)
    for name in ("logit", "probit", "cloglog", "loglog"):
        link = glm_families.make_link(name)
        mu = link.inverse(eta)
        a, b = link.ratios(eta, mu)
        assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), name
        inner = slice(1, 4)
        m1 = link.derivative(eta, mu)[inner]
        assert_allclose((a * mu)[inner].numpy(), m1.numpy(), rtol=1e-9, atol=1e-300)
        assert_allclose((b * link.complement(eta, mu))[inner].numpy(), m1.numpy(), rtol=1e-9, atol=1e-300)
    # 1 - Phi(8) is Phi(-8), not zero.
    probit = glm_families.make_link("probit")
    assert float(probit.complement(eta, probit.inverse(eta))[3]) == pytest.approx(stats.norm.sf(8.0), rel=1e-12)
    mid = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    for name in glm_families.LINK_NAMES:
        link = glm_families.make_link(name, power=0.5, k=0.7)
        assert_allclose(link.inverse(link.link(mid)).numpy(), mid.numpy(), rtol=1e-12)


def test_separation_is_detected_instead_of_reporting_huge_coefficients(data):
    separated = data.assign(perfect=(data.x1 > 0.1).astype(float))
    for options in ({}, {"link": "probit"}, {"optimizer": "irls"}, {"link": "cloglog", "optimizer": "irls"}):
        with pytest.raises(AnalysisError) as caught:
            oe.glm(data=separated, y="perfect", x=X, family="binomial", **options)
        assert caught.value.code == "separation_detected", options
        assert "separation" in str(caught.value)
    # Quasi-complete separation: one category predicts the outcome perfectly.
    quasi = data.assign(flag=(np.arange(len(data)) < 40).astype(float))
    quasi.loc[quasi.flag == 1, "binary"] = 1.0
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=quasi, y="binary", x=["x1", "flag"], family="binomial")
    assert caught.value.code == "separation_detected"
    # A strong but imperfect predictor still converges, with large finite coefficients.
    rng = np.random.default_rng(8)
    strong = pd.DataFrame({"x": rng.normal(size=400)})
    strong["y"] = rng.binomial(1, 1 / (1 + np.exp(-6 * strong.x))).astype(float)
    result = oe.glm(data=strong, y="y", x=["x"], family="binomial")
    assert 3 < result.coefficients[1].estimate < 12


def test_categorical_predictors_collinearity_and_missing_policy(data):
    frame = data.assign(twice=2 * data.x1, one=1.0)
    result = oe.glm(data=frame, y="count", x=["x1", "sector", "twice", "one", "x2"], categorical=["sector"],
                    family="poisson")
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "sector[b]", "sector[c]", "x2"]
    assert result.provenance["omitted_terms"] == ["twice", "one"]
    assert "Omitted because of collinearity: twice, one." in result.warnings
    dummies = pd.get_dummies(frame.sector, drop_first=True, dtype=float, prefix="sector")
    ref = sm.GLM(frame["count"], sm.add_constant(pd.concat([frame[["x1"]], dummies, frame[["x2"]]], axis=1)),
                 family=sm.families.Poisson()).fit(tol=1e-13)
    assert_allclose(estimates(result), ref.params.values, rtol=1e-8)
    assert result.tests["model"]["df"] == 4
    no_constant = oe.glm(data=data, y="count", x=X, family="poisson", intercept=False)
    ref = sm.GLM(data["count"], data[X], family=sm.families.Poisson()).fit(tol=1e-13)
    assert [c.term for c in no_constant.coefficients] == X
    assert_allclose(estimates(no_constant), ref.params.values, rtol=1e-8)
    holes = data.copy()
    holes.loc[[3, 17], "x1"] = np.nan
    holes.loc[5, "count"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=holes, y="count", x=X, family="poisson")
    assert caught.value.code == "missing_values"
    dropped = oe.glm(data=holes, y="count", x=X, family="poisson", missing="drop")
    assert dropped.nobs == 497 and dropped.dropped_rows == 3 and 5 not in dropped.sample_positions
    assert "Excluded 3 observation(s)" in dropped.warnings[0]
    complete = oe.glm(data=holes.dropna(subset=["x1", "count"]), y="count", x=X, family="poisson")
    assert_allclose(estimates(dropped), estimates(complete), rtol=1e-12)


def test_error_codes(data):
    cases = [
        ({"x": "x1"}, "invalid_spec"),
        ({"family": "weibull"}, "invalid_spec"),
        ({"link": "cauchit"}, "invalid_spec"),
        ({"optimizer": "bfgs"}, "invalid_spec"),
        ({"covariance": "HC3"}, "invalid_spec"),
        ({"weights": "aw"}, "invalid_spec"),
        ({"max_iterations": 0}, "invalid_spec"),
        ({"x": ["nope"]}, "missing_columns"),
        ({"y": "state"}, "non_numeric_column"),
        ({"family": "poisson", "y": "normal"}, "invalid_count_outcome"),
        ({"family": "gamma", "y": "count"}, "invalid_positive_outcome"),
        ({"family": "inverse_gaussian", "y": "normal"}, "invalid_positive_outcome"),
        ({"family": "binomial", "y": "count"}, "invalid_binary_outcome"),
        ({"family": "binomial", "y": "count", "trials": "trials"}, "invalid_binomial_outcome"),
        ({"family": "binomial", "y": "successes", "trials": "expo"}, "invalid_trials"),
        ({"family": "poisson", "y": "count", "trials": "trials"}, "invalid_spec"),
        ({"family": "binomial", "y": "binary", "link": "reciprocal"}, "invalid_option"),
        ({"family": "poisson", "y": "count", "link": "logit"}, "invalid_option"),
        ({"family": "poisson", "y": "count", "link": "nbinomial"}, "invalid_option"),
        ({"link": "power"}, "invalid_option"),
        ({"power": 0.5}, "invalid_option"),
        ({"family": "poisson", "y": "count", "dispersion": 2.0}, "invalid_option"),
        ({"family": "nbinomial", "y": "over", "dispersion": -1.0}, "invalid_option"),
        ({"scale": "pearson"}, "invalid_option"),
        ({"scale": -2}, "invalid_option"),
        ({"tolerance": 2.0}, "invalid_option"),
    ]
    for arguments, code in cases:
        with pytest.raises(AnalysisError) as caught:
            oe.glm(**{"data": data, "y": "normal", "x": X, **arguments})
        assert caught.value.code == code, arguments
        assert str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=data.assign(zero=0.0), y="zero", x=X, family="binomial")
    assert caught.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=data.head(3), y="count", x=X, family="poisson")
    assert caught.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=data.assign(exact=1 + 2 * data.x1), y="exact", x=["x1"])
    assert caught.value.code == "perfect_fit"
    # A non-integer count is allowed with a recorded note (Stata).
    noted = oe.glm(data=data, y="gamma", x=X, family="poisson")
    assert any("non-integer" in warning for warning in noted.warnings)
    # The registry rejects undeclared fields before any data are read.
    for arguments in [{"options": {"family": "weibull"}}, {"covariance": "hac"}, {"columns": {"absorb": "firm"}},
                      {"cluster": ["firm", "state", "sector"]}, {"panel": "firm"}]:
        with pytest.raises(ValidationError):
            ModelSpec(estimator="glm", outcome="count", predictors=X, **arguments)
    # A nonconverged fit is an error, never a partial result.
    with pytest.raises(AnalysisError) as caught:
        oe.glm(data=data, y="gamma", x=X, family="gamma", link="identity", optimizer="irls", max_iterations=1)
    assert caught.value.code == "nonconvergence"


def test_result_round_trip_summary_and_latex(data):
    result = oe.glm(data=data, y="gamma", x=["x1", "x2", "sector"], categorical=["sector"], family="gamma",
                    link="log", cluster="firm")
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored == result
    payload = json.loads(result.model_dump_json())
    assert payload["tests"]["model"]["distribution"] == "chi2" and payload["extra"]["family"] == "gamma"
    text = result.summary()
    assert "Generalized linear model — gamma" in text and "Covariance: cluster" in text
    assert "Wald chi2 test of the slopes: chi2(4)" in text and "deviance:" in text and "sector[b]" in text
    assert "  z  " in text
    latex = str(result.to_latex())
    assert "sector[b]" in latex.replace(r"\_", "_") and "Log likelihood" in latex
    spec = ModelSpec(estimator="glm", outcome="count", predictors=X, covariance="robust",
                     options={"family": "poisson"})
    assert fit(spec, data=data).spec == spec
    assert ModelSpec.model_validate(json.loads(result.spec.model_dump_json())) == result.spec
    assert result.provenance["optimizer"]["converged"] is True
    assert result.provenance["solver"] == "newton_observed_hessian"


def test_glm_handles_a_large_sample_in_linear_time():
    rng = np.random.default_rng(5)
    n = 200_000
    frame = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    frame["g"] = rng.integers(0, 4000, size=n)
    frame["y"] = rng.gamma(2.0, np.exp(0.3 + 0.2 * frame.x0 - 0.1 * frame.x1) / 2.0)
    columns = [f"x{i}" for i in range(6)]
    result = oe.glm(data=frame, y="y", x=columns, family="gamma", link="log", cluster="g")
    assert result.nobs == n and len(result.predictions) == 400
    assert abs(result.coefficients[1].estimate - 0.2) < 0.02
    irls = oe.glm(data=frame, y="y", x=columns, family="gamma", link="log", optimizer="irls")
    assert_allclose(estimates(irls), estimates(result), rtol=1e-5, atol=1e-7)
