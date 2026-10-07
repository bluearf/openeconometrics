"""Ordered and multinomial models against independent oracles.

ologit / oprobit are compared with statsmodels' OrderedModel (delta method for
its threshold parameterization), mlogit with statsmodels' MNLogit, and every
covariance estimator, weight type and offset with explicit NumPy algebra on
independently written likelihoods (numerical scores and Hessians, brute-force
SciPy maximization). Analytic derivatives are checked against numerical ones.
"""

import json
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import optimize as sopt
from scipy import special, stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics import registry
from openecon.econometrics.discrete import ESTIMATORS, EXPORTS
from openecon.econometrics.discrete.kernels import MultinomialObjective, OrderedObjective
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec, ResultBundle

X = ["x1", "x2"]
LINKS = {"ologit": "logit", "oprobit": "probit"}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20262)
    n = 600
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "firm": np.repeat(np.arange(60), 10), "state": rng.integers(0, 12, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "off": rng.normal(scale=0.4, size=n),
    })
    latent = 0.8 * frame.x1 - 0.5 * frame.x2 + rng.logistic(size=n)
    frame["rating"] = np.digitize(latent, [-1.0, 0.3, 1.5])
    frame["grade"] = pd.Categorical.from_codes(
        frame.rating.to_numpy(), categories=["poor", "fair", "good", "best"], ordered=True)
    utility = np.column_stack([
        np.zeros(n), 0.4 + 0.8 * frame.x1 - 0.3 * frame.x2,
        -0.2 - 0.5 * frame.x1 + 0.6 * frame.x2, 0.9 * frame.x2])
    frame["choice"] = (utility + rng.gumbel(size=(n, 4))).argmax(axis=1)
    frame["mode"] = np.array(["bus", "car", "rail", "walk"])[frame.choice]
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


# ---- independent likelihoods ---------------------------------------------------------


def _cdf(value, link):
    return special.ndtr(value) if link == "probit" else 1 / (1 + np.exp(-value))


def ordered_loglik_obs(theta, x, codes, link, offset=None):
    """Per-observation ordered log likelihood (complex-step safe)."""
    k = x.shape[1]
    categories = len(theta) - k + 1
    eta = x @ theta[:k] + (0 if offset is None else offset)
    cuts = theta[k:]
    upper = np.where(codes < categories - 1, _cdf(cuts[np.minimum(codes, categories - 2)] - eta, link), 1)
    lower = np.where(codes > 0, _cdf(cuts[np.maximum(codes - 1, 0)] - eta, link), 0)
    return np.log(upper - lower)


def mlogit_loglik_obs(theta, x, codes, base, categories):
    """Coefficients stacked by non-base category, as OpenEcon reports them."""
    k = x.shape[1]
    eta = np.zeros((len(x), categories), dtype=theta.dtype)
    others = [j for j in range(categories) if j != base]
    eta[:, others] = x @ theta.reshape(len(others), k).T
    eta = eta - eta.real.max(axis=1, keepdims=True)
    return eta[np.arange(len(x)), codes] - np.log(np.exp(eta).sum(axis=1))


def ml_covariances(loglik_obs, theta, weights=None, groups=None, nobs=None):
    """OIM, OPG, robust and cluster covariances of an independently written likelihood.

    Scores are complex-step derivatives (exact to rounding); the Hessian is the
    central difference of the complex-step gradient.
    """
    w = np.ones(len(loglik_obs(theta))) if weights is None else weights
    hessian = approx_fprime(theta, lambda t: approx_fprime_cs(t, lambda u: (w * loglik_obs(u)).sum()),
                            centered=True)
    hessian = (hessian + hessian.T) / 2
    scores = approx_fprime_cs(theta, loglik_obs) * w[:, None]
    n = len(w) if nobs is None else nobs
    bread = np.linalg.inv(-hessian)
    out = {"nonrobust": bread, "opg": np.linalg.inv(scores.T @ scores),
           "robust": n / (n - 1) * bread @ scores.T @ scores @ bread}
    if groups is not None:
        labels, index = np.unique(groups, return_inverse=True)
        sums = np.zeros((len(labels), len(theta)))
        np.add.at(sums, index, scores)
        g = len(labels)
        out["cluster"] = g / (g - 1) * bread @ sums.T @ sums @ bread
    return out


def brute_force(loglik_obs, start, weights=None):
    """Maximize an independently written likelihood with SciPy's BFGS (numerical gradient)."""
    def negative(t):
        with np.errstate(all="ignore"):
            values = loglik_obs(t)
            total = -(values if weights is None else weights * values).sum()
        return total if np.isfinite(total) else 1e12

    return sopt.minimize(negative, start, method="BFGS", options={"gtol": 1e-9})


# ---- manifest -----------------------------------------------------------------------


def test_manifest_registers_the_family_and_public_functions():
    names = ["ologit", "oprobit", "mlogit", "clogit", "hetprobit", "biprobit"]
    assert [info.name for info in ESTIMATORS] == names and EXPORTS == {}
    assert all(info.family == "discrete" and info.inference == "z" for info in ESTIMATORS)
    for name in names:
        assert callable(getattr(oe, name)) and name in dir(oe)
        info = registry.get(name)
        assert info.function == name and info.default_covariance == "nonrobust"
        assert info.covariances == ("nonrobust", "opg", "robust", "cluster")
    assert registry.get("ologit").intercept == "never" == registry.get("clogit").intercept
    assert registry.get("clogit").role("group").required
    assert registry.get("clogit").weights == ("fweight", "iweight", "pweight")
    assert registry.get("hetprobit").role("het").many and registry.get("hetprobit").role("het").required
    assert registry.get("biprobit").role("outcome2").required
    assert not registry.get("biprobit").role("predictors2").required
    record = oe.capabilities()["estimators"]
    assert record["mlogit"]["options"]["base"]["type"] == "json"
    assert record["ologit"]["stata"] == ["ologit"] and record["ologit"]["intercept"] is False
    assert record["clogit"]["stata"] == ["clogit", "xtlogit, fe"]
    assert record["biprobit"]["binary_outcome"] is True
    assert "discrete" in oe.capabilities()["families"]
    # The manifest must stay importable without the tensor runtime.
    code = ("import sys, openecon.econometrics.discrete as d; "
            "assert 'torch' not in sys.modules and 'pandas' not in sys.modules; "
            "print(len(d.ESTIMATORS))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "6"


# ---- ordered models -------------------------------------------------------------------


@pytest.mark.parametrize("command", ["ologit", "oprobit"])
def test_ordered_matches_statsmodels_ordered_model(data, command):
    result = getattr(oe, command)(data=data, y="rating", x=X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = OrderedModel(data.rating, data[X], distr=LINKS[command]).fit(
            method="newton", disp=0, tol=1e-13, maxiter=200)
    cuts = ref.model.transform_threshold_params(ref.params.values)[1:-1]
    assert [c.term for c in result.coefficients] == ["x1", "x2", "/cut1", "/cut2", "/cut3"]
    assert [c.equation for c in result.coefficients] == ["rating", "rating", None, None, None]
    assert_allclose(estimates(result), np.r_[ref.params.values[:2], cuts], rtol=1e-8, atol=1e-9)
    # statsmodels estimates cut1 and log increments: delta method to OpenEcon's cutpoints.
    jacobian = np.eye(5)
    increments = np.exp(ref.params.values[3:])
    jacobian[3, 2:4] = [1, increments[0]]
    jacobian[4, 2:5] = [1, increments[0], increments[1]]
    # (statsmodels differentiates numerically, hence the tolerance.)
    assert_allclose(covariance(result), jacobian @ ref.cov_params().values @ jacobian.T,
                    rtol=2e-5, atol=1e-8)
    n = len(data)
    counts = np.bincount(data.rating)
    null = float((counts * np.log(counts / n)).sum())
    assert result.nobs == n and result.metrics["n_categories"] == 4
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-12)
    assert_allclose(result.extra["null_log_likelihood"], null, rtol=1e-12)
    assert_allclose(result.metrics["pseudo_r_squared"], 1 - ref.llf / null, rtol=1e-10)
    assert_allclose(result.metrics["aic"], -2 * ref.llf + 2 * 5, rtol=1e-12)
    assert_allclose(result.metrics["bic"], -2 * ref.llf + np.log(n) * 5, rtol=1e-12)
    model = result.tests["model"]
    assert model["distribution"] == "chi2" and model["df"] == 2 and "LR" in model["label"]
    assert_allclose(model["statistic"], 2 * (ref.llf - null), rtol=1e-9)
    assert_allclose(model["p_value"], stats.chi2.sf(model["statistic"], 2), rtol=1e-8)
    assert result.extra["categories"] == [0, 1, 2, 3]
    assert_allclose(result.extra["cutpoints"], cuts, rtol=1e-8)
    assert result.extra["category_counts"] == counts.tolist()
    assert result.inference["use_t"] is False and result.inference["correction"] == "observed information"
    z = estimates(result) / errors(result)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.norm.sf(np.abs(z)), rtol=1e-8)
    assert result.predictions == [] and result.provenance["stata_parity_validated"] is False


@pytest.mark.parametrize("command", ["ologit", "oprobit"])
def test_ordered_covariances_follow_stata_ml_conventions(data, command):
    x, codes = data[X].to_numpy(), data.rating.to_numpy()
    base = getattr(oe, command)(data=data, y="rating", x=X)
    theta = estimates(base)
    expected = ml_covariances(lambda t: ordered_loglik_obs(t, x, codes, LINKS[command]), theta,
                              groups=data.firm.to_numpy())
    assert_allclose(covariance(base), expected["nonrobust"], rtol=2e-6, atol=1e-9)
    for kind in ("opg", "robust", "cluster"):
        result = getattr(oe, command)(data=data, y="rating", x=X, covariance=kind,
                                      cluster="firm" if kind == "cluster" else None)
        assert_allclose(estimates(result), theta, rtol=1e-12)
        assert_allclose(covariance(result), expected[kind], rtol=2e-6, atol=1e-9)
        model = result.tests["model"]
        slopes = theta[:2]
        wald = slopes @ np.linalg.solve(expected[kind][:2, :2], slopes)
        assert "Wald" in model["label"] and model["df"] == 2
        assert_allclose(model["statistic"], wald, rtol=1e-5)
    assert result.inference["cluster_count"] == 60 and "G/(G-1)" in result.inference["correction"]
    assert_allclose(result.inference["small_sample_correction"], 60 / 59)
    implied = oe.ologit(data=data, y="rating", x=X, cluster="firm")
    assert implied.spec.covariance == "cluster"
    twoway = oe.ologit(data=data, y="rating", x=X, cluster=["firm", "state"])
    assert twoway.inference["cluster_columns"] == ["firm", "state"]
    assert np.isfinite(errors(twoway)).all()


def test_ordered_weights_follow_stata_semantics(data):
    x, codes = data[X].to_numpy(), data.rating.to_numpy()
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "opg", "robust", "cluster"):
        extra = {"covariance": kind, "cluster": "firm" if kind == "cluster" else None}
        weighted = oe.oprobit(data=data, y="rating", x=X, weights="fw", weight_type="fweight", **extra)
        duplicated = oe.oprobit(data=repeated, y="rating", x=X, **extra)
        assert weighted.nobs == len(repeated) == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(duplicated), rtol=1e-9, atol=1e-10)
        assert_allclose(covariance(weighted), covariance(duplicated), rtol=1e-8, atol=1e-11)
        for name in ("log_likelihood", "pseudo_r_squared", "aic", "bic"):
            assert_allclose(weighted.metrics[name], duplicated.metrics[name], rtol=1e-10)
        assert_allclose(weighted.tests["model"]["statistic"], duplicated.tests["model"]["statistic"],
                        rtol=1e-8)
    # aweights are rescaled to sum to N; iweights are used as given.
    scaled = data.aw.to_numpy() * len(data) / data.aw.sum()

    def loglik(t):
        return ordered_loglik_obs(t, x, codes, "logit")

    analytic = oe.ologit(data=data, y="rating", x=X, weights="aw", weight_type="aweight")
    brute = brute_force(loglik, estimates(analytic) * 0.9, scaled)
    assert_allclose(estimates(analytic), brute.x, rtol=2e-5, atol=2e-6)
    expected = ml_covariances(loglik, estimates(analytic), weights=scaled)
    assert_allclose(covariance(analytic), expected["nonrobust"], rtol=2e-6, atol=1e-9)
    assert_allclose(analytic.metrics["log_likelihood"], (scaled * loglik(estimates(analytic))).sum(),
                    rtol=1e-12)
    assert analytic.nobs == len(data)
    importance = oe.ologit(data=data, y="rating", x=X, weights="aw", weight_type="iweight")
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-8)
    assert_allclose(covariance(importance) * data.aw.sum() / len(data), covariance(analytic), rtol=1e-8)
    # pweights: same point estimates, robust covariance by default, scale invariant.
    sampling = oe.ologit(data=data, y="rating", x=X, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    raw = ml_covariances(loglik, estimates(sampling), weights=data.aw.to_numpy())
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-8)
    assert_allclose(covariance(sampling), raw["robust"], rtol=2e-6, atol=1e-9)
    doubled = oe.ologit(data=data.assign(aw=2 * data.aw), y="rating", x=X, weights="aw",
                        weight_type="pweight")
    assert_allclose(covariance(doubled), covariance(sampling), rtol=1e-9)
    with pytest.raises(AnalysisError) as caught:
        fit(ModelSpec(estimator="ologit", outcome="rating", predictors=X, intercept=False,
                      covariance="nonrobust", weights="aw", weight_type="pweight"), data=data)
    assert caught.value.code == "unsupported_covariance"


def test_ordered_offset_and_null_model_with_offset(data):
    x, codes, offset = data[X].to_numpy(), data.rating.to_numpy(), data.off.to_numpy()
    result = oe.oprobit(data=data, y="rating", x=X, offset="off")
    brute = brute_force(lambda t: ordered_loglik_obs(t, x, codes, "probit", offset),
                        estimates(result) * 0.9)
    assert_allclose(estimates(result), brute.x, rtol=2e-5, atol=2e-6)
    assert_allclose(result.metrics["log_likelihood"], -brute.fun, rtol=1e-10)
    empty = np.empty((len(data), 0))
    null = brute_force(lambda t: ordered_loglik_obs(t, empty, codes, "probit", offset),
                       np.array([-0.6, 0.2, 0.9]))
    assert_allclose(result.extra["null_log_likelihood"], -null.fun, rtol=1e-9)
    assert_allclose(result.tests["model"]["statistic"], 2 * (null.fun - brute.fun), rtol=1e-6)
    assert not np.allclose(estimates(result), estimates(oe.oprobit(data=data, y="rating", x=X)))


def test_ordered_categories_come_from_values_or_an_ordered_categorical(data):
    numeric = oe.ologit(data=data, y="rating", x=X)
    labelled = oe.ologit(data=data, y="grade", x=X)
    assert labelled.extra["categories"] == ["poor", "fair", "good", "best"]
    assert labelled.extra["category_order"] == "ordered Categorical"
    assert_allclose(estimates(labelled), estimates(numeric), rtol=1e-12)
    # Only observed categories count, and unequally spaced numeric values keep their order.
    spaced = data.assign(rating=data.rating.map({0: -3.5, 1: 0.0, 2: 10.0, 3: 99.0}))
    result = oe.ologit(data=spaced, y="rating", x=X)
    assert result.extra["categories"] == [-3.5, 0, 10, 99]
    assert_allclose(estimates(result), estimates(numeric), rtol=1e-12)
    subset = data[data.grade != "fair"]
    assert oe.ologit(data=subset, y="grade", x=X).extra["categories"] == ["poor", "good", "best"]
    # Two categories: the ordered logit is the binary logit with cut1 = -constant.
    binary = data.assign(high=(data.rating >= 2) * 1.0)
    two = oe.ologit(data=binary, y="high", x=X)
    ref = sm.Logit(binary.high, sm.add_constant(binary[X])).fit(disp=0, tol=1e-12)
    assert_allclose(estimates(two), np.r_[ref.params.values[1:], -ref.params.values[0]], rtol=1e-8)
    assert_allclose(errors(two), np.r_[ref.bse.values[1:], ref.bse.values[0]], rtol=1e-7)
    assert_allclose(two.metrics["pseudo_r_squared"], ref.prsquared, rtol=1e-9)
    for column, code in (("mode", "invalid_ordered_outcome"),):
        with pytest.raises(AnalysisError) as caught:
            oe.ologit(data=data, y=column, x=X)
        assert caught.value.code == code
    unordered = data.assign(grade=data.grade.cat.as_unordered())
    with pytest.raises(AnalysisError) as caught:
        oe.oprobit(data=unordered, y="grade", x=X)
    assert caught.value.code == "invalid_ordered_outcome" and "mlogit" in str(caught.value)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_ordered_analytic_derivatives_match_numerical_ones(data, link):
    x = torch.tensor(data[X].to_numpy())
    codes = torch.tensor(data.rating.to_numpy())
    objective = OrderedObjective(x, codes, 4, torch.tensor(data.aw.to_numpy()),
                                 torch.tensor(data.off.to_numpy()), link)
    theta = torch.tensor([0.3, -0.2, -1.2, 0.1, 1.0], dtype=torch.float64)
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    assert report["hessian_asymmetry"] < 1e-10
    value, gradient, _ = objective(theta)
    rows = objective.score_rows(theta)
    assert_allclose((rows * objective.w[:, None]).sum(dim=0).numpy(), gradient.numpy(), atol=1e-9)
    expected = ordered_loglik_obs(theta.numpy(), data[X].to_numpy(), data.rating.to_numpy(), link,
                                  data.off.to_numpy())
    assert_allclose(float(value), (data.aw.to_numpy() * expected).sum(), rtol=1e-12)
    assert_allclose(float(objective.value(theta)), float(value), rtol=1e-14)
    # A trial point with unordered cutpoints is rejected with a non-finite value.
    unordered = torch.tensor([0.3, -0.2, 0.5, 0.1, 1.0], dtype=torch.float64)
    assert not np.isfinite(float(objective.value(unordered)))
    assert not np.isfinite(float(objective(unordered)[0]))
    # Far tails: interval probabilities are formed where both tails are small.
    extreme = torch.tensor([[40.0, 0.0], [-40.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    tail = OrderedObjective(extreme, torch.tensor([2, 0, 1]), 3, torch.ones(3, dtype=torch.float64),
                            None, link)
    point = torch.tensor([1.0, 0.0, -0.5, 0.5], dtype=torch.float64)
    value, gradient, hessian = tail(point)
    assert torch.isfinite(value) and torch.isfinite(gradient).all() and torch.isfinite(hessian).all()
    assert float(tail.probabilities(point)[0]) == 1.0
    wrong = OrderedObjective(extreme, torch.tensor([0, 2, 1]), 3, torch.ones(3, dtype=torch.float64),
                             None, link)
    value, gradient, hessian = wrong(point)
    assert torch.isfinite(value) and float(value) < -30
    assert torch.isfinite(gradient).all() and torch.isfinite(hessian).all()


def test_ordered_collinearity_missing_values_and_categorical_regressors(data):
    frame = data.assign(one=1.0, twin=2 * data.x1)
    result = oe.ologit(data=frame, y="rating", x=["x1", "one", "x2", "twin"])
    assert [c.term for c in result.coefficients][:2] == ["x1", "x2"]
    assert result.provenance["omitted_terms"] == ["one", "twin"]
    assert any("collinearity" in warning and "one, twin" in warning for warning in result.warnings)
    assert_allclose(estimates(result), estimates(oe.ologit(data=data, y="rating", x=X)), rtol=1e-10)
    expanded = oe.oprobit(data=data, y="rating", x=["x1", "sector"], categorical=["sector"])
    assert [c.term for c in expanded.coefficients][:3] == ["x1", "sector[b]", "sector[c]"]
    dummies = data.assign(b=(data.sector == "b") * 1.0, c=(data.sector == "c") * 1.0)
    assert_allclose(estimates(expanded), estimates(oe.oprobit(data=dummies, y="rating", x=["x1", "b", "c"])),
                    rtol=1e-10)
    holes = data.copy()
    holes.loc[[3, 40], "x1"] = np.nan
    holes.loc[7, "rating"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=holes, y="rating", x=X)
    assert caught.value.code == "missing_values"
    dropped = oe.ologit(data=holes, y="rating", x=X, missing="drop")
    complete = oe.ologit(data=holes.dropna(subset=["x1", "rating"]), y="rating", x=X)
    assert dropped.nobs == len(data) - 3 and dropped.dropped_rows == 3
    assert 3 not in dropped.sample_positions and 7 not in dropped.sample_positions
    assert_allclose(estimates(dropped), estimates(complete), rtol=1e-12)


def test_ordered_separation_and_error_codes(data):
    separated = data.assign(top=(data.rating == 3) * 1.0)
    for command in (oe.ologit, oe.oprobit):
        with pytest.raises(AnalysisError) as caught:
            command(data=separated, y="rating", x=["x1", "top"])
        assert caught.value.code == "separation_detected" and "predicted perfectly" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=data.assign(rating=1), y="rating", x=X)
    assert caught.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=data.assign(rating=data.x1), y="rating", x=["x2"])
    assert caught.value.code == "too_many_categories"
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=data, y="rating", x="x1")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=data, y="rating", x=X, covariance="HC3")
    assert caught.value.code == "invalid_spec" and "covariance" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=data.assign(aw=-data.aw), y="rating", x=X, weights="aw", weight_type="aweight")
    assert caught.value.code == "negative_weights"
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=data, y="rating", x=X, weights="aw", weight_type="fweight")
    assert caught.value.code == "noninteger_frequency_weights"
    with pytest.raises(AnalysisError) as caught:
        oe.ologit(data=data, y="rating", x=["x1", "nope"])
    assert caught.value.code == "missing_columns"
    for arguments in [{"intercept": True}, {"options": {"base": 1}}, {"panel": "firm"},
                      {"columns": {"group": "firm"}}, {"cluster": ["firm", "state", "sector"]}]:
        fields = {"intercept": False, **arguments}
        with pytest.raises(ValidationError):
            ModelSpec(estimator="ologit", outcome="rating", predictors=X, **fields)


# ---- multinomial logit ------------------------------------------------------------------


def test_mlogit_matches_statsmodels_mnlogit(data):
    result = oe.mlogit(data=data, y="choice", x=X, base=0)
    ref = sm.MNLogit(data.choice, sm.add_constant(data[X])).fit(method="newton", disp=0, tol=1e-13)
    terms = [f"{j}:{term}" for j in (1, 2, 3) for term in ("Intercept", "x1", "x2")]
    assert [c.term for c in result.coefficients] == terms
    assert [c.equation for c in result.coefficients] == [t.split(":")[0] for t in terms]
    assert_allclose(estimates(result).reshape(3, 3), ref.params.values.T, rtol=1e-9, atol=1e-10)
    assert_allclose(errors(result).reshape(3, 3), ref.bse.values.T, rtol=1e-8)
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-12)
    assert_allclose(result.metrics["pseudo_r_squared"], ref.prsquared, rtol=1e-8)
    assert_allclose(result.metrics["aic"], ref.aic, rtol=1e-12)
    assert_allclose(result.metrics["bic"], ref.bic, rtol=1e-12)
    model = result.tests["model"]
    assert model["df"] == 6 and "LR" in model["label"]
    assert_allclose(model["statistic"], ref.llr, rtol=1e-9)
    assert_allclose(model["p_value"], ref.llr_pvalue, rtol=1e-7)
    assert result.extra["categories"] == [0, 1, 2, 3] and result.extra["base"] == 0
    assert result.extra["equations"] == ["1", "2", "3"] and result.metrics["n_categories"] == 4
    assert result.title == "Multinomial logistic regression" and result.predictions == []


def test_mlogit_base_category_rules(data):
    counts = data.choice.value_counts()
    most = int(counts.idxmax())
    default = oe.mlogit(data=data, y="choice", x=X)
    assert default.extra["base"] == most and "most frequent" in default.extra["base_rule"]
    explicit = oe.mlogit(data=data, y="choice", x=X, base=most)
    assert_allclose(estimates(default), estimates(explicit), rtol=1e-13)
    # Changing the base is a reparameterization: b_j(base B) = b_j(base 0) - b_B(base 0).
    zero = estimates(oe.mlogit(data=data, y="choice", x=X, base=0)).reshape(3, 3)
    full = np.vstack([np.zeros(3), zero])
    other = [j for j in range(4) if j != most]
    assert_allclose(estimates(default).reshape(3, 3), full[other] - full[most], rtol=1e-8, atol=1e-10)
    assert_allclose(default.metrics["log_likelihood"], explicit.metrics["log_likelihood"], rtol=1e-13)
    # String labels and a Categorical give the same model with labelled equations.
    labelled = oe.mlogit(data=data, y="mode", x=X, base="bus")
    assert labelled.extra["categories"] == ["bus", "car", "rail", "walk"]
    assert [c.term for c in labelled.coefficients][:3] == ["car:Intercept", "car:x1", "car:x2"]
    assert_allclose(estimates(labelled).reshape(3, 3), zero, rtol=1e-12)
    ordered = data.assign(mode=pd.Categorical(data["mode"], categories=["walk", "rail", "car", "bus"]))
    reordered = oe.mlogit(data=ordered, y="mode", x=X, base="bus")
    assert reordered.extra["equations"] == ["walk", "rail", "car"]
    assert_allclose(estimates(reordered).reshape(3, 3), zero[::-1], rtol=1e-12)
    # Weighted frequencies choose the default base.
    heavy = data.assign(w=np.where(data.choice == 2, 50.0, 1.0))
    assert oe.mlogit(data=heavy, y="choice", x=X, weights="w", weight_type="fweight").extra["base"] == 2
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=data, y="choice", x=X, base=9)
    assert caught.value.code == "invalid_base_category" and "0, 1, 2, 3" in str(caught.value)


def test_mlogit_covariances_and_weights(data):
    x = sm.add_constant(data[X]).to_numpy()
    codes = data.choice.to_numpy()

    def loglik(t):
        return mlogit_loglik_obs(t, x, codes, 0, 4)

    base = oe.mlogit(data=data, y="choice", x=X, base=0)
    theta = estimates(base)
    expected = ml_covariances(loglik, theta, groups=data.firm.to_numpy())
    assert_allclose(covariance(base), expected["nonrobust"], rtol=2e-6, atol=1e-9)
    slopes = [1, 2, 4, 5, 7, 8]
    for kind in ("opg", "robust", "cluster"):
        result = oe.mlogit(data=data, y="choice", x=X, base=0, covariance=kind,
                           cluster="firm" if kind == "cluster" else None)
        assert_allclose(covariance(result), expected[kind], rtol=2e-6, atol=1e-9)
        wald = theta[slopes] @ np.linalg.solve(expected[kind][np.ix_(slopes, slopes)], theta[slopes])
        assert result.tests["model"]["df"] == 6 and "Wald" in result.tests["model"]["label"]
        assert_allclose(result.tests["model"]["statistic"], wald, rtol=1e-5)
    hc0 = sm.MNLogit(data.choice, x).fit(method="newton", disp=0, tol=1e-13, cov_type="HC0")
    robust = oe.mlogit(data=data, y="choice", x=X, base=0, covariance="robust")
    n = len(data)
    assert_allclose(errors(robust).reshape(3, 3), np.asarray(hc0.bse).T * np.sqrt(n / (n - 1)),
                    rtol=1e-7)
    # fweights replicate rows under every covariance estimator.
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "opg", "robust", "cluster"):
        extra = {"covariance": kind, "cluster": "firm" if kind == "cluster" else None, "base": 0}
        weighted = oe.mlogit(data=data, y="choice", x=X, weights="fw", weight_type="fweight", **extra)
        duplicated = oe.mlogit(data=repeated, y="choice", x=X, **extra)
        assert weighted.nobs == len(repeated)
        assert_allclose(estimates(weighted), estimates(duplicated), rtol=1e-9, atol=1e-10)
        assert_allclose(covariance(weighted), covariance(duplicated), rtol=1e-8, atol=1e-11)
        assert_allclose(weighted.metrics["bic"], duplicated.metrics["bic"], rtol=1e-11)
        assert_allclose(weighted.tests["model"]["statistic"], duplicated.tests["model"]["statistic"],
                        rtol=1e-8)
    # aweights (rescaled to N), iweights (as given) and pweights (robust).
    scaled = data.aw.to_numpy() * n / data.aw.sum()
    analytic = oe.mlogit(data=data, y="choice", x=X, base=0, weights="aw", weight_type="aweight")
    brute = brute_force(loglik, estimates(analytic) * 0.9, scaled)
    assert_allclose(estimates(analytic), brute.x, rtol=2e-5, atol=2e-6)
    weighted_cov = ml_covariances(loglik, estimates(analytic), weights=scaled)
    assert_allclose(covariance(analytic), weighted_cov["nonrobust"], rtol=2e-6, atol=1e-9)
    shares = np.bincount(codes, weights=scaled) / n
    assert_allclose(analytic.extra["null_log_likelihood"], n * (shares * np.log(shares)).sum(), rtol=1e-12)
    importance = oe.mlogit(data=data, y="choice", x=X, base=0, weights="aw", weight_type="iweight")
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-8)
    assert_allclose(importance.metrics["log_likelihood"] * n / data.aw.sum(),
                    analytic.metrics["log_likelihood"], rtol=1e-10)
    sampling = oe.mlogit(data=data, y="choice", x=X, base=0, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    raw = ml_covariances(loglik, estimates(sampling), weights=data.aw.to_numpy())
    assert_allclose(covariance(sampling), raw["robust"], rtol=2e-6, atol=1e-9)


def test_mlogit_without_constant_collinearity_and_missing_values(data):
    x = data[X].to_numpy()
    result = oe.mlogit(data=data, y="choice", x=X, base=0, intercept=False)
    brute = brute_force(lambda t: mlogit_loglik_obs(t, x, data.choice.to_numpy(), 0, 4), np.zeros(6))
    assert_allclose(estimates(result), brute.x, rtol=2e-5, atol=2e-6)
    null = len(data) * np.log(1 / 4)
    assert_allclose(result.extra["null_log_likelihood"], null, rtol=1e-13)
    assert result.tests["model"]["df"] == 6 and "equal-probability" in result.tests["model"]["label"]
    assert_allclose(result.tests["model"]["statistic"], 2 * (-brute.fun - null), rtol=1e-8)
    frame = data.assign(one=1.0, twin=data.x1 - data.x2)
    omitted = oe.mlogit(data=frame, y="choice", x=["x1", "x2", "one", "twin"], base=0)
    assert omitted.provenance["omitted_terms"] == ["one", "twin"] and len(omitted.coefficients) == 9
    assert_allclose(estimates(omitted), estimates(oe.mlogit(data=data, y="choice", x=X, base=0)), rtol=1e-10)
    expanded = oe.mlogit(data=data, y="choice", x=["x1", "sector"], categorical=["sector"], base=0)
    assert [c.term for c in expanded.coefficients][:4] == ["1:Intercept", "1:x1", "1:sector[b]", "1:sector[c]"]
    holes = data.copy()
    holes.loc[[5, 9], "choice"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=holes, y="choice", x=X)
    assert caught.value.code == "missing_values"
    dropped = oe.mlogit(data=holes, y="choice", x=X, base=0, missing="drop")
    assert dropped.nobs == len(data) - 2 and dropped.extra["categories"] == [0, 1, 2, 3]
    assert_allclose(estimates(dropped), estimates(oe.mlogit(data=holes.dropna(), y="choice", x=X, base=0)),
                    rtol=1e-12)


def test_mlogit_derivatives_separation_and_error_codes(data):
    x = torch.tensor(sm.add_constant(data[X]).to_numpy())
    codes = torch.tensor(data.choice.to_numpy())
    objective = MultinomialObjective(x, codes, 4, 1, torch.tensor(data.aw.to_numpy()))
    theta = torch.tensor(np.random.default_rng(3).normal(size=9) * 0.4)
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    value, gradient, hessian = objective(theta)
    rows = objective.score_rows(theta)
    assert_allclose((rows * objective.w[:, None]).sum(dim=0).numpy(), gradient.numpy(), atol=1e-9)
    expected = mlogit_loglik_obs(theta.numpy(), x.numpy(), codes.numpy(), 1, 4)
    assert_allclose(float(value), (data.aw.to_numpy() * expected).sum(), rtol=1e-12)
    assert_allclose(float(objective.value(theta)), float(value), rtol=1e-14)
    assert float(torch.linalg.eigvalsh(hessian)[-1]) < 0
    # A huge index does not overflow the softmax.
    wide = torch.tensor(np.r_[800.0, np.zeros(8)])
    assert torch.isfinite(objective(wide)[0]) and torch.isfinite(objective(wide)[2]).all()
    separated = data.assign(flag=(data.choice == 3) * 1.0)
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=separated, y="choice", x=["x1", "flag"])
    assert caught.value.code == "separation_detected"
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=data.assign(choice=2), y="choice", x=X)
    assert caught.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=data.assign(choice=np.arange(len(data))), y="choice", x=X)
    assert caught.value.code == "too_many_categories"
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=data, y="choice", x=X, covariance="cluster")
    assert caught.value.code == "invalid_spec" and "cluster column" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.mlogit(data=data.groupby("choice").head(2), y="choice", x=X)      # 8 rows, 9 parameters
    assert caught.value.code == "insufficient_observations"
    with pytest.raises(ValidationError):
        ModelSpec(estimator="mlogit", outcome="choice", predictors=X, options={"baseline": 1})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="mlogit", outcome="choice", predictors=X, columns={"offset": "off"})


# ---- rendering and scale ------------------------------------------------------------------


def test_result_round_trip_summary_and_latex(data):
    ordered = oe.oprobit(data=data, y="grade", x=["x1", "sector"], categorical=["sector"], cluster="firm")
    multinomial = oe.mlogit(data=data, y="mode", x=X, base="bus", covariance="robust")
    for result in (ordered, multinomial):
        restored = type(result).model_validate_json(result.model_dump_json())
        assert restored == result and isinstance(restored, ResultBundle)
        assert ModelSpec.model_validate(json.loads(result.spec.model_dump_json())) == result.spec
        assert fit(result.spec, data=data).spec == result.spec
        assert result.provenance["optimizer"]["converged"] is True
        assert result.provenance["solver"] == "newton_observed_hessian"
        assert result.provenance["family"] == "discrete"
    payload = json.loads(ordered.model_dump_json())
    assert payload["extra"]["categories"] == ["poor", "fair", "good", "best"]
    assert payload["tests"]["model"]["distribution"] == "chi2"
    text = ordered.summary()
    assert "Ordered probit regression — grade" in text and "Covariance: cluster" in text
    assert "/cut3" in text and "sector[b]" in text and "[grade]" in text and "  z  " in text
    assert "Wald chi2 test of the slopes: chi2(3)" in text and "n_categories: 4" in text
    latex = str(ordered.to_latex())
    assert "sector[b]" in latex.replace(r"\_", "_") and "cut1" in latex
    text = multinomial.summary()
    assert "Multinomial logistic regression — mode" in text and "[car]" in text and "car:x1" in text
    assert "walk:Intercept" in str(multinomial.to_latex())
    assert multinomial.extra["base"] == "bus" and multinomial.spec.options == {"base": "bus"}


def test_ordered_and_multinomial_models_scale_linearly():
    rng = np.random.default_rng(11)
    n = 200_000
    frame = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    columns = list(frame.columns)
    index = frame.to_numpy() @ np.linspace(-0.5, 0.5, 6)
    frame["rating"] = np.digitize(index + rng.logistic(size=n), [-1.5, -0.3, 0.6, 1.8])
    utility = np.column_stack([np.zeros(n), 0.3 + index, -0.2 - index, 0.5 * index])
    frame["choice"] = (utility + rng.gumbel(size=(n, 4))).argmax(axis=1)
    frame["g"] = rng.integers(0, 3000, size=n)
    start = time.perf_counter()
    ordered = oe.ologit(data=frame, y="rating", x=columns, cluster="g")
    multinomial = oe.mlogit(data=frame, y="choice", x=columns, base=0)
    elapsed = time.perf_counter() - start
    assert ordered.nobs == n == multinomial.nobs and elapsed < 30
    assert_allclose(estimates(ordered)[:6], np.linspace(-0.5, 0.5, 6), atol=0.03)
    assert_allclose(estimates(multinomial).reshape(3, 7)[0, 1:], np.linspace(-0.5, 0.5, 6), atol=0.05)
