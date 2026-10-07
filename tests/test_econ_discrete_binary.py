"""Heteroskedastic probit and bivariate probit against independent oracles.

hetprobit and biprobit are compared with brute-force SciPy maximization of
independently written NumPy likelihoods; every covariance estimator is rebuilt
from complex-step scores and a numerical Hessian of those likelihoods. The
bivariate normal distribution function is checked against high-precision
quadrature (mpmath) and scipy.stats.multivariate_normal; the NumPy oracle used
for the biprobit likelihood is a 400-point Gauss-Legendre evaluation of
Plackett's identity, itself validated against SciPy.
"""

import json
import time

import mpmath
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import optimize as sopt
from scipy import special, stats
from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics.discrete.bivariate import (
    BiprobitObjective, bvn_cdf, bvn_upper, log_bvn_cdf,
)
from openecon.econometrics.discrete.kernels import HetprobitObjective, mills_ratio
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec, ResultBundle

X = ["x1", "x2"]
Z = ["z1", "z2"]
KINDS = ("nonrobust", "opg", "robust", "cluster")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20264)
    n = 900
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "z1": rng.normal(size=n), "z2": rng.uniform(-1, 1, size=n),
        "firm": np.repeat(np.arange(90), 10), "state": rng.integers(0, 12, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
    })
    latent = 0.3 + 0.9 * frame.x1 - 0.5 * frame.x2
    frame["y"] = (latent + np.exp(0.5 * frame.z1 - 0.4 * frame.z2) * rng.normal(size=n) > 0) * 1.0
    shocks = rng.multivariate_normal([0, 0], [[1, 0.55], [0.55, 1]], size=n)
    frame["work"] = (0.3 + 0.8 * frame.x1 - 0.4 * frame.x2 + shocks[:, 0] > 0) * 1.0
    frame["insured"] = (-0.2 + 0.5 * frame.x1 - 0.6 * frame.z1 + shocks[:, 1] > 0) * 1.0
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def design(frame, columns):
    return np.column_stack([np.ones(len(frame)), frame[columns].to_numpy()])


# ---- independent likelihoods ---------------------------------------------------------


def hetprobit_loglik_obs(theta, x, z, y):
    """ln Phi(q x'b / exp(z'g)) per observation (complex-step safe)."""
    k = x.shape[1]
    index = (x @ theta[:k]) / np.exp(z @ theta[k:])
    return np.log(special.ndtr((2 * y - 1) * index))


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(400)


def phi2(a, b, r):
    """Bivariate normal cdf by Plackett's identity with a 400-point rule (complex-step safe)."""
    half = np.arcsin(r) / 2
    theta = half[:, None] * (1 + _NODES)
    sine, cosine2 = np.sin(theta), np.cos(theta) ** 2
    integrand = np.exp(-((a ** 2 + b ** 2)[:, None] - 2 * (a * b)[:, None] * sine) / (2 * cosine2))
    return special.ndtr(a) * special.ndtr(b) + half * (integrand @ _WEIGHTS) / (2 * np.pi)


def biprobit_loglik_obs(theta, x1, x2, y1, y2):
    k1, k2 = x1.shape[1], x2.shape[1]
    q1, q2 = 2 * y1 - 1, 2 * y2 - 1
    rho = np.tanh(theta[-1])
    return np.log(phi2(q1 * (x1 @ theta[:k1]), q2 * (x2 @ theta[k1:k1 + k2]), q1 * q2 * rho))


def ml_covariances(loglik_obs, theta, weights=None, groups=None, nobs=None):
    """OIM, OPG, robust and cluster covariances of an independently written likelihood."""
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
    def negative(t):
        with np.errstate(all="ignore"):
            values = loglik_obs(t)
            total = -(values if weights is None else weights * values).sum()
        return total if np.isfinite(total) else 1e12

    return sopt.minimize(negative, start, method="BFGS", options={"gtol": 1e-9})


def wald(theta, cov, indices):
    return theta[indices] @ np.linalg.solve(cov[np.ix_(indices, indices)], theta[indices])


# ---- heteroskedastic probit ------------------------------------------------------------


def test_hetprobit_matches_brute_force_likelihood(data):
    x, z, y = design(data, X), data[Z].to_numpy(), data.y.to_numpy()

    def loglik(t):
        return hetprobit_loglik_obs(t, x, z, y)

    result = oe.hetprobit(data=data, y="y", x=X, het=Z)
    brute = brute_force(loglik, np.r_[0.2, 0.5, -0.3, 0.0, 0.0])
    terms = ["Intercept", "x1", "x2", "lnsigma:z1", "lnsigma:z2"]
    assert [c.term for c in result.coefficients] == terms
    assert [c.equation for c in result.coefficients] == ["y"] * 3 + ["lnsigma"] * 2
    assert_allclose(estimates(result), brute.x, rtol=2e-5, atol=2e-6)
    assert_allclose(result.metrics["log_likelihood"], -brute.fun, rtol=1e-11)
    n = len(data)
    assert result.nobs == n and result.title == "Heteroskedastic probit regression"
    assert_allclose(result.metrics["aic"], 2 * brute.fun + 2 * 5, rtol=1e-11)
    assert_allclose(result.metrics["bic"], 2 * brute.fun + np.log(n) * 5, rtol=1e-11)
    expected = ml_covariances(loglik, estimates(result), groups=data.firm.to_numpy())
    assert_allclose(covariance(result), expected["nonrobust"], rtol=2e-6, atol=1e-9)
    # Stata's header: Wald chi2 of the mean-equation slopes; LR test of lnsigma = 0 below.
    model = result.tests["model"]
    assert model["df"] == 2 and model["distribution"] == "chi2" and "Wald" in model["label"]
    assert_allclose(model["statistic"], wald(estimates(result), expected["nonrobust"], [1, 2]), rtol=1e-5)
    probit = sm.Probit(data.y, x).fit(disp=0, tol=1e-12)
    homoskedastic = result.tests["lnsigma"]
    assert homoskedastic["df"] == 2 and "LR test of lnsigma = 0" in homoskedastic["label"]
    assert_allclose(result.extra["probit_log_likelihood"], probit.llf, rtol=1e-11)
    assert_allclose(homoskedastic["statistic"], 2 * (-brute.fun - probit.llf), rtol=1e-8)
    assert_allclose(homoskedastic["p_value"], stats.chi2.sf(homoskedastic["statistic"], 2), rtol=1e-8)
    assert result.extra["zero_outcomes"] == int((y == 0).sum())
    assert result.extra["nonzero_outcomes"] == int(y.sum())
    assert result.extra["variance_terms"] == ["lnsigma:z1", "lnsigma:z2"]
    assert result.inference["use_t"] is False and result.inference["correction"] == "observed information"
    z_stat = estimates(result) / errors(result)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.norm.sf(np.abs(z_stat)), rtol=1e-8)
    # The chart sample holds Pr(y = 1) = Phi(x'b / exp(z'g)).
    theta = estimates(result)
    fitted = special.ndtr((x @ theta[:3]) / np.exp(z @ theta[3:]))
    rows = [p["row"] for p in result.predictions]
    assert_allclose([p["fitted"] for p in result.predictions], fitted[rows], rtol=1e-10)
    for kind in ("opg", "robust", "cluster"):
        other = oe.hetprobit(data=data, y="y", x=X, het=Z, covariance=kind,
                             cluster="firm" if kind == "cluster" else None)
        assert_allclose(estimates(other), theta, rtol=1e-12)
        assert_allclose(covariance(other), expected[kind], rtol=2e-6, atol=1e-9)
        assert_allclose(other.tests["model"]["statistic"], wald(theta, expected[kind], [1, 2]), rtol=1e-5)
        test = other.tests["lnsigma"]
        if kind == "opg":
            assert "LR" in test["label"]
            assert_allclose(test["statistic"], homoskedastic["statistic"], rtol=1e-12)
        else:
            assert "Wald test of lnsigma = 0" in test["label"] and test["df"] == 2
            assert_allclose(test["statistic"], wald(theta, expected[kind], [3, 4]), rtol=1e-5)
    assert other.inference["cluster_count"] == 90 and "G/(G-1)" in other.inference["correction"]
    twoway = oe.hetprobit(data=data, y="y", x=X, het=Z, cluster=["firm", "state"])
    assert twoway.inference["cluster_columns"] == ["firm", "state"] and np.isfinite(errors(twoway)).all()


def test_hetprobit_weights_follow_stata_semantics(data):
    x, z, y = design(data, X), data[Z].to_numpy(), data.y.to_numpy()

    def loglik(t):
        return hetprobit_loglik_obs(t, x, z, y)

    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in KINDS:
        extra = {"covariance": kind, "cluster": "firm" if kind == "cluster" else None}
        weighted = oe.hetprobit(data=data, y="y", x=X, het=Z, weights="fw", weight_type="fweight", **extra)
        duplicated = oe.hetprobit(data=repeated, y="y", x=X, het=Z, **extra)
        assert weighted.nobs == len(repeated) == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(duplicated), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance(weighted), covariance(duplicated), rtol=1e-7, atol=1e-11)
        for name in ("log_likelihood", "aic", "bic"):
            assert_allclose(weighted.metrics[name], duplicated.metrics[name], rtol=1e-10)
        for name in ("model", "lnsigma"):
            assert_allclose(weighted.tests[name]["statistic"], duplicated.tests[name]["statistic"], rtol=1e-7)
    n = len(data)
    scaled = data.aw.to_numpy() * n / data.aw.sum()
    analytic = oe.hetprobit(data=data, y="y", x=X, het=Z, weights="aw", weight_type="aweight")
    brute = brute_force(loglik, estimates(analytic) * 0.9, scaled)
    assert_allclose(estimates(analytic), brute.x, rtol=2e-5, atol=2e-6)
    expected = ml_covariances(loglik, estimates(analytic), weights=scaled)
    assert_allclose(covariance(analytic), expected["nonrobust"], rtol=2e-6, atol=1e-9)
    assert analytic.nobs == n
    importance = oe.hetprobit(data=data, y="y", x=X, het=Z, weights="aw", weight_type="iweight")
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-7)
    assert_allclose(covariance(importance) * data.aw.sum() / n, covariance(analytic), rtol=1e-7)
    sampling = oe.hetprobit(data=data, y="y", x=X, het=Z, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust" and "Wald" in sampling.tests["lnsigma"]["label"]
    raw = ml_covariances(loglik, estimates(sampling), weights=data.aw.to_numpy())
    assert_allclose(covariance(sampling), raw["robust"], rtol=2e-6, atol=1e-9)
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data, y="y", x=X, het=Z, weights="aw", weight_type="pweight", covariance="opg")
    assert caught.value.code == "unsupported_covariance"


def test_hetprobit_derivatives_and_stable_mills_ratio(data):
    x = torch.tensor(design(data, X))
    z = torch.tensor(data[Z].to_numpy())
    y = torch.tensor(data.y.to_numpy())
    objective = HetprobitObjective(x, z, y, torch.tensor(data.aw.to_numpy()))
    theta = torch.tensor([0.2, 0.7, -0.3, 0.4, -0.2], dtype=torch.float64)
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    assert report["hessian_asymmetry"] < 1e-10
    value, gradient, _ = objective(theta)
    rows = objective.score_rows(theta)
    assert_allclose((rows * objective.w[:, None]).sum(dim=0).numpy(), gradient.numpy(), atol=1e-9)
    expected = hetprobit_loglik_obs(theta.numpy(), x.numpy(), z.numpy(), y.numpy())
    assert_allclose(float(value), (data.aw.to_numpy() * expected).sum(), rtol=1e-12)
    assert_allclose(float(objective.value(theta)), float(value), rtol=1e-14)
    # Without variance regressors the objective is the probit likelihood.
    empty = torch.empty((len(data), 0), dtype=torch.float64)
    probit = HetprobitObjective(x, empty, y, torch.ones(len(data), dtype=torch.float64))
    report = check_derivatives(probit, theta[:3])
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    # Inverse Mills ratio and curvature against 50-digit arithmetic in both tails.
    mpmath.mp.dps = 50
    points = [-45.0, -12.0, -3.0, -0.5, 0.0, 0.7, 4.0, 9.0]
    log_cdf, ratio, curvature = mills_ratio(torch.tensor(points, dtype=torch.float64))
    for i, t in enumerate(points):
        cdf = mpmath.ncdf(t)
        exact = mpmath.npdf(t) / cdf
        assert_allclose(float(log_cdf[i]), float(mpmath.log(cdf)), rtol=1e-13)
        assert_allclose(float(ratio[i]), float(exact), rtol=1e-13)
        assert_allclose(float(curvature[i]), float(exact * (exact + t)), rtol=1e-9, atol=1e-300)
    far = mills_ratio(torch.tensor([-1e5, -1e8, 40.0], dtype=torch.float64))
    assert torch.isfinite(torch.stack(far)).all()
    assert_allclose(far[2][:2].numpy(), [1 - 1e-10, 1.0], rtol=1e-12)
    assert float(far[1][2]) == 0.0 and float(far[2][2]) == 0.0


def test_hetprobit_collinearity_missing_values_and_categoricals(data):
    base = oe.hetprobit(data=data, y="y", x=X, het=Z)
    frame = data.assign(one=1.0, twin=data.x1 + data.x2, zsum=data.z1 - 2 * data.z2)
    result = oe.hetprobit(data=frame, y="y", x=["x1", "x2", "twin"], het=["one", "z1", "z2", "zsum"])
    assert [c.term for c in result.coefficients] == [c.term for c in base.coefficients]
    assert result.provenance["omitted_terms"] == ["twin", "lnsigma:one", "lnsigma:zsum"]
    assert any("lnsigma:one" in warning for warning in result.warnings)
    assert_allclose(estimates(result), estimates(base), rtol=1e-9)
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=frame, y="y", x=X, het=["one"])
    assert caught.value.code == "no_variance_regressors" and "ordinary probit" in str(caught.value)
    # A variable may enter both equations; categoricals expand in either one.
    both = oe.hetprobit(data=data, y="y", x=["x1", "x2", "sector"], het=["x1", "sector"],
                        categorical=["sector"])
    assert [c.term for c in both.coefficients] == [
        "Intercept", "x1", "x2", "sector[b]", "sector[c]",
        "lnsigma:x1", "lnsigma:sector[b]", "lnsigma:sector[c]"]
    dummies = data.assign(b=(data.sector == "b") * 1.0, c=(data.sector == "c") * 1.0)
    plain = oe.hetprobit(data=dummies, y="y", x=["x1", "x2", "b", "c"], het=["x1", "b", "c"])
    assert_allclose(estimates(both), estimates(plain), rtol=1e-9)
    without = oe.hetprobit(data=data, y="y", x=X, het=Z, intercept=False)
    x, z, y = data[X].to_numpy(), data[Z].to_numpy(), data.y.to_numpy()
    brute = brute_force(lambda t: hetprobit_loglik_obs(t, x, z, y), np.r_[0.5, -0.3, 0.0, 0.0])
    assert [c.term for c in without.coefficients][:2] == X and without.tests["model"]["df"] == 2
    assert_allclose(estimates(without), brute.x, rtol=2e-5, atol=2e-6)
    holes = data.copy()
    holes.loc[[3, 40], "z1"] = np.nan
    holes.loc[7, "y"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=holes, y="y", x=X, het=Z)
    assert caught.value.code == "missing_values"
    dropped = oe.hetprobit(data=holes, y="y", x=X, het=Z, missing="drop")
    complete = oe.hetprobit(data=holes.dropna(subset=["z1", "y"]), y="y", x=X, het=Z)
    assert dropped.nobs == len(data) - 3 and dropped.dropped_rows == 3
    assert 40 not in dropped.sample_positions and 7 not in dropped.sample_positions
    assert_allclose(estimates(dropped), estimates(complete), rtol=1e-12)


def test_hetprobit_is_invariant_to_the_location_of_variance_regressors(data):
    base = oe.hetprobit(data=data, y="y", x=X, het=Z, covariance="robust")
    theta, cov = estimates(base), covariance(base)
    # A moderate shift, checked in the model's own parameterization by brute force.
    shift = np.array([3.0, -2.0])
    moved = data.assign(z1=data.z1 + shift[0], z2=data.z2 + shift[1])
    result = oe.hetprobit(data=moved, y="y", x=X, het=Z, covariance="robust")
    x, z, y = design(moved, X), moved[Z].to_numpy(), moved.y.to_numpy()

    def loglik(t):
        return hetprobit_loglik_obs(t, x, z, y)

    brute = brute_force(loglik, estimates(result) * 0.97)
    assert_allclose(estimates(result), brute.x, rtol=5e-5, atol=5e-6)
    assert_allclose(covariance(result), ml_covariances(loglik, estimates(result))["robust"],
                    rtol=5e-6, atol=1e-9)
    # The shift is an exact reparameterization: b' = b exp(c'g), V' = J V J'.
    factor = np.exp(shift @ theta[3:])
    assert_allclose(estimates(result), np.r_[theta[:3] * factor, theta[3:]], rtol=1e-8)
    jacobian = np.eye(5)
    jacobian[:3, :3] *= factor
    jacobian[:3, 3:] = np.outer(theta[:3] * factor, shift)
    assert_allclose(covariance(result), jacobian @ cov @ jacobian.T, rtol=1e-7)
    assert_allclose(result.metrics["log_likelihood"], base.metrics["log_likelihood"], rtol=1e-12)
    assert_allclose(result.tests["lnsigma"]["statistic"], base.tests["lnsigma"]["statistic"], rtol=1e-7)
    assert_allclose(result.extra["variance_regressor_means"], moved[Z].mean().to_numpy(), rtol=1e-12)
    # A variance regressor with a large mean (an age, a year) still converges; the mean
    # equation is then on the scale exp(mean(z)'g), which the result points out.
    far = oe.hetprobit(data=data.assign(z1=data.z1 + 60.0), y="y", x=X, het=Z)
    plain = oe.hetprobit(data=data, y="y", x=X, het=Z)
    assert_allclose(far.metrics["log_likelihood"], plain.metrics["log_likelihood"], rtol=1e-12)
    assert_allclose(estimates(far)[3:], estimates(plain)[3:], rtol=1e-7)
    assert_allclose(estimates(far)[:3], estimates(plain)[:3] * np.exp(60 * estimates(plain)[3]), rtol=1e-6)
    assert_allclose((estimates(far) / errors(far))[3:], (estimates(plain) / errors(plain))[3:], rtol=1e-6)
    assert any("large means" in warning for warning in far.warnings) and not plain.warnings
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data.assign(z1=data.z1 + 5000.0), y="y", x=X, het=Z)
    assert caught.value.code == "variance_scale_overflow" and "Centre" in str(caught.value)


def test_hetprobit_error_codes(data):
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data.assign(y=data.y + 1), y="y", x=X, het=Z)
    assert caught.value.code == "invalid_binary_outcome" and "0/1" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data.assign(y=1.0), y="y", x=X, het=Z)
    assert caught.value.code == "constant_outcome"
    separated = data.assign(flag=data.y)
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=separated, y="y", x=["x1", "flag"], het=Z)
    assert caught.value.code == "separation_detected" and "probit starting values" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data, y="y", x=X, het="z1")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data, y="y", x=X, het=[])
    assert caught.value.code == "invalid_spec" and "het" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data, y="y", x=X, het=["y"])
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data, y="y", x=X, het=Z, covariance="HC1")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data, y="y", x=X, het=["z1", "nope"])
    assert caught.value.code == "missing_columns"
    with pytest.raises(AnalysisError) as caught:
        oe.hetprobit(data=data.head(5), y="y", x=X, het=Z)
    assert caught.value.code in {"insufficient_observations", "separation_detected", "constant_outcome"}
    for fields in [{"columns": {}}, {"columns": {"het": Z, "group": "firm"}},
                   {"columns": {"het": Z}, "options": {"base": 1}},
                   {"columns": {"het": Z}, "panel": "firm"}]:
        with pytest.raises(ValidationError):
            ModelSpec(estimator="hetprobit", outcome="y", predictors=X, **fields)


# ---- bivariate normal distribution function ---------------------------------------------


def exact_log_phi2(a, b, r):
    """ln Phi2(a, b; r) by 30-digit quadrature of phi(x) Phi((b - r x)/s) over x <= min(a, b)."""
    mpmath.mp.dps = 30
    a, b, r = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(r)
    limit, other = min(a, b), max(a, b)
    scale = mpmath.sqrt(1 - r * r)

    def integrand(v):
        return mpmath.npdf(v) * mpmath.ncdf((other - r * v) / scale)

    # Panels: geometric in units of the decay length at the limit, unit steps further out,
    # and the neighbourhood of the point where the conditional probability switches.
    step = mpmath.mpf(10) ** -10
    rate = abs(mpmath.log(integrand(limit)) - mpmath.log(integrand(limit - step))) / step
    width = 1 / max(rate, 1)
    panels = {limit - j * width for j in (0, 1, 2, 4, 8, 16, 32, 64)}
    panels |= {limit - j for j in (1, 2, 3, 5, 8, 14)}
    if r != 0:
        switch, spread = other / r, scale / abs(r)
        panels |= {switch + j * spread for j in (-8, -3, 0, 3, 8)}
        panels |= {limit - j * spread for j in (1, 2, 4, 8, 16, 32)}
    panels = sorted(point for point in panels if limit - 14 <= point <= limit)
    return float(mpmath.log(mpmath.quad(integrand, panels)))


def test_bivariate_normal_cdf_matches_high_precision_quadrature():
    rng = np.random.default_rng(0)
    cases = []
    for r in (-0.999, -0.925, -0.5, 0.0, 0.76, 0.924, 0.926, 0.99999):
        cases += [(*rng.uniform(-3, 3, size=2), r) for _ in range(2)]
        cases += [(0.0, 0.0, r), (0.3, 0.3000001, r)]
    a, b, r = (torch.tensor(column, dtype=torch.float64) for column in zip(*cases))
    ours = bvn_cdf(a, b, r).numpy()
    reference = np.exp([exact_log_phi2(*case) for case in cases])
    assert_allclose(ours, reference, rtol=0, atol=3e-15)
    # A dense comparison with SciPy's bivariate normal, including both branches.
    points = rng.uniform(-4, 4, size=(400, 2))
    correlations = rng.choice([-0.98, -0.93, -0.6, -0.1, 0.2, 0.7, 0.92, 0.95, 0.999], size=400)
    scipy_values = np.array([
        stats.multivariate_normal.cdf(p, mean=[0, 0], cov=[[1, c], [c, 1]])
        for p, c in zip(points, correlations)])
    ours = bvn_cdf(torch.tensor(points[:, 0]), torch.tensor(points[:, 1]), torch.tensor(correlations))
    assert_allclose(ours.numpy(), scipy_values, rtol=0, atol=5e-14)
    # The NumPy oracle used for the biprobit likelihood agrees with both.
    moderate = np.abs(correlations) < 0.9
    assert_allclose(phi2(points[moderate, 0], points[moderate, 1], correlations[moderate]),
                    scipy_values[moderate], rtol=0, atol=5e-14)


def test_log_bivariate_normal_cdf_keeps_relative_accuracy_in_the_tails():
    # Outlying arguments: Genz's absolute accuracy of 1e-15 is useless for the logarithm here.
    cases = [(-6.3, 0.2, -0.56), (-32.0, -1.0, 0.67), (-32.0, -1.0, -0.67), (-2.0, -2.0, -0.95),
             (-7.0, 7.5, -0.5), (-9.0, -9.0, 0.999), (-12.0, -11.0, 0.5), (-45.0, 2.0, 0.3),
             (-8.0, -3.0, 0.0), (-5.0, -5.0, -0.999), (-3.0, -8.0, 0.7), (-20.0, -19.99, 0.9999),
             (-10.0, 9.0, -0.9999), (-1.5, -4.2, 0.93), (1.1, -8.2, -0.97)]
    a, b, r = (torch.tensor(column, dtype=torch.float64) for column in zip(*cases))
    ours = log_bvn_cdf(a, b, r).numpy()
    reference = np.array([exact_log_phi2(*case) for case in cases])
    assert_allclose(ours, reference, rtol=1e-12, atol=1e-11)
    plain = torch.log(bvn_cdf(a, b, r)).numpy()
    assert np.abs(plain - reference).max() > 1e-3            # what the tail evaluation repairs
    assert_allclose(log_bvn_cdf(b, a, r).numpy(), ours, rtol=1e-13)
    # Ordinary probabilities are the logarithm of the Genz value, and the switch is seamless.
    rng = np.random.default_rng(2)
    x = torch.tensor(rng.uniform(-2, 3, size=300))
    y = torch.tensor(rng.uniform(-2, 3, size=300))
    c = torch.tensor(rng.uniform(-0.95, 0.95, size=300))
    ordinary = bvn_cdf(x, y, c) > 1e-4
    assert int(ordinary.sum()) > 250
    assert torch.equal(log_bvn_cdf(x, y, c)[ordinary], torch.log(bvn_cdf(x, y, c))[ordinary])
    edge = torch.linspace(-4.6, -3.9, 200, dtype=torch.float64)          # Phi2 crosses 1e-5 here
    values = log_bvn_cdf(edge, edge + 0.5, 0.4)
    exact = np.array([exact_log_phi2(v, v + 0.5, 0.4) for v in edge[::40].tolist()])
    assert_allclose(values[::40].numpy(), exact, rtol=1e-11)
    assert bool((values[1:] > values[:-1]).all())
    # Closed forms far beyond the float64 range of the probability itself.
    far = torch.tensor([-200.0, -1e3], dtype=torch.float64)
    other = torch.tensor([-300.0, -40.0], dtype=torch.float64)
    independent = torch.special.log_ndtr(far) + torch.special.log_ndtr(other)
    assert_allclose(log_bvn_cdf(far, other, 0.0).numpy(), independent.numpy(), rtol=1e-13)
    infinite = torch.tensor([float("inf")] * 2, dtype=torch.float64)
    assert_allclose(log_bvn_cdf(far, infinite, 0.6).numpy(), torch.special.log_ndtr(far).numpy(), rtol=1e-13)
    assert_allclose(log_bvn_cdf(far, -far, 0.6).numpy(), torch.special.log_ndtr(far).numpy(), rtol=1e-13)
    nothing = log_bvn_cdf(torch.tensor([-float("inf"), 0.5]), torch.tensor([0.0, -0.6]),
                          torch.tensor([0.3, -1.0], dtype=torch.float64))
    assert nothing.tolist() == [-float("inf"), -float("inf")]
    assert log_bvn_cdf(torch.zeros(2, 3, dtype=torch.float64), -9.0, 0.2).shape == (2, 3)


def test_bivariate_normal_cdf_identities_and_limits():
    rng = np.random.default_rng(1)
    a = torch.tensor(rng.uniform(-3, 3, size=200))
    b = torch.tensor(rng.uniform(-3, 3, size=200))
    r = torch.tensor(rng.uniform(-0.99, 0.99, size=200))
    cdf_a, cdf_b = torch.special.ndtr(a), torch.special.ndtr(b)
    assert_allclose(bvn_cdf(a, b, 0.0).numpy(), (cdf_a * cdf_b).numpy(), rtol=0, atol=1e-15)
    assert_allclose(bvn_cdf(a, b, r).numpy(), bvn_cdf(b, a, r).numpy(), rtol=0, atol=1e-15)
    assert_allclose((bvn_cdf(a, b, r) + bvn_cdf(a, -b, -r)).numpy(), cdf_a.numpy(), rtol=0, atol=2e-15)
    assert_allclose(bvn_upper(a, b, r).numpy(), bvn_cdf(-a, -b, r).numpy(), rtol=0, atol=1e-15)
    upper = 1 - cdf_a - cdf_b + bvn_cdf(a, b, r)
    assert_allclose(bvn_upper(a, b, r).numpy(), upper.numpy(), rtol=0, atol=3e-15)
    zero = torch.zeros(200, dtype=torch.float64)
    assert_allclose(bvn_cdf(zero, zero, r).numpy(), 0.25 + np.arcsin(r.numpy()) / (2 * np.pi),
                    rtol=0, atol=1e-15)
    # Perfect correlation: Phi2(a, b; 1) = Phi(min(a, b)), Phi2(a, b; -1) = max(0, Phi(a) + Phi(b) - 1).
    assert_allclose(bvn_cdf(a, b, 1.0).numpy(), torch.special.ndtr(torch.minimum(a, b)).numpy(),
                    rtol=0, atol=1e-15)
    assert_allclose(bvn_cdf(a, b, -1.0).numpy(), (cdf_a + cdf_b - 1).clamp_min(0).numpy(),
                    rtol=0, atol=1e-15)
    # Infinite and extreme arguments are exact marginals, zero or one.
    big = torch.tensor([50.0, -50.0, 50.0, 0.5, float("inf")], dtype=torch.float64)
    other = torch.tensor([50.0, 1.0, 0.5, -50.0, 0.25], dtype=torch.float64)
    expected = [1.0, 0.0, stats.norm.cdf(0.5), 0.0, stats.norm.cdf(0.25)]
    assert_allclose(bvn_cdf(big, other, 0.5).numpy(), expected, rtol=0, atol=1e-15)
    assert_allclose(bvn_cdf(big, other, -0.97).numpy(), expected, rtol=0, atol=1e-15)
    # The complement 1 - r^2 may be supplied when r is within rounding of one.
    alpha = 12.0
    point = torch.tensor([0.2], dtype=torch.float64)
    value = bvn_cdf(point, point + 1e-6, np.tanh(alpha), 1 / np.cosh(alpha) ** 2)
    assert 0 < stats.norm.cdf(0.2) - float(value) < 1e-5
    assert bvn_cdf(torch.zeros(3, 2, dtype=torch.float64), 0.3, 0.4).shape == (3, 2)


# ---- bivariate probit ---------------------------------------------------------------------


def test_biprobit_matches_brute_force_likelihood(data):
    x1, x2 = design(data, X), design(data, ["x1", "z1"])
    y1, y2 = data.work.to_numpy(), data.insured.to_numpy()

    def loglik(t):
        return biprobit_loglik_obs(t, x1, x2, y1, y2)

    result = oe.biprobit(data=data, y1="work", y2="insured", x=X, x2=["x1", "z1"])
    terms = ["work:Intercept", "work:x1", "work:x2", "insured:Intercept", "insured:x1", "insured:z1",
             "/athrho"]
    assert [c.term for c in result.coefficients] == terms
    assert [c.equation for c in result.coefficients] == ["work"] * 3 + ["insured"] * 3 + [None]
    assert result.title == "Seemingly unrelated bivariate probit"
    brute = brute_force(loglik, np.r_[0.2, 0.6, -0.3, -0.1, 0.4, -0.5, 0.3])
    theta = estimates(result)
    assert_allclose(theta, brute.x, rtol=3e-5, atol=3e-6)
    assert_allclose(result.metrics["log_likelihood"], -brute.fun, rtol=1e-11)
    # The likelihood at the estimates against SciPy's bivariate normal, observation by observation.
    q1, q2 = 2 * y1 - 1, 2 * y2 - 1
    rho = np.tanh(theta[-1])
    joint = [stats.multivariate_normal.cdf([a, b], mean=[0, 0], cov=[[1, c], [c, 1]])
             for a, b, c in zip(q1 * (x1 @ theta[:3]), q2 * (x2 @ theta[3:6]), q1 * q2 * rho)]
    assert_allclose(result.metrics["log_likelihood"], np.log(joint).sum(), rtol=1e-12)
    n = len(data)
    assert result.nobs == n
    assert_allclose(result.metrics["aic"], 2 * brute.fun + 2 * 7, rtol=1e-11)
    assert_allclose(result.metrics["bic"], 2 * brute.fun + np.log(n) * 7, rtol=1e-11)
    expected = ml_covariances(loglik, theta, groups=data.firm.to_numpy())
    assert_allclose(covariance(result), expected["nonrobust"], rtol=2e-6, atol=1e-9)
    # rho: delta-method standard error and the tanh of the athrho interval (Stata's display).
    record = result.extra["rho"]
    se = np.sqrt(expected["nonrobust"][-1, -1])
    crit = stats.norm.ppf(0.975)
    assert_allclose(result.metrics["rho"], rho, rtol=1e-14)
    assert_allclose(record["estimate"], rho, rtol=1e-14)
    assert_allclose(record["std_error"], (1 - rho ** 2) * se, rtol=1e-6)
    assert_allclose([record["ci_low"], record["ci_high"]],
                    np.tanh([theta[-1] - crit * se, theta[-1] + crit * se]), rtol=1e-6)
    last = result.coefficients[-1]
    assert_allclose([np.tanh(last.ci_low), np.tanh(last.ci_high)], [record["ci_low"], record["ci_high"]],
                    rtol=1e-12)
    # Header Wald test of all slopes; LR test of rho = 0 against the two separate probits.
    slopes = [1, 2, 4, 5]
    model = result.tests["model"]
    assert model["df"] == 4 and "Wald" in model["label"]
    assert_allclose(model["statistic"], wald(theta, expected["nonrobust"], slopes), rtol=1e-5)
    separate = [sm.Probit(y1, x1).fit(disp=0, tol=1e-12).llf, sm.Probit(y2, x2).fit(disp=0, tol=1e-12).llf]
    assert_allclose(result.extra["probit_log_likelihoods"], separate, rtol=1e-11)
    assert_allclose(result.extra["comparison_log_likelihood"], sum(separate), rtol=1e-11)
    independence = result.tests["rho"]
    assert independence["df"] == 1 and "LR test of rho = 0" in independence["label"]
    assert_allclose(independence["statistic"], 2 * (-brute.fun - sum(separate)), rtol=1e-8)
    assert_allclose(independence["p_value"], stats.chi2.sf(independence["statistic"], 1), rtol=1e-8)
    table = pd.crosstab(data.work, data.insured)
    assert result.extra["outcome_counts"] == {f"{a}{b}": float(table.loc[a, b]) for a in (0, 1) for b in (0, 1)}
    assert result.extra["seemingly_unrelated"] is True and result.extra["recursive"] is False
    assert result.extra["hessian"] == "analytic" and result.extra["equations"] == ["work", "insured"]
    rows = [p["row"] for p in result.predictions]
    assert_allclose([p["fitted"] for p in result.predictions], special.ndtr(x1 @ theta[:3])[rows], rtol=1e-10)
    for kind in ("opg", "robust", "cluster"):
        other = oe.biprobit(data=data, y1="work", y2="insured", x=X, x2=["x1", "z1"], covariance=kind,
                            cluster="firm" if kind == "cluster" else None)
        assert_allclose(estimates(other), theta, rtol=1e-12)
        assert_allclose(covariance(other), expected[kind], rtol=2e-6, atol=1e-9)
        assert_allclose(other.tests["model"]["statistic"], wald(theta, expected[kind], slopes), rtol=1e-5)
        test = other.tests["rho"]
        if kind == "opg":
            assert "LR" in test["label"]
        else:
            assert test["label"] == "Wald test of rho = 0" and test["df"] == 1
            assert_allclose(test["statistic"], theta[-1] ** 2 / expected[kind][-1, -1], rtol=1e-5)
    assert other.inference["cluster_count"] == 90
    assert_allclose(other.inference["small_sample_correction"], 90 / 89)
    twoway = oe.biprobit(data=data, y1="work", y2="insured", x=X, cluster=["firm", "state"])
    assert twoway.inference["cluster_columns"] == ["firm", "state"] and np.isfinite(errors(twoway)).all()
    narrow = oe.biprobit(data=data, y1="work", y2="insured", x=X, x2=["x1", "z1"], alpha=0.1)
    crit = stats.norm.ppf(0.95)
    assert_allclose(narrow.extra["rho"]["ci_high"], np.tanh(theta[-1] + crit * se), rtol=1e-6)


def test_biprobit_common_regressors_recursive_model_and_spec_path(data):
    result = oe.biprobit(data=data, y1="work", y2="insured", x=["x1", "x2", "z1"])
    assert result.title == "Bivariate probit regression" and result.extra["seemingly_unrelated"] is False
    assert [c.term for c in result.coefficients] == [
        f"{name}:{term}" for name in ("work", "insured") for term in ("Intercept", "x1", "x2", "z1")
    ] + ["/athrho"]
    assert result.tests["model"]["df"] == 6
    assert "predictors2" not in result.spec.columns and result.spec.columns == {"outcome2": "insured"}
    explicit = oe.biprobit(data=data, y1="work", y2="insured", x=["x1", "x2", "z1"], x2=["x1", "x2", "z1"])
    assert_allclose(estimates(explicit), estimates(result), rtol=1e-13)
    assert explicit.title == "Bivariate probit regression"
    spec = ModelSpec(estimator="biprobit", outcome="work", predictors=["x1", "x2", "z1"],
                     columns={"outcome2": "insured"})
    assert spec.covariance == "nonrobust"
    assert_allclose(estimates(fit(spec, data=data)), estimates(result), rtol=1e-13)
    # Swapping the equations is the same model.
    swapped = oe.biprobit(data=data, y1="insured", y2="work", x=["x1", "x2", "z1"])
    assert_allclose(swapped.metrics["log_likelihood"], result.metrics["log_likelihood"], rtol=1e-12)
    assert_allclose(estimates(swapped)[[4, 5, 6, 7, 0, 1, 2, 3, 8]], estimates(result), rtol=1e-7, atol=1e-9)
    # Recoding one outcome flips the sign of its equation and of rho.
    flipped = oe.biprobit(data=data.assign(insured=1 - data.insured), y1="work", y2="insured",
                          x=["x1", "x2", "z1"])
    sign = np.r_[np.ones(4), -np.ones(5)]
    assert_allclose(estimates(flipped), sign * estimates(result), rtol=1e-7, atol=1e-9)
    assert_allclose(flipped.metrics["rho"], -result.metrics["rho"], rtol=1e-7)
    # Recursive bivariate probit: the first outcome is a regressor of the second equation.
    rng = np.random.default_rng(5)
    n = len(data)
    shocks = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], size=n)
    frame = data.assign(treated=(0.2 + 0.9 * data.x1 + 0.7 * data.z2 + shocks[:, 0] > 0) * 1.0)
    frame["outcome"] = (-0.3 + 0.8 * frame.treated - 0.5 * frame.z1 + shocks[:, 1] > 0) * 1.0
    recursive = oe.biprobit(data=frame, y1="treated", y2="outcome", x=["x1", "z2"], x2=["treated", "z1"])
    assert recursive.extra["recursive"] is True
    x1, x2 = design(frame, ["x1", "z2"]), design(frame, ["treated", "z1"])
    brute = brute_force(lambda t: biprobit_loglik_obs(t, x1, x2, frame.treated.to_numpy(),
                                                      frame.outcome.to_numpy()),
                        estimates(recursive) * 0.9)
    assert_allclose(estimates(recursive), brute.x, rtol=5e-5, atol=5e-6)
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=frame, y1="treated", y2="outcome", x=["x1", "outcome"], x2=["treated", "z1"])
    assert caught.value.code == "invalid_spec" and "recursive" in str(caught.value)


def test_biprobit_weights_follow_stata_semantics(data):
    x1, x2 = design(data, X), design(data, ["x1", "z1"])
    y1, y2 = data.work.to_numpy(), data.insured.to_numpy()

    def loglik(t):
        return biprobit_loglik_obs(t, x1, x2, y1, y2)

    arguments = {"y1": "work", "y2": "insured", "x": X, "x2": ["x1", "z1"]}
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in KINDS:
        extra = {"covariance": kind, "cluster": "firm" if kind == "cluster" else None, **arguments}
        weighted = oe.biprobit(data=data, weights="fw", weight_type="fweight", **extra)
        duplicated = oe.biprobit(data=repeated, **extra)
        assert weighted.nobs == len(repeated) == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(duplicated), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance(weighted), covariance(duplicated), rtol=1e-7, atol=1e-11)
        for name in ("log_likelihood", "aic", "bic", "rho"):
            assert_allclose(weighted.metrics[name], duplicated.metrics[name], rtol=1e-9)
        for name in ("model", "rho"):
            assert_allclose(weighted.tests[name]["statistic"], duplicated.tests[name]["statistic"], rtol=1e-7)
        assert weighted.extra["outcome_counts"] == duplicated.extra["outcome_counts"]
    n = len(data)
    scaled = data.aw.to_numpy() * n / data.aw.sum()
    analytic = oe.biprobit(data=data, weights="aw", weight_type="aweight", **arguments)
    brute = brute_force(loglik, estimates(analytic) * 0.9, scaled)
    assert_allclose(estimates(analytic), brute.x, rtol=3e-5, atol=3e-6)
    expected = ml_covariances(loglik, estimates(analytic), weights=scaled)
    assert_allclose(covariance(analytic), expected["nonrobust"], rtol=2e-6, atol=1e-9)
    assert_allclose(analytic.metrics["log_likelihood"], (scaled * loglik(estimates(analytic))).sum(), rtol=1e-11)
    importance = oe.biprobit(data=data, weights="aw", weight_type="iweight", **arguments)
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-7)
    assert_allclose(covariance(importance) * data.aw.sum() / n, covariance(analytic), rtol=1e-7)
    sampling = oe.biprobit(data=data, weights="aw", weight_type="pweight", **arguments)
    assert sampling.spec.covariance == "robust" and sampling.tests["rho"]["label"] == "Wald test of rho = 0"
    raw = ml_covariances(loglik, estimates(sampling), weights=data.aw.to_numpy())
    assert_allclose(covariance(sampling), raw["robust"], rtol=2e-6, atol=1e-9)
    doubled = oe.biprobit(data=data.assign(aw=2 * data.aw), weights="aw", weight_type="pweight", **arguments)
    assert_allclose(covariance(doubled), covariance(sampling), rtol=1e-8)
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=data, weights="aw", weight_type="pweight", covariance="nonrobust", **arguments)
    assert caught.value.code == "unsupported_covariance"


@pytest.mark.parametrize("rho", [0.3, -0.6, 0.95, -0.97, 0.995])
def test_biprobit_analytic_derivatives_match_numerical_ones(rho):
    rng = np.random.default_rng(int(1000 * abs(rho)))
    n = 500
    x1 = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    x2 = np.column_stack([np.ones(n), rng.normal(size=n)])
    shocks = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    b1, b2 = np.array([0.2, 0.5, -0.3]), np.array([-0.1, 0.4])
    y1 = (x1 @ b1 + shocks[:, 0] > 0) * 1.0
    y2 = (x2 @ b2 + shocks[:, 1] > 0) * 1.0
    weights = torch.tensor(rng.uniform(0.5, 2, size=n))
    objective = BiprobitObjective(torch.tensor(x1), torch.tensor(x2), torch.tensor(y1), torch.tensor(y2),
                                  weights)
    theta = torch.tensor(np.r_[b1 + 0.05, b2 - 0.05, 0.97 * np.arctanh(rho)])
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    assert report["hessian_asymmetry"] < 1e-10
    value, gradient, _ = objective(theta)
    rows = objective.score_rows(theta)
    assert_allclose((rows * weights[:, None]).sum(dim=0).numpy(), gradient.numpy(), atol=1e-8)
    assert_allclose(float(objective.value(theta)), float(value), rtol=1e-14)
    if abs(rho) < 0.9:
        expected = biprobit_loglik_obs(theta.numpy(), x1, x2, y1, y2)
        assert_allclose(float(value), (weights.numpy() * expected).sum(), rtol=1e-12)
    # A trial point outside the parameter space is rejected with a non-finite value.
    outside = theta.clone()
    outside[-1] = float("inf")
    assert not np.isfinite(float(objective.value(outside)))
    assert not np.isfinite(float(objective(outside)[0]))


def test_biprobit_is_stable_with_outlying_observations(data):
    # Two gross outliers with the "wrong" outcome: fitted joint probabilities of about 1e-8.
    frame = data.copy()
    frame.loc[0, ["x1", "work"]] = [12.0, 0.0]
    frame.loc[1, ["x1", "work"]] = [-12.0, 1.0]
    result = oe.biprobit(data=frame, y1="work", y2="insured", x=X, x2=["x1", "z1"])
    theta = estimates(result)
    x1, x2 = design(frame, X), design(frame, ["x1", "z1"])
    y1, y2 = frame.work.to_numpy(), frame.insured.to_numpy()
    q1, q2 = 2 * y1 - 1, 2 * y2 - 1
    w1, w2, r = q1 * (x1 @ theta[:3]), q2 * (x2 @ theta[3:6]), q1 * q2 * np.tanh(theta[-1])
    joint = np.array([stats.multivariate_normal.cdf([a, b], mean=[0, 0], cov=[[1, c], [c, 1]])
                      for a, b, c in zip(w1, w2, r)])
    log_joint = np.log(np.maximum(joint, 1e-300))
    small = joint < 1e-5
    assert small.sum() == 2 and joint.min() < 1e-7
    log_joint[small] = [exact_log_phi2(a, b, c) for a, b, c in zip(w1[small], w2[small], r[small])]
    assert_allclose(result.metrics["log_likelihood"], log_joint.sum(), rtol=1e-12)
    objective = BiprobitObjective(torch.tensor(x1), torch.tensor(x2), torch.tensor(y1), torch.tensor(y2),
                                  torch.ones(len(frame), dtype=torch.float64))
    value, gradient, hessian = objective(torch.tensor(theta))
    assert float(gradient.abs().max()) < 1e-6 and float(torch.linalg.eigvalsh(hessian)[-1]) < 0
    report = check_derivatives(objective, torch.tensor(theta) + 0.02)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    assert_allclose(covariance(result), np.linalg.inv(-hessian.numpy()), rtol=1e-8)
    # Probabilities far below the float64 range still give a finite, differentiable likelihood.
    wild = torch.tensor(theta * np.r_[1, 60, 1, 1, 1, 1, 1])
    value, gradient, hessian = objective(wild)
    assert float(value) < -1e3 and torch.isfinite(gradient).all() and torch.isfinite(hessian).all()
    report = check_derivatives(objective, wild)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-7


def test_biprobit_collinearity_missing_values_and_categoricals(data):
    base = oe.biprobit(data=data, y1="work", y2="insured", x=X, x2=["x1", "z1"])
    frame = data.assign(one=1.0, twin=data.x1 - data.x2, zz=3 * data.z1)
    result = oe.biprobit(data=frame, y1="work", y2="insured", x=["x1", "x2", "twin"],
                         x2=["x1", "one", "z1", "zz"])
    assert [c.term for c in result.coefficients] == [c.term for c in base.coefficients]
    assert result.provenance["omitted_terms"] == ["work:twin", "insured:one", "insured:zz"]
    assert sum("collinearity" in warning for warning in result.warnings) == 2
    assert_allclose(estimates(result), estimates(base), rtol=1e-9)
    expanded = oe.biprobit(data=data, y1="work", y2="insured", x=["x1", "sector"], x2=["z1", "sector"],
                           categorical=["sector"])
    assert [c.term for c in expanded.coefficients] == [
        "work:Intercept", "work:x1", "work:sector[b]", "work:sector[c]",
        "insured:Intercept", "insured:z1", "insured:sector[b]", "insured:sector[c]", "/athrho"]
    without = oe.biprobit(data=data, y1="work", y2="insured", x=X, x2=["x1", "z1"], intercept=False)
    assert [c.term for c in without.coefficients][:2] == ["work:x1", "work:x2"]
    assert without.tests["model"]["df"] == 4 and len(without.coefficients) == 5
    holes = data.copy()
    holes.loc[[3, 40], "z1"] = np.nan
    holes.loc[7, "insured"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=holes, y1="work", y2="insured", x=X, x2=["x1", "z1"])
    assert caught.value.code == "missing_values"
    dropped = oe.biprobit(data=holes, y1="work", y2="insured", x=X, x2=["x1", "z1"], missing="drop")
    complete = oe.biprobit(data=holes.dropna(subset=["z1", "insured"]), y1="work", y2="insured", x=X,
                           x2=["x1", "z1"])
    assert dropped.nobs == len(data) - 3 and dropped.dropped_rows == 3
    assert 7 not in dropped.sample_positions and 40 not in dropped.sample_positions
    assert_allclose(estimates(dropped), estimates(complete), rtol=1e-12)


def test_biprobit_boundary_separation_and_error_codes(data):
    for column, sign in ((data.work, "+1"), (1 - data.work, "-1")):
        with pytest.raises(AnalysisError) as caught:
            oe.biprobit(data=data.assign(insured=column), y1="work", y2="insured", x=X)
        assert caught.value.code == "boundary_solution" and f"runs to {sign}" in str(caught.value)
    # An empty cell of the 2x2 table alone is not a boundary: the estimate of rho is high but interior.
    nested = data.assign(insured=data.work * (data.z1 > 0))
    interior = oe.biprobit(data=nested, y1="work", y2="insured", x=X, x2=["x1"])
    assert interior.extra["outcome_counts"]["01"] == 0 and 0.9 < interior.metrics["rho"] < 1
    x1, x2 = design(nested, X), design(nested, ["x1"])
    brute = brute_force(lambda t: biprobit_loglik_obs(t, x1, x2, nested.work.to_numpy(),
                                                      nested.insured.to_numpy()),
                        estimates(interior) * 0.95)
    assert_allclose(interior.metrics["log_likelihood"], -brute.fun, rtol=1e-8)
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=data.assign(flag=data.insured), y1="work", y2="insured", x=X, x2=["x1", "flag"])
    assert caught.value.code == "separation_detected" and "probit for insured" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=data.assign(insured=data.insured * 2), y1="work", y2="insured", x=X)
    assert caught.value.code == "invalid_binary_outcome"
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=data.assign(insured=0.0), y1="work", y2="insured", x=X)
    assert caught.value.code == "constant_outcome" and "insured" in str(caught.value)
    for arguments, text in (({"y2": "work"}, "different"), ({"x2": ["x1", "insured"]}, "own equation"),
                            ({"x2": []}, "x2"), ({"x2": "x1"}, "x2"), ({"covariance": "HC1"}, "covariance"),
                            ({"y2": None}, "outcome2")):
        with pytest.raises(AnalysisError) as caught:
            oe.biprobit(**{"data": data, "y1": "work", "y2": "insured", "x": X, **arguments})
        assert caught.value.code == "invalid_spec" and text in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=data, y1="work", y2="nope", x=X)
    assert caught.value.code == "missing_columns"
    with pytest.raises(AnalysisError) as caught:
        oe.biprobit(data=data.assign(aw=-data.aw), y1="work", y2="insured", x=X, weights="aw",
                    weight_type="iweight")
    assert caught.value.code == "negative_weights"
    for fields in [{"columns": {}}, {"columns": {"outcome2": ["insured", "y"]}},
                   {"columns": {"outcome2": "insured", "het": ["z1"]}},
                   {"columns": {"outcome2": "insured"}, "options": {"base": 1}},
                   {"columns": {"outcome2": "insured"}, "covariance": "HC3"}]:
        with pytest.raises(ValidationError):
            ModelSpec(estimator="biprobit", outcome="work", predictors=X, **fields)


# ---- rendering and scale -------------------------------------------------------------------


def test_binary_models_round_trip_summary_and_latex(data):
    het = oe.hetprobit(data=data, y="y", x=["x1", "sector"], het=Z, categorical=["sector"], cluster="firm")
    bivariate = oe.biprobit(data=data, y1="work", y2="insured", x=X, x2=["x1", "z1"], covariance="robust")
    for result in (het, bivariate):
        restored = type(result).model_validate_json(result.model_dump_json())
        assert restored == result and isinstance(restored, ResultBundle)
        assert ModelSpec.model_validate(json.loads(result.spec.model_dump_json())) == result.spec
        assert fit(result.spec, data=data).spec == result.spec
        assert result.provenance["optimizer"]["converged"] is True
        assert result.provenance["solver"] == "newton_observed_hessian"
        assert result.provenance["family"] == "discrete"
        assert result.provenance["stata_parity_validated"] is False
        assert len(result.predictions) == 400
    text = het.summary()
    assert "Heteroskedastic probit regression — y" in text and "Covariance: cluster" in text
    assert "[lnsigma]" in text and "lnsigma:z1" in text and "sector[b]" in text
    assert "Wald test of lnsigma = 0 (homoskedastic probit): chi2(2)" in text
    assert "lnsigma:z2" in str(het.to_latex()).replace(r"\_", "_")
    text = bivariate.summary()
    assert "Seemingly unrelated bivariate probit — work" in text and "/athrho" in text
    assert "[insured]" in text and "insured:z1" in text and "rho: 0." in text
    assert "Wald test of rho = 0: chi2(1)" in text
    assert "athrho" in str(bivariate.to_latex())
    payload = json.loads(bivariate.model_dump_json())
    assert set(payload["extra"]["rho"]) >= {"estimate", "std_error", "ci_low", "ci_high"}
    assert payload["spec"]["columns"] == {"outcome2": "insured", "predictors2": ["x1", "z1"]}
    assert oe.capabilities()["estimators"]["hetprobit"]["columns"]["het"]["required"] is True


def test_binary_models_scale_linearly():
    rng = np.random.default_rng(12)
    n = 200_000
    frame = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    columns = list(frame.columns)
    beta = np.linspace(-0.5, 0.5, 6)
    index = frame.to_numpy() @ beta
    frame["y"] = (0.2 + index + np.exp(0.4 * frame.x0 - 0.3 * frame.x1) * rng.normal(size=n) > 0) * 1.0
    shocks = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
    frame["a"] = (0.2 + index + shocks[:, 0] > 0) * 1.0
    frame["b"] = (-0.1 - 0.5 * index + shocks[:, 1] > 0) * 1.0
    frame["g"] = rng.integers(0, 3000, size=n)
    start = time.perf_counter()
    het = oe.hetprobit(data=frame, y="y", x=columns, het=["x0", "x1"], cluster="g")
    bivariate = oe.biprobit(data=frame, y1="a", y2="b", x=columns)
    elapsed = time.perf_counter() - start
    assert het.nobs == n == bivariate.nobs and elapsed < 30
    assert_allclose(estimates(het)[1:7], beta, atol=0.03)
    assert_allclose(estimates(het)[7:], [0.4, -0.3], atol=0.03)
    assert_allclose(estimates(bivariate)[1:7], beta, atol=0.03)
    assert_allclose(bivariate.metrics["rho"], 0.6, atol=0.02)
