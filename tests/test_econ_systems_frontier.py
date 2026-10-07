"""Independent oracles for the stochastic frontier (oe.frontier, oe.frontier_efficiency).

The three log likelihoods are rewritten here with scipy.stats and maximized by
brute force (scipy.optimize), the covariance is the inverse of a numerically
differentiated Hessian (statsmodels approx_hess), sandwich covariances use
numerically differentiated per-observation scores, and the efficiency scores
are integrated numerically from the conditional density of u given e.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import integrate, stats
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.systems.frontier_kernels import FrontierObjective
from openecon.engines.optimize import check_derivatives


def make_frontier(seed=4, n=400):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    v = rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "g": rng.integers(0, 30, n),
                          "f": rng.integers(1, 4, n).astype(float)})
    base = 1 + 0.5 * x1 + 0.3 * x2 + v
    frame["yh"] = base - np.abs(rng.normal(scale=0.6, size=n))
    frame["ye"] = base - rng.exponential(0.5, size=n)
    frame["yt"] = base - np.abs(0.3 + rng.normal(scale=0.6, size=n))
    frame["yc"] = base + np.abs(rng.normal(scale=0.6, size=n))
    return frame


@pytest.fixture(scope="module")
def data():
    return make_frontier()


def loglik(theta, y, x, dist, s=1.0):
    k = x.shape[1]
    e = y - x @ theta[:k]
    if dist == "hnormal":
        sv2, su2 = np.exp(theta[k]), np.exp(theta[k + 1])
        sig = np.sqrt(sv2 + su2)
        lam = np.sqrt(su2 / sv2)
        return (np.log(2) - np.log(sig) + stats.norm.logpdf(e / sig)
                + stats.norm.logcdf(-s * e * lam / sig))
    if dist == "exponential":
        sv, su = np.exp(theta[k] / 2), np.exp(theta[k + 1] / 2)
        return (-np.log(su) + sv ** 2 / (2 * su ** 2) + s * e / su
                + stats.norm.logcdf(-s * e / sv - sv / su))
    mu, sig2, gam = theta[k], np.exp(theta[k + 1]), 1 / (1 + np.exp(-theta[k + 2]))
    su2, sv2 = gam * sig2, (1 - gam) * sig2
    mustar = (mu * sv2 - s * e * su2) / sig2
    sstar = np.sqrt(su2 * sv2 / sig2)
    return (stats.norm.logpdf((s * e + mu) / np.sqrt(sig2)) - 0.5 * np.log(sig2)
            + stats.norm.logcdf(mustar / sstar) - stats.norm.logcdf(mu / np.sqrt(su2)))


def oracle(frame, ycol, dist, s=1.0, w=None):
    x = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    y = frame[ycol].to_numpy()
    w = np.ones(len(y)) if w is None else w
    start = np.r_[np.linalg.lstsq(x, y, rcond=None)[0], -2.0, -1.0] if dist != "tnormal" \
        else np.r_[np.linalg.lstsq(x, y, rcond=None)[0], 0.0, -1.0, 0.0]
    fun = lambda t: -(w * loglik(t, y, x, dist, s)).sum()  # noqa: E731
    opt = minimize(fun, start, method="BFGS", options={"gtol": 1e-9, "maxiter": 20000})
    opt = minimize(fun, opt.x, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    opt = minimize(fun, opt.x, method="BFGS", options={"gtol": 1e-10})
    hess = approx_hess(opt.x, lambda t: (w * loglik(t, y, x, dist, s)).sum())
    return opt.x, np.linalg.inv(-hess), -opt.fun, x, y


def est(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


@pytest.mark.parametrize("dist,ycol", [("hnormal", "yh"), ("exponential", "ye"),
                                       ("tnormal", "yt")])
def test_maximum_likelihood_matches_brute_force(data, dist, ycol):
    theta, cov, ll, _, _ = oracle(data, ycol, dist)
    result = oe.frontier(data=data, y=ycol, x=["x1", "x2"], distribution=dist)
    b, se = est(result)
    tol = 1e-4 if dist == "tnormal" else 1e-5
    assert_allclose(b, theta, rtol=tol, atol=tol)
    assert_allclose(se, np.sqrt(np.diag(cov)), rtol=1e-3)
    assert_allclose(result.metrics["log_likelihood"], ll, rtol=1e-9)
    names = [c.term for c in result.coefficients]
    if dist == "tnormal":
        assert names[3:] == ["/mu", "/lnsigma2", "/ilgtgamma"]
        sig2, gam = np.exp(b[4]), 1 / (1 + np.exp(-b[5]))
        assert_allclose(result.metrics["gamma"], gam, rtol=1e-12)
        assert_allclose(result.metrics["sigma_u2"], gam * sig2, rtol=1e-12)
        grad = np.array([0, gam * sig2, gam * (1 - gam) * sig2])
        se_su2 = np.sqrt(grad @ np.array(result.covariance_matrix)[3:, 3:] @ grad)
        assert_allclose(result.extra["ancillary"]["sigma_u2"]["std_error"], se_su2, rtol=1e-10)
    else:
        assert names[3:] == ["/lnsig2v", "/lnsig2u"]
        sv, su = np.exp(b[3] / 2), np.exp(b[4] / 2)
        assert_allclose(result.metrics["lambda"], su / sv, rtol=1e-12)
        assert_allclose(result.metrics["sigma2"], sv ** 2 + su ** 2, rtol=1e-12)
        grad = np.array([-su / sv / 2, su / sv / 2])
        se_l = np.sqrt(grad @ np.array(result.covariance_matrix)[3:, 3:] @ grad)
        assert_allclose(result.extra["ancillary"]["lambda"]["std_error"], se_l, rtol=1e-10)
    # LR test of sigma_u = 0 against OLS, chibar2(01).
    x = np.column_stack([np.ones(len(data)), data.x1, data.x2])
    e = data[ycol].to_numpy() - x @ np.linalg.lstsq(x, data[ycol], rcond=None)[0]
    ll_ols = stats.norm.logpdf(e, scale=np.sqrt(e @ e / len(e))).sum()
    lr = 2 * (ll - ll_ols)
    assert_allclose(result.tests["sigma_u"]["statistic"], lr, rtol=1e-6)
    assert_allclose(result.tests["sigma_u"]["p_value"], 0.5 * stats.chi2.sf(lr, 1), rtol=1e-5)
    assert result.inference["use_t"] is False


def test_cost_frontier_and_efficiency(data):
    theta, cov, ll, x, y = oracle(data, "yc", "hnormal", s=-1.0)
    result = oe.frontier(data=data, y="yc", x=["x1", "x2"], cost=True)
    assert_allclose(est(result)[0], theta, rtol=1e-5, atol=1e-5)
    eff = oe.frontier_efficiency(result, data)
    b = est(result)[0]
    sv2, su2 = np.exp(b[3]), np.exp(b[4])
    e = y - x @ b[:3]
    for i in (0, 5, 17):
        # Conditional density of u given e, integrated numerically (cost: e = v + u).
        dens = lambda u, ei=e[i]: stats.norm.pdf(ei - u, scale=np.sqrt(sv2)) * \
            stats.halfnorm.pdf(u, scale=np.sqrt(su2))  # noqa: E731
        norm = integrate.quad(dens, 0, np.inf)[0]
        mean = integrate.quad(lambda u: u * dens(u), 0, np.inf)[0] / norm
        te = integrate.quad(lambda u: np.exp(u) * dens(u), 0, 30)[0] / norm
        assert_allclose(eff["u"][i], mean, rtol=1e-7)
        assert_allclose(eff["te"][i], te, rtol=1e-7)
        assert eff["row"][i] == i
    assert eff.attrs["frontier"] == "cost"


@pytest.mark.parametrize("dist,ycol", [("exponential", "ye"), ("tnormal", "yt")])
def test_efficiency_production(data, dist, ycol):
    result = oe.frontier(data=data, y=ycol, x=["x1", "x2"], distribution=dist)
    eff = oe.frontier_efficiency(result, data)
    b = est(result)[0]
    x = np.column_stack([np.ones(len(data)), data.x1, data.x2])
    e = data[ycol].to_numpy() - x @ b[:3]
    if dist == "exponential":
        sv, su = np.exp(b[3] / 2), np.exp(b[4] / 2)
        prior = lambda u: stats.expon.pdf(u, scale=su)  # noqa: E731
    else:
        mu, sig2, gam = b[3], np.exp(b[4]), 1 / (1 + np.exp(-b[5]))
        sv, su = np.sqrt((1 - gam) * sig2), np.sqrt(gam * sig2)
        prior = lambda u: stats.truncnorm.pdf(u, -mu / su, np.inf, loc=mu, scale=su)  # noqa: E731
    for i in (1, 9):
        dens = lambda u, ei=e[i]: stats.norm.pdf(ei + u, scale=sv) * prior(u)  # noqa: E731
        norm = integrate.quad(dens, 0, np.inf)[0]
        mean = integrate.quad(lambda u: u * dens(u), 0, np.inf)[0] / norm
        te = integrate.quad(lambda u: np.exp(-u) * dens(u), 0, np.inf)[0] / norm
        assert_allclose(eff["u"][i], mean, rtol=1e-6)
        assert_allclose(eff["te"][i], te, rtol=1e-6)
    assert (eff["te"] < 1).all() and (eff["u_mode"] >= 0).all()


@pytest.mark.parametrize("dist", ["hnormal", "exponential", "tnormal"])
@pytest.mark.parametrize("cost", [False, True])
def test_analytic_derivatives(data, dist, cost):
    x = torch.tensor(np.column_stack([np.ones(len(data)), data.x1, data.x2]))
    y = torch.tensor(data.yh.to_numpy())
    w = torch.tensor(data.f.to_numpy())
    objective = FrontierObjective(x, y, w, dist, cost)
    theta = torch.tensor([0.9, 0.4, 0.2, -2.0, -1.0] if dist != "tnormal"
                         else [0.9, 0.4, 0.2, 0.3, -0.8, 0.5], dtype=torch.float64)
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-8
    assert report["hessian_max_rel_error"] < 1e-7
    rows = objective.score_rows(theta)
    assert_allclose((w[:, None] * rows).sum(0).numpy(), objective(theta)[1].numpy(), rtol=1e-10)


def test_sandwich_covariances(data):
    theta, _, _, x, y = oracle(data, "yh", "hnormal")
    result = oe.frontier(data=data, y="yh", x=["x1", "x2"])
    b = est(result)[0]
    hess = approx_hess(b, lambda t: loglik(t, y, x, "hnormal").sum())
    scores = approx_fprime(b, lambda t: loglik(t, y, x, "hnormal"), centered=True)
    bread = np.linalg.inv(-hess)
    n = len(y)
    robust = oe.frontier(data=data, y="yh", x=["x1", "x2"], covariance="robust")
    v = n / (n - 1) * bread @ scores.T @ scores @ bread
    assert_allclose(est(robust)[1], np.sqrt(np.diag(v)), rtol=1e-4)
    assert "sigma_u" not in robust.tests
    opg = oe.frontier(data=data, y="yh", x=["x1", "x2"], covariance="opg")
    assert_allclose(est(opg)[1], np.sqrt(np.diag(np.linalg.inv(scores.T @ scores))), rtol=1e-4)
    cluster = oe.frontier(data=data, y="yh", x=["x1", "x2"], cluster="g")
    sums = pd.DataFrame(scores).groupby(data.g.to_numpy()).sum().to_numpy()
    groups = len(sums)
    v = groups / (groups - 1) * bread @ sums.T @ sums @ bread
    assert_allclose(est(cluster)[1], np.sqrt(np.diag(v)), rtol=1e-4)


def test_weights_and_offsets(data):
    expanded = data.loc[data.index.repeat(data.f.astype(int))].reset_index(drop=True)
    weighted = oe.frontier(data=data, y="yh", x=["x1", "x2"], weights="f", weight_type="fweight")
    duplicated = oe.frontier(data=expanded, y="yh", x=["x1", "x2"])
    assert weighted.nobs == len(expanded)
    assert_allclose(est(weighted)[0], est(duplicated)[0], rtol=1e-8)
    assert_allclose(est(weighted)[1], est(duplicated)[1], rtol=1e-7)
    assert_allclose(weighted.metrics["log_likelihood"], duplicated.metrics["log_likelihood"],
                    rtol=1e-10)
    pw = oe.frontier(data=data, y="yh", x=["x1", "x2"], weights="f", weight_type="pweight")
    assert pw.spec.covariance == "robust"
    assert_allclose(est(pw)[0], est(weighted)[0], rtol=1e-8)
    shifted = data.assign(x1=data.x1 + 1e6)
    big = oe.frontier(data=shifted, y="yh", x=["x1", "x2"])
    base = oe.frontier(data=data, y="yh", x=["x1", "x2"])
    assert_allclose(est(big)[0][1:], est(base)[0][1:], rtol=1e-7)
    assert_allclose(est(big)[1][1:], est(base)[1][1:], rtol=1e-6)


def test_boundary_and_errors(data):
    wrong = data.assign(yw=data.yc)
    with pytest.raises(AnalysisError) as error:
        oe.frontier(data=wrong, y="yw", x=["x1", "x2"])
    assert error.value.code == "boundary_solution"
    with pytest.raises(AnalysisError) as error:
        oe.frontier(data=data, y="yh", x=["x1", "x2"], cost=True)
    assert error.value.code == "boundary_solution"
    with pytest.raises(AnalysisError) as error:
        oe.frontier(data=data, y="yh", x=["x1", "x2"], weights="f", weight_type="pweight",
                    covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"
    with pytest.raises(AnalysisError) as error:
        oe.frontier(data=data, y="yh", x=["x1"], distribution="gamma")
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.frontier(data=data, y="yh", x=["x1"], weights="f", weight_type="aweight")
    assert error.value.code == "invalid_spec"
    result = oe.frontier(data=data, y="yh", x=["x1", "x2"])
    with pytest.raises(AnalysisError) as error:
        oe.frontier_efficiency(result, data.head(50))
    assert error.value.code == "sample_mismatch"
    with pytest.raises(AnalysisError) as error:
        oe.frontier_efficiency(oe.sureg(data=data, equations=[
            {"y": "yh", "x": ["x1"]}, {"y": "ye", "x": ["x2"]}]), data)
    assert error.value.code == "invalid_result"


def test_round_trip_collinearity_missing(data):
    frame = data.assign(dup=2 * data.x1)
    result = oe.frontier(data=frame, y="yh", x=["x1", "dup", "x2"])
    assert "dup" not in [c.term for c in result.coefficients]
    assert result.provenance["omitted_terms"] == ["dup"]
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert json.loads(result.model_dump_json())["extra"]["distribution"] == "hnormal"
    text = result.summary()
    assert "/lnsig2u" in text and "chibar2" in text
    assert "lnsig2u" in result.to_latex()
    missing = data.copy()
    missing.loc[2, "x2"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.frontier(data=missing, y="yh", x=["x1", "x2"])
    assert error.value.code == "missing_values"
    dropped = oe.frontier(data=missing, y="yh", x=["x1", "x2"], missing="drop")
    eff = oe.frontier_efficiency(dropped, missing)
    assert len(eff) == len(data) - 1 and 2 not in list(eff["row"])
    assert callable(oe.frontier_efficiency) and oe.frontier.__doc__


def test_importance_weights(data):
    iw = oe.frontier(data=data, y="ye", x=["x1", "x2"], distribution="exponential",
                     weights="f", weight_type="iweight")
    fw = oe.frontier(data=data, y="ye", x=["x1", "x2"], distribution="exponential",
                     weights="f", weight_type="fweight")
    assert_allclose(est(iw)[0], est(fw)[0], rtol=1e-9)
    assert_allclose(iw.metrics["log_likelihood"], fw.metrics["log_likelihood"], rtol=1e-12)
    theta, cov, ll, _, _ = oracle(data, "ye", "exponential", w=data.f.to_numpy())
    assert_allclose(est(iw)[0], theta, rtol=1e-5, atol=1e-6)
    assert_allclose(est(iw)[1], np.sqrt(np.diag(cov)), rtol=1e-3)
