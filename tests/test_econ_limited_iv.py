"""ivprobit and ivtobit (maximum likelihood and Newey's two-step) against independent oracles.

The ML estimators are compared with brute-force maximization of a NumPy
likelihood written directly in Stata's parameterization (coefficients, reduced
forms, atanh correlations and log standard deviations of the full error
covariance matrix), which is unrelated to the recursive working
parameterization of the kernel; covariances are rebuilt from complex-step
scores and numerical Hessians. The two-step estimator is compared with Newey's
minimum chi-squared formula coded in NumPy on statsmodels probits (and a
brute-force tobit).
"""

import time

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import special, stats
from test_econ_limited import (
    KINDS, brute_force, check_fit, covariance, errors, estimates, hessian_of, round_trip,
    terms, tobit_loglik,
)

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics.limited.iv_kernels import IVObjective, ProbitRows
from openecon.econometrics.limited.kernels import CensoredRows
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec

EXOG = ["x1"]
INSTRUMENTS = ["z1", "z2", "z3"]


# ---- independent likelihood -----------------------------------------------------------


def iv_loglik(theta, w, z, y2, y, limits):
    """Joint log likelihood per observation in Stata's parameterization.

    ``theta = (d, Pi rows, athrho for every pair of equations (i > j), lnsigma)``;
    equation 1 is the outcome equation. ``limits=None`` is the probit model
    (``var(u) = 1``, no ``lnsigma1``); otherwise ``limits = (ll, ul)`` of a tobit.
    """
    kw, kz, p = w.shape[1], z.shape[1], y2.shape[1]
    delta = theta[:kw]
    pi = theta[kw:kw + p * kz].reshape(p, kz)
    pairs = p * (p + 1) // 2
    athrho = theta[kw + p * kz:kw + p * kz + pairs]
    log_sd = theta[kw + p * kz + pairs:]
    if limits is None:
        log_sd = np.concatenate([np.zeros(1, dtype=theta.dtype), log_sd])
    correlation = np.eye(p + 1, dtype=theta.dtype)
    position = 0
    for i in range(1, p + 1):
        for j in range(i):
            correlation[i, j] = correlation[j, i] = np.tanh(athrho[position])
            position += 1
    sd = np.exp(log_sd)
    sigma = correlation * np.outer(sd, sd)
    v = y2 - z @ pi.T
    inverse = np.linalg.inv(sigma[1:, 1:])
    slope = inverse @ sigma[1:, 0]
    omega = np.sqrt(sigma[0, 0] - sigma[0, 1:] @ slope)
    mean = w @ delta + v @ slope
    normal = -0.5 * np.einsum("ij,jk,ik->i", v, inverse, v) \
        - 0.5 * np.log(np.linalg.det(sigma[1:, 1:])) - 0.5 * p * np.log(2 * np.pi)
    if limits is None:
        return np.log(special.ndtr((2 * y - 1) * mean / omega)) + normal
    ll, ul = limits
    left = np.zeros(len(y), bool) if ll is None else y <= ll
    right = np.zeros(len(y), bool) if ul is None else y >= ul
    density = -0.5 * ((y - mean) / omega) ** 2 - np.log(omega) - 0.5 * np.log(2 * np.pi)
    lower = np.log(special.ndtr(((ll or 0.0) - mean) / omega))
    upper = np.log(special.ndtr((mean - (ul or 0.0)) / omega))
    return np.where(left, lower, np.where(right, upper, density)) + normal


# ---- data -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20263)
    n = 800
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "z1": rng.normal(size=n), "z2": rng.normal(size=n),
        "z3": rng.uniform(-1, 1, size=n), "firm": np.repeat(np.arange(80), 10),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "group": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
    })
    shocks = rng.multivariate_normal(
        [0, 0, 0], [[1.0, 0.45, -0.3], [0.45, 1.2, 0.25], [-0.3, 0.25, 0.8]], size=n)
    frame["e1"] = 0.4 * frame.x1 + 0.8 * frame.z1 - 0.3 * frame.z2 + shocks[:, 1]
    frame["e2"] = -0.2 * frame.x1 + 0.3 * frame.z2 + 0.9 * frame.z3 + shocks[:, 2]
    index = 0.2 + 0.6 * frame.x1 - 0.7 * frame.e1
    frame["y"] = (index + shocks[:, 0] > 0) * 1.0
    frame["yy"] = (index + 0.5 * frame.e2 + shocks[:, 0] > 0) * 1.0
    frame["t"] = (0.5 + index + 0.5 * frame.e2 + 1.3 * shocks[:, 0]).clip(lower=0.0, upper=3.0)
    return frame


def blocks(frame, endog, exog=EXOG, instruments=INSTRUMENTS):
    ones = np.ones(len(frame))
    x1 = np.column_stack([ones, frame[exog].to_numpy(dtype=float)])
    y2 = frame[endog].to_numpy(dtype=float)
    z = np.column_stack([x1, frame[instruments].to_numpy(dtype=float)])
    return np.column_stack([x1, y2]), z, y2


def start_values(w, z, y2, tobit=False):
    """Oracle starting point: zero slopes, first-stage coefficients, independent errors."""
    p = y2.shape[1]
    pi = np.linalg.lstsq(z, y2, rcond=None)[0]
    log_sd = np.log((y2 - z @ pi).std(axis=0))
    return np.r_[np.zeros(w.shape[1]), pi.T.reshape(-1), np.zeros(p * (p + 1) // 2),
                 np.zeros(1) if tobit else [], log_sd]


def ivprobit(frame, endog=("e1",), **options):
    return oe.ivprobit(data=frame, y="y" if len(endog) == 1 else "yy", x=EXOG, endog=list(endog),
                       instruments=INSTRUMENTS, **options)


def ivtobit(frame, endog=("e1", "e2"), **options):
    return oe.ivtobit(data=frame, y="t", x=EXOG, endog=list(endog), instruments=INSTRUMENTS,
                      ll=0.0, ul=3.0, **options)


@pytest.fixture(scope="module")
def solutions(data):
    """Brute-force maxima of the three benchmark models (shared by the covariance cases)."""
    ones = np.ones(len(data))
    out = {}
    w, z, y2 = blocks(data, ["e1"])
    out["probit1"] = brute_force(iv_loglik, start_values(w, z, y2), ones, w, z, y2,
                                 data.y.to_numpy(), None)
    w, z, y2 = blocks(data, ["e1", "e2"])
    out["probit2"] = brute_force(iv_loglik, start_values(w, z, y2), ones, w, z, y2,
                                 data.yy.to_numpy(), None)
    out["tobit2"] = brute_force(iv_loglik, start_values(w, z, y2, True), ones, w, z, y2,
                                data.t.to_numpy(), (0.0, 3.0))
    return out


# ---- maximum likelihood ---------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_ivprobit_ml_matches_brute_force_likelihood(data, solutions, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = ivprobit(data, **options)
    w, z, y2 = blocks(data, ["e1"])
    theta, log_likelihood = check_fit(result, iv_loglik, None, w, z, y2, data.y.to_numpy(), None,
                                      groups=data.firm, solution=solutions["probit1"])
    assert terms(result) == ["Intercept", "x1", "e1", "e1:Intercept", "e1:x1", "e1:z1", "e1:z2",
                             "e1:z3", "/athrho2_1", "/lnsigma2"]
    assert [c.equation for c in result.coefficients] == ["y"] * 3 + ["e1"] * 5 + [None] * 2
    v, metrics = covariance(result), result.metrics
    assert_allclose(metrics["aic"], -2 * log_likelihood + 20, rtol=1e-10)
    assert_allclose(metrics["bic"], -2 * log_likelihood + 10 * np.log(len(data)), rtol=1e-10)
    model = result.tests["model"]
    assert model["df"] == 2
    assert_allclose(model["statistic"], theta[1:3] @ np.linalg.solve(v[1:3, 1:3], theta[1:3]),
                    rtol=1e-5)
    exogeneity = result.tests["exogeneity"]
    assert exogeneity["df"] == 1 and exogeneity["distribution"] == "chi2"
    assert_allclose(exogeneity["statistic"], theta[8] ** 2 / v[8, 8], rtol=1e-5)
    assert_allclose(exogeneity["p_value"], stats.chi2.sf(exogeneity["statistic"], 1), rtol=1e-6)
    corr = result.extra["correlations"]["/athrho2_1"]
    assert corr["label"] == "corr(e.e1, e.y)"
    assert_allclose(corr["estimate"], np.tanh(theta[8]), rtol=1e-6)
    assert_allclose(corr["std_error"], (1 - np.tanh(theta[8]) ** 2) * np.sqrt(v[8, 8]), rtol=1e-9)
    sd = result.extra["standard_deviations"]["/lnsigma2"]
    assert sd["label"] == "sd(e.e1)"
    assert_allclose(sd["estimate"], np.exp(theta[9]), rtol=1e-6)
    assert result.extra["instruments"] == INSTRUMENTS and result.extra["n_instruments"] == 3
    assert not result.inference["use_t"] and len(result.predictions) == 400


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "cluster"])
def test_ivprobit_ml_with_two_endogenous_regressors(data, solutions, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = ivprobit(data, endog=("e1", "e2"), **options)
    w, z, y2 = blocks(data, ["e1", "e2"])
    theta, _ = check_fit(result, iv_loglik, None, w, z, y2, data.yy.to_numpy(), None,
                         groups=data.firm, solution=solutions["probit2"], atol=2e-8)
    assert terms(result)[-5:] == ["/athrho2_1", "/athrho3_1", "/athrho3_2", "/lnsigma2",
                                  "/lnsigma3"]
    assert terms(result)[:4] == ["Intercept", "x1", "e1", "e2"]
    assert [c.equation for c in result.coefficients][4:14] == ["e1"] * 5 + ["e2"] * 5
    v = covariance(result)
    exogeneity = result.tests["exogeneity"]
    assert exogeneity["df"] == 2
    assert_allclose(exogeneity["statistic"],
                    theta[14:16] @ np.linalg.solve(v[14:16, 14:16], theta[14:16]), rtol=1e-5)
    labels = {name: record["label"] for name, record in result.extra["correlations"].items()}
    assert labels == {"/athrho2_1": "corr(e.e1, e.yy)", "/athrho3_1": "corr(e.e2, e.yy)",
                      "/athrho3_2": "corr(e.e2, e.e1)"}


@pytest.mark.parametrize("kind", KINDS)
def test_ivtobit_ml_matches_brute_force_likelihood(data, solutions, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = ivtobit(data, **options)
    w, z, y2 = blocks(data, ["e1", "e2"])
    y = data.t.to_numpy()
    theta, log_likelihood = check_fit(result, iv_loglik, None, w, z, y2, y, (0.0, 3.0),
                                      groups=data.firm, solution=solutions["tobit2"], atol=2e-8)
    assert terms(result)[-6:] == ["/athrho2_1", "/athrho3_1", "/athrho3_2", "/lnsigma1",
                                  "/lnsigma2", "/lnsigma3"]
    metrics = result.metrics
    assert metrics["n_left_censored"] == (y <= 0).sum()
    assert metrics["n_right_censored"] == (y >= 3).sum()
    assert metrics["n_uncensored"] == ((y > 0) & (y < 3)).sum()
    assert_allclose(metrics["sigma"], np.exp(theta[17]), rtol=1e-6)
    assert_allclose(metrics["aic"], -2 * log_likelihood + 2 * 20, rtol=1e-10)
    v = covariance(result)
    exogeneity = result.tests["exogeneity"]
    assert_allclose(exogeneity["statistic"],
                    theta[14:16] @ np.linalg.solve(v[14:16, 14:16], theta[14:16]), rtol=1e-5)
    assert result.extra["limits"] == {"lower": 0.0, "upper": 3.0}
    assert not result.inference["use_t"]


def test_ivtobit_with_one_endogenous_regressor_and_one_limit(data):
    result = oe.ivtobit(data=data, y="t", x=EXOG, endog=["e1"], instruments=INSTRUMENTS, ll="min")
    w, z, y2 = blocks(data, ["e1"])
    y = data.t.to_numpy()
    check_fit(result, iv_loglik, start_values(w, z, y2, True), w, z, y2, y, (y.min(), None))
    assert terms(result)[-3:] == ["/athrho2_1", "/lnsigma1", "/lnsigma2"]
    assert result.metrics["n_right_censored"] == 0


def test_iv_ml_weights_follow_stata_semantics(data):
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for options in ({"covariance": "nonrobust"}, {"covariance": "robust"}, {"cluster": "firm"},
                    {"covariance": "opg"}):
        for model in (ivprobit, ivtobit):
            weighted = model(data, weights="fw", weight_type="fweight", **options)
            replicated = model(expanded, **options)
            assert weighted.nobs == replicated.nobs == int(data.fw.sum())
            assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-7, atol=1e-9)
            assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-6, atol=1e-11)
            assert_allclose(weighted.tests["exogeneity"]["statistic"],
                            replicated.tests["exogeneity"]["statistic"], rtol=1e-6)
    w, z, y2 = blocks(data, ["e1"])
    aw = data.aw.to_numpy()
    analytic = ivprobit(data, weights="aw", weight_type="aweight")
    check_fit(analytic, iv_loglik, start_values(w, z, y2), w, z, y2, data.y.to_numpy(), None,
              weights=aw * len(aw) / aw.sum())
    sampling = ivprobit(data, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    check_fit(sampling, iv_loglik, start_values(w, z, y2), w, z, y2, data.y.to_numpy(), None,
              weights=aw)
    with pytest.raises(AnalysisError) as error:
        ivprobit(data, weights="aw", weight_type="pweight", covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"


def test_just_identified_ml_equals_the_control_function_estimator(data):
    # With as many instruments as endogenous regressors the ML estimates are the first-stage
    # least squares and the control-function probit, rescaled by sd(u | v).
    result = oe.ivprobit(data=data, y="y", x=EXOG, endog=["e1"], instruments=["z1"])
    w, z, y2 = blocks(data, ["e1"], instruments=["z1"])
    first = sm.OLS(y2[:, 0], z).fit()
    control = sm.Probit(data.y.to_numpy(), np.column_stack([w, first.resid])).fit(
        disp=0, tol=1e-13)
    b = estimates(result)
    sigma = np.sqrt(first.ssr / len(data))
    omega = 1 / np.sqrt(1 + (control.params[3] * sigma) ** 2)
    assert_allclose(b[:3], control.params[:3] * omega, rtol=1e-7)
    assert_allclose(b[3:6], first.params, rtol=1e-8)
    assert_allclose(b[7], np.log(sigma), rtol=1e-8, atol=1e-10)
    assert_allclose(np.sinh(b[6]), control.params[3] * sigma, rtol=1e-7)
    assert_allclose(result.metrics["log_likelihood"],
                    control.llf + stats.norm.logpdf(first.resid, scale=sigma).sum(), rtol=1e-10)
    # The two-step estimator then reproduces the control-function coefficients themselves.
    two = oe.ivprobit(data=data, y="y", x=EXOG, endog=["e1"], instruments=["z1"],
                      method="twostep")
    assert_allclose(estimates(two), control.params[:3], rtol=1e-7)


# ---- Newey's two-step estimator -------------------------------------------------------


def newey_oracle(y, w, z, y2, k1, outcome_fit, weights=None):
    """Minimum chi-squared estimator from explicit algebra."""
    n = len(y) if weights is None else weights.sum()
    kz, p = z.shape[1], y2.shape[1]
    kw = k1 + p
    wls = np.ones(len(y)) if weights is None else weights
    gram = np.linalg.inv((z * wls[:, None]).T @ z)
    pi = gram @ (z * wls[:, None]).T @ y2
    v = y2 - z @ pi
    reduced, reduced_cov = outcome_fit(np.column_stack([z, v]))
    conditional, conditional_cov = outcome_fit(np.column_stack([w, v]))
    target = y2 @ (reduced[kz:kz + p] - conditional[k1:kw])
    coefficient = gram @ (z * wls[:, None]).T @ target
    s2 = (wls * (target - z @ coefficient) ** 2).sum() / (n - kz)
    omega = reduced_cov[:kz, :kz] + s2 * gram
    distance = np.column_stack([np.eye(kz)[:, :k1], pi])
    variance = np.linalg.inv(distance.T @ np.linalg.solve(omega, distance))
    delta = variance @ distance.T @ np.linalg.solve(omega, reduced[:kz])
    lam, lam_cov = conditional[kw:kw + p], conditional_cov[kw:kw + p, kw:kw + p]
    return delta, variance, lam @ np.linalg.solve(lam_cov, lam)


@pytest.mark.parametrize("endog", [("e1",), ("e1", "e2")])
def test_ivprobit_two_step_matches_neweys_formula(data, endog):
    result = ivprobit(data, endog=endog, method="twostep")
    y = data.y.to_numpy() if len(endog) == 1 else data.yy.to_numpy()
    w, z, y2 = blocks(data, list(endog))

    def probit(design):
        fitted = sm.Probit(y, design).fit(disp=0, tol=1e-13, maxiter=200)
        return np.asarray(fitted.params), np.asarray(fitted.cov_params())

    delta, variance, wald = newey_oracle(y, w, z, y2, 2, probit)
    assert terms(result) == ["Intercept", "x1", *endog]
    assert all(c.equation is None for c in result.coefficients)
    assert_allclose(estimates(result), delta, rtol=1e-7)
    assert_allclose(covariance(result), variance, rtol=1e-6)
    exogeneity = result.tests["exogeneity"]
    assert exogeneity["df"] == len(endog)
    assert_allclose(exogeneity["statistic"], wald, rtol=1e-6)
    model = result.tests["model"]
    assert_allclose(model["statistic"], delta[1:] @ np.linalg.solve(variance[1:, 1:], delta[1:]),
                    rtol=1e-6)
    assert "log_likelihood" not in result.metrics and "normalization" in result.extra
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.norm.sf(np.abs(delta / np.sqrt(np.diag(variance)))), rtol=1e-6,
                    atol=1e-300)
    # Consistent for the ML coefficients divided by sd(u | v): the ratios agree.
    ml = ivprobit(data, endog=endog)
    ratio = estimates(result) / estimates(ml)[:len(delta)]
    assert_allclose(ratio[1:], ratio[1:].mean(), rtol=0.1)
    round_trip(result)


def test_ivtobit_two_step_matches_neweys_formula(data):
    result = ivtobit(data, method="twostep")
    y = data.t.to_numpy()
    w, z, y2 = blocks(data, ["e1", "e2"])

    def tobit(design):
        k = design.shape[1]
        ones = np.ones(len(y))
        theta, _ = brute_force(tobit_loglik, np.r_[np.zeros(k), 1.0], ones, design, y, 0.0, 3.0)
        cov = np.linalg.inv(-hessian_of(tobit_loglik, theta, ones, design, y, 0.0, 3.0))
        return theta, cov

    delta, variance, wald = newey_oracle(y, w, z, y2, 2, tobit)
    assert_allclose(estimates(result), delta, rtol=1e-6)
    assert_allclose(covariance(result), variance, rtol=2e-5)
    assert_allclose(result.tests["exogeneity"]["statistic"], wald, rtol=2e-5)
    assert result.metrics["n_left_censored"] == (y <= 0).sum()
    # Two-step and ML estimate the same structural coefficients for the tobit model.
    ml = ivtobit(data)
    assert_allclose(estimates(result), estimates(ml)[:4], atol=3 * errors(ml)[:4].max())
    round_trip(result)


def test_two_step_frequency_weights_and_restrictions(data):
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for model in (ivprobit, ivtobit):
        weighted = model(data, method="twostep", weights="fw", weight_type="fweight")
        replicated = model(expanded, method="twostep")
        assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-7)
        assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-6)
        assert_allclose(weighted.tests["exogeneity"]["statistic"],
                        replicated.tests["exogeneity"]["statistic"], rtol=1e-6)
        assert weighted.nobs == int(data.fw.sum())
        for options, code in (({"covariance": "robust"}, "unsupported_covariance"),
                              ({"cluster": "firm"}, "unsupported_covariance"),
                              ({"weights": "aw", "weight_type": "aweight"},
                               "unsupported_weights")):
            with pytest.raises(AnalysisError) as error:
                model(data, method="twostep", **options)
            assert error.value.code == code


# ---- kernel ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("model", ["probit", "tobit"])
def test_iv_kernel_derivatives_and_reparameterization(data, model, p):
    endog = ["e1", "e2"][:p]
    _, z, y2 = blocks(data, endog)
    weights = torch.tensor(data.aw.to_numpy())
    if model == "probit":
        rows = ProbitRows(torch.tensor(data.yy.to_numpy()))
    else:
        y = data.t.to_numpy()
        rows = CensoredRows(torch.tensor(np.where(y <= 0, -np.inf, np.where(y >= 3, 3.0, y))),
                            torch.tensor(np.where(y >= 3, np.inf, np.where(y <= 0, 0.0, y))))
    objective = IVObjective(torch.tensor(z), torch.tensor(y2), 2, rows, weights)
    rng = np.random.default_rng(p)
    theta = torch.tensor(rng.normal(scale=0.3, size=objective.size))
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-8
    assert report["hessian_max_rel_error"] < 1e-7
    assert report["hessian_asymmetry"] < 1e-9
    value, gradient, _ = objective(theta)
    assert_allclose(float(objective.value(theta)), float(value), rtol=1e-12)
    assert_allclose((objective.score_rows(theta) * weights[:, None]).sum(dim=0).numpy(),
                    gradient.numpy(), rtol=1e-8, atol=1e-8)
    # The working likelihood equals the likelihood in Stata's parameterization.
    reported, jacobian = objective.structural(theta)
    w = np.column_stack([z[:, :2], y2])
    outcome = data.yy.to_numpy() if model == "probit" else data.t.to_numpy()
    expected = (data.aw.to_numpy() * iv_loglik(
        reported.numpy(), w, z, y2, outcome, None if model == "probit" else (0.0, 3.0))).sum()
    assert_allclose(float(value), expected, rtol=1e-11)
    numerical = np.empty((objective.size, objective.size))
    for column in range(objective.size):
        step = torch.zeros(objective.size, dtype=torch.float64)
        step[column] = 1e-6
        numerical[:, column] = ((objective.structural(theta + step)[0]
                                 - objective.structural(theta - step)[0]) / 2e-6).numpy()
    assert_allclose(jacobian.numpy(), numerical, rtol=1e-6, atol=1e-8)
    assert len(objective.ancillary_names()) == p * (p + 1) // 2 + p + (model == "tobit")


# ---- sample handling and failures -----------------------------------------------------


def test_iv_sample_collinearity_missing_values_and_options(data):
    reference = ivprobit(data)
    frame = data.assign(copy=2 * data.x1, zcopy=data.z1 - data.x1)
    result = oe.ivprobit(data=frame, y="y", x=["x1", "copy"], endog=["e1"],
                         instruments=["z1", "zcopy", "z2", "z3"])
    assert result.provenance["omitted_terms"] == ["copy", "zcopy"]
    assert result.extra["instruments"] == INSTRUMENTS
    assert_allclose(estimates(result), estimates(reference), rtol=1e-7, atol=1e-9)
    holes = data.copy()
    holes.loc[[2, 9], "z2"] = np.nan
    holes.loc[4, "e1"] = np.nan
    with pytest.raises(AnalysisError) as error:
        ivprobit(holes)
    assert error.value.code == "missing_values"
    dropped = ivprobit(holes, missing="drop")
    assert dropped.nobs == len(data) - 3
    assert_allclose(estimates(dropped), estimates(ivprobit(holes.dropna(subset=["z2", "e1"]))),
                    rtol=1e-10)
    # No exogenous regressors, no constant, categorical instruments.
    bare = oe.ivprobit(data=data, y="y", endog=["e1"], instruments=["z1", "z2"],
                       intercept=False)
    assert terms(bare) == ["e1", "e1:z1", "e1:z2", "/athrho2_1", "/lnsigma2"]
    coded = oe.ivprobit(data=data, y="y", x=EXOG, endog=["e1"], instruments=["z1", "group"],
                        categorical=["group"])
    assert coded.extra["instruments"] == ["z1", "group[b]", "group[c]"]
    spec = ModelSpec(estimator="ivprobit", outcome="y", predictors=EXOG,
                     columns={"endogenous": ["e1"], "instruments": INSTRUMENTS})
    assert_allclose(estimates(fit(spec, data=data)), estimates(reference), rtol=1e-12)
    first = reference.extra["first_stage"]["e1"]
    w, z, y2 = blocks(data, ["e1"])
    full = sm.OLS(y2[:, 0], z).fit()
    test = full.f_test(np.eye(5)[2:])
    assert_allclose(first["f_statistic"], float(test.fvalue), rtol=1e-8)
    assert_allclose(first["f_p_value"], float(test.pvalue), rtol=1e-6, atol=1e-300)
    assert_allclose(first["r_squared"], full.rsquared, rtol=1e-10)
    assert_allclose(first["adjusted_r_squared"], full.rsquared_adj, rtol=1e-10)
    assert_allclose(first["rmse"], np.sqrt(full.mse_resid), rtol=1e-10)
    assert first["f_df1"] == 3 and first["f_df2"] == len(data) - 5


def test_iv_error_codes(data):
    def code(function=oe.ivprobit, frame=data, **arguments):
        base = {"y": "y", "x": EXOG, "endog": ["e1"], "instruments": INSTRUMENTS}
        with pytest.raises(AnalysisError) as error:
            function(data=frame, **{**base, **arguments})
        return error.value.code

    assert code(instruments=["x1", "z1"]) == "invalid_spec"
    assert code(instruments=["e1"]) == "invalid_spec"
    assert code(endog=["x1"]) == "invalid_spec"
    assert code(endog=["y"]) == "invalid_spec"
    assert code(endog=[]) == "invalid_spec"
    assert code(instruments=[]) == "invalid_spec"
    assert code(endog="e1") == "invalid_spec"
    assert code(method="liml") == "invalid_spec"
    assert code(endog=["e1", "e2"], instruments=["z1"]) == "underidentified"
    assert code(frame=data.assign(dup=3 * data.x1 + 1), instruments=["dup"]) == "underidentified"
    exact = data.assign(e1=data.x1 + data.z1)
    assert code(frame=exact) == "collinear_endogenous"
    assert code(frame=data.assign(y=data.y + 1)) == "invalid_binary_outcome"
    assert code(frame=data.assign(y=1.0)) == "constant_outcome"
    assert code(oe.ivtobit, y="t", ll=5.0, ul=1.0) == "invalid_limits"
    assert code(oe.ivtobit, y="t", ll=50.0) == "no_uncensored_observations"
    assert code(covariance="HC1") == "invalid_spec"
    # An instrument that cannot explain the endogenous regressor: rank condition.
    rng = np.random.default_rng(4)
    weak = pd.DataFrame({"x1": rng.normal(size=400), "z1": rng.normal(size=400)})
    weak["e1"] = weak.x1 + rng.normal(size=400)
    weak["y"] = (weak.e1 + rng.normal(size=400) > 0) * 1.0
    weak["z1"] = 0.0 * weak.z1 + weak.x1 ** 0        # a constant: no excluded instrument left
    assert code(frame=weak, instruments=["z1"]) == "underidentified"


def test_iv_results_round_trip_render_and_are_exported(data):
    results = {"ivprobit": ivprobit(data, cluster="firm"),
               "ivtobit": ivtobit(data, covariance="robust")}
    for name, result in results.items():
        text = round_trip(result)
        assert result.spec.estimator == name and result.provenance["family"] == "limited"
        assert result.provenance["stata_parity_validated"] is False
        assert "Wald test of exogeneity" in text and "/athrho2_1" in text
        assert name in oe.capabilities()["estimators"]
        assert callable(getattr(oe, name)) and getattr(oe, name).__doc__
        assert result.provenance["optimizer"]["converged"]
        assert result.extra["hessian"] == "analytic"


def test_iv_models_scale_linearly():
    rng = np.random.default_rng(9)
    n = 200_000
    columns = [f"x{i}" for i in range(5)]
    frame = pd.DataFrame(rng.normal(size=(n, 5)), columns=columns)
    frame["z1"], frame["z2"] = rng.normal(size=n), rng.normal(size=n)
    shocks = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    slopes = np.linspace(-0.4, 0.4, 5)
    frame["e"] = frame[columns].to_numpy() @ slopes + 0.7 * frame.z1 - 0.5 * frame.z2 \
        + shocks[:, 1]
    index = 0.2 + frame[columns].to_numpy() @ slopes - 0.6 * frame.e
    frame["y"] = (index + shocks[:, 0] > 0) * 1.0
    frame["t"] = np.maximum(0.0, index + shocks[:, 0])
    arguments = {"data": frame, "x": columns, "endog": ["e"], "instruments": ["z1", "z2"]}
    started = time.perf_counter()
    fits = [oe.ivprobit(y="y", **arguments), oe.ivtobit(y="t", ll=0, **arguments),
            oe.ivtobit(y="t", ll=0, method="twostep", **arguments)]
    elapsed = time.perf_counter() - started
    assert elapsed < 40
    for result in fits:
        assert_allclose(estimates(result)[1:6], slopes, atol=0.03)
        assert abs(estimates(result)[6] + 0.6) < 0.03
