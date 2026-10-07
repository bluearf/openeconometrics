"""heckman (ML and two-step) and heckprobit against independent oracles.

The ML estimators are compared with brute-force maximization of independently
written NumPy likelihoods (the bivariate normal distribution function of the
heckprobit oracle is a 100-point Gauss-Legendre evaluation of Plackett's
identity, checked against SciPy); covariances are rebuilt from complex-step
scores and numerical Hessians. The two-step estimator is compared with
Heckman's / Greene's explicit formula coded in NumPy on top of a statsmodels
probit.
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
    KINDS, brute_force, check_fit, covariance, design, errors, estimates, round_trip, terms,
)

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics.limited.selection_kernels import (
    HeckmanObjective, HeckprobitObjective,
)
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec

X = ["x1", "x2"]
Z = ["x1", "x2", "z1"]
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(100)


# ---- independent likelihoods ----------------------------------------------------------


def heckman_loglik(theta, x, z, y, s):
    """Heckman selection log likelihood per observation in (b, g, athrho, lnsigma)."""
    k, q = x.shape[1], z.shape[1]
    rho, sigma = np.tanh(theta[k + q]), np.exp(theta[k + q + 1])
    index = z @ theta[k:k + q]
    e = (np.where(s, y, 0.0) - x @ theta[:k]) / sigma
    chosen = np.log(special.ndtr((index + rho * e) / np.sqrt(1 - rho ** 2))) \
        - 0.5 * e ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    return np.where(s, chosen, np.log(special.ndtr(-index)))


def phi2(a, b, r):
    """Bivariate normal cdf by Plackett's identity (complex-step safe)."""
    half = np.arcsin(r) / 2
    sine = np.sin(np.multiply.outer(half, 1 + _NODES))
    a2, b2 = np.asarray(a)[..., None], np.asarray(b)[..., None]
    integrand = np.exp(-(a2 ** 2 + b2 ** 2 - 2 * a2 * b2 * sine) / (2 * (1 - sine ** 2)))
    return special.ndtr(a) * special.ndtr(b) \
        + (integrand * _WEIGHTS).sum(axis=-1) * half / (2 * np.pi)


def heckprobit_loglik(theta, x, z, y, s):
    """Probit-with-selection log likelihood per observation in (b, g, athrho)."""
    k, q = x.shape[1], z.shape[1]
    rho = np.tanh(theta[k + q])
    sign = np.where(s, 2 * y - 1, 1.0)
    joint = phi2(sign * (x @ theta[:k]), z @ theta[k:k + q], sign * rho)
    return np.where(s, np.log(joint), np.log(special.ndtr(-(z @ theta[k:k + q]))))


# ---- data -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20262)
    n = 900
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n), "z1": rng.normal(size=n),
        "firm": np.repeat(np.arange(90), 10), "state": rng.integers(0, 9, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
    })
    shocks = rng.multivariate_normal([0, 0], [[1, 0.55], [0.55, 1]], size=n)
    frame["s"] = (0.3 + 0.5 * frame.x1 - 0.3 * frame.x2 + 0.9 * frame.z1 + shocks[:, 1] > 0) * 1.0
    wage = 1.0 + 0.8 * frame.x1 - 0.5 * frame.x2 + 1.4 * shocks[:, 0]
    frame["y"] = np.where(frame.s == 1, wage, np.nan)
    frame["d"] = np.where(frame.s == 1, (0.2 + 0.8 * frame.x1 - 0.5 * frame.x2
                                         + shocks[:, 0] > 0) * 1.0, np.nan)
    return frame


def arrays(frame, outcome="y"):
    selected = frame.s.to_numpy() == 1
    return (design(frame, X), design(frame, Z), np.nan_to_num(frame[outcome].to_numpy()),
            selected)


HECKMAN_START = np.r_[1.0, 0.5, -0.5, 0.3, 0.5, -0.3, 0.9, 0.3, 0.2]
HECKPROBIT_START = np.r_[0.2, 0.5, -0.5, 0.3, 0.5, -0.3, 0.9, 0.2]


def heckman(frame, **options):
    return oe.heckman(data=frame, y="y", x=X, select="s", select_x=Z, **options)


def heckprobit(frame, **options):
    return oe.heckprobit(data=frame, y="d", x=X, select="s", select_x=Z, **options)


def two_step_oracle(frame, weights=None):
    """Heckman's two-step estimator and covariance, coded from Greene's formula."""
    x, z, y, selected = arrays(frame)
    w = np.ones(len(frame)) if weights is None else weights
    probit = sm.Probit(selected.astype(float), z).fit(disp=0, tol=1e-13, maxiter=200) \
        if weights is None else sm.GLM(selected.astype(float), z, freq_weights=w,
                                       family=sm.families.Binomial(
                                           sm.families.links.Probit())).fit(tol=1e-13)
    gamma = np.asarray(probit.params)
    index = z @ gamma
    # Observed-information covariance of the probit (statsmodels' GLM reports the expected one).
    q = 2 * selected - 1
    ratio = q * stats.norm.pdf(index) / stats.norm.cdf(q * index)
    v_probit = np.linalg.inv((z * (w * ratio * (ratio + index))[:, None]).T @ z)
    x1, z1, y1, w1 = x[selected], z[selected], y[selected], w[selected]
    mills = stats.norm.pdf(index[selected]) / stats.norm.cdf(index[selected])
    delta = mills * (mills + index[selected])
    star = np.column_stack([x1, mills])
    gram = np.linalg.inv((star * w1[:, None]).T @ star)
    beta = gram @ (star * w1[:, None]).T @ y1
    resid = y1 - star @ beta
    lam = beta[-1]
    sigma2 = ((w1 * resid ** 2).sum() + lam ** 2 * (w1 * delta).sum()) / w1.sum()
    rho2 = lam ** 2 / sigma2
    link = (star * (w1 * delta)[:, None]).T @ z1
    inner = (star * (w1 * (1 - rho2 * delta))[:, None]).T @ star + rho2 * link @ v_probit @ link.T
    return {"beta": beta, "gamma": gamma, "sigma": np.sqrt(sigma2), "rho": lam / np.sqrt(sigma2),
            "v_beta": sigma2 * gram @ inner @ gram, "v_gamma": v_probit,
            "cross": lam * gram @ link @ v_probit}


# ---- heckman, maximum likelihood ------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_heckman_ml_matches_brute_force_likelihood(data, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = heckman(data, **options)
    x, z, y, selected = arrays(data)
    theta, log_likelihood = check_fit(result, heckman_loglik, HECKMAN_START, x, z, y, selected,
                                      groups=data.firm)
    assert terms(result) == ["Intercept", "x1", "x2", "select:Intercept", "select:x1",
                             "select:x2", "select:z1", "/athrho", "/lnsigma"]
    assert [c.equation for c in result.coefficients] == ["y"] * 3 + ["select"] * 4 + [None] * 2
    assert result.nobs == len(data) and result.dropped_rows == 0
    metrics, v = result.metrics, covariance(result)
    assert metrics["n_selected"] == selected.sum()
    assert metrics["n_censored"] == (~selected).sum()
    rho, sigma = np.tanh(theta[7]), np.exp(theta[8])
    assert_allclose([metrics["rho"], metrics["sigma"], metrics["lambda"]],
                    [rho, sigma, rho * sigma], rtol=1e-6)
    assert_allclose(metrics["aic"], -2 * log_likelihood + 18, rtol=1e-10)
    assert_allclose(metrics["bic"], -2 * log_likelihood + 9 * np.log(len(data)), rtol=1e-10)
    extra = result.extra
    assert_allclose(extra["rho"]["std_error"], (1 - rho ** 2) * np.sqrt(v[7, 7]), rtol=1e-9)
    assert_allclose(extra["sigma"]["std_error"], sigma * np.sqrt(v[8, 8]), rtol=1e-9)
    gradient = np.array([sigma * (1 - rho ** 2), rho * sigma])
    assert_allclose(extra["lambda"]["std_error"], np.sqrt(gradient @ v[7:, 7:] @ gradient),
                    rtol=1e-9)
    z975 = stats.norm.ppf(0.975)
    assert_allclose([extra["rho"]["ci_low"], extra["rho"]["ci_high"]],
                    np.tanh(theta[7] + np.array([-1, 1]) * z975 * np.sqrt(v[7, 7])), rtol=1e-8)
    model = result.tests["model"]
    assert model["df"] == 2 and model["distribution"] == "chi2"
    assert_allclose(model["statistic"], theta[1:3] @ np.linalg.solve(v[1:3, 1:3], theta[1:3]),
                    rtol=1e-6)
    test = result.tests["rho"]
    if kind in {"nonrobust", "opg"}:
        probit = sm.Probit(selected.astype(float), z).fit(disp=0, tol=1e-12)
        ols = sm.OLS(y[selected], x[selected]).fit()
        comparison = probit.llf + ols.llf
        assert_allclose(extra["comparison_log_likelihood"], comparison, rtol=1e-9)
        assert_allclose(test["statistic"], 2 * (log_likelihood - comparison), rtol=1e-7)
        assert_allclose(test["p_value"], stats.chi2.sf(test["statistic"], 1), rtol=1e-7)
    else:
        assert_allclose(test["statistic"], theta[7] ** 2 / v[7, 7], rtol=1e-6)
    assert not result.inference["use_t"]
    assert len(result.predictions) == 400


@pytest.mark.parametrize("kind", ["nonrobust", "robust", "cluster"])
def test_heckman_ml_frequency_weights_replicate_rows(data, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    weighted = heckman(data, weights="fw", weight_type="fweight", **options)
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    replicated = heckman(expanded, **options)
    assert weighted.nobs == replicated.nobs == int(data.fw.sum())
    assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-8)
    assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-6, atol=1e-12)
    for name in ("log_likelihood", "n_selected", "n_censored", "lambda"):
        assert_allclose(weighted.metrics[name], replicated.metrics[name], rtol=1e-8)
    assert_allclose(weighted.tests["rho"]["statistic"], replicated.tests["rho"]["statistic"],
                    rtol=1e-6)


def test_heckman_ml_analytic_sampling_and_importance_weights(data):
    x, z, y, selected = arrays(data)
    aw = data.aw.to_numpy()
    analytic = heckman(data, weights="aw", weight_type="aweight")
    check_fit(analytic, heckman_loglik, HECKMAN_START, x, z, y, selected,
              weights=aw * len(aw) / aw.sum())
    importance = heckman(data, weights="aw", weight_type="iweight", covariance="opg")
    check_fit(importance, heckman_loglik, HECKMAN_START, x, z, y, selected, weights=aw)
    sampling = heckman(data, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    check_fit(sampling, heckman_loglik, HECKMAN_START, x, z, y, selected, weights=aw)
    with pytest.raises(AnalysisError) as error:
        heckman(data, weights="aw", weight_type="pweight", covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"


# ---- heckman, two-step ----------------------------------------------------------------


def test_heckman_two_step_matches_greenes_formula(data):
    result = heckman(data, method="twostep")
    oracle = two_step_oracle(data)
    assert terms(result) == ["Intercept", "x1", "x2", "select:Intercept", "select:x1",
                             "select:x2", "select:z1", "mills:lambda"]
    assert [c.equation for c in result.coefficients] == ["y"] * 3 + ["select"] * 4 + ["mills"]
    b, v = estimates(result), covariance(result)
    outcome = [0, 1, 2, 7]
    assert_allclose(b[outcome], oracle["beta"], rtol=1e-8)
    assert_allclose(b[3:7], oracle["gamma"], rtol=1e-7)
    assert_allclose(v[np.ix_(outcome, outcome)], oracle["v_beta"], rtol=1e-6)
    assert_allclose(v[3:7, 3:7], oracle["v_gamma"], rtol=1e-6)
    assert_allclose(v[np.ix_(outcome, [3, 4, 5, 6])], oracle["cross"], rtol=1e-6, atol=1e-12)
    probit = sm.Probit((data.s == 1).astype(float), design(data, Z)).fit(disp=0, tol=1e-13)
    assert_allclose(errors(result)[3:7], probit.bse, rtol=1e-6)
    metrics = result.metrics
    assert_allclose([metrics["rho"], metrics["sigma"], metrics["lambda"]],
                    [oracle["rho"], oracle["sigma"], oracle["beta"][-1]], rtol=1e-8)
    assert metrics["n_selected"] == (data.s == 1).sum()
    assert "log_likelihood" not in metrics and not result.extra["rho_truncated"]
    model = result.tests["model"]
    assert_allclose(model["statistic"], b[1:3] @ np.linalg.solve(v[1:3, 1:3], b[1:3]), rtol=1e-7)
    assert model["df"] == 2
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.norm.sf(np.abs(b / errors(result))), rtol=1e-8, atol=1e-300)
    # Both estimators are consistent for the same parameters.
    ml = heckman(data)
    assert_allclose(b[:7], estimates(ml)[:7], atol=4 * errors(ml)[:7].max())
    assert_allclose(metrics["lambda"], ml.metrics["lambda"], atol=0.25)
    round_trip(result)


def test_heckman_two_step_frequency_weights_and_restrictions(data):
    weighted = heckman(data, method="twostep", weights="fw", weight_type="fweight")
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    replicated = heckman(expanded, method="twostep")
    assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-8)
    assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-6, atol=1e-13)
    assert weighted.nobs == int(data.fw.sum())
    oracle = two_step_oracle(data, data.fw.to_numpy())
    assert_allclose(estimates(weighted)[[0, 1, 2, 7]], oracle["beta"], rtol=1e-7)
    assert_allclose(covariance(weighted)[np.ix_([0, 1, 2, 7], [0, 1, 2, 7])], oracle["v_beta"],
                    rtol=1e-5)
    for options, code in (({"covariance": "robust"}, "unsupported_covariance"),
                          ({"cluster": "firm"}, "unsupported_covariance"),
                          ({"weights": "aw", "weight_type": "aweight"}, "unsupported_weights"),
                          ({"weights": "aw", "weight_type": "pweight"}, "unsupported_weights")):
        with pytest.raises(AnalysisError) as error:
            heckman(data, method="twostep", **options)
        assert error.value.code == code
    with pytest.raises(AnalysisError) as error:
        heckman(data, method="gmm")
    assert error.value.code == "invalid_spec"


def test_heckman_two_step_without_selection_effect_and_truncated_rho():
    rng = np.random.default_rng(11)
    n = 4000
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "z1": rng.normal(size=n)})
    frame["s"] = (0.2 + 0.4 * frame.x1 + 0.8 * frame.z1 + rng.normal(size=n) > 0) * 1.0
    frame["y"] = np.where(frame.s == 1, 1 + 0.5 * frame.x1 - 0.5 * frame.x2
                          + rng.normal(size=n), np.nan)
    result = heckman(frame, method="twostep")
    lam = result.coefficients[-1]
    assert abs(lam.statistic) < 2.5 and abs(result.metrics["rho"]) < 0.2
    chosen = frame[frame.s == 1]
    ols = sm.OLS(chosen.y, design(chosen, X)).fit()
    assert_allclose(estimates(result)[:3], ols.params, atol=3 * errors(result)[:3].max())
    assert_allclose(result.metrics["sigma"], np.sqrt(ols.ssr / len(chosen)), rtol=0.02)
    # An estimate of rho outside [-1, 1] is truncated and sigma follows (Stata's rhosigma).
    rng = np.random.default_rng(0)
    small = pd.DataFrame({"x": rng.normal(size=150), "z": rng.normal(size=150)})
    u = rng.multivariate_normal([0, 0], [[1, 0.95], [0.95, 1]], size=150)
    small["s"] = (0.2 + 0.5 * small.x + 0.4 * small.z + u[:, 1] > 0) * 1.0
    small["y"] = np.where(small.s == 1, 1 + 0.7 * small.x + u[:, 0], np.nan)
    limited = oe.heckman(data=small, y="y", x=["x"], select="s", select_x=["x", "z"],
                         method="twostep")
    assert limited.extra["rho_truncated"] and limited.metrics["rho"] == 1.0
    assert_allclose(limited.metrics["sigma"], abs(limited.metrics["lambda"]))
    assert any("truncated" in warning for warning in limited.warnings)
    assert (errors(limited) > 0).all()


# ---- heckprobit -----------------------------------------------------------------------


def test_bivariate_normal_oracle_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(6):
        a, b, r = rng.normal(), rng.normal(), rng.uniform(-0.9, 0.9)
        expected = stats.multivariate_normal(cov=[[1, r], [r, 1]]).cdf([a, b])
        assert_allclose(phi2(np.array([a]), np.array([b]), np.array([r]))[0], expected,
                        atol=1e-7)


@pytest.fixture(scope="module")
def heckprobit_solution(data):
    x, z, y, selected = arrays(data, "d")
    return brute_force(heckprobit_loglik, HECKPROBIT_START, np.ones(len(data)), x, z, y, selected)


@pytest.mark.parametrize("kind", KINDS)
def test_heckprobit_matches_brute_force_likelihood(data, heckprobit_solution, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = heckprobit(data, **options)
    x, z, y, selected = arrays(data, "d")
    theta, log_likelihood = check_fit(result, heckprobit_loglik, HECKPROBIT_START, x, z, y,
                                      selected, groups=data.firm, solution=heckprobit_solution)
    assert terms(result) == ["Intercept", "x1", "x2", "select:Intercept", "select:x1",
                             "select:x2", "select:z1", "/athrho"]
    assert [c.equation for c in result.coefficients] == ["d"] * 3 + ["select"] * 4 + [None]
    v, metrics = covariance(result), result.metrics
    assert_allclose(metrics["rho"], np.tanh(theta[7]), rtol=1e-6)
    assert metrics["n_selected"] == selected.sum() and metrics["n_censored"] == (~selected).sum()
    assert_allclose(metrics["aic"], -2 * log_likelihood + 16, rtol=1e-10)
    assert_allclose(result.extra["rho"]["std_error"],
                    (1 - np.tanh(theta[7]) ** 2) * np.sqrt(v[7, 7]), rtol=1e-9)
    model = result.tests["model"]
    assert_allclose(model["statistic"], theta[1:3] @ np.linalg.solve(v[1:3, 1:3], theta[1:3]),
                    rtol=1e-6)
    test = result.tests["rho"]
    if kind in {"nonrobust", "opg"}:
        selection = sm.Probit(selected.astype(float), z).fit(disp=0, tol=1e-12)
        outcome = sm.Probit(y[selected], x[selected]).fit(disp=0, tol=1e-12)
        assert_allclose(result.extra["comparison_log_likelihood"], selection.llf + outcome.llf,
                        rtol=1e-9)
        assert_allclose(test["statistic"], 2 * (log_likelihood - selection.llf - outcome.llf),
                        rtol=1e-6)
    else:
        assert_allclose(test["statistic"], theta[7] ** 2 / v[7, 7], rtol=1e-6)
    counts = result.extra["outcome_counts"]
    assert counts["selected_one"] == (y[selected] == 1).sum()
    assert counts["selected_zero"] == (y[selected] == 0).sum()


def test_heckprobit_weights_follow_stata_semantics(data):
    for options in ({"covariance": "nonrobust"}, {"covariance": "robust"}, {"cluster": "firm"}):
        weighted = heckprobit(data, weights="fw", weight_type="fweight", **options)
        expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
        replicated = heckprobit(expanded, **options)
        assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-8)
        assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-6, atol=1e-12)
        assert weighted.nobs == int(data.fw.sum())
    x, z, y, selected = arrays(data, "d")
    aw = data.aw.to_numpy()
    analytic = heckprobit(data, weights="aw", weight_type="aweight")
    check_fit(analytic, heckprobit_loglik, HECKPROBIT_START, x, z, y, selected,
              weights=aw * len(aw) / aw.sum())
    sampling = heckprobit(data, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    # pweights share the aweight point estimates; the robust covariance ignores their scale.
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-8)
    doubled = heckprobit(data.assign(aw=2 * data.aw), weights="aw", weight_type="pweight")
    assert_allclose(covariance(doubled), covariance(sampling), rtol=1e-7)
    robust = heckprobit(data, weights="aw", weight_type="aweight", covariance="robust")
    assert_allclose(covariance(sampling), covariance(robust), rtol=1e-7)


# ---- derivatives ----------------------------------------------------------------------


@pytest.mark.parametrize("athrho", [-1.2, 0.0, 0.4, 2.0])
def test_selection_kernels_have_exact_derivatives(data, athrho):
    x, z = torch.tensor(design(data, X)), torch.tensor(design(data, Z))
    selected = torch.tensor(data.s.to_numpy() == 1)
    weights = torch.tensor(data.aw.to_numpy())
    cases = [
        (HeckmanObjective(x, z, torch.tensor(np.nan_to_num(data.y.to_numpy())), selected,
                          weights),
         [0.8, 0.6, -0.3, 0.2, 0.4, -0.2, 0.7, athrho, 0.25]),
        (HeckprobitObjective(x, z, torch.tensor(np.nan_to_num(data.d.to_numpy())), selected,
                             weights),
         [0.1, 0.6, -0.3, 0.2, 0.4, -0.2, 0.7, athrho]),
    ]
    for objective, point in cases:
        theta = torch.tensor(point, dtype=torch.float64)
        report = check_derivatives(objective, theta)
        assert report["gradient_max_rel_error"] < 1e-8
        assert report["hessian_max_rel_error"] < 1e-7
        assert report["hessian_asymmetry"] < 1e-9
        value, gradient, _ = objective(theta)
        assert_allclose(float(objective.value(theta)), float(value), rtol=1e-13)
        rows = objective.score_rows(theta)
        assert rows.shape == (len(data), len(point))
        assert_allclose((rows * weights[:, None]).sum(dim=0).numpy(), gradient.numpy(),
                        rtol=1e-9, atol=1e-9)


# ---- sample handling and failures -----------------------------------------------------


def test_selection_sample_missing_values_and_collinearity(data):
    reference = heckman(data)
    # Outcome values of nonselected rows are ignored, whatever they are.
    noisy = data.assign(y=data.y.where(data.s == 1, 123.0))
    assert_allclose(estimates(heckman(noisy)), estimates(reference), rtol=1e-12)
    holes = data.copy()
    chosen = holes.index[holes.s == 1][:3]
    holes.loc[chosen, "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        heckman(holes)
    assert error.value.code == "missing_values"
    dropped = heckman(holes, missing="drop")
    assert dropped.nobs == len(data) - 3 and dropped.dropped_rows == 3
    assert any("missing outcome" in warning for warning in dropped.warnings)
    assert_allclose(estimates(dropped), estimates(heckman(holes.drop(index=chosen))), rtol=1e-12)
    gaps = data.copy()
    gaps.loc[5, "z1"] = np.nan
    with pytest.raises(AnalysisError) as error:
        heckman(gaps)
    assert error.value.code == "missing_values"
    assert heckman(gaps, missing="drop").nobs == len(data) - 1
    frame = data.assign(copy=2 * data.x1, zcopy=-data.z1)
    for function, outcome in ((oe.heckman, "y"), (oe.heckprobit, "d")):
        result = function(data=frame, y=outcome, x=["x1", "copy", "x2"], select="s",
                          select_x=["x1", "x2", "z1", "zcopy"])
        assert result.provenance["omitted_terms"] == ["select:zcopy", "copy"]
        assert "copy" not in terms(result) and "select:zcopy" not in terms(result)
    coded = oe.heckman(data=data, y="y", x=X, select="s", select_x=[*Z, "sector"],
                       categorical=["sector"])
    assert "select:sector[b]" in terms(coded) and "select:sector[c]" in terms(coded)
    spec = ModelSpec(estimator="heckman", outcome="y", predictors=X,
                     columns={"select": "s", "select_x": Z})
    assert_allclose(estimates(fit(spec, data=data)), estimates(reference), rtol=1e-12)
    assert spec.covariance == "nonrobust"
    # An outcome regressor that varies only among the nonselected rows is omitted.
    idle = data.assign(extra=np.where(data.s == 1, 0.0, data.z1))
    omitted = oe.heckman(data=idle, y="y", x=[*X, "extra"], select="s", select_x=Z)
    assert omitted.provenance["omitted_terms"] == ["extra"]


def test_selection_error_codes(data):
    def code(function, frame=data, **arguments):
        base = {"y": "y", "x": X, "select": "s", "select_x": Z}
        with pytest.raises(AnalysisError) as error:
            function(data=frame, **{**base, **arguments})
        return error.value.code

    assert code(oe.heckman, frame=data.assign(s=data.s + 1)) == "invalid_binary_outcome"
    assert code(oe.heckman, frame=data.assign(s=1.0, y=data.y.fillna(0))) == "constant_selection"
    assert code(oe.heckman, select="y") == "invalid_spec"
    assert code(oe.heckman, select_x=["s", "z1"]) == "invalid_spec"
    assert code(oe.heckman, select_x=["y", "z1"]) == "invalid_spec"
    assert code(oe.heckman, select_x="z1") == "invalid_spec"
    assert code(oe.heckman, select_x=[]) == "invalid_spec"
    assert code(oe.heckman, covariance="HC3") == "invalid_spec"
    assert code(oe.heckman, select_x=["nope"]) == "missing_columns"
    assert code(oe.heckprobit) == "invalid_binary_outcome"
    assert code(oe.heckprobit, y="d", frame=data.assign(d=np.where(data.s == 1, 1.0, np.nan))) \
        == "constant_outcome"
    assert code(oe.heckprobit, y="d", select_x=[]) == "invalid_spec"
    # Perfectly correlated errors: the likelihood has no interior maximum.
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"x1": rng.normal(size=800), "x2": rng.normal(size=800),
                          "z1": rng.normal(size=800)})
    shock = rng.normal(size=800)
    frame["s"] = (0.3 + 0.5 * frame.x1 + 0.8 * frame.z1 + shock > 0) * 1.0
    frame["y"] = np.where(frame.s == 1, 1 + 0.7 * frame.x1 + 1.2 * shock, np.nan)
    frame["d"] = np.where(frame.s == 1, (0.2 + 0.7 * frame.x1 + shock > 0) * 1.0, np.nan)
    assert code(oe.heckman, frame=frame) == "boundary_solution"
    assert code(oe.heckprobit, y="d", frame=frame) == "boundary_solution"
    # The selection indicator is predicted perfectly.
    separated = data.assign(s=(data.z1 > 0) * 1.0, y=data.y.fillna(0.0))
    assert code(oe.heckman, frame=separated) == "separation_detected"


def test_selection_results_round_trip_render_and_are_exported(data):
    results = {"heckman": heckman(data, cluster="firm"),
               "heckprobit": heckprobit(data, covariance="robust")}
    for name, result in results.items():
        text = round_trip(result)
        assert result.spec.estimator == name and result.provenance["family"] == "limited"
        assert result.provenance["stata_parity_validated"] is False
        assert "rho" in text and "select:z1" in text and "[select]" in text
        assert name in oe.capabilities()["estimators"]
        assert callable(getattr(oe, name)) and getattr(oe, name).__doc__
        assert result.provenance["optimizer"]["converged"]
    assert results["heckman"].inference["cluster_count"] == 90


def test_selection_models_scale_linearly():
    rng = np.random.default_rng(8)
    n = 200_000
    columns = [f"x{i}" for i in range(6)]
    frame = pd.DataFrame(rng.normal(size=(n, 6)), columns=columns)
    frame["z"] = rng.normal(size=n)
    shocks = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    slopes = np.linspace(-0.4, 0.4, 6)
    index = 0.3 + frame[columns].to_numpy() @ slopes
    frame["s"] = (index + 0.8 * frame.z + shocks[:, 1] > 0) * 1.0
    frame["y"] = np.where(frame.s == 1, 1 + index + shocks[:, 0], np.nan)
    frame["d"] = np.where(frame.s == 1, (index + shocks[:, 0] > 0) * 1.0, np.nan)
    started = time.perf_counter()
    ml = oe.heckman(data=frame, y="y", x=columns, select="s", select_x=[*columns, "z"])
    two = oe.heckman(data=frame, y="y", x=columns, select="s", select_x=[*columns, "z"],
                     method="twostep")
    probit = oe.heckprobit(data=frame, y="d", x=columns, select="s", select_x=[*columns, "z"])
    elapsed = time.perf_counter() - started
    assert elapsed < 40
    for result in (ml, two, probit):
        assert_allclose(estimates(result)[1:7], slopes, atol=0.03)
    assert abs(ml.metrics["rho"] - 0.5) < 0.03 and abs(probit.metrics["rho"] - 0.5) < 0.06
