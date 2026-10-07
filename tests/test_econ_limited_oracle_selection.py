"""Independent oracles for heckman (ML and two-step) and heckprobit (verification stage).

The likelihoods are written from the model definitions in the natural
parameters ``(b, g, rho, sigma)`` / ``(b, g, rho)`` -- not the working
``(athrho, lnsigma)`` -- and maximized with the complex-step toolkit of
``test_econ_limited_oracle``; the reported parameters and their covariance
follow from the Jacobian of the reparameterization. The bivariate normal
distribution function of the heckprobit oracle is Plackett's integral
``Phi2(h, k; r) = Phi(h) Phi(k) + int_0^r phi2(h, k; t) dt`` by Gauss-Legendre
quadrature. The two-step oracle is Greene's formula on a statsmodels probit.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import special, stats
from test_econ_limited_oracle import (
    KINDS, LOG_SQRT_2PI, check_table, check_test, covariance_of, dummies, estimates,
    jacobian_cs, likelihood_pieces, likelihood_weights, newton, options_for, stata_covariance,
    table, wald,
)

import openecon as oe
from openecon.analysis import AnalysisError

NODES, WEIGHTS = np.polynomial.legendre.leggauss(32)
SOLVED = {}                # oracle maxima, shared by the covariance kinds of one weighting


def phi2_cdf(h, k, rho):
    """Standard bivariate normal distribution function (Plackett's identity)."""
    t = 0.5 * rho * (NODES[:, None] + 1)                          # [m, n] or [m, 1]
    density = np.exp(-(h * h - 2 * t * h * k + k * k) / (2 * (1 - t * t))) \
        / (2 * np.pi * np.sqrt(1 - t * t))
    return special.ndtr(h) * special.ndtr(k) + 0.5 * rho * (WEIGHTS[:, None] * density).sum(axis=0)


def test_plackett_oracle_is_accurate():
    for rho in (-0.8, -0.3, 0.0, 0.5, 0.9):
        assert_allclose(phi2_cdf(np.zeros(1), np.zeros(1), rho),
                        0.25 + np.arcsin(rho) / (2 * np.pi), rtol=1e-13)
        point = np.array([0.7, -1.2])
        expected = stats.multivariate_normal(cov=[[1, rho], [rho, 1]]).cdf(point)
        assert_allclose(phi2_cdf(point[:1], point[1:], rho), expected, atol=1e-7)


# ---- data ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(55021)
    n = 520
    sizes = rng.integers(3, 16, size=70)
    firm = np.repeat(np.arange(len(sizes)), sizes)[:n]
    firm = np.concatenate([firm, np.full(n - len(firm), len(sizes))])
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 2, size=n),
        "z1": rng.normal(size=n), "kind": rng.choice(["a", "b", "c"], size=n),
        "firm": firm, "fw": rng.integers(1, 4, size=n).astype(float),
        "aw": rng.gamma(3.0, 0.5, size=n) + 0.2,
    })
    shocks = rng.multivariate_normal([0, 0], [[1, 0.55], [0.55, 1]], size=n)
    effect = frame.kind.map({"a": 0.0, "b": 0.4, "c": -0.3}).to_numpy()
    frame["s"] = (0.3 + 0.6 * frame.x1 - 0.9 * frame.z1 + effect + shocks[:, 1] > 0) * 1.0
    latent = 0.5 + 0.9 * frame.x1 - 0.4 * frame.x2 + 1.3 * shocks[:, 0]
    frame["y"] = np.where(frame.s == 1, latent, np.nan)
    frame["d"] = np.where(frame.s == 1, (0.2 + 0.7 * frame.x1 - 0.3 * frame.x2
                                         + shocks[:, 0] > 0) * 1.0, np.nan)
    return frame


XO = ["x1", "x2"]
ZS = ["x1", "z1", "kind"]
CAT = ["kind"]


def arrays(frame, outcome):
    x, x_names = dummies(frame, XO)
    z, z_names = dummies(frame, ZS, CAT)
    selected = frame.s.to_numpy() == 1
    y = np.where(selected, frame[outcome].to_numpy(), 0.0)
    return x, x_names, z, [f"select:{name}" for name in z_names], y, selected


def probit(y, x, w=None):
    """statsmodels probit (GLM for non-integer weights): coefficients, covariance, ll."""
    if w is None:
        fit = sm.Probit(y, x).fit(disp=0, tol=1e-12, maxiter=200, method="newton")
        return fit.params, fit.cov_params(), fit.llf
    family = sm.families.Binomial(link=sm.families.links.Probit())
    fit = sm.GLM(y, x, family=family, freq_weights=w).fit(tol=1e-13, maxiter=200)
    index = x @ fit.params
    sign = 2 * y - 1
    value = float(w @ np.log(special.ndtr(sign * index)))
    ratio = stats.norm.pdf(index) / special.ndtr(sign * index)
    curvature = ratio * (ratio + sign * index)
    return fit.params, np.linalg.inv((x * (w * curvature)[:, None]).T @ x), value


# ---- heckman, maximum likelihood -----------------------------------------------------


def heckman_natural(x, z, y, selected):
    """Per-observation log likelihood in ``(b, g, rho, sigma)``."""
    k, q = x.shape[1], z.shape[1]

    def obs(theta):
        beta, gamma, rho, sigma = theta[:k], theta[k:k + q], theta[k + q], theta[k + q + 1]
        e = (y - x @ beta) / sigma
        zg = z @ gamma
        on = np.log(special.ndtr((zg + rho * e) / np.sqrt(1 - rho * rho))) - 0.5 * e * e \
            - np.log(sigma) - LOG_SQRT_2PI
        return np.where(selected, on, np.log(special.ndtr(-zg)))

    def reported(theta):
        return np.concatenate([theta[:k + q], [np.arctanh(theta[k + q])],
                               [np.log(theta[k + q + 1])]])

    return obs, reported


def heckman_oracle(frame, kind, *, weights=None, weight_type=None):
    x, x_names, z, z_names, y, selected = arrays(frame, "y")
    k, q = x.shape[1], z.shape[1]
    w, nobs = likelihood_weights(frame, weights, weight_type)
    obs, reported = heckman_natural(x, z, y, selected)
    gamma = probit(selected.astype(float), z, w)[0]
    root = np.sqrt(w[selected])
    beta = np.linalg.lstsq(x[selected] * root[:, None], y[selected] * root, rcond=None)[0]
    start = np.r_[beta, gamma, 0.0, np.std(y[selected] - x[selected] @ beta)]
    key = ("heckman", weights, "pweight" if weight_type == "iweight" else weight_type)
    if key not in SOLVED:
        theta, value = newton(obs, start, w)
        SOLVED[key] = theta, value, likelihood_pieces(obs, theta, w)
    theta, value, pieces = SOLVED[key]
    natural = stata_covariance(obs, theta, kind, w=w, weight_type=weight_type,
                               groups=frame.firm, pieces=pieces)
    jacobian = jacobian_cs(reported, theta)
    return {"names": [*x_names, *z_names, "/athrho", "/lnsigma"], "theta": theta,
            "params": reported(theta), "covariance": jacobian @ natural @ jacobian.T,
            "natural": natural, "value": value, "nobs": nobs, "w": w, "k": k, "q": q,
            "arrays": (x, z, y, selected)}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighting", [None, "fweight", "aweight", "iweight", "pweight"])
def test_heckman_ml_every_covariance_and_weight_type(data, kind, weighting):
    call = {"data": data, "y": "y", "x": XO, "select": "s", "select_x": ZS, "categorical": CAT}
    if weighting == "pweight" and kind in ("nonrobust", "opg"):
        with pytest.raises(AnalysisError) as caught:
            oe.heckman(**call, covariance=kind, weights="aw", weight_type="pweight")
        assert caught.value.code == "unsupported_covariance"
        return
    column = {None: None, "fweight": "fw"}.get(weighting, "aw")
    extra = {} if weighting is None else {"weights": column, "weight_type": weighting}
    result = oe.heckman(**call, **options_for(kind), **extra)
    oracle = heckman_oracle(data, kind, weights=column, weight_type=weighting)
    k, q, w, nobs = oracle["k"], oracle["q"], oracle["w"], oracle["nobs"]
    x, z, y, selected = oracle["arrays"]
    params, covariance, value = oracle["params"], oracle["covariance"], oracle["value"]
    assert [c.term for c in result.coefficients] == oracle["names"]
    assert [c.equation for c in result.coefficients] == ["y"] * k + ["select"] * q + [None] * 2
    check_table(result, params, covariance)
    size = k + q + 2
    metrics = result.metrics
    assert result.nobs == nobs
    assert_allclose(metrics["log_likelihood"], value, rtol=1e-10)
    assert_allclose(metrics["aic"], -2 * value + 2 * size, rtol=1e-10)
    assert_allclose(metrics["bic"], -2 * value + np.log(nobs) * size, rtol=1e-10)
    counts = w if weighting == "fweight" else np.ones(len(y))
    assert metrics["n_selected"] == int(counts[selected].sum())
    assert metrics["n_censored"] == int(counts[~selected].sum())
    # rho, sigma and lambda with standard errors straight from the natural parameterization.
    rho, sigma = oracle["theta"][k + q], oracle["theta"][k + q + 1]
    natural = oracle["natural"]
    assert_allclose([metrics["rho"], metrics["sigma"], metrics["lambda"]],
                    [rho, sigma, rho * sigma], rtol=1e-6)
    record = result.extra
    assert_allclose(record["rho"]["std_error"], np.sqrt(natural[k + q, k + q]), rtol=1e-5)
    assert_allclose(record["sigma"]["std_error"], np.sqrt(natural[k + q + 1, k + q + 1]),
                    rtol=1e-5)
    direction = np.array([sigma, rho])
    block = natural[k + q:, k + q:]
    assert_allclose(record["lambda"]["std_error"], np.sqrt(direction @ block @ direction),
                    rtol=1e-5)
    crit = stats.norm.ppf(0.975)
    a, a_se = params[k + q], np.sqrt(covariance[k + q, k + q])
    t, t_se = params[k + q + 1], np.sqrt(covariance[k + q + 1, k + q + 1])
    assert_allclose([record["rho"]["ci_low"], record["rho"]["ci_high"]],
                    np.tanh([a - crit * a_se, a + crit * a_se]), rtol=1e-5)
    assert_allclose([record["sigma"]["ci_low"], record["sigma"]["ci_high"]],
                    np.exp([t - crit * t_se, t + crit * t_se]), rtol=1e-5)
    # Model test: outcome slopes. Independence: LR against probit + regression, or Wald.
    check_test(result.tests["model"], wald(params, covariance, range(1, k)), k - 1, rtol=1e-5)
    if kind in ("nonrobust", "opg"):
        probit_ll = probit(selected.astype(float), z, None if weighting is None else w)[2]
        w1 = w[selected]
        beta = np.linalg.lstsq(x[selected] * np.sqrt(w1)[:, None], y[selected] * np.sqrt(w1),
                               rcond=None)[0]
        ssr, total = w1 @ (y[selected] - x[selected] @ beta) ** 2, w1.sum()
        regression_ll = -0.5 * total * (np.log(2 * np.pi * ssr / total) + 1)
        check_test(result.tests["rho"], 2 * (value - probit_ll - regression_ll), 1, rtol=1e-6)
    else:
        check_test(result.tests["rho"], (a / a_se) ** 2, 1, rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_selection_frequency_weights_equal_replicated_rows(data, kind):
    replicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for function, outcome in ((oe.heckman, "y"), (oe.heckprobit, "d")):
        call = {"y": outcome, "x": XO, "select": "s", "select_x": ZS, "categorical": CAT,
                **options_for(kind)}
        weighted = function(data=data, weights="fw", weight_type="fweight", **call)
        expanded = function(data=replicated, **call)
        assert weighted.nobs == expanded.nobs == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-7, atol=1e-9)
        assert_allclose(covariance_of(weighted), covariance_of(expanded), rtol=2e-6, atol=1e-11)
        for key, value in expanded.metrics.items():
            assert_allclose(weighted.metrics[key], value, rtol=1e-8), key
        for name in ("model", "rho"):
            assert_allclose(weighted.tests[name]["statistic"], expanded.tests[name]["statistic"],
                            rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("kind", ["nonrobust", "opg"])
def test_selection_integer_importance_weights_equal_frequency_weights(data, kind):
    for function, outcome in ((oe.heckman, "y"), (oe.heckprobit, "d")):
        call = {"data": data, "y": outcome, "x": XO, "select": "s", "select_x": ZS,
                "categorical": CAT, "covariance": kind, "weights": "fw"}
        frequency = function(weight_type="fweight", **call)
        importance = function(weight_type="iweight", **call)
        assert_allclose(estimates(importance), estimates(frequency), rtol=1e-8)
        assert_allclose(covariance_of(importance), covariance_of(frequency), rtol=1e-7,
                        atol=1e-12)


# ---- heckman, two-step ---------------------------------------------------------------


def two_step_oracle(frame, fweights=None, columns=None):
    """Heckman (1979) / Greene: estimates and covariance in the order (b, g, lambda)."""
    x, x_names, z, z_names, y, selected = columns or arrays(frame, "y")
    w = np.ones(len(y)) if fweights is None else frame[fweights].to_numpy(dtype=float)
    gamma, v_probit, _ = probit(selected.astype(float), z, None if fweights is None else w)
    index = (z @ gamma)[selected]
    mills = stats.norm.pdf(index) / stats.norm.cdf(index)
    delta = mills * (mills + index)
    xs = np.column_stack([x[selected], mills])
    w1, y1, z1 = w[selected], y[selected], z[selected]
    cross = np.linalg.inv((xs * w1[:, None]).T @ xs)
    beta = cross @ (xs * w1[:, None]).T @ y1
    resid = y1 - xs @ beta
    n1 = w1.sum()
    b_lambda = beta[-1]
    sigma2 = (w1 @ resid ** 2 + b_lambda ** 2 * (w1 @ delta)) / n1
    rho = b_lambda / np.sqrt(sigma2)
    if abs(rho) > 1:                     # Stata's default rhosigma rule
        rho, sigma2 = np.sign(rho), b_lambda ** 2
    link = (xs * (w1 * delta)[:, None]).T @ z1                       # X*' D Z
    middle = (xs * (w1 * (1 - rho ** 2 * delta))[:, None]).T @ xs \
        + rho ** 2 * link @ v_probit @ link.T
    v_outcome = sigma2 * cross @ middle @ cross
    v_cross = b_lambda * cross @ link @ v_probit
    k, q = x.shape[1], z.shape[1]
    order = [*range(k), k + q]
    size = k + q + 1
    covariance = np.zeros((size, size))
    covariance[np.ix_(order, order)] = v_outcome
    covariance[k:k + q, k:k + q] = v_probit
    covariance[np.ix_(order, range(k, k + q))] = v_cross
    covariance[np.ix_(range(k, k + q), order)] = v_cross.T
    params = np.r_[beta[:k], gamma, b_lambda]
    return {"names": [*x_names, *z_names, "mills:lambda"], "params": params,
            "covariance": covariance, "rho": rho, "sigma": np.sqrt(sigma2), "k": k, "q": q,
            "n": int(w.sum()), "n1": int(n1)}


@pytest.mark.parametrize("fweights", [None, "fw"])
def test_heckman_two_step_matches_greene(data, fweights):
    extra = {} if fweights is None else {"weights": "fw", "weight_type": "fweight"}
    result = oe.heckman(data=data, y="y", x=XO, select="s", select_x=ZS, categorical=CAT,
                        method="twostep", **extra)
    oracle = two_step_oracle(data, fweights)
    k, q = oracle["k"], oracle["q"]
    assert [c.term for c in result.coefficients] == oracle["names"]
    assert [c.equation for c in result.coefficients] == ["y"] * k + ["select"] * q + ["mills"]
    check_table(result, oracle["params"], oracle["covariance"], rtol=2e-6)
    assert_allclose([result.metrics["rho"], result.metrics["sigma"], result.metrics["lambda"]],
                    [oracle["rho"], oracle["sigma"], oracle["params"][-1]], rtol=1e-6)
    assert result.nobs == oracle["n"] and result.metrics["n_selected"] == oracle["n1"]
    assert result.metrics["n_censored"] == oracle["n"] - oracle["n1"]
    check_test(result.tests["model"], wald(oracle["params"], oracle["covariance"], range(1, k)),
               k - 1, rtol=1e-5)
    assert "log_likelihood" not in result.metrics and "rho" not in result.tests
    if fweights:
        replicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
        expanded = oe.heckman(data=replicated, y="y", x=XO, select="s", select_x=ZS,
                              categorical=CAT, method="twostep")
        assert_allclose(estimates(result), estimates(expanded), rtol=1e-8)
        assert_allclose(covariance_of(result), covariance_of(expanded), rtol=1e-7, atol=1e-12)


def test_heckman_two_step_rejects_what_stata_rejects(data):
    call = {"data": data, "y": "y", "x": XO, "select": "s", "select_x": ZS, "categorical": CAT,
            "method": "twostep"}
    for options, code in (({"covariance": "robust"}, "unsupported_covariance"),
                          ({"cluster": "firm"}, "unsupported_covariance"),
                          ({"weights": "aw", "weight_type": "pweight"}, "unsupported_weights"),
                          ({"weights": "aw", "weight_type": "aweight"}, "unsupported_weights")):
        with pytest.raises(AnalysisError) as caught:
            oe.heckman(**call, **options)
        assert caught.value.code == code, options


def test_heckman_two_step_and_ml_agree_in_large_samples():
    """Both are consistent: on a large sample they agree within sampling error."""
    rng = np.random.default_rng(31)
    n = 20000
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    u = rng.multivariate_normal([0, 0], [[1, -0.5], [-0.5, 1]], size=n)
    frame["s"] = (0.2 + 0.5 * frame.x + frame.z + u[:, 1] > 0) * 1.0
    frame["y"] = np.where(frame.s == 1, 1 + 2 * frame.x + 1.5 * u[:, 0], np.nan)
    call = {"data": frame, "y": "y", "x": ["x"], "select": "s", "select_x": ["x", "z"]}
    ml, two = oe.heckman(**call), oe.heckman(method="twostep", **call)
    assert_allclose(estimates(ml)[:2], [1.0, 2.0], atol=0.06)
    assert_allclose(estimates(two)[:2], estimates(ml)[:2], atol=0.05)
    assert_allclose([ml.metrics["rho"], ml.metrics["sigma"]], [-0.5, 1.5], atol=0.06)
    assert_allclose([two.metrics["rho"], two.metrics["sigma"]], [-0.5, 1.5], atol=0.08)
    # The two-step standard errors exceed the efficient ML ones but are of the same order.
    ratio = table(two)["std_error"][:2] / table(ml)["std_error"][:2]
    assert np.all(ratio > 0.99) and np.all(ratio < 2.0)


# ---- heckprobit ----------------------------------------------------------------------


def heckprobit_natural(x, z, y, selected):
    """Per-observation log likelihood in ``(b, g, rho)``."""
    k, q = x.shape[1], z.shape[1]
    positive = selected & (y == 1)

    def obs(theta):
        xb, zg, rho = x @ theta[:k], z @ theta[k:k + q], theta[k + q]
        both = phi2_cdf(xb, zg, rho)                                  # Pr(y* > 0, selected)
        on = np.where(positive, np.log(both), np.log(special.ndtr(zg) - both))
        return np.where(selected, on, np.log(special.ndtr(-zg)))

    def reported(theta):
        return np.concatenate([theta[:k + q], [np.arctanh(theta[k + q])]])

    return obs, reported


def heckprobit_oracle(frame, kind, *, weights=None, weight_type=None):
    x, x_names, z, z_names, y, selected = arrays(frame, "d")
    k, q = x.shape[1], z.shape[1]
    w, nobs = likelihood_weights(frame, weights, weight_type)
    obs, reported = heckprobit_natural(x, z, y, selected)
    weighted = None if weights is None else w
    gamma, _, selection_ll = probit(selected.astype(float), z, weighted)
    beta, _, outcome_ll = probit(y[selected], x[selected],
                                 None if weights is None else w[selected])
    key = ("heckprobit", weights, "pweight" if weight_type == "iweight" else weight_type)
    if key not in SOLVED:
        theta, value = newton(obs, np.r_[beta, gamma, 0.0], w)
        SOLVED[key] = theta, value, likelihood_pieces(obs, theta, w)
    theta, value, pieces = SOLVED[key]
    natural = stata_covariance(obs, theta, kind, w=w, weight_type=weight_type,
                               groups=frame.firm, pieces=pieces)
    jacobian = jacobian_cs(reported, theta)
    return {"names": [*x_names, *z_names, "/athrho"], "theta": theta, "natural": natural,
            "params": reported(theta), "covariance": jacobian @ natural @ jacobian.T,
            "value": value, "nobs": nobs, "k": k, "q": q,
            "comparison": selection_ll + outcome_ll}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighting", [None, "fweight", "aweight", "iweight", "pweight"])
def test_heckprobit_every_covariance_and_weight_type(data, kind, weighting):
    if weighting == "pweight" and kind in ("nonrobust", "opg"):
        return
    column = {None: None, "fweight": "fw"}.get(weighting, "aw")
    extra = {} if weighting is None else {"weights": column, "weight_type": weighting}
    result = oe.heckprobit(data=data, y="d", x=XO, select="s", select_x=ZS, categorical=CAT,
                           **options_for(kind), **extra)
    oracle = heckprobit_oracle(data, kind, weights=column, weight_type=weighting)
    k, q, nobs = oracle["k"], oracle["q"], oracle["nobs"]
    params, covariance, value = oracle["params"], oracle["covariance"], oracle["value"]
    assert [c.term for c in result.coefficients] == oracle["names"]
    assert [c.equation for c in result.coefficients] == ["d"] * k + ["select"] * q + [None]
    check_table(result, params, covariance, rtol=2e-6)
    size = k + q + 1
    assert result.nobs == nobs
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-10)
    assert_allclose(result.metrics["aic"], -2 * value + 2 * size, rtol=1e-10)
    assert_allclose(result.metrics["bic"], -2 * value + np.log(nobs) * size, rtol=1e-10)
    rho = oracle["theta"][k + q]
    assert_allclose(result.metrics["rho"], rho, rtol=1e-6)
    assert_allclose(result.extra["rho"]["std_error"],
                    np.sqrt(oracle["natural"][k + q, k + q]), rtol=1e-4)
    check_test(result.tests["model"], wald(params, covariance, range(1, k)), k - 1, rtol=1e-4)
    if kind in ("nonrobust", "opg"):
        check_test(result.tests["rho"], 2 * (value - oracle["comparison"]), 1, rtol=1e-6)
    else:
        check_test(result.tests["rho"], params[k + q] ** 2 / covariance[k + q, k + q], 1,
                   rtol=1e-4)


def test_selection_missing_outcomes_and_recorded_values_of_nonselected_rows(data):
    """The outcome of a nonselected row is ignored whether missing or recorded."""
    filled = data.assign(y=data.y.fillna(123.0), d=data.d.fillna(1.0))
    for function, outcome in ((oe.heckman, "y"), (oe.heckprobit, "d")):
        call = {"y": outcome, "x": XO, "select": "s", "select_x": ZS, "categorical": CAT}
        left, right = function(data=data, **call), function(data=filled, **call)
        assert_allclose(estimates(left), estimates(right), rtol=1e-12)
        assert_allclose(covariance_of(left), covariance_of(right), rtol=1e-12)
        assert left.dropped_rows == 0 and left.nobs == len(data)
    two = oe.heckman(data=data, y="y", x=XO, select="s", select_x=ZS, categorical=CAT,
                     method="twostep")
    filled_two = oe.heckman(data=filled, y="y", x=XO, select="s", select_x=ZS, categorical=CAT,
                            method="twostep")
    assert_allclose(estimates(two), estimates(filled_two), rtol=1e-12)


def test_selection_missing_values_collinear_columns_and_units(data):
    holes = data.copy()
    holes.loc[[2, 30, 31], "z1"] = np.nan
    holes.loc[[7], "x2"] = np.nan
    selected_rows = holes.index[holes.s == 1][:3]
    holes.loc[selected_rows, ["y", "d"]] = np.nan            # selected, outcome missing
    holes["copy"] = 3 * holes.x1 - 2                         # collinear in both equations
    lost = sorted({2, 7, 30, 31, *selected_rows})
    clean = holes.drop(index=lost).reset_index(drop=True)
    for function, outcome, options in ((oe.heckman, "y", {}), (oe.heckprobit, "d", {}),
                                       (oe.heckman, "y", {"method": "twostep"})):
        with pytest.raises(AnalysisError) as caught:
            function(data=holes, y=outcome, x=XO, select="s", select_x=ZS, categorical=CAT,
                     **options)
        assert caught.value.code == "missing_values"
        result = function(data=holes, y=outcome, x=["x1", "copy", "x2"], select="s",
                          select_x=["x1", "copy", "z1", "kind"], categorical=CAT,
                          missing="drop", **options)
        expected = function(data=clean, y=outcome, x=XO, select="s", select_x=ZS,
                            categorical=CAT, **options)
        assert [c.term for c in result.coefficients] == [c.term for c in expected.coefficients]
        assert_allclose(estimates(result), estimates(expected), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance_of(result), covariance_of(expected), rtol=1e-7, atol=1e-12)
        assert result.provenance["omitted_terms"] == ["select:copy", "copy"]
        assert result.sample_positions == [i for i in range(len(holes)) if i not in lost]
        assert result.nobs == len(clean) and result.dropped_rows == len(lost)
        # Units of a regressor: x1 -> 1e6 x1 divides its coefficients and standard errors.
        scaled = function(data=clean.assign(x1=clean.x1 * 1e6), y=outcome, x=XO, select="s",
                          select_x=ZS, categorical=CAT, **options)
        factor = np.array([1e-6 if c.term.endswith("x1") else 1.0 for c in expected.coefficients])
        assert_allclose(estimates(scaled), estimates(expected) * factor, rtol=1e-6, atol=1e-12)
        assert_allclose(table(scaled)["std_error"], table(expected)["std_error"] * factor,
                        rtol=1e-6)


def test_selection_row_order_does_not_matter(data):
    shuffled = data.sample(frac=1.0, random_state=11).reset_index(drop=True)
    for function, outcome, options in ((oe.heckman, "y", {"cluster": "firm"}),
                                       (oe.heckprobit, "d", {"cluster": "firm"}),
                                       (oe.heckman, "y", {"method": "twostep"})):
        call = {"y": outcome, "x": XO, "select": "s", "select_x": ZS, "categorical": CAT,
                **options}
        left, right = function(data=data, **call), function(data=shuffled, **call)
        assert_allclose(estimates(left), estimates(right), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance_of(left), covariance_of(right), rtol=1e-7, atol=1e-12)


def test_heckman_two_step_truncates_rho_as_stata_rhosigma():
    """Without an exclusion restriction the two-step rho often leaves [-1, 1]: it is set to
    +-1 and sigma to |b_lambda|, and both enter the covariance."""
    for seed, sign in ((2, 1.0), (3, -1.0)):
        rng = np.random.default_rng(seed)
        n = 300
        frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
        u = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
        frame["s"] = (0.2 + 0.5 * frame.x1 + u[:, 1] > 0) * 1.0
        frame["y"] = np.where(frame.s == 1,
                              0.3 + 0.8 * frame.x1 - 0.5 * frame.x2 + u[:, 0], np.nan)
        result = oe.heckman(data=frame, y="y", x=["x1", "x2"], select="s",
                            select_x=["x1", "x2"], method="twostep")
        x, x_names = dummies(frame, ["x1", "x2"])
        selected = frame.s.to_numpy() == 1
        columns = (x, x_names, x, [f"select:{name}" for name in x_names],
                   np.where(selected, frame.y.to_numpy(), 0.0), selected)
        oracle = two_step_oracle(frame, columns=columns)
        assert oracle["rho"] == sign and result.extra["rho_truncated"] is True
        check_table(result, oracle["params"], oracle["covariance"], rtol=2e-5)
        assert_allclose([result.metrics["rho"], result.metrics["sigma"]],
                        [sign, abs(oracle["params"][-1])], rtol=1e-6)
        assert any("truncated" in warning for warning in result.warnings)
