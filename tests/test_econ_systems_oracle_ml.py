"""Independent oracles for gmm and frontier (verification suite).

GMM: the two-step, iterated and one-step estimators are recomputed with
explicit NumPy algebra (linear moments) or by brute-force SciPy minimization
of ``g'Wg`` with finite-difference Jacobians (nonlinear and multi-equation
moments); the moment covariances (robust, cluster, HAC Bartlett, unadjusted,
centred) are written out from their definitions.

frontier: the half-normal, exponential and truncated-normal log likelihoods
are written independently with ``scipy.special.log_ndtr`` and maximized by
SciPy followed by Newton polishing on finite differences; covariances come
from numerically differentiated Hessians and scores; efficiency scores are
checked by numerical integration of the conditional density of ``u`` given
the composed error.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, optimize, stats
from scipy.special import expit, log_ndtr

import openecon as oe


def table(result):
    c = result.coefficients
    return (np.array([v.estimate for v in c]), np.array([v.std_error for v in c]),
            np.array([v.p_value for v in c]), np.array([v.ci_low for v in c]))


def num_grad(f, x, h=1e-6):
    x = np.asarray(x, float)
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))]).T


def num_hess(f, x, h=1e-4):
    k = len(x)
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            out[i, j] = out[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                                     + f(x - ei - ej)) / (4 * h * h)
    return out


def polish(f, x, steps=6):
    """Newton steps on finite differences: maximizes f to ~1e-9 from a close start."""
    for _ in range(steps):
        x = x - np.linalg.solve(num_hess(f, x), num_grad(f, x))
    return x


# ---- GMM ------------------------------------------------------------------------------------


def iv_data(seed=31, n=300):
    rng = np.random.default_rng(seed)
    z1, z2, x1 = rng.normal(size=(3, n))
    v = rng.normal(size=n)
    x2 = 0.6 * z1 + 0.4 * z2 + v
    noise = (0.7 * v + rng.normal(size=n)) * (1 + 0.6 * np.abs(x1))
    d = pd.DataFrame({"y": 1 + 0.5 * x1 - x2 + noise, "x1": x1, "x2": x2, "z1": z1, "z2": z2,
                      "w": rng.uniform(0.3, 2.5, n), "fw": rng.integers(1, 4, n).astype(float),
                      "g": rng.integers(0, 30, n), "t": rng.permutation(n)})
    return d


LINEAR = ["y - {b0} - {b1}*x1 - {b2}*x2"]


def linear_parts(d):
    n = len(d)
    x = np.column_stack([np.ones(n), d.x1, d.x2])
    z = np.column_stack([np.ones(n), d.x1, d.z1, d.z2])
    return x, z, d.y.to_numpy()


def gmm_linear(x, z, y, weight, w):
    zw = z * w[:, None]
    a = x.T @ zw @ weight @ zw.T @ x
    return np.linalg.solve(a, x.T @ zw @ weight @ zw.T @ y), a


@pytest.mark.parametrize("weighting", ["none", "aweight", "pweight", "fweight"])
def test_linear_two_step_gmm_explicit(weighting):
    d = iv_data()
    x, z, y = linear_parts(d)
    n = len(d)
    if weighting == "fweight":
        w, sq, kw = d.fw.to_numpy(), d.fw.to_numpy(), {"weights": "fw", "weight_type": "fweight"}
        nobs = int(d.fw.sum())
    elif weighting == "none":
        w, sq, kw, nobs = np.ones(n), np.ones(n), {}, n
    else:
        w = d.w.to_numpy() * n / d.w.sum()
        sq, kw, nobs = w ** 2, {"weights": "w", "weight_type": weighting}, n
    first, _ = gmm_linear(x, z, y, np.linalg.inv(z.T @ (w[:, None] * z)), w)
    u1 = y - x @ first
    s = (z * (sq * u1 ** 2)[:, None]).T @ z
    weight = np.linalg.inv(s)
    beta, info = gmm_linear(x, z, y, weight, w)
    cov = np.linalg.inv(info)
    result = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"],
                    winitial="unadjusted", **kw)
    est, se, p, lo = table(result)
    assert_allclose(est, beta, rtol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-8)
    assert_allclose(p, 2 * stats.norm.sf(np.abs(beta / np.sqrt(np.diag(cov)))), rtol=1e-6)
    assert_allclose(lo, beta - stats.norm.ppf(0.975) * np.sqrt(np.diag(cov)), rtol=1e-8)
    g = z.T @ (w * (y - x @ beta))
    j = g @ weight @ g
    test = result.tests["hansen_j"]
    assert_allclose(test["statistic"], j, rtol=1e-8)
    assert test["df"] == 1
    assert_allclose(test["p_value"], stats.chi2.sf(j, 1), rtol=1e-7)
    assert_allclose(result.metrics["criterion"], j / nobs, rtol=1e-8)
    assert result.nobs == nobs


def test_linear_gmm_frequency_weights_equal_duplicates():
    d = iv_data(seed=32, n=120)
    a = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"], weights="fw",
               weight_type="fweight", winitial="unadjusted")
    expanded = d.loc[d.index.repeat(d.fw.astype(int))].reset_index(drop=True)
    b = oe.gmm(data=expanded, moments=LINEAR, instruments=["x1", "z1", "z2"],
               winitial="unadjusted")
    for left, right in zip(table(a), table(b)):
        assert_allclose(left, right, rtol=1e-8)
    assert_allclose(a.tests["hansen_j"]["statistic"], b.tests["hansen_j"]["statistic"], rtol=1e-8)


def test_one_step_identity_gmm_sandwich():
    d = iv_data(seed=33)
    x, z, y = linear_parts(d)
    n = len(d)
    beta, info = gmm_linear(x, z, y, np.eye(4), np.ones(n))
    u = y - x @ beta
    s = (z * (u ** 2)[:, None]).T @ z
    bread = np.linalg.inv(info)
    grad = x.T @ z
    cov = bread @ grad @ s @ grad.T @ bread
    result = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"], twostep=False)
    est, se = table(result)[:2]
    assert_allclose(est, beta, rtol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-8)
    assert "hansen_j" not in result.tests


def test_unadjusted_weight_matrix_and_nonrobust_covariance():
    d = iv_data(seed=34)
    x, z, y = linear_parts(d)
    n = len(d)
    proj = z @ np.linalg.solve(z.T @ z, z.T)
    tsls = np.linalg.solve(x.T @ proj @ x, x.T @ proj @ y)
    sigma2 = np.sum((y - x @ tsls) ** 2) / n
    result = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"],
                    winitial="unadjusted", wmatrix="unadjusted", covariance="nonrobust")
    est = table(result)[0]
    assert_allclose(est, tsls, rtol=1e-9)
    assert_allclose(np.array(result.covariance_matrix),
                    sigma2 * np.linalg.inv(x.T @ proj @ x), rtol=1e-8)
    # The robust sandwich after an unadjusted weight matrix.
    robust = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"],
                    winitial="unadjusted", wmatrix="unadjusted", covariance="robust")
    u = y - x @ tsls
    weight = np.linalg.inv(sigma2 * z.T @ z)
    grad = x.T @ z
    bread = np.linalg.inv(grad @ weight @ grad.T)
    s = (z * (u ** 2)[:, None]).T @ z
    assert_allclose(np.array(robust.covariance_matrix),
                    bread @ grad @ weight @ s @ weight @ grad.T @ bread, rtol=1e-8)


def test_cluster_and_centred_weight_matrices():
    d = iv_data(seed=35)
    x, z, y = linear_parts(d)
    n = len(d)
    codes = d.g.to_numpy()
    groups = len(np.unique(codes))

    def cluster_s(u, centre=False):
        m = z * u[:, None]
        if centre:
            m = m - m.mean(axis=0)
        sums = np.zeros((codes.max() + 1, z.shape[1]))
        np.add.at(sums, codes, m)
        return sums.T @ sums

    first, _ = gmm_linear(x, z, y, np.linalg.inv(z.T @ z), np.ones(n))
    weight = np.linalg.inv(cluster_s(y - x @ first))
    beta, info = gmm_linear(x, z, y, weight, np.ones(n))
    result = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"],
                    winitial="unadjusted", cluster="g")
    assert_allclose(table(result)[0], beta, rtol=1e-9)
    # Documented convention: the efficient form times G/(G-1).
    assert_allclose(np.array(result.covariance_matrix),
                    np.linalg.inv(info) * groups / (groups - 1), rtol=1e-8)
    g = z.T @ (y - x @ beta)
    assert_allclose(result.tests["hansen_j"]["statistic"], g @ weight @ g, rtol=1e-8)
    # center: robust S of the demeaned moments.
    m1 = z * (y - x @ first)[:, None]
    m1 = m1 - m1.mean(axis=0)
    weight = np.linalg.inv(m1.T @ m1)
    beta, info = gmm_linear(x, z, y, weight, np.ones(n))
    centred = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"],
                     winitial="unadjusted", center=True)
    assert_allclose(table(centred)[0], beta, rtol=1e-9)
    assert_allclose(np.array(centred.covariance_matrix), np.linalg.inv(info), rtol=1e-8)


def test_hac_weight_matrix_in_time_order():
    d = iv_data(seed=36, n=200)
    x, z, y = linear_parts(d)
    n = len(d)
    order = np.argsort(d.t.to_numpy())
    lags = 3

    def hac_s(u):
        m = (z * u[:, None])[order]
        s = m.T @ m
        for lag in range(1, lags + 1):
            gamma = m[lag:].T @ m[:-lag]
            s += (1 - lag / (lags + 1)) * (gamma + gamma.T)
        return s

    first, _ = gmm_linear(x, z, y, np.linalg.inv(z.T @ z), np.ones(n))
    weight = np.linalg.inv(hac_s(y - x @ first))
    beta, info = gmm_linear(x, z, y, weight, np.ones(n))
    result = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"],
                    winitial="unadjusted", wmatrix="hac", lags=lags, time="t")
    assert_allclose(table(result)[0], beta, rtol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), np.linalg.inv(info), rtol=1e-8)


def test_iterated_gmm_fixed_point():
    d = iv_data(seed=37)
    x, z, y = linear_parts(d)
    n = len(d)
    beta, _ = gmm_linear(x, z, y, np.linalg.inv(z.T @ z), np.ones(n))
    for _ in range(500):
        u = y - x @ beta
        weight = np.linalg.inv((z * (u ** 2)[:, None]).T @ z)
        new, info = gmm_linear(x, z, y, weight, np.ones(n))
        if np.max(np.abs(new - beta)) < 1e-14:
            break
        beta = new
    result = oe.gmm(data=d, moments=LINEAR, instruments=["x1", "z1", "z2"],
                    winitial="unadjusted", igmm=True, igmm_tolerance=1e-13)
    assert_allclose(table(result)[0], new, rtol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), np.linalg.inv(info), rtol=1e-7)


def test_nonlinear_exponential_moments_brute_force():
    rng = np.random.default_rng(38)
    n = 500
    x1, x2, z = rng.normal(size=(3, n))
    y = rng.poisson(np.exp(0.2 + 0.4 * x1 - 0.3 * x2)).astype(float)
    d = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z})
    inst = np.column_stack([np.ones(n), x1, x2, z])

    def gbar(t):
        return inst.T @ (y - np.exp(t[0] + t[1] * x1 + t[2] * x2))

    def solve(weight, start):
        fit = optimize.minimize(lambda t: gbar(t) @ weight @ gbar(t), start, method="BFGS",
                                options={"gtol": 1e-11})
        return polish(lambda t: -gbar(t) @ weight @ gbar(t), fit.x, steps=3)

    first = solve(np.eye(4), np.zeros(3))
    u = y - np.exp(first[0] + first[1] * x1 + first[2] * x2)
    weight = np.linalg.inv((inst * (u ** 2)[:, None]).T @ inst)
    beta = solve(weight, first)
    jac = num_grad(gbar, beta)
    cov = np.linalg.inv(jac.T @ weight @ jac)
    result = oe.gmm(data=d, moments=["y - exp({b0} + {b1}*x1 + {b2}*x2)"],
                    instruments=["x1", "x2", "z"])
    assert_allclose(table(result)[0], beta, rtol=1e-6, atol=1e-8)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-5)
    assert_allclose(result.tests["hansen_j"]["statistic"], gbar(beta) @ weight @ gbar(beta),
                    rtol=1e-6)


def test_two_equation_moments_with_shared_parameter():
    rng = np.random.default_rng(39)
    n = 400
    x, z1, z2 = rng.normal(size=(3, n))
    e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    y1 = 1 + 0.5 * x + e[:, 0]
    y2 = -1 + 0.5 * z1 + 0.3 * z2 + e[:, 1]
    d = pd.DataFrame({"y1": y1, "y2": y2, "x": x, "z1": z1, "z2": z2})
    za = np.column_stack([np.ones(n), x, z1])
    zb = np.column_stack([np.ones(n), z1, z2])

    def resid(t):
        return np.column_stack([y1 - t[0] - t[1] * x, y2 - t[2] - t[1] * z1 - t[3] * z2])

    def gbar(t):
        u = resid(t)
        return np.concatenate([za.T @ u[:, 0], zb.T @ u[:, 1]])

    def rows(t):
        u = resid(t)
        return np.column_stack([za * u[:, :1], zb * u[:, 1:]])

    def solve(weight, start):
        fit = optimize.minimize(lambda t: gbar(t) @ weight @ gbar(t), start, method="BFGS",
                                options={"gtol": 1e-11})
        return polish(lambda t: -gbar(t) @ weight @ gbar(t), fit.x, steps=3)

    # winitial unadjusted with unit residual covariance: blockdiag (Z_j'Z_j)^-1.
    w0 = np.zeros((6, 6))
    w0[:3, :3], w0[3:, 3:] = np.linalg.inv(za.T @ za), np.linalg.inv(zb.T @ zb)
    first = solve(w0, np.zeros(4))
    m = rows(first)
    weight = np.linalg.inv(m.T @ m)
    beta = solve(weight, first)
    jac = num_grad(gbar, beta)
    cov = np.linalg.inv(jac.T @ weight @ jac)
    result = oe.gmm(data=d, moments={"a": "y1 - {a0} - {b}*x", "b": "y2 - {c0} - {b}*z1 - {c2}*z2"},
                    instruments=[["x", "z1"], ["z1", "z2"]], winitial="unadjusted")
    assert [c.term for c in result.coefficients] == ["a0", "b", "c0", "c2"]
    assert_allclose(table(result)[0], beta, rtol=1e-7, atol=1e-9)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-6)
    assert result.tests["hansen_j"]["df"] == 2
    # Unadjusted S across equations: s_jk Z_j'Z_k with s_jk = u_j'u_k / N.
    u = resid(first)
    sig = u.T @ u / n
    s = np.block([[sig[0, 0] * za.T @ za, sig[0, 1] * za.T @ zb],
                  [sig[1, 0] * zb.T @ za, sig[1, 1] * zb.T @ zb]])
    weight = np.linalg.inv(s)
    beta = solve(weight, first)
    jac = num_grad(gbar, beta)
    unadj = oe.gmm(data=d, moments={"a": "y1 - {a0} - {b}*x", "b": "y2 - {c0} - {b}*z1 - {c2}*z2"},
                   instruments=[["x", "z1"], ["z1", "z2"]], winitial="unadjusted",
                   wmatrix="unadjusted", covariance="nonrobust")
    assert_allclose(table(unadj)[0], beta, rtol=1e-7, atol=1e-9)
    assert_allclose(np.array(unadj.covariance_matrix), np.linalg.inv(jac.T @ weight @ jac),
                    rtol=1e-6)


# ---- frontier -------------------------------------------------------------------------------


def frontier_data(seed=41, n=400, cost=False, dist="hnormal"):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    if dist == "exponential":
        u = rng.exponential(0.5, n)
    elif dist == "tnormal":
        u = stats.truncnorm.rvs(-0.4 / 0.6, np.inf, loc=0.4, scale=0.6, size=n, random_state=rng)
    else:
        u = np.abs(rng.normal(0, 0.7, n))
    s = -1 if cost else 1
    y = 1 + 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.35, n) - s * u
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "fw": rng.integers(1, 4, n).astype(float),
                         "pw": rng.uniform(0.5, 2.0, n), "g": rng.integers(0, 40, n)})


def loglik_rows(theta, y, x, dist, s):
    """Per-observation log densities in Stata's parameterization (written from [R] frontier)."""
    k = x.shape[1]
    e = y - x @ theta[:k]
    if dist == "tnormal":
        mu, s2, gamma = theta[k], np.exp(theta[k + 1]), expit(theta[k + 2])
        su2, sv2 = gamma * s2, (1 - gamma) * s2
        mu_star = (mu * sv2 - s * e * su2) / s2
        s_star = np.sqrt(su2 * sv2 / s2)
        return (-0.5 * np.log(2 * np.pi * s2) - (s * e + mu) ** 2 / (2 * s2)
                + log_ndtr(mu_star / s_star) - log_ndtr(mu / np.sqrt(su2)))
    sv, su = np.exp(theta[k] / 2), np.exp(theta[k + 1] / 2)
    if dist == "hnormal":
        sig = np.sqrt(sv ** 2 + su ** 2)
        return (0.5 * np.log(2 / np.pi) - np.log(sig) + log_ndtr(-s * e * (su / sv) / sig)
                - e ** 2 / (2 * sig ** 2))
    return -np.log(su) + sv ** 2 / (2 * su ** 2) + s * e / su + log_ndtr(-s * e / sv - sv / su)


def brute_force_frontier(d, dist, cost, w=None):
    y = d.y.to_numpy()
    x = np.column_stack([np.ones(len(d)), d.x1, d.x2])
    w = np.ones(len(d)) if w is None else w
    s = -1 if cost else 1

    def ll(t):
        return float(w @ loglik_rows(t, y, x, dist, s))

    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    start = np.concatenate([ols, [0.0, np.log(0.3), 0.0] if dist == "tnormal"
                            else [np.log(0.1), np.log(0.3)]])
    fit = optimize.minimize(lambda t: -ll(t), start, method="BFGS",
                            options={"gtol": 1e-9, "maxiter": 5000})
    theta = polish(ll, fit.x)
    return theta, ll, (y, x, s)


CASES = [(dist, cost) for dist in ("hnormal", "exponential", "tnormal") for cost in (False, True)]


@pytest.mark.parametrize("dist,cost", CASES)
def test_frontier_maximum_likelihood_brute_force(dist, cost):
    d = frontier_data(seed=41 + CASES.index((dist, cost)), cost=cost, dist=dist)
    theta, ll, _ = brute_force_frontier(d, dist, cost)
    cov = np.linalg.inv(-num_hess(ll, theta))
    result = oe.frontier(data=d, y="y", x=["x1", "x2"], distribution=dist, cost=cost)
    est, se, p, lo = table(result)
    assert_allclose(est, theta, rtol=1e-6, atol=1e-7)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=2e-4, atol=1e-9)
    assert_allclose(p, 2 * stats.norm.sf(np.abs(est / se)), rtol=1e-6, atol=1e-300)
    names = ["Intercept", "x1", "x2"] + (["/mu", "/lnsigma2", "/ilgtgamma"] if dist == "tnormal"
                                         else ["/lnsig2v", "/lnsig2u"])
    assert [c.term for c in result.coefficients] == names
    n, k = len(d), len(theta)
    assert_allclose(result.metrics["log_likelihood"], ll(theta), rtol=1e-10)
    assert_allclose(result.metrics["aic"], -2 * ll(theta) + 2 * k, rtol=1e-10)
    assert_allclose(result.metrics["bic"], -2 * ll(theta) + np.log(n) * k, rtol=1e-10)
    # Derived variance parameters and delta-method standard errors.
    if dist == "tnormal":
        transforms = {"sigma2": lambda t: np.exp(t[4]), "gamma": lambda t: expit(t[5]),
                      "sigma_u2": lambda t: expit(t[5]) * np.exp(t[4]),
                      "sigma_v2": lambda t: (1 - expit(t[5])) * np.exp(t[4])}
    else:
        transforms = {"sigma_v": lambda t: np.exp(t[3] / 2), "sigma_u": lambda t: np.exp(t[4] / 2),
                      "sigma2": lambda t: np.exp(t[3]) + np.exp(t[4]),
                      "lambda": lambda t: np.exp((t[4] - t[3]) / 2)}
    for name, f in transforms.items():
        grad = num_grad(f, theta)
        assert_allclose(result.metrics[name], f(theta), rtol=1e-6)
        assert_allclose(result.extra["ancillary"][name]["std_error"], np.sqrt(grad @ cov @ grad),
                        rtol=5e-4)
    # LR test of sigma_u = 0 against OLS, chibar2(01).
    resid = d.y - np.column_stack([np.ones(n), d.x1, d.x2]) @ np.linalg.lstsq(
        np.column_stack([np.ones(n), d.x1, d.x2]), d.y, rcond=None)[0]
    ll_ols = stats.norm(0, np.sqrt(np.mean(resid ** 2))).logpdf(resid).sum()
    lr = 2 * (ll(theta) - ll_ols)
    test = result.tests["sigma_u"]
    assert_allclose(test["statistic"], lr, rtol=1e-8)
    assert_allclose(test["p_value"], 0.5 * stats.chi2.sf(lr, 1), rtol=1e-6)


@pytest.mark.parametrize("dist", ["hnormal", "exponential", "tnormal"])
def test_frontier_robust_opg_cluster_covariances(dist):
    d = frontier_data(seed=51, dist=dist)
    theta, ll, (y, x, s) = brute_force_frontier(d, dist, False)
    hess = num_hess(ll, theta)
    scores = num_grad(lambda t: loglik_rows(t, y, x, dist, s), theta)
    bread = np.linalg.inv(-hess)
    n = len(d)
    robust = oe.frontier(data=d, y="y", x=["x1", "x2"], distribution=dist, covariance="robust")
    assert_allclose(np.array(robust.covariance_matrix),
                    n / (n - 1) * bread @ scores.T @ scores @ bread, rtol=5e-4, atol=1e-10)
    opg = oe.frontier(data=d, y="y", x=["x1", "x2"], distribution=dist, covariance="opg")
    assert_allclose(np.array(opg.covariance_matrix), np.linalg.inv(scores.T @ scores), rtol=1e-5)
    clustered = oe.frontier(data=d, y="y", x=["x1", "x2"], distribution=dist, cluster="g")
    codes = d.g.to_numpy()
    sums = np.zeros((codes.max() + 1, len(theta)))
    np.add.at(sums, codes, scores)
    groups = len(np.unique(codes))
    assert_allclose(np.array(clustered.covariance_matrix),
                    groups / (groups - 1) * bread @ sums.T @ sums @ bread, rtol=5e-4, atol=1e-10)
    assert "sigma_u" not in robust.tests and "sigma_u" not in clustered.tests


def test_frontier_weights():
    d = frontier_data(seed=52, n=250)
    weighted = oe.frontier(data=d, y="y", x=["x1", "x2"], weights="fw", weight_type="fweight")
    expanded = d.loc[d.index.repeat(d.fw.astype(int))].reset_index(drop=True)
    plain = oe.frontier(data=expanded, y="y", x=["x1", "x2"])
    for a, b in zip(table(weighted), table(plain)):
        assert_allclose(a, b, rtol=1e-7, atol=1e-10)
    assert weighted.nobs == plain.nobs
    assert_allclose(weighted.metrics["log_likelihood"], plain.metrics["log_likelihood"],
                    rtol=1e-10)
    assert_allclose(weighted.tests["sigma_u"]["statistic"], plain.tests["sigma_u"]["statistic"],
                    rtol=1e-7)
    # pweights: weighted likelihood, robust sandwich of the weighted scores.
    pw = d.pw.to_numpy()
    theta, ll, (y, x, s) = brute_force_frontier(d, "hnormal", False, w=pw)
    scores = num_grad(lambda t: loglik_rows(t, y, x, "hnormal", s), theta) * pw[:, None]
    bread = np.linalg.inv(-num_hess(ll, theta))
    n = len(d)
    result = oe.frontier(data=d, y="y", x=["x1", "x2"], weights="pw", weight_type="pweight")
    assert result.spec.covariance == "robust"
    assert_allclose(table(result)[0], theta, rtol=1e-6, atol=1e-7)
    assert_allclose(np.array(result.covariance_matrix),
                    n / (n - 1) * bread @ scores.T @ scores @ bread, rtol=5e-4, atol=1e-10)


def test_frontier_tnormal_without_interior_maximum_is_refused():
    # On this sample the truncated-normal likelihood keeps rising as mu -> -inf (the
    # exponential limit; checked with Nelder-Mead): no estimate may be reported.
    d = frontier_data(seed=66, n=300, cost=True, dist="tnormal")
    with pytest.raises(oe.AnalysisError) as error:
        oe.frontier(data=d, y="y", x=["x1", "x2"], distribution="tnormal", cost=True)
    assert error.value.code == "boundary_solution" and "exponential" in str(error.value)


@pytest.mark.parametrize("dist,cost", CASES)
def test_frontier_efficiency_by_numerical_integration(dist, cost):
    d = frontier_data(seed=71 + CASES.index((dist, cost)), n=300, cost=cost, dist=dist)
    result = oe.frontier(data=d, y="y", x=["x1", "x2"], distribution=dist, cost=cost)
    eff = oe.frontier_efficiency(result, d)
    b = {c.term: c.estimate for c in result.coefficients}
    e = d.y - b["Intercept"] - b["x1"] * d.x1 - b["x2"] * d.x2
    assert_allclose(eff["residual"], e, rtol=1e-10, atol=1e-12)
    s = -1 if cost else 1
    if dist == "tnormal":
        s2, gamma = np.exp(b["/lnsigma2"]), expit(b["/ilgtgamma"])
        su, sv, mu = np.sqrt(gamma * s2), np.sqrt((1 - gamma) * s2), b["/mu"]
        density_u = stats.truncnorm(-mu / su, np.inf, loc=mu, scale=su).pdf
    else:
        sv, su = np.exp(b["/lnsig2v"] / 2), np.exp(b["/lnsig2u"] / 2)
        density_u = (stats.halfnorm(scale=su).pdf if dist == "hnormal"
                     else stats.expon(scale=su).pdf)
    for i in [0, 7, 50, 123, 299]:
        joint = lambda u, ei=e[i]: stats.norm.pdf(ei + s * u, scale=sv) * density_u(u)  # noqa: E731
        top = 20 * (su + sv) + abs(e[i])
        mass = integrate.quad(joint, 0, top, epsabs=0, epsrel=1e-11, limit=200)[0]
        mean = integrate.quad(lambda u: u * joint(u), 0, top, epsabs=0, epsrel=1e-11,
                              limit=200)[0] / mass
        te = integrate.quad(lambda u: np.exp(-s * u) * joint(u), 0, top, epsabs=0, epsrel=1e-11,
                            limit=200)[0] / mass
        assert_allclose(eff["u"][i], mean, rtol=1e-6)
        assert_allclose(eff["te"][i], te, rtol=1e-6)
    assert np.all(eff["u"] > 0) and np.all((eff["te"] < 1) if not cost else (eff["te"] > 1))
