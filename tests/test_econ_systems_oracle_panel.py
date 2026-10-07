"""Independent oracles for xtgls and xtpcse (verification suite).

Small panels are estimated with the explicit N x N error covariance ``Omega``
and the explicit block-diagonal Prais-Winsten matrix ``P`` (rows
``sqrt(1 - rho^2) e_1``, ``e_t - rho e_{t-1}``), never with the
implementation's whitening or grid algebra: GLS is
``(X'Omega^-1 X)^-1 X'Omega^-1 y`` and the Beck-Katz covariance is
``(X'X)^-1 X'Omega X (X'X)^-1`` with ``Omega`` assembled element by element
from the documented ``Sigma`` estimators. The iterated xtgls is also checked
against a brute-force maximization of the Gaussian likelihood.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import optimize, stats
from scipy.linalg import block_diag

import openecon as oe


def panel_data(seed=81, m=4, periods=14, rho=0.5, drop=None):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.5, 0.2, 0.1], [0.5, 2.0, 0.3, 0.0], [0.2, 0.3, 1.5, 0.4],
                    [0.1, 0.0, 0.4, 0.8]])[:m, :m]
    e = np.zeros((m, periods))
    for t in range(periods):
        e[:, t] = rng.multivariate_normal(np.zeros(m), cov) + (rho * e[:, t - 1] if t else 0)
    ids, ts = np.repeat(np.arange(m), periods), np.tile(np.arange(periods), m) + 1990
    x1, x2 = rng.normal(size=(2, m * periods))
    d = pd.DataFrame({"y": 1 + 2 * x1 - x2 + e.reshape(-1), "x1": x1, "x2": x2,
                      "id": ids, "t": ts})
    if drop is not None:
        d = d.drop(index=drop)
    return d.sample(frac=1, random_state=seed).reset_index(drop=True)   # shuffled input


def arrays(d):
    s = d.sort_values(["id", "t"])
    x = np.column_stack([np.ones(len(s)), s.x1, s.x2])
    return x, s.y.to_numpy(), s.id.to_numpy(), s.t.to_numpy()


def table(result):
    c = result.coefficients
    return np.array([v.estimate for v in c]), np.array([v.std_error for v in c])


def check(result, beta, cov):
    est, se = table(result)
    assert_allclose(est, beta, rtol=1e-9, atol=1e-12)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-8, atol=1e-14)
    p = np.array([c.p_value for c in result.coefficients])
    assert_allclose(p, 2 * stats.norm.sf(np.abs(beta / np.sqrt(np.diag(cov)))), rtol=1e-6,
                    atol=1e-300)
    slopes = beta[1:]
    assert_allclose(result.tests["model"]["statistic"],
                    slopes @ np.linalg.solve(cov[1:, 1:], slopes), rtol=1e-8)


def gls(x, y, omega):
    oi = np.linalg.inv(omega)
    cov = np.linalg.inv(x.T @ oi @ x)
    return cov @ x.T @ oi @ y, cov


def ols(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def rho_by_type(e, rhotype, k):
    """AR(1) coefficient of one panel's residual series (Stata's rhotype definitions)."""
    t = len(e)
    cross = np.sum(e[1:] * e[:-1])
    if rhotype == "regress":
        return cross / np.sum(e[:-1] ** 2)
    if rhotype == "freg":
        return cross / np.sum(e[1:] ** 2)
    tscorr = cross / np.sum(e ** 2)
    dw = 1 - np.sum(np.diff(e) ** 2) / np.sum(e ** 2) / 2
    return {"tscorr": tscorr, "dw": dw, "theil": tscorr * (t - k) / t,
            "nagar": (dw * t ** 2 + k ** 2) / (t ** 2 - k ** 2)}[rhotype]


def pw_matrix(rho, t):
    p = np.eye(t)
    p[0, 0] = np.sqrt(1 - rho ** 2)
    for i in range(1, t):
        p[i, i - 1] = -rho
    return p


def panel_rhos(e, ids, rhotype, k):
    return np.array([rho_by_type(e[ids == i], rhotype, k) for i in np.unique(ids)])


def transform(x, y, ids, rhos):
    sizes = [int(np.sum(ids == i)) for i in np.unique(ids)]
    p = block_diag(*[pw_matrix(r, t) for r, t in zip(rhos, sizes)])
    return p @ x, p @ y


# ---- xtgls ----------------------------------------------------------------------------------


def omega_of(e, ids, ts, panels):
    n = len(e)
    if panels == "iid":
        return np.eye(n) * np.mean(e ** 2)
    if panels == "heteroskedastic":
        s2 = {i: np.mean(e[ids == i] ** 2) for i in np.unique(ids)}
        return np.diag([s2[i] for i in ids])
    periods = len(np.unique(ts))
    grid = e.reshape(len(np.unique(ids)), periods)
    sigma = grid @ grid.T / periods
    return np.kron(sigma, np.eye(periods))


@pytest.mark.parametrize("panels", ["iid", "heteroskedastic", "correlated"])
@pytest.mark.parametrize("corr", ["independent", "ar1", "psar1"])
def test_xtgls_explicit_omega(panels, corr):
    d = panel_data()
    x, y, ids, ts = arrays(d)
    k = x.shape[1]
    if corr != "independent":
        rhos = panel_rhos(y - x @ ols(x, y), ids, "regress", k)
        if corr == "ar1":
            rhos = np.full_like(rhos, rhos.mean())
        x, y = transform(x, y, ids, rhos)
    e = y - x @ ols(x, y)
    beta, cov = gls(x, y, omega_of(e, ids, ts, panels))
    result = oe.xtgls(data=d, y="y", x=["x1", "x2"], panel="id", time="t", panels=panels,
                      corr=corr)
    check(result, beta, cov)
    if corr == "ar1":
        assert_allclose(result.metrics["rho"], rhos[0], rtol=1e-10)
    if corr == "psar1":
        assert_allclose(result.extra["rho"], rhos, rtol=1e-10)
    m = 4
    assert result.metrics["estimated_covariances"] == {"iid": 1, "heteroskedastic": m,
                                                       "correlated": m * (m + 1) // 2}[panels]
    assert result.metrics["estimated_autocorrelations"] == {"independent": 0, "ar1": 1,
                                                            "psar1": m}[corr]


@pytest.mark.parametrize("rhotype", ["regress", "freg", "tscorr", "dw", "theil", "nagar"])
def test_xtgls_rhotypes(rhotype):
    d = panel_data(seed=82, periods=20, rho=0.3)
    x, y, ids, ts = arrays(d)
    rhos = panel_rhos(y - x @ ols(x, y), ids, rhotype, 3)
    xs, ys = transform(x, y, ids, rhos)
    e = ys - xs @ ols(xs, ys)
    beta, cov = gls(xs, ys, omega_of(e, ids, ts, "heteroskedastic"))
    result = oe.xtgls(data=d, y="y", x=["x1", "x2"], panel="id", time="t",
                      panels="heteroskedastic", corr="psar1", rhotype=rhotype)
    check(result, beta, cov)
    assert_allclose(result.extra["rho"], rhos, rtol=1e-10)


def test_xtgls_unbalanced_common_rho_weights_by_pairs():
    # Panels of different lengths and start dates, gap free.
    d = panel_data(seed=83, periods=15, drop=[0, 1, 2, 15 * 2 + 14, 15 * 3 + 13, 15 * 3 + 14])
    x, y, ids, ts = arrays(d)
    sizes = np.array([np.sum(ids == i) for i in np.unique(ids)])
    rhos = panel_rhos(y - x @ ols(x, y), ids, "regress", 3)
    common = np.sum(rhos * (sizes - 1)) / np.sum(sizes - 1)
    xs, ys = transform(x, y, ids, np.full(4, common))
    e = ys - xs @ ols(xs, ys)
    beta, cov = gls(xs, ys, omega_of(e, ids, ts, "heteroskedastic"))
    result = oe.xtgls(data=d, y="y", x=["x1", "x2"], panel="id", time="t",
                      panels="heteroskedastic", corr="ar1")
    check(result, beta, cov)
    assert_allclose(result.metrics["rho"], common, rtol=1e-10)
    assert result.metrics["obs_per_group_min"] == sizes.min()


@pytest.mark.parametrize("panels", ["heteroskedastic", "correlated"])
def test_xtgls_iterated_is_maximum_likelihood(panels):
    d = panel_data(seed=84, periods=16, rho=0.0)
    x, y, ids, ts = arrays(d)
    m, periods = 4, 16

    def loglik(beta):
        e = y - x @ beta
        if panels == "heteroskedastic":
            s2 = np.array([np.mean(e[ids == i] ** 2) for i in range(m)])
            return -0.5 * np.sum(periods * (np.log(2 * np.pi * s2) + 1))
        grid = e.reshape(m, periods)
        sign, logdet = np.linalg.slogdet(grid @ grid.T / periods)
        return -0.5 * (len(e) * (np.log(2 * np.pi) + 1) + periods * logdet)

    fit = optimize.minimize(lambda b: -loglik(b), ols(x, y), method="BFGS",
                            options={"gtol": 1e-11})
    result = oe.xtgls(data=d, y="y", x=["x1", "x2"], panel="id", time="t", panels=panels,
                      igls=True, tolerance=1e-12, max_iterations=500)
    est = table(result)[0]
    assert_allclose(est, fit.x, rtol=0, atol=2e-6)
    assert_allclose(result.metrics["log_likelihood"], -fit.fun, rtol=1e-9)
    # At the fixed point the covariance is the GLS one with Omega from the final residuals.
    e = y - x @ est
    beta, cov = gls(x, y, omega_of(e, ids, ts, panels))
    assert_allclose(est, beta, rtol=1e-8)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=1e-6)


def test_xtgls_iid_is_ols_with_n_divisor_and_likelihood():
    d = panel_data(seed=85)
    x, y, ids, ts = arrays(d)
    beta = ols(x, y)
    e = y - x @ beta
    n = len(y)
    s2 = e @ e / n
    result = oe.xtgls(data=d, y="y", x=["x1", "x2"], panel="id", time="t")
    check(result, beta, s2 * np.linalg.inv(x.T @ x))
    assert_allclose(result.metrics["log_likelihood"],
                    stats.norm(0, np.sqrt(s2)).logpdf(e).sum(), rtol=1e-10)


# ---- xtpcse ---------------------------------------------------------------------------------


def pcse_omega(e, ids, ts, kind):
    """Omega of the Beck-Katz covariance, element by element."""
    panels, periods = np.unique(ids), np.unique(ts)
    common = [t for t in periods if all(np.any((ids == i) & (ts == t)) for i in panels)]
    series = {i: dict(zip(ts[ids == i], e[ids == i])) for i in panels}
    n = len(e)
    if kind == "independent":
        return np.eye(n) * np.mean(e ** 2)

    def sigma(i, j):
        if kind in ("casewise", "het_casewise"):
            shared = common
        else:
            shared = sorted(set(series[i]) & set(series[j]))
        return np.mean([series[i][t] * series[j][t] for t in shared])

    omega = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if ts[a] != ts[b]:
                continue
            if kind.startswith("het") and ids[a] != ids[b]:
                continue
            omega[a, b] = sigma(ids[a], ids[b])
    return omega


def beck_katz(x, e, omega):
    bread = np.linalg.inv(x.T @ x)
    return bread @ x.T @ omega @ x @ bread


@pytest.mark.parametrize("option", ["casewise", "pairwise", "het_casewise", "het_pairwise",
                                    "independent"])
@pytest.mark.parametrize("balanced", [True, False])
def test_xtpcse_explicit_beck_katz(option, balanced):
    drop = None if balanced else [0, 1, 15 + 7, 15 * 2 + 14, 15 * 3 + 3]
    d = panel_data(seed=86, periods=15, drop=drop)
    x, y, ids, ts = arrays(d)
    beta = ols(x, y)
    e = y - x @ beta
    cov = beck_katz(x, e, pcse_omega(e, ids, ts, option))
    flags = {"pairwise": option.endswith("pairwise"), "hetonly": option.startswith("het"),
             "independent": option == "independent"}
    result = oe.xtpcse(data=d, y="y", x=["x1", "x2"], panel="id", time="t", **flags)
    check(result, beta, cov)
    assert_allclose(result.metrics["r_squared"], 1 - e @ e / np.sum((y - y.mean()) ** 2),
                    rtol=1e-10)


@pytest.mark.parametrize("correlation", ["ar1", "psar1"])
@pytest.mark.parametrize("np1", [False, True])
def test_xtpcse_prais_winsten(correlation, np1):
    d = panel_data(seed=87, periods=15, drop=[0, 1, 15 * 3 + 14])
    x, y, ids, ts = arrays(d)
    sizes = np.array([np.sum(ids == i) for i in np.unique(ids)])
    rhos = panel_rhos(y - x @ ols(x, y), ids, "regress", 3)
    if correlation == "ar1":
        weights = sizes if np1 else sizes - 1
        rhos = np.full(4, np.sum(rhos * weights) / np.sum(weights))
    xs, ys = transform(x, y, ids, rhos)
    beta = ols(xs, ys)
    e = ys - xs @ beta
    cov = beck_katz(xs, e, pcse_omega(e, ids, ts, "casewise"))
    result = oe.xtpcse(data=d, y="y", x=["x1", "x2"], panel="id", time="t",
                       correlation=correlation, np1=np1)
    check(result, beta, cov)
    assert_allclose(result.metrics["r_squared"], 1 - e @ e / np.sum((ys - ys.mean()) ** 2),
                    rtol=1e-10)
    if correlation == "ar1":
        assert_allclose(result.metrics["rho"], rhos[0], rtol=1e-10)


def test_xtpcse_singleton_panel_with_common_rho():
    # A panel observed once carries no autocorrelation; np1 must not let it poison rho.
    d = panel_data(seed=88, periods=12)
    single = pd.DataFrame({"y": [0.3], "x1": [0.1], "x2": [-0.2], "id": [9], "t": [1995]})
    d = pd.concat([d, single], ignore_index=True)
    x, y, ids, ts = arrays(d)
    sizes = np.array([np.sum(ids == i) for i in np.unique(ids)])
    e0 = y - x @ ols(x, y)
    rhos = np.array([rho_by_type(e0[ids == i], "regress", 3) if np.sum(ids == i) > 1 else 0.0
                     for i in np.unique(ids)])
    keep = sizes > 1
    for np1 in (False, True):
        weights = (sizes if np1 else sizes - 1) * keep
        common = np.sum(rhos * weights) / np.sum(weights)
        result = oe.xtpcse(data=d, y="y", x=["x1", "x2"], panel="id", time="t",
                           correlation="ar1", np1=np1, pairwise=True)
        assert_allclose(result.metrics["rho"], common, rtol=1e-10)
        xs, ys = transform(x, y, ids, np.full(len(sizes), common))
        beta = ols(xs, ys)
        e = ys - xs @ beta
        check(result, beta, beck_katz(xs, e, pcse_omega(e, ids, ts, "pairwise")))
