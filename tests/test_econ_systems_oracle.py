"""Independent oracles for sureg, mvreg and reg3 (verification suite).

Everything here is derived from the textbook definitions and written without
looking at the implementation's algebra: the stacked system is built
explicitly as ``X = blockdiag(X_1 .. X_M)`` with the full Kronecker weight
``Sigma^-1 kron D`` (``D = diag(w)`` for SUR, ``D = W Z (Z'WZ)^-1 Z'W`` for 3SLS)
on small data, iterated SUR is checked against a brute-force SciPy maximization
of the full Gaussian likelihood, mvreg and the equation-by-equation methods
against statsmodels OLS / WLS / IV2SLS, and the estimators against the
invariances they must satisfy (frequency weights equal duplicated rows, row
permutations, rescaled regressors, identical regressors, exact identification).
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import optimize, stats
from statsmodels.sandbox.regression.gmm import IV2SLS

import openecon as oe

RTOL = 1e-7


# ---- data ---------------------------------------------------------------------------------


def sur_data(seed=3, n=90):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n, 5)), columns=["a", "b", "c", "f", "h"])
    e = rng.multivariate_normal([0, 0, 0], [[1.0, 0.6, -0.3], [0.6, 2.0, 0.4],
                                            [-0.3, 0.4, 0.7]], size=n)
    d["y1"] = 1 + d.a - 0.5 * d.b + e[:, 0]
    d["y2"] = -2 + 0.3 * d.c + d.a + 0.2 * d.f + e[:, 1]
    d["y3"] = 0.5 * d.h - d.b + e[:, 2]
    d["w"] = rng.uniform(0.2, 3.0, n)
    d["fw"] = rng.integers(1, 4, n).astype(float)
    d["g"] = rng.choice(["p", "q", "r"], n)
    return d


EQS = [{"y": "y1", "x": ["a", "b"]}, {"y": "y2", "x": ["c", "a", "f"]},
       {"y": "y3", "x": ["h", "b"]}]


def matrices(d, eqs, constant=True):
    xs = []
    for eq in eqs:
        cols = [d[c].to_numpy(float) for c in eq["x"]]
        if eq.get("constant", constant):
            cols = [np.ones(len(d))] + cols
        xs.append(np.column_stack(cols))
    return xs, [d[eq["y"]].to_numpy(float) for eq in eqs]


def blockdiag(xs):
    n = xs[0].shape[0]
    out = np.zeros((n * len(xs), sum(x.shape[1] for x in xs)))
    col = 0
    for i, x in enumerate(xs):
        out[i * n:(i + 1) * n, col:col + x.shape[1]] = x
        col += x.shape[1]
    return out


def system_gls(xs, ys, sigma, d_matrix):
    """GLS with Var(e) = Sigma kron D^-1, i.e. weight Omega^-1 = Sigma^-1 kron D."""
    big, y = blockdiag(xs), np.concatenate(ys)
    omega_inv = np.kron(np.linalg.inv(sigma), d_matrix)
    information = big.T @ omega_inv @ big
    cov = np.linalg.inv(information)
    return cov @ big.T @ omega_inv @ y, cov


def split_resid(xs, ys, beta):
    out, col = [], 0
    for x, y in zip(xs, ys):
        out.append(y - x @ beta[col:col + x.shape[1]])
        col += x.shape[1]
    return np.column_stack(out)


def ols_each(xs, ys, w):
    return np.concatenate([np.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * y))
                           for x, y in zip(xs, ys)])


def table(result):
    c = result.coefficients
    return (np.array([v.estimate for v in c]), np.array([v.std_error for v in c]),
            np.array([v.statistic for v in c]), np.array([v.p_value for v in c]),
            np.array([v.ci_low for v in c]), np.array([v.ci_high for v in c]))


def check_inference(result, beta, cov, df=None, alpha=0.05):
    est, se, stat, p, lo, hi = table(result)
    sd = np.sqrt(np.diag(cov))
    assert_allclose(est, beta, rtol=RTOL, atol=1e-10)
    assert_allclose(np.array(result.covariance_matrix), cov, rtol=RTOL, atol=1e-13)
    assert_allclose(se, sd, rtol=RTOL)
    assert_allclose(stat, beta / sd, rtol=1e-6)
    dist = stats.norm if df is None else stats.t(df)
    assert_allclose(p, 2 * dist.sf(np.abs(beta / sd)), rtol=1e-6, atol=1e-300)
    crit = dist.ppf(1 - alpha / 2)
    assert_allclose(lo, beta - crit * sd, rtol=1e-7, atol=1e-12)
    assert_allclose(hi, beta + crit * sd, rtol=1e-7, atol=1e-12)


def wald(beta, cov, index):
    b = beta[index]
    return float(b @ np.linalg.solve(cov[np.ix_(index, index)], b))


def slope_index(xs, constant=True):
    out, col = [], 0
    for x in xs:
        out.append(list(range(col + int(constant), col + x.shape[1])))
        col += x.shape[1]
    return out


def bp_stat(sigma, n):
    corr = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
    return n * np.sum(np.tril(corr, -1) ** 2)


# ---- sureg ----------------------------------------------------------------------------------


@pytest.mark.parametrize("weighting", ["none", "aweight", "fweight"])
def test_sureg_two_step_against_kronecker_gls(weighting):
    d = sur_data()
    xs, ys = matrices(d, EQS)
    n_rows = len(d)
    if weighting == "aweight":
        w = d.w.to_numpy() * n_rows / d.w.sum()
        nobs = n_rows
        kw = {"weights": "w", "weight_type": "aweight"}
    elif weighting == "fweight":
        w, nobs = d.fw.to_numpy(), int(d.fw.sum())
        kw = {"weights": "fw", "weight_type": "fweight"}
    else:
        w, nobs, kw = np.ones(n_rows), n_rows, {}
    e0 = split_resid(xs, ys, ols_each(xs, ys, w))
    sigma = (e0 * w[:, None]).T @ e0 / nobs
    beta, cov = system_gls(xs, ys, sigma, np.diag(w))
    result = oe.sureg(data=d, equations=EQS, **kw)
    check_inference(result, beta, cov)
    assert result.nobs == nobs
    e = split_resid(xs, ys, beta)
    rss = (w[:, None] * e ** 2).sum(axis=0)
    for i, eq in enumerate(EQS):
        y = ys[i]
        tss = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        assert_allclose(result.metrics[f"{eq['y']}:rmse"], np.sqrt(rss[i] / nobs), rtol=1e-9)
        assert_allclose(result.metrics[f"{eq['y']}:r_squared"], 1 - rss[i] / tss, rtol=1e-9)
    slopes = slope_index(xs)
    for i, eq in enumerate(EQS):
        test = result.tests[f"{eq['y']}:model"]
        assert_allclose(test["statistic"], wald(beta, cov, slopes[i]), rtol=1e-8)
        assert test["df"] == len(slopes[i])
        assert_allclose(test["p_value"], stats.chi2.sf(test["statistic"], test["df"]), rtol=1e-8)
    every = sum(slopes, [])
    assert_allclose(result.tests["model"]["statistic"], wald(beta, cov, every), rtol=1e-8)
    assert result.metrics["df_model"] == len(every)
    bp = result.tests["breusch_pagan"]
    assert_allclose(bp["statistic"], bp_stat(sigma, nobs), rtol=1e-9)
    assert bp["df"] == 3
    assert_allclose(bp["p_value"], stats.chi2.sf(bp["statistic"], 3), rtol=1e-9)
    final = (e * w[:, None]).T @ e / nobs
    ll = -nobs / 2 * (3 * (1 + np.log(2 * np.pi)) + np.log(np.linalg.det(final)))
    assert_allclose(result.metrics["log_likelihood"], ll, rtol=1e-10)
    assert_allclose(np.array(result.extra["sigma"]), sigma, rtol=1e-9)


def test_sureg_dfk_and_small():
    d = sur_data(seed=8, n=40)
    xs, ys = matrices(d, EQS)
    n = len(d)
    k = np.array([x.shape[1] for x in xs])
    e0 = split_resid(xs, ys, ols_each(xs, ys, np.ones(n)))
    sigma = e0.T @ e0 / np.sqrt(np.outer(n - k, n - k))
    beta, cov = system_gls(xs, ys, sigma, np.eye(n))
    result = oe.sureg(data=d, equations=EQS, dfk=True, small=True)
    check_inference(result, beta, cov, df=n - k[0])
    assert result.inference["df_inference"] == n - k[0]
    e = split_resid(xs, ys, beta)
    slopes = slope_index(xs)
    for i, eq in enumerate(EQS):
        assert_allclose(result.metrics[f"{eq['y']}:rmse"],
                        np.sqrt(np.sum(e[:, i] ** 2) / (n - k[i])), rtol=1e-9)
        test = result.tests[f"{eq['y']}:model"]
        q = len(slopes[i])
        assert test["distribution"] == "F" and test["df"] == q and test["df2"] == n - k[i]
        assert_allclose(test["statistic"], wald(beta, cov, slopes[i]) / q, rtol=1e-8)
        assert_allclose(test["p_value"], stats.f.sf(test["statistic"], q, n - k[i]), rtol=1e-7)
    # small alone changes only the reference distributions; dfk alone only the divisor.
    plain = oe.sureg(data=d, equations=EQS)
    small_only = oe.sureg(data=d, equations=EQS, small=True)
    assert_allclose(table(plain)[1], table(small_only)[1], rtol=1e-12)
    dfk_only = oe.sureg(data=d, equations=EQS, dfk=True)
    assert_allclose(table(dfk_only)[1], np.sqrt(np.diag(cov)), rtol=RTOL)


def test_sureg_iterated_is_the_gaussian_ml_estimate():
    d = sur_data(seed=11, n=70)
    eqs = EQS[:2]
    xs, ys = matrices(d, eqs)
    n = len(d)
    k1 = xs[0].shape[1]

    def negll(theta):
        b = theta[:-3]
        chol = np.array([[np.exp(theta[-3]), 0.0], [theta[-2], np.exp(theta[-1])]])
        e = split_resid(xs, ys, b)
        return -stats.multivariate_normal(np.zeros(2), chol @ chol.T).logpdf(e).sum()

    start = np.concatenate([ols_each(xs, ys, np.ones(n)), [0.0, 0.0, 0.0]])
    fit = optimize.minimize(negll, start, method="BFGS", options={"gtol": 1e-10})
    result = oe.sureg(data=d, equations=eqs, iterate=True, tolerance=1e-13)
    est = table(result)[0]
    assert_allclose(est, fit.x[:-3], rtol=0, atol=2e-6)
    assert_allclose(result.metrics["log_likelihood"], -fit.fun, rtol=1e-9)
    # The fixed point of the FGLS map, iterated in NumPy to machine precision.
    beta = ols_each(xs, ys, np.ones(n))
    for _ in range(500):
        e = split_resid(xs, ys, beta)
        new, cov = system_gls(xs, ys, e.T @ e / n, np.eye(n))
        if np.max(np.abs(new - beta)) < 1e-15:
            break
        beta = new
    check_inference(result, new, cov)
    assert result.extra["converged"] and est.shape == (k1 + xs[1].shape[1],)


def test_sureg_identical_regressors_reduce_to_ols():
    d = sur_data(seed=4, n=50)
    eqs = [{"y": "y1", "x": ["a", "b"]}, {"y": "y2", "x": ["a", "b"]}]
    result = oe.sureg(data=d, equations=eqs)
    for i, eq in enumerate(eqs):
        ols = sm.OLS(d[eq["y"]], sm.add_constant(d[["a", "b"]])).fit()
        assert_allclose(table(result)[0][3 * i:3 * i + 3], ols.params.to_numpy(), rtol=1e-10)


def test_sureg_frequency_weights_equal_duplicated_rows_and_invariances():
    d = sur_data(seed=5, n=45)
    weighted = oe.sureg(data=d, equations=EQS, weights="fw", weight_type="fweight", dfk=True,
                        small=True)
    expanded = d.loc[d.index.repeat(d.fw.astype(int))].reset_index(drop=True)
    plain = oe.sureg(data=expanded, equations=EQS, dfk=True, small=True)
    for a, b in zip(table(weighted), table(plain)):
        assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    assert weighted.nobs == plain.nobs == int(d.fw.sum())
    assert_allclose(weighted.tests["breusch_pagan"]["statistic"],
                    plain.tests["breusch_pagan"]["statistic"], rtol=1e-9)
    # Row order does not matter; aweights are scale free.
    shuffled = oe.sureg(data=d.sample(frac=1, random_state=2), equations=EQS)
    base = oe.sureg(data=d, equations=EQS)
    assert_allclose(table(shuffled)[0], table(base)[0], rtol=1e-10)
    assert_allclose(table(shuffled)[1], table(base)[1], rtol=1e-10)
    a1 = oe.sureg(data=d, equations=EQS, weights="w", weight_type="aweight")
    a2 = oe.sureg(data=d.assign(w=d.w * 1e5), equations=EQS, weights="w", weight_type="aweight")
    assert_allclose(table(a1)[1], table(a2)[1], rtol=1e-10)
    # Rescaling a regressor rescales its coefficient and standard error only.
    scaled = oe.sureg(data=d.assign(a=d.a * 1e6), equations=EQS)
    terms = [c.term for c in base.coefficients]
    factor = np.array([1e-6 if t.endswith(":a") else 1.0 for t in terms])
    assert_allclose(table(scaled)[0], table(base)[0] * factor, rtol=1e-9)
    assert_allclose(table(scaled)[1], table(base)[1] * factor, rtol=1e-9)


def test_sureg_badly_scaled_and_offset_data_keep_precision():
    d = sur_data(seed=6, n=80)
    shifted = d.assign(a=d.a + 1e7, c=d.c * 1e-6, y2=d.y2 * 1e5)
    xs, ys = matrices(shifted, EQS)
    n = len(d)
    # Oracle on centred, rescaled data mapped back exactly (well conditioned).
    xs_c = [np.column_stack([x[:, 0]] + [x[:, j] - x[:, j].mean() for j in range(1, x.shape[1])])
            for x in xs]
    e0 = split_resid(xs_c, ys, ols_each(xs_c, ys, np.ones(n)))
    beta_c, cov_c = system_gls(xs_c, ys, e0.T @ e0 / n, np.eye(n))
    jac, col = np.eye(len(beta_c)), 0
    for x in xs:
        jac[col, col + 1:col + x.shape[1]] = -x[:, 1:].mean(axis=0)
        col += x.shape[1]
    result = oe.sureg(data=shifted, equations=EQS)
    est, se = table(result)[:2]
    assert_allclose(est, jac @ beta_c, rtol=1e-8)
    assert_allclose(se, np.sqrt(np.diag(jac @ cov_c @ jac.T)), rtol=1e-8)


def test_sureg_constraints_equal_restricted_gls():
    d = sur_data(seed=9, n=60)
    xs, ys = matrices(d, EQS)
    n = len(d)
    e0 = split_resid(xs, ys, ols_each(xs, ys, np.ones(n)))
    beta, cov = system_gls(xs, ys, e0.T @ e0 / n, np.eye(n))
    # [y1]a = [y2]a and [y3]b = -1: positions 1, 5 and 9 of the stacked vector.
    r_mat = np.zeros((2, len(beta)))
    r_mat[0, 1], r_mat[0, 5], r_mat[1, 9] = 1, -1, 1
    r_val = np.array([0.0, -1.0])
    middle = np.linalg.inv(r_mat @ cov @ r_mat.T)
    beta_r = beta - cov @ r_mat.T @ middle @ (r_mat @ beta - r_val)
    cov_r = cov - cov @ r_mat.T @ middle @ r_mat @ cov
    result = oe.sureg(data=d, equations=EQS, constraints=[
        {"terms": {"y1:a": 1, "y2:a": -1}, "value": 0}, {"terms": {"y3:b": 1}, "value": -1}])
    terms = [c.term for c in result.coefficients]
    assert "y3:b" not in terms and result.extra["constrained_terms"] == {"y3:b": -1.0}
    keep = [i for i in range(len(beta)) if i != 9]
    est, se = table(result)[:2]
    assert_allclose(est, beta_r[keep], rtol=1e-8)
    assert_allclose(se, np.sqrt(np.diag(cov_r))[keep], rtol=1e-7)
    assert_allclose(est[terms.index("y1:a")], est[terms.index("y2:a")], rtol=1e-12)


def test_sureg_missing_categorical_and_collinear_designs():
    d = sur_data(seed=10, n=70)
    d.loc[[3, 8], "f"] = np.nan
    d.loc[5, "y3"] = np.nan
    d["a2"] = 2 * d.a
    eqs = [{"y": "y1", "x": ["a", "g", "a2"]}, {"y": "y2", "x": ["c", "f"]},
           {"y": "y3", "x": ["h", "b"], "constant": False}]
    result = oe.sureg(data=d, equations=eqs, categorical=["g"], missing="drop")
    assert result.nobs == 67 and "y1:a2" in result.provenance["omitted_terms"]
    sample = d.dropna(subset=["f", "y3"]).reset_index(drop=True)
    sample = sample.assign(gq=(sample.g == "q").astype(float), gr=(sample.g == "r").astype(float))
    xs, ys = matrices(sample, [{"y": "y1", "x": ["a", "gq", "gr"]}, {"y": "y2", "x": ["c", "f"]},
                               {"y": "y3", "x": ["h", "b"], "constant": False}])
    n = len(sample)
    e0 = split_resid(xs, ys, ols_each(xs, ys, np.ones(n)))
    beta, cov = system_gls(xs, ys, e0.T @ e0 / n, np.eye(n))
    check_inference(result, beta, cov)
    assert [c.term for c in result.coefficients][:4] == ["y1:Intercept", "y1:a", "y1:g[q]",
                                                         "y1:g[r]"]
    # y3 has no constant: its R-squared is uncentred.
    e3 = split_resid(xs, ys, beta)[:, 2]
    assert_allclose(result.metrics["y3:r_squared"], 1 - e3 @ e3 / (ys[2] @ ys[2]), rtol=1e-9)


def test_sureg_tiny_sample():
    d = sur_data(seed=12, n=7)
    eqs = [{"y": "y1", "x": ["a"]}, {"y": "y2", "x": ["c"]}]
    xs, ys = matrices(d, eqs)
    e0 = split_resid(xs, ys, ols_each(xs, ys, np.ones(7)))
    beta, cov = system_gls(xs, ys, e0.T @ e0 / 7, np.eye(7))
    check_inference(oe.sureg(data=d, equations=eqs), beta, cov)


# ---- mvreg ----------------------------------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
def test_mvreg_against_statsmodels(weighted):
    d = sur_data(seed=13, n=55)
    outcomes, regs = ["y1", "y2", "y3"], ["a", "b", "c"]
    kw = {"weights": "w", "weight_type": "aweight"} if weighted else {}
    result = oe.mvreg(data=d, y=outcomes, x=regs, corr=True, **kw)
    x = sm.add_constant(d[regs])
    n, k = len(d), 4
    w = d.w.to_numpy() * n / d.w.sum() if weighted else np.ones(n)
    fits = [sm.WLS(d[y], x, weights=w).fit() for y in outcomes]
    est, se, stat, p, lo, hi = table(result)
    assert_allclose(est, np.concatenate([f.params for f in fits]), rtol=1e-9)
    assert_allclose(se, np.concatenate([f.bse for f in fits]), rtol=1e-9)
    assert_allclose(p, np.concatenate([f.pvalues for f in fits]), rtol=1e-7)
    ci = np.vstack([f.conf_int().to_numpy() for f in fits])
    assert_allclose(lo, ci[:, 0], rtol=1e-8)
    assert_allclose(hi, ci[:, 1], rtol=1e-8)
    assert result.inference["df_inference"] == n - k
    resid = np.column_stack([f.resid for f in fits])
    sigma = (resid * w[:, None]).T @ resid / (n - k)
    xtx_inv = np.linalg.inv(x.to_numpy().T @ (w[:, None] * x.to_numpy()))
    assert_allclose(np.array(result.covariance_matrix), np.kron(sigma, xtx_inv), rtol=1e-8,
                    atol=1e-14)
    for f, y in zip(fits, outcomes):
        assert_allclose(result.metrics[f"{y}:rmse"], np.sqrt(f.mse_resid), rtol=1e-9)
        assert_allclose(result.metrics[f"{y}:r_squared"], f.rsquared, rtol=1e-9)
        test = result.tests[f"{y}:model"]
        assert_allclose(test["statistic"], f.fvalue, rtol=1e-8)
        assert_allclose(test["p_value"], f.f_pvalue, rtol=1e-6)
        assert (test["df"], test["df2"]) == (3, n - k)
    assert_allclose(result.tests["breusch_pagan"]["statistic"], bp_stat(sigma, n), rtol=1e-9)


def test_mvreg_frequency_weights_equal_duplicates():
    d = sur_data(seed=14, n=30)
    weighted = oe.mvreg(data=d, y=["y1", "y2"], x=["a", "b"], weights="fw", weight_type="fweight")
    expanded = d.loc[d.index.repeat(d.fw.astype(int))].reset_index(drop=True)
    plain = oe.mvreg(data=expanded, y=["y1", "y2"], x=["a", "b"])
    for a, b in zip(table(weighted), table(plain)):
        assert_allclose(a, b, rtol=1e-9)


# ---- reg3 -----------------------------------------------------------------------------------


def simultaneous(seed=21, n=120):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    u = rng.multivariate_normal([0, 0, 0], [[1, 0.5, 0.2], [0.5, 1.5, -0.3],
                                            [0.2, -0.3, 0.8]], size=n)
    # y1 = 1 + 0.5 y2 + z1 + u1;  y2 = -1 + 0.3 y1 + z2 - z3 + u2;  y3 = 2 + y1 + z4 + u3
    a = np.array([[1, -0.5, 0], [-0.3, 1, 0], [-1, 0, 1]])
    rhs = np.column_stack([1 + z[:, 0] + u[:, 0], -1 + z[:, 1] - z[:, 2] + u[:, 1],
                           2 + z[:, 3] + u[:, 2]])
    y = np.linalg.solve(a, rhs.T).T
    d = pd.DataFrame({"y1": y[:, 0], "y2": y[:, 1], "y3": y[:, 2], "z1": z[:, 0],
                      "z2": z[:, 1], "z3": z[:, 2], "z4": z[:, 3]})
    d["w"] = rng.uniform(0.3, 2.0, n)
    d["fw"] = rng.integers(1, 3, n).astype(float)
    d["z5"] = rng.normal(size=n)
    return d


R3 = [{"y": "y1", "x": ["y2", "z1"]}, {"y": "y2", "x": ["y1", "z2", "z3"]},
      {"y": "y3", "x": ["y1", "z4"]}]


def projection(z, w):
    zw = z * w[:, None]
    return zw @ np.linalg.solve(z.T @ zw, zw.T)


def two_sls_each(xs, ys, proj):
    return np.concatenate([np.linalg.solve(x.T @ proj @ x, x.T @ proj @ y) for x, y in zip(xs, ys)])


@pytest.mark.parametrize("weighting", ["none", "aweight", "fweight"])
def test_reg3_three_stage_against_kronecker_algebra(weighting):
    d = simultaneous()
    n = len(d)
    if weighting == "aweight":
        w, nobs, kw = d.w.to_numpy() * n / d.w.sum(), n, {"weights": "w", "weight_type": "aweight"}
    elif weighting == "fweight":
        w, nobs = d.fw.to_numpy(), int(d.fw.sum())
        kw = {"weights": "fw", "weight_type": "fweight"}
    else:
        w, nobs, kw = np.ones(n), n, {}
    xs, ys = matrices(d, R3)
    z = np.column_stack([np.ones(n), d[["z1", "z2", "z3", "z4"]].to_numpy()])
    proj = projection(z, w)
    e0 = split_resid(xs, ys, two_sls_each(xs, ys, proj))
    sigma = (e0 * w[:, None]).T @ e0 / nobs
    beta, cov = system_gls(xs, ys, sigma, proj)
    result = oe.reg3(data=d, equations=R3, **kw)
    check_inference(result, beta, cov)
    assert result.extra["instruments"] == ["Intercept", "z1", "z2", "z3", "z4"]
    assert set(result.extra["endogenous"]) == {"y1", "y2"}
    e = split_resid(xs, ys, beta)
    for i, eq in enumerate(R3):
        rss = np.sum(w * e[:, i] ** 2)
        tss = np.sum(w * (ys[i] - np.average(ys[i], weights=w)) ** 2)
        assert_allclose(result.metrics[f"{eq['y']}:rmse"], np.sqrt(rss / nobs), rtol=1e-9)
        assert_allclose(result.metrics[f"{eq['y']}:r_squared"], 1 - rss / tss, rtol=1e-9)
    slopes = slope_index(xs)
    for i, eq in enumerate(R3):
        assert_allclose(result.tests[f"{eq['y']}:model"]["statistic"],
                        wald(beta, cov, slopes[i]), rtol=1e-8)


def test_reg3_dfk_small_and_iterated():
    d = simultaneous(seed=22, n=60)
    n = len(d)
    xs, ys = matrices(d, R3)
    k = np.array([x.shape[1] for x in xs])
    z = np.column_stack([np.ones(n), d[["z1", "z2", "z3", "z4"]].to_numpy()])
    proj = projection(z, np.ones(n))
    e0 = split_resid(xs, ys, two_sls_each(xs, ys, proj))
    divisor = np.sqrt(np.outer(n - k, n - k))
    beta, cov = system_gls(xs, ys, e0.T @ e0 / divisor, proj)
    check_inference(oe.reg3(data=d, equations=R3, dfk=True, small=True), beta, cov, df=n - k[0])
    beta = two_sls_each(xs, ys, proj)
    for _ in range(1000):
        e = split_resid(xs, ys, beta)
        new, cov = system_gls(xs, ys, e.T @ e / n, proj)
        if np.max(np.abs(new - beta)) < 1e-14:
            break
        beta = new
    check_inference(oe.reg3(data=d, equations=R3, ireg3=True, tolerance=1e-13), new, cov)


def test_reg3_two_sls_and_ols_methods_against_statsmodels():
    d = simultaneous(seed=23, n=80)
    n = len(d)
    xs, ys = matrices(d, R3)
    z = np.column_stack([np.ones(n), d[["z1", "z2", "z3", "z4"]].to_numpy()])
    two = oe.reg3(data=d, equations=R3, method="2sls")
    fits = [IV2SLS(y, x, z).fit() for x, y in zip(xs, ys)]
    est, se, _, p, _, _ = table(two)
    assert_allclose(est, np.concatenate([f.params for f in fits]), rtol=1e-9)
    assert_allclose(se, np.concatenate([f.bse for f in fits]), rtol=1e-9)
    # Stata: t statistics with the first equation's degrees of freedom.
    df1 = n - xs[0].shape[1]
    assert two.inference["df_inference"] == df1
    assert_allclose(p, 2 * stats.t.sf(np.abs(est / se), df1), rtol=1e-7)
    # Independent equations: zero cross-equation covariance.
    cov = np.array(two.covariance_matrix)
    assert np.all(cov[:3, 3:] == 0)
    ols = oe.reg3(data=d, equations=R3, method="ols")
    fits = [sm.OLS(y, x).fit() for x, y in zip(xs, ys)]
    est, se = table(ols)[:2]
    assert_allclose(est, np.concatenate([f.params for f in fits]), rtol=1e-9)
    assert_allclose(se, np.concatenate([f.bse for f in fits]), rtol=1e-9)
    for f, eq in zip(fits, R3):
        assert_allclose(ols.tests[f"{eq['y']}:model"]["statistic"], f.fvalue, rtol=1e-8)
        assert_allclose(ols.metrics[f"{eq['y']}:rmse"], np.sqrt(f.mse_resid), rtol=1e-9)


def test_reg3_sure_method_equals_sureg():
    d = simultaneous(seed=24, n=70)
    a = oe.reg3(data=d, equations=R3, method="sure")
    b = oe.sureg(data=d, equations=R3)
    for left, right in zip(table(a), table(b)):
        assert_allclose(left, right, rtol=1e-10)
    exog = [{"y": "y1", "x": ["z1", "z2"]}, {"y": "y2", "x": ["z2", "z3"]},
            {"y": "y3", "x": ["z4"]}]
    a = oe.reg3(data=d, equations=exog, method="sure", ireg3=True)
    b = oe.sureg(data=d, equations=exog, iterate=True)
    assert_allclose(table(a)[0], table(b)[0], rtol=1e-10)
    # Iterating SUR on a simultaneous system heads for a singular covariance (the
    # Gaussian SUR criterion is unbounded there): refused with an explanation.
    with pytest.raises(oe.AnalysisError) as error:
        oe.reg3(data=d, equations=R3, method="sure", ireg3=True)
    assert error.value.code == "singular_sigma" and "unbounded" in str(error.value)


def test_reg3_exactly_identified_system_equals_two_sls():
    d = simultaneous(seed=25, n=90)
    eqs = [{"y": "y1", "x": ["y2", "z1"]}, {"y": "y2", "x": ["y1", "z2"]}]
    three = oe.reg3(data=d, equations=eqs)
    two = oe.reg3(data=d, equations=eqs, method="2sls")
    assert_allclose(table(three)[0], table(two)[0], rtol=1e-9)


def test_reg3_instrument_lists():
    d = simultaneous(seed=26, n=100)
    n = len(d)
    # exogenous adds z5; endogenous marks z1 as endogenous in equation 1.
    result = oe.reg3(data=d, equations=R3, exogenous=["z5"], endogenous=["z4"])
    xs, ys = matrices(d, R3)
    z = np.column_stack([np.ones(n), d[["z1", "z2", "z3", "z5"]].to_numpy()])
    proj = projection(z, np.ones(n))
    e0 = split_resid(xs, ys, two_sls_each(xs, ys, proj))
    beta, cov = system_gls(xs, ys, e0.T @ e0 / n, proj)
    check_inference(result, beta, cov)
    assert set(result.extra["endogenous"]) == {"y1", "y2", "z4"}
    # instruments= is the complete exogenous list (Stata inst()).
    full = oe.reg3(data=d, equations=R3, instruments=["z1", "z2", "z3", "z5"])
    assert_allclose(table(full)[0], beta, rtol=1e-9)
