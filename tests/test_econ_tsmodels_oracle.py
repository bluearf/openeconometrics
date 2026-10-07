"""Independent oracles for the tsmodels family (verification pass).

These checks are written from first principles and do not reuse the
implementation's formulas or the implementer's test code:

- prais: GLS with the explicit AR(1) correlation matrix (small n), the rho fixed
  point found by a scalar root finder, Stata's ANOVA model F, invariances;
- ardl: OLS of the conditional error-correction regression, restricted and
  unrestricted SSR for the bounds F, HC covariances under reparameterization;
- filters: dense normal equations (HP) and statsmodels (BK, CF);
- threshold: global brute-force search over every admissible split;
- ucm: a NumPy exact-diffuse Kalman filter and RTS smoother for the local level;
- mswitch: a NumPy sequential Hamilton filter and Kim smoother, maximized by SciPy.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import optimize, stats

import openecon as oe

warnings.filterwarnings("ignore", module="statsmodels")


def coef(fit):
    return np.array([c.estimate for c in fit.coefficients])


def ses(fit):
    return np.array([c.std_error for c in fit.coefficients])


def cov(fit):
    return np.array(fit.covariance_matrix)


# ---------------------------------------------------------------------------------------------
# prais
# ---------------------------------------------------------------------------------------------


def prais_data(seed=5, n=60, rho=0.7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n).cumsum() * 0.3 + rng.normal(size=n)
    x2 = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + rng.normal()
    y = 2.0 + 1.5 * x1 - 0.7 * x2 + 0.5 * (group == "b") + u
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": group, "t": np.arange(n) + 1})


def toeplitz_inverse(rho, n):
    index = np.arange(n)
    return np.linalg.inv(rho ** np.abs(index[:, None] - index[None, :]))


def rho_formula(u, rhotype, k):
    n = len(u)
    num = float(np.sum(u[1:] * u[:-1]))
    d = float(np.sum(np.diff(u) ** 2) / np.sum(u ** 2))
    return {"regress": num / np.sum(u[:-1] ** 2), "freg": num / np.sum(u[1:] ** 2),
            "tscorr": num / np.sum(u ** 2), "dw": 1 - d / 2,
            "theil": num / np.sum(u ** 2) * (n - k) / n,
            "nagar": ((1 - d / 2) * n * n + k * k) / (n * n - k * k)}[rhotype]


def gls_at(y, X, rho, prais):
    """GLS through the inverse AR(1) correlation matrix (Prais) or conditional on y_1 (corc)."""
    n = len(y)
    if prais:
        w = toeplitz_inverse(rho, n) * (1 - rho * rho)        # = P'P
    else:
        d = np.eye(n)[1:] - rho * np.eye(n)[:-1]               # quasi-differences, row 1 dropped
        w = d.T @ d
    xtwx = X.T @ w @ X
    beta = np.linalg.solve(xtwx, X.T @ w @ y)
    return beta, w, xtwx


@pytest.mark.parametrize("method", ["prais", "corc"])
@pytest.mark.parametrize("rhotype", ["regress", "freg", "tscorr", "dw", "theil", "nagar"])
def test_prais_fixed_point_and_gls_matrix(method, rhotype):
    df = prais_data()
    prais = method == "prais"
    y, X = df.y.to_numpy(), np.column_stack([np.ones(len(df)), df.x1, df.x2])
    k = X.shape[1]

    def gap(r):
        beta = gls_at(y, X, r, prais)[0]
        return rho_formula(y - X @ beta, rhotype, k) - r

    root = optimize.brentq(gap, -0.95, 0.99, xtol=1e-14)
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], time="t", method=method, rhotype=rhotype,
                   tolerance=1e-13, max_iterations=1000)
    assert_allclose(fit.metrics["rho"], root, rtol=1e-8)
    beta, w, xtwx = gls_at(y, X, fit.metrics["rho"], prais)
    assert_allclose(coef(fit), beta, rtol=1e-9)
    resid = y - X @ beta
    ssr = float(resid @ w @ resid)
    nstar = len(y) if prais else len(y) - 1
    s2 = ssr / (nstar - k)
    assert_allclose(cov(fit), s2 * np.linalg.inv(xtwx), rtol=1e-8)
    assert_allclose(fit.metrics["rmse"], np.sqrt(s2), rtol=1e-9)
    tstat = beta / np.sqrt(np.diag(s2 * np.linalg.inv(xtwx)))
    assert_allclose([c.p_value for c in fit.coefficients],
                    2 * stats.t.sf(np.abs(tstat), nstar - k), rtol=1e-7)
    crit = stats.t.ppf(0.975, nstar - k)
    assert_allclose([c.ci_low for c in fit.coefficients],
                    beta - crit * np.sqrt(np.diag(s2 * np.linalg.inv(xtwx))), rtol=1e-8)


def test_prais_reports_stata_anova_f_and_r2():
    """Stata's prais reports the ANOVA F of the transformed regression (regress, hascons).

    In Stata's own example (prais usr idle) F(1, 28) = 7.12 while t(idle)^2 = 8.25,
    so the reported F is (R2 / (K-1)) / ((1 - R2) / (N - K)) with R2 = 1 - SSR*/TSS*
    (TSS* centered at the mean of y*), not the Wald test of the slope.
    """
    df = prais_data()
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], time="t")
    rho = fit.metrics["rho"]
    y, X = df.y.to_numpy(), np.column_stack([np.ones(len(df)), df.x1, df.x2])
    ys = np.r_[np.sqrt(1 - rho ** 2) * y[0], y[1:] - rho * y[:-1]]
    xs = np.vstack([np.sqrt(1 - rho ** 2) * X[:1], X[1:] - rho * X[:-1]])
    e = ys - xs @ coef(fit)
    tss, ssr = np.sum((ys - ys.mean()) ** 2), e @ e
    n, k = len(ys), 3
    r2 = 1 - ssr / tss
    assert_allclose(fit.metrics["r_squared"], r2, rtol=1e-10)
    assert_allclose(fit.metrics["adjusted_r_squared"], 1 - (1 - r2) * (n - 1) / (n - k), rtol=1e-10)
    f = (r2 / (k - 1)) / ((1 - r2) / (n - k))
    model = fit.tests["model"]
    assert_allclose(model["statistic"], f, rtol=1e-9)
    assert (model["df"], model["df2"]) == (2, n - k)
    assert_allclose(model["p_value"], stats.f.sf(f, 2, n - k), rtol=1e-8)
    # With a robust covariance the model test is the Wald F of the slopes.
    robust = oe.prais(data=df, y="y", x=["x1", "x2"], time="t", covariance="robust")
    V = cov(robust)[1:, 1:]
    b = coef(robust)[1:]
    assert_allclose(robust.tests["model"]["statistic"], b @ np.linalg.solve(V, b) / 2, rtol=1e-9)
    # Cochrane-Orcutt: the transformed constant is a constant, so ANOVA F = Wald F.
    corc = oe.prais(data=df, y="y", x=["x1", "x2"], time="t", method="corc")
    V, b = cov(corc)[1:, 1:], coef(corc)[1:]
    assert_allclose(corc.tests["model"]["statistic"], b @ np.linalg.solve(V, b) / 2, rtol=1e-8)


@pytest.mark.parametrize("kind", ["HC1", "HC2", "HC3"])
def test_prais_white_covariances_on_transformed_data(kind):
    df = prais_data(seed=8)
    fit = oe.prais(data=df, y="y", x=["x1", "x2"], time="t", covariance=kind)
    rho = fit.metrics["rho"]
    y, X = df.y.to_numpy(), np.column_stack([np.ones(len(df)), df.x1, df.x2])
    ys = np.r_[np.sqrt(1 - rho ** 2) * y[0], y[1:] - rho * y[:-1]]
    xs = np.vstack([np.sqrt(1 - rho ** 2) * X[:1], X[1:] - rho * X[:-1]])
    bread = np.linalg.inv(xs.T @ xs)
    e = ys - xs @ (bread @ xs.T @ ys)
    h = np.einsum("ij,jk,ik->i", xs, bread, xs)
    n, k = xs.shape
    scale = {"HC1": e ** 2 * n / (n - k), "HC2": e ** 2 / (1 - h), "HC3": e ** 2 / (1 - h) ** 2}
    V = bread @ (xs.T * scale[kind]) @ xs @ bread
    assert_allclose(cov(fit), V, rtol=1e-8)


def test_prais_invariances_categorical_and_collinear():
    df = prais_data(seed=9)
    base = oe.prais(data=df, y="y", x=["x1", "x2", "g"], categorical=["g"], time="t")
    shuffled = df.sample(frac=1.0, random_state=3)
    again = oe.prais(data=shuffled, y="y", x=["x1", "x2", "g"], categorical=["g"], time="t")
    assert_allclose(coef(again), coef(base), rtol=1e-12)
    assert_allclose(cov(again), cov(base), rtol=1e-11)
    assert [c.term for c in base.coefficients] == ["Intercept", "x1", "x2", "g[b]", "g[c]"]
    # NumPy with explicit dummies at the reported rho.
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2, df.g == "b", df.g == "c"]).astype(float)
    beta = gls_at(df.y.to_numpy(), X, base.metrics["rho"], True)[0]
    assert_allclose(coef(base), beta, rtol=1e-9)
    # Rescaling a regressor rescales its coefficient only.
    scaled = df.assign(x1=df.x1 * 1e6)
    big = oe.prais(data=scaled, y="y", x=["x1", "x2", "g"], categorical=["g"], time="t")
    assert_allclose(coef(big) * [1, 1e6, 1, 1, 1], coef(base), rtol=1e-8)
    assert_allclose([c.statistic for c in big.coefficients],
                    [c.statistic for c in base.coefficients], rtol=1e-8)
    # An exactly collinear regressor is omitted Stata-style with a warning.
    collinear = oe.prais(data=df.assign(x3=2 * df.x1 - df.x2), y="y", x=["x1", "x2", "x3"],
                         time="t")
    plain = oe.prais(data=df, y="y", x=["x1", "x2"], time="t")
    assert [c.term for c in collinear.coefficients] == ["Intercept", "x1", "x2"]
    assert any("x3" in w for w in collinear.warnings)
    assert_allclose(coef(collinear), coef(plain), rtol=1e-10)


def test_prais_missing_at_the_edges_equals_trimmed_sample():
    df = prais_data(seed=10)
    holed = df.copy()
    holed.loc[[0, 1], "x2"] = np.nan
    holed.loc[len(df) - 1, "y"] = np.nan
    dropped = oe.prais(data=holed, y="y", x=["x1", "x2"], time="t", missing="drop")
    trimmed = oe.prais(data=df.iloc[2:len(df) - 1], y="y", x=["x1", "x2"], time="t")
    assert_allclose(coef(dropped), coef(trimmed), rtol=1e-12)
    assert dropped.nobs == len(df) - 3
    assert dropped.sample_positions == list(range(2, len(df) - 1))


# ---------------------------------------------------------------------------------------------
# ardl
# ---------------------------------------------------------------------------------------------


def ardl_data(seed=21, n=160):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).cumsum(0) * 0.5
    w = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (0.5 + 0.6 * y[t - 1] - 0.15 * y[t - 2] + 0.4 * x[t, 0] - 0.1 * x[t - 1, 0]
                + 0.2 * x[t, 1] + 0.3 * w[t] + 0.01 * t + rng.normal() * 0.7)
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "w": w, "t": np.arange(n) + 1})


def lagged(a, i, rows):
    return a[rows - i]


def ec_regression(df, p, q, trend, start, exog=False):
    """OLS of D.y on (L.y, x_t levels, LD.y.., D.x.., deterministic) on rows >= start."""
    y, X = df.y.to_numpy(), df[["x1", "x2"]].to_numpy()
    rows = np.arange(start, len(y))
    cols, names = [], []
    if trend != "none":
        cols.append(np.ones(len(rows)))
        names.append("Intercept")
    if trend == "trend":
        cols.append(rows + 1.0)
        names.append("trend")
    cols.append(lagged(y, 1, rows))
    names.append("ADJ")
    for j in range(2):
        cols.append(X[rows, j])
        names.append(f"lvl{j}")
    for i in range(1, p):
        cols.append(lagged(y, i, rows) - lagged(y, i + 1, rows))
        names.append(f"dy{i}")
    for j in range(2):
        for lag_ in range(q[j]):
            cols.append(lagged(X[:, j], lag_, rows) - lagged(X[:, j], lag_ + 1, rows))
            names.append(f"dx{j}_{lag_}")
    if exog:
        cols.append(df.w.to_numpy()[rows])
        names.append("w")
    Z = np.column_stack(cols)
    target = y[rows] - y[rows - 1]
    return Z, target, names


def ols(Z, target, kind="nonrobust"):
    bread = np.linalg.inv(Z.T @ Z)
    beta = bread @ Z.T @ target
    e = target - Z @ beta
    n, k = Z.shape
    h = np.einsum("ij,jk,ik->i", Z, bread, Z)
    if kind == "nonrobust":
        V = bread * (e @ e) / (n - k)
    else:
        scale = {"HC1": e ** 2 * n / (n - k), "HC2": e ** 2 / (1 - h),
                 "HC3": e ** 2 / (1 - h) ** 2}[kind]
        V = bread @ (Z.T * scale) @ Z @ bread
    return beta, V, e


@pytest.mark.parametrize("kind", ["nonrobust", "HC1", "HC3"])
@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_ardl_ec_form_is_a_reparameterized_ols(kind, trend):
    df = ardl_data()
    p, q = 2, [2, 1]
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[p, *q], trend=trend, ec=True,
                  covariance=kind, exog=["w"])
    Z, target, names = ec_regression(df, p, q, trend, start=2, exog=True)
    beta, V, e = ols(Z, target, kind)
    est = {c.term: (c.estimate, c.std_error) for c in fit.coefficients}
    pairs = {"ADJ:L.y": "ADJ", "SR:LD.y": "dy1", "SR:D.x1": "dx0_0", "SR:LD.x1": "dx0_1",
             "SR:D.x2": "dx1_0", "SR:w": "w", "SR:Intercept": "Intercept"}
    if trend == "trend":
        pairs["SR:trend"] = "trend"
    assert set(est) == set(pairs) | {"LR:x1", "LR:x2"}
    for ours, theirs in pairs.items():
        i = names.index(theirs)
        assert_allclose(est[ours], (beta[i], np.sqrt(V[i, i])), rtol=1e-8, atol=1e-12)
    a = names.index("ADJ")
    for j, term in enumerate(["LR:x1", "LR:x2"]):
        i = names.index(f"lvl{j}")
        value = -beta[i] / beta[a]
        grad = np.zeros(len(beta))
        grad[i], grad[a] = -1 / beta[a], beta[i] / beta[a] ** 2
        assert_allclose(est[term], (value, np.sqrt(grad @ V @ grad)), rtol=1e-8)
    tss = np.sum((target - target.mean()) ** 2)
    r2 = 1 - (e @ e) / tss
    assert_allclose(fit.metrics["r_squared"], r2, rtol=1e-10)
    # The model test of the EC form tests every reported coefficient except the constant.
    slopes = [i for i, name in enumerate(names) if name != "Intercept"]
    stat = beta[slopes] @ np.linalg.solve(V[np.ix_(slopes, slopes)], beta[slopes]) / len(slopes)
    assert_allclose(fit.tests["model"]["statistic"], stat, rtol=1e-8)
    assert fit.tests["model"]["df"] == len(slopes)
    if kind == "nonrobust":
        n, k = Z.shape
        assert_allclose(stat, (r2 / (k - 1)) / ((1 - r2) / (n - k)), rtol=1e-8)


@pytest.mark.parametrize("trend,restricted,case", [("constant", False, 3), ("constant", True, 2),
                                                   ("trend", False, 5), ("trend", True, 4),
                                                   ("none", False, 1)])
def test_ardl_bounds_f_and_t_from_restricted_regressions(trend, restricted, case):
    df = ardl_data(seed=4)
    p, q = 2, [1, 1]
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[p, *q], trend=trend,
                  restricted=restricted)
    # Conditional EC regression with lagged levels (PSS 2001, eq. 23): D.y on L.y, L.x, ...
    y, X = df.y.to_numpy(), df[["x1", "x2"]].to_numpy()
    rows = np.arange(2, len(y))
    levels = [y[rows - 1], X[rows - 1, 0], X[rows - 1, 1]]
    deterministic = []
    if trend != "none":
        deterministic.append(np.ones(len(rows)))
    if trend == "trend":
        deterministic.append(rows + 1.0)
    short = [y[rows - 1] - y[rows - 2]]
    for j in range(2):
        short.append(X[rows, j] - X[rows - 1, j])
    target = y[rows] - y[rows - 1]
    unrestricted = np.column_stack(deterministic + levels + short)
    restricted_det = deterministic[:]
    if case == 2:
        restricted_det = []                       # constant restricted to the long run
    if case == 4:
        restricted_det = deterministic[:1]        # trend restricted
    restricted_z = np.column_stack(restricted_det + short)
    ssr_u = np.sum((target - unrestricted @ np.linalg.lstsq(unrestricted, target, rcond=None)[0]) ** 2)
    ssr_r = np.sum((target - restricted_z @ np.linalg.lstsq(restricted_z, target, rcond=None)[0]) ** 2)
    n, k = unrestricted.shape
    qn = k - restricted_z.shape[1]
    f = ((ssr_r - ssr_u) / qn) / (ssr_u / (n - k))
    bounds = fit.tests["bounds_f"]
    assert_allclose(bounds["statistic"], f, rtol=1e-8)
    assert bounds["df"] == qn and bounds["df2"] == n - k and bounds["case"] == case
    if case in (1, 3, 5):
        beta, V, _ = ols(unrestricted, target)
        i = len(deterministic)
        assert_allclose(fit.tests["bounds_t"]["statistic"], beta[i] / np.sqrt(V[i, i]), rtol=1e-8)
    else:
        assert "bounds_t" not in fit.tests


def test_ardl_distributed_lag_and_datetime_time():
    df = ardl_data(seed=6)
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], lags=[0, 2, 1])
    y, X = df.y.to_numpy(), df[["x1", "x2"]].to_numpy()
    rows = np.arange(2, len(y))
    Z = np.column_stack([np.ones(len(rows)), X[rows, 0], X[rows - 1, 0], X[rows - 2, 0],
                         X[rows, 1], X[rows - 1, 1]])
    beta, V, e = ols(Z, y[rows])
    assert_allclose(coef(fit), beta, rtol=1e-9)
    assert_allclose(cov(fit), V, rtol=1e-8)
    n = len(rows)
    ll = -n / 2 * (1 + np.log(2 * np.pi) + np.log(e @ e / n))
    assert_allclose(fit.metrics["log_likelihood"], ll, rtol=1e-12)
    assert_allclose(fit.metrics["bic"], -2 * ll + np.log(n) * 6, rtol=1e-12)
    dated = df.assign(t=pd.date_range("1990-01-01", periods=len(df), freq="QS")).iloc[::-1]
    again = oe.ardl(data=dated, y="y", x=["x1", "x2"], lags=[0, 2, 1], time="t")
    assert_allclose(coef(again), beta, rtol=1e-9)


def test_ardl_lag_search_with_trend_and_exog_matches_brute_force():
    df = ardl_data(seed=7, n=120)
    maxlags = [3, 2, 2]
    fit = oe.ardl(data=df, y="y", x=["x1", "x2"], maxlags=maxlags, ic="bic", trend="trend",
                  exog=["w"])
    y, X, w = df.y.to_numpy(), df[["x1", "x2"]].to_numpy(), df.w.to_numpy()
    rows = np.arange(3, len(y))
    scores = {}
    for combo in itertools.product(range(1, 4), range(3), range(3)):
        cols = [np.ones(len(rows)), rows + 1.0, w[rows]]
        cols += [y[rows - i] for i in range(1, combo[0] + 1)]
        for j in range(2):
            cols += [X[rows - lag_, j] for lag_ in range(combo[j + 1] + 1)]
        Z = np.column_stack(cols)
        e = y[rows] - Z @ np.linalg.lstsq(Z, y[rows], rcond=None)[0]
        n = len(rows)
        ll = -n / 2 * (1 + np.log(2 * np.pi) + np.log(e @ e / n))
        scores[combo] = -2 * ll + np.log(n) * Z.shape[1]
    best = min(scores, key=scores.get)
    assert fit.extra["lag_selection"]["selected"] == list(best)
    assert_allclose(fit.metrics["bic"], scores[best], rtol=1e-10)


# ---------------------------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------------------------


def filter_data(seed=31, n=150):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 0.02 * t + np.cumsum(rng.normal(size=n)) * 0.3 + np.sin(2 * np.pi * t / 20)
    return pd.DataFrame({"y": y, "t": t + 1})


@pytest.mark.parametrize("smooth", [6.25, 1600.0, 1e5, 1e8])
def test_hp_filter_solves_dense_normal_equations(smooth):
    """The exact solution of (I + lambda K'K) tau = y in 40-digit arithmetic (mpmath).

    The system's condition number is about 1 + 16 lambda, so a float64 solver is
    accurate to roughly eps * sqrt(cond) here (LAPACK's banded Cholesky is no better).
    """
    import mpmath

    df = filter_data(n=110)
    n = len(df)
    second = np.zeros((n - 2, n))
    for i in range(n - 2):
        second[i, i:i + 3] = [1.0, -2.0, 1.0]
    mpmath.mp.dps = 40
    system = mpmath.matrix((smooth * second.T @ second + np.eye(n)).tolist())
    trend = np.array(mpmath.lu_solve(system, mpmath.matrix(df.y.tolist())).tolist(),
                     dtype=float).ravel()
    out = oe.tsfilter(df.sample(frac=1.0, random_state=1), "y", method="hp", time="t",
                      smooth=smooth)
    assert list(out["period"]) == list(df.t)
    tolerance = 1e-13 * np.sqrt(1 + 16 * smooth) * np.abs(df.y).max()
    assert_allclose(out["trend"].to_numpy(), trend, rtol=0, atol=tolerance)
    assert_allclose(out["cycle"].to_numpy(), df.y.to_numpy() - trend, rtol=0, atol=tolerance)


@pytest.mark.parametrize("low,high,order", [(6, 32, 12), (2, 8, 3), (10, 40, 20)])
def test_bk_and_cf_filters_match_statsmodels(low, high, order):
    from statsmodels.tsa.filters.bk_filter import bkfilter
    from statsmodels.tsa.filters.cf_filter import cffilter

    df = filter_data(seed=32)
    y = df.y.to_numpy()
    bk = oe.tsfilter(df, "y", method="bk", minperiod=low, maxperiod=high, smaorder=order)
    cycle = bk["cycle"].to_numpy()
    assert np.isnan(cycle[:order]).all() and np.isnan(cycle[len(y) - order:]).all()
    assert_allclose(cycle[order:len(y) - order], bkfilter(y, low, high, order), rtol=1e-10,
                    atol=1e-12)
    for drift in (True, False):
        cf = oe.tsfilter(df, "y", method="cf", minperiod=low, maxperiod=high, drift=drift)
        reference, _ = cffilter(y, low, high, drift)
        assert_allclose(cf["cycle"].to_numpy(), reference, rtol=1e-9, atol=1e-10)
        assert_allclose(cf["trend"].to_numpy(), y - reference, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("h,p", [(8, 4), (2, 1), (24, 12)])
def test_hamilton_filter_is_ols_of_the_h_step_ahead_value(h, p):
    df = filter_data(seed=33, n=200)
    y = df.y.to_numpy()
    out = oe.tsfilter(df, "y", method="hamilton", hamilton_h=h, hamilton_p=p)
    rows = np.arange(h + p - 1, len(y))
    Z = np.column_stack([np.ones(len(rows))] + [y[rows - h - j] for j in range(p)])
    beta = np.linalg.lstsq(Z, y[rows], rcond=None)[0]
    trend = out["trend"].to_numpy()
    assert np.isnan(trend[:h + p - 1]).all()
    assert_allclose(trend[h + p - 1:], Z @ beta, rtol=1e-10)
    assert_allclose(out["cycle"].to_numpy()[h + p - 1:], y[rows] - Z @ beta, atol=1e-9)
    coefficients = out.attrs["coefficients"]
    assert_allclose([coefficients["Intercept"]] + [coefficients[f"L{j}"] for j in range(p)], beta,
                    rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------------------------


def threshold_data(seed=41, n=140, two=False):
    rng = np.random.default_rng(seed)
    q = np.round(rng.normal(size=n), 2)                   # ties in the threshold variable
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    region = (q > 0.2).astype(int) + ((q > 1.0).astype(int) if two else 0)
    a = np.array([1.0, -1.0, 2.0])[region]
    b = np.array([0.5, 2.0, -1.0])[region]
    y = a + b * x1 + 0.3 * x2 + rng.normal(size=n) * 0.5
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "q": q})


def region_design(df, gammas, varying, common):
    region = np.zeros(len(df), int)
    for g in gammas:
        region += (df.q.to_numpy() > g)
    base = {"Intercept": np.ones(len(df)), "x1": df.x1.to_numpy(), "x2": df.x2.to_numpy()}
    cols = [base[name] for name in common]
    for r in range(len(gammas) + 1):
        cols += [base[name] * (region == r) for name in varying]
    return np.column_stack(cols), region


def ssr_of(Z, y):
    return float(np.sum((y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]) ** 2))


@pytest.mark.parametrize("varying,common", [(["Intercept", "x1", "x2"], []),
                                            (["Intercept", "x1"], ["x2"])])
def test_threshold_single_global_brute_force(varying, common):
    df = threshold_data()
    n, trim = len(df), 0.15
    minimum = int(np.ceil(trim * n))
    fit = oe.threshold(data=df, y="y", x=["x1", "x2"], threshold_var="q", trim=trim,
                       regions=None if not common else varying)
    best = None
    for g in np.unique(df.q):
        sizes = [(df.q <= g).sum(), (df.q > g).sum()]
        if min(sizes) < minimum:
            continue
        Z, _ = region_design(df, [g], varying, common)
        value = ssr_of(Z, df.y.to_numpy())
        if best is None or value < best[0]:
            best = (value, g)
    assert_allclose(fit.extra["thresholds"], [best[1]])
    Z, region = region_design(df, [best[1]], varying, common)
    for kind in ("nonrobust", "HC1", "HC3"):
        res = oe.threshold(data=df, y="y", x=["x1", "x2"], threshold_var="q", trim=trim,
                           regions=None if not common else varying, covariance=kind)
        beta, V, e = ols(Z, df.y.to_numpy(), kind)
        assert_allclose(coef(res), beta, rtol=1e-9)
        assert_allclose(cov(res), V, rtol=1e-8, atol=1e-14 * np.abs(V).max())
    assert_allclose(fit.metrics["ssr"], best[0], rtol=1e-10)
    assert fit.extra["region_sizes"] == [int((region == r).sum()) for r in range(2)]
    k = Z.shape[1]
    assert fit.inference["df_inference"] == n - k
    assert_allclose(fit.metrics["bic"], n * np.log(best[0] / n) + k * np.log(n), rtol=1e-12)


def test_threshold_two_sequential_equals_global_pair_search():
    df = threshold_data(seed=42, n=150, two=True)
    n, trim = len(df), 0.1
    minimum = int(np.ceil(trim * n))
    fit = oe.threshold(data=df, y="y", x=["x1", "x2"], threshold_var="q", trim=trim,
                       nthresholds=2)
    values = np.unique(df.q)
    best = None
    y = df.y.to_numpy()
    for g1, g2 in itertools.combinations(values, 2):
        sizes = [(df.q <= g1).sum(), ((df.q > g1) & (df.q <= g2)).sum(), (df.q > g2).sum()]
        if min(sizes) < minimum:
            continue
        value = ssr_of(region_design(df, [g1, g2], ["Intercept", "x1", "x2"], [])[0], y)
        if best is None or value < best[0]:
            best = (value, [g1, g2])
    assert_allclose(fit.extra["thresholds"], best[1])
    assert_allclose(fit.metrics["ssr"], best[0], rtol=1e-10)


# ---------------------------------------------------------------------------------------------
# ucm: local level with and without a regressor
# ---------------------------------------------------------------------------------------------


def ucm_data(seed=51, n=120):
    rng = np.random.default_rng(seed)
    level = 5 + np.cumsum(rng.normal(size=n) * 0.5)
    x = rng.normal(size=n)
    y = level + 0.8 * x + rng.normal(size=n) * 0.9
    return pd.DataFrame({"y": y, "x": x, "t": np.arange(n) + 1})


def local_level_filter(y, q, h):
    """Exact diffuse filter of y_t = mu_t + e_t, mu_(t+1) = mu_t + eta_t (NumPy, sequential).

    The first observation is diffuse: it contributes -1/2 log 2 pi only (F_inf = 1) and
    gives a_2 = y_1, P_2 = h + q. Returns (loglik, a, P, v, F) with a, P predicted states.
    """
    n = len(y)
    a, p = np.zeros(n + 1), np.zeros(n + 1)
    v, f = np.zeros(n), np.zeros(n)
    a[1], p[1] = y[0], h + q
    ll = -0.5 * np.log(2 * np.pi)
    for t in range(1, n):
        v[t] = y[t] - a[t]
        f[t] = p[t] + h
        k = p[t] / f[t]
        a[t + 1] = a[t] + k * v[t]
        p[t + 1] = p[t] * (1 - k) + q
        ll += -0.5 * (np.log(2 * np.pi) + np.log(f[t]) + v[t] ** 2 / f[t])
    return ll, a, p, v, f


def dense_smoothed_level(y, q, h):
    """E[mu_t | y] with a flat prior on mu_1, by dense GLS (independent of any recursion)."""
    n = len(y)
    index = np.arange(n)
    walk = q * np.minimum(index[:, None], index[None, :])     # Cov(mu_t - mu_1, mu_s - mu_1)
    s = walk + h * np.eye(n)
    ones = np.ones(n)
    solved = np.linalg.solve(s, np.column_stack([ones, y]))
    mu1 = (ones @ solved[:, 1]) / (ones @ solved[:, 0])
    return mu1 + walk @ np.linalg.solve(s, y - mu1)


def numerical_hessian(fun, x, step=1e-4):
    x = np.asarray(x, float)
    k = len(x)
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * step * max(1, abs(x[i])), np.eye(k)[j] * step * max(1, abs(x[j]))
            out[i, j] = (fun(x + ei + ej) - fun(x + ei - ej) - fun(x - ei + ej)
                         + fun(x - ei - ej)) / (4 * np.linalg.norm(ei) * np.linalg.norm(ej))
    return out


def test_ucm_local_level_against_numpy_filter_and_dense_smoother():
    df = ucm_data()
    y = df.y.to_numpy()
    fit = oe.ucm(data=df, y="y", model="llevel", time="t")
    est = {c.term: c.estimate for c in fit.coefficients}
    q, h = est["/var(level)"], est["/var(e)"]
    ll = local_level_filter(y, q, h)[0]
    assert_allclose(fit.metrics["log_likelihood"], ll, rtol=1e-10)

    def negative(theta):
        return -local_level_filter(y, np.exp(theta[0]), np.exp(theta[1]))[0]

    best = optimize.minimize(negative, np.log([q * 1.5, h * 0.7]), method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    assert -best.fun <= ll + 1e-7                              # ours is the maximum
    assert_allclose(np.exp(best.x), [q, h], rtol=1e-4)
    # Observed-information covariance of (var(level), var(e)).
    hessian = numerical_hessian(lambda v: local_level_filter(y, v[0], v[1])[0], [q, h], 1e-5)
    assert_allclose(cov(fit), np.linalg.inv(-hessian), rtol=2e-4)
    assert_allclose(fit.metrics["aic"], -2 * ll + 4, rtol=1e-12)
    assert_allclose(fit.metrics["bic"], -2 * ll + 2 * np.log(len(y)), rtol=1e-12)
    parts = oe.ucm_components(fit, df)
    assert_allclose(parts["level"].to_numpy(), dense_smoothed_level(y, q, h), rtol=1e-8)
    # Forecasts: a_(n+1) with MSE P_(n+1) + (j - 1) q + h.
    _, a, p, _, _ = local_level_filter(y, q, h)
    forecast = oe.forecast(fit, steps=5)
    assert_allclose(forecast["forecast"].to_numpy(), np.full(5, a[-1]), rtol=1e-9)
    assert_allclose(forecast["std_error"].to_numpy(), np.sqrt(p[-1] + np.arange(5) * q + h),
                    rtol=1e-9)
    assert list(forecast["period"]) == list(range(len(y) + 1, len(y) + 6))
    # Variances: one-sided p-values and intervals truncated at zero (Stata's ucm).
    for c in fit.coefficients:
        assert_allclose(c.p_value, stats.norm.sf(c.estimate / c.std_error), rtol=1e-8)
        assert c.ci_low >= 0.0


def test_ucm_local_level_with_regressor_by_brute_force_ml():
    df = ucm_data(seed=52)
    y, x = df.y.to_numpy(), df.x.to_numpy()
    fit = oe.ucm(data=df, y="y", x=["x"], model="llevel")
    est = {c.term: c.estimate for c in fit.coefficients}

    def loglik(theta):
        return local_level_filter(y - theta[2] * x, theta[0], theta[1])[0]

    point = [est["/var(level)"], est["/var(e)"], est["x"]]
    assert_allclose(fit.metrics["log_likelihood"], loglik(point), rtol=1e-10)
    best = optimize.minimize(lambda th: -loglik([np.exp(th[0]), np.exp(th[1]), th[2]]),
                             [np.log(point[0]) + 0.3, np.log(point[1]) - 0.2, point[2] + 0.1],
                             method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 6000})
    assert -best.fun <= loglik(point) + 1e-7
    assert_allclose([np.exp(best.x[0]), np.exp(best.x[1]), best.x[2]], point, rtol=2e-4)
    hessian = numerical_hessian(lambda v: loglik([v[1], v[2], v[0]]),
                                [point[2], point[0], point[1]], 1e-5)
    assert [c.term for c in fit.coefficients] == ["x", "/var(level)", "/var(e)"]
    assert_allclose(cov(fit), np.linalg.inv(-hessian), rtol=5e-4)


# ---------------------------------------------------------------------------------------------
# mswitch
# ---------------------------------------------------------------------------------------------


def ms_data(seed=61, n=300, k=2):
    rng = np.random.default_rng(seed)
    p = np.array([[0.95, 0.05], [0.10, 0.90]]) if k == 2 else \
        np.array([[0.9, 0.05, 0.05], [0.1, 0.85, 0.05], [0.05, 0.1, 0.85]])
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = rng.choice(k, p=p[s[t - 1]])
    x = rng.normal(size=n)
    mu = np.array([-1.0, 1.5, 4.0])[:k]
    sigma = np.array([0.6, 1.2, 0.8])[:k]
    y = mu[s] + 0.7 * x + sigma[s] * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "t": np.arange(n) + 1})


def ms_unpack(fit, k, varswitch):
    est = {c.term: c.estimate for c in fit.coefficients}
    mu = np.array([est[f"state{s + 1}:Intercept"] for s in range(k)])
    sigma = np.exp([est[f"/lnsigma{s + 1}"] for s in range(k)]) if varswitch else \
        np.full(k, np.exp(est["/lnsigma"]))
    logits = np.array([[est[f"/lgt(p{a + 1}{c + 1})"] for c in range(k - 1)] for a in range(k)])
    full = np.column_stack([logits, np.zeros(k)])
    p = np.exp(full) / np.exp(full).sum(1, keepdims=True)
    return mu, est.get("x", 0.0), sigma, p


def hamilton_numpy(y, x, mu, beta, sigma, p):
    """Sequential Hamilton filter and Kim smoother started from the ergodic distribution."""
    n, k = len(y), len(mu)
    eigvals, eigvecs = np.linalg.eig(p.T)
    pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    pi = pi / pi.sum()
    dens = stats.norm.pdf(y[:, None], mu[None, :] + beta * x[:, None], sigma[None, :])
    predicted, filtered = np.zeros((n, k)), np.zeros((n, k))
    ll, current = 0.0, pi
    for t in range(n):
        predicted[t] = current @ p if t else pi
        joint = predicted[t] * dens[t]
        ll += np.log(joint.sum())
        filtered[t] = joint / joint.sum()
        current = filtered[t]
    smoothed = np.zeros((n, k))
    smoothed[-1] = filtered[-1]
    for t in range(n - 2, -1, -1):
        smoothed[t] = filtered[t] * (p @ (smoothed[t + 1] / predicted[t + 1]))
    return ll, filtered, smoothed, predicted


@pytest.mark.parametrize("varswitch", [False, True])
def test_mswitch_dynamic_regression_against_numpy_hamilton_filter(varswitch):
    df = ms_data()
    y, x = df.y.to_numpy(), df.x.to_numpy()
    fit = oe.mswitch(data=df, y="y", x=["x"], time="t", varswitch=varswitch)
    mu, beta, sigma, p = ms_unpack(fit, 2, varswitch)
    ll, filtered, smoothed, predicted = hamilton_numpy(y, x, mu, beta, sigma, p)
    assert_allclose(fit.metrics["log_likelihood"], ll, rtol=1e-10)
    assert mu[0] < mu[1]                                      # states ordered by the constant
    probs = oe.mswitch_probabilities(fit, df)
    for s in range(2):
        assert_allclose(probs[f"filtered_state{s + 1}"].to_numpy(), filtered[:, s], atol=1e-10)
        assert_allclose(probs[f"smoothed_state{s + 1}"].to_numpy(), smoothed[:, s], atol=1e-10)
        assert_allclose(probs[f"predicted_state{s + 1}"].to_numpy(), predicted[:, s], atol=1e-10)
    theta0 = coef(fit)
    names = [c.term for c in fit.coefficients]

    def loglik(theta):
        est = dict(zip(names, theta))
        m = np.array([est["state1:Intercept"], est["state2:Intercept"]])
        sg = np.exp([est["/lnsigma1"], est["/lnsigma2"]]) if varswitch else \
            np.full(2, np.exp(est["/lnsigma"]))
        full = np.array([[est["/lgt(p11)"], 0.0], [est["/lgt(p21)"], 0.0]])
        pm = np.exp(full) / np.exp(full).sum(1, keepdims=True)
        return hamilton_numpy(y, x, m, est["x"], sg, pm)[0]

    best = optimize.minimize(lambda th: -loglik(th), theta0 + 0.05, method="BFGS",
                             options={"gtol": 1e-8})
    assert -best.fun <= ll + 1e-6
    assert_allclose(best.x, theta0, rtol=1e-3, atol=1e-4)
    hessian = numerical_hessian(loglik, theta0, 1e-4)
    assert_allclose(cov(fit), np.linalg.inv(-hessian), rtol=2e-3, atol=1e-7)
    k = len(theta0)
    assert_allclose(fit.metrics["aic"], -2 * ll + 2 * k, rtol=1e-12)
    assert_allclose(fit.metrics["bic"], -2 * ll + np.log(len(y)) * k, rtol=1e-12)
    assert_allclose(fit.metrics["duration_state1"], 1 / (1 - p[0, 0]), rtol=1e-10)
    # Transition probabilities with delta-method standard errors (extra).
    V = cov(fit)
    i = names.index("/lgt(p11)")
    se = p[0, 0] * (1 - p[0, 0]) * np.sqrt(V[i, i])
    cell = fit.extra["transition_matrix"][0][0]
    assert_allclose([cell["probability"], cell["std_error"]], [p[0, 0], se], rtol=1e-8)


def test_mswitch_three_states_and_ar1_likelihoods():
    df = ms_data(seed=62, n=400, k=3)
    y, x = df.y.to_numpy(), df.x.to_numpy()
    fit = oe.mswitch(data=df, y="y", x=["x"], states=3, varswitch=True)
    mu, beta, sigma, p = ms_unpack(fit, 3, True)
    assert_allclose(fit.metrics["log_likelihood"], hamilton_numpy(y, x, mu, beta, sigma, p)[0],
                    rtol=1e-10)
    # Hamilton (1989) AR(1): y_t - mu(s_t) = phi (y_(t-1) - mu(s_(t-1))) + e_t, conditional on y_1,
    # with (s_1, s_0) drawn from the ergodic joint distribution pi_(s_0) p_(s_0 s_1).
    ar = oe.mswitch(data=df, y="y", ar=1)
    est = {c.term: c.estimate for c in ar.coefficients}
    m = np.array([est["state1:Intercept"], est["state2:Intercept"]])
    phi, sg = est["L1.ar"], np.exp(est["/lnsigma"])
    full = np.array([[est["/lgt(p11)"], 0.0], [est["/lgt(p21)"], 0.0]])
    pm = np.exp(full) / np.exp(full).sum(1, keepdims=True)
    w, vecs = np.linalg.eig(pm.T)
    pi = np.real(vecs[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    joint = pi[:, None] * pm                                   # [s_prev, s_now]
    ll = 0.0
    for t in range(1, len(y)):
        dens = stats.norm.pdf(y[t] - m[None, :] - phi * (y[t - 1] - m[:, None]), 0, sg)
        if t > 1:
            joint = joint.sum(0)[:, None] * pm
        mass = joint * dens
        ll += np.log(mass.sum())
        joint = mass / mass.sum()
    assert_allclose(ar.metrics["log_likelihood"], ll, rtol=1e-10)
    assert ar.nobs == len(y) - 1


def test_threshold_bootstrap_with_two_thresholds_tests_the_best_single_threshold():
    """Regression: with nthresholds=2 the bootstrap statistic used the lower refined threshold."""
    df = threshold_data(seed=43, n=120, two=True)
    n, trim = len(df), 0.1
    minimum = int(np.ceil(trim * n))
    y = df.y.to_numpy()
    fit = oe.threshold(data=df, y="y", x=["x1", "x2"], threshold_var="q", trim=trim,
                       nthresholds=2, bootstrap=5, seed=1)
    single = min(ssr_of(region_design(df, [g], ["Intercept", "x1", "x2"], [])[0], y)
                 for g in np.unique(df.q)
                 if min((df.q <= g).sum(), (df.q > g).sum()) >= minimum)
    linear = ssr_of(np.column_stack([np.ones(n), df.x1, df.x2]), y)
    assert_allclose(fit.tests["threshold_effect"]["statistic"], n * (linear - single) / single,
                    rtol=1e-9)
