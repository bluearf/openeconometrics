"""Independent numerical oracles for the panel family (xtreg, hausman, xtfmb).

Every expected value here is derived from the textbook / Stata
Methods-and-formulas definition and computed directly in NumPy (dummy-variable
least squares, explicit within / GLS transforms, explicit sandwich sums, a
brute-force SciPy maximization of a likelihood written with dense per-panel
covariance matrices) or taken from statsmodels where the identical convention
exists. Nothing is shared with the implementation or with its own test file.
"""

import time

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import openecon as oe
from openecon.analysis_contracts import AnalysisError

COLS = ["x1", "x2", "z"]          # z is time invariant
VARY = ["x1", "x2"]


# ---- synthetic designs ----------------------------------------------------------------


def make_data(seed=11, n=24, periods=7, unbalanced=True, gaps=False, hetero=True):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), periods)
    year = np.tile(np.arange(2000, 2000 + periods), n)
    u, z = rng.normal(size=n), rng.normal(size=n)
    x1 = rng.normal(size=ids.size) + 0.3 * u[ids]
    x2 = rng.normal(size=ids.size) + 0.3 * z[ids]
    cat = rng.choice(["a", "b", "c"], size=ids.size)
    scale = np.exp(0.6 * rng.normal(size=n))[ids] if hetero else 1.0
    e = rng.normal(size=ids.size) * scale
    y = (1.0 + 0.5 * x1 - 0.3 * x2 + 0.2 * z[ids] + 0.4 * (cat == "b") - 0.5 * (cat == "c")
         + u[ids] + e)
    frame = pd.DataFrame({"id": ids, "year": year, "y": y, "x1": x1, "x2": x2, "z": z[ids],
                          "cat": cat})
    frame["wa"] = rng.uniform(0.5, 3.0, size=n)[ids]           # aweight, constant in panel
    frame["wf"] = rng.integers(1, 4, size=n)[ids].astype(float)  # fweight, constant in panel
    frame["wrow"] = rng.uniform(0.5, 2.5, size=ids.size)        # varies within panel
    frame["cl"] = ids % 5                                       # nests the panels
    frame["cl2"] = (ids + year) % 4                             # does not nest the panels
    if unbalanced and gaps:
        inner = np.flatnonzero((year > 2000) & (year < 2000 + periods - 1))
        frame = frame.drop(index=rng.choice(inner, size=ids.size // 6, replace=False))
    elif unbalanced:
        tail = ((ids % 3 == 0) & (year >= 2000 + periods - 2)) | (
            (ids % 3 == 1) & (year == 2000 + periods - 1))
        frame = frame.drop(index=np.flatnonzero(tail))
    return frame.reset_index(drop=True)


@pytest.fixture(scope="module")
def balanced():
    return make_data(seed=3, unbalanced=False)


@pytest.fixture(scope="module")
def unbalanced():
    return make_data(seed=11, unbalanced=True)


@pytest.fixture(scope="module")
def gappy():
    return make_data(seed=23, unbalanced=True, gaps=True)


# ---- generic linear-algebra oracles ---------------------------------------------------


def codes_of(frame):
    codes = pd.factorize(frame.id)[0]
    return codes, int(codes.max()) + 1


def gmean(values, codes, n, w=None):
    w = np.ones(len(codes)) if w is None else w
    values = np.asarray(values, float)
    if values.ndim == 1:
        return np.bincount(codes, weights=w * values, minlength=n) / np.bincount(codes, weights=w, minlength=n)
    return np.column_stack([gmean(values[:, j], codes, n, w) for j in range(values.shape[1])])


def wls(x, y, w=None):
    """Weighted least squares: beta, residuals, (X'WX)^-1, weighted SSR."""
    w = np.ones(len(y)) if w is None else w
    xtwx = x.T @ (w[:, None] * x)
    beta = np.linalg.solve(xtwx, x.T @ (w * y))
    resid = y - x @ beta
    return beta, resid, np.linalg.inv(xtwx), float(w @ resid ** 2)


def wcorr(a, b, w=None):
    w = np.ones(len(a)) if w is None else w
    da, db = a - (w @ a) / w.sum(), b - (w @ b) / w.sum()
    return float((w @ (da * db)) / np.sqrt((w @ da ** 2) * (w @ db ** 2)))


def cr1(x, resid, w, codes, nobs, kk):
    """CR1 sandwich: G/(G-1) (N-1)/(N-K) B (sum_g s_g s_g') B, s_g = sum_{i in g} w_i e_i x_i."""
    w = np.ones(len(resid)) if w is None else w
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    scores = x * (w * resid)[:, None]
    groups = int(codes.max()) + 1
    totals = np.zeros((groups, x.shape[1]))
    np.add.at(totals, codes, scores)
    factor = groups / (groups - 1) * (nobs - 1) / (nobs - kk)
    return factor * bread @ (totals.T @ totals) @ bread, groups


def hc1(x, resid, w, nobs, kk):
    w = np.ones(len(resid)) if w is None else w
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    scores = x * (w * resid)[:, None]
    return nobs / (nobs - kk) * bread @ (scores.T @ scores) @ bread


def driscoll_kraay(x, resid, w, time, lags, nobs, kk):
    """Hoechle's xtscc: Bartlett HAC of the cross-sectional score sums, T/(T-1) (N-1)/(N-K)."""
    w = np.ones(len(resid)) if w is None else w
    periods = np.unique(time)
    big_t = len(periods)
    h = np.zeros((big_t, x.shape[1]))
    for t, period in enumerate(periods):
        rows = time == period
        h[t] = (x[rows] * (w * resid)[rows, None]).sum(0)
    meat = h.T @ h
    for lag in range(1, lags + 1):
        omega = h[lag:].T @ h[:-lag]            # sum_t h_t h_{t-lag}'
        meat += (1 - lag / (lags + 1)) * (omega + omega.T)
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    return bread @ meat @ bread * (big_t / (big_t - 1)) * ((nobs - 1) / (nobs - kk)), big_t


def coef(result):
    return {c.term: c for c in result.coefficients}


def block(result, terms):
    index = {c.term: i for i, c in enumerate(result.coefficients)}
    rows = [index[t] for t in terms]
    return np.array(result.covariance_matrix)[np.ix_(rows, rows)]


def check_inference(result, terms, beta, v, *, df=None, alpha=0.05, rtol=1e-7):
    """Compare estimate, SE, statistic, p-value and CI of every term with the oracle."""
    table = coef(result)
    se = np.sqrt(np.diag(v))
    for i, term in enumerate(terms):
        c = table[term]
        assert_allclose(c.estimate, beta[i], rtol=rtol, err_msg=term)
        assert_allclose(c.std_error, se[i], rtol=rtol, err_msg=term)
        stat = beta[i] / se[i]
        assert_allclose(c.statistic, stat, rtol=rtol, err_msg=term)
        if df is None:
            p, crit = 2 * stats.norm.sf(abs(stat)), stats.norm.ppf(1 - alpha / 2)
        else:
            p, crit = 2 * stats.t.sf(abs(stat), df), stats.t.ppf(1 - alpha / 2, df)
        assert_allclose(c.p_value, p, rtol=1e-6, atol=1e-12, err_msg=term)
        assert_allclose(c.ci_low, beta[i] - crit * se[i], rtol=rtol, err_msg=term)
        assert_allclose(c.ci_high, beta[i] + crit * se[i], rtol=rtol, err_msg=term)
    assert_allclose(block(result, terms), v, rtol=rtol, atol=1e-14)


def check_f(test, statistic, df1, df2):
    assert test["distribution"] == "F"
    assert test["df"] == df1 and test["df2"] == df2
    assert_allclose(test["statistic"], statistic, rtol=1e-7)
    assert_allclose(test["p_value"], stats.f.sf(statistic, df1, df2), rtol=1e-6, atol=1e-14)


def wald_f(beta, v, df2):
    q = len(beta)
    statistic = float(beta @ np.linalg.solve(v, beta)) / q
    return statistic, q, df2


# ---- fixed effects --------------------------------------------------------------------


def lsdv(frame, cols, w=None):
    """Explicit dummy-variable least squares: slopes, nonrobust SEs, sigma_e^2, SSR, effects."""
    codes, n = codes_of(frame)
    dummies = np.eye(n)[codes]
    design = np.column_stack([dummies, frame[cols].to_numpy()])
    y = frame.y.to_numpy()
    w = np.ones(len(y)) if w is None else w * len(y) / w.sum()
    beta, resid, bread, ssr = wls(design, y, w)
    s2 = ssr / (len(y) - n - len(cols))
    return beta[n:], np.sqrt(np.diag(s2 * bread)[n:]), s2, ssr, beta[:n]


def fe_oracle(frame, cols, w=None):
    """Stata's transformed regression: y - ybar_i + ybar on [1, x - xbar_i + xbar]."""
    codes, n = codes_of(frame)
    y, x = frame.y.to_numpy(), frame[cols].to_numpy(float)
    big_n, k = len(y), len(cols)
    w = np.ones(big_n) if w is None else w * big_n / w.sum()
    ybar, xbar = gmean(y, codes, n, w), gmean(x, codes, n, w)
    grand_y, grand_x = (w @ y) / w.sum(), (w[:, None] * x).sum(0) / w.sum()
    ys = y - ybar[codes] + grand_y
    xs = np.column_stack([np.ones(big_n), x - xbar[codes] + grand_x])
    beta, resid, bread, ssr = wls(xs, ys, w)
    df_resid = big_n - n - k
    s2 = ssr / df_resid
    slopes = beta[1:]
    xb_bar = xbar @ slopes
    u = ybar - xb_bar
    sigma_u = float(np.std(u, ddof=1))
    pooled_ssr = wls(np.column_stack([np.ones(big_n), x]), y, w)[3]
    return {
        "beta": beta, "v": s2 * bread, "resid": resid, "xs": xs, "w": w, "ssr": ssr,
        "df_resid": df_resid, "sigma_e": np.sqrt(s2), "sigma_u": sigma_u,
        "rho": sigma_u ** 2 / (sigma_u ** 2 + s2),
        # Stata's e(corr) = corr(u_i, x_it b) over the N observations (not over panels).
        "corr_u_xb": wcorr(u[codes], x @ slopes, w),
        "r2w": wcorr((x - xbar[codes]) @ slopes, y - ybar[codes], w) ** 2,
        "r2b": wcorr(xb_bar, ybar) ** 2, "r2o": wcorr(x @ slopes, y, w) ** 2,
        "f_u": ((pooled_ssr - ssr) / (n - 1)) / (ssr / df_resid), "n": n, "codes": codes,
        "sizes": np.bincount(codes).astype(float), "nobs": big_n,   # T_i are row counts
    }


def check_fe_metrics(result, o):
    m = result.metrics
    assert list(m) == ["r_squared_within", "r_squared_between", "r_squared_overall", "sigma_u",
                       "sigma_e", "rho", "corr_u_xb", "rmse", "df_resid", "n_groups", "t_min",
                       "t_avg", "t_max"]
    assert_allclose([m["r_squared_within"], m["r_squared_between"], m["r_squared_overall"]],
                    [o["r2w"], o["r2b"], o["r2o"]], rtol=1e-8)
    assert_allclose([m["sigma_u"], m["sigma_e"], m["rho"], m["corr_u_xb"], m["rmse"]],
                    [o["sigma_u"], o["sigma_e"], o["rho"], o["corr_u_xb"], o["sigma_e"]], rtol=1e-8)
    assert m["df_resid"] == o["df_resid"] and m["n_groups"] == o["n"]
    assert_allclose([m["t_min"], m["t_avg"], m["t_max"]],
                    [o["sizes"].min(), o["nobs"] / o["n"], o["sizes"].max()], rtol=1e-12)


@pytest.mark.parametrize("name", ["balanced", "unbalanced", "gappy"])
def test_fe_matches_lsdv_and_transformed_regression(request, name):
    frame = request.getfixturevalue(name)
    result = oe.xtreg(data=frame, y="y", x=COLS, panel="id", time="year")
    slopes, se, s2, ssr, _ = lsdv(frame, VARY)
    o = fe_oracle(frame, VARY)
    assert result.inference["use_t"] is True
    assert result.nobs == len(frame)
    # z has no within variation: omitted and recorded, never silently.
    assert "z" not in coef(result) and "z" in result.provenance["omitted_terms"]
    assert any("z" in w for w in result.warnings)
    assert_allclose([coef(result)[t].estimate for t in VARY], slopes, rtol=1e-9)
    assert_allclose([coef(result)[t].std_error for t in VARY], se, rtol=1e-8)
    assert_allclose(result.metrics["sigma_e"], np.sqrt(s2), rtol=1e-10)
    check_inference(result, ["Intercept", *VARY], o["beta"], o["v"], df=o["df_resid"])
    assert result.inference["df_inference"] == o["df_resid"]
    check_fe_metrics(result, o)
    stat, q, df2 = wald_f(o["beta"][1:], o["v"][1:, 1:], o["df_resid"])
    check_f(result.tests["model"], stat, q, df2)
    check_f(result.tests["fixed_effects"], o["f_u"], o["n"] - 1, o["df_resid"])
    assert_allclose(result.extra["ssr"], ssr, rtol=1e-10)


def test_fe_sigma_u_is_sd_of_lsdv_effects(unbalanced):
    """u_i from LSDV differ from ybar_i - xbar_i'b by the constant only: same sd."""
    result = oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id")
    _, _, _, _, effects = lsdv(unbalanced, VARY)
    assert_allclose(result.metrics["sigma_u"], np.std(effects, ddof=1), rtol=1e-9)


def test_fe_categorical_and_collinear_columns(gappy):
    frame = gappy.copy()
    frame["x3"] = 2 * frame.x1 + 1          # collinear with x1 (and the constant)
    result = oe.xtreg(data=frame, y="y", x=["x1", "x2", "x3", "cat", "z"], panel="id",
                      categorical=["cat"], missing="raise")
    frame["cat[b]"] = (frame.cat == "b").astype(float)
    frame["cat[c]"] = (frame.cat == "c").astype(float)
    cols = ["x1", "x2", "cat[b]", "cat[c]"]
    assert set(result.provenance["omitted_terms"]) == {"x3", "z"}
    o = fe_oracle(frame, cols)
    check_inference(result, ["Intercept", *cols], o["beta"], o["v"], df=o["df_resid"])
    check_fe_metrics(result, o)
    check_f(result.tests["fixed_effects"], o["f_u"], o["n"] - 1, o["df_resid"])


def test_fe_robust_is_cluster_on_panel_with_k_plus_one(unbalanced, gappy):
    for frame in (unbalanced, gappy):
        result = oe.xtreg(data=frame, y="y", x=VARY, panel="id", covariance="robust")
        o = fe_oracle(frame, VARY)
        v, groups = cr1(o["xs"], o["resid"], None, o["codes"], o["nobs"], len(VARY) + 1)
        check_inference(result, ["Intercept", *VARY], o["beta"], v, df=groups - 1)
        assert result.inference["df_inference"] == groups - 1
        assert result.inference["cluster_count"] == groups
        assert_allclose(result.inference["small_sample_correction"],
                        groups / (groups - 1) * (o["nobs"] - 1) / (o["nobs"] - 3), rtol=1e-12)
        # The same thing through statsmodels (cluster on the transformed regression).
        sm_fit = sm.OLS(o["xs"] @ o["beta"] + o["resid"], o["xs"]).fit(
            cov_type="cluster", cov_kwds={"groups": o["codes"]}, use_t=True)
        assert_allclose(block(result, ["Intercept", *VARY]), sm_fit.cov_params(), rtol=1e-9)
        assert_allclose([coef(result)[t].p_value for t in VARY], sm_fit.pvalues[1:], rtol=1e-6)
        stat, q, df2 = wald_f(o["beta"][1:], v[1:, 1:], groups - 1)
        check_f(result.tests["model"], stat, q, df2)
        assert "fixed_effects" not in result.tests


def test_fe_cluster_requires_nested_panels(unbalanced):
    """xtreg, fe / re: the panel variable must be nested within the cluster variable.

    Stata refuses otherwise (r(498) 'panels are not nested within clusters'); regress
    (model='pooled') accepts any cluster column.
    """
    o = fe_oracle(unbalanced, VARY)
    k, big_n = len(VARY), o["nobs"]
    # cl nests the panels: K = k + 1.
    nested = oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", covariance="cluster",
                      cluster="cl")
    codes = pd.factorize(unbalanced.cl)[0]
    v, groups = cr1(o["xs"], o["resid"], None, codes, big_n, k + 1)
    check_inference(nested, ["Intercept", *VARY], o["beta"], v, df=groups - 1)
    assert nested.inference["k_small_sample"] == k + 1
    assert not any("absorbed" in w for w in nested.warnings)
    # cl2 does not nest the panels: an actionable error for fe and re, never the areg K.
    for model in ("fe", "re"):
        with pytest.raises(AnalysisError) as err:
            oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", model=model,
                     covariance="cluster", cluster="cl2")
        assert err.value.code == "cluster_not_nested"
        assert "cl2" in str(err.value) and "pooled" in str(err.value)
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", time="year", covariance="cluster",
                 cluster=["cl", "year"])
    assert err.value.code == "cluster_not_nested" and "year" in str(err.value)
    pooled = oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", model="pooled",
                      covariance="cluster", cluster="cl2")
    assert pooled.inference["cluster_count"] == unbalanced.cl2.nunique()


def pooled_oracle(frame, cols):
    xs = np.column_stack([np.ones(len(frame)), frame[cols].to_numpy(float)])
    beta, resid, _, _ = wls(xs, frame.y.to_numpy())
    return {"beta": beta, "xs": xs, "resid": resid, "nobs": len(frame)}


def two_way_oracle(frame, o):
    """CGM: M = M_cl + M_cl2 - M_(cl x cl2), G_min/(G_min-1) (N-1)/(N-K), df G_min - 1."""
    xs, resid, big_n, kk = o["xs"], o["resid"], o["nobs"], o["xs"].shape[1]
    bread = np.linalg.inv(xs.T @ xs)
    scores = xs * resid[:, None]

    def meat(codes):
        totals = np.zeros((codes.max() + 1, kk))
        np.add.at(totals, codes, scores)
        return totals.T @ totals

    a = pd.factorize(frame.cl)[0]
    b = pd.factorize(frame.cl2)[0]
    ab = pd.factorize(pd.Series(a * (b.max() + 1) + b))[0]
    g_min = min(a.max() + 1, b.max() + 1)
    factor = g_min / (g_min - 1) * (big_n - 1) / (big_n - kk)
    return factor * bread @ (meat(a) + meat(b) - meat(ab)) @ bread, g_min


def test_pooled_two_way_cluster_inclusion_exclusion():
    """Crossed clusters are legal for model='pooled' (regress): CGM inclusion-exclusion."""
    frame = make_data(seed=5, n=60, periods=8, unbalanced=True, gaps=True)
    frame["cl"] = frame.id % 12                      # nests the panels
    frame["cl2"] = (frame.id * 7 + frame.year) % 9   # crosses them
    o = pooled_oracle(frame, VARY)
    v, g_min = two_way_oracle(frame, o)
    assert np.linalg.eigvalsh(v).min() > 0
    result = oe.xtreg(data=frame, y="y", x=VARY, panel="id", model="pooled",
                      covariance="cluster", cluster=["cl", "cl2"])
    check_inference(result, ["Intercept", *VARY], o["beta"], v, df=g_min - 1)
    assert result.inference["cluster_counts"] == [12, 9]
    assert result.inference["psd_adjusted"] is False
    # The same two columns are refused for fe (cl2 crosses the panels).
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=frame, y="y", x=VARY, panel="id", covariance="cluster",
                 cluster=["cl", "cl2"])
    assert err.value.code == "cluster_not_nested"


def test_fe_and_re_two_way_cluster_with_both_columns_nesting_the_panels():
    """Multiway clustering is legal for fe when every column nests the panels (Stata's rule).

    CGM inclusion-exclusion on Stata's transformed regression with K = k + 1, one factor
    G_min/(G_min-1) (N-1)/(N-K) and G_min - 1 df. The fit runs on centered slopes; a
    sandwich without eigenvalue repair is equivariant, so it equals the oracle on
    [1, x - xbar_i + xbar]. For re (a single cluster variable in Stata) the same sandwich on
    the GLS-transformed regression is an OpenEcon extension.
    """
    frame = make_data(seed=5, n=200, periods=8, unbalanced=True, gaps=True)
    frame["cl"] = frame.id % 20                      # both nest the panels and cross each other
    frame["cl2"] = frame.id % 15
    o = fe_oracle(frame, VARY)
    v, g_min = two_way_oracle(frame, o)
    assert g_min == 15 and np.linalg.eigvalsh(v[1:, 1:]).min() > 0
    fe = oe.xtreg(data=frame, y="y", x=VARY, panel="id", covariance="cluster",
                  cluster=["cl", "cl2"])
    assert fe.inference["psd_adjusted"] is False and fe.inference["cluster_counts"] == [20, 15]
    assert fe.inference["k_small_sample"] == 3
    check_inference(fe, ["Intercept", *VARY], o["beta"], v, df=g_min - 1)
    r = re_oracle(frame, VARY)
    ro = {"xs": r["xs"], "resid": r["resid"], "nobs": r["nobs"]}
    v_re, _ = two_way_oracle(frame, ro)
    assert np.linalg.eigvalsh(v_re).min() > 0
    re = oe.xtreg(data=frame, y="y", x=VARY, panel="id", model="re", covariance="cluster",
                  cluster=["cl", "cl2"])
    check_inference(re, ["Intercept", *VARY], r["beta"], v_re, df=None)
    assert re.inference["cluster_df"] == g_min - 1 and re.inference["use_t"] is False
    # 12 x 9 nested clusters on 60 panels: the meat is indefinite. The eigenvalue fix is not
    # equivariant, so it is applied where regress would apply it, on Stata's transformed
    # (uncentered) design, and the oracle's fix on that design must be reproduced.
    small = make_data(seed=5, n=60, periods=8, unbalanced=True, gaps=True)
    small["cl"], small["cl2"] = small.id % 12, small.id % 9
    for model, fit in (("fe", fe_oracle(small, VARY)), ("re", re_oracle(small, VARY))):
        v, _ = two_way_oracle(small, fit)
        assert np.linalg.eigvalsh(v).min() < -1e-5
        xtx = fit["xs"].T @ fit["xs"]
        meat = xtx @ v @ xtx
        values, vectors = np.linalg.eigh((meat + meat.T) / 2)
        repaired = (vectors * np.clip(values, 0, None)) @ vectors.T
        expected = np.linalg.solve(xtx, np.linalg.solve(xtx, repaired).T)
        result = oe.xtreg(data=small, y="y", x=VARY, panel="id", model=model,
                          covariance="cluster", cluster=["cl", "cl2"])
        assert result.inference["psd_adjusted"] is True, model
        assert any("positive semidefinite" in w for w in result.warnings)
        assert_allclose(np.array(result.covariance_matrix), expected, rtol=1e-7, atol=1e-14,
                        err_msg=model)


def test_fe_constant_with_zero_mean_regressors_is_flagged(unbalanced):
    """Var(_cons) under panel clustering is xbar'V xbar: within residuals sum to zero in
    every panel. With standardized regressors that is rounding noise (SE ~ 1e-17, p = 0);
    Stata prints it, OpenEcon prints it with a warning. Slopes are untouched."""
    centered = unbalanced.assign(x1=unbalanced.x1 - unbalanced.x1.mean(),
                                 x2=unbalanced.x2 - unbalanced.x2.mean())
    plain = oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", covariance="robust")
    assert not any("Intercept" in w for w in plain.warnings)
    o = fe_oracle(unbalanced, VARY)
    v, _ = cr1(o["xs"], o["resid"], None, o["codes"], o["nobs"], 3)
    xbar = unbalanced[VARY].to_numpy().mean(0)
    assert_allclose(coef(plain)["Intercept"].std_error ** 2, xbar @ v[1:, 1:] @ xbar, rtol=1e-8)
    for covariance, extra in (("robust", {}), ("cluster", {"cluster": "cl"})):
        flagged = oe.xtreg(data=centered, y="y", x=VARY, panel="id", covariance=covariance,
                           **extra)
        assert any("zero up to rounding" in w for w in flagged.warnings), covariance
        assert coef(flagged)["Intercept"].std_error < 1e-12
        reference = oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", covariance=covariance,
                             **extra)
        for term in VARY:
            assert_allclose(coef(flagged)[term].std_error, coef(reference)[term].std_error,
                            rtol=1e-9)
    # The conventional and Driscoll-Kraay variances of the constant are ordinary numbers.
    for covariance in ("nonrobust", "driscoll_kraay"):
        fit = oe.xtreg(data=centered, y="y", x=VARY, panel="id", time="year",
                       covariance=covariance)
        assert coef(fit)["Intercept"].std_error > 1e-3
        assert not any("zero up to rounding" in w for w in fit.warnings)


def test_pooled_two_way_cluster_indefinite_meat_gets_cgm_fix():
    """With 5 x 4 crossed clusters the inclusion-exclusion meat is indefinite.

    The shared covariance layer applies the Cameron-Gelbach-Miller eigenvalue fix to the
    meat (negative eigenvalues set to zero), records inference['psd_adjusted'] and warns.
    The oracle applies the same fix to its own meat.
    """
    frame = make_data(seed=2, n=24, periods=7, unbalanced=True)
    o = pooled_oracle(frame, VARY)
    v, g_min = two_way_oracle(frame, o)
    assert np.linalg.eigvalsh(v).min() < 0
    result = oe.xtreg(data=frame, y="y", x=VARY, panel="id", model="pooled",
                      covariance="cluster", cluster=["cl", "cl2"])
    assert result.inference["psd_adjusted"] is True
    assert any("positive semidefinite" in w for w in result.warnings)
    xtx = o["xs"].T @ o["xs"]
    meat = xtx @ v @ xtx                                  # the factored CGM meat
    values, vectors = np.linalg.eigh((meat + meat.T) / 2)
    fixed = (vectors * np.clip(values, 0, None)) @ vectors.T
    expected = np.linalg.solve(xtx, np.linalg.solve(xtx, fixed).T)
    assert_allclose(np.array(result.covariance_matrix), expected, rtol=1e-7, atol=1e-14)
    assert np.linalg.eigvalsh(expected).min() >= -1e-12
    assert result.inference["df_inference"] == g_min - 1


def test_fe_driscoll_kraay(balanced, gappy):
    for frame, lags in ((balanced, 2), (gappy, 1), (balanced, 0)):
        result = oe.xtreg(data=frame, y="y", x=VARY, panel="id", time="year",
                          covariance="driscoll_kraay", lags=lags)
        o = fe_oracle(frame, VARY)
        v, big_t = driscoll_kraay(o["xs"], o["resid"], None, frame.year.to_numpy(), lags,
                                  o["nobs"], len(VARY) + 1)
        check_inference(result, ["Intercept", *VARY], o["beta"], v, df=big_t - 1)
        assert result.inference["lags"] == lags and result.inference["df_inference"] == big_t - 1
        stat, q, df2 = wald_f(o["beta"][1:], v[1:, 1:], big_t - 1)
        check_f(result.tests["model"], stat, q, df2)
    # Default bandwidth floor(4 (T/100)^(2/9)); T = 7 -> 2.
    default = oe.xtreg(data=balanced, y="y", x=VARY, panel="id", time="year",
                       covariance="driscoll_kraay")
    assert default.inference["lags"] == int(np.floor(4 * (7 / 100) ** (2 / 9))) == 2
    two = oe.xtreg(data=balanced, y="y", x=VARY, panel="id", time="year",
                   covariance="driscoll_kraay", lags=2)
    assert_allclose(default.covariance_matrix, two.covariance_matrix, rtol=1e-14)


def test_fe_aweights_pweights_and_fweights(unbalanced):
    frame = unbalanced
    wa = frame.wa.to_numpy()
    o = fe_oracle(frame, VARY, wa)
    a = oe.xtreg(data=frame, y="y", x=VARY, panel="id", weights="wa", weight_type="aweight")
    check_inference(a, ["Intercept", *VARY], o["beta"], o["v"], df=o["df_resid"])
    check_fe_metrics(a, o)
    check_f(a.tests["fixed_effects"], o["f_u"], o["n"] - 1, o["df_resid"])
    slopes, se, s2, _, _ = lsdv(frame, VARY, wa)
    assert_allclose([coef(a)[t].std_error for t in VARY], se, rtol=1e-8)
    # pweights: same point estimates, covariance must be the panel-cluster sandwich on w x e.
    p = oe.xtreg(data=frame, y="y", x=VARY, panel="id", weights="wa", weight_type="pweight",
                 covariance="robust")
    v, groups = cr1(o["xs"], o["resid"], o["w"], o["codes"], o["nobs"], len(VARY) + 1)
    check_inference(p, ["Intercept", *VARY], o["beta"], v, df=groups - 1)
    ar = oe.xtreg(data=frame, y="y", x=VARY, panel="id", weights="wa", weight_type="aweight",
                  covariance="robust")
    assert_allclose(ar.covariance_matrix, p.covariance_matrix, rtol=1e-12)
    # fweights: identical to the row-replicated dataset without weights.
    f = oe.xtreg(data=frame, y="y", x=VARY, panel="id", weights="wf", weight_type="fweight")
    expanded = frame.loc[frame.index.repeat(frame.wf.astype(int))].reset_index(drop=True)
    e = oe.xtreg(data=expanded, y="y", x=VARY, panel="id")
    assert f.nobs == e.nobs == len(expanded)
    assert_allclose([c.estimate for c in f.coefficients], [c.estimate for c in e.coefficients],
                    rtol=1e-10)
    assert_allclose(f.covariance_matrix, e.covariance_matrix, rtol=1e-9)
    for name in f.metrics:
        assert_allclose(f.metrics[name], e.metrics[name], rtol=1e-9, err_msg=name)
    for name in ("model", "fixed_effects"):
        assert_allclose(f.tests[name]["statistic"], e.tests[name]["statistic"], rtol=1e-9)
        assert f.tests[name]["df2"] == e.tests[name]["df2"]
    fr = oe.xtreg(data=frame, y="y", x=VARY, panel="id", weights="wf", weight_type="fweight",
                  covariance="robust")
    er = oe.xtreg(data=expanded, y="y", x=VARY, panel="id", covariance="robust")
    assert_allclose(fr.covariance_matrix, er.covariance_matrix, rtol=1e-9)


def test_fe_errors(unbalanced):
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", weights="wrow",
                 weight_type="aweight")
    assert err.value.code == "weights_vary_within_panel"
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", weights="wa",
                 weight_type="pweight")
    assert err.value.code == "unsupported_covariance"
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", covariance="driscoll_kraay")
    assert err.value.code == "invalid_spec"          # no time column
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", lags=2)
    assert err.value.code == "invalid_spec"          # lags without Driscoll-Kraay
    dup = pd.concat([unbalanced, unbalanced.head(3)]).reset_index(drop=True)
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=dup, y="y", x=VARY, panel="id", time="year")
    assert err.value.code == "repeated_time_values"
    holes = unbalanced.copy()
    holes.loc[[0, 5], "x1"] = np.nan
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=holes, y="y", x=VARY, panel="id")
    assert err.value.code == "missing_values"
    dropped = oe.xtreg(data=holes, y="y", x=VARY, panel="id", missing="drop")
    clean = oe.xtreg(data=holes.dropna(), y="y", x=VARY, panel="id")
    assert dropped.nobs == len(holes) - 2 and dropped.dropped_rows == 2
    assert_allclose(dropped.covariance_matrix, clean.covariance_matrix, rtol=1e-12)
    # N - n - k = 0: no residual degrees of freedom.
    tiny = make_data(seed=5, n=3, periods=2, unbalanced=False)
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=tiny, y="y", x=["x1", "x2", "cat"], panel="id", categorical=["cat"])
    assert err.value.code in {"insufficient_observations", "singular_design"}


def test_fe_tiny_sample_df_one():
    frame = make_data(seed=9, n=4, periods=3, unbalanced=False)[["id", "year", "y", "x1", "x2"]]
    frame = frame.iloc[:11]                      # N = 11, n = 4, k = 2 -> df = 5
    result = oe.xtreg(data=frame, y="y", x=VARY, panel="id")
    o = fe_oracle(frame, VARY)
    assert o["df_resid"] == 5
    check_inference(result, ["Intercept", *VARY], o["beta"], o["v"], df=5)
    check_fe_metrics(result, o)


def test_fe_large_offset_regressor_with_small_within_variation():
    """A regressor like a price level (1e4) with within sd 0.05 has real within variation.

    Stata keeps it (its collinearity check runs on the within-transformed data). The
    transformed design [1, x - xbar_i + xbar] has condition number ~2e9 here, which the
    normal equations of ``wls`` cannot resolve, so the oracle solves the within regression
    (condition number ~20) and rebuilds (X*'X*)^-1 from the exact block inverse
    [[1/N + m'A^-1 m, -m'A^-1], [-A^-1 m, A^-1]] with A = Xw'Xw and m the grand means.
    """
    frame = make_data(seed=4, unbalanced=False)
    rng = np.random.default_rng(1)
    frame["level"] = 1e4 + 0.05 * rng.normal(size=len(frame))
    frame["y"] = frame.y + 3.0 * (frame.level - 1e4)
    cols = ["x1", "x2", "level"]
    result = oe.xtreg(data=frame, y="y", x=cols, panel="id")
    assert "level" in coef(result), result.warnings
    codes, n = codes_of(frame)
    y, x = frame.y.to_numpy(), frame[cols].to_numpy(float)
    big_n, k = len(y), len(cols)
    wx, wy = x - gmean(x, codes, n)[codes], y - gmean(y, codes, n)[codes]
    assert np.linalg.cond(wx) < 100
    slopes = np.linalg.lstsq(wx, wy, rcond=None)[0]
    resid = wy - wx @ slopes
    s2 = (resid @ resid) / (big_n - n - k)
    m = x.mean(0)
    a_inv = np.linalg.inv(wx.T @ wx)
    v = np.empty((k + 1, k + 1))
    v[0, 0] = 1 / big_n + m @ a_inv @ m
    v[0, 1:] = v[1:, 0] = -a_inv @ m
    v[1:, 1:] = a_inv
    beta = np.concatenate([[y.mean() - m @ slopes], slopes])
    check_inference(result, ["Intercept", *cols], beta, s2 * v, df=big_n - n - k)
    assert_allclose(result.metrics["sigma_e"], np.sqrt(s2), rtol=1e-10)


# ---- random effects -------------------------------------------------------------------


def re_oracle(frame, cols, sa=False):
    """Swamy-Arora GLS from first principles (harmonic-mean T_bar or Baltagi-Chang).

    The conventional covariance is the OLS one of the transformed regression,
    RSS*/(N - K) (X*'X*)^-1 (Stata: e(rmse) is the 'root mean squared error of GLS
    regression'), not sigma_e^2 (X*'X*)^-1.
    """
    codes, n = codes_of(frame)
    y, x = frame.y.to_numpy(), frame[cols].to_numpy(float)
    big_n, k = len(y), len(cols)
    sizes = np.bincount(codes).astype(float)
    w = np.ones(big_n)
    xbar, ybar = gmean(x, codes, n), gmean(y, codes, n)
    within_x, within_y = x - xbar[codes], y - ybar[codes]
    varying = [j for j in range(k)
               if (within_x[:, j] ** 2).sum() > 1e-10 * ((x[:, j] - x[:, j].mean()) ** 2).sum()]
    ssr_w = wls(within_x[:, varying], within_y)[3]
    s2e = ssr_w / (big_n - n - len(varying))
    zbar = np.column_stack([np.ones(n), xbar])
    ssr_b = wls(zbar, ybar)[3]
    df_b = n - zbar.shape[1]
    t_bar = n / (1 / sizes).sum()
    s2u = max(0.0, ssr_b / df_b - s2e / t_bar)
    if sa:
        # E[RSS of the T_i-weighted between regression] = s2u [N - tr((Z'PZ)^-1 Z'P~Z)] + s2e (n - K_b)
        ssr_bw = wls(zbar, ybar, sizes)[3]
        ztpz = zbar.T @ (sizes[:, None] * zbar)
        ztppz = zbar.T @ (sizes[:, None] ** 2 * zbar)
        s2u = max(0.0, (ssr_bw - df_b * s2e) / (big_n - np.trace(np.linalg.solve(ztpz, ztppz))))
    theta = 1 - np.sqrt(s2e / (sizes * s2u + s2e))
    xs = np.column_stack([1 - theta[codes], x - theta[codes, None] * xbar[codes]])
    ys = y - theta[codes] * ybar[codes]
    beta, resid, bread, ssr = wls(xs, ys)
    rmse2 = ssr / (big_n - xs.shape[1])
    slopes = beta[1:]
    pooled_resid = wls(np.column_stack([np.ones(big_n), x]), y)[1]
    sums = np.bincount(codes, weights=pooled_resid)
    lm = big_n ** 2 / (2 * ((sizes ** 2).sum() - big_n)) * (
        (sums ** 2).sum() / (pooled_resid ** 2).sum() - 1) ** 2
    return {
        "beta": beta, "v": rmse2 * bread, "xs": xs, "resid": resid, "w": w, "codes": codes,
        "n": n, "nobs": big_n, "sizes": sizes, "s2e": s2e, "s2u": s2u, "t_bar": t_bar,
        "theta": theta, "rmse": np.sqrt(rmse2),
        "r2w": wcorr(within_x @ slopes, within_y) ** 2,
        "r2b": wcorr(xbar @ slopes, ybar) ** 2, "r2o": wcorr(x @ slopes, y) ** 2,
        "lm": lm, "ssr_w": ssr_w, "ssr_b": ssr_b,
    }


def check_re(result, o, terms):
    assert result.inference["use_t"] is False and result.inference["df_inference"] is None
    check_inference(result, terms, o["beta"], o["v"], df=None)
    m = result.metrics
    assert_allclose([m["r_squared_within"], m["r_squared_between"], m["r_squared_overall"]],
                    [o["r2w"], o["r2b"], o["r2o"]], rtol=1e-8)
    assert_allclose([m["sigma_u"], m["sigma_e"], m["rho"], m["rmse"]],
                    [np.sqrt(o["s2u"]), np.sqrt(o["s2e"]), o["s2u"] / (o["s2u"] + o["s2e"]),
                     o["rmse"]], rtol=1e-9)
    assert m["n_groups"] == o["n"]
    assert_allclose([m["t_min"], m["t_avg"], m["t_max"]],
                    [o["sizes"].min(), o["nobs"] / o["n"], o["sizes"].max()], rtol=1e-12)
    balanced = np.all(o["sizes"] == o["sizes"][0])
    if balanced:
        assert_allclose(m["theta"], o["theta"][0], rtol=1e-10)
    else:
        assert m["theta"] is None
        assert_allclose([result.extra["theta"]["min"], result.extra["theta"]["max"]],
                        [o["theta"].min(), o["theta"].max()], rtol=1e-10, atol=1e-12)
        assert_allclose(result.extra["theta"]["median"], np.median(o["theta"]), rtol=1e-10,
                        atol=1e-12)
    wald = result.tests["model"]
    slopes = o["beta"][1:]
    assert wald["distribution"] == "chi2" and wald["df"] == len(slopes)
    stat = float(slopes @ np.linalg.solve(o["v"][1:, 1:], slopes))
    assert_allclose(wald["statistic"], stat, rtol=1e-7)
    assert_allclose(wald["p_value"], stats.chi2.sf(stat, len(slopes)), rtol=1e-6, atol=1e-14)
    bp = result.tests["breusch_pagan"]
    assert_allclose(bp["statistic"], o["lm"], rtol=1e-8)
    assert_allclose(bp["p_value"], 0.5 * stats.chi2.sf(o["lm"], 1), rtol=1e-6, atol=1e-14)
    assert bp["df"] == 1 and "halved" in bp["label"]


@pytest.mark.parametrize("name", ["balanced", "unbalanced", "gappy"])
def test_re_swamy_arora(request, name):
    frame = request.getfixturevalue(name)
    result = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="re")
    o = re_oracle(frame, COLS)
    check_re(result, o, ["Intercept", *COLS])
    components = result.extra["variance_components"]
    assert_allclose(components["sigma_e_squared"], o["s2e"], rtol=1e-10)
    assert_allclose(components["sigma_u_squared"], o["s2u"], rtol=1e-10)
    assert_allclose(components["t_bar_harmonic"], o["t_bar"], rtol=1e-12)
    # sigma_e is the within one: identical to the fe fit.
    fe = oe.xtreg(data=frame, y="y", x=COLS, panel="id")
    assert_allclose(result.metrics["sigma_e"], fe.metrics["sigma_e"], rtol=1e-12)


def test_re_sa_balanced_equals_default_and_unbalanced_formula(balanced, gappy):
    plain = oe.xtreg(data=balanced, y="y", x=COLS, panel="id", model="re")
    sa = oe.xtreg(data=balanced, y="y", x=COLS, panel="id", model="re", sa=True)
    assert_allclose(sa.covariance_matrix, plain.covariance_matrix, rtol=1e-10)
    assert_allclose(sa.metrics["sigma_u"], plain.metrics["sigma_u"], rtol=1e-10)
    result = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="re", sa=True)
    o = re_oracle(gappy, COLS, sa=True)
    assert o["s2u"] != pytest.approx(re_oracle(gappy, COLS)["s2u"])
    check_re(result, o, ["Intercept", *COLS])


def independent_columns(x, tol=1e-9):
    """Left-to-right greedy rank screen (Stata's omission order)."""
    kept = []
    for j in range(x.shape[1]):
        trial = x[:, [*kept, j]]
        if np.linalg.matrix_rank(trial, tol=tol * np.linalg.norm(trial)) > len(kept):
            kept.append(j)
    return kept


def test_within_collinear_regressor(gappy):
    """x3 = x1 + c_i is collinear with x1 inside panels only.

    fe omits x3 (left to right); re drops it from the within step (k_w = 2, so sigma_e^2
    has N - n - 2 df) but keeps it in the between and GLS regressions.
    """
    frame = gappy.copy()
    rng = np.random.default_rng(77)
    frame["x3"] = frame.x1 + rng.normal(size=frame.id.nunique())[pd.factorize(frame.id)[0]]
    cols = ["x1", "x2", "x3", "z"]
    fe = oe.xtreg(data=frame, y="y", x=cols, panel="id")
    assert set(fe.provenance["omitted_terms"]) == {"x3", "z"}
    o = fe_oracle(frame, VARY)
    check_inference(fe, ["Intercept", *VARY], o["beta"], o["v"], df=o["df_resid"])
    re = oe.xtreg(data=frame, y="y", x=cols, panel="id", model="re")
    codes, n = codes_of(frame)
    y, x = frame.y.to_numpy(), frame[cols].to_numpy(float)
    big_n = len(y)
    sizes = np.bincount(codes).astype(float)
    xbar, ybar = gmean(x, codes, n), gmean(y, codes, n)
    within_x, within_y = x - xbar[codes], y - ybar[codes]
    varying = [j for j in range(x.shape[1]) if (within_x[:, j] ** 2).sum() > 1e-10 * (x[:, j] ** 2).sum()]
    within_cols = [varying[i] for i in independent_columns(within_x[:, varying])]
    assert within_cols == [0, 1]
    ssr_w = wls(within_x[:, within_cols], within_y)[3]
    s2e = ssr_w / (big_n - n - len(within_cols))
    zbar = np.column_stack([np.ones(n), xbar])
    ssr_b = wls(zbar, ybar)[3]
    s2u = max(0.0, ssr_b / (n - zbar.shape[1]) - s2e / (n / (1 / sizes).sum()))
    theta = 1 - np.sqrt(s2e / (sizes * s2u + s2e))
    xs = np.column_stack([1 - theta[codes], x - theta[codes, None] * xbar[codes]])
    beta, resid, bread, ssr = wls(xs, y - theta[codes] * ybar[codes])
    assert re.provenance["omitted_terms"] == []
    check_inference(re, ["Intercept", *cols], beta, ssr / (big_n - xs.shape[1]) * bread, df=None)
    assert re.extra["variance_components"]["within_regressors"] == 2
    assert re.extra["variance_components"]["df_within"] == big_n - n - 2
    assert_allclose(re.metrics["sigma_e"], np.sqrt(s2e), rtol=1e-10)
    assert_allclose(re.metrics["sigma_u"], np.sqrt(s2u), rtol=1e-10)


def test_re_categorical(gappy):
    frame = gappy.copy()
    result = oe.xtreg(data=frame, y="y", x=["x1", "cat", "z"], panel="id", model="re",
                      categorical=["cat"])
    frame["cat[b]"] = (frame.cat == "b").astype(float)
    frame["cat[c]"] = (frame.cat == "c").astype(float)
    cols = ["x1", "cat[b]", "cat[c]", "z"]
    check_re(result, re_oracle(frame, cols), ["Intercept", *cols])


def test_re_robust_and_cluster(unbalanced):
    o = re_oracle(unbalanced, COLS)
    kk = len(COLS) + 1
    robust = oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", model="re", covariance="robust")
    v, groups = cr1(o["xs"], o["resid"], None, o["codes"], o["nobs"], kk)
    check_inference(robust, ["Intercept", *COLS], o["beta"], v, df=None)
    assert robust.inference["cluster_count"] == groups
    clustered = oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", model="re",
                         covariance="cluster", cluster="cl")
    v2, _ = cr1(o["xs"], o["resid"], None, pd.factorize(unbalanced.cl)[0], o["nobs"], kk)
    check_inference(clustered, ["Intercept", *COLS], o["beta"], v2, df=None)
    wald = robust.tests["model"]
    stat = float(o["beta"][1:] @ np.linalg.solve(v[1:, 1:], o["beta"][1:]))
    assert wald["distribution"] == "chi2"
    assert_allclose(wald["statistic"], stat, rtol=1e-7)


def test_re_rejects_weights(unbalanced):
    """Stata's xtreg, re takes no weights (the manual allows them for fe only, iweights for mle)."""
    for column, weight_type, covariance in (("wa", "aweight", None), ("wa", "pweight", "robust"),
                                            ("wf", "fweight", None)):
        with pytest.raises(AnalysisError) as err:
            oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", model="re", weights=column,
                     weight_type=weight_type, covariance=covariance)
        assert err.value.code == "unsupported_weights"
        assert "model='re'" in str(err.value) and "'fe'" in str(err.value)


def test_re_sigma_u_truncated_at_zero_reduces_to_pooled():
    frame = make_data(seed=8, n=20, periods=5, unbalanced=False)
    rng = np.random.default_rng(2)
    frame["y"] = 1 + 0.5 * frame.x1 + rng.normal(size=len(frame))    # no panel effect
    re = oe.xtreg(data=frame, y="y", x=["x1"], panel="id", model="re")
    o = re_oracle(frame, ["x1"])
    if o["s2u"] == 0:
        assert re.metrics["sigma_u"] == 0
        assert any("zero" in w for w in re.warnings)
        pooled = oe.xtreg(data=frame, y="y", x=["x1"], panel="id", model="pooled")
        assert_allclose([c.estimate for c in re.coefficients],
                        [c.estimate for c in pooled.coefficients], rtol=1e-10)
    check_re(re, o, ["Intercept", "x1"])


# ---- between ---------------------------------------------------------------------------


def be_oracle(frame, cols, wls_weights=False):
    codes, n = codes_of(frame)
    y, x = frame.y.to_numpy(), frame[cols].to_numpy(float)
    sizes = np.bincount(codes).astype(float)
    xbar, ybar = gmean(x, codes, n), gmean(y, codes, n)
    zbar = np.column_stack([np.ones(n), xbar])
    w = sizes if wls_weights else None
    beta, resid, bread, ssr = wls(zbar, ybar, w)
    kk = zbar.shape[1]
    df_resid = n - kk
    s2 = ssr / df_resid
    slopes = beta[1:]
    ww = np.ones(n) if w is None else w
    centered = ybar - (ww @ ybar) / ww.sum()
    return {
        "beta": beta, "v": s2 * bread, "zbar": zbar, "resid": resid, "w": w, "df_resid": df_resid,
        "rmse": np.sqrt(s2), "r2b": 1 - ssr / (ww @ centered ** 2),
        "r2w": wcorr((x - xbar[codes]) @ slopes, y - ybar[codes]) ** 2,
        "r2o": wcorr(x @ slopes, y) ** 2, "n": n, "sizes": sizes, "nobs": len(y), "ssr": ssr,
    }


@pytest.mark.parametrize("name", ["balanced", "gappy"])
@pytest.mark.parametrize("weighted", [False, True])
def test_be(request, name, weighted):
    frame = request.getfixturevalue(name)
    result = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="be", wls=weighted)
    o = be_oracle(frame, COLS, weighted)
    terms = ["Intercept", *COLS]
    assert result.inference["use_t"] is True
    check_inference(result, terms, o["beta"], o["v"], df=o["df_resid"])
    m = result.metrics
    assert_allclose([m["r_squared_between"], m["r_squared_within"], m["r_squared_overall"]],
                    [o["r2b"], o["r2w"], o["r2o"]], rtol=1e-8)
    assert_allclose(m["rmse"], o["rmse"], rtol=1e-10)
    assert m["df_resid"] == o["df_resid"] and m["n_groups"] == o["n"]
    assert result.nobs == o["nobs"]
    stat, q, df2 = wald_f(o["beta"][1:], o["v"][1:, 1:], o["df_resid"])
    check_f(result.tests["model"], stat, q, df2)
    assert_allclose(result.extra["ssr"], o["ssr"], rtol=1e-10)
    assert result.extra["weighted_by_panel_length"] is weighted


def test_be_robust_hc1_and_cluster(gappy):
    o = be_oracle(gappy, COLS)
    terms = ["Intercept", *COLS]
    robust = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="be", covariance="robust")
    v = hc1(o["zbar"], o["resid"], None, o["n"], len(COLS) + 1)
    check_inference(robust, terms, o["beta"], v, df=o["df_resid"])
    sm_fit = sm.OLS(o["zbar"] @ o["beta"] + o["resid"], o["zbar"]).fit(cov_type="HC1")
    assert_allclose(block(robust, terms), sm_fit.cov_params(), rtol=1e-9)
    ow = be_oracle(gappy, COLS, True)
    weighted = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="be", wls=True,
                        covariance="robust")
    vw = hc1(ow["zbar"], ow["resid"], ow["w"], ow["n"], len(COLS) + 1)
    check_inference(weighted, terms, ow["beta"], vw, df=ow["df_resid"])
    sm_w = sm.WLS(ow["zbar"] @ ow["beta"] + ow["resid"], ow["zbar"], weights=ow["w"]).fit(
        cov_type="HC1")
    assert_allclose(block(weighted, terms), sm_w.cov_params(), rtol=1e-9)
    clustered = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="be", covariance="cluster",
                         cluster="cl")
    panel_cluster = gappy.groupby("id", sort=False).cl.first().to_numpy()
    codes = pd.factorize(panel_cluster)[0]
    v2, groups = cr1(o["zbar"], o["resid"], None, codes, o["n"], len(COLS) + 1)
    check_inference(clustered, terms, o["beta"], v2, df=groups - 1)
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="be", covariance="cluster",
                 cluster="cl2")
    assert err.value.code == "cluster_varies_within_panel"
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="be", weights="wa",
                 weight_type="aweight")
    assert err.value.code == "unsupported_weights"


def test_be_tiny_sample_and_insufficient():
    frame = make_data(seed=13, n=4, periods=3, unbalanced=False)
    result = oe.xtreg(data=frame, y="y", x=VARY, panel="id", model="be")
    o = be_oracle(frame, VARY)
    assert o["df_resid"] == 1
    check_inference(result, ["Intercept", *VARY], o["beta"], o["v"], df=1)
    three = make_data(seed=13, n=3, periods=3, unbalanced=False)
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=three, y="y", x=VARY, panel="id", model="be")
    assert err.value.code == "insufficient_observations"


# ---- first differences -----------------------------------------------------------------


def fd_oracle(frame, cols, w=None, weight_type=None):
    frame = frame.sort_values(["id", "year"]).reset_index(drop=True)
    same = (frame.id.to_numpy()[1:] == frame.id.to_numpy()[:-1]) & (
        frame.year.to_numpy()[1:] - frame.year.to_numpy()[:-1] == 1)
    rows = np.flatnonzero(same) + 1
    y, x = frame.y.to_numpy(), frame[cols].to_numpy(float)
    dy, dx = y[rows] - y[rows - 1], x[rows] - x[rows - 1]
    design = np.column_stack([np.ones(len(rows)), dx])
    codes = pd.factorize(frame.id.to_numpy()[rows])[0]
    nobs = len(rows)
    if w is not None:
        w = w[rows]
        if weight_type == "fweight":
            nobs = int(w.sum())
        else:
            w = w * len(rows) / w.sum()
    beta, resid, bread, ssr = wls(design, dy, w)
    kk = design.shape[1]
    df_resid = nobs - kk
    ww = np.ones(len(rows)) if w is None else w
    centered = dy - (ww @ dy) / ww.sum()
    r2 = 1 - ssr / (ww @ centered ** 2)
    return {"beta": beta, "v": ssr / df_resid * bread, "design": design, "resid": resid, "w": w,
            "codes": codes, "nobs": nobs, "df_resid": df_resid, "r2": r2,
            "adj_r2": 1 - (nobs - 1) / (nobs - kk) * (1 - r2), "rmse": np.sqrt(ssr / df_resid),
            "dropped": len(frame) - len(rows), "n": codes.max() + 1,
            "sizes": np.bincount(codes, weights=ww)}


@pytest.mark.parametrize("name", ["balanced", "unbalanced", "gappy"])
def test_fd(request, name):
    frame = request.getfixturevalue(name)
    result = oe.xtreg(data=frame, y="y", x=COLS, panel="id", time="year", model="fd")
    o = fd_oracle(frame, VARY)
    assert "z" in result.provenance["omitted_terms"]       # D.z = 0
    terms = ["Intercept", *VARY]
    assert result.inference["use_t"] is True
    check_inference(result, terms, o["beta"], o["v"], df=o["df_resid"])
    assert result.nobs == o["nobs"]
    assert result.extra["dropped_for_differencing"] == o["dropped"]
    assert any("consecutive" in w for w in result.warnings)
    m = result.metrics
    assert_allclose([m["r_squared"], m["adjusted_r_squared"], m["rmse"]],
                    [o["r2"], o["adj_r2"], o["rmse"]], rtol=1e-9)
    assert m["df_resid"] == o["df_resid"] and m["df_model"] == 2 and m["n_groups"] == o["n"]
    assert_allclose([m["t_min"], m["t_max"]], [o["sizes"].min(), o["sizes"].max()])
    stat, q, df2 = wald_f(o["beta"][1:], o["v"][1:, 1:], o["df_resid"])
    check_f(result.tests["model"], stat, q, df2)
    # Explicit check against statsmodels on the differenced sample.
    sm_fit = sm.OLS(o["design"] @ o["beta"] + o["resid"], o["design"]).fit()
    assert_allclose(block(result, terms), sm_fit.cov_params(), rtol=1e-9)


def test_fd_robust_cluster_and_weights(gappy):
    o = fd_oracle(gappy, VARY)
    terms = ["Intercept", *VARY]
    robust = oe.xtreg(data=gappy, y="y", x=VARY, panel="id", time="year", model="fd",
                      covariance="robust")
    v, groups = cr1(o["design"], o["resid"], None, o["codes"], o["nobs"], 3)
    check_inference(robust, terms, o["beta"], v, df=groups - 1)
    stat, q, df2 = wald_f(o["beta"][1:], v[1:, 1:], groups - 1)
    check_f(robust.tests["model"], stat, q, df2)
    # Row-varying weights are legal for fd (regress D.y D.x [aw=w]).
    wrow = gappy.sort_values(["id", "year"]).wrow.to_numpy()
    oa = fd_oracle(gappy, VARY, wrow, "aweight")
    a = oe.xtreg(data=gappy, y="y", x=VARY, panel="id", time="year", model="fd", weights="wrow",
                 weight_type="aweight")
    check_inference(a, terms, oa["beta"], oa["v"], df=oa["df_resid"])
    assert_allclose(a.metrics["rmse"], oa["rmse"], rtol=1e-10)
    wf = gappy.sort_values(["id", "year"]).wf.to_numpy()
    of = fd_oracle(gappy, VARY, wf, "fweight")
    f = oe.xtreg(data=gappy, y="y", x=VARY, panel="id", time="year", model="fd", weights="wf",
                 weight_type="fweight")
    assert f.nobs == of["nobs"]
    check_inference(f, terms, of["beta"], of["v"], df=of["df_resid"])
    fr = oe.xtreg(data=gappy, y="y", x=VARY, panel="id", time="year", model="fd", weights="wf",
                  weight_type="fweight", covariance="robust")
    vf, groups = cr1(of["design"], of["resid"], of["w"], of["codes"], of["nobs"], 3)
    check_inference(fr, terms, of["beta"], vf, df=groups - 1)
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=gappy, y="y", x=VARY, panel="id", model="fd")
    assert err.value.code == "invalid_spec"
    single = gappy.drop_duplicates("id")
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=single, y="y", x=VARY, panel="id", time="year", model="fd")
    assert err.value.code == "empty_sample"


# ---- pooled OLS ------------------------------------------------------------------------


def test_pooled_matches_statsmodels_and_panel_cluster(unbalanced):
    frame = unbalanced
    x = np.column_stack([np.ones(len(frame)), frame[COLS].to_numpy()])
    y = frame.y.to_numpy()
    terms = ["Intercept", *COLS]
    plain = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="pooled")
    fit = sm.OLS(y, x).fit()
    check_inference(plain, terms, fit.params, fit.cov_params(), df=fit.df_resid)
    assert_allclose([plain.metrics["r_squared"], plain.metrics["adjusted_r_squared"]],
                    [fit.rsquared, fit.rsquared_adj], rtol=1e-10)
    assert_allclose(plain.metrics["rmse"], np.sqrt(fit.mse_resid), rtol=1e-10)
    assert plain.metrics["n_groups"] == frame.id.nunique()
    check_f(plain.tests["model"], fit.fvalue, len(COLS), fit.df_resid)
    codes, _ = codes_of(frame)
    robust = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="pooled", covariance="robust")
    cl = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": codes}, use_t=True)
    check_inference(robust, terms, cl.params, cl.cov_params(), df=cl.df_resid_inference)
    assert_allclose([coef(robust)[t].p_value for t in terms], cl.pvalues, rtol=1e-6)
    # Row-varying weights are fine for pooled OLS.
    w = frame.wrow.to_numpy()
    a = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="pooled", weights="wrow",
                 weight_type="aweight")
    ws = sm.WLS(y, x, weights=w * len(w) / w.sum()).fit()
    check_inference(a, terms, ws.params, ws.cov_params(), df=ws.df_resid)
    f = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="pooled", weights="wf",
                 weight_type="fweight", covariance="robust")
    expanded = frame.loc[frame.index.repeat(frame.wf.astype(int))].reset_index(drop=True)
    e = oe.xtreg(data=expanded, y="y", x=COLS, panel="id", model="pooled", covariance="robust")
    assert_allclose(f.covariance_matrix, e.covariance_matrix, rtol=1e-9)
    assert f.nobs == e.nobs


def test_pooled_driscoll_kraay(balanced, gappy):
    terms = ["Intercept", *COLS]
    for frame, lags in ((balanced, 3), (gappy, 2)):
        result = oe.xtreg(data=frame, y="y", x=COLS, panel="id", time="year", model="pooled",
                          covariance="driscoll_kraay", lags=lags, kernel="bartlett")
        x = np.column_stack([np.ones(len(frame)), frame[COLS].to_numpy()])
        beta, resid, _, _ = wls(x, frame.y.to_numpy())
        v, big_t = driscoll_kraay(x, resid, None, frame.year.to_numpy(), lags, len(frame),
                                  x.shape[1])
        check_inference(result, terms, beta, v, df=big_t - 1)
        assert_allclose(result.inference["small_sample_correction"],
                        big_t / (big_t - 1) * (len(frame) - 1) / (len(frame) - 4), rtol=1e-12)
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=balanced, y="y", x=COLS, panel="id", time="year", model="re",
                 covariance="driscoll_kraay")
    assert err.value.code == "unsupported_covariance"
    # Row-varying aweights: scores w_i e_i x_i with the weights rescaled to sum N.
    w = gappy.wrow.to_numpy()
    w = w * len(w) / w.sum()
    x = np.column_stack([np.ones(len(gappy)), gappy[COLS].to_numpy()])
    beta, resid, _, _ = wls(x, gappy.y.to_numpy(), w)
    v, big_t = driscoll_kraay(x, resid, w, gappy.year.to_numpy(), 2, len(gappy), x.shape[1])
    weighted = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model="pooled",
                        covariance="driscoll_kraay", lags=2, weights="wrow",
                        weight_type="aweight")
    check_inference(weighted, terms, beta, v, df=big_t - 1)


def test_driscoll_kraay_with_frequency_and_sampling_weights(gappy):
    """An OpenEcon extension (Stata's own vce(dkraay) refuses fweights and pweights).

    fweights replicate rows: scores f_i x_i e_i summed within periods and N = sum f in the
    factor T/(T-1) (N-1)/(N-K); pweights use the same scores as aweights (rescaled to N).
    """
    terms = ["Intercept", *COLS]
    year = gappy.year.to_numpy()
    x = np.column_stack([np.ones(len(gappy)), gappy[COLS].to_numpy()])
    y = gappy.y.to_numpy()
    f = gappy.wf.to_numpy()
    beta, resid, _, _ = wls(x, y, f)
    v, big_t = driscoll_kraay(x, resid, f, year, 2, int(f.sum()), x.shape[1])
    result = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model="pooled",
                      covariance="driscoll_kraay", lags=2, weights="wf", weight_type="fweight")
    assert result.nobs == int(f.sum())
    check_inference(result, terms, beta, v, df=big_t - 1)
    w = gappy.wa.to_numpy() * len(gappy) / gappy.wa.sum()
    beta, resid, _, _ = wls(x, y, w)
    v, big_t = driscoll_kraay(x, resid, w, year, 2, len(gappy), x.shape[1])
    for weight_type in ("pweight", "aweight"):
        result = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model="pooled",
                          covariance="driscoll_kraay", lags=2, weights="wa",
                          weight_type=weight_type)
        check_inference(result, terms, beta, v, df=big_t - 1)
    # fe: the same scores on Stata's transformed regression, weights constant within panel.
    o = fe_oracle(gappy, VARY, gappy.wf.to_numpy())
    f_fe = gappy.wf.to_numpy()
    xs, ys = o["xs"], o["xs"] @ o["beta"] + o["resid"]
    beta, resid, _, _ = wls(xs, ys, f_fe)
    v, big_t = driscoll_kraay(xs, resid, f_fe, year, 1, int(f_fe.sum()), 3)
    fe = oe.xtreg(data=gappy, y="y", x=VARY, panel="id", time="year",
                  covariance="driscoll_kraay", lags=1, weights="wf", weight_type="fweight")
    check_inference(fe, ["Intercept", *VARY], beta, v, df=big_t - 1, rtol=1e-6)


# ---- random-effects maximum likelihood --------------------------------------------------


def re_loglik(theta, x, y, codes, n):
    """Gaussian random-effects log likelihood with dense per-panel covariance matrices."""
    k = x.shape[1]
    s2u, s2e = np.exp(2 * theta[k]), np.exp(2 * theta[k + 1])
    r = y - x @ theta[:k]
    total = 0.0
    for i in range(n):
        ri = r[codes == i]
        v = s2e * np.eye(len(ri)) + s2u * np.ones((len(ri), len(ri)))
        total += stats.multivariate_normal.logpdf(ri, mean=np.zeros(len(ri)), cov=v)
    return total


def re_hessian(theta, x, y, codes, n):
    """Hessian of the dense-matrix likelihood from the matrix-calculus formulas.

    ll_i = -1/2 ln|V_i| - 1/2 r_i' V_i^-1 r_i with V_i = e^{2b} I + e^{2a} 11', so that
    d ll/d beta = X' V^-1 r, d ll/d phi = -1/2 tr(V^-1 V_phi) + 1/2 r' V^-1 V_phi V^-1 r and
    the second derivatives follow by differentiating V^-1 once more.
    """
    k = x.shape[1]
    s2u, s2e = np.exp(2 * theta[k]), np.exp(2 * theta[k + 1])
    r = y - x @ theta[:k]
    h = np.zeros((k + 2, k + 2))
    for i in range(n):
        rows = codes == i
        xi, ri = x[rows], r[rows]
        t = len(ri)
        ones, eye = np.ones((t, t)), np.eye(t)
        vinv = np.linalg.inv(s2e * eye + s2u * ones)
        first = {k: 2 * s2u * ones, k + 1: 2 * s2e * eye}
        second = {(k, k): 4 * s2u * ones, (k + 1, k + 1): 4 * s2e * eye,
                  (k, k + 1): np.zeros((t, t)), (k + 1, k): np.zeros((t, t))}
        h[:k, :k] -= xi.T @ vinv @ xi
        for p in (k, k + 1):
            cross = -xi.T @ vinv @ first[p] @ vinv @ ri
            h[:k, p] += cross
            h[p, :k] += cross
            for q in (k, k + 1):
                a, b, c = first[p], first[q], second[(p, q)]
                term = 0.5 * np.trace(vinv @ b @ vinv @ a) - 0.5 * np.trace(vinv @ c)
                inner = vinv @ c @ vinv - vinv @ b @ vinv @ a @ vinv - vinv @ a @ vinv @ b @ vinv
                h[p, q] += term + 0.5 * ri @ inner @ ri
    return h


def maximize_loglik(x, y, codes, n, start):
    objective = lambda t: -re_loglik(t, x, y, codes, n)       # noqa: E731
    fit = minimize(objective, start, method="BFGS", options={"gtol": 1e-10, "maxiter": 500})
    # Newton polish with numerical derivatives.
    theta = fit.x
    for _ in range(3):
        g = approx_fprime(theta, objective, centered=True)
        h = approx_hess(theta, objective)
        theta = theta - np.linalg.solve(h, g)
    return theta, -objective(theta)


@pytest.mark.parametrize("name", ["balanced", "gappy"])
def test_mle_against_brute_force_likelihood(request, name):
    frame = request.getfixturevalue(name)
    cols = COLS
    result = oe.xtreg(data=frame, y="y", x=cols, panel="id", model="mle")
    codes, n = codes_of(frame)
    x = np.column_stack([np.ones(len(frame)), frame[cols].to_numpy()])
    y = frame.y.to_numpy()
    kk = x.shape[1]
    table = coef(result)
    terms = ["Intercept", *cols]
    theta_hat = np.array([table[t].estimate for t in terms] + [np.log(table["/sigma_u"].estimate),
                                                                np.log(table["/sigma_e"].estimate)])
    # (1) The reported log likelihood is the likelihood at the reported estimates.
    ll_hat = re_loglik(theta_hat, x, y, codes, n)
    assert_allclose(result.metrics["log_likelihood"], ll_hat, rtol=1e-10)
    # (2) It is a stationary point and the brute-force maximum is no higher.
    g = approx_fprime(theta_hat, lambda t: re_loglik(t, x, y, codes, n), centered=True)
    assert np.abs(g).max() < 1e-5
    theta_opt, ll_opt = maximize_loglik(x, y, codes, n, theta_hat + 0.05)
    assert ll_opt <= ll_hat + 1e-7
    assert_allclose(theta_hat, theta_opt, rtol=1e-6, atol=1e-7)
    # (3) OIM covariance: inverse negative Hessian (analytic dense-matrix form, cross-checked
    #     against central differences), delta method for sigma.
    h = re_hessian(theta_hat, x, y, codes, n)
    numerical = approx_hess(theta_hat, lambda t: re_loglik(t, x, y, codes, n))
    assert_allclose(h, numerical, rtol=1e-3, atol=1e-3 * np.abs(h).max())
    v_log = np.linalg.inv(-h)
    jac = np.ones(kk + 2)
    jac[kk], jac[kk + 1] = np.exp(theta_hat[kk]), np.exp(theta_hat[kk + 1])
    v = v_log * jac[:, None] * jac[None, :]
    all_terms = [*terms, "/sigma_u", "/sigma_e"]
    se = np.sqrt(np.diag(v))
    assert_allclose([table[t].std_error for t in all_terms], se, rtol=1e-7)
    assert_allclose(np.array(result.covariance_matrix), v, rtol=1e-7, atol=1e-14)
    assert result.inference["use_t"] is False
    for t in terms:
        assert table[t].equation == "y"
    assert table["/sigma_u"].equation is None and table["/sigma_e"].equation is None
    # (4) Metrics and tests.
    su, se_ = table["/sigma_u"].estimate, table["/sigma_e"].estimate
    assert_allclose([result.metrics["sigma_u"], result.metrics["sigma_e"], result.metrics["rho"]],
                    [su, se_, su ** 2 / (su ** 2 + se_ ** 2)], rtol=1e-12)
    assert_allclose(result.metrics["aic"], -2 * ll_hat + 2 * (kk + 2), rtol=1e-10)
    assert_allclose(result.metrics["bic"], -2 * ll_hat + np.log(len(y)) * (kk + 2), rtol=1e-10)
    ones = np.ones((len(y), 1))
    _, ll_null = maximize_loglik(ones, y, codes, n, np.array([y.mean(), theta_hat[kk],
                                                              theta_hat[kk + 1]]))
    lr = 2 * (ll_hat - ll_null)
    model = result.tests["model"]
    assert model["df"] == kk - 1 and model["distribution"] == "chi2"
    assert_allclose(model["statistic"], lr, rtol=1e-6)
    assert_allclose(model["p_value"], stats.chi2.sf(lr, kk - 1), rtol=1e-5, atol=1e-14)
    ssr = wls(x, y)[3]
    ll_ols = -0.5 * len(y) * (np.log(2 * np.pi) + np.log(ssr / len(y)) + 1)
    sig = result.tests["sigma_u"]
    assert_allclose(sig["statistic"], 2 * (ll_hat - ll_ols), rtol=1e-8)
    assert_allclose(sig["p_value"], 0.5 * stats.chi2.sf(2 * (ll_hat - ll_ols), 1), rtol=1e-6,
                    atol=1e-14)
    assert sig["df"] == 1 and "halved" in sig["label"]
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=frame, y="y", x=cols, panel="id", model="mle", covariance="robust")
    assert err.value.code == "unsupported_covariance"
    with pytest.raises(AnalysisError) as err:
        oe.xtreg(data=frame, y="y", x=cols, panel="id", model="mle", weights="wa",
                 weight_type="aweight")
    assert err.value.code == "unsupported_weights"


def test_mle_without_panel_variance_is_not_a_crash():
    """sigma_u -> 0: Stata's xtreg, mle reports sigma_u ~ 0 with chibar2 = 0, p = 1.

    The family may instead refuse with a documented boundary_solution error; anything
    else (nonconvergence, numerical_failure, a generic covariance error) is a defect.
    """
    frame = make_data(seed=8, n=20, periods=5, unbalanced=False)
    rng = np.random.default_rng(2)
    frame["y"] = 1 + 0.5 * frame.x1 + rng.normal(size=len(frame))
    try:
        result = oe.xtreg(data=frame, y="y", x=["x1"], panel="id", model="mle")
    except AnalysisError as err:
        assert err.code == "boundary_solution", (err.code, str(err))
        assert "sigma_u" in str(err)
    else:
        assert result.metrics["sigma_u"] < 1e-2 * result.metrics["sigma_e"]
        assert result.tests["sigma_u"]["p_value"] > 0.4


def test_mle_sigma_ci_follows_stata_log_transform(balanced):
    """Stata's xtreg, mle displays the /sigma CIs as exp(ln sigma -/+ z se_ln) (asymmetric).

    With delta-method SE = sigma * se_ln the lower bound must be sigma * exp(-z se_ln),
    not the symmetric sigma - z * sigma * se_ln.
    """
    result = oe.xtreg(data=balanced, y="y", x=COLS, panel="id", model="mle")
    z = stats.norm.ppf(0.975)
    # The rule itself, on the rows printed in the [XT] xtreg manual's xtreg, mle example:
    # /sigma_u .2485556 (.0035017) [.2417863, .2555144], /sigma_e .2918458 (.001352)
    # [.289208, .2945076]. The symmetric interval would be [.2416924, .2554188].
    for est, se, low, high in ((0.2485556, 0.0035017, 0.2417863, 0.2555144),
                               (0.2918458, 0.001352, 0.289208, 0.2945076)):
        assert_allclose([est * np.exp(-z * se / est), est * np.exp(z * se / est)], [low, high],
                        rtol=2e-6)
        assert abs(est - z * se - low) > 5e-6
    for term in ("/sigma_u", "/sigma_e"):
        c = coef(result)[term]
        se_ln = c.std_error / c.estimate
        assert_allclose([c.ci_low, c.ci_high],
                        [c.estimate * np.exp(-z * se_ln), c.estimate * np.exp(z * se_ln)],
                        rtol=1e-8, err_msg=term)
        assert c.ci_high - c.estimate > c.estimate - c.ci_low > 0      # asymmetric, positive
        # The ln-scale estimate and standard error are reported too.
        ln = result.extra["ln_sigma"][term[1:]]
        assert_allclose([ln["estimate"], ln["std_error"]], [np.log(c.estimate), se_ln], rtol=1e-10)
    # Regression coefficients keep the symmetric normal interval, at any alpha.
    wide = oe.xtreg(data=balanced, y="y", x=COLS, panel="id", model="mle", alpha=0.10)
    z90 = stats.norm.ppf(0.95)
    c = coef(wide)["x1"]
    assert_allclose([c.ci_low, c.ci_high],
                    [c.estimate - z90 * c.std_error, c.estimate + z90 * c.std_error], rtol=1e-9)
    s = coef(wide)["/sigma_u"]
    assert_allclose(s.ci_low, s.estimate * np.exp(-z90 * s.std_error / s.estimate), rtol=1e-8)
    again = type(wide).model_validate_json(wide.model_dump_json())
    assert coef(again)["/sigma_u"].ci_low == s.ci_low


# ---- Hausman ----------------------------------------------------------------------------


def hausman_oracle(consistent, efficient, terms, scale_c=1.0, scale_e=1.0):
    bc = np.array([coef(consistent)[t].estimate for t in terms])
    be = np.array([coef(efficient)[t].estimate for t in terms])
    v = block(consistent, terms) * scale_c - block(efficient, terms) * scale_e
    v = (v + v.T) / 2
    d = bc - be
    values, vectors = np.linalg.eigh(v)
    keep = np.abs(values) > 1e-12 * np.abs(values).max()
    stat = float(((vectors[:, keep].T @ d) ** 2 / values[keep]).sum())
    return stat, int(keep.sum()), d, v


def test_hausman_fe_vs_re(unbalanced, balanced):
    for frame in (unbalanced, balanced):
        fe = oe.xtreg(data=frame, y="y", x=COLS, panel="id")
        re = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="re")
        out = oe.hausman(fe, re)
        stat, rank, d, v = hausman_oracle(fe, re, VARY)
        assert out["terms"] == VARY and out["df"] == rank == 2
        assert_allclose(out["statistic"], stat, rtol=1e-9)
        assert_allclose(out["difference"], d, rtol=1e-12)
        if stat >= 0 and not out["negative_definite"]:
            assert_allclose(out["statistic"], float(d @ np.linalg.solve(v, d)), rtol=1e-9)
            assert_allclose(out["p_value"], stats.chi2.sf(stat, rank), rtol=1e-8)
            assert out["note"] is None
        # Without sigmamore V_c - V_e may be indefinite (as in Stata): then no p-value.
        assert out["reject"] == (None if out["p_value"] is None else out["p_value"] < 0.05)
        se_d = [np.sqrt(x) if x > 0 else None for x in np.diag(v)]
        for got, want in zip(out["se_difference"], se_d, strict=True):
            assert (got is None) == (want is None)
            if want is not None:
                assert_allclose(got, want, rtol=1e-9)


def test_hausman_sigmamore_and_sigmaless(gappy):
    fe = oe.xtreg(data=gappy, y="y", x=COLS, panel="id")
    re = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="re")
    fd = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model="fd")
    # help hausman: e(sigma_e) is read after xtreg fe/mle, e(rmse) after xtreg re. The two
    # differ (rmse^2 = RSS*/(N-K) of the GLS regression), so sigmamore is NOT a no-op.
    s2_fe, s2_re = fe.metrics["sigma_e"] ** 2, re.metrics["rmse"] ** 2
    assert abs(s2_re / s2_fe - 1) > 1e-3
    plain = oe.hausman(fe, re)
    assert plain["sigma"] is None and plain["sigma2"] is None
    more = oe.hausman(fe, re, sigmamore=True)
    assert more["sigma"] == "sigmamore"
    assert_allclose([more["sigma2"]["consistent"], more["sigma2"]["efficient"]], [s2_fe, s2_re],
                    rtol=1e-12)
    stat, rank, _, v = hausman_oracle(fe, re, VARY, scale_c=s2_re / s2_fe)
    assert_allclose(more["statistic"], stat, rtol=1e-9)
    assert abs(more["statistic"] / plain["statistic"] - 1) > 1e-3
    # Both covariances on the efficient sigma^2: V_c - V_e is positive definite for fe vs re.
    assert np.linalg.eigvalsh(v).min() > 0 and more["negative_definite"] is False
    assert more["df"] == rank == 2 and more["p_value"] is not None
    less = oe.hausman(fe, re, sigmaless=True)
    stat, rank, _, v = hausman_oracle(fe, re, VARY, scale_e=s2_fe / s2_re)
    assert_allclose(less["statistic"], stat, rtol=1e-9) and less["sigma"] == "sigmaless"
    assert np.linalg.eigvalsh(v).min() > 0
    # sigmaless is sigmamore times s2_re / s2_fe (the same matrix on the other variance).
    assert_allclose(less["statistic"], more["statistic"] * s2_re / s2_fe, rtol=1e-9)
    # mle stores e(sigma_e) as fe does.
    mle = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", model="mle")
    assert_allclose(oe.hausman(fe, mle, sigmamore=True)["sigma2"]["efficient"],
                    mle.metrics["sigma_e"] ** 2, rtol=1e-12)
    # Every other estimator stores e(rmse): first differences (regress D.) against fe.
    s2_fd = fd.metrics["rmse"] ** 2
    more = oe.hausman(fd, fe, sigmamore=True)
    stat, rank, _, _ = hausman_oracle(fd, fe, VARY, scale_c=s2_fe / s2_fd)
    assert_allclose(more["statistic"], stat, rtol=1e-9) and more["sigma"] == "sigmamore"
    less = oe.hausman(fd, fe, sigmaless=True)
    stat, rank, _, _ = hausman_oracle(fd, fe, VARY, scale_e=s2_fd / s2_fe)
    assert_allclose(less["statistic"], stat, rtol=1e-9)
    with pytest.raises(AnalysisError):
        oe.hausman(fe, re, sigmamore=True, sigmaless=True)
    with pytest.raises(AnalysisError) as err:
        oe.hausman(fe, "re")
    assert err.value.code == "invalid_spec"


def test_hausman_requires_conventional_covariances(unbalanced):
    """Stata: 'hausman cannot be used with vce(robust), vce(cluster cvar), or p-weighted data'.

    With a robust V_c the difference V_c - V_e is not the variance of b_c - b_e, so the
    statistic is not a chi2; it used to be computed silently. force=True (Stata's force
    option) computes it regardless and says so.
    """
    fe = oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", time="year")
    re = oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", model="re")
    ordinary = oe.hausman(fe, re, sigmamore=True)
    assert ordinary["forced"] is False and ordinary["negative_definite"] is False
    robust_fits = [
        oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", covariance="robust"),
        oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", covariance="cluster", cluster="cl"),
        oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", time="year",
                 covariance="driscoll_kraay"),
        oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", weights="wa", weight_type="pweight",
                 covariance="robust"),
    ]
    for fit in robust_fits:
        with pytest.raises(AnalysisError) as err:
            oe.hausman(fit, re)
        assert err.value.code == "unsupported_covariance"
        assert "consistent" in str(err.value) and "force=True" in str(err.value)
    robust_re = oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", model="re",
                         covariance="robust")
    with pytest.raises(AnalysisError) as err:
        oe.hausman(fe, robust_re, sigmamore=True)
    assert err.value.code == "unsupported_covariance" and "efficient" in str(err.value)
    forced = oe.hausman(robust_fits[0], re, force=True)
    stat, rank, _, _ = hausman_oracle(robust_fits[0], re, VARY)
    assert forced["forced"] is True and "forced" in forced["note"]
    assert_allclose(forced["statistic"], stat, rtol=1e-9)
    # Analytic and frequency weights with the conventional covariance are fine (fe only).
    weighted = oe.xtreg(data=unbalanced, y="y", x=COLS, panel="id", weights="wa",
                        weight_type="aweight")
    assert oe.hausman(weighted, re)["forced"] is False


def test_hausman_negative_statistic_is_reported_with_note():
    """Construct V_c - V_e indefinite: consistent = re (small V), efficient = fe (large V)."""
    frame = make_data(seed=32, n=12, periods=4, unbalanced=False)
    fe = oe.xtreg(data=frame, y="y", x=VARY, panel="id")
    re = oe.xtreg(data=frame, y="y", x=VARY, panel="id", model="re")
    assert re.metrics["sigma_u"] > 0
    out = oe.hausman(re, fe)
    stat, rank, _, v = hausman_oracle(re, fe, VARY)
    assert np.linalg.eigvalsh(v).min() < 0
    assert out["negative_definite"] is True and out["note"] is not None
    assert_allclose(out["statistic"], stat, rtol=1e-9)
    if stat < 0:
        assert out["p_value"] is None and out["reject"] is None


# ---- Fama-MacBeth ------------------------------------------------------------------------


def fmb_oracle(frame, cols, lags=None):
    frame = frame.sort_values(["year", "id"])
    periods = np.sort(frame.year.unique())
    estimates, r2 = [], []
    for period in periods:
        chunk = frame[frame.year == period]
        x = np.column_stack([np.ones(len(chunk)), chunk[cols].to_numpy()])
        beta, resid, _, ssr = wls(x, chunk.y.to_numpy())
        estimates.append(beta)
        r2.append(1 - ssr / ((chunk.y - chunk.y.mean()) ** 2).sum())
    b = np.array(estimates)
    big_t = len(periods)
    mean = b.mean(0)
    dev = b - mean
    if lags is None:
        v = dev.T @ dev / (big_t * (big_t - 1))
    else:
        s = dev.T @ dev
        for lag in range(1, lags + 1):
            omega = dev[lag:].T @ dev[:-lag]
            s += (1 - lag / (lags + 1)) * (omega + omega.T)
        v = s * (big_t / (big_t - 1)) / big_t ** 2
    return mean, v, big_t, float(np.mean(r2))


def test_xtfmb(balanced, gappy):
    terms = ["Intercept", *COLS]
    for frame in (balanced, gappy):
        result = oe.xtfmb(data=frame, y="y", x=COLS, panel="id", time="year")
        mean, v, big_t, r2 = fmb_oracle(frame, COLS)
        check_inference(result, terms, mean, v, df=big_t - 1)
        assert_allclose(result.metrics["r_squared"], r2, rtol=1e-10)
        assert result.metrics["n_periods"] == big_t
        stat, q, df2 = wald_f(mean[1:], v[1:, 1:], big_t - 1)
        check_f(result.tests["model"], stat, q, df2)
        for lags in (0, 2):
            hac = oe.xtfmb(data=frame, y="y", x=COLS, panel="id", time="year", covariance="hac",
                           lags=lags)
            mean, v, big_t, _ = fmb_oracle(frame, COLS, lags)
            check_inference(hac, terms, mean, v, df=big_t - 1)
        # Newey-West with 0 lags is the plain estimator; the statsmodels check of the same.
        zero = oe.xtfmb(data=frame, y="y", x=COLS, panel="id", time="year", covariance="hac",
                        lags=0)
        assert_allclose(zero.covariance_matrix, result.covariance_matrix, rtol=1e-12)
    with pytest.raises(AnalysisError) as err:
        oe.xtfmb(data=balanced, y="y", x=COLS, panel="id", time="year", lags=2)
    assert err.value.code == "invalid_spec"


def test_xtfmb_newey_west_matches_statsmodels(balanced):
    """Regress the period estimates on a constant with Newey-West: identical covariance."""
    frame = balanced
    periods = np.sort(frame.year.unique())
    b = np.array([wls(np.column_stack([np.ones((frame.year == p).sum()),
                                       frame.loc[frame.year == p, COLS].to_numpy()]),
                      frame.loc[frame.year == p, "y"].to_numpy())[0] for p in periods])
    hac = oe.xtfmb(data=frame, y="y", x=COLS, panel="id", time="year", covariance="hac", lags=2)
    for j, term in enumerate(["Intercept", *COLS]):
        fit = sm.OLS(b[:, j], np.ones(len(periods))).fit(cov_type="HAC",
                                                           cov_kwds={"maxlags": 2, "use_correction": True})
        assert_allclose(coef(hac)[term].std_error, fit.bse[0], rtol=1e-9, err_msg=term)
        assert_allclose(coef(hac)[term].estimate, fit.params[0], rtol=1e-12)


# ---- invariances across models -------------------------------------------------------------


@pytest.mark.parametrize("model", ["fe", "re", "be", "fd", "pooled", "mle"])
def test_row_permutation_invariance(gappy, model):
    shuffled = gappy.sample(frac=1, random_state=5).reset_index(drop=True)
    a = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model=model)
    b = oe.xtreg(data=shuffled, y="y", x=COLS, panel="id", time="year", model=model)
    assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients],
                    rtol=1e-9)
    assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=1e-8)
    for name in a.metrics:
        if a.metrics[name] is not None:
            assert_allclose(a.metrics[name], b.metrics[name], rtol=1e-8, err_msg=name)


@pytest.mark.parametrize("model", ["fe", "re", "be", "fd", "pooled"])
def test_rescaling_regressors_rescales_coefficients(gappy, model):
    scaled = gappy.copy()
    scaled["x1"] = scaled.x1 * 1e6
    scaled["y"] = scaled.y * 1e3 + 5e4
    a = oe.xtreg(data=gappy, y="y", x=VARY, panel="id", time="year", model=model)
    b = oe.xtreg(data=scaled, y="y", x=VARY, panel="id", time="year", model=model)
    ca, cb = coef(a), coef(b)
    assert_allclose(cb["x1"].estimate, ca["x1"].estimate * 1e3 / 1e6, rtol=1e-9)
    assert_allclose(cb["x2"].estimate, ca["x2"].estimate * 1e3, rtol=1e-9)
    assert_allclose(cb["x1"].statistic, ca["x1"].statistic, rtol=1e-9)
    assert_allclose(cb["x2"].statistic, ca["x2"].statistic, rtol=1e-9)
    assert_allclose(b.tests["model"]["statistic"], a.tests["model"]["statistic"], rtol=1e-9)
    for name in a.metrics:
        if name.startswith("r_squared") or name in {"rho", "corr_u_xb"}:
            assert_allclose(b.metrics[name], a.metrics[name], rtol=1e-8, err_msg=name)


def test_singleton_panels_pooled_cluster_equals_hc1(balanced):
    """Every panel a single row: G = N, so G/(G-1) (N-1)/(N-K) = N/(N-K) and CR1 is HC1."""
    frame = balanced[balanced.year == 2000].reset_index(drop=True)
    x = np.column_stack([np.ones(len(frame)), frame[COLS].to_numpy()])
    y = frame.y.to_numpy()
    result = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="pooled", covariance="robust")
    fit = sm.OLS(y, x).fit(cov_type="HC1")
    assert_allclose(np.array(result.covariance_matrix), fit.cov_params(), rtol=1e-10)


def test_cluster_on_panel_column_equals_robust(unbalanced):
    for model in ("fe", "re", "pooled"):
        robust = oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", model=model,
                          covariance="robust")
        clustered = oe.xtreg(data=unbalanced, y="y", x=VARY, panel="id", model=model,
                             covariance="cluster", cluster="id")
        assert_allclose(robust.covariance_matrix, clustered.covariance_matrix, rtol=1e-12)
        assert robust.inference["df_inference"] == clustered.inference["df_inference"]


def test_pooled_with_fe_cluster_equals_regress_cluster(unbalanced):
    """model='pooled', covariance='robust' is regress y x, vce(cluster id) exactly."""
    frame = unbalanced
    x = np.column_stack([np.ones(len(frame)), frame[COLS].to_numpy()])
    beta, resid, _, _ = wls(x, frame.y.to_numpy())
    codes, _ = codes_of(frame)
    v, groups = cr1(x, resid, None, codes, len(frame), x.shape[1])
    result = oe.xtreg(data=frame, y="y", x=COLS, panel="id", model="pooled", covariance="robust")
    check_inference(result, ["Intercept", *COLS], beta, v, df=groups - 1)


def test_json_round_trip_and_summary(gappy):
    for model in ("fe", "re", "mle"):
        result = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model=model)
        text = result.summary()
        assert "Intercept" in text
        from openecon.models import ResultBundle
        again = ResultBundle.model_validate_json(result.model_dump_json())
        assert again.metrics == result.metrics and again.tests == result.tests


# ---- regressions for the repaired defects ---------------------------------------------------


def test_degenerate_outcomes_raise_instead_of_noise_standard_errors(balanced):
    """A constant, panel-level or exactly fitted outcome used to give SEs of 1e-16 and p = 0."""
    frame = balanced
    degenerate = {
        "constant": frame.assign(y=3.0),
        "panel level": frame.assign(y=frame.id.astype(float)),
        "exact fit": frame.assign(y=2 * frame.x1 - frame.x2 + 1),
    }
    for name, data in degenerate.items():
        for model in ("fe", "re"):
            with pytest.raises(AnalysisError) as err:
                oe.xtreg(data=data, y="y", x=VARY, panel="id", model=model)
            assert err.value.code == "no_within_variation", (name, model, err.value.code)
            assert "'y'" in str(err.value)
        with pytest.raises(AnalysisError) as err:
            oe.xtreg(data=data, y="y", x=VARY, panel="id", covariance="robust")
        assert err.value.code == "no_within_variation", name
    # The OLS-type models name the problem as well (previously a generic invalid_covariance).
    for model, data, code in (
            ("pooled", degenerate["constant"], "constant_outcome"),
            ("fd", degenerate["constant"], "constant_outcome"),
            ("be", degenerate["constant"], "constant_outcome"),
            ("pooled", degenerate["exact fit"], "perfect_fit"),
            ("fd", degenerate["exact fit"], "perfect_fit"),
            ("be", degenerate["exact fit"], "perfect_fit")):
        with pytest.raises(AnalysisError) as err:
            oe.xtreg(data=data, y="y", x=VARY, panel="id", time="year", model=model)
        assert err.value.code == code, (model, err.value.code)
    # A panel-level outcome is a legitimate between / pooled regression.
    assert oe.xtreg(data=degenerate["panel level"], y="y", x=VARY, panel="id",
                    model="be").nobs == len(frame)
    # Weighted panel means of a constant are not exactly that constant: the criterion must
    # judge the within deviations against the rounding floor of the level of y, not against
    # a centered total that is rounding noise as well.
    for weight_type in ("aweight", "fweight"):
        with pytest.raises(AnalysisError) as err:
            oe.xtreg(data=frame.assign(y=0.1), y="y", x=VARY, panel="id",
                     weights="wa" if weight_type == "aweight" else "wf", weight_type=weight_type)
        assert err.value.code == "no_within_variation", weight_type
    # An exact fit of an outcome with a large level leaves residuals of about eps * level
    # (1e-6 here), far above any centered cutoff: the floor is 1e-28 * sum y^2.
    level = frame.assign(y=1e10 + 2 * frame.x1 - frame.x2)
    for model, code in (("fe", "no_within_variation"), ("re", "no_within_variation"),
                        ("mle", "no_within_variation"), ("pooled", "perfect_fit"),
                        ("fd", "perfect_fit"), ("be", "perfect_fit")):
        with pytest.raises(AnalysisError) as err:
            oe.xtreg(data=level, y="y", x=VARY, panel="id", time="year", model=model)
        assert err.value.code == code, (model, err.value.code)
    # A tiny but real signal on a large level is NOT refused: sd/level = 1e-9 >> 1e-14.
    rng = np.random.default_rng(12)
    small = frame.assign(y=1e6 + 1e-3 * (frame.x1 + rng.normal(size=len(frame))))
    for model in ("fe", "re", "pooled", "fd", "be", "mle"):
        fit = oe.xtreg(data=small, y="y", x=VARY, panel="id", time="year", model=model)
        assert_allclose(coef(fit)["x1"].estimate, 1e-3, rtol=0.6, err_msg=model)
        assert 1e-6 < coef(fit)["x1"].std_error < 1e-2, model
    # Tiny within variation next to large between variation is real too (ratio 1e-16 in
    # sums of squares, which a centered relative cutoff of 1e-13 would have refused).
    between = frame.assign(y=1e2 * frame.id + 1e-6 * (frame.x1 + rng.normal(size=len(frame))))
    fit = oe.xtreg(data=between, y="y", x=VARY, panel="id")
    assert_allclose(coef(fit)["x1"].estimate, 1e-6, rtol=0.6)


def test_xtfmb_degenerate_outcomes_and_offsets(balanced):
    for data, code in ((balanced.assign(y=3.0), "constant_outcome"),
                       (balanced.assign(y=balanced.year.astype(float)), "constant_outcome"),
                       (balanced.assign(y=2 * balanced.x1 + 1), "perfect_fit")):
        with pytest.raises(AnalysisError) as err:
            oe.xtfmb(data=data, y="y", x=VARY, panel="id", time="year")
        assert err.value.code == code
    base = oe.xtfmb(data=balanced, y="y", x=VARY, panel="id", time="year", covariance="hac",
                    lags=1)
    moved = oe.xtfmb(data=balanced.assign(x1=balanced.x1 + 1e8), y="y", x=VARY, panel="id",
                     time="year", covariance="hac", lags=1)
    b, m = coef(base), coef(moved)
    for term in VARY:
        assert_allclose(m[term].estimate, b[term].estimate, rtol=1e-6)
        assert_allclose(m[term].std_error, b[term].std_error, rtol=1e-5)
    assert_allclose(m["Intercept"].estimate, b["Intercept"].estimate - 1e8 * b["x1"].estimate,
                    rtol=1e-6)
    assert_allclose(moved.metrics["r_squared"], base.metrics["r_squared"], rtol=1e-6)
    assert_allclose(moved.extra["period_estimates_sd"][1:], base.extra["period_estimates_sd"][1:],
                    rtol=1e-5)


@pytest.mark.parametrize("model,options", [
    ("fe", {}), ("fe", {"covariance": "robust"}),
    ("fe", {"covariance": "driscoll_kraay", "lags": 1}),
    ("fe", {"covariance": "cluster", "cluster": "cl"}),
    ("re", {}), ("re", {"covariance": "robust"}), ("re", {"sa": True}),
    ("be", {}), ("be", {"covariance": "robust"}), ("be", {"wls": True}),
    ("fd", {}), ("fd", {"covariance": "robust"}),
    ("pooled", {}), ("pooled", {"covariance": "robust"}),
    ("pooled", {"covariance": "driscoll_kraay", "lags": 2}),
    ("pooled", {"covariance": "cluster", "cluster": ["cl", "cl2"]}),
    ("mle", {}),
])
def test_regressor_level_does_not_change_slopes_or_their_inference(model, options):
    """x1 + 1e8 has relative variation 1e-8, numerically collinear with the constant in
    uncentered terms. Every regression is run on centered slopes (Stata sweeps the constant
    first), so slopes, their covariance, tests and fit statistics equal those of x1 to the
    precision float64 leaves in x1 + 1e8 (about 1e-8), and only the Intercept moves.
    """
    frame = make_data(seed=5, n=60, periods=8, unbalanced=True, gaps=True)
    frame["cl"] = frame.id % 12
    frame["cl2"] = (frame.id * 7 + frame.year) % 9
    shifted = frame.assign(x1=frame.x1 + 1e8)
    base = oe.xtreg(data=frame, y="y", x=VARY, panel="id", time="year", model=model, **options)
    moved = oe.xtreg(data=shifted, y="y", x=VARY, panel="id", time="year", model=model,
                     **options)
    assert moved.provenance["omitted_terms"] == [] and moved.inference.get("psd_adjusted") in (
        None, False)
    b, m = coef(base), coef(moved)
    for term in VARY:
        assert_allclose(m[term].estimate, b[term].estimate, rtol=2e-6, err_msg=term)
        assert_allclose(m[term].std_error, b[term].std_error, rtol=2e-5, err_msg=term)
    assert_allclose(block(moved, VARY), block(base, VARY), rtol=1e-4, atol=1e-12)
    if model == "fd":            # the level differences out: the drift does not move either
        assert_allclose(m["Intercept"].estimate, b["Intercept"].estimate, rtol=1e-5, atol=1e-8)
        assert_allclose(m["Intercept"].std_error, b["Intercept"].std_error, rtol=1e-5)
    else:
        assert_allclose(m["Intercept"].estimate,
                        b["Intercept"].estimate - 1e8 * b["x1"].estimate, rtol=1e-6)
        # Var(Intercept) = Var(Intercept_c) + m'V m - 2 m'Cov: dominated by 1e16 * Var(b1).
        assert_allclose(m["Intercept"].std_error, 1e8 * b["x1"].std_error, rtol=1e-3)
    assert_allclose(moved.tests["model"]["statistic"], base.tests["model"]["statistic"],
                    rtol=1e-4)
    for name, value in base.metrics.items():
        if value is not None and name != "theta":
            assert_allclose(moved.metrics[name], value, rtol=1e-5, atol=1e-9, err_msg=name)
    if "fixed_effects" in base.tests:
        assert_allclose(moved.tests["fixed_effects"]["statistic"],
                        base.tests["fixed_effects"]["statistic"], rtol=1e-5)


def test_large_offset_regressor_is_handled_by_re_and_mle(balanced):
    """x1 + 1e5 has the same within variation as x1.

    re used to drop it silently from the variance-component within step (uncentered
    screen) and mle failed to converge on the unconditioned likelihood; both must now be
    shift invariant: same slopes and standard errors, Intercept moved by -1e5 * b1.
    """
    shifted = balanced.assign(x1=balanced.x1 + 1e5)
    for model in ("re", "mle", "fe"):
        base = oe.xtreg(data=balanced, y="y", x=VARY, panel="id", model=model)
        moved = oe.xtreg(data=shifted, y="y", x=VARY, panel="id", model=model)
        assert moved.provenance["omitted_terms"] == [], model
        b, m = coef(base), coef(moved)
        for term in VARY:
            assert_allclose(m[term].estimate, b[term].estimate, rtol=1e-7, err_msg=model)
            assert_allclose(m[term].std_error, b[term].std_error, rtol=1e-6, err_msg=model)
        assert_allclose(m["Intercept"].estimate,
                        b["Intercept"].estimate - 1e5 * b["x1"].estimate, rtol=1e-7)
        assert_allclose([moved.metrics["sigma_u"], moved.metrics["sigma_e"]],
                        [base.metrics["sigma_u"], base.metrics["sigma_e"]], rtol=1e-7)
        if model == "re":
            assert moved.extra["variance_components"]["within_regressors"] == 2
            assert_allclose(moved.metrics["rmse"], base.metrics["rmse"], rtol=1e-8)
        if model == "mle":
            assert moved.provenance["optimizer"]["converged"] is True
            assert moved.provenance["optimizer"]["iterations"] <= 10
            assert_allclose(moved.metrics["log_likelihood"], base.metrics["log_likelihood"],
                            rtol=1e-10)
            assert_allclose(moved.tests["model"]["statistic"], base.tests["model"]["statistic"],
                            rtol=1e-7)
    # The reviewer's reproduction: 30 x 6 panel, offsets 1e4 .. 1e7 (1e5 failed before).
    rng = np.random.default_rng(42)
    ids = np.repeat(np.arange(30), 6)
    u = rng.normal(size=30)[ids]
    x1, x2 = rng.normal(size=180) + 0.4 * u, rng.normal(size=180)
    frame = pd.DataFrame({"id": ids, "x1": x1, "x2": x2,
                          "y": 1 + 0.5 * x1 - 0.3 * x2 + u + rng.normal(size=180)})
    reference = coef(oe.xtreg(data=frame, y="y", x=VARY, panel="id", model="mle"))
    for offset in (1e4, 1e5, 1e6, 1e7):
        fit = coef(oe.xtreg(data=frame.assign(x1=frame.x1 + offset), y="y", x=VARY, panel="id",
                            model="mle"))
        assert_allclose([fit[t].estimate for t in (*VARY, "/sigma_u", "/sigma_e")],
                        [reference[t].estimate for t in (*VARY, "/sigma_u", "/sigma_e")],
                        rtol=1e-6, err_msg=str(offset))


def test_mle_is_invariant_to_units(gappy):
    """The internal centering/scaling is an exact reparameterization: rescaling the data
    rescales coefficients, sigmas and the log likelihood (Jacobian) exactly."""
    a = oe.xtreg(data=gappy, y="y", x=VARY, panel="id", model="mle")
    scaled = gappy.assign(x1=gappy.x1 * 1e6 - 3e8, y=gappy.y * 1e3 + 5e4)
    b = oe.xtreg(data=scaled, y="y", x=VARY, panel="id", model="mle")
    ca, cb = coef(a), coef(b)
    assert_allclose(cb["x1"].estimate, ca["x1"].estimate * 1e3 / 1e6, rtol=1e-7)
    assert_allclose(cb["x2"].estimate, ca["x2"].estimate * 1e3, rtol=1e-7)
    for term in VARY:
        assert_allclose(cb[term].statistic, ca[term].statistic, rtol=1e-6)
    for term in ("/sigma_u", "/sigma_e"):
        assert_allclose(cb[term].estimate, ca[term].estimate * 1e3, rtol=1e-7)
        assert_allclose([cb[term].ci_low, cb[term].ci_high],
                        [ca[term].ci_low * 1e3, ca[term].ci_high * 1e3], rtol=1e-6)
    assert_allclose(b.metrics["log_likelihood"],
                    a.metrics["log_likelihood"] - len(gappy) * np.log(1e3), rtol=1e-10)
    assert_allclose(b.tests["sigma_u"]["statistic"], a.tests["sigma_u"]["statistic"], rtol=1e-6)
    assert_allclose(b.metrics["rho"], a.metrics["rho"], rtol=1e-7)


def test_grunfeld_textbook_digits():
    """Grunfeld (10 firms, 1935-54): the digits printed for this textbook example.

    These are the values quoted by the adversarial review from Stata's output for
    ``xtreg invest value capital`` (not re-run by us; parity stays unvalidated).
    fe: sigma_u 85.732315, sigma_e 52.767964, rho .72525012, F(9, 188) = 49.18 for u_i = 0,
    R2 within/between/overall .7668/.8194/.8060, corr(u_i, Xb) = -0.1517 (over observations;
    over panels it would be -0.1698). re (Swamy-Arora; the textbook variance components for
    this data set are sigma_u^2 = 7089.80, sigma_e^2 = 2784.46, Baltagi, Econometric Analysis
    of Panel Data, ch. 2): theta .8612, standard errors 28.8989 / .0104927 / .0171805, which
    only rmse^2 (X*'X*)^-1 reproduces (sigma_e^2 (X*'X*)^-1 gives 28.8893 / .0104892 /
    .0171747).
    """
    data = sm.datasets.grunfeld.load_pandas().data
    data = data[data.firm != "American Steel"].reset_index(drop=True)
    assert data.firm.nunique() == 10 and len(data) == 200
    fe = oe.xtreg(data=data, y="invest", x=["value", "capital"], panel="firm", time="year")
    m = fe.metrics
    assert_allclose([m["sigma_u"], m["sigma_e"], m["rho"]], [85.732315, 52.767964, 0.72525012],
                    rtol=5e-6)
    assert_allclose([m["r_squared_within"], m["r_squared_between"], m["r_squared_overall"]],
                    [0.7668, 0.8194, 0.8060], atol=5e-5)
    assert_allclose(m["corr_u_xb"], -0.1517, atol=5e-5)
    test = fe.tests["fixed_effects"]
    assert (test["df"], test["df2"]) == (9, 188)
    assert_allclose(test["statistic"], 49.18, atol=5e-3)
    re = oe.xtreg(data=data, y="invest", x=["value", "capital"], panel="firm", model="re")
    assert_allclose([re.metrics["sigma_u"], re.metrics["sigma_e"]], [84.20095, 52.767964],
                    rtol=5e-7)
    assert_allclose([re.metrics["sigma_u"] ** 2, re.metrics["sigma_e"] ** 2], [7089.80, 2784.46],
                    atol=6e-3)
    assert_allclose(re.metrics["theta"], 0.8612, atol=5e-5)
    table = coef(re)
    assert_allclose([table[t].std_error for t in ("Intercept", "value", "capital")],
                    [28.8989, 0.0104927, 0.0171805], rtol=5e-6)
    old = np.sqrt(re.metrics["sigma_e"] ** 2 / re.metrics["rmse"] ** 2)
    assert abs(table["value"].std_error * old - 0.0104927) > 3e-6    # the old convention fails
    assert_allclose(re.metrics["rmse"] ** 2, re.extra["ssr_transformed"] / (200 - 3), rtol=1e-12)
    assert "rmse^2" in re.inference["correction"]


ACCEPTED_COVARIANCES = {
    "fe": {"nonrobust", "robust", "cluster", "driscoll_kraay"},
    "pooled": {"nonrobust", "robust", "cluster", "driscoll_kraay"},
    "re": {"nonrobust", "robust", "cluster"}, "be": {"nonrobust", "robust", "cluster"},
    "fd": {"nonrobust", "robust", "cluster"}, "mle": {"nonrobust"},
}
WEIGHTED_MODELS = {"fe", "fd", "pooled"}          # Stata: xtreg, re / be take none, mle iweights


@pytest.mark.parametrize("model", ["fe", "re", "be", "fd", "pooled", "mle"])
def test_every_declared_combination_is_fitted_or_refused_with_its_code(gappy, model):
    """Manifest contract: each model x covariance x weight type either fits (finite,
    serializable, positive standard errors) or raises exactly the documented code."""
    from openecon.models import ResultBundle

    declared = oe.capabilities()["estimators"]["xtreg"]
    assert set(declared["covariances"]) == {"nonrobust", "robust", "cluster", "driscoll_kraay"}
    assert set(declared["weights"]) == {"aweight", "fweight", "pweight"}
    for covariance in declared["covariances"]:
        for weight_type in (None, *declared["weights"]):
            options = {"covariance": covariance}
            if covariance == "cluster":
                options["cluster"] = "cl"
            if weight_type is not None:
                options.update(weights="wf" if weight_type == "fweight" else "wa",
                               weight_type=weight_type)
            if covariance not in ACCEPTED_COVARIANCES[model]:
                expected = "unsupported_covariance"
            elif weight_type is not None and model not in WEIGHTED_MODELS:
                expected = "unsupported_weights"
            elif weight_type == "pweight" and covariance == "nonrobust":
                expected = "unsupported_covariance"
            else:
                expected = None
            label = (model, covariance, weight_type)
            try:
                result = oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year",
                                  model=model, **options)
            except AnalysisError as err:
                assert err.code == expected, (label, err.code, str(err))
                continue
            assert expected is None, label
            assert result.inference["covariance"] == covariance, label
            assert result.provenance["stata_parity_validated"] is False
            assert all(np.isfinite(c.estimate) and c.std_error > 0 and np.isfinite(c.p_value)
                       and c.ci_low < c.ci_high for c in result.coefficients), label
            assert np.isfinite(np.array(result.covariance_matrix)).all(), label
            assert ResultBundle.model_validate_json(result.model_dump_json()) == result, label
            assert result.inference["use_t"] is (model not in {"re", "mle"}), label
    # Options that belong to another model or covariance are refused, never ignored.
    for options in ({"sa": True}, {"wls": True}, {"lags": 1}, {"kernel": "parzen"}):
        owner = {"sa": "re", "wls": "be"}.get(next(iter(options)))
        if owner == model:
            assert oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model=model,
                            **options).nobs > 0
            continue
        with pytest.raises(AnalysisError) as err:
            oe.xtreg(data=gappy, y="y", x=COLS, panel="id", time="year", model=model, **options)
        assert err.value.code == "invalid_spec", (model, options)
    # Two cluster columns: both nest the panels, so every model that clusters accepts them.
    if "cluster" in ACCEPTED_COVARIANCES[model]:
        frame = gappy.assign(cl3=gappy.id % 7)
        two = oe.xtreg(data=frame, y="y", x=COLS, panel="id", time="year", model=model,
                       covariance="cluster", cluster=["cl", "cl3"])
        assert two.inference["cluster_counts"] == [5, 7] and two.inference["cluster_df"] == 4
        assert all(c.std_error > 0 for c in two.coefficients)


# ---- performance ------------------------------------------------------------------------


def test_dense_performance_one_million_rows(monkeypatch):
    # Preserve the resident-kernel benchmark. Disk replay and spill timings are
    # measured separately on physical files, including actual large panels.
    from openecon.econometrics import streaming_registry
    monkeypatch.setattr(streaming_registry, "supports_spec", lambda spec: False)
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1024")
    rng = np.random.default_rng(0)
    big_n, n, k = 1_000_000, 100_000, 8
    ids = np.repeat(np.arange(n), 10)
    year = np.tile(np.arange(10), n)
    x = rng.normal(size=(big_n, k))
    u = rng.normal(size=n)[ids]
    y = x @ np.linspace(0.1, 0.8, k) + u + rng.normal(size=big_n)
    frame = pd.DataFrame(x, columns=[f"x{j}" for j in range(k)])
    frame["id"], frame["year"], frame["y"] = ids, year, y
    cols = [f"x{j}" for j in range(k)]
    timings = {}
    for model, extra in (("fe", {}), ("fe", {"covariance": "robust"}), ("re", {}),
                         ("pooled", {}), ("fe", {"covariance": "driscoll_kraay"})):
        start = time.perf_counter()
        result = oe.xtreg(data=frame, y="y", x=cols, panel="id", time="year", model=model, **extra)
        timings[(model, extra.get("covariance", "nonrobust"))] = time.perf_counter() - start
        assert result.nobs == big_n
    assert all(elapsed < 20 for elapsed in timings.values()), timings
