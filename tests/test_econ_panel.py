"""Independent oracles for the panel-data linear family (xtreg, hausman, xtfmb).

Fixed effects are checked against explicit dummy-variable least squares in
NumPy, random effects against the Swamy-Arora formulas coded independently,
robust/cluster factors against statsmodels OLS on the transformed data, the
random-effects likelihood against a brute-force SciPy maximization of a
likelihood written with explicit per-panel covariance matrices, and the
tests (Breusch-Pagan, Hausman, F for u_i = 0) against their textbook formulas.
"""

import json
import time

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.panel import kernels
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec, ResultBundle

X_COLS = ["x1", "x2", "z"]
VARYING = ["x1", "x2"]


def make_panel(seed=42, n=30, periods=6, unbalanced=True):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), periods)
    year = np.tile(np.arange(2001, 2001 + periods), n)
    u = rng.normal(size=n)[ids]
    x1 = rng.normal(size=n * periods) + 0.4 * u
    x2 = rng.normal(size=n * periods)
    z = rng.normal(size=n)[ids]
    y = 1 + 0.5 * x1 - 0.3 * x2 + 0.2 * z + u + rng.normal(size=n * periods)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z, "id": ids, "year": year})
    frame["w"] = rng.integers(1, 4, size=n)[ids].astype(float)
    frame["grp"] = ids % 6                     # nests the panels
    frame["tgrp"] = year % 3                   # does not nest the panels
    frame["sector"] = pd.Categorical(np.where(x2 > 0, "a", np.where(x1 > 0, "b", "c")))
    if unbalanced:
        candidates = np.flatnonzero(year >= 2003)
        frame = frame.drop(index=rng.choice(candidates, size=25, replace=False))
    return frame.reset_index(drop=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def balanced():
    return make_panel(seed=7, unbalanced=False)


def coefficients(result):
    return {c.term: c for c in result.coefficients}


def estimates(result, terms):
    table = coefficients(result)
    return np.array([table[t].estimate for t in terms]), np.array([table[t].std_error for t in terms])


def dummies(frame):
    return pd.get_dummies(frame.id).to_numpy(float)


def lsdv(frame, cols, weights=None):
    """Weighted dummy-variable OLS: slopes, SEs (df N-n-k), sigma_e^2, SSR, effects."""
    design = np.column_stack([dummies(frame), frame[cols].to_numpy()])
    y = frame.y.to_numpy()
    w = np.ones(len(y)) if weights is None else weights * len(y) / weights.sum()
    root = np.sqrt(w)
    beta = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    resid = y - design @ beta
    n, k = dummies(frame).shape[1], len(cols)
    s2 = (w * resid ** 2).sum() / (len(y) - n - k)
    cov = s2 * np.linalg.inv((design * w[:, None]).T @ design)
    return beta[n:], np.sqrt(np.diag(cov)[n:]), s2, (w * resid ** 2).sum(), beta[:n]


def within_data(frame, cols, weights=None):
    """Stata's transformed fe regression: y - ybar_i + ybar on [1, x - xbar_i + xbar]."""
    w = np.ones(len(frame)) if weights is None else weights * len(frame) / weights.sum()
    values = frame[["y", *cols]].to_numpy()
    totals = pd.Series(w).groupby(frame.id.to_numpy()).transform("sum").to_numpy()
    means = pd.DataFrame(values * w[:, None]).groupby(frame.id.to_numpy()).transform("sum")
    means = means.to_numpy() / totals[:, None]
    grand = (values * w[:, None]).sum(0) / w.sum()
    transformed = values - means + grand
    return np.column_stack([np.ones(len(frame)), transformed[:, 1:]]), transformed[:, 0], w


def swamy_arora(frame, cols, sa=False):
    """Independent Stata-style random-effects GLS (harmonic-mean or Baltagi-Chang)."""
    ids = frame.id.to_numpy()
    groups = frame.groupby("id", sort=False)
    x, y = frame[cols].to_numpy(), frame.y.to_numpy()
    xbar, ybar = groups[cols].transform("mean").to_numpy(), groups.y.transform("mean").to_numpy()
    sizes = groups.size().to_numpy().astype(float)
    codes = pd.factorize(ids)[0]
    n, big_n = len(sizes), len(y)
    within_x, within_y = x - xbar, y - ybar
    keep = within_x.std(0) > 1e-8
    bw = np.linalg.lstsq(within_x[:, keep], within_y, rcond=None)[0]
    ssr_w = ((within_y - within_x[:, keep] @ bw) ** 2).sum()
    s2e = ssr_w / (big_n - n - keep.sum())
    zb = np.column_stack([np.ones(n), groups[cols].mean().to_numpy()])
    yb = groups.y.mean().to_numpy()
    bb = np.linalg.lstsq(zb, yb, rcond=None)[0]
    ssr_b = ((yb - zb @ bb) ** 2).sum()
    t_bar = n / (1 / sizes).sum()
    s2u = max(0.0, ssr_b / (n - zb.shape[1]) - s2e / t_bar)
    if sa:
        root = np.sqrt(sizes)
        bbw = np.linalg.lstsq(zb * root[:, None], yb * root, rcond=None)[0]
        ssr_bw = (sizes * (yb - zb @ bbw) ** 2).sum()
        trace = np.trace(np.linalg.solve((zb * sizes[:, None]).T @ zb, (zb * sizes[:, None] ** 2).T @ zb))
        s2u = max(0.0, (ssr_bw - (n - zb.shape[1]) * s2e) / (big_n - trace))
    theta = 1 - np.sqrt(s2e / (sizes * s2u + s2e))
    shrink = theta[codes]
    zs = np.column_stack([np.ones(big_n), x]) - shrink[:, None] * np.column_stack([np.ones(big_n), xbar])
    ys = y - shrink * ybar
    beta = np.linalg.lstsq(zs, ys, rcond=None)[0]
    # Stata's conventional VCE is the OLS one of the transformed regression: rmse^2 (X*'X*)^-1.
    rmse2 = ((ys - zs @ beta) ** 2).sum() / (big_n - zs.shape[1])
    cov = rmse2 * np.linalg.inv(zs.T @ zs)
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "s2e": s2e, "s2u": s2u, "theta": theta,
            "zs": zs, "ys": ys, "t_bar": t_bar, "rmse": np.sqrt(rmse2)}


def driscoll_kraay(xs, resid, time, lags):
    bread = np.linalg.inv(xs.T @ xs)
    scores = xs * resid[:, None]
    periods = np.unique(time)
    h = np.array([scores[time == t].sum(0) for t in periods])
    meat = h.T @ h
    for lag in range(1, lags + 1):
        cross = h[lag:].T @ h[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    big_n, k = xs.shape
    return bread @ meat @ bread * len(periods) / (len(periods) - 1) * (big_n - 1) / (big_n - k)


def duplicated(frame):
    return frame.loc[np.repeat(frame.index, frame.w.astype(int))].reset_index(drop=True)


# ---- fixed effects ----------------------------------------------------------------------


def test_fe_matches_dummy_variable_regression(panel):
    result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year")
    beta, se, s2, ssr, effects = lsdv(panel, VARYING)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert result.provenance["omitted_terms"] == ["z"]
    assert any("absorbed fixed effects" in w for w in result.warnings)
    actual, actual_se = estimates(result, VARYING)
    assert_allclose(actual, beta, rtol=1e-10)
    assert_allclose(actual_se, se, rtol=1e-10)
    grand_x, grand_y = panel[VARYING].to_numpy().mean(0), panel.y.mean()
    xs, ys, _ = within_data(panel, VARYING)
    cons = coefficients(result)["Intercept"]
    assert cons.estimate == pytest.approx(grand_y - grand_x @ beta, rel=1e-10)
    assert cons.std_error == pytest.approx(np.sqrt(s2 * np.linalg.inv(xs.T @ xs)[0, 0]), rel=1e-10)
    n, big_n = panel.id.nunique(), len(panel)
    df_resid = big_n - n - 2
    assert result.metrics["df_resid"] == df_resid and result.inference["df_inference"] == df_resid
    assert result.metrics["sigma_e"] == pytest.approx(np.sqrt(s2), rel=1e-12)
    means = panel.groupby("id")
    u = means.y.mean().to_numpy() - means[VARYING].mean().to_numpy() @ beta
    assert result.metrics["sigma_u"] == pytest.approx(u.std(ddof=1), rel=1e-10)
    xb_bar = means[VARYING].mean().to_numpy() @ beta
    # Stata's e(corr) is corr(u_i, x_it b) over the observations, not over the panel means.
    by_row = u[panel.id.to_numpy()]
    assert result.metrics["corr_u_xb"] == pytest.approx(
        np.corrcoef(by_row, panel[VARYING].to_numpy() @ beta)[0, 1], rel=1e-10)
    assert result.metrics["corr_u_xb"] != pytest.approx(np.corrcoef(u, xb_bar)[0, 1], rel=1e-3)
    rho = u.var(ddof=1) / (u.var(ddof=1) + s2)
    assert result.metrics["rho"] == pytest.approx(rho, rel=1e-10)
    within_x = panel[VARYING].to_numpy() - means[VARYING].transform("mean").to_numpy()
    within_y = panel.y.to_numpy() - means.y.transform("mean").to_numpy()
    assert result.metrics["r_squared_within"] == pytest.approx(
        np.corrcoef(within_x @ beta, within_y)[0, 1] ** 2, rel=1e-10)
    assert result.metrics["r_squared_between"] == pytest.approx(
        np.corrcoef(xb_bar, means.y.mean())[0, 1] ** 2, rel=1e-10)
    assert result.metrics["r_squared_overall"] == pytest.approx(
        np.corrcoef(panel[VARYING].to_numpy() @ beta, panel.y)[0, 1] ** 2, rel=1e-10)
    pooled = sm.OLS(panel.y, sm.add_constant(panel[VARYING])).fit()
    f_u = ((pooled.ssr - ssr) / (n - 1)) / (ssr / df_resid)
    test = result.tests["fixed_effects"]
    assert test["statistic"] == pytest.approx(f_u, rel=1e-10) and test["df"] == n - 1
    assert test["p_value"] == pytest.approx(stats.f.sf(f_u, n - 1, df_resid), rel=1e-9)
    model = result.tests["model"]
    wald = beta @ np.linalg.solve(np.diag(se) @ np.corrcoef(np.eye(2)) @ np.diag(se), beta) / 2
    assert model["df"] == 2 and model["df2"] == df_resid and model["distribution"] == "F"
    block = np.asarray(result.covariance_matrix)[1:, 1:]
    assert model["statistic"] == pytest.approx(beta @ np.linalg.solve(block, beta) / 2, rel=1e-10)
    assert list(result.metrics)[:7] == ["r_squared_within", "r_squared_between", "r_squared_overall",
                                        "sigma_u", "sigma_e", "rho", "corr_u_xb"]
    assert result.metrics["n_groups"] == n and result.metrics["t_max"] == 6
    assert result.provenance["sample_order"] == "sorted by id, year"
    del wald


def test_fe_with_categorical_regressor_matches_lsdv(panel):
    result = oe.xtreg(data=panel, y="y", x=["x1", "sector"], panel="id", categorical=["sector"])
    frame = panel.assign(**{"sector[b]": (panel.sector == "b").astype(float),
                            "sector[c]": (panel.sector == "c").astype(float)})
    beta, se, _, _, _ = lsdv(frame, ["x1", "sector[b]", "sector[c]"])
    actual, actual_se = estimates(result, ["x1", "sector[b]", "sector[c]"])
    assert_allclose(actual, beta, rtol=1e-10)
    assert_allclose(actual_se, se, rtol=1e-10)
    assert result.provenance["categorical_encoding"]["sector"]["reference"] == "a"


def test_fe_robust_is_cluster_on_panel_with_stata_factor(panel):
    result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", covariance="robust")
    xs, ys, _ = within_data(panel, VARYING)
    fit = sm.OLS(ys, xs).fit(cov_type="cluster", cov_kwds={"groups": panel.id.to_numpy()})
    assert_allclose(np.asarray(result.covariance_matrix), fit.cov_params(), rtol=1e-9, atol=1e-14)
    n = panel.id.nunique()
    assert result.inference["df_inference"] == n - 1 and result.inference["cluster_count"] == n
    assert result.inference["cluster_column"] == "id" and result.inference["covariance"] == "robust"
    assert "vce(cluster panel)" in result.inference["correction"]
    assert result.inference["k_small_sample"] == 3
    actual, se = estimates(result, VARYING)
    assert_allclose([c.p_value for c in result.coefficients][1:],
                    2 * stats.t.sf(np.abs(actual / se), n - 1), rtol=1e-9)
    assert result.tests["model"]["df2"] == n - 1 and "fixed_effects" not in result.tests
    assert not any("panels" in w for w in result.warnings)
    few = oe.xtreg(data=panel[panel.id < 20], y="y", x=X_COLS, panel="id", covariance="robust")
    assert any("Only 20 panels" in w for w in few.warnings)
    same = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", covariance="cluster", cluster="id")
    assert_allclose(np.asarray(same.covariance_matrix), fit.cov_params(), rtol=1e-9, atol=1e-14)


def test_fe_and_re_cluster_require_panels_nested_in_clusters(panel):
    nested = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", covariance="cluster", cluster="grp")
    xs, ys, _ = within_data(panel, VARYING)
    fit = sm.OLS(ys, xs).fit(cov_type="cluster", cov_kwds={"groups": panel.grp.to_numpy()})
    assert_allclose(np.asarray(nested.covariance_matrix), fit.cov_params(), rtol=1e-9, atol=1e-14)
    assert not any("not nested" in w for w in nested.warnings)
    assert nested.inference["k_small_sample"] == 3
    # Stata's xtreg, fe / re refuse a cluster variable that does not nest the panels.
    n = panel.id.nunique()
    for model, cluster in (("fe", "tgrp"), ("re", "tgrp"), ("fe", ["id", "year"]),
                           ("re", ["grp", "tgrp"])):
        with pytest.raises(AnalysisError) as caught:
            oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", model=model,
                     covariance="cluster", cluster=cluster)
        assert caught.value.code == "cluster_not_nested"
        assert "not nested" in str(caught.value)
    # regress (model='pooled') accepts crossed clusters, including two-way (panel, time).
    two_way = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", model="pooled",
                       covariance="cluster", cluster=["id", "year"])
    assert two_way.inference["cluster_counts"][:2] == [n, 6] and two_way.inference["cluster_df"] == 5
    assert all(c.std_error > 0 for c in two_way.coefficients)
    crossed = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="pooled",
                       covariance="cluster", cluster="tgrp")
    fit = sm.OLS(panel.y, sm.add_constant(panel[X_COLS])).fit(
        cov_type="cluster", cov_kwds={"groups": panel.tgrp.to_numpy()})
    assert_allclose(np.asarray(crossed.covariance_matrix), fit.cov_params(), rtol=1e-9)


def test_driscoll_kraay_matches_explicit_hac_of_period_sums(panel):
    xs, ys, _ = within_data(panel, VARYING)
    resid = ys - xs @ np.linalg.lstsq(xs, ys, rcond=None)[0]
    for lags in (None, 0, 3):
        result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year",
                          covariance="driscoll_kraay", lags=lags)
        used = int(np.floor(4 * (6 / 100) ** (2 / 9))) if lags is None else lags
        assert result.inference["lags"] == used and result.inference["periods"] == 6
        expected = driscoll_kraay(xs, resid, panel.year.to_numpy(), used)
        assert_allclose(np.asarray(result.covariance_matrix), expected, rtol=1e-9, atol=1e-14)
        assert result.inference["df_inference"] == 5 and result.tests["model"]["df2"] == 5
    pooled = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", model="pooled",
                      covariance="driscoll_kraay", lags=2, kernel="parzen")
    assert pooled.inference["kernel"] == "parzen" and pooled.inference["lags"] == 2


def test_fe_weights_follow_stata(panel):
    weights = panel.w.to_numpy()
    aweighted = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", weights="w", weight_type="aweight")
    beta, se, s2, _, _ = lsdv(panel, VARYING, weights)
    actual, actual_se = estimates(aweighted, VARYING)
    assert_allclose(actual, beta, rtol=1e-10)
    assert_allclose(actual_se, se, rtol=1e-10)
    assert aweighted.metrics["sigma_e"] == pytest.approx(np.sqrt(s2), rel=1e-10)
    frequency = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", weights="w", weight_type="fweight")
    expanded = oe.xtreg(data=duplicated(panel), y="y", x=X_COLS, panel="id")
    assert frequency.nobs == expanded.nobs == int(weights.sum())
    for name in ("estimate", "std_error", "p_value"):
        assert_allclose([getattr(c, name) for c in frequency.coefficients],
                        [getattr(c, name) for c in expanded.coefficients], rtol=1e-9)
    for key in ("sigma_u", "sigma_e", "r_squared_within", "r_squared_between", "t_avg"):
        assert frequency.metrics[key] == pytest.approx(expanded.metrics[key], rel=1e-9)
    assert frequency.tests["fixed_effects"]["statistic"] == pytest.approx(
        expanded.tests["fixed_effects"]["statistic"], rel=1e-9)
    probability = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", weights="w",
                           weight_type="pweight", covariance="robust")
    robust = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", weights="w",
                      weight_type="aweight", covariance="robust")
    assert_allclose(np.asarray(probability.covariance_matrix), np.asarray(robust.covariance_matrix),
                    rtol=1e-12)
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", weights="w", weight_type="pweight")
    assert caught.value.code == "unsupported_covariance"
    varying = panel.assign(w=panel.w + (panel.year == 2001))
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=varying, y="y", x=X_COLS, panel="id", weights="w", weight_type="aweight")
    assert caught.value.code == "weights_vary_within_panel"
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="be", weights="w",
                 weight_type="aweight")
    assert caught.value.code == "unsupported_weights"


# ---- random effects -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["panel", "balanced"])
def test_re_matches_independent_swamy_arora(name, request):
    frame = request.getfixturevalue(name)
    result = oe.xtreg(data=frame, y="y", x=X_COLS, panel="id", model="re")
    oracle = swamy_arora(frame, X_COLS)
    actual, se = estimates(result, ["Intercept", *X_COLS])
    assert_allclose(actual, oracle["beta"], rtol=1e-10)
    assert_allclose(se, oracle["se"], rtol=1e-10)
    assert result.metrics["sigma_e"] == pytest.approx(np.sqrt(oracle["s2e"]), rel=1e-10)
    assert result.metrics["sigma_u"] == pytest.approx(np.sqrt(oracle["s2u"]), rel=1e-10)
    assert result.metrics["rho"] == pytest.approx(oracle["s2u"] / (oracle["s2u"] + oracle["s2e"]))
    assert result.metrics["rmse"] == pytest.approx(oracle["rmse"], rel=1e-10)
    fit = sm.OLS(oracle["ys"], oracle["zs"]).fit()        # the transformed regression itself
    assert_allclose(np.asarray(result.covariance_matrix), fit.cov_params(), rtol=1e-9)
    assert result.inference["use_t"] is False and result.inference["df_inference"] is None
    components = result.extra["variance_components"]
    assert components["t_bar_harmonic"] == pytest.approx(oracle["t_bar"]) and components["within_regressors"] == 2
    if name == "balanced":
        assert result.metrics["theta"] == pytest.approx(oracle["theta"][0], rel=1e-10)
    else:
        assert result.metrics["theta"] is None
        assert result.extra["theta"]["min"] == pytest.approx(oracle["theta"].min(), rel=1e-10)
        assert result.extra["theta"]["max"] == pytest.approx(oracle["theta"].max(), rel=1e-10)
    block = np.asarray(result.covariance_matrix)[1:, 1:]
    wald = actual[1:] @ np.linalg.solve(block, actual[1:])
    assert result.tests["model"]["statistic"] == pytest.approx(wald, rel=1e-10)
    assert result.tests["model"]["distribution"] == "chi2" and result.tests["model"]["df"] == 3
    pooled = sm.OLS(frame.y, sm.add_constant(frame[X_COLS])).fit()
    sums = pd.Series(pooled.resid.to_numpy()).groupby(frame.id.to_numpy()).sum()
    sizes = frame.groupby("id").size().to_numpy()
    big_n = len(frame)
    lm = big_n ** 2 / (2 * ((sizes ** 2).sum() - big_n)) * ((sums ** 2).sum() / (pooled.resid ** 2).sum() - 1) ** 2
    assert result.tests["breusch_pagan"]["statistic"] == pytest.approx(lm, rel=1e-10)
    assert result.tests["breusch_pagan"]["p_value"] == pytest.approx(0.5 * stats.chi2.sf(lm, 1), rel=1e-9)


def test_re_baltagi_chang_option(panel, balanced):
    result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="re", sa=True)
    oracle = swamy_arora(panel, X_COLS, sa=True)
    actual, se = estimates(result, ["Intercept", *X_COLS])
    assert_allclose(actual, oracle["beta"], rtol=1e-10)
    assert_allclose(se, oracle["se"], rtol=1e-10)
    assert result.metrics["sigma_u"] == pytest.approx(np.sqrt(oracle["s2u"]), rel=1e-10)
    assert result.extra["variance_components"]["method"] == "baltagi_chang_sa"
    assert result.metrics["sigma_u"] != pytest.approx(
        oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="re").metrics["sigma_u"])
    # In a balanced panel the Baltagi-Chang formula reduces to the Swamy-Arora one.
    plain = oe.xtreg(data=balanced, y="y", x=X_COLS, panel="id", model="re")
    adjusted = oe.xtreg(data=balanced, y="y", x=X_COLS, panel="id", model="re", sa=True)
    assert adjusted.metrics["sigma_u"] == pytest.approx(plain.metrics["sigma_u"], rel=1e-10)


def test_re_robust_and_cluster_use_transformed_regression(panel):
    oracle = swamy_arora(panel, X_COLS)
    for covariance, cluster, groups in (("robust", None, panel.id), ("cluster", "grp", panel.grp)):
        result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="re",
                          covariance=covariance, cluster=cluster)
        fit = sm.OLS(oracle["ys"], oracle["zs"]).fit(cov_type="cluster",
                                                     cov_kwds={"groups": groups.to_numpy()})
        assert_allclose(np.asarray(result.covariance_matrix), fit.cov_params(), rtol=1e-9, atol=1e-14)
        assert result.inference["use_t"] is False
        assert result.tests["model"]["distribution"] == "chi2"


def test_re_takes_no_weights_as_in_stata(panel):
    for weight_type, covariance in (("fweight", None), ("aweight", None), ("pweight", "robust")):
        with pytest.raises(AnalysisError) as caught:
            oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="re", weights="w",
                     weight_type=weight_type, covariance=covariance)
        assert caught.value.code == "unsupported_weights"
        assert "model='re'" in str(caught.value)
    # The unweighted fit on explicitly duplicated rows is how frequency weights are obtained.
    expanded = oe.xtreg(data=duplicated(panel), y="y", x=X_COLS, panel="id", model="re")
    assert expanded.nobs == int(panel.w.sum())


# ---- between, first differences, pooled ----------------------------------------------------


def test_between_matches_panel_mean_regression(panel):
    groups = panel.groupby("id")
    zb, yb = sm.add_constant(groups[X_COLS].mean()), groups.y.mean()
    sizes = groups.size().to_numpy().astype(float)
    for wls in (False, True):
        result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="be", wls=wls)
        fit = (sm.WLS(yb, zb, weights=sizes) if wls else sm.OLS(yb, zb)).fit()
        actual, se = estimates(result, ["Intercept", *X_COLS])
        assert_allclose(actual, fit.params.to_numpy(), rtol=1e-10)
        assert_allclose(se, fit.bse.to_numpy(), rtol=1e-10)
        assert result.metrics["r_squared_between"] == pytest.approx(fit.rsquared, rel=1e-10)
        assert result.metrics["rmse"] == pytest.approx(np.sqrt(fit.mse_resid), rel=1e-10)
        assert result.metrics["df_resid"] == 26 and result.inference["df_inference"] == 26
        assert result.tests["model"]["statistic"] == pytest.approx(fit.fvalue, rel=1e-10)
        assert result.extra["weighted_by_panel_length"] is wls
    robust = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="be", covariance="robust")
    hc1 = sm.OLS(yb, zb).fit(cov_type="HC1")
    assert_allclose(np.asarray(robust.covariance_matrix), hc1.cov_params(), rtol=1e-9)
    clustered = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="be", covariance="cluster",
                         cluster="grp")
    fit = sm.OLS(yb, zb).fit(cov_type="cluster", cov_kwds={"groups": groups.grp.first().to_numpy()})
    assert_allclose(np.asarray(clustered.covariance_matrix), fit.cov_params(), rtol=1e-9)
    assert clustered.inference["df_inference"] == 5
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="be", covariance="cluster",
                 cluster="tgrp")
    assert caught.value.code == "cluster_varies_within_panel"


def test_first_differences_use_consecutive_periods_only(panel):
    result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", model="fd")
    ordered = panel.sort_values(["id", "year"]).reset_index(drop=True)
    previous = ordered.shift(1)
    valid = (previous.id == ordered.id) & (ordered.year - previous.year == 1)
    diff = (ordered[["y", *VARYING]] - previous[["y", *VARYING]])[valid]
    fit = sm.OLS(diff.y, sm.add_constant(diff[VARYING])).fit()
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert result.provenance["omitted_terms"] == ["z"]
    actual, se = estimates(result, ["Intercept", *VARYING])
    assert_allclose(actual, fit.params.to_numpy(), rtol=1e-10)
    assert_allclose(se, fit.bse.to_numpy(), rtol=1e-10)
    assert result.nobs == int(valid.sum()) and result.dropped_rows == len(panel) - int(valid.sum())
    assert result.extra["dropped_for_differencing"] == result.dropped_rows
    assert any("consecutive periods" in w for w in result.warnings)
    assert result.metrics["r_squared"] == pytest.approx(fit.rsquared, rel=1e-10)
    assert result.metrics["df_resid"] == int(valid.sum()) - 3
    assert result.predictions[0]["observed"] == pytest.approx(diff.y.iloc[0])
    robust = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", model="fd",
                      covariance="robust")
    fit = sm.OLS(diff.y, sm.add_constant(diff[VARYING])).fit(
        cov_type="cluster", cov_kwds={"groups": ordered.id[valid].to_numpy()})
    assert_allclose(np.asarray(robust.covariance_matrix), fit.cov_params(), rtol=1e-9)
    assert robust.inference["df_inference"] == panel.id.nunique() - 1
    frequency = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", model="fd",
                         weights="w", weight_type="fweight")
    assert frequency.nobs == int(ordered.w[valid].sum())


def test_pooled_matches_statsmodels_with_panel_covariances(panel):
    result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="pooled")
    fit = sm.OLS(panel.y, sm.add_constant(panel[X_COLS])).fit()
    actual, se = estimates(result, ["Intercept", *X_COLS])
    assert_allclose(actual, fit.params.to_numpy(), rtol=1e-10)
    assert_allclose(se, fit.bse.to_numpy(), rtol=1e-10)
    assert result.metrics["adjusted_r_squared"] == pytest.approx(fit.rsquared_adj, rel=1e-10)
    assert result.metrics["n_groups"] == 30 and result.tests["model"]["df2"] == len(panel) - 4
    robust = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="pooled", covariance="robust")
    clustered = sm.OLS(panel.y, sm.add_constant(panel[X_COLS])).fit(
        cov_type="cluster", cov_kwds={"groups": panel.id.to_numpy()})
    assert_allclose(np.asarray(robust.covariance_matrix), clustered.cov_params(), rtol=1e-9)
    assert robust.inference["df_inference"] == 29
    duplicate = oe.xtreg(data=panel.assign(twice=2 * panel.x1), y="y", x=["x1", "twice", "x2"],
                         panel="id", model="pooled")
    assert duplicate.provenance["omitted_terms"] == ["twice"]


# ---- maximum likelihood ------------------------------------------------------------------


def negative_log_likelihood(theta, x, y, codes, n):
    beta, s_u, s_e = theta[:-2], np.exp(2 * theta[-2]), np.exp(2 * theta[-1])
    resid = y - x @ beta
    total = 0.0
    for i in range(n):
        r = resid[codes == i]
        size = len(r)
        v = s_e * np.eye(size) + s_u * np.ones((size, size))
        total += -0.5 * (size * np.log(2 * np.pi) + np.linalg.slogdet(v)[1] + r @ np.linalg.solve(v, r))
    return -total


def test_mle_matches_brute_force_likelihood_maximization(panel):
    result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="mle")
    ordered = panel.sort_values(["id", "year"])
    x = sm.add_constant(ordered[X_COLS]).to_numpy()
    y, codes = ordered.y.to_numpy(), pd.factorize(ordered.id)[0]
    n = codes.max() + 1
    gls = swamy_arora(panel, X_COLS)
    start = np.concatenate([gls["beta"], [0.5 * np.log(gls["s2u"]), 0.5 * np.log(gls["s2e"])]])
    brute = minimize(negative_log_likelihood, start, args=(x, y, codes, n), method="BFGS",
                     options={"gtol": 1e-9})
    terms = ["Intercept", *X_COLS, "/sigma_u", "/sigma_e"]
    assert [c.term for c in result.coefficients] == terms
    assert [c.equation for c in result.coefficients] == ["y"] * 4 + [None, None]
    actual, se = estimates(result, terms)
    expected = np.concatenate([brute.x[:4], np.exp(brute.x[4:])])
    assert_allclose(actual, expected, rtol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(-brute.fun, rel=1e-10)
    ln_sigma = result.extra["ln_sigma"]
    theta = np.concatenate([actual[:4], [ln_sigma["sigma_u"]["estimate"],
                                         ln_sigma["sigma_e"]["estimate"]]])
    hessian = approx_hess(theta, negative_log_likelihood, args=(x, y, codes, n))
    jacobian = np.concatenate([np.ones(4), np.exp(theta[4:])])
    expected_se = np.sqrt(np.diag(np.linalg.inv(hessian))) * jacobian
    assert_allclose(se, expected_se, rtol=2e-4)
    assert result.metrics["aic"] == pytest.approx(2 * brute.fun + 2 * 6, rel=1e-10)
    pooled = sm.OLS(y, x).fit()
    lr = 2 * (-brute.fun - pooled.llf)
    assert result.tests["sigma_u"]["statistic"] == pytest.approx(lr, rel=1e-8)
    assert result.tests["sigma_u"]["p_value"] == pytest.approx(0.5 * stats.chi2.sf(lr, 1), rel=1e-6)
    null = minimize(negative_log_likelihood, np.array([y.mean(), *brute.x[4:]]),
                    args=(np.ones((len(y), 1)), y, codes, n), method="BFGS", options={"gtol": 1e-9})
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (null.fun - brute.fun), rel=1e-7)
    assert result.tests["model"]["df"] == 3 and result.inference["use_t"] is False
    assert result.provenance["optimizer"]["converged"] is True


def test_random_effects_likelihood_derivatives_are_analytic(panel):
    ordered = panel.sort_values(["id", "year"])
    x = torch.as_tensor(sm.add_constant(ordered[X_COLS]).to_numpy(), dtype=torch.float64)
    y = torch.as_tensor(ordered.y.to_numpy(), dtype=torch.float64)
    codes = torch.as_tensor(pd.factorize(ordered.id)[0], dtype=torch.int64)
    like = kernels.RandomEffectsLikelihood(x, y, codes, int(codes.max()) + 1)
    theta = torch.tensor([0.8, 0.4, -0.2, 0.1, -0.3, 0.1], dtype=torch.float64)
    report = check_derivatives(like, theta)
    assert report["gradient_max_rel_error"] < 1e-8
    assert report["hessian_max_rel_error"] < 1e-7 and report["hessian_asymmetry"] < 1e-12
    value = negative_log_likelihood(theta.numpy(), x.numpy(), y.numpy(), codes.numpy(), like.n)
    assert float(like.value(theta)) == pytest.approx(-value, rel=1e-12)


def test_mle_reports_boundary_solution_without_panel_variance():
    rng = np.random.default_rng(3)
    ids = np.repeat(np.arange(25), 5)
    noise = rng.normal(size=125)
    noise -= pd.Series(noise).groupby(ids).transform("mean").to_numpy()   # no between variation
    frame = pd.DataFrame({"y": noise + 0.3 * rng.normal(size=125), "x": rng.normal(size=125), "id": ids})
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=frame, y="y", x=["x"], panel="id", model="mle")
    assert caught.value.code == "boundary_solution"
    gls = oe.xtreg(data=frame, y="y", x=["x"], panel="id", model="re")
    assert gls.metrics["sigma_u"] == 0 and any("reduces to pooled OLS" in w for w in gls.warnings)


# ---- Hausman -------------------------------------------------------------------------------


def test_hausman_matches_explicit_formula(panel):
    fe = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id")
    re = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="re")
    result = oe.hausman(fe, re)
    b_c, _ = estimates(fe, VARYING)
    b_e, _ = estimates(re, VARYING)
    v_c = np.asarray(fe.covariance_matrix)[1:, 1:]
    v_e = np.asarray(re.covariance_matrix)[1:3, 1:3]
    d = b_c - b_e
    statistic = d @ np.linalg.solve(v_c - v_e, d)
    assert result["terms"] == VARYING and result["df"] == 2
    assert result["statistic"] == pytest.approx(statistic, rel=1e-10)
    assert result["p_value"] == pytest.approx(stats.chi2.sf(statistic, 2), rel=1e-9)
    assert_allclose(result["difference"], d, rtol=1e-12)
    assert_allclose(result["se_difference"], np.sqrt(np.diag(v_c - v_e)), rtol=1e-12)
    assert result["negative_definite"] is False and result["note"] is None
    more = oe.hausman(fe, re, sigmamore=True)
    # help hausman: e(sigma_e) after xtreg, fe; e(rmse) after xtreg, re.
    ratio = re.metrics["rmse"] ** 2 / fe.metrics["sigma_e"] ** 2
    assert abs(ratio - 1) > 1e-3
    expected = d @ np.linalg.solve(v_c * ratio - v_e, d)
    assert more["statistic"] == pytest.approx(expected, rel=1e-10) and more["sigma"] == "sigmamore"
    assert more["statistic"] != pytest.approx(statistic, rel=1e-3)
    assert np.linalg.eigvalsh(v_c * ratio - v_e).min() > 0 and more["negative_definite"] is False
    less = oe.hausman(fe, re, sigmaless=True)
    assert less["statistic"] == pytest.approx(d @ np.linalg.solve(v_c - v_e / ratio, d), rel=1e-10)
    reversed_roles = oe.hausman(re, fe)
    assert reversed_roles["statistic"] < 0 and reversed_roles["p_value"] is None
    assert reversed_roles["negative_definite"] is True and "chi2 < 0" in reversed_roles["note"]
    with pytest.raises(AnalysisError) as caught:
        oe.hausman(fe, re, sigmamore=True, sigmaless=True)
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.hausman(fe, oe.xtreg(data=panel, y="y", x=["z"], panel="id", model="re"))
    assert caught.value.code == "no_common_terms"


# ---- Fama-MacBeth ---------------------------------------------------------------------


def test_fama_macbeth_matches_period_by_period_regressions(panel):
    result = oe.xtfmb(data=panel, y="y", x=X_COLS, panel="id", time="year")
    periods = sorted(panel.year.unique())
    estimates_by_period = np.array([
        sm.OLS(panel.y[panel.year == t], sm.add_constant(panel[X_COLS][panel.year == t])).fit().params
        for t in periods])
    mean = estimates_by_period.mean(0)
    actual, se = estimates(result, ["Intercept", *X_COLS])
    assert_allclose(actual, mean, rtol=1e-10)
    assert_allclose(se, estimates_by_period.std(0, ddof=1) / np.sqrt(len(periods)), rtol=1e-10)
    assert result.inference["df_inference"] == 5 and result.metrics["n_periods"] == 6
    hac = oe.xtfmb(data=panel, y="y", x=X_COLS, panel="id", time="year", covariance="hac", lags=2)
    deviations = estimates_by_period - mean
    meat = deviations.T @ deviations
    for lag in (1, 2):
        cross = deviations[lag:].T @ deviations[:-lag]
        meat += (1 - lag / 3) * (cross + cross.T)
    expected = meat / 36 * 6 / 5
    assert_allclose(np.asarray(hac.covariance_matrix), expected, rtol=1e-9, atol=1e-14)
    zero = oe.xtfmb(data=panel, y="y", x=X_COLS, panel="id", time="year", covariance="hac", lags=0)
    assert_allclose(np.asarray(zero.covariance_matrix), np.asarray(result.covariance_matrix), rtol=1e-12)
    with pytest.raises(AnalysisError) as caught:
        oe.xtfmb(data=panel, y="y", x=X_COLS, panel="id", time="year", lags=2)
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.xtfmb(data=panel.head(8), y="y", x=X_COLS, panel="id", time="year")
    assert caught.value.code == "insufficient_observations"


# ---- contract: errors, missing data, serialization, exports ---------------------------------


def test_failure_contract(panel):
    cases = [
        ({"model": "re", "covariance": "driscoll_kraay", "time": "year"}, "unsupported_covariance"),
        ({"model": "mle", "covariance": "robust"}, "unsupported_covariance"),
        ({"lags": 2}, "invalid_spec"),
        ({"covariance": "driscoll_kraay"}, "invalid_spec"),
        ({"model": "fd"}, "invalid_spec"),
        ({"sa": True}, "invalid_spec"),
        ({"wls": True}, "invalid_spec"),
        ({"model": "mle", "weights": "w", "weight_type": "fweight"}, "unsupported_weights"),
        ({"model": "re", "weights": "w", "weight_type": "fweight"}, "unsupported_weights"),
        ({"model": "be", "weights": "w", "weight_type": "aweight"}, "unsupported_weights"),
        ({"covariance": "cluster", "cluster": "tgrp"}, "cluster_not_nested"),
        ({"model": "re", "covariance": "cluster", "cluster": "tgrp"}, "cluster_not_nested"),
        ({"x": "x1"}, "invalid_spec"),
    ]
    for update, code in cases:
        arguments = {"data": panel, "y": "y", "x": X_COLS, "panel": "id", **update}
        with pytest.raises(AnalysisError) as caught:
            oe.xtreg(**arguments)
        assert caught.value.code == code, update
    with pytest.raises((ValidationError, AnalysisError)):
        ModelSpec(estimator="xtreg", outcome="y", predictors=["x1"], panel="id", options={"model": "ols"})
    with pytest.raises((ValidationError, AnalysisError)):
        ModelSpec(estimator="xtreg", outcome="y", predictors=["x1"], panel="id", intercept=False)
    with pytest.raises((ValidationError, AnalysisError)):
        ModelSpec(estimator="xtreg", outcome="y", predictors=["x1"])
    missing = panel.copy()
    missing.loc[[2, 9], "x1"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=missing, y="y", x=X_COLS, panel="id")
    assert caught.value.code == "missing_values"
    dropped = oe.xtreg(data=missing, y="y", x=X_COLS, panel="id", missing="drop")
    assert dropped.nobs == len(panel) - 2 and dropped.dropped_rows == 2
    assert sorted(set(range(len(panel))) - set(dropped.sample_positions)) == [2, 9]
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=pd.concat([panel, panel.iloc[:1]]), y="y", x=X_COLS, panel="id", time="year")
    assert caught.value.code == "repeated_time_values"
    tiny = panel[panel.id < 2]
    with pytest.raises(AnalysisError) as caught:
        oe.xtreg(data=tiny, y="y", x=X_COLS, panel="id", model="be")
    assert caught.value.code == "insufficient_observations"
    with pytest.raises((ValidationError, AnalysisError)):     # the registry rejects a cluster column without cluster vce
        oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", covariance="robust", cluster="grp")


def test_serialization_rendering_and_exports(panel):
    for model in ("fe", "re", "be", "fd", "pooled", "mle"):
        result = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", model=model)
        restored = ResultBundle.model_validate_json(result.model_dump_json())
        assert restored == result
        json.loads(result.model_dump_json())
        assert result.provenance["stata_parity_validated"] is False
        assert result.provenance["family"] == "panel" and result.provenance["model"] == model
    fe = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", time="year", covariance="driscoll_kraay")
    text = fe.summary()
    assert "Fixed-effects (within) regression — y" in text and "F test of the slopes: F(2, 5)" in text
    assert "sigma_u:" in text and r"R^{2}" not in text
    latex = str(fe.to_latex())
    assert "x1" in latex and "begin{tabular}" in latex
    mle = oe.xtreg(data=panel, y="y", x=X_COLS, panel="id", model="mle")
    assert "/sigma_u" in mle.summary() and "chibar2(1)" in mle.summary()
    assert "sigma" in str(mle.to_latex(style="diagnostic"))
    assert callable(oe.xtreg) and callable(oe.hausman) and callable(oe.xtfmb)
    assert "xtreg" in dir(oe) and "hausman" in dir(oe)
    capability = oe.capabilities()["estimators"]["xtreg"]
    assert capability["panel"] == "required" and capability["options"]["model"]["default"] == "fe"
    assert capability["covariances"] == ["nonrobust", "robust", "cluster", "driscoll_kraay"]
    assert capability["inference"] == "Student t"
    spec = ModelSpec(estimator="xtreg", outcome="y", predictors=X_COLS, panel="id",
                     options={"model": "re"})
    assert oe.fit(spec, data=panel).title == "Random-effects GLS regression"


def test_large_panel_runs_in_seconds():
    rng = np.random.default_rng(5)
    n, periods = 20_000, 10
    ids, year = np.repeat(np.arange(n), periods), np.tile(np.arange(periods), n)
    x = rng.normal(size=(n * periods, 6))
    y = x.sum(1) + rng.normal(size=n)[ids] + rng.normal(size=n * periods)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)]).assign(y=y, id=ids, year=year)
    names = [f"x{i}" for i in range(6)]
    start = time.perf_counter()
    fe = oe.xtreg(data=frame, y="y", x=names, panel="id", time="year", covariance="robust")
    re = oe.xtreg(data=frame, y="y", x=names, panel="id", model="re")
    assert time.perf_counter() - start < 20
    assert fe.nobs == re.nobs == n * periods and fe.metrics["n_groups"] == n
    assert_allclose([c.estimate for c in fe.coefficients][1:], np.ones(6), atol=0.02)
