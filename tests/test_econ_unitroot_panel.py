"""Independent oracles for the panel unit-root and panel cointegration tests.

oe.xtunitroot (Levin-Lin-Chu, Im-Pesaran-Shin, Fisher, Hadri, Breitung) and
oe.xtcointtest (Kao) are compared with statistics computed panel by panel in
NumPy / statsmodels from the published formulas and the typed tables; the
tables themselves are checked by interpolation identities and Monte Carlo.
"""

import json
import math
import time

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.tsa.adfvalues as adfvalues
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.tsa.stattools import adfuller

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.unitroot import critical, tables
from openecon.frame import DataFrame

SM_CASE = {"none": "n", "constant": "c", "trend": "ct"}


def code(function, *args, **kwargs):
    with pytest.raises(AnalysisError) as error:
        function(*args, **kwargs)
    return error.value.code


def make_panel(seed=3, n=12, periods=40, rho=1.0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, periods)) * rng.uniform(0.5, 2.0, size=(n, 1))
    y = np.zeros((n, periods))
    for t in range(1, periods):
        y[:, t] = rho * y[:, t - 1] + e[:, t] + 0.3 * e[:, t - 1]
    y += rng.normal(size=(n, 1)) * 3
    return pd.DataFrame({"id": np.repeat([f"u{i:02d}" for i in range(n)], periods),
                         "year": np.tile(np.arange(1980, 1980 + periods), n), "y": y.ravel()})


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def series_of(frame):
    return [group["y"].to_numpy() for _, group in frame.sort_values(["id", "year"]).groupby("id")]


def deterministic(n, trend):
    columns = []
    if trend in ("constant", "trend"):
        columns.append(np.ones(n))
    if trend == "trend":
        columns.append(np.arange(1.0, n + 1))
    return columns


def bartlett_lrv(d, lags):
    n = len(d)
    return d @ d / n + 2 * sum((1 - j / (lags + 1)) * (d[j:] @ d[:-j]) / n
                               for j in range(1, lags + 1))


# ---- tables -------------------------------------------------------------------------


def test_llc_adjustment_table_and_interpolation():
    assert critical.llc_adjustment("constant", 25)[:2] == (-0.554, 0.919)
    assert critical.llc_adjustment("trend", 100)[:2] == (-0.566, 0.695)
    mean, deviation, note = critical.llc_adjustment("constant", 27.5)
    assert (mean, deviation) == pytest.approx((-0.550, 0.904)) and note is None
    mean, deviation, note = critical.llc_adjustment("trend", 20)
    assert (mean, deviation) == (-0.703, 1.003) and "T = 25" in note
    mean, deviation, _ = critical.llc_adjustment("constant", 500)
    assert mean == pytest.approx(-0.5045) and deviation == pytest.approx(0.7245)
    assert critical.llc_adjustment("none", 1e9)[:2] == pytest.approx((0.0, 1.0), abs=1e-6)
    # Asymptotic rows: -1/2 and sqrt(1/2) with panel means, -1/2 and 1/2 with trends.
    assert tables.LLC_ADJUST["constant"][-1] == (-0.5, 0.707)
    assert tables.LLC_ADJUST["trend"][-1] == (-0.5, 0.5)
    for case in tables.LLC_ADJUST:
        rows = np.array(tables.LLC_ADJUST[case])
        assert len(rows) == len(tables.LLC_T)
        assert np.all(np.diff(rows[:, 1]) <= 0)        # sigma* decreases with T


def test_ips_moment_table_interpolation_and_limits():
    assert critical.ips_moments("constant", 0, 10) == (-1.504, 1.069)
    assert critical.ips_moments("trend", 8, 100) == (-2.088, 0.670)
    assert critical.ips_moments("trend", 3, 5000) == (-2.158, 0.625)
    mean, variance = critical.ips_moments("constant", 2, 35)
    assert mean == pytest.approx((-1.460 - 1.476) / 2)
    assert variance == pytest.approx((0.865 + 0.830) / 2)
    with pytest.raises(AnalysisError) as error:
        critical.ips_moments("constant", 9, 50)
    assert error.value.code == "ips_moments_unavailable"
    with pytest.raises(AnalysisError):
        critical.ips_moments("constant", 6, 20)        # blank cell of the table
    with pytest.raises(AnalysisError):
        critical.ips_moments("trend", 0, 8)
    for case in ("constant", "trend"):
        assert len(tables.IPS_MEAN[case]) == len(tables.IPS_VARIANCE[case]) == 9
        for means, variances in zip(tables.IPS_MEAN[case], tables.IPS_VARIANCE[case],
                                    strict=True):
            assert len(means) == len(variances) == len(tables.IPS_T)
            assert [m is None for m in means] == [v is None for v in variances]


@pytest.mark.parametrize("case,lags,nobs", [("constant", 0, 10), ("constant", 2, 25),
                                            ("trend", 1, 15), ("trend", 4, 40)])
def test_ips_moments_agree_with_simulation(case, lags, nobs):
    rng = np.random.default_rng(1000 + nobs + lags)
    reps = 40000
    y = rng.standard_normal((reps, nobs + lags + 1)).cumsum(axis=1)
    dy = np.diff(y, axis=1)
    columns = [y[:, lags:-1]] + [dy[:, lags - j:-j] for j in range(1, lags + 1)]
    columns += [np.broadcast_to(c, (reps, nobs)) for c in deterministic(nobs, case)]
    x = np.stack(columns, axis=2)
    target = dy[:, lags:]
    xtx = np.einsum("rnk,rnl->rkl", x, x)
    beta = np.linalg.solve(xtx, np.einsum("rnk,rn->rk", x, target)[..., None])[..., 0]
    resid = target - np.einsum("rnk,rk->rn", x, beta)
    s2 = (resid ** 2).sum(axis=1) / (nobs - x.shape[2])
    t = beta[:, 0] / np.sqrt(s2 * np.linalg.inv(xtx)[:, 0, 0])
    mean, variance = critical.ips_moments(case, lags, nobs)
    assert t.mean() == pytest.approx(mean, abs=0.02)
    assert t.var() == pytest.approx(variance, rel=0.06)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic_spectral"])
def test_batched_long_run_variance_equals_the_shared_hac_kernel(kernel):
    import torch

    from openecon.econometrics.unitroot.panel_kernels import long_run_variances
    from openecon.engines.covariance import meat_hac

    generator = torch.Generator().manual_seed(5)
    d = torch.randn((4, 37), generator=generator, dtype=torch.float64)
    for lags in (0, 3, 9):
        batched = long_run_variances(d, lags, kernel)
        for row in range(4):
            expected = meat_hac(d[row][:, None].contiguous(), lags, kernel)[0, 0] / 37
            assert float(batched[row]) == pytest.approx(float(expected), rel=1e-10)


# ---- Levin-Lin-Chu ---------------------------------------------------------------------


def _llc_by_hand(frame, lags, trend, kernel_lags, lrv="paper"):
    lags = lags if isinstance(lags, list) else [lags] * frame["id"].nunique()
    e_all, v_all, ratios = [], [], []
    for y, p in zip(series_of(frame), lags, strict=True):
        total = len(y)
        dy = np.diff(y)
        n = total - p - 1
        target, level = dy[p:], y[p:total - 1]
        z = [dy[p - j:total - 1 - j] for j in range(1, p + 1)] + deterministic(n, trend)
        if z:
            z = np.column_stack(z)
            e = target - z @ np.linalg.lstsq(z, target, rcond=None)[0]
            v = level - z @ np.linalg.lstsq(z, level, rcond=None)[0]
        else:
            e, v = target, level
        rho = e @ v / (v @ v)
        sigma = np.sqrt(np.sum((e - rho * v) ** 2) / n)
        e_all.append(e / sigma)
        v_all.append(v / sigma)
        # Step 2: Delta y minus its mean when panel means or trends are in the model;
        # lrv="null" keeps Delta y as it is with panel means.
        d = dy.copy()
        if trend == "trend" or (trend == "constant" and lrv == "paper"):
            d = d - d.mean()
        ratios.append(np.sqrt(bartlett_lrv(d, kernel_lags)) / sigma)
    e, v = np.concatenate(e_all), np.concatenate(v_all)
    delta = e @ v / (v @ v)
    m = len(e)
    s2 = np.sum((e - delta * v) ** 2) / m
    se = np.sqrt(s2 / (v @ v))
    t_delta = delta / se
    s_n = np.mean(ratios)
    mean, deviation, _ = critical.llc_adjustment(trend, m / len(lags))
    return t_delta, (t_delta - m * s_n * se * mean / s2) / deviation, s_n


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("lags", [0, 2])
def test_llc_matches_hand_computation(panel, trend, lags):
    periods = 40
    default_lags = int(3.21 * periods ** (1 / 3))           # Stata: int(3.21 T^(1/3))
    unadjusted, adjusted, s_n = _llc_by_hand(panel, lags, trend, default_lags)
    result = oe.xtunitroot(panel, "y", "id", "year", test="llc", lags=lags, trend=trend)
    attrs = result.attrs
    assert attrs["unadjusted_t"] == pytest.approx(unadjusted, rel=1e-8)
    assert attrs["statistic"] == pytest.approx(adjusted, rel=1e-8)
    assert attrs["s_n"] == pytest.approx(s_n, rel=1e-8)
    assert attrs["p_value"] == pytest.approx(stats.norm.cdf(adjusted), rel=1e-8)
    assert attrs["kernel_lags"] == default_lags == 10
    assert attrs["t_tilde"] == periods - lags - 1
    assert (attrs["n_panels"], attrs["periods"], attrs["nobs"]) == (12, 40.0, 480)
    assert list(result.index) == ["unadjusted_t", "adjusted_t"]
    assert result.loc["adjusted_t", "statistic"] == attrs["statistic"]
    for kernel_lags, lrv in ((3, "paper"), (5, "null"), (0, "null")):
        expected = _llc_by_hand(panel, lags, trend, kernel_lags, lrv)
        other = oe.xtunitroot(panel, "y", "id", "year", test="llc", lags=lags, trend=trend,
                              kernel_lags=kernel_lags, lrv=lrv)
        assert other.attrs["statistic"] == pytest.approx(expected[1], rel=1e-8)
        assert other.attrs["unadjusted_t"] == pytest.approx(unadjusted, rel=1e-8)


def _panel_lag_choice(y, trend, method, maxlag):
    """AIC / BIC lag order (1..maxlag, as Stata) of one ADF regression on the common sample."""
    total = len(y)
    dy = np.diff(y)
    n = total - maxlag - 1
    best = (np.inf, 0)
    for p in range(maxlag, 0, -1):
        columns = [y[maxlag:total - 1]] + [dy[maxlag - j:total - 1 - j] for j in range(1, p + 1)]
        columns += deterministic(n, trend)
        ssr = sm.OLS(dy[maxlag:], np.column_stack(columns)).fit().ssr
        penalty = 2 if method == "aic" else np.log(n)
        value = n * np.log(ssr / n) + len(columns) * penalty
        if value <= best[0]:
            best = (value, p)
    return best[1]


@pytest.mark.parametrize("method", ["aic", "bic"])
def test_llc_with_panel_specific_lag_orders(panel, method):
    chosen = [_panel_lag_choice(y, "constant", method, 4) for y in series_of(panel)]
    assert len(set(chosen)) > (1 if method == "aic" else 0)
    result = oe.xtunitroot(panel, "y", "id", "year", test="llc", lags=method, maxlag=4)
    attrs = result.attrs
    assert attrs["lags"] == pytest.approx(np.mean(chosen))
    assert (attrs["lags_min"], attrs["lags_max"]) == (min(chosen), max(chosen))
    assert attrs["lag_method"] == method
    assert min(chosen) >= 1
    expected = _llc_by_hand(panel, chosen, "constant", 10)
    assert attrs["unadjusted_t"] == pytest.approx(expected[0], rel=1e-8)
    assert attrs["statistic"] == pytest.approx(expected[1], rel=1e-8)
    assert attrs["t_tilde"] == pytest.approx(40 - np.mean(chosen) - 1)


def test_llc_kernels_demean_and_null_distribution(panel):
    base = oe.xtunitroot(panel, "y", "id", "year", test="llc", lags=1)
    for kernel in ("parzen", "quadratic_spectral"):
        other = oe.xtunitroot(panel, "y", "id", "year", test="llc", lags=1, kernel=kernel)
        assert other.attrs["unadjusted_t"] == pytest.approx(base.attrs["unadjusted_t"])
        assert other.attrs["s_n"] != pytest.approx(base.attrs["s_n"])
        assert other.attrs["kernel"] == kernel
    centred = panel.assign(y=panel["y"] - panel.groupby("year")["y"].transform("mean"))
    for test in ("llc", "ips", "fisher", "hadri", "breitung", "ht"):
        demeaned = oe.xtunitroot(panel, "y", "id", "year", test=test, demean=True)
        direct = oe.xtunitroot(centred, "y", "id", "year", test=test)
        assert demeaned.attrs["statistic"] == pytest.approx(direct.attrs["statistic"], rel=1e-9)
        assert demeaned.attrs["demean"] is True
    # Under the null with many panels the statistic of the no-deterministics model and of
    # the null-consistent long-run variance are approximately standard normal.
    rng = np.random.default_rng(2)
    draws = {"none": [], "constant": [], "trend": []}
    n, periods = 120, 51
    index = pd.DataFrame({"id": np.repeat(np.arange(n), periods),
                          "year": np.tile(np.arange(periods), n)})
    for _ in range(40):
        frame = index.assign(y=rng.standard_normal((n, periods)).cumsum(axis=1).ravel())
        for trend in draws:
            draws[trend].append(oe.xtunitroot(frame, "y", "id", "year", test="llc",
                                              trend=trend, lrv="null").attrs["statistic"])
    for trend, values in draws.items():
        assert abs(np.mean(values)) < 0.6, trend
        assert 0.7 < np.std(values) < 1.35, trend


# ---- Im-Pesaran-Shin and Fisher --------------------------------------------------------


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("lags", [0, 3])
def test_ips_matches_statsmodels_statistics_and_table_moments(panel, trend, lags):
    statistics = [adfuller(y, maxlag=lags, regression=SM_CASE[trend], autolag=None)[0]
                  for y in series_of(panel)]
    nobs = 40 - lags - 1
    mean, variance = critical.ips_moments(trend, lags, nobs)
    t_bar = np.mean(statistics)
    w = np.sqrt(12) * (t_bar - mean) / np.sqrt(variance)
    result = oe.xtunitroot(panel, "y", "id", "year", test="ips", lags=lags, trend=trend)
    attrs = result.attrs
    assert attrs["t_bar"] == pytest.approx(t_bar, rel=1e-9)
    assert attrs["statistic"] == pytest.approx(w, rel=1e-9)
    assert attrs["p_value"] == pytest.approx(stats.norm.cdf(w), rel=1e-9)
    assert attrs["mean_adjustment"] == pytest.approx(mean)
    assert attrs["variance_adjustment"] == pytest.approx(variance)
    assert list(result.index) == ["t_bar", "w_t_bar"]
    assert attrs["h0"] == "All panels contain unit roots"


def test_ips_and_fisher_on_an_unbalanced_panel_with_automatic_lags(panel):
    rng = np.random.default_rng(9)
    keep = np.ones(len(panel), dtype=bool)
    for i, unit in enumerate(sorted(panel["id"].unique())):
        cut = int(rng.integers(0, 12)) if i % 2 else 0
        keep &= ~((panel["id"] == unit) & (panel["year"] < 1980 + cut))
    unbalanced = panel[keep].sample(frac=1.0, random_state=4)
    groups = series_of(unbalanced)
    assert len({len(y) for y in groups}) > 2
    chosen = [_panel_lag_choice(y, "constant", "bic", 3) for y in groups]
    statistics, means, variances, p_values = [], [], [], []
    for y, p in zip(groups, chosen, strict=True):
        fit = adfuller(y, maxlag=p, regression="c", autolag=None)
        statistics.append(fit[0])
        p_values.append(fit[1])
        mean, variance = critical.ips_moments("constant", p, len(y) - p - 1)
        means.append(mean)
        variances.append(variance)
    count = len(groups)
    w = np.sqrt(count) * (np.mean(statistics) - np.mean(means)) / np.sqrt(np.mean(variances))
    result = oe.xtunitroot(unbalanced, "y", "id", "year", test="ips", lags="bic", maxlag=3)
    assert result.attrs["balanced"] is False
    assert result.attrs["statistic"] == pytest.approx(w, rel=1e-8)
    assert result.attrs["lags"] == pytest.approx(np.mean(chosen))
    assert result.attrs["periods"] == pytest.approx(np.mean([len(y) for y in groups]))
    fisher = oe.xtunitroot(unbalanced, "y", "id", "year", test="fisher", lags="bic", maxlag=3)
    assert fisher.loc["P", "statistic"] == pytest.approx(-2 * np.sum(np.log(p_values)), rel=1e-8)
    for test in ("llc", "hadri", "breitung", "ht"):
        assert code(oe.xtunitroot, unbalanced, "y", "id", "year", test=test) == "unbalanced_panel"


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
def test_fisher_combinations_match_scipy(panel, trend):
    count, lags = 12, 2
    p = np.array([adfuller(y, maxlag=lags, regression=SM_CASE[trend], autolag=None)[1]
                  for y in series_of(panel)])
    result = oe.xtunitroot(panel, "y", "id", "year", test="fisher", lags=lags, trend=trend)
    chi2 = -2 * np.log(p).sum()
    z = stats.norm.ppf(p).sum() / np.sqrt(count)
    k = 3 * (5 * count + 4) / (np.pi ** 2 * count * (5 * count + 2))
    l_star = np.sqrt(k) * np.log(p / (1 - p)).sum()
    modified = -(np.log(p) + 1).sum() / np.sqrt(count)
    assert list(result.index) == ["P", "Z", "L*", "Pm"]
    assert_allclose(result["statistic"], [chi2, z, l_star, modified], rtol=1e-7)
    assert_allclose(result["p_value"],
                    [stats.chi2.sf(chi2, 2 * count), stats.norm.cdf(z),
                     stats.t.cdf(l_star, 5 * count + 4), stats.norm.sf(modified)], rtol=1e-7)
    assert result.loc["P", "df"] == 2 * count and result.loc["L*", "df"] == 5 * count + 4
    assert modified == pytest.approx((chi2 - 2 * count) / (2 * np.sqrt(count)))
    assert result.attrs["statistic"] == pytest.approx(chi2, rel=1e-7)
    assert result.attrs["method"] == "dfuller"


def test_fisher_phillips_perron(panel):
    lags = 3
    p_values = []
    for y in series_of(panel):
        n = len(y) - 1
        ols = sm.OLS(y[1:], np.column_stack([y[:-1], np.ones(n)])).fit()
        u = ols.resid
        gamma0, lambda2, s2 = u @ u / n, bartlett_lrv(u, lags), u @ u / (n - 2)
        z_t = np.sqrt(gamma0 / lambda2) * (ols.params[0] - 1) / ols.bse[0] \
            - 0.5 * (lambda2 - gamma0) * n * ols.bse[0] / np.sqrt(lambda2 * s2)
        p_values.append(adfvalues.mackinnonp(z_t, regression="c"))
    result = oe.xtunitroot(panel, "y", "id", "year", test="fisher", method="pperron", lags=lags)
    assert result.loc["P", "statistic"] == pytest.approx(-2 * np.sum(np.log(p_values)), rel=1e-7)
    single = panel[panel["id"] == "u03"]
    assert adfvalues.mackinnonp(oe.pperron(single, "y", lags=lags).attrs["z_t"],
                                regression="c") == pytest.approx(p_values[3], rel=1e-8)
    assert code(oe.xtunitroot, panel, "y", "id", "year", test="fisher", method="pperron",
                lags="aic") == "invalid_lags"
    assert code(oe.xtunitroot, panel, "y", "id", "year", test="fisher", method="pperron",
                lags=60) == "invalid_lags"


# ---- Hadri and Breitung ---------------------------------------------------------------


@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_hadri_matches_hand_computation(panel, trend):
    periods, count = 40, 12
    terms = 1 if trend == "constant" else 2
    partial, variances, long_run = [], [], []
    for y in series_of(panel):
        c = np.column_stack(deterministic(periods, trend))
        e = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
        partial.append(np.sum(np.cumsum(e) ** 2) / periods ** 2)
        variances.append(e @ e / (periods - terms))
        long_run.append(bartlett_lrv(e, 4))                  # Stata: divisor T
    mean, variance = (1 / 6, 1 / 45) if trend == "constant" else (1 / 15, 11 / 6300)

    def z(lm):
        return np.sqrt(count) * (lm - mean) / np.sqrt(variance)

    plain = oe.xtunitroot(panel, "y", "id", "year", test="hadri", trend=trend)
    lm = np.mean(partial) / np.mean(variances)
    assert plain.attrs["lm"] == pytest.approx(lm, rel=1e-9)
    assert plain.attrs["statistic"] == pytest.approx(z(lm), rel=1e-9)
    assert plain.attrs["p_value"] == pytest.approx(stats.norm.sf(z(lm)), rel=1e-6, abs=1e-300)
    robust = oe.xtunitroot(panel, "y", "id", "year", test="hadri", trend=trend, robust=True)
    assert robust.attrs["statistic"] == pytest.approx(
        z(np.mean(np.array(partial) / np.array(variances))), rel=1e-9)
    kernel = oe.xtunitroot(panel, "y", "id", "year", test="hadri", trend=trend, kernel_lags=4)
    assert kernel.attrs["statistic"] == pytest.approx(z(np.mean(partial) / np.mean(long_run)),
                                                     rel=1e-9)
    assert kernel.attrs["kernel"] == "bartlett" and plain.attrs["kernel"] is None
    assert list(plain.index) == ["lm", "z"] and plain.attrs["h0"] == "All panels are stationary"
    # Random walks are rejected; stationary panels are not.
    assert plain.attrs["p_value"] < 1e-6
    calm = oe.xtunitroot(make_panel(seed=8, rho=0.0), "y", "id", "year", test="hadri",
                         kernel_lags=2)
    assert calm.attrs["p_value"] > 0.01


def _breitung_by_hand(frame, lags, trend):
    """Stata's Methods and formulas for xtunitroot breitung, one panel at a time."""
    num = den = 0.0
    for y in series_of(frame):
        total = len(y)
        dy = np.diff(y)
        n = total - 1 - lags
        d = dy[lags:]
        z = [dy[lags - j:total - 1 - j] for j in range(1, lags + 1)]
        if trend == "trend":
            level = y[lags:].copy()                          # y_{t-1} and the last level
            if lags:
                gamma = sm.OLS(d, np.column_stack([*z, np.ones(n)])).fit().params[:lags]
                d = d - sum(gamma[j - 1] * z[j - 1] for j in range(1, lags + 1))
                level = level - sum(gamma[j - 1] * y[lags - j:total - j]
                                    for j in range(1, lags + 1))
            s2 = np.var(d, ddof=1)
            y_star, x_star = np.empty(n - 1), np.empty(n - 1)
            for i in range(1, n):                            # s = 1..n-1
                y_star[i - 1] = np.sqrt((n - i) / (n - i + 1)) * (d[i - 1] - d[i:].mean())
                x_star[i - 1] = level[i - 1] - level[0] - (i - 1) * d.mean()
        else:
            level = y[lags:total - 1].copy()
            if trend == "constant":
                level = level - level[0]
            if lags:
                zz = np.column_stack(z)
                d = sm.OLS(d, zz).fit().resid
                level = sm.OLS(level, zz).fit().resid
            s2 = d @ d / (n - 1)
            y_star, x_star = d, level
        num += x_star @ y_star / s2
        den += x_star @ x_star / s2
    return num / np.sqrt(den)


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("lags", [0, 2])
def test_breitung_matches_hand_computation(panel, trend, lags):
    expected = _breitung_by_hand(panel, lags, trend)
    result = oe.xtunitroot(panel, "y", "id", "year", test="breitung", lags=lags, trend=trend)
    assert result.attrs["statistic"] == pytest.approx(expected, rel=1e-8)
    assert result.attrs["p_value"] == pytest.approx(stats.norm.cdf(expected), rel=1e-8)
    assert list(result.index) == ["lambda"]


def test_breitung_orthogonalization_is_unbiased_under_the_null():
    # The transformed pairs have zero covariance under a random walk with drift, so the
    # statistic is centred without any tabulated adjustment.
    rng = np.random.default_rng(6)
    n, periods = 80, 30
    index = pd.DataFrame({"id": np.repeat(np.arange(n), periods),
                          "year": np.tile(np.arange(periods), n)})
    values = []
    for _ in range(60):
        y = (rng.standard_normal((n, periods)) + rng.normal(size=(n, 1))).cumsum(axis=1)
        values.append(oe.xtunitroot(index.assign(y=y.ravel()), "y", "id", "year",
                                    test="breitung", trend="trend").attrs["statistic"])
    assert abs(np.mean(values)) < 0.4 and 0.75 < np.std(values) < 1.3


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("altt", [False, True])
def test_harris_tzavalis_matches_hand_computation(panel, trend, altt):
    count, t = 12, 39 if altt else 40                # Stata: T = periods; altt: T - 1
    num = den = 0.0
    for y in series_of(panel):
        current, lagged = y[1:], y[:-1]
        columns = deterministic(39, trend)
        if columns:
            c = np.column_stack(columns)
            current = current - c @ np.linalg.lstsq(c, current, rcond=None)[0]
            lagged = lagged - c @ np.linalg.lstsq(c, lagged, rcond=None)[0]
        num += lagged @ current
        den += lagged @ lagged
    rho = num / den
    mean, variance = {
        "none": (1.0, 2 / (t * (t - 1))),
        "constant": (1 - 3 / (t + 1),
                     3 * (17 * t ** 2 - 20 * t + 17) / (5 * (t - 1) * (t + 1) ** 3)),
        "trend": (1 - 15 / (2 * (t + 2)),
                  15 * (193 * t ** 2 - 728 * t + 1147) / (112 * (t + 2) ** 3 * (t - 2))),
    }[trend]
    z = np.sqrt(count) * (rho - mean) / np.sqrt(variance)
    result = oe.xtunitroot(panel, "y", "id", "year", test="ht", trend=trend, altt=altt)
    assert result.attrs["altt"] is altt and result.attrs["moment_periods"] == t
    assert result.attrs["rho"] == pytest.approx(rho, rel=1e-9)
    assert result.attrs["statistic"] == pytest.approx(z, rel=1e-8)
    assert result.attrs["p_value"] == pytest.approx(stats.norm.cdf(z), rel=1e-8)
    assert list(result.index) == ["rho", "z"]
    assert code(oe.xtunitroot, panel, "y", "id", "year", test="ht", lags=1) == "invalid_lags"


def test_harris_tzavalis_moments_agree_with_simulation():
    # The published mean and variance of the pooled coefficient are exact for fixed T,
    # with T the observations per panel in the regression (altt=True).
    rng = np.random.default_rng(14)
    n, levels, reps = 1500, 8, 400
    index = pd.DataFrame({"id": np.repeat(np.arange(n), levels),
                          "year": np.tile(np.arange(levels), n)})
    for trend in ("constant", "trend"):
        values = [oe.xtunitroot(index.assign(
            y=(rng.standard_normal((n, levels)) + 0.5).cumsum(axis=1).ravel()
            if trend == "trend" else rng.standard_normal((n, levels)).cumsum(axis=1).ravel()),
            "y", "id", "year", test="ht", trend=trend, altt=True).attrs["statistic"]
            for _ in range(reps)]
        assert abs(np.mean(values)) < 0.2, trend
        assert 0.85 < np.std(values) < 1.15, trend


# ---- contract --------------------------------------------------------------------------


def test_panel_results_are_tables_and_stationary_panels_reject(panel):
    stationary = make_panel(seed=11, rho=0.3)
    for test in ("llc", "ips", "fisher", "breitung"):
        result = oe.xtunitroot(stationary, "y", "id", "year", test=test, lags=1)
        assert isinstance(result, DataFrame)
        assert result.attrs["p_value"] < 0.01, test
        assert json.loads(json.dumps(result.attrs)) == result.attrs
        assert "\\begin{tabular}" in str(result.to_latex())
        for key in ("statistic", "p_value", "distribution", "n_panels", "periods", "lags",
                    "trend", "label", "notes", "h0", "ha", "test"):
            assert key in result.attrs
        assert oe.xtunitroot(panel, "y", "id", "year", test=test, lags=1).attrs["p_value"] > 0.01
    dated = panel.assign(date=pd.to_datetime(panel["year"].astype(str) + "-01-01"))
    assert oe.xtunitroot(dated, "y", "id", "date", test="ips").attrs["statistic"] \
        == pytest.approx(oe.xtunitroot(panel, "y", "id", "year", test="ips").attrs["statistic"])


def test_panel_errors(panel):
    def run(**kwargs):
        return code(oe.xtunitroot, panel, "y", "id", "year", **kwargs)

    assert run(test="pesaran") == "invalid_option"
    assert run(trend="drift") == "invalid_option"
    assert run(kernel="tukey") == "invalid_option"
    assert run(lags="hq") == "invalid_option"
    assert run(lags=-1) == "invalid_lags"
    assert run(lags=1, maxlag=3) == "invalid_option"
    assert run(test="ips", trend="none") == "invalid_option"
    assert run(test="hadri", trend="none") == "invalid_option"
    assert run(test="ips", lags=9) == "ips_moments_unavailable"
    assert run(demean=1) == "invalid_option"
    assert run(test="hadri", lags=2) == "invalid_lags"
    assert run(test="ips", kernel_lags=3) == "invalid_option"
    assert run(test="breitung", kernel="parzen") == "invalid_option"
    assert run(test="llc", method="pperron") == "invalid_option"
    assert run(test="llc", robust=True) == "invalid_option"
    assert run(test="ips", lrv="null") == "invalid_option"
    assert run(lrv="other") == "invalid_option"
    assert run(lags=25) == "insufficient_observations"
    assert code(oe.xtunitroot, panel, "y", "id", "missing") == "missing_columns"
    assert code(oe.xtunitroot, panel, ["y"], "id", "year") == "invalid_spec"
    assert code(oe.xtunitroot, panel[panel["id"] == "u01"], "y", "id", "year") \
        == "insufficient_panels"
    assert code(oe.xtunitroot, panel[panel["year"] < 1984], "y", "id", "year") \
        == "insufficient_observations"
    assert code(oe.xtunitroot, pd.concat([panel, panel.iloc[:1]]), "y", "id", "year") \
        == "repeated_time_values"
    assert code(oe.xtunitroot, panel.drop(index=45), "y", "id", "year") == "time_gaps"
    holes = panel.copy()
    holes.loc[7, "y"] = np.nan
    assert code(oe.xtunitroot, holes, "y", "id", "year") == "missing_values"
    flat = panel.copy()
    flat.loc[flat["id"] == "u02", "y"] = 1.0
    assert code(oe.xtunitroot, flat, "y", "id", "year", test="ips") in {"collinear_regressors",
                                                                       "perfect_fit"}
    assert code(oe.xtunitroot, flat, "y", "id", "year", test="hadri") == "constant_series"
    assert code(oe.xtunitroot, panel.assign(year="x"), "y", "id", "year") == "invalid_time"


# ---- Kao --------------------------------------------------------------------------------


def make_coint_panel(seed=21, n=15, periods=45, cointegrated=True):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal((n, periods)).cumsum(axis=1)
    x2 = rng.standard_normal((n, periods)).cumsum(axis=1)
    noise = rng.standard_normal((n, periods))
    u = noise if cointegrated else noise.cumsum(axis=1)
    y = rng.normal(size=(n, 1)) * 2 + 0.7 * x1 - 0.4 * x2 + u
    return pd.DataFrame({"id": np.repeat(np.arange(n), periods),
                         "year": np.tile(np.arange(periods), n), "y": y.ravel(),
                         "x1": x1.ravel(), "x2": x2.ravel()})


def _kao_by_hand(frame, names, lags, kernel_lags):
    frame = frame.sort_values(["id", "year"])
    count, periods = frame["id"].nunique(), frame["year"].nunique()
    dummies = pd.get_dummies(frame["id"]).to_numpy(dtype=float)
    lsdv = sm.OLS(frame["y"].to_numpy(), np.column_stack([frame[names].to_numpy(),
                                                          dummies])).fit()
    e = lsdv.resid.reshape(count, periods)
    lagged, current = e[:, :-1].ravel(), e[:, 1:].ravel()
    rho = lagged @ current / (lagged @ lagged)
    s2 = np.sum((current - rho * lagged) ** 2) / len(current)
    t_rho = (rho - 1) * np.sqrt(lagged @ lagged) / np.sqrt(s2)
    de = np.diff(e, axis=1)
    columns = [e[:, lags:periods - 1].ravel()] + [de[:, lags - j:periods - 1 - j].ravel()
                                                  for j in range(1, lags + 1)]
    t_adf = sm.OLS(de[:, lags:].ravel(), np.column_stack(columns)).fit().tvalues[0]
    levels = frame[["y", *names]].to_numpy().reshape(count, periods, -1)
    w = np.diff(levels, axis=1)
    m = count * (periods - 1)
    sigma = sum(w[i].T @ w[i] for i in range(count)) / m
    omega = sigma.copy()
    for i in range(count):
        for j in range(1, kernel_lags + 1):
            cross = w[i][j:].T @ w[i][:-j]
            omega += (1 - j / (kernel_lags + 1)) * (cross + cross.T) / m

    def conditional(a):
        return a[0, 0] - a[0, 1:] @ np.linalg.solve(a[1:, 1:], a[1:, 0])

    s2_v, s2_0v = conditional(sigma), conditional(omega)
    ratio = s2_v / s2_0v
    bias = np.sqrt(count) * periods * (rho - 1)
    scale = np.sqrt(s2_0v / (2 * s2_v) + 3 * s2_v / (10 * s2_0v))
    shift = np.sqrt(6 * count) * np.sqrt(ratio) / 2
    return {
        "modified_df": (bias + 3 * np.sqrt(count) * ratio) / np.sqrt(3 + 36 * ratio ** 2 / 5),
        "df": (t_rho + shift) / scale, "adf": (t_adf + shift) / scale,
        "unadjusted_modified_df": (bias + 3 * np.sqrt(count)) / np.sqrt(10.2),
        "unadjusted_df": np.sqrt(1.25) * t_rho + np.sqrt(1.875 * count),
    }, lsdv.params[:len(names)], rho


@pytest.mark.parametrize("lags,kernel_lags", [(1, None), (0, 0), (3, 5)])
def test_kao_matches_hand_computation(lags, kernel_lags):
    frame = make_coint_panel(cointegrated=False)
    used = int(4 * (45 / 100) ** (2 / 9)) if kernel_lags is None else kernel_lags
    expected, beta, rho = _kao_by_hand(frame, ["x1", "x2"], lags, used)
    result = oe.xtcointtest(frame.sample(frac=1.0, random_state=0), "y", ["x1", "x2"], "id",
                            "year", lags=lags, kernel_lags=kernel_lags)
    assert list(result.index) == list(expected)
    assert_allclose(result["statistic"], list(expected.values()), rtol=1e-7)
    assert_allclose(result["p_value"], stats.norm.cdf(list(expected.values())), rtol=1e-6)
    attrs = result.attrs
    assert attrs["statistic"] == pytest.approx(expected["adf"], rel=1e-7)
    assert attrs["rho"] == pytest.approx(rho, rel=1e-9)
    assert attrs["cointegrating_vector"] == pytest.approx({"x1": beta[0], "x2": beta[1]},
                                                          rel=1e-8)
    assert (attrs["n_panels"], attrs["periods"], attrs["kernel_lags"]) == (15, 45, used)
    assert json.loads(json.dumps(attrs)) == attrs


def test_kao_power_options_and_errors():
    related = make_coint_panel(cointegrated=True)
    unrelated = make_coint_panel(cointegrated=False)
    assert oe.xtcointtest(related, "y", ["x1", "x2"], "id", "year").attrs["p_value"] < 1e-6
    assert oe.xtcointtest(unrelated, "y", ["x1", "x2"], "id", "year").attrs["p_value"] > 0.01
    frame = related.assign(dup=related["x1"] * 2, level=related["id"] * 1.0)
    result = oe.xtcointtest(frame, "y", ["x1", "dup", "level", "x2"], "id", "year")
    assert result.attrs["omitted"] == ["dup", "level"]
    assert_allclose(result["statistic"],
                    oe.xtcointtest(related, "y", ["x1", "x2"], "id", "year")["statistic"],
                    rtol=1e-8)
    centred = related.copy()
    for name in ("y", "x1", "x2"):
        centred[name] -= centred.groupby("year")[name].transform("mean")
    assert_allclose(
        oe.xtcointtest(related, "y", ["x1", "x2"], "id", "year", demean=True)["statistic"],
        oe.xtcointtest(centred, "y", ["x1", "x2"], "id", "year")["statistic"], rtol=1e-8)
    other = oe.xtcointtest(related, "y", ["x1"], "id", "year", kernel="parzen", kernel_lags=4)
    assert other.attrs["kernel"] == "parzen" and isinstance(other, DataFrame)
    assert "\\begin{tabular}" in str(other.to_latex())

    def run(*args, **kwargs):
        return code(oe.xtcointtest, *args, **kwargs)

    assert oe.xtcointtest(related, "y", ["x1"], "id", "year", test="pedroni").attrs["test"] == "pedroni"
    assert oe.xtcointtest(related, "y", ["x1"], "id", "year", test="westerlund").attrs["test"] == "westerlund"
    assert run(related, "y", [], "id", "year") == "invalid_spec"
    assert run(related, "y", "x1", "id", "year") == "invalid_spec"
    assert run(related, "y", ["x1"], "id", "year", lags=-1) == "invalid_lags"
    assert run(related, "y", ["x1"], "id", "year", kernel_lags=60) == "invalid_lags"
    assert run(related, "y", ["x1"], "id", "year", kernel="tukey") == "invalid_option"
    assert run(related.iloc[:-3], "y", ["x1"], "id", "year") == "unbalanced_panel"
    assert run(related[related["id"] == 0], "y", ["x1"], "id", "year") == "insufficient_panels"
    assert run(frame, "y", ["level"], "id", "year") == "collinear_regressors"
    assert run(related[related["year"] < 5], "y", ["x1"], "id", "year") \
        == "insufficient_observations"
    holes = related.copy()
    holes.loc[4, "x1"] = np.nan
    assert run(holes, "y", ["x1"], "id", "year") == "missing_values"


def test_panel_tests_scale_to_a_million_rows():
    rng = np.random.default_rng(0)
    n, periods = 10_000, 100
    y = rng.standard_normal((n, periods)).cumsum(axis=1)
    x = rng.standard_normal((n, periods)).cumsum(axis=1)
    frame = pd.DataFrame({"id": np.repeat(np.arange(n), periods),
                          "year": np.tile(np.arange(periods), n), "y": y.ravel(),
                          "x": x.ravel()})
    started = time.perf_counter()
    for test in ("llc", "ips", "fisher", "breitung"):
        result = oe.xtunitroot(frame, "y", "id", "year", test=test, lags=2)
        assert math.isfinite(result.attrs["statistic"])
    for test in ("hadri", "ht"):
        result = oe.xtunitroot(frame, "y", "id", "year", test=test)
        assert math.isfinite(result.attrs["statistic"])
    assert math.isfinite(oe.xtcointtest(frame, "y", ["x"], "id", "year").attrs["statistic"])
    assert time.perf_counter() - started < 40
