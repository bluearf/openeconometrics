"""Independent oracles for the single-series unit-root and stationarity tests.

oe.dfuller, oe.pperron, oe.dfgls and oe.kpss are compared with statsmodels
(adfuller, kpss), with explicit NumPy algebra written from the published
formulas, and the typed tables with statsmodels' transcription of MacKinnon's
tables and with Monte Carlo draws.
"""

import io
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.tsa.adfvalues as adfvalues
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.stats.sandwich_covariance import S_hac_simple
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss as sm_kpss

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import TableSet
from openecon.econometrics.unitroot import critical, tables
from openecon.frame import DataFrame

SM_CASE = {"none": "n", "constant": "c", "trend": "ct"}
LEVELS = ("1%", "5%", "10%")


@pytest.fixture(scope="module")
def walk():
    rng = np.random.default_rng(20261003)
    e = rng.standard_normal(180)
    y = np.cumsum(e + 0.3 * np.r_[0.0, e[:-1]]) + 5.0
    return pd.DataFrame({"y": y, "t": np.arange(1960, 1960 + len(y))})


@pytest.fixture(scope="module")
def stationary():
    rng = np.random.default_rng(99)
    e = rng.standard_normal(240)
    y = np.zeros(240)
    for i in range(1, 240):
        y[i] = 0.5 * y[i - 1] + e[i]
    return pd.DataFrame({"y": y + 2.0, "t": np.arange(240)})


# ---- tables ------------------------------------------------------------------------


def test_mackinnon_tables_match_an_independent_transcription():
    for case, key in (("n", "n"), ("c", "c"), ("ct", "ct"), ("ctt", "ctt")):
        assert_allclose(tables.TAU_STAR[case], adfvalues._tau_stars[key])
        assert_allclose(tables.TAU_MIN[case], adfvalues._tau_mins[key])
        assert_allclose(tables.TAU_MAX[case], adfvalues._tau_maxs[key])
        small = np.array(tables.TAU_SMALL_P[case]) * np.array(tables.TAU_SMALL_SCALE)
        large = np.array(tables.TAU_LARGE_P[case]) * np.array(tables.TAU_LARGE_SCALE)
        assert_allclose(small, adfvalues._tau_smallps[key], rtol=1e-13)
        assert_allclose(large, adfvalues._tau_largeps[key], rtol=1e-13)
        rows = np.array(tables.TAU_CRIT_2010[case])
        assert_allclose(rows, adfvalues.tau_2010s[key][:len(rows)], rtol=1e-13)


@pytest.mark.parametrize("case", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("n_series", [1, 2, 4, 6])
def test_mackinnon_functions_match_statsmodels(case, n_series):
    for statistic in (-30.0, -6.2, -4.1, -3.3, -2.5, -1.4, -0.2, 0.6, 3.0):
        expected = adfvalues.mackinnonp(statistic, regression=case, N=n_series)
        assert critical.mackinnon_p(statistic, case, n_series) == pytest.approx(expected,
                                                                                abs=1e-14)
    if case != "n" or n_series == 1:
        for nobs in (25, 143, 5000):
            ours = critical.mackinnon_critical(case, n_series, nobs)
            theirs = adfvalues.mackinnoncrit(N=n_series, regression=case, nobs=nobs)
            assert_allclose([ours[level] for level in LEVELS], theirs, rtol=1e-13)
        asymptotic = critical.mackinnon_critical(case, n_series, math.inf)
        assert_allclose([asymptotic[level] for level in LEVELS],
                        adfvalues.mackinnoncrit(N=n_series, regression=case), rtol=1e-13)
    else:
        assert critical.mackinnon_critical(case, n_series, 100) is None
        implied = critical.mackinnon_asymptotic_critical(case, n_series)
        for level, target in zip(LEVELS, (0.01, 0.05, 0.10), strict=True):
            assert adfvalues.mackinnonp(implied[level], regression=case,
                                        N=n_series) == pytest.approx(target, abs=1e-9)


def _mackinnon_z_p(z, case):
    """MacKinnon (1994) p-value of the normalized-bias statistic (statsmodels' z tables)."""
    star = {"n": adfvalues.z_star_nc, "c": adfvalues.z_star_c, "ct": adfvalues.z_star_ct}[case][0]
    if z <= star:
        c = {"n": adfvalues.z_nc_smallp, "c": adfvalues.z_c_smallp,
             "ct": adfvalues.z_ct_smallp}[case][0]
        log_z = np.log(abs(z))
        return stats.norm.cdf(c[0] + c[1] * log_z + c[2] * log_z ** 2 + c[3] * log_z ** 3)
    c = {"n": adfvalues.z_nc_largep, "c": adfvalues.z_c_largep,
         "ct": adfvalues.z_ct_largep}[case][0]
    return stats.norm.cdf(np.polyval(c[::-1], z))


def test_fuller_table_is_consistent_with_mackinnon_and_simulation():
    # Asymptotic row against MacKinnon's distribution function of the z statistic.
    for case, row in tables.FULLER_RHO.items():
        for value, level in zip(row[-1], (0.01, 0.05, 0.10), strict=True):
            assert _mackinnon_z_p(value, case) == pytest.approx(level, abs=0.0015)
        # Critical values grow in magnitude with n and across deterministic cases.
        assert all(np.all(np.diff(np.array(row)[:, j]) <= 0) for j in range(3))
    # Finite-sample row n = 100 (Fuller: n(rho-1) with n-1 regression observations).
    rng = np.random.default_rng(4)
    n, reps = 100, 60000
    y = rng.standard_normal((reps, n)).cumsum(axis=1)
    lag, cur = y[:, :-1], y[:, 1:]
    lag_c, cur_c = lag - lag.mean(1, keepdims=True), cur - cur.mean(1, keepdims=True)
    rho = (lag_c * cur_c).sum(1) / (lag_c ** 2).sum(1)
    simulated = np.quantile(n * (rho - 1), [0.01, 0.05, 0.10])
    assert_allclose(simulated, tables.FULLER_RHO["c"][2], atol=0.35)
    # Stata's manual prints -27.687 / -20.872 / -17.643 for N = 143 with a trend.
    values = critical.fuller_rho_critical("ct", 143)
    assert_allclose([values[level] for level in LEVELS], [-27.687, -20.872, -17.643], atol=5e-4)
    assert critical.fuller_rho_critical("c", 10) == dict(zip(LEVELS, (-17.2, -12.5, -10.2),
                                                             strict=True))
    assert critical.fuller_rho_critical("n", 9000)["5%"] == -8.1


def test_ers_table_against_simulation_and_interpolation():
    rng = np.random.default_rng(8)
    total, reps = 100, 40000
    y = rng.standard_normal((reps, total)).cumsum(axis=1)
    alpha = 1 - 13.5 / total
    z = np.column_stack([np.ones(total), np.arange(1.0, total + 1)])
    zq, yq = z.copy(), y.copy()
    zq[1:] -= alpha * z[:-1]
    yq[:, 1:] -= alpha * y[:, :-1]
    detrended = y - (np.linalg.solve(zq.T @ zq, zq.T @ yq.T).T) @ z.T
    lag, diff = detrended[:, :-1], np.diff(detrended, axis=1)
    rho = (lag * diff).sum(1) / (lag ** 2).sum(1)
    s2 = ((diff - rho[:, None] * lag) ** 2).sum(1) / (total - 2)
    simulated = np.quantile(rho / np.sqrt(s2 / (lag ** 2).sum(1)), [0.01, 0.05, 0.10])
    assert_allclose(simulated, tables.ERS_TREND[1], atol=0.07)
    assert critical.ers_trend_critical(30) == dict(zip(LEVELS, tables.ERS_TREND[0], strict=True))
    assert critical.ers_trend_critical(75)["5%"] == pytest.approx(-3.11)
    # Stata's rule: the asymptotic row above 200 observations, no interpolation.
    assert critical.ers_trend_critical(200)["10%"] == pytest.approx(-2.64)
    assert critical.ers_trend_critical(201) == dict(zip(LEVELS, tables.ERS_TREND[3],
                                                        strict=True))
    # Stata's manual (dfgls ln_inv, 92 observations) prints -3.610 at 1%.
    assert critical.ers_trend_critical(92)["1%"] == pytest.approx(-3.6104)


# ---- dfuller ------------------------------------------------------------------------


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("lags", [0, 1, 4])
def test_dfuller_matches_statsmodels(walk, trend, lags):
    result = oe.dfuller(walk, "y", lags=lags, trend=trend, time="t")
    expected = adfuller(walk["y"].to_numpy(), maxlag=lags, regression=SM_CASE[trend],
                        autolag=None)
    attrs = result.attrs
    assert attrs["statistic"] == pytest.approx(expected[0], rel=1e-10)
    assert attrs["p_value"] == pytest.approx(expected[1], rel=1e-9, abs=1e-14)
    assert attrs["nobs"] == expected[3] == len(walk) - lags - 1
    for level in LEVELS:
        assert attrs["critical_values"][level] == pytest.approx(expected[4][level], rel=1e-12)
    assert attrs["lags"] == lags and attrs["trend"] == trend
    assert attrs["distribution"] == "Dickey-Fuller"
    assert result.loc["Z(t)", "statistic"] == attrs["statistic"]
    assert result.loc["Z(t)", "critical_5pct"] == attrs["critical_values"]["5%"]


def test_dfuller_drift_uses_student_t_and_regression_table_matches_ols(walk):
    y = walk["y"].to_numpy()
    lags = 2
    result = oe.dfuller(walk, "y", lags=lags, trend="drift", regress=True)
    assert isinstance(result, TableSet) and set(result) == {"test", "regression"}
    dy = np.diff(y)
    design = np.column_stack([y[lags:-1], dy[lags - 1:-1], dy[lags - 2:-2], np.ones(len(y) - 3)])
    ols = sm.OLS(dy[lags:], design).fit()
    table = result["regression"]
    assert list(table.index) == ["L1.y", "LD.y", "L2D.y", "Intercept"]
    assert_allclose(table["coefficient"], ols.params, rtol=1e-9)
    assert_allclose(table["std_error"], ols.bse, rtol=1e-9)
    assert_allclose(table["statistic"], ols.tvalues, rtol=1e-9)
    assert_allclose(table["p_value"], ols.pvalues, rtol=1e-8)
    assert_allclose(table[["ci_low", "ci_high"]], ols.conf_int(), rtol=1e-8)
    attrs = result.attrs
    df = int(ols.df_resid)
    assert attrs["df"] == df and attrs["distribution"] == "t"
    assert attrs["statistic"] == pytest.approx(ols.tvalues[0], rel=1e-10)
    assert attrs["p_value"] == pytest.approx(stats.t.cdf(ols.tvalues[0], df), rel=1e-10)
    assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                    stats.t.ppf([0.01, 0.05, 0.10], df), rtol=1e-10)
    assert attrs["coefficient"] == pytest.approx(ols.params[0], rel=1e-10)


def test_dfuller_trend_regression_and_time_sorting(walk):
    shuffled = walk.sample(frac=1.0, random_state=3)
    result = oe.dfuller(shuffled, "y", lags=1, trend="trend", regress=True, time="t")
    y = walk["y"].to_numpy()
    dy = np.diff(y)
    n = len(y) - 2
    # Stata's _trend is t - 1: its first value in a regression with one lag is 2.
    design = np.column_stack([y[1:-1], dy[:-1], np.arange(2.0, n + 2), np.ones(n)])
    ols = sm.OLS(dy[1:], design).fit()
    assert list(result["regression"].index) == ["L1.y", "LD.y", "_trend", "Intercept"]
    assert_allclose(result["regression"]["coefficient"], ols.params, rtol=1e-8)
    assert result.attrs["statistic"] == pytest.approx(ols.tvalues[0], rel=1e-9)
    # Without the time column the shuffled row order is a different series.
    assert oe.dfuller(shuffled, "y", lags=1).attrs["statistic"] \
        != pytest.approx(oe.dfuller(walk, "y", lags=1).attrs["statistic"])


def test_dfuller_rejects_for_a_stationary_series(stationary):
    attrs = oe.dfuller(stationary, "y", lags=1).attrs
    assert attrs["p_value"] < 1e-6 and attrs["statistic"] < attrs["critical_values"]["1%"]


# ---- pperron ------------------------------------------------------------------------


def _pperron_by_hand(y, lags, trend):
    """Stata's Methods and formulas, with statsmodels' Bartlett long-run variance."""
    n = len(y) - 1
    columns = [y[:-1]]
    if trend == "trend":
        columns.append(np.arange(1.0, n + 1))
    if trend != "none":
        columns.append(np.ones(n))
    ols = sm.OLS(y[1:], np.column_stack(columns)).fit()
    rho, sigma, u = ols.params[0], ols.bse[0], ols.resid
    gamma0 = u @ u / n
    lambda2 = float(S_hac_simple(u[:, None], nlags=lags)[0, 0]) / n
    s2 = u @ u / (n - len(columns))
    z_rho = n * (rho - 1) - 0.5 * n ** 2 * sigma ** 2 / s2 * (lambda2 - gamma0)
    z_t = np.sqrt(gamma0 / lambda2) * (rho - 1) / sigma \
        - 0.5 * (lambda2 - gamma0) / np.sqrt(lambda2) * n * sigma / np.sqrt(s2)
    return z_rho, z_t, ols


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("lags", [None, 0, 6])
def test_pperron_matches_hand_computation(walk, trend, lags):
    y = walk["y"].to_numpy()
    n = len(y) - 1
    used = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
    z_rho, z_t, ols = _pperron_by_hand(y, used, trend)
    result = oe.pperron(walk, "y", lags=lags, trend=trend, regress=True)
    attrs = result.attrs
    assert attrs["lags"] == used and attrs["nobs"] == n
    assert attrs["z_rho"] == pytest.approx(z_rho, rel=1e-9)
    assert attrs["z_t"] == pytest.approx(z_t, rel=1e-9)
    assert attrs["statistic"] == attrs["z_t"]
    case = SM_CASE[trend]
    assert attrs["p_value"] == pytest.approx(adfvalues.mackinnonp(z_t, regression=case),
                                             rel=1e-9)
    assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                    adfvalues.mackinnoncrit(N=1, regression=case, nobs=n), rtol=1e-12)
    assert attrs["critical_values_rho"] == critical.fuller_rho_critical(case, n)
    test = result["test"]
    assert list(test.index) == ["Z(rho)", "Z(t)"]
    assert np.isnan(test.loc["Z(rho)", "p_value"])
    assert_allclose(result["regression"]["coefficient"], ols.params, rtol=1e-9)
    assert_allclose(result["regression"]["std_error"], ols.bse, rtol=1e-9)
    if used == 0:
        # Without Newey-West lags Z(t) is the Dickey-Fuller t statistic itself.
        assert attrs["z_t"] == pytest.approx((ols.params[0] - 1) / ols.bse[0], rel=1e-9)
        assert attrs["z_t"] == pytest.approx(
            oe.dfuller(walk, "y", trend=trend).attrs["statistic"], rel=1e-9)


# ---- dfgls --------------------------------------------------------------------------


def _dfgls_by_hand(y, maxlag, trend):
    total = len(y)
    alpha = 1 + (-13.5 if trend else -7.0) / total
    z = np.column_stack([np.ones(total), np.arange(1.0, total + 1)]) if trend \
        else np.ones((total, 1))
    yq, zq = y.copy(), z.copy()
    yq[1:] = y[1:] - alpha * y[:-1]
    zq[1:] = z[1:] - alpha * z[:-1]
    detrended = y - z @ np.linalg.lstsq(zq, yq, rcond=None)[0]
    d = np.diff(detrended)
    n = total - maxlag - 1
    out = {}
    for k in range(1, maxlag + 1):
        columns = [detrended[maxlag:-1]] + [d[maxlag - j:total - 1 - j] for j in range(1, k + 1)]
        ols = sm.OLS(d[maxlag:], np.column_stack(columns)).fit()
        s2 = ols.ssr / n
        tau = ols.params[0] ** 2 * np.sum(detrended[maxlag:-1] ** 2) / s2
        out[k] = {"statistic": ols.tvalues[0], "rmse": np.sqrt(s2), "last_t": ols.tvalues[-1],
                  "sc": np.log(s2) + (k + 1) * np.log(n) / n,
                  "maic": np.log(s2) + 2 * (tau + k) / n}
    return out


@pytest.mark.parametrize("trend", [True, False])
def test_dfgls_matches_hand_computation(walk, trend):
    y = walk["y"].to_numpy()
    maxlag = 6
    result = oe.dfgls(walk, "y", maxlag=maxlag, trend=trend)
    expected = _dfgls_by_hand(y, maxlag, trend)
    assert list(result["lags"]) == list(range(1, maxlag + 1))
    for name in ("statistic", "rmse", "sc", "maic"):
        assert_allclose(result[name], [expected[k][name] for k in expected], rtol=1e-8)
    attrs = result.attrs
    n = len(y) - maxlag - 1
    assert attrs["nobs"] == n and attrs["maxlag"] == maxlag
    assert attrs["lags_maic"] == min(expected, key=lambda k: expected[k]["maic"])
    assert attrs["lags_sc"] == min(expected, key=lambda k: expected[k]["sc"])
    significant = [k for k in expected if abs(expected[k]["last_t"]) > stats.norm.ppf(0.95)]
    assert attrs["lags_seq_t"] == (max(significant) if significant else 0)
    assert attrs["lags"] == attrs["lags_maic"]
    assert attrs["statistic"] == pytest.approx(expected[attrs["lags_maic"]]["statistic"],
                                               rel=1e-8)
    if trend:
        assert attrs["p_value"] is None
        assert attrs["critical_values"] == critical.ers_trend_critical(len(y))
        assert attrs["nobs_series"] == len(y)
    else:
        assert attrs["p_value"] == pytest.approx(
            adfvalues.mackinnonp(attrs["statistic"], regression="n"), rel=1e-9)
        assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                        adfvalues.mackinnoncrit(N=1, regression="n", nobs=n), rtol=1e-12)


def test_dfgls_default_maxlag_and_zero_lags(walk):
    total = len(walk)
    result = oe.dfgls(walk, "y")
    assert result.attrs["maxlag"] == int(12 * (total / 100) ** 0.25)
    single = oe.dfgls(walk, "y", maxlag=0)
    assert list(single["lags"]) == [0] and single.attrs["lags"] == 0
    y = walk["y"].to_numpy()
    alpha = 1 - 13.5 / total
    z = np.column_stack([np.ones(total), np.arange(1.0, total + 1)])
    yq, zq = y.copy(), z.copy()
    yq[1:], zq[1:] = y[1:] - alpha * y[:-1], z[1:] - alpha * z[:-1]
    detrended = y - z @ np.linalg.lstsq(zq, yq, rcond=None)[0]
    ols = sm.OLS(np.diff(detrended), detrended[:-1]).fit()
    assert single.attrs["statistic"] == pytest.approx(ols.tvalues[0], rel=1e-8)


# ---- kpss ---------------------------------------------------------------------------


@pytest.mark.parametrize("trend", [False, True])
def test_kpss_matches_statsmodels_for_every_truncation(walk, stationary, trend):
    for frame in (walk, stationary):
        y = frame["y"].to_numpy()
        result = oe.kpss(frame, "y", lags=7, trend=trend)
        assert list(result["lags"]) == list(range(8))
        for order, statistic, p_value in result.itertuples(index=False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = sm_kpss(y, regression="ct" if trend else "c", nlags=int(order))
            assert statistic == pytest.approx(expected[0], rel=1e-9)
            assert p_value == pytest.approx(expected[1], rel=1e-9)
        attrs = result.attrs
        assert attrs["statistic"] == result["statistic"].iloc[-1] and attrs["lags"] == 7
        assert attrs["critical_values"] == expected[3]
        assert attrs["trend"] == ("trend" if trend else "constant")


def test_kpss_default_and_automatic_bandwidth(stationary):
    y = stationary["y"].to_numpy()
    total = len(y)
    default = oe.kpss(stationary, "y")
    assert default.attrs["lags"] == int(12 * (total / 100) ** 0.25)
    assert default.attrs["bandwidth"] == "schwert" and len(default) == default.attrs["lags"] + 1
    # Hobijn, Franses and Ooms (1998): Newey-West automatic bandwidth for the Bartlett kernel.
    e = y - y.mean()
    n = int(4 * (total / 100) ** (2 / 9))
    gamma = [e[j:] @ e[:total - j] / total for j in range(n + 1)]
    s0 = gamma[0] + 2 * sum(gamma[1:])
    s1 = 2 * sum(j * gamma[j] for j in range(1, n + 1))
    bandwidth = min(total - 1, int(1.1447 * ((s1 / s0) ** 2) ** (1 / 3) * total ** (1 / 3)))
    auto = oe.kpss(stationary, "y", auto=True)
    assert auto.attrs["lags"] == bandwidth and auto.attrs["bandwidth"] == "auto"
    assert len(auto) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = sm_kpss(y, regression="c", nlags=bandwidth)
    assert auto.attrs["statistic"] == pytest.approx(expected[0], rel=1e-9)
    assert 0.01 <= auto.attrs["p_value"] <= 0.10


def test_kpss_p_value_interpolation_and_bounds():
    low, note_low = critical.kpss_p(0.2, "level")
    high, note_high = critical.kpss_p(1.5, "level")
    assert low == 0.10 and "above 0.10" in note_low
    assert high == 0.01 and "below 0.01" in note_high
    middle, note = critical.kpss_p(0.5185, "level")
    assert note is None and middle == pytest.approx(0.0375)
    assert critical.kpss_p(0.146, "trend")[0] == pytest.approx(0.05)


# ---- contract -----------------------------------------------------------------------


def test_results_are_tables_with_json_safe_attrs_and_latex(walk):
    results = {
        "dfuller": oe.dfuller(walk, "y", lags=2),
        "pperron": oe.pperron(walk, "y"),
        "dfgls": oe.dfgls(walk, "y", maxlag=4),
        "kpss": oe.kpss(walk, "y", lags=3),
    }
    for name, result in results.items():
        assert isinstance(result, DataFrame)
        attrs = json.loads(json.dumps(result.attrs))
        assert attrs == result.attrs
        assert attrs["test"] == name and attrs["series"] == "y"
        for key in ("statistic", "p_value", "distribution", "critical_values", "lags",
                    "trend", "nobs", "label", "notes"):
            assert key in attrs
        assert math.isfinite(attrs["statistic"])
        latex = str(result.to_latex())
        assert "\\begin{tabular}" in latex and "statistic" in latex
        assert pd.read_json(io.StringIO(result.to_json())).shape == result.shape
    both = oe.dfuller(walk, "y", regress=True)
    assert json.loads(json.dumps(both.attrs)) == both.attrs
    assert "L1.y" in str(both) and "\\begin{tabular}" in both.to_latex()


def test_public_names_are_exported_without_estimators():
    exports = registry.public_exports()
    for name in ("dfuller", "dfgls", "pperron", "kpss", "zandrews", "egranger", "xtunitroot",
                 "xtcointtest", "chow", "sbsingle", "cusum"):
        assert exports[name][0].startswith("openecon.econometrics.unitroot.")
        assert callable(getattr(oe, name)) and name in dir(oe)
        assert len(getattr(oe, name).__doc__) > 400
    import openecon.econometrics.unitroot as family
    assert family.ESTIMATORS == ()


def test_inputs_as_mapping_and_records(walk):
    expected = oe.dfuller(walk, "y", lags=1).attrs["statistic"]
    assert oe.dfuller({"y": walk["y"].tolist()}, "y", lags=1).attrs["statistic"] \
        == pytest.approx(expected)
    records = walk.to_dict("records")
    assert oe.dfuller(records, "y", lags=1, time="t").attrs["statistic"] == pytest.approx(expected)
    dated = walk.assign(date=pd.date_range("2001-01-01", periods=len(walk), freq="MS"))
    assert oe.pperron(dated.sample(frac=1, random_state=1), "y", time="date").attrs["z_t"] \
        == pytest.approx(oe.pperron(walk, "y").attrs["z_t"])


@pytest.mark.parametrize("function", [oe.dfuller, oe.pperron, oe.dfgls, oe.kpss])
def test_data_errors(function, walk):
    def code(*args, **kwargs):
        with pytest.raises(AnalysisError) as error:
            function(*args, **kwargs)
        return error.value.code

    assert code(walk, "missing") == "missing_columns"
    assert code(walk, ["y"]) == "invalid_spec"
    assert code(walk.iloc[:0], "y") == "empty_data"
    assert code([1, 2, 3], "y") == "invalid_data"
    holes = walk.copy()
    holes.loc[10, "y"] = np.nan
    assert code(holes, "y") == "missing_values"
    assert code(walk.assign(y="a"), "y") == "non_numeric_column"
    assert code(walk.assign(y=1.5), "y") == "constant_series"
    assert code(walk.iloc[:3], "y") == "insufficient_observations"
    assert code(walk.drop(index=20), "y", time="t") == "time_gaps"
    assert code(pd.concat([walk, walk.iloc[:1]]), "y", time="t") == "repeated_time_values"
    assert code(walk.assign(t=walk["t"] + 0.5), "y", time="t") == "invalid_time"
    assert code(walk.assign(t="x"), "y", time="t") == "invalid_time"
    assert code(walk, "y", time="y") == "invalid_spec"
    infinite = walk.copy()
    infinite.loc[3, "y"] = np.inf
    assert code(infinite, "y") == "non_finite_values"


def test_option_errors(walk):
    def code(function, *args, **kwargs):
        with pytest.raises(AnalysisError) as error:
            function(*args, **kwargs)
        return error.value.code

    assert code(oe.dfuller, walk, "y", trend="quadratic") == "invalid_option"
    assert code(oe.dfuller, walk, "y", lags=-1) == "invalid_lags"
    assert code(oe.dfuller, walk, "y", lags=1.5) == "invalid_lags"
    assert code(oe.dfuller, walk, "y", lags=True) == "invalid_lags"
    assert code(oe.dfuller, walk, "y", regress="yes") == "invalid_option"
    assert code(oe.dfuller, walk, "y", lags=120) == "insufficient_observations"
    assert code(oe.pperron, walk, "y", trend="drift") == "invalid_option"
    assert code(oe.pperron, walk, "y", lags=500) == "invalid_lags"
    assert code(oe.dfgls, walk, "y", trend="yes") == "invalid_option"
    assert code(oe.dfgls, walk, "y", maxlag=100) == "invalid_lags"
    assert code(oe.kpss, walk, "y", lags=3, auto=True) == "invalid_option"
    assert code(oe.kpss, walk, "y", lags=len(walk)) == "invalid_lags"
    assert code(oe.kpss, walk, "y", trend=1) == "invalid_option"
    line = pd.DataFrame({"y": np.arange(40.0)})
    assert code(oe.dfuller, line, "y", trend="trend") in {"collinear_regressors", "perfect_fit"}
    assert code(oe.kpss, line, "y", trend=True) == "constant_series"


def test_large_series_run_quickly():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": rng.standard_normal(1_000_000).cumsum()})
    started = time.perf_counter()
    assert math.isfinite(oe.dfuller(frame, "y", lags=8, trend="trend").attrs["statistic"])
    assert math.isfinite(oe.pperron(frame, "y").attrs["z_t"])
    assert math.isfinite(oe.dfgls(frame, "y", maxlag=8).attrs["statistic"])
    assert math.isfinite(oe.kpss(frame, "y", lags=20).attrs["statistic"])
    assert time.perf_counter() - started < 20
