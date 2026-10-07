"""Independent oracles for the break and cointegration tests of the unitroot family.

oe.zandrews is compared with a brute-force NumPy scan and with statsmodels'
zivot_andrews; oe.egranger with statsmodels' coint and explicit OLS; oe.chow,
oe.sbsingle and oe.cusum with explicit split-sample regressions and
statsmodels' recursive residuals.
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
from scipy import optimize, stats
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tsa.stattools import adfuller, coint, zivot_andrews

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.unitroot import critical, tables
from openecon.frame import DataFrame

LEVELS = ("1%", "5%", "10%")


def code(function, *args, **kwargs):
    with pytest.raises(AnalysisError) as error:
        function(*args, **kwargs)
    return error.value.code


@pytest.fixture(scope="module")
def broken():
    rng = np.random.default_rng(31)
    total = 150
    t = np.arange(total)
    y = np.zeros(total)
    e = rng.standard_normal(total)
    for i in range(1, total):
        y[i] = 0.7 * y[i - 1] + e[i]
    y = y + 0.05 * t + 3.0 * (t > 80)
    return pd.DataFrame({"y": y, "year": 1900 + t})


# ---- Zivot-Andrews --------------------------------------------------------------------


def _zandrews_by_brute_force(y, lags, model, trim):
    total = len(y)
    dy = np.diff(y)
    n = total - lags - 1
    t = np.arange(lags + 2, total + 1)
    edge = int(trim * total)
    best = (np.inf, None)
    for tb in range(max(edge + 1, lags + 3), min(total - edge, total - 2) + 1):
        columns = [y[lags:total - 1], np.ones(n), t.astype(float)]
        columns += [dy[lags - j:total - 1 - j] for j in range(1, lags + 1)]
        if model in ("intercept", "both"):
            columns.append((t > tb).astype(float))
        if model in ("trend", "both"):
            columns.append(np.where(t > tb, t - tb, 0.0))
        statistic = sm.OLS(dy[lags:], np.column_stack(columns)).fit().tvalues[0]
        if statistic < best[0]:
            best = (statistic, tb)
    return best


@pytest.mark.parametrize("model", ["intercept", "trend", "both"])
@pytest.mark.parametrize("lags", [0, 3])
def test_zandrews_matches_brute_force_scan(broken, model, lags):
    y = broken["y"].to_numpy()
    statistic, tb = _zandrews_by_brute_force(y, lags, model, 0.15)
    result = oe.zandrews(broken, "y", break_=model, lags=lags, time="year")
    attrs = result.attrs
    assert attrs["statistic"] == pytest.approx(statistic, rel=1e-8)
    assert attrs["break_index"] == tb - 1
    assert attrs["break_period"] == 1900 + tb - 1
    assert attrs["nobs"] == len(y) - lags - 1 and attrs["lags"] == lags
    assert attrs["p_value"] is None and attrs["break"] == model
    assert attrs["critical_values"] == dict(zip(LEVELS, tables.ZIVOT_ANDREWS[model], strict=True))
    assert result.loc["min t", "statistic"] == attrs["statistic"]
    other = oe.zandrews(broken, "y", break_=model, lags=lags, trim=0.3)
    assert other.attrs["statistic"] == pytest.approx(
        _zandrews_by_brute_force(y, lags, model, 0.3)[0], rel=1e-8)
    assert other.attrs["candidates"] < attrs["candidates"]


@pytest.mark.parametrize("model,regression", [("intercept", "c"), ("both", "ct")])
def test_zandrews_matches_statsmodels(broken, model, regression):
    y = broken["y"].to_numpy()
    expected = zivot_andrews(y, maxlag=2, regression=regression, autolag=None)
    result = oe.zandrews(broken, "y", break_=model, lags=2)
    assert result.attrs["statistic"] == pytest.approx(expected[0], rel=1e-8)
    assert result.attrs["break_index"] == expected[4]
    # Automatic lag choice from the no-break regression with constant and trend.
    for method, autolag in (("aic", "AIC"), ("bic", "BIC"), ("t", "t-stat")):
        chosen = adfuller(y, maxlag=9, regression="ct", autolag=autolag)[2]
        automatic = oe.zandrews(broken, "y", break_=model, lags=method, maxlag=9)
        assert automatic.attrs["lags"] == chosen and automatic.attrs["lag_method"] == method
        if chosen:
            expected = zivot_andrews(y, maxlag=chosen, regression=regression, autolag=None)
            assert automatic.attrs["statistic"] == pytest.approx(expected[0], rel=1e-8)


def test_zandrews_finds_the_break_and_errors(broken):
    result = oe.zandrews(broken, "y", break_="intercept", lags=1, time="year")
    assert abs(result.attrs["break_period"] - 1980) <= 3
    assert result.attrs["statistic"] < result.attrs["critical_values"]["5%"]
    assert isinstance(result, DataFrame)
    assert json.loads(json.dumps(result.attrs)) == result.attrs
    assert "\\begin{tabular}" in str(result.to_latex())
    assert code(oe.zandrews, broken, "y", break_="level") == "invalid_option"
    assert code(oe.zandrews, broken, "y", trim=0.5) == "invalid_option"
    assert code(oe.zandrews, broken, "y", lags="hq") == "invalid_option"
    assert code(oe.zandrews, broken, "y", lags=-1) == "invalid_lags"
    assert code(oe.zandrews, broken, "y", lags=2, maxlag=4) == "invalid_option"
    assert code(oe.zandrews, broken, "y", lags=80) == "insufficient_observations"
    assert code(oe.zandrews, broken.iloc[:8], "y") == "insufficient_observations"
    assert code(oe.zandrews, broken.assign(y=3.0), "y") == "constant_series"
    assert code(oe.zandrews, pd.DataFrame({"y": np.arange(60.0)}), "y") \
        in {"collinear_regressors", "perfect_fit"}


# ---- Engle-Granger ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cointegrated():
    rng = np.random.default_rng(77)
    total = 160
    x1 = rng.standard_normal(total).cumsum()
    x2 = rng.standard_normal(total).cumsum() + 0.1 * np.arange(total)
    u = np.zeros(total)
    e = rng.standard_normal(total)
    for i in range(1, total):
        u[i] = 0.6 * u[i - 1] + e[i]
    frame = pd.DataFrame({"y": 2 + 0.8 * x1 - 0.5 * x2 + u, "x1": x1, "x2": x2,
                          "t": np.arange(total)})
    frame["z"] = rng.standard_normal(total).cumsum()
    return frame


@pytest.mark.parametrize("trend,regression", [("none", "n"), ("constant", "c"), ("trend", "ct"),
                                              ("quadratic", "ctt")])
@pytest.mark.parametrize("lags", [0, 2])
def test_egranger_matches_statsmodels_coint(cointegrated, trend, regression, lags):
    y = cointegrated["y"].to_numpy()
    x = cointegrated[["x1", "x2"]].to_numpy()
    expected = coint(y, x, trend=regression, maxlag=lags, autolag=None)
    result = oe.egranger(cointegrated, "y", ["x1", "x2"], lags=lags, trend=trend)
    attrs = result.attrs
    assert isinstance(result, TableSet)
    assert attrs["statistic"] == pytest.approx(expected[0], rel=1e-8)
    assert attrs["p_value"] == pytest.approx(expected[1], rel=1e-7, abs=1e-300)
    assert attrs["n_series"] == 3 and attrs["nobs"] == len(y) - lags - 1
    values = [attrs["critical_values"][level] for level in LEVELS]
    if trend == "none":
        for value, target in zip(values, (0.01, 0.05, 0.10), strict=True):
            assert adfvalues.mackinnonp(value, regression="n", N=3) == pytest.approx(target,
                                                                                    abs=1e-9)
    else:
        assert_allclose(values, adfvalues.mackinnoncrit(N=3, regression=regression,
                                                        nobs=len(y) - lags - 1), rtol=1e-12)
    assert result["test"].loc["Z(t)", "statistic"] == attrs["statistic"]


def test_egranger_regressions_ecm_and_lag_choice(cointegrated):
    y = cointegrated["y"].to_numpy()
    x = cointegrated[["x1", "x2"]].to_numpy()
    total = len(y)
    result = oe.egranger(cointegrated, "y", ["x1", "x2"], lags=1, regress=True, ecm=True,
                         time="t")
    assert list(result) == ["test", "cointegrating_regression", "adf_regression", "ecm"]
    first = sm.OLS(y, np.column_stack([x, np.ones(total)])).fit()
    table = result["cointegrating_regression"]
    assert list(table.index) == ["x1", "x2", "Intercept"]
    assert_allclose(table["coefficient"], first.params, rtol=1e-9)
    assert_allclose(table["std_error"], first.bse, rtol=1e-8)
    assert result.attrs["cointegrating_vector"] == pytest.approx(
        dict(zip(table.index, first.params, strict=True)))
    e = first.resid
    de = np.diff(e)
    second = sm.OLS(de[1:], np.column_stack([e[1:-1], de[:-1]])).fit()
    assert list(result["adf_regression"].index) == ["L1.e", "LD.e"]
    assert_allclose(result["adf_regression"]["coefficient"], second.params, rtol=1e-8)
    assert result.attrs["statistic"] == pytest.approx(second.tvalues[0], rel=1e-8)
    # Error-correction model: Delta y on e_{t-1}, lagged differences of y and x, constant.
    dy, dx = np.diff(y), np.diff(x, axis=0)
    design = np.column_stack([e[1:-1], dy[:-1], dx[:-1], np.ones(total - 2)])
    third = sm.OLS(dy[1:], design).fit()
    ecm = result["ecm"]
    assert list(ecm.index) == ["L1.e", "LD.y", "LD.x1", "LD.x2", "Intercept"]
    assert_allclose(ecm["coefficient"], third.params, rtol=1e-8)
    assert_allclose(ecm["std_error"], third.bse, rtol=1e-8)
    assert_allclose(ecm["p_value"], third.pvalues, rtol=1e-7)
    assert result.attrs["adjustment"] == pytest.approx(third.params[0], rel=1e-8)
    assert result.attrs["adjustment"] < 0
    # Without lags the ECM has the lagged residual and a constant only.
    plain = oe.egranger(cointegrated, "y", ["x1", "x2"], ecm=True)["ecm"]
    assert list(plain.index) == ["L1.e", "Intercept"]
    assert_allclose(plain["coefficient"],
                    sm.OLS(dy, np.column_stack([e[:-1], np.ones(total - 1)])).fit().params,
                    rtol=1e-8)
    # Automatic lag order of the residual regression (no deterministic terms).
    for method, autolag in (("aic", "AIC"), ("bic", "BIC"), ("t", "t-stat")):
        expected = adfuller(e, maxlag=6, regression="n", autolag=autolag)
        automatic = oe.egranger(cointegrated, "y", ["x1", "x2"], lags=method, maxlag=6)
        assert automatic.attrs["lags"] == expected[2]
        assert automatic.attrs["statistic"] == pytest.approx(expected[0], rel=1e-8)


def test_egranger_omits_collinear_regressors_and_reports_errors(cointegrated):
    frame = cointegrated.assign(dup=2 * cointegrated["x1"], one=1.0)
    result = oe.egranger(frame, "y", ["x1", "dup", "one", "x2"])
    reference = oe.egranger(frame, "y", ["x1", "x2"])
    assert result.attrs["omitted"] == ["dup", "one"]
    assert result.attrs["n_series"] == 3
    assert result.attrs["statistic"] == pytest.approx(reference.attrs["statistic"], rel=1e-9)
    assert any("Omitted" in note for note in result.attrs["notes"])
    assert json.loads(json.dumps(result.attrs)) == result.attrs
    assert "cointegrating" in result.to_latex() and "x2" in result.to_latex()
    unrelated = oe.egranger(frame, "z", ["x1"])
    assert unrelated.attrs["p_value"] > 0.05
    assert reference.attrs["p_value"] < 0.01
    assert code(oe.egranger, frame, "y", []) == "invalid_spec"
    assert code(oe.egranger, frame, "y", "x1") == "invalid_spec"
    assert code(oe.egranger, frame, "y", ["x1", "x1"]) == "invalid_spec"
    assert code(oe.egranger, frame, "y", ["y"]) == "invalid_spec"
    assert code(oe.egranger, frame, "y", ["one"]) == "collinear_regressors"
    assert code(oe.egranger, frame, "y", ["x1"], trend="drift") == "invalid_option"
    assert code(oe.egranger, frame, "y", ["x1"], lags=2, maxlag=3) == "invalid_option"
    assert code(oe.egranger, frame, "dup", ["x1"]) == "perfect_fit"
    many = frame.assign(**{f"w{i}": np.random.default_rng(i).standard_normal(len(frame)).cumsum()
                           for i in range(6)})
    assert code(oe.egranger, many, "y", [f"w{i}" for i in range(6)]) == "too_many_regressors"
    holes = frame.copy()
    holes.loc[5, "x1"] = np.nan
    assert code(oe.egranger, holes, "y", ["x1"]) == "missing_values"


# ---- Chow -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression():
    rng = np.random.default_rng(5)
    total = 120
    x1, x2 = rng.standard_normal(total), rng.standard_normal(total)
    late = np.arange(total) >= 70
    y = 1 + 0.5 * x1 - 0.2 * x2 + 0.9 * late * x1 + rng.standard_normal(total)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "year": np.arange(1900, 1900 + total)})


def _ssr(frame, rows, columns=("x1", "x2")):
    data = frame.iloc[rows]
    design = np.column_stack([np.ones(len(data))] + [data[c] for c in columns])
    return sm.OLS(data["y"].to_numpy(), design).fit().ssr


def test_chow_matches_split_sample_regressions(regression):
    total, k, position = len(regression), 3, 70
    pooled = _ssr(regression, slice(0, total))
    first, second = _ssr(regression, slice(0, position)), _ssr(regression, slice(position, total))
    f_stat = ((pooled - first - second) / k) / ((first + second) / (total - 2 * k))
    forecast = ((pooled - first) / (total - position)) / (first / (position - k))
    result = oe.chow(regression, "y", ["x1", "x2"], 1970, time="year")
    assert list(result.index) == ["chow_f", "wald", "lr", "forecast_f"]
    assert_allclose(result["statistic"],
                    [f_stat, k * f_stat, total * np.log(pooled / (first + second)), forecast],
                    rtol=1e-9)
    assert_allclose(result["p_value"],
                    [stats.f.sf(f_stat, k, total - 2 * k), stats.chi2.sf(k * f_stat, k),
                     stats.chi2.sf(total * np.log(pooled / (first + second)), k),
                     stats.f.sf(forecast, total - position, position - k)], rtol=1e-8)
    assert list(result["df"]) == [k, k, k, total - position]
    attrs = result.attrs
    assert attrs["statistic"] == pytest.approx(f_stat) and attrs["df2"] == total - 2 * k
    assert attrs["break_period"] == 1970 and attrs["break_index"] == position
    assert (attrs["nobs_1"], attrs["nobs_2"]) == (position, total - position)
    assert attrs["ssr"] == pytest.approx(pooled) and attrs["ssr_1"] == pytest.approx(first)
    assert attrs["terms"] == ["Intercept", "x1", "x2"]
    # The same break given as a row position, and the interacted regression's F test.
    by_position = oe.chow(regression, "y", ["x1", "x2"], position)
    assert by_position.attrs["statistic"] == pytest.approx(f_stat)
    late = (np.arange(total) >= position).astype(float)
    base = np.column_stack([np.ones(total), regression["x1"], regression["x2"]])
    full = sm.OLS(regression["y"].to_numpy(), np.column_stack([base, base * late[:, None]])).fit()
    assert attrs["statistic"] == pytest.approx(
        float(full.f_test(np.eye(6)[3:]).fvalue), rel=1e-8)
    dated = regression.assign(date=pd.date_range("2000-01-01", periods=total, freq="QS"))
    by_date = oe.chow(dated, "y", ["x1", "x2"], "2017-07-01", time="date")
    assert by_date.attrs["break_index"] == position
    assert by_date.attrs["statistic"] == pytest.approx(f_stat)
    assert by_date.attrs["break_period"].startswith("2017-07-01")


def test_chow_forecast_only_when_second_regime_is_short_and_collinearity(regression):
    total = len(regression)
    result = oe.chow(regression, "y", ["x1", "x2"], total - 2)
    assert result["statistic"].iloc[:3].isna().all()
    first, pooled = _ssr(regression, slice(0, total - 2)), _ssr(regression, slice(0, total))
    forecast = ((pooled - first) / 2) / (first / (total - 2 - 3))
    assert result.loc["forecast_f", "statistic"] == pytest.approx(forecast, rel=1e-9)
    assert result.attrs["statistic"] is None and result.attrs["forecast_statistic"] > 0
    assert any("undefined" in note for note in result.attrs["notes"])
    frame = regression.assign(dup=regression["x1"] * 3, one=2.0)
    omitted = oe.chow(frame, "y", ["x1", "dup", "x2", "one"], 70)
    assert omitted.attrs["omitted"] == ["dup", "one"]
    assert omitted.attrs["statistic"] == pytest.approx(
        oe.chow(regression, "y", ["x1", "x2"], 70).attrs["statistic"], rel=1e-9)
    mean_shift = oe.chow(regression, "y", [], 70)
    y = regression["y"].to_numpy()
    restricted = np.sum((y - y.mean()) ** 2)
    split = np.sum((y[:70] - y[:70].mean()) ** 2) + np.sum((y[70:] - y[70:].mean()) ** 2)
    assert mean_shift.attrs["statistic"] == pytest.approx(
        (restricted - split) / (split / (total - 2)), rel=1e-9)
    no_constant = oe.chow(regression, "y", ["x1"], 70, intercept=False)
    assert no_constant.attrs["terms"] == ["x1"] and no_constant.attrs["df"] == 1
    assert code(oe.chow, regression, "y", ["x1"], 0) == "invalid_break"
    assert code(oe.chow, regression, "y", ["x1"], total) == "invalid_break"
    assert code(oe.chow, regression, "y", ["x1"], 2.5) == "invalid_break"
    assert code(oe.chow, regression, "y", ["x1"], "soon", time="year") == "invalid_break"
    assert code(oe.chow, regression, "y", ["x1", "x2"], 2) == "singular_subsample"
    assert code(oe.chow, regression, "y", [], 70, intercept=False) == "invalid_spec"
    assert code(oe.chow, regression, "y", ["x1"], 70, intercept="no") == "invalid_option"
    assert code(oe.chow, regression.assign(y=regression["x1"]), "y", ["x1"], 70) == "perfect_fit"


# ---- sup-Wald ---------------------------------------------------------------------------


def _wald_path(frame, first, last, columns=("x1", "x2")):
    total, k = len(frame), 1 + len(columns)
    pooled = _ssr(frame, slice(0, total), columns)
    values = []
    for n1 in range(first, last + 1):
        split = _ssr(frame, slice(0, n1), columns) + _ssr(frame, slice(n1, total), columns)
        values.append((pooled - split) / (split / (total - 2 * k)))
    return np.array(values)


@pytest.mark.parametrize("trim", [0.15, 0.25])
def test_sbsingle_matches_brute_force(regression, trim):
    total, k = len(regression), 3
    edge = max(k + 1, math.ceil(trim * total - 1e-9))      # Stata: ceil(trim T) per regime
    wald = _wald_path(regression, edge, total - edge)
    result = oe.sbsingle(regression, "y", ["x1", "x2"], trim=trim, time="year")
    attrs = result.attrs
    assert attrs["sup_wald"] == pytest.approx(wald.max(), rel=1e-8)
    assert attrs["ave_wald"] == pytest.approx(wald.mean(), rel=1e-8)
    assert attrs["exp_wald"] == pytest.approx(np.log(np.mean(np.exp(wald / 2))), rel=1e-8)
    assert attrs["break_index"] == edge + int(wald.argmax())
    assert attrs["break_period"] == 1900 + attrs["break_index"]
    assert attrs["candidates"] == len(wald) and attrs["df"] == k
    assert attrs["statistic"] == attrs["sup_wald"]
    assert attrs["p_value"] == pytest.approx(
        critical.supwald_p(wald.max(), k, edge / total, (total - edge) / total))
    assert attrs["p_value"] < 0.01
    if trim == 0.15:
        assert attrs["critical_values"] == pytest.approx(
            {"1%": 3 * 6.02, "5%": 3 * 4.71, "10%": 3 * 4.09})
        assert attrs["statistic"] > attrs["critical_values"]["1%"]
    else:
        assert attrs["critical_values"] is None
    for test, key in (("avewald", "ave_wald"), ("expwald", "exp_wald")):
        other = oe.sbsingle(regression, "y", ["x1", "x2"], trim=trim, test=test)
        assert other.attrs["statistic"] == pytest.approx(attrs[key])
        assert other.attrs["p_value"] is None and other.attrs["critical_values"] is None
        assert list(other.index) == [test]
    # At the chosen date the Wald statistic is K times the Chow F statistic.
    chow = oe.chow(regression, "y", ["x1", "x2"], attrs["break_index"])
    assert attrs["sup_wald"] == pytest.approx(k * chow.attrs["statistic"], rel=1e-8)


def test_supwald_tail_approximation_reproduces_the_tabulated_critical_values():
    # DeLong's formula against the Andrews (2003) table: every entry within 15% of its level.
    for q, row in enumerate(tables.QLR_F_15, start=1):
        for value, level in zip(row, (0.10, 0.05, 0.01), strict=True):
            approx = critical.supwald_p(q * value, q, 0.15, 0.85)
            assert level * 0.97 <= approx <= level * 1.19, (q, level, approx)
    assert critical.supwald_critical(21, 0.15) is None
    assert critical.supwald_critical(2, 0.10) is None
    assert critical.supwald_p(1.0, 3, 0.15, 0.85) == 1.0
    # The exact asymptotic distribution for q = 1 by simulation of a Brownian bridge.
    rng = np.random.default_rng(12)
    steps, reps = 2000, 20000
    increments = rng.standard_normal((reps, steps)) / math.sqrt(steps)
    path = increments.cumsum(axis=1)
    grid = np.arange(1, steps + 1) / steps
    bridge = path - grid * path[:, -1:]
    inside = (grid >= 0.15) & (grid <= 0.85)
    sup = (bridge[:, inside] ** 2 / (grid[inside] * (1 - grid[inside]))).max(axis=1)
    assert np.mean(sup > 8.68) == pytest.approx(0.05, abs=0.008)
    assert critical.supwald_p(8.68, 1, 0.15, 0.85) == pytest.approx(np.mean(sup > 8.68),
                                                                    abs=0.01)


def test_sbsingle_collinearity_errors_and_no_break(regression):
    rng = np.random.default_rng(2)
    stable = regression.assign(y=1 + regression["x1"] + rng.standard_normal(len(regression)))
    result = oe.sbsingle(stable, "y", ["x1", "x2"])
    assert result.attrs["p_value"] > 0.05
    frame = regression.assign(dup=-regression["x2"])
    omitted = oe.sbsingle(frame, "y", ["x1", "x2", "dup"])
    assert omitted.attrs["omitted"] == ["dup"]
    assert omitted.attrs["statistic"] == pytest.approx(
        oe.sbsingle(regression, "y", ["x1", "x2"]).attrs["statistic"], rel=1e-9)
    assert json.loads(json.dumps(omitted.attrs)) == omitted.attrs
    late = regression.assign(dummy=(np.arange(len(regression)) >= 100).astype(float))
    assert code(oe.sbsingle, late, "y", ["x1", "dummy"]) == "singular_subsample"
    assert code(oe.sbsingle, regression, "y", ["x1"], test="lr") == "invalid_option"
    assert code(oe.sbsingle, regression, "y", ["x1"], trim=0.6) == "invalid_option"
    assert code(oe.sbsingle, regression.iloc[:5], "y", ["x1", "x2"]) \
        == "insufficient_observations"


# ---- CUSUM ------------------------------------------------------------------------------


def test_cusum_matches_recursive_residuals(regression):
    total, k = len(regression), 3
    design = np.column_stack([np.ones(total), regression["x1"], regression["x2"]])
    ols = sm.OLS(regression["y"].to_numpy(), design).fit()
    expected = recursive_olsresiduals(ols, skip=k, alpha=0.95)
    w = expected[4][k:]
    result = oe.cusum(regression, "y", ["x1", "x2"], time="year")
    assert len(result) == total - k
    assert_allclose(result["recursive_residual"], w, rtol=1e-7, atol=1e-9)
    assert list(result["period"][:2]) == [1900 + k, 1901 + k]
    # Brown, Durbin and Evans: the recursive residuals reproduce the residual sum of squares.
    assert np.sum(result["recursive_residual"] ** 2) == pytest.approx(ols.ssr, rel=1e-9)
    assert result.attrs["sigma_ols"] == pytest.approx(np.sqrt(ols.ssr / (total - k)), rel=1e-9)
    # Stata's estat sbcusum: variance of the recursive residuals about their mean, T - K.
    sigma = np.sqrt(np.sum((w - w.mean()) ** 2) / (total - k))
    assert result.attrs["sigma"] == pytest.approx(sigma, rel=1e-7)
    assert_allclose(result["cusum"], np.cumsum(w) / sigma, rtol=1e-7, atol=1e-9)
    step = np.arange(1, total - k + 1)
    a = optimize.brentq(lambda c: stats.norm.sf(3 * c) + np.exp(-4 * c * c) * stats.norm.cdf(c)
                        - 0.025, 0.1, 3.0, xtol=1e-14)
    assert a == pytest.approx(0.9479, abs=5e-5)              # Stata prints 0.9479
    bound = a * np.sqrt(total - k) + 2 * a * step / np.sqrt(total - k)
    assert_allclose(result["cusum_upper"], bound, rtol=1e-10)
    assert_allclose(result["cusum_lower"], -bound, rtol=1e-10)
    # statsmodels draws the same lines one step earlier (from 0) with a rounded to 0.948.
    assert_allclose(result["cusum_upper"].to_numpy()[:-1], expected[6][1][1:], rtol=2e-4)
    squares = np.cumsum(w ** 2) / np.sum(w ** 2)
    assert_allclose(result["cusumsq"], squares, rtol=1e-8)
    m = (total - k) / 2 - 1
    c0 = 1.3581015 / np.sqrt(m) - 0.6701218 / m - 0.8858694 / m ** 1.5
    assert result.attrs["cusumsq_critical"] == pytest.approx(c0)
    assert_allclose(result["cusumsq_upper"], step / (total - k) + c0, rtol=1e-12)
    attrs = result.attrs
    statistic = np.max(np.abs(np.cumsum(w) / sigma) / (np.sqrt(total - k)
                                                        * (1 + 2 * step / (total - k))))
    assert attrs["statistic"] == pytest.approx(statistic, rel=1e-7)
    assert_allclose([attrs["critical_values"][level] for level in ("1%", "5%", "10%")],
                    [1.1430, 0.9479, 0.8499], atol=5e-5)     # as printed by Stata
    assert_allclose([attrs["critical_values"][level] for level in ("1%", "5%", "10%")],
                    [tables.CUSUM_A[level] for level in ("1%", "5%", "10%")], atol=6e-4)
    assert attrs["cusum_crosses"] == bool(
        np.any(np.abs(result["cusum"]) > result["cusum_upper"]))
    assert attrs["cusumsq_crosses"] == bool(attrs["cusumsq_statistic"] > c0)
    assert attrs["n_recursive"] == total - k and attrs["start"] == k
    assert json.loads(json.dumps(attrs)) == attrs


def test_cusum_detects_a_mean_shift_and_handles_early_collinearity(regression):
    total = len(regression)
    shifted = regression.assign(y=regression["y"] + 4.0 * (np.arange(total) >= 60))
    result = oe.cusum(shifted, "y", ["x1", "x2"], level="1%")
    assert result.attrs["cusum_crosses"] and result.attrs["statistic"] > 1.143
    assert result.attrs["level"] == "1%"
    stable = oe.cusum(regression.assign(y=np.random.default_rng(1).standard_normal(total)),
                      "y", ["x1"])
    assert not stable.attrs["cusum_crosses"]
    # A dummy that switches on late is not identified by the first observations.
    late = regression.assign(dummy=(np.arange(total) >= 10).astype(float))
    path = oe.cusum(late, "y", ["x1", "dummy"])
    assert path.attrs["start"] == 11 and len(path) == total - 11
    assert any("first 11 observations" in note for note in path.attrs["notes"])
    design = np.column_stack([np.ones(total), late["x1"], late["dummy"]])
    ols = sm.OLS(late["y"].to_numpy(), design).fit()
    expected = recursive_olsresiduals(ols, skip=11)[4][11:]
    assert_allclose(path["recursive_residual"], expected, rtol=1e-6, atol=1e-8)
    omitted = oe.cusum(regression.assign(dup=regression["x1"]), "y", ["x1", "dup"])
    assert omitted.attrs["omitted"] == ["dup"] and omitted.attrs["terms"] == ["Intercept", "x1"]
    assert code(oe.cusum, regression, "y", ["x1"], level="2%") == "invalid_option"
    assert code(oe.cusum, regression, "y", "x1") == "invalid_spec"
    assert code(oe.cusum, regression.iloc[:4], "y", ["x1", "x2"]) == "insufficient_observations"
    holes = regression.copy()
    holes.loc[3, "x2"] = np.nan
    assert code(oe.cusum, holes, "y", ["x1", "x2"]) == "missing_values"


def test_break_tests_scale_to_a_million_rows():
    rng = np.random.default_rng(0)
    n, names = 1_000_000, [f"x{i}" for i in range(9)]
    frame = pd.DataFrame(rng.standard_normal((n, 9)), columns=names)
    frame["y"] = frame.sum(axis=1) + rng.standard_normal(n)
    frame["walk"] = rng.standard_normal(n).cumsum()
    started = time.perf_counter()
    assert math.isfinite(oe.sbsingle(frame, "y", names).attrs["statistic"])
    assert len(oe.cusum(frame, "y", names)) == n - 10
    assert math.isfinite(oe.chow(frame, "y", names, 400_000).attrs["statistic"])
    assert math.isfinite(oe.zandrews(frame, "walk", break_="both", lags=4).attrs["statistic"])
    assert math.isfinite(oe.egranger(frame, "walk", ["x0", "x1"], lags=2).attrs["statistic"])
    assert time.perf_counter() - started < 30
