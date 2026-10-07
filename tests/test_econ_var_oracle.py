"""Independent verification of the var family: published output and second-source tables.

Part 1 of the verification suite (the algebraic oracles are in
``test_econ_var_oracle_algebra.py``, hostile inputs in
``test_econ_var_adversarial.py``).

* Johansen critical values: every shipped number against independent published
  transcriptions of Osterwald-Lenum (1992) -- the matrices of the Stata user
  command ``johans`` (SSC S418701, all five tables), the arrays of the R package
  urca (Tables 1* and 2*), the output printed in the Stata manual (Table 1) --
  and, as plausibility checks, against the MacKinnon-Haug-Michelis values that
  ship with statsmodels, Johansen's (1995) tables and a simulation of the limit
  distribution.
* Output printed in the Stata 19 manuals for the Lutkepohl data (shipped with
  statsmodels' test-suite): [TS] var (exogenous variables), varsoc
  (post-estimation, with exogenous variables), vargranger, varwle, varnorm,
  varlmar, varstable and irf table (orthogonalized responses and variance
  decompositions with their standard errors, two orderings), and the arithmetic
  of the [TS] vecrank and [TS] vec tables.
"""

import os

import numpy as np
import pandas as pd
import pytest
import statsmodels
from numpy.testing import assert_allclose
from statsmodels.tsa.coint_tables import c_sja, c_sjt

import openecon as oe
from openecon.econometrics.var import critical_values

TRENDS = ["none", "rconstant", "constant", "rtrend", "trend"]

# ---- Osterwald-Lenum (1992) as transcribed in johans.ado (95% and 99% columns) -------------
# Rows: K - r = 1..11. Keys: (maximum-eigenvalue 5%, 1%, trace 5%, 1%).
JOHANS = {
    "none": (   # Case 0
        [3.84, 11.44, 17.89, 23.80, 30.04, 36.36, 41.51, 47.99, 53.69, 59.06, 65.30],
        [6.51, 15.69, 22.99, 28.82, 35.17, 41.00, 47.15, 53.90, 59.78, 65.21, 72.36],
        [3.84, 12.53, 24.31, 39.89, 59.46, 82.49, 109.99, 141.20, 175.77, 212.67, 255.27],
        [6.51, 16.31, 29.75, 45.58, 66.52, 90.45, 119.80, 152.32, 187.31, 226.40, 269.81]),
    "rconstant": (   # Case 1*
        [9.24, 15.67, 22.00, 28.14, 34.40, 40.30, 46.45, 52.00, 57.42, 63.57, 69.74],
        [12.97, 20.20, 26.81, 33.24, 39.79, 46.82, 51.91, 57.95, 63.71, 69.94, 76.63],
        [9.24, 19.96, 34.91, 53.12, 76.07, 102.14, 131.70, 165.58, 202.92, 244.15, 291.40],
        [12.97, 24.60, 41.07, 60.16, 84.45, 111.01, 143.09, 177.20, 215.74, 257.68, 307.64]),
    "constant": (    # Case 1
        [3.76, 14.07, 20.97, 27.07, 33.46, 39.37, 45.28, 51.42, 57.12, 62.81, 68.83],
        [6.65, 18.63, 25.52, 32.24, 38.77, 45.10, 51.57, 57.69, 62.80, 69.09, 75.95],
        [3.76, 15.41, 29.68, 47.21, 68.52, 94.15, 124.24, 156.00, 192.89, 233.13, 277.71],
        [6.65, 20.04, 35.65, 54.46, 76.07, 103.18, 133.57, 168.36, 204.95, 247.18, 293.44]),
    "rtrend": (      # Case 2*
        [12.25, 18.96, 25.54, 31.46, 37.52, 43.97, 49.42, 55.50, 61.29, 66.23, 72.72],
        [16.26, 23.65, 30.34, 36.65, 42.36, 49.51, 54.71, 62.46, 67.88, 73.73, 79.23],
        [12.25, 25.32, 42.44, 62.99, 87.31, 114.90, 146.76, 182.82, 222.21, 263.42, 310.81],
        [16.26, 30.45, 48.45, 70.05, 96.58, 124.75, 158.49, 196.08, 234.41, 279.07, 327.45]),
    "trend": (       # Case 2
        [3.74, 16.87, 23.78, 30.33, 36.41, 42.48, 48.45, 54.25, 60.29, 66.10, 71.68],
        [6.40, 21.47, 28.83, 35.68, 41.58, 48.17, 54.48, 60.81, 66.91, 72.96, 78.51],
        [3.74, 18.17, 34.55, 54.64, 77.74, 104.94, 136.61, 170.80, 208.97, 250.84, 295.99],
        [6.40, 23.46, 40.49, 61.24, 85.78, 114.36, 146.99, 182.51, 222.46, 263.94, 312.58]),
}

# ---- urca::ca.jo, arrays cv.const and cv.trend: [11 dims] x [10%, 5%, 1%] x [eigen, trace] --
URCA = {
    "rconstant": [
        7.52, 13.75, 19.77, 25.56, 31.66, 37.45, 43.25, 48.91, 54.35, 60.25, 66.02,
        9.24, 15.67, 22.00, 28.14, 34.40, 40.30, 46.45, 52.00, 57.42, 63.57, 69.74,
        12.97, 20.20, 26.81, 33.24, 39.79, 46.82, 51.91, 57.95, 63.71, 69.94, 76.63,
        7.52, 17.85, 32.00, 49.65, 71.86, 97.18, 126.58, 159.48, 196.37, 236.54, 282.45,
        9.24, 19.96, 34.91, 53.12, 76.07, 102.14, 131.70, 165.58, 202.92, 244.15, 291.40,
        12.97, 24.60, 41.07, 60.16, 84.45, 111.01, 143.09, 177.20, 215.74, 257.68, 307.64],
    "rtrend": [
        10.49, 16.85, 23.11, 29.12, 34.75, 40.91, 46.32, 52.16, 57.87, 63.18, 69.26,
        12.25, 18.96, 25.54, 31.46, 37.52, 43.97, 49.42, 55.50, 61.29, 66.23, 72.72,
        16.26, 23.65, 30.34, 36.65, 42.36, 49.51, 54.71, 62.46, 67.88, 73.73, 79.23,
        10.49, 22.76, 39.06, 59.14, 83.20, 110.42, 141.01, 176.67, 215.17, 256.72, 303.13,
        12.25, 25.32, 42.44, 62.99, 87.31, 114.90, 146.76, 182.82, 222.21, 263.42, 310.81,
        16.26, 30.45, 48.45, 70.05, 96.58, 124.75, 158.49, 196.08, 234.41, 279.07, 327.45],
}

# ---- Johansen (1995), Tables 15.1-15.5, 95% trace quantiles (as quoted in RATS johmle.src) --
JOHANSEN_1995 = {
    "none": [4.14, 12.21, 24.08, 39.71, 59.24, 82.61, 109.93, 140.74, 175.47, 214.07, 256.23],
    "constant": [3.84, 15.34, 29.38, 47.21, 68.68, 93.92, 123.04, 155.75, 192.30, 232.60,
                 276.37],
    "trend": [3.84, 18.15, 34.56, 54.11, 77.79, 104.76, 135.66, 170.15, 208.53, 250.53, 296.02],
    "rconstant": [9.13, 19.99, 34.80, 53.42, 75.74, 101.84, 132.00, 165.73, 203.34, 244.56,
                  289.71],
    "rtrend": [12.39, 25.47, 42.20, 62.61, 86.96, 114.96, 146.75, 182.45, 221.56, 264.23,
               311.13],
}


def shipped(trend, statistic, level):
    return [critical_values.lookup(trend, d, statistic, level) for d in range(1, 12)]


@pytest.mark.parametrize("trend", TRENDS)
def test_critical_values_equal_the_johans_ado_transcription(trend):
    max5, max1, trace5, trace1 = JOHANS[trend]
    assert critical_values.available(trend) == critical_values.MAX_DIMENSION == 11
    assert shipped(trend, "max", 5) == max5
    assert shipped(trend, "max", 1) == max1
    assert shipped(trend, "trace", 5) == trace5
    assert shipped(trend, "trace", 1) == trace1
    assert critical_values.lookup(trend, 12, "trace", 5) is None
    assert critical_values.lookup(trend, 0, "trace", 5) is None


@pytest.mark.parametrize("trend", ["rconstant", "rtrend"])
def test_critical_values_equal_the_urca_arrays(trend):
    table = np.array(URCA[trend]).reshape(2, 3, 11)        # [statistic, level, dimension]
    assert shipped(trend, "max", 5) == table[0, 1].tolist()
    assert shipped(trend, "max", 1) == table[0, 2].tolist()
    assert shipped(trend, "trace", 5) == table[1, 1].tolist()
    assert shipped(trend, "trace", 1) == table[1, 2].tolist()


def test_critical_values_equal_the_stata_manual_output():
    """[TS] vecrank, examples 1-3: vecrank y i c, lags(5) [level99] [max levela]."""
    assert shipped("constant", "trace", 5)[:3][::-1] == [29.68, 15.41, 3.76]
    assert shipped("constant", "trace", 1)[:3][::-1] == [35.65, 20.04, 6.65]
    assert shipped("constant", "max", 5)[:3][::-1] == [20.97, 14.07, 3.76]
    assert shipped("constant", "max", 1)[:3][::-1] == [25.52, 18.63, 6.65]


@pytest.mark.parametrize("trend, order", [("none", -1), ("constant", 0), ("trend", 1)])
def test_critical_values_are_close_to_mackinnon_haug_michelis(trend, order):
    """Same deterministic case, more accurate response surfaces: a wrong table would be far off.

    statsmodels' det_order -1 / 0 / 1 are MacKinnon-Haug-Michelis' cases without
    deterministic terms, with a constant and with a linear trend.
    """
    for d in range(1, 12):
        loose = trend == "none" and d == 1            # 3.84 against 4.13: known discrepancy
        tolerance = 0.075 if loose else 0.04
        trace, maximum = c_sjt(d, order), c_sja(d, order)       # [90%, 95%, 99%]
        assert critical_values.lookup(trend, d, "trace", 5) == pytest.approx(trace[1],
                                                                             rel=tolerance)
        assert critical_values.lookup(trend, d, "trace", 1) == pytest.approx(trace[2],
                                                                             rel=tolerance)
        assert critical_values.lookup(trend, d, "max", 5) == pytest.approx(maximum[1],
                                                                           rel=tolerance)
        assert critical_values.lookup(trend, d, "max", 1) == pytest.approx(maximum[2],
                                                                           rel=tolerance)
    # The neighbouring tables are much further away than the tolerance.
    assert abs(critical_values.lookup("rconstant", 3, "trace", 5) / c_sjt(3, 0)[1] - 1) > 0.15


@pytest.mark.parametrize("trend", TRENDS)
def test_trace_critical_values_are_close_to_johansen_1995(trend):
    ours = np.array(shipped(trend, "trace", 5))
    theirs = np.array(JOHANSEN_1995[trend])
    relative = np.abs(ours / theirs - 1)
    assert relative[2:].max() < 0.012
    assert relative[:2].max() < (0.075 if trend == "none" else 0.03)


def limit_trace(rng, trend, d, steps, reps):
    """Draws of tr{int dW F' (int F F')^-1 int F dW'} for the five deterministic cases."""
    eps = rng.standard_normal((reps, steps, d))
    lag = np.concatenate([np.zeros((reps, 1, d)), np.cumsum(eps, axis=1)[:, :-1]], axis=1)
    time = np.arange(1, steps + 1) / steps
    ones = np.ones((reps, steps, 1))
    line = np.broadcast_to(time[None, :, None], (reps, steps, 1))
    if trend == "none":
        f = lag
    elif trend == "rconstant":
        f = np.concatenate([lag, ones], axis=2)
    elif trend == "constant":                     # demeaned, last component replaced by time
        f = np.concatenate([lag[:, :, :-1], line], axis=2)
        f = f - f.mean(axis=1, keepdims=True)
    elif trend == "rtrend":
        f = np.concatenate([lag, line], axis=2)
        f = f - f.mean(axis=1, keepdims=True)
    else:                                         # detrended, last component replaced by t^2
        f = np.concatenate([lag[:, :, :-1], line ** 2], axis=2)
        basis = np.column_stack([np.ones(steps), time])
        project = np.eye(steps) - basis @ np.linalg.solve(basis.T @ basis, basis.T)
        f = np.einsum("st,bti->bsi", project, f)
    sef = np.einsum("bti,btj->bij", eps, f)
    sff = np.einsum("bti,btj->bij", f, f)
    return np.trace(sef @ np.linalg.solve(sff, sef.transpose(0, 2, 1)), axis1=1, axis2=2)


@pytest.mark.parametrize("trend, d", [("none", 7), ("none", 9), ("trend", 6), ("trend", 8),
                                      ("trend", 3), ("constant", 5), ("rconstant", 4),
                                      ("rtrend", 4)])
def test_single_source_dimensions_agree_with_a_simulation_of_the_limit(trend, d):
    """The completed dimensions (Table 0: 7-11, Table 2: 6-11) sit where a simulation puts them.

    Random walks of length 400 as in the original paper; 2,500 replications
    give a standard error of about 0.5 for the 95% quantile, and the adjacent
    dimensions are more than 20 away.
    """
    trace = limit_trace(np.random.default_rng(1000 + d), trend, d, 400, 2500)
    expected = critical_values.lookup(trend, d, "trace", 5)
    assert np.quantile(trace, 0.95) == pytest.approx(expected, rel=0.025)
    assert np.quantile(trace, 0.99) == pytest.approx(
        critical_values.lookup(trend, d, "trace", 1), rel=0.04)
    for neighbour in (d - 1, d + 1):
        assert abs(critical_values.lookup(trend, neighbour, "trace", 5) - expected) > 15


def test_single_source_entries_are_flagged_in_the_result():
    assert critical_values.SINGLE_SOURCE == {"none": 7, "trend": 6}
    assert not critical_values.single_source("none", 6)
    assert critical_values.single_source("none", 7) and critical_values.single_source("trend", 11)
    assert not critical_values.single_source("constant", 11)
    rng = np.random.default_rng(3)
    names = [f"v{i}" for i in range(7)]
    frame = pd.DataFrame(rng.normal(size=(400, 7)).cumsum(axis=0), columns=names)
    assert oe.vecrank(data=frame, y=names[:6], trend="none").attrs["critical_values_flag"] is None
    flagged = oe.vecrank(data=frame, y=names, trend="none")
    assert "johans.ado" in flagged.attrs["critical_values_flag"]
    assert "K - r >= 7" in flagged.attrs["critical_values_flag"]
    assert flagged["trace_cv5"][0] == 109.99 and flagged["max_cv1"][0] == 47.15
    assert oe.vecrank(data=frame, y=names).attrs["critical_values_flag"] is None
    assert oe.vecrank(data=frame, y=names[:5], trend="trend").attrs[
        "critical_values_flag"] is None
    six = oe.vecrank(data=frame, y=names[:6], trend="trend")
    assert "K - r >= 6" in six.attrs["critical_values_flag"] and six["trace_cv5"][0] == 104.94
    model = oe.vec(data=frame, y=names[:6], trend="trend", rank=1)
    assert model.extra["critical_values_flag"] == six.attrs["critical_values_flag"]
    assert oe.vec(data=frame, y=names[:3], rank=1).extra["critical_values_flag"] is None


# ---- output printed in the Stata manuals (Lutkepohl data) -----------------------------------

NAMES = ["dln_inv", "dln_inc", "dln_consump"]


@pytest.fixture(scope="module")
def lutkepohl():
    path = os.path.join(os.path.dirname(statsmodels.__file__), "tsa", "tests", "results",
                        "lutkepohl2.dta")
    if not os.path.exists(path):
        pytest.skip("statsmodels does not ship the Lutkepohl data")
    frame = pd.read_stata(path)
    assert len(frame) == 92 and str(frame.qtr.iloc[0])[:10] == "1960-01-01"
    return frame


@pytest.fixture(scope="module")
def until_1978(lutkepohl):
    """`if qtr<=tq(1978q4)`: the differences start in 1960q2, 73 observations with two lags."""
    return lutkepohl[lutkepohl.qtr <= "1978-12-31"].iloc[1:].reset_index(drop=True)


def test_published_vargranger(until_1978):
    """[TS] vargranger, example 1: var ..., dfk small; vargranger (F, df, df_r, Prob > F)."""
    result = oe.var(data=until_1978, y=NAMES, time="qtr", lags=2, dfk=True, small=True)
    published = [
        ("dln_inv", "dln_inc", .04847, 2, 0.9527), ("dln_inv", "dln_consump", 1.5004, 2, 0.2306),
        ("dln_inv", "ALL", 1.5917, 4, 0.1869), ("dln_inc", "dln_inv", 1.7683, 2, 0.1786),
        ("dln_inc", "dln_consump", 1.7184, 2, 0.1873), ("dln_inc", "ALL", 1.9466, 4, 0.1130),
        ("dln_consump", "dln_inv", .97147, 2, 0.3839), ("dln_consump", "dln_inc", 6.1465, 2, 0.0036),
        ("dln_consump", "ALL", 3.7746, 4, 0.0080)]
    rows = result.extra["granger"]
    assert [(r["equation"], r["excluded"]) for r in rows] == [(e, x) for e, x, *_ in published]
    for row, (_, _, statistic, df, p) in zip(rows, published, strict=True):
        assert row["distribution"] == "F" and row["df"] == df and row["df2"] == 66
        assert row["statistic"] == pytest.approx(statistic, rel=1.3e-4)   # five printed digits
        assert row["p_value"] == pytest.approx(p, abs=6e-5)
    # "test [dln_inv]L.dln_inv ... accumulate": the equation's F statistic, F(6, 66) = 1.62.
    equation = result.extra["equations"][0]
    assert (equation["df"], equation["df2"]) == (6, 66)
    assert equation["statistic"] == pytest.approx(1.62, abs=6e-3)
    assert equation["p_value"] == pytest.approx(0.1547, abs=6e-5)
    assert result.tests["granger_all_dln_consump"]["statistic"] == pytest.approx(3.7746, rel=6e-5)


def test_published_varwle(until_1978):
    """[TS] varwle, example 1: var ..., dfk small; varwle."""
    result = oe.var(data=until_1978, y=NAMES, time="qtr", lags=2, dfk=True, small=True)
    published = {
        (1, "dln_inv"): (2.64902, 3, 0.0560), (2, "dln_inv"): (1.25799, 3, 0.2960),
        (1, "dln_inc"): (2.19276, 3, 0.0971), (2, "dln_inc"): (.907499, 3, 0.4423),
        (1, "dln_consump"): (1.80804, 3, 0.1543), (2, "dln_consump"): (5.57645, 3, 0.0018),
        (1, "ALL"): (3.78884, 9, 0.0007), (2, "ALL"): (2.96811, 9, 0.0050)}
    rows = {(r["lag"], r["equation"]): r for r in result.extra["lag_exclusion"]}
    assert set(rows) == set(published)
    for key, (statistic, df, p) in published.items():
        assert rows[key]["df"] == df and rows[key]["df2"] == 66
        assert rows[key]["statistic"] == pytest.approx(statistic, rel=6e-6)
        assert rows[key]["p_value"] == pytest.approx(p, abs=6e-5)
    assert result.tests["lag_exclusion_L2"]["statistic"] == pytest.approx(2.96811, rel=6e-6)


def test_published_varwle_chi2(lutkepohl):
    """[TS] varwle, example 2: the VAR underlying `svar dln_inc dln_consump` (89 observations)."""
    frame = lutkepohl.iloc[1:].reset_index(drop=True)
    result = oe.var(data=frame, y=["dln_inc", "dln_consump"], time="qtr", lags=2)
    assert result.nobs == 89
    published = {
        (1, "dln_inc"): (6.88775, 2, 0.032), (2, "dln_inc"): (1.873546, 2, 0.392),
        (1, "dln_consump"): (9.938547, 2, 0.007), (2, "dln_consump"): (13.89996, 2, 0.001),
        (1, "ALL"): (34.54276, 4, 0.000), (2, "ALL"): (19.44093, 4, 0.001)}
    rows = {(r["lag"], r["equation"]): r for r in result.extra["lag_exclusion"]}
    for key, (statistic, df, p) in published.items():
        assert rows[key]["distribution"] == "chi2" and rows[key]["df"] == df
        assert rows[key]["statistic"] == pytest.approx(statistic, rel=6e-7)
        assert rows[key]["p_value"] == pytest.approx(p, abs=6e-4)


def test_published_varnorm_and_varlmar(until_1978):
    """[TS] varnorm example 1 and [TS] varlmar example 1: var ..., dfk; varnorm; varlmar, mlag(5)."""
    result = oe.var(data=until_1978, y=NAMES, time="qtr", lags=2, dfk=True, lm_lags=5)
    published = {   # skewness, chi2, p | kurtosis, chi2, p | Jarque-Bera chi2, p
        "dln_inv": (.11935, 0.173, 0.67718, 3.9331, 2.648, 0.10367, 2.821, 0.24397),
        "dln_inc": (-.38316, 1.786, 0.18139, 3.7396, 1.664, 0.19710, 3.450, 0.17817),
        "dln_consump": (-.31275, 1.190, 0.27532, 2.6484, 0.376, 0.53973, 1.566, 0.45702)}
    rows = {r["equation"]: r for r in result.extra["normality"]}
    for name, (skew, s, sp, kurt, q, qp, jb, jbp) in published.items():
        row = rows[name]
        assert row["skewness"] == pytest.approx(skew, abs=6e-6)
        assert row["kurtosis"] == pytest.approx(kurt, abs=6e-5)
        assert row["skewness_chi2"] == pytest.approx(s, abs=6e-4)
        assert row["kurtosis_chi2"] == pytest.approx(q, abs=6e-4)
        assert row["jarque_bera"] == pytest.approx(jb, abs=6e-4)
        assert row["skewness_p"] == pytest.approx(sp, abs=6e-6)
        assert row["kurtosis_p"] == pytest.approx(qp, abs=6e-6)
        assert row["jarque_bera_p"] == pytest.approx(jbp, abs=6e-6) and row["jarque_bera_df"] == 2
    joint = rows["ALL"]
    assert joint["jarque_bera"] == pytest.approx(7.838, abs=6e-4) and joint["jarque_bera_df"] == 6
    assert joint["jarque_bera_p"] == pytest.approx(0.25025, abs=6e-6)
    assert joint["skewness_chi2"] == pytest.approx(3.150, abs=6e-4)
    assert joint["skewness_p"] == pytest.approx(0.36913, abs=6e-6)
    assert joint["kurtosis_chi2"] == pytest.approx(4.688, abs=6e-4)
    assert joint["kurtosis_p"] == pytest.approx(0.19613, abs=6e-6)
    assert result.tests["normality"]["statistic"] == pytest.approx(7.838, abs=6e-4)
    # The normality test uses the dfk estimate of Sigma ("dfk estimator used in computations").
    plain = oe.var(data=until_1978, y=NAMES, time="qtr", lags=2, lm_lags=5)
    assert plain.tests["normality"]["statistic"] != pytest.approx(7.838, abs=1e-2)
    lm = [(5.5871, 0.78043), (6.3189, 0.70763), (8.4022, 0.49418), (11.8742, 0.22049),
          (5.2914, 0.80821)]
    for fit in (result, plain):           # "varlmar always uses the ML estimator of Sigma"
        for lag, (statistic, p) in enumerate(lm, start=1):
            test = fit.tests[f"lm_autocorrelation_L{lag}"]
            assert test["statistic"] == pytest.approx(statistic, abs=6e-5) and test["df"] == 9
            assert test["p_value"] == pytest.approx(p, abs=6e-6)


def test_published_varstable(lutkepohl):
    """[TS] varstable, example 1: var ... if qtr>=tq(1961q2) & qtr<=tq(1978q4) (71 observations)."""
    frame = lutkepohl[(lutkepohl.qtr >= "1960-10-01") & (lutkepohl.qtr <= "1978-12-31")]
    result = oe.var(data=frame.reset_index(drop=True), y=NAMES, time="qtr", lags=2)
    assert result.nobs == 71
    published = [(.5456253, 0.0, .545625), (-.3785754, .3853982, .540232),
                 (-.3785754, -.3853982, .540232), (-.0643276, .4595944, .464074),
                 (-.0643276, -.4595944, .464074), (-.3698058, 0.0, .369806)]
    roots = result.extra["stability"]["eigenvalues"]
    assert result.extra["stability"]["stable"] is True and len(roots) == 6
    for root, (real, imag, modulus) in zip(roots, published, strict=True):
        assert root["real"] == pytest.approx(real, abs=6e-8)
        assert abs(root["imag"]) == pytest.approx(abs(imag), abs=6e-8)
        assert root["modulus"] == pytest.approx(modulus, abs=6e-7)


# irf table oirf fevd, impulse(dln_inc) response(dln_consump) noci std: step, oirf, S.E., fevd, S.E.
ORDER_A = [(.005123, .000878, 0, 0), (.001635, .000984, .288494, .077483),
           (.002948, .000993, .294288, .073722), (-.000221, .000662, .322454, .075562),
           (.000811, .000586, .319227, .074063), (.000462, .000333, .322579, .075019),
           (.000044, .000275, .323552, .075371), (.000151, .000162, .323383, .075314),
           (.000091, .000114, .323499, .075386)]
ORDER_B = [(.005461, .000925, 0, 0), (.001578, .000988, .327807, .08159),
           (.003307, .001042, .328795, .077519), (-.00019, .000676, .370775, .080604),
           (.000846, .000617, .366896, .079019), (.000491, .000349, .370399, .079941),
           (.000069, .000292, .371487, .080323), (.000158, .000172, .371315, .080287),
           (.000096, .000122, .371438, .080366)]
# irf table fevd, ... individual: 95% lower and upper bounds of the FEVD (ordering a).
FEVD_BOUNDS_A = [(0, 0), (.13663, .440357), (.149797, .43878), (.174356, .470552),
                 (.174066, .464389), (.175544, .469613), (.175826, .471277), (.17577, .470995),
                 (.175744, .471253)]


@pytest.mark.parametrize("order, published", [
    (["dln_inv", "dln_inc", "dln_consump"], ORDER_A),
    (["dln_inc", "dln_inv", "dln_consump"], ORDER_B)])
def test_published_irf_table(lutkepohl, order, published):
    """[TS] irf table, example 1: var dln_inv dln_inc dln_consump; irf create ..., step(8)."""
    frame = lutkepohl.iloc[1:].reset_index(drop=True)
    result = oe.var(data=frame, y=order, time="qtr", lags=2)
    assert result.nobs == 89
    table = oe.irf(result, steps=8, kind="orthogonalized")
    block = table[(table.impulse == "dln_inc") & (table.response == "dln_consump")]
    assert list(block.step) == list(range(9))
    expected = np.array(published)
    assert_allclose(block.irf, expected[:, 0], atol=6e-7)
    assert_allclose(block.std_error, expected[:, 1], atol=6e-7)
    assert_allclose(block.fevd, expected[:, 2], atol=6e-7)
    assert_allclose(block.fevd_std_error, expected[:, 3], atol=6e-7)
    # The arrays stored with the fit are the same numbers.
    stored = result.extra["irf"]
    r, c = order.index("dln_consump"), order.index("dln_inc")
    assert_allclose(np.array(stored["orthogonalized"])[:, r, c], expected[:, 0], atol=6e-7)
    assert_allclose(np.array(stored["se"]["fevd"])[:, r, c], expected[:, 3], atol=6e-7)
    if order[0] == "dln_inv":
        bounds = np.array(FEVD_BOUNDS_A)
        z = 1.959963984540054
        assert_allclose(block.fevd - z * block.fevd_std_error, bounds[:, 0], atol=6e-7)
        assert_allclose(block.fevd + z * block.fevd_std_error, bounds[:, 1], atol=6e-7)


def test_published_var_with_exogenous_variable(until_1978):
    """[TS] var, example 2: var dln_inc dln_consump if qtr<=tq(1978q4), dfk exog(dln_inv)."""
    result = oe.var(data=until_1978, y=["dln_inc", "dln_consump"], x=["dln_inv"], time="qtr",
                    lags=2, dfk=True)
    metrics = result.metrics
    assert result.nobs == 73
    assert metrics["log_likelihood"] == pytest.approx(478.5663, abs=6e-5)
    assert metrics["aic_per_obs"] == pytest.approx(-12.78264, abs=6e-6)
    assert metrics["hqic_per_obs"] == pytest.approx(-12.63259, abs=6e-6)
    assert metrics["sbic_per_obs"] == pytest.approx(-12.40612, abs=6e-6)
    assert metrics["fpe"] == pytest.approx(9.64e-09, rel=6e-4)
    assert metrics["det_sigma_ml"] == pytest.approx(6.93e-09, rel=1e-3)
    header = [("dln_inc", 6, .011917, 0.0702, 5.059587, 0.4087),
              ("dln_consump", 6, .009197, 0.2794, 25.97262, 0.0001)]
    for row, (name, parms, rmse, r2, chi2, p) in zip(result.extra["equations"], header,
                                                     strict=True):
        assert (row["equation"], row["parms"], row["df"]) == (name, parms, 5)
        assert row["rmse"] == pytest.approx(rmse, abs=6e-7)
        assert row["r_squared"] == pytest.approx(r2, abs=6e-5)
        assert row["statistic"] == pytest.approx(chi2, rel=6e-7)
        assert row["p_value"] == pytest.approx(p, abs=6e-5)
    published = {   # coefficient, std. err., [95% conf. interval]
        "dln_inc:L1.dln_inc": (-.1343345, .1391074, -.4069801, .1383111),
        "dln_inc:L2.dln_inc": (.0120331, .1380346, -.2585097, .2825759),
        "dln_inc:L1.dln_consump": (.3235342, .1652769, -.0004027, .647471),
        "dln_inc:L2.dln_consump": (.0754177, .1648624, -.2477066, .398542),
        "dln_inc:dln_inv": (.0151546, .0302319, -.0440987, .074408),
        "dln_inc:Intercept": (.0145136, .0043815, .0059259, .0231012),
        "dln_consump:L1.dln_inc": (.2425719, .1073561, .0321578, .452986),
        "dln_consump:L2.dln_inc": (.3487949, .1065281, .1400036, .5575862),
        "dln_consump:L1.dln_consump": (-.3119629, .1275524, -.5619611, -.0619648),
        "dln_consump:L2.dln_consump": (-.0128502, .1272325, -.2622213, .2365209),
        "dln_consump:dln_inv": (.0503616, .0233314, .0046329, .0960904),
        "dln_consump:Intercept": (.0131013, .0033814, .0064738, .0197288)}
    rows = {c.term: c for c in result.coefficients}
    assert list(rows) == list(published)
    for term, (estimate, error, low, high) in published.items():
        assert rows[term].estimate == pytest.approx(estimate, abs=6e-8)
        assert rows[term].std_error == pytest.approx(error, abs=6e-8)
        assert rows[term].ci_low == pytest.approx(low, abs=6e-7)
        assert rows[term].ci_high == pytest.approx(high, abs=6e-7)


def test_published_varsoc_after_var_with_exogenous_variable(lutkepohl):
    """[TS] varsoc, example 2: var dln_inc dln_consump ..., lutstats exog(l.dln_inv); varsoc.

    The printed criteria are Lutkepohl's versions, ln|Sigma| + penalty * j K^2 / T:
    Stata's standard ones minus K (1 + ln 2 pi) and minus the penalty of the
    two exogenous terms (L.dln_inv and the constant).
    """
    frame = lutkepohl[lutkepohl.qtr <= "1978-12-31"].copy()
    frame["l_dln_inv"] = frame.dln_inv.shift(1)
    frame = frame.iloc[1:].reset_index(drop=True)
    # The first row only supplies lagged values; its (missing) exogenous value is never used.
    frame.loc[0, "l_dln_inv"] = 0.0
    result = oe.var(data=frame, y=["dln_inc", "dln_consump"], x=["l_dln_inv"], time="qtr", lags=2)
    record = result.extra["lag_order_selection"]
    assert record["n_obs"] == 73 and record["maxlag"] == 2
    rows = record["rows"]
    assert_allclose([r["ll"] for r in rows], [460.646, 467.606, 477.087], atol=6e-4)
    assert_allclose([r["lr"] for r in rows[1:]], [13.919, 18.962], atol=6e-4)
    assert [r["df"] for r in rows[1:]] == [4, 4]
    assert_allclose([r["p_value"] for r in rows[1:]], [0.008, 0.001], atol=6e-4)
    assert_allclose([r["fpe"] for r in rows], [1.3e-08, 1.2e-08, 1.0e-08], rtol=4e-2)
    t, k, exogenous = 73, 2, 2
    constant = k * (1 + np.log(2 * np.pi))
    published = {"aic": ([-18.2962, -18.3773, -18.5275], 2.0),
                 "hqic": ([-18.2962, -18.3273, -18.4274], 2.0 * np.log(np.log(t))),
                 "sbic": ([-18.2962, -18.2518, -18.2764], np.log(t))}
    for name, (values, penalty) in published.items():
        ours = [r[name] - constant - penalty * k * exogenous / t for r in rows]
        assert_allclose(ours, values, atol=6e-5)
    assert record["selected"] == {"fpe": 2, "aic": 2, "hqic": 2, "sbic": 0, "lr": 2}
    assert result.metrics["log_likelihood"] == pytest.approx(477.087, abs=6e-4)


def test_published_vecrank_and_vec_table_arithmetic():
    """The conventions behind the tables printed in [TS] vecrank and [TS] vec.

    The data of those examples (balance2, rdinc) are not available here, so the
    test recomputes the printed columns from the other printed columns with
    OpenEcon's formulas: parameter counts, information criteria, the
    degrees-of-freedom d and the equation chi2.
    """
    # vecrank y i c, lags(5): K = 3, T = 91, trend(constant): m1 = 3, m2 = 3 * 4 + 1.
    k, t, m1, m2 = 3, 91, 3, 13
    ll = [1231.1041, 1245.3882, 1252.5055, 1254.1787]
    parms = [k * m2 + (k + m1 - r) * r for r in range(4)]
    assert parms == [39, 44, 47, 48]
    sbic = [(-2 * a + np.log(t) * b) / t for a, b in zip(ll, parms, strict=True)]
    hqic = [(-2 * a + 2 * np.log(np.log(t)) * b) / t for a, b in zip(ll, parms, strict=True)]
    aic = [(-2 * a + 2 * b) / t for a, b in zip(ll, parms, strict=True)]
    assert_allclose(sbic, [-25.12401, -25.19009, -25.19781, -25.18501], atol=1e-5)
    assert_allclose(hqic, [-25.76596, -25.91435, -25.97144, -25.97511], atol=1e-5)
    assert_allclose(aic, [-26.20009, -26.40414, -26.49463, -26.50942], atol=1e-5)
    eigenvalues = np.array([0.26943, 0.14480, 0.03611])
    trace = [-t * np.log(1 - eigenvalues[r:]).sum() for r in range(3)]
    assert_allclose(trace, [46.1492, 17.5810, 3.3465], atol=2e-3)
    assert_allclose(-t * np.log(1 - eigenvalues), [28.5682, 14.2346, 3.3465], atol=2e-3)
    assert_allclose(2 * np.diff(ll), [28.5682, 14.2346, 3.3465], atol=2e-4)
    # vec ln_ne ln_se: K = 2, r = 1, lags 2, T = 53: 9 free parameters, d = 4.
    k, r, t, m1, m2 = 2, 1, 53, 2, 3
    count = k * m2 + (k + m1 - r) * r
    d = count // k
    assert (count, d) == (9, 4)
    ll = 300.6224
    assert (-2 * ll + 2 * count) / t == pytest.approx(-11.00462, abs=1e-5)
    assert (-2 * ll + 2 * np.log(np.log(t)) * count) / t == pytest.approx(-10.87595, abs=1e-5)
    assert (-2 * ll + np.log(t) * count) / t == pytest.approx(-10.67004, abs=1e-5)
    # One parameter more or fewer moves the criteria by at least 2 / T = 0.038.
    # Equation table: chi2 = (T - d) R2 / (1 - R2) with the printed (rounded) R-squared; T
    # instead of T - d misses the printed value. (The identity holds for the uncentered
    # R-squared with all coefficients and for the centered one with the slopes only; the
    # size of the printed R-squared of a growth rate, 0.93, points to the uncentered one.)
    for r2, chi2 in ((0.9313, 664.4668), (0.9292, 642.7179)):
        low, high = [(t - d) * v / (1 - v) for v in (r2 - 5e-5, r2 + 5e-5)]
        assert low < chi2 < high
        assert not t * (r2 - 5e-5) / (1 - r2 + 5e-5) < chi2 < t * (r2 + 5e-5) / (1 - r2 - 5e-5)
    # Cointegrating equation: chi2(1) is the squared z of the single free coefficient.
    assert (-.9433708 / .0054643) ** 2 == pytest.approx(29805.02, rel=2e-4)
