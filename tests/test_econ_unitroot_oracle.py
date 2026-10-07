"""Independent verification of the unit-root family (verify stage).

Four kinds of evidence, none of which reuses the implementation's formulas:

1. Stata's own printed results. The worked examples of the Stata 19 manuals
   ([TS] dfuller, [TS] pperron, [TS] dfgls) use two classic series that are
   typed in below (Box and Jenkins's airline passengers; Luetkepohl's West
   German investment), so the printed statistics, regression tables, lag
   choices and critical values are reproduced digit for digit. Conventions
   that the manuals' other examples pin down ([XT] xtunitroot ht, [TS] estat
   sbsingle, [TS] estat sbcusum) are checked by arithmetic.
2. Second transcriptions of the typed tables: R's plm (Levin-Lin-Chu and
   Im-Pesaran-Shin), R's urca (Elliott-Rothenberg-Stock, Zivot-Andrews),
   statsmodels (KPSS), and a simulation of the limiting process for Andrews's
   sup-Wald critical values.
3. Statistics recomputed with explicit loops, statsmodels OLS and different
   algebra (refitting instead of running cross products, Wald tests on
   interacted regressions, prediction-error forms).
4. Invariances that the theory implies (level, scale, row order, per-panel
   transformations) and Monte Carlo evidence on the two conventions that
   matter for size (the LLC long-run variance step and the Harris-Tzavalis T).
5. A second pass: Fuller's tables as stored in R's fUnitRoots and tseries, the
   Stata formulas of the Breitung, Fisher and Kao statistics with explicit
   loops, lag-order choices, and the nested factorization behind them.
6. Monte Carlo under the null for every tabulated reference that had none
   (MacKinnon, KPSS, Zivot-Andrews, Hadri, CUSUM), with coarse tolerances.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss as sm_kpss

import openecon as oe
from openecon.econometrics.unitroot import critical, tables

LEVELS = ("1%", "5%", "10%")

# Box, Jenkins and Reinsel, Series G: international airline passengers, monthly, 1949-1960.
AIR = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]
# Luetkepohl (1993), Table E.1: West German fixed investment, quarterly, 1960q1-1982q4.
INVESTMENT = [
    180, 179, 185, 192, 211, 202, 207, 214, 231, 229, 234, 237, 206, 250, 259, 263,
    264, 280, 282, 292, 286, 302, 304, 307, 317, 314, 306, 304, 292, 275, 273, 301,
    280, 289, 303, 322, 315, 339, 364, 371, 375, 432, 453, 460, 475, 496, 494, 498,
    526, 519, 516, 531, 573, 551, 538, 532, 558, 524, 525, 519, 526, 510, 519, 538,
    549, 570, 559, 584, 611, 597, 603, 619, 635, 658, 675, 700, 692, 759, 782, 816,
    844, 830, 853, 852, 833, 860, 870, 830, 801, 824, 831, 830,
]


@pytest.fixture(scope="module")
def air():
    return pd.DataFrame({"air": np.array(AIR, dtype=float)})


@pytest.fixture(scope="module")
def investment():
    return pd.DataFrame({"ln_inv": np.log(np.array(INVESTMENT, dtype=float))})


def make_panel(seed, n, periods, drift=0.0, rho=1.0, ma=0.0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, periods + 1)) * rng.uniform(0.5, 2.0, size=(n, 1))
    u = e[:, 1:] + ma * e[:, :-1] + drift
    y = np.zeros((n, periods))
    for t in range(1, periods):
        y[:, t] = rho * y[:, t - 1] + u[:, t]
    y += rng.normal(size=(n, 1)) * 4
    frame = pd.DataFrame({"id": np.repeat(np.arange(n), periods),
                          "year": np.tile(np.arange(1970, 1970 + periods), n), "y": y.ravel()})
    return frame, y


# ---- 1. Stata's printed results ----------------------------------------------------------


def test_stata_manual_dfuller_example(air):
    # . dfuller air, lags(3) trend regress        ([TS] dfuller, example 1)
    result = oe.dfuller(air, "air", lags=3, trend="trend", regress=True)
    attrs = result.attrs
    assert attrs["nobs"] == 140 and attrs["lags"] == 3
    assert attrs["statistic"] == pytest.approx(-6.936, abs=5e-4)
    assert attrs["p_value"] < 5e-5                              # printed as 0.0000
    table = result["regression"]
    assert list(table.index) == ["L1.air", "LD.air", "L2D.air", "L3D.air", "_trend",
                                 "Intercept"]
    assert_allclose(table["coefficient"],
                    [-.5217089, .5572871, .095912, .14511, 1.407534, 44.49164], rtol=2e-6)
    assert_allclose(table["std_error"],
                    [.0752195, .0799894, .0876692, .0879922, .2098378, 7.78335], rtol=2e-6)
    assert_allclose(table["statistic"], [-6.94, 6.97, 1.09, 1.65, 6.71, 5.72], atol=5e-3)
    assert_allclose(table["p_value"], [0, 0, .276, .101, 0, 0], atol=5e-4)
    assert_allclose(table["ci_low"],
                    [-.67048, .399082, -.0774825, -.0289232, .9925118, 29.09753], rtol=3e-6)
    assert_allclose(table["ci_high"],
                    [-.3729379, .7154923, .2693065, .3191433, 1.822557, 59.88575], rtol=3e-6)
    # Stata prints critical values interpolated in Fuller's table (-4.027, -3.445, -3.145);
    # MacKinnon's response surface, reported here, agrees to the third decimal.
    assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                    [-4.027, -3.445, -3.145], atol=4e-3)


def test_stata_manual_dfuller_on_investment(investment):
    # . dfuller ln_inv, lag(4) trend   and   lag(7) trend      ([TS] dfgls, example 1)
    for lags, statistic, p_value, nobs in ((4, -3.133, 0.0987, 87), (7, -3.994, 0.0090, 84)):
        attrs = oe.dfuller(investment, "ln_inv", lags=lags, trend="trend").attrs
        assert attrs["nobs"] == nobs
        assert attrs["statistic"] == pytest.approx(statistic, abs=5e-4)
        assert attrs["p_value"] == pytest.approx(p_value, abs=5e-5)


def test_stata_manual_pperron_example(air):
    # . pperron air, lags(4) trend regress        ([TS] pperron, example 1)
    result = oe.pperron(air, "air", lags=4, trend="trend", regress=True)
    attrs = result.attrs
    assert attrs["nobs"] == 143 and attrs["lags"] == 4
    assert attrs["z_rho"] == pytest.approx(-46.405, abs=5e-4)
    assert attrs["z_t"] == pytest.approx(-5.049, abs=5e-4)
    assert attrs["p_value"] == pytest.approx(0.0002, abs=5e-5)
    # Z(rho): Fuller's table interpolated in n, exactly as printed.
    assert_allclose([attrs["critical_values_rho"][level] for level in LEVELS],
                    [-27.687, -20.872, -17.643], atol=5e-4)
    # Z(t): Stata prints -4.026, -3.444, -3.144 (Fuller); MacKinnon agrees to 3e-3.
    assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                    [-4.026, -3.444, -3.144], atol=4e-3)
    table = result["regression"]
    assert list(table.index) == ["L1.air", "_trend", "Intercept"]
    assert_allclose(table["coefficient"], [.7318116, .7107559, 25.95168], rtol=2e-6)
    assert_allclose(table["std_error"], [.0578092, .1670563, 7.325951], rtol=2e-6)
    assert_allclose(table["statistic"], [12.66, 4.25, 3.54], atol=5e-3)
    assert_allclose(table["ci_low"], [.6175196, .3804767, 11.46788], rtol=3e-6)
    assert_allclose(table["ci_high"], [.8461035, 1.041035, 40.43547], rtol=3e-6)
    # The default truncation of the same call is int(4 (143/100)^(2/9)) = 4.
    assert oe.pperron(air, "air", trend="trend").attrs["lags"] == 4


def test_stata_manual_dfgls_example(investment):
    # . dfgls ln_inv                               ([TS] dfgls, example 1)
    result = oe.dfgls(investment, "ln_inv")
    attrs = result.attrs
    assert attrs["maxlag"] == 11 and attrs["nobs"] == 80        # Schwert criterion; N = 80
    printed = {11: -2.925, 10: -2.671, 9: -2.766, 8: -3.259, 7: -3.536, 6: -3.115,
               5: -3.054, 4: -3.016, 3: -2.071, 2: -1.675, 1: -1.752}
    by_lag = result.set_index("lags")
    for lag, tau in printed.items():
        assert by_lag.loc[lag, "statistic"] == pytest.approx(tau, abs=5e-4)
    # Opt lag (Ng-Perron seq t) = 7 with RMSE = .0388771
    assert attrs["lags_seq_t"] == 7
    assert by_lag.loc[7, "rmse"] == pytest.approx(.0388771, abs=5e-8)
    # Min SIC = -6.169137 at lag 4 with RMSE = .0398949
    assert attrs["lags_sc"] == 4
    assert by_lag.loc[4, "sc"] == pytest.approx(-6.169137, abs=5e-7)
    assert by_lag["sc"].min() == by_lag.loc[4, "sc"]
    assert by_lag.loc[4, "rmse"] == pytest.approx(.0398949, abs=5e-8)
    # Min MAIC = -6.136692 at lag 1 with RMSE = .0440319
    assert attrs["lags_maic"] == 1 and attrs["lags"] == 1
    assert by_lag.loc[1, "maic"] == pytest.approx(-6.136692, abs=1e-6)
    assert by_lag["maic"].min() == by_lag.loc[1, "maic"]
    assert by_lag.loc[1, "rmse"] == pytest.approx(.0440319, abs=5e-8)
    assert attrs["statistic"] == pytest.approx(-1.752, abs=5e-4)
    # The 1% critical value printed for every lag is the Elliott-Rothenberg-Stock entry
    # interpolated at the 92 observations of the series: -3.610.
    assert attrs["nobs_series"] == 92
    assert attrs["critical_values"]["1%"] == pytest.approx(-3.610, abs=5e-4)
    assert (result["critical_1pct"] == attrs["critical_values"]["1%"]).all()
    # Stata's default 5% / 10% values are Cheung-Lai's (lag dependent, not reproduced); at
    # one lag they are -3.055 and -2.762, next to the ERS values reported here.
    assert attrs["critical_values"]["5%"] == pytest.approx(-3.055, abs=2e-3)
    assert attrs["critical_values"]["10%"] == pytest.approx(-2.762, abs=3e-3)


def test_stata_manual_conventions_by_arithmetic():
    # [XT] xtunitroot ht, example 2: 151 panels, 34 periods, rho = 0.8184, z = -13.1239.
    # The printed z needs T = 34 (the number of periods) in the moment formulas; with
    # T - 1 = 33 (Stata's altt, the definition in Harris and Tzavalis) it would be -12.43.
    def z(rho, t, count=151):
        mean = 1 - 3 / (t + 1)
        variance = 3 * (17 * t * t - 20 * t + 17) / (5 * (t - 1) * (t + 1) ** 3)
        return math.sqrt(count) * (rho - mean) / math.sqrt(variance)

    assert z(0.81835, 34) < -13.1239 < z(0.81845, 34)           # rho is printed to 4 digits
    assert not z(0.81835, 33) < -13.1239 < z(0.81845, 33)
    frame, _ = make_panel(1, 9, 20)
    default = oe.xtunitroot(frame, "y", "id", "year", test="ht")
    alternative = oe.xtunitroot(frame, "y", "id", "year", test="ht", altt=True)
    rho = default.attrs["rho"]
    assert alternative.attrs["rho"] == rho
    assert default.attrs["statistic"] == pytest.approx(z(rho, 20, 9), rel=1e-10)
    assert alternative.attrs["statistic"] == pytest.approx(z(rho, 19, 9), rel=1e-10)
    assert any("altt=True" in note for note in default.attrs["notes"])
    assert not any("altt=True" in note for note in alternative.attrs["notes"])

    # [TS] estat sbsingle, example 1: 222 observations, 15% trimming. "Trimmed sample:
    # 1964q1 thru 2002q3" are observations 35 and 189: the first regime holds 34..188
    # observations, i.e. ceil(0.15 * 222) = 34 at each end.
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"t": np.arange(1, 223), "x": rng.standard_normal(222),
                         "y": rng.standard_normal(222)})
    attrs = oe.sbsingle(data, "y", ["x"], time="t").attrs
    assert (attrs["first_candidate"], attrs["last_candidate"]) == (35, 189)
    assert attrs["candidates"] == 155

    # [TS] estat sbcusum prints the critical values 1.1430, 0.9479, 0.8499.
    values = oe.cusum(data, "y", ["x"]).attrs["critical_values"]
    assert_allclose([values[level] for level in LEVELS], [1.1430, 0.9479, 0.8499], atol=5e-5)
    for level, alpha in zip(LEVELS, (0.01, 0.05, 0.10), strict=True):
        a = values[level]                                       # Brown-Durbin-Evans eq. (2.6)
        assert stats.norm.sf(3 * a) + math.exp(-4 * a * a) * stats.norm.cdf(a) \
            == pytest.approx(alpha / 2, rel=1e-9)

    # [XT] xtunitroot llc, example 1: 34 periods, "10.00 lags average (chosen by LLC)".
    frame, _ = make_panel(2, 6, 34)
    assert oe.xtunitroot(frame, "y", "id", "year", test="llc").attrs["kernel_lags"] == 10 \
        == int(3.21 * 34 ** (1 / 3))


# ---- 2. Second transcriptions of the tables ---------------------------------------------

# R package plm, R/test_uroot.R (adj.levinlin): mu* and sigma* for T~ = 25..250 and the
# asymptotic values, for the models without deterministics, with intercepts, with trends.
PLM_LLC_T = (25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 250)
PLM_LLC = {
    "none": ((0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000,
              0.000, 0.000),
             (1.049, 1.035, 1.027, 1.021, 1.017, 1.014, 1.011, 1.008, 1.007, 1.006, 1.005,
              1.001, 1.000)),
    "constant": ((-0.554, -0.546, -0.541, -0.537, -0.533, -0.531, -0.527, -0.524, -0.521,
                  -0.520, -0.518, -0.509, -0.500),
                 (0.919, 0.889, 0.867, 0.850, 0.837, 0.826, 0.810, 0.798, 0.789, 0.782,
                  0.776, 0.742, 0.707)),
    "trend": ((-0.703, -0.674, -0.653, -0.637, -0.624, -0.614, -0.598, -0.587, -0.578,
               -0.571, -0.566, -0.533, -0.500),
              (1.003, 0.949, 0.906, 0.871, 0.842, 0.818, 0.780, 0.751, 0.728, 0.710, 0.695,
               0.603, 0.500)),
}
# R package plm (adj.ips.wtbar.*): rows p = 0..8, columns T = 10, 15, 20, 25, 30, 40, 50,
# 60, 70, 100; nan where Im, Pesaran and Shin leave the cell empty.
_ = float("nan")
PLM_IPS = {
    ("constant", "mean"): (
        -1.504, -1.514, -1.522, -1.520, -1.526, -1.523, -1.527, -1.519, -1.524, -1.532,
        -1.488, -1.503, -1.516, -1.514, -1.519, -1.520, -1.524, -1.519, -1.522, -1.530,
        -1.319, -1.387, -1.428, -1.443, -1.460, -1.476, -1.493, -1.490, -1.498, -1.514,
        -1.306, -1.366, -1.413, -1.433, -1.453, -1.471, -1.489, -1.486, -1.495, -1.512,
        -1.171, -1.260, -1.329, -1.363, -1.394, -1.428, -1.454, -1.458, -1.470, -1.495,
        _, _, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494,
        _, _, _, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476,
        _, _, _, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474,
        _, _, _, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456),
    ("constant", "variance"): (
        1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735,
        1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745,
        1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754,
        1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761,
        2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771,
        _, _, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781,
        _, _, _, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795,
        _, _, _, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806,
        _, _, _, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818),
    ("trend", "mean"): (
        -2.166, -2.167, -2.168, -2.167, -2.172, -2.173, -2.176, -2.174, -2.174, -2.177,
        -2.173, -2.169, -2.172, -2.172, -2.173, -2.177, -2.180, -2.178, -2.176, -2.179,
        -1.914, -1.999, -2.047, -2.074, -2.095, -2.120, -2.137, -2.143, -2.146, -2.158,
        -1.922, -1.977, -2.032, -2.065, -2.091, -2.117, -2.137, -2.142, -2.146, -2.158,
        -1.750, -1.823, -1.911, -1.968, -2.009, -2.057, -2.091, -2.103, -2.114, -2.135,
        _, _, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135,
        _, _, _, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113,
        _, _, _, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112,
        _, _, _, -1.761, -1.835, -1.925, -1.987, -2.024, -2.046, -2.088),
    ("trend", "variance"): (
        1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597,
        1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605,
        1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613,
        2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625,
        3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629,
        _, _, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638,
        _, _, _, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650,
        _, _, _, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661,
        _, _, _, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670),
}
# R package urca: ur.ers (DF-GLS with trend; T = 50, 100, 200, infinity at 1%, 5%, 10%)
# and ur.za (models intercept, trend, both at 1%, 5%, 10%).
URCA_ERS = np.array([[-3.77, -3.58, -3.46, -3.48], [-3.19, -3.03, -2.93, -2.89],
                     [-2.89, -2.74, -2.64, -2.57]]).T
URCA_ZA = {"intercept": (-5.34, -4.8, -4.58), "trend": (-4.93, -4.42, -4.11),
           "both": (-5.57, -5.08, -4.82)}


def test_llc_and_ips_tables_equal_the_plm_transcription():
    assert tables.LLC_T[:-1] == tuple(float(t) for t in PLM_LLC_T)
    assert math.isinf(tables.LLC_T[-1])
    for case, (means, deviations) in PLM_LLC.items():
        ours = np.array(tables.LLC_ADJUST[case])
        assert_allclose(ours[:, 0], means, rtol=0, atol=0)
        assert_allclose(ours[:, 1], deviations, rtol=0, atol=0)
    assert tables.IPS_T == (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0)
    for (case, moment), flat in PLM_IPS.items():
        table = tables.IPS_MEAN[case] if moment == "mean" else tables.IPS_VARIANCE[case]
        ours = np.array([[np.nan if value is None else value for value in row]
                         for row in table])
        assert ours.shape == (9, 10)
        assert_allclose(ours, np.array(flat).reshape(9, 10), rtol=0, atol=0, equal_nan=True)


def test_ers_zivot_andrews_and_kpss_tables_equal_urca_and_statsmodels():
    assert tables.ERS_T[:3] == (50.0, 100.0, 200.0) and math.isinf(tables.ERS_T[3])
    assert_allclose(np.array(tables.ERS_TREND), URCA_ERS, rtol=0, atol=0)
    for model, values in URCA_ZA.items():
        assert tables.ZIVOT_ANDREWS[model] == values
        assert critical.zivot_andrews_critical(model) == dict(zip(LEVELS, values, strict=True))
    rng = np.random.default_rng(3)
    x = rng.standard_normal(80)
    for regression, case in (("c", "level"), ("ct", "trend")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = sm_kpss(x, regression=regression, nlags=2)[3]
        assert critical.kpss_critical(case) == theirs


def test_andrews_table_against_a_simulation_of_the_limiting_process():
    # sup over [0.15, 0.85] of ||B(r) - r B(1)||^2 / (r (1 - r)) for a q-vector Brownian
    # motion; the grid makes the simulated supremum slightly too small (about 1%).
    rng = np.random.default_rng(77)
    grid, reps, largest = 600, 6000, 10
    r = np.arange(1, grid + 1) / grid
    keep = (r >= 0.15) & (r <= 0.85)
    sup = np.empty((reps, largest))
    for start in range(0, reps, 500):
        steps = rng.standard_normal((500, grid, largest)) / math.sqrt(grid)
        bridge = steps.cumsum(axis=1)
        bridge -= r[None, :, None] * bridge[:, -1:, :]
        ratio = bridge[:, keep, :] ** 2 / (r * (1 - r))[keep][None, :, None]
        sup[start:start + 500] = ratio.cumsum(axis=2).max(axis=1)
    simulated = np.quantile(sup, [0.90, 0.95], axis=0).T                    # [q, level]
    for q in (1, 2, 3, 5, 8, 10):
        ten, five, _one = tables.QLR_F_15[q - 1]
        assert_allclose(simulated[q - 1], [ten * q, five * q], rtol=0.045)
        values = critical.supwald_critical(q, 0.15)
        assert values["10%"] == pytest.approx(ten * q) and values["5%"] == pytest.approx(five * q)


def test_fuller_table_with_trend_against_simulation():
    rng = np.random.default_rng(11)
    n, reps = 50, 60000                       # Fuller: n (rho-hat - 1), n regression rows
    y = rng.standard_normal((reps, n + 1)).cumsum(axis=1)
    x = np.column_stack([np.ones(n), np.arange(1.0, n + 1)])
    q = np.linalg.qr(x)[0]
    lag, cur = y[:, :-1], y[:, 1:]
    lag = lag - (lag @ q) @ q.T
    cur = cur - (cur @ q) @ q.T
    rho = (lag * cur).sum(axis=1) / (lag ** 2).sum(axis=1)
    assert_allclose(np.quantile(n * (rho - 1), [0.01, 0.05, 0.10]), tables.FULLER_RHO["ct"][1],
                    atol=0.45)


# ---- 3. Own recomputation ----------------------------------------------------------------


def ols(y, x):
    return sm.OLS(y, x).fit()


def test_single_series_tests_are_invariant_to_level_scale_and_row_order():
    rng = np.random.default_rng(5)
    total = 150
    e = rng.standard_normal(total)
    series = np.cumsum(e + 0.4 * np.r_[0, e[:-1]]) + 0.05 * np.arange(total)
    base = pd.DataFrame({"t": np.arange(total), "y": series})

    def run(frame):
        return {
            "adf": oe.dfuller(frame, "y", lags=2, trend="trend", time="t").attrs["statistic"],
            "pp_rho": oe.pperron(frame, "y", time="t").attrs["z_rho"],
            "pp_t": oe.pperron(frame, "y", time="t").attrs["z_t"],
            "gls": oe.dfgls(frame, "y", maxlag=3, time="t").attrs["statistic"],
            "kpss": oe.kpss(frame, "y", lags=4, trend=True, time="t").attrs["statistic"],
            "za": oe.zandrews(frame, "y", lags=1, break_="both", time="t").attrs["statistic"],
        }

    expected = run(base)
    for label, frame, rtol in (
        ("shuffled", base.sample(frac=1.0, random_state=1), 1e-12),
        ("scale 1e8", base.assign(y=base["y"] * 1e8), 1e-9),
        ("scale 1e-8", base.assign(y=base["y"] * 1e-8), 1e-9),
        ("level 1e6", base.assign(y=base["y"] + 1e6), 1e-7),
        ("level 1e9", base.assign(y=base["y"] + 1e9), 1e-4),
        ("scale and level", base.assign(y=base["y"] * 1e-6 - 3.0), 1e-7),
    ):
        got = run(frame)
        for name, value in expected.items():
            assert got[name] == pytest.approx(value, rel=rtol), (label, name)
    # Without a constant the level matters: the no-constant test is not shift invariant.
    plain = oe.dfuller(base, "y", trend="none").attrs["statistic"]
    assert oe.dfuller(base.assign(y=base["y"] + 50), "y", trend="none").attrs["statistic"] \
        != pytest.approx(plain, rel=1e-3)
    # The regression tables of the shifted series are those of OLS on the shifted series.
    shifted = base.assign(y=base["y"] + 1234.5)
    y = shifted["y"].to_numpy()
    dy = np.diff(y)
    n = total - 2
    fit = ols(dy[1:], np.column_stack([y[1:-1], dy[:-1], np.arange(2.0, n + 2), np.ones(n)]))
    table = oe.dfuller(shifted, "y", lags=1, trend="trend", regress=True)["regression"]
    assert_allclose(table["coefficient"], fit.params, rtol=1e-8)
    assert_allclose(table["std_error"], fit.bse, rtol=1e-8)
    fit = ols(y[1:], np.column_stack([y[:-1], np.ones(total - 1)]))
    table = oe.pperron(shifted, "y", regress=True)["regression"]
    assert_allclose(table["coefficient"], fit.params, rtol=1e-8)
    assert_allclose(table["std_error"], fit.bse, rtol=1e-8)
    assert_allclose(table["ci_low"], fit.conf_int()[:, 0], rtol=1e-7)


def test_kpss_and_dfuller_agree_with_statsmodels_on_badly_scaled_series():
    rng = np.random.default_rng(8)
    for scale, shift in ((1e-8, 0.0), (1e8, 0.0), (1.0, 1e7)):
        y = (rng.standard_normal(90).cumsum() + 0.1 * np.arange(90)) * scale + shift
        frame = pd.DataFrame({"y": y})
        for trend, case in (("constant", "c"), ("trend", "ct")):
            expected = adfuller(y - y[0], maxlag=2, regression=case, autolag=None)
            attrs = oe.dfuller(frame, "y", lags=2, trend=trend).attrs
            assert attrs["statistic"] == pytest.approx(expected[0], rel=1e-7)
            assert attrs["p_value"] == pytest.approx(expected[1], rel=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = sm_kpss(y - y[0], regression="ct", nlags=3)
        assert oe.kpss(frame, "y", lags=3, trend=True).attrs["statistic"] \
            == pytest.approx(expected[0], rel=1e-7)


@pytest.mark.parametrize("model", ["intercept", "trend", "both"])
@pytest.mark.parametrize("lags", [0, 2])
def test_zandrews_equals_one_regression_per_break_date(model, lags):
    rng = np.random.default_rng(21 + lags)
    total = 70
    y = np.cumsum(rng.standard_normal(total)) + 3.0 * (np.arange(total) >= 40) + 7.0
    dy = np.diff(y)
    edge = int(0.15 * total)
    best = (np.inf, None)
    count = 0
    for tb in range(max(edge + 1, lags + 3), min(total - edge, total - 2) + 1):   # 1-based TB
        t = np.arange(lags + 2, total + 1)                                        # 1-based t
        columns = [np.ones(len(t)), t.astype(float), y[t - 2]]
        columns += [dy[t - 2 - j] for j in range(1, lags + 1)]
        if model in ("intercept", "both"):
            columns.append((t > tb).astype(float))
        if model in ("trend", "both"):
            columns.append(np.where(t > tb, t - tb, 0.0))
        fit = ols(dy[t - 2], np.column_stack(columns))
        count += 1
        if fit.tvalues[2] < best[0]:
            best = (fit.tvalues[2], tb)
    result = oe.zandrews(pd.DataFrame({"y": y}), "y", break_=model, lags=lags)
    attrs = result.attrs
    assert attrs["statistic"] == pytest.approx(best[0], rel=1e-8)
    assert attrs["break_index"] == best[1] - 1 and attrs["break_period"] == best[1] - 1
    assert attrs["candidates"] == count
    assert attrs["critical_values"] == dict(zip(LEVELS, URCA_ZA[model], strict=True))
    assert attrs["p_value"] is None


def test_egranger_error_correction_step_and_level_invariance():
    rng = np.random.default_rng(13)
    total = 140
    x1 = np.cumsum(rng.standard_normal(total))
    x2 = np.cumsum(rng.standard_normal(total))
    u = np.zeros(total)
    for t in range(1, total):
        u[t] = 0.6 * u[t - 1] + rng.standard_normal()
    y = 2.0 + 0.8 * x1 - 0.5 * x2 + u
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    lags = 2
    result = oe.egranger(frame, "y", ["x1", "x2"], lags=lags, ecm=True, regress=True)
    first = ols(y, np.column_stack([x1, x2, np.ones(total)]))
    e = first.resid
    de = np.diff(e)
    adf = ols(de[lags:], np.column_stack([e[lags:-1], de[1:-1], de[:-2]]))
    assert result.attrs["statistic"] == pytest.approx(adf.tvalues[0], rel=1e-8)
    assert_allclose(result["adf_regression"]["coefficient"], adf.params, rtol=1e-7)
    assert_allclose(result["cointegrating_regression"]["coefficient"], first.params, rtol=1e-8)
    assert_allclose(result["cointegrating_regression"]["std_error"], first.bse, rtol=1e-8)
    d = np.diff(np.column_stack([y, x1, x2]), axis=0)
    rows = np.arange(lags, total - 1)                          # Delta at t = rows + 1
    design = [e[rows]] + [d[rows - j, c] for j in (1, 2) for c in range(3)]
    ecm = ols(d[rows, 0], np.column_stack([*design, np.ones(len(rows))]))
    table = result["ecm"]
    assert list(table.index) == ["L1.e", "LD.y", "LD.x1", "LD.x2", "L2D.y", "L2D.x1", "L2D.x2",
                                 "Intercept"]
    assert_allclose(table["coefficient"], ecm.params, rtol=1e-7)
    assert_allclose(table["std_error"], ecm.bse, rtol=1e-7)
    assert_allclose(table["p_value"], ecm.pvalues, rtol=1e-6, atol=1e-12)
    assert result.attrs["adjustment"] == pytest.approx(ecm.params[0], rel=1e-8)
    # Levels that dwarf the variation used to make a regressor "collinear" with the
    # constant and silently drop it: the test must not depend on the level at all.
    moved = frame.assign(y=frame["y"] + 1e9, x1=frame["x1"] - 3e9, x2=frame["x2"] * 1e-6 + 5.0)
    other = oe.egranger(moved, "y", ["x1", "x2"], lags=lags)
    assert other.attrs["omitted"] == []
    assert other.attrs["statistic"] == pytest.approx(result.attrs["statistic"], rel=1e-5)
    assert other.attrs["cointegrating_vector"]["x1"] == pytest.approx(first.params[0], rel=1e-6)


def test_chow_equals_the_interacted_regression_and_the_prediction_error_form():
    rng = np.random.default_rng(31)
    total, split = 90, 55
    x = np.column_stack([np.ones(total), rng.standard_normal((total, 2))])
    y = x @ [1.0, 0.5, -0.3] + rng.standard_normal(total) + 0.8 * (np.arange(total) >= split)
    frame = pd.DataFrame({"y": y, "a": x[:, 1], "b": x[:, 2], "t": np.arange(2000, 2000 + total)})
    result = oe.chow(frame, "y", ["a", "b"], 2000 + split, time="t")
    k = 3
    after = (np.arange(total) >= split).astype(float)[:, None]
    interacted = ols(y, np.column_stack([x, x * after]))
    restriction = np.hstack([np.zeros((k, k)), np.eye(k)])
    f_test = interacted.f_test(restriction)
    assert result.loc["chow_f", "statistic"] == pytest.approx(float(f_test.fvalue), rel=1e-8)
    assert result.loc["chow_f", "p_value"] == pytest.approx(float(f_test.pvalue), rel=1e-7)
    assert (result.loc["chow_f", "df"], result.loc["chow_f", "df2"]) == (k, total - 2 * k)
    wald = interacted.wald_test(restriction, use_f=False, scalar=True)
    assert result.loc["wald", "statistic"] == pytest.approx(float(wald.statistic), rel=1e-8)
    assert result.loc["wald", "p_value"] == pytest.approx(float(wald.pvalue), rel=1e-7)
    pooled = ols(y, x)
    likelihood_ratio = 2 * (interacted.llf - pooled.llf)
    assert result.loc["lr", "statistic"] == pytest.approx(likelihood_ratio, rel=1e-8)
    assert result.loc["lr", "p_value"] == pytest.approx(stats.chi2.sf(likelihood_ratio, k),
                                                        rel=1e-7)
    # Chow's predictive test from the prediction errors of the first-regime fit.
    x1, x2, y1, y2 = x[:split], x[split:], y[:split], y[split:]
    first = ols(y1, x1)
    error = y2 - x2 @ first.params
    spread = np.eye(total - split) + x2 @ np.linalg.inv(x1.T @ x1) @ x2.T
    forecast = (error @ np.linalg.solve(spread, error) / (total - split)) / first.mse_resid
    assert result.loc["forecast_f", "statistic"] == pytest.approx(forecast, rel=1e-8)
    assert result.loc["forecast_f", "p_value"] == pytest.approx(
        stats.f.sf(forecast, total - split, split - k), rel=1e-7)
    assert result.attrs["break_period"] == 2000 + split and result.attrs["break_index"] == split
    # Level and scale of the regressors do not matter when the constant breaks too.
    moved = frame.assign(a=frame["a"] * 1e6 + 4e9, y=frame["y"] * 1e-7 - 2.0)
    other = oe.chow(moved, "y", ["a", "b"], 2000 + split, time="t")
    assert_allclose(other["statistic"], result["statistic"], rtol=1e-5)


@pytest.mark.parametrize("trim", [0.15, 0.2, 0.07])
def test_sbsingle_equals_wald_tests_on_interacted_regressions(trim):
    rng = np.random.default_rng(41)
    total, k = 83, 3
    x = np.column_stack([np.ones(total), rng.standard_normal((total, 2))])
    y = x @ [0.3, 1.0, 0.2] + rng.standard_normal(total) + 0.9 * x[:, 1] * (np.arange(total) > 50)
    frame = pd.DataFrame({"y": y, "a": x[:, 1], "b": x[:, 2]})
    edge = max(k + 1, math.ceil(trim * total - 1e-9))
    restriction = np.hstack([np.zeros((k, k)), np.eye(k)])
    path = []
    for n1 in range(edge, total - edge + 1):
        after = (np.arange(total) >= n1).astype(float)[:, None]
        fit = ols(y, np.column_stack([x, x * after]))
        path.append(float(fit.wald_test(restriction, use_f=False, scalar=True).statistic))
    path = np.array(path)
    attrs = oe.sbsingle(frame, "y", ["a", "b"], trim=trim).attrs
    assert attrs["candidates"] == len(path)
    assert attrs["sup_wald"] == pytest.approx(path.max(), rel=1e-7)
    assert attrs["ave_wald"] == pytest.approx(path.mean(), rel=1e-7)
    assert attrs["exp_wald"] == pytest.approx(math.log(np.mean(np.exp(path / 2))), rel=1e-7)
    assert attrs["break_index"] == edge + int(path.argmax()) and attrs["df"] == k
    if trim == 0.15:
        assert attrs["critical_values"] == pytest.approx(
            {"10%": 4.09 * 3, "5%": 4.71 * 3, "1%": 6.02 * 3})
    else:
        assert attrs["critical_values"] is None


def test_sup_wald_p_value_against_the_table_and_the_stata_manual():
    # The DeLong / Andrews tail approximation is approximate: at the tabulated critical
    # values it returns between 0.9 and 1.25 times the nominal level.
    for q in (1, 2, 3, 5, 10, 20):
        values = critical.supwald_critical(q, 0.15)
        for level, alpha in zip(LEVELS, (0.01, 0.05, 0.10), strict=True):
            p = critical.supwald_p(values[level], q, 0.15, 0.85)
            assert 0.9 * alpha < p < 1.25 * alpha, (q, level, p)
    # [TS] estat sbsingle, example 1: sup-Wald 14.1966 with 3 coefficients and candidates
    # 34/222..188/222; Stata prints Hansen's (1997) p-value 0.0440. The approximation used
    # here gives 0.051: the same order, not the same digits.
    p = critical.supwald_p(14.1966, 3, 34 / 222, 188 / 222)
    assert p == pytest.approx(0.0440, abs=0.01) and p > 0.0440


def test_cusum_equals_refitting_the_regression_at_every_date():
    rng = np.random.default_rng(51)
    total, k = 64, 3
    x = np.column_stack([np.ones(total), rng.standard_normal((total, 2)).cumsum(axis=0)])
    y = x @ [1.0, 0.4, -0.2] + rng.standard_normal(total) + 1.5 * (np.arange(total) >= 40)
    frame = pd.DataFrame({"y": y, "a": x[:, 1], "b": x[:, 2]})
    w = []
    for t in range(k, total):
        history, target = x[:t], y[:t]
        beta = np.linalg.lstsq(history, target, rcond=None)[0]
        leverage = x[t] @ np.linalg.solve(history.T @ history, x[t])
        w.append((y[t] - x[t] @ beta) / math.sqrt(1 + leverage))
    w = np.array(w)
    result = oe.cusum(frame, "y", ["a", "b"], level="10%")
    assert_allclose(result["recursive_residual"], w, rtol=1e-7, atol=1e-9)
    count = total - k
    sigma = math.sqrt(np.sum((w - w.mean()) ** 2) / count)      # Stata: centred, T - K
    step = np.arange(1, count + 1)
    assert_allclose(result["cusum"], np.cumsum(w) / sigma, rtol=1e-7, atol=1e-9)
    a = result.attrs["critical_values"]["10%"]
    assert_allclose(result["cusum_upper"], a * math.sqrt(count) * (1 + 2 * step / count),
                    rtol=1e-12)
    statistic = np.max(np.abs(np.cumsum(w) / sigma) / (math.sqrt(count)
                                                       * (1 + 2 * step / count)))
    assert result.attrs["statistic"] == pytest.approx(statistic, rel=1e-7)
    assert result.attrs["cusum_crosses"] == bool(statistic > a)
    assert result.attrs["sigma"] == pytest.approx(sigma, rel=1e-7)
    assert result.attrs["sigma_ols"] == pytest.approx(math.sqrt(ols(y, x).mse_resid), rel=1e-8)
    assert_allclose(result["cusumsq"], np.cumsum(w ** 2) / np.sum(w ** 2), rtol=1e-7)
    # Reparametrizing the regressors (levels, scales) leaves recursive residuals unchanged.
    moved = frame.assign(a=frame["a"] * 1e5 + 7e8)
    other = oe.cusum(moved, "y", ["a", "b"], level="10%")
    assert_allclose(other["recursive_residual"], w, rtol=1e-4, atol=1e-6)


def interpolate(nodes, values, at):
    return float(np.interp(at, nodes, values))


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("lags", [0, 1, 3])
def test_llc_from_statsmodels_regressions_and_the_plm_table(trend, lags):
    frame, y = make_panel(61, 9, 43, drift=0.2 if trend == "trend" else 0.0, ma=0.3)
    count, periods = y.shape
    n = periods - lags - 1
    kernel_lags = int(3.21 * periods ** (1 / 3))
    e_all, v_all, ratios = [], [], []
    for series in y:
        dy = np.diff(series)
        others = [dy[lags - j:periods - 1 - j] for j in range(1, lags + 1)]
        if trend != "none":
            others.append(np.ones(n))
        if trend == "trend":
            others.append(np.arange(1.0, n + 1))
        target, level = dy[lags:], series[lags:periods - 1]
        if others:
            others = np.column_stack(others)
            e, v = ols(target, others).resid, ols(level, others).resid
        else:
            e, v = target, level
        sigma = math.sqrt(ols(e, v).ssr / n)
        e_all.append(e / sigma)
        v_all.append(v / sigma)
        d = dy - dy.mean() if trend != "none" else dy           # LLC step 2 / Stata
        m = len(d)
        variance = d @ d / m + 2 * sum((1 - j / (kernel_lags + 1)) * (d[j:] @ d[:-j]) / m
                                       for j in range(1, kernel_lags + 1))
        ratios.append(math.sqrt(variance) / sigma)
    pooled = ols(np.concatenate(e_all), np.concatenate(v_all))
    observations = count * n
    sigma2 = pooled.ssr / observations                          # divisor N T~, no df correction
    se = math.sqrt(sigma2 / np.sum(np.concatenate(v_all) ** 2))
    t_delta = pooled.params[0] / se
    mu = interpolate(PLM_LLC_T, PLM_LLC[trend][0][:-1], n)
    sd = interpolate(PLM_LLC_T, PLM_LLC[trend][1][:-1], n)
    t_star = (t_delta - observations * np.mean(ratios) * se * mu / sigma2) / sd
    attrs = oe.xtunitroot(frame, "y", "id", "year", test="llc", lags=lags, trend=trend).attrs
    assert attrs["kernel_lags"] == kernel_lags and attrs["t_tilde"] == n
    assert attrs["unadjusted_t"] == pytest.approx(t_delta, rel=1e-8)
    assert attrs["coefficient"] == pytest.approx(pooled.params[0], rel=1e-8)
    assert attrs["s_n"] == pytest.approx(np.mean(ratios), rel=1e-8)
    assert (attrs["mu_star"], attrs["sigma_star"]) == pytest.approx((mu, sd))
    assert attrs["statistic"] == pytest.approx(t_star, rel=1e-8)
    assert attrs["p_value"] == pytest.approx(stats.norm.cdf(t_star), rel=1e-7)
    # lrv="null" differs from the published step only with panel means.
    null = oe.xtunitroot(frame, "y", "id", "year", test="llc", lags=lags, trend=trend,
                         lrv="null").attrs
    if trend == "constant":
        assert null["s_n"] > attrs["s_n"] and null["statistic"] > attrs["statistic"]
        assert any("lrv='null'" in note for note in attrs["notes"])
    else:
        assert null["statistic"] == attrs["statistic"]


def test_panel_tests_are_invariant_to_panel_specific_levels_scales_and_trends():
    frame, y = make_panel(71, 10, 36, drift=0.1, ma=0.2)
    count, periods = y.shape
    rng = np.random.default_rng(72)
    scale = rng.uniform(1e-3, 1e3, size=(count, 1))
    level = rng.normal(size=(count, 1)) * 1e6
    slope = rng.normal(size=(count, 1)) * 10
    time = np.arange(periods)[None, :]

    def statistics(values, trend, tests):
        data = frame.assign(y=values.ravel())
        out = {}
        for test, options in tests:
            key = (test, *sorted(options.items()))
            out[key] = oe.xtunitroot(data, "y", "id", "year", test=test, trend=trend,
                                     **options).attrs["statistic"]
        return out

    scale_free = [("llc", {"lags": 1}), ("ips", {"lags": 1}), ("fisher", {"lags": 2}),
                  ("fisher", {"method": "pperron", "lags": 2}), ("breitung", {"lags": 1}),
                  ("hadri", {"robust": True})]
    for trend in ("constant", "trend"):
        moved = y * scale + level + (slope * time if trend == "trend" else 0.0)
        base, other = statistics(y, trend, scale_free), statistics(moved, trend, scale_free)
        for key, value in base.items():
            assert other[key] == pytest.approx(value, rel=2e-6), (trend, key)
        # Pooled statistics are invariant to levels (and trends) and to a common scale.
        pooled = [("ht", {}), ("ht", {"altt": True}), ("hadri", {}), ("hadri", {"kernel_lags": 3})]
        moved = y * 250.0 + level + (slope * time if trend == "trend" else 0.0)
        base, other = statistics(y, trend, pooled), statistics(moved, trend, pooled)
        for key, value in base.items():
            assert other[key] == pytest.approx(value, rel=2e-6), (trend, key)
    # Row order and labels do not matter either.
    shuffled = frame.sample(frac=1.0, random_state=3).assign(
        id=lambda d: d["id"].map(lambda i: f"unit-{9 - i}"))
    for test in ("llc", "ips", "fisher", "hadri", "breitung", "ht"):
        assert oe.xtunitroot(shuffled, "y", "id", "year", test=test).attrs["statistic"] \
            == pytest.approx(oe.xtunitroot(frame, "y", "id", "year", test=test)
                             .attrs["statistic"], rel=1e-10)


@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_ips_from_statsmodels_and_the_plm_table_on_an_unbalanced_panel(trend):
    frame, y = make_panel(81, 8, 44, drift=0.1 if trend == "trend" else 0.0)
    keep = ~(((frame["id"] == 2) & (frame["year"] < 1979)) | ((frame["id"] == 5)
                                                                & (frame["year"] > 2001)))
    unbalanced = frame[keep]
    lags = 2
    t_columns = np.array(tables.IPS_T)
    key = "constant" if trend == "constant" else "trend"
    means = np.array(PLM_IPS[(key, "mean")]).reshape(9, 10)[lags]
    variances = np.array(PLM_IPS[(key, "variance")]).reshape(9, 10)[lags]
    t_values, expected_mean, expected_variance = [], [], []
    for _, group in unbalanced.sort_values(["id", "year"]).groupby("id"):
        series = group["y"].to_numpy()
        t_values.append(adfuller(series, maxlag=lags, autolag=None,
                                 regression="c" if trend == "constant" else "ct")[0])
        nobs = len(series) - lags - 1
        expected_mean.append(interpolate(t_columns, means, nobs))
        expected_variance.append(interpolate(t_columns, variances, nobs))
    w = math.sqrt(8) * (np.mean(t_values) - np.mean(expected_mean)) \
        / math.sqrt(np.mean(expected_variance))
    attrs = oe.xtunitroot(unbalanced, "y", "id", "year", test="ips", lags=lags,
                          trend=trend).attrs
    assert attrs["balanced"] is False
    assert attrs["t_bar"] == pytest.approx(np.mean(t_values), rel=1e-9)
    assert attrs["statistic"] == pytest.approx(w, rel=1e-9)
    assert attrs["p_value"] == pytest.approx(stats.norm.cdf(w), rel=1e-8)


def test_hadri_with_kernel_follows_the_stata_formula():
    frame, y = make_panel(91, 7, 40, rho=0.5)
    count, periods = y.shape
    for trend, d, (mu, variance) in (("constant", 1, (1 / 6, 1 / 45)),
                                     ("trend", 2, (1 / 15, 11 / 6300))):
        x = np.ones((periods, 1)) if trend == "constant" else \
            np.column_stack([np.ones(periods), np.arange(1.0, periods + 1)])
        partial, plain, long_run = [], [], []
        m = 3
        for series in y:
            e = ols(series, x).resid
            partial.append(np.sum(np.cumsum(e) ** 2) / periods ** 2)
            plain.append(e @ e / (periods - d))                 # T' = T - 1 or T - 2
            long_run.append(e @ e / periods + 2 / periods * sum(
                (1 - j / (m + 1)) * (e[j:] @ e[:-j]) for j in range(1, m + 1)))
        for options, lm in (({}, np.mean(partial) / np.mean(plain)),
                            ({"robust": True}, np.mean(np.array(partial) / np.array(plain))),
                            ({"kernel_lags": m}, np.mean(partial) / np.mean(long_run))):
            z = math.sqrt(count) * (lm - mu) / math.sqrt(variance)
            attrs = oe.xtunitroot(frame, "y", "id", "year", test="hadri", trend=trend,
                                  **options).attrs
            assert attrs["lm"] == pytest.approx(lm, rel=1e-9), options
            assert attrs["statistic"] == pytest.approx(z, rel=1e-8), options
            assert attrs["p_value"] == pytest.approx(stats.norm.sf(z), rel=1e-7, abs=1e-300)


def test_kao_statistics_follow_from_the_reported_components_and_are_scale_free():
    rng = np.random.default_rng(101)
    count, periods = 12, 50
    x1 = rng.standard_normal((count, periods)).cumsum(axis=1)
    x2 = rng.standard_normal((count, periods)).cumsum(axis=1)
    u = rng.standard_normal((count, periods))
    for t in range(1, periods):
        u[:, t] += 0.5 * u[:, t - 1]
    y = rng.normal(size=(count, 1)) + 0.7 * x1 - 0.4 * x2 + u
    frame = pd.DataFrame({"id": np.repeat(np.arange(count), periods),
                          "year": np.tile(np.arange(periods), count),
                          "y": y.ravel(), "x1": x1.ravel(), "x2": x2.ravel()})
    result = oe.xtcointtest(frame, "y", ["x1", "x2"], "id", "year", lags=2, kernel_lags=3)
    attrs = result.attrs
    rho, t_rho, t_adf = attrs["rho"], attrs["t_rho"], attrs["t_adf"]
    s2, w2 = attrs["sigma2_v"], attrs["sigma2_0v"]
    root = math.sqrt(count)
    # Kao (1999) / Stata's Methods and formulas, from the reported components.
    expected = {
        "unadjusted_modified_df": (root * periods * (rho - 1) + 3 * root) / math.sqrt(51 / 5),
        "unadjusted_df": math.sqrt(5 / 4) * t_rho + math.sqrt(15 * count / 8),
        "modified_df": (root * periods * (rho - 1) + 3 * root * s2 / w2)
        / math.sqrt(3 + 36 * s2 ** 2 / (5 * w2 ** 2)),
        "df": (t_rho + math.sqrt(6 * count) * math.sqrt(s2) / (2 * math.sqrt(w2)))
        / math.sqrt(w2 / (2 * s2) + 3 * s2 / (10 * w2)),
        "adf": (t_adf + math.sqrt(6 * count) * math.sqrt(s2) / (2 * math.sqrt(w2)))
        / math.sqrt(w2 / (2 * s2) + 3 * s2 / (10 * w2)),
    }
    for name, value in expected.items():
        assert result.loc[name, "statistic"] == pytest.approx(value, rel=1e-10)
        assert result.loc[name, "p_value"] == pytest.approx(stats.norm.cdf(value), rel=1e-7,
                                                            abs=1e-300)
    # The within estimator and the pooled Dickey-Fuller coefficient, panel by panel.
    demeaned = [values - values.mean(axis=1, keepdims=True) for values in (y, x1, x2)]
    slope = np.linalg.lstsq(np.column_stack([demeaned[1].ravel(), demeaned[2].ravel()]),
                            demeaned[0].ravel(), rcond=None)[0]
    assert_allclose([attrs["cointegrating_vector"][name] for name in ("x1", "x2")], slope,
                    rtol=1e-9)
    e = demeaned[0] - slope[0] * demeaned[1] - slope[1] * demeaned[2]
    assert rho == pytest.approx(np.sum(e[:, 1:] * e[:, :-1]) / np.sum(e[:, :-1] ** 2), rel=1e-9)
    # Units of y and x, and panel-specific levels, do not matter.
    moved = frame.assign(y=frame["y"] * 1e4 + frame["id"] * 1e7, x1=frame["x1"] * 1e-3 - 5e5)
    other = oe.xtcointtest(moved, "y", ["x1", "x2"], "id", "year", lags=2, kernel_lags=3)
    assert_allclose(other["statistic"], result["statistic"], rtol=1e-6)


# ---- 4. Monte Carlo evidence on two conventions ------------------------------------------


def test_llc_adjustments_are_centred_for_the_long_run_variance_under_the_null():
    """E[sum(v e)] / T~ = mu* E[s_i] must hold for t* to be centred (LLC 2002, eq. 12).

    Simulated random walks show which long-run variance step the published mu* belongs
    to: Delta y as it is with panel means, Delta y minus its mean with panel trends. The
    published / Stata step demeans with panel means as well, which makes s_i too small
    and t* too negative in short panels (documented in docs/econometrics/unitroot.md).
    """
    rng = np.random.default_rng(2002)
    t_tilde, panels = 50, 60000
    y = rng.standard_normal((panels, t_tilde + 1)).cumsum(axis=1)
    dy, lag = np.diff(y, axis=1), y[:, :-1]
    lags = int(3.21 * (t_tilde + 1) ** (1 / 3))

    def long_run(d):
        out = (d * d).sum(axis=1) / t_tilde
        for j in range(1, lags + 1):
            out += 2 * (1 - j / (lags + 1)) * (d[:, j:] * d[:, :-j]).sum(axis=1) / t_tilde
        return np.sqrt(out)

    demeaned = dy - dy.mean(axis=1, keepdims=True)
    for trend, x in (("constant", np.ones((t_tilde, 1))),
                     ("trend", np.column_stack([np.ones(t_tilde),
                                                np.arange(1.0, t_tilde + 1)]))):
        q = np.linalg.qr(x)[0]
        e, v = dy - (dy @ q) @ q.T, lag - (lag @ q) @ q.T
        rho = (e * v).sum(axis=1) / (v * v).sum(axis=1)
        sigma = np.sqrt(((e - rho[:, None] * v) ** 2).sum(axis=1) / t_tilde)
        moment = ((e * v).sum(axis=1) / sigma ** 2).mean() / t_tilde
        mu = dict(zip(PLM_LLC_T, PLM_LLC[trend][0], strict=False))[t_tilde]
        needed = {"raw": moment / (long_run(dy) / sigma).mean(),
                  "demeaned": moment / (long_run(demeaned) / sigma).mean()}
        if trend == "constant":
            assert needed["raw"] == pytest.approx(mu, abs=0.012)
            assert needed["demeaned"] < mu - 0.06
        else:
            assert needed["demeaned"] == pytest.approx(mu, abs=0.015)
            assert needed["raw"] > mu + 0.05


def test_harris_tzavalis_is_centred_with_altt_and_not_with_the_stata_default():
    rng = np.random.default_rng(1999)
    count, periods, reps = 400, 8, 150
    index = pd.DataFrame({"id": np.repeat(np.arange(count), periods),
                          "year": np.tile(np.arange(periods), count)})
    default, alternative = [], []
    for _ in range(reps):
        frame = index.assign(y=rng.standard_normal((count, periods)).cumsum(axis=1).ravel())
        default.append(oe.xtunitroot(frame, "y", "id", "year", test="ht").attrs["statistic"])
        alternative.append(oe.xtunitroot(frame, "y", "id", "year", test="ht",
                                         altt=True).attrs["statistic"])
    assert abs(np.mean(alternative)) < 0.25 and 0.85 < np.std(alternative) < 1.15
    assert np.mean(default) < -1.5                    # T = periods over-rejects in short panels


def test_breitung_statistic_is_standard_normal_under_the_null():
    rng = np.random.default_rng(2000)
    count, periods, reps = 50, 30, 120
    index = pd.DataFrame({"id": np.repeat(np.arange(count), periods),
                          "year": np.tile(np.arange(periods), count)})
    draws = {"none": [], "constant": [], "trend": []}
    for _ in range(reps):
        e = rng.standard_normal((count, periods)) * rng.uniform(0.3, 3.0, size=(count, 1))
        for trend in draws:
            y = (e + (0.5 if trend == "trend" else 0.0)).cumsum(axis=1)
            if trend != "none":
                y = y + rng.normal(size=(count, 1)) * 10
            draws[trend].append(oe.xtunitroot(index.assign(y=y.ravel()), "y", "id", "year",
                                              test="breitung", trend=trend).attrs["statistic"])
    for trend, values in draws.items():
        assert abs(np.mean(values)) < 0.3, trend
        assert 0.8 < np.std(values) < 1.2, trend


# ---- 5. Second verification pass: sources read from the R packages and Stata manuals ----

# R package fUnitRoots, adfTable(): Hamilton (1994), Tables B.5 and B.6 = Fuller (1976),
# Tables 8.5.1 and 8.5.2. One row per sample size 25, 50, 100, 250, 500, infinity; the
# columns are the probabilities 0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99.
FUNITROOTS_N_RHO = {
    "n": ((-11.9, -9.3, -7.3, -5.3, 1.01, 1.40, 1.79, 2.28),
          (-12.9, -9.9, -7.7, -5.5, 0.97, 1.35, 1.70, 2.16),
          (-13.3, -10.2, -7.9, -5.6, 0.95, 1.31, 1.65, 2.09),
          (-13.6, -10.3, -8.0, -5.7, 0.93, 1.28, 1.62, 2.04),
          (-13.7, -10.4, -8.0, -5.7, 0.93, 1.28, 1.61, 2.04),
          (-13.8, -10.5, -8.1, -5.7, 0.93, 1.28, 1.60, 2.03)),
    "c": ((-17.2, -14.6, -12.5, -10.2, -0.76, 0.01, 0.65, 1.40),
          (-18.9, -15.7, -13.3, -10.7, -0.81, -0.07, 0.53, 1.22),
          (-19.8, -16.3, -13.7, -11.0, -0.83, -0.10, 0.47, 1.14),
          (-20.3, -16.6, -14.0, -11.2, -0.84, -0.12, 0.43, 1.09),
          (-20.5, -16.8, -14.0, -11.2, -0.84, -0.13, 0.42, 1.06),
          (-20.7, -16.9, -14.1, -11.3, -0.85, -0.13, 0.41, 1.04)),
    "ct": ((-22.5, -19.9, -17.9, -15.6, -3.66, -2.51, -1.53, -0.43),
           (-25.7, -22.4, -19.8, -16.8, -3.71, -2.60, -1.66, -0.65),
           (-27.4, -23.6, -20.7, -17.5, -3.74, -2.62, -1.73, -0.75),
           (-28.4, -24.4, -21.3, -18.0, -3.75, -2.64, -1.78, -0.82),
           (-28.9, -24.8, -21.5, -18.1, -3.76, -2.65, -1.78, -0.84),
           (-29.5, -25.1, -21.8, -18.3, -3.77, -2.66, -1.79, -0.87)),
}
FUNITROOTS_TAU = {
    "n": ((-2.66, -2.26, -1.95, -1.60, 0.92, 1.33, 1.70, 2.16),
          (-2.62, -2.25, -1.95, -1.61, 0.91, 1.31, 1.66, 2.08),
          (-2.60, -2.24, -1.95, -1.61, 0.90, 1.29, 1.64, 2.03),
          (-2.58, -2.23, -1.95, -1.62, 0.89, 1.29, 1.63, 2.01),
          (-2.58, -2.23, -1.95, -1.62, 0.89, 1.28, 1.62, 2.00),
          (-2.58, -2.23, -1.95, -1.62, 0.89, 1.28, 1.62, 2.00)),
    "c": ((-3.75, -3.33, -3.00, -2.63, -0.37, 0.00, 0.34, 0.72),
          (-3.58, -3.22, -2.93, -2.60, -0.40, -0.03, 0.29, 0.66),
          (-3.51, -3.17, -2.89, -2.58, -0.42, -0.05, 0.26, 0.63),
          (-3.46, -3.14, -2.88, -2.57, -0.42, -0.06, 0.24, 0.62),
          (-3.44, -3.13, -2.87, -2.57, -0.43, -0.07, 0.24, 0.61),
          (-3.43, -3.12, -2.86, -2.57, -0.44, -0.07, 0.23, 0.60)),
    "ct": ((-4.38, -3.95, -3.60, -3.24, -1.14, -0.80, -0.50, -0.15),
           (-4.15, -3.80, -3.50, -3.18, -1.19, -0.87, -0.58, -0.24),
           (-4.04, -3.73, -3.45, -3.15, -1.22, -0.90, -0.62, -0.28),
           (-3.99, -3.69, -3.43, -3.13, -1.23, -0.92, -0.64, -0.31),
           (-3.98, -3.68, -3.42, -3.13, -1.24, -0.93, -0.65, -0.32),
           (-3.96, -3.66, -3.41, -3.12, -1.25, -0.94, -0.66, -0.33)),
}
# R package tseries, pp.test(): the trend case again, stored column by column (one vector
# per probability 0.01, 0.025, 0.05, 0.10 over n = 25, 50, 100, 250, 500, infinity) and
# with the sign dropped.
TSERIES_RHO_CT = ((22.5, 25.7, 27.4, 28.4, 28.9, 29.5), (19.9, 22.4, 23.6, 24.4, 24.8, 25.1),
                  (17.9, 19.8, 20.7, 21.3, 21.5, 21.8), (15.6, 16.8, 17.5, 18.0, 18.1, 18.3))
TSERIES_TAU_CT = ((4.38, 4.15, 4.04, 3.99, 3.98, 3.96), (3.95, 3.80, 3.73, 3.69, 3.68, 3.66),
                  (3.60, 3.50, 3.45, 3.43, 3.42, 3.41), (3.24, 3.18, 3.15, 3.13, 3.13, 3.12))


def test_fuller_tables_equal_the_funitroots_and_tseries_transcriptions():
    assert tables.FULLER_RHO_N[:5] == (25.0, 50.0, 100.0, 250.0, 500.0)
    assert math.isinf(tables.FULLER_RHO_N[5])
    for ours, theirs in ((tables.FULLER_RHO, FUNITROOTS_N_RHO), (tables.FULLER_TAU,
                                                                  FUNITROOTS_TAU)):
        assert set(ours) == {"n", "c", "ct"}
        for case, rows in theirs.items():
            assert_allclose(np.array(ours[case]), np.array(rows)[:, [0, 2, 3]], rtol=0, atol=0)
    for ours, theirs in ((tables.FULLER_RHO["ct"], TSERIES_RHO_CT),
                         (tables.FULLER_TAU["ct"], TSERIES_TAU_CT)):
        assert_allclose(np.array(ours), -np.array(theirs)[[0, 2, 3]].T, rtol=0, atol=0)


def test_fuller_tau_table_agrees_with_mackinnon_and_the_stata_manuals(air):
    import statsmodels.tsa.adfvalues as adfvalues

    # MacKinnon's (2010) response surfaces at the tabulated sample sizes, and his (1994)
    # distribution function at the asymptotic row.
    for case, rows in tables.FULLER_TAU.items():
        for n, row in zip(tables.FULLER_RHO_N[:5], rows[:5], strict=True):
            assert_allclose(row, adfvalues.mackinnoncrit(N=1, regression=case, nobs=n),
                            atol=0.04)
            assert critical.fuller_tau_critical(case, int(n)) == dict(zip(LEVELS, row,
                                                                          strict=True))
        for value, level in zip(rows[5], (0.01, 0.05, 0.10), strict=True):
            assert adfvalues.mackinnonp(value, regression=case, N=1) == pytest.approx(
                level, abs=0.003)
        # Outside the table: the first row below n = 25, the asymptotic row above 500.
        assert critical.fuller_tau_critical(case, 12) == dict(zip(LEVELS, rows[0], strict=True))
        assert critical.fuller_tau_critical(case, 501) == dict(zip(LEVELS, rows[5],
                                                                   strict=True))
        midway = critical.fuller_tau_critical(case, 175)
        assert_allclose([midway[level] for level in LEVELS],
                        (np.array(rows[2]) + np.array(rows[3])) / 2, rtol=1e-14)
    # "Interpolated Dickey-Fuller" critical values as printed by Stata:
    # . dfuller air, lags(3) trend  -> -4.027  -3.445  -3.145   (140 observations)
    # . pperron air, lags(4) trend  -> -4.026  -3.444  -3.144   (143 observations)
    values = oe.dfuller(air, "air", lags=3, trend="trend").attrs["critical_values_fuller"]
    assert_allclose([values[level] for level in LEVELS], [-4.027, -3.445, -3.145], atol=5e-4)
    values = oe.pperron(air, "air", lags=4, trend="trend").attrs["critical_values_fuller"]
    assert_allclose([values[level] for level in LEVELS], [-4.026, -3.444, -3.144], atol=5e-4)
    # The drift variant has a Student t reference: no Fuller values.
    assert "critical_values_fuller" not in oe.dfuller(air, "air", trend="drift").attrs
    for trend, case in (("none", "n"), ("constant", "c")):
        got = oe.dfuller(air, "air", trend=trend).attrs["critical_values_fuller"]
        assert got == critical.fuller_tau_critical(case, 143)


def test_cusum_of_squares_coefficients_equal_statsmodels():
    from statsmodels.regression.recursive_ls import _cusum_squares_scalars as theirs

    # statsmodels stores Edgerton and Wells's coefficients by one-sided level alpha / 2.
    columns = {"10%": 1, "5%": 2, "1%": 4}                    # 0.05, 0.025, 0.005
    for level, column in columns.items():
        assert_allclose(tables.CUSUMSQ_COEFFICIENTS[level], theirs[:, column], rtol=0, atol=0)
    m = 0.5 * 57 - 1
    a1, a2, a3 = theirs[:, 2]
    assert critical.cusumsq_critical(57, "5%") == pytest.approx(
        a1 / math.sqrt(m) + a2 / m + a3 / m ** 1.5, rel=1e-14)


def test_kpss_critical_values_against_simulation():
    rng = np.random.default_rng(1992)
    total, reps = 400, 40000
    noise = rng.standard_normal((reps, total))
    for case, x in (("level", np.ones((total, 1))),
                    ("trend", np.column_stack([np.ones(total), np.arange(1.0, total + 1)]))):
        q = np.linalg.qr(x)[0]
        u = noise - (noise @ q) @ q.T
        eta = (np.cumsum(u, axis=1) ** 2).sum(axis=1) / total ** 2 / ((u ** 2).sum(axis=1)
                                                                      / total)
        assert_allclose(np.quantile(eta, [0.90, 0.95, 0.975, 0.99]), tables.KPSS_CRIT[case],
                        rtol=0.04)


def test_zivot_andrews_critical_values_against_simulation():
    # Minimum Dickey-Fuller t over break dates for driftless random walks (no lags). The
    # regressors other than y_{t-1} are common to all replications, so each candidate date
    # is one projection. Finite samples sit slightly to the left of the asymptotic values.
    rng = np.random.default_rng(1993)
    total, reps = 160, 2500
    y = rng.standard_normal((reps, total)).cumsum(axis=1)
    dy, lag = np.diff(y, axis=1), y[:, :-1]
    n = total - 1
    t = np.arange(2.0, total + 1)
    edge = int(0.15 * total)
    smallest = {model: np.full(reps, np.inf) for model in ("intercept", "trend", "both")}
    for tb in range(edge + 1, total - edge + 1):
        du, dt = (t > tb).astype(float), np.where(t > tb, t - tb, 0.0)
        for model, extra in (("intercept", [du]), ("trend", [dt]), ("both", [du, dt])):
            z = np.column_stack([np.ones(n), t, *extra])
            q = np.linalg.qr(z)[0]
            level = lag - (lag @ q) @ q.T
            change = dy - (dy @ q) @ q.T
            b = (level * change).sum(axis=1) / (level * level).sum(axis=1)
            ssr = ((change - b[:, None] * level) ** 2).sum(axis=1)
            se = np.sqrt(ssr / (n - z.shape[1] - 1) / (level * level).sum(axis=1))
            smallest[model] = np.minimum(smallest[model], b / se)
    for model, values in smallest.items():
        simulated = np.quantile(values, [0.01, 0.05, 0.10])
        assert_allclose(simulated, tables.ZIVOT_ANDREWS[model], atol=0.22)
        assert (simulated < np.array(tables.ZIVOT_ANDREWS[model]) + 0.08).all()
    # The implementation computes the same statistic on one of the simulated series.
    frame = pd.DataFrame({"y": y[0]})
    for model in smallest:
        got = oe.zandrews(frame, "y", break_=model).attrs["statistic"]
        assert got == pytest.approx(smallest[model][0], rel=1e-8)


def stata_breitung(y, trend, p):
    """lambda of [XT] xtunitroot, Methods and formulas (Breitung), with explicit loops.

    The time index t is 1-based as in the manual. With trends the manual prints
    v_is = u_is - u_i1 - (T - p - 1) mean(Du); Breitung (2000) has (s - 1) mean(Du), which is
    what makes the statistic invariant to the panel trends and is used here.
    """
    count, total = y.shape
    numerator = denominator = 0.0
    for series in y:
        level = lambda t: series[t - 1]                              # y_t  # noqa: E731
        change = lambda t: series[t - 1] - series[t - 2]             # Delta y_t  # noqa: E731
        times = range(p + 2, total + 1)
        d = np.array([change(t) for t in times])
        lagged = np.array([[change(t - j) for j in range(1, p + 1)] for t in times])
        lagged = lagged.reshape(len(d), p)
        if trend != "trend":
            base = level(p + 1) if trend == "constant" else 0.0
            x = np.array([level(t - 1) - base for t in times])
            if p:
                d = d - lagged @ np.linalg.lstsq(lagged, d, rcond=None)[0]
                x = x - lagged @ np.linalg.lstsq(lagged, x, rcond=None)[0]
            variance = np.sum(d ** 2) / (total - p - 2)
            numerator += np.sum(x * d) / variance
            denominator += np.sum(x ** 2) / variance
            continue
        design = np.column_stack([np.ones(len(d)), lagged])
        alpha = np.linalg.lstsq(design, d, rcond=None)[0][1:]
        du = np.array([change(t) - sum(alpha[j - 1] * change(t - j) for j in range(1, p + 1))
                       for t in times])
        u = np.array([level(t - 1) - sum(alpha[j - 1] * level(t - j - 1)
                                         for j in range(1, p + 1)) for t in times])
        n = total - p - 1
        variance = np.sum((du - du.mean()) * du) / (total - p - 2)
        for s in range(1, n):
            ahead = math.sqrt((n - s) / (n - s + 1)) * (du[s - 1] - du[s:].mean())
            behind = u[s - 1] - u[0] - (s - 1) * du.mean()
            numerator += behind * ahead / variance
            denominator += behind ** 2 / variance
    return numerator / math.sqrt(denominator)


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("lags", [0, 1, 3])
def test_breitung_follows_the_stata_formulas(trend, lags):
    frame, y = make_panel(5, 7, 28, drift=0.1, ma=0.3)
    attrs = oe.xtunitroot(frame, "y", "id", "year", test="breitung", trend=trend,
                          lags=lags).attrs
    expected = stata_breitung(y, trend, lags)
    assert attrs["statistic"] == pytest.approx(expected, rel=1e-9)
    assert attrs["p_value"] == pytest.approx(stats.norm.cdf(expected), rel=1e-8)
    assert attrs["lags"] == lags and attrs["distribution"] == "normal"


def phillips_perron_t(series, q, case):
    n = len(series) - 1
    columns = [series[:-1]]
    if case in ("c", "ct"):
        columns.append(np.ones(n))
    if case == "ct":
        columns.append(np.arange(1.0, n + 1))
    fit = ols(series[1:], np.column_stack(columns))
    u = fit.resid
    gamma0 = u @ u / n
    lambda2 = gamma0 + 2 * sum((1 - j / (q + 1)) * (u[j:] @ u[:-j]) / n for j in range(1, q + 1))
    rho, se = fit.params[0], fit.bse[0]
    return math.sqrt(gamma0 / lambda2) * (rho - 1) / se \
        - 0.5 * (lambda2 - gamma0) * n * se / math.sqrt(lambda2 * fit.mse_resid)


@pytest.mark.parametrize("trend,case", [("none", "n"), ("constant", "c"), ("trend", "ct")])
@pytest.mark.parametrize("method", ["dfuller", "pperron"])
def test_fisher_combinations_on_an_unbalanced_panel(trend, case, method):
    from statsmodels.tsa.adfvalues import mackinnonp

    frame, _ = make_panel(6, 9, 40, ma=0.4)
    unbalanced = frame[~((frame["id"] == 1) & (frame["year"] < 1982))]
    p_values = []
    for _, group in unbalanced.sort_values(["id", "year"]).groupby("id"):
        series = group["y"].to_numpy()
        if method == "dfuller":
            p_values.append(adfuller(series, maxlag=2, autolag=None, regression=case)[1])
        else:
            p_values.append(mackinnonp(phillips_perron_t(series, 2, case), regression=case,
                                       N=1))
    p = np.array(p_values)
    count = len(p)
    inverse_chi2 = -2 * np.log(p).sum()
    inverse_normal = stats.norm.ppf(p).sum() / math.sqrt(count)
    k = 3 * (5 * count + 4) / (math.pi ** 2 * count * (5 * count + 2))
    logit = math.sqrt(k) * np.log(p / (1 - p)).sum()
    modified = -(np.log(p) + 1).sum() / math.sqrt(count)
    expected = {"P": (inverse_chi2, 2 * count, stats.chi2.sf(inverse_chi2, 2 * count)),
                "Z": (inverse_normal, None, stats.norm.cdf(inverse_normal)),
                "L*": (logit, 5 * count + 4, stats.t.cdf(logit, 5 * count + 4)),
                "Pm": (modified, None, stats.norm.sf(modified))}
    result = oe.xtunitroot(unbalanced, "y", "id", "year", test="fisher", trend=trend, lags=2,
                           method=method)
    assert list(result.index) == list(expected)
    for name, (statistic, df, p_value) in expected.items():
        assert result.loc[name, "statistic"] == pytest.approx(statistic, rel=1e-9)
        assert result.loc[name, "p_value"] == pytest.approx(p_value, rel=1e-8)
        if df is not None:
            assert result.loc[name, "df"] == df
    assert result.attrs["statistic"] == pytest.approx(inverse_chi2, rel=1e-9)
    assert result.attrs["balanced"] is False and result.attrs["n_panels"] == count


@pytest.mark.parametrize("method", ["aic", "bic", "hqic"])
@pytest.mark.parametrize("trend,case", [("none", "n"), ("constant", "c"), ("trend", "ct")])
def test_panel_lag_orders_minimize_the_gaussian_information_criterion(method, trend, case):
    # Stata: regressions with 1..pmax lags on the sample that pmax lags leave, panel by
    # panel; (-2 ln L + k c) / M with c = 2, ln M or 2 ln ln M.
    frame, y = make_panel(7, 6, 45, ma=0.5)
    largest = 4
    chosen = []
    for series in y:
        total = len(series)
        dy = np.diff(series)
        rows = np.arange(largest, total - 1)
        best = None
        for p in range(1, largest + 1):
            columns = [series[rows]] + [dy[rows - j] for j in range(1, p + 1)]
            if case in ("c", "ct"):
                columns.append(np.ones(len(rows)))
            if case == "ct":
                columns.append(np.arange(1.0, len(rows) + 1))
            fit = ols(dy[rows], np.column_stack(columns))
            m = len(rows)
            penalty = {"aic": 2.0, "bic": math.log(m), "hqic": 2 * math.log(math.log(m))}[method]
            criterion = (-2 * fit.llf + len(columns) * penalty) / m
            if best is None or criterion < best[0]:
                best = (criterion, p)
        chosen.append(best[1])
    for test in ("fisher", "llc", "breitung"):
        attrs = oe.xtunitroot(frame, "y", "id", "year", test=test, trend=trend, lags=method,
                              maxlag=largest).attrs
        assert attrs["lags"] == pytest.approx(np.mean(chosen))
        assert (attrs["lags_min"], attrs["lags_max"]) == (min(chosen), max(chosen))
        assert attrs["lag_method"] == method
    # LLC enters Table 2 at T - mean(p_i) - 1.
    attrs = oe.xtunitroot(frame, "y", "id", "year", test="llc", trend=trend, lags=method,
                          maxlag=largest).attrs
    assert attrs["t_tilde"] == pytest.approx(45 - np.mean(chosen) - 1)


@pytest.mark.parametrize("lags,kernel_lags", [(1, 2), (2, 0), (0, 4)])
def test_kao_components_from_dummy_variable_regressions(lags, kernel_lags):
    rng = np.random.default_rng(101)
    count, periods = 9, 40
    x1 = rng.standard_normal((count, periods)).cumsum(axis=1)
    x2 = rng.standard_normal((count, periods)).cumsum(axis=1)
    u = rng.standard_normal((count, periods))
    for t in range(1, periods):
        u[:, t] += 0.6 * u[:, t - 1]
    y = rng.normal(size=(count, 1)) * 3 + 0.7 * x1 - 0.4 * x2 + u
    frame = pd.DataFrame({"id": np.repeat(np.arange(count), periods),
                          "year": np.tile(np.arange(periods), count),
                          "y": y.ravel(), "x1": x1.ravel(), "x2": x2.ravel()})
    dummies = np.zeros((count * periods, count))
    dummies[np.arange(count * periods), np.repeat(np.arange(count), periods)] = 1.0
    lsdv = ols(y.ravel(), np.column_stack([x1.ravel(), x2.ravel(), dummies]))
    e = lsdv.resid.reshape(count, periods)
    behind, ahead = e[:, :-1].ravel(), e[:, 1:].ravel()
    rho = behind @ ahead / (behind @ behind)
    s2 = np.sum((ahead - rho * behind) ** 2) / len(ahead)
    t_rho = (rho - 1) * math.sqrt(behind @ behind) / math.sqrt(s2)
    targets, regressors = [], []
    for i in range(count):
        for t in range(lags + 1, periods):
            targets.append(e[i, t] - e[i, t - 1])
            regressors.append([e[i, t - 1]] + [e[i, t - j] - e[i, t - j - 1]
                                               for j in range(1, lags + 1)])
    t_adf = ols(np.array(targets), np.array(regressors)).tvalues[0]
    w = np.stack([np.diff(y, axis=1), np.diff(x1, axis=1), np.diff(x2, axis=1)], axis=2)
    size = count * (periods - 1)
    sigma = sum(w[i].T @ w[i] for i in range(count)) / size
    omega = sigma.copy()
    for j in range(1, kernel_lags + 1):
        gamma = sum(w[i][j:].T @ w[i][:-j] for i in range(count)) / size
        omega += (1 - j / (kernel_lags + 1)) * (gamma + gamma.T)

    def conditional(a):
        return a[0, 0] - a[0, 1:] @ np.linalg.solve(a[1:, 1:], a[1:, 0])

    attrs = oe.xtcointtest(frame, "y", ["x1", "x2"], "id", "year", lags=lags,
                           kernel_lags=kernel_lags).attrs
    assert attrs["rho"] == pytest.approx(rho, rel=1e-10)
    assert attrs["t_rho"] == pytest.approx(t_rho, rel=1e-10)
    assert attrs["t_adf"] == pytest.approx(t_adf, rel=1e-9)
    assert attrs["sigma2_v"] == pytest.approx(conditional(sigma), rel=1e-10)
    assert attrs["sigma2_0v"] == pytest.approx(conditional(omega), rel=1e-10)
    assert_allclose([attrs["cointegrating_vector"][name] for name in ("x1", "x2")],
                    lsdv.params[:2], rtol=1e-9)


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
def test_harris_tzavalis_rho_is_the_lsdv_coefficient(trend):
    frame, y = make_panel(8, 12, 15, drift=0.2)
    count, periods = y.shape
    columns = [y[:, :-1].ravel()]
    if trend != "none":
        dummies = np.zeros((count * (periods - 1), count))
        dummies[np.arange(count * (periods - 1)), np.repeat(np.arange(count), periods - 1)] = 1
        columns.append(dummies)
    if trend == "trend":
        columns.append(dummies * np.tile(np.arange(1.0, periods), count)[:, None])
    rho = ols(y[:, 1:].ravel(), np.column_stack(columns)).params[0]
    attrs = oe.xtunitroot(frame, "y", "id", "year", test="ht", trend=trend).attrs
    assert attrs["rho"] == pytest.approx(rho, rel=1e-10)
    # demean=True is the same test on the cross-sectionally demeaned series.
    demeaned = frame.assign(y=(y - y.mean(axis=0, keepdims=True)).ravel())
    for test in ("llc", "ips", "fisher", "hadri", "breitung", "ht"):
        if test in ("ips", "hadri") and trend == "none":
            continue
        assert oe.xtunitroot(frame, "y", "id", "year", test=test, trend=trend,
                             demean=True).attrs["statistic"] == pytest.approx(
            oe.xtunitroot(demeaned, "y", "id", "year", test=test,
                          trend=trend).attrs["statistic"], rel=1e-9)


def adf_fit(series, lags, case, start):
    total = len(series)
    dy = np.diff(series)
    rows = np.arange(start, total - 1)
    columns = [series[rows]] + [dy[rows - j] for j in range(1, lags + 1)]
    if case in ("c", "ct", "ctt"):
        columns.append(np.ones(len(rows)))
    if case in ("ct", "ctt"):
        columns.append(np.arange(start + 1.0, total))
    if case == "ctt":
        columns.append(np.arange(start + 1.0, total) ** 2)
    return ols(dy[rows], np.column_stack(columns))


@pytest.mark.parametrize("case", ["n", "c", "ct", "ctt"])
def test_nested_factorization_equals_separate_regressions(case, monkeypatch):
    import torch

    from openecon.econometrics.unitroot import series as module

    rng = np.random.default_rng(17)
    total, largest = 400, 9
    e = rng.standard_normal(total)
    y = np.cumsum(e + 0.6 * np.r_[0.0, e[:-1]]) + 0.02 * np.arange(total)
    for block in (1 << 22, 64):                     # one block, then many small blocks
        monkeypatch.setattr(module, "_BLOCK_ELEMENTS", block)
        fits = module.nested_adf(torch.as_tensor(y), largest, case, "test")
        assert [fit.lags for fit in fits] == list(range(largest + 1))
        for fit in fits:
            theirs = adf_fit(y, fit.lags, case, largest)
            assert fit.nobs == theirs.nobs and fit.parameters == len(theirs.params)
            assert fit.ssr == pytest.approx(theirs.ssr, rel=1e-9)
            assert fit.coefficient == pytest.approx(theirs.params[0], rel=1e-7)
            assert fit.statistic == pytest.approx(theirs.tvalues[0], rel=1e-7)
            assert fit.level_ss == pytest.approx(np.sum(y[largest:total - 1] ** 2), rel=1e-12)
            if fit.lags:
                assert fit.last_t == pytest.approx(theirs.tvalues[fit.lags], rel=1e-7)
            else:
                assert fit.last_t is None


@pytest.mark.parametrize("method", ["aic", "bic", "t"])
def test_lag_choice_of_zandrews_and_egranger(method):
    rng = np.random.default_rng(42)
    total, largest = 120, 5
    e = rng.standard_normal(total)
    y = np.cumsum(e + 0.5 * np.r_[0.0, e[:-1]]) + 10
    x = np.cumsum(rng.standard_normal(total))
    frame = pd.DataFrame({"y": y, "x": x})

    def choose(series, case):
        fits = {p: adf_fit(series, p, case, largest) for p in range(largest + 1)}
        if method == "t":
            return next((p for p in range(largest, 0, -1)
                         if abs(fits[p].tvalues[p]) >= stats.norm.ppf(0.95)), 0)
        penalty = (lambda n: 2.0) if method == "aic" else math.log
        values = {p: fit.nobs * math.log(fit.ssr / fit.nobs)
                  + len(fit.params) * penalty(fit.nobs) for p, fit in fits.items()}
        return min(values, key=values.get)

    attrs = oe.zandrews(frame, "y", lags=method, maxlag=largest).attrs
    assert attrs["lags"] == choose(y, "ct") and attrs["lag_method"] == method
    residual = ols(y, np.column_stack([x, np.ones(total)])).resid
    attrs = oe.egranger(frame, "y", ["x"], lags=method, maxlag=largest).attrs
    assert attrs["lags"] == choose(residual, "n") and attrs["lag_method"] == method
    assert attrs["statistic"] == pytest.approx(
        adf_fit(residual, attrs["lags"], "n", attrs["lags"]).tvalues[0], rel=1e-8)


@pytest.mark.parametrize("trend", [True, False])
def test_dfgls_from_explicit_gls_detrending(trend):
    rng = np.random.default_rng(43)
    total, largest = 120, 6
    e = rng.standard_normal(total)
    y = np.cumsum(e + 0.5 * np.r_[0.0, e[:-1]]) + 10
    a = 1 + (-13.5 if trend else -7.0) / total
    z = np.column_stack([np.ones(total), np.arange(1.0, total + 1)]) if trend \
        else np.ones((total, 1))
    quasi_y = np.r_[y[0], y[1:] - a * y[:-1]]
    quasi_z = np.vstack([z[0], z[1:] - a * z[:-1]])
    detrended = y - z @ np.linalg.lstsq(quasi_z, quasi_y, rcond=None)[0]
    result = oe.dfgls(pd.DataFrame({"y": y}), "y", maxlag=largest, trend=trend)
    by_lag = result.set_index("lags")
    n = total - largest - 1
    criteria, last = {}, {}
    for k in range(1, largest + 1):
        fit = adf_fit(detrended, k, "n", largest)
        s2 = fit.ssr / n
        tau = fit.params[0] ** 2 * np.sum(detrended[largest:total - 1] ** 2) / s2
        criteria[k] = (math.log(s2) + (k + 1) * math.log(n) / n,
                       math.log(s2) + 2 * (tau + k) / n)
        last[k] = fit.tvalues[k]
        assert by_lag.loc[k, "statistic"] == pytest.approx(fit.tvalues[0], rel=1e-8)
        assert by_lag.loc[k, "rmse"] == pytest.approx(math.sqrt(s2), rel=1e-10)
        assert by_lag.loc[k, "sc"] == pytest.approx(criteria[k][0], rel=1e-10)
        assert by_lag.loc[k, "maic"] == pytest.approx(criteria[k][1], rel=1e-9)
    attrs = result.attrs
    assert attrs["lags_sc"] == min(criteria, key=lambda k: criteria[k][0])
    assert attrs["lags_maic"] == min(criteria, key=lambda k: criteria[k][1])
    assert attrs["lags_seq_t"] == next(
        (k for k in range(largest, 0, -1) if abs(last[k]) > stats.norm.ppf(0.95)), 0)
    assert attrs["nobs"] == n and attrs["statistic"] == by_lag.loc[attrs["lags_maic"],
                                                                   "statistic"]
    if trend:
        # Elliott-Rothenberg-Stock at the 120 observations of the series, between the rows
        # for T = 100 and T = 200.
        assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                        URCA_ERS[1] + 0.2 * (URCA_ERS[2] - URCA_ERS[1]), rtol=1e-12)
        assert attrs["p_value"] is None
    else:
        import statsmodels.tsa.adfvalues as adfvalues

        assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                        adfvalues.mackinnoncrit(N=1, regression="n", nobs=n), rtol=1e-12)
        assert attrs["p_value"] == pytest.approx(
            adfvalues.mackinnonp(attrs["statistic"], regression="n", N=1), rel=1e-9)


@pytest.mark.parametrize("trend", [False, True])
def test_kpss_automatic_bandwidth_follows_hobijn_franses_ooms(trend):
    rng = np.random.default_rng(44)
    total = 130
    e = rng.standard_normal(total)
    y = np.cumsum(0.3 * e) + e + 5
    x = np.column_stack([np.ones(total), np.arange(1.0, total + 1)]) if trend \
        else np.ones((total, 1))
    u = ols(y, x).resid
    n = int(4 * (total / 100) ** (2 / 9))
    gamma = [u[j:] @ u[:total - j] / total for j in range(n + 1)]
    s0 = gamma[0] + 2 * sum(gamma[1:])
    s1 = 2 * sum(j * gamma[j] for j in range(1, n + 1))
    bandwidth = min(total - 1, int(1.1447 * ((s1 / s0) ** 2) ** (1 / 3) * total ** (1 / 3)))
    result = oe.kpss(pd.DataFrame({"y": y}), "y", trend=trend, auto=True)
    assert result.attrs["lags"] == bandwidth and result.attrs["bandwidth"] == "auto"
    assert len(result) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = sm_kpss(y, regression="ct" if trend else "c", nlags=bandwidth)[0]
    assert result.attrs["statistic"] == pytest.approx(expected, rel=1e-9)


def test_dfuller_drift_is_student_t_with_residual_degrees_of_freedom():
    rng = np.random.default_rng(45)
    y = np.cumsum(rng.standard_normal(90) + 0.3)
    fit = adf_fit(y, 2, "c", 2)
    attrs = oe.dfuller(pd.DataFrame({"y": y}), "y", lags=2, trend="drift").attrs
    assert attrs["statistic"] == pytest.approx(fit.tvalues[0], rel=1e-9)
    assert attrs["df"] == fit.df_resid == 90 - 3 - 4
    assert attrs["p_value"] == pytest.approx(stats.t.cdf(fit.tvalues[0], fit.df_resid), rel=1e-9)
    assert_allclose([attrs["critical_values"][level] for level in LEVELS],
                    stats.t.ppf([0.01, 0.05, 0.10], fit.df_resid), rtol=1e-9)
    assert attrs["distribution"] == "t"


def test_sup_wald_p_value_against_simulated_quantiles_and_single_candidate():
    # Simulated sup of the tied-down Bessel process over [0.15, 0.85]: the tail formula is
    # usable up to a p-value of about 0.5 and understates larger ones.
    rng = np.random.default_rng(78)
    grid, reps = 500, 6000
    r = np.arange(1, grid + 1) / grid
    keep = (r >= 0.15) & (r <= 0.85)
    for q in (1, 4):
        steps = rng.standard_normal((reps, grid, q)) / math.sqrt(grid)
        bridge = steps.cumsum(axis=1)
        bridge -= r[None, :, None] * bridge[:, -1:, :]
        sup = ((bridge[:, keep, :] ** 2).sum(axis=2) / (r * (1 - r))[keep][None, :]).max(axis=1)
        for probability, tolerance in ((0.5, 0.10), (0.3, 0.06), (0.2, 0.05), (0.1, 0.03),
                                       (0.05, 0.02), (0.01, 0.006)):
            point = np.quantile(sup, 1 - probability)
            assert critical.supwald_p(point, q, 0.15, 0.85) == pytest.approx(probability,
                                                                             abs=tolerance)
        assert critical.supwald_p(np.quantile(sup, 0.1), q, 0.15, 0.85) < 0.9
    # One candidate date: an ordinary Wald test, chi-squared(q).
    assert critical.supwald_p(7.3, 3, 0.5, 0.5) == pytest.approx(stats.chi2.sf(7.3, 3), rel=1e-10)
    rng = np.random.default_rng(79)
    data = pd.DataFrame({"x": rng.standard_normal(60), "y": rng.standard_normal(60)})
    data["y"] += 2.0 * (np.arange(60) >= 30)
    attrs = oe.sbsingle(data, "y", ["x"], trim=0.49).attrs
    assert attrs["candidates"] == 1 and attrs["critical_values"] is None
    chow_test = oe.chow(data, "y", ["x"], 30)
    assert attrs["statistic"] == pytest.approx(chow_test.loc["wald", "statistic"], rel=1e-9)
    assert attrs["p_value"] == pytest.approx(chow_test.loc["wald", "p_value"], rel=1e-8)
    assert any("one candidate" in note for note in attrs["notes"])
    # The usual case says where the critical values come from and what was checked.
    attrs = oe.sbsingle(data, "x", [], trim=0.15).attrs
    assert any("not independently verified" in note for note in attrs["notes"])
    if attrs["p_value"] > 0.5:
        assert any("understates" in note for note in attrs["notes"])


# ---- 6. Monte Carlo under the null for the remaining tabulated references ---------------


def dickey_fuller_t(level, change, deterministic):
    """t ratios of the lagged level for many replications sharing the deterministic terms."""
    n, k = change.shape[1], 0
    if deterministic is not None:
        q = np.linalg.qr(deterministic)[0]
        level = level - (level @ q) @ q.T
        change = change - (change @ q) @ q.T
        k = deterministic.shape[1]
    b = (level * change).sum(axis=1) / (level * level).sum(axis=1)
    ssr = ((change - b[:, None] * level) ** 2).sum(axis=1)
    return b / np.sqrt(ssr / (n - k - 1) / (level * level).sum(axis=1))


def test_mackinnon_surfaces_against_simulated_dickey_fuller_and_engle_granger_tests():
    rng = np.random.default_rng(1994)
    total, reps = 200, 20000
    y = rng.standard_normal((reps, total)).cumsum(axis=1)
    n = total - 1
    cases = {"n": None, "c": np.ones((n, 1)),
             "ct": np.column_stack([np.ones(n), np.arange(1.0, n + 1)])}
    for case, deterministic in cases.items():
        t = dickey_fuller_t(y[:, :-1], np.diff(y, axis=1), deterministic)
        values = critical.mackinnon_critical(case, 1, n)
        assert_allclose(np.quantile(t, [0.01, 0.05, 0.10]),
                        [values[level] for level in LEVELS], atol=0.07)
        p = np.array([critical.mackinnon_p(value, case, 1) for value in t[:4000]])
        for level in (0.05, 0.25, 0.5, 0.9):
            assert np.mean(p < level) == pytest.approx(level, abs=0.03)
    # Engle-Granger with one regressor and a constant: N = 2 series.
    x = rng.standard_normal((reps, total)).cumsum(axis=1)
    xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    e = yc - ((xc * yc).sum(axis=1) / (xc * xc).sum(axis=1))[:, None] * xc
    t = dickey_fuller_t(e[:, :-1], np.diff(e, axis=1), None)
    values = critical.mackinnon_critical("c", 2, n)
    assert_allclose(np.quantile(t, [0.01, 0.05, 0.10]), [values[level] for level in LEVELS],
                    atol=0.07)
    p = np.array([critical.mackinnon_p(value, "c", 2) for value in t[:4000]])
    assert np.mean(p < 0.05) == pytest.approx(0.05, abs=0.02)
    # One replication through the public functions.
    frame = pd.DataFrame({"y": y[0], "x": x[0]})
    assert oe.egranger(frame, "y", ["x"]).attrs["statistic"] == pytest.approx(t[0], rel=1e-8)


def test_hadri_and_cusum_references_against_simulation():
    rng = np.random.default_rng(2000)
    count, periods, reps = 100, 100, 120
    index = pd.DataFrame({"id": np.repeat(np.arange(count), periods),
                          "t": np.tile(np.arange(periods), count)})
    draws = {"constant": [], "trend": []}
    for _ in range(reps):
        noise = rng.standard_normal((count, periods)) + rng.normal(size=(count, 1)) * 3
        for trend, values in draws.items():
            y = noise + (0.1 * np.arange(periods) if trend == "trend" else 0.0)
            values.append(oe.xtunitroot(index.assign(y=y.ravel()), "y", "id", "t", test="hadri",
                                        trend=trend).attrs["statistic"])
    for trend, values in draws.items():           # Hadri's z is slightly oversized for finite T
        assert abs(np.mean(values)) < 0.45, trend
        assert 0.8 < np.std(values) < 1.35, trend
    # Brown-Durbin-Evans's lines are crossed a little less often than the nominal 5%, the
    # CUSUM-of-squares bounds about as often.
    reps, total = 500, 120
    crossed = squares = 0
    for _ in range(reps):
        data = pd.DataFrame({"y": rng.standard_normal(total), "x": rng.standard_normal(total)})
        attrs = oe.cusum(data, "y", ["x"]).attrs
        crossed += attrs["cusum_crosses"]
        squares += attrs["cusumsq_crosses"]
    assert 0.01 <= crossed / reps <= 0.07
    assert 0.02 <= squares / reps <= 0.09
