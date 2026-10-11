"""Fourier ADF / KPSS / LM: NumPy and statsmodels oracles, critical-value provenance, contracts.

Statistics are compared with explicit NumPy least squares written from the
papers' equations (and statsmodels OLS for one case). Critical values are
checked against (a) a second transcription of every published entry (the GAUSS
library TSPDLIB, github.com/aptech/tspdlib) and (b) the seeded simulation in
tests/fixtures/unitroot_fourier/simulated_quantiles.json
(scripts/generate_unitroot_fourier_oracles.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot import fourier_tables as tables

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "unitroot_fourier"
SIMULATED = json.loads((FIXTURES / "simulated_quantiles.json").read_text())
LEVELS = ("1%", "5%", "10%")


def series(seed=11, n=140, kind="walk"):
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    e = rng.normal(size=n)
    if kind == "walk":
        y = np.cumsum(e + 0.3 * np.r_[0.0, e[:-1]]) + 3 * np.sin(2 * np.pi * 2 * t / n)
    else:
        y = np.zeros(n)
        y[0] = e[0]
        for i in range(1, n):
            y[i] = 0.6 * y[i - 1] + e[i]
        y = y + 2 * np.cos(2 * np.pi * t / n) + 0.01 * t
    return pd.DataFrame({"y": y + 50.0, "t": t})


# ---- NumPy oracles --------------------------------------------------------------------


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ssr = float(resid @ resid)
    df = X.shape[0] - X.shape[1]
    cov = ssr / df * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), ssr, df


def fourier_cols(T, k):
    t = np.arange(1, T + 1)
    return np.sin(2 * np.pi * k * t / T), np.cos(2 * np.pi * k * t / T)


def adf_oracle(y, k, trend, p):
    """Enders-Lee (2012) eq. (3) with p lagged differences on t = p+2..T."""
    T = len(y)
    t = np.arange(1, T + 1, dtype=float)
    s, c = fourier_cols(T, k)
    dy = np.diff(y)                                   # dy[i] = y[i+1] - y[i], obs i+2
    rows = np.arange(p + 1, T)                        # 0-based observation index
    lagcols = [dy[rows - 1 - j] for j in range(1, p + 1)]
    det = [np.ones(len(rows))] + ([t[rows]] if trend == "trend" else [])
    X_u = np.column_stack([y[rows - 1], *det, s[rows], c[rows], *lagcols])
    X_r = np.column_stack([y[rows - 1], *det, *lagcols])
    target = dy[rows - 1]
    b, se, ssr_u, df_u = ols(X_u, target)
    _, _, ssr_r, _ = ols(X_r, target)
    return b[0] / se[0], ((ssr_r - ssr_u) / 2) / (ssr_u / df_u), ssr_u


def kpss_oracle(y, k, trend, lags):
    T = len(y)
    t = np.arange(1, T + 1, dtype=float)
    s, c = fourier_cols(T, k)
    base = [np.ones(T)] + ([t] if trend == "trend" else [])
    X = np.column_stack(base + [s, c])
    _, _, ssr, _ = ols(X, y)
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    S = np.cumsum(e)
    gamma = [float(e[j:] @ e[:T - j]) / T for j in range(lags + 1)]
    lrv = gamma[0] + 2 * sum((1 - j / (lags + 1)) * gamma[j] for j in range(1, lags + 1))
    _, _, ssr0, _ = ols(np.column_stack(base), y)
    return float(S @ S) / T ** 2 / lrv, ((ssr0 - ssr) / 2) / (ssr / (T - X.shape[1])), ssr


def lm_oracle(y, k, p):
    """Enders-Lee (2012, OBES) eqs. (8)-(10); F as in the authors' RATS program."""
    T = len(y)
    t = np.arange(1, T + 1, dtype=float)
    s, c = fourier_cols(T, k)
    dy = np.diff(y)
    ds, dc = np.diff(s), np.diff(c)
    delta = np.linalg.lstsq(np.column_stack([np.ones(T - 1), ds, dc]), dy, rcond=None)[0]
    psi = y[0] - delta[0] - delta[1] * s[0] - delta[2] * c[0]
    S = y - psi - delta[0] * t - delta[1] * s - delta[2] * c
    dS = np.diff(S)
    rows = np.arange(p + 1, T)
    lagcols = [dS[rows - 1 - j] for j in range(1, p + 1)]
    X = np.column_stack([S[rows - 1], np.ones(len(rows)), ds[rows - 1], dc[rows - 1], *lagcols])
    b, se, ssr, _ = ols(X, dy[rows - 1])
    d0 = dy.mean()
    SL = y - y[0] - d0 * (t - 1)
    dSL = np.diff(SL)
    lagL = [dSL[rows - 1 - j] for j in range(1, p + 1)]
    Xu = np.column_stack([SL[rows - 1], np.ones(len(rows)), ds[rows - 1], dc[rows - 1], *lagL])
    Xr = np.column_stack([SL[rows - 1], np.ones(len(rows)), *lagL])
    _, _, ssr_u, df_u = ols(Xu, dy[rows - 1])
    _, _, ssr_r, _ = ols(Xr, dy[rows - 1])
    return b[0] / se[0], ((ssr_r - ssr_u) / 2) / (ssr_u / df_u), ssr


def select_lags(fits_ssr, nobs, params, method, last_t=None):
    """Explicit AIC/BIC over orders (ties to the smaller order) on a common sample."""
    if method == "t":
        for order in range(len(fits_ssr) - 1, 0, -1):
            if abs(last_t[order]) >= 1.6448536269514722:
                return order
        return 0
    penalty = 2.0 if method == "aic" else math.log(nobs)
    values = [nobs * math.log(ssr / nobs) + (params + order) * penalty
              for order, ssr in enumerate(fits_ssr)]
    return int(np.argmin(values))


# ---- statistics ------------------------------------------------------------------------


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("p", [0, 2])
def test_fourier_adf_matches_numpy_regression(trend, k, p):
    frame = series()
    result = oe.fourier_adf(frame, "y", time="t", trend=trend, frequency=k, lags=p)
    tau, f_stat, ssr = adf_oracle(frame.y.to_numpy(), k, trend, p)
    assert result.attrs["statistic"] == pytest.approx(tau, rel=1e-10, abs=1e-10)
    assert result.attrs["f_statistic"] == pytest.approx(f_stat, rel=1e-9, abs=1e-10)
    assert result["frequencies"]["selection_ssr"].iloc[0] == pytest.approx(ssr, rel=1e-9)
    assert result.attrs["nobs"] == len(frame) - 1 - p
    assert result.attrs["df_resid"] == len(frame) - 1 - p - (p + (4 if trend == "trend" else 3) + 1)
    assert result.attrs["f_critical_values"] is None
    assert "fixed frequency" in result.attrs["f_inference_reason"]


def test_fourier_adf_matches_statsmodels_ols():
    sm = pytest.importorskip("statsmodels.api")
    frame = series(seed=5)
    y = frame.y.to_numpy()
    T = len(y)
    s, c = fourier_cols(T, 2)
    rows = np.arange(2, T)
    dy = np.diff(y)
    X = np.column_stack([y[rows - 1], np.ones(len(rows)), np.arange(1, T + 1)[rows], s[rows],
                         c[rows], dy[rows - 2]])
    fit = sm.OLS(dy[rows - 1], X).fit()
    result = oe.fourier_adf(frame, "y", trend="trend", frequency=2, lags=1)
    assert result.attrs["statistic"] == pytest.approx(fit.tvalues[0], rel=1e-9)
    restriction = np.zeros((2, X.shape[1]))
    restriction[0, 3] = restriction[1, 4] = 1.0          # the sin and cos columns
    assert result.attrs["f_statistic"] == pytest.approx(
        float(np.squeeze(fit.f_test(restriction).fvalue)), rel=1e-8)


@pytest.mark.parametrize("method", ["aic", "bic", "t"])
@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_fourier_adf_frequency_and_lag_selection(method, trend):
    frame = series(seed=23, n=160)
    y = frame.y.to_numpy()
    T, maxlag = len(y), 5
    result = oe.fourier_adf(frame, "y", trend=trend, lags=method, maxlag=maxlag)
    rows = np.arange(maxlag + 1, T)
    dy = np.diff(y)
    t = np.arange(1, T + 1, dtype=float)
    candidates = []
    for k in range(1, 6):
        s, c = fourier_cols(T, k)
        det = [np.ones(len(rows))] + ([t[rows]] if trend == "trend" else [])
        ssrs, last_t = [], [0.0]
        for order in range(maxlag + 1):
            X = np.column_stack([*det, s[rows], c[rows], y[rows - 1],
                                 *[dy[rows - 1 - j] for j in range(1, order + 1)]])
            b, se, ssr, _ = ols(X, dy[rows - 1])
            ssrs.append(ssr)
            if order:
                last_t.append(b[-1] / se[-1])
        chosen = select_lags(ssrs, len(rows), len(det) + 3, method, last_t)
        candidates.append((ssrs[chosen], k, chosen))
    best_ssr, best_k, best_p = min(candidates)
    assert result.attrs["frequency"] == best_k
    assert result.attrs["lags"] == best_p
    np.testing.assert_allclose(result["frequencies"]["selection_ssr"], [c[0] for c in candidates],
                               rtol=1e-9)
    tau, f_stat, _ = adf_oracle(y, best_k, trend, best_p)
    assert result.attrs["statistic"] == pytest.approx(tau, rel=1e-9)
    assert result.attrs["f_statistic"] == pytest.approx(f_stat, rel=1e-8)
    assert result.attrs["f_critical_values"] is not None


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("lags", [0, 3, 8])
def test_fourier_kpss_matches_numpy(trend, k, lags):
    frame = series(kind="stationary")
    result = oe.fourier_kpss(frame, "y", time="t", trend=trend, frequency=k, lags=lags)
    stat, f_stat, ssr = kpss_oracle(frame.y.to_numpy(), k, trend, lags)
    assert result.attrs["statistic"] == pytest.approx(stat, rel=1e-10)
    assert result.attrs["f_statistic"] == pytest.approx(f_stat, rel=1e-9)
    assert result["frequencies"]["ssr"].iloc[0] == pytest.approx(ssr, rel=1e-10)
    assert result.attrs["bandwidth"] == "user"


def test_fourier_kpss_auto_frequency_and_bandwidths():
    frame = series(kind="stationary", n=200)
    y = frame.y.to_numpy()
    result = oe.fourier_kpss(frame, "y")
    ssrs = [kpss_oracle(y, k, "constant", 1)[2] for k in range(1, 6)]
    assert result.attrs["frequency"] == int(np.argmin(ssrs)) + 1
    default = min(len(y) - 1, int(12 * (len(y) / 100) ** 0.25))
    assert result.attrs["lags"] == default and result.attrs["bandwidth"] == "schwert"
    assert result.attrs["statistic"] == pytest.approx(
        kpss_oracle(y, result.attrs["frequency"], "constant", default)[0], rel=1e-10)
    automatic = oe.fourier_kpss(frame, "y", auto=True)
    assert automatic.attrs["bandwidth"] == "auto"
    assert automatic.attrs["statistic"] == pytest.approx(
        kpss_oracle(y, automatic.attrs["frequency"], "constant", automatic.attrs["lags"])[0],
        rel=1e-10)
    assert result.attrs["f_critical_values"] == pytest.approx(
        dict(zip(LEVELS, np.array(tables.KPSS_SUPF["constant"][0])
                 + (200 - 100) / 400 * (np.array(tables.KPSS_SUPF["constant"][1])
                                        - np.array(tables.KPSS_SUPF["constant"][0])))))


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("p", [0, 1, 4])
def test_fourier_lm_matches_paper_equations(k, p):
    frame = series(seed=31, n=130)
    result = oe.fourier_lm(frame, "y", time="t", frequency=k, lags=p)
    tau, f_stat, ssr = lm_oracle(frame.y.to_numpy(), k, p)
    assert result.attrs["statistic"] == pytest.approx(tau, rel=1e-9)
    assert result.attrs["f_statistic"] == pytest.approx(f_stat, rel=1e-8)
    assert result["frequencies"]["selection_ssr"].iloc[0] == pytest.approx(ssr, rel=1e-9)
    assert result.attrs["trend"] == "trend"
    # Panel b applies to a given frequency.
    expected, _ = row_at(tables.LM_T, tables.LM_F[k - 1], len(frame))
    assert result.attrs["f_critical_values"] == pytest.approx(expected)


def test_fourier_lm_auto_frequency_uses_regression_ten_ssr():
    frame = series(seed=8, n=180)
    y = frame.y.to_numpy()
    result = oe.fourier_lm(frame, "y", lags=1)
    ssrs = [lm_oracle(y, k, 1)[2] for k in range(1, 6)]
    assert result.attrs["frequency"] == int(np.argmin(ssrs)) + 1
    np.testing.assert_allclose(result["frequencies"]["selection_ssr"], ssrs, rtol=1e-9)
    expected, _ = row_at(tables.LM_T, tables.LM_SUPF, len(y))
    assert result.attrs["f_critical_values"] == pytest.approx(expected)


def test_invariance_to_level_scale_and_row_order():
    frame = series(seed=3)
    for function, options in [(oe.fourier_adf, {"lags": 2}), (oe.fourier_lm, {"lags": 1}),
                              (oe.fourier_kpss, {"lags": 4})]:
        base = function(frame, "y", time="t", **options).attrs
        for scale, shift in [(1e-60, 0.0), (-7.0, 1e7), (1e60, 0.0)]:
            moved = frame.assign(y=frame.y * scale + shift).sample(frac=1.0, random_state=4)
            other = function(moved, "y", time="t", **options).attrs
            assert other["statistic"] == pytest.approx(base["statistic"], rel=1e-7, abs=1e-8)
            assert other["f_statistic"] == pytest.approx(base["f_statistic"], rel=1e-6, abs=1e-6)
            assert other["frequency"] == base["frequency"]


# ---- critical values: transcription, second transcription, simulation ---------------------


def row_at(nodes, rows, total):
    if total <= nodes[0]:
        return dict(zip(LEVELS, rows[0])), "first"
    if total >= nodes[-1]:
        return dict(zip(LEVELS, rows[-1])), "last"
    i = next(i for i in range(1, len(nodes)) if total <= nodes[i])
    w = (total - nodes[i - 1]) / (nodes[i] - nodes[i - 1])
    return dict(zip(LEVELS, [a + w * (b - a) for a, b in zip(rows[i - 1], rows[i])])), "inside"


# TSPDLIB (github.com/aptech/tspdlib, src/fourier_*.src): an independent transcription of the
# same published tables, typed here from those files. TSPDLIB buckets T <= 150, 151-349,
# 350-500 and > 500 for the 100/200/500/2500 rows.
TSPDLIB_ADF = {
    "constant": [
        [(-4.42, -3.81, -3.49), (-3.97, -3.27, -2.91), (-3.77, -3.07, -2.71), (-3.64, -2.97, -2.64), (-3.58, -2.93, -2.60)],
        [(-4.37, -3.78, -3.47), (-3.93, -3.26, -2.92), (-3.74, -3.06, -2.72), (-3.62, -2.98, -2.65), (-3.55, -2.94, -2.62)],
        [(-4.35, -3.76, -3.46), (-3.91, -3.26, -2.91), (-3.70, -3.06, -2.72), (-3.62, -2.97, -2.66), (-3.56, -2.94, -2.62)],
        [(-4.31, -3.75, -3.45), (-3.89, -3.25, -2.90), (-3.69, -3.05, -2.71), (-3.61, -2.96, -2.64), (-3.53, -2.93, -2.61)]],
    "trend": [
        [(-4.95, -4.35, -4.05), (-4.69, -4.05, -3.71), (-4.45, -3.78, -3.44), (-4.29, -3.65, -3.29), (-4.20, -3.56, -3.22)],
        [(-4.87, -4.31, -4.02), (-4.62, -4.01, -3.69), (-4.38, -3.77, -3.43), (-4.27, -3.63, -3.31), (-4.18, -3.56, -3.24)],
        [(-4.81, -4.29, -4.01), (-4.57, -3.99, -3.67), (-4.38, -3.76, -3.43), (-4.25, -3.64, -3.31), (-4.18, -3.56, -3.25)],
        [(-4.80, -4.27, -4.00), (-4.58, -3.98, -3.67), (-4.38, -3.75, -3.43), (-4.24, -3.63, -3.30), (-4.16, -3.55, -3.24)]],
}
TSPDLIB_ADF_F = {"constant": [(10.35, 7.58, 6.35), (10.02, 7.41, 6.25), (9.78, 7.29, 6.16), (9.72, 7.25, 6.11)],
                 "trend": [(12.21, 9.14, 7.78), (11.70, 8.88, 7.62), (11.52, 8.76, 7.53), (11.35, 8.71, 7.50)]}
TSPDLIB_KPSS = {"constant": [[(0.2699, 0.1720, 0.1318), (0.6671, 0.4152, 0.3150), (0.7182, 0.4480, 0.3393), (0.7222, 0.4592, 0.3476), (0.7386, 0.4626, 0.3518)],
                             [(0.2709, 0.1696, 0.1294), (0.6615, 0.4075, 0.3053), (0.7046, 0.4424, 0.3309), (0.7152, 0.4491, 0.3369), (0.7344, 0.4571, 0.3415)]],
                "trend": [[(0.0716, 0.0546, 0.0471), (0.2022, 0.1321, 0.1034), (0.2103, 0.1423, 0.1141), (0.2170, 0.1478, 0.1189), (0.2177, 0.1484, 0.1201)],
                          [(0.0720, 0.0539, 0.0463), (0.1968, 0.1278, 0.0995), (0.2091, 0.1404, 0.1123), (0.2111, 0.1441, 0.1155), (0.2178, 0.1465, 0.1178)]]}
TSPDLIB_LM = [
    [(-4.69, -4.10, -3.82), (-4.25, -3.57, -3.23), (-3.98, -3.31, -2.96), (-3.85, -3.18, -2.86), (-3.75, -3.11, -2.81)],
    [(-4.61, -4.07, -3.79), (-4.18, -3.55, -3.23), (-3.94, -3.30, -2.98), (-3.80, -3.18, -2.88), (-3.73, -3.12, -2.83)],
    [(-4.57, -4.05, -3.78), (-4.13, -3.54, -3.22), (-3.94, -3.31, -2.98), (-3.81, -3.19, -2.88), (-3.75, -3.14, -2.83)],
    [(-4.56, -4.03, -3.77), (-4.15, -3.54, -3.22), (-3.94, -3.30, -2.98), (-3.80, -3.19, -2.88), (-3.74, -3.13, -2.83)]]


def test_published_entries_equal_second_transcription():
    for case in ("constant", "trend"):
        for row in range(4):
            for k in range(5):
                assert tables.ADF_TAU[case][k][row] == TSPDLIB_ADF[case][row][k]
            assert tables.ADF_SUPF[case][row] == TSPDLIB_ADF_F[case][row]
        for row in range(2):
            for k in range(3):          # k = 4, 5 are simulated here (no primary copy)
                assert tables.KPSS_TAU[case][k][row] == TSPDLIB_KPSS[case][row][k]
        assert tables.KPSS_SUPF_PUBLISHED[case] == (
            {"constant": ((6.730, 4.929, 4.133), (6.281, 4.651, 3.935)),
             "trend": ((6.873, 4.972, 4.162), (6.315, 4.669, 3.928))}[case])
    for row in range(4):
        for k in range(5):
            assert tables.LM_TAU[k][row] == TSPDLIB_LM[row][k]
    # Panels b/c and the F rows are printed 10%, 5%, 1%: stored values must decrease.
    for rows in [*tables.LM_F, tables.LM_SUPF, *tables.ADF_SUPF.values(), *tables.KPSS_SUPF.values()]:
        for row in rows:
            assert row[0] > row[1] > row[2]


def tolerance(sim, published_reps, sim_reps, decimals):
    """Combined Monte Carlo error of both simulations plus the published rounding."""
    se = sim["mc_se"]
    return 4.5 * se * math.sqrt(1.0 + sim_reps / published_reps) + 0.5 * 10.0 ** -decimals


def test_published_tables_replicate_in_seeded_simulation():
    checked = 0
    for case in ("constant", "trend"):
        for row, total in enumerate((100, 200, 500)):
            cell = SIMULATED["fourier_adf"][f"{case}/{total}"]
            for k in range(1, 6):
                for level, value in zip(LEVELS, tables.ADF_TAU[case][k - 1][row]):
                    sim = cell["tau"][str(k)][level]
                    assert abs(value - sim["value"]) <= tolerance(sim, 20_000, cell["reps"], 2)
                    checked += 1
            for level, value in zip(LEVELS, tables.ADF_SUPF[case][row]):
                sim = cell["sup_f"][level]
                assert abs(value - sim["value"]) <= tolerance(sim, 20_000, cell["reps"], 2)
                checked += 1
        for row, total in enumerate((100, 500)):
            cell = SIMULATED["fourier_kpss"][f"{case}/{total}"]
            for k in range(1, 6):
                for level, value in zip(LEVELS, tables.KPSS_TAU[case][k - 1][row]):
                    sim = cell["tau"][str(k)][level]
                    if k in tables.KPSS_SIMULATED_FREQUENCIES:
                        assert value == round(sim["value"], 4)
                    else:
                        assert abs(value - sim["value"]) <= tolerance(sim, 50_000, cell["reps"], 4)
                    checked += 1
            for level, value in zip(LEVELS, tables.KPSS_SUPF[case][row]):
                assert value == round(cell["sup_f"][level]["value"], 3)
    for row, total in enumerate((100, 200, 500)):
        cell = SIMULATED["fourier_lm"][str(total)]
        for k in range(1, 6):
            for level, value in zip(LEVELS, tables.LM_TAU[k - 1][row]):
                sim = cell["tau"][str(k)][level]
                assert abs(value - sim["value"]) <= tolerance(sim, 100_000, cell["reps"], 2)
            for level, value in zip(LEVELS, tables.LM_F[k - 1][row]):
                sim = cell["f"][str(k)][level]
                assert abs(value - sim["value"]) <= tolerance(sim, 100_000, cell["reps"], 2)
                checked += 2
        for level, value in zip(LEVELS, tables.LM_SUPF[row]):
            sim = cell["sup_f"][level]
            assert abs(value - sim["value"]) <= tolerance(sim, 100_000, cell["reps"], 2)
    assert checked == 258


def test_published_kpss_supremum_f_does_not_replicate_at_t100():
    # Documented reason for using simulated supremum-F critical values.
    for case in ("constant", "trend"):
        cell = SIMULATED["fourier_kpss"][f"{case}/100"]["sup_f"]
        for level, published in zip(LEVELS, tables.KPSS_SUPF_PUBLISHED[case][0]):
            sim = cell[level]
            assert (published - sim["value"]) / sim["mc_se"] > 5.0


def test_critical_value_interpolation_and_notes():
    frame = series(n=150)
    result = oe.fourier_adf(frame, "y", frequency=3, trend="trend")
    expected, _ = row_at(tables.ADF_T, tables.ADF_TAU["trend"][2], 150)
    assert result.attrs["critical_values"] == pytest.approx(expected)
    assert result.attrs["critical_value_source"].startswith("Enders and Lee (2012)")
    short = oe.fourier_adf(series(n=40), "y", frequency=1)
    assert short.attrs["critical_values"] == dict(zip(LEVELS, tables.ADF_TAU["constant"][0][0]))
    assert any("tabulated from T = 100" in note for note in short.attrs["notes"])
    kpss_simulated = oe.fourier_kpss(series(kind="stationary"), "y", frequency=5)
    assert "simulation" in kpss_simulated.attrs["critical_value_source"]
    kpss_published = oe.fourier_kpss(series(kind="stationary"), "y", frequency=2)
    assert "Becker, Enders and Lee" in kpss_published.attrs["critical_value_source"]


def test_rejection_flags_and_no_p_values():
    result = oe.fourier_adf(series(kind="stationary", n=200), "y", lags="bic")
    tau = result.attrs["statistic"]
    assert result.attrs["reject"] == {lv: tau < cv for lv, cv in result.attrs["critical_values"].items()}
    assert result.attrs["p_value"] is None and result["test"].p_value.isna().all()
    assert "quantiles only" in result.attrs["p_value_reason"]
    kpss = oe.fourier_kpss(series(n=200), "y")
    assert kpss.attrs["reject"] == {lv: kpss.attrs["statistic"] > cv
                                    for lv, cv in kpss.attrs["critical_values"].items()}


def test_max_frequency_below_five_has_no_supremum_critical_values():
    for function in (oe.fourier_adf, oe.fourier_kpss, oe.fourier_lm):
        result = function(series(), "y", max_frequency=3)
        assert result.attrs["max_frequency"] == 3
        assert result.attrs["f_critical_values"] is None
        assert "max_frequency=3" in result.attrs["f_inference_reason"]
        assert len(result["frequencies"]) == 3


# ---- contracts -------------------------------------------------------------------------


@pytest.mark.parametrize("function", [oe.fourier_adf, oe.fourier_kpss, oe.fourier_lm])
def test_failure_contract(function):
    frame = series()
    with pytest.raises(AnalysisError) as error:
        function(frame.assign(y=frame.y.where(frame.index != 5)), "y")
    assert error.value.code == "missing_values"
    with pytest.raises(AnalysisError) as error:
        function(frame.iloc[:8], "y")
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        function(frame.assign(y=4.0), "y")
    assert error.value.code == "constant_series"
    for bad in ({"frequency": 0}, {"frequency": 6}, {"frequency": True}, {"frequency": "best"},
                {"frequency": 2, "max_frequency": 3}, {"max_frequency": 7}):
        with pytest.raises(AnalysisError) as error:
            function(frame, "y", **bad)
        assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        function(frame.drop(index=[10]), "y", time="t")
    assert error.value.code == "time_gaps"
    with pytest.raises(AnalysisError) as error:
        function(frame, "missing")
    assert error.value.code == "missing_columns"


def test_lag_and_option_errors():
    frame = series()
    for function in (oe.fourier_adf, oe.fourier_lm):
        with pytest.raises(AnalysisError) as error:
            function(frame, "y", lags=-1)
        assert error.value.code == "invalid_lags"
        with pytest.raises(AnalysisError) as error:
            function(frame, "y", lags="hq")
        assert error.value.code == "invalid_option"
        with pytest.raises(AnalysisError) as error:
            function(frame, "y", lags=2, maxlag=4)
        assert error.value.code == "invalid_option"
        with pytest.raises(AnalysisError) as error:
            function(frame, "y", lags="aic", maxlag=200)
        assert error.value.code == "invalid_lags"
    with pytest.raises(AnalysisError) as error:
        oe.fourier_adf(frame, "y", trend="quadratic")
    assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        oe.fourier_kpss(frame, "y", lags=3, auto=True)
    assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        oe.fourier_kpss(frame, "y", lags=len(frame))
    assert error.value.code == "invalid_lags"


def test_deterministic_fourier_path_is_refused():
    t = np.arange(1, 121)
    frame = pd.DataFrame({"y": 2.0 + np.sin(2 * np.pi * 2 * t / 120)})
    with pytest.raises(AnalysisError) as error:
        oe.fourier_kpss(frame, "y", frequency=2)
    assert error.value.code == "constant_series"
    walk = pd.DataFrame({"y": np.cumsum(np.ones(120)) + np.sin(2 * np.pi * t / 120)})
    with pytest.raises(AnalysisError) as error:
        oe.fourier_adf(walk, "y", trend="trend", frequency=1)
    assert error.value.code in {"perfect_fit", "collinear_regressors"}


def test_json_text_and_latex_rendering():
    for result in (oe.fourier_adf(series(), "y", lags="t", maxlag=4),
                   oe.fourier_kpss(series(kind="stationary"), "y", trend="trend"),
                   oe.fourier_lm(series(), "y", frequency=1)):
        json.dumps(result.attrs, allow_nan=False)
        json.dumps(result["test"].attrs, allow_nan=False)
        text = str(result)
        assert "[test]" in text and "[frequencies]" in text
        assert "\\begin{tabular}" in result.to_latex()
        assert set(result["test"].index) == {"tau", "F"}


def test_public_exports_and_docstrings():
    from openecon.econometrics import registry

    exports = registry.public_exports()
    for name in ("fourier_adf", "fourier_kpss", "fourier_lm", "ers_pt", "hegy"):
        assert exports[name][0].startswith("openecon.econometrics.unitroot.")
        assert callable(getattr(oe, name)) and name in dir(oe)
        assert len(getattr(oe, name).__doc__) > 1500
