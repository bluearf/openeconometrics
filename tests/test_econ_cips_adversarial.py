"""Independent equation-54 and numerical contracts for CADF/CIPS.

The oracle uses explicit calendar rows and single-unit NumPy QR/SVD. It does
not import the Torch design, sample, fit, mean, or lag-selection helpers.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot.cips import xtcips
from openecon.econometrics.unitroot.cips_tables import critical_values


def panel_data(n=13, total=73, seed=88412, integer=False):
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(n, total))
    innovations += np.linspace(0.4, 1.6, n)[:, None] * rng.normal(size=total)
    if integer:
        innovations = np.rint(innovations * 8)
    else:
        for j in range(1, total):
            innovations[:, j] += 0.4 * innovations[:, j - 1]
    values = innovations.cumsum(axis=1)
    values += (np.arange(n) * 3)[:, None]
    frame = pd.DataFrame({
        "unit": np.repeat(np.arange(n), total),
        "period": np.tile(np.arange(total), n),
        "value": values.reshape(-1),
    })
    return frame, values


def oracle(values, unit, order, trend, start=None):
    start = order if start is None else start
    total = values.shape[1]
    mean = values.mean(axis=0)
    rows, target = [], []
    for t in range(start + 1, total):
        row = [values[unit, t - 1], mean[t - 1]]
        row += [mean[t - j] - mean[t - j - 1] for j in range(order + 1)]
        row += [values[unit, t - j] - values[unit, t - j - 1]
                for j in range(1, order + 1)]
        if trend != "none":
            row += [1.0]
        if trend == "trend":
            row += [float(t)]
        rows.append(row)
        target.append(values[unit, t] - values[unit, t - 1])
    x, y = np.asarray(rows), np.asarray(target)
    # Orthogonalize to compute classical covariance independently of Torch QR.
    scale = np.max(np.abs(x), axis=0)
    scaled = x / scale
    q, r = np.linalg.qr(scaled, mode="reduced")
    beta_scaled = np.linalg.lstsq(scaled, y, rcond=None)[0]
    residual = y - scaled @ beta_scaled
    ssr = float(residual @ residual)
    variance = ssr / (len(y) - x.shape[1])
    inverse = np.linalg.solve(r, np.eye(x.shape[1]))
    se_scaled = np.sqrt(variance * np.sum(inverse * inverse, axis=1))
    return {
        "statistic": float(beta_scaled[0] / se_scaled[0]),
        "beta": float(beta_scaled[0] / scale[0]),
        "se": float(se_scaled[0] / scale[0]),
        "ssr": ssr, "nobs": len(y), "k": x.shape[1],
    }


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_explicit_equation54_oracle_on_shuffled_calendar_rows(trend, order):
    frame, values = panel_data()
    answer = xtcips(frame.sample(frac=1, random_state=703), "value", "unit", "period",
                    lags=order, trend=trend, individual=True, inference="none", block_size=4)
    expected = [oracle(values, i, order, trend) for i in range(len(values))]
    for column, key in (("cadf", "statistic"), ("level_coefficient", "beta"), ("std_error", "se")):
        assert_allclose(answer[column], [item[key] for item in expected], rtol=2e-11, atol=2e-11)
    assert list(answer.index) == list(range(len(values)))
    assert_allclose(answer.nobs, values.shape[1] - 1 - order, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["aic", "bic", "hqic"])
@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
def test_automatic_lags_use_common_holdback_then_individual_refit(method, trend):
    frame, values = panel_data(seed=9829)
    reach = 4
    answer = xtcips(frame, "value", "unit", "period", lags=method, maxlag=reach,
                    trend=trend, individual=True, inference="none", block_size=5)
    expected_orders, expected_stats = [], []
    for i in range(len(values)):
        candidates = [oracle(values, i, p, trend, start=reach) for p in range(reach + 1)]
        nobs = candidates[0]["nobs"]
        penalty = {"aic": 2, "bic": math.log(nobs), "hqic": 2 * math.log(math.log(nobs))}[method]
        scores = [nobs * math.log(fit["ssr"] / nobs) + penalty * fit["k"] for fit in candidates]
        selected = int(np.argmin(scores))
        expected_orders.append(selected)
        expected_stats.append(oracle(values, i, selected, trend)["statistic"])
    assert list(answer.lags) == expected_orders
    assert_allclose(answer.cadf, expected_stats, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("scale", [-1e120, 1e-120])
def test_global_unit_scaling_preserves_statistics_with_native_rescaling(trend, scale):
    frame, _ = panel_data(seed=2104)
    base = xtcips(frame, "value", "unit", "period", lags=2, trend=trend,
                  individual=True, inference="none")
    frame.value *= scale
    answer = xtcips(frame, "value", "unit", "period", lags=2, trend=trend,
                    individual=True, inference="none")
    assert_allclose(answer.cadf, base.cadf, rtol=2e-11, atol=2e-11)
    assert_allclose(answer.level_coefficient, base.level_coefficient, rtol=2e-11, atol=2e-11)
    assert_allclose(answer.std_error, base.std_error, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_large_exact_unit_offsets_are_absorbed_without_mutating_input(trend):
    frame, values = panel_data(integer=True)
    original = frame.copy(deep=True)
    base = xtcips(frame, "value", "unit", "period", lags=2, trend=trend, individual=True)
    pd.testing.assert_frame_equal(frame, original)
    frame.value += np.repeat((np.arange(len(values)) + 1) * 1e14, values.shape[1])
    shifted = frame.copy(deep=True)
    answer = xtcips(frame, "value", "unit", "period", lags=2, trend=trend, individual=True)
    pd.testing.assert_frame_equal(frame, shifted)
    assert_allclose(answer.cadf, base.cadf, rtol=2e-12, atol=2e-12)


def test_table_T_is_potential_differences_not_refit_nobs_or_levels():
    frame, _ = panel_data(n=20, total=16, seed=619)
    answer = xtcips(frame, "value", "unit", "period", lags=1, trend="constant", individual=True)
    # Final TableI(b): N20/T15. Lost augmentation row must not choose T14.
    assert answer.attrs["critical_T"] == 15
    assert_allclose(answer.nobs, 14)
    assert_allclose(answer.critical_1pct, -4.62)
    assert_allclose(answer.critical_5pct, -3.54)


def test_sparse_final_parentheses_retain_column_positions_and_raw_cells():
    # Final TableII(a), T10/N15 is unchanged at5%, changed at10%.
    assert critical_values("cips", "none", 15, 10, True)["5%"] == -1.71
    assert critical_values("cips", "none", 15, 10, True)["10%"] == -1.55
    # TableII(c), T15/N100 only1%/5% are unchanged; neighboring cells differ.
    assert critical_values("cips", "trend", 100, 15, True)["1%"] == -2.74
    assert critical_values("cips", "trend", 100, 15, True)["5%"] == -2.59
    assert critical_values("cips", "trend", 100, 15, True)["10%"] == -2.51


def test_outside_tables_returns_no_decision_and_permits_valid_small_panels():
    frame, _ = panel_data(n=7, total=25)
    with pytest.raises(AnalysisError) as caught:
        xtcips(frame, "value", "unit", "period")
    assert caught.value.code == "critical_values_unavailable"
    answer = xtcips(frame, "value", "unit", "period", lags=1, inference="none")
    assert answer.attrs["p_value"] is None
    assert answer[["critical_1pct", "reject_1pct"]].isna().all().all()


def test_calendar_months_and_large_exact_integer_periods_preserve_lags():
    frame, _ = panel_data(n=11, total=51)
    base = xtcips(frame, "value", "unit", "period", lags=3, inference="none", individual=True)
    month = frame.copy()
    month.period = np.tile(pd.date_range("2020-01-01", periods=51, freq="MS"), 11)
    dated = xtcips(month, "value", "unit", "period", lags=3, inference="none", individual=True)
    large = frame.copy()
    large.period += 2**60
    integer = xtcips(large, "value", "unit", "period", lags=3, inference="none", individual=True)
    assert_allclose(dated.cadf, base.cadf, rtol=0, atol=0)
    assert_allclose(integer.cadf, base.cadf, rtol=0, atol=0)


def test_rank_deficiency_in_one_unit_fails_without_dropping_it():
    frame, _ = panel_data()
    frame.loc[frame.unit == 5, "value"] = 6.0
    with pytest.raises(AnalysisError) as caught:
        xtcips(frame, "value", "unit", "period", lags=1, inference="none", block_size=2)
    assert caught.value.code in {"collinear_regressors", "constant_series"}


def test_blocks_do_not_change_selected_lags_or_all_unit_outputs():
    frame, _ = panel_data(n=17, total=87)
    one = xtcips(frame, "value", "unit", "period", lags="aic", maxlag=4,
                 individual=True, inference="none", block_size=1)
    many = xtcips(frame, "value", "unit", "period", lags="aic", maxlag=4,
                  individual=True, inference="none", block_size=17)
    assert list(one.lags) == list(many.lags)
    assert_allclose(one.cadf, many.cadf, rtol=2e-13, atol=2e-13)
