"""oe.ltable against hand-computed actuarial life tables and explicit NumPy loops."""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.duration.survfunc import survdiff

import openecon as oe
from openecon.analysis_contracts import AnalysisError

SMALL = {"t": [1, 2, 2, 3, 5, 6, 8, 9, 12, 15], "d": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]}


def test_hand_computed_table():
    table = oe.ltable(SMALL, "t", failure="d", intervals=5)["table"]
    # [0,5): N=10, d=3, m=1 -> n=9.5, q=3/9.5; [5,10): N=6, d=3, m=1 -> n=5.5
    assert list(table.entering) == [10.0, 6.0, 2.0, 1.0]
    assert list(table.deaths) == [3.0, 3.0, 0.0, 1.0]
    assert list(table.lost) == [1.0, 1.0, 1.0, 0.0]
    assert_allclose(table.at_risk, [9.5, 5.5, 1.5, 1.0])
    q1, q2 = 3 / 9.5, 3 / 5.5
    assert_allclose(table.survival[:2], [1 - q1, (1 - q1) * (1 - q2)])
    se2 = (1 - q1) * (1 - q2) * math.sqrt(3 / (9.5 * 6.5) + 3 / (5.5 * 2.5))
    assert table.std_error[1] == pytest.approx(se2)
    hazard1 = q1 / ((1 - q1 / 2) * 5)
    assert table.hazard[0] == pytest.approx(hazard1)
    assert table.hazard_std_error[0] == pytest.approx(
        hazard1 * math.sqrt((1 - (5 * hazard1 / 2) ** 2) / 3))
    assert table.density[1] == pytest.approx((1 - q1) * q2 / 5)
    assert table.density_std_error[1] == pytest.approx(
        (1 - q1) * q2 / 5 * math.sqrt(q1 / (9.5 * (1 - q1)) + (1 - q2) / (5.5 * q2)))
    z = stats.norm.isf(0.025)
    sigma = math.sqrt(3 / (9.5 * 6.5)) / abs(math.log(6.5 / 9.5))
    assert table.ci_low[0] == pytest.approx((1 - q1) ** math.exp(z * sigma))
    assert table.ci_high[0] == pytest.approx((1 - q1) ** math.exp(-z * sigma))


def test_noadjust_cutpoints_and_chi2_hazard_interval():
    table = oe.ltable(SMALL, "t", failure="d", intervals=[4, 10], noadjust=True,
                      alpha=0.1)["table"]
    assert list(table.interval_start) == [0.0, 4.0, 10.0]
    assert np.isnan(table.interval_end.iloc[-1]) and np.isnan(table.hazard.iloc[-1])
    first = table.iloc[0]
    assert first.at_risk == first.entering == 10.0
    hazard = first.deaths / 10 / 4
    assert first.hazard == pytest.approx(hazard)
    assert first.hazard_std_error == pytest.approx(hazard / math.sqrt(first.deaths))
    d = first.deaths
    assert first.hazard_ci_low == pytest.approx(hazard * stats.chi2.ppf(0.05, 2 * d) / (2 * d))
    assert first.hazard_ci_high == pytest.approx(hazard * stats.chi2.ppf(0.95, 2 * d) / (2 * d))


def test_groups_weights_and_homogeneity_tests():
    rng = np.random.default_rng(5)
    n = 200
    frame = pd.DataFrame({"g": rng.integers(0, 3, n), "w": rng.integers(1, 3, n)})
    t = rng.exponential(4 / (1 + 0.5 * frame.g))
    c = rng.exponential(6, n)
    frame["t"], frame["d"] = np.round(np.minimum(t, c), 2) + 0.01, (t <= c).astype(float)
    out = oe.ltable(frame, "t", failure="d", by="g", intervals=2)
    tests = out["tests"]
    deaths = frame.groupby("g").d.sum().to_numpy()
    total = frame.groupby("g").t.sum().to_numpy()
    lr = 2 * (deaths.sum() * np.log(total.sum() / deaths.sum())
              - np.sum(deaths * np.log(total / deaths)))
    assert tests.loc["likelihood_ratio", "statistic"] == pytest.approx(lr, rel=1e-12)
    assert tests.loc["likelihood_ratio", "df"] == 2
    chi2, _ = survdiff(frame.t, frame.d, frame.g)
    assert tests.loc["logrank", "statistic"] == pytest.approx(chi2, rel=1e-10)
    for level in range(3):
        rows = out["table"][out["table"].group == level]
        sub = frame[frame.g == level]
        assert rows.deaths.sum() == sub.d.sum() and rows.lost.sum() == (1 - sub.d).sum()
    weighted = oe.ltable(frame, "t", failure="d", weights="w", intervals=2)["table"]
    expanded = frame.loc[frame.index.repeat(frame.w)]
    plain = oe.ltable(expanded, "t", failure="d", intervals=2)["table"]
    assert_allclose(weighted.survival, plain.survival, rtol=1e-12)
    assert_allclose(weighted.hazard, plain.hazard, rtol=1e-12)
    assert "Life table" in str(out) and "tabular" in out.to_latex()


def test_failure_contract():
    with pytest.raises(AnalysisError) as error:
        oe.ltable(SMALL, "t", failure="d", intervals=-1)
    assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        oe.ltable(SMALL, "t", failure="d", intervals=[3, 2])
    assert error.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as error:
        oe.ltable({"t": [1e9, 1.0]}, "t", intervals=1)
    assert error.value.code == "too_many_intervals"
    with pytest.raises(AnalysisError) as error:
        oe.ltable({"t": [-1.0, 1.0]}, "t")
    assert error.value.code == "invalid_survival_time"
    zero = oe.ltable({"t": [0.0, 0.5, 1.5], "d": [1, 1, 1]}, "t", failure="d")
    assert zero["table"].deaths.iloc[0] == 2.0
