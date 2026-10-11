"""ERS point-optimal P_T: NumPy oracle, R urca::ur.ers fixture, Table 1 provenance, contracts.

R values: tests/fixtures/unitroot_fourier/ers_urca.json from
scripts/generate_unitroot_fourier_oracles.R (urca 1.3-4). Simulated quantiles:
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
from openecon.econometrics.unitroot import ers

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "unitroot_fourier"
URCA = json.loads((FIXTURES / "ers_urca.json").read_text())
SIMULATED = json.loads((FIXTURES / "simulated_quantiles.json").read_text())


def frame(seed=2, n=120):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    return pd.DataFrame({"y": 100 + np.cumsum(e + 0.4 * np.r_[0.0, e[:-1]]),
                         "t": np.arange(n)})


def oracle(y, trend, p):
    """ERS eqs. (13)-(15), (20)-(21) written out with NumPy."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    c = -7.0 if trend == "constant" else -13.5
    a = 1 + c / T
    z = np.ones((T, 1)) if trend == "constant" else np.column_stack([np.ones(T), np.arange(1, T + 1)])

    def ssr(alpha):
        yq = np.r_[y[0], y[1:] - alpha * y[:-1]]
        zq = np.vstack([z[:1], z[1:] - alpha * z[:-1]])
        b = np.linalg.lstsq(zq, yq, rcond=None)[0]
        return float(np.sum((yq - zq @ b) ** 2))

    dy = np.diff(y)
    rows = np.arange(p + 1, T)
    X = np.column_stack([np.ones(len(rows)), y[rows - 1], *[dy[rows - 1 - j] for j in range(1, p + 1)]])
    b = np.linalg.lstsq(X, dy[rows - 1], rcond=None)[0]
    resid = dy[rows - 1] - X @ b
    s2 = float(resid @ resid) / (len(rows) - X.shape[1])
    omega = s2 / (1 - b[2:].sum()) ** 2
    return (ssr(a) - a * ssr(1.0)) / omega, omega


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("p", [0, 1, 3, 6])
def test_matches_numpy_equations(trend, p):
    data = frame()
    result = oe.ers_pt(data, "y", time="t", trend=trend, lags=p)
    expected, omega = oracle(data.y, trend, p)
    assert result.attrs["statistic"] == pytest.approx(expected, rel=1e-10)
    assert result.attrs["spectral_variance"] == pytest.approx(omega, rel=1e-10)
    assert result.attrs["nobs"] == len(data) - 1 - p
    assert result.attrs["gls_alpha"] == 1 + ers.GLS_C[trend] / len(data)


def test_matches_r_urca_reference():
    worst = 0.0
    for case in URCA["cases"]:
        y = np.asarray(case["y"])
        result = oe.ers_pt(pd.DataFrame({"y": y}), "y", trend=case["model"], lags=case["lag"])
        expected = case["teststat"]
        if case["model"] == "trend":
            # urca's S(1) demeans (0, Delta y) over all T entries; the GLS residual sum
            # under alpha = 1 is smaller by (y_T - y_1)^2 / (T (T - 1)).
            n = len(y)
            expected += (1 - 13.5 / n) * (y[-1] - y[0]) ** 2 / (n * (n - 1)) \
                / result.attrs["spectral_variance"]
        worst = max(worst, abs(result.attrs["statistic"] - expected) / abs(expected))
        # urca's critical values: Table 1 rows, chosen by T < 50, < 100, <= 200, > 200.
        n = len(y)
        row = 0 if n < 50 else 1 if n < 100 else 2 if n <= 200 else 4
        table = ers.PT_CRITICAL[case["model"]][row]
        np.testing.assert_allclose(case["cval"], [table[0], table[2], table[3]])
    assert worst < 1e-11
    assert len(URCA["cases"]) == 36 and URCA["meta"]["urca"] == "1.3.4"


def test_trend_case_urca_difference_is_the_documented_identity():
    case = next(c for c in URCA["cases"] if c["model"] == "trend" and c["kind"] == "trending")
    y = np.asarray(case["y"])
    ours = oe.ers_pt(pd.DataFrame({"y": y}), "y", trend="trend", lags=case["lag"]).attrs
    assert ours["statistic"] - case["teststat"] > 0


def test_bic_selection_on_common_sample():
    data = frame(seed=9, n=150)
    y = data.y.to_numpy()
    maxlag, minlag = 6, 2
    result = oe.ers_pt(data, "y", minlag=minlag, maxlag=maxlag)
    dy = np.diff(y)
    rows = np.arange(maxlag + 1, len(y))
    values = []
    for p in range(minlag, maxlag + 1):
        X = np.column_stack([np.ones(len(rows)), y[rows - 1], *[dy[rows - 1 - j] for j in range(1, p + 1)]])
        b = np.linalg.lstsq(X, dy[rows - 1], rcond=None)[0]
        ssr = float(np.sum((dy[rows - 1] - X @ b) ** 2))
        values.append(len(rows) * math.log(ssr / len(rows)) + X.shape[1] * math.log(len(rows)))
    chosen = minlag + int(np.argmin(values))
    assert result.attrs["lags"] == chosen and result.attrs["lag_method"] == "bic"
    assert result.attrs["statistic"] == pytest.approx(oracle(y, "constant", chosen)[0], rel=1e-10)
    assert [row["lags"] for row in result.attrs["lag_selection_results"]] == list(range(2, 7))
    for method in ("aic", "t"):
        assert oe.ers_pt(data, "y", lags=method, maxlag=4).attrs["lag_method"] == method


def test_table_one_transcription_and_simulation():
    # NBER TWP 130, Table 1, panels A and B (1%, 2.5%, 5%, 10%).
    assert ers.ERS_T == (50.0, 100.0, 200.0, 500.0, float("inf"))
    assert ers.PT_CRITICAL["constant"][4] == (1.99, 2.55, 3.26, 4.48)
    assert ers.PT_CRITICAL["trend"][4] == (3.96, 4.78, 5.62, 6.89)
    for case in ("constant", "trend"):
        for row, total in enumerate((50, 100, 200, 500)):
            cell = SIMULATED["ers_pt"][f"{case}/{total}"]
            for level, value in zip(("1%", "2.5%", "5%", "10%"), ers.PT_CRITICAL[case][row]):
                sim = cell["p_t"][level]
                # ERS used 20,000 replications and two decimals.
                tolerance = 4.5 * sim["mc_se"] * math.sqrt(1 + cell["reps"] / 20_000) + 0.005
                assert abs(value - sim["value"]) <= tolerance, (case, total, level)


def test_critical_value_interpolation_and_rejection():
    data = frame(n=150)
    result = oe.ers_pt(data, "y", lags=2)
    lower, upper = ers.PT_CRITICAL["constant"][1], ers.PT_CRITICAL["constant"][2]
    expected = [a + 0.5 * (b - a) for a, b in zip(lower, upper)]
    assert list(result.attrs["critical_values"].values()) == pytest.approx(expected)
    assert result.attrs["reject"] == {lv: result.attrs["statistic"] < cv
                                      for lv, cv in result.attrs["critical_values"].items()}
    big = oe.ers_pt(frame(n=700), "y", lags=1)
    assert big.attrs["critical_values"] == dict(zip(ers.LEVELS, ers.PT_CRITICAL["constant"][4]))
    assert any("asymptotic row" in note for note in big.attrs["notes"])
    small = oe.ers_pt(frame(n=30), "y", lags=1)
    assert small.attrs["critical_values"] == dict(zip(ers.LEVELS, ers.PT_CRITICAL["constant"][0]))
    stationary = pd.DataFrame({"y": np.random.default_rng(1).normal(size=200)})
    assert oe.ers_pt(stationary, "y", lags=0).attrs["reject"]["1%"]
    assert result.attrs["p_value"] is None and result.p_value.isna().all()


def test_invariance_and_rendering():
    data = frame(seed=4)
    base = oe.ers_pt(data, "y", time="t", trend="trend", lags=2).attrs["statistic"]
    for scale, shift in [(-1e-30, 0.0), (1e30, 0.0), (3.0, 1e6)]:
        moved = data.assign(y=data.y * scale + shift).sample(frac=1.0, random_state=1)
        assert oe.ers_pt(moved, "y", time="t", trend="trend", lags=2).attrs["statistic"] == \
            pytest.approx(base, rel=1e-7)
    result = oe.ers_pt(data, "y")
    json.dumps(result.attrs, allow_nan=False)
    assert "\\begin{tabular}" in result.to_latex()
    assert "P_T" in str(result)


def test_failure_contract():
    data = frame()
    cases = [
        (dict(data=data.assign(y=data.y.where(data.index != 3))), "missing_values"),
        (dict(data=data.iloc[:6]), "insufficient_observations"),
        (dict(data=data.assign(y=1.5)), "constant_series"),
        (dict(data=data, trend="none"), "invalid_option"),
        (dict(data=data, lags="hq"), "invalid_option"),
        (dict(data=data, lags=-2), "invalid_lags"),
        (dict(data=data, lags=2, maxlag=4), "invalid_option"),
        (dict(data=data, lags=2, minlag=1), "invalid_option"),
        (dict(data=data, minlag=5, maxlag=3), "invalid_lags"),
        (dict(data=data, maxlag=80), "invalid_lags"),
        (dict(data=data.drop(index=[7]), time="t"), "time_gaps"),
    ]
    for kwargs, code in cases:
        options = {"y": "y", **kwargs}
        with pytest.raises(AnalysisError) as error:
            oe.ers_pt(**options)
        assert error.value.code == code, kwargs
    linear = pd.DataFrame({"y": np.arange(60.0) * 2.0})
    with pytest.raises(AnalysisError) as error:
        oe.ers_pt(linear, "y", lags=1)
    assert error.value.code in {"perfect_fit", "collinear_regressors"}
