"""HEGY seasonal unit-root tests: R uroot statistics, NumPy oracle, simulated critical values.

* tests/fixtures/unitroot_hegy/uroot_reference.json (scripts/generate_unitroot_hegy_oracles.R):
  uroot 2.1-3 ``hegy.test`` statistics and chosen lags on seeded series, the
  HEGY (1990) and Beaulieu-Miron (1993) tables as uroot stores them, the
  Diaz-Emparanza (2014) response-surface quantiles and a 3,000-replication size
  check of those response surfaces.
* tests/fixtures/unitroot_hegy/simulated_critical_values.json
  (scripts/generate_unitroot_hegy_oracles.py): the seeded simulation behind
  ``unitroot/hegy_tables.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot import hegy_tables
from openecon.econometrics.unitroot.hegy import _critical

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "unitroot_hegy"
UROOT = json.loads((FIXTURES / "uroot_reference.json").read_text())
SIMULATED = json.loads((FIXTURES / "simulated_critical_values.json").read_text())
LEVELS = ("1%", "2.5%", "5%", "10%")


def data(seasons=4, n=96, seed=3, kind="walk"):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(n):
        x[t] = (x[t - seasons] if t >= seasons and kind == "walk" else 0.0) + e[t]
        if kind == "stationary" and t >= 1:
            x[t] += 0.5 * x[t - 1]
    pattern = np.resize([3.0, -1.0, 0.5, -2.5, 1.0, 0.0, -0.5, 1.5, -2.0, 0.7, 0.3, -1.0][:seasons], n)
    return pd.DataFrame({"y": x + pattern + 0.03 * np.arange(n), "t": np.arange(1, n + 1)})


# ---- NumPy oracle written from the filter definitions --------------------------------------


def filter_series(x, coefficients):
    """sum_j c_j x_{t-j} for t >= len(c) - 1 (NaN before)."""
    out = np.full(len(x), np.nan)
    for t in range(len(coefficients) - 1, len(x)):
        out[t] = sum(c * x[t - j] for j, c in enumerate(coefficients))
    return out


def oracle(x, seasons, case, p):
    """HEGY regression with the filters Delta_S / (1 - L), -Delta_S / (1 + L) and
    Delta_S / (1 - 2 cos(w) L + L^2) (two lags each), the representation of
    Smith and Taylor (1999); polynomial division gives the filter coefficients."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    delta = np.zeros(seasons + 1)
    delta[0], delta[seasons] = 1.0, -1.0                            # 1 - L^S, low order first
    columns = []
    q, r = np.polynomial.polynomial.polydiv(delta, [1.0, -1.0])
    assert np.allclose(r, 0)
    columns.append(np.r_[np.nan, filter_series(x, q)[:-1]])         # y1_{t-1}
    q, r = np.polynomial.polynomial.polydiv(delta, [1.0, 1.0])
    assert np.allclose(r, 0)
    columns.append(np.r_[np.nan, -filter_series(x, q)[:-1]])        # y2_{t-1}
    for harmonic in range(1, seasons // 2):
        w = 2 * np.pi * harmonic / seasons
        q, r = np.polynomial.polynomial.polydiv(delta, [1.0, -2 * np.cos(w), 1.0])
        assert np.allclose(r, 0, atol=1e-12)
        filtered = filter_series(x, q)
        columns.append(np.r_[np.nan, filtered[:-1]])
        columns.append(np.r_[np.nan, np.nan, filtered[:-2]])
    target = np.r_[np.full(seasons, np.nan), x[seasons:] - x[:-seasons]]
    H = np.column_stack(columns)
    rows = np.arange(seasons + p, T)
    n = T - seasons
    det = []
    if case != "none":
        det.append(np.ones(T))
    if case in ("trend", "seasonal_trend"):
        det.append(np.arange(T, dtype=float))
    if case in ("seasonal", "seasonal_trend"):
        det.extend((np.arange(T) % seasons == s).astype(float) for s in range(1, seasons))
    lags = [np.r_[np.full(j, np.nan), target[:-j]] for j in range(1, p + 1)]
    X = np.column_stack([*det, H, *lags])[rows]
    y = target[rows]
    assert not np.isnan(X).any() and n == T - seasons
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ssr = float(np.sum((y - X @ beta) ** 2))
    df = len(y) - X.shape[1]
    se = np.sqrt(ssr / df * np.diag(np.linalg.inv(X.T @ X)))
    k = len(det)

    def F(drop):
        keep = [i for i in range(X.shape[1]) if i not in {k + d for d in drop}]
        b, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
        ssr_r = float(np.sum((y - X[:, keep] @ b) ** 2))
        return ((ssr_r - ssr) / len(drop)) / (ssr / df)

    out = {"t_1": beta[k] / se[k], "t_2": beta[k + 1] / se[k + 1]}
    for harmonic in range(1, seasons // 2):
        out[f"F_{2 * harmonic + 1}_{2 * harmonic + 2}"] = F([2 * harmonic, 2 * harmonic + 1])
    out[f"F_2_{seasons}"] = F(list(range(1, seasons)))
    out[f"F_1_{seasons}"] = F(list(range(seasons)))
    return out, df


@pytest.mark.parametrize("seasons,n", [(4, 72), (12, 180)])
@pytest.mark.parametrize("case", ["none", "constant", "trend", "seasonal", "seasonal_trend"])
@pytest.mark.parametrize("p", [0, 2])
def test_statistics_match_numpy_filter_oracle(seasons, n, case, p):
    frame = data(seasons=seasons, n=n)
    result = oe.hegy(frame, "y", time="t", seasons=seasons, deterministic=case, lags=p)
    expected, df = oracle(frame.y.to_numpy(), seasons, case, p)
    assert list(result.index) == list(expected)
    np.testing.assert_allclose(result.statistic.to_numpy(dtype=float), list(expected.values()),
                               rtol=1e-8, atol=1e-9)
    assert result.attrs["df_resid"] == df and result.attrs["nobs"] == n - seasons - p


def test_statistics_and_lag_choice_match_r_uroot():
    series = {item["name"]: np.asarray(item["y"], dtype=float) for item in UROOT["series"]}
    worst = 0.0
    for case in UROOT["cases"]:
        frame = pd.DataFrame({"y": series[case["series"]]})
        lags = case["maxlag"] if case["lag_method"] == "fixed" else case["lag_method"].lower()
        options = {} if case["lag_method"] == "fixed" else {"maxlag": case["maxlag"]}
        result = oe.hegy(frame, "y", seasons=case["seasons"], deterministic=case["deterministic"],
                         lags=lags, **options)
        assert result.attrs["lags"] == case["lag_order"], case
        assert result.attrs["df_resid"] == case["df_resid"]
        names = [name.replace(":", "_") for name in case["statistic_names"]]
        assert list(result.index) == names
        ours = result.statistic.to_numpy(dtype=float)
        theirs = np.asarray(case["statistics"])
        worst = max(worst, float(np.max(np.abs(ours - theirs) / np.maximum(1.0, np.abs(theirs)))))
    assert worst < 1e-9
    assert len(UROOT["cases"]) == 128 and UROOT["meta"]["uroot"] == "2.1-3"


def test_simulated_tables_are_the_committed_simulation():
    assert SIMULATED["meta"]["seed"] == 19900101
    for seasons in (4, 12):
        block = SIMULATED[str(seasons)]
        assert tuple(float(n) for n in block["nodes"]) == hegy_tables.NODES[seasons]
        for case, cells in block["cases"].items():
            for name in ("t_1", "t_2", "pair", "F_seasonal", "F_all"):
                rows = hegy_tables.CRITICAL[seasons][case][name]
                for row, node in zip(rows, block["nodes"]):
                    expected = tuple(round(cells[str(node)][name][lv]["value"], 3) for lv in LEVELS)
                    assert row == expected
                    tail_ok = (row[0] < row[1] < row[2] < row[3]) if name.startswith("t") \
                        else (row[0] > row[1] > row[2] > row[3])
                    assert tail_ok


def test_monthly_pairs_share_the_pooled_distribution():
    for case, cells in SIMULATED["12"]["cases"].items():
        for node, cell in cells.items():
            pooled = cell["pair"]["5%"]
            for name, pair in cell["pairs"].items():
                gap = abs(pair["5%"]["value"] - pooled["value"])
                assert gap <= 4.5 * pair["5%"]["mc_se"], (case, node, name)


def test_critical_values_interpolate_in_reciprocal_sample_size():
    nodes = hegy_tables.NODES[4]
    rows = hegy_tables.CRITICAL[4]["seasonal"]["pair"]
    n = 120
    weight = (1 / 80 - 1 / n) / (1 / 80 - 1 / 160)
    expected = [a + weight * (b - a) for a, b in zip(rows[1], rows[2])]
    assert list(_critical(4, "seasonal", "pair", n).values()) == pytest.approx(expected)
    assert list(_critical(4, "seasonal", "pair", 10).values()) == list(rows[0])
    assert list(_critical(4, "seasonal", "pair", 5000).values()) == list(rows[-1])
    assert nodes[0] == 40.0
    frame = data(n=124)
    result = oe.hegy(frame, "y", deterministic="seasonal")
    assert result.attrs["nobs"] == 120
    assert result.attrs["critical_values"]["F_3_4"] == pytest.approx(dict(zip(LEVELS, expected)))
    short = oe.hegy(data(n=30), "y")
    assert any("below the smallest simulated sample" in note for note in short.attrs["notes"])


def test_rejection_tails_and_decisions():
    stationary = oe.hegy(data(n=200, kind="stationary"), "y", deterministic="seasonal")
    walk = oe.hegy(data(n=200), "y", deterministic="seasonal")
    for result in (stationary, walk):
        for name in result.index:
            tail = result.loc[name, "tail"]
            statistic = result.loc[name, "statistic"]
            decision = result.attrs["reject"][name]
            for level, cv in result.attrs["critical_values"][name].items():
                assert decision[level] == (statistic < cv if tail == "lower" else statistic > cv)
        assert result.attrs["p_value"] is None and result.p_value.isna().all()
    # Seasonal unit roots are rejected for the stationary series at 1%; the seasonal random
    # walk's F statistics are far smaller.
    assert stationary.attrs["reject"]["F_2_4"]["1%"] and stationary.attrs["reject"]["F_1_4"]["1%"]
    assert walk.loc["F_2_4", "statistic"] < stationary.loc["F_2_4", "statistic"] / 4
    assert list(stationary["frequency"]) == ["0", "pi", "+-pi/2", "all seasonal", "all"]
    monthly = oe.hegy(data(seasons=12, n=240), "y", seasons=12, deterministic="constant")
    assert list(monthly.index) == ["t_1", "t_2", "F_3_4", "F_5_6", "F_7_8", "F_9_10", "F_11_12",
                                   "F_2_12", "F_1_12"]
    assert list(monthly["frequency"])[2:7] == ["+-pi/6", "+-pi/3", "+-pi/2", "+-2pi/3", "+-5pi/6"]


def test_published_sources_that_did_not_replicate_are_documented():
    codes = {"100": "constant", "101": "seasonal", "110": "trend", "111": "seasonal_trend",
             "000": "none"}
    names = {"zero": "t_1", "pi": "t_2", "pair": "pair"}

    def gaps(key, seasons):
        out = []
        for block in UROOT[key]:
            for stat in block["values"]:
                for row in stat["rows"]:
                    ours = _critical(seasons, codes[block["code"]], names[stat["statistic"]],
                                     row["T"] - seasons)
                    out.append(max(abs(v - ours[lv]) for v, lv in zip(row["values"], LEVELS)))
        return out

    # HEGY (1990) as stored in uroot: t(pi_1) with seasonal dummies at T = 48 is 0.3-0.5
    # more negative than the simulation (and uroot's own hegy.test agrees with the simulation).
    hegy_1990 = gaps("hegy_1990", 4)
    assert max(hegy_1990) > 0.3
    # Beaulieu and Miron (1993) monthly tables agree with the simulation within 0.2.
    assert max(gaps("beaulieu_miron_1993", 12)) < 0.2
    # Diaz-Emparanza (2014) response surfaces as ported in uroot: size distortion at 5% ...
    size = UROOT["response_surface_size"]
    assert size["statistic_names"][2] == "F_3:4" and size["T"] == 100
    assert size["rejection_5pct"][2] > 0.07 and size["rejection_5pct"][3] > 0.07
    # ... and response-surface quantiles far from the simulation at its own grid.
    far = 0
    for row in UROOT["response_surface"]:
        for name in ("t_1", "t_2", "pair", "F_seasonal", "F_all"):
            cell = SIMULATED[str(row["seasons"])]["cases"][row["deterministic"]][
                str(row["nobs"])][name]
            far += any(abs(value - cell[level]["value"]) > 5 * cell[level]["mc_se"]
                       for value, level in zip(row[name], ("1%", "5%", "10%")))
    assert len(UROOT["response_surface"]) * 5 == 180 and far >= 80
    # uroot's own hegy.test under the null (3,000 replications) reproduces the simulation,
    # not the HEGY (1990) table.
    names = ("t_1", "t_2", "pair", "F_seasonal", "F_all")
    for key in ("response_surface_size", "uroot_null_t48"):
        run = UROOT[key]
        for name, quantiles in zip(names, run["quantiles"]):
            ours = _critical(4, "seasonal", name, run["T"] - 4)
            limit = 0.12 if name.startswith("t") else 0.35
            for level in ("5%", "10%"):
                assert abs(quantiles[LEVELS.index(level)] - ours[level]) < limit, (key, name)
    table_48 = next(stat for block in UROOT["hegy_1990"] if block["code"] == "101"
                    for stat in block["values"] if stat["statistic"] == "zero")["rows"][0]
    assert table_48["T"] == 48
    assert table_48["values"][2] - UROOT["uroot_null_t48"]["quantiles"][0][2] < -0.25


def test_invariance_and_rendering():
    frame = data(n=100)
    base = oe.hegy(frame, "y", time="t", deterministic="seasonal_trend", lags=1)
    for scale, shift in [(1e-40, 0.0), (-5.0, 1e8), (1e40, 0.0)]:
        moved = frame.assign(y=frame.y * scale + shift).sample(frac=1.0, random_state=2)
        other = oe.hegy(moved, "y", time="t", deterministic="seasonal_trend", lags=1)
        # Target and regressors scale together: every statistic is unchanged.
        np.testing.assert_allclose(other.statistic.to_numpy(dtype=float),
                                   base.statistic.to_numpy(dtype=float), rtol=1e-6)
    json.dumps(base.attrs, allow_nan=False)
    assert "\\begin{tabular}" in base.to_latex()
    assert "F_3_4" in str(base)


def test_failure_contract():
    frame = data()
    cases = [
        (dict(data=frame.assign(y=frame.y.where(frame.index != 9))), "missing_values"),
        (dict(data=frame.iloc[:12]), "insufficient_observations"),
        (dict(data=frame.assign(y=2.0)), "constant_series"),
        (dict(data=frame, seasons=6), "invalid_option"),
        (dict(data=frame, seasons=True), "invalid_option"),
        (dict(data=frame, deterministic="dummies"), "invalid_option"),
        (dict(data=frame, lags=-1), "invalid_lags"),
        (dict(data=frame, lags="hq"), "invalid_option"),
        (dict(data=frame, lags=1, maxlag=3), "invalid_option"),
        (dict(data=frame, lags="aic", maxlag=60), "invalid_lags"),
        (dict(data=frame, lags=50), "insufficient_observations"),
        (dict(data=frame.drop(index=[20]), time="t"), "time_gaps"),
    ]
    for kwargs, code in cases:
        with pytest.raises(AnalysisError) as error:
            oe.hegy(y="y", **kwargs)
        assert error.value.code == code, kwargs
    seasonal_pattern = pd.DataFrame({"y": np.resize([1.0, 4.0, -2.0, 0.5], 64)})
    with pytest.raises(AnalysisError) as error:
        oe.hegy(seasonal_pattern, "y", deterministic="seasonal")
    assert error.value.code in {"perfect_fit", "collinear_regressors"}
