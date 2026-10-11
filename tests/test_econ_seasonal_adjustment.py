"""Native seasonal adjustment (STL, X-11 method) against independent references.

References (development-only; never runtime adapters):
- tests/fixtures/seasonal/r_stl.json: R 4.6 stats::stl and forecast::forecast.stl
  (scripts/generate_seasonal_oracles.R);
- statsmodels STL (repository dev dependency), live;
- a test-only Python port of R's netlib partial sort ``psort`` that explains the one
  robust R case that differs from the exact median used here and in statsmodels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError

torch.set_num_threads(1)
FIXTURES = Path(__file__).parent / "fixtures" / "seasonal"
R_STL = json.loads((FIXTURES / "r_stl.json").read_text())
CASES = {case["name"]: case for case in R_STL["stl"]}


def _r_arguments(case):
    """Only the arguments the R call passed explicitly (R defaults are derived natively)."""
    call = case["call"]
    keyword = {"s.window": "seasonal", "s.degree": "seasonal_degree", "t.window": "trend",
               "l.window": "low_pass", "s.jump": "seasonal_jump", "t.jump": "trend_jump",
               "l.jump": "low_pass_jump", "robust": "robust", "inner": "inner", "outer": "outer"}
    out = {}
    for r_name, name in keyword.items():
        marker = f"{r_name} = "
        if marker in call:
            raw = call.split(marker, 1)[1].split(",", 1)[0].rstrip(")")
            out[name] = (raw.strip('"') if raw.startswith('"') else
                         raw == "TRUE" if raw in ("TRUE", "FALSE") else int(raw))
    return out


def _r_psort(values, ind):
    """Port of R's netlib ``psort`` (src/library/stats/src/stl.c), oracle only."""
    a = [None] + list(values)
    ind = [None] + list(ind)
    il, iu, indl, indu = [0] * 16, [0] * 16, [0] * 16, [0] * 16
    jl, ju, i, j, m = 1, len(ind) - 1, 1, len(values), 0
    indl[0], indu[0] = 1, len(ind) - 1
    state = "outer"
    while True:
        if state == "outer":
            state = "partition" if i < j else "pop"
        elif state == "pop":
            m -= 1
            if m < 0:
                return a[1:]
            i, j, jl, ju = il[m], iu[m], indl[m], indu[m]
            state = "check" if jl <= ju else "pop"
        elif state == "check":
            state = "partition" if j - i > 10 else "insert"
        elif state == "partition":
            k, lo, ij = i, j, (i + j) // 2
            t = a[ij]
            if a[i] > t:
                a[ij], a[i], t = a[i], t, a[i]
            if a[j] < t:
                a[ij], a[j], t = a[j], t, a[j]
                if a[i] > t:
                    a[ij], a[i], t = a[i], t, a[i]
            while True:
                lo -= 1
                if a[lo] <= t:
                    tt = a[lo]
                    k += 1
                    while a[k] < t:
                        k += 1
                    if k > lo:
                        break
                    a[lo], a[k] = a[k], tt
            indl[m], indu[m] = jl, ju
            p1, m = m, m + 1
            if lo - i <= j - k:
                il[p1], iu[p1], j = k, j, lo
                while jl <= ju and ind[ju] > j:
                    ju -= 1
                if jl > ju:
                    state = "pop"
                else:
                    indl[p1], state = ju + 1, "check"
            else:
                il[p1], iu[p1], i = i, lo, k
                while jl <= ju and ind[jl] < i:
                    jl += 1
                if jl > ju:
                    state = "pop"
                else:
                    indu[p1], state = jl - 1, "check"
        else:  # insertion sort of a short segment
            if i != 1:
                i -= 1
                while True:
                    i += 1
                    if i == j:
                        break
                    t = a[i + 1]
                    if a[i] > t:
                        k = i
                        while True:
                            a[k + 1] = a[k]
                            k -= 1
                            if t >= a[k]:
                                break
                        a[k + 1] = t
                state = "pop"
            else:
                state = "outer"


def _r_weights(y, fit):
    r = (y - fit).abs().numpy()
    n = len(r)
    mid1 = n // 2 + 1
    mid2 = n - mid1 + 1
    ordered = _r_psort(r, [mid1, mid2])
    cmad = 3.0 * (ordered[mid1 - 1] + ordered[mid2 - 1])
    weights = np.where(r <= 0.001 * cmad, 1.0,
                       np.where(r <= 0.999 * cmad, (1 - (r / cmad) ** 2) ** 2, 0.0))
    return torch.tensor(weights, dtype=torch.float64)


def _components(result):
    return result["components"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_stl_matches_r_stats_stl_with_r_defaults(name, monkeypatch):
    case = CASES[name]
    args = _r_arguments(case)
    result = oe.stl({"y": case["y"]}, "y", period=case["period"], **args)
    settings = result.attrs["settings"]
    # Every default not passed explicitly reproduces R's effective window/degree/jump/iterations.
    assert [settings[k] for k in ("seasonal", "trend", "low_pass")] == [case["win"][k] for k in "stl"]
    assert [settings[f"{k}_degree"] for k in ("seasonal", "trend", "low_pass")] == [case["deg"][k] for k in "stl"]
    assert [settings[f"{k}_jump"] for k in ("seasonal", "trend", "low_pass")] == [case["jump"][k] for k in "stl"]
    assert (settings["inner"], settings["outer"]) == (case["inner"], case["outer"])
    table = _components(result)
    deviation = max(np.abs(table[col].to_numpy() - case[key]).max()
                    for col, key in (("seasonal", "seasonal"), ("trend", "trend"),
                                     ("remainder", "remainder"), ("robustness_weight", "weights")))
    if name == "quarterly_custom":
        # R's psort returns a non-median order statistic in one robustness pass here.
        assert deviation > 1e-3
        import openecon.econometrics.tsworkflows.stl as module

        monkeypatch.setattr(module, "_robustness_weights", _r_weights)
        table = _components(oe.stl({"y": case["y"]}, "y", period=case["period"], **args))
        deviation = max(np.abs(table[col].to_numpy() - case[key]).max()
                        for col, key in (("seasonal", "seasonal"), ("trend", "trend"),
                                         ("robustness_weight", "weights")))
    assert deviation < 1e-9


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("options", [
    {},
    {"seasonal": 13, "trend": 25, "low_pass": 13, "seasonal_deg": 0, "seasonal_jump": 3,
     "trend_jump": 4, "low_pass_jump": 2},
    {"seasonal": 9, "trend_deg": 0, "low_pass_deg": 0},
])
def test_stl_matches_statsmodels_defaults_and_options(robust, options):
    from statsmodels.tsa.seasonal import STL

    y = np.asarray(CASES["monthly_s13_robust"]["y"])
    reference = STL(y, period=12, robust=robust, **options).fit()
    names = {"seasonal_deg": "seasonal_degree", "trend_deg": "trend_degree",
             "low_pass_deg": "low_pass_degree"}
    result = oe.stl({"y": y}, "y", period=12, robust=robust, convention="statsmodels",
                    **{names.get(k, k): v for k, v in options.items()})
    table = _components(result)
    np.testing.assert_allclose(table["seasonal"], reference.seasonal, atol=1e-9, rtol=0)
    np.testing.assert_allclose(table["trend"], reference.trend, atol=1e-9, rtol=0)
    np.testing.assert_allclose(table["remainder"], reference.resid, atol=1e-9, rtol=0)
    np.testing.assert_allclose(table["robustness_weight"], reference.weights, atol=1e-12, rtol=0)
    assert result.attrs["settings"]["inner"] == (2 if robust else 5)


def test_stl_components_reconstruct_and_periodic_is_exactly_periodic():
    case = CASES["monthly_periodic"]
    result = oe.stl({"y": case["y"]}, "y", period=12, seasonal="periodic")
    table = _components(result)
    total = table["trend"] + table["seasonal"] + table["remainder"]
    np.testing.assert_allclose(total, case["y"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(table["seasonally_adjusted"], np.asarray(case["y"]) - table["seasonal"],
                               atol=1e-12, rtol=0)
    seasonal = table["seasonal"].to_numpy()
    np.testing.assert_allclose(seasonal[12:], seasonal[:-12], atol=1e-12, rtol=0)
    assert result.attrs["reconstruction_max_abs_error"] < 1e-12
    assert 0 <= result.attrs["strength_seasonal"] <= 1 and 0 <= result.attrs["strength_trend"] <= 1
    assert result.attrs["sample"]["weights"] == "not supported"
    assert result.attrs["sample"]["nobs"] == len(case["y"])
    assert "no standard errors" in result.attrs["inference"]


def test_stl_calendar_validation_and_row_order():
    y = CASES["monthly_s7"]["y"]
    months = pd.date_range("2010-01-01", periods=len(y), freq="MS")
    frame = pd.DataFrame({"month": months, "y": y})
    reference = oe.stl({"y": y}, "y", period=12, seasonal=7)
    dated = oe.stl(frame.sample(frac=1.0, random_state=3), "y", time="month", seasonal=7)
    assert dated.attrs["settings"]["period"] == 12
    assert dated.attrs["calendar"] == {"kind": "datetime", "frequency": "MS"}
    np.testing.assert_allclose(_components(dated)["trend"], _components(reference)["trend"], atol=0)
    assert list(_components(dated)["period"]) == list(months)
    with pytest.raises(AnalysisError) as error:
        oe.stl(frame, "y", time="month", period=4, seasonal=7)
    assert error.value.code == "period_mismatch"
    quarters = pd.DataFrame({"q": pd.date_range("2000-01-01", periods=40, freq="QS"),
                             "y": np.arange(40.0) % 4 + np.arange(40.0)})
    assert oe.stl(quarters, "y", time="q", seasonal=7).attrs["settings"]["period"] == 4
    irregular = frame.drop(index=[30]).reset_index(drop=True)
    with pytest.raises(AnalysisError) as error:
        oe.stl(irregular, "y", time="month", seasonal=7)
    assert error.value.code == "time_gaps"
    gaps = pd.DataFrame({"t": [*range(1, 31), *range(32, 62)], "y": np.arange(60.0)})
    with pytest.raises(AnalysisError) as error:
        oe.stl(gaps, "y", time="t", period=12, seasonal=7)
    assert error.value.code == "time_gaps"
    with pytest.raises(AnalysisError) as error:
        oe.stl(pd.DataFrame({"t": range(60), "y": np.arange(60.0)}), "y", time="t", seasonal=7)
    assert error.value.code == "missing_period"
    annual = pd.DataFrame({"t": pd.date_range("1950-01-01", periods=60, freq="YS"), "y": np.arange(60.0)})
    with pytest.raises(AnalysisError) as error:
        oe.stl(annual, "y", time="t", period=2, seasonal=7)
    assert error.value.code == "not_seasonal"


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"period": 12}, "invalid_option"),                                  # R has no default s.window
    ({"period": 12, "seasonal": 8}, "invalid_option"),                   # even window
    ({"period": 12, "seasonal": 7, "trend": 2}, "invalid_option"),
    ({"period": 12, "seasonal": 7, "seasonal_degree": 2}, "invalid_option"),
    ({"period": 12, "seasonal": "periodic", "seasonal_degree": 1}, "invalid_option"),
    ({"period": 12, "seasonal": 7, "convention": "x13"}, "invalid_option"),
    ({"period": 70, "seasonal": 7}, "insufficient_observations"),
    ({"period": 1, "seasonal": 7}, "invalid_option"),
    ({"period": 12, "seasonal": 7, "max_work": 10}, "search_budget"),
])
def test_stl_failure_contract(kwargs, code):
    with pytest.raises(AnalysisError) as error:
        oe.stl({"y": CASES["monthly_s7"]["y"]}, "y", **kwargs)
    assert error.value.code == code


def test_stl_rejects_missing_values_and_nonnumeric():
    y = list(CASES["monthly_s7"]["y"])
    y[40] = float("nan")
    with pytest.raises(AnalysisError) as error:
        oe.stl({"y": y}, "y", period=12, seasonal=7)
    assert error.value.code == "missing_values"
    with pytest.raises(AnalysisError):
        oe.stl({"y": ["a"] * 40}, "y", period=12, seasonal=7)


def test_stl_saved_state_round_trip_and_tamper_detection():
    case = CASES["monthly_s13_robust"]
    months = pd.date_range("2001-01-31", periods=len(case["y"]), freq="ME")
    result = oe.stl(pd.DataFrame({"m": months, "y": case["y"]}), "y", time="m", seasonal=13, robust=True)
    text = result.to_json()
    restored = oe.seasonal_restore(text)
    pd.testing.assert_frame_equal(pd.DataFrame(restored["components"]), pd.DataFrame(result["components"]))
    assert restored.attrs["settings"] == result.attrs["settings"]
    assert restored.to_json() == text
    assert oe.seasonal_restore(json.loads(text)).to_json() == text
    record = json.loads(text)
    record["components"]["seasonal"][5] += 1e-3
    with pytest.raises(AnalysisError) as error:
        oe.seasonal_restore(record)
    assert error.value.code == "invalid_state"                           # checksum
    from openecon.econometrics.tsworkflows.ssdiffuse import _digest

    record.pop("sha256")
    record["sha256"] = _digest(record)
    with pytest.raises(AnalysisError) as error:
        oe.seasonal_restore(record)
    assert error.value.code == "invalid_state"                           # replay mismatch
    record = json.loads(text)
    record["settings"]["seasonal"] = 8
    record.pop("sha256")
    record["sha256"] = _digest(record)
    with pytest.raises(AnalysisError) as error:
        oe.seasonal_restore(record)
    assert error.value.code in {"invalid_state", "invalid_option"}
    with pytest.raises(AnalysisError) as error:
        oe.seasonal_restore("{\"schema\": \"other\"}")
    assert error.value.code == "invalid_state"


def test_stl_summary_and_latex_render():
    result = oe.stl({"y": CASES["monthly_s7"]["y"]}, "y", period=12, seasonal=7)
    text = str(result)
    assert "STL decomposition" in text and "components" in text
    assert "\\end{longtable}" in result.to_latex()


@pytest.mark.parametrize("method", ["naive", "rwdrift"])
def test_stl_forecast_matches_forecast_stl(method):
    reference = next(item for item in R_STL["stl_forecast"]["forecasts"] if item["method"] == method)
    case = CASES["monthly_s7"]
    result = oe.stl({"y": case["y"]}, "y", period=12, seasonal=7)
    for alpha, low, high in ((0.2, "lower80", "upper80"), (0.05, "lower95", "upper95")):
        out = oe.stl_forecast(result, reference["h"], method="naive" if method == "naive" else "drift",
                              alpha=alpha)
        np.testing.assert_allclose(out["forecast"], reference["mean"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(out["ci_low"], reference[low], atol=1e-9, rtol=0)
        np.testing.assert_allclose(out["ci_high"], reference[high], atol=1e-9, rtol=0)
    assert list(out["period"]) == list(range(len(case["y"]), len(case["y"]) + reference["h"]))


def test_stl_forecast_reuses_native_ets_and_arima():
    case = CASES["monthly_s7"]
    months = pd.date_range("2010-01-01", periods=len(case["y"]), freq="MS")
    result = oe.stl(pd.DataFrame({"m": months, "y": case["y"]}), "y", time="m", seasonal=7)
    table = _components(result)
    adjusted = pd.DataFrame({"adjusted": table["seasonally_adjusted"].to_numpy()})
    seasonal = table["seasonal"].to_numpy()
    expected_seasonal = np.resize(seasonal[-12:], 14)
    ets_out = oe.stl_forecast(result, 14, method="ets", ets_model="ANN")
    direct = oe.forecast(oe.ets(data=adjusted, y="adjusted", model="ANN"), 14)
    np.testing.assert_allclose(ets_out["forecast"], direct["forecast"].to_numpy() + expected_seasonal,
                               atol=1e-9, rtol=0)
    np.testing.assert_allclose(ets_out["std_error"], direct["std_error"], atol=1e-9, rtol=0)
    arima_out = oe.stl_forecast(result, 14, method="arima", arima_order=(1, 1, 0))
    direct = oe.forecast(oe.arima(data=adjusted, y="adjusted", order=(1, 1, 0)), 14)
    np.testing.assert_allclose(arima_out["forecast"], direct["forecast"].to_numpy() + expected_seasonal,
                               atol=1e-9, rtol=0)
    np.testing.assert_allclose(arima_out["ci_high"] - arima_out["ci_low"],
                               direct["ci_high"] - direct["ci_low"], atol=1e-9, rtol=0)
    assert list(arima_out["period"]) == list(pd.date_range("2020-06-01", periods=14, freq="MS"))
    restored = oe.seasonal_restore(result.to_json())
    pd.testing.assert_frame_equal(pd.DataFrame(oe.stl_forecast(restored, 14)),
                                  pd.DataFrame(oe.stl_forecast(result, 14)))


def test_stl_naive_interval_coverage_in_a_bounded_simulation():
    """Seeded Monte Carlo (120 series): random walk + fixed monthly pattern, n = 480.

    With a long sample the seasonal estimate is accurate and the forecast::rwf
    intervals of the adjusted series attain their nominal coverage; on short samples
    they under-cover because the seasonal component is treated as known (documented).
    """
    rng = np.random.default_rng(20261010)
    pattern = 5 * np.sin(2 * np.pi * np.arange(12) / 12)
    reps, n, h = 120, 480, 6
    hits95, hits80 = np.zeros(h), np.zeros(h)
    for _ in range(reps):
        y = 50 + np.cumsum(rng.normal(0, 1, n + h)) + np.resize(pattern, n + h)
        fit = oe.stl({"y": y[:n]}, "y", period=12, seasonal="periodic")
        for alpha, hits in ((0.05, hits95), (0.2, hits80)):
            out = oe.stl_forecast(fit, h, method="naive", alpha=alpha)
            hits += (y[n:] >= out["ci_low"].to_numpy()) & (y[n:] <= out["ci_high"].to_numpy())
    assert 0.90 <= hits95.mean() / reps <= 0.99
    assert 0.72 <= hits80.mean() / reps <= 0.88


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"steps": 0}, "invalid_steps"),
    ({"steps": 3, "method": "seasonal"}, "invalid_option"),
    ({"steps": 3, "alpha": 1.5}, "invalid_option"),
    ({"steps": 3, "method": "ets", "ets_model": "AAA"}, "invalid_option"),
])
def test_stl_forecast_failure_contract(kwargs, code):
    result = oe.stl({"y": CASES["monthly_s7"]["y"]}, "y", period=12, seasonal=7)
    with pytest.raises(AnalysisError) as error:
        oe.stl_forecast(result, **kwargs)
    assert error.value.code == code
    with pytest.raises(AnalysisError) as error:
        oe.stl_forecast({"components": []}, 3)
    assert error.value.code == "invalid_result"


# ---- X-11 method (native) ------------------------------------------------------------

X11_REFERENCE = json.loads((FIXTURES / "numpy_x11.json").read_text())
X11_CASES = {case["name"]: case for case in X11_REFERENCE["x11"]}


def _x11_options(case):
    options = dict(case["options"])
    if "sigma_limits" in options:
        options["sigma_limits"] = tuple(options["sigma_limits"])
    return options


@pytest.mark.parametrize("name", sorted(X11_CASES))
def test_x11_matches_independent_loop_reference(name):
    case = X11_CASES[name]
    result = oe.x11({"y": case["y"]}, "y", period=case["period"], **_x11_options(case))
    tables = result["tables"]
    reference = case["reference"]
    for key, values in reference.items():
        if key in ("henderson", "final_ic_ratio"):
            continue
        np.testing.assert_allclose(tables[key].to_numpy(dtype=float), np.asarray(values, dtype=float),
                                   atol=1e-9, rtol=0, err_msg=key)
    steps = {step: record["henderson"] for step, record in result.attrs["henderson_steps"].items()}
    assert steps == reference["henderson"]
    assert result.attrs["final_ic_ratio"] == pytest.approx(reference["final_ic_ratio"], abs=1e-10)
    components = result["components"]
    np.testing.assert_allclose(components["seasonal"], reference["d10"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(components["trend"], reference["d12"], atol=1e-9, rtol=0)
    assert result.attrs["reconstruction_max_abs_error"] < 1e-9
    assert result.attrs["method"] == "X-11 method (native)"
    assert "X-13ARIMA-SEATS" in result.attrs["not_x13"]


def test_x11_filter_weights_are_the_published_ones():
    from openecon.econometrics.tsworkflows import x11 as module

    h13 = module.henderson_weights(13).numpy()
    # Symmetric 13-term Henderson weights as tabulated (5 decimals), e.g. NI LabVIEW TSA
    # Moving Average documentation.
    published = [-0.01935, -0.02786, 0.0, 0.06549, 0.14736, 0.21434, 0.24006]
    np.testing.assert_allclose(h13[:7], published, atol=5e-6)
    for length in (5, 7, 9, 13, 23):
        weights = module.henderson_weights(length).numpy()
        lags = np.arange(-(length // 2), length // 2 + 1)
        # Henderson filters reproduce cubic polynomials exactly.
        np.testing.assert_allclose([weights.sum(), weights @ lags, weights @ lags**2, weights @ lags**3],
                                   [1, 0, 0, 0], atol=1e-12)
    for kind in ("3x3", "3x5"):
        symmetric, ends = module.seasonal_filters(kind)
        assert float(symmetric.sum()) == pytest.approx(1.0)
        assert all(float(w.sum()) == pytest.approx(1.0, abs=1e-12) for w in ends)
    ma = module.trend_ma_weights(12).numpy()
    pattern = np.sin(2 * np.pi * np.arange(40) / 12) + 0.3 * np.cos(4 * np.pi * np.arange(40) / 12)
    np.testing.assert_allclose(np.convolve(pattern, ma, mode="valid"), 0.0, atol=1e-12)


@pytest.mark.parametrize(("length", "period"), [(9, 12), (13, 12), (23, 12), (5, 4), (7, 4)])
def test_musgrave_end_weights_solve_the_linear_trend_revision_criterion(length, period):
    """Independent derivation: minimize sum (u - w)^2 + D (sum j (u - w))^2 s.t. sum u = 1."""
    from openecon.econometrics.tsworkflows import x11 as module

    w = module.henderson_weights(length).numpy()
    ratio = module.musgrave_ratio(length, period)
    d = 4.0 / (np.pi * ratio**2)
    m = length // 2
    lags = np.arange(-m, m + 1, dtype=float)
    for k, filt in enumerate(module.musgrave_filters(module.henderson_weights(length), ratio)):
        size = length - 1 - k
        # Unknown u on the first `size` lags; missing u are zero.
        a = np.eye(length)[:, :size]
        a = np.vstack([a, np.sqrt(d) * lags[None, :] @ a])
        target = np.concatenate([w, [np.sqrt(d) * lags @ w]])
        kkt = np.block([[2 * a.T @ a, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
        rhs = np.concatenate([2 * a.T @ target, [1.0]])
        solution = np.linalg.solve(kkt, rhs)[:size]
        np.testing.assert_allclose(filt.numpy(), solution, atol=1e-12)


@pytest.mark.parametrize("mode", ["multiplicative", "additive"])
def test_x11_recovers_a_constant_level_with_a_fixed_seasonal_pattern(mode):
    pattern = np.array([0.9, 0.95, 1.05, 1.1, 1.08, 1.02, 0.97, 0.93, 0.96, 1.01, 1.04, 0.99])
    pattern = pattern / pattern.mean()
    if mode == "multiplicative":
        y, seasonal, irregular = 250.0 * np.tile(pattern, 9), np.tile(pattern, 9), 1.0
    else:
        seasonal = np.tile(40 * (pattern - 1), 9)
        y, irregular = 250.0 + seasonal, 0.0
    result = oe.x11({"y": y}, "y", period=12, mode=mode)
    components = result["components"]
    np.testing.assert_allclose(components["seasonal"], seasonal, atol=1e-12)
    np.testing.assert_allclose(components["trend"], 250.0, atol=1e-10)
    np.testing.assert_allclose(components["irregular"], irregular, atol=1e-12)
    assert result.attrs["extreme_values_c17"] == 0


def test_x11_calendar_first_season_and_restore():
    case = X11_CASES["monthly_multiplicative_start_april"]
    months = pd.date_range("2003-04-01", periods=len(case["y"]), freq="MS")
    frame = pd.DataFrame({"m": months, "y": case["y"]}).sample(frac=1.0, random_state=1)
    dated = oe.x11(frame, "y", time="m")
    assert dated.attrs["settings"]["first_season"] == 4
    plain = oe.x11({"y": case["y"]}, "y", period=12, first_season=4)
    np.testing.assert_allclose(dated["components"]["seasonally_adjusted"],
                               plain["components"]["seasonally_adjusted"], atol=0)
    text = dated.to_json()
    restored = oe.seasonal_restore(text)
    pd.testing.assert_frame_equal(pd.DataFrame(restored["components"]), pd.DataFrame(dated["components"]))
    pd.testing.assert_frame_equal(pd.DataFrame(restored["tables"]), pd.DataFrame(dated["tables"]))
    assert restored.to_json() == text
    from openecon.econometrics.tsworkflows.ssdiffuse import _digest

    record = json.loads(text)
    record["components"]["trend"][3] *= 1.0001
    record.pop("sha256")
    record["sha256"] = _digest(record)
    with pytest.raises(AnalysisError) as error:
        oe.seasonal_restore(record)
    assert error.value.code == "invalid_state"
    quarters = pd.DataFrame({"q": pd.date_range("1990-07-01", periods=58, freq="QS"),
                             "y": X11_CASES["quarterly_multiplicative_auto"]["y"]})
    assert oe.x11(quarters, "y", time="q").attrs["settings"]["first_season"] == 3
    assert "\\end{longtable}" in dated.to_latex()
    assert "X-11 method (native)" in str(dated)


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"period": 7}, "unsupported_period"),
    ({"period": 12, "mode": "log"}, "invalid_option"),
    ({"period": 12, "seasonal_filter": "3x9"}, "invalid_option"),
    ({"period": 12, "henderson": 5}, "invalid_option"),
    ({"period": 12, "henderson": 11}, "invalid_option"),
    ({"period": 12, "sigma_limits": (2.5, 1.5)}, "invalid_option"),
    ({"period": 12, "sigma_limits": (0.4, 1.5)}, "invalid_option"),
    ({"period": 12, "first_season": 13}, "invalid_option"),
])
def test_x11_failure_contract(kwargs, code):
    y = X11_CASES["monthly_smooth_low_ic"]["y"]                          # 96 months
    with pytest.raises(AnalysisError) as error:
        oe.x11({"y": y}, "y", **kwargs)
    assert error.value.code == code


def test_x11_rejects_nonpositive_multiplicative_and_short_or_dated_conflicts():
    y = list(X11_CASES["monthly_smooth_low_ic"]["y"])
    y[10] = -1.0
    with pytest.raises(AnalysisError) as error:
        oe.x11({"y": y}, "y", period=12)
    assert error.value.code == "nonpositive_values"
    assert oe.x11({"y": y}, "y", period=12, mode="additive").attrs["mode"] == "additive"
    with pytest.raises(AnalysisError) as error:
        oe.x11({"y": y[:83]}, "y", period=12, mode="additive")             # 3x5 needs 7 years
    assert error.value.code == "insufficient_observations"
    short = oe.x11({"y": y[:72]}, "y", period=12, mode="additive", seasonal_filter="3x3")
    assert short.attrs["nobs"] == 72                                    # 3x3/3x3 needs 6 years
    with pytest.raises(AnalysisError) as error:                         # initial 3x5 needs 8 years
        oe.x11({"y": y[:95]}, "y", period=12, mode="additive", initial_seasonal_filter="3x5")
    assert error.value.code == "insufficient_observations"
    months = pd.date_range("2000-01-01", periods=96, freq="MS")
    with pytest.raises(AnalysisError) as error:
        oe.x11(pd.DataFrame({"m": months, "y": X11_CASES["monthly_smooth_low_ic"]["y"]}), "y",
               time="m", first_season=2)
    assert error.value.code == "invalid_option"
    days = pd.DataFrame({"d": pd.date_range("2000-01-01", periods=96, freq="D"), "y": np.arange(1.0, 97)})
    with pytest.raises(AnalysisError) as error:
        oe.x11(days, "y", time="d", period=12)
    assert error.value.code == "unsupported_calendar"
