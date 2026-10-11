"""Calendar/intervention regressors, intermittent demand and forecast combination.

References (development-only): brute-force day enumeration with the Python
``datetime`` module, statsmodels SARIMAX (regression with ARMA errors), committed
fixtures from scripts/generate_seasonal_oracles.{R,py} and explicit NumPy loops.
"""

from __future__ import annotations

import datetime as dt
import io
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


# ---- calendar regressors ------------------------------------------------------------


def _brute_force(first: dt.date, months: int):
    """Weekday counts, leap-year value and length of the period by enumerating days."""
    year, month = first.year + (first.month - 1 + months) // 12, (first.month - 1 + months) % 12 + 1
    stop = dt.date(year, month, 1)
    counts = [0] * 7
    day, leap = first, 0.0
    while day < stop:
        counts[day.weekday()] += 1
        if day.month == 2 and day.day == 1:
            leap = 0.75 if (day.year % 4 == 0 and day.year % 100 != 0) or day.year % 400 == 0 else -0.25
        day += dt.timedelta(days=1)
    return counts, leap, (stop - first).days


@pytest.mark.parametrize(("frequency", "start", "periods", "months", "shift"), [
    ("MS", "1995-01-01", 160, 1, 0),
    ("ME", "2094-11-30", 140, 1, 0),           # crosses the non-leap year 2100
    ("QS", "1998-01-01", 70, 3, 0),
    ("QE-DEC", "1999-03-31", 60, 3, -2),
    ("QS-FEB", "2001-02-01", 50, 3, 0),        # fiscal quarters Feb-Apr, ...
])
def test_calendar_regressors_match_day_enumeration(frequency, start, periods, months, shift):
    table = oe.calendar_regressors(start=start, periods=periods, frequency=frequency, horizon=5,
                                   length_of_period=True)
    assert len(table) == periods + 5 and table["future"].sum() == 5
    assert table.attrs["seasons_per_year"] == 12 // months
    expected = pd.date_range(start, periods=periods + 5, freq=frequency)
    assert list(table["period"]) == list(expected)
    for row, stamp in zip(table.itertuples(), expected, strict=True):
        month = stamp.month + shift
        first = dt.date(stamp.year + (month - 1) // 12, (month - 1) % 12 + 1, 1)
        counts, leap, days = _brute_force(first, months)
        assert [row.td_mon, row.td_tue, row.td_wed, row.td_thu, row.td_fri, row.td_sat] == [
            counts[k] - counts[6] for k in range(6)]
        assert row.leap_year == leap
        assert row.length_of_period == pytest.approx(days - 365.25 * months / 12, abs=1e-12)


def test_calendar_weekday_contrast_and_holidays():
    months = pd.date_range("2010-01-01", periods=36, freq="MS")
    holidays = {"easter": ["2010-04-04", "2011-04-24", "2012-04-08", "2013-03-31", "2009-04-12"],
                "fair": ["2011-06-03", "2011-06-04", "2011-06-05"]}
    table = oe.calendar_regressors(months, trading_day="one", leap_year=False, holidays=holidays)
    assert list(table.columns) == ["period", "future", "td_weekday", "holiday_easter", "holiday_fair"]
    for row, stamp in zip(table.itertuples(), months, strict=True):
        counts, _, _ = _brute_force(stamp.date(), 1)
        assert row.td_weekday == sum(counts[:5]) - 2.5 * sum(counts[5:])
    easter = table.set_index("period")["holiday_easter"]
    assert easter[pd.Timestamp("2010-04-01")] == 1 and easter[pd.Timestamp("2012-04-01")] == 1
    assert easter.sum() == 3 and table["holiday_fair"].sum() == 3
    assert table.set_index("period")["holiday_fair"][pd.Timestamp("2011-06-01")] == 3
    assert table.attrs["holidays"]["easter"] == {"listed": 5, "used": 3, "outside_calendar": 2}


def test_calendar_regressors_drive_regarima_like_statsmodels_sarimax():
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    reg = oe.calendar_regressors(start="2001-01-01", periods=132, frequency="MS", horizon=6)
    sample, future = reg[~reg["future"]], reg[reg["future"]]
    names = ["td_mon", "td_tue", "td_wed", "td_thu", "td_fri", "td_sat", "leap_year"]
    rng = np.random.default_rng(162)
    x = sample[names].to_numpy(dtype=float)
    beta = np.array([0.8, 0.5, 0.3, 0.2, 1.1, -0.9, 2.0])
    errors = np.zeros(len(sample))
    shocks = rng.normal(0, 1.0, len(sample))
    for t in range(len(sample)):
        errors[t] = (0.6 * errors[t - 1] if t else 0.0) + shocks[t]
    data = pd.DataFrame({"month": sample["period"].to_numpy(), "y": 50 + x @ beta + errors})
    data = pd.concat([data, sample[names].reset_index(drop=True)], axis=1)
    fit = oe.arima(data=data, y="y", x=names, time="month", order=(1, 0, 0))
    estimates = {c.term: c.estimate for c in fit.coefficients}
    model = SARIMAX(data["y"].to_numpy(), exog=x, order=(1, 0, 0), trend="c")
    # SARIMAX puts the constant in the ARMA equation: c = mean * (1 - phi).
    phi = estimates["ARMA:L1.ar"]
    ours = np.array([estimates["Intercept"] * (1 - phi), *[estimates[n] for n in names], phi,
                     estimates["/sigma"] ** 2])
    # Same exact Gaussian likelihood at the native estimates ...
    assert fit.metrics["log_likelihood"] == pytest.approx(model.loglike(ours), abs=1e-6)
    # ... which is at least the optimum statsmodels' own optimizer reaches.
    reference = model.fit(disp=False, method="lbfgs", maxiter=5000, pgtol=1e-10)
    assert fit.metrics["log_likelihood"] >= reference.llf - 1e-6
    for name, value in zip(names, beta, strict=True):
        assert abs(estimates[name] - value) < 1.0                        # recovers the effects
    out = oe.forecast(fit, 6, exog=future[names].reset_index(drop=True))
    expected = model.filter(ours).get_forecast(6, exog=future[names].to_numpy(dtype=float))
    np.testing.assert_allclose(out["forecast"], expected.predicted_mean, atol=1e-6)
    np.testing.assert_allclose(out["std_error"], np.sqrt(expected.var_pred_mean), atol=1e-6)


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"dates": pd.date_range("2000-01-01", periods=30, freq="W")}, "unsupported_calendar"),
    ({"dates": pd.date_range("2000-01-01", periods=30, freq="YS")}, "unsupported_calendar"),
    ({"dates": list(pd.date_range("2000-01-01", periods=30, freq="MS").delete(7))}, "time_gaps"),
    ({"dates": ["2000-01-01", "x"]}, "invalid_time"),
    ({"start": "2000-01-01", "periods": 24}, "invalid_option"),
    ({"dates": pd.date_range("2000-01-01", periods=30, freq="MS"), "start": "2000-01-01"}, "invalid_option"),
    ({"start": "2000-01-01", "periods": 24, "frequency": "MS", "trading_day": "seven"}, "invalid_option"),
    ({"start": "2000-01-01", "periods": 24, "frequency": "MS", "trading_day": None,
      "leap_year": False}, "invalid_option"),
    ({"start": "2000-01-01", "periods": 24, "frequency": "MS", "holidays": {"bad name": ["2000-01-01"]}},
     "invalid_option"),
    ({"start": "2000-01-01", "periods": 24, "frequency": "MS", "holidays": {"x": ["2000-01-01"] * 2}},
     "invalid_option"),
    ({"start": "2000-01-01", "periods": 24, "frequency": "MS", "horizon": -1}, "invalid_option"),
])
def test_calendar_regressors_failure_contract(kwargs, code):
    with pytest.raises(AnalysisError) as error:
        oe.calendar_regressors(**kwargs)
    assert error.value.code == code


def test_calendar_regressors_json_and_latex():
    table = oe.calendar_regressors(start="2020-01-01", periods=12, frequency="MS")
    assert json.loads(json.dumps(table.attrs))["regressors"][0] == "td_mon"
    assert "\\end{tabular}" in table.to_latex()


# ---- intermittent demand ------------------------------------------------------------

R_CROSTON = json.loads((FIXTURES / "r_croston.json").read_text())["croston"]


def _croston_reference(y, method, alpha, beta):
    """Explicit loop: Croston/SBA/SBJ/TSB with the documented initialization."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    first = int(np.flatnonzero(y > 0)[0]) + 1
    size, interval, prob = y[first - 1], float(first), 1.0 / first
    since = 0
    fitted = np.full(n, np.nan)
    sizes = np.full(n, np.nan)
    for t in range(first + 1, n + 1):
        if method == "tsb":
            fitted[t - 1] = prob * size
        else:
            factor = {"croston": 1.0, "sba": 1 - beta / 2, "sbj": 1 - beta / (2 - beta)}[method]
            fitted[t - 1] = factor * size / interval
        value = y[t - 1]
        since += 1
        if value > 0:
            size += alpha * (value - size)
            if method != "tsb":
                interval += beta * (since - interval)
            since = 0
        if method == "tsb":
            prob += beta * ((value > 0) - prob)
        sizes[t - 1] = size
    sizes[first - 1] = y[first - 1]
    if method == "tsb":
        point = prob * size
    else:
        point = {"croston": 1.0, "sba": 1 - beta / 2, "sbj": 1 - beta / (2 - beta)}[method] * size / interval
    return fitted, sizes, point, first


@pytest.mark.parametrize("case", R_CROSTON, ids=[c["name"] for c in R_CROSTON])
def test_croston_matches_r_forecast_croston_model(case):
    result = oe.croston({"y": case["y"]}, "y", method=case["type"], alpha=case["alpha"], steps=5)
    fitted = result["fitted"]
    first = result.attrs["first_demand_position"]
    reference = np.array([np.nan if v is None else v for v in case["fitted"]], dtype=float)
    np.testing.assert_allclose(fitted["fitted"].to_numpy()[first:], reference[first:], atol=1e-12)
    assert np.isnan(fitted["fitted"].to_numpy()[:first]).all()
    demand = fitted["size_level"].to_numpy()[np.asarray(case["y"]) > 0]
    np.testing.assert_allclose(demand, case["fit_demand"], atol=1e-12)
    interval = fitted["interval_level"].to_numpy()[np.asarray(case["y"]) > 0]
    np.testing.assert_allclose(interval, case["fit_interval"], atol=1e-12)
    factor = {"croston": 1.0, "sba": 1 - case["alpha"] / 2,
              "sbj": 1 - case["alpha"] / (2 - case["alpha"])}[case["type"]]
    own = factor * case["fit_demand"][-1] / case["fit_interval"][-1]
    assert result.attrs["forecast"] == pytest.approx(own, abs=1e-12)
    np.testing.assert_allclose(result["forecast"]["forecast"], own, atol=1e-12)
    if case["y"][-1] > 0:
        # forecast 9.x returns the ratio before the final demand when the series ends with one.
        assert abs(case["forecast"][0] - own) > 0.1
    else:
        np.testing.assert_allclose(result["forecast"]["forecast"], case["forecast"], atol=1e-12)


@pytest.mark.parametrize("method", ["croston", "sba", "sbj", "tsb"])
@pytest.mark.parametrize(("alpha", "beta"), [(0.1, 0.25), (0.6, 0.05), (1.0, 0.3)])
def test_intermittent_methods_match_explicit_loops(method, alpha, beta):
    y = R_CROSTON[4]["y"]
    result = oe.croston({"y": y}, "y", method=method, alpha=alpha, beta=beta, steps=3)
    fitted, sizes, point, first = _croston_reference(y, method, alpha, beta)
    np.testing.assert_allclose(result["fitted"]["fitted"], fitted, atol=1e-12)
    np.testing.assert_allclose(result["fitted"]["size_level"].to_numpy()[first - 1:], sizes[first - 1:],
                               atol=1e-12)
    assert result.attrs["forecast"] == pytest.approx(point, abs=1e-12)
    error = (np.asarray(y) - fitted)[first:]
    assert result.attrs["in_sample_loss"] == pytest.approx(np.mean(error**2), abs=1e-12)


@pytest.mark.parametrize(("method", "beta", "loss"), [("croston", None, "mse"), ("sba", "estimate", "mse"),
                                                       ("tsb", "estimate", "mae"), ("sbj", None, "mae")])
def test_intermittent_estimation_minimizes_the_declared_loss(method, beta, loss):
    y = R_CROSTON[0]["y"]
    result = oe.croston({"y": y}, "y", method=method, alpha="estimate", beta=beta, loss=loss)
    alpha_hat, beta_hat = result.attrs["alpha"], result.attrs["beta"]
    assert 0.01 <= alpha_hat <= 1 and 0.01 <= beta_hat <= 1

    def loss_at(a, b):
        fitted, _, _, first = _croston_reference(y, method, a, b)
        error = (np.asarray(y) - fitted)[first:]
        return np.mean(error**2) if loss == "mse" else np.mean(np.abs(error))

    assert result.attrs["in_sample_loss"] == pytest.approx(loss_at(alpha_hat, beta_hat), abs=1e-10)
    fine = np.arange(0.01, 1.0001, 0.01)
    if beta is None:
        brute = min(loss_at(a, a) for a in fine)
        assert alpha_hat == beta_hat and result.attrs["estimated"] == ["alpha"]
    else:
        brute = min(loss_at(a, b) for a in fine[::4] for b in fine[::4])
        assert result.attrs["estimated"] == ["alpha", "beta"]
    # Golden-section tolerance 1e-7 in the constants; the brute-force grid is coarser.
    assert result.attrs["in_sample_loss"] <= brute + 1e-6


def test_croston_calendar_json_and_failures():
    y = R_CROSTON[0]["y"]
    weeks = pd.date_range("2024-01-07", periods=len(y), freq="W-SUN")
    frame = pd.DataFrame({"week": weeks, "units": y}).sample(frac=1.0, random_state=4)
    result = oe.croston(frame, "units", time="week", steps=4, method="sba", alpha=0.1)
    assert list(result["forecast"]["period"]) == list(pd.date_range(weeks[-1], periods=5, freq="W-SUN")[1:])
    reference = oe.croston({"units": y}, "units", steps=4, method="sba", alpha=0.1)
    np.testing.assert_allclose(result["fitted"]["fitted"], reference["fitted"]["fitted"], atol=0)
    restored = pd.read_json(io.StringIO(result["fitted"].to_json(orient="split")), orient="split")
    np.testing.assert_allclose(restored["fitted"].to_numpy(dtype=float), result["fitted"]["fitted"], atol=1e-12)
    assert json.loads(json.dumps(result.attrs))["method"] == "sba"
    assert "SBA" in str(result) and "\\end{" in result.to_latex()
    cases = [
        ({"y": [0, 1, -2, 3]}, {}, "negative_values"),
        ({"y": [0, 0, 5, 0]}, {}, "insufficient_demands"),
        ({"y": [0, 1, float("nan"), 3]}, {}, "missing_values"),
        ({"y": y}, {"alpha": 0}, "invalid_option"),
        ({"y": y}, {"alpha": 1.5}, "invalid_option"),
        ({"y": y}, {"beta": "free"}, "invalid_option"),
        ({"y": y}, {"method": "ses"}, "invalid_option"),
        ({"y": y}, {"loss": "mape"}, "invalid_option"),
        ({"y": y}, {"steps": 0}, "invalid_option"),
        ({"y": [0, 0, 2, 0, 3]}, {"time": None}, None),
    ]
    for data, options, code in cases:
        if code is None:
            assert oe.croston(data, "y", **options).attrs["demands"] == 2
            continue
        with pytest.raises(AnalysisError) as error:
            oe.croston(data, "y", **options)
        assert error.value.code == code


# ---- forecast combination -----------------------------------------------------------


def _history(seed=7, rows=40):
    rng = np.random.default_rng(seed)
    actual = rng.normal(100, 5, rows)
    common = rng.normal(0, 1, rows)
    return pd.DataFrame({"actual": actual,
                         "arima": actual + 1.0 * common + rng.normal(0, 1.0, rows),
                         "ets": actual + 0.5 * common + rng.normal(0, 2.0, rows),
                         "naive": actual + rng.normal(0.5, 3.0, rows)})


def _components():
    periods = list(range(121, 127))
    return {"arima": pd.DataFrame({"period": periods, "forecast": [101.0, 102, 103, 104, 105, 106]}),
            "ets": pd.DataFrame({"period": periods, "forecast": [100.0, 100.5, 101, 101.5, 102, 102.5]}),
            "naive": pd.DataFrame({"period": periods, "forecast": [99.0] * 6})}


def test_forecast_combination_weights_match_closed_forms():
    forecasts, history = _components(), _history()
    f = np.column_stack([forecasts[n]["forecast"] for n in ("arima", "ets", "naive")])
    equal = oe.forecast_combination(forecasts)
    np.testing.assert_allclose(equal["forecast"], f.mean(1), atol=1e-12)
    errors = history[["arima", "ets", "naive"]].to_numpy() - history[["actual"]].to_numpy()
    mse = (errors**2).mean(0)
    inverse = oe.forecast_combination(forecasts, method="inverse_mse", history=history)
    w = (1 / mse) / (1 / mse).sum()
    np.testing.assert_allclose(list(inverse.attrs["weights"].values()), w, atol=1e-12)
    np.testing.assert_allclose(inverse["forecast"], f @ w, atol=1e-12)
    bg = oe.forecast_combination(forecasts, method="bates_granger", history=history)
    s = errors.T @ errors / len(errors)
    solved = np.linalg.solve(s, np.ones(3))
    w = solved / solved.sum()
    np.testing.assert_allclose(list(bg.attrs["weights"].values()), w, atol=1e-12)
    np.testing.assert_allclose(bg["forecast"], f @ w, atol=1e-10)
    assert sum(bg.attrs["weights"].values()) == pytest.approx(1.0, abs=1e-12)
    assert bg.attrs["history_combined_mse"] == pytest.approx(w @ s @ w, abs=1e-10)
    assert bg.attrs["history_combined_mse"] <= min(bg.attrs["history_mse"].values()) + 1e-12
    np.testing.assert_allclose(bg["forecast_ets"], forecasts["ets"]["forecast"], atol=0)
    assert json.loads(json.dumps(bg.attrs))["method"] == "bates_granger"


def test_forecast_combination_of_native_forecasts_requires_matching_periods():
    case = json.loads((FIXTURES / "r_stl.json").read_text())["stl"][0]
    frame = pd.DataFrame({"t": range(1, len(case["y"]) + 1), "y": case["y"]})
    stl = oe.stl(frame, "y", time="t", period=12, seasonal=7)
    arima = oe.arima(data=frame, y="y", time="t", order=(1, 1, 0), seasonal=(0, 1, 1), period=12)
    first, second = oe.stl_forecast(stl, 6, method="drift"), oe.forecast(arima, 6)
    assert list(first["period"]) == list(second["period"]) == list(range(126, 132))
    combined = oe.forecast_combination({"stl": first, "sarima": second}, origins={"stl": 125, "sarima": 125})
    np.testing.assert_allclose(combined["forecast"], (first["forecast"] + second["forecast"]) / 2, atol=1e-12)
    steps = oe.forecast(oe.arima(data=frame[["y"]], y="y", order=(0, 1, 1)), 6)     # step labels 1..6
    assert list(steps["period"]) == list(range(1, 7))
    with pytest.raises(AnalysisError) as error:
        oe.forecast_combination({"stl": first, "arima_steps": steps})
    assert error.value.code == "mismatched_forecasts"


@pytest.mark.parametrize(("change", "code"), [
    (lambda f, h: ({"arima": f["arima"]}, {}), "invalid_forecasts"),
    (lambda f, h: ({**f, "ets": f["ets"].iloc[:5]}, {}), "mismatched_forecasts"),
    (lambda f, h: ({**f, "ets": f["ets"].assign(period=f["ets"]["period"] + 1)}, {}), "mismatched_forecasts"),
    (lambda f, h: ({**f, "ets": f["ets"].assign(forecast=np.nan)}, {}), "missing_values"),
    (lambda f, h: (f, {"origins": {"arima": 120, "ets": 120, "naive": 119}}), "mismatched_origins"),
    (lambda f, h: (f, {"origins": {"arima": 120}}), "mismatched_origins"),
    (lambda f, h: (f, {"method": "median"}), "invalid_option"),
    (lambda f, h: (f, {"method": "inverse_mse"}), "missing_history"),
    (lambda f, h: (f, {"method": "equal", "history": h}), "invalid_option"),
    (lambda f, h: (f, {"method": "bates_granger", "history": h.iloc[:3]}), "insufficient_history"),
    (lambda f, h: (f, {"method": "bates_granger", "history": h.assign(naive=h["arima"])}),
     "singular_error_covariance"),
    (lambda f, h: (f, {"method": "inverse_mse", "history": h.drop(columns="ets")}), "missing_columns"),
    (lambda f, h: (f, {"method": "inverse_mse", "history": h.assign(ets=h["actual"])}), "degenerate_history"),
    (lambda f, h: (f, {"method": "inverse_mse", "history": h.assign(ets=np.nan)}), "missing_values"),
])
def test_forecast_combination_failure_contract(change, code):
    forecasts, options = change(_components(), _history())
    with pytest.raises(AnalysisError) as error:
        oe.forecast_combination(forecasts, **options)
    assert error.value.code == code


# ---- intervention regressors ---------------------------------------------------------


def _event_reference(total, at, kind, end=None, decay=None):
    out = np.zeros(total)
    for t in range(total):
        lag = t - at
        if kind == "pulse":
            out[t] = (1.0 if lag == 0 else 0.0) if decay is None else (decay**lag if lag >= 0 else 0.0)
        elif kind == "step":
            out[t] = 0.0 if lag < 0 else (1.0 if decay is None else sum(decay**j for j in range(lag + 1)))
        elif end is None:
            out[t] = max(0, lag)
        else:
            out[t] = 0.0 if t <= at else (1.0 if t >= end else (t - at) / (end - at))
    return out


def test_intervention_regressors_match_explicit_loops():
    months = pd.date_range("2015-01-01", periods=48, freq="MS")
    events = [{"name": "ao", "type": "pulse", "at": "2016-03-01"},
              {"name": "tc", "type": "pulse", "at": "2016-03-01", "decay": 0.7},
              {"name": "ls", "type": "step", "at": "2017-01-01"},
              {"name": "gradual", "type": "step", "at": "2017-01-01", "decay": 0.5},
              {"name": "rp", "type": "ramp", "at": "2015-06-01", "end": "2016-02-01"},
              {"name": "slope", "type": "ramp", "at": "2018-06-01"},
              {"name": "late", "type": "step", "at": "2019-02-01"}]           # in the horizon
    table = oe.intervention_regressors(months, events, horizon=3)
    assert len(table) == 51 and table["future"].sum() == 3
    positions = {"ao": 14, "tc": 14, "ls": 24, "gradual": 24, "rp": 5, "slope": 41, "late": 49}
    for event in events:
        expected = _event_reference(51, positions[event["name"]], event["type"],
                                    end=13 if event["name"] == "rp" else None, decay=event.get("decay"))
        np.testing.assert_allclose(table[event["name"]], expected, atol=1e-14, err_msg=event["name"])
    assert [e["in_sample"] for e in table.attrs["events"]] == [True] * 6 + [False]
    numbered = oe.intervention_regressors(range(1990, 2030), [{"name": "s", "type": "step", "at": 2001}],
                                          horizon=2)
    assert list(numbered["period"])[-1] == 2031 and numbered["s"].sum() == 31
    rows = oe.intervention_regressors(30, [{"name": "p", "type": "pulse", "at": 7}])
    assert rows["p"].tolist() == [1.0 if i == 7 else 0.0 for i in range(30)]


def test_level_shift_regarima_matches_sarimax_likelihood():
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    reg = oe.intervention_regressors(range(1, 101), [
        {"name": "shift", "type": "step", "at": 60},
        {"name": "spike", "type": "pulse", "at": 30, "decay": 0.6}], horizon=4)
    sample, future = reg[~reg["future"]], reg[reg["future"]]
    rng = np.random.default_rng(6)
    errors = np.zeros(100)
    for t in range(100):
        errors[t] = (0.5 * errors[t - 1] if t else 0.0) + rng.normal()
    x = sample[["shift", "spike"]].to_numpy(dtype=float)
    data = pd.DataFrame({"t": range(1, 101), "y": 10 + x @ np.array([4.0, -6.0]) + errors,
                         "shift": x[:, 0], "spike": x[:, 1]})
    fit = oe.arima(data=data, y="y", x=["shift", "spike"], time="t", order=(1, 0, 0))
    est = {c.term: c.estimate for c in fit.coefficients}
    assert abs(est["shift"] - 4.0) < 1.0 and abs(est["spike"] + 6.0) < 1.5
    phi = est["ARMA:L1.ar"]
    model = SARIMAX(data["y"].to_numpy(), exog=x, order=(1, 0, 0), trend="c")
    params = np.array([est["Intercept"] * (1 - phi), est["shift"], est["spike"], phi, est["/sigma"] ** 2])
    assert fit.metrics["log_likelihood"] == pytest.approx(model.loglike(params), abs=1e-6)
    assert fit.covariance_matrix is not None and len(fit.covariance_matrix) == 5
    out = oe.forecast(fit, 4, exog=future[["shift", "spike"]].reset_index(drop=True))
    expected = model.filter(params).get_forecast(4, exog=future[["shift", "spike"]].to_numpy(dtype=float))
    np.testing.assert_allclose(out["forecast"], expected.predicted_mean, atol=1e-6)
    assert list(future["period"]) == [101, 102, 103, 104]


@pytest.mark.parametrize(("periods", "events", "code"), [
    (24, [], "invalid_option"),
    (24, [{"name": "a", "type": "spike", "at": 3}], "invalid_option"),
    (24, [{"name": "a", "type": "step", "at": 30}], "invalid_option"),
    (24, [{"name": "a", "type": "step", "at": 3, "decay": 1.0}], "unbounded_transfer"),
    (24, [{"name": "a", "type": "pulse", "at": 3, "decay": -0.2}], "unbounded_transfer"),
    (24, [{"name": "a", "type": "ramp", "at": 3, "decay": 0.5}], "unbounded_transfer"),
    (24, [{"name": "a", "type": "ramp", "at": 5, "end": 5}], "invalid_option"),
    (24, [{"name": "a", "type": "step", "at": 5, "end": 9}], "invalid_option"),
    (24, [{"name": "a", "type": "step", "at": 5}, {"name": "a", "type": "pulse", "at": 2}], "invalid_option"),
    (24, [{"name": "period", "type": "step", "at": 5}], "invalid_option"),
    (24, [{"name": "a", "type": "step", "at": 5, "size": 2}], "invalid_option"),
    ([1, 2, 4, 5], [{"name": "a", "type": "step", "at": 2}], "time_gaps"),
    (["a", "b", "c"], [{"name": "a", "type": "step", "at": "b"}], "invalid_time"),
])
def test_intervention_regressors_failure_contract(periods, events, code):
    with pytest.raises(AnalysisError) as error:
        oe.intervention_regressors(periods, events)
    assert error.value.code == code


# ---- rolling-origin backtests ---------------------------------------------------------

STL_SERIES = json.loads((FIXTURES / "r_stl.json").read_text())["stl"][0]["y"]
BACKTEST_MODELS = {"stl_drift": {"model": "stl", "forecast": "drift", "seasonal": 7},
                   "stl_periodic": {"model": "stl", "forecast": "naive", "seasonal": "periodic"},
                   "sba": {"model": "croston", "variant": "sba", "alpha": "estimate"}}


def test_backtest_refits_within_each_origin_without_lookahead():
    months = pd.date_range("2010-01-01", periods=len(STL_SERIES), freq="MS")
    frame = pd.DataFrame({"m": months, "y": STL_SERIES})
    result = oe.forecast_backtest(frame, "y", time="m", models=BACKTEST_MODELS, steps=3,
                                  minimum=96, step=7)
    assert result.attrs["period"] == 12 and result.attrs["origins"] == 5
    assert result.attrs["failures"] == []
    for origin in (96, 117):
        train = {"y": STL_SERIES[:origin]}
        rows = result[result["origin"] == months[origin - 1]]
        horizon = min(3, len(STL_SERIES) - origin)
        assert list(rows["period"]) == list(months[origin:origin + horizon])
        np.testing.assert_allclose(rows["actual"], STL_SERIES[origin:origin + horizon], atol=0)
        expected = oe.stl_forecast(oe.stl(train, "y", period=12, seasonal=7), horizon, method="drift")
        np.testing.assert_allclose(rows["stl_drift"], expected["forecast"], atol=1e-12)
        expected = oe.croston(train, "y", steps=horizon, method="sba", alpha="estimate")
        np.testing.assert_allclose(rows["sba"], expected["forecast"]["forecast"], atol=1e-12)
    # Changing observations after an origin cannot change that origin's forecasts.
    altered = frame.copy()
    altered.loc[103:, "y"] = altered.loc[103:, "y"] * 3.0 + 50.0
    again = oe.forecast_backtest(altered, "y", time="m", models=BACKTEST_MODELS, steps=3,
                                 minimum=96, step=7)
    early = result["origin"] <= months[102]
    pd.testing.assert_frame_equal(pd.DataFrame(again[early][list(BACKTEST_MODELS)]),
                                  pd.DataFrame(result[early][list(BACKTEST_MODELS)]))
    assert not np.allclose(again[~early]["stl_drift"], result[~early]["stl_drift"])
    history = result[result["horizon"] == 1][["actual", "stl_drift", "stl_periodic"]]
    combo = oe.forecast_combination(
        {"stl_drift": oe.stl_forecast(oe.stl(frame, "y", time="m", seasonal=7), 2, method="drift"),
         "stl_periodic": oe.stl_forecast(oe.stl(frame, "y", time="m", seasonal="periodic"), 2)},
        method="inverse_mse", history=history)
    assert set(combo.attrs["weights"]) == {"stl_drift", "stl_periodic"}
    accuracy = result.attrs["accuracy"]["stl_drift"]
    assert set(accuracy) == {"1", "2", "3"} and accuracy["1"]["count"] == 5      # JSON-safe keys
    assert json.loads(json.dumps(result.attrs))["origins"] == 5


def test_backtest_records_failures_explicitly():
    demand = [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 1, 0, 5, 0]
    result = oe.forecast_backtest({"y": demand}, "y", models={"c": {"model": "croston", "alpha": 0.2}},
                                  steps=2, minimum=6)
    failed = {item["origin"] for item in result.attrs["failures"]}
    assert failed == set(range(5, 10)) and {i["code"] for i in result.attrs["failures"]} == {"insufficient_demands"}
    assert result[result["origin"].isin(failed)]["c"].isna().all()
    assert result[~result["origin"].isin(failed)]["c"].notna().all()


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"models": {}}, "invalid_option"),
    ({"models": {"m": {"model": "prophet"}}}, "invalid_option"),
    ({"models": {"m": {"model": "stl", "seasonal": 7, "window": 3}}}, "invalid_option"),
    ({"models": {"actual": {"model": "croston"}}}, "invalid_option"),
    ({"models": {"m": {"model": "stl", "seasonal": 7}}}, "missing_period"),
    ({"models": {"m": {"model": "croston"}}, "max_origins": 3}, "search_budget"),
    ({"models": {"m": {"model": "croston"}}, "steps": 0}, "invalid_option"),
])
def test_backtest_failure_contract(kwargs, code):
    options = {"steps": 2, "minimum": 60, **kwargs}
    with pytest.raises(AnalysisError) as error:
        oe.forecast_backtest({"y": STL_SERIES}, "y", **options)
    assert error.value.code == code
