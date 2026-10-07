"""Adversarial inputs and regression tests for the arima family (verify stage).

Every invalid input or numerical failure must raise ``AnalysisError`` with a code and a
message that says what to change; nothing may leak a raw exception, a NaN or a silently
changed model. The second half pins the defects fixed during verification.
"""

import json
import math
import sys
import subprocess

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.signal import lfilter

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry


def simulate(n, phi=(), theta=(), seed=0, burn=300):
    e = np.random.default_rng(seed).normal(size=n + burn)
    return lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi, float)], e)[burn:]


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def std_errors(result):
    return np.array([c.std_error for c in result.coefficients])


@pytest.fixture(scope="module")
def frame():
    n = 120
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"t": np.arange(n), "x1": rng.normal(size=n),
                         "y": 1.0 + simulate(n, [0.6], [0.3], seed=1)})
    data["g"] = np.resize(["a", "b", "c"], n)
    return data


def code_of(function, *args, **kwargs):
    with pytest.raises(AnalysisError) as error:
        function(*args, **kwargs)
    assert str(error.value)                                   # a message for the user
    return error.value.code


def assert_finite(result):
    """No NaN or infinity anywhere in a persisted result."""
    text = result.model_dump_json()
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text)
    for row in payload["coefficients"]:
        assert all(math.isfinite(row[key]) for key in
                   ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"))
        assert row["std_error"] > 0
    assert all(value is None or math.isfinite(value) for value in payload["metrics"].values())
    assert all(math.isfinite(row["fitted"]) for row in payload["predictions"])


# ---- degenerate samples ------------------------------------------------------------------------


def test_empty_tiny_and_degenerate_samples(frame):
    base = dict(y="y", order=(1, 0, 0))
    assert code_of(oe.arima, data=frame.iloc[:0], **base) == "empty_data"
    assert code_of(oe.arima, data=frame.iloc[:1], **base) == "insufficient_observations"
    assert code_of(oe.arima, data=frame.iloc[:3], **base) == "insufficient_observations"
    assert code_of(oe.arima, data=frame.iloc[:4], y="y", x=["x1"], order=(2, 0, 2)) \
        == "insufficient_observations"
    assert code_of(oe.arima, data=frame.iloc[:5], y="y", order=(1, 2, 0), seasonal=(0, 1, 0),
                   period=4) == "insufficient_observations"            # differencing eats it all
    # The smallest admissible samples give a finite fit.
    for rows, order in ((5, (1, 0, 0)), (8, (1, 0, 1)), (12, (5, 0, 0))):
        fit = oe.arima(data=frame.iloc[:rows], y="y", order=order)
        assert fit.nobs == rows
        assert_finite(fit)
    assert code_of(oe.arima, data=frame.assign(y=2.0), **base) == "constant_outcome"
    assert code_of(oe.arima, data=frame.assign(y=3.0 * frame.t), y="y", order=(1, 1, 0)) \
        == "constant_outcome"                                          # constant after differencing
    assert code_of(oe.arima, data=frame.assign(z=2 * frame.y + 1), y="y", x=["z"],
                   order=(1, 0, 0)) == "perfect_fit"
    assert code_of(oe.arima, data=frame.assign(x1=np.nan), y="y", x=["x1"], order=(1, 0, 0)) \
        == "missing_values"
    assert code_of(oe.arima, data=frame.assign(x1=np.nan), y="y", x=["x1"], order=(1, 0, 0),
                   missing="drop") == "empty_sample"
    assert code_of(oe.arima, data=frame.assign(y=frame.y.where(frame.t != 3, np.inf)), **base) \
        == "non_finite_values"
    assert code_of(oe.arima, data=frame.assign(y=frame.y.astype(str)), **base) \
        == "non_numeric_column"
    assert code_of(oe.arima, data=frame, y="y", x=["g"], order=(1, 0, 0)) == "non_numeric_column"
    assert code_of(oe.arima, data=frame.assign(g="a"), y="y", x=["g"], categorical=["g"],
                   order=(1, 0, 0)) == "constant_predictor"
    assert code_of(oe.arima, data=frame, y="nope", order=(1, 0, 0)) == "missing_columns"
    assert code_of(oe.arima, data=None, **base) == "invalid_data"
    assert code_of(oe.arima, data="text", **base) == "invalid_data"
    # A constant regressor next to the constant, and an all-zero one, are omitted and recorded.
    plain = oe.arima(data=frame, y="y", x=["x1"], order=(1, 0, 0))
    for junk in (3.0, 0.0):
        fit = oe.arima(data=frame.assign(c=junk), y="y", x=["x1", "c"], order=(1, 0, 0))
        assert fit.provenance["omitted_terms"] == ["c"]
        assert_allclose(estimates(fit), estimates(plain), rtol=1e-10)
    # Mappings and row records are tables too.
    by_columns = oe.arima(data={"y": frame.y.tolist()}, **base)
    by_records = oe.arima(data=frame[["y"]].to_dict("records"), **base)
    assert_allclose(estimates(by_columns), estimates(by_records), rtol=1e-13)


def test_time_gaps_duplicates_and_missing_values(frame):
    base = dict(y="y", order=(1, 0, 0), time="t")
    assert code_of(oe.arima, data=frame.drop(index=[50]), **base) == "time_gaps"
    assert code_of(oe.arima, data=pd.concat([frame, frame.iloc[[5]]]), **base) \
        == "repeated_time_values"
    assert code_of(oe.arima, data=frame.assign(t=frame.t + 0.5), **base) == "invalid_time"
    assert code_of(oe.arima, data=frame.assign(t=frame.t * 2), **base) == "time_gaps"
    interior = frame.assign(y=frame.y.where(frame.t != 30))
    assert code_of(oe.arima, data=interior, **base) == "missing_values"
    assert code_of(oe.arima, data=interior, missing="drop", **base) == "time_gaps"
    assert code_of(oe.arima, data=interior, y="y", order=(1, 0, 0), missing="drop") == "time_gaps"
    dated = interior.assign(t=pd.date_range("2000-01-01", periods=len(frame)))
    assert code_of(oe.arima, data=dated, missing="drop", **base) == "time_gaps"
    stamp = frame.t.astype(float)
    assert code_of(oe.arima, data=frame.assign(t=stamp.where(frame.t != 40)), missing="drop",
                   **base) == "time_gaps"
    # Incomplete rows at either end are dropped on request and counted.
    for hole, rows in ((frame.t > 2, 117), (frame.t < 110, 110)):
        fit = oe.arima(data=frame.assign(y=frame.y.where(hole)), missing="drop", **base)
        assert fit.nobs == rows and fit.dropped_rows == len(frame) - rows
        assert any("missing" in warning for warning in fit.warnings)
    edge = oe.arima(data=frame.assign(t=stamp.where(frame.t != 0)), missing="drop", **base)
    assert edge.nobs == len(frame) - 1


def test_invalid_options_never_leak_validation_errors(frame):
    """Regression: oe.arima used to let pydantic's ValidationError escape for these."""
    base = dict(data=frame, y="y")
    spec_mistakes = [
        dict(order=(1, 0, 0), covariance="HC1"), dict(order=(1, 0, 0), covariance="cluster"),
        dict(order=(1, 0, 0), method="mle"), dict(order=(1, 0, 0), alpha=0.0),
        dict(order=(1, 0, 0), alpha=1.0), dict(order=(1, 0, 0), alpha="a"),
        dict(order=(1, 0, 0), missing="skip"), dict(order=(1, 0, 0), max_iterations=0),
        dict(order=(1, 0, 0), max_iterations=2.5), dict(order=(1, 0, 0), ljung_lags=0),
        dict(order=(1, 0, 0), seasonal=(1, 0, 0), period=1),
        dict(order=(1, 0, 0), seasonal=(1, 0, 0), period=2.5),
        dict(order=(1, 0, 0), tolerance="small"), dict(order=(1, 0, 0), tolerance=float("inf")),
        dict(order=(1, 0, 0), constant="yes"), dict(order=(1, 0, 0), x=["y"]),
        dict(order=(1, 0, 0), x=["x1", "x1"]), dict(order=(1, 0, 0), categorical=["g"]),
        dict(order=(1, 0, 0), time=["t"]), dict(order=(1, 0, 0), time="y"),
        dict(order=(1, 0, 0), x="x1"),
    ]
    for options in spec_mistakes:
        assert code_of(oe.arima, **base, **options) == "invalid_spec", options
    assert code_of(oe.arima, data=frame, y=["y"], order=(1, 0, 0)) == "invalid_spec"
    order_mistakes = [
        dict(order=None), dict(order="101"), dict(order=(1.0, 0, 1)), dict(order=(True, 0, 1)),
        dict(order=(-1, 0, 1)), dict(order=(1, 0)), dict(order=(1, 0, 1, 1)), dict(order=7),
        dict(order=(1, 4, 0)), dict(order=(1, 0, 0), seasonal=(1, 0, 0)),
        dict(order=(1, 0, 0), period=4), dict(order=(1, 0, 0), seasonal=(0, 3, 0), period=4),
        dict(order=(1, 0, 0), seasonal="100", period=4),
    ]
    for options in order_mistakes:
        assert code_of(oe.arima, **base, **options) == "invalid_order", options
    assert code_of(oe.arima, **base, order=(1, 0, 0), seasonal=(1, 0, 0), period=200) \
        == "model_too_large"
    assert code_of(oe.arima, **base, order=(1, 0, 1), tolerance=0.0) == "invalid_option"
    assert code_of(oe.arima, **base, order=(1, 0, 1), tolerance=-1.0) == "invalid_option"
    assert code_of(oe.arima, **base, order=(1, 0, 0), ljung_lags=len(frame)) == "invalid_lags"
    assert code_of(oe.arima, **base, order=(2, 0, 2), max_iterations=1) == "nonconvergence"
    # Integer arrays and NumPy integers are integers (regression: they were refused).
    reference = oe.arima(**base, order=(1, 0, 1))
    assert_allclose(estimates(oe.arima(**base, order=np.array([1, 0, 1]))), estimates(reference),
                    rtol=1e-13)
    seasonal = oe.arima(**base, order=[1, 0, 0], seasonal=np.array([1, 0, 0]),
                        period=np.int64(4), ljung_lags=np.int32(8), max_iterations=np.int64(50))
    assert seasonal.spec.options["period"] == 4 and seasonal.tests["ljung_box"]["lags"] == 8
    # Options at their bounds are accepted.
    assert oe.arima(**base, order=(1, 0, 0), ljung_lags=len(frame) - 1).tests["ljung_box"][
        "df"] == len(frame) - 1
    assert oe.arima(**base, order=(1, 0, 0), alpha=0.5).inference["confidence_level"] == 0.5
    assert oe.arima(**base, order=(0, 3, 0), seasonal=(0, 2, 0), period=2).dropped_rows == 7
    assert oe.arima(**base, order=(1, 0, 0), seasonal=(0, 0, 1), period=2).nobs == len(frame)
    assert oe.arima(**base, order=(1, 0, 1), tolerance=1).nobs == len(frame)


def test_time_column_must_be_numeric_or_datetime(frame):
    """Regression: a text time column was reported as 'declare a categorical predictor'."""
    text = frame.assign(t=frame.t.astype(str))
    flags = frame.assign(t=frame.t > 5)
    for data in (text, flags):
        assert code_of(oe.arima, data=data, y="y", order=(1, 0, 0), time="t") == "invalid_time"
        assert code_of(oe.corrgram, data, "y", time="t") == "invalid_time"
        assert code_of(oe.wntestq, data, "y", time="t") == "invalid_time"
        assert code_of(oe.archlm, data, "y", time="t") == "invalid_time"
        assert code_of(oe.tssmooth, data=data, y="y", time="t") == "invalid_time"
    with pytest.raises(AnalysisError, match="to_datetime"):
        oe.arima(data=text, y="y", order=(1, 0, 0), time="t")
    assert code_of(oe.corrgram, frame, "y", time=["t"]) == "invalid_spec"
    assert code_of(oe.corrgram, frame, "y", time="y") == "invalid_spec"


# ---- extreme magnitudes --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ml", "css"])
def test_units_and_levels_of_the_data_do_not_matter(frame, method):
    """Regression: a level shift of 1e7 standard deviations (or a regressor with such an
    offset, or mixed units) ended in 'nonconvergence' because the numerical Hessian was
    taken in raw units. Estimation is now standardized internally."""
    options = dict(y="y", x=["x1"], order=(1, 0, 1), method=method)
    n = len(frame)
    for kind in ("opg", "nonrobust", "robust"):
        base = oe.arima(data=frame, covariance=kind, **options)
        b, se = estimates(base), std_errors(base)
        for c in (1e-8, 1e8):
            fit = oe.arima(data=frame.assign(y=frame.y * c), covariance=kind, **options)
            factor = np.array([c, c, 1.0, 1.0, c])
            assert_allclose(estimates(fit), b * factor, rtol=1e-7)
            assert_allclose(std_errors(fit), se * factor, rtol=1e-5)
            assert fit.metrics["log_likelihood"] == pytest.approx(
                base.metrics["log_likelihood"] - n * math.log(c), rel=1e-10)
            assert_finite(fit)
        for c in (1e-8, 1e8):
            fit = oe.arima(data=frame.assign(x1=frame.x1 * c), covariance=kind, **options)
            factor = np.array([1.0, 1 / c, 1.0, 1.0, 1.0])
            assert_allclose(estimates(fit), b * factor, rtol=1e-7)
            assert_allclose(std_errors(fit), se * factor, rtol=1e-5)
        mixed = oe.arima(data=frame.assign(y=frame.y * 1e5, x1=frame.x1 * 1e-3), covariance=kind,
                         **options)
        assert_allclose(estimates(mixed), b * np.array([1e5, 1e8, 1.0, 1.0, 1e5]), rtol=1e-7)
    base = oe.arima(data=frame, **options)
    b, se = estimates(base), std_errors(base)
    for shift in (1e4, 1e7, 1e9):
        fit = oe.arima(data=frame.assign(y=frame.y + shift), **options)
        assert_allclose(estimates(fit)[1:], b[1:], rtol=1e-5, atol=1e-7)
        assert estimates(fit)[0] == pytest.approx(b[0] + shift, rel=1e-12)
        assert_allclose(std_errors(fit), se, rtol=1e-4)
        assert fit.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"],
                                                              abs=1e-5)
    for shift in (1e3, 1e6, 1e8):
        fit = oe.arima(data=frame.assign(x1=frame.x1 + shift), **options)
        assert_allclose(estimates(fit)[1:], b[1:], rtol=1e-5, atol=1e-7)
        assert estimates(fit)[0] == pytest.approx(b[0] - shift * b[1], rel=1e-6)
        assert_allclose(std_errors(fit)[1:], se[1:], rtol=1e-4)
    assert "standardization" in base.provenance
    # Without a constant nothing is centered: the model through the origin is unchanged.
    origin = oe.arima(data=frame, constant=False, **options)
    scaled = oe.arima(data=frame.assign(y=frame.y * 1e6), constant=False, **options)
    assert_allclose(estimates(scaled), estimates(origin) * np.array([1e6, 1.0, 1.0, 1e6]),
                    rtol=1e-7)


# ---- boundaries of the parameter space -----------------------------------------------------------


def test_unit_roots_overdifferencing_and_explosive_series(frame):
    walk = frame.assign(y=np.cumsum(frame.y))
    near = oe.arima(data=walk, y="y", order=(1, 0, 0))
    assert 0.99 < estimates(near)[1] < 1.0                        # exact ML stays stationary
    assert any("unit root" in warning for warning in near.warnings)
    assert_finite(near)
    conditional = oe.arima(data=walk, y="y", order=(1, 0, 0), method="css")
    assert estimates(conditional)[1] > 1.0
    assert any("not stationary" in warning for warning in conditional.warnings)
    assert_finite(conditional)
    explosive = frame.assign(y=1.05 ** frame.t + 0.01 * frame.y)
    fit = oe.arima(data=explosive, y="y", order=(1, 0, 0), method="css")
    assert estimates(fit)[1] == pytest.approx(1.05, abs=2e-3)
    assert_finite(fit)
    assert code_of(oe.forecast, fit, 10_000) == "non_finite_result"
    # Over-differencing piles the MA estimate up on -1: OPG is degenerate there, the
    # observed information is reported and the substitution is recorded.
    for kind in ("opg", "robust", "nonrobust"):
        over = oe.arima(data=frame, y="y", order=(0, 2, 1), covariance=kind)
        assert estimates(over)[1] == pytest.approx(-1.0, abs=1e-5)
        assert any("invertibility" in warning for warning in over.warnings)
        assert over.inference["covariance"] == "nonrobust"
        assert over.inference.get("requested_covariance") == (None if kind == "nonrobust"
                                                              else kind)
        assert_finite(over)
    # An over-parameterized model either fits with a warning or fails with a clear code.
    noise = pd.DataFrame({"y": np.random.default_rng(5).normal(size=150)})
    try:
        fit = oe.arima(data=noise, y="y", order=(2, 0, 2))
    except AnalysisError as error:
        assert error.code in {"nonconvergence", "singular_information"}
    else:
        assert_finite(fit)


def test_optimizer_record_counts_every_iteration(frame):
    """Regression: 'iterations' was 0 whenever the screening run had already converged."""
    mixed = oe.arima(data=frame, y="y", order=(1, 0, 1))
    record = mixed.provenance["optimizer"]
    assert record["iterations"] == record["screening_iterations"] + record["final_iterations"]
    assert record["screening_iterations"] > 0 and mixed.metrics["iterations"] > 0
    assert mixed.metrics["iterations"] == record["iterations"]
    assert mixed.metrics["log_likelihood"] == pytest.approx(
        max(value for value in record["starts"].values() if value is not None), abs=1e-6)
    pure = oe.arima(data=frame, y="y", order=(1, 0, 0))
    record = pure.provenance["optimizer"]
    assert record["screening_iterations"] == 0 and record["iterations"] > 0
    assert oe.arima(data=frame, y="y", order=(0, 0, 0), constant=False).metrics["iterations"] == 0


# ---- forecasts -----------------------------------------------------------------------------------


def test_forecast_rejects_bad_requests(frame):
    fit = oe.arima(data=frame, y="y", x=["g", "x1"], categorical=["g"], order=(1, 0, 0), time="t")
    exog = {"g": ["a", "b"], "x1": [0.0, 1.0]}
    assert len(oe.forecast(fit, 2, exog=exog)) == 2
    assert len(oe.forecast(fit, 2, exog=[{"g": "a", "x1": 0.0}, {"g": "b", "x1": 1.0}])) == 2
    assert len(oe.forecast(fit, 2, exog={**exog, "unused": [1, 2]})) == 2
    cases = [
        (dict(steps=0, exog=exog), "invalid_steps"), (dict(steps=-1, exog=exog), "invalid_steps"),
        (dict(steps=2.0, exog=exog), "invalid_steps"),
        (dict(steps="2", exog=exog), "invalid_steps"),
        (dict(steps=10_001), "invalid_steps"), (dict(steps=2), "missing_exog"),
        (dict(steps=3, exog=exog), "invalid_exog"),
        (dict(steps=2, exog={"g": ["a", "zz"], "x1": [0.0, 1.0]}), "invalid_exog"),
        (dict(steps=2, exog={"g": ["a", "b"], "x1": [0.0, np.nan]}), "missing_values"),
        (dict(steps=2, exog={"g": ["a", "b"], "x1": ["u", "v"]}), "non_numeric_column"),
        (dict(steps=2, exog={"x1": [0.0, 1.0]}), "missing_columns"),
        (dict(steps=2, exog="text"), "invalid_data"),
        (dict(steps=2, exog=exog, alpha=0.0), "invalid_option"),
        (dict(steps=2, exog=exog, alpha="a"), "invalid_option"),
        (dict(steps=2, exog=exog, data=frame.iloc[:2]), "missing_columns"),   # category c absent
        (dict(steps=2, exog=exog, data=frame[["t", "y"]]), "missing_columns"),
        (dict(steps=2, exog=exog, data=frame.drop(index=[9])), "time_gaps"),
    ]
    for options, code in cases:
        steps = options.pop("steps")
        assert code_of(oe.forecast, fit, steps, **options) == code, (steps, options)
    # Three rows are enough to rebuild the state of an AR(1) model.
    assert len(oe.forecast(fit, 2, exog=exog, data=frame.iloc[:3])) == 2
    assert code_of(oe.arima_forecast, "not a result", 2) == "invalid_result"
    assert code_of(oe.arima_forecast, oe.ols(data=frame, y="y", x=["x1"]), 2) == "invalid_result"
    # Regression: a category the model never saw must not pass as the reference category.
    unseen = frame.copy()
    unseen.loc[5, "g"] = "zz"
    assert code_of(oe.forecast, fit, 2, exog=exog, data=unseen) == "invalid_data"
    fewer = frame[frame.g != "c"].assign(t=np.arange(80))
    assert code_of(oe.forecast, fit, 2, exog=exog, data=fewer) == "missing_columns"
    plain = oe.arima(data=frame, y="y", order=(1, 1, 1))
    assert code_of(oe.forecast, plain, 2, exog={"x1": [0.0, 1.0]}) == "invalid_exog"
    assert code_of(oe.forecast, plain, 2, data=frame.iloc[:3]) == "insufficient_observations"
    assert code_of(oe.forecast, plain, 2, data=frame.iloc[:1]) == "insufficient_observations"
    assert code_of(oe.forecast, plain, 2, data=frame.iloc[:0]) == "empty_data"
    table = oe.forecast(plain, 10_000)
    assert len(table) == 10_000 and np.isfinite(table.to_numpy()).all()
    assert table["period"].tolist()[:2] == [1, 2] and table.attrs["period_unit"] == "steps ahead"
    # The series is stationary, so d = 1 over-differences it: the MA unit root cancels the
    # integration and the forecast error variance stays bounded.
    assert table["std_error"].iloc[-1] == pytest.approx(table["std_error"].iloc[5000], rel=1e-9)
    # A truly integrated series: the forecast standard error grows without bound.
    walk = oe.arima(data=frame.assign(y=np.cumsum(frame.y)), y="y", order=(1, 1, 0))
    table = oe.forecast(walk, 2_000)
    assert (np.diff(table["std_error"]) > 0).all()
    assert table["std_error"].iloc[-1] > 20 * table["std_error"].iloc[0]


# ---- diagnostics and smoothing -------------------------------------------------------------------


def test_diagnostics_reject_degenerate_series(frame):
    for function in (oe.corrgram, oe.wntestq, oe.jarque_bera, oe.archlm):
        assert code_of(function, frame.iloc[:0], "y") == "empty_data"
        assert code_of(function, frame.assign(y=1.0), "y") == "constant_series"
        assert code_of(function, frame.assign(y=frame.y.where(frame.t != 4)), "y") \
            == "missing_values"
        assert code_of(function, frame.assign(y="a"), "y") == "non_numeric_column"
        assert code_of(function, frame, "nope") == "missing_columns"
        assert code_of(function, None, "y") == "invalid_data"
        assert code_of(function, frame, ["y"]) == "invalid_spec"
        assert code_of(function, frame.iloc[:3], "y") == "insufficient_observations"
    n = len(frame)
    assert code_of(oe.corrgram, frame, "y", lags=n) == "invalid_lags"
    assert code_of(oe.corrgram, frame, "y", lags=0) == "invalid_lags"
    assert code_of(oe.corrgram, frame, "y", lags=3.0) == "invalid_lags"
    assert code_of(oe.corrgram, frame, "y", pacf="burg") == "invalid_option"
    assert len(oe.corrgram(frame, "y", lags=n - 1)) == n - 1                 # the largest lag
    assert np.isfinite(oe.corrgram(frame, "y", lags=n - 1).to_numpy()).all()
    assert code_of(oe.corrgram, frame, "y", lags=n - 1, pacf="regression") == "collinear_lags"
    assert code_of(oe.wntestq, frame, "y", lags=True) == "invalid_lags"
    assert code_of(oe.archlm, frame, "y", lags=[1, 1]) == "invalid_lags"
    assert code_of(oe.archlm, frame, "y", lags=[]) == "invalid_lags"
    assert code_of(oe.archlm, frame, "y", lags=59) == "insufficient_observations"
    assert oe.archlm(frame, "y", lags=58).attrs["nobs"] == n - 58
    assert code_of(oe.archlm, frame, "y", demean="yes") == "invalid_option"
    assert code_of(oe.corrgram, frame.drop(index=[7]), "y", time="t") == "time_gaps"
    assert code_of(oe.corrgram, pd.concat([frame, frame.iloc[[3]]]), "y", time="t") \
        == "repeated_time_values"
    # A series that alternates exactly: squared values are constant, lags are collinear.
    flip = pd.DataFrame({"y": np.tile([1.0, -1.0], 40)})
    assert code_of(oe.archlm, flip, "y") == "constant_series"
    assert code_of(oe.corrgram, flip, "y", lags=5, pacf="regression") == "collinear_lags"
    table = oe.corrgram(flip, "y", lags=5)
    assert np.isfinite(table.to_numpy()).all() and table["acf"][0] == pytest.approx(-79 / 80)
    assert oe.jarque_bera(flip, "y").attrs["statistic"] == pytest.approx(80 / 6 * 4 / 4)
    # Extreme units leave the statistics unchanged.
    for c in (1e-8, 1e8):
        scaled = frame.assign(y=frame.y * c)
        assert oe.wntestq(scaled, "y", lags=5).attrs["statistic"] == pytest.approx(
            oe.wntestq(frame, "y", lags=5).attrs["statistic"], rel=1e-9)
        assert oe.archlm(scaled, "y", lags=2, demean=True).attrs["statistic"] == pytest.approx(
            oe.archlm(frame, "y", lags=2, demean=True).attrs["statistic"], rel=1e-7)


def test_smoothing_rejects_bad_requests_and_handles_extremes(frame):
    base = dict(data=frame, y="y")
    cases = [
        (dict(method="holt"), "invalid_option"), (dict(additive="yes"), "invalid_option"),
        (dict(period=4), "invalid_option"), (dict(method="shwinters"), "invalid_option"),
        (dict(method="shwinters", period=1), "invalid_option"),
        (dict(method="shwinters", period=2.0), "invalid_option"),
        (dict(method="shwinters", period=61), "insufficient_observations"),
        (dict(alpha=1.5), "invalid_parameter"), (dict(alpha=-0.1), "invalid_parameter"),
        (dict(alpha="a"), "invalid_parameter"), (dict(alpha=True), "invalid_parameter"),
        (dict(gamma=0.2), "invalid_parameter"),
        (dict(method="exponential", beta=0.2), "invalid_parameter"),
        (dict(forecast=-1), "invalid_steps"), (dict(forecast=1.5), "invalid_steps"),
        (dict(forecast=10_001), "invalid_steps"),
    ]
    for options, code in cases:
        assert code_of(oe.tssmooth, **base, **options) == code, options
    assert code_of(oe.tssmooth, data=frame.iloc[:0], y="y") == "empty_data"
    assert code_of(oe.tssmooth, data=frame.iloc[:3], y="y") == "insufficient_observations"
    assert code_of(oe.tssmooth, data=frame.assign(y=1.0), y="y") == "constant_series"
    assert code_of(oe.tssmooth, data=frame.assign(y=2.0 * frame.t), y="y") == "perfect_fit"
    assert code_of(oe.tssmooth, data=frame.assign(y=frame.y.where(frame.t != 4)), y="y") \
        == "missing_values"
    assert code_of(oe.tssmooth, data=frame.assign(y=frame.y - 30), y="y", method="shwinters",
                   period=4, additive=False) == "nonpositive_series"
    assert code_of(oe.tssmooth, data=frame.drop(index=[7]), y="y", time="t") == "time_gaps"
    # The largest admissible season, and scale equivariance at extreme units.
    assert oe.tssmooth(**base, method="shwinters", period=60).attrs["nobs"] == len(frame)
    positive = frame.assign(y=frame.y + 10)
    for options in (dict(method="exponential"), dict(method="dexponential"),
                    dict(method="hwinters"), dict(method="shwinters", period=4),
                    dict(method="shwinters", period=4, additive=False)):
        reference = oe.tssmooth(data=positive, y="y", forecast=2, **options)
        assert np.isfinite(reference["forecast"]).all()
        for c in (1e-8, 1e8):
            scaled = oe.tssmooth(data=positive.assign(y=positive.y * c), y="y", forecast=2,
                                 **options)
            assert scaled.attrs["rmse"] == pytest.approx(reference.attrs["rmse"] * c, rel=1e-6)
            assert_allclose(scaled["forecast"], reference["forecast"] * c, rtol=1e-5)
    dated = oe.tssmooth(data=frame.assign(t=pd.date_range("2001-01-01", periods=len(frame),
                                                          freq="MS")),
                        y="y", time="t", forecast=2)
    assert dated["period"].iloc[-1] == pd.Timestamp("2011-02-01")


def test_smoothing_finds_the_global_minimum_of_a_bimodal_criterion():
    """Regression: Holt's SSE for this series has two local minima, (alpha, beta) =
    (1, 0) and about (0.80, 1); the search used to stop in the second, 1.8 % worse."""
    rng = np.random.default_rng(11)
    t = np.arange(120)
    x = 50 + np.cumsum(rng.normal(0.2, 1.0, size=120)) + 5 * np.sin(2 * np.pi * t / 12) \
        + rng.normal(size=120)
    half = 60
    slope, intercept = np.polyfit(np.arange(1, half + 1), x[:half], 1)

    def sse(point):
        level, trend, total = intercept, slope, 0.0
        for value in x:
            total += (value - level - trend) ** 2
            new = point[0] * value + (1 - point[0]) * (level + trend)
            trend = point[1] * (new - level) + (1 - point[1]) * trend
            level = new
        return total

    table = oe.tssmooth(data=pd.DataFrame({"x": x}), y="x", method="hwinters")
    assert table.attrs["parameters"] == {"alpha": 1.0, "beta": 0.0}
    assert set(table.attrs["at_bounds"]) == {"alpha", "beta"}
    assert table.attrs["sse"] == pytest.approx(sse([1.0, 0.0]), rel=1e-10)
    other = minimize(sse, [0.8, 0.9], bounds=[(0, 1)] * 2, method="L-BFGS-B")
    assert other.fun > table.attrs["sse"] * 1.01                  # the local minimum it avoids
    grid = min(sse([a, b]) for a in np.linspace(0, 1, 21) for b in np.linspace(0, 1, 21))
    assert table.attrs["sse"] <= grid * (1 + 1e-12)


def test_smoothing_search_on_a_long_series_reaches_the_full_sample_minimum():
    """Beyond 5,000 observations the starting basin is chosen on the first 5,000 and the
    parameters are then optimized on the whole series: the result must be the minimum of
    the full-sample criterion, not of the screening sample."""
    rng = np.random.default_rng(21)
    n = 7_000
    x = 10 + np.cumsum(rng.normal(0.01, 0.3, size=n)) + rng.normal(size=n)
    x[5_000:] += np.cumsum(rng.normal(0.0, 1.0, size=n - 5_000))      # the regime changes late
    level0 = x[:n // 2].mean()

    def sse(alpha, values=x, start=level0):
        # e_t = x_t - S_(t-1) and S_t = S_(t-1) + alpha e_t give
        # e_t = (x_t - x_(t-1)) + (1 - alpha) e_(t-1), with x_0 = S_0 and e_0 = 0.
        errors = lfilter([1.0], [1.0, alpha - 1.0], np.diff(np.r_[start, values]))
        return float(errors @ errors)

    grid = np.linspace(0.0, 1.0, 2001)
    best = grid[np.argmin([sse(a) for a in grid])]
    table = oe.tssmooth(data=pd.DataFrame({"x": x}), y="x", method="exponential")
    alpha = table.attrs["parameters"]["alpha"]
    assert table.attrs["sse"] == pytest.approx(sse(alpha), rel=1e-9)
    assert abs(alpha - best) < 1e-3 and table.attrs["sse"] <= sse(best) * (1 + 1e-9)
    short = oe.tssmooth(data=pd.DataFrame({"x": x[:5_000]}), y="x", method="exponential")
    assert abs(short.attrs["parameters"]["alpha"] - alpha) > 0.05   # the two samples disagree


# ---- the public surface --------------------------------------------------------------------------


def test_manifest_is_light_and_every_declared_feature_is_handled(frame):
    probe = ("import sys; import openecon.econometrics.arima as m; "
             "assert 'torch' not in sys.modules and 'pandas' not in sys.modules; "
             "print(len(m.ESTIMATORS), sorted(m.EXPORTS), sorted(m.FORECAST))")
    output = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            check=True).stdout
    assert output.split()[0] == "1" and "arima_forecast" in output
    info = registry.get("arima")
    assert info.function == "arima" and info.family == "arima" and info.inference == "z"
    assert info.default_covariance == "opg" and info.weights == () and info.panel == "none"
    for kind in info.covariances:
        assert oe.arima(data=frame, y="y", order=(1, 0, 0), covariance=kind).inference[
            "covariance"] == kind
    declared = {option.name for option in info.options}
    assert declared == {"order", "seasonal", "period", "method", "constant", "ljung_lags",
                        "max_iterations", "tolerance"}
    through_spec = oe.fit(oe.ModelSpec(
        estimator="arima", outcome="y", predictors=["x1"], time="t",
        options={"order": [1, 0, 1], "seasonal": [0, 0, 1], "period": 4, "method": "css",
                 "constant": False, "ljung_lags": 6, "max_iterations": 150, "tolerance": 1e-9}),
        data=frame)
    direct = oe.arima(data=frame, y="y", x=["x1"], time="t", order=(1, 0, 1), seasonal=(0, 0, 1),
                      period=4, method="css", constant=False, ljung_lags=6, max_iterations=150,
                      tolerance=1e-9)
    assert_allclose(estimates(through_spec), estimates(direct), rtol=1e-12)
    assert [c.term for c in direct.coefficients] == ["x1", "ARMA:L1.ar", "ARMA:L1.ma",
                                                     "ARMA4:L1.ma", "/sigma"]
    assert direct.tests["ljung_box"]["lags"] == 6 and direct.extra["method"] == "css"
    exports = registry.public_exports()
    for name in ("arima", "arima_forecast", "forecast", "tssmooth", "corrgram", "wntestq",
                 "jarque_bera", "archlm"):
        assert name in exports and callable(getattr(oe, name))
        text = getattr(oe, name).__doc__
        assert text and len(text) > 200, name
    for name in ("arima", "arima_forecast", "tssmooth", "corrgram", "wntestq", "archlm"):
        assert "Stata" in getattr(oe, name).__doc__ and ">>>" in getattr(oe, name).__doc__
    for word in ("order", "seasonal", "period", "method", "constant", "covariance",
                 "categorical", "ljung_lags", "max_iterations", "tolerance", "missing", "alpha",
                 "opg", "nonrobust", "robust", "log_likelihood", "last_state"):
        assert word in oe.arima.__doc__, word
    restored = type(direct).model_validate_json(direct.model_dump_json())
    assert_allclose(oe.forecast(restored, 3, exog={"x1": [0.0, 0.1, 0.2]})["forecast"],
                    oe.forecast(direct, 3, exog={"x1": [0.0, 0.1, 0.2]})["forecast"], rtol=1e-13)
    assert direct.provenance["stata_parity_validated"] is False
    assert "Std. error" in direct.summary() and "ARMA4:L1.ma" in direct.summary()
