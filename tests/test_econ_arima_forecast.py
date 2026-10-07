"""Independent oracles for oe.forecast / oe.arima_forecast (dynamic forecasts of oe.arima).

Point forecasts and forecast standard errors are compared with statsmodels'
Kalman-filter forecasts at the same parameters (stationary, integrated, seasonal
and regression models), with an explicit NumPy recursion for the conditional
estimator, and with the textbook MA(infinity) weights of the integrated process.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.signal import lfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.frame import DataFrame
from openecon.models import ResultBundle


def simulate(n, phi=(), theta=(), seed=0, burn=300):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + burn)
    return lfilter(np.r_[1.0, theta], np.r_[1.0, -np.asarray(phi)], e)[burn:]


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def statsmodels_params(result):
    params = estimates(result)
    params[-1] **= 2                       # statsmodels carries sigma^2
    return params


def sm_forecast(model, params, steps, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.smooth(params).get_forecast(steps, **kwargs)
        return np.asarray(prediction.predicted_mean), np.asarray(prediction.se_mean)


@pytest.fixture(scope="module")
def armax():
    n = 40                                  # short sample, strong MA: the end-of-sample
    rng = np.random.default_rng(5)          # disturbance is uncertain under exact ML
    x1 = rng.normal(size=n + 6)
    y = 2 + 0.7 * x1[:n] + simulate(n, [0.5, -0.3], [0.85], seed=2)
    frame = pd.DataFrame({"y": y, "x1": x1[:n], "t": np.arange(2001, 2001 + n)})
    future = pd.DataFrame({"x1": x1[n:]})
    fit = oe.arima(data=frame, y="y", x=["x1"], order=(2, 0, 1), time="t")
    return frame, future, fit


def test_stationary_armax_forecast_matches_statsmodels(armax):
    frame, future, fit = armax
    table = oe.arima_forecast(fit, 6, exog=future)
    n = len(frame)
    design = np.column_stack([np.ones(n), frame["x1"]])
    model = SARIMAX(frame["y"].to_numpy(), exog=design, order=(2, 0, 1))
    mean, se = sm_forecast(model, statsmodels_params(fit), 6,
                           exog=np.column_stack([np.ones(6), future["x1"]]))
    assert_allclose(table["forecast"], mean, rtol=1e-10, atol=1e-10)
    assert_allclose(table["std_error"], se, rtol=1e-10)
    z = stats.norm.ppf(0.975)
    assert_allclose(table["ci_low"], mean - z * se, rtol=1e-9, atol=1e-9)
    assert_allclose(table["ci_high"], mean + z * se, rtol=1e-9, atol=1e-9)
    assert isinstance(table, DataFrame)
    assert list(table.columns) == ["period", "forecast", "std_error", "ci_low", "ci_high"]
    assert table["period"].tolist() == list(range(2041, 2047))        # the time column continues
    assert table.attrs["model"] == "ARIMA(2,0,1)" and table.attrs["outcome"] == "y"
    assert table.attrs["origin"] == "end of the estimation sample"
    assert table.attrs["period_unit"] == "time value"
    assert table.attrs["sigma"] == pytest.approx(fit.metrics["sigma"])
    # The uncertainty about the last disturbance matters here: the pure MA(infinity)
    # formula sigma * sqrt(cumsum(psi^2)) is strictly smaller at h = 1.
    assert fit.extra["last_state"]["disturbance_covariance"] is not None
    assert table["std_error"].iloc[0] > fit.metrics["sigma"] * (1 + 1e-4)
    wide = oe.arima_forecast(fit, 6, exog=future, alpha=0.2)
    assert_allclose(wide["ci_high"], mean + stats.norm.ppf(0.9) * se, rtol=1e-9)
    assert wide.attrs["confidence_level"] == pytest.approx(0.8)
    assert "forecast" in str(table.to_latex())


def test_integrated_model_with_drift_matches_statsmodels_and_psi_weights():
    n = 120
    y = np.cumsum(0.3 + simulate(n, [0.5], [0.4], seed=4))
    fit = oe.arima(data=pd.DataFrame({"y": y}), y="y", order=(1, 1, 1))
    table = oe.arima_forecast(fit, 8)
    # Stata's constant of the differenced series is a linear trend in the level equation.
    model = SARIMAX(y, exog=np.arange(1, n + 1.0), order=(1, 1, 1))
    mean, se = sm_forecast(model, statsmodels_params(fit), 8, exog=np.arange(n + 1, n + 9.0))
    assert_allclose(table["forecast"], mean, rtol=1e-9)
    assert_allclose(table["std_error"], se, rtol=1e-8)
    assert table["period"].tolist() == list(range(1, 9))              # no time column: steps
    assert table.attrs["period_unit"] == "steps ahead"
    # Textbook variance: sigma^2 * cumulative sum of squared MA(infinity) weights of the
    # integrated process theta(L) / (phi(L) (1 - L)).
    drift, phi, theta, sigma = estimates(fit)
    impulse = np.zeros(8)
    impulse[0] = 1.0
    psi = lfilter([1.0, theta], np.convolve([1.0, -phi], [1.0, -1.0]), impulse)
    assert_allclose(table["std_error"], sigma * np.sqrt(np.cumsum(psi ** 2)), rtol=1e-6)
    # Long-run slope of the forecast function is the drift.
    far = oe.arima_forecast(fit, 400)
    assert far["forecast"].diff().iloc[-1] == pytest.approx(drift, rel=1e-8)
    assert bool((far["std_error"].diff().dropna() > 0).all())


def test_seasonal_airline_forecast_and_second_differences():
    rng = np.random.default_rng(41)
    n, s = 180, 12
    w = lfilter(np.convolve([1, -0.4], np.r_[1, np.zeros(11), -0.6]), [1.0], rng.normal(size=n))
    y = lfilter([1.0], np.convolve([1, -1], np.r_[1, np.zeros(11), -1]), w)
    frame = pd.DataFrame({"y": y})
    fit = oe.arima(data=frame, y="y", order=(0, 1, 1), seasonal=(0, 1, 1), period=s,
                   constant=False)
    table = oe.arima_forecast(fit, 30)
    model = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, s))
    mean, se = sm_forecast(model, statsmodels_params(fit), 30)
    assert_allclose(table["forecast"], mean, rtol=1e-7, atol=1e-7)
    assert_allclose(table["std_error"], se, rtol=1e-7)
    assert table.attrs["model"] == "ARIMA(0,1,1)x(0,1,1)[12]"
    # A long horizon keeps its accuracy: the MA(infinity) weights of theta(L) Theta(L^12) /
    # ((1 - L)(1 - L^12)) have a double root at 1 and eleven seasonal unit roots. (The
    # conditional estimator has no end-of-sample disturbance uncertainty to add.)
    css = oe.arima(data=frame, y="y", order=(0, 1, 1), seasonal=(0, 1, 1), period=s,
                   constant=False, method="css")
    theta, big_theta, sigma = estimates(css)
    impulse = np.zeros(3000)
    impulse[0] = 1.0
    psi = lfilter(np.convolve([1, theta], np.r_[1, np.zeros(s - 1), big_theta]),
                  np.convolve([1, -1], np.r_[1, np.zeros(s - 1), -1]), impulse)
    far = oe.arima_forecast(css, 3000)
    assert_allclose(far["std_error"], sigma * np.sqrt(np.cumsum(psi ** 2)), rtol=1e-9)
    assert bool(np.isfinite(far["forecast"]).all())

    twice = np.cumsum(np.cumsum(simulate(150, [0.4], seed=8)))
    second = oe.arima(data=pd.DataFrame({"y": twice}), y="y", order=(1, 2, 0), constant=False)
    model = SARIMAX(twice, order=(1, 2, 0))
    mean, se = sm_forecast(model, statsmodels_params(second), 12)
    out = oe.arima_forecast(second, 12)
    assert_allclose(out["forecast"], mean, rtol=1e-8)
    assert_allclose(out["std_error"], se, rtol=1e-7)

    multiplicative = oe.arima(data=frame, y="y", order=(1, 0, 0), seasonal=(1, 1, 0), period=s,
                              covariance="nonrobust")
    model = SARIMAX(y, exog=np.arange(1, n + 1.0), order=(1, 0, 0), seasonal_order=(1, 1, 0, s))
    params = statsmodels_params(multiplicative)
    params[0] /= s          # constant of (1 - L^12) y is a drift of b0 / 12 per period
    mean, se = sm_forecast(model, params, 15, exog=np.arange(n + 1, n + 16.0))
    out = oe.arima_forecast(multiplicative, 15)
    assert_allclose(out["forecast"], mean, rtol=1e-7, atol=1e-7)
    assert_allclose(out["std_error"], se, rtol=1e-7)


def test_conditional_estimator_forecast_matches_explicit_recursion():
    n = 90
    rng = np.random.default_rng(21)
    x1 = rng.normal(size=n + 5).cumsum()
    y = np.cumsum(0.2 + simulate(n, [0.6], [0.5], seed=22)) + 1.5 * x1[:n]
    frame = pd.DataFrame({"y": y, "x1": x1[:n]})
    fit = oe.arima(data=frame, y="y", x=["x1"], order=(1, 1, 1), method="css")
    b0, b1, phi, theta, sigma = estimates(fit)
    u = np.diff(y - b1 * x1[:n]) - b0
    e = np.zeros(n - 1)
    for t in range(n - 1):                               # presample u and e are zero
        e[t] = u[t] - (phi * u[t - 1] + theta * e[t - 1] if t else 0.0)
    steps, level, last_u = 5, (y - b1 * x1[:n])[-1], u[-1]
    expected = []
    for h in range(1, steps + 1):
        last_u = phi * last_u + (theta * e[-1] if h == 1 else 0.0)
        level += b0 + last_u
        expected.append(level + b1 * x1[n + h - 1])
    table = oe.arima_forecast(fit, steps, exog=pd.DataFrame({"x1": x1[n:]}))
    assert_allclose(table["forecast"], expected, rtol=1e-10)
    impulse = np.zeros(steps)
    impulse[0] = 1.0
    psi = lfilter([1.0, theta], np.convolve([1.0, -phi], [1.0, -1.0]), impulse)
    assert_allclose(table["std_error"], sigma * np.sqrt(np.cumsum(psi ** 2)), rtol=1e-10)
    assert fit.extra["last_state"]["disturbance_covariance"] is None


def test_forecast_from_new_data_and_after_json_round_trip(armax):
    frame, future, fit = armax
    same = oe.arima_forecast(fit, 6, data=frame, exog=future)
    assert_allclose(same[["forecast", "std_error"]], oe.arima_forecast(fit, 6, exog=future)
                    [["forecast", "std_error"]], rtol=1e-12)
    assert same.attrs["origin"] == "end of the supplied data"
    # Forecast origin moved back five periods: the fitted model runs over the shorter sample.
    shorter = frame.iloc[:-5]
    moved = oe.arima_forecast(fit, 4, data=shorter, exog=pd.DataFrame({"x1": frame["x1"].iloc[-5:-1]
                                                                 .to_numpy()}))
    design = np.column_stack([np.ones(len(shorter)), shorter["x1"]])
    model = SARIMAX(shorter["y"].to_numpy(), exog=design, order=(2, 0, 1))
    mean, se = sm_forecast(model, statsmodels_params(fit), 4,
                           exog=np.column_stack([np.ones(4), frame["x1"].iloc[-5:-1]]))
    assert_allclose(moved["forecast"], mean, rtol=1e-10, atol=1e-10)
    assert_allclose(moved["std_error"], se, rtol=1e-10)
    assert moved["period"].tolist() == [int(shorter["t"].iloc[-1]) + h for h in range(1, 5)]
    restored = ResultBundle.model_validate_json(fit.model_dump_json())
    again = oe.arima_forecast(restored, 6, exog=future)
    assert_allclose(again[["forecast", "std_error", "ci_low", "ci_high"]],
                    oe.arima_forecast(fit, 6, exog=future)[["forecast", "std_error", "ci_low",
                                                      "ci_high"]], rtol=1e-13)
    records = oe.arima_forecast(fit, 2, exog=[{"x1": 0.5}, {"x1": -0.5}])   # rows as records
    columns = oe.arima_forecast(fit, 2, exog={"x1": [0.5, -0.5]})          # mapping of columns
    assert_allclose(records["forecast"], columns["forecast"], rtol=1e-14)
    shifted = oe.arima_forecast(fit, 2, exog={"x1": [1.5, 0.5]})
    slope = fit.coefficients[1].estimate
    assert_allclose(shifted["forecast"] - columns["forecast"], [slope, slope], rtol=1e-10)


def test_categorical_regressors_in_forecasts():
    n = 150
    group = np.array(["a", "b", "c"])[np.arange(n) % 3]
    effect = {"a": 0.0, "b": 1.0, "c": -2.0}
    y = np.array([effect[g] for g in group]) + simulate(n, [0.5], seed=4)
    frame = pd.DataFrame({"y": y, "g": group})
    fit = oe.arima(data=frame, y="y", x=["g"], categorical=["g"], order=(1, 0, 0))
    dummies = frame.assign(b=(group == "b") * 1.0, c=(group == "c") * 1.0)
    manual = oe.arima(data=dummies, y="y", x=["b", "c"], order=(1, 0, 0))
    left = oe.arima_forecast(fit, 3, exog=pd.DataFrame({"g": ["a", "b", "c"]}))
    right = oe.arima_forecast(manual, 3, exog=pd.DataFrame({"b": [0.0, 1.0, 0.0],
                                                            "c": [0.0, 0.0, 1.0]}))
    assert_allclose(left["forecast"], right["forecast"], rtol=1e-8)
    with pytest.raises(AnalysisError) as error:
        oe.arima_forecast(fit, 1, exog=pd.DataFrame({"g": ["z"]}))
    assert error.value.code == "invalid_exog"


def test_white_noise_and_pure_ma_forecasts():
    y = 5.0 + simulate(80, seed=11)
    fit = oe.arima(data=pd.DataFrame({"y": y}), y="y", order=(0, 0, 0))
    table = oe.arima_forecast(fit, 3)
    assert_allclose(table["forecast"], [y.mean()] * 3, rtol=1e-12)
    assert_allclose(table["std_error"], [fit.metrics["sigma"]] * 3, rtol=1e-12)
    ma = oe.arima(data=pd.DataFrame({"y": simulate(200, theta=[0.6, 0.3], seed=12)}), y="y",
                  order=(0, 0, 2))
    out = oe.arima_forecast(ma, 5)
    constant, t1, t2, sigma = estimates(ma)
    assert_allclose(out["forecast"].iloc[2:], [constant] * 3, rtol=1e-12)   # beyond q: the mean
    assert out["std_error"].iloc[-1] == pytest.approx(sigma * math.sqrt(1 + t1 ** 2 + t2 ** 2),
                                                      rel=1e-6)


def test_forecast_error_codes_and_exports(armax):
    frame, future, fit = armax
    cases = [
        (dict(result="fit", steps=3), "invalid_result"),
        (dict(result=fit, steps=0, exog=future), "invalid_steps"),
        (dict(result=fit, steps=2.0, exog=future), "invalid_steps"),
        (dict(result=fit, steps=True, exog=future), "invalid_steps"),
        (dict(result=fit, steps=10_001), "invalid_steps"),
        (dict(result=fit, steps=6), "missing_exog"),
        (dict(result=fit, steps=3, exog=future), "invalid_exog"),            # 6 rows for 3 steps
        (dict(result=fit, steps=6, exog=future.rename(columns={"x1": "z"})), "missing_columns"),
        (dict(result=fit, steps=6, exog=future.assign(x1=np.nan)), "missing_values"),
        (dict(result=fit, steps=6, exog=future, alpha=1.5), "invalid_option"),
        (dict(result=fit, steps=6, exog=future, data=frame[["y", "t"]]), "missing_columns"),
        (dict(result=fit, steps=6, exog=future, data=frame.iloc[:3]), "insufficient_observations"),
    ]
    for kwargs, code in cases:
        result, steps = kwargs.pop("result"), kwargs.pop("steps")
        with pytest.raises(AnalysisError) as error:
            oe.arima_forecast(result, steps, **kwargs)
        assert error.value.code == code, code
    no_regressors = oe.arima(data=frame, y="y", order=(1, 0, 0))
    with pytest.raises(AnalysisError) as error:
        oe.arima_forecast(no_regressors, 2, exog=future.iloc[:2])
    assert error.value.code == "invalid_exog"
    other = oe.ols(data=frame, y="y", x=["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.arima_forecast(other, 2)
    assert error.value.code == "invalid_result"


def test_forecast_is_published_as_oe_forecast(armax):
    """oe.forecast is the shared dispatcher; the arima manifest registers this family's
    function for arima results and also exports it as oe.arima_forecast."""
    frame, future, fit = armax
    exports = registry.public_exports()
    assert exports["arima_forecast"] == ("openecon.econometrics.arima.forecast", "forecast")
    assert exports["forecast"] == ("openecon.econometrics.core", "forecast")
    assert registry.forecasters()["arima"] == ("openecon.econometrics.arima.forecast", "forecast")
    assert_allclose(oe.forecast(fit, 6, exog=future)["forecast"],
                    oe.arima_forecast(fit, 6, exog=future)["forecast"], rtol=1e-14)
    assert_allclose(oe.forecast(fit, steps=2, exog=future.iloc[:2])["std_error"],
                    oe.arima_forecast(fit, 2, exog=future.iloc[:2])["std_error"], rtol=1e-14)
    # NumPy integers are accepted as the horizon; an integer alpha is still refused.
    assert len(oe.forecast(fit, np.int64(6), exog=future)) == 6
    with pytest.raises(AnalysisError) as error:
        oe.forecast(fit, 6, exog=future, alpha=1)
    assert error.value.code == "invalid_option"
