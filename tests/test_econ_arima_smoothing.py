"""Independent oracles for oe.tssmooth (exponential smoothing and Holt-Winters).

Every smoother is compared, at fixed parameters, with its textbook recursion written
out in NumPy (the form of Stata's [TS] tssmooth manual entries) and with statsmodels
where its recursion is the same; the estimated parameters are compared with SciPy
minimizations of the explicit sum of squared one-step errors.
"""

import math
import time
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize, minimize_scalar
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.arima.smoothing import METHODS, _Smoother
from openecon.engines.optimize import check_derivatives
from openecon.frame import DataFrame

M = 12


@pytest.fixture(scope="module")
def seasonal():
    rng = np.random.default_rng(1)
    n = 96
    t = np.arange(1, n + 1)
    y = 20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / M) + rng.normal(size=n)
    return pd.DataFrame({"y": y, "month": np.arange(601, 601 + n)})


@pytest.fixture(scope="module")
def trending():
    rng = np.random.default_rng(2)
    n = 80
    level = 10 + np.cumsum(0.4 + 0.5 * rng.normal(size=n))
    return pd.DataFrame({"y": level + rng.normal(size=n)})


# ---- textbook recursions -------------------------------------------------------------------


def first_half_line(y):
    half = max(2, len(y) // 2)
    slope, intercept = np.polyfit(np.arange(1, half + 1), y[:half], 1)
    return intercept, slope


def seasonal_start(y, m, additive):
    """Level, trend and seasonal terms from the complete seasons of the first half."""
    years = max(2, min(len(y) // m, (len(y) // 2) // m))
    block = y[:years * m].reshape(years, m)
    means = block.mean(axis=1)
    trend = (means[-1] - means[0]) / ((years - 1) * m)
    level = means[0] - trend * (m + 1) / 2
    line = level + trend * np.arange(1, years * m + 1).reshape(years, m)
    if additive:
        terms = (block - line).mean(axis=0)
        return level, trend, terms - terms.mean()
    terms = (block / line).mean(axis=0)
    return level, trend, terms / terms.mean()


def simple_recursion(y, alpha, s0, steps=0):
    forecast, smoothed, state = np.zeros(len(y)), np.zeros(len(y)), s0
    for t, x in enumerate(y):
        forecast[t] = state
        state = alpha * x + (1 - alpha) * state
        smoothed[t] = state
    return forecast, smoothed, np.full(steps, state)


def brown_recursion(y, alpha, a0, b0, steps=0):
    """Double exponential smoothing through its two smoothed series S and S[2]."""
    ratio = (1 - alpha) / alpha
    s1, s2 = a0 - ratio * b0, a0 - 2 * ratio * b0
    forecast, smoothed = np.zeros(len(y)), np.zeros(len(y))
    for t, x in enumerate(y):
        forecast[t] = (2 + alpha / (1 - alpha)) * s1 - (1 + alpha / (1 - alpha)) * s2
        s1 = alpha * x + (1 - alpha) * s1
        s2 = alpha * s1 + (1 - alpha) * s2
        smoothed[t] = 2 * s1 - s2
    level, trend = 2 * s1 - s2, alpha / (1 - alpha) * (s1 - s2)
    return forecast, smoothed, level + trend * np.arange(1, steps + 1)


def holt_recursion(y, alpha, beta, a0, b0, steps=0):
    a, b = a0, b0
    forecast, smoothed = np.zeros(len(y)), np.zeros(len(y))
    for t, x in enumerate(y):
        forecast[t] = a + b
        new = alpha * x + (1 - alpha) * (a + b)
        b = beta * (new - a) + (1 - beta) * b
        a = new
        smoothed[t] = a
    return forecast, smoothed, a + b * np.arange(1, steps + 1)


def winters_recursion(y, alpha, beta, gamma, a0, b0, s0, m, additive, steps=0):
    a, b, s = a0, b0, list(s0)
    n = len(y)
    forecast, smoothed = np.zeros(n), np.zeros(n)
    for t, x in enumerate(y):
        i = t % m
        if additive:
            forecast[t] = a + b + s[i]
            new = alpha * (x - s[i]) + (1 - alpha) * (a + b)
        else:
            forecast[t] = (a + b) * s[i]
            new = alpha * x / s[i] + (1 - alpha) * (a + b)
        b = beta * (new - a) + (1 - beta) * b
        a = new
        s[i] = gamma * (x - a if additive else x / a) + (1 - gamma) * s[i]
        smoothed[t] = a + s[i] if additive else a * s[i]
    ahead = np.array([(a + b * h) + s[(n + h - 1) % m] if additive
                      else (a + b * h) * s[(n + h - 1) % m] for h in range(1, steps + 1)])
    return forecast, smoothed, ahead


def check(table, forecast, smoothed, ahead, y):
    n = len(y)
    assert_allclose(table["forecast"].to_numpy()[:n], forecast, rtol=1e-9, atol=1e-9)
    assert_allclose(table["smoothed"].to_numpy()[:n], smoothed, rtol=1e-9, atol=1e-9)
    assert_allclose(table["forecast"].to_numpy()[n:], ahead, rtol=1e-9, atol=1e-9)
    assert_allclose(table["observed"].to_numpy()[:n], y, rtol=0, atol=0)
    assert table["observed"].iloc[n:].isna().all() and table["smoothed"].iloc[n:].isna().all()
    sse = np.sum((y - forecast) ** 2)
    assert table.attrs["sse"] == pytest.approx(sse, rel=1e-9)
    assert table.attrs["rmse"] == pytest.approx(math.sqrt(sse / n), rel=1e-9)
    assert table.attrs["nobs"] == n and table.attrs["forecast_steps"] == len(ahead)


# ---- fixed parameters ----------------------------------------------------------------------


def test_simple_exponential_smoothing(trending):
    y = trending["y"].to_numpy()
    s0 = y[:len(y) // 2].mean()                       # Stata: mean of the first half
    table = oe.tssmooth(data=trending, y="y", method="exponential", alpha=0.3, forecast=4)
    check(table, *simple_recursion(y, 0.3, s0, 4), y)
    assert table.attrs["initial"] == {"level": pytest.approx(s0), "trend": 0.0, "seasonal": None}
    assert table.attrs["parameters"] == {"alpha": 0.3} and table.attrs["estimated"] == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = SimpleExpSmoothing(y, initialization_method="known", initial_level=s0).fit(
            smoothing_level=0.3, optimized=False)
    assert_allclose(table["forecast"].to_numpy()[:len(y)], reference.fittedvalues, rtol=1e-10)
    assert_allclose(table["forecast"].to_numpy()[len(y):], reference.forecast(4), rtol=1e-10)
    assert isinstance(table, DataFrame)
    assert list(table.columns) == ["period", "observed", "smoothed", "forecast"]
    assert table["period"].tolist() == list(range(len(y) + 4))            # row numbers
    for alpha in (0.0, 1.0):                                              # the bounds are legal
        edge = oe.tssmooth(data=trending, y="y", method="exponential", alpha=alpha)
        check(edge, *simple_recursion(y, alpha, s0), y)


def test_double_exponential_smoothing(trending):
    y = trending["y"].to_numpy()
    a0, b0 = first_half_line(y)                       # Stata: regression on the first half
    table = oe.tssmooth(data=trending, y="y", method="dexponential", alpha=0.25, forecast=5)
    check(table, *brown_recursion(y, 0.25, a0, b0, 5), y)
    assert table.attrs["initial"]["level"] == pytest.approx(a0)
    assert table.attrs["initial"]["trend"] == pytest.approx(b0)


def test_holt_linear_trend_smoothing(trending):
    y = trending["y"].to_numpy()
    a0, b0 = first_half_line(y)
    table = oe.tssmooth(data=trending, y="y", method="hwinters", alpha=0.4, beta=0.2, forecast=6)
    check(table, *holt_recursion(y, 0.4, 0.2, a0, b0, 6), y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = Holt(y, initialization_method="known", initial_level=a0,
                         initial_trend=b0).fit(smoothing_level=0.4, smoothing_trend=0.2,
                                               optimized=False)
    assert_allclose(table["forecast"].to_numpy()[:len(y)], reference.fittedvalues, rtol=1e-10)
    assert_allclose(table["forecast"].to_numpy()[len(y):], reference.forecast(6), rtol=1e-10)
    assert_allclose(table["smoothed"].to_numpy()[:len(y)], reference.level, rtol=1e-10)
    default = oe.tssmooth(data=trending, y="y", alpha=0.4, beta=0.2)      # hwinters is the default
    assert default.attrs["method"] == "hwinters"
    assert_allclose(default["forecast"], table["forecast"].iloc[:len(y)], rtol=1e-13)


@pytest.mark.parametrize("additive", [True, False])
def test_seasonal_holt_winters(seasonal, additive):
    y = seasonal["y"].to_numpy()
    a0, b0, s0 = seasonal_start(y, M, additive)
    table = oe.tssmooth(data=seasonal, y="y", method="shwinters", period=M, additive=additive,
                        alpha=0.3, beta=0.1, gamma=0.2, forecast=15)
    check(table, *winters_recursion(y, 0.3, 0.1, 0.2, a0, b0, s0, M, additive, 15), y)
    initial = table.attrs["initial"]
    assert initial["level"] == pytest.approx(a0) and initial["trend"] == pytest.approx(b0)
    assert_allclose(initial["seasonal"], s0, rtol=1e-10)
    assert sum(initial["seasonal"]) == pytest.approx(0.0 if additive else M, abs=1e-9)
    assert table.attrs["seasonal_period"] == M and table.attrs["additive"] is additive
    if additive:
        # statsmodels updates the seasonal term with the one-step error, which is the
        # classical recursion with gamma* = gamma (1 - alpha).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = ExponentialSmoothing(
                y, trend="add", seasonal="add", seasonal_periods=M,
                initialization_method="known", initial_level=a0, initial_trend=b0,
                initial_seasonal=s0).fit(smoothing_level=0.3, smoothing_trend=0.1,
                                         smoothing_seasonal=0.2 * (1 - 0.3), optimized=False)
        assert_allclose(table["forecast"].to_numpy()[:len(y)], reference.fittedvalues, rtol=1e-9)
        ours, theirs = table["forecast"].to_numpy()[len(y):], np.asarray(reference.forecast(15))
        # At horizon h = m statsmodels reuses the seasonal term of one cycle earlier; the
        # classical forecast a_n + h b_n + s_(n+h-m) uses the latest one, s_n.
        keep = np.arange(15) != M - 1
        assert_allclose(ours[keep], theirs[keep], rtol=1e-9)
        assert ours[M - 1] == pytest.approx(
            reference.level[-1] + M * reference.trend[-1] + reference.season[-1], rel=1e-9)


# ---- estimated parameters ------------------------------------------------------------------


def test_estimated_parameters_minimize_the_explicit_sse(trending, seasonal):
    y = trending["y"].to_numpy()
    s0 = y[:len(y) // 2].mean()
    a0, b0 = first_half_line(y)
    simple = oe.tssmooth(data=trending, y="y", method="exponential")
    best = minimize_scalar(lambda a: np.sum((y - simple_recursion(y, a, s0)[0]) ** 2),
                           bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
    assert simple.attrs["parameters"]["alpha"] == pytest.approx(best.x, abs=1e-6)
    assert simple.attrs["sse"] <= best.fun * (1 + 1e-10)
    assert simple.attrs["estimated"] == ["alpha"] and simple.attrs["at_bounds"] == []

    brown = oe.tssmooth(data=trending, y="y", method="dexponential")
    best = minimize_scalar(lambda a: np.sum((y - brown_recursion(y, a, a0, b0)[0]) ** 2),
                           bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-12})
    assert brown.attrs["parameters"]["alpha"] == pytest.approx(best.x, abs=1e-6)
    assert brown.attrs["sse"] <= best.fun * (1 + 1e-10)

    holt = oe.tssmooth(data=trending, y="y", method="hwinters")
    objective = lambda p: np.sum((y - holt_recursion(y, p[0], p[1], a0, b0)[0]) ** 2)  # noqa: E731
    best = min((minimize(objective, start, bounds=[(0, 1)] * 2, method="L-BFGS-B",
                         options={"ftol": 1e-15, "gtol": 1e-10})
                for start in ([0.3, 0.1], [0.7, 0.5], [0.1, 0.9])), key=lambda r: r.fun)
    assert holt.attrs["sse"] <= best.fun * (1 + 1e-9)
    assert_allclose([holt.attrs["parameters"][k] for k in ("alpha", "beta")], best.x, atol=2e-4)
    partial = oe.tssmooth(data=trending, y="y", method="hwinters", alpha=0.5)   # beta only
    best = minimize_scalar(lambda b: objective([0.5, b]), bounds=(0, 1), method="bounded",
                           options={"xatol": 1e-12})
    assert partial.attrs["parameters"]["alpha"] == 0.5 and partial.attrs["estimated"] == ["beta"]
    assert partial.attrs["parameters"]["beta"] == pytest.approx(best.x, abs=1e-5)

    ys = seasonal["y"].to_numpy()
    for additive in (True, False):
        a0s, b0s, s0s = seasonal_start(ys, M, additive)
        fit = oe.tssmooth(data=seasonal, y="y", method="shwinters", period=M, additive=additive)

        def sse(p, additive=additive, start=(a0s, b0s, s0s)):
            return np.sum((ys - winters_recursion(ys, *p, *start, M, additive)[0]) ** 2)

        best = min((minimize(sse, start, bounds=[(0, 1)] * 3, method="L-BFGS-B",
                             options={"ftol": 1e-15, "gtol": 1e-10})
                    for start in ([0.3, 0.1, 0.1], [0.5, 0.5, 0.5], [0.1, 0.01, 0.3],
                                  [0.9, 0.1, 0.9])), key=lambda r: r.fun)
        assert fit.attrs["sse"] <= best.fun * (1 + 1e-8)
        estimated = [fit.attrs["parameters"][k] for k in ("alpha", "beta", "gamma")]
        assert sse(estimated) == pytest.approx(fit.attrs["sse"], rel=1e-9)
        assert fit.attrs["estimated"] == ["alpha", "beta", "gamma"]
        assert fit.attrs["iterations"] > 0


def test_parameters_on_the_bounds_and_unidentified_parameters(trending):
    walk = pd.DataFrame({"y": np.cumsum(np.random.default_rng(8).normal(size=300)) + 50})
    naive = oe.tssmooth(data=walk, y="y", method="exponential")
    y = walk["y"].to_numpy()
    best = minimize_scalar(lambda a: np.sum((y - simple_recursion(y, a, y[:150].mean())[0]) ** 2),
                           bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
    assert naive.attrs["sse"] <= best.fun * (1 + 1e-10)
    alpha = naive.attrs["parameters"]["alpha"]
    assert (alpha == 1.0 and naive.attrs["at_bounds"] == ["alpha"]) or \
        (alpha == pytest.approx(best.x, abs=1e-5) and naive.attrs["at_bounds"] == [])
    # A deterministic line plus noise around the first-half fit: the level never needs
    # updating, alpha goes to 0 and beta then has no effect on the recursion.
    rng = np.random.default_rng(4)
    line = pd.DataFrame({"y": 3 + 0.5 * np.arange(1, 201) + 0.01 * rng.normal(size=200)})
    frozen = oe.tssmooth(data=line, y="y", method="hwinters", alpha=0.0)
    assert frozen.attrs["parameters"] == {"alpha": 0.0, "beta": 0.0}
    assert frozen.attrs["not_identified"] == ["beta"] and frozen.attrs["estimated"] == ["beta"]
    a0, b0 = first_half_line(line["y"].to_numpy())
    assert_allclose(frozen["forecast"], a0 + b0 * np.arange(1, 201), rtol=1e-9)
    for beta in (0.1, 0.9):                              # beta really is irrelevant at alpha = 0
        other = oe.tssmooth(data=line, y="y", method="hwinters", alpha=0.0, beta=beta)
        assert other.attrs["sse"] == pytest.approx(frozen.attrs["sse"], rel=1e-10)
    seasonal_free = oe.tssmooth(data=pd.DataFrame({"y": np.tile([1.0, 3.0, 2.0, 5.0], 12)
                                                   + 0.1 * rng.normal(size=48)}),
                                y="y", method="shwinters", period=4, alpha=1.0, beta=0.3)
    assert seasonal_free.attrs["not_identified"] == ["gamma"]
    assert seasonal_free.attrs["parameters"]["gamma"] == 0.0


def test_analytic_derivatives_of_every_smoother(seasonal):
    y = torch.from_numpy(seasonal["y"].to_numpy())
    for method, additive in (("exponential", True), ("dexponential", True), ("hwinters", True),
                             ("shwinters", True), ("shwinters", False)):
        model = _Smoother(y, method, M if method == "shwinters" else 1, additive)
        free = list(METHODS[method])
        function, curvature = model.objective(free, {})
        theta = torch.tensor([0.5, 0.9, 0.7][:len(free)], dtype=torch.float64)
        report = check_derivatives(function, theta)
        assert report["gradient_max_rel_error"] < 1e-8, (method, additive)
        hessian = curvature(theta)
        assert hessian.shape == (len(free), len(free))
        assert_allclose(hessian.numpy(), hessian.numpy().T, atol=1e-12)


# ---- time handling, errors, exports -----------------------------------------------------------


def test_time_column_and_future_periods(seasonal):
    shuffled = seasonal.sample(frac=1.0, random_state=5)
    ordered = oe.tssmooth(data=seasonal, y="y", method="hwinters", alpha=0.3, beta=0.1,
                          time="month", forecast=3)
    again = oe.tssmooth(data=shuffled, y="y", method="hwinters", alpha=0.3, beta=0.1,
                        time="month", forecast=3)
    assert_allclose(again["forecast"], ordered["forecast"], rtol=1e-13)
    n = len(seasonal)
    assert ordered["period"].tolist() == list(range(601, 601 + n + 3))    # integer time continues
    dated = seasonal.assign(date=pd.date_range("2010-01-01", periods=n, freq="MS"))
    monthly = oe.tssmooth(data=dated, y="y", method="hwinters", alpha=0.3, beta=0.1, time="date",
                          forecast=2)
    assert monthly["period"].iloc[-1] == pd.Timestamp("2018-02-01")
    assert_allclose(monthly["forecast"], ordered["forecast"].iloc[:n + 2], rtol=1e-13)
    as_columns = oe.tssmooth(data={"y": seasonal["y"].tolist()}, y="y", method="exponential",
                             alpha=0.5)
    assert len(as_columns) == n and "forecast" in str(as_columns.to_latex())


def test_smoothing_error_codes(seasonal):
    holes = seasonal.copy()
    holes.loc[7, "y"] = np.nan
    cases = [
        (dict(method="holt"), "invalid_option"),
        (dict(method="exponential", alpha=1.5), "invalid_parameter"),
        (dict(method="exponential", alpha="a"), "invalid_parameter"),
        (dict(method="exponential", beta=0.2), "invalid_parameter"),          # not used
        (dict(method="hwinters", gamma=0.2), "invalid_parameter"),
        (dict(method="shwinters"), "invalid_option"),                         # period missing
        (dict(method="shwinters", period=1), "invalid_option"),
        (dict(method="hwinters", period=12), "invalid_option"),
        (dict(method="shwinters", period=60), "insufficient_observations"),
        (dict(method="exponential", forecast=-1), "invalid_steps"),
        (dict(method="exponential", forecast=2.5), "invalid_steps"),
        (dict(method="shwinters", period=12, additive="yes"), "invalid_option"),
        (dict(method="exponential", y="nope"), "missing_columns"),
        (dict(method="exponential", data=holes), "missing_values"),
        (dict(method="exponential", data=seasonal.drop(index=[20]), time="month"), "time_gaps"),
        (dict(method="exponential", data=seasonal.assign(y=4.0)), "constant_series"),
        (dict(method="exponential", data=seasonal.iloc[:3]), "insufficient_observations"),
        (dict(method="shwinters", period=12, additive=False,
              data=seasonal.assign(y=seasonal["y"] - 40)), "nonpositive_series"),
        (dict(method="hwinters", data=pd.DataFrame({"y": 2.0 + 3.0 * np.arange(40)})),
         "perfect_fit"),
    ]
    for kwargs, code in cases:
        arguments = {"data": seasonal, "y": "y", **kwargs}
        with pytest.raises(AnalysisError) as error:
            oe.tssmooth(**arguments)
        assert error.value.code == code, (kwargs, code)


def test_tssmooth_export_and_speed():
    assert registry.public_exports()["tssmooth"] == ("openecon.econometrics.arima.smoothing",
                                                     "tssmooth")
    assert "tssmooth" not in registry.names()             # a table function, not an estimator
    assert oe.tssmooth.__doc__ and "Stata" in oe.tssmooth.__doc__
    rng = np.random.default_rng(0)
    n = 300_000
    t = np.arange(1, n + 1)
    y = 100 + 0.001 * t + 5 * np.sin(2 * np.pi * t / M) + np.cumsum(0.05 * rng.normal(size=n)) \
        + rng.normal(size=n)
    frame = pd.DataFrame({"y": y})
    start = time.perf_counter()
    for kwargs in (dict(method="exponential"), dict(method="dexponential"),
                   dict(method="hwinters"), dict(method="shwinters", period=M)):
        table = oe.tssmooth(data=frame, y="y", forecast=12, **kwargs)
        assert len(table) == n + 12 and table.attrs["rmse"] < 3.0
    assert time.perf_counter() - start < 30.0
    # The long series keeps full accuracy: the filtered errors of the seasonal smoother
    # equal the explicit recursion although its error polynomial has 13 roots at the
    # unit circle (alpha, beta and gamma are small here).
    parameters = [table.attrs["parameters"][k] for k in ("alpha", "beta", "gamma")]
    explicit = winters_recursion(y, *parameters, *seasonal_start(y, M, True), M, True)[0]
    assert_allclose(table["forecast"].to_numpy()[:n], explicit, rtol=1e-10, atol=1e-9)
    short = pd.DataFrame({"y": y[:20_000]})
    start = time.perf_counter()
    table = oe.tssmooth(data=short, y="y", method="shwinters", period=M, additive=False)
    assert time.perf_counter() - start < 20.0 and table.attrs["rmse"] < 3.0
