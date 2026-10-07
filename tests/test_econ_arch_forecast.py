"""oe.arch_forecast / oe.forecast against hand recursions.

The reference continues the oracle's in-sample path (``test_econ_arch_oracle``) with an
explicit loop written from the textbook recursions, with SciPy for the absolute moments
and quantiles of the innovation distributions. Closed forms are checked where they
exist (GARCH mean reversion, IGARCH linear growth).
"""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, stats
from scipy.special import gamma as gamma_function
from test_econ_arch_oracle import ABS_NORMAL, oracle_of, simulate, unpack

import openecon as oe
from openecon.analysis import AnalysisError


def innovation(dist, tau):
    """The fitted unit-variance innovation distribution as a frozen SciPy distribution."""
    if dist == "t":
        nu = 2.0 + math.exp(tau)
        return stats.t(nu, scale=math.sqrt((nu - 2.0) / nu))
    if dist == "ged":
        shape = math.exp(tau)
        return stats.gennorm(shape, scale=math.sqrt(gamma_function(1 / shape)
                                                    / gamma_function(3 / shape)))
    return stats.norm()


def absolute_moment(frozen, power):
    return integrate.quad(lambda v: abs(v) ** power * frozen.pdf(v), -np.inf, np.inf)[0]


def hand_forecast(result, frame, steps, x_future=None, z_future=None):
    """(mean, variance, std_error) forecasts by explicit recursion on the oracle's path."""
    options = result.spec.options
    model = options.get("model", "garch")
    kind = {"arch": "garch", "igarch": "garch", "tarch": "gjr"}.get(model, model)
    dist, archm = options.get("dist", "normal"), options.get("archm")
    theta, loglike = oracle_of(result, frame)
    p = unpack([c.term for c in result.coefficients], theta, model)
    e, h, u = (list(series) for series in loglike.path(theta))
    n = len(e)
    frozen = innovation(dist, p["tau"])
    phi = p["power"]
    variance, mean = [], []
    for step in range(steps):
        t = n + step
        if z_future is not None:
            index = p["het_const"] + float(np.dot(z_future[step], p["het"]))
            w = index if kind == "egarch" else math.exp(index)
        else:
            w = p["omega"]

        def past(lag):
            return t - lag < n                      # an observed period

        if kind == "egarch":
            log_h = w
            for i, c in p["a"].items():
                log_h += c * (e[t - i] / math.sqrt(h[t - i]) if past(i) else 0.0)
            for i, c in p["g"].items():
                size = abs(e[t - i]) / math.sqrt(h[t - i]) if past(i) \
                    else absolute_moment(frozen, 1.0)
                log_h += c * (size - ABS_NORMAL)
            for j, c in p["b"].items():
                log_h += c * math.log(h[t - j])
            ht = math.exp(log_h)
        elif kind == "parch":
            s = w
            for i, c in p["a"].items():
                s += c * (abs(e[t - i]) ** phi if past(i)
                          else absolute_moment(frozen, phi) * h[t - i] ** (phi / 2))
            for j, c in p["b"].items():
                s += c * h[t - j] ** (phi / 2)
            ht = s ** (2 / phi)
        else:
            ht = w
            for i, c in p["a"].items():
                ht += c * (e[t - i] ** 2 if past(i) else h[t - i])
            for i, c in p["g"].items():
                ht += c * ((e[t - i] ** 2 if e[t - i] < 0 else 0.0) if past(i) else h[t - i] / 2)
            for j, c in p["b"].items():
                ht += c * h[t - j]
        ut = sum(c * u[t - j] for j, c in p["ar"].items()) \
            + sum(c * e[t - k] for k, c in p["ma"].items())
        beta = np.asarray(p["beta"])
        if result.spec.intercept:
            regression = beta[0] + (float(np.dot(x_future[step], beta[1:])) if len(beta) > 1
                                    else 0.0)
        else:
            regression = float(np.dot(x_future[step], beta)) if len(beta) else 0.0
        in_mean = {None: 0.0, "variance": ht, "sd": math.sqrt(ht), "log": math.log(ht)}[archm]
        h.append(ht)
        e.append(0.0)
        u.append(ut)
        variance.append(ht)
        mean.append(regression + p["psi"] * in_mean + ut)
    # MA(infinity) weights of the ARMA disturbance.
    weights = [1.0]
    for j in range(1, steps):
        weights.append(p["ma"].get(j, 0.0)
                       + sum(c * weights[j - lag] for lag, c in p["ar"].items() if lag <= j))
    mse = [sum(weights[j] ** 2 * variance[k - j] for j in range(k + 1)) for k in range(steps)]
    critical = frozen.isf(0.025)
    return np.array(mean), np.array(variance), np.sqrt(mse), critical


def check(result, frame, steps=12, x_future=None, z_future=None, exog=None):
    table = oe.arch_forecast(result, steps, exog=exog)
    mean, variance, std_error, critical = hand_forecast(result, frame, steps, x_future, z_future)
    assert list(table.columns) == ["period", "mean_forecast", "variance_forecast", "std_error",
                                   "ci_low", "ci_high"]
    assert_allclose(table["variance_forecast"], variance, rtol=1e-9)
    assert_allclose(table["mean_forecast"], mean, rtol=1e-9, atol=1e-11)
    assert_allclose(table["std_error"], std_error, rtol=1e-9)
    assert_allclose(table["ci_high"], mean + critical * std_error, rtol=1e-7)
    assert_allclose(table["ci_low"], mean - critical * std_error, rtol=1e-7)
    assert table.attrs["critical_value"] == pytest.approx(critical, rel=1e-8)
    return table


@pytest.fixture(scope="module")
def garch():
    frame = simulate(700, 11)
    return frame, oe.arch(data=frame, y="y", time="t")


def test_garch_forecast_reverts_to_the_unconditional_variance(garch):
    frame, result = garch
    table = check(result, frame, steps=400)
    by_term = {c.term: c.estimate for c in result.coefficients}
    persistence = by_term["ARCH:L1.arch"] + by_term["ARCH:L1.garch"]
    long_run = by_term["ARCH:Intercept"] / (1 - persistence)
    first = table["variance_forecast"].iloc[0]
    state = result.extra["state"]
    assert first == pytest.approx(
        by_term["ARCH:Intercept"] + by_term["ARCH:L1.arch"] * state["residuals"][-1] ** 2
        + by_term["ARCH:L1.garch"] * state["variances"][-1], rel=1e-12)
    # Closed form: h(k) = long run + persistence^(k-1) (h(1) - long run).
    steps = np.arange(400)
    assert_allclose(table["variance_forecast"],
                    long_run + persistence ** steps * (first - long_run), rtol=1e-9)
    assert table["variance_forecast"].iloc[-1] == pytest.approx(
        result.metrics["unconditional_variance"], rel=1e-6)
    assert_allclose(table["mean_forecast"], by_term["Intercept"], rtol=1e-12)
    assert list(table["period"][:3]) == [701, 702, 703]
    assert table.attrs["period_unit"] == "time value"
    assert table.attrs["origin"] == "end of the estimation sample"
    assert "conditional expectation" in table.attrs["variance_forecast_definition"]


def test_dispatch_through_oe_forecast_and_alpha(garch):
    frame, result = garch
    direct = oe.arch_forecast(result, 6, alpha=0.1)
    shared = oe.forecast(result, 6, alpha=0.1)
    pd.testing.assert_frame_equal(direct, shared)
    critical = stats.norm.isf(0.05)
    assert_allclose(direct["ci_high"] - direct["mean_forecast"], critical * direct["std_error"],
                    rtol=1e-9)
    assert direct.attrs["confidence_level"] == pytest.approx(0.9)
    assert oe.arch_forecast(result, np.int64(3)).shape[0] == 3


@pytest.mark.parametrize("keywords", [
    dict(model="gjr"),
    dict(model="gjr", dist="t"),
    dict(arch=2, garch=2),
    dict(arch=[1, 3], garch=[2]),
    dict(model="arch", arch=2),
    dict(dist="ged"),
], ids=["gjr", "gjr_t", "garch22", "sparse", "arch2", "ged"])
def test_variance_forecasts_of_the_linear_models(keywords):
    simulation = {key: keywords[key] for key in ("dist",) if key in keywords}
    if keywords.get("model") == "gjr":
        simulation["model"] = "gjr"
    frame = simulate(700, 23, **simulation)
    result = oe.arch(data=frame, y="y", time="t", **keywords)
    check(result, frame)


def test_igarch_variance_forecast_grows_linearly():
    frame = simulate(700, 31)
    result = oe.arch(data=frame, y="y", model="igarch")
    table = check(result, frame, steps=30)
    omega = {c.term: c.estimate for c in result.coefficients}["ARCH:Intercept"]
    assert_allclose(np.diff(table["variance_forecast"]), omega, rtol=1e-8)
    assert list(table["period"][:2]) == [1, 2] and table.attrs["period_unit"] == "steps ahead"


@pytest.mark.parametrize("dist,seed", [("normal", 13), ("t", 14)])
def test_egarch_forecast(dist, seed):
    frame = simulate(700, seed, model="egarch", dist=dist)
    result = oe.arch(data=frame, y="y", time="t", model="egarch", dist=dist)
    table = check(result, frame, steps=40)
    assert "exp(E[ln h])" in table.attrs["variance_forecast_definition"]
    by_term = {c.term: c.estimate for c in result.coefficients}
    if dist == "normal":
        # ln h(k) = omega + b ln h(k-1) beyond the first step.
        logs = np.log(table["variance_forecast"].to_numpy())
        assert_allclose(logs[1:], by_term["ARCH:Intercept"] + by_term["ARCH:L1.egarch"] * logs[:-1],
                        rtol=1e-10, atol=1e-12)
        assert logs[-1] == pytest.approx(result.extra["unconditional_log_variance"], abs=2e-2)


def test_power_arch_forecast():
    frame = simulate(700, 45, model="parch")
    result = oe.arch(data=frame, y="y", x=["x"], model="parch")
    future = np.linspace(-1.0, 1.0, 8)[:, None]
    table = check(result, frame, steps=8, x_future=future, exog={"x": future[:, 0]})
    assert "E[s]^(2/power)" in table.attrs["variance_forecast_definition"]


def test_mean_forecast_with_regressors_arma_and_arch_in_mean():
    frame = simulate(700, 61, ar=0.5, ma=0.3, archm="sd", psi=0.4)
    result = oe.arch(data=frame, y="y", x=["x"], time="t", ar=1, ma=1, archm="sd")
    future = np.array([0.5, -1.0, 0.0, 2.0, 1.0, -0.5])[:, None]
    table = check(result, frame, steps=6, x_future=future,
                  exog=pd.DataFrame({"x": future[:, 0]}))
    # The ARMA part raises the forecast error above the innovation standard deviation.
    assert (table["std_error"].iloc[1:] > np.sqrt(table["variance_forecast"].iloc[1:])).all()
    assert table["std_error"].iloc[0] == pytest.approx(
        math.sqrt(table["variance_forecast"].iloc[0]), rel=1e-12)
    sparse = oe.arch(data=frame, y="y", x=["x"], ar=[1, 3], ma=[2], model="gjr")
    check(sparse, frame, steps=9, x_future=np.zeros((9, 1)), exog={"x": np.zeros(9)})
    in_mean = oe.arch(data=simulate(700, 62, model="egarch", archm="sd", psi=0.4), y="y",
                      model="egarch", archm="sd")
    check(in_mean, simulate(700, 62, model="egarch", archm="sd", psi=0.4), steps=5)


def test_variance_regressors_need_future_values():
    frame = simulate(700, 71, het=0.5)
    result = oe.arch(data=frame, y="y", x=["x"], variance_x=["z"], time="t")
    x_future, z_future = np.array([[0.2], [0.4], [-0.3]]), np.array([[1.0], [0.0], [-1.0]])
    exog = pd.DataFrame({"x": x_future[:, 0], "z": z_future[:, 0]})
    check(result, frame, steps=3, x_future=x_future, z_future=z_future, exog=exog)
    egarch = oe.arch(data=simulate(700, 72, model="egarch", het=0.3), y="y", model="egarch",
                     variance_x=["z"])
    check(egarch, simulate(700, 72, model="egarch", het=0.3), steps=3, z_future=z_future,
          exog={"z": z_future[:, 0]})
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(result, 3)
    assert error.value.code == "missing_exog"
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(result, 3, exog=exog.iloc[:2])
    assert error.value.code == "invalid_exog"
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(result, 3, exog=exog[["x"]])
    assert error.value.code == "missing_columns"
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(result, 3, exog=exog.assign(z=[1.0, np.nan, 2.0]))
    assert error.value.code == "missing_values"


def test_categorical_regressors_in_future_values():
    frame = simulate(700, 81).assign(
        regime=lambda d: np.where(np.arange(len(d)) % 3 == 0, "a",
                                  np.where(np.arange(len(d)) % 3 == 1, "b", "c")))
    result = oe.arch(data=frame, y="y", x=["x", "regime"], categorical=["regime"])
    exog = pd.DataFrame({"x": [0.0, 1.0, 2.0], "regime": ["c", "a", "b"]})
    table = oe.arch_forecast(result, 3, exog=exog)
    by_term = {c.term: c.estimate for c in result.coefficients}
    expected = by_term["Intercept"] + by_term["x"] * exog["x"].to_numpy() + np.array(
        [by_term["regime[c]"], 0.0, by_term["regime[b]"]])
    assert_allclose(table["mean_forecast"], expected, rtol=1e-12)
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(result, 3, exog=exog.assign(regime=["c", "a", "zzz"]))
    assert error.value.code == "invalid_exog"


def test_forecast_from_other_data_runs_the_fitted_model_over_it(garch):
    frame, result = garch
    same = oe.arch_forecast(result, 5, data=frame)
    pd.testing.assert_frame_equal(same, oe.arch_forecast(result, 5), check_exact=False,
                                  rtol=1e-10)
    assert same.attrs["origin"] == "end of the supplied data"
    # A shorter sample: the state is rebuilt with the fitted parameters on that sample.
    shorter = frame.iloc[:500]
    table = oe.arch_forecast(result, 4, data=shorter)
    by_term = {c.term: c.estimate for c in result.coefficients}
    residual = shorter["y"].to_numpy() - by_term["Intercept"]
    h = np.empty(500)
    start = np.mean(residual ** 2)
    previous_e2, previous_h = start, start
    for t in range(500):
        h[t] = by_term["ARCH:Intercept"] + by_term["ARCH:L1.arch"] * previous_e2 \
            + by_term["ARCH:L1.garch"] * previous_h
        previous_e2, previous_h = residual[t] ** 2, h[t]
    first = by_term["ARCH:Intercept"] + by_term["ARCH:L1.arch"] * residual[-1] ** 2 \
        + by_term["ARCH:L1.garch"] * h[-1]
    assert table["variance_forecast"].iloc[0] == pytest.approx(first, rel=1e-10)
    assert list(table["period"]) == [501, 502, 503, 504]
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(result, 2, data=frame.drop(columns=["y"]))
    assert error.value.code == "missing_columns"
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(result, 2, data=frame.iloc[:1])
    assert error.value.code == "insufficient_observations"


def test_datetime_time_columns_forecast_in_steps():
    frame = simulate(500, 91).assign(
        day=lambda d: pd.date_range("2021-01-01", periods=len(d), freq="D"))
    result = oe.arch(data=frame, y="y", time="day")
    table = oe.arch_forecast(result, 3)
    assert list(table["period"]) == [1, 2, 3] and table.attrs["period_unit"] == "steps ahead"


def test_forecast_errors(garch):
    frame, result = garch

    def code(*args, **keywords):
        with pytest.raises(AnalysisError) as error:
            oe.arch_forecast(*args, **keywords)
        return error.value.code

    assert code("not a result", 3) == "invalid_result"
    other = oe.arima(data=frame, y="y", order=(1, 0, 0))
    assert code(other, 3) == "invalid_result"
    for steps in (0, -1, 2.5, "3", True, 10_001):
        assert code(result, steps) == "invalid_steps"
    for alpha in (0.0, 1.0, "0.05", True):
        assert code(result, 3, alpha=alpha) == "invalid_option"
    assert code(result, 3, exog={"x": [1.0, 2.0, 3.0]}) == "invalid_exog"


def test_power_arch_forecast_needs_the_absolute_moment():
    # With Student-t errors E|z|^power does not exist once power >= degrees of freedom.
    frame = simulate(600, 52, model="parch", dist="t", archm="sd", psi=0.3)
    result = oe.arch(data=frame, y="y", x=["x"], model="parch", dist="t", archm="sd")
    exog = {"x": [0.0, 0.0]}
    assert oe.arch_forecast(result, 2, exog=exog).shape[0] == 2
    heavy = result.model_copy(deep=True)
    by_term = {c.term: c for c in heavy.coefficients}
    by_term["/lndfm2"].estimate = math.log(0.5)              # 2.5 degrees of freedom
    by_term["POWER:power"].estimate = 3.0
    with pytest.raises(AnalysisError) as error:
        oe.arch_forecast(heavy, 2, exog=exog)
    assert error.value.code == "non_finite_result"
    one_step = oe.arch_forecast(result, 1, exog={"x": [0.0]})
    assert one_step["variance_forecast"].iloc[0] > 0
