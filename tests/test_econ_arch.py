"""oe.arch against brute-force SciPy maximization of an independent likelihood.

The oracle (``test_econ_arch_oracle.oracle_loglike``) is an explicit loop over time
with SciPy's densities. Estimates are compared with ``scipy.optimize`` maximizing that
likelihood from the true parameter values; the observed-information, OPG and robust
covariances with numerically differentiated Hessians and per-observation scores of the
oracle. The file also covers the public API, the failure contract, collinearity
omission, the missing-data policy and the rendering of results.
"""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera as sm_jarque_bera
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from test_econ_arch_oracle import oracle_of, oracle_recursion, simulate, unpack

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.models import ModelSpec

TRUE = {
    "garch": [0.3, 0.5, 0.12, 0.78, 0.1],
    "gjr": [0.3, 0.5, 0.05, 0.15, 0.75, 0.1],
    "egarch": [0.3, 0.5, -0.12, 0.25, 0.85, 0.0],
    "garch_t": [0.3, 0.5, 0.12, 0.78, 0.1, math.log(5.0)],
}
SETTINGS = {
    "garch": dict(model="garch"),
    "gjr": dict(model="gjr"),
    "egarch": dict(model="egarch"),
    "garch_t": dict(model="garch", dist="t"),
}


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def brute_force(loglike, start):
    """Maximize the oracle with SciPy: BFGS on numerical gradients, then a simplex polish."""

    def objective(point):
        value = loglike(point)
        return 1e10 if value is None or not np.isfinite(value).all() else -value.sum()

    first = minimize(objective, start, method="BFGS", options={"gtol": 1e-6})
    polished = minimize(objective, first.x, method="Nelder-Mead",
                        options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 6000, "maxfev": 6000})
    return polished.x, -polished.fun


@pytest.fixture(scope="module")
def frames():
    return {name: simulate(700, seed, model=options["model"], dist=options.get("dist", "normal"))
            for seed, (name, options) in enumerate(SETTINGS.items(), start=11)}


@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_estimates_match_brute_force_maximization(name, frames):
    frame = frames[name]
    result = oe.arch(data=frame, y="y", x=["x"], time="t", **SETTINGS[name])
    theta, loglike = oracle_of(result, frame)
    assert result.metrics["log_likelihood"] == pytest.approx(loglike(theta).sum(), rel=1e-11)
    best, value = brute_force(loglike, np.array(TRUE[name]))
    assert result.metrics["log_likelihood"] >= value - 1e-7
    assert result.metrics["log_likelihood"] == pytest.approx(value, abs=1e-5)
    assert_allclose(theta, best, rtol=2e-3, atol=2e-4)
    # The estimate is a stationary point of the independent likelihood.
    gradient = approx_fprime(theta, lambda point: loglike(point).sum(), centered=True)
    standard_errors = np.array([c.std_error for c in result.coefficients])
    assert np.abs(gradient * standard_errors).max() < 1e-4
    k, n = len(theta), len(frame)
    assert result.nobs == n
    assert result.metrics["aic"] == pytest.approx(-2 * value + 2 * k, abs=1e-4)
    assert result.metrics["bic"] == pytest.approx(-2 * value + k * math.log(n), abs=1e-4)
    assert result.inference["use_t"] is False
    assert result.provenance["stata_parity_validated"] is False
    assert result.provenance["optimizer"]["gradient"].startswith("analytic")


@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_covariances_match_numerical_derivatives_of_the_oracle(name, frames):
    frame = frames[name]
    fits = {kind: oe.arch(data=frame, y="y", x=["x"], time="t", covariance=kind,
                          **SETTINGS[name]) for kind in ("nonrobust", "opg", "robust")}
    theta, loglike = oracle_of(fits["nonrobust"], frame)
    n = len(frame)
    hessian = approx_hess(theta, lambda point: loglike(point).sum())
    scores = approx_fprime(theta, loglike, centered=True)
    bread = np.linalg.inv(-hessian)
    expected = {"nonrobust": bread, "opg": np.linalg.inv(scores.T @ scores),
                "robust": n / (n - 1) * bread @ (scores.T @ scores) @ bread}
    for kind, result in fits.items():
        assert_allclose(estimates(result), theta, rtol=1e-9, atol=1e-10)
        covariance = np.array(result.covariance_matrix)
        scale = np.sqrt(np.outer(np.diag(expected[kind]), np.diag(expected[kind])))
        assert_allclose(covariance / scale, expected[kind] / scale, atol=2e-3)
        assert result.inference["covariance"] == kind
    assert "observed information" in fits["nonrobust"].inference["correction"]
    assert fits["robust"].inference["small_sample_correction"] == pytest.approx(n / (n - 1))
    assert "outer product" in fits["opg"].inference["correction"]


def test_default_covariance_is_the_outer_product_of_gradients(frames):
    # Stata's arch reports OPG standard errors unless vce(oim) or vce(robust) is given.
    result = oe.arch(data=frames["garch"], y="y", x=["x"])
    explicit = oe.arch(data=frames["garch"], y="y", x=["x"], covariance="opg")
    assert result.spec.covariance == "opg" == result.inference["covariance"]
    assert "outer product" in result.inference["correction"]
    assert result.covariance_matrix == explicit.covariance_matrix
    assert result.inference["distribution"] == "normal"
    assert registry.get("arch").default_covariance == "opg"


# ---- public API -------------------------------------------------------------------------------


def test_registry_entry_and_public_access():
    info = registry.get("arch")
    assert info.family == "arch" and info.stata == ("arch",)
    assert info.covariances == ("opg", "nonrobust", "robust") and info.inference == "z"
    assert info.time == "optional" and info.predictors == "optional" and info.weights == ()
    exports = registry.public_exports()
    assert exports["arch"] == ("openecon.econometrics.arch.estimators", "arch")
    assert exports["arch_forecast"][0] == "openecon.econometrics.arch.forecast"
    # The ARCH-LM test is published once, by the arima family.
    assert exports["archlm"][0] == "openecon.econometrics.arima.diagnostics"
    assert registry.forecasters()["arch"] == ("openecon.econometrics.arch.forecast",
                                              "arch_forecast")
    assert callable(oe.arch) and callable(oe.arch_forecast)
    assert "arch" in oe.capabilities()["estimators"]
    for text in ("Stata", "EViews", "Example", "covariance", "model"):
        assert text in oe.arch.__doc__


def test_fit_through_model_spec_equals_the_convenience_function(frames):
    frame = frames["gjr"]
    direct = oe.fit(ModelSpec(estimator="arch", outcome="y", predictors=["x"], time="t",
                              options={"model": "gjr", "arch": [1], "garch": [1]}), data=frame)
    convenient = oe.arch(data=frame, y="y", x=["x"], time="t", model="gjr")
    alias = oe.arch(data=frame, y="y", x=["x"], time="t", model="tarch")
    assert_allclose(estimates(direct), estimates(convenient), rtol=1e-12)
    assert_allclose(estimates(alias), estimates(convenient), rtol=1e-12)
    assert [c.term for c in convenient.coefficients] == [
        "Intercept", "x", "ARCH:L1.arch", "ARCH:L1.tarch", "ARCH:L1.garch", "ARCH:Intercept"]
    assert [c.equation for c in convenient.coefficients] == ["y", "y", "ARCH", "ARCH", "ARCH",
                                                             "ARCH"]
    assert convenient.title == "GJR-GARCH(1,1) regression, normal errors"
    # Defaults: one ARCH and one GARCH lag.
    defaults = oe.fit(ModelSpec(estimator="arch", outcome="y", predictors=["x"]), data=frame)
    assert [c.term for c in defaults.coefficients][2:] == ["ARCH:L1.arch", "ARCH:L1.garch",
                                                           "ARCH:Intercept"]


def test_lag_orders_and_lag_lists(frames):
    frame = frames["garch"]
    by_order = oe.arch(data=frame, y="y", x=["x"], arch=2, garch=1)
    by_list = oe.arch(data=frame, y="y", x=["x"], arch=[2, 1], garch=[1])
    assert by_order.spec.options["arch"] == [1, 2] == by_list.spec.options["arch"]
    assert_allclose(estimates(by_order), estimates(by_list), rtol=1e-12)
    sparse = oe.arch(data=frame, y="y", x=["x"], arch=[1, 3], garch=[2])
    assert [c.term for c in sparse.coefficients][2:] == [
        "ARCH:L1.arch", "ARCH:L3.arch", "ARCH:L2.garch", "ARCH:Intercept"]
    theta, loglike = oracle_of(sparse, frame)
    assert sparse.metrics["log_likelihood"] == pytest.approx(loglike(theta).sum(), rel=1e-11)
    # model="arch" and garch=0 are the same pure ARCH model.
    pure = oe.arch(data=frame, y="y", x=["x"], model="arch", arch=2)
    no_garch = oe.arch(data=frame, y="y", x=["x"], model="garch", arch=2, garch=0)
    assert_allclose(estimates(pure), estimates(no_garch), rtol=1e-9)
    assert pure.title == "ARCH(2) regression, normal errors"
    assert [c.term for c in pure.coefficients][2:] == ["ARCH:L1.arch", "ARCH:L2.arch",
                                                       "ARCH:Intercept"]


def test_no_constant_and_no_regressors(frames):
    frame = frames["garch"].assign(centered=lambda d: d["y"] - d["y"].mean())
    result = oe.arch(data=frame, y="centered", constant=False)
    assert [c.term for c in result.coefficients] == ["ARCH:L1.arch", "ARCH:L1.garch",
                                                     "ARCH:Intercept"]
    assert "model" not in result.tests
    theta, loglike = oracle_of(result, frame)
    assert result.metrics["log_likelihood"] == pytest.approx(loglike(theta).sum(), rel=1e-11)
    with_constant = oe.arch(data=frame, y="centered")
    assert [c.term for c in with_constant.coefficients][0] == "Intercept"
    assert with_constant.metrics["log_likelihood"] >= result.metrics["log_likelihood"]


def test_metrics_tests_and_extra(frames):
    frame = frames["garch"]
    result = oe.arch(data=frame, y="y", x=["x"], time="t")
    theta = unpack([c.term for c in result.coefficients], estimates(result), "garch")
    persistence = theta["a"][1] + theta["b"][1]
    assert list(result.metrics) == ["log_likelihood", "aic", "bic", "persistence",
                                    "unconditional_variance", "iterations"]
    assert result.metrics["persistence"] == pytest.approx(persistence, rel=1e-12)
    assert result.metrics["unconditional_variance"] == pytest.approx(
        theta["omega"] / (1 - persistence), rel=1e-12)
    # Wald test of the slope.
    slope = result.coefficients[1]
    assert result.tests["model"]["statistic"] == pytest.approx(slope.statistic ** 2, rel=1e-9)
    assert result.tests["model"]["df"] == 1 and result.tests["model"]["distribution"] == "chi2"
    # Diagnostics of the standardized residuals against statsmodels.
    y, x = frame["y"].to_numpy(), np.c_[np.ones(len(frame)), frame["x"].to_numpy()]
    e, h, _ = oracle_recursion(theta, y, x)
    z = e / np.sqrt(h)
    lags = result.tests["ljung_box"]["lags"]
    assert lags == min(len(z) // 2 - 2, 40)
    reference = acorr_ljungbox(z, lags=[lags])
    assert result.tests["ljung_box"]["statistic"] == pytest.approx(
        reference["lb_stat"].iloc[0], rel=1e-8)
    assert result.tests["ljung_box"]["p_value"] == pytest.approx(
        reference["lb_pvalue"].iloc[0], rel=1e-7)
    reference = acorr_ljungbox(z ** 2, lags=[lags])
    assert result.tests["ljung_box_squared"]["statistic"] == pytest.approx(
        reference["lb_stat"].iloc[0], rel=1e-8)
    lm, lm_p, _, _ = het_arch(z, nlags=5)
    assert result.tests["arch_lm_residuals"]["statistic"] == pytest.approx(lm, rel=1e-8)
    assert result.tests["arch_lm_residuals"]["p_value"] == pytest.approx(lm_p, rel=1e-7)
    jb, jb_p, _, _ = sm_jarque_bera(z)
    assert result.tests["jarque_bera"]["statistic"] == pytest.approx(jb, rel=1e-9)
    # Predictions are the conditional mean; the stored state is the end of the sample.
    assert len(result.predictions) == 400
    first = result.predictions[0]
    assert first["residual"] == pytest.approx(e[first["row"]], rel=1e-9)
    state = result.extra["state"]
    assert_allclose(state["residuals"], e[-1:], rtol=1e-10)
    assert_allclose(state["variances"], h[-1:], rtol=1e-10)
    assert state["last_period"] == 700
    assert state["presample_variance"] == pytest.approx(np.mean(e ** 2), rel=1e-12)
    assert_allclose(result.extra["conditional_variance_tail"], h[-400:], rtol=1e-10)
    assert result.extra["model"]["label"] == "GARCH(1,1)"
    assert result.extra["distribution"] == {"name": "normal"}
    shorter = oe.arch(data=frame, y="y", x=["x"], test_lags=3)
    assert shorter.tests["ljung_box"]["lags"] == 3 == shorter.tests["arch_lm_residuals"]["lags"]


def test_distribution_parameters_are_reported_on_the_log_scale(frames):
    frame = frames["garch_t"]
    result = oe.arch(data=frame, y="y", x=["x"], dist="t", alpha=0.1)
    last = result.coefficients[-1]
    assert last.term == "/lndfm2" and last.equation is None
    record = result.extra["distribution"]
    assert record["name"] == "t"
    assert record["df"] == pytest.approx(2 + math.exp(last.estimate), rel=1e-12)
    assert record["df_ci"] == pytest.approx([2 + math.exp(last.ci_low),
                                             2 + math.exp(last.ci_high)], rel=1e-9)
    assert result.title == "GARCH(1,1) regression, Student t errors"
    ged = oe.arch(data=simulate(600, 5, dist="ged"), y="y", x=["x"], dist="ged")
    assert ged.coefficients[-1].term == "/lnshape"
    assert ged.extra["distribution"]["shape"] == pytest.approx(
        math.exp(ged.coefficients[-1].estimate), rel=1e-12)
    theta, loglike = oracle_of(ged, simulate(600, 5, dist="ged"))
    assert ged.metrics["log_likelihood"] == pytest.approx(loglike(theta).sum(), rel=1e-11)


def test_json_round_trip_summary_and_latex(frames):
    result = oe.arch(data=frames["egarch"], y="y", x=["x"], model="egarch", dist="normal")
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    for piece in ("EGARCH(1,1) regression, normal errors", "[ARCH]", "ARCH:L1.earch_a",
                  "ARCH:L1.egarch", "log_likelihood", "persistence", "Ljung-Box"):
        assert piece in text
    latex = result.to_latex()
    assert "earch" in latex and "tabular" in latex
    assert result.summary(format="latex") == latex
    assert result.extra["unconditional_log_variance"] == pytest.approx(
        result.coefficients[-1].estimate / (1 - result.metrics["persistence"]), rel=1e-12)
    assert result.metrics["unconditional_variance"] is None


# ---- sample handling --------------------------------------------------------------------------


def test_rows_are_sorted_by_time_and_datetimes_are_accepted(frames):
    frame = frames["garch"]
    reference = oe.arch(data=frame, y="y", x=["x"], time="t")
    shuffled = frame.sample(frac=1.0, random_state=3)
    result = oe.arch(data=shuffled, y="y", x=["x"], time="t")
    assert_allclose(estimates(result), estimates(reference), rtol=1e-12)
    assert result.provenance["sample_order"] == "sorted by t"
    without_time = oe.arch(data=frame, y="y", x=["x"])
    assert_allclose(estimates(without_time), estimates(reference), rtol=1e-12)
    assert without_time.extra["state"]["last_period"] is None
    dated = frame.assign(day=pd.date_range("2020-01-01", periods=len(frame), freq="D"))
    result = oe.arch(data=dated, y="y", x=["x"], time="day")
    assert_allclose(estimates(result), estimates(reference), rtol=1e-12)
    assert any("consecutive periods" in warning for warning in result.warnings)


def test_missing_data_policy(frames):
    frame = frames["garch"].copy()
    frame.loc[[0, 1, 699], "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=frame, y="y", x=["x"], time="t")
    assert error.value.code == "missing_values"
    dropped = oe.arch(data=frame, y="y", x=["x"], time="t", missing="drop")
    trimmed = oe.arch(data=frame.iloc[2:699], y="y", x=["x"], time="t")
    assert dropped.nobs == 697 and dropped.dropped_rows == 3
    assert_allclose(estimates(dropped), estimates(trimmed), rtol=1e-12)
    assert dropped.sample_positions[0] == 2
    assert any("Excluded 3 observation(s)" in warning for warning in dropped.warnings)
    frame.loc[300, "x"] = np.nan                       # a hole inside the series
    for time in ("t", None):
        with pytest.raises(AnalysisError) as error:
            oe.arch(data=frame, y="y", x=["x"], time=time, missing="drop")
        assert error.value.code == "time_gaps"


def test_collinear_regressors_are_omitted_and_recorded(frames):
    frame = frames["garch"].assign(twice=lambda d: 2 * d["x"], zcopy=lambda d: d["z"] + 1.0)
    reference = oe.arch(data=frame, y="y", x=["x"], variance_x=["z"])
    result = oe.arch(data=frame, y="y", x=["x", "twice"], variance_x=["z", "zcopy"])
    assert result.provenance["omitted_terms"] == ["twice", "HET:zcopy"]
    assert any("Omitted because of collinearity: twice" in w for w in result.warnings)
    assert [c.term for c in result.coefficients] == [c.term for c in reference.coefficients]
    assert_allclose(estimates(result), estimates(reference), rtol=1e-9)


def test_results_are_equivariant_to_the_units_of_the_outcome(frames):
    frame = frames["gjr"]
    base = oe.arch(data=frame, y="y", x=["x"], model="gjr", covariance="robust")
    scaled = oe.arch(data=frame.assign(y=lambda d: d["y"] / 250.0), y="y", x=["x"], model="gjr",
                     covariance="robust")
    factor = np.array([1 / 250, 1 / 250, 1, 1, 1, 1 / 250 ** 2])
    assert_allclose(estimates(scaled), estimates(base) * factor, rtol=2e-6)
    assert_allclose([c.std_error for c in scaled.coefficients],
                    np.array([c.std_error for c in base.coefficients]) * factor, rtol=2e-5)
    assert scaled.metrics["log_likelihood"] == pytest.approx(
        base.metrics["log_likelihood"] + len(frame) * math.log(250.0), rel=1e-10)
    assert scaled.metrics["persistence"] == pytest.approx(base.metrics["persistence"], rel=1e-6)


def test_categorical_regressors_in_both_equations(frames):
    frame = frames["garch"].assign(
        regime=lambda d: np.where(np.arange(len(d)) % 3 == 0, "a",
                                  np.where(np.arange(len(d)) % 3 == 1, "b", "c")))
    result = oe.arch(data=frame, y="y", x=["x", "regime"], variance_x=["regime"],
                     categorical=["regime"])
    terms = [c.term for c in result.coefficients]
    assert terms == ["Intercept", "x", "regime[b]", "regime[c]", "HET:regime[b]", "HET:regime[c]",
                     "HET:Intercept", "ARCH:L1.arch", "ARCH:L1.garch"]
    assert result.provenance["categorical_encoding"]["regime"]["reference"] == "a"
    indicators = pd.get_dummies(frame["regime"], dtype=float)[["b", "c"]]
    numeric = pd.concat([frame, indicators], axis=1)
    reference = oe.arch(data=numeric, y="y", x=["x", "b", "c"], variance_x=["b", "c"])
    assert_allclose(estimates(result), estimates(reference), rtol=1e-9)


# ---- failure contract -------------------------------------------------------------------------


def error_code(**keywords):
    with pytest.raises(AnalysisError) as error:
        oe.arch(**keywords)
    return error.value.code


def test_specification_errors(frames):
    frame = frames["garch"]
    base = dict(data=frame, y="y", x=["x"])
    assert error_code(**base, model="figarch") == "invalid_spec"
    assert error_code(**base, dist="cauchy") == "invalid_spec"
    assert error_code(**base, archm="cube") == "invalid_spec"
    assert error_code(**base, covariance="cluster") == "invalid_spec"
    assert error_code(**base, covariance="hac") == "invalid_spec"
    assert error_code(**base, model="arch", garch=1) == "invalid_spec"
    assert error_code(**base, model="igarch", garch=0) == "invalid_spec"
    assert error_code(**base, arch=0) == "invalid_spec"
    assert error_code(**base, arch=-1) == "invalid_lags"
    assert error_code(**base, arch=[1, 1]) == "invalid_lags"
    assert error_code(**base, garch=[0]) == "invalid_lags"
    assert error_code(**base, ar=1.5) == "invalid_lags"
    assert error_code(**base, ma="1") == "invalid_lags"
    assert error_code(**base, arch=500) == "invalid_lags"
    assert error_code(data=frame, y="y", x="x") == "invalid_spec"
    assert error_code(**base, variance_x="z") == "invalid_spec"
    assert error_code(**base, constant="yes") == "invalid_spec"
    assert error_code(**base, alpha=1.5) == "invalid_spec"
    assert error_code(**base, missing="ignore") == "invalid_spec"
    assert error_code(**base, max_iterations=0) == "invalid_spec"
    assert error_code(**base, tolerance=-1.0) == "invalid_option"
    assert error_code(**base, time="y") == "invalid_spec"
    assert error_code(**base, time="x") == "invalid_spec"
    assert error_code(**base, variance_x=["y"]) == "invalid_spec"
    assert error_code(data=frame, y="y", x=["nope"]) == "missing_columns"
    assert error_code(data=frame.iloc[:0], y="y", x=["x"]) == "empty_data"
    # A ModelSpec built directly keeps pydantic's ValidationError.
    for options in ({"model": "figarch"}, {"arch": 1}, {"lags": 3}):
        with pytest.raises(ValidationError):
            ModelSpec(estimator="arch", outcome="y", options=options)
    with pytest.raises(ValidationError):
        ModelSpec(estimator="arch", outcome="y", weights="x", weight_type="aweight")
    with pytest.raises(ValidationError):
        ModelSpec(estimator="arch", outcome="y", cluster="x")
    with pytest.raises(AnalysisError) as error:
        oe.fit(ModelSpec(estimator="arch", outcome="y", options={"arch": [0]}), data=frame)
    assert error.value.code == "invalid_lags"


def test_data_errors(frames):
    frame = frames["garch"]
    gaps = frame.drop(index=[350]).reset_index(drop=True)
    assert error_code(data=gaps, y="y", x=["x"], time="t") == "time_gaps"
    repeated = frame.assign(t=lambda d: d["t"].where(d["t"] != 5, 4))
    assert error_code(data=repeated, y="y", x=["x"], time="t") == "repeated_time_values"
    fractional = frame.assign(t=lambda d: d["t"] + 0.5)
    assert error_code(data=fractional, y="y", x=["x"], time="t") == "invalid_time"
    labelled = frame.assign(t=lambda d: d["t"].astype(str))
    assert error_code(data=labelled, y="y", x=["x"], time="t") == "invalid_time"
    assert error_code(data=frame.iloc[:8], y="y", x=["x"]) == "insufficient_observations"
    assert error_code(data=frame.assign(y=1.0), y="y", x=["x"]) == "constant_outcome"
    assert error_code(data=frame.assign(y=lambda d: 1 + 2 * d["x"]), y="y", x=["x"]) \
        == "perfect_fit"
    assert error_code(data=frame.assign(x="a"), y="y", x=["x"]) == "non_numeric_column"
    infinite = frame.assign(x=lambda d: d["x"].where(d.index != 3, np.inf))
    assert error_code(data=infinite, y="y", x=["x"]) == "non_finite_values"


def test_normal_data_with_t_errors_reports_diverging_degrees_of_freedom():
    # The t likelihood of Gaussian data keeps rising in the degrees of freedom.
    frame = simulate(1500, 3)
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=frame, y="y", x=["x"], dist="t")
    assert error.value.code == "nonconvergence"
    assert "dist='normal'" in str(error.value)
