"""Adversarial inputs for the tsmodels family: every case works or raises AnalysisError.

Also the regression tests of the verification pass: scale equivariance of
mswitch and ucm (standardized estimation, zero-variance boundary), constant
outcomes, perfect fits and option bounds.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import make_spec

warnings.filterwarnings("ignore", module="statsmodels")

N = 80


@pytest.fixture(scope="module")
def base():
    rng = np.random.default_rng(7)
    u = np.zeros(N)
    for t in range(1, N):
        u[t] = 0.5 * u[t - 1] + rng.normal()
    x1, x2, q = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    y = 1 + x1 - 0.5 * x2 + np.where(q > 0, 1.0, -1.0) + u
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "q": q, "t": np.arange(N) + 2000})


ESTIMATORS = {
    "prais": lambda d, **k: oe.prais(data=d, y="y", x=["x1"], **k),
    "ardl": lambda d, **k: oe.ardl(data=d, y="y", x=["x1"], **k),
    "ucm": lambda d, **k: oe.ucm(data=d, y="y", model="llevel", **k),
    "mswitch": lambda d, **k: oe.mswitch(data=d, y="y", **k),
    "threshold": lambda d, **k: oe.threshold(data=d, y="y", x=["x1"], threshold_var="q", **k),
}
FILTERS = {method: (lambda d, method=method, **k: oe.tsfilter(d, "y", method=method, **k))
           for method in ("hp", "bk", "cf", "hamilton")}
EVERYTHING = {**ESTIMATORS, **FILTERS}


def code_of(call):
    with pytest.raises(AnalysisError) as info:
        call()
    message = str(info.value)
    assert message and "Traceback" not in message
    return info.value.code


@pytest.mark.parametrize("name", list(EVERYTHING))
def test_tiny_and_empty_samples(base, name):
    call = EVERYTHING[name]
    assert code_of(lambda: call(base.iloc[:0])) in {"empty_data", "empty_sample"}
    for rows in (1, 2, 3):
        if name == "prais" and rows == 3:          # 3 rows, 2 coefficients: rho = 1 here
            assert code_of(lambda: call(base.iloc[:rows])) == "rho_out_of_range"
            continue
        assert code_of(lambda: call(base.iloc[:rows])) == "insufficient_observations"


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_constant_outcome_and_missing_columns(base, name):
    call = ESTIMATORS[name]
    assert code_of(lambda: call(base.assign(y=2.5))) == "constant_outcome"
    assert code_of(lambda: call(base.drop(columns="y"))) == "missing_columns"
    holed = base.assign(y=np.r_[base.y[:40], np.nan, base.y[41:]])
    assert code_of(lambda: call(holed)) == "missing_values"
    if name in ("prais", "ardl", "threshold"):
        # Every row loses its regressor: nothing remains.
        assert code_of(lambda: call(base.assign(x1=np.nan), missing="drop")) == "empty_sample"


@pytest.mark.parametrize("name", list(EVERYTHING))
def test_non_numeric_and_non_finite_values(base, name):
    call = EVERYTHING[name]
    assert code_of(lambda: call(base.assign(y=["a"] * N))) == "non_numeric_column"
    assert code_of(lambda: call(base.assign(y=np.r_[np.inf, base.y[1:]]))) == "non_finite_values"


@pytest.mark.parametrize("name", list(EVERYTHING))
def test_time_structure_errors(base, name):
    call = EVERYTHING[name]
    duplicated = base.copy()
    duplicated.loc[5, "t"] = duplicated.loc[4, "t"]
    assert code_of(lambda: call(duplicated, time="t")) == "repeated_time_values"
    assert code_of(lambda: call(base.assign(t=base.t + 0.5), time="t")) == "invalid_time"
    gapped = base.assign(t=np.r_[np.arange(40), np.arange(41, N + 1)])
    if name == "threshold":                    # the time column only orders the rows
        assert oe.threshold(data=gapped, y="y", x=["x1"], threshold_var="q",
                            time="t").nobs == N
    else:
        assert code_of(lambda: call(gapped, time="t")) == "time_gaps"
    if name in ESTIMATORS and name != "threshold":
        holed = base.assign(y=np.r_[base.y[:40], np.nan, base.y[41:]])
        assert code_of(lambda: call(holed, time="t", missing="drop")) == "time_gaps"


def test_scale_equivariance_of_the_likelihood_estimators(base):
    """Regression: mswitch did not converge on y * 1e-8, ucm on y * 1e8 (verification pass)."""
    rng = np.random.default_rng(3)
    states = np.zeros(300, int)
    for t in range(1, 300):
        states[t] = states[t - 1] if rng.random() < 0.93 else 1 - states[t - 1]
    x = rng.normal(size=300)
    y = np.where(states == 1, 2.0, -1.0) + 0.5 * x + rng.normal(size=300)
    frame = pd.DataFrame({"y": y, "x": x})
    reference = oe.mswitch(data=frame, y="y", x=["x"], varswitch=True)
    for scale in (1e-8, 1e8):
        scaled = oe.mswitch(data=frame.assign(y=y * scale, x=x / scale), y="y", x=["x"],
                            varswitch=True)
        multiplier = np.array([scale, scale, scale ** 2, 1, 1, 1, 1])
        offset = np.array([0, 0, 0, np.log(scale), np.log(scale), 0, 0])
        assert_allclose(coef(scaled), coef(reference) * multiplier + offset, rtol=1e-5,
                        atol=1e-6)
        assert_allclose(ses(scaled), ses(reference) * multiplier, rtol=1e-4)
        assert_allclose(scaled.metrics["log_likelihood"],
                        reference.metrics["log_likelihood"] - 300 * np.log(scale), rtol=1e-9)
        probabilities = oe.mswitch_probabilities(scaled, frame.assign(y=y * scale, x=x / scale))
        assert_allclose(probabilities["smoothed_state2"].to_numpy(),
                        oe.mswitch_probabilities(reference, frame)["smoothed_state2"].to_numpy(),
                        atol=1e-5)
    walk = pd.DataFrame({"y": np.cumsum(rng.normal(size=80))})
    level = oe.ucm(data=walk, y="y", model="llevel")
    for scale in (1e-8, 1e8):
        scaled = oe.ucm(data=walk.assign(y=walk.y * scale), y="y", model="llevel")
        assert scaled.extra["fixed_zero"] == level.extra["fixed_zero"]
        assert_allclose(coef(scaled) / scale ** 2, coef(level), rtol=1e-4)
        # The exact diffuse likelihood excludes the diffuse first period from the Jacobian.
        assert_allclose(scaled.metrics["log_likelihood"],
                        level.metrics["log_likelihood"] - 79 * np.log(scale), rtol=1e-6)


def test_large_offsets_and_tiny_regressors(base):
    shifted = base.assign(y=base.y + 1e6, x1=base.x1 * 1e-3 + 1e2)
    plain = base.assign(x1=base.x1 * 1e-3)
    for name in ("prais", "ardl", "threshold"):
        a, b = ESTIMATORS[name](shifted), ESTIMATORS[name](plain)
        slopes_a = {c.term: c.estimate for c in a.coefficients if "Intercept" not in c.term}
        slopes_b = {c.term: c.estimate for c in b.coefficients if "Intercept" not in c.term}
        assert slopes_a.keys() == slopes_b.keys()
        for term in slopes_a:
            assert_allclose(slopes_a[term], slopes_b[term], rtol=1e-6, atol=1e-9)


def coef(fit):
    return np.array([c.estimate for c in fit.coefficients])


def ses(fit):
    return np.array([c.std_error for c in fit.coefficients])


def test_perfect_fits_and_degenerate_designs(base):
    exact = base.assign(y=1 + 2 * base.x1 + 0.5 * base.x2)
    assert code_of(lambda: oe.prais(data=exact, y="y", x=["x1", "x2"])) == "perfect_fit"
    assert code_of(lambda: oe.threshold(data=exact, y="y", x=["x1", "x2"],
                                        threshold_var="q")) == "perfect_fit"
    assert code_of(lambda: oe.threshold(data=base.assign(q=1.0), y="y", x=["x1"],
                                        threshold_var="q")) == "constant_threshold_variable"
    # A regressor collinear with the constant is dropped with a warning, never silently.
    fit = oe.prais(data=base.assign(c=3.0), y="y", x=["x1", "c"])
    assert [c.term for c in fit.coefficients] == ["Intercept", "x1"]
    assert any("c" in warning and "Omitted" in warning for warning in fit.warnings)
    assert code_of(lambda: oe.ardl(data=base.assign(c=3.0), y="y", x=["c"],
                                   lags=[1, 0])) == "collinear_lags"
    # A binary outcome has no continuous regime structure.
    assert code_of(lambda: oe.mswitch(data=base.assign(y=(base.y > 0) * 1.0), y="y")) in {
        "nonconvergence", "numerical_failure", "degenerate_model"}


def test_option_bounds(base):
    assert code_of(lambda: oe.prais(data=base, y="y", x=["x1"], max_iterations=1)) == \
        "nonconvergence"
    assert code_of(lambda: oe.prais(data=base, y="y", x=["x1"], tolerance=0.0)) == "invalid_option"
    assert code_of(lambda: oe.prais(data=base, y="y", x=["x1"], method="ols")) == "invalid_spec"
    assert code_of(lambda: oe.prais(data=base, y="y", x="x1")) == "invalid_spec"
    assert code_of(lambda: oe.ardl(data=base, y="y", x=["x1"], lags=[1])) == "invalid_lags"
    assert code_of(lambda: oe.ardl(data=base, y="y", x=["x1"], lags=[1, -1])) == "invalid_lags"
    assert code_of(lambda: oe.ardl(data=base, y="y", x=["x1"], maxlags=0)) == "invalid_lags"
    assert code_of(lambda: oe.ardl(data=base, y="y", x=["x1"], trend="none",
                                   restricted=True)) == "invalid_option"
    assert code_of(lambda: oe.ardl(data=base, y="y", x=["x1"], lags=[0, 1],
                                   ec=True)) == "invalid_option"
    assert code_of(lambda: oe.ardl(data=base, y="y", x=["x1"], exog=["x1"])) == "invalid_spec"
    assert code_of(lambda: oe.ardl(data=base, y="y", x=["x1"], maxlags=40)) == \
        "insufficient_observations"
    assert code_of(lambda: oe.threshold(data=base, y="y", x=["x1"], threshold_var="q",
                                        trim=0.5)) == "invalid_option"
    assert code_of(lambda: oe.threshold(data=base, y="y", x=["x1"], threshold_var="q",
                                        trim=0.2, nthresholds=5)) == "insufficient_observations"
    assert code_of(lambda: oe.threshold(data=base, y="y", x=["x1"], threshold_var="q",
                                        regions=["x9"])) == "invalid_option"
    assert code_of(lambda: oe.mswitch(data=base, y="y", states=6)) == "insufficient_observations"
    assert code_of(lambda: oe.mswitch(data=base, y="y", switch=[])) == "invalid_option"
    assert code_of(lambda: oe.mswitch(data=base, y="y", switch=["x9"])) == "invalid_option"
    assert code_of(lambda: oe.ucm(data=base, y="y", model="none")) == "invalid_model"
    assert code_of(lambda: oe.ucm(data=base, y="y", model="llevel", cycle=True,
                                  cycle_frequency=4.0)) == "invalid_option"
    assert code_of(lambda: oe.ucm(data=base, y="y", model="bogus")) == "invalid_spec"
    assert code_of(lambda: oe.tsfilter(base, "y", method="xx")) == "invalid_option"
    assert code_of(lambda: oe.tsfilter(base, "y", method="hp", smooth=0)) == "invalid_option"
    assert code_of(lambda: oe.tsfilter(base, "y", method="bk", minperiod=32,
                                       maxperiod=6)) == "invalid_option"
    assert code_of(lambda: oe.tsfilter(base, "y", method="bk", smaorder=50)) == \
        "insufficient_observations"
    assert code_of(lambda: oe.tsfilter(base, "y", method="hamilton", hamilton_h=0)) == \
        "invalid_option"
    assert code_of(lambda: oe.tsfilter(base, "y", method="hamilton", hamilton_h=74)) == \
        "insufficient_observations"


def test_undeclared_weights_and_covariances_are_rejected():
    for estimator in ("prais", "ardl", "ucm", "mswitch", "threshold"):
        with pytest.raises(AnalysisError) as info:
            make_spec(estimator, outcome="y", predictors=["x1"], weights="w",
                      weight_type="aweight")
        assert info.value.code == "invalid_spec"
        with pytest.raises(AnalysisError):
            make_spec(estimator, outcome="y", predictors=["x1"], covariance="cluster",
                      cluster="g")
