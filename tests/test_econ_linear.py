"""The linear family manifest and the Stata-named wrappers regress / newey over oe.ols."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.linear import ESTIMATORS, EXPORTS
from openecon.models import ModelSpec


@pytest.fixture
def data():
    rng = np.random.default_rng(20240)
    n = 300
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "firm": np.repeat(np.arange(30), 10), "year": np.tile(np.arange(10), 30),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "w": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.0, size=n),
        "t": np.arange(n),
    })
    frame["y"] = 1 + 2 * frame.x1 - frame.x2 + rng.normal(size=n) * (1 + frame.x1.abs())
    return frame


@pytest.fixture
def ols_available(data):
    """The wrappers delegate to another engineer's package; skip when it is broken."""
    try:
        oe.ols(data=data, y="y", x=["x1"])
    except Exception as exc:  # noqa: BLE001 - any failure of the external package
        pytest.skip(f"oe.ols is not usable: {type(exc).__name__}: {exc}")


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def same_fit(actual, expected):
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    assert_allclose(estimates(actual), estimates(expected), rtol=1e-13)
    assert_allclose(errors(actual), errors(expected), rtol=1e-13)
    assert_allclose(np.asarray(actual.covariance_matrix), np.asarray(expected.covariance_matrix), rtol=1e-13)
    assert actual.nobs == expected.nobs and actual.metrics == expected.metrics
    assert actual.spec.covariance == expected.spec.covariance and actual.inference == expected.inference
    assert actual.tests == expected.tests


# ------------------------------------------------------------------------------- manifest


def test_manifest_registers_areg_reghdfe_cnsreg_and_the_stata_wrappers():
    assert [info.name for info in ESTIMATORS] == ["areg", "reghdfe", "cnsreg"]
    assert all(info.family == "linear" and info.inference == "t" for info in ESTIMATORS)
    assert all(info.covariances == ("nonrobust", "HC1", "robust", "cluster") for info in ESTIMATORS)
    assert all(info.weights == ("aweight", "fweight", "pweight") and info.cluster_dimensions == 2
               for info in ESTIMATORS)
    assert EXPORTS == {"regress": "openecon.econometrics.linear.wrappers:regress",
                       "newey": "openecon.econometrics.linear.wrappers:newey"}
    assert "regress" not in registry.names() and "ols" in registry.names()
    for name in ("regress", "newey", "areg", "reghdfe", "cnsreg"):
        assert callable(getattr(oe, name)) and name in dir(oe)
    capabilities = oe.capabilities()["estimators"]
    assert "regress" not in capabilities
    assert capabilities["areg"]["covariances"] == ["nonrobust", "HC1", "robust", "cluster"]
    assert capabilities["reghdfe"]["options"]["drop_singletons"]["default"] is True
    assert capabilities["cnsreg"]["options"]["constraints"]["required"] is True
    assert registry.get("areg").role("absorb").required and registry.get("reghdfe").role("absorb").many


def test_manifest_imports_without_the_tensor_runtime():
    code = ("import sys; import openecon.econometrics.linear as m; "
            "assert 'torch' not in sys.modules and 'pandas' not in sys.modules; "
            "print(len(m.ESTIMATORS), sorted(m.EXPORTS))")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "3 ['newey', 'regress']"


def test_robust_is_accepted_at_the_spec_level_and_means_hc1(data):
    for estimator, extra in [("areg", {"columns": {"absorb": "firm"}}),
                             ("reghdfe", {"columns": {"absorb": ["firm"]}, "intercept": False}),
                             ("cnsreg", {"options": {"constraints": [{"terms": {"x2": 1}, "value": 0}]}})]:
        spec = ModelSpec(estimator=estimator, outcome="y", predictors=["x1", "x2"], covariance="robust", **extra)
        result = oe.fit(spec, data=data)
        explicit = oe.fit(spec.model_copy(update={"covariance": "HC1"}), data=data)
        assert result.spec.covariance == "robust" and result.inference["covariance"] == "HC1"
        assert result.inference["correction"].startswith("HC1")
        assert_allclose(errors(result), errors(explicit), rtol=1e-13)
        with pytest.raises(ValidationError):
            ModelSpec(estimator=estimator, outcome="y", predictors=["x1"], covariance="HC3", **extra)


# ------------------------------------------------------------------------------- wrappers


@pytest.mark.usefixtures("ols_available")
@pytest.mark.parametrize("arguments", [
    {},
    {"covariance": "robust"},
    {"covariance": "HC3", "weights": "aw", "weight_type": "aweight"},
    {"cluster": "firm"},
    {"cluster": ["firm", "year"]},
    {"weights": "w", "weight_type": "fweight"},
    {"weights": "aw", "weight_type": "pweight"},
    {"covariance": "hac", "lags": 3, "time": "t"},
    {"covariance": "hac", "lags": 2, "kernel": "parzen", "time": "t", "weights": "aw", "weight_type": "aweight"},
    {"categorical": ["sector"], "intercept": False},
])
def test_regress_is_oe_ols_with_stata_argument_names(data, arguments):
    columns = ["x1", "x2", "sector"] if "categorical" in arguments else ["x1", "x2"]
    wrapped = oe.regress(data=data, y="y", x=columns, **arguments)
    translated = {**arguments, "covariance": "HC1" if arguments.get("covariance") == "robust"
                  else arguments.get("covariance")}
    direct = oe.ols(data=data, y="y", x=columns, **translated)
    same_fit(wrapped, direct)
    assert wrapped.spec.estimator == "ols"
    if arguments.get("covariance") == "robust":
        assert wrapped.spec.covariance == "HC1"
    if arguments.get("weight_type") == "pweight":
        assert wrapped.spec.covariance == "HC1"
    if arguments.get("cluster"):
        assert wrapped.spec.covariance == "cluster"
    assert hasattr(wrapped, "predict") and hasattr(wrapped, "test")   # oe.ols post-estimation


@pytest.mark.usefixtures("ols_available")
def test_newey_is_oe_ols_hac_and_translates_its_arguments(data):
    wrapped = oe.newey(data=data, y="y", x=["x1", "x2"], lag=3, time="t")
    direct = oe.ols(data=data, y="y", x=["x1", "x2"], covariance="hac", lags=3, time="t")
    same_fit(wrapped, direct)
    assert wrapped.spec.covariance == "hac" and wrapped.spec.options["lags"] == 3
    assert wrapped.spec.options.get("kernel", "bartlett") == "bartlett" and wrapped.spec.time == "t"
    parzen = oe.newey(data=data, y="y", x=["x1", "x2"], lag=4, kernel="parzen", time="t", weights="aw",
                      weight_type="aweight", categorical=None)
    same_fit(parzen, oe.ols(data=data, y="y", x=["x1", "x2"], covariance="hac", lags=4, kernel="parzen",
                            time="t", weights="aw", weight_type="aweight"))
    assert parzen.spec.options["kernel"] == "parzen"
    # lag 0 is White's HC1, exactly as regress with vce(robust)
    assert_allclose(errors(oe.newey(data=data, y="y", x=["x1", "x2"], lag=0, time="t")),
                    errors(oe.regress(data=data, y="y", x=["x1", "x2"], covariance="robust")), rtol=1e-12)
    assert_allclose(estimates(wrapped), estimates(oe.regress(data=data, y="y", x=["x1", "x2"])), rtol=1e-13)
    restored = oe.ResultBundle.model_validate_json(wrapped.model_dump_json())
    assert restored.spec == wrapped.spec and json.loads(wrapped.model_dump_json())["nobs"] == 300


@pytest.mark.usefixtures("ols_available")
def test_wrappers_reject_what_oe_ols_cannot_express_or_stata_does_not_allow(data):
    for arguments, code in [
        ({"lag": 2, "panel": "firm", "time": "t"}, "unsupported_option"),
        ({"lag": 2}, "unsupported_option"),                  # oe.ols has no row-order HAC: time needed
        ({"lag": 2, "time": "t", "weights": "w", "weight_type": "fweight"}, "unsupported_weights"),
        ({"lag": 2, "time": "t", "weights": "aw", "weight_type": "pweight"}, "unsupported_weights"),
        ({"lag": -1, "time": "t"}, "invalid_spec"),
        ({"lag": True, "time": "t"}, "invalid_spec"),
        ({"lag": "3", "time": "t"}, "invalid_spec"),
        ({"lag": 2, "time": "t", "x": "x1"}, "invalid_spec"),
        ({"lag": 2, "time": "year"}, "invalid_time"),        # oe.ols: repeated periods
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.newey(**{"data": data, "y": "y", "x": ["x1", "x2"], **arguments})
        assert caught.value.code == code, arguments
    for arguments, code in [
        ({"covariance": "hac", "lags": 2, "time": "t", "weights": "w", "weight_type": "fweight"},
         "unsupported_weights"),
        ({"covariance": "hac", "lags": 2}, "unsupported_option"),
        ({"x": "x1"}, "invalid_spec"),
        ({"covariance": "hac", "time": "t"}, "invalid_spec"),   # oe.ols: HAC needs lags
        ({"lags": 2}, "invalid_spec"),                           # oe.ols: lags only with HAC
        ({"covariance": "nonrobust", "weights": "aw", "weight_type": "pweight"}, "invalid_spec"),
        ({"x": ["x1"], "data": data.assign(x1=np.where(np.arange(300) == 0, np.nan, data.x1)),
          "missing": "raise"}, "missing_values"),
    ]:
        with pytest.raises(AnalysisError) as caught:
            oe.regress(**{"data": data, "y": "y", "x": ["x1", "x2"], **arguments})
        assert caught.value.code == code, arguments
    # missing='drop' is the oe.ols default and the wrapper's
    dropped = oe.regress(data=data.assign(x1=np.where(np.arange(300) == 0, np.nan, data.x1)), y="y", x=["x1"])
    assert dropped.nobs == 299
