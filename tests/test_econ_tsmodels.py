"""Family-level contract of tsmodels: manifest, public API, dispatch and failure contract."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry

ESTIMATORS = ["prais", "ardl", "ucm", "mswitch", "threshold", "nardl"]
HELPERS = ["tsfilter", "ucm_components", "ucm_forecast", "mswitch_probabilities", "nardl_multipliers"]


def test_manifest_is_torch_free():
    code = ("import sys, openecon.econometrics.tsmodels as m; "
            "assert 'torch' not in sys.modules and 'pandas' not in sys.modules; "
            "print(len(m.ESTIMATORS))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(len(ESTIMATORS))


def test_registry_and_public_names():
    names = registry.names()
    exports = registry.public_exports()
    for name in ESTIMATORS:
        assert name in names
        info = registry.get(name)
        assert info.family == "tsmodels" and info.function == name
        assert callable(getattr(oe, name))
        assert getattr(oe, name).__doc__ and "Stata" in getattr(oe, name).__doc__
        record = registry.describe(info)
        assert record["family"] == "tsmodels" and record["default_covariance"] in record["covariances"]
    for name in HELPERS:
        assert name in exports and callable(getattr(oe, name))
        assert "Example" in getattr(oe, name).__doc__
    assert registry.forecasters()["ucm"] == ("openecon.econometrics.tsmodels.ucm", "ucm_forecast")
    assert {registry.get(n).inference for n in ("prais", "ardl", "threshold")} == {"t"}
    assert {registry.get(n).inference for n in ("ucm", "mswitch")} == {"z"}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    n = 160
    x = rng.normal(size=n).cumsum() * 0.3
    y = 1 + 0.8 * x + np.convolve(rng.normal(size=n), [1, 0.5], mode="same")
    return pd.DataFrame({"y": y, "x": x, "q": rng.normal(size=n), "t": np.arange(n) + 1})


def test_fit_through_generic_spec(data):
    from openecon.econometrics.core import make_spec

    spec = make_spec("prais", outcome="y", predictors=["x"], time="t", options={"method": "corc"})
    fit = oe.fit(spec, data=data)
    assert fit.spec.estimator == "prais" and fit.nobs == len(data) - 1
    with pytest.raises(AnalysisError) as err:
        make_spec("ardl", outcome="y", predictors=["x"], options={"bogus": 1})
    assert err.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as err:
        make_spec("ucm", outcome="y", intercept=True)
    assert err.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as err:
        make_spec("threshold", outcome="y", predictors=["x"])
    assert err.value.code == "invalid_spec"


def test_every_estimator_round_trips(data):
    fits = [
        oe.prais(data=data, y="y", x=["x"], time="t"),
        oe.ardl(data=data, y="y", x=["x"], maxlags=2, time="t", ec=True),
        oe.ucm(data=data, y="y", x=["x"], model="llevel", time="t"),
        oe.mswitch(data=data, y="y", time="t"),
        oe.threshold(data=data, y="y", x=["x"], threshold_var="q"),
        oe.nardl(data=data, y="y", x=["x"], maxlags=2, time="t"),
    ]
    for fit in fits:
        assert type(fit).model_validate_json(fit.model_dump_json()) == fit
        assert fit.summary() and fit.to_latex()
        assert fit.provenance["family"] == "tsmodels"
        assert fit.provenance["stata_parity_validated"] is False
        assert fit.inference["correction"]
        assert all(np.isfinite(c.std_error) and c.std_error > 0 for c in fit.coefficients)


def test_forecast_dispatch_and_failures(data):
    fit = oe.ucm(data=data, y="y", model="llevel", time="t")
    table = oe.forecast(fit, steps=4)
    assert list(table.period) == [len(data) + i for i in range(1, 5)]
    with pytest.raises(AnalysisError) as err:
        oe.forecast(oe.prais(data=data, y="y", x=["x"]), steps=2)
    assert err.value.code == "unsupported_forecast"
    with pytest.raises(AnalysisError) as err:
        oe.forecast(fit, steps=0)
    assert err.value.code == "invalid_steps"
    with pytest.raises(AnalysisError) as err:
        oe.ucm_components(oe.prais(data=data, y="y", x=["x"]), data)
    assert err.value.code == "invalid_result"
    with pytest.raises(AnalysisError) as err:
        oe.prais(data=data.assign(t=1), y="y", x=["x"], time="t")
    assert err.value.code == "repeated_time_values"
    with pytest.raises(AnalysisError) as err:
        oe.ardl(data=data.assign(t="a"), y="y", x=["x"], time="t")
    assert err.value.code == "invalid_time"
