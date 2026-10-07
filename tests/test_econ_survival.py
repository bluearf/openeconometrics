"""Survival family: registry contract, public access, special functions, edge cases and speed.

The estimator-specific oracles live in test_econ_survival_sts.py (Kaplan-Meier and
log-rank family), test_econ_survival_cox.py (stcox, stcurve),
test_econ_survival_streg.py (parametric models) and test_econ_survival_ltable.py.
"""

import time

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import special

import openecon as oe
from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.survival.special import log_gamma_tails
from openecon.models import ModelSpec


def make_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": rng.integers(0, 3, n)})
    frame["t"] = np.round(rng.exponential(np.exp(-0.5 * frame.x)), 2) + 0.01
    frame["d"] = (rng.random(n) < 0.7).astype(float)
    return frame


def test_registry_and_public_access():
    assert {"stcox", "streg"} <= set(registry.names())
    exports = registry.public_exports()
    for name in ("sts", "stcox", "streg", "stcurve", "ltable"):
        assert name in exports and callable(getattr(oe, name))
    info = registry.get("stcox")
    assert info.intercept == "never" and info.inference == "z"
    capabilities = oe.capabilities()
    assert "stcox" in str(capabilities) and "streg" in str(capabilities)
    for function in (oe.sts, oe.stcox, oe.streg, oe.stcurve, oe.ltable):
        assert "Stata" in function.__doc__ and "Example" in function.__doc__


def test_specs_validate_through_the_registry():
    data = make_data()
    spec = ModelSpec(estimator="stcox", outcome="t", predictors=["x"], intercept=False,
                     columns={"failure": "d"}, options={"ties": "efron"})
    direct = fit(spec, data=data)
    assert direct.coefficients[0].estimate == pytest.approx(
        oe.stcox(data=data, time="t", failure="d", x=["x"], ties="efron")
        .coefficients[0].estimate)
    with pytest.raises(ValidationError):
        ModelSpec(estimator="stcox", outcome="t", predictors=["x"], intercept=False,
                  options={"ties": "average"})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="stcox", outcome="t", predictors=["x"])          # has no constant
    with pytest.raises(ValidationError):
        ModelSpec(estimator="streg", outcome="t", predictors=["x"], covariance="HC1")
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=data, time="t", failure="d", x=["x"], ties="average")
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=data, time="t", failure="d", x=["x"], weights="g", weight_type="aweight")
    assert error.value.code == "invalid_spec"


def test_incomplete_gamma_tails_match_scipy():
    for shape in (0.05, 0.4, 1.0, 3.7, 25.0, 400.0, 5000.0, 2e4, 3e5):
        width = 1.5 if shape < 50 else 8 / np.sqrt(shape)
        x = shape * np.exp(np.linspace(-1, 1, 33) * width)
        log_p, log_q = log_gamma_tails(shape, torch.tensor(np.log(x / shape)))
        expected_p = np.log(special.gammainc(shape, x))
        expected_q = np.log(special.gammaincc(shape, x))
        finite = np.isfinite(expected_p) & np.isfinite(expected_q)
        assert_allclose(log_p.numpy()[finite], expected_p[finite], rtol=1e-9, atol=1e-12)
        assert_allclose(log_q.numpy()[finite], expected_q[finite], rtol=1e-9, atol=1e-12)
    far = log_gamma_tails(2.0, torch.tensor([np.log(200.0)], dtype=torch.float64))[1]
    assert float(far) == pytest.approx(np.log(special.gammaincc(2.0, 400.0)), rel=1e-12)


def test_edge_cases_are_handled_or_explained():
    data = make_data()
    tied = oe.stcox(data=data.assign(t=1.0, d=1.0), time="t", failure="d", x=["x"])
    assert tied.coefficients[0].estimate == pytest.approx(0.0, abs=1e-12)
    level = oe.stcox(data=data.assign(x=data.x + 1e8), time="t", failure="d", x=["x"])
    plain = oe.stcox(data=data, time="t", failure="d", x=["x"])
    assert level.coefficients[0].estimate == pytest.approx(plain.coefficients[0].estimate,
                                                           rel=1e-7)
    no_events = data.assign(d=np.where(data.g == 2, 0.0, data.d))
    with pytest.raises(AnalysisError) as error:
        oe.streg(data=no_events, time="t", failure="d", x=["x"], strata="g")
    assert error.value.code == "separation_detected"
    strata = oe.stcox(data=no_events, time="t", failure="d", x=["x"], strata="g")
    assert strata.metrics["n_failures"] == no_events.d.sum()
    with pytest.raises(AnalysisError) as error:
        oe.stcox(data=data.assign(c=1.0), time="t", failure="d", x=["c"])
    assert error.value.code == "no_covariates"
    curve = oe.stcurve(oe.stcox(data=data, time="t", failure="d", x=["x"], strata="g"),
                       data=data)
    assert set(curve.stratum) == {0, 1, 2}
    assert (np.diff(curve[curve.stratum == 0].survivor) <= 0).all()


def test_dense_large_samples_run_in_seconds(monkeypatch):
    # This benchmark measures the original resident-data kernels. Physical
    # Dataset replay validates memory bounds and complete inference separately.
    from openecon.econometrics import streaming_registry
    monkeypatch.setattr(streaming_registry, "supports_spec", lambda spec: False)
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1024")
    rng = np.random.default_rng(1)
    n, k = 200_000, 5
    x = rng.normal(size=(n, k))
    t = rng.exponential(np.exp(-x @ np.linspace(-0.3, 0.3, k)))
    c = rng.exponential(1.0, n)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(k)])
    frame["t"] = np.round(np.minimum(t, c), 3) + 0.001
    frame["d"] = (t <= c).astype(float)
    frame["g"] = rng.integers(0, 3, n)
    start = time.perf_counter()
    oe.sts(frame, "t", failure="d", by="g")
    assert time.perf_counter() - start < 3
    start = time.perf_counter()
    for ties in ("breslow", "efron"):
        oe.stcox(data=frame, time="t", failure="d", x=list(frame.columns[:k]), ties=ties)
    oe.streg(data=frame, time="t", failure="d", x=list(frame.columns[:k]))
    assert time.perf_counter() - start < 20
