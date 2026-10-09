"""Independent admission, calendar, device and persistence boundary checks."""

from copy import deepcopy
import hashlib
import importlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


ssmodel = importlib.import_module("openecon.econometrics.tsworkflows.ssmodel")
sspace = importlib.import_module("openecon.econometrics.tsworkflows.sspace")


def system():
    return {"T": [[0.5]], "Z": [[1.0]], "Q": [[0.2]], "H": [[0.3]],
            "a0": [0.1], "P0": [[0.4]], "c": [0.2], "d": [0.0]}


@pytest.mark.parametrize("key,value", [
    ("Z", [[1.0, 2.0]]), ("Q", [[np.inf]]), ("P0", [[np.nan]]),
    ("H", np.array([[1+2j]])), ("a0", [1j]),
])
def test_invalid_real_matrix_domains_are_refused(key, value):
    with pytest.raises(AnalysisError):
        oe.sspace(data={"y": [0.3, 0.2, 0.1]}, y="y", system={**system(), key: value})


def test_oversized_parameter_schema_refuses_before_tensor_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical conversion began before schema admission")
    monkeypatch.setattr(ssmodel, "finite_tensor", forbidden)
    record = {**system(), "parameters": [
        {"name": f"p{i}", "matrix": "d", "row": 0} for i in range(21)
    ]}
    with pytest.raises(AnalysisError, match="At most20"):
        ssmodel.System(record, 1)


def test_oversized_schedule_refuses_before_numeric_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized path was numerically allocated")
    monkeypatch.setattr(ssmodel, "finite_tensor", forbidden)
    record = {**system(), "schedules": {"T": [[[0.5]]] * 20001}}
    with pytest.raises(AnalysisError, match="20000"):
        ssmodel.System(record, 1)


def test_workspace_refusal_precedes_likelihood_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Likelihood executed after a refused live-buffer plan")
    monkeypatch.setattr(sspace, "kalman", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        oe.sspace(data={"y": np.zeros(4000)}, y="y", system=system())
    assert caught.value.code == "workspace_limit"


@pytest.mark.parametrize("bad", [None, np.inf, -np.inf, 1+2j, "numeric-looking"])
def test_exogenous_missing_nonfinite_or_nonreal_values_never_remove_dates(bad):
    record = {**system(), "exogenous": {
        "state": {"columns": ["x"], "coefficients": [[0.3]]}
    }}
    with pytest.raises(AnalysisError):
        oe.sspace(data={"y": [0.2, None, 0.5], "time": [0, 1, 2], "x": [0.1, bad, 0.2]},
                  y="y", time="time", system=record, missing="mask")


@pytest.mark.parametrize("dates", [[0, 1, 1], [0, 1, 3], [False, True, True]])
def test_invalid_calendars_refuse_before_filtering(dates, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid calendar reached the filter")
    monkeypatch.setattr(sspace, "kalman", forbidden)
    with pytest.raises(AnalysisError):
        oe.sspace(data={"y": [0.2, None, 0.5], "time": dates}, y="y", time="time",
                  system=system(), missing="mask")


def test_monthly_calendar_with_missing_february_preserves_outgoing_transition():
    frame = pd.DataFrame({"date": ["2024-02-01", "2024-04-01", "2024-01-01", "2024-03-01"],
                          "y": [None, 0.5, 1.0, 0.7]}, index=[9, 9, -1, 2])
    before = frame.copy(deep=True)
    record = {**system(), "schedules": {"c": [[0.2], [0.3], [0.4], [0.1]]}}
    result = oe.sspace(data=frame, y="y", time="date", system=record, missing="mask")
    pd.testing.assert_frame_equal(frame, before)
    assert result.sample_positions == [2, 0, 3, 1]
    assert result.extra["periods"] == [f"2024-0{i}-01T00:00:00" for i in range(1, 5)]
    january = 0.1 + 0.4/(0.4+0.3)*(1.0-0.1)
    february = 0.5*january + 0.4
    march = 0.5*february + 0.2
    np.testing.assert_allclose(np.asarray(result.extra["prior_mean"])[1:3, 0], [february, march], atol=1e-13)
    assert result.extra["observed_count"] == [1, 0, 1, 1]
    assert result.dropped_rows == 0 and result.extra["log_likelihood_contributions"][1] == 0


def test_equation_specific_exogenous_order_and_exact_future_saved_paths():
    x, z = np.array([1.0, 2.0, -3.0, 0.5]), np.array([0.3, -1.0, 2.0, 0.0])
    record = {**system(), "T": [[0.0]], "Q": [[0.0]], "P0": [[0.0]], "a0": [0.2],
              "c": [0.1], "d": [0.4], "exogenous": {
                  "state": {"columns": ["z", "x"], "coefficients": [[0.2, -0.5]]},
                  "measurement": {"columns": ["x", "z"], "coefficients": [[0.7, 0.1]]},
              }}
    c = 0.1 + 0.2*z - 0.5*x
    d = 0.4 + 0.7*x + 0.1*z
    expected = np.r_[0.2, c[:-1]] + d
    permutation = [2, 0, 3, 1]
    frame = pd.DataFrame({"time": np.arange(4)[permutation], "y": np.zeros(4),
                          "x": x[permutation], "z": z[permutation]})
    result = oe.sspace(data=frame, y="y", time="time", system=record)
    np.testing.assert_allclose(np.asarray(result.extra["predicted"])[:, 0], expected, atol=1e-13)
    assert result.extra["system_base"]["c"] == [0.1]
    assert result.extra["system_base"]["d"] == [0.4]
    restored = oe.ResultBundle.model_validate_json(result.model_dump_json())
    drivers = pd.DataFrame({"z": [2.0, -1.0], "x": [1.0, 3.0]}, index=[5, 5])
    before = drivers.copy(deep=True)
    future = oe.forecast(restored, 2, future={"exogenous": drivers})
    pd.testing.assert_frame_equal(drivers, before)
    np.testing.assert_allclose(future.forecast, [c[-1]+1.3, 0.0+2.4], atol=1e-13)
    np.testing.assert_allclose(future.attrs["future_paths"]["c"], [[0.0], [-1.6]], atol=1e-13)
    np.testing.assert_allclose(future.attrs["future_paths"]["d"], [[1.3], [2.4]], atol=1e-13)
    json.dumps(future.attrs, allow_nan=False)


def test_accepted_numpy_equations_are_canonical_and_caller_owned_values_unchanged():
    record = system()
    for key in ("Z", "Q", "H", "a0", "P0", "c", "d"):
        record[key] = np.asarray(record[key], dtype=np.float64)
    record["schedules"] = {"d": [np.array([v]) for v in (0.1, 0.2, 0.3)]}
    record["exogenous"] = {
        "measurement": {"columns": ["x"], "coefficients": np.array([[0.4]])}
    }
    before = deepcopy(record)
    result = oe.sspace(data={"y": [0.2, None, 0.7], "x": [1.0, 2.0, 3.0]},
                       y="y", system=record, missing="mask")
    for key in ("Z", "Q", "H", "a0", "P0", "c", "d"):
        np.testing.assert_array_equal(record[key], before[key])
    for raw, old in zip(record["schedules"]["d"], before["schedules"]["d"], strict=True):
        np.testing.assert_array_equal(raw, old)
    np.testing.assert_array_equal(record["exogenous"]["measurement"]["coefficients"], [[0.4]])
    restored = oe.ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.extra["system_template"] == restored.spec.options["system"]
    assert restored.extra["system_template"]["schedules"]["d"] == [[0.1], [0.2], [0.3]]
    np.testing.assert_allclose(restored.extra["system"]["d"], [[0.5], [1.0], [1.5]], atol=1e-13)
    json.dumps(restored.model_dump(mode="json"), allow_nan=False)
    oe.sspace_smooth(restored)


@pytest.fixture(scope="module")
def gaussian_ml():
    observed = np.array([0.2, 1.5, -0.4, 2.1, 0.9, 0.5])
    record = {**system(), "T": [[0.0]], "Q": [[0.0]], "P0": [[0.0]], "a0": [0.0],
              "c": [0.0], "d": [0.0], "H": [[1.0]], "parameters": [
                  {"name": "mean", "matrix": "d", "row": 0},
                  {"name": "variance", "matrix": "H", "row": 0, "col": 0, "transform": "positive"},
              ]}
    result = oe.sspace(data={"y": [0.2, 1.5, None, -0.4, 2.1, 0.9, None, 0.5]},
                       y="y", system=record, missing="mask", alpha=0.1, tolerance=1e-9)
    return result, observed


def test_closed_form_ml_full_stats_survive_json_restore(gaussian_ml):
    result, observed = gaussian_ml
    restored = oe.ResultBundle.model_validate_json(result.model_dump_json())
    mean, variance, n = observed.mean(), observed.var(), len(observed)
    covariance = np.diag([variance/n, 2*variance**2/n])
    np.testing.assert_allclose(restored.covariance_matrix, covariance, rtol=2e-6, atol=2e-8)
    for i, coefficient in enumerate(restored.coefficients):
        beta, se = [mean, variance][i], np.sqrt(covariance[i, i])
        assert coefficient.estimate == pytest.approx(beta, rel=2e-6, abs=2e-8)
        assert coefficient.std_error == pytest.approx(se, rel=2e-6, abs=2e-8)
        assert coefficient.p_value == pytest.approx(2*norm.sf(abs(beta/se)), rel=2e-5, abs=2e-8)
        assert coefficient.ci_low == pytest.approx(beta-norm.isf(0.05)*se, rel=2e-5, abs=2e-8)
        assert coefficient.ci_high == pytest.approx(beta+norm.isf(0.05)*se, rel=2e-5, abs=2e-8)
    assert restored.model_dump(mode="json") == result.model_dump(mode="json")
    assert oe.sspace_smooth(restored).attrs["state_sha256"] == result.extra["state_sha256"]


@pytest.mark.parametrize("field", ["coefficient_se", "coefficient_ci", "covariance", "likelihood", "alpha", "positions", "periods"])
def test_saved_digest_covers_full_scientific_state(gaussian_ml, field):
    result, _ = gaussian_ml
    saved = deepcopy(result.model_dump(mode="json"))
    if field == "coefficient_se":
        saved["coefficients"][0]["std_error"] += 0.01
    elif field == "coefficient_ci":
        saved["coefficients"][0]["ci_high"] += 0.01
    elif field == "covariance":
        saved["covariance_matrix"][0][0] += 0.01
    elif field == "likelihood":
        saved["metrics"]["log_likelihood"] += 0.01
    elif field == "alpha":
        saved["spec"]["alpha"] = 0.2
    elif field == "positions":
        saved["sample_positions"][0], saved["sample_positions"][1] = 1, 0
    else:
        saved["extra"]["periods"][0] = 100
    with pytest.raises(AnalysisError, match="digest"):
        oe.sspace_disturbances(json.dumps(saved))


def test_rehashed_invalid_saved_path_refuses_before_tensor_conversion(gaussian_ml, monkeypatch):
    result, _ = gaussian_ml
    saved = deepcopy(result.model_dump(mode="json"))
    saved["extra"]["system"]["Q"] = [[0.0]*4 for _ in range(4)]
    saved["extra"].pop("state_sha256")
    saved["extra"]["state_sha256"] = hashlib.sha256(json.dumps(
        saved, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()).hexdigest()
    original = sspace.finite_tensor
    def guarded(value, shape=None, name="matrix"):
        if name == "Q":
            raise AssertionError("Malformed saved Q reached unbounded tensor conversion")
        return original(value, shape, name)
    monkeypatch.setattr(sspace, "finite_tensor", guarded)
    with pytest.raises(AnalysisError):
        oe.sspace_smooth(json.dumps(saved))


def test_explicit_cpu_float64_route_survives_ambient_device_and_inference_mode():
    default_device, default_dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        with torch.inference_mode():
            result = oe.sspace(data={"y": [0.2, None, 0.5]}, y="y", system=system(), missing="mask")
            restored = oe.ResultBundle.model_validate_json(result.model_dump_json())
            smooth = oe.sspace_smooth(restored)
            oe.sspace_autocov(restored)
            oe.sspace_disturbances(restored)
            forecast = oe.forecast(restored, 2)
        assert str(torch.get_default_device()) == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert np.isfinite(smooth.state_1).all() and np.isfinite(forecast.std_error).all()
    finally:
        torch.set_default_device(default_device)
        torch.set_default_dtype(default_dtype)
