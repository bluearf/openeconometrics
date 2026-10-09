"""Dense joint Gaussian oracles for masked, scheduled proper-prior systems."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load("sspace_scheduled_fixtures", "docs/examples/sspace_scheduled_eight.py")
oracle = load("sspace_joint_gaussian_oracle", "scripts/verify_sspace_scheduled_oracles.py")


def fit_case(*, schedules=False, exogenous=False, shuffle=False, missing=True):
    data, system = fixtures.fixture(schedules=schedules, exogenous=exogenous, shuffle=shuffle, missing=missing)
    frame = pd.DataFrame(data)
    frame.index = [8, 2, 8, -1, 2, 7, 7]
    before, system_before = frame.copy(deep=True), deepcopy(system)
    case = {"data": data, "system": system, "responses": ["y", "other"], "time": "period"}
    result = oe.sspace(data=frame, y="y", responses=["other"], time="period", system=system, missing="mask")
    y, path, order = oracle.ordered_case(case)
    pd.testing.assert_frame_equal(frame, before)
    assert system == system_before
    return result, case, oracle.joint_reference(y, path), order


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("exogenous", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_full_filtered_state_mask_and_likelihood_against_joint_density(scheduled, exogenous, shuffle, missing):
    result, case, expected, order = fit_case(schedules=scheduled, exogenous=exogenous, shuffle=shuffle, missing=missing)
    assert result.nobs == result.nobs_original == 7
    assert result.dropped_rows == 0
    assert result.sample_positions == order.tolist()
    np.testing.assert_array_equal(result.extra["observed_mask"], expected["observed_mask"])
    np.testing.assert_array_equal(result.extra["observed_count"], expected["observed_count"])
    for key in oracle.FILTER_KEYS:
        oracle.compare(result.extra[key], expected[key], key)
    assert result.metrics["log_likelihood"] == pytest.approx(expected["log_likelihood"], abs=2e-10)
    assert sum(result.extra["log_likelihood_contributions"]) == pytest.approx(expected["log_likelihood"], abs=2e-10)
    if missing:
        assert result.extra["log_likelihood_contributions"][2] == pytest.approx(0, abs=1e-14)
        np.testing.assert_allclose(result.extra["filtered"][2], result.extra["prior_mean"][2])
        np.testing.assert_allclose(result.extra["filtered_covariance"][2], result.extra["prior_covariance"][2])
    table = oe.sspace_filter(data=pd.DataFrame(case["data"]), y="y", responses=["other"],
                             time="period", system=case["system"], missing="mask")
    np.testing.assert_allclose(table[["state_1", "state_2"]], expected["filtered"], atol=2e-10)
    assert table.row.tolist() == order.tolist()


@pytest.mark.parametrize("scheduled,exogenous", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("saved_form", ["model", "json", "dict"])
def test_rts_lag_and_every_noise_cross_block_from_saved_state(scheduled, exogenous, saved_form):
    result, _, expected, order = fit_case(schedules=scheduled, exogenous=exogenous, shuffle=True)
    value = result if saved_form == "model" else result.model_dump_json() if saved_form == "json" else result.model_dump(mode="json")
    smooth, lag, noise = oe.sspace_smooth(value), oe.sspace_autocov(value), oe.sspace_disturbances(value)
    for key in oracle.SMOOTH_KEYS:
        oracle.compare(smooth.attrs[key], expected[key], key)
    oracle.compare(lag.attrs["lag_one_covariance"], expected["lag_one_covariance"], "lag orientation")
    for key in oracle.DISTURBANCE_KEYS:
        oracle.compare(noise.attrs[key], expected[key], key)
    np.testing.assert_allclose(smooth[["state_1", "state_2"]], expected["smoothed"], atol=2e-10)
    np.testing.assert_allclose(lag[["cov_1_1", "cov_1_2", "cov_2_1", "cov_2_2"]], expected["lag_one_covariance"].reshape(6, 4), atol=2e-10)
    np.testing.assert_allclose(noise[["process_1", "process_2"]], expected["process_disturbance"], atol=2e-10)
    np.testing.assert_allclose(noise[["measurement_1", "measurement_2"]], expected["measurement_disturbance"], atol=2e-10)
    assert smooth.row.tolist() == noise.row.tolist() == order.tolist()
    assert lag.row.tolist() == order[:-1].tolist()
    assert lag.next_row.tolist() == order[1:].tolist()
    assert np.max(abs(expected["lag_one_covariance"]-expected["lag_one_covariance"].transpose(0, 2, 1))) > 1e-3
    assert abs(expected["measurement_disturbance"][1, 0]) > 1e-3
    assert np.max(abs(expected["process_measurement_covariance"][:-1])) > 1e-3
    np.testing.assert_array_equal(noise.attrs["process_disturbance"][-1], [0, 0])
    np.testing.assert_allclose(noise.attrs["process_disturbance_covariance"][-1], expected["process_disturbance_covariance"][-1], atol=1e-13)


@pytest.mark.parametrize("scheduled,exogenous", [(False, False), (True, False), (False, True), (True, True)])
def test_saved_future_known_path_and_full_forecast_covariance(scheduled, exogenous):
    result, case, expected, _ = fit_case(schedules=scheduled, exogenous=exogenous, shuffle=True)
    future = fixtures.future_fixture()
    if not scheduled:
        future.pop("schedules")
    if not exogenous:
        future.pop("exogenous")
    before = deepcopy(future)
    restored = oe.ResultBundle.model_validate_json(result.model_dump_json())
    table = oe.forecast(restored, 3, future=future, alpha=0.1)
    forecast = oracle.forecast_reference(expected, case["system"], future, 3)
    assert future == before
    for key, value in forecast.items():
        oracle.compare(table.attrs[key], value, key)
    se = np.sqrt(forecast["measurement_covariance"][:, 0, 0])
    np.testing.assert_allclose(table.forecast, forecast["measurement_mean"][:, 0], atol=2e-10)
    np.testing.assert_allclose(table.std_error, se, atol=2e-10)
    np.testing.assert_allclose(table.ci_low, table.forecast-stats.norm.ppf(0.95)*se, atol=2e-9)
    np.testing.assert_allclose(table.ci_high, table.forecast+stats.norm.ppf(0.95)*se, atol=2e-9)
    assert table.attrs["origin_period"] == 6
    if scheduled or exogenous:
        with pytest.raises(AnalysisError):
            oe.forecast(restored, 3)
    if scheduled:
        bad = deepcopy(future)
        del bad["schedules"]["T"]
        with pytest.raises(AnalysisError):
            oe.forecast(restored, 3, future=bad)
        bad = deepcopy(future)
        bad["schedules"]["T"] = bad["schedules"]["T"][:-1]
        with pytest.raises(AnalysisError):
            oe.forecast(restored, 3, future=bad)
    if exogenous:
        bad = deepcopy(future)
        bad["exogenous"]["x"][1] = None
        with pytest.raises(AnalysisError):
            oe.forecast(restored, 3, future=bad)


def test_masked_scheduled_ml_full_observed_information_and_physical_transforms():
    data, system = fixtures.ml_fixture()
    case = {"data": data, "system": system, "responses": ["y"], "time": "period"}
    result = oe.sspace(data=pd.DataFrame(data), y="y", time="period", system=system,
                       missing="mask", tolerance=1e-9)
    beta, covariance = oracle.ml_reference(case)
    actual = [c.estimate for c in result.coefficients]
    np.testing.assert_allclose(actual, beta, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(result.covariance_matrix, covariance, atol=2e-6, rtol=2e-5)
    assert abs(covariance[0, 1]) > 0.01
    assert result.nobs == result.nobs_original == 12 and result.dropped_rows == 0
    assert sum(result.extra["observed_count"]) == 10
    for i, coefficient in enumerate(result.coefficients):
        se = np.sqrt(covariance[i, i])
        assert coefficient.std_error == pytest.approx(se, abs=2e-6, rel=2e-5)
        assert coefficient.p_value == pytest.approx(2*stats.norm.sf(abs(beta[i]/se)), abs=2e-6, rel=2e-5)
    y, path, _ = oracle.ordered_case(case, result.model_dump(mode="json"))
    expected = oracle.joint_reference(y, path)
    smoothed = oe.sspace_smooth(result)
    disturbances = oe.sspace_disturbances(result)
    oracle.compare(smoothed.attrs["smoothed"], expected["smoothed"], "deterministic fitted state")
    np.testing.assert_array_equal(smoothed.attrs["smoothed_covariance"], np.zeros((12, 1, 1)))
    np.testing.assert_array_equal(disturbances.attrs["process_disturbance_covariance"], np.zeros((12, 1, 1)))


@pytest.mark.parametrize("scheduled", [False, True])
def test_all_missing_fixed_system_preserves_calendar_prior_and_zero_likelihood(scheduled):
    data, system = fixtures.fixture(schedules=scheduled)
    data["y"], data["other"] = [None]*7, [None]*7
    result = oe.sspace(data=pd.DataFrame(data), y="y", responses=["other"], time="period",
                       system=system, missing="mask")
    case = {"data": data, "system": system, "responses": ["y", "other"], "time": "period"}
    y, path, _ = oracle.ordered_case(case)
    expected = oracle.joint_reference(y, path)
    assert result.nobs == 7
    assert result.metrics["log_likelihood"] == 0
    np.testing.assert_array_equal(result.extra["observed_count"], np.zeros(7))
    noise = oe.sspace_disturbances(result)
    for key in oracle.DISTURBANCE_KEYS:
        oracle.compare(noise.attrs[key], expected[key], key)


def test_rank_deficient_proper_prior_and_zero_process_noise_use_exact_conditional_moments():
    data, system = fixtures.fixture()
    system.update(P0=[[0.4, 0.2], [0.2, 0.1]], Q=[[0.0, 0.0], [0.0, 0.0]])
    case = {"data": data, "system": system, "responses": ["y", "other"], "time": "period"}
    result = oe.sspace(data=pd.DataFrame(data), y="y", responses=["other"], time="period",
                       system=system, missing="mask")
    y, path, _ = oracle.ordered_case(case)
    expected = oracle.joint_reference(y, path)
    smoothed, noise = oe.sspace_smooth(result), oe.sspace_disturbances(result)
    for key in oracle.SMOOTH_KEYS:
        oracle.compare(smoothed.attrs[key], expected[key], key, atol=2e-11)
    for key in oracle.DISTURBANCE_KEYS:
        oracle.compare(noise.attrs[key], expected[key], key, atol=2e-11)
    np.testing.assert_array_equal(noise.attrs["process_disturbance_covariance"], np.zeros((7, 2, 2)))
    np.testing.assert_array_equal(noise.attrs["process_state_covariance"], np.zeros((7, 2, 2)))
    np.testing.assert_array_equal(noise.attrs["process_measurement_covariance"], np.zeros((7, 2, 2)))


def test_stationary_proper_prior_with_known_measurement_paths_matches_joint_density():
    data, system = fixtures.fixture(schedules=True)
    system["initialization"] = "stationary"
    for key in ("T", "Q", "c"):
        system["schedules"].pop(key)
    result = oe.sspace(data=pd.DataFrame(data), y="y", responses=["other"], time="period",
                       system=system, missing="mask")
    t, q, c = (np.asarray(system[key]) for key in ("T", "Q", "c"))
    reference_system = deepcopy(system)
    reference_system["a0"] = np.linalg.solve(np.eye(2)-t, c).tolist()
    reference_system["P0"] = np.linalg.solve(np.eye(4)-np.kron(t, t), q.reshape(-1)).reshape(2, 2).tolist()
    case = {"data": data, "system": reference_system, "responses": ["y", "other"], "time": "period"}
    y, path, _ = oracle.ordered_case(case)
    expected = oracle.joint_reference(y, path)
    for key in oracle.FILTER_KEYS:
        oracle.compare(result.extra[key], expected[key], key)
    smooth = oe.sspace_smooth(result)
    for key in oracle.SMOOTH_KEYS:
        oracle.compare(smooth.attrs[key], expected[key], key)


def test_finite_horizon_known_transition_growth_uses_its_declared_path():
    data = {"period": list(range(5)), "y": [0.2, None, -0.1, 0.8, 0.4]}
    system = {"Z": [[1.0]], "T": [[0.5]], "Q": [[0.2]], "H": [[0.3]],
              "a0": [0.1], "P0": [[0.4]], "c": [0.0], "d": [0.0],
              "schedules": {"T": [[[value]] for value in [1.1, 1.2, 0.9, 1.3, 1.05]]}}
    result = oe.sspace(data=pd.DataFrame(data), y="y", time="period", system=system, missing="mask")
    case = {"data": data, "system": system, "responses": ["y"], "time": "period"}
    y, path, _ = oracle.ordered_case(case)
    expected = oracle.joint_reference(y, path)
    for key in oracle.FILTER_KEYS:
        oracle.compare(result.extra[key], expected[key], key)
    np.testing.assert_allclose(result.extra["next_covariance"], expected["next_covariance"], atol=2e-10)


@pytest.mark.parametrize("missing", ["raise", "drop"])
def test_default_and_drop_do_not_silently_bridge_missing_interior_dates(missing):
    data, system = fixtures.fixture()
    with pytest.raises(AnalysisError):
        oe.sspace(data=pd.DataFrame(data), y="y", responses=["other"], time="period",
                  system=system, missing=missing)


@pytest.mark.parametrize("role", ["y", "other", "period", "x", "z"])
def test_mask_only_admits_absent_measurements_and_never_nonfinite_or_missing_drivers(role):
    data, system = fixtures.fixture(schedules=True, exogenous=True)
    data[role][1] = float("inf") if role in {"y", "other"} else None
    with pytest.raises(AnalysisError):
        oe.sspace(data=pd.DataFrame(data), y="y", responses=["other"], time="period",
                  system=system, missing="mask")


@pytest.mark.parametrize("change", ["length", "shape", "nonfinite", "negative_q", "asymmetric_h", "scheduled_free"])
def test_schedule_geometry_covariance_and_parameter_guards(change):
    data, system = fixtures.fixture(schedules=True)
    if change == "length":
        system["schedules"]["T"].pop()
    elif change == "shape":
        system["schedules"]["Z"][1] = [[1.0]]
    elif change == "nonfinite":
        system["schedules"]["c"][1][0] = float("inf")
    elif change == "negative_q":
        system["schedules"]["Q"][1][0][0] = -0.2
    elif change == "asymmetric_h":
        system["schedules"]["H"][1][0][1] += 0.2
    else:
        system["parameters"] = [{"name": "variance", "matrix": "H", "row": 0, "col": 0, "transform": "positive"}]
    with pytest.raises(AnalysisError):
        oe.sspace(data=pd.DataFrame(data), y="y", responses=["other"], time="period",
                  system=system, missing="mask")


@pytest.mark.parametrize("helper", [oe.sspace_smooth, oe.sspace_autocov, oe.sspace_disturbances])
def test_saved_state_mutation_is_detected_before_helper_reuse(helper):
    result, _, _, _ = fit_case()
    state = result.model_dump(mode="json")
    state["extra"]["filtered"][0][0] += 0.1
    with pytest.raises((AnalysisError, ValueError)):
        helper(json.dumps(state))


def test_native_example_payload_passes_independent_dense_oracle():
    saved, outputs, inputs = fixtures.run()
    receipt = oracle.verify({"states": saved, "poststates": outputs, "oracle_inputs": inputs})
    assert receipt["eight_outputs_verified"] is True
    assert max(receipt["maximum_absolute_differences"].values()) < 3e-6


def test_cpu_public_calls_preserve_ambient_meta_device_and_inference_mode():
    result, case, expected, _ = fit_case(schedules=True, exogenous=True, shuffle=True)
    previous_device = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        with torch.inference_mode():
            value = oe.sspace(data=pd.DataFrame(case["data"]), y="y", responses=["other"],
                              time="period", system=case["system"], missing="mask")
            np.testing.assert_allclose(value.extra["filtered"], expected["filtered"], atol=2e-10)
            restored = oe.ResultBundle.model_validate_json(value.model_dump_json())
            for helper in (oe.sspace_smooth, oe.sspace_autocov, oe.sspace_disturbances):
                actual = helper(restored)
                pd.testing.assert_frame_equal(actual, helper(result))
            oe.forecast(restored, 3, future=fixtures.future_fixture())
            data, system = fixtures.ml_fixture()
            fitted = oe.sspace(data=pd.DataFrame(data), y="y", time="period", system=system,
                               missing="mask", tolerance=1e-9)
            beta, covariance = oracle.ml_reference({"data": data, "system": system, "responses": ["y"], "time": "period"})
            np.testing.assert_allclose([c.estimate for c in fitted.coefficients], beta, atol=2e-6)
            np.testing.assert_allclose(fitted.covariance_matrix, covariance, atol=2e-6, rtol=2e-5)
            assert torch.get_default_device().type == "meta"
            assert torch.is_inference_mode_enabled()
    finally:
        torch.set_default_device(previous_device)
    assert torch.get_default_device() == previous_device
