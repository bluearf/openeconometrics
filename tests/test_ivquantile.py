"""Scalar iid IVQR numerical, persistence and explicit refusal contracts."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.ivquantile import ivqreg, ivqreg_predict, ivqreg_restore
from openecon.econometrics.ivquantile.common import seal
from openecon.econometrics.summary_state import summary_state
from openecon.resources import use_workspace_budget


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def sample(seed=627, n=241, q=1):
    generator = np.random.default_rng(seed)
    x, z, u, e, z2 = generator.normal(size=(5, n))
    d = 1.5 * z + 0.6 * x + u + (0.4 * z2 if q == 2 else 0)
    y = 0.7 + 1.1 * d - 0.4 * x + 0.5 * u + e
    return pd.DataFrame(dict(y=y, d=d, x=x, z=z, z2=z2))


def run(data=None, **options):
    args = dict(data=sample() if data is None else data, y="y", endogenous="d",
                instruments=["z"], x=["x"], grid=np.linspace(-0.5, 2.5, 31).tolist())
    args.update(options)
    return ivqreg(**args)


@pytest.fixture(scope="module")
def fitted():
    return run(inference="identified")


@pytest.fixture(scope="module")
def tiny_outcome_fitted():
    data = sample()
    data["y"] *= 1e-6
    return run(data, inference="identified", grid=(np.linspace(-.5, 2.5, 31) * 1e-6).tolist(),
               bandwidth=.65e-6, refinement_tolerance=1e-10)


def test_weak_mode_has_no_structural_wald(fitted):
    value = run()
    assert value["coefficients"]["std_error"].isna().all()
    assert value.attrs["state"]["joint"] is None
    assert "joint_covariance" not in value
    assert value.attrs["weak_id_df"] == 1
    assert value.attrs["outer_tails"] == "unknown"
    assert value.attrs["between_grid_points"].startswith("untested")
    assert value.attrs["vendor_parity_validated"] is False
    assert fitted["coefficients"]["std_error"].gt(0).all()


def test_complete_state_replays_without_optimizer(fitted, monkeypatch):
    from openecon.econometrics.quantile import kernels
    monkeypatch.setattr(kernels, "solve", lambda *a, **k: pytest.fail("Refitted QR"))
    for input_value in (fitted, fitted.attrs["state"], json.dumps(fitted.attrs["state"]), summary_state(fitted)):
        value = ivqreg_restore(input_value)
        assert value.attrs["state"] == fitted.attrs["state"]
        for name in fitted:
            pd.testing.assert_frame_equal(value[name], fitted[name])
    assert "weak\\_id\\_grid" in fitted.to_latex()


@pytest.mark.parametrize("field", ["coefficients", "basis", "dual_basis", "bread", "covariance", "statistic",
                                 "p_value", "criterion", "qr_objective", "bandwidth", "density_support",
                                 "structural_moments", "qr_iterations", "qr_pivots", "qr_gap"])
def test_rehashed_profile_corruption_is_refused(fitted, field):
    state = copy.deepcopy(fitted.attrs["state"])
    record = state["evaluations"][5]
    if field == "basis":
        record[field][0] = record[field][1]
    elif field in {"qr_iterations", "qr_pivots"}:
        record[field] = 10000
    elif field == "qr_gap":
        record[field] = -1
    elif isinstance(record[field], list):
        if isinstance(record[field][0], list):
            record[field][0][0] += 0.5
        else:
            record[field][0] += 0.5
    else:
        record[field] += 1
    state.pop("sha256")
    with pytest.raises(AnalysisError):
        ivqreg_restore(seal(state))


@pytest.mark.parametrize("field", ["sample_positions", "terms", "selected_evaluation", "accepted", "runs", "search", "joint"])
def test_rehashed_selection_and_joint_corruption_is_refused(fitted, field):
    state = copy.deepcopy(fitted.attrs["state"])
    if field == "sample_positions":
        state[field][0] = 1
    elif field == "terms":
        state[field][0] = "y"
    elif field == "selected_evaluation":
        state[field] = 0
    elif field == "accepted":
        state[field][0] = not state[field][0]
    elif field == "runs":
        state[field] = []
    elif field == "search":
        state[field]["coarse_selected"] = 0
    else:
        state[field]["covariance"][0][1] += 0.5
    state.pop("sha256")
    with pytest.raises(AnalysisError):
        ivqreg_restore(seal(state))


@pytest.mark.parametrize("target,entry", [("profile", (0, 0)), ("profile", (0, 1)),
                                         ("joint", (0, 0)), ("joint", (0, 1))])
def test_tiny_covariance_rehashed_relative_corruption_refuses(tiny_outcome_fitted, target, entry, monkeypatch):
    from openecon.econometrics.quantile import kernels
    monkeypatch.setattr(kernels, "solve", lambda *a, **k: pytest.fail("Refitted QR"))
    restored = ivqreg_restore(tiny_outcome_fitted.attrs["state"])
    assert restored.attrs["state"] == tiny_outcome_fitted.attrs["state"]
    state = copy.deepcopy(tiny_outcome_fitted.attrs["state"])
    covariance = state["evaluations"][5]["covariance"] if target == "profile" else state["joint"]["covariance"]
    row, col = entry
    assert 0 < abs(covariance[row][col]) < 1e-10
    covariance[row][col] *= 1.01
    covariance[col][row] = covariance[row][col]
    state.pop("sha256")
    with pytest.raises(AnalysisError, match="covariance fails semantic"):
        ivqreg_restore(seal(state))


def test_tiny_dimensionful_profile_criterion_rehash_refuses(tiny_outcome_fitted):
    state = copy.deepcopy(tiny_outcome_fitted.attrs["state"])
    assert 0 < state["evaluations"][5]["criterion"] < 1e-10
    state["evaluations"][5]["criterion"] *= 1.01
    state.pop("sha256")
    with pytest.raises(AnalysisError, match="criterion fails semantic"):
        ivqreg_restore(seal(state))


def test_native_kernel_replay_allocations_preserve_ambient_meta_device_and_defaults(fitted):
    from openecon.econometrics.ivquantile.common import geometry
    from openecon.econometrics.ivquantile.kernels import certificate, identified_covariance
    state = fitted.attrs["state"]
    old_dtype, rng = torch.get_default_dtype(), torch.random.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            y, d, x, _, w, a, _ = geometry(sample(), state["spec"], state["config"])
            selected = state["evaluations"][state["selected_evaluation"]]
            beta, residual, dual, _ = certificate(w, y-d*selected["alpha"], selected["coefficients"],
                                                 selected["basis"], state["config"]["quantile"],
                                                 witness=selected["dual_basis"])
            joint = identified_covariance(y, d, x, w, a, selected, state["config"])
            assert beta.device.type == residual.device.type == dual.device.type == "cpu"
            assert beta.dtype == residual.dtype == dual.dtype == torch.float64
            assert joint == state["joint"]
            assert torch.empty(0).device.type == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(torch.random.get_rng_state(), rng)
    finally:
        torch.set_default_dtype(old_dtype)


def test_missing_sample_dtype_and_typed_index_are_exact():
    frame = sample()
    frame.index = pd.MultiIndex.from_arrays([np.arange(len(frame)) % 3, [f"r{i}" for i in range(len(frame))]],
                                           names=["group", "physical"])
    frame["x"] = frame["x"].astype("Float64")
    frame.loc[frame.index[2], "x"] = pd.NA
    with pytest.raises(AnalysisError, match="Missing"):
        run(frame)
    value = run(frame, missing="drop")
    assert 2 not in value.attrs["state"]["sample_positions"]
    assert len(value.attrs["state"]["sample_positions"]) == 240
    restored = ivqreg_restore(value)
    assert restored.attrs["state"]["source"] == value.attrs["state"]["source"]
    query = frame.iloc[:6].copy()
    prediction = ivqreg_predict(result=restored, data=query, missing="drop")
    assert prediction["predictions"].index.equals(query.index[[0, 1, 3, 4, 5]])
    assert prediction.attrs["sample_positions"] == [0, 1, 3, 4, 5]


def test_prediction_joint_uncertainty_retains_cross_rows(fitted):
    query = sample().iloc[:8].copy()
    query.index = pd.Index(["same"] * len(query), name="identity")
    result = ivqreg_predict(result=fitted, data=query)
    design = np.column_stack([query["d"], np.ones(len(query)), query["x"]])
    cov = np.asarray(fitted.attrs["state"]["joint"]["covariance"])
    np.testing.assert_allclose(result.attrs["jacobian"], design)
    np.testing.assert_allclose(result.attrs["target_covariance"], design @ cov @ design.T, atol=1e-12)
    assert result["predictions"].index.equals(query.index)
    np.testing.assert_allclose(result["predictions"]["std_error"], np.sqrt(np.diag(design @ cov @ design.T)))
    assert result.attrs["future_observation_interval"] is False


@pytest.mark.parametrize("options", [dict(device="cuda"), dict(weights="w"), dict(cluster="g"),
                                     dict(missing="auto"), dict(intercept=1), dict(quantile=True),
                                     dict(quantile=0.01), dict(confidence=1), dict(bandwidth=0),
                                     dict(inference="robust"), dict(grid=[0, 0, 1]),
                                     dict(grid=[1, 0, -1]), dict(grid=[0, 1]),
                                     dict(max_work=1), dict(max_evaluations=30),
                                     dict(max_evaluations=True), dict(refinement_tolerance=0)])
def test_explicit_unsupported_options(options):
    with pytest.raises(AnalysisError):
        run(**options)


@pytest.mark.parametrize("change", ["boolean", "string", "infinity", "zero_instrument", "duplicate_role", "small", "perfect"])
def test_input_and_identification_refusals(change):
    frame = sample()
    options = {}
    if change == "boolean":
        frame["z"] = frame["z"] > 0
    elif change == "string":
        frame["z"] = frame["z"].astype(str)
    elif change == "infinity":
        frame.loc[0, "y"] = np.inf
    elif change == "zero_instrument":
        frame["z"] = 0
    elif change == "duplicate_role":
        options["instruments"] = ["x"]
    elif change == "small":
        frame = frame.iloc[:20]
    else:
        frame["y"] = frame["d"] + frame["x"]
        options["grid"] = [0, 1, 2]
    with pytest.raises(AnalysisError):
        run(frame, **options)


def test_resource_admission_precedes_qr(monkeypatch, fitted):
    from openecon.econometrics.quantile import kernels
    monkeypatch.setattr(kernels, "solve", lambda *a, **k: pytest.fail("allocated QR"))
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        run()
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        ivqreg_restore(json.dumps(fitted.attrs["state"]))


def test_declared_dtype_device_rng_are_preserved():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    rng = torch.random.get_rng_state().clone()
    try:
        value = run()
        assert value.attrs["dtype"] == "float64"
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(rng, torch.random.get_rng_state())
    finally:
        torch.set_default_dtype(old)


def test_no_intercept_query_and_scaled_instruments():
    frame = sample()
    value = run(frame, intercept=False)
    assert value.attrs["state"]["terms"] == ["d", "x"]
    pred = ivqreg_predict(result=value, data=frame.iloc[:3])
    assert np.asarray(pred.attrs["jacobian"]).shape == (3, 2)
    frame["z"] *= -8
    scaled = run(frame, intercept=False)
    np.testing.assert_allclose(value["weak_id_grid"]["criterion"], scaled["weak_id_grid"]["criterion"], rtol=1e-8)
    np.testing.assert_allclose(value["weak_id_grid"]["weak_id_statistic"], scaled["weak_id_grid"]["weak_id_statistic"], rtol=1e-8)


def test_absent_structural_identification_does_not_create_wald_inference():
    frame = sample()
    frame["d"] = 0.0
    result = run(frame, grid=[-100, 0, 100])
    assert result.attrs["state"]["joint"] is None
    np.testing.assert_allclose(result["weak_id_grid"]["weak_id_statistic"],
                               result["weak_id_grid"]["weak_id_statistic"].iloc[0])
    assert "unestablished" in result.attrs["point_estimate_status"]
    with pytest.raises(AnalysisError):
        run(frame, grid=[-100, 0, 100], inference="identified")


@pytest.mark.parametrize("field", ["alpha", "statistic", "coefficients", "selected_evaluation", "sample_positions", "accepted"])
def test_saved_boolean_numeric_confusion_is_refused(fitted, field):
    state = copy.deepcopy(fitted.attrs["state"])
    if field in {"alpha", "statistic"}:
        state["evaluations"][0][field] = False
    elif field == "coefficients":
        state["evaluations"][0][field][0] = True
    elif field == "selected_evaluation":
        state[field] = True
    elif field == "sample_positions":
        state[field][0] = False
    else:
        state[field][0] = 0
    state.pop("sha256")
    with pytest.raises(AnalysisError):
        ivqreg_restore(seal(state))


def test_no_exogenous_no_intercept_route():
    frame = sample()
    result = run(frame, x=[], intercept=False, grid=[0, 1, 2])
    assert result.attrs["state"]["terms"] == ["d"]
    prediction = ivqreg_predict(result=result, data=frame.iloc[:4])
    assert np.asarray(prediction.attrs["jacobian"]).shape == (4, 1)
