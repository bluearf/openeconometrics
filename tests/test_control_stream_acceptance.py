"""Independent scientific and end-to-end streamed CF fit acceptance."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks/control_stream_fit_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("cf_stream_independent", BENCHMARK)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)


@pytest.mark.parametrize("kind", oracle.KINDS)
def test_independent_outcome_score_and_observed_derivative(kind):
    frame = oracle.fixture(kind, 80)
    y = frame.y.dropna().to_numpy()[:32]
    eta = np.linspace(-0.9, 0.8, len(y))
    criterion, score, derivative, mean = oracle.quantities(kind, y, eta)
    step = 1e-5
    upper = oracle.quantities(kind, y, eta+step)
    lower = oracle.quantities(kind, y, eta-step)
    np.testing.assert_allclose((upper[0]-lower[0])/(2*step), score, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose((upper[1]-lower[1])/(2*step), derivative, atol=2e-8, rtol=2e-8)
    assert np.isfinite(criterion).all() and np.isfinite(mean).all()


@pytest.mark.parametrize("kind", oracle.KINDS)
@pytest.mark.parametrize("covariance", ("robust", "cluster"))
def test_independent_fit_has_full_cross_stage_uncertainty(kind, covariance):
    frame = oracle.fixture(kind, 400)
    fitted = oracle.fit_oracle(kind, frame, covariance)
    bread, joint = fitted["bread"], fitted["joint_covariance"]
    np.testing.assert_array_equal(bread[:4, 4:], np.zeros((4, 4)))
    assert np.max(np.abs(bread[4:, :4])) > 1
    assert np.max(np.abs(joint[:4, 4:])) > 1e-7
    np.testing.assert_allclose(joint, joint.T, atol=1e-14)
    assert np.linalg.eigvalsh(joint)[0] > 0
    assert fitted["sample"]["nobs"] < len(frame)
    # The stacked bread is minus the derivative of all estimating equations.
    required = ["y", "x", "d", "z1", "z2"]
    if covariance == "cluster":
        required.append("cluster")
    selected = frame.loc[frame[required].notna().all(axis=1)]
    z = np.column_stack((np.ones(len(selected)), selected[["x", "z1", "z2"]]))
    x = np.column_stack((np.ones(len(selected)), selected[["x", "d"]]))
    d, y = selected.d.to_numpy(), selected.y.to_numpy()

    def equations(p):
        residual = d-z@p[:4]
        q = np.column_stack((x, residual))
        score = oracle.quantities(kind, y, q@p[4:])[1]
        return np.r_[z.T@residual, q.T@score]

    numeric = -oracle.jacobian(equations, np.r_[fitted["gamma"], fitted["beta"]])
    np.testing.assert_allclose(bread, numeric, atol=2e-7, rtol=2e-7)


def test_protocol_fixes_all_eight_physical_files_and_separate_covariance_samples(tmp_path):
    manifest = oracle.generate(tmp_path / "inputs", source_pin="fixed-base", rows=400)
    assert manifest["schema"] == oracle.PROTOCOL_SCHEMA
    assert manifest["max_iterations"] == 20 and manifest["batch_rows"] == 8192
    assert len(manifest["plan"]) == 8
    assert {plan["format"] for plan in manifest["plan"]} == {"csv", "parquet"}
    for plan in manifest["plan"]:
        path = Path(plan["file"])
        assert path.is_file() and oracle.digest(path) == plan["sha256"]
        assert plan["samples"]["cluster"]["nobs"] == plan["samples"]["robust"]["nobs"]-1
        assert plan["samples"]["cluster"]["positions_sha256"] != plan["samples"]["robust"]["positions_sha256"]
    with pytest.raises(FileExistsError):
        oracle.generate(tmp_path / "inputs", source_pin="fixed-base", rows=400)


def test_oracle_process_imports_no_tensor_or_estimation_sdk():
    code = f'''import importlib.util, sys
spec = importlib.util.spec_from_file_location("cf_stream_oracle", {str(BENCHMARK)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for kind in module.KINDS:
    module.fit_oracle(kind, module.fixture(kind, 128), "robust")
assert not any(n.split(".")[0] in {{"openecon", "torch", "scipy", "statsmodels"}} for n in sys.modules)
print("independent")
'''
    result = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert result.strip() == "independent"


@pytest.mark.parametrize("kind", oracle.KINDS)
@pytest.mark.parametrize("covariance", ("robust", "cluster"))
def test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich(
        kind, covariance, tmp_path):
    import openecon as oe

    frame = oracle.fixture(kind, 6001)
    suffix = "csv" if oracle.KINDS.index(kind) % 2 == 0 else "parquet"
    path = tmp_path / (kind+"."+suffix)
    if suffix == "csv":
        frame.to_csv(path, index=False, float_format="%.17g")
        frame = pd.read_csv(path)
    else:
        frame.to_parquet(path, index=False, row_group_size=257)
        frame = pd.read_parquet(path)
    source = oe.scan(path)
    result = getattr(oe, oracle.APIS[kind])(data=source, y="y", endogenous="d",
        x=["x"], instruments=["z1", "z2"], covariance=covariance,
        cluster="cluster" if covariance == "cluster" else None,
        missing="drop", max_iterations=20, batch_rows=257)
    independent = oracle.fit_oracle(kind, frame, covariance)
    state = result.extra["control_function_state"]
    assert state["schema"] == "openecon.control_function.stream.v2"
    assert result.nobs > 5000 and result.nobs_original == 6001
    for key in ("gamma", "beta", "bread", "meat", "joint_covariance", "criterion"):
        np.testing.assert_allclose(state[key], independent[key], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(result.covariance_matrix, independent["joint_covariance"],
                               atol=2e-7, rtol=2e-6)
    assert not {"z", "x", "d", "y", "design", "row_scores", "cluster_codes",
                "source_index", "source_values", "sample_positions"}.intersection(state)
    encoded = result.model_dump_json()
    assert len(encoded.encode()) < 262144
    restored = oe.cf_restore(result=encoded, data=source, batch_rows=257)
    query = source.head(41)
    pd.testing.assert_frame_equal(oe.predict(result, query), oe.predict(restored, query))
    pd.testing.assert_frame_equal(oe.margins(result, ["x", "d", "z1"], data=query),
                                  oe.margins(restored, ["x", "d", "z1"], data=query))
    saved = json.loads(encoded)
    assert saved["nobs"] == independent["sample"]["nobs"]
    assert saved["dropped_rows"] == independent["sample"]["dropped_rows"]
