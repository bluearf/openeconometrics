"""A completed transport/process alone cannot certify physical science cases."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load("guard_replay", "benchmarks/native_model_matrix_replay.py")
sys.path.insert(0, str(ROOT / "scripts"))
try:
    helpers = load("guard_helpers", "scripts/verify_frozen_helpers.py")
finally:
    sys.path.pop(0)


def model():
    positions = [0, 2, 3]
    position_hash = hashlib.sha256(b"".join(struct.pack("<q", p) for p in positions)).hexdigest()
    return {
        "spec": {"estimator": "ols"},
        "nobs": 3,
        "nobs_original": 4,
        "dropped_rows": 1,
        "coefficients": [{"estimate": 1.0, "std_error": 0.1}],
        "covariance_matrix": [[0.01]],
        "metrics": {"rss": 1.0},
        "inference": {"df_inference": 2},
        "tests": {},
        "sample_positions": positions,
        "provenance": {
            "sample_positions_hash": position_hash,
            "sample_position_count": len(positions),
        },
        "predictions": [
            {"row": 0, "actual": 1.0, "fitted": 0.9},
            {"row": 2, "actual": 2.0, "fitted": 1.9},
        ],
    }


def compare(first, second):
    return replay.compare(first, second, relative=5e-5, absolute=2e-6)


def test_resident_positions_and_omitted_replay_hash_have_the_same_full_membership():
    first, second = model(), model()
    first["sample_positions"] = [3, 0, 2]  # Resident temporary time/panel order.
    second["sample_positions"] = []
    assert compare(first, second)["status"] == "passed"
    second["provenance"]["sample_positions_hash"] = "b" * 64
    assert compare(first, second)["status"] == "failed"


@pytest.mark.parametrize("field", ["nobs", "nobs_original", "dropped_rows"])
def test_counts_are_exact_even_when_relative_tolerance_would_allow_fifty_rows(field):
    first = model()
    first[field] = 1_000_000
    second = deepcopy(first)
    second[field] += 1
    assert compare(first, second)["status"] == "failed"


def test_equal_counts_do_not_hide_different_or_duplicate_retained_positions():
    first, second = model(), model()
    second["sample_positions"] = [0, 1, 3]
    assert compare(first, second)["status"] == "failed"
    second["sample_positions"] = [0, 0, 3]
    assert compare(first, second)["status"] == "failed"


def test_preview_rows_align_by_physical_position_and_compare_actual_outcomes():
    first, second = model(), model()
    second["predictions"].reverse()
    assert compare(first, second)["status"] == "passed"
    second["predictions"][0]["actual"] += 0.1
    assert compare(first, second)["status"] == "failed"


def test_duplicate_and_out_of_source_preview_positions_are_rejected():
    first, second = model(), model()
    second["predictions"].append(second["predictions"][0])
    assert compare(first, second)["status"] == "failed"
    second = model()
    second["predictions"][0]["row"] = 4
    assert compare(first, second)["status"] == "failed"


@pytest.mark.parametrize("flag", [False, 0.0])
def test_explicit_nonconvergence_is_not_ignored_as_optimizer_effort(flag):
    first, second = model(), model()
    first["metrics"]["converged"] = True
    second["metrics"]["converged"] = flag
    assert compare(first, second)["status"] == "failed"
    # An absent resident flag must not hide explicit replay failure either.
    del first["metrics"]["converged"]
    assert compare(first, second)["status"] == "failed"


@pytest.mark.parametrize(
    "cases,requested", [([], []), ([{"id": "owned", "model": "ols", "rows": 3}], ["missing"])]
)
def test_empty_or_absent_requested_model_cannot_return_success(
    monkeypatch, tmp_path, cases, requested
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": ["ols"], "cases": cases}))
    monkeypatch.setattr(replay, "code_identity", lambda: {"aggregate_sha256": "controlled"})
    args = SimpleNamespace(
        manifest=manifest,
        directory=tmp_path / "output",
        models=requested,
        cases_per_model=1,
        skip_warm=True,
        timeout=2,
    )
    with pytest.raises(ValueError, match="nonempty|Requested models"):
        replay.run(args)


@pytest.mark.parametrize("termination", ["nonzero", "timeout"])
def test_real_nonzero_or_cancelled_worker_overrides_its_previously_written_pass(
    monkeypatch, tmp_path, termination
):
    stub = tmp_path / "worker.py"
    stub.write_text(
        """import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('operation');p.add_argument('--case');p.add_argument('--directory');p.add_argument('--mode');p.add_argument('--skip-warm',action='store_true');a=p.parse_args()
Path(a.directory,'receipt.json').write_text(json.dumps({'status':'passed','source_unchanged':True,'scratch_remaining':{}}))
"""
        + ("raise SystemExit(9)\n" if termination == "nonzero" else "time.sleep(30)\n")
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"models": ["ols"], "cases": [{"id": "owned", "model": "ols", "rows": 3}]})
    )
    monkeypatch.setattr(replay, "__file__", str(stub))
    monkeypatch.setattr(
        replay,
        "code_identity",
        lambda: {"aggregate_sha256": "controlled", "git_commit": "controlled"},
    )
    output = tmp_path / "output"
    assert (
        replay.run(
            SimpleNamespace(
                manifest=manifest,
                directory=output,
                models=[],
                cases_per_model=1,
                skip_warm=True,
                timeout=0.3 if termination == "timeout" else 5,
            )
        )
        == 1
    )
    saved = json.loads((output / "matrix.json").read_text())["cases"][0]
    assert saved["status"] == "failed"
    for receipt in saved["modes"].values():
        assert receipt["worker_reported_status"] == "passed"
        assert receipt["status"] == ("timeout" if termination == "timeout" else "worker_failed")
        assert receipt["exit_code"] != 0
        assert receipt["controller_timed_out"] == (termination == "timeout")


def hdfe_proof():
    case = {"id": "owned", "path": "/owned/synthetic.csv"}
    proof = {
        "status": "passed",
        "case": case,
        "source_unchanged": True,
        "scratch_clean": True,
        "measurements": [
            {"nobs": 3, "nobs_original": 3, "fit_seconds": 0.01, "result_sha256": "a" * 64}
        ]
        * 2,
    }
    return case, proof


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("source_unchanged", False),
        ("scratch_clean", False),
        ("measurements", []),
    ],
)
def test_ok_console_cannot_hide_a_failed_or_incomplete_hdfe_helper(field, value):
    case, proof = hdfe_proof()
    proof[field] = value
    with pytest.raises(ValueError, match="HDFE helper"):
        helpers.validate_helper_result(proof, "hdfe", [case])


def test_complete_hdfe_helper_contract_and_empty_planned_refusal():
    case, proof = hdfe_proof()
    helpers.validate_helper_result(proof, "hdfe", [case])
    with pytest.raises(ValueError, match="nonempty"):
        helpers.validate_helper_result(proof, "hdfe", [])


def test_smoothing_helper_must_cover_all_eight_predeclared_method_format_cases():
    cases = [
        {
            "method": method,
            "format": format_,
            "rows": 8,
            "source_sha256": "a" * 64,
            "source_bytes": 100,
            "reference": {"nobs": 3},
        }
        for method in ("bspline_regress", "rcs_regress", "fp_regress", "mfp_regress")
        for format_ in ("csv", "parquet")
    ]
    records = [
        {
            **case,
            "retained_rows": 3,
            "independent_full_inference": True,
            "saved_dataset_prediction_and_weighted_margins": True,
        }
        for case in cases
    ]
    proof = {
        "cases": records,
        "publication_and_persistence": True,
        "resident_smoothing_fit_and_dataset_head_guards": True,
    }
    helpers.validate_helper_result(proof, "smoothing", cases)
    proof["cases"] = records[:-1]
    with pytest.raises(ValueError, match="omitted planned"):
        helpers.validate_helper_result(proof, "smoothing", cases)
    with pytest.raises(ValueError, match="exactly once"):
        helpers.validate_planned_cases(cases[:-1], "smoothing")
