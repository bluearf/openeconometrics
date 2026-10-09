"""Owned frozen runtime: complete strict rank-ordered choice models and queries, replay and restart.

Never accesses human projects; a temporary loopback server is terminated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("openecon.econometrics.discrete", "openecon.econometrics.discrete.rank_ordered",
           "openecon.econometrics.discrete.rank_ordered_postestimation",
           "openecon.econometrics.registry", "openecon.analysis_contracts",
           "openecon.engines.optimize", "openecon.resources", "openecon.econometrics.core",
           "openecon.econometrics.nonparametric.common")
METHODS = ("oim", "hc0", "cr0", "first", "stages", "effects", "elasticities")
MARKER = "RANK_ORDERED_ACCEPTANCE_OK "
EXAMPLE = ROOT / "docs/examples/rank_ordered.py"
FIT_TABLES = ("inputs", "parameters", "information", "bread", "meat", "covariance", "case_scores", "cluster_scores", "stage_scores", "stages", "case_likelihood", "probabilities", "probability_jacobian", "fit_summary")
PREDICTION_TABLES = ("predictions", "covariance", "log_probability_covariance", "joint_covariance", "joint_jacobian", "log_odds_covariance", "jacobian", "log_probability_jacobian", "log_odds_jacobian", "coefficient_covariance")
MARGIN_TABLES = ("margins", "per_case", "support", "covariance", "jacobian", "per_case_jacobian", "coefficient_covariance")


def expected_tables(method):
    return FIT_TABLES if method in ("oim", "hc0", "cr0") else PREDICTION_TABLES if method in ("first", "stages") else MARGIN_TABLES


def wire_scalar(value):
    """Exact bounded console encoding for these synthetic table cells."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, int):
        return str(value) if abs(value) > 2**53-1 else value
    if isinstance(value, float):
        assert math.isfinite(value), "Nonfinite complete table cell"
        return value
    return str(value)[:500]


def saved_frames(payload):
    from openecon.frame import DataFrame

    frames = {}
    for key, saved in payload["tables"].items():
        assert set(saved) == {"index", "columns", "data"}, key
        assert len(saved["index"]) == len(saved["data"]), key
        assert all(len(row) == len(saved["columns"]) for row in saved["data"]), key
        frames[key] = DataFrame(saved["data"], columns=saved["columns"], index=saved["index"])
    return frames


def source_identity(runtime):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    hashes = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        assert normalized(archive.extract(name)) == normalized(compile(path.read_text(), str(path), "exec", dont_inherit=True)), name
        hashes[name] = digest(path)
    return hashes


def header(directory):
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n"
            f"RANK_ORDERED_RESULT_DIRECTORY = {str(directory)!r}\n")


def verify_outputs(run, directory):
    assert run["status"] == "ok", run.get("error")
    assert "Display limit reached" not in run["stdout"]
    markers = [line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(markers) == 1, "Expected one complete acceptance marker"
    proof = json.loads(markers[0])
    assert proof["frozen"] and proof["all_four_scopes_verified"] and proof["full_state_replay_verified"]
    assert proof["saved_tables"] == 76 == sum(len(row["table_rows"]) for row in proof["methods"].values())
    assert proof["displayed_tables"] == 7
    assert proof["third_party_estimation_imports"] == []
    assert tuple(proof["methods"]) == tuple(sorted(METHODS))
    assert [item["type"] for item in run["outputs"]] == ["table"] * proof["displayed_tables"]
    offset = 0
    for name in METHODS:
        rows = proof["methods"][name]["table_rows"]
        assert set(rows) == set(expected_tables(name))
        path = directory / (name + ".json")
        assert digest(path) == proof["methods"][name]["sha256"]
        payload = json.loads(path.read_text())
        frames = saved_frames(payload)
        expected_display = ["parameters"] if name in ("oim", "hc0", "cr0") else ["predictions"] if name in ("first", "stages") else ["margins"]
        assert proof["methods"][name]["displayed_keys"] == expected_display
        for key in proof["methods"][name]["displayed_keys"]:
            output = run["outputs"][offset]
            frame = payload["tables"][key]
            data = output["data"]
            assert len(data["rows"]) == data["total_rows"] == rows[key]
            assert data["columns"] == frame["columns"], (name, key, "columns")
            assert data["total_columns"] == len(frame["columns"]), (name, key, "column truncation")
            assert data["rows"] == [[wire_scalar(cell) for cell in row] for row in frame["data"]], (name, key, "values")
            assert data["index"] == [[wire_scalar(value)] for value in frame["index"]], (name, key, "index")
            assert data["index_names"] == [None], (name, key, "index names")
            expected_latex = str(frames[key].to_latex())
            assert output["latex"] == expected_latex, (name, key, "LaTeX")
            assert hashlib.sha256(expected_latex.encode()).hexdigest() == proof["methods"][name]["displayed_latex_sha256"][key], (name, key, "source-table LaTeX binding")
            offset += 1
    assert offset == proof["displayed_tables"]
    return proof


def verify_files(directory, proof):
    from openecon.econometrics.core import TableSet

    hashes = {}
    for name in METHODS:
        path = directory / (name + ".json")
        hashes[name] = digest(path)
        assert hashes[name] == proof["methods"][name]["sha256"]
        payload = json.loads(path.read_text())
        assert set(payload) == {"attrs", "tables", "latex"}
        assert set(payload["tables"]) == set(expected_tables(name))
        assert {key:len(frame["data"]) for key,frame in payload["tables"].items()} == proof["methods"][name]["table_rows"]
        # Every saved frame contributes its complete captioned LaTeX export.
        frames = saved_frames(payload)
        assert payload["latex"] == TableSet({key: frames[key] for key in expected_tables(name)}).to_latex(), (name, "full LaTeX")
        assert payload["attrs"]["contract"] in ("rank_ordered_v1", "rank_ordered_postestimation_v1")
        json.dumps(payload, allow_nan=False)
    return hashes


def verify(runtime):
    runtime = runtime.resolve(strict=True)
    sources = source_identity(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-rank-ordered-") as temporary:
        root = Path(temporary)
        directory = root / "complete-results"
        code = header(directory) + EXAMPLE.read_text()
        owned = OwnedRuntime(runtime, root, "unused")
        try:
            owned.open_project(create=True)
            project = owned.project
            owned.call("/console/script", {"code":code,"name":"analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code":code,"timeout_seconds":180})
            proof = verify_outputs(run, directory)
            hashes = verify_files(directory,proof)
            original = {key:run[key] for key in ("code","stdout","outputs","events")}
            document = owned.call("/console/script")
            owned.call("/console/reset", {})
            saved = next(row for row in owned.call("/console")["history"] if row["id"] == run["id"])
            assert original == {key:saved[key] for key in original}
        finally:
            owned.close()
        owned = OwnedRuntime(runtime,root,project)
        try:
            owned.open_project()
            saved = next(row for row in owned.call("/console")["history"] if row["id"] == run["id"])
            assert original == {key:saved[key] for key in original}
            assert owned.call("/console/script") == document
            assert verify_files(directory,proof) == hashes
            status = owned.request("/api/desktop/status",desktop=True)
            assert status["console"]["pid"] is None
        finally:
            owned.close()
    return dict(status="passed", source_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                compiled_modules_equal_source=sources, runtime_sha256=digest(runtime), example_sha256=digest(EXAMPLE),
                execution_id=run["id"],duration_ms=run["duration_ms"], proof=proof, complete_result_hashes=hashes,
                saved_tables=proof["saved_tables"], displayed_tables=proof["displayed_tables"], all_four_scopes_saved=True,
                all_saved_inputs_replayed=True, code_stdout_outputs_events_equal_after_worker_reset=True,
                code_stdout_outputs_events_equal_after_server_restart=True, complete_files_unchanged_after_restart=True,
                saved_script_equal_after_restart=True, no_worker_needed_to_read_history=True,
                source_path_injected=False, owned_runtime_stopped=True, owned_temporary_data_removed=not root.exists(),
                human_data_access=False,native_window_verified=False,public_release_delivered=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("status","duration_ms","saved_tables","displayed_tables")}))
