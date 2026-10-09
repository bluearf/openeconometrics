"""Verify four full prospective contracts in a frozen server across restart.

Uses the owned runtime lifecycle helper, never human projects or native UI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
MARKER = "PROSPECTIVE_PLANNING_OK "
METHODS = ("mean", "proportion", "correlation", "precision")


def verify_outputs(run):
    if run["status"] != "ok":
        raise RuntimeError("Planning execution failed: " + str(run.get("error")))
    proofs = [json.loads(line[len(MARKER):]) for line in run["stdout"].splitlines()
              if line.startswith(MARKER)]
    if len(proofs) != 1 or not proofs[0]["frozen"]:
        raise RuntimeError("Missing frozen planning proof")
    proof = proofs[0]
    outputs = run["outputs"]
    assert len(outputs) == 13
    assert [len(item["data"]["rows"]) for item in outputs] == proof["output_rows"]
    assert all(item["type"] == "table" and len(item["data"]["rows"]) == item["data"]["total_rows"]
               for item in outputs)
    assert all(any(term in item.get("latex", "") for term in
                   ("\\begin{tabular}", "\\begin{longtable}")) for item in outputs)
    for position, name in enumerate(METHODS):
        saved_plan = outputs[3 * position]["data"]
        assert [dict(zip(saved_plan["columns"], row, strict=True))
                for row in saved_plan["rows"]] == proof["plan_rows"][name]
        settings = {key: json.loads(value) for key, value in outputs[3 * position + 2]["data"]["rows"]}
        assert settings == proof["methods"][name]
        assert settings["prospective"] and settings["observations_used"] is False
    assert proof["all_solve_modes_verified"] and proof["third_party_estimation_imports"] == []
    return proof


def verify(runtime):
    from PyInstaller.archive.readers import CArchiveReader
    runtime = runtime.resolve(strict=True)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in ("econometrics.stats.planning", "econometrics.stats.__init__"):
        source = ROOT / "src/openecon" / (name.replace(".", "/") + ".py")
        module = ("openecon." + name).removesuffix(".__init__")
        assert normalized(archive.extract(module)) == normalized(
            compile(source.read_text(), str(source), "exec", dont_inherit=True))
        modules[module] = digest(source)
    example = ROOT / "docs/examples/prospective_planning.py"
    code = """import sys
from pathlib import Path
import importlib
assert getattr(sys, 'frozen', False)
module = importlib.import_module('openecon.econometrics.stats.planning')
assert Path(module.__file__).is_relative_to(Path(sys._MEIPASS))
""" + example.read_text()
    with tempfile.TemporaryDirectory(prefix="openecon-planning-proof-") as temporary:
        root = Path(temporary)
        owned = OwnedRuntime(runtime, root, "unused")
        try:
            owned.open_project(create=True)
            project = owned.project
            owned.call("/console/script", {"code": code, "name": "analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code": code, "timeout_seconds": 120})
            proof = verify_outputs(run)
            document = owned.call("/console/script")
            original = {key: run[key] for key in ("code", "stdout", "outputs", "events")}
        finally:
            owned.close()
        owned = OwnedRuntime(runtime, root, project)
        try:
            owned.open_project()
            after = next(row for row in owned.call("/console")["history"] if row["id"] == run["id"])
            assert original == {key: after[key] for key in original}
            assert owned.call("/console/script") == document
            status = owned.request("/api/desktop/status", desktop=True)
            assert status["console"]["pid"] is None
        finally:
            owned.close()
    return dict(status="passed", proof=proof, compiled_modules_equal_source=modules,
                runtime_sha256=digest(runtime), example_sha256=digest(example),
                execution_id=run["id"], duration_ms=run["duration_ms"],
                output_rows=proof["output_rows"], all_four_scientific_settings_saved=True,
                code_stdout_outputs_events_equal_after_full_server_restart=True,
                saved_script_equal_after_restart=True, no_worker_needed_to_read_history=True,
                source_path_injected=False, owned_runtime_stopped=True,
                owned_temporary_data_removed=not root.exists(), human_data_access=False,
                native_window_verified=False, public_release_delivered=False,
                outputs_sha256=hashlib.sha256(json.dumps(run["outputs"], sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "duration_ms", "output_rows")}))
