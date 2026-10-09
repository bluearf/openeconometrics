"""Owned frozen runtime: eight complete design results, replay and restart.

Never accesses human projects; a temporary loopback server is terminated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("openecon.econometrics.stats", "openecon.econometrics.stats.planning",
           "openecon.econometrics.stats.power_designs", "openecon.econometrics.stats.power_distributions",
           "openecon.engines.distributions", "openecon.econometrics.registry", "openecon.analysis_contracts")
METHODS = ("power_tmean", "power_ttwomeans", "power_tpaired", "power_anova",
           "power_regression", "power_cluster_mean", "power_gof", "power_independence")
MARKER = "POWER_DESIGN_ACCEPTANCE_OK "
EXAMPLE = ROOT / "docs/examples/power_designs.py"


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
            f"POWER_DESIGN_RESULT_DIRECTORY = {str(directory)!r}\n")


def verify_outputs(run):
    assert run["status"] == "ok", run.get("error")
    assert [item["type"] for item in run["outputs"]] == ["table"] * 16
    assert "Display limit reached" not in run["stdout"]
    proof = json.loads(next(line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)))
    assert proof["frozen"] and proof["all_solve_modes_verified"] and proof["full_input_replay_verified"]
    assert proof["third_party_estimation_imports"] == []
    assert proof["saved_tables"] == 26 and tuple(proof["methods"]) == tuple(sorted(METHODS))
    for j, name in enumerate(METHODS):
        rows = proof["methods"][name]["table_rows"]
        for index, key in ((2*j,"plan"),(2*j+1,"scenarios")):
            output = run["outputs"][index]
            assert len(output["data"]["rows"]) == output["data"]["total_rows"] == rows[key]
            assert any("\\begin{"+env+"}" in output["latex"] for env in ("tabular","longtable"))
    return proof


def verify_files(directory, proof):
    hashes = {}
    for name in METHODS:
        path = directory / (name + ".json")
        hashes[name] = digest(path)
        assert hashes[name] == proof["methods"][name]["sha256"]
        payload = json.loads(path.read_text())
        assert {key:len(frame["data"]) for key,frame in payload["tables"].items()} == proof["methods"][name]["table_rows"]
        assert payload["attrs"]["prospective"] and payload["attrs"]["observations_used"] is False
        assert "\\begin{tabular}" in payload["latex"]
    return hashes


def verify(runtime):
    runtime = runtime.resolve(strict=True)
    sources = source_identity(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-power-designs-") as temporary:
        root = Path(temporary)
        directory = root / "complete-results"
        code = header(directory) + EXAMPLE.read_text()
        owned = OwnedRuntime(runtime, root, "unused")
        try:
            owned.open_project(create=True)
            project = owned.project
            owned.call("/console/script", {"code":code,"name":"analysis.py"}, method="PUT")
            run = owned.call("/console/execute", {"code":code,"timeout_seconds":180})
            proof = verify_outputs(run)
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
                saved_tables=26, displayed_tables=16, all_eight_scientific_settings_saved=True,
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
