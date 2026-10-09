"""Verify all eight scalar design contracts in two frozen runs across full restart."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from verify_bai_perron_runtime import OwnedRuntime, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
MARKER = "EIGHT_DESIGNS_OK "
METHODS = {
    1: ("power_paired_mean", "power_two_proportions", "power_two_correlations", "power_slope"),
    2: ("power_logrank", "power_mcnemar", "precision_mean_unknown", "precision_variance"),
}


def hashed(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def verify_outputs(run):
    assert run["status"] == "ok", run.get("error")
    proofs = [
        json.loads(line[len(MARKER) :])
        for line in run["stdout"].splitlines()
        if line.startswith(MARKER)
    ]
    assert len(proofs) == 1 and proofs[0]["frozen"]
    proof = proofs[0]
    assert proof["part"] in METHODS and list(proof["methods"]) == sorted(METHODS[proof["part"]])
    if proof["part"] == 2:
        assert proof["exact_dyadic_rejection_boundary_verified"]
    outputs = run["outputs"]
    assert len(outputs) == 12
    assert [len(item["data"]["rows"]) for item in outputs] == proof["output_rows"]
    assert all(
        item["type"] == "table" and len(item["data"]["rows"]) == item["data"]["total_rows"]
        for item in outputs
    )
    assert all(
        any(term in item.get("latex", "") for term in ("\\begin{tabular}", "\\begin{longtable}"))
        for item in outputs
    )
    for position, name in enumerate(METHODS[proof["part"]]):
        plan = outputs[3 * position]["data"]
        assert [dict(zip(plan["columns"], row, strict=True)) for row in plan["rows"]] == proof[
            "plan_rows"
        ][name]
        settings = {
            key: json.loads(value) for key, value in outputs[3 * position + 2]["data"]["rows"]
        }
        assert settings == proof["methods"][name]
        assert settings["prospective"] and settings["observations_used"] is False
    assert proof["all_solve_modes_verified"] and proof["third_party_estimation_imports"] == []
    return proof


def example_code(part):
    return """import sys
from pathlib import Path
import importlib
assert getattr(sys, 'frozen', False)
assert Path(importlib.import_module('openecon.econometrics.stats.planning_extended').__file__).is_relative_to(Path(sys._MEIPASS))
""" + (ROOT / f"docs/examples/eight_prospective_designs_{part}.py").read_text()


def verify(runtime):
    from PyInstaller.archive.readers import CArchiveReader

    runtime = runtime.resolve(strict=True)
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in (
        "econometrics.stats.planning",
        "econometrics.stats.planning_extended",
        "econometrics.stats.__init__",
    ):
        source = ROOT / "src/openecon" / (name.replace(".", "/") + ".py")
        module = ("openecon." + name).removesuffix(".__init__")
        assert normalized(archive.extract(module)) == normalized(
            compile(source.read_text(), str(source), "exec", dont_inherit=True)
        )
        modules[module] = digest(source)
    runs = []
    with tempfile.TemporaryDirectory(prefix="openecon-eight-designs-proof-") as temporary:
        root = Path(temporary)
        owned = OwnedRuntime(runtime, root, "unused")
        try:
            project = owned.request(
                "/api/desktop/local-projects",
                {"name": "Eight Prospective Designs QA"},
                desktop=True,
            )["id"]
            owned.project = project
            owned.open_project()
            for part in (1, 2):
                code = example_code(part)
                owned.call("/console/script", {"code": code, "name": "analysis.py"}, method="PUT")
                run = owned.call("/console/execute", {"code": code, "timeout_seconds": 120})
                proof = verify_outputs(run)
                runs.append(dict(run=run, proof=proof))
            document = owned.call("/console/script")
        finally:
            owned.close()
        owned = OwnedRuntime(runtime, root, project)
        try:
            owned.open_project()
            history = owned.call("/console")["history"]
            assert len(history) == 2
            for item in runs:
                after = next(row for row in history if row["id"] == item["run"]["id"])
                assert all(
                    after[key] == item["run"][key]
                    for key in ("code", "stdout", "outputs", "events")
                )
            assert owned.call("/console/script") == document
            assert owned.request("/api/desktop/status", desktop=True)["console"]["pid"] is None
        finally:
            owned.close()
    return dict(
        status="passed",
        compiled_modules_equal_source=modules,
        runtime_sha256=digest(runtime),
        examples_sha256={
            part: digest(ROOT / f"docs/examples/eight_prospective_designs_{part}.py")
            for part in (1, 2)
        },
        runs=[
            dict(
                run_id=item["run"]["id"],
                duration_ms=item["run"]["duration_ms"],
                proof=item["proof"],
                hashes={
                    key: hashed(item["run"][key]) for key in ("code", "stdout", "outputs", "events")
                },
            )
            for item in runs
        ],
        all_eight_scientific_contracts_saved=True,
        complete_output_tables=24,
        code_stdout_outputs_events_equal_after_full_server_restart=True,
        active_second_script_equal_after_restart=True,
        first_script_retained_in_run_history=True,
        no_worker_needed_to_read_history=True,
        source_path_injected=False,
        owned_runtime_stopped=True,
        owned_temporary_data_removed=not root.exists(),
        human_data_access=False,
        native_window_verified=False,
        public_release_delivered=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "complete_output_tables",
                    "all_eight_scientific_contracts_saved",
                )
            }
        )
    )
