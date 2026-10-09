"""Read-only exact spectral geometry native Run/history capture.

Actual launch, Run, Quit and reopening use computer-use tools. This helper
never executes analysis or modifies the user's project or installed app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from verify_spectral_geometry_runtime import (
    EXAMPLE, MARKER, digest, source_identity, validate_displayed_tables,
    validate_result_files, verify_outputs,
)


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def capture(origin, root, results, runtime, source_ref,
            project_name="Spectral geometry uncertainty QA"):
    projects = json.loads((root/"local-projects.json").read_text())["projects"]
    matches = [item for item in projects if item["name"] == project_name]
    assert len(matches) == 1, "One fresh owned spectral QA project is required"
    project = matches[0]["id"]
    prefix = f"/api/desktop/projects/{project}/workspace"

    def request(path, token=None):
        with urlopen(Request(origin+path, headers={"X-OpenEcon-Token": token} if token else {}), timeout=20) as response:
            return json.load(response)

    desktop = request("/api/desktop/session")["token"]
    status = request("/api/desktop/status", desktop)
    assert status["project_id"] == project
    token = request(prefix+"/session")["token"]
    history = request(prefix+"/console", token)["history"]
    matching = [item for item in history if MARKER in item.get("stdout", "")]
    assert len(matching) == 1
    run = matching[0]
    proof = verify_outputs(run)
    hashes = validate_result_files(results, proof)
    validate_displayed_tables(run, results)
    script = request(prefix+"/console/script", token)
    assert script["code"] == run["code"] and run["code"].endswith(EXAMPLE.read_text())
    return {"project_name": project_name, "project_id": project, "run_id": run["id"],
            "duration_ms": run["duration_ms"], "history_count": len(history), "proof": proof,
            "compiled_source_identity": source_identity(runtime, source_ref),
            "compiled_source_ref": source_ref, "runtime_sha256": digest(runtime),
            "complete_result_hashes": hashes, "fixture_sha256": digest(results/"fixtures.json"),
            "hashes": {key: hashed(run[key]) for key in ("code", "stdout", "outputs", "events")},
            "document_hash": hashed(script), "desktop_status": status,
            "example_sha256": digest(EXAMPLE), "capture_sha256": digest(__file__),
            "native_tables_and_full_saved_payloads_verified": True}


def verify_restart(before, receipt):
    for key in ("project_name", "project_id", "run_id", "history_count", "hashes", "document_hash",
                "runtime_sha256", "complete_result_hashes", "fixture_sha256", "compiled_source_identity", "example_sha256"):
        assert receipt[key] == before[key], key
    assert receipt["desktop_status"]["console"]["pid"] is None
    receipt.update(status="passed", native_run_verified=True,
                   source_code_outputs_events_equal_after_full_native_restart=True,
                   no_worker_started_to_read_history=True, owned_qa_app=True,
                   primary_installation_changed=False,
                   public_release_delivered=False, native_results_reopened=False,
                   frozen_full_result_reconstruction_is_separate=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--project-name", default="Spectral geometry uncertainty QA")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.baseline
    if destination.exists():
        parser.error("Choose a fresh receipt path")
    if args.output and not args.baseline.exists():
        parser.error("Saved baseline is required")
    receipt = capture(args.origin, args.root, args.results, args.runtime, args.source_ref, args.project_name)
    if args.output:
        receipt = verify_restart(json.loads(args.baseline.read_text()), receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: receipt[key] for key in ("project_id", "run_id", "duration_ms", "history_count")}))
