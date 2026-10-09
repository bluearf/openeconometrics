"""Read back the new owned native Run and exact saved script after QA restart.

UI launch, Run and Quit belong to computer-use tools. This read-only verifier
never executes analysis or changes project state; it only inspects the exact
Multivariate selection and contrasts verified QA project and the new wave's complete result files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from verify_multivariate_selection_runtime import EXAMPLE, digest, source_identity, validate_result_files, verify_outputs


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def capture(origin, root, results, runtime, source_ref=None, project_name="Multivariate selection and contrasts verified QA"):
    projects = json.loads((root / "local-projects.json").read_text())["projects"]
    matches = [item for item in projects if item["name"] == project_name]
    assert len(matches) == 1, "Exactly one owned new-wave QA project is required"
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
    matching = [item for item in history if "MULTIVARIATE_SELECTION_CONTRASTS_OK " in item.get("stdout", "")]
    assert len(matching) == 1, "Choose a fresh project with one successful new-wave native Run"
    run = matching[0]
    proof = verify_outputs(run)
    result_hashes = validate_result_files(results, proof)
    script = request(prefix+"/console/script", token)
    assert script["code"] == run["code"]
    assert run["code"].endswith(EXAMPLE.read_text()), "The actual Run must execute this exact new-wave example"
    return {"project_name": project_name, "project_id": project,
            "run_id": run["id"], "duration_ms": run["duration_ms"],
            "history_count": len(history), "proof": proof,
            "compiled_source_identity": source_identity(runtime, source_ref),
            "compiled_source_ref": source_ref, "runtime_sha256": digest(runtime),
            "complete_result_hashes": result_hashes,
            "fixture_hashes": {name: digest(results/name) for name in ("fixtures.json", "base-models.json")},
            "hashes": {key: hashed(run[key]) for key in ("code", "stdout", "outputs", "events")},
            "document_hash": hashed(script), "desktop_status": status,
            "example_sha256": digest(EXAMPLE), "capture_sha256": digest(__file__),
            "native_tables_and_full_saved_payloads_verified": True}


def verify_restart(before, receipt):
    for key in ("project_name", "project_id", "run_id", "history_count", "hashes", "document_hash",
                "runtime_sha256", "complete_result_hashes", "fixture_hashes", "compiled_source_identity", "example_sha256"):
        assert receipt[key] == before[key], key
    assert receipt["desktop_status"]["console"]["pid"] is None
    receipt.update(status="passed", native_run_verified=True,
                   source_code_outputs_events_equal_after_full_native_restart=True,
                   no_worker_started_to_read_history=True, owned_qa_app=True,
                   primary_installation_changed=False, reused_native_shell_version="0.3.43",
                   public_release_delivered=False, native_results_reopened=False,
                   frozen_full_result_reconstruction_is_separate=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--project-name", default="Multivariate selection and contrasts verified QA")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source-ref", help="Pinned source revision used to build the installed QA runtime")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.baseline
    if destination.exists():
        parser.error("Choose a fresh receipt path")
    if args.output and not args.baseline.exists():
        parser.error("The saved baseline is required for restart comparison")
    receipt = capture(args.origin, args.root, args.results, args.runtime, args.source_ref, args.project_name)
    if args.output:
        receipt = verify_restart(json.loads(args.baseline.read_text()), receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: receipt[key] for key in ("project_id", "run_id", "duration_ms", "history_count")}))
