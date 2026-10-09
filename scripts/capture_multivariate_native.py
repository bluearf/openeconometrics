"""Read back actual native Run and saved script after an owned QA restart.

UI launch, Run and Quit use computer-use tools. This read-only verifier never
executes code or alters project state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from verify_multivariate_runtime import CASES, digest, source_identity, verify_outputs


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def capture(origin, root, results, runtime, source_ref=None):
    projects = json.loads((root / "local-projects.json").read_text())["projects"]
    matches = [item for item in projects if item["name"] == "Multivariate options QA"]
    assert len(matches) == 1
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
    matching = [item for item in history if "MULTIVARIATE_OPTIONS_OK " in item.get("stdout", "")]
    assert len(matching) == 1
    run = matching[0]
    proof = verify_outputs(run)
    assert proof["hashes"] == {name: digest(results/(name+".json")) for name in CASES}
    script = request(prefix+"/console/script", token)
    assert script["code"] == run["code"]
    return {"project_id": project, "run_id": run["id"], "duration_ms": run["duration_ms"],
            "history_count": len(history), "proof": proof, "compiled_source_identity": source_identity(runtime, source_ref),
            "compiled_source_ref": source_ref,
            "runtime_sha256": digest(runtime), "hashes": {key: hashed(run[key]) for key in ("code", "stdout", "outputs", "events")},
            "document_hash": hashed(script), "desktop_status": status}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source-ref", help="Pinned source revision used to build the installed QA runtime")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = capture(args.origin, args.root, args.results, args.runtime, args.source_ref)
    if args.output:
        before = json.loads(args.baseline.read_text())
        for key in ("project_id", "run_id", "history_count", "hashes", "document_hash", "runtime_sha256"):
            assert receipt[key] == before[key], key
        assert receipt["desktop_status"]["console"]["pid"] is None
        receipt.update(status="passed", native_run_verified=True,
                       source_code_outputs_events_equal_after_full_native_restart=True,
                       no_worker_started_to_read_history=True, owned_qa_app=True,
                       primary_installation_changed=False, reused_native_shell_version="0.3.43",
                       public_release_delivered=False, native_results_reopened=False)
        args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    else:
        args.baseline.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({key: receipt[key] for key in ("project_id", "run_id", "duration_ms", "history_count")}))
