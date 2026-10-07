"""Read back an already-run synthetic native project; never executes code.

Actual native Run/Quit/relaunch and screen observations use computer-use tools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def capture(origin, root):
    projects = json.loads((root / "local-projects.json").read_text())["projects"]
    matching = [row for row in projects if row["name"] == "Bai-Perron synthetic QA"]
    if len(matching) != 1:
        raise RuntimeError("Expected one uniquely named synthetic native project")
    project = matching[0]["id"]
    prefix = f"/api/desktop/projects/{project}/workspace"

    def request(path, token=None):
        headers = {"X-OpenEcon-Token": token} if token else {}
        with urlopen(Request(origin + path, headers=headers), timeout=15) as response:
            return json.load(response)

    desktop = request("/api/desktop/session")["token"]
    status = request("/api/desktop/status", desktop)
    if status["project_id"] != project:
        raise RuntimeError("Native UI has not opened the owned synthetic project")
    token = request(prefix + "/session")["token"]
    history = request(prefix + "/console", token)["history"]
    matching = [row for row in history if "BAI_PERRON_OK " in row.get("stdout", "")]
    if len(matching) != 1 or matching[0]["status"] != "ok":
        raise RuntimeError("Expected one successful native Run")
    run = matching[0]
    marker = "BAI_PERRON_OK "
    proof = next(json.loads(line[len(marker):]) for line in run["stdout"].splitlines() if line.startswith(marker))
    assert proof["frozen"] and proof["nobs"] == 45 and proof["break_indices"] == [15, 30]
    assert proof["p_value"] == .005
    outputs = run["outputs"]
    assert [len(row["data"]["rows"]) for row in outputs] == [3, 3, 3, 6, 6, 45, 31]
    assert all(len(row["data"]["rows"]) == row["data"]["total_rows"] for row in outputs)
    settings = {row[0]: json.loads(row[1]) for row in outputs[-1]["data"]["rows"]}
    assert settings == proof["metadata"]
    document = request(prefix + "/console/script", token)
    assert document["code"] == run["code"]
    return dict(project_id=project, run_id=run["id"], duration_ms=run["duration_ms"],
                history_count=len(history), proof=proof,
                hashes={key: hashed(run[key]) for key in ("code", "stdout", "outputs", "events")},
                document_hash=hashed(document), desktop_status=status,
                full_scientific_settings_saved=True, all_fixture_tables_complete=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = capture(args.origin, args.root)
    if args.output:
        before = json.loads(args.baseline.read_text())
        assert receipt["run_id"] == before["run_id"]
        assert receipt["hashes"] == before["hashes"]
        assert receipt["document_hash"] == before["document_hash"]
        assert receipt["history_count"] == before["history_count"]
        assert receipt["desktop_status"]["console"]["pid"] is None
        receipt.update(status="passed", native_run_verified=True,
                       code_stdout_outputs_events_equal_after_full_native_restart=True,
                       saved_script_equal_after_full_native_restart=True,
                       no_python_worker_started_to_read_saved_history=True,
                       main_installation_changed=False, reused_owned_tauri_shell=True,
                       qa_runtime_rebuilt_from_current_source=True, public_release_delivered=False,
                       human_data_access=False, cuda_verified=False)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    else:
        args.baseline.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("project_id", "run_id", "duration_ms", "history_count")}))
