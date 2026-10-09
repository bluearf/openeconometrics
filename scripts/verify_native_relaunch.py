"""Read-only persistence/process checks for the owned binary-mediation QA app.

The human/GUI operator launches, selects the saved project and fully Quits the
app. Run baseline, stopped1, after1, stopped2, after2 in that order. This helper
never launches/Quits an app, opens a project, executes code or changes QA data.
Only the evidence receipt is written. Native visibility is separate GUI proof.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
from pathlib import Path
import plistlib
import socket
import subprocess
import time
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Library/Application Support/org.openecon.qa.binarymediation"
APP = Path.home() / "Applications/OpenEconometrics Binary Mediation QA.app"
EVIDENCE = ROOT / "docs/evidence/binary-mediation-eight-2026-10-07"
PROJECT = "58aff430e12140a793b6550e09d21d6c"
SCRIPT = "ed12206d4b23467282107fa775b98dac"
EXECUTION = "882d7fcd-6f5e-4ebc-a676-ac223f83f4f9"
NAME = "Eight Binary Mediation Domains QA"
PHASES = ("baseline", "stopped1", "after1", "stopped2", "after2")
METHODS = tuple(a + "_" + b for a in ("logit", "probit")
                for b in ("gaussian", "logit", "probit", "poisson"))
TABLES = {"inputs", "parameters", "information", "bread", "scores", "model_loglikelihood",
          "covariance", "counterfactual_rows", "means", "means_covariance", "effects",
          "effects_covariance", "delta_jacobian", "fit_summary"}
MARKER = "BINARY_MEDIATION_ACCEPTANCE_OK "


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    require(not path.is_symlink() and path.is_file(), "Missing or linked owned QA file.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    digest(path)  # Refuse linked inputs before opening them.
    return json.loads(path.read_text())


def json_hash(value):
    # Match the historical installed execution receipt's complete-record hash.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def saved_snapshot():
    """Hash every saved console document/history byte and all 112 result tables."""
    require(DATA.is_dir() and not DATA.is_symlink(), "The owned QA data folder is unavailable.")
    for folder in (DATA / "projects", DATA / "projects" / PROJECT,
                   DATA / "projects" / PROJECT / "console", DATA / "complete-binary-mediation"):
        require(folder.is_dir() and not folder.is_symlink(), "A linked/missing QA folder is refused.")
    installed = read_json(EVIDENCE / "installed.json")
    require(installed["project_id"] == PROJECT and installed["script_id"] == SCRIPT
            and installed["execution_id"] == EXECUTION, "Unexpected historical installed receipt.")
    catalog = read_json(DATA / "local-projects.json")
    require(len(catalog["projects"]) == 1 and catalog["projects"][0]["id"] == PROJECT
            and catalog["projects"][0]["name"] == NAME, "The QA project catalog changed.")
    console = DATA / "projects" / PROJECT / "console"
    files = sorted(console.iterdir())
    require(all(f.is_file() and not f.is_symlink() and f.suffix == ".json" for f in files),
            "Unexpected or linked QA console entry.")
    hashes = {f.name: digest(f) for f in files}
    require({"history.json", "scripts.json", "script.json", SCRIPT + ".json"} <= set(hashes),
            "Saved console state is incomplete.")
    history = read_json(console / "history.json")["history"]
    require(isinstance(history, list) and all(isinstance(row, dict) for row in history),
            "Invalid saved history.")
    matches = [row for row in history if row.get("id") == EXECUTION]
    require(len(matches) == 1, "The installed execution is missing or duplicated.")
    run = matches[0]
    require(json_hash(run) == installed["execution_sha256"], "The complete installed execution changed.")
    script = read_json(console / (SCRIPT + ".json"))
    require(script["id"] == SCRIPT and script["code"] == run["code"], "Saved script code changed.")
    outputs = run["outputs"]
    require(run["status"] == "ok" and len(outputs) == 16
            and all(item["type"] == "table" for item in outputs), "Incomplete installed outputs.")
    markers = [line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    require(len(markers) == 1, "Missing or duplicate saved acceptance marker.")
    proof = json.loads(markers[0])
    require(proof["saved_tables"] == 112 and proof["frozen"]
            and proof["all_eight_contracts_verified"] and proof["full_state_replay_verified"],
            "Incomplete saved scientific proof.")
    directory = DATA / "complete-binary-mediation"
    require({f.name for f in directory.iterdir()} == {name + ".json" for name in METHODS},
            "Unexpected or missing complete result file.")
    result_hashes, shapes = {}, {}
    for j, name in enumerate(METHODS):
        path = directory / (name + ".json")
        result_hashes[name] = digest(path)
        require(result_hashes[name] == installed["complete_result_hashes"][name]
                == proof["methods"][name]["sha256"], "A complete installed result changed.")
        payload = read_json(path)
        require(set(payload["tables"]) == TABLES and "binary_mediation_state" in payload["attrs"]
                and "\\begin{tabular}" in payload["latex"], "Incomplete saved table/state/LaTeX payload.")
        shapes[name] = {}
        for key, frame in payload["tables"].items():
            columns, rows, index = frame["columns"], frame["data"], frame["index"]
            require(len(index) == len(rows) and all(len(row) == len(columns) for row in rows),
                    "Incomplete saved table schema or rows.")
            require(len(rows) == proof["methods"][name]["table_rows"][key], "Saved table rows changed.")
            shapes[name][key] = {"rows": len(rows), "columns": len(columns)}
        require(proof["methods"][name]["displayed_keys"] == ["parameters", "effects"],
                "Unexpected saved display order.")
        for offset, key in enumerate(("parameters", "effects")):
            frame = outputs[2*j + offset]["data"]
            require(len(frame["rows"]) == frame["total_rows"] == shapes[name][key]["rows"],
                    "Saved native output was truncated.")
    return {"catalog_sha256": digest(DATA / "local-projects.json"), "console_file_sha256": hashes,
            "complete_history_sha256": json_hash(history), "history_count": len(history),
            "history_ids": [row["id"] for row in history], "saved_script_code_sha256": json_hash(script["code"]),
            "execution_id": EXECUTION, "execution_sha256": json_hash(run),
            "complete_result_hashes": result_hashes, "saved_table_shapes": shapes,
            "saved_tables": 112, "displayed_tables": 16}


def bundle_identity():
    frozen, bundle = read_json(EVIDENCE / "frozen.json"), read_json(EVIDENCE / "qa-bundle.json")
    require(frozen["status"] == bundle["status"] == "passed"
            and frozen["source_head"] == bundle["source_head"], "Historical bundle receipts disagree.")
    require(bundle["build"]["identifier"] == "org.openecon.qa.binarymediation",
            "Unexpected QA bundle receipt.")
    with (APP / "Contents/Info.plist").open("rb") as stream:
        require(plistlib.load(stream)["CFBundleIdentifier"] == "org.openecon.qa.binarymediation",
                "The installed app is not the owned QA bundle.")
    native = APP / "Contents/MacOS/openecon-desktop"
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    require(digest(runtime) == frozen["runtime_sha256"], "The frozen QA runtime changed.")
    return {"source_head": frozen["source_head"], "native_sha256": digest(native),
            "runtime_sha256": frozen["runtime_sha256"],
            "frozen_receipt_sha256": digest(EVIDENCE / "frozen.json"),
            "qa_bundle_receipt_sha256": digest(EVIDENCE / "qa-bundle.json")}


def process_table():
    """No command arguments/environment are printed or persisted."""
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,lstart=,comm="], text=True)
    result = []
    for line in rows.splitlines():
        parts = line.split(None, 7)
        if len(parts) == 8:
            result.append({"pid": int(parts[0]), "ppid": int(parts[1]),
                           "started": " ".join(parts[2:7]), "executable": parts[7]})
    return result


def owned_processes(rows, *, running):
    native = str(APP / "Contents/MacOS/openecon-desktop")
    runtime = str(APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime")
    apps = [r for r in rows if r["executable"] == native]
    runtimes = [r for r in rows if r["executable"] == runtime]
    if not running:
        require(not apps and not runtimes, "The owned app or bundled runtime is still running.")
        return []
    require(len(apps) == 1 and runtimes, "The owned native app/runtime is not running.")
    owned = {apps[0]["pid"]}
    while True:
        added = {r["pid"] for r in rows if r["ppid"] in owned} - owned
        if not added:
            break
        owned |= added
    require(all(r["pid"] in owned for r in runtimes), "QA runtime is not a descendant of the owned app.")
    return [r for r in rows if r["pid"] in owned]


def get_json(port, path, token=None):
    require(path in {"/api/desktop/session", "/api/desktop/status"}, "Only passive desktop GETs are allowed.")
    headers = {"X-OpenEcon-Token": token} if token else {}
    request = Request(f"http://127.0.0.1:{port}" + path, headers=headers, method="GET")
    # Disable proxy handling for the explicitly owned loopback endpoint.
    from urllib.request import ProxyHandler, build_opener
    with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
        return json.load(response)


def live_status(port):
    token = get_json(port, "/api/desktop/session")["token"]
    status = get_json(port, "/api/desktop/status", token)
    require(status["project_id"] == PROJECT, "Select the owned QA project through the GUI first.")
    console = status["console"]
    require(isinstance(console, dict) and console.get("pid") is None
            and console.get("running") is False and console.get("execution_id") is None,
            "A console worker/execution exists; read-only relaunch proof is invalid.")
    return {"project_id": PROJECT, "console_pid": None, "console_running": False,
            "console_execution_id": None, "session_generation": console["session_generation"]}


def port_closed(port):
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", port)) == errno.ECONNREFUSED


def stopped_state(previous, *, timeout=15):
    deadline = time.monotonic() + timeout
    while True:
        rows = process_table()
        remaining = {(r["pid"], r["started"]) for r in rows}
        gone = all((r["pid"], r["started"]) not in remaining for r in previous["owned_processes"])
        paths_gone = not any(r["executable"] in {
            str(APP / "Contents/MacOS/openecon-desktop"),
            str(APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime")} for r in rows)
        if gone and paths_gone and port_closed(previous["port"]):
            return {"tracked_owned_process_identities_gone": True, "owned_executable_paths_absent": True,
                    "loopback_connection_refused": True, "port": previous["port"]}
        require(time.monotonic() < deadline, "Full Quit did not stop owned processes and refuse the old loopback port.")
        time.sleep(.2)


def verify_phase(phase, receipt):
    receipt = receipt.expanduser().resolve()
    require(phase in PHASES, "Unknown verification phase.")
    require(not receipt.is_relative_to(DATA) and not receipt.is_relative_to(APP),
            "Evidence must be written outside the owned app/data profile.")
    record = read_json(receipt) if receipt.exists() else {"schema": "owned_native_relaunch_v1", "phases": []}
    require([row["phase"] for row in record["phases"]] == list(PHASES[:PHASES.index(phase)]),
            "Run each verification phase once, in baseline/stopped1/after1/stopped2/after2 order.")
    snapshot, identity = saved_snapshot(), bundle_identity()
    if phase == "baseline":
        record.update(baseline=snapshot, bundle_identity=identity)
    else:
        require(snapshot == record["baseline"], "Complete code/history/result state changed across relaunch.")
        require(identity == record["bundle_identity"], "The installed bundle changed across relaunch.")
    row = {"phase": phase, "at_utc": datetime.now(timezone.utc).isoformat(),
           "complete_saved_state_unchanged": True, "native_ui_verified_by_helper": False}
    if phase.startswith("stopped"):
        row.update(stopped_state(record["phases"][-1]))
    else:
        port = read_json(DATA / ".runtime-port.json")["port"]
        require(type(port) is int and 1024 <= port <= 65535, "Invalid owned runtime port.")
        processes = owned_processes(process_table(), running=True)
        if phase != "baseline":
            previous = record["phases"][-2]
            old = {(r["pid"], r["started"]) for r in previous["owned_processes"]}
            require(all((r["pid"], r["started"]) not in old for r in processes),
                    "The app/runtime was reused instead of cold-started.")
        row.update(port=port, owned_processes=processes, status=live_status(port))
    record["phases"].append(row)
    record.update(status="passed" if phase == "after2" else "in_progress", owned_qa_only=True,
                  helper_executes_code=False, helper_opens_project=False, helper_changes_qa_state=False,
                  native_ui_verified_by_helper=False, startup_root_cause_established=False,
                  public_release_delivered=False)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return {"status": record["status"], "phase": phase, "saved_tables": 112,
            "helper_executes_code": False, "native_ui_verified_by_helper": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_phase(args.phase, args.receipt.expanduser().resolve())
    except Exception as exc:
        # Never echo HTTP headers, session descriptors, response bodies or code.
        if isinstance(exc, RuntimeError):
            parser.exit(1, str(exc) + "\n")
        parser.exit(1, f"Read-only relaunch verification failed ({type(exc).__name__}).\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
