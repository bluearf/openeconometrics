"""Read-only full-Quit/relaunch evidence for the owned Rank Ordered QA app.

Run baseline, stopped, after around GUI-operated Quit/relaunch/project selection.
Baseline permits an idle console worker left by the completed native Run. This
helper never launches, quits, opens a project, runs code, or changes QA data.
Only a machine-readable evidence receipt outside the app/profile is written.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import plistlib
import re

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "org.openecon.qa.rankordered"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
APP = Path.home() / "Applications/OpenEconometrics Rank Ordered QA.app"
EVIDENCE = ROOT / "docs/evidence/rank-ordered-four-2026-10-07"
NAME = "Rank Ordered Choice QA"
METHODS = ("oim", "hc0", "cr0", "first", "stages", "effects", "elasticities")
PHASES = ("baseline", "stopped", "after")
MARKER = "RANK_ORDERED_ACCEPTANCE_OK "
FIT_TABLES = {
    "inputs",
    "parameters",
    "information",
    "bread",
    "meat",
    "covariance",
    "case_scores",
    "cluster_scores",
    "stage_scores",
    "stages",
    "case_likelihood",
    "probabilities",
    "probability_jacobian",
    "fit_summary",
}
PREDICT_TABLES = {
    "predictions",
    "covariance",
    "log_probability_covariance",
    "joint_covariance",
    "joint_jacobian",
    "log_odds_covariance",
    "jacobian",
    "log_probability_jacobian",
    "log_odds_jacobian",
    "coefficient_covariance",
}
MARGIN_TABLES = {
    "margins",
    "per_case",
    "support",
    "covariance",
    "jacobian",
    "per_case_jacobian",
    "coefficient_covariance",
}

# Load only the reviewed process/passive-HTTP helpers, with no runtime imports.
_spec = importlib.util.spec_from_file_location(
    "rank_owned_process_protocol", Path(__file__).with_name("verify_native_relaunch.py")
)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)
process_table = native.process_table
port_closed = native.port_closed
get_json = native.get_json
require = native.require


def no_links(path):
    path = path.expanduser().absolute()
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "Linked QA/evidence paths are refused.",
    )
    return path


def digest(path):
    path = no_links(path)
    require(path.is_file(), "Missing owned QA/evidence file.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    digest(path)
    return json.loads(path.read_text())


def json_hash(value):
    # Identical complete-execution hash convention to installed.json.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def installed_receipt():
    installed = read_json(EVIDENCE / "installed.json")
    require(
        installed["status"] == "passed"
        and installed["saved_tables"] == 76
        and installed["displayed_tables"] == 7
        and installed["all_four_scopes_saved"] is True
        and installed["all_saved_inputs_replayed"] is True,
        "Incomplete installed acceptance receipt.",
    )
    for key in ("project_id", "script_id", "execution_id"):
        require(
            isinstance(installed[key], str)
            and re.fullmatch(
                r"[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", installed[key]
            ),
            "Invalid owned installed identifier.",
        )
    require(
        set(installed["complete_result_hashes"]) == set(METHODS),
        "Incomplete installed result hashes.",
    )
    return installed


def saved_snapshot():
    """Snapshot every console JSON byte and all seven complete 76-table files."""
    installed = installed_receipt()
    project, script_id, execution = (
        installed[key] for key in ("project_id", "script_id", "execution_id")
    )
    console, directory = DATA / "projects" / project / "console", DATA / "complete-rank-ordered"
    for folder in (DATA, DATA / "projects", console.parent, console, directory):
        require(no_links(folder).is_dir(), "Missing owned QA directory.")
    catalog = read_json(DATA / "local-projects.json")
    require(
        len(catalog["projects"]) == 1
        and catalog["projects"][0]["id"] == project
        and catalog["projects"][0]["name"] == NAME,
        "The owned QA project catalog changed.",
    )
    entries = sorted(console.iterdir())
    require(
        all(not f.is_symlink() and f.is_file() and f.suffix == ".json" for f in entries),
        "Unexpected/linked QA console entry.",
    )
    console_hashes = {f.name: digest(f) for f in entries}
    require(
        {"history.json", "scripts.json", "script.json", script_id + ".json"} <= set(console_hashes),
        "Incomplete saved console documents.",
    )
    history = read_json(console / "history.json")["history"]
    require(
        isinstance(history, list) and all(isinstance(row, dict) for row in history),
        "Invalid saved console history.",
    )
    matches = [row for row in history if row.get("id") == execution]
    require(
        len(matches) == 1 and json_hash(matches[0]) == installed["execution_sha256"],
        "Complete installed execution missing, duplicated or changed.",
    )
    run = matches[0]
    script = read_json(console / (script_id + ".json"))
    require(
        script["id"] == script_id and script["code"] == run["code"],
        "The saved execution/script code disagrees.",
    )
    markers = [
        line[len(MARKER) :] for line in run["stdout"].splitlines() if line.startswith(MARKER)
    ]
    require(len(markers) == 1, "Missing or duplicated acceptance marker.")
    proof = json.loads(markers[0])
    require(
        run["status"] == "ok"
        and proof["frozen"] is True
        and proof["all_four_scopes_verified"] is True
        and proof["full_state_replay_verified"] is True
        and proof["third_party_estimation_imports"] == []
        and proof["saved_tables"] == 76
        and proof["displayed_tables"] == 7
        and set(proof["methods"]) == set(METHODS),
        "Incomplete saved scientific acceptance proof.",
    )
    outputs = run["outputs"]
    require(
        len(outputs) == 7 and all(item["type"] == "table" for item in outputs),
        "Incomplete installed rendered outputs.",
    )
    require(
        {f.name for f in directory.iterdir()} == {name + ".json" for name in METHODS},
        "Missing or unexpected complete result file.",
    )
    hashes, shapes = {}, {}
    for offset, name in enumerate(METHODS):
        path = directory / (name + ".json")
        hashes[name] = digest(path)
        require(
            hashes[name]
            == installed["complete_result_hashes"][name]
            == proof["methods"][name]["sha256"],
            "A complete installed result changed.",
        )
        payload = read_json(path)
        tables = (
            FIT_TABLES
            if name in METHODS[:3]
            else PREDICT_TABLES
            if name in METHODS[3:5]
            else MARGIN_TABLES
        )
        require(
            set(payload) == {"tables", "attrs", "latex"}
            and set(payload["tables"]) == tables
            and "\\begin{tabular}" in payload["latex"],
            "Incomplete complete-result schema or LaTeX.",
        )
        attrs = payload["attrs"]
        fit = name in METHODS[:3]
        require(
            attrs["contract"] == ("rank_ordered_v1" if fit else "rank_ordered_postestimation_v1")
            and ("rank_ordered_state" if fit else "source_fit_state") in attrs
            and isinstance(attrs["settings"], dict),
            "Incomplete saved scientific attrs/state.",
        )
        shapes[name] = {}
        for key, frame in payload["tables"].items():
            columns, rows, index = frame["columns"], frame["data"], frame["index"]
            require(
                len(set(columns)) == len(columns)
                and len(index) == len(rows)
                and all(len(row) == len(columns) for row in rows)
                and len(rows) == proof["methods"][name]["table_rows"][key],
                "Incomplete complete table columns, index or rows.",
            )
            shapes[name][key] = {"rows": len(rows), "columns": len(columns)}
        key = "parameters" if fit else "predictions" if name in METHODS[3:5] else "margins"
        require(
            proof["methods"][name]["displayed_keys"] == [key], "Unexpected saved display selection."
        )
        output = outputs[offset]
        require(
            output["data"]["total_rows"] == len(output["data"]["rows"]) == shapes[name][key]["rows"]
            and any("\\begin{" + env + "}" in output["latex"] for env in ("tabular", "longtable")),
            "Installed display truncated or LaTeX missing.",
        )
    require(sum(len(rows) for rows in shapes.values()) == 76, "Incomplete saved table count.")
    return dict(
        installed_receipt_sha256=digest(EVIDENCE / "installed.json"),
        project_id=project,
        script_id=script_id,
        execution_id=execution,
        execution_sha256=json_hash(run),
        catalog_sha256=digest(DATA / "local-projects.json"),
        console_file_sha256=console_hashes,
        complete_history_sha256=json_hash(history),
        history_count=len(history),
        history_ids=[row["id"] for row in history],
        complete_result_hashes=hashes,
        saved_table_shapes=shapes,
        saved_tables=76,
        displayed_tables=7,
    )


def bundle_identity():
    frozen, bundle = read_json(EVIDENCE / "frozen.json"), read_json(EVIDENCE / "qa-bundle.json")
    require(
        frozen["status"] == bundle["status"] == "passed"
        and frozen["source_head"] == bundle["source_head"],
        "Frozen/native bundle receipts disagree.",
    )
    require(bundle["build"]["identifier"] == IDENTIFIER, "Unexpected owned QA bundle receipt.")
    info = no_links(APP / "Contents/Info.plist")
    digest(info)
    with info.open("rb") as stream:
        require(
            plistlib.load(stream)["CFBundleIdentifier"] == IDENTIFIER,
            "Installed app is not the owned Rank Ordered QA bundle.",
        )
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    require(digest(runtime) == frozen["runtime_sha256"], "The accepted frozen QA runtime changed.")
    return dict(
        source_head=frozen["source_head"],
        runtime_sha256=digest(runtime),
        native_sha256=digest(APP / "Contents/MacOS/openecon-desktop"),
        frozen_receipt_sha256=digest(EVIDENCE / "frozen.json"),
        qa_bundle_receipt_sha256=digest(EVIDENCE / "qa-bundle.json"),
    )


@contextmanager
def rebound_helpers():
    # Private imported module, rebound only for this synchronous helper call.
    previous = native.APP, native.process_table, native.port_closed
    native.APP, native.process_table, native.port_closed = APP, process_table, port_closed
    try:
        yield
    finally:
        native.APP, native.process_table, native.port_closed = previous


def owned_processes(rows, *, running):
    with rebound_helpers():
        return native.owned_processes(rows, running=running)


def stopped_state(previous, *, timeout=15):
    with rebound_helpers():
        return native.stopped_state(previous, timeout=timeout)


def live_status(port, project, *, allow_worker):
    token = get_json(port, "/api/desktop/session")["token"]
    status = get_json(port, "/api/desktop/status", token)
    require(
        status["project_id"] == project, "Select the owned saved QA project through the GUI first."
    )
    console = status["console"]
    require(
        isinstance(console, dict)
        and console.get("running") is False
        and console.get("execution_id") is None,
        "A running execution cannot establish read-only persistence.",
    )
    pid = console.get("pid")
    require(
        pid is None or (allow_worker and type(pid) is int and pid > 0),
        "A console worker exists after relaunch.",
    )
    require(
        type(console.get("session_generation")) is int and console["session_generation"] >= 0,
        "Invalid console session generation.",
    )
    return dict(
        project_id=project,
        console_pid=pid,
        console_running=False,
        console_execution_id=None,
        session_generation=console["session_generation"],
        passive_routes=["/api/desktop/session", "/api/desktop/status"],
    )


def verify_phase(phase, receipt):
    receipt = no_links(receipt).resolve()
    require(phase in PHASES, "Unknown verification phase.")
    require(
        not receipt.is_relative_to(no_links(DATA).resolve())
        and not receipt.is_relative_to(no_links(APP).resolve()),
        "Evidence must remain outside the owned app/profile.",
    )
    record = (
        read_json(receipt)
        if receipt.exists()
        else {"schema": "rank_ordered_native_relaunch_v1", "phases": []}
    )
    require(
        record["schema"] == "rank_ordered_native_relaunch_v1"
        and [row["phase"] for row in record["phases"]] == list(PHASES[: PHASES.index(phase)]),
        "Use each phase once in baseline/stopped/after order.",
    )
    snapshot, identity = saved_snapshot(), bundle_identity()
    if phase == "baseline":
        record.update(baseline=snapshot, bundle_identity=identity)
    else:
        require(
            snapshot == record["baseline"],
            "Complete saved console/history/result bytes changed across relaunch.",
        )
        require(
            identity == record["bundle_identity"],
            "The installed bundle or accepted receipts changed across relaunch.",
        )
    row = dict(
        phase=phase,
        at_utc=datetime.now(timezone.utc).isoformat(),
        complete_saved_state_unchanged=True,
        native_ui_verified_by_helper=False,
    )
    if phase == "stopped":
        row.update(stopped_state(record["phases"][0]))
    else:
        port = read_json(DATA / ".runtime-port.json")["port"]
        require(type(port) is int and 1024 <= port <= 65535, "Invalid owned runtime port.")
        processes = owned_processes(process_table(), running=True)
        status = live_status(port, snapshot["project_id"], allow_worker=phase == "baseline")
        if status["console_pid"] is not None:
            require(
                status["console_pid"] in {p["pid"] for p in processes},
                "Baseline worker is not an owned descendant.",
            )
        if phase == "after":
            old = record["phases"][0]["owned_processes"]
            runtime_path = str(APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime")
            require(
                sum(p["executable"] == runtime_path for p in processes) == 1,
                "An extra bundled-runtime child exists after relaunch.",
            )
            require(
                not ({p["pid"] for p in processes} & {p["pid"] for p in old}),
                "Owned processes were reused instead of fresh cold-start PIDs.",
            )
        require(
            saved_snapshot() == snapshot, "Saved state changed during passive status verification."
        )
        row.update(port=port, owned_processes=processes, status=status)
    record["phases"].append(row)
    record.update(
        status="passed" if phase == "after" else "in_progress",
        owned_qa_only=True,
        helper_executes_code=False,
        helper_opens_project=False,
        helper_launches_or_quits=False,
        helper_changes_qa_state=False,
        native_ui_verified_by_helper=False,
        startup_root_cause_established=False,
        public_release_delivered=False,
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return dict(
        status=record["status"],
        phase=phase,
        saved_tables=76,
        displayed_tables=7,
        helper_executes_code=False,
        native_ui_verified_by_helper=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_phase(args.phase, args.receipt)
    except Exception as exc:
        # Never expose HTTP bodies, tokens, headers, code or arbitrary OS errors.
        print(
            json.dumps(
                dict(
                    status="failed",
                    phase=args.phase,
                    error=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__,
                )
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
