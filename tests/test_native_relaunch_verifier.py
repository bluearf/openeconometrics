"""Protocol tests use synthetic files/processes; they never launch a native app."""
import errno
import importlib.util
import json
from pathlib import Path
import plistlib
from types import SimpleNamespace

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "native_relaunch_verifier", Path(__file__).parents[1] / "scripts/verify_native_relaunch.py")
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def owned(tmp_path, monkeypatch):
    data, app, evidence = tmp_path / "qa-data", tmp_path / "Owned QA.app", tmp_path / "prior-evidence"
    monkeypatch.setattr(verifier, "DATA", data)
    monkeypatch.setattr(verifier, "APP", app)
    monkeypatch.setattr(verifier, "EVIDENCE", evidence)
    console = data / "projects" / verifier.PROJECT / "console"
    results = data / "complete-binary-mediation"
    proof = {"frozen": True, "all_eight_contracts_verified": True,
             "full_state_replay_verified": True, "saved_tables": 112, "methods": {}}
    hashes, outputs = {}, []
    for method in verifier.METHODS:
        payload = {"attrs": {"binary_mediation_state": {"full_state": True}},
                   "latex": "\\begin{tabular} saved complete output",
                   "tables": {key: {"columns": ["value"], "data": [[1]], "index": [0]}
                              for key in sorted(verifier.TABLES)}}
        path = results / (method + ".json")
        write_json(path, payload)
        hashes[method] = verifier.digest(path)
        proof["methods"][method] = {"sha256": hashes[method], "displayed_keys": ["parameters", "effects"],
                                    "table_rows": {key: 1 for key in verifier.TABLES}}
        outputs.extend({"type": "table", "data": {"columns": ["value"], "rows": [[1]], "total_rows": 1}}
                       for _ in range(2))
    run = {"id": verifier.EXECUTION, "status": "ok", "code": "saved synthetic code",
           "stdout": verifier.MARKER + json.dumps(proof), "outputs": outputs, "events": []}
    write_json(console / "history.json", {"history": [run]})
    write_json(console / "scripts.json", {"schema": 1, "analysis": {}, "scripts": [verifier.SCRIPT]})
    write_json(console / "script.json", {"name": "analysis.py", "code": "other saved code"})
    write_json(console / (verifier.SCRIPT + ".json"),
               {"id": verifier.SCRIPT, "name": "example.py", "code": run["code"], "version": 3})
    write_json(data / "local-projects.json", {"projects": [{"id": verifier.PROJECT,
                                                          "name": verifier.NAME, "description": "synthetic"}]})
    write_json(data / ".runtime-port.json", {"port": 12345})
    write_json(evidence / "installed.json", {"project_id": verifier.PROJECT, "script_id": verifier.SCRIPT,
                                            "execution_id": verifier.EXECUTION,
                                            "execution_sha256": verifier.json_hash(run),
                                            "complete_result_hashes": hashes})
    native = app / "Contents/MacOS/openecon-desktop"
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    native.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    native.write_bytes(b"owned native")
    runtime.write_bytes(b"owned frozen runtime")
    with (app / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleIdentifier": "org.openecon.qa.binarymediation"}, stream)
    write_json(evidence / "frozen.json", {"status": "passed", "source_head": "original6759",
                                         "runtime_sha256": verifier.digest(runtime)})
    write_json(evidence / "qa-bundle.json", {"status": "passed", "source_head": "original6759",
                                            "build": {"identifier": "org.openecon.qa.binarymediation"}})
    env = SimpleNamespace(data=data, app=app, console=console, results=results, evidence=evidence,
                          receipt=tmp_path / "new-evidence/readback.json", processes=[])
    def processes(start):
        return [{"pid": start, "ppid": 1, "started": f"app start {start}", "executable": str(native)},
                {"pid": start+1, "ppid": start, "started": f"runtime start {start}", "executable": str(runtime)},
                {"pid": start+2, "ppid": start+1, "started": f"child start {start}", "executable": "/owned/child"}]
    env.processes_for = processes
    env.processes = processes(100)
    monkeypatch.setattr(verifier, "process_table", lambda: env.processes)
    monkeypatch.setattr(verifier, "live_status", lambda port: {"project_id": verifier.PROJECT,
                        "console_pid": None, "console_running": False, "console_execution_id": None,
                        "session_generation": 2})
    monkeypatch.setattr(verifier, "port_closed", lambda port: True)
    return env


def test_two_full_cold_starts_preserve_all_state_and_track_descendants(owned):
    verifier.verify_phase("baseline", owned.receipt)
    baseline = json.loads(owned.receipt.read_text())["baseline"]
    assert baseline["saved_tables"] == sum(len(v) for v in baseline["saved_table_shapes"].values()) == 112
    assert baseline["console_file_sha256"].keys() == {"history.json", "scripts.json", "script.json", verifier.SCRIPT+".json"}
    for stopped, after, pid in (("stopped1", "after1", 200), ("stopped2", "after2", 300)):
        owned.processes = []
        verifier.verify_phase(stopped, owned.receipt)
        owned.processes = owned.processes_for(pid)
        verifier.verify_phase(after, owned.receipt)
    receipt = json.loads(owned.receipt.read_text())
    assert receipt["status"] == "passed"
    assert [r["phase"] for r in receipt["phases"]] == list(verifier.PHASES)
    assert len(receipt["phases"][0]["owned_processes"]) == 3
    assert not receipt["helper_executes_code"] and not receipt["helper_opens_project"]
    assert not receipt["native_ui_verified_by_helper"] and not receipt["startup_root_cause_established"]


@pytest.mark.parametrize("target", ["script.json", "scripts.json", "new-script.json", "history.json"])
def test_complete_console_state_mutations_refused_without_receipt_write(owned, target):
    verifier.verify_phase("baseline", owned.receipt)
    before = owned.receipt.read_bytes()
    if target == "history.json":
        state = verifier.read_json(owned.console / target)
        state["history"].append({"id": "new-run", "code": "unexpected execution"})
        write_json(owned.console / target, state)
    else:
        write_json(owned.console / target, {"code": "unexpected changed or added script"})
    owned.processes = []
    with pytest.raises(RuntimeError, match="Complete code/history/result state changed"):
        verifier.verify_phase("stopped1", owned.receipt)
    assert owned.receipt.read_bytes() == before


def test_old_installed_execution_is_verified_before_new_baseline(owned):
    history = verifier.read_json(owned.console / "history.json")
    history["history"][0]["events"] = [{"type": "changed"}]
    write_json(owned.console / "history.json", history)
    with pytest.raises(RuntimeError, match="complete installed execution changed"):
        verifier.verify_phase("baseline", owned.receipt)
    assert not owned.receipt.exists()


def test_complete_json_bytes_and_not_only_table_shapes_are_verified(owned):
    path = owned.results / (verifier.METHODS[0] + ".json")
    state = verifier.read_json(path)
    state["tables"]["covariance"]["data"][0][0] = 99
    write_json(path, state)
    with pytest.raises(RuntimeError, match="complete installed result changed"):
        verifier.saved_snapshot()


def test_symlinked_console_or_results_are_refused(owned):
    path = owned.console / "script.json"
    path.rename(owned.data / "linked-code.json")
    path.symlink_to(owned.data / "linked-code.json")
    with pytest.raises(RuntimeError, match="linked QA console"):
        verifier.saved_snapshot()


def test_frozen_runtime_and_receipt_identity_are_pinned(owned):
    runtime = owned.app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    runtime.write_bytes(b"changed runtime")
    with pytest.raises(RuntimeError, match="frozen QA runtime changed"):
        verifier.bundle_identity()


def test_stage_order_and_reused_processes_do_not_prove_cold_start(owned):
    with pytest.raises(RuntimeError, match="Run each verification phase"):
        verifier.verify_phase("after1", owned.receipt)
    verifier.verify_phase("baseline", owned.receipt)
    owned.processes = []
    verifier.verify_phase("stopped1", owned.receipt)
    owned.processes = owned.processes_for(100)
    with pytest.raises(RuntimeError, match="reused instead of cold-started"):
        verifier.verify_phase("after1", owned.receipt)


def test_stopped_requires_orphan_descendants_gone_and_negative_tcp(owned, monkeypatch):
    previous = {"owned_processes": owned.processes, "port": 12345}
    owned.processes = [{**owned.processes[-1], "ppid": 1}]
    with pytest.raises(RuntimeError, match="Full Quit did not stop"):
        verifier.stopped_state(previous, timeout=0)
    owned.processes = []
    monkeypatch.setattr(verifier, "port_closed", lambda port: False)
    with pytest.raises(RuntimeError, match="Full Quit did not stop"):
        verifier.stopped_state(previous, timeout=0)


def test_exact_process_paths_and_parentage_exclude_other_apps(owned):
    other = {"pid": 900, "ppid": 1, "started": "elsewhere", "executable": "/Other QA.app/openecon-runtime"}
    assert verifier.owned_processes([*owned.processes, other], running=True) == owned.processes
    detached = [{**owned.processes[1], "ppid": 1}, owned.processes[0]]
    with pytest.raises(RuntimeError, match="not a descendant"):
        verifier.owned_processes(detached, running=True)


@pytest.mark.parametrize("change", [{"pid": 987}, {"running": True}, {"execution_id": "new-run"}])
def test_live_status_refuses_worker_or_execution(monkeypatch, change):
    console = {"pid": None, "running": False, "execution_id": None, "session_generation": 2, **change}
    monkeypatch.setattr(verifier, "get_json", lambda port, path, token=None:
                        {"token": "never-persisted"} if path.endswith("/session")
                        else {"project_id": verifier.PROJECT, "console": console})
    with pytest.raises(RuntimeError, match="console worker/execution exists"):
        verifier.live_status(12345)


def test_live_status_does_not_activate_inactive_project(monkeypatch):
    calls = []
    def get(port, path, token=None):
        calls.append(path)
        return {"token": "never-persisted"} if path.endswith("/session") else {"project_id": None, "console": None}
    monkeypatch.setattr(verifier, "get_json", get)
    with pytest.raises(RuntimeError, match="through the GUI first"):
        verifier.live_status(12345)
    assert calls == ["/api/desktop/session", "/api/desktop/status"]


def test_api_allowlist_refuses_even_get_project_session():
    with pytest.raises(RuntimeError, match="Only passive desktop GETs"):
        verifier.get_json(12345, f"/api/desktop/projects/{verifier.PROJECT}/workspace/session")


@pytest.mark.parametrize("result,expected", [(errno.ECONNREFUSED, True), (errno.ETIMEDOUT, False), (0, False)])
def test_only_connection_refused_proves_loopback_closed(monkeypatch, result, expected):
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def settimeout(self, timeout):
            assert timeout == 1
        def connect_ex(self, address):
            assert address == ("127.0.0.1", 12345)
            return result
    monkeypatch.setattr(verifier.socket, "socket", Connection)
    assert verifier.port_closed(12345) is expected


def test_receipt_cannot_be_written_into_app_data(owned):
    with pytest.raises(RuntimeError, match="outside the owned app/data"):
        verifier.verify_phase("baseline", owned.data / "receipt.json")


def test_failure_output_does_not_echo_tokens_or_http_bodies(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["verify", "--phase", "baseline", "--receipt", "/tmp/receipt.json"])
    def fail(*args):
        raise OSError("HTTP body contained token SECRET-NOT-LOGGED")
    monkeypatch.setattr(verifier, "verify_phase", fail)
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 1
    assert "SECRET-NOT-LOGGED" not in capsys.readouterr().err
