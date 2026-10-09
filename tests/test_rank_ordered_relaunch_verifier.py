"""Synthetic protocol/files/process tests; no native app or QA profile is accessed."""

import importlib.util
import json
from pathlib import Path
import plistlib
from types import SimpleNamespace

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "rank_relaunch_verifier", Path(__file__).parents[1] / "scripts/verify_rank_ordered_relaunch.py"
)
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def owned(tmp_path, monkeypatch):
    data, app, evidence = tmp_path / "profile", tmp_path / "Rank QA.app", tmp_path / "accepted"
    project, script_id, execution = "1" * 32, "2" * 32, "33333333-3333-3333-3333-333333333333"
    for name, value in (("DATA", data), ("APP", app), ("EVIDENCE", evidence)):
        monkeypatch.setattr(verifier, name, value)
    console, directory = data / "projects" / project / "console", data / "complete-rank-ordered"
    proof = dict(
        frozen=True,
        all_four_scopes_verified=True,
        full_state_replay_verified=True,
        third_party_estimation_imports=[],
        saved_tables=76,
        displayed_tables=7,
        methods={},
    )
    hashes, outputs = {}, []
    for name in verifier.METHODS:
        fit = name in verifier.METHODS[:3]
        tables = (
            verifier.FIT_TABLES
            if fit
            else verifier.PREDICT_TABLES
            if name in verifier.METHODS[3:5]
            else verifier.MARGIN_TABLES
        )
        key = "parameters" if fit else "predictions" if name in verifier.METHODS[3:5] else "margins"
        payload = dict(
            attrs={
                "contract": "rank_ordered_v1" if fit else "rank_ordered_postestimation_v1",
                "settings": {},
                "rank_ordered_state" if fit else "source_fit_state": {"complete": True},
            },
            latex="\\begin{tabular} full76table payload",
            tables={t: dict(columns=["value"], index=[0], data=[[1.0]]) for t in tables},
        )
        path = directory / (name + ".json")
        write_json(path, payload)
        hashes[name] = verifier.digest(path)
        proof["methods"][name] = dict(
            sha256=hashes[name], table_rows={t: 1 for t in tables}, displayed_keys=[key]
        )
        outputs.append(
            dict(
                type="table",
                data=dict(columns=["value"], rows=[[1.0]], total_rows=1),
                latex="\\begin{tabular} complete",
            )
        )
    run = dict(
        id=execution,
        status="ok",
        code="complete synthetic saved code",
        stdout=verifier.MARKER + json.dumps(proof),
        outputs=outputs,
        events=[],
    )
    write_json(console / "history.json", {"history": [run]})
    write_json(console / "scripts.json", {"scripts": [script_id]})
    write_json(console / "script.json", {"code": "other saved document"})
    write_json(console / (script_id + ".json"), {"id": script_id, "code": run["code"]})
    write_json(data / "local-projects.json", {"projects": [dict(id=project, name=verifier.NAME)]})
    write_json(data / ".runtime-port.json", {"port": 12345})
    write_json(
        evidence / "installed.json",
        dict(
            status="passed",
            project_id=project,
            script_id=script_id,
            execution_id=execution,
            execution_sha256=verifier.json_hash(run),
            complete_result_hashes=hashes,
            saved_tables=76,
            displayed_tables=7,
            all_four_scopes_saved=True,
            all_saved_inputs_replayed=True,
        ),
    )
    native = app / "Contents/MacOS/openecon-desktop"
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    native.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    native.write_bytes(b"owned native executable")
    runtime.write_bytes(b"accepted frozen runtime")
    with (app / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleIdentifier": verifier.IDENTIFIER}, stream)
    write_json(
        evidence / "frozen.json",
        dict(status="passed", source_head="science-pin", runtime_sha256=verifier.digest(runtime)),
    )
    write_json(
        evidence / "qa-bundle.json",
        dict(status="passed", source_head="science-pin", build={"identifier": verifier.IDENTIFIER}),
    )
    env = SimpleNamespace(
        data=data,
        app=app,
        evidence=evidence,
        console=console,
        directory=directory,
        receipt=tmp_path / "read-only-evidence/relaunch.json",
        project=project,
        worker=102,
    )

    def processes(start, worker=False):
        result = [
            dict(pid=start, ppid=1, started=f"app {start}", executable=str(native)),
            dict(pid=start + 1, ppid=start, started=f"runtime {start}", executable=str(runtime)),
        ]
        if worker:
            result.append(
                dict(
                    pid=start + 2,
                    ppid=start + 1,
                    started=f"worker {start}",
                    executable=str(runtime),
                )
            )
        return result

    env.processes_for = processes
    env.processes = processes(100, worker=True)
    monkeypatch.setattr(verifier, "process_table", lambda: env.processes)
    monkeypatch.setattr(verifier, "port_closed", lambda port: True)
    calls = []

    def get(port, path, token=None):
        calls.append(path)
        if path.endswith("/session"):
            return {"token": "NEVER-PERSIST-THIS-TOKEN"}
        assert token == "NEVER-PERSIST-THIS-TOKEN"
        return {
            "project_id": env.project,
            "console": dict(pid=env.worker, running=False, execution_id=None, session_generation=2),
        }

    env.calls = calls
    monkeypatch.setattr(verifier, "get_json", get)
    return env


def baseline_stopped(owned):
    verifier.verify_phase("baseline", owned.receipt)
    owned.processes = []
    verifier.verify_phase("stopped", owned.receipt)
    owned.processes = owned.processes_for(200)
    owned.worker = None


def test_complete76tables_idle_baseline_worker_stopped_and_fresh_no_worker_after(owned):
    baseline_stopped(owned)
    verifier.verify_phase("after", owned.receipt)
    receipt = json.loads(owned.receipt.read_text())
    assert receipt["status"] == "passed"
    assert len(receipt["phases"][0]["owned_processes"]) == 3
    assert len(receipt["phases"][2]["owned_processes"]) == 2
    assert (
        receipt["baseline"]["saved_tables"]
        == sum(len(rows) for rows in receipt["baseline"]["saved_table_shapes"].values())
        == 76
    )
    assert set(receipt["baseline"]["console_file_sha256"]) == {
        "history.json",
        "scripts.json",
        "script.json",
        "2" * 32 + ".json",
    }
    assert owned.calls == ["/api/desktop/session", "/api/desktop/status"] * 2
    assert "NEVER-PERSIST" not in owned.receipt.read_text()
    assert not receipt["helper_opens_project"] and not receipt["helper_launches_or_quits"]
    assert not receipt["helper_executes_code"] and not receipt["native_ui_verified_by_helper"]


@pytest.mark.parametrize("target", ["script.json", "scripts.json", "new.json", "history.json"])
def test_every_console_byte_mutation_or_new_document_refused_without_receipt_write(owned, target):
    verifier.verify_phase("baseline", owned.receipt)
    previous = owned.receipt.read_bytes()
    write_json(owned.console / target, {"modified": True})
    owned.processes = []
    with pytest.raises((RuntimeError, KeyError)):
        verifier.verify_phase("stopped", owned.receipt)
    assert owned.receipt.read_bytes() == previous


@pytest.mark.parametrize("part", ["tables", "attrs", "latex"])
def test_full_result_points_attrs_and_latex_not_only_shapes_are_pinned(owned, part):
    path = owned.directory / "effects.json"
    saved = verifier.read_json(path)
    if part == "tables":
        saved[part]["covariance"]["data"][0][0] = 99
    elif part == "attrs":
        saved[part]["source_fit_state"]["complete"] = False
    else:
        saved[part] += "changed"
    write_json(path, saved)
    with pytest.raises(RuntimeError, match="complete installed result changed"):
        verifier.saved_snapshot()


@pytest.mark.parametrize("which", ["profile", "console", "result", "receipt_parent"])
def test_linked_paths_and_parent_folders_refused(owned, which, tmp_path):
    target = {
        "profile": owned.data,
        "console": owned.console,
        "result": owned.directory / "first.json",
        "receipt_parent": owned.receipt.parent,
    }[which]
    if which == "receipt_parent":
        target.mkdir()
    actual = tmp_path / ("actual-" + which)
    target.rename(actual)
    target.symlink_to(actual, target_is_directory=actual.is_dir())
    with pytest.raises(RuntimeError, match="Linked|linked"):
        verifier.verify_phase("baseline", owned.receipt)


def test_baseline_requires_original_installed_execution_and_safe_ids(owned):
    receipt = verifier.read_json(owned.evidence / "installed.json")
    receipt["execution_sha256"] = "changed"
    write_json(owned.evidence / "installed.json", receipt)
    with pytest.raises(RuntimeError, match="installed execution"):
        verifier.saved_snapshot()
    receipt["script_id"] = "../outside"
    write_json(owned.evidence / "installed.json", receipt)
    with pytest.raises(RuntimeError, match="identifier"):
        verifier.saved_snapshot()


def test_frozen_runtime_and_baseline_native_identity_are_pinned(owned):
    verifier.verify_phase("baseline", owned.receipt)
    native = owned.app / "Contents/MacOS/openecon-desktop"
    native.write_bytes(b"changed native")
    with pytest.raises(RuntimeError, match="installed bundle"):
        verifier.verify_phase("stopped", owned.receipt)
    runtime = owned.app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    runtime.write_bytes(b"changed runtime")
    with pytest.raises(RuntimeError, match="frozen QA runtime"):
        verifier.bundle_identity()


def test_rebound_helpers_restore_original_app_and_only_own_descendants(owned):
    previous = verifier.native.APP
    other = dict(pid=900, ppid=1, started="other", executable="/Other.app/runtime")
    assert verifier.owned_processes([*owned.processes, other], running=True) == owned.processes
    assert verifier.native.APP == previous
    detached = [owned.processes[0], {**owned.processes[1], "ppid": 1}]
    with pytest.raises(RuntimeError, match="not a descendant"):
        verifier.owned_processes(detached, running=True)
    assert verifier.native.APP == previous


def test_full_quit_requires_recorded_orphan_worker_gone_and_refused_port(owned, monkeypatch):
    previous = dict(owned_processes=owned.processes, port=12345)
    owned.processes = [{**owned.processes[-1], "ppid": 1}]
    with pytest.raises(RuntimeError, match="Full Quit"):
        verifier.stopped_state(previous, timeout=0)
    owned.processes = []
    monkeypatch.setattr(verifier, "port_closed", lambda port: False)
    with pytest.raises(RuntimeError, match="Full Quit"):
        verifier.stopped_state(previous, timeout=0)


@pytest.mark.parametrize("which", ["worker", "reused", "extra_runtime"])
def test_after_no_worker_and_fresh_processes_are_required(owned, which):
    baseline_stopped(owned)
    if which == "worker":
        owned.worker = 202
    elif which == "reused":
        owned.processes = owned.processes_for(100)
    else:
        owned.processes = owned.processes_for(200, worker=True)
    previous = owned.receipt.read_bytes()
    with pytest.raises(RuntimeError, match="worker|fresh|extra bundled"):
        verifier.verify_phase("after", owned.receipt)
    assert owned.receipt.read_bytes() == previous


def test_inactive_project_is_not_activated_and_get_allowlist_is_passive(owned):
    owned.project = None
    with pytest.raises(RuntimeError, match="through the GUI"):
        verifier.verify_phase("baseline", owned.receipt)
    assert owned.calls == ["/api/desktop/session", "/api/desktop/status"]
    with pytest.raises(RuntimeError, match="Only passive"):
        verifier.native.get_json(12345, "/api/desktop/projects/owned/workspace/session")


def test_phase_order_and_profile_receipt_writes_refused(owned):
    with pytest.raises(RuntimeError, match="phase once"):
        verifier.verify_phase("after", owned.receipt)
    with pytest.raises(RuntimeError, match="outside"):
        verifier.verify_phase("baseline", owned.data / "receipt.json")


def test_unexpected_http_error_is_machine_readable_without_secret(owned, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["verify", "--phase", "baseline", "--receipt", str(owned.receipt)]
    )

    def fail(*args):
        raise OSError("HTTP body token SECRET-NEVER-ECHO")

    monkeypatch.setattr(verifier, "verify_phase", fail)
    with pytest.raises(SystemExit) as error:
        verifier.main()
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert json.loads(output)["error"] == "OSError"
    assert "SECRET" not in output
