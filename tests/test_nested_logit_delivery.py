"""Owned delivery proof regressions using synthetic files and mocked processes.

No native app, real process inventory, HTTP endpoint, or user profile is used.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import plistlib
import sys
from types import SimpleNamespace

import pytest


_SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
_CONFIGURATIONS = {
    "nested_logit": dict(module="verify_nested_logit_delivery", directory="complete-nested-choice",
                         saved_tables=65, displayed_tables=7, script_name="nested_logit.py"),
    "nonlinear_sur": dict(module="verify_nonlinear_sur_delivery", directory="complete-nonlinear-sur",
                          saved_tables=86, displayed_tables=11, script_name="nonlinear_sur.py"),
    "mprobit": dict(module="verify_mprobit_delivery", directory="complete-mprobit",
                    saved_tables=11, displayed_tables=11, script_name="mprobit.py"),
}
_VERIFIERS = {}
sys.path.insert(0, str(_SCRIPTS))
try:
    for _name, _config in _CONFIGURATIONS.items():
        _spec = importlib.util.spec_from_file_location(_config["module"], _SCRIPTS/(_config["module"]+".py"))
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        _VERIFIERS[_name] = _module
finally:
    sys.path.remove(str(_SCRIPTS))

verifier = _VERIFIERS["nested_logit"]
configuration = _CONFIGURATIONS["nested_logit"]


@pytest.fixture(autouse=True, params=tuple(_CONFIGURATIONS), ids=tuple(_CONFIGURATIONS))
def select_verifier(request, monkeypatch):
    """Run each owned protocol regression unchanged against the method couriers."""
    monkeypatch.setitem(globals(), "verifier", _VERIFIERS[request.param])
    monkeypatch.setitem(globals(), "configuration", _CONFIGURATIONS[request.param])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def owned(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    data, app = root/"owned-profile", root/"Synthetic QA.app"
    monkeypatch.setattr(verifier, "DATA", data)
    monkeypatch.setattr(verifier, "APP", app)
    project, script = "1"*32, "2"*32
    execution = "33333333-3333-3333-3333-333333333333"
    console = data/"projects"/project/"console"
    results = data/configuration["directory"]
    proof = dict(frozen=True, all_eight_scopes_verified=True, full_state_replay_verified=True,
                 third_party_estimation_imports=[], saved_tables=configuration["saved_tables"],
                 displayed_tables=configuration["displayed_tables"])
    run = dict(id=execution, status="ok", code="synthetic accepted complete script",
               stdout=verifier.MARKER+json.dumps(proof),
               outputs=[dict(type="table", data=dict(columns=["value"], rows=[[1.]], total_rows=1)) for _ in range(configuration["displayed_tables"])],
               events=[], error=None, duration_ms=123.)
    write_json(console/"history.json", {"history": [run]})
    document = dict(id=script, name=configuration["script_name"], code=run["code"])
    write_json(console/(script+".json"), document)
    write_json(console/"script.json", document)
    write_json(console/"scripts.json", {"scripts": [script]})
    write_json(data/"local-projects.json", {"projects": [dict(id=project, name=verifier.NAME)]})
    hashes = {}
    for method in verifier.METHODS:
        # The cold protocol must preserve prior accepted bytes. Numerical
        # payload correctness belongs to source/frozen/readback verification.
        path = results/(method+".json")
        write_json(path, {"synthetic_previously_accepted_result": method})
        hashes[method] = verifier.digest(path)
    native = app/"Contents/MacOS/openecon-desktop"
    runtime = app/"Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    native.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    native.write_bytes(b"synthetic owned native")
    runtime.write_bytes(b"synthetic accepted frozen runtime")
    (app/"Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": verifier.IDENTIFIER}))
    installed = root/"outside-evidence/installed.json"
    accepted = dict(status="passed", project_id=project, script_id=script, execution_id=execution,
                    execution_sha256=verifier.json_hash(run), complete_result_hashes=hashes, proof=proof,
                    runtime_sha256=verifier.digest(runtime), native_sha256=verifier.digest(native),
                    bundle_info_sha256=verifier.digest(app/"Contents/Info.plist"),
                    native_gui_observed_by_helper=False, separate_gui_evidence_required=True)
    write_json(installed, accepted)
    env = SimpleNamespace(root=root, data=data, app=app, console=console, results=results,
                          project=project, script=script, execution=execution, run=run,
                          installed=installed, receipt=root/"outside-evidence/cold.json",
                          accepted=accepted, runtime=runtime, native=native, calls=[], worker=102)

    def processes(start, worker=False):
        rows = [dict(pid=start, ppid=1, started=f"native {start}", executable=str(native)),
                dict(pid=start+1, ppid=start, started=f"runtime {start}", executable=str(runtime))]
        if worker:
            rows.append(dict(pid=start+2, ppid=start+1, started=f"worker {start}", executable=str(runtime)))
        return rows
    env.processes_for, env.processes = processes, processes(100, worker=True)
    monkeypatch.setattr(verifier.processes, "process_table", lambda: env.processes)
    monkeypatch.setattr(verifier.processes, "owned_processes", lambda rows, **kwargs: list(rows))

    def stopped(previous):
        assert env.processes == []
        assert previous["owned_processes"]
        return dict(tracked_owned_process_identities_gone=True, owned_executable_paths_absent=True,
                    loopback_connection_refused=True, port=previous["port"])
    monkeypatch.setattr(verifier.processes, "stopped_state", stopped)

    def request(path, body=None, *, token=None, method=None):
        env.calls.append((path, body, method))
        assert path == "/api/desktop/status" and body is None and method is None
        assert token == "SYNTHETIC-TOKEN-MUST-NOT-BE-PERSISTED"
        return dict(project_id=env.project, console=dict(running=False, execution_id=None, pid=env.worker))
    monkeypatch.setattr(verifier, "native_session", lambda: (12345, request, "SYNTHETIC-TOKEN-MUST-NOT-BE-PERSISTED"))

    def forbidden(*args, **kwargs):
        pytest.fail("protocol test attempted an actual app/process/HTTP operation")
    monkeypatch.setattr(verifier.subprocess, "check_output", forbidden)
    monkeypatch.setattr(verifier, "build_opener", forbidden)
    monkeypatch.setattr(verifier, "OwnedRuntime", forbidden)
    return env


def phase(env, name):
    value = verifier.relaunch(name, env.receipt, env.installed)
    write_json(env.receipt, value)
    return value


def stop_and_prepare_cold(env):
    phase(env, "baseline")
    env.processes = []
    phase(env, "stopped")
    env.processes = env.processes_for(200)
    env.worker = None


def test_complete_accepted_execution_survives_passive_cold_protocol(owned):
    stop_and_prepare_cold(owned)
    record = phase(owned, "after")
    assert record["status"] == "passed"
    assert [row["phase"] for row in record["phases"]] == ["baseline", "stopped", "after"]
    assert set(record["saved_snapshot"]["console_files"]) == {"history.json", "script.json", "scripts.json", owned.script+".json"}
    assert record["phases"][1]["tracked_owned_process_identities_gone"]
    assert record["phases"][1]["loopback_connection_refused"]
    assert record["phases"][2]["no_computation_worker"]
    assert all(row["native_gui_observed_by_helper"] is False for row in record["phases"])
    assert owned.calls == [("/api/desktop/status", None, None)]*2
    assert "SYNTHETIC-TOKEN" not in owned.receipt.read_text()


def test_empty_console_cannot_certify_missing_accepted_execution(owned):
    for path in owned.console.iterdir():
        path.unlink()
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert not owned.receipt.exists() and owned.calls == []


@pytest.mark.parametrize("file", ("history.json", "script.json", "scripts.json", "accepted_script"))
def test_each_required_console_document_is_mandatory(owned, file):
    (owned.console/(owned.script+".json" if file == "accepted_script" else file)).unlink()
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert not owned.receipt.exists() and owned.calls == []


@pytest.mark.parametrize("field", ("code", "stdout", "outputs", "events", "status", "error", "duration_ms"))
def test_every_accepted_execution_field_is_bound_before_baseline(owned, field):
    run = copy.deepcopy(owned.run)
    run[field] = {"code": "changed source", "stdout": "changed output", "outputs": [], "events": ["changed"],
                  "status": "error", "error": "new error", "duration_ms": 456.}[field]
    write_json(owned.console/"history.json", {"history": [run]})
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert not owned.receipt.exists() and owned.calls == []


@pytest.mark.parametrize("history", ([], "duplicate", "different_id"))
def test_missing_duplicated_or_wrong_execution_is_rejected(owned, history):
    if history == "duplicate":
        rows = [owned.run, owned.run]
    elif history == "different_id":
        rows = [dict(owned.run, id="4"*32)]
    else:
        rows = history
    write_json(owned.console/"history.json", {"history": rows})
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert owned.calls == []


@pytest.mark.parametrize("field", ("id", "code"))
def test_accepted_script_identity_and_code_must_match_history(owned, field):
    document = dict(id=owned.script, code=owned.run["code"])
    document[field] = "5"*32 if field == "id" else "different script source"
    write_json(owned.console/(owned.script+".json"), document)
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert owned.calls == []


@pytest.mark.parametrize("change", ("project_id", "project_name", "other_project"))
def test_owned_catalogue_must_identify_exactly_the_accepted_synthetic_project(owned, change):
    catalogue = dict(projects=[dict(id=owned.project, name=verifier.NAME)])
    if change == "other_project":
        catalogue["projects"].append(dict(id="6"*32, name="unrelated synthetic project"))
    else:
        catalogue["projects"][0]["id" if change == "project_id" else "name"] = "unrecognized"
    write_json(owned.data/"local-projects.json", catalogue)
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert owned.calls == []


@pytest.mark.parametrize("field", ("project_id", "script_id", "execution_id"))
@pytest.mark.parametrize("bad", ("../outside", "/absolute/path", "1"*31, True))
def test_identifier_guard_rejects_traversal_or_invalid_types_before_profile_reads(owned, monkeypatch, field, bad):
    accepted = dict(owned.accepted, **{field: bad})
    write_json(owned.installed, accepted)
    actual = verifier.read_json
    def guarded_read(path):
        assert path == owned.installed, "identifier validation reached owned-profile files"
        return actual(path)
    monkeypatch.setattr(verifier, "read_json", guarded_read)
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert owned.calls == []


@pytest.mark.parametrize("target", ("installed", "cold", "history", "catalogue"))
def test_linked_receipt_or_console_input_is_refused(owned, target):
    path = {"installed": owned.installed, "cold": owned.receipt,
            "history": owned.console/"history.json", "catalogue": owned.data/"local-projects.json"}[target]
    source = owned.root/(target+"-link-target.json")
    if path.exists():
        source.write_bytes(path.read_bytes())
        path.unlink()
    else:
        write_json(source, {"phases": []})
    path.symlink_to(source)
    with pytest.raises(AssertionError):
        verifier.relaunch("baseline", owned.receipt, owned.installed)
    assert owned.calls == []


def test_linked_console_parent_is_refused_before_history_read(owned):
    source = owned.root/"real-synthetic-console"
    owned.console.rename(source)
    owned.console.symlink_to(source, target_is_directory=True)
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert owned.calls == []


def linked_output(env, kind):
    """Create a guarded path while preserving a synthetic external sentinel."""
    target = env.root/("external-"+kind)
    if kind == "existing_file":
        target.write_text("unrelated synthetic evidence must remain unchanged\n")
    elif kind == "existing_parent":
        target.mkdir()
        (target/"sentinel.txt").write_text("unrelated synthetic evidence must remain unchanged\n")
    link = env.root/("linked-"+kind)
    link.symlink_to(target, target_is_directory=kind.endswith("parent"))
    output = link/"new-directory"/"receipt.json" if kind.endswith("parent") else link
    before = target.read_bytes() if target.is_file() else None
    return output, target, before


@pytest.mark.parametrize("kind", ("dangling_file", "existing_parent", "dangling_parent"))
def test_missing_cold_receipt_rejects_linked_path_before_baseline_work(owned, kind):
    receipt, target, _ = linked_output(owned, kind)
    assert not receipt.exists()
    with pytest.raises(AssertionError, match="Linked QA path"):
        verifier.relaunch("baseline", receipt, owned.installed)
    assert owned.calls == [] and not receipt.exists()
    if kind != "existing_parent":
        assert not target.exists()
    else:
        assert {p.name for p in target.iterdir()} == {"sentinel.txt"}


@pytest.mark.parametrize("mode", ("seed", "readback"))
def test_installed_receipt_dangling_link_rejected_before_source_or_native_access(owned, monkeypatch, mode):
    receipt, target, _ = linked_output(owned, "dangling_file")
    def forbidden(*args, **kwargs):
        pytest.fail("linked receipt reached source or native inspection")
    monkeypatch.setattr(verifier, "source_pin", forbidden)
    monkeypatch.setattr(verifier, "native_session", forbidden)
    with pytest.raises(AssertionError, match="Linked QA path"):
        verifier.installed(mode, receipt, None)
    assert not target.exists()


@pytest.mark.parametrize("mode", ("source", "frozen", "seed", "readback", "baseline", "stopped", "after"))
@pytest.mark.parametrize("kind", ("existing_file", "dangling_file", "existing_parent", "dangling_parent"))
def test_cli_linked_output_is_rejected_before_mode_work_or_external_write(owned, monkeypatch, mode, kind):
    output, target, before = linked_output(owned, kind)
    def forbidden(*args, **kwargs):
        pytest.fail("linked output reached source, native, or cold protocol work")
    for name in ("source", "frozen", "installed", "relaunch"):
        monkeypatch.setattr(verifier, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["delivery-verifier", mode, "--output", str(output)])
    with pytest.raises(AssertionError, match="Linked QA path"):
        verifier.main()
    assert owned.calls == []
    if kind == "existing_file":
        assert target.read_bytes() == before
    elif kind == "existing_parent":
        assert {p.name for p in target.iterdir()} == {"sentinel.txt"}
    else:
        assert not target.exists()


@pytest.mark.parametrize("target", ("history", "script", "results", "accepted_receipt"))
def test_cold_phase_rejects_changes_to_any_accepted_saved_artifact(owned, target):
    phase(owned, "baseline")
    before = owned.receipt.read_bytes()
    if target == "history":
        write_json(owned.console/"history.json", {"history": [dict(owned.run, error="changed")]})
    elif target == "script":
        write_json(owned.console/(owned.script+".json"), dict(id=owned.script, code="changed"))
    elif target == "results":
        write_json(owned.results/"oim.json", {"changed": "previously accepted bytes"})
    else:
        write_json(owned.installed, dict(owned.accepted, source_pin_after={"changed": "receipt provenance"}))
    owned.processes = []
    with pytest.raises(AssertionError):
        phase(owned, "stopped")
    assert owned.receipt.read_bytes() == before


def test_reused_process_identity_fails_cold_after(owned):
    phase(owned, "baseline")
    original = list(owned.processes)
    owned.processes = []
    phase(owned, "stopped")
    owned.processes = original
    owned.worker = None
    with pytest.raises(AssertionError):
        phase(owned, "after")


def test_worker_after_cold_start_cannot_be_read_only_history_proof(owned):
    stop_and_prepare_cold(owned)
    owned.worker = 999
    with pytest.raises(AssertionError):
        phase(owned, "after")


@pytest.mark.parametrize("target", ("runtime", "native", "bundle_info"))
def test_accepted_binary_identity_cannot_change_before_cold_baseline(owned, target):
    path = {"runtime": owned.runtime, "native": owned.native,
            "bundle_info": owned.app/"Contents/Info.plist"}[target]
    path.write_bytes(b"different bytes than the accepted installed artifact")
    with pytest.raises(AssertionError):
        phase(owned, "baseline")
    assert owned.calls == []


@pytest.mark.parametrize("changed", ("scientific_module", "example", "delivery_verifier"))
def test_committed_source_pin_rejects_uncommitted_scientific_source(tmp_path, monkeypatch, changed):
    root = tmp_path.resolve()
    package = root/"src/openecon"
    package.mkdir(parents=True)
    (package/"__init__.py").write_text("# synthetic package\n")
    (package/"math.py").write_text("ANSWER = 42\n")
    example = root/"example.py"
    example.write_text("# synthetic example\n")
    delivery = root/"delivery_verifier.py"
    delivery.write_text("# synthetic delivery verifier\n")
    original = {str(path.relative_to(root)): path.read_bytes() for path in (package/"__init__.py", package/"math.py", example, delivery)}
    monkeypatch.setattr(verifier, "ROOT", root)
    monkeypatch.setattr(verifier, "EXAMPLE", example)
    monkeypatch.setattr(verifier, "__file__", str(delivery))
    monkeypatch.setattr(verifier, "MODULES", ("openecon.math",))
    monkeypatch.setattr(verifier, "identity", lambda: "synthetic-immutable-head")
    def committed_blob(command, **kwargs):
        assert command[:2] == ["git", "show"] and kwargs["cwd"] == root
        head, path = command[2].split(":", 1)
        assert head == "synthetic-immutable-head"
        return original[path]
    monkeypatch.setattr(verifier.subprocess, "check_output", committed_blob)
    pinned = verifier.source_pin()
    assert set(pinned["committed_file_hashes"]) == {"example", "delivery_verifier", "openecon", "openecon.math"}
    path = {"scientific_module": package/"math.py", "example": example, "delivery_verifier": delivery}[changed]
    path.write_text("# uncommitted source change\n")
    with pytest.raises(AssertionError, match="Commit the scientific source"):
        verifier.source_pin()
