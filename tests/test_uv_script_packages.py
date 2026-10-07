"""Installer syntax, conditional file reads and persistent project declarations."""
import ast
from copy import deepcopy

import pytest

from openecon.console_worker import validate_install_request, WorkerProtocolError
from openecon.package_requirements import parse_specifications
from openecon.project_packages import PackageError, validate_manifest
from openecon.script_packages import _use_installer, install, rewrite_install_commands
from openecon.team_store import TeamError, validate_environment_manifest
from test_script_package_console import script_console as script_console
from test_script_package_manager import manager_fixture as manager_fixture, wait_job


@pytest.mark.parametrize("prefix", ["%uv pip install", "!uv pip install", "%uv add", "!uv add", "%pip install", "!pip install"])
def test_common_install_syntax_preserves_python_execution_order(prefix):
    calls = []
    code = f"saved = 37\n{prefix} 'demo[fast]>=1,<2'\nvalue = saved + 1\n"
    def callback(rows, **options):
        assert namespace["saved"] == 37 and "value" not in namespace
        calls.append((rows, options))
        return {"installed": [{"name": "demo", "version": "1.2.3"}]}
    namespace = {}
    with _use_installer(callback):
        exec(compile(rewrite_install_commands(code), "<syntax>", "exec"), namespace)
    assert calls[0][0] == [{"name": "demo", "version": None}]
    assert calls[0][1]["specifications"] == ["demo[fast]<2,>=1"]
    assert calls[0][1].get("installer", "pip") == ("uv" if "uv" in prefix else "pip")
    assert namespace["value"] == 38


@pytest.mark.parametrize("quote", ['"""', "'''", 'f"""', 'r"""'])
def test_uv_commands_inside_strings_are_not_installs(quote):
    ending = quote[-3:]
    code = f"text = {quote}first\n%uv pip install demo\n!uv add demo\nlast{ending}\n"
    assert rewrite_install_commands(code) == code
    ast.parse(code)


def test_requirements_file_includes_ranges_extras_and_comments_at_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nested.txt").write_text("demo[fast]>=1,<2 # note\n")
    (tmp_path / "requirements.txt").write_text("# comment\n-r nested.txt\nanother==2.0\n")
    calls = []
    def callback(rows, **options):
        calls.append((rows, options))
        return {"installed": [{"name": "demo", "version": "1.2.3"}, {"name": "another", "version": "2.0"}]}
    # A file made by a preceding Python line is read only when install executes.
    code = "if False:\n    %uv pip install -r missing.txt\n%uv pip install -r requirements.txt\n"
    with _use_installer(callback):
        exec(rewrite_install_commands(code), {})
    assert len(calls) == 1
    assert calls[0][1]["specifications"] == ["demo[fast]<2,>=1", "another==2.0"]


def test_cyclic_and_oversized_requirements_do_not_invoke_installer(tmp_path):
    path = tmp_path / "cycle.txt"
    path.write_text("-r cycle.txt\n")
    calls = []
    with _use_installer(calls.append), pytest.raises(PackageError):
        install(requirements_file=path, installer="uv")
    path.write_bytes(b"x" * (256 * 1024 + 1))
    with _use_installer(calls.append), pytest.raises(PackageError):
        install(requirements_file=path, installer="uv")
    assert not calls


def test_inactive_marker_does_not_install_or_modify_project(capsys):
    calls = []
    with _use_installer(calls.append):
        install('demo; python_version < "1"', installer="uv")
    assert calls == []
    assert "No package requirements apply" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["%uv run example.py", "%uv sync", "%uv install demo", "%uv pip install --system demo", "%uv pip install --python other demo", "%uv add ./source", "%uv pip install 'demo @ https://example.test/a.whl'"])
def test_other_environment_or_source_commands_are_rejected(command):
    with pytest.raises(PackageError):
        rewrite_install_commands(command)


def test_uv_install_retains_state_and_extras_when_adding_plain_packages(script_console):
    session, manager = script_console
    first = session.execute("saved = 37\n%uv add 'demo[fast]>=1,<2'\nimport demo\nprint(demo.VALUE)")
    assert first["status"] == "ok", first
    before = manager.snapshot()
    assert before["manifest"]["schema"] == 2
    assert before["manifest"]["specifications"] == ["demo[fast]<2,>=1"]
    second = session.execute("oe.install('another')\nimport demo.late, another\nsaved")
    assert second["status"] == "ok", second
    assert second["outputs"][0]["data"] == "37"
    assert "demo[fast]<2,>=1" in manager.snapshot()["manifest"]["specifications"]
    assert session.status()["session_generation"] == first["session_generation"]


def test_uv_matching_range_avoids_new_job_or_worker_restart(script_console):
    session, manager = script_console
    first = session.execute("%uv pip install 'demo>=1,<2'\nimport demo\nsaved=41")
    assert first["status"] == "ok", first
    job, path, pid = manager.snapshot()["job"]["id"], manager.active_path(), session.status()["pid"]
    repeated = session.execute("%uv pip install 'demo>=1,<2'\nsaved")
    assert repeated["status"] == "ok", repeated
    assert manager.snapshot()["job"]["id"] == job
    assert manager.active_path() == path and session.status()["pid"] == pid


def test_uv_failed_install_preserves_previous_manifest(script_console):
    session, manager = script_console
    assert session.execute("%uv add demo\nimport demo\nsaved=19")["status"] == "ok"
    before = manager.snapshot()["manifest"]
    failed = session.execute("%uv pip install failure")
    assert failed["status"] == "error"
    assert manager.snapshot()["manifest"] == before
    assert session.execute("saved")["outputs"][0]["data"] == "19"


def modern_manifest():
    return {"schema": 2, "python": "3.13", "core": {"numpy": "2.3.3"},
            "requirements": [{"name": "demo", "version": "1.2.3"}],
            "locked": [{"name": "demo", "version": "1.2.3"}, {"name": "helper", "version": "2.0"}],
            "installer": "uv", "specifications": ['demo[fast]>=1,<2; platform_system == "Darwin"']}


def test_shared_specs_remain_inert_on_a_different_server_platform():
    value = modern_manifest()
    normalized = validate_manifest(value)
    assert validate_environment_manifest(value) == normalized
    assert normalized["specifications"] == ['demo[fast]<2,>=1; platform_system == "Darwin"']


@pytest.mark.parametrize("change", [
    {"schema": 2.0}, {"installer": "shell"}, {"installer": {}},
    {"specifications": "demo"}, {"specifications": ["demo>=2"]},
    {"specifications": ["demo @ https://example.test/a.whl"]},
    {"specifications": ["--index-url=https://example.test"]},
    {"specifications": ["demo", "Demo"]}, {"surprise": True},
])
def test_invalid_shared_specs_are_rejected_in_both_validators(change):
    value = deepcopy(modern_manifest()) | change
    with pytest.raises(PackageError):
        validate_manifest(value)
    with pytest.raises(TeamError) as error:
        validate_environment_manifest(value)
    assert error.value.code == "INVALID_ENVIRONMENT"


def test_worker_rejects_mismatched_specs_and_unrecognized_installer_options():
    rows, specs = parse_specifications(["demo>=1,<2"])
    message = {"kind": "package_install", "id": "execution", "request_id": "request", "requirements": rows,
               "loaded_versions": {}, "installer": "uv", "specifications": specs}
    assert validate_install_request(message, "execution")["installer"] == "uv"
    for change in [{"specifications": ["other"]}, {"installer": []}, {"upgrade": "yes"}, {"target": "/tmp"}]:
        with pytest.raises(WorkerProtocolError):
            validate_install_request(message | change, "execution")


def test_application_and_installer_upgrade_retains_existing_project_libraries(manager_fixture, monkeypatch):
    from openecon import project_packages
    manager = manager_fixture.create()
    manager.start_install("demo")
    before = wait_job(manager)
    path, generation = manager.active_path(), manager._generation
    manager.close()
    core = {**before["manifest"]["core"], "openecon": "0.3.7a1", "uv": "0.9.26"}
    monkeypatch.setattr(project_packages, "protected_versions", lambda: core)
    reopened = manager_fixture.create()
    try:
        assert reopened.active_path() == path and reopened._generation == generation
        assert reopened.snapshot()["manifest"]["core"] == core
        assert reopened.snapshot()["manifest"]["locked"] == before["manifest"]["locked"]
        assert len(manager_fixture.actions) == 1
    finally:
        reopened.close()


def test_matching_uv_core_request_does_not_persist_a_missing_overlay(manager_fixture):
    manager = manager_fixture.create()
    rows, specs = parse_specifications(["numpy>=2,<3"])
    snapshot = manager.start_install_many(rows, installer="uv", specifications=specs)
    assert snapshot["job"] is None and snapshot["manifest"]["schema"] == 1
    assert not (manager.root / "active.json").exists()
    manager.close()
    assert manager_fixture.create().snapshot()["requirements"] == []


@pytest.mark.parametrize("numeric_change", [True, False])
def test_incompatible_core_upgrade_keeps_original_pointer(manager_fixture, monkeypatch, numeric_change):
    from openecon import project_packages
    manager = manager_fixture.create()
    manager.start_install("demo")
    before = wait_job(manager)
    pointer = manager.root / "active.json"
    original = pointer.read_bytes()
    core = {**before["manifest"]["core"], "openecon": "0.3.7a1", "uv": "0.9.26"}
    if numeric_change:
        core["numpy"] = "9.0"
    else:
        metadata = next(manager.active_path().glob("demo-*.dist-info/METADATA"))
        old_version = before["manifest"]["core"]["openecon"]
        metadata.write_text(metadata.read_text() + f"Requires-Dist: openecon=={old_version}\n")
    manager.close()
    monkeypatch.setattr(project_packages, "protected_versions", lambda: core)
    with pytest.raises(PackageError) as error:
        manager_fixture.create()
    assert error.value.code == "INCOMPATIBLE_CORE"
    assert pointer.read_bytes() == original
