"""Actual worker/installer processes for synchronous project script installs."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time

import pytest

from openecon.console import ConsoleSession
from openecon import project_packages
from openecon.project_packages import ProjectPackages
from openecon.workspace import Workspace

INSTALLER = r'''
import json, pathlib, sys, time
action = pathlib.Path(sys.argv[1])
data = json.loads(action.read_text())
stage = action.parent
if any(row['name'] == 'slow' for row in data['requirements']):
    print('Waiting in script installer', flush=True)
    time.sleep(30)
if any(row['name'] == 'failure' for row in data['requirements']):
    (stage / 'result.json').write_text(json.dumps({'state':'error','code':'NO_WHEEL','message':'No compatible wheel'}))
    raise SystemExit(2)
overlay = stage / 'site-packages'
overlay.mkdir()
locked=[]
for row in data['requirements']:
    name, version = row['name'], row['version'] or '1.2.3'
    package = overlay / name.replace('-', '_')
    package.mkdir()
    (package / '__init__.py').write_text('VALUE = '+repr(version)+'\n')
    (package / 'late.py').write_text('VALUE = '+repr(version)+'\n')
    metadata = overlay / (name.replace('-', '_')+'-'+version+'.dist-info')
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
    locked.append({'name':name,'version':version})
manifest={'schema':1,'python':data['python'],'core':data['core'],'requirements':locked,'locked':locked}
if data.get('installer', 'pip') != 'pip' or data.get('specifications') is not None:
    manifest.update(schema=2, installer=data.get('installer','pip'),
                    specifications=data.get('specifications') or [row['name'] + ('==' + row['version'] if row['version'] else '') for row in data['requirements']])
(stage / 'result.json').write_text(json.dumps({'state':'complete','manifest':manifest}))
'''


@pytest.fixture
def script_console(tmp_path, monkeypatch):
    usage = project_packages.shutil.disk_usage
    monkeypatch.setattr(project_packages.shutil, "disk_usage",
                        lambda path: usage(path)._replace(free=4 * 1024**3))
    monkeypatch.setattr(project_packages, "installer_command",
                        lambda action: [sys.executable, "-c", INSTALLER, str(action)])
    manager = ProjectPackages(tmp_path / "workspace")
    session = ConsoleSession(Workspace(tmp_path / "workspace"), default_timeout_seconds=None,
                             max_timeout_seconds=None)
    session.packages = manager
    yield session, manager
    session.close()
    manager.close()


def test_script_install_import_and_output_order_preserve_variables(script_console):
    session, manager = script_console
    result = session.execute("saved = 37\nprint('before')\n"
                             "oe.install('demo')\nimport demo\nprint(demo.VALUE)\ndisplay(saved)")
    assert result["status"] == "ok", result
    assert result["stdout"] == "before\nPackages ready: demo==1.2.3\n1.2.3\n"
    assert result["events"] == [{"type": "stdout", "text": result["stdout"]},
                                {"type": "output", "index": 0}]
    assert result["outputs"][0]["data"] == "37"
    assert manager.snapshot()["requirements"] == [{"name": "demo", "version": "1.2.3"}]
    assert session.execute("saved + 1")["outputs"][0]["data"] == "38"


def test_percent_pip_batch_and_repeated_noop(script_console):
    session, manager = script_console
    code = "%pip install demo another==2.0\nimport demo, another\nprint(another.VALUE)"
    first = session.execute(code)
    assert first["status"] == "ok", first
    before, pid = manager.active_path(), session.status()["pid"]
    job = manager.snapshot()["job"]["id"]
    repeated = session.execute(code)
    assert repeated["status"] == "ok", repeated
    assert manager.active_path() == before
    assert manager.snapshot()["job"]["id"] == job
    assert session.status()["pid"] == pid
    assert session.workspace.console_history()[-1]["code"] == code


def test_new_package_preserves_imported_old_package_and_late_submodule(script_console):
    session, manager = script_console
    first = session.execute("oe.install('demo')\nimport demo\nsaved = 19")
    assert first["status"] == "ok", first
    previous = manager.active_path()
    second = session.execute("oe.install('another')\nimport demo.late, another\n"
                              "print(demo.late.VALUE)\nsaved")
    assert second["status"] == "ok", second
    assert second["outputs"][0]["data"] == "19"
    assert previous.is_dir()
    active = manager.active_path()
    session.reset()
    assert not previous.exists()
    assert active.is_dir()
    assert session.execute("import demo, another\nprint(demo.VALUE)")["status"] == "ok"


def test_imported_upgrade_rejected_before_promotion(script_console):
    session, manager = script_console
    assert session.execute("oe.install('demo')\nimport demo\nsaved = 9")["status"] == "ok"
    previous = manager.active_path()
    result = session.execute("oe.install('demo==2.0')\nprint('must not run')")
    assert result["status"] == "error", result
    assert "Restart Python" in result["error"]["message"]
    assert "must not run" not in result["stdout"]
    assert manager.active_path() == previous
    assert session.execute("print(demo.VALUE)\nsaved")["outputs"][0]["data"] == "9"
    session.reset()
    updated = session.execute("oe.install('demo==2.0')\nimport demo\ndemo.VALUE")
    assert updated["status"] == "ok", updated
    assert updated["outputs"][0]["data"] == "2.0"


def test_install_failure_preserves_active_environment_and_session(script_console):
    session, manager = script_console
    assert session.execute("oe.install('demo')\nimport demo\nsaved = 8")["status"] == "ok"
    previous = manager.active_path()
    result = session.execute("print('before')\noe.install('failure')\nprint('after')")
    assert result["status"] == "error", result
    assert result["stdout"] == "before\n"
    assert "No compatible wheel" in result["error"]["message"]
    assert manager.active_path() == previous
    assert session.execute("saved")["outputs"][0]["data"] == "8"


def _wait_running(manager):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        job = manager.snapshot().get("job") or {}
        if "Waiting" in job.get("log", ""):
            return
        time.sleep(.02)
    raise AssertionError("Test installer did not start")


def _wait_terminal(manager):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if manager.snapshot()["job"]["state"] != "running":
            return
        time.sleep(.02)
    raise AssertionError("Test installer did not stop")


def test_stop_cancels_owned_script_installer(script_console):
    session, manager = script_console
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(session.execute, "oe.install('slow')\nprint('after')")
        _wait_running(manager)
        session.interrupt()
        result = pending.result(timeout=10)
    _wait_terminal(manager)
    assert result["status"] == "interrupted", result
    assert manager.snapshot()["installed"] == []
    assert not list(manager.root.glob("stage-*"))
    assert session.execute("1 + 1")["outputs"][0]["data"] == "2"


def test_timeout_cancels_owned_script_installer(script_console):
    session, manager = script_console
    result = session.execute("oe.install('slow')", timeout_seconds=1)
    _wait_terminal(manager)
    assert result["status"] == "timeout", result
    assert manager.snapshot()["installed"] == []
    assert not list(manager.root.glob("stage-*"))
    assert session.execute("1 + 1")["status"] == "ok"


def test_worker_crash_stops_install_without_execution_timeout(script_console):
    session, manager = script_console
    result = session.execute("import os, threading, time\n"
                             "def crash():\n    time.sleep(.3)\n    os._exit(7)\n"
                             "threading.Thread(target=crash).start()\noe.install('slow')")
    _wait_terminal(manager)
    assert result["error"]["type"] == "WORKER_EXITED", result
    assert manager.snapshot()["installed"] == []
    assert not list(manager.root.glob("stage-*"))
    assert session.execute("1 + 1")["status"] == "ok"


def test_invalid_code_does_not_install_any_package(script_console):
    session, manager = script_console
    result = session.execute("%pip install demo\nfor:")
    assert result["error"]["type"] == "SyntaxError", result
    assert manager.snapshot()["job"] is None


def test_non_desktop_session_rejects_install(tmp_path):
    session = ConsoleSession(Workspace(tmp_path))
    try:
        result = session.execute("oe.install('demo')")
        assert result["status"] == "error"
        assert "desktop project" in result["error"]["message"]
        assert session.execute("2 + 3")["outputs"][0]["data"] == "5"
    finally:
        session.close()


def test_package_request_protocol_rejects_foreign_execution_and_extras():
    from openecon.console_worker import WorkerProtocolError, validate_install_request
    message = {"kind": "package_install", "id": "run", "request_id": "package",
               "requirements": [{"name": "demo", "version": None}], "loaded_versions": {}}
    assert validate_install_request(message, "run")["requirements"] == message["requirements"]
    for value in ({**message, "id": "other"}, {**message, "op": "shell"},
                  {**message, "requirements": [{"name": "--target", "version": None}]},
                  {**message, "loaded_versions": {"x": "https://evil"}}):
        with pytest.raises(WorkerProtocolError):
            validate_install_request(value, "run")
    assert json.loads(json.dumps(message)) == message
