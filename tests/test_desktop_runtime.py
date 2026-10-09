"""Desktop project persistence and local-process trust boundaries."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading

from fastapi.testclient import TestClient
import pytest

from openecon.console import ConsoleSession
from openecon.console_worker import (
    MAX_WORKER_BYTES, WorkerProtocolError, receive_json, send_json, validate_result,
)
from openecon.desktop_runtime import create_desktop_app, worker_environment
from openecon.workspace import Workspace

A = "a" * 32
B = "b" * 32


def prefix(project=A):
    return f"/api/desktop/projects/{project}/workspace"


def connect(client, project=A):
    response = client.get(prefix(project) + "/session")
    assert response.status_code == 200, response.text
    return {"X-OpenEcon-Token": response.json()["token"]}


def test_opening_and_saving_never_executes_a_draft(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/auth/config").json() == {"mode": "desktop"}
        assert client.get("/api/session").json()["environment"] == "local"
        headers = connect(client)
        assert app.state.desktop_projects.active_app.state.console._process is None
        sentinel = tmp_path / "did-not-run"
        code = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')"
        assert client.put(prefix() + "/console/script", headers=headers,
                          json={"name": "analysis.py", "code": code}).status_code == 200
        assert client.get(prefix() + "/console/script", headers=headers).json()["code"] == code
        assert client.get(prefix() + "/console", headers=headers).json()["history"] == []
        assert not sentinel.exists()
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(prefix() + "/console/script", headers=headers).json()["code"] == code
        assert not sentinel.exists()


def test_persistent_torch_session_switch_and_project_tokens(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers_a = connect(client)
        first = client.post(prefix() + "/console/execute", headers=headers_a,
                            json={"code": "tensor = torch.tensor([1., 2.], dtype=torch.float64)\ntensor.sum().item()"}).json()
        assert first["status"] == "ok", first
        assert first["outputs"][0]["data"] == "3.0"
        console_a = app.state.desktop_projects.active_app.state.console
        pid = console_a.snapshot()["status"]["pid"]
        second = client.post(prefix() + "/console/execute", headers=headers_a,
                             json={"code": "tensor.add_(2)\ntensor.sum().item()"}).json()
        assert second["outputs"][0]["data"] == "7.0"
        assert console_a.snapshot()["status"]["pid"] == pid
        headers_b = connect(client, B)
        assert console_a._process is None
        assert client.get(prefix() + "/console", headers=headers_a).status_code == 409
        assert client.post(prefix(B) + "/console/execute", headers=headers_a,
                           json={"code": "42"}).status_code == 401
        new = client.post(prefix(B) + "/console/execute", headers=headers_b,
                          json={"code": "'tensor' in globals()"}).json()
        assert new["outputs"][0]["data"] == "False"
        new_headers_a = connect(client)
        assert new_headers_a != headers_a
        state = client.get(prefix() + "/console", headers=new_headers_a).json()
        assert len(state["history"]) == 2
        assert state["status"]["pid"] is None
        assert state["variables"] == []
        assert (tmp_path / "projects" / A / "console" / "history.json").is_file()
    assert app.state.desktop_projects.active_app is None
    assert app.state.desktop_projects.closed


def test_sync_state_persists_and_does_not_overwrite_pending_code(tmp_path):
    state = {"cloud_version": 7, "base_code": "original = 1", "role": "editor"}
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(prefix() + "/desktop-sync-state", headers=headers).json() is None
        assert client.put(prefix() + "/desktop-sync-state", headers=headers, json=state).json() == state
        assert client.put(prefix() + "/console/script", headers=headers,
                          json={"code": "pending = 2"}).status_code == 200
        assert client.put(prefix() + "/desktop-sync-state", headers=headers,
                          json={**state, "cloud_version": "8"}).status_code == 422
        assert client.put(prefix() + "/desktop-sync-state", headers=headers,
                          json={**state, "role": "admin"}).status_code == 422
        assert client.get(prefix() + "/desktop-sync-state").status_code == 401
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(prefix() + "/desktop-sync-state", headers=headers).json() == state
        assert client.get(prefix() + "/console/script", headers=headers).json()["code"] == "pending = 2"


def test_result_outbox_is_durable_idempotent_and_project_scoped(tmp_path):
    item = {"record": {"id": "58f5ef78-3c66-4856-a7f9-584c6f90d08d", "code": "raise RuntimeError('never run')",
                       "status": "ok", "outputs": []},
            "input_files": [{"id": "c" * 32, "data_hash": "d" * 64}]}
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        assert client.put(prefix() + "/desktop-outbox", json=item).status_code == 401
        assert client.get(prefix() + "/desktop-outbox", headers=headers).json() == {"items": []}
        saved = client.put(prefix() + "/desktop-outbox", headers=headers, json=item)
        assert saved.json() == {"stored": True, "id": item["record"]["id"]}
        assert client.put(prefix() + "/desktop-outbox", headers=headers, json=item).status_code == 200
        conflict = {**item, "record": {**item["record"], "status": "error"}}
        assert client.put(prefix() + "/desktop-outbox", headers=headers, json=conflict).status_code == 409
        assert client.get(prefix() + "/desktop-outbox", headers=headers).json() == {"items": [item]}
        assert app.state.desktop_projects.active_app.state.console._process is None
        headers_b = connect(client, B)
        assert client.get(prefix(B) + "/desktop-outbox", headers=headers_b).json() == {"items": []}
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(prefix() + "/desktop-outbox", headers=headers).json() == {"items": [item]}
        path = prefix() + "/desktop-outbox/" + item["record"]["id"]
        assert client.delete(path, headers=headers).json()["deleted"]
        assert client.delete(path, headers=headers).json()["deleted"]
        assert client.get(prefix() + "/desktop-outbox", headers=headers).json() == {"items": []}


def test_result_outbox_rejects_invalid_json_identifiers_references_and_limits(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        for invalid in (
            {"record": {"id": "../bad"}, "input_files": []},
            {"record": {"id": "safe"}, "input_files": [{"id": A, "data_hash": "bad"}]},
            {"record": {"id": "safe"}, "input_files": [{"id": A, "data_hash": "d" * 64}] * 2},
        ):
            assert client.put(prefix() + "/desktop-outbox", headers=headers, json=invalid).status_code == 422
        assert client.put(prefix() + "/desktop-outbox", headers={**headers, "Content-Type": "application/json"},
                          content=b'{"record":{"id":"safe","number":NaN},"input_files":[]}').status_code == 422
        huge = {"record": {"id": "huge", "data": "x" * (3 * 1024 * 1024)}, "input_files": []}
        assert client.put(prefix() + "/desktop-outbox", headers=headers, json=huge).status_code == 413
        for index in range(20):
            item = {"record": {"id": f"run-{index}"}, "input_files": []}
            assert client.put(prefix() + "/desktop-outbox", headers=headers, json=item).status_code == 200
        assert client.put(prefix() + "/desktop-outbox", headers=headers,
                          json={"record": {"id": "one-too-many"}, "input_files": []}).status_code == 409
        assert len(client.get(prefix() + "/desktop-outbox", headers=headers).json()["items"]) == 20


def test_result_outbox_total_size_is_bounded_and_failure_preserves_pending(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        for index in range(4):
            item = {"record": {"id": f"run-{index}", "data": "x" * (2 * 1024 * 1024)}, "input_files": []}
            response = client.put(prefix() + "/desktop-outbox", headers=headers, json=item)
            assert response.status_code == (200 if index < 3 else 409)
        assert len(client.get(prefix() + "/desktop-outbox", headers=headers).json()["items"]) == 3
        assert (tmp_path / "projects" / A / "desktop-outbox.json").stat().st_size <= 8 * 1024 * 1024


def test_cached_import_validates_checksum_and_reuses_snapshot(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        source = tmp_path / "projects" / A / "observations.csv"
        source.write_bytes(b"x,y\n1,2\n2,4\n3,7\n")
        body = {"name": source.name, "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "cloud_id": "c" * 32}
        wrong = client.post(prefix() + "/datasets/import-cached", headers=headers,
                            json={**body, "expected_sha256": "0" * 64})
        assert wrong.status_code == 422
        assert wrong.json()["detail"]["code"] == "DATA_INTEGRITY"
        imported = client.post(prefix() + "/datasets/import-cached", headers=headers, json=body)
        assert imported.status_code == 200, imported.text
        assert imported.json()["row_count"] == 3
        assert imported.json()["python_path"] == source.name
        assert imported.json()["cloud_id"] == body["cloud_id"]
        cached = client.get(prefix() + "/desktop-cached-files", headers=headers).json()
        expected_cache = {"files": [{"cloud_id": body["cloud_id"], "sha256": body["expected_sha256"],
                                     "name": body["name"], "dataset_id": imported.json()["id"]}],
                          "local_only": []}
        assert cached == expected_cache
        assert imported.json()["data_hash"] != cached["files"][0]["sha256"]
        again = client.post(prefix() + "/datasets/import-cached", headers=headers, json=body)
        assert again.json()["id"] == imported.json()["id"]
        assert len(client.get(prefix() + "/datasets", headers=headers).json()["datasets"]) == 1
        execution = client.post(prefix() + "/console/execute", headers=headers,
                                json={"code": "len(oe.read('observations.csv'))"}).json()
        assert execution["outputs"][0]["data"] == "3"
        for name in ("../observations.csv", ".hidden.csv", "a:b.csv", "ğ" * 100 + ".csv"):
            assert client.post(prefix() + "/datasets/import-cached", headers=headers,
                               json={**body, "name": name}).status_code == 422
        invalid_unicode = json.dumps({**body, "name": "\ud800.csv"}).encode("ascii")
        assert client.post(prefix() + "/datasets/import-cached",
                           headers={**headers, "Content-Type": "application/json"},
                           content=invalid_unicode).status_code == 422
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(prefix() + "/desktop-cached-files", headers=headers).json() == expected_cache


@pytest.mark.parametrize("entry", [
    {"name": "../outside.csv", "dataset_id": "58f5ef78-3c66-4856-a7f9-584c6f90d08d", "sha256": "d" * 64},
    {"name": "safe.csv", "dataset_id": "../outside", "sha256": "d" * 64},
    {"name": "safe.csv", "dataset_id": "58f5ef78-3c66-4856-a7f9-584c6f90d08d", "sha256": "bad"},
])
def test_cached_file_metadata_rejects_corrupt_paths_ids_and_hashes(tmp_path, entry):
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        (tmp_path / "projects" / A / "desktop-cache-index.json").write_text(json.dumps({"c" * 32: entry}))
        response = client.get(prefix() + "/desktop-cached-files", headers=headers)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "CORRUPT_LOCAL_STATE"


def test_origin_host_cloud_credentials_traversal_and_symlinks(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        for headers in ({"Origin": "https://elsewhere.example"}, {"Sec-Fetch-Site": "cross-site"}):
            assert client.get(prefix() + "/session", headers=headers).status_code == 403
        assert client.get(prefix() + "/session", headers={"Host": "evil.example"}).status_code == 400
        assert client.get(prefix() + "/session", headers={"Authorization": "Bearer cloud-token"}).status_code == 400
        for invalid in ("not-a-project", "A" * 32, "%2e%2e", A + "%2f.."):
            assert client.get(prefix(invalid) + "/session").status_code == 404
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "projects" / A).symlink_to(outside, target_is_directory=True)
        assert client.get(prefix() + "/session").status_code == 422
        assert not (outside / "console").exists()
        (tmp_path / "projects" / A).unlink()
        headers = connect(client)
        external = outside / "secret.json"
        external.write_text('{"history":[{"code":"secret"}]}')
        (tmp_path / "projects" / A / "console" / "history.json").symlink_to(external)
        assert client.get(prefix() + "/console", headers=headers).status_code == 422


def test_desktop_worker_does_not_inherit_cloud_or_auth_environment(tmp_path, monkeypatch):
    secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/private/credential.json", "AWS_SECRET_ACCESS_KEY": "secret",
               "OPENECON_FIREBASE_CONFIG": "secret", "OPENECON_ACCESS_TOKEN": "secret",
               "HTTPS_PROXY": "http://private-proxy", "PYTHONSTARTUP": "/private/code.py"}
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    assert not set(secrets) & worker_environment().keys()
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        execution = client.post(prefix() + "/console/execute", headers=headers,
                                json={"code": f"import os\nall(name not in os.environ for name in {list(secrets)!r})"}).json()
        assert execution["status"] == "ok", execution
        assert execution["outputs"][0]["data"] == "True"


class _Exploit:
    def __init__(self, sentinel):
        self.sentinel = sentinel

    def __reduce__(self):
        return exec, (f"from pathlib import Path; Path({self.sentinel!r}).write_text('executed in parent')",)


def _pickle_worker(connection, workspace):
    send_json(connection, {"kind": "ready", "ready": True, "process_group": None})
    receive_json(connection)
    connection.send_bytes(pickle.dumps(_Exploit(str(Path(workspace) / "pickle-executed"))))
    connection.close()


def _invalid_result_worker(connection, workspace):
    send_json(connection, {"kind": "ready", "ready": True, "process_group": None})
    command = receive_json(connection)
    send_json(connection, {"kind": "result", "id": command["id"], "result": {
        "status": [], "stdout": "", "outputs": [], "variables": [], "error": None}})
    connection.close()


def _oversized_worker(connection, workspace):
    send_json(connection, {"kind": "ready", "ready": True, "process_group": None})
    receive_json(connection)
    try:
        connection.send_bytes(b" " * (MAX_WORKER_BYTES + 1))
    except OSError:
        pass
    connection.close()


@pytest.mark.parametrize("target", [_pickle_worker, _invalid_result_worker, _oversized_worker])
def test_malformed_worker_messages_stop_session_without_unpickling(tmp_path, monkeypatch, target):
    import openecon.console_worker as worker
    monkeypatch.setattr(worker, "worker_main", target)
    session = ConsoleSession(Workspace(tmp_path))
    try:
        result = session.execute("42")
        assert result["status"] == "error"
        assert result["state_reset"]
        assert session.snapshot()["status"]["pid"] is None
        assert session.snapshot()["variables"] == []
        assert not (tmp_path / "pickle-executed").exists()
    finally:
        session.close()


@pytest.mark.parametrize("payload", [b"[]", b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e1000}',
                                    b'not json', b'{"a":"\\ud800"}', b'{"\\ud800":0}'])
def test_json_pipe_rejects_invalid_payloads(payload):
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        sender.send_bytes(payload)
        with pytest.raises(WorkerProtocolError):
            receive_json(receiver)
    finally:
        receiver.close()
        sender.close()


def test_worker_result_identifier_and_schema_are_validated():
    result = {"status": "ok", "stdout": "", "outputs": [], "variables": [], "error": None}
    with pytest.raises(WorkerProtocolError):
        validate_result({"kind": "result", "id": "stale", "result": result}, "current")
    with pytest.raises(WorkerProtocolError):
        validate_result({"kind": "result", "id": "current", "result": {**result, "outputs": [{}]}}, "current")


def test_worker_result_checks_timeline_and_accepts_legacy_without_it():
    result = {"status": "ok", "stdout": "last\n", "variables": [], "error": None,
              "outputs": [{"type": "text", "data": "first"}]}
    message = {"kind": "result", "id": "current", "result": result}
    assert validate_result(message, "current") == result
    result['events'] = [{'type': 'output', 'index': 0}, {'type': 'stdout', 'text': 'last\n'}]
    assert validate_result(message, 'current')['events'] == result['events']
    result['events'][0]['index'] = True
    with pytest.raises(WorkerProtocolError, match='timeline'):
        validate_result(message, 'current')


def test_explicit_close_cleans_running_worker_and_keeps_history(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        client.post(prefix() + "/console/execute", headers=headers, json={"code": "saved = 10"})
        console = app.state.desktop_projects.active_app.state.console
        assert console.snapshot()["status"]["pid"] is not None
        assert client.post("/api/desktop/close").status_code == 401
        root_token = client.get("/api/desktop/session").json()["token"]
        assert root_token != headers["X-OpenEcon-Token"]
        assert client.post("/api/desktop/close", headers={"X-OpenEcon-Token": root_token}).json() == {"status": "closed"}
        assert console._process is None
        headers = connect(client)
        assert len(client.get(prefix() + "/console", headers=headers).json()["history"]) == 1


@pytest.mark.parametrize("termination", ["shutdown", "eof"])
def test_entry_announces_ephemeral_port_and_stdin_shutdown(tmp_path, termination):
    import httpx
    process = subprocess.Popen([sys.executable, "-m", "openecon.desktop_entry", "--port", "0",
                                "--data-root", str(tmp_path)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    worker_handle = None
    kernel32 = None
    try:
        announcements = queue.Queue(maxsize=1)

        def read_ready():
            try:
                record = process.stdout.readline(16385)
                if len(record) > 16384 or not record.endswith(b"\n"):
                    raise RuntimeError("Desktop runtime readiness is absent or oversized")
                announcements.put(json.loads(record))
            except BaseException as error:
                announcements.put(error)

        threading.Thread(target=read_ready, daemon=True).start()
        try:
            descriptor = announcements.get(timeout=30)
        except queue.Empty as error:
            raise AssertionError("Desktop runtime did not announce startup") from error
        if isinstance(descriptor, BaseException):
            raise descriptor
        assert descriptor["type"] == "ready"
        assert descriptor["port"] > 0
        assert descriptor["url"] == f"http://127.0.0.1:{descriptor['port']}"
        assert httpx.get(descriptor["url"] + "/api/auth/config", trust_env=False).json() == {"mode": "desktop"}
        with httpx.Client(base_url=descriptor["url"], trust_env=False, timeout=30) as client:
            headers = connect(client)
            result = client.post(prefix() + "/console/execute", headers=headers, json={"code": "6 * 7"}).json()
            assert result["status"] == "ok", result
            assert result["outputs"][0]["data"] == "42"
            worker_pid = client.get(prefix() + "/console", headers=headers).json()["status"]["pid"]
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            # Hold this exact owned worker before shutdown, so PID reuse cannot
            # turn the exit assertion into a query or signal of another process.
            worker_handle = kernel32.OpenProcess(0x00100000, False, worker_pid)
            if not worker_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            assert kernel32.WaitForSingleObject(worker_handle, 0) == 258
        if termination == "shutdown":
            process.stdin.write(b'{"type":"shutdown"}\n')
            process.stdin.flush()
        else:
            process.stdin.close()
        process.wait(timeout=10)
        assert process.returncode == 0
        assert process.stdout.read() == b""
        if os.name == "nt":
            assert kernel32.WaitForSingleObject(worker_handle, 5000) == 0
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(worker_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if worker_handle:
            kernel32.CloseHandle(worker_handle)
