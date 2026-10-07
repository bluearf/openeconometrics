"""Local execution and cloud delivery are separate, durable operations."""
from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from openecon.desktop_runtime import create_desktop_app
from openecon import desktop_sharing

PROJECT = "a" * 32
PREFIX = f"/api/desktop/projects/{PROJECT}/workspace"
ACTOR = "synthetic-owner"
REFS = [{"id": "b" * 32, "data_hash": "c" * 64}]


def connect(client):
    response = client.get(PREFIX + "/session")
    assert response.status_code == 200, response.text
    return {"X-OpenEcon-Token": response.json()["token"]}


def execute(client, headers, **kwargs):
    return client.post(PREFIX + "/desktop-console/execute", headers=headers,
                       json={"code": "6 * 7", "actor_uid": ACTOR, "input_files": REFS, **kwargs})


def queue(client, headers):
    return client.get(PREFIX + "/desktop-outbox", headers=headers).json()["items"]


def states(client, headers):
    response = client.get(PREFIX + "/desktop-sharing", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def history(client, headers):
    return client.get(PREFIX + "/console", headers=headers).json()["history"]


def fake_execution(app, monkeypatch, *, outputs=None):
    console = app.state.desktop_projects.active_app.state.console
    calls = []

    def run(code, *, timeout_seconds=None, record_metadata=None):
        calls.append(code)
        record = {"id": str(uuid4()), "code": code, "created_at": "2026-10-06T12:00:00+00:00",
                  "status": "ok", "stdout": "", "outputs": outputs or [], "variables": [],
                  "events": [], "error": None, "duration_ms": 1.0,
                  "session_generation": 1, "active_session_generation": 1}
        if record_metadata:
            record.update(record_metadata(record))
        console.workspace.append_console_history(record)
        return record

    monkeypatch.setattr(console, "execute", run)
    return calls


def test_execution_context_is_durable_and_original_envelope_survives_restart(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        result = execute(client, headers).json()
        assert result["status"] == "ok", result
        assert result["outputs"][0]["data"] == "42"
        assert result["sharing"]["state"] == "pending"
        assert result["sharing_context"] == {"actor_uid": ACTOR, "input_files": REFS, "local_only": False}
        original = queue(client, headers)[0]
        assert original["input_files"] == REFS
        assert original["record"]["actor_uid"] == ACTOR
        assert not {"sharing_context", "sharing", "local_only"} & original["record"].keys()
        assert history(client, headers)[0]["sharing_context"] == result["sharing_context"]
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert states(client, headers)["pending_count"] == 1
        assert queue(client, headers) == [original]
        # A different current inventory must not replace the run's original refs.
        client.post(PREFIX + "/datasets/example", headers=headers)
        failed = client.put(PREFIX + f"/desktop-sharing/{result['id']}", headers=headers,
                            json={"actor_uid": ACTOR, "state": "failed",
                                  "error": {"code": "OFFLINE", "message": "Connection unavailable.", "status": 503}})
        assert failed.json()["state"] == "failed"
        retry = client.post(PREFIX + f"/desktop-sharing/{result['id']}/retry", headers=headers,
                            json={"actor_uid": ACTOR})
        assert retry.json()["state"] == "pending"
        assert queue(client, headers) == [original]
        assert client.get(PREFIX + "/console", headers=headers).json()["status"]["pid"] is None
        assert len(history(client, headers)) == 1


def test_full_queue_preserves_execution_and_retry_never_recomputes(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        for number in range(20):
            response = client.put(PREFIX + "/desktop-outbox", headers=headers,
                                  json={"record": {"id": f"legacy-{number}"}, "input_files": []})
            assert response.status_code == 200
        result = execute(client, headers, code="counter = globals().get('counter', 0) + 1\ncounter").json()
        assert result["status"] == "ok" and result["outputs"][0]["data"] == "1"
        assert result["sharing"]["state"] == "failed"
        assert result["sharing"]["error"]["code"] == "OUTBOX_FULL"
        saved_id = result["id"]
        assert history(client, headers)[0]["sharing_context"]["input_files"] == REFS
        assert states(client, headers)["failed_count"] == 1
        assert len(queue(client, headers)) == 20
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert next(item for item in states(client, headers)["records"] if item["id"] == saved_id)["state"] == "failed"
        client.delete(PREFIX + "/desktop-outbox/legacy-0", headers=headers)
        retry = client.post(PREFIX + f"/desktop-sharing/{saved_id}/retry", headers=headers, json={"actor_uid": ACTOR})
        assert retry.status_code == 200, retry.text
        assert queue(client, headers)[-1]["record"]["outputs"][0]["data"] == "1"
        state = client.get(PREFIX + "/console", headers=headers).json()
        assert state["status"]["pid"] is None
        assert len(state["history"]) == 1
        assert states(client, headers)["failed_count"] == 0


def test_oversized_result_retains_context_and_durable_failure(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        calls = fake_execution(app, monkeypatch, outputs=[{"type": "text", "data": "x" * (2 * 1024 * 1024)}] * 2)
        response = execute(client, headers)
        assert response.status_code == 200, response.text[:1000]
        result = response.json()
        assert result["status"] == "ok" and result["sharing"]["state"] == "failed"
        assert result["sharing"]["error"]["code"] == "OUTBOX_LIMIT"
        assert result["sharing"]["retryable"] is False
        retry = client.post(PREFIX + f"/desktop-sharing/{result['id']}/retry", headers=headers, json={"actor_uid": ACTOR})
        assert retry.status_code == 413
        assert calls == ["6 * 7"]
        assert history(client, headers)[0]["sharing_context"]["input_files"] == REFS
        assert queue(client, headers) == []
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert states(client, headers)["records"][0]["error"]["code"] == "OUTBOX_LIMIT"


@pytest.mark.parametrize("before,after", [(True, False), (False, True), (False, "unreadable")])
def test_local_only_dependencies_are_conservative_before_and_after_run(tmp_path, monkeypatch, before, after):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        fake_execution(app, monkeypatch)
        store = app.state.desktop_projects.active_app.state.workspace
        calls = 0

        def inventory():
            nonlocal calls
            calls += 1
            current = before if calls == 1 else after
            if current == "unreadable":
                raise ValueError("Dataset record unavailable.")
            return [{"local_only": current}]

        monkeypatch.setattr(store, "list_datasets", inventory)
        result = execute(client, headers).json()
        assert result["status"] == "ok"
        assert result["local_only"] is True
        assert result["sharing_context"]["local_only"] is True
        assert result["sharing"]["state"] == "local_only"
        assert queue(client, headers) == []
        path = PREFIX + f"/desktop-sharing/{result['id']}"
        assert client.post(path + "/retry", headers=headers, json={"actor_uid": ACTOR}).status_code == 409
        assert client.put(path, headers=headers, json={"actor_uid": ACTOR, "state": "failed"}).status_code == 409


def test_actor_role_and_access_denied_guards_preserve_history(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        calls = fake_execution(app, monkeypatch)
        result = execute(client, headers).json()
        path = PREFIX + f"/desktop-sharing/{result['id']}"
        for method, suffix, body in (("post", "/retry", {"actor_uid": "another-owner"}),
                                     ("put", "", {"actor_uid": "another-owner", "state": "failed"})):
            response = getattr(client, method)(path + suffix, headers=headers, json=body)
            assert response.status_code == 403
        assert client.delete(PREFIX + f"/desktop-outbox/{result['id']}?actor_uid=another-owner", headers=headers).status_code == 403
        for role, denied in (("viewer", False), ("editor", True)):
            client.put(PREFIX + "/desktop-sync-state", headers=headers,
                       json={"cloud_version": 1, "base_code": "", "role": role, "access_denied": denied})
            assert execute(client, headers).status_code == 403
            assert client.post(path + "/retry", headers=headers, json={"actor_uid": ACTOR}).status_code == 403
        assert calls == ["6 * 7"]
        assert len(history(client, headers)) == 1
        assert len(queue(client, headers)) == 1


def test_exact_envelope_conflict_and_confirmed_ack_is_terminal(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        fake_execution(app, monkeypatch)
        result = execute(client, headers).json()
        original = queue(client, headers)[0]
        changed = {**original, "input_files": [{"id": "d" * 32, "data_hash": "e" * 64}]}
        assert client.put(PREFIX + "/desktop-outbox", headers=headers, json=changed).status_code == 409
        ack = PREFIX + f"/desktop-outbox/{result['id']}?actor_uid={ACTOR}"
        assert client.delete(ack, headers=headers).status_code == 200
        assert client.delete(ack, headers=headers).status_code == 200
        assert states(client, headers)["records"][0]["state"] == "shared"
        assert client.put(PREFIX + "/desktop-outbox", headers=headers, json=original).json()["stored"]
        assert client.put(PREFIX + "/desktop-outbox", headers=headers, json=changed).status_code == 409
        assert queue(client, headers) == []
        retry_path = PREFIX + f"/desktop-sharing/{result['id']}"
        assert client.post(retry_path + "/retry", headers=headers, json={"actor_uid": ACTOR}).json()["state"] == "shared"
        assert client.put(retry_path, headers=headers, json={"actor_uid": ACTOR, "state": "failed"}).status_code == 409
        assert client.put(PREFIX + "/desktop-sharing/unknown", headers=headers,
                          json={"actor_uid": ACTOR, "state": "failed"}).status_code == 404


def test_acknowledgement_crash_preserves_shared_terminal_before_queue_removal(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        headers = connect(client)
        fake_execution(app, monkeypatch)
        result = execute(client, headers).json()
        original_write = desktop_sharing._write_outbox

        def failure(*args):
            raise OSError("Synthetic failure after confirmed delivery.")

        monkeypatch.setattr(desktop_sharing, "_write_outbox", failure)
        ack = PREFIX + f"/desktop-outbox/{result['id']}?actor_uid={ACTOR}"
        assert client.delete(ack, headers=headers).status_code == 500
        assert states(client, headers)["records"][0]["state"] == "shared"
        assert len(queue(client, headers)) == 1
        monkeypatch.setattr(desktop_sharing, "_write_outbox", original_write)
        assert client.delete(ack, headers=headers).status_code == 200
        assert queue(client, headers) == []
        assert states(client, headers)["records"][0]["state"] == "shared"


def test_validation_auth_project_boundaries_and_json_size_limits(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        calls = fake_execution(app, monkeypatch)
        assert execute(client, {}).status_code == 401
        for invalid in ({"actor_uid": ""}, {"actor_uid": "x" * 129},
                        {"input_files": REFS * 2}, {"input_files": [{"id": "bad", "data_hash": "c" * 64}]}):
            assert execute(client, headers, **invalid).status_code == 422
        invalid_unicode = json.dumps({"code": "42", "actor_uid": "\ud800", "input_files": []}).encode("ascii")
        assert client.post(PREFIX + "/desktop-console/execute", headers={**headers, "Content-Type": "application/json"},
                           content=invalid_unicode).status_code == 422
        assert calls == []
        state_path = tmp_path / "projects" / PROJECT / "desktop-sharing.json"
        state_path.write_text(json.dumps({"records": []}) + " " * desktop_sharing.MAX_SHARING_BYTES)
        response = client.get(PREFIX + "/desktop-sharing", headers=headers)
        assert response.status_code == 422
        state_path.unlink()
        state_path.symlink_to(tmp_path / "missing")
        assert client.get(PREFIX + "/desktop-sharing", headers=headers).status_code == 422
        state_path.unlink()
        result = execute(client, headers).json()
        client.get("/api/desktop/projects/" + "f" * 32 + "/workspace/session")
        assert client.post(PREFIX + f"/desktop-sharing/{result['id']}/retry", headers=headers,
                           json={"actor_uid": ACTOR}).status_code == 409


def test_non_python_named_script_never_executes(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        calls = fake_execution(app, monkeypatch)
        created = client.post(PREFIX + "/console/scripts", headers=headers, json={"name": "paper.tex", "code": "hello"}).json()
        response = execute(client, headers, script_id=created["id"])
        assert response.status_code == 422
        assert calls == []


def test_local_computation_survives_broken_sharing_status_storage(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        calls = fake_execution(app, monkeypatch)
        console = app.state.desktop_projects.active_app.state.console
        original_execute = console.execute

        def broken_storage(*args, **kwargs):
            result = original_execute(*args, **kwargs)
            (tmp_path / "projects" / PROJECT / "desktop-sharing.json").write_text("not JSON")
            return result

        monkeypatch.setattr(console, "execute", broken_storage)
        result = execute(client, headers).json()
        assert result["status"] == "ok"
        assert result["sharing"]["state"] == "failed"
        assert result["sharing"]["error"]["code"] == "CORRUPT_LOCAL_STATE"
        assert calls == ["6 * 7"]
        assert history(client, headers)[0]["sharing_context"]["input_files"] == REFS


def test_legacy_pending_result_can_report_failure_without_original_context(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        store = app.state.desktop_projects.active_app.state.workspace
        store.append_console_history({"id": "legacy", "code": "42", "status": "ok", "outputs": [],
                                      "stdout": "", "variables": [], "error": None, "session_generation": 1})
        original = {"record": {"id": "legacy", "actor_uid": ACTOR}, "input_files": REFS}
        assert client.put(PREFIX + "/desktop-outbox", headers=headers, json=original).status_code == 200
        response = client.put(PREFIX + "/desktop-sharing/legacy", headers=headers,
                              json={"actor_uid": ACTOR, "state": "failed",
                                    "error": {"code": "NETWORK", "message": "Offline.", "status": 0}})
        assert response.status_code == 200, response.text
        assert response.json()["state"] == "failed"
        assert response.json()["error"]["status"] == 0
        assert response.json()["retryable"] is True
        assert client.post(PREFIX + "/desktop-sharing/legacy/retry", headers=headers,
                           json={"actor_uid": ACTOR}).status_code == 200
        assert queue(client, headers) == [original]


def test_historical_membership_denial_can_retry_after_access_is_restored(tmp_path, monkeypatch):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        calls = fake_execution(app, monkeypatch)
        result = execute(client, headers).json()
        original = queue(client, headers)[0]
        path = PREFIX + f"/desktop-sharing/{result['id']}"
        response = client.put(path, headers=headers,
                              json={"actor_uid": ACTOR, "state": "failed", "error": {
                                  "code": "ROLE_REQUIRED", "message": "Membership unavailable.", "status": 403}})
        assert response.json()["retryable"] is True
        client.put(PREFIX + "/desktop-sync-state", headers=headers,
                   json={"cloud_version": 1, "base_code": "", "role": "editor", "access_denied": True})
        assert client.post(path + "/retry", headers=headers, json={"actor_uid": ACTOR}).status_code == 403
        client.put(PREFIX + "/desktop-sync-state", headers=headers,
                   json={"cloud_version": 2, "base_code": "", "role": "editor", "access_denied": False})
        assert client.post(path + "/retry", headers=headers, json={"actor_uid": ACTOR}).status_code == 200
        assert queue(client, headers) == [original]
        assert calls == ["6 * 7"]


def test_expired_history_legacy_queue_retries_exact_body_after_restart(tmp_path):
    item = {"record": {"id": "expired", "actor_uid": ACTOR, "code": "never rerun", "status": "ok"},
            "input_files": REFS}
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.put(PREFIX + "/desktop-outbox", headers=headers, json=item).status_code == 200
        assert client.put(PREFIX + "/desktop-sharing/expired", headers=headers,
                          json={"actor_uid": ACTOR, "state": "failed", "error": {
                              "code": "OFFLINE", "message": "Offline.", "status": 0}}).status_code == 200
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        client.post(PREFIX + "/datasets/example", headers=headers)
        assert states(client, headers)["records"][0]["retryable"] is True
        retry = client.post(PREFIX + "/desktop-sharing/expired/retry", headers=headers, json={"actor_uid": ACTOR})
        assert retry.status_code == 200, retry.text
        assert queue(client, headers) == [item]
        assert history(client, headers) == []
        assert client.get(PREFIX + "/console", headers=headers).json()["status"]["pid"] is None
        assert client.post(PREFIX + "/desktop-sharing/expired/retry", headers=headers,
                           json={"actor_uid": "wrong-owner"}).status_code == 403


def test_metadata_retention_is_bounded_by_actual_encoded_bytes(tmp_path):
    from openecon.workspace import Workspace
    archive = desktop_sharing.DesktopSharing(tmp_path, Workspace(tmp_path))
    records = {f"old-{index}": {"id": f"old-{index}", "actor_uid": "\U00010000" * 128,
                                "state": "failed", "error": {
                                    "code": "NETWORK", "message": "\U00010000" * 512, "status": 0}}
               for index in range(1000)}
    archive._save_metadata(records)
    assert archive.metadata_path.stat().st_size <= desktop_sharing.MAX_SHARING_BYTES
    restored = archive._metadata()
    assert 500 < len(restored) < 1000
    assert "old-999" in restored and "old-0" not in restored


def test_contextless_history_without_pending_body_has_no_false_sharing_failure(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        app.state.desktop_projects.active_app.state.workspace.append_console_history({
            "id": "old-untracked-run", "code": "42", "status": "ok", "outputs": []})
        snapshot = states(client, headers)
        assert snapshot["failed_count"] == snapshot["pending_count"] == 0
        assert snapshot["records"][0]["state"] == "local_only"
        assert snapshot["records"][0]["retryable"] is False


def test_live_snapshot_is_at_most_history_plus_queue_count(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        archive = tmp_path / "projects" / PROJECT / "console" / "history.json"
        archive.write_text(json.dumps({"history": [{"id": f"old-{index}"} for index in range(500)]}))
        for index in range(20):
            assert client.put(PREFIX + "/desktop-outbox", headers=headers,
                              json={"record": {"id": f"queued-{index}", "actor_uid": ACTOR}, "input_files": []}).status_code == 200
        snapshot = states(client, headers)
        assert len(snapshot["records"]) == 520
        assert snapshot["pending_count"] == 20
        assert snapshot["failed_count"] == 0


def test_network_archive_text_is_captured_before_history_and_stable_after_formatter_upgrade(tmp_path, monkeypatch):
    network = {"type": "plot", "data": {"kind": "network", "title": "Synthetic collaboration",
               "artifact": {"url": "/console/plots/synthetic", "id": "synthetic"}},
               "latex": "saved publication LaTeX", "latex_style": "publication-v1"}
    model = {"type": "text", "data": "42"}
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        calls = fake_execution(app, monkeypatch, outputs=[model, network])
        result = execute(client, headers).json()
        context = result["sharing_context"]
        assert context["network_summaries"] == [{"index": 1, "summary": desktop_sharing.legacy_network_summary_v1(network["data"])}]
        assert set(context) == {"actor_uid", "input_files", "local_only", "network_summaries"}
        saved = history(client, headers)[0]
        assert saved["outputs"][1]["type"] == "plot"
        assert saved["outputs"][1]["data"]["artifact"] == network["data"]["artifact"]
        original = queue(client, headers)[0]
        assert original["record"]["outputs"][1] == {**network, "type": "text", "data": context["network_summaries"][0]["summary"]}
        assert original["record"]["outputs"][0] == model
        assert not {"sharing_context", "sharing", "local_only"} & original["record"].keys()
        path = PREFIX + f"/desktop-sharing/{result['id']}"
        client.put(path, headers=headers, json={"actor_uid": ACTOR, "state": "failed"})
        monkeypatch.setattr(desktop_sharing, "legacy_network_summary_v1", lambda data: "An upgraded UI formatter with different text")
        assert client.post(path + "/retry", headers=headers, json={"actor_uid": ACTOR}).status_code == 200
        assert queue(client, headers) == [original]
        assert calls == ["6 * 7"]
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.post(path + "/retry", headers=headers, json={"actor_uid": ACTOR}).status_code == 200
        assert queue(client, headers) == [original]
        assert len(history(client, headers)) == 1
        assert client.get(PREFIX + "/console", headers=headers).json()["status"]["pid"] is None


def test_network_summary_keeps_event_indices_and_legacy_raw_queue_identity(tmp_path):
    from openecon.workspace import Workspace
    project_path = tmp_path / "projects" / PROJECT
    workspace = Workspace(project_path)
    record = {"id": "old-network", "actor_uid": ACTOR, "code": "never recompute", "status": "ok", "stdout": "hello",
              "outputs": [{"type": "plot", "data": {"kind": "network", "title": "Legacy network", "artifact": {"id": "legacy"}}}],
              "events": [{"type": "stdout", "text": "hello"}, {"type": "output", "index": 0}],
              "sharing_context": {"actor_uid": ACTOR, "input_files": REFS, "local_only": False}}
    workspace.append_console_history(record)
    archive = desktop_sharing.DesktopSharing(project_path, workspace)
    prepared = archive.envelope(record)
    assert prepared["record"]["events"] == record["events"]
    assert prepared["record"]["outputs"][0]["data"] == desktop_sharing.legacy_network_summary_v1(record["outputs"][0]["data"])
    raw = {"record": {key: value for key, value in record.items() if key != "sharing_context"}, "input_files": REFS}
    desktop_sharing._write_outbox(project_path / "desktop-outbox.json", {"items": [raw]})
    archive._set(record["id"], ACTOR, "failed", digest=archive._digest(raw))
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        response = client.post(PREFIX + "/desktop-sharing/old-network/retry", headers=headers, json={"actor_uid": ACTOR})
        assert response.status_code == 200, response.text
        assert queue(client, headers) == [raw]
        assert client.get(PREFIX + "/console", headers=headers).json()["status"]["pid"] is None
        assert client.delete(PREFIX + "/desktop-outbox/old-network?actor_uid=" + ACTOR, headers=headers).status_code == 200
        assert client.put(PREFIX + "/desktop-outbox", headers=headers, json=raw).status_code == 200
        assert queue(client, headers) == []


@pytest.mark.parametrize("summaries", [[{"index": 0, "summary": "wrong output"}],
                                       [{"index": 1, "summary": "summary"}] * 2,
                                       [{"index": 20, "summary": "summary"}],
                                       [{"index": 1, "summary": "x" * 513}]])
def test_saved_network_summary_validation_cannot_replace_model_outputs(summaries):
    record = {"actor_uid": ACTOR, "outputs": [{"type": "text", "data": "original model"},
               {"type": "plot", "data": {"kind": "network", "artifact": {}}}],
              "sharing_context": {"input_files": [], "network_summaries": summaries}}
    with pytest.raises(desktop_sharing.DesktopError, match="sharing context"):
        desktop_sharing.DesktopSharing.envelope(record)
