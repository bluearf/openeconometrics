"""Durable named-file cloud bases and optimistic desktop-only backfill."""
from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from openecon.desktop_runtime import create_desktop_app

PROJECT = "a" * 32
FILE = "b" * 32
OTHER = "c" * 32
PREFIX = f"/api/desktop/projects/{PROJECT}/workspace"


def connect(client):
    response = client.get(PREFIX + "/session")
    assert response.status_code == 200
    return {"X-OpenEcon-Token": response.json()["token"]}


def state():
    return {"cloud_version": 10, "base_code": "legacy = 1", "role": "editor", "scripts": {
        FILE: {"name": "empty.py", "cloud_version": None, "local_version": 0, "base_code": ""},
        OTHER: {"name": "other.py", "cloud_version": 52, "local_version": 3, "base_code": "base = 2", "conflict": True},
    }, "script_catalog": [{"id": "analysis", "name": "analysis.py", "version": 10},
                            {"id": OTHER, "name": "other.py", "version": 52}], "access_denied": False}


def test_per_file_sync_bases_survive_restart_without_changing_the_legacy_shape(tmp_path):
    legacy = {"cloud_version": 1, "base_code": "", "role": "owner"}
    saved = state()
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.put(PREFIX + "/desktop-sync-state", headers=headers, json=legacy).json() == legacy
        assert client.get(PREFIX + "/desktop-sync-state", headers=headers).json() == legacy
        assert client.put(PREFIX + "/desktop-sync-state", headers=headers, json=saved).json() == saved
        assert client.put(PREFIX + f"/desktop-scripts/{FILE}", headers=headers,
                          json={"name": "empty.py", "code": "", "version": 0}).json() == {
                              "id": FILE, "name": "empty.py", "code": "", "version": 0}
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(PREFIX + "/desktop-sync-state", headers=headers).json() == saved
        assert client.get(PREFIX + f"/console/scripts/{FILE}", headers=headers).json()["code"] == ""
        assert client.app.state.desktop_projects.active_app.state.console._process is None


def test_backfill_requires_local_token_and_local_version_and_never_overwrites_another_name(tmp_path):
    app = create_desktop_app(tmp_path)
    with TestClient(app) as client:
        headers = connect(client)
        path = PREFIX + f"/desktop-scripts/{FILE}"
        initial = {"name": "first.py", "code": "raise RuntimeError('not executed')", "version": 0}
        assert client.put(path, json=initial).status_code == 401
        assert client.put(path, headers=headers, json=initial).json()["version"] == 0
        assert client.put(path, headers=headers, json=initial).json()["version"] == 0
        updated = client.put(path, headers=headers, json={**initial, "code": "changed = 1"})
        assert updated.json()["version"] == 1
        assert client.put(path, headers=headers, json={**initial, "code": "wrong cloud version", "version": 40}).status_code == 409
        assert client.put(path, headers=headers, json=initial).status_code == 409
        assert client.put(PREFIX + f"/desktop-scripts/{OTHER}", headers=headers,
                          json={"name": "first.py", "code": "", "version": 0}).status_code == 409
        assert client.get(PREFIX + f"/console/scripts/{FILE}", headers=headers).json()["code"] == "changed = 1"
        assert client.get(PREFIX + f"/console/scripts/{OTHER}", headers=headers).status_code == 404
        assert app.state.desktop_projects.active_app.state.console._process is None


@pytest.mark.parametrize("invalid", [
    {"scripts": {"analysis": {"name": "a.py", "cloud_version": 0, "local_version": 0, "base_code": ""}}},
    {"scripts": {FILE: {"name": "../bad.py", "cloud_version": 0, "local_version": 0, "base_code": ""}}},
    {"scripts": {FILE: {"name": "a.py", "cloud_version": True, "local_version": 0, "base_code": ""}}},
    {"scripts": {FILE: {"name": "a.py", "cloud_version": 0, "local_version": "0", "base_code": ""}}},
    {"script_catalog": [{"id": FILE, "name": "a.py", "version": 0, "code": "must not be metadata"}]},
    {"script_catalog": [{"id": "analysis", "name": "wrong.py", "version": 0}]},
    {"script_catalog": [{"id": "analysis", "name": "analysis.py", "version": 0}] * 2},
    {"access_denied": "false"},
])
def test_invalid_per_file_state_is_rejected_without_replacing_previous_bases(tmp_path, invalid):
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.put(PREFIX + "/desktop-sync-state", headers=headers, json=state()).status_code == 200
        assert client.put(PREFIX + "/desktop-sync-state", headers=headers, json={**state(), **invalid}).status_code == 422
        assert client.get(PREFIX + "/desktop-sync-state", headers=headers).json() == state()


def test_valid_multi_file_state_can_exceed_the_old_one_megabyte_read_limit(tmp_path):
    saved = {"cloud_version": 0, "base_code": "", "role": "editor", "scripts": {
        f"{index:032x}": {"name": f"file_{index}.py", "cloud_version": index, "local_version": 0, "base_code": "x" * 60000}
        for index in range(20)
    }}
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.put(PREFIX + "/desktop-sync-state", headers=headers, json=saved).status_code == 200
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(PREFIX + "/desktop-sync-state", headers=headers).json() == saved


@pytest.mark.parametrize("extension", ["md", "tex"])
def test_document_cache_and_sync_catalog_survive_restart_without_execution(tmp_path, extension):
    saved = state()
    saved["scripts"][FILE]["name"] = f"notes.{extension}"
    saved["scripts"][FILE]["base_code"] = "source text"
    saved["script_catalog"].append({"id": FILE, "name": f"notes.{extension}", "version": 0})
    document = {"id": FILE, "name": f"notes.{extension}", "code": "source text", "version": 0}
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.put(PREFIX + "/desktop-sync-state", headers=headers, json=saved).status_code == 200
        assert client.put(PREFIX + f"/desktop-scripts/{FILE}", headers=headers,
                          json={key: value for key, value in document.items() if key != "id"}).json() == document
        assert client.put(PREFIX + "/desktop-scripts/analysis", headers=headers,
                          json={"name": f"analysis.{extension}", "code": "", "version": 0}).status_code == 409
    with TestClient(create_desktop_app(tmp_path)) as client:
        headers = connect(client)
        assert client.get(PREFIX + "/desktop-sync-state", headers=headers).json() == saved
        assert client.get(PREFIX + f"/console/scripts/{FILE}", headers=headers).json() == document
        rejection = client.post(PREFIX + "/console/execute", headers=headers,
                                json={"code": "1 + 2", "script_id": FILE})
        assert rejection.status_code == 422
        assert rejection.json()["detail"]["code"] == "DOCUMENT_NOT_EXECUTABLE"
        assert client.app.state.desktop_projects.active_app.state.console._process is None
