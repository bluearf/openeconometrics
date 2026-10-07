"""Project package changes and user code share one serialized local session."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from openecon.console import ConsoleError, ConsoleSession
from openecon.desktop_runtime import create_desktop_app
from openecon.server import create_app
from openecon.workspace import Workspace


def test_package_install_requires_project_desktop_and_local_token(tmp_path):
    with TestClient(create_app(tmp_path / "plain")) as client:
        token = client.get("/api/session").json()["token"]
        assert client.post("/api/environment/install", headers={"X-OpenEcon-Token": token},
                           json={"name": "humanize"}).status_code == 404
    with TestClient(create_desktop_app(tmp_path / "desktop")) as client:
        prefix = "/api/desktop/projects/" + "a" * 32 + "/workspace"
        token = client.get(prefix + "/session").json()["token"]
        assert client.get(prefix + "/environment").status_code == 401
        headers = {"X-OpenEcon-Token": token}
        assert client.get(prefix + "/environment", headers=headers).status_code == 200
        assert client.post(prefix + "/environment/install", headers=headers,
                           json={"name": "torch", "version": "1.0"}).status_code == 422
        assert client.post(prefix + "/environment/install", headers=headers,
                           json={"name": "https://evil.example/pkg.whl"}).status_code == 422
        assert client.post(prefix + "/environment/install", headers=headers,
                           json={"name": "humanize", "index": "https://evil.example"}).status_code == 422


def test_running_install_blocks_execution_and_running_code_blocks_changes(tmp_path):
    console = ConsoleSession(Workspace(tmp_path))
    try:
        console.packages = SimpleNamespace(snapshot=lambda: {"job": {"state": "running"}})
        record = console.execute("42")
        assert record["error"]["type"] == "ENVIRONMENT_BUSY"
        assert console.status()["pid"] is None
        console._run_lock.acquire()
        try:
            called = []
            with pytest.raises(ConsoleError, match="Stop the running code"):
                console.change_environment(lambda: called.append(True))
            assert called == []
        finally:
            console._run_lock.release()
    finally:
        console.close()


def test_package_overlay_is_importable_only_in_its_project_worker(tmp_path):
    overlay = tmp_path / "extra"
    overlay.mkdir()
    (overlay / "openecon_test_extra.py").write_text("answer = 73\n")
    first = ConsoleSession(Workspace(tmp_path / "one"))
    second = ConsoleSession(Workspace(tmp_path / "two"))
    first.packages = SimpleNamespace(snapshot=lambda: {"job": None}, active_path=lambda: overlay)
    try:
        result = first.execute("import openecon_test_extra\nopenecon_test_extra.answer")
        assert result["status"] == "ok", result
        assert result["outputs"][0]["data"] == "73"
        unavailable = second.execute("import openecon_test_extra")
        assert unavailable["error"]["type"] == "ModuleNotFoundError"
        generation = first.status()["session_generation"]
        first.change_environment(lambda: {"accepted": True})
        assert first.status()["pid"] is None
        assert first.status()["session_generation"] == generation + 1
        assert first.snapshot()["variables"] == []
        assert len(first.snapshot()["history"]) == 1
    finally:
        first.close()
        second.close()
