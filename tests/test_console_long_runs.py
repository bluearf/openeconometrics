"""Large local analyses remain cancellable without changing cloud deadlines."""
from concurrent.futures import ThreadPoolExecutor
import time

from fastapi.testclient import TestClient
import pytest

from openecon.console import ConsoleError, ConsoleSession
from openecon.server import create_app
from openecon.workspace import Workspace


def test_local_http_accepts_long_runtime_without_changing_output_contract(tmp_path):
    app = create_app(tmp_path)
    assert app.state.console.default_timeout_seconds is None
    assert app.state.console.max_timeout_seconds is None
    with TestClient(app) as client:
        token = client.get("/api/session").json()["token"]
        response = client.post("/api/console/execute", headers={"X-OpenEcon-Token": token},
                               json={"code": "print('before')\n42", "timeout_seconds": 3600})
        assert response.status_code == 200
        record = response.json()
        assert record["status"] == "ok"
        assert record["outputs"][0]["data"] == "42"
        assert record["events"] == [{"type": "stdout", "text": "before\n"},
                                    {"type": "output", "index": 0}]


def test_default_cloud_console_still_rejects_unbounded_explicit_deadline(tmp_path):
    session = ConsoleSession(Workspace(tmp_path))
    try:
        assert session.default_timeout_seconds == 60
        assert session.max_timeout_seconds == 120
        with pytest.raises(ConsoleError) as error:
            session.execute("42", timeout_seconds=121)
        assert error.value.code == "INVALID_TIMEOUT"
        assert session.status()["pid"] is None
    finally:
        session.close()


def test_unlimited_local_execution_can_be_interrupted_and_restarted(tmp_path):
    session = ConsoleSession(Workspace(tmp_path), default_timeout_seconds=None,
                             max_timeout_seconds=None)
    try:
        assert session.execute("x = 41")["status"] == "ok"
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(session.execute, "while True: pass")
            deadline = time.monotonic() + 5
            while not session.status()["running"] and time.monotonic() < deadline:
                time.sleep(.01)
            assert session.status()["running"]
            assert session.interrupt()["status"] == "interrupted"
            assert pending.result(timeout=5)["status"] == "interrupted"
        assert session.status()["running"] is False
        assert session.execute("42")["outputs"][0]["data"] == "42"
    finally:
        session.close()


@pytest.mark.parametrize("invalid", [True, float("inf"), float("nan"), -1, 0, "120"])
def test_unlimited_local_mode_keeps_invalid_deadlines_rejected(tmp_path, invalid):
    session = ConsoleSession(Workspace(tmp_path), default_timeout_seconds=None,
                             max_timeout_seconds=None)
    try:
        with pytest.raises(ConsoleError) as error:
            session.execute("42", timeout_seconds=invalid)
        assert error.value.code == "INVALID_TIMEOUT"
    finally:
        session.close()
