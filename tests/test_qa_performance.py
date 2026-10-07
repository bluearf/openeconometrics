"""The timing recorder is opt-in, protected, bounded and content-free."""

import json

from fastapi.testclient import TestClient

from openecon.desktop_runtime import create_desktop_app


def test_normal_desktop_has_no_qa_recorder(tmp_path):
    with TestClient(create_desktop_app(tmp_path)) as client:
        token = client.get("/api/desktop/session").json()["token"]
        response = client.post(
            "/api/desktop/qa-performance/event",
            headers={"X-OpenEcon-Token": token},
            json={"phase": "typing", "duration_ms": 20, "epoch_ms": 1000},
        )
        assert response.status_code in (401, 404)
        assert not any(
            getattr(route, "path", "").startswith("/api/desktop/qa-performance")
            for route in client.app.routes
        )
    assert not (tmp_path / "qa-performance.jsonl").exists()


def test_explicit_qa_only_accepts_times_labels_and_local_token(tmp_path):
    with TestClient(create_desktop_app(tmp_path, qa_metrics=True)) as client:
        payload = {"phase": "typing", "duration_ms": 20, "epoch_ms": 1000}
        assert client.post("/api/desktop/qa-performance/event", json=payload).status_code == 401
        token = client.get("/api/desktop/session").json()["token"]
        headers = {"X-OpenEcon-Token": token}
        assert (
            client.post(
                "/api/desktop/qa-performance/event",
                headers=headers,
                json={**payload, "code": "private"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/desktop/qa-performance/event", headers=headers, json=payload
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/desktop/qa-performance/control",
                headers=headers,
                json={"scenario": "latency_250", "delay_ms": 250},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/desktop/qa-performance/event", headers=headers, json=payload
            ).status_code
            == 200
        )
    lines = [json.loads(s) for s in (tmp_path / "qa-performance.jsonl").read_text().splitlines()]
    assert lines[0] == {**payload, "scenario": "local_off", "delay_ms": 0}
    assert lines[1] == {**payload, "scenario": "latency_250", "delay_ms": 250}
