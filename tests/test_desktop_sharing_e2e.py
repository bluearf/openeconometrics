"""Real local Torch output, durable retry and another member's cloud API readback.

The cloud storage/identity transports are test fixtures. This does not establish
native GUI operation or deployment to the live cloud service.
"""
import json

from fastapi.testclient import TestClient

from openecon.desktop_runtime import create_desktop_app
from test_desktop_cloud import api as api, headers


def test_local_ols_survives_offline_restart_lost_ack_and_member_readback(tmp_path, api):
    project = api.pid
    local = f"/api/desktop/projects/{project}/workspace"
    cloud = f"/api/projects/{project}/workspace"
    code = """import openecon as oe
run_count = globals().get('run_count', 0) + 1
data = oe.example()
model = oe.ols(data=data, y='wage', x=['education', 'experience'], covariance='HC3')
display(model)
print(run_count)
"""
    with TestClient(create_desktop_app(tmp_path)) as client:
        token = {"X-OpenEcon-Token": client.get(local + "/session").json()["token"]}
        response = client.post(local + "/desktop-console/execute", headers=token,
                               json={"code": code, "actor_uid": "editor", "input_files": []})
        assert response.status_code == 200, response.text
        computed = response.json()
        assert computed["status"] == "ok" and computed["stdout"] == "1\n", computed
        output = computed["outputs"][0]
        assert output["type"] == "model" and output["data"]["nobs"] == 480
        assert output["data"]["provenance"]["backend"] == "openecon.torch"
        assert "\\toprule" in output["latex"]
        original = client.get(local + "/desktop-outbox", headers=token).json()["items"][0]
        assert "sharing_context" not in original["record"]
        encoded = json.dumps(original, sort_keys=True)
        assert client.get(local + "/desktop-sharing", headers=token).json()["pending_count"] == 1
        # Offline: no cloud request has been issued when the app exits.

    with TestClient(create_desktop_app(tmp_path)) as client:
        token = {"X-OpenEcon-Token": client.get(local + "/session").json()["token"]}
        assert client.get(local + "/console", headers=token).json()["status"]["pid"] is None
        saved = client.get(local + "/desktop-outbox", headers=token).json()["items"][0]
        assert json.dumps(saved, sort_keys=True) == encoded
        first = api.client.post(cloud + "/desktop/results", headers=headers("editor"), json=saved)
        assert first.status_code == 201, first.text
        # Simulate loss of the local acknowledgement, leaving the exact queue item.
        assert client.get(local + "/desktop-sharing", headers=token).json()["pending_count"] == 1

    with TestClient(create_desktop_app(tmp_path)) as client:
        token = {"X-OpenEcon-Token": client.get(local + "/session").json()["token"]}
        retried = client.post(local + f"/desktop-sharing/{computed['id']}/retry", headers=token,
                              json={"actor_uid": "editor"})
        assert retried.status_code == 200
        item = client.get(local + "/desktop-outbox", headers=token).json()["items"][0]
        assert json.dumps(item, sort_keys=True) == encoded
        duplicate = api.client.post(cloud + "/desktop/results", headers=headers("editor"), json=item)
        assert duplicate.status_code == 201 and duplicate.json() == first.json()
        ack = client.delete(local + f"/desktop-outbox/{computed['id']}?actor_uid=editor", headers=token)
        assert ack.status_code == 200
        state = client.get(local + "/desktop-sharing", headers=token).json()
        assert state["pending_count"] == 0 and state["records"][0]["state"] == "shared"
        history = client.get(local + "/console", headers=token).json()
        assert history["status"]["pid"] is None and len(history["history"]) == 1
        assert history["history"][0]["stdout"] == "1\n"

    member = api.client.get(cloud + "/console", headers=headers("viewer"))
    assert member.status_code == 200
    shared = member.json()["history"]
    assert len(shared) == 1 and shared[0]["stdout"] == computed["stdout"]
    assert shared[0]["events"] == computed["events"]
    assert shared[0]["outputs"][0]["latex"] == output["latex"]
    assert shared[0]["outputs"][0]["data"]["coefficients"] == output["data"]["coefficients"]
    assert api.client.post(cloud + "/desktop/results", headers=headers("viewer"), json=item).status_code == 403
    assert api.client.get(cloud + "/console", headers=headers("outsider")).status_code == 404
