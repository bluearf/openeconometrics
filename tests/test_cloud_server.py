"""The cloud identity gate must protect every workspace route, including bootstrap."""
import pytest
from fastapi.testclient import TestClient

from openecon.cloud import cloud_app
from openecon.cloud_access import CloudAccess, CloudAuthError
from openecon.server import create_app


@pytest.fixture
def cloud_client(tmp_path, monkeypatch):
    def verify(self, assertion):
        if assertion != "verified-owner-assertion":
            raise CloudAuthError("Not the owner")
        return {"email": self.owner_email, "sub": "owner-subject"}

    monkeypatch.setattr(CloudAccess, "verify", verify)
    access = CloudAccess("https://openecon.example", "/projects/123/locations/us-central1/services/openecon",
                         "owner@example.com")
    with TestClient(create_app(tmp_path, cloud_access=access), base_url=access.public_origin) as client:
        yield client


@pytest.mark.parametrize("path", ["/", "/api/session", "/api/console", "/api/datasets",
                                   "/chart-assets/renderer.js"])
def test_all_cloud_surfaces_require_verified_owner(cloud_client, path):
    response = cloud_client.get(path, headers={"x-goog-authenticated-user-email": "accounts.google.com:owner@example.com"})
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "OWNER_REQUIRED"


def authorize(client):
    client.headers["x-goog-iap-jwt-assertion"] = "verified-owner-assertion"
    response = client.get("/api/session")
    assert response.status_code == 200
    client.headers["x-openecon-token"] = response.json()["token"]
    return response.json()


def test_owner_session_keeps_csrf_and_origin_checks(cloud_client):
    session = authorize(cloud_client)
    assert session["environment"] == "cloud" and not session["persistent"]
    assert session["upload_limit_bytes"] == 24 * 1024 * 1024
    assert cloud_client.get("/api/console").status_code == 200
    assert cloud_client.get("/api/console", headers={"x-openecon-token": "invalid"}).status_code == 401
    assert cloud_client.get("/api/session", headers={"Origin": "https://openecon.example"}).status_code == 200
    for origin in ("https://evil.example", "http://openecon.example", "https://openecon.example.evil.test"):
        assert cloud_client.get("/api/session", headers={"Origin": origin}).status_code == 403
    assert cloud_client.get("/api/session", headers={"Host": "evil.example"}).status_code == 400


def test_navigation_from_signin_allowed_but_cross_site_api_denied(cloud_client):
    authorize(cloud_client)
    headers = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}
    assert cloud_client.get("/", headers=headers).status_code == 200
    assert cloud_client.get("/api/session", headers=headers).status_code == 403
    assert cloud_client.post("/api/console/reset", headers=headers).status_code == 403


def test_cloud_does_not_advertise_container_local_mcp(cloud_client):
    authorize(cloud_client)
    config = cloud_client.get("/api/config").json()
    assert config["mcp_available"] is False
    assert config["codex_command"] == config["claude_command"] == ""
    assert config["notice"]


def test_health_probe_reveals_no_workspace_state(cloud_client):
    response = cloud_client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_cloud_cannot_start_without_explicit_identity_configuration(monkeypatch):
    for key in ("OPENECON_PUBLIC_ORIGIN", "OPENECON_IAP_AUDIENCE", "OPENECON_OWNER_EMAIL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="requires"):
        cloud_app()
