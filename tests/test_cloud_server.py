"""The cloud origin gate stays strict; the local server has no cloud execution mode."""
import inspect

import pytest
from fastapi.testclient import TestClient

from openecon.server import create_app
from openecon.team_server import source_commit, validate_public_origin


@pytest.mark.parametrize("origin", [
    "http://app.example.com", "https://app.example.com/", "https://app.example.com/path",
    "https://app.example.com:443", "https://user@app.example.com", "https://app.example.com?q=1",
    "https://app.example.com#fragment", "HTTPS://app.example.com", "https://APP.example.com",
    " https://app.example.com", "https://app.example.com\n", "https://app.example.com.",
    "https://app..example.com", "https://*.example.com", "https://app_bad.example.com",
    "https://127.0.0.1", "https://[::1]", "https://localhost", "https://café.example.com",
    "https://-app.example.com", "https://app-.example.com", "https://[malformed", None,
])
def test_origin_must_be_one_canonical_https_dns_origin(origin):
    with pytest.raises(ValueError, match="origin"):
        validate_public_origin(origin)


def test_canonical_origin_is_accepted():
    validate_public_origin("https://openecon-291739190496.us-central1.run.app")


@pytest.mark.parametrize("value,expected", [
    ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
    ("a" * 64, "a" * 64),
    (None, None), ("", None), ("unknown", None), ("0123456", None),
    ("0123456789ABCDEF0123456789ABCDEF01234567", None), ("a" * 41, None),
    ("0123456789abcdef0123456789abcdef01234567\n", None), (12345, None),
])
def test_source_commit_accepts_only_full_lowercase_commit_ids(value, expected):
    assert source_commit(value) == expected


def test_local_server_has_no_cloud_access_mode(tmp_path):
    assert "cloud_access" not in inspect.signature(create_app).parameters
    with TestClient(create_app(tmp_path)) as client:
        session = client.get("/api/session").json()
        assert session["environment"] == "local" and session["persistent"] is True
        assert client.get("/api/auth/config").json() == {"mode": "local"}
        # IAP assertions are not an identity source for the loopback server.
        assert client.get("/api/console", headers={"x-goog-iap-jwt-assertion": "x"}).status_code == 401
        assert client.get("/healthz").status_code != 200
