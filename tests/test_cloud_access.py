"""Real ES256 verification with offline Google key retrieval fixtures."""
from datetime import datetime, timezone
from types import SimpleNamespace

from cryptography.hazmat.primitives.asymmetric import ec
import jwt
import pytest

from openecon.cloud_access import CloudAccess, CloudAuthError, IAP_ISSUER, IAP_JWKS_URL

ORIGIN = "https://openecon-123456789.us-central1.run.app"
AUDIENCE = "/projects/123456789/locations/us-central1/services/openecon"
OWNER = "owner@example.com"


@pytest.fixture(scope="module")
def signing_key():
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture
def access(monkeypatch, signing_key):
    result = CloudAccess(ORIGIN, AUDIENCE, OWNER)
    monkeypatch.setattr(result._jwks, "get_signing_key_from_jwt",
                        lambda value: SimpleNamespace(key=signing_key.public_key()))
    return result


def token(key, *, updates=None, omit=(), headers=None):
    now = int(datetime.now(timezone.utc).timestamp())
    payload = {"iss": IAP_ISSUER, "aud": AUDIENCE, "sub": "accounts.google.com:12345",
               "email": OWNER, "iat": now, "exp": now + 600}
    payload.update(updates or {})
    for name in omit:
        payload.pop(name)
    return jwt.encode(payload, key, algorithm="ES256", headers={"kid": "test-key", **(headers or {})})


def test_real_signed_owner_assertion_is_accepted(access, signing_key):
    claims = access.verify(token(signing_key))
    assert claims["email"] == OWNER
    assert claims["sub"] == "accounts.google.com:12345"


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
        CloudAccess(origin, AUDIENCE, OWNER)


@pytest.mark.parametrize("audience", [
    "", "https://openecon.run.app", "/projects/project-id/locations/us-central1/services/openecon",
    "/projects/123/locations/us-central1/services/*", AUDIENCE + "/", AUDIENCE + "\n",
    "/projects/0/locations/us-central1/services/openecon", None,
])
def test_audience_must_identify_one_cloud_run_service(audience):
    with pytest.raises(ValueError, match="audience"):
        CloudAccess(ORIGIN, audience, OWNER)


@pytest.mark.parametrize("owner", ["", None, " owner@example.com", "owner@example.com ",
    "one@example.com,two@example.com", "one@example.com;two@example.com", "*", "group:team@example.com",
    "owner@example..com", "owner@-example.com", "owner@example.com\n"])
def test_owner_must_be_a_single_explicit_email(owner):
    with pytest.raises(ValueError, match="owner_email"):
        CloudAccess(ORIGIN, AUDIENCE, owner)


@pytest.mark.parametrize("updates", [
    {"email": "other@example.com"}, {"email": "Owner@example.com"},
    {"email": [OWNER]}, {"aud": "another-service"}, {"aud": [AUDIENCE]},
    {"iss": "https://accounts.google.com"}, {"sub": ""}, {"sub": " "}, {"sub": 12345},
    {"sub": " padded "}, {"exp": True}, {"iat": False}, {"exp": "9999999999"},
    {"iat": "0"}, {"exp": float("inf")}, {"iat": float("nan")},
])
def test_signed_but_invalid_claims_are_denied(access, signing_key, updates):
    with pytest.raises(CloudAuthError):
        access.verify(token(signing_key, updates=updates))


@pytest.mark.parametrize("missing", ["exp", "iat", "sub", "email", "iss", "aud"])
def test_required_claims_cannot_be_omitted(access, signing_key, missing):
    with pytest.raises(CloudAuthError):
        access.verify(token(signing_key, omit=[missing]))


def test_time_windows_and_lifetime_are_enforced(access, signing_key):
    now = int(datetime.now(timezone.utc).timestamp())
    for dates in [
        {"iat": now - 700, "exp": now - 40},
        {"iat": now + 60, "exp": now + 660},
        {"iat": now, "exp": now + 661},
        {"iat": now, "exp": now},
        {"iat": now, "exp": now - 1},
    ]:
        with pytest.raises(CloudAuthError):
            access.verify(token(signing_key, updates=dates))
    for dates in [
        {"iat": now - 610, "exp": now - 10},
        {"iat": now + 10, "exp": now + 610},
        {"iat": now, "exp": now + 660},
    ]:
        assert access.verify(token(signing_key, updates=dates))["email"] == OWNER


def test_signature_from_another_key_is_denied(access):
    another = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(CloudAuthError):
        access.verify(token(another))


@pytest.mark.parametrize("assertion", [None, b"token", "", "not-a-jwt", "a.b.c", "x" * 16385])
def test_malformed_assertions_are_denied(access, assertion):
    with pytest.raises(CloudAuthError):
        access.verify(assertion)


def test_algorithm_confusion_and_missing_key_id_are_denied_before_fetch(access, signing_key, monkeypatch):
    calls = []
    monkeypatch.setattr(access._jwks, "get_signing_key_from_jwt", lambda value: calls.append(value))
    now = int(datetime.now(timezone.utc).timestamp())
    payload = {"iss": IAP_ISSUER, "aud": AUDIENCE, "email": OWNER,
               "sub": "12345", "iat": now, "exp": now + 600}
    samples = [
        jwt.encode(payload, "a-long-test-hmac-key-with-at-least-32-bytes", algorithm="HS256", headers={"kid": "test-key"}),
        jwt.encode(payload, None, algorithm="none", headers={"kid": "test-key"}),
        jwt.encode(payload, signing_key, algorithm="ES256"),
        token(signing_key, headers={"kid": ""}),
    ]
    for sample in samples:
        with pytest.raises(CloudAuthError):
            access.verify(sample)
    assert calls == []


def test_key_retrieval_failure_fails_closed_without_exposing_assertion(access, signing_key, monkeypatch):
    def unavailable(value):
        raise jwt.PyJWKClientConnectionError("upstream unavailable")
    monkeypatch.setattr(access._jwks, "get_signing_key_from_jwt", unavailable)
    assertion = token(signing_key)
    with pytest.raises(CloudAuthError) as error:
        access.verify(assertion)
    assert assertion not in str(error.value)
    assert "upstream" not in str(error.value)


def test_key_resolver_has_fixed_url_finite_cache_and_timeout(monkeypatch):
    captured = {}
    def client(uri, **kwargs):
        captured.update(uri=uri, **kwargs)
        return object()
    monkeypatch.setattr(jwt, "PyJWKClient", client)
    CloudAccess(ORIGIN, AUDIENCE, OWNER)
    assert captured == {"uri": IAP_JWKS_URL, "cache_jwk_set": True, "lifespan": 300,
                        "cache_keys": False, "timeout": 5}
