"""Real signature checks plus the broker's pre-body authentication boundary."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient
import httpx
import jwt
import pytest

from openecon import team_sandbox_auth as auth
from openecon import team_sandbox_broker as broker

AUDIENCE = "https://openecon-sandbox-291739190496.us-central1.run.app"
EMAIL = "openecon-control@openecon-workbench.iam.gserviceaccount.com"
SUBJECT = "108062605432490492564"


@pytest.fixture(scope="module")
def signing():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "offline-test")])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(hours=1))
            .not_valid_after(now + timedelta(days=1)).sign(key, hashes.SHA256()))
    return key, cert.public_bytes(serialization.Encoding.PEM).decode("ascii")


def token(signing, *, claims=None, remove=(), headers=None, key=None):
    now = int(datetime.now(timezone.utc).timestamp())
    body = {"iss": "https://accounts.google.com", "aud": AUDIENCE, "sub": SUBJECT,
            "email": EMAIL, "email_verified": True, "iat": now, "exp": now + 3600,
            **(claims or {})}
    for name in remove:
        body.pop(name, None)
    return jwt.encode(body, key or signing[0], algorithm="RS256",
                      headers={"kid": "google-offline-key", **(headers or {})})


def verifier():
    return auth.BrokerAuth(AUDIENCE, EMAIL, SUBJECT)


def network(monkeypatch, signing, handler=None):
    configurations, requests = [], []
    original = httpx.AsyncClient

    def receive(request):
        requests.append(request)
        if handler:
            return handler(request)
        return httpx.Response(200, json={"google-offline-key": signing[1]},
                              headers={"Cache-Control": "public, max-age=3600"})

    def client(**kwargs):
        configurations.append(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(receive))

    monkeypatch.setattr(auth.httpx, "AsyncClient", client)
    return configurations, requests


def test_real_google_format_signature_and_cached_public_certs(monkeypatch, signing):
    configurations, requests = network(monkeypatch, signing)
    gate = verifier()

    async def scenario():
        await gate.verify(["Bearer " + token(signing)])
        await gate.verify(["Bearer " + token(signing, claims={"iss": "accounts.google.com"})])
    asyncio.run(scenario())
    assert configurations == [{"trust_env": False, "follow_redirects": False, "timeout": 2}]
    assert len(requests) == 1
    assert str(requests[0].url) == auth.CERTIFICATE_URL
    assert requests[0].headers["accept-encoding"] == "identity"
    assert "authorization" not in requests[0].headers
    assert "metadata" not in str(requests[0].url)


@pytest.mark.parametrize("claims,remove", [
    ({"aud": AUDIENCE + "/"}, ()), ({"aud": [AUDIENCE]}, ()),
    ({"iss": "https://accounts.google.com.evil.example"}, ()),
    ({"sub": "999999999999999999999"}, ()),
    ({"email": "openecon-compute@openecon-workbench.iam.gserviceaccount.com"}, ()),
    ({"email_verified": False}, ()), ({"email_verified": "true"}, ()),
    ({"exp": 1}, ()), ({"iat": 9999999999, "exp": 10000000000}, ()),
    ({"iat": 1}, ()), ({"iat": True}, ()),
    ({}, ("exp",)), ({}, ("sub",)), ({}, ("email_verified",)),
])
def test_valid_signature_cannot_authorize_wrong_identity_scope_or_time(
        monkeypatch, signing, claims, remove):
    network(monkeypatch, signing)
    with pytest.raises(auth.BrokerAuthError) as error:
        asyncio.run(verifier().verify(["Bearer " + token(signing, claims=claims, remove=remove)]))
    assert error.value.status_code == 401
    assert str(error.value) == "Compute authentication required."


def test_wrong_signature_rejected_even_with_all_trusted_claims(monkeypatch, signing):
    network(monkeypatch, signing)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(auth.BrokerAuthError):
        asyncio.run(verifier().verify(["Bearer " + token(signing, key=other)]))


@pytest.mark.parametrize("headers", [{"alg": "HS256"}, {"alg": "none"},
                                      {"kid": "../../metadata"}, {"jku": "http://metadata/"},
                                      {"jwk": {}}, {"typ": "unexpected"}])
def test_untrusted_header_cannot_select_algorithm_key_or_network(monkeypatch, signing, headers):
    _, requests = network(monkeypatch, signing)
    valid = token(signing)
    parts = valid.split(".")
    parts[0] = jwt.utils.base64url_encode(json.dumps(
        {"alg": "RS256", "kid": "google-offline-key", **headers}).encode()).decode()
    with pytest.raises(auth.BrokerAuthError):
        asyncio.run(verifier().verify(["Bearer " + ".".join(parts)]))
    assert requests == []


@pytest.mark.parametrize("authorization", [[], [""], ["Basic secret"], ["Bearer "],
                                           ["Bearer a.b.c", "Bearer d.e.f"],
                                           ["Bearer a.b.c\nsecret"],
                                           ["Bearer " + "x" * 8193]])
def test_invalid_header_fails_before_network(monkeypatch, signing, authorization):
    _, requests = network(monkeypatch, signing)
    with pytest.raises(auth.BrokerAuthError):
        asyncio.run(verifier().verify(authorization))
    assert not requests


@pytest.mark.parametrize("kind", ["redirect", "oversize", "encoding", "malformed", "duplicate",
                                  "wrong_key", "wrong_shape", "timeout"])
def test_certificate_failures_are_bounded_and_fail_closed(monkeypatch, signing, kind):
    def response(request):
        if kind == "redirect":
            return httpx.Response(302, headers={"Location": "http://metadata.google.internal/"})
        if kind == "oversize":
            return httpx.Response(200, content=b" " * 65537)
        if kind == "encoding":
            return httpx.Response(200, content=b"opaque", headers={"Content-Encoding": "unexpected"})
        if kind == "malformed":
            return httpx.Response(200, content=b"not-json-provider-secret")
        if kind == "duplicate":
            value = json.dumps(signing[1])
            return httpx.Response(200, content=(f'{{"key":{value},"key":{value}}}').encode())
        if kind == "wrong_key":
            return httpx.Response(200, json={"google-offline-key": "not-a-certificate"})
        if kind == "wrong_shape":
            return httpx.Response(200, json=[])
        raise httpx.ReadTimeout("provider-secret", request=request)
    _, requests = network(monkeypatch, signing, response)
    gate = verifier()
    for _ in range(2):
        with pytest.raises(auth.BrokerAuthError) as error:
            asyncio.run(gate.verify(["Bearer " + token(signing)]))
        assert error.value.status_code == 503
        assert str(error.value) == "Compute authentication unavailable."
    assert len(requests) == 1


def test_rotation_refresh_and_unknown_key_flood_are_single_flight(monkeypatch, signing):
    _, requests = network(monkeypatch, signing)
    clock = [100.0]
    monkeypatch.setattr(auth, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    gate = verifier()

    async def unknown():
        with pytest.raises(auth.BrokerAuthError) as error:
            await gate.verify(["Bearer " + token(signing, headers={"kid": "new-key"})])
        assert error.value.status_code == 401

    async def scenario():
        await gate.verify(["Bearer " + token(signing)])
        await asyncio.gather(*(unknown() for _ in range(20)))
        assert len(requests) == 1
        clock[0] += 61
        await asyncio.gather(*(unknown() for _ in range(20)))
        assert len(requests) == 2
        await gate.verify(["Bearer " + token(signing)])
    asyncio.run(scenario())


def test_expired_certificate_cache_is_not_used_during_provider_failure(monkeypatch, signing):
    calls = []

    def response(request):
        calls.append(request)
        if len(calls) > 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"google-offline-key": signing[1]},
                              headers={"Cache-Control": "max-age=60"})
    network(monkeypatch, signing, response)
    clock = [100.0]
    monkeypatch.setattr(auth, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    gate = verifier()

    async def scenario():
        await gate.verify(["Bearer " + token(signing)])
        clock[0] += 61
        with pytest.raises(auth.BrokerAuthError) as error:
            await gate.verify(["Bearer " + token(signing)])
        assert error.value.status_code == 503
    asyncio.run(scenario())


def test_google_cdn_age_is_subtracted_before_local_ttl_cap(monkeypatch, signing):
    _, requests = network(monkeypatch, signing, lambda _: httpx.Response(
        200, json={"google-offline-key": signing[1]},
        headers={"Cache-Control": "public, max-age=20000", "Age": "5000"}))
    clock = [100.0]
    monkeypatch.setattr(auth, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    gate = verifier()

    async def scenario():
        await gate.verify(["Bearer " + token(signing)])
        clock[0] += 100
        await gate.verify(["Bearer " + token(signing)])
    asyncio.run(scenario())
    assert gate._expires == 3700 and len(requests) == 1


def test_unknown_key_does_not_hide_successful_google_rotation(monkeypatch, signing):
    calls = []

    def response(request):
        calls.append(request)
        kid = "google-offline-key" if len(calls) == 1 else "new-key"
        return httpx.Response(200, json={kid: signing[1]}, headers={"Cache-Control": "max-age=3600"})
    network(monkeypatch, signing, response)
    clock = [100.0]
    monkeypatch.setattr(auth, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    gate = verifier()

    async def scenario():
        await gate.verify(["Bearer " + token(signing)])
        clock[0] += 61
        await gate.verify(["Bearer " + token(signing, headers={"kid": "new-key"})])
    asyncio.run(scenario())
    assert len(calls) == 2


@pytest.mark.parametrize("headers", [[], [(b"x-serverless-authorization", b"Bearer fake.signature")],
                                      [(b"authorization", b"Bearer forged.token.signature")],
                                      [(b"authorization", b"Bearer a.b.c"),
                                       (b"authorization", b"Bearer a.b.c")]])
def test_unauthenticated_direct_http_cannot_read_body_acquire_or_touch_urls(headers):
    supervisor = Mock(spec=broker.Broker)
    supervisor.acquire.side_effect = AssertionError("Authentication must precede acquire")
    app = broker.create_app(supervisor, auth=verifier())
    receive = AsyncMock(side_effect=AssertionError("Authentication must precede body read"))
    events = []

    async def send(event):
        events.append(event)
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "path": "/execute", "raw_path": b"/execute",
             "query_string": b"", "root_path": "", "scheme": "http", "headers": headers,
             "client": ("10.0.0.2", 1234), "server": ("10.0.0.1", 8080)}
    asyncio.run(app(scope, receive, send))
    assert events[0]["status"] == 401
    supervisor.acquire.assert_not_called()
    receive.assert_not_called()


def test_signed_control_caller_reaches_existing_body_validation(monkeypatch, signing):
    network(monkeypatch, signing)
    runtime, watchdog = Mock(), Mock()
    supervisor = broker.Broker(runtime=runtime, watchdog=watchdog)
    with TestClient(broker.create_app(supervisor, auth=verifier())) as client:
        result = client.post("/execute", json={}, headers={
            "Authorization": "Bearer " + token(signing),
            "X-Serverless-Authorization": "Bearer signature-stripped.untrusted.token"})
    assert result.status_code == 400
    watchdog.arm.assert_called_once_with(240)
    watchdog.disarm.assert_called_once()
    assert not runtime.mock_calls and not supervisor.busy


def test_service_auth_has_no_missing_config_or_emulator_bypass(monkeypatch):
    for name in ("OPENECON_BROKER_AUDIENCE", "OPENECON_BROKER_CALLER_EMAIL", "OPENECON_BROKER_CALLER_SUB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "localhost:9099")
    with pytest.raises(ValueError):
        broker.create_app()
    monkeypatch.setenv("OPENECON_BROKER_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("OPENECON_BROKER_CALLER_EMAIL", EMAIL)
    monkeypatch.setenv("OPENECON_BROKER_CALLER_SUB", SUBJECT)
    assert auth.BrokerAuth.from_environment().caller_sub == SUBJECT


@pytest.mark.parametrize("values", [(AUDIENCE + "/", EMAIL, SUBJECT),
                                    ("http://metadata.google.internal", EMAIL, SUBJECT),
                                    (AUDIENCE, "someone@gmail.com", SUBJECT),
                                    (AUDIENCE, EMAIL, "not-a-subject")])
def test_invalid_trusted_configuration_is_refused(values):
    with pytest.raises(ValueError):
        auth.BrokerAuth(*values)
