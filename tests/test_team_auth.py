"""Authentication boundaries without network calls or Firebase credentials."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import sys

import pytest

from openecon.team_auth import TeamAuth, TeamAuthError, TeamIdentity

PROJECT = "openecon-workbench"


def claims(**changes):
    return {"uid": "person_123", "sub": "person_123", "email": "Person@Example.com",
            "name": " Person ", "email_verified": True, "aud": PROJECT,
            "iss": f"https://securetoken.google.com/{PROJECT}",
            "firebase": {"sign_in_provider": "password"}, **changes}


def authenticate(payload):
    return TeamAuth(PROJECT, verifier=lambda token: payload)


@pytest.mark.parametrize("provider", ["password", "google.com"])
def test_verified_identity_uses_uid_and_does_not_trust_role_claims(provider):
    identity = authenticate(claims(role="owner", enabled=True,
                                  firebase={"sign_in_provider": provider})).verify("signed-token")
    assert identity == TeamIdentity("person_123", "person@example.com", "Person", True)
    assert not hasattr(identity, "role") and not hasattr(identity, "enabled")
    with pytest.raises(FrozenInstanceError):
        identity.uid = "another-person"


@pytest.mark.parametrize("provider", ["password", "google.com"])
def test_unverified_identity_only_available_when_explicitly_requested(provider):
    auth = authenticate(claims(email_verified=False, firebase={"sign_in_provider": provider}))
    with pytest.raises(TeamAuthError) as error:
        auth.verify("signed-token")
    assert error.value.code == "EMAIL_VERIFICATION_REQUIRED"
    assert error.value.status_code == 403
    identity = auth.verify("signed-token", require_verified=False)
    assert identity.email_verified is False
    assert identity.uid == "person_123"


@pytest.mark.parametrize("changes", [
    {"aud": "foreign-project"}, {"aud": [PROJECT]}, {"iss": "https://accounts.google.com"},
    {"sub": "different-person"}, {"uid": ""}, {"uid": "user/another-document"},
    {"uid": ".."}, {"uid": "x" * 129}, {"uid": None},
    {"email": "not-an-email"}, {"email": " person@example.com"},
    {"email": "person@example.com\n"}, {"email": "person@exa<mple.com"},
    {"email_verified": "true"}, {"email_verified": 1}, {"email_verified": None},
    {"firebase": {"sign_in_provider": "anonymous"}},
    {"firebase": {"sign_in_provider": "custom"}},
    {"firebase": {"sign_in_provider": "password", "tenant": "another-tenant"}},
    {"firebase": None},
])
def test_invalid_identity_cannot_pass_even_account_status_verification(changes):
    with pytest.raises(TeamAuthError) as error:
        authenticate(claims(**changes)).verify("signed-token", require_verified=False)
    assert error.value.code == "INVALID_TOKEN"
    assert error.value.status_code == 401


@pytest.mark.parametrize("provider", [None, "", "anonymous", "custom", "github.com", "apple.com",
                                      "phone", "Google.com", "google.com ", ["google.com"],
                                      {"provider": "google.com"}])
def test_only_explicit_password_and_google_providers_are_supported(provider):
    with pytest.raises(TeamAuthError) as error:
        authenticate(claims(firebase={"sign_in_provider": provider})).verify("signed-token")
    assert error.value.code == "INVALID_TOKEN" and error.value.status_code == 401


@pytest.mark.parametrize("changes", [
    {"aud": "foreign-project"}, {"iss": "https://accounts.google.com"},
    {"sub": "another-uid"}, {"email_verified": "true"},
    {"firebase": {"sign_in_provider": "google.com", "tenant": "foreign-tenant"}},
])
def test_google_provider_does_not_relax_firebase_token_boundaries(changes):
    payload = claims(firebase={"sign_in_provider": "google.com"})
    payload.update(changes)
    with pytest.raises(TeamAuthError) as error:
        authenticate(payload).verify("signed-token")
    assert error.value.code == "INVALID_TOKEN"


def test_linked_google_and_password_identity_has_same_uid():
    password = authenticate(claims()).verify("password-token")
    google = authenticate(claims(firebase={"sign_in_provider": "google.com"})).verify("google-token")
    assert password == google


@pytest.mark.parametrize("value", [None, {}, [], "claims"])
def test_missing_or_non_mapping_claims_are_rejected(value):
    with pytest.raises(TeamAuthError):
        authenticate(value).verify("signed-token")


@pytest.mark.parametrize("token", [None, "", " ", "a\nb", "x" * 16385])
def test_malformed_token_is_rejected_before_verifier(token):
    def verifier(_):
        pytest.fail("Malformed token reached the Firebase verifier")
    with pytest.raises(TeamAuthError) as error:
        TeamAuth(PROJECT, verifier=verifier).verify(token)
    assert error.value.status_code == 401


def test_provider_failure_does_not_disclose_token_or_exception_details():
    def revoked(token):
        raise ValueError(f"revoked token: {token}, internal credential detail")
    with pytest.raises(TeamAuthError) as error:
        TeamAuth(PROJECT, verifier=revoked).verify("private-token-content")
    assert "private-token-content" not in str(error.value)
    assert "credential" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("project", ["", "Mixed-Case", "a/b/project", "test ", None])
def test_project_id_must_be_explicit_and_canonical(project):
    with pytest.raises(ValueError, match="project ID"):
        TeamAuth(project)


def fake_firebase(monkeypatch, *, failure=None):
    calls = []
    app = SimpleNamespace(project_id=PROJECT)
    classes = {name: type(name, (Exception,), {}) for name in [
        "CertificateFetchError", "InsufficientPermissionError", "UnavailableError",
        "DeadlineExceededError", "InternalError", "ResourceExhaustedError",
        "PermissionDeniedError", "InvalidIdTokenError", "UserDisabledError",
        "TenantIdMismatchError"]}

    def verify(token, **kwargs):
        calls.append((token, kwargs))
        if failure:
            raise classes[failure]("private provider details")
        return claims()

    auth = SimpleNamespace(verify_id_token=verify, **classes)
    created = []

    def get_app(name):
        if not created:
            raise ValueError("No app")
        assert name == f"openecon-team-{PROJECT}"
        return app

    def initialize_app(**kwargs):
        created.append(kwargs)
        return app

    monkeypatch.setitem(sys.modules, "firebase_admin", SimpleNamespace(
        auth=auth, exceptions=SimpleNamespace(**classes), get_app=get_app,
        initialize_app=initialize_app))
    monkeypatch.delenv("FIREBASE_AUTH_EMULATOR_HOST", raising=False)
    return app, created, calls


def test_sdk_adapter_pins_project_and_checks_revocation_on_every_request(monkeypatch):
    app, created, calls = fake_firebase(monkeypatch)
    verifier = TeamAuth(PROJECT)
    verifier.verify("first-token")
    verifier.verify("second-token")
    assert created == [{"options": {"projectId": PROJECT, "httpTimeout": 10},
                        "name": f"openecon-team-{PROJECT}"}]
    assert calls == [("first-token", {"app": app, "check_revoked": True}),
                     ("second-token", {"app": app, "check_revoked": True})]


def test_sdk_adapter_rejects_mismatched_existing_app(monkeypatch):
    app, _, calls = fake_firebase(monkeypatch)
    app.project_id = "foreign-project"
    with pytest.raises(TeamAuthError) as error:
        TeamAuth(PROJECT).verify("signed-token")
    assert error.value.status_code == 503
    assert calls == []


def test_sdk_initialization_outage_is_not_a_bad_user_token(monkeypatch):
    fake_firebase(monkeypatch)
    def unavailable(**kwargs):
        raise RuntimeError("private credential setup details")
    monkeypatch.setattr(sys.modules["firebase_admin"], "initialize_app", unavailable)
    with pytest.raises(TeamAuthError) as error:
        TeamAuth(PROJECT).verify("signed-token")
    assert error.value.status_code == 503
    assert "private credential" not in str(error.value)


@pytest.mark.parametrize("failure", ["InvalidIdTokenError", "UserDisabledError", "TenantIdMismatchError"])
def test_sdk_credential_rejection_remains_unauthorized(monkeypatch, failure):
    fake_firebase(monkeypatch, failure=failure)
    with pytest.raises(TeamAuthError) as error:
        TeamAuth(PROJECT).verify("signed-token")
    assert error.value.status_code == 401
    assert error.value.code == "INVALID_TOKEN"


@pytest.mark.parametrize("failure", ["CertificateFetchError", "InsufficientPermissionError",
                                  "UnavailableError", "PermissionDeniedError"])
def test_verification_outage_fails_closed_without_telling_user_to_reauthenticate(monkeypatch, failure):
    fake_firebase(monkeypatch, failure=failure)
    with pytest.raises(TeamAuthError) as error:
        TeamAuth(PROJECT).verify("signed-token")
    assert error.value.code == "AUTH_UNAVAILABLE"
    assert error.value.status_code == 503
    assert "private provider details" not in str(error.value)


def test_emulator_environment_cannot_disable_signature_checks(monkeypatch):
    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "localhost:9099")
    with pytest.raises(TeamAuthError) as error:
        TeamAuth(PROJECT).verify("unsigned-emulator-token")
    assert error.value.code == "AUTH_UNAVAILABLE"
    assert error.value.status_code == 503
