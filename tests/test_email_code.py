"""Email proof, abuse bounds, one-use races and Firebase UID preservation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from openecon.email_code import EmailCodes, SmtpCodeDelivery, configured_email_codes, firebase_email_identity
from openecon.team_auth import TeamAuth, TeamAuthError
from openecon.team_server import create_team_app
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamError, TeamStore

EMAIL = 'person@example.test'
VERIFIER = 'a' * 64
CHALLENGE = hashlib.sha256(VERIFIER.encode()).hexdigest()
ORIGIN = 'https://app.example.test'


@pytest.fixture
def service():
    instant = [datetime(2026, 10, 11, tzinfo=timezone.utc)]
    db = MemoryDocuments()
    send = Mock()
    identity = Mock(return_value='existing-google-uid')
    issue = Mock(return_value='synthetic-custom-token')
    value = EmailCodes(db, secret=b'x' * 32, send=send, identity=identity,
                       issue=issue, clock=lambda: instant[0])
    return SimpleNamespace(value=value, db=db, send=send, identity=identity, issue=issue, instant=instant)


def begin(service):
    result = service.value.begin(EMAIL, CHALLENGE)
    return result['request_id'], service.send.call_args.args[1]


def test_code_never_stored_returned_or_used_before_verification(service):
    request_id, code = begin(service)
    assert len(code) == 6 and code.isdigit()
    row = service.db.get('oe_email_codes/' + request_id)
    assert code not in json.dumps(row, default=str)
    assert VERIFIER not in json.dumps(service.db.data, default=str)
    service.identity.assert_not_called()
    service.issue.assert_not_called()
    result = service.value.verify(request_id, code, VERIFIER)
    assert result == {'custom_token': 'synthetic-custom-token'}
    service.identity.assert_called_once_with(EMAIL)
    service.issue.assert_called_once_with('existing-google-uid')
    with pytest.raises(TeamError, match='incorrect or has expired'):
        service.value.verify(request_id, code, VERIFIER)
    assert service.issue.call_count == 1


def test_wrong_codes_and_foreign_client_proofs_persist_attempt_limit(service):
    request_id, code = begin(service)
    for n in range(5):
        wrong_code = f'{(int(code) + 1) % 1000000:06d}'
        with pytest.raises(TeamError):
            service.value.verify(request_id, wrong_code if n % 2 else code,
                                 VERIFIER if n % 2 else 'b' * 64)
        assert service.db.get('oe_email_codes/' + request_id)['attempts'] == n + 1
    with pytest.raises(TeamError):
        service.value.verify(request_id, code, VERIFIER)
    service.issue.assert_not_called()


def test_expiry_and_resend_invalidate_previous_challenge(service):
    old_id, old_code = begin(service)
    with pytest.raises(TeamError) as limit:
        begin(service)
    assert limit.value.status_code == 429
    service.instant[0] += timedelta(seconds=60)
    new_id, new_code = begin(service)
    assert old_id != new_id
    with pytest.raises(TeamError):
        service.value.verify(old_id, old_code, VERIFIER)
    service.instant[0] += timedelta(seconds=600)
    with pytest.raises(TeamError):
        service.value.verify(new_id, new_code, VERIFIER)
    service.identity.assert_not_called()


def test_email_and_global_limits_bound_sends(service):
    for _ in range(5):
        begin(service)
        service.instant[0] += timedelta(seconds=60)
    with pytest.raises(TeamError):
        begin(service)
    assert service.send.call_count == 5
    service.instant[0] += timedelta(hours=1)
    begin(service)
    key = f'oe_limits/email-code-global-{service.instant[0]:%Y%m%d}'
    service.db.put(key, {'count': 512})
    with pytest.raises(TeamError):
        service.value.begin('another@example.test', CHALLENGE)
    assert service.send.call_count == 6


def test_delivery_failure_cannot_advance_to_usable_code_or_expose_transport_secret(service):
    service.send.side_effect = RuntimeError('smtp-credential-must-not-escape')
    with pytest.raises(TeamError) as failure:
        begin(service)
    assert failure.value.status_code == 503
    assert 'smtp-credential' not in str(failure.value)
    assert not service.db.scan('oe_email_codes')
    service.issue.assert_not_called()


def test_two_concurrent_verifications_issue_exactly_one_session(service):
    request_id, code = begin(service)
    def verify():
        try:
            return service.value.verify(request_id, code, VERIFIER)
        except TeamError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda _: verify(), range(2)))
    assert sum(value is not None for value in result) == 1
    service.identity.assert_called_once()
    service.issue.assert_called_once()


def test_identity_failure_still_consumes_the_code(service):
    request_id, code = begin(service)
    service.identity.side_effect = TeamAuthError('AUTH_UNAVAILABLE', 'Unavailable', 503)
    with pytest.raises(TeamAuthError):
        service.value.verify(request_id, code, VERIFIER)
    with pytest.raises(TeamError):
        service.value.verify(request_id, code, VERIFIER)
    assert service.identity.call_count == 1
    service.issue.assert_not_called()


def client(service=None):
    db = service.db if service else MemoryDocuments()
    store = TeamStore(db, owner_email='owner@example.test', public_origin=ORIGIN)
    app = create_team_app(store=store, storage=MemoryStorage(), auth=TeamAuth('openecon-test'),
                          public_origin=ORIGIN, firebase_config={
                              'projectId': 'openecon-test', 'authDomain': 'openecon-test.firebaseapp.com'},
                          email_codes=service.value if service else None, passwordless_only=True)
    return TestClient(app)


def test_public_code_endpoints_origin_strict_bodies_no_enumeration_or_member_writes(service):
    with client(service) as http:
        config = http.get('/api/auth/config').json()
        assert config['email_code_available'] and not config['account_link_available']
        assert http.post('/api/desktop/account-link', json={}).status_code == 410
        assert http.post('/api/auth/email-code', json={'email': EMAIL, 'challenge': CHALLENGE},
                         headers={'Origin': 'https://foreign.test'}).status_code == 403
        assert not service.db.data
        result = http.post('/api/auth/email-code', json={'email': ' Person@Example.test ', 'challenge': CHALLENGE})
        assert result.status_code == 202
        assert set(result.json()) == {'request_id', 'expires_at', 'retry_after'}
        request_id = result.json()['request_id']
        code = service.send.call_args.args[1]
        for payload in [{'code': code, 'verifier': VERIFIER, 'uid': 'foreign'},
                        {'code': 123456, 'verifier': VERIFIER},
                        {'code': 'sensitive-not-echoed', 'verifier': VERIFIER}]:
            response = http.post('/api/auth/email-code/' + request_id + '/verify', json=payload)
            assert response.status_code == 422
            assert code not in response.text and 'sensitive-not-echoed' not in response.text
        assert http.post('/api/auth/email-code/' + request_id + '/verify',
                         json={'code': code, 'verifier': VERIFIER}).status_code == 200
        assert not service.db.scan('oe_projects') and not service.db.scan('oe_users')


def test_unconfigured_sender_is_unavailable_and_never_claims_delivery():
    with client() as http:
        assert http.get('/api/auth/config').json()['email_code_available'] is False
        result = http.post('/api/auth/email-code', json={'email': EMAIL, 'challenge': CHALLENGE})
        assert result.status_code == 503


@pytest.mark.parametrize('email', ['x\r\nBcc:third@example.test', 'a@b', 'x y@example.test', 'ü@example.test'])
def test_invalid_email_never_reaches_delivery(service, email):
    with client(service) as http:
        assert http.post('/api/auth/email-code', json={'email': email, 'challenge': CHALLENGE}).status_code == 422
        service.send.assert_not_called()


def test_production_accepts_google_and_verified_email_code_but_rejects_password_tokens():
    claims = {'uid': 'retained', 'sub': 'retained', 'email': EMAIL, 'email_verified': True,
              'aud': 'openecon-test', 'iss': 'https://securetoken.google.com/openecon-test',
              'firebase': {'sign_in_provider': 'google.com'}}
    auth = TeamAuth('openecon-test', verifier=lambda _: claims, passwordless_only=True)
    assert auth.verify('signed').uid == 'retained'
    claims['firebase']['sign_in_provider'] = 'password'
    with pytest.raises(TeamAuthError):
        auth.verify('signed')
    claims['firebase']['sign_in_provider'] = 'custom'
    for value in [False, 'true', 1, None]:
        claims['openecon_email_code'] = value
        with pytest.raises(TeamAuthError):
            auth.verify('signed')
    claims['openecon_email_code'] = True
    assert auth.verify('signed').uid == 'retained'
    claims['email_verified'] = False
    with pytest.raises(TeamAuthError):
        auth.verify('signed')


def test_google_uid_and_provider_are_preserved_and_disabled_users_are_not_reenabled(monkeypatch):
    import firebase_admin
    from firebase_admin import auth
    monkeypatch.setattr(firebase_admin, 'get_app', lambda _: SimpleNamespace(project_id='openecon-test'))
    user = SimpleNamespace(uid='google-uid', email_verified=True, disabled=False)
    monkeypatch.setattr(auth, 'get_user_by_email', lambda email, app: user)
    create, update = Mock(), Mock()
    monkeypatch.setattr(auth, 'create_user', create)
    monkeypatch.setattr(auth, 'update_user', update)
    assert firebase_email_identity('openecon-test', EMAIL) == 'google-uid'
    create.assert_not_called()
    update.assert_not_called()
    user.disabled = True
    with pytest.raises(TeamError):
        firebase_email_identity('openecon-test', EMAIL)
    update.assert_not_called()


def test_first_verified_code_creates_no_password_and_unverified_existing_uid_is_retained(monkeypatch):
    import firebase_admin
    from firebase_admin import auth
    monkeypatch.setattr(firebase_admin, 'get_app', lambda _: SimpleNamespace(project_id='openecon-test'))
    user = SimpleNamespace(uid='new-uid', email_verified=True, disabled=False)
    lookup = Mock(side_effect=auth.UserNotFoundError('Not found'))
    create = Mock(return_value=user)
    monkeypatch.setattr(auth, 'get_user_by_email', lookup)
    monkeypatch.setattr(auth, 'create_user', create)
    assert firebase_email_identity('openecon-test', EMAIL) == 'new-uid'
    assert create.call_args.kwargs['email_verified'] is True
    assert set(create.call_args.kwargs) == {'email', 'email_verified', 'app'}
    existing = SimpleNamespace(uid='retained-uid', email_verified=False, disabled=False)
    monkeypatch.setattr(auth, 'get_user_by_email', Mock(return_value=existing))
    revoked, updated = Mock(), Mock(return_value=existing)
    monkeypatch.setattr(auth, 'revoke_refresh_tokens', revoked)
    monkeypatch.setattr(auth, 'update_user', updated)
    assert firebase_email_identity('openecon-test', EMAIL) == 'retained-uid'
    assert revoked.call_args.args == ('retained-uid',)
    assert updated.call_args.kwargs['email_verified'] is True


@pytest.mark.parametrize('environment', ['emulator', 'foreign-project'])
def test_email_identity_refuses_emulator_and_wrong_project_before_lookup(monkeypatch, environment):
    import firebase_admin
    from firebase_admin import auth
    lookup = Mock()
    monkeypatch.setattr(auth, 'get_user_by_email', lookup)
    monkeypatch.setattr(firebase_admin, 'get_app', lambda _: SimpleNamespace(project_id='foreign'))
    if environment == 'emulator':
        monkeypatch.setenv('FIREBASE_AUTH_EMULATOR_HOST', 'localhost:9099')
    else:
        monkeypatch.delenv('FIREBASE_AUTH_EMULATOR_HOST', raising=False)
    with pytest.raises(TeamAuthError):
        firebase_email_identity('openecon-test', EMAIL)
    lookup.assert_not_called()


def test_sender_and_shared_secret_must_be_configured_together(monkeypatch):
    monkeypatch.delenv('OPENECON_EMAIL_CODE_SMTP', raising=False)
    monkeypatch.delenv('OPENECON_EMAIL_CODE_KEY', raising=False)
    assert configured_email_codes(None, None, 'openecon-test') is None
    monkeypatch.setenv('OPENECON_EMAIL_CODE_KEY', 'a' * 64)
    with pytest.raises(ValueError):
        configured_email_codes(None, None, 'openecon-test')
    monkeypatch.setenv('OPENECON_EMAIL_CODE_SMTP', '{}')
    store = SimpleNamespace(db=MemoryDocuments())
    with pytest.raises(ValueError):
        configured_email_codes(store, None, 'openecon-test')
    config = dict(host='smtp.example.test', port=465, username='service',
                  password='synthetic-secret', sender='login@example.test')
    monkeypatch.setenv('OPENECON_EMAIL_CODE_SMTP', json.dumps(config))
    assert isinstance(configured_email_codes(store, None, 'openecon-test'), EmailCodes)
    monkeypatch.setenv('OPENECON_EMAIL_CODE_KEY', 'a' * 63)
    with pytest.raises(ValueError):
        configured_email_codes(store, None, 'openecon-test')


def test_smtp_uses_tls_authenticated_sender_and_only_the_requested_recipient(monkeypatch):
    import smtplib
    config = dict(host='smtp.example.test', port=587, username='service', password='synthetic-secret', sender='login@example.test')
    smtp = Mock()
    smtp.__enter__ = Mock(return_value=smtp)
    smtp.__exit__ = Mock(return_value=False)
    smtp.send_message.return_value = {}
    constructor = Mock(return_value=smtp)
    monkeypatch.setattr(smtplib, 'SMTP', constructor)
    SmtpCodeDelivery(config)(EMAIL, '123456')
    constructor.assert_called_once_with(config['host'], 587, timeout=15)
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with('service', 'synthetic-secret')
    message = smtp.send_message.call_args.args[0]
    assert message['To'] == EMAIL and message['From'] == 'login@example.test'
    assert '123456' in message.get_content() and 'synthetic-secret' not in message.as_string()
    with pytest.raises(ValueError):
        SmtpCodeDelivery({**config, 'port': 25})
    monkeypatch.setattr(smtplib, 'SMTP_SSL', constructor)
    SmtpCodeDelivery({**config, 'port': 465})(EMAIL, '654321')
    assert constructor.call_args.kwargs['context'].check_hostname is True
    assert smtp.starttls.call_count == 1
    smtp.send_message.return_value = {EMAIL: (550, b'rejected')}
    with pytest.raises(RuntimeError, match='rejected'):
        SmtpCodeDelivery(config)(EMAIL, '123456')
