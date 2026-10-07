"""One-use browser login and data-only publication, with no cloud execution."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_server import create_team_app
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamStore

ORIGIN = 'https://desktop-test.example.com'
PROJECT = 'openecon-desktop-test'
VERIFIER = 'a' * 64
CHALLENGE = hashlib.sha256(VERIFIER.encode()).hexdigest()


def claims(uid, provider='google.com', desktop=None):
    result = {'uid': uid, 'sub': uid, 'email': f'{uid}@example.com', 'email_verified': True,
              'aud': PROJECT, 'iss': f'https://securetoken.google.com/{PROJECT}',
              'firebase': {'sign_in_provider': provider}}
    if desktop is not None:
        result['openecon_desktop'] = desktop
    return result


@pytest.fixture
def api():
    store = TeamStore(MemoryDocuments(), owner_email='owner@example.com', public_origin=ORIGIN)
    users = {uid: TeamIdentity(uid, f'{uid}@example.com', email_verified=True)
             for uid in ('owner', 'editor', 'viewer', 'outsider')}
    for user in users.values():
        store.me(user)
    project = store.create_project(users['owner'], 'Desktop')
    for role in ('editor', 'viewer'):
        invite = store.invite(project['id'], users['owner'], users[role].email, role)
        store.accept(invite['id'], users[role])
    storage, runner, issuer = MemoryStorage(), Mock(), Mock(side_effect=lambda uid: f'custom-{uid}')
    auth = TeamAuth(PROJECT, verifier=lambda token: claims(token))
    app = create_team_app(store=store, storage=storage, auth=auth, runner=runner,
                          public_origin=ORIGIN, firebase_config={'projectId': PROJECT,
                          'authDomain': f'{PROJECT}.firebaseapp.com'}, desktop_token_issuer=issuer)
    with TestClient(app, base_url=ORIGIN) as client:
        yield SimpleNamespace(client=client, store=store, storage=storage, runner=runner,
                              issuer=issuer, pid=project['id'])


def headers(uid='owner'):
    return {'Authorization': f'Bearer {uid}', 'Origin': ORIGIN}


def begin(api):
    response = api.client.post('/api/desktop/login', json={'challenge': CHALLENGE})
    assert response.status_code == 201
    return response.json()['request_id']


def test_login_proof_is_bound_pending_and_one_use(api):
    request_id = begin(api)
    path = f'/api/desktop/login/{request_id}'
    assert api.client.post(path+'/exchange', json={'verifier': VERIFIER}).status_code == 202
    assert api.client.post(path+'/authorize').status_code == 401
    assert api.client.post(path+'/authorize', headers=headers('editor')).status_code == 200
    assert api.client.post(path+'/exchange', json={'verifier': 'b'*64}).status_code == 403
    response = api.client.post(path+'/exchange', json={'verifier': VERIFIER})
    assert response.json() == {'custom_token': 'custom-editor'}
    api.issuer.assert_called_once_with('editor')
    assert api.client.post(path+'/exchange', json={'verifier': VERIFIER}).status_code == 410
    assert api.store.db.get(f'oe_desktop_logins/{request_id}') is None


def test_login_cannot_reassign_account_or_accept_cross_origin(api):
    request_id = begin(api)
    path = f'/api/desktop/login/{request_id}/authorize'
    assert api.client.post(path, headers=headers('owner')).status_code == 200
    assert api.client.post(path, headers=headers('editor')).status_code == 409
    assert api.client.post('/api/desktop/login', json={'challenge': CHALLENGE},
                           headers={'Origin': 'https://evil.example'}).status_code == 403


def test_expired_or_malformed_login_is_rejected_without_minting(api):
    request_id = begin(api)
    entry = api.store.db.get(f'oe_desktop_logins/{request_id}')
    entry['expires_at'] = datetime.now(timezone.utc)-timedelta(seconds=1)
    api.store.db.put(f'oe_desktop_logins/{request_id}', entry)
    assert api.client.post(f'/api/desktop/login/{request_id}/authorize', headers=headers()).status_code == 410
    assert api.client.post('/api/desktop/login', json={'challenge': '../wrong'}).status_code == 422
    assert api.client.post('/api/desktop/login', json={'challenge': CHALLENGE, 'uid': 'owner'}).status_code == 422
    api.issuer.assert_not_called()


def test_login_begin_has_transactional_daily_budget(api):
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    api.store.db.put(f'oe_limits/desktop-login-{today}', {'count': 512})
    assert api.client.post('/api/desktop/login', json={'challenge': CHALLENGE}).status_code == 429


@pytest.mark.parametrize('marker,accepted', [(True,True),(False,False),('true',False),(None,False)])
def test_custom_provider_requires_explicit_desktop_claim(marker, accepted):
    auth = TeamAuth(PROJECT, verifier=lambda token: claims('owner', 'custom', marker))
    if accepted:
        assert auth.verify('opaque').uid == 'owner'
    else:
        with pytest.raises(ValueError):
            auth.verify('opaque')


def record(actor='owner'):
    return {'id': 'local-run-1', 'actor_uid': actor, 'code': 'raise RuntimeError("MUST NOT EXECUTE")',
            'status': 'ok', 'stdout': '42', 'outputs': [{'type':'text','data':'42'}],
            'error': None, 'duration_ms': 4, 'session_generation': 1}


def test_local_result_publication_is_data_only_shared_and_idempotent(api):
    path = f'/api/projects/{api.pid}/workspace/desktop/results'
    response = api.client.post(path, headers=headers('editor'), json={'record':record('editor'),'input_files':[]})
    assert response.status_code == 201
    assert api.client.post(path, headers=headers('editor'), json={'record':record('editor'),'input_files':[]}).json() == response.json()
    changed = record('editor')
    changed['stdout'] = 'different result with the same identifier'
    assert api.client.post(path, headers=headers('editor'), json={'record':changed,'input_files':[]}).status_code == 409
    history = api.client.get(f'/api/projects/{api.pid}/workspace/console', headers=headers('viewer')).json()
    saved = history['history'][0]
    assert saved['execution_origin'] == 'desktop'
    assert saved['actor_email'] == 'editor@example.com'
    assert saved['code'] == record()['code']
    assert 'events' not in saved
    assert api.store.project(api.pid, TeamIdentity('owner','owner@example.com',email_verified=True))['active_run'] is None
    api.runner.start.assert_not_called()


def test_desktop_archive_preserves_output_order_and_persisted_member_readback(api):
    events = [{'type': 'stdout', 'text': 'Before\n'}, {'type': 'output', 'index': 0},
              {'type': 'stdout', 'text': 'Between\n'}, {'type': 'output', 'index': 1}]
    submitted = record('editor')
    submitted.update(stdout='Before\nBetween\n', events=events, outputs=[
        {'type': 'text', 'data': 'first display'}, {'type': 'latex', 'data': r'\text{Second display}'},
    ])
    path = f'/api/projects/{api.pid}/workspace/desktop/results'
    response = api.client.post(path, headers=headers('editor'),
                               json={'record': submitted, 'input_files': []})
    assert response.status_code == 201
    run = api.store.db.get(f'oe_projects/{api.pid}/runs/{response.json()["id"]}')
    archived = json.loads(api.storage.get(run['result']))
    assert archived['events'] == events and archived['stdout'] == submitted['stdout']
    history = api.client.get(f'/api/projects/{api.pid}/workspace/console', headers=headers('viewer')).json()
    saved = history['history'][0]
    assert saved['events'] == events and saved['stdout'] == submitted['stdout']
    assert [item['type'] for item in saved['outputs']] == ['text', 'latex']
    assert saved['execution_origin'] == 'desktop'
    assert saved['actor_email'] == 'editor@example.com'
    api.runner.start.assert_not_called()


@pytest.mark.parametrize('events', [
    None,
    [{'type': 'output', 'index': True}],
    [{'type': 'output', 'index': 0}, {'type': 'output', 'index': 0}],
    [{'type': 'stdout', 'text': '42'}],
    [{'type': 'stdout', 'text': 'different'}, {'type': 'output', 'index': 0}],
    [{'type': 'stdout', 'text': '42', 'extra': 'untrusted'}, {'type': 'output', 'index': 0}],
])
def test_desktop_archive_rejects_malformed_output_order_before_any_save(api, events):
    submitted = record('editor')
    submitted['events'] = events
    response = api.client.post(f'/api/projects/{api.pid}/workspace/desktop/results',
                               headers=headers('editor'), json={'record': submitted, 'input_files': []})
    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'INVALID_RESULT'
    assert not api.storage.objects
    assert api.store.runs(api.pid, TeamIdentity('owner', 'owner@example.com', email_verified=True)) == []
    api.runner.start.assert_not_called()


@pytest.mark.parametrize('uid', ['viewer','outsider'])
def test_local_results_cannot_bypass_project_write_membership(api, uid):
    response = api.client.post(f'/api/projects/{api.pid}/workspace/desktop/results',
                               headers=headers(uid), json={'record':record(),'input_files':[]})
    assert response.status_code in {403,404}
    assert not api.storage.objects
    api.runner.start.assert_not_called()


def test_local_result_rejects_unknown_input_and_malformed_table(api):
    path = f'/api/projects/{api.pid}/workspace/desktop/results'
    assert api.client.post(path, headers=headers(), json={'record':record(),
                           'input_files':[{'id':'1'*32,'data_hash':'a'*64}]}).status_code == 409
    invalid = record()
    invalid['outputs'] = [{'type':'table','data':{'columns':['x'],'rows':[[1,2]],'total_rows':1,'total_columns':1}}]
    assert api.client.post(path, headers=headers(), json={'record':invalid,'input_files':[]}).status_code == 422
    assert not api.storage.objects


def test_publication_does_not_attribute_another_accounts_pending_result_to_current_user(api):
    response = api.client.post(f'/api/projects/{api.pid}/workspace/desktop/results',
        headers=headers('editor'), json={'record': record('owner'), 'input_files': []})
    assert response.status_code == 403
    assert not api.storage.objects
    api.runner.start.assert_not_called()


def test_result_archive_body_budget_is_scoped_and_streamed(api):
    from openecon.team_server import request_body_limit
    path = f'/api/projects/{api.pid}/workspace/desktop/results'
    assert request_body_limit(path) == 3 * 1024**2
    assert request_body_limit('/api/desktop/login') == 512 * 1024
    assert request_body_limit('/api/projects/wrong/workspace/desktop/results') == 512 * 1024
    response = api.client.post(path, headers=headers(),
                               content=iter([b' ' * (1024**2)] * 4))
    assert response.status_code == 413
    assert not api.storage.objects
    api.runner.start.assert_not_called()


def test_custom_token_has_short_expiry_and_no_extra_authority():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from google.auth import crypt, jwt
    from openecon.desktop_cloud import firebase_custom_token
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    signer = crypt.RSASigner.from_string(private)
    email = 'transfer@test-project.iam.gserviceaccount.com'
    token = firebase_custom_token(SimpleNamespace(signer=SimpleNamespace(
        signer=signer, service_account_email=email)), 'disposable-uid')
    public = key.public_key().public_bytes(serialization.Encoding.PEM,
                                          serialization.PublicFormat.SubjectPublicKeyInfo)
    decoded = jwt.decode(token, certs=public)
    assert decoded['iss'] == decoded['sub'] == email
    assert decoded['uid'] == 'disposable-uid'
    assert decoded['exp'] - decoded['iat'] == 300
    assert decoded['claims'] == {'openecon_desktop': True}
    assert decoded['aud'].endswith('v1.IdentityToolkit')
