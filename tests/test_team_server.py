"""HTTP authorization, tenant boundaries, revocation, untrusted output and retired cloud runs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_server import create_team_app, validate_worker_result
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamError, TeamStore, now

ORIGIN = 'https://app.example.com'
FIREBASE_PROJECT = 'openecon-test'
FIREBASE_CONFIG = {'projectId': FIREBASE_PROJECT,
                   'authDomain': f'{FIREBASE_PROJECT}.firebaseapp.com'}


@pytest.mark.parametrize('case', json.loads((Path(__file__).parent / 'fixtures/project-create.json').read_text()),
                         ids=lambda case: case['id'])
def test_project_creation_field_contract_and_readback(api, case):
    before = deepcopy(api.store.db.data)
    response = api.client.post('/api/projects', headers={**header('owner'), 'content-type': 'application/json'},
                               content=json.dumps({key: case[key] for key in ('name', 'description')}, ensure_ascii=True))
    if case['valid']:
        assert response.status_code == 201, response.text
        project = response.json()
        assert project['name'] == case['normalized_name']
        assert project['description'] == case['normalized_description']
        saved = api.store.project(project['id'], api.users['owner'])
        assert saved['name'] == project['name'] and saved['description'] == project['description']
        listed = api.client.get('/api/projects', headers=header('owner')).json()['projects']
        assert next(value for value in listed if value['id'] == project['id']) == project
    else:
        assert response.status_code == 422, response.text
        fields = {error['loc'][-1] for error in response.json()['detail']}
        assert fields == set(case['fields'])
        assert all(set(error) == {'loc', 'msg', 'type'} for error in response.json()['detail'])
        assert api.store.db.data == before
    assert api.storage.objects == {}


@pytest.mark.parametrize('field,value', [('name', None), ('name', 123), ('name', True),
                                       ('description', None), ('description', 123), ('description', [])])
def test_project_creation_rejects_nontext_without_writes(api, field, value):
    before = deepcopy(api.store.db.data)
    response = api.client.post('/api/projects', headers=header('owner'),
                               json={'name': 'Study', 'description': 'Retained', field: value})
    assert response.status_code == 422
    assert api.store.db.data == before


def test_project_creation_malformed_unicode_extra_field_has_safe_errors(api):
    before = deepcopy(api.store.db.data)
    response = api.client.post('/api/projects', headers={**header('owner'), 'content-type': 'application/json'},
        content=json.dumps({'name': 'Study', 'description': 'Kept', '\ud800': {'nested': 'Not echoed'}}, ensure_ascii=True))
    assert response.status_code == 422
    # Pydantic rejects a malformed Unicode key at the dictionary boundary.
    assert response.json()['detail'][0]['loc'] == ['body']
    assert 'Not echoed' not in response.text and 'input' not in response.json()['detail'][0]
    assert api.store.db.data == before


def package_manifest():
    return {'schema': 1, 'python': '3.13',
            'core': {'openecon': '0.3.3a1', 'torch': '2.9.0'},
            'requirements': [{'name': 'requests', 'version': '2.32.5'}],
            'locked': [{'name': 'requests', 'version': '2.32.5'},
                       {'name': 'urllib3', 'version': '2.5.0'}]}


def test_environment_metadata_round_trip_and_bootstrap_share_no_binaries(api):
    initial = api.client.get(api.prefix + '/environment', headers=header('viewer'))
    assert initial.status_code == 200
    assert initial.json() == {'manifest': None, 'version': 0}
    saved = api.client.put(api.prefix + '/environment', headers=header('editor'),
                           json={'manifest': package_manifest(), 'version': 0})
    assert saved.status_code == 200, saved.text
    assert saved.json() == {'manifest': package_manifest(), 'version': 1}
    assert saved.headers['cache-control'] == 'no-store'
    assert api.client.get(api.prefix + '/environment', headers=header('viewer')).json() == saved.json()
    bootstrap = api.client.get(api.prefix + '/bootstrap', headers=header('owner')).json()
    assert bootstrap['environment'] == saved.json()
    assert api.client.get(f'/api/projects/{api.other}/workspace/environment',
                          headers=header('owner')).json() == {'manifest': None, 'version': 0}
    assert api.storage.objects == {}


@pytest.mark.parametrize('uid,status', [('owner', 200), ('editor', 200), ('viewer', 403),
                                      ('outsider', 404), ('unverified', 403)])
def test_environment_write_requires_verified_editor_membership(api, uid, status):
    response = api.client.put(api.prefix + '/environment', headers=header(uid),
                               json={'manifest': package_manifest(), 'version': 0})
    assert response.status_code == status, response.text
    if status != 200:
        assert api.store.environment(api.pid, api.users['owner']) == {'manifest': None, 'version': 0}


@pytest.mark.parametrize('uid,status', [('owner', 200), ('editor', 200), ('viewer', 200),
                                      ('outsider', 404), ('unverified', 403)])
def test_environment_read_requires_verified_project_membership(api, uid, status):
    assert api.client.get(api.prefix + '/environment', headers=header(uid)).status_code == status


def test_environment_optimistic_conflict_preserves_manifest(api):
    saved = api.client.put(api.prefix + '/environment', headers=header('owner'),
                           json={'manifest': package_manifest(), 'version': 0}).json()
    response = api.client.put(api.prefix + '/environment', headers=header('editor'),
                              json={'manifest': package_manifest(), 'version': 0})
    assert response.status_code == 409
    assert response.json()['detail']['code'] == 'VERSION_CONFLICT'
    assert api.client.get(api.prefix + '/environment', headers=header('viewer')).json() == saved


def test_environment_revoked_token_cannot_read_or_write(api):
    api.revoked.add('editor')
    assert api.client.get(api.prefix + '/environment', headers=header('editor')).status_code == 401
    response = api.client.put(api.prefix + '/environment', headers=header('editor'),
                              json={'manifest': package_manifest(), 'version': 0})
    assert response.status_code == 401
    assert api.store.environment(api.pid, api.users['owner']) == {'manifest': None, 'version': 0}


@pytest.mark.parametrize('role', [None, 'viewer'])
def test_environment_write_rechecks_membership_when_transaction_starts(api, role):
    original = api.store.save_environment

    def revoked_before_save(project_id, user, manifest, version):
        api.store.change_member(project_id, user.uid, api.users['owner'], role)
        return original(project_id, user, manifest, version)

    api.store.save_environment = revoked_before_save
    response = api.client.put(api.prefix + '/environment', headers=header('editor'),
                              json={'manifest': package_manifest(), 'version': 0})
    assert response.status_code == (404 if role is None else 403)
    assert api.store.environment(api.pid, api.users['owner']) == {'manifest': None, 'version': 0}


@pytest.mark.parametrize('suffix', ['/environment', '/bootstrap'])
def test_environment_read_rechecks_membership_after_manifest_document_read(api, suffix):
    api.store.save_environment(api.pid, api.users['owner'], package_manifest(), 0)
    original = api.store.db.get
    removed = False

    def racing_read(path):
        nonlocal removed
        result = original(path)
        if path.endswith('/workspace/environment') and not removed:
            removed = True
            api.store.change_member(api.pid, 'editor', api.users['owner'])
        return result

    api.store.db.get = racing_read
    response = api.client.get(api.prefix + suffix, headers=header('editor'))
    assert response.status_code == 404
    assert 'requests' not in response.text


@pytest.mark.parametrize('change', ['schema_bool', 'url', 'path', 'vcs', 'extra', 'unlocked',
                                  'reserved_core', 'too_many'])
def test_environment_api_rejects_unsafe_manifests_before_storage(api, change):
    manifest = package_manifest()
    if change == 'schema_bool':
        manifest['schema'] = True
    elif change in {'url', 'path', 'vcs'}:
        manifest['locked'][0]['version'] = {
            'url': 'https://example.com/package.whl', 'path': '../package.whl',
            'vcs': 'git+https://example.com/package',
        }[change]
    elif change == 'extra':
        manifest['command'] = 'pip install requests'
    elif change == 'unlocked':
        manifest['locked'] = []
    elif change == 'reserved_core':
        manifest['locked'].append({'name': 'torch', 'version': '2.9.0'})
    else:
        manifest['locked'] = [{'name': f'package-{i}', 'version': '1.0'} for i in range(101)]
    response = api.client.put(api.prefix + '/environment', headers=header('editor'),
                              json={'manifest': manifest, 'version': 0})
    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'INVALID_ENVIRONMENT'
    assert api.store.environment(api.pid, api.users['owner']) == {'manifest': None, 'version': 0}
    assert api.storage.objects == {}


@pytest.mark.parametrize('body', [
    {'manifest': None, 'version': 0}, {'manifest': package_manifest(), 'version': True},
    {'manifest': package_manifest(), 'version': -1},
    {'manifest': package_manifest(), 'version': 0, 'install': True},
    {'manifest': package_manifest(), 'version': 2**63 - 1},
])
def test_environment_http_body_is_strict(api, body):
    response = api.client.put(api.prefix + '/environment', headers=header('editor'), json=body)
    assert response.status_code == 422
    assert api.store.environment(api.pid, api.users['owner']) == {'manifest': None, 'version': 0}


def claims(uid, verified=True):
    return {'uid': uid, 'sub': uid, 'email': f'{uid}@example.com', 'name': uid,
            'aud': FIREBASE_PROJECT, 'iss': f'https://securetoken.google.com/{FIREBASE_PROJECT}',
            'firebase': {'sign_in_provider': 'password'}, 'email_verified': verified}


class StubStorage(MemoryStorage):
    def __init__(self):
        super().__init__()
        self.on_get = None

    def get(self, reference, maximum=24 * 1024**2):
        value = super().get(reference, maximum)
        if self.on_get:
            self.on_get(reference)
        return value


@pytest.fixture
def api():
    db, storage = MemoryDocuments(), StubStorage()
    users = {uid: TeamIdentity(uid, f'{uid}@example.com', uid, uid != 'unverified')
             for uid in ('owner', 'editor', 'viewer', 'outsider', 'unverified')}
    store = TeamStore(db, owner_email=users['owner'].email, public_origin=ORIGIN)
    store.me(users['owner'])
    pid = store.create_project(users['owner'], 'Shared project')['id']
    other = store.create_project(users['owner'], 'Other project')['id']
    for uid, role in [('editor', 'editor'), ('viewer', 'viewer')]:
        invitation = store.invite(pid, users['owner'], users[uid].email, role)
        store.accept(invitation['id'], users[uid])
    revoked = set()
    token_claims = {}
    def verify(token):
        if token in token_claims and token not in revoked:
            return token_claims[token]
        if token not in users or token in revoked:
            raise ValueError('invalid')
        return claims(token, token != 'unverified')
    app = create_team_app(store=store, storage=storage, auth=TeamAuth(FIREBASE_PROJECT, verifier=verify),
                          public_origin=ORIGIN, firebase_config=FIREBASE_CONFIG)
    with TestClient(app, base_url=ORIGIN) as client:
        yield SimpleNamespace(client=client, store=store, storage=storage,
                              users=users, pid=pid, other=other, revoked=revoked, token_claims=token_claims,
                              prefix=f'/api/projects/{pid}/workspace')


def header(uid):
    return {'Authorization': f'Bearer {uid}'}


def update_run(api, run_id, **changes):
    """Edit a persisted synthetic run document; the service never writes runs."""
    path = f'oe_projects/{api.pid}/runs/{run_id}'
    run = api.store.db.get(path)
    run.update(changes)
    api.store.db.put(path, run)
    return run


def legacy_cloud_run(api, *, state='running', active=True):
    """A run accepted by the retired cloud backend that never reached a terminal state."""
    run_id = 'c' * 32
    run = {'id': run_id, 'project_id': api.pid, 'uid': 'editor', 'email': 'editor@example.com',
           'code': 'print("legacy")', 'timeout_seconds': 60, 'state': state, 'generation': 1,
           'created_at': '2026-10-02T10:00:00+00:00', 'deadline': '2026-10-02T10:18:00+00:00',
           'files': [], 'operation': 'sandbox:' + api.pid + ':' + run_id,
           'execution': 'sandbox:' + api.pid + ':' + run_id, 'result': None, 'cancel_requested': False}
    api.store.db.put(f'oe_projects/{api.pid}/runs/{run_id}', run)
    project = api.store.db.get(f'oe_projects/{api.pid}')
    project.update(run_count=1, active_run=run_id if active else None)
    api.store.db.put(f'oe_projects/{api.pid}', project)
    return run


@pytest.mark.parametrize('suffix', ['/console/execute', '/console/interrupt', '/console/reset'])
@pytest.mark.parametrize('uid', ['owner', 'editor', 'viewer'])
@pytest.mark.parametrize('body', [
    None, {'code': '1 + 2'}, {'code': '1', 'wait_for_result': False},
    {'code': '1', 'script_id': 'analysis', 'timeout_seconds': 30}, {'unexpected': True},
])
def test_retired_cloud_execution_routes_refuse_old_clients_clearly(api, suffix, uid, body):
    before = deepcopy(api.store.db.data)
    response = api.client.post(api.prefix + suffix, headers=header(uid), json=body)
    assert response.status_code == 410, response.text
    detail = response.json()['detail']
    assert detail['code'] == 'CLOUD_EXECUTION_RETIRED'
    assert 'desktop app' in detail['message']
    assert response.headers['cache-control'] == 'no-store'
    # Nothing is reserved, recorded, staged or launched for the request.
    assert api.store.db.data == before and api.storage.objects == {}


@pytest.mark.parametrize(('uid', 'status'), [('outsider', 404), ('unverified', 403)])
def test_retired_cloud_execution_keeps_membership_checks_before_refusal(api, uid, status):
    response = api.client.post(api.prefix + '/console/execute', headers=header(uid), json={'code': '1'})
    assert response.status_code == status
    assert api.client.post(api.prefix + '/console/execute', json={'code': '1'}).status_code == 401
    api.revoked.add('editor')
    assert api.client.post(api.prefix + '/console/execute', headers=header('editor'),
                           json={'code': '1'}).status_code == 401


def test_no_route_can_start_cloud_execution(api):
    from fastapi.routing import APIRoute
    import inspect
    from openecon import team_server
    assert 'runner' not in inspect.signature(create_team_app).parameters
    routes = {(method, route.path): route for route in api.client.app.routes
              if isinstance(route, APIRoute) for method in route.methods}
    for suffix in ('execute', 'interrupt', 'reset'):
        route = routes[('POST', '/api/projects/{project_id}/workspace/console/' + suffix)]
        assert route.endpoint.__name__ == 'cloud_execution_retired'
    assert not hasattr(team_server, 'ExecuteBody')
    assert not any(name in api.store.__class__.__dict__ for name in
                   ('begin_run', 'finish_run', 'request_cancel', 'claim_run_dispatch', 'attach_run_execution'))


def test_legacy_unfinished_cloud_run_is_read_as_stopped_without_changing_saved_data(api):
    run = legacy_cloud_run(api)
    before = deepcopy(api.store.db.data)
    boot = api.client.get(api.prefix + '/bootstrap', headers=header('editor')).json()
    assert boot['status'] == {'running': False, 'session_generation': 1, 'pid': None}
    state = api.client.get(api.prefix + '/console', headers=header('viewer')).json()
    assert state['status']['running'] is False
    record = state['history'][-1]
    assert record['id'] == run['id'] and record['code'] == run['code']
    assert record['status'] == 'interrupted' and 'desktop app' in record['error']['message']
    saved = api.client.get(api.prefix + f'/runs/{run["id"]}/record', headers=header('viewer'))
    assert saved.status_code == 200 and saved.json() == record
    page = api.client.get(api.prefix + '/console/history', headers=header('viewer')).json()
    assert page['runs'][0]['id'] == run['id'] and page['runs'][0]['state'] == 'running'
    # Reads tolerate the record; the stored lock and run document are preserved.
    assert api.store.db.data == before
    assert api.store.db.get(f'oe_projects/{api.pid}')['active_run'] == run['id']


def test_legacy_cloud_result_and_generated_file_remain_readable(api):
    from openecon.team_server import json_bytes
    run = legacy_cloud_run(api, state='finished', active=False)
    record = {'id': run['id'], 'code': run['code'], 'created_at': run['created_at'], 'status': 'ok',
              'stdout': '3\n', 'outputs': [{'type': 'text', 'data': '3'}], 'variables': [], 'error': None,
              'duration_ms': 12, 'session_generation': 1, 'active_session_generation': 1,
              'state_reset': True, 'actor_email': 'editor@example.com',
              'artifacts': [{'name': 'answer.txt', 'size_bytes': 2,
                             'url': api.prefix + f'/runs/{run["id"]}/files/0'}]}
    reference = api.storage.put(f'projects/{api.pid}/results/{run["id"]}/p/record.json', json_bytes(record))
    artifact = api.storage.put(f'projects/{api.pid}/results/{run["id"]}/p/answer.txt', b'42')
    update_run(api, run['id'], result=reference, record_summary={'status': 'ok'},
               artifacts=[{'name': 'answer.txt', 'size_bytes': 2, 'blob': artifact}])
    history = api.client.get(api.prefix + '/console', headers=header('viewer')).json()['history']
    assert history[-1]['id'] == run['id'] and history[-1]['outputs'][0]['data'] == '3'
    assert '3' in history[-1]['outputs'][0]['latex']
    download = api.client.get(record['artifacts'][0]['url'], headers=header('viewer'))
    assert download.status_code == 200 and download.content == b'42'
    assert api.client.get(record['artifacts'][0]['url'], headers=header('outsider')).status_code == 404


@pytest.mark.parametrize('suffix', ['/api/me', '/api/projects', '/api/projects/id/members',
    '/api/projects/id/workspace/bootstrap',
    '/api/projects/id/workspace/session', '/api/projects/id/workspace/console',
    '/api/projects/id/workspace/environment',
    '/api/projects/id/workspace/datasets', '/api/projects/id/workspace/files/f/download'])
def test_anonymous_api_reads_are_denied(api, suffix):
    assert api.client.get(suffix).status_code == 401


def test_auth_configuration_only_is_public_and_tokens_are_not_in_cache(api):
    response = api.client.get('/api/auth/config')
    assert response.status_code == 200
    assert response.json()['mode'] == 'teams'
    assert response.json()['cloud_execution_available'] is False
    assert response.headers['cache-control'] == 'no-store'
    assert api.client.get('/api/me', headers=header('unverified')).status_code == 200
    assert api.client.get('/api/projects', headers=header('unverified')).status_code == 403
    api.revoked.add('editor')
    assert api.client.get('/api/projects', headers=header('editor')).status_code == 401


@pytest.mark.parametrize('path', ['/', '/api/auth/config', '/api/me'])
def test_google_popup_headers_only_allow_required_trusted_sources(api, path):
    response = api.client.get(path)
    policy = {parts[0]: parts[1:] for entry in response.headers['content-security-policy'].split(';')
              if (parts := entry.split())}
    assert policy['script-src'] == ["'self'", 'https://apis.google.com']
    assert policy['frame-src'] == [f'https://{FIREBASE_PROJECT}.firebaseapp.com']
    assert policy['connect-src'] == ["'self'", 'https://identitytoolkit.googleapis.com',
                                     'https://securetoken.googleapis.com']
    assert policy['frame-ancestors'] == ["'none'"]
    assert policy['object-src'] == ["'none'"]
    assert policy['form-action'] == ["'self'"]
    assert "'unsafe-inline'" not in policy['script-src']
    assert "'unsafe-eval'" not in policy['script-src']
    assert '*' not in response.headers['content-security-policy']
    assert response.headers['cross-origin-opener-policy'] == 'same-origin-allow-popups'
    assert response.headers['x-frame-options'] == 'DENY'


@pytest.mark.parametrize('domain', [None, '', 'foreign-project.firebaseapp.com',
    'openecon-test.firebaseapp.com.evil.example', 'https://openecon-test.firebaseapp.com',
    'openecon-test.firebaseapp.com:443', 'openecon-test.firebaseapp.com/__/auth/iframe',
    'openecon-test.firebaseapp.com?x=1', 'openecon-test.firebaseapp.com#fragment',
    'openecon-test.firebaseapp.com@evil.example', 'openecon-test.firebaseapp.com; script-src *',
    'openecon-test.firebaseapp.com\r\nX-Injected: true', '*.firebaseapp.com',
    'openecon-test.web.app', 'openecon-test.firebaseapp.com.', ['openecon-test.firebaseapp.com']])
def test_invalid_auth_domain_fails_closed_at_startup(domain):
    with pytest.raises(ValueError, match='authDomain'):
        create_team_app(store=None, storage=None, auth=TeamAuth(FIREBASE_PROJECT),
                        public_origin=ORIGIN, firebase_config={**FIREBASE_CONFIG, 'authDomain': domain})


def test_foreign_firebase_project_cannot_set_csp_even_with_valid_domain():
    with pytest.raises(ValueError, match='authDomain'):
        create_team_app(store=None, storage=None, auth=TeamAuth(FIREBASE_PROJECT),
                        public_origin=ORIGIN, firebase_config={**FIREBASE_CONFIG, 'projectId': 'foreign-project'})


def test_auth_configuration_is_snapshotted_with_validated_domain():
    config = dict(FIREBASE_CONFIG)
    app = create_team_app(store=None, storage=None, auth=TeamAuth(FIREBASE_PROJECT),
                          public_origin=ORIGIN, firebase_config=config)
    config['authDomain'] = 'foreign.firebaseapp.com'
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.get('/api/auth/config').json()['firebase']['authDomain'] == FIREBASE_CONFIG['authDomain']


def test_google_login_retains_same_uid_membership_and_revocation(api):
    api.token_claims['google-editor'] = {**claims('editor'), 'firebase': {'sign_in_provider': 'google.com'}}
    response = api.client.get('/api/me', headers=header('google-editor'))
    assert response.status_code == 200
    assert response.json()['user']['uid'] == 'editor'
    assert api.client.get(api.prefix + '/session', headers=header('google-editor')).status_code == 200
    api.revoked.add('google-editor')
    assert api.client.get(api.prefix + '/session', headers=header('google-editor')).status_code == 401
    api.revoked.remove('google-editor')
    assert api.client.delete(f'/api/projects/{api.pid}/members/editor', headers=header('owner')).status_code == 200
    assert api.client.get(api.prefix + '/session', headers=header('google-editor')).status_code == 404


def test_google_same_email_different_uid_cannot_inherit_existing_membership(api):
    api.token_claims['google-impostor'] = {
        **claims('different-uid'), 'email': 'editor@example.com',
        'firebase': {'sign_in_provider': 'google.com'}}
    assert api.client.get('/api/me', headers=header('google-impostor')).status_code == 200
    assert api.client.get(api.prefix + '/session', headers=header('google-impostor')).status_code == 404


def test_google_unverified_email_is_not_assumed_verified(api):
    api.token_claims['google-unverified'] = {
        **claims('editor', verified=False), 'firebase': {'sign_in_provider': 'google.com'}}
    assert api.client.get('/api/me', headers=header('google-unverified')).status_code == 200
    assert api.client.get(api.prefix + '/session', headers=header('google-unverified')).status_code == 403


@pytest.mark.parametrize('method,suffix,body', [
    ('put', '/console/script', {'code': '1', 'version': 0}),
    ('put', '/environment', {'manifest': package_manifest(), 'version': 0}),
    ('post', '/datasets/example', None),
])
def test_viewer_cannot_mutate_workspace(api, method, suffix, body):
    response = api.client.request(method, api.prefix + suffix, json=body, headers=header('viewer'))
    assert response.status_code == 403


def test_viewer_cannot_upload_invite_promote_or_remove(api):
    response = api.client.post(api.prefix + '/datasets/upload', headers=header('viewer'),
                               files={'file': ('data.csv', b'x,y\n1,2\n', 'text/csv')})
    assert response.status_code == 403
    assert api.client.post(f'/api/projects/{api.pid}/invitations', headers=header('viewer'),
                           json={'email': 'new@example.com', 'role': 'editor'}).status_code == 403
    for role in ('editor', 'viewer'):
        assert api.client.patch(f'/api/projects/{api.pid}/members/{role}', headers=header('editor'),
                                json={'role': 'owner'}).status_code == 403
    assert api.client.delete(f'/api/projects/{api.pid}/members/editor', headers=header('viewer')).status_code == 403


def test_foreign_project_and_file_ids_do_not_cross_tenant_boundary(api):
    uploaded = api.client.post(f'/api/projects/{api.other}/workspace/datasets/upload', headers=header('owner'),
                              files={'file': ('private.csv', b'secret\n42\n', 'text/csv')}).json()
    for uid in ('viewer', 'outsider'):
        assert api.client.get(f'/api/projects/{api.other}/workspace/datasets', headers=header(uid)).status_code == 404
        assert api.client.get(f'/api/projects/{api.other}/workspace/files/{uploaded["id"]}/download',
                              headers=header(uid)).status_code == 404
    assert api.client.get(api.prefix + f'/files/{uploaded["id"]}/download', headers=header('viewer')).status_code == 404
    assert api.client.get(api.prefix + '/session', headers=header('viewer')).json()['read_only'] is True


def test_removed_member_loses_access_with_the_same_token(api):
    assert api.client.get(api.prefix + '/console/script', headers=header('editor')).status_code == 200
    assert api.client.delete(f'/api/projects/{api.pid}/members/editor', headers=header('owner')).status_code == 200
    for suffix in ('/session', '/bootstrap', '/console', '/console/script', '/environment', '/datasets'):
        assert api.client.get(api.prefix + suffix, headers=header('editor')).status_code == 404
    assert api.client.post(api.prefix + '/console/execute', headers=header('editor'), json={'code': '1'}).status_code == 404


@pytest.mark.parametrize(('uid', 'status'), [('unverified', 403), ('outsider', 404)])
def test_workspace_bootstrap_requires_verified_project_membership(api, uid, status):
    assert api.client.get(api.prefix + '/bootstrap', headers=header(uid)).status_code == status


def test_workspace_bootstrap_rechecks_token_revocation(api):
    assert api.client.get(api.prefix + '/bootstrap', headers=header('editor')).status_code == 200
    api.revoked.add('editor')
    assert api.client.get(api.prefix + '/bootstrap', headers=header('editor')).status_code == 401


@pytest.mark.parametrize(('uid', 'read_only'), [('owner', False), ('editor', False), ('viewer', True)])
def test_workspace_bootstrap_matches_scoped_content_without_reading_history(api, uid, read_only):
    draft = api.store.save_script(api.pid, api.users['editor'], 'print("saved draft")', 0)
    api.store.save_script(api.other, api.users['owner'], 'private other project', 0)
    uploaded = api.client.post(api.prefix + '/datasets/upload', headers=header('editor'),
                               files={'file': ('data.csv', b'a,b\n1,2', 'text/csv')}).json()
    expected_session = api.client.get(api.prefix + '/session', headers=header(uid)).json()
    expected_datasets = api.client.get(api.prefix + '/datasets', headers=header(uid)).json()['datasets']
    api.store.runs = Mock(side_effect=AssertionError('History must not block the editor.'))
    api.storage.get = Mock(side_effect=AssertionError('Bootstrap must not read result blobs.'))
    response = api.client.get(api.prefix + '/bootstrap', headers=header(uid))
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    data = response.json()
    assert data['session'] == expected_session
    assert data['session']['read_only'] is read_only
    assert data['draft'] == draft
    assert data['environment'] == {'manifest': None, 'version': 0}
    assert data['datasets'] == expected_datasets
    assert data['datasets'][0]['id'] == uploaded['id']
    assert 'blob' not in data['datasets'][0]
    assert data['status'] == {'running': False, 'session_generation': 0, 'pid': None}
    assert 'history' not in data and 'private other project' not in response.text


@pytest.mark.parametrize('role', [None, 'viewer'])
def test_workspace_bootstrap_rechecks_removal_or_downgrade_during_draft_read(api, role):
    original = api.store.script
    def racing_read(project_id, user):
        draft = original(project_id, user)
        api.store.change_member(project_id, user.uid, api.users['owner'], role)
        return draft
    api.store.script = racing_read
    response = api.client.get(api.prefix + '/bootstrap', headers=header('editor'))
    if role is None:
        assert response.status_code == 404
    else:
        assert response.status_code == 200
        assert response.json()['session']['read_only'] is True


def test_download_rechecks_membership_after_storage_read(api):
    item = api.client.post(api.prefix + '/datasets/upload', headers=header('editor'),
                           files={'file': ('x.csv', b'x\n1\n', 'text/csv')}).json()
    def remove(reference):
        api.storage.on_get = None
        api.store.change_member(api.pid, 'editor', api.users['owner'])
    api.storage.on_get = remove
    assert api.client.get(api.prefix + f'/files/{item["id"]}/download', headers=header('editor')).status_code == 404


def shared_result(api):
    from openecon.team_server import json_bytes
    run = legacy_cloud_run(api, state='finished', active=False)
    record = {'id': run['id'], 'code': run['code'], 'created_at': run['created_at'], 'status': 'ok',
              'stdout': '', 'outputs': [], 'variables': [], 'error': None, 'duration_ms': 1,
              'session_generation': 1}
    reference = api.storage.put(f'projects/{api.pid}/results/{run["id"]}/p/record.json', json_bytes(record))
    update_run(api, run['id'], result=reference, record_summary={'status': 'ok'})
    return run


def test_console_rechecks_membership_after_result_read(api):
    shared_result(api)
    def remove(reference):
        if reference['key'].endswith('/record.json'):
            api.storage.on_get = None
            api.store.change_member(api.pid, 'editor', api.users['owner'])
    api.storage.on_get = remove
    assert api.client.get(api.prefix + '/console', headers=header('editor')).status_code == 404


def test_cross_site_bearer_requests_and_unknown_fields_are_denied(api):
    for headers in ({'Origin': 'https://evil.example'}, {'Sec-Fetch-Site': 'cross-site'}):
        assert api.client.get('/api/projects', headers={**header('owner'), **headers}).status_code == 403
    assert api.client.put(api.prefix + '/console/script', headers=header('editor'),
                          json={'code': '1', 'version': 0, 'project_id': api.other}).status_code == 422


def valid_payload():
    run = {'id': 'run1', 'code': '1', 'created_at': '2026-10-01', 'generation': 1, 'email': 'actor@example.com'}
    payload = {'execution_id': 'run1', 'record': {'status': 'ok', 'outputs': [], 'stdout': ''}, 'generated_files': []}
    return run, payload


@pytest.mark.parametrize('data', [
    {'columns': ['x'], 'rows': [[1]], 'total_rows': 1, 'total_columns': 1, 'index_names': 'not-a-list'},
    {'columns': ['x'], 'rows': [[1]], 'total_rows': 1, 'total_columns': 1, 'index': ['not-a-row']},
])
def test_malformed_table_indexes_cannot_be_saved_for_other_members(data):
    run, payload = valid_payload()
    payload['record']['outputs'] = [{'type': 'table', 'data': data}]
    with pytest.raises(TeamError):
        validate_worker_result(payload, run)


def test_incomplete_model_cannot_be_saved_for_other_members():
    run, payload = valid_payload()
    payload['record']['outputs'] = [{'type': 'model', 'data': {
        'spec': {'outcome': 'y', 'predictors': ['x']}, 'coefficients': [], 'nobs': 1, 'metrics': {},
        'warnings': {'malformed': 'not a list'},
    }}]
    with pytest.raises(TeamError):
        validate_worker_result(payload, run)


@pytest.mark.parametrize('generated', [
    [{'name': '../x', 'content_base64': 'eA==', 'size': 1}],
    [{'name': 'x', 'content_base64': 'invalid-base64!', 'size': 1}],
    [{'name': 'x', 'content_base64': 'eA==', 'size': 2}],
    [{'name': 'x', 'content_base64': 'eA==', 'size': 1}] * 2,
])
def test_untrusted_generated_payloads_fail_closed(generated):
    run, payload = valid_payload()
    payload['generated_files'] = generated
    with pytest.raises(TeamError):
        validate_worker_result(payload, run)


def test_readback_storage_error_is_safe_and_preserves_saved_records(api, caplog):
    shared_result(api)
    before = deepcopy(api.store.db.data)
    api.storage.get = Mock(side_effect=RuntimeError('https://storage.googleapis.com/private?X-Goog-Signature=secret'))
    failed = api.client.get(api.prefix + '/console', headers=header('viewer'))
    assert failed.status_code == 503
    assert 'X-Goog-Signature' not in failed.text + caplog.text
    assert api.store.db.data == before


def test_chunked_json_body_is_bounded_before_parsing(api):
    def chunks():
        yield b'{"code":"'
        for _ in range(17):
            yield b'x' * 32768
        yield b'"}'
    before = deepcopy(api.store.db.data)
    request = api.client.build_request('PUT', api.prefix + '/console/script',
        headers={**header('editor'), 'Content-Type': 'application/json'}, content=chunks())
    assert 'content-length' not in request.headers
    assert api.client.send(request).status_code == 413
    assert api.store.db.data == before


def test_chunked_multipart_body_is_bounded_before_storing(api, monkeypatch):
    from openecon import team_server
    monkeypatch.setattr(team_server, 'MAX_TRANSFER_BYTES', 8192)
    def chunks():
        yield (b'--test-boundary\r\nContent-Disposition: form-data; name="file"; filename="x.csv"\r\n'
               b'Content-Type: text/csv\r\n\r\n')
        for _ in range(20):
            yield b'x' * 4096
        yield b'\r\n--test-boundary--\r\n'
    request = api.client.build_request('POST', api.prefix + '/datasets/upload',
        headers={**header('editor'), 'Content-Type': 'multipart/form-data; boundary=test-boundary'}, content=chunks())
    assert 'content-length' not in request.headers
    assert api.client.send(request).status_code == 413
    assert api.storage.objects == {}


def test_owner_can_rename_project_and_all_member_summaries_read_the_same_version(api):
    pending = api.store.invite(api.pid, api.users['owner'], 'pending@example.com', 'viewer')
    api.store.save_script(api.pid, api.users['editor'], 'unsaved = (\n    42', 0)
    before = deepcopy(api.store.db.data)
    response = api.client.patch(f'/api/projects/{api.pid}', headers=header('owner'),
                                json={'name': '  Türkiye Araştırması 👩‍💻  ', 'name_version': 0})
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved['id'] == api.pid and saved['name'] == 'Türkiye Araştırması 👩‍💻'
    assert saved['name_version'] == 1 and saved['role'] == 'owner'
    assert response.headers['cache-control'] == 'no-store'
    for uid in ('owner', 'editor', 'viewer'):
        listing = api.client.get('/api/projects', headers=header(uid)).json()['projects']
        project = next(value for value in listing if value['id'] == api.pid)
        assert project['name'] == saved['name'] and project['name_version'] == 1
        profile = api.client.get('/api/me', headers=header(uid)).json()
        assert next(value for value in profile['projects'] if value['id'] == api.pid)['name_version'] == 1
    invitation = api.client.get(f'/api/projects/{api.pid}/members', headers=header('owner')).json()['invitations'][0]
    assert invitation['project_name'] == saved['name']
    assert {key: value for key, value in invitation.items() if key != 'project_name'} == {
        key: value for key, value in pending.items() if key != 'project_name'}
    assert api.store.db.get(f'oe_projects/{api.other}') == before[f'oe_projects/{api.other}']
    for path, value in before.items():
        if path.startswith(f'oe_projects/{api.pid}/workspace/') or path.startswith('oe_users/'):
            assert api.store.db.get(path) == value
    assert api.store.script(api.pid, api.users['viewer'])['code'] == 'unsaved = (\n    42'
    assert api.storage.objects == {}


def test_project_rename_http_legacy_summary_defaults_to_zero_without_a_migration(api):
    path = f'oe_projects/{api.pid}'
    legacy = api.store.db.get(path)
    legacy.pop('name_version')
    api.store.db.put(path, legacy)
    listed = api.client.get('/api/projects', headers=header('owner')).json()['projects']
    assert next(project for project in listed if project['id'] == api.pid)['name_version'] == 0
    response = api.client.patch(f'/api/projects/{api.pid}', headers=header('owner'),
                                json={'name': 'Legacy renamed', 'name_version': 0})
    assert response.status_code == 200 and response.json()['name_version'] == 1


@pytest.mark.parametrize('uid,status', [('editor', 403), ('viewer', 403), ('outsider', 404),
                                      ('unverified', 403), (None, 401)])
def test_project_rename_http_requires_verified_owner(api, uid, status):
    before = deepcopy(api.store.db.data)
    response = api.client.patch(f'/api/projects/{api.pid}', headers=header(uid) if uid else {},
                                json={'name': 'Forbidden', 'name_version': 0})
    assert response.status_code == status, response.text
    assert api.store.db.data == before
    assert api.storage.objects == {}


@pytest.mark.parametrize('uid', ['owner', 'editor'])
def test_project_rename_http_revoked_authentication_cannot_mutate(api, uid):
    api.revoked.add(uid)
    before = deepcopy(api.store.db.data)
    response = api.client.patch(f'/api/projects/{api.pid}', headers=header(uid),
                                json={'name': 'Revoked', 'name_version': 0})
    assert response.status_code == 401
    assert api.store.db.data == before


@pytest.mark.parametrize('role', [None, 'editor', 'viewer'])
def test_project_rename_http_rechecks_owner_after_authentication(api, role):
    original = api.store.rename_project
    def changed_before_transaction(project_id, user, name, name_version):
        path = f'oe_projects/{project_id}'
        project = api.store.db.get(path)
        if role is None:
            project['members'].pop(user.uid)
        else:
            project['members'][user.uid]['role'] = role
        api.store.db.put(path, project)
        return original(project_id, user, name, name_version)
    api.store.rename_project = changed_before_transaction
    response = api.client.patch(f'/api/projects/{api.pid}', headers=header('owner'),
                                json={'name': 'Race denied', 'name_version': 0})
    assert response.status_code == (404 if role is None else 403)
    assert api.store.db.get(f'oe_projects/{api.pid}')['name'] == 'Shared project'
    assert not any(event['action'] == 'project.renamed' for event in api.store.db.scan(f'oe_projects/{api.pid}/audit'))


def test_project_rename_http_cas_rejects_stale_same_name_and_idempotent_current_name(api):
    path = f'/api/projects/{api.pid}'
    first = api.client.patch(path, headers=header('owner'), json={'name': 'New project', 'name_version': 0})
    assert first.status_code == 200
    before = deepcopy(api.store.db.data)
    stale = api.client.patch(path, headers=header('owner'), json={'name': 'New project', 'name_version': 0})
    assert stale.status_code == 409 and stale.json()['detail']['code'] == 'VERSION_CONFLICT'
    assert api.store.db.data == before
    same = api.client.patch(path, headers=header('owner'), json={'name': ' New project ', 'name_version': 1})
    assert same.status_code == 200 and same.json() == first.json()
    assert api.store.db.data == before


@pytest.mark.parametrize('body', [
    {}, {'name': 'New'}, {'name_version': 0}, {'name': None, 'name_version': 0},
    {'name': True, 'name_version': 0}, {'name': 1, 'name_version': 0},
    {'name': [], 'name_version': 0}, {'name': {}, 'name_version': 0},
    {'name': '', 'name_version': 0}, {'name': '   ', 'name_version': 0},
    {'name': 'a' * 101, 'name_version': 0}, {'name': 'line\n', 'name_version': 0},
    {'name': '\ttab', 'name_version': 0}, {'name': 'name\x00', 'name_version': 0},
    {'name': 'name\x85', 'name_version': 0}, {'name': 'name\u2028', 'name_version': 0},
    {'name': 'name\u2029', 'name_version': 0},
    {'name': 'New', 'name_version': True}, {'name': 'New', 'name_version': False},
    {'name': 'New', 'name_version': -1}, {'name': 'New', 'name_version': '0'},
    {'name': 'New', 'name_version': 0.0}, {'name': 'New', 'name_version': None},
    {'name': 'New', 'name_version': 2**53},
    {'name': 'New', 'name_version': 0, 'id': 'other'},
    {'name': 'New', 'name_version': 0, 'owner_uid': 'outsider'},
    {'name': 'New', 'name_version': 0, 'members': {}},
    {'name': 'New', 'name_version': 0, 'description': 'Changed'},
    {'name': 'New', 'name_version': 0, 'files': []},
    {'name': 'New', 'name_version': 0, 'active_run': 'forged'},
])
def test_project_rename_http_rejects_invalid_or_mass_assignment_bodies(api, body):
    before = deepcopy(api.store.db.data)
    response = api.client.patch(f'/api/projects/{api.pid}', headers=header('owner'), json=body)
    assert response.status_code == 422, response.text
    assert api.store.db.data == before
    assert api.storage.objects == {}


def test_project_rename_http_unicode_limits_apply_after_trim(api):
    name = '😀' * 100
    saved = api.client.patch(f'/api/projects/{api.pid}', headers=header('owner'),
                             json={'name': '  ' + name + '  ', 'name_version': 0})
    assert saved.status_code == 200 and saved.json()['name'] == name
    rejected = api.client.patch(f'/api/projects/{api.pid}', headers=header('owner'),
                                json={'name': name + '😀', 'name_version': 1})
    assert rejected.status_code == 422
    assert api.store.project(api.pid, api.users['owner'])['name'] == name


@pytest.mark.parametrize('headers', [{'Origin': 'https://foreign.example.com'}, {'Sec-Fetch-Site': 'cross-site'}])
def test_project_rename_http_preserves_origin_guards(api, headers):
    before = deepcopy(api.store.db.data)
    response = api.client.patch(f'/api/projects/{api.pid}', headers={**header('owner'), **headers},
                                json={'name': 'Forbidden', 'name_version': 0})
    assert response.status_code == 403 and response.json()['detail']['code'] == 'ORIGIN_DENIED'
    assert api.store.db.data == before


def seed_history(api, count=137, *, tied=False, code=None):
    """Persisted synthetic rows, with no worker or user project involved."""
    for index in range(count):
        run_id = f'{index:032x}'
        api.store.db.put(f'oe_projects/{api.pid}/runs/{run_id}', {
            'id': run_id, 'created_at': f'2026-10-01T00:{0 if tied else index // 60:02d}:{0 if tied else index % 60:02d}+00:00',
            'code': code or f'print("legacy_marker_{index}")', 'state': 'finished',
            'record_summary': {'status': 'ok'}, 'result': None, 'generation': index + 1,
            'email': 'editor@example.com', 'files': [{'private': 'not returned'}],
        })
    return [f'{index:032x}' for index in reversed(range(count))]


def history_request(api, **params):
    return api.client.get(api.prefix + '/console/history', headers=header('viewer'), params=params)


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('limit,max_bytes', [(1, 4096), (7, 4096), (50, 4096), (50, 262144)])
def test_history_pages_find_old_runs_preserve_budgets_and_survive_new_insertions(api, tied, limit, max_bytes):
    expected = seed_history(api, tied=tied, code='🌍' * 64000)
    cursor, found, requests = None, [], []
    original_scan = api.store.db.history_scan
    def bounded_scan(*args, **kwargs):
        requests.append(kwargs)
        return original_scan(*args, **kwargs)
    api.store.db.history_scan = bounded_scan
    while True:
        response = history_request(api, limit=limit, max_bytes=max_bytes, **({'cursor': cursor} if cursor else {}))
        assert response.status_code == 200, response.text
        assert len(response.content) <= max_bytes
        page = response.json()
        assert len(page['runs']) <= limit and page['scanned'] <= 100
        found.extend(run['id'] for run in page['runs'])
        if cursor is None:
            new_id = 'f' * 32
            api.store.db.put(f'oe_projects/{api.pid}/runs/{new_id}', {
                **api.store.db.get(f'oe_projects/{api.pid}/runs/{expected[0]}'),
                'id': new_id, 'created_at': now(),
            })
        cursor = page['next_cursor']
        if cursor is None:
            break
        assert len(found) <= len(expected)
    assert found == expected and len(set(found)) == 137
    assert history_request(api).json()['runs'][0]['id'] == new_id
    assert all(request['limit'] == 101 for request in requests)
    assert api.storage.objects == {}
    assert 'not returned' not in response.text and 'result' not in page['runs'][-1]


@pytest.mark.parametrize('params,expected', [
    ({'query': 'LEGACY_MARKER_0"'}, [f'{0:032x}']),
    ({'query': f'{1:032x}'}, [f'{1:032x}']),
    ({'since': '2026-10-01', 'until': '2026-10-01'}, [f'{i:032x}' for i in reversed(range(137))]),
    ({'since': '2026-10-02'}, []),
    ({'until': '2026-09-30'}, []),
])
def test_history_search_id_code_and_utc_dates_with_bounded_empty_pages(api, params, expected):
    seed_history(api)
    found, cursor, pages = [], None, 0
    while True:
        response = history_request(api, **params, **({'cursor': cursor} if cursor else {}))
        assert response.status_code == 200, response.text
        page = response.json()
        found.extend(row['id'] for row in page['runs'])
        cursor = page['next_cursor']
        pages += 1
        assert pages < 10 and page['scanned'] <= 100
        if not cursor:
            break
    assert found == expected
    if params.get('query') == 'LEGACY_MARKER_0"':
        assert pages == 2  # First search batch has no match, but a cursor advances.


@pytest.mark.parametrize('params', [
    {'limit': 0}, {'limit': 51}, {'max_bytes': 4095}, {'max_bytes': 262145},
    {'cursor': ''}, {'cursor': 'not-a-cursor'}, {'cursor': 'a' * 2049},
    {'query': '\x00'}, {'query': 'a' * 257}, {'since': '2026-02-30'},
    {'since': '2026-10-02', 'until': '2026-10-01'}, {'until': 'tomorrow'},
])
def test_invalid_history_requests_fail_before_repository_scan(api, params):
    api.store.db.history_scan = Mock(side_effect=AssertionError('must not scan'))
    response = history_request(api, **params)
    assert response.status_code == 422, response.text
    assert not api.store.db.history_scan.called


def test_history_cursor_is_scoped_to_query_and_project_and_rechecks_membership(api):
    seed_history(api)
    page = history_request(api, limit=1).json()
    cursor = page['next_cursor']
    assert history_request(api, cursor=cursor, query='different').status_code == 422
    other = api.client.get(f'/api/projects/{api.other}/workspace/console/history',
                           headers=header('owner'), params={'cursor': cursor})
    assert other.status_code == 422
    api.store.change_member(api.pid, 'viewer', api.users['owner'], None)
    assert history_request(api, cursor=cursor).status_code == 404
    assert api.client.get(api.prefix + f'/runs/{page["runs"][0]["id"]}/record', headers=header('viewer')).status_code == 404


def test_history_membership_is_checked_after_metadata_and_blob_read(api):
    seed_history(api, 1)
    original_scan = api.store.db.history_scan
    def revoke_after_scan(*args, **kwargs):
        rows = original_scan(*args, **kwargs)
        api.store.change_member(api.pid, 'viewer', api.users['owner'], None)
        return rows
    api.store.db.history_scan = revoke_after_scan
    assert history_request(api).status_code == 404


def test_history_opens_full_saved_record_without_reevaluating_or_truncating(api):
    from openecon.team_server import json_bytes
    seed_history(api, 1, code='print("🌍")\n' * 1000)
    run_id = f'{0:032x}'
    run = api.store.run(api.pid, run_id)
    record = {'id': run_id, 'code': run['code'], 'created_at': run['created_at'], 'status': 'ok',
              'stdout': 'original', 'outputs': [], 'variables': [], 'error': None,
              'duration_ms': 123, 'session_generation': 1}
    reference = api.storage.put('synthetic/old-record.json', json_bytes(record))
    update_run(api, run_id, result=reference)
    reads = []
    api.storage.on_get = lambda reference: reads.append(reference)
    page = history_request(api).json()
    assert not reads
    assert len(page['runs'][0]['code_preview']) == 240 and page['runs'][0]['code_preview_truncated']
    response = api.client.get(api.prefix + f'/runs/{run_id}/record', headers=header('viewer'))
    assert response.status_code == 200 and response.json() == record
    assert len(reads) == 1
    api.storage.on_get = lambda _: api.store.change_member(api.pid, 'viewer', api.users['owner'], None)
    assert api.client.get(api.prefix + f'/runs/{run_id}/record', headers=header('viewer')).status_code == 404


@pytest.mark.parametrize('uid,status', [('outsider', 404), ('unverified', 403)])
def test_history_requires_verified_membership(api, uid, status):
    assert api.client.get(api.prefix + '/console/history', headers=header(uid)).status_code == status
    assert api.client.get(api.prefix + '/runs/unknown/record', headers=header(uid)).status_code == status


def test_history_record_missing_unfinished_and_failed_states_are_explicit(api):
    seed_history(api, 1)
    run_id = f'{0:032x}'
    path = api.prefix + f'/runs/{run_id}/record'
    assert api.client.get(api.prefix + '/runs/unknown/record', headers=header('viewer')).status_code == 404
    update_run(api, run_id, state='running')
    unfinished = api.client.get(path, headers=header('viewer'))
    assert unfinished.status_code == 200 and unfinished.json()['status'] == 'interrupted'
    assert api.store.run(api.pid, run_id)['state'] == 'running'
    update_run(api, run_id, state='failed', record_summary={'status': 'timeout', 'message': 'original timeout'})
    result = api.client.get(path, headers=header('viewer'))
    assert result.status_code == 200 and result.json()['status'] == 'timeout'
    assert result.json()['error']['message'] == 'original timeout'


def test_history_snapshot_cursor_replays_first_page_after_new_runs(api):
    from openecon.team_store import now
    seed_history(api, 60)
    original = history_request(api, limit=7).json()
    new_id = 'f' * 32
    api.store.db.put(f'oe_projects/{api.pid}/runs/{new_id}', {
        **api.store.run(api.pid, f'{59:032x}'), 'id': new_id, 'created_at': now(),
    })
    replay = history_request(api, limit=7, cursor=original['page_cursor']).json()
    assert replay == original
    assert history_request(api, limit=7).json()['runs'][0]['id'] == new_id


def test_unicode_search_and_maximum_metadata_fit_smallest_history_page(api):
    seed_history(api, 2, code='🌍' * 64000)
    for index in range(2):
        update_run(api, f'{index:032x}', email='🧪' * 256)
    first = history_request(api, query='🌍' * 256, max_bytes=4096).json()
    assert first['runs'] and first['next_cursor']
    next_page = history_request(api, query='🌍' * 256, max_bytes=4096, cursor=first['next_cursor']).json()
    assert next_page['runs'] and next_page['runs'][0]['id'] != first['runs'][0]['id']
