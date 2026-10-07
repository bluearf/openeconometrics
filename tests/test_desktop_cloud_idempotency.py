"""Desktop result retries archive inert data once, preserving access checks."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Barrier

import pytest

from test_desktop_cloud import api as api, headers, record


def path(api):
    return f'/api/projects/{api.pid}/workspace/desktop/results'


def body():
    return {'record': record('editor'), 'input_files': []}


def stored_run(api, identifier):
    return api.store.db.get(f'oe_projects/{api.pid}/runs/{identifier}')


def seed_input(api):
    file = {'id': '1' * 32, 'name': 'wages.csv', 'data_hash': 'a' * 64}
    project_path = f'oe_projects/{api.pid}'
    project = api.store.db.get(project_path)
    project['files'] = [deepcopy(file)]
    api.store.db.put(project_path, project)
    return file


def change_project(api, change):
    project_path = f'oe_projects/{api.pid}'
    def update(db):
        project = db.get(project_path)
        change(project)
        db.put(project_path, project)
    api.store.db.atomic(update)


@pytest.mark.parametrize('change', ['replaced', 'deleted'])
def test_acknowledgement_retry_survives_changed_input_without_rewriting_archive(api, change):
    file = seed_input(api)
    payload = body()
    payload['input_files'] = [{'id': file['id'], 'data_hash': file['data_hash']}]
    first = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert first.status_code == 201
    run = stored_run(api, first.json()['id'])
    archived = api.storage.get(run['result'])
    if change == 'replaced':
        change_project(api, lambda project: project['files'][0].update(data_hash='b' * 64))
    else:
        change_project(api, lambda project: project.update(files=[]))
    repeated = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert repeated.status_code == 201 and repeated.json() == first.json()
    assert api.storage.get(run['result']) == archived
    history = api.client.get(f'/api/projects/{api.pid}/workspace/console',
                             headers=headers('viewer')).json()['history']
    assert len(history) == 1 and history[0]['input_files'] == [file]
    api.runner.start.assert_not_called()


def concurrent_posts(api, monkeypatch, payloads):
    ready = Barrier(2)
    put = api.storage.put
    def overlap(key, payload, content_type='application/octet-stream'):
        reference = put(key, payload, content_type)
        ready.wait(timeout=10)
        return reference
    monkeypatch.setattr(api.storage, 'put', overlap)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(api.client.post, path(api), headers=headers('editor'),
                               json=payload) for payload in payloads]
        return [future.result(timeout=20) for future in futures]


def test_concurrent_identical_posts_share_one_generation_and_both_acknowledge(api, monkeypatch):
    responses = concurrent_posts(api, monkeypatch, [body(), body()])
    assert [response.status_code for response in responses] == [201, 201]
    assert responses[0].json() == responses[1].json()
    run = stored_run(api, responses[0].json()['id'])
    assert set(api.storage.objects) == {run['result']['key']}
    archived = json.loads(api.storage.get(run['result']))
    assert archived['stdout'] == '42' and archived['actor_email'] == 'editor@example.com'
    events = api.store.db.scan(f'oe_projects/{api.pid}/audit', limit=100)
    assert sum(event['action'] == 'desktop.result_shared' for event in events) == 1
    api.runner.start.assert_not_called()


def test_concurrent_different_bodies_conflict_without_deleting_winning_archive(api, monkeypatch):
    changed = body()
    changed['record']['stdout'] = 'different result'
    responses = concurrent_posts(api, monkeypatch, [body(), changed])
    assert sorted(response.status_code for response in responses) == [201, 409]
    winner = next(response for response in responses if response.status_code == 201)
    loser = next(response for response in responses if response.status_code == 409)
    assert loser.json()['detail']['code'] == 'RESULT_CONFLICT'
    run = stored_run(api, winner.json()['id'])
    assert set(api.storage.objects) == {run['result']['key']}
    assert json.loads(api.storage.get(run['result']))['stdout'] in {'42', 'different result'}
    api.runner.start.assert_not_called()


@pytest.mark.parametrize('field', ['stdout', 'input_files'])
def test_same_identifier_with_changed_envelope_is_not_an_acknowledgement(api, field):
    payload = body()
    first = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert first.status_code == 201
    if field == 'stdout':
        payload['record']['stdout'] = 'different result'
    else:
        payload['input_files'] = [{'id': '1' * 32, 'data_hash': 'a' * 64}]
    response = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert response.status_code == 409 and response.json()['detail']['code'] == 'RESULT_CONFLICT'
    run = stored_run(api, first.json()['id'])
    assert set(api.storage.objects) == {run['result']['key']}


@pytest.mark.parametrize('uid,status', [('owner', 403), ('viewer', 403), ('outsider', 404)])
def test_existing_result_acknowledgement_does_not_bypass_actor_or_membership(api, uid, status):
    payload = body()
    first = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert first.status_code == 201
    response = api.client.post(path(api), headers=headers(uid), json=payload)
    assert response.status_code == status
    run = stored_run(api, first.json()['id'])
    assert set(api.storage.objects) == {run['result']['key']}


@pytest.mark.parametrize('membership,status', [('viewer', 403), (None, 404)])
def test_existing_result_retry_rechecks_removed_or_downgraded_member(api, membership, status):
    payload = body()
    first = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert first.status_code == 201
    def modify(project):
        if membership is None:
            project['members'].pop('editor')
        else:
            project['members']['editor']['role'] = membership
    change_project(api, modify)
    response = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert response.status_code == status
    run = stored_run(api, first.json()['id'])
    assert set(api.storage.objects) == {run['result']['key']}


@pytest.mark.parametrize('change,status,code', [
    ('input', 409, 'DATA_INTEGRITY'), ('membership', 404, 'NOT_FOUND'),
])
def test_publication_rechecks_membership_and_inputs_after_upload_and_cleans_own_blob(
    api, monkeypatch, change, status, code,
):
    file = seed_input(api)
    payload = body()
    payload['input_files'] = [{'id': file['id'], 'data_hash': file['data_hash']}]
    put = api.storage.put
    def upload_then_change(key, payload, content_type='application/octet-stream'):
        reference = put(key, payload, content_type)
        if change == 'input':
            change_project(api, lambda project: project.update(files=[]))
        else:
            change_project(api, lambda project: project['members'].pop('editor'))
        return reference
    monkeypatch.setattr(api.storage, 'put', upload_then_change)
    response = api.client.post(path(api), headers=headers('editor'), json=payload)
    assert response.status_code == status and response.json()['detail']['code'] == code
    assert not api.storage.objects
    assert not api.store.db.scan(f'oe_projects/{api.pid}/runs')
    api.runner.start.assert_not_called()


@pytest.mark.parametrize('failed_confirmation_read', [False, True])
def test_lost_database_commit_acknowledgement_preserves_committed_blob_for_retry(
    api, monkeypatch, failed_confirmation_read,
):
    atomic = api.store.db.atomic
    get = api.store.db.get
    calls = 0
    committed_without_acknowledgement = False
    confirmation_failed = False
    def lose_acknowledgement(fn):
        nonlocal calls, committed_without_acknowledgement
        calls += 1
        result = atomic(fn)
        if calls == 2:
            committed_without_acknowledgement = True
            raise RuntimeError('Simulated lost database acknowledgement after commit')
        return result
    def interrupted_confirmation(path):
        nonlocal confirmation_failed
        if (failed_confirmation_read and committed_without_acknowledgement and not confirmation_failed
                and '/runs/' in path):
            confirmation_failed = True
            raise RuntimeError('Simulated unavailable commit confirmation')
        return get(path)
    monkeypatch.setattr(api.store.db, 'atomic', lose_acknowledgement)
    monkeypatch.setattr(api.store.db, 'get', interrupted_confirmation)
    first = api.client.post(path(api), headers=headers('editor'), json=body())
    assert first.status_code == 503 and first.json()['detail']['code'] == 'SERVICE_UNAVAILABLE'
    runs = api.store.db.scan(f'oe_projects/{api.pid}/runs')
    assert len(runs) == 1
    reference = runs[0]['result']
    assert api.storage.get(reference)
    repeated = api.client.post(path(api), headers=headers('editor'), json=body())
    assert repeated.status_code == 201 and repeated.json()['id'] == runs[0]['id']
    assert set(api.storage.objects) == {reference['key']}
    api.runner.start.assert_not_called()
