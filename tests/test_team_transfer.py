"""Private bounded transfer contracts; disk transport is an isolated test double."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from openecon.team_auth import TeamIdentity
from openecon.team_store import MemoryDocuments, TeamError, TeamStore
from openecon.team_transfer import (
    DatasetTransfers, MAX_DATA_BYTES, MAX_MANIFEST_BYTES, PART_BYTES, TransferBegin,
)


class DiskBlobs:
    """Physical files, fixed-size reads and generation pins; never a cloud proof."""
    def __init__(self, root):
        self.root = root
        self.refs = {}
        self.get_sizes = []
        self.put_sizes = []
        self.fail_delete = False
        self.after_get = None

    def put(self, key, payload, content_type='application/octet-stream'):
        assert key not in self.refs
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        ref = {'key': key, 'generation': len(self.refs) + 1, 'size': len(payload)}
        self.refs[key] = ref
        self.put_sizes.append(len(payload))
        return ref

    def get(self, ref, maximum=PART_BYTES):
        assert self.refs[ref['key']] == ref
        with (self.root / ref['key']).open('rb') as stream:
            data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise TeamError('RESULT_LIMIT', 'Too large.', 413)
        self.get_sizes.append(len(data))
        if self.after_get:
            action, self.after_get = self.after_get, None
            action()
        return data

    def delete(self, ref):
        if self.fail_delete:
            raise OSError('isolated delete failure')
        current = self.refs.get(ref['key'])
        if current is None:
            return
        assert current == ref
        (self.root / ref['key']).unlink()
        del self.refs[ref['key']]

    def find(self, key, maximum=PART_BYTES):
        ref = self.refs.get(key)
        if ref and ref['size'] > maximum:
            raise TeamError('RESULT_LIMIT', 'Too large.', 413)
        return deepcopy(ref)


class NoScanTransactionDocuments(MemoryDocuments):
    """Production Firestore transactions have no collection-scan method."""
    def atomic(self, fn):
        class Transaction:
            def get(_, path):
                return self.get(path)

            def put(_, path, value):
                return self.put(path, value)

            def delete(_, path):
                return self.delete(path)
        return super().atomic(lambda _: fn(Transaction()))


@pytest.fixture
def env(tmp_path):
    db = NoScanTransactionDocuments()
    users = {name: TeamIdentity(name, f'{name}@example.com', name, True)
             for name in ('owner', 'editor', 'editor2', 'viewer', 'outsider')}
    store = TeamStore(db, owner_email=users['owner'].email,
                      public_origin='https://synthetic.example.com')
    store.me(users['owner'])
    pid = store.create_project(users['owner'], 'Synthetic transfer')['id']
    other = store.create_project(users['owner'], 'Other')['id']
    for uid in ('editor', 'editor2', 'viewer'):
        invite = store.invite(pid, users['owner'], users[uid].email,
                              'viewer' if uid == 'viewer' else 'editor')
        store.accept(invite['id'], users[uid])
    storage = DiskBlobs(tmp_path / 'blobs')
    return SimpleNamespace(db=db, store=store, pid=pid, other=other, users=users,
                           storage=storage, transfers=DatasetTransfers(store, storage))


def body(payload, *, name='large.csv', request_id=None):
    return TransferBegin(request_id=request_id or uuid4().hex, name=name,
                         size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def put(env, transfer, payload, index=0, uid='editor'):
    return env.transfers.put_part(env.pid, transfer['id'], env.users[uid], index, payload,
                                  hashlib.sha256(payload).hexdigest())


def ready(env, payload=b'x,y\n1,2\n'):
    transfer = env.transfers.begin(env.pid, env.users['editor'], body(payload))
    put(env, transfer, payload)
    return transfer, env.transfers.complete(env.pid, transfer['id'], env.users['editor'])


@pytest.mark.parametrize('field,value', [('size_bytes', True), ('size_bytes', 1.0),
    ('size_bytes', MAX_DATA_BYTES + 1), ('size_bytes', 0), ('sha256', 'A' * 64),
    ('request_id', '1/../2'), ('extra', 'hidden')])
def test_begin_metadata_is_strict(field, value):
    data = body(b'x,y\n1,2\n').model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        TransferBegin(**data)


@pytest.mark.parametrize('uid,status', [('viewer', 403), ('outsider', 404)])
def test_noneditors_cannot_begin_or_buffer(env, uid, status):
    before = deepcopy(env.db.data)
    with pytest.raises(TeamError) as error:
        env.transfers.begin(env.pid, env.users[uid], body(b'abc'))
    assert error.value.status_code == status
    assert env.db.data == before and not env.storage.refs


def test_idempotency_conflicts_and_partial_is_not_catalogue(env):
    source = b'x,y\n1,2\n'
    spec = body(source)
    transfer = env.transfers.begin(env.pid, env.users['editor'], spec)
    assert env.transfers.begin(env.pid, env.users['editor'], spec) == transfer
    assert env.store.project(env.pid, env.users['viewer'])['files'] == []
    changed = spec.model_copy(update={'sha256': '0' * 64})
    with pytest.raises(TeamError, match='different content'):
        env.transfers.begin(env.pid, env.users['editor'], changed)
    with pytest.raises(TeamError) as error:
        env.transfers.complete(env.pid, transfer['id'], env.users['editor'])
    assert error.value.code == 'TRANSFER_INCOMPLETE'
    assert not env.storage.refs
    first = put(env, transfer, source)
    second = put(env, transfer, source)
    assert first['reused'] is False and second['reused'] is True
    assert len(env.storage.refs) == 1
    with pytest.raises(TeamError) as error:
        put(env, transfer, b'x,y\n3,4\n')
    assert error.value.code == 'TRANSFER_CONFLICT'
    metadata = env.transfers.complete(env.pid, transfer['id'], env.users['editor'])
    assert env.transfers.complete(env.pid, transfer['id'], env.users['owner']) == metadata
    assert len(env.store.project(env.pid, env.users['viewer'])['files']) == 1
    assert len(env.storage.refs) == 2
    assert 'blob' not in metadata


@pytest.mark.parametrize('uid,status', [('editor2', 403), ('viewer', 403), ('outsider', 404)])
@pytest.mark.parametrize('operation', ['status', 'part', 'complete', 'cancel'])
def test_upload_journal_is_creator_or_owner_scoped(env, uid, status, operation):
    transfer = env.transfers.begin(env.pid, env.users['editor'], body(b'abc'))
    user = env.users[uid]
    actions = {
        'status': lambda: env.transfers.status(env.pid, transfer['id'], user),
        'part': lambda: put(env, transfer, b'abc', uid=uid),
        'complete': lambda: env.transfers.complete(env.pid, transfer['id'], user),
        'cancel': lambda: env.transfers.cancel(env.pid, transfer['id'], user),
    }
    with pytest.raises(TeamError) as error:
        actions[operation]()
    assert error.value.status_code == status
    assert not env.storage.refs


def test_membership_is_rechecked_after_manifest_and_part_read(env):
    _, metadata = ready(env)
    for action in ('manifest', 'download'):
        project = env.db.get(f'oe_projects/{env.pid}')
        project['members']['viewer'] = {'role': 'viewer', 'email': env.users['viewer'].email}
        env.db.put(f'oe_projects/{env.pid}', project)
        def revoke():
            value = env.db.get(f'oe_projects/{env.pid}')
            del value['members']['viewer']
            env.db.put(f'oe_projects/{env.pid}', value)
        env.storage.after_get = revoke
        with pytest.raises(TeamError) as error:
            if action == 'manifest':
                env.transfers.manifest(env.pid, metadata['id'], env.users['viewer'])
            else:
                env.transfers.download_part(env.pid, metadata['id'], env.users['viewer'], 0)
        assert error.value.status_code == 404


def test_other_project_cannot_read_manifest_or_parts(env):
    _, metadata = ready(env)
    reads = len(env.storage.get_sizes)
    with pytest.raises(TeamError) as error:
        env.transfers.manifest(env.other, metadata['id'], env.users['owner'])
    assert error.value.status_code == 404 and len(env.storage.get_sizes) == reads


@pytest.mark.parametrize('damage', ['part-bytes', 'whole-hash'])
def test_integrity_failures_never_publish(env, damage):
    payload = b'x,y\n1,2\n'
    spec = body(payload)
    if damage == 'whole-hash':
        spec = spec.model_copy(update={'sha256': '0' * 64})
    transfer = env.transfers.begin(env.pid, env.users['editor'], spec)
    put(env, transfer, payload)
    if damage == 'part-bytes':
        ref = next(iter(env.storage.refs.values()))
        (env.storage.root / ref['key']).write_bytes(b'x,y\n3,4\n')
    with pytest.raises(TeamError) as error:
        env.transfers.complete(env.pid, transfer['id'], env.users['editor'])
    assert error.value.code == 'DATA_INTEGRITY'
    assert env.store.project(env.pid, env.users['viewer'])['files'] == []


@pytest.mark.parametrize('damage', ['index-bool', 'negative-generation', 'tenant-key', 'size-bool',
                                  'part-bytes-float', 'injected-file', 'extra-part'])
def test_untrusted_manifest_is_rejected(env, damage):
    _, metadata = ready(env)
    stored = env.store.project(env.pid, env.users['owner'])['files'][0]
    ref = stored['blob']
    data = json.loads(env.storage.get(ref, maximum=MAX_MANIFEST_BYTES))
    if damage == 'index-bool':
        data['parts'][0]['index'] = False
    elif damage == 'size-bool':
        data['size_bytes'] = True
    elif damage == 'negative-generation':
        data['parts'][0]['blob']['generation'] = -1
    elif damage == 'part-bytes-float':
        data['part_bytes'] = float(PART_BYTES)
    elif damage == 'injected-file':
        data['file'] = {'name': 'injected.csv', 'data_hash': '0' * 64}
    elif damage == 'extra-part':
        data['parts'][0]['extra'] = 'unexpected'
    else:
        data['parts'][0]['blob']['key'] = data['parts'][0]['blob']['key'].replace(env.pid, env.other)
    changed = json.dumps(data).encode()
    (env.storage.root / ref['key']).write_bytes(changed)
    env.storage.refs[ref['key']] = {**ref, 'size': len(changed)}
    stored['blob']['size'] = len(changed)
    project = env.db.get(f'oe_projects/{env.pid}')
    project['files'][0] = stored
    env.db.put(f'oe_projects/{env.pid}', project)
    with pytest.raises(TeamError) as error:
        env.transfers.manifest(env.pid, metadata['id'], env.users['viewer'])
    assert error.value.code == 'DATA_INTEGRITY'


def test_cancel_keeps_budget_until_storage_cleanup_and_is_retryable(env):
    transfer = env.transfers.begin(env.pid, env.users['editor'], body(b'abc'))
    put(env, transfer, b'abc')
    env.storage.fail_delete = True
    with pytest.raises(OSError):
        env.transfers.cancel(env.pid, transfer['id'], env.users['editor'])
    assert transfer['id'] in env.store.project(env.pid, env.users['owner'])['transfer_reservations']
    with pytest.raises(TeamError) as error:
        put(env, transfer, b'abc')
    assert error.value.code == 'TRANSFER_CANCELLED'
    env.storage.fail_delete = False
    assert env.transfers.cancel(env.pid, transfer['id'], env.users['owner']) == {'cancelled': True}
    assert env.transfers.cancel(env.pid, transfer['id'], env.users['editor']) == {'cancelled': True}
    assert not env.storage.refs
    assert not env.store.project(env.pid, env.users['owner'])['transfer_reservations']


def test_expired_reservations_cannot_bypass_project_budget(env):
    for index in range(4):
        transfer = env.transfers.begin(env.pid, env.users['editor'], body(b'abc', name=f'{index}.csv'))
        path = f'oe_projects/{env.pid}/transfers/{transfer["id"]}'
        journal = env.db.get(path)
        journal['expires_at'] = '2000-01-01T00:00:00+00:00'
        env.db.put(path, journal)
    with pytest.raises(TeamError) as error:
        env.transfers.begin(env.pid, env.users['editor'], body(b'abc', name='fifth.csv'))
    assert error.value.code == 'TRANSFER_LIMIT'


@pytest.mark.parametrize('phase', ['object-ack', 'db-before', 'db-after'])
def test_ambiguous_part_ack_retries_the_same_immutable_key(env, monkeypatch, phase):
    payload = b'x,y\n1,2\n'
    transfer = env.transfers.begin(env.pid, env.users['editor'], body(payload))
    part_path = f'oe_projects/{env.pid}/transfers/{transfer["id"]}/parts/0000'
    if phase == 'object-ack':
        original_put = env.storage.put
        failed = False
        def lost_ack(*args, **kwargs):
            nonlocal failed
            result = original_put(*args, **kwargs)
            if not failed:
                failed = True
                raise OSError('lost immutable object acknowledgement')
            return result
        monkeypatch.setattr(env.storage, 'put', lost_ack)
        put(env, transfer, payload)
    else:
        original_atomic = env.db.atomic
        failed = False
        def maybe_lost_ack(fn):
            nonlocal failed
            def run(db):
                nonlocal failed
                result = fn(db)
                current = db.get(part_path)
                if not failed and current and current.get('blob'):
                    failed = True
                    if phase == 'db-before':
                        raise OSError('failed before part commit')
                return result
            result = original_atomic(run)
            if failed and phase == 'db-after':
                monkeypatch.setattr(env.db, 'atomic', original_atomic)
                raise OSError('lost part database acknowledgement')
            return result
        monkeypatch.setattr(env.db, 'atomic', maybe_lost_ack)
        with pytest.raises(OSError):
            put(env, transfer, payload)
        assert len(env.storage.refs) == 1
        put(env, transfer, payload)
    assert len(env.storage.refs) == 1
    assert len(env.transfers.status(env.pid, transfer['id'], env.users['editor'])['parts']) == 1
    ready_file = env.transfers.complete(env.pid, transfer['id'], env.users['editor'])
    assert ready_file['data_hash'] == hashlib.sha256(payload).hexdigest()


def test_cancel_pending_claim_holds_budget_until_lease_and_recovers_unacknowledged_object(env):
    payload = b'abc'
    transfer = env.transfers.begin(env.pid, env.users['editor'], body(payload))
    part_path = f'oe_projects/{env.pid}/transfers/{transfer["id"]}/parts/0000'
    checksum = hashlib.sha256(payload).hexdigest()
    key = f'projects/{env.pid}/transfers/{transfer["id"]}/0000/{checksum}'
    env.db.put(part_path, {'index': 0, 'sha256': checksum, 'size_bytes': 3,
                          'blob': None, 'key': key, 'lease_until': '2999-01-01T00:00:00+00:00'})
    env.storage.put(key, payload)
    with pytest.raises(TeamError) as error:
        env.transfers.cancel(env.pid, transfer['id'], env.users['owner'])
    assert error.value.code == 'CLEANUP_PENDING'
    assert not env.storage.refs
    assert transfer['id'] in env.store.project(env.pid, env.users['owner'])['transfer_reservations']
    claim = env.db.get(part_path)
    claim['lease_until'] = '2000-01-01T00:00:00+00:00'
    env.db.put(part_path, claim)
    assert env.transfers.cancel(env.pid, transfer['id'], env.users['owner'])['cancelled'] is True
    assert not env.store.project(env.pid, env.users['owner'])['transfer_reservations']


def test_small_upload_and_large_reservations_share_casefold_name_and_total_budget(env):
    env.transfers.begin(env.pid, env.users['editor'], body(b'abc', name='Large.csv'))
    with pytest.raises(TeamError) as error:
        env.store.add_file(env.pid, env.users['editor'], {'name': 'large.csv', 'size_bytes': 3})
    assert error.value.code == 'FILE_EXISTS'
    project = env.db.get(f'oe_projects/{env.pid}')
    project['files'] = [{'name': f'{i}.csv', 'size_bytes': MAX_DATA_BYTES,
                         'transfer': 'chunked-v1'} for i in range(4)]
    project['transfer_reservations'] = {}
    env.db.put(f'oe_projects/{env.pid}', project)
    with pytest.raises(TeamError) as error:
        env.store.add_file(env.pid, env.users['editor'], {'name': 'small.csv', 'size_bytes': 3})
    assert error.value.code == 'FILE_LIMIT'


@pytest.fixture
def http(env):
    from fastapi.testclient import TestClient
    from openecon.team_auth import TeamAuth
    from openecon.team_server import create_team_app
    origin = 'https://synthetic.example.com'
    def verify(token):
        return {'uid': token, 'sub': token, 'email': f'{token}@example.com',
                'email_verified': True, 'name': token, 'aud': 'openecon-test',
                'iss': 'https://securetoken.google.com/openecon-test',
                'firebase': {'sign_in_provider': 'password'}}
    app = create_team_app(store=env.store, storage=env.storage,
        auth=TeamAuth('openecon-test', verifier=verify), runner=object(), public_origin=origin,
        firebase_config={'projectId': 'openecon-test', 'authDomain': 'openecon-test.firebaseapp.com'})
    with TestClient(app, base_url=origin) as client:
        yield client


def test_http_owner_upload_viewer_download_strict_errors_and_local_compute(env, http):
    prefix = f'/api/projects/{env.pid}/workspace'
    headers = {'Authorization': 'Bearer editor'}
    payload = b'x,y\n1,2\n'
    response = http.post(prefix + '/transfers', json=body(payload).model_dump(), headers=headers)
    assert response.status_code == 201, response.text
    transfer = response.json()
    part = prefix + f'/transfers/{transfer["id"]}/parts/0'
    assert http.put(part, content=payload, headers=headers).status_code == 422
    assert http.put(part, content=b'x' * (PART_BYTES + 1), headers=headers).status_code == 413
    response = http.put(part, content=payload, headers={**headers,
        'X-OpenEcon-SHA256': hashlib.sha256(payload).hexdigest()})
    assert response.status_code == 200, response.text
    completed = http.post(prefix + f'/transfers/{transfer["id"]}/complete', headers=headers)
    assert completed.status_code == 200, completed.text
    file_id = completed.json()['id']
    viewer = {'Authorization': 'Bearer viewer'}
    assert http.get(prefix + f'/files/{file_id}/manifest', headers=viewer).status_code == 200
    response = http.get(prefix + f'/files/{file_id}/parts/0', headers=viewer)
    assert response.status_code == 200 and response.content == payload
    assert response.headers['X-OpenEcon-SHA256'] == hashlib.sha256(payload).hexdigest()
    assert http.get(prefix + f'/files/{file_id}/download', headers=viewer).status_code == 409
    response = http.post(prefix + '/console/execute', json={'code': '1 + 1'}, headers=headers)
    assert response.status_code == 409 and response.json()['detail']['code'] == 'LOCAL_COMPUTE_REQUIRED'
    assert env.store.project(env.pid, env.users['owner'])['active_run'] is None


@pytest.mark.parametrize('uid,status', [('viewer', 403), ('outsider', 404)])
def test_http_rechecks_membership_before_stream_body(env, http, uid, status):
    transfer = env.transfers.begin(env.pid, env.users['editor'], body(b'abc'))
    path = f'/api/projects/{env.pid}/workspace/transfers/{transfer["id"]}/parts/0'
    response = http.put(path, content=iter([b'x' * (PART_BYTES + 1)]),
                        headers={'Authorization': f'Bearer {uid}'})
    assert response.status_code == status
    assert not env.storage.refs


def test_legacy_small_upload_keeps_committed_blob_after_lost_database_ack(env, http, monkeypatch):
    original_atomic = env.db.atomic
    failed = False
    def lost_ack(fn):
        nonlocal failed
        result = original_atomic(fn)
        if not failed and env.store.project(env.pid, env.users['owner'])['files']:
            failed = True
            raise OSError('lost committed file acknowledgement')
        return result
    monkeypatch.setattr(env.db, 'atomic', lost_ack)
    response = http.post(f'/api/projects/{env.pid}/workspace/datasets/example',
                         headers={'Authorization': 'Bearer editor'})
    assert response.status_code == 201, response.text
    metadata = response.json()
    stored = env.store.project(env.pid, env.users['viewer'])['files'][0]
    assert stored['id'] == metadata['id']
    assert stored['blob']['key'] in env.storage.refs
    data = env.storage.get(stored['blob'], maximum=PART_BYTES)
    assert hashlib.sha256(data).hexdigest() == metadata['data_hash']


def test_physical_large_csv_offline_resume_second_member_and_local_torch_fit(env, tmp_path, record_property):
    """Real >24 MiB disk file and Torch Dataset fit, isolated HTTP/storage provenance."""
    import openecon as oe
    source = tmp_path / 'large.csv'
    line = '1,3\n2,5\n3,7\n4,9\n'
    with source.open('w') as output:
        output.write('x,y\n')
        for _ in range(1_600_000):
            output.write(line)
    size = source.stat().st_size
    assert size > 24 * 1024**2
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for part in iter(lambda: stream.read(PART_BYTES), b''):
            digest.update(part)
    spec = TransferBegin(request_id=uuid4().hex, name='large.csv', size_bytes=size,
                         sha256=digest.hexdigest())
    transfer = env.transfers.begin(env.pid, env.users['editor'], spec)
    # Stop after two acknowledged parts; no catalogue entry is visible offline.
    with source.open('rb') as stream:
        for index in range(2):
            put(env, transfer, stream.read(PART_BYTES), index)
    assert not env.store.project(env.pid, env.users['viewer'])['files']
    acknowledged = env.transfers.status(env.pid, transfer['id'], env.users['editor'])['parts']
    assert [p['index'] for p in acknowledged] == [0, 1]
    with source.open('rb') as stream:
        stream.seek(2 * PART_BYTES)
        for index in range(2, transfer['part_count']):
            put(env, transfer, stream.read(PART_BYTES), index)
    metadata = env.transfers.complete(env.pid, transfer['id'], env.users['editor'])
    manifest = env.transfers.manifest(env.pid, metadata['id'], env.users['viewer'])
    assert all('blob' not in part for part in manifest['parts'])
    downloaded = tmp_path / 'second-member.csv'
    downloaded_hash = hashlib.sha256()
    with downloaded.open('wb') as output:
        for index in range(manifest['file']['size_bytes'] // PART_BYTES + 1):
            if index >= len(manifest['parts']):
                break
            data, _ = env.transfers.download_part(env.pid, metadata['id'], env.users['viewer'], index)
            output.write(data)
            downloaded_hash.update(data)
    assert downloaded_hash.hexdigest() == spec.sha256
    result = oe.ols(data=oe.scan(downloaded), y='y', x=['x'], covariance='HC1')
    assert result.nobs == 6_400_000
    assert result.coefficients[0].estimate == pytest.approx(1.0, abs=1e-7)
    assert result.coefficients[1].estimate == pytest.approx(2.0, abs=1e-7)
    assert max(env.storage.get_sizes) <= PART_BYTES
    assert max(env.storage.put_sizes) <= PART_BYTES
    record_property('physical_bytes', size)
    record_property('source_sha256', spec.sha256)
    record_property('part_count', transfer['part_count'])
    record_property('observations', result.nobs)
    record_property('intercept', result.coefficients[0].estimate)
    record_property('slope', result.coefficients[1].estimate)
    record_property('maximum_blob_read_bytes', max(env.storage.get_sizes))
    record_property('maximum_blob_write_bytes', max(env.storage.put_sizes))
    record_property('transport_provenance', 'isolated_disk_storage_double_not_cloud_or_installed_app')
