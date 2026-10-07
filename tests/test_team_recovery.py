"""Consistent backups, isolated publication, tamper checks and preserved access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openecon import team_recovery as recovery
from openecon.team_storage import MemoryStorage
from openecon.team_store import FirestoreDocuments, MemoryDocuments

spec = importlib.util.spec_from_file_location('recovery_drill', Path(__file__).parents[1] / 'scripts/verify_team_recovery.py')
drill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drill)


@pytest.fixture
def backed_up(tmp_path):
    team = drill.synthetic_team(runs=3)
    directory = tmp_path / 'backup'
    receipt = recovery.export_project(team.db, team.storage, team.pid, directory, source=drill.SOURCE)
    return SimpleNamespace(team=team, directory=directory, receipt=receipt,
                           db=MemoryDocuments(), blobs=MemoryStorage())


def restore(fixture):
    return recovery.restore_project(fixture.db, fixture.blobs, fixture.directory,
                                    fixture.receipt['manifest_sha256'], destination=drill.DESTINATION)


def rehash(fixture, change):
    path = fixture.directory / 'manifest.json'
    value = json.loads(path.read_bytes())
    change(value)
    payload = recovery.encoded(value)
    path.write_bytes(payload)
    fixture.receipt['manifest_sha256'] = recovery.digest(payload)


def test_complete_roundtrip_preserves_identity_metadata_bytes_and_api_permissions(backed_up):
    fixture = backed_up
    source = deepcopy(fixture.team.db.data)
    source_blobs = deepcopy(fixture.team.storage.objects)
    manifest = recovery.validate_bundle(fixture.directory, fixture.receipt['manifest_sha256'])
    receipt = restore(fixture)
    inverse = {recovery.encoded(row['destination']): row['source'] for row in receipt['object_map']}
    def original(value):
        if isinstance(value, dict):
            return inverse.get(recovery.encoded(value), {key: original(item) for key, item in value.items()})
        if isinstance(value, list):
            return [original(item) for item in value]
        return value
    restored = {path: original(value) for path, value in fixture.db.data.items() if not path.startswith('oe_recoveries/')}
    assert restored == manifest['documents']
    for row in receipt['object_map']:
        assert fixture.blobs.get(row['destination']) == fixture.team.storage.get(row['source'])
    assert len(receipt['object_map']) == fixture.receipt['blob_references']
    assert fixture.team.unrelated not in recovery.encoded(manifest).decode()
    assert fixture.db.get('oe_limits/compute') is None
    assert all(not key.startswith('staging/') for key in fixture.blobs.objects)
    assert drill.verify_api(fixture.db, fixture.blobs, fixture.team.pid)['restored_history_runs'] == 5
    assert fixture.team.db.data == source and fixture.team.storage.objects == source_blobs


def test_snapshot_is_not_changed_by_a_later_source_write(tmp_path):
    team = drill.synthetic_team(runs=1)
    old = team.store.script(team.pid, team.users['owner'])
    get = team.storage.get
    entered = False
    def changed(reference, **kwargs):
        nonlocal entered
        if not entered:
            entered = True
            team.store.save_script(team.pid, team.users['owner'], 'print("later version")', old['version'])
        return get(reference, **kwargs)
    team.storage.get = changed
    receipt = recovery.export_project(team.db, team.storage, team.pid, tmp_path / 'backup', source=drill.SOURCE)
    manifest = recovery.validate_bundle(tmp_path / 'backup', receipt['manifest_sha256'])
    assert manifest['documents'][f'oe_projects/{team.pid}/workspace/script'] == old
    assert team.store.script(team.pid, team.users['owner'])['code'] != old['code']


@pytest.mark.parametrize('kind', ['active', 'pending', 'documents', 'metadata', 'data_version', 'incomplete_artifact'])
def test_incomplete_or_over_budget_source_never_publishes_a_partial_backup(tmp_path, kind):
    team = drill.synthetic_team(runs=1)
    root = f'oe_projects/{team.pid}'
    if kind == 'active':
        team.db.data[root]['active_run'] = 'inflight'
    elif kind == 'pending':
        team.db.data[f'{root}/runs/{1:032x}']['state'] = 'starting'
    elif kind == 'documents':
        for i in range(recovery.MAX_DOCUMENTS):
            team.db.put(f'{root}/audit/event{i}', {'id': str(i), 'created_at': '2026-10-01'})
    elif kind == 'metadata':
        team.db.data[root]['description'] = 'x' * recovery.MAX_METADATA_BYTES
    elif kind == 'data_version':
        team.db.data[root]['files'][0]['data_hash'] = '0' * 64
    else:
        team.db.data[f'{root}/runs/{1:032x}']['artifacts'] = []
    before, blobs = deepcopy(team.db.data), deepcopy(team.storage.objects)
    output = tmp_path / 'backup'
    with pytest.raises(recovery.RecoveryError):
        recovery.export_project(team.db, team.storage, team.pid, output, source=drill.SOURCE)
    assert not output.exists()
    assert team.db.data == before and team.storage.objects == blobs


def test_existing_backup_directory_and_source_failure_preserve_user_files(tmp_path):
    team = drill.synthetic_team(runs=1)
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'user.txt').write_bytes(b'keep me')
    with pytest.raises(FileExistsError):
        recovery.export_project(team.db, team.storage, team.pid, output, source=drill.SOURCE)
    assert (output / 'user.txt').read_bytes() == b'keep me'
    team.storage.get = Mock(side_effect=TimeoutError('synthetic storage timeout'))
    with pytest.raises(TimeoutError):
        recovery.export_project(team.db, team.storage, team.pid, tmp_path / 'failed', source=drill.SOURCE)
    assert not (tmp_path / 'failed').exists()


@pytest.mark.parametrize('kind', ['manifest', 'blob', 'blob_symlink', 'directory_symlink', 'metadata_scope',
                                 'foreign_blob', 'missing_blob', 'core', 'owner', 'profile', 'result_source'])
def test_tampered_or_incompatible_bundle_has_no_destination_effect(backed_up, tmp_path, kind):
    f = backed_up
    if kind == 'manifest':
        with (f.directory / 'manifest.json').open('ab') as stream:
            stream.write(b' ')
    elif kind in {'blob', 'blob_symlink'}:
        manifest = recovery.validate_bundle(f.directory, f.receipt['manifest_sha256'])
        path = f.directory / 'blobs' / manifest['blobs'][0]['sha256']
        if kind == 'blob':
            path.write_bytes(b'changed')
        else:
            payload = path.read_bytes()
            path.unlink()
            foreign = tmp_path / 'foreign'
            foreign.write_bytes(payload)
            path.symlink_to(foreign)
    elif kind == 'directory_symlink':
        linked = tmp_path / 'linked'
        linked.symlink_to(f.directory)
        f.directory = linked
    else:
        def mutate(value):
            root = f'oe_projects/{f.team.pid}'
            if kind == 'metadata_scope':
                value['documents']['oe_projects/foreign'] = {'id': 'foreign'}
            elif kind == 'foreign_blob':
                value['blobs'][0]['source']['key'] = 'projects/foreign/file.csv'
            elif kind == 'missing_blob':
                value['blobs'].pop()
            elif kind == 'core':
                value['openecon_version'] = '999'
            elif kind == 'owner':
                value['documents'][root]['members']['viewer']['role'] = 'owner'
            elif kind == 'profile':
                value['documents']['oe_users/viewer']['project_ids'].append('foreign')
            elif kind == 'result_source':
                value['documents'][f'{root}/runs/{1:032x}']['code'] = 'print("forged source")'
        rehash(f, mutate)
    with pytest.raises(recovery.RecoveryError):
        restore(f)
    assert f.db.data == {} and f.blobs.objects == {}


@pytest.mark.parametrize('path', ['oe_projects/other', 'oe_users/owner', 'oe_invitations/other', 'orphan'])
def test_nonempty_destination_never_overwrites_existing_data(backed_up, path):
    f = backed_up
    path = f'oe_projects/{f.team.pid}/workspace/script' if path == 'orphan' else path
    f.db.put(path, {'human_data': 'unchanged'})
    before = deepcopy(f.db.data)
    with pytest.raises(recovery.RecoveryError, match='empty|orphaned'):
        restore(f)
    assert f.db.data == before and f.blobs.objects == {}


@pytest.mark.parametrize('field', ['gcp_project', 'bucket'])
def test_source_and_destination_must_be_independent(backed_up, field):
    location = {**drill.DESTINATION, field: drill.SOURCE[field]}
    with pytest.raises(recovery.RecoveryError, match='separate'):
        recovery.restore_project(backed_up.db, backed_up.blobs, backed_up.directory,
                                 backed_up.receipt['manifest_sha256'], destination=location)
    assert backed_up.db.data == {} and backed_up.blobs.objects == {}


@pytest.mark.parametrize('failure', ['readback', 'upload', 'stop', 'changed_file'])
def test_prepublication_failure_cleans_only_owned_uploaded_objects(backed_up, failure):
    f = backed_up
    f.blobs.objects['existing/unrelated'] = b'keep'
    if failure == 'readback':
        f.blobs.get = Mock(return_value=b'corrupted readback')
    else:
        original_put = f.blobs.put
        manifest = recovery.validate_bundle(f.directory, f.receipt['manifest_sha256'])
        calls = 0
        def failing_put(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2 and failure != 'changed_file':
                raise KeyboardInterrupt() if failure == 'stop' else TimeoutError('upload timed out')
            value = original_put(*args, **kwargs)
            if calls == 1 and failure == 'changed_file':
                (f.directory / 'blobs' / manifest['blobs'][1]['sha256']).write_bytes(b'changed after validation')
            return value
        f.blobs.put = failing_put
    with pytest.raises((recovery.RecoveryError, TimeoutError, KeyboardInterrupt)):
        restore(f)
    assert f.db.data == {} and f.blobs.objects == {'existing/unrelated': b'keep'}


def test_unknown_metadata_commit_keeps_referenced_blobs_and_refuses_blind_retry(backed_up, monkeypatch):
    f = backed_up
    publish = recovery._publish
    def lost_response(*args):
        publish(*args)
        raise TimeoutError('response lost after commit')
    monkeypatch.setattr(recovery, '_publish', lost_response)
    with pytest.raises(recovery.RecoveryError, match='reconciliation'):
        restore(f)
    assert f.db.get(f'oe_projects/{f.team.pid}') is not None
    before, blobs = deepcopy(f.db.data), deepcopy(f.blobs.objects)
    for reference in recovery.references(f.db.data, f.team.pid):
        f.blobs.get(reference)
    with pytest.raises(recovery.RecoveryError, match='empty'):
        restore(f)
    assert f.db.data == before and f.blobs.objects == blobs


def test_lost_upload_reply_removes_only_the_unpublished_owned_key(backed_up):
    f = backed_up
    put = f.blobs.put
    f.blobs.objects['existing/unrelated'] = b'keep'
    def accepted_but_reply_lost(*args, **kwargs):
        put(*args, **kwargs)
        raise TimeoutError('accepted upload reply lost')
    f.blobs.put = accepted_but_reply_lost
    with pytest.raises(recovery.RecoveryError, match='Copy stage failed: recovery'):
        restore(f)
    assert f.db.data == {} and f.blobs.objects == {'existing/unrelated': b'keep'}


def test_destination_created_during_copy_is_not_overwritten(backed_up):
    f = backed_up
    put = f.blobs.put
    human = {'id': 'human', 'content': 'keep'}
    def concurrent_destination(*args, **kwargs):
        value = put(*args, **kwargs)
        f.db.put('oe_projects/human', human)
        return value
    f.blobs.put = concurrent_destination
    with pytest.raises(recovery.RecoveryError, match='reconciliation'):
        restore(f)
    assert f.db.data == {'oe_projects/human': human}
    assert f.blobs.objects  # Publication was attempted: retain for reconciliation.


def test_failed_cleanup_reports_the_owned_scope_without_hiding_other_data(backed_up):
    f = backed_up
    f.blobs.get = Mock(return_value=b'corrupted destination')
    f.blobs.delete = Mock(side_effect=TimeoutError('delete unavailable'))
    with pytest.raises(recovery.RecoveryError, match='Copy cleanup needs reconciliation: recovery'):
        restore(f)
    assert f.db.data == {} and f.blobs.objects


def test_cli_rejects_tampering_without_loading_cloud_credentials(backed_up, monkeypatch):
    cloud = Mock(side_effect=AssertionError('No cloud clients for invalid bundles'))
    monkeypatch.setattr(recovery, '_cloud', cloud)
    with pytest.raises(SystemExit) as caught:
        recovery.main(['restore', '--gcp-project', 'synthetic-recovery', '--bucket', 'synthetic-recovery-blobs',
                       '--input', str(backed_up.directory), '--manifest-sha256', '0' * 64])
    assert caught.value.code == 2
    cloud.assert_not_called()


def test_google_sdk_queries_share_readonly_snapshot_and_restore_uses_atomic_creates(backed_up, monkeypatch):
    firestore = pytest.importorskip('google.cloud.firestore')
    from google.auth.credentials import AnonymousCredentials
    from google.cloud.firestore_v1.document import DocumentReference
    from google.cloud.firestore_v1.query import Query
    f = backed_up
    source = object.__new__(FirestoreDocuments)
    source.client = firestore.Client(project='synthetic-source', credentials=AnonymousCredentials())
    target = object.__new__(FirestoreDocuments)
    target.client = firestore.Client(project='synthetic-recovery', credentials=AnonymousCredentials())
    transactions, reads = [], []
    def transactional(function):
        def run(transaction):
            transactions.append(transaction)
            return function(transaction)
        return run
    monkeypatch.setattr(firestore, 'transactional', transactional)
    def get(reference, *, transaction, timeout):
        assert timeout == 25
        reads.append(transaction)
        data = f.team.db.get(reference.path) if reference._client is source.client else None
        return SimpleNamespace(to_dict=lambda: data, exists=data is not None)
    monkeypatch.setattr(DocumentReference, 'get', get)
    def stream(query, *, transaction, timeout):
        assert timeout == 25
        reads.append(transaction)
        proto = query._to_protobuf()
        assert proto.limit == (401 if query._client is source.client else 1)
        if query._client is target.client:
            return iter([])
        collection = '/'.join(query._parent._path)
        if collection == 'oe_invitations':
            assert proto.where.field_filter.field.field_path == 'project_id'
            assert proto.where.field_filter.value.string_value == f.team.pid
        return iter([SimpleNamespace(reference=SimpleNamespace(path=path), to_dict=lambda value=value: deepcopy(value))
                     for path, value in f.team.db.data.items() if path.startswith(collection + '/')
                     and path.count('/') == collection.count('/') + 1
                     and (collection != 'oe_invitations' or value['project_id'] == f.team.pid)])
    monkeypatch.setattr(Query, 'stream', stream)
    values = recovery.snapshot(source, f.team.pid)
    assert values == recovery.snapshot(f.team.db, f.team.pid)
    assert transactions[0]._read_only is True and transactions[0]._write_pbs == []
    assert all(transaction is transactions[0] for transaction in reads)
    reads.clear()
    receipt = {'schema': 1, 'recovery_id': 'a' * 32, 'project_id': f.team.pid}
    recovery._publish(target, values, receipt)
    tx = transactions[1]
    assert tx._read_only is False and tx._max_attempts == 1
    assert all(transaction is tx for transaction in reads)
    assert len(tx._write_pbs) == len(values) + 1 <= 401
    assert all(write.current_document.exists is False for write in tx._write_pbs)
    assert sum(write._pb.ByteSize() for write in tx._write_pbs) < 8 * 1024**2
