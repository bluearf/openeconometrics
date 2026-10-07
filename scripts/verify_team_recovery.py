"""Disposable, offline recovery drill across fresh processes and disk stores.

Only synthetic identities/data are used. File adapters are test dependencies,
never a fallback for the cloud service. No ADC, Firebase or network is contacted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from uuid import uuid4

from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_recovery import (digest, encoded, export_project, references, restore_project,
                                    validate_bundle)
from openecon.team_server import create_team_app, validate_worker_result
from openecon.team_storage import MAX_TRANSFER_BYTES, MemoryStorage
from openecon.team_store import MemoryDocuments, TeamError, TeamStore, now

SOURCE = {'gcp_project': 'synthetic-source', 'bucket': 'synthetic-source-blobs'}
DESTINATION = {'gcp_project': 'synthetic-recovery', 'bucket': 'synthetic-recovery-blobs'}
ORIGIN = 'https://recovery.example.com'


class FileDocuments(MemoryDocuments):
    """Isolated single-process test adapter; persistence crosses stage boundaries."""
    def __init__(self, directory):
        super().__init__()
        self.path = Path(directory) / 'documents.json'
        self.path.parent.mkdir(exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_bytes())

    def persist(self):
        self.path.write_bytes(encoded(self.data))

    def atomic(self, function):
        value = super().atomic(function)
        self.persist()
        return value


class FileStorage(MemoryStorage):
    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory) / 'objects'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = self.directory / 'index.json'
        self.names = json.loads(self.index.read_bytes()) if self.index.exists() else {}

    def put(self, key, payload, content_type='application/octet-stream'):
        if key in self.names or len(payload) > MAX_TRANSFER_BYTES:
            raise ValueError('The synthetic immutable object is invalid or exists.')
        self.names[key] = digest(payload)
        (self.directory / digest(payload)).write_bytes(payload)
        self.index.write_bytes(encoded(self.names))
        return {'key': key, 'generation': 1, 'size': len(payload)}

    def get(self, reference, maximum=MAX_TRANSFER_BYTES):
        if reference.get('generation') != 1 or reference['key'] not in self.names:
            raise TeamError('NOT_FOUND', 'Synthetic object not found.', 404)
        payload = (self.directory / self.names[reference['key']]).read_bytes()
        if len(payload) > maximum:
            raise TeamError('RESULT_LIMIT', 'Synthetic object exceeds the limit.', 413)
        return payload

    def delete(self, reference):
        self.names.pop(reference['key'], None)
        self.index.write_bytes(encoded(self.names))


def synthetic_team(db=None, storage=None, runs=137):
    db = db if db is not None else MemoryDocuments()
    storage = storage if storage is not None else MemoryStorage()
    users = {role: TeamIdentity(role, f'{role}@example.com', role.title(), True)
             for role in ('owner', 'editor', 'viewer', 'outsider')}
    store = TeamStore(db, owner_email=users['owner'].email, public_origin=ORIGIN)
    store.me(users['owner'])
    pid = store.create_project(users['owner'], 'Synthetic recovery · İstanbul', 'Offline recovery fixture')['id']
    unrelated = store.create_project(users['owner'], 'Unrelated synthetic project')['id']
    for role in ('editor', 'viewer'):
        invite = store.invite(pid, users['owner'], users[role].email, role)
        store.accept(invite['id'], users[role])
    store.invite(pid, users['owner'], 'pending@example.com', 'viewer')
    store.save_script(pid, users['owner'], 'print("preserved source · İstanbul")', 0)
    script = store.create_script(pid, users['editor'], 'auxiliary.py', 'print("second source")')
    manifest = {'schema': 1, 'python': '3.13', 'core': {'openecon': '0.3.18a1', 'torch': '2.9.0'},
                'requirements': [{'name': 'requests', 'version': '2.32.5'}],
                'locked': [{'name': 'requests', 'version': '2.32.5'}]}
    store.save_environment(pid, users['editor'], manifest, 0)
    data = ('group,value\n' + 'İstanbul,42\n' * 10000).encode()
    fid = uuid4().hex
    reference = storage.put(f'projects/{pid}/files/{fid}/source.csv', data)
    file = {'id': fid, 'name': 'source.csv', 'python_path': 'source.csv', 'source': 'upload',
            'size_bytes': len(data), 'data_hash': digest(data), 'columns': [], 'preview': [],
            'row_count': None, 'column_count': None, 'created_at': now(), 'blob': reference}
    store.add_file(pid, users['owner'], file)
    folder = uuid4().hex
    layout = {'version': 0, 'entries': [
        {'kind': 'folder', 'id': folder, 'name': 'Research', 'parent': None},
        {'kind': 'script', 'id': 'analysis', 'name': 'analysis.py', 'parent': None},
        {'kind': 'script', 'id': script['id'], 'name': 'auxiliary.py', 'parent': folder},
        {'kind': 'dataset', 'id': fid, 'name': 'source.csv', 'parent': folder}]}
    store.put_file_layout(pid, users['owner'], layout)
    for index in range(runs):
        rid = f'{index + 1:032x}'
        run = {'id': rid, 'project_id': pid, 'uid': 'editor', 'email': users['editor'].email,
               'code': f'print("saved result {index}")', 'created_at': f'2026-10-05T{index // 60:02d}:{index % 60:02d}:00+00:00',
               'generation': index + 1, 'state': 'finished', 'files': [deepcopy(file)],
               'result': None, 'artifacts': [], 'record_summary': {'status': 'ok'},
               'input_blob': {'key': f'staging/{pid}/{rid}/input.json', 'generation': 1, 'size': 8},
               'operation': 'expired-operation', 'execution': 'expired-execution'}
        record, _ = validate_worker_result({'execution_id': rid, 'record': {
            'status': 'ok', 'stdout': f'saved result {index}\n',
            'outputs': [{'type': 'table', 'data': {'columns': ['n'], 'rows': [[index]],
                                                'total_rows': 1, 'total_columns': 1}}]}}, run)
        if index == 0:
            payload = b'preserved generated artifact\n'
            artifact = {'name': 'answer.txt', 'size_bytes': len(payload),
                        'blob': storage.put(f'projects/{pid}/results/{rid}/answer.txt', payload)}
            run['artifacts'] = [artifact]
            record['artifacts'] = [{'name': artifact['name'], 'size_bytes': artifact['size_bytes'],
                                   'url': f'/api/projects/{pid}/workspace/runs/{rid}/files/0'}]
        if index == 1:
            # Actual desktop archive schema has no project_id or files member.
            run.pop('project_id')
            run.pop('files')
            run['execution_origin'] = 'desktop'
            record.update(execution_origin='desktop', input_files=[{
                'id': fid, 'name': file['name'], 'data_hash': file['data_hash']}])
        run['result'] = storage.put(f'projects/{pid}/results/{rid}/record.json', encoded(record))
        db.put(f'oe_projects/{pid}/runs/{rid}', run)
    for status in ('failed', 'cancelled'):
        rid = uuid4().hex
        db.put(f'oe_projects/{pid}/runs/{rid}', {
            'id': rid, 'project_id': pid, 'uid': 'editor', 'email': users['editor'].email,
            'code': f'# {status}', 'created_at': '2026-10-04T00:00:00+00:00', 'generation': 0,
            'state': status, 'files': [deepcopy(file)], 'result': None, 'artifacts': [],
            'record_summary': {'status': 'error' if status == 'failed' else 'interrupted'}})
    if hasattr(db, 'persist'):
        db.persist()
    return SimpleNamespace(db=db, storage=storage, store=store, pid=pid, users=users,
                           unrelated=unrelated, fid=fid, script_id=script['id'])


def verify_api(db, blobs, pid):
    from fastapi.testclient import TestClient
    project_name = 'synthetic-recovery'
    def claims(uid):
        return {'uid': uid, 'sub': uid, 'aud': project_name,
                'iss': f'https://securetoken.google.com/{project_name}',
                'email': f'{uid}@example.com', 'name': uid.title(), 'email_verified': True,
                'firebase': {'sign_in_provider': 'password'}}
    auth = TeamAuth(project_name, verifier=claims)
    store = TeamStore(db, owner_email='owner@example.com', public_origin=ORIGIN)
    runner = SimpleNamespace(start=lambda *a, **k: (_ for _ in ()).throw(AssertionError('No execution in recovery')))
    app = create_team_app(store=store, storage=blobs, auth=auth, runner=runner, public_origin=ORIGIN,
                          firebase_config={'projectId': project_name, 'authDomain': f'{project_name}.firebaseapp.com'})
    checks = {}
    before = deepcopy(db.data)
    with TestClient(app, base_url=ORIGIN) as client:
        prefix = f'/api/projects/{pid}/workspace'
        for role in ('owner', 'editor', 'viewer'):
            headers = {'Authorization': f'Bearer {role}', 'Origin': ORIGIN}
            projects = client.get('/api/projects', headers=headers)
            assert projects.status_code == 200, projects.text
            assert projects.json()['projects'][0]['role'] == role
            for route in ('console/script', 'console/scripts', 'environment', 'files/layout'):
                response = client.get(f'{prefix}/{route}', headers=headers)
                assert response.status_code == 200, response.text
            checks[f'{role}_read'] = True
            member = TeamIdentity(role, f'{role}@example.com', role.title(), True)
            if role in ('owner', 'editor'):
                store.project(pid, member, 'editor')
            if role == 'owner':
                store.project(pid, member, 'owner')
        viewer = {'Authorization': 'Bearer viewer', 'Origin': ORIGIN}
        assert client.put(prefix + '/console/script', headers=viewer, json={'code': 'changed', 'version': 1}).status_code == 403
        assert client.post(f'/api/projects/{pid}/invitations', headers={**viewer, 'Authorization': 'Bearer editor'},
                           json={'email': 'denied@example.com', 'role': 'viewer'}).status_code == 403
        assert client.get(prefix + '/console/script', headers={'Authorization': 'Bearer outsider'}).status_code == 404
        assert client.get(prefix + '/console/script', headers={'Authorization': 'Bearer viewer'}).json()['code'] == before[f'oe_projects/{pid}/workspace/script']['code']
        found, cursor = [], None
        while True:
            page = client.get(prefix + '/console/history', headers=viewer,
                              params={'limit': 7, **({'cursor': cursor} if cursor else {})})
            assert page.status_code == 200, page.text
            payload = page.json()
            found.extend(item['id'] for item in payload['runs'])
            cursor = payload['next_cursor']
            if not cursor:
                break
        expected = {value['id'] for path, value in before.items() if path.startswith(f'oe_projects/{pid}/runs/')}
        assert len(found) == len(set(found)) == len(expected) and set(found) == expected
        rid = f'{1:032x}'
        record = client.get(f'{prefix}/runs/{rid}/record', headers=viewer)
        assert record.status_code == 200 and record.json()['stdout'] == 'saved result 0\n'
        artifact = client.get(f'{prefix}/runs/{rid}/files/0', headers=viewer)
        assert artifact.status_code == 200 and artifact.content == b'preserved generated artifact\n'
        file = before[f'oe_projects/{pid}']['files'][0]
        data = client.get(f'{prefix}/files/{file["id"]}/download', headers=viewer)
        assert data.status_code == 200 and digest(data.content) == file['data_hash']
        checks.update(viewer_write_denied=True, editor_owner_action_denied=True, outsider_denied=True,
                      restored_history_runs=len(found), source_download_hash_verified=True,
                      original_result_opened=True, generated_artifact_downloaded=True)
    assert db.data == before  # Reads/denied writes never modify recovered metadata.
    return checks


def stage(name, root):
    root = Path(root)
    if name == 'seed':
        team = synthetic_team(FileDocuments(root / 'source'), FileStorage(root / 'source'))
        return {'project_id': team.pid, 'source_full_metadata_sha256': digest(encoded(team.db.data)),
                'source_object_index_sha256': digest(encoded(team.storage.names)), 'unrelated_project': team.unrelated}
    seed = json.loads((root / 'seed.json').read_bytes())
    pid = seed['project_id']
    if name == 'export':
        return export_project(FileDocuments(root / 'source'), FileStorage(root / 'source'), pid,
                              root / 'backup', source=SOURCE)
    receipt = json.loads((root / 'export.json').read_bytes())
    if name == 'restore':
        result = restore_project(FileDocuments(root / 'destination'), FileStorage(root / 'destination'),
                                 root / 'backup', receipt['manifest_sha256'], destination=DESTINATION)
        return {key: value for key, value in result.items() if key != 'object_map'}
    manifest = validate_bundle(root / 'backup', receipt['manifest_sha256'])
    db, blobs = FileDocuments(root / 'destination'), FileStorage(root / 'destination')
    saved = next(value for path, value in db.data.items() if path.startswith('oe_recoveries/'))
    inverse = {encoded(row['destination']): row['source'] for row in saved['object_map']}
    def undo(value):
        if isinstance(value, dict):
            return inverse.get(encoded(value), {key: undo(item) for key, item in value.items()})
        if isinstance(value, list):
            return [undo(item) for item in value]
        return value
    restored = {path: undo(value) for path, value in db.data.items() if not path.startswith('oe_recoveries/')}
    assert restored == manifest['documents']
    for row in saved['object_map']:
        assert digest(blobs.get(row['destination'])) == digest(FileStorage(root / 'source').get(row['source']))
    source_db, source_blobs = FileDocuments(root / 'source'), FileStorage(root / 'source')
    assert digest(encoded(source_db.data)) == seed['source_full_metadata_sha256']
    assert digest(encoded(source_blobs.names)) == seed['source_object_index_sha256']
    assert seed['unrelated_project'] not in encoded(manifest['documents']).decode()
    api = verify_api(db, blobs, pid)
    return {'metadata_matches_after_blob_reference_translation': True,
            'scoped_metadata_sha256': digest(encoded(restored)), 'all_blob_hashes_match': True,
            'blob_references_verified': len(references(manifest['documents'], pid)),
            'source_and_unrelated_project_unchanged': True, 'api': api,
            'human_projects_read_or_written': 0, 'package_installs': 0, 'computation_launches': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--stage', choices=('seed', 'export', 'restore', 'verify'))
    parser.add_argument('--work-root', type=Path)
    args = parser.parse_args()
    if args.stage:
        started = time.perf_counter()
        value = stage(args.stage, args.work_root)
        value.update(elapsed_seconds=time.perf_counter() - started,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                     * (1 if sys.platform == 'darwin' else 1024))
        print(encoded(value).decode())
        return
    stages = {}
    recovery_started = None
    with tempfile.TemporaryDirectory(prefix='openecon-recovery-drill-') as temporary:
        root = Path(temporary)
        for name in ('seed', 'export', 'restore', 'verify'):
            if name == 'restore':
                recovery_started = time.perf_counter()
            process_started = time.perf_counter()
            result = subprocess.run([sys.executable, __file__, '--stage', name, '--work-root', str(root)],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f'The synthetic {name} stage failed:\n{result.stderr}')
            stages[name] = json.loads(result.stdout)
            stages[name]['process_wall_seconds'] = time.perf_counter() - process_started
            (root / f'{name}.json').write_bytes(encoded(stages[name]))
        recovery_wall_seconds = time.perf_counter() - recovery_started
    report = {'schema': 1, 'recorded_at': now(), 'proof': 'offline synthetic disk stores / four fresh processes',
              'live_firestore_gcs_verified': False, 'firebase_provider_login_verified': False,
              'temporary_drill_files_removed': True, 'recovery_wall_seconds': recovery_wall_seconds,
              'stages': stages}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(encoded(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
