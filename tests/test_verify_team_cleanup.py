"""Disposable QA cleanup fails before any deletion when a run is unresolved."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'team_live_verify', Path(__file__).parents[1] / 'scripts/verify_team_live.py')
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)


class Snapshot:
    def __init__(self, document):
        self.document = document

    def to_dict(self):
        return deepcopy(self.document)


class Reference:
    def __init__(self, project, runs):
        self.project, self.runs = project, runs

    def get(self):
        return Snapshot(self.project)

    def collection(self, name):
        assert name == 'runs'
        return self

    def stream(self):
        return [Snapshot(run) for run in self.runs]


class Database:
    def __init__(self, projects):
        self.projects = projects
        self.read_paths = []

    def document(self, path):
        self.read_paths.append(path)
        return Reference(*self.projects[path])


@pytest.mark.parametrize('project,runs', [
    ({'active_run': 'private-run-sentinel'}, [{'state': 'finished'}]),
    ({'active_run': None}, [{'state': 'starting'}]),
    ({'active_run': None}, [{'state': 'running'}]),
    (None, [{'state': 'running'}]),
    ({'active_run': None}, [{}]),
])
def test_active_or_unresolved_disposable_run_refuses_cleanup_without_private_values(project, runs):
    database = Database({'oe_projects/qa-private-project-sentinel': (project, runs)})
    with pytest.raises(SystemExit) as error:
        verify.ensure_cleanup_idle(database, ['qa-private-project-sentinel'])
    assert 'private-run-sentinel' not in str(error.value)
    assert 'qa-private-project-sentinel' not in str(error.value)
    assert database.read_paths == ['oe_projects/qa-private-project-sentinel']


def test_finished_cancelled_failed_and_missing_empty_projects_can_be_cleaned():
    database = Database({
        'oe_projects/own-first': ({'active_run': None}, [{'state': state}
            for state in ['finished', 'cancelled', 'failed']]),
        'oe_projects/own-empty': (None, []),
        'oe_projects/foreign': ({'active_run': 'active'}, [{'state': 'running'}]),
    })
    verify.ensure_cleanup_idle(database, ['own-first', 'own-empty'])
    assert database.read_paths == ['oe_projects/own-first', 'oe_projects/own-empty']


def test_all_disposable_projects_are_checked_before_cleanup_returns():
    database = Database({
        'oe_projects/first': ({'active_run': None}, [{'state': 'finished'}]),
        'oe_projects/second': ({'active_run': None}, [{'state': 'running'}]),
    })
    with pytest.raises(SystemExit):
        verify.ensure_cleanup_idle(database, ['first', 'second'])
    assert database.read_paths == ['oe_projects/first', 'oe_projects/second']


def test_cleanup_main_never_partly_deletes_when_later_project_is_active(tmp_path, monkeypatch):
    state = tmp_path / 'qa-state.json'
    manifest, database = cleanup_fixture()
    manifest['projects'].append(LATER_PROJECT)
    database.documents['oe_projects/' + LATER_PROJECT] = {
        'owner_uid': OWNER, 'members': {OWNER: {}}, 'active_run': 'active'}
    state.write_text(json.dumps(manifest))
    database.recursive_delete = Mock()
    delete_identity = Mock()
    monkeypatch.setattr(verify, 'STATE', state)
    monkeypatch.setattr('sys.argv', ['verify_team_live.py', 'cleanup'])
    monkeypatch.setattr(verify.subprocess, 'check_output', lambda *args, **kwargs: 'dummy-test-token')
    monkeypatch.setattr(verify.firebase_admin, 'initialize_app', lambda *args, **kwargs: object())
    monkeypatch.setattr(verify.firestore, 'Client', lambda *args, **kwargs: database)
    monkeypatch.setattr(verify.storage, 'Client', lambda *args, **kwargs: SimpleNamespace(bucket=lambda _: object()))
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    with pytest.raises(SystemExit):
        verify.main()
    database.recursive_delete.assert_not_called()
    delete_identity.assert_not_called()
    assert state.exists()


OWN_PROJECT = '1' * 32
OTHER_PROJECT = '5' * 32
LATER_PROJECT = '2' * 32
INVITATION = '3' * 32
GRANT = '4' * 32
OWNER = 'private-owner-sentinel'
EDITOR = 'private-editor-sentinel'


class CleanupSnapshot(Snapshot):
    def __init__(self, path, document):
        super().__init__(document)
        self.id = path.rsplit('/', 1)[-1]


class CleanupReference:
    def __init__(self, database, path):
        self.database, self.path = database, path

    def get(self):
        if self.path in self.database.read_errors:
            raise RuntimeError('private-provider-error-sentinel')
        return CleanupSnapshot(self.path, self.database.documents.get(self.path))

    def collection(self, name):
        return CleanupCollection(self.database, self.path + '/' + name)

    def delete(self):
        self.database.deleted.append(self.path)
        self.database.documents.pop(self.path, None)

    def update(self, changes):
        assert set(changes) == {'project_ids'}
        assert isinstance(changes['project_ids'], verify.firestore.ArrayRemove)
        document = self.database.documents[self.path]
        # Simulate a membership appended after the cleanup's final profile read.
        if self.database.concurrent_index:
            document['project_ids'].append(self.database.concurrent_index)
        remove = changes['project_ids'].values
        document['project_ids'] = [value for value in document['project_ids'] if value not in remove]
        self.database.updated.append(self.path)


class CleanupCollection:
    def __init__(self, database, path):
        self.database, self.path = database, path

    def stream(self):
        items = list(self.database.documents.items())
        for path, document in items:
            if path.startswith(self.path + '/') and path.count('/') == self.path.count('/') + 1:
                yield CleanupSnapshot(path, document)
        if self.path in self.database.stream_errors:
            raise RuntimeError('private-provider-error-sentinel')


class CleanupDatabase:
    def __init__(self, documents):
        self.documents = deepcopy(documents)
        self.deleted, self.updated = [], []
        self.read_errors, self.stream_errors = set(), set()
        self.concurrent_index = None

    def document(self, path):
        return CleanupReference(self, path)

    def collection(self, path):
        return CleanupCollection(self, path)

    def recursive_delete(self, reference):
        for path in list(self.documents):
            if path == reference.path or path.startswith(reference.path + '/'):
                self.deleted.append(path)
                del self.documents[path]


def cleanup_fixture():
    state = {'projects': [OWN_PROJECT], 'users': {
        'owner': {'uid': OWNER, 'password': 'private-password-sentinel'},
        'editor': {'uid': EDITOR}}, 'invites': [INVITATION],
        'desktop_login_ids': [GRANT], 'checks': ['Approved offline check']}
    documents = {
        'oe_projects/' + OWN_PROJECT: {'owner_uid': OWNER, 'members': {
            OWNER: {'role': 'owner'}, EDITOR: {'role': 'editor'}}, 'active_run': None},
        'oe_projects/' + OWN_PROJECT + '/runs/done': {'state': 'finished'},
        'oe_users/' + OWNER: {'project_ids': [OWN_PROJECT], 'enabled': True, 'name': 'Unchanged'},
        'oe_users/' + EDITOR: {'project_ids': [OWN_PROJECT]},
        'oe_invitations/' + INVITATION: {'project_id': OWN_PROJECT},
        'oe_desktop_logins/' + GRANT: {'uid': OWNER},
    }
    return state, CleanupDatabase(documents)


def execute_cleanup(database, state, monkeypatch):
    delete_identity = Mock()
    blobs = SimpleNamespace(list_blobs=Mock(return_value=[]))
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    report = verify.cleanup_qa(database, blobs, object(), state)
    return report, delete_identity, blobs


@pytest.mark.parametrize('association', ['profile_only', 'owner_only', 'member_only', 'missing_profile'])
def test_non_manifest_association_preserves_auth_profile_and_project(association, monkeypatch):
    state, database = cleanup_fixture()
    original_state = deepcopy(state)
    if association == 'profile_only':
        database.documents['oe_users/' + OWNER]['project_ids'].append(OTHER_PROJECT)
    else:
        database.documents['oe_projects/' + OTHER_PROJECT] = {
            'owner_uid': OWNER if association != 'member_only' else 'human-user',
            'members': {OWNER: {'role': 'editor'}}, 'active_run': 'human-analysis'}
    if association == 'missing_profile':
        del database.documents['oe_users/' + OWNER]
    else:
        database.concurrent_index = LATER_PROJECT
    report, delete_identity, blobs = execute_cleanup(database, state, monkeypatch)
    assert report['identities_preserved'] == 1 and report['identities_removed'] == 1
    assert report['state_retained'] is True
    assert [call.args[0] for call in delete_identity.call_args_list] == [EDITOR]
    assert 'oe_users/' + OWNER not in database.deleted
    assert 'oe_projects/' + OTHER_PROJECT not in database.deleted
    if association != 'profile_only':
        assert database.documents['oe_projects/' + OTHER_PROJECT]['active_run'] == 'human-analysis'
    if association != 'missing_profile':
        profile = database.documents['oe_users/' + OWNER]
        assert OWN_PROJECT not in profile['project_ids']
        assert LATER_PROJECT in profile['project_ids']
        assert profile['name'] == 'Unchanged' and profile['enabled'] is True
        if association == 'profile_only':
            assert OTHER_PROJECT in profile['project_ids']
    assert {call.kwargs['prefix'] for call in blobs.list_blobs.call_args_list} == {
        f'projects/{OWN_PROJECT}/', f'staging/{OWN_PROJECT}/'}
    assert state == original_state
    assert not any(secret in json.dumps(report) for secret in [OWNER, EDITOR, 'private-password-sentinel'])


@pytest.mark.parametrize('failure', ['profile', 'partial_project_stream', 'runs', 'invitation', 'grant'])
def test_cleanup_read_failure_prevents_all_mutations_and_hides_provider_error(failure, monkeypatch):
    state, database = cleanup_fixture()
    if failure == 'profile':
        database.read_errors.add('oe_users/' + OWNER)
    elif failure == 'partial_project_stream':
        database.stream_errors.add('oe_projects')
    elif failure == 'runs':
        database.stream_errors.add('oe_projects/' + OWN_PROJECT + '/runs')
    elif failure == 'invitation':
        database.read_errors.add('oe_invitations/' + INVITATION)
    else:
        database.read_errors.add('oe_desktop_logins/' + GRANT)
    delete_identity = Mock()
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    with pytest.raises(SystemExit) as error:
        verify.cleanup_qa(database, object(), object(), state)
    assert 'private-provider-error-sentinel' not in str(error.value)
    assert OWNER not in str(error.value)
    assert database.deleted == [] and database.updated == []
    delete_identity.assert_not_called()


@pytest.mark.parametrize('foreign_resource', ['owner', 'member', 'invitation', 'grant', 'late_invalid_grant'])
def test_cleanup_refuses_adopted_or_malformed_manifest_resources_before_any_delete(foreign_resource, monkeypatch):
    state, database = cleanup_fixture()
    if foreign_resource == 'owner':
        database.documents['oe_projects/' + OWN_PROJECT]['owner_uid'] = 'human-user'
    elif foreign_resource == 'member':
        database.documents['oe_projects/' + OWN_PROJECT]['members']['human-user'] = {'role': 'viewer'}
    elif foreign_resource == 'invitation':
        database.documents['oe_invitations/' + INVITATION]['project_id'] = OTHER_PROJECT
    elif foreign_resource == 'grant':
        database.documents['oe_desktop_logins/' + GRANT]['uid'] = 'human-user'
    else:
        state['desktop_login_ids'].append('../private-path-sentinel')
    delete_identity = Mock()
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    with pytest.raises(SystemExit):
        verify.cleanup_qa(database, object(), object(), state)
    assert database.deleted == [] and database.updated == []
    delete_identity.assert_not_called()


def test_owned_only_cleanup_removes_disposable_accounts_and_resources(monkeypatch):
    state, database = cleanup_fixture()
    report, delete_identity, _ = execute_cleanup(database, state, monkeypatch)
    assert report['state_retained'] is False and report['identities_preserved'] == 0
    assert report['identities_removed'] == 2 and report['projects_removed'] == 1
    assert {call.args[0] for call in delete_identity.call_args_list} == {OWNER, EDITOR}
    assert database.documents == {}


def test_new_non_manifest_association_during_project_cleanup_still_preserves_identity(monkeypatch):
    state, database = cleanup_fixture()
    delete_project = database.recursive_delete

    def add_membership_after_project_delete(reference):
        delete_project(reference)
        database.documents['oe_projects/' + OTHER_PROJECT] = {
            'owner_uid': 'human-user', 'members': {EDITOR: {'role': 'editor'}}}
        database.documents['oe_users/' + EDITOR]['project_ids'].append(OTHER_PROJECT)

    database.recursive_delete = add_membership_after_project_delete
    report, delete_identity, _ = execute_cleanup(database, state, monkeypatch)
    assert report['identities_preserved'] == 1
    assert [call.args[0] for call in delete_identity.call_args_list] == [OWNER]
    assert database.documents['oe_users/' + EDITOR]['project_ids'] == [OTHER_PROJECT]


def test_cleanup_main_preserves_exact_private_state_bytes_when_identity_adopted(tmp_path, monkeypatch, capsys):
    state, database = cleanup_fixture()
    database.documents['oe_projects/' + OTHER_PROJECT] = {'owner_uid': OWNER, 'members': {OWNER: {}}}
    private_path = tmp_path / 'qa-state.json'
    original_bytes = json.dumps(state, indent=3).encode()
    private_path.write_bytes(original_bytes)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, 'STATE', private_path)
    monkeypatch.setattr('sys.argv', ['verify_team_live.py', 'cleanup'])
    get_token = Mock(return_value='dummy-test-token')
    monkeypatch.setattr(verify.subprocess, 'check_output', get_token)
    monkeypatch.setattr(verify.firebase_admin, 'initialize_app', lambda *args, **kwargs: object())
    monkeypatch.setattr(verify.firestore, 'Client', lambda *args, **kwargs: database)
    monkeypatch.setattr(verify.storage, 'Client', lambda *args, **kwargs: SimpleNamespace(
        bucket=lambda _: SimpleNamespace(list_blobs=lambda **kwargs: [])))
    delete_identity = Mock()
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    verify.main()
    assert private_path.read_bytes() == original_bytes
    assert '--project=' + verify.PROJECT in get_token.call_args.args[0]
    stdout = capsys.readouterr().out
    public_output = stdout + (tmp_path / 'artifacts/verification/team-live-qa.json').read_text()
    assert not any(secret in public_output for secret in [OWNER, EDITOR, OTHER_PROJECT, 'private-password-sentinel'])
    assert json.loads(stdout)['state_retained'] is True
