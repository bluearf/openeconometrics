"""Automatic QA cleanup is refused; ownership inspection is always read-only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from google.cloud import storage
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


def test_finished_cancelled_failed_and_missing_empty_projects_can_be_inspected():
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


@pytest.mark.parametrize('arguments', [[], ['--url', 'https://owned.example.com'],
                                     ['--resume-run', 'claimed-quiet-run'], ['--quiet'],
                                     ['--cleanup-receipt', 'claimed-quiet-receipt.json']])
def test_cleanup_main_refuses_before_credentials_private_state_or_any_sdk_call(tmp_path, monkeypatch, arguments):
    state = tmp_path / 'qa-state.json'
    manifest, database = cleanup_fixture()
    manifest['projects'].append(LATER_PROJECT)
    database.documents['oe_projects/' + LATER_PROJECT] = {
        'owner_uid': OWNER, 'members': {OWNER: {}}, 'active_run': 'active'}
    manifest['quiescence_enforced'] = True
    manifest['cleanup_receipt'] = {'safe_to_delete': True}
    original = json.dumps(manifest, indent=3).encode()
    state.write_bytes(original)
    private_state = Mock(wraps=state)
    private_state.read_text.side_effect = AssertionError('Private manifest must not be read')
    monkeypatch.setattr(verify, 'STATE', private_state)
    monkeypatch.setattr('sys.argv', ['verify_team_live.py', 'cleanup', *arguments])
    calls = []
    for module, name in [(verify.subprocess, 'check_output'), (verify, 'Credentials'),
                         (verify.firebase_admin, 'initialize_app'), (verify.firestore, 'Client'),
                         (storage, 'Client'), (verify.auth, 'delete_user')]:
        call = Mock(side_effect=AssertionError('Cleanup must refuse before SDK access'))
        monkeypatch.setattr(module, name, call)
        calls.append(call)
    with pytest.raises(SystemExit):
        verify.main()
    assert all(not call.called for call in calls)
    assert private_state.mock_calls == []
    assert state.read_bytes() == original
    assert not (tmp_path / 'artifacts/verification/team-live-qa.json').exists()


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
        snapshot = CleanupSnapshot(self.path, self.database.documents.get(self.path))
        if self.database.after_get:
            self.database.after_get(self.path)
        return snapshot

    def collection(self, name):
        return CleanupCollection(self.database, self.path + '/' + name)

    def delete(self):
        self.database.deleted.append(self.path)
        self.database.documents.pop(self.path, None)

    def update(self, changes):
        self.database.updated.append(self.path)
        self.database.documents[self.path].update(changes)


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
        self.after_get = None

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


def execute_inspection(database, state, monkeypatch):
    delete_identity = Mock()
    blobs = SimpleNamespace(list_blobs=Mock(return_value=[]))
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    report = verify.inspect_cleanup_qa(database, state)
    assert report['inspection_only'] is True
    assert report['safe_to_delete'] is False and report['quiescence_enforced'] is False
    assert report['state_retained'] is True
    assert not any('removed' in key for key in report)
    assert database.deleted == [] and database.updated == []
    delete_identity.assert_not_called()
    blobs.list_blobs.assert_not_called()
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
    original_documents = deepcopy(database.documents)
    report, delete_identity, blobs = execute_inspection(database, state, monkeypatch)
    assert report['identities_with_other_project_associations'] == 1
    assert report['identities_inspected'] == 2
    assert report['state_retained'] is True
    delete_identity.assert_not_called()
    assert 'oe_users/' + OWNER not in database.deleted
    assert 'oe_projects/' + OTHER_PROJECT not in database.deleted
    if association != 'profile_only':
        assert database.documents['oe_projects/' + OTHER_PROJECT]['active_run'] == 'human-analysis'
    if association != 'missing_profile':
        profile = database.documents['oe_users/' + OWNER]
        assert OWN_PROJECT in profile['project_ids']
        assert profile['name'] == 'Unchanged' and profile['enabled'] is True
        if association == 'profile_only':
            assert OTHER_PROJECT in profile['project_ids']
    assert database.documents == original_documents
    blobs.list_blobs.assert_not_called()
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
        verify.inspect_cleanup_qa(database, state)
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
        verify.inspect_cleanup_qa(database, state)
    assert database.deleted == [] and database.updated == []
    delete_identity.assert_not_called()


def test_owned_only_inspection_preserves_every_disposable_account_and_resource(monkeypatch):
    state, database = cleanup_fixture()
    original = deepcopy(database.documents)
    report, delete_identity, _ = execute_inspection(database, state, monkeypatch)
    assert report['identities_with_other_project_associations'] == 0
    assert report['identities_inspected'] == 2 and report['projects_inspected'] == 1
    assert report['invitations_inspected'] == 1 and report['desktop_grants_inspected'] == 1
    delete_identity.assert_not_called()
    assert database.documents == original


def test_inspection_cannot_promote_manifest_quiet_or_deletion_claims(monkeypatch):
    state, database = cleanup_fixture()
    state.update({'quiet': True, 'inspection_only': False, 'safe_to_delete': True,
                  'quiescence_enforced': True,
                  'cleanup_receipt': {'safe_to_delete': True, 'quiescence_enforced': True}})
    original_state, original_documents = deepcopy(state), deepcopy(database.documents)
    report, _, _ = execute_inspection(database, state, monkeypatch)
    assert report['inspection_only'] is True
    assert report['safe_to_delete'] is False and report['quiescence_enforced'] is False
    assert state == original_state and database.documents == original_documents


@pytest.mark.parametrize('delayed_change', ['running_run', 'adopted_membership'])
def test_delayed_change_after_idle_inspection_never_authorizes_cleanup(tmp_path, monkeypatch, delayed_change):
    state, database = cleanup_fixture()
    private_path = tmp_path / 'qa-state.json'
    original = json.dumps(state, indent=3).encode()
    private_path.write_bytes(original)
    triggered = []

    def change_after_final_preflight_read(path):
        if path != 'oe_desktop_logins/' + GRANT:
            return
        triggered.append(path)
        if delayed_change == 'running_run':
            database.documents['oe_projects/' + OWN_PROJECT]['active_run'] = 'new-active-run'
            database.documents['oe_projects/' + OWN_PROJECT + '/runs/late'] = {'state': 'running'}
        else:
            database.documents['oe_projects/' + OTHER_PROJECT] = {
                'owner_uid': 'human-user', 'members': {EDITOR: {'role': 'editor'}}}
            database.documents['oe_users/' + EDITOR]['project_ids'].append(OTHER_PROJECT)

    database.after_get = change_after_final_preflight_read
    report, delete_identity, blobs = execute_inspection(database, state, monkeypatch)
    assert triggered == ['oe_desktop_logins/' + GRANT]
    assert report['safe_to_delete'] is False and report['quiescence_enforced'] is False
    after_external_change = deepcopy(database.documents)
    with pytest.raises(SystemExit, match='Automatic deletion is disabled'):
        verify.cleanup_qa(database, blobs, object(), state, quiet=True, receipt=report)
    assert database.documents == after_external_change
    assert database.deleted == [] and database.updated == []
    delete_identity.assert_not_called()
    assert private_path.read_bytes() == original


def test_inspection_main_preserves_exact_private_state_and_creates_no_report_file(tmp_path, monkeypatch, capsys):
    state, database = cleanup_fixture()
    database.documents['oe_projects/' + OTHER_PROJECT] = {'owner_uid': OWNER, 'members': {OWNER: {}}}
    private_path = tmp_path / 'qa-state.json'
    original_bytes = json.dumps(state, indent=3).encode()
    private_path.write_bytes(original_bytes)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, 'STATE', private_path)
    monkeypatch.setattr('sys.argv', ['verify_team_live.py', 'inspect-cleanup'])
    get_token = Mock(return_value='dummy-test-token')
    monkeypatch.setattr(verify.subprocess, 'check_output', get_token)
    initialize_app = Mock(side_effect=AssertionError('Inspection does not initialize Firebase Auth'))
    monkeypatch.setattr(verify.firebase_admin, 'initialize_app', initialize_app)
    monkeypatch.setattr(verify.firestore, 'Client', lambda *args, **kwargs: database)
    storage_client = Mock(side_effect=AssertionError('Inspection does not initialize Cloud Storage'))
    monkeypatch.setattr(storage, 'Client', storage_client)
    delete_identity = Mock()
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    verify.main()
    assert private_path.read_bytes() == original_bytes
    assert '--project=' + verify.PROJECT in get_token.call_args.args[0]
    stdout = capsys.readouterr().out
    assert not (tmp_path / 'artifacts/verification/team-live-qa.json').exists()
    public_output = stdout
    assert not any(secret in public_output for secret in [OWNER, EDITOR, OTHER_PROJECT, 'private-password-sentinel'])
    report = json.loads(stdout)
    assert report['inspection_only'] is True and report['state_retained'] is True
    assert report['safe_to_delete'] is False and report['quiescence_enforced'] is False
    assert report['identities_with_other_project_associations'] == 1
    initialize_app.assert_not_called()
    storage_client.assert_not_called()
    delete_identity.assert_not_called()
    assert database.deleted == [] and database.updated == []


def test_inspection_reads_only_recorded_qa_account_link_quota_keys(monkeypatch):
    state, database = cleanup_fixture()
    key = f'account-link-{OWNER}-20261009'
    state['account_link_limit_ids'] = [key, key]
    unrelated = {
        f'oe_limits/account-link-{OWNER}-20261008': {'count': 2},
        'oe_limits/account-link-human-user-20261009': {'count': 6},
        'oe_limits/desktop-login-20261009': {'count': 9},
        'oe_limits/compute': {'day': '20261009', 'runs': 1, 'active': {}},
    }
    database.documents.update(deepcopy(unrelated))
    database.documents['oe_limits/' + key] = {'count': 32}
    original = deepcopy(database.documents)
    report, _, _ = execute_inspection(database, state, monkeypatch)
    assert database.documents == original
    assert report['recorded_account_link_quotas_inspected'] == 1
    assert report['recorded_account_link_quotas_present'] == 1
    assert key not in json.dumps(report) and OWNER not in json.dumps(report)


@pytest.mark.parametrize('key,quota', [
    ('account-link-human-user-20261009', {'count': 1}),
    (f'account-link-{OWNER}-20260230', {'count': 1}),
    (f'account-link-{OWNER}-20261009/foreign', {'count': 1}),
    (f'account-link-{OWNER}-20261009', {'count': True}),
    (f'account-link-{OWNER}-20261009', {'count': 33}),
    (f'account-link-{OWNER}-20261009', {'count': 1, 'private': 'sentinel'}),
])
def test_foreign_or_malformed_quota_refuses_cleanup_before_any_mutation(key, quota, monkeypatch):
    state, database = cleanup_fixture()
    state['account_link_limit_ids'] = [key]
    database.documents['oe_limits/' + key] = quota
    delete_identity = Mock()
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    with pytest.raises(SystemExit) as error:
        verify.inspect_cleanup_qa(database, state)
    assert key not in str(error.value) and OWNER not in str(error.value)
    assert database.deleted == [] and database.updated == []
    delete_identity.assert_not_called()


def test_quota_read_failure_preserves_every_resource_and_redacts_provider_error(monkeypatch):
    state, database = cleanup_fixture()
    key = f'account-link-{EDITOR}-20261009'
    state['account_link_limit_ids'] = [key]
    database.read_errors.add('oe_limits/' + key)
    delete_identity = Mock()
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    with pytest.raises(SystemExit) as error:
        verify.inspect_cleanup_qa(database, state)
    assert 'private-provider-error-sentinel' not in str(error.value)
    assert database.deleted == [] and database.updated == []
    delete_identity.assert_not_called()


def test_inspection_preserves_adopted_identity_and_its_account_link_quota(monkeypatch):
    state, database = cleanup_fixture()
    key = f'account-link-{EDITOR}-20261009'
    state['account_link_limit_ids'] = [key]
    quota = {'count': 7}
    database.documents['oe_limits/' + key] = deepcopy(quota)
    database.documents['oe_projects/' + OTHER_PROJECT] = {
        'owner_uid': 'human-user', 'members': {EDITOR: {'role': 'viewer'}}}
    original = deepcopy(database.documents)
    report, delete_identity, _ = execute_inspection(database, state, monkeypatch)
    assert report['identities_with_other_project_associations'] == 1
    assert report['recorded_account_link_quotas_present'] == 1
    assert report['state_retained'] is True
    assert database.documents['oe_limits/' + key] == quota
    assert database.documents == original
    delete_identity.assert_not_called()


def test_missing_recorded_quota_still_cannot_authorize_cleanup(monkeypatch):
    state, database = cleanup_fixture()
    key = f'account-link-{OWNER}-20261009'
    state['account_link_limit_ids'] = [key]
    original = deepcopy(database.documents)
    report, _, _ = execute_inspection(database, state, monkeypatch)
    assert report['recorded_account_link_quotas_present'] == 0
    assert report['recorded_account_link_quotas_inspected'] == 1
    assert database.documents == original


@pytest.mark.parametrize('claims', [{}, {'quiet': True}, {'quiescence_enforced': True},
                                  {'receipt': {'safe_to_delete': True, 'quiescence_enforced': True}}])
def test_legacy_cleanup_cannot_be_admitted_by_claimed_quiet_flags_or_receipts(monkeypatch, claims):
    database, blobs, app, private_state = [Mock() for _ in range(4)]
    delete_identity = Mock()
    monkeypatch.setattr(verify.auth, 'delete_user', delete_identity)
    with pytest.raises(SystemExit, match='Automatic deletion is disabled'):
        verify.cleanup_qa(database, blobs, app, private_state, **claims)
    assert all(obj.mock_calls == [] for obj in [database, blobs, app, private_state])
    delete_identity.assert_not_called()
