"""Project authorization, invitations, races and publication boundaries."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace

import pytest

from openecon.team_auth import TeamIdentity
from openecon.team_store import (FirestoreDocuments, MemoryDocuments, TeamError, TeamStore,
                                 filename, validate_environment_manifest)


@pytest.mark.parametrize('case', json.loads((Path(__file__).parent / 'fixtures/project-create.json').read_text()),
                         ids=lambda case: case['id'])
def test_project_creation_shared_text_contract_without_partial_writes(team, case):
    before = deepcopy(team.db.data)
    if case['valid']:
        saved = team.store.create_project(team.owner, case['name'], case['description'])
        assert saved['name'] == case['normalized_name']
        assert saved['description'] == case['normalized_description']
    else:
        with pytest.raises(TeamError) as caught:
            team.store.create_project(team.owner, case['name'], case['description'])
        assert caught.value.status_code == 422 and caught.value.code == 'INVALID_PROJECT'
        assert team.db.data == before


def package_manifest():
    return {'schema': 1, 'python': '3.13',
            'core': {'openecon': '0.3.3a1', 'torch': '2.9.0'},
            'requirements': [{'name': 'requests', 'version': '2.32.5'}],
            'locked': [{'name': 'requests', 'version': '2.32.5'},
                       {'name': 'urllib3', 'version': '2.5.0'}]}


def test_environment_is_versioned_shared_inert_metadata(team):
    assert team.store.environment(team.pid, team.viewer) == {'manifest': None, 'version': 0}
    saved = team.store.save_environment(team.pid, team.owner, package_manifest(), 0)
    assert saved == {'manifest': package_manifest(), 'version': 1}
    assert team.store.environment(team.pid, team.viewer) == saved
    updated = package_manifest()
    updated['requirements'] = []
    second = team.store.save_environment(team.pid, team.editor, updated, 1)
    assert second['version'] == 2
    stored = team.db.get(f'oe_projects/{team.pid}/workspace/environment')
    assert stored['manifest'] == updated
    assert stored['updated_by'] == team.editor.uid
    assert team.store.project(team.pid, team.owner)['updated_at'] == stored['updated_at']
    assert any(event['action'] == 'environment.updated'
               for event in team.db.scan(f'oe_projects/{team.pid}/audit'))
    # The only persisted data are package/version metadata and an audit event.
    assert team.store.project(team.pid, team.owner)['files'] == []


def test_environment_write_conflict_preserves_saved_manifest(team):
    saved = team.store.save_environment(team.pid, team.owner, package_manifest(), 0)
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        team.store.save_environment(team.pid, team.editor, package_manifest(), 0)
    assert caught.value.code == 'VERSION_CONFLICT' and caught.value.status_code == 409
    assert team.db.data == before
    assert team.store.environment(team.pid, team.viewer) == saved


def test_concurrent_environment_writes_have_exactly_one_winner(team):
    outcomes = race([
        lambda: team.store.save_environment(team.pid, team.owner, package_manifest(), 0),
        lambda: team.store.save_environment(team.pid, team.editor, package_manifest(), 0),
    ])
    assert sorted(status for status, _ in outcomes) == ['VERSION_CONFLICT', 'ok']
    winner = next(value for status, value in outcomes if status == 'ok')
    assert team.store.environment(team.pid, team.viewer) == winner


def test_environment_read_returns_detached_metadata(team):
    manifest = package_manifest()
    saved = team.store.save_environment(team.pid, team.editor, manifest, 0)
    manifest['core']['torch'] = '999'
    saved['manifest']['requirements'].clear()
    read = team.store.environment(team.pid, team.viewer)
    read['manifest']['locked'][0]['version'] = '999'
    assert team.store.environment(team.pid, team.viewer)['manifest'] == package_manifest()


@pytest.mark.parametrize('role', [None, 'viewer'])
def test_environment_write_checks_membership_inside_transaction(team, role):
    original = team.db.atomic
    entered = False

    def revoked_before_transaction(function):
        nonlocal entered
        if not entered:
            entered = True
            team.store.change_member(team.pid, team.editor.uid, team.owner, role)
        return original(function)

    team.db.atomic = revoked_before_transaction
    with pytest.raises(TeamError) as caught:
        team.store.save_environment(team.pid, team.editor, package_manifest(), 0)
    assert caught.value.code == ('NOT_FOUND' if role is None else 'ROLE_REQUIRED')
    assert team.db.get(f'oe_projects/{team.pid}/workspace/environment') is None


def test_environment_read_rechecks_membership_after_document_read(team):
    team.store.save_environment(team.pid, team.owner, package_manifest(), 0)
    original = team.db.get
    removed = False

    def removed_during_read(path):
        nonlocal removed
        value = original(path)
        if path.endswith('/workspace/environment') and not removed:
            removed = True
            team.store.change_member(team.pid, team.editor.uid, team.owner)
        return value

    team.db.get = removed_during_read
    with pytest.raises(TeamError) as caught:
        team.store.environment(team.pid, team.editor)
    assert caught.value.code == 'NOT_FOUND'


def test_environment_normalizes_distribution_aliases_and_sorts():
    manifest = package_manifest()
    manifest['core'] = {'Torch': '2.9.0', 'OpenEcon': '0.3.3a1'}
    manifest['requirements'] = [{'name': 'My_Package.Name', 'version': '1.0rc1.post2.dev3+cpu.1'}]
    manifest['locked'] = list(reversed(manifest['locked'])) + manifest['requirements']
    normalized = validate_environment_manifest(manifest)
    assert list(normalized['core']) == ['openecon', 'torch']
    assert normalized['requirements'][0]['name'] == 'my-package-name'
    assert [row['name'] for row in normalized['locked']] == ['my-package-name', 'requests', 'urllib3']


@pytest.mark.parametrize('change', [
    'schema_bool', 'schema_extra', 'python_patch', 'python_url', 'extra_field',
    'core_list', 'core_alias_duplicate', 'core_version_url', 'core_locked_overlap',
    'requirements_missing_lock', 'requirements_mismatched_version', 'locked_alias_duplicate',
    'entry_extra', 'entry_url', 'entry_path', 'entry_operator', 'entry_vcs',
    'entry_option', 'entry_unicode', 'entry_too_long', 'locked_too_many',
    'requirements_too_many', 'core_too_many', 'locked_not_list',
])
def test_environment_rejects_unsafe_or_unbounded_manifests_without_writes(team, change):
    manifest = package_manifest()
    if change == 'schema_bool':
        manifest['schema'] = True
    elif change == 'schema_extra':
        manifest['schema'] = 2
    elif change == 'python_patch':
        manifest['python'] = '3.13.1'
    elif change == 'python_url':
        manifest['python'] = 'https://python.example'
    elif change == 'extra_field':
        manifest['index_url'] = 'https://packages.example'
    elif change == 'core_list':
        manifest['core'] = []
    elif change == 'core_alias_duplicate':
        manifest['core']['Torch'] = '2.9.0'
    elif change == 'core_version_url':
        manifest['core']['torch'] = 'https://packages.example/torch.whl'
    elif change == 'core_locked_overlap':
        manifest['locked'].append({'name': 'TORCH', 'version': '2.9.0'})
    elif change == 'requirements_missing_lock':
        manifest['locked'] = []
    elif change == 'requirements_mismatched_version':
        manifest['requirements'][0]['version'] = '2.32.4'
    elif change == 'locked_alias_duplicate':
        manifest['locked'].append({'name': 'Requests', 'version': '2.32.5'})
    elif change == 'entry_extra':
        manifest['locked'][0]['url'] = 'https://packages.example'
    elif change in {'entry_url', 'entry_path', 'entry_operator', 'entry_vcs', 'entry_option'}:
        manifest['locked'][0]['version'] = {
            'entry_url': 'https://packages.example/a.whl', 'entry_path': '../package.whl',
            'entry_operator': '>=2', 'entry_vcs': 'git+https://example.com/repo',
            'entry_option': '--index-url=example',
        }[change]
    elif change == 'entry_unicode':
        manifest['locked'][0]['name'] = 'requésts'
    elif change == 'entry_too_long':
        manifest['locked'][0]['version'] = '1' * 101
    elif change in {'locked_too_many', 'requirements_too_many'}:
        section = 'locked' if change == 'locked_too_many' else 'requirements'
        manifest[section] = [{'name': f'package-{i}', 'version': '1.0'} for i in range(101)]
    elif change == 'core_too_many':
        manifest['core'] = {f'core-{i}': '1.0' for i in range(101)}
    elif change == 'locked_not_list':
        manifest['locked'] = 'requests==2.32.5'
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        team.store.save_environment(team.pid, team.owner, manifest, 0)
    assert caught.value.code == 'INVALID_ENVIRONMENT' and caught.value.status_code == 422
    assert team.db.data == before


def test_environment_allows_one_hundred_core_and_additional_distributions():
    manifest = package_manifest()
    manifest['core'] = {f'core-{i}': '1.0' for i in range(100)}
    manifest['locked'] = [{'name': f'package-{i}', 'version': '1.0'} for i in range(100)]
    manifest['requirements'] = deepcopy(manifest['locked'])
    normalized = validate_environment_manifest(manifest)
    assert len(normalized['core']) == len(normalized['locked']) == len(normalized['requirements']) == 100


@pytest.mark.parametrize('version', [True, -1, 1.5, '0', 2**63 - 1])
def test_environment_version_is_a_bounded_strict_integer(team, version):
    with pytest.raises(TeamError) as caught:
        team.store.save_environment(team.pid, team.owner, package_manifest(), version)
    assert caught.value.code == 'INVALID_ENVIRONMENT'


@pytest.fixture
def team():
    db = MemoryDocuments()
    owner = TeamIdentity("owner", "owner@example.com", "Owner", True)
    editor = TeamIdentity("editor", "editor@example.com", "Editor", True)
    viewer = TeamIdentity("viewer", "viewer@example.com", "Viewer", True)
    outsider = TeamIdentity("outsider", "outsider@example.com", "Outside", True)
    store = TeamStore(db, owner_email=owner.email, public_origin="https://app.example.com")
    store.me(owner)
    project = store.create_project(owner, "First project")
    for user, role in [(editor, "editor"), (viewer, "viewer")]:
        invite = store.invite(project["id"], owner, user.email, role)
        store.accept(invite["id"], user)
    return SimpleNamespace(db=db, store=store, owner=owner, editor=editor,
                           viewer=viewer, outsider=outsider, pid=project["id"])


def invoke(team, action, user):
    store, pid = team.store, team.pid
    calls = {
        "project": lambda: store.project(pid, user),
        "members": lambda: store.members(pid, user),
        "script": lambda: store.script(pid, user),
        "environment": lambda: store.environment(pid, user),
        "runs": lambda: store.runs(pid, user),
        "save_script": lambda: store.save_script(pid, user, "print(1)", 0),
        "save_environment": lambda: store.save_environment(pid, user, package_manifest(), 0),
        "invite": lambda: store.invite(pid, user, "recipient@example.com", "viewer"),
        "change_member": lambda: store.change_member(pid, team.editor.uid, user, "viewer"),
        "revoke_invite": lambda: store.revoke_invite(pid, "unknown", user),
        "add_file": lambda: store.add_file(pid, user, {"id": "file1", "name": "data.csv", "size_bytes": 5}),
    }
    return calls[action]()


def test_signup_does_not_enable_application_and_verified_bootstrap_is_required(team):
    unverified_owner = replace(team.owner, uid="new-owner", email_verified=False)
    assert not team.store.me(unverified_owner)["enabled"]
    assert not team.store.me(team.outsider)["enabled"]
    assert team.store.me(team.owner)["enabled"]
    with pytest.raises(TeamError) as caught:
        team.store.create_project(team.outsider, "Unauthorized")
    assert caught.value.code == "INVITATION_REQUIRED"
    with pytest.raises(TeamError) as caught:
        team.store.create_project(unverified_owner, "Unverified")
    assert caught.value.code == "EMAIL_UNVERIFIED"


def test_accepted_invitation_enables_account_without_changing_project_role(team):
    profile = team.store.me(team.viewer)
    assert profile["enabled"]
    assert profile["projects"][0]["role"] == "viewer"
    own_project = team.store.create_project(team.viewer, "My separate project")
    assert own_project["role"] == "owner"
    assert team.store.project(team.pid, team.viewer)["members"][team.viewer.uid]["role"] == "viewer"


@pytest.mark.parametrize("action", ["project", "members", "script", "environment", "runs", "save_script", "save_environment",
                                    "invite", "change_member", "revoke_invite", "add_file"])
def test_nonmember_cannot_read_or_mutate_even_with_exact_project_id(team, action):
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        invoke(team, action, team.outsider)
    assert caught.value.code == "NOT_FOUND" and caught.value.status_code == 404
    assert team.db.data == before


@pytest.mark.parametrize("action", ["save_script", "save_environment", "invite", "change_member", "revoke_invite",
                                    "add_file"])
def test_viewer_cannot_mutate(team, action):
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        invoke(team, action, team.viewer)
    assert caught.value.code == "ROLE_REQUIRED" and caught.value.status_code == 403
    assert team.db.data == before


@pytest.mark.parametrize("action", ["invite", "change_member", "revoke_invite"])
def test_editor_cannot_manage_members_or_escalate_own_role(team, action):
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        invoke(team, action, team.editor)
    assert caught.value.code == "ROLE_REQUIRED"
    assert team.db.data == before


def test_unverified_identity_cannot_reuse_an_existing_membership(team):
    user = replace(team.editor, email_verified=False)
    assert team.store.projects(user) == []
    with pytest.raises(TeamError) as caught:
        team.store.save_script(team.pid, user, "print(1)", 0)
    assert caught.value.code == "NOT_FOUND"


def test_invitation_requires_exact_verified_recipient_and_cannot_be_replayed(team):
    invite = team.store.invite(team.pid, team.owner, " New@Example.com ", "editor")
    recipient = TeamIdentity("new", "new@example.com", "New", True)
    assert invite["email"] == recipient.email
    with pytest.raises(TeamError) as caught:
        team.store.accept(invite["id"], team.outsider)
    assert caught.value.code == "INVITATION_INVALID"
    with pytest.raises(TeamError) as caught:
        team.store.accept(invite["id"], replace(recipient, email_verified=False))
    assert caught.value.code == "EMAIL_UNVERIFIED"
    accepted = team.store.accept(invite["id"], recipient)
    assert accepted["role"] == "editor"
    with pytest.raises(TeamError) as caught:
        team.store.accept(invite["id"], recipient)
    assert caught.value.code == "INVITATION_INVALID"
    assert len(team.store.project(team.pid, team.owner)["members"]) == 4


@pytest.mark.parametrize("state", ["expired", "revoked", "detached"])
def test_inactive_invitation_cannot_enable_account(team, state):
    invite = team.store.invite(team.pid, team.owner, team.outsider.email, "viewer")
    if state == "expired":
        team.db.data[f'oe_invitations/{invite["id"]}']["expires_at"] = "2000-01-01T00:00:00+00:00"
    elif state == "revoked":
        team.store.revoke_invite(team.pid, invite["id"], team.owner)
    else:
        team.db.data[f"oe_projects/{team.pid}"]["invitations"] = []
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        team.store.accept(invite["id"], team.outsider)
    assert caught.value.code == "INVITATION_INVALID"
    assert team.db.data == before


def test_invitation_and_membership_management_stay_in_their_project(team):
    invite = team.store.invite(team.pid, team.owner, team.outsider.email, "viewer")
    another = team.store.create_project(team.owner, "Another project")
    with pytest.raises(TeamError) as caught:
        team.store.revoke_invite(another["id"], invite["id"], team.owner)
    assert caught.value.code == "NOT_FOUND"
    assert team.db.get(f'oe_invitations/{invite["id"]}')["status"] == "pending"
    assert team.store.members(team.pid, team.viewer)["invitations"] == []


@pytest.mark.parametrize("role", [None, "viewer", "editor", "owner"])
def test_owner_cannot_be_removed_demoted_or_transferred(team, role):
    with pytest.raises(TeamError) as caught:
        team.store.change_member(team.pid, team.owner.uid, team.owner, role)
    assert caught.value.code == "OWNER_PROTECTED"
    assert team.store.project(team.pid, team.owner)["owner_uid"] == team.owner.uid


def test_demotion_and_removal_apply_to_existing_identity_immediately(team):
    team.store.change_member(team.pid, team.editor.uid, team.owner, "viewer")
    assert team.store.project(team.pid, team.editor)["members"][team.editor.uid]["role"] == "viewer"
    with pytest.raises(TeamError) as caught:
        team.store.save_script(team.pid, team.editor, "print(1)", 0)
    assert caught.value.code == "ROLE_REQUIRED"
    team.store.change_member(team.pid, team.editor.uid, team.owner)
    with pytest.raises(TeamError) as caught:
        team.store.project(team.pid, team.editor)
    assert caught.value.code == "NOT_FOUND"
    assert team.store.projects(team.editor) == []


def test_stale_draft_write_is_rejected_without_losing_saved_code(team):
    saved = team.store.save_script(team.pid, team.owner, "first = 1", 0)
    assert saved["version"] == 1
    with pytest.raises(TeamError) as caught:
        team.store.save_script(team.pid, team.editor, "stale = 2", 0)
    assert caught.value.code == "VERSION_CONFLICT"
    assert team.store.script(team.pid, team.viewer)["code"] == "first = 1"


def race(functions):
    barrier = Barrier(len(functions))
    def call(function):
        barrier.wait(timeout=5)
        try:
            return ("ok", function())
        except TeamError as exc:
            return (exc.code, None)
    with ThreadPoolExecutor(max_workers=len(functions)) as pool:
        return list(pool.map(call, functions))


def test_concurrent_draft_writes_have_exactly_one_winner(team):
    outcomes = race([
        lambda: team.store.save_script(team.pid, team.owner, "owner = 1", 0),
        lambda: team.store.save_script(team.pid, team.editor, "editor = 2", 0),
    ])
    assert sorted(status for status, _ in outcomes) == ["VERSION_CONFLICT", "ok"]
    winner = next(value for status, value in outcomes if status == "ok")
    assert team.store.script(team.pid, team.viewer) == winner


def test_concurrent_invitation_acceptance_cannot_replay_membership_creation(team):
    invite = team.store.invite(team.pid, team.owner, team.outsider.email, "editor")
    outcomes = race([lambda: team.store.accept(invite["id"], team.outsider),
                     lambda: team.store.accept(invite["id"], team.outsider)])
    assert sorted(status for status, _ in outcomes) == ["INVITATION_INVALID", "ok"]
    assert team.db.get(f"oe_users/{team.outsider.uid}")["project_ids"] == [team.pid]


def test_project_limit_is_atomic_for_simultaneous_last_slot(team):
    for index in range(3):
        team.store.create_project(team.owner, f"Existing {index}")
    outcomes = race([
        lambda: team.store.create_project(team.owner, "Last slot A"),
        lambda: team.store.create_project(team.owner, "Last slot B"),
    ])
    assert sorted(status for status, _ in outcomes) == ["PROJECT_LIMIT", "ok"]
    assert len(team.store.projects(team.owner)) == 5


def test_store_has_no_run_lifecycle_after_cloud_execution_retirement(team):
    for name in ('begin_run', 'update_run', 'finish_run', 'request_cancel',
                 'claim_run_dispatch', 'attach_run_execution'):
        assert not hasattr(team.store, name)
    legacy = {'id': 'e' * 32, 'state': 'running', 'created_at': '2026-10-02T10:00:00+00:00'}
    team.db.put(f'oe_projects/{team.pid}/runs/{legacy["id"]}', legacy)
    assert team.store.run(team.pid, legacy['id']) == legacy
    assert team.store.runs(team.pid, team.viewer) == [legacy]


def test_file_quota_and_duplicate_names_do_not_mutate_project(team):
    metadata = {"id": "full", "name": "large.csv", "size_bytes": 64 * 1024**2}
    team.store.add_file(team.pid, team.editor, metadata)
    with pytest.raises(TeamError) as caught:
        team.store.add_file(team.pid, team.editor, {"id": "extra", "name": "extra.csv", "size_bytes": 1})
    assert caught.value.code == "FILE_LIMIT"
    with pytest.raises(TeamError) as caught:
        team.store.add_file(team.pid, team.editor, metadata)
    assert caught.value.code == "FILE_EXISTS"
    assert len(team.store.project(team.pid, team.owner)["files"]) == 1


def test_filename_rejects_reserved_and_escaping_names():
    for name in ["../escape", "/absolute", "a\\b.csv", "a:b.csv", ".hidden", "a\x00b", "a\nb", "a\x7fb.csv",
                 " analysis.csv", "analysis.py", "manifest.json", "result.json", "ğ" * 91]:
        with pytest.raises(TeamError) as caught:
            filename(name)
        assert caught.value.code == "INVALID_FILENAME"
    assert filename("ölçümler.csv") == "ölçümler.csv"


def test_latest_history_remains_current_after_more_than_one_thousand_runs(team):
    for index in range(1100):
        run = {"id": str(index),
               "created_at": f"2026-10-01T00:{index // 60:02d}:{index % 60:02d}+00:00"}
        team.db.put(f"oe_projects/{team.pid}/runs/run-{index}", run)
    assert [run["id"] for run in team.store.runs(team.pid, team.viewer)] == [
        str(index) for index in range(1050, 1100)]


def test_memory_transaction_rolls_back_all_prior_mutations():
    db = MemoryDocuments()
    db.put("records/original", {"value": 1})
    def fail(transaction):
        transaction.delete("records/original")
        transaction.put("records/new", {"value": 2})
        raise TeamError("FAILED", "Failed")
    with pytest.raises(TeamError):
        db.atomic(fail)
    assert db.data == {"records/original": {"value": 1}}


def test_firestore_adapter_stages_writes_until_every_read_finishes(monkeypatch):
    firestore = pytest.importorskip("google.cloud.firestore")
    monkeypatch.setattr(firestore, "transactional", lambda function: function)
    events = []
    class Transaction:
        def set(self, reference, value):
            events.append(("write", reference.path, value))
        def delete(self, reference):
            events.append(("delete", reference.path))
    transaction = Transaction()
    class Reference:
        def __init__(self, path):
            self.path = path
        def get(self, *, transaction):
            assert not any(event[0] in {"write", "delete"} for event in events)
            events.append(("read", self.path))
            return SimpleNamespace(to_dict=lambda: {"value": self.path})
    documents = FirestoreDocuments.__new__(FirestoreDocuments)
    documents.client = SimpleNamespace(document=Reference, transaction=lambda **kwargs: transaction)
    def change(batch):
        batch.put("records/first", {"value": "staged"})
        assert batch.get("records/first") == {"value": "staged"}
        assert batch.get("records/other") == {"value": "records/other"}
        batch.delete("records/deleted")
        return "done"
    assert documents.atomic(change) == "done"
    assert events == [("read", "records/other"), ("write", "records/first", {"value": "staged"}),
                      ("delete", "records/deleted")]


def test_project_rename_only_changes_name_metadata_pending_labels_and_one_audit(team, monkeypatch):
    other = team.store.create_project(team.owner, 'Other project')['id']
    team.store.save_script(team.pid, team.owner, 'unsaved = (\n    42', 0)
    team.store.save_environment(team.pid, team.owner, package_manifest(), 0)
    team.store.add_file(team.pid, team.owner, {'id': 'd' * 32, 'name': 'data.csv', 'size_bytes': 42})
    run = {'id': 'e' * 32, 'state': 'running', 'created_at': '2026-10-02T10:00:00+00:00'}
    team.db.put(f'oe_projects/{team.pid}/runs/{run["id"]}', run)
    project = team.db.get(f'oe_projects/{team.pid}')
    project['active_run'] = run['id']  # Retired cloud backend marker, preserved as stored.
    team.db.put(f'oe_projects/{team.pid}', project)
    pending = team.store.invite(team.pid, team.owner, 'pending@example.com', 'editor')
    expired = team.store.invite(team.pid, team.owner, 'expired@example.com', 'viewer')
    expired_path = f'oe_invitations/{expired["id"]}'
    expired_record = team.db.get(expired_path)
    expired_record['expires_at'] = '2000-01-01T00:00:00+00:00'
    team.db.put(expired_path, expired_record)
    before = deepcopy(team.db.data)
    timestamp = '2028-01-01T00:00:00+00:00'
    monkeypatch.setattr('openecon.team_store.now', lambda: timestamp)
    result = team.store.rename_project(team.pid, team.owner, '  Türkiye Araştırması 👩‍💻  ', 0)
    assert result['name'] == 'Türkiye Araştırması 👩‍💻'
    assert result['id'] == team.pid and result['name_version'] == 1
    assert result['updated_at'] == timestamp
    assert next(p for p in team.store.projects(team.viewer) if p['id'] == team.pid)['name'] == result['name']
    assert team.store.project(other, team.owner) == before[f'oe_projects/{other}']
    expected = deepcopy(before)
    expected[f'oe_projects/{team.pid}'].update(name=result['name'], name_version=1, updated_at=timestamp)
    for item in (pending, expired):
        expected[f'oe_invitations/{item["id"]}']['project_name'] = result['name']
    added = set(team.db.data) - set(before)
    assert len(added) == 1
    audit_path = added.pop()
    audit = team.db.data[audit_path]
    assert audit['action'] == 'project.renamed' and audit['subject'] == result['name']
    assert audit['actor_uid'] == team.owner.uid
    expected[audit_path] = audit
    assert team.db.data == expected
    assert team.store.project(team.pid, team.owner)['active_run'] == run['id']
    assert team.store.script(team.pid, team.viewer)['code'] == 'unsaved = (\n    42'
    assert team.store.invitation_summary(team.db.get(f'oe_invitations/{pending["id"]}'))['url'] == pending['url']


def test_project_rename_legacy_zero_version_and_same_name_idempotence(team):
    path = f'oe_projects/{team.pid}'
    legacy = team.db.get(path)
    legacy.pop('name_version')
    team.db.put(path, legacy)
    assert team.store.projects(team.owner)[0]['name_version'] == 0
    before = deepcopy(team.db.data)
    unchanged = team.store.rename_project(team.pid, team.owner, ' First project ', 0)
    assert unchanged['name_version'] == 0 and team.db.data == before
    saved = team.store.rename_project(team.pid, team.owner, 'Renamed', 0)
    assert saved['name_version'] == 1
    before = deepcopy(team.db.data)
    assert team.store.rename_project(team.pid, team.owner, 'Renamed', 1) == saved
    assert team.db.data == before
    with pytest.raises(TeamError) as caught:
        team.store.rename_project(team.pid, team.owner, 'Renamed', 0)
    assert caught.value.code == 'VERSION_CONFLICT' and caught.value.status_code == 409
    assert team.db.data == before


@pytest.mark.parametrize('name', [None, True, 1, [], {}, '', '   ', 'a' * 101,
                                'name\n', '\tname', 'name\x00', 'name\x7f', 'name\x85',
                                'name\u2028', 'name\u2029', 'name\ud800'])
def test_project_rename_rejects_invalid_names_without_writing(team, name):
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        team.store.rename_project(team.pid, team.owner, name, 0)
    assert caught.value.code == 'INVALID_PROJECT' and caught.value.status_code == 422
    assert team.db.data == before


@pytest.mark.parametrize('version', [None, True, False, -1, 1.0, 0.0, '0', [], 2**53])
def test_project_rename_version_is_a_strict_safe_nonnegative_integer(team, version):
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        team.store.rename_project(team.pid, team.owner, 'New name', version)
    assert caught.value.code == 'INVALID_PROJECT' and caught.value.status_code == 422
    assert team.db.data == before


def test_project_rename_unicode_limit_counts_characters_after_trim(team):
    name = '😀' * 100
    assert team.store.rename_project(team.pid, team.owner, '  ' + name + '  ', 0)['name'] == name
    with pytest.raises(TeamError):
        team.store.rename_project(team.pid, team.owner, name + '😀', 1)


@pytest.mark.parametrize('identity,code,status', [('editor', 'ROLE_REQUIRED', 403),
                                               ('viewer', 'ROLE_REQUIRED', 403),
                                               ('outsider', 'NOT_FOUND', 404),
                                               ('unverified', 'NOT_FOUND', 404)])
def test_project_rename_requires_verified_owner_membership(team, identity, code, status):
    user = replace(team.owner, email_verified=False) if identity == 'unverified' else getattr(team, identity)
    before = deepcopy(team.db.data)
    with pytest.raises(TeamError) as caught:
        team.store.rename_project(team.pid, user, 'No access', 0)
    assert (caught.value.code, caught.value.status_code) == (code, status)
    assert team.db.data == before


@pytest.mark.parametrize('role', [None, 'editor', 'viewer'])
def test_project_rename_rechecks_owner_membership_inside_transaction(team, role):
    original = team.db.atomic
    entered = False
    def changed_before_transaction(function):
        nonlocal entered
        if not entered:
            entered = True
            path = f'oe_projects/{team.pid}'
            project = team.db.get(path)
            if role is None:
                project['members'].pop(team.owner.uid)
            else:
                project['members'][team.owner.uid]['role'] = role
            team.db.put(path, project)
        return original(function)
    team.db.atomic = changed_before_transaction
    with pytest.raises(TeamError) as caught:
        team.store.rename_project(team.pid, team.owner, 'Cannot rename', 0)
    assert caught.value.code == ('NOT_FOUND' if role is None else 'ROLE_REQUIRED')
    assert team.db.get(f'oe_projects/{team.pid}')['name'] == 'First project'
    assert not any(event['action'] == 'project.renamed' for event in team.db.scan(f'oe_projects/{team.pid}/audit'))


def test_project_rename_concurrent_cas_has_one_winner_and_one_audit(team):
    outcomes = race([
        lambda: team.store.rename_project(team.pid, team.owner, 'First name', 0),
        lambda: team.store.rename_project(team.pid, team.owner, 'Second name', 0),
    ])
    assert sorted(status for status, _ in outcomes) == ['VERSION_CONFLICT', 'ok']
    winner = next(value for status, value in outcomes if status == 'ok')
    assert team.store.project(team.pid, team.owner)['name'] == winner['name']
    assert winner['name_version'] == 1
    assert len([event for event in team.db.scan(f'oe_projects/{team.pid}/audit')
                if event['action'] == 'project.renamed']) == 1


def test_project_rename_and_invitation_creation_commit_consistent_labels(team):
    outcomes = race([
        lambda: team.store.rename_project(team.pid, team.owner, 'Current project name', 0),
        lambda: team.store.invite(team.pid, team.owner, 'concurrent@example.com', 'viewer'),
    ])
    assert all(status == 'ok' for status, _ in outcomes)
    invite = next(value for _, value in outcomes if 'project_name' in value)
    stored = team.db.get(f'oe_invitations/{invite["id"]}')
    assert stored['project_name'] == 'Current project name'
    assert stored['status'] == 'pending' and stored['role'] == 'viewer'
    assert stored['expires_at'] == invite['expires_at']


def test_project_rename_does_not_lose_concurrent_script_commit(team):
    outcomes = race([
        lambda: team.store.rename_project(team.pid, team.owner, 'Renamed', 0),
        lambda: team.store.save_script(team.pid, team.editor, 'exact = (\n    7', 0),
    ])
    assert all(status == 'ok' for status, _ in outcomes)
    assert team.store.project(team.pid, team.owner)['name_version'] == 1
    assert team.store.project(team.pid, team.owner)['name'] == 'Renamed'
    assert team.store.script(team.pid, team.viewer)['code'] == 'exact = (\n    7'


def test_project_rename_cannot_relabel_a_different_projects_invitation(team):
    other = team.store.create_project(team.owner, 'Other project')['id']
    invite = team.store.invite(other, team.owner, 'other@example.com', 'editor')
    path = f'oe_projects/{team.pid}'
    corrupted = team.db.get(path)
    corrupted['invitations'].append(invite['id'])
    team.db.put(path, corrupted)
    before = team.db.get(f'oe_invitations/{invite["id"]}')
    team.store.rename_project(team.pid, team.owner, 'This project', 0)
    assert team.db.get(f'oe_invitations/{invite["id"]}') == before


def test_project_rename_rolls_back_project_and_all_labels_when_invitation_write_fails(team):
    first = team.store.invite(team.pid, team.owner, 'first@example.com', 'editor')
    second = team.store.invite(team.pid, team.owner, 'second@example.com', 'viewer')
    before = deepcopy(team.db.data)
    original = team.db.put
    def fail_second(path, value):
        if path == f'oe_invitations/{second["id"]}':
            raise TeamError('FAILED', 'Write failed')
        return original(path, value)
    team.db.put = fail_second
    with pytest.raises(TeamError, match='Write failed'):
        team.store.rename_project(team.pid, team.owner, 'Attempted', 0)
    assert team.db.data == before
    assert team.db.get(f'oe_invitations/{first["id"]}')['project_name'] == 'First project'


def test_firestore_project_rename_reads_authorization_and_invitations_before_any_write(team, monkeypatch):
    firestore = pytest.importorskip('google.cloud.firestore')
    monkeypatch.setattr(firestore, 'transactional', lambda function: function)
    invitation = team.store.invite(team.pid, team.owner, 'pending@example.com', 'editor')
    records = deepcopy(team.db.data)
    events = []
    required = {f'oe_projects/{team.pid}', f'oe_invitations/{invitation["id"]}'}
    class Transaction:
        def set(self, reference, value):
            assert required.issubset({path for operation, path in events if operation == 'read'})
            events.append(('write', reference.path))
            records[reference.path] = deepcopy(value)
    transaction = Transaction()
    class Reference:
        def __init__(self, path):
            self.path = path
        def get(self, *, transaction):
            assert not any(operation == 'write' for operation, _ in events)
            events.append(('read', self.path))
            return SimpleNamespace(to_dict=lambda: deepcopy(records.get(self.path)))
    documents = FirestoreDocuments.__new__(FirestoreDocuments)
    documents.client = SimpleNamespace(document=Reference, transaction=lambda **kwargs: transaction)
    store = TeamStore(documents, owner_email=team.owner.email, public_origin='https://app.example.com')
    summary = store.rename_project(team.pid, team.owner, 'Atomic cloud name', 0)
    assert summary['name_version'] == 1
    assert records[f'oe_projects/{team.pid}']['name'] == 'Atomic cloud name'
    assert records[f'oe_invitations/{invitation["id"]}']['project_name'] == 'Atomic cloud name'
    assert [operation for operation, _ in events][:2] == ['read', 'read']


def test_firestore_history_query_uses_project_projection_bounds_and_complete_cursor(monkeypatch):
    from google.auth.credentials import AnonymousCredentials
    from google.cloud.firestore_v1.client import Client
    from google.cloud.firestore_v1.query import Query
    from openecon.team_history import HISTORY_FIELDS
    documents = FirestoreDocuments.__new__(FirestoreDocuments)
    documents.client = Client(project='history-fixture', credentials=AnonymousCredentials())
    queries = []
    def stream(query):
        queries.append(query._to_protobuf())
        return []
    monkeypatch.setattr(Query, 'stream', stream)
    collection = 'oe_projects/project/runs'
    snapshot, timestamp = '2026-10-06T00:00:00+00:00', '2026-10-01T00:00:00+00:00'
    documents.history_scan(collection, snapshot=snapshot, before=(timestamp, 'a' * 32), limit=101)
    query = queries[0]
    assert [field.field_path for field in query.select.fields] == list(HISTORY_FIELDS)
    assert query.from_[0].collection_id == 'runs' and not query.from_[0].all_descendants
    assert [order.field.field_path for order in query.order_by] == ['created_at', '__name__']
    assert all(order.direction == 2 for order in query.order_by)
    assert query.limit == 101 and query.offset == 0
    assert query.where.field_filter.field.field_path == 'created_at'
    assert query.where.field_filter.value.string_value == snapshot
    assert query.start_at.values[0].string_value == timestamp and not query.start_at.before
    assert query.start_at.values[1].reference_value.endswith(f'/documents/{collection}/' + 'a' * 32)
