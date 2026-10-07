"""Temporary transfer signing preserves role ownership and unrelated IAM state."""
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from scripts import sandbox_qa_signer as signer

NOW = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)
BASELINE = {'role': f'projects/{signer.PROJECT}/roles/openeconBlobSigner',
            'members': [f'serviceAccount:openecon-control@{signer.PROJECT}.iam.gserviceaccount.com']}


class Client:
    def __init__(self, role=None, *, create_failure=None, create_commits=False,
                 delete_failure=None, delete_commits=False):
        self.current = {'version': 3, 'etag': 'baseline-policy-etag', 'bindings': [deepcopy(BASELINE)]}
        self.current_role = deepcopy(role)
        self.calls = []
        self.create_failure, self.create_commits = create_failure, create_commits
        self.delete_failure, self.delete_commits = delete_failure, delete_commits

    def policy(self):
        return deepcopy(self.current)

    def role(self):
        return deepcopy(self.current_role)

    def call(self, method, path, body=None, *, etag=None):
        self.calls.append((method, path, deepcopy(body), etag))
        if (method, path) == ('POST', f'projects/{signer.PROJECT}/roles'):
            failure, self.create_failure = self.create_failure, None
            if failure is None or self.create_commits:
                self.current_role = {'name': signer.ROLE, 'etag': 'created-role-etag', **deepcopy(body['role'])}
            if failure:
                raise failure
            return self.role()
        if (method, path) == ('POST', signer.ACCOUNT_PATH + ':setIamPolicy'):
            assert body['policy']['etag'] == self.current['etag']
            self.current = deepcopy(body['policy'])
            self.current['etag'] = 'fresh-policy-etag'
            return self.policy()
        if (method, path) == ('DELETE', signer.ROLE):
            assert etag == self.current_role['etag']
            failure, self.delete_failure = self.delete_failure, None
            if failure is None or self.delete_commits:
                self.current_role.update(deleted=True, etag='deleted-role-etag')
            if failure:
                raise failure
            return self.role()
        raise AssertionError('Unexpected IAM request scope')


def state(client):
    return signer.make_state(client.policy(), 60, role=client.role(), now=NOW)


def existing_role():
    return {'name': signer.ROLE, 'includedPermissions': [signer.PERMISSION], 'stage': 'GA',
            'etag': 'preexisting-role-etag', 'title': signer.TITLE, 'description': 'Existing unrelated owner'}


def test_new_role_exact_permission_and_transfer_only_binding_then_verified_cleanup():
    client = Client()
    saved = state(client)
    persist = Mock()
    signer.ensure_role(client, saved, save=persist)
    assert saved['role_created'] and not saved['role_preexisting']
    assert client.current_role['includedPermissions'] == ['iam.serviceAccounts.signBlob']
    signer.shared.reconcile_binding(client, saved['binding'], grant=True, account_path=signer.ACCOUNT_PATH)
    assert client.current['bindings'] == [BASELINE, saved['binding']]
    signer.shared.reconcile_binding(client, saved['binding'], grant=False, account_path=signer.ACCOUNT_PATH)
    signer.remove_created_role(client, saved)
    assert client.current['bindings'] == [BASELINE]
    assert client.current_role['deleted']
    assert client.calls[-1][:2] == ('DELETE', signer.ROLE)
    assert client.calls[-1][3] == saved['created_role_etag']


def test_preexisting_exact_role_is_never_deleted_or_changed():
    role = existing_role()
    client = Client(role)
    saved = state(client)
    signer.ensure_role(client, saved, save=Mock())
    signer.shared.reconcile_binding(client, saved['binding'], grant=True, account_path=signer.ACCOUNT_PATH)
    signer.shared.reconcile_binding(client, saved['binding'], grant=False, account_path=signer.ACCOUNT_PATH)
    signer.remove_created_role(client, saved)
    assert client.current_role == role
    assert all(call[:2] == ('POST', signer.ACCOUNT_PATH + ':setIamPolicy') for call in client.calls)


def test_concurrently_created_foreign_role_is_never_claimed_or_deleted():
    class ConcurrentClient(Client):
        def call(self, method, path, body=None, *, etag=None):
            if (method, path) == ('POST', f'projects/{signer.PROJECT}/roles'):
                self.current_role = existing_role()
                raise signer.shared.CloudError(409)
            return super().call(method, path, body, etag=etag)

    client = ConcurrentClient()
    saved = state(client)
    signer.ensure_role(client, saved, save=Mock())
    assert saved['role_preexisting'] and not saved['role_created']
    signer.remove_created_role(client, saved)
    assert client.current_role == existing_role()
    assert not any(call[0] == 'DELETE' for call in client.calls)


@pytest.mark.parametrize('commits', [False, True])
def test_unknown_role_create_is_read_back_and_ownership_retained(commits):
    client = Client(create_failure=signer.shared.CloudError(), create_commits=commits)
    saved = state(client)
    signer.ensure_role(client, saved, save=Mock())
    assert saved['role_created']
    assert client.current_role['description'] == signer.role_description(saved)
    assert len(client.calls) == (1 if commits else 2)


@pytest.mark.parametrize('commits', [False, True])
def test_unknown_role_delete_is_read_back_before_retry(commits):
    client = Client(delete_failure=signer.shared.CloudError(), delete_commits=commits)
    saved = state(client)
    signer.ensure_role(client, saved, save=Mock())
    signer.remove_created_role(client, saved)
    assert client.current_role['deleted']
    deletes = [call for call in client.calls if call[0] == 'DELETE']
    assert len(deletes) == (1 if commits else 2)


def test_unrelated_role_metadata_change_blocks_deletion_and_new_ownership_adoption():
    client = Client()
    saved = state(client)
    signer.ensure_role(client, saved, save=Mock())
    client.current_role.update(etag='unrelated-change-etag', title='Unrelated updated title')
    before = deepcopy(client.current_role)
    with pytest.raises(ValueError):
        signer.ensure_role(client, saved, save=Mock())
    with pytest.raises(ValueError):
        signer.remove_created_role(client, saved)
    assert client.current_role == before
    assert not any(call[0] == 'DELETE' for call in client.calls)


def test_unrelated_transfer_reference_preserved_and_created_role_not_deleted():
    client = Client()
    saved = state(client)
    signer.ensure_role(client, saved, save=Mock())
    client.current['bindings'].append({'role': signer.ROLE, 'members': ['user:unrelated@example.com']})
    with pytest.raises(ValueError):
        signer.remove_created_role(client, saved)
    assert not any(call[0] == 'DELETE' for call in client.calls)


@pytest.mark.parametrize('change', [
    lambda s: s.update(service_account=signer.shared.ACCOUNT),
    lambda s: s['binding'].update(role='roles/iam.serviceAccountTokenCreator'),
    lambda s: s['binding'].update(members=['allUsers']),
    lambda s: s.update(expires_at_utc='2026-10-02T15:00:00Z'),
    lambda s: s.update(role_created=True, role_preexisting=True),
])
def test_state_rejects_expanded_scope_or_expiry(change):
    saved = state(Client())
    change(saved)
    with pytest.raises(ValueError):
        signer.validate_state(saved)


def test_signer_client_rejects_control_identity_and_broad_role_path():
    client = object.__new__(signer.Client)
    for method, path in [('POST', signer.shared.ACCOUNT_PATH + ':setIamPolicy'),
                         ('GET', 'roles/iam.serviceAccountTokenCreator')]:
        with pytest.raises(ValueError):
            client.call(method, path)


def test_preexisting_broad_role_is_rejected_without_mutation():
    role = existing_role()
    role['includedPermissions'].append('iam.serviceAccounts.getAccessToken')
    client = Client(role)
    with pytest.raises(ValueError):
        signer.ensure_role(client, state(client), save=Mock())
    assert client.calls == []


def test_signer_state_private_atomic_rewrite_and_shared_identity_state_unchanged(tmp_path):
    path = tmp_path / 'signer-state.json'
    client = Client()
    saved = state(client)
    signer.shared.save_state(saved, path, validator=signer.validate_state)
    signer.ensure_role(client, saved, save=lambda changed: signer.rewrite_state(changed, path))
    assert signer.shared.read_state(path, validator=signer.validate_state) == saved
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_revoke_main_preserves_state_when_role_cleanup_fails(tmp_path, monkeypatch):
    path = tmp_path / 'signer-state.json'
    client = Client(delete_failure=signer.shared.CloudError(403))
    saved = state(client)
    signer.ensure_role(client, saved, save=Mock())
    signer.shared.save_state(saved, path, validator=signer.validate_state)
    client.current['bindings'].append(saved['binding'])
    monkeypatch.setattr(signer, 'STATE', path)
    monkeypatch.setattr(signer, 'Client', lambda: client)
    monkeypatch.setattr('sys.argv', ['sandbox_qa_signer.py', '--revoke'])
    with pytest.raises(signer.shared.CloudError):
        signer.main()
    assert path.exists()
    assert client.current['bindings'] == [BASELINE]
    signer.main()
    assert not path.exists()
    assert client.current_role['deleted']
