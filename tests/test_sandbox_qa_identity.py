"""Temporary QA IAM grants retain exact scope, bounded expiry and safe cleanup."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'sandbox_qa_identity', Path(__file__).parents[1] / 'scripts/sandbox_qa_identity.py')
identity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(identity)
NOW = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)
OTHER = {'role': 'roles/viewer', 'members': ['user:unrelated@example.com']}


class PolicyClient:
    def __init__(self, policy=None, *, failure=None, commit_on_failure=False):
        self.current = deepcopy(policy or {'version': 1, 'etag': 'baseline-etag', 'bindings': [OTHER]})
        self.writes = []
        self.failure = failure
        self.commit_on_failure = commit_on_failure

    def policy(self):
        return deepcopy(self.current)

    def call(self, method, path, body):
        assert (method, path) == ('POST', identity.ACCOUNT_PATH + ':setIamPolicy')
        assert body['policy']['etag'] == self.current['etag']
        self.writes.append(deepcopy(body))
        failure, self.failure = self.failure, None
        if failure is None or self.commit_on_failure:
            self.current = deepcopy(body['policy'])
            self.current['etag'] = 'fresh-etag'
        if failure is not None:
            raise failure
        return self.policy()


def state(client=None, minutes=60):
    client = client or PolicyClient()
    return identity.make_state(client.policy(), minutes, now=NOW)


def test_grant_uses_exact_oidc_only_role_unique_condition_and_retains_other_access():
    client = PolicyClient()
    first, second = state(client), state(client)
    assert first['binding']['condition']['title'] != second['binding']['condition']['title']
    assert first['binding']['role'] == identity.ROLE
    assert first['binding']['members'] == ['user:' + identity.OPERATOR]
    assert identity.validate_state(first) == NOW + timedelta(minutes=60)
    identity.reconcile_binding(client, first['binding'], grant=True)
    assert client.current['version'] == 3
    assert client.current['bindings'] == [OTHER, first['binding']]
    assert client.writes[0]['policy']['etag'] == 'baseline-etag'
    identity.reconcile_binding(client, first['binding'], grant=True)
    assert len(client.writes) == 1


@pytest.mark.parametrize('minutes', [0, 121, True, 3.5])
def test_unbounded_or_invalid_expiry_is_rejected(minutes):
    with pytest.raises(ValueError):
        state(minutes=minutes)


@pytest.mark.parametrize('commit', [False, True])
def test_unknown_grant_outcome_is_reconciled_before_retry(commit):
    client = PolicyClient(failure=identity.CloudError(), commit_on_failure=commit)
    owned = state(client)['binding']
    identity.reconcile_binding(client, owned, grant=True)
    assert identity.binding_present(client.current, owned)
    assert len(client.writes) == (1 if commit else 2)
    assert client.current['bindings'].count(owned) == 1


def test_revoke_uses_fresh_policy_and_removes_only_owned_conditional_binding():
    client = PolicyClient()
    owned = state(client)['binding']
    unrelated_qa = deepcopy(owned)
    unrelated_qa['condition']['title'] = 'openecon-sandbox-qa-' + 'c' * 32
    client.current.update(version=3, etag='current-not-saved-etag',
                          bindings=[OTHER, unrelated_qa, owned], auditConfigs=[{'service': 'allServices'}])
    identity.reconcile_binding(client, owned, grant=False)
    assert client.current['bindings'] == [OTHER, unrelated_qa]
    assert client.current['auditConfigs'] == [{'service': 'allServices'}]
    assert client.writes[0]['policy']['etag'] == 'current-not-saved-etag'


def test_modified_same_title_binding_is_never_removed():
    client = PolicyClient()
    owned = state(client)['binding']
    changed = deepcopy(owned)
    changed['members'].append('user:someone-else@example.com')
    client.current['bindings'].append(changed)
    with pytest.raises(ValueError):
        identity.reconcile_binding(client, owned, grant=False)
    assert client.writes == []


def test_permanent_iam_rejection_does_not_retry_or_replace_unrelated_policy():
    client = PolicyClient(failure=identity.CloudError(403))
    with pytest.raises(identity.CloudError):
        identity.reconcile_binding(client, state(client)['binding'], grant=True)
    assert len(client.writes) == 1
    assert client.current['bindings'] == [OTHER]


@pytest.mark.parametrize('change', [
    lambda s: s.update(project='foreign-project'),
    lambda s: s.update(service_account='foreign-account@example.com'),
    lambda s: s['binding'].update(role='roles/iam.serviceAccountTokenCreator'),
    lambda s: s['binding'].update(members=['allUsers']),
    lambda s: s['binding']['condition'].update(expression='true'),
    lambda s: s.update(expires_at_utc='2026-10-02T15:00:00Z'),
    lambda s: s.update(binding=None),
])
def test_state_cannot_expand_ownership_or_expiry(change):
    saved = state()
    change(saved)
    with pytest.raises(ValueError):
        identity.validate_state(saved)


def test_state_is_private_minimal_and_saved_without_other_policy_or_token(tmp_path):
    path = tmp_path / 'qa-state.json'
    saved = state()
    identity.save_state(saved, path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert identity.read_state(path) == saved
    assert 'unrelated@example.com' not in path.read_text()
    assert set(json.loads(path.read_text())) == set(saved)
    with pytest.raises(FileExistsError):
        identity.save_state(saved, path)
    path.chmod(0o644)
    with pytest.raises(ValueError):
        identity.read_state(path)


def test_other_active_operator_fails_before_token_request(monkeypatch):
    cli = Mock(return_value='someone-else@example.com')
    monkeypatch.setattr(identity.subprocess, 'check_output', cli)
    with pytest.raises(ValueError):
        identity.Client()
    assert cli.call_count == 1


def test_role_verification_rejects_broader_permissions():
    client = object.__new__(identity.Client)
    client.call = Mock(return_value={'includedPermissions': [identity.PERMISSION, 'iam.serviceAccounts.getAccessToken']})
    with pytest.raises(ValueError):
        client.verify_role()


def test_client_rejects_any_outside_service_account_scope():
    client = object.__new__(identity.Client)
    with pytest.raises(ValueError):
        client.call('POST', 'projects/foreign/serviceAccounts/foreign@example.com:setIamPolicy', {})


def test_revoke_main_preserves_saved_state_until_verified_readback(tmp_path, monkeypatch):
    path = tmp_path / 'qa-state.json'
    saved = state()
    identity.save_state(saved, path)
    client = PolicyClient(failure=identity.CloudError(403))
    client.current['bindings'].append(saved['binding'])
    client.verify_role = Mock()
    monkeypatch.setattr(identity, 'STATE', path)
    monkeypatch.setattr(identity, 'Client', lambda: client)
    monkeypatch.setattr('sys.argv', ['sandbox_qa_identity.py', '--revoke'])
    with pytest.raises(identity.CloudError):
        identity.main()
    assert path.exists()
    identity.main()
    assert not path.exists()
    assert client.current['bindings'] == [OTHER]
