"""Manage expiring QA signing access only on the fixed transfer service account.

The operator already administers this project, but cannot create its scoped GCS
capabilities without signBlob. This adds one permission temporarily; it never
grants TokenCreator. Role ownership survives unknown mutation responses, and a
preexisting custom role is never deleted or changed by cleanup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import requests

if __package__:
    from . import sandbox_qa_identity as shared
else:
    import sandbox_qa_identity as shared

PROJECT = shared.PROJECT
OPERATOR = shared.OPERATOR
ACCOUNT = f'openecon-transfer@{PROJECT}.iam.gserviceaccount.com'
ACCOUNT_PATH = f'projects/{PROJECT}/serviceAccounts/{ACCOUNT}'
ROLE_ID = 'openeconSandboxQaSigner'
ROLE = f'projects/{PROJECT}/roles/{ROLE_ID}'
PERMISSION = 'iam.serviceAccounts.signBlob'
STATE = Path('artifacts/team-setup/sandbox-qa-signer.json')
TITLE = 'OpenEcon sandbox QA signing'
DESCRIPTION = 'Temporary sandbox signing verification; expires automatically.'


class Client(shared.Client):
    def call(self, method, path, body=None, *, allow_missing=False, etag=None):
        allowed = {
            ('GET', ROLE), ('POST', f'projects/{PROJECT}/roles'), ('DELETE', ROLE),
            ('POST', ACCOUNT_PATH + ':getIamPolicy'), ('POST', ACCOUNT_PATH + ':setIamPolicy'),
        }
        if (method, path) not in allowed or (etag is not None and (method, path) != ('DELETE', ROLE)):
            raise ValueError('QA signing operations must use the fixed transfer identity and custom role.')
        try:
            response = self.session.request(
                method, 'https://iam.googleapis.com/v1/' + path, json=body,
                params={'etag': etag} if etag is not None else None,
                timeout=25, allow_redirects=False)
        except requests.RequestException:
            raise shared.CloudError() from None
        if allow_missing and response.status_code == 404:
            return None
        if not 200 <= response.status_code < 300:
            raise shared.CloudError(response.status_code)
        try:
            return response.json()
        except ValueError:
            raise shared.CloudError() from None

    def policy(self):
        policy = self.call('POST', ACCOUNT_PATH + ':getIamPolicy',
                           {'options': {'requestedPolicyVersion': 3}})
        if (not isinstance(policy, dict) or not isinstance(policy.get('etag'), str)
                or not policy['etag'] or not isinstance(policy.get('bindings', []), list)):
            raise ValueError('A fresh versioned transfer policy is required.')
        return policy

    def role(self):
        return self.call('GET', ROLE, allow_missing=True)


def verify_role(role, *, allow_deleted=False):
    if (not isinstance(role, dict) or role.get('name') != ROLE
            or role.get('includedPermissions') != [PERMISSION]
            or role.get('stage') != 'GA' or not isinstance(role.get('etag'), str) or not role['etag']
            or role.get('deleted') and not allow_deleted):
        raise ValueError('The QA signer role must contain only the exact signing permission.')


def role_description(state):
    return 'Temporary OpenEcon sandbox QA signer owned by ' + state['binding']['condition']['title']


def make_state(policy, minutes, *, role, now=None):
    if type(minutes) is not int or not 1 <= minutes <= 120:
        raise ValueError('Temporary QA signing access must expire within 120 minutes.')
    created = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    expires = shared.utc_text(created + timedelta(minutes=minutes))
    return {
        'schema_version': 1, 'project': PROJECT, 'service_account': ACCOUNT,
        'operator': OPERATOR, 'original_etag': policy['etag'],
        'created_at_utc': shared.utc_text(created), 'expires_at_utc': expires,
        'role_preexisting': role is not None, 'role_created': False, 'created_role_etag': None,
        'binding': {'role': ROLE, 'members': ['user:' + OPERATOR], 'condition': {
            'title': 'openecon-sandbox-signer-qa-' + uuid4().hex, 'description': DESCRIPTION,
            'expression': f'request.time < timestamp("{expires}")',
        }},
    }


def validate_state(state):
    keys = {'schema_version', 'project', 'service_account', 'operator', 'original_etag',
            'created_at_utc', 'expires_at_utc', 'role_preexisting', 'role_created',
            'created_role_etag', 'binding'}
    if (not isinstance(state, dict) or set(state) != keys or state.get('schema_version') != 1
            or state.get('project') != PROJECT or state.get('service_account') != ACCOUNT
            or state.get('operator') != OPERATOR or not isinstance(state.get('original_etag'), str)
            or not state['original_etag'] or type(state.get('role_preexisting')) is not bool
            or type(state.get('role_created')) is not bool
            or state['role_created'] and state['role_preexisting']
            or state['role_created'] and (not isinstance(state.get('created_role_etag'), str)
                                         or not state['created_role_etag'])
            or not state['role_created'] and state.get('created_role_etag') is not None):
        raise ValueError('QA signer state does not match its fixed ownership scope.')
    binding = state['binding']
    condition = binding.get('condition', {}) if isinstance(binding, dict) else {}
    if (not isinstance(binding, dict) or not isinstance(condition, dict)
            or set(binding) != {'role', 'members', 'condition'}
            or set(condition) != {'title', 'description', 'expression'}
            or binding.get('role') != ROLE or binding.get('members') != ['user:' + OPERATOR]
            or not isinstance(condition.get('title'), str)
            or not re.fullmatch(r'openecon-sandbox-signer-qa-[0-9a-f]{32}', condition['title'])
            or condition.get('description') != DESCRIPTION):
        raise ValueError('Only this helper\'s exact conditional signer binding can be managed.')
    for key in ('created_at_utc', 'expires_at_utc'):
        if not isinstance(state[key], str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', state[key]):
            raise ValueError('QA signing expiry must be an explicit UTC instant.')
    created = datetime.fromisoformat(state['created_at_utc'].replace('Z', '+00:00'))
    expires = datetime.fromisoformat(state['expires_at_utc'].replace('Z', '+00:00'))
    if (not timedelta(minutes=1) <= expires - created <= timedelta(hours=2)
            or condition.get('expression') != f'request.time < timestamp("{state["expires_at_utc"]}")'):
        raise ValueError('QA signing expiry exceeds its fixed contract.')
    return expires


def rewrite_state(state, path=STATE):
    validate_state(state)
    # Refuse replacement of a nonprivate/symlink state. Persist ownership before
    # advancing from role creation to a service-account policy grant.
    shared.read_state(path, validator=validate_state)
    pending = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        shared.save_state(state, pending, validator=validate_state)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def ensure_role(client, state, *, save):
    for _ in range(3):
        existing = client.role()
        if existing is not None:
            verify_role(existing)
            owns = (not state['role_preexisting'] and existing.get('description') == role_description(state))
            if owns:
                if state['role_created'] and existing['etag'] != state['created_role_etag']:
                    raise ValueError('The created QA role changed; unrelated changes were preserved.')
                state.update(role_created=True, created_role_etag=existing['etag'])
            elif not state['role_created']:
                state.update(role_preexisting=True, role_created=False, created_role_etag=None)
            else:
                raise ValueError('The created QA role changed; its ownership cannot be verified.')
            save(state)
            return
        if state['role_preexisting'] or state['role_created']:
            raise ValueError('A previously existing QA role is unavailable; no replacement was created.')
        failure = None
        try:
            client.call('POST', f'projects/{PROJECT}/roles', {'roleId': ROLE_ID, 'role': {
                'title': TITLE, 'stage': 'GA', 'includedPermissions': [PERMISSION],
                'description': role_description(state),
            }})
        except shared.CloudError as error:
            failure = error
        # A lost response may have created the role. Always inspect the exact
        # immutable role ID/permission and nonce before deciding ownership.
        observed = client.role()
        if observed is not None:
            verify_role(observed)
            owns = observed.get('description') == role_description(state)
            state.update(role_created=owns, role_preexisting=not owns,
                         created_role_etag=observed['etag'] if owns else None)
            save(state)
            return
        if failure and failure.status not in {None, 409, 412, 500, 502, 503, 504}:
            raise failure
    raise shared.CloudError()


def remove_created_role(client, state):
    if not state['role_created']:
        return  # Preexisting roles, including concurrent creations, are kept.
    if any(item.get('role') == ROLE for item in client.policy().get('bindings', [])):
        raise ValueError('The QA role still has transfer-policy references; unrelated access was preserved.')
    for _ in range(3):
        role = client.role()
        if role is None:
            return
        verify_role(role, allow_deleted=True)
        if role.get('description') != role_description(state):
            raise ValueError('The QA role ownership changed; the role was preserved.')
        if role.get('deleted'):
            return
        if role.get('etag') != state['created_role_etag'] or role.get('title') != TITLE:
            raise ValueError('The created QA role changed; unrelated changes were preserved.')
        failure = None
        try:
            client.call('DELETE', ROLE, etag=role['etag'])
        except shared.CloudError as error:
            failure = error
        observed = client.role()
        if observed is None or observed.get('deleted'):
            return
        if failure and failure.status not in {None, 409, 412, 500, 502, 503, 504}:
            raise failure
    raise shared.CloudError()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--grant', action='store_true')
    actions.add_argument('--revoke', action='store_true')
    parser.add_argument('--minutes', type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.minutes <= 120:
        parser.error('Temporary QA signing access must expire within 120 minutes.')
    client = Client()
    if args.grant:
        if STATE.exists():
            state = shared.read_state(STATE, validator=validate_state)
        else:
            role = client.role()
            if role is not None:
                verify_role(role)
            state = make_state(client.policy(), args.minutes, role=role)
            shared.save_state(state, STATE, validator=validate_state)
        if validate_state(state) <= datetime.now(timezone.utc):
            raise SystemExit('Existing QA signing access expired; revoke its saved binding before a new grant.')
        ensure_role(client, state, save=lambda saved: rewrite_state(saved, STATE))
        shared.reconcile_binding(client, state['binding'], grant=True, account_path=ACCOUNT_PATH)
        print(json.dumps({'action': 'grant', 'binding_verified': True,
                          'role_permission_verified': True, 'expires_at_utc': state['expires_at_utc']}))
    else:
        if not STATE.exists():
            raise SystemExit('No owned QA signer state exists; no IAM resource was changed.')
        state = shared.read_state(STATE, validator=validate_state)
        # Reconcile role ownership if a create response/state update was lost.
        if not state['role_preexisting'] and not state['role_created']:
            role = client.role()
            if role is not None and role.get('description') == role_description(state):
                verify_role(role, allow_deleted=True)
                state.update(role_created=True, created_role_etag=role['etag'])
                rewrite_state(state, STATE)
        shared.reconcile_binding(client, state['binding'], grant=False, account_path=ACCOUNT_PATH)
        remove_created_role(client, state)
        STATE.unlink()
        print(json.dumps({'action': 'revoke', 'owned_binding_absent': True,
                          'created_role_removed': state['role_created'], 'cleanup_complete': True}))


if __name__ == '__main__':
    main()
