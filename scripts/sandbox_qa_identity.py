"""Grant/revoke one expiring OIDC-only QA binding on the fixed control identity.

No access tokens, keys, full IAM policies or private configuration are saved.
Interrupted requests are reconciled from fresh policies before a bounded retry.
The unique conditional binding expires within two hours even if cleanup fails.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from uuid import uuid4

import requests

PROJECT = 'openecon-workbench'
ACCOUNT = f'openecon-control@{PROJECT}.iam.gserviceaccount.com'
OPERATOR = 'anilsen@bluearf.com'
ROLE = 'roles/iam.serviceAccountOpenIdTokenCreator'
PERMISSION = 'iam.serviceAccounts.getOpenIdToken'
ACCOUNT_PATH = f'projects/{PROJECT}/serviceAccounts/{ACCOUNT}'
STATE = Path('artifacts/team-setup/sandbox-qa-identity.json')
DESCRIPTION = 'Temporary sandbox integration verification; expires automatically.'


class CloudError(RuntimeError):
    def __init__(self, status=None):
        self.status = status
        super().__init__('QA identity request failed; private provider details were withheld.')


class Client:
    def __init__(self):
        try:
            active = subprocess.check_output(
                ['gcloud', 'config', 'get-value', 'core/account', '--project=' + PROJECT],
                text=True, stderr=subprocess.PIPE).strip()
            if active != OPERATOR:
                raise ValueError('The fixed authorized QA operator must be the active Cloud account.')
            token = subprocess.check_output(
                ['gcloud', 'auth', 'print-access-token', '--account=' + OPERATOR, '--project=' + PROJECT],
                text=True, stderr=subprocess.PIPE).strip()
        except subprocess.CalledProcessError:
            raise CloudError() from None
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({'Authorization': 'Bearer ' + token})

    def call(self, method, path, body=None):
        if (method, path) not in {
            ('GET', ROLE), ('POST', ACCOUNT_PATH + ':getIamPolicy'),
            ('POST', ACCOUNT_PATH + ':setIamPolicy'),
        }:
            raise ValueError('QA identity operations are confined to their fixed service account and role.')
        try:
            response = self.session.request(
                method, 'https://iam.googleapis.com/v1/' + path, json=body,
                timeout=25, allow_redirects=False)
        except requests.RequestException:
            raise CloudError() from None
        if not 200 <= response.status_code < 300:
            raise CloudError(response.status_code)
        try:
            return response.json()
        except ValueError:
            raise CloudError() from None

    def policy(self):
        policy = self.call('POST', ACCOUNT_PATH + ':getIamPolicy',
                           {'options': {'requestedPolicyVersion': 3}})
        if (not isinstance(policy, dict) or not isinstance(policy.get('etag'), str)
                or not policy['etag'] or not isinstance(policy.get('bindings', []), list)):
            raise ValueError('A fresh versioned service-account policy is required.')
        return policy

    def verify_role(self):
        role = self.call('GET', ROLE)
        if role.get('deleted') or role.get('includedPermissions') != [PERMISSION]:
            raise ValueError('The QA role must permit only OIDC identity-token creation.')


def utc_text(value):
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def make_state(policy, minutes, *, now=None):
    if type(minutes) is not int or not 1 <= minutes <= 120:
        raise ValueError('Temporary QA access must expire between one and 120 minutes.')
    created = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    expires = created + timedelta(minutes=minutes)
    binding = {'role': ROLE, 'members': ['user:' + OPERATOR], 'condition': {
        'title': 'openecon-sandbox-qa-' + uuid4().hex,
        'description': DESCRIPTION,
        'expression': f'request.time < timestamp("{utc_text(expires)}")',
    }}
    return {'schema_version': 1, 'project': PROJECT, 'service_account': ACCOUNT,
            'operator': OPERATOR, 'original_etag': policy['etag'],
            'created_at_utc': utc_text(created), 'expires_at_utc': utc_text(expires),
            'binding': binding}


def validate_state(state):
    if (not isinstance(state, dict) or set(state) != {
            'schema_version', 'project', 'service_account', 'operator', 'original_etag',
            'created_at_utc', 'expires_at_utc', 'binding'}
            or state.get('schema_version') != 1 or state.get('project') != PROJECT
            or state.get('service_account') != ACCOUNT or state.get('operator') != OPERATOR
            or not isinstance(state.get('original_etag'), str) or not state['original_etag']):
        raise ValueError('QA state does not match its fixed ownership scope.')
    binding = state['binding']
    condition = binding.get('condition', {}) if isinstance(binding, dict) else {}
    if (not isinstance(binding, dict) or not isinstance(condition, dict)
            or set(binding) != {'role', 'members', 'condition'}
            or set(condition) != {'title', 'description', 'expression'}
            or binding.get('role') != ROLE or binding.get('members') != ['user:' + OPERATOR]
            or not isinstance(condition.get('title'), str)
            or not re.fullmatch(r'openecon-sandbox-qa-[0-9a-f]{32}', condition.get('title', ''))
            or condition.get('description') != DESCRIPTION):
        raise ValueError('Only this helper\'s exact conditional QA binding can be managed.')
    for key in ('created_at_utc', 'expires_at_utc'):
        if not isinstance(state[key], str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', state[key]):
            raise ValueError('QA expiry must be an explicit UTC instant.')
    created = datetime.fromisoformat(state['created_at_utc'].replace('Z', '+00:00'))
    expires = datetime.fromisoformat(state['expires_at_utc'].replace('Z', '+00:00'))
    if (not timedelta(minutes=1) <= expires - created <= timedelta(hours=2)
            or condition.get('expression') != f'request.time < timestamp("{state["expires_at_utc"]}")'):
        raise ValueError('QA expiry exceeds its narrow conditional-access contract.')
    return expires


def save_state(state, path=STATE, *, validator=None):
    (validator or validate_state)(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def read_state(path=STATE, *, validator=None):
    details = path.lstat()
    if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid()
            or details.st_mode & 0o077):
        raise ValueError('QA state must be an owner-only regular local file.')
    state = json.loads(path.read_text())
    (validator or validate_state)(state)
    return state


def binding_present(policy, binding):
    title = binding['condition']['title']
    candidates = [item for item in policy.get('bindings', [])
                  if item.get('condition', {}).get('title') == title]
    if candidates and (len(candidates) != 1 or candidates[0] != binding):
        raise ValueError('The unique QA binding changed; unrelated IAM access will not be modified.')
    return bool(candidates)


def reconcile_binding(client, binding, *, grant, account_path=ACCOUNT_PATH):
    """Change only the owned binding with a fresh etag and verified readback."""
    for _ in range(3):
        policy = client.policy()
        if binding_present(policy, binding) == grant:
            return
        desired = deepcopy(policy)
        desired['version'] = 3
        if grant:
            desired.setdefault('bindings', []).append(deepcopy(binding))
        else:
            desired['bindings'] = [item for item in desired.get('bindings', []) if item != binding]
        failure = None
        try:
            client.call('POST', account_path + ':setIamPolicy', {'policy': desired})
        except CloudError as error:
            failure = error
        # Even a lost response may have committed. Read the current policy
        # before retrying; never replace it with an old saved full policy.
        observed = client.policy()
        if binding_present(observed, binding) == grant:
            return
        if failure and failure.status not in {None, 409, 412, 500, 502, 503, 504}:
            raise failure
    raise CloudError()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--grant', action='store_true')
    actions.add_argument('--revoke', action='store_true')
    parser.add_argument('--minutes', type=int, default=60, help='Expiry for a new grant, maximum 120 minutes')
    args = parser.parse_args()
    if not 1 <= args.minutes <= 120:
        parser.error('Temporary QA access must expire between one and 120 minutes.')
    client = Client()
    client.verify_role()
    if args.grant:
        if STATE.exists():
            state = read_state(STATE)
        else:
            state = make_state(client.policy(), args.minutes)
            # Persist the exact owned binding before any IAM write, allowing
            # cleanup/reconciliation after an interrupted grant response.
            save_state(state, STATE)
        if validate_state(state) <= datetime.now(timezone.utc):
            raise SystemExit('Existing QA access has expired; revoke its saved binding before a new grant.')
        reconcile_binding(client, state['binding'], grant=True)
        print(json.dumps({'action': 'grant', 'binding_verified': True,
                          'expires_at_utc': state['expires_at_utc'], 'role': ROLE}))
    else:
        if not STATE.exists():
            raise SystemExit('No owned QA state exists; no IAM binding was modified.')
        state = read_state(STATE)
        reconcile_binding(client, state['binding'], grant=False)
        STATE.unlink()
        print(json.dumps({'action': 'revoke', 'owned_binding_absent': True, 'cleanup_complete': True}))


if __name__ == '__main__':
    main()
