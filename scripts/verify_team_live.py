"""Authorized team-sync QA using disposable test identities, never an owner token.

Creates no emails. Test credentials live only in an ignored, mode-0600 local file.
The sync backend never executes code: `verify` checks that old execution routes
are refused, never that a cloud analysis runs. Automatic cleanup is disabled:
ownership inspection cannot enforce quiescence across Firestore, Cloud Storage
and Firebase Auth.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import secrets
import subprocess
import time
from uuid import uuid4

import firebase_admin
from firebase_admin import auth, credentials
from google.cloud import firestore
from google.oauth2.credentials import Credentials
import httpx

PROJECT = 'openecon-workbench'
STATE = Path('artifacts/team-setup/qa-state.json')
def ensure_cleanup_idle(db, project_ids):
    """Read all disposable projects before deleting any project or identity."""
    terminal = {'finished', 'failed', 'cancelled'}
    for project_id in project_ids:
        ref = db.document('oe_projects/' + project_id)
        project = ref.get().to_dict() or {}
        if project.get('active_run'):
            raise SystemExit('QA cleanup refused: a disposable analysis is still active. Stop it and wait for completion.')
        for snapshot in ref.collection('runs').stream():
            run = snapshot.to_dict() or {}
            if run.get('state') not in terminal:
                raise SystemExit('QA cleanup refused: a disposable analysis is still active. Stop it and wait for completion.')


def _cleanup_ids(values):
    if not isinstance(values, list) or any(
        not isinstance(value, str) or len(value) != 32
        or any(character not in '0123456789abcdef' for character in value)
        for value in values
    ):
        raise SystemExit('QA cleanup refused: invalid disposable resource manifest.')
    return set(values)


def _cleanup_document(snapshot):
    document = snapshot.to_dict()
    if document is not None and not isinstance(document, dict):
        raise SystemExit('QA cleanup refused: resource ownership could not be established.')
    return document


def _cleanup_account_link_limits(db, user_ids, values):
    """Inspect exact manifest keys; never scan or reset a shared quota collection."""
    if not isinstance(values, list):
        raise SystemExit('QA cleanup refused: invalid disposable quota manifest.')
    limits = {}
    for key in values:
        if not isinstance(key, str):
            raise SystemExit('QA cleanup refused: invalid disposable quota manifest.')
        owners = [uid for uid in user_ids if key.startswith(f'account-link-{uid}-')]
        if len(owners) != 1:
            raise SystemExit('QA cleanup refused: a quota belongs to a non-QA identity.')
        uid = owners[0]
        day = key[len(f'account-link-{uid}-'):]
        if len(day) != 8 or any(character not in '0123456789' for character in day):
            raise SystemExit('QA cleanup refused: invalid disposable quota manifest.')
        try:
            datetime.strptime(day, '%Y%m%d')
        except ValueError:
            raise SystemExit('QA cleanup refused: invalid disposable quota manifest.') from None
        quota = _cleanup_document(db.document('oe_limits/' + key).get())
        if quota is not None and (
            set(quota) != {'count'} or type(quota['count']) is not int
            or not 0 <= quota['count'] <= 32
        ):
            raise SystemExit('QA cleanup refused: quota ownership could not be established.')
        limits[key] = (uid, quota is not None)
    return limits


def _inspect_cleanup_users(db, user_ids, project_ids):
    """Use both profile indexes and authoritative projects; neither is sufficient alone.

    A complete project scan also finds memberships whose profile index is stale or
    missing. Finish every read before returning, including the full stream, so a
    partial query result can never authorize identity deletion.
    """
    protected, profiles = set(), {}
    for user_id in user_ids:
        profile = _cleanup_document(db.document('oe_users/' + user_id).get())
        profiles[user_id] = profile
        indexed = profile.get('project_ids', []) if profile is not None else []
        indexed = _cleanup_ids(indexed)
        if indexed - project_ids:
            protected.add(user_id)
    for snapshot in db.collection('oe_projects').stream():
        project = _cleanup_document(snapshot)
        if project is None:
            raise SystemExit('QA cleanup refused: resource ownership could not be established.')
        members, owner = project.get('members'), project.get('owner_uid')
        if not isinstance(members, dict) or not isinstance(owner, str):
            raise SystemExit('QA cleanup refused: resource ownership could not be established.')
        project_id = snapshot.id
        _cleanup_ids([project_id])
        associated = set(members) | {owner}
        if project_id not in project_ids:
            protected.update(associated & user_ids)
        elif owner not in user_ids or associated - user_ids:
            # An adopted QA project may contain human data even if its ID was
            # originally recorded by this script. Require explicit review.
            raise SystemExit('QA cleanup refused: a disposable project has non-QA ownership or membership.')
    return protected, profiles


def refuse_automatic_cleanup():
    raise SystemExit(
        'QA cleanup refused: no enforced admission or quiescence contract exists. '
        'Automatic deletion is disabled; inspect-cleanup is read-only and cannot authorize deletion.')


def cleanup_qa(db, blobs, app, state, **claims):
    """Refuse legacy callers without inspecting state or trusting quiet claims."""
    refuse_automatic_cleanup()


def inspect_cleanup_qa(db, state):
    """Read exact ownership and quota associations without authorizing deletion.

    These independent reads are only a snapshot. A new run or membership can
    appear immediately afterward; an idle snapshot is not enforced quiescence.
    """
    try:
        project_ids = _cleanup_ids(state['projects'])
        invitation_ids = _cleanup_ids(state['invites'])
        grant_ids = _cleanup_ids(state.get('desktop_login_ids', []))
        users = state['users']
        if not isinstance(users, dict) or any(
            not isinstance(user, dict) or not isinstance(user.get('uid'), str)
            or not user['uid'] or len(user['uid']) > 128
            or '/' in user['uid'] or any(ord(character) < 32 for character in user['uid'])
            for user in users.values()
        ):
            raise SystemExit('QA cleanup refused: invalid disposable identity manifest.')
        user_ids = {user['uid'] for user in users.values()}
        protected, _ = _inspect_cleanup_users(db, user_ids, project_ids)
        ensure_cleanup_idle(db, sorted(project_ids))
        # Inspect every exact recorded related resource. This path has no writes.
        for invitation_id in invitation_ids:
            invitation = _cleanup_document(db.document('oe_invitations/' + invitation_id).get())
            if invitation is not None and invitation.get('project_id') not in project_ids:
                raise SystemExit('QA cleanup refused: an invitation belongs to a non-QA project.')
        for request_id in grant_ids:
            grant = _cleanup_document(db.document('oe_desktop_logins/' + request_id).get())
            if grant is not None and grant.get('uid') is not None and grant['uid'] not in user_ids:
                raise SystemExit('QA cleanup refused: a desktop grant belongs to a non-QA identity.')
        account_link_limits = _cleanup_account_link_limits(
            db, user_ids, state.get('account_link_limit_ids', []))

        return {
            'inspection_only': True, 'safe_to_delete': False,
            'quiescence_enforced': False, 'state_retained': True,
            'projects_inspected': len(project_ids),
            'identities_inspected': len(user_ids),
            'identities_with_other_project_associations': len(protected),
            'invitations_inspected': len(invitation_ids),
            'desktop_grants_inspected': len(grant_ids),
            'recorded_account_link_quotas_inspected': len(account_link_limits),
            'recorded_account_link_quotas_present': sum(existed for _, existed in account_link_limits.values()),
        }
    except Exception:
        # SDK errors may contain private document paths or credential details.
        # Keep the private manifest on all failures and never print raw errors.
        raise SystemExit('QA cleanup inspection stopped: resource ownership could not be safely inspected. Private QA state was retained.') from None


def platform_request(client, method, url, **kwargs):
    """Retry only read-only requests rejected by Cloud Run before the app.

    Firebase/application JSON errors and every mutation are returned immediately.
    This allows a fresh public service's IAM propagation to settle without
    repeating an upload, creating projects or altering membership.
    """
    for attempt in range(5):
        response = client.request(method, url, **kwargs)
        transient = (method == 'GET' and response.status_code == 401
                     and 'text/html' in response.headers.get('content-type', '')
                     and '<title>401 Unauthorized</title>' in response.text)
        if not transient or attempt == 4:
            return response
        time.sleep(2)
    raise AssertionError('Unreachable retry state.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify', 'cleanup', 'inspect-cleanup'])
    parser.add_argument('--url', default='https://openecon-teams-preview-291739190496.us-central1.run.app')
    parser.add_argument('--expect-commit', help='Require this full source commit in /api/auth/config')
    args = parser.parse_args()
    if args.action == 'cleanup':
        # Refuse before acquiring credentials, reading the private manifest, or
        # constructing any SDK client. No caller-supplied receipt admits cleanup.
        refuse_automatic_cleanup()
    token = subprocess.check_output(['gcloud', 'auth', 'print-access-token', '--project=' + PROJECT], text=True).strip()
    credential = Credentials(token, quota_project_id=PROJECT)

    if args.action == 'inspect-cleanup':
        db = firestore.Client(project=PROJECT, credentials=credential)
        state = json.loads(STATE.read_text())
        print(json.dumps(inspect_cleanup_qa(db, state), indent=2))
        return

    class AdminCredential(credentials.Base):
        def get_credential(self):
            return credential

    app = firebase_admin.initialize_app(AdminCredential(), {'projectId': PROJECT})
    db = firestore.Client(project=PROJECT, credentials=credential)
    if args.action == 'prepare':
        if STATE.exists():
            raise SystemExit('Existing QA state must be retained and explicitly reviewed before another prepare.')
        state = {'users': {}, 'projects': [], 'invites': [], 'checks': [],
                 'account_link_limit_ids': []}
        STATE.parent.mkdir(exist_ok=True, parents=True)
        STATE.touch(mode=0o600)
        def save():
            STATE.write_text(json.dumps(state, indent=2))
        save()
        for role in ['owner', 'editor', 'viewer', 'outsider', 'unverified']:
            email = f'openecon-qa-{role}-{uuid4().hex[:12]}@example.com'
            password = 'Qa9-' + secrets.token_urlsafe(30)
            user = auth.create_user(email=email, password=password, email_verified=role != 'unverified',
                                    display_name=f'QA {role.title()}', app=app)
            state['users'][role] = {'uid': user.uid, 'email': email, 'password': password}
            save()
        owner = state['users']['owner']
        db.document('oe_users/' + owner['uid']).set({'uid': owner['uid'], 'email': owner['email'],
            'name': 'QA Owner', 'enabled': True, 'project_ids': [], 'created_at': '2026-10-01T00:00:00+00:00'})
        print('Prepared five disposable QA identities; no emails sent.')
        return
    state = json.loads(STATE.read_text())
    state['url'] = args.url.rstrip('/')
    client = httpx.Client(timeout=60, follow_redirects=False)
    public = client.get(state['url'] + '/api/auth/config').json()
    if public.get('cloud_execution_available') is not False:
        raise AssertionError('The sync backend must report that cloud execution is unavailable.')
    if args.expect_commit and public.get('source_commit') != args.expect_commit:
        raise AssertionError('The live service does not report the expected source commit.')
    print('Service source commit:', public.get('source_commit'), flush=True)
    config = public['firebase']
    tokens = {}
    for role, user in state['users'].items():
        response = client.post('https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword',
            params={'key': config['apiKey']}, json={'email': user['email'], 'password': user['password'],
                                                 'returnSecureToken': True})
        if response.status_code != 200:
            raise RuntimeError(f'QA sign-in failed: {role}, HTTP {response.status_code}')
        tokens[role] = response.json()['idToken']

    def save():
        STATE.write_text(json.dumps(state, indent=2))

    def check(name):
        state['checks'].append(name)
        save()
        print('PASS', name, flush=True)

    def call(method, path, role='owner', status=200, **kwargs):
        response = platform_request(client, method, state['url'] + '/api' + path,
            headers={'Authorization': 'Bearer ' + tokens[role], 'Origin': state['url']}, **kwargs)
        if response.status_code != status:
            # Only our API's safe response is printed; never provider tokens.
            raise AssertionError(f'{method} {path}: HTTP {response.status_code}: {response.text[:600]}')
        return response.json() if 'application/json' in response.headers.get('content-type', '') else response.content

    assert client.get(state['url'] + '/api/projects').status_code == 401
    check('Anonymous project API denied; login page available')
    assert call('GET', '/me', 'unverified')['user']['email_verified'] is False
    call('GET', '/projects', 'unverified', status=403)
    check('Unverified email cannot access projects')
    for role in state['users']:
        call('GET', '/me', role)
    call('POST', '/projects', 'outsider', status=403, json={'name': 'Unauthorized'})
    check('Uninvited account cannot create a project')
    project = call('POST', '/projects', status=201, json={'name': 'Ekip doğrulama', 'description': 'Geçici QA projesi'})
    project_id = project['id']
    state['projects'].append(project_id)
    save()
    workspace = f'/projects/{project_id}/workspace'
    call('GET', workspace + '/console', 'outsider', status=404)
    for role in ['editor', 'viewer']:
        invite = call('POST', f'/projects/{project_id}/invitations', status=201,
                      json={'email': state['users'][role]['email'], 'role': role})
        state['invites'].append(invite['id'])
        save()
        call('POST', '/invitations/' + invite['id'] + '/accept', 'outsider', status=404)
        call('POST', '/invitations/' + invite['id'] + '/accept', role)
        call('POST', '/invitations/' + invite['id'] + '/accept', role, status=404)
    check('Email-bound invitations accepted once; cross-project access denied')
    script = 'import openecon as oe\ndf = oe.example()\ndisplay(df.head())\n'
    call('PUT', workspace + '/console/script', 'editor', json={'code': script, 'version': 0})
    call('PUT', workspace + '/console/script', 'owner', status=409, json={'code': 'stale', 'version': 0})
    call('PUT', workspace + '/console/script', 'viewer', status=403, json={'code': 'forbidden', 'version': 1})
    assert call('GET', workspace + '/console/script', 'viewer')['code'] == script
    check('Shared draft persists; stale writes and viewer edits denied')
    dataset = call('POST', workspace + '/datasets/upload', 'editor', status=201,
                   files={'file': ('team-data.csv', b'x,y\n1,2\n2,4\n3,6\n', 'text/csv')})
    assert call('GET', workspace + '/files/' + dataset['id'] + '/download', 'viewer') == b'x,y\n1,2\n2,4\n3,6\n'
    check('Project upload/download persist for members')
    for role in ['editor', 'viewer']:
        for route in ['execute', 'interrupt', 'reset']:
            refused = call('POST', workspace + '/console/' + route, role, status=410, json={'code': 'print(1)'})
            assert refused['detail']['code'] == 'CLOUD_EXECUTION_RETIRED'
    call('POST', workspace + '/console/execute', 'outsider', status=404, json={'code': 'print(1)'})
    synced = call('GET', workspace + '/console', 'viewer')
    assert synced['status']['running'] is False and synced['history'] == []
    check('Cloud execution routes refuse old clients; no run was created')
    call('PATCH', f'/projects/{project_id}/members/' + state['users']['editor']['uid'],
         json={'role': 'viewer'})
    call('PUT', workspace + '/console/script', 'editor', status=403, json={'code': 'demoted', 'version': 1})
    call('DELETE', f'/projects/{project_id}/members/' + state['users']['viewer']['uid'])
    call('GET', workspace + '/console', 'viewer', status=404)
    check('Role changes and member removal take effect on the next request')
    print('Live sync integration finished; QA identities remain preserved for UI verification. '
          'Automatic cleanup is disabled.', flush=True)


if __name__ == '__main__':
    main()
