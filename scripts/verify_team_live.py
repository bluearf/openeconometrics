"""Authorized team-sync QA using disposable test identities, never an owner token.

Creates no emails. Test credentials live only in an ignored, mode-0600 local file.
The sync backend never executes code: `verify` checks that old execution routes
are refused, never that a cloud analysis runs.

`prepare` labels every identity with a random session in Firebase Auth. `cleanup
--confirm-session` removes only that session's resources, and only from a
service that advertises the reviewed write-quiescence contract
(`openecon.team_quiescence`): disable and revoke the identities, wait until every
write admitted for them has finished, re-inventory, delete in order and read
back. Each phase is journaled in the private state, so an interrupted cleanup
resumes safely. Anything shared with another account refuses before any change.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
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
BUCKET = PROJECT + '-projects'
STATE = Path('artifacts/team-setup/qa-state.json')
SESSION_CLAIM = 'openecon_qa_session'
QA_EMAIL = re.compile(r'openecon-qa-(?:owner|editor|viewer|outsider|unverified)-[0-9a-f]{12}@example\.com')
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
        'QA cleanup refused: quiet flags, receipts and inspection snapshots cannot authorize deletion. '
        'Automatic deletion is disabled; run `cleanup --confirm-session SESSION`, which enforces '
        'admission and write quiescence first.')


def cleanup_qa(db, blobs, app, state, **claims):
    """Refuse legacy callers without inspecting state or trusting quiet claims."""
    refuse_automatic_cleanup()


def _refuse(reason):
    raise SystemExit('QA cleanup refused: ' + reason + ' No resource was changed by this check.')


def require_quiescence_contract(advertised):
    """Return the wait only for a service enforcing exactly the reviewed contract."""
    from openecon.team_quiescence import CONTRACT
    if advertised != CONTRACT:
        _refuse('the service does not advertise the reviewed write-quiescence contract.')
    return CONTRACT['quiescence_seconds']


class FirebaseIdentities:
    """The four Firebase Auth operations cleanup needs, bound to one Admin app."""

    def __init__(self, app):
        self.app = app

    def get(self, uid):
        try:
            return auth.get_user(uid, app=self.app)
        except auth.UserNotFoundError:
            return None

    def label(self, uid, session):
        auth.set_custom_user_claims(uid, {SESSION_CLAIM: session}, app=self.app)

    def disable(self, uid):
        auth.update_user(uid, disabled=True, app=self.app)
        auth.revoke_refresh_tokens(uid, app=self.app)

    def delete(self, uid):
        try:
            auth.delete_user(uid, app=self.app)
        except auth.UserNotFoundError:
            pass


def _session_users(state):
    session, users = state.get('session'), state.get('users')
    if not isinstance(session, str) or not re.fullmatch('[0-9a-f]{32}', session):
        _refuse('the private QA manifest has no session label; run `label-session` after review.')
    if not isinstance(users, dict) or not users or any(
        not isinstance(user, dict) or not isinstance(user.get('uid'), str)
        or not re.fullmatch('[A-Za-z0-9]{1,128}', user['uid'])
        or not isinstance(user.get('email'), str) or not QA_EMAIL.fullmatch(user['email'])
        for user in users.values()
    ) or len({user['uid'] for user in users.values()}) != len(users):
        _refuse('invalid disposable identity manifest.')
    return session, {user['uid']: user['email'] for user in users.values()}


def label_session(identities, state):
    """Label a reviewed legacy manifest's identities; never relabel another session."""
    if state.get('session') is not None:
        _refuse('this QA manifest is already labelled.')
    session = uuid4().hex
    _, emails = _session_users({**state, 'session': session})
    for uid, email in emails.items():
        record = identities.get(uid)
        if record is None or (record.email or '').lower() != email or record.custom_claims:
            _refuse('a manifest identity is missing, has another email or already carries claims.')
    for uid in emails:
        identities.label(uid, session)
    state['session'] = session
    return session


def session_identities(identities, state):
    """Map each manifest UID to its Auth record; None means it no longer exists.

    The Auth label, not the local file alone, proves session ownership.
    """
    session, emails = _session_users(state)
    records = {}
    for uid, email in emails.items():
        record = identities.get(uid)
        if record is not None and (
            (record.email or '').lower() != email
            or (record.custom_claims or {}).get(SESSION_CLAIM) != session
        ):
            _refuse('a manifest identity is not labelled for this QA session.')
        records[uid] = record
    return records


def _where(db, collection, field, value):
    from google.cloud.firestore_v1.base_query import FieldFilter
    return db.collection(collection).where(filter=FieldFilter(field, '==', value))


def _account_link_limits(db, uid):
    """Every account-link quota document of one identity, by exact document-ID range."""
    from google.cloud.firestore_v1.base_query import FieldFilter
    from google.cloud.firestore_v1.field_path import FieldPath
    prefix = f'account-link-{uid}-'
    query = (db.collection('oe_limits')
             .where(filter=FieldFilter(FieldPath.document_id(), '>=', db.document('oe_limits/' + prefix)))
             .where(filter=FieldFilter(FieldPath.document_id(), '<', db.document('oe_limits/' + prefix + '\uf8ff'))))
    keys = []
    for snapshot in query.stream():
        quota, day = _cleanup_document(snapshot), snapshot.id[len(prefix):]
        if (not snapshot.id.startswith(prefix) or not re.fullmatch('[0-9]{8}', day)
                or quota is None or set(quota) != {'count'} or type(quota['count']) is not int
                or not 0 <= quota['count'] <= 32):
            _refuse('an account-link quota could not be attributed to this QA session.')
        keys.append(snapshot.id)
    return keys


def session_inventory(db, state, uids):
    """Read every resource of this session completely; refuse anything shared.

    A project belongs to the session only when its owner and all members are
    session identities and every invitation it ever issued targeted one of them.
    """
    _, emails = _session_users(state)
    manifest_projects = _cleanup_ids(state['projects'])
    projects = set()
    for snapshot in db.collection('oe_projects').stream():
        project = _cleanup_document(snapshot)
        if project is None or not isinstance(project.get('members'), dict) \
                or not isinstance(project.get('owner_uid'), str):
            _refuse('resource ownership could not be established.')
        _cleanup_ids([snapshot.id])
        associated = set(project['members']) | {project['owner_uid']}
        if snapshot.id in manifest_projects or associated & uids:
            if project['owner_uid'] not in uids or associated - uids:
                _refuse('a session identity or project is shared with another account.')
            projects.add(snapshot.id)
    for uid in uids:
        profile = _cleanup_document(db.document('oe_users/' + uid).get())
        indexed = _cleanup_ids(profile.get('project_ids', []) if profile else [])
        for project_id in indexed - projects:
            if db.document('oe_projects/' + project_id).get().to_dict() is not None:
                _refuse('a session identity is indexed in another project.')
    session_emails = set(emails.values())
    invitations = set()
    for project_id in projects:
        for snapshot in _where(db, 'oe_invitations', 'project_id', project_id).stream():
            invitation = _cleanup_document(snapshot) or {}
            if invitation.get('email') not in session_emails:
                _refuse('a session project invited another account.')
            invitations.add(snapshot.id)
    for invitation_id in _cleanup_ids(state['invites']):
        invitation = _cleanup_document(db.document('oe_invitations/' + invitation_id).get())
        if invitation is not None and invitation.get('project_id') not in projects:
            _refuse('an invitation belongs to another project.')
    grants = set()
    for uid in uids:
        grants.update(snapshot.id for snapshot in _where(db, 'oe_desktop_logins', 'uid', uid).stream())
    for request_id in _cleanup_ids(state.get('desktop_login_ids', [])):
        grant = _cleanup_document(db.document('oe_desktop_logins/' + request_id).get())
        if grant is not None:
            if grant.get('uid') is not None and grant['uid'] not in uids:
                _refuse('a desktop grant belongs to another account.')
            grants.add(request_id)
    limits = set(_cleanup_account_link_limits(db, uids, state.get('account_link_limit_ids', [])))
    for uid in uids:
        limits.update(_account_link_limits(db, uid))
    recoveries = set()
    for project_id in projects:
        recoveries.update(snapshot.id for snapshot in _where(db, 'oe_recoveries', 'project_id', project_id).stream())
    return {'projects': sorted(projects), 'invitations': sorted(invitations),
            'desktop_logins': sorted(grants), 'account_link_limits': sorted(limits),
            'recoveries': sorted(recoveries), 'profiles': sorted(uids)}


DOCUMENT_GROUPS = (('oe_invitations', 'invitations'), ('oe_recoveries', 'recoveries'),
                   ('oe_desktop_logins', 'desktop_logins'), ('oe_limits', 'account_link_limits'),
                   ('oe_users', 'profiles'))


def delete_session(db, bucket, identities, inventory):
    """Idempotently delete blobs, project trees, related documents, then identities."""
    from google.api_core.exceptions import NotFound
    blobs = 0
    for project_id in inventory['projects']:
        for blob in bucket.list_blobs(prefix=f'projects/{project_id}/'):
            try:
                blob.delete()
                blobs += 1
            except NotFound:
                pass
        db.recursive_delete(db.document('oe_projects/' + project_id))
    for collection, group in DOCUMENT_GROUPS:
        for identifier in inventory[group]:
            db.document(f'{collection}/{identifier}').delete()
    for uid in inventory['profiles']:
        identities.delete(uid)
    return blobs


def residual_resources(db, bucket, identities, inventory):
    """Count every remaining session resource with fresh reads."""
    count = 0
    for project_id in inventory['projects']:
        count += len(list(bucket.list_blobs(prefix=f'projects/{project_id}/', max_results=1)))
        reference = db.document('oe_projects/' + project_id)
        count += reference.get().to_dict() is not None
        count += sum(len(list(collection.limit(1).stream())) for collection in reference.collections())
        for collection in ('oe_invitations', 'oe_recoveries'):
            count += len(list(_where(db, collection, 'project_id', project_id).limit(1).stream()))
    for collection, group in DOCUMENT_GROUPS:
        count += sum(db.document(f'{collection}/{identifier}').get().to_dict() is not None
                     for identifier in inventory[group])
    for uid in inventory['profiles']:
        count += len(list(_where(db, 'oe_desktop_logins', 'uid', uid).limit(1).stream()))
        count += len(_account_link_limits(db, uid))
        count += identities.get(uid) is not None
    return count


def cleanup_session(db, bucket, identities, state, *, quiescence_seconds, save,
                    clock=time.time, sleep=time.sleep, log=print):
    """Admission stop -> enforced quiescence -> inventory -> deletion -> readback.

    The journal in ``state['cleanup']`` records the phase. Rerunning after any
    interruption repeats only idempotent work and never shortens the wait.
    """
    journal = state.setdefault('cleanup', {'phase': 'admission'})
    try:
        while True:
            phase = journal['phase']
            if phase == 'admission':
                records = session_identities(identities, state)
                session_inventory(db, state, set(records))
                for uid, record in records.items():
                    if record is not None:
                        identities.disable(uid)
                if any(record is not None and not record.disabled
                       for record in (identities.get(uid) for uid in records)):
                    raise SystemExit('QA cleanup stopped: an identity could not be disabled. '
                                     'Private QA state was retained; rerun cleanup.')
                journal.update(phase='quiescence', identities=sorted(records),
                               admission_completed_at=clock())
                save()
                log('Disabled and revoked every session identity; waiting for write quiescence.')
            elif phase in ('quiescence', 'deletion', 'readback'):
                records = session_identities(identities, state)
                if set(records) != set(journal['identities']) or any(
                        record is not None and not record.disabled for record in records.values()):
                    # Someone re-enabled an identity: its writes are admitted again.
                    journal.clear()
                    journal['phase'] = 'admission'
                    save()
                    continue
                if phase == 'quiescence':
                    elapsed = clock() - journal['admission_completed_at']
                    if elapsed < 0:
                        journal['admission_completed_at'] = clock()
                        save()
                        continue
                    if elapsed < quiescence_seconds:
                        remaining = quiescence_seconds - elapsed
                        log(f'Waiting {math.ceil(remaining)} seconds for enforced write quiescence.')
                        sleep(remaining)
                        continue
                    journal.update(phase='deletion', inventory=session_inventory(db, state, set(records)))
                    save()
                elif phase == 'deletion':
                    journal['blobs_removed'] = journal.get('blobs_removed', 0) + delete_session(
                        db, bucket, identities, journal['inventory'])
                    journal['phase'] = 'readback'
                    save()
                else:
                    residual = residual_resources(db, bucket, identities, journal['inventory'])
                    if residual:
                        journal['phase'] = 'deletion'
                        save()
                        raise SystemExit(f'QA cleanup readback found {residual} remaining session resources. '
                                         'Private QA state was retained; rerun cleanup.')
                    journal['phase'] = 'complete'
                    save()
            elif phase == 'complete':
                inventory = journal['inventory']
                return {'session_cleanup': 'complete', 'residual_resources': 0,
                        'quiescence_seconds': quiescence_seconds,
                        'projects_removed': len(inventory['projects']),
                        'identities_removed': len(inventory['profiles']),
                        'invitations_removed': len(inventory['invitations']),
                        'desktop_grants_removed': len(inventory['desktop_logins']),
                        'account_link_quotas_removed': len(inventory['account_link_limits']),
                        'recovery_receipts_removed': len(inventory['recoveries']),
                        'blobs_removed': journal.get('blobs_removed', 0)}
            else:
                _refuse('the private cleanup journal is invalid.')
    except SystemExit:
        save()
        raise
    except Exception:
        # SDK errors may contain private paths; keep the journal and resume later.
        save()
        raise SystemExit(f'QA cleanup stopped safely during {journal.get("phase")}. '
                         'Private QA state was retained; rerun cleanup to resume.') from None


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


def save_private_state(path, state):
    temporary = path.with_name(path.name + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps(state, indent=2))
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify', 'cleanup', 'inspect-cleanup', 'label-session'])
    parser.add_argument('--url', default='https://openecon-teams-preview-291739190496.us-central1.run.app')
    parser.add_argument('--expect-commit', help='Require this full source commit in /api/auth/config')
    parser.add_argument('--confirm-session', help='The session label printed by prepare; required by cleanup')
    args = parser.parse_args()
    if args.action == 'cleanup' and not args.confirm_session:
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
    if args.action in ('cleanup', 'label-session'):
        state = json.loads(STATE.read_text())
        identities = FirebaseIdentities(app)
        if args.action == 'label-session':
            session = label_session(identities, state)
            save_private_state(STATE, state)
            print('Labelled the reviewed QA identities. Cleanup confirmation session:', session)
            return
        if state.get('session') != args.confirm_session:
            _refuse('the confirmed session does not match the private QA manifest.')
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            public = client.get(args.url.rstrip('/') + '/api/auth/config').json()
        seconds = require_quiescence_contract(public.get('write_quiescence'))
        from google.cloud import storage as cloud_storage
        bucket = cloud_storage.Client(project=PROJECT, credentials=credential).bucket(BUCKET)
        report = cleanup_session(db, bucket, identities, state, quiescence_seconds=seconds,
                                 save=lambda: save_private_state(STATE, state))
        # Keep the reviewed manifest without passwords; a new prepare may follow.
        for user in state['users'].values():
            user.pop('password', None)
        save_private_state(STATE.with_name(f'qa-state.cleaned-{state["session"]}.json'), state)
        STATE.unlink()
        print(json.dumps(report, indent=2))
        return
    if args.action == 'prepare':
        if STATE.exists():
            raise SystemExit('Existing QA state must be retained and explicitly reviewed before another prepare.')
        state = {'session': uuid4().hex, 'users': {}, 'projects': [], 'invites': [], 'checks': [],
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
            auth.set_custom_user_claims(user.uid, {SESSION_CLAIM: state['session']}, app=app)
        owner = state['users']['owner']
        db.document('oe_users/' + owner['uid']).set({'uid': owner['uid'], 'email': owner['email'],
            'name': 'QA Owner', 'enabled': True, 'project_ids': [], 'created_at': '2026-10-01T00:00:00+00:00'})
        print('Prepared five disposable QA identities; no emails sent. '
              'Cleanup confirmation session:', state['session'])
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
          'Remove them with `cleanup --confirm-session SESSION` when finished.', flush=True)


if __name__ == '__main__':
    main()
