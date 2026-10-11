"""Session-labelled QA cleanup: admission stop, enforced quiescence, ordered deletion, readback."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from google.api_core.exceptions import NotFound
import pytest

from openecon.team_quiescence import CONTRACT, QUIESCENCE_SECONDS

_SPEC = importlib.util.spec_from_file_location(
    'team_live_session_cleanup', Path(__file__).parents[1] / 'scripts/verify_team_live.py')
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)

SESSION, OTHER_SESSION = 'a' * 32, 'b' * 32
PROJECT, UNRECORDED, HUMAN_PROJECT, OTHER_QA_PROJECT = '1' * 32, '2' * 32, '3' * 32, '4' * 32
INVITE_EDITOR, INVITE_VIEWER, INVITE_HUMAN = 'c' * 32, 'd' * 32, 'e' * 32
GRANT, RECORDED_GRANT, HUMAN_GRANT = '5' * 32, '6' * 32, '7' * 32
EMAILS = {role: f'openecon-qa-{role}-00000000000{index}@example.com'
          for index, role in enumerate(['owner', 'editor', 'viewer'], 1)}
OWNER, EDITOR, VIEWER = 'qaowner', 'qaeditor', 'qaviewer'
QUOTA = f'account-link-{OWNER}-20261009'


class Snapshot:
    def __init__(self, path, document):
        self.id, self._document = path.rsplit('/', 1)[-1], deepcopy(document)

    def to_dict(self):
        return deepcopy(self._document)


class Reference:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def get(self):
        return Snapshot(self.path, self.db.docs.get(self.path))

    def delete(self):
        self.db.events.append(('delete', self.path))
        self.db.docs.pop(self.path, None)

    def collection(self, name):
        return Query(self.db, self.path + '/' + name)

    def collections(self):
        depth = self.path.count('/') + 2
        names = {path[len(self.path) + 1:].split('/', 1)[0] for path in self.db.docs
                 if path.startswith(self.path + '/') and path.count('/') == depth}
        return [Query(self.db, self.path + '/' + name) for name in sorted(names)]


class Query:
    def __init__(self, db, path, filters=(), limit=None):
        self.db, self.path, self.filters, self.count = db, path, filters, limit

    def where(self, *, filter):
        return Query(self.db, self.path, (*self.filters, filter), self.count)

    def limit(self, count):
        return Query(self.db, self.path, self.filters, count)

    def _matches(self, path, document):
        for condition in self.filters:
            if condition.field_path == '__name__':
                bound = condition.value.path
                if not {'>=': path >= bound, '<': path < bound}[condition.op_string]:
                    return False
            elif condition.op_string != '==' or document.get(condition.field_path) != condition.value:
                return False
        return True

    def stream(self):
        rows = [Snapshot(path, document) for path, document in sorted(self.db.docs.items())
                if path.startswith(self.path + '/') and path.count('/') == self.path.count('/') + 1
                and self._matches(path, document)]
        return iter(rows[:self.count] if self.count is not None else rows)


class Database:
    def __init__(self, docs):
        self.docs, self.events = deepcopy(docs), []

    def document(self, path):
        return Reference(self, path)

    def collection(self, path):
        return Query(self, path)

    def recursive_delete(self, reference):
        for path in sorted(self.docs):
            if path == reference.path or path.startswith(reference.path + '/'):
                self.events.append(('delete', path))
                del self.docs[path]


class Bucket:
    def __init__(self, objects, events):
        self.objects, self.events, self.fail_once = dict(objects), events, set()
        self.unlisted_for_deletion = set()

    def list_blobs(self, prefix, max_results=None):
        # A full listing (deletion) may miss objects; the readback probe never does.
        names = sorted(name for name in self.objects if name.startswith(prefix)
                       and (max_results is not None or name not in self.unlisted_for_deletion))
        blobs = [SimpleNamespace(name=name, delete=lambda name=name: self._delete(name)) for name in names]
        return blobs[:max_results] if max_results is not None else blobs

    def _delete(self, name):
        if name in self.fail_once:
            self.fail_once.discard(name)
            raise RuntimeError('private-provider-error-sentinel')
        if name not in self.objects:
            raise NotFound('gone')
        self.events.append(('delete', 'gs://' + name))
        del self.objects[name]


class Identities:
    def __init__(self, records, events):
        self.records, self.events = deepcopy(records), events

    def get(self, uid):
        record = self.records.get(uid)
        return None if record is None else SimpleNamespace(**deepcopy(record))

    def label(self, uid, session):
        self.events.append(('label', uid))
        self.records[uid]['custom_claims'] = {verify.SESSION_CLAIM: session}

    def disable(self, uid):
        self.events.append(('disable', uid))
        self.records[uid]['disabled'] = True

    def delete(self, uid):
        self.events.append(('delete', 'auth:' + uid))
        self.records.pop(uid, None)


def identity(email, session=SESSION, disabled=False):
    return {'email': email, 'disabled': disabled,
            'custom_claims': {verify.SESSION_CLAIM: session} if session else {}}


def session_documents():
    members = {uid: {'role': role} for uid, role in [(OWNER, 'owner'), (EDITOR, 'editor'), (VIEWER, 'viewer')]}
    return {
        f'oe_projects/{PROJECT}': {'owner_uid': OWNER, 'members': members, 'invitations': []},
        f'oe_projects/{PROJECT}/runs/r1': {'state': 'finished'},
        f'oe_projects/{PROJECT}/audit/a1': {'action': 'project.create'},
        f'oe_projects/{PROJECT}/scripts/s1': {'code': 'x = 1'},
        f'oe_projects/{PROJECT}/workspace/script': {'code': 'y = 2'},
        f'oe_projects/{PROJECT}/transfers/t1': {'status': 'complete'},
        f'oe_projects/{UNRECORDED}': {'owner_uid': OWNER, 'members': {OWNER: {'role': 'owner'}}},
        f'oe_users/{OWNER}': {'project_ids': [PROJECT, UNRECORDED]},
        f'oe_users/{EDITOR}': {'project_ids': [PROJECT]},
        f'oe_users/{VIEWER}': {'project_ids': [PROJECT]},
        f'oe_invitations/{INVITE_EDITOR}': {'project_id': PROJECT, 'email': EMAILS['editor']},
        f'oe_invitations/{INVITE_VIEWER}': {'project_id': PROJECT, 'email': EMAILS['viewer']},
        f'oe_desktop_logins/{GRANT}': {'uid': OWNER},
        f'oe_desktop_logins/{RECORDED_GRANT}': {'uid': None},
        f'oe_limits/{QUOTA}': {'count': 1},
        'oe_recoveries/' + 'f' * 32: {'project_id': PROJECT},
    }


def foreign_documents():
    return {
        f'oe_projects/{HUMAN_PROJECT}': {'owner_uid': 'human', 'members': {'human': {'role': 'owner'}}},
        f'oe_projects/{HUMAN_PROJECT}/runs/r1': {'state': 'finished'},
        f'oe_projects/{OTHER_QA_PROJECT}': {'owner_uid': 'otherqa', 'members': {'otherqa': {'role': 'owner'}}},
        'oe_users/human': {'project_ids': [HUMAN_PROJECT]},
        'oe_users/otherqa': {'project_ids': [OTHER_QA_PROJECT]},
        f'oe_invitations/{INVITE_HUMAN}': {'project_id': HUMAN_PROJECT, 'email': 'colleague@example.org'},
        f'oe_desktop_logins/{HUMAN_GRANT}': {'uid': 'human'},
        'oe_limits/account-link-human-20261009': {'count': 2},
        f'oe_limits/account-link-{OWNER}X-20261009': {'count': 3},
        'oe_limits/desktop-login-20261009': {'count': 9},
        'oe_recoveries/' + '9' * 32: {'project_id': HUMAN_PROJECT},
    }


SESSION_BLOBS = {f'projects/{PROJECT}/files/f1/data.csv': b'x', f'projects/{PROJECT}/transfers/t1/0000/ab': b'y',
                 f'projects/{PROJECT}/desktop-results/r1/p1/record.json': b'{}',
                 f'projects/{UNRECORDED}/files/f2/data.csv': b'z'}
FOREIGN_BLOBS = {f'projects/{HUMAN_PROJECT}/files/f3/private.csv': b'human',
                 f'projects/{OTHER_QA_PROJECT}/files/f4/data.csv': b'other'}
FOREIGN_IDENTITIES = {'human': {'email': 'person@example.org', 'disabled': False, 'custom_claims': {}},
                      f'{OWNER}X': {'email': 'person2@example.org', 'disabled': False, 'custom_claims': {}},
                      'otherqa': identity('openecon-qa-owner-0000000000ff@example.com', OTHER_SESSION)}


def fixture():
    events = []
    state = {'session': SESSION, 'projects': [PROJECT], 'invites': [INVITE_EDITOR, INVITE_VIEWER],
             'desktop_login_ids': [RECORDED_GRANT], 'account_link_limit_ids': [QUOTA], 'checks': [],
             'users': {role: {'uid': uid, 'email': EMAILS[role], 'password': 'private-password-sentinel'}
                       for role, uid in [('owner', OWNER), ('editor', EDITOR), ('viewer', VIEWER)]}}
    db = Database({**session_documents(), **foreign_documents()})
    db.events = events
    bucket = Bucket({**SESSION_BLOBS, **FOREIGN_BLOBS}, events)
    identities = Identities({**{uid: identity(EMAILS[role]) for role, uid in
                                [('owner', OWNER), ('editor', EDITOR), ('viewer', VIEWER)]},
                             **FOREIGN_IDENTITIES}, events)
    return SimpleNamespace(state=state, db=db, bucket=bucket, identities=identities, events=events)


class Clock:
    def __init__(self):
        self.now, self.sleeps, self.during_sleep = 1_000_000.0, [], None

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if self.during_sleep:
            self.during_sleep()
        self.now += seconds


def run(f, clock, saves=None):
    saved = saves if saves is not None else []
    return verify.cleanup_session(
        f.db, f.bucket, f.identities, f.state, quiescence_seconds=QUIESCENCE_SECONDS,
        save=lambda: saved.append(deepcopy(f.state['cleanup'])), clock=clock, sleep=clock.sleep,
        log=lambda *args: None)


def assert_foreign_untouched(f):
    for path, document in foreign_documents().items():
        assert f.db.docs[path] == document
    assert {name: f.bucket.objects[name] for name in FOREIGN_BLOBS} == FOREIGN_BLOBS
    for uid, record in FOREIGN_IDENTITIES.items():
        assert f.identities.records[uid] == record


def assert_session_gone(f):
    assert not any(PROJECT in path or UNRECORDED in path for path in f.db.docs)
    assert not any(name in f.bucket.objects for name in SESSION_BLOBS)
    assert not any(uid in f.identities.records for uid in (OWNER, EDITOR, VIEWER))
    remaining = set(f.db.docs) - set(foreign_documents())
    assert remaining == set()


def test_full_cycle_removes_only_this_session_and_reads_back_zero():
    f, clock, saves = fixture(), Clock(), []
    report = run(f, clock, saves)
    assert report == {'session_cleanup': 'complete', 'residual_resources': 0,
                      'quiescence_seconds': QUIESCENCE_SECONDS, 'projects_removed': 2,
                      'identities_removed': 3, 'invitations_removed': 2, 'desktop_grants_removed': 2,
                      'account_link_quotas_removed': 1, 'recovery_receipts_removed': 1, 'blobs_removed': 4}
    assert_session_gone(f)
    assert_foreign_untouched(f)
    assert sum(clock.sleeps) >= QUIESCENCE_SECONDS
    kinds = [event for event, _ in f.events]
    last_disable = max(i for i, kind in enumerate(kinds) if kind == 'disable')
    first_delete = kinds.index('delete')
    assert last_disable < first_delete
    assert {uid for kind, uid in f.events if kind == 'disable'} == {OWNER, EDITOR, VIEWER}
    targets = [target for kind, target in f.events if kind == 'delete']
    assert all(target.startswith('auth:') for target in targets[-3:])
    assert not any(target.startswith('auth:') for target in targets[:-3])
    assert [save['phase'] for save in saves] == ['quiescence', 'deletion', 'readback', 'complete']
    assert 'private-password-sentinel' not in json.dumps(report)


@pytest.mark.parametrize('scenario', [
    'no_session', 'other_session_label', 'unlabelled', 'email_mismatch', 'human_member',
    'identity_in_human_project', 'invitation_to_human', 'manifest_invitation_foreign',
    'manifest_grant_foreign', 'manifest_quota_foreign', 'malformed_quota', 'indexed_human_project',
])
def test_shared_or_unlabelled_resources_refuse_before_any_change(scenario):
    f, clock = fixture(), Clock()
    if scenario == 'no_session':
        del f.state['session']
    elif scenario == 'other_session_label':
        f.identities.records[EDITOR]['custom_claims'] = {verify.SESSION_CLAIM: OTHER_SESSION}
    elif scenario == 'unlabelled':
        f.identities.records[VIEWER]['custom_claims'] = {}
    elif scenario == 'email_mismatch':
        f.identities.records[OWNER]['email'] = 'person@example.org'
    elif scenario == 'human_member':
        f.db.docs[f'oe_projects/{PROJECT}']['members']['human'] = {'role': 'viewer'}
    elif scenario == 'identity_in_human_project':
        f.db.docs[f'oe_projects/{HUMAN_PROJECT}']['members'][EDITOR] = {'role': 'editor'}
    elif scenario == 'invitation_to_human':
        f.db.docs[f'oe_invitations/{"8" * 32}'] = {'project_id': PROJECT, 'email': 'colleague@example.org'}
    elif scenario == 'manifest_invitation_foreign':
        f.state['invites'].append(INVITE_HUMAN)
    elif scenario == 'manifest_grant_foreign':
        f.state['desktop_login_ids'].append(HUMAN_GRANT)
    elif scenario == 'manifest_quota_foreign':
        f.state['account_link_limit_ids'].append('account-link-human-20261009')
    elif scenario == 'malformed_quota':
        f.db.docs[f'oe_limits/{QUOTA}']['private'] = 'sentinel'
    else:
        f.db.docs[f'oe_users/{OWNER}']['project_ids'].append(HUMAN_PROJECT)
    docs, blobs, records = deepcopy(f.db.docs), dict(f.bucket.objects), deepcopy(f.identities.records)
    with pytest.raises(SystemExit) as error:
        run(f, clock)
    assert 'private-password-sentinel' not in str(error.value) and 'human' not in str(error.value)
    assert f.events == [] and clock.sleeps == []
    assert f.db.docs == docs and f.bucket.objects == blobs and f.identities.records == records
    assert f.state['cleanup'] == {'phase': 'admission'}


@pytest.mark.parametrize('advertised', [None, {}, {**CONTRACT, 'version': 2},
                                        {**CONTRACT, 'quiescence_seconds': 1},
                                        {**CONTRACT, 'side_effect_seconds': 3600}])
def test_only_the_exact_reviewed_service_contract_is_accepted(advertised):
    with pytest.raises(SystemExit, match='write-quiescence contract'):
        verify.require_quiescence_contract(advertised)
    assert verify.require_quiescence_contract(dict(CONTRACT)) == QUIESCENCE_SECONDS


def test_interrupted_wait_resumes_without_shortening_quiescence():
    f, clock = fixture(), Clock()
    started = clock.now

    def interrupt():
        clock.now += 200
        raise RuntimeError('private-provider-error-sentinel')
    clock.during_sleep = interrupt
    with pytest.raises(SystemExit, match='during quiescence') as error:
        run(f, clock)
    assert 'private-provider-error-sentinel' not in str(error.value)
    assert f.state['cleanup']['phase'] == 'quiescence'
    assert not any(kind == 'delete' for kind, _ in f.events)
    clock.during_sleep = None
    run(f, clock)
    assert clock.sleeps[-1] == pytest.approx(QUIESCENCE_SECONDS - 200)
    assert clock.now - started >= QUIESCENCE_SECONDS
    assert_session_gone(f)
    assert_foreign_untouched(f)


def test_failure_during_deletion_resumes_idempotently_without_waiting_again():
    f, clock = fixture(), Clock()
    f.bucket.fail_once.add(f'projects/{PROJECT}/transfers/t1/0000/ab')
    with pytest.raises(SystemExit, match='during deletion') as error:
        run(f, clock)
    assert 'private-provider-error-sentinel' not in str(error.value)
    assert f.state['cleanup']['phase'] == 'deletion'
    assert OWNER in f.identities.records
    sleeps = len(clock.sleeps)
    report = run(f, clock)
    assert len(clock.sleeps) == sleeps
    assert report['residual_resources'] == 0
    assert_session_gone(f)
    assert_foreign_untouched(f)


def test_reenabled_identity_restarts_admission_and_the_full_wait():
    f, clock = fixture(), Clock()
    clock.during_sleep = Mock(side_effect=RuntimeError('stop'))
    with pytest.raises(SystemExit):
        run(f, clock)
    f.identities.records[EDITOR]['disabled'] = False
    clock.during_sleep = None
    clock.now += QUIESCENCE_SECONDS * 2
    run(f, clock)
    assert [uid for kind, uid in f.events if kind == 'disable'].count(EDITOR) == 2
    assert clock.sleeps[-1] == pytest.approx(QUIESCENCE_SECONDS)
    assert_session_gone(f)


def test_writes_landing_during_quiescence_are_inventoried_and_removed():
    f, clock = fixture(), Clock()
    late_project = '0' * 32

    def in_flight_writes_finish():
        f.bucket.objects[f'projects/{PROJECT}/files/late/data.csv'] = b'late'
        f.db.docs[f'oe_projects/{PROJECT}/scripts/late'] = {'code': 'late'}
        f.db.docs[f'oe_projects/{late_project}'] = {'owner_uid': OWNER, 'members': {OWNER: {'role': 'owner'}}}
        f.bucket.objects[f'projects/{late_project}/files/a/b.csv'] = b'late'
        f.db.docs[f'oe_limits/account-link-{OWNER}-20261010'] = {'count': 1}
    clock.during_sleep = in_flight_writes_finish
    report = run(f, clock)
    assert report['projects_removed'] == 3 and report['account_link_quotas_removed'] == 2
    assert not any(late_project in path for path in f.db.docs)
    assert not any(late_project in name or 'late' in name for name in f.bucket.objects)
    assert_session_gone(f)
    assert_foreign_untouched(f)


def test_membership_shared_during_quiescence_refuses_before_any_deletion():
    f, clock = fixture(), Clock()

    def human_joins():
        f.db.docs[f'oe_projects/{PROJECT}']['members']['human'] = {'role': 'viewer'}
    clock.during_sleep = human_joins
    with pytest.raises(SystemExit, match='shared with another account'):
        run(f, clock)
    assert not any(kind == 'delete' for kind, _ in f.events)
    assert f.state['cleanup']['phase'] == 'quiescence'
    assert all(name in f.bucket.objects for name in SESSION_BLOBS)


def test_readback_residual_keeps_state_and_a_rerun_completes():
    f, clock = fixture(), Clock()
    f.bucket.unlisted_for_deletion.add(f'projects/{PROJECT}/files/f1/data.csv')
    with pytest.raises(SystemExit, match='readback found 1 remaining'):
        run(f, clock)
    assert f.state['cleanup']['phase'] == 'deletion'
    assert OWNER not in f.identities.records
    f.bucket.unlisted_for_deletion.clear()
    report = run(f, clock)
    assert report['residual_resources'] == 0 and report['blobs_removed'] == 4
    assert_session_gone(f)
    assert_foreign_untouched(f)


def test_label_session_labels_only_unclaimed_matching_legacy_identities():
    f = fixture()
    del f.state['session']
    for uid in (OWNER, EDITOR, VIEWER):
        f.identities.records[uid]['custom_claims'] = {}
    session = verify.label_session(f.identities, f.state)
    assert f.state['session'] == session and len(session) == 32
    assert all(f.identities.records[uid]['custom_claims'] == {verify.SESSION_CLAIM: session}
               for uid in (OWNER, EDITOR, VIEWER))
    assert f.identities.records['otherqa'] == FOREIGN_IDENTITIES['otherqa']
    with pytest.raises(SystemExit, match='already labelled'):
        verify.label_session(f.identities, f.state)


@pytest.mark.parametrize('problem', ['claims', 'email', 'missing'])
def test_label_session_refuses_claimed_mismatched_or_missing_identities(problem):
    f = fixture()
    del f.state['session']
    for uid in (OWNER, EDITOR, VIEWER):
        f.identities.records[uid]['custom_claims'] = {}
    if problem == 'claims':
        f.identities.records[EDITOR]['custom_claims'] = {verify.SESSION_CLAIM: OTHER_SESSION}
    elif problem == 'email':
        f.identities.records[EDITOR]['email'] = 'person@example.org'
    else:
        del f.identities.records[EDITOR]
    with pytest.raises(SystemExit):
        verify.label_session(f.identities, f.state)
    assert 'session' not in f.state and not any(kind == 'label' for kind, _ in f.events)


def test_cleanup_main_checks_session_and_live_contract_before_any_identity_change(tmp_path, monkeypatch):
    f = fixture()
    state_path = tmp_path / 'qa-state.json'
    state_path.write_text(json.dumps(f.state))
    monkeypatch.setattr(verify, 'STATE', state_path)
    monkeypatch.setattr(verify.subprocess, 'check_output', Mock(return_value='dummy-test-token'))
    monkeypatch.setattr(verify.firebase_admin, 'initialize_app', Mock(return_value=object()))
    monkeypatch.setattr(verify.firestore, 'Client', lambda *args, **kwargs: f.db)
    mutations = Mock(side_effect=AssertionError('No identity may change'))
    for name in ('update_user', 'revoke_refresh_tokens', 'delete_user', 'set_custom_user_claims'):
        monkeypatch.setattr(verify.auth, name, mutations)
    monkeypatch.setattr(verify.auth, 'get_user', Mock(side_effect=AssertionError('Not reached')))
    advertised = {}
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.return_value.json.side_effect = lambda: {'write_quiescence': advertised.get('value')}
    monkeypatch.setattr(verify.httpx, 'Client', Mock(return_value=client))
    monkeypatch.setattr('sys.argv', ['verify_team_live.py', 'cleanup', '--confirm-session', OTHER_SESSION])
    with pytest.raises(SystemExit, match='confirmed session does not match'):
        verify.main()
    client.get.assert_not_called()
    monkeypatch.setattr('sys.argv', ['verify_team_live.py', 'cleanup', '--confirm-session', SESSION])
    with pytest.raises(SystemExit, match='write-quiescence contract'):
        verify.main()
    mutations.assert_not_called()
    assert json.loads(state_path.read_text()) == f.state
