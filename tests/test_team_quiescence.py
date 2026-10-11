"""Every admitted team request may start backend writes only inside its window."""
from pathlib import Path
import re
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from openecon import team_quiescence as quiescence
from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_server import create_team_app
from openecon.team_storage import MemoryStorage
from openecon.team_store import FirestoreDocuments, MemoryDocuments, TeamError, TeamStore

ORIGIN = 'https://app.example.com'
FIREBASE_PROJECT = 'openecon-test'
FIREBASE_CONFIG = {'projectId': FIREBASE_PROJECT, 'authDomain': f'{FIREBASE_PROJECT}.firebaseapp.com'}


class Clock:
    def __init__(self):
        self.now = 5_000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(quiescence, 'time', fake)
    return fake


@pytest.fixture
def api(clock):
    db, storage = MemoryDocuments(), MemoryStorage()
    owner = TeamIdentity('owner', 'owner@example.com', 'owner', True)
    store = TeamStore(db, owner_email=owner.email, public_origin=ORIGIN)
    store.me(owner)
    pid = store.create_project(owner, 'Shared project')['id']
    delays = {}

    def verify(token):
        # Authentication happens after admission; a slow check consumes the window.
        clock.now += delays.get(token, 0)
        return {'uid': 'owner', 'sub': 'owner', 'email': owner.email, 'name': 'owner',
                'aud': FIREBASE_PROJECT, 'iss': f'https://securetoken.google.com/{FIREBASE_PROJECT}',
                'firebase': {'sign_in_provider': 'password'}, 'email_verified': True}
    app = create_team_app(store=store, storage=storage, auth=TeamAuth(FIREBASE_PROJECT, verifier=verify),
                          public_origin=ORIGIN, firebase_config=FIREBASE_CONFIG)
    with TestClient(app, base_url=ORIGIN) as client:
        yield SimpleNamespace(client=client, db=db, storage=storage, delays=delays,
                              prefix=f'/api/projects/{pid}/workspace')


def headers(token):
    return {'Authorization': f'Bearer {token}'}


def test_service_advertises_the_exact_contract(api):
    advertised = api.client.get('/api/auth/config').json()['write_quiescence']
    assert advertised == quiescence.CONTRACT == {'version': 1, 'side_effect_seconds': 360,
                                                 'quiescence_seconds': 690}


def test_write_window_equals_the_enforced_cloud_run_request_timeout():
    deploy = (Path(__file__).parents[1] / 'scripts/deploy_team_control.py').read_text()
    assert re.findall(r"template\.get\('timeout'\) == '(\d+)s'", deploy) == [str(quiescence.SIDE_EFFECT_SECONDS)]
    assert quiescence.QUIESCENCE_SECONDS == (quiescence.SIDE_EFFECT_SECONDS + quiescence.TRANSACTION_SECONDS
                                             + quiescence.CLOCK_MARGIN_SECONDS)


@pytest.mark.parametrize('delay,admitted', [(0, True), (quiescence.SIDE_EFFECT_SECONDS, True),
                                            (quiescence.SIDE_EFFECT_SECONDS + 1, False)])
def test_storage_and_firestore_writes_need_an_open_request_window(api, delay, admitted):
    api.delays['slow'] = delay
    before_docs, before_objects = dict(api.db.data), dict(api.storage.objects)
    upload = api.client.post(api.prefix + '/datasets/upload', headers=headers('slow'),
                             files={'file': ('data.csv', b'x,y\n1,2\n', 'text/csv')})
    script = api.client.put(api.prefix + '/console/script', headers=headers('slow'),
                            json={'code': 'x = 1', 'version': 0})
    if admitted:
        assert upload.status_code == 201 and script.status_code == 200
        assert len(api.storage.objects) == len(before_objects) + 1
    else:
        for response in (upload, script):
            assert response.status_code == 503
            assert response.json()['detail']['code'] == 'REQUEST_EXPIRED'
        assert api.storage.objects == before_objects
        assert api.db.data == before_docs


def test_each_request_gets_a_fresh_window(api, clock):
    api.delays['slow'] = quiescence.SIDE_EFFECT_SECONDS + 1
    assert api.client.put(api.prefix + '/console/script', headers=headers('slow'),
                          json={'code': 'late', 'version': 0}).status_code == 503
    assert api.client.put(api.prefix + '/console/script', headers=headers('fast'),
                          json={'code': 'x = 1', 'version': 0}).status_code == 200


def test_code_outside_a_request_has_no_window(clock):
    clock.now += 10 * quiescence.SIDE_EFFECT_SECONDS
    MemoryDocuments().atomic(lambda db: db.put('oe_test/doc', {'ok': True}))
    MemoryStorage().put('projects/x/file', b'data')


def test_every_firestore_transaction_attempt_needs_the_window(clock, monkeypatch):
    from google.cloud import firestore

    class Retry(Exception):
        pass

    class Transaction:
        def __init__(self):
            self.writes = []

        def set(self, reference, value):
            self.writes.append(reference.path)

        def delete(self, reference):
            self.writes.append(reference.path)

    transaction, reads = Transaction(), []

    def document(path):
        def get(transaction=None):
            reads.append(path)
            return SimpleNamespace(to_dict=lambda: None)
        return SimpleNamespace(path=path, get=get)

    def transactional(apply):
        def run(transaction):
            try:
                return apply(transaction)
            except Retry:
                clock.now += quiescence.SIDE_EFFECT_SECONDS + 1  # contention retry starts late
                return apply(transaction)
        return run

    monkeypatch.setattr(firestore, 'transactional', transactional)
    documents = object.__new__(FirestoreDocuments)
    documents.client = SimpleNamespace(document=document, transaction=lambda max_attempts: transaction)
    attempts = []

    def mutate(db):
        attempts.append(db.get('oe_projects/p'))
        db.put('oe_projects/p', {'name': 'changed'})
        if len(attempts) == 1:
            raise Retry
    token = quiescence.admit()
    try:
        with pytest.raises(TeamError) as error:
            documents.atomic(mutate)
    finally:
        quiescence.release(token)
    assert error.value.code == 'REQUEST_EXPIRED'
    assert len(attempts) == 1 and reads == ['oe_projects/p'] and transaction.writes == []
    token = quiescence.admit()
    try:
        documents.atomic(lambda db: db.put('oe_projects/q', {'name': 'saved'}))
    finally:
        quiescence.release(token)
    assert transaction.writes == ['oe_projects/q']


@pytest.mark.parametrize('production', [False, True])
def test_storage_uploads_need_the_window_before_touching_the_bucket(clock, production):
    from openecon.team_storage import TeamStorage
    created = []
    if production:
        storage = object.__new__(TeamStorage)
        storage.bucket = SimpleNamespace(blob=lambda key: created.append(key) or SimpleNamespace(
            upload_from_string=lambda *args, **kwargs: None, generation=1))
    else:
        storage = MemoryStorage()
    token = quiescence.admit()
    try:
        storage.put('projects/p/files/early', b'x')
        clock.now += quiescence.SIDE_EFFECT_SECONDS + 1
        with pytest.raises(TeamError) as error:
            storage.put('projects/p/files/late', b'x')
    finally:
        quiescence.release(token)
    assert error.value.code == 'REQUEST_EXPIRED'
    keys = created if production else list(storage.objects)
    assert keys == ['projects/p/files/early']
