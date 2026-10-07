"""Offline checks of supervisor integrity, capability scope and streaming leases."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from unittest.mock import Mock

import pytest

from openecon import team_sandbox_runner as module
from openecon.team_auth import TeamIdentity
from openecon.team_runner import JobRunnerError
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamError, TeamStore

PID, RID = 'a' * 32, 'b' * 32
BUCKET = 'openecon-test-runs'
ORIGIN = 'https://openecon-compute-123456789012.us-central1.run.app'
NAME = 'projects/openecon-test/locations/us-central1/services/openecon-compute'
IMAGE = 'us-central1-docker.pkg.dev/openecon-test/apps/compute@sha256:' + 'c' * 64
SA = 'openecon-compute@openecon-test.iam.gserviceaccount.com'
CALLER_SUBJECT = '108062605432490492564'
OPERATION = f'sandbox:{PID}:{RID}'
COMPLETION = f'staging/{PID}/{RID}/completion.json'
CANCEL = f'staging/{PID}/{RID}/cancel.json'
INPUT = (f'https://storage.googleapis.com/{BUCKET}/staging/{PID}/{RID}/input.json'
         '?generation=123&X-Goog-Expires=1800&X-Goog-Signature=input-capability')


def service_config():
    revision = NAME + '/revisions/openecon-compute-00001-abc'
    return {
        'name': NAME, 'generation': '3', 'observedGeneration': '3',
        'uri': ORIGIN, 'urls': [ORIGIN], 'reconciling': False,
        'latestReadyRevision': revision, 'latestCreatedRevision': revision,
        'terminalCondition': {'state': 'CONDITION_SUCCEEDED'},
        'trafficStatuses': [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION',
                             'revision': revision, 'percent': 100}],
        'scaling': {'maxInstanceCount': 4},
        'template': {
            'serviceAccount': SA, 'executionEnvironment': 'EXECUTION_ENVIRONMENT_GEN2',
            'maxInstanceRequestConcurrency': 1, 'timeout': '360s',
            'containers': [{
                'image': IMAGE, 'sandboxLauncher': True,
                'command': ['/usr/bin/tini'],
                'args': ['--', '/opt/venv/bin/python', '-m', 'openecon.team_sandbox_broker'],
                'resources': {'limits': {'cpu': '2', 'memory': '4Gi'}},
                'env': [
                    {'name': 'OPENECON_MODE', 'value': 'sandbox-broker'},
                    {'name': 'OPENECON_SANDBOX_ROOTFS', 'value': '/opt/openecon-sandbox-rootfs'},
                    {'name': 'OPENECON_SANDBOX_ENABLED', 'value': '1'},
                    {'name': 'OPENECON_BROKER_AUDIENCE', 'value': ORIGIN},
                    {'name': 'OPENECON_BROKER_CALLER_EMAIL',
                     'value': 'openecon-control@openecon-test.iam.gserviceaccount.com'},
                    {'name': 'OPENECON_BROKER_CALLER_SUB', 'value': CALLER_SUBJECT},
                ],
            }],
        },
    }


class Store:
    def __init__(self):
        self.value = {
            'id': RID, 'project_id': PID, 'uid': 'creator-uid', 'email': 'creator@example.com',
            'timeout_seconds': 60, 'state': 'starting', 'operation': None,
            'deadline': (datetime.now(timezone.utc) + timedelta(minutes=18)).isoformat(),
            'cancel_requested': False,
        }
        self.role = 'editor'
        self.checked = []

    def run(self, project_id, run_id):
        assert (project_id, run_id) == (PID, RID)
        return deepcopy(self.value)

    def claim_run_dispatch(self, project_id, run_id, *, operation, execution):
        assert (project_id, run_id) == (PID, RID)
        if self.value.get('operation'):
            raise TeamError('ALREADY_STARTED', 'already dispatched', 409)
        self.value.update(operation=operation, execution=execution, state='running')
        return deepcopy(self.value)

    def project(self, project_id, user, minimum):
        assert project_id == PID and minimum == 'editor'
        self.checked.append(user)
        if self.role != 'editor':
            raise TeamError('ROLE_REQUIRED', 'permission changed', 403)
        return {'members': {user.uid: {'role': self.role}}}

    def update_run(self, project_id, run_id, **changes):
        assert (project_id, run_id) == (PID, RID)
        self.value.update(changes)


class Storage(MemoryStorage):
    def __init__(self):
        super().__init__()
        self.policies = []

    def signed_cancel_url(self, project_id, run_id):
        assert (project_id, run_id) == (PID, RID)
        return (f'https://storage.googleapis.com/{BUCKET}/{CANCEL}'
                '?X-Goog-Expires=1800&X-Goog-Signature=cancel-capability')

    def output_policy(self, key, maximum):
        self.policies.append((key, maximum))
        return {'url': f'https://storage.googleapis.com/{BUCKET}/',
                'fields': {'key': key, 'policy': 'bounded-policy', 'x-goog-signature': 'parent-secret'}}


class Response(BytesIO):
    status = 200

    def __init__(self, payload, *, on_line=None):
        super().__init__(payload)
        self.on_line = on_line
        self.read_limits = []

    def readline(self, limit=-1):
        self.read_limits.append(limit)
        if self.on_line:
            self.on_line()
        return super().readline(limit)


def event(status='running', run_id=RID):
    return json.dumps({'execution_id': run_id, 'status': status}).encode() + b'\n'


def complete(runner, status='succeeded', **changes):
    marker = {'execution_id': RID, 'status': status, 'cleanup_confirmed': True, **changes}
    runner.storage.objects[COMPLETION] = json.dumps(marker).encode()


@pytest.fixture
def runner(monkeypatch):
    result = module.GoogleSandboxRunner(
        'openecon-test', 'us-central1', 'openecon-compute', origin=ORIGIN, bucket=BUCKET,
        service_account=SA, image=IMAGE, storage=Storage(), store=Store(), caller_subject=CALLER_SUBJECT)
    result._services = Mock()
    result._services.get_service.return_value = service_config()
    result._services.get_iam_policy.return_value = {'bindings': [
        {'role': 'roles/run.invoker',
         'members': ['serviceAccount:openecon-control@openecon-test.iam.gserviceaccount.com']},
    ]}
    monkeypatch.setattr(module, '_identity_token', Mock(return_value='private-control-id-token'))
    monkeypatch.setattr(module, '_open_rpc', Mock())
    return result


def test_start_claims_before_rpc_and_sends_only_parent_capabilities(runner):
    def open_rpc(request, timeout):
        assert runner.store.value['operation'] == OPERATION
        assert runner.store.value['execution'] == OPERATION
        assert timeout == 90
        assert request.full_url == ORIGIN + '/execute'
        assert request.get_header('X-serverless-authorization') == 'Bearer private-control-id-token'
        assert request.get_header('Authorization') == request.get_header('X-serverless-authorization')
        body = json.loads(request.data)
        assert set(body) == {'execution_id', 'input_url', 'cancel_url', 'completion_upload', 'timeout_seconds'}
        assert body['execution_id'] == RID and body['input_url'] == INPUT
        assert body['timeout_seconds'] == 60
        assert body['completion_upload']['fields']['key'] == COMPLETION
        assert 'generation' not in body['cancel_url']
        complete(runner)
        return Response(event() + event('succeeded'))

    module._open_rpc.side_effect = open_rpc
    result = runner.start(INPUT, 60)
    assert result == {'operation': OPERATION, 'execution': OPERATION,
                      'status': 'succeeded', 'message': None}
    assert runner.storage.policies == [(COMPLETION, 4096)]
    assert runner.store.checked == [TeamIdentity('creator-uid', 'creator@example.com', email_verified=True)] * 3
    module._identity_token.assert_called_once_with(ORIGIN)
    runner._services.get_service.assert_called_once_with(request={'name': NAME}, retry=None, timeout=10)


@pytest.mark.parametrize('short', [False, True])
def test_explicit_traffic_accepts_only_exact_ready_full_resource_or_basename(runner, short):
    service = runner._services.get_service.return_value
    ready = service['latestReadyRevision']
    service['trafficStatuses'][0]['revision'] = ready.rsplit('/', 1)[1] if short else ready
    runner._verify_worker()


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(name=NAME.replace('openecon-test', 'foreign-test')),
    lambda s: s.update(reconciling=True),
    lambda s: s.update(observedGeneration='2'),
    lambda s: s.update(latestCreatedRevision=NAME + '/revisions/other'),
    lambda s: s.update(terminalCondition={'state': 'CONDITION_FAILED'}),
    lambda s: s['trafficStatuses'].append({'revision': NAME + '/revisions/other', 'percent': 1}),
    lambda s: s['trafficStatuses'][0].update(percent=50),
    lambda s: s['trafficStatuses'][0].update(revision='openecon-compute-00002-other'),
    lambda s: s['trafficStatuses'][0].update(revision=s['latestReadyRevision'].replace('/services/openecon-compute/', '/services/other-service/')),
    lambda s: s['trafficStatuses'][0].update(revision='other-service-00001-abc'),
    lambda s: s['trafficStatuses'][0].update(type='TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST'),
    lambda s: s.update(trafficStatuses=[{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST', 'percent': 100}]),
    lambda s: s['trafficStatuses'][0].pop('type'),
    lambda s: s.update(invokerIamDisabled=True),
    lambda s: s.update(sshEnabled=True),
    lambda s: s.update(multiRegionSettings={'regions': ['us-central1', 'europe-west1']}),
    lambda s: s.update(uri='https://foreign.run.app', urls=[]),
    lambda s: s['template'].update(serviceAccount='privileged@openecon-test.iam.gserviceaccount.com'),
    lambda s: s['template'].update(executionEnvironment='EXECUTION_ENVIRONMENT_GEN1'),
    lambda s: s['template'].update(maxInstanceRequestConcurrency=80),
    lambda s: s['scaling'].update(maxInstanceCount=5),
    lambda s: s['scaling'].update(maxInstanceCount=0),
    lambda s: s['scaling'].update(minInstanceCount=1),
    lambda s: s['scaling'].update(scalingMode='MANUAL'),
    lambda s: s['template'].update(volumes=[{'name': 'shared'}]),
    lambda s: s['template'].update(vpcAccess={'connector': 'private-network'}),
    lambda s: s['template'].update(timeout='900s'),
    lambda s: s['template']['containers'].append(deepcopy(s['template']['containers'][0])),
    lambda s: s['template']['containers'][0].update(image=IMAGE.replace('c' * 64, 'd' * 64)),
    lambda s: s['template']['containers'][0].update(sandboxLauncher=False),
    lambda s: s['template']['containers'][0].pop('sandboxLauncher'),
    lambda s: s['template']['containers'][0].update(volumeMounts=[{'name': 'shared'}]),
    lambda s: s['template']['containers'][0].update(command=['sh'], args=['-c', 'evil']),
    lambda s: s['template']['containers'][0].update(command=['python'], args=['-m', 'openecon.team_sandbox_broker']),
    lambda s: s['template']['containers'][0].update(command=['/usr/bin/tini', '--', '/opt/venv/bin/python', '-m', 'openecon.team_sandbox_broker'], args=[]),
    lambda s: s['template']['containers'][0].update(workingDir='/tmp'),
    lambda s: s['template']['containers'][0]['env'].append({'name': 'PYTHONPATH', 'value': '/tmp'}),
    lambda s: s['template']['containers'][0]['env'][1].update(value='/'),
    lambda s: s['template']['containers'][0]['env'][0].update(valueSource={'secretKeyRef': {'secret': 'private'}}),
    lambda s: s['template']['containers'][0]['env'][3].update(value='https://other-123456789012.us-central1.run.app'),
    lambda s: s['template']['containers'][0]['env'][4].update(value='attacker@openecon-test.iam.gserviceaccount.com'),
    lambda s: s['template']['containers'][0]['env'][5].update(value='999999999999999999999'),
    lambda s: s['template']['containers'][0]['env'].pop(),
    lambda s: s['template']['containers'][0]['env'].append(deepcopy(s['template']['containers'][0]['env'][-1])),
    lambda s: s['template']['containers'][0]['resources']['limits'].update(memory='8Gi'),
    lambda s: s['template']['containers'][0]['resources'].update(cpuIdle=False),
])
def test_service_drift_fails_closed_before_any_rpc_or_capability_grant(runner, mutation):
    mutation(runner._services.get_service.return_value)
    with pytest.raises(JobRunnerError):
        runner.start(INPUT, 60)
    module._open_rpc.assert_not_called()
    module._identity_token.assert_not_called()
    assert not runner.storage.policies
    assert runner.status(OPERATION)['status'] == 'failed'  # Known: no sandbox existed.


@pytest.mark.parametrize('bindings', [
    [],
    [{'role': 'roles/run.invoker', 'members': ['allUsers']}],
    [{'role': 'roles/run.invoker', 'members': ['allAuthenticatedUsers']}],
    [{'role': 'roles/run.invoker', 'members': ['serviceAccount:foreign@openecon-test.iam.gserviceaccount.com']}],
    [{'role': 'roles/run.invoker',
      'members': ['serviceAccount:openecon-control@openecon-test.iam.gserviceaccount.com'],
      'condition': {'expression': 'true'}}],
    [{'role': 'roles/run.invoker',
      'members': ['serviceAccount:openecon-control@openecon-test.iam.gserviceaccount.com']},
     {'role': 'roles/run.admin', 'members': ['allUsers']}],
    [{'role': 'roles/run.viewer',
      'members': ['serviceAccount:openecon-control@openecon-test.iam.gserviceaccount.com']}],
])
def test_public_or_unexpected_invoker_iam_fails_closed(runner, bindings):
    runner._services.get_iam_policy.return_value = {'bindings': bindings}
    with pytest.raises(JobRunnerError):
        runner.start(INPUT, 60)
    module._open_rpc.assert_not_called()
    module._identity_token.assert_not_called()
    assert not runner.storage.policies


def test_unreadable_iam_cannot_grant_capabilities(runner):
    runner._services.get_iam_policy.side_effect = PermissionError('private upstream detail')
    with pytest.raises(JobRunnerError) as caught:
        runner.start(INPUT, 60)
    assert 'upstream' not in str(caught.value)
    module._open_rpc.assert_not_called()


@pytest.mark.parametrize('url', [
    INPUT.replace(BUCKET, 'foreign-bucket'), INPUT.replace('/input.json', '/output.json'),
    INPUT.replace('/staging/', '/projects/'), INPUT.replace(PID, PID.upper()),
    INPUT.replace('/input.json', '/%69nput.json'), INPUT.replace('generation=123&', ''),
    INPUT.replace('storage.googleapis.com', 'storage.googleapis.com.evil.test'),
])
def test_input_must_be_pinned_to_exact_bucket_project_run(runner, url):
    with pytest.raises(ValueError):
        runner.start(url, 60)
    runner._services.get_service.assert_not_called()
    module._open_rpc.assert_not_called()
    assert runner.store.value['operation'] is None


@pytest.mark.parametrize('timeout', [0, 121, True, '60', float('nan'), float('inf')])
def test_timeout_bounds_and_reserved_run_are_preserved(runner, timeout):
    with pytest.raises(ValueError):
        runner.start(INPUT, timeout)
    module._open_rpc.assert_not_called()


def test_already_dispatched_execution_is_never_retried(runner):
    runner.store.value.update(operation=OPERATION, state='running')
    with pytest.raises(JobRunnerError):
        runner.start(INPUT, 60)
    module._open_rpc.assert_not_called()


def test_reserved_timeout_cannot_be_changed_during_dispatch(runner):
    with pytest.raises(JobRunnerError):
        runner.start(INPUT, 120)
    module._open_rpc.assert_not_called()


def test_failed_atomic_claim_does_not_cancel_someone_elses_dispatch(runner):
    runner.store.claim_run_dispatch = Mock(side_effect=TeamError('ALREADY_STARTED', 'already running', 409))
    with pytest.raises(TeamError):
        runner.start(INPUT, 60)
    assert not runner.storage.objects
    module._open_rpc.assert_not_called()


@pytest.mark.parametrize('change', ['cancel', 'demote'])
def test_cancel_or_creator_demotion_during_heartbeat_writes_marker(runner, change):
    count = 0
    def on_line():
        nonlocal count
        count += 1
        if count == 1:
            if change == 'cancel':
                runner.store.value['cancel_requested'] = True
            else:
                runner.store.role = 'viewer'
        if count == 2:
            assert json.loads(runner.storage.objects[CANCEL]) == {'execution_id': RID, 'cancel_requested': True}
            complete(runner, 'cancelled')
    module._open_rpc.return_value = Response(event() + event('cancelled'), on_line=on_line)
    assert runner.start(INPUT, 60)['status'] == 'cancelled'
    assert runner.store.value['cancel_requested']


def test_cancel_before_dispatch_never_launches_or_grants_supervisor_caps(runner):
    runner.store.value['cancel_requested'] = True
    assert runner.start(INPUT, 60)['status'] == 'cancelled'
    module._open_rpc.assert_not_called()
    module._identity_token.assert_not_called()
    assert not runner.storage.policies
    assert runner.status(OPERATION)['status'] == 'cancelled'


def test_unknown_transport_outcome_has_one_post_and_no_fake_completion(runner):
    module._open_rpc.side_effect = OSError('private token and URL must not escape')
    with pytest.raises(JobRunnerError) as caught:
        runner.start(INPUT, 60)
    assert module._open_rpc.call_count == 1
    assert COMPLETION not in runner.storage.objects
    assert CANCEL in runner.storage.objects
    assert runner.status(OPERATION)['status'] == 'running'
    assert 'private token' not in str(caught.value)


@pytest.mark.parametrize('payload', [
    event('succeeded'), b'x' * 4097 + b'\n', b'{"status":"running"}',
    event('running', run_id='c' * 32), b'[]\n',
    ('{"execution_id":"' + RID + '","status":"running","status":"succeeded"}\n').encode(),
])
def test_bad_or_unconfirmed_stream_cannot_release_capacity(runner, payload):
    response = Response(payload)
    module._open_rpc.return_value = response
    with pytest.raises(JobRunnerError):
        runner.start(INPUT, 60)
    assert COMPLETION not in runner.storage.objects
    assert CANCEL in runner.storage.objects
    assert set(response.read_limits) == {4097}


def test_heartbeat_count_is_bounded_even_when_every_line_is_valid(runner, monkeypatch):
    monkeypatch.setattr(module, 'MAX_EVENTS', 3)
    response = Response(event() * 4)
    module._open_rpc.return_value = response
    with pytest.raises(JobRunnerError):
        runner.start(INPUT, 60)
    assert len(response.read_limits) == 3
    assert CANCEL in runner.storage.objects


def test_user_result_does_not_prove_sandbox_exit(runner):
    runner.storage.put(f'staging/{PID}/{RID}/output.json', b'{"status":"success"}')
    assert runner.status(OPERATION)['status'] == 'running'


@pytest.mark.parametrize('marker', [
    {'execution_id': RID, 'status': 'succeeded'},
    {'execution_id': RID, 'status': 'succeeded', 'cleanup_confirmed': 1},
    {'execution_id': 'c' * 32, 'status': 'succeeded', 'cleanup_confirmed': True},
    {'execution_id': RID, 'status': 'running', 'cleanup_confirmed': True},
    {'execution_id': RID, 'status': 'succeeded', 'cleanup_confirmed': True, 'untrusted': 'extra'},
    [], 'success',
])
def test_only_scoped_parent_completion_with_confirmed_cleanup_is_terminal(runner, marker):
    runner.storage.objects[COMPLETION] = json.dumps(marker).encode()
    with pytest.raises(JobRunnerError):
        runner.status(OPERATION)


def test_oversized_completion_fails_closed(runner):
    runner.storage.objects[COMPLETION] = b' ' * 4097
    with pytest.raises(JobRunnerError):
        runner.status(OPERATION)


@pytest.mark.parametrize('name', [
    f'sandbox:{PID}:../other', f'sandbox:{PID}:{RID}:suffix', OPERATION.upper(),
    NAME + '/executions/task', None,
])
def test_cancel_and_status_accept_only_canonical_operation_names(runner, name):
    for method in (runner.cancel, runner.status):
        with pytest.raises(ValueError):
            method(name)
    assert not runner.storage.objects


def test_cancel_is_idempotent_and_never_claims_completion(runner):
    assert runner.cancel(OPERATION)['status'] == 'cancelling'
    before = deepcopy(runner.storage.objects)
    runner.cancel(OPERATION)
    assert runner.storage.objects == before
    assert runner.status(OPERATION)['status'] == 'running'


@pytest.mark.parametrize('origin', [
    'http://example.run.app', ORIGIN + '/', ORIGIN + '/execute', ORIGIN + ':443',
    ORIGIN + '?x=y', 'https://user@example.run.app', 'https://example.run.app.evil.test',
    'https://openecon-compute-hashed-uc.a.run.app',
    ORIGIN.replace('openecon-compute', 'other-service'), ORIGIN.replace('us-central1', 'europe-west1'),
])
def test_origin_is_fixed_canonical_cloud_run_https(origin):
    with pytest.raises(ValueError):
        module.GoogleSandboxRunner('openecon-test', 'us-central1', 'openecon-compute',
            origin=origin, bucket=BUCKET, service_account=SA, image=IMAGE, storage=None, store=None,
            caller_subject=CALLER_SUBJECT)


@pytest.mark.parametrize('subject', [None, '', 'control@example.com', '123', '000000000000000000000',
                                   108062605432490492564, '1' * 33, '1\n23456789'])
def test_control_numeric_subject_must_be_explicit_and_canonical(subject):
    with pytest.raises(ValueError, match='numeric identity'):
        module.GoogleSandboxRunner('openecon-test', 'us-central1', 'openecon-compute',
            origin=ORIGIN, bucket=BUCKET, service_account=SA, image=IMAGE, storage=None, store=None,
            caller_subject=subject)


def test_raw_rest_reader_preserves_preview_sandbox_flag(monkeypatch):
    auth = pytest.importorskip('google.auth')
    transport = pytest.importorskip('google.auth.transport.requests')
    response = Mock(status_code=200)
    response.raw.read.return_value = json.dumps(service_config()).encode()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    session = Mock()
    session.get.return_value = response
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(auth, 'default', Mock(return_value=(object(), 'openecon-test')))
    monkeypatch.setattr(transport, 'AuthorizedSession', Mock(return_value=session))
    raw = module._ServiceClient().get_service(request={'name': NAME}, retry=None, timeout=10)
    assert raw['template']['containers'][0]['sandboxLauncher'] is True
    session.get.assert_called_once_with('https://run.googleapis.com/v2/' + NAME,
                                       timeout=10, allow_redirects=False, stream=True)
    response.raw.read.assert_called_once_with(262145, decode_content=True)
    session.get.reset_mock()
    module._ServiceClient().get_iam_policy(request={'name': NAME}, retry=None, timeout=10)
    session.get.assert_called_once_with(
        'https://run.googleapis.com/v2/' + NAME + ':getIamPolicy?options.requestedPolicyVersion=3',
        timeout=10, allow_redirects=False, stream=True)


def test_real_store_reserves_dispatch_and_concurrent_publication_is_not_reopened(runner):
    owner = TeamIdentity('owner', 'owner@example.com', email_verified=True)
    store = TeamStore(MemoryDocuments(), owner_email=owner.email, public_origin='https://app.example.com')
    store.me(owner)
    project = store.create_project(owner, 'Analysis')
    pid = project['id']
    run = store.begin_run(pid, owner, '1 + 1', 60)
    rid = run['id']
    name = f'sandbox:{pid}:{rid}'
    runner.store, runner.storage = store, MemoryStorage()
    runner.storage.signed_cancel_url = Mock(return_value='bounded-cancel-capability')
    runner.storage.output_policy = Mock(return_value={'bounded': 'parent-only-policy'})

    def open_rpc(request, timeout):
        current = store.run(pid, rid)
        assert current['operation'] == current['execution'] == name
        assert store.project(pid, owner)['active_run'] == rid
        with pytest.raises(TeamError) as caught:
            store.begin_run(pid, owner, 'second execution', 60)
        assert caught.value.code == 'CONSOLE_BUSY'
        runner.storage.put(f'staging/{pid}/{rid}/completion.json', json.dumps({
            'execution_id': rid, 'status': 'succeeded', 'cleanup_confirmed': True,
        }).encode())
        # A concurrent console poll can publish the trusted completion before
        # the still-open dispatch request receives its final HTTP heartbeat.
        store.finish_run(pid, rid, None, {'status': 'success'})
        return Response(event('succeeded', run_id=rid))

    module._open_rpc.side_effect = open_rpc
    result = runner.start(INPUT.replace(PID, pid).replace(RID, rid), 60)
    assert result['status'] == 'succeeded'
    attached = store.attach_run_execution(pid, rid, operation=result['operation'], execution=result['execution'])
    assert attached['state'] == 'finished'
    assert store.project(pid, owner)['active_run'] is None
    with pytest.raises(JobRunnerError):
        runner.start(INPUT.replace(PID, pid).replace(RID, rid), 60)
    assert module._open_rpc.call_count == 1
