"""No network: assert job scoping, launch semantics, polling, and cancellation."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openecon import team_runner as module

PARENT = "projects/openecon-test/locations/us-central1"
JOB = PARENT + "/jobs/openecon-worker"
OPERATION = PARENT + "/operations/run-123"
EXECUTION = JOB + "/executions/openecon-worker-abc"
URL = "https://storage.googleapis.com/openecon-runs/input.json?X-Goog-Signature=x&X-Goog-Expires=900"
SERVICE_ACCOUNT = "openecon-worker@openecon-test.iam.gserviceaccount.com"


@pytest.fixture
def runner(monkeypatch):
    jobs, executions = Mock(), Mock()
    monkeypatch.setattr(module, "_clients", lambda: (jobs, executions))
    monkeypatch.setattr(module, "_operation_dict", lambda value: value)
    return module.GoogleJobRunner("openecon-test", "us-central1", "openecon-worker")


@pytest.mark.parametrize('code_seconds,task_seconds', [(45, 165), (60, 180), (120, 240)])
def test_start_limits_overrides_to_one_fixed_job_and_run_capability(runner, code_seconds, task_seconds):
    runner._jobs.run_job.return_value = SimpleNamespace(operation={
        "name": OPERATION, "metadata": {"name": EXECUTION}})
    result = runner.start(URL, timeout_seconds=code_seconds)
    assert result == {"operation": OPERATION, "execution": EXECUTION, "status": "queued", "message": None}
    call = runner._jobs.run_job.call_args
    assert call.kwargs["retry"] is None
    assert call.kwargs["timeout"] == 20
    assert call.kwargs["request"] == {"name": JOB, "overrides": {
        "task_count": 1, "timeout": {"seconds": task_seconds}, "container_overrides": [{
            "env": [{"name": "OPENECON_RUN_INPUT_URL", "value": URL}]}]}}


@pytest.mark.parametrize("raw,status", [
    ({"metadata": {"name": EXECUTION}}, "queued"),
    ({"metadata": {"name": EXECUTION, "start_time": "2026-10-01T12:00:00Z"}}, "running"),
    ({"done": True, "response": {"name": EXECUTION, "succeeded_count": 1}}, "succeeded"),
    ({"done": True, "error": {"code": 1, "message": "secret-capability"}}, "cancelled"),
    ({"done": True, "error": {"code": 13, "message": "secret-capability"}}, "failed"),
    ({"done": True, "response": {"name": EXECUTION, "failed_count": 1}}, "failed"),
    ({"done": True, "response": {"name": EXECUTION, "cancelled_count": 1}}, "cancelled"),
    ({"done": True}, "failed"),
    ({"done": True, "response": {"name": EXECUTION, "conditions": [
        {"type": "Completed", "state": "CONDITION_SUCCEEDED"}]}}, "succeeded"),
    ({"done": True, "response": {"name": EXECUTION, "conditions": [
        {"type": "Completed", "state": "CONDITION_FAILED"}]}}, "failed"),
])
def test_platform_status_not_user_results_controls_lifecycle(runner, raw, status):
    runner._jobs.transport.operations_client.get_operation.return_value = {"name": OPERATION, **raw}
    result = runner.status(OPERATION)
    assert result["status"] == status
    assert "secret-capability" not in str(result)


def test_cancel_scopes_to_configured_job(runner):
    runner._executions.cancel_execution.return_value = SimpleNamespace(operation={"name": OPERATION})
    assert runner.cancel(EXECUTION)["status"] == "cancelling"
    assert runner._executions.cancel_execution.call_args.kwargs["request"] == {"name": EXECUTION}
    with pytest.raises(ValueError):
        runner.cancel(EXECUTION.replace("openecon-worker", "other-worker"))
    with pytest.raises(ValueError):
        runner.status(OPERATION.replace("openecon-test", "other-project"))


def test_unknown_launch_outcome_is_not_retried(runner):
    runner._jobs.run_job.side_effect = RuntimeError(URL)
    with pytest.raises(module.JobRunnerError, match="do not automatically retry") as exc:
        runner.start(URL)
    assert URL not in str(exc.value)
    assert runner._jobs.run_job.call_count == 1


@pytest.mark.parametrize("duration", [0, -1, 121, True, float("inf")])
def test_runner_enforces_code_timeout(runner, duration):
    with pytest.raises(ValueError):
        runner.start(URL, timeout_seconds=duration)
    runner._jobs.run_job.assert_not_called()


def configured_job():
    run_v2 = pytest.importorskip('google.cloud.run_v2')
    return run_v2.Job(name=JOB, template={'task_count': 1, 'parallelism': 1, 'template': {
        'service_account': SERVICE_ACCOUNT, 'max_retries': 0,
        'containers': [{'image': 'example/image@sha256:a',
                        'env': [{'name': 'OMP_NUM_THREADS', 'value': '2'}]}]}})


def test_explicit_identity_checks_real_sdk_job_before_every_launch(runner):
    runner.service_account = SERVICE_ACCOUNT
    runner._jobs.get_job.return_value = configured_job()
    runner._jobs.run_job.return_value = SimpleNamespace(operation={'name': OPERATION})
    runner.start(URL)
    runner.start(URL)
    assert runner._jobs.get_job.call_count == 2
    assert runner._jobs.get_job.call_args.kwargs == {'request': {'name': JOB}, 'retry': None, 'timeout': 10}
    calls = [call[0] for call in runner._jobs.mock_calls]
    assert calls == ['get_job', 'run_job', 'get_job', 'run_job']


@pytest.mark.parametrize('mutation', ['identity', 'retries', 'tasks', 'parallelism', 'sidecar',
                                      'secret', 'volume', 'connector', 'network', 'foreign_job'])
def test_drifted_worker_is_rejected_before_a_capability_is_sent(runner, mutation):
    runner.service_account = SERVICE_ACCOUNT
    job = configured_job()
    task = job.template.template
    if mutation == 'identity':
        task.service_account = 'privileged@openecon-test.iam.gserviceaccount.com'
    elif mutation == 'retries':
        task.max_retries = 3
    elif mutation == 'tasks':
        job.template.task_count = 2
    elif mutation == 'parallelism':
        job.template.parallelism = 2
    elif mutation == 'sidecar':
        task.containers.append({'image': 'example/sidecar'})
    elif mutation == 'secret':
        task.containers[0].env.append({'name': 'TOKEN', 'value_source': {
            'secret_key_ref': {'secret': 'secret', 'version': 'latest'}}})
    elif mutation == 'volume':
        task.volumes.append({'name': 'credentials', 'secret': {'secret': 'secret'}})
    elif mutation == 'connector':
        task.vpc_access.connector = 'private-vpc'
    elif mutation == 'network':
        task.vpc_access.network_interfaces.append({'network': 'default'})
    elif mutation == 'foreign_job':
        job.name = JOB.replace('openecon-worker', 'other-worker')
    runner._jobs.get_job.return_value = job
    with pytest.raises(module.JobRunnerError, match='isolation requirements'):
        runner.start(URL)
    runner._jobs.run_job.assert_not_called()


def test_unreadable_worker_is_rejected_before_launch(runner):
    runner.service_account = SERVICE_ACCOUNT
    runner._jobs.get_job.side_effect = RuntimeError('private diagnostic')
    with pytest.raises(module.JobRunnerError, match='no run was launched') as error:
        runner.start(URL)
    assert 'private diagnostic' not in str(error.value)
    runner._jobs.run_job.assert_not_called()


@pytest.mark.parametrize('email', ['', 'default', 'a@example.com', SERVICE_ACCOUNT + '\n'])
def test_service_account_configuration_must_be_explicit_email(runner, email):
    with pytest.raises(ValueError, match='service account'):
        module.GoogleJobRunner('openecon-test', 'us-central1', 'openecon-worker', service_account=email)


@pytest.mark.parametrize('state,expected', [('CONDITION_SUCCEEDED', 'succeeded'), ('CONDITION_FAILED', 'failed')])
def test_real_protobuf_completed_condition_uses_type_underscored(runner, monkeypatch, state, expected):
    run_v2 = pytest.importorskip('google.cloud.run_v2')
    from google.longrunning.operations_pb2 import Operation
    from google.protobuf.json_format import MessageToDict
    execution = run_v2.Execution(name=EXECUTION, conditions=[{'type_': 'Completed', 'state': state}])
    operation = Operation(name=OPERATION, done=True)
    operation.response.Pack(run_v2.Execution.pb(execution))
    monkeypatch.setattr(module, '_operation_dict',
        lambda raw: MessageToDict(raw, preserving_proto_field_name=True))
    runner._jobs.transport.operations_client.get_operation.return_value = operation
    assert runner.status(OPERATION)['status'] == expected
