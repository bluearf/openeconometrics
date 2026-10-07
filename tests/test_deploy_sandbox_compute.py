"""Deployment binds both broker authentication layers to one fixed caller."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'sandbox_compute_deploy', Path(__file__).parents[1] / 'scripts/deploy_sandbox_compute.py')
deploy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy)
IMAGE = 'us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:' + 'a' * 64
SUBJECT = '108062605432490492564'


def test_broker_deployment_pins_full_application_auth_and_existing_isolation_contract():
    config = deploy.service_config('openecon-sandbox', IMAGE, caller_subject=SUBJECT)
    container = config['template']['containers'][0]
    assert {entry['name']: entry['value'] for entry in container['env']} == {
        'OPENECON_MODE': 'sandbox-broker',
        'OPENECON_SANDBOX_ROOTFS': '/opt/openecon-sandbox-rootfs',
        'OPENECON_SANDBOX_ENABLED': '1',
        'OPENECON_BROKER_AUDIENCE': 'https://openecon-sandbox-291739190496.us-central1.run.app',
        'OPENECON_BROKER_CALLER_EMAIL': deploy.CONTROL,
        'OPENECON_BROKER_CALLER_SUB': SUBJECT,
    }
    assert container['command'] == ['/usr/bin/tini']
    assert container['args'] == ['--', '/opt/venv/bin/python', '-m', 'openecon.team_sandbox_broker']
    assert container['resources'] == {'limits': {'cpu': '2', 'memory': '4Gi'},
                                      'cpuIdle': True, 'startupCpuBoost': True}
    assert container['sandboxLauncher'] is True
    assert config['template']['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 4}
    assert config['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 4}
    assert config['template']['maxInstanceRequestConcurrency'] == 1
    assert config['template']['serviceAccount'] == deploy.COMPUTE


@pytest.mark.parametrize('subject', [None, '', 'operator@example.com', 123456789, '123',
                                   '0123456789', '1' * 33])
def test_broker_config_cannot_omit_or_guess_control_subject(subject):
    with pytest.raises(ValueError, match='numeric control'):
        deploy.service_config('openecon-sandbox', IMAGE, caller_subject=subject)


def test_disposable_fixed_probe_remains_separate_from_production_broker():
    config = deploy.service_config('openecon-sandbox-check', IMAGE, probe_source='fixed_probe()')
    container = config['template']['containers'][0]
    assert container['env'] == []
    assert container['args'] == ['--', '/opt/venv/bin/python', '-c', 'fixed_probe()']
    assert config['template']['scaling']['maxInstanceCount'] == 1
    assert config['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 1}


def test_compute_helper_reads_fixed_iam_account_subject_without_changing_iam():
    operator = object.__new__(deploy.Operator)
    operator.call = Mock(return_value={'email': deploy.CONTROL, 'uniqueId': SUBJECT})
    assert operator.control_subject() == SUBJECT
    operator.call.assert_called_once_with('GET',
        f'https://iam.googleapis.com/v1/projects/{deploy.PROJECT}/serviceAccounts/{deploy.CONTROL}')


@pytest.mark.parametrize('account', [
    {'email': 'other@example.com', 'uniqueId': SUBJECT},
    {'email': deploy.CONTROL, 'uniqueId': SUBJECT, 'disabled': True},
    {'email': deploy.CONTROL}, {'email': deploy.CONTROL, 'uniqueId': 123456789},
])
def test_compute_helper_rejects_unverified_or_disabled_identity(account):
    operator = object.__new__(deploy.Operator)
    operator.call = Mock(return_value=account)
    with pytest.raises(RuntimeError, match='identity'):
        operator.control_subject()


def ready_service(config):
    revision = config['name'] + '/revisions/' + config['name'].rsplit('/', 1)[1] + '-00002-qa'
    return {
        'name': config['name'], 'etag': 'fresh-service-version',
        'generation': '2', 'observedGeneration': '2',
        'latestReadyRevision': revision, 'latestCreatedRevision': revision,
        'terminalCondition': {'state': 'CONDITION_SUCCEEDED'},
        'template': deepcopy(config['template']),
        # REST omits the zero minimum; the maximum must still be explicit.
        'scaling': {'maxInstanceCount': config['scaling']['maxInstanceCount']},
        'trafficStatuses': [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST', 'percent': 100}],
    }


@pytest.mark.parametrize('short', [False, True])
def test_pin_traffic_uses_ready_immutable_revision_and_etag_then_waits_for_readback(short):
    config = deploy.service_config('openecon-sandbox', IMAGE, caller_subject=SUBJECT)
    service = ready_service(config)
    published = deepcopy(service)
    ready = service['latestReadyRevision']
    published['trafficStatuses'] = [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION',
        'revision': ready.rsplit('/', 1)[1] if short else ready, 'percent': 100}]
    operator = Mock()
    operation = {'name': 'traffic-operation'}
    operator.call.side_effect = [service, operation, published]
    assert deploy.pin_ready_traffic(operator, config) == published
    url = 'https://run.googleapis.com/v2/' + config['name']
    assert operator.call.call_args_list[1].args == ('PATCH', url + '?updateMask=traffic', {
        'name': config['name'], 'etag': 'fresh-service-version',
        'traffic': [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION',
            'revision': 'openecon-sandbox-00002-qa', 'percent': 100}],
    })
    operator.wait.assert_called_once_with(operation)
    assert [call[0] for call in operator.method_calls] == ['call', 'call', 'wait', 'call']


@pytest.mark.parametrize('change', [
    lambda s: s.update(reconciling=True),
    lambda s: s.update(observedGeneration='1'),
    lambda s: s.update(latestCreatedRevision=s['latestReadyRevision'] + '-newer'),
    lambda s: s.update(latestReadyRevision='projects/other/revisions/not-ours'),
    lambda s: s['terminalCondition'].update(state='CONDITION_FAILED'),
    lambda s: s['template']['containers'][0].update(image=IMAGE[:-64] + 'b' * 64),
    lambda s: s.update(etag=''),
])
def test_pin_refuses_unready_changed_or_unversioned_service_before_mutation(change):
    config = deploy.service_config('openecon-sandbox', IMAGE, caller_subject=SUBJECT)
    service = ready_service(config)
    change(service)
    operator = Mock()
    operator.call.return_value = service
    with pytest.raises(RuntimeError):
        deploy.pin_ready_traffic(operator, config)
    assert operator.call.call_count == 1
    operator.wait.assert_not_called()


@pytest.mark.parametrize('change', [
    lambda s: s['scaling'].update(maxInstanceCount=100),
    lambda s: s['scaling'].update(minInstanceCount=1),
    lambda s: s['trafficStatuses'][0].update(percent=50),
    lambda s: s['trafficStatuses'][0].update(revision='another-revision'),
    lambda s: s['trafficStatuses'][0].update(revision=s['latestReadyRevision'].replace('/services/openecon-sandbox/', '/services/other-service/')),
    lambda s: s['trafficStatuses'][0].update(type='TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST'),
    lambda s: s.update(trafficStatuses=[{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST', 'percent': 100}]),
])
def test_pin_fails_closed_when_readback_retains_default_scaling_or_unpinned_traffic(change):
    config = deploy.service_config('openecon-sandbox', IMAGE, caller_subject=SUBJECT)
    service = ready_service(config)
    published = deepcopy(service)
    published['trafficStatuses'] = [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION',
        'revision': service['latestReadyRevision'], 'percent': 100}]
    change(published)
    operator = Mock()
    operator.call.side_effect = [service, {'name': 'traffic-operation'}, published]
    with pytest.raises(RuntimeError, match='traffic or service scaling'):
        deploy.pin_ready_traffic(operator, config)


@pytest.mark.parametrize('exists', [True, False])
def test_main_applies_service_scaling_and_waits_before_pinning(monkeypatch, capsys, exists):
    config = deploy.service_config('openecon-sandbox', IMAGE, caller_subject=SUBJECT)
    service = ready_service(config)
    operator = Mock()
    operator.control_subject.return_value = SUBJECT
    events = []
    def call(method, url, body=None, **kwargs):
        events.append(('call', method, url, body))
        if url.endswith('/roles/openeconSandboxVerifier'):
            return {'includedPermissions': ['run.services.get', 'run.services.getIamPolicy']}
        if method == 'GET':
            return service if not kwargs.get('allow_missing') or exists else None
        return {'name': 'initial-operation'}
    operator.call.side_effect = call
    operator.wait.side_effect = lambda operation: events.append(('wait', operation))
    monkeypatch.setattr(deploy, 'Operator', lambda: operator)
    monkeypatch.setattr(deploy, 'pin_ready_traffic', lambda op, cfg: events.append(('pin', cfg)))
    monkeypatch.setattr('sys.argv', ['deploy', '--image', IMAGE])
    deploy.main()
    mutation = next(item for item in events if item[0] == 'call' and item[1] in {'PATCH', 'POST'})
    assert mutation[3]['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 4}
    if exists:
        assert mutation[1] == 'PATCH'
        assert mutation[2].endswith('?updateMask=template,traffic,launchStage,ingress,scaling')
    else:
        assert mutation[1] == 'POST' and 'name' not in mutation[3]
    assert events.index(next(event for event in events if event[0] == 'wait')) < events.index(
        next(event for event in events if event[0] == 'pin'))
    assert '"control_switched": false' in capsys.readouterr().out


@pytest.mark.parametrize('service', sorted(deploy.COMPUTE_SERVICES))
def test_each_allowlisted_broker_binds_its_own_service_audience_and_same_fixed_identity(service):
    config = deploy.service_config(service, IMAGE, caller_subject=SUBJECT)
    env = {entry['name']: entry['value'] for entry in config['template']['containers'][0]['env']}
    assert config['name'] == f'{deploy.PARENT}/services/{service}'
    assert env['OPENECON_BROKER_AUDIENCE'] == f'https://{service}-{deploy.NUMBER}.{deploy.REGION}.run.app'
    assert env['OPENECON_BROKER_CALLER_EMAIL'] == deploy.CONTROL
    assert env['OPENECON_BROKER_CALLER_SUB'] == SUBJECT
    assert config['template']['serviceAccount'] == deploy.COMPUTE
    assert config['template']['maxInstanceRequestConcurrency'] == 1
    assert config['template']['timeout'] == '360s'
    assert config['template']['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 4}
    assert config['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 4}


@pytest.mark.parametrize('service,source', [
    ('another-sandbox', None), ('openecon-sandbox-latex-extra', None),
    ('openecon-sandbox', 'fixed_probe()'), ('openecon-sandbox-latex', 'fixed_probe()'),
    ('openecon-sandbox-check', None), ('openecon-sandbox-check', ''),
])
def test_config_rejects_unapproved_services_or_crossing_probe_boundary(service, source):
    with pytest.raises(ValueError):
        deploy.service_config(service, IMAGE, caller_subject=SUBJECT, probe_source=source)


@pytest.mark.parametrize('flags', [
    ['--compute-service', 'another-sandbox'],
    ['--compute-service', 'openecon-sandbox-check'],
    ['--compute-service', 'openecon-sandbox-latex', '--service', 'openecon-sandbox'],
    ['--compute-service', 'openecon-sandbox-latex', '--probe-source', 'not-read.py'],
])
def test_invalid_compute_cli_selection_is_rejected_before_auth_or_any_management_call(monkeypatch, flags):
    operator = Mock(side_effect=AssertionError('Management must not be reached.'))
    monkeypatch.setattr(deploy, 'Operator', operator)
    monkeypatch.setattr('sys.argv', ['deploy', '--image', IMAGE, *flags])
    with pytest.raises(SystemExit):
        deploy.main()
    operator.assert_not_called()


def test_blue_green_compute_creation_never_mutates_or_reads_original_private_service(monkeypatch, capsys):
    green = 'openecon-sandbox-latex'
    config = deploy.service_config(green, IMAGE, caller_subject=SUBJECT)
    service = ready_service(config)
    operator = Mock()
    operator.control_subject.return_value = SUBJECT
    events = []

    def call(method, url, body=None, **kwargs):
        events.append((method, url, body))
        if url.endswith('/roles/openeconSandboxVerifier'):
            return {'includedPermissions': ['run.services.get', 'run.services.getIamPolicy']}
        if method == 'GET':
            return None if kwargs.get('allow_missing') else service
        return {'name': 'create-green-operation'}

    operator.call.side_effect = call
    monkeypatch.setattr(deploy, 'Operator', lambda: operator)
    monkeypatch.setattr(deploy, 'pin_ready_traffic', lambda op, cfg: None)
    monkeypatch.setattr('sys.argv', ['deploy', '--image', IMAGE, '--compute-service', green])
    deploy.main()
    create = next(event for event in events if event[0] == 'POST' and '?serviceId=' in event[1])
    assert create[1].endswith('?serviceId=openecon-sandbox-latex')
    assert create[2]['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 4}
    assert 'name' not in create[2]
    iam = next(event for event in events if event[1].endswith(':setIamPolicy'))
    assert iam[1].endswith('/services/openecon-sandbox-latex:setIamPolicy')
    assert iam[2]['policy'] == {'version': 3, 'bindings': [
        {'role': 'roles/run.invoker', 'members': ['serviceAccount:' + deploy.CONTROL]},
        {'role': deploy.VERIFIER, 'members': ['serviceAccount:' + deploy.CONTROL]},
    ]}
    assert all('/services/openecon-sandbox:' not in url and not url.endswith('/services/openecon-sandbox')
               for _, url, _ in events)
    assert json.loads(capsys.readouterr().out)['service'] == green


@pytest.mark.parametrize('changed', [None, 'image', 'audience'])
def test_green_configuration_passes_existing_runtime_verifier_without_relaxing_digest_or_audience(changed):
    from openecon.team_runner import JobRunnerError
    from openecon.team_sandbox_runner import GoogleSandboxRunner
    green = 'openecon-sandbox-latex'
    config = deploy.service_config(green, IMAGE, caller_subject=SUBJECT)
    service = ready_service(config)
    origin = f'https://{green}-{deploy.NUMBER}.{deploy.REGION}.run.app'
    service['uri'] = origin
    service['trafficStatuses'] = [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION',
                                  'revision': service['latestReadyRevision'], 'percent': 100}]
    if changed == 'image':
        service['template']['containers'][0]['image'] = IMAGE[:-64] + 'b' * 64
    elif changed == 'audience':
        environment = service['template']['containers'][0]['env']
        next(item for item in environment if item['name'] == 'OPENECON_BROKER_AUDIENCE')['value'] = (
            'https://openecon-sandbox-291739190496.us-central1.run.app')
    runner = GoogleSandboxRunner(deploy.PROJECT, deploy.REGION, green, origin=origin,
        bucket=deploy.PROJECT + '-projects', service_account=deploy.COMPUTE,
        image=IMAGE, storage=None, store=None, caller_subject=SUBJECT)
    runner._services = Mock()
    runner._services.get_service.return_value = service
    runner._services.get_iam_policy.return_value = {'bindings': [
        {'role': 'roles/run.invoker', 'members': ['serviceAccount:' + deploy.CONTROL]},
        {'role': deploy.VERIFIER, 'members': ['serviceAccount:' + deploy.CONTROL]},
    ]}
    if changed:
        with pytest.raises(JobRunnerError, match='isolation requirements'):
            runner._verify_worker()
    else:
        runner._verify_worker()
    runner._services.get_service.assert_called_once_with(
        request={'name': f'{deploy.PARENT}/services/{green}'}, retry=None, timeout=10)
