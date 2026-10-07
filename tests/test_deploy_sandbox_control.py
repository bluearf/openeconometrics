"""Control publication preserves private config and scoped rollback settings."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock, call

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'sandbox_control_deploy', Path(__file__).parents[1] / 'scripts/deploy_sandbox_control.py')
deploy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy)
IMAGE = 'us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:' + 'a' * 64
COMPUTE_IMAGE = IMAGE[:-64] + 'b' * 64
SUBJECT = '108062605432490492564'


def test_worker_management_audit_reuses_operator_for_exact_service_and_policy():
    operator = Mock()
    name = f'{deploy.PARENT}/services/{deploy.COMPUTE_SERVICE}'
    services = deploy.ManagementServices(operator, name)
    services.get_service(request={'name': name})
    services.get_iam_policy(request={'name': name})
    assert operator.call.call_args_list == [
        call('GET', name),
        call('GET', name + ':getIamPolicy?options.requestedPolicyVersion=3'),
    ]


@pytest.mark.parametrize('method', ['get_service', 'get_iam_policy'])
def test_worker_management_audit_rejects_other_service_or_extra_request_fields(method):
    operator = Mock()
    name = f'{deploy.PARENT}/services/{deploy.COMPUTE_SERVICE}'
    services = deploy.ManagementServices(operator, name)
    for request in ({'name': name + '-other'}, {'name': name, 'extra': True}):
        with pytest.raises(ValueError, match='Only the fixed compute'):
            getattr(services, method)(request=request)
    operator.call.assert_not_called()


def production():
    env = {
        'OPENECON_MODE': 'teams', 'OPENECON_PROJECT_ID': deploy.PROJECT,
        'OPENECON_REGION': deploy.REGION, 'OPENECON_BUCKET': deploy.PROJECT + '-projects',
        'OPENECON_COMPUTE_JOB': 'openecon-compute', 'OPENECON_COMPUTE_EMAIL': deploy.COMPUTE,
        'OPENECON_PUBLIC_ORIGIN': deploy.origin(deploy.PRODUCTION),
        'OPENECON_FIREBASE_CONFIG': 'private-config-sentinel',
    }
    return {
        'name': f'{deploy.PARENT}/services/openecon', 'ingress': 'INGRESS_TRAFFIC_ALL',
        'template': {
            'revision': 'old-revision', 'serviceAccount': deploy.CONTROL,
            'maxInstanceRequestConcurrency': 16, 'timeout': '360s',
            'scaling': {'maxInstanceCount': 2}, 'annotations': {'reviewed': 'preserved'},
            'containers': [{
                'image': IMAGE, 'env': [{'name': k, 'value': v} for k, v in env.items()]
                    + [{'name': 'KEEP_SECRET_REFERENCE', 'valueSource': {'secretKeyRef': {'secret': 'source-secret', 'version': '1'}}}],
                'resources': {'limits': {'cpu': '2', 'memory': '4Gi'}, 'startupCpuBoost': True},
                'startupProbe': {'httpGet': {'path': '/healthz'}},
            }],
        },
    }


@pytest.mark.parametrize('backend', ['sandbox', 'jobs'])
def test_preview_preserves_private_settings_and_both_backend_handles(backend):
    source = production()
    before = deepcopy(source)
    candidate = deploy.candidate_config(
        source, service=deploy.PREVIEW, image=IMAGE,
        compute_origin=deploy.origin(deploy.COMPUTE_SERVICE), compute_image=COMPUTE_IMAGE,
        backend=backend, caller_subject=SUBJECT)
    assert source == before
    assert 'name' not in candidate
    template = candidate['template']
    assert 'revision' not in template
    assert template['serviceAccount'] == deploy.CONTROL
    assert template['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 2}
    assert template['annotations'] == before['template']['annotations']
    container = template['containers'][0]
    assert container['resources'] == before['template']['containers'][0]['resources']
    assert container['startupProbe'] == before['template']['containers'][0]['startupProbe']
    env = {item['name']: item for item in container['env']}
    assert env['OPENECON_FIREBASE_CONFIG']['value'] == 'private-config-sentinel'
    assert env['KEEP_SECRET_REFERENCE']['valueSource']['secretKeyRef']['secret'] == 'source-secret'
    assert env['OPENECON_PUBLIC_ORIGIN']['value'] == deploy.origin(deploy.PREVIEW)
    assert env['OPENECON_RUNNER']['value'] == backend
    assert env['OPENECON_COMPUTE_JOB']['value'] == 'openecon-compute'
    assert env['OPENECON_COMPUTE_IMAGE']['value'] == COMPUTE_IMAGE
    assert env['OPENECON_BROKER_CALLER_SUB']['value'] == SUBJECT


@pytest.mark.parametrize('change', [
    lambda s: s['template'].update(serviceAccount=deploy.COMPUTE),
    lambda s: s['template']['scaling'].update(minInstanceCount=1),
    lambda s: s['template']['scaling'].update(maxInstanceCount=20),
    lambda s: s['template'].update(maxInstanceRequestConcurrency=80),
    lambda s: s['template'].update(volumes=[{'name': 'shared'}]),
    lambda s: s.update(invokerIamDisabled=True),
    lambda s: s['template']['containers'][0]['resources']['limits'].update(memory='16Gi'),
    lambda s: s['template']['containers'][0]['env'].append({'name': 'OPENECON_MODE', 'value': 'legacy'}),
])
def test_unreviewed_control_drift_is_rejected(change):
    source = production()
    change(source)
    with pytest.raises(ValueError):
        deploy.candidate_config(source, service=deploy.PRODUCTION, image=IMAGE,
                                compute_origin=deploy.origin(deploy.COMPUTE_SERVICE),
                                compute_image=COMPUTE_IMAGE, backend='sandbox', caller_subject=SUBJECT)


def test_baseline_contains_only_safe_summary_and_preserves_first_rollback_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    safe = {'baseline_image': IMAGE, 'baseline_ready_revision': 'old-revision', 'applied': False}
    deploy.save_baseline(safe, True)
    deploy.save_baseline({'baseline_ready_revision': 'changed-revision'}, True)
    path = tmp_path / 'artifacts/verification/sandbox-control-production-baseline.json'
    assert json.loads(path.read_text()) == safe
    assert path.stat().st_mode & 0o777 == 0o600


def test_public_web_policy_rejects_additional_public_privilege():
    public = {'bindings': [{'role': 'roles/run.invoker', 'members': ['allUsers']}]}
    assert deploy.public_web_policy(public)
    public['bindings'].append({'role': 'roles/run.admin', 'members': ['allUsers']})
    assert not deploy.public_web_policy(public)


def test_control_helper_reads_subject_from_fixed_iam_identity():
    operator = object.__new__(deploy.Operator)
    operator.session = Mock()
    response = operator.session.get.return_value
    response.status_code = 200
    response.json.return_value = {'email': deploy.CONTROL, 'uniqueId': SUBJECT}
    assert operator.control_subject() == SUBJECT
    operator.session.get.assert_called_once_with(
        f'https://iam.googleapis.com/v1/projects/{deploy.PROJECT}/serviceAccounts/{deploy.CONTROL}',
        timeout=15, allow_redirects=False)


@pytest.mark.parametrize('status,account', [
    (403, {}), (200, {'email': 'other@example.com', 'uniqueId': SUBJECT}),
    (200, {'email': deploy.CONTROL, 'uniqueId': SUBJECT, 'disabled': True}),
    (200, {'email': deploy.CONTROL, 'uniqueId': None}),
])
def test_control_helper_fails_closed_for_unverified_identity(status, account):
    operator = object.__new__(deploy.Operator)
    operator.session = Mock()
    operator.session.get.return_value.status_code = status
    operator.session.get.return_value.json.return_value = account
    with pytest.raises(RuntimeError, match='identity'):
        operator.control_subject()


@pytest.mark.parametrize('subject', [None, '', 'operator@example.com', 123456789])
def test_control_candidate_requires_numeric_subject_even_for_jobs_rollback(subject):
    with pytest.raises(ValueError, match='numeric control'):
        deploy.candidate_config(production(), service=deploy.PREVIEW, image=IMAGE,
            compute_origin=deploy.origin(deploy.COMPUTE_SERVICE), compute_image=COMPUTE_IMAGE,
            backend='jobs', caller_subject=subject)


@pytest.mark.parametrize('service', sorted(deploy.COMPUTE_SERVICES))
@pytest.mark.parametrize('backend', ['sandbox', 'jobs'])
def test_control_candidate_explicitly_binds_selected_broker_service_origin_and_immutable_image(service, backend):
    source = production()
    before = deepcopy(source)
    config = deploy.candidate_config(source, service=deploy.PREVIEW, image=IMAGE,
        compute_origin=deploy.origin(service), compute_image=COMPUTE_IMAGE,
        compute_service=service, backend=backend, caller_subject=SUBJECT)
    assert source == before
    env = {entry['name']: entry.get('value') for entry in config['template']['containers'][0]['env']}
    assert env['OPENECON_COMPUTE_SERVICE'] == service
    assert env['OPENECON_COMPUTE_ORIGIN'] == deploy.origin(service)
    assert env['OPENECON_COMPUTE_IMAGE'] == COMPUTE_IMAGE
    assert env['OPENECON_BROKER_CALLER_SUB'] == SUBJECT
    assert env['OPENECON_COMPUTE_EMAIL'] == deploy.COMPUTE
    assert env['OPENECON_COMPUTE_JOB'] == 'openecon-compute'
    assert config['template']['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 2}


@pytest.mark.parametrize('service,origin', [
    ('openecon-sandbox-latex', deploy.origin('openecon-sandbox')),
    ('openecon-sandbox', deploy.origin('openecon-sandbox-latex')),
    ('openecon-sandbox-check', deploy.origin('openecon-sandbox-check')),
    ('other', deploy.origin('other')),
    ('openecon-sandbox-latex', 'https://openecon-sandbox-latex.example.com'),
    ('openecon-sandbox-latex', 'https://openecon-sandbox-latex-999999999999.us-central1.run.app'),
])
def test_control_candidate_rejects_unapproved_service_or_mismatched_exact_origin(service, origin):
    with pytest.raises(ValueError):
        deploy.candidate_config(production(), service=deploy.PREVIEW, image=IMAGE,
            compute_origin=origin, compute_image=COMPUTE_IMAGE, compute_service=service,
            backend='sandbox', caller_subject=SUBJECT)


@pytest.mark.parametrize('production_target', [False, True])
@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('backend', ['sandbox', 'jobs'])
def test_main_verifies_exact_selected_green_broker_then_only_changes_selected_control(
        monkeypatch, capsys, production_target, apply, backend):
    source = production()
    operator = Mock()
    operator.control_subject.return_value = SUBJECT
    events = []

    def call(method, path, body=None, **kwargs):
        events.append((method, path, body))
        if path.endswith(':getIamPolicy'):
            return {'bindings': [{'role': 'roles/run.invoker', 'members': ['allUsers']}]}
        if method == 'GET':
            return None if kwargs.get('allow_missing') and not production_target else source
        return {'name': 'control-operation'}

    operator.call.side_effect = call
    verifier = Mock()
    verifier_class = Mock(return_value=verifier)
    monkeypatch.setattr(deploy, 'Operator', lambda: operator)
    monkeypatch.setattr(deploy, 'save_baseline', Mock())
    monkeypatch.setattr('openecon.team_sandbox_runner.GoogleSandboxRunner', verifier_class)
    flags = ['deploy', '--image', IMAGE, '--compute-image', COMPUTE_IMAGE,
             '--compute-service', 'openecon-sandbox-latex', '--backend', backend]
    if production_target:
        flags.append('--production')
    if apply:
        flags.append('--apply')
    monkeypatch.setattr('sys.argv', flags)
    deploy.main()
    verifier_class.assert_called_once_with(
        deploy.PROJECT, deploy.REGION, 'openecon-sandbox-latex',
        origin=deploy.origin('openecon-sandbox-latex'), bucket=deploy.PROJECT + '-projects',
        service_account=deploy.COMPUTE, image=COMPUTE_IMAGE, storage=None, store=None,
        caller_subject=SUBJECT)
    assert verifier._verify_worker.call_count == (1 if backend == 'sandbox' else 0)
    changes = [event for event in events if event[0] != 'GET']
    if not apply:
        assert changes == []
        operator.wait.assert_not_called()
    else:
        selected = deploy.PRODUCTION if production_target else deploy.PREVIEW
        mutation = changes[0]
        if production_target:
            assert mutation[0] == 'PATCH'
            assert mutation[1].startswith(f'{deploy.PARENT}/services/{selected}?')
        else:
            assert mutation[0] == 'POST'
            assert mutation[1] == deploy.PARENT + '/services?serviceId=' + selected
        env = {entry['name']: entry.get('value') for entry in mutation[2]['template']['containers'][0]['env']}
        assert env['OPENECON_COMPUTE_SERVICE'] == 'openecon-sandbox-latex'
        assert env['OPENECON_COMPUTE_ORIGIN'] == deploy.origin('openecon-sandbox-latex')
        assert all('/services/openecon-sandbox' not in path for _, path, _ in changes)
    summary = json.loads(capsys.readouterr().out)
    assert summary['compute_service'] == 'openecon-sandbox-latex'
    assert summary['compute_origin'] == deploy.origin('openecon-sandbox-latex')
    assert 'private-config-sentinel' not in str(summary)
