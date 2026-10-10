"""Control publication preserves private config and deploys only the sync service."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'team_control_deploy', Path(__file__).parents[1] / 'scripts/deploy_team_control.py')
deploy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy)
IMAGE = 'us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon@sha256:' + 'a' * 64
COMPUTE = f'openecon-compute@{deploy.PROJECT}.iam.gserviceaccount.com'
LEGACY_COMPUTE_ENV = {
    'OPENECON_RUNNER': 'sandbox', 'OPENECON_COMPUTE_JOB': 'openecon-compute',
    'OPENECON_COMPUTE_EMAIL': COMPUTE, 'OPENECON_COMPUTE_SERVICE': 'openecon-sandbox',
    'OPENECON_COMPUTE_ORIGIN': deploy.origin('openecon-sandbox'),
    'OPENECON_COMPUTE_IMAGE': IMAGE[:-64] + 'b' * 64,
    'OPENECON_BROKER_CALLER_SUB': '108062605432490492564',
}


def production(*, legacy=True):
    env = {
        'OPENECON_MODE': 'teams', 'OPENECON_PROJECT_ID': deploy.PROJECT,
        'OPENECON_REGION': deploy.REGION, 'OPENECON_BUCKET': deploy.PROJECT + '-projects',
        'OPENECON_PUBLIC_ORIGIN': deploy.origin(deploy.PRODUCTION),
        'OPENECON_OWNER_EMAIL': 'owner@example.com',
        'OPENECON_SIGNER_EMAIL': f'openecon-transfer@{deploy.PROJECT}.iam.gserviceaccount.com',
        'OPENECON_FIREBASE_CONFIG': 'private-config-sentinel',
        **(LEGACY_COMPUTE_ENV if legacy else {}),
    }
    return {
        'name': f'{deploy.PARENT}/services/openecon', 'ingress': 'INGRESS_TRAFFIC_ALL',
        'latestReadyRevision': 'openecon-old', 'template': {
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


def environment(config):
    return {item['name']: item for item in config['template']['containers'][0]['env']}


@pytest.mark.parametrize('legacy', [True, False])
@pytest.mark.parametrize('service', [deploy.PREVIEW, deploy.PRODUCTION])
def test_candidate_preserves_private_settings_and_removes_retired_compute(service, legacy):
    source = production(legacy=legacy)
    before = deepcopy(source)
    candidate = deploy.candidate_config(source, service=service, image=IMAGE)
    assert source == before
    assert 'name' not in candidate
    template = candidate['template']
    assert 'revision' not in template
    assert template['serviceAccount'] == deploy.CONTROL
    assert template['scaling'] == {'minInstanceCount': 0, 'maxInstanceCount': 2}
    assert template['annotations'] == before['template']['annotations']
    container = template['containers'][0]
    assert container['image'] == IMAGE
    assert container['resources'] == before['template']['containers'][0]['resources']
    assert container['startupProbe'] == before['template']['containers'][0]['startupProbe']
    assert 'sandboxLauncher' not in container
    env = environment(candidate)
    assert env['OPENECON_FIREBASE_CONFIG']['value'] == 'private-config-sentinel'
    assert env['OPENECON_OWNER_EMAIL']['value'] == 'owner@example.com'
    assert env['KEEP_SECRET_REFERENCE']['valueSource']['secretKeyRef']['secret'] == 'source-secret'
    assert env['OPENECON_PUBLIC_ORIGIN']['value'] == deploy.origin(service)
    assert env['OPENECON_MODE']['value'] == 'teams'
    assert not set(env) & deploy.RETIRED_SETTINGS
    assert not any('COMPUTE' in name or 'RUNNER' in name or 'BROKER' in name for name in env)
    assert len(container['env']) == len(env)
    assert deploy.retired_settings(source) == (sorted(LEGACY_COMPUTE_ENV) if legacy else [])


def test_service_variable_cannot_override_the_image_source_commit():
    source = production()
    source['template']['containers'][0]['env'].append(
        {'name': 'OPENECON_SOURCE_COMMIT', 'value': 'f' * 40})
    env = environment(deploy.candidate_config(source, service=deploy.PREVIEW, image=IMAGE))
    assert 'OPENECON_SOURCE_COMMIT' not in env


@pytest.mark.parametrize('change', [
    lambda s: s['template'].update(serviceAccount=COMPUTE),
    lambda s: s['template']['scaling'].update(minInstanceCount=1),
    lambda s: s['template']['scaling'].update(maxInstanceCount=20),
    lambda s: s['template'].update(maxInstanceRequestConcurrency=80),
    lambda s: s['template'].update(volumes=[{'name': 'shared'}]),
    lambda s: s.update(invokerIamDisabled=True),
    lambda s: s['template']['containers'][0]['resources']['limits'].update(memory='16Gi'),
    lambda s: s['template']['containers'][0].update(sandboxLauncher=True),
    lambda s: s['template']['containers'][0]['env'].append({'name': 'OPENECON_MODE', 'value': 'legacy'}),
    lambda s: s['template']['containers'][0]['env'].__setitem__(0, {'name': 'OPENECON_MODE', 'value': 'single-owner'}),
    lambda s: s['template']['containers'][0].update(env=[
        item for item in s['template']['containers'][0]['env'] if item['name'] != 'OPENECON_SIGNER_EMAIL']),
])
def test_unreviewed_control_drift_is_rejected(change):
    source = production()
    change(source)
    with pytest.raises(ValueError):
        deploy.candidate_config(source, service=deploy.PRODUCTION, image=IMAGE)


@pytest.mark.parametrize('image', [
    'us-central1-docker.pkg.dev/openecon-workbench/openecon/openecon:latest',
    'us-central1-docker.pkg.dev/other/openecon/openecon@sha256:' + 'a' * 64,
    IMAGE + '\n', IMAGE[:-1], None,
])
def test_candidate_requires_an_immutable_reviewed_digest(image):
    with pytest.raises(ValueError, match='digest'):
        deploy.candidate_config(production(), service=deploy.PREVIEW, image=image)


@pytest.mark.parametrize('service', ['openecon-sandbox', 'openecon-sandbox-latex', 'other'])
def test_only_the_fixed_control_services_can_be_targeted(service):
    with pytest.raises(ValueError, match='fixed control'):
        deploy.candidate_config(production(), service=service, image=IMAGE)


def test_baseline_contains_only_safe_summary_and_preserves_first_rollback_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    safe = {'baseline_image': IMAGE, 'baseline_ready_revision': 'old-revision', 'applied': False}
    deploy.save_baseline(safe, True)
    deploy.save_baseline({'baseline_ready_revision': 'changed-revision'}, True)
    path = tmp_path / 'artifacts/verification/team-control-production-baseline.json'
    assert json.loads(path.read_text()) == safe
    assert path.stat().st_mode & 0o777 == 0o600


def test_public_web_policy_rejects_additional_public_privilege():
    public = {'bindings': [{'role': 'roles/run.invoker', 'members': ['allUsers']}]}
    assert deploy.public_web_policy(public)
    public['bindings'].append({'role': 'roles/run.admin', 'members': ['allUsers']})
    assert not deploy.public_web_policy(public)


@pytest.mark.parametrize('retired', [
    ['--compute-image', IMAGE], ['--backend', 'jobs'], ['--compute-service', 'openecon-sandbox'],
])
def test_retired_compute_options_are_rejected_before_any_cloud_call(monkeypatch, retired):
    operator = Mock(side_effect=AssertionError('No management call may happen'))
    monkeypatch.setattr(deploy, 'Operator', operator)
    monkeypatch.setattr('sys.argv', ['deploy', '--image', IMAGE, *retired])
    with pytest.raises(SystemExit):
        deploy.main()
    operator.assert_not_called()


@pytest.mark.parametrize('production_target', [False, True])
@pytest.mark.parametrize('apply', [False, True])
def test_main_plans_and_changes_only_the_selected_control_service(
        monkeypatch, capsys, production_target, apply):
    source = production()
    operator = Mock()
    events = []

    def call(method, path, body=None, **kwargs):
        events.append((method, path, body))
        if path.endswith(':getIamPolicy'):
            return {'bindings': [{'role': 'roles/run.invoker', 'members': ['allUsers']}]}
        if method == 'GET':
            return None if kwargs.get('allow_missing') and not production_target else source
        return {'name': 'control-operation'}

    operator.call.side_effect = call
    baseline = Mock()
    monkeypatch.setattr(deploy, 'Operator', lambda: operator)
    monkeypatch.setattr(deploy, 'save_baseline', baseline)
    flags = ['deploy', '--image', IMAGE]
    if production_target:
        flags.append('--production')
    if apply:
        flags.append('--apply')
    monkeypatch.setattr('sys.argv', flags)
    deploy.main()
    selected = deploy.PRODUCTION if production_target else deploy.PREVIEW
    # Every read or write names one fixed control service; no compute resource is touched.
    for _, path, _ in events:
        assert path.startswith((f'{deploy.PARENT}/services/{deploy.PRODUCTION}',
                                f'{deploy.PARENT}/services/{selected}', f'{deploy.PARENT}/services?'))
        assert 'sandbox' not in path and '/jobs' not in path
    changes = [event for event in events if event[0] != 'GET']
    summary = json.loads(capsys.readouterr().out)
    if not apply:
        assert changes == []
        operator.wait.assert_not_called()
        baseline.assert_not_called()
        assert summary['applied'] is False
    else:
        mutation = changes[0]
        if production_target:
            assert mutation[0] == 'PATCH'
            assert mutation[1].startswith(f'{deploy.PARENT}/services/{selected}?')
            assert len(changes) == 1
        else:
            assert mutation[0] == 'POST'
            assert mutation[1] == deploy.PARENT + '/services?serviceId=' + selected
            assert changes[1][1] == f'{deploy.PARENT}/services/{selected}:setIamPolicy'
        env = {entry['name']: entry.get('value')
               for entry in mutation[2]['template']['containers'][0]['env']}
        assert not set(env) & deploy.RETIRED_SETTINGS
        assert env['OPENECON_PUBLIC_ORIGIN'] == deploy.origin(selected)
        assert mutation[2]['template']['containers'][0]['image'] == IMAGE
        operator.wait.assert_called_once_with({'name': 'control-operation'})
        assert summary['applied'] is True
    assert summary['service'] == selected and summary['cloud_execution'] is False
    assert summary['retired_settings_removed'] == sorted(LEGACY_COMPUTE_ENV)
    assert not any(key.startswith('compute') or key == 'backend' for key in summary)
    assert 'private-config-sentinel' not in str(summary)
