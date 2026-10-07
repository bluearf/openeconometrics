"""Deploy a private sandbox broker; keep production control switching explicit.

The operator's OAuth token exists only in memory. This script never enables
public invocation or changes minimum-instance billing. Preview API fields are
sent as raw v2 JSON because older Cloud SDK/protobufs silently discard them.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time

import requests

PROJECT = 'openecon-workbench'
REGION = 'us-central1'
NUMBER = '291739190496'
CONTROL = f'openecon-control@{PROJECT}.iam.gserviceaccount.com'
COMPUTE = f'openecon-compute@{PROJECT}.iam.gserviceaccount.com'
PARENT = f'projects/{PROJECT}/locations/{REGION}'
VERIFIER = f'projects/{PROJECT}/roles/openeconSandboxVerifier'
DEFAULT_COMPUTE_SERVICE = 'openecon-sandbox'
COMPUTE_SERVICES = frozenset({DEFAULT_COMPUTE_SERVICE, 'openecon-sandbox-latex'})
PROBE_SERVICE = 'openecon-sandbox-check'
IMAGE_PATTERN = re.compile(
    r'us-central1-docker\.pkg\.dev/openecon-workbench/openecon/openecon@sha256:[0-9a-f]{64}')


def service_config(service, image, *, caller_subject=None, probe_source=None):
    if not isinstance(image, str) or not IMAGE_PATTERN.fullmatch(image):
        raise ValueError('An immutable verified OpenEcon image digest is required.')
    if service not in COMPUTE_SERVICES | {PROBE_SERVICE}:
        raise ValueError('Only the fixed private brokers or disposable probe service are supported.')
    if (probe_source is not None) != (service == PROBE_SERVICE):
        raise ValueError('Probe code is confined to the disposable private verification service.')
    if probe_source is not None and (not isinstance(probe_source, str)
                                     or not probe_source or len(probe_source) > 32768):
        raise ValueError('Probe source exceeds its configuration budget or is empty.')
    command = ['/usr/bin/tini']
    args = ['--', '/opt/venv/bin/python', '-m', 'openecon.team_sandbox_broker']
    environment = [
        {'name': 'OPENECON_MODE', 'value': 'sandbox-broker'},
        {'name': 'OPENECON_SANDBOX_ROOTFS', 'value': '/opt/openecon-sandbox-rootfs'},
        {'name': 'OPENECON_SANDBOX_ENABLED', 'value': '1'},
    ]
    if probe_source is not None:
        args, environment = ['--', '/opt/venv/bin/python', '-c', probe_source], []
    else:
        if (service not in COMPUTE_SERVICES or not isinstance(caller_subject, str)
                or not re.fullmatch(r'[1-9][0-9]{5,31}', caller_subject)):
            raise ValueError('The fixed broker and verified numeric control identity are required.')
        environment.extend([
            {'name': 'OPENECON_BROKER_AUDIENCE',
             'value': f'https://{service}-{NUMBER}.{REGION}.run.app'},
            {'name': 'OPENECON_BROKER_CALLER_EMAIL', 'value': CONTROL},
            {'name': 'OPENECON_BROKER_CALLER_SUB', 'value': caller_subject},
        ])
    return {
        'name': f'{PARENT}/services/{service}', 'launchStage': 'BETA',
        'ingress': 'INGRESS_TRAFFIC_ALL',
        'scaling': {'minInstanceCount': 0, 'maxInstanceCount': 1 if probe_source is not None else 4},
        'template': {
            'serviceAccount': COMPUTE, 'executionEnvironment': 'EXECUTION_ENVIRONMENT_GEN2',
            'maxInstanceRequestConcurrency': 1, 'timeout': '360s',
            'scaling': {'minInstanceCount': 0, 'maxInstanceCount': 1 if probe_source is not None else 4},
            'containers': [{
                'image': image, 'command': command, 'args': args, 'env': environment,
                'sandboxLauncher': True, 'ports': [{'containerPort': 8080}],
                'resources': {'limits': {'cpu': '2', 'memory': '4Gi'},
                              'cpuIdle': True, 'startupCpuBoost': True},
                'startupProbe': {'httpGet': {'path': '/healthz'}, 'periodSeconds': 1,
                                 'timeoutSeconds': 1, 'failureThreshold': 120},
            }],
        },
        'traffic': [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST', 'percent': 100}],
    }


class Operator:
    def __init__(self):
        token = subprocess.check_output(['gcloud', 'auth', 'print-access-token'], text=True).strip()
        self.session = requests.Session()
        self.session.headers.update({'Authorization': 'Bearer ' + token})
        self.session.trust_env = False

    def call(self, method, url, body=None, *, allow_missing=False):
        response = self.session.request(method, url, json=body, timeout=45, allow_redirects=False)
        if allow_missing and response.status_code == 404:
            return None
        if not 200 <= response.status_code < 300:
            # No provider response, request body or Authorization value is
            # emitted; CLI failures carry only endpoint class and HTTP status.
            raise RuntimeError(f'Cloud management request failed: HTTP {response.status_code}')
        return response.json()

    def wait(self, operation):
        name = operation['name']
        if not name.startswith(PARENT + '/operations/'):
            raise RuntimeError('Unexpected cloud operation scope.')
        deadline = time.monotonic() + 600
        while not operation.get('done') and time.monotonic() < deadline:
            time.sleep(4)
            operation = self.call('GET', 'https://run.googleapis.com/v2/' + name)
        if not operation.get('done') or operation.get('error'):
            raise RuntimeError('Sandbox deployment did not complete successfully.')
        return operation.get('response', {})

    def control_subject(self):
        account = self.call('GET', f'https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts/{CONTROL}')
        subject = account.get('uniqueId')
        if (account.get('email') != CONTROL or account.get('disabled', False)
                or not isinstance(subject, str) or not re.fullmatch(r'[1-9][0-9]{5,31}', subject)):
            raise RuntimeError('The fixed control service account identity could not be verified.')
        return subject


def ready_revision(service, config):
    """Bind traffic to the completed deployment, never another concurrent image."""
    ready = service.get('latestReadyRevision')
    if (service.get('name') != config['name'] or service.get('reconciling', False)
            or service.get('generation') is None
            or service.get('observedGeneration') != service.get('generation')
            or not isinstance(ready, str)
            or not re.fullmatch(re.escape(config['name']) + r'/revisions/[a-z0-9-]+', ready)
            or ready != service.get('latestCreatedRevision')
            or service.get('terminalCondition', {}).get('state') != 'CONDITION_SUCCEEDED'
            or service.get('template', {}).get('containers', [{}])[0].get('image')
            != config['template']['containers'][0]['image']):
        raise RuntimeError('The reviewed sandbox revision is not ready for traffic pinning.')
    return ready


def pin_ready_traffic(operator, config):
    url = 'https://run.googleapis.com/v2/' + config['name']
    service = operator.call('GET', url)
    ready = ready_revision(service, config)
    etag = service.get('etag')
    if not isinstance(etag, str) or not etag:
        raise RuntimeError('A fresh service version is required for traffic pinning.')
    # TrafficTarget input takes the short revision ID. The ready revision above
    # is first verified as a full resource in this exact service's scope.
    traffic = [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION',
                'revision': ready.rsplit('/', 1)[1], 'percent': 100}]
    operation = operator.call('PATCH', url + '?updateMask=traffic', {
        'name': config['name'], 'etag': etag, 'traffic': traffic,
    })
    operator.wait(operation)
    published = operator.call('GET', url)
    status = published.get('trafficStatuses', [])
    if (ready_revision(published, config) != ready or len(status) != 1
            or status[0].get('type') != 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION'
            or status[0].get('revision') not in {ready, ready.rsplit('/', 1)[1]}
            or status[0].get('percent') != 100
            or published.get('scaling', {}).get('minInstanceCount', 0) != 0
            or published.get('scaling', {}).get('maxInstanceCount') != config['scaling']['maxInstanceCount']):
        raise RuntimeError('Sandbox traffic or service scaling did not match the reviewed deployment.')
    return published


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--compute-service', choices=sorted(COMPUTE_SERVICES),
                          help='Select the fixed private broker for blue/green deployment')
    selector.add_argument('--service', choices=sorted(COMPUTE_SERVICES | {PROBE_SERVICE}),
                          help='Legacy service selector; required for the disposable probe')
    parser.add_argument('--probe-source')
    args = parser.parse_args()
    service = args.compute_service or args.service or DEFAULT_COMPUTE_SERVICE
    if not IMAGE_PATTERN.fullmatch(args.image):
        raise SystemExit('An immutable verified OpenEcon image digest is required.')
    if bool(args.probe_source) != (service == PROBE_SERVICE):
        raise SystemExit('Probe code is confined to the disposable private verification service.')
    if args.probe_source:
        from pathlib import Path
        source = Path(args.probe_source).read_text()
        if len(source) > 32768:
            raise SystemExit('Probe source exceeds its configuration budget.')
    else:
        source = None
    operator = Operator()
    subject = operator.control_subject() if source is None else None
    config = service_config(service, args.image, caller_subject=subject, probe_source=source)
    base = 'https://run.googleapis.com/v2/'
    existing = operator.call('GET', base + config['name'], allow_missing=True)
    if existing:
        operation = operator.call('PATCH', base + config['name']
                                  + '?updateMask=template,traffic,launchStage,ingress,scaling', config)
    else:
        # The create route takes serviceId separately; a populated resource
        # name is only accepted by the subsequent patch/read routes.
        create_body = {key: value for key, value in config.items() if key != 'name'}
        operation = operator.call('POST', base + PARENT + '/services?serviceId=' + service,
                                  create_body)
    operator.wait(operation)
    pin_ready_traffic(operator, config)
    if source is None:
        role_url = 'https://iam.googleapis.com/v1/' + VERIFIER
        role = operator.call('GET', role_url, allow_missing=True)
        permissions = ['run.services.get', 'run.services.getIamPolicy']
        if role:
            if sorted(role.get('includedPermissions', [])) != sorted(permissions) or role.get('deleted'):
                raise RuntimeError('Existing verifier role differs from its narrow contract.')
        else:
            operator.call('POST', f'https://iam.googleapis.com/v1/projects/{PROJECT}/roles', {
                'roleId': 'openeconSandboxVerifier', 'role': {
                    'title': 'OpenEcon sandbox verification', 'stage': 'GA',
                    'includedPermissions': permissions,
                },
            })
        policy = {'version': 3, 'bindings': [
            {'role': 'roles/run.invoker', 'members': ['serviceAccount:' + CONTROL]},
            {'role': VERIFIER, 'members': ['serviceAccount:' + CONTROL]},
        ]}
        operator.call('POST', base + config['name'] + ':setIamPolicy', {'policy': policy})
    service = operator.call('GET', base + config['name'])
    print(json.dumps({'service': config['name'].rsplit('/', 1)[1], 'uri': service.get('uri'),
                      'urls': service.get('urls'), 'revision': service.get('latestReadyRevision'),
                      'sandboxLauncher': service['template']['containers'][0].get('sandboxLauncher'),
                      'minimum_instances': 0, 'control_switched': False}))


if __name__ == '__main__':
    main()
