"""Stage the sandbox control plane without exporting its private environment.

By default this previews the plan for the fixed disposable web service. Use
--apply to create/update it, and --production explicitly for the live service.
Rollback keeps the candidate image and scoped sandbox settings, changing only
which backend starts new runs; accepted sandbox/Job handles remain routable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import time

import requests

PROJECT = 'openecon-workbench'
REGION = 'us-central1'
NUMBER = '291739190496'
PARENT = f'projects/{PROJECT}/locations/{REGION}'
CONTROL = f'openecon-control@{PROJECT}.iam.gserviceaccount.com'
COMPUTE = f'openecon-compute@{PROJECT}.iam.gserviceaccount.com'
PRODUCTION = 'openecon'
PREVIEW = 'openecon-teams-preview'
COMPUTE_SERVICE = 'openecon-sandbox'
COMPUTE_SERVICES = frozenset({COMPUTE_SERVICE, 'openecon-sandbox-latex'})
IMAGE_PATTERN = re.compile(
    r'us-central1-docker\.pkg\.dev/openecon-workbench/openecon/openecon@sha256:[0-9a-f]{64}')


class Operator:
    def __init__(self):
        try:
            token = subprocess.check_output(
                ['gcloud', 'auth', 'print-access-token', '--project=' + PROJECT],
                text=True, stderr=subprocess.PIPE).strip()
        except subprocess.CalledProcessError:
            raise RuntimeError('Google Cloud management authentication is unavailable.') from None
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({'Authorization': 'Bearer ' + token})

    def call(self, method, path, body=None, *, allow_missing=False):
        response = self.session.request(
            method, 'https://run.googleapis.com/v2/' + path, json=body,
            timeout=45, allow_redirects=False)
        if allow_missing and response.status_code == 404:
            return None
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f'Cloud management request failed: HTTP {response.status_code}')
        return response.json()

    def wait(self, operation):
        name = operation.get('name', '')
        if not name.startswith(PARENT + '/operations/'):
            raise RuntimeError('Unexpected Cloud Run operation scope.')
        deadline = time.monotonic() + 600
        while not operation.get('done') and time.monotonic() < deadline:
            time.sleep(4)
            operation = self.call('GET', name)
        if not operation.get('done') or operation.get('error'):
            raise RuntimeError('Control deployment did not complete successfully.')

    def control_subject(self):
        response = self.session.get(
            f'https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts/{CONTROL}',
            timeout=15, allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError('The fixed control service account identity could not be read.')
        account = response.json()
        subject = account.get('uniqueId')
        if (account.get('email') != CONTROL or account.get('disabled', False)
                or not isinstance(subject, str) or not re.fullmatch(r'[1-9][0-9]{5,31}', subject)):
            raise RuntimeError('The fixed control service account identity could not be verified.')
        return subject


class ManagementServices:
    """Use the same renewed management login for the fail-closed worker audit.

    Deployment must not silently use a second, stale local ADC identity. The
    production verifier still checks every template and IAM boundary unchanged.
    """
    def __init__(self, operator, name):
        self.operator, self.name = operator, name

    def get_service(self, *, request, retry=None, timeout=10):
        if request != {'name': self.name}:
            raise ValueError('Only the fixed compute service can be inspected.')
        return self.operator.call('GET', self.name)

    def get_iam_policy(self, *, request, retry=None, timeout=10):
        if request != {'name': self.name}:
            raise ValueError('Only the fixed compute policy can be inspected.')
        return self.operator.call('GET', self.name + ':getIamPolicy?options.requestedPolicyVersion=3')


def origin(service):
    return f'https://{service}-{NUMBER}.{REGION}.run.app'


def public_web_policy(policy):
    bindings = policy.get('bindings', [])
    invokers = [binding for binding in bindings if binding.get('role') == 'roles/run.invoker']
    return (len(invokers) == 1 and not invokers[0].get('condition')
            and invokers[0].get('members') == ['allUsers']
            and not any(member in {'allUsers', 'allAuthenticatedUsers'}
                        for binding in bindings if binding.get('role') != 'roles/run.invoker'
                        for member in binding.get('members', [])))


def candidate_config(source, *, service, image, compute_origin, compute_image, backend,
                     caller_subject, compute_service=COMPUTE_SERVICE):
    if service not in {PRODUCTION, PREVIEW} or backend not in {'sandbox', 'jobs'}:
        raise ValueError('Only fixed control services and supported backends are allowed.')
    if not IMAGE_PATTERN.fullmatch(image) or not IMAGE_PATTERN.fullmatch(compute_image):
        raise ValueError('Both images must use immutable reviewed OpenEcon digests.')
    if compute_service not in COMPUTE_SERVICES:
        raise ValueError('Only the two fixed private compute services are allowed.')
    if compute_origin != origin(compute_service):
        raise ValueError('The fixed private compute origin is required.')
    if not isinstance(caller_subject, str) or not re.fullmatch(r'[1-9][0-9]{5,31}', caller_subject):
        raise ValueError('The verified numeric control service identity is required.')
    template = deepcopy(source.get('template', {}))
    containers = template.get('containers', [])
    scaling = source.get('scaling', {})
    revision_scaling = template.get('scaling', {})
    safe = (
        source.get('name') == f'{PARENT}/services/{PRODUCTION}'
        and not source.get('invokerIamDisabled', False)
        and source.get('ingress') == 'INGRESS_TRAFFIC_ALL'
        and template.get('serviceAccount') == CONTROL
        and template.get('maxInstanceRequestConcurrency') == 16
        and template.get('timeout') == '360s'
        and not scaling.get('minInstanceCount', 0)
        and not revision_scaling.get('minInstanceCount', 0)
        and all(value == 2 for value in (
            scaling.get('maxInstanceCount', 2), revision_scaling.get('maxInstanceCount', 2)))
        and (scaling.get('maxInstanceCount') == 2 or revision_scaling.get('maxInstanceCount') == 2)
        and not template.get('volumes') and not template.get('vpcAccess')
        and not template.get('serviceMesh') and len(containers) == 1
    )
    if not safe:
        raise ValueError('Production control configuration differs from the reviewed baseline.')
    container = containers[0]
    resources = container.get('resources', {})
    if (resources.get('limits') != {'cpu': '2', 'memory': '4Gi'}
            or resources.get('cpuIdle', True) is not True
            or container.get('sandboxLauncher') or container.get('volumeMounts')):
        raise ValueError('Control resource and isolation settings must remain unchanged.')
    environment = container.get('env', [])
    values = {item.get('name'): item.get('value') for item in environment}
    expected = {
        'OPENECON_MODE': 'teams', 'OPENECON_PROJECT_ID': PROJECT,
        'OPENECON_REGION': REGION, 'OPENECON_BUCKET': PROJECT + '-projects',
        'OPENECON_COMPUTE_JOB': 'openecon-compute', 'OPENECON_COMPUTE_EMAIL': COMPUTE,
        'OPENECON_PUBLIC_ORIGIN': origin(PRODUCTION),
    }
    if (len(values) != len(environment)
            or any(values.get(key) != value for key, value in expected.items())):
        raise ValueError('Control identity, storage, or legacy Job configuration changed.')
    # Preserve every other environment setting verbatim, including secret
    # references. The source service, not a local artifact, owns these values.
    updates = {
        'OPENECON_PUBLIC_ORIGIN': origin(service), 'OPENECON_RUNNER': backend,
        'OPENECON_COMPUTE_SERVICE': compute_service,
        'OPENECON_COMPUTE_ORIGIN': compute_origin, 'OPENECON_COMPUTE_IMAGE': compute_image,
        'OPENECON_BROKER_CALLER_SUB': caller_subject,
    }
    container['env'] = [item for item in environment if item['name'] not in updates]
    container['env'].extend({'name': name, 'value': value} for name, value in updates.items())
    container['image'] = image
    template.pop('revision', None)
    template.setdefault('scaling', {})['minInstanceCount'] = 0
    template['scaling']['maxInstanceCount'] = 2
    result = {
        'template': template, 'ingress': source['ingress'],
        'launchStage': source.get('launchStage', 'GA'),
        'traffic': [{'type': 'TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST', 'percent': 100}],
    }
    if scaling:
        result['scaling'] = deepcopy(scaling)
    return result


def save_baseline(summary, production):
    path = Path('artifacts/verification') / (
        'sandbox-control-production-baseline.json' if production else 'sandbox-control-preview-baseline.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return  # Preserve the first rollback reference; never export full env.
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(summary, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Immutable candidate web/control digest')
    parser.add_argument('--compute-image', required=True, help='Immutable reviewed private compute digest')
    parser.add_argument('--compute-service', choices=sorted(COMPUTE_SERVICES), default=COMPUTE_SERVICE,
                        help='Select the fixed private broker; separate services preserve old runs')
    parser.add_argument('--backend', choices=['sandbox', 'jobs'], default='sandbox')
    parser.add_argument('--production', action='store_true', help='Explicitly select the live web service')
    parser.add_argument('--apply', action='store_true', help='Apply the reviewed plan; default is read-only')
    args = parser.parse_args()
    if not IMAGE_PATTERN.fullmatch(args.image) or not IMAGE_PATTERN.fullmatch(args.compute_image):
        raise SystemExit('Both candidate images must be immutable reviewed OpenEcon digests.')
    operator = Operator()
    source_name = f'{PARENT}/services/{PRODUCTION}'
    source = operator.call('GET', source_name)
    source_policy = operator.call('GET', source_name + ':getIamPolicy')
    if not public_web_policy(source_policy):
        raise SystemExit('Public web invocation differs from the reviewed baseline.')
    service = PRODUCTION if args.production else PREVIEW
    compute_origin = origin(args.compute_service)
    caller_subject = operator.control_subject()
    # Use exactly the same fail-closed API/template/IAM verifier as production.
    # These calls read configuration; they never launch Python or signed inputs.
    from openecon.team_sandbox_runner import GoogleSandboxRunner
    verifier = GoogleSandboxRunner(
        PROJECT, REGION, args.compute_service, origin=compute_origin,
        bucket=PROJECT + '-projects', service_account=COMPUTE,
        image=args.compute_image, storage=None, store=None, caller_subject=caller_subject)
    verifier._services = ManagementServices(operator, verifier.name)
    if args.backend == 'sandbox':
        verifier._verify_worker()
    # A jobs-start rollback must remain possible when the sandbox service is
    # unhealthy. Its accepted handles still read scoped parent markers and can
    # receive cancellation markers; no new sandbox launch is permitted.
    config = candidate_config(source, service=service, image=args.image,
                              compute_origin=compute_origin, compute_image=args.compute_image,
                              backend=args.backend, caller_subject=caller_subject,
                              compute_service=args.compute_service)
    summary = {
        'project': PROJECT, 'service': service, 'origin': origin(service),
        'backend': args.backend, 'candidate_image': args.image,
        'compute_service': args.compute_service, 'compute_origin': compute_origin,
        'compute_image': args.compute_image, 'minimum_instances': 0,
        'baseline_ready_revision': source.get('latestReadyRevision'),
        'baseline_image': source['template']['containers'][0]['image'],
        'applied': False,
    }
    if not args.apply:
        print(json.dumps(summary))
        return
    save_baseline(summary, args.production)
    name = f'{PARENT}/services/{service}'
    existing = operator.call('GET', name, allow_missing=True)
    if existing:
        config['name'] = name
        operation = operator.call('PATCH', name + '?updateMask=template,traffic,ingress,launchStage,scaling', config)
    else:
        # Cloud Run service.name is omitted on create; serviceId owns its name.
        operation = operator.call('POST', PARENT + '/services?serviceId=' + service, config)
    operator.wait(operation)
    if not args.production:
        policy = {'version': 3, 'bindings': [
            {'role': 'roles/run.invoker', 'members': ['allUsers']},
        ]}
        operator.call('POST', name + ':setIamPolicy', {'policy': policy})
    published = operator.call('GET', name)
    summary.update(applied=True, ready_revision=published.get('latestReadyRevision'))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
