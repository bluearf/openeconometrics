"""Deploy the team sync control service from one reviewed, digest-pinned image.

The service never executes user code or estimators, so there is no compute
service, job or backend to select. By default this prints a read-only plan for
the fixed disposable preview service. Use --apply to create/update the preview,
and --production --apply explicitly for the live service. The live service's
private environment is read in memory, preserved and never exported; retired
cloud-execution settings are removed from the candidate.
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
PRODUCTION = 'openecon'
PREVIEW = 'openecon-teams-preview'
IMAGE_PATTERN = re.compile(
    r'us-central1-docker\.pkg\.dev/openecon-workbench/openecon/openecon@sha256:[0-9a-f]{64}')
# Settings of the retired cloud execution backends. A candidate never keeps them.
RETIRED_SETTINGS = frozenset({
    'OPENECON_RUNNER', 'OPENECON_COMPUTE_JOB', 'OPENECON_COMPUTE_EMAIL',
    'OPENECON_COMPUTE_SERVICE', 'OPENECON_COMPUTE_ORIGIN', 'OPENECON_COMPUTE_IMAGE',
    'OPENECON_BROKER_CALLER_SUB',
})
# The image binds its own source commit at build time; a service variable must
# not be able to report a different revision.
IMAGE_OWNED_SETTINGS = frozenset({'OPENECON_SOURCE_COMMIT'})
REQUIRED_SYNC_SETTINGS = ('OPENECON_OWNER_EMAIL', 'OPENECON_SIGNER_EMAIL', 'OPENECON_FIREBASE_CONFIG')


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


def retired_settings(source):
    containers = source.get('template', {}).get('containers', [])
    names = {item.get('name') for container in containers for item in container.get('env', [])}
    return sorted(names & RETIRED_SETTINGS)


def candidate_config(source, *, service, image):
    if service not in {PRODUCTION, PREVIEW}:
        raise ValueError('Only the fixed control services are allowed.')
    if not isinstance(image, str) or not IMAGE_PATTERN.fullmatch(image):
        raise ValueError('The control image must use an immutable reviewed OpenEcon digest.')
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
        'OPENECON_PUBLIC_ORIGIN': origin(PRODUCTION),
    }
    if (len(values) != len(environment)
            or any(values.get(key) != value for key, value in expected.items())
            or any(key not in values for key in REQUIRED_SYNC_SETTINGS)):
        raise ValueError('Control identity, storage or sync configuration changed.')
    # Preserve every other environment setting verbatim, including secret
    # references. The source service, not a local artifact, owns these values.
    dropped = RETIRED_SETTINGS | IMAGE_OWNED_SETTINGS | {'OPENECON_PUBLIC_ORIGIN'}
    container['env'] = [item for item in environment if item['name'] not in dropped]
    container['env'].append({'name': 'OPENECON_PUBLIC_ORIGIN', 'value': origin(service)})
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
        'team-control-production-baseline.json' if production else 'team-control-preview-baseline.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return  # Preserve the first rollback reference; never export full env.
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(summary, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Immutable reviewed control image digest')
    parser.add_argument('--production', action='store_true', help='Explicitly select the live web service')
    parser.add_argument('--apply', action='store_true', help='Apply the reviewed plan; default is read-only')
    args = parser.parse_args()
    if not IMAGE_PATTERN.fullmatch(args.image):
        raise SystemExit('The candidate image must be an immutable reviewed OpenEcon digest.')
    operator = Operator()
    source_name = f'{PARENT}/services/{PRODUCTION}'
    source = operator.call('GET', source_name)
    source_policy = operator.call('GET', source_name + ':getIamPolicy')
    if not public_web_policy(source_policy):
        raise SystemExit('Public web invocation differs from the reviewed baseline.')
    service = PRODUCTION if args.production else PREVIEW
    config = candidate_config(source, service=service, image=args.image)
    summary = {
        'project': PROJECT, 'service': service, 'origin': origin(service),
        'candidate_image': args.image, 'minimum_instances': 0, 'cloud_execution': False,
        'retired_settings_removed': retired_settings(source),
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
