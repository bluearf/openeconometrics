"""Trusted streaming bridge to a fixed, private Cloud Run sandbox supervisor.

Only the supervisor receives completion/cancellation capabilities. Submitted
Python runs in a fresh platform sandbox, never in this control-plane process.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from openecon.team_auth import TeamIdentity
from openecon.team_job import _NoRedirect, storage_url
from openecon.team_runner import JobRunnerError
from openecon.team_store import TeamError

MAX_COMPLETION_BYTES = 4096
MAX_EVENT_BYTES = 4096
MAX_EVENTS = 512
MAX_RPC_BYTES = 65536
_ID = r"[0-9a-f]{32}"
_TERMINAL = frozenset({'succeeded', 'failed', 'cancelled'})


def _json_object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('Non-finite JSON value')

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError('An object is required')
    return result


class _ServiceClient:
    """Read REST JSON: older run_v2 protobufs silently omit sandboxLauncher."""
    def get_service(self, *, request, retry=None, timeout=10):
        return self._read(request['name'], timeout)

    def get_iam_policy(self, *, request, retry=None, timeout=10):
        return self._read(request['name'] + ':getIamPolicy?options.requestedPolicyVersion=3', timeout)

    @staticmethod
    def _read(path, timeout):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform'])
        with AuthorizedSession(credentials) as session:
            with session.get('https://run.googleapis.com/v2/' + path,
                             timeout=timeout, allow_redirects=False, stream=True) as response:
                if response.status_code != 200:
                    raise ValueError('Service configuration unavailable')
                raw = response.raw.read(262145, decode_content=True)
                if len(raw) > 262144:
                    raise ValueError('Service configuration too large')
                return _json_object(raw)


def _identity_token(origin: str) -> str:
    # The deployed control service uses its own attached identity. Never accept
    # a caller token, key file, or a token returned by the compute service.
    from google.auth.compute_engine import IDTokenCredentials
    from google.auth.transport.requests import Request as AuthRequest

    request = AuthRequest()
    credentials = IDTokenCredentials(
        request=request, target_audience=origin, use_metadata_identity_endpoint=True)
    credentials.refresh(request)
    if not isinstance(credentials.token, str) or not credentials.token:
        raise ValueError('Service identity unavailable')
    return credentials.token


def _open_rpc(request: Request, *, timeout: float):
    # Fixed HTTPS origin and no redirects/proxy environment: the ID token and
    # per-run capabilities must never be forwarded to an alternate endpoint.
    return build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)


class GoogleSandboxRunner:
    """Keep the HTTP request open until the supervisor has destroyed its sandbox.

    The configured service account must separately be provisioned with no IAM
    roles; reading a service template cannot establish effective IAM grants.
    """
    def __init__(self, project: str, region: str, service: str, *, origin: str,
                 bucket: str, service_account: str, image: str, storage, store,
                 caller_subject: str | None = None):
        if not isinstance(project, str) or not re.fullmatch(
                r'[a-z][a-z0-9-]{4,61}[a-z0-9]', project):
            raise ValueError('A Google Cloud project ID is required.')
        if not isinstance(region, str) or not re.fullmatch(r'[a-z]+(?:-[a-z]+)+[0-9]+', region):
            raise ValueError('A Cloud Run region is required.')
        if not isinstance(service, str) or not re.fullmatch(
                r'[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?', service):
            raise ValueError('A fixed Cloud Run service is required.')
        if not isinstance(origin, str) or not re.fullmatch(
                r'https://' + re.escape(service) + r'-[1-9][0-9]{5,19}\.'
                + re.escape(region) + r'\.run\.app', origin):
            raise ValueError('A fixed numeric Cloud Run HTTPS origin is required.')
        if not isinstance(caller_subject, str) or not re.fullmatch(r'[1-9][0-9]{5,31}', caller_subject):
            raise ValueError('The control service account numeric identity must be explicitly configured.')
        if not isinstance(bucket, str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]', bucket):
            raise ValueError('An explicit execution bucket is required.')
        if not isinstance(service_account, str) or not re.fullmatch(
                r'[a-z][a-z0-9-]{4,28}[a-z0-9]@' + re.escape(project)
                + r'\.iam\.gserviceaccount\.com', service_account):
            raise ValueError('An explicit roleless service account is required.')
        if not isinstance(image, str) or not re.fullmatch(
                r'[a-z0-9][a-z0-9./_-]+@sha256:[0-9a-f]{64}', image):
            raise ValueError('The reviewed compute image must be pinned by digest.')
        self.project, self.region, self.service = project, region, service
        self.origin, self.bucket = origin, bucket
        self.service_account, self.image = service_account, image
        self.caller_subject = caller_subject
        self.caller_email = f'openecon-control@{project}.iam.gserviceaccount.com'
        self.storage, self.store = storage, store
        self.name = f'projects/{project}/locations/{region}/services/{service}'
        self._services = _ServiceClient()

    @staticmethod
    def _parts(name: str) -> tuple[str, str]:
        match = re.fullmatch(rf'sandbox:({_ID}):({_ID})', name) if isinstance(name, str) else None
        if match is None:
            raise ValueError('A scoped sandbox execution identifier is required.')
        return match.group(1), match.group(2)

    def _input_parts(self, input_url: str) -> tuple[str, str]:
        storage_url(input_url, generation=True)
        path = urlsplit(input_url).path
        match = re.fullmatch(rf'/{re.escape(self.bucket)}/staging/({_ID})/({_ID})/input\.json', path)
        if match is None:
            raise ValueError('The input must belong to the configured execution bucket.')
        return match.group(1), match.group(2)

    def _verify_worker(self) -> None:
        try:
            service = self._services.get_service(request={'name': self.name}, retry=None, timeout=10)
            template = service['template']
            containers = template['containers']
            scaling = service.get('scaling', {})
            revision_scaling = template.get('scaling', {})
            limits = [x for x in (scaling.get('maxInstanceCount'),
                                  revision_scaling.get('maxInstanceCount')) if x is not None]
            ready = service['latestReadyRevision']
            traffic = service['trafficStatuses']
            safe = (
                service['name'] == self.name and not service.get('reconciling', False)
                and service.get('observedGeneration') == service['generation']
                and ready == service['latestCreatedRevision']
                and ready.startswith(self.name + '/revisions/')
                and service.get('terminalCondition', {}).get('state') == 'CONDITION_SUCCEEDED'
                and len(traffic) == 1
                and traffic[0].get('type') == 'TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION'
                and traffic[0].get('revision') in {ready, ready.rsplit('/', 1)[1]}
                and traffic[0].get('percent') == 100
                and not service.get('invokerIamDisabled', False)
                and not service.get('sshEnabled', False)
                and not service.get('multiRegionSettings')
                and self.origin in {service.get('uri'), *service.get('urls', [])}
                and template['serviceAccount'] == self.service_account
                and template['executionEnvironment'] == 'EXECUTION_ENVIRONMENT_GEN2'
                and template['maxInstanceRequestConcurrency'] == 1
                and bool(limits) and all(type(limit) is int and 1 <= limit <= 4 for limit in limits)
                and not scaling.get('minInstanceCount', 0)
                and not revision_scaling.get('minInstanceCount', 0)
                and scaling.get('scalingMode', 'AUTOMATIC') in {'AUTOMATIC', 'SCALING_MODE_UNSPECIFIED'}
                and not template.get('volumes') and not template.get('vpcAccess')
                and not template.get('serviceMesh') and len(containers) == 1
            )
            if safe:
                container = containers[0]
                environment = container.get('env', [])
                expected_environment = {
                    'OPENECON_MODE': 'sandbox-broker',
                    'OPENECON_SANDBOX_ROOTFS': '/opt/openecon-sandbox-rootfs',
                    'OPENECON_SANDBOX_ENABLED': '1',
                    'OPENECON_BROKER_AUDIENCE': self.origin,
                    'OPENECON_BROKER_CALLER_EMAIL': self.caller_email,
                    'OPENECON_BROKER_CALLER_SUB': self.caller_subject,
                }
                safe = (
                    container['image'] == self.image and container.get('sandboxLauncher') is True
                    and not container.get('volumeMounts') and not container.get('sourceCode')
                    and not container.get('baseImageUri')
                    and not container.get('workingDir')
                    and not any(variable.get('valueSource') for variable in environment)
                    and len(environment) == len(expected_environment)
                    and {v['name']: v.get('value') for v in environment} == expected_environment
                    # The broker must not become Linux PID 1, which treats
                    # default-fatal signals differently. Tini preserves the
                    # kernel-enforced SIGALRM watchdog boundary.
                    and container.get('command') == ['/usr/bin/tini']
                    and container.get('args') == [
                        '--', '/opt/venv/bin/python', '-m', 'openecon.team_sandbox_broker']
                    and container.get('resources', {}).get('limits') == {'cpu': '2', 'memory': '4Gi'}
                    and container.get('resources', {}).get('cpuIdle', True) is True
                    and template.get('timeout') == '360s'
                )
            if safe:
                policy = self._services.get_iam_policy(
                    request={'name': self.name}, retry=None, timeout=10)
                bindings = policy.get('bindings', [])
                invokers = [binding for binding in bindings if binding.get('role') == 'roles/run.invoker']
                expected_invoker = 'serviceAccount:' + self.caller_email
                safe = (
                    bool(invokers)
                    and all(not binding.get('condition')
                            and binding.get('members') == [expected_invoker] for binding in invokers)
                    and not any(member in {'allUsers', 'allAuthenticatedUsers'}
                                for binding in bindings for member in binding.get('members', []))
                )
        except Exception:
            raise JobRunnerError('Compute configuration could not be verified; no run was launched.') from None
        if not safe:
            raise JobRunnerError('Compute configuration does not meet isolation requirements; no run was launched.')

    def _summary(self, project_id: str, run_id: str, status: str) -> dict:
        name = f'sandbox:{project_id}:{run_id}'
        return {'operation': name, 'execution': name, 'status': status,
                'message': 'Cloud execution failed.' if status == 'failed' else None}

    def _should_cancel(self, project_id: str, run_id: str) -> bool:
        run = self.store.run(project_id, run_id)
        if run is None:
            return True
        if run.get('cancel_requested') or run.get('state') in {'cancelled', 'failed'}:
            return True
        if datetime.fromisoformat(run['deadline']) <= datetime.now(timezone.utc):
            return True
        owner = TeamIdentity(run['uid'], run['email'], email_verified=True)
        try:
            self.store.project(project_id, owner, 'editor')
        except TeamError as exc:
            if exc.status_code not in (403, 404):
                raise
            self.store.update_run(project_id, run_id, cancel_requested=True)
            return True
        return False

    def start(self, input_url: str, timeout_seconds: float = 120) -> dict:
        project_id, run_id = self._input_parts(input_url)
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not .05 <= timeout_seconds <= 120):
            raise ValueError('Execution timeout must be between 0.05 and 120 seconds.')
        run = self.store.run(project_id, run_id)
        if (run is None or run.get('id') != run_id or run.get('project_id') != project_id
                or run.get('state') != 'starting' or run.get('operation')
                or run.get('timeout_seconds') != timeout_seconds):
            raise JobRunnerError('Only a new, reserved project execution can be dispatched.')
        name = f'sandbox:{project_id}:{run_id}'
        completion_key = f'staging/{project_id}/{run_id}/completion.json'
        self.store.claim_run_dispatch(project_id, run_id, operation=name, execution=name)
        attempted = False
        try:
            self._verify_worker()
            if self._should_cancel(project_id, run_id):
                self.cancel(name)
                self.storage.put(completion_key, json.dumps({
                    'execution_id': run_id, 'status': 'cancelled', 'cleanup_confirmed': True,
                }).encode(), 'application/json')
                return self._summary(project_id, run_id, 'cancelled')
            body = {
                'execution_id': run_id, 'input_url': input_url,
                'cancel_url': self.storage.signed_cancel_url(project_id, run_id),
                'completion_upload': self.storage.output_policy(completion_key, maximum=MAX_COMPLETION_BYTES),
                'timeout_seconds': timeout_seconds,
            }
            encoded = json.dumps(body, allow_nan=False, separators=(',', ':')).encode()
            if len(encoded) > MAX_RPC_BYTES:
                raise ValueError('Compute request exceeds its bounded capability envelope')
            # Cloud Run validates X-Serverless-Authorization and strips that
            # header's signature. Preserve the same complete Google-signed JWT
            # separately for the broker's independent application verification.
            authorization = 'Bearer ' + _identity_token(self.origin)
            request = Request(self.origin + '/execute', data=encoded, method='POST', headers={
                'Content-Type': 'application/json', 'Accept': 'application/x-ndjson',
                'X-Serverless-Authorization': authorization, 'Authorization': authorization,
            })
            deadline = time.monotonic() + math.ceil(timeout_seconds) + 180
            attempted = True
            with _open_rpc(request, timeout=90) as response:
                if response.status != 200:
                    raise ValueError('Compute request was not accepted')
                for _ in range(MAX_EVENTS):
                    if time.monotonic() >= deadline:
                        raise ValueError('Compute dispatch deadline exceeded')
                    line = response.readline(MAX_EVENT_BYTES + 1)
                    if not line:
                        break
                    if len(line) > MAX_EVENT_BYTES or not line.endswith(b'\n'):
                        raise ValueError('Invalid compute heartbeat')
                    event = _json_object(line)
                    if (event.get('execution_id') != run_id
                            or event.get('status') not in {'running', *_TERMINAL}):
                        raise ValueError('Invalid compute heartbeat')
                    if self._should_cancel(project_id, run_id):
                        self.cancel(name)
                else:
                    raise ValueError('Compute emitted too many events')
            # A worker-controlled result or HTTP frame cannot release a lease.
            # Only the separate parent-written cleanup marker establishes exit.
            result = self.status(name)
            if result['status'] not in _TERMINAL:
                raise ValueError('Compute stream ended without confirmed cleanup')
            return result
        except Exception:
            if attempted:
                try:
                    self.cancel(name)
                except Exception:
                    pass
            else:
                # No request was sent. The trusted control plane can attest
                # that there is no sandbox to clean up, even when signing or
                # configuration verification failed before dispatch.
                try:
                    self.storage.put(completion_key, json.dumps({
                        'execution_id': run_id, 'status': 'failed', 'cleanup_confirmed': True,
                    }).encode(), 'application/json')
                except Exception:
                    pass
            raise JobRunnerError('Compute launch could not be confirmed; do not automatically retry.') from None

    def status(self, operation_name: str) -> dict:
        project_id, run_id = self._parts(operation_name)
        try:
            raw = self.storage.get({'key': f'staging/{project_id}/{run_id}/completion.json'},
                                   maximum=MAX_COMPLETION_BYTES)
        except TeamError as exc:
            if exc.status_code == 404:
                return self._summary(project_id, run_id, 'running')
            raise JobRunnerError('Compute completion is temporarily unavailable.') from None
        except Exception:
            raise JobRunnerError('Compute completion is temporarily unavailable.') from None
        try:
            if len(raw) > MAX_COMPLETION_BYTES:
                raise ValueError('Oversized completion')
            result = _json_object(raw)
            if (set(result) != {'execution_id', 'status', 'cleanup_confirmed'}
                    or result['execution_id'] != run_id or result['status'] not in _TERMINAL
                    or result['cleanup_confirmed'] is not True):
                raise ValueError('Unconfirmed cleanup')
        except Exception:
            raise JobRunnerError('Compute completion could not be verified.') from None
        return self._summary(project_id, run_id, result['status'])

    def cancel(self, execution_name: str) -> dict:
        project_id, run_id = self._parts(execution_name)
        try:
            self.storage.request_cancel(project_id, run_id)
        except Exception:
            raise JobRunnerError('The execution cancellation could not be confirmed.') from None
        return self._summary(project_id, run_id, 'cancelling')
