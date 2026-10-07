"""Authorized private-broker QA using disposable staging objects only.

Run only after the platform runtime proof has passed and the private broker URL
has been approved for this task. No Firestore, account, membership, email or user
project changes occur. ADC, ID tokens and signed capabilities stay in memory.
Unknown launch outcomes never cause a second execution. Objects are removed only
after the separate parent-written cleanup marker has been verified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

PROJECT = 'openecon-workbench'
BUCKET = PROJECT + '-projects'
SIGNER = 'openecon-transfer@' + PROJECT + '.iam.gserviceaccount.com'
CONTROL = 'openecon-control@' + PROJECT + '.iam.gserviceaccount.com'
ORIGIN = 'https://openecon-sandbox-291739190496.us-central1.run.app'
MAX_FRAME = 4096
MAX_FRAMES = 512
MAX_RESULT = 24 * 1024**2
TERMINAL = {'succeeded', 'failed', 'cancelled'}
ANALYSIS = """df = oe.example()
display(df.head(3))
model = oe.ols(data=df, y='wage', x=['education', 'experience'])
display(model)
display(oe.plot.coefficients(model))
oe.plot.scatter(data=df, x='education', y='wage')
"""
ORPHAN = """import os, time
child = os.fork()
if child == 0:
    os.setsid()
    fd = os.open('/dev/null', os.O_RDWR)
    os.dup2(fd, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd > 2: os.close(fd)
    time.sleep(300)
    os._exit(0)
print('fixed-orphan-created')
"""
BEACON = """import json, os, time
from pathlib import Path
from urllib.request import Request, ProxyHandler, build_opener
from openecon.team_job import _NoRedirect
spec = json.loads(Path('beacon-capability.json').read_bytes())
child = os.fork()
if child == 0:
    os.setsid()
    fd = os.open('/dev/null', os.O_RDWR)
    for target in (0, 1, 2): os.dup2(fd, target)
    if fd > 2: os.close(fd)
    time.sleep(BEACON_DELAY)
    payload = json.dumps({'execution_id': spec['execution_id'], 'nonce': spec['nonce'],
                          'proof': 'fixed-beacon'}, sort_keys=True, separators=(',', ':')).encode()
    request = Request(spec['upload']['url'], data=payload, method='PUT',
                      headers=spec['upload']['headers'])
    try:
        with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=8) as response:
            response.read(4096)
    finally:
        os._exit(0)
print('fixed-beacon-child-created')
time.sleep(PARENT_DELAY)
"""
GUEST_BROKER_AUTH = """import base64, ipaddress, json, struct, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise URLError('Redirect refused')

checks = {'gateway_discovered': False, 'probe_completed': False,
          'missing_auth_denied': False, 'forged_auth_denied': False,
          'serverless_only_denied': False}
gateways = set()
try:
    for row in Path('/proc/net/route').read_text().splitlines()[1:]:
        fields = row.split()
        if len(fields) >= 8 and fields[1] == '00000000' and fields[7] == '00000000' and int(fields[3], 16) & 3 == 3:
            address = ipaddress.IPv4Address(struct.pack('<I', int(fields[2], 16)))
            if address.is_private and not (address.is_loopback or address.is_multicast or address.is_unspecified):
                gateways.add(str(address))
    if len(gateways) == 1:
        checks['gateway_discovered'] = True
        endpoint = 'http://' + next(iter(gateways)) + ':8080/execute'
        def encoded(value):
            return base64.urlsafe_b64encode(json.dumps(value, separators=(',', ':')).encode()).rstrip(b'=').decode()
        now = int(time.time())
        forged = '.'.join((encoded({'alg': 'RS256', 'kid': 'fixed-invalid-proof-key', 'typ': 'JWT'}),
            encoded({'iss': 'https://accounts.google.com',
                     'aud': 'https://openecon-sandbox-291739190496.us-central1.run.app',
                     'sub': '108062605432490492564',
                     'email': 'openecon-control@openecon-workbench.iam.gserviceaccount.com',
                     'email_verified': True, 'iat': now, 'exp': now + 300}),
            base64.urlsafe_b64encode(bytes(256)).rstrip(b'=').decode()))
        opener = build_opener(ProxyHandler({}), NoRedirect())
        completed = 0
        for name, extra in (('missing_auth_denied', {}),
                            ('forged_auth_denied', {'Authorization': 'Bearer ' + forged}),
                            ('serverless_only_denied', {'X-Serverless-Authorization': 'Bearer ' + forged})):
            request = Request(endpoint, data=b'{}', method='POST',
                              headers={'Content-Type': 'application/json', **extra})
            try:
                with opener.open(request, timeout=3):
                    pass
            except HTTPError as error:
                checks[name] = error.code == 401
                error.close()
            except (URLError, TimeoutError, OSError):
                pass
            completed += 1
        checks['probe_completed'] = completed == 3
except Exception:
    pass
print(json.dumps(checks, sort_keys=True, separators=(',', ':')))
"""


@dataclass(frozen=True)
class Case:
    name: str
    code: str
    timeout: float = 10
    record_status: str = 'ok'
    cancel: bool = False
    beacon: str = ''


CASES = (
    Case('initial_analysis', ANALYSIS),
    Case('guest_gateway_broker_auth', GUEST_BROKER_AUTH, timeout=15),
    Case('analysis_after_guest_auth_probe', ANALYSIS),
    Case('repeat_analysis_1', ANALYSIS),
    Case('repeat_analysis_2', ANALYSIS),
    Case('repeat_analysis_3', ANALYSIS),
    Case('cancellation', 'import time\ntime.sleep(300)\n', timeout=120, cancel=True),
    Case('python_timeout', 'while True: pass\n', timeout=.1, record_status='timeout'),
    Case('bounded_stdout', "print('x' * 1000000)\n"),
    Case('detached_fork', ORPHAN),
    Case('beacon_child_positive_control', BEACON.replace('BEACON_DELAY', '1').replace('PARENT_DELAY', '8'),
         timeout=15, beacon='positive'),
    Case('deleted_sandbox_orphan_beacon', BEACON.replace('BEACON_DELAY', '20').replace('PARENT_DELAY', '0'),
         beacon='deleted'),
    Case('analysis_after_adversarial_cases', ANALYSIS),
)


class ProofError(RuntimeError):
    """Use fixed messages only; do not include requests, outputs or capabilities."""


def control_identity_token() -> str:
    # This proof must use the same caller identity as production. An operator
    # token must never satisfy the broker's application-level authorization.
    # Direct ID-token issuance requires only getOpenIdToken. gcloud's generic
    # impersonation flow also requests an access token, which this QA identity
    # deliberately cannot mint. No IAM grant or operator-token fallback occurs.
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession, Request as AuthRequest
        import requests

        credentials, _ = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform'], quota_project_id=PROJECT)
        with requests.Session() as refresh_session:
            refresh_session.trust_env = False
            with AuthorizedSession(credentials, auth_request=AuthRequest(session=refresh_session)) as session:
                session.trust_env = False
                with session.post(
                    'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/'
                    + CONTROL + ':generateIdToken',
                    json={'audience': ORIGIN, 'includeEmail': True},
                    timeout=30, allow_redirects=False, stream=True,
                ) as response:
                    if response.status_code != 200:
                        raise ValueError('ID-token issuance unavailable')
                    raw = response.raw.read(32769, decode_content=True)
                    if len(raw) > 32768:
                        raise ValueError('ID-token response exceeds its bound')
                    payload = json_object(raw)
        token = payload.get('token') if set(payload) == {'token'} else None
    except Exception:
        raise ProofError('The fixed control identity could not be impersonated.') from None
    if not isinstance(token, str) or not 1 <= len(token) <= 8192 or not re.fullmatch(
            r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', token):
        raise ProofError('A complete signed control identity token is required.')
    return token


def json_object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProofError('Duplicate JSON field.')
            result[key] = value
        return result

    def invalid(_):
        raise ProofError('Nonfinite JSON value.')

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ProofError('A JSON object is required.')
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ProofError('Redirect refused.')


def validate_beacon_upload(upload: dict, project_id: str, run_id: str, payload: bytes) -> None:
    if (any(not re.fullmatch(r'[0-9a-f]{32}', value) for value in (project_id, run_id))
            or not isinstance(upload, dict) or set(upload) != {'url', 'headers'}
            or not 1 <= len(payload) <= 4096):
        raise ProofError('Fixed beacon capability is invalid.')
    parsed = urlsplit(upload['url'])
    query = parse_qs(parsed.query, keep_blank_values=True)
    expected_query = {'X-Goog-Algorithm', 'X-Goog-Credential', 'X-Goog-Date',
                      'X-Goog-Expires', 'X-Goog-SignedHeaders', 'X-Goog-Signature'}
    if (parsed.scheme != 'https' or parsed.netloc != 'storage.googleapis.com' or parsed.fragment
            or parsed.path != f'/{BUCKET}/staging/{project_id}/{run_id}/beacon.json'
            or set(query) != expected_query or any(len(values) != 1 for values in query.values())
            or query['X-Goog-Algorithm'] != ['GOOG4-RSA-SHA256']
            or query['X-Goog-SignedHeaders'] != ['content-length;content-type;host;x-goog-if-generation-match']
            or not re.fullmatch(re.escape(SIGNER) + r'/[0-9]{8}/auto/storage/goog4_request',
                                query['X-Goog-Credential'][0])
            or not re.fullmatch(r'[0-9a-f]{512}', query['X-Goog-Signature'][0])
            or upload['headers'] != {'Content-Type': 'application/json',
                                      'Content-Length': str(len(payload)),
                                      'x-goog-if-generation-match': '0'}):
        raise ProofError('Fixed beacon capability scope is invalid.')
    try:
        expires = int(query['X-Goog-Expires'][0])
        dated = datetime.strptime(query['X-Goog-Date'][0], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dated).total_seconds()
        if not 30 <= expires <= 300 or not -60 <= age < expires:
            raise ValueError
    except (ValueError, TypeError):
        raise ProofError('Fixed beacon capability expiry is invalid.') from None


def beacon_spec(storage, project_id: str, run_id: str) -> tuple[dict, bytes]:
    nonce = uuid4().hex
    payload = json.dumps({'execution_id': run_id, 'nonce': nonce, 'proof': 'fixed-beacon'},
                         sort_keys=True, separators=(',', ':')).encode()
    headers = {'Content-Type': 'application/json', 'Content-Length': str(len(payload)),
               'x-goog-if-generation-match': '0'}
    url = storage.bucket.blob(f'staging/{project_id}/{run_id}/beacon.json').generate_signed_url(
        version='v4', expiration=timedelta(minutes=3), method='PUT', content_type='application/json',
        headers={'Content-Length': headers['Content-Length'], 'x-goog-if-generation-match': '0'},
        credentials=storage.signer)
    upload = {'url': url, 'headers': headers}
    validate_beacon_upload(upload, project_id, run_id, payload)
    return {'execution_id': run_id, 'nonce': nonce, 'upload': upload}, payload


def verify_beacon(storage, project_id: str, run_id: str, expected: bytes, present: bool) -> bool:
    from openecon.team_store import TeamError
    try:
        raw = storage.get({'key': f'staging/{project_id}/{run_id}/beacon.json'}, maximum=4096)
    except TeamError as error:
        if error.status_code == 404:
            return not present
        raise ProofError('Fixed beacon read failed.') from None
    return present and raw == expected


def beacon_capability_positive_control(storage, project_id: str, run_id: str,
                                      spec: dict, payload: bytes) -> bool:
    upload = spec['upload']
    request = Request(upload['url'], data=payload, method='PUT', headers=upload['headers'])
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=15) as response:
        if response.status not in {200, 201, 204}:
            raise ProofError('Fixed beacon positive upload failed.')
        response.read(4096)
    if not verify_beacon(storage, project_id, run_id, payload, True):
        raise ProofError('Fixed beacon positive read failed.')
    blob = storage.bucket.blob(f'staging/{project_id}/{run_id}/beacon.json')
    blob.reload(timeout=15)
    blob.delete(if_generation_match=int(blob.generation), timeout=15, retry=None)
    return verify_beacon(storage, project_id, run_id, payload, False)


def completion(storage, project_id: str, run_id: str) -> dict | None:
    from openecon.team_store import TeamError
    try:
        raw = storage.get({'key': f'staging/{project_id}/{run_id}/completion.json'}, maximum=4096)
    except TeamError as error:
        if error.status_code == 404:
            return None
        raise ProofError('Completion read failed.') from None
    marker = json_object(raw)
    if (set(marker) != {'execution_id', 'status', 'cleanup_confirmed'}
            or marker['execution_id'] != run_id or marker['status'] not in TERMINAL
            or marker['cleanup_confirmed'] is not True):
        raise ProofError('Parent cleanup marker is invalid.')
    return marker


def remove_confirmed_objects(storage, project_id: str, run_id: str) -> bool:
    from google.api_core.exceptions import NotFound
    prefix = f'staging/{project_id}/{run_id}/'
    for suffix in ('input.json', 'output.json', 'cancel.json', 'completion.json',
                   'beacon-input.json', 'beacon.json'):
        blob = storage.bucket.blob(prefix + suffix)
        try:
            blob.reload(timeout=15)
            generation = int(blob.generation)
            blob.delete(if_generation_match=generation, timeout=15, retry=None)
        except NotFound:
            pass
    return True


def fork_stdout_classification(case: Case, stdout: str) -> tuple[bool, bool]:
    """Recognize only the fixed marker and Python's exact fork warning.

    The console captures stderr alongside stdout. Python may warn at os.fork()
    after native libraries have created threads. No other warning or output is
    accepted; actual child death is proved separately by the external beacon.
    """
    marker = 'fixed-orphan-created' if case.name == 'detached_fork' else 'fixed-beacon-child-created'
    line = case.code.splitlines().index('child = os.fork()') + 1
    exact_marker = re.escape(marker) + r'\n?'
    warning = (rf'<openecon:[0-9a-f]{{8}}(?:-[0-9a-f]{{4}}){{3}}-[0-9a-f]{{12}}>:{line}: '
               r'DeprecationWarning: This process \(pid=[1-9][0-9]*\) is multi-threaded, '
               r'use of fork\(\) may lead to deadlocks in the child\.\n')
    known_warning = re.fullmatch(warning + exact_marker, stdout) is not None
    return re.fullmatch(exact_marker, stdout) is not None or known_warning, known_warning


def fixed_output_checks(case: Case, record: dict, generated: list) -> dict:
    checks = {'record_status_matches': record['status'] == case.record_status,
              'no_generated_artifacts': generated == []}
    if case.code == ANALYSIS:
        outputs = record['outputs']
        checks.update(
            table_model_two_plots=[x['type'] for x in outputs] == ['table', 'model', 'plot', 'plot'],
            observations_480=len(outputs) > 1 and outputs[1]['data'].get('nobs') == 480,
            native_hc3=len(outputs) > 1 and outputs[1]['data'].get('spec', {}).get('covariance') == 'HC3',
        )
    elif case.name == 'bounded_stdout':
        checks.update(stdout_bounded=len(record['stdout'].encode()) <= 70 * 1024,
                      stdout_truncation_visible='truncated' in record['stdout'])
    elif case.name == 'detached_fork':
        checks['fork_code_completed'] = fork_stdout_classification(case, record['stdout'])[0]
    elif case.name == 'guest_gateway_broker_auth':
        expected = {'gateway_discovered', 'probe_completed', 'missing_auth_denied',
                    'forged_auth_denied', 'serverless_only_denied'}
        observed = json_object(record['stdout'].encode())
        checks['guest_auth_probe_exact_shape'] = set(observed) == expected
        checks.update({name: observed.get(name) is True for name in expected})
    elif case.beacon:
        checks['beacon_child_created'] = fork_stdout_classification(case, record['stdout'])[0]
    return checks


def run_case(case: Case, storage, token: str, evidence: dict, save=None) -> dict:
    from openecon.team_sandbox_broker import validate_body
    from openecon.team_server import validate_worker_result

    project_id, run_id = uuid4().hex, uuid4().hex
    scope = {'project_id': project_id, 'run_id': run_id}
    evidence['pending_staging_scopes'].append(scope)
    if save is not None:
        save()  # Journal only random scope IDs before the first cloud mutation.
    run = {'id': run_id, 'code': case.code, 'timeout_seconds': case.timeout, 'files': [],
           'generation': 1, 'email': 'sandbox-proof@example.invalid',
           'created_at': datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    marker = None
    attempted = False
    cancel_thread = None
    cancel_done = threading.Event()
    cancel_failed = threading.Event()
    frame_count = 0
    first_frame_at = None
    cancellation_at = None
    result = {'name': case.name, 'checks': {}, 'passed': False}
    try:
        if case.beacon:
            spec, beacon_payload = beacon_spec(storage, project_id, run_id)
            if not beacon_capability_positive_control(storage, project_id, run_id, spec, beacon_payload):
                raise ProofError('Fixed beacon capability positive control failed.')
            reference = storage.put(f'staging/{project_id}/{run_id}/beacon-input.json',
                                    json.dumps(spec, separators=(',', ':')).encode(), 'application/json')
            run['files'] = [{'name': 'beacon-capability.json', 'blob': reference}]
        input_url, output_ref, _ = storage.manifest(project_id, run)
        body = {'execution_id': run_id, 'input_url': input_url,
                'cancel_url': storage.signed_cancel_url(project_id, run_id),
                'completion_upload': storage.output_policy(
                    f'staging/{project_id}/{run_id}/completion.json', maximum=4096),
                'timeout_seconds': case.timeout}
        encoded = json.dumps(body, allow_nan=False, separators=(',', ':')).encode()
        validate_body(encoded)  # The actual Google SDK policy must fit the broker contract.
        prepared_at = time.monotonic()

        def cancel_after_heartbeat():
            nonlocal cancellation_at
            time.sleep(3)
            try:
                storage.request_cancel(project_id, run_id)
                cancellation_at = time.monotonic()
                cancel_done.set()
            except Exception:
                cancel_failed.set()

        request = Request(ORIGIN + '/execute', data=encoded, method='POST', headers={
            'Content-Type': 'application/json', 'Accept': 'application/x-ndjson',
            'X-Serverless-Authorization': 'Bearer ' + token,
            'Authorization': 'Bearer ' + token})
        opener = build_opener(ProxyHandler({}), NoRedirect())
        attempted = True
        terminal_frame = None
        hard_deadline = time.monotonic() + math.ceil(case.timeout) + 180
        with opener.open(request, timeout=60) as response:
            if response.status != 200:
                raise ProofError('Broker dispatch failed.')
            for _ in range(MAX_FRAMES):
                if time.monotonic() >= hard_deadline:
                    raise ProofError('Broker response exceeded the proof deadline.')
                raw = response.readline(MAX_FRAME + 1)
                if not raw:
                    break
                if len(raw) > MAX_FRAME or not raw.endswith(b'\n'):
                    raise ProofError('Broker frame exceeded its bound.')
                frame = json_object(raw)
                if (set(frame) != {'execution_id', 'status'} or frame['execution_id'] != run_id
                        or frame['status'] not in {'running', *TERMINAL}):
                    raise ProofError('Broker frame is invalid.')
                frame_count += 1
                if first_frame_at is None:
                    first_frame_at = time.monotonic()
                    if case.cancel:
                        cancel_thread = threading.Thread(target=cancel_after_heartbeat, daemon=True)
                        cancel_thread.start()
                if frame['status'] in TERMINAL:
                    terminal_frame = frame['status']
            else:
                raise ProofError('Broker frame count exceeded its bound.')
        if cancel_thread:
            cancel_thread.join(timeout=50)
            if cancel_thread.is_alive() or cancel_failed.is_set():
                raise ProofError('Cancellation could not be confirmed.')
        marker_deadline = time.monotonic() + 10
        while marker is None and time.monotonic() < marker_deadline:
            marker = completion(storage, project_id, run_id)
            if marker is None:
                time.sleep(.2)
        if marker is None:
            raise ProofError('Broker ended without a trusted cleanup marker.')
        confirmed_at = time.monotonic()
        result['checks'] = {'parent_cleanup_marker_confirmed': True,
                            'terminal_frame_matches_parent': terminal_frame == marker['status']}
        result['completion_status'] = marker['status']
        if case.cancel:
            result['checks'].update(cancel_requested_after_heartbeat=cancel_done.is_set(),
                                    cancellation_completed=marker['status'] == 'cancelled')
        else:
            result['checks']['parent_execution_succeeded'] = marker['status'] == 'succeeded'
            if marker['status'] != 'succeeded':
                raise ProofError('Fixed worker execution did not succeed.')
            # This is intentionally after the parent marker. Worker JSON alone
            # cannot prove termination or authorize result publication.
            raw = storage.get(output_ref, maximum=MAX_RESULT)
            payload = json_object(raw)
            record, generated = validate_worker_result(payload, run)
            result['checks'].update(fixed_output_checks(case, record, generated))
            if case.name == 'detached_fork' or case.beacon:
                result['known_fork_warning_observed'] = fork_stdout_classification(case, record['stdout'])[1]
            result['record_status'] = record['status']
            result['result_bytes'] = len(raw)
            result['stdout_bytes'] = len(record['stdout'].encode())
            result['record_duration_ms'] = record['duration_ms']
            if case.beacon:
                if case.beacon == 'deleted':
                    # Wait beyond the fixed child upload delay after confirmed
                    # deletion. The exact signed capability was exercised and
                    # reset above; GCS object reads are strongly consistent.
                    time.sleep(35)
                result['checks']['signed_beacon_capability_positive_control'] = True
                result['checks']['external_beacon_expected_presence'] = verify_beacon(
                    storage, project_id, run_id, beacon_payload, case.beacon == 'positive')
        ended = time.monotonic()
        result['timings'] = {
            'input_preparation_seconds': prepared_at - started,
            'dispatch_to_parent_marker_seconds': confirmed_at - prepared_at,
            'result_validation_seconds': ended - confirmed_at,
            'total_seconds': ended - started,
            'first_heartbeat_seconds': None if first_frame_at is None else first_frame_at - prepared_at,
            'cancellation_to_parent_marker_seconds': (None if cancellation_at is None
                                                       else confirmed_at - cancellation_at),
        }
        result['heartbeat_frames'] = frame_count
        result['passed'] = all(result['checks'].values())
    except Exception:
        result['error'] = 'Fixed private broker proof failed.'
        if attempted and marker is None:
            try:
                storage.request_cancel(project_id, run_id)
                marker = completion(storage, project_id, run_id)
            except Exception:
                marker = None
    finally:
        # A delayed cancellation writer must finish before scoped object cleanup.
        if cancel_thread and cancel_thread.is_alive():
            cancel_thread.join(timeout=50)
        if not attempted:
            safe_to_remove = True
        else:
            safe_to_remove = marker is not None and (cancel_thread is None or not cancel_thread.is_alive())
        if safe_to_remove:
            try:
                result['objects_removed'] = remove_confirmed_objects(storage, project_id, run_id)
                evidence['pending_staging_scopes'].remove(scope)
            except Exception:
                result['objects_removed'] = False
                result['passed'] = False
        else:
            result['objects_removed'] = False
            result['passed'] = False
    return result


def verify(core_proof: Path, report: Path) -> dict:
    proof = json_object(core_proof.read_bytes())
    checks = proof.get('checks', {})
    if (proof.get('passed') is not True or proof.get('kernel_alarm_after_disconnect_killed_parent') is not True
            or proof.get('kernel_watchdog_child_death_verified') is not True
            or checks.get('sandbox_urandom_available') is not True
            or checks.get('sandbox_null_available') is not True):
        raise ProofError('A passing actual platform proof is required before broker execution.')
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(Request(ORIGIN + '/execute', data=b'{}', method='POST',
                                 headers={'Content-Type': 'application/json'}), timeout=30):
            raise ProofError('The broker permits anonymous invocation.')
    except HTTPError as error:
        if error.code not in {401, 403}:
            raise ProofError('Private IAM invocation could not be established.') from None
    token = control_identity_token()
    from openecon.team_storage import TeamStorage
    storage = TeamStorage(PROJECT, BUCKET, SIGNER)  # Fresh ADC; nothing saved or printed.
    evidence = {'schema_version': 1, 'verified_at': datetime.now(timezone.utc).isoformat(),
                'origin': ORIGIN, 'anonymous_invocation_denied': True, 'cases': [],
                'pending_staging_scopes': [], 'warm_instance_reuse_confirmed': False,
                'guest_gateway_auth_rejections_verified': False,
                'analysis_after_guest_auth_probe_succeeded': False,
                'orphan_external_beacon_death_verified': False, 'passed': False}

    def save():
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')

    for case in CASES:
        result = run_case(case, storage, token, evidence, save=save)
        evidence['cases'].append(result)
        save()
        print(json.dumps({'case': case.name, 'passed': result['passed'],
                          'objects_removed': result['objects_removed']}), flush=True)
        if not result['passed']:
            break  # Never retry a launch or proceed past a failing isolation gate.
    evidence['passed'] = (len(evidence['cases']) == len(CASES)
                          and all(item['passed'] for item in evidence['cases'])
                          and not evidence['pending_staging_scopes'])
    evidence['orphan_external_beacon_death_verified'] = all(
        any(item['name'] == name and item['passed'] for item in evidence['cases'])
        for name in ('beacon_child_positive_control', 'deleted_sandbox_orphan_beacon'))
    evidence['guest_gateway_auth_rejections_verified'] = any(
        item['name'] == 'guest_gateway_broker_auth' and item['passed'] for item in evidence['cases'])
    evidence['analysis_after_guest_auth_probe_succeeded'] = any(
        item['name'] == 'analysis_after_guest_auth_probe' and item['passed'] for item in evidence['cases'])
    save()
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--verify', metavar='APPROVED_PRIVATE_BROKER_ORIGIN')
    parser.add_argument('--core-proof', type=Path)
    parser.add_argument('--report', type=Path,
                        default=Path('artifacts/verification/sandbox-broker-live.json'))
    args = parser.parse_args()
    if args.check:
        for case in CASES:
            compile(case.code, '<fixed-' + case.name + '>', 'exec')
        print(json.dumps({'fixed_cases_compiled': len(CASES), 'cloud_calls': False}))
        return 0
    if args.verify != ORIGIN or args.core_proof is None:
        print(json.dumps({'passed': False, 'error': 'The fixed private origin and core proof are required.'}))
        return 1
    try:
        evidence = verify(args.core_proof, args.report)
    except Exception:
        print(json.dumps({'passed': False, 'error': 'Private broker verification unavailable.'}))
        return 1
    print(json.dumps({'passed': evidence['passed'], 'report': str(args.report)}))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
