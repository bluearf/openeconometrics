"""Explicit disposable QA: roleless IAM denial and fresh compute isolation.

Uses existing ignored QA credentials; never changes identities or memberships.
Two normal console runs persist their approved assertions and a harmless marker
artifact. Metadata/Firebase tokens stay in memory and are never printed or saved.
This script and artifacts/ are excluded from Cloud Build by .gcloudignore.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

PROJECT = 'openecon-workbench'
REGION = 'us-central1'
PREVIEW = 'https://openecon-teams-preview-291739190496.us-central1.run.app'
STATE = Path('artifacts/team-setup/qa-state.json')
REPORT = Path('artifacts/verification/team-isolation-live.json')
PREFIX = 'OPENECON_ISOLATION_RESULT '


def first_run_code(project_id: str, marker: str, nonce: str, sign_blob: bool) -> str:
    # Only public resource names and a random nonsecret marker enter saved code.
    settings = json.dumps({'cloud_project': PROJECT, 'region': REGION,
        'project_id': project_id, 'marker': marker, 'nonce': nonce, 'sign_blob': sign_blob})
    return '''import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

_settings = json.loads(SETTINGS_LITERAL)
_fresh_variable = "isolation_probe_sentinel" not in globals()

class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Probe redirect refused")

def _probe():
    # All bearer material is function-local, with no display or file operations.
    opener = build_opener(ProxyHandler({}), _RejectRedirect())
    metadata = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/"
    try:
        request = Request(metadata + "email", headers={"Metadata-Flavor": "Google"})
        with opener.open(request, timeout=10) as response:
            identity = response.read(512).decode().strip()
        request = Request(metadata + "token", headers={"Metadata-Flavor": "Google"})
        with opener.open(request, timeout=10) as response:
            raw = response.read(16385)
        if len(raw) > 16384:
            raise ValueError("Metadata limit")
        token_data = json.loads(raw)
        token = token_data["access_token"]
        if token_data.get("token_type", "").lower() != "bearer" or not isinstance(token, str) or len(token) < 100:
            raise ValueError("Metadata token shape")
        del raw, token_data
    except Exception:
        raise RuntimeError("Worker metadata identity could not be verified") from None

    project = _settings["cloud_project"]
    expected_identity = "openecon-compute@" + project + ".iam.gserviceaccount.com"
    results = {"worker_identity_confirmed": identity == expected_identity,
               "prior_output_absent": not Path("qa-output.csv").exists(),
               "initial_marker_absent": not Path(_settings["marker"]).exists(),
               "initial_variable_absent": _fresh_variable}
    # Examine names only, never environment values or the scoped input capability.
    needles = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PRIVATE_KEY", "API_KEY", "ACCESS_KEY", "BEARER")
    forbidden = {"OPENECON_FIREBASE_CONFIG", "OPENECON_SIGNER_EMAIL", "OPENECON_OWNER_EMAIL"}
    results["credential_environment_absent"] = not any(
        name in forbidden or any(word in name.upper() for word in needles) for name in os.environ)
    root = "https://firestore.googleapis.com/v1/projects/" + project + "/databases/(default)/documents/oe_projects"
    targets = [
        ("firestore_document", root + "/" + _settings["project_id"], None),
        ("firestore_collection", root + "?pageSize=1", None),
        ("gcs_bucket_listing", "https://storage.googleapis.com/storage/v1/b/" + project + "-projects/o?maxResults=1", None),
        ("cloud_run_job", "https://run.googleapis.com/v2/projects/" + project + "/locations/" + _settings["region"] + "/jobs/openecon-compute", None),
    ]
    if _settings["sign_blob"]:
        # Fixed inert challenge, never a JWT, authorization request or credential.
        targets.append(("iam_sign_blob", "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/openecon-transfer@" + project + ".iam.gserviceaccount.com:signBlob",
                        b'{"payload":"b3BlbmVjb24taXNvbGF0aW9uLXByb2Jl"}'))
    for name, url, payload in targets:
        request = Request(url, data=payload, headers={"Authorization": "Bearer " + token,
                          "Content-Type": "application/json"}, method="POST" if payload else "GET")
        try:
            with opener.open(request, timeout=12) as response:
                status = response.status
                # Even unexpected success never reads or reveals protected data.
        except HTTPError as error:
            status = error.code
            error.close()
        except Exception:
            status = 0
        results[name] = status
    del token
    return results

_checks = _probe()
print("OPENECON_ISOLATION_RESULT " + json.dumps(_checks, sort_keys=True))
assert all(value is True if isinstance(value, bool) else value == 403 for value in _checks.values()), "Isolation probe failed; inspect approved status assertions"
isolation_probe_sentinel = _settings["nonce"]
Path(_settings["marker"]).write_text("Harmless isolation QA marker.\\n")
'''.replace('SETTINGS_LITERAL', repr(settings))


def second_run_code(marker: str) -> str:
    return f'''import json
from pathlib import Path
_checks = {{"previous_variable_absent": "isolation_probe_sentinel" not in globals(),
           "previous_marker_absent": not Path({marker!r}).exists(),
           "prior_output_absent": not Path("qa-output.csv").exists()}}
print({PREFIX!r} + json.dumps(_checks, sort_keys=True))
assert all(_checks.values()), "Fresh-run isolation failed"
'''


def approved_checks(record: dict, permitted: set[str]) -> dict:
    """Publish only an explicit bool/status whitelist, never arbitrary stdout."""
    lines = record.get('stdout', '').splitlines()
    matching = [line[len(PREFIX):] for line in lines if line.startswith(PREFIX)]
    if len(matching) != 1:
        return {}
    try:
        checks = json.loads(matching[0])
    except (ValueError, TypeError):
        return {}
    if not isinstance(checks, dict) or set(checks) != permitted:
        return {}
    if not all(isinstance(value, bool) or type(value) is int and 0 <= value <= 599
               for value in checks.values()):
        return {}
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=PREVIEW)
    parser.add_argument('--project-id', required=True, help='Existing disposable application QA project ID')
    parser.add_argument('--probe-sign-blob', action='store_true',
                        help='Also attempt signing a fixed inert challenge; expect IAM denial')
    args = parser.parse_args()
    origin = args.url.rstrip('/')
    parsed = urlsplit(origin)
    if (parsed.scheme != 'https' or not parsed.hostname or not parsed.hostname.endswith('.run.app')
            or parsed.netloc != parsed.hostname or parsed.path or parsed.query or parsed.fragment):
        raise SystemExit('Use a canonical Cloud Run HTTPS origin.')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', args.project_id):
        raise SystemExit('An explicit application project ID is required.')

    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'url': origin,
              'cloud_project': PROJECT, 'application_project_id': args.project_id,
              'sign_blob_probe_enabled': args.probe_sign_blob, 'runs': [], 'passed': False}
    stage = 'load_qa_credentials'
    try:
        state = json.loads(STATE.read_text())
        owner = state['users']['owner']
        with httpx.Client(timeout=330, follow_redirects=False, trust_env=False) as client:
            def request(method, path, *, token=None, **kwargs):
                headers = {'Origin': origin}
                if token:
                    headers['Authorization'] = 'Bearer ' + token
                response = client.request(method, origin + '/api' + path, headers=headers, **kwargs)
                if response.status_code != 200:
                    raise RuntimeError(f'API request returned HTTP {response.status_code}')
                return response.json()

            stage = 'public_config'
            config = request('GET', '/auth/config')['firebase']
            if config['projectId'] != PROJECT:
                raise RuntimeError('Unexpected Firebase project')
            stage = 'qa_signin'
            response = client.post('https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword',
                params={'key': config['apiKey']}, json={'email': owner['email'],
                    'password': owner['password'], 'returnSecureToken': True})
            if response.status_code != 200:
                raise RuntimeError(f'QA sign-in returned HTTP {response.status_code}')
            token = response.json()['idToken']
            del owner, state, response
            workspace = f'/projects/{args.project_id}/workspace'
            stage = 'preconditions'
            datasets = request('GET', workspace + '/datasets', token=token)['datasets']
            if any(item['name'] == 'qa-output.csv' for item in datasets):
                raise RuntimeError('Prior generated output is an explicit input; choose a different QA project')
            history = request('GET', workspace + '/console', token=token)
            if history['status']['running']:
                raise RuntimeError('QA project already has an active run')
            marker = 'qa-isolation-marker-' + uuid4().hex + '.txt'
            first_keys = {'worker_identity_confirmed', 'prior_output_absent', 'initial_marker_absent',
                'initial_variable_absent', 'credential_environment_absent', 'firestore_document',
                'firestore_collection', 'gcs_bucket_listing', 'cloud_run_job'}
            if args.probe_sign_blob:
                first_keys.add('iam_sign_blob')
            for label, code, keys, timeout in [
                ('iam_boundaries', first_run_code(args.project_id, marker, uuid4().hex, args.probe_sign_blob), first_keys, 90),
                ('fresh_execution', second_run_code(marker), {'previous_variable_absent',
                    'previous_marker_absent', 'prior_output_absent'}, 30),
            ]:
                stage = label
                print(f'Starting {label}; no credentials will be printed.', flush=True)
                started = time.monotonic()
                record = request('POST', workspace + '/console/execute', token=token,
                                 json={'code': code, 'timeout_seconds': timeout})
                checks = approved_checks(record, keys)
                passed = (record.get('status') == 'ok' and bool(checks)
                    and all(value is True if isinstance(value, bool) else value == 403
                            for value in checks.values()) and record.get('state_reset') is True)
                if label == 'iam_boundaries':
                    passed = passed and any(item.get('name') == marker for item in record.get('artifacts', []))
                report['runs'].append({'probe': label, 'run_id': record.get('id'),
                    'duration_seconds': round(time.monotonic() - started, 2), 'checks': checks, 'passed': passed})
                if not passed:
                    raise RuntimeError('Approved probe assertions failed')
                print('PASS ' + label + ': ' + json.dumps(checks, sort_keys=True), flush=True)
            del token
            report['passed'] = True
    except Exception as error:
        # Provider exceptions/record errors may contain capabilities; only class
        # and our fixed stage name are approved for the local report or console.
        report['failure'] = {'stage': stage, 'error_type': type(error).__name__}
        print(f'FAIL {stage} ({type(error).__name__}); no provider details emitted.', flush=True)
    finally:
        report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {REPORT}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
