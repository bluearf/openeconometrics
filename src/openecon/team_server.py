"""Authenticated team sync backend. It never executes user Python or estimators.

Analyses run in the local desktop app. This service stores accounts, project
membership, files and desktop-computed results that members choose to share.
"""
from __future__ import annotations

import base64
import hashlib
from importlib.resources import files
import ipaddress
import json
import logging
import math
from pathlib import Path
import re
from urllib.parse import quote, urlparse, urlsplit
from uuid import uuid4

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from openecon import __version__
from openecon.output_events import validate_output_events
from openecon.team_auth import TeamAuthError
from openecon.team_storage import MAX_TRANSFER_BYTES
from openecon.team_store import (MAX_PROJECT_DESCRIPTION_LENGTH, MAX_PROJECT_NAME_LENGTH,
                                MAX_PROJECT_NAME_VERSION, TeamError, filename, now,
                                project_description, project_name)

_DNS_LABEL = re.compile(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z')
_SOURCE_COMMIT = re.compile(r'[0-9a-f]{40}(?:[0-9a-f]{24})?\Z')
CLOUD_EXECUTION_RETIRED = (
    'Analyses run only in the OpenEconometrics desktop app. Open this project in the desktop '
    'app to run code; results you share from the desktop appear in the project history.')


def validate_public_origin(value: str) -> None:
    """Accept one canonical HTTPS DNS origin, without path, port or credentials."""
    if not isinstance(value, str) or not value.isascii():
        raise ValueError('Cloud public_origin must be a canonical HTTPS origin.')
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ''
        valid = (
            parsed.scheme == 'https'
            and value == f'https://{host}'
            and 0 < len(host) <= 253
            and len(host.split('.')) >= 2
            and all(_DNS_LABEL.fullmatch(label) for label in host.split('.'))
        )
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            valid = False
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Cloud public_origin must be a canonical HTTPS DNS origin without '
                         'a path, port, credentials, query, or fragment.')


def source_commit(value) -> str | None:
    """Return a full lowercase source commit ID, or None when the build bound none."""
    return value if isinstance(value, str) and _SOURCE_COMMIT.fullmatch(value) else None


class VersionedStaticFiles(StaticFiles):
    """Vite content hashes allow public files to survive browser reloads."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if (response.status_code in {200, 304}
                and re.fullmatch(r'[\w.-]+-[\w-]{8,}\.[\w]+', Path(path).name)):
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        return response


def request_body_limit(path: str) -> int:
    if re.fullmatch(r'/api/projects/[0-9a-f]{32}/workspace/transfers/[0-9a-f]{32}/parts/[0-9]+', path):
        from openecon.team_transfer import PART_BYTES
        return PART_BYTES
    if path.endswith('/datasets/upload'):
        return MAX_TRANSFER_BYTES + 65536
    if re.fullmatch(r'/api/projects/[0-9a-f]{32}/workspace/desktop/results', path):
        return 3 * 1024**2
    return 512 * 1024


class BoundedRequestBody:
    """Bound streamed/chunked bodies before multipart parsing or disk spooling."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        limit = request_body_limit(scope.get('path', ''))
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > limit:
                    from starlette.exceptions import HTTPException
                    raise HTTPException(413, 'The request exceeds the size limit.')
            return message
        await self.app(scope, bounded_receive, send)


class StrictBody(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ProjectBody(StrictBody):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(min_length=1, max_length=MAX_PROJECT_NAME_LENGTH)
    description: str = Field(default='', max_length=MAX_PROJECT_DESCRIPTION_LENGTH)

    @field_validator('name', mode='before')
    @classmethod
    def normalize_name(cls, value):
        return project_name(value)

    @field_validator('description', mode='before')
    @classmethod
    def normalize_description(cls, value):
        return project_description(value)


class ProjectRenameBody(StrictBody):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str
    name_version: int = Field(ge=0, le=MAX_PROJECT_NAME_VERSION)


class InviteBody(StrictBody):
    email: str = Field(min_length=3, max_length=254)
    role: str


class RoleBody(StrictBody):
    role: str


class ScriptBody(StrictBody):
    code: str = Field(max_length=64000)
    name: str = Field(default='analysis.py', pattern=r'^analysis\.py$')
    version: int = Field(ge=0, strict=True)


class ScriptCreateBody(StrictBody):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(default='untitled.py', max_length=180)
    code: str = Field(default='', max_length=64000)
    id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')


class NamedScriptBody(StrictBody):
    model_config = ConfigDict(extra='forbid', strict=True)
    code: str = Field(max_length=64000)
    version: int = Field(ge=0, lt=2**63 - 1)


class EnvironmentBody(StrictBody):
    manifest: dict
    version: int = Field(ge=0, lt=2**63 - 1, strict=True)


class FileLayoutBody(StrictBody):
    model_config = ConfigDict(extra='forbid', strict=True)
    version: int = Field(ge=0, lt=2**63 - 1)
    entries: list[dict] = Field(max_length=2000)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


def error_record(run, message, status='error'):
    return {'id': run['id'], 'code': run['code'], 'created_at': run['created_at'],
            'status': status, 'stdout': '', 'outputs': [], 'variables': [],
            'error': {'type': status.upper(), 'message': message, 'traceback': ''},
            'duration_ms': 0, 'session_generation': run['generation'],
            'active_session_generation': run['generation'], 'state_reset': True}


def validate_worker_result(payload, run):
    """Untrusted JSON is data; reject oversized/malformed output before publish.

    Desktop-shared results use this schema gate. Historical cloud records were
    published through the same checks and remain readable unchanged.
    """
    if not isinstance(payload, dict) or payload.get('execution_id') != run['id']:
        raise TeamError('INVALID_RESULT', 'The computation result could not be verified.', 422)
    record = payload.get('record')
    if not isinstance(record, dict) or record.get('status') not in {'ok', 'error', 'timeout', 'interrupted'}:
        raise TeamError('INVALID_RESULT', 'The computation result could not be verified.', 422)
    stdout, outputs = record.get('stdout', ''), record.get('outputs', [])
    if (not isinstance(stdout, str) or len(stdout.encode()) > 70 * 1024
            or not isinstance(outputs, list) or len(outputs) > 20
            or len(json_bytes(outputs)) > 2 * 1024**2):
        raise TeamError('RESULT_LIMIT', 'The computation output exceeded the size limit.', 422)
    if 'events' in record:
        try:
            validate_output_events(record['events'], stdout, outputs)
        except ValueError:
            raise TeamError('INVALID_RESULT', 'The output order is invalid.', 422) from None
    for item in outputs:
        if not isinstance(item, dict) or item.get('type') not in {'model', 'plot', 'table', 'text', 'latex'}:
            raise TeamError('INVALID_RESULT', 'The output format is invalid.', 422)
        data = item.get('data')
        if item['type'] in {'text', 'latex'} and not isinstance(data, str):
            raise TeamError('INVALID_RESULT', 'The text output is invalid.', 422)
        if item['type'] not in {'text', 'latex'} and not isinstance(data, dict):
            raise TeamError('INVALID_RESULT', 'The output format is invalid.', 422)
        from openecon.output_latex import validate_latex_fields
        try:
            validate_latex_fields(item)
        except ValueError:
            raise TeamError('INVALID_RESULT', 'The LaTeX output is invalid.', 422) from None
        if item['type'] == 'table':
            cols, rows = data.get('columns'), data.get('rows')
            indexes, index_names = data.get('index'), data.get('index_names')
            if (not isinstance(cols, list) or len(cols) > 30 or not all(isinstance(c, str) for c in cols)
                    or not isinstance(rows, list) or len(rows) > 50
                    or not all(isinstance(r, list) and len(r) == len(cols) for r in rows)
                    or not isinstance(data.get('total_rows'), int)
                    or not isinstance(data.get('total_columns'), int)
                    or data['total_rows'] < len(rows) or data['total_columns'] < len(cols)
                    or (index_names is not None and (not isinstance(index_names, list)
                        or not all(x is None or isinstance(x, str) for x in index_names)))
                    or (indexes is not None and (not isinstance(indexes, list) or len(indexes) != len(rows)
                        or not all(isinstance(x, list) and len(x) <= 30 for x in indexes)))):
                raise TeamError('INVALID_RESULT', 'The table output is invalid.', 422)
        if item['type'] == 'model':
            from openecon.models import ResultBundle
            try:
                if not isinstance(data.get('coefficients'), list) or len(data['coefficients']) > 500:
                    raise ValueError('coefficients')
                fields = {k: v for k, v in data.items() if k != 'display_omitted'}
                fields.setdefault('covariance_matrix', [])
                fields.setdefault('sample_positions', [])
                validated = ResultBundle.model_validate(fields)
                item['data'] = validated.model_dump(mode='json', exclude={'covariance_matrix', 'sample_positions'})
                item['data']['display_omitted'] = ['covariance_matrix', 'sample_positions']
            except (ValueError, TypeError):
                raise TeamError('INVALID_RESULT', 'The model output is invalid.', 422) from None
        if item['type'] == 'plot':
            from openecon.team_output import validate_plot
            try:
                item['data'] = validate_plot(data)
            except (ValueError, TypeError):
                raise TeamError('INVALID_RESULT', 'The chart output is invalid.', 422) from None
    error = record.get('error')
    if error is not None and (not isinstance(error, dict)
                             or not all(isinstance(error.get(k), str) and len(error[k]) <= 20000
                                        for k in ('type', 'message', 'traceback'))):
        raise TeamError('INVALID_RESULT', 'The error output is invalid.', 422)
    duration = record.get('duration_ms', 0)
    if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
        raise TeamError('INVALID_RESULT', 'The duration output is invalid.', 422)
    trusted = {'id': run['id'], 'code': run['code'], 'created_at': run['created_at'],
               'status': record['status'], 'stdout': stdout, 'outputs': outputs, 'error': error,
               'variables': [], 'duration_ms': min(duration, 300000),
               'session_generation': run['generation'], 'active_session_generation': run['generation'],
               'state_reset': True, 'actor_email': run['email']}
    if 'events' in record:
        trusted['events'] = [dict(event) for event in record['events']]
    generated = payload.get('generated_files', [])
    if not isinstance(generated, list) or len(generated) > 20:
        raise TeamError('RESULT_LIMIT', 'Up to 20 output files can be saved.', 422)
    decoded, total, names = [], 0, set()
    for output in generated:
        if not isinstance(output, dict):
            raise TeamError('INVALID_RESULT', 'The file output is invalid.', 422)
        name = filename(output.get('name'))
        if name in names:
            raise TeamError('INVALID_RESULT', 'Duplicate output file.', 422)
        names.add(name)
        try:
            body = base64.b64decode(output['content_base64'], validate=True)
        except (KeyError, ValueError, TypeError):
            raise TeamError('INVALID_RESULT', 'The file output is invalid.', 422) from None
        total += len(body)
        if total > 16 * 1024**2 or output.get('size') != len(body):
            raise TeamError('RESULT_LIMIT', 'The output files exceed the combined 16 MiB limit.', 422)
        decoded.append((name, body))
    return trusted, decoded


def create_team_app(*, store, storage, auth, public_origin, firebase_config, desktop_token_issuer=None,
                    source_commit_id=None):
    validate_public_origin(public_origin)
    commit = source_commit(source_commit_id)
    # The web configuration may not expand our script/frame trust boundary.
    # Only this verified Firebase project's default authentication host is
    # supported; custom domains require a separate, reviewed allowlist change.
    auth_domain = f'{auth.project_id}.firebaseapp.com'
    if (not isinstance(firebase_config, dict)
            or firebase_config.get('projectId') != auth.project_id
            or firebase_config.get('authDomain') != auth_domain):
        raise ValueError('Firebase authDomain must be this project\'s exact firebaseapp.com domain.')
    firebase_config = dict(firebase_config)
    content_security_policy = (
        "default-src 'self'; script-src 'self' https://apis.google.com; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
        "connect-src 'self' https://identitytoolkit.googleapis.com https://securetoken.googleapis.com; "
        f"frame-src https://{auth_domain}; font-src 'self'; object-src 'none'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    app = FastAPI(title='OpenEconometrics Teams', version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.team_store, app.state.team_storage = store, storage
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[urlparse(public_origin).hostname, 'testserver'])
    app.add_middleware(BoundedRequestBody)

    @app.exception_handler(TeamError)
    @app.exception_handler(TeamAuthError)
    async def expected_error(request, exc):
        return JSONResponse({'detail': {'code': exc.code, 'message': str(exc)}}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_fields(request, exc):
        if request.method == 'POST' and request.url.path == '/api/projects':
            # Field errors must survive malformed Unicode without echoing drafts
            # or serializing validator exception objects from Pydantic's context.
            errors = [{key: error[key] for key in ('loc', 'msg', 'type')} for error in exc.errors()]
            return Response(json.dumps({'detail': errors}, ensure_ascii=True), status_code=422,
                            media_type='application/json')
        return await request_validation_exception_handler(request, exc)

    @app.middleware('http')
    async def guards(request: Request, call_next):
        path = request.url.path
        if path == '/healthz' and request.method in {'GET', 'HEAD'}:
            return JSONResponse({'status': 'ok'})
        try:
            # No ambient cookies are used: authorization always needs an explicit
            # bearer token, and writes must originate from this interface.
            origin = request.headers.get('origin')
            if origin and origin != public_origin:
                raise TeamError('ORIGIN_DENIED', 'This origin is not allowed.', 403)
            if path.startswith('/api/'):
                if request.headers.get('sec-fetch-site') == 'cross-site':
                    raise TeamError('ORIGIN_DENIED', 'Cross-site API requests are not allowed.', 403)
                length = request.headers.get('content-length')
                if length and (not length.isdigit() or int(length) > request_body_limit(path)):
                    raise TeamError('FILE_LIMIT', 'The request exceeds the size limit.', 413)
                from openecon.desktop_cloud import public_desktop_login
                if path != '/api/auth/config' and not public_desktop_login(path, request.method):
                    authorization = request.headers.get('authorization', '')
                    token = authorization[7:] if authorization.startswith('Bearer ') else ''
                    request.state.user = await run_in_threadpool(
                        auth.verify, token, require_verified=path != '/api/me')
            response = await call_next(request)
        except (TeamError, TeamAuthError) as exc:
            response = JSONResponse({'detail': {'code': exc.code, 'message': str(exc)}}, status_code=exc.status_code)
        except Exception as exc:
            # Cloud SDK exceptions can contain signed capability URLs. Do not
            # expose them in HTTP responses or application traceback logs.
            logging.getLogger('openecon.teams').error('Control request failed: %s', type(exc).__name__)
            response = JSONResponse({'detail': {'code': 'SERVICE_UNAVAILABLE',
                'message': 'The operation could not be completed right now. Your saved files are preserved.'}}, status_code=503)
        response.headers.update({
            'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
            'X-Frame-Options': 'DENY', 'Cross-Origin-Opener-Policy': 'same-origin-allow-popups',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': content_security_policy,
        })
        if path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/auth/config')
    def config():
        return {'mode': 'teams', 'firebase': firebase_config, 'desktop_login_available': True,
                'account_link_available': True, 'dataset_transfer_available': True,
                'cloud_execution_available': False, 'source_commit': commit}

    from openecon.desktop_cloud import attach_desktop_cloud_routes
    attach_desktop_cloud_routes(app, store=store, storage=storage, token_issuer=desktop_token_issuer)
    from openecon.team_transfer import attach_transfer_routes
    attach_transfer_routes(app, store=store, storage=storage)

    @app.get('/api/me')
    def me(request: Request):
        return store.me(request.state.user)

    @app.get('/api/projects')
    def projects(request: Request):
        return {'projects': store.projects(request.state.user)}

    @app.post('/api/projects', status_code=201)
    def create_project(body: ProjectBody, request: Request):
        return store.create_project(request.state.user, body.name, body.description)

    @app.patch('/api/projects/{project_id}')
    def rename_project(project_id: str, body: ProjectRenameBody, request: Request):
        return store.rename_project(project_id, request.state.user, body.name, body.name_version)

    @app.get('/api/projects/{project_id}/members')
    def members(project_id: str, request: Request):
        return store.members(project_id, request.state.user)

    @app.post('/api/projects/{project_id}/invitations', status_code=201)
    def invite(project_id: str, body: InviteBody, request: Request):
        return store.invite(project_id, request.state.user, body.email, body.role)

    @app.post('/api/invitations/{invitation_id}/accept')
    def accept(invitation_id: str, request: Request):
        return store.accept(invitation_id, request.state.user)

    @app.delete('/api/projects/{project_id}/invitations/{invitation_id}')
    def revoke(project_id: str, invitation_id: str, request: Request):
        store.revoke_invite(project_id, invitation_id, request.state.user)
        return {'status': 'revoked'}

    @app.patch('/api/projects/{project_id}/members/{uid}')
    def role(project_id: str, uid: str, body: RoleBody, request: Request):
        store.change_member(project_id, uid, request.state.user, body.role)
        return {'status': 'updated'}

    @app.delete('/api/projects/{project_id}/members/{uid}')
    def remove(project_id: str, uid: str, request: Request):
        store.change_member(project_id, uid, request.state.user)
        return {'status': 'removed'}

    prefix = '/api/projects/{project_id}/workspace'

    def session_payload(project, user):
        return {'token': '', 'version': __version__, 'environment': 'team', 'persistent': True,
                'upload_limit_bytes': MAX_TRANSFER_BYTES, 'execution_mode': 'desktop',
                'read_only': project['members'][user.uid]['role'] == 'viewer'}

    def idle_status(project):
        # No cloud computation exists. A legacy active_run marker is retained
        # unchanged for preservation, but it no longer reports a live run.
        return {'running': False, 'session_generation': project['run_count'], 'pid': None}

    @app.get(prefix + '/session')
    def session(project_id: str, request: Request):
        return session_payload(store.project(project_id, request.state.user), request.state.user)

    @app.get(prefix + '/bootstrap')
    def bootstrap(project_id: str, request: Request):
        # script() authorizes before reading the draft. Recheck membership after
        # that read so removal/downgrade cannot leak a draft or editable session.
        draft = store.script(project_id, request.state.user)
        environment = store.environment(project_id, request.state.user)
        project = store.project(project_id, request.state.user)
        return {'session': session_payload(project, request.state.user), 'draft': draft,
                'environment': environment,
                'datasets': [file_public(f) for f in project['files']],
                'status': idle_status(project)}

    @app.get(prefix + '/config')
    def workspace_config(project_id: str, request: Request):
        store.project(project_id, request.state.user)
        return {'version': __version__, 'mcp_available': False, 'mcp_command': '',
                'codex_command': '', 'claude_command': '',
                'notice': 'Remote agent connections are not yet available in the team version.'}

    @app.get(prefix + '/console/script')
    def script(project_id: str, request: Request):
        return store.script(project_id, request.state.user)

    @app.put(prefix + '/console/script')
    def save_script(project_id: str, body: ScriptBody, request: Request):
        return store.save_script(project_id, request.state.user, body.code, body.version)

    @app.get(prefix + '/console/scripts')
    def scripts(project_id: str, request: Request):
        return store.scripts(project_id, request.state.user)

    @app.post(prefix + '/console/scripts', status_code=201)
    def create_script(project_id: str, body: ScriptCreateBody, request: Request):
        return store.create_script(project_id, request.state.user, body.name, body.code, script_id=body.id)

    @app.get(prefix + '/console/scripts/{script_id}')
    def named_script(project_id: str, script_id: str, request: Request):
        return store.named_script(project_id, request.state.user, script_id)

    @app.put(prefix + '/console/scripts/{script_id}')
    def save_named_script(project_id: str, script_id: str, body: NamedScriptBody, request: Request):
        return store.save_named_script(project_id, request.state.user, script_id, body.code, body.version)

    @app.get(prefix + '/files/layout')
    def get_file_layout(project_id: str, request: Request):
        return store.get_file_layout(project_id, request.state.user)

    @app.put(prefix + '/files/layout')
    def put_file_layout(project_id: str, body: FileLayoutBody, request: Request):
        return store.put_file_layout(project_id, request.state.user, body.model_dump())

    @app.get(prefix + '/environment')
    def environment(project_id: str, request: Request):
        return store.environment(project_id, request.state.user)

    @app.put(prefix + '/environment')
    def save_environment(project_id: str, body: EnvironmentBody, request: Request):
        return store.save_environment(project_id, request.state.user, body.manifest, body.version)

    def file_public(metadata):
        return {k: v for k, v in metadata.items() if k != 'blob'}

    @app.get(prefix + '/datasets')
    def datasets(project_id: str, request: Request):
        return {'datasets': [file_public(f) for f in store.project(project_id, request.state.user)['files']]}

    def save_upload(project_id, user, name, body, source='upload'):
        store.project(project_id, user, 'editor')
        name = filename(name)
        file_id = uuid4().hex
        reference = storage.put(f'projects/{project_id}/files/{file_id}/{name}', body)
        metadata = {'id': file_id, 'name': name, 'python_path': name, 'source': source,
                    'size_bytes': len(body), 'data_hash': hashlib.sha256(body).hexdigest(),
                    'row_count': 480 if source == 'example' else None,
                    'column_count': 6 if source == 'example' else None,
                    'columns': [], 'preview': [], 'created_at': now(), 'blob': reference}
        try:
            store.add_file(project_id, user, metadata)
        except Exception:
            # Firestore may have committed before a transport acknowledgement
            # was lost. Never delete the generation a published file references.
            saved = store.db.get(f'oe_projects/{project_id}')
            committed = next((f for f in (saved or {}).get('files', []) if f['id'] == file_id), None)
            if committed and committed.get('blob') == reference:
                store.project(project_id, user, 'editor')
                return file_public(committed)
            storage.delete(reference)
            raise
        return file_public(metadata)

    @app.post(prefix + '/datasets/upload', status_code=201)
    async def upload(project_id: str, request: Request, file: UploadFile = File(...)):
        try:
            await run_in_threadpool(store.project, project_id, request.state.user, 'editor')
            name = filename(file.filename or '')
            if Path(name).suffix.lower() not in {'.csv', '.parquet', '.xlsx', '.dta'}:
                raise TeamError('UNSUPPORTED_FORMAT', 'Upload a CSV, Parquet, XLSX or DTA file.', 422)
            data = bytearray()
            while chunk := await file.read(1024**2):
                data.extend(chunk)
                if len(data) > MAX_TRANSFER_BYTES:
                    raise TeamError('FILE_LIMIT', 'The file size limit is 24 MiB.', 413)
            return await run_in_threadpool(save_upload, project_id, request.state.user, name, bytes(data))
        finally:
            await file.close()

    @app.post(prefix + '/datasets/example', status_code=201)
    def example(project_id: str, request: Request):
        return save_upload(project_id, request.state.user, 'wages.csv',
                           files('openecon').joinpath('examples/wages.csv').read_bytes(), 'example')

    @app.get(prefix + '/files/{file_id}/download')
    def download_file(project_id: str, file_id: str, request: Request):
        project = store.project(project_id, request.state.user)
        item = next((f for f in project['files'] if f['id'] == file_id), None)
        if item is None:
            raise TeamError('NOT_FOUND', 'File not found.', 404)
        if item.get('transfer') == 'chunked-v1':
            raise TeamError('CHUNKED_DOWNLOAD', 'Download this data through the verified manifest and parts.', 409)
        data = storage.get(item['blob'])
        store.project(project_id, request.state.user)
        return Response(data, media_type='application/octet-stream',
                        headers={'Content-Disposition': f"attachment; filename*=UTF-8''{quote(item['name'])}"})

    terminal = {'finished', 'failed', 'cancelled'}

    def read_record(run):
        if run['state'] not in terminal:
            # Cloud execution was retired. A historical run that never reached
            # a terminal state is shown as stopped; its stored document is unchanged.
            return error_record(run, 'This cloud run did not finish before cloud execution was retired. '
                                'Run the analysis in the desktop app.', 'interrupted')
        if run.get('result'):
            from openecon.output_latex import enrich_record
            return enrich_record(json.loads(storage.get(run['result'], maximum=3 * 1024**2)))
        summary = run.get('record_summary') or {}
        return error_record(run, summary.get('message', 'The computation could not be completed.'), summary.get('status', 'error'))

    @app.get(prefix + '/console/history')
    def paged_history(project_id: str, request: Request, cursor: str | None = Query(default=None, max_length=2048),
                      query: str = Query(default='', max_length=256), since: str = '', until: str = '',
                      limit: int = Query(default=20, ge=1, le=50),
                      max_bytes: int = Query(default=64 * 1024, ge=4096, le=256 * 1024)):
        from openecon.team_history import encoded, history_page
        page = history_page(store, project_id, request.state.user, cursor=cursor, query=query,
                            since=since, until=until, limit=limit, max_bytes=max_bytes)
        return Response(encoded(page), media_type='application/json')

    @app.get(prefix + '/runs/{run_id}/record')
    def historical_record(project_id: str, run_id: str, request: Request):
        store.project(project_id, request.state.user)
        run = store.run(project_id, run_id)
        if not run:
            raise TeamError('NOT_FOUND', 'Run not found.', 404)
        data = json_bytes(read_record(run))
        store.project(project_id, request.state.user)
        if len(data) > 8 * 1024**2:
            raise TeamError('RESULT_LIMIT', 'The saved result exceeds the response limit.', 413)
        return Response(data, media_type='application/json')

    @app.get(prefix + '/console')
    def console(project_id: str, request: Request):
        project = store.project(project_id, request.state.user)
        history, size = [], 0
        for run in reversed(store.runs(project_id, request.state.user)):
            estimate = (run.get('result') or {}).get('size', 1024)
            if history and size + estimate > 8 * 1024**2:
                break
            record = read_record(run)
            size += len(json_bytes(record))
            history.append(record)
        history.reverse()
        store.project(project_id, request.state.user)
        return {'history': history, 'variables': [], 'status': idle_status(project)}

    @app.post(prefix + '/console/execute')
    @app.post(prefix + '/console/interrupt')
    @app.post(prefix + '/console/reset')
    def cloud_execution_retired(project_id: str, request: Request):
        # Older browser clients still call these routes. Authentication and the
        # membership check run first; the request body and code are never read.
        store.project(project_id, request.state.user)
        raise TeamError('CLOUD_EXECUTION_RETIRED', CLOUD_EXECUTION_RETIRED, 410)

    @app.get(prefix + '/runs/{run_id}/files/{index}')
    def artifact(project_id: str, run_id: str, index: int, request: Request):
        store.project(project_id, request.state.user)
        run = store.run(project_id, run_id)
        items = (run or {}).get('artifacts', [])
        if not run or run['state'] != 'finished' or not 0 <= index < len(items):
            raise TeamError('NOT_FOUND', 'Output file not found.', 404)
        item = items[index]
        data = storage.get(item['blob'])
        store.project(project_id, request.state.user)
        return Response(data, media_type='application/octet-stream',
                        headers={'Content-Disposition': f"attachment; filename*=UTF-8''{quote(item['name'])}"})

    static = Path(str(files('openecon').joinpath('static')))
    chart_assets = Path(str(files('openecon_charts').joinpath('assets')))
    app.mount('/chart-assets', StaticFiles(directory=chart_assets), name='charts')
    if (static / 'assets').is_dir():
        app.mount('/assets', VersionedStaticFiles(directory=static / 'assets'), name='assets')

    @app.get('/')
    def index():
        return FileResponse(static / 'index.html', headers={'Cache-Control': 'no-cache'})

    return app
