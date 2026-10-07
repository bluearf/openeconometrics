"""Loopback-only workspaces for the desktop shell.

Cloud sign-in and file transfers belong to the native shell. This module never
accepts cloud credentials, imports the cloud runner, or executes saved drafts
when a project is opened. Python has the local user's permissions.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
from typing import Annotated, Literal
from urllib.parse import urlparse
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from openecon import __version__
from openecon.data import DataError, MAX_FILE_BYTES
from openecon.server import ConsoleRequest, FileLayoutRequest, create_app
from openecon.workspace import _atomic_json
from openecon import file_layout, script_contracts
from openecon.desktop_sharing import (
    DesktopError, DesktopSharing, InputFile, OutboxItem, SharingUpdate, RetryRequest,
    _load_json, actor_uid, network_summaries,
)

_PROJECT_ID = re.compile(r"[0-9a-f]{32}\Z")
_PROJECT_ROUTE = re.compile(r"/([0-9a-f]{32})/workspace(/.*)?\Z")
_RESERVED = {"datasets", "results", "console", "desktop-sync-state.json", "desktop-cache-index.json",
             "desktop-outbox.json", "desktop-sharing.json", "desktop-file-layout-sync.json", ".packages"}
# JSON may escape each valid control character into six bytes. The underlying
# acknowledged Python text is independently capped at the project byte limit.
MAX_SYNC_STATE_BYTES = script_contracts.MAX_SCRIPT_BYTES * 6 + 1024 * 1024


class LocalProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)

    @field_validator("name", "description")
    @classmethod
    def clean_text(cls, value):
        if any(ord(char) < 32 and char not in "\n\t" for char in value):
            raise ValueError("Project text contains invalid characters.")
        return value.strip()

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value):
        if not value:
            raise ValueError("Enter a project name.")
        return value
_SAFE_ENVIRONMENT = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR", "APPDATA", "LOCALAPPDATA",
    "USERPROFILE", "VIRTUAL_ENV",
}


def worker_environment() -> dict[str, str]:
    """Keep OS essentials; never propagate auth, cloud, proxy or Python hooks."""
    environment = {key: value for key, value in os.environ.items() if key in _SAFE_ENVIRONMENT}
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    return environment


class DesktopConsoleRequest(ConsoleRequest):
    model_config = ConfigDict(extra="forbid", strict=True)
    actor_uid: str
    input_files: list[InputFile] = Field(max_length=20)
    _actor = field_validator("actor_uid")(actor_uid)

    @field_validator("input_files")
    @classmethod
    def unique_references(cls, value):
        if len({item.id for item in value}) != len(value):
            raise ValueError("The input file references contain duplicates.")
        return value


class ScriptSyncState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    cloud_version: int | None = Field(ge=0, lt=script_contracts.MAX_SCRIPT_VERSION)
    local_version: int = Field(ge=0, lt=script_contracts.MAX_SCRIPT_VERSION)
    base_code: str = Field(max_length=64000)
    conflict: bool = False

    _name = field_validator("name")(script_contracts.script_name)
    _code = field_validator("base_code")(script_contracts.script_code)


class ScriptSyncMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^(?:analysis|[0-9a-f]{32})$")
    name: str
    version: int = Field(ge=0, lt=script_contracts.MAX_SCRIPT_VERSION)

    _name = field_validator("name")(script_contracts.script_name)


class SyncState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cloud_version: int = Field(ge=0)
    base_code: str = Field(max_length=64000)
    role: Literal["owner", "editor", "viewer"]
    scripts: dict[Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")], ScriptSyncState] = Field(
        default_factory=dict, max_length=script_contracts.MAX_SCRIPTS - 1)
    script_catalog: list[ScriptSyncMetadata] = Field(default_factory=list, max_length=script_contracts.MAX_SCRIPTS)
    access_denied: bool = False

    _code = field_validator("base_code")(script_contracts.script_code)

    @model_validator(mode="after")
    def bounded_scripts(self):
        if len(self.base_code.encode("utf-8")) + sum(len(entry.base_code.encode("utf-8"))
                                                  for entry in self.scripts.values()) > script_contracts.MAX_SCRIPT_BYTES:
            raise ValueError("The saved synchronization state exceeds the project limit.")
        identifiers = [entry.id for entry in self.script_catalog]
        if (len(set(identifiers)) != len(identifiers)
                or (identifiers and "analysis" not in identifiers)
                or any(entry.id == "analysis" and entry.name != "analysis.py" for entry in self.script_catalog)):
            raise ValueError("The saved source file catalog is invalid.")
        return self


class CachedScript(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    code: str = Field(max_length=64000)
    version: int = Field(ge=0, lt=script_contracts.MAX_SCRIPT_VERSION)

    _name = field_validator("name")(script_contracts.script_name)
    _code = field_validator("code")(script_contracts.script_code)


class FileLayoutSyncCache(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = Field(ge=0, lt=file_layout.MAX_VERSION)
    cloud_version: int | None = Field(ge=0, lt=file_layout.MAX_VERSION)
    base_entries: list[dict] = Field(max_length=file_layout.MAX_ENTRIES)
    conflict: bool

    @field_validator("base_entries")
    @classmethod
    def valid_entries(cls, value):
        return file_layout.validate({"version": 0, "entries": value})["entries"]


class CachedFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=200)
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cloud_id: str = Field(pattern=r"^[0-9a-f]{32}$")


def _cached_name(name: str) -> str:
    try:
        encoded_name = name.encode("utf-8")
    except UnicodeError as exc:
        raise DesktopError("INVALID_FILENAME", "The project file name is invalid.") from exc
    if (not 1 <= len(encoded_name) <= 180 or name.startswith(".") or name != name.strip()
            or name in _RESERVED or "/" in name or "\\" in name or ":" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name) or Path(name).name != name):
        raise DesktopError("INVALID_FILENAME", "Use a simple file name inside this project.")
    return name


def _safe_directory(path: Path, parent: Path) -> None:
    if path.is_symlink() or path.resolve().parent != parent.resolve():
        raise DesktopError("UNSAFE_PROJECT_PATH", "The local project directory is unsafe.")
    try:
        path.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise DesktopError("UNSAFE_PROJECT_PATH", "The local project directory is unavailable.") from exc
    if not path.is_dir():
        raise DesktopError("UNSAFE_PROJECT_PATH", "The local project directory is unavailable.")


def _check_workspace(path: Path) -> None:
    """Do not follow links in files produced by arbitrary local Python code."""
    if path.is_symlink():
        raise DesktopError("UNSAFE_PROJECT_PATH", "The local project directory is unsafe.")
    for folder in (path, path / "datasets", path / "results", path / "console"):
        if folder.is_symlink() or not folder.is_dir() or folder.resolve().parent != (
            path.parent if folder == path else path
        ).resolve():
            raise DesktopError("UNSAFE_PROJECT_PATH", "The local project directory is unsafe.")
        for item in folder.iterdir():
            if item.is_symlink():
                raise DesktopError("UNSAFE_PROJECT_PATH", "A linked local project file cannot be used.")


class DesktopProjects:
    """One active worker; durable files and fresh tokens belong to each project."""
    def __init__(self, data_root: str | Path):
        self.root = Path(data_root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.projects_path = self.root / "projects"
        _safe_directory(self.projects_path, self.root)
        self.lock = threading.RLock()
        self.active_project: str | None = None
        self.active_app: FastAPI | None = None
        self.closed = False

    def project_path(self, project_id: str) -> Path:
        if not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id):
            raise DesktopError("INVALID_PROJECT", "A canonical project identifier is required.", 404)
        if self.projects_path.is_symlink() or self.projects_path.resolve().parent != self.root:
            raise DesktopError("UNSAFE_PROJECT_PATH", "The local projects directory is unsafe.")
        path = self.projects_path / project_id
        _safe_directory(path, self.projects_path)
        for name in ("datasets", "results", "console"):
            _safe_directory(path / name, path)
        _check_workspace(path)
        return path

    def activate(self, project_id: str) -> FastAPI:
        with self.lock:
            if self.closed:
                raise DesktopError("DESKTOP_CLOSED", "The desktop runtime is closed.", 409)
            path = self.project_path(project_id)
            if self.active_project == project_id:
                return self.active_app
            if self.active_app is not None:
                self.active_app.state.packages.close()
                self.active_app.state.console.close()
            app = create_app(path, project_packages=True)
            # The original local API stays unchanged. Only desktop workers use
            # the environment allowlist; neither opening nor saving starts one.
            app.state.console.worker_environment = worker_environment()
            self._add_project_routes(app, path)
            self.active_project, self.active_app = project_id, app
            return app

    def current(self, project_id: str) -> FastAPI:
        with self.lock:
            if self.active_project != project_id or self.active_app is None:
                raise DesktopError("PROJECT_INACTIVE", "Open this project again to establish a local session.", 409)
            self.project_path(project_id)
            return self.active_app

    def _add_project_routes(self, app: FastAPI, path: Path):
        store = app.state.workspace
        sharing = DesktopSharing(path, store)

        def require_editing():
            _check_workspace(path)
            raw = _load_json(path / "desktop-sync-state.json", None,
                             maximum=MAX_SYNC_STATE_BYTES)
            if raw is None:
                return
            try:
                saved = SyncState.model_validate(raw)
            except ValueError as exc:
                raise DesktopError("CORRUPT_LOCAL_STATE", "The project record could not be read.") from exc
            if saved.role == "viewer" or saved.access_denied:
                raise DesktopError("ROLE_REQUIRED", "You do not have permission to share results for this project.", 403)

        def layout_catalog(*, editing=False):
            _check_workspace(path)
            raw = _load_json(path / "desktop-sync-state.json", None,
                             maximum=MAX_SYNC_STATE_BYTES)
            if raw is None:
                return []
            try:
                saved = SyncState.model_validate(raw)
            except ValueError as exc:
                raise DesktopError("CORRUPT_LOCAL_STATE", "The project record could not be read.") from exc
            if editing and (saved.role == "viewer" or saved.access_denied):
                raise DesktopError("ROLE_REQUIRED", "You do not have permission to edit this project.", 403)
            return [entry.model_dump() for entry in saved.script_catalog]

        @app.get("/api/desktop/file-layout")
        def get_file_layout():
            with self.lock:
                return store.get_file_layout(known_scripts=layout_catalog())

        @app.put("/api/desktop/file-layout")
        def put_file_layout(body: FileLayoutRequest):
            with self.lock:
                return store.put_file_layout(body.model_dump(),
                                             known_scripts=layout_catalog(editing=True))

        def read_layout_cache():
            _check_workspace(path)
            raw = _load_json(path / "desktop-file-layout-sync.json", None,
                             maximum=2 * file_layout.MAX_BYTES)
            if raw is None:
                return None
            try:
                return FileLayoutSyncCache.model_validate(raw).model_dump()
            except ValueError as exc:
                raise DesktopError("CORRUPT_LOCAL_STATE", "The file layout record could not be read.") from exc

        @app.get("/api/desktop/file-layout-sync")
        def get_layout_sync_cache():
            with self.lock:
                return read_layout_cache()

        @app.put("/api/desktop/file-layout-sync")
        def put_layout_sync_cache(body: FileLayoutSyncCache):
            with self.lock:
                current = read_layout_cache()
                expected = current["version"] if current else 0
                if body.version != expected:
                    raise DesktopError("VERSION_CONFLICT", "The file layout record has changed. Open it again.", 409)
                if expected >= file_layout.MAX_VERSION - 1:
                    raise DesktopError("VERSION_CONFLICT", "The file layout version limit has been reached.", 409)
                saved = {**body.model_dump(), "version": expected + 1}
                if len(json.dumps(saved, ensure_ascii=False).encode("utf-8")) > file_layout.MAX_BYTES:
                    raise DesktopError("INVALID_FILE_LAYOUT", "The file layout is too large.")
                _atomic_json(path / "desktop-file-layout-sync.json", saved)
                return saved

        @app.post("/api/desktop-console/execute")
        def execute_and_queue(body: DesktopConsoleRequest):
            with self.lock:
                require_editing()
                if body.script_id is not None:
                    document = store.named_console_script(body.script_id)
                    name = file_layout.source_name(store.get_file_layout(), body.script_id, document["name"])
                    try:
                        script_contracts.require_python(name)
                    except script_contracts.ScriptValidationError as exc:
                        raise DataError(str(exc), exc.code) from exc
                references = [item.model_dump() for item in body.input_files]
                local_only_before = any(item.get("local_only") is True for item in store.list_datasets())

            def metadata(computed_record):
                with self.lock:
                    try:
                        _check_workspace(path)
                        local_only = local_only_before or any(item.get("local_only") is True
                                                             for item in store.list_datasets())
                    except (DesktopError, DataError, OSError, ValueError):
                        local_only = True
                    context = {"actor_uid": body.actor_uid, "input_files": references, "local_only": local_only}
                    summaries = network_summaries(computed_record)
                    if summaries:
                        context["network_summaries"] = summaries
                    return {"actor_uid": body.actor_uid, "local_only": local_only, "sharing_context": context}

            record = app.state.console.execute(body.code, timeout_seconds=body.timeout_seconds,
                                               record_metadata=metadata)
            with self.lock:
                try:
                    _check_workspace(path)
                    state = sharing.after_execute(record)
                except (DesktopError, OSError) as exc:
                    # The immutable context was already saved in console history.
                    # Even a broken queue/status file must not erase computation.
                    state = {"id": record["id"], "actor_uid": body.actor_uid,
                             "state": "failed", "error": sharing.failure(exc), "retryable": True}
            return {**record, "sharing": state}

        @app.get("/api/desktop-sharing")
        def get_sharing():
            with self.lock:
                _check_workspace(path)
                return sharing.snapshot()

        @app.put("/api/desktop-sharing/{record_id}")
        def update_sharing(record_id: str, body: SharingUpdate):
            with self.lock:
                if body.state == "pending":
                    require_editing()
                else:
                    _check_workspace(path)
                return sharing.update(record_id, body)

        @app.post("/api/desktop-sharing/{record_id}/retry")
        def retry_sharing(record_id: str, body: RetryRequest):
            with self.lock:
                require_editing()
                return sharing.retry(record_id, body.actor_uid)

        @app.get("/api/desktop-outbox")
        def get_outbox():
            with self.lock:
                _check_workspace(path)
                return sharing.outbox()

        @app.put("/api/desktop-outbox")
        def put_outbox(body: OutboxItem):
            with self.lock:
                require_editing()
                return sharing.enqueue(body.model_dump())

        @app.delete("/api/desktop-outbox/{record_id}")
        def acknowledge_outbox(record_id: str, actor_uid: str | None = None):
            with self.lock:
                _check_workspace(path)
                return sharing.acknowledge(record_id, actor_uid)

        @app.get("/api/desktop-sync-state")
        def get_sync_state():
            with self.lock:
                state = _load_json(path / "desktop-sync-state.json", None, maximum=MAX_SYNC_STATE_BYTES)
                if state is not None:
                    try:
                        state = SyncState.model_validate(state).model_dump(exclude_unset=True)
                    except ValueError as exc:
                        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved synchronization state is invalid.") from exc
                return state

        @app.put("/api/desktop-sync-state")
        def put_sync_state(body: SyncState):
            with self.lock:
                _check_workspace(path)
                state = body.model_dump(exclude_unset=True)
                _atomic_json(path / "desktop-sync-state.json", state)
                return state

        @app.put("/api/desktop-scripts/{script_id}")
        def cache_script(script_id: str, body: CachedScript):
            with self.lock:
                _check_workspace(path)
                return store.cache_console_script(script_id, body.name, body.code, body.version)

        @app.get("/api/desktop-cached-files")
        def cached_files():
            with self.lock:
                _check_workspace(path)
                index = _load_json(path / "desktop-cache-index.json", {})
                if not isinstance(index, dict) or len(index) > 1000:
                    raise DesktopError("CORRUPT_LOCAL_STATE", "The saved file cache is invalid.")
                files = []
                for cloud_id, entry in index.items():
                    try:
                        if not isinstance(entry, dict) or set(entry) != {"dataset_id", "sha256", "name"}:
                            raise ValueError("Invalid cache entry.")
                        body = CachedFile.model_validate({"cloud_id": cloud_id, "name": entry["name"],
                                                         "expected_sha256": entry["sha256"]})
                        name = _cached_name(body.name)
                        dataset_id = entry["dataset_id"]
                        if not isinstance(dataset_id, str) or str(UUID(dataset_id)) != dataset_id:
                            raise ValueError("Invalid dataset identifier.")
                    except (ValueError, TypeError, AttributeError) as exc:
                        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved file cache is invalid.") from exc
                    files.append({"cloud_id": body.cloud_id, "sha256": body.expected_sha256,
                                  "name": name, "dataset_id": dataset_id})
                return {"files": files, "local_only": [profile["id"] for profile in store.list_datasets()
                                                        if profile.get("local_only") is True]}

        @app.post("/api/datasets/import-cached")
        def import_cached(body: CachedFile):
            name = _cached_name(body.name)
            with self.lock:
                _check_workspace(path)
                source = path / name
                if not source.is_file() or source.is_symlink() or source.resolve().parent != path:
                    raise DesktopError("NOT_FOUND", "The cached project file was not found.", 404)
                if source.suffix.lower() not in {".csv", ".parquet"} and source.stat().st_size > MAX_FILE_BYTES:
                    raise DataError("The cached dataset exceeds the local file limit.", "DATA_LIMIT")
                digest = hashlib.sha256()
                with source.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if not secrets.compare_digest(digest.hexdigest(), body.expected_sha256):
                    raise DesktopError("DATA_INTEGRITY", "The cached file does not match its cloud checksum.")
                index_path = path / "desktop-cache-index.json"
                index = _load_json(index_path, {})
                if not isinstance(index, dict):
                    raise DesktopError("CORRUPT_LOCAL_STATE", "The saved file cache is invalid.")
                previous = index.get(body.cloud_id)
                if isinstance(previous, dict) and previous.get("sha256") == body.expected_sha256:
                    profile = store.get_dataset(previous.get("dataset_id", ""))
                    store.load_frame(profile["id"])
                else:
                    profile = store.import_file(source, name=name)
                    index[body.cloud_id] = {"dataset_id": profile["id"], "sha256": body.expected_sha256,
                                           "name": name}
                    _atomic_json(index_path, index)
                return {**profile, "cloud_id": body.cloud_id, "python_path": name,
                        "sha256": body.expected_sha256}

        @app.exception_handler(DesktopError)
        async def project_error(request, exc):
            return _error_response(exc)

        @app.exception_handler(RequestValidationError)
        async def invalid_project_request(request, exc):
            # Echoing raw invalid strings can itself fail UTF-8 serialization;
            # submitted input also need not appear in the validation response.
            fields = [{"type": str(error.get("type", "invalid")).encode("utf-8", "replace").decode("utf-8"),
                       "loc": [str(part).encode("utf-8", "replace").decode("utf-8")
                               for part in error.get("loc", [])],
                       "msg": str(error.get("msg", "Invalid input.")).encode("utf-8", "replace").decode("utf-8")}
                      for error in exc.errors()]
            return JSONResponse({"detail": fields}, status_code=422)

    def status(self) -> dict:
        with self.lock:
            console = self.active_app.state.console if self.active_app is not None else None
            return {"project_id": self.active_project, "persistent": True,
                    "execution_mode": "persistent", "console": console.status() if console else None}

    def deactivate(self):
        with self.lock:
            if self.active_app is not None:
                self.active_app.state.packages.close()
                self.active_app.state.console.close()
            self.active_project = self.active_app = None

    def close(self):
        with self.lock:
            self.closed = True
            self.deactivate()


def _error_response(exc: DesktopError):
    return JSONResponse({"detail": {"code": exc.code, "message": str(exc)}}, status_code=exc.status_code)


class _ProjectRouter:
    def __init__(self, projects: DesktopProjects):
        self.projects = projects

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        # Starlette retains the original path for mounts and exposes root_path.
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path):]
        match = _PROJECT_ROUTE.fullmatch(path)
        if not match or any(part in {".", ".."} for part in path.split("/")) or "\\" in path:
            await _error_response(DesktopError("INVALID_PROJECT", "The local project route is invalid.", 404))(scope, receive, send)
            return
        project_id, remainder = match.groups()
        remainder = remainder or "/"
        try:
            if remainder == "/session" and scope.get("method") == "GET":
                app = await run_in_threadpool(self.projects.activate, project_id)
            else:
                app = await run_in_threadpool(self.projects.current, project_id)
        except DesktopError as exc:
            await _error_response(exc)(scope, receive, send)
            return
        forwarded = dict(scope)
        forwarded.update(path="/api" + remainder, raw_path=("/api" + remainder).encode("utf-8"), root_path="")
        await app(forwarded, receive, send)


def create_desktop_app(data_root: str | Path, *, auth_config: dict | None = None, qa_metrics: bool = False) -> FastAPI:
    projects = DesktopProjects(data_root)
    # Existing local endpoints remain useful for development and backward
    # compatibility. Their workspace and token are separate from all projects.
    local = create_app(projects.root / "local")
    local.state.console.worker_environment = worker_environment()
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            projects.close()
            local.state.console.close()

    app = FastAPI(title="OpenEconometrics Desktop", version=__version__, docs_url=None,
                  redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.desktop_projects, app.state.token = projects, token
    @app.exception_handler(DesktopError)
    async def desktop_error(_request, error):
        return _error_response(error)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def desktop_access(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            try:
                valid = (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                         and parsed.port in {request.url.port or 80, 5173})
            except ValueError:
                valid = False
            if not valid:
                return _error_response(DesktopError("ORIGIN_DENIED", "Only this OpenEconometrics interface can access the local runtime.", 403))
        if request.headers.get("sec-fetch-site") == "cross-site":
            return _error_response(DesktopError("ORIGIN_DENIED", "Cross-site requests are not allowed.", 403))
        if request.headers.get("authorization"):
            return _error_response(DesktopError("CLOUD_TOKEN_DENIED", "Cloud credentials must not be sent to the Python runtime.", 400))
        if (request.url.path in {"/api/desktop/status", "/api/desktop/close"}
                or request.url.path.startswith(("/api/desktop/local-projects", "/api/desktop/qa-performance"))) and not secrets.compare_digest(
            request.headers.get("x-openecon-token", ""), token
        ):
            return _error_response(DesktopError("TOKEN_REQUIRED", "Establish a local desktop session.", 401))
        response = await call_next(request)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                                 "X-Frame-Options": "DENY", "Cache-Control": "no-store"})
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; font-src 'self'; "
            "connect-src 'self' ipc: http://ipc.localhost https://*.googleapis.com https://*.firebaseapp.com; "
            "frame-src https://*.firebaseapp.com https://accounts.google.com; "
            "object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        )
        return response

    if qa_metrics:
        from openecon.qa_performance import attach
        attach(app, projects.root)

    @app.get("/api/auth/config")
    def config():
        return dict(auth_config) if auth_config is not None else {"mode": "desktop"}

    @app.get("/api/desktop/session")
    def session():
        return {"token": token, "version": __version__, "environment": "desktop", "persistent": True,
                "execution_mode": "persistent", "upload_limit_bytes": None,
                "eager_format_limit_bytes": MAX_FILE_BYTES}

    @app.get("/api/desktop/status")
    def status():
        return projects.status()

    def local_catalog():
        value = _load_json(projects.root / "local-projects.json", {"projects": []}, maximum=512 * 1024)
        if not isinstance(value, dict) or set(value) != {"projects"} or not isinstance(value["projects"], list) or len(value["projects"]) > 200:
            raise DesktopError("CORRUPT_LOCAL_STATE", "The local project list could not be read.")
        seen = set()
        for row in value["projects"]:
            if not isinstance(row, dict) or set(row) != {"id", "name", "description"} or not _PROJECT_ID.fullmatch(str(row["id"])) or row["id"] in seen:
                raise DesktopError("CORRUPT_LOCAL_STATE", "The local project list could not be read.")
            try:
                LocalProjectInput.model_validate({"name": row["name"], "description": row["description"]})
            except ValueError as exc:
                raise DesktopError("CORRUPT_LOCAL_STATE", "The local project list could not be read.") from exc
            seen.add(row["id"])
        return value

    @app.get("/api/desktop/local-projects")
    def list_local_projects():
        with projects.lock:
            return local_catalog()

    @app.post("/api/desktop/local-projects")
    def create_local_project(body: LocalProjectInput):
        with projects.lock:
            catalog = local_catalog()
            if len(catalog["projects"]) >= 200:
                raise DesktopError("PROJECT_LIMIT", "The local project limit has been reached.", 409)
            identifier = uuid4().hex
            if (projects.projects_path / identifier).exists():
                raise DesktopError("PROJECT_EXISTS", "Create the project again.", 409)
            project = {"id": identifier, **body.model_dump()}
            projects.project_path(identifier)
            _atomic_json(projects.root / "local-projects.json", {"projects": [*catalog["projects"], project]})
            return project

    @app.post("/api/desktop/local-projects/{identifier}/open")
    def open_local_project(identifier: str):
        with projects.lock:
            project = next((row for row in local_catalog()["projects"] if row["id"] == identifier), None)
            if project is None:
                raise DesktopError("LOCAL_PROJECT_NOT_FOUND", "This is not a local project.", 404)
            path = projects.project_path(identifier)
            if (path / "desktop-sync-state.json").exists():
                raise DesktopError("PROJECT_KIND_CONFLICT", "A team project cannot be opened as a local project.", 403)
            projects.activate(identifier)
            return project

    @app.post("/api/desktop/close")
    def close():
        projects.deactivate()
        return {"status": "closed"}

    app.mount("/api/desktop/projects", _ProjectRouter(projects), name="desktop-projects")
    app.mount("/", local)
    return app
