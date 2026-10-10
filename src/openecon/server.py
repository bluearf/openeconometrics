"""Local loopback workspace API for the desktop app and `openecon serve`.

Analyses always run on the user's machine; there is no cloud execution mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from contextlib import asynccontextmanager
import secrets
import tempfile
from urllib.parse import urlparse

from openecon.optional_dependencies import require_extra
try:
    from fastapi import FastAPI, File, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError:
    require_extra("server", "fastapi", "uvicorn", "multipart")
    raise
from importlib.resources import files
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.background import BackgroundTask

from openecon import __version__
from openecon.console import ConsoleError, ConsoleSession
from openecon.analysis_contracts import AnalysisError, capabilities
from openecon.data import DataError, SUPPORTED
from openecon.models import ModelSpec
from openecon.workspace import Workspace
from openecon import file_layout, script_contracts


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    spec: ModelSpec


class PortableEnvironmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: dict


class ConsoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1, max_length=64000)
    timeout_seconds: float | None = Field(default=None, ge=.05, allow_inf_nan=False)
    script_id: str | None = Field(default=None, pattern=r"^(?:analysis|[0-9a-f]{32})$")


class ScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(max_length=64000)
    name: str = "analysis.py"


class ScriptCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(default="untitled.py", max_length=180)
    code: str = Field(default="", max_length=64000)
    id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class NamedScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str = Field(max_length=64000)
    version: int = Field(ge=0, lt=2**63 - 1)


class FileLayoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = Field(ge=0, lt=2**63 - 1)
    entries: list[dict] = Field(max_length=2000)


class PackageInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=100)
    version: str | None = Field(default=None, min_length=1, max_length=100)


class PackageRemoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=100)


class PackageRestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    manifest: dict


def create_app(workspace: str | Path = ".openecon", *, project_packages: bool = False) -> FastAPI:
    require_extra("server", "fastapi", "uvicorn", "multipart")
    store = Workspace(workspace)
    console = ConsoleSession(store, default_timeout_seconds=None, max_timeout_seconds=None)
    packages = None
    if project_packages:
        from openecon.project_packages import PackageError, ProjectPackages
        packages = ProjectPackages(store.path)
        console.packages = packages

    @asynccontextmanager
    async def lifespan(app):
        yield
        console.close()
        if packages is not None:
            packages.close()

    app = FastAPI(
        title="OpenEconometrics", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    token = secrets.token_urlsafe(32)
    app.state.workspace = store
    app.state.console = console
    app.state.packages = packages
    app.state.token = token
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin:
            origin_url = urlparse(origin)
            valid_hosts = {"127.0.0.1", "localhost", "::1"}
            valid_ports = {request.url.port or 80, 5173}
            valid_origin = not (
                origin_url.scheme != "http"
                or origin_url.hostname not in valid_hosts
                or origin_url.port not in valid_ports
            )
            if not valid_origin:
                return JSONResponse(
                    {
                        "detail": {
                            "code": "ORIGIN_DENIED",
                            "message": "Only this OpenEconometrics interface can access this API.",
                        }
                    },
                    status_code=403,
                )
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse(
                {
                    "detail": {
                        "code": "ORIGIN_DENIED",
                        "message": "Cross-site requests are not allowed.",
                    }
                },
                status_code=403,
            )
        if request.url.path.startswith("/api/") and request.url.path not in {"/api/session", "/api/auth/config"}:
            if not secrets.compare_digest(request.headers.get("x-openecon-token", ""), token):
                return JSONResponse(
                    {
                        "detail": {
                            "code": "TOKEN_REQUIRED",
                            "message": "Refresh the app to establish a session.",
                        }
                    },
                    status_code=401,
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(DataError)
    @app.exception_handler(AnalysisError)
    async def expected_error(request, exc):
        return JSONResponse(
            {"detail": {"code": exc.code, "message": str(exc)}},
            status_code=(404 if exc.code == "NOT_FOUND" else
                         409 if exc.code in {"VERSION_CONFLICT", "SCRIPT_ID_CONFLICT", "SCRIPT_NAME_CONFLICT", "FILE_INVENTORY_CONFLICT"}
                         else 429 if exc.code == "SCRIPT_LIMIT" else 422),
        )

    @app.exception_handler(ConsoleError)
    async def console_error(request, exc):
        return JSONResponse({"detail": {"code": exc.code, "message": str(exc)}},
                            status_code=409 if exc.code == "CONSOLE_BUSY" else 422)

    if packages is not None:
        @app.exception_handler(PackageError)
        async def package_error(request, exc):
            return JSONResponse({"detail": {"code": exc.code, "message": str(exc)}},
                                status_code=409 if exc.code in {"ENVIRONMENT_BUSY", "PACKAGES_BUSY"} else 422)

        @app.get("/api/environment")
        def project_environment():
            return packages.snapshot()

        @app.get("/api/environment/export")
        def package_export():
            return packages.export_portable()

        @app.post("/api/environment/import/preview")
        def package_import_preview(body: PortableEnvironmentRequest):
            return packages.preview_portable(body.document)

        @app.post("/api/environment/import/restore")
        def package_import_restore(body: PortableEnvironmentRequest):
            return console.change_environment(lambda: packages.restore_portable(body.document))

        @app.post("/api/environment/install")
        def package_install(body: PackageInstallRequest):
            return console.change_environment(lambda: packages.start_install(body.name, body.version))

        @app.post("/api/environment/remove")
        def package_remove(body: PackageRemoveRequest):
            return console.change_environment(lambda: packages.start_remove(body.name))

        @app.post("/api/environment/restore")
        def package_restore(body: PackageRestoreRequest):
            return console.change_environment(lambda: packages.start_restore(body.manifest))

        @app.post("/api/environment/cancel")
        def package_cancel():
            return packages.cancel()

    @app.get("/api/console")
    def console_state():
        return console.snapshot()

    @app.get("/api/console/plots/{plot_id}")
    def console_plot(plot_id: str):
        from openecon.plot_artifacts import checked_plot_path, history_reference
        reference = history_reference(store.console_history(), plot_id)
        return FileResponse(checked_plot_path(store.path, reference), media_type="application/json")

    @app.post("/api/console/execute")
    def console_execute(body: ConsoleRequest):
        if body.script_id is not None:
            document = store.named_console_script(body.script_id)
            name = file_layout.source_name(store.get_file_layout(), body.script_id, document["name"])
            try:
                script_contracts.require_python(name)
            except script_contracts.ScriptValidationError as exc:
                raise DataError(str(exc), exc.code) from exc
        return console.execute(body.code, timeout_seconds=body.timeout_seconds)

    @app.post("/api/console/interrupt")
    def console_interrupt():
        return console.interrupt()

    @app.post("/api/console/reset")
    def console_reset():
        return console.reset()

    @app.get("/api/console/script")
    def console_script():
        return store.console_script()

    @app.put("/api/console/script")
    def save_console_script(body: ScriptRequest):
        return store.save_console_script(body.code, body.name)

    @app.get("/api/files/layout")
    def get_file_layout():
        return store.get_file_layout()

    @app.put("/api/files/layout")
    def put_file_layout(body: FileLayoutRequest):
        return store.put_file_layout(body.model_dump())

    @app.get("/api/console/scripts")
    def console_scripts():
        return store.console_scripts()

    @app.post("/api/console/scripts", status_code=201)
    def create_console_script(body: ScriptCreateRequest):
        return store.create_console_script(body.name, body.code, script_id=body.id)

    @app.get("/api/console/scripts/{script_id}")
    def named_console_script(script_id: str):
        return store.named_console_script(script_id)

    @app.put("/api/console/scripts/{script_id}")
    def save_named_console_script(script_id: str, body: NamedScriptRequest):
        return store.save_named_console_script(script_id, body.code, body.version)

    @app.get("/api/session")
    def session():
        return {"token": token, "version": __version__,
                "environment": "local", "persistent": True, "upload_limit_bytes": None}

    @app.get('/api/auth/config')
    def auth_config():
        # Team sign-in is served by the cloud sync backend; this server is local.
        return {'mode': 'local'}

    @app.get("/api/config")
    def config():
        from openecon.mcp_launcher import connection_config
        return {"version": __version__, **connection_config(store.path)}

    @app.get("/api/capabilities")
    def get_capabilities():
        return capabilities()

    @app.get("/api/datasets")
    def datasets():
        return {"datasets": store.list_datasets()}

    @app.post("/api/datasets/example")
    def example():
        return store.create_example()

    @app.post("/api/datasets/upload")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "upload").name[:200]
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise DataError("Supported formats: CSV, Parquet, XLSX and DTA.", "UNSUPPORTED_FORMAT")
        fd, temp = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    stream.write(chunk)
            from starlette.concurrency import run_in_threadpool

            return await run_in_threadpool(store.import_file, temp, name=name, local_only=True)
        finally:
            Path(temp).unlink(missing_ok=True)
            await file.close()

    @app.get("/api/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        return store.get_dataset(dataset_id)

    @app.post("/api/analyses")
    def analyze(body: AnalysisRequest):
        return store.run_analysis(body.dataset_id, body.spec)

    @app.get("/api/results")
    def results():
        return {"results": store.list_results()}

    @app.get("/api/results/revision")
    def result_revision():
        return {"revision": store.agent_results_revision()}

    @app.get("/api/results/{result_id}")
    def result(result_id: str):
        return store.get_result(result_id)

    @app.get("/api/results/{result_id}/python", response_class=PlainTextResponse)
    def python(result_id: str):
        return PlainTextResponse(
            store.python_script(result_id),
            headers={"Content-Disposition": 'attachment; filename="analysis.py"'},
        )

    @app.get("/api/results/{result_id}/bundle")
    def bundle(result_id: str):
        fd, path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        try:
            store.export_bundle(result_id, destination=path)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        return FileResponse(path, media_type="application/zip", filename="openecon-analysis.zip",
                            background=BackgroundTask(Path(path).unlink, missing_ok=True))

    app.mount(
        "/chart-assets",
        StaticFiles(directory=str(files("openecon_charts").joinpath("assets"))),
        name="chart-assets",
    )
    static = Path(__file__).parent / "static"
    if (static / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/")
    def index():
        if not (static / "index.html").is_file():
            return PlainTextResponse(
                "Build the web app first: cd web && npm ci && npm run build", status_code=503
            )
        return FileResponse(static / "index.html")

    return app
