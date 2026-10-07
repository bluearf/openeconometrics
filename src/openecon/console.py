"""Lifecycle and durable history for an explicitly executed local Python session.

Code has the permissions of the local user. The child process allows interruption
and recovery; it is deliberately not presented as a security sandbox.
"""
from __future__ import annotations

import atexit
import math
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import signal
import shutil
import tempfile
import threading
import time
from uuid import uuid4
from typing import Callable

from openecon.console_worker import (
    MAX_COMMAND_BYTES, WorkerProtocolError, receive_json, send_json, validate_install_request,
    validate_ready, validate_result,
)


class ConsoleError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def load_dataset(dataset_id: str, *, workspace: str | Path | None = None):
    """Load an immutable imported dataset from the active local workspace."""
    from openecon.workspace import Workspace
    location = workspace or os.environ.get("OPENECON_WORKSPACE", ".openecon")
    return Workspace(location).load_frame(dataset_id)


class ConsoleSession:
    def __init__(self, workspace, *, startup_timeout: float = 30,
                 worker_environment: dict[str, str] | None = None,
                 default_timeout_seconds: float | None = 60,
                 max_timeout_seconds: float | None = 120):
        self.workspace = workspace
        self.startup_timeout = startup_timeout
        self.worker_environment = dict(worker_environment) if worker_environment is not None else None
        self.default_timeout_seconds = default_timeout_seconds
        self.max_timeout_seconds = max_timeout_seconds
        self._process = None
        self._connection = None
        self._group_pid = None
        self._scratch_directory = None
        self._run_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._running = False
        self._current_id = None
        self._interrupted = threading.Event()
        self._variables = []
        self._closed = False
        self.packages = None
        self._script_package_job = None
        history = workspace.console_history()
        self._generation = max((item.get("session_generation", 0) for item in history), default=0) + 1
        atexit.register(self.close)

    def snapshot(self) -> dict:
        from openecon.output_latex import enrich_record
        with self._state_lock:
            return {"history": [enrich_record(item) for item in self.workspace.display_history()],
                    "status": self.status(), "variables": list(self._variables)}

    def status(self) -> dict:
        """Inspect the live worker without loading saved execution outputs."""
        with self._state_lock:
            return {"running": self._running, "session_generation": self._generation,
                    "pid": self._process.pid if self._process and self._process.is_alive() else None,
                    "execution_id": self._current_id}

    def _start(self):
        from openecon.console_worker import worker_main
        with self._state_lock:
            if self._closed:
                raise ConsoleError("CONSOLE_CLOSED", "The Python session is closed.")
            if self._process is not None and self._process.is_alive():
                return
            self._dispose(increment=False)
            try:
                overlay = self.packages.active_path() if self.packages is not None else None
            except ValueError as exc:
                raise ConsoleError("INVALID_ENVIRONMENT", str(exc)) from exc
            parent, child = multiprocessing.get_context("spawn").Pipe()
            try:
                self._scratch_directory = Path(tempfile.mkdtemp(
                    prefix=".openecon-scratch-", dir=self.workspace.path))
                arguments = (child, str(self.workspace.path), self.worker_environment,
                             str(overlay) if overlay is not None else None, str(self._scratch_directory))
                process = multiprocessing.get_context("spawn").Process(
                    target=worker_main, args=arguments, name="OpenEconometrics Python")
                process.start()
            except BaseException:
                parent.close()
                child.close()
                self._dispose(increment=False)
                raise
            worker_pid = process.pid
            child.close()
            self._connection, self._process = parent, process
        try:
            if not parent.poll(self.startup_timeout):
                raise ConsoleError("WORKER_START_FAILED", "Python startup timed out; the next run will start a fresh session.")
            ready = validate_ready(receive_json(parent))
            if not ready.get("ready"):
                raise ConsoleError("WORKER_START_FAILED", ready.get("error", "Python could not start."))
            with self._state_lock:
                if self._interrupted.is_set() or self._process is not process:
                    raise ConsoleError("INTERRUPTED", "Python was interrupted during startup.")
                if ready.get("process_group") == worker_pid:
                    self._group_pid = worker_pid
        except (EOFError, OSError) as exc:
            self._dispose()
            raise ConsoleError("WORKER_START_FAILED", "Python exited during startup; check the local installation.") from exc
        except WorkerProtocolError as exc:
            self._dispose()
            raise ConsoleError("WORKER_PROTOCOL", "Python returned an invalid startup message; the session was stopped.") from exc
        except ConsoleError:
            self._dispose()
            raise

    def _dispose(self, *, increment: bool = True):
        with self._state_lock:
            process, connection = self._process, self._connection
            group_pid = self._group_pid
            scratch = self._scratch_directory
            self._scratch_directory = None
            self._process = self._connection = self._group_pid = None
            if group_pid is not None and os.name == "posix":
                try:
                    os.killpg(group_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # macOS can return EPERM for a group whose leader just
                    # exited. Reclaim its scratch even in that race; if the
                    # owned child is still alive, stop it directly.
                    if process is not None and process.is_alive():
                        process.kill()
            if process is not None:
                if process.is_alive() and group_pid is None:
                    if os.name == "posix":
                        try:
                            # The worker establishes a separate process group.
                            if os.getpgid(process.pid) == process.pid:
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                process.join(timeout=2)
                if not process.is_alive():
                    process.close()
            if connection is not None:
                connection.close()
            if scratch is not None:
                # Only this supervisor-created nonce directory is removed.
                # This also reclaims spill files after an abrupt worker kill.
                shutil.rmtree(scratch, ignore_errors=True)
            self._variables = []
            if self.packages is not None:
                prune = getattr(self.packages, "prune_generations", None)
                if prune is not None:
                    prune()
            if increment:
                self._generation += 1

    def _install_from_script(self, request: dict, deadline: float | None) -> dict:
        """Service one worker request while retaining the execution lock."""
        from openecon.project_packages import PackageError
        reply = {"kind": "package_result", "id": request["id"],
                 "request_id": request["request_id"], "ok": False,
                 "path": None, "installed": [], "error": None}
        try:
            if self.packages is None:
                raise PackageError("PROJECT_INSTALL_UNAVAILABLE", "Run package installation in an OpenEconometrics desktop project.")
            previous_job = (self.packages.snapshot().get("job") or {}).get("id")
            snapshot = self.packages.start_install_many(
                request["requirements"], loaded_versions=request["loaded_versions"],
                preserve_previous=True,
                **{name: request[name] for name in ("installer", "specifications", "upgrade", "source_options") if name in request},
            )
            job = snapshot.get("job") or {}
            if job.get("id") != previous_job:
                self._script_package_job = job.get("id")
                while job.get("state") == "running":
                    if self._interrupted.is_set():
                        raise ConsoleError("INTERRUPTED", "Python was interrupted; session variables were cleared.")
                    with self._state_lock:
                        if self._process is None or not self._process.is_alive():
                            raise ConsoleError("WORKER_EXITED", "Python exited during package installation; the installer was stopped.")
                    if deadline is not None and time.monotonic() >= deadline:
                        raise ConsoleError("TIMEOUT", "Package installation exceeded the execution time limit; Python was stopped.")
                    self._interrupted.wait(.05)
                    snapshot = self.packages.snapshot()
                    job = snapshot.get("job") or {}
                if job.get("state") != "complete":
                    raise PackageError(job.get("code", "PACKAGE_INSTALL_FAILED"),
                                       job.get("message", "Package installation did not complete."))
            if self._interrupted.is_set():
                raise ConsoleError("INTERRUPTED", "Python was interrupted; session variables were cleared.")
            versions = {row["name"]: row["version"]
                        for row in [*snapshot["base"], *snapshot["installed"]]}
            path = self.packages.active_path()
            reply.update(ok=True, path=str(path) if path is not None else None,
                         installed=[{"name": row["name"], "version": versions[row["name"]]}
                                    for row in request["requirements"]])
        except PackageError as exc:
            reply["error"] = {"code": exc.code, "message": str(exc)[:4000]}
        except ConsoleError:
            self._cancel_script_install()
            self._dispose()
            raise
        finally:
            self._script_package_job = None
        return reply

    def _cancel_script_install(self):
        if self.packages is not None and self._script_package_job is not None:
            job = self.packages.snapshot().get("job") or {}
            if job.get("id") == self._script_package_job and job.get("state") == "running":
                self.packages.cancel()

    def execute(self, code: str, *, timeout_seconds: float | None = None,
                record_metadata: Callable[[dict], dict] | None = None) -> dict:
        if not isinstance(code, str) or not code.strip() or len(code) > 64000:
            raise ConsoleError("INVALID_CODE", "Enter between 1 and 64,000 characters of Python code.")
        timeout_seconds = self.default_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout_seconds is None:
            if self.max_timeout_seconds is not None:
                raise ConsoleError("INVALID_TIMEOUT", "Unlimited execution is only available in the local desktop runtime.")
        elif (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
              or not math.isfinite(timeout_seconds) or timeout_seconds < .05
              or (self.max_timeout_seconds is not None and timeout_seconds > self.max_timeout_seconds)):
            raise ConsoleError("INVALID_TIMEOUT", "Execution timeout is outside this runtime's permitted range.")
        if not self._run_lock.acquire(blocking=False):
            raise ConsoleError("CONSOLE_BUSY", "Python is already running. Wait or interrupt the current execution.")
        execution_id = str(uuid4())
        start = time.monotonic()
        created = datetime.now(timezone.utc).isoformat()
        self._interrupted.clear()
        with self._state_lock:
            self._running = True
            self._current_id = execution_id
        result = None
        try:
            if self.packages is not None and (self.packages.snapshot().get("job") or {}).get("state") == "running":
                raise ConsoleError("ENVIRONMENT_BUSY", "Wait for the package installation to finish or cancel it.")
            self._start()
            with self._state_lock:
                connection = self._connection
                run_generation = self._generation
                if self._interrupted.is_set() or connection is None:
                    raise ConsoleError("INTERRUPTED", "Python was interrupted; session variables were cleared.")
                send_json(connection, {"op": "execute", "code": code, "id": execution_id},
                          maximum=MAX_COMMAND_BYTES)
            deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
            while True:
                if self._interrupted.is_set():
                    raise ConsoleError("INTERRUPTED", "Python was interrupted; session variables were cleared.")
                if deadline is not None and time.monotonic() >= deadline:
                    self._dispose()
                    raise ConsoleError("TIMEOUT", "Python exceeded the time limit. The worker was stopped and session variables were cleared.")
                try:
                    wait_seconds = min(.05, max(0, deadline - time.monotonic())) if deadline is not None else .05
                    if connection.poll(wait_seconds):
                        message = receive_json(connection)
                        if message.get("kind") == "package_install":
                            request = validate_install_request(message, execution_id)
                            reply = self._install_from_script(request, deadline)
                            send_json(connection, reply, maximum=MAX_COMMAND_BYTES)
                            continue
                        result = validate_result(message, execution_id)
                        break
                except (EOFError, OSError) as exc:
                    if self._interrupted.is_set():
                        raise ConsoleError("INTERRUPTED", "Python was interrupted; session variables were cleared.") from exc
                    self._dispose()
                    raise ConsoleError("WORKER_EXITED", "Python exited unexpectedly. Session variables were cleared; run the setup code again.") from exc
            with self._state_lock:
                if self._interrupted.is_set():
                    raise ConsoleError("INTERRUPTED", "Python was interrupted; session variables were cleared.")
                self._variables = result.get("variables", [])
        except WorkerProtocolError:
            self._dispose()
            result = {"status": "error", "stdout": "", "outputs": [], "variables": [],
                      "error": {"type": "WORKER_PROTOCOL", "message": "Python returned an invalid message; the session was stopped.", "traceback": ""},
                      "state_reset": True}
            run_generation = locals().get("run_generation", self._generation)
        except (OSError, EOFError, RuntimeError) as exc:
            self._dispose()
            result = {"status": "error", "stdout": "", "outputs": [], "variables": [],
                      "error": {"type": "WORKER_EXITED", "message": f"Python could not continue: {str(exc)[:1000]}", "traceback": ""},
                      "state_reset": True}
            run_generation = locals().get("run_generation", self._generation)
        except ConsoleError as exc:
            failure = (ConsoleError("INTERRUPTED", "Python was interrupted; session variables were cleared.")
                       if self._interrupted.is_set() else exc)
            status = {"TIMEOUT": "timeout", "INTERRUPTED": "interrupted"}.get(failure.code, "error")
            result = {"status": status, "stdout": "", "outputs": [], "variables": [],
                      "error": {"type": failure.code, "message": str(failure), "traceback": ""}, "state_reset": True}
            run_generation = locals().get("run_generation", self._generation)
        finally:
            with self._state_lock:
                generation = self._generation
                self._running = False
                self._current_id = None
            if result is not None:
                result.update(id=execution_id, code=code, created_at=created,
                              duration_ms=round((time.monotonic() - start) * 1000, 2),
                              session_generation=run_generation, active_session_generation=generation)
                try:
                    if record_metadata is not None:
                        result.update(record_metadata(result))
                    self.workspace.append_console_history(result)
                finally:
                    self._run_lock.release()
            else:
                self._run_lock.release()
        return result

    def change_environment(self, operation):
        """Serialize an explicit package change with code execution."""
        if not self._run_lock.acquire(blocking=False):
            raise ConsoleError("CONSOLE_BUSY", "Stop the running code before changing packages.")
        try:
            if self._closed:
                raise ConsoleError("CONSOLE_CLOSED", "The Python session is closed.")
            result = operation()
            self._dispose()
            return result
        finally:
            self._run_lock.release()

    def interrupt(self) -> dict:
        if self._running:
            self._interrupted.set()
            self._cancel_script_install()
        with self._state_lock:
            running = self._running
            if running:
                self._interrupted.set()
                self._dispose()
            return {"status": "interrupted" if running else "idle", "session_generation": self._generation,
                    "message": "Python stopped; variables were cleared." if running else "No Python code is running."}

    def reset(self) -> dict:
        if self._running:
            self._interrupted.set()
            self._cancel_script_install()
        with self._state_lock:
            if self._running:
                self._interrupted.set()
            self._dispose()
            return {"status": "reset", "session_generation": self._generation,
                    "message": "Python variables were cleared. Saved execution history is preserved."}

    def close(self):
        self._interrupted.set()
        self._cancel_script_install()
        with self._state_lock:
            self._closed = True
            self._interrupted.set()
            self._dispose(increment=False)
        atexit.unregister(self.close)
