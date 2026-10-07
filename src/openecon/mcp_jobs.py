"""Bounded, durable MCP analysis jobs with owned, interruptible worker processes.

The lease elects one computing process per workspace. The separate record lock
lets other MCP connections request cancellation without waiting for the fit.
Request IDs are retained, including failures, so retries never recompute a job.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import stat
import tempfile
import threading
from time import monotonic
from uuid import NAMESPACE_URL, uuid5

from openecon.console_worker import receive_json, send_json
from openecon.data import DataError
from openecon.models import ModelSpec, ResultBundle
from openecon.workspace import Workspace, _atomic_json

MAX_JOBS = 128
MAX_SPEC_BYTES = 64 * 1024
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 128 * 1024
ACTIVE = {"running", "cancelling", "publishing"}
TERMINAL = {"completed", "cancelled", "failed", "timed_out", "interrupted"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _lock(path: Path, *, blocking: bool = True):
    """Return a held OS lock, or None for a busy nonblocking lease."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise DataError("Invalid analysis job lock.", "CORRUPT_JOB")
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        else:
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        os.close(fd)
        if not blocking and exc.errno in {11, 13, 35, 36}:
            return None
        raise
    return fd


def _unlock(fd):
    if fd is not None:
        if os.name == "posix":
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        else:
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)


def _read(path: Path, maximum: int):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError("Invalid job record size.")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(maximum + 1)
        if len(payload) > maximum:
            raise ValueError("Job record exceeds its limit.")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Invalid job record.")
        return value, hashlib.sha256(payload).hexdigest()
    finally:
        os.close(fd)


def analysis_worker(connection, workspace: str, scratch: str, job: dict):
    """Spawn entry point: fixed ModelSpec computation, never client-supplied code."""
    group = False
    if os.name == "posix":
        try:
            os.setsid()
            group = True
        except OSError:
            pass
    # A killed STDIO owner must not leave a fit consuming resources indefinitely.
    parent = multiprocessing.parent_process()

    def watch_parent():
        while parent is not None and parent.is_alive():
            threading.Event().wait(.2)
        if group:
            os.killpg(os.getpid(), signal.SIGKILL)
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    # Numerical libraries must not print into the inherited MCP stdout pipe.
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.close(sink)
    os.environ["TMPDIR"] = scratch
    tempfile.tempdir = scratch
    started = monotonic()
    try:
        store = Workspace(workspace)
        if store.get_dataset(job["dataset_id"])["data_hash"] != job["data_hash"]:
            raise DataError("The dataset changed after job submission.", "DATA_INTEGRITY")
        result = store.compute_analysis(job["dataset_id"], job["spec"])
        result.update(id=job["id"], created_at=job["created_at"])
        path = Path(scratch) / "result.json"
        # Check before allocating the indented disk representation too.
        if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > MAX_RESULT_BYTES:
            raise DataError("The analysis result exceeds the background job limit.", "RESULT_LIMIT")
        _atomic_json(path, result)
        if path.stat().st_size > MAX_RESULT_BYTES:
            raise DataError("The analysis result exceeds the background job limit.", "RESULT_LIMIT")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        send_json(connection, {"kind": "completed", "digest": digest,
                               "duration_ms": (monotonic() - started) * 1000}, maximum=8192)
    except Exception as exc:
        from openecon.analysis import AnalysisError
        expected = isinstance(exc, (DataError, AnalysisError))
        send_json(connection, {"kind": "failed", "code": exc.code if expected else "JOB_FAILED",
                               "message": str(exc)[:1000] if expected else "The analysis worker failed."},
                  maximum=8192)
    finally:
        connection.close()


class AnalysisJobs:
    def __init__(self, store: Workspace):
        self.store = store
        self.directory = store.path / ".mcp-jobs"
        self.directory.mkdir(exist_ok=True)
        if self.directory.is_symlink():
            raise DataError("Invalid analysis job directory.", "CORRUPT_JOB")
        self._mutex = threading.RLock()
        self._process = self._connection = self._lease = self._thread = None
        self._closing = threading.Event()
        atexit.register(self.close)

    @contextmanager
    def _guard(self):
        with self._mutex:
            if self.directory.is_symlink() or self.directory.resolve().parent != self.store.path:
                raise DataError("Invalid analysis job directory.", "CORRUPT_JOB")
            fd = _lock(self.directory / "records.lock")
            try:
                yield
            finally:
                _unlock(fd)

    def _path(self, job_id):
        return self.directory / f"{self.store._id(job_id)}.json"

    def _record(self, job_id):
        try:
            job, _ = _read(self._path(job_id), MAX_RECORD_BYTES)
            if job["id"] != job_id or job["state"] not in ACTIVE | TERMINAL or job["schema"] != 1:
                raise ValueError("Invalid job state.")
            return job
        except FileNotFoundError as exc:
            raise DataError("The analysis job was not found.", "NOT_FOUND") from exc
        except (OSError, ValueError, KeyError) as exc:
            raise DataError("The analysis job record cannot be read.", "CORRUPT_JOB") from exc

    def _write(self, job, **changes):
        job.update(changes, updated_at=_now())
        if len(json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False).encode()) > MAX_RECORD_BYTES:
            raise DataError("The analysis job record exceeds its limit.", "JOB_LIMIT")
        _atomic_json(self._path(job["id"]), job)

    def _scratch(self, job):
        path = self.directory / job["id"]
        if path.is_symlink():
            raise DataError("Invalid analysis job staging directory.", "CORRUPT_JOB")
        return path

    def _cleanup(self, job):
        shutil.rmtree(self._scratch(job), ignore_errors=True)

    def _prepared(self, job):
        try:
            result, digest = _read(self._scratch(job) / "result.json", MAX_RESULT_BYTES)
            if digest != job["digest"] or result["id"] != job["id"] or result["dataset_id"] != job["dataset_id"]:
                raise ValueError("Invalid staged identity.")
            if result["spec"] != job["spec"] or result["provenance"]["dataset_snapshot_hash"] != job["data_hash"]:
                raise ValueError("Invalid staged specification.")
            ResultBundle.model_validate({k: v for k, v in result.items() if k not in {"dataset_id", "dataset_name"}})
            display = self.store.agent_result_record(result, duration_ms=job["duration_ms"])
            return result, display
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, DataError):
                raise
            raise DataError("The staged analysis result is invalid.", "CORRUPT_JOB") from exc

    def _publish(self, job, prepared=None):
        result, display = prepared or self._prepared(job)
        # The durable publishing state commits the decision before either write.
        # Recovery replaces these same IDs; it never reruns the estimator.
        _atomic_json(self.store.result_path / f"{job['id']}.json", result)
        directory = self.store.console_path / "mcp-results"
        directory.mkdir(exist_ok=True)
        _atomic_json(directory / f"{job['id']}.json", display)
        self._write(job, state="completed", result_id=job["id"], error=None)
        self._cleanup(job)

    def _recover(self):
        if self._lease is not None:
            return
        lease = _lock(self.directory / "controller.lock", blocking=False)
        if lease is None:
            return
        try:
            for path in self.directory.glob("*.json"):
                job = self._record(path.stem)
                if job["state"] == "publishing":
                    self._publish(job)
                elif job["state"] in ACTIVE:
                    self._write(job, state="interrupted", error={"code": "JOB_INTERRUPTED",
                                "message": "The owning MCP server stopped. Submit a new request ID to run again."})
                    self._cleanup(job)
        finally:
            _unlock(lease)

    def _status(self, job):
        from openecon.mcp_server import compact_result
        status = {k: job[k] for k in ("id", "request_id", "dataset_id", "state", "created_at", "updated_at", "timeout_seconds", "error")}
        if job["state"] == "completed":
            status["result_id"] = job["result_id"]
            status["result"] = compact_result(self.store.get_result(job["result_id"]))
        return status

    def start(self, dataset_id: str, spec: dict, request_id: str, timeout_seconds: int = 300):
        request_id = self.store._id(request_id)
        dataset_id = self.store._id(dataset_id)
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
            raise DataError("The job deadline must be between 1 and 3600 seconds.", "JOB_LIMIT")
        if len(json.dumps(spec, ensure_ascii=False, allow_nan=False).encode()) > MAX_SPEC_BYTES:
            raise DataError("The model specification exceeds the job limit.", "JOB_LIMIT")
        validated = ModelSpec.model_validate(spec).model_dump(mode="json")
        fingerprint = hashlib.sha256(json.dumps([dataset_id, validated, timeout_seconds], sort_keys=True).encode()).hexdigest()
        job_id = str(uuid5(NAMESPACE_URL, "openecon:mcp-analysis:" + request_id))
        with self._guard():
            if self._closing.is_set():
                raise DataError("The MCP server is shutting down.", "JOB_CLOSED")
            self._recover()
            if self._path(job_id).exists():
                job = self._record(job_id)
                if job["fingerprint"] != fingerprint:
                    raise DataError("This request ID already belongs to a different analysis.", "JOB_CONFLICT")
                return self._status(job)
            profile = self.store.get_dataset(dataset_id)
            job = dict(schema=1, id=job_id, request_id=request_id, dataset_id=dataset_id,
                       spec=validated, data_hash=profile["data_hash"], fingerprint=fingerprint,
                       timeout_seconds=timeout_seconds, created_at=_now(), updated_at=_now(),
                       state="running", error=None)
            # Keep room for terminal metadata in the actual indented disk format.
            if len(json.dumps(job, ensure_ascii=False, indent=2).encode()) > MAX_RECORD_BYTES - 8192:
                raise DataError("The model specification exceeds the job record limit.", "JOB_LIMIT")
            if sum(1 for _ in self.directory.glob("*.json")) >= MAX_JOBS:
                raise DataError("The workspace retains 128 job IDs. Use a new workspace for further jobs.", "JOB_LIMIT")
            if self._lease is not None:
                raise DataError("An analysis job is already running in this workspace.", "JOB_BUSY")
            lease = _lock(self.directory / "controller.lock", blocking=False)
            if lease is None:
                raise DataError("An analysis job is already running in this workspace.", "JOB_BUSY")
            self._lease = lease
            try:
                self._write(job)
                scratch = self._scratch(job)
                scratch.mkdir()
                context = multiprocessing.get_context("spawn")
                self._connection, child = context.Pipe(duplex=False)
                self._process = context.Process(target=analysis_worker,
                    args=(child, str(self.store.path), str(scratch), job), daemon=True)
                try:
                    self._process.start()
                finally:
                    child.close()
                started = monotonic()
                self._thread = threading.Thread(target=self._monitor, args=(job_id, started), daemon=True)
                self._thread.start()
                return self._status(job)
            except Exception:
                try:
                    self._stop()
                    self._write(job, state="failed", error={"code": "JOB_START_FAILED", "message": "The analysis worker could not start."})
                    self._cleanup(job)
                finally:
                    _unlock(self._lease)
                    self._lease = None
                raise DataError("The analysis worker could not start.", "JOB_START_FAILED") from None

    def get(self, job_id: str):
        job_id = self.store._id(job_id)
        with self._guard():
            self._recover()
            return self._status(self._record(job_id))

    def cancel(self, job_id: str):
        job_id = self.store._id(job_id)
        with self._guard():
            self._recover()
            job = self._record(job_id)
            if job["state"] == "running":
                self._write(job, state="cancelling")
            return self._status(job)

    def _stop(self):
        process, connection = self._process, self._connection
        if process is not None and process.pid is not None:
            try:
                if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.is_alive():
                    process.kill()
            except ProcessLookupError:
                pass
            process.join(timeout=2)
            if process.is_alive():
                raise DataError("The analysis worker has not stopped.", "JOB_STOP_FAILED")
            process.close()
        elif process is not None:
            process.close()
        if connection is not None:
            connection.close()
        self._process = self._connection = None

    def _monitor(self, job_id, started):
        try:
            while True:
                with self._guard():
                    job = self._record(job_id)
                    stop = ("interrupted" if self._closing.is_set() else
                            "cancelled" if job["state"] == "cancelling" else
                            "timed_out" if monotonic() - started >= job["timeout_seconds"] else None)
                    if stop:
                        self._stop()
                        self._cleanup(job)
                        self._write(job, state=stop, error=None if stop == "cancelled" else
                                    {"code": "JOB_TIMEOUT" if stop == "timed_out" else "JOB_INTERRUPTED",
                                     "message": "The analysis worker was stopped."})
                        return
                    if self._connection.poll():
                        message = receive_json(self._connection, maximum=8192)
                        self._stop()
                        if message.get("kind") == "failed":
                            if set(message) != {"kind", "code", "message"} or not all(isinstance(message[k], str) for k in ("code", "message")):
                                raise ValueError("Invalid failure message.")
                            self._write(job, state="failed", error={"code": message["code"], "message": message["message"]})
                        elif (set(message) == {"kind", "digest", "duration_ms"} and message["kind"] == "completed"
                              and isinstance(message["digest"], str) and len(message["digest"]) == 64
                              and type(message["duration_ms"]) in {int, float} and message["duration_ms"] >= 0):
                            job.update(digest=message["digest"], duration_ms=message["duration_ms"])
                            prepared = self._prepared(job)
                            self._write(job, state="publishing")
                            self._publish(job, prepared)
                        else:
                            raise ValueError("Invalid completion message.")
                        self._cleanup(job)
                        return
                    if not self._process.is_alive():
                        raise ValueError("The worker exited without a result.")
                self._closing.wait(.05)
        except Exception as exc:
            with self._guard():
                job = self._record(job_id)
                # A publication interrupted between writes remains replayable.
                if job["state"] != "publishing":
                    self._write(job, state="failed", error={"code": exc.code if isinstance(exc, DataError) else "JOB_FAILED",
                                "message": str(exc) if isinstance(exc, DataError) else "The analysis worker failed."})
                    self._cleanup(job)
        finally:
            with self._guard():
                self._stop()
                _unlock(self._lease)
                self._lease = None

    def close(self):
        self._closing.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        atexit.unregister(self.close)
