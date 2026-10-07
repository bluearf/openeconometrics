"""Private, roleless Cloud Run supervisor; submitted Python never runs here.

The platform and this broker authenticate the invoker. Only a fresh sandbox receives
the input capability. Completion/cancellation capabilities stay in this process.
There is deliberately no subprocess fallback when the sandbox CLI is unavailable.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import time
from typing import Callable
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import anyio
from fastapi import FastAPI, Request
import httpx
from starlette.responses import JSONResponse, Response

from openecon.team_sandbox_auth import BrokerAuth, BrokerAuthError

MAX_REQUEST = 65536
MAX_MARKER = 4096
MAX_WALL_SECONDS = 240
ROOTFS = "/opt/openecon-sandbox-rootfs"
SANDBOX_BIN = "/usr/local/gcp/bin/sandbox"
_ID = r"[0-9a-f]{32}"
_FIELDS = frozenset({"execution_id", "input_url", "cancel_url", "completion_upload",
                     "timeout_seconds"})
_UPLOAD_FIELDS = frozenset({"key", "policy", "x-goog-algorithm", "x-goog-credential",
                            "x-goog-date", "x-goog-signature", "Content-Type"})


class BrokerError(ValueError):
    """A generic failure; never include capabilities or user output in the message."""


def json_object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BrokerError("Duplicate field.")
            result[key] = value
        return result

    def invalid(_):
        raise BrokerError("Invalid number.")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise BrokerError("An object is required.")
    return result


def _storage_url(value: str, *, pinned: bool) -> tuple[str, str]:
    if (not isinstance(value, str) or not value.isascii() or not 1 <= len(value) <= 16384
            or any(ch.isspace() or ord(ch) < 32 for ch in value)):
        raise BrokerError("Invalid capability.")
    parsed = urlsplit(value)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if (parsed.scheme != "https" or parsed.netloc != "storage.googleapis.com" or parsed.fragment
            or any(len(values) != 1 or not values[0] for values in query.values())
            or not query.get("X-Goog-Signature") or not query.get("X-Goog-Expires")):
        raise BrokerError("Invalid capability.")
    try:
        if not 1 <= int(query["X-Goog-Expires"][0]) <= 1800:
            raise ValueError
        if pinned and not query.get("generation", [""])[0].isdigit():
            raise ValueError
        if not pinned and "generation" in query:
            raise ValueError
    except (ValueError, TypeError):
        raise BrokerError("Invalid capability.") from None
    match = re.fullmatch(r"/([a-z0-9][a-z0-9._-]{1,220}[a-z0-9])/(.+)", parsed.path)
    if not match:
        raise BrokerError("Invalid capability.")
    return match.group(1), match.group(2)


def validate_body(raw: bytes, *, now: datetime | None = None) -> dict:
    if len(raw) > MAX_REQUEST:
        raise BrokerError("Request too large.")
    body = json_object(raw)
    if set(body) != _FIELDS or not isinstance(body["execution_id"], str):
        raise BrokerError("Invalid execution request.")
    run_id = body["execution_id"]
    if not re.fullmatch(_ID, run_id):
        raise BrokerError("Invalid execution identifier.")
    duration = body["timeout_seconds"]
    if (type(duration) not in (float, int) or not math.isfinite(duration)
            or not .05 <= duration <= 120):
        raise BrokerError("Invalid execution timeout.")
    bucket, key = _storage_url(body["input_url"], pinned=True)
    match = re.fullmatch(rf"staging/({_ID})/{run_id}/input\.json", key)
    if not match:
        raise BrokerError("Invalid input scope.")
    prefix = f"staging/{match.group(1)}/{run_id}/"
    if _storage_url(body["cancel_url"], pinned=False) != (bucket, prefix + "cancel.json"):
        raise BrokerError("Invalid cancellation scope.")
    upload = body["completion_upload"]
    if (not isinstance(upload, dict) or set(upload) != {"url", "fields"}
            or upload["url"] not in {f"https://storage.googleapis.com/{bucket}",
                                     f"https://storage.googleapis.com/{bucket}/"}
            or not isinstance(upload["fields"], dict) or set(upload["fields"]) != _UPLOAD_FIELDS):
        raise BrokerError("Invalid completion scope.")
    fields = upload["fields"]
    if (any(not isinstance(v, str) or not v or len(v) > 16384 or "\r" in v or "\n" in v
            for v in fields.values()) or sum(map(len, fields.values())) > 32768
            or fields["key"] != prefix + "completion.json"
            or fields["Content-Type"] != "application/json"
            or fields["x-goog-algorithm"] != "GOOG4-RSA-SHA256"):
        raise BrokerError("Invalid completion capability.")
    try:
        policy = json_object(base64.b64decode(fields["policy"], validate=True))
        expires = datetime.fromisoformat(policy["expiration"].replace("Z", "+00:00"))
        remaining = (expires - (now or datetime.now(timezone.utc))).total_seconds()
        conditions = policy["conditions"]
        # No starts-with wildcard, duplicate condition, alternate key or relaxed
        # size condition can broaden the sole parent-owned completion object.
        required = [{"key": fields["key"]}, {"Content-Type": "application/json"},
                    ["content-length-range", 1, MAX_MARKER]]
        allowed = required + [{"bucket": bucket}] + [
            {name: fields[name]} for name in ("x-goog-algorithm", "x-goog-credential", "x-goog-date")]
        if (set(policy) != {"expiration", "conditions"} or not 0 < remaining <= 1800
                or not isinstance(conditions, list) or len(conditions) > len(allowed)
                or any(conditions.count(item) != 1 for item in required)
                or any(item not in allowed or conditions.count(item) != 1 for item in conditions)):
            raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError):
        raise BrokerError("Invalid completion policy.") from None
    return body


class WallWatchdog:
    """A kernel real-time alarm, not a Python thread or Python signal handler.

    Run on the main event-loop thread. The default SIGALRM disposition kills the
    broker even if Python stalls. Container teardown semantics still require the
    deployed Cloud Run isolation probe; unit tests cannot establish that boundary.
    """
    def arm(self, seconds: float) -> None:
        if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
            raise RuntimeError("The hard watchdog must use the default fatal signal disposition.")
        signal.setitimer(signal.ITIMER_REAL, max(.001, seconds))

    def disarm(self) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)


class SandboxRuntime:
    """Fixed launcher and bounded HTTPS only; no Google credentials or cloud SDK."""
    async def launch(self, name: str, input_url: str):
        process = await asyncio.create_subprocess_exec(
            SANDBOX_BIN, "run", name, "--rootfs", ROOTFS, "--write", "--allow-egress",
            "--workdir", "/tmp", "--", "/opt/venv/bin/python", "-I", "-m",
            "openecon.team_sandbox_worker", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True, env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"})
        return process

    async def input(self, process, input_url: str) -> None:
        process.stdin.write(input_url.encode("ascii") + b"\n")
        await asyncio.wait_for(process.stdin.drain(), 5)
        process.stdin.close()
        await asyncio.wait_for(process.stdin.wait_closed(), 5)

    async def delete(self, name: str) -> None:
        process = await asyncio.create_subprocess_exec(
            SANDBOX_BIN, "delete", name, "--force", stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True, env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"})
        try:
            if await asyncio.wait_for(process.wait(), 10) != 0:
                raise BrokerError("Sandbox deletion was not confirmed.")
        finally:
            if process.returncode is None:
                await self.reap(process)

    async def reap(self, process) -> None:
        if process is None:
            return
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.wait_for(process.wait(), 5)

    async def cancelled(self, body: dict) -> bool:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=2) as client:
            async with client.stream("GET", body["cancel_url"]) as response:
                if response.status_code == 404:
                    return False
                if response.status_code != 200:
                    raise BrokerError("Cancellation status unavailable.")
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_MARKER:
                        raise BrokerError("Cancellation status exceeds its limit.")
                marker = json_object(bytes(chunks))
                if (set(marker) != {"execution_id", "cancel_requested"}
                        or marker["execution_id"] != body["execution_id"]
                        or marker["cancel_requested"] is not True):
                    raise BrokerError("Invalid cancellation status.")
                return True

    async def complete(self, body: dict, status: str) -> None:
        marker = json.dumps({"execution_id": body["execution_id"], "status": status,
                             "cleanup_confirmed": True}, separators=(",", ":")).encode()
        if status not in {"succeeded", "failed", "cancelled"} or len(marker) > MAX_MARKER:
            raise BrokerError("Invalid completion marker.")
        upload = body["completion_upload"]
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=4) as client:
            async with client.stream("POST", upload["url"], data=upload["fields"],
                                     files={"file": ("completion.json", marker, "application/json")}) as response:
                if not 200 <= response.status_code < 300:
                    raise BrokerError("Completion could not be stored.")
                # No response body is useful or trusted. Closing the stream
                # prevents an unbounded body or a slow response from delaying exit.


@dataclass
class Execution:
    accepted_at: float
    body: dict | None = None
    name: str = ""
    process: object | None = None
    launch_attempted: bool = False
    status: str = "failed"
    cleanup_confirmed: bool = False
    finished: bool = False


class Broker:
    def __init__(self, runtime=None, watchdog=None, *, terminate: Callable = os._exit,
                 clock: Callable = time.monotonic, heartbeat_seconds: float = 1):
        self.runtime = runtime or SandboxRuntime()
        self.watchdog = watchdog or WallWatchdog()
        self.terminate, self.clock = terminate, clock
        self.heartbeat_seconds = heartbeat_seconds
        self.busy = False
        self.poisoned = False

    def acquire(self) -> Execution | None:
        if self.busy or self.poisoned:
            return None
        self.busy = True
        context = Execution(self.clock())
        try:
            self.watchdog.arm(MAX_WALL_SECONDS)
        except BaseException:
            self.busy = False
            raise
        return context

    def configure(self, context: Execution, body: dict) -> None:
        context.body, context.name = body, "oe-" + uuid4().hex
        remaining = context.accepted_at + math.ceil(body["timeout_seconds"]) + 120 - self.clock()
        self.watchdog.arm(remaining)

    def poison(self) -> None:
        self.poisoned = True
        self.terminate(70)
        raise BrokerError("Sandbox cleanup was not confirmed; supervisor is unavailable.")

    def release(self, context: Execution) -> None:
        if self.poisoned or (context.launch_attempted and not context.cleanup_confirmed):
            self.poison()
        self.watchdog.disarm()
        self.busy = False

    async def finish(self, context: Execution) -> None:
        if context.finished:
            return
        if context.launch_attempted:
            try:
                # Delete the entire namespace/overlay first. Reaping the launcher
                # alone would not establish that an escaped process group died.
                await asyncio.wait_for(self.runtime.delete(context.name), 15)
                await asyncio.wait_for(self.runtime.reap(context.process), 5)
                context.cleanup_confirmed = True
            except BaseException:
                self.poison()
        else:
            context.cleanup_confirmed = True
        if context.body is not None:
            try:
                await asyncio.wait_for(self.runtime.complete(context.body, context.status), 5)
            except Exception:
                # No success is invented if storing the trusted marker failed.
                # The controller keeps the lease until its conservative deadline.
                context.status = "failed"
        context.finished = True

    @staticmethod
    def frame(context: Execution, status: str) -> bytes:
        return (json.dumps({"execution_id": context.body["execution_id"], "status": status},
                           separators=(",", ":")) + "\n").encode()

    async def run(self, context: Execution, send_frame: Callable) -> None:
        body = context.body
        # Reserve the last 30 seconds of the kernel hard bound for cleanup and
        # marker upload, including input download and interpreter startup time.
        cutoff = context.accepted_at + math.ceil(body["timeout_seconds"]) + 90
        try:
            await send_frame(self.frame(context, "running"))
            if await asyncio.wait_for(self.runtime.cancelled(body), 3):
                context.status = "cancelled"
                return
            context.launch_attempted = True
            context.process = await asyncio.wait_for(self.runtime.launch(context.name, body["input_url"]), 10)
            await asyncio.wait_for(self.runtime.input(context.process, body["input_url"]), 6)
            while True:
                if self.clock() >= cutoff:
                    break
                if await asyncio.wait_for(self.runtime.cancelled(body), 3):
                    context.status = "cancelled"
                    break
                try:
                    code = await asyncio.wait_for(context.process.wait(), self.heartbeat_seconds)
                    context.status = "succeeded" if code == 0 else "failed"
                    break
                except TimeoutError:
                    await send_frame(self.frame(context, "running"))
        except asyncio.CancelledError:
            context.status = "cancelled"
            raise
        except Exception:
            context.status = "failed"


class ExecutionResponse(Response):
    media_type = "application/x-ndjson"

    def __init__(self, broker: Broker, context: Execution):
        super().__init__(content=b"", status_code=200,
                         headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
        self.raw_headers = [(k, v) for k, v in self.raw_headers if k != b"content-length"]
        self.broker, self.context = broker, context

    async def __call__(self, scope, receive, send):
        async def send_frame(frame):
            await send({"type": "http.response.body", "body": frame, "more_body": True})

        async def disconnected(group):
            while True:
                if (await receive())["type"] == "http.disconnect":
                    self.context.status = "cancelled"
                    group.cancel_scope.cancel()
                    return

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(disconnected, group)
                try:
                    await send({"type": "http.response.start", "status": 200,
                                "headers": self.raw_headers})
                    await self.broker.run(self.context, send_frame)
                finally:
                    # Disconnect cancellation must never skip deletion. This
                    # bounded cleanup executes before the ASGI request returns.
                    with anyio.CancelScope(shield=True):
                        await self.broker.finish(self.context)
                    group.cancel_scope.cancel()
            await send_frame(self.broker.frame(self.context, self.context.status))
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            # Also covers an ASGI send/receive exception before execution starts.
            with anyio.CancelScope(shield=True):
                if not self.context.finished and not self.broker.poisoned:
                    await self.broker.finish(self.context)
                if not self.broker.poisoned:
                    self.broker.release(self.context)


def create_app(broker: Broker | None = None, *, auth: BrokerAuth | None = None) -> FastAPI:
    supervisor = broker or Broker()
    authorizer = auth if auth is not None else BrokerAuth.from_environment()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.broker = supervisor

    @app.get("/healthz")
    async def health():
        return JSONResponse({"ok": not supervisor.poisoned}, status_code=503 if supervisor.poisoned else 200)

    @app.post("/execute")
    async def execute(request: Request):
        try:
            await authorizer.verify(request.headers.getlist("authorization"))
        except BrokerAuthError as error:
            return JSONResponse({"error": str(error)}, status_code=error.status_code)
        context = supervisor.acquire()
        if context is None:
            return JSONResponse({"error": "Compute is unavailable."}, status_code=503)
        handed_off = False
        try:
            if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
                raise BrokerError("JSON is required.")
            if "content-encoding" in request.headers:
                raise BrokerError("Encoded request bodies are not accepted.")
            raw = bytearray()
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > MAX_REQUEST:
                        raise BrokerError("Request too large.")
                    raw.extend(chunk)
            supervisor.configure(context, validate_body(bytes(raw)))
            handed_off = True
            return ExecutionResponse(supervisor, context)
        except (ValueError, TypeError, TimeoutError):
            return JSONResponse({"error": "Invalid execution request."}, status_code=400)
        finally:
            if not handed_off:
                supervisor.release(context)

    return app


def verify_runtime() -> None:
    # Only the trusted launcher needs host root. Submitted Python runs in a
    # fresh managed guest; Tini remains PID 1 so the hard alarm can kill Python.
    if (os.environ.get("OPENECON_MODE") != "sandbox-broker"
            or os.environ.get("OPENECON_SANDBOX_ENABLED") != "1"
            or os.environ.get("OPENECON_SANDBOX_ROOTFS") != ROOTFS
            or os.getuid() != 0 or os.geteuid() != 0
            or os.getpid() == 1 or not Path(SANDBOX_BIN).is_file()):
        raise RuntimeError("The isolated compute runtime is unavailable.")
    root = Path(ROOTFS)
    if (root.is_symlink() or root.stat().st_uid != 0 or root.stat().st_mode & 0o222
            or not (root / ".openecon-clean-rootfs").is_file()
            or any((root / "tmp").iterdir()) or any((root / "run").iterdir())):
        raise RuntimeError("The compute root filesystem is not immutable and clean.")


def main() -> None:
    verify_runtime()
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                access_log=False, log_level="critical", workers=1, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
