"""Lifecycle/transport tests; real namespace isolation needs the Cloud Run probe."""
import asyncio
import base64
from datetime import datetime, timedelta, timezone
import json
import os
import signal
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, Mock

from fastapi.testclient import TestClient
import httpx
import pytest

from openecon import team_sandbox_broker as broker
from openecon import team_sandbox_worker as worker

PID, RID = "a" * 32, "b" * 32
# Direct validation tests use an explicit fixed clock. HTTP validation uses the
# actual broker clock, so its request is stamped when the test sends it.
NOW = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)


def request_body(*, now=NOW):
    prefix = f"staging/{PID}/{RID}/"
    fields = {"key": prefix + "completion.json", "Content-Type": "application/json",
              "x-goog-algorithm": "GOOG4-RSA-SHA256", "x-goog-credential": "signer/credential",
              "x-goog-date": now.strftime("%Y%m%dT%H%M%SZ"), "x-goog-signature": "completion-secret"}
    policy = {"expiration": (now + timedelta(minutes=25)).isoformat(), "conditions": [
        {"key": fields["key"]}, {"Content-Type": "application/json"},
        ["content-length-range", 1, 4096], {"bucket": "execution-bucket"},
        *[{k: fields[k]} for k in ("x-goog-algorithm", "x-goog-credential", "x-goog-date")]]}
    fields["policy"] = base64.b64encode(json.dumps(policy).encode()).decode()
    origin = "https://storage.googleapis.com/execution-bucket/" + prefix
    query = "?X-Goog-Signature=input-secret&X-Goog-Expires=1800"
    return {"execution_id": RID, "input_url": origin + "input.json" + query + "&generation=123",
            "cancel_url": origin + "cancel.json" + query,
            "completion_upload": {"url": "https://storage.googleapis.com/execution-bucket",
                                  "fields": fields}, "timeout_seconds": 120}


def validated(**changes):
    return broker.validate_body(json.dumps({**request_body(), **changes}).encode(), now=NOW)


def test_valid_scoped_capabilities_and_short_deadline():
    assert validated(timeout_seconds=.05)["execution_id"] == RID


def test_call_time_fixture_refresh_preserves_expired_capability_rejection():
    after_long_collection = NOW + timedelta(minutes=49)
    with pytest.raises(broker.BrokerError, match="Invalid completion policy"):
        broker.validate_body(json.dumps(request_body()).encode(), now=after_long_collection)
    fresh_body = request_body(now=after_long_collection)
    assert broker.validate_body(json.dumps(fresh_body).encode(),
                                now=after_long_collection)["execution_id"] == RID


@pytest.mark.parametrize("changes", [
    {"execution_id": f"sandbox:{PID}:{RID}"}, {"execution_id": "../escape"},
    {"execution_id": None}, {"timeout_seconds": True}, {"timeout_seconds": "120"},
    {"timeout_seconds": .01}, {"timeout_seconds": 121}, {"timeout_seconds": float("nan")},
    {"untrusted_command": "python"}, {"input_url": "http://metadata.google.internal/"},
    {"input_url": "https://storage.googleapis.com.evil.example/"},
])
def test_strict_request_rejects_extra_fields_and_wrong_types(changes):
    with pytest.raises((broker.BrokerError, ValueError)):
        validated(**changes)


@pytest.mark.parametrize("field, transform", [
    ("input_url", lambda s: s.replace("generation=123", "generation=")),
    ("input_url", lambda s: s + "&generation=456"),
    ("input_url", lambda s: s + "&X-Goog-Signature=duplicate"),
    ("input_url", lambda s: s.replace("input.json", "input%2ejson")),
    ("input_url", lambda s: s.replace(RID, "c" * 32)),
    ("cancel_url", lambda s: s.replace(PID, "c" * 32)),
    ("cancel_url", lambda s: s.replace("execution-bucket", "other-bucket")),
    ("cancel_url", lambda s: s + "&generation=123"),
    ("cancel_url", lambda s: s + "#fragment"),
    ("cancel_url", lambda s: s.replace("1800", "1801")),
])
def test_capabilities_are_bound_to_exact_bucket_project_run_and_generation(field, transform):
    body = request_body()
    body[field] = transform(body[field])
    with pytest.raises(broker.BrokerError):
        broker.validate_body(json.dumps(body).encode(), now=NOW)


@pytest.mark.parametrize("mutation", ["unbounded", "wildcard", "wrong_key", "wrong_bucket",
                                      "extra", "expired", "long", "duplicate", "invalid_base64"])
def test_completion_policy_cannot_broaden_parent_only_capability(mutation):
    body = request_body()
    upload = body["completion_upload"]
    policy = json.loads(base64.b64decode(upload["fields"]["policy"]))
    if mutation == "unbounded":
        policy["conditions"][2][2] = 5000
    elif mutation == "wildcard":
        policy["conditions"][0] = ["starts-with", "$key", "staging/"]
    elif mutation == "wrong_key":
        upload["fields"]["key"] = "staging/other/completion.json"
    elif mutation == "wrong_bucket":
        upload["url"] = "https://storage.googleapis.com/other-bucket"
    elif mutation == "extra":
        upload["fields"]["file"] = "capability"
    elif mutation == "expired":
        policy["expiration"] = (NOW - timedelta(seconds=1)).isoformat()
    elif mutation == "long":
        policy["expiration"] = (NOW + timedelta(hours=1)).isoformat()
    elif mutation == "duplicate":
        policy["conditions"].append(policy["conditions"][0])
    upload["fields"]["policy"] = base64.b64encode(json.dumps(policy).encode()).decode()
    if mutation == "invalid_base64":
        upload["fields"]["policy"] = "!invalid!"
    with pytest.raises(broker.BrokerError):
        broker.validate_body(json.dumps(body).encode(), now=NOW)


def test_json_duplicates_and_oversized_envelope_rejected():
    raw = json.dumps(request_body()).encode()
    with pytest.raises(broker.BrokerError):
        broker.validate_body(raw[:-1] + b',"timeout_seconds":120}', now=NOW)
    with pytest.raises(broker.BrokerError):
        broker.validate_body(b" " * 65537)


class FakeWatchdog:
    def __init__(self):
        self.arms = []
        self.disarmed = False

    def arm(self, seconds):
        self.arms.append(seconds)

    def disarm(self):
        self.disarmed = True


class FakeProcess:
    def __init__(self, complete=True, code=0):
        self.returncode = code if complete else None
        self.exited = asyncio.Event()
        if complete:
            self.exited.set()

    async def wait(self):
        await self.exited.wait()
        return self.returncode


class FakeRuntime:
    def __init__(self, *, complete=True, code=0, cancellations=None, deletion_fails=False):
        self.complete_immediately, self.code = complete, code
        self.cancellations = list(cancellations or [])
        self.deletion_fails = deletion_fails
        self.events = []
        self.process = None
        self.started = None

    async def cancelled(self, body):
        self.events.append("cancel_read")
        return self.cancellations.pop(0) if self.cancellations else False

    async def launch(self, name, url):
        self.events.append(("launch", name, url))
        self.process = FakeProcess(self.complete_immediately, self.code)
        if self.started:
            self.started.set()
        return self.process

    async def input(self, process, url):
        self.events.append(("stdin", url))

    async def delete(self, name):
        self.events.append(("delete", name))
        if self.deletion_fails:
            raise TimeoutError

    async def reap(self, process):
        self.events.append("reap")
        if process is not None and process.returncode is None:
            process.returncode = -9
            process.exited.set()

    async def complete(self, body, status):
        self.events.append(("marker", body["execution_id"], status))


def supervise(runtime=None, **kwargs):
    return broker.Broker(runtime or FakeRuntime(), FakeWatchdog(), heartbeat_seconds=.005, **kwargs)


async def lifecycle(supervisor):
    context = supervisor.acquire()
    supervisor.configure(context, validated())
    frames = []

    async def emit(frame):
        frames.append(json.loads(frame))

    await supervisor.run(context, emit)
    await supervisor.finish(context)
    supervisor.release(context)
    return context, frames


@pytest.mark.parametrize("code,status", [(0, "succeeded"), (1, "failed"), (-9, "failed")])
def test_worker_exit_then_namespace_delete_then_marker(code, status):
    runtime = FakeRuntime(code=code)
    supervisor = supervise(runtime)
    context, frames = asyncio.run(lifecycle(supervisor))
    assert context.status == status and context.cleanup_confirmed
    assert frames == [{"execution_id": RID, "status": "running"}]
    assert runtime.events[-3:] == [("delete", context.name), "reap", ("marker", RID, status)]
    assert context.name.startswith("oe-") and RID not in context.name
    assert supervisor.watchdog.disarmed and not supervisor.busy


def test_cancel_before_launch_produces_no_child():
    runtime = FakeRuntime(cancellations=[True])
    context, _ = asyncio.run(lifecycle(supervise(runtime)))
    assert context.status == "cancelled" and not context.launch_attempted
    assert runtime.events == ["cancel_read", ("marker", RID, "cancelled")]


def test_cancel_during_execution_deletes_entire_sandbox_before_marker():
    runtime = FakeRuntime(complete=False, cancellations=[False, False, True])
    context, frames = asyncio.run(lifecycle(supervise(runtime)))
    assert len(frames) == 2 and context.status == "cancelled"
    assert runtime.events[-3:] == [("delete", context.name), "reap", ("marker", RID, "cancelled")]
    assert runtime.process.returncode == -9


def test_timeout_includes_download_startup_and_reserves_cleanup_budget():
    times = iter([100, 105, 311])
    runtime = FakeRuntime(complete=False)
    supervisor = supervise(runtime, clock=lambda: next(times))
    context, _ = asyncio.run(lifecycle(supervisor))
    assert context.status == "failed" and runtime.process.returncode == -9
    assert supervisor.watchdog.arms == [240, 235]
    assert runtime.events[-1] == ("marker", RID, "failed")


def test_unconfirmed_deletion_poison_terminates_without_marker_or_unlock():
    terminated = []
    runtime = FakeRuntime(deletion_fails=True)
    supervisor = supervise(runtime, terminate=terminated.append)
    with pytest.raises(broker.BrokerError):
        asyncio.run(lifecycle(supervisor))
    assert terminated == [70] and supervisor.poisoned and supervisor.busy
    assert supervisor.acquire() is None
    assert not supervisor.watchdog.disarmed
    assert not any(isinstance(e, tuple) and e[0] == "marker" for e in runtime.events)


def test_marker_failure_keeps_cleanup_safe_and_does_not_invent_success():
    runtime = FakeRuntime()
    runtime.complete = AsyncMock(side_effect=OSError)
    supervisor = supervise(runtime)
    context, _ = asyncio.run(lifecycle(supervisor))
    assert context.cleanup_confirmed and context.status == "failed"
    assert not supervisor.busy and supervisor.watchdog.disarmed


def test_ambiguous_launcher_failure_still_deletes_name_before_marker():
    runtime = FakeRuntime()
    runtime.launch = AsyncMock(side_effect=TimeoutError)
    context, _ = asyncio.run(lifecycle(supervise(runtime)))
    assert context.launch_attempted and context.cleanup_confirmed
    assert runtime.events[-3:] == [("delete", context.name), "reap", ("marker", RID, "failed")]


def test_disconnect_cleanup_is_shielded_and_no_second_request_can_start():
    async def scenario():
        runtime = FakeRuntime(complete=False)
        runtime.started = asyncio.Event()
        supervisor = supervise(runtime)
        context = supervisor.acquire()
        supervisor.configure(context, validated())
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            await runtime.started.wait()
            assert supervisor.acquire() is None
            return {"type": "http.disconnect"}

        await broker.ExecutionResponse(supervisor, context)({}, receive, send)
        assert context.cleanup_confirmed and context.status == "cancelled"
        assert runtime.events[-3:] == [("delete", context.name), "reap", ("marker", RID, "cancelled")]
        assert not supervisor.busy and supervisor.watchdog.disarmed
    asyncio.run(scenario())


def test_asgi_send_failure_before_launch_never_leaves_busy_watchdog():
    async def scenario():
        supervisor = supervise()
        context = supervisor.acquire()
        supervisor.configure(context, validated())

        async def send(_):
            raise OSError("disconnected")

        async def receive():
            await asyncio.Event().wait()

        # AnyIO wraps an ASGI send error, but request-finally still cleans up.
        try:
            await broker.ExecutionResponse(supervisor, context)({}, receive, send)
        except ExceptionGroup:
            pass
        assert context.cleanup_confirmed and not context.launch_attempted
        assert not supervisor.busy and supervisor.watchdog.disarmed
    asyncio.run(scenario())


def test_http_body_limit_generic_errors_and_success_stream():
    supervisor = supervise()
    with TestClient(broker.create_app(supervisor, auth=Mock(verify=AsyncMock()))) as client:
        assert client.get("/healthz").json() == {"ok": True}
        for body in [b"{}", b" " * 65537, b'{"secret":"do-not-echo"}']:
            response = client.post("/execute", content=body, headers={"Content-Type": "application/json"})
            assert response.status_code == 400
            assert response.json() == {"error": "Invalid execution request."}
            assert supervisor.watchdog.disarmed and not supervisor.busy
        response = client.post("/execute", json=request_body(now=datetime.now(timezone.utc)))
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1] == {"execution_id": RID, "status": "succeeded"}
        assert all(len(line) <= 4096 for line in response.content.splitlines(keepends=True))


def test_fixed_launcher_never_passes_headers_marker_or_cancel_capability(monkeypatch):
    process = Mock()
    process.stdin.drain = AsyncMock()
    process.stdin.wait_closed = AsyncMock()
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def scenario():
        runtime = broker.SandboxRuntime()
        await runtime.launch("oe-" + "c" * 32, request_body()["input_url"])
        await runtime.input(process, request_body()["input_url"])
    asyncio.run(scenario())
    args, kwargs = spawn.call_args
    assert args == (broker.SANDBOX_BIN, "run", "oe-" + "c" * 32, "--rootfs", broker.ROOTFS,
                    "--write", "--allow-egress", "--workdir", "/tmp", "--",
                    "/opt/venv/bin/python", "-I", "-m", "openecon.team_sandbox_worker")
    assert kwargs["stdout"] == kwargs["stderr"] == asyncio.subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert set(kwargs["env"]) == {"PATH", "HOME"}
    assert "secret" not in repr(args) + repr(kwargs)
    process.stdin.write.assert_called_once_with(request_body()["input_url"].encode() + b"\n")


def network(monkeypatch, handler):
    original = httpx.AsyncClient
    configurations = []

    def make(**kwargs):
        configurations.append(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", make)
    return configurations


@pytest.mark.parametrize("status,content,expected", [
    (404, b"", False), (200, json.dumps({"execution_id": RID, "cancel_requested": True}).encode(), True),
    (302, b"", None), (403, b"", None), (200, b"x" * 4097, None),
    (200, json.dumps({"execution_id": RID, "cancel_requested": 1}).encode(), None),
    (200, json.dumps({"execution_id": PID, "cancel_requested": True}).encode(), None),
])
def test_cancel_transport_bounded_no_redirect_and_strict_marker(monkeypatch, status, content, expected):
    configurations = network(monkeypatch, lambda _: httpx.Response(status, content=content))
    call = broker.SandboxRuntime().cancelled(validated())
    if expected is None:
        with pytest.raises((broker.BrokerError, ValueError)):
            asyncio.run(call)
    else:
        assert asyncio.run(call) is expected
    assert configurations == [{"trust_env": False, "follow_redirects": False, "timeout": 2}]


def test_completion_upload_contains_only_small_trusted_marker(monkeypatch):
    received = []

    def handler(request):
        received.append(request)
        return httpx.Response(204)

    network(monkeypatch, handler)
    asyncio.run(broker.SandboxRuntime().complete(validated(), "cancelled"))
    request = received[0]
    assert request.method == "POST"
    assert b'{"execution_id":"' + RID.encode() + b'","status":"cancelled","cleanup_confirmed":true}' in request.content
    assert b"input-secret" not in request.content
    assert len(request.content) < 8192


def test_kernel_hard_alarm_terminates_even_without_python_handler():
    script = "import signal,time; signal.signal(signal.SIGALRM, signal.SIG_DFL); signal.setitimer(signal.ITIMER_REAL, .05); time.sleep(20)"
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=3)
    assert completed.returncode == -signal.SIGALRM


@pytest.mark.parametrize("uid,euid,pid", [(0, 0, 1), (10001, 10001, 2), (10001, 0, 2), (0, 10001, 2)])
def test_runtime_refuses_guest_identity_or_python_as_pid_one(monkeypatch, uid, euid, pid):
    monkeypatch.setenv("OPENECON_MODE", "sandbox-broker")
    monkeypatch.setenv("OPENECON_SANDBOX_ENABLED", "1")
    monkeypatch.setenv("OPENECON_SANDBOX_ROOTFS", broker.ROOTFS)
    monkeypatch.setattr(os, "getuid", lambda: uid)
    monkeypatch.setattr(os, "geteuid", lambda: euid)
    monkeypatch.setattr(os, "getpid", lambda: pid)
    # The immutable rootfs and launcher are not even inspected in this state.
    path = Mock(side_effect=AssertionError("Invalid parent identity must be rejected first"))
    monkeypatch.setattr(broker, "Path", path)
    with pytest.raises(RuntimeError, match="unavailable"):
        broker.verify_runtime()
    path.assert_not_called()


@pytest.mark.parametrize("invalid", [None, "launcher", "symlink", "owner", "writable", "marker", "tmp", "run"])
def test_trusted_root_parent_still_requires_clean_immutable_guest_rootfs(monkeypatch, invalid):
    monkeypatch.setenv("OPENECON_MODE", "sandbox-broker")
    monkeypatch.setenv("OPENECON_SANDBOX_ENABLED", "1")
    monkeypatch.setenv("OPENECON_SANDBOX_ROOTFS", broker.ROOTFS)
    monkeypatch.setattr(os, "getuid", lambda: 0)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(os, "getpid", lambda: 2)
    launcher = Mock()
    launcher.is_file.return_value = invalid != "launcher"
    root = MagicMock()
    root.is_symlink.return_value = invalid == "symlink"
    root.stat.return_value.st_uid = 10001 if invalid == "owner" else 0
    root.stat.return_value.st_mode = 0o40755 if invalid == "writable" else 0o40555
    children = {name: Mock() for name in (".openecon-clean-rootfs", "tmp", "run")}
    children[".openecon-clean-rootfs"].is_file.return_value = invalid != "marker"
    for name in ("tmp", "run"):
        children[name].iterdir.return_value = iter([object()] if invalid == name else [])
    root.__truediv__.side_effect = children.__getitem__
    monkeypatch.setattr(broker, "Path", lambda path: root if path == broker.ROOTFS else launcher)
    if invalid:
        with pytest.raises(RuntimeError):
            broker.verify_runtime()
    else:
        broker.verify_runtime()


def test_launcher_delete_uses_force_and_requires_success(monkeypatch):
    async def scenario():
        process = FakeProcess(code=1)
        spawn = AsyncMock(return_value=process)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        with pytest.raises(broker.BrokerError, match="deletion"):
            await broker.SandboxRuntime().delete("oe-fixed")
        args, kwargs = spawn.call_args
        assert args == (broker.SANDBOX_BIN, "delete", "oe-fixed", "--force")
        assert kwargs["stdout"] == kwargs["stderr"] == asyncio.subprocess.DEVNULL
    asyncio.run(scenario())


def test_launcher_reap_kills_entire_process_group_then_waits(monkeypatch):
    async def scenario():
        process = FakeProcess(complete=False)
        process.pid = 4567
        calls = []

        def killpg(pid, kind):
            calls.append((pid, kind))
            process.returncode = -9
            process.exited.set()

        monkeypatch.setattr(os, "killpg", killpg)
        await broker.SandboxRuntime().reap(process)
        assert calls == [(4567, signal.SIGKILL)]
        assert process.returncode == -9
    asyncio.run(scenario())


def test_worker_clears_credentials_and_only_keeps_fixed_runtime_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {"GOOGLE_APPLICATION_CREDENTIALS": "secret",
                                      "OPENECON_COMPLETION_URL": "secret", "HTTP_PROXY": "secret"})
    worker.clean_environment()
    assert not any("secret" in v for v in os.environ.values())
    assert set(os.environ) == {"PATH", "HOME", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
                               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "OPENECON_WORKSPACE"}


def test_worker_uses_managed_guest_privilege_reduction(monkeypatch):
    reduction = Mock()
    monkeypatch.setattr(worker, "reduce_guest_privileges", reduction)
    worker.drop_privileges()
    reduction.assert_called_once_with()


def test_worker_stops_before_input_or_execution_when_privilege_guard_fails(monkeypatch):
    reduction = Mock(side_effect=RuntimeError("Guest privilege reduction failed."))
    clean = Mock(side_effect=AssertionError("No startup work may follow failed reduction"))
    stdin = Mock()
    stdin.buffer.read.side_effect = AssertionError("Input must remain unread")
    monkeypatch.setattr(worker, "reduce_guest_privileges", reduction)
    monkeypatch.setattr(worker, "clean_environment", clean)
    monkeypatch.setattr(worker.sys, "stdin", stdin)
    assert worker.main() == 1
    reduction.assert_called_once_with()
    clean.assert_not_called()
    stdin.buffer.read.assert_not_called()


def test_worker_reduces_privileges_before_environment_rootfs_and_input(monkeypatch):
    events = []
    monkeypatch.setattr(worker, "reduce_guest_privileges", lambda: events.append("privileges"))
    monkeypatch.setattr(worker, "clean_environment", lambda: events.append("environment"))
    marker = Mock()
    marker.is_file.side_effect = lambda: events.append("rootfs") or True
    monkeypatch.setattr(worker, "Path", Mock(return_value=marker))
    stdin = Mock()
    stdin.buffer.read.side_effect = lambda maximum: events.append("input") or b""
    monkeypatch.setattr(worker.sys, "stdin", stdin)
    assert worker.main() == 1
    assert events == ["privileges", "environment", "rootfs", "input"]


def test_metadata_guard_refuses_reachable_endpoint_without_reading_tokens(monkeypatch):
    opener = Mock()
    opener.open.return_value.__enter__ = Mock(return_value=Mock())
    opener.open.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(worker, "build_opener", lambda *args: opener)
    with pytest.raises(RuntimeError):
        worker.verify_metadata_blocked()
    request = opener.open.call_args.args[0]
    assert request.full_url == "http://169.254.169.254/computeMetadata/v1/instance/id"
    assert "token" not in request.full_url and opener.open.call_args.kwargs["timeout"] == 1


@pytest.mark.parametrize("denial", [worker.URLError("blocked"), TimeoutError("blocked")])
def test_metadata_guard_checks_both_addresses_with_blocked_connections(monkeypatch, denial):
    opener = Mock()
    opener.open.side_effect = denial
    monkeypatch.setattr(worker, "build_opener", lambda *args: opener)
    worker.verify_metadata_blocked()
    assert opener.open.call_count == 2
    assert worker._NoRedirect().redirect_request is not None
    with pytest.raises(RuntimeError):
        worker._NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere/")
