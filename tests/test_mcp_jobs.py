from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from openecon.data import DataError
from openecon.mcp_jobs import ACTIVE, AnalysisJobs, analysis_worker
from openecon.workspace import Workspace


SPEC = {"outcome": "wage", "predictors": ["education", "experience"], "covariance": "HC3"}


def _slow_worker(connection, workspace, scratch, job):
    if os.name == "posix":
        os.setsid()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pids = [os.getpid(), child.pid]
    else:
        pids = [os.getpid()]
    Path(scratch, "pids.json").write_text(json.dumps(pids))
    time.sleep(60)


def _counting_worker(connection, workspace, scratch, job):
    with Path(workspace, "computations.txt").open("a") as stream:
        stream.write(job["id"] + "\n")
    analysis_worker(connection, workspace, scratch, job)


def _limited_worker(connection, workspace, scratch, job):
    import openecon.mcp_jobs as module
    module.MAX_RESULT_BYTES = 1
    analysis_worker(connection, workspace, scratch, job)


def _malformed_worker(connection, workspace, scratch, job):
    connection.send_bytes(b'{"kind":"completed","kind":"failed"}')
    time.sleep(60)


def _slow_compute(self, dataset, spec):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (self.path / "worker-pids.json").write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(60)


def _watched_worker(connection, workspace, scratch, job):
    Workspace.compute_analysis = _slow_compute
    analysis_worker(connection, workspace, scratch, job)


def _crashing_owner(workspace, dataset):
    import openecon.mcp_jobs as module
    module.analysis_worker = _watched_worker
    manager = AnalysisJobs(Workspace(workspace))
    job = manager.start(dataset, SPEC, str(uuid4()))
    Path(workspace, "owner-job.json").write_text(json.dumps(job))
    time.sleep(60)


@pytest.fixture
def jobs(tmp_path):
    store = Workspace(tmp_path)
    dataset = store.create_example()
    manager = AnalysisJobs(store)
    try:
        yield manager, dataset["id"]
    finally:
        manager.close()


def _wait(manager, job_id, *, deadline=25):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        status = manager.get(job_id)
        if status["state"] not in ACTIVE:
            return status
        time.sleep(.05)
    pytest.fail("Analysis job did not reach a terminal state.")


def _pids(manager, job_id):
    path = manager.directory / job_id / "pids.json"
    end = time.monotonic() + 20
    while not path.exists() and time.monotonic() < end:
        time.sleep(.05)
    return json.loads(path.read_text())


def _gone(pid):
    # On Linux an exited grandchild may briefly remain as an init-owned zombie.
    proc = Path(f"/proc/{pid}/stat")
    if proc.exists() and proc.read_text().split()[2] == "Z":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_background_completion_and_retry_survive_restart(jobs, monkeypatch):
    import openecon.mcp_jobs as module
    monkeypatch.setattr(module, "analysis_worker", _counting_worker)
    manager, dataset = jobs
    request = str(uuid4())
    started = time.monotonic()
    job = manager.start(dataset, SPEC, request)
    assert job["state"] == "running" and time.monotonic() - started < 1
    assert manager.start(dataset, SPEC, request)["id"] == job["id"]
    completed = _wait(manager, job["id"])
    assert completed["state"] == "completed", completed
    result = completed["result"]
    assert result["nobs"] == 480 and result["observation_data_included"] is False
    assert not {"sample_positions", "predictions", "residuals", "fitted_values"} & result.keys()
    assert result["coefficients"] == manager.store.get_result(result["id"])["coefficients"]
    paths = [manager.store.result_path / f"{result['id']}.json",
             manager.store.console_path / "mcp-results" / f"{result['id']}.json"]
    mtimes = [path.stat().st_mtime_ns for path in paths]
    manager.close()
    reopened = AnalysisJobs(manager.store)
    try:
        assert reopened.start(dataset, SPEC, request) == completed
        assert reopened.cancel(job["id"]) == completed
        assert [path.stat().st_mtime_ns for path in paths] == mtimes
        assert len(manager.store.list_results()) == len(manager.store.display_history()) == 1
        assert manager.store.console_history() == []
        assert (manager.store.path / "computations.txt").read_text().splitlines() == [job["id"]]
    finally:
        reopened.close()


@pytest.mark.skipif(os.name != "posix", reason="Owned descendant cleanup is a POSIX process-group check.")
@pytest.mark.parametrize("action,expected", [("cancel", "cancelled"), ("close", "interrupted")])
def test_cancellation_and_disconnect_release_worker_and_descendants(jobs, monkeypatch, action, expected):
    import openecon.mcp_jobs as module
    monkeypatch.setattr(module, "analysis_worker", _slow_worker)
    manager, dataset = jobs
    request = str(uuid4())
    job = manager.start(dataset, SPEC, request)
    pids = _pids(manager, job["id"])
    other = AnalysisJobs(manager.store)
    try:
        assert other.start(dataset, SPEC, request)["id"] == job["id"]
        with pytest.raises(DataError) as busy:
            other.start(dataset, SPEC, str(uuid4()))
        assert busy.value.code == "JOB_BUSY"
        if action == "cancel":
            assert other.cancel(job["id"])["state"] == "cancelling"
        else:
            manager.close()
        result = _wait(other, job["id"])
        assert result["state"] == expected
        assert other.start(dataset, SPEC, request)["state"] == expected
        deadline = time.monotonic() + 3
        while not all(_gone(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.05)
        assert all(_gone(pid) for pid in pids)
        assert not (manager.directory / job["id"]).exists()
        assert manager.store.list_results() == manager.store.display_history() == []
    finally:
        other.close()
        for pid in pids:
            if not _gone(pid):
                os.kill(pid, signal.SIGKILL)


def test_deadline_stops_computation_and_retry_is_terminal(jobs, monkeypatch):
    import openecon.mcp_jobs as module
    monkeypatch.setattr(module, "analysis_worker", _slow_worker)
    manager, dataset = jobs
    request = str(uuid4())
    job = manager.start(dataset, SPEC, request, timeout_seconds=1)
    pid = manager._process.pid
    done = _wait(manager, job["id"])
    assert done["state"] == "timed_out" and done["error"]["code"] == "JOB_TIMEOUT"
    assert _gone(pid)
    assert manager.start(dataset, SPEC, request, timeout_seconds=1) == done
    assert manager.store.list_results() == []


def test_concurrent_connections_deduplicate_and_conflicting_reuse_fails(jobs, monkeypatch):
    import openecon.mcp_jobs as module
    monkeypatch.setattr(module, "analysis_worker", _slow_worker)
    first, dataset = jobs
    second = AnalysisJobs(first.store)
    request = str(uuid4())
    try:
        with ThreadPoolExecutor(2) as executor:
            futures = [executor.submit(manager.start, dataset, SPEC, request) for manager in (first, second)]
            statuses = [future.result() for future in futures]
        assert statuses[0]["id"] == statuses[1]["id"]
        assert sum(manager._process is not None for manager in (first, second)) == 1
        with pytest.raises(DataError) as conflict:
            second.start(dataset, {**SPEC, "predictors": ["education"]}, request)
        assert conflict.value.code == "JOB_CONFLICT"
        second.cancel(statuses[0]["id"])
        assert _wait(second, statuses[0]["id"])["state"] == "cancelled"
    finally:
        second.close()


def test_failed_analysis_preserves_core_code_without_result(jobs):
    manager, dataset = jobs
    request = str(uuid4())
    job = manager.start(dataset, {**SPEC, "predictors": ["unknown"]}, request)
    done = _wait(manager, job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "missing_columns"
    assert manager.start(dataset, {**SPEC, "predictors": ["unknown"]}, request) == done
    assert manager.store.list_results() == manager.store.display_history() == []


def test_interrupted_publication_replays_same_result_without_refitting(jobs, monkeypatch):
    import openecon.mcp_jobs as module
    monkeypatch.setattr(module, "analysis_worker", _counting_worker)
    manager, dataset = jobs
    original = module._atomic_json
    failed = False

    def interrupt(path, data):
        nonlocal failed
        if path.parent.name == "mcp-results" and not failed:
            failed = True
            raise OSError("Simulated interrupted UI write")
        return original(path, data)

    monkeypatch.setattr(module, "_atomic_json", interrupt)
    request = str(uuid4())
    job = manager.start(dataset, SPEC, request)
    done = _wait(manager, job["id"])
    assert failed and done["state"] == "completed", done
    assert len(manager.store.list_results()) == len(manager.store.display_history()) == 1
    assert (manager.store.path / "computations.txt").read_text().splitlines() == [job["id"]]
    assert manager.start(dataset, SPEC, request)["result_id"] == job["id"]


def test_orphan_status_is_interrupted_and_never_restarted(jobs):
    manager, dataset = jobs
    request = str(uuid4())
    from openecon.mcp_jobs import _now
    from openecon.workspace import _atomic_json
    _atomic_json(manager._path(request), {"schema": 1, "id": request, "request_id": request,
        "dataset_id": dataset, "state": "running", "created_at": _now(), "updated_at": _now(),
        "timeout_seconds": 300, "error": None})
    result = manager.get(request)
    assert result["state"] == "interrupted" and result["error"]["code"] == "JOB_INTERRUPTED"
    assert manager._process is None and manager.store.list_results() == []


@pytest.mark.skipif(os.name != "posix", reason="Force-killed owner and descendant checks require POSIX.")
def test_force_killed_owner_does_not_leave_computation_alive(jobs):
    import multiprocessing
    manager, dataset = jobs
    owner = multiprocessing.get_context("spawn").Process(target=_crashing_owner,
                                                         args=(str(manager.store.path), dataset))
    pids = []
    try:
        owner.start()
        marker = manager.store.path / "worker-pids.json"
        owner_record = manager.store.path / "owner-job.json"
        deadline = time.monotonic() + 20
        while not (marker.exists() and owner_record.exists()) and time.monotonic() < deadline:
            time.sleep(.05)
        pids = json.loads(marker.read_text())
        job = json.loads(owner_record.read_text())
        owner.kill()
        owner.join(timeout=3)
        deadline = time.monotonic() + 5
        while not all(_gone(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.05)
        assert all(_gone(pid) for pid in pids), "The real analysis worker watchdog must stop orphaned compute."
        assert manager.get(job["id"])["state"] == "interrupted"
        assert not (manager.directory / job["id"]).exists()
        assert manager.store.list_results() == manager.store.display_history() == []
    finally:
        if owner.is_alive():
            owner.kill()
        owner.join(timeout=3)
        owner.close()
        for pid in pids:
            if not _gone(pid):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("timeout", [0, 3601, True, 1.5])
def test_invalid_deadlines_rejected_before_spawning(jobs, timeout):
    manager, dataset = jobs
    with pytest.raises(DataError) as limit:
        manager.start(dataset, SPEC, str(uuid4()), timeout)
    assert limit.value.code == "JOB_LIMIT" and manager._process is None


def test_admission_limits_and_path_ids(jobs, monkeypatch):
    import openecon.mcp_jobs as module
    manager, dataset = jobs
    with pytest.raises(DataError) as bad_id:
        manager.get("../../outside")
    assert bad_id.value.code == "NOT_FOUND"
    with pytest.raises(DataError) as spec_limit:
        manager.start(dataset, {"outcome": "x" * (module.MAX_SPEC_BYTES + 1)}, str(uuid4()))
    assert spec_limit.value.code == "JOB_LIMIT"
    monkeypatch.setattr(module, "MAX_JOBS", 0)
    with pytest.raises(DataError) as retention:
        manager.start(dataset, SPEC, str(uuid4()))
    assert retention.value.code == "JOB_LIMIT"
    assert manager._process is None


def test_oversized_display_is_rejected_before_publishing(jobs, monkeypatch):
    manager, dataset = jobs

    def reject(*args, **kwargs):
        raise DataError("The agent result display exceeds the local history limit.", "DISPLAY_LIMIT")

    monkeypatch.setattr(manager.store, "agent_result_record", reject)
    job = manager.start(dataset, SPEC, str(uuid4()))
    done = _wait(manager, job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "DISPLAY_LIMIT"
    assert manager.store.list_results() == manager.store.display_history() == []


@pytest.mark.parametrize("worker,code", [(_limited_worker, "RESULT_LIMIT"), (_malformed_worker, "JOB_FAILED")])
def test_worker_limits_and_malformed_messages_cannot_publish(jobs, monkeypatch, worker, code):
    import openecon.mcp_jobs as module
    monkeypatch.setattr(module, "analysis_worker", worker)
    manager, dataset = jobs
    job = manager.start(dataset, SPEC, str(uuid4()))
    pid = manager._process.pid
    done = _wait(manager, job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == code
    assert _gone(pid)
    assert manager.store.list_results() == manager.store.display_history() == []


def test_indented_record_limit_is_checked_before_spawning(jobs, monkeypatch):
    import openecon.mcp_jobs as module
    monkeypatch.setattr(module, "MAX_RECORD_BYTES", 8192)
    manager, dataset = jobs
    with pytest.raises(DataError) as limit:
        manager.start(dataset, SPEC, str(uuid4()))
    assert limit.value.code == "JOB_LIMIT" and manager._process is None


def test_start_failure_releases_lease_and_remains_terminal(jobs, monkeypatch):
    from multiprocessing.process import BaseProcess
    manager, dataset = jobs
    request = str(uuid4())

    def fail(*args):
        raise OSError("Simulated spawn failure")

    with monkeypatch.context() as patch:
        patch.setattr(BaseProcess, "start", fail)
        with pytest.raises(DataError) as error:
            manager.start(dataset, SPEC, request)
        assert error.value.code == "JOB_START_FAILED"
    assert manager._lease is manager._process is None
    assert manager.start(dataset, SPEC, request)["state"] == "failed"
    assert _wait(manager, manager.start(dataset, SPEC, str(uuid4()))["id"])["state"] == "completed"
