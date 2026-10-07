"""Atomic script package batches, live-import guards and generation retention."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from openecon import project_packages as packages
from openecon.project_packages import PackageError, ProjectPackages


CORE = {"openecon": "0.3.4a1", "numpy": "2.5.3", "pip": "26.2.1"}
FAKE_RESOLVER = r'''
import json, pathlib, sys, time
action = pathlib.Path(sys.argv[1])
data = json.loads(action.read_text())
stage = action.parent
roots = {row['name']: row['version'] or '1.0' for row in data['requirements']}
if 'wait' in roots:
    (stage / 'entered').write_text('owned fixture')
    print('Waiting in the owned resolver', flush=True)
    deadline = time.monotonic() + 20
    while not (stage / 'release').exists() and time.monotonic() < deadline:
        time.sleep(.01)
if 'failure' in roots:
    (stage / 'result.json').write_text(json.dumps(
        {'state': 'error', 'code': 'NO_WHEEL', 'message': 'No compatible wheel'}))
    raise SystemExit(2)
locked = dict(roots)
if 'demo' in roots and roots['demo'] != '2.0':
    locked.setdefault('helper', '2.0' if 'upgrade-dependency' in roots else '1.0')
overlay = stage / 'site-packages'
overlay.mkdir()
for name, version in sorted(locked.items()):
    module = name.replace('-', '_')
    (overlay / (module + '.py')).write_text('VALUE = ' + repr(version) + '\n')
    metadata = overlay / (module + '-' + version + '.dist-info')
    metadata.mkdir()
    (metadata / 'METADATA').write_text(
        'Metadata-Version: 2.1\nName: ' + name + '\nVersion: ' + version + '\n')
records = lambda values: [{'name': name, 'version': version} for name, version in sorted(values.items())]
(stage / 'result.json').write_text(json.dumps({'state': 'complete', 'manifest': {
    'schema': 1, 'python': data['python'], 'core': data['core'],
    'requirements': records(roots), 'locked': records(locked)}}))
print('Owned resolver complete', flush=True)
'''


@pytest.fixture
def manager_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(packages, "protected_versions", lambda: dict(CORE))
    monkeypatch.setattr(packages.shutil, "disk_usage", lambda _path: SimpleNamespace(free=4 * 1024**3))
    actions, managers = [], []

    def command(action):
        actions.append(json.loads(action.read_text()))
        return [sys.executable, "-c", FAKE_RESOLVER, str(action)]

    monkeypatch.setattr(packages, "installer_command", command)

    def create(name="workspace"):
        manager = ProjectPackages(tmp_path / name)
        managers.append(manager)
        return manager

    yield SimpleNamespace(create=create, actions=actions, tmp_path=tmp_path)
    for manager in managers:
        manager.close()


def rows(**versions):
    return [{"name": name, "version": version} for name, version in versions.items()]


def wait_job(manager):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = manager.snapshot()
        if snapshot["job"] and snapshot["job"]["state"] != "running":
            return snapshot
        time.sleep(.01)
    pytest.fail("The owned package fixture did not terminate")


def wait_stage(manager):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.snapshot()["job"]
        stage = manager.root / f"stage-{job['id']}"
        if (stage / "entered").is_file():
            return stage
        time.sleep(.01)
    pytest.fail("The owned package fixture did not enter its wait")


def install_demo(manager):
    manager.start_install_many(rows(demo="1.0"))
    result = wait_job(manager)
    assert result["job"]["state"] == "complete"
    return result


def assert_rejected_atomically(manager, requirement, code, **kwargs):
    before = manager.snapshot()
    paths = set(manager.root.iterdir())
    with pytest.raises(PackageError) as exc:
        manager.start_install_many(requirement, **kwargs)
    assert exc.value.code == code
    assert manager.snapshot() == before
    assert set(manager.root.iterdir()) == paths


def test_two_roots_install_in_one_atomic_job(manager_fixture):
    manager = manager_fixture.create()
    pending = manager.start_install_many(rows(**{"new-one": None, "new-two": "3.0"}))
    assert pending["job"]["state"] == "running"
    result = wait_job(manager)
    assert result["job"]["state"] == "complete"
    assert result["requirements"] == rows(**{"new-one": "1.0", "new-two": "3.0"})
    assert len(manager_fixture.actions) == 1
    assert manager_fixture.actions[0]["requirements"] == rows(**{"new-one": None, "new-two": "3.0"})
    assert len(list(manager.root.glob("generation-*"))) == 1
    assert not list(manager.root.glob("stage-*"))


@pytest.mark.parametrize(("requirement", "code"), [
    ([{"name": "new-one", "version": None}, {"name": "https://invalid/pkg", "version": None}], "INVALID_PACKAGE"),
    ([{"name": "new-one", "version": None}, {"name": "bad", "version": ">=1"}], "INVALID_VERSION"),
    ([{"name": "Demo_pkg", "version": None}, {"name": "demo-pkg", "version": "1.0"}], "INVALID_MANIFEST"),
    ([{"name": "new-one", "version": None, "url": "ignored"}], "INVALID_MANIFEST"),
    ([{"name": "new-one"}], "INVALID_MANIFEST"),
    ({"name": "new-one", "version": None}, "INVALID_MANIFEST"),
    (None, "INVALID_MANIFEST"),
    ([{"name": f"pkg{i}", "version": None} for i in range(101)], "INVALID_MANIFEST"),
    ([{"name": "numpy", "version": "9.0"}, {"name": "new-one", "version": None}], "PROTECTED_PACKAGE"),
    ([{"name": "os", "version": None}], "PROTECTED_PACKAGE"),
    ([{"name": "openecon-charts", "version": None}], "PROTECTED_PACKAGE"),
])
def test_invalid_batches_do_not_start_or_mutate(manager_fixture, requirement, code):
    manager = manager_fixture.create()
    assert_rejected_atomically(manager, requirement, code)
    assert manager_fixture.actions == []


@pytest.mark.parametrize("requirement", [
    [], rows(numpy=None), rows(numpy="2.5.3.0"), rows(openecon="0.3.4a1"),
    rows(numpy=None, openecon="0.3.4a1"),
])
def test_core_and_empty_batches_are_identical_noops(manager_fixture, requirement):
    manager = manager_fixture.create()
    manager._available = False
    before = manager.snapshot()
    assert manager.start_install_many(requirement) == before
    assert list(manager.root.iterdir()) == []
    assert manager_fixture.actions == []


@pytest.mark.parametrize("version", [None, "1.0", "1.0.0"])
def test_existing_direct_pin_noop_keeps_job_generation_and_avoids_network(manager_fixture, version):
    manager = manager_fixture.create()
    before = install_demo(manager)
    path = manager.active_path()
    assert manager.start_install_many(rows(demo=version)) == before
    assert manager.active_path() == path
    assert len(manager_fixture.actions) == 1


def test_noop_does_not_replay_an_old_failure(manager_fixture):
    manager = manager_fixture.create()
    install_demo(manager)
    manager.start_install_many(rows(failure=None))
    before = wait_job(manager)
    assert before["job"]["code"] == "NO_WHEEL"
    assert manager.start_install_many(rows(demo=None)) == before
    assert len(manager_fixture.actions) == 2


def test_existing_transitive_request_becomes_pinned_root_without_new_job(manager_fixture):
    manager = manager_fixture.create()
    before = install_demo(manager)
    path = manager.active_path()
    result = manager.start_install_many(rows(helper=None))
    assert result["job"] == before["job"]
    assert result["requirements"] == rows(demo="1.0", helper="1.0")
    assert result["installed"] == before["installed"]
    assert manager.active_path() == path
    assert len(manager_fixture.actions) == 1
    reopened = manager_fixture.create()
    assert reopened.snapshot()["requirements"] == result["requirements"]
    manager.start_remove("demo")
    assert wait_job(manager)["installed"] == rows(helper="1.0")


def test_direct_root_promotion_rolls_back_on_atomic_write_failure(manager_fixture, monkeypatch):
    manager = manager_fixture.create()
    before = install_demo(manager)
    active = (manager.root / "active.json").read_bytes()

    def fail_write(_path, _value):
        raise OSError("Owned fixture write failure")

    monkeypatch.setattr(packages, "_write_json", fail_write)
    with pytest.raises(PackageError) as exc:
        manager.start_install_many(rows(helper="1.0"))
    assert exc.value.code == "PACKAGE_STATE_WRITE_FAILED"
    assert manager.snapshot() == before
    assert (manager.root / "active.json").read_bytes() == active
    assert len(manager_fixture.actions) == 1


def test_existing_roots_remain_pinned_in_merged_single_resolve(manager_fixture):
    manager = manager_fixture.create()
    install_demo(manager)
    manager.start_install_many(rows(**{"demo": None, "helper": None, "new-one": None}))
    result = wait_job(manager)
    assert result["job"]["state"] == "complete"
    assert manager_fixture.actions[-1]["requirements"] == rows(**{"demo": "1.0", "helper": "1.0", "new-one": None})
    assert result["requirements"] == rows(**{"demo": "1.0", "helper": "1.0", "new-one": "1.0"})
    assert len(manager_fixture.actions) == 2


@pytest.mark.parametrize(("requested", "loaded"), [
    (rows(demo="2.0"), {"demo": "1.0"}),
    (rows(**{"upgrade-dependency": None}), {"helper": "1.0"}),
    (rows(demo="2.0"), {"helper": "1.0"}),
])
def test_imported_roots_and_transitives_cannot_change_or_disappear(manager_fixture, requested, loaded):
    manager = manager_fixture.create()
    before = install_demo(manager)
    path = manager.active_path()
    active = (manager.root / "active.json").read_bytes()
    manager.start_install_many(requested, loaded_versions=loaded, preserve_previous=True)
    result = wait_job(manager)
    assert result["job"]["state"] == "error"
    assert result["job"]["code"] == "PACKAGE_RESTART_REQUIRED"
    assert result["job"]["message"] == "Restart Python before changing an imported package."
    assert result["manifest"] == before["manifest"]
    assert manager.active_path() == path
    assert (manager.root / "active.json").read_bytes() == active
    assert len(list(manager.root.glob("generation-*"))) == 1
    assert not list(manager.root.glob("stage-*"))


def test_matching_live_dependencies_and_core_allow_addition_with_old_paths_retained(manager_fixture):
    manager = manager_fixture.create()
    install_demo(manager)
    old_path = manager.active_path()
    manager.start_install_many(rows(**{"new-one": None}),
                               loaded_versions={"demo": "1.0.0", "helper": "1.0", "numpy": "2.5.3"},
                               preserve_previous=True)
    result = wait_job(manager)
    assert result["job"]["state"] == "complete"
    assert manager.active_path() != old_path
    assert (old_path / "helper.py").is_file()
    assert manager.prune_generations() == 1
    assert not old_path.exists()
    assert manager.active_path().is_dir()


def test_only_valid_inactive_owned_generation_directories_are_pruned(manager_fixture):
    manager = manager_fixture.create()
    install_demo(manager)
    for name in ["new-one", "new-two"]:
        manager.start_install_many(rows(**{name: None}), preserve_previous=True)
        assert wait_job(manager)["job"]["state"] == "complete"
    active = manager.active_path()
    unrelated = [manager.root / "generation-other", manager.root / "stage-" / "owned",
                 manager.root / ("generation-" + "A" * 32)]
    for path in unrelated:
        path.mkdir(parents=True)
    foreign = manager_fixture.tmp_path / "owned-foreign-fixture"
    foreign.mkdir()
    (foreign / "keep").write_text("safe")
    link = manager.root / f"generation-{uuid4().hex}"
    link.symlink_to(foreign, target_is_directory=True)
    regular_file = manager.root / f"generation-{uuid4().hex}"
    regular_file.write_text("keep")
    stale = manager.root / f"generation-{uuid4().hex}"
    stale.mkdir()
    assert manager.prune_generations() == 3
    assert active.is_dir()
    assert all(path.is_dir() for path in unrelated)
    assert regular_file.read_text() == "keep"
    assert link.is_symlink() and (foreign / "keep").read_text() == "safe"
    assert manager.prune_generations() == 0


def test_prune_skips_running_job_and_cancel_retains_prior_generations(manager_fixture):
    manager = manager_fixture.create()
    install_demo(manager)
    old_path = manager.active_path()
    manager.start_install_many(rows(**{"new-one": None}), preserve_previous=True)
    assert wait_job(manager)["job"]["state"] == "complete"
    active = manager.active_path()
    manager.start_install_many(rows(wait=None), preserve_previous=True)
    wait_stage(manager)
    assert manager.prune_generations() == 0
    assert old_path.is_dir() and active.is_dir()
    assert_rejected_atomically(manager, rows(demo=None), "PACKAGES_BUSY")
    manager.cancel()
    result = wait_job(manager)
    assert result["job"]["state"] == "cancelled"
    assert result["job"]["code"] == "PACKAGE_CANCELLED"
    assert manager.active_path() == active
    assert manager.prune_generations() == 1
    assert active.is_dir() and not old_path.exists()


def test_batch_and_loaded_versions_are_copied_before_background_job(manager_fixture):
    manager = manager_fixture.create()
    install_demo(manager)
    requested = rows(**{"wait": None, "upgrade-dependency": None})
    loaded = {"helper": "1.0"}
    manager.start_install_many(requested, loaded_versions=loaded, preserve_previous=True)
    stage = wait_stage(manager)
    loaded["helper"] = "2.0"
    requested.clear()
    (stage / "release").write_text("continue")
    result = wait_job(manager)
    assert result["job"]["code"] == "PACKAGE_RESTART_REQUIRED"
    assert manager_fixture.actions[-1]["requirements"] == rows(**{"demo": "1.0", "upgrade-dependency": None, "wait": None})


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"preserve_previous": 1}, "INVALID_MANIFEST"),
    ({"loaded_versions": []}, "INVALID_MANIFEST"),
    ({"loaded_versions": {"Demo_pkg": "1.0", "demo-pkg": "1.0"}}, "INVALID_MANIFEST"),
    ({"loaded_versions": {"demo": ">=1"}}, "INVALID_VERSION"),
    ({"loaded_versions": {"--unsafe": "1.0"}}, "INVALID_PACKAGE"),
    ({"loaded_versions": {f"pkg{i}": "1.0" for i in range(201)}}, "INVALID_MANIFEST"),
])
def test_invalid_live_install_options_are_atomic(manager_fixture, kwargs, code):
    manager = manager_fixture.create()
    assert_rejected_atomically(manager, rows(demo=None), code, **kwargs)
    assert manager_fixture.actions == []


def test_prune_and_new_job_are_serialized(manager_fixture, monkeypatch):
    manager = manager_fixture.create()
    install_demo(manager)
    old_generation = manager.active_path().parent
    manager.start_install_many(rows(**{"new-one": None}), preserve_previous=True)
    assert wait_job(manager)["job"]["state"] == "complete"
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    original = packages.shutil.rmtree

    def delayed(path, *args, **kwargs):
        if Path(path) == old_generation:
            entered.set()
            assert release.wait(5)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(packages.shutil, "rmtree", delayed)
    prune = threading.Thread(target=manager.prune_generations)

    def start():
        manager.start_install_many(rows(**{"new-two": None}), preserve_previous=True)
        started.set()

    install = threading.Thread(target=start)
    try:
        prune.start()
        assert entered.wait(5)
        install.start()
        assert not started.wait(.05)
    finally:
        release.set()
        prune.join(5)
        if install.ident is not None:
            install.join(5)
    assert not prune.is_alive() and not install.is_alive()
    assert started.is_set()
    assert wait_job(manager)["job"]["state"] == "complete"


@pytest.mark.skipif(packages.os.name != "posix", reason="POSIX process-group cancellation")
@pytest.mark.parametrize("state", ["exited", "alive", "term-denied"])
def test_cancel_permission_error_targets_only_owned_live_leader(manager_fixture, monkeypatch, state):
    manager = manager_fixture.create()
    calls = []

    class OwnedProcess:
        pid = 123456789
        returncode = 0 if state == "exited" else None

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append("terminate")
            if state == "term-denied":
                raise PermissionError("Owned fixture denied terminate")
            self.returncode = 0

        def kill(self):
            calls.append("kill")
            self.returncode = 0

        def wait(self, *, timeout):
            assert timeout == .5
            if self.returncode is None:
                raise packages.subprocess.TimeoutExpired("owned fixture", timeout)
            return self.returncode

    def denied_group(pid, _signal):
        assert pid == OwnedProcess.pid
        raise PermissionError("Owned fixture exited process group")

    manager._job = {"id": uuid4().hex, "state": "running", "message": "fixture", "log": ""}
    manager._process = OwnedProcess()
    monkeypatch.setattr(packages.os, "killpg", denied_group)
    assert manager.cancel()["job"]["state"] == "running"
    assert manager._cancelled.is_set()
    assert calls == {"exited": [], "alive": ["terminate"], "term-denied": ["terminate", "kill"]}[state]
    manager._job["state"] = "cancelled"
