"""Project isolation, atomic package promotion and wheel-only safety contracts."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
import zipfile

import pytest

from openecon import package_installer as installer
from openecon import project_packages as packages
from openecon.project_packages import PackageError, ProjectPackages


CORE = {"openecon": "0.3.2a1", "numpy": "2.5.3", "pip": "26.2.1"}
PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def manifest(*names):
    records = [{"name": name, "version": "1.2.3"} for name in sorted(names)]
    return {"schema": 1, "python": PYTHON, "core": CORE,
            "requirements": records, "locked": records}


def write_overlay(path: Path, name="demo", version="1.2.3", module=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{module or name}.py").write_text(f"VALUE = {version!r}\n", encoding="utf-8")
    metadata = path / f"{name.replace('-', '_')}-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")


FAKE_INSTALLER = r'''
import json, pathlib, sys, time
action = pathlib.Path(sys.argv[1])
data = json.loads(action.read_text())
stage = action.parent
roots = data['requirements']
if any(row['name'] == 'slow' for row in roots):
    print('Waiting in the owned installer', flush=True)
    (stage / 'entered').write_text('yes')
    time.sleep(20)
if any(row['name'] == 'failure' for row in roots):
    (stage / 'result.json').write_text(json.dumps({'state':'error','code':'NO_WHEEL','message':'No compatible wheel'}))
    raise SystemExit(2)
overlay = stage / 'site-packages'
overlay.mkdir()
locked=[]
for row in roots:
    name, version = row['name'], row['version'] or '1.2.3'
    (overlay / (name.replace('-', '_')+'.py')).write_text('VALUE = '+repr(version)+'\n')
    metadata = overlay / (name.replace('-', '_')+'-'+version+'.dist-info')
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
    locked.append({'name':name,'version':version})
result = {'state':'complete','manifest':{'schema':1,'python':data['python'],'core':data['core'],'requirements':locked,'locked':locked}}
(stage / 'result.json').write_text(json.dumps(result))
print('fake owned installer complete', flush=True)
'''


@pytest.fixture
def managers(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(packages.shutil, "disk_usage", lambda path: SimpleNamespace(free=4 * 1024**3))
    monkeypatch.setattr(packages, "protected_versions", lambda: dict(CORE))
    monkeypatch.setattr(packages, "installer_command", lambda action: [sys.executable, "-c", FAKE_INSTALLER, str(action)])
    created = []

    def new(name="workspace"):
        manager = ProjectPackages(tmp_path / name)
        created.append(manager)
        return manager

    yield new
    for manager in created:
        manager.close()


def wait_job(manager, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        snapshot = manager.snapshot()
        if snapshot["job"]["state"] != "running":
            return snapshot
        time.sleep(.01)
    raise AssertionError("Owned test package job did not terminate")


def wait_process(manager):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if manager._process is not None and "Waiting" in manager.snapshot()["job"]["log"]:
            return
        time.sleep(.01)
    raise AssertionError("Owned test installer did not start")


@pytest.mark.parametrize("name", ["https://a/b", "x>=1", "--target", "../pkg", "x[extra]", "", "a" * 101])
def test_name_injection_rejected(name):
    with pytest.raises(PackageError) as exc:
        packages.canonical_name(name)
    assert exc.value.code == "INVALID_PACKAGE"


@pytest.mark.parametrize("version", [">=1", "1;rm", "https://a", "../1", "v1.0", "", "1 " , True])
def test_nonexact_versions_rejected(version):
    with pytest.raises(PackageError) as exc:
        packages.canonical_version(version)
    assert exc.value.code == "INVALID_VERSION"


def test_canonical_pypi_names_and_pep440_versions():
    assert packages.canonical_name("Demo__Pkg.Name") == "demo-pkg-name"
    assert packages.canonical_version("1!2.3RC1.post2.dev3+CPU") == "1!2.3rc1.post2.dev3+cpu"


def test_manifest_requires_exact_shape_subset_and_fixed_core():
    valid = manifest("demo")
    assert packages.validate_manifest(valid, core=CORE, python=PYTHON) == valid
    variants = [
        {**valid, "extra": 1}, {**valid, "schema": True},
        {**valid, "requirements": [{"name": "missing", "version": "1.2.3"}]},
        {**valid, "locked": valid["locked"] * 2},
        {**valid, "locked": [{"name": "numpy", "version": "2.5.3"}]},
        {**valid, "requirements": [{"name": "demo", "version": "1.2.3", "url": "x"}]},
    ]
    for variant in variants:
        with pytest.raises(PackageError):
            packages.validate_manifest(variant)
    with pytest.raises(PackageError) as exc:
        packages.validate_manifest(valid, core={"openecon": "9.0"})
    assert exc.value.code == "INCOMPATIBLE_CORE"
    with pytest.raises(PackageError) as exc:
        packages.validate_manifest(valid, python="3.99")
    assert exc.value.code == "INCOMPATIBLE_PYTHON"


def test_manifest_package_count_bounded():
    value = manifest(*[f"demo{i}" for i in range(101)])
    with pytest.raises(PackageError) as exc:
        packages.validate_manifest(value)
    assert exc.value.code == "INVALID_MANIFEST"


def test_snapshot_is_immutable_and_project_open_does_not_install(managers):
    manager = managers()
    snapshot = manager.snapshot()
    snapshot["manifest"]["core"]["numpy"] = "9.0"
    snapshot["requirements"].append({"name": "demo", "version": "1.2.3"})
    assert manager.snapshot()["manifest"]["core"] == CORE
    assert manager.snapshot()["requirements"] == []
    assert manager.active_path() is None
    assert manager.snapshot()["job"] is None
    assert list(manager.root.iterdir()) == []


def test_install_promotes_complete_generation_and_reopen_retains_it(managers):
    manager = managers()
    started = manager.start_install("demo")
    assert started["job"]["state"] == "running"
    result = wait_job(manager)
    assert result["job"]["state"] == "complete"
    assert result["requirements"] == [{"name": "demo", "version": "1.2.3"}]
    path = manager.active_path()
    assert path and (path / "demo.py").is_file()
    assert not list(manager.root.glob("stage-*"))
    reopened = ProjectPackages(manager.workspace)
    try:
        assert reopened.active_path() == path
        assert reopened.snapshot()["manifest"] == result["manifest"]
        assert reopened.snapshot()["job"] is None
    finally:
        reopened.close()


def test_project_packages_do_not_leak_into_another_project(managers):
    first, second = managers("first"), managers("second")
    first.start_install("demo", "1.2.3")
    assert wait_job(first)["job"]["state"] == "complete"
    assert second.active_path() is None
    assert second.snapshot()["requirements"] == []


def test_failed_job_retains_previous_active_environment(managers):
    manager = managers()
    manager.start_install("demo", "1.2.3")
    before = wait_job(manager)
    previous_path = manager.active_path()
    manager.start_install("failure")
    result = wait_job(manager)
    assert result["job"]["state"] == "error"
    assert result["job"]["message"] == "No compatible wheel"
    assert result["manifest"] == before["manifest"]
    assert manager.active_path() == previous_path
    assert not list(manager.root.glob("stage-*"))


def test_cancel_kills_owned_installer_and_retains_prior_state(managers):
    manager = managers()
    manager.start_install("demo")
    before = wait_job(manager)
    manager.start_install("slow")
    wait_process(manager)
    process = manager._process
    manager.cancel()
    result = wait_job(manager)
    assert result["job"]["state"] == "cancelled"
    assert process.poll() is not None
    assert result["manifest"] == before["manifest"]
    assert manager.active_path() is not None


def test_close_cancels_job_and_disallows_new_work(managers):
    manager = managers()
    manager.start_install("slow")
    wait_process(manager)
    manager.close()
    assert manager.snapshot()["job"]["state"] == "cancelled"
    with pytest.raises(PackageError) as exc:
        manager.start_install("demo")
    assert exc.value.code == "PACKAGES_CLOSED"


def test_remove_rebuilds_graph_and_last_removal_returns_no_overlay(managers):
    manager = managers()
    manager.start_install("demo")
    wait_job(manager)
    first_path = manager.active_path()
    manager.start_install("other")
    wait_job(manager)
    assert first_path and not first_path.exists()
    manager.start_remove("demo")
    assert wait_job(manager)["requirements"] == [{"name": "other", "version": "1.2.3"}]
    assert not (manager.active_path() / "demo.py").exists()
    manager.start_remove("other")
    assert wait_job(manager)["installed"] == []
    assert manager.active_path() is None


def test_restore_pins_manifest_without_automatic_install_on_open(managers):
    manager = managers()
    manager.start_restore(manifest("demo"))
    assert wait_job(manager)["manifest"] == manifest("demo")


@pytest.mark.parametrize("name", ["numpy", "pip", "math", "os", "openecon-charts"])
def test_base_or_stdlib_roots_cannot_be_requested(managers, name):
    with pytest.raises(PackageError) as exc:
        managers().start_install(name)
    assert exc.value.code == "PROTECTED_PACKAGE"


def test_busy_and_missing_root_errors_are_actionable(managers):
    manager = managers()
    with pytest.raises(PackageError) as exc:
        manager.start_remove("missing")
    assert exc.value.code == "PACKAGE_NOT_REQUESTED"
    manager.start_install("slow")
    with pytest.raises(PackageError) as exc:
        manager.start_install("demo")
    assert exc.value.code == "PACKAGES_BUSY"


def test_terminal_state_cannot_start_new_job_during_previous_cleanup(managers, monkeypatch):
    manager = managers()
    manager.start_install("demo")
    wait_job(manager)
    old_generation = manager.active_path().parent
    entered, release = threading.Event(), threading.Event()
    original = packages.shutil.rmtree

    def blocked(path, *args, **kwargs):
        if Path(path) == old_generation:
            entered.set()
            assert release.wait(3)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(packages.shutil, "rmtree", blocked)
    manager.start_install("other")
    try:
        assert entered.wait(3)
        assert manager.snapshot()["job"]["state"] == "running"
        with pytest.raises(PackageError) as exc:
            manager.start_install("slow")
        assert exc.value.code == "PACKAGES_BUSY"
    finally:
        release.set()
    assert wait_job(manager)["job"]["state"] == "complete"
    manager.start_install("slow")
    wait_process(manager)
    process = manager._process
    manager.cancel()
    assert wait_job(manager)["job"]["state"] == "cancelled"
    assert process.poll() is not None


def test_progress_log_is_bounded(managers):
    manager = managers()
    manager._job = {"id": "0" * 32, "state": "running", "message": "", "log": ""}
    manager._append_log("x" * (2 * packages.MAX_LOG_BYTES))
    assert len(manager.snapshot()["job"]["log"]) == packages.MAX_LOG_BYTES


@pytest.mark.parametrize("name", ["os.py", "NumPy", "torch", "pip", "typing_extensions.py", "math.cpython-313-darwin.so"])
def test_overlay_cannot_shadow_stdlib_or_core_imports(tmp_path, name):
    overlay = tmp_path / "site-packages"
    overlay.mkdir()
    path = overlay / name
    if "." in name:
        path.write_bytes(b"not imported")
    else:
        path.mkdir()
    with pytest.raises(PackageError) as exc:
        packages.validate_overlay(overlay)
    assert exc.value.code == "PROTECTED_PACKAGE"


@pytest.mark.parametrize("name", ["bin", "Scripts", "custom.pth"])
def test_overlay_rejects_executable_entry_hooks(tmp_path, name):
    overlay = tmp_path / "site-packages"
    overlay.mkdir()
    path = overlay / name
    path.write_text("import os", encoding="utf-8")
    with pytest.raises(PackageError) as exc:
        packages.validate_overlay(overlay)
    assert exc.value.code == "UNSAFE_PACKAGE_FILES"


def test_overlay_rejects_links_and_excess_bytes(tmp_path, monkeypatch):
    overlay = tmp_path / "site-packages"
    overlay.mkdir()
    (overlay / "link.py").symlink_to(tmp_path / "elsewhere.py")
    with pytest.raises(PackageError) as exc:
        packages.validate_overlay(overlay)
    assert exc.value.code == "UNSAFE_PACKAGE_FILES"
    (overlay / "link.py").unlink()
    (overlay / "demo.py").write_bytes(b"123456")
    monkeypatch.setattr(packages, "MAX_OVERLAY_BYTES", 5)
    with pytest.raises(PackageError) as exc:
        packages.validate_overlay(overlay)
    assert exc.value.code == "PACKAGE_LIMIT"


def test_active_overlay_metadata_must_match_exact_lock(managers):
    manager = managers()
    manager.start_install("demo")
    wait_job(manager)
    path = manager.active_path()
    metadata = next(path.glob("*.dist-info")) / "METADATA"
    metadata.write_text("Metadata-Version: 2.1\nName: demo\nVersion: 9.0\n", encoding="utf-8")
    with pytest.raises(PackageError) as exc:
        manager.active_path()
    assert exc.value.code == "PACKAGE_LOCK_MISMATCH"


def test_symlinked_package_root_or_pointer_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(packages, "protected_versions", lambda: dict(CORE))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".packages").symlink_to(tmp_path / "other")
    with pytest.raises(PackageError):
        ProjectPackages(workspace)
    (workspace / ".packages").unlink()
    manager = ProjectPackages(workspace)
    manager.close()
    (workspace / ".packages/active.json").symlink_to(tmp_path / "pointer.json")
    with pytest.raises(PackageError):
        ProjectPackages(workspace)


def test_source_and_frozen_installer_commands_use_own_runtime(tmp_path, monkeypatch):
    action = tmp_path / "action.json"
    command = packages.installer_command(action)
    assert command[:2] == [sys.executable, "-c"] and command[-1] == str(action)
    assert "from openecon.package_installer import main" in command[2]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert packages.installer_command(action) == [sys.executable, "--package-installer", str(action)]


def test_installer_environment_omits_cloud_credentials_and_python_hooks(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "must not propagate")
    monkeypatch.setenv("PYTHONPATH", "must not propagate")
    monkeypatch.setenv("PIP_INDEX_URL", "https://not-pypi.example")
    environment = packages.installer_environment()
    assert not {"GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH", "PIP_INDEX_URL"}.intersection(environment)


@pytest.mark.parametrize("url", ["http://files.pythonhosted.org/demo.whl", "https://evil.example/demo.whl",
                                 "file:///tmp/demo.whl", "https://files.pythonhosted.org/demo.tar.gz",
                                 "https://files.pythonhosted.org/demo.whl?token=x", "https://a@files.pythonhosted.org/demo.whl",
                                 "https://files.pythonhosted.org:bad/demo.whl"])
def test_wheel_locations_are_fixed_https_pypi_only(url):
    with pytest.raises(PackageError) as exc:
        installer._safe_wheel_url(url)
    assert exc.value.code == "INVALID_WHEEL"


def report_row(name="demo", version="1.2.3", digest="a" * 64):
    return {"metadata": {"name": name, "version": version},
            "download_info": {"url": f"https://files.pythonhosted.org/{name}-1.2.3-py3-none-any.whl",
                              "archive_info": {"hashes": {"sha256": digest}}}}


def test_resolution_uses_fixed_constraints_and_skips_matching_core(tmp_path, monkeypatch):
    calls = []

    def fake(stage, arguments):
        calls.append(arguments)
        (stage / "resolve.json").write_text(json.dumps({"install": [report_row(), report_row("numpy", "2.5.3")]}))

    monkeypatch.setattr(installer, "_run_pip", fake)
    result = installer._resolve(tmp_path, [{"name": "demo", "version": None}], CORE, None)
    assert [row["name"] for row in result] == ["demo"]
    assert "--dry-run" in calls[0] and "--ignore-installed" in calls[0]
    assert "--only-binary=:all:" in calls[0]
    assert "numpy==2.5.3" in (tmp_path / "constraints.txt").read_text()
    assert installer.INDEX_URL in calls[0]


def test_resolution_cannot_change_core_or_ignore_restore_lock(tmp_path, monkeypatch):
    def fake(stage, arguments):
        (stage / "resolve.json").write_text(json.dumps({"install": [report_row("numpy", "9.0")]}))

    monkeypatch.setattr(installer, "_run_pip", fake)
    with pytest.raises(PackageError) as exc:
        installer._resolve(tmp_path, [{"name": "demo", "version": None}], CORE, None)
    assert exc.value.code == "PROTECTED_PACKAGE"
    monkeypatch.setattr(installer, "_run_pip", lambda stage, arguments: (stage / "resolve.json").write_text(json.dumps({"install": [report_row()]})))
    with pytest.raises(PackageError) as exc:
        installer._resolve(tmp_path, [{"name": "demo", "version": None}], CORE, [{"name": "demo", "version": "9.0"}])
    assert exc.value.code == "PACKAGE_LOCK_MISMATCH"


def zip_bytes(name="demo.py", content=b"value=1", *, symlink=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        entry = zipfile.ZipInfo(name)
        if symlink:
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, content)
    return buffer.getvalue()


class Response(io.BytesIO):
    def geturl(self):
        return "https://files.pythonhosted.org/demo-1.2.3-py3-none-any.whl"


@pytest.mark.parametrize("name,symlink", [("../outside.py", False), ("/outside.py", False),
                                          ("bad\\name.py", False), ("demo.py", True)])
def test_wheel_archive_paths_and_links_rejected_before_pip(tmp_path, monkeypatch, name, symlink):
    content = zip_bytes(name, symlink=symlink)
    monkeypatch.setattr(installer, "urlopen", lambda request, timeout: Response(content))
    with pytest.raises(PackageError) as exc:
        installer._download_wheels(tmp_path, [{"name": "demo", "version": "1.2.3",
            "url": Response().geturl(), "sha256": hashlib.sha256(content).hexdigest()}])
    assert exc.value.code == "UNSAFE_PACKAGE_FILES"


def test_download_hash_and_unpacked_size_limits_are_verified(tmp_path, monkeypatch):
    content = zip_bytes()
    monkeypatch.setattr(installer, "urlopen", lambda request, timeout: Response(content))
    row = {"name": "demo", "version": "1.2.3", "url": Response().geturl(), "sha256": "0" * 64}
    with pytest.raises(PackageError) as exc:
        installer._download_wheels(tmp_path, [row])
    assert exc.value.code == "WHEEL_HASH_MISMATCH"
    import shutil
    shutil.rmtree(tmp_path / "wheels")
    monkeypatch.setattr(installer, "MAX_OVERLAY_BYTES", 1)
    row["sha256"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(PackageError) as exc:
        installer._download_wheels(tmp_path, [row])
    assert exc.value.code == "PACKAGE_LIMIT"


def test_combined_wheel_entry_limit_is_checked_before_extraction(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("resources/", b"")
        archive.writestr("demo.py", b"value=1")
    content = buffer.getvalue()
    monkeypatch.setattr(installer, "urlopen", lambda request, timeout: Response(content))
    monkeypatch.setattr(installer, "MAX_FILES", 3)
    selected = [{"name": name, "version": "1.2.3",
                 "url": f"https://files.pythonhosted.org/{name}-1.2.3-py3-none-any.whl",
                 "sha256": hashlib.sha256(content).hexdigest()}
                for name in ["demo", "other"]]
    with pytest.raises(PackageError) as exc:
        installer._download_wheels(tmp_path, selected)
    assert exc.value.code == "PACKAGE_LIMIT"
    assert not (tmp_path / "site-packages").exists()


def test_guard_blocks_direct_dependency_urls_and_file_requirements_in_resolver(tmp_path):
    # Run the private guard in a fresh process, as production does, so monkey
    # patches cannot contaminate another test's pip modules.
    code = """
from pathlib import Path
from openecon.package_installer import _guard_pip_links
from openecon.project_packages import PackageError
from pip._internal.models.link import Link
_guard_pip_links(['install','--dry-run'], Path('.'))
for value in ['https://evil.example/demo.whl','file:///tmp/demo.whl','http://pypi.org/simple/demo']:
    try:
        Link(value)
    except PackageError:
        pass
    else:
        raise AssertionError(value)
Link('https://files.pythonhosted.org/demo.whl#sha256='+'0'*64)
print('guard-ok')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "guard-ok"
