"""Real owned pip/uv installs, snapshot builds, rollback and inert source metadata."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import zipfile

import pytest

from openecon import package_installer as installer
from openecon.package_sources import validate_sources, check_build_tools, prepare_sources
from openecon.project_packages import PackageError, ProjectPackages
from openecon.script_packages import _use_installer, install
from openecon.script_packages import _parse_install_command
from openecon.console_worker import validate_install_request, WorkerProtocolError


def wheel(path, version="1.0.0", requires=""):
    target = path / f"oe_source_probe-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("oe_source_probe.py", f"VALUE = {version!r}\n")
        base = f"oe_source_probe-{version}.dist-info/"
        archive.writestr(
            base + "METADATA",
            f"Metadata-Version: 2.1\nName: oe-source-probe\nVersion: {version}\n" + requires,
        )
        archive.writestr(
            base + "WHEEL",
            "Wheel-Version: 1.0\nGenerator: OpenEcon QA\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(base + "RECORD", "")
    return target


def source(path, slow=False):
    path.mkdir()
    (path / "pyproject.toml").write_text(
        '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n'
    )
    (path / "backend.py").write_text(
        """import zipfile,time
def build_wheel(wheel_directory,config_settings=None,metadata_directory=None):
    DELAY
    name="oe_source_probe-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_directory+"/"+name,"w") as archive:
        archive.writestr("oe_source_probe.py","VALUE=42\\n")
        base="oe_source_probe-1.0.0.dist-info/"
        archive.writestr(base+"METADATA","Metadata-Version: 2.1\\nName: oe-source-probe\\nVersion: 1.0.0\\n")
        archive.writestr(base+"WHEEL","Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")
        archive.writestr(base+"RECORD","")
    return name
""".replace("DELAY", "time.sleep(30)" if slow else "pass")
    )
    return path


def wait(manager, limit=30):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        snapshot = manager.snapshot()
        if snapshot["job"]["state"] != "running":
            return snapshot
        time.sleep(0.05)
    manager.cancel()
    raise AssertionError("Owned installer did not complete within the test deadline")


@pytest.mark.parametrize("backend", ["pip", "uv"])
def test_real_hashed_local_wheel_and_atomic_rollback(tmp_path, backend):
    artifact = wheel(tmp_path)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    options = {"artifacts": [{"name": "oe-source-probe", "wheel": str(artifact), "sha256": digest}]}
    manager = ProjectPackages(tmp_path / backend)
    try:
        manager.start_install_many(
            [{"name": "oe-source-probe", "version": "1.0.0"}],
            installer=backend,
            source_options=options,
        )
        snapshot = wait(manager)
        assert snapshot["job"]["state"] == "complete", snapshot["job"]
        path = manager.active_path()
        assert path is not None
        spec = importlib.util.spec_from_file_location("qa_probe", path / "oe_source_probe.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.VALUE == "1.0.0"
        provenance = json.loads((path.parent / "source-provenance.json").read_text())
        assert provenance["sources"][0]["wheel_sha256"] == digest
        assert provenance["sources"][0]["python_abi"] and provenance["sources"][0][
            "wheel_tags"
        ] == ["py3-none-any"]
        portable = json.dumps(manager.export_portable())
        assert (
            str(artifact) not in portable
            and "source_options" not in portable
            and digest not in portable
        )
        bad = {"artifacts": [{**options["artifacts"][0], "sha256": "0" * 64}]}
        manager.start_install_many(
            [{"name": "oe-source-probe", "version": "1.0.0"}], installer=backend, source_options=bad
        )
        failed = wait(manager)
        assert failed["job"]["code"] == "WHEEL_HASH_MISMATCH"
        assert manager.active_path() == path and not list(manager.root.glob("stage-*"))
        manager.close()
        reopened = ProjectPackages(tmp_path / backend)
        try:
            assert reopened.active_path() == path and reopened.snapshot()["job"] is None
        finally:
            reopened.close()
    finally:
        manager.close()


def test_explicit_local_source_build_real_pipeline(tmp_path):
    project = source(tmp_path / "source")
    manager = ProjectPackages(tmp_path / "workspace")
    try:
        manager.start_install_many(
            [{"name": "oe-source-probe", "version": None}],
            source_options={
                "artifacts": [{"name": "oe-source-probe", "directory": str(project), "build": True}]
            },
        )
        snapshot = wait(manager)
        assert snapshot["job"]["state"] == "complete", snapshot["job"]
        path = manager.active_path()
        record = json.loads((path.parent / "source-provenance.json").read_text())["sources"][0]
        assert (
            record["kind"] == "local-build"
            and len(record["source_sha256"]) == 64
            and record["build_isolation"] is False
        )
        assert not (project / "build").exists() and not list(manager.root.glob("stage-*"))
    finally:
        manager.close()


@pytest.mark.parametrize("operation", ["cancelled", "failed"])
def test_failed_or_cancelled_source_build_preserves_active_environment(tmp_path, operation):
    manager = ProjectPackages(tmp_path / "workspace")
    project = source(tmp_path / "source", slow=operation == "cancelled")
    if operation == "failed":
        backend = project / "backend.py"
        backend.write_text(
            backend.read_text().replace(
                "    pass\n", '    raise RuntimeError("QA build failure")\n'
            )
        )
    try:
        artifact = wheel(tmp_path)
        manager.start_install_many(
            [{"name": "oe-source-probe", "version": "1.0.0"}],
            source_options={
                "artifacts": [
                    {
                        "name": "oe-source-probe",
                        "wheel": str(artifact),
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ]
            },
        )
        assert wait(manager)["job"]["state"] == "complete"
        previous = manager.active_path()
        assert previous is not None
        manager.start_install_many(
            [{"name": "oe-source-probe", "version": None}],
            source_options={
                "artifacts": [{"name": "oe-source-probe", "directory": str(project), "build": True}]
            },
        )
        deadline = time.monotonic() + 10
        while (
            "Building an explicitly" not in manager.snapshot()["job"].get("log", "")
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if operation == "cancelled":
            manager.cancel()
        snapshot = wait(manager)
        assert snapshot["job"]["state"] == ("error" if operation == "failed" else "cancelled")
        assert manager.active_path() == previous
        if operation == "failed":
            assert snapshot["job"]["code"] == "PACKAGE_RESOLUTION_FAILED"
        assert "VALUE = '1.0.0'" in (previous / "oe_source_probe.py").read_text()
        assert not list(manager.root.glob("stage-*")) and project.is_dir()
    finally:
        manager.close()


@pytest.mark.parametrize(
    "value",
    [
        {"index_url": "https://user:secret@example.org/simple"},
        {"index_url": "https://example.org/simple?token=secret"},
        {
            "artifacts": [
                {
                    "name": "oe-source-probe",
                    "git": "https://example.org/repo.git",
                    "commit": "main",
                    "build": True,
                }
            ]
        },
        {"artifacts": [{"name": "oe-source-probe", "directory": "/tmp/source", "build": False}]},
        {"editable": True},
    ],
)
def test_invalid_sources_never_reveal_credentials(value):
    with pytest.raises(PackageError) as error:
        validate_sources(value, {"oe-source-probe"})
    assert "secret" not in str(error.value)


def test_protocol_and_public_install_pass_explicit_sources(tmp_path):
    artifact = wheel(tmp_path)
    options = {
        "artifacts": [
            {
                "name": "oe-source-probe",
                "wheel": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
    }
    calls = []

    def callback(requested, **kwargs):
        calls.append((requested, kwargs))
        return {"installed": [{"name": "oe-source-probe", "version": "1.0.0"}]}

    with _use_installer(callback):
        install("oe-source-probe", source_options=options)
    assert calls[0][1]["source_options"] == options
    request = {
        "kind": "package_install",
        "id": "run",
        "request_id": "request",
        "requirements": calls[0][0],
        "loaded_versions": {},
        "source_options": options,
    }
    assert validate_install_request(request, "run")["source_options"] == options
    with pytest.raises(WorkerProtocolError):
        validate_install_request(
            {**request, "source_options": {"index_url": "https://user:secret@example.org"}}, "run"
        )
    command = _parse_install_command(
        "uv",
        ["pip", "install", "--index-url", "https://packages.example.org/simple", "oe-source-probe"],
    )
    assert command["source_options"] == {"index_url": "https://packages.example.org/simple"}


def test_pinned_git_snapshot_and_commit_record(tmp_path, monkeypatch):
    commit = "a" * 40
    options = validate_sources(
        {
            "artifacts": [
                {
                    "name": "oe-source-probe",
                    "git": "https://example.org/repo.git",
                    "commit": commit,
                    "build": True,
                }
            ]
        },
        {"oe-source-probe"},
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "init" in command:
            checkout = Path(command[-1])
            (checkout / "project.txt").write_text("pinned source")
            (checkout / "pyproject.toml").write_text(
                '[build-system]\nrequires=[]\nbuild-backend="backend"\n'
            )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: commit + "\n")

    def build(snapshot, output):
        assert (snapshot / "project.txt").read_text() == "pinned source"
        wheel(output)

    _, records = prepare_sources(tmp_path, options, build)
    assert records[0]["commit"] == commit and len(records[0]["source_sha256"]) == 64
    assert any("fetch" in cmd and commit in cmd for cmd in commands)
    assert not list((tmp_path / "sources").glob("*-checkout"))


def test_custom_index_resolution_policy_and_embedded_constraints(tmp_path, monkeypatch):
    observed = []

    def resolve(stage, arguments):
        observed.extend(arguments)
        (stage / "resolve.json").write_text(json.dumps({"install": []}))

    monkeypatch.setattr(installer, "_run_pip", resolve)
    monkeypatch.setattr(
        installer,
        "_SOURCE_POLICY",
        {"index_url": "https://packages.example.org/simple", "wheel_hosts": ["wheels.example.org"]},
    )
    installer._resolve(tmp_path, [{"name": "demo", "version": None}], {"torch": "2.12.0"}, None)
    assert observed[observed.index("--index-url") + 1] == "https://packages.example.org/simple"
    assert (tmp_path / "constraints.txt").read_text() == "torch==2.12.0\n"
    assert installer._safe_wheel_url(
        "https://wheels.example.org/demo-1.0-py3-none-any.whl"
    ).endswith(".whl")
    with pytest.raises(PackageError):
        installer._safe_wheel_url("https://elsewhere.example.org/demo-1.0-py3-none-any.whl")
    with pytest.raises(PackageError):
        installer._safe_wheel_url(
            "https://wheels.example.org/demo-1.0-py3-none-any.whl?token=secret"
        )


def test_missing_build_tools_and_abi_guard(tmp_path):
    project = source(tmp_path / "source")
    (project / "pyproject.toml").write_text(
        '[build-system]\nrequires=["oe-missing-build-backend==1.0"]\nbuild-backend="missing"\n'
    )
    with pytest.raises(PackageError) as error:
        check_build_tools(project)
    assert error.value.code == "BUILD_TOOLS_UNAVAILABLE"
    path = wheel(tmp_path)
    bad = path.with_name(path.name.replace("py3-none-any", "cp27-cp27m-win32"))
    path.rename(bad)
    options = validate_sources(
        {
            "artifacts": [
                {
                    "name": "oe-source-probe",
                    "wheel": str(bad),
                    "sha256": hashlib.sha256(bad.read_bytes()).hexdigest(),
                }
            ]
        },
        {"oe-source-probe"},
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(PackageError) as error:
        prepare_sources(stage, options, lambda *args: pytest.fail("Wheel must not trigger a build"))
    assert error.value.code == "INVALID_WHEEL"
