"""uv lock identity, embedded discovery, and real offline wheel installation."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace
from uuid import uuid4
import zipfile

import pytest

from openecon import package_installer as installer
from openecon import uv_runtime
from openecon.project_packages import PackageError


def row(name="demo-uv", version="1.2.3", *, filename=None, url=None, digest="a" * 64):
    return {"name": name, "version": version, "wheels": [{
        "url": url or "https://files.pythonhosted.org/" + (filename or f"{name.replace('-', '_')}-{version}-py3-none-any.whl"),
        "hashes": {"sha256": digest}}]}


def write_lock(path, rows, version="1.0"):
    lines = [f"lock-version = {json.dumps(version)}", 'created-by = "uv"']
    for item in rows:
        lines.append("[[packages]]")
        for key in ("name", "version", "marker", "directory"):
            if key in item:
                lines.append(f"{key} = {json.dumps(item[key])}")
        if "wheels" in item:
            wheels = ["{ url = " + json.dumps(wheel["url"]) + ", hashes = { sha256 = "
                      + json.dumps(wheel["hashes"]["sha256"]) + " } }" for wheel in item["wheels"]]
            lines.append("wheels = [" + ", ".join(wheels) + "]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wheel_bytes(name="demo-uv", version="1.2.3", *, requirements=(), extras=()):
    normalized = name.replace("-", "_")
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(normalized + "/__init__.py", "VALUE = 'isolated verified wheel'\n")
        directory = f"{normalized}-{version}.dist-info/"
        metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: >=3.9\n"
        metadata += "".join(f"Requires-Dist: {value}\n" for value in requirements)
        metadata += "".join(f"Provides-Extra: {value}\n" for value in extras)
        archive.writestr(directory + "METADATA", metadata)
        archive.writestr(directory + "WHEEL", "Wheel-Version: 1.0\nGenerator: independent-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(directory + "RECORD", "")
    return data.getvalue()


@pytest.fixture
def native_uv():
    try:
        path = uv_runtime.uv_executable()
    except PackageError:
        pytest.skip("A native uv binary is not installed in this test environment")
    return path


def embedded_probe_executable(tmp_path):
    probe = tmp_path / "embedded-python"
    source = Path(uv_runtime.__file__).resolve().parents[1]
    probe.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(source)!r})\n"
                     "from openecon.uv_runtime import maybe_uv_python_probe\n"
                     f"sys.executable = {str(probe)!r}\n"
                     "raise SystemExit(0 if maybe_uv_python_probe(sys.argv[1:]) else 47)\n")
    probe.chmod(0o700)
    return probe


def test_uv_resolution_retains_full_specs_and_core_constraints(tmp_path, monkeypatch):
    calls = []

    def resolve(stage, arguments):
        calls.append(arguments)
        write_lock(stage / "pylock.toml", [row(), row("numpy", "2.3.3")])

    monkeypatch.setattr(installer, "_run_uv", resolve)
    selected = installer._resolve_uv(tmp_path, [{"name": "demo-uv", "version": None}],
                                     {"numpy": "2.3.3"}, None, ["demo-uv[extras]>=1,<2"])
    assert [(value["name"], value["version"]) for value in selected] == [("demo-uv", "1.2.3")]
    assert (tmp_path / "requirements.in").read_text() == "demo-uv[extras]>=1,<2\n"
    assert (tmp_path / "constraints.txt").read_text() == "numpy==2.3.3\n"
    assert "compile" == calls[0][0]
    for required in ["--no-build", "--no-sources", "--constraint", "--keyring-provider", "--quiet"]:
        assert required in calls[0]
    assert calls[0][calls[0].index("--default-index") + 1] == "https://pypi.org/simple"


def test_uv_resolution_restores_exact_transitive_graph(tmp_path, monkeypatch):
    locks = [{"name": "demo-uv", "version": "1.2.3"}, {"name": "dependency", "version": "4.5.6"}]
    monkeypatch.setattr(installer, "_run_uv", lambda stage, args: write_lock(stage / "pylock.toml", [row(), row("dependency", "4.5.6")]))
    selected = installer._resolve_uv(tmp_path, [{"name": "demo-uv", "version": "1.2.3"}], {}, locks,
                                     ["demo-uv[optional]>=1"])
    assert [{"name": value["name"], "version": value["version"]} for value in selected] == locks
    assert (tmp_path / "constraints.txt").read_text() == "demo-uv==1.2.3\ndependency==4.5.6\n"


def project_stage(tmp_path):
    stage = tmp_path / ".packages" / f"stage-{uuid4().hex}"
    stage.mkdir(parents=True)
    return stage


def downloadable_row(content, name="demo-uv", version="1.2.3"):
    return {"name": name, "version": version, "url": row(name, version)["wheels"][0]["url"],
            "sha256": hashlib.sha256(content).hexdigest()}


class DownloadResponse(io.BytesIO):
    def __init__(self, content, url):
        super().__init__(content)
        self.url = url

    def geturl(self):
        return self.url


def test_verified_project_wheel_cache_reuses_bytes_without_network(tmp_path, monkeypatch):
    content = wheel_bytes()
    selected = downloadable_row(content)
    calls = []

    def download(request, timeout):
        calls.append((request.full_url, timeout))
        return DownloadResponse(content, request.full_url)

    monkeypatch.setattr(installer, "urlopen", download)
    first = project_stage(tmp_path)
    first_wheel = installer._download_wheels(first, [selected])[0]
    assert first_wheel.read_bytes() == content
    assert calls == [(selected["url"], 15)]
    cached = first.parent / "wheel-cache" / (selected["sha256"] + ".whl")
    assert cached.read_bytes() == content
    assert json.loads((first / "wheel-downloads.json").read_text()) == {"cache_hits": 0, "downloads": 1}
    monkeypatch.setattr(installer, "urlopen", lambda *args, **kwargs: pytest.fail("Verified repeated wheel should not use the network"))
    second = project_stage(tmp_path)
    second_wheel = installer._download_wheels(second, [selected])[0]
    assert second_wheel.read_bytes() == content
    assert second_wheel.stat().st_ino != cached.stat().st_ino, "Staged bytes must not mutate the retained cache through a hardlink"
    assert json.loads((second / "wheel-downloads.json").read_text()) == {"cache_hits": 1, "downloads": 0}


def test_corrupt_project_wheel_cache_is_redownloaded_and_repaired(tmp_path, monkeypatch):
    content = wheel_bytes()
    selected = downloadable_row(content)
    stage = project_stage(tmp_path)
    cache = stage.parent / "wheel-cache"
    cache.mkdir()
    cached = cache / (selected["sha256"] + ".whl")
    cached.write_bytes(b"changed local cache bytes")
    calls = []

    def download(request, timeout):
        calls.append(request.full_url)
        return DownloadResponse(content, request.full_url)

    monkeypatch.setattr(installer, "urlopen", download)
    downloaded = installer._download_wheels(stage, [selected])
    assert downloaded[0].read_bytes() == content
    assert calls == [selected["url"]]
    assert cached.read_bytes() == content
    assert json.loads((stage / "wheel-downloads.json").read_text()) == {"cache_hits": 0, "downloads": 1}
    assert not list(stage.glob("cached-wheel-*"))
    monkeypatch.setattr(installer, "urlopen", lambda *args, **kwargs: pytest.fail("Repaired wheel must become reusable"))
    installer._download_wheels(project_stage(tmp_path), [selected])


@pytest.mark.parametrize("link_kind", ["directory", "wheel"])
def test_project_wheel_cache_links_fail_before_network_or_outside_writes(tmp_path, monkeypatch, link_kind):
    content = wheel_bytes()
    selected = downloadable_row(content)
    stage = project_stage(tmp_path)
    cache = stage.parent / "wheel-cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "untouched.whl"
    sentinel.write_bytes(content)
    if link_kind == "directory":
        cache.symlink_to(outside, target_is_directory=True)
    else:
        cache.mkdir()
        (cache / (selected["sha256"] + ".whl")).symlink_to(sentinel)
    monkeypatch.setattr(installer, "urlopen", lambda *args, **kwargs: pytest.fail("Unsafe cache path reached a download"))
    with pytest.raises(PackageError) as error:
        installer._download_wheels(stage, [selected])
    assert error.value.code == "UNSAFE_PACKAGE_FILES"
    assert sentinel.read_bytes() == content
    assert list(outside.iterdir()) == [sentinel]


def test_cached_wheel_still_checks_archive_paths_before_installation(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../outside.py", "unsafe")
    content = buffer.getvalue()
    selected = downloadable_row(content)
    stage = project_stage(tmp_path)
    cache = stage.parent / "wheel-cache"
    cache.mkdir()
    (cache / (selected["sha256"] + ".whl")).write_bytes(content)
    monkeypatch.setattr(installer, "urlopen", lambda *args, **kwargs: pytest.fail("Cached test bytes should avoid network"))
    with pytest.raises(PackageError) as error:
        installer._download_wheels(stage, [selected])
    assert error.value.code == "UNSAFE_PACKAGE_FILES"
    assert not (tmp_path / "outside.py").exists()


def test_cached_wheels_obey_aggregate_download_budget(tmp_path, monkeypatch):
    stage = project_stage(tmp_path)
    cache = stage.parent / "wheel-cache"
    cache.mkdir()
    first, second = wheel_bytes("first"), wheel_bytes("second")
    selected = [downloadable_row(first, "first"), downloadable_row(second, "second")]
    for value, content in zip(selected, [first, second], strict=True):
        (cache / (value["sha256"] + ".whl")).write_bytes(content)
    monkeypatch.setattr(installer, "MAX_DOWNLOAD_BYTES", len(first) + len(second) - 1)
    monkeypatch.setattr(installer, "urlopen", lambda *args, **kwargs: pytest.fail("Both test wheels are already cached"))
    with pytest.raises(PackageError) as error:
        installer._download_wheels(stage, selected)
    assert error.value.code == "PACKAGE_LIMIT"


def test_project_wheel_cache_evicts_old_verified_wheels_to_remain_bounded(tmp_path, monkeypatch):
    first, second = wheel_bytes("first"), wheel_bytes("second")
    contents = {row("first")["wheels"][0]["url"]: first,
                row("second")["wheels"][0]["url"]: second}
    monkeypatch.setattr(installer, "MAX_DOWNLOAD_BYTES", max(len(first), len(second)))
    monkeypatch.setattr(installer, "urlopen", lambda request, timeout: DownloadResponse(contents[request.full_url], request.full_url))
    installer._download_wheels(project_stage(tmp_path), [downloadable_row(first, "first")])
    installer._download_wheels(project_stage(tmp_path), [downloadable_row(second, "second")])
    cache = tmp_path / ".packages" / "wheel-cache"
    entries = list(cache.glob("*.whl"))
    assert len(entries) == 1
    assert entries[0].name == hashlib.sha256(second).hexdigest() + ".whl"
    assert entries[0].read_bytes() == second


@pytest.mark.parametrize("saved_pin,expected", [("1.2.3", "1.2.3"), (None, "1.7.0")])
def test_native_uv_broad_range_preserves_saved_pin_until_explicit_upgrade(tmp_path, monkeypatch, native_uv, saved_pin, expected):
    wheels = tmp_path / "local-wheels"
    wheels.mkdir()
    for version in ["1.2.3", "1.7.0", "2.0.0"]:
        (wheels / f"demo_uv-{version}-py3-none-any.whl").write_bytes(wheel_bytes(version=version))
    original_run = installer._run_uv

    def local_resolution(stage, arguments):
        adjusted = list(arguments)
        index = adjusted.index("--default-index")
        adjusted[index:index + 2] = ["--offline", "--no-index", "--find-links", str(wheels)]
        original_run(stage, adjusted)

    # Only the fixture index changes. The actual native resolver, production
    # specifications, and newly retained root constraint file remain untouched.
    monkeypatch.setattr(installer, "_run_uv", local_resolution)
    monkeypatch.setattr(uv_runtime, "uv_executable", lambda: native_uv)
    probe = embedded_probe_executable(tmp_path)
    monkeypatch.setattr(installer.sys, "executable", str(probe))
    monkeypatch.setenv("PATH", str(tmp_path / "no-system-python"))

    def read_fixture(path, core, locked):
        records = tomllib.loads(path.read_text())["packages"]
        return [{"name": value["name"], "version": value["version"]} for value in records]

    # Fixture wheels use file URLs; the separate production URL validation tests
    # ensure these sources are rejected outside this independent local resolver.
    monkeypatch.setattr(installer, "_read_uv_lock", read_fixture)
    result = installer._resolve_uv(tmp_path, [{"name": "demo-uv", "version": saved_pin}], {}, None,
                                   ["demo-uv>=1,<2"])
    assert result == [{"name": "demo-uv", "version": expected}]
    assert (tmp_path / "constraints.txt").read_text() == ("demo-uv==1.2.3\n" if saved_pin else "")


def test_uv_native_platform_wheel_rank_is_checked_independently(tmp_path):
    incompatible = row(filename="demo_uv-1.2.3-cp39-cp39-win32.whl")["wheels"][0]
    compatible = row()["wheels"][0]
    report = tmp_path / "pylock.toml"
    write_lock(report, [{**row(), "wheels": [incompatible, compatible]}])
    result = installer._read_uv_lock(report, {}, None)
    assert result[0]["url"] == compatible["url"]


@pytest.mark.parametrize("items,code", [
    ([row(), row()], "INVALID_RESOLUTION"),
    ([row(digest="not-a-digest")], "INVALID_RESOLUTION"),
    ([row(url="https://evil.example/demo_uv-1.2.3-py3-none-any.whl")], "INVALID_WHEEL"),
    ([row(url="http://files.pythonhosted.org/demo_uv-1.2.3-py3-none-any.whl")], "INVALID_WHEEL"),
    ([row(filename="another-1.2.3-py3-none-any.whl")], "INVALID_WHEEL"),
    ([row(filename="demo_uv-9.9.9-py3-none-any.whl")], "INVALID_WHEEL"),
    ([row(filename="not-a-wheel.whl")], "INVALID_WHEEL"),
    ([row(filename="demo_uv-1.2.3-cp39-cp39-win32.whl")], "INVALID_WHEEL"),
    ([{**row(), "wheels": []}], "INVALID_RESOLUTION"),
    ([{"name": "demo-uv", "version": "1.2.3"}], "INVALID_RESOLUTION"),
    ([{**row(), "directory": "../unsafe"}], "INVALID_RESOLUTION"),
    ([{**row(), "marker": 'sys_platform == "other"'}], "INVALID_RESOLUTION"),
])
def test_uv_lock_rejects_unverified_or_nonconcrete_solution(tmp_path, items, code):
    report = tmp_path / "pylock.toml"
    write_lock(report, items)
    with pytest.raises(PackageError) as error:
        installer._read_uv_lock(report, {}, None)
    assert error.value.code == code


def test_uv_lock_cannot_change_core_or_drop_saved_dependency(tmp_path):
    report = tmp_path / "pylock.toml"
    write_lock(report, [row("numpy", "9.9.9")])
    with pytest.raises(PackageError, match="fixed core"):
        installer._read_uv_lock(report, {"numpy": "2.3.3"}, None)
    write_lock(report, [row()])
    with pytest.raises(PackageError) as error:
        installer._read_uv_lock(report, {}, [{"name": "demo-uv", "version": "1.2.3"}, {"name": "missing", "version": "2.0"}])
    assert error.value.code == "PACKAGE_LOCK_MISMATCH"


def test_uv_empty_platform_solution_is_valid_and_still_checks_restore_lock(tmp_path):
    report = tmp_path / "pylock.toml"
    write_lock(report, [])
    assert installer._read_uv_lock(report, {}, None) == []
    with pytest.raises(PackageError) as error:
        installer._read_uv_lock(report, {}, [{"name": "demo-uv", "version": "1.2.3"}])
    assert error.value.code == "PACKAGE_LOCK_MISMATCH"


def test_uv_lock_file_and_graph_are_bounded(tmp_path, monkeypatch):
    report = tmp_path / "pylock.toml"
    write_lock(report, [row()])
    alias = tmp_path / "alias.toml"
    alias.symlink_to(report)
    with pytest.raises(PackageError):
        installer._read_uv_lock(alias, {}, None)
    monkeypatch.setattr(installer, "MAX_REPORT_BYTES", 10)
    with pytest.raises(PackageError):
        installer._read_uv_lock(report, {}, None)
    monkeypatch.setattr(installer, "MAX_REPORT_BYTES", 10000)
    monkeypatch.setattr(installer, "MAX_PACKAGES", 1)
    write_lock(report, [row(), row("second", "1.0"), row("third", "1.0")])
    with pytest.raises(PackageError):
        installer._read_uv_lock(report, {}, None)


def test_uv_native_child_ignores_user_configuration_and_python_selectors(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_INDEX", "https://unsafe.example/simple")
    monkeypatch.setenv("UV_PYTHON", "/unsafe/python")
    monkeypatch.setenv("UV_CONFIG_FILE", "/unsafe/uv.toml")
    monkeypatch.setenv("PYTHONHOME", "/unsafe/python-home")
    monkeypatch.setenv("VIRTUAL_ENV", "/unsafe/environment")
    calls = []
    monkeypatch.setattr(uv_runtime, "uv_executable", lambda: Path("/bundled/tools/uv"))
    monkeypatch.setattr(installer.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(returncode=0))
    installer._run_uv(tmp_path, ["compile", str(tmp_path / "input.in"), "--no-build"])
    args, options = calls[0]
    command = args[0]
    assert command[0] == "/bundled/tools/uv"
    assert command[-2:] == ["--python", sys.executable]
    for flag in ["--no-config", "--no-cache", "--no-python-downloads", "--no-managed-python"]:
        assert flag in command
    for name in ["UV_INDEX", "UV_PYTHON", "UV_CONFIG_FILE", "PYTHONHOME", "VIRTUAL_ENV"]:
        assert name not in options["env"]
    assert options["env"]["UV_PYTHON_DOWNLOADS"] == "never"
    assert options["cwd"] == tmp_path
    assert options["stdin"] is subprocess.DEVNULL
    assert not options.get("start_new_session", False), "uv must remain in the cancellable installer process group"


def test_uv_child_failure_is_transactional_package_error(tmp_path, monkeypatch):
    monkeypatch.setattr(uv_runtime, "uv_executable", lambda: Path("/bundled/tools/uv"))
    monkeypatch.setattr(installer.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=17))
    with pytest.raises(PackageError) as error:
        installer._run_uv(tmp_path, ["compile"])
    assert error.value.code == "PACKAGE_RESOLUTION_FAILED"


def test_uv_frozen_discovery_never_uses_ambient_path(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(uv_runtime.shutil, "which", lambda name: pytest.fail("Frozen builds must not search PATH"))
    with pytest.raises(PackageError) as error:
        uv_runtime.uv_executable()
    assert error.value.code == "UV_UNAVAILABLE"
    candidate = tmp_path / "tools" / ("uv.exe" if os.name == "nt" else "uv")
    candidate.parent.mkdir()
    candidate.write_bytes(b"bundled uv")
    candidate.chmod(0o700)
    assert uv_runtime.uv_executable() == candidate


def test_uv_probe_outputs_current_data_without_importing_temp_scripts(tmp_path, capsys):
    scripts = tmp_path / "python"
    scripts.mkdir()
    sentinel = tmp_path / "must-not-execute"
    (scripts / "get_interpreter_info.py").write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n")
    script = f"import sys; sys.path = [{json.dumps(str(tmp_path))}] + sys.path; from python.get_interpreter_info import main; main()"
    assert uv_runtime.maybe_uv_python_probe(["-I", "-B", "-c", script])
    info = json.loads(capsys.readouterr().out)
    assert info["result"] == "success"
    assert info["sys_executable"] == sys.executable
    assert info["markers"]["python_version"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    assert not sentinel.exists()


@pytest.mark.parametrize("arguments", [
    ["-c", "print('arbitrary code')"],
    ["-I", "-B", "-c", "import os; os.system('unsafe')"],
    ["-I", "-B", "-c", 'import sys; sys.path = [__import__("os").system("unsafe")] + sys.path; from python.get_interpreter_info import main; main()'],
    ["-I", "-B", "-c", 'import sys; sys.path = ["relative"] + sys.path; from python.get_interpreter_info import main; main()'],
    ["-I", "-B", "-c", 'import sys; sys.path = ["/tmp"] + sys.path; from python.get_interpreter_info import main; main(); print("extra")'],
])
def test_uv_probe_does_not_accept_general_python_execution(arguments, capsys):
    assert not uv_runtime.maybe_uv_python_probe(arguments)
    assert not capsys.readouterr().out


def test_real_uv_installs_verified_local_wheel_without_system_python_or_network(tmp_path, monkeypatch, native_uv):
    wheel = tmp_path / "demo_uv-1.2.3-py3-none-any.whl"
    content = wheel_bytes()
    wheel.write_bytes(content)
    lock = tmp_path / "wheel-lock.txt"
    lock.write_text(f"{wheel.as_uri()} --hash=sha256:{hashlib.sha256(content).hexdigest()}\n")
    overlay = tmp_path / "site-packages"
    overlay.mkdir()
    # Emulate the frozen entry's exact protocol. uv cannot discover another
    # interpreter: PATH is empty, Python downloads disabled, and --python points
    # to this read-only protocol executable. Its general -c support is absent.
    probe = embedded_probe_executable(tmp_path)
    monkeypatch.setattr(uv_runtime, "uv_executable", lambda: native_uv)
    monkeypatch.setattr(installer.sys, "executable", str(probe))
    monkeypatch.setenv("PATH", str(tmp_path / "no-system-python"))
    installer._install_uv(tmp_path, overlay, lock)
    assert (overlay / "demo_uv" / "__init__.py").read_text() == "VALUE = 'isolated verified wheel'\n"
    assert (overlay / "demo_uv-1.2.3.dist-info" / "METADATA").is_file()
    assert not (tmp_path / ".venv").exists()


def test_real_uv_compiles_ranges_extras_and_fixed_core_without_other_python(tmp_path, monkeypatch, native_uv):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name, version, kwargs in [
        ("demo-uv", "1.2.3", {}),
        ("demo-uv", "1.7.0", {"extras": ["addon"], "requirements": ["numpy>=2", 'dependency<2; extra == "addon"']}),
        ("demo-uv", "2.0.0", {}),
        ("dependency", "1.5.0", {}),
        ("dependency", "2.5.0", {}),
        ("numpy", "2.3.3", {}),
        ("numpy", "9.9.9", {}),
    ]:
        path = wheels / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        path.write_bytes(wheel_bytes(name, version, **kwargs))
    inputs = tmp_path / "requirements.in"
    inputs.write_text('demo-uv[addon]>=1,<2\nignored-package; python_version < "2"\n')
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy==2.3.3\n")
    probe = embedded_probe_executable(tmp_path)
    monkeypatch.setattr(uv_runtime, "uv_executable", lambda: native_uv)
    monkeypatch.setattr(installer.sys, "executable", str(probe))
    monkeypatch.setenv("PATH", str(tmp_path / "no-system-python"))
    output = tmp_path / "pylock.toml"
    installer._run_uv(tmp_path, ["compile", "--quiet", "--offline", "--no-index", "--no-build",
                                "--find-links", str(wheels), "--constraint", str(constraints),
                                "--format", "pylock.toml", "--output-file", str(output), str(inputs)])
    result = tomllib.loads(output.read_text())
    assert [(value["name"], value["version"]) for value in result["packages"]] == [
        ("demo-uv", "1.7.0"), ("dependency", "1.5.0"), ("numpy", "2.3.3")]
    assert not (tmp_path / ".venv").exists()


def test_install_uv_returns_reproducible_manifest_and_keeps_core_out_of_overlay(tmp_path, monkeypatch, native_uv):
    core = {"openecon": "0.3.6a1", "numpy": "2.3.3"}
    monkeypatch.setattr(installer, "protected_versions", lambda: core)
    monkeypatch.setattr(uv_runtime, "uv_executable", lambda: native_uv)
    stage = tmp_path / ".packages" / f"stage-{uuid4().hex}"
    stage.mkdir(parents=True)
    content = wheel_bytes()
    selected = [{"name": "demo-uv", "version": "1.2.3", "url": row()["wheels"][0]["url"],
                 "sha256": hashlib.sha256(content).hexdigest()}]
    monkeypatch.setattr(installer, "_resolve_uv", lambda *args: selected)

    def download(folder, resolved):
        destination = folder / "wheels"
        destination.mkdir()
        wheel = destination / "demo_uv-1.2.3-py3-none-any.whl"
        wheel.write_bytes(content)
        return [wheel]

    monkeypatch.setattr(installer, "_download_wheels", download)
    result = installer.install({"schema": 1, "stage": str(stage), "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                                "core": core, "requirements": [{"name": "demo-uv", "version": None}], "locked": None,
                                "installer": "uv", "specifications": ["demo-uv>=1,<2"]})
    assert result["state"] == "complete"
    assert result["manifest"]["schema"] == 2
    assert result["manifest"]["installer"] == "uv"
    assert result["manifest"]["specifications"] == ["demo-uv<2,>=1"]
    assert result["manifest"]["requirements"] == [{"name": "demo-uv", "version": "1.2.3"}]
    assert result["manifest"]["locked"] == result["manifest"]["requirements"]
    assert not (stage / "wheels").exists()
    assert not (stage / "site-packages" / "numpy").exists()
    assert json.loads((stage / "wheel-provenance.json").read_text())["wheels"] == selected


@pytest.mark.parametrize("extra", [
    {"installer": "not-an-installer"}, {"installer": ["uv"]},
    {"installer": "uv", "specifications": ["different>=1"]},
    {"installer": "uv", "specifications": ["demo-uv @ https://evil.example/package.whl"]},
])
def test_invalid_uv_actions_are_rejected_before_any_resolver(tmp_path, monkeypatch, extra):
    core = {"openecon": "0.3.6a1"}
    stage = tmp_path / ".packages" / f"stage-{uuid4().hex}"
    stage.mkdir(parents=True)
    monkeypatch.setattr(installer, "protected_versions", lambda: core)
    monkeypatch.setattr(installer, "_resolve_uv", lambda *args: pytest.fail("Invalid inputs reached uv"))
    with pytest.raises(PackageError):
        installer.install({"schema": 1, "stage": str(stage), "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                           "core": core, "requirements": [{"name": "demo-uv", "version": None}], "locked": None, **extra})
