"""Independently constructed Mach-O metadata and relocated payload fixtures."""

import importlib.util
from pathlib import Path
import plistlib
import struct

import pytest


spec = importlib.util.spec_from_file_location(
    "installer_audit", Path(__file__).parents[1] / "desktop/scripts/macos_contract.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def binary(minimum=0x000F0000):
    # arm64 MH_EXECUTE with one 24-byte LC_BUILD_VERSION for macOS.
    return struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 2, 1, 24, 0, 0) + \
        struct.pack("<IIIIII", 0x32, 24, 1, minimum, 0x001A0000, 0)


def test_thin_universal_and_legacy_minimum_versions():
    data = binary()
    assert module.macho_minimums(data) == [
        {"cpu": 0x0100000C, "platform": 1, "minimum": "15.0.0", "sdk": "26.0.0"}]
    fat = struct.pack(">II", 0xCAFEBABE, 1) + struct.pack(">IIIII", 0x0100000C, 0, 28, len(data), 0) + data
    assert module.macho_minimums(fat) == module.macho_minimums(data)
    legacy = struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 2, 1, 16, 0, 0) + \
        struct.pack("<IIII", 0x24, 16, 0x000E0100, 0x000F0000)
    assert module.macho_minimums(legacy)[0]["minimum"] == "14.1.0"
    assert module.macho_minimums(b"ordinary data") == []


@pytest.mark.parametrize("data", [binary()[:-1], binary()[:36] + struct.pack("<I", 128) + binary()[40:]])
def test_truncated_load_commands_fail_closed(data):
    with pytest.raises(ValueError):
        module.macho_minimums(data)


def test_relocated_payload_hash_and_minimum_os_violation(tmp_path):
    import shutil

    app = tmp_path / "original.app"
    content = app / "Contents"
    content.mkdir(parents=True)
    (content / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": "org.openecon.qa.fixture", "CFBundleShortVersionString": "0.3.43",
        "LSMinimumSystemVersion": "15.0"}))
    (content / "engine").write_bytes(binary())
    (content / "alias").symlink_to("engine")
    before = module.audit(app)
    other = tmp_path / "relocated.app"
    shutil.copytree(app, other, symlinks=True)
    assert module.audit(other)["payload_sha256"] == before["payload_sha256"]
    assert before["minimum_os_violations"] == []
    (other / "Contents/engine").write_bytes(binary(0x00100000))
    assert module.audit(other)["minimum_os_violations"] == ["Contents/engine"]
    assert module.audit(other)["payload_sha256"] != before["payload_sha256"]


@pytest.mark.parametrize("absolute", [False, True])
def test_alias_cannot_escape_bundle_or_pin_install_location(tmp_path, absolute):
    content = tmp_path / "bad.app/Contents"
    content.mkdir(parents=True)
    (content / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": "org.openecon.qa.fixture", "CFBundleShortVersionString": "0.3.43",
        "LSMinimumSystemVersion": "15.0"}))
    (tmp_path / "outside").write_bytes(b"outside")
    (content / "engine").write_bytes(binary())
    (content / "escape").symlink_to(content / "engine" if absolute else "../../outside")
    with pytest.raises(ValueError):
        module.audit(content.parent)


def test_finalizer_rejects_newer_library_before_installer_creation(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "package_runtime", SimpleNamespace(
        optimize_runtime=lambda _: {}, prepare_distribution_notices=lambda _: None))
    monkeypatch.setitem(sys.modules, "macos_contract", module)
    source = Path(__file__).parents[1] / "desktop/scripts/package_macos.py"
    package_spec = importlib.util.spec_from_file_location("macos_guard_fixture", source)
    package = importlib.util.module_from_spec(package_spec)
    package_spec.loader.exec_module(package)
    monkeypatch.setattr(package.sys, "platform", "darwin")
    prepared = tmp_path / "desktop/suggestions"
    prepared.mkdir(parents=True)
    (prepared / "manifest.json").write_text(json.dumps({"files": []}))
    monkeypatch.setattr(package, "DESKTOP", prepared.parent)
    commands = []
    monkeypatch.setattr(package.subprocess, "run", lambda command, **_: commands.append(command))
    app = tmp_path / "OpenEconometrics.app"
    content = app / "Contents"
    runtime = content / "Resources/runtime"
    runtime.mkdir(parents=True)
    (runtime / "runtime-manifest.json").write_text("{}")
    (content / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": "org.openecon.qa.fixture", "CFBundleShortVersionString": "0.3.43",
        "LSMinimumSystemVersion": "15.0"}))
    (runtime / "engine").write_bytes(binary(0x001A0000))
    with pytest.raises(RuntimeError, match="Bundled binaries require a newer macOS"):
        package.package(app, tmp_path / "invalid.dmg")
    assert not (tmp_path / "invalid.dmg").exists()
    assert not any(command[0] == "hdiutil" for command in commands)
