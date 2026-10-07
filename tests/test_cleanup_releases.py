"""Release-cache cleanup preserves proof, source, and unowned local data.

Every repository and generated package in this module lives under pytest's
temporary directory. No installed application or real release cache is used.
"""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import plistlib
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "desktop/scripts/cleanup_releases.py"
CURRENT_VERSION = "0.3.18"


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, tuple[str, bytes | str]]:
    """Include symlinks without following them and ignore Git's private state."""
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            result[str(relative)] = ("symlink", str(path.readlink()))
        elif path.is_file():
            result[str(relative)] = ("file", path.read_bytes())
        elif path.is_dir():
            result[str(relative)] = ("directory", b"")
    return result


@pytest.fixture
def cleanup_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("cleanup_releases_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.real_system_guard = module._system_guard
    # Only host process/mount state and macOS signing need platform fixtures.
    # Source identity, retained-package hashes, and candidate paths stay real.
    monkeypatch.setattr(module, "_system_guard", lambda repo, releases: None)
    monkeypatch.setattr(module, "_verify_signature", lambda app: None)
    return module


@pytest.fixture(params=["OpenEcon", "OpenEconometrics"])
def repository(tmp_path, request):
    repo = tmp_path / "repository"
    repo.mkdir()
    config = repo / "desktop/src-tauri/tauri.conf.json"
    product = request.param
    write_json(config, {"productName": product, "version": CURRENT_VERSION})
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "desktop/src-tauri/tauri.conf.json"],
                   check=True, capture_output=True)
    subprocess.run([
        "git", "-C", str(repo), "-c", "user.name=Cleanup test",
        "-c", "user.email=cleanup-test@example.invalid", "commit", "-qm", "Fixture",
    ], check=True, capture_output=True)
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    releases = repo / "desktop/build/releases"
    current = releases / CURRENT_VERSION
    source = current / "source"
    write_bytes(source / "desktop/src-tauri/tauri.conf.json", config.read_bytes())
    app = current / f"target/release/bundle/macos/{product}.app"
    write_bytes(app / "Contents/Info.plist", plistlib.dumps({
        "CFBundleExecutable": "openecon-desktop",
        "CFBundleIdentifier": "org.openecon.desktop",
        "CFBundleShortVersionString": CURRENT_VERSION,
        "CFBundleVersion": CURRENT_VERSION,
    }))
    native = write_bytes(app / "Contents/MacOS/openecon-desktop", b"native fixture\n")
    native.chmod(0o755)
    runtime = write_bytes(
        app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime",
        b"runtime fixture\n",
    )
    runtime.chmod(0o755)
    engine_manifest = app / "Contents/Resources/suggestions/manifest.json"
    write_json(engine_manifest, {"engine": "llama.cpp", "files": []})
    runtime_manifest = app / "Contents/Resources/runtime/runtime-manifest.json"
    write_json(runtime_manifest, {"ui_release": {
        "desktop_version": CURRENT_VERSION,
        "source_commit": source_commit,
        "runtime_executable_sha256": digest(runtime),
        "engine_manifest_sha256": digest(engine_manifest),
    }})
    dmg = write_bytes(
        current / f"target/release/bundle/dmg/{product}_{CURRENT_VERSION}_aarch64.dmg",
        b"verified compressed disk image fixture\n",
    )
    artifacts = repo / f"desktop/artifacts/release-{CURRENT_VERSION}/build"
    validation = artifacts / "native-package-validation.json"
    write_json(validation, {
        "status": "passed",
        "source_commit": source_commit,
        "native_version": CURRENT_VERSION,
        "app": str(app),
        "dmg": str(dmg),
        "dmg_bytes": dmg.stat().st_size,
        "dmg_sha256": digest(dmg),
        "app_and_packed_app": {
            "runtime_executable_sha256": digest(runtime),
            "runtime_manifest_sha256": digest(runtime_manifest),
            "engine_manifest_sha256": digest(engine_manifest),
            "native_executable_sha256": digest(native),
            "strict_signature_verified": True,
        },
        "signature_and_resources_unchanged_by_packing": True,
        "readonly_mount_verified": True,
        "owned_mount_detached": True,
    })
    write_json(artifacts / "source-freeze.json", {
        "source_commit": source_commit,
        "source": str(source),
        "native_version": CURRENT_VERSION,
        "foreign_changes_excluded": True,
    })
    write_json(artifacts / "packaged-runtime-smoke.json", {
        "status": "ok",
        "native_version": CURRENT_VERSION,
        "source_commit": source_commit,
        "bundled": True,
        "execution": {"status": "ok", "error": None},
        "human_project_accessed": False,
    })
    return {
        "repo": repo, "releases": releases, "current": current,
        "source": source, "app": app, "dmg": dmg, "native": native,
        "runtime": runtime, "runtime_manifest": runtime_manifest,
        "engine_manifest": engine_manifest, "artifacts": artifacts,
        "validation": validation, "source_commit": source_commit, "product": product,
    }


def change_json(path: Path, **changes) -> None:
    document = json.loads(path.read_text())
    document.update(changes)
    write_json(path, document)


GENERATED_SUBPATHS = (
    "target",
    "source/desktop/runtime/openecon-runtime",
    "source/web/node_modules",
    "source/desktop/node_modules",
    "source/desktop/build/pyinstaller",
    "source/desktop/src-tauri/target",
)


def old_release(repository, module, version="0.3.17", *, marked=True):
    root = repository["releases"] / version
    for relative in GENERATED_SUBPATHS:
        write_bytes(root / relative / "generated.bin", b"owned generated cache\n")
    write_bytes(root / "source/analysis.py", b"print('preserved source')\n")
    write_bytes(root / "source/desktop/runtime/runtime-manifest.json", b"{}\n")
    write_bytes(root / "notes.json", b'{"preserve": true}\n')
    if marked:
        write_json(root / module.MARKER, {
            "format": "openecon-generated-release-cache-v1",
            "repository": str(repository["repo"].resolve()),
            "version": version,
            "source_commit": repository["source_commit"],
        })
    return root


def relative_candidate_paths(report):
    return {item["path"] for item in report["candidates"]}


def test_dry_run_lists_owned_generated_paths_without_writing_markers(
    repository, cleanup_module,
):
    old = old_release(repository, cleanup_module)
    before = snapshot(repository["repo"])

    report = cleanup_module.cleanup(repository["repo"])

    expected = {str((old / path).relative_to(repository["repo"]))
                for path in GENERATED_SUBPATHS}
    assert relative_candidate_paths(report) == expected
    assert report["current_version"] == CURRENT_VERSION
    assert report["retained_versions"] == [CURRENT_VERSION]
    assert report["applied"] is False
    assert report["deleted"] == []
    assert all(item["bytes"] > 0 for item in report["candidates"])
    assert snapshot(repository["repo"]) == before


def test_apply_only_removes_owned_cache_and_preserves_retained_packages_and_source(
    repository, cleanup_module,
):
    old = old_release(repository, cleanup_module)
    unmarked = old_release(repository, cleanup_module, "0.3.16", marked=False)
    backup = write_bytes(
        repository["releases"] / "manual-backup/source/analysis.py", b"human backup\n"
    )
    proof = write_bytes(
        repository["repo"] / "desktop/artifacts/release-0.3.17/build/proof.json",
        b'{"keep": true}\n',
    )
    unrelated = [
        write_bytes(repository["repo"] / "web/node_modules/manual.txt", b"outside\n"),
        write_bytes(repository["repo"] / "desktop/runtime/manual.txt", b"outside\n"),
    ]
    retained = snapshot(repository["current"])
    untouched = {path: path.read_bytes() for path in [backup, proof, *unrelated]}
    legacy = snapshot(unmarked)

    report = cleanup_module.cleanup(repository["repo"], apply=True)

    assert report["applied"] is True
    assert len(report["deleted"]) == len(GENERATED_SUBPATHS)
    for path in GENERATED_SUBPATHS:
        assert not (old / path).exists()
    assert (old / "source/analysis.py").read_bytes() == b"print('preserved source')\n"
    assert (old / "source/desktop/runtime/runtime-manifest.json").exists()
    assert (old / "notes.json").exists()
    assert snapshot(unmarked) == legacy
    for path, contents in untouched.items():
        assert path.read_bytes() == contents
    current_after = snapshot(repository["current"])
    current_after.pop(cleanup_module.MARKER)
    assert current_after == retained
    marker = json.loads((repository["current"] / cleanup_module.MARKER).read_text())
    assert marker == {
        "format": "openecon-generated-release-cache-v1",
        "repository": str(repository["repo"].resolve()),
        "version": CURRENT_VERSION,
        "source_commit": repository["source_commit"],
    }
    second = cleanup_module.cleanup(repository["repo"], apply=True)
    assert second["deleted"] == []
    assert repository["app"].is_dir() and repository["dmg"].is_file()


def test_semver_retention_keeps_previous_version_and_all_future_versions(
    repository, cleanup_module,
):
    previous = old_release(repository, cleanup_module, "0.3.10")
    older = old_release(repository, cleanup_module, "0.3.2-blue")
    future = [old_release(repository, cleanup_module, version)
              for version in ("0.3.19", "0.4.0-beta")]
    preserved = {root: snapshot(root) for root in [previous, *future]}

    report = cleanup_module.cleanup(repository["repo"], keep=2, apply=True)

    assert "0.3.10" in report["retained_versions"]
    assert "0.3.2-blue" not in report["retained_versions"]
    assert not (older / "target").exists()
    for root, original in preserved.items():
        assert snapshot(root) == original


@pytest.mark.parametrize("changes", [
    {"format": "unknown-format"},
    {"repository": "/another/repository"},
    {"version": "0.3.16"},
    {"source_commit": "not-a-commit"},
])
def test_invalid_old_markers_are_preserved(repository, cleanup_module, changes):
    old = old_release(repository, cleanup_module)
    change_json(old / cleanup_module.MARKER, **changes)
    before = snapshot(old)

    report = cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(old) == before
    assert report["deleted"] == []


def test_malformed_marker_and_non_version_human_directories_are_untouched(
    repository, cleanup_module,
):
    old = old_release(repository, cleanup_module)
    (old / cleanup_module.MARKER).write_text("{ malformed json")
    human = old_release(repository, cleanup_module, "before-install-backup")
    original = {root: snapshot(root) for root in [old, human]}

    cleanup_module.cleanup(repository["repo"], apply=True)

    assert all(snapshot(root) == contents for root, contents in original.items())


@pytest.mark.parametrize("bad_path", ["dmg", "native", "runtime", "runtime_manifest",
                                      "engine_manifest"])
def test_changed_retained_package_hash_blocks_all_deletions(
    repository, cleanup_module, bad_path,
):
    old_release(repository, cleanup_module)
    path = repository[bad_path]
    path.write_bytes(path.read_bytes() + b"unexpected change\n")
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("record,changes", [
    ("native-package-validation.json", {"status": "failed"}),
    ("native-package-validation.json", {"native_version": "0.3.17"}),
    ("native-package-validation.json", {"source_commit": "0" * 40}),
    ("source-freeze.json", {"native_version": "0.3.17"}),
    ("source-freeze.json", {"source_commit": "0" * 40}),
    ("packaged-runtime-smoke.json", {"status": "failed"}),
    ("packaged-runtime-smoke.json", {"native_version": "0.3.17"}),
    ("packaged-runtime-smoke.json", {"source_commit": "0" * 40}),
    ("packaged-runtime-smoke.json", {"bundled": False}),
    ("packaged-runtime-smoke.json", {"human_project_accessed": True}),
    ("packaged-runtime-smoke.json", {"execution": {"status": "error"}}),
])
def test_invalid_current_proof_blocks_all_mutations(
    repository, cleanup_module, record, changes,
):
    old_release(repository, cleanup_module)
    change_json(repository["artifacts"] / record, **changes)
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("record", ["source-freeze.json", "native-package-validation.json",
                                    "packaged-runtime-smoke.json"])
def test_missing_current_proof_blocks_all_mutations(repository, cleanup_module, record):
    old_release(repository, cleanup_module)
    (repository["artifacts"] / record).unlink()
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


def test_frozen_source_version_must_match_current_source(repository, cleanup_module):
    old_release(repository, cleanup_module)
    change_json(repository["source"] / "desktop/src-tauri/tauri.conf.json", version="0.3.17")
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


def test_app_native_version_mismatch_blocks_cleanup(repository, cleanup_module):
    old_release(repository, cleanup_module)
    info = repository["app"] / "Contents/Info.plist"
    document = plistlib.loads(info.read_bytes())
    document["CFBundleShortVersionString"] = "0.3.17"
    info.write_bytes(plistlib.dumps(document))
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


def test_runtime_source_commit_must_match_frozen_source_even_with_valid_hash(
    repository, cleanup_module,
):
    old_release(repository, cleanup_module)
    runtime_manifest = repository["runtime_manifest"]
    document = json.loads(runtime_manifest.read_text())
    document["ui_release"]["source_commit"] = "0" * 40
    write_json(runtime_manifest, document)
    proof = json.loads(repository["validation"].read_text())
    proof["app_and_packed_app"]["runtime_manifest_sha256"] = digest(runtime_manifest)
    write_json(repository["validation"], proof)
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("name", [".openecon", "user-data", "userdata", "data-root",
                                  "backups", "backup", "rollback"])
def test_user_data_in_marked_cache_blocks_global_cleanup(
    repository, cleanup_module, name,
):
    old_release(repository, cleanup_module)
    old = old_release(repository, cleanup_module, "0.3.16")
    write_bytes(old / "target" / name / "private-data", b"user analysis\n")
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("git_entry", ["directory", "worktree-pointer"])
def test_embedded_git_repository_blocks_global_cleanup(
    repository, cleanup_module, git_entry,
):
    old = old_release(repository, cleanup_module)
    dot_git = old / "target/.git"
    if git_entry == "directory":
        dot_git.mkdir()
    else:
        write_bytes(dot_git, b"gitdir: /some/private/worktree\n")
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert dot_git.exists()
    assert snapshot(repository["repo"]) == before


def test_tracked_file_in_generated_cache_blocks_global_cleanup(repository, cleanup_module):
    old = old_release(repository, cleanup_module)
    tracked = old / "target/tracked-source.txt"
    write_bytes(tracked, b"tracked and not disposable\n")
    subprocess.run(["git", "-C", str(repository["repo"]), "add",
                    str(tracked.relative_to(repository["repo"]))],
                   check=True, capture_output=True)
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("relative", ["target", "source"])
def test_candidate_symlink_or_symlink_ancestor_blocks_global_cleanup(
    repository, cleanup_module, tmp_path, relative,
):
    old = old_release(repository, cleanup_module)
    external = tmp_path / "external"
    external.mkdir()
    write_bytes(external / "web/node_modules/private-file", b"outside repository\n")
    # Rename the real path aside so the symlink cannot hide fixture data.
    candidate = old / relative
    candidate.rename(old / f"original-{relative}")
    candidate.symlink_to(external, target_is_directory=True)
    before = snapshot(repository["repo"])
    external_before = snapshot(external)

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before
    assert snapshot(external) == external_before


@pytest.mark.parametrize("guard", ["_system_guard", "_verify_signature"])
def test_active_or_mount_guard_and_signature_failure_prevent_partial_cleanup(
    repository, cleanup_module, monkeypatch, guard,
):
    old_release(repository, cleanup_module)
    before = snapshot(repository["repo"])

    def refuse(*args):
        raise cleanup_module.CleanupError("Active build/mount or failed signature")

    monkeypatch.setattr(cleanup_module, guard, refuse)
    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


def test_current_cache_removes_only_exact_old_dmg_names(repository, cleanup_module):
    directory = repository["dmg"].parent
    old = [write_bytes(directory / f"{product}_0.3.10_{arch}.dmg", b"old image\n")
           for product in ("OpenEcon", "OpenEconometrics")
           for arch in ("aarch64", "arm64", "x86_64")]
    preserve = [write_bytes(directory / name, b"human or future image\n") for name in (
        "OpenEcon_0.3.10_aarch64-copy.dmg", "OpenEcon_0.3.10_other.dmg",
        "OpenEcon_0.3.19_aarch64.dmg", "my-analysis.dmg",
        "OpenEconometrics_0.3.10_aarch64-copy.dmg", "OpenEconometrics_0.3.19_aarch64.dmg",
        "OpenEconometrics_0.3.10_other.dmg", "OpenEconometrics-custom_0.3.10_aarch64.dmg",
    )]

    report = cleanup_module.cleanup(repository["repo"], apply=True)

    assert relative_candidate_paths(report) == {
        str(path.relative_to(repository["repo"])) for path in old
    }
    assert all(not path.exists() for path in old)
    assert all(path.exists() for path in [repository["dmg"], *preserve])


def test_current_app_and_dmg_must_have_the_same_approved_product_name(repository, cleanup_module):
    other = "OpenEconometrics" if repository["product"] == "OpenEcon" else "OpenEcon"
    moved = repository["dmg"].with_name(f"{other}_{CURRENT_VERSION}_aarch64.dmg")
    repository["dmg"].rename(moved)
    change_json(repository["validation"], dmg=str(moved))
    before = snapshot(repository["repo"])
    with pytest.raises(cleanup_module.CleanupError, match="App and DMG paths"):
        cleanup_module.cleanup(repository["repo"], apply=True)
    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("product", ["OtherProduct", "../OpenEconometrics", "OpenEconometrics/other"])
def test_unknown_or_changed_frozen_product_prevents_cleanup(repository, cleanup_module, product):
    path = repository["source"] / "desktop/src-tauri/tauri.conf.json"
    change_json(path, productName=product)
    before = snapshot(repository["repo"])
    with pytest.raises(cleanup_module.CleanupError, match="product name"):
        cleanup_module.cleanup(repository["repo"], apply=True)
    assert snapshot(repository["repo"]) == before


def test_dmg_for_retained_previous_release_is_preserved(repository, cleanup_module):
    old_release(repository, cleanup_module, "0.3.17")
    directory = repository["dmg"].parent
    previous = write_bytes(directory / "OpenEcon_0.3.17_aarch64.dmg", b"rollback image\n")
    older = write_bytes(directory / "OpenEcon_0.3.16_aarch64.dmg", b"old image\n")

    cleanup_module.cleanup(repository["repo"], keep=2, apply=True)

    assert previous.exists() and repository["dmg"].exists()
    assert not older.exists()


@pytest.mark.parametrize("keep", [0, -1, True])
def test_invalid_retention_count_never_mutates(repository, cleanup_module, keep):
    old_release(repository, cleanup_module)
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], keep=keep, apply=True)

    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("record,field", [
    ("native-package-validation.json", "app"),
    ("native-package-validation.json", "dmg"),
    ("source-freeze.json", "source"),
])
def test_proof_cannot_redirect_package_or_source_to_another_release(
    repository, cleanup_module, record, field,
):
    old = old_release(repository, cleanup_module)
    change_json(repository["artifacts"] / record, **{field: str(old / field)})
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


def test_retained_proof_path_cannot_be_an_external_symlink(
    repository, cleanup_module, tmp_path,
):
    old_release(repository, cleanup_module)
    proof = repository["validation"]
    external = tmp_path / "external-proof.json"
    external.write_bytes(proof.read_bytes())
    proof.unlink()
    proof.symlink_to(external)
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before


def test_release_root_cannot_be_an_external_symlink(repository, cleanup_module, tmp_path):
    old = old_release(repository, cleanup_module)
    external = tmp_path / "external-release"
    old.rename(external)
    old.symlink_to(external, target_is_directory=True)
    before = snapshot(repository["repo"])
    external_before = snapshot(external)

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before
    assert snapshot(external) == external_before


def test_current_marker_symlink_is_checked_before_deletion(
    repository, cleanup_module, tmp_path,
):
    old_release(repository, cleanup_module)
    external = write_bytes(tmp_path / "human-marker", b"human file\n")
    (repository["current"] / cleanup_module.MARKER).symlink_to(external)
    before = snapshot(repository["repo"])

    with pytest.raises((cleanup_module.CleanupError, OSError)):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before
    assert external.read_bytes() == b"human file\n"


@pytest.mark.parametrize("marker", [".build-in-progress", ".active-build", ".build.lock"])
def test_real_active_build_markers_refuse_cleanup(repository, cleanup_module, monkeypatch, marker):
    old = old_release(repository, cleanup_module)
    write_bytes(old / marker, b"active fixture\n")
    monkeypatch.setattr(cleanup_module, "_system_guard", cleanup_module.real_system_guard)
    monkeypatch.setattr(cleanup_module.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, stdout="", stderr=""))
    before = snapshot(repository["repo"])

    # Call the real host guard directly: Git subprocesses remain untouched in
    # cleanup tests, and only the OS process-list read is faked here.
    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module._system_guard(repository["repo"], repository["releases"])

    assert snapshot(repository["repo"]) == before


def test_real_mount_guard_refuses_mounted_release_image(
    repository, cleanup_module, monkeypatch,
):
    mounted = plistlib.dumps({"images": [{"image-path": str(repository["dmg"])}]})

    def fake_run(command, **kwargs):
        output = "" if command[0] == "ps" else mounted
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(cleanup_module.sys, "platform", "darwin")
    monkeypatch.setattr(cleanup_module.subprocess, "run", fake_run)
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.real_system_guard(repository["repo"], repository["releases"])

    assert snapshot(repository["repo"]) == before


@pytest.mark.parametrize("command", [
    "python desktop/scripts/build_desktop.py --bundles app,dmg",
    "cargo build --release",
    "python -m PyInstaller runtime.spec",
])
def test_real_process_guard_refuses_active_builds(
    repository, cleanup_module, monkeypatch, command,
):
    monkeypatch.setattr(cleanup_module.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(
                            args[0], 0, stdout="999999 " + command + "\n", stderr=""))
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.real_system_guard(repository["repo"], repository["releases"])

    assert snapshot(repository["repo"]) == before


def test_existing_cleanup_lock_is_preserved_and_blocks_mutation(repository, cleanup_module):
    old_release(repository, cleanup_module)
    lock = write_bytes(repository["releases"] / ".cleanup.lock", b"another cleanup\n")
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before
    assert lock.read_bytes() == b"another cleanup\n"


def test_shared_node_modules_leaf_links_and_targets_are_preserved(
    repository, cleanup_module,
):
    old = old_release(repository, cleanup_module)
    links = []
    shared = []
    for source_relative, shared_relative in (
        ("source/web/node_modules", "web/node_modules"),
        ("source/desktop/node_modules", "desktop/node_modules"),
    ):
        target = repository["repo"] / shared_relative
        write_bytes(target / "human-installed-package/package.json", b'{"private": true}\n')
        link = old / source_relative
        link.rename(link.with_name("preserved-original-node_modules"))
        link.symlink_to(target, target_is_directory=True)
        links.append(link)
        shared.append((target, snapshot(target)))

    report = cleanup_module.cleanup(repository["repo"], apply=True)

    assert len(report["deleted"]) == len(GENERATED_SUBPATHS) - len(links)
    assert all(link.is_symlink() for link in links)
    assert all(str(link.relative_to(repository["repo"])) not in relative_candidate_paths(report)
               for link in links)
    assert all(snapshot(target) == original for target, original in shared)


def test_proof_path_with_parent_traversal_is_refused_before_deletion(
    repository, cleanup_module,
):
    old_release(repository, cleanup_module)
    change_json(repository["artifacts"] / "source-freeze.json",
                source=str(repository["source"] / ".." / "source"))
    before = snapshot(repository["repo"])

    with pytest.raises(cleanup_module.CleanupError):
        cleanup_module.cleanup(repository["repo"], apply=True)

    assert snapshot(repository["repo"]) == before
