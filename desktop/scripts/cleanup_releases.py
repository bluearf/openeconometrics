"""Prune generated release caches only after a verified installation.

Dry-run is the default. This script never removes source text, verification
records, application backups or the installed application.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys

MARKER = ".openecon-generated-cache.json"
FORMAT = "openecon-generated-release-cache-v1"
VERSION = r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([A-Za-z0-9.-]+))?(?:\+([A-Za-z0-9.-]+))?"
GENERATED = (
    "target",
    "source/desktop/runtime/openecon-runtime",
    "source/web/node_modules",
    "source/desktop/node_modules",
    "source/desktop/build/pyinstaller",
    "source/desktop/build/pyinstaller-config",
    "source/desktop/src-tauri/target",
)
PROTECTED = {".git", ".openecon", "user-data", "userdata", "data-root", "backups", "backup", "rollback"}
BUNDLE_PRODUCTS = {"OpenEconometrics", "OpenEcon"}
DMG_NAME = r"(OpenEconometrics|OpenEcon)_(.+)_(aarch64|arm64|x86_64)\.dmg"


class CleanupError(RuntimeError):
    """A safety or verification prerequisite was not satisfied."""


def _version(value):
    match = re.fullmatch(VERSION, value)
    if not match:
        raise CleanupError("Invalid release version.")
    prerelease, build = match.group(4), match.group(5)
    for identifiers in (prerelease, build):
        if identifiers is not None and any(not item for item in identifiers.split(".")):
            raise CleanupError("Invalid release version.")
    if prerelease and any(item.isdigit() and len(item) > 1 and item.startswith("0") for item in prerelease.split(".")):
        raise CleanupError("Invalid release version.")
    suffix = tuple((0, int(item)) if item.isdigit() else (1, item) for item in (prerelease or "").split("."))
    return (*map(int, match.group(1, 2, 3)), prerelease is None, suffix)


def _git(repo, *arguments):
    result = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True)
    if result.returncode:
        raise CleanupError("The canonical Git repository could not be verified.")
    return result.stdout


def _safe_path(repo, path):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise CleanupError("A path contains a parent-directory component.")
    try:
        parts = path.relative_to(repo).parts
    except ValueError:
        raise CleanupError("A path leaves the canonical repository.") from None
    current = repo
    for part in parts:
        current /= part
        if current.is_symlink():
            raise CleanupError("A cache or verification path is a symbolic link.")
    return path


def _json(repo, path):
    try:
        value = json.loads(_safe_path(repo, path).read_text())
    except (OSError, ValueError):
        raise CleanupError("A required verification record is missing or invalid.") from None
    if not isinstance(value, dict):
        raise CleanupError("A verification record must be an object.")
    return value


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_signature(app):
    if sys.platform != "darwin":
        raise CleanupError("This verified App/DMG cleanup currently requires macOS.")
    result = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True)
    if result.returncode:
        raise CleanupError("The retained application's signature is invalid.")


def _system_guard(repo, releases):
    result = subprocess.run(["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True)
    if result.returncode:
        raise CleanupError("Active builds could not be checked.")
    build = re.compile(r"(?:^|[/\s])(?:build_desktop|package_runtime|package_macos)\.py(?:\s|$)|(?:^|[/\s])(?:cargo|tauri)\s+build(?:\s|$)|\bPyInstaller\b")
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, _, command = line.strip().partition(" ")
        if pid != str(os.getpid()) and (build.search(command) or str(releases) + os.sep in command):
            raise CleanupError("A build or application using the release cache is active.")
    for directory in (releases, *releases.iterdir()):
        if directory.is_dir() and any((directory / name).exists() for name in (".build-in-progress", ".active-build", ".build.lock")):
            raise CleanupError("A release build marker is active.")
    if sys.platform == "darwin":
        mounted = subprocess.run(["hdiutil", "info", "-plist"], capture_output=True)
        if mounted.returncode:
            raise CleanupError("Mounted disk images could not be checked.")
        try:
            images = plistlib.loads(mounted.stdout)["images"]
            for image in images:
                if Path(image["image-path"]).resolve().is_relative_to(releases):
                    raise CleanupError("A release disk image is still mounted.")
        except (KeyError, ValueError, plistlib.InvalidFileException):
            raise CleanupError("Mounted disk image information is invalid.") from None


def _verify_retained(repo, releases, version):
    current = _safe_path(repo, releases / version)
    records = repo / "desktop/artifacts" / ("release-" + version) / "build"
    proof = _json(repo, records / "native-package-validation.json")
    freeze = _json(repo, records / "source-freeze.json")
    smoke = _json(repo, records / "packaged-runtime-smoke.json")
    commit = proof.get("source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CleanupError("The retained release has no exact source commit.")
    frozen_config = json.loads(_git(repo, "show", commit + ":desktop/src-tauri/tauri.conf.json"))
    if not all(record.get("source_commit") == commit and record.get("native_version") == version for record in (proof, freeze, smoke)) or frozen_config.get("version") != version:
        raise CleanupError("Source and native release verification disagree.")
    if proof.get("status") != "passed" or smoke.get("status") != "ok" or smoke.get("bundled") is not True or smoke.get("execution", {}).get("status") != "ok" or smoke.get("human_project_accessed") is not False:
        raise CleanupError("The retained package and isolated runtime smoke test must pass.")
    if _safe_path(repo, freeze.get("source", "")) != current / "source":
        raise CleanupError("The verified frozen source belongs to a different cache.")
    source_config = _json(repo, current / "source/desktop/src-tauri/tauri.conf.json")
    product = frozen_config.get("productName")
    if product not in BUNDLE_PRODUCTS or source_config.get("productName") != product:
        raise CleanupError("The retained frozen source has an unknown or different product name.")
    if source_config.get("version") != version:
        raise CleanupError("The retained frozen source has a different version.")
    app = current / f"target/release/bundle/macos/{product}.app"
    dmg = _safe_path(repo, Path(proof.get("dmg", "")))
    expected_dmg = re.fullmatch(DMG_NAME, dmg.name)
    if _safe_path(repo, Path(proof.get("app", ""))) != app or dmg.parent != current / "target/release/bundle/dmg" or not expected_dmg or expected_dmg.group(1) != product or expected_dmg.group(2) != version:
        raise CleanupError("The retained App and DMG paths do not belong to the current release.")
    info = plistlib.loads(_safe_path(repo, app / "Contents/Info.plist").read_bytes())
    if info.get("CFBundleVersion") != version or info.get("CFBundleShortVersionString") != version:
        raise CleanupError("The retained application has a different native version.")
    resources = app / "Contents/Resources"
    hashes = proof.get("app_and_packed_app", {})
    expected = {
        "native_executable_sha256": app / "Contents/MacOS/openecon-desktop",
        "runtime_executable_sha256": resources / "runtime/openecon-runtime/openecon-runtime",
        "runtime_manifest_sha256": resources / "runtime/runtime-manifest.json",
        "engine_manifest_sha256": resources / "suggestions/manifest.json",
    }
    for key, path in expected.items():
        if not re.fullmatch(r"[0-9a-f]{64}", hashes.get(key, "")) or _sha(_safe_path(repo, path)) != hashes[key]:
            raise CleanupError("A retained application payload differs from its verification record.")
    runtime = _json(repo, resources / "runtime/runtime-manifest.json")
    if runtime.get("ui_release", {}).get("source_commit") != commit:
        raise CleanupError("The retained runtime belongs to a different source commit.")
    if hashes.get("strict_signature_verified") is not True or not all(proof.get(key) is True for key in ("signature_and_resources_unchanged_by_packing", "readonly_mount_verified", "owned_mount_detached")):
        raise CleanupError("Package signature and read-only installer verification are required.")
    if dmg.stat().st_size != proof.get("dmg_bytes") or _sha(dmg) != proof.get("dmg_sha256"):
        raise CleanupError("The retained DMG differs from its verification record.")
    _verify_signature(app)
    return commit, dmg


def _candidate(repo, path):
    path = _safe_path(repo, path)
    relative = path.relative_to(repo)
    if _git(repo, "ls-files", "-z", "--", str(relative)):
        raise CleanupError("A cleanup candidate contains tracked source files.")
    size = 0
    entries = [path]
    if path.is_dir():
        for parent, directories, files in os.walk(path, followlinks=False):
            entries.extend(Path(parent) / name for name in directories + files)
    for entry in entries:
        name = entry.name.casefold()
        if name in PROTECTED or name.endswith((".bak", ".backup")):
            raise CleanupError("A cleanup candidate contains user data, a repository or a backup.")
        if entry.is_symlink():
            if not entry.resolve().is_relative_to(path):
                raise CleanupError("A generated tree contains an external symbolic link.")
        elif entry.is_file():
            size += entry.stat().st_size
        elif not entry.is_dir():
            raise CleanupError("A generated tree contains a special file.")
    return {"path": str(relative), "bytes": size, "kind": "directory" if path.is_dir() else "file"}


def _plan(repo, keep):
    repo = Path(repo).absolute()
    if repo != repo.resolve() or Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
        raise CleanupError("Cleanup must run from the canonical repository.")
    if type(keep) is not int or keep < 1:
        raise CleanupError("Keep must be a positive release count.")
    version = _json(repo, repo / "desktop/src-tauri/tauri.conf.json")["version"]
    current_key = _version(version)
    releases = _safe_path(repo, repo / "desktop/build/releases")
    _system_guard(repo, releases)
    commit, dmg = _verify_retained(repo, releases, version)
    current_marker = _safe_path(repo, releases / version / MARKER)
    if _git(repo, "ls-files", "-z", "--", str(current_marker.relative_to(repo))):
        raise CleanupError("The generated-cache marker is a tracked source file.")
    if current_marker.exists():
        existing = _json(repo, current_marker)
        if existing.get("format") != FORMAT or existing.get("repository") != str(repo) or existing.get("version") != version or not re.fullmatch(r"[0-9a-f]{40}", existing.get("source_commit", "")):
            raise CleanupError("The current generated-cache marker is invalid.")
    eligible, skipped = [], []
    for directory in releases.iterdir():
        try:
            key = _version(directory.name)
        except CleanupError:
            continue
        _safe_path(repo, directory)
        if not directory.is_dir() or key >= current_key:
            continue
        if not (directory / MARKER).exists():
            skipped.append(directory.name)
            continue
        try:
            marker = _json(repo, directory / MARKER)
        except CleanupError:
            skipped.append(directory.name)
            continue
        if marker.get("format") != FORMAT or marker.get("repository") != str(repo) or marker.get("version") != directory.name or not re.fullmatch(r"[0-9a-f]{40}", marker.get("source_commit", "")):
            skipped.append(directory.name)
            continue
        eligible.append((key, directory))
    ordered = sorted(eligible, key=lambda row: row[0], reverse=True)
    retained = [version, *(directory.name for _, directory in ordered[:keep - 1])]
    candidates = []
    for _, directory in ordered[keep - 1:]:
        for relative in GENERATED:
            path = directory / relative
            if relative.endswith("/node_modules"):
                _safe_path(repo, path.parent)
                if path.is_symlink():
                    # Frozen source uses tiny links to the shared dependency
                    # cache. Keep the link and its shared target untouched.
                    continue
            if path.exists() or path.is_symlink():
                candidates.append(_candidate(repo, path))
    for retained_version in retained:
        images = releases / retained_version / "target/release/bundle/dmg"
        _safe_path(repo, images)
        if images.is_dir():
            for image in images.iterdir():
                match = re.fullmatch(DMG_NAME, image.name)
                if match:
                    try:
                        old = _version(match.group(2)) < current_key
                    except CleanupError:
                        continue
                    if old and match.group(2) not in retained and image != dmg:
                        candidates.append(_candidate(repo, image))
    return repo, releases, commit, {"current_version": version, "retained_versions": retained, "candidates": candidates, "skipped_versions": sorted(skipped), "applied": False, "deleted": []}


@contextmanager
def _parent_fd(repo, path):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(repo, flags)
    try:
        for part in path.parent.relative_to(repo).parts:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def cleanup(repo: Path, *, keep: int = 1, apply: bool = False) -> dict:
    repo, releases, commit, plan = _plan(repo, keep)
    if not apply:
        return plan
    if not shutil.rmtree.avoids_symlink_attacks:
        raise CleanupError("Safe directory deletion is unavailable on this platform.")
    lock = releases / ".cleanup.lock"
    descriptor = None
    with _parent_fd(repo, lock) as parent:
        try:
            descriptor = os.open(lock.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            # Revalidate artifacts and all candidates under the exclusive cleanup
            # lock; an old preview is never authorization for a later deletion.
            _, _, commit, plan = _plan(repo, keep)
            for record in plan["candidates"]:
                path = repo / record["path"]
                _system_guard(repo, releases)
                _candidate(repo, path)
                with _parent_fd(repo, path) as candidate_parent:
                    if record["kind"] == "directory":
                        shutil.rmtree(path.name, dir_fd=candidate_parent)
                    else:
                        os.unlink(path.name, dir_fd=candidate_parent)
                plan["deleted"].append(record["path"])
            marker = releases / plan["current_version"] / MARKER
            body = {"format": FORMAT, "repository": str(repo), "version": plan["current_version"], "source_commit": commit}
            with _parent_fd(repo, marker) as marker_parent:
                marker_fd = os.open(marker.name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600, dir_fd=marker_parent)
                with os.fdopen(marker_fd, "w") as output:
                    output.write(json.dumps(body, indent=2) + "\n")
            plan["applied"] = True
            return plan
        except FileExistsError:
            raise CleanupError("Another cleanup is active.") from None
        finally:
            if descriptor is not None:
                if os.stat(lock.name, dir_fd=parent, follow_symlinks=False).st_ino == os.fstat(descriptor).st_ino:
                    os.unlink(lock.name, dir_fd=parent)
                os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete only verified generated-cache candidates.")
    parser.add_argument("--keep", type=int, default=1, help="Full release caches to retain, including the current release (default: 1).")
    args = parser.parse_args()
    try:
        report = cleanup(Path(__file__).resolve().parents[2], keep=args.keep, apply=args.apply)
    except (CleanupError, OSError, ValueError) as error:
        parser.exit(1, "Cleanup refused: " + str(error) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
