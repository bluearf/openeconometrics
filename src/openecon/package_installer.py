"""Private wheel-only installer entry point for an owned project subprocess.

pip runs in fresh bundled-runtime subprocesses; uv uses its bundled native
executable and queries only embedded Python. Both resolvers ignore user settings,
obey exact base-runtime constraints, and install verified wheels into staging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4
import zipfile

from openecon.project_packages import (
    MAX_FILES, MAX_OVERLAY_BYTES, MAX_PACKAGES, PackageError, _records, _write_json,
    canonical_name, canonical_version, installer_command, installer_environment,
    protected_versions, validate_manifest, validate_overlay,
)

MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
INDEX_URL = "https://pypi.org/simple"
WHEEL_HOST = "files.pythonhosted.org"
_SOURCE_WHEELS = {}
_SOURCE_POLICY = {}
_SOURCE_ROOT = None


def _source_roots(specifications):
    from packaging.requirements import Requirement
    result=[]
    for specification in specifications:
        requirement=Requirement(specification)
        name=canonical_name(requirement.name)
        if name in _SOURCE_WHEELS:
            extras="["+",".join(sorted(requirement.extras))+"]" if requirement.extras else ""
            result.append(name+extras+" @ "+_SOURCE_WHEELS[name])
        else:
            result.append(specification)
    return result


def _safe_wheel_url(value: str) -> str:
    if not isinstance(value, str) or len(value) > 3000:
        raise PackageError("INVALID_WHEEL", "The resolver returned an invalid wheel location.")
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise PackageError("INVALID_WHEEL", "The resolver returned an invalid wheel location.") from exc
    name = unquote(PurePosixPath(parsed.path).name)
    if parsed.scheme == "file" and _SOURCE_ROOT is not None:
        from urllib.request import url2pathname
        candidate = Path(url2pathname(parsed.path))
        if (parsed.netloc not in {"", "localhost"} or candidate.parent != _SOURCE_ROOT
                or candidate.is_symlink() or not candidate.is_file() or candidate.suffix != ".whl"):
            raise PackageError("INVALID_WHEEL", "Only this operation's verified source wheels are allowed.")
        return candidate.name
    hosts={WHEEL_HOST, *_SOURCE_POLICY.get("wheel_hosts", [])}
    if _SOURCE_POLICY.get("index_url"):
        hosts.add(urlparse(_SOURCE_POLICY["index_url"]).hostname)
    if (parsed.scheme != "https" or parsed.hostname not in hosts or port not in {None, 443}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not name.endswith(".whl") or "/" in name or "\\" in name or ".." in name):
        raise PackageError("INVALID_WHEEL", "Wheel downloads must retain an explicitly allowed HTTPS source host.")
    return name


def _run_pip(stage: Path, arguments: list[str]) -> None:
    action = stage / f"pip-{uuid4().hex}.json"
    _write_json(action, {"operation": "pip", "arguments": arguments})
    environment=installer_environment()
    if (stage/"source-policy.json").exists():
        scratch=stage/"installer-scratch"
        scratch.mkdir(mode=0o700,exist_ok=True)
        environment.update(TMPDIR=str(scratch),TEMP=str(scratch),TMP=str(scratch))
    process = subprocess.run(installer_command(action), stdin=subprocess.DEVNULL,
                             env=environment, check=False,
                             **({"stdout":subprocess.DEVNULL,"stderr":subprocess.DEVNULL} if (stage/"source-policy.json").exists() else {}))
    if process.returncode:
        raise PackageError("PACKAGE_RESOLUTION_FAILED", "Compatible wheels could not be resolved or installed; see the package log.")


def _resolve(stage: Path, requests: list[dict], core: dict, locked: list[dict] | None,
             specifications: list[str] | None = None) -> list[dict]:
    if not requests and not specifications:
        return []
    constraints = stage / "constraints.txt"
    rows = {**core, **{row["name"]: row["version"] for row in requests if row["version"] is not None},
            **({row["name"]: row["version"] for row in locked} if locked is not None else {})}
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(rows.items())), encoding="utf-8")
    report = stage / "resolve.json"
    roots = specifications if specifications is not None else [
        row["name"] + (f"=={row['version']}" if row["version"] is not None else "") for row in requests]
    roots = _source_roots(roots)
    print("Resolving compatible wheels", flush=True)
    _run_pip(stage, ["--isolated", "--disable-pip-version-check", "--no-input", "install",
                     "--dry-run", "--ignore-installed", "--only-binary=:all:", "--no-cache-dir",
                     "--progress-bar", "off", "--timeout", "15", "--retries", "1",
                     "--index-url", _SOURCE_POLICY.get("index_url", INDEX_URL), "--constraint", str(constraints),
                     "--report", str(report), *roots])
    if report.is_symlink() or not report.is_file() or report.stat().st_size > MAX_REPORT_BYTES:
        raise PackageError("INVALID_RESOLUTION", "The package resolution report is invalid.")
    try:
        result = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise PackageError("INVALID_RESOLUTION", "The package resolution report could not be read.") from exc
    selected, seen = [], set()
    items = result.get("install")
    if not isinstance(items, list) or len(items) > 2 * MAX_PACKAGES:
        raise PackageError("PACKAGE_LIMIT", "The dependency graph exceeds the package limit.")
    for item in items:
        try:
            name = canonical_name(item["metadata"]["name"])
            version = canonical_version(item["metadata"]["version"])
            url = item["download_info"]["url"]
            digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
        except (KeyError, TypeError) as exc:
            raise PackageError("INVALID_RESOLUTION", "Every selected package needs a wheel and SHA-256 digest.") from exc
        if name in seen or not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            raise PackageError("INVALID_RESOLUTION", "The resolver returned duplicate or invalid packages.")
        seen.add(name)
        _safe_wheel_url(url)
        if name in core:
            if version != core[name]:
                raise PackageError("PROTECTED_PACKAGE", "A dependency would change OpenEconometrics's fixed core.")
            continue
        selected.append({"name": name, "version": version, "url": url, "sha256": digest.lower()})
    if len(selected) > MAX_PACKAGES:
        raise PackageError("PACKAGE_LIMIT", "Additional packages support at most 100 distributions.")
    selected.sort(key=lambda row: row["name"])
    if locked is not None and [{"name": row["name"], "version": row["version"]} for row in selected] != locked:
        raise PackageError("PACKAGE_LOCK_MISMATCH", "The platform's resolved packages differ from this saved manifest.")
    return selected


def _run_uv(stage: Path, arguments: list[str]) -> None:
    from openecon.uv_runtime import uv_environment, uv_executable
    # The chosen Python is always this runtime. Frozen desktop entry answers
    # uv's narrow, read-only discovery query without executing supplied code.
    command = [str(uv_executable()), "--no-config", "--no-cache", "--no-python-downloads",
               "--no-managed-python", "--color", "never", "--no-progress", "pip", *arguments,
               "--python", sys.executable]
    process = subprocess.run(command, stdin=subprocess.DEVNULL, env=uv_environment(),
                             cwd=stage, check=False,
                             **({"stdout":subprocess.DEVNULL,"stderr":subprocess.DEVNULL} if _SOURCE_POLICY else {}))
    if process.returncode:
        raise PackageError("PACKAGE_RESOLUTION_FAILED",
                           "Compatible wheels could not be resolved or installed with uv; see the package log.")


def _read_uv_lock(path: Path, core: dict, locked: list[dict] | None) -> list[dict]:
    """Accept a concrete PEP 751 solution and independently verify wheel identity."""
    from packaging.tags import sys_tags
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from packaging.version import Version
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise PackageError("INVALID_RESOLUTION", "The uv package lock is invalid.")
    try:
        result = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise PackageError("INVALID_RESOLUTION", "The uv package lock could not be read.") from exc
    # uv omits the packages table when every requested platform marker is false.
    items = result.get("packages", [])
    if result.get("lock-version") != "1.0" or not isinstance(items, list) or len(items) > 2 * MAX_PACKAGES:
        raise PackageError("INVALID_RESOLUTION", "The uv package lock must contain a bounded concrete wheel solution.")
    tag_order = {tag: index for index, tag in enumerate(sys_tags())}
    selected, seen = [], set()
    for item in items:
        if isinstance(item,dict) and "archive" in item and _SOURCE_ROOT is not None:
            archive=item["archive"]
            location=archive.get("url") or archive.get("path")
            if isinstance(location,str) and not location.startswith("file:"):
                candidate=Path(location)
                location=(candidate if candidate.is_absolute() else path.parent/candidate).resolve().as_uri()
            _safe_wheel_url(location)
            item={**item,"wheels":[{"url":location,"hashes":archive.get("hashes",{})}]}
            del item["archive"]
        if not isinstance(item, dict) or any(key in item for key in ("directory", "vcs", "archive", "marker")):
            raise PackageError("INVALID_RESOLUTION", "Only concrete PyPI wheel packages are supported.")
        try:
            name, version = canonical_name(item["name"]), canonical_version(item["version"])
            wheels = item["wheels"]
        except (KeyError, TypeError) as exc:
            raise PackageError("INVALID_RESOLUTION", "Every uv package needs a version and compatible wheel.") from exc
        if name in seen or not isinstance(wheels, list) or not wheels or len(wheels) > MAX_FILES:
            raise PackageError("INVALID_RESOLUTION", "The uv resolver returned duplicate or invalid packages.")
        seen.add(name)
        candidates = []
        for wheel in wheels:
            try:
                url, digest = wheel["url"], wheel["hashes"]["sha256"]
            except (KeyError, TypeError) as exc:
                raise PackageError("INVALID_RESOLUTION", "Every uv wheel needs a URL and SHA-256 digest.") from exc
            filename = _safe_wheel_url(url)
            if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                raise PackageError("INVALID_RESOLUTION", "The uv resolver returned an invalid wheel digest.")
            try:
                wheel_name, wheel_version, _, tags = parse_wheel_filename(filename)
            except InvalidWheelFilename as exc:
                raise PackageError("INVALID_WHEEL", "The uv resolver returned an invalid wheel filename.") from exc
            if wheel_name != name or wheel_version != Version(version):
                raise PackageError("INVALID_WHEEL", "A uv wheel does not match its locked package identity.")
            matches = [tag_order[tag] for tag in tags if tag in tag_order]
            if matches:
                candidates.append((min(matches), filename, url, digest.lower()))
        if name in core:
            if version != core[name]:
                raise PackageError("PROTECTED_PACKAGE", "A dependency would change OpenEconometrics's fixed core.")
            continue
        if not candidates:
            raise PackageError("INVALID_WHEEL", "A resolved package has no wheel compatible with embedded Python.")
        _, _, url, digest = min(candidates)
        selected.append({"name": name, "version": version, "url": url, "sha256": digest})
    if len(selected) > MAX_PACKAGES:
        raise PackageError("PACKAGE_LIMIT", "Additional packages support at most 100 distributions.")
    selected.sort(key=lambda row: row["name"])
    if locked is not None and [{"name": row["name"], "version": row["version"]} for row in selected] != locked:
        raise PackageError("PACKAGE_LOCK_MISMATCH", "The platform's uv packages differ from this saved manifest.")
    return selected


def _resolve_uv(stage: Path, requests: list[dict], core: dict, locked: list[dict] | None,
                specifications: list[str]) -> list[dict]:
    if not requests and not specifications:
        return []
    constraints = stage / "constraints.txt"
    rows = {**core, **{row["name"]: row["version"] for row in requests if row["version"] is not None},
            **({row["name"]: row["version"] for row in locked} if locked is not None else {})}
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(rows.items())), encoding="utf-8")
    inputs = stage / "requirements.in"
    inputs.write_text("".join(f"{specification}\n" for specification in _source_roots(specifications)), encoding="utf-8")
    report = stage / "pylock.toml"
    print("Resolving compatible wheels with uv", flush=True)
    # uv 0.9.26 splits requirement-file arguments on whitespace, even when
    # subprocess receives one argv element. The macOS Application Support
    # directory always contains a space. Our generated files live in cwd;
    # use their fixed relative names rather than absolute profile paths.
    _run_uv(stage, ["compile", "--quiet", "--format", "pylock.toml", "--output-file", report.name,
                    "--no-header", "--no-annotate", "--no-build", "--no-sources",
                    "--keyring-provider", "disabled", "--default-index", _SOURCE_POLICY.get("index_url",INDEX_URL),
                    "--constraint", constraints.name, inputs.name])
    return _read_uv_lock(report, core, locked)


def _install_uv(stage: Path, overlay: Path, wheel_lock: Path) -> None:
    # The resolver's URLs are never passed to the installer. It only receives
    # our bounded, hashed, path-checked local wheel files and runs offline.
    _run_uv(stage, ["install", "--offline", "--no-index", "--no-deps", "--no-build",
                    "--require-hashes", "--link-mode", "copy", "--target", str(overlay),
                    "--keyring-provider", "disabled", "--requirement", wheel_lock.name])


def _download_wheels(stage: Path, selected: list[dict]) -> list[Path]:
    folder = stage / "wheels"
    folder.mkdir(mode=0o700)
    downloaded, compressed_size, extracted_size, entry_count = [], 0, 0, 0
    cache = stage.parent / "wheel-cache" if stage.parent.name == ".packages" else None
    if cache is not None:
        if cache.is_symlink():
            raise PackageError("UNSAFE_PACKAGE_FILES", "The project wheel cache cannot be a link.")
        cache.mkdir(mode=0o700, exist_ok=True)
    cache_hits = 0
    for row in selected:
        name = _safe_wheel_url(row["url"])
        target = folder / name
        if target.exists():
            raise PackageError("INVALID_WHEEL", "Two packages resolved to the same wheel file.")
        print(f"Downloading {row['name']} {row['version']}", flush=True)
        digest = hashlib.sha256()
        try:
            cached = cache / (row["sha256"] + ".whl") if cache is not None else None
            if cached is not None and cached.is_symlink():
                raise PackageError("UNSAFE_PACKAGE_FILES", "Cached wheels cannot be links.")
            cached_valid = False
            if cached is not None and cached.is_file() and cached.stat().st_size <= MAX_DOWNLOAD_BYTES:
                cache_digest = hashlib.sha256()
                with cached.open("rb") as candidate:
                    while block := candidate.read(1024 * 1024):
                        cache_digest.update(block)
                cached_valid = cache_digest.hexdigest() == row["sha256"]
            request = Request(row["url"], headers={"User-Agent": "OpenEconometrics project packages"})
            from urllib.request import url2pathname
            local_source = urlparse(row["url"]).scheme == "file"
            response = cached.open("rb") if cached_valid else Path(url2pathname(urlparse(row["url"]).path)).open("rb") if local_source else urlopen(request, timeout=15)
            if cached_valid:
                cache_hits += 1
            with response, target.open("xb") as output:
                if not cached_valid and not local_source:
                    _safe_wheel_url(response.geturl())
                while block := response.read(1024 * 1024):
                    compressed_size += len(block)
                    if compressed_size > MAX_DOWNLOAD_BYTES:
                        raise PackageError("PACKAGE_LIMIT", "Wheel downloads exceed the 256 MiB project limit.")
                    digest.update(block)
                    output.write(block)
        except PackageError:
            raise
        except OSError as exc:
            raise PackageError("PACKAGE_DOWNLOAD_FAILED", "A wheel could not be downloaded from PyPI.") from exc
        if digest.hexdigest() != row["sha256"]:
            raise PackageError("WHEEL_HASH_MISMATCH", "A downloaded wheel did not match its verified hash.")
        try:
            with zipfile.ZipFile(target) as archive:
                for entry in archive.infolist():
                    entry_count += 1
                    if entry_count > MAX_FILES:
                        raise PackageError("PACKAGE_LIMIT", "Wheels exceed the 50,000-entry project limit.")
                    components = entry.filename.rstrip("/").split("/")
                    if (not components or any(component in {"", ".", ".."} for component in components)
                            or "\\" in entry.filename or ":" in entry.filename
                            or stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFLNK):
                        raise PackageError("UNSAFE_PACKAGE_FILES", "A wheel contains unsafe paths or links.")
                    extracted_size += entry.file_size
                    if extracted_size > MAX_OVERLAY_BYTES:
                        raise PackageError("PACKAGE_LIMIT", "Unpacked packages exceed the 512 MiB project limit.")
        except zipfile.BadZipFile as exc:
            raise PackageError("INVALID_WHEEL", "A downloaded wheel is not a valid archive.") from exc
        if cache is not None and not cached_valid:
            entries = [path for path in cache.iterdir()
                       if re.fullmatch(r"[a-f0-9]{64}\.whl", path.name) and path.is_file() and not path.is_symlink()]
            entries.sort(key=lambda path: path.stat().st_mtime)
            used = sum(path.stat().st_size for path in entries)
            while entries and (used + target.stat().st_size > MAX_DOWNLOAD_BYTES or len(entries) >= 200):
                old = entries.pop(0)
                used -= old.stat().st_size
                old.unlink()
            temporary = stage / ("cached-wheel-" + uuid4().hex)
            shutil.copyfile(target, temporary)
            temporary.replace(cached)
        if cached_valid:
            cached.touch()
        downloaded.append(target)
    _write_json(stage / "wheel-downloads.json", {"cache_hits": cache_hits, "downloads": len(selected) - cache_hits})
    return downloaded


def install(action: dict) -> dict:
    global _SOURCE_WHEELS, _SOURCE_POLICY, _SOURCE_ROOT
    try:
        return _install_action(action)
    finally:
        _SOURCE_WHEELS, _SOURCE_POLICY, _SOURCE_ROOT = {}, {}, None


def _install_action(action: dict) -> dict:
    global _SOURCE_WHEELS, _SOURCE_POLICY, _SOURCE_ROOT
    required = {"schema", "stage", "python", "core", "requirements", "locked"}
    if (not isinstance(action, dict) or not required.issubset(action)
            or set(action) - required - {"installer", "specifications", "source_options"}
            or type(action["schema"]) is not int or action["schema"] != 1):
        raise PackageError("INVALID_INSTALL_ACTION", "The package installation request is invalid.")
    backend = action.get("installer", "pip")
    if not isinstance(backend, str) or backend not in {"pip", "uv"}:
        raise PackageError("INVALID_INSTALL_ACTION", "Choose the pip or uv package installer.")
    stage = Path(action["stage"])
    if (not stage.is_absolute() or stage.is_symlink() or not stage.is_dir()
            or stage.parent.name != ".packages" or stage.parent.is_symlink()
            or not re.fullmatch(r"stage-[a-f0-9]{32}", stage.name)):
        raise PackageError("INVALID_INSTALL_ACTION", "The private package staging directory is invalid.")
    requests = _records(action["requirements"], nullable=True)
    locked = None if action["locked"] is None else _records(action["locked"])
    # Validate fixed core and Python shape without requiring null requests to
    # pretend that the resolver has already observed their installed versions.
    skeleton = validate_manifest({"schema": 1, "python": action["python"], "core": action["core"],
                                  "requirements": [], "locked": []}, core=protected_versions())
    if skeleton["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise PackageError("INCOMPATIBLE_PYTHON", "The installer uses a different Python version.")
    if any(row["name"] in skeleton["core"] for row in requests):
        raise PackageError("PROTECTED_PACKAGE", "OpenEconometrics's bundled package versions cannot be replaced.")
    _SOURCE_WHEELS, _SOURCE_POLICY, _SOURCE_ROOT = {}, {}, None
    source_records=[]
    if action.get("source_options"):
        from openecon.package_sources import validate_sources, prepare_sources
        _SOURCE_POLICY=validate_sources(action["source_options"],{row["name"] for row in requests})
        _write_json(stage/"source-policy.json", _SOURCE_POLICY)
        _SOURCE_ROOT=stage/"sources"
        def build(source,output):
            # Explicit trusted source execution; no download/install of build tools
            # and no mutation of embedded Python. Backends/tools must exist already.
            print("Building an explicitly requested source snapshot",flush=True)
            _run_pip(stage,["--isolated","--disable-pip-version-check","--no-input","wheel",
                            "--no-index","--no-deps","--no-build-isolation","--no-cache-dir",
                            "--wheel-dir",str(output),str(source)])
        _SOURCE_WHEELS,source_records=prepare_sources(stage,_SOURCE_POLICY,build)
    specifications = action.get("specifications")
    if specifications is not None:
        from openecon.package_requirements import parse_specifications
        active, specifications = parse_specifications(specifications)
        if {row["name"] for row in active} - set(skeleton["core"]) != {row["name"] for row in requests}:
            raise PackageError("INVALID_INSTALL_ACTION", "Package specifications must match the requested roots.")
        from packaging.requirements import Requirement
        for specification in specifications:
            requirement = Requirement(specification)
            name = canonical_name(requirement.name)
            if (name in skeleton["core"] and (requirement.marker is None or requirement.marker.evaluate())
                    and not requirement.specifier.contains(skeleton["core"][name], prereleases=True)):
                raise PackageError("PROTECTED_PACKAGE", "A package requirement would change OpenEconometrics's fixed core.")
    elif backend == "uv":
        specifications = [row["name"] + (f"=={row['version']}" if row["version"] is not None else "")
                          for row in requests]
    if backend == "uv":
        assert specifications is not None
        selected = _resolve_uv(stage, requests, skeleton["core"], locked, specifications)
    elif specifications is not None:
        selected = _resolve(stage, requests, skeleton["core"], locked, specifications)
    else:
        selected = _resolve(stage, requests, skeleton["core"], locked)
    versions = {row["name"]: row["version"] for row in selected}
    if any(row["name"] not in versions or row["version"] is not None and versions[row["name"]] != row["version"]
           for row in requests):
        raise PackageError("INVALID_RESOLUTION", "Resolved packages do not satisfy every requested root.")
    manifest_data = {**skeleton,
        "requirements": [{"name": row["name"], "version": versions[row["name"]]} for row in requests],
        "locked": [{"name": row["name"], "version": row["version"]} for row in selected]}
    if backend == "uv" or specifications is not None:
        manifest_data.update({"schema": 2, "installer": backend, "specifications": specifications})
    manifest = validate_manifest(manifest_data)
    wheels = _download_wheels(stage, selected)
    overlay = stage / "site-packages"
    overlay.mkdir(mode=0o700)
    if wheels:
        wheel_lock = stage / "wheel-lock.txt"
        wheel_lock.write_text("".join(f"{path.as_uri()} --hash=sha256:{row['sha256']}\n"
                                     for path, row in zip(wheels, selected, strict=True)), encoding="utf-8")
        print("Installing verified wheels", flush=True)
        if backend == "uv":
            _install_uv(stage, overlay, wheel_lock)
        else:
            _run_pip(stage, ["--isolated", "--disable-pip-version-check", "--no-input", "install",
                             "--no-index", "--no-deps", "--only-binary=:all:", "--no-cache-dir",
                             "--no-compile", "--require-hashes", "--target", str(overlay),
                             "--requirement", str(wheel_lock)])
    # Executable CLI wrappers are deliberately outside this library-only API.
    for name in ["bin", "Scripts"]:
        candidate = overlay / name
        if candidate.is_symlink():
            raise PackageError("UNSAFE_PACKAGE_FILES", "Generated package scripts cannot be links.")
        if candidate.is_dir():
            shutil.rmtree(candidate)
    for candidate in overlay.glob("*.pth"):
        if candidate.is_symlink():
            raise PackageError("UNSAFE_PACKAGE_FILES", "Package import hooks cannot be links.")
        candidate.unlink()
    validate_overlay(overlay, manifest)
    # Wheel URLs and hashes stay local. Shared manifests contain versions only.
    _write_json(stage / "wheel-provenance.json", {"schema": 1, "wheels": selected})
    if _SOURCE_POLICY:
        _write_json(stage/"source-provenance.json",{"schema":1,"sources":source_records,
                   "index_url":_SOURCE_POLICY.get("index_url",INDEX_URL),"installer":backend,
                   "resolved":[{"name":row["name"],"version":row["version"],"sha256":row["sha256"]} for row in selected]})
        if _SOURCE_ROOT.exists():
            shutil.rmtree(_SOURCE_ROOT)
        (stage/"source-policy.json").unlink(missing_ok=True)
    shutil.rmtree(stage / "wheels")
    print("Project packages ready", flush=True)
    return {"state": "complete", "manifest": manifest}


def _guard_pip_links(arguments: list[str], stage: Path) -> None:
    """Reject direct dependency URLs and redirect hosts outside our PyPI source."""
    from pip._internal.models.link import Link
    from pip._vendor.requests.sessions import Session
    original_link, original_send = Link.__init__, Session.send
    local_install = "--no-index" in arguments
    wheel_root = stage / "wheels"
    source_root = stage / "sources"
    policy_path=stage/"source-policy.json"
    policy=json.loads(policy_path.read_text()) if policy_path.is_file() and not policy_path.is_symlink() else {}
    from openecon.package_sources import validate_sources
    policy=validate_sources(policy,{row["name"] for row in policy.get("artifacts",[])}) if policy else {}
    hosts={"pypi.org",WHEEL_HOST,*policy.get("wheel_hosts",[])}
    if policy.get("index_url"):
        hosts.add(urlparse(policy["index_url"]).hostname)
    source_build="wheel" in arguments and "--no-build-isolation" in arguments

    def allowed(value: str) -> None:
        parsed = urlparse(value)
        if parsed.scheme == "file" and (local_install or policy):
            from urllib.request import url2pathname
            candidate = Path(url2pathname(parsed.path))
            permitted=(candidate.parent==wheel_root and local_install or candidate.parent==source_root and candidate.suffix==".whl")
            if source_build and candidate.is_dir() and candidate.parent==source_root:
                permitted=True
            if source_build and candidate.suffix==".whl" and candidate.is_relative_to(stage/"installer-scratch"):
                permitted=True
            if (parsed.netloc not in {"", "localhost"} or not permitted or candidate.is_symlink()
                    or not candidate.exists() or not source_build and candidate.suffix != ".whl"):
                raise PackageError("INVALID_WHEEL", "Only this operation's verified local wheels are allowed.")
            return
        if (parsed.scheme != "https" or parsed.hostname not in hosts
                or parsed.username or parsed.password or parsed.port not in {None, 443}):
            raise PackageError("INVALID_WHEEL", "Dependencies must resolve to trusted PyPI wheels.")

    def checked_link(self, url, *args, **kwargs):
        allowed(url)
        return original_link(self, url, *args, **kwargs)

    def checked_send(self, request, *args, **kwargs):
        allowed(request.url)
        return original_send(self, request, *args, **kwargs)

    Link.__init__, Session.send = checked_link, checked_send


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenEconometrics private project package installer")
    parser.add_argument("action", type=Path)
    arguments = parser.parse_args(argv)
    result_path = None
    try:
        if arguments.action.is_symlink() or arguments.action.stat().st_size > 256 * 1024:
            raise PackageError("INVALID_INSTALL_ACTION", "The installer action file is invalid.")
        action = json.loads(arguments.action.read_text(encoding="utf-8"))
        if isinstance(action, dict) and set(action) == {"operation", "arguments"} and action["operation"] == "pip":
            if not isinstance(action["arguments"], list) or not all(isinstance(value, str) for value in action["arguments"]):
                raise PackageError("INVALID_INSTALL_ACTION", "The private pip action is invalid.")
            _guard_pip_links(action["arguments"], arguments.action.parent)
            from pip._internal.cli.main import main as pip_main
            return pip_main(action["arguments"])
        result_path = arguments.action.parent / "result.json"
        result = install(action)
        _write_json(result_path, result)
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, PackageError) else "PACKAGE_INSTALL_FAILED"
        message = str(exc)[:1000] if isinstance(exc, PackageError) else "Package installation failed; the previous environment remains active."
        if result_path is not None:
            _write_json(result_path, {"state": "error", "code": code, "message": message})
        print(f"{code}: {message}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
