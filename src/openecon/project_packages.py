"""Atomic, opt-in project package overlays for the bundled desktop Python.

Package installation runs in an owned subprocess. It never modifies the base
runtime, uses another interpreter, or installs automatically when a project opens.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
from uuid import uuid4

MAX_PACKAGES = 100
MAX_FILES = 50_000
MAX_OVERLAY_BYTES = 512 * 1024 * 1024
MAX_LOG_BYTES = 16 * 1024
MAX_STATE_BYTES = 256 * 1024
NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?$")
VERSION_PATTERN = re.compile(
    r"^(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?$",
    re.IGNORECASE,
)

# Explicit desktop/runtime dependencies only: never enumerate the developer's
# full environment (which also contains reference estimators and test tools).
PROTECTED_IMPORTS = {
    "openecon": ("openecon",), "openecon-charts": ("openecon_charts",),
    "torch": ("torch", "functorch", "torchgen"), "numpy": ("numpy",),
    "pandas": ("pandas",), "pyarrow": ("pyarrow",),
    "fastapi": ("fastapi",), "starlette": ("starlette",), "uvicorn": ("uvicorn",),
    "pydantic": ("pydantic",), "pydantic-core": ("pydantic_core",),
    "python-multipart": ("python_multipart", "multipart"),
    "openpyxl": ("openpyxl",), "pyreadstat": ("pyreadstat",),
    "pip": ("pip",), "uv": ("uv",), "packaging": ("packaging",), "setuptools": ("setuptools", "pkg_resources"),
    "threadpoolctl": ("threadpoolctl",), "typer": ("typer",), "mcp": ("mcp",),
    "rich": ("rich",), "shellingham": ("shellingham",),
    "typing-extensions": ("typing_extensions",), "typing-inspection": ("typing_inspection",),
    "annotated-types": ("annotated_types",), "annotated-doc": ("annotated_doc",),
    "sympy": ("sympy",), "mpmath": ("mpmath",), "networkx": ("networkx",),
    "jinja2": ("jinja2",), "markupsafe": ("markupsafe",), "filelock": ("filelock",),
    "fsspec": ("fsspec",), "pytz": ("pytz",), "tzdata": ("tzdata",),
    "python-dateutil": ("dateutil",), "six": ("six",), "et-xmlfile": ("et_xmlfile",),
    "anyio": ("anyio",), "sniffio": ("sniffio",), "click": ("click",),
    "h11": ("h11",), "httpx": ("httpx",), "httpcore": ("httpcore",),
    "certifi": ("certifi",), "idna": ("idna",), "charset-normalizer": ("charset_normalizer",),
    "requests": ("requests",), "urllib3": ("urllib3",),
    "cryptography": ("cryptography",), "cffi": ("cffi", "_cffi_backend"),
    "pycparser": ("pycparser",), "pyjwt": ("jwt",), "attrs": ("attrs", "attr"),
    "h2": ("h2",), "hpack": ("hpack",), "hyperframe": ("hyperframe",),
    "importlib-metadata": ("importlib_metadata",), "zipp": ("zipp",),
}
PROTECTED_DISTRIBUTIONS = tuple(PROTECTED_IMPORTS)


class PackageError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical_name(value: str) -> str:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise PackageError("INVALID_PACKAGE", "Use a plain PyPI package name, without URLs or options.")
    return re.sub(r"[-_.]+", "-", value).lower()


def canonical_version(value: str) -> str:
    if (not isinstance(value, str) or len(value) > 100
            or not VERSION_PATTERN.fullmatch(value)):
        raise PackageError("INVALID_VERSION", "Use one exact package version, without operators or URLs.")
    from packaging.version import InvalidVersion, Version
    try:
        return str(Version(value))
    except InvalidVersion as exc:
        raise PackageError("INVALID_VERSION", "The exact package version is invalid.") from exc


def _same_version(first: str, second: str) -> bool:
    from packaging.version import Version
    return Version(first) == Version(second)


def _loaded_versions(value: dict[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 2 * MAX_PACKAGES:
        raise PackageError("INVALID_MANIFEST", "Loaded package versions must be a bounded name/version mapping.")
    result = {}
    for name, version in value.items():
        name = canonical_name(name)
        if name in result:
            raise PackageError("INVALID_MANIFEST", "Loaded package names must be unique.")
        result[name] = canonical_version(version)
    return result


def protected_versions() -> dict[str, str]:
    result = {}
    for name, roots in PROTECTED_IMPORTS.items():
        # Frozen builds only protect dependencies actually available in their
        # import archive; copied metadata alone does not make a module available.
        if not any(PathFinder.find_spec(root, sys.path) is not None for root in roots):
            continue
        try:
            result[name] = canonical_version(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            continue
    from openecon import __version__
    result["openecon"] = canonical_version(__version__)
    return dict(sorted(result.items()))


def _records(value, *, nullable: bool = False) -> list[dict]:
    if not isinstance(value, list) or len(value) > MAX_PACKAGES:
        raise PackageError("INVALID_MANIFEST", "A package list supports at most 100 entries.")
    result, seen = [], set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"name", "version"}:
            raise PackageError("INVALID_MANIFEST", "Package records need exactly name and version.")
        name = canonical_name(row["name"])
        version = None if nullable and row["version"] is None else canonical_version(row["version"])
        if name in seen:
            raise PackageError("INVALID_MANIFEST", "Package names must be unique.")
        seen.add(name)
        result.append({"name": name, "version": version})
    return sorted(result, key=lambda row: row["name"])


def validate_manifest(value, *, core: dict[str, str] | None = None,
                      python: str | None = None) -> dict:
    schema = value.get("schema") if isinstance(value, dict) else None
    fields = {"schema", "python", "core", "requirements", "locked"}
    if schema == 2:
        fields |= {"installer", "specifications"}
    if (not isinstance(value, dict) or set(value) != fields
            or type(schema) is not int or schema not in {1, 2}
            or not isinstance(value["python"], str)
            or not re.fullmatch(r"[0-9]+\.[0-9]+", value["python"])
            or not isinstance(value["core"], dict) or len(value["core"]) > MAX_PACKAGES):
        raise PackageError("INVALID_MANIFEST", "The package manifest format is invalid.")
    pinned_core = {}
    for name, version in value["core"].items():
        normalized = canonical_name(name)
        if normalized in pinned_core:
            raise PackageError("INVALID_MANIFEST", "Core package names must be unique.")
        pinned_core[normalized] = canonical_version(version)
    requirements, locked = _records(value["requirements"]), _records(value["locked"])
    locks = {row["name"]: row["version"] for row in locked}
    if any(row["name"] in pinned_core for row in locked):
        raise PackageError("INVALID_MANIFEST", "Additional packages cannot replace the fixed core.")
    if any(locks.get(row["name"]) != row["version"] for row in requirements):
        raise PackageError("INVALID_MANIFEST", "Every requested package must have an exact lock.")
    if core is not None and pinned_core != core:
        raise PackageError("INCOMPATIBLE_CORE", "This package manifest belongs to a different OpenEconometrics core.")
    if python is not None and value["python"] != python:
        raise PackageError("INCOMPATIBLE_PYTHON", "This package manifest uses a different Python minor version.")
    result = {"schema": schema, "python": value["python"], "core": dict(sorted(pinned_core.items())),
              "requirements": requirements, "locked": locked}
    if schema == 2:
        from openecon.package_requirements import validate_specification_pins
        if not isinstance(value["installer"], str) or value["installer"] not in {"pip", "uv"}:
            raise PackageError("INVALID_MANIFEST", "Use the pip or uv package installer.")
        result.update(installer=value["installer"], specifications=validate_specification_pins(
            value["specifications"], requirements, pinned_core))
    return result


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_STATE_BYTES:
        raise PackageError("INVALID_PACKAGE_STATE", "The saved package state is invalid.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise PackageError("INVALID_PACKAGE_STATE", "The saved package state could not be read.") from exc


def validate_overlay(path: str | Path, manifest: dict | None = None) -> None:
    """Check import roots, installed distribution identity and all file paths.

    This prevents accidental replacement of base imports. It is not an OS
    sandbox: trusted third-party Python still has the local user's permissions.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise PackageError("UNSAFE_PACKAGE_FILES", "The project package directory is invalid.")
    protected = {name.casefold() for name in sys.stdlib_module_names}
    for imports in PROTECTED_IMPORTS.values():
        if any(PathFinder.find_spec(name, sys.path) is not None for name in imports):
            protected.update(name.casefold() for name in imports)
    count = size = 0
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in [*folders, *files]:
            entry = Path(directory) / name
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise PackageError("UNSAFE_PACKAGE_FILES", "Package files cannot contain links or special files.")
            if entry.parent == root:
                if name in {"bin", "Scripts"} or name.endswith(".pth"):
                    raise PackageError("UNSAFE_PACKAGE_FILES", "Executable package hooks and scripts are not supported.")
                import_name = name.split(".", 1)[0]
                if import_name.casefold() in protected:
                    raise PackageError("PROTECTED_PACKAGE", f"The package would replace protected import '{import_name}'.")
            if stat.S_ISREG(info.st_mode):
                count += 1
                size += info.st_size
                if count > MAX_FILES or size > MAX_OVERLAY_BYTES:
                    raise PackageError("PACKAGE_LIMIT", "Additional packages exceed the project file or 512 MiB limit.")
    installed = []
    for distribution in importlib.metadata.distributions(path=[str(root)]):
        installed.append({"name": canonical_name(distribution.metadata["Name"]),
                          "version": canonical_version(distribution.version)})
    installed = _records(installed)
    if manifest is not None and installed != validate_manifest(manifest)["locked"]:
        raise PackageError("PACKAGE_LOCK_MISMATCH", "Installed packages do not match the project manifest.")


def installer_command(action: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--package-installer", str(action)]
    # A source checkout can coexist with another editable installation. Keep
    # every owned installer subprocess on the exact package running its parent.
    source_root = str(Path(__file__).resolve().parent.parent)
    bootstrap = ("import sys; sys.path.insert(0, " + repr(source_root) + "); "
                 "from openecon.package_installer import main; raise SystemExit(main())")
    return [sys.executable, "-c", bootstrap, str(action)]


def installer_environment() -> dict[str, str]:
    safe = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
            "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR", "APPDATA", "LOCALAPPDATA", "USERPROFILE"}
    environment = {name: value for name, value in os.environ.items() if name in safe}
    if getattr(sys, "frozen", False):
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return environment


class ProjectPackages:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / ".packages"
        if self.root.is_symlink():
            raise PackageError("UNSAFE_PACKAGE_FILES", "The project package root cannot be a symbolic link.")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._core = protected_versions()
        self._python = f"{sys.version_info.major}.{sys.version_info.minor}"
        self._available = PathFinder.find_spec("pip", sys.path) is not None
        self._manifest = {"schema": 1, "python": self._python, "core": self._core,
                          "requirements": [], "locked": []}
        self._generation = None
        self._job = None
        self._process = None
        self._cancelled = threading.Event()
        self._thread = None
        self._closed = False
        active = self.root / "active.json"
        if active.exists() or active.is_symlink():
            value = _read_json(active)
            if (not isinstance(value, dict) or set(value) != {"generation", "manifest"}
                    or not isinstance(value["generation"], str)
                    or not re.fullmatch(r"[0-9a-f]{32}", value["generation"])):
                raise PackageError("INVALID_PACKAGE_STATE", "The active package generation is invalid.")
            saved = validate_manifest(value["manifest"], python=self._python)
            if saved["core"] != self._core:
                # Only the application version and bundled installer may change
                # without replacing a project's already installed libraries.
                stable = {name: version for name, version in self._core.items() if name not in {"openecon", "uv"}}
                previous = {name: version for name, version in saved["core"].items() if name not in {"openecon", "uv"}}
                if stable != previous or any(row["name"] in self._core for row in saved["locked"]):
                    raise PackageError("INCOMPATIBLE_CORE", "This package manifest belongs to a different OpenEconometrics core.")
                overlay = self.root / ("generation-" + value["generation"]) / "site-packages"
                validate_overlay(overlay, saved)
                from packaging.requirements import InvalidRequirement, Requirement
                from packaging.version import Version
                from email.parser import Parser
                extras = {canonical_name(Requirement(spec).name): Requirement(spec).extras
                          for spec in saved.get("specifications", [])}
                for metadata in overlay.glob("*.dist-info/METADATA"):
                    message = Parser().parsestr(metadata.read_text(encoding="utf-8"))
                    name = canonical_name(message.get("Name", ""))
                    for declaration in message.get_all("Requires-Dist", []):
                        try:
                            dependency = Requirement(declaration)
                        except InvalidRequirement as exc:
                            raise PackageError("INCOMPATIBLE_CORE", "An installed library has invalid dependency metadata.") from exc
                        dependency_name = canonical_name(dependency.name)
                        applies = dependency.marker is None or any(dependency.marker.evaluate({"extra": extra})
                                                                  for extra in extras.get(name, set()) | {""})
                        if (applies and dependency_name in self._core
                                and not dependency.specifier.contains(Version(self._core[dependency_name]), prereleases=True)):
                            raise PackageError("INCOMPATIBLE_CORE", "An installed library requires an incompatible bundled package.")
                saved = validate_manifest({**saved, "core": self._core}, core=self._core, python=self._python)
                _write_json(active, {"generation": value["generation"], "manifest": saved})
            self._manifest = saved
            self._generation = value["generation"]

    def snapshot(self) -> dict:
        with self._lock:
            return {"available": self._available,
                    "base": [{"name": name, "version": version} for name, version in self._core.items()],
                    "requirements": deepcopy(self._manifest["requirements"]),
                    "installed": deepcopy(self._manifest["locked"]),
                    "manifest": deepcopy(self._manifest), "job": deepcopy(self._job)}

    def export_portable(self) -> dict:
        from openecon.portable_environment import export_document
        with self._lock:
            return export_document(self._manifest)

    def preview_portable(self, document: dict) -> dict:
        from openecon.portable_environment import validate_document
        with self._lock:
            manifest = validate_document(document, core=self._core, python=self._python)
            return {"manifest": manifest, "matches": manifest == self._manifest}

    def restore_portable(self, document: dict) -> dict:
        # Validate again at the installation boundary; preview is inert.
        return self.start_restore(self.preview_portable(document)["manifest"])

    def active_path(self) -> Path | None:
        with self._lock:
            if self._generation is None or not self._manifest["locked"]:
                return None
            generation = self.root / f"generation-{self._generation}"
            if self.root.is_symlink() or generation.is_symlink() or not generation.is_dir():
                raise PackageError("UNSAFE_PACKAGE_FILES", "The active package generation is unavailable.")
            path = generation / "site-packages"
            validate_overlay(path, self._manifest)
            return path

    def _start(self, requirements: list[dict], locked: list[dict] | None = None, *,
               loaded_versions: dict[str, str] | None = None, preserve_previous: bool = False,
               installer: str = "pip", specifications: list[str] | None = None,
               source_options: dict | None = None) -> dict:
        with self._lock:
            if self._closed:
                raise PackageError("PACKAGES_CLOSED", "The project package manager is closed.")
            if not self._available:
                raise PackageError("INSTALLER_UNAVAILABLE", "This runtime does not include the package installer.")
            if self._job and self._job["state"] == "running":
                raise PackageError("PACKAGES_BUSY", "A package operation is already running.")
            requirements = _records(requirements, nullable=True)
            if source_options is not None:
                from openecon.package_sources import validate_sources
                source_options = validate_sources(source_options, {row["name"] for row in requirements})
            if not isinstance(installer, str) or installer not in {"pip", "uv"}:
                raise PackageError("INVALID_PACKAGE", "Use the pip or uv package installer.")
            if specifications is not None:
                from openecon.package_requirements import parse_specifications
                active, specifications = parse_specifications(specifications)
                if {row["name"] for row in active} - set(self._core) != {row["name"] for row in requirements}:
                    raise PackageError("INCOMPATIBLE_PACKAGE_MARKERS", "The saved package markers do not match this computer's environment.")
            loaded_versions = _loaded_versions(loaded_versions)
            if not isinstance(preserve_previous, bool):
                raise PackageError("INVALID_MANIFEST", "preserve_previous must be a boolean.")
            if any(row["name"] in self._core or row["name"] in {"openecon", "openecon-charts"}
                   or row["name"] in sys.stdlib_module_names
                   for row in requirements):
                raise PackageError("PROTECTED_PACKAGE", "OpenEconometrics's bundled packages have fixed versions.")
            if shutil.disk_usage(self.root).free < 2 * 1024**3:
                raise PackageError("PACKAGE_DISK_LIMIT", "Package installation needs at least 2 GiB free space.")
            job_id = uuid4().hex
            self._job = {"id": job_id, "state": "running", "message": "Preparing packages", "log": ""}
            self._cancelled.clear()
            self._thread = threading.Thread(target=self._run,
                                            args=(job_id, requirements, locked, loaded_versions, preserve_previous,
                                                  installer, specifications, source_options),
                                            name="OpenEconometrics project packages", daemon=True)
            self._thread.start()
            return self.snapshot()

    def start_install(self, name: str, version: str | None = None) -> dict:
        name = canonical_name(name)
        version = None if version is None else canonical_version(version)
        with self._lock:
            if self._manifest["schema"] == 2:
                return self.start_install_many([{"name": name, "version": version}],
                                               installer=self._manifest["installer"])
            requirements = {row["name"]: row["version"] for row in self._manifest["requirements"]}
            requirements[name] = version
            return self._start([{"name": name, "version": value} for name, value in requirements.items()])
    def start_install_many(self, requirements: list[dict], *, loaded_versions: dict[str, str] | None = None,
                           preserve_previous: bool = False, installer: str = "pip",
                           specifications: list[str] | None = None, upgrade: bool = False,
                           source_options: dict | None = None) -> dict:
        requested = _records(requirements, nullable=True)
        if source_options is not None:
            from openecon.package_sources import validate_sources
            source_options = validate_sources(source_options, {row["name"] for row in requested})
        loaded = _loaded_versions(loaded_versions)
        if not isinstance(preserve_previous, bool) or not isinstance(upgrade, bool):
            raise PackageError("INVALID_MANIFEST", "Package installation switches must be booleans.")
        if not isinstance(installer, str) or installer not in {"pip", "uv"}:
            raise PackageError("INVALID_PACKAGE", "Use the pip or uv package installer.")
        from packaging.requirements import Requirement
        from openecon.package_requirements import parse_specifications, satisfies
        normalized = None
        if specifications is not None:
            active, normalized = parse_specifications(specifications)
            if _records(active, nullable=True) != requested:
                raise PackageError("INVALID_PACKAGE", "Package names must match their requirement specifications.")
        with self._lock:
            if self._closed:
                raise PackageError("PACKAGES_CLOSED", "The project package manager is closed.")
            if self._job and self._job["state"] == "running":
                raise PackageError("PACKAGES_BUSY", "A package operation is already running.")
            installed = {row["name"]: row["version"] for row in self._manifest["locked"]}
            roots = {row["name"]: row["version"] for row in self._manifest["requirements"]}
            old_specs = {Requirement(spec).name: spec for spec in self._manifest.get("specifications", [])}
            new_specs = {Requirement(spec).name: spec for spec in normalized or []}
            merged_specs = {name: old_specs.get(name, name + "==" + version)
                            for name, version in roots.items()}
            merged_specs.update({name: spec for name, spec in old_specs.items() if name not in roots})
            advanced = installer == "uv" or normalized is not None or self._manifest["schema"] == 2
            satisfied = not source_options
            for row in requested:
                name, version = row["name"], row["version"]
                specification = new_specs.get(name, name + ("==" + version if version else ""))
                requirement = Requirement(specification)
                old_extras = Requirement(old_specs.get(name, name)).extras
                if old_extras:
                    requirement.extras |= old_extras
                    specification = str(requirement)
                extras_changed = not requirement.extras <= old_extras
                if name in self._core:
                    if not satisfies(specification, self._core[name]):
                        raise PackageError("PROTECTED_PACKAGE", "OpenEconometrics's bundled packages have fixed versions.")
                    if requirement.extras:
                        merged_specs[name] = specification
                        satisfied = satisfied and not extras_changed
                    continue
                if name in {"openecon", "openecon-charts"} or name in sys.stdlib_module_names:
                    raise PackageError("PROTECTED_PACKAGE", "OpenEconometrics's bundled packages have fixed versions.")
                existing = installed.get(name)
                matches = existing is not None and satisfies(specification, existing)
                satisfied = satisfied and matches and not extras_changed and not upgrade
                roots[name] = existing if matches and not upgrade and not source_options else version
                merged_specs[name] = specification
            for name, specification in new_specs.items():
                if name not in {row["name"] for row in requested} and name not in roots:
                    merged_specs[name] = specification
            if not advanced:
                merged_specs = None
            else:
                merged_specs = sorted(merged_specs.values())
            if satisfied:
                pinned = [{"name": name, "version": version} for name, version in sorted(roots.items())]
                candidate = {**self._manifest, "requirements": pinned}
                if advanced:
                    candidate.update(schema=2, installer=installer, specifications=merged_specs)
                manifest = validate_manifest(candidate, core=self._core, python=self._python)
                if manifest != self._manifest and self._generation is None and not manifest["locked"]:
                    # A core-only no-op has no overlay generation to persist.
                    return self.snapshot()
                if manifest != self._manifest:
                    if self.root.is_symlink():
                        raise PackageError("UNSAFE_PACKAGE_FILES", "The project package root changed.")
                    try:
                        _write_json(self.root / "active.json", {"generation": self._generation, "manifest": manifest})
                    except OSError as exc:
                        raise PackageError("PACKAGE_STATE_WRITE_FAILED", "Package requirements could not be saved; the previous environment remains active.") from exc
                    self._manifest = manifest
                return self.snapshot()
            return self._start([{"name": name, "version": version} for name, version in roots.items()],
                               loaded_versions=loaded, preserve_previous=preserve_previous,
                               installer=installer, specifications=merged_specs, source_options=source_options)

    def start_remove(self, name: str) -> dict:
        name = canonical_name(name)
        with self._lock:
            if name not in {row["name"] for row in self._manifest["requirements"]}:
                raise PackageError("PACKAGE_NOT_REQUESTED", "Remove a directly requested package; dependencies are managed automatically.")
            from packaging.requirements import Requirement
            specifications = self._manifest.get("specifications")
            if specifications is not None:
                specifications = [spec for spec in specifications if Requirement(spec).name != name]
            return self._start([row for row in self._manifest["requirements"] if row["name"] != name],
                               installer=self._manifest.get("installer", "pip"), specifications=specifications)

    def start_restore(self, manifest: dict) -> dict:
        manifest = validate_manifest(manifest, core=self._core, python=self._python)
        return self._start(manifest["requirements"], manifest["locked"],
                           installer=manifest.get("installer", "pip"), specifications=manifest.get("specifications"))

    def _append_log(self, value: str) -> None:
        with self._lock:
            self._job["log"] = (self._job["log"] + value).encode("utf-8")[-MAX_LOG_BYTES:].decode("utf-8", errors="replace")

    def _run(self, job_id: str, requirements: list[dict], locked: list[dict] | None,
             loaded_versions: dict[str, str], preserve_previous: bool,
             installer: str = "pip", specifications: list[str] | None = None,
             source_options: dict | None = None):
        stage = self.root / f"stage-{job_id}"
        promoted = None
        terminal_state = "error"
        terminal_message = "The package operation failed; the previous environment remains active."
        terminal_code = None
        try:
            if self.root.is_symlink():
                raise PackageError("UNSAFE_PACKAGE_FILES", "The project package root changed.")
            stage.mkdir(mode=0o700)
            action = stage / "action.json"
            operation = {"schema": 1, "stage": str(stage), "python": self._python,
                         "core": self._core, "requirements": requirements, "locked": locked}
            if installer != "pip" or specifications is not None:
                operation.update(installer=installer, specifications=specifications)
            if source_options:
                operation["source_options"] = source_options
            _write_json(action, operation)
            with self._lock:
                if self._cancelled.is_set():
                    raise PackageError("PACKAGE_CANCELLED", "Package operation cancelled.")
                self._process = subprocess.Popen(installer_command(action), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=installer_environment(),
                    start_new_session=os.name == "posix")
                process = self._process
            assert process.stdout is not None
            while block := process.stdout.read1(4096):
                self._append_log(block.decode("utf-8", errors="replace"))
            process.stdout.close()
            result_code = process.wait()
            if self._cancelled.is_set():
                raise PackageError("PACKAGE_CANCELLED", "Package operation cancelled.")
            if result_code and not (stage / "result.json").is_file():
                raise PackageError("PACKAGE_INSTALL_FAILED", "The installer stopped; see the package log. The previous environment remains active.")
            result = _read_json(stage / "result.json")
            if not isinstance(result, dict):
                raise PackageError("INVALID_PACKAGE_STATE", "The installer returned an invalid result.")
            if result_code or result.get("state") != "complete":
                raise PackageError(result.get("code", "PACKAGE_INSTALL_FAILED"),
                                   result.get("message", "The package operation failed; the previous environment remains active.")[:1000])
            manifest = validate_manifest(result["manifest"], core=self._core, python=self._python)
            if installer == "uv" or specifications is not None:
                if (manifest.get("installer") != installer
                        or specifications is not None and set(manifest.get("specifications", [])) != set(specifications)):
                    raise PackageError("INVALID_PACKAGE_STATE", "The installer did not preserve the requested package specifications.")
            validate_overlay(stage / "site-packages", manifest)
            with self._lock:
                if self._cancelled.is_set():
                    raise PackageError("PACKAGE_CANCELLED", "Package operation cancelled.")
                resolved = {row["name"]: row["version"] for row in manifest["locked"]}
                for name, version in loaded_versions.items():
                    if name not in self._core and (name not in resolved or not _same_version(resolved[name], version)):
                        raise PackageError("PACKAGE_RESTART_REQUIRED", "Restart Python before changing an imported package.")
                previous = self._generation
                promoted = self.root / f"generation-{job_id}"
                stage.replace(promoted)
                _write_json(self.root / "active.json", {"generation": job_id, "manifest": manifest})
                self._generation, self._manifest = job_id, manifest
                self._job["message"] = "Finalizing packages"
                terminal_state, terminal_message = "complete", "Packages ready"
            if previous and not preserve_previous:
                old = self.root / f"generation-{previous}"
                if old.is_dir() and not old.is_symlink():
                    shutil.rmtree(old, ignore_errors=True)
        except BaseException as exc:
            with self._lock:
                cancelled = self._cancelled.is_set() or isinstance(exc, PackageError) and exc.code == "PACKAGE_CANCELLED"
                message = str(exc)[:1000] if isinstance(exc, PackageError) else "The package operation failed; the previous environment remains active."
                terminal_state = "cancelled" if cancelled else "error"
                terminal_message = message
                terminal_code = "PACKAGE_CANCELLED" if cancelled else (
                    exc.code if isinstance(exc, PackageError) else "PACKAGE_INSTALL_FAILED")
        finally:
            if stage.is_dir() and not stage.is_symlink():
                shutil.rmtree(stage, ignore_errors=True)
            if promoted and self._generation != job_id and promoted.is_dir() and not promoted.is_symlink():
                shutil.rmtree(promoted, ignore_errors=True)
            # A terminal state permits the next job. Publish it only after
            # cleanup and clearing this job's subprocess pointer, together.
            with self._lock:
                if self._job and self._job["id"] == job_id:
                    self._process = None
                    self._job.update(state=terminal_state, message=terminal_message)
                    if terminal_code is not None:
                        self._job["code"] = terminal_code

    def prune_generations(self) -> int:
        """Delete inactive owned generations after the caller disposes Python."""
        with self._lock:
            if self._job and self._job["state"] == "running":
                return 0
            if self.root.is_symlink():
                raise PackageError("UNSAFE_PACKAGE_FILES", "The project package root changed.")
            active = f"generation-{self._generation}" if self._generation is not None else None
            removed = 0
            for path in self.root.iterdir():
                if (path.name != active and re.fullmatch(r"generation-[0-9a-f]{32}", path.name)
                        and not path.is_symlink() and path.is_dir()):
                    shutil.rmtree(path)
                    removed += 1
            return removed

    def cancel(self) -> dict:
        with self._lock:
            if self._job and self._job["state"] == "running":
                self._cancelled.set()
                process = self._process
                if process is not None and (os.name == "posix" or process.poll() is None):
                    try:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            taskkill = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "taskkill.exe"
                            subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, timeout=2, check=False)
                    except ProcessLookupError:
                        pass
                    except (OSError, subprocess.TimeoutExpired):
                        if process.poll() is None:
                            try:
                                process.terminate()
                            except OSError:
                                pass
                    try:
                        process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        if os.name != "posix":
                            process.kill()
                    # A leader can exit while a child ignores SIGTERM and keeps
                    # stdout open. Always finish the owned group on POSIX.
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            if process.poll() is None:
                                try:
                                    process.kill()
                                except OSError:
                                    pass
            return self.snapshot()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.cancel()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
