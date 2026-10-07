"""Explicit package sources, copied/built only inside an owned installer stage."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from urllib.parse import urlparse, unquote
from urllib.request import urlopen
import zipfile

from openecon.project_packages import PackageError, canonical_name

LIMIT = 256 * 1024**2


def check_build_tools(source):
    """No automatic build-tool installation into embedded Python."""
    from packaging.requirements import Requirement

    project = source / "pyproject.toml"
    try:
        if project.exists():
            if project.stat().st_size > 256 * 1024:
                raise ValueError("project metadata")
            system = tomllib.loads(project.read_text())["build-system"]
            required = system.get("requires", [])
            paths = system.get("backend-path", [])
            if not isinstance(paths, list) or any(
                not isinstance(path, str)
                or not (source / path).resolve().is_relative_to(source.resolve())
                for path in paths
            ):
                raise ValueError("backend path")
        else:
            required = ["setuptools>=40.8.0"]
        if not isinstance(required, list) or len(required) > 100:
            raise ValueError("build requirements")
        for text in required:
            requirement = Requirement(text)
            if requirement.url is not None:
                raise ValueError("build tool URL")
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            version = importlib.metadata.version(requirement.name)
            if not requirement.specifier.contains(version, prereleases=True):
                raise importlib.metadata.PackageNotFoundError(requirement.name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise PackageError(
            "BUILD_TOOLS_UNAVAILABLE",
            "The source build requires tools or versions not bundled in this Python. Build a compatible wheel externally and install it by hash.",
        ) from exc
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise PackageError(
            "INVALID_SOURCE_BUILD", "The source build metadata or backend path is unsupported."
        ) from exc


def clean_url(value):
    if not isinstance(value, str) or len(value) > 3000:
        raise PackageError("INVALID_SOURCE", "Specify a bounded HTTPS source URL.")
    try:
        url = urlparse(value)
        port = url.port
    except ValueError as exc:
        raise PackageError("INVALID_SOURCE", "The source URL is invalid.") from exc
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or port not in (None, 443)
        or any(ch.isspace() for ch in value)
    ):
        raise PackageError(
            "SOURCE_CREDENTIALS_UNSUPPORTED",
            "Use HTTPS sources without embedded credentials, query tokens or fragments. Authenticated sources must be downloaded separately and supplied as local wheels.",
        )
    return value


def validate_sources(value, names):
    if not isinstance(value, dict) or set(value) - {"index_url", "wheel_hosts", "artifacts"}:
        raise PackageError(
            "INVALID_SOURCE", "Supported source options are index_url, wheel_hosts and artifacts."
        )
    result = {}
    if "index_url" in value:
        result["index_url"] = clean_url(value["index_url"])
    hosts = value.get("wheel_hosts", [])
    if (
        not isinstance(hosts, list)
        or len(hosts) > 10
        or any(
            not isinstance(h, str) or not re.fullmatch(r"[a-z0-9.-]+", h) or ".." in h
            for h in hosts
        )
    ):
        raise PackageError(
            "INVALID_SOURCE", "wheel_hosts must contain at most ten plain lowercase host names."
        )
    if hosts:
        result["wheel_hosts"] = hosts
    artifacts = value.get("artifacts", [])
    if not isinstance(artifacts, list) or len(artifacts) > 10:
        raise PackageError("INVALID_SOURCE", "Specify at most ten explicit artifacts.")
    normalized = []
    seen = set()
    for row in artifacts:
        if not isinstance(row, dict) or "name" not in row:
            raise PackageError("INVALID_SOURCE", "Every artifact needs its package name.")
        name = canonical_name(row["name"])
        if name not in names or name in seen:
            raise PackageError(
                "INVALID_SOURCE", "Artifact names must uniquely match requested package roots."
            )
        seen.add(name)
        item = {**row, "name": name}
        if set(row) == {"name", "wheel", "sha256"}:
            wheel = row["wheel"]
            if not isinstance(row["sha256"], str) or not re.fullmatch(
                r"[a-fA-F0-9]{64}", row["sha256"]
            ):
                raise PackageError(
                    "INVALID_SOURCE", "Wheel sources require an explicit SHA-256 digest."
                )
            item["sha256"] = row["sha256"].lower()
            if isinstance(wheel, str) and wheel.startswith("https:"):
                clean_url(wheel)
            elif not isinstance(wheel, str) or not Path(wheel).is_absolute():
                raise PackageError(
                    "INVALID_SOURCE", "Specify an absolute local wheel path or HTTPS wheel URL."
                )
            if not unquote(urlparse(wheel).path).endswith(".whl"):
                raise PackageError("INVALID_SOURCE", "An artifact must be a wheel.")
        elif set(row) == {"name", "git", "commit", "build"}:
            clean_url(row["git"])
            if (
                row["build"] is not True
                or not isinstance(row["commit"], str)
                or not re.fullmatch(r"[a-fA-F0-9]{40}", row["commit"])
            ):
                raise PackageError(
                    "INVALID_SOURCE", "Git sources need a full 40-character commit and build=True."
                )
            item["commit"] = row["commit"].lower()
        elif set(row) == {"name", "directory", "build"}:
            if (
                row["build"] is not True
                or not isinstance(row["directory"], str)
                or not Path(row["directory"]).is_absolute()
            ):
                raise PackageError(
                    "INVALID_SOURCE",
                    "Local source builds need an absolute directory and build=True.",
                )
        else:
            raise PackageError(
                "INVALID_SOURCE",
                "Use a hashed wheel, pinned Git build, or explicit local directory build. Editable installs and arbitrary flags are unsupported.",
            )
        normalized.append(item)
    if normalized:
        result["artifacts"] = normalized
    return result


def _copy_source(source, destination):
    if source.is_symlink() or not source.is_dir():
        raise PackageError("INVALID_SOURCE", "The source directory must be a real local directory.")
    destination.mkdir()
    digest = hashlib.sha256()
    total = 0
    count = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(
            part in {".git", "__pycache__", ".venv", "build", "dist"} for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise PackageError(
                "UNSAFE_PACKAGE_FILES", "Source snapshots cannot contain symbolic links."
            )
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            count += 1
            total += path.stat().st_size
            if total > LIMIT or count > 50000:
                raise PackageError(
                    "PACKAGE_LIMIT", "Source snapshots exceed the 256 MiB or 50,000-file limit."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded = str(relative).encode()
            digest.update(len(encoded).to_bytes(8, "big") + encoded)
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            with path.open("rb") as inp, target.open("xb") as out:
                while block := inp.read(1024**2):
                    digest.update(block)
                    out.write(block)
        else:
            raise PackageError(
                "UNSAFE_PACKAGE_FILES", "Source snapshots contain an unsupported file type."
            )
    return digest.hexdigest()


def _wheel_identity(path, name):
    from packaging.utils import parse_wheel_filename
    from packaging.tags import sys_tags

    try:
        parsed, version, _, tags = parse_wheel_filename(path.name)
        if parsed != name or not tags.intersection(set(sys_tags())):
            raise ValueError("identity or ABI")
        with zipfile.ZipFile(path) as archive:
            records = [item for item in archive.namelist() if item.endswith(".dist-info/METADATA")]
            if len(records) != 1 or archive.getinfo(records[0]).file_size > 1024**2:
                raise ValueError("metadata")
            from email.parser import BytesParser

            meta = BytesParser().parsebytes(archive.read(records[0]))
            if canonical_name(meta["Name"]) != name or str(version) != meta["Version"]:
                raise ValueError("metadata identity")
    except (ValueError, KeyError, zipfile.BadZipFile, TypeError) as exc:
        raise PackageError(
            "INVALID_WHEEL", "The artifact identity or Python/platform ABI is incompatible."
        ) from exc
    return str(version), sorted(map(str, tags))


def prepare_sources(stage, options, build_callback):
    """Return name -> verified wheel URI plus local-only provenance."""
    folder = stage / "sources"
    folder.mkdir(mode=0o700)
    paths = {}
    records = []
    for row in options.get("artifacts", []):
        name = row["name"]
        record = {"name": name}
        if "wheel" in row:
            location = row["wheel"]
            remote = location.startswith("https:")
            filename = (
                unquote(urlparse(location).path.split("/")[-1]) if remote else Path(location).name
            )
            if "/" in filename or "\\" in filename or ".." in filename:
                raise PackageError("INVALID_WHEEL", "The wheel filename is invalid.")
            path = folder / filename
            if not remote and (Path(location).is_symlink() or not Path(location).is_file()):
                raise PackageError("INVALID_SOURCE", "The local wheel is unavailable.")
            try:
                response = urlopen(location, timeout=15) if remote else Path(location).open("rb")
                with response, path.open("xb") as output:
                    if remote:
                        final = clean_url(response.geturl())
                        if urlparse(final).hostname != urlparse(location).hostname:
                            raise PackageError(
                                "INVALID_SOURCE",
                                "Wheel source redirects must retain the requested host.",
                            )
                    total = 0
                    digest = hashlib.sha256()
                    while block := response.read(1024**2):
                        total += len(block)
                        if total > LIMIT:
                            raise PackageError("PACKAGE_LIMIT", "A source wheel exceeds 256 MiB.")
                        digest.update(block)
                        output.write(block)
            except OSError as exc:
                raise PackageError(
                    "SOURCE_DOWNLOAD_FAILED", "The requested source wheel could not be read."
                ) from exc
            if digest.hexdigest() != row["sha256"]:
                raise PackageError(
                    "WHEEL_HASH_MISMATCH", "The source wheel did not match its requested SHA-256."
                )
            record.update(kind="url-wheel" if remote else "local-wheel", location=location)
        else:
            source = folder / (name + "-source")
            if "git" in row:
                git = shutil.which("git")
                if git is None:
                    raise PackageError(
                        "BUILD_TOOLS_UNAVAILABLE", "Pinned Git builds require a Git executable."
                    )
                checkout = folder / (name + "-checkout")
                checkout.mkdir()
                environment = {
                    "PATH": str(Path(git).parent),
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": str(folder / "absent-git-config"),
                }
                commands = [
                    [git, "init", "--quiet", str(checkout)],
                    [
                        git,
                        "-C",
                        str(checkout),
                        "-c",
                        "protocol.file.allow=never",
                        "fetch",
                        "--quiet",
                        "--depth=1",
                        row["git"],
                        row["commit"],
                    ],
                    [git, "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
                ]
                try:
                    for command in commands:
                        subprocess.run(
                            command,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True,
                            timeout=120,
                        )
                    actual = subprocess.check_output(
                        [git, "-C", str(checkout), "rev-parse", "HEAD"], env=environment, text=True
                    ).strip()
                except (subprocess.SubprocessError, OSError) as exc:
                    raise PackageError(
                        "SOURCE_FETCH_FAILED", "The pinned Git source could not be fetched."
                    ) from exc
                if actual != row["commit"]:
                    raise PackageError(
                        "SOURCE_COMMIT_MISMATCH",
                        "The Git checkout did not match the requested commit.",
                    )
                snapshot = _copy_source(checkout, source)
                shutil.rmtree(checkout)
                record.update(kind="git", location=row["git"], commit=actual)
            else:
                snapshot = _copy_source(Path(row["directory"]), source)
                record.update(kind="local-build", location=row["directory"])
            record["source_sha256"] = snapshot
            check_build_tools(source)
            build_folder = folder / (name + "-built")
            build_folder.mkdir()
            build_callback(source, build_folder)
            wheels = list(build_folder.glob("*.whl"))
            if len(wheels) != 1:
                raise PackageError(
                    "SOURCE_BUILD_FAILED", "Source build must produce exactly one compatible wheel."
                )
            path = folder / wheels[0].name
            shutil.move(wheels[0], path)
            shutil.rmtree(build_folder)
            shutil.rmtree(source)
            record["build_isolation"] = False
            record["build_tools"] = {
                key: importlib.metadata.version(key)
                for key in ("pip", "setuptools", "wheel")
                if importlib.util.find_spec(key)
            }
        version, tags = _wheel_identity(path, name)
        record.update(
            version=version,
            wheel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            wheel_tags=tags,
            python_abi=sysconfig.get_config_var("SOABI"),
            python=sys.version.split()[0],
        )
        paths[name] = path.as_uri()
        records.append(record)
    return paths, records
