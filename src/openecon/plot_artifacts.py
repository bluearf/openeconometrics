"""Content-addressed local network plots; history carries only a small reference.

Large graphs never travel through the small worker pipe or inflate history.
This store is scoped to one workspace and deliberately is not a cloud URL.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time

from openecon.data import DataError

MAX_PLOT_BYTES = 128 * 1024 * 1024
MAX_STORE_BYTES = 1024 * 1024 * 1024
INLINE_PLOT_BYTES = 256 * 1024
PLOT_GRACE_SECONDS = 300
MAX_HISTORY_BYTES = 32 * 1024 * 1024  # Physical indented JSON; logical history is 8 MiB.
MAX_MANAGED_FILES = 20_000
_ID = re.compile(r"[0-9a-f]{64}\Z")
_HEADER = re.compile(rb'\{\s*"kind"\s*:\s*"network"\s*,')


def _directory(workspace: str | Path, *, create=False) -> Path:
    root = Path(workspace).resolve()
    console = root / "console"
    directory = console / "network-plots"
    if console.is_symlink() or directory.is_symlink():
        raise DataError("Linked plot storage is not allowed.", "PLOT_INTEGRITY")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    if (console.exists() and not console.is_dir()) or (directory.exists() and not directory.is_dir()):
        raise DataError("Plot storage is not a directory.", "PLOT_INTEGRITY")
    return directory


def validate_reference(value) -> dict:
    if (type(value) is not dict or set(value) != {"version", "id", "bytes"}
            or value["version"] != 1 or type(value["version"]) is not int
            or type(value["id"]) is not str or not _ID.fullmatch(value["id"])
            or type(value["bytes"]) is not int or not 1 <= value["bytes"] <= MAX_PLOT_BYTES):
        raise DataError("The network plot reference is invalid.", "PLOT_INTEGRITY")
    return dict(value)


def _history_ids(workspace, supplied):
    if supplied is None:
        path = Path(workspace).resolve() / "console" / "history.json"
        if path.is_symlink():
            raise DataError("Linked analysis history cannot be used to collect plots.", "PLOT_INTEGRITY")
        if not path.exists():
            supplied = []
        else:
            if not path.is_file() or path.stat().st_size > MAX_HISTORY_BYTES:
                raise DataError("Analysis history is too large or invalid; local plots were preserved.", "PLOT_INTEGRITY")
            try:
                with path.open("rb") as stream:
                    raw = stream.read(MAX_HISTORY_BYTES + 1)
                if len(raw) > MAX_HISTORY_BYTES:
                    raise ValueError("history size")
                value = json.loads(raw)
                if type(value) is not dict:
                    raise ValueError("history shape")
                supplied = value.get("history")
            except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                raise DataError("Analysis history cannot be verified; local plots were preserved.", "PLOT_INTEGRITY") from exc
    if type(supplied) is not list or len(supplied) > 500:
        raise DataError("Analysis history cannot be verified; local plots were preserved.", "PLOT_INTEGRITY")
    referenced = set()
    try:
        # Bound work and the logical payload before collecting any references.
        size = 0
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for chunk in encoder.iterencode(supplied):
            size += len(chunk.encode("utf-8"))
            if size > 8 * 1024 * 1024 + 4096:
                raise ValueError("history size")
        for record in supplied:
            if type(record) is not dict:
                raise ValueError("history record")
            outputs = record.get("outputs", [])
            if type(outputs) is not list or len(outputs) > 20:
                raise ValueError("history outputs")
            for output in outputs:
                if type(output) is not dict:
                    raise ValueError("history output")
                data = output.get("data")
                if output.get("type") == "plot" and type(data) is dict and "artifact" in data:
                    referenced.add(validate_reference(data["artifact"])["id"])
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise DataError("Analysis history cannot be verified; local plots were preserved.", "PLOT_INTEGRITY") from exc
    return referenced


def collect_plot_garbage(workspace: str | Path, *, history=None, now=None,
                         grace_seconds=PLOT_GRACE_SECONDS) -> dict:
    """Delete only verified, expired cache files absent from bounded history.

    SHA filenames, a network PlotSpec header and the complete content checksum
    identify managed cache entries. Foreign files, links, corrupt files, current
    references and newly created files are preserved. Unlink uses an open
    directory descriptor, so a changed directory link cannot target user data.
    The grace period protects plots created before their execution is recorded.
    """
    if (isinstance(grace_seconds, bool) or not isinstance(grace_seconds, (int, float))
            or not 0 <= grace_seconds <= 86400):
        raise DataError("Plot collection grace must be between 0 and 86,400 seconds.", "PLOT_INTEGRITY")
    current = time.time() if now is None else now
    if (isinstance(current, bool) or not isinstance(current, (int, float))
            or not -1e12 < current < 1e12):
        raise DataError("The plot collection time is invalid.", "PLOT_INTEGRITY")
    directory = _directory(workspace)
    result = {"deleted_files": 0, "deleted_bytes": 0, "retained_referenced": 0,
              "retained_young": 0, "retained_unverified": 0, "retained_bytes": 0,
              "cleanup_supported": True}
    if not directory.exists():
        return result
    referenced = _history_ids(workspace, history)
    if (os.scandir not in os.supports_fd or os.unlink not in os.supports_dir_fd
            or os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd):
        # Windows does not offer these directory-relative Unix operations.
        # Retaining cache files is safer than falling back to race-prone paths;
        # storing/reopening plots still works, with the same explicit disk cap.
        result["cleanup_supported"] = False
        for path in directory.iterdir():
            if path.suffix == ".json" and _ID.fullmatch(path.stem) and not path.is_symlink() and path.is_file():
                result["retained_unverified"] += 1
                result["retained_bytes"] += path.stat().st_size
                if result["retained_unverified"] > MAX_MANAGED_FILES:
                    raise DataError("The local plot cache has too many entries to inspect safely.", "PLOT_LIMIT")
        return result
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        raise DataError("The local plot directory cannot be verified.", "PLOT_INTEGRITY") from exc
    try:
        candidates = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if not entry.name.endswith(".json") or not _ID.fullmatch(entry.name[:-5]):
                    continue
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                if len(candidates) >= MAX_MANAGED_FILES:
                    raise DataError("The local plot cache has too many entries to collect safely.", "PLOT_LIMIT")
                candidates.append((entry.name, info))
        for name, info in candidates:
            if name[:-5] in referenced:
                result["retained_referenced"] += 1
                result["retained_bytes"] += info.st_size
                continue
            if current - info.st_mtime < grace_seconds:
                result["retained_young"] += 1
                result["retained_bytes"] += info.st_size
                continue
            verified = False
            if 1 <= info.st_size <= MAX_PLOT_BYTES:
                try:
                    fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
                    with os.fdopen(fd, "rb") as source:
                        captured = os.fstat(source.fileno())
                        if stat.S_ISREG(captured.st_mode) and _HEADER.match(source.read(256)):
                            source.seek(0)
                            verified = hashlib.file_digest(source, "sha256").hexdigest() == name[:-5]
                    latest = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    verified &= (latest.st_dev, latest.st_ino, latest.st_size, latest.st_mtime_ns) == (
                        captured.st_dev, captured.st_ino, captured.st_size, captured.st_mtime_ns)
                except (OSError, ValueError):
                    verified = False
            if verified:
                os.unlink(name, dir_fd=descriptor)
                result["deleted_files"] += 1
                result["deleted_bytes"] += info.st_size
            else:
                result["retained_unverified"] += 1
                result["retained_bytes"] += info.st_size
        return result
    finally:
        os.close(descriptor)


def store_plot(workspace: str | Path, plot: dict) -> dict:
    """Atomically store a validated plot, preserving every displayed node/edge."""
    from openecon_charts import PlotSpec
    if plot.get("kind") != "network":
        raise DataError("Only network plots use local graph storage.", "PLOT_INTEGRITY")
    from openecon_charts.timeline import unpack
    plot = PlotSpec(**unpack(plot)).transport_dump()  # Check expanded limits before compact storage.
    # Managed entries have a canonical identifying header regardless of the
    # caller's dictionary insertion order. Nested data stays unchanged.
    plot = {"kind": plot["kind"], **{key: value for key, value in plot.items() if key != "kind"}}
    directory = _directory(workspace, create=True)
    collection = collect_plot_garbage(workspace)
    existing_bytes = sum(path.stat().st_size for path in directory.glob("*.json")
                         if _ID.fullmatch(path.stem) and path.is_file() and not path.is_symlink())
    digest, size = hashlib.sha256(), 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".writing-", delete=False) as target:
            temporary = Path(target.name)
            encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for chunk in encoder.iterencode(plot):
                encoded = chunk.encode("utf-8")
                size += len(encoded)
                if size > MAX_PLOT_BYTES:
                    raise DataError("The network plot exceeds 128 MiB; choose a smaller view.", "PLOT_LIMIT")
                digest.update(encoded)
                target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        identifier = digest.hexdigest()
        destination = directory / f"{identifier}.json"
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise DataError("The network plot storage was changed.", "PLOT_INTEGRITY")
        if destination.exists():
            checked_plot_path(workspace, {"version": 1, "id": identifier, "bytes": size})
        else:
            if existing_bytes + size > MAX_STORE_BYTES:
                message = ("Local network plot storage exceeds 1 GiB. Referenced plots and plots created in the last five minutes are preserved; export views or clear old analysis history."
                           if collection["cleanup_supported"] else
                           "Local network plot storage exceeds 1 GiB. Safe automatic cache cleanup is unavailable on this platform; export views and manage the local plot cache explicitly.")
                raise DataError(message, "PLOT_LIMIT")
            os.replace(temporary, destination)
        return {"kind": "network", "title": plot["title"], "x_label": "", "y_label": "", "data": [],
                "sample_n": plot["sample_n"], "total_n": plot["total_n"], "dropped_n": 0,
                "artifact": {"version": 1, "id": identifier, "bytes": size}}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checked_plot_path(workspace: str | Path, reference: dict) -> Path:
    reference = validate_reference(reference)
    path = _directory(workspace) / f"{reference['id']}.json"
    if path.is_symlink() or not path.is_file():
        raise DataError("The local network plot is unavailable in this workspace.", "NOT_FOUND")
    if path.stat().st_size != reference["bytes"]:
        raise DataError("The saved network plot size changed.", "PLOT_INTEGRITY")
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != reference["id"]:
            raise DataError("The saved network plot checksum changed.", "PLOT_INTEGRITY")
    return path


def history_reference(history: list[dict], identifier: str) -> dict:
    """Serve only a plot actually referenced by this project's recorded history."""
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
        raise DataError("The network plot identifier is invalid.", "NOT_FOUND")
    for record in reversed(history):
        for output in record.get("outputs", []):
            data = output.get("data")
            if output.get("type") == "plot" and isinstance(data, dict):
                reference = data.get("artifact")
                if isinstance(reference, dict) and reference.get("id") == identifier:
                    return validate_reference(reference)
    raise DataError("The network plot is not part of this project's history.", "NOT_FOUND")
