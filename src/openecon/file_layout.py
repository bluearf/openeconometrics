"""Inert, versioned project explorer metadata; names never become disk paths."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import stat
import unicodedata

from openecon import script_contracts

MAX_ENTRIES = 2000
MAX_FOLDERS = 200
MAX_DEPTH = 16
MAX_BYTES = 1024 * 1024
MAX_VERSION = 2**63 - 1
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_FOLDER_ID = re.compile(r"[0-9a-f]{32}\Z")
_SCRIPT_ID = re.compile(r"(?:analysis|[0-9a-f]{32})\Z")
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z", re.IGNORECASE)
_UNSAFE = '/\\:<>?*|"'


class FileLayoutError(ValueError):
    def __init__(self, message: str, code: str = "INVALID_FILE_LAYOUT"):
        super().__init__(message)
        self.code = code


def version(value: object) -> int:
    if type(value) is not int or not 0 <= value < MAX_VERSION:
        raise FileLayoutError("The file layout version is invalid.")
    return value


def safe_name(value: object) -> str:
    if not isinstance(value, str):
        raise FileLayoutError("Enter a file or folder name.")
    name = unicodedata.normalize("NFC", value)
    try:
        size = len(name.encode("utf-8"))
    except UnicodeError as exc:
        raise FileLayoutError("The file or folder name is invalid.") from exc
    if (not 1 <= size <= 180 or name != name.strip() or name.startswith(".")
            or name.endswith((".", " ")) or any(c in _UNSAFE for c in name)
            or any(unicodedata.category(c).startswith("C") or c in "\u2028\u2029" for c in name)
            or _DEVICE.fullmatch(name.split(".", 1)[0])):
        raise FileLayoutError("Use a short, safe file or folder name.")
    return name


def _extension(name: str) -> str:
    return Path(name).suffix.casefold()


def validate(value: object) -> dict:
    """Validate a bounded tree without resolving project data or following links."""
    if (not isinstance(value, dict) or set(value) != {"version", "entries"}
            or not isinstance(value.get("entries"), list) or len(value["entries"]) > MAX_ENTRIES):
        raise FileLayoutError("The file layout is invalid.")
    document = {"version": version(value["version"]), "entries": []}
    ids, siblings, folders = set(), set(), {}
    for row in value["entries"]:
        if not isinstance(row, dict) or set(row) != {"kind", "id", "name", "parent"}:
            raise FileLayoutError("An entry in the file layout is invalid.")
        kind, identifier, parent = row["kind"], row["id"], row["parent"]
        if (not isinstance(kind, str) or kind not in {"script", "dataset", "folder"}
                or not isinstance(identifier, str) or not _ID.fullmatch(identifier)
                or (kind, identifier) in ids
                or (kind == "folder" and not _FOLDER_ID.fullmatch(identifier))
                or (kind == "script" and not _SCRIPT_ID.fullmatch(identifier))
                or (parent is not None and (not isinstance(parent, str) or not _FOLDER_ID.fullmatch(parent)))):
            raise FileLayoutError("The file or folder ID is invalid.")
        name = safe_name(row["name"])
        if kind == "script":
            try:
                script_contracts.script_name(name)
            except script_contracts.ScriptValidationError as exc:
                raise FileLayoutError(str(exc)) from exc
            if identifier == "analysis" and not name.endswith(".py"):
                raise FileLayoutError("Keep the .py extension of the main analysis file.")
        sibling = (parent, name.casefold())
        if sibling in siblings:
            raise FileLayoutError("This name is already in use in the same folder.")
        ids.add((kind, identifier))
        siblings.add(sibling)
        normalized = {"kind": kind, "id": identifier, "name": name, "parent": parent}
        document["entries"].append(normalized)
        if kind == "folder":
            folders[identifier] = normalized
    if len(folders) > MAX_FOLDERS:
        raise FileLayoutError("The project folder limit has been reached.")
    for row in document["entries"]:
        parent, seen, depth = row["parent"], ({row["id"]} if row["kind"] == "folder" else set()), 0
        while parent is not None:
            if parent not in folders or parent in seen:
                raise FileLayoutError("The folder relationship is invalid.")
            seen.add(parent)
            depth += 1
            if depth > MAX_DEPTH:
                raise FileLayoutError("Folders can be nested up to 16 levels deep.")
            parent = folders[parent]["parent"]
    if len(json.dumps(document, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        raise FileLayoutError("The file layout exceeds the size limit.")
    return document


def inventory(scripts: list[dict], datasets: list[dict]) -> list[dict]:
    return [{"kind": kind, "id": item["id"], "name": item["name"], "parent": None}
            for kind, records in (("script", scripts), ("dataset", datasets)) for item in records]


def stored_document(value: object) -> dict | None:
    if value is None:
        return None
    try:
        if not isinstance(value, dict):
            raise ValueError("Invalid explorer record.")
        return validate({key: value[key] for key in ("version", "entries")})
    except (ValueError, KeyError, TypeError) as exc:
        raise FileLayoutError("The saved file layout could not be read.", "CORRUPT_FILE_LAYOUT") from exc


def _default_name(value: str) -> str:
    """Display old imports safely, retaining their immutable original metadata."""
    name = unicodedata.normalize("NFC", value)
    name = "".join("_" if c in _UNSAFE or unicodedata.category(c).startswith("C")
                   or c in "\u2028\u2029" else c for c in name).strip().lstrip(".").rstrip(" .")
    if not name:
        name = "data"
    if _DEVICE.fullmatch(name.split(".", 1)[0]):
        name = "_" + name
    suffix = Path(name).suffix
    stem = name[:-len(suffix)] if suffix else name
    while len((stem + suffix).encode("utf-8")) > 180:
        if stem:
            stem = stem[:-1]
        else:
            suffix = suffix[:-1]
    return safe_name(stem.rstrip(" .") + suffix)


def _unique_name(name: str, used: set[str]) -> str:
    if name.casefold() not in used:
        return name
    suffix = Path(name).suffix
    base = name[:-len(suffix)] if suffix else name
    for number in range(2, MAX_ENTRIES + 2):
        tail = f" ({number})" + suffix
        stem = base
        while len((stem + tail).encode("utf-8")) > 180:
            stem = stem[:-1]
        candidate = stem.rstrip(" .") + tail
        if candidate.casefold() not in used:
            return candidate
    raise FileLayoutError("The filename limit has been reached.")


def reconcile(stored: dict | None, current: list[dict]) -> dict:
    """Keep saved ordering and names; append freshly created files at the root."""
    old = validate(stored) if stored is not None else {"version": 0, "entries": []}
    records = {(row["kind"], row["id"]): row for row in current}
    rows = [deepcopy(row) for row in old["entries"]
            if row["kind"] == "folder" or (row["kind"], row["id"]) in records]
    present = {(row["kind"], row["id"]) for row in rows}
    used = {row["name"].casefold() for row in rows if row["parent"] is None}
    for row in current:
        if (row["kind"], row["id"]) not in present:
            name = _unique_name(_default_name(row["name"]), used)
            rows.append({**row, "name": name, "parent": None})
            used.add(name.casefold())
    return validate({"version": old["version"], "entries": rows})


def validate_inventory(document: dict, current: list[dict]) -> dict:
    validated = validate(document)
    records = {(row["kind"], row["id"]): row for row in current}
    supplied = {(row["kind"], row["id"]) for row in validated["entries"] if row["kind"] != "folder"}
    if supplied != set(records):
        raise FileLayoutError("The project files have changed. Reload the current list.", "FILE_INVENTORY_CONFLICT")
    for row in validated["entries"]:
        if row["kind"] == "dataset" and _extension(row["name"]) != _extension(records[(row["kind"], row["id"])]["name"]):
            raise FileLayoutError("Keep the data file extension.")
    return validated


def source_name(document: dict, identifier: str, fallback: str) -> str:
    """A validated explorer alias is the effective name of its source document."""
    return next((row["name"] for row in document["entries"]
                 if row["kind"] == "script" and row["id"] == identifier), fallback)


def read(path: Path) -> dict | None:
    """Bounded descriptor read; a saved explorer record cannot be a symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FileLayoutError("The saved file layout could not be read.", "CORRUPT_FILE_LAYOUT") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("Invalid explorer record.")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ValueError("Explorer record too large.")
        return validate(json.loads(payload.decode("utf-8")))
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise FileLayoutError("The saved file layout could not be read.", "CORRUPT_FILE_LAYOUT") from exc
    finally:
        os.close(fd)


@contextmanager
def storage_lock(directory: Path):
    """Serialize local layout replacements across processes on Unix desktops."""
    if os.name != "posix":
        yield
        return
    import fcntl
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
