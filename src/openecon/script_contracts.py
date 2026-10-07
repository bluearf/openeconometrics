"""Inert named-editor metadata shared by local and team persistence."""
from __future__ import annotations

import re
import unicodedata

MAX_SCRIPTS = 200
MAX_SCRIPT_CHARACTERS = 64000
MAX_SCRIPT_BYTES = 8 * 1024 * 1024
MAX_SCRIPT_VERSION = 2**63 - 1
SOURCE_EXTENSIONS = frozenset({".py", ".md", ".tex"})
_ID = re.compile(r"[0-9a-f]{32}\Z")
_WINDOWS_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z", re.IGNORECASE)


class ScriptValidationError(ValueError):
    def __init__(self, message: str, code: str = "INVALID_SCRIPT"):
        super().__init__(message)
        self.code = code


def script_id(value: str, *, creating: bool = False) -> str:
    if isinstance(value, str) and ((value == "analysis" and not creating) or _ID.fullmatch(value)):
        return value
    raise ScriptValidationError("The file ID is invalid.",
                                "INVALID_SCRIPT" if creating else "NOT_FOUND")


def script_name(value: str) -> str:
    if not isinstance(value, str):
        raise ScriptValidationError("Enter a filename.")
    name = unicodedata.normalize("NFC", value)
    try:
        size = len(name.encode("utf-8"))
    except UnicodeError as exc:
        raise ScriptValidationError("The filename is invalid.") from exc
    stem, separator, extension = name.rpartition(".")
    if (not 1 <= size <= 180 or name != name.strip() or name.startswith(".")
            or not separator or "." + extension not in SOURCE_EXTENSIONS
            or not stem or stem.endswith((" ", "."))
            or any(c in name for c in '/\\:<>?*|"')
            or any(unicodedata.category(c).startswith("C") or c in "\u2028\u2029" for c in name)
            or _WINDOWS_DEVICE.fullmatch(name.split(".", 1)[0])):
        raise ScriptValidationError("Use a short, safe .py, .md or .tex filename.")
    return name


def require_python(name: str) -> None:
    """Identified source documents cannot accidentally enter the Python runner."""
    if not script_name(name).endswith(".py"):
        raise ScriptValidationError("This document cannot be run as Python code.",
                                    "DOCUMENT_NOT_EXECUTABLE")


def script_code(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_SCRIPT_CHARACTERS:
        raise ScriptValidationError("The file can contain up to 64,000 characters.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ScriptValidationError("The file contains invalid text.") from exc
    return value


def script_version(value: int) -> int:
    if type(value) is not int or not 0 <= value < MAX_SCRIPT_VERSION:
        raise ScriptValidationError("The file version is invalid.")
    return value


def unique_script_name(name: str, used: list[str]) -> str:
    name = script_name(name)
    existing = {unicodedata.normalize("NFC", item).casefold() for item in used}
    if name.casefold() not in existing:
        return name
    stem, _, extension = name.rpartition(".")
    for suffix in range(2, MAX_SCRIPTS + 2):
        tail = f"_{suffix}.{extension}"
        shortened = stem
        while len((shortened + tail).encode("utf-8")) > 180:
            shortened = shortened[:-1]
        candidate = shortened.rstrip(" .") + tail
        if candidate.casefold() not in existing:
            return script_name(candidate)
    raise ScriptValidationError("The project file limit has been reached.", "SCRIPT_LIMIT")


def validate_script_rows(value: object) -> list[dict]:
    """Validate a bounded index; it contains metadata and byte counts only."""
    if not isinstance(value, list) or len(value) >= MAX_SCRIPTS:
        raise ScriptValidationError("The file list is invalid.")
    rows, ids, names = [], set(), {"analysis.py"}
    for row in value:
        if not isinstance(row, dict) or set(row) != {"id", "name", "version", "size"}:
            raise ScriptValidationError("The file list is invalid.")
        identifier = script_id(row["id"], creating=True)
        name, version = script_name(row["name"]), script_version(row["version"])
        size = row["size"]
        if (identifier in ids or name.casefold() in names or type(size) is not int
                or not 0 <= size <= MAX_SCRIPT_CHARACTERS * 4):
            raise ScriptValidationError("The file list is invalid.")
        ids.add(identifier)
        names.add(name.casefold())
        rows.append({"id": identifier, "name": name, "version": version, "size": size})
    return rows


def metadata(document: dict) -> dict:
    return {key: document[key] for key in ("id", "name", "version")}
