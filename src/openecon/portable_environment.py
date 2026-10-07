"""Inert, bounded environment documents with explicit runtime compatibility."""

from __future__ import annotations

import hashlib
import hmac
import json
import platform
import re
import sys

from openecon.project_packages import PackageError, validate_manifest

FORMAT = "openecon.environment.v1"
MAX_DOCUMENT_BYTES = 64 * 1024


def runtime_identity() -> dict:
    return {
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "implementation": sys.implementation.name,
        "abi": sys.implementation.cache_tag,
    }


def _bytes(value) -> bytes:
    try:
        data = json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise PackageError(
            "INVALID_MANIFEST", "The environment document is not valid JSON."
        ) from exc
    if len(data) > MAX_DOCUMENT_BYTES:
        raise PackageError("INVALID_MANIFEST", "The environment document exceeds 64 KiB.")
    return data


def export_document(manifest: dict) -> dict:
    payload = {
        "format": FORMAT,
        "runtime": runtime_identity(),
        "manifest": validate_manifest(manifest),
    }
    document = {**payload, "sha256": hashlib.sha256(_bytes(payload)).hexdigest()}
    _bytes(document)
    return document


def validate_document(value, *, core: dict, python: str) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "runtime", "manifest", "sha256"}
        or value["format"] != FORMAT
        or not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise PackageError(
            "INVALID_MANIFEST", "Choose an exported OpenEconometrics environment document."
        )
    _bytes(value)
    payload = {key: value[key] for key in ("format", "runtime", "manifest")}
    expected = hashlib.sha256(_bytes(payload)).hexdigest()
    if not hmac.compare_digest(value["sha256"], expected):
        raise PackageError(
            "MANIFEST_CHECKSUM",
            "The environment document has changed or is damaged. Export it again.",
        )
    if value["runtime"] != runtime_identity():
        raise PackageError(
            "INCOMPATIBLE_PLATFORM",
            "This environment uses a different platform, architecture or Python ABI.",
        )
    return validate_manifest(value["manifest"], core=core, python=python)
