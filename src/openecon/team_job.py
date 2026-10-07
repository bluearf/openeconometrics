"""One isolated Cloud Run Job task; never use this inside the control API.

The task identity must have no IAM permissions. Its only data capabilities are
short-lived signed URLs and one size-bounded signed POST policy for this run.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID, uuid4

MAX_INPUT_FILE = 24 * 1024 * 1024
MAX_INPUT_TOTAL = 64 * 1024 * 1024
MAX_FILES = 20
MAX_GENERATED_TOTAL = 16 * 1024 * 1024
MAX_RESULT_BYTES = 24 * 1024 * 1024 - 65536
MAX_MANIFEST_BYTES = 1024 * 1024


class JobInputError(ValueError):
    """An execution manifest, transfer, or generated artifact is invalid."""


def safe_filename(name: str) -> str:
    if (not isinstance(name, str) or not 1 <= len(name) <= 180
            or name != name.strip() or name.startswith(".")
            or re.search(r"[\x00-\x1f\x7f/\\:]", name)):
        raise JobInputError("Execution files must have simple, non-hidden filenames.")
    return name


def storage_url(value: str, *, signed: bool = True, generation: bool = False) -> str:
    if not isinstance(value, str) or not value.isascii() or len(value) > 16384:
        raise JobInputError("A bounded Cloud Storage HTTPS URL is required.")
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == "https" and parsed.netloc == "storage.googleapis.com"
                 and not parsed.fragment and parsed.path.startswith("/")
                 and len(parsed.path.split("/")) >= (3 if signed else 2)
                 and not any(ch.isspace() for ch in value))
        query = parse_qs(parsed.query)
        if signed:
            valid = (valid and len(query.get("X-Goog-Signature", [])) == 1
                     and bool(query["X-Goog-Signature"][0])
                     and len(query.get("X-Goog-Expires", [])) == 1
                     and 0 < int(query["X-Goog-Expires"][0]) <= 1800)
        else:
            valid = valid and not parsed.query
        if generation:
            valid = (valid and len(query.get("generation", [])) == 1
                     and query["generation"][0].isdigit())
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise JobInputError("Only scoped, short-lived Cloud Storage URLs are accepted.")
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise JobInputError("Cloud Storage transfers must not redirect.")


def download(url: str, limit: int) -> bytes:
    storage_url(url)
    with build_opener(_NoRedirect()).open(url, timeout=15) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > limit:
            raise JobInputError("The execution input exceeds its size limit.")
        content = response.read(limit + 1)
    if len(content) > limit:
        raise JobInputError("The execution input exceeds its size limit.")
    return content


def _validate_upload(upload: dict) -> dict:
    if not isinstance(upload, dict) or not isinstance(upload.get("fields"), dict):
        raise JobInputError("The execution requires a signed output POST policy.")
    storage_url(upload.get("url"), signed=False)
    fields = upload["fields"]
    if (not 1 <= len(fields) <= 20 or "key" not in fields or "policy" not in fields
            or "x-goog-signature" not in fields
            or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", k)
                   or k.lower() == "file" or not isinstance(v, str)
                   or len(v) > 16384 or "\r" in v or "\n" in v for k, v in fields.items())
            or sum(len(k) + len(v) for k, v in fields.items()) > 32768):
        raise JobInputError("The execution output POST policy is invalid.")
    return upload


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise JobInputError("The execution manifest must be an object.")
    try:
        execution_id = manifest["execution_id"]
        if not isinstance(execution_id, str):
            raise ValueError
        UUID(execution_id)
    except (KeyError, ValueError, AttributeError):
        raise JobInputError("A valid execution identifier is required.") from None
    code = manifest.get("code")
    duration = manifest.get("timeout_seconds", 120)
    if not isinstance(code, str) or not code.strip() or len(code) > 64000:
        raise JobInputError("Execution code must contain between 1 and 64,000 characters.")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not .05 <= duration <= 120):
        raise JobInputError("Execution timeout must be between 0.05 and 120 seconds.")
    files = manifest.get("files", [])
    if not isinstance(files, list) or len(files) > MAX_FILES:
        raise JobInputError("An execution accepts at most 20 input files.")
    names = set()
    for item in files:
        if not isinstance(item, dict):
            raise JobInputError("Each input file needs a name and download URL.")
        name = safe_filename(item.get("name"))
        if name in names:
            raise JobInputError("Input filenames must be unique.")
        names.add(name)
        storage_url(item.get("download_url"), generation=True)
    _validate_upload(manifest.get("output_upload"))
    return {**manifest, "files": files, "timeout_seconds": duration}


class _JobWorkspace:
    """Only the child lifecycle adapter: no history database in the compute job."""
    def __init__(self, directory: Path):
        self.path = directory

    def console_history(self):
        return []

    def append_console_history(self, record):
        pass


def collect_generated(directory: Path, input_names: set[str]) -> list[dict]:
    candidates = []
    with os.scandir(directory) as entries:
        for index, entry in enumerate(entries):
            if index >= 256:
                raise JobInputError("The execution created too many filesystem entries.")
            if entry.name in input_names or entry.name.startswith("."):
                continue
            if entry.is_symlink():
                raise JobInputError("Generated files must not be symbolic links.")
            if entry.is_dir(follow_symlinks=False):
                continue  # Only top-level files are artifact outputs.
            safe_filename(entry.name)
            candidates.append(entry.name)
            if len(candidates) > MAX_FILES:
                raise JobInputError("An execution can save at most 20 generated files.")
    result, total = [], 0
    root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in sorted(candidates):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
            with os.fdopen(fd, "rb") as stream:
                details = os.fstat(stream.fileno())
                if not stat.S_ISREG(details.st_mode):
                    raise JobInputError("Generated artifacts must be regular files.")
                remaining = MAX_GENERATED_TOTAL - total
                if details.st_size > remaining:
                    raise JobInputError("Generated files exceed the 16 MiB aggregate limit.")
                content = stream.read(remaining + 1)
                total += len(content)
                if total > MAX_GENERATED_TOTAL:
                    raise JobInputError("Generated files exceed the 16 MiB aggregate limit.")
                result.append({"name": name, "size": len(content),
                               "content_base64": base64.b64encode(content).decode("ascii")})
    finally:
        os.close(root_fd)
    return result


def run_manifest(manifest: dict, directory: Path) -> dict:
    from openecon.console import ConsoleSession

    manifest = validate_manifest(manifest)
    directory.mkdir(parents=True, exist_ok=False)
    total = 0
    for item in manifest["files"]:
        content = download(item["download_url"], min(MAX_INPUT_FILE, MAX_INPUT_TOTAL - total))
        total += len(content)
        (directory / item["name"]).write_bytes(content)
    session = ConsoleSession(_JobWorkspace(directory))
    try:
        record = session.execute(manifest["code"], timeout_seconds=manifest["timeout_seconds"])
    finally:
        session.close()
    record.update(id=manifest["execution_id"], variables=[], state_reset=True,
                  session_generation=1, active_session_generation=1)
    try:
        generated = collect_generated(directory, {item["name"] for item in manifest["files"]})
    except (JobInputError, OSError):
        generated = []
        record.update(status="error", error={"type": "ARTIFACT_LIMIT",
            "message": "Generated files must be ordinary top-level files, at most 20 files and 16 MiB in total.",
            "traceback": ""})
    return {"execution_id": manifest["execution_id"], "record": record, "generated_files": generated}


def upload_result(upload: dict, result: dict) -> None:
    _validate_upload(upload)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        raise JobInputError("The serialized execution result exceeds the output limit.")
    boundary = "openecon-" + uuid4().hex
    parts = []
    for name, value in upload["fields"].items():
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                      f'{value}\r\n').encode("utf-8"))
    parts.extend([
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="result.json"\r\n'
         'Content-Type: application/json\r\n\r\n').encode("ascii"), encoded,
        f'\r\n--{boundary}--\r\n'.encode("ascii"),
    ])
    request = Request(upload["url"], data=b"".join(parts), method="POST",
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with build_opener(_NoRedirect()).open(request, timeout=30) as response:
        if not 200 <= response.status < 300:
            raise JobInputError("The execution output could not be stored.")
        response.read(4096)


def main() -> int:
    manifest = None
    try:
        input_url = storage_url(os.environ.get("OPENECON_RUN_INPUT_URL"))
        manifest = validate_manifest(json.loads(download(input_url, MAX_MANIFEST_BYTES)))
        with tempfile.TemporaryDirectory(prefix="openecon-run-") as temporary:
            result = run_manifest(manifest, Path(temporary) / "workspace")
            upload_result(manifest["output_upload"], result)
        return 0
    except Exception:
        # Capability URLs and Python outputs never go to shared platform logs.
        if manifest is not None:
            now = datetime.now(timezone.utc).isoformat()
            failure = {"execution_id": manifest["execution_id"], "generated_files": [],
                       "record": {"id": manifest["execution_id"], "code": manifest["code"],
                           "status": "error", "stdout": "", "outputs": [], "variables": [],
                           "state_reset": True, "session_generation": 1, "active_session_generation": 1,
                           "created_at": now, "completed_at": now, "duration_ms": 0,
                           "error": {"type": "JOB_FAILED", "message": "The isolated execution could not complete.",
                                     "traceback": ""}}}
            try:
                upload_result(manifest["output_upload"], failure)
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
