"""Bounded local delivery metadata and immutable result envelopes.

This module never sends a request to the cloud or executes analysis code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from pydantic import BaseModel, ConfigDict, Field, field_validator

_RECORD_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
MAX_OUTBOX_BYTES = 8 * 1024 * 1024
MAX_OUTBOX_ITEM_BYTES = 3 * 1024 * 1024
MAX_SHARING_BYTES = 2 * 1024 * 1024
MAX_SHARING_RECORDS = 1000
MAX_HISTORY_READ_BYTES = 32 * 1024 * 1024

class DesktopError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code, self.status_code = code, status_code


class InputFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    data_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OutboxItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record: dict
    input_files: list[InputFile] = Field(max_length=20)


def _encoded_outbox(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise DesktopError("INVALID_OUTBOX", "The saved result must contain valid JSON data.") from exc


def _outbox_item(item: dict) -> dict:
    try:
        item = OutboxItem.model_validate(item).model_dump()
    except ValueError as exc:
        raise DesktopError("INVALID_OUTBOX", "The saved result or file references are invalid.") from exc
    record_id = item["record"].get("id")
    if not isinstance(record_id, str) or not _RECORD_ID.fullmatch(record_id):
        raise DesktopError("INVALID_OUTBOX", "The saved result identifier is invalid.")
    if "actor_uid" in item["record"]:
        try:
            actor_uid(item["record"]["actor_uid"])
        except ValueError as exc:
            raise DesktopError("INVALID_OUTBOX", "The saved result owner is invalid.") from exc
    references = [reference["id"] for reference in item["input_files"]]
    if len(set(references)) != len(references):
        raise DesktopError("INVALID_OUTBOX", "The input file references contain duplicates.")
    if len(_encoded_outbox(item)) > MAX_OUTBOX_ITEM_BYTES:
        raise DesktopError("OUTBOX_LIMIT", "A saved result exceeds the local archive limit.", 413)
    return item


def _read_outbox(path: Path) -> dict:
    value = _load_json(path, {"items": []}, maximum=MAX_OUTBOX_BYTES)
    if not isinstance(value, dict) or set(value) != {"items"} or not isinstance(value["items"], list):
        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved result archive cannot be read.")
    if len(value["items"]) > 20:
        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved result archive exceeds its limit.")
    items = [_outbox_item(item) for item in value["items"]]
    identifiers = [item["record"]["id"] for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved result archive contains duplicate identifiers.")
    return {"items": items}


def _write_outbox(path: Path, document: dict) -> None:
    encoded = _encoded_outbox(document)
    if len(encoded) > MAX_OUTBOX_BYTES or len(document["items"]) > 20:
        raise DesktopError("OUTBOX_FULL", "The local result archive is full. Synchronize pending results first.", 409)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_json(path: Path, default, *, maximum: int = 1024 * 1024):
    if path.is_symlink():
        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved local project state cannot be read.")
    if not path.is_file():
        return default
    if path.stat().st_size > maximum:
        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved local project state cannot be read.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DesktopError("CORRUPT_LOCAL_STATE", "The saved local project state cannot be read.") from exc



def actor_uid(value: str) -> str:
    try:
        valid = (isinstance(value, str) and 1 <= len(value) <= 128
                 and len(value.encode("utf-8")) <= 512
                 and not any(ord(char) < 32 or ord(char) == 127 for char in value))
    except UnicodeError:
        valid = False
    if not valid:
        raise ValueError("The result owner is invalid.")
    return value


class SharingError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=512)
    status: int = Field(default=0, ge=0, le=599)

    @field_validator("status")
    @classmethod
    def valid_status(cls, value):
        if value != 0 and value < 100:
            raise ValueError("The sharing status must be zero or a valid HTTP status.")
        return value

    @field_validator("message")
    @classmethod
    def valid_message(cls, value):
        if len(value.encode("utf-8")) > 2048:
            raise ValueError("The sharing error is too long.")
        return value


class RetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actor_uid: str
    _actor = field_validator("actor_uid")(actor_uid)


class SharingUpdate(RetryRequest):
    state: str = Field(pattern=r"^(?:pending|failed)$")
    error: SharingError | None = None


class SharingEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # Empty is retained only for ownerless legacy records. Such records cannot
    # be recovered with the new authenticated retry endpoint.
    actor_uid: str = Field(max_length=128)
    state: str = Field(pattern=r"^(?:pending|shared|failed|local_only)$")
    error: SharingError | None = None
    envelope_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("actor_uid")
    @classmethod
    def valid_actor(cls, value):
        return actor_uid(value) if value else value


def _network_artifact(output: dict) -> bool:
    data = output.get("data")
    return (output.get("type") == "plot" and isinstance(data, dict)
            and data.get("kind") == "network" and "artifact" in data)


def legacy_network_summary_v1(data: dict) -> str:
    """Pinned archive wording for queued records from previous app versions.

    Keep this helper stable: cloud idempotency hashes the complete request body.
    New executions additionally capture the actual text before history is saved.
    """
    title = data.get("title")
    # Match the existing web archive's first 256 UTF-16 units. Avoid retaining
    # a split surrogate, which cannot be written as a valid UTF-8 JSON string.
    name = (title.encode("utf-16-le")[:512].decode("utf-16-le", errors="ignore")
            if isinstance(title, str) and title.strip() else "Network chart")
    return (f"{name}: the complete network chart is stored on the computer where this run was executed. "
            "The chart was not uploaded. Export and share its HTML file to share the complete view.")


def network_summaries(record: dict) -> list[dict]:
    """Capture only short replacement texts, leaving full local plots intact."""
    return [{"index": index, "summary": legacy_network_summary_v1(output["data"])}
            for index, output in enumerate(record.get("outputs", []))
            if _network_artifact(output)]


class NetworkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    index: int = Field(ge=0, lt=20)
    summary: str = Field(min_length=1, max_length=512)

    @field_validator("summary")
    @classmethod
    def valid_summary(cls, value):
        if len(value.encode("utf-8")) > 2048:
            raise ValueError("The saved network summary is too long.")
        return value


class DesktopSharing:
    """Serialize calls under the enclosing project's lock.

    Durable console history owns the original context. This compact file owns
    delivery status; it cannot change the immutable request body or rerun code.
    """
    def __init__(self, path: Path, workspace):
        self.path, self.workspace = path, workspace
        self.outbox_path = path / "desktop-outbox.json"
        self.metadata_path = path / "desktop-sharing.json"

    def _history(self) -> list[dict]:
        value = _load_json(self.path / "console" / "history.json", {"history": []},
                           maximum=MAX_HISTORY_READ_BYTES)
        if (not isinstance(value, dict) or not isinstance(value.get("history"), list)
                or len(value["history"]) > 500
                or any(not isinstance(item, dict) for item in value["history"])):
            raise DesktopError("CORRUPT_LOCAL_STATE", "The saved result history cannot be read.")
        return value["history"]

    def _metadata(self) -> dict[str, dict]:
        value = _load_json(self.metadata_path, {"records": []}, maximum=MAX_SHARING_BYTES)
        if (not isinstance(value, dict) or set(value) != {"records"}
                or not isinstance(value["records"], list)
                or len(value["records"]) > MAX_SHARING_RECORDS):
            raise DesktopError("CORRUPT_LOCAL_STATE", "The saved sharing status cannot be read.")
        try:
            records = [SharingEntry.model_validate(item).model_dump(exclude_none=True)
                       for item in value["records"]]
        except ValueError as exc:
            raise DesktopError("CORRUPT_LOCAL_STATE", "The saved sharing status is invalid.") from exc
        if len({item["id"] for item in records}) != len(records):
            raise DesktopError("CORRUPT_LOCAL_STATE", "The saved sharing status contains duplicate results.")
        return {item["id"]: item for item in records}

    def _save_metadata(self, records: dict[str, dict]) -> None:
        # The live history and queue remain recoverable. Older terminal status
        # can expire with the bounded history, without discarding pending work.
        protected = {item.get("id") for item in self._history()}
        protected.update(item["record"]["id"] for item in _read_outbox(self.outbox_path)["items"])
        candidates = iter([key for key in records if key not in protected])
        sizes = {key: len(_encoded_outbox(entry)) for key, entry in records.items()}
        encoded_size = len(b'{"records":[]}') + sum(sizes.values()) + max(0, len(records) - 1)
        while len(records) > MAX_SHARING_RECORDS or encoded_size > MAX_SHARING_BYTES:
            key = next(candidates, None)
            if key is None:
                raise DesktopError("SHARING_LIMIT", "The local sharing status archive is full.", 409)
            encoded_size -= sizes[key] + (1 if len(records) > 1 else 0)
            records.pop(key)
        # Compact JSON and a measured write bound, rather than a larger indented
        # representation whose disk size differs from the bound.
        encoded = _encoded_outbox({"records": list(records.values())})
        descriptor, temporary = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.metadata_path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def envelope(record: dict) -> dict:
        context = record.get("sharing_context")
        if not isinstance(context, dict):
            raise DesktopError("SHARING_CONTEXT_MISSING", "The original result sharing context was not saved.", 409)
        cloud_record = {key: value for key, value in record.items()
                        if key not in {"sharing_context", "sharing", "local_only"}}
        outputs = record.get("outputs", [])
        expected = {index for index, output in enumerate(outputs) if _network_artifact(output)}
        if "network_summaries" in context:
            try:
                supplied = context["network_summaries"]
                if not isinstance(supplied, list) or len(supplied) > 20:
                    raise ValueError("Invalid network summary list.")
                entries = [NetworkSummary.model_validate(item) for item in supplied]
                replacements = {item.index: item.summary for item in entries}
                if len(replacements) != len(entries) or set(replacements) != expected:
                    raise ValueError("The saved network summaries do not match the original outputs.")
            except (ValueError, UnicodeError) as exc:
                raise DesktopError("CORRUPT_LOCAL_STATE", "The saved result sharing context is invalid.") from exc
        else:
            # Migration for history captured before this feature. The canonical
            # v1 formatter is permanently pinned, rather than today's UI text.
            replacements = {index: legacy_network_summary_v1(outputs[index]["data"]) for index in expected}
        if replacements:
            cloud_record["outputs"] = [{**output, "type": "text", "data": replacements[index]}
                                       if index in replacements else output
                                       for index, output in enumerate(outputs)]
        return {"record": cloud_record, "input_files": context.get("input_files")}

    @staticmethod
    def _digest(item: dict) -> str:
        return hashlib.sha256(_encoded_outbox(item)).hexdigest()

    def _record(self, record_id: str) -> dict:
        if not isinstance(record_id, str) or not _RECORD_ID.fullmatch(record_id):
            raise DesktopError("INVALID_OUTBOX", "The saved result identifier is invalid.")
        record = next((item for item in self._history() if item.get("id") == record_id), None)
        if record is None or "actor_uid" not in record:
            queued = next((item["record"] for item in _read_outbox(self.outbox_path)["items"]
                           if item["record"]["id"] == record_id), None)
            record = queued or record
        if record is None:
            raise DesktopError("SHARING_NOT_FOUND", "This result is no longer in the local history.", 404)
        return record

    @staticmethod
    def _owner(record: dict, actor: str) -> None:
        if record.get("actor_uid") != actor:
            raise DesktopError("RESULT_ACTOR", "This result belongs to another sign-in session.", 403)

    @staticmethod
    def failure(exc: Exception) -> dict:
        return {"code": getattr(exc, "code", "SHARING_STORAGE"),
                "message": str(exc)[:512] or "The result could not be queued for sharing.",
                "status": getattr(exc, "status_code", 500)}

    def _initial(self, record: dict, queued: bool) -> dict:
        context = record.get("sharing_context")
        local_only = (record.get("local_only") is True
                      or (isinstance(context, dict) and context.get("local_only") is True)
                      or (not queued and not isinstance(context, dict)))
        state = "local_only" if local_only else ("pending" if queued else "failed")
        item = {"id": record["id"], "actor_uid": record.get("actor_uid", ""), "state": state}
        if state == "failed":
            item["error"] = {"code": "SHARING_QUEUE_MISSING" if isinstance(context, dict) else "SHARING_CONTEXT_MISSING",
                             "message": "This result has not been queued for sharing.", "status": 409}
        return item

    def snapshot(self) -> dict:
        metadata = self._metadata()
        queued = {item["record"]["id"]: item for item in _read_outbox(self.outbox_path)["items"]}
        history = {item["id"]: item for item in self._history()
                   if isinstance(item.get("id"), str) and _RECORD_ID.fullmatch(item["id"])}
        records = []
        # Keep entries for queued legacy work even when its console run expired.
        ordered_ids = list(history) + [key for key in queued if key not in history]
        for key in ordered_ids:
            record = history.get(key) or queued[key]["record"]
            if "actor_uid" not in record and key in queued:
                record = queued[key]["record"]
            item = dict(metadata.get(key) or self._initial(record, key in queued))
            context = record.get("sharing_context")
            item.pop("envelope_hash", None)
            item["retryable"] = bool(
                item["state"] in {"pending", "failed"}
                and (isinstance(context, dict) or key in queued)
                and not (context or {}).get("local_only")
                and not record.get("local_only", False)
                and (item.get("error") or {}).get("code") not in {
                    "OUTBOX_LIMIT", "RESULT_LIMIT", "OUTBOX_CONFLICT", "RESULT_CONFLICT", "DATA_INTEGRITY"
                }
            )
            records.append(item)
        return {"records": records,
                "pending_count": sum(item["state"] == "pending" for item in records),
                "failed_count": sum(item["state"] == "failed" for item in records)}

    def _set(self, record_id: str, actor: str, state: str, *, error=None, digest=None) -> dict:
        metadata = self._metadata()
        previous = metadata.get(record_id)
        if previous and previous["actor_uid"] != actor:
            raise DesktopError("RESULT_ACTOR", "This result belongs to another sign-in session.", 403)
        if previous and previous["state"] == "shared":
            return previous
        entry = {"id": record_id, "actor_uid": actor, "state": state}
        if error:
            entry["error"] = SharingError.model_validate(error).model_dump(exclude_none=True)
        if digest or (previous or {}).get("envelope_hash"):
            entry["envelope_hash"] = digest or previous["envelope_hash"]
        metadata[record_id] = entry
        self._save_metadata(metadata)
        return entry

    def enqueue(self, item: dict) -> dict:
        item = _outbox_item(item)
        key, actor = item["record"]["id"], item["record"].get("actor_uid", "")
        if actor:
            try:
                actor_uid(actor)
            except ValueError as exc:
                raise DesktopError("RESULT_ACTOR", "The result owner is invalid.", 422) from exc
        metadata = self._metadata()
        prior = metadata.get(key)
        digest = self._digest(item)
        if prior and (prior["actor_uid"] != actor or
                      (prior.get("envelope_hash") and prior["envelope_hash"] != digest)):
            raise DesktopError("OUTBOX_CONFLICT", "A different result already uses this archive identifier.", 409)
        if prior and prior["state"] == "shared" and prior.get("envelope_hash"):
            return {"stored": True, "id": key}
        document = _read_outbox(self.outbox_path)
        previous = next((entry for entry in document["items"] if entry["record"]["id"] == key), None)
        history = next((record for record in self._history() if record.get("id") == key), None)
        if history and isinstance(history.get("sharing_context"), dict):
            context = history["sharing_context"]
            legacy_identical = ("network_summaries" not in context and previous is not None
                                and self._digest(previous) == digest)
            if self._digest(self.envelope(history)) != digest and not legacy_identical:
                raise DesktopError("OUTBOX_CONFLICT", "The original result sharing context cannot be changed.", 409)
            if history["sharing_context"].get("local_only"):
                raise DesktopError("SHARING_LOCAL_ONLY", "Results using local-only data cannot be shared.", 409)
        if prior and prior["state"] == "shared":
            return {"stored": True, "id": key}
        if previous is not None:
            if _encoded_outbox(previous) != _encoded_outbox(item):
                raise DesktopError("OUTBOX_CONFLICT", "A different result already uses this archive identifier.", 409)
        else:
            document["items"].append(item)
            _write_outbox(self.outbox_path, document)
        self._set(key, actor, "pending", digest=digest)
        return {"stored": True, "id": key}

    def after_execute(self, record: dict) -> dict:
        actor, key = record["actor_uid"], record["id"]
        if record["sharing_context"]["local_only"]:
            self._set(key, actor, "local_only")
        else:
            try:
                self.enqueue(self.envelope(record))
            except (DesktopError, OSError) as exc:
                self._set(key, actor, "failed", error=self.failure(exc))
        return next(item for item in self.snapshot()["records"] if item["id"] == key)

    def update(self, record_id: str, body: SharingUpdate) -> dict:
        record = self._record(record_id)
        self._owner(record, body.actor_uid)
        context = record.get("sharing_context")
        if record.get("local_only") is True or (context or {}).get("local_only"):
            raise DesktopError("SHARING_LOCAL_ONLY", "Results using local-only data cannot be shared.", 409)
        previous = self._metadata().get(record_id)
        if previous and previous["state"] == "shared":
            raise DesktopError("SHARING_COMPLETE", "This result has already been shared.", 409)
        if body.state == "pending" and not any(item["record"]["id"] == record_id
                                               for item in _read_outbox(self.outbox_path)["items"]):
            raise DesktopError("SHARING_QUEUE_MISSING", "Retry this result before marking it pending.", 409)
        self._set(record_id, body.actor_uid, body.state,
                  error=body.error.model_dump(exclude_none=True) if body.error else None)
        return next(item for item in self.snapshot()["records"] if item["id"] == record_id)

    def retry(self, record_id: str, actor: str) -> dict:
        record = self._record(record_id)
        self._owner(record, actor)
        previous = self._metadata().get(record_id)
        if previous and previous["state"] == "shared":
            return {"stored": False, "id": record_id, "state": "shared"}
        if record.get("local_only") is True or (record.get("sharing_context") or {}).get("local_only"):
            raise DesktopError("SHARING_LOCAL_ONLY", "Results using local-only data cannot be shared.", 409)
        try:
            context = record.get("sharing_context")
            queued = next((entry for entry in _read_outbox(self.outbox_path)["items"]
                           if entry["record"]["id"] == record_id), None)
            if isinstance(context, dict) and ("network_summaries" in context or queued is None):
                item = self.envelope(record)
            elif queued is not None:
                # Prior app versions stored actor/input references in the queue,
                # sometimes with raw network artifact outputs. Keep that exact
                # body; the frontend applies only the permanently pinned v1 text.
                item = queued
                self._owner(item["record"], actor)
            else:
                raise DesktopError("SHARING_CONTEXT_MISSING", "The original result sharing context was not saved.", 409)
            self.enqueue(item)
        except (DesktopError, OSError) as exc:
            self._set(record_id, actor, "failed", error=self.failure(exc))
            if isinstance(exc, OSError):
                raise DesktopError("SHARING_STORAGE", "The result could not be queued for sharing.", 500) from exc
            raise
        return {"stored": True, "id": record_id, "state": "pending"}

    def acknowledge(self, record_id: str, actor: str | None = None) -> dict:
        if not _RECORD_ID.fullmatch(record_id):
            raise DesktopError("INVALID_OUTBOX", "The saved result identifier is invalid.")
        document = _read_outbox(self.outbox_path)
        item = next((entry for entry in document["items"] if entry["record"]["id"] == record_id), None)
        previous = self._metadata().get(record_id)
        owner = item["record"].get("actor_uid", "") if item else (previous or {}).get("actor_uid", "")
        if actor is not None and owner != actor:
            raise DesktopError("RESULT_ACTOR", "This result belongs to another sign-in session.", 403)
        if item is not None or (previous and previous["state"] == "shared"):
            # Persist confirmed cloud success before deleting the pending body.
            # A crash here is recoverable by an idempotent repeated acknowledgement.
            self._set(record_id, owner, "shared", digest=self._digest(item) if item else None)
        document["items"] = [entry for entry in document["items"] if entry["record"]["id"] != record_id]
        _write_outbox(self.outbox_path, document)
        return {"deleted": True, "id": record_id}

    def outbox(self) -> dict:
        return _read_outbox(self.outbox_path)
