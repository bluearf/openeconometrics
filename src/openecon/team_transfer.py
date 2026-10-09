"""Membership-scoped, immutable, bounded dataset transfer journals.

Only complete SHA-256-verified files enter the project catalogue. Parts remain
private objects, and every operation rechecks current membership. No dataframe
parsing, user code, signed URLs, or client filesystem paths enter this service.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from openecon.team_store import TeamError, filename, now

PART_BYTES = 4 * 1024**2
MAX_DATA_BYTES = 2 * 1024**3
MAX_PROJECT_DATA_BYTES = 8 * 1024**3
MAX_PARTS = MAX_DATA_BYTES // PART_BYTES
MAX_MANIFEST_BYTES = 256 * 1024
TRANSFER_VERSION = "chunked-v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[0-9a-f]{32}\Z")


class TransferBegin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=180)
    size_bytes: int = Field(ge=1, le=MAX_DATA_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _path(project_id, transfer_id):
    if not _ID.fullmatch(project_id) or not _ID.fullmatch(transfer_id):
        raise TeamError("INVALID_TRANSFER", "Invalid transfer identifier.", 422)
    return f"oe_projects/{project_id}/transfers/{transfer_id}"


def _public_file(metadata):
    return {key: value for key, value in metadata.items() if key != "blob"}


class DatasetTransfers:
    def __init__(self, store, storage):
        self.store, self.storage = store, storage

    def _active(self, db, project_id, transfer_id, user, *, ready=False):
        project = self.store._project(db, project_id, user, "editor")
        path = _path(project_id, transfer_id)
        record = db.get(path)
        if record is None:
            raise TeamError("NOT_FOUND", "Transfer not found.", 404)
        if record["uid"] != user.uid and project["owner_uid"] != user.uid:
            raise TeamError("ROLE_REQUIRED", "This transfer belongs to another member.", 403)
        if record["state"] == "cancelled":
            raise TeamError("TRANSFER_CANCELLED", "This transfer was cancelled.", 410)
        if record["state"] == "ready" and not ready:
            raise TeamError("TRANSFER_COMPLETE", "This transfer is already complete.", 409)
        if record["state"] != "ready" and record["expires_at"] <= now():
            raise TeamError("TRANSFER_EXPIRED", "Start the expired transfer again.", 410)
        return path, record, project

    @staticmethod
    def _part_size(record, index):
        if type(index) is not int or not 0 <= index < record["part_count"]:
            raise TeamError("INVALID_PART", "Invalid part index.", 422)
        return min(PART_BYTES, record["size_bytes"] - index * PART_BYTES)

    def begin(self, project_id, user, body):
        name = filename(body.name)
        if Path(name).suffix.lower() not in {".csv", ".parquet"}:
            raise TeamError("UNSUPPORTED_FORMAT", "Share large data as CSV or Parquet.", 422)
        transfer_id = hashlib.sha256(f"{user.uid}:{body.request_id}".encode()).hexdigest()[:32]
        path = _path(project_id, transfer_id)
        signature = {"name": name, "size_bytes": body.size_bytes, "sha256": body.sha256}

        def create(db):
            project = self.store._project(db, project_id, user, "editor")
            prior = db.get(path)
            if prior:
                if prior["uid"] != user.uid or any(prior[key] != value for key, value in signature.items()):
                    raise TeamError("TRANSFER_CONFLICT", "The transfer ID has different content.", 409)
                self._active(db, project_id, transfer_id, user, ready=True)
                return prior
            # Reservations live in the project transaction, not a truncated
            # scan of historical journals. Expired uploads reserve their budget
            # until cancelled and cleaned; expiry cannot create unbounded blobs.
            reservations = project.get("transfer_reservations", {})
            pending = list(reservations.values())
            if len(pending) >= 4:
                raise TeamError("TRANSFER_LIMIT", "Finish or cancel an existing transfer first.", 429)
            if any(item["name"].casefold() == name.casefold() for item in [*project["files"], *pending]):
                raise TeamError("FILE_EXISTS", "This filename is already in use.", 409)
            if len(project["files"]) + len(pending) >= 20 or (
                sum(item["size_bytes"] for item in project["files"]) +
                sum(row["size_bytes"] for row in pending) + body.size_bytes > MAX_PROJECT_DATA_BYTES
            ):
                raise TeamError("FILE_LIMIT", "The project transfer budget is 20 files and 8 GiB.", 429)
            record = {"id": transfer_id, "uid": user.uid, **signature, "state": "uploading",
                      "part_count": (body.size_bytes + PART_BYTES - 1) // PART_BYTES,
                      "created_at": now(), "expires_at": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
                      "file_id": uuid4().hex}
            db.put(path, record)
            reservations[transfer_id] = {"name": name, "size_bytes": body.size_bytes}
            project["transfer_reservations"] = reservations
            db.put(f"oe_projects/{project_id}", project)
            return record
        return self._status(self.store.db.atomic(create))

    @staticmethod
    def _status(record):
        return {key: record[key] for key in ("id", "state", "name", "size_bytes", "sha256", "part_count", "expires_at", "file_id")} | {
            "schema": TRANSFER_VERSION, "part_bytes": PART_BYTES}

    def status(self, project_id, transfer_id, user):
        path, _, _ = self.store.db.atomic(lambda db: self._active(db, project_id, transfer_id, user, ready=True))
        parts = self.store.db.scan(path + "/parts", limit=MAX_PARTS)
        _, record, _ = self.store.db.atomic(lambda db: self._active(db, project_id, transfer_id, user, ready=True))
        return self._status(record) | {"parts": sorted(
            [{"index": p["index"], "sha256": p["sha256"], "size_bytes": p["size_bytes"]}
             for p in parts if p.get("blob")],
            key=lambda part: part["index"])}

    def authorize_part(self, project_id, transfer_id, user, index):
        def check(db):
            _, record, _ = self._active(db, project_id, transfer_id, user)
            self._part_size(record, index)
        self.store.db.atomic(check)

    def put_part(self, project_id, transfer_id, user, index, payload, expected):
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise TeamError("DATA_INTEGRITY", "A SHA-256 part checksum is required.", 422)
        path, record, _ = self.store.db.atomic(lambda db: self._active(db, project_id, transfer_id, user))
        size = self._part_size(record, index)
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected:
            raise TeamError("DATA_INTEGRITY", "The uploaded part failed size/checksum verification.", 409)
        part_path = f"{path}/parts/{index:04d}"
        existing = self.store.db.get(part_path)
        recovered = None
        if existing and not existing.get("blob") and existing.get("sha256") == expected:
            recovered = self.storage.find(existing["key"], maximum=PART_BYTES)
            if recovered:
                data = self.storage.get(recovered, maximum=PART_BYTES)
                if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
                    raise TeamError("DATA_INTEGRITY", "The immutable part failed verification.", 409)

        def claim(db):
            self._active(db, project_id, transfer_id, user)
            prior = db.get(part_path)
            if prior and (prior["sha256"] != expected or prior["size_bytes"] != size):
                raise TeamError("TRANSFER_CONFLICT", "This part already has different content.", 409)
            if prior and not prior.get("blob") and prior["lease_until"] > now() and recovered is None:
                raise TeamError("PART_BUSY", "The part is still being acknowledged. Retry shortly.", 409)
            if not prior or not prior.get("blob"):
                prior = {"index": index, "sha256": expected, "size_bytes": size, "blob": None,
                         "key": f"projects/{project_id}/transfers/{transfer_id}/{index:04d}/{expected}",
                         "lease_until": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}
                db.put(part_path, prior)
            return prior
        prior = self.store.db.atomic(claim)
        if prior.get("blob"):
            return {"index": index, "sha256": expected, "size_bytes": size, "reused": True}
        # The durable claim has exactly one stable object key. Ambiguous object
        # or DB acknowledgements cannot allocate unlimited unreferenced blobs.
        try:
            reference = recovered or self._immutable_put(prior["key"], payload)
        except BaseException:
            def retryable(db):
                current = db.get(part_path)
                if current and not current.get("blob"):
                    current["lease_until"] = now()
                    db.put(part_path, current)
            self.store.db.atomic(retryable)
            raise
        part = {**prior, "blob": reference}

        def save(db):
            self._active(db, project_id, transfer_id, user)
            prior = db.get(part_path)
            if prior and prior.get("blob"):
                if prior["sha256"] != expected or prior["size_bytes"] != size:
                    raise TeamError("TRANSFER_CONFLICT", "This part already has different content.", 409)
                return False
            db.put(part_path, part)
            return True
        try:
            saved = self.store.db.atomic(save)
        except BaseException:
            # Unknown database acknowledgement is resolved on exact retry;
            # never delete a generation that may have become the winning one.
            journal = self.store.db.get(path)
            if journal and journal.get("state") == "cancelled":
                self.storage.delete(reference)
            raise
        return {"index": index, "sha256": expected, "size_bytes": size, "reused": not saved}

    def _immutable_put(self, key, payload, content_type="application/octet-stream"):
        try:
            return self.storage.put(key, payload, content_type)
        except Exception:
            reference = self.storage.find(key, maximum=max(PART_BYTES, MAX_MANIFEST_BYTES))
            if reference is None:
                raise
            # A pre-existing generation is reusable only with exact bytes.
            existing = self.storage.get(reference, maximum=max(PART_BYTES, MAX_MANIFEST_BYTES))
            if len(existing) != len(payload) or hashlib.sha256(existing).digest() != hashlib.sha256(payload).digest():
                raise TeamError("DATA_INTEGRITY", "The immutable object has different content.", 409) from None
            return reference

    def complete(self, project_id, transfer_id, user):
        def acquire(db):
            path, record, _ = self._active(db, project_id, transfer_id, user, ready=True)
            if record["state"] == "ready":
                return path, record, None
            if record.get("finalize_until", "") > now():
                raise TeamError("TRANSFER_BUSY", "Whole-file verification is running. Retry status shortly.", 409)
            lease = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
            record["finalize_until"] = lease
            db.put(path, record)
            return path, record, lease
        path, record, lease = self.store.db.atomic(acquire)
        if record["state"] == "ready":
            return self.manifest(project_id, record["file_id"], user)["file"]
        try:
            return self._complete(project_id, transfer_id, user, path, record)
        finally:
            def release(db):
                current = db.get(path)
                if current and current.get("finalize_until") == lease:
                    current.pop("finalize_until", None)
                    db.put(path, current)
            self.store.db.atomic(release)

    def _complete(self, project_id, transfer_id, user, path, record):
        parts = self.store.db.scan(path + "/parts", limit=MAX_PARTS)
        indexed = {part["index"]: part for part in parts}
        if len(indexed) != record["part_count"] or any(not part.get("blob") for part in parts):
            raise TeamError("TRANSFER_INCOMPLETE", "Upload every part before completing the file.", 409)
        digest = hashlib.sha256()
        ordered = []
        deadline = time.monotonic() + 10 * 60
        for index in range(record["part_count"]):
            if time.monotonic() > deadline:
                raise TeamError("TRANSFER_TIMEOUT", "Whole-file verification exceeded its ten-minute budget.", 503)
            # A revoked/cancelled transfer stops verification between parts.
            _, current, _ = self.store.db.atomic(lambda db: self._active(db, project_id, transfer_id, user, ready=True))
            if current["state"] == "ready":
                return self.manifest(project_id, current["file_id"], user)["file"]
            part = indexed.get(index)
            if part is None:
                raise TeamError("TRANSFER_INCOMPLETE", "A part is missing.", 409)
            data = self.storage.get(part["blob"], maximum=PART_BYTES)
            size = self._part_size(record, index)
            if len(data) != size or hashlib.sha256(data).hexdigest() != part["sha256"]:
                raise TeamError("DATA_INTEGRITY", "A stored part failed verification.", 409)
            digest.update(data)
            ordered.append({key: part[key] for key in ("index", "sha256", "size_bytes", "blob")})
        if digest.hexdigest() != record["sha256"]:
            raise TeamError("DATA_INTEGRITY", "The complete file failed SHA-256 verification.", 409)
        manifest = {"schema": TRANSFER_VERSION, "size_bytes": record["size_bytes"], "sha256": record["sha256"],
                    "part_bytes": PART_BYTES, "parts": ordered}
        data = json.dumps(manifest, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > MAX_MANIFEST_BYTES:
            raise TeamError("TRANSFER_LIMIT", "The file manifest exceeds its budget.", 413)
        reference = self._immutable_put(f"projects/{project_id}/files/{record['file_id']}/manifest.json", data, "application/json")
        metadata = {"id": record["file_id"], "name": record["name"], "python_path": record["name"],
                    "source": "upload", "size_bytes": record["size_bytes"], "data_hash": record["sha256"],
                    "row_count": None, "column_count": None, "columns": [], "preview": [],
                    "created_at": now(), "transfer": TRANSFER_VERSION, "blob": reference}

        def commit(db):
            _, current, project = self._active(db, project_id, transfer_id, user, ready=True)
            if current["state"] == "ready":
                return False
            if any(item["name"].casefold() == record["name"].casefold() for item in project["files"]):
                raise TeamError("FILE_EXISTS", "This filename is already in use.", 409)
            if len(project["files"]) >= 20 or sum(f["size_bytes"] for f in project["files"]) + record["size_bytes"] > MAX_PROJECT_DATA_BYTES:
                raise TeamError("FILE_LIMIT", "The project transfer budget was exceeded.", 429)
            project["files"].append(metadata)
            project.get("transfer_reservations", {}).pop(transfer_id, None)
            project["updated_at"] = now()
            db.put(f"oe_projects/{project_id}", project)
            current.update(state="ready", manifest=reference)
            db.put(path, current)
            self.store._audit(db, project_id, user, "file.shared", record["file_id"])
            return True
        # The stable manifest key is retained on an unknown acknowledgement;
        # exact retry resolves whether the project publication committed.
        self.store.db.atomic(commit)
        return self.manifest(project_id, record["file_id"], user)["file"]

    def manifest(self, project_id, file_id, user, *, private=False):
        project = self.store.project(project_id, user)
        metadata = next((f for f in project["files"] if f["id"] == file_id), None)
        if not metadata or metadata.get("transfer") != TRANSFER_VERSION:
            raise TeamError("NOT_FOUND", "Shared data manifest not found.", 404)
        data = self.storage.get(metadata["blob"], maximum=MAX_MANIFEST_BYTES)
        try:
            m = json.loads(data)
            if (not isinstance(m, dict) or set(m) != {"schema", "size_bytes", "sha256", "part_bytes", "parts"}
                    or m["schema"] != TRANSFER_VERSION or type(m["size_bytes"]) is not int
                    or type(m["part_bytes"]) is not int or not isinstance(m["parts"], list)
                    or m["size_bytes"] != metadata["size_bytes"] or m["sha256"] != metadata["data_hash"]):
                raise ValueError()
            count = (m["size_bytes"] + PART_BYTES - 1) // PART_BYTES
            if not 1 <= m["size_bytes"] <= MAX_DATA_BYTES or len(m["parts"]) != count or m["part_bytes"] != PART_BYTES:
                raise ValueError()
            for index, part in enumerate(m["parts"]):
                expected_size = min(PART_BYTES, m["size_bytes"] - index * PART_BYTES)
                if (not isinstance(part, dict) or set(part) != {"index", "sha256", "size_bytes", "blob"}
                        or type(part["index"]) is not int or type(part["size_bytes"]) is not int
                        or part["index"] != index or part["size_bytes"] != expected_size
                        or not _HASH.fullmatch(part["sha256"])):
                    raise ValueError()
                blob = part["blob"]
                if (not isinstance(blob, dict) or set(blob) != {"key", "generation", "size"}
                        or type(blob["generation"]) is not int or blob["generation"] <= 0
                        or type(blob["size"]) is not int or blob["size"] != expected_size
                        or not re.fullmatch(f"projects/{project_id}/transfers/[0-9a-f]{{32}}/{index:04d}/{part['sha256']}", blob["key"])):
                    raise ValueError()
        except (TypeError, KeyError, ValueError):
            raise TeamError("DATA_INTEGRITY", "The shared data manifest is invalid.", 409) from None
        self.store.project(project_id, user)
        return {**m, "file": _public_file(metadata), "parts": [
            part if private else {k: v for k, v in part.items() if k != "blob"} for part in m["parts"]]}

    def download_part(self, project_id, file_id, user, index):
        manifest = self.manifest(project_id, file_id, user, private=True)
        if type(index) is not int or not 0 <= index < len(manifest["parts"]):
            raise TeamError("INVALID_PART", "Invalid part index.", 422)
        part = manifest["parts"][index]
        data = self.storage.get(part["blob"], maximum=PART_BYTES)
        if len(data) != part["size_bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise TeamError("DATA_INTEGRITY", "The shared data part failed verification.", 409)
        self.store.project(project_id, user)
        return data, part

    def cancel(self, project_id, transfer_id, user):
        def cancel(db):
            project = self.store._project(db, project_id, user, "editor")
            path = _path(project_id, transfer_id)
            record = db.get(path)
            if not record:
                raise TeamError("NOT_FOUND", "Transfer not found.", 404)
            if record["uid"] != user.uid and project["owner_uid"] != user.uid:
                raise TeamError("ROLE_REQUIRED", "This transfer belongs to another member.", 403)
            if record["state"] == "ready":
                raise TeamError("TRANSFER_COMPLETE", "A complete project file cannot be cancelled.", 409)
            record["state"] = "cancelled"
            db.put(path, record)
            return path, record
        path, record = self.store.db.atomic(cancel)
        # State is durable before cleanup; an interrupted delete is retryable.
        pending = record.get("finalize_until", "") > now()
        for part in self.store.db.scan(path + "/parts", limit=MAX_PARTS):
            self.store.project(project_id, user, "editor")
            reference = part.get("blob") or self.storage.find(part["key"], maximum=PART_BYTES)
            if reference:
                self.store.project(project_id, user, "editor")
                self.storage.delete(reference)
            if not part.get("blob") and part["lease_until"] > now():
                pending = True
        if pending:
            raise TeamError("CLEANUP_PENDING", "Cancellation is saved; retry cleanup after the upload lease expires.", 409)
        manifest = self.storage.find(f"projects/{project_id}/files/{record['file_id']}/manifest.json",
                                     maximum=MAX_MANIFEST_BYTES)
        if manifest:
            self.store.project(project_id, user, "editor")
            self.storage.delete(manifest)
        def release(db):
            project = self.store._project(db, project_id, user, "editor")
            project.get("transfer_reservations", {}).pop(transfer_id, None)
            db.put(f"oe_projects/{project_id}", project)
        self.store.db.atomic(release)
        return {"cancelled": True}


def attach_transfer_routes(app, *, store, storage):
    transfers = DatasetTransfers(store, storage)
    app.state.dataset_transfers = transfers
    prefix = "/api/projects/{project_id}/workspace"

    @app.post(prefix + "/transfers", status_code=201)
    def begin(project_id: str, body: TransferBegin, request: Request):
        return transfers.begin(project_id, request.state.user, body)

    @app.get(prefix + "/transfers/{transfer_id}")
    def status(project_id: str, transfer_id: str, request: Request):
        return transfers.status(project_id, transfer_id, request.state.user)

    @app.put(prefix + "/transfers/{transfer_id}/parts/{index}")
    async def upload_part(project_id: str, transfer_id: str, index: int, request: Request):
        # Auth/membership before any part buffering, including unknown bodies.
        await run_in_threadpool(transfers.authorize_part, project_id, transfer_id, request.state.user, index)
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > PART_BYTES:
                raise TeamError("FILE_LIMIT", "The data part exceeds 4 MiB.", 413)
            data.extend(chunk)
        return await run_in_threadpool(transfers.put_part, project_id, transfer_id, request.state.user,
                                       index, bytes(data), request.headers.get("X-OpenEcon-SHA256"))

    @app.post(prefix + "/transfers/{transfer_id}/complete")
    def complete(project_id: str, transfer_id: str, request: Request):
        return transfers.complete(project_id, transfer_id, request.state.user)

    @app.delete(prefix + "/transfers/{transfer_id}")
    def cancel(project_id: str, transfer_id: str, request: Request):
        return transfers.cancel(project_id, transfer_id, request.state.user)

    @app.get(prefix + "/files/{file_id}/manifest")
    def manifest(project_id: str, file_id: str, request: Request):
        return transfers.manifest(project_id, file_id, request.state.user)

    @app.get(prefix + "/files/{file_id}/parts/{index}")
    def download(project_id: str, file_id: str, index: int, request: Request):
        data, part = transfers.download_part(project_id, file_id, request.state.user, index)
        return Response(data, media_type="application/octet-stream", headers={
            "X-OpenEcon-SHA256": part["sha256"], "Content-Length": str(len(data))})
