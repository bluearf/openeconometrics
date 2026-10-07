"""Cloud login grants and archived results for locally executing clients.

Grants store a UID and SHA-256 challenge, never refresh credentials. Results
are untrusted data; this module never executes Python code.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import re
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from openecon.team_store import TeamError


class LoginBegin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    challenge: str = Field(pattern=r"^[0-9a-f]{64}$")


class LoginExchange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    verifier: str = Field(pattern=r"^[0-9a-f]{64}$")


class DesktopResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record: dict
    input_files: list[dict] = Field(max_length=20)


def public_desktop_login(path: str, method: str) -> bool:
    return method == "POST" and (path == "/api/desktop/login" or bool(
        re.fullmatch(r"/api/desktop/login/[0-9a-f]{32}/exchange", path)))


def firebase_custom_token(storage, uid: str) -> str:
    from google.auth import jwt
    timestamp = int(datetime.now(timezone.utc).timestamp())
    return jwt.encode(storage.signer.signer, {
        "iss": storage.signer.service_account_email,
        "sub": storage.signer.service_account_email,
        "aud": "https://identitytoolkit.googleapis.com/google.identity.identitytoolkit.v1.IdentityToolkit",
        "iat": timestamp, "exp": timestamp + 300, "uid": uid,
        "claims": {"openecon_desktop": True},
    }).decode("ascii")


def attach_desktop_cloud_routes(app, *, store, storage, token_issuer=None):
    issuer = token_issuer or (lambda uid: firebase_custom_token(storage, uid))

    def login_path(request_id):
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise TeamError("LOGIN_NOT_FOUND", "Sign-in request not found.", 404)
        return f"oe_desktop_logins/{request_id}"

    def active(db, request_id):
        path = login_path(request_id)
        grant = db.get(path)
        if not grant or grant["expires_at"] <= datetime.now(timezone.utc):
            raise TeamError("LOGIN_EXPIRED", "Sign-in request expired. Start again from the app.", 410)
        return path, grant

    @app.post("/api/desktop/login", status_code=201)
    def begin(body: LoginBegin):
        request_id = uuid4().hex
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=5)
        def create(db):
            quota_path = f"oe_limits/desktop-login-{now:%Y%m%d}"
            quota = db.get(quota_path) or {"count": 0}
            if quota["count"] >= 512:
                raise TeamError("LOGIN_LIMIT", "The sign-in limit has been reached. Try again later.", 429)
            quota["count"] += 1
            db.put(quota_path, quota)
            db.put(login_path(request_id), {"id": request_id, "challenge": body.challenge,
                                           "expires_at": expires, "uid": None})
        store.db.atomic(create)
        return {"request_id": request_id, "expires_at": expires.isoformat(), "code": request_id[:8].upper()}

    @app.post("/api/desktop/login/{request_id}/authorize")
    def authorize(request_id: str, request: Request):
        def approve(db):
            path, grant = active(db, request_id)
            if grant["uid"] is not None:
                raise TeamError("LOGIN_ALREADY_AUTHORIZED", "This sign-in request has already been approved.", 409)
            grant["uid"] = request.state.user.uid
            db.put(path, grant)
        store.db.atomic(approve)
        return {"authorized": True}

    @app.post("/api/desktop/login/{request_id}/exchange")
    def exchange(request_id: str, body: LoginExchange):
        digest = hashlib.sha256(body.verifier.encode("ascii")).hexdigest()
        def consume(db):
            path, grant = active(db, request_id)
            if not hmac.compare_digest(grant["challenge"], digest):
                raise TeamError("INVALID_LOGIN_PROOF", "The sign-in request could not be verified.", 403)
            if grant["uid"] is None:
                return None
            db.delete(path)
            return grant["uid"]
        uid = store.db.atomic(consume)
        if uid is None:
            return JSONResponse({"pending": True}, status_code=202)
        return {"custom_token": issuer(uid)}

    @app.post("/api/projects/{project_id}/workspace/desktop/results", status_code=201)
    def publish_result(project_id: str, body: DesktopResult, request: Request):
        from openecon.team_server import json_bytes, validate_worker_result
        user = request.state.user
        project = store.project(project_id, user, "editor")
        record = body.record
        if record.get("actor_uid") != user.uid:
            raise TeamError("RESULT_ACTOR", "This result was produced in another session.", 403)
        client_id, code = record.get("id"), record.get("code")
        if (not isinstance(client_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", client_id)
                or not isinstance(code, str) or not 1 <= len(code) <= 64000):
            raise TeamError("INVALID_RESULT", "The local analysis record is invalid.", 422)
        run_id = hashlib.sha256(f"{user.uid}:{client_id}".encode()).hexdigest()[:32]
        try:
            request_hash = hashlib.sha256(json.dumps(body.model_dump(), sort_keys=True,
                ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        except (ValueError, UnicodeError):
            raise TeamError("INVALID_RESULT", "The local analysis record is invalid.", 422) from None
        run_path = f"oe_projects/{project_id}/runs/{run_id}"

        def identical(prior):
            if (prior.get("desktop_client_id") != client_id or prior.get("uid") != user.uid
                    or prior.get("desktop_request_hash") != request_hash):
                raise TeamError("RESULT_CONFLICT", "The result ID is already in use.", 409)

        def existing(db):
            current = store._project(db, project_id, user, "editor")
            prior = db.get(run_path)
            if prior:
                identical(prior)
                return None
            return current

        # A lost acknowledgement must remain repeatable after the input files
        # change. Membership and the exact original request are still checked.
        project = store.db.atomic(existing)
        if project is None:
            return {"id": run_id, "shared": True}

        def input_versions(current):
            inputs = []
            for supplied in body.input_files:
                file = next((f for f in current["files"] if f["id"] == supplied.get("id")), None)
                if file is None or file["data_hash"] != supplied.get("data_hash"):
                    raise TeamError("DATA_INTEGRITY", "The data version for this result could not be verified.", 409)
                inputs.append({"id": file["id"], "name": file["name"], "data_hash": file["data_hash"]})
            return inputs

        inputs = input_versions(project)
        created = datetime.now(timezone.utc).isoformat()
        run = {"id": run_id, "code": code, "created_at": created, "generation": 0, "email": user.email}
        trusted, _ = validate_worker_result({"execution_id": run_id, "record": record}, run)
        trusted.update(execution_origin="desktop", input_files=inputs,
                       verification="Client-produced output; archived without rerunning or independently verifying calculations.")
        payload = json_bytes(trusted)
        if len(payload) > 3 * 1024**2:
            raise TeamError("RESULT_LIMIT", "The result exceeds the size limit.", 413)
        # Each attempt owns its immutable generation. A concurrent loser can
        # never overwrite, collide with or delete the winning archived payload.
        publication = uuid4().hex
        reference = storage.put(f"projects/{project_id}/desktop-results/{run_id}/{publication}/record.json",
                                payload, "application/json")
        def save(db):
            current = store._project(db, project_id, user, "editor")
            prior = db.get(run_path)
            if prior:
                identical(prior)
                return False
            input_versions(current)
            db.put(run_path, {**run, "uid": user.uid, "state": "finished", "result": reference,
                              "desktop_client_id": client_id, "execution_origin": "desktop",
                              "desktop_request_hash": request_hash,
                              "record_summary": {"status": record["status"]}, "artifacts": []})
            store._audit(db, project_id, user, "desktop.result_shared", run_id)
            current["updated_at"] = created
            db.put(f"oe_projects/{project_id}", current)
            return True
        try:
            saved = store.db.atomic(save)
        except TeamError:
            storage.delete(reference)
            raise
        except BaseException:
            # A database transport failure can occur after committing. Retain
            # this generation unless another winner proves it is unreferenced;
            # an exact retry resolves the acknowledgement without recomputation.
            try:
                prior = store.db.get(run_path)
            except BaseException:
                prior = None
            if prior and prior.get("result") != reference:
                storage.delete(reference)
            raise
        if not saved:
            storage.delete(reference)
        return {"id": run_id, "shared": True}
