"""Synthetic shared uv manifest contract; no package installation or execution.

Live mode uses two fresh Firebase identities and one disposable owned project.
Credentials remain in memory. Every owned resource is removed and read back.
The tested Python modules must come from the clean 0.3.7 release source.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4

PROJECT = "openecon-workbench"
ORIGIN = "https://openecon-291739190496.us-central1.run.app"
ROOT = Path(__file__).resolve().parents[1]
RELEASE_SOURCE = ROOT / "desktop/build/releases/0.3.7/source"


class ProbeFailure(RuntimeError):
    """Only fixed messages, never credentials or provider response bodies."""


def require(condition, message):
    if not condition:
        raise ProbeFailure(message)


def modern_manifest():
    # Nonexistent QA distributions are intentional: sharing must never resolve
    # packages, install wheels, or evaluate the server's platform markers.
    return {"schema": 2, "python": "3.13", "core": {"openecon": "0.3.7", "torch": "2.9.0"},
            "requirements": [{"name": "oe-qa-demo", "version": "1.2.3"}],
            "locked": [{"name": "oe-qa-demo", "version": "1.2.3"},
                       {"name": "oe-qa-helper", "version": "2.0"}],
            "installer": "uv", "specifications": [
                'oe-qa-demo[fast]<2,>=1; platform_system == "Darwin"',
                'oe-qa-only-platform>=4; sys_platform == "win32"']}


def legacy_manifest():
    value = modern_manifest()
    return {key: (1 if key == "schema" else value) for key, value in value.items()
            if key not in {"installer", "specifications"}}


def invalid_manifests():
    changes = {
        "direct_url": {"specifications": ["oe-qa-demo @ https://example.invalid/a.whl"]},
        "index_option": {"specifications": ["--index-url=https://example.invalid"]},
        "pin_mismatch": {"specifications": ["oe-qa-demo>=2"]},
        "unexpected_field": {"command": "uv pip install oe-qa-demo"},
        "invalid_installer": {"installer": "shell"},
        "noninteger_schema": {"schema": 2.0},
        "locked_url_field": {"locked": [
            {"name": "oe-qa-demo", "version": "1.2.3", "url": "https://example.invalid/a.whl"}]},
    }
    return {name: modern_manifest() | deepcopy(change) for name, change in changes.items()}


def verify_contract(call, project_id):
    """Run the same HTTP contract against a live client or TestClient."""
    prefix = f"/projects/{project_id}/workspace"
    path = prefix + "/environment"
    checks = []
    require(call("GET", path) == {"manifest": None, "version": 0}, "Fresh manifest was not empty.")
    expected = {"manifest": modern_manifest(), "version": 1}
    require(call("PUT", path, json={"manifest": modern_manifest(), "version": 0}) == expected,
            "Schema 2 write changed installer or specifications.")

    def readback(value):
        for role in ("owner", "viewer"):
            require(call("GET", path, role=role) == value, "Manifest readback changed.")
            require(call("GET", prefix + "/bootstrap", role=role)["environment"] == value,
                    "Bootstrap manifest readback changed.")

    readback(expected)
    checks.append("schema2_uv_ranges_extras_platform_markers_exact_owner_viewer_bootstrap_readback")
    call("PUT", path, role="viewer", expected=403,
         json={"manifest": modern_manifest(), "version": 1})
    readback(expected)
    checks.append("viewer_reads_200_writes_403_without_mutation")
    conflict = call("PUT", path, expected=409,
                    json={"manifest": modern_manifest() | {"installer": "pip"}, "version": 0})
    require(conflict.get("detail", {}).get("code") == "VERSION_CONFLICT", "Conflict code changed.")
    readback(expected)
    checks.append("optimistic_conflict_409_preserves_manifest")
    for name, invalid in invalid_manifests().items():
        rejected = call("PUT", path, expected=422, json={"manifest": invalid, "version": 1})
        require(rejected.get("detail", {}).get("code") == "INVALID_ENVIRONMENT",
                "Invalid manifest rejection code changed.")
        readback(expected)
        checks.append(name + "_422_without_mutation")
    call("PUT", path, expected=422,
         json={"manifest": modern_manifest(), "version": 1, "install": True})
    readback(expected)
    checks.append("unexpected_request_field_422_without_mutation")
    expected = {"manifest": legacy_manifest(), "version": 2}
    require(call("PUT", path, json={"manifest": legacy_manifest(), "version": 1}) == expected,
            "Legacy schema 1 write changed.")
    readback(expected)
    checks.append("legacy_schema1_exact_roundtrip")
    expected = {"manifest": modern_manifest(), "version": 3}
    require(call("PUT", path, json={"manifest": modern_manifest(), "version": 2}) == expected,
            "Schema 1 to 2 transition changed.")
    readback(expected)
    checks.append("legacy_to_schema2_exact_roundtrip")
    return checks


def configure_release_source():
    expected = RELEASE_SOURCE / "src/openecon"
    require((expected / "team_store.py").is_file(), "Clean 0.3.7 release source is unavailable.")
    sys.path.insert(0, str(RELEASE_SOURCE / "src"))
    import openecon
    require(Path(openecon.__file__).resolve().parent == expected.resolve(),
            "QA must import the clean 0.3.7 release source.")
    return openecon


def cleanup_helper():
    spec = importlib.util.spec_from_file_location(
        "uv_manifest_release_cleanup", ROOT / "scripts/verify_release_contract_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cleanup


def run_live(report_path):
    configure_release_source()
    from openecon.team_store import validate_environment_manifest
    require(validate_environment_manifest(modern_manifest()) == modern_manifest(),
            "Canonical schema 2 fixture does not match this release.")
    import firebase_admin
    from firebase_admin import auth, credentials
    from google.cloud import firestore, storage
    from google.oauth2.credentials import Credentials
    import httpx

    cleanup = cleanup_helper()
    report = {"release_source": "desktop/build/releases/0.3.7/source", "project": PROJECT,
              "origin": ORIGIN, "live_checks": [], "cleanup": None, "success": False}
    marker = "Disposable uv schema2 " + uuid4().hex
    identities = {role: {"uid": "oeqa_uv037_" + uuid4().hex,
                         "email": "openecon-uv-037-" + uuid4().hex + "@example.com",
                         "name": marker + " " + role} for role in ("owner", "viewer")}
    project_ids, invitations = [], []
    admin_app = db = bucket = None
    phase = "admin_authorization"
    try:
        process = subprocess.run(["gcloud", "auth", "print-access-token", "--project=" + PROJECT],
                                 capture_output=True, text=True, timeout=45)
        require(process.returncode == 0 and process.stdout.strip(), "Cloud authorization unavailable.")
        google_credential = Credentials(process.stdout.strip(), quota_project_id=PROJECT)
        del process

        class AdminCredential(credentials.Base):
            def get_credential(self):
                return google_credential

        admin_app = firebase_admin.initialize_app(
            AdminCredential(), {"projectId": PROJECT}, name="uv-manifest-" + uuid4().hex)
        db = firestore.Client(project=PROJECT, credentials=google_credential)
        bucket = storage.Client(project=PROJECT, credentials=google_credential).bucket(PROJECT + "-projects")
        with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
            phase = "public_auth_configuration"
            config = client.get(ORIGIN + "/api/auth/config")
            require(config.status_code == 200, "Public auth configuration unavailable.")
            firebase = config.json()["firebase"]
            require(firebase["projectId"] == PROJECT, "Unexpected authentication project.")
            require(client.get(ORIGIN + "/api/projects").status_code == 401, "Anonymous project access allowed.")
            phase = "fresh_identities"
            for identity in identities.values():
                password = "Qa9-" + secrets.token_urlsafe(32)
                auth.create_user(uid=identity["uid"], email=identity["email"], password=password,
                                 email_verified=True, display_name=identity["name"], app=admin_app)
                response = client.post("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                                       params={"key": firebase["apiKey"]},
                                       json={"email": identity["email"], "password": password, "returnSecureToken": True})
                require(response.status_code == 200, "Synthetic sign-in failed.")
                identity["token"] = response.json()["idToken"]
                del password, response
            owner = identities["owner"]
            db.document("oe_users/" + owner["uid"]).set({
                "uid": owner["uid"], "email": owner["email"], "name": owner["name"],
                "enabled": True, "project_ids": [], "created_at": datetime.now(timezone.utc).isoformat()})

            def call(method, path, *, role="owner", expected=200, **kwargs):
                response = client.request(method, ORIGIN + "/api" + path,
                                          headers={"Authorization": "Bearer " + identities[role]["token"],
                                                   "Origin": ORIGIN}, **kwargs)
                require(response.status_code == expected,
                        "Synthetic API status mismatch (HTTP " + str(response.status_code) + ").")
                require(response.headers.get("cache-control") == "no-store", "API caching policy changed.")
                return response.json()

            phase = "fresh_project_membership"
            for role in identities:
                call("GET", "/me", role=role)
            project = call("POST", "/projects", expected=201, json={"name": marker})
            project_ids.append(project["id"])
            project_id = project["id"]
            invitation = call("POST", f"/projects/{project_id}/invitations", expected=201,
                              json={"email": identities["viewer"]["email"], "role": "viewer"})
            invitations.append(invitation["id"])
            call("POST", "/invitations/" + invitation["id"] + "/accept", role="viewer")
            phase = "shared_manifest_contract"
            report["live_checks"] = verify_contract(call, project_id)
            phase = "inert_manifest_readback"
            reference = db.document("oe_projects/" + project_id)
            project = reference.get().to_dict()
            require(not project.get("active_run") and not project.get("files"), "Manifest created execution state.")
            require(not list(reference.collection("runs").stream()), "Manifest created a compute run.")
            for prefix in (f"projects/{project_id}/", f"staging/{project_id}/"):
                require(not list(bucket.list_blobs(prefix=prefix)), "Manifest created data artifacts.")
            stored = db.document(f"oe_projects/{project_id}/workspace/environment").get().to_dict()
            require(stored["manifest"] == modern_manifest() and stored["version"] == 3,
                    "Persisted manifest changed.")
            report["live_checks"].append("firestore_exact_inert_metadata_no_compute_runs_or_blobs")
            report["success"] = True
    except Exception:
        report["success"] = False
        report["failed_phase"] = phase
    finally:
        if admin_app is not None and db is not None and bucket is not None:
            try:
                # A lost create response may still have created the known UID.
                # Refuse deletion unless its exact fresh email/name also match.
                for identity in identities.values():
                    try:
                        actual = auth.get_user(identity["uid"], app=admin_app)
                    except auth.UserNotFoundError:
                        continue
                    require(actual.email == identity["email"] and actual.display_name == identity["name"],
                            "QA identity ownership changed; cleanup stopped.")
                report["cleanup"] = cleanup(db, bucket, admin_app, identities, project_ids, invitations, marker)
            except Exception:
                report["success"] = False
                report["cleanup"] = {"readback_confirmed": False}
                recovery = report_path.with_suffix(".recovery.json")
                recovery.parent.mkdir(parents=True, exist_ok=True)
                recovery.touch(mode=0o600, exist_ok=True)
                recovery.chmod(0o600)
                recovery.write_text(json.dumps({"project": PROJECT, "marker": marker,
                    "projects": project_ids, "invitations": invitations,
                    "uids": [identity["uid"] for identity in identities.values()]}, indent=2) + "\n")
        if admin_app is not None:
            firebase_admin.delete_app(admin_app)
        report["checked_at"] = datetime.now(timezone.utc).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run-live", action="store_true")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "artifacts/verification/uv-manifest-live.json")
    args = parser.parse_args()
    if args.check:
        configure_release_source()
        from openecon.team_store import validate_environment_manifest
        require(validate_environment_manifest(modern_manifest()) == modern_manifest(), "Schema 2 fixture changed.")
        require(validate_environment_manifest(legacy_manifest()) == legacy_manifest(), "Legacy fixture changed.")
        print(json.dumps({"release_source_verified": True, "fixtures_validated": True, "cloud_calls": False}))
        return 0
    report = run_live(args.report)
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit("Manifest verification unavailable; no provider response or credentials printed.") from None
