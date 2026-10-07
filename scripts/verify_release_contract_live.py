"""One release's synthetic archive proof; credentials exist only in memory.

Never uses browser sessions, existing QA state, or an existing application user.
Every write is scoped to two explicit fresh identities and one fresh QA project.
"""

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import tempfile
from uuid import uuid4

import firebase_admin
from firebase_admin import auth, credentials
from google.cloud import firestore, storage
from google.oauth2.credentials import Credentials
import httpx

import openecon
from openecon.console import ConsoleSession
from openecon.workspace import Workspace

PROJECT = "openecon-workbench"
ORIGIN = "https://openecon-291739190496.us-central1.run.app"


class ProbeFailure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ProbeFailure(message)


def make_records():
    require(
        Path(openecon.__file__).resolve().parent
        == Path(__file__).resolve().parents[1] / "src/openecon",
        "QA must run against the isolated release source.",
    )
    records = {}
    cases = {
        "ordinary": "covariance='HC3'",
        "weighted": "covariance='HC2', weights='weight', weight_type='aweight'",
        "hac": "covariance='hac', time='tick', lags=2, kernel='bartlett'",
    }
    with tempfile.TemporaryDirectory(prefix="openecon-release-console-") as directory:
        session = ConsoleSession(Workspace(Path(directory)))
        try:
            for name, arguments in cases.items():
                code = f"""import openecon as oe
frame = oe.example()
frame['weight'] = 1 + frame.index % 5
frame['tick'] = range(len(frame))
model = oe.ols(data=frame, y='wage', x=['education', 'experience'], {arguments})
model.title = 'OpenEcon 0.3.6 synthetic archive verification'
print('before-model')
display(model)
print('before-chart')
display(oe.plot.coefficients(model))
display(oe.Latex(model.to_latex()))
"""
                record = session.execute(code)
                require(record["status"] == "ok", "Synthetic local analysis failed.")
                require(
                    [item["type"] for item in record["outputs"]] == ["model", "plot", "latex"],
                    "Synthetic output ordering changed.",
                )
                records[name] = record
        finally:
            session.close()
    return records


def cleanup(db, bucket, admin_app, identities, project_ids, invitations, marker):
    """Preflight owned resources and refuse deletion if fixture ownership drifted."""
    uid_set = {item["uid"] for item in identities.values()}
    owner = identities["owner"]["uid"]
    # Recover a successfully created fixture if its HTTP response was lost.
    owned = db.collection("oe_projects").where(
        filter=firestore.FieldFilter("owner_uid", "==", owner)
    )
    for snapshot in owned.stream():
        value = snapshot.to_dict()
        require(
            value.get("name") == marker and set(value.get("members", {})) <= uid_set,
            "QA identity acquired an unexpected project; cleanup stopped.",
        )
        if snapshot.id not in project_ids:
            project_ids.append(snapshot.id)
    project_set = set(project_ids)
    for project_id in project_ids:
        related = db.collection("oe_invitations").where(
            filter=firestore.FieldFilter("project_id", "==", project_id)
        )
        for snapshot in related.stream():
            if snapshot.id not in invitations:
                invitations.append(snapshot.id)
    for project_id in project_ids:
        ref = db.document("oe_projects/" + project_id)
        value = ref.get().to_dict()
        if value is not None:
            require(
                value.get("name") == marker
                and value.get("owner_uid") == owner
                and set(value.get("members", {})) <= uid_set
                and not value.get("active_run"),
                "QA resource ownership changed; cleanup stopped.",
            )
        for run in ref.collection("runs").stream():
            require(
                run.to_dict().get("state") in {"finished", "failed", "cancelled"},
                "QA execution remains active; cleanup stopped.",
            )
    for user_id in uid_set:
        profile = db.document("oe_users/" + user_id).get().to_dict() or {}
        require(
            set(profile.get("project_ids", [])) <= project_set,
            "QA identity acquired external membership; cleanup stopped.",
        )
        for field in ("owner_uid", "members." + user_id + ".uid"):
            query = db.collection("oe_projects").where(
                filter=firestore.FieldFilter(field, "==", user_id)
            )
            require(
                all(snapshot.id in project_set for snapshot in query.stream()),
                "QA identity acquired an external project; cleanup stopped.",
            )
    for invitation_id in invitations:
        value = db.document("oe_invitations/" + invitation_id).get().to_dict()
        require(
            value is None or value.get("project_id") in project_set,
            "QA invitation ownership changed; cleanup stopped.",
        )
    for project_id in project_ids:
        db.recursive_delete(db.document("oe_projects/" + project_id))
        for prefix in (f"projects/{project_id}/", f"staging/{project_id}/"):
            for blob in bucket.list_blobs(prefix=prefix):
                blob.delete(if_generation_match=int(blob.generation))
    for invitation_id in invitations:
        db.document("oe_invitations/" + invitation_id).delete()
    for user_id in uid_set:
        try:
            auth.delete_user(user_id, app=admin_app)
        except auth.UserNotFoundError:
            pass
        db.document("oe_users/" + user_id).delete()
    remaining = {
        "project_documents": 0,
        "project_subcollection_documents": 0,
        "project_blobs": 0,
        "staging_blobs": 0,
        "invitations": 0,
        "profiles": 0,
        "auth_identities": 0,
    }
    for project_id in project_ids:
        reference = db.document("oe_projects/" + project_id)
        remaining["project_documents"] += int(reference.get().exists)
        for collection in reference.collections():
            remaining["project_subcollection_documents"] += sum(1 for _ in collection.stream())
        remaining["project_blobs"] += sum(
            1 for _ in bucket.list_blobs(prefix=f"projects/{project_id}/")
        )
        remaining["staging_blobs"] += sum(
            1 for _ in bucket.list_blobs(prefix=f"staging/{project_id}/")
        )
    for invitation_id in invitations:
        remaining["invitations"] += int(db.document("oe_invitations/" + invitation_id).get().exists)
    for user_id in uid_set:
        remaining["profiles"] += int(db.document("oe_users/" + user_id).get().exists)
        try:
            auth.get_user(user_id, app=admin_app)
        except auth.UserNotFoundError:
            continue
        remaining["auth_identities"] += 1
    require(not any(remaining.values()), "QA cleanup readback found remaining resources.")
    return {
        "projects_removed": len(project_ids),
        "identities_removed": len(uid_set),
        "invitations_removed": len(invitations),
        "remaining": remaining,
        "readback_confirmed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-live", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "source_commit": "a694e18",
        "local_integration_tests": 306,
        "baseline_archive_http": 422,
        "baseline_rejected_fields": [
            "spec.weights",
            "spec.weight_type",
            "spec.time",
            "spec.panel",
            "spec.options",
            "title",
            "tests",
        ],
        "live_checks": [],
        "cleanup": None,
        "success": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    phase = "synthetic_analysis"
    marker = "Disposable release 0.3.6 " + uuid4().hex[:12]
    records = make_records()
    phase = "admin_authorization"
    token_process = subprocess.run(
        ["gcloud", "auth", "print-access-token", "--project=" + PROJECT],
        capture_output=True,
        text=True,
    )
    require(token_process.returncode == 0, "Cloud authorization unavailable.")
    google_credential = Credentials(token_process.stdout.strip(), quota_project_id=PROJECT)
    del token_process

    class AdminCredential(credentials.Base):
        def get_credential(self):
            return google_credential

    admin_app = firebase_admin.initialize_app(
        AdminCredential(), {"projectId": PROJECT}, name="release-" + uuid4().hex
    )
    db = firestore.Client(project=PROJECT, credentials=google_credential)
    bucket = storage.Client(project=PROJECT, credentials=google_credential).bucket(
        PROJECT + "-projects"
    )
    identities = {role: {"uid": "oeqa036_" + uuid4().hex} for role in ("owner", "viewer")}
    projects, invitations = [], []
    failed = False
    try:
        with httpx.Client(timeout=60, follow_redirects=False) as client:
            config = client.get(ORIGIN + "/api/auth/config")
            require(config.status_code == 200, "Public auth configuration unavailable.")
            firebase = config.json()["firebase"]
            require(firebase["projectId"] == PROJECT, "Unexpected authentication project.")
            require(
                client.get(ORIGIN + "/").status_code == 200,
                "Anonymous application availability check failed.",
            )
            require(
                client.get(ORIGIN + "/api/projects").status_code == 401,
                "Anonymous project access was not denied.",
            )
            report["live_checks"].append("anonymous_application_auth_config_and_project_denial")
            phase = "fresh_identities"
            for role, identity in identities.items():
                email = "openecon-release-036-" + uuid4().hex + "@example.com"
                password = "Qa9-" + secrets.token_urlsafe(32)
                auth.create_user(
                    uid=identity["uid"],
                    email=email,
                    password=password,
                    email_verified=True,
                    display_name="Disposable release " + role,
                    app=admin_app,
                )
                response = client.post(
                    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                    params={"key": firebase["apiKey"]},
                    json={"email": email, "password": password, "returnSecureToken": True},
                )
                require(response.status_code == 200, "Synthetic sign-in failed.")
                identity.update(email=email, token=response.json()["idToken"])
                del password, response
            owner = identities["owner"]
            db.document("oe_users/" + owner["uid"]).set(
                {
                    "uid": owner["uid"],
                    "email": owner["email"],
                    "name": "Disposable release owner",
                    "enabled": True,
                    "project_ids": [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            def call(method, path, *, role="owner", expected=200, **kwargs):
                response = client.request(
                    method,
                    ORIGIN + "/api" + path,
                    headers={
                        "Authorization": "Bearer " + identities[role]["token"],
                        "Origin": ORIGIN,
                    },
                    **kwargs,
                )
                require(
                    response.status_code == expected,
                    "Synthetic API response did not match expected status (HTTP "
                    + str(response.status_code)
                    + ").",
                )
                return response.json()

            phase = "fresh_project_membership"
            for role in identities:
                call("GET", "/me", role=role)
            project = call("POST", "/projects", expected=201, json={"name": marker})
            projects.append(project["id"])
            project_id = project["id"]
            invitation = call(
                "POST",
                "/projects/" + project_id + "/invitations",
                expected=201,
                json={"email": identities["viewer"]["email"], "role": "viewer"},
            )
            invitations.append(invitation["id"])
            call("POST", "/invitations/" + invitation["id"] + "/accept", role="viewer")
            expected_outputs = {}
            path = "/projects/" + project_id + "/workspace/desktop/results"
            for name, source in records.items():
                phase = "archive_" + name
                submitted = deepcopy(source)
                submitted["actor_uid"] = owner["uid"]
                published = call(
                    "POST", path, expected=201, json={"record": submitted, "input_files": []}
                )
                require(
                    call("POST", path, expected=201, json={"record": submitted, "input_files": []})
                    == published,
                    "Archive idempotency failed.",
                )
                outputs = deepcopy(source["outputs"])
                outputs[0]["data"]["display_omitted"] = ["covariance_matrix", "sample_positions"]
                expected_outputs[published["id"]] = (outputs, source["events"], source["stdout"])
                report["live_checks"].append(name + "_archive_201_and_idempotent")
            phase = "viewer_readback"
            history = call("GET", "/projects/" + project_id + "/workspace/console", role="viewer")[
                "history"
            ]
            require(len(history) == len(expected_outputs), "Unexpected archived result count.")
            for record in history:
                outputs, events, stdout = expected_outputs[record["id"]]
                require(
                    record["outputs"] == outputs
                    and record["events"] == events
                    and record["stdout"] == stdout,
                    "Shared model, D3, LaTeX or event readback changed.",
                )
                require(
                    record["execution_origin"] == "desktop",
                    "Archive unexpectedly executed remotely.",
                )
                spec = record["outputs"][0]["data"]["spec"]
                require(
                    {"weights", "weight_type", "time", "panel", "options"} <= spec.keys(),
                    "New model metadata was lost.",
                )
                require(record["outputs"][0]["data"]["tests"], "Model inference tests were lost.")
            report["live_checks"].append(
                "viewer_readback_preserves_title_tests_weights_hac_d3_latex_events"
            )
            phase = "invalid_result_rejection"
            for invalid in ("unknown_field", "invalid_weights"):
                submitted = deepcopy(records["ordinary"])
                submitted.update(id="invalid-" + uuid4().hex, actor_uid=owner["uid"])
                data = submitted["outputs"][0]["data"]
                if invalid == "unknown_field":
                    data["unknown_release_probe"] = True
                else:
                    data["spec"]["weight_type"] = "aweight"
                rejected = call(
                    "POST", path, expected=422, json={"record": submitted, "input_files": []}
                )
                require(
                    rejected["detail"]["code"] == "INVALID_RESULT",
                    "Malformed model rejection changed.",
                )
                report["live_checks"].append(invalid + "_rejected_422")
            forbidden = deepcopy(records["ordinary"])
            forbidden.update(id="viewer-" + uuid4().hex, actor_uid=identities["viewer"]["uid"])
            call(
                "POST",
                path,
                role="viewer",
                expected=403,
                json={"record": forbidden, "input_files": []},
            )
            require(
                len(
                    call("GET", "/projects/" + project_id + "/workspace/console", role="viewer")[
                        "history"
                    ]
                )
                == 3,
                "Rejected publication unexpectedly persisted.",
            )
            report["live_checks"].append(
                "viewer_publication_denied_403_and_no_invalid_records_saved"
            )
            report["success"] = True
    except Exception:
        failed = True
        report["failed_phase"] = phase
    finally:
        try:
            report["cleanup"] = cleanup(
                db, bucket, admin_app, identities, projects, invitations, marker
            )
        except Exception:
            failed = True
            report["success"] = False
            report["cleanup"] = {"readback_confirmed": False}
            # Only resource references, never email/password/token, for exact recovery.
            recovery = args.report.with_suffix(".recovery.json")
            recovery.touch(mode=0o600, exist_ok=True)
            recovery.chmod(0o600)
            recovery.write_text(
                json.dumps(
                    {
                        "project": PROJECT,
                        "marker": marker,
                        "projects": projects,
                        "invitations": invitations,
                        "uids": [v["uid"] for v in identities.values()],
                    },
                    indent=2,
                )
            )
        firebase_admin.delete_app(admin_app)
        report["checked_at"] = datetime.now(timezone.utc).isoformat()
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(
            "Release verification did not complete; see sanitized phase and cleanup report."
        )


if __name__ == "__main__":
    try:
        main()
    except ProbeFailure:
        raise SystemExit(
            "Release verification stopped before completion; no provider response or credentials printed."
        ) from None
