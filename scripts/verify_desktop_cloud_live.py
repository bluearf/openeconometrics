"""Disposable integration QA: desktop login and local results, never cloud compute.

Use verify_team_live.py prepare/cleanup to own the test identities. Secrets stay
in the existing ignored mode-0600 QA file and memory. Only exact test IDs enter
the cleanup list. No emails, real user projects or cloud Python runs are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import tempfile

import httpx

PROJECT = "openecon-workbench"
ORIGINS = {
    "https://openecon-teams-preview-291739190496.us-central1.run.app",
    "https://openecon-291739190496.us-central1.run.app",
}
STATE = Path("artifacts/team-setup/qa-state.json")
REPORT = Path("artifacts/verification/desktop-cloud-live.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, choices=sorted(ORIGINS))
    args = parser.parse_args()
    if STATE.is_symlink() or STATE.stat().st_mode & 0o077:
        raise SystemExit("The owned QA file must be a private regular file.")
    state = json.loads(STATE.read_text())
    checks = []
    report = {"origin": args.url, "cloud_project": PROJECT,
              "started_at": datetime.now(timezone.utc).isoformat(), "passed": False,
              "checks": checks, "cloud_computations_started": 0}
    def save():
        STATE.write_text(json.dumps(state, indent=2))
    def checked(name):
        checks.append(name)
        print("PASS", name, flush=True)
    with httpx.Client(timeout=90, follow_redirects=False, trust_env=False) as client:
        def call(method, path, *, token=None, expected=200, **kwargs):
            if "/console/execute" in path:
                raise RuntimeError("This QA never invokes cloud computation.")
            headers = {"Origin": args.url}
            if token:
                headers["Authorization"] = "Bearer " + token
            response = client.request(method, args.url + "/api" + path, headers=headers, **kwargs)
            if response.status_code != expected:
                raise RuntimeError(f"QA API returned HTTP {response.status_code}; provider details withheld.")
            return response
        try:
            config = call("GET", "/auth/config").json()
            assert config["firebase"]["projectId"] == PROJECT
            assert config["desktop_login_available"] is True
            assert config.get("cloud_execution_available") is False
            # Build-bound, read-only identity of the deployed sync service.
            report["source_commit"] = config.get("source_commit")
            config = config["firebase"]
            tokens = {}
            for role in ("owner", "viewer", "outsider"):
                user = state["users"][role]
                assert user["email"].startswith("openecon-qa-") and user["email"].endswith("@example.com")
                response = client.post("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                    params={"key": config["apiKey"]}, json={"email": user["email"],
                    "password": user["password"], "returnSecureToken": True})
                if response.status_code != 200:
                    raise RuntimeError("Disposable QA sign-in failed; credentials withheld.")
                tokens[role] = response.json()["idToken"]
            verifier = secrets.token_hex(32)
            challenge = hashlib.sha256(verifier.encode("ascii")).hexdigest()
            grant = call("POST", "/desktop/login", expected=201, json={"challenge": challenge}).json()
            login = "/desktop/login/" + grant["request_id"]
            state.setdefault("desktop_login_ids", []).append(grant["request_id"])
            save()
            call("POST", login + "/exchange", expected=202, json={"verifier": verifier})
            call("POST", login + "/authorize", expected=401)
            call("POST", login + "/authorize", token=tokens["owner"])
            call("POST", login + "/exchange", expected=403, json={"verifier": "b" * 64})
            custom = call("POST", login + "/exchange", json={"verifier": verifier}).json()["custom_token"]
            response = client.post("https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken",
                params={"key": config["apiKey"]}, json={"token": custom, "returnSecureToken": True})
            del custom
            if response.status_code != 200:
                raise RuntimeError("Firebase desktop exchange failed; provider details withheld.")
            tokens["owner"] = response.json()["idToken"]
            profile = call("GET", "/me", token=tokens["owner"]).json()
            assert profile["user"]["uid"] == state["users"]["owner"]["uid"]
            call("POST", login + "/exchange", expected=410, json={"verifier": verifier})
            checked("one-use browser approval, proof binding and real Firebase custom sign-in")
            project = call("POST", "/projects", token=tokens["owner"], expected=201,
                           json={"name": "Desktop QA " + secrets.token_hex(4)}).json()
            pid = project["id"]
            state["projects"].append(pid)
            save()
            prefix = "/projects/" + pid + "/workspace"
            invitation = call("POST", "/projects/" + pid + "/invitations",
                token=tokens["owner"], expected=201,
                json={"email": state["users"]["viewer"]["email"], "role": "viewer"}).json()
            state["invites"].append(invitation["id"])
            save()
            call("POST", "/invitations/" + invitation["id"] + "/accept", token=tokens["viewer"])
            boot = call("GET", prefix + "/bootstrap", token=tokens["owner"]).json()
            draft = call("PUT", prefix + "/console/script", token=tokens["owner"],
                json={"code": "print('saved draft; never executed in cloud')", "version": boot["draft"]["version"]}).json()
            call("PUT", prefix + "/console/script", token=tokens["owner"], expected=409,
                 json={"code": "concurrent edit", "version": boot["draft"]["version"]})
            assert call("GET", prefix + "/console/script", token=tokens["owner"]).json() == draft
            checked("draft persistence and conflict rejection")
            file = call("POST", prefix + "/datasets/example", token=tokens["owner"], expected=201).json()
            raw = call("GET", prefix + "/files/" + file["id"] + "/download", token=tokens["owner"]).content
            assert hashlib.sha256(raw).hexdigest() == file["data_hash"]
            # A real local worker generates the record. Its code is never sent to
            # the cloud execution route; publication carries sanitized data only.
            from openecon.console import ConsoleSession
            from openecon.workspace import Workspace
            with tempfile.TemporaryDirectory(prefix="openecon-desktop-cloud-qa-") as folder:
                root = Path(folder)
                (root / "wages.csv").write_bytes(raw)
                console = ConsoleSession(Workspace(root))
                try:
                    record = console.execute('import openecon as oe\n'
                        'df = oe.read("wages.csv")\n'
                        'm = oe.ols(data=df, y="wage", x=["education", "experience"], covariance="HC3")\n'
                        'display(df.head())\ndisplay(m)\ndisplay(oe.plot.coefficients(m))')
                finally:
                    console.close()
            assert record["status"] == "ok"
            record["actor_uid"] = state["users"]["owner"]["uid"]
            payload = {"record": record, "input_files": [{"id": file["id"], "data_hash": file["data_hash"]}]}
            shared = call("POST", prefix + "/desktop/results", token=tokens["owner"], expected=201, json=payload).json()
            assert call("POST", prefix + "/desktop/results", token=tokens["owner"], expected=201, json=payload).json() == shared
            call("POST", prefix + "/desktop/results", token=tokens["viewer"], expected=403, json=payload)
            call("POST", prefix + "/desktop/results", token=tokens["outsider"], expected=404, json=payload)
            history = call("GET", prefix + "/console", token=tokens["viewer"]).json()["history"]
            assert len(history) == 1 and history[0]["execution_origin"] == "desktop"
            assert history[0]["input_files"][0]["data_hash"] == file["data_hash"]
            assert [output["type"] for output in history[0]["outputs"]] == ["table", "model", "plot"]
            assert all(output["latex"] for output in history[0]["outputs"])
            assert history[0]["outputs"][1]["data"]["nobs"] == 480
            final = call("GET", prefix + "/bootstrap", token=tokens["owner"]).json()
            assert final["status"]["running"] is False and final["status"]["session_generation"] == 0
            checked("real local OLS, D3 and LaTeX archived once with no cloud compute")
            checked("viewer readback and write denial; outsider isolation")
            report["passed"] = True
        finally:
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(report, indent=2))
    print("Desktop cloud QA passed; remove its exact disposable resources with verify_team_live.py cleanup.")


if __name__ == "__main__":
    main()
