"""Seed/read back the dedicated Survey DEFF QA app; native Run is external.

This helper writes only the new synthetic project and its script during seed.
It never executes or resets a native console. Full app Quit, visible relaunch,
and rendered-table observation belong to the external native UI controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
from urllib.request import Request, urlopen

import verify_multiple_testing_installed as native
from verify_survey_deff_runtime import (
    EXAMPLE, MARKER, MODULES, artifact_files, digest, fingerprint,
    header, proof_from_execution, save_json, source_identity,
)

IDENTIFIER = "org.openecon.qa.surveydeff"
PRODUCT = "OpenEconometrics Survey DEFF QA"
APP = Path.home() / "Applications" / (PRODUCT + ".app")
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Survey DEFF QA"


def installed_identity(source_ref):
    native.require_unaliased_directory(APP)
    native.require_unaliased_directory(DATA)
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    if info.get("CFBundleIdentifier") != IDENTIFIER or APP.name != PRODUCT + ".app":
        raise RuntimeError("Only the new dedicated owned Survey DEFF QA bundle is allowed")
    executable = info["CFBundleExecutable"]
    if Path(executable).name != executable:
        raise RuntimeError("The native executable name must be a simple bundle filename")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    signature = subprocess.run(["codesign", "-dv", str(APP)], check=True,
                               capture_output=True, text=True).stderr
    if "Signature=adhoc" not in signature:
        raise RuntimeError("Require the separately ad-hoc signed synthetic QA bundle")
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    return runtime, {
        "app": str(APP), "identifier": IDENTIFIER, "product": PRODUCT,
        "version": info["CFBundleShortVersionString"],
        "minimum_system_version": info["LSMinimumSystemVersion"],
        "native_executable_sha256": digest(APP / "Contents/MacOS" / executable),
        "runtime_sha256": digest(runtime),
        "runtime_manifest_sha256": digest(APP / "Contents/Resources/runtime/runtime-manifest.json"),
        "compiled_modules_equal_source": source_identity(runtime, source_ref),
        "scientific_source_ref": source_ref,
        "signature": "ad-hoc owned QA; no Developer ID/notarization/public release claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--ui-observed", action="store_true",
                        help="Actual native Run/rendered tables were independently observed")
    parser.add_argument("--after-app-restart", action="store_true")
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.seed and (args.ui_observed or args.after_app_restart):
        parser.error("Seed and native completed-run readback are separate stages")
    if args.seed and args.receipt.exists():
        parser.error("A new seed receipt is required; never reseed accepted history")
    runtime, identity = installed_identity(args.source_ref)
    pids = native.runtime_pids(str(runtime))
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    if type(port) is not int or not 1024 <= port <= 65535:
        raise RuntimeError("The dedicated loopback descriptor has an invalid port")
    native.verify_listener(pids, port)
    token = None

    def call(path, body=None):
        # Prevent this acceptance reader from executing, resetting, or touching
        # any other route. Every write is explicitly scoped to the seed stage.
        if "/console/execute" in path or "/console/reset" in path:
            raise RuntimeError("Native acceptance never executes or resets a console")
        if body is not None and not args.seed:
            raise RuntimeError("Completed native readback is read-only")
        native.verify_listener(pids, port)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        request = Request(f"http://127.0.0.1:{port}" + path, headers=headers,
                          data=json.dumps(body).encode() if body is not None else None)
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call("/api/desktop/local-projects", {
            "name": PROJECT,
            "description": "Owned synthetic MARKET-210 single-stage unequal/equal-weight DEFF acceptance.",
        })
        projects = call("/api/desktop/local-projects")["projects"]
    if len(projects) != 1 or projects[0]["name"] != PROJECT:
        raise RuntimeError("Only the single dedicated synthetic Survey DEFF QA project is allowed")
    project = projects[0]["id"]
    if not isinstance(project, str) or not re.fullmatch(r"[0-9a-f]{32}", project):
        raise RuntimeError("The dedicated project ID is invalid")
    workspace = DATA / "projects" / project
    native.require_unaliased_directory(workspace)
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    artifacts = workspace / "survey-deff-complete-artifacts"
    code = header(artifacts) + EXAMPLE.read_text()
    source_hash = hashlib.sha256(code.encode()).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.seed:
        if call(prefix + "/console")["history"]:
            raise RuntimeError("Seed only a new synthetic project with no execution history")
        script = call(prefix + "/console/scripts", {"name": EXAMPLE.name, "code": code})
        if call(prefix + "/console/scripts/" + script["id"])["code"] != code:
            raise RuntimeError("The seeded native script did not persist exactly")
        record = dict(status="seeded", identity=identity, project_id=project,
                      script_id=script["id"], source_sha256=source_hash,
                      native_run_and_rendered_tables_observed=False,
                      example_sha256=digest(EXAMPLE), seed_runtime_pids=pids)
        save_json(args.receipt, record)
    else:
        record = json.loads(args.receipt.read_text())
        if (record["identity"] != identity or record["project_id"] != project
                or record["source_sha256"] != source_hash
                or record["example_sha256"] != digest(EXAMPLE)):
            raise RuntimeError("The dedicated identity/project/seeded example changed")
        script = call(prefix + "/console/scripts/" + record["script_id"])
        if script["code"] != code:
            raise RuntimeError("The saved native script changed")
        console = call(prefix + "/console")
        history = console["history"]
        if len(history) != 1 or history[0].get("code") != code or MARKER not in history[0].get("stdout", ""):
            raise RuntimeError("Require one original actual native Run with the exact seeded source")
        execution = history[0]
        proof = proof_from_execution(execution)
        hashes = artifact_files(artifacts, proof, execution)
        execution_hash, history_hash = fingerprint(execution), fingerprint(history)
        if args.after_app_restart:
            if not record.get("execution_sha256"):
                raise RuntimeError("Pin the completed native run before verifying an app restart")
            native.require_pids_exited(record["readback_runtime_pids"])
            if console["status"]["pid"] is not None:
                raise RuntimeError("A cold app readback must have no computation worker")
            if (record["execution_sha256"] != execution_hash
                    or record["history_sha256"] != history_hash
                    or record["artifact_files_sha256"] != hashes
                    or record["saved_script_sha256"] != fingerprint(script)):
                raise RuntimeError("Original execution/history/script/full artifacts changed on app restart")
            record["app_restart_readback_unchanged"] = True
        elif record.get("execution_sha256") and (
                record["execution_sha256"] != execution_hash
                or record["artifact_files_sha256"] != hashes):
            raise RuntimeError("A pinned original native run/artifact changed")
        stage = "after-app-restart" if args.after_app_restart else "native-readback"
        target = args.receipt.parent / stage
        target.mkdir(exist_ok=False)
        save_json(target / "console.json", console)
        save_json(target / "original-execution.json", execution)
        save_json(target / "saved-script.json", script)
        shutil.copytree(artifacts, target / "complete-artifacts")
        for index, output in enumerate(execution["outputs"], 1):
            (target / f"output-{index}.tex").write_text(output["latex"], encoding="utf-8")
        if artifact_files(target / "complete-artifacts", proof, execution) != hashes:
            raise RuntimeError("The durable full native artifact copy changed")
        record.update(
            status="native-restarted" if args.after_app_restart else "native-verified",
            proof=proof, execution_id=execution["id"], execution_sha256=execution_hash,
            history_sha256=history_hash, saved_script_sha256=fingerprint(script),
            artifact_files_sha256=hashes, current_runtime_pids=pids,
            actual_output_count=len(execution["outputs"]), full_artifacts_retained=True,
            frozen_execution=True, external_oracle_packages_absent=True,
        )
        if args.ui_observed:
            record["native_run_and_rendered_tables_observed"] = True
        if not args.after_app_restart:
            record["readback_runtime_pids"] = pids
        record.update(verifier_sha256=digest(__file__), scoped_modules=list(MODULES),
                      primary_app_untouched=True, human_project_untouched=True,
                      public_release_delivered=False)
        temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
        save_json(temporary, record)
        temporary.replace(args.receipt)
    print(json.dumps({"status": record["status"], "project": PROJECT,
                      "restart": record.get("app_restart_readback_unchanged", False)}))


if __name__ == "__main__":
    main()
