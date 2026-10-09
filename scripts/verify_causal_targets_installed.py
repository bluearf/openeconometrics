"""Seed/read back the dedicated causal-targets QA app; Run occurs through its UI.

Only org.openecon.qa.causaltargets and its single synthetic project are accessed.
This helper never calls a console execution endpoint. A native app restart must
be performed externally before --after-app-restart pins unchanged readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
from urllib.request import Request, urlopen

from verify_causal_targets_runtime import (
    EXAMPLE,
    MODULES,
    NAMES,
    artifact_files,
    digest,
    fingerprint,
    header,
    proof_from_execution,
    source_identity,
)

DATA = Path.home() / "Library/Application Support/org.openecon.qa.causaltargets"
APP = Path.home() / "Applications/OpenEconometrics Causal Targets QA.app"
NAME = "Causal Targets Eight QA"


def runtime_listener(port, runtime):
    """Read only the dedicated loopback listener and verify its owned binary."""
    listeners = subprocess.run(
        ["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(set(listeners)) != 1:
        raise RuntimeError("The dedicated QA port must have one runtime listener")
    pid = int(listeners[0])
    command = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], check=True, capture_output=True, text=True
    ).stdout.strip()
    if not command.startswith(str(runtime)):
        raise RuntimeError("The QA loopback listener does not belong to its installed runtime")
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument(
        "--ui-observed",
        action="store_true",
        help="Record native Run and rendered tables already observed through computer use",
    )
    parser.add_argument(
        "--after-app-restart",
        action="store_true",
        help="Verify the pinned completed execution after an actual externally performed app restart",
    )
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-ref", help="Immutable accepted source SHA for compiled-module identity")
    args = parser.parse_args()
    if args.seed and (args.ui_observed or args.after_app_restart):
        parser.error("Seeding and completed-run readback are separate proof steps")
    bundle = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    if bundle.get("CFBundleIdentifier") != "org.openecon.qa.causaltargets":
        raise RuntimeError("The installed app must be the dedicated owned causal-targets QA bundle")
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    if type(port) is not int or not 1024 <= port <= 65535:
        raise RuntimeError("The owned QA app port descriptor is invalid")
    token = None

    def call(path, body=None):
        # The only writes permitted in this helper are owned project/script
        # creation and opening that project. Scientific execution belongs to UI.
        if "/console/execute" in path or "/console/reset" in path:
            raise RuntimeError("Installed verification does not execute or reset a console")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        request = Request(
            f"http://127.0.0.1:{port}" + path,
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call(
            "/api/desktop/local-projects",
            {
                "name": NAME,
                "description": "Owned MARKET-481..488 acceptance; synthetic observational targets, known-nuisance survival and declared OVB sensitivity assumptions.",
            },
        )
        projects = call("/api/desktop/local-projects")["projects"]
    if len(projects) != 1 or projects[0]["name"] != NAME:
        raise RuntimeError(
            "Only the single owned synthetic Causal Targets Eight QA project is allowed"
        )
    project = projects[0]["id"]
    if args.seed:
        if args.receipt.exists():
            raise RuntimeError(
                "Choose a new seed receipt path; an accepted run cannot be overwritten by reseeding"
            )
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    listener_pid = runtime_listener(port, runtime)
    sources = source_identity(runtime, args.source_ref)
    directory = DATA / "causal-targets-complete-artifacts"
    code = header(directory) + EXAMPLE.read_text()
    if args.seed:
        script = call(prefix + "/console/scripts", {"name": EXAMPLE.name, "code": code})
        record = dict(
            status="seeded",
            project_id=project,
            script_id=script["id"],
            native_run_and_rendered_tables_observed=False,
        )
    else:
        record = json.loads(args.receipt.read_text())
        if record["project_id"] != project:
            raise RuntimeError("The receipt belongs to another project")
        if call(prefix + "/console/scripts/" + record["script_id"])["code"] != code:
            raise RuntimeError("Saved native script differs from the current complete example")
        completed = [
            item
            for item in call(prefix + "/console")["history"]
            if item.get("status") == "ok" and "CAUSAL_TARGETS_EIGHT_OK " in item.get("stdout", "")
        ]
        if record.get("execution_id"):
            selected = [item for item in completed if item["id"] == record["execution_id"]]
        else:
            selected = completed
        if len(selected) != 1:
            raise RuntimeError("A unique completed native example run must be present or pinned")
        execution = selected[0]
        if execution["code"] != code:
            raise RuntimeError("The completed native run differs from the exact seeded code")
        proof = proof_from_execution(execution)
        hashes = artifact_files(directory, proof)
        # Restoration here checks all complete tables and state with the source
        # helper, without fitting or executing inside the installed app.
        from openecon.econometrics.causal_design.common import (
            causal_design_load,
            causal_design_save,
        )

        for name in NAMES:
            artifact = json.loads((directory / (name + ".json")).read_text())
            restored = causal_design_load(artifact)
            if causal_design_save(restored) != artifact:
                raise RuntimeError(f"The full native artifact did not round-trip: {name}")
            if not any(
                "\\begin{" + kind + "}" in restored.to_latex() for kind in ("tabular", "longtable")
            ):
                raise RuntimeError(f"The full native artifact lacks LaTeX tables: {name}")
        execution_hash = fingerprint(execution)
        if record.get("execution_sha256"):
            if (
                record["execution_sha256"] != execution_hash
                or record["artifact_files_sha256"] != hashes
                or record["executed_code_sha256"] != hashlib.sha256(code.encode()).hexdigest()
            ):
                raise RuntimeError(
                    "Pinned execution/code/full artifacts changed between native readbacks"
                )
            if args.after_app_restart:
                if record.get("runtime_listener_pid") == listener_pid:
                    raise RuntimeError(
                        "A native app restart must replace the pinned owned runtime listener"
                    )
                record["app_restart_readback_unchanged"] = True
                record["runtime_listener_changed_after_app_restart"] = True
        elif args.after_app_restart:
            raise RuntimeError("Pin the completed run before verifying an actual app restart")
        if args.ui_observed:
            record["native_run_and_rendered_tables_observed"] = True
        record.update(
            status="passed",
            frozen_execution=True,
            execution_id=execution["id"],
            execution_sha256=execution_hash,
            executed_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            created_at=execution["created_at"],
            completed_at=execution["completed_at"],
            console_duration_ms=execution["duration_ms"],
            proof=proof,
            ordered_output_types=["table"] * 8,
            representative_outputs_have_latex=True,
            representative_table_rows=[len(item["data"]["rows"]) for item in execution["outputs"]],
            full_artifact_payload_sha256=proof["artifact_sha256s"],
            artifact_files_sha256=hashes,
            eight_full_artifact_files_verified=True,
            full_artifacts_roundtrip_equal=True,
            saved_source_matches_example=True,
            external_oracle_packages_absent=True,
        )
    record.update(
        installed_app=str(APP),
        application_bundle_identifier=bundle["CFBundleIdentifier"],
        runtime_sha256=digest(runtime),
        compiled_modules_equal_source=sources,
        scientific_source_ref=args.source_ref,
        runtime_listener_pid=listener_pid,
        current_scoped_modules=list(MODULES),
        example_sha256=digest(EXAMPLE),
        verifier_sha256=digest(__file__),
        primary_app_untouched=True,
        human_project_untouched=True,
        public_release_delivered=False,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": record["status"],
                "project": NAME,
                "restart_readback": record.get("app_restart_readback_unchanged", False),
            }
        )
    )


if __name__ == "__main__":
    main()
