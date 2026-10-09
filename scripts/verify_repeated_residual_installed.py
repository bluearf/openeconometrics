"""Seed/read back an owned QA project; Run and Quit occur through native UI.

This never calls the console execute endpoint or opens a human project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from verify_repeated_residual_eight import header, source_identity, RESULT_NAMES, POST_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Library/Application Support/org.openecon.qa.repeatedgls"
APP = Path.home() / "Applications/OpenEconometrics Repeated GLS QA.app"
EXAMPLE = ROOT / "docs/examples/repeated_residual_eight.py"
NAME = "Repeated residual eight QA"
ALL_RESULT_NAMES = (*RESULT_NAMES, *POST_NAMES)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_code():
    directory = DATA / "repeated-gls-complete-results"
    return header(directory) + EXAMPLE.read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--ui-observed", action="store_true")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    assert type(port) is int and 1024 <= port <= 65535
    token = None

    def call(path, body=None):
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
        call("/api/desktop/local-projects", {
            "name": NAME,
            "description": "Owned synthetic MARKET-648..655 repeated residual acceptance.",
        })
        projects = call("/api/desktop/local-projects")["projects"]
    assert len(projects) == 1 and projects[0]["name"] == NAME
    project = projects[0]["id"]
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    code = native_code()
    if args.seed:
        script = call(prefix + "/console/scripts", {"name": EXAMPLE.name, "code": code})
        record = {"status": "seeded", "project_id": project, "script_id": script["id"]}
    else:
        record = json.loads(args.receipt.read_text())
        assert record["project_id"] == project
        assert call(prefix + "/console/scripts/" + record["script_id"])["code"] == code
        completed = [item for item in call(prefix + "/console")["history"]
                     if item["status"] == "ok"
                     and "REPEATED_GLS_ACCEPTANCE_OK " in item.get("stdout", "")]
        if record.get("execution_id"):
            execution = next(item for item in completed if item["id"] == record["execution_id"])
        else:
            assert len(completed) == 1, [item["id"] for item in completed]
            execution = completed[0]
        assert execution["code"] == code
        assert [item["type"] for item in execution["outputs"]] == ["table"] * 8
        assert "Display limit reached" not in execution["stdout"]
        assert all("\\begin{tabular}" in item.get("latex", "")
                   or "\\begin{longtable}" in item.get("latex", "")
                   for item in execution["outputs"])
        proof = json.loads(next(line.split("REPEATED_GLS_ACCEPTANCE_OK ", 1)[1]
                                for line in execution["stdout"].splitlines()
                                if line.startswith("REPEATED_GLS_ACCEPTANCE_OK ")))
        fingerprint = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()
        result_root = DATA / "repeated-gls-complete-results"
        hashes = {name: digest(result_root / (name + ".json")) for name in ALL_RESULT_NAMES}
        counts = {}
        for name in ALL_RESULT_NAMES:
            payload = json.loads((result_root / (name + ".json")).read_text())
            assert payload["tables"] and payload["attrs"] and len(payload["latex"]) > 100
            counts[name] = {key: len(value["data"]) for key, value in payload["tables"].items()}
        if record.get("execution_sha256"):
            assert record["execution_sha256"] == fingerprint
            assert record["complete_result_hashes"] == hashes
            record["app_restart_readback_unchanged"] = True
        if args.ui_observed:
            record["native_run_and_rendered_tables_observed"] = True
        record.update(
            status="passed", execution_id=execution["id"], execution_sha256=fingerprint,
            complete_result_hashes=hashes, complete_table_counts=counts, proof=proof,
            ordered_table_outputs=8, publication_latex=True, saved_source_matches_example=True,
            console_duration_ms=execution["duration_ms"],
        )
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    record.update(
        installed_app=str(APP), runtime_sha256=digest(runtime), example_sha256=digest(EXAMPLE),
        compiled_modules_equal_source=source_identity(runtime),
        verifier_sha256=digest(__file__), primary_app_untouched=True,
        human_project_untouched=True, release_delivered=False,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "project": NAME,
                      "restart_readback": record.get("app_restart_readback_unchanged", False)}))


if __name__ == "__main__":
    main()
