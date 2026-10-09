"""Seed/read back the dedicated QA app; execute Run through its native UI.

Only org.openecon.qa.causallearning's owned synthetic project is accessed.
This helper never invokes the console execute endpoint or a human project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from verify_causal_learning_runtime import digest, header, source_identity, NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Library/Application Support/org.openecon.qa.causallearning"
APP = Path.home() / "Applications/OpenEconometrics Causal Learning QA.app"
EXAMPLE = ROOT / "docs/examples/causal_learning.py"
NAME = "Causal Learning QA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument(
        "--ui-observed",
        action="store_true",
        help="Record a native Run and rendered tables already observed through computer use",
    )
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
        call(
            "/api/desktop/local-projects",
            {
                "name": NAME,
                "description": "Owned MARKET-157 acceptance; stationary synthetic data only.",
            },
        )
        projects = call("/api/desktop/local-projects")["projects"]
    assert len(projects) == 1 and projects[0]["name"] == NAME
    project = projects[0]["id"]
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    sources = source_identity(runtime)
    result_root = DATA / "causal-learning-complete-results"
    code = header(result_root) + EXAMPLE.read_text()
    if args.seed:
        script = call(prefix + "/console/scripts", {"name": EXAMPLE.name, "code": code})
        record = {"status": "seeded", "project_id": project, "script_id": script["id"]}
    else:
        record = json.loads(args.receipt.read_text())
        assert record["project_id"] == project
        assert call(prefix + "/console/scripts/" + record["script_id"])["code"] == code
        completed = [
            item
            for item in call(prefix + "/console")["history"]
            if item["status"] == "ok" and "CAUSAL_LEARNING_ACCEPTANCE_OK " in item.get("stdout", "")
        ]
        if record.get("execution_id"):
            execution = next(item for item in completed if item["id"] == record["execution_id"])
        else:
            assert len(completed) == 1, [item["id"] for item in completed]
            execution = completed[0]
        assert execution["code"] == code
        assert [item["type"] for item in execution["outputs"]] == ["model"] * 9 + ["table"]
        assert all(
            any(
                "\\begin{" + environment + "}" in item.get("latex", "")
                for environment in ("tabular", "longtable")
            )
            for item in execution["outputs"]
        )
        proof = json.loads(
            next(
                line.split("CAUSAL_LEARNING_ACCEPTANCE_OK ", 1)[1]
                for line in execution["stdout"].splitlines()
                if line.startswith("CAUSAL_LEARNING_ACCEPTANCE_OK ")
            )
        )
        fingerprint = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()
        hashes = {name: digest(result_root / (name + ".json")) for name in NAMES}
        from openecon.models import ResultBundle

        for name in NAMES:
            restored = ResultBundle.model_validate_json(
                (result_root / (name + ".json")).read_text()
            )
            assert len(restored.covariance_matrix) == len(restored.coefficients)
            assert restored.provenance["stata_parity_validated"] is False
            assert len(restored.to_latex()) > 100
        if record.get("execution_sha256"):
            assert record["execution_sha256"] == fingerprint
            assert record["complete_result_hashes"] == hashes
            record["app_restart_readback_unchanged"] = True
        if args.ui_observed:
            record["native_run_and_rendered_tables_observed"] = True
        record.update(
            status="passed",
            execution_id=execution["id"],
            execution_sha256=fingerprint,
            complete_result_hashes=hashes,
            proof=proof,
            ordered_model_outputs=9,
            ordered_table_outputs=1,
            publication_latex=True,
            saved_source_matches_example=True,
            console_duration_ms=execution["duration_ms"],
        )
    record.update(
        installed_app=str(APP),
        runtime_sha256=digest(runtime),
        compiled_modules_equal_source=sources,
        example_sha256=digest(EXAMPLE),
        verifier_sha256=digest(__file__),
        primary_app_untouched=True,
        human_project_untouched=True,
        release_delivered=False,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + "\n")
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
