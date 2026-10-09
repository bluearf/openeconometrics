"""Seed/read back the dedicated local QA app; the native Run action is manual/UI.

Requires the separately built org.openecon.qa.finite application. It never
uses a human project, primary application, cloud account or external endpoint.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from verify_finite_runtime import MODULES, PROCEDURES, digest, normalized

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Library/Application Support/org.openecon.qa.finite"
EXAMPLE = ROOT / "docs/examples/finite_eight.py"
APP = Path.home() / "Applications/OpenEconometrics Finite Regression QA.app"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument(
        "--accept-new-run",
        action="store_true",
        help="Pin the latest completed native run after an explicit rebuild",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    assert type(port) is int and 1024 <= port <= 65535
    origin = f"http://127.0.0.1:{port}"
    token = None

    def call(path, body=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        request = Request(
            origin + path,
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urlopen(request, timeout=20) as response:
            return json.load(response)

    token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call(
            "/api/desktop/local-projects",
            {
                "name": "Finite Regression Eight QA",
                "description": "Owned acceptance of MARKET-385..392; public fixtures and synthetic rows.",
            },
        )
        projects = call("/api/desktop/local-projects")["projects"]
    assert len(projects) == 1 and projects[0]["name"] == "Finite Regression Eight QA"
    project = projects[0]["id"]
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    from PyInstaller.archive.readers import CArchiveReader

    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for name in MODULES:
        path = ROOT / "src/openecon" / (name.replace(".", "/") + ".py")
        module = ("openecon." + name).removesuffix(".__init__")
        expected = compile(path.read_text(), str(path), "exec", dont_inherit=True)
        assert normalized(archive.extract(module)) == normalized(expected), module
        sources[module] = digest(path)
    header = """import sys, importlib
from pathlib import Path
assert getattr(sys, 'frozen', False)
assert importlib.util.find_spec('scipy') is None
assert importlib.util.find_spec('statsmodels') is None
for name in %r:
    assert Path(importlib.import_module(name).__file__).is_relative_to(Path(sys._MEIPASS))
""" % (tuple(sources),)
    code = (
        header
        + "ARTIFACT_DIRECTORY = "
        + repr(str(DATA / "finite-artifacts"))
        + "\n"
        + EXAMPLE.read_text()
    )
    if args.seed:
        script = call(prefix + "/console/scripts", {"name": EXAMPLE.name, "code": code})
        record = {
            "status": "seeded",
            "project_id": project,
            "script_id": script["id"],
            "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        }
    else:
        record = json.loads(args.receipt.read_text())
        assert record["project_id"] == project
        script = call(prefix + "/console/scripts/" + record["script_id"])
        assert script["code"] == code
        state = call(prefix + "/console")
        history = state["history"]
        completed = [
            item
            for item in history
            if item.get("status") == "ok" and "FINITE_EIGHT_OK " in item.get("stdout", "")
        ]
        assert completed, [item.get("status") for item in history]
        if args.accept_new_run:
            execution = max(completed, key=lambda item: item["created_at"])
            record.pop("execution_sha256", None)
            record.pop("restart_readback_unchanged", None)
        elif record.get("execution_id"):
            execution = next(item for item in completed if item["id"] == record["execution_id"])
        else:
            assert len(completed) == 1
            execution = completed[0]
        assert execution["code"] == code
        proof = json.loads(
            next(
                line.split("FINITE_EIGHT_OK ", 1)[1]
                for line in execution["stdout"].splitlines()
                if line.startswith("FINITE_EIGHT_OK ")
            )
        )
        assert (
            proof["procedures"] == list(PROCEDURES)
            and set(proof["artifact_sha256"]) == set(PROCEDURES)
            and proof["complete_artifacts_equal"]
            and proof["frozen"]
        )
        assert len(execution["outputs"]) == 8 and all(
            item.get("latex") for item in execution["outputs"]
        )
        fingerprint = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()
        artifacts = {}
        for name, sha in proof["artifact_sha256"].items():
            path = DATA / "finite-artifacts" / (name + ".json")
            artifact = json.loads(path.read_text())
            payload_sha = hashlib.sha256(
                json.dumps(
                    artifact["payload"],
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            assert payload_sha == artifact["sha256"] == sha
            artifacts[name] = digest(path)
        if record.get("execution_sha256"):
            assert record["execution_sha256"] == fingerprint
            assert record["artifact_files_sha256"] == artifacts
            record["restart_readback_unchanged"] = True
        record.update(
            status="passed",
            execution_id=execution["id"],
            duration_ms=execution["duration_ms"],
            created_at=execution["created_at"],
            completed_at=execution["completed_at"],
            execution_sha256=fingerprint,
            proof=proof,
            ordered_outputs=[item["type"] for item in execution["outputs"]],
            saved_source_matches_example=True,
            all_outputs_have_latex=True,
            artifact_files_sha256=artifacts,
            eight_full_artifact_files_verified=True,
        )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record.update(
        installed_app=str(APP),
        runtime_sha256=digest(runtime),
        compiled_modules_equal_source=sources,
        primary_app_untouched=True,
        human_project_untouched=True,
    )
    args.receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
