"""Seed/read back org.openecon.qa.binarymediation; native Run is performed via UI.

This helper never invokes execute. It reads only its owned synthetic QA project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from verify_binary_mediation_runtime import EXAMPLE, header, source_identity, verify_files, verify_outputs

DATA = Path.home() / "Library/Application Support/org.openecon.qa.binarymediation"
APP = Path.home() / "Applications/OpenEconometrics Binary Mediation QA.app"
NAME = "Eight Binary Mediation Domains QA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",action="store_true")
    parser.add_argument("--ui-observed",action="store_true")
    parser.add_argument("--receipt",type=Path,required=True)
    args = parser.parse_args()
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    assert type(port) is int and 1024 <= port <= 65535
    token = desktop_token = None
    def call(path,body=None):
        headers = {"Content-Type":"application/json"}
        request_token = desktop_token if path=="/api/desktop/status" else token
        if request_token:
            headers["X-OpenEcon-Token"] = request_token
        request = Request(f"http://127.0.0.1:{port}"+path,headers=headers,
                          data=json.dumps(body).encode() if body is not None else None)
        with urlopen(request,timeout=30) as response:
            return json.load(response)
    token = desktop_token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call("/api/desktop/local-projects",{"name":NAME,"description":"Owned MARKET-441..448 synthetic binary mediation acceptance."})
        projects = call("/api/desktop/local-projects")["projects"]
    assert len(projects)==1 and projects[0]["name"]==NAME
    project = projects[0]["id"]
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open",{})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix+"/session")["token"]
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    sources = source_identity(runtime)
    directory = DATA / "complete-binary-mediation"
    code = header(directory)+EXAMPLE.read_text()
    if args.seed:
        script = call(prefix+"/console/scripts",{"name":EXAMPLE.name,"code":code})
        record = dict(status="seeded",project_id=project,script_id=script["id"])
    else:
        record = json.loads(args.receipt.read_text())
        assert record["project_id"]==project
        assert call(prefix+"/console/scripts/"+record["script_id"])["code"]==code
        matching = [row for row in call(prefix+"/console")["history"]
                    if "BINARY_MEDIATION_ACCEPTANCE_OK " in row.get("stdout","")]
        if record.get("execution_id"):
            run = next(row for row in matching if row["id"]==record["execution_id"])
        else:
            assert len(matching)==1
            run = matching[0]
        proof = verify_outputs(run)
        assert run["code"]==code
        hashes = verify_files(directory,proof)
        execution_hash = hashlib.sha256(json.dumps(run,sort_keys=True).encode()).hexdigest()
        if record.get("execution_sha256"):
            assert record["execution_sha256"]==execution_hash
            assert record["complete_result_hashes"]==hashes
            record["repeated_execution_and_files_readback_unchanged"] = True
            assert call("/api/desktop/status")["console"]["pid"] is None
        if args.ui_observed:
            record["native_run_and_rendered_tables_observed"] = True
        record.update(status="passed",execution_id=run["id"],execution_sha256=execution_hash,
                      complete_result_hashes=hashes,compiled_modules_equal_source=sources,
                      duration_ms=run["duration_ms"],displayed_tables=16,saved_tables=proof["saved_tables"],
                      all_eight_contracts_saved=True,all_saved_inputs_replayed=True,
                      owned_qa_app=True,human_data_access=False,public_release_delivered=False,
                      minimum_macos="26.0; development Python binary floor, no macOS 15 compatibility claim")
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({key:record[key] for key in ("status","project_id")}))


if __name__ == "__main__":
    main()
