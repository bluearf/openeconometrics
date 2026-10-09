"""Read back native UI Runs and complete model and target state; never executes code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from verify_saved_targets_eight_runtime import MARKER, hashed, verify_outputs


def capture(origin,root):
    projects = json.loads((root/"local-projects.json").read_text())["projects"]
    matches = [row for row in projects if row["name"] == "Saved Targets Eight QA"]
    assert len(matches) == 1
    project = matches[0]["id"]
    prefix = f"/api/desktop/projects/{project}/workspace"
    def request(path,token=None):
        with urlopen(Request(origin+path,headers={"X-OpenEcon-Token":token} if token else {}),timeout=15) as response:
            return json.load(response)
    desktop = request("/api/desktop/session")["token"]
    status = request("/api/desktop/status",desktop)
    assert status["project_id"] == project
    token = request(prefix+"/session")["token"]
    history = request(prefix+"/console",token)["history"]
    matching = [row for row in history if MARKER in row.get("stdout","")]
    assert len(matching) == 2
    assert all(row["status"] == "error" for row in history if row not in matching)
    runs = sorted(matching,key=lambda run:verify_outputs(run)["part"])
    proofs = [verify_outputs(run) for run in runs]
    assert [proof["part"] for proof in proofs] == [1, 2]
    assert sum(len(proof["output_contracts"]) for proof in proofs) == 32
    assert len({name for proof in proofs for name in proof["methods"]}) == 8
    document = request(prefix+"/console/script",token)
    assert document["code"] == runs[1]["code"]
    return dict(project_id=project,history_count=len(history),accepted_run_count=2,
        retained_failed_input_attempts=[dict(run_id=run["id"],status=run["status"],
            hashes={key:hashed(run[key]) for key in ("code","stdout","outputs","events")})
            for run in history if run not in matching],
        complete_output_tables=sum(len(proof["output_contracts"]) for proof in proofs),
        runs=[dict(run_id=run["id"],duration_ms=run["duration_ms"],proof=proof,
                   hashes={key:hashed(run[key]) for key in ("code","stdout","outputs","events")}) for run,proof in zip(runs,proofs,strict=True)],
        document_hash=hashed(document),desktop_status=status)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin",required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    receipt=capture(args.origin,args.root)
    if args.output:
        before=json.loads(args.baseline.read_text())
        assert all(receipt[key] == before[key] for key in ("project_id","history_count","accepted_run_count","retained_failed_input_attempts","runs","document_hash","complete_output_tables"))
        assert receipt["desktop_status"]["console"]["pid"] is None
        receipt.update(status="passed",native_two_runs_verified=True,all_eight_complete_states_retained=True,
            full_code_stdout_outputs_events_equal_after_native_quit_relaunch=True,no_worker_started_for_history=True,
            owned_qa_app=True,main_installation_changed=False,public_release_delivered=False)
        args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    else:
        args.baseline.parent.mkdir(parents=True,exist_ok=True)
        args.baseline.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("project_id","history_count","complete_output_tables")}))
