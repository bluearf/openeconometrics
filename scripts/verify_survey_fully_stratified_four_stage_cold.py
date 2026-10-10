"""Seed/read back a second actual native Run: fresh frozen worker, all fits disabled.

Run after normal Quit/full-PID exit/reopen and --after-restart verification.
This helper only seeds saved source and reads history; it never executes code.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import verify_survey_fully_stratified_four_stage_installed as installed

SCRIPT = "survey_fully_stratified_four_stage_cold_replay.py"
MARKER = "SURVEY_FULLY_STRATIFIED_FOUR_STAGE_COLD:"


def code_for(receipt):
    workspace = installed.DATA / "projects" / receipt["project_id"]
    proof = receipt["proof"]
    return f"""import importlib, importlib.util, json, os, sys
from pathlib import Path
import pandas as pd
import openecon as oe
assert getattr(sys,"frozen",False)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
_workspace=Path({str(workspace)!r}).resolve()
assert Path.cwd().resolve()==_workspace
_original={proof!r}
assert os.getpid()!=_original["worker_pid"]
for _name in {installed.MODULES!r}:
    assert Path(importlib.import_module(_name).__file__).resolve().is_relative_to(Path(sys._MEIPASS).resolve())
from openecon.econometrics.survey import fully_stratified_four_stage_regression as _regression
def _forbidden(*args,**kwargs):
    raise AssertionError("A cold saved-state replay must not fit a model")
_regression._fit=_forbidden
for _name in ("mean","total","ratio","proportion","regress","logit","probit","poisson"):
    setattr(oe,"survey_fully_stratified_four_stage_"+_name,_forbidden)
_payloads={{}}
for _key,_entry in _original["files"].items():
    _path=_workspace/_entry["file"]
    assert not _path.is_symlink()
    _bytes=_path.read_bytes()
    assert len(_bytes)==_entry["bytes"]<8*1024*1024
    assert __import__("hashlib").sha256(_bytes).hexdigest()==_entry["sha256"]
    _payloads[_key]=json.loads(_bytes)
def table_state(table):
    return dict(index=table.index.tolist(),columns=table.columns.tolist(),data=table.astype(object).where(pd.notna(table),None).values.tolist(),attrs=table.attrs)
def digest(value):
    return __import__("hashlib").sha256(json.dumps(value,allow_nan=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
for _name,_state in _payloads["states"].items():
    _cls=oe.SurveyFullyStratifiedFourStageResult if _name in ("mean","total","ratio","proportion") else oe.SurveyFullyStratifiedFourStageRegressionResult
    _model=_cls.model_validate_json(json.dumps(_state,allow_nan=False))
    assert _model.model_dump(mode="json")==_state
    _table=_model.to_frame()
    assert table_state(_table)==_payloads["postestimation"][_name]
    assert oe.to_latex(_table)
    if _name in ("mean","total","ratio","proportion"):
        _targets={{"contrast":table_state(_model.contrast([1.]+[0.]*(len(_model.labels)-1)))}}
    else:
        _targets={{"lincom":table_state(_model.lincom([0.,1.,0.])),"test":table_state(_model.test([[0.,1.,0.],[0.,0.,1.]])),"predict":table_state(_model.predict(pd.DataFrame({{"x":[-.5,.5],"z":[.2,-.2]}})))}}
    assert {{k:digest(v) for k,v in _targets.items()}}==_original["saved_postestimation"][_name]
    display(_table)
print({MARKER!r}+json.dumps(dict(ok=True,cases=8,source_ref={receipt["identity"]["source_ref"]!r},worker_pid=os.getpid(),original_worker_pid=_original["worker_pid"],frozen=True,fits_disabled=True,files=_original["files"]),sort_keys=True))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", action="store_true")
    args = parser.parse_args()
    original = json.loads(args.native_receipt.read_text())
    assert original["status"] == "native-restarted"
    source = original["identity"]["source_ref"]
    assert (
        installed.pinned_source(source, "scripts/verify_survey_fully_stratified_four_stage_cold.py")
        == Path(__file__).read_text()
    )
    runtime = installed.APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    assert installed.digest(runtime) == original["identity"]["runtime_sha256"]
    installed.source_parity(runtime, source)
    pids = installed.native.runtime_pids(str(runtime))
    port = json.loads((installed.DATA / ".runtime-port.json").read_text())["port"]
    assert type(port) is int and 1024 <= port <= 65535
    token = None

    def call(path, body=None):
        installed.native.verify_listener(pids, port)
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
    assert (
        len(projects) == 1
        and projects[0]["id"] == original["project_id"]
        and projects[0]["name"] == installed.PROJECT
    )
    prefix = f"/api/desktop/projects/{original['project_id']}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    code = code_for(original)
    history = call(prefix + "/console")["history"]
    first = [run for run in history if run["id"] == original["execution_id"]]
    assert len(first) == 1 and installed.object_digest(first[0]) == original["execution_sha256"]
    if args.seed:
        assert not args.receipt.exists() and len(history) == 1
        script = call(prefix + "/console/scripts", dict(name=SCRIPT, code=code))
        assert call(prefix + "/console/scripts/" + script["id"])["code"] == code
        result = dict(
            status="seeded",
            source_ref=source,
            script_id=script["id"],
            source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        )
    else:
        result = json.loads(args.receipt.read_text())
        assert (
            len(history) == 2
            and call(prefix + "/console/scripts/" + result["script_id"])["code"] == code
        )
        second = [run for run in history if run.get("status") == "ok" and run.get("code") == code]
        assert len(second) == 1
        run = second[0]
        markers = [
            line[len(MARKER) :] for line in run["stdout"].splitlines() if line.startswith(MARKER)
        ]
        assert len(markers) == 1
        proof = json.loads(markers[0])
        assert proof["ok"] and proof["cases"] == 8 and proof["fits_disabled"] and proof["frozen"]
        assert (
            proof["source_ref"] == source and proof["worker_pid"] != original["proof"]["worker_pid"]
        )
        assert (
            proof["original_worker_pid"] == original["proof"]["worker_pid"]
            and proof["files"] == original["proof"]["files"]
        )
        outputs = run["outputs"]
        assert len(outputs) == 8
        for output in outputs:
            assert output["type"] == "table" and output["latex"] and output["data"]["rows"]
        for i, output in enumerate(outputs):
            assert (
                hashlib.sha256(output["latex"].encode()).hexdigest()
                == original["outputs"][i]["latex_sha256"]
            )
        for entry in proof["files"].values():
            path = installed.DATA / "projects" / original["project_id"] / entry["file"]
            assert (
                path.stat().st_size == entry["bytes"] and installed.digest(path) == entry["sha256"]
            )
        result.update(
            status="cold-replayed",
            proof=proof,
            execution_id=run["id"],
            execution_sha256=installed.object_digest(run),
            history_sha256=installed.object_digest(history),
            runtime_sha256=installed.digest(runtime),
        )
        args.receipt.with_name("cold-native-execution.json").write_text(json.dumps(run, indent=2))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result["status"], source_ref=source)))


if __name__ == "__main__":
    main()
