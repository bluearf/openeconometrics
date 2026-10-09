"""Seed/read back one dedicated native CF prediction project; never submit Run.

Run is clicked in the actual app. All full predictions and joint state are
independently checked by the offline acceptance checker, outside the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
from urllib.request import Request, urlopen

import verify_control_function_native as base
import verify_multiple_testing_installed as native

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "Control Prediction QA"
IDENTIFIER = "org.openecon.qa.control-prediction-eight-20261008"
APP = Path.home() / "Applications" / (PRODUCT + ".app")
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Conditional Prediction Eight QA"
MARKER = "CONTROL_PREDICTION_INSTALLED:"
QUERY_MODULES = [
    "openecon", "openecon.models", "openecon.frame", "openecon.dataset",
    "openecon.resources", "openecon.prediction_capabilities",
    "openecon.econometrics.postest.prediction", "openecon.econometrics.postest.control_prediction",
    "openecon.econometrics.postest.streaming_prediction", "openecon.econometrics.postest.index_codec",
    "openecon.econometrics.postest.inference", "openecon.econometrics.control_function.commands",
    "openecon.econometrics.control_function.kernels", "openecon.econometrics.control_function.state",
]


def code_for(workspace, manifest, benchmark_hash):
    benchmark = ROOT / "benchmarks/control_prediction_acceptance.py"
    return f'''import hashlib, importlib, importlib.util, json, os, runpy, sys, torch
from pathlib import Path
import pandas as pd
assert getattr(sys, "frozen", False)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
assert Path.cwd().resolve() == Path({str(workspace)!r}).resolve()
torch.set_num_threads(2)
_root = Path(sys._MEIPASS).resolve()
_modules = {{}}
for _name in {QUERY_MODULES!r}:
    _module = importlib.import_module(_name)
    _file = Path(_module.__file__).resolve()
    assert _file.is_relative_to(_root)
    _modules[_name] = str(_file)
_benchmark = Path({str(benchmark)!r})
assert hashlib.sha256(_benchmark.read_bytes()).hexdigest() == {benchmark_hash!r}
_acceptance = runpy.run_path(str(_benchmark))
_directory = Path.cwd() / "control-prediction-acceptance"
_record = _acceptance["run"](Path({str(manifest)!r}), _directory)
assert _record["status"] == "passed" and _record["frozen"]
assert len(_record["cases"]) == 10 and len(_record["small_cases"]) == 16
for _item in _record["small_cases"]:
    if _item["id"].endswith("_hc0"):
        _case = json.loads(Path(_item["file"]).read_text())
        _table = pd.DataFrame(_case["margins"]["ame"]["rows"])
        _table.attrs["publication_notes"] = [
            _case["estimator"], "Conditional derivatives; full Gamma/Beta HC0 covariance.",
            "Observed endogenous and instruments; structural APE unidentified."]
        display(_table)
print({MARKER!r}+json.dumps({{"frozen":True,"root":str(_root),"modules":_modules,
    "worker_pid":os.getpid(),"cases":16,"physical_cases":10,
    "run_sha256":_acceptance["digest"](_directory/"run.json"),
    "no_scipy_loaded":not _record["scipy_loaded"],
    "no_statsmodels_loaded":not _record["statsmodels_loaded"]}},sort_keys=True))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--after-restart", action="store_true")
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_ref = subprocess.check_output(["git", "rev-parse", args.source_ref + "^{commit}"], cwd=ROOT, text=True).strip()
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == IDENTIFIER and not APP.is_symlink()
    assert APP.name == PRODUCT + ".app"
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    # Compare every bundled local SDK module against the explicit source pin.
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    base.MODULES = sorted(name for name in archive.toc if name == "openecon" or name.startswith("openecon."))
    assert set(QUERY_MODULES) <= set(base.MODULES)
    identity = {"app": str(APP), "identifier": IDENTIFIER, "source_ref": source_ref,
                "runtime_sha256": base.digest(runtime),
                "runtime_manifest_sha256": base.digest(APP / "Contents/Resources/runtime/runtime-manifest.json"),
                "source_parity": base.source_parity(runtime, source_ref),
                "benchmark_sha256": base.digest(ROOT / "benchmarks/control_prediction_acceptance.py"),
                "manifest_sha256": base.digest(args.manifest),
                "signature": "ad-hoc isolated QA; no public release claim"}
    pids = native.runtime_pids(str(runtime))
    native.require_unaliased_directory(DATA)
    native.require_unaliased_directory(DATA / "projects", may_be_missing=True)
    port_file = DATA / ".runtime-port.json"
    assert not port_file.is_symlink() and port_file.stat().st_size < 256
    port = json.loads(port_file.read_text())["port"]
    assert type(port) is int and 1024 <= port <= 65535
    token = None

    def call(path, body=None):
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
        call("/api/desktop/local-projects", {"name": PROJECT, "description": "Synthetic local-only MARKET-553–560 acceptance."})
        projects = call("/api/desktop/local-projects")["projects"]
    assert len(projects) == 1 and projects[0]["name"] == PROJECT
    project = projects[0]["id"]
    workspace = DATA / "projects" / project
    native.require_unaliased_directory(workspace)
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    code = code_for(workspace, args.manifest.resolve(), identity["benchmark_sha256"])
    if args.seed:
        assert not args.receipt.exists()
        script = call(prefix + "/console/scripts", {"name": "conditional_prediction_eight.py", "code": code})
        assert call(prefix + "/console/scripts/" + script["id"])["code"] == code
        record = {"status": "seeded", "identity": identity, "project_id": project,
                  "script_id": script["id"], "runtime_pids": pids,
                  "code_sha256": hashlib.sha256(code.encode()).hexdigest()}
    else:
        record = json.loads(args.receipt.read_text())
        assert record["identity"] == identity and record["project_id"] == project
        assert call(prefix + "/console/scripts/" + record["script_id"])["code"] == code
        history = call(prefix + "/console")["history"]
        runs = [r for r in history if r.get("code") == code]
        assert len(runs) == 1 and runs[0]["status"] == "ok", [(r["status"],r.get("error")) for r in runs]
        run = runs[0]
        proof = json.loads(next(line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)))
        assert proof["frozen"] and proof["no_scipy_loaded"] and proof["no_statsmodels_loaded"]
        assert proof["cases"] == 16 and proof["physical_cases"] == 10
        assert Path(proof["root"]) == runtime.parent / "_internal"
        assert set(proof["modules"]) == set(QUERY_MODULES)
        assert len(run["outputs"]) == 8 and all(o["type"] == "table" and o.get("latex") for o in run["outputs"])
        directory = workspace / "control-prediction-acceptance"
        native.require_unaliased_directory(directory)
        assert base.digest(directory / "run.json") == proof["run_sha256"]
        measured = json.loads((directory / "run.json").read_text())
        hc0 = [item for item in measured["small_cases"] if item["id"].endswith("_hc0")]
        for output, item in zip(run["outputs"], hc0, strict=True):
            case = json.loads(Path(item["file"]).read_text())
            expected = case["margins"]["ame"]["rows"]
            table = output["data"]
            assert table["total_rows"] == len(expected) and len(table["rows"]) == len(expected)
            assert table["columns"] == list(expected[0])
            assert table["rows"] == [[row[name] for name in table["columns"]] for row in expected]
        hashes = {p.name: base.digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
        raw_history = workspace / "console/history.json"
        assert not raw_history.is_symlink() and raw_history.is_file()
        raw_history_sha256 = base.digest(raw_history)
        execution_hash = hashlib.sha256(json.dumps(run, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if args.after_restart:
            native.require_pids_exited(record["runtime_pids"])
            assert record["execution_sha256"] == execution_hash and record["output_files"] == hashes
            assert record["raw_history_sha256"] == raw_history_sha256
        else:
            assert proof["worker_pid"] in pids
            record["runtime_pids"] = pids
        record.update(status="native-restarted" if args.after_restart else "native-verified",
                      execution_sha256=execution_hash, output_files=hashes, proof=proof,
                      execution_record=run, current_runtime_pids=pids,
                      output_directory=str(directory), raw_history_sha256=raw_history_sha256,
                      all_eight_display_tables_match_full_saved_margins=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "project": project, "source": source_ref}))


if __name__ == "__main__":
    main()
