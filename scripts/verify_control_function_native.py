"""Verify source-pinned frozen code and dedicated native Control Function QA outputs.

Seeding writes only synthetic local project/script state; native Run is always
clicked through the actual app. Readback never submits a console execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
from types import CodeType
from urllib.request import Request, urlopen

import verify_multiple_testing_installed as native

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "org.openecon.qa.control-function-eight-20261007"
PRODUCT = "Control Function QA"
APP = Path.home() / "Applications" / f"{PRODUCT}.app"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Control Function Eight QA"
STATE = "control-functions-eight-state.json"
MARKER = "CONTROL_FUNCTION_INSTALLED:"
MODULES = [
    'openecon.engines.separation',
    'openecon',
    'openecon.models',
    'openecon.analysis',
    'openecon.analysis_contracts',
    'openecon.frame',
    'openecon.resources',
    'openecon.console_worker',
    'openecon.latex',
    'openecon.econometrics',
    'openecon.econometrics.registry',
    'openecon.econometrics.core',
    'openecon.econometrics.control_function',
    'openecon.econometrics.control_function.commands',
    'openecon.econometrics.control_function.kernels',
    'openecon.econometrics.control_function.state',
    'openecon.econometrics.control_function.postest',
    'openecon.econometrics.glm',
    'openecon.econometrics.glm.kernels',
    'openecon.econometrics.glm.families',
    'openecon.econometrics.glm.common',
    'openecon.engines.linalg',
    'openecon.engines.optimize',
    'openecon.engines.covariance',
    'openecon.engines.distributions',
    'openecon.engines.inference',
    'openecon.engines.contracts',
    'openecon.engines.torch_engine',
    'openecon.engines.execution',
    'openecon.econometrics.mi.common',
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_parity(runtime, source_ref=None):
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")

    def normalize(code):
        return code.replace(
            co_filename="<pinned>",
            co_consts=tuple(normalize(v) if isinstance(v, CodeType) else v for v in code.co_consts),
        )

    hashes = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = (
            path.read_text()
            if source_ref is None
            else subprocess.check_output(
                ["git", "show", f"{source_ref}:{path.relative_to(ROOT).as_posix()}"],
                cwd=ROOT,
                text=True,
            )
        )
        if normalize(archive.extract(name)) != normalize(
            compile(source, str(path), "exec", dont_inherit=True)
        ):
            raise RuntimeError(f"Frozen source mismatch: {name}")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    return hashes


def code_for(project, source_ref=None):
    workspace = DATA / "projects" / project
    example = ((ROOT / "examples/control_functions_eight.py").read_text() if source_ref is None
               else subprocess.check_output(
                   ["git", "show", f"{source_ref}:examples/control_functions_eight.py"],
                   cwd=ROOT, text=True,
               ))
    header = f"""import importlib, importlib.util, os, sys, torch
from pathlib import Path
assert getattr(sys, "frozen", False)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
torch.set_num_threads(1)
_frozen_root = Path(sys._MEIPASS).resolve()
_workspace = Path({str(workspace)!r}).resolve()
assert Path.cwd().resolve() == _workspace
_modules = {{}}
for _name in {MODULES!r}:
    _module = importlib.import_module(_name)
    _file = Path(_module.__file__).resolve()
    assert _file.is_relative_to(_frozen_root)
    _modules[_name] = str(_file)
"""
    footer = f"""
from openecon.resources import use_workspace_budget
with use_workspace_budget(1):
    try:
        oe.cfregress(data=pd.DataFrame(raw_inputs["gaussian"]["data"]),
                     y="y",endogenous="d",x=["x"],instruments=["z1","z2"])
    except oe.AnalysisError as _exc:
        assert _exc.code == "workspace_limit"
    else:
        raise AssertionError("Frozen control-function fit failed its allocation gate")
_bytes = json.dumps(states,sort_keys=True,allow_nan=False,separators=(",", ":")).encode()
assert len(_bytes) < 32*1024*1024
_file = _workspace / {STATE!r}
assert not _file.is_symlink()
_temp = _workspace / ({STATE!r} + ".tmp")
assert not _temp.is_symlink()
_temp.write_bytes(_bytes)
_temp.replace(_file)
assert json.loads(_file.read_text()) == states
_input_bytes = json.dumps(plain(raw_inputs),sort_keys=True,allow_nan=False,separators=(",", ":")).encode()
_input_file = _workspace / "control-function-inputs.json"
assert not _input_file.is_symlink()
_input_file.write_bytes(_input_bytes)
print({MARKER!r}+json.dumps({{"frozen":True,"root":str(_frozen_root),"modules":_modules,
    "workspace":str(_workspace),"worker_pid":os.getpid(),"stages":8,"cases":16,
    "states_sha256":__import__("hashlib").sha256(_bytes).hexdigest(),"states_bytes":len(_bytes),
    "inputs_sha256":__import__("hashlib").sha256(_input_bytes).hexdigest(),
    "source_oracles_available":False,"restored_equal":True,"resource_guard_passed":True}},sort_keys=True))
"""
    return header + example + footer


def displayed_table_checks(outputs, cases):
    """Bind every actual rendered coefficient/SE/p/CI row to a full HC0 state."""
    kinds = ("gaussian", "logit", "probit", "cloglog", "poisson", "gamma", "inverse_gaussian", "fractional_logit")
    selected = {case["kind"]: case for case in cases if case["case_id"].endswith("_hc0")}
    checks = {}
    for kind, output in zip(kinds, outputs, strict=True):
        if output.get("type") != "table" or not output.get("latex"):
            raise RuntimeError("Require actual exportable coefficient table for " + kind)
        table = output["data"]
        coefficients = selected[kind]["result"]["coefficients"]
        if table["total_rows"] != len(coefficients) or len(table["rows"]) != len(coefficients):
            raise RuntimeError("Displayed coefficient rows were truncated: " + kind)
        for row, coefficient in zip(table["rows"], coefficients, strict=True):
            for name, expected in coefficient.items():
                if name not in table["columns"] or row[table["columns"].index(name)] != expected:
                    raise RuntimeError("Displayed coefficient output differs: " + kind + ":" + name)
        checks[kind] = {"full_coefficient_rows":len(coefficients), "latex_available":True, "saved_inference_equal":True}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--seed", action="store_true")
    actions.add_argument("--after-restart", action="store_true")
    parser.add_argument(
        "--source-ref",
        required=True,
        help="Explicit committed source for a retained QA bundle; never an automatic fallback",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_ref = (
        None
        if args.source_ref is None
        else subprocess.check_output(
            ["git", "rev-parse", "--verify", args.source_ref + "^{commit}"], cwd=ROOT, text=True
        ).strip()
    )
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    if (info["CFBundleIdentifier"] != IDENTIFIER or APP.is_symlink()
            or APP.name != PRODUCT + ".app"):
        raise RuntimeError("Only the dedicated synthetic Control Function QA app is allowed.")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    static_root = runtime.parent / "_internal/openecon/static"
    frontend = {path.relative_to(static_root).as_posix(): digest(path)
                for path in static_root.rglob("*") if path.is_file()}
    identity = {
        "app": str(APP),
        "identifier": IDENTIFIER,
        "version": info["CFBundleShortVersionString"],
        "source_ref": source_ref,
        "native_executable_sha256": digest(APP / "Contents/MacOS" / info["CFBundleExecutable"]),
        "runtime_manifest_sha256": digest(APP / "Contents/Resources/runtime/runtime-manifest.json"),
        "runtime_sha256": digest(runtime),
        "frontend_sha256": hashlib.sha256(json.dumps(frontend, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_parity": source_parity(runtime, source_ref),
        "signature": "ad-hoc QA; no Developer ID/notarization/public release claim",
    }
    signature = subprocess.run(["codesign", "-dv", str(APP)], check=True, text=True, capture_output=True).stderr
    if "Signature=adhoc" not in signature:
        raise RuntimeError("Require ad-hoc dedicated QA signature; no release identity is accepted.")
    pids = native.runtime_pids(str(runtime))
    native.require_unaliased_directory(DATA)
    native.require_unaliased_directory(DATA / "projects", may_be_missing=True)
    port_file = DATA / ".runtime-port.json"
    if port_file.is_symlink() or port_file.stat().st_size > 256:
        raise RuntimeError("Unexpected dedicated runtime port alias/size.")
    port = json.loads(port_file.read_text())["port"]
    if type(port) is not int or not 1024 <= port <= 65535:
        raise RuntimeError("Invalid dedicated local port.")
    native.verify_listener(pids, port)
    token = None

    def call(path, body=None):
        native.verify_listener(pids, port)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        req = Request(
            f"http://127.0.0.1:{port}" + path,
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urlopen(req, timeout=30) as response:
            return json.load(response)

    token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call(
            "/api/desktop/local-projects",
            {
                "name": PROJECT,
                "description": "Synthetic local-only MARKET-506 through MARKET-513 acceptance.",
            },
        )
        projects = call("/api/desktop/local-projects")["projects"]
    if len(projects) != 1 or projects[0]["name"] != PROJECT:
        raise RuntimeError("Only the single dedicated synthetic project is allowed.")
    project = projects[0]["id"]
    native.require_unaliased_directory(DATA / "projects" / project)
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    code = code_for(project, source_ref)
    if args.seed:
        if args.receipt.exists():
            raise RuntimeError("Refuse to overwrite a prior acceptance seed.")
        script = call(
            prefix + "/console/scripts", {"name": "control_functions_eight_installed.py", "code": code}
        )
        if call(prefix + "/console/scripts/" + script["id"])["code"] != code:
            raise RuntimeError("Saved script readback differs.")
        record = {
            "status": "seeded",
            "identity": identity,
            "project_id": project,
            "script_id": script["id"],
            "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "runtime_pids": pids,
        }
    else:
        record = json.loads(args.receipt.read_text())
        if record["identity"] != identity or record["project_id"] != project:
            raise RuntimeError("Native identity/source changed.")
        if call(prefix + "/console/scripts/" + record["script_id"])["code"] != code:
            raise RuntimeError("Saved source changed.")
        history = call(prefix + "/console")["history"]
        runs = [
            r
            for r in history
            if r.get("status") == "ok" and r.get("code") == code and MARKER in r.get("stdout", "")
        ]
        if len(runs) != 1:
            raise RuntimeError("Require one completed actual native Run with the exact source.")
        run = runs[0]
        proof = json.loads(
            next(
                line[len(MARKER) :]
                for line in run["stdout"].splitlines()
                if line.startswith(MARKER)
            )
        )
        if (
            proof["stages"] != 8
            or proof["cases"] != 16
            or not proof["frozen"]
            or not proof["restored_equal"]
            or not proof["resource_guard_passed"]
            or proof["source_oracles_available"]
        ):
            raise RuntimeError("Invalid native execution proof.")
        workspace = DATA / "projects" / project
        frozen_root = runtime.parent / "_internal"
        if (proof["workspace"] != str(workspace.resolve())
                or proof["root"] != str(frozen_root.resolve())
                or set(proof["modules"]) != set(MODULES)
                or any(not Path(path).is_relative_to(frozen_root.resolve())
                       for path in proof["modules"].values())):
            raise RuntimeError("Native module/workspace provenance differs from the selected bundle.")
        if len(run["outputs"]) != 8 or not all(output.get("latex") for output in run["outputs"]):
            raise RuntimeError("Eight complete rendered/exportable native results are required.")
        path = DATA / "projects" / project / STATE
        if path.is_symlink() or digest(path) != proof["states_sha256"] or path.stat().st_size != proof["states_bytes"]:
            raise RuntimeError("Persisted result state changed.")
        states = json.loads(path.read_text())
        cases = states["cases"]
        kinds = {"gaussian", "logit", "probit", "cloglog", "poisson", "gamma", "inverse_gaussian", "fractional_logit"}
        expected = {kind+suffix for kind in kinds for suffix in ("_hc0", "_cr0")}
        if len(cases) != 16 or {case["case_id"] for case in cases} != expected:
            raise RuntimeError("Require every full HC0/CR0 outcome state.")
        for case in cases:
            if case["result"] != case["restored_result"] or not case["predictions"]:
                raise RuntimeError("Complete native save/restore/prediction differs.")
            if case["result"]["extra"]["control_function_state"]["kind"] != case["kind"]:
                raise RuntimeError("Native full state kind differs.")
        displayed = displayed_table_checks(run["outputs"], cases)
        input_path = DATA / "projects" / project / "control-function-inputs.json"
        if input_path.is_symlink():
            raise RuntimeError("Unexpected synthetic input alias.")
        args.receipt.with_name("native-inputs.json").write_bytes(input_path.read_bytes())
        execution_hash = hashlib.sha256(
            json.dumps(run, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inputs_hash = digest(input_path)
        if inputs_hash != proof["inputs_sha256"]:
            raise RuntimeError("Native original input hash differs.")
        if not args.after_restart and proof["worker_pid"] not in pids:
            raise RuntimeError("Observed native worker PID is absent from the selected runtime.")
        if args.after_restart:
            native.require_pids_exited(record["runtime_pids"])
            if record["execution_sha256"] != execution_hash:
                raise RuntimeError("History/output/events changed after full native restart.")
            if record["inputs_sha256"] != inputs_hash:
                raise RuntimeError("Saved synthetic inputs changed after full native restart.")
        record.update(
            status="native-restarted" if args.after_restart else "native-verified",
            execution_id=run["id"],
            execution_sha256=execution_hash,
            execution_record=run,
            proof=proof,
            outputs=run["outputs"],
            complete_states=states,
            cases=cases,
            displayed_table_checks=displayed,
            inputs=json.loads(input_path.read_text()),
            states_sha256=digest(path),
            inputs_sha256=inputs_hash,
            current_runtime_pids=pids,
        )
        if not args.after_restart:
            record["runtime_pids"] = pids
        args.receipt.with_name("native-execution.json").write_text(json.dumps(run, indent=2))
        args.receipt.with_name("persisted-states.json").write_text(json.dumps(states, indent=2))
        for i, output in enumerate(run["outputs"], 1):
            args.receipt.with_name(f"output-{i}.tex").write_text(output["latex"])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2))
    print(
        json.dumps(
            {"status": record["status"], "stages": 8, "runtime_sha256": identity["runtime_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
