"""Verify source-pinned frozen code and dedicated native MI Extensions QA outputs.

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
IDENTIFIER = "org.openecon.qa.mi-extensions-eight-20261007"
PRODUCT = "OpenEconometrics MI Extensions QA"
APP = Path.home() / "Applications" / f"{PRODUCT}.app"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "MI Extensions Inference QA"
STATE = "mi-extensions-eight-state.json"
MARKER = "MI_EXTENSIONS_INSTALLED:"
MODULES = [
    "openecon",
    "openecon.models",
    "openecon.analysis",
    "openecon.linear_ols",
    "openecon.linear_ols.spec",
    "openecon.linear_ols.design",
    "openecon.linear_ols.estimation",
    "openecon.engines.torch_engine",
    "openecon.engines.inference",
    "openecon.console_worker",
    "openecon.econometrics.mi",
    "openecon.econometrics.mi.common",
    "openecon.econometrics.mi.diagnostics",
    "openecon.econometrics.mi.generation",
    "openecon.econometrics.mi.chained",
    "openecon.econometrics.mi.pooling",
    "openecon.econometrics.mi.joint",
    "openecon.econometrics.mi.discrete",
    "openecon.econometrics.mi.sensitivity",
    "openecon.econometrics.mi.passive",
    "openecon.econometrics.mi.lincom",
    "openecon.econometrics.registry",
    "openecon.analysis_contracts",
    "openecon.econometrics.core",
    "openecon.resources",
    "openecon.engines.distributions",
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


def code_for(project):
    workspace = DATA / "projects" / project
    example = (ROOT / "docs/examples/mi_extensions_eight.py").read_text()
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
        oe.mi_pool([[0.]*32]*100, [torch.eye(32,dtype=torch.float64).tolist()]*100,
                   imputation_description="Budget refusal synthetic oracle")
    except oe.AnalysisError as _exc:
        assert _exc.code == "workspace_limit"
    else:
        raise AssertionError("Frozen pooling failed its allocation gate")
_bytes = json.dumps(states,sort_keys=True,allow_nan=False,separators=(",", ":")).encode()
_file = _workspace / {STATE!r}
_file.write_bytes(_bytes)
assert json.loads(_file.read_text()) == states
_inputs = {{}}
for _name, _frame in {{"poisson":poisson_input,"ordinal":ordinal_input,
                     "multinomial":multinomial_input,"normal_delta":normal_input,
                     "logit_delta":logit_input,"poisson_delta":poisson_input}}.items():
    _inputs[_name] = {{k:[None if pd.isna(v) else v.item() if hasattr(v,"item") else v for v in _frame[k]] for k in _frame}}
    _inputs[_name]["__index__"] = list(_frame.index)
_inputs["pool_tables"] = {{_name: {{"columns":list(_frame.columns),
    "rows":[[None if pd.isna(v) else v.item() if hasattr(v,"item") else v for v in _row]
            for _row in _frame.itertuples(index=False,name=None)],"index":[str(v) for v in _frame.index]}}
    for _name,_frame in pool.tables.items()}}
(_workspace / "mi-inputs.json").write_text(json.dumps(_inputs,sort_keys=True,allow_nan=False))
print({MARKER!r}+json.dumps({{"frozen":True,"root":str(_frozen_root),"modules":_modules,
    "workspace":str(_workspace),"worker_pid":os.getpid(),"stages":8,
    "states_sha256":__import__("hashlib").sha256(_bytes).hexdigest(),"states_bytes":len(_bytes),
    "source_oracles_available":False,"restored_equal":True,"resource_guard_passed":True}},sort_keys=True))
"""
    return header + example + footer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--after-restart", action="store_true")
    parser.add_argument(
        "--source-ref",
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
    if info["CFBundleIdentifier"] != IDENTIFIER or APP.is_symlink():
        raise RuntimeError("Only the dedicated synthetic MI Extensions QA app is allowed.")
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
    pids = native.runtime_pids(str(runtime))
    native.require_unaliased_directory(DATA)
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
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
                "description": "Synthetic local-only MARKET-433 through MARKET-440 acceptance.",
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
    code = code_for(project)
    if args.seed:
        if args.receipt.exists():
            raise RuntimeError("Refuse to overwrite a prior acceptance seed.")
        script = call(
            prefix + "/console/scripts", {"name": "mi_extensions_eight_installed.py", "code": code}
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
        if set(states) != {"poisson", "ordinal", "multinomial", "normal_delta", "logit_delta", "poisson_delta", "passive", "lincom"}:
            raise RuntimeError("Eight full missing-data states were not preserved.")
        from openecon import MIDiscreteResult, MIDeltaResult, MIPassiveResult, MILincomResult

        for name, state in states.items():
            cls = (MIDiscreteResult if name in {"poisson", "ordinal", "multinomial"}
                   else MIDeltaResult if name.endswith("_delta")
                   else MIPassiveResult if name == "passive" else MILincomResult)
            cls.model_validate(state)
        input_path = DATA / "projects" / project / "mi-inputs.json"
        if input_path.is_symlink():
            raise RuntimeError("Unexpected synthetic input alias.")
        args.receipt.with_name("native-inputs.json").write_bytes(input_path.read_bytes())
        execution_hash = hashlib.sha256(
            json.dumps(run, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inputs_hash = digest(input_path)
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
