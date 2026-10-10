"""Seed/read back the dedicated six-stage QA app; Run is a native UI action.

No console execution route is called. Full source/PYZ identity, local listener,
actual output, saved state and process-exit/reopen evidence remain distinct.
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
PRODUCT = "Foundations QA"
IDENTIFIER = "org.openecon.qa.foundations-20261008"
APP = Path.home() / "Applications" / (PRODUCT + ".app")
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Four Method Foundations QA"
STATE = "six-model-state.json"
MARKER = "SIX_MODEL_NATIVE:"
RESTORE_MARKER = "SIX_MODEL_RESTORE:"
MODULES = [
    "openecon",
    "openecon.analysis_contracts",
    "openecon.models",
    "openecon.frame",
    "openecon.console_worker",
    "openecon.resources",
    "openecon.econometrics",
    "openecon.econometrics.registry",
    "openecon.econometrics.core",
    "openecon.econometrics.mi.common",
    "openecon.econometrics.resident_cpu",
    "openecon.engines.contracts",
    "openecon.engines.inference",
    "openecon.engines.distributions",
    "openecon.econometrics.summary_state",
    "openecon.econometrics.state_lifecycle",
    "openecon.econometrics.bayesian",
    "openecon.econometrics.bayesian.core",
    "openecon.econometrics.bayesian.posterior",
    "openecon.econometrics.bayesian.commands",
    "openecon.econometrics.latent",
    "openecon.econometrics.latent.cfa",
    "openecon.econometrics.ivquantile",
    "openecon.econometrics.ivquantile.common",
    "openecon.econometrics.ivquantile.kernels",
    "openecon.econometrics.ivquantile.api",
    "openecon.econometrics.quantile",
    "openecon.econometrics.quantile.kernels",
    "openecon.econometrics.tsworkflows",
    "openecon.econometrics.tsworkflows.ssengine",
    "openecon.econometrics.tsworkflows.ssdiffuse",
    "openecon.econometrics.tsworkflows.sspace",
    "openecon.econometrics.bayesian.hypothesis",
    "openecon.econometrics.bayesian.hypothesis_kernels",
    "openecon.econometrics.latent.sem",
    "openecon.econometrics.tsworkflows.dfm",
    "openecon.econometrics.weakiv",
    "openecon.econometrics.weakiv.api",
    "openecon.econometrics.weakiv.common",
    "openecon.econometrics.weakiv.kernels",
    "openecon.econometrics.mixtures",
    "openecon.econometrics.mixtures.gaussian",
    "openecon.econometrics.supervised",
    "openecon.econometrics.supervised.common",
    "openecon.econometrics.supervised.cart",
    "openecon.econometrics.supervised.split",
    "openecon.econometrics.supervised.metrics",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def committed(pin, path):
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=ROOT, text=True)


def parity(runtime, pin):
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")

    def normalized(code):
        return code.replace(
            co_filename="<source-pin>",
            co_consts=tuple(
                normalized(v) if isinstance(v, CodeType) else v for v in code.co_consts
            ),
        )

    result = {}
    for name in MODULES:
        source = ROOT / "src" / Path(*name.split("."))
        source = source / "__init__.py" if source.is_dir() else source.with_suffix(".py")
        text = committed(pin, source.relative_to(ROOT).as_posix())
        if normalized(archive.extract(name)) != normalized(
            compile(text, str(source), "exec", dont_inherit=True)
        ):
            raise RuntimeError("Frozen scientific/source dependency mismatch: " + name)
        result[name] = hashlib.sha256(text.encode()).hexdigest()
    return result


def code_for(project, pin, *, restore=False, batch=0):
    workspace = DATA / "projects" / project
    samples = {
        name: committed(pin, "examples/" + name + ".py")
        for name in ("bayesian_hypothesis", "latent_sem", "dynamic_factor", "weakiv_clr",
                     "finite_mixture", "supervised_cart")
    }
    prefix = f"""import importlib, importlib.util, os, sys, torch
from pathlib import Path
assert getattr(sys, "frozen", False)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
torch.set_num_threads(1)
_WORKSPACE = Path({str(workspace)!r}).resolve()
_frozen_root = Path(sys._MEIPASS).resolve()
assert Path.cwd().resolve() == _WORKSPACE
assert not (_WORKSPACE / {STATE!r}).is_symlink()
_EXAMPLES = {samples!r}
_DISPLAY_BATCH = {batch!r}
_modules = {{}}
for _name in {MODULES!r}:
    _module = importlib.import_module(_name)
    _file = Path(_module.__file__).resolve()
    assert _file.is_relative_to(_frozen_root)
    _modules[_name] = str(_file)
"""
    path = "examples/six_model_restore.py" if restore else "examples/six_model_installed.py"
    body = committed(pin, path)
    marker = (RESTORE_MARKER if restore else MARKER) + str(batch) + ":"
    suffix = f"""
_state_bytes = (_WORKSPACE / {STATE!r}).read_bytes()
print({marker!r}+json.dumps({{"stages":6,"frozen":True,"source_ref":{pin!r},
    "root":str(_frozen_root),"workspace":str(_WORKSPACE),"worker_pid":os.getpid(),
    "modules":_modules,"full_covariance":True,"semantic_replay":True,
    "groups_complete":False,"cart_uncertainty_available":False,"tables":_table_count,
    "batch":{batch!r},"displayed_tables":_display_count,
    "states_sha256":__import__("hashlib").sha256(_state_bytes).hexdigest(),
    "states_bytes":len(_state_bytes)}},sort_keys=True))
"""
    return prefix + body + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=range(4), default=0)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--after-restart", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    pin = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_ref + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    if info["CFBundleIdentifier"] != IDENTIFIER or APP.name != PRODUCT + ".app":
        raise RuntimeError("Require the dedicated synthetic Foundations QA application.")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    signature = subprocess.run(
        ["codesign", "-dv", str(APP)], capture_output=True, text=True, check=True
    ).stderr
    if "Signature=adhoc" not in signature:
        raise RuntimeError("Require separately ad-hoc signed local QA identity.")
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    static = runtime.parent / "_internal/openecon/static"
    if not (static / "index.html").is_file() or not (static / "assets").is_dir():
        raise RuntimeError("The installed frontend is missing.")
    frontend = {
        p.relative_to(static).as_posix(): digest(p) for p in static.rglob("*") if p.is_file()
    }
    identity = {
        "source_ref": pin,
        "identifier": IDENTIFIER,
        "app": str(APP),
        "version": info["CFBundleShortVersionString"],
        "minimum_system_version": info["LSMinimumSystemVersion"],
        "native_executable_sha256": digest(APP / "Contents/MacOS" / info["CFBundleExecutable"]),
        "runtime_sha256": digest(runtime),
        "manifest_sha256": digest(APP / "Contents/Resources/runtime/runtime-manifest.json"),
        "frontend_sha256": hashlib.sha256(
            json.dumps(frontend, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_parity": parity(runtime, pin),
        "signature": "ad-hoc local QA; no signed/notarized/public-release claim",
    }
    pids = native.runtime_pids(str(runtime))
    native.require_unaliased_directory(DATA)
    native.require_unaliased_directory(DATA / "projects", may_be_missing=True)
    port_file = DATA / ".runtime-port.json"
    if port_file.is_symlink() or port_file.stat().st_size > 256:
        raise RuntimeError("Invalid dedicated runtime port descriptor.")
    port = json.loads(port_file.read_text())["port"]
    if type(port) is not int or not 1024 <= port <= 65535:
        raise RuntimeError("Invalid dedicated runtime port.")
    token = None

    def call(path, body=None):
        native.verify_listener(pids, port)
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
                "name": PROJECT,
                "description": "Synthetic source-pinned MARKET-646/647/680–683 six-stage acceptance; all eight parents remain open.",
            },
        )
        projects = call("/api/desktop/local-projects")["projects"]
    if len(projects) != 1 or projects[0]["name"] != PROJECT:
        raise RuntimeError("Require the single dedicated synthetic project.")
    project = projects[0]["id"]
    native.require_unaliased_directory(DATA / "projects" / project)
    if args.seed:
        call(f"/api/desktop/local-projects/{project}/open", {})
    prefix = f"/api/desktop/projects/{project}/workspace"
    token = None
    token = call(prefix + "/session")["token"]
    code = code_for(project, pin, restore=args.restore, batch=args.batch)
    marker = (RESTORE_MARKER if args.restore else MARKER) + str(args.batch) + ":"
    if args.seed:
        if args.receipt.exists():
            raise RuntimeError("Refuse to overwrite a previous native acceptance seed.")
        stem = "six_models_restore" if args.restore else "six_models_installed"
        name = stem + "_" + pin[:8] + "_batch" + str(args.batch) + ".py"
        script = call(prefix + "/console/scripts", {"name": name, "code": code})
        if call(prefix + "/console/scripts/" + script["id"])["code"] != code:
            raise RuntimeError("Saved script readback mismatch.")
        record = {
            "status": "seeded",
            "identity": identity,
            "project_id": project,
            "script_id": script["id"],
            "runtime_pids": pids,
            "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        }
    else:
        record = json.loads(args.receipt.read_text())
        if record["identity"] != identity or record["project_id"] != project:
            raise RuntimeError("Source/application identity changed.")
        if call(prefix + "/console/scripts/" + record["script_id"])["code"] != code:
            raise RuntimeError("Saved source changed.")
        history = call(prefix + "/console")["history"]
        runs = [
            r
            for r in history
            if r.get("status") == "ok" and r.get("code") == code and marker in r.get("stdout", "")
        ]
        if len(runs) != 1:
            raise RuntimeError("Require exactly one completed actual native UI Run.")
        run = runs[0]
        proof = json.loads(
            next(
                line[len(marker) :]
                for line in run["stdout"].splitlines()
                if line.startswith(marker)
            )
        )
        if (
            proof["stages"] != 6
            or not proof["frozen"]
            or not proof["full_covariance"]
            or not proof["semantic_replay"]
            or proof["groups_complete"]
            or proof.get("source_ref") != pin
        ):
            raise RuntimeError("Invalid scoped native execution proof.")
        frozen_root = (runtime.parent / "_internal").resolve()
        workspace = DATA / "projects" / project
        if (
            proof["root"] != str(frozen_root)
            or proof["workspace"] != str(workspace.resolve())
            or set(proof["modules"]) != set(MODULES)
            or any(not Path(p).is_relative_to(frozen_root) for p in proof["modules"].values())
        ):
            raise RuntimeError("Native module/workspace provenance changed.")
        state_file = workspace / STATE
        if (
            state_file.is_symlink()
            or digest(state_file) != proof["states_sha256"]
            or state_file.stat().st_size != proof["states_bytes"]
        ):
            raise RuntimeError("Full saved native result state changed.")
        states = json.loads(state_file.read_text())
        if states.get("schema") != "openecon.six-model-stages.qa.v1" or set(states) != {
            "schema", "bayes", "sem", "dfm", "clr", "mixture", "cart",
        }:
            raise RuntimeError("Incomplete six-stage native state.")
        tables = [o for o in run["outputs"] if o.get("type") == "table"]
        expected_tables = 79 if args.restore else 72
        if proof["tables"] != expected_tables:
            raise RuntimeError("Incomplete rendered six-family table coverage.")
        displayed_tables = min(20, expected_tables - 20 * args.batch)
        if proof.get("batch") != args.batch or proof.get("displayed_tables") != displayed_tables:
            raise RuntimeError("Native rendered table batch coverage changed.")
        if len(tables) != displayed_tables or any(not o.get("latex") for o in tables):
            raise RuntimeError("Require all actual rendered/exportable native result tables.")
        execution_sha = hashlib.sha256(
            json.dumps(run, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if args.after_restart:
            native.require_pids_exited(record["runtime_pids"])
            if execution_sha != record["execution_sha256"] or proof != record["proof"]:
                raise RuntimeError("Retained execution/history changed after full restart.")
        elif proof["worker_pid"] not in pids:
            raise RuntimeError(
                "The observed native execution worker is absent from the selected runtime."
            )
        record.update(
            status="native-restarted" if args.after_restart else "native-verified",
            execution_sha256=execution_sha,
            execution_id=run["id"],
            proof=proof,
            runtime_pids=pids,
            complete_states=states,
            outputs=run["outputs"],
            execution_record=run,
        )
        args.receipt.with_name("native-execution.json").write_text(json.dumps(run, indent=2))
        args.receipt.with_name("persisted-states.json").write_bytes(state_file.read_bytes())
        for i, output in enumerate(tables):
            args.receipt.with_name(f"output-{i}.tex").write_text(output["latex"])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2))
    print(
        json.dumps({"status": record["status"], "project": project, "receipt": str(args.receipt)})
    )


if __name__ == "__main__":
    main()
