"""Verify source-pinned frozen code and dedicated native Fractional QA outputs.

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
IDENTIFIER = "org.openecon.qa.fractional-20261007-v2"
PRODUCT = "OpenEconometrics Fractional QA v2"
APP = Path.home() / "Applications" / f"{PRODUCT}.app"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Fractional Memory QA"
STATE = "fractional-state.json"
MARKER = "FRACTIONAL_INSTALLED:"
MODULES = ["openecon", "openecon.models", "openecon.console_worker",
           "openecon.econometrics.fractional", "openecon.econometrics.fractional.kernels",
           "openecon.econometrics.fractional.procedures", "openecon.econometrics.fractional.models",
           "openecon.econometrics.registry", "openecon.analysis_contracts", "openecon.econometrics.core"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_parity(runtime, source_ref=None):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    def normalize(code):
        return code.replace(co_filename="<pinned>", co_consts=tuple(normalize(v) if isinstance(v, CodeType) else v for v in code.co_consts))
    hashes = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = path.read_text() if source_ref is None else subprocess.check_output(
            ["git", "show", f"{source_ref}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT, text=True)
        if normalize(archive.extract(name)) != normalize(compile(source, str(path), "exec", dont_inherit=True)):
            raise RuntimeError(f"Frozen source mismatch: {name}")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    return hashes


def code_for(project):
    workspace = DATA / "projects" / project
    example = (ROOT / "docs/examples/fractional_memory.py").read_text()
    header = f'''import importlib, importlib.util, os, sys
from pathlib import Path
assert getattr(sys, "frozen", False)
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
_frozen_root = Path(sys._MEIPASS).resolve()
_workspace = Path({str(workspace)!r}).resolve()
assert Path.cwd().resolve() == _workspace
_modules = {{}}
for _name in {MODULES!r}:
    _module = importlib.import_module(_name)
    _file = Path(_module.__file__).resolve()
    assert _file.is_relative_to(_frozen_root)
    _modules[_name] = str(_file)
'''
    footer = f'''
_bytes = json.dumps(states, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
_file = _workspace / {STATE!r}
_file.write_bytes(_bytes)
assert json.loads(_file.read_text()) == states
print({MARKER!r} + json.dumps({{"frozen": True, "root": str(_frozen_root), "modules": _modules,
    "workspace": str(_workspace), "worker_pid": os.getpid(), "stages": 4,
    "states_sha256": __import__("hashlib").sha256(_bytes).hexdigest(), "states_bytes": len(_bytes),
    "source_oracles_available": False, "restored_equal": True}}, sort_keys=True))
'''
    return header + example + footer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--after-restart", action="store_true")
    parser.add_argument("--source-ref", help="Explicit committed source for a retained QA bundle; never an automatic fallback")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_ref = None if args.source_ref is None else subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_ref + "^{commit}"], cwd=ROOT, text=True).strip()
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    if info["CFBundleIdentifier"] != IDENTIFIER or APP.is_symlink():
        raise RuntimeError("Only the dedicated synthetic Fractional QA app is allowed.")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    identity = {"app": str(APP), "identifier": IDENTIFIER, "version": info["CFBundleShortVersionString"],
                "runtime_sha256": digest(runtime), "source_parity": source_parity(runtime, source_ref),
                "signature": "ad-hoc QA; no Developer ID/notarization/public release claim"}
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
        req = Request(f"http://127.0.0.1:{port}" + path, headers=headers,
                      data=json.dumps(body).encode() if body is not None else None)
        with urlopen(req, timeout=30) as response:
            return json.load(response)
    token = call("/api/desktop/session")["token"]
    projects = call("/api/desktop/local-projects")["projects"]
    if args.seed and not projects:
        call("/api/desktop/local-projects", {"name": PROJECT, "description": "Synthetic local-only MARKET-203/204/205/206 acceptance."})
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
        script = call(prefix + "/console/scripts", {"name": "fractional_memory_installed.py", "code": code})
        if call(prefix + "/console/scripts/" + script["id"])["code"] != code:
            raise RuntimeError("Saved script readback differs.")
        record = {"status": "seeded", "identity": identity, "project_id": project,
                  "script_id": script["id"], "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                  "runtime_pids": pids}
    else:
        record = json.loads(args.receipt.read_text())
        if record["identity"] != identity or record["project_id"] != project:
            raise RuntimeError("Native identity/source changed.")
        if call(prefix + "/console/scripts/" + record["script_id"])["code"] != code:
            raise RuntimeError("Saved source changed.")
        history = call(prefix + "/console")["history"]
        runs = [r for r in history if r.get("status") == "ok" and r.get("code") == code and MARKER in r.get("stdout", "")]
        if len(runs) != 1:
            raise RuntimeError("Require one completed actual native Run with the exact source.")
        run = runs[0]
        proof = json.loads(next(line[len(MARKER):] for line in run["stdout"].splitlines() if line.startswith(MARKER)))
        if proof["stages"] != 4 or not proof["frozen"] or not proof["restored_equal"] or proof["source_oracles_available"]:
            raise RuntimeError("Invalid native execution proof.")
        if len(run["outputs"]) != 4 or not all(output.get("latex") for output in run["outputs"]):
            raise RuntimeError("Four complete rendered/exportable native results are required.")
        path = DATA / "projects" / project / STATE
        if path.is_symlink() or digest(path) != proof["states_sha256"]:
            raise RuntimeError("Persisted result state changed.")
        states = json.loads(path.read_text())
        if len(states["model"]["covariance_matrix"]) != 3 or len(states["forecast"]["attrs"]["covariance_matrix"]) != 8:
            raise RuntimeError("Full model/forecast covariance was not preserved.")
        execution_hash = hashlib.sha256(json.dumps(run, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if args.after_restart:
            native.require_pids_exited(record["runtime_pids"])
            if record["execution_sha256"] != execution_hash:
                raise RuntimeError("History/output/events changed after full native restart.")
        record.update(status="native-restarted" if args.after_restart else "native-verified",
                      execution_id=run["id"], execution_sha256=execution_hash, proof=proof,
                      outputs=[{"type": output["type"], "latex_sha256": hashlib.sha256(output["latex"].encode()).hexdigest()} for output in run["outputs"]],
                      states_sha256=digest(path), current_runtime_pids=pids)
        if not args.after_restart:
            record["runtime_pids"] = pids
        args.receipt.with_name("native-execution.json").write_text(json.dumps(run, indent=2))
        args.receipt.with_name("persisted-states.json").write_text(json.dumps(states, indent=2))
        for i, output in enumerate(run["outputs"], 1):
            args.receipt.with_name(f"output-{i}.tex").write_text(output["latex"])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2))
    print(json.dumps({"status": record["status"], "stages": 4, "runtime_sha256": identity["runtime_sha256"]}))


if __name__ == "__main__":
    main()
