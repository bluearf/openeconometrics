"""Seed/read back source-pinned, isolated native survey-regression-replication acceptance.

This helper never submits a console execution request. Seed the dedicated
synthetic project, click its actual native Run, read back, then quit/relaunch
the QA app and use --after-restart to prove retained history and all eight coefficient/helper states.
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
IDENTIFIER = "org.openecon.qa.survey-regression-replication-eight-20261007"
PRODUCT = "OpenEconometrics Survey Replicate QA"
APP = Path.home() / "Applications" / f"{PRODUCT}.app"
DATA = Path.home() / "Library/Application Support" / IDENTIFIER
PROJECT = "Survey Replicate QA"
SCRIPT = "survey_regression_replication_eight.py"
STATE = "survey-regression-replication-states.json"
POSTSTATE = "survey-regression-replication-postestimation.json"
MARKER = "SURVEY_REGRESSION_REPLICATION_INSTALLED:"
MODULES = ['openecon', 'openecon.models', 'openecon.console_worker', 'openecon.survey', 'openecon.econometrics.survey', 'openecon.econometrics.survey.common', 'openecon.econometrics.survey.targets', 'openecon.econometrics.survey.replication', 'openecon.econometrics.survey.regression_common', 'openecon.econometrics.survey.regression', 'openecon.econometrics.survey.regression_replication', 'openecon.econometrics.survey.regression_postest', 'openecon.econometrics.registry', 'openecon.analysis_contracts', 'openecon.econometrics.core', 'openecon.resources', 'openecon.engines.inference', 'openecon.engines.distributions', 'openecon.engines.separation', 'openecon.engines.contracts', 'openecon.engines']


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def source_parity(runtime, source_ref):
    """Compare pinned compiled module code, normalizing filenames only."""
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")

    def normalized(code):
        return code.replace(
            co_filename="<pinned>",
            co_consts=tuple(normalized(value) if isinstance(value, CodeType) else value
                           for value in code.co_consts),
        )

    hashes = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = subprocess.check_output(
            ["git", "show", f"{source_ref}:{path.relative_to(ROOT).as_posix()}"],
            cwd=ROOT, text=True,
        )
        if normalized(archive.extract(name)) != normalized(
                compile(source, str(path), "exec", dont_inherit=True)):
            raise RuntimeError("Frozen source mismatch: " + name)
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    return hashes


def code_for(project):
    workspace = DATA / "projects" / project
    example = (ROOT / "docs/examples/survey_regression_replication_eight.py").read_text()
    header = f"""import importlib, importlib.util, os, sys
from pathlib import Path
assert getattr(sys, "frozen", False), "Not the frozen runtime"
assert importlib.util.find_spec("scipy") is None
assert importlib.util.find_spec("statsmodels") is None
_frozen_root = Path(sys._MEIPASS).resolve()
_workspace = Path({str(workspace)!r}).resolve()
assert Path.cwd().resolve() == _workspace
_modules = {{}}
for _name in {MODULES!r}:
    _module = importlib.import_module(_name)
    _file = Path(_module.__file__).resolve()
    assert _file.is_relative_to(_frozen_root), _name
    _modules[_name] = str(_file)
"""
    footer = f"""
assert len(states) == 8 and set(states) == set(poststates)
for _name, _state in states.items():
    _restored = oe.SurveyRegressionResult.model_validate(_state)
    assert _restored.model_dump(mode="json") == _state
    assert _restored.family == _name.split("_")[0]
    assert _restored.metadata["replication"]["method"] == _name.split("_")[1]
    assert _restored.metadata["replication"]["failed_replicates"] == []
    assert _restored.to_frame().attrs["covariance_matrix"] == _state["covariance"]
_file_proofs = {{}}
for _key, _name, _value in [("states", {STATE!r}, states), ("postestimation", {POSTSTATE!r}, poststates), ("oracle_inputs", "survey-replica-oracle-inputs.json", oracle_inputs)]:
    _bytes = json.dumps(_value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    assert len(_bytes) < 1024 * 1024, "Synthetic acceptance state exceeded its bounded serialization"
    _path = _workspace / _name
    _temporary = _workspace / (_name + ".tmp")
    assert not _path.is_symlink() and not _temporary.is_symlink()
    _temporary.write_bytes(_bytes)
    _temporary.replace(_path)
    assert json.loads(_path.read_text()) == _value
    _file_proofs[_key] = {{"file": _name, "sha256": __import__("hashlib").sha256(_bytes).hexdigest(), "bytes": len(_bytes)}}
print({MARKER!r} + json.dumps({{"frozen": True, "root": str(_frozen_root), "modules": _modules,
    "workspace": str(_workspace), "worker_pid": os.getpid(), "stages": 8, "rows": 48,
    "files": _file_proofs, "source_oracles_available": False, "restored_equal": True}}, sort_keys=True))
"""
    return header + example + footer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--after-restart", action="store_true")
    parser.add_argument("--source-ref", required=True,
                        help="Explicit committed frozen-source pin; no fallback to another revision")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.seed and args.after_restart:
        raise RuntimeError("Seed and restart readback are separate acceptance stages.")
    if args.seed and args.receipt.exists():
        raise RuntimeError("Refuse to overwrite a prior acceptance seed.")
    source_ref = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.source_ref + "^{commit}"],
        cwd=ROOT, text=True,
    ).strip()
    native.require_unaliased_directory(APP)
    info = plistlib.loads((APP / "Contents/Info.plist").read_bytes())
    if info["CFBundleIdentifier"] != IDENTIFIER:
        raise RuntimeError("Only the dedicated synthetic survey-regression-replication QA app is allowed.")
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(APP)], check=True)
    signature = subprocess.run(["codesign", "-dv", str(APP)], check=True,
                               text=True, capture_output=True).stderr
    if "Signature=adhoc" not in signature:
        raise RuntimeError("Require the separately ad-hoc signed QA bundle.")
    runtime = APP / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    manifest = APP / "Contents/Resources/runtime/runtime-manifest.json"
    identity = {
        "app": str(APP), "identifier": IDENTIFIER,
        "version": info["CFBundleShortVersionString"],
        "minimum_system_version": info["LSMinimumSystemVersion"],
        "source_ref": source_ref,
        "runtime_sha256": digest(runtime),
        "native_executable_sha256": digest(APP / "Contents/MacOS" / info["CFBundleExecutable"]),
        "runtime_manifest_sha256": digest(manifest),
        "source_parity": source_parity(runtime, source_ref),
        "signature": "ad-hoc local QA; no Developer ID/notarization/public release claim",
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
        call("/api/desktop/local-projects", {
            "name": PROJECT,
            "description": "Synthetic local-only single-stage survey coefficient replication and saved inference acceptance.",
        })
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
        script = call(prefix + "/console/scripts", {"name": SCRIPT, "code": code})
        if call(prefix + "/console/scripts/" + script["id"])["code"] != code:
            raise RuntimeError("Saved script readback differs.")
        record = {
            "status": "seeded", "identity": identity,
            "project_id": project, "script_id": script["id"],
            "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "runtime_pids": pids,
        }
    else:
        record = json.loads(args.receipt.read_text())
        if record["identity"] != identity or record["project_id"] != project:
            raise RuntimeError("Native identity/source changed.")
        if hashlib.sha256(code.encode()).hexdigest() != record["source_sha256"]:
            raise RuntimeError("Acceptance source changed from the seeded script.")
        if call(prefix + "/console/scripts/" + record["script_id"])["code"] != code:
            raise RuntimeError("Saved source changed.")
        history = call(prefix + "/console")["history"]
        runs = [run for run in history if run.get("status") == "ok"
                and run.get("code") == code and MARKER in run.get("stdout", "")]
        if len(runs) != 1:
            raise RuntimeError("Require one completed actual native Run with the exact seeded source.")
        run = runs[0]
        marker_lines = [line[len(MARKER):] for line in run["stdout"].splitlines()
                        if line.startswith(MARKER)]
        if len(marker_lines) != 1:
            raise RuntimeError("Native execution must retain one complete installed proof.")
        proof = json.loads(marker_lines[0])
        if (proof["stages"] != 8 or proof["rows"] != 48
                or proof["frozen"] is not True or proof["restored_equal"] is not True
                or proof["source_oracles_available"] is not False
                or set(proof["modules"]) != set(MODULES)
                or proof["workspace"] != str(DATA / "projects" / project)):
            raise RuntimeError("Invalid native execution proof.")
        outputs = run["outputs"]
        if len(outputs) != 8 or not all(output.get("type") == "table"
                                      and output.get("latex") and output["data"].get("rows")
                                      for output in outputs):
            raise RuntimeError("Eight complete rendered/exportable native results are required.")
        paths = {"states": DATA / "projects" / project / STATE,
                 "postestimation": DATA / "projects" / project / POSTSTATE,
                 "oracle_inputs": DATA / "projects" / project / "survey-replica-oracle-inputs.json"}
        files = {}
        for key, path in paths.items():
            if path.is_symlink() or digest(path) != proof["files"][key]["sha256"]:
                raise RuntimeError("Persisted acceptance state changed: " + key)
            if path.stat().st_size != proof["files"][key]["bytes"]:
                raise RuntimeError("Persisted acceptance state size changed: " + key)
            files[key] = {"sha256": digest(path), "bytes": path.stat().st_size}
        states = json.loads(paths["states"].read_text())
        poststates = json.loads(paths["postestimation"].read_text())
        expected = {f"{family}_{method}" for family in ("linear", "logit")
                    for method in ("brr", "fay", "jackknife", "bootstrap")}
        if set(states) != expected or set(poststates) != expected:
            raise RuntimeError("Eight complete coefficient fit/helper states were not preserved.")
        for name, state in states.items():
            if (state["family"] != name.split("_")[0]
                    or state["metadata"]["replication"]["method"] != name.split("_")[1]
                    or state["metadata"]["replication"]["failed_replicates"] != []
                    or state["metadata"]["n_design"] != 48
                    or len(state["covariance"]) != len(state["coefficients"])):
                raise RuntimeError("Complete coefficient covariance/sample state missing: " + name)
        inputs = DATA / "projects" / project / "survey-replica-oracle-inputs.json"
        if inputs.is_symlink():
            raise RuntimeError("Oracle fixture path is aliased.")
        args.receipt.with_name("oracle-inputs.json").write_bytes(inputs.read_bytes())
        args.receipt.with_name("oracle-payload.json").write_text(json.dumps({"states": states, "poststates": poststates, "oracle_inputs": json.loads(inputs.read_text())}, allow_nan=False))
        execution_hash, history_hash = object_digest(run), object_digest(history)
        if args.after_restart:
            if record.get("status") != "native-verified":
                raise RuntimeError("Restart proof requires a preceding actual native verification.")
            native.require_pids_exited(record["runtime_pids"])
            if (record["execution_sha256"] != execution_hash
                    or record["history_sha256"] != history_hash or record["files"] != files):
                raise RuntimeError("History/output/events or saved states changed after full native restart.")
        record.update(
            status="native-restarted" if args.after_restart else "native-verified",
            execution_id=run["id"], execution_sha256=execution_hash,
            history_sha256=history_hash, proof=proof, files=files,
            outputs=[{"type": output["type"],
                      "latex_sha256": hashlib.sha256(output["latex"].encode()).hexdigest()}
                     for output in outputs],
            current_runtime_pids=pids,
        )
        if not args.after_restart:
            record["runtime_pids"] = pids
        args.receipt.with_name("native-execution.json").write_text(json.dumps(run, indent=2))
        args.receipt.with_name("persisted-states.json").write_text(json.dumps(states, indent=2))
        args.receipt.with_name("postestimation-states.json").write_text(json.dumps(poststates, indent=2))
        for index, output in enumerate(outputs, 1):
            args.receipt.with_name(f"output-{index}.tex").write_text(output["latex"])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2))
    print(json.dumps({"status": record["status"], "stages": 8,
                      "runtime_sha256": identity["runtime_sha256"]}))


if __name__ == "__main__":
    main()
