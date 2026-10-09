"""Verify eight complete repeated-GLS artifacts at a declared execution layer.

Source, isolated installation and owned frozen console are separate modes.
Actual installed native Run and Quit/relaunch need their own observed receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from types import CodeType
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/repeated_residual_eight.py"
MODULES = (
    "openecon", "openecon.analysis", "openecon.analysis_contracts", "openecon.models",
    "openecon.dataset", "openecon.frame", "openecon.resources", "openecon.econometrics",
    "openecon.econometrics.registry", "openecon.econometrics.core",
    "openecon.econometrics.resident_cpu", "openecon.econometrics.mixed",
    "openecon.econometrics.mixed.extended_common", "openecon.econometrics.mixed.repeated",
    "openecon.econometrics.mixed.repeated_kernels", "openecon.econometrics.meta",
    "openecon.econometrics.meta.common", "openecon.econometrics.meta.dependent",
    "openecon.econometrics.meta.dependent_kernels", "openecon.econometrics.meta.dependent_post",
    "openecon.engines", "openecon.engines.contracts", "openecon.engines.optimize",
    "openecon.engines.linalg", "openecon.engines.distributions",
    "openecon_charts", "openecon_charts.latex",
)
RESULT_NAMES = tuple(f"{structure}_{method}" for structure in
                     ("cs", "ar1", "diagonal", "unstructured") for method in ("ml", "reml"))
POST_NAMES = tuple(f"{name}-{operation}" for name in RESULT_NAMES for operation in ("mean", "contrast"))
PROCEDURES = ("repeated_gls",) * 8
MARKER = "REPEATED_GLS_ACCEPTANCE_OK "
IDENTITY_MARKER = "REPEATED_GLS_MODULE_IDENTITY "
RESTORE_MARKER = "REPEATED_GLS_FULL_RESULTS_REOPENED"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_path(name):
    directory = "packages/openecon-charts/src" if name.startswith("openecon_charts") else "src"
    source = ROOT / directory / Path(*name.split("."))
    return source / "__init__.py" if source.is_dir() else source.with_suffix(".py")


def source_hashes():
    return {name: digest(source_path(name)) for name in MODULES}


def normalized(code):
    return code.replace(co_filename="<verified-source>", co_consts=tuple(
        normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts))


def source_identity(runtime):
    """Compare whole compiled modules, normalizing only source filename paths."""
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    for name in MODULES:
        source = source_path(name)
        expected = compile(source.read_text(), str(source), "exec", dont_inherit=True)
        if normalized(archive.extract(name)) != normalized(expected):
            raise RuntimeError(f"Compiled module differs from source: {name}")
    return source_hashes()


def receipt(stdout, marker=MARKER):
    records = [json.loads(line[len(marker):]) for line in stdout.splitlines() if line.startswith(marker)]
    if len(records) != 1:
        raise RuntimeError(f"Missing unique complete receipt: {marker.strip()}")
    return records[0]


def artifact_hashes(directory):
    return {name: digest(Path(directory) / (name + ".json")) for name in (*RESULT_NAMES, *POST_NAMES)}


def check_proof(directory, proof):
    if (proof["results"] != list(RESULT_NAMES) or proof["procedures"] != list(PROCEDURES)
            or proof["issues"] != [f"MARKET-{number}" for number in range(648, 656)]):
        raise RuntimeError("Unexpected eight-stage procedure or issue order")
    for flag in ("complete_artifacts_equal", "state_restore_equal", "saved_operations_equal", "all_latex"):
        if proof.get(flag) is not True:
            raise RuntimeError(f"Missing complete saved-result check: {flag}")
    hashes = artifact_hashes(directory)
    if hashes != {**proof["artifact_sha256"], **proof["post_artifact_sha256"]}:
        raise RuntimeError("Complete artifact file bytes do not match the example receipt")
    for name in RESULT_NAMES:
        saved = json.loads((Path(directory) / (name + ".json")).read_bytes())
        if saved["attrs"]["state"]["schema"] != "openecon.repeated.gls.v1":
            raise RuntimeError("Unexpected retained state schema")
        if "\\begin{tabular}" not in saved["latex"]:
            raise RuntimeError("A complete fit lacks LaTeX")
        if set(saved["table_order"]) != set(saved["tables"]):
            raise RuntimeError("A complete result lost table order or contents")
    return hashes


def header(directory):
    """Reusable native-console header: bundled imports and absent external oracles."""
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n"
            f"REPEATED_GLS_RESULT_DIRECTORY = {str(directory)!r}\n")


def restore_code(directory, hashes, *, frozen=True):
    """Canonical fit and saved operations after process reset, with no optimizer."""
    return f"""import hashlib, json
from pathlib import Path
import pandas as pd
import openecon as oe
from openecon.econometrics.core import TableSet
assert bool(getattr(__import__('sys'), 'frozen', False)) == {frozen!r}
directory = Path({str(directory)!r})
expected = {hashes!r}
def full_payload(result):
    return {{'title': result.title, 'table_order': list(result),
            'tables': {{key: frame.to_dict(orient='split') for key, frame in result.items()}},
            'table_dtypes': {{key: [str(dtype) for dtype in frame.dtypes] for key, frame in result.items()}},
            'table_attrs': {{key: frame.attrs for key, frame in result.items()}},
            'attrs': result.attrs, 'latex': result.to_latex()}}
def full_bytes(result):
    return json.dumps(full_payload(result), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def saved_result(saved):
    frames = {{key: oe.DataFrame(**saved['tables'][key]) for key in saved['table_order']}}
    for key, frame in frames.items():
        for column, dtype in zip(frame.columns, saved['table_dtypes'][key]):
            frame[column] = frame[column].astype(dtype)
            if dtype == 'object':
                frame[column] = frame[column].where(frame[column].notna(), None)
        frame.attrs.update(saved['table_attrs'][key])
    return TableSet(frames, title=saved['title'], **saved['attrs'])
for name, sha in expected.items():
    raw = (directory/(name+'.json')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    assert full_bytes(saved_result(json.loads(raw))) == raw
future = pd.DataFrame({{'x': [-.5, .0, .8], 'z': [.2, -.4, .1]}},
                      index=['profile-a', 'profile-b', 'profile-c'])
targets = pd.DataFrame([[0., 1., 0.], [0., 0., 1.]],
                       columns=['Intercept', 'x', 'z'], index=['x-effect', 'z-effect'])
for name in {RESULT_NAMES!r}:
    raw = (directory/(name+'.json')).read_bytes()
    state = json.loads(raw)['attrs']['state']
    assert full_bytes(oe.restore_repeated_gls(state)) == raw
    assert full_bytes(oe.repeated_gls_predict(state, data=future)) == (directory/(name+'-mean.json')).read_bytes()
    assert full_bytes(oe.repeated_gls_contrast(state, contrast=targets, null=[.1, -.1])) == (directory/(name+'-contrast.json')).read_bytes()
print({RESTORE_MARKER!r})
"""


def execute_source(directory, python, *, installed_site=None, timeout=300):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       PYTHONNOUSERSITE="1")
    environment.pop("PYTHONPATH", None)
    expected = source_hashes()
    if installed_site is None:
        environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")))
        guard = ""
    else:
        installed_site = Path(installed_site).resolve(strict=True)
        environment["PYTHONPATH"] = str(installed_site)
        guard = ("assert importlib.util.find_spec('scipy') is None\n"
                 "assert importlib.util.find_spec('statsmodels') is None\n"
                 f"for module_name in {MODULES!r}:\n"
                 "    location = Path(importlib.import_module(module_name).__file__).resolve()\n"
                 f"    assert location.is_relative_to(Path({str(installed_site)!r}))\n"
                 f"    assert not location.is_relative_to(Path({str(ROOT)!r})/'src')\n")
    code = ("import runpy, importlib, importlib.util, hashlib, json\nfrom pathlib import Path\n" + guard
            + f"expected_modules = {expected!r}\n"
            + "identities = {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() for name in expected_modules}\n"
            + "assert identities == expected_modules\n"
            + f"print({IDENTITY_MARKER!r}+json.dumps(identities,sort_keys=True))\n"
            + f"runpy.run_path({str(EXAMPLE)!r}, init_globals="
            + repr({"REPEATED_GLS_RESULT_DIRECTORY": str(directory)})
            + " | {'display': lambda value: None})\n")
    run = subprocess.run([str(python), "-c", code], cwd=directory, env=environment,
                         capture_output=True, text=True, timeout=timeout)
    (directory / "execution.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f"Example exited {run.returncode}; inspect {directory / 'execution.log'}")
    proof = receipt(run.stdout)
    if proof["frozen"]:
        raise RuntimeError("Expected a non-frozen source or isolated-installation run")
    hashes = check_proof(directory, proof)
    identities = receipt(run.stdout, IDENTITY_MARKER)
    reopen_imports = "import importlib, importlib.util\nfrom pathlib import Path\n"
    reopen = subprocess.run([str(python), "-c", reopen_imports + guard + restore_code(directory, hashes, frozen=False)],
                            cwd=directory, env=environment, capture_output=True, text=True, timeout=timeout)
    (directory / "reopen.log").write_text(reopen.stdout + reopen.stderr)
    if reopen.returncode or RESTORE_MARKER not in reopen.stdout:
        raise RuntimeError(f"Fresh-process saved restoration failed; inspect {directory / 'reopen.log'}")
    return {"proof": proof, "complete_result_hashes": hashes, "module_bytes_equal_source": identities,
            "fresh_process_full_restoration": True, "installed_site": str(installed_site) if installed_site else None,
            "source_path_injected": installed_site is None,
            "external_oracle_packages_absent": installed_site is not None,
            "python": str(Path(python).resolve())}


def verify_frozen(runtime, directory, python, *, timeout=180):
    runtime = Path(runtime).resolve(strict=True)
    fingerprint = digest(runtime)
    sources = source_identity(runtime)
    source = execute_source(directory / "source-artifacts", python, timeout=max(timeout, 300))
    data = directory / "owned-frozen-profile"
    data.mkdir()
    destination = data / "complete-results"
    prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    process = descriptor = token = None
    with (directory / "runtime-errors.log").open("ab") as errors:
        def call(path, body=None):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-OpenEcon-Token"] = token
            request = Request(descriptor["url"] + prefix + path, headers=headers,
                              data=json.dumps(body).encode() if body is not None else None)
            with urlopen(request, timeout=timeout+30) as response:
                return json.load(response)

        def start():
            nonlocal process, descriptor, token
            process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", str(data)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                       env=environment, start_new_session=True)
            descriptor = token = None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], .25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                if process.poll() is not None:
                    raise RuntimeError("Owned runtime exited before readiness")
            if not descriptor or descriptor.get("type") != "ready":
                raise RuntimeError("Owned runtime did not become ready")
            if descriptor["url"] != f"http://127.0.0.1:{descriptor['port']}":
                raise RuntimeError("Invalid owned loopback descriptor")
            token = call("/session")["token"]

        def stop():
            if process and process.poll() is None:
                try:
                    process.stdin.write(b'{"type":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=12)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=3)
            if process:
                process.stdin.close()
                process.stdout.close()

        try:
            start()
            run = call("/console/execute", {"code": header(destination) + EXAMPLE.read_text(),
                                            "timeout_seconds": timeout})
            (directory / "frozen-execution.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
            if run.get("status") != "ok":
                raise RuntimeError(f"Frozen example failed: {run.get('error')}")
            proof = receipt(run["stdout"])
            outputs = run["outputs"]
            if [output["type"] for output in outputs] != ["table"] * 8:
                raise RuntimeError("Frozen example did not display exactly eight ordered summaries")
            if not proof["frozen"] or any("\\begin{tabular}" not in output.get("latex", "") for output in outputs):
                raise RuntimeError("A frozen summary lacks complete LaTeX")
            if "Display limit reached" in run["stdout"]:
                raise RuntimeError("Summary rendering exceeded the display limit")
            hashes = check_proof(destination, proof)
            if hashes != source["complete_result_hashes"]:
                raise RuntimeError("Complete frozen fit/post bytes differ from fresh source execution")
            call("/console/reset", {})
            reopened = call("/console/execute", {"code": header(destination) + restore_code(destination, hashes),
                                                 "timeout_seconds": timeout})
            if reopened.get("status") != "ok" or RESTORE_MARKER not in reopened["stdout"]:
                raise RuntimeError("Full saved results failed after worker reset")
            saved = next(row for row in call("/console")["history"] if row["id"] == run["id"])
            if any(saved[key] != run[key] for key in ("code", "outputs", "events")):
                raise RuntimeError("Saved execution changed after worker reset")
            stop()
            start()
            saved = next(row for row in call("/console")["history"] if row["id"] == run["id"])
            if any(saved[key] != run[key] for key in ("code", "outputs", "events")):
                raise RuntimeError("Saved execution changed after full runtime restart")
            reopened = call("/console/execute", {"code": header(destination) + restore_code(destination, hashes),
                                                 "timeout_seconds": timeout})
            if reopened.get("status") != "ok" or RESTORE_MARKER not in reopened["stdout"]:
                raise RuntimeError("Full saved results failed after runtime restart")
            if digest(runtime) != fingerprint or hashes != artifact_hashes(destination):
                raise RuntimeError("Runtime or complete artifacts changed during verification")
            record = {"frozen_execution": True, "compiled_modules_equal_source": sources,
                      "runtime_sha256": fingerprint, "source": source, "proof": proof,
                      "complete_result_hashes": hashes, "complete_source_frozen_artifacts_equal": True,
                      "worker_reset_readback": True, "runtime_restart_readback": True,
                      "saved_code_outputs_events_equal": True, "ordered_output_types": ["table"] * 8,
                      "table_rows": [len(output["data"]["rows"]) for output in outputs],
                      "console_duration_ms": run["duration_ms"], "external_oracle_packages_absent": True,
                      "source_path_injected": False, "owned_data_directory": str(data)}
        finally:
            stop()
        record["owned_runtime_stopped"] = process.poll() is not None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "installed", "frozen"), default="source")
    parser.add_argument("--directory", type=Path, required=True, help="New persistent evidence directory")
    parser.add_argument("--output", type=Path, help="New JSON receipt; defaults to DIRECTORY/verification.json")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--site", type=Path, help="Only installed mode: isolated installed package directory")
    parser.add_argument("--runtime", type=Path, help="Only frozen mode: immutable owned runtime executable")
    parser.add_argument("--source-revision", help="Require current HEAD to equal this scientific source revision")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if (args.mode == "installed") != (args.site is not None) or (args.mode == "frozen") != (args.runtime is not None):
        parser.error("--site belongs to installed mode; --runtime belongs to frozen mode")
    if not 30 <= args.timeout <= 600:
        parser.error("Choose a timeout between 30 and 600 seconds")
    directory = args.directory.resolve()
    output = args.output.resolve() if args.output else directory / "verification.json"
    if directory.exists() or output.exists():
        parser.error("Choose a new evidence directory and receipt path")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.source_revision and head != args.source_revision:
        parser.error("Current HEAD does not equal the requested source revision")
    before = source_hashes()
    directory.mkdir(parents=True)
    started = time.monotonic()
    record = {"status": "running", "mode": args.mode, "source_head": head,
              "source_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "source_modules": before, "example_sha256": digest(EXAMPLE), "verifier_sha256": digest(__file__),
              "native_window_verified": False, "installation_changed": False, "human_data_access": False,
              "public_release_delivered": False, "cuda_verified": False}
    try:
        if args.mode == "frozen":
            record.update(verify_frozen(args.runtime, directory, args.python, timeout=args.timeout))
        else:
            record.update(execute_source(directory / "complete-results", args.python,
                                         installed_site=args.site, timeout=max(args.timeout, 300)))
        if source_hashes() != before or digest(EXAMPLE) != record["example_sha256"]:
            raise RuntimeError("Source or editable example changed during verification; run against a stable pin")
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "mode", "source_head", "duration_seconds")}))


if __name__ == "__main__":
    main()
