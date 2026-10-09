"""Verify owned frozen dependent-meta execution, full artifacts and process restart.

This starts a temporary loopback console. Actual installed native Run and
application quit/relaunch require a separate observed acceptance receipt.
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
import tempfile
import time
from types import CodeType
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("openecon", "openecon.analysis", "openecon.analysis_contracts", "openecon.frame",
           "openecon.econometrics.registry", "openecon.econometrics.core",
           "openecon.econometrics.resident_cpu", "openecon.econometrics.meta",
           "openecon.econometrics.meta.common", "openecon.econometrics.meta.dependent",
           "openecon.econometrics.meta.dependent_kernels", "openecon.econometrics.meta.dependent_post",
           "openecon.engines.distributions", "openecon.resources", "openecon_charts.latex")
RESULT_NAMES = ("common", "effect", "study", "robust", "contrast", "mean", "latent", "diagnostics")
PROCEDURES = ("meta_dependent", "meta_dependent", "meta_dependent", "meta_dependent_robust",
              "meta_dependent_contrast", "meta_dependent_predict", "meta_dependent_predict_effect",
              "meta_dependent_diagnostics")
MARKER = "DEPENDENT_META_ACCEPTANCE_OK "


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(code):
    return code.replace(co_filename="<verified-source>", co_consts=tuple(
        normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts))


def source_identity(runtime):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for name in MODULES:
        directory = "packages/openecon-charts/src" if name.startswith("openecon_charts") else "src"
        source = ROOT/directory/Path(*name.split("."))
        source = source/"__init__.py" if source.is_dir() else source.with_suffix(".py")
        expected = compile(source.read_text(), str(source), "exec", dont_inherit=True)
        if normalized(archive.extract(name)) != normalized(expected):
            raise RuntimeError(f"Compiled module differs from source: {name}")
        sources[name] = digest(source)
    return sources


def receipt(stdout):
    records = [json.loads(line[len(MARKER):]) for line in stdout.splitlines() if line.startswith(MARKER)]
    if len(records) != 1:
        raise RuntimeError("Missing unique complete dependent-meta example receipt")
    return records[0]


def header(directory):
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n"
            f"DEPENDENT_META_RESULT_DIRECTORY = {str(directory)!r}\n")


def restore_code(directory, hashes):
    return f"""import hashlib, json
from pathlib import Path
import openecon as oe
from openecon.econometrics.core import TableSet
from openecon.econometrics.meta.dependent import load_state
assert getattr(__import__('sys'), 'frozen', False)
directory = Path({str(directory)!r})
expected = {hashes!r}
for name, sha in expected.items():
    raw = (directory/(name+'.json')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    payload = json.loads(raw)
    frames = {{key: oe.DataFrame(**payload['tables'][key]) for key in payload['table_order']}}
    for key in frames:
        for column, dtype in zip(frames[key].columns, payload['table_dtypes'][key]):
            frames[key][column] = frames[key][column].astype(dtype)
            if dtype == 'object':
                frames[key][column] = frames[key][column].where(frames[key][column].notna(), None)
        frames[key].attrs.update(payload['table_attrs'][key])
    restored = TableSet(frames, title=payload['title'], **payload['attrs'])
    assert restored.to_latex() == payload['latex']
    assert {{key: frame.to_dict(orient='split') for key, frame in restored.items()}} == payload['tables']
    load_state(restored.attrs['prediction_state'])
print('DEPENDENT_META_FULL_RESULTS_REOPENED')
"""


def verify(runtime):
    runtime = Path(runtime).resolve(strict=True)
    fingerprint = digest(runtime)
    sources = source_identity(runtime)
    example = ROOT/"docs/examples/dependent_meta_eight.py"
    with tempfile.TemporaryDirectory(prefix="openecon-dependent-meta-") as temporary:
        root = Path(temporary)
        expected_directory = root/"source-artifacts"
        source_environment = os.environ.copy()
        source_environment["PYTHONPATH"] = str(ROOT/"src")+os.pathsep+str(ROOT/"packages/openecon-charts/src")
        source_code = ("import runpy\n"+f"runpy.run_path({str(example)!r}, init_globals="
                       +repr({"DEPENDENT_META_RESULT_DIRECTORY": str(expected_directory)})
                       +" | {'display': lambda value: None})\n")
        source_run = subprocess.run([sys.executable, "-c", source_code], cwd=ROOT, env=source_environment,
                                    capture_output=True, text=True, timeout=180, check=True)
        source_proof = receipt(source_run.stdout)
        if source_proof["frozen"]:
            raise RuntimeError("Expected independent source execution before frozen comparison")
        expected = {name: digest(expected_directory/(name+".json")) for name in RESULT_NAMES}
        if expected != source_proof["artifact_sha256"]:
            raise RuntimeError("Complete source artifact files do not match their receipt")
        data = root/"frozen-profile"
        data.mkdir()
        destination = data/"complete-results"
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        process = descriptor = token = None
        with (root/"runtime-errors.log").open("ab") as errors:
            def call(path, body=None):
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["X-OpenEcon-Token"] = token
                request = Request(descriptor["url"]+prefix+path, headers=headers,
                                  data=json.dumps(body).encode() if body is not None else None)
                with urlopen(request, timeout=180) as response:
                    return json.load(response)

            def start():
                nonlocal process, descriptor, token
                process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", str(data)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                           env=environment, start_new_session=True)
                descriptor = token = None
                deadline = time.monotonic()+60
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
                code = header(destination)+example.read_text()
                run = call("/console/execute", {"code": code, "timeout_seconds": 180})
                if run.get("status") != "ok":
                    raise RuntimeError(f"Frozen example failed: {run.get('error')}")
                proof = receipt(run["stdout"])
                outputs = run["outputs"]
                if [output["type"] for output in outputs] != ["table"]*8:
                    raise RuntimeError("The frozen example did not display exactly eight ordered summaries")
                if not proof["frozen"] or proof["procedures"] != list(PROCEDURES) or proof["results"] != list(RESULT_NAMES):
                    raise RuntimeError("Unexpected frozen procedure order")
                if any("\\begin{tabular}" not in output.get("latex", "") for output in outputs):
                    raise RuntimeError("A frozen summary lacks complete LaTeX")
                if "Display limit reached" in run["stdout"]:
                    raise RuntimeError("Summary rendering exceeded the display limit")
                hashes = {name: digest(destination/(name+".json")) for name in RESULT_NAMES}
                if hashes != expected or hashes != proof["artifact_sha256"]:
                    raise RuntimeError("Full frozen result bytes differ from independent source execution")
                call("/console/reset", {})
                reopened = call("/console/execute", {"code": restore_code(destination, hashes), "timeout_seconds": 120})
                if reopened.get("status") != "ok" or "DEPENDENT_META_FULL_RESULTS_REOPENED" not in reopened["stdout"]:
                    raise RuntimeError("Complete results failed restoration after worker reset")
                saved = next(row for row in call("/console")["history"] if row["id"] == run["id"])
                if any(saved[key] != run[key] for key in ("code", "outputs", "events")):
                    raise RuntimeError("Saved execution changed after worker reset")
                stop()
                start()
                saved = next(row for row in call("/console")["history"] if row["id"] == run["id"])
                if any(saved[key] != run[key] for key in ("code", "outputs", "events")):
                    raise RuntimeError("Saved execution changed after full runtime restart")
                reopened = call("/console/execute", {"code": restore_code(destination, hashes), "timeout_seconds": 120})
                if reopened.get("status") != "ok" or "DEPENDENT_META_FULL_RESULTS_REOPENED" not in reopened["stdout"]:
                    raise RuntimeError("Complete results failed restoration after runtime restart")
                if digest(runtime) != fingerprint or hashes != {name: digest(destination/(name+".json")) for name in RESULT_NAMES}:
                    raise RuntimeError("Runtime or full artifact bytes changed during verification")
                record = {"status": "passed", "frozen_execution": True,
                          "compiled_modules_equal_source": sources,
                          "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                          "runtime_sha256": fingerprint, "example_sha256": digest(example), "verifier_sha256": digest(__file__),
                          "source_proof": source_proof, "proof": proof, "complete_result_hashes": hashes,
                          "complete_source_frozen_artifacts_equal": True,
                          "worker_reset_readback": True, "runtime_restart_readback": True,
                          "saved_code_outputs_events_equal": True,
                          "ordered_output_types": ["table"]*8,
                          "table_rows": [len(output["data"]["rows"]) for output in outputs],
                          "console_duration_ms": run["duration_ms"],
                          "external_oracle_packages_absent": True, "source_path_injected": False,
                          "native_window_verified": False, "installation_changed": False,
                          "human_data_access": False, "public_release_delivered": False, "cuda_verified": False}
            finally:
                stop()
            record["owned_runtime_stopped"] = process.poll() is not None
    record["owned_temporary_data_removed"] = not root.exists()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path")
    record = verify(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: record[key] for key in ("status", "frozen_execution", "table_rows")}))


if __name__ == "__main__":
    main()
