"""Owned frozen console, source identity, full results and restart acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time
from types import CodeType
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("openecon.econometrics.var.frequency_causality",
           "openecon.econometrics.var.panel_causality", "openecon.econometrics.var",
           "openecon.econometrics.var.toda_yamamoto", "openecon.econometrics.var.kernels")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(code):
    return code.replace(co_filename="<verified-source>", co_consts=tuple(
        normalized(c) if isinstance(c, CodeType) else c for c in code.co_consts))


def source_identity(runtime):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    hashes = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        assert normalized(archive.extract(name)) == normalized(compile(path.read_text(), str(path), "exec", dont_inherit=True)), name
        hashes[name] = digest(path)
    return hashes


def header(directory):
    return ("import sys, importlib.util, importlib\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n"
            f"CAUSALITY_RESULT_DIRECTORY = {str(directory)!r}\n")


def verify(runtime):
    runtime = runtime.resolve(strict=True)
    fingerprint = digest(runtime)
    sources = source_identity(runtime)
    example = ROOT / "docs/examples/frequency_panel_causality.py"
    with tempfile.TemporaryDirectory(prefix="openecon-causality-frozen-") as temporary:
        data = Path(temporary)
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        result_directory = data / "complete-results"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        process = descriptor = token = None

        def call(path, body=None):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-OpenEcon-Token"] = token
            request = Request(descriptor["url"] + prefix + path, headers=headers,
                              data=json.dumps(body).encode() if body is not None else None)
            with urlopen(request, timeout=180) as response:
                return json.load(response)

        def start():
            nonlocal process, descriptor, token
            process = subprocess.Popen([str(runtime), "--port", "0", "--data-root", str(data)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=(data / "runtime-errors.log").open("ab"),
                                       start_new_session=True, env=environment)
            descriptor = token = None
            deadline = time.monotonic()+60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], .25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                assert process.poll() is None, "Owned runtime exited before ready"
            assert descriptor and descriptor["type"] == "ready"
            assert descriptor["url"] == f"http://127.0.0.1:{descriptor['port']}"
            token = call("/session")["token"]

        def stop():
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)

        try:
            start()
            code = header(result_directory) + example.read_text()
            run = call("/console/execute", {"code": code, "timeout_seconds": 180})
            assert run["status"] == "ok", run.get("error")
            assert [o["type"] for o in run["outputs"]] == ["table"]*6
            assert all(any("\\begin{" + environment + "}" in o.get("latex", "")
                           for environment in ("tabular", "longtable")) for o in run["outputs"])
            proof = json.loads(next(line.split("CAUSALITY_ACCEPTANCE_OK ", 1)[1]
                                    for line in run["stdout"].splitlines()
                                    if line.startswith("CAUSALITY_ACCEPTANCE_OK ")))
            hashes = {name: digest(result_directory/(name+".json")) for name in ("frequency", "panel")}
            call("/console/reset", {})
            saved = next(row for row in call("/console")["history"] if row["id"] == run["id"])
            assert saved["code"] == run["code"] and saved["outputs"] == run["outputs"] and saved["events"] == run["events"]
            stop()
            start()
            saved = next(row for row in call("/console")["history"] if row["id"] == run["id"])
            assert saved["outputs"] == run["outputs"] and saved["events"] == run["events"]
            reopened = call("/console/execute", {"code": f"""import json
from pathlib import Path
import openecon as oe
from openecon.econometrics.core import TableSet
root = Path({str(result_directory)!r})
for name, counts in [('frequency', {{'tests': 3, 'coefficients': 14, 'restrictions': 9}}),
                     ('panel', {{'tests': 3, 'individual': 12, 'coefficients': 60}})]:
    payload = json.loads((root/(name+'.json')).read_text())
    frames = {{key: oe.DataFrame(**value) for key, value in payload['tables'].items()}}
    restored = TableSet(frames, **payload['attrs'])
    assert {{key: len(frame) for key, frame in frames.items()}} == counts
    assert restored.attrs['cause'] == 'investment' and restored.attrs['dtype'] == 'float64'
    assert len(restored.to_latex()) > 1000
print('CAUSALITY_FULL_RESULTS_REOPENED')
""", "timeout_seconds": 120})
            assert reopened["status"] == "ok", reopened.get("error")
            assert "CAUSALITY_FULL_RESULTS_REOPENED" in reopened["stdout"]
            assert hashes == {name: digest(result_directory/(name+".json")) for name in hashes}
            assert digest(runtime) == fingerprint
            receipt = {"status": "passed", "frozen": True, "source_identity": sources,
                       "runtime_sha256": fingerprint, "example_sha256": digest(example),
                       "verifier_sha256": digest(__file__), "proof": proof,
                       "ordered_table_outputs": 6, "publication_latex": True,
                       "worker_reset_readback": True, "runtime_restart_readback": True,
                       "full_results_reopened": True, "complete_result_hashes": hashes,
                       "external_oracle_packages_absent": True, "installed_ui_verified": False,
                       "human_projects_accessed": False, "release_delivered": False,
                       "console_duration_ms": run["duration_ms"]}
        finally:
            stop()
        receipt["owned_runtime_stopped"] = process.poll() is not None
    receipt["owned_temporary_profile_removed"] = not data.exists()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output)}))
