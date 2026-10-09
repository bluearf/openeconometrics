"""Owned frozen acceptance, compiled identity and saved full-result restart.

Native Run/UI proof is separate. No human project or system Python is used by
the runtime. The caller must choose a fresh output path.
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
import tempfile
import time
from types import CodeType
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODULES = tuple("openecon.econometrics.multivariate" + ("." + name if name else "")
                for name in ("", "pca", "factor", "extraction", "rotation", "summary", "weighted", "supplementary", "scaling", "common", "replay")) + ("openecon.analysis_contracts",)
CASES = ("frequency_pca", "summary_pca", "summary_factor", "minres", "anderson_rubin", "target", "geomin", "supplementary")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(code):
    return code.replace(co_filename="<verified-source>", co_consts=tuple(
        normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts))


def source_identity(runtime, source_ref=None):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = (subprocess.check_output(["git", "show", f"{source_ref}:{path.relative_to(ROOT)}"], cwd=ROOT)
                  if source_ref else path.read_bytes())
        assert normalized(archive.extract(name)) == normalized(compile(source, str(path), "exec", dont_inherit=True)), name
        sources[name] = hashlib.sha256(source).hexdigest()
    return sources


def verify_outputs(run):
    assert run["status"] == "ok", run.get("error")
    proof = json.loads(next(line.split("MULTIVARIATE_OPTIONS_OK ", 1)[1]
                            for line in run["stdout"].splitlines() if line.startswith("MULTIVARIATE_OPTIONS_OK ")))
    assert proof["cases"] == list(CASES) and proof["frozen"]
    assert proof["physical_fixture_rows"] == 1203 and proof["weight_total"] == 2403
    assert proof["complete_score_rows"] == {name: 1203 for name in CASES[:-1]}
    assert len(run["outputs"]) == 8
    assert [item["type"] for item in run["outputs"]] == ["table"] * 8
    assert [len(item["data"]["rows"]) for item in run["outputs"]] == [6]*7+[2]
    assert all("\\begin{tabular}" in item["latex"] for item in run["outputs"])
    assert "Display limit reached" not in run["stdout"]
    return proof


def frozen_header():
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n")


def reopen_code(directory):
    return frozen_header() + f"""
import json
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.core import TableSet
root = Path({str(directory)!r})
fixture = json.loads((root/'fixture.json').read_text())
frame = pd.DataFrame(**fixture)
for name in {CASES[:-1]!r}:
    payload = json.loads((root/(name+'.json')).read_text())
    result = TableSet({{key: pd.DataFrame(**value) for key, value in payload['tables'].items()}}, **payload['attrs'])
    scorer = oe.pca_scores if result.attrs['procedure'] == 'pca' else oe.factor_scores
    score = scorer(result, frame)
    expected = pd.DataFrame(**payload['scores'])
    assert len(score) == 1203 and float(torch.as_tensor(score.to_numpy()-expected.to_numpy()).abs().max()) < 1e-10
    assert '\\\\begin{{tabular}}' in payload['latex']
payload = json.loads((root/'active_ca.json').read_text())
active = TableSet({{key: pd.DataFrame(**value) for key, value in payload['tables'].items()}}, **payload['attrs'])
row = oe.ca_project(active, {{'c0': [35], 'c1': [12], 'c2': [3]}})
assert abs(float(row.dim1.iloc[0])-float(active['rows'].dim1.iloc[0])) < 1e-10
print('MULTIVARIATE_FULL_RESULTS_REOPENED')
"""


def verify(runtime, result_directory):
    runtime = runtime.resolve(strict=True)
    fingerprint, sources = digest(runtime), source_identity(runtime)
    result_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    with tempfile.TemporaryDirectory(prefix="openecon-multivariate-frozen-") as temporary:
        data = Path(temporary)
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        process = descriptor = token = None
        errors = (data / "errors.log").open("ab")

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
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, start_new_session=True, env=environment)
            descriptor = token = None
            deadline = time.monotonic()+60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], .25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                assert process.poll() is None, "Owned runtime exited before readiness"
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
            code = frozen_header() + f"MULTIVARIATE_RESULT_DIRECTORY = {str(result_directory)!r}\n" + (ROOT / "docs/examples/multivariate_options.py").read_text()
            run = call("/console/execute", {"code": code, "timeout_seconds": 180})
            proof = verify_outputs(run)
            hashes = {name: digest(result_directory/(name+".json")) for name in CASES}
            assert hashes == proof["hashes"]
            call("/console/reset", {})
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            stop()
            start()
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            reopen = call("/console/execute", {"code": reopen_code(result_directory), "timeout_seconds": 180})
            assert reopen["status"] == "ok", reopen.get("error")
            assert "MULTIVARIATE_FULL_RESULTS_REOPENED" in reopen["stdout"]
            assert hashes == {name: digest(result_directory/(name+".json")) for name in CASES}
            assert digest(runtime) == fingerprint
            receipt = {"status": "passed", "frozen": True, "source_identity": sources,
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "runtime_sha256": fingerprint, "proof": proof, "duration_ms": run["duration_ms"],
                "example_sha256": digest(ROOT / "docs/examples/multivariate_options.py"), "verifier_sha256": digest(__file__),
                "worker_reset_readback": True, "runtime_restart_readback": True, "full_results_reopened": True,
                "external_oracle_packages_absent": True, "native_ui_verified": False, "public_release_delivered": False}
        finally:
            stop()
            errors.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    receipt = verify(args.runtime, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: receipt[key] for key in ("status", "frozen", "full_results_reopened", "duration_ms")}))
