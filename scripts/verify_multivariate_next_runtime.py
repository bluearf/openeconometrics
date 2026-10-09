"""Frozen identity, full saved uncertainty and moment-hypothesis acceptance.

Native Run and application Quit/reopen are collected separately. This helper
owns only a fresh temporary runtime root and fresh result/receipt paths.
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
from urllib.request import Request, urlopen
from uuid import uuid4

from verify_multivariate_weight_matrix_runtime import digest, finite_json, normalized

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/multivariate_uncertainty_factorial.py"
MARKER = "MULTIVARIATE_NEXT_EIGHT_OK "
CASES = ("covariance_pca", "correlation_pca", "multifactor_pf", "target_pf",
         "frequency_rm", "summary_rm", "frequency_factorial", "summary_factorial")
MODULES = (
    "openecon.econometrics.multivariate", "openecon.econometrics.multivariate.pca_uncertainty",
    "openecon.econometrics.multivariate.factor_uncertainty", "openecon.econometrics.multivariate.pca",
    "openecon.econometrics.multivariate.factor", "openecon.econometrics.multivariate.common",
    "openecon.econometrics.multivariate.uncertainty", "openecon.econometrics.multivariate.summary",
    "openecon.econometrics.multivariate.weighted", "openecon.econometrics.multivariate.rotation",
    "openecon.econometrics.multivariate.rotation_extensions", "openecon.econometrics.multivariate.extraction",
    "openecon.econometrics.multivariate.extraction_extensions", "openecon.econometrics.multivariate.replay",
    "openecon.econometrics.stats", "openecon.econometrics.stats.manova_factorial",
    "openecon.econometrics.stats.rm_moments", "openecon.econometrics.stats.manova_options",
    "openecon.econometrics.stats.manova", "openecon.econometrics.stats.anova",
    "openecon.econometrics.stats.rm_anova", "openecon.econometrics.stats.rm_contrast",
    "openecon.econometrics.stats.common", "openecon.econometrics.stats.glm",
    "openecon.econometrics.summary_state", "openecon.econometrics.core",
    "openecon.econometrics.postest.index_codec", "openecon.econometrics.resident_cpu",
    "openecon.engines.inference", "openecon.engines.linalg", "openecon.engines.distributions",
    "openecon.resources", "openecon.analysis_contracts",
)


def source_identity(runtime, source_ref):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = subprocess.check_output(["git", "show", f"{source_ref}:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert normalized(archive.extract(name)) == normalized(compile(source, str(path), "exec", dont_inherit=True)), name
        sources[name] = hashlib.sha256(source).hexdigest()
    return sources


def verify_outputs(run, *, require_frozen=True):
    assert run["status"] == "ok", run.get("error")
    lines = [line for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(lines) == 1
    proof = json.loads(lines[0][len(MARKER):])
    assert proof["cases"] == list(CASES)
    if require_frozen:
        assert proof["frozen"]
    assert proof["bootstrap_replications"] == 199
    assert proof["fixture_rows"] == 603 and proof["rm_long_rows"] == 2412
    assert proof["full_saved_roundtrip"]
    assert len(run["outputs"]) == 8
    assert all(item["type"] == "table" and "\\begin{tabular}" in item["latex"] for item in run["outputs"])
    assert "Display limit reached" not in run["stdout"]
    return proof


def validate_result_files(directory, proof):
    hashes = {name: digest(directory / (name + ".json")) for name in CASES}
    assert hashes == proof["hashes"]
    for name in CASES:
        payload = json.loads((directory / (name + ".json")).read_text())
        assert set(payload) == {"fit", "post"} and finite_json(payload)
        for state in (payload["fit"], payload["post"]):
            if state is None:
                continue
            assert set(state) == {"summary", "latex", "table_dtypes", "table_order"}
            assert any(token in state["latex"] for token in ("\\begin{tabular}", "\\begin{longtable}"))
            saved = json.loads(state["summary"])
            assert saved["schema"] == "openecon.summary.v1"
            for frame in saved["tables"].values():
                assert len(frame["index"]) == len(frame["data"])
                assert all(len(row) == len(frame["columns"]) for row in frame["data"])
            assert set(state["table_dtypes"]) == set(saved["tables"])
            assert set(state["table_order"]) == set(saved["tables"])
            assert len(state["table_order"]) == len(saved["tables"])
            assert all(len(state["table_dtypes"][key]) == len(frame["columns"])
                       for key,frame in saved["tables"].items())
        tables = json.loads(payload["fit"]["summary"])["tables"]
        assert len(tables) == proof["fit_table_counts"][name]
        if name in CASES[:4]:
            assert len(tables["replicates"]["data"]) == 199
            assert len(tables["covariance"]["data"]) == len(tables["covariance"]["columns"])
    fixtures = json.loads((directory / "fixtures.json").read_text())
    assert finite_json(fixtures)
    return hashes


def validate_displayed_tables(run, directory):
    """These eight primary tables fit the preview; compare every shown cell."""
    for name,item in zip(CASES,run["outputs"],strict=True):
        payload = json.loads((directory/(name+".json")).read_text())
        key = "estimates" if name in CASES[:4] else "within" if name.endswith("rm") else "multivariate"
        frame = json.loads(payload["fit"]["summary"])["tables"][key]
        shown = item["data"]
        assert len(frame["data"]) <= 50 and len(frame["columns"]) <= 30
        assert shown["total_rows"] == len(frame["data"])
        assert shown["total_columns"] == len(frame["columns"])
        assert shown["columns"] == [str(value) for value in frame["columns"]]
        assert shown["rows"] == frame["data"], name
        assert shown["index"] == [[value] for value in frame["index"]]
        assert shown["index_names"] == frame["index_names"]


def frozen_header():
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n")


def reconstruction_code(directory):
    return f"""
import json
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
torch.set_num_threads(2)
root = Path({str(directory)!r})
fixtures = json.loads((root/'fixtures.json').read_text())
def restore_pack(state):
    result = oe.restore_summary(state['summary'])
    assert json.loads(oe.summary_state(result)) == json.loads(state['summary'])
    for key,dtypes in state['table_dtypes'].items():
        result[key] = result[key].astype(dict(zip(result[key].columns,dtypes)))
    result = type(result)({{key:result[key] for key in state['table_order']}},
                          title=result.title,**result.attrs)
    assert json.loads(oe.summary_state(result)) == json.loads(state['summary'])
    assert result.to_latex() == state['latex']
    return result
for name in {CASES!r}:
    payload = json.loads((root/(name+'.json')).read_text())
    result = restore_pack(payload['fit'])
    if name in {CASES[:4]!r}:
        draws = torch.tensor(result['replicates'].to_numpy(dtype='float64'), dtype=torch.float64)
        centered = draws-draws.mean(0)
        expected = centered.T@centered/(len(draws)-1)
        actual = torch.tensor(result['covariance'].to_numpy(dtype='float64'), dtype=torch.float64)
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        estimates = result['estimates']
        se = expected.diagonal().clamp_min(0).sqrt()
        assert torch.allclose(torch.tensor(estimates['std_error'].to_numpy(), dtype=torch.float64), se, rtol=1e-12, atol=1e-12)
        tails = torch.tensor([(1-result.attrs['confidence'])/2,(1+result.attrs['confidence'])/2],dtype=torch.float64)
        intervals = torch.quantile(draws,tails,dim=0,interpolation='linear')
        for column, values in [('ci_lower',intervals[0]),('ci_upper',intervals[1])]:
            assert torch.allclose(torch.tensor(estimates[column].to_numpy(),dtype=torch.float64),values,rtol=1e-12,atol=1e-12)
    else:
        specification = fixtures['contrasts'][name]
        function = oe.rm_mtest if name.endswith('rm') else oe.manova_contrast
        tested = function(result, specification['L'], M=specification['M'], null=specification['null'])
        assert json.loads(oe.summary_state(tested)) == json.loads(payload['post']['summary'])
        assert tested.to_latex() == payload['post']['latex']
        restore_pack(payload['post'])
print('MULTIVARIATE_NEXT_FULL_RESULTS_REOPENED')
"""


def verify(runtime, result_directory, source_ref):
    runtime = runtime.resolve(strict=True)
    fingerprint, sources = digest(runtime), source_identity(runtime, source_ref)
    result_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    with tempfile.TemporaryDirectory(prefix="openecon-next-multivariate-frozen-") as temporary:
        data = Path(temporary)
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        process = descriptor = token = None
        errors = (data / "errors.log").open("ab")

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
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                start_new_session=True, env=environment)
            descriptor = token = None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], .25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                assert process.poll() is None, "Owned runtime exited before readiness"
            assert descriptor and descriptor["type"] == "ready"
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
            code = frozen_header() + f"MULTIVARIATE_NEXT_RESULT_DIRECTORY = {str(result_directory)!r}\n" + EXAMPLE.read_text()
            run = call("/console/execute", {"code": code, "timeout_seconds": 180})
            proof = verify_outputs(run)
            hashes = validate_result_files(result_directory, proof)
            validate_displayed_tables(run,result_directory)
            call("/console/reset", {})
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            stop()
            start()
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            reopened = call("/console/execute", {"code": frozen_header() + reconstruction_code(result_directory), "timeout_seconds": 180})
            assert reopened["status"] == "ok", reopened.get("error")
            assert "MULTIVARIATE_NEXT_FULL_RESULTS_REOPENED" in reopened["stdout"]
            assert hashes == validate_result_files(result_directory, proof)
            assert digest(runtime) == fingerprint
            receipt = {"status": "passed", "frozen": True, "compiled_source_ref": source_ref,
                "source_identity": sources, "runtime_sha256": fingerprint, "proof": proof,
                "duration_ms": run["duration_ms"], "complete_payload_hashes": hashes,
                "example_sha256": digest(EXAMPLE), "verifier_sha256": digest(__file__),
                "worker_reset_readback": True, "runtime_restart_readback": True,
                "full_results_reopened": True, "full_bootstrap_covariance_SE_CI_rederived": 4,
                "primary_display_tables_full_cells_equal": 8,
                "saved_full_hypotheses_reconstructed": 4,
                "external_oracle_packages_absent": True, "native_ui_verified": False,
                "public_release_delivered": False}
        finally:
            stop()
            errors.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    result = verify(args.runtime, args.results, args.source_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "frozen", "full_results_reopened", "duration_ms")}))
