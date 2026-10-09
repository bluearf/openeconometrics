"""Fresh owned frozen acceptance, compiled identity and complete saved restart.

Native Run/UI proof is separate. This verifier owns a temporary runtime root
and must be given fresh result and receipt paths; previous waves stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    for name in ("", "pca", "factor", "extraction", "extraction_extensions", "rotation",
                 "rotation_extensions", "summary", "weighted", "uncertainty", "supplementary",
                 "scaling", "common", "replay", "canon", "discrim", "hierarchical", "kmeans", "reliability")) + ("openecon.analysis_contracts",)
CASES = ("frequency_factor", "alpha", "image_covariance", "cf_orthogonal", "cf_oblique",
         "partial_target_orthogonal", "partial_target_oblique", "loading_bootstrap")
MARKER = "MULTIVARIATE_EXTENSION_OK "
EXAMPLE = ROOT / "docs/examples/multivariate_extension_acceptance.py"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_result_files(directory, proof):
    """Inspect every complete stored table/vector against the run's exact hashes."""
    hashes = {name: digest(directory/(name+".json")) for name in CASES}
    assert hashes == proof["hashes"]
    for name in CASES:
        payload = json.loads((directory/(name+".json")).read_text())
        attrs, tables = payload["attrs"], payload["tables"]
        assert {key: len(value["data"]) for key, value in tables.items()} == proof["saved_table_counts"][name]
        for table in tables.values():
            assert len(table["index"]) == len(table["data"])
            assert all(len(row) == len(table["columns"]) for row in table["data"])
            assert all(value is None or isinstance(value, (int, float)) and math.isfinite(value)
                       for row in table["data"] for value in row)
        assert "\\begin{tabular}" in payload["latex"]
        if name != "loading_bootstrap":
            assert attrs["procedure"] == "factor" and attrs["factors"] == 2
            score = payload["scores"]
            assert len(score["data"]) == len(score["index"]) == 1203
            assert len(score["columns"]) == 2
            assert all(len(row) == 2 and all(isinstance(value, (int, float)) and math.isfinite(value) for value in row)
                       for row in score["data"])
            assert "score_coefficients" in tables and "descriptives" in tables
        else:
            assert attrs["procedure"] == "factor_bootstrap" and payload["scores"] is None
            assert attrs["successful_replications"] == attrs["replications"] == 199
            assert not attrs["failed_replications"] and attrs["sign_anchor"] == "a"
            assert attrs["inference_target"] == "fixed one-factor principal-factor estimator functional"
            assert len(tables["replicates"]["columns"]) == 12
            assert len(tables["covariance"]["columns"]) == 12
            assert not attrs["p_values_available"] and not attrs["inference_df_available"]
            for key in ("p_value", "df"):
                column = tables["estimates"]["columns"].index(key)
                assert all(row[column] is None for row in tables["estimates"]["data"])
    return hashes


def normalized(code):
    return code.replace(co_filename="<verified-source>", co_consts=tuple(
        normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts))


def source_identity(runtime, source_ref=None):
    """Compare every packaged multivariate module to live or explicitly pinned source."""
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


def verify_outputs(run, *, require_frozen=True):
    assert run["status"] == "ok", run.get("error")
    lines = [line for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(lines) == 1, "Exactly one new-wave completion marker is required"
    proof = json.loads(lines[0].split(MARKER, 1)[1])
    assert proof["cases"] == list(CASES)
    if require_frozen:
        assert proof["frozen"], "A source run is not frozen/native evidence"
    assert proof["physical_fixture_rows"] == 1203 and proof["weight_total"] == 2403
    assert proof["latent_correlation"] == .35
    assert proof["complete_score_rows"] == {name: 1203 for name in CASES[:-1]}
    assert proof["bootstrap_replications"] == 199 and proof["bootstrap_covariance_dimension"] == 12
    assert proof["saved_table_counts"]["loading_bootstrap"]["replicates"] == 199
    assert proof["saved_table_counts"]["loading_bootstrap"]["covariance"] == 12
    assert proof["saved_table_counts"]["loading_bootstrap"]["sample"] == 1203
    assert len(run["outputs"]) == 8
    assert [item["type"] for item in run["outputs"]] == ["table"] * 8
    assert [len(item["data"]["rows"]) for item in run["outputs"]] == [6]*7+[12]
    assert all("\\begin{tabular}" in item["latex"] for item in run["outputs"])
    assert "Display limit reached" not in run["stdout"]
    assert set(proof["hashes"]) == set(CASES)
    return proof


def frozen_header():
    return ("import sys, importlib, importlib.util\nfrom pathlib import Path\n"
            "assert getattr(sys, 'frozen', False)\n"
            "assert importlib.util.find_spec('scipy') is None\n"
            "assert importlib.util.find_spec('statsmodels') is None\n"
            f"for module_name in {MODULES!r}:\n"
            "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n")


def reconstruction_code(directory):
    """Source-testable full reconstruction; the frozen caller prepends its header."""
    return f"""
import json
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.core import TableSet
torch.set_num_threads(2)
root = Path({str(directory)!r})
frame = pd.DataFrame(**json.loads((root/'fixture.json').read_text()))
def restored_result(payload):
    return TableSet({{key: pd.DataFrame(**value) for key, value in payload['tables'].items()}}, **payload['attrs'])
def equal_tables(first, second):
    assert set(first) == set(second)
    for key in first:
        assert list(first[key].index) == list(second[key].index)
        assert list(first[key].columns) == list(second[key].columns)
        one = torch.as_tensor(first[key].to_numpy(dtype='float64'))
        two = torch.as_tensor(second[key].to_numpy(dtype='float64'))
        assert torch.allclose(one, two, atol=1e-12, rtol=1e-12, equal_nan=True), key
for name in {CASES[:-1]!r}:
    payload = json.loads((root/(name+'.json')).read_text())
    result = restored_result(payload)
    score = oe.factor_scores(result, frame)
    expected = pd.DataFrame(**payload['scores'])
    assert len(score) == 1203 and score.notna().all().all()
    assert list(score.index) == list(expected.index) and list(score.columns) == list(expected.columns)
    assert torch.allclose(torch.as_tensor(score.to_numpy()), torch.as_tensor(expected.to_numpy()), atol=1e-10, rtol=0)
    # Full restored table values/order and null cells survive a second JSON cycle.
    second = json.loads(json.dumps({{'attrs': result.attrs, 'tables': {{key: table.astype(object).where(table.notna(), None).to_dict(orient='split') for key, table in result.items()}}}}, allow_nan=False))
    equal_tables(result, restored_result(second))
    assert '\\\\begin{{tabular}}' in payload['latex']
    if name.startswith('partial_target'):
        mask = result['rotation_target_mask'].to_numpy().astype(bool)
        assert result['rotation_target'].isna().to_numpy().tolist() == (~mask).tolist()
        assert result.attrs['target_jacobian_rank'] == result.attrs['target_tangent_dimension']
payload = json.loads((root/'loading_bootstrap.json').read_text())
result = restored_result(payload)
attrs = result.attrs
draws = torch.as_tensor(result['replicates'].to_numpy(dtype='float64'))
assert draws.shape == (199, 12) and bool(torch.isfinite(draws).all())
centred = draws-draws.mean(0)
covariance = centred.T @ centred/198
assert torch.allclose(covariance, torch.as_tensor(result['covariance'].to_numpy(dtype='float64')), atol=1e-12, rtol=1e-12)
assert torch.allclose(covariance.diagonal().sqrt(), torch.as_tensor(result['estimates'].std_error.to_numpy(dtype='float64')), atol=1e-12, rtol=1e-12)
intervals = torch.quantile(draws, torch.tensor([.025, .975], dtype=torch.float64), dim=0)
assert torch.allclose(intervals.T, torch.as_tensor(result['estimates'][['ci_lower','ci_upper']].to_numpy(dtype='float64')), atol=1e-12, rtol=1e-12)
assert result['estimates'][['p_value','df']].isna().all().all()
assert attrs['successful_replications'] == 199 and not attrs['failed_replications']
# Refit from the exact saved raw complete sample with the fixed recorded seed
# and anchor: every loading/uniqueness vector, diagnostic and interval matches.
replay = oe.factor_bootstrap(result['sample'], attrs['variables'], method=attrs['method'],
    replications=attrs['replications'], confidence=attrs['confidence'], seed=attrs['seed'], anchor=attrs['sign_anchor'])
equal_tables(result, replay)
assert replay.attrs['source_content_sha256'] == attrs['source_content_sha256']
assert replay.attrs['inference_target'] == 'fixed one-factor principal-factor estimator functional'
assert not replay.attrs['p_values_available'] and not replay.attrs['inference_df_available']
assert '\\\\begin{{tabular}}' in payload['latex']
print('MULTIVARIATE_EXTENSION_FULL_RESULTS_REOPENED')
"""


def reopen_code(directory):
    return frozen_header() + reconstruction_code(directory)


def verify(runtime, result_directory, source_ref=None):
    runtime = runtime.resolve(strict=True)
    fingerprint, sources = digest(runtime), source_identity(runtime, source_ref)
    result_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    with tempfile.TemporaryDirectory(prefix="openecon-multivariate-extension-frozen-") as temporary:
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
            code = frozen_header() + f"MULTIVARIATE_EXTENSION_RESULT_DIRECTORY = {str(result_directory)!r}\n" + EXAMPLE.read_text()
            run = call("/console/execute", {"code": code, "timeout_seconds": 180})
            proof = verify_outputs(run)
            hashes = validate_result_files(result_directory, proof)
            call("/console/reset", {})
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            stop()
            start()
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            reopen = call("/console/execute", {"code": reopen_code(result_directory), "timeout_seconds": 180})
            assert reopen["status"] == "ok", reopen.get("error")
            assert "MULTIVARIATE_EXTENSION_FULL_RESULTS_REOPENED" in reopen["stdout"]
            assert hashes == {name: digest(result_directory/(name+".json")) for name in CASES}
            assert digest(runtime) == fingerprint
            receipt = {"status": "passed", "frozen": True, "source_identity": sources,
                "compiled_source_ref": source_ref,
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "runtime_sha256": fingerprint, "proof": proof, "duration_ms": run["duration_ms"],
                "example_sha256": digest(EXAMPLE), "verifier_sha256": digest(__file__),
                "worker_reset_readback": True, "runtime_restart_readback": True, "full_results_reopened": True,
                "seven_complete_saved_scores_reconstructed": True, "full_bootstrap_replicates_covariance_ci_reconstructed": True,
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
    parser.add_argument("--source-ref", help="Explicit source revision used to build this owned runtime")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    receipt = verify(args.runtime, args.results, args.source_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: receipt[key] for key in ("status", "frozen", "full_results_reopened", "duration_ms")}))
