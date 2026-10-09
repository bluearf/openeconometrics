"""Fresh frozen acceptance, compiled identity and full saved weight/matrix restart.

Native Run/UI proof is separate. All result and receipt paths must be fresh;
the verifier starts an owned temporary runtime and leaves earlier waves intact.
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
                 "scaling", "common", "replay", "canon", "canon_options", "discrim",
                 "discrim_options", "repeated", "hierarchical", "kmeans", "reliability")) + (
    "openecon.analysis_contracts", "openecon.econometrics.stats.rm_anova",
    "openecon.econometrics.stats.rm_contrast",
    "openecon.econometrics.stats.glm", "openecon.econometrics.stats.common",
    "openecon.econometrics.stats.replay", "openecon.econometrics.postest.index_codec",
    "openecon.econometrics.core", "openecon.engines.linalg",
    "openecon.engines.distributions", "openecon.resources", "openecon.dataset")
CASES = ("frequency_reliability", "frequency_adequacy", "frequency_discriminant",
         "summary_discriminant", "frequency_canonical", "summary_canonical",
         "wide_repeated", "repeated_contrasts")
METHODS = {name: ("lda", "qda") if "discriminant" in name else ("main",) for name in CASES}
POST_CASES = CASES[2:6]
MARKER = "MULTIVARIATE_WEIGHT_MATRIX_OK "
EXAMPLE = ROOT / "docs/examples/multivariate_weight_matrix.py"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite_json(value):
    """Permit mixed labels, nulls and finite scalars, including nested metadata."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and finite_json(item) for key, item in value.items())
    return False


def validate_table(table, axes):
    assert set(table) == {"index", "columns", "data"}
    assert len(table["index"]) == len(table["data"])
    assert all(len(row) == len(table["columns"]) for row in table["data"])
    assert finite_json(table)
    assert set(axes) == {"index", "columns"}
    assert axes["index"] and axes["columns"] and finite_json(axes)
    assert all(value is None or isinstance(value, (str, bool, int, float))
               for row in table["data"] for value in row)


def validate_result_files(directory, proof):
    """Inspect every full table, mixed cell, axis name, attribute and exact hash."""
    hashes = {name: digest(directory / (name + ".json")) for name in CASES}
    assert hashes == proof["hashes"]
    for name in CASES:
        payload = json.loads((directory / (name + ".json")).read_text())
        assert set(payload) == {"fits", "post", "post_axis_names"}
        assert set(payload["fits"]) == set(METHODS[name])
        assert finite_json(payload)
        for method, fit in payload["fits"].items():
            attrs = fit["attrs"]
            assert "\\begin{tabular}" in fit["latex"]
            assert attrs["n"] == (2403 if name.startswith("frequency") else
                                  7218 if "repeated" in name or name == "repeated_contrasts" else 1203)
            if fit["kind"] == "tableset":
                assert set(fit) == {"kind", "attrs", "tables", "axis_names", "latex"}
                assert len(fit["tables"]) == proof["fit_table_counts"][name][method]
                assert set(fit["tables"]) == set(fit["axis_names"])
                for key, table in fit["tables"].items():
                    validate_table(table, fit["axis_names"][key])
            else:
                assert fit["kind"] == "frame" and name == "frequency_adequacy"
                assert set(fit) == {"kind", "attrs", "table", "axis_names", "latex"}
                assert proof["fit_table_counts"][name][method] == 1
                validate_table(fit["table"], fit["axis_names"])
            if name in ("frequency_reliability", "frequency_adequacy", "frequency_canonical"):
                assert attrs["physical_rows"] == 962 and attrs["n_zero_weight"] == 241
                assert attrs["covariance_divisor"] == 2402
            if "discriminant" in name:
                assert attrs["procedure"] == "discrim" and attrs["method"] == method
                assert len(attrs["discriminant_state_sha256"]) == 64
                assert attrs["training_classification_available"] == name.startswith("frequency")
                if name.startswith("summary"):
                    assert not attrs["loo_available"]
                    assert not any("classification_table" in key for key in fit["tables"])
            if "canonical" in name:
                assert attrs["procedure"] == "canon" and attrs["canonical_pairs"] == 3
                assert {"descriptives", "raw_coefficients", "joint_covariance"} <= set(fit["tables"])
            if name in ("wide_repeated", "repeated_contrasts"):
                assert attrs["n_subjects"] == 1203 and attrs["state_schema"] == "openecon.rm_wide.v1"
                assert len(fit["tables"]["sample"]["data"]) == 1203
                assert len(attrs["source_content_sha256"]) == 64
            if name == "repeated_contrasts":
                assert attrs["q"] == 2 and attrs["df_numerator"] == 2 and attrs["df_denominator"] == 1201
                assert attrs["df_marginal"] == 1202
                assert len(fit["tables"]["subject_contrasts"]["data"]) == 1203
        assert set(payload["post"]) == set(payload["post_axis_names"])
        assert set(payload["post"]) == (set(METHODS[name]) if name in POST_CASES else set())
        for method, table in payload["post"].items():
            validate_table(table, payload["post_axis_names"][method])
            assert len(table["data"]) == proof["full_post_rows"][name][method] == 1203
            assert len(table["columns"]) == (4 if "discriminant" in name else 6)
            assert all(isinstance(value, (int, float)) and math.isfinite(value)
                       for row in table["data"] for value in row)
    fixtures = json.loads((directory / "fixtures.json").read_text())
    assert set(fixtures) == {"canonical", "discriminant", "repeated", "C"}
    assert finite_json(fixtures)
    for name in ("canonical", "discriminant", "repeated"):
        validate_table(fixtures[name], {"index": [None], "columns": [None]})
        assert len(fixtures[name]["data"]) == 1203
    return hashes


def normalized(code):
    return code.replace(co_filename="<verified-source>", co_consts=tuple(
        normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts))


def source_identity(runtime, source_ref=None):
    """Compare packaged methods and numerical dependencies to live/pinned source."""
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
    assert proof["fixture_rows"] == 1203 and proof["weight_total"] == 2403
    assert proof["full_post_rows"] == {name: {method: 1203 for method in METHODS[name]} for name in POST_CASES}
    assert proof["full_subject_contrast_rows"] == 1203
    assert set(proof["fit_table_counts"]) == set(CASES)
    assert all(set(proof["fit_table_counts"][name]) == set(METHODS[name]) for name in CASES)
    assert len(run["outputs"]) == 8
    assert [item["type"] for item in run["outputs"]] == ["table"] * 8
    assert [len(item["data"]["rows"]) for item in run["outputs"]] == [6, 7, 3, 2, 3, 3, 8, 2]
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
    """Self-contained source-testable reconstruction; frozen caller adds its header."""
    return rf'''
import json
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.core import TableSet
from openecon.econometrics.stats import common as sc
torch.set_num_threads(2)
root = Path({str(directory)!r})
fixtures = json.loads((root/'fixtures.json').read_text())
canonical = pd.DataFrame(**fixtures['canonical'])
discriminant = pd.DataFrame(**fixtures['discriminant'])
repeated = pd.DataFrame(**fixtures['repeated'])
names = list(canonical.columns)
def decode_table(spec, axes):
    frame = pd.DataFrame(**spec)
    frame.index.names = axes['index']
    frame.columns.names = axes['columns']
    return frame
def restore_fit(fit):
    if fit['kind'] == 'tableset':
        result = TableSet({{key: decode_table(table, fit['axis_names'][key])
                           for key, table in fit['tables'].items()}}, **fit['attrs'])
    else:
        result = decode_table(fit['table'], fit['axis_names'])
        result.attrs.update(fit['attrs'])
    assert result.attrs == fit['attrs']
    return result
def same_frame(actual, expected):
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected),
        check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
def same_fit(actual, expected):
    if isinstance(actual, TableSet):
        assert set(actual) == set(expected)
        for key in actual:
            same_frame(actual[key], expected[key])
    else:
        same_frame(actual, expected)
def repack(fit):
    result = restore_fit(fit)
    if isinstance(result, TableSet):
        saved = {{'kind': 'tableset', 'attrs': result.attrs,
            'tables': {{key: table.astype(object).where(table.notna(), None).to_dict(orient='split')
                       for key, table in result.items()}},
            'axis_names': {{key: {{'index': table.index.names, 'columns': table.columns.names}}
                           for key, table in result.items()}}}}
    else:
        saved = {{'kind': 'frame', 'attrs': result.attrs,
            'table': result.astype(object).where(result.notna(), None).to_dict(orient='split'),
            'axis_names': {{'index': result.index.names, 'columns': result.columns.names}}}}
    again = restore_fit(json.loads(json.dumps(saved, allow_nan=False)))
    same_fit(result, again)
    assert result.attrs == again.attrs
    return result
saved = {{}}
for name in {CASES!r}:
    payload = json.loads((root/(name+'.json')).read_text())
    saved[name] = {{}}
    for method, fit in payload['fits'].items():
        result = repack(fit)
        saved[name][method] = result
        assert '\\begin{{tabular}}' in fit['latex']
        if 'discriminant' in name:
            actual = oe.discrim_predict(result, discriminant)
        elif 'canonical' in name:
            actual = oe.canon_scores(result, canonical)
        else:
            actual = None
        if actual is not None:
            expected = decode_table(payload['post'][method], payload['post_axis_names'][method])
            same_frame(actual, expected)
            assert len(actual) == 1203 and actual.notna().all().all()
        if name in ('wide_repeated', 'repeated_contrasts'):
            restored = oe.rm_restore(result)
            same_fit(restored, result)
            assert restored.attrs['source_content_sha256'] == result.attrs['source_content_sha256']
            assert restored.attrs['measurement_columns'] == names
            assert restored.attrs['n_subjects'] == 1203
wide = saved['wide_repeated']['main']
contrast = saved['repeated_contrasts']['main']
replay = oe.rm_contrasts(oe.rm_restore(wide), contrast['contrasts'],
    null=contrast['null'].iloc[:, 0].tolist(), alpha=contrast.attrs['alpha'],
    sampling_model=contrast.attrs['sampling_model'])
same_fit(replay, contrast)
assert replay.attrs['contrast_content_sha256'] == contrast.attrs['contrast_content_sha256']
assert replay.attrs['df_numerator'] == 2 and replay.attrs['df_denominator'] == 1201
assert replay.attrs['df_marginal'] == 1202
x = torch.as_tensor(repeated.to_numpy(dtype='float64'))
C = torch.as_tensor(contrast['contrasts'].to_numpy(dtype='float64'))
null = torch.as_tensor(contrast['null'].to_numpy(dtype='float64').reshape(-1))
vectors = (x-x[:, :1]) @ C.T
n, q = vectors.shape
offset = vectors-vectors[0]
location = offset.mean(0)
location += (offset-location).mean(0)
estimate = vectors[0]+location
dev = offset-location
covariance = dev.T @ dev/((n-1)*n)
standard = covariance.diag().sqrt()
critical = sc.t_critical(contrast.attrs['alpha'], n-1)
assert torch.allclose(vectors, torch.as_tensor(contrast['subject_contrasts'].to_numpy(dtype='float64')), atol=1e-12, rtol=1e-12)
assert torch.allclose(covariance, torch.as_tensor(contrast['joint_covariance'].to_numpy(dtype='float64')), atol=1e-12, rtol=1e-12)
expected = torch.stack([estimate, standard, null, (estimate-null)/standard,
    torch.full_like(estimate, n-1), estimate-critical*standard, estimate+critical*standard], dim=1)
columns = ['estimate', 'std_error', 'null', 't', 'df', 'ci_lower', 'ci_upper']
assert torch.allclose(expected, torch.as_tensor(contrast['estimates'][columns].to_numpy(dtype='float64')), atol=1e-12, rtol=1e-12)
delta = estimate-null
t_squared = float(delta @ torch.linalg.solve(covariance, delta))
joint = contrast['joint'].iloc[0]
assert abs(joint.t_squared-t_squared) <= 1e-10*max(1, abs(t_squared))
assert abs(joint.statistic-t_squared*(n-q)/(q*(n-1))) <= 1e-10*max(1, abs(joint.statistic))
assert joint.df_numerator == q and joint.df_denominator == n-q
print('MULTIVARIATE_WEIGHT_MATRIX_FULL_RESULTS_REOPENED')
'''


def reopen_code(directory):
    return frozen_header() + reconstruction_code(directory)

def verify(runtime, result_directory, source_ref=None):
    runtime = runtime.resolve(strict=True)
    fingerprint, sources = digest(runtime), source_identity(runtime, source_ref)
    result_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    with tempfile.TemporaryDirectory(prefix="openecon-multivariate-weight-matrix-frozen-") as temporary:
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
            code = frozen_header() + f"MULTIVARIATE_WEIGHT_MATRIX_RESULT_DIRECTORY = {str(result_directory)!r}\n" + EXAMPLE.read_text()
            run = call("/console/execute", {"code": code, "timeout_seconds": 180})
            proof = verify_outputs(run)
            hashes = validate_result_files(result_directory, proof)
            fixture_fingerprint = digest(result_directory / "fixtures.json")
            call("/console/reset", {})
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            stop()
            start()
            stored = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert all(stored[key] == run[key] for key in ("code", "stdout", "outputs", "events"))
            reopen = call("/console/execute", {"code": reopen_code(result_directory), "timeout_seconds": 180})
            assert reopen["status"] == "ok", reopen.get("error")
            assert "MULTIVARIATE_WEIGHT_MATRIX_FULL_RESULTS_REOPENED" in reopen["stdout"]
            assert hashes == {name: digest(result_directory/(name+".json")) for name in CASES}
            assert digest(result_directory / "fixtures.json") == fixture_fingerprint
            assert digest(runtime) == fingerprint
            receipt = {"status": "passed", "frozen": True, "source_identity": sources,
                "compiled_source_ref": source_ref,
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "runtime_sha256": fingerprint, "proof": proof, "duration_ms": run["duration_ms"],
                "example_sha256": digest(EXAMPLE), "verifier_sha256": digest(__file__),
                "worker_reset_readback": True, "runtime_restart_readback": True, "full_results_reopened": True,
                "four_complete_discriminant_predictions_reconstructed": True,
                "two_complete_canonical_score_tables_reconstructed": True,
                "all_complete_rm_states_restored": True, "full_rm_contrasts_covariance_df_ci_reconstructed": True,
                "fixtures_sha256": fixture_fingerprint,
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
