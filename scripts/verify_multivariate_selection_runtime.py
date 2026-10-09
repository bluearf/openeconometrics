"""Owned frozen-runtime acceptance for eight selection and contrast contracts.

The actual native Run/Quit/reopen proof is collected separately. This helper
uses fresh temporary runtime storage and never changes a user's project.
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

from verify_multivariate_weight_matrix_runtime import digest, finite_json, normalized, validate_table

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/multivariate_selection_contrasts.py"
MARKER = "MULTIVARIATE_SELECTION_CONTRASTS_OK "
CASES = ("forward_selection", "backward_selection", "stepwise_selection", "frequency_selection",
         "frequency_manova", "summary_manova", "saved_manova_contrast", "saved_rm_joint")
MODULES = (
    "openecon.econometrics.multivariate", "openecon.econometrics.multivariate.discrim_selection",
    "openecon.econometrics.multivariate.discrim", "openecon.econometrics.multivariate.discrim_options",
    "openecon.econometrics.multivariate.common", "openecon.econometrics.multivariate.weighted",
    "openecon.econometrics.multivariate.replay", "openecon.econometrics.stats",
    "openecon.econometrics.stats.manova_options", "openecon.econometrics.stats.manova",
    "openecon.econometrics.stats.streaming_manova", "openecon.econometrics.stats.anova",
    "openecon.econometrics.stats.streaming_anova", "openecon.econometrics.stats.rm_anova",
    "openecon.econometrics.stats.streaming_rm", "openecon.econometrics.stats.rm_contrast",
    "openecon.econometrics.stats.glm", "openecon.econometrics.stats.common",
    "openecon.econometrics.stats.replay", "openecon.econometrics.stats.posthoc",
    "openecon.econometrics.summary_state", "openecon.econometrics.resident_cpu",
    "openecon.econometrics.postest.index_codec", "openecon.econometrics.core",
    "openecon.engines.linalg", "openecon.engines.distributions", "openecon.engines.inference",
    "openecon.resources", "openecon.dataset", "openecon.analysis_contracts",
)


def source_identity(runtime, source_ref):
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for name in MODULES:
        path = ROOT / "src" / Path(*name.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        source = subprocess.check_output(["git", "show", f"{source_ref}:{path.relative_to(ROOT)}"], cwd=ROOT)
        expected = compile(source, str(path), "exec", dont_inherit=True)
        assert normalized(archive.extract(name)) == normalized(expected), name
        sources[name] = hashlib.sha256(source).hexdigest()
    return sources


def validate_fit(spec):
    assert finite_json(spec)
    assert any(marker in spec["latex"] for marker in ("\\begin{tabular}", "\\begin{longtable}"))
    if spec["kind"] == "tableset":
        assert set(spec) == {"kind", "title", "attrs", "tables", "axis_names", "latex"}
        assert set(spec["tables"]) == set(spec["axis_names"])
        for key, frame in spec["tables"].items():
            validate_table(frame, spec["axis_names"][key])
    else:
        assert spec["kind"] == "frame"
        assert set(spec) == {"kind", "attrs", "table", "axis_names", "latex"}
        validate_table(spec["table"], spec["axis_names"])


def validate_result_files(directory, proof):
    hashes = {name: digest(directory/(name+".json")) for name in CASES}
    assert hashes == proof["hashes"]
    for name in CASES:
        payload = json.loads((directory/(name+".json")).read_text())
        assert set(payload) == {"fit", "post"}
        validate_fit(payload["fit"])
        assert len(payload["fit"]["tables"]) == proof["fit_table_counts"][name]
        if payload["post"] is not None:
            validate_fit(payload["post"])
        if name.endswith("selection"):
            prediction = payload["post"]["table"]
            assert len(prediction["data"]) == 603 and len(prediction["columns"]) == 4
            assert all(value is None for value in prediction["data"][7])
            assert all(all(value is not None for value in row)
                       for i, row in enumerate(prediction["data"]) if i != 7)
        else:
            assert "multivariate" in payload["fit"]["tables"]
    fixtures = json.loads((directory/"fixtures.json").read_text())
    bases = json.loads((directory/"base-models.json").read_text())
    assert finite_json(fixtures) and finite_json(bases)
    assert len(fixtures["data"]["data"]) == len(fixtures["query"]["data"]) == 603
    assert len(fixtures["rm_data"]["data"]) == 2412
    assert set(bases) == {"manova", "manova_dataset", "rm", "empty_selection", "empty_predictions"}
    for spec in bases.values():
        validate_fit(spec)
    assert len(bases["empty_predictions"]["table"]["data"]) == 7
    return {name: digest(directory/name) for name in [*(case+".json" for case in CASES),
                                                    "fixtures.json", "base-models.json"]}


def verify_outputs(run, *, require_frozen=True):
    assert run["status"] == "ok", run.get("error")
    lines = [line for line in run["stdout"].splitlines() if line.startswith(MARKER)]
    assert len(lines) == 1
    proof = json.loads(lines[0][len(MARKER):])
    assert proof["cases"] == list(CASES)
    if require_frozen:
        assert proof["frozen"]
    assert proof["fixture_rows"] == 603 and proof["rm_long_rows"] == 2412
    assert proof["weight_total"] == 1203 and proof["empty_prediction_rows"] == 7
    assert proof["prediction_rows"] == {name: 603 for name in CASES[:4]}
    assert len(run["outputs"]) == 8
    assert [item["type"] for item in run["outputs"]] == ["table"]*8
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


def reconstruction_code(directory):
    return rf'''
import json
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.core import TableSet
torch.set_num_threads(2)
root = Path({str(directory)!r})
fixtures = json.loads((root/'fixtures.json').read_text())
bases = json.loads((root/'base-models.json').read_text())
def restore_frame(spec, axes):
    frame = pd.DataFrame(**spec)
    frame.index.names, frame.columns.names = axes['index'], axes['columns']
    return frame
def restore(spec):
    if spec['kind'] == 'tableset':
        return TableSet({{key: restore_frame(value, spec['axis_names'][key])
                         for key, value in spec['tables'].items()}}, title=spec['title'], **spec['attrs'])
    frame = restore_frame(spec['table'], spec['axis_names'])
    frame.attrs.update(spec['attrs'])
    return frame
def same(first, second):
    if isinstance(first, TableSet):
        assert first.title == second.title and first.attrs == second.attrs
        assert set(first) == set(second)
        for key in first:
            same(first[key], second[key])
    else:
        pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second),
            check_dtype=False, check_exact=False,
                                      check_index_type=False if first.index.empty and second.index.empty else "equiv",
                                      check_column_type=False if first.columns.empty and second.columns.empty else "equiv", rtol=1e-12, atol=1e-12)
query = pd.DataFrame(**fixtures['query'])
stored = {{}}
for name in {CASES!r}:
    payload = json.loads((root/(name+'.json')).read_text())
    result = restore(payload['fit'])
    stored[name] = result
    same(result, oe.restore_summary(oe.summary_state(result)))
    if name.endswith('selection'):
        predicted = oe.discrim_stepwise_predict(result, query)
        same(predicted, restore(payload['post']))
        assert len(predicted) == 603 and predicted.index.equals(query.index)
    elif name in ('frequency_manova', 'summary_manova'):
        tested = oe.manova_contrast(result, fixtures['L_oneway'], M=fixtures['M'], null=fixtures['null'],
            contrast_names=['early-control', 'late-control'], transform_names=['first-gap', 'second-gap'])
        same(tested, restore(payload['post']))
for key in ('manova', 'manova_dataset'):
    result = oe.manova_contrast(restore(bases[key]), fixtures['L'], M=fixtures['M'], null=fixtures['null'],
        contrast_names=['group-effect', 'covariate'], transform_names=['first-gap', 'second-gap'])
    expected = stored['saved_manova_contrast'] if key == 'manova' else restore(json.loads((root/'saved_manova_contrast.json').read_text())['post'])
    same(result, expected)
result = oe.rm_mtest(restore(bases['rm']), fixtures['L_rm'], M=fixtures['M'], null=fixtures['null'],
    contrast_names=['group-1', 'group-2'], transform_names=['first-gap', 'second-gap'])
same(result, stored['saved_rm_joint'])
same(oe.discrim_stepwise_predict(restore(bases['empty_selection']), pd.DataFrame(index=range(7))),
     restore(bases['empty_predictions']))
print('MULTIVARIATE_SELECTION_FULL_RESULTS_REOPENED')
'''


def verify(runtime, result_directory, source_ref):
    runtime = runtime.resolve(strict=True)
    fingerprint, sources = digest(runtime), source_identity(runtime, source_ref)
    result_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    with tempfile.TemporaryDirectory(prefix="openecon-selection-contrasts-frozen-") as temporary:
        data = Path(temporary)
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        process = descriptor = token = None
        errors = (data/"errors.log").open("ab")

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
                start_new_session=True, env=environment)
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
            code = frozen_header()+f"MULTIVARIATE_SELECTION_RESULT_DIRECTORY = {str(result_directory)!r}\n"+EXAMPLE.read_text()
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
            reopened = call("/console/execute", {"code": frozen_header()+reconstruction_code(result_directory), "timeout_seconds": 180})
            assert reopened["status"] == "ok", reopened.get("error")
            assert "MULTIVARIATE_SELECTION_FULL_RESULTS_REOPENED" in reopened["stdout"]
            assert hashes == validate_result_files(result_directory, proof)
            assert digest(runtime) == fingerprint
            receipt = {"status": "passed", "frozen": True, "compiled_source_ref": source_ref,
                "source_identity": sources, "runtime_sha256": fingerprint, "proof": proof,
                "duration_ms": run["duration_ms"], "complete_payload_hashes": hashes,
                "example_sha256": digest(EXAMPLE), "verifier_sha256": digest(__file__),
                "worker_reset_readback": True, "runtime_restart_readback": True,
                "full_results_reopened": True, "complete_predictions_reconstructed": 4,
                "prior_only_empty_predictions_reconstructed": True,
                "oneway_summary_and_resident_dataset_manova_contrasts_reconstructed": True,
                "joint_rm_hypotheses_full_covariance_and_inference_reconstructed": True,
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
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "frozen", "full_results_reopened", "duration_ms")}))
