"""Smoke an owned local runtime with real Parquet data and dense fit oracles.

All projects, files and processes belong to a temporary verification directory.
No human application, account, workspace or cloud service is accessed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4


MARKER = "__OPENECON_STREAMING_CHECK__"
COMBINATIONS = [("ols", covariance) for covariance in ("nonrobust", "HC1", "HC3", "cluster")]
COMBINATIONS += [(estimator, covariance) for estimator in ("logit", "probit")
                 for covariance in ("nonrobust", "cluster")]

FIXTURE = r'''
import importlib.metadata
import json
import os
from pathlib import Path
import pandas as pd
import torch
import openecon.analysis as analysis
import openecon.streaming_analysis as streaming
import openecon.streaming_design as design_module
import openecon.linear_ols.streaming as ols_streaming
from openecon.dataset import Dataset

torch.set_num_threads(min(torch.get_num_threads(), 2))
generator = torch.Generator().manual_seed(31003)
n = 120001
x = torch.randn(n, generator=generator, dtype=torch.float64)
z = torch.randn(n, generator=generator, dtype=torch.float64)
category_code = torch.arange(n, dtype=torch.int64) % 4
category_effect = category_code.to(torch.float64) * .17
linear = -.2 + .65 * x - .35 * z + category_effect
frame = pd.DataFrame({
    "x": x.tolist(), "z": z.tolist(),
    "category": pd.Categorical(["abcd"[int(value)] for value in category_code], categories=list("abcd")),
    "group": (torch.arange(n, dtype=torch.int64) % 5000).tolist(),
    "y": (1.2 + .7 * x - .4 * z + category_effect + torch.randn(n, generator=generator, dtype=torch.float64)).tolist(),
    "binary": (torch.rand(n, generator=generator, dtype=torch.float64) < torch.sigmoid(linear)).to(torch.float64).tolist(),
})
frame.loc[frame.index % 971 == 0, "x"] = float("nan")
frame.loc[frame.index % 1319 == 0, "category"] = None
frame.loc[frame.index % 1999 == 0, "group"] = float("nan")
fixture_path = Path.cwd() / "streaming-fixture.parquet"
frame.to_parquet(fixture_path, index=False, row_group_size=4096)
assert n > 100000 and oe.scan(fixture_path).row_count == n
constants = {
    "max_parameters": streaming.MAX_PARAMETERS,
    "working_bytes": streaming.WORKING_BYTES,
    "max_batch_rows": streaming.MAX_BATCH_ROWS,
    "category_bytes": design_module.MAX_CATEGORY_BYTES,
    "cluster_key_bytes": design_module.MAX_CLUSTER_BYTES,
}
assert constants == {"max_parameters": 384, "working_bytes": 128*1024*1024,
                     "max_batch_rows": 65536, "category_bytes": 2*1024*1024,
                     "cluster_key_bytes": 6*1024*1024}, constants
streaming.MAX_BATCH_ROWS = 4096
# Native OLS v5 has its own reader/numerical planner. Change this constant
# only in this disposable worker, and verify the observed encoded peak and
# every actual QR input below rather than inferring bounds from declarations.
assert ols_streaming._READER_ROWS == 65536
assert ols_streaming._WORKING_BYTES == 128*1024*1024
ols_streaming._READER_ROWS = 4096
print("__OPENECON_STREAMING_CHECK__" + json.dumps({
    "version": oe.__version__, "distribution_version": importlib.metadata.version("openecon"),
    "fixture_rows": n, "fixture_bytes": fixture_path.stat().st_size,
    "physical_parquet": True, "cluster_groups_before_missing": 5000,
    "constants": constants,
}))
'''

MODEL = r'''
estimator = __ESTIMATOR__
covariance = __COVARIANCE__
outcome = "y" if estimator == "ols" else "binary"
options = {"y": outcome, "x": ["x", "z", "category"], "categorical": ["category"],
           "covariance": covariance, "missing": "drop"}
if covariance == "cluster":
    options["cluster"] = "group"
method = getattr(oe, estimator)
reference = method(data=frame, **options)
selected = [outcome, "x", "z", "category"] + (["group"] if covariance == "cluster" else [])
physical = frame.index[frame[selected].notna().all(axis=1)].tolist()
saved_prepare, saved_hash = analysis._prepare_data, analysis._frame_hasher
saved_qr = torch.linalg.qr
def forbidden_dense(*args, **kwargs):
    raise AssertionError("A scan fit attempted a full dense preparation/hash")
def bounded_qr(matrix, *args, **kwargs):
    assert matrix.shape[0] <= 4096, matrix.shape
    return saved_qr(matrix, *args, **kwargs)
analysis._prepare_data = analysis._frame_hasher = forbidden_dense
torch.linalg.qr = bounded_qr
try:
    model = method(data=oe.scan(fixture_path), **options)
finally:
    analysis._prepare_data, analysis._frame_hasher = saved_prepare, saved_hash
    torch.linalg.qr = saved_qr
assert model.nobs == reference.nobs == len(physical)
assert model.nobs_original == 120001
assert model.dropped_rows == 120001 - len(physical) > 0
assert model.sample_positions == [] and len(model.predictions) == 400
assert [row["row"] for row in model.predictions] == physical[:400]
assert model.provenance["sample_positions_omitted"] is True
assert model.provenance["streaming"]["row_limit"] is None
if estimator == "ols":
    # OLS moved into linear_ols with explicit reporting-only state. Binary
    # results retain the original streaming_analysis provenance contract.
    maximum_encoded_rows = model.provenance["streaming"]["maximum_encoded_rows"]
    assert model._state["samples_only"] is True
    assert "original_frame" not in model._state
    for name in ("x", "y", "weights", "resid", "fitted", "leverage", "sample_positions"):
        assert len(model._state[name]) <= 400, (name, len(model._state[name]))
else:
    maximum_encoded_rows = model.provenance["streaming"]["maximum_encoded_batch_rows"]
    assert model.provenance["solver_diagnostics"]["dense_observation_matrix"] is False
assert 0 < maximum_encoded_rows <= 4096
assert [value.term for value in model.coefficients] == [value.term for value in reference.coefficients]
actual = torch.tensor([[value.estimate, value.std_error] for value in model.coefficients], dtype=torch.float64)
expected = torch.tensor([[value.estimate, value.std_error] for value in reference.coefficients], dtype=torch.float64)
torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)
torch.testing.assert_close(torch.tensor(model.covariance_matrix, dtype=torch.float64),
                          torch.tensor(reference.covariance_matrix, dtype=torch.float64), rtol=2e-8, atol=2e-8)
assert abs(model.metrics["log_likelihood"] - reference.metrics["log_likelihood"]) <= 2e-8 * (1 + abs(reference.metrics["log_likelihood"]))
payload_bytes = len(model.model_dump_json().encode("utf-8"))
assert payload_bytes < 100000, payload_bytes
latex = model.to_latex()
assert "\\toprule" in latex and "\\midrule" in latex and "\\bottomrule" in latex
diagnostics = {} if estimator == "ols" else model.provenance["solver_diagnostics"]
if covariance == "cluster":
    if estimator == "ols":
        subsets = model.provenance["streaming"]["cluster_subsets"]
        assert len(subsets) == 1 and subsets[0]["columns"] == ["group"]
        diagnostics = subsets[0]
    assert model.inference["cluster_count"] == reference.inference["cluster_count"] == 5000
    assert diagnostics["cluster_aggregation"] == "sqlite_spill"
    assert diagnostics["cluster_cache_capacity"] < 5000
    assert diagnostics["cluster_spill_writes"] > 0 and diagnostics["cluster_scratch_bytes"] > 0
scratch = Path(os.environ["OPENECON_SCRATCH_DIRECTORY"])
assert not list(scratch.rglob("groups.sqlite3"))
display(model)
print("__OPENECON_STREAMING_CHECK__" + json.dumps({
    "estimator": estimator, "covariance": covariance, "nobs": model.nobs,
    "dropped_rows": model.dropped_rows, "cluster_count": model.inference["cluster_count"],
    "solver": model.provenance["solver"], "passes": model.provenance["streaming"]["passes"],
    "batch_rows": maximum_encoded_rows,
    "encoded_peak_field": "maximum_encoded_rows" if estimator == "ols" else "maximum_encoded_batch_rows",
    "coefficient_se_max_abs_error": float((actual-expected).abs().max()),
    "result_json_bytes": payload_bytes, "latex_booktabs": True,
    "no_full_positions_or_dense_scan_matrix": True,
    "cluster_spill_writes": diagnostics.get("cluster_spill_writes", 0),
    "normal_scratch_reclaimed": True,
}))
'''

FAILURES = r'''
from openecon.analysis_contracts import AnalysisError
separation_codes = {}
for estimator in ("logit", "probit"):
    separated = pd.DataFrame({"x": [-2., -1., 1., 2.], "binary": [0., 0., 1., 1.]})
    try:
        getattr(oe, estimator)(data=Dataset.from_frame(separated), y="binary", x=["x"])
    except AnalysisError as error:
        assert error.code == "separation_detected", (estimator, error.code)
        separation_codes[estimator] = error.code
    else:
        raise AssertionError("Separated fit returned success")
changed_codes = {}
for estimator in ("ols", "logit", "probit"):
    state = [0]
    outcome = "y" if estimator == "ols" else "binary"
    def changing_batches():
        state[0] += 1
        changed = frame.iloc[:3000].copy()
        if state[0] > 1:
            changed.loc[1, outcome] = changed.loc[1, outcome] + 1 if estimator == "ols" else 1 - changed.loc[1, outcome]
        yield changed
    try:
        getattr(oe, estimator)(data=Dataset.from_batches(changing_batches, list(frame.columns), row_count=3000),
                              y=outcome, x=["x", "z", "category"], categorical=["category"], missing="drop")
    except AnalysisError as error:
        assert error.code == "source_changed", (estimator, error.code)
        changed_codes[estimator] = error.code
    else:
        raise AssertionError("Changed source returned success")
print("__OPENECON_STREAMING_CHECK__" + json.dumps({"separation_codes": separation_codes,
                                                "callback_domain_codes": changed_codes}))
'''

SCRATCH = r'''
from openecon.engines.streaming_groups import ClusterAccumulator
owned_scratch = Path(os.environ["OPENECON_SCRATCH_DIRECTORY"])
cancel_probe = ClusterAccumulator(2, scratch_directory=owned_scratch)
cancel_probe.add([str(value).encode() for value in range(6000)], torch.ones((6000,2), dtype=torch.float64))
assert cancel_probe.diagnostics["cluster_spill_writes"] > 0
assert list(owned_scratch.rglob("groups.sqlite3"))
print("__OPENECON_STREAMING_CHECK__" + json.dumps({"owned_scratch": str(owned_scratch),
                                                "spill_writes": cancel_probe.diagnostics["cluster_spill_writes"]}))
'''


def verify(runtime: Path | None, output: Path, *, expected_version: str | None = None) -> dict:
    runtime = runtime.expanduser().resolve(strict=True) if runtime else None
    report = {"status": "running", "human_data_access": False, "system_python_required": runtime is None,
              "runtime": str(runtime) if runtime else "source", "models": []}
    with tempfile.TemporaryDirectory(prefix="openecon-streaming-check-") as directory:
        root = Path(directory)
        stderr_path = root / "runtime-stderr.log"
        process = None
        stderr_file = None
        descriptor = None
        token = None
        project = uuid4().hex
        prefix = f"/api/desktop/projects/{project}/workspace"

        def call(path, body=None, *, timeout=180):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-OpenEcon-Token"] = token
            request = Request(descriptor["url"] + prefix + path, headers=headers,
                              data=json.dumps(body).encode() if body is not None else None)
            try:
                with urlopen(request, timeout=timeout) as response:
                    return json.load(response)
            except HTTPError as error:
                raise RuntimeError(f"Owned API HTTP {error.code}: {error.read(4000).decode()}") from None

        def execute(code):
            return call("/console/execute", {"code": code, "timeout_seconds": 180})

        def checked(code):
            response = execute(code)
            if response["status"] != "ok":
                raise RuntimeError(json.dumps(response.get("error"), ensure_ascii=False)[:8000])
            lines = [line[len(MARKER):] for line in response["stdout"].splitlines() if line.startswith(MARKER)]
            if not lines:
                raise RuntimeError("Owned console did not return the verification marker")
            return json.loads(lines[-1]), response

        def stop():
            if descriptor is not None and token is not None and process is not None and process.poll() is None:
                try:
                    call("/console/reset", {}, timeout=10)
                except (OSError, RuntimeError):
                    pass
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write(b'{"type":"shutdown"}\n')
                        process.stdin.flush()
                    except BrokenPipeError:
                        pass
                    try:
                        process.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait(timeout=3)
                process.stdin.close()
                process.stdout.close()
            if stderr_file is not None:
                stderr_file.close()

        try:
            env = os.environ.copy()
            if runtime:
                env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            stderr_file = stderr_path.open("ab")
            command = [str(runtime)] if runtime else [sys.executable, "-m", "openecon.desktop_entry"]
            process = subprocess.Popen(command + ["--data-root", str(root)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr_file, env=env,
                                       start_new_session=os.name == "posix")
            started = time.monotonic()
            deadline = started + 60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], max(0, min(1, deadline-time.monotonic())))[0]:
                    descriptor = json.loads(process.stdout.readline())
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Owned runtime exited at startup: {process.returncode}")
            if descriptor is None or descriptor.get("type") != "ready":
                raise RuntimeError("Owned runtime was not ready within 60 seconds")
            assert urlparse(descriptor["url"]).hostname in {"127.0.0.1", "localhost", "::1"}
            report["startup_seconds"] = round(time.monotonic()-started, 3)
            token = call("/session")["token"]
            report["phase"] = "fixture"
            metadata, _ = checked(FIXTURE)
            report.update(metadata)
            if expected_version:
                assert metadata["version"] == expected_version, metadata
            if runtime:
                assert metadata["distribution_version"] == metadata["version"], metadata
                report["distribution_metadata_matches_source"] = True
            for estimator, covariance in COMBINATIONS:
                report["phase"] = f"{estimator}/{covariance}"
                code = MODEL.replace("__ESTIMATOR__", repr(estimator)).replace("__COVARIANCE__", repr(covariance))
                record, response = checked(code)
                assert any("\\toprule" in item.get("latex", "") for item in response["outputs"]), response["outputs"]
                report["models"].append(record)
                print(f"Verified {estimator}/{covariance}: {record['nobs']} retained rows", flush=True)
            report["phase"] = "failure contracts"
            failures, _ = checked(FAILURES)
            report.update(failures)
            report["phase"] = "interrupt scratch cleanup"
            scratch, _ = checked(SCRATCH)
            owned_scratch = Path(scratch["owned_scratch"]).resolve()
            assert owned_scratch.is_relative_to(root.resolve()) and owned_scratch.exists()
            assert list(owned_scratch.rglob("groups.sqlite3"))
            with ThreadPoolExecutor(max_workers=1) as pool:
                running = pool.submit(execute, "import time\ntime.sleep(120)")
                deadline = time.monotonic()+10
                while time.monotonic() < deadline:
                    if call("/console")["status"]["running"]:
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError("Owned cancellation probe did not start")
                assert call("/console/interrupt", {})["status"] == "interrupted"
                cancelled = running.result(timeout=15)
                assert cancelled["status"] == "interrupted", cancelled
            assert not owned_scratch.exists(), "Supervised worker scratch survived Stop"
            restarted = execute("assert 'cancel_probe' not in globals()\nprint(oe.__version__)")
            assert restarted["status"] == "ok" and metadata["version"] in restarted["stdout"], restarted
            report["stop_reclaims_spilled_sqlite_and_restarts"] = True
            report["phase"] = "complete"
            report["status"] = "ok"
            return report
        except Exception as error:
            report["status"] = "error"
            report["error"] = str(error)[:10000]
            if process is not None:
                report["runtime_exit_code"] = process.poll()
            raise
        finally:
            stop()
            if stderr_path.exists():
                with stderr_path.open("rb") as stream:
                    stream.seek(max(0, stderr_path.stat().st_size-32768))
                    report["runtime_stderr"] = stream.read().decode("utf-8", errors="replace")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-version")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.runtime, arguments.output, expected_version=arguments.expected_version)))
