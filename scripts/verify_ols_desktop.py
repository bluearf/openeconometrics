"""Verify comprehensive OLS through an owned desktop runtime's real console.

Uses only synthetic data, a random loopback port and a temporary project. This
is a packaging/integration check, not a claim of external Stata parity. The
frozen-runtime mode needs no system Python, package installation or sign-in.
"""
from __future__ import annotations

import argparse
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


MARKER = "__OPENECON_OLS_DESKTOP_CHECK__"

FIXTURE = r'''
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch

torch.set_num_threads(min(2, torch.get_num_threads()))
generator = torch.Generator().manual_seed(36104)
n = 641
x = torch.randn(n, generator=generator, dtype=torch.float64)
z = torch.randn(n, generator=generator, dtype=torch.float64)
index = torch.arange(n)
frame = pd.DataFrame({
    "x": x.tolist(), "x_dup": x.tolist(), "z": z.tolist(),
    "sector": ["abcd"[i % 4] for i in range(n)],
    "firm": (index % 29).tolist(), "market": ((index // 7) % 17).tolist(),
    "third": (index % 11).tolist(), "fourth": ((index // 3) % 7).tolist(),
    "time": (index + index // 8).tolist(),
    "wf": (index % 4 + 1).tolist(),
    "wa": (.5 + torch.rand(n, generator=generator, dtype=torch.float64)).tolist(),
    "y": (1.4 + 1.2*x - .6*z + .2*x*z +
           torch.randn(n, generator=generator, dtype=torch.float64)*(.6 + .2*x.abs())).tolist(),
})
frame.loc[10, "x"] = float("nan")
frame.loc[37, "y"] = float("nan")
frame.loc[62, "sector"] = None
frame.loc[111, ["wf", "wa"]] = 0
fixture_path = Path.cwd() / "ols-owned-fixture.parquet"
frame.to_parquet(fixture_path, index=False, row_group_size=37)
source = oe.scan(fixture_path)
assert source.row_count == n
def number(value):
    return torch.as_tensor(value, dtype=torch.float64)
def close(actual, expected, rtol=2e-8, atol=2e-10, equal_nan=False):
    torch.testing.assert_close(number(actual), number(expected), rtol=rtol, atol=atol, equal_nan=equal_nan)
def coefficients(model):
    return [[c.estimate, c.std_error, c.statistic, c.p_value, c.ci_low, c.ci_high]
            for c in model.coefficients]
def compare(options, *, covariance=True):
    dense = oe.ols(data=frame, **options)
    stream = oe.ols(data=source, **options)
    assert stream.nobs == dense.nobs
    assert stream.nobs_original == dense.nobs_original == n
    assert stream.dropped_rows == dense.dropped_rows
    assert [c.term for c in stream.coefficients] == [c.term for c in dense.coefficients]
    close([c.estimate for c in stream.coefficients], [c.estimate for c in dense.coefficients])
    if covariance:
        close(stream.covariance_matrix, dense.covariance_matrix)
        close(coefficients(stream), coefficients(dense))
    for key in ("rmse", "df_resid", "r_squared", "adjusted_r_squared", "ss_resid", "log_likelihood"):
        close(stream.metrics[key], dense.metrics[key])
    assert stream.sample_positions == [] and len(stream.predictions) == 400
    assert stream.provenance["streaming"]["full_sample_positions_retained"] is False
    assert stream.provenance["streaming"]["row_limit"] is None
    assert len(stream.model_dump_json().encode()) < 150000
    assert "\\toprule" in stream.to_latex() and "\\bottomrule" in stream.to_latex()
    scratch = Path(os.environ["OPENECON_SCRATCH_DIRECTORY"])
    assert not list(scratch.rglob("*.sqlite3")), "Normal fit left an owned SQLite spill"
    error = float((number([c.estimate for c in stream.coefficients]) -
                   number([c.estimate for c in dense.coefficients])).abs().max())
    return dense, stream, {"covariance": stream.spec.covariance,
                          "weight_type": stream.spec.weight_type,
                          "nobs": stream.nobs, "physical_rows": n-stream.dropped_rows,
                          "terms": [c.term for c in stream.coefficients],
                          "coefficient_max_abs_error": error,
                          "dense_covariance_oracle": covariance,
                          "passes": stream.provenance["streaming"]["passes"],
                          "bounded_result": True, "spill_reclaimed": True}
print("__OPENECON_OLS_DESKTOP_CHECK__" + json.dumps({
    "version": oe.__version__, "distribution_version": importlib.metadata.version("openecon"),
    "fixture_rows": n, "fixture_bytes": fixture_path.stat().st_size,
    "physical_parquet": True, "external_stata_parity_claimed": False,
}))
'''

FORMULAS_WEIGHTS = r'''
records = []
for formula in ("y ~ x * C(sector) + I(z**2)", "y ~ x + x_dup + z", "y ~ L(x) + D(z)"):
    options = {"formula": formula, "covariance": "HC3"}
    if "L(" in formula:
        options["time"] = "time"
    dense, model, record = compare(options)
    if "x_dup" in formula:
        assert "x_dup" in model.provenance["omitted_terms"]
        record["collinearity_omission"] = True
    record["formula"] = formula
    records.append(record)
for covariance in ("nonrobust", "HC0", "HC1", "HC2", "HC3"):
    _, _, record = compare({"y": "y", "x": ["x", "z"], "covariance": covariance})
    records.append(record)
for weight_type in ("aweight", "fweight", "pweight", "iweight"):
    weight = "wf" if weight_type == "fweight" else "wa"
    dense, model, record = compare({"y": "y", "x": ["x", "z"], "weights": weight,
                                   "weight_type": weight_type, "covariance": "HC3"})
    if weight_type == "fweight":
        retained = frame.dropna(subset=["y", "x", "z", "wf"])
        expanded = retained.loc[retained.index.repeat(retained.wf.astype(int))]
        literal = oe.ols(data=expanded, y="y", x=["x", "z"], covariance="HC3")
        close(coefficients(model), coefficients(literal))
        close(model.covariance_matrix, literal.covariance_matrix)
        record["literal_frequency_replication"] = True
    records.append(record)
    if weight_type != "pweight":
        _, _, record = compare({"y": "y", "x": ["x", "z"], "weights": weight,
                               "weight_type": weight_type, "covariance": "nonrobust"})
        records.append(record)
_, default_pw, record = compare({"y": "y", "x": ["x", "z"], "weights": "wa", "weight_type": "pweight"})
assert default_pw.spec.covariance == "HC1"
record["pweight_default_hc1"] = True
records.append(record)
print("__OPENECON_OLS_DESKTOP_CHECK__" + json.dumps({"formula_weight_models": records}))
'''

CLUSTER_HAC = r'''
records = []
for dimensions in (1, 2, 3, 4):
    clusters = ["firm", "market", "third", "fourth"][:dimensions]
    _, model, record = compare({"y": "y", "x": ["x", "z"], "covariance": "cluster", "cluster": clusters})
    assert model.inference["cluster_dimensions"] == dimensions
    record["cluster_dimensions"] = dimensions
    records.append(record)
for covariance, adjustment in (("HC2", "dfadjust"), ("HC3", "hansen"),
                               ("cluster_hc2", "dfadjust"), ("cluster_hc3", "hansen")):
    options = {"y": "y", "x": ["x", "z"], "covariance": covariance, adjustment: True}
    if covariance.startswith("cluster"):
        options["cluster"] = "firm"
    dense, model, record = compare(options)
    actual, expected = model.lincom({"x": 1., "z": -.3}), dense.lincom({"x": 1., "z": -.3})
    for key in ("std_error", "statistic", "df", "p_value", "ci_low", "ci_high", "statistic_scale"):
        close(actual[key], expected[key])
    assert model.inference[adjustment] is True
    record["contrast_adjustment"] = adjustment
    records.append(record)
for kernel in ("bartlett", "parzen", "truncated", "quadratic_spectral"):
    _, model, record = compare({"y": "y", "x": ["x", "z"], "covariance": "hac",
                               "time": "time", "kernel": kernel, "lags": 4})
    record["kernel"] = kernel
    records.append(record)
for kernel in ("bartlett", "parzen", "quadratic_spectral"):
    _, model, record = compare({"y": "y", "x": ["x", "z"], "covariance": "hac",
                               "time": "time", "kernel": kernel, "lags": "auto"})
    assert model.inference["automatic_lags"] is True
    record.update(kernel=kernel, automatic_lags=True)
    records.append(record)
print("__OPENECON_OLS_DESKTOP_CHECK__" + json.dumps({"cluster_hac_models": records}))
'''

RESAMPLING = r'''
records = []
for cluster in (None, "firm"):
    options = {"y": "y", "x": ["x", "z"], "covariance": "jackknife", "cluster": cluster}
    _, _, record = compare(options)
    record["resampling_units"] = "clusters" if cluster else "rows"
    records.append(record)
options = {"y": "y", "x": ["x", "z"], "covariance": "bootstrap", "weights": "wf",
           "weight_type": "fweight", "reps": 13, "seed": 361}
_, model, record = compare(options)
assert model.inference["successful_reps"] == 13 and model.inference["distribution"] == "normal"
record["exact_frequency_bootstrap_same_seed"] = True
records.append(record)
for cluster in (None, "firm"):
    options = {"y": "y", "x": ["x", "z"], "covariance": "bootstrap", "cluster": cluster,
               "reps": 13, "seed": 362}
    _, model, record = compare(options, covariance=False)
    repeated = oe.ols(data=source, **options)
    close(model.covariance_matrix, repeated.covariance_matrix, rtol=0., atol=0.)
    eigenvalues = torch.linalg.eigvalsh(number(model.covariance_matrix))
    assert float(eigenvalues.min()) >= -1e-12 and float(eigenvalues.max()) > 0
    assert model.inference["successful_reps"] == 13
    record.update(resampling_units="clusters" if cluster else "pairs",
                  exact_stream_repeatability=True, reps=13,
                  oracle_note="Exact resampling distribution; dense uses a different random draw sequence.")
    records.append(record)
print("__OPENECON_OLS_DESKTOP_CHECK__" + json.dumps({"resampling_models": records}))
'''

POSTESTIMATION = r'''
dense, model, _ = compare({"formula": "y ~ x * C(sector) + I(z**2)", "covariance": "HC3"})
contrast = {"x": 2., "I(z**2)": -.3}
tests = {}
for name, calculate in (("lincom", lambda m: m.lincom(contrast)),
                        ("nlcom", lambda m: m.nlcom(lambda b: b["x"] / b["I(z**2)"])),
                        ("joint_test", lambda m: m.test(["x", "I(z**2)"])),
                        ("testparm", lambda m: m.testparm("sector*"))):
    expected, actual = calculate(dense), calculate(model)
    keys = [key for key in ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high")
            if key in actual]
    for key in keys:
        close(actual[key], expected[key])
    tests[name] = {key: actual[key] for key in keys}
shifted_checks = []
for covariance in ("nonrobust", "HC3", "cluster"):
    options = {"y": "y", "x": ["x", "z"], "covariance": covariance}
    if covariance == "cluster":
        options["cluster"] = "firm"
    reference = oe.ols(data=frame, **options)
    shifted = frame.assign(x=frame.x + 1e8, z=frame.z - 2e8, y=frame.y + 1e8)
    shifted_path = Path.cwd() / "ols-owned-shifted.parquet"
    shifted.to_parquet(shifted_path, index=False, row_group_size=37)
    for execution, data in (("dense", shifted), ("source", oe.scan(shifted_path))):
        current = oe.ols(data=data, **options)
        mapping = {"Intercept": 1., "x": 1e8, "z": -2e8}
        actual = current.lincom(mapping, constant=-1e8)
        expected = reference.lincom({"Intercept": 1.})
        for key in ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"):
            close(actual[key], expected[key], rtol=2e-6, atol=2e-7)
        actual = current.test([mapping, {"x": 1.}], value=[1e8, 0.])
        expected = reference.test(["Intercept", "x"])
        for key in ("statistic", "p_value"):
            close(actual[key], expected[key], rtol=2e-6, atol=2e-7)
        close(current.predict(data=shifted.iloc[:8], kind="stdp").to_numpy(),
              reference.predict(data=frame.iloc[:8], kind="stdp").to_numpy(),
              rtol=2e-6, atol=2e-7, equal_nan=True)
        shifted_checks.append({"covariance": covariance, "execution": execution,
                               "lincom_joint_wald_prediction": True, "offset": 1e8})
for method in ("ame", "mem"):
    actual = model.margins(method=method, at={"x": [-.5, 1.]})
    expected = dense.margins(method=method, at={"x": [-.5, 1.]})
    assert actual.variable.tolist() == expected.variable.tolist()
    close(actual[["estimate", "std_error", "ci_low", "ci_high"]].to_numpy(),
          expected[["estimate", "std_error", "ci_low", "ci_high"]].to_numpy())
    tests["margins_" + method] = {"rows": len(actual), "dense_oracle": True}
prediction_checks = []
prediction_dense, prediction_model, _ = compare({"formula": "y ~ x * C(sector) + I(z**2)"})
for kind in ("xb", "residual", "stdp", "stdf", "stdr", "leverage", "cook", "dfbeta"):
    count, max_rows = 0, 0
    expected = prediction_dense.predict(kind=kind).reindex(frame.index)
    for part in prediction_model.iter_predict(batch_rows=19, kind=kind):
        assert len(part) <= 19
        positions = part.attrs["physical_positions"]
        assert positions == list(range(count, count+len(part)))
        close(part.to_numpy(), expected.iloc[positions].to_numpy(), equal_nan=True)
        count += len(part)
        max_rows = max(max_rows, len(part))
    assert count == n > 400
    prediction_checks.append({"kind": kind, "rows": count, "max_batch_rows": max_rows})
try:
    model.predict()
except oe.AnalysisError as error:
    assert "iter_predict" in str(error)
else:
    raise AssertionError("Streaming predict allocated a full result")
close(model.predict(data=frame.iloc[:3]).to_numpy(), dense.predict(data=frame.iloc[:3]).to_numpy())
diagnostic_dense, diagnostic_model, _ = compare({"y": "y", "x": ["x", "z"], "time": "time"})
expected, actual = diagnostic_dense.diagnostics(lags=2), diagnostic_model.diagnostics(lags=2)
for name in ("breusch_pagan", "white", "reset", "breusch_godfrey", "jarque_bera", "durbin_watson"):
    assert actual[name].get("available", True), actual[name]
    for key in ("statistic", "p_value"):
        if key in actual[name]:
            close(actual[name][key], expected[name][key])
close(actual["vif"].vif.to_numpy(), expected["vif"].vif.to_numpy())
assert actual["influence"]["available"] and actual["influence"]["rows"] > 400
latex = model.to_latex()
assert all(command in latex for command in ("\\toprule", "\\midrule", "\\bottomrule"))
roundtrip = oe.ResultBundle.model_validate(model.model_dump())
assert roundtrip.nobs == model.nobs
display(model)
display(model.margins(variables=["x"]))
print("__OPENECON_OLS_DESKTOP_CHECK__" + json.dumps({
    "postestimation": tests, "prediction_batches": prediction_checks,
    "shifted_postestimation": shifted_checks,
    "full_sample_diagnostics": {"physical_rows": actual["influence"]["rows"],
                                "methods": list(actual), "dense_oracle": True},
    "latex_booktabs": True, "model_serialization_roundtrip": True,
}))
'''

ORDER = r'''
print("OLS_ORDER_1")
print("Türkçe: ölçüm, ücret, gözlem")
display(frame.head(2))
print("OLS_ORDER_2")
display(model)
print("OLS_ORDER_3")
display(oe.plot.scatter(data=frame.head(8), x="x", y="y"))
print("OLS_ORDER_4")
assert model.nobs > 400
saved_ols_version = oe.__version__
print("__OPENECON_OLS_DESKTOP_CHECK__" + json.dumps({"source_order": True, "persistent_variables": True}))
'''


def verify(runtime: Path | None, output: Path, *, expected_version: str | None = None) -> dict:
    runtime = runtime.expanduser().resolve(strict=True) if runtime else None
    output = output.expanduser().resolve()
    report = {"status": "running", "runtime": str(runtime) if runtime else "source",
              "human_data_access": False, "system_python_required": runtime is None,
              "package_installation_required": False, "checks": {}, "console_runs": []}
    with tempfile.TemporaryDirectory(prefix="openecon-ols-desktop-check-") as directory:
        root = Path(directory)
        stderr_path = root / "stderr.log"
        process = descriptor = token = None
        stderr_file = stderr_path.open("wb")
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"

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

        def checked(code):
            response = call("/console/execute", {"code": code, "timeout_seconds": 180})
            if response["status"] != "ok":
                raise RuntimeError(json.dumps(response.get("error"), ensure_ascii=False)[:10000])
            assert response.get("completed_at") and response.get("id"), "Missing console completion record"
            snapshot = call("/console")
            assert snapshot["status"]["running"] is False
            persisted = next(item for item in snapshot["history"] if item["id"] == response["id"])
            assert persisted["status"] == "ok" and persisted["completed_at"] == response["completed_at"]
            markers = [line[len(MARKER):] for line in response["stdout"].splitlines() if line.startswith(MARKER)]
            assert len(markers) == 1, "Expected one owned structured result marker"
            record = json.loads(markers[0])
            report["console_runs"].append({"phase": report["phase"], "status": response["status"],
                                           "run_complete": True, "history_persisted": True,
                                           "duration_ms": response["duration_ms"]})
            return record, response

        def stop():
            if process is None:
                return
            if descriptor is not None and token is not None and process.poll() is None:
                try:
                    call("/console/reset", {}, timeout=10)
                except (OSError, RuntimeError):
                    pass
            if process.poll() is None:
                try:
                    process.stdin.write(b'{"type":"shutdown"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait(timeout=3)
            process.stdin.close()
            process.stdout.close()
            report["owned_runtime_stopped"] = process.poll() is not None

        try:
            report["phase"] = "startup"
            environment = os.environ.copy()
            if runtime:
                environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            command = [str(runtime)] if runtime else [sys.executable, "-m", "openecon.desktop_entry"]
            process = subprocess.Popen(command + ["--port", "0", "--data-root", str(root)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file,
                                       env=environment, start_new_session=os.name == "posix")
            started = time.monotonic()
            deadline = started + 60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], min(1, max(0, deadline-time.monotonic())))[0]:
                    line = process.stdout.readline(8193)
                    if not line or len(line) > 8192:
                        raise RuntimeError("Owned runtime returned an invalid startup descriptor")
                    descriptor = json.loads(line)
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Owned runtime exited at startup: {process.returncode}")
            if descriptor is None or descriptor.get("type") != "ready":
                raise RuntimeError("Owned runtime did not become ready within 60 seconds")
            parsed = urlparse(descriptor["url"])
            assert parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            assert parsed.port and parsed.port == descriptor["port"]
            report["startup_seconds"] = round(time.monotonic()-started, 3)
            token = call("/session")["token"]
            for phase, code in (("fixture", FIXTURE), ("formulas and weights", FORMULAS_WEIGHTS),
                                ("clusters and HAC", CLUSTER_HAC), ("resampling", RESAMPLING),
                                ("postestimation", POSTESTIMATION), ("source output order", ORDER)):
                report["phase"] = phase
                record, response = checked(code)
                report.update(record)
                if phase == "fixture":
                    if expected_version:
                        assert record["version"] == expected_version, record
                    if runtime:
                        assert record["distribution_version"] == record["version"], record
                        report["checks"]["embedded_distribution_metadata"] = True
                if phase == "postestimation":
                    assert [item["type"] for item in response["outputs"]] == ["model", "table"]
                    assert all("\\toprule" in item.get("latex", "") for item in response["outputs"])
                    report["checks"]["console_model_and_dataframe_latex"] = True
                if phase == "source output order":
                    assert [event["type"] for event in response["events"]] == [
                        "stdout", "output", "stdout", "output", "stdout", "output", "stdout"]
                    assert [event["index"] for event in response["events"] if event["type"] == "output"] == [0, 1, 2]
                    assert [item["type"] for item in response["outputs"]] == ["table", "model", "plot"]
                    for ordinal, event in zip(range(1, 5), response["events"][::2], strict=True):
                        assert event["text"].startswith(f"OLS_ORDER_{ordinal}\n")
                    assert "Türkçe: ölçüm, ücret, gözlem" in response["events"][0]["text"]
                    report["checks"]["utf8_output_preserved"] = True
                    report["checks"]["interleaved_text_table_model_chart_order"] = True
                report["checks"][phase] = True
                print(f"Verified desktop OLS: {phase}", flush=True)
            report["phase"] = "persistent state"
            persisted, _ = checked("assert model.nobs > 400\nassert saved_ols_version == oe.__version__\n"
                                   "print('" + MARKER + "' + json.dumps({'persistent_state': True}))")
            report.update(persisted)
            report["checks"]["run_complete_and_history"] = True
            report["phase"] = "complete"
            report["status"] = "ok"
        except Exception as error:
            report.update(status="error", error=f"{type(error).__name__}: {error}"[:12000])
            raise
        finally:
            try:
                stop()
            finally:
                stderr_file.close()
                if report["status"] == "error":
                    with stderr_path.open("rb") as stream:
                        stream.seek(max(0, stderr_path.stat().st_size-8000))
                        report["stderr_tail"] = stream.read().decode("utf-8", errors="replace")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    report["owned_temporary_data_removed"] = not root.exists()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, help="Frozen desktop Python executable; omit for the source runtime.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-version")
    arguments = parser.parse_args()
    result = verify(arguments.runtime, arguments.output, expected_version=arguments.expected_version)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))
