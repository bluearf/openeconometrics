"""Physical-file fit → saved prediction → global margins → paper-table probe.

Run in a fresh subprocess by default. The independent NumPy oracle reads only
bounded physical Parquet blocks; it does not invoke an external estimator.
The default workload is 10 million OLS rows and 1 million logit rows. This is
an opt-in benchmark, never a test-suite workload or a hardware extrapolation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/evidence/postest-streaming-10-million-2026-10-06.json"
THREADS = {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
PREDICTORS = ["x1", "x2", "x3"]
SEED = 20261006


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024**2):
            value.update(chunk)
    return value.hexdigest()


def snapshot():
    paths = [Path(__file__).resolve(), *sorted((ROOT / "src/openecon").rglob("*.py"))]
    files = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"scope": "benchmark and all OpenEcon production Python sources",
            "sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def phase(case, name, action):
    print(json.dumps({"stage": name, "estimator": case["estimator"], "rows": case["physical_rows"]}),
          file=sys.stderr, flush=True)
    started = time.perf_counter()
    result = action()
    case["phases"][name] = {"seconds": time.perf_counter() - started,
                            "cumulative_peak_process_rss_bytes": peak_rss()}
    return result


def physical_source(path, rows, block_rows, estimator):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(SEED + (estimator == "logit"))
    beta = np.array([.3, .5, -.2, .1])
    writer = None
    try:
        for start in range(0, rows, block_rows):
            count = min(block_rows, rows - start)
            x = rng.normal(size=(count, 3))
            eta = beta[0] + x @ beta[1:]
            if estimator == "ols":
                y = eta + (.5 + .3 * np.abs(x[:, 0])) * rng.normal(size=count)
            else:
                y = (rng.random(count) < 1 / (1 + np.exp(-eta))).astype(np.int64)
            frame = pd.DataFrame(x, columns=PREDICTORS,
                                 index=pd.Index(10 + 3 * np.arange(start, start + count), name="row_id"))
            frame["y"] = y
            table = pa.Table.from_pandas(frame, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=block_rows)
    finally:
        if writer is not None:
            writer.close()
    with pq.ParquetFile(path) as stored:
        assert stored.metadata.num_rows == rows
        groups = stored.metadata.num_row_groups
    return {"format": "Parquet", "physical_rows": rows, "row_groups": groups,
            "file_bytes": path.stat().st_size, "sha256": digest(path),
            "index": {"name": "row_id", "formula": "10 + 3 * original row position"},
            "generation_block_rows": block_rows, "seed": SEED + (estimator == "logit"),
            "population_beta": beta.tolist(), "complete_rows": rows}


def physical_blocks(path, block_rows):
    import numpy as np
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as reader:
        for block in reader.iter_batches(batch_size=block_rows, columns=[*PREDICTORS, "y", "row_id"]):
            frame = block.to_pandas()
            x = np.column_stack([np.ones(len(frame)), frame[PREDICTORS].to_numpy(dtype=float)])
            yield x, frame.y.to_numpy(dtype=float), frame.index.to_numpy()


def ols_oracle(path, block_rows):
    import numpy as np
    xx, xy, rows = np.zeros((4, 4)), np.zeros(4), 0
    for x, y, _ in physical_blocks(path, block_rows):
        xx += x.T @ x
        xy += x.T @ y
        rows += len(y)
    beta, bread = np.linalg.solve(xx, xy), np.linalg.inv(xx)
    meat = np.zeros((4, 4))
    for x, y, _ in physical_blocks(path, block_rows):
        residual = y - x @ beta
        leverage = np.einsum("nk,kl,nl->n", x, bread, x)
        score = x * (residual / (1 - leverage))[:, None]
        meat += score.T @ score
    return beta, bread @ meat @ bread, {"rows_read_per_pass": rows, "passes": 2,
                                        "definition": "normal equations; HC3 score sandwich using independent leverage",
                                        "condition_number_xtx": float(np.linalg.cond(xx))}


def sigmoid(eta):
    import numpy as np
    # Moderate well-conditioned synthetic indexes; no production Torch link.
    return 1 / (1 + np.exp(-eta))


def logit_oracle(path, block_rows):
    import numpy as np
    beta = np.zeros(4)
    iterations = 0
    for iterations in range(1, 21):
        score, information, rows = np.zeros(4), np.zeros((4, 4)), 0
        for x, y, _ in physical_blocks(path, block_rows):
            p = sigmoid(x @ beta)
            score += x.T @ (y - p)
            information += (x.T * (p * (1 - p))) @ x
            rows += len(y)
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    else:
        raise AssertionError("Independent Newton logit did not converge.")
    information, score = np.zeros((4, 4)), np.zeros(4)
    for x, y, _ in physical_blocks(path, block_rows):
        p = sigmoid(x @ beta)
        information += (x.T * (p * (1 - p))) @ x
        score += x.T @ (y - p)
    return beta, np.linalg.inv(information), {"rows_read_per_pass": rows, "passes": iterations + 1,
                                               "newton_iterations": iterations,
                                               "maximum_final_score_absolute": float(np.max(np.abs(score))),
                                               "definition": "independent bounded NumPy Newton score/information"}


def saved_fit(source, estimator, folder):
    import openecon as oe
    from openecon.models import ResultBundle
    options = {"covariance": "HC3"} if estimator == "ols" else {}
    fitted = getattr(oe, estimator)(data=source, y="y", x=PREDICTORS, **options)
    file = folder / f"{estimator}-result.json"
    file.write_text(fitted.model_dump_json(), encoding="utf-8")
    restored = ResultBundle.model_validate_json(file.read_text(encoding="utf-8"))
    assert type(restored) is ResultBundle  # never exercise the fitted OLS private method
    assert restored.model_dump_json() == fitted.model_dump_json()
    return restored, {"json_bytes": file.stat().st_size, "json_sha256": digest(file),
                      "restored_type": type(restored).__name__, "fitted_type": type(fitted).__name__}


def check_model(saved, beta, covariance):
    import numpy as np
    terms = [item.term for item in saved.coefficients]
    assert terms == ["Intercept", *PREDICTORS]
    actual_beta = np.array([item.estimate for item in saved.coefficients])
    actual_covariance = np.array(saved.covariance_matrix)
    np.testing.assert_allclose(actual_beta, beta, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(actual_covariance, covariance, rtol=3e-8, atol=2e-13)
    return {"maximum_beta_absolute_error": float(np.max(np.abs(actual_beta - beta))),
            "maximum_covariance_absolute_error": float(np.max(np.abs(actual_covariance - covariance))),
            "independent_beta": beta.tolist(), "independent_covariance": covariance.tolist(),
            "fitted_beta": actual_beta.tolist(), "fitted_covariance": actual_covariance.tolist()}


def critical_95(df=None):
    z = 1.959963984540054
    if df is None:
        return z
    # Independent large-df Student-t expansion; benchmark OLS df ≥ 996.
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def check_predictions(output, path, block_rows, estimator, beta, covariance, rows):
    import numpy as np
    prediction_iterator = output.iter_batches(batch_rows=block_rows)
    checked, maximum = 0, 0
    errors = dict.fromkeys(["response", "std_error", "ci_low", "ci_high"], 0.)
    try:
        for x, _, index in physical_blocks(path, block_rows):
            actual = next(prediction_iterator)
            assert actual.columns.tolist() == list(errors)
            np.testing.assert_array_equal(actual.index.to_numpy(), index)
            assert actual.index.name == "row_id"
            eta = x @ beta
            response = eta if estimator == "ols" else sigmoid(eta)
            jacobian = x if estimator == "ols" else (response * (1 - response))[:, None] * x
            se = np.sqrt(np.einsum("nk,kl,nl->n", jacobian, covariance, jacobian))
            critical = critical_95(rows - 4 if estimator == "ols" else None)
            expected = {"response": response, "std_error": se,
                        "ci_low": response - critical * se, "ci_high": response + critical * se}
            for name, value in expected.items():
                actual_values = actual[name].to_numpy(dtype=float)
                np.testing.assert_allclose(actual_values, value, rtol=3e-8, atol=3e-10)
                errors[name] = max(errors[name], float(np.max(np.abs(actual_values - value))))
            checked += len(actual)
            maximum = max(maximum, len(actual))
        assert next(prediction_iterator, None) is None
    finally:
        prediction_iterator.close()
    assert checked == rows
    return {"all_prediction_rows_exhausted": checked, "maximum_validation_block_rows": maximum,
            "index_values_and_name_preserved": True, "columns_checked": list(errors),
            "maximum_absolute_errors": errors}


def effect_oracle(path, block_rows, estimator, beta, covariance, method, settings):
    import numpy as np
    values, gradients = [], []
    for setting in settings:
        mass, design_sum = 0, np.zeros(4)
        estimate_sum, gradient_sum = np.zeros(3), np.zeros((3, 4))
        for x, _, _ in physical_blocks(path, block_rows):
            for name, value in setting.items():
                x[:, 1 + PREDICTORS.index(name)] = value
            mass += len(x)
            if method == "mem":
                design_sum += x.sum(axis=0)
                continue
            if estimator == "ols":
                estimate_sum += len(x) * beta[1:]
                gradient_sum[:, 1:] += len(x) * np.eye(3)
            else:
                p = sigmoid(x @ beta)
                derivative = p * (1 - p)
                common = (derivative * (1 - 2 * p)) @ x
                estimate_sum += beta[1:] * derivative.sum()
                gradient_sum += beta[1:, None] * common[None, :]
                gradient_sum[:, 1:] += np.eye(3) * derivative.sum()
        if method == "mem":
            x = design_sum / mass
            if estimator == "ols":
                estimates, gradient = beta[1:], np.column_stack([np.zeros(3), np.eye(3)])
            else:
                p = float(sigmoid(x @ beta))
                derivative = p * (1 - p)
                estimates = beta[1:] * derivative
                gradient = beta[1:, None] * derivative * (1 - 2 * p) * x[None, :]
                gradient[:, 1:] += derivative * np.eye(3)
        else:
            estimates, gradient = estimate_sum / mass, gradient_sum / mass
        for index, estimate in enumerate(estimates):
            values.append({"variable": PREDICTORS[index], "method": method,
                           **{f"at[{name}]": value for name, value in setting.items()},
                           "estimate": float(estimate), "std_error": math.sqrt(float(gradient[index] @ covariance @ gradient[index]))})
            gradients.append(gradient[index])
    return values, np.array(gradients)


def check_effects(actual, expected, gradients, rows):
    import numpy as np
    assert len(actual) == len(expected)
    for index, item in enumerate(expected):
        for field in ("variable", "method", *[name for name in item if name.startswith("at[")]):
            assert actual.iloc[index][field] == item[field]
    wanted = np.array([[item["estimate"], item["std_error"]] for item in expected])
    measured = actual[["estimate", "std_error"]].to_numpy(dtype=float)
    np.testing.assert_allclose(measured, wanted, rtol=3e-8, atol=3e-11)
    actual_gradient = np.array(actual.attrs["delta_gradients"])
    np.testing.assert_allclose(actual_gradient, gradients, rtol=3e-8, atol=3e-11)
    assert actual.attrs["input_evaluation_rows"] == rows
    assert actual.attrs["evaluation_rows"] == rows
    return {"rows": actual.to_dict(orient="records"), "delta_gradients": actual_gradient.tolist(),
            "maximum_effect_or_se_absolute_error": float(np.max(np.abs(measured - wanted))),
            "maximum_global_gradient_absolute_error": float(np.max(np.abs(actual_gradient - gradients))),
            "whole_source_metadata": actual.attrs}


def run_case(folder, rows, block_rows, estimator):
    import openecon as oe
    case = {"estimator": estimator, "physical_rows": rows, "phases": {}}
    path = folder / f"{estimator}.parquet"
    case["source"] = phase(case, "physical_generation_write_and_hash", lambda: physical_source(path, rows, block_rows, estimator))
    source = oe.scan(path)
    saved, serialization = phase(case, "native_fit_and_json_save_restore", lambda: saved_fit(source, estimator, folder))
    assert saved.nobs == saved.nobs_original == rows
    case["serialization"] = serialization
    case["fit_provenance"] = saved.provenance
    case["fit_inference"] = saved.inference
    oracle = ols_oracle if estimator == "ols" else logit_oracle
    beta, covariance, oracle_record = phase(case, "independent_bounded_fit_and_covariance_oracle", lambda: oracle(path, block_rows))
    case["fit_oracle"] = {**oracle_record, **check_model(saved, beta, covariance)}
    output = phase(case, "saved_predict_full_source_to_owned_output", lambda: oe.predict(saved, source, interval="mean", batch_rows=block_rows))
    case["prediction_validation"] = phase(case, "exhaust_all_predictions_with_independent_oracle", lambda output=output: check_predictions(output, path, block_rows, estimator, beta, covariance, rows))
    case["prediction_metadata"] = output.metadata
    owned = Path(output._owned_prediction_output.name)
    case["owned_prediction_disk_bytes"] = sum(file.stat().st_size for file in owned.rglob("*") if file.is_file())
    del output
    gc.collect()
    assert not owned.exists()
    case["owned_prediction_cleanup_verified"] = True
    case["margins"] = {}
    tables = []
    for label, method, at, settings in [("ame", "ame", None, [{}]), ("mem", "mem", None, [{}]),
                                        ("mem_at", "mem", {"x1": [-.5, .5]}, [{"x1": -.5}, {"x1": .5}])]:
        actual = phase(case, f"saved_margins_{label}_whole_source", lambda method=method, at=at: oe.margins(saved, PREDICTORS, data=source, method=method, at=at, batch_rows=block_rows))
        expected, gradients = phase(case, f"independent_margins_{label}_oracle", lambda method=method, settings=settings: effect_oracle(path, block_rows, estimator, beta, covariance, method, settings))
        case["margins"][label] = check_effects(actual, expected, gradients, rows)
        tables.append(str(actual.to_latex(index=False, caption=f"{estimator.upper()} {label} marginal effects", label=f"tab:{estimator}-{label}")))
    model_table = str(saved.to_latex(caption=f"{estimator.upper()} physical-file analysis", label=f"tab:{estimator}-model"))
    for source_text in [model_table, *tables]:
        assert all(token in source_text for token in (r"\toprule", r"\midrule", r"\bottomrule"))
        assert r"\input{" not in source_text
    case["paper_tables"] = {"model_latex": model_table, "margins_latex": tables,
                             "booktabs_verified": True, "compilation_verified": False,
                             "scope": "escaped publication table fragments; no external TeX compiler used"}
    case["case_final_cumulative_peak_rss_bytes"] = peak_rss()
    case["checks_passed"] = True
    return case


def worker(options):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    import openecon as oe
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = snapshot()
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "execution_layer": "source_run", "scope": "native CPU physical-file fit, saved-model postestimation and LaTeX fragments",
              "environment": {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                              "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                              "pyarrow": pa.__version__, "torch": torch.__version__, "openecon": oe.__version__,
                              "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
              "import_baseline_peak_process_rss_bytes": peak_rss(), "source_snapshot_before": before,
              "limitations": ["One run on one Mac; operating-system file cache was not purged",
                              "Measured process RSS includes imports, NumPy oracle, Arrow, outputs and allocator overhead; resource plans bound named buffers, not RSS",
                              "Synthetic well-conditioned numeric designs; these oracles do not prove Stata equivalence",
                              "No frozen-desktop-runtime, GPU, UI, cloud or all-estimator performance claim",
                              "No extrapolation beyond the actual physical row counts; LaTeX compilation not performed"], "cases": []}
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory(prefix="openecon-postest-physical-") as directory:
        folder = Path(directory)
        old_scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
        os.environ["OPENECON_SCRATCH_DIRECTORY"] = str(folder)
        try:
            report["environment"]["observed_threadpools"] = [{key: pool.get(key) for key in ("user_api", "internal_api", "num_threads")} for pool in threadpool_info()]
            report["cases"].append(run_case(folder, options.rows, options.batch_rows, "ols"))
            if options.logit_rows:
                report["cases"].append(run_case(folder, options.logit_rows, options.batch_rows, "logit"))
        finally:
            if old_scratch is None:
                os.environ.pop("OPENECON_SCRATCH_DIRECTORY", None)
            else:
                os.environ["OPENECON_SCRATCH_DIRECTORY"] = old_scratch
    report["owned_physical_fixture_cleanup_verified"] = not folder.exists()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["source_snapshot_after"] = snapshot()
    report["source_snapshot_unchanged"] = report["source_snapshot_after"] == before
    report["peak_process_rss_bytes"] = peak_rss()
    report["all_checks_passed"] = all(case["checks_passed"] for case in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--logit-rows", type=int, default=1_000_000)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    if options.rows < 1000 or options.logit_rows < 0 or 0 < options.logit_rows < 1000 or not 1 <= options.batch_rows <= 65536:
        parser.error("Use at least 1000 OLS/logit rows (or 0 to skip logit), and block rows from 1 through 65536.")
    if options.worker:
        print(json.dumps(worker(options), allow_nan=False), flush=True)
        return
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--rows", str(options.rows),
               "--logit-rows", str(options.logit_rows), "--batch-rows", str(options.batch_rows)]
    process = subprocess.run(command, env={**os.environ, **THREADS}, text=True, stdout=subprocess.PIPE, check=True)
    report = json.loads(process.stdout)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(options.output), "checks_passed": report["all_checks_passed"],
                      "source_snapshot_unchanged": report["source_snapshot_unchanged"],
                      "peak_process_rss_mib": report["peak_process_rss_bytes"] / 1024**2,
                      "physical_rows": [case["physical_rows"] for case in report["cases"]]}), flush=True)


if __name__ == "__main__":
    main()
