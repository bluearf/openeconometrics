"""Predeclared common CF targets: physical query replay and independent oracle.

generate fixes eight 100k Parquet queries and two 1m CSV queries before runs.
run uses only the installed/source public API; check imports no Torch or SDK
and compares every persisted response/derivative/CI plus global AME/MEM.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "docs/evidence/control-functions-eight-2026-10-07/persisted-states.json.gz"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


def generate(directory):
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=False)
    cases = json.loads(gzip.decompress(LEGACY.read_bytes()))["cases"]
    fixture = directory / "training.json"
    write(
        fixture,
        [
            {
                "case_id": c["case_id"],
                "kind": c["kind"],
                "inputs": c["inputs"],
                "estimator": c["result"]["spec"]["estimator"],
            }
            for c in cases
        ],
    )
    plans = []
    for case in cases[::2]:
        for rows, suffix in [
            (100000, "parquet"),
            *([(1000000, "csv")] if case["kind"] in {"gaussian", "poisson"} else []),
        ]:
            base = pd.DataFrame(case["inputs"]["data"])[["x", "d", "z1", "z2"]]
            query = (
                pd.concat([base] * math.ceil(rows / len(base)), ignore_index=True)
                .iloc[:rows]
                .copy()
            )
            query.loc[query.index % 997 == 0, "z1"] = float("nan")
            file = directory / f"{case['kind']}-{rows}.{suffix}"
            if suffix == "csv":
                query.to_csv(file, index=False)
            else:
                query.to_parquet(file, index=False)
            plans.append(
                {
                    "id": f"{case['kind']}-{rows}-{suffix}",
                    "case_id": case["case_id"],
                    "rows": rows,
                    "format": suffix,
                    "file": str(file.resolve()),
                    "sha256": digest(file),
                    "bytes": file.stat().st_size,
                    "missing_rows": int(query.z1.isna().sum()),
                }
            )
    manifest = {
        "schema": 1,
        "training": str(fixture.resolve()),
        "training_sha256": digest(fixture),
        "historical_fixture_sha256": digest(LEGACY),
        "plan": plans,
        "batch_rows": 8192,
        "model_fit_rows": 240,
        "oracle": "independent formula and parameter finite differences; no SDK/Torch import in check",
        "tolerances": {"atol": 2e-8, "rtol": 2e-7},
        "scope": "conditional mean/derivative inference and global AME/MEM; fit is resident, query is streamed",
        "vendor_parity": False,
        "cold_disk_cache": False,
        "gpu": False,
    }
    write(directory / "manifest.json", manifest)
    return manifest


def frame_record(frame):
    return {"rows": frame.to_dict("records"), "attrs": frame.attrs}


def run(manifest_path, output):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import openecon as oe
    from openecon.models import ResultBundle

    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": 1,
        "status": "running",
        "manifest_sha256": digest(manifest_path),
        "cases": [],
        "small_cases": [],
        "sdk_version": oe.__version__,
        "refit_query": False,
    }
    try:
        assert digest(manifest["training"]) == manifest["training_sha256"]
        training = json.loads(Path(manifest["training"]).read_text())
        models = {}
        for case in training:
            inputs = case["inputs"]
            data = pd.DataFrame(inputs["data"])
            options = {key: value for key, value in inputs.items() if key != "data"}
            options["missing"] = "drop"
            fitted = getattr(oe, case["estimator"])(data=data, **options)
            model_path = output / (case["case_id"] + "-model.json")
            model_path.write_text(fitted.model_dump_json())
            restored = ResultBundle.model_validate_json(model_path.read_text())
            query = data.iloc[:31].drop(columns=[inputs["y"], "cluster"], errors="ignore")
            predictions = {}
            for kind in ["response", "xb", "stdp", "derivative"]:
                predictions[kind] = frame_record(
                    oe.predict(
                        restored,
                        query,
                        kind=kind,
                        term="d" if kind == "derivative" else None,
                        interval=None if kind == "stdp" else "mean",
                    )
                )
            margins = {
                method: frame_record(
                    oe.margins(
                        restored,
                        ["x", "d", "z1"],
                        data=query,
                        method=method,
                        at={"z2": [-0.4, 0.5]},
                    )
                )
                for method in ["ame", "mem"]
            }
            record = {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "estimator": case["estimator"],
                "model": str(model_path.resolve()),
                "model_sha256": digest(model_path),
                "predictions": predictions,
                "margins": margins,
            }
            write(output / (case["case_id"] + "-small.json"), record)
            receipt["small_cases"].append(
                {
                    "id": case["case_id"],
                    "status": "passed",
                    "file": str((output / (case["case_id"] + "-small.json")).resolve()),
                }
            )
            models[case["case_id"]] = restored
        for plan in manifest["plan"]:
            item = {"id": plan["id"], "status": "running", "rows": plan["rows"], "targets": {}}
            receipt["cases"].append(item)
            write(output / "run.json", receipt)
            assert digest(plan["file"]) == plan["sha256"]
            source = oe.scan(plan["file"])
            model = models[plan["case_id"]]
            started = time.monotonic()
            for kind in ["response", "derivative"]:
                prediction = oe.predict(
                    model,
                    source,
                    kind=kind,
                    term="d" if kind == "derivative" else None,
                    interval="mean",
                    batch_rows=manifest["batch_rows"],
                )
                path = output / (plan["id"] + "-" + kind + ".parquet")
                writer, count = None, 0
                try:
                    for frame in prediction.iter_batches(batch_rows=8192):
                        block = pa.Table.from_pandas(pd.DataFrame(frame), preserve_index=True)
                        if writer is None:
                            writer = pq.ParquetWriter(path, block.schema)
                        writer.write_table(block)
                        count += len(frame)
                finally:
                    if writer is not None:
                        writer.close()
                assert count == plan["rows"]
                item["targets"][kind] = {
                    "file": str(path.resolve()),
                    "sha256": digest(path),
                    "rows": count,
                    "metadata": prediction.metadata,
                }
                owned = Path(prediction._owned_prediction_output.name)
                del prediction
                gc.collect()
                assert not owned.exists()
            item["prediction_seconds"] = time.monotonic() - started
            started = time.monotonic()
            item["margins"] = {
                method: frame_record(
                    oe.margins(model, ["x", "d", "z1"], data=source, method=method, batch_rows=8192)
                )
                for method in ["ame", "mem"]
            }
            item["margins_seconds"] = time.monotonic() - started
            assert digest(plan["file"]) == plan["sha256"]
            item.update(status="passed", scratch_cleaned=True, source_unchanged=True)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt.update(
            status="passed",
            peak_process_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
            scipy_loaded=any(name == "scipy" or name.startswith("scipy.") for name in sys.modules),
            statsmodels_loaded=any(
                name == "statsmodels" or name.startswith("statsmodels.") for name in sys.modules
            ),
            frozen=bool(getattr(sys, "frozen", False)),
        )
        assert not receipt["scipy_loaded"] and not receipt["statsmodels_loaded"]
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "run.json", receipt)
    return receipt


def oracle(kind, query, saved, p, variable=None, target="response"):
    import numpy as np

    def design(terms):
        return np.column_stack(
            [
                np.ones(len(query)) if name == "Intercept" else query[name].to_numpy(dtype=float)
                for name in terms
            ]
        )

    kz = len(saved["gamma"])
    gamma, beta = p[:kz], p[kz:]
    z, x = design(saved["z_terms"]), design(saved["x_terms"])
    residual = query.d.to_numpy() - z @ gamma
    eta = x @ beta[:-1] + residual * beta[-1]
    if target in {"xb", "stdp"} and variable is None:
        return eta
    if kind == "gaussian":
        mean, first = eta, np.ones(len(eta))
    elif kind in {"logit", "fractional_logit"}:
        mean = 1 / (1 + np.exp(-eta))
        first = mean * (1 - mean)
    elif kind == "probit":
        mean = np.array([math.erfc(-v / math.sqrt(2)) / 2 for v in eta])
        first = np.exp(-(eta**2) / 2) / math.sqrt(2 * math.pi)
    elif kind == "cloglog":
        mean, first = -np.expm1(-np.exp(eta)), np.exp(eta - np.exp(eta))
    else:
        mean = first = np.exp(eta)
    if variable is None:
        return mean
    slope = beta[saved["x_terms"].index(variable)] if variable in saved["x_terms"] else 0.0
    partial = float(variable == "d")
    if variable in saved["z_terms"]:
        partial -= gamma[saved["z_terms"].index(variable)]
    slope += beta[-1] * partial
    return first * slope


def check(manifest_path, directory):
    import numpy as np
    import pandas as pd

    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((directory / "run.json").read_text())
    training = {c["case_id"]: c for c in json.loads(Path(manifest["training"]).read_text())}
    expected = {p["id"] for p in manifest["plan"]}
    assert receipt["status"] == "passed" and {c["id"] for c in receipt["cases"]} == expected
    assert len(receipt["cases"]) == len(expected) and len(receipt["small_cases"]) == 16
    comparisons = []

    def jacobian(function, p):
        cols = []
        for j, v in enumerate(p):
            step = 1e-5 * max(1.0, abs(v))
            upper, lower = p.copy(), p.copy()
            upper[j] += step
            lower[j] -= step
            cols.append((function(upper) - function(lower)) / (2 * step))
        return np.asarray(cols).T

    def close(a, b):
        np.testing.assert_allclose(a, b, **manifest["tolerances"], equal_nan=True)

    small_checks = []
    for item in receipt["small_cases"]:
        record = json.loads(Path(item["file"]).read_text())
        assert record["case_id"] == item["id"] and item["status"] == "passed"
        model_path = Path(record["model"])
        assert digest(model_path) == record["model_sha256"]
        model = json.loads(model_path.read_text())
        state = model["extra"]["control_function_state"]
        p, cov = np.r_[state["gamma"], state["beta"]], np.asarray(state["joint_covariance"])
        case = training[item["id"]]
        kind = case["kind"]
        query = pd.DataFrame(case["inputs"]["data"]).iloc[:31]
        for target, prediction in record["predictions"].items():

            def fn(p):
                return oracle(
                    kind,
                    query,
                    state,
                    p,
                    variable="d" if target == "derivative" else None,
                    target=target,
                )

            values = fn(p)
            j = jacobian(fn, p)
            se = np.sqrt(np.einsum("nk,kl,nl->n", j, cov, j))
            columns = np.asarray(pd.DataFrame(prediction["rows"]), dtype=float)
            expected_columns = (
                se[:, None]
                if target == "stdp"
                else np.column_stack(
                    (values, se, values - 1.959963984540054 * se, values + 1.959963984540054 * se)
                )
            )
            close(columns, expected_columns)
        for method, margins in record["margins"].items():
            for grid_index, z2 in enumerate([-0.4, 0.5]):
                grid = query.assign(z2=z2)
                grid = grid if method == "ame" else grid[["x", "d", "z1", "z2"]].mean().to_frame().T
                for v_index, variable in enumerate(["x", "d", "z1"]):

                    def fn(p):
                        return np.array([oracle(kind, grid, state, p, variable=variable).mean()])

                    j = jacobian(fn, p).ravel()
                    value = float(fn(p)[0])
                    se = math.sqrt(j @ cov @ j)
                    i = grid_index * 3 + v_index
                    row = margins["rows"][i]
                    close(margins["attrs"]["delta_gradients"][i], j)
                    close(
                        [row["estimate"], row["std_error"], row["ci_low"], row["ci_high"]],
                        [value, se, value - 1.959963984540054 * se, value + 1.959963984540054 * se],
                    )
        small_checks.append(
            {"id": item["id"], "status": "passed", "full_targets_and_joint_gradients": True}
        )

    for plan in manifest["plan"]:
        item = next(c for c in receipt["cases"] if c["id"] == plan["id"])
        model = json.loads((directory / (plan["case_id"] + "-model.json")).read_text())
        state = model["extra"]["control_function_state"]
        p, cov = np.r_[state["gamma"], state["beta"]], np.asarray(state["joint_covariance"])
        kind = training[plan["case_id"]]["kind"]
        source = (
            pd.read_csv(plan["file"]) if plan["format"] == "csv" else pd.read_parquet(plan["file"])
        )
        valid = ~source.isna().any(axis=1)
        q = source.loc[valid]
        result_errors = {}
        for target in ["response", "derivative"]:
            file = Path(item["targets"][target]["file"])
            assert digest(file) == item["targets"][target]["sha256"]
            actual = pd.read_parquet(file)
            assert len(actual) == plan["rows"] and actual.index.equals(source.index)
            assert int(actual.iloc[:, 0].isna().sum()) == plan["missing_rows"]
            maxerror = 0.0
            # Offline oracle has a bounded block too; no large N-by-K Jacobian.
            for start in range(0, len(source), 8192):
                query = source.iloc[start : start + 8192]
                keep = ~query.isna().any(axis=1)
                selected = query.loc[keep]

                def fn(p):
                    return oracle(
                        kind, selected, state, p, variable="d" if target == "derivative" else None
                    )

                values = fn(p)
                j = jacobian(fn, p)
                se = np.sqrt(np.einsum("nk,kl,nl->n", j, cov, j))
                expected_block = np.column_stack(
                    (values, se, values - 1.959963984540054 * se, values + 1.959963984540054 * se)
                )
                observed = actual.iloc[start : start + 8192].loc[keep].to_numpy()
                close(observed, expected_block)
                maxerror = max(
                    maxerror, float(np.max(np.abs(observed - expected_block), initial=0))
                )
            result_errors[target] = maxerror
        for method in ["ame", "mem"]:
            query = q if method == "ame" else q.mean().to_frame().T
            rows = item["margins"][method]["rows"]
            gradients = item["margins"][method]["attrs"]["delta_gradients"]
            for i, variable in enumerate(["x", "d", "z1"]):

                def fn(p):
                    return np.array([oracle(kind, query, state, p, variable=variable).mean()])

                gradient = jacobian(fn, p).ravel()
                value = float(fn(p)[0])
                se = math.sqrt(gradient @ cov @ gradient)
                close(gradients[i], gradient)
                close(
                    [
                        rows[i]["estimate"],
                        rows[i]["std_error"],
                        rows[i]["ci_low"],
                        rows[i]["ci_high"],
                    ],
                    [value, se, value - 1.959963984540054 * se, value + 1.959963984540054 * se],
                )
        comparisons.append(
            {
                "id": plan["id"],
                "status": "passed",
                "rows": plan["rows"],
                "all_response_and_derivative_intervals": True,
                "max_absolute_errors": result_errors,
                "global_ame_mem": True,
                "joint_covariance": True,
            }
        )
    summary = {
        "status": "passed",
        "cases": comparisons,
        "physical_query_rows": sum(p["rows"] for p in manifest["plan"]),
        "small_case_count": 16,
        "small_cases": small_checks,
        "peak_process_bytes": receipt["peak_process_bytes"],
        "manifest_sha256": digest(manifest_path),
        "run_sha256": digest(directory / "run.json"),
        "oracle_imports": "NumPy/pandas offline only; no SciPy/SDK/Torch",
    }
    write(directory / "independent-check.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "run", "check"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = (
        generate(args.directory.resolve())
        if args.action == "generate"
        else run(args.manifest.resolve(), args.directory.resolve())
        if args.action == "run"
        else check(args.manifest.resolve(), args.directory.resolve())
    )
    print(
        json.dumps(
            {
                "status": result.get("status", "generated"),
                "cases": len(result.get("plan", result.get("cases", []))),
            }
        )
    )


if __name__ == "__main__":
    main()
