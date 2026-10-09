"""Post-hoc failure diagnosis with wholly independent NumPy DGP/RNG/SVD.

This does not replace or tune the frozen full calibration. The full-null T80
cell was selected after observing a failed marginal-size gate. Two NEW outer
seed streams and both prespecified conventions use 500 outer / 999 inner each.
An independent rejection-rate replication can corroborate failure, not prove
that a redesigned test is valid. NumPy is a development oracle, not runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from benchmarks.research.nardl_bounds_numpy_oracle import NULLS, bootstrap, sample_from_shocks


def wilson(count, denominator):
    z, p = 1.959963984540054, count / denominator
    d = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / d
    half = z * np.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / d
    return [float(max(0, center - half)), float(min(1, center + half))]


def run(checkpoint=None):
    from benchmarks.research.nardl_bounds_grid import GRID

    root = Path(__file__).resolve().parents[2]
    cell = GRID[0]
    if cell.name != "null_T80" or cell.lags != (1, 1, 1):
        raise ValueError("Diagnostic is pinned to the original full-null T80 cell.")
    source_files = (
        "benchmarks/research/nardl_bounds_numpy_oracle.py",
        "benchmarks/research/nardl_bounds_independent_size.py",
        "benchmarks/research/nardl_bounds_grid.py",
    )
    specification = {
        "schema": "openecon.nardl-independent-size-diagnostic-protocol.v1",
        "post_hoc_cell_selection": "null_T80 chosen after the native matrix first-cell size failure; no tuning or replacement of that protocol",
        "grid": asdict(cell),
        "outer_per_run": 500,
        "inner_per_null": 999,
        "seeds": [9164927, 1295130197],
        "alphas": [0.01, 0.05, 0.1],
        "conventions": [
            {"initial": "fixed_prefix", "recenter": "pool", "residual_scale": "df"},
            {"initial": "original_block", "recenter": "draw", "residual_scale": "none"},
        ],
        "outer_seed_formula": "seed+1009*outer_index",
        "inner_seed_formula": "outer_seed+524287",
        "rng": "NumPy PCG64, independent of Torch streams; samples and index streams are newly generated",
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files
        },
        "invalid_draw_replacement": 0,
        "failure_denominator": "all500plannedcalls; failure assignment bounds",
        "cpu_scope": "predeclared80rows/999replications; bounded small resident development experiment; no native-install, CUDA or streaming claim",
        "refit_operation_proxy": 4 * 500 * 3 * 999 * 79 * 36,
        "aggregate_proxy_budget": 20_000_000_000,
    }
    if specification["refit_operation_proxy"] > specification["aggregate_proxy_budget"]:
        raise ValueError("Independent diagnostic exceeds aggregate work budget.")
    protocol_hash = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    start, results = time.perf_counter(), []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for seed in specification["seeds"]:
            for convention in specification["conventions"]:
                records, failures = [], []
                for oi in range(500):
                    outer_seed = seed + 1009 * oi
                    try:
                        rng = np.random.default_rng(outer_seed)
                        y, raw = sample_from_shocks(
                            cell, rng.standard_normal(144), rng.standard_normal(144)
                        )
                        rng = np.random.default_rng(outer_seed + 524287)
                        starts = (
                            rng.integers(80, size=999)
                            if convention["initial"] == "original_block"
                            else np.zeros(999, dtype=int)
                        )
                        indices = rng.integers(79, size=(999, 79))
                        result = bootstrap(
                            y,
                            raw,
                            (1, 1, 1),
                            indices,
                            starts,
                            recenter=convention["recenter"],
                            residual_scale=convention["residual_scale"],
                        )
                        if not all(
                            np.isfinite(value["draw_statistics"]).all()
                            for value in result["nulls"].values()
                        ):
                            raise ValueError(
                                "Nonfinite independent statistic; invalidate entire call."
                            )
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        failures.append(
                            {
                                "outer_index": oi,
                                "outer_seed": outer_seed,
                                "error": str(error),
                                "whole_call_invalid": True,
                            }
                        )
                        continue
                    records.append(
                        {
                            "outer_index": oi,
                            "outer_seed": outer_seed,
                            "inner_seed": outer_seed + 524287,
                            "sample_sha256": hashlib.sha256(
                                np.column_stack((y, raw)).astype("<f8").tobytes()
                            ).hexdigest(),
                            "index_stream_sha256": hashlib.sha256(
                                indices.astype("<i8").tobytes()
                            ).hexdigest(),
                            "observed_statistics": result["observed_statistics"].tolist(),
                            "tail_probabilities": {
                                null: result["nulls"][null]["tail_probability"] for null in NULLS
                            },
                        }
                    )
                rates = {}
                for alpha in specification["alphas"]:
                    counts = {null: 0 for null in (*NULLS, "all_three")}
                    for record in records:
                        decisions = [record["tail_probabilities"][null] <= alpha for null in NULLS]
                        for null, rejected in zip(NULLS, decisions, strict=True):
                            counts[null] += int(rejected)
                        counts["all_three"] += int(all(decisions))
                    rates[str(alpha)] = {
                        null: {
                            "rejected_successful_calls": count,
                            "planned_outer_denominator": 500,
                            "failed_outer_calls": len(failures),
                            "rate_bounds_with_failed_calls": [
                                count / 500,
                                (count + len(failures)) / 500,
                            ],
                            "wilson95_no_failed_call_rejects": wilson(count, 500),
                            "wilson95_all_failed_calls_reject": wilson(count + len(failures), 500),
                        }
                        for null, count in counts.items()
                    }
                results.append(
                    {
                        "seed": seed,
                        "convention": convention,
                        "records": records,
                        "failures": failures,
                        "completed_outer_calls": len(records),
                        "failed_outer_calls": len(failures),
                        "rates": rates,
                    }
                )
                if checkpoint:
                    checkpoint(
                        {
                            "seed": seed,
                            "convention": convention,
                            "completed_outer_calls": len(records),
                            "failed_outer_calls": len(failures),
                            "rates_at_05": {
                                null: item["rate_bounds_with_failed_calls"]
                                for null, item in rates["0.05"].items()
                            },
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    )
    return {
        "schema": "openecon.nardl-bounds-independent-size-diagnostic.v1",
        "protocol": specification,
        "protocol_sha256": protocol_hash,
        "private_research_only": True,
        "inferential_validity_established": False,
        "acceptance_complete": False,
        "public_bounds_inference_enabled": False,
        "post_hoc_diagnostic_only": True,
        "planned_outer_calls": 2000,
        "completed_outer_calls": sum(item["completed_outer_calls"] for item in results),
        "failed_outer_calls": sum(item["failed_outer_calls"] for item in results),
        "invalid_draws_replaced": 0,
        "elapsed_seconds": time.perf_counter() - start,
        "runtime": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.system(),
            "device": "cpu",
            "dtype": "float64",
        },
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(checkpoint=lambda row: print(json.dumps(row), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as handle,
    ):
        handle.write(
            (
                json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
        )


if __name__ == "__main__":
    main()
