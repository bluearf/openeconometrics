"""Predeclared full PRIVATE calibration; passing simulation is not a theorem.

Run ``--protocol-only`` before the matrix, then ``--run 0`` through ``--run 3``.
The four runs use the same fixed 15-cell grid, two independent seed streams,
and two explicit conventions. Each cell has 500 outer and 999 inner draws.
No test is exposed by this development-only module.
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

import torch

from benchmarks.research.nardl_bounds import NULLS, ResearchFailure, research_bootstrap
from benchmarks.research.nardl_bounds_grid import ACCEPTANCE, GRID, generate_sample, wilson

ROOT = Path(__file__).resolve().parents[2]
NUMERIC_SOURCES = (
    "benchmarks/research/nardl_bounds.py",
    "benchmarks/research/nardl_bounds_grid.py",
    "benchmarks/research/nardl_bounds_calibration.py",
)
OUTER, INNER = 500, 999
ALPHAS = (0.01, 0.05, 0.1)
RUNS = (
    {
        "name": "fixed-pool-df-seed-a",
        "seed": 364729,
        "initial": "fixed_prefix",
        "recenter": "pool",
        "residual_scale": "df",
    },
    {
        "name": "fixed-pool-df-seed-b",
        "seed": 1719533379,
        "initial": "fixed_prefix",
        "recenter": "pool",
        "residual_scale": "df",
    },
    {
        "name": "block-draw-none-seed-a",
        "seed": 364729,
        "initial": "original_block",
        "recenter": "draw",
        "residual_scale": "none",
    },
    {
        "name": "block-draw-none-seed-b",
        "seed": 1719533379,
        "initial": "original_block",
        "recenter": "draw",
        "residual_scale": "none",
    },
)


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def protocol():
    """The complete protocol is fixed independently of realized test statistics."""
    work = (
        sum(3 * INNER * (cell.total - max(cell.lags)) * (sum(cell.lags) + 3) ** 2 for cell in GRID)
        * OUTER
    )
    return {
        "schema": "openecon.nardl-bounds-calibration-protocol.v1",
        "private_research_only": True,
        "inferential_validity_established": False,
        "public_bounds_inference_enabled": False,
        "grid": [asdict(cell) for cell in GRID],
        "runs": list(RUNS),
        "outer_per_cell": OUTER,
        "inner_per_null": INNER,
        "alphas": list(ALPHAS),
        "acceptance": ACCEPTANCE,
        "numerical_size_gate": "At alpha=.05, every maintained-class true-null Wilson95 upper <= .08, including full-null all-three rejection; every whole call must succeed.",
        "numerical_power_gate": "asymmetric_T160 at alpha=.05, all-three Wilson95 lower >= .7.",
        "nonlinear_validity_gate": "Still requires independent nonlinear validity/replication; a numerical pass cannot activate public inference.",
        "convention_contrast": "Two prespecified bundled conventions; not attribution to individual initialization/centering/scaling changes, and no choice by best observed rates.",
        "outer_seed_formula": "run_seed+100003*cell_index+1009*outer_index",
        "inner_seed_formula": "outer_seed+524287",
        "denominator": "All 500 planned calls; failures retained, no retries or replacement draws; Wilson intervals under both failure assignments.",
        "decision_rule": "inclusive tails, (1+extreme)/(999+1)<=alpha; all-three is intersection of the separate-null decisions",
        "critical_value_convention": "ceil((1-alpha)*B)-1 for upper tail, ceil(alpha*B)-1 for lower tail; displayed quantiles do not override the discrete tail decision",
        "scope": "case III, one raw I(1) iid-increment path, fixed supplied lags; I(2) cell is excluded stress",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in NUMERIC_SOURCES
        },
        "work_budget": {
            "per_run_refit_operation_proxy": work,
            "maximum_per_run": 150_000_000_000,
            "outer_call_batch_size": 512,
            "outer_call_tensor_memory_mb": 128,
            "torch_threads": 2,
            "no_gpu_or_streaming_claim": True,
            "tensor_budget_is_not_process_rss_or_blas_allocator_guarantee": True,
        },
    }


def true_nulls(cell):
    if not cell.within_maintained_class:
        return ()
    if cell.hypothesis_class == "all_levels_null":
        return NULLS
    if cell.hypothesis_class == "no_explanatory_levels":
        return ("explanatory_levels",)
    return ()


def rates(records, failures, planned, alpha):
    """All planned calls remain in the denominator, including total failure."""
    counts = {null: 0 for null in (*NULLS, "all_three")}
    for record in records:
        decisions = [record["tail_probabilities"][null] <= alpha for null in NULLS]
        for null, rejected in zip(NULLS, decisions, strict=True):
            counts[null] += int(rejected)
        counts["all_three"] += int(all(decisions))
    return {
        null: {
            "rejected_successful_calls": count,
            "planned_outer_denominator": planned,
            "failed_outer_calls": len(failures),
            "rate_bounds_with_failed_calls": [count / planned, (count + len(failures)) / planned],
            "wilson95_no_failed_call_rejects": wilson(count, planned),
            "wilson95_all_failed_calls_reject": wilson(count + len(failures), planned),
        }
        for null, count in counts.items()
    }


def evaluate(cells):
    violations = []
    for result in cells:
        cell = next(cell for cell in GRID if cell.name == result["specification"]["name"])
        if result["failed_outer_calls"]:
            violations.append(
                {
                    "cell": cell.name,
                    "gate": "whole_call_success",
                    "failures": result["failed_outer_calls"],
                }
            )
        screened = (
            *true_nulls(cell),
            *(("all_three",) if cell.hypothesis_class == "all_levels_null" else ()),
        )
        for null in screened:
            upper = result["rates"]["0.05"][null]["wilson95_all_failed_calls_reject"][1]
            if upper > 0.08:
                violations.append(
                    {
                        "cell": cell.name,
                        "gate": "size_at_05",
                        "null": null,
                        "wilson95_upper": upper,
                        "maximum": 0.08,
                    }
                )
        if cell.name == "asymmetric_T160":
            lower = result["rates"]["0.05"]["all_three"]["wilson95_no_failed_call_rejects"][0]
            if lower < 0.7:
                violations.append(
                    {
                        "cell": cell.name,
                        "gate": "power_at_05",
                        "wilson95_lower": lower,
                        "minimum": 0.7,
                    }
                )
    return {
        "numerical_acceptance_passed": not violations,
        "violations": violations,
        "inferential_validity_established": False,
        "acceptance_complete": False,
        "public_bounds_inference_enabled": False,
    }


def run_calibration(specification, run_index, checkpoint=None):
    if digest(specification) != digest(protocol()):
        raise ValueError(
            "Predeclared protocol or numeric sources changed; refuse to run a different specification."
        )
    if (
        isinstance(run_index, bool)
        or not isinstance(run_index, int)
        or not 0 <= run_index < len(RUNS)
    ):
        raise ValueError("Run index must be 0..3.")
    if (
        specification["work_budget"]["per_run_refit_operation_proxy"]
        > specification["work_budget"]["maximum_per_run"]
    ):
        raise ValueError("Full calibration exceeds its explicit aggregate work budget.")
    run = RUNS[run_index]
    start, cells = time.perf_counter(), []
    with torch.device("cpu"), torch.inference_mode():
        for ci, cell in enumerate(GRID):
            records, failures, dgp = [], [], None
            for oi in range(OUTER):
                outer_seed = run["seed"] + 100003 * ci + 1009 * oi
                try:
                    y, x, dgp = generate_sample(cell, outer_seed)
                    p, qp, qn = cell.lags
                    result = research_bootstrap(
                        y,
                        x,
                        p=p,
                        q_positive=qp,
                        q_negative=qn,
                        replications=INNER,
                        seed=outer_seed + 524287,
                        initial=run["initial"],
                        recenter=run["recenter"],
                        residual_scale=run["residual_scale"],
                        batch_size=512,
                        memory_mb=128,
                    )
                except (ResearchFailure, RuntimeError) as error:
                    if isinstance(error, RuntimeError):
                        error = ResearchFailure(
                            "torch_numerical",
                            "Native outer DGP or null bootstrap failed.",
                            native_error=str(error),
                        )
                    failures.append(
                        {
                            "outer_index": oi,
                            "outer_seed": outer_seed,
                            "code": error.code,
                            "message": str(error),
                            "details": error.details,
                        }
                    )
                    continue
                records.append(
                    {
                        "outer_index": oi,
                        "outer_seed": outer_seed,
                        "inner_seed": outer_seed + 524287,
                        "sample_sha256": result["input_sha256"],
                        "index_stream_sha256": result["index_stream_sha256"],
                        "draw_statistics_sha256": digest(result["draw_statistics"]),
                        "observed_statistics": result["observed_statistics"],
                        "extreme_draws": {
                            null: result["tests"][null]["extreme_draws"] for null in NULLS
                        },
                        "tail_probabilities": {
                            null: result["tests"][null]["research_tail_probability_plus_one"]
                            for null in NULLS
                        },
                    }
                )
            cells.append(
                {
                    "specification": asdict(cell),
                    "true_nulls": list(true_nulls(cell)),
                    "dgp": dgp,
                    "completed_outer_calls": len(records),
                    "failed_outer_calls": len(failures),
                    "invalid_draws_replaced": 0,
                    "rates": {
                        str(alpha): rates(records, failures, OUTER, alpha) for alpha in ALPHAS
                    },
                    "records": records,
                    "failures": failures,
                }
            )
            if checkpoint:
                checkpoint(
                    {
                        "run": run["name"],
                        "cell": cell.name,
                        "completed_cells": len(cells),
                        "completed_outer_calls": len(records),
                        "failed_outer_calls": len(failures),
                        "elapsed_seconds": time.perf_counter() - start,
                        "rates_at_05": {
                            key: value["rate_bounds_with_failed_calls"]
                            for key, value in cells[-1]["rates"]["0.05"].items()
                        },
                    }
                )
    return {
        "schema": "openecon.nardl-bounds-full-calibration.v1",
        "protocol_sha256": digest(specification),
        "run_index": run_index,
        "run": run,
        "private_research_only": True,
        "planned_outer_calls": len(GRID) * OUTER,
        "completed_outer_calls": sum(cell["completed_outer_calls"] for cell in cells),
        "failed_outer_calls": sum(cell["failed_outer_calls"] for cell in cells),
        "invalid_draws_replaced": 0,
        "elapsed_seconds": time.perf_counter() - start,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.system(),
            "device": "cpu",
            "dtype": "float64",
            "threads": torch.get_num_threads(),
        },
        "acceptance": evaluate(cells),
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--run", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.protocol_only:
        args.protocol.parent.mkdir(parents=True, exist_ok=True)
        with args.protocol.open("xb") as handle:
            handle.write(encoded(protocol()))
        print(
            json.dumps(
                {
                    "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                    "runs": len(RUNS),
                }
            )
        )
        return
    if args.run is None or args.output is None:
        parser.error("--run and --output are required after --protocol-only")
    # The pinned two-thread setting bounds resource demand. No global RNG used.
    torch.set_num_threads(2)
    specification = json.loads(args.protocol.read_text())
    result = run_calibration(
        specification, args.run, checkpoint=lambda row: print(json.dumps(row), flush=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as handle,
    ):
        handle.write(encoded(result))
    print(
        json.dumps(
            {
                "run": result["run"]["name"],
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                **result["acceptance"],
            }
        )
    )


if __name__ == "__main__":
    main()
