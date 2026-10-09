"""Audit saved whole-call denominators and rebuild the full research receipt."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

from benchmarks.research import nardl_bounds_calibration as calibration

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "docs/evidence/nardl-bounds"


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads(
        gzip.decompress((ROOT / path).read_bytes())
        if path.endswith(".gz")
        else (ROOT / path).read_text()
    )


def validate_cell(cell, ci, run):
    if calibration.digest(cell["specification"]) != calibration.digest(
        calibration.protocol()["grid"][ci]
    ):
        raise ValueError("Cell specification changed.")
    records, failures = cell["records"], cell["failures"]
    positions = [row["outer_index"] for row in (*records, *failures)]
    if sorted(positions) != list(range(500)):
        raise ValueError("Every planned outer call must occur exactly once, including failures.")
    for row in (*records, *failures):
        if row["outer_seed"] != run["seed"] + 100003 * ci + 1009 * row["outer_index"]:
            raise ValueError("Outer seed changed.")
    for row in records:
        if row["inner_seed"] != row["outer_seed"] + 524287:
            raise ValueError("Inner seed changed.")
        for null in calibration.NULLS:
            extreme = row["extreme_draws"][null]
            if isinstance(extreme, bool) or not isinstance(extreme, int) or not 0 <= extreme <= 999:
                raise ValueError("Invalid saved discrete tail count.")
            if row["tail_probabilities"][null] != (1 + extreme) / 1000:
                raise ValueError("Tail probability disagrees with the actual saved count.")
    if cell["completed_outer_calls"] != len(records) or cell["failed_outer_calls"] != len(failures):
        raise ValueError("Saved completed/failed totals disagree with actual records.")
    if cell["invalid_draws_replaced"] != 0:
        raise ValueError("Replaced invalid draws cannot be hidden.")
    for alpha in calibration.ALPHAS:
        if cell["rates"][str(alpha)] != calibration.rates(records, failures, 500, alpha):
            raise ValueError("Saved rejection summary disagrees with whole-call records.")


def receipt():
    protocol_path = f"{PREFIX}-protocol-2026-10-07.json"
    specification = read(protocol_path)
    if calibration.digest(specification) != calibration.digest(calibration.protocol()):
        raise ValueError("Frozen numeric source pins no longer match.")
    paths = [f"{PREFIX}-calibration-2026-10-07-run-{index}.json.gz" for index in range(4)]
    results, summaries = [read(path) for path in paths], []
    for index, result in enumerate(results):
        if result["run_index"] != index or result["run"] != specification["runs"][index]:
            raise ValueError("Unexpected run/seed/convention.")
        if (
            result["protocol_sha256"] != calibration.digest(specification)
            or len(result["cells"]) != 15
        ):
            raise ValueError("Incomplete or differently pinned full calibration.")
        for ci, cell in enumerate(result["cells"]):
            validate_cell(cell, ci, result["run"])
        if (
            result["planned_outer_calls"] != 7500
            or result["completed_outer_calls"]
            != sum(cell["completed_outer_calls"] for cell in result["cells"])
            or result["failed_outer_calls"]
            != sum(cell["failed_outer_calls"] for cell in result["cells"])
            or result["invalid_draws_replaced"] != 0
        ):
            raise ValueError("Saved full-run totals disagree with the actual whole-call records.")
        assessed = calibration.evaluate(result["cells"])
        if assessed != result["acceptance"]:
            raise ValueError("Saved acceptance disagrees with recomputed gates.")
        summaries.append(
            {
                "path": paths[index],
                "sha256": sha(paths[index]),
                "run": result["run"],
                "elapsed_seconds": result["elapsed_seconds"],
                "runtime": result["runtime"],
                "planned_outer_calls": 7500,
                "completed_outer_calls": sum(
                    cell["completed_outer_calls"] for cell in result["cells"]
                ),
                "failed_outer_calls": sum(cell["failed_outer_calls"] for cell in result["cells"]),
                "acceptance": assessed,
                "cells": [
                    {
                        key: value
                        for key, value in cell.items()
                        if key not in {"records", "failures", "dgp"}
                    }
                    for cell in result["cells"]
                ],
            }
        )
    for first, second in ((0, 2), (1, 3)):
        for ci in range(15):
            a = {
                row["outer_index"]: row["sample_sha256"]
                for row in results[first]["cells"][ci]["records"]
            }
            b = {
                row["outer_index"]: row["sample_sha256"]
                for row in results[second]["cells"][ci]["records"]
            }
            if any(a[index] != b[index] for index in a.keys() & b.keys()):
                raise ValueError("Bundled convention contrast failed to hold the outer DGP fixed.")
    oracle_path, independent_path = (
        f"{PREFIX}-independent-oracle-2026-10-07.json",
        f"{PREFIX}-independent-size-2026-10-07.json.gz",
    )
    oracle, independent = read(oracle_path), read(independent_path)
    if oracle["source_sha256"] != sha("benchmarks/research/nardl_bounds_numpy_oracle.py"):
        raise ValueError("Independent numerical oracle source pin changed.")
    for path, expected in independent["protocol"]["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Independent size-diagnostic source pin changed.")
    diagnostic_runs = []
    for run in independent["runs"]:
        if sorted(row["outer_index"] for row in (*run["records"], *run["failures"])) != list(
            range(500)
        ):
            raise ValueError("Independent failure diagnosis lost planned calls.")
        if run["completed_outer_calls"] != len(run["records"]) or run["failed_outer_calls"] != len(
            run["failures"]
        ):
            raise ValueError("Independent diagnostic counts disagree with whole-call records.")
        if run["rates"] != {
            str(alpha): calibration.rates(run["records"], run["failures"], 500, alpha)
            for alpha in calibration.ALPHAS
        }:
            raise ValueError("Independent diagnostic summaries disagree with records.")
        diagnostic_runs.append(
            {key: value for key, value in run.items() if key not in {"records", "failures"}}
        )
    if (
        independent["planned_outer_calls"] != 2000
        or independent["completed_outer_calls"]
        != sum(run["completed_outer_calls"] for run in diagnostic_runs)
        or independent["failed_outer_calls"]
        != sum(run["failed_outer_calls"] for run in diagnostic_runs)
        or independent["invalid_draws_replaced"] != 0
    ):
        raise ValueError("Saved independent totals disagree with actual records.")
    sources = (
        *calibration.NUMERIC_SOURCES,
        "benchmarks/research/nardl_bounds_numpy_oracle.py",
        "benchmarks/research/nardl_bounds_independent_size.py",
        "benchmarks/research/nardl_bounds_receipt.py",
        "tests/test_econ_nardl_bounds_research.py",
        "tests/test_econ_nardl_bounds_calibration.py",
        "docs/econometrics/nardl_bounds_research.md",
    )
    logs = {
        kind: f"{PREFIX}-validation-2026-10-07-{kind}.log.gz"
        for kind in ("tests", "ruff-check", "ruff-format")
    }
    test_log = gzip.decompress((ROOT / logs["tests"]).read_bytes()).decode()
    found = re.search(r"(\d+) passed in ([0-9.]+)s", test_log)
    if not found or " failed" in test_log or " error" in test_log:
        raise ValueError("Saved focused numeric suite did not pass.")
    if (
        "All checks passed!"
        not in gzip.decompress((ROOT / logs["ruff-check"]).read_bytes()).decode()
    ):
        raise ValueError("Saved source lint proof did not pass.")
    if (
        "5 files already formatted"
        not in gzip.decompress((ROOT / logs["ruff-format"]).read_bytes()).decode()
    ):
        raise ValueError("Saved source formatting proof did not pass.")
    return {
        "schema": "openecon.nardl-bounds-calibration-review.v1",
        "date": "2026-10-07",
        "status": {
            "private_research_only": True,
            "acceptance_complete": False,
            "numerical_acceptance_passed": all(
                run["acceptance"]["numerical_acceptance_passed"] for run in summaries
            ),
            "inferential_validity_established": False,
            "public_bounds_inference_enabled": False,
            "public_sdk_modified": False,
            "closure_blocker": "Predeclared marginal-null size screens fail; independent nonlinear validity remains unestablished. Do not close MARKET-112 as implemented.",
        },
        "protocol": {"path": protocol_path, "sha256": sha(protocol_path)},
        "source_sha256": {path: sha(path) for path in sources},
        "validation": {
            "focused_tests_passed": int(found[1]),
            "focused_tests_elapsed_seconds": float(found[2]),
            "ruff_check_passed": True,
            "ruff_format_passed": True,
            "logs": {kind: {"path": path, "sha256": sha(path)} for kind, path in logs.items()},
        },
        "full_matrix": {
            "planned_outer_calls": 30000,
            "completed_outer_calls": sum(run["completed_outer_calls"] for run in summaries),
            "failed_outer_calls": sum(run["failed_outer_calls"] for run in summaries),
            "inner_unrestricted_refits_if_all_calls_succeed": 30000 * 999 * 3,
            "invalid_draws_replaced": 0,
            "convention_samples_paired": True,
            "all_counts_rates_and_source_pins_recomputed": True,
            "runs": summaries,
        },
        "independent_numerical_oracle": {
            "path": oracle_path,
            "sha256": sha(oracle_path),
            "records": oracle["records"],
            "inferential_validity_established": False,
        },
        "independent_size_diagnostic": {
            "path": independent_path,
            "sha256": sha(independent_path),
            "post_hoc_diagnostic_only": True,
            "planned_outer_calls": 2000,
            "completed_outer_calls": independent["completed_outer_calls"],
            "failed_outer_calls": independent["failed_outer_calls"],
            "invalid_draws_replaced": 0,
            "runs": diagnostic_runs,
        },
        "limits": [
            "CPU float64 small resident research; no runtime NumPy/SciPy backend, native-install/CUDA/GPU/streaming validation",
            "caseIII one supplied rawI1 iid-increment predictor, fixedlags only; I2 stress excluded",
            "No tested new calibration, public p-values, tables or automatic decision; numerical equality cannot establish inferential validity",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = receipt()
    with args.output.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(value["status"]))


if __name__ == "__main__":
    main()
