"""Predeclared bounded HKSJ coverage grid; every failure remains a miss."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import openecon as oe
from openecon.analysis_contracts import AnalysisError

SEEDS = (730194, 910197)
REPS = 400
GATE = (.91, .99)
GRID = ((10, 0.), (10, .3), (30, 0.), (30, .3))


def validate():
    torch.set_num_threads(1)
    results = []
    for seed in SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for k, tau in GRID:
            covered = failed = 0
            errors = {}
            for _ in range(REPS):
                y = .4+torch.randn(k, generator=generator, dtype=torch.float64)*math.sqrt(tau+.2)
                try:
                    fit = oe.meta_pool(data={"yi": y.tolist(), "vi": [.2]*k}, method="REML", inference="hksj")
                    row = fit["coefficients"].iloc[0]
                    covered += row.ci_low <= .4 <= row.ci_high
                except AnalysisError as error:
                    failed += 1
                    errors[error.code] = errors.get(error.code, 0)+1
            rate = covered/REPS
            results.append({"seed": seed, "k": k, "tau2_true": tau, "vi": .2, "planned": REPS,
                            "completed": REPS-failed, "failed": failed, "failure_codes": errors,
                            "coverage_including_failures_as_misses": rate,
                            "binomial_mcse": math.sqrt(rate*(1-rate)/REPS),
                            "passed": GATE[0] <= rate <= GATE[1] and failed == 0})
            print(json.dumps(results[-1]), flush=True)
    return {"status": "passed" if all(r["passed"] for r in results) else "failed", "gate_predeclared": GATE,
            "method": "REML+hksj", "level": .95, "grid": results,
            "scope": "Independent normal studies with known equal sampling variances; 8 cells, no broad coverage claim.",
            "oracles": "Published BCG author display and independent NumPy QR/SciPy likelihood are separate tests."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    assert receipt["status"] == "passed", receipt
