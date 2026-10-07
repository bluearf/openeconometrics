"""Declared independent-seed iid size/power grid, with complete denominators.

Run from a checkout using PYTHONPATH=src. This is a bounded Monte Carlo study,
not proof for serially dependent errors or partial structural change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import openecon as oe

ROOT = Path(__file__).resolve().parents[1]


def validate():
    torch.set_num_threads(1)
    outer, inner = 200, 199
    grid = [(40, False), (80, True)]
    levels = [.01, .05, .10]
    rows = []
    started = time.monotonic()
    for cell, (n, predictor) in enumerate(grid):
        for shifted in (False, True):
            counts = np.zeros(3, dtype=int)
            failures = []
            selected = []
            for draw in range(outer):
                # Disjoint data and calibration streams, no outcome-tuned seeds.
                data_seed = 831_000 + cell*10_000 + int(shifted)*1000 + draw
                calibration_seed = 9_427_000 + cell*10_000 + int(shifted)*1000 + draw
                rng = np.random.default_rng(data_seed)
                x = rng.normal(size=n)
                y = 2 + (.7*x if predictor else 0) + rng.normal(size=n)
                if shifted:
                    y[n//3:2*n//3] += 3
                frame = pd.DataFrame({"y": y, "x": x})
                try:
                    result = oe.bai_perron(frame, "y", ["x"] if predictor else [],
                                          max_breaks=2, replications=inner, seed=calibration_seed)
                    counts += np.array([result.attrs["p_value"] <= alpha for alpha in levels])
                    selected.append(result.attrs["selected_breaks"])
                except Exception as exc:
                    failures.append({"draw": draw, "type": type(exc).__name__, "message": str(exc)})
            rates = counts / outer  # failed fits remain in the declared denominator
            tolerances = [3.29*np.sqrt(alpha*(1-alpha)/outer) + 1/(inner+1) for alpha in levels]
            passed = (not failures and (rates[1] >= .85 if shifted else
                                       all(abs(rate-alpha) <= tol for rate, alpha, tol
                                           in zip(rates, levels, tolerances))))
            row = dict(n=n, parameters=2 if predictor else 1, max_breaks=2,
                       alternative="two mean shifts of 3 sigma" if shifted else "constant coefficients",
                       outer_replications=outer, inner_replications=inner,
                       rejection_denominator=outer, failures=failures,
                       levels=levels, rejections=counts.tolist(), rejection_rates=rates.tolist(),
                       null_tolerances=tolerances, power_5pct_minimum=.85,
                       selected_break_histogram={str(k): selected.count(k) for k in range(3)}, passed=bool(passed))
            rows.append(row)
            print(json.dumps(row), flush=True)
    source = ROOT / "src/openecon/econometrics/unitroot/bai_perron.py"
    return dict(status="passed" if all(row["passed"] for row in rows) else "failed",
                method="Bai–Perron pure-change Gaussian iid conditional design",
                scope="two declared T/q cells; no serial dependence, heteroskedasticity or partial-change claim",
                seed_rule="data=831000+cell*10000+alternative*1000+draw; calibration=9427000+same offsets",
                rejection_rule="UDmax Monte Carlo p <= alpha; fixed 2-break search repeated per draw",
                acceptance_rule="null within 3.29 binomial standard errors plus 1/(B+1); power at 5%>=.85; zero failures",
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                elapsed_seconds=time.monotonic()-started, grid=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if receipt["status"] == "passed" else 1)
