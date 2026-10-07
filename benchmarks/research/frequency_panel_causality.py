"""Predeclared size/power grid for two bounded noncausality procedures.

This is numerical evidence in specified DGPs, not blanket finite-sample validity.
Every planned call remains in the denominator; failures are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.var.frequency_causality import bccaustest
from openecon.econometrics.var.panel_causality import dhcausality

ROOT = Path(__file__).resolve().parents[2]
ALPHAS = (.01, .05, .1)
SEEDS = (18073, 90149)
BC_GRID = (
    ("null_p3_n160", 3, 160, "null"),
    ("null_p6_n320", 6, 320, "null"),
    ("spectral_null_p3_n320", 3, 320, "spectral_null"),
    ("causal_p3_n160", 3, 160, "causal"),
)
DH_GRID = (
    ("null_N20_T80_K1", 20, 80, 1, 0.),
    ("null_N50_T40_K2", 50, 40, 2, 0.),
    ("half_causal_N20_T80_K1", 20, 80, 1, .5),
    ("half_causal_N50_T40_K2", 50, 40, 2, .5),
)


def wilson(count, denominator):
    z = 1.959963984540054
    fraction = count / denominator
    center = (fraction + z*z/(2*denominator)) / (1+z*z/denominator)
    half = z*math.sqrt(fraction*(1-fraction)/denominator + z*z/(4*denominator**2)) \
        / (1+z*z/denominator)
    return [center-half, center+half]


def bc_data(periods, regime, seed):
    rng = torch.Generator().manual_seed(seed)
    values = torch.randn((periods+200, 2), generator=rng, dtype=torch.float64)
    beta = torch.zeros(3, dtype=torch.float64)
    if regime == "causal":
        beta[0] = .35
    elif regime == "spectral_null":
        beta[:] = torch.tensor([.25, -.5*math.cos(.8), .25], dtype=torch.float64)
    for t in range(3, len(values)):
        values[t, 0] += .15 + .35*values[t-1, 0] + beta @ values[t-3:t, 1].flip(0)
        values[t, 1] += .1 + .4*values[t-1, 1]
    return pd.DataFrame(values[-periods:].numpy(), columns=["y", "x"])


def dh_data(units, periods, lags, causal_share, seed):
    rng = torch.Generator().manual_seed(seed)
    # Original paper benchmark Eq (18) has heterogeneous AR slopes, intercepts,
    # innovation variances, and some/all causal units. This declared grid uses
    # stable rho in [-.8,.8] and stationary AR(.4) x, making no exact table-cell
    # claim where original x generation/initialization is not fully specified.
    rho = torch.rand(units, generator=rng, dtype=torch.float64)*1.6-.8
    intercept = torch.randn(units, generator=rng, dtype=torch.float64)
    sigma = (torch.rand(units, generator=rng, dtype=torch.float64)+.5).sqrt()
    beta = torch.zeros(units, dtype=torch.float64)
    beta[:int(units*causal_share)] = .4
    values = torch.randn((units, periods+200, 2), generator=rng, dtype=torch.float64)
    values[:, :, 0] *= sigma[:, None]
    for t in range(1, periods+200):
        values[:, t, 0] += intercept + rho*values[:, t-1, 0] + beta*values[:, t-1, 1]
        values[:, t, 1] += .4*values[:, t-1, 1]
    frame = pd.DataFrame(values[:, -periods:].reshape(-1, 2).numpy(), columns=["y", "x"])
    frame["unit"] = [unit for unit in range(units) for _ in range(periods)]
    frame["t"] = list(range(periods))*units
    return frame


def run(outer):
    if outer < 500:
        raise ValueError("The acceptance grid requires at least 500 outer replications per cell and seed")
    started = time.monotonic()
    records = []
    for seed in SEEDS:
        for method, grid in (("bccaustest", BC_GRID), ("dhcausality", DH_GRID)):
            for cell_index, cell in enumerate(grid):
                identifier = cell[0]
                frequencies = [.8] if method == "bccaustest" and cell[3] == "spectral_null" \
                    else [.2, .8, 2.4]
                rejections = [[0]*3 for _ in (frequencies if method == "bccaustest" else [None])]
                failures = {}
                for replicate in range(outer):
                    local_seed = seed + cell_index*1000003 + replicate*1009
                    try:
                        if method == "bccaustest":
                            _, lags, periods, regime = cell
                            frame = bc_data(periods, regime, local_seed)
                            result = bccaustest(data=frame, y="y", x="x", lags=lags,
                                               frequencies=frequencies)
                            probabilities = result["tests"].p_value.tolist()
                        else:
                            _, units, periods, lags, causal_share = cell
                            frame = dh_data(units, periods, lags, causal_share, local_seed)
                            result = dhcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=lags)
                            probabilities = [result["tests"].loc[2, "p_value"]]
                        for i, probability in enumerate(probabilities):
                            for j, alpha in enumerate(ALPHAS):
                                rejections[i][j] += probability < alpha
                    except AnalysisError as exc:
                        failures[exc.code] = failures.get(exc.code, 0) + 1
                failed = sum(failures.values())
                for i, counts in enumerate(rejections):
                    rates = [{"alpha": alpha, "rejections": count, "planned_denominator": outer,
                              "rate_failure_as_nonreject": count/outer,
                              "rate_failure_as_reject": (count+failed)/outer,
                              "wilson_95": wilson(count, outer)} for alpha, count in zip(ALPHAS, counts)]
                    null = identifier.startswith("null") or "spectral_null" in identifier
                    primary = rates[1]
                    passed = failed == 0 and (primary["wilson_95"][1] <= .10 if null
                                              else primary["wilson_95"][0] >= .70)
                    records.append({"method": method, "cell": list(cell), "seed": seed,
                                    "frequency": frequencies[i] if method == "bccaustest" else None,
                                    "failures": failures, "rates": rates, "passed_declared_gate": passed})
                print(json.dumps({"completed": method+":"+identifier, "seed": seed,
                                  "failures": failed}), flush=True)
    paths = ["src/openecon/econometrics/var/frequency_causality.py",
             "src/openecon/econometrics/var/panel_causality.py",
             "benchmarks/research/frequency_panel_causality.py"]
    return {"schema": 1, "outer_per_cell_per_seed": outer, "seeds": list(SEEDS),
            "planned_calls": len(SEEDS)*(len(BC_GRID)+len(DH_GRID))*outer,
            "alpha_grid": list(ALPHAS), "null_gate": "Wilson95 upper<=.10 at .05; zero failed calls",
            "power_gate": "Wilson95 lower>=.70 at .05; zero failed calls",
            "scope": "specified stationary iid independent DGPs only; no blanket finite-T proof",
            "cells": records, "passed_declared_grid": all(r["passed_declared_gate"] for r in records),
            "duration_seconds": time.monotonic()-started,
            "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = run(args.outer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    if not receipt["passed_declared_grid"]:
        raise SystemExit("Declared size/power acceptance grid failed; inspect evidence without changing the gates")
