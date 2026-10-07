"""Seeded development experiment separating Gaussian efficiency/contamination.

The numerical estimator is native Torch. SciPy quadrature is a separate
oracle for Gaussian asymptotic efficiencies. The finite Monte Carlo is
diagnostic evidence, not a universal breakdown or efficiency guarantee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad

from openecon.econometrics.robust.smm import calibrated_tune, mm_estimate, s_estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    c0 = calibrated_tune(0.5)
    c1 = 4.685061
    seed = 20261007
    n = 300
    rng = np.random.default_rng(seed)
    errors = {"ols": [], "s": [], "mm": []}
    for replication in range(args.replications):
        x = torch.tensor(np.c_[np.ones(n), rng.normal(size=n)], dtype=torch.float64)
        beta = torch.tensor([1.0, 2.0], dtype=torch.float64)
        y = x @ beta + torch.tensor(rng.normal(size=n), dtype=torch.float64)
        sf = s_estimate(
            x, y, c=c0, b=0.5, starts=40, seed=replication, max_iterations=200, tolerance=1e-8
        )
        mb = mm_estimate(x, y, sf, c1, 200, 1e-8)[0]
        ob = torch.linalg.lstsq(x, y).solution
        for name, b in [("ols", ob), ("s", sf.beta), ("mm", mb)]:
            errors[name].append((b - beta).numpy())
    errors = {name: np.array(v) for name, v in errors.items()}
    mse = {name: np.mean(v**2, axis=0).tolist() for name, v in errors.items()}

    def normal(z):
        return np.exp(-z * z / 2) / np.sqrt(2 * np.pi)

    def theory(c):
        def psi(u):
            return u * max(1 - (u / c) ** 2, 0) ** 2

        def derivative(u):
            return (1 - (u / c) ** 2) * (1 - 5 * (u / c) ** 2) if abs(u) < c else 0.0

        a = quad(lambda z: derivative(z) * normal(z), -c, c, epsabs=1e-12)[0]
        b = quad(lambda z: psi(z) ** 2 * normal(z), -c, c, epsabs=1e-12)[0]
        return a * a / b

    efficiency = {
        name: (np.array(mse["ols"]) / np.array(mse[name])).tolist() for name in ["s", "mm"]
    }
    # Monte Carlo uncertainty: nonparametric bootstrap of the paired simulations.
    mc_rng = np.random.default_rng(seed + 1)
    ratios = []
    for _ in range(1000):
        rows = mc_rng.integers(0, args.replications, args.replications)
        ratios.append(
            np.mean(errors["ols"][rows] ** 2, axis=0) / np.mean(errors["mm"][rows] ** 2, axis=0)
        )
    uncertainty = np.quantile(ratios, [0.025, 0.975], axis=0).tolist()
    contamination = []
    x0 = rng.normal(size=150)
    noise = rng.normal(size=150) * 0.5
    for fraction in [0.2, 0.4]:
        m = int(150 * fraction)
        for amplitude in [1e3, 1e6]:
            xx = x0.copy()
            yy = 1 + 2 * xx + noise
            xx[:m] = 50 + np.arange(m) / m
            yy[:m] = amplitude - 70 * xx[:m]
            x = torch.tensor(np.c_[np.ones(150), xx], dtype=torch.float64)
            y = torch.tensor(yy, dtype=torch.float64)
            sf = s_estimate(
                x, y, c=c0, b=0.5, starts=300, seed=19, max_iterations=200, tolerance=1e-8
            )
            mb, _, before, after = mm_estimate(x, y, sf, c1, 200, 1e-8)
            contamination.append(
                {
                    "fraction": fraction,
                    "amplitude": amplitude,
                    "s": sf.beta.tolist(),
                    "mm": mb.tolist(),
                    "ols": torch.linalg.lstsq(x, y).solution.tolist(),
                    "s_scale": sf.scale,
                    "mm_objective_descent": after <= before,
                    "scale_equation_error": sf.scale_equation_error,
                }
            )
    report = {
        "seed": seed,
        "replications": args.replications,
        "n": n,
        "true_coefficients": [1, 2],
        "errors": "independent standard Gaussian",
        "starts": 40,
        "gaussian_asymptotic_efficiency_oracle": {"s": theory(c0), "mm": theory(c1)},
        "monte_carlo_mse": mse,
        "monte_carlo_relative_efficiency": efficiency,
        "mm_relative_efficiency_95pct_paired_mc_bootstrap": uncertainty,
        "contamination": contamination,
        "limits": "A finite seeded experiment does not certify global-minimum S breakdown. Monte Carlo efficiency is uncertain; Gaussian asymptotic efficiency is independently integrated. Covariance and published panel fixtures are tested separately.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
