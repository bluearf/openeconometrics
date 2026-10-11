"""Monte-Carlo-free limiting rejection rates of Ng--Perron Table I thresholds.

QA-only; not imported by the runtime. Under the unit-root null with known
long-run variance, the Theorem-1 limits of MZa, MSB and MPT are quadratic forms
in Brownian motion. On an N-step grid, each event {MZa < x}, {MSB < x} and
{MPT < x} becomes {e'Be < q} for iid standard normal e and a symmetric matrix B
built from the same functional the validator simulates
(scripts/validate_unitroot_methods.py): W for the constant case and
V(r) = W(r) - r[lam W(1) + 3(1 - lam) int s W(s) ds] for the trend case,
with trapezoid integrals. Its probability is a weighted chi-square CDF, evaluated
by Imhof (1961) numerical inversion with a rigorous truncation bound. MZt is a
ratio rather than a quadratic form and is not covered.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

# Ng & Perron (2001), Econometrica 69, Table I, p. 1524 (1%, 5%, 10%).
TABLE_I = {
    "constant": {"MZa": (-13.8, -8.1, -5.7), "MSB": (0.174, 0.233, 0.275), "MPT": (1.78, 3.17, 4.45)},
    "trend": {"MZa": (-23.8, -17.3, -14.2), "MSB": (0.143, 0.168, 0.185), "MPT": (4.03, 5.48, 6.67)},
}
CBAR = {"constant": -7.0, "trend": -13.5}


def functional(steps, trend):
    """Return (C_I, c_E): integral = e'C_I e and endpoint = (c_E'e)**2."""
    paths = np.tril(np.ones((steps + 1, steps + 1)), -1)[:, :steps] / math.sqrt(steps)
    r = np.arange(steps + 1) / steps
    weights = np.full(steps + 1, 1.0 / steps)
    weights[0] = weights[-1] = 0.5 / steps
    if trend == "trend":
        c = CBAR[trend]
        lam = (1 - c) / (1 - c + c * c / 3)
        slope = lam * paths[-1] + 3 * (1 - lam) * (weights * r) @ paths
        paths = paths - np.outer(r, slope)
    return paths.T @ (weights[:, None] * paths), paths[-1]


def imhof_cdf(lam, q, step=0.02, tol=1e-11, limit=1e7):
    """P(sum_i lam_i chi2_1 < q) for nonnegative weights (Imhof 1961).

    The integral is truncated at U once the bound
    int_U^inf du/(u rho(u)) <= (2/m) U^(-m/2) prod_{j<=m} lam_j^(-1/2)
    (largest m weights, minimized over m) is below tol.
    """
    lam = np.sort(lam[lam > 1e-15 * lam.max()])[::-1]
    cumulative = np.cumsum(np.log(lam))
    counts = np.arange(1, len(lam) + 1)
    total, start, last = 0.0, 0.0, 0.5 * (lam.sum() - q)
    while True:
        u = start + step * np.arange(1, 4097)
        scaled = np.outer(u, lam)
        theta = 0.5 * np.arctan(scaled).sum(axis=1) - 0.5 * q * u
        log_rho = 0.25 * np.log1p(scaled * scaled).sum(axis=1)
        values = np.concatenate(([last], np.sin(theta) / (u * np.exp(log_rho))))
        total += step / 3 * (
            values[0] + 4 * values[1:-1:2].sum() + 2 * values[2:-1:2].sum() + values[-1]
        )
        last, start = values[-1], u[-1]
        bound = np.min(np.log(2 / counts) - 0.5 * counts * np.log(start) - 0.5 * cumulative)
        if bound < math.log(tol):
            return 1.0 - (0.5 + total / math.pi)
        if start > limit:
            raise RuntimeError("Imhof integration did not converge.")
        if start > 2000:
            step = min(step * 2, 1.0)


def chi_square_cdf_even(df, q):
    """Closed form for even degrees of freedom, used only as a self-test."""
    half = q / 2
    return 1.0 - math.exp(-half) * sum(half**j / math.factorial(j) for j in range(df // 2))


def self_test():
    checks = []
    for df, q in [(10, 8.0), (16, 12.0), (40, 30.0)]:
        exact = chi_square_cdf_even(df, q)
        value = float(imhof_cdf(np.ones(df), q))
        checks.append(
            dict(df=df, q=q, imhof=value, closed_form=exact, passed=bool(abs(value - exact) < 1e-8))
        )
    return checks


def rates(steps, trend):
    integral, endpoint = functional(steps, trend)
    c = CBAR[trend]
    end_weight = -c if trend == "constant" else 1 - c
    outer = np.outer(endpoint, endpoint)
    lam_integral = np.linalg.eigvalsh(integral)
    lam_mpt = np.linalg.eigvalsh(c * c * integral + end_weight * outer)
    out = {}
    for name, thresholds in TABLE_I[trend].items():
        row = []
        for cv in thresholds:
            if name == "MZa":  # (E - 1) / (2I) < cv  <=>  E - 2 cv I < 1, since I > 0
                probability = imhof_cdf(np.linalg.eigvalsh(outer - 2 * cv * integral), 1.0)
            elif name == "MSB":  # sqrt(I) < cv
                probability = imhof_cdf(lam_integral, cv * cv)
            else:  # c^2 I + k E < cv
                probability = imhof_cdf(lam_mpt, cv)
            row.append(float(probability))
        out[name] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 2000])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path.")
    start = time.monotonic()
    tests = self_test()
    if not all(check["passed"] for check in tests):
        raise SystemExit(f"Imhof self-test failed: {tests}")
    grids = {
        str(steps): {trend: rates(steps, trend) for trend in ("constant", "trend")}
        for steps in args.steps
    }
    record = dict(
        source="Ng & Perron (2001), Econometrica 69, Table I, p. 1524",
        method="Imhof inversion of the validator's Theorem-1 functional; no Monte Carlo",
        levels=[0.01, 0.05, 0.10],
        self_test=tests,
        rejection_rates=grids,
        python=platform.python_version(),
        numpy=np.__version__,
        elapsed_seconds=time.monotonic() - start,
        limits="Known long-run variance; MZt not covered; trapezoid discretization at the listed step counts.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"elapsed_seconds": record["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
