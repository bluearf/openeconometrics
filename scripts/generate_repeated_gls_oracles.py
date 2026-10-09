#!/usr/bin/env python3
"""Reproduce independent static R/nlme residual-GLS fixtures; no OpenEcon import.

Regular scientific tests use the checked-in JSON and need no R installation.
The generator keeps all fitted residual blocks and both nlme's reported fixed
covariance and the independent unadjusted GLS matrix: nlme inflates its ML fixed
covariance by N/(N-p), whereas its fitted residual covariance is ML scaled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = ("cs", "ar1", "diagonal", "unstructured")


def fixture(structure: str) -> pd.DataFrame:
    rng = np.random.default_rng(87421 + STRUCTURES.index(structure))
    times = np.array([0, 1, 3, 4])
    if structure == "cs":
        shape = (1.0 + .24) * np.eye(4) - .24 * np.ones((4, 4))
    elif structure == "ar1":
        shape = np.power(-.62, np.abs(times[:, None] - times))
    elif structure == "diagonal":
        shape = np.diag(np.array([.45, .85, 1.3, 1.8])**2)
    else:
        factor = np.array([[.8, 0, 0, 0], [-.45, 1.25, 0, 0],
                           [.35, .25, .5, 0], [.1, -.35, .3, .9]])
        shape = factor @ factor.T
    records = []
    for subject in range(32):
        # At least six complete subjects; other deletion patterns independent
        # of x/y and not restricted to suffixes or adjacent occasion pairs.
        keep = np.ones(4, dtype=bool) if subject < 6 else rng.random(4) > .24
        if keep.sum() < 2:
            keep[rng.choice(4, size=2, replace=False)] = True
        x, z = rng.normal(size=(2, 4))
        residual = np.linalg.cholesky(shape) @ rng.normal(size=4)
        y = 1.7 + .8*x - .45*z + residual
        for occasion in np.flatnonzero(keep):
            records.append({"subject": f"s{subject:02d}", "occasion": int(times[occasion]),
                            "occasion_code": int(occasion + 1), "x": float(x[occasion]),
                            "z": float(z[occasion]), "y": float(y[occasion])})
    return pd.DataFrame(records)


def dense_likelihood(frame, covariance, method):
    x = np.column_stack([np.ones(len(frame)), frame[["x", "z"]]])
    y = frame.y.to_numpy()
    inverse_x = np.linalg.solve(covariance, x)
    bread = np.linalg.inv(x.T @ inverse_x)
    beta = bread @ inverse_x.T @ y
    residual = y - x @ beta
    logdet = np.linalg.slogdet(covariance)[1]
    q = residual @ np.linalg.solve(covariance, residual)
    value = -.5*(len(y)*np.log(2*np.pi) + logdet + q)
    if method == "REML":
        value += .5*(x.shape[1]*np.log(2*np.pi) - np.linalg.slogdet(x.T @ inverse_x)[1])
    return beta, bread, float(value), float(q)


def generate(rscript: str, destination: Path):
    with tempfile.TemporaryDirectory(prefix="repeated-gls-nlme-") as directory:
        root = Path(directory)
        cases = {}
        versions = None
        for structure in STRUCTURES:
            frame = fixture(structure)
            source = root / f"{structure}.csv"
            frame.to_csv(source, index=False, float_format="%.17g")
            output = root / structure
            subprocess.run([rscript, str(ROOT / "scripts/repeated_gls_nlme_oracle.R"),
                            str(source), structure, str(output)], check=True)
            versions = (output / "runtime.txt").read_text().splitlines()
            for method in ("ML", "REML"):
                prefix = output / method.lower()
                def matrix(suffix):
                    return pd.read_csv(f"{prefix}-{suffix}.csv").to_numpy(dtype=float)
                coefficients = pd.read_csv(f"{prefix}-coefficients.csv", index_col=0)
                cov = matrix("residual-covariance")
                beta, fixed_cov, likelihood, q = dense_likelihood(frame, cov, method)
                fit = pd.read_csv(f"{prefix}-fit.csv").iloc[0].to_dict()
                # Self-check the external fit against direct, independent dense
                # Gaussian likelihood, including restricted normalization.
                np.testing.assert_allclose(beta, coefficients.Value, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(likelihood, fit["log_likelihood"], rtol=1e-10, atol=1e-10)
                reported = matrix("reported-coefficient-covariance")
                adjusted = fixed_cov*(len(frame)/(len(frame)-3) if method == "ML" else 1)
                np.testing.assert_allclose(adjusted, reported, rtol=1e-9, atol=1e-11)
                for key in ("rho",):
                    if not np.isfinite(fit[key]):
                        fit[key] = None
                cases[f"{structure}_{method.lower()}"] = {
                    "structure": structure, "method": method,
                    "data": frame.to_dict("records"), "n": len(frame), "terms": ["Intercept", "x", "z"],
                    "coefficients": beta.tolist(), "conditional_coefficient_covariance": fixed_cov.tolist(),
                    "nlme_reported_coefficient_covariance": reported.tolist(),
                    "nlme_t_table": coefficients.to_dict("split"), "fit": fit,
                    "occasion_covariance": matrix("occasion-covariance").tolist(),
                    "weighted_residual_ss": q,
                }
        receipt = {"schema": "openecon.independent-repeated-gls-oracles.v1",
                   "external_runtime": versions, "generator_seed_base": 87421,
                   "source": "R/nlme gls, corCompSymm, corAR1, varIdent, corSymm",
                   "reference_urls": ["https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/gls.html",
                       "https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corAR1.html",
                       "https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/corSymm.html"],
                   "nlme_ml_fixed_covariance_correction": "N/(N-p); preserve separately from unadjusted conditional GLS",
                   "cases": cases}
        for script in ("generate_repeated_gls_oracles.py", "repeated_gls_nlme_oracle.R"):
            receipt.setdefault("script_sha256", {})[script] = hashlib.sha256((ROOT / "scripts" / script).read_bytes()).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"cases": len(cases), "destination": str(destination), "external_runtime": versions}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/repeated_gls/nlme.json")
    args = parser.parse_args()
    generate(args.rscript, args.output)
