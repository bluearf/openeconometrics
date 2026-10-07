"""Regenerate MARKET-124's development-only independent scientific receipt.

PYTHONPATH=src:packages/openecon-charts/src python scripts/validate_spatial_market124.py
SciPy is an independent oracle dependency here, never in the estimator runtime.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy import optimize
import torch

import openecon as oe


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "docs/evidence/market-124-spatial"
    spec = importlib.util.spec_from_file_location(
        "spatial_scientific_oracle", root / "tests/test_spatial_models.py"
    )
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_spatial_models.py",
            "tests/test_spatial_weights_moran.py",
            "-q",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        print(check.stdout + check.stderr)
        raise SystemExit(check.returncode)
    data, weights, w = oracle.fixture_data()
    model_records = {}
    exports = {}
    for model in ("sar", "sem", "sac", "sdm"):
        start = time.perf_counter()
        result = getattr(oe, model)(data, "y", ["x", "z"], key="id", spatial_weights=weights)
        elapsed = time.perf_counter() - start
        x = np.column_stack((np.ones(len(data)), data.x, data.z))
        if model == "sdm":
            x = np.column_stack((x, w @ data[["x", "z"]].to_numpy()))
        y = data.y.to_numpy()
        params = np.array([c.estimate for c in result.coefficients])

        def likelihood(q):
            return oracle.covariance_density(q, y, x, w, model)

        independent_cov = np.linalg.inv(-oracle.finite_hessian(likelihood, params))

        def objective(spatial):
            return -oracle.profile_oracle(np.atleast_1d(spatial), y, x, w, model)[0]

        if model == "sac":
            optimum = optimize.differential_evolution(
                objective, [(-0.99998, 0.99998)] * 2, seed=9, tol=1e-11, polish=True
            )
        else:
            grid = np.linspace(-0.99998, 0.99998, 41)
            fits = [
                optimize.minimize_scalar(
                    objective, bounds=(a, b), method="bounded", options={"xatol": 1e-12}
                )
                for a, b in zip(grid[:-1], grid[1:], strict=True)
            ]
            optimum = min(fits, key=lambda fit: fit.fun)
        oracle_ll, oracle_params = oracle.profile_oracle(np.atleast_1d(optimum.x), y, x, w, model)
        model_records[model] = {
            "n": len(data),
            "nnz": weights.nnz,
            "parameters": [c.model_dump() for c in result.coefficients],
            "log_likelihood": result.metrics["log_likelihood"],
            "independent_gaussian_density_log_likelihood": likelihood(params),
            "independent_optimizer_log_likelihood": oracle_ll,
            "max_abs_parameter_error": float(np.abs(params - oracle_params).max()),
            "max_abs_covariance_error": float(
                np.abs(np.array(result.covariance_matrix) - independent_cov).max()
            ),
            "log_likelihood_error": abs(result.metrics["log_likelihood"] - oracle_ll),
            "elapsed_fit_seconds_on_this_host": elapsed,
            "solver": result.provenance["solver_diagnostics"],
            "resource_plans": result.provenance["resource_plans"],
            "result_json_roundtrip": result.model_dump()
            == type(result).model_validate_json(result.model_dump_json()).model_dump(),
        }
        exports[f"{model}.tex"] = result.to_latex()
        if model in {"sar", "sac", "sdm"}:
            exports[f"{model}-impacts.tex"] = oe.spatial_impacts(result).to_latex(index=False)
    keys = ["a", "b", "c", "d", "e"]
    small = oe.spatial_weights(
        keys,
        [
            ("a", "b", 2),
            ("a", "c", 1),
            ("b", "a", 1),
            ("b", "e", 2),
            ("c", "a", 1),
            ("c", "d", 2),
            ("d", "e", 1),
            ("e", "b", 3),
        ],
    )
    values = np.array([1.0, 2.0, -1.0, 4.0, 7.0])
    matrix = small.dense().numpy()

    def statistic(z):
        return len(z) / matrix.sum() * (z @ matrix @ z) / (z @ z)

    z = values - values.mean()
    null = np.array([statistic(z[list(order)]) for order in itertools.permutations(range(5))])
    moran = oe.moran(
        pd.DataFrame({"id": keys, "y": values}),
        "y",
        key="id",
        spatial_weights=small,
        permutations=149,
        seed=38,
    )
    receipt = {
        "issue": "MARKET-124",
        "evidence_layer": "local source scientific validation",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy_oracle_only": scipy.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "pytest": {"returncode": check.returncode, "output": check.stdout.strip()},
        "independence": "NumPy reduced-form multivariate Gaussian covariance density; finite-difference full information; independent Brent/differential-evolution optimizers; exhaustive finite permutation null; covariate and parameter perturbations for impacts",
        "models": model_records,
        "moran": {
            "enumerated_assignments": len(null),
            "mean_error": abs(moran["expected"] - null.mean()),
            "variance_error": abs(moran["variance"] - null.var()),
            "result": moran,
        },
        "limits": [
            "Cross-sectional CPU float64; fixed exogenous W; Gaussian ML",
            "Exact dense matrices with max_n=512 default and workspace guard; no measured large-data claim",
            "Conservative stable interval; multiple starts do not guarantee a global SAC maximum",
            "No Stata parity, frozen desktop, GPU, deployment, project-save UI or public-release proof",
        ],
    }
    target.mkdir(parents=True, exist_ok=True)
    for name, latex in exports.items():
        (target / name).write_text(latex.rstrip() + "\n")
    (target / "oracle.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(target / "oracle.json"),
                "models": {
                    name: {
                        key: value
                        for key, value in record.items()
                        if key
                        in {
                            "max_abs_parameter_error",
                            "max_abs_covariance_error",
                            "log_likelihood_error",
                            "elapsed_fit_seconds_on_this_host",
                        }
                    }
                    for name, record in model_records.items()
                },
                "moran_variance_error": receipt["moran"]["variance_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
