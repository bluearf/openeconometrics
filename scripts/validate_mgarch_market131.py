"""Regenerate MARKET-131's source-only independent scientific receipt.

PYTHONPATH=src:packages/openecon-charts/src python scripts/validate_mgarch_market131.py
NumPy and actual R references are development oracles, never runtime dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

import openecon as oe
from openecon.econometrics.mgarch import kernels


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "docs/evidence/market-131-mgarch"
    test_file = root / "tests/test_mgarch_models.py"
    spec = importlib.util.spec_from_file_location("mgarch_independent_oracle", test_file)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_mgarch_models.py",
            "-q",
            "-o",
            "cache_dir=artifacts/mgarch-validation/pytest-cache",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        print(check.stdout + check.stderr)
        raise SystemExit(check.returncode)
    reference_file = target / "published-reference.json"
    reference = json.loads(reference_file.read_text())
    reference_errors = {}
    for kind in ("ccc", "dcc", "bekk"):
        record = reference[kind]
        matrices = {
            k: torch.tensor(v, dtype=torch.float64) for k, v in record["parameters"].items()
        }
        matrices["mu"] = torch.tensor(reference["mean"], dtype=torch.float64)
        initial = (
            torch.tensor(record["initializer"], dtype=torch.float64)
            if kind == "bekk"
            else torch.diag(torch.tensor(record["initializer_variances"], dtype=torch.float64))
        )
        y = torch.tensor(reference["returns"], dtype=torch.float64)
        path, _, state = kernels.covariance_path(matrices, y, initial, kind)
        correlation = (
            path
            / path.diagonal(dim1=1, dim2=2).sqrt()[:, :, None]
            / path.diagonal(dim1=1, dim2=2).sqrt()[:, None, :]
        )
        forecast = kernels.forecast_path(matrices, state, kind, reference["horizons"])
        forecast_corr = (
            forecast
            / forecast.diagonal(dim1=1, dim2=2).sqrt()[:, :, None]
            / forecast.diagonal(dim1=1, dim2=2).sqrt()[:, None, :]
        )
        reference_errors[kind] = {
            "max_abs_covariance_path_error": float(
                np.abs(path.numpy() - record["covariance_path"]).max()
            ),
            "max_abs_correlation_path_error": float(
                np.abs(correlation.numpy() - record["correlation_path"]).max()
            ),
            "max_abs_covariance_forecast_error": float(
                np.abs(forecast.numpy() - record["covariance_forecast"]).max()
            ),
            "max_abs_correlation_forecast_error": float(
                np.abs(forecast_corr.numpy() - record["correlation_forecast"]).max()
            ),
            "gaussian_log_likelihood_error": abs(
                float(kernels.matrix_log_likelihood(matrices, y, initial, kind))
                - record["gaussian_log_likelihood"]
            ),
            "forecast_reference_convention": record["forecast_method"],
        }
    models = {}
    exports = {}
    for kind in ("ccc", "dcc", "bekk"):
        data = (
            oracle.simulated_data(kind="bekk", n=400, seed=18)
            if kind == "bekk"
            else oracle.simulated_data()
        )
        started = time.perf_counter()
        result = getattr(oe, "mgarch_" + kind)(data, ["x", "y"], intercept=False, tolerance=1e-8)
        elapsed = time.perf_counter() - started
        initial = np.array(result.extra["initialization"]["initial_covariance"])
        parameters = np.array([coefficient.estimate for coefficient in result.coefficients])

        def objective(q):
            return oracle.independent_ll(q, data.to_numpy(), initial, kind)

        independent_information = -oracle.numerical_hessian(objective, parameters)
        independent_covariance = np.linalg.inv(independent_information)
        delta = 1e-5 * (1 + np.abs(parameters))
        finite_score = np.array(
            [
                (
                    objective(parameters + np.eye(len(parameters))[i] * delta[i])
                    - objective(parameters - np.eye(len(parameters))[i] * delta[i])
                )
                / (2 * delta[i])
                for i in range(len(parameters))
            ]
        )
        restored = type(result).model_validate_json(result.model_dump_json())
        forecast = oe.forecast(restored, 5)
        models[kind] = {
            "nobs": result.nobs,
            "dimension": 2,
            "constant_mean": False,
            "parameters": [coefficient.model_dump() for coefficient in result.coefficients],
            "initialization": result.extra["initialization"],
            "gaussian_log_likelihood": result.metrics["log_likelihood"],
            "independent_gaussian_log_likelihood": objective(parameters),
            "log_likelihood_error": abs(result.metrics["log_likelihood"] - objective(parameters)),
            "max_abs_covariance_error": float(
                np.abs(np.array(result.covariance_matrix) - independent_covariance).max()
            ),
            "independent_finite_difference_score_max": float(np.abs(finite_score).max()),
            "independent_scaled_score": float(finite_score @ independent_covariance @ finite_score),
            "independent_information_eigenvalues": np.linalg.eigvalsh(
                independent_information
            ).tolist(),
            "elapsed_fit_seconds_on_this_host": elapsed,
            "solver": result.provenance["solver_diagnostics"],
            "resource_plans": result.provenance["resource_plans"],
            "strict_json_roundtrip": result.model_dump() == restored.model_dump(),
            "forecasts_after_json_restore": forecast.attrs,
        }
        exports[f"{kind}.tex"] = result.to_latex()
        exports[f"{kind}-forecast.tex"] = forecast.to_latex(index=False)
    receipt = {
        "issue": "MARKET-131",
        "evidence_layer": "local source scientific validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy_reference_only": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "pytest": {"returncode": check.returncode, "output": check.stdout.strip()},
        "independent_oracle_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
        "reference_fixture_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "independence": "Actual R source-extracted published forecast loop and independent R paper recursions; separate NumPy physical-parameter Gaussian density and finite-difference full score/information; no production map or optimizer in NumPy likelihood oracle",
        "published_reference": {
            "engine": reference["reference_engine"],
            "runtime": reference["reference_runtime"],
            "primary_sources": reference["primary_sources"],
            "horizons": reference["horizons"],
            "literal_bekk_forecast_block_executed": True,
            "setup_sha256": reference["evaluated_R_setup_sha256"],
            "models": reference_errors,
            "limits": reference["limitations"],
        },
        "joint_fits": models,
        "additional_test_coverage": [
            "Complete physical-unit score/Hessian away from the optimum for every model",
            "Three-dimensional CCC/DCC joint constant-mean fits and nonzero nuisance covariance blocks",
            "Full off-diagonal A and B, physical transform roundtrip and full Kronecker stationarity",
            "Independent Gaussian quadrature demonstrates CCC multi-step covariance plug-in gap",
            "Direct and serialized-console LaTeX preserves forecast approximation and parameter-conditioning disclosures",
            "Dimension, sample, time, missing, positivity, stationarity, resource, nonconvergence, persisted-state and Dataset guards",
        ],
        "limits": [
            "Gaussian order (1,1), constant/zero mean, CPU float64; CCC/DCC dimension 2..3 and full BEKK 2",
            "Full joint Gaussian conditional ML with estimated DCC target; not the conventional two-stage empirical-target estimator",
            "Fixed recorded initializer conditioned upon; no backcast uncertainty, robust covariance or global optimum guarantee",
            "Exact full one-step covariance for all models and all-horizon full BEKK covariance; CCC/DCC later full covariances are labeled plug-in approximations",
            "Exact marginal variance expectations for CCC/DCC; normalized forecast covariance is not an expectation of random conditional correlations",
            "Default max_n=512 and workspace guards; no Dataset/GPU or measured large-scale execution claim",
            "Supplied stationary parameter reference, not a published empirical fit or complete external-package coefficient/Hessian/OPG parity",
            "No installed-app, frozen desktop, project-save UI, deployment or public-release proof",
        ],
    }
    target.mkdir(parents=True, exist_ok=True)
    for name, latex in exports.items():
        (target / name).write_text(latex + "\n")
    (target / "oracle.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(target / "oracle.json"),
                "reference_errors": reference_errors,
                "joint_fit_errors": {
                    kind: {
                        key: record[key]
                        for key in (
                            "log_likelihood_error",
                            "max_abs_covariance_error",
                            "independent_finite_difference_score_max",
                            "independent_scaled_score",
                            "elapsed_fit_seconds_on_this_host",
                        )
                    }
                    for kind, record in models.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
