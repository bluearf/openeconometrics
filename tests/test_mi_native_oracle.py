"""Check independent-oracle gradients and refusal of corrupted display proof."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative


_PATH = Path(__file__).resolve().parents[1] / "scripts/verify_missing_data_oracles.py"
_SPEC = importlib.util.spec_from_file_location("mi_native_oracle", _PATH)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)


def test_observed_likelihood_complete_gradient_matches_independent_finite_difference():
    rng = np.random.default_rng(8162)
    y = rng.normal(size=(40, 3))
    y[10:16, 0] = np.nan
    y[20:29, 1:] = np.nan
    objective, _, pack = oracle.gaussian_objective(y)
    position = pack(
        np.array([0.2, -0.6, 1.1]), np.array([[1.8, 0.3, -0.1], [0.3, 1.2, 0.1], [-0.1, 0.1, 0.7]])
    )
    reference = approx_derivative(lambda z: objective(z)[0], position, method="3-point").ravel()
    np.testing.assert_allclose(objective(position)[1], reference, rtol=2e-8, atol=2e-8)
    fitted, _, _ = oracle.gaussian_fit(y)
    assert fitted["gradient_max_abs"] < 2e-5


def test_native_display_count_corruption_is_detected_without_model_imports():
    original = [[1.0, 2.0], [2.0, None], [3.0, 4.0], [4.0, 3.0]]
    state = {
        "method": "mi_chained",
        "columns": ["x", "y"],
        "seed": 1,
        "original": original,
        "completed_matrices": [
            [[1.0, 2.0], [2.0, 3.2], [3.0, 4.0], [4.0, 3.0]],
            [[1.0, 2.0], [2.0, 3.4], [3.0, 4.0], [4.0, 3.0]],
        ],
        "metadata": {
            "n": 4,
            "p": 2,
            "m": 2,
            "missing_mask": [[False, False], [False, True], [False, False], [False, False]],
            "missing_counts": [0, 1],
            "imputation_ids": [1, 2],
            "sample": {"nobs_original": 4, "nobs": 4, "dropped_rows": 0, "positions": [0, 1, 2, 3]},
            "index": {
                "kind": "index",
                "dtype": "object",
                "name": {"type": "scalar", "value": "participant"},
                "values": [
                    {"type": "scalar", "value": v} for v in ["row-0", "row-0", "row-1", "row-1"]
                ],
            },
            "converged": False,
            "stata_parity_validated": False,
            "methods": {"y": "normal"},
        },
    }
    state["integrity_sha256"] = oracle.canonical_hash(state)
    inputs = {
        "incomplete": {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, None, 4.0, 3.0]},
        "base": {"participant": ["row-0", "row-0", "row-1", "row-1"]},
    }
    shown = pd.DataFrame(
        {"column": ["x", "y"], "observed": [4, 3], "missing": [0, 1], "imputations": [2, 2]}
    )
    assert oracle.generation("normal", state, inputs, shown)["pass"]
    shown.loc[1, "missing"] = 0
    report = oracle.generation("normal", state, inputs, shown)
    assert not report["pass"]
    failed = [check["name"] for check in report["checks"] if not check["pass"]]
    assert failed == ["actual displayed missing counts"]


def test_full_d1_zero_between_and_finite_df_analytical_limit():
    pool = {
        "estimates": [[1.0, 2.0]] * 8,
        "covariances": [[[2.0, 0.8], [0.8, 3.0]]] * 8,
        "complete_df": 100.0,
    }
    result = oracle.d1_values(pool, [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "reiter2007")
    assert result["relative_increase_variance"] == 0
    assert result["df2"] == 100 * 101 / 103
    covariance = np.array(pool["covariances"][0])
    mean = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        result["statistic"], mean @ np.linalg.solve(covariance, mean) / 2, atol=1e-15, rtol=0
    )
