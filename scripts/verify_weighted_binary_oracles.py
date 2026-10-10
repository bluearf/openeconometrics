"""Offline independent numerical checks of actual native saved results and queries."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_weighted_binary import reference  # noqa: E402


def verify(directory):
    frame = pd.DataFrame(json.loads((directory / "weighted-binary-inputs.json").read_text()))
    results = json.loads((directory / "weighted-binary-results.json").read_text())
    queries = json.loads((directory / "weighted-binary-queries.json").read_text())
    latex = json.loads((directory / "weighted-binary-latex.json").read_text())
    assert len(results) == len(queries) == len(latex) == 8
    checks = []
    for name, result in results.items():
        model, weight = name.split("_")
        kind = result["spec"]["covariance"]
        beta, v, ll, n, _, _, w = reference(frame, model, weight, kind)
        coefficients = result["coefficients"]
        actual = np.array(
            [
                [
                    c[k]
                    for k in ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high")
                ]
                for c in coefficients
            ]
        )
        se = np.sqrt(np.diag(v))
        z = beta / se
        critical = norm.ppf(0.975)
        expected = np.column_stack(
            [beta, se, z, 2 * norm.sf(abs(z)), beta - critical * se, beta + critical * se]
        )
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=8e-8)
        np.testing.assert_allclose(result["covariance_matrix"], v, atol=2e-10, rtol=5e-8)
        assert result["nobs"] == n and result["sample_positions"] == list(range(len(frame)))
        assert abs(result["metrics"]["log_likelihood"] - ll) < 1e-7
        x = np.column_stack([np.ones(len(frame)), frame[["x", "z"]]])
        eta = x @ beta
        p = expit(eta) if model == "logit" else ndtr(eta)
        first = p * (1 - p) if model == "logit" else norm.pdf(eta)
        second = first * (1 - 2 * p) if model == "logit" else -eta * first
        gradient = first[:7, None] * x[:7]
        errors = np.sqrt(np.einsum("ni,ij,nj->n", gradient, v, gradient))
        np.testing.assert_allclose(
            queries[name]["predict"]["data"],
            np.column_stack([p[:7], errors, p[:7] - critical * errors, p[:7] + critical * errors]),
            atol=2e-8,
            rtol=5e-8,
        )
        normalized = w / w.sum()
        for j, row in enumerate(queries[name]["margins"]["data"], start=1):
            estimate = beta[j] * (normalized @ first)
            derivative = (normalized * second * beta[j]) @ x
            derivative[j] += normalized @ first
            error = np.sqrt(derivative @ v @ derivative)
            statistic = estimate / error
            np.testing.assert_allclose(
                [row[i] for i in (2, 3, 4, 7, 8, 9)],
                [
                    estimate,
                    error,
                    statistic,
                    2 * norm.sf(abs(statistic)),
                    estimate - critical * error,
                    estimate + critical * error,
                ],
                atol=2e-8,
                rtol=8e-8,
            )
        contrast = np.array([0.0, 1.0, -0.5])
        value = contrast @ beta
        error = np.sqrt(contrast @ v @ contrast)
        np.testing.assert_allclose(
            [queries[name]["lincom"][k] for k in ("estimate", "std_error")],
            [value, error],
            atol=2e-8,
        )
        statistic = beta[1:] @ np.linalg.solve(v[1:, 1:], beta[1:])
        np.testing.assert_allclose(
            [queries[name]["test"][k] for k in ("statistic", "p_value")],
            [statistic, chi2.sf(statistic, 2)],
            atol=2e-8,
        )
        assert latex[name] and not result["provenance"]["stata_parity_validated"]
        checks.append(
            {
                "domain": name,
                "covariance": kind,
                "parameters": len(beta),
                "status": "passed",
                "full_covariance_inference_prediction_margins_contrast_test": True,
            }
        )
    return {
        "status": "passed",
        "domains": checks,
        "native_input_rows": len(frame),
        "oracle": "independent NumPy/SciPy weighted score root and observed information; no runtime oracle dependency",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "domains": len(report["domains"])}))
