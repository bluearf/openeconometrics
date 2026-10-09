"""Development-only independent NumPy/SciPy survey and official-manual oracles.

Never imported by production kernels. Reference datasets are not redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def fixture():
    return pd.DataFrame(
        {
            "w": [2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 4.0, 6.0],
            "p": [1, 1, 2, 2, 1, 1, 2, 2],
            "h": ["a"] * 4 + ["b"] * 4,
            "y": [1.0, 3.0, 2.0, 5.0, 6.0, 8.0, 7.0, 11.0],
            "x": [2.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 6.0],
            "domain": [1, 1, 1, 1, 0, 0, 0, 0],
            "category": ["a", "a", "a", "b", "b", "a", "b", "b"],
        }
    )


def target(frame, weights, kind, columns, denominators=None, domain=None, categories=None):
    use = (
        np.ones(len(frame), dtype=bool) if domain is None else frame[domain].to_numpy().astype(bool)
    )
    if kind == "proportion":
        values = np.column_stack(
            [frame[columns[0]].to_numpy() == level for level in categories]
        ).astype(float)
    else:
        use &= frame[columns + (denominators or [])].notna().all(axis=1).to_numpy()
        values = frame[columns].fillna(0).to_numpy()
    numerator = (weights * use) @ values
    if kind == "total":
        return numerator, use[:, None] * values
    x = frame[denominators].fillna(0).to_numpy() if kind == "ratio" else np.ones_like(values)
    denominator = (weights * use) @ x
    theta = numerator / denominator
    return theta, use[:, None] * (values - x * theta) / denominator


def taylor(frame, weights, score):
    covariance = np.zeros((score.shape[1], score.shape[1]))
    for _, block in frame.groupby("h", sort=False):
        totals = []
        for _, rows in block.groupby("p", sort=False):
            idx = rows.index.to_numpy()
            totals.append((weights[idx, None] * score[idx]).sum(0))
        totals = np.asarray(totals)
        centered = totals - totals.mean(0)
        covariance += len(totals) / (len(totals) - 1) * (centered.T @ centered)
    return covariance


def comparison(actual, expected, *, atol=2e-10):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=atol)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def check_states(path):
    states = json.loads(path.read_text())
    frame = fixture()
    w = frame.w.to_numpy()
    checks = {}
    for name, state in states.items():
        metadata = state["metadata"]
        args = (
            state["target"],
            metadata["outcomes"],
            metadata["denominators"] or None,
            metadata["domain_column"],
            ["a", "b", "absent"] if name == "proportion" else None,
        )
        theta, score = target(frame, w, *args)
        if state["method"] == "taylor":
            covariance = taylor(frame, w, score)
        else:
            replicas, multipliers, strata = [], [], []
            if name == "jackknife":
                for h in ("a", "b"):
                    for p in (1, 2):
                        replica = w.copy()
                        replica[(frame.h.to_numpy() == h) & (frame.p.to_numpy() == p)] = 0.0
                        replica[(frame.h.to_numpy() == h) & (frame.p.to_numpy() != p)] *= 2.0
                        replicas.append(target(frame, replica, *args)[0])
                        multipliers.append(0.5)
                        strata.append(h)
            else:
                rho = 0.3 if name == "fay" else 0.0
                choices = (
                    list(product((1, 2), repeat=2))
                    if name == "bootstrap"
                    else [(1, 1), (2, 1), (1, 2), (2, 2)]
                )
                for pair in choices:
                    factors = np.array(
                        [
                            2.0 - rho if p == pair[0 if h == "a" else 1] else rho
                            for h, p in zip(frame.h, frame.p)
                        ]
                    )
                    replicas.append(target(frame, w * factors, *args)[0])
                    multipliers.append(1 / (4 * (1 - rho) ** 2))
            replicas = np.asarray(replicas)
            comparison(metadata["replicate_estimates"], replicas)
            comparison(metadata["variance_multipliers"], multipliers)
            if name == "jackknife":
                centered = replicas.copy()
                for h in ("a", "b"):
                    idx = np.array(strata) == h
                    centered[idx] -= replicas[idx].mean(0)
            elif metadata["centering"] == "replicate_mean":
                centered = replicas - replicas.mean(0)
            else:
                centered = replicas - theta
            covariance = (centered.T * np.array(multipliers)) @ centered
        se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        critical = stats.t.isf(state["alpha"] / 2, state["df"])
        table = {
            "estimate": theta.tolist(),
            "std_error": se.tolist(),
            "p_value": [
                float(2 * stats.t.sf(abs((a - b) / s), state["df"])) if s else None
                for a, b, s in zip(theta, state["null"], se)
            ],
            "ci_low": (theta - critical * se).tolist(),
            "ci_high": (theta + critical * se).tolist(),
        }
        checks[name] = {
            "estimate_max_abs_error": comparison(state["estimates"], theta),
            "covariance_max_abs_error": comparison(state["covariance"], covariance),
            "independent_table": table,
            "df": state["df"],
            "sample": metadata["sample_positions"],
        }
    execution = json.loads(path.with_name("native-execution.json").read_text())
    # The native example preserves insertion order, while the canonical state
    # file sorts keys. Match each actual displayed output explicitly.
    order = ["mean", "total", "ratio", "proportion", "brr", "fay", "jackknife", "bootstrap"]
    assert len(execution["outputs"]) == 8
    for name, output in zip(order, execution["outputs"]):
        data = output["data"]
        expected = checks[name]["independent_table"]
        for column, values in expected.items():
            position = data["columns"].index(column)
            observed = [row[position] for row in data["rows"]]
            for value, truth in zip(observed, values):
                if truth is None:
                    assert value is None
                else:
                    comparison(value, truth)
        checks[name]["actual_native_displayed_inference_compared"] = True
    return {
        "status": "independent-native-states-passed",
        "methods": checks,
        "oracle": "NumPy direct weighted scores/PSU sums and exhaustive first-stage PSU enumeration; SciPy Student t",
        "production_helpers_used": False,
        "licensed_stata_execution": False,
    }


def check_manual(path):
    import openecon as oe

    frame = pd.read_stata(path, convert_categoricals=False)
    design = oe.survey_design(frame, weights="finalwgt", psu="psu", strata="strata")
    result = oe.survey_mean(frame, design, ["tcresult", "tgresult"], missing="drop")
    oracle = frame.rename(columns={"strata": "h", "psu": "p"})
    w = oracle.finalwgt.to_numpy()
    theta, score = target(oracle, w, "mean", ["tcresult", "tgresult"])
    covariance = taylor(oracle, w, score)
    comparison(result.estimates, theta, atol=1e-9)
    comparison(result.covariance, covariance, atol=1e-9)
    table = result.to_frame()
    manual = np.array(
        [[211.3975, 1.252274, 208.8435, 213.9515], [138.5760, 2.071934, 134.3503, 142.8018]]
    )
    observed = table[["estimate", "std_error", "ci_low", "ci_high"]].to_numpy()
    np.testing.assert_allclose(observed[:, 0], manual[:, 0], rtol=0.0, atol=0.00005)
    np.testing.assert_allclose(observed[:, 1], manual[:, 1], rtol=0.0, atol=0.0000005)
    np.testing.assert_allclose(observed[:, 2:], manual[:, 2:], rtol=0.0, atol=0.00005)
    assert result.metadata["n_used"] == 5050 and result.df == 31
    assert result.metadata["sum_target_weights"] == 56820832.0
    critical = stats.t.isf(0.025, 31)
    comparison(table.ci_low, theta - critical * np.sqrt(np.diag(covariance)), atol=1e-9)
    comparison(table.p_value, 2 * stats.t.sf(abs(theta / np.sqrt(np.diag(covariance))), 31))
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "status": "official-manual-and-independent-full-inference-passed",
        "dataset_url": "https://www.stata-press.com/data/r19/nhanes2.dta",
        "manual_url": "https://www.stata.com/manuals/svyestat.pdf",
        "manual_example": "Example 1, page 7",
        "dataset_sha256": sha,
        "dataset_redistributed": False,
        "design_rows": len(frame),
        "sample_rows": 5050,
        "psus": 62,
        "strata": 31,
        "df": 31,
        "target_weight_sum": 56820832.0,
        "full_covariance": covariance.tolist(),
        "table": table.to_dict(orient="index"),
        "licensed_stata_execution": False,
        "stata_parity_validated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.states is None) == (args.dataset is None):
        parser.error(
            "Declare exactly one local states file or explicitly downloaded reference dataset."
        )
    record = check_states(args.states) if args.states else check_manual(args.dataset)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"]}))


if __name__ == "__main__":
    main()
