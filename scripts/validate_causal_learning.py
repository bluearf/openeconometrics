"""Bounded DGP calibration and reproducible independent DoubleML fixtures.

Run source mode with the checkout on PYTHONPATH. Reference mode runs in an
isolated DoubleML 0.11 environment and never becomes a runtime dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


def sample(n, seed, *, hetero=False, iv=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-0.3 * x[:, 0] + 0.2 * x[:, 1]))
    u = rng.uniform(size=n)
    z = rng.binomial(1, p)
    d = (u < 0.15 + 0.65 * z).astype(int) if iv else rng.binomial(1, p)
    tau = 2 + 1.5 * x[:, 0] if hetero else np.repeat(2.0, n)
    y = 1 + tau * d + 0.5 * x[:, 0] + rng.normal(size=n)
    if iv:
        y += 3 * (u - 0.5)  # endogenous treatment, valid independent instrument
    df = pd.DataFrame(x, columns=["x1", "x2", "x3"])
    df["y"], df["d"], df["z"] = y, d, z
    df["group"] = np.where(x[:, 0] > 0, "high", "low")
    df["policy"] = (x[:, 0] > 0).astype(float)
    return df


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - radius), float(center + radius)]


def local_forest_truth(state, query):
    """Independent traversal: admitted-C feature conditional mean target.

    This target averages the known DGP CATE with the fitted local leaf weights;
    it makes no statement about approximation bias at the queried point.
    """
    x = np.array(state["estimation_x"])
    weights = np.array(state["importance_weights"])

    def leaf(tree, point):
        node = 0
        while "feature" in tree[node]:
            item = tree[node]
            node = item["left"] if point[item["feature"]] <= item["threshold"] else item["right"]
        return node

    mass = np.zeros(len(x))
    for tree in state["trees"]:
        selected = np.array([leaf(tree, row) == leaf(tree, query) for row in x])
        mass += weights * selected / (weights * selected).sum() / len(state["trees"])
    return float(mass @ (2 + 1.5 * x[:, 0]))


def reference(root):
    import doubleml as dml
    from sklearn.linear_model import LinearRegression, LogisticRegression

    assert dml.__version__ == "0.11.0"
    records = {}
    for name in ("irm", "atet", "iivm"):
        df = pd.read_csv(root / (name + ".csv"))
        native = json.loads((root / (name + "-native.json")).read_text())
        data = dml.DoubleMLData(
            df,
            y_col="y",
            d_cols="d",
            x_cols=["x1", "x2", "x3"],
            z_cols="z" if name == "iivm" else None,
        )
        classifier = LogisticRegression(C=np.inf, l1_ratio=0, max_iter=10000, tol=1e-12)
        model = (
            dml.DoubleMLIIVM(data, LinearRegression(), classifier, classifier, n_folds=5)
            if name == "iivm"
            else dml.DoubleMLIRM(
                data,
                LinearRegression(),
                classifier,
                n_folds=5,
                score="ATTE" if name == "atet" else "ATE",
            )
        )
        model.set_sample_splitting(
            [
                (np.array(r["train_positions"]), np.array(r["test_positions"]))
                for r in native["extra"]["fold_records"]
            ]
        )
        model.fit()
        row = native["coefficients"][0]
        np.testing.assert_allclose(
            [row["estimate"], row["std_error"]], [model.coef[0], model.se[0]], atol=5e-7, rtol=2e-6
        )
        np.testing.assert_allclose(
            [row["ci_low"], row["ci_high"]], model.confint().to_numpy()[0], atol=1e-6
        )
        records[name] = {
            "reference": "DoubleML",
            "version": dml.__version__,
            "estimate": float(model.coef[0]),
            "std_error": float(model.se[0]),
            "native_estimate": row["estimate"],
            "native_std_error": row["std_error"],
            "difference": [
                row["estimate"] - float(model.coef[0]),
                row["std_error"] - float(model.se[0]),
            ],
            "data_sha256": hashlib.sha256((root / (name + ".csv")).read_bytes()).hexdigest(),
        }
    (root / "doubleml-reference.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


def source(root, repetitions):
    import openecon as oe

    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    for name, seed in (("irm", 701), ("atet", 701), ("iivm", 719)):
        df = sample(
            1800 if name == "iivm" else 1200, seed, hetero=name != "iivm", iv=name == "iivm"
        )
        result = (
            oe.dmliivm(data=df, y="y", treatment="d", instrument="z", x=["x1", "x2", "x3"])
            if name == "iivm"
            else oe.dmlirm(
                data=df,
                y="y",
                treatment="d",
                x=["x1", "x2", "x3"],
                estimand="atet" if name == "atet" else "ate",
            )
        )
        df.to_csv(root / (name + ".csv"), index=False)
        (root / (name + "-native.json")).write_text(result.model_dump_json(indent=2))
    # Independent Gauss-Hermite/Stein population ATET, with E[P(D=1|X)]=1/2.
    nodes, weights = np.polynomial.hermite.hermgauss(60)
    probabilities = 1 / (1 + np.exp(-np.sqrt(0.26) * nodes))
    true_atet = 2 + 0.9 * np.sum(weights * probabilities * (1 - probabilities)) / np.sqrt(np.pi)
    records = {
        name: []
        for name in [
            "IRM_constant",
            "IRM_heterogeneous",
            "ATET",
            "GATE_low",
            "GATE_high",
            "CATE_intercept",
            "CATE_slope",
            "policy",
            "IIVM",
            "forest_local_low",
            "forest_local_high",
        ]
    }
    for rep in range(repetitions):
        seed = 81731 + rep
        fixed, varied, iv = (
            sample(800, seed),
            sample(800, seed, hetero=True),
            sample(1000, seed, iv=True),
        )
        common = dict(y="y", treatment="d", x=["x1", "x2", "x3"])
        outputs = [
            ("IRM_constant", oe.dmlirm(data=fixed, **common), [2.0]),
            ("IRM_heterogeneous", oe.dmlirm(data=varied, **common), [2.0]),
            ("ATET", oe.dmlirm(data=varied, **common, estimand="atet"), [true_atet]),
            ("GATE", oe.dmlgate(data=varied, **common, group="group"), None),
            ("CATE", oe.dmlcate(data=varied, **common, basis=["x1"]), [2.0, 1.5]),
            (
                "policy",
                oe.policyvalue(data=varied, **common, policy="policy", cost=0.25),
                [1 + (2 - 0.25) / 2 + 1.5 / np.sqrt(2 * np.pi)],
            ),
            ("IIVM", oe.dmliivm(data=iv, **common, instrument="z"), [2.0]),
        ]
        for name, result, truth in outputs:
            if name == "GATE":
                truth = [
                    2 + (1 if label == "high" else -1) * 1.5 * np.sqrt(2 / np.pi)
                    for label in result.extra["group_labels"]
                ]
            for j, value in enumerate(truth):
                row = result.coefficients[j]
                key = (
                    ("GATE_" + result.extra["group_labels"][j])
                    if name == "GATE"
                    else ("CATE_intercept" if j == 0 else "CATE_slope")
                    if name == "CATE"
                    else name
                )
                records[key].append(
                    {
                        "truth": float(value),
                        "estimate": row.estimate,
                        "se": row.std_error,
                        "covered": bool(row.ci_low <= value <= row.ci_high),
                    }
                )
        queries = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        forest = oe.causalforest(
            data=varied, **common, trees=12, max_depth=3, min_leaf=12, query=queries
        )
        for j, point in enumerate(queries):
            truth = local_forest_truth(forest.extra["cate_state"], point)
            row = forest.extra["cate"][j]
            key = "forest_local_low" if j == 0 else "forest_local_high"
            records[key].append(
                {
                    "truth": truth,
                    "estimate": row["cate"],
                    "se": row["std_error"],
                    "covered": bool(row["ci_low"] <= truth <= row["ci_high"]),
                }
            )
    receipt = {
        "replications": repetitions,
        "seed": 81731,
        "domains": "Declared IID linear/heterogeneous binary IRM, endogenous binary-IV and honest forest conditional local targets; nominal 95% intervals. Forest target excludes pointwise approximation bias.",
        "elapsed_seconds": time.perf_counter() - started,
        "experiments": {},
    }
    for name, rows in records.items():
        successes = int(sum(row["covered"] for row in rows))
        interval = wilson(successes, len(rows))
        receipt["experiments"][name] = {
            "coverage": successes / len(rows),
            "covered": successes,
            "wilson95": interval,
            "bias": float(np.mean([r["estimate"] - r["truth"] for r in rows])),
            "mean_standard_error": float(np.mean([r["se"] for r in rows])),
            "runs": rows,
        }
        assert successes / len(rows) >= 0.80 and interval[1] >= 0.95, (
            name,
            receipt["experiments"][name],
        )
    (root / "bounded-coverage.json").write_text(json.dumps(receipt, indent=2))
    print(
        json.dumps(
            {
                name: {k: v for k, v in record.items() if k != "runs"}
                for name, record in receipt["experiments"].items()
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/causal-validation"))
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--replications", type=int, default=40)
    args = parser.parse_args()
    if args.reference:
        reference(args.output)
    else:
        if args.replications < 20:
            raise ValueError("At least 20 replications are required.")
        source(args.output, args.replications)


if __name__ == "__main__":
    main()
