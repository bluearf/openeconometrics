"""Physical CSV/Parquet references and portable native verification for MARKET519.

Generation is a development step (SciPy/NumPy independent references). ``run``
uses native OpenEcon only; call ``verify`` inside a frozen/native runtime to
repeat the same physical files. The caller must independently prove package,
installed process and source pins; a stage label alone is not that proof.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations_with_replacement
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

METHODS = ("bspline_regress", "rcs_regress", "fp_regress", "mfp_regress")
POWERS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _basis(frame, transforms):
    """Independent dev reference, SciPy B-splines and literal FP/RCS formulas."""
    import numpy as np
    from scipy.interpolate import BSpline

    pieces = [np.ones((len(frame), 1))]
    for t in transforms:
        x = frame[t["column"]].to_numpy()
        if t["kind"] == "bs":
            lo, hi = t["boundary"]
            degree = t["degree"]
            knots = [lo] * (degree + 1) + t["knots"] + [hi] * (degree + 1)
            pieces.append(BSpline.design_matrix(x, knots, degree).toarray()[:, 1:])
        elif t["kind"] == "rcs":
            knots = np.asarray(t["knots"])
            z = (x - knots[0]) / (knots[-1] - knots[0])
            knots = (knots - knots[0]) / (knots[-1] - knots[0])
            pieces.append(
                np.column_stack(
                    [
                        z,
                        *[
                            np.maximum(z - a, 0) ** 3
                            - np.maximum(z - knots[-2], 0) ** 3
                            * (knots[-1] - a)
                            / (knots[-1] - knots[-2])
                            + np.maximum(z - knots[-1], 0) ** 3
                            * (knots[-2] - a)
                            / (knots[-1] - knots[-2])
                            for a in knots[:-2]
                        ],
                    ]
                )
            )
        elif t["kind"] == "fp":
            z = x / t["scale"]
            fp = []
            for j, p in enumerate(t["powers"]):
                value = np.log(z) if p == 0 else z**p
                fp.append(value * np.log(z) if j and p == t["powers"][j - 1] else value)
            if fp:
                pieces.append(np.column_stack(fp))
        else:
            pieces.append(x[:, None])
    return np.column_stack(pieces)


def _reference(frame, method):
    import numpy as np
    from scipy.stats import f, t

    frame = frame.dropna(subset=["x", "z", "y"])
    y = frame.y.to_numpy()
    candidates, closed, selected = None, None, None
    if method == "bspline_regress":
        transforms = [
            {
                "kind": "bs",
                "column": "x",
                "knots": np.quantile(frame.x, [0.25, 0.5, 0.75]).tolist(),
                "boundary": [float(frame.x.min()), float(frame.x.max())],
                "degree": 3,
            }
        ]
    elif method == "rcs_regress":
        transforms = [
            {
                "kind": "rcs",
                "column": "x",
                "knots": np.quantile(frame.x, np.linspace(0.05, 0.95, 5)).tolist(),
            }
        ]
    else:
        transforms = [
            {"kind": "fp", "column": "x", "powers": [0.0, 0.0], "scale": 2.0},
            {"kind": "linear", "column": "z"},
        ]
    if method == "mfp_regress":
        candidates = []
        for powers in [
            [],
            *[[p] for p in POWERS],
            *[list(p) for p in combinations_with_replacement(POWERS, 2)],
        ]:
            transforms[0].update(powers=powers, scale=1.0)
            x = _basis(frame, transforms)
            b = np.linalg.lstsq(x, y, rcond=None)[0]
            candidates.append(
                {"powers": powers, "rss": float(np.sum((y - x @ b) ** 2)), "n_parameters": len(b)}
            )
        best1 = min(range(1, 9), key=lambda j: (candidates[j]["rss"], j))
        best2 = min(range(9, 45), key=lambda j: (candidates[j]["rss"], j))
        full = candidates[best2]
        df = len(frame) - full["n_parameters"]
        linear = next(j for j, c in enumerate(candidates) if c["powers"] == [1.0])
        closed = []
        for reduced, dnum in [(0, 4), (linear, 3), (best1, 2)]:
            statistic = max(0.0, (candidates[reduced]["rss"] - full["rss"]) / dnum) / (
                full["rss"] / df
            )
            closed.append({"statistic": statistic, "p_value": float(f.sf(statistic, dnum, df))})
        selected = (
            0
            if closed[0]["p_value"] >= 0.05
            else linear
            if closed[1]["p_value"] >= 0.05
            else best1
            if closed[2]["p_value"] >= 0.05
            else best2
        )
        transforms[0]["powers"] = candidates[selected]["powers"]
    x = _basis(frame, transforms)
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    rss = float(np.sum((y - x @ b) ** 2))
    df = len(frame) - len(b)
    covariance = rss / df * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(covariance))
    critical = t.ppf(0.975, df)
    return {
        "nobs": len(frame),
        "parameters": b.tolist(),
        "covariance": covariance.tolist(),
        "rss": rss,
        "df": df,
        "p_value": (2 * t.sf(np.abs(b / se), df)).tolist(),
        "ci_low": (b - critical * se).tolist(),
        "ci_high": (b + critical * se).tolist(),
        "transforms": transforms,
        "candidates": candidates,
        "closed_tests": closed,
        "selected": selected,
    }


def generate(directory, rows):
    import numpy as np
    import pandas as pd

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if rows < 100001:
        raise ValueError("The physical fixture must exceed the resident smoothing row domain.")
    i = np.arange(rows)
    x, z = 0.25 + 4.5 * i / (rows - 1), np.cos(i * 0.47)
    frame = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": 2 + 1.3 * np.log(x) + 0.2 * z + 0.08 * np.sin(i * 0.61),
            "w": 1 + i % 7,
        }
    )
    frame.loc[[31, rows // 2, rows - 23], "y"] = np.nan
    frame.loc[[49, rows - 79], "x"] = np.nan
    frame.to_csv(directory / "training.csv", index=False)
    frame.to_parquet(
        directory / "training.parquet", index=False, row_group_size=8192, compression="zstd"
    )
    # CSV's own parser defines its actual float64 sample; reference each format.
    physical = {
        "csv": pd.read_csv(directory / "training.csv"),
        "parquet": pd.read_parquet(directory / "training.parquet"),
    }
    query_i = np.arange(67)
    query = pd.DataFrame(
        {"x": 0.3 + 4.3 * query_i / 66, "z": np.cos(query_i * 0.47), "w": 1 + query_i % 7}
    )
    query.loc[[0, 39], "x"] = np.nan
    query.to_parquet(directory / "query.parquet", index=False)
    cases = []
    for extension, parsed in physical.items():
        path = directory / ("training." + extension)
        for method in METHODS:
            reference = _reference(parsed, method)
            complete = query.dropna()
            basis = _basis(complete, reference["transforms"])
            prediction = basis @ np.asarray(reference["parameters"])
            covariance = np.asarray(reference["covariance"])
            se = np.sqrt(np.einsum("ij,jk,ik->i", basis, covariance, basis))
            average = np.average(basis, axis=0, weights=complete.w)
            reference.update(
                query_prediction=prediction.tolist(),
                query_std_error=se.tolist(),
                average_mean=float(average @ reference["parameters"]),
                average_std_error=float(np.sqrt(average @ covariance @ average)),
            )
            cases.append(
                {
                    "method": method,
                    "format": extension,
                    "path": path.name,
                    "source_sha256": digest(path),
                    "source_bytes": path.stat().st_size,
                    "rows": rows,
                    "options": {"powers": [0.0, 0.0], "scale": 2.0}
                    if method == "fp_regress"
                    else {},
                    "reference": reference,
                }
            )
    manifest = {
        "schema": 1,
        "scope": "physical files containing synthetic QA data; independent development NumPy/SciPy references; no customer/hardware scaling claim",
        "generator": "scripts/validate_dataset_smoothing.py",
        "oracle": "complete parsed source NumPy least squares, literal FP/RCS, SciPy B-spline and Student-t/F",
        "query_sha256": digest(directory / "query.parquet"),
        "cases": cases,
    }
    save(directory / "manifest.json", manifest)
    return manifest


def verify(directory, workspace_mb=32, output_directory=None):
    """Portable runtime helper; no SciPy or external estimation backend imports."""
    import numpy as np
    import pandas as pd
    import torch
    import openecon as oe
    from openecon.dataset import Dataset
    from openecon.models import ResultBundle
    from openecon.resources import use_workspace_budget

    directory = Path(directory)
    output = Path(output_directory or directory / "verification")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(directory / "query.parquet") != manifest["query_sha256"]:
        raise RuntimeError("Query source hash differs from the independent reference.")
    from openecon.econometrics.smoothing import common

    original_head, original_prepare = Dataset.head, common.prepare

    def no_collect(*args, **kwargs):
        raise RuntimeError("A whole-source Dataset collection was attempted.")

    Dataset.head = no_collect
    common.prepare = no_collect
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    records = []
    try:
        with use_workspace_budget(workspace_mb):
            for case in manifest["cases"]:
                path = directory / case["path"]
                if digest(path) != case["source_sha256"]:
                    raise RuntimeError("Physical training source changed.")
                start = time.perf_counter()
                result = getattr(oe, case["method"])(
                    data=oe.scan(path),
                    y="y",
                    x=["x", "z"] if "fp" in case["method"] else ["x"],
                    missing="drop",
                    **case["options"],
                )
                elapsed = time.perf_counter() - start
                reference = case["reference"]

                def values(name):
                    return [getattr(c, name) for c in result.coefficients]

                np.testing.assert_allclose(
                    values("estimate"), reference["parameters"], rtol=2e-7, atol=2e-9
                )
                np.testing.assert_allclose(
                    result.covariance_matrix, reference["covariance"], rtol=2e-7, atol=2e-10
                )
                for name in ("p_value", "ci_low", "ci_high"):
                    np.testing.assert_allclose(values(name), reference[name], rtol=2e-7, atol=2e-9)
                assert result.nobs == reference["nobs"] == case["rows"] - 5
                assert result.nobs_original == case["rows"] and result.dropped_rows == 5
                np.testing.assert_allclose(result.metrics["rss"], reference["rss"], rtol=2e-9)
                assert result.inference["df_inference"] == reference["df"]
                if case["method"] == "mfp_regress":
                    state = result.extra["smoothing_state"]
                    assert state["selected"] == reference["selected"]
                    np.testing.assert_allclose(
                        [c["rss"] for c in state["candidates"]],
                        [c["rss"] for c in reference["candidates"]],
                        rtol=2e-8,
                    )
                stem = case["method"] + "-" + case["format"]
                envelope = output / (stem + ".json")
                save(envelope, result.model_dump(mode="json"))
                restored = ResultBundle.model_validate_json(envelope.read_text())
                query = oe.scan(directory / "query.parquet")
                prediction = oe.smoothing_predict(
                    restored, data=query, interval=True, missing="drop"
                )
                # The owned output has65 rows; the physical fitting source is never collected.
                frame = pd.concat(list(prediction.iter_batches()), ignore_index=True)
                np.testing.assert_allclose(
                    frame["mean"], reference["query_prediction"], rtol=2e-7, atol=2e-9
                )
                np.testing.assert_allclose(
                    frame["std_error"], reference["query_std_error"], rtol=2e-7, atol=2e-10
                )
                margin = oe.smoothing_margins(
                    restored, data=query, averaging_weights="w", interval=True, missing="drop"
                )
                np.testing.assert_allclose(
                    margin.estimate.iloc[0], reference["average_mean"], rtol=2e-7, atol=2e-9
                )
                np.testing.assert_allclose(
                    margin.std_error.iloc[0], reference["average_std_error"], rtol=2e-7, atol=2e-10
                )
                latex = oe.to_latex(restored)
                margin_latex = oe.to_latex(margin)
                assert "selection uncertainty" in latex and "selection uncertainty" in margin_latex
                (output / (stem + ".tex")).write_text(latex)
                (output / (stem + "-margins.tex")).write_text(margin_latex)
                records.append(
                    {
                        "method": case["method"],
                        "format": case["format"],
                        "rows": case["rows"],
                        "retained_rows": result.nobs,
                        "source_sha256": case["source_sha256"],
                        "source_bytes": case["source_bytes"],
                        "seconds": elapsed,
                        "independent_full_inference": True,
                        "saved_dataset_prediction_and_weighted_margins": True,
                        "result_envelope_sha256": digest(envelope),
                        "latex_sha256": digest(output / (stem + ".tex")),
                        "margins_latex_sha256": digest(output / (stem + "-margins.tex")),
                        "streaming": result.provenance["streaming"],
                        "solver_diagnostics": result.provenance["solver_diagnostics"],
                    }
                )
    finally:
        Dataset.head, common.prepare = original_head, original_prepare
        torch.set_num_threads(previous_threads)
    return {
        "schema": 1,
        "scope": "eight physical source cases; source/frozen/installed identity must be joined externally; four new bounded routes, no all148 streaming claim",
        "runtime": {
            "executable": sys.executable,
            "frozen": bool(getattr(sys, "frozen", False)),
            "bundle_directory": getattr(sys, "_MEIPASS", None),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "scipy_loaded": "scipy" in sys.modules,
            "statsmodels_loaded": "statsmodels" in sys.modules,
        },
        "configured_workspace_mb": workspace_mb,
        "max_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_manifest_sha256": digest(directory / "manifest.json"),
        "cases": records,
        "resident_smoothing_fit_and_dataset_head_guards": True,
        "publication_and_persistence": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "run"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=250000)
    parser.add_argument("--workspace-mb", type=int, default=32)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    value = (
        generate(args.directory, args.rows)
        if args.command == "generate"
        else verify(args.directory, args.workspace_mb)
    )
    if args.receipt:
        save(args.receipt, value)
    print(json.dumps({"cases": len(value["cases"]), "command": args.command}))


if __name__ == "__main__":
    main()
