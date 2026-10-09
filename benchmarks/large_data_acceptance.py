"""Predeclared, physical 100k/1m IV/FE/count/panel/time-series acceptance.

Source and frozen/installed execution are separate stages. Fixtures are seeded
synthetic development data. Resident comparisons share implementation primitives;
this is full reported-inference consistency, not an independent vendor oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages/openecon-charts/src"))
from native_model_matrix_replay import digest, write, run  # noqa: E402

MODELS = ["ivregress", "areg", "poisson", "xtreg", "prais"]
SIZES = [100_000, 1_000_000]


def fixture(name, rows):
    from native_model_matrix_fixtures import fixture as original
    if name != "xtreg":
        return original(name, rows)
    import pandas as pd
    import torch
    from openecon.models import ModelSpec
    generator = torch.Generator().manual_seed(20261007)
    x, z, error = torch.randn((3, rows), generator=generator, dtype=torch.float64)
    group = torch.arange(rows) // 50
    time = torch.arange(rows) % 50
    y = .5*x-.3*z+.2*torch.sin(group.to(torch.float64))+error
    frame = pd.DataFrame({"x": x.numpy(), "z": z.numpy(), "y": y.numpy(),
                          "g": group.numpy(), "t": time.numpy(),
                          "w": (1+group % 3).numpy()})
    frame.loc[23::9973, "x"] = float("nan")
    spec = ModelSpec(estimator="xtreg", outcome="y", predictors=["x", "z"],
        panel="g", time="t", covariance="cluster", cluster="g", weights="w",
        weight_type="fweight", missing="drop", options={"model": "fe"})
    return frame, spec


def generate(directory, sizes=SIZES):
    import torch
    torch.set_num_threads(2)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {"schema": 1, "models": MODELS, "physical_rows": sizes,
        "format_by_size": {str(rows): "csv" if index % 2 == 0 else "parquet"
                           for index, rows in enumerate(sizes)},
        "model_geometry": {
            "ivregress": "2SLS; one endogenous/two excluded instruments; robust V; full first-stage diagnostics",
            "areg": "two numeric regressors;31absorbed groups; robust V",
            "poisson": "numeric and categorical regressors;23clusters;frequency weights;zero-weight and missing rows",
            "xtreg": "FE;50periods per growing group;group-cluster V;panel-constant frequency weights;missing x",
            "prais": "one numeric regressor;sorted actual periods;AR(1) errors;robust V"},
        "worker_threads": 2, "timeout_seconds_per_mode": 600,
        "relative_tolerance": 5e-5, "absolute_tolerance": 2e-6,
        "comparison": "complete reported coefficients/V/SE/df/p/CI/sample/metrics/diagnostics plus bounded saved helpers",
        "reference_boundary": "Resident route on identical parsed physical rows; shared kernels, not independent oracle",
        "cache_boundary": "fresh processes; hashing warms OS pages; no cold-cache claim",
        "failure_policy": "Retain every predeclared case, refusal and timeout in denominator",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "generator_sha256": digest(__file__)}
    write(directory / "protocol.json", protocol)  # Frozen before any measurements.
    cases = []
    for rows in sizes:
        extension = protocol["format_by_size"][str(rows)]
        for name in MODELS:
            frame, spec = fixture(name, rows)
            assert len(frame) == rows
            path = directory / f"{name}-{rows}.{extension}"
            if extension == "csv":
                frame.to_csv(path, index=False)
            else:
                frame.to_parquet(path, index=False, row_group_size=8192, compression="zstd")
            cases.append({"id": f"{name}-{rows}-{digest(path)[:12]}", "model": name,
                "path": str(path.resolve()), "rows": rows, "spec": spec.model_dump(mode="json"),
                "source_sha256": digest(path), "source_bytes": path.stat().st_size,
                "metadata": frame.attrs.get("metadata", {}),
                "geometry": protocol["model_geometry"][name]})
            print(json.dumps({"generated": name, "rows": rows, "format": extension}), flush=True)
    manifest = {"schema": 1, "stage": "predeclared_physical_scale", "cases": cases,
                "models": MODELS, "protocol_sha256": digest(directory / "protocol.json")}
    write(directory / "manifest.json", manifest)


def summarize(directory):
    report = json.loads((directory / "matrix.json").read_text())
    assert len(report["cases"]) == len(MODELS)*len(SIZES)
    keys = {(case["case"]["model"], case["case"]["rows"]) for case in report["cases"]}
    assert keys == {(name, rows) for name in MODELS for rows in SIZES}
    rows = []
    for case in report["cases"]:
        modes = case["modes"]
        replay = modes["replay"]
        rows.append({"model": case["case"]["model"], "physical_rows": case["case"]["rows"],
            "status": case["status"], "fit_seconds": replay.get("fit_seconds"),
            "warm_fit_seconds": replay.get("warm", {}).get("fit_seconds"),
            "completed_fit_source_passes": replay.get("cold_source_reads", {}).get("completed_passes"),
            "fit_peak_rss_MiB": replay.get("cold_process_peak_rss_bytes", 0)/1024**2,
            "total_peak_rss_MiB": replay.get("process_peak_rss_bytes", 0)/1024**2,
            "sampled_scratch_peak_bytes": replay.get("sampled_scratch_peak_bytes"),
            "comparison": case["comparison"], "failure": replay.get("error")})
    summary = {"schema": 1, "cases": rows, "passed": report["passed"], "failed": report["failed"],
        "source_commit": report["code_before"]["git_commit"],
        "code_aggregate_sha256": report["code_before"]["aggregate_sha256"],
        "matrix_sha256": digest(directory / "matrix.json"),
        "installed_or_frozen_verified": False, "native_ui_verified": False,
        "independent_vendor_reference": False, "other_models_or_geometries_measured": False}
    write(directory / "summary.json", summary)
    text = ["# Physical scale acceptance", "", "Synthetic CPU float64 data; fresh processes with warm OS pages.",
            "Resident entry shares kernels; not independent Stata parity. Installed/UI proof is separate.", "",
            "| Model | Physical rows | Status | Cold fit s | Warm fit s | Fit passes | Total peak MiB |",
            "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['model']} | {row['physical_rows']} | {row['status']} | "
                    f"{row['fit_seconds']} | {row['warm_fit_seconds']} | "
                    f"{row['completed_fit_source_passes']} | {row['total_peak_rss_MiB']:.1f} |")
    (directory / "measurements.md").write_text("\n".join(text)+"\n")
    print(json.dumps({"passed": summary["passed"], "failed": summary["failed"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    maker = sub.add_parser("generate")
    maker.add_argument("--directory", type=Path, required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--manifest", type=Path, required=True)
    runner.add_argument("--directory", type=Path, required=True)
    report = sub.add_parser("summarize")
    report.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.directory.resolve())
    elif args.command == "run":
        args.models, args.cases_per_model, args.skip_warm, args.timeout = MODELS, 2, False, 600
        raise SystemExit(run(args))
    else:
        summarize(args.directory)
