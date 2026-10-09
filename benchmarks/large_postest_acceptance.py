"""Compare every physical prediction row and global margins from saved fits.

The resident/replay fit results come from the predeclared physical scale run.
Evaluation shares the public prediction implementation; it is a saved-fit and
full-covariance consistency check, not an independent statistical oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_case(case, source_directory, output_directory):
    import pandas as pd
    import torch
    import openecon as oe
    torch.set_num_threads(2)
    started = time.monotonic()
    assert digest(case["path"]) == case["source_sha256"], "Physical source changed"
    base = Path(source_directory) / case["id"]
    resident = oe.ResultBundle.model_validate_json((base / "resident/result.json").read_text())
    replay = oe.ResultBundle.model_validate_json((base / "replay/result.json").read_text())
    data = oe.scan(case["path"])
    options = {"kind": "xb"} if case["model"] in {"areg", "xtreg"} else {}
    prediction = oe.predict(replay, data=data, interval="mean", batch_rows=8192, **options)
    observed = prediction.iter_batches(batch_rows=8192)
    original = data.iter_batches(batch_rows=8192)
    rows = batches = missing = 0
    maximum_error = 0.0
    try:
        for raw, actual in zip(original, observed, strict=True):
            expected = oe.predict(resident, data=raw, interval="mean", **options)
            pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True),
                                          check_dtype=False, rtol=5e-5, atol=2e-6)
            expected, actual = expected.reset_index(drop=True), actual.reset_index(drop=True)
            values = (expected-actual).abs()
            maximum_error = max(maximum_error, float(values.max().max()))
            rows += len(actual)
            batches += 1
            missing += int(actual.isna().any(axis=1).sum())
        assert rows == case["rows"]
        assert batches > 1
        prediction_metadata = prediction.provenance
    finally:
        original.close()
        observed.close()
        prediction.close()
    variables = [name for name in replay.spec.predictors if name not in replay.spec.categorical]
    actual_margins = oe.margins(replay, data=data, variables=variables, batch_rows=8192, **options)
    expected_margins = oe.margins(resident, data=data, variables=variables, batch_rows=8192, **options)
    pd.testing.assert_frame_equal(expected_margins, actual_margins, check_dtype=False, rtol=5e-5, atol=2e-6)
    assert digest(case["path"]) == case["source_sha256"], "Physical source changed during evaluation"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {"status": "passed", "model": case["model"], "physical_rows": rows,
        "prediction_batches": batches, "maximum_batch_rows": 8192,
        "missing_prediction_rows": missing, "maximum_prediction_inference_error": maximum_error,
        "source_sha256": case["source_sha256"], "source_identity_unchanged": True,
        "resident_result_sha256": digest(base / "resident/result.json"),
        "replay_result_sha256": digest(base / "replay/result.json"),
        "prediction_provenance": prediction_metadata,
        "margins": json.loads(actual_margins.to_json(orient="split", double_precision=15)),
        "seconds": time.monotonic()-started, "frozen": bool(getattr(sys, "frozen", False)),
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss*1024),
        "no_refit": True, "whole_source_frame_collected": False,
        "reference": "Same public evaluation implementation, separately saved resident fit versus replay fit; not independent oracle"}
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / (case["id"]+".json")).write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"model": case["model"], "rows": rows, "status": "passed"}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    records = []
    cases = json.loads(args.manifest.read_text())["cases"]
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for case in cases:
        try:
            records.append(verify_case(case, args.source_directory, args.directory))
        except Exception as exc:
            record = {"status": "failed", "model": case["model"], "physical_rows": case["rows"],
                      "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            (args.directory / (case["id"]+".json")).write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps(record), flush=True)
    passed = sum(row["status"] == "passed" for row in records)
    (args.directory / "summary.json").write_text(json.dumps({"cases": records,
        "planned": len(cases), "passed": passed, "failed": len(cases)-passed,
        "source_commit": source_commit,
        "scope": "full-row saved predict/margins, CPU float64"}, indent=2)+"\n")
    raise SystemExit(0 if passed == len(cases) and passed else 1)
