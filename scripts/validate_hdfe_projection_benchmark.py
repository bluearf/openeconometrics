"""Audit all frozen physical FE cases against their original source results.

Standard-library-only auditor; unavailable/failed cases stay in the denominator.
The input receipts/results are read-only and remain separate from this report.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a, b, name, *, rtol, atol):
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        if not isinstance(b, (int, float)) or isinstance(b, bool):
            raise AssertionError(f"{name}: incompatible numeric type")
        if (
            not math.isfinite(a)
            or not math.isfinite(b)
            or not math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
        ):
            raise AssertionError(f"{name}: {a!r} != {b!r}")
        return
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f"{name}: different fields")
        for key in a:
            compare(a[key], b[key], name + "." + key, rtol=rtol, atol=atol)
        return
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError(f"{name}: different lengths")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{name}[{i}]", rtol=rtol, atol=atol)
        return
    if a != b:
        raise AssertionError(f"{name}: {a!r} != {b!r}")


def audit(protocol, manifest, baseline, optimized):
    gates = protocol["numeric_gates"]
    performance = protocol["performance_gates"]
    outputs = []
    for case in manifest["cases"]:
        row = {"id": case["id"], "status": "failed", "failures": []}
        errors = row["failures"]
        try:
            before = read(baseline / case["id"] / "receipt.json")
            after = read(optimized / case["id"] / "receipt.json")
            for item in (before, after):
                assert item["status"] == "passed", item.get("error", "worker failed")
                assert item["case"] == case, "physical fixture/spec mismatch"
                assert item["threads"] == protocol["threads"], "thread mismatch"
                assert item["scratch_clean"] and item["source_unchanged"], (
                    "scratch/source guard failed"
                )
            assert before["python"] == after["python"] and before["torch"] == after["torch"], (
                "runtime versions differ"
            )
            fits = []
            for i in range(protocol["fits_per_case"]):
                p0 = baseline / case["id"] / f"result-{i}.json.gz"
                p1 = optimized / case["id"] / f"result-{i}.json.gz"
                assert sha(p0) == before["measurements"][i]["result_sha256"], (
                    "baseline result checksum"
                )
                assert sha(p1) == after["measurements"][i]["result_sha256"], (
                    "optimized result checksum"
                )
                a, b = read(p0), read(p1)
                numeric_errors = []
                for field, rtol, atol in (
                    ("coefficients", gates["coefficient_rtol"], gates["coefficient_atol"]),
                    ("covariance_matrix", gates["covariance_rtol"], gates["covariance_atol"]),
                    ("metrics", gates["likelihood_rtol"], gates["likelihood_atol"]),
                    ("inference", gates["inference_rtol"], gates["inference_atol"]),
                    ("tests", gates["inference_rtol"], gates["inference_atol"]),
                    ("predictions", gates["coefficient_rtol"], gates["coefficient_atol"]),
                    ("extra", gates["inference_rtol"], gates["inference_atol"]),
                ):
                    try:
                        compare(a[field], b[field], field, rtol=rtol, atol=atol)
                    except AssertionError as error:
                        numeric_errors.append(str(error))
                for field in (
                    "spec",
                    "nobs",
                    "nobs_original",
                    "dropped_rows",
                    "sample_positions",
                    "warnings",
                ):
                    if a[field] != b[field]:
                        numeric_errors.append(f"exact {field} differs")
                for field in (
                    "data_hash",
                    "sample_hash",
                    "sample_positions_hash",
                    "input_columns",
                    "design_terms",
                    "categorical_encoding",
                    "omitted_terms",
                    "weights",
                    "precision",
                ):
                    if a["provenance"][field] != b["provenance"][field]:
                        numeric_errors.append(f"exact provenance.{field} differs")
                first, second = before["measurements"][i], after["measurements"][i]
                ratio = second["fit_seconds"] / first["fit_seconds"]
                reduction = 1 - second["passes"] / first["passes"]
                if ratio > performance["maximum_fit_time_ratio_each_case"]:
                    errors.append(f"fit{i} timing ratio exceeds gate: {ratio}")
                if reduction < performance["minimum_source_pass_reduction_each_case"]:
                    errors.append(f"fit{i} pass reduction below gate: {reduction}")
                errors.extend(numeric_errors)
                fits.append(
                    {
                        "fit": i,
                        "baseline_seconds": first["fit_seconds"],
                        "optimized_seconds": second["fit_seconds"],
                        "time_ratio": ratio,
                        "baseline_passes": first["passes"],
                        "optimized_passes": second["passes"],
                        "pass_reduction": reduction,
                        "numerical_equivalence": not numeric_errors,
                        "all_checked_scientific_fields_bit_identical": all(
                            a[f] == b[f]
                            for f in (
                                "coefficients",
                                "covariance_matrix",
                                "metrics",
                                "inference",
                                "tests",
                                "predictions",
                                "extra",
                            )
                        ),
                    }
                )
            addition = after["peak_process_rss_bytes"] - before["peak_process_rss_bytes"]
            if addition > performance["maximum_RSS_addition_bytes"]:
                errors.append(f"parent-process RSS addition exceeds gate: {addition}")
            row.update(
                fits=fits,
                baseline_RSS=before["peak_process_rss_bytes"],
                optimized_RSS=after["peak_process_rss_bytes"],
                RSS_addition=addition,
                baseline_scratch=before["sampled_scratch_peak_bytes"],
                optimized_scratch=after["sampled_scratch_peak_bytes"],
                baseline_feature_sha256=before["feature_sha256"],
                optimized_feature_sha256=after["feature_sha256"],
            )
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        row["status"] = "failed" if errors else "passed"
        outputs.append(row)
    return {
        "schema": 1,
        "denominator": len(manifest["cases"]),
        "passed": sum(x["status"] == "passed" for x in outputs),
        "failed": sum(x["status"] == "failed" for x in outputs),
        "status": "passed" if all(x["status"] == "passed" for x in outputs) else "failed",
        "boundaries": protocol["boundaries"],
        "cases": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "manifest", "baseline", "optimized", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    protocol, manifest = read(args.protocol), read(args.manifest)
    expected = {
        f"{model}-{rows}-{suffix}"
        for model in protocol["models"]
        for rows in protocol["physical_rows"]
        for suffix in protocol["formats"]
    }
    observed = [case["id"] for case in manifest["cases"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(
            "The physical manifest does not include every frozen protocol case exactly once"
        )
    value = audit(protocol, manifest, args.baseline, args.optimized)
    value["protocol_sha256"] = sha(args.protocol)
    value["manifest_sha256"] = sha(args.manifest)
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value[key] for key in ("status", "denominator", "passed", "failed")}))
    return int(value["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
