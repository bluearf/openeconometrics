"""Independently compare the exact frozen physical8 fixture results to source.

No native-window claim: helper execution/reset/history, full result persistence,
compiled SDK identity, result/sample checksums and numerical fields are separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_hdfe_projection_benchmark import compare, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "manifest", "source", "frozen", "helper", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    protocol, manifest = read(args.protocol), read(args.manifest)
    identity, summary = read(args.frozen / "identity.json"), read(args.frozen / "summary.json")
    assert identity["all_local_analysis_sdk_embedded_code_matches_source"] is True
    assert identity["strict_ad_hoc_signature"] is True
    expected = {
        f"{model}-{rows}-{suffix}"
        for model in protocol["models"]
        for rows in protocol["physical_rows"]
        for suffix in protocol["formats"]
    }
    ids = [x["id"] for x in manifest["cases"]]
    assert len(ids) == len(expected) and set(ids) == expected
    assert summary["planned"] == len(expected) and summary["passed"] == len(expected)
    assert {r["name"] for r in summary["cases"]} == expected
    rows = []
    for case in manifest["cases"]:
        row = {"id": case["id"], "status": "failed", "failures": []}
        errors = row["failures"]
        try:
            frozen_dir = args.frozen / case["id"]
            acceptance = read(frozen_dir / "acceptance.json")
            assert acceptance["status"] == "passed"
            for flag in ("frozen", "history_equal_after_reset", "owned_project_removed"):
                assert acceptance[flag] is True, flag
            assert acceptance["helper_sha256"] == sha(args.helper), "helper identity differs"
            before, after = (
                read(args.source / case["id"] / "receipt.json"),
                read(frozen_dir / "results/receipt.json"),
            )
            assert after == read(frozen_dir / "helper.json"), (
                "persisted helper/result receipt mismatch"
            )
            assert before["case"] == after["case"] == case
            assert before["status"] == after["status"] == "passed"
            assert after["threads"] == protocol["threads"]
            assert after["scratch_clean"] and after["source_unchanged"]
            sdk = after["sdk_path"]
            assert sdk.startswith(
                str(Path(identity["installed_app"]) / "Contents/Resources/runtime/")
            ), "SDK outside installed bundle"
            fits = []
            for i in range(protocol["fits_per_case"]):
                p0 = args.source / case["id"] / f"result-{i}.json.gz"
                p1 = frozen_dir / "results" / f"result-{i}.json.gz"
                assert sha(p0) == before["measurements"][i]["result_sha256"]
                assert sha(p1) == after["measurements"][i]["result_sha256"]
                a, b = read(p0), read(p1)
                fields = (
                    "coefficients",
                    "covariance_matrix",
                    "metrics",
                    "inference",
                    "tests",
                    "predictions",
                    "extra",
                )
                gates = protocol["numeric_gates"]
                for field, rtol, atol in (
                    ("coefficients", gates["coefficient_rtol"], gates["coefficient_atol"]),
                    ("covariance_matrix", gates["covariance_rtol"], gates["covariance_atol"]),
                    ("metrics", gates["likelihood_rtol"], gates["likelihood_atol"]),
                    ("inference", gates["inference_rtol"], gates["inference_atol"]),
                    ("tests", gates["inference_rtol"], gates["inference_atol"]),
                    ("predictions", gates["coefficient_rtol"], gates["coefficient_atol"]),
                    ("extra", gates["inference_rtol"], gates["inference_atol"]),
                ):
                    compare(a[field], b[field], field, rtol=rtol, atol=atol)
                for field in (
                    "spec",
                    "nobs",
                    "nobs_original",
                    "dropped_rows",
                    "sample_positions",
                    "warnings",
                ):
                    assert a[field] == b[field], f"exact {field} differs"
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
                    assert a["provenance"][field] == b["provenance"][field], (
                        f"exact provenance.{field} differs"
                    )
                assert (
                    a["provenance"]["streaming"]["passes"] == b["provenance"]["streaming"]["passes"]
                )
                fits.append(
                    {
                        "fit": i,
                        "numerical_equivalence": True,
                        "all_scientific_fields_bit_identical": all(a[f] == b[f] for f in fields),
                        "source_passes": a["provenance"]["streaming"]["passes"],
                        "frozen_passes": b["provenance"]["streaming"]["passes"],
                        "source_result_sha256": sha(p0),
                        "frozen_result_sha256": sha(p1),
                        "frozen_seconds": after["measurements"][i]["fit_seconds"],
                    }
                )
            row.update(
                fits=fits,
                native_window_verified=acceptance["native_window_verified"],
                executed_helper_sha256=acceptance["helper_sha256"],
                frozen_sdk_path=sdk,
                python=after["python"],
                torch=after["torch"],
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        row["status"] = "failed" if errors else "passed"
        rows.append(row)
    value = {
        "schema": 1,
        "status": "passed" if all(r["status"] == "passed" for r in rows) else "failed",
        "planned_cases": len(expected),
        "passed_cases": sum(r["status"] == "passed" for r in rows),
        "paired_fits": sum(len(r.get("fits", [])) for r in rows),
        "all_scientific_fields_bit_identical": all(
            f["all_scientific_fields_bit_identical"] for r in rows for f in r.get("fits", [])
        )
        and all(r["status"] == "passed" for r in rows),
        "protocol_sha256": sha(args.protocol),
        "manifest_sha256": sha(args.manifest),
        "helper_sha256": sha(args.helper),
        "bundle_identity": identity,
        "cases": rows,
        "boundaries": [
            "Frozen helper from installed bundle; actual native-window/restart proof is separately root-owned",
            "Timings on contended host, not a controlled baseline comparison; no frozen timing/RSS improvement gate is inferred",
            "Same8 synthetic physical fixtures/options/source geometry; no universal model/million-row/CUDA claim",
            "Initial relative-output execution-only attempt is preserved separately as persistence harness failure",
        ],
    }
    args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "status",
                    "planned_cases",
                    "passed_cases",
                    "paired_fits",
                    "all_scientific_fields_bit_identical",
                )
            }
        )
    )
    return int(value["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
