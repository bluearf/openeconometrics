"""Owned frozen helpers, separate from source and actual native-window proof."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_bai_perron_runtime import OwnedRuntime  # noqa: E402
from verify_native_matrix_desktop import bundled_identity  # noqa: E402


def validate_planned_cases(cases, mode):
    if not isinstance(cases, list) or not cases:
        raise ValueError("A nonempty planned helper case inventory is required")
    if mode == "hdfe":
        names = [case["id"] for case in cases]
        if len(names) != len(set(names)) or any(
            not n or n in {".", ".."} or Path(n).name != n for n in names
        ):
            raise ValueError("HDFE cases require unique safe identifiers")
    elif mode == "smoothing":
        names = [(case["method"], case["format"]) for case in cases]
        required = {
            (method, format_)
            for method in ("bspline_regress", "rcs_regress", "fp_regress", "mfp_regress")
            for format_ in ("csv", "parquet")
        }
        if len(names) != 8 or set(names) != required:
            raise ValueError("Smoothing requires every declared method/format case exactly once")
    else:
        raise ValueError("Unknown helper contract")


def validate_helper_result(proof, mode, planned_cases):
    """An OK console marker is transport proof, not a scientific pass."""
    validate_planned_cases(planned_cases, mode)
    if not isinstance(proof, dict):
        raise ValueError("Helper did not return a result mapping")
    if mode == "hdfe":
        if (
            len(planned_cases) != 1
            or proof.get("status") != "passed"
            or proof.get("case") != planned_cases[0]
            or proof.get("source_unchanged") is not True
            or proof.get("scratch_clean") is not True
        ):
            raise ValueError("HDFE helper failed or changed its planned case/source/cleanup")
        measurements = proof.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != 2:
            raise ValueError("HDFE helper must retain both planned first/warm fits")
        for measurement in measurements:
            if (
                type(measurement.get("nobs")) is not int
                or measurement["nobs"] <= 0
                or type(measurement.get("nobs_original")) is not int
                or measurement["nobs_original"] <= 0
                or not isinstance(measurement.get("fit_seconds"), (int, float))
                or not math.isfinite(measurement["fit_seconds"])
                or measurement["fit_seconds"] < 0
                or not isinstance(measurement.get("result_sha256"), str)
                or len(measurement["result_sha256"]) != 64
            ):
                raise ValueError("HDFE helper has incomplete fit measurements")
    else:
        records = proof.get("cases")
        if not isinstance(records, list) or len(records) != len(planned_cases):
            raise ValueError("Smoothing helper omitted planned cases")
        if (
            proof.get("publication_and_persistence") is not True
            or proof.get("resident_smoothing_fit_and_dataset_head_guards") is not True
        ):
            raise ValueError("Smoothing helper lacks declared persistence/resource checks")
        for expected, measured in zip(planned_cases, records, strict=True):
            if (
                any(
                    measured.get(key) != expected[key]
                    for key in ("method", "format", "rows", "source_sha256", "source_bytes")
                )
                or measured.get("retained_rows") != expected["reference"]["nobs"]
                or measured.get("independent_full_inference") is not True
                or measured.get("saved_dataset_prediction_and_weighted_margins") is not True
            ):
                raise ValueError(
                    "Smoothing helper result does not match its planned case/reference"
                )


class LongRuntime(OwnedRuntime):
    def request(self, path, body=None, method=None, *, desktop=False):
        d = self.descriptor
        req = Request(
            d["url"] + path,
            headers={
                "Content-Type": "application/json",
                "X-OpenEcon-Token": d["token"] if desktop else self.token,
            },
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
        )
        with urlopen(req, timeout=1860) as response:
            return json.load(response)


def run_one(app, source, expression, output, name, *, mode, planned_cases):
    validate_planned_cases(planned_cases, mode)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="openecon-priority-helper-owned-") as temporary:
        root = Path(temporary)
        runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
        owned = LongRuntime(runtime, root, "unused")
        code = f"""import importlib.util,sys,json,hashlib
from pathlib import Path
import openecon as oe
assert getattr(sys,'frozen',False)
assert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
assert hashlib.sha256(Path({str(source)!r}).read_bytes()).hexdigest()=={source_sha256!r}
spec=importlib.util.spec_from_file_location('owned_priority_helper', {str(source)!r})
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
proof={expression}
assert hashlib.sha256(Path({str(source)!r}).read_bytes()).hexdigest()=={source_sha256!r}
assert not any(n=='scipy' or n.startswith('scipy.') for n in sys.modules)
Path({str(output / "helper.json")!r}).write_text(json.dumps(proof,indent=2))
print('FROZEN_HELPER_OK ' + {name!r})
"""
        try:
            owned.open_project(create=True)
            run = owned.call("/console/execute", {"code": code, "timeout_seconds": 1800})
            (output / "console.json").write_text(json.dumps(run, indent=2))
            owned.call("/console/reset", {})
            saved = next(row for row in owned.call("/console")["history"] if row["id"] == run["id"])
            fields = ("code", "stdout", "outputs", "events")
            equal = {key: saved[key] for key in fields if key in saved} == {
                key: run[key] for key in fields if key in run
            }
            passed = run["status"] == "ok" and "FROZEN_HELPER_OK " + name in run["stdout"] and equal
            receipt = {
                "name": name,
                "status": "passed" if passed else "failed",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "helper_sha256": source_sha256,
                "history_equal_after_reset": equal,
                "duration_ms": run["duration_ms"],
                "frozen": passed,
                "native_window_verified": False,
                "error": run.get("error"),
            }
        finally:
            owned.close()
    receipt["owned_project_removed"] = not root.exists()
    # A successful console call alone is insufficient: the helper result must
    # survive outside the disposable execution project.
    payload = output / "helper.json"
    receipt["helper_payload_persisted"] = payload.is_file()
    if payload.is_file():
        receipt["helper_payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
        try:
            validate_helper_result(json.loads(payload.read_text()), mode, planned_cases)
            receipt["helper_semantic_contract_passed"] = True
        except (ValueError, KeyError, TypeError) as error:
            receipt["helper_semantic_contract_passed"] = False
            receipt["status"] = "failed"
            receipt["error"] = f"{type(error).__name__}: {error}"
    else:
        receipt["status"] = "failed"
        receipt["error"] = "Helper result did not survive owned project cleanup"
    (output / "acceptance.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoothing", "hdfe"], required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, help="Smoothing physical fixture directory")
    parser.add_argument("--manifest", type=Path, help="HDFE predeclared physical manifest")
    args = parser.parse_args()
    args.app = args.app.resolve()
    args.directory = args.directory.resolve()
    if args.mode == "smoothing" and not args.fixtures:
        parser.error("--fixtures is required for smoothing")
    if args.mode == "hdfe" and not args.manifest:
        parser.error("--manifest is required for hdfe")
    args.directory.mkdir(parents=True, exist_ok=False)
    identity = bundled_identity(args.app, "org.openecon.qa.priorityseven")
    (args.directory / "identity.json").write_text(json.dumps(identity, indent=2))
    if args.mode == "smoothing":
        fixtures = args.fixtures.resolve()
        planned_cases = json.loads((fixtures / "manifest.json").read_text())["cases"]
        validate_planned_cases(planned_cases, "smoothing")
        outputs = args.directory / "results"
        expression = (
            f"helper.verify(Path({str(fixtures)!r}),32,output_directory=Path({str(outputs)!r}))"
        )
        receipts = [
            run_one(
                args.app,
                ROOT / "scripts/validate_dataset_smoothing.py",
                expression,
                args.directory / "worker",
                "smoothing-eight",
                mode="smoothing",
                planned_cases=planned_cases,
            )
        ]
    else:
        manifest = json.loads(args.manifest.resolve().read_text())
        cases = manifest["cases"]
        validate_planned_cases(cases, "hdfe")
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = []
            for case in cases:
                name = case["id"]
                outputs = args.directory / name / "results"
                expression = f"helper.run_case({case!r},Path({str(outputs)!r}))"
                tasks.append(
                    pool.submit(
                        run_one,
                        args.app,
                        ROOT / "benchmarks/hdfe_projection_benchmark.py",
                        expression,
                        args.directory / name,
                        name,
                        mode="hdfe",
                        planned_cases=[case],
                    )
                )
            receipts = []
            for case, task in zip(cases, tasks, strict=True):
                try:
                    receipts.append(task.result())
                except Exception as exc:
                    receipts.append({"name": case["id"], "status": "failed", "error": repr(exc)})
    (args.directory / "summary.json").write_text(
        json.dumps(
            {
                "cases": receipts,
                "passed": sum(r["status"] == "passed" for r in receipts),
                "planned": len(receipts),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if receipts and all(r["status"] == "passed" for r in receipts) else 1)
