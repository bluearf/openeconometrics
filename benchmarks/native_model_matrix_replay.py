"""Fresh-process physical replay and bounded resident-reference matrix.

Run capture first with native_model_matrix.py, then:
  python benchmarks/native_model_matrix_replay.py run --manifest captured.json \
      --directory /absolute/owned/evidence --models ols logit

Physical inputs are deterministic development fixtures, not customer data.
Resident reference routes can share primitives or algorithms with replay; the
receipt names the entry point and never calls them independent oracles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packages/openecon-charts/src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_identity():
    files = [*sorted((ROOT / "src/openecon").rglob("*.py")),
             *sorted((ROOT / "packages/openecon-charts/src").rglob("*.py")),
             *sorted((ROOT / "benchmarks").glob("native_model_matrix*.py")),
             ROOT / "benchmarks/native_model_matrix_tests.json"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    return {"git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "aggregate_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "files_sha256": hashes}


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def postest(result, path, metadata):
    """Explicit saved-result helper scope; no universal helper claim."""
    import pandas as pd
    import openecon as oe
    from openecon.models import ResultBundle
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    name = restored.spec.estimator
    # Only nine complete rows are needed for this explicitly bounded helper.
    # Loading the whole file here polluted post-fit RSS for large replay cases.
    pieces, count = [], 0
    iterator = oe.scan(path).iter_batches(batch_rows=8192)
    try:
        for batch in iterator:
            piece = batch.dropna().head(9-count)
            pieces.append(piece)
            count += len(piece)
            if count == 9:
                break
    finally:
        iterator.close()
    evaluation = pd.concat(pieces).copy()
    evaluation.attrs["metadata"] = metadata

    def rows(table):
        return json.loads(table.to_json(orient="records", double_precision=15))

    if name in {"arima", "arch", "var", "vec", "ucm"}:
        options = {"exog": evaluation.head(3)} if name in {"arima", "arch", "var", "ucm"} else {}
        return {"helper": "forecast from saved state", "rows": rows(oe.forecast(restored, 3, **options))}
    if name == "nardl":
        return {"helper": "nardl_multipliers from saved state",
                "rows": rows(oe.nardl_multipliers(restored, steps=3))}
    if name in {"ols", "logit", "probit", "areg", "cnsreg", "reghdfe", "ppmlhdfe",
                "ivregress", "xtivreg", "ivreghdfe", "nl", "rreg", "qreg", "bsqreg", "sqreg", "iqreg"}:
        options = {"kind": "xb"} if name in {"areg", "reghdfe", "ppmlhdfe", "xtivreg", "ivreghdfe"} else {}
        if name == "sqreg":
            options["outcome"] = restored.coefficients[0].equation
        return {"helper": "external-row prediction and full-covariance mean interval",
                "rows": rows(oe.predict(restored, data=evaluation, interval="mean", **options))}
    if name == "teffects":
        evaluated = oe.causal_evaluate(restored, oe.Dataset.from_frame(evaluation),
            target="standardized_outcome", treatment="1", population="fixed_evaluation")
        try:
            return {"helper": "saved causal standardized outcome",
                    "rows": rows(pd.concat(list(evaluated.iter_batches())))}
        finally:
            evaluated.close()
    return {"helper": "persisted physical-row prediction preview and model diagnostics",
            "rows": restored.predictions, "tests": restored.tests,
            "boundary": "No generic external-target or forecast assertion for this case; original method-specific development assertions are separate"}


def worker(args):
    import pandas as pd
    import torch
    import openecon as oe
    from openecon.dataset import Dataset
    from openecon.models import ModelSpec, ResultBundle
    from openecon.econometrics import registry

    if not getattr(sys, "frozen", False):
        assert Path(oe.__file__).resolve().is_relative_to(ROOT / "src"), \
            "Source measurement imported an unrelated installed SDK"

    case = json.loads(args.case.read_text())
    specification = ModelSpec.model_validate(case["spec"])
    path = Path(case["path"])
    before = digest(path)
    if before != case["source_sha256"]:
        raise ValueError("The physical source differs from the captured fixture identity")
    scratch = args.directory / "scratch"
    state = args.directory / "group-state"
    scratch.mkdir(exist_ok=False)
    state.mkdir(exist_ok=False)
    os.environ["OPENECON_SCRATCH_DIRECTORY"] = str(scratch)
    os.environ["OPENECON_GROUP_STATE_DIRECTORY"] = str(state)
    torch.set_num_threads(2)
    torch.manual_seed(8127)
    counters = {"started_passes": 0, "completed_passes": 0,
                "source_rows_yielded": 0, "maximum_source_batch_rows": 0}
    started = time.perf_counter()
    if args.mode == "resident":
        data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        data.attrs["metadata"] = case.get("metadata", {})
        counters.update(started_passes=1, completed_passes=1,
                        source_rows_yielded=len(data), maximum_source_batch_rows=len(data))
        if registry.get(specification.estimator).legacy:
            def fit():
                return oe.fit(specification, data=data)
            entry = "openecon.analysis:fit (legacy resident)"
        else:
            entry = registry.get(specification.estimator).entry
            def fit():
                return registry.load_entry(registry.get(specification.estimator))(specification, data)
    else:
        data = oe.scan(path)
        data._metadata = case.get("metadata", {})
        original = Dataset.iter_batches

        def counted(source, *a, **kw):
            if source is not data:
                yield from original(source, *a, **kw)
                return
            counters["started_passes"] += 1
            for batch in original(source, *a, **kw):
                counters["source_rows_yielded"] += len(batch)
                counters["maximum_source_batch_rows"] = max(
                    counters["maximum_source_batch_rows"], len(batch))
                yield batch
            counters["completed_passes"] += 1
        Dataset.iter_batches = counted
        def fit():
            return oe.fit(specification, data=data)
        entry = "openecon.analysis:fit -> Dataset adapter"
    input_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        result = fit()
        fit_seconds = time.perf_counter() - started
        cold_reads = dict(counters)
        cold_peak_rss = peak_rss_bytes()
        warm_evidence = {"boundary": "No repeat requested"}
        if args.mode == "replay" and not getattr(args, "skip_warm", False):
            torch.manual_seed(8127)
            started = time.perf_counter()
            repeated = fit()
            warm_evidence = {"boundary": "Second fit in same worker and same scanned source; OS cache already warmed",
                "fit_seconds": time.perf_counter()-started,
                "source_reads": {k: counters[k]-cold_reads[k] for k in counters
                                 if k != "maximum_source_batch_rows"},
                "maximum_source_batch_rows": counters["maximum_source_batch_rows"],
                "comparison": compare(result.model_dump(), repeated.model_dump(), relative=1e-7, absolute=1e-9)}
            if warm_evidence["comparison"]["status"] != "passed":
                raise ValueError("Warm replay changed reported inference: " +
                                 repr(warm_evidence["comparison"]["errors"][:3]))
        payload = result.model_dump_json()
        result_path = args.directory / "result.json"
        result_path.write_text(payload + "\n")
        restored = ResultBundle.model_validate_json(result_path.read_text())
        assert restored.model_dump() == result.model_dump()
        latex_path = args.directory / "result.tex"
        latex_path.write_text(restored.to_latex())
        assert "\\begin{tabular}" in latex_path.read_text()
        helper = postest(restored, path, case.get("metadata", {}))
        outputs = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                   for p in (result_path, latex_path)}
        receipt = {"status": "passed", "entry_point": entry,
                   "input_seconds": input_seconds, "fit_seconds": fit_seconds,
                   "nobs": result.nobs, "nobs_original": result.nobs_original,
                   "dropped_rows": result.dropped_rows, "outputs": outputs,
                   "result_roundtrip_exact": True, "latex_readback": True,
                   "streaming": result.provenance.get("streaming"),
                   "inference": result.inference, "model_test_keys": sorted(result.tests)}
        receipt["postest"] = helper
        receipt.update(cold_source_reads=cold_reads, cold_process_peak_rss_bytes=cold_peak_rss,
                       warm=warm_evidence)
    except Exception as error:
        import traceback
        traceback.print_exc()
        receipt = {"status": "failed", "entry_point": entry,
                   "error_type": type(error).__name__, "error": str(error),
                   "error_code": getattr(error, "code", None)}
    receipt.update(mode=args.mode, model=specification.estimator,
        source_sha256_before=before, source_sha256_after=digest(path),
        source_unchanged=before == digest(path), source_bytes=path.stat().st_size,
        actual_source_rows=case["rows"], source_reads=counters,
        process_peak_rss_bytes=peak_rss_bytes(),
        rss_scope="Process high-water mark, including imports and input; not incremental allocation",
        scratch_remaining={str(p.relative_to(scratch)): p.stat().st_size
                           for p in scratch.rglob("*") if p.is_file()},
        persistent_group_state_bytes=sum(p.stat().st_size for p in state.rglob("*") if p.is_file()),
        python=sys.version, torch_version=torch.__version__, frozen=getattr(sys, "frozen", False),
        cache_boundary="Fresh worker; input hashing and fixture generation warm OS pages; no cold-cache claim")
    write(args.directory / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "mode": args.mode,
                      "model": specification.estimator, "error": receipt.get("error")}))
    return 0 if receipt["status"] == "passed" else 1


def compare(expected, observed, *, relative, absolute):
    errors, counts, maximum = [], {"numeric": 0, "identity": 0}, 0.0

    def visit(a, b, path):
        nonlocal maximum
        if isinstance(a, dict):
            if not isinstance(b, dict):
                errors.append(path + ": mapping type differs")
                return
            for k, value in a.items():
                if k == "label" and path.startswith("tests"):
                    continue  # Human title; statistic/df/distribution/p are checked.
                if k not in b:
                    errors.append(path + "/" + k + ": absent")
                else:
                    visit(value, b[k], path + "/" + k)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                errors.append(path + ": list length differs")
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, path + "/" + str(i))
        elif isinstance(a, int) and not isinstance(a, bool):
            # Counts, dimensions and integer policy/option values are identities:
            # 5e-5 at one million must not admit fifty missing observations.
            counts["identity"] += 1
            if not isinstance(b, (int, float)) or isinstance(b, bool) or a != b:
                errors.append(f"{path}: integer identity {a!r} != {b!r}")
        elif isinstance(a, float):
            counts["numeric"] += 1
            if not isinstance(b, (int, float)) or isinstance(b, bool):
                errors.append(path + ": numeric type differs")
            else:
                delta = abs(a - b)
                maximum = max(maximum, delta)
                if not math.isclose(a, b, rel_tol=relative, abs_tol=absolute):
                    errors.append(f"{path}: {a} != {b}")
        else:
            counts["identity"] += 1
            if a != b:
                errors.append(f"{path}: {a!r} != {b!r}")

    for field in (
        "spec",
        "nobs",
        "nobs_original",
        "dropped_rows",
        "coefficients",
        "covariance_matrix",
        "metrics",
        "inference",
        "tests",
    ):
        if field == "metrics":
            for label, value in (("resident", expected), ("replay", observed)):
                converged = value[field].get("converged")
                if converged is not None and converged != 1:
                    errors.append(label + ": explicitly nonconverged fit")
            effort = {"iterations", "n_iterations"}
            visit(
                {k: v for k, v in expected[field].items() if k not in effort},
                observed[field],
                field,
            )
        elif field == "inference":
            # Explanatory prose differs between resident and replay adapters.
            # Preserve it in both result files; compare the policy identifiers,
            # all degrees of freedom/factors and full covariance numerically.
            prose = {"correction", "degrees_of_freedom_convention"}
            visit(
                {k: v for k, v in expected[field].items() if k not in prose}, observed[field], field
            )
        else:
            visit(expected[field], observed[field], field)

    def sample_identity(result, label):
        positions = result.get("sample_positions", [])
        provenance = result.get("provenance", {})
        recorded = provenance.get("sample_positions_hash")
        count = provenance.get("sample_position_count")
        if positions:
            if not isinstance(positions, list) or any(
                type(position) is not int or not 0 <= position < result["nobs_original"]
                for position in positions
            ):
                errors.append(label + ": invalid zero-based physical sample positions")
                return None
            # Membership is canonical physical source order, irrespective of a
            # resident fit's temporary panel/time sort. Replay hashes represent
            # retained physical scan positions as signed little-endian int64.
            ordered = sorted(positions)
            if any(first == second for first, second in zip(ordered, ordered[1:])):
                errors.append(label + ": duplicate physical sample positions")
                return None
            h = hashlib.sha256()
            for position in ordered:
                h.update(struct.pack("<q", position))
            if count is not None and (type(count) is not int or count != len(ordered)):
                errors.append(label + ": sample_position_count disagrees with physical positions")
            return {"count": len(ordered), "sha256": h.hexdigest()}
        if (
            type(count) is not int
            or count <= 0
            or not isinstance(recorded, str)
            or len(recorded) != 64
            or any(c not in "0123456789abcdef" for c in recorded)
        ):
            errors.append(label + ": complete physical sample identity unavailable")
            return None
        return {"count": count, "sha256": recorded}

    first_sample = sample_identity(expected, "resident sample")
    second_sample = sample_identity(observed, "replay sample")
    if first_sample is not None and second_sample is not None:
        visit(first_sample, second_sample, "physical_sample")
    # Chart sampling can be thinner for replay, so compare every shared physical
    # row, and require a nonempty overlap whenever resident has chart rows.
    reference = {row["row"]: row for row in expected["predictions"] if "row" in row}
    if expected["predictions"] and len(reference) != len(expected["predictions"]):
        errors.append("resident predictions: missing or duplicate physical position")
    if any(
        type(position) is not int or not 0 <= position < expected["nobs_original"]
        for position in reference
    ):
        errors.append("resident predictions: invalid physical position")
    overlap = 0
    seen = set()
    for row in observed["predictions"]:
        position = row.get("row")
        if (
            type(position) is not int
            or not 0 <= position < observed["nobs_original"]
            or position in seen
        ):
            errors.append("replay predictions: missing, invalid or duplicate physical position")
            continue
        seen.add(position)
        if position in reference:
            overlap += 1
            visit(reference[row["row"]], row, "predictions/" + str(row["row"]))
    if reference and not overlap:
        errors.append("predictions: no matching physical row")
    for name, encoding in expected["provenance"].get("categorical_encoding", {}).items():
        actual_encoding = observed["provenance"].get("categorical_encoding", {}).get(name)
        if actual_encoding is None:
            errors.append("categorical_encoding/" + name + ": absent")
        else:
            visit(encoding, actual_encoding, "categorical_encoding/" + name)
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "comparisons": counts,
        "prediction_rows_compared": overlap,
        "maximum_absolute_difference": maximum,
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "inference_description_boundary": "correction/convention prose is retained, not asserted text-identical; covariance, policy codes, df and correction factors are compared",
        "physical_sample": {
            "resident": first_sample,
            "replay": second_sample,
            "canonicalization": "Complete resident zero-based physical positions sorted into source order; SHA256 over signed little-endian int64. Replay uses persisted retained physical scan-position hash and exact count. Undocumented differently ordered hashes cannot be normalized from counts alone.",
            "scope": "Exact retained physical membership/count; frequency-weight nobs remains separately exact. Temporary fit order prose is not asserted identical.",
        },
        "convergence": {
            "resident": expected["metrics"].get("converged"),
            "replay": observed["metrics"].get("converged"),
            "scope": "Explicit nonconvergence fails. Missing flags do not establish convergence; iteration counts may differ.",
        },
        "fields": [
            "spec",
            "sample counts",
            "every coefficient/inference field",
            "full covariance matrix",
            "all resident fit metrics (optimizer effort recorded separately)",
            "inference policy",
            "all resident diagnostics",
            "complete physical sample identity",
            "shared physical prediction rows",
        ],
        "extra_scope": "Model-specific extras/forecasts checked by original development tests; not blanket extra identity",
    }


def run(args):
    manifest = json.loads(args.manifest.read_text())
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    identity = code_identity()
    chosen, counts = [], {}
    for case in sorted(manifest["cases"], key=lambda c: (c["model"], -c["rows"], c["id"])):
        model = case["model"]
        if args.models and model not in args.models:
            continue
        if counts.get(model, 0) < args.cases_per_model:
            chosen.append(case)
            counts[model] = counts.get(model, 0) + 1
    if not chosen:
        raise ValueError("A nonempty selected physical case inventory is required")
    missing = set(args.models or []) - set(counts)
    if missing:
        raise ValueError(
            "Requested models have no selected physical cases: " + repr(sorted(missing))
        )
    names = [case["id"] for case in chosen]
    if len(names) != len(set(names)) or any(
        not name or name in {".", ".."} or Path(name).name != name for name in names
    ):
        raise ValueError("Physical cases require unique safe identifiers")
    rows = []
    for case in chosen:
        base = directory / case["id"]
        if base.exists():
            raise SystemExit(f"Refusing to reuse an existing case output directory: {base}")
        base.mkdir()
        write(base / "case.json", case)
        evidence = {"case": case, "modes": {}}
        for mode in ("resident", "replay"):
            target = base / mode
            target.mkdir()
            command = [
                sys.executable,
                str(Path(__file__)),
                "worker",
                "--case",
                str(base / "case.json"),
                "--directory",
                str(target),
                "--mode",
                mode,
            ]
            if args.skip_warm:
                command.append("--skip-warm")
            scratch_peak = 0
            started = time.perf_counter()
            timed_out = False
            with (target / "worker.log").open("w") as output:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    start_new_session=True,
                )
                while process.poll() is None:
                    try:
                        scratch_peak = max(
                            scratch_peak,
                            sum(
                                p.stat().st_size
                                for p in (target / "scratch").rglob("*")
                                if p.is_file()
                            ),
                        )
                    except FileNotFoundError:
                        pass
                    if time.perf_counter() - started > args.timeout:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
                    time.sleep(0.05)
            receipt_path = target / "receipt.json"
            receipt = (
                json.loads(receipt_path.read_text())
                if receipt_path.exists()
                else {
                    "status": "timeout"
                    if time.perf_counter() - started > args.timeout
                    else "worker_failed",
                    "exit_code": process.returncode,
                    "log": (target / "worker.log").read_text()[-2000:],
                }
            )
            if timed_out or process.returncode != 0:
                receipt["worker_reported_status"] = receipt.get("status")
                receipt["status"] = "timeout" if timed_out else "worker_failed"
            receipt["exit_code"] = process.returncode
            receipt["controller_timed_out"] = timed_out
            receipt.update(
                sampled_scratch_peak_bytes=scratch_peak,
                monitor_interval_seconds=0.05,
                process_wall_seconds=time.perf_counter() - started,
            )
            if receipt["status"] != "passed":
                scratch = target / "scratch"
                abandoned = sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())
                shutil.rmtree(scratch, ignore_errors=False) if scratch.exists() else None
                receipt["failure_cleanup"] = {
                    "owned_scratch_bytes_removed": abandoned,
                    "scratch_exists_after_cleanup": scratch.exists(),
                }
            evidence["modes"][mode] = receipt
        if all(m["status"] == "passed" for m in evidence["modes"].values()):
            expected = json.loads((base / "resident/result.json").read_text())
            actual = json.loads((base / "replay/result.json").read_text())
            # State-space observed information uses finite differences; retain
            # its documented wider tolerance rather than quietly broadening all.
            relative = 1e-3 if case["model"] in {"ucm", "mswitch", "arch", "arima"} else 5e-5
            evidence["comparison"] = compare(expected, actual, relative=relative, absolute=2e-6)
            one, two = (evidence["modes"][m]["postest"] for m in ("resident", "replay"))
            if one["helper"] != two["helper"]:
                evidence["comparison"]["errors"].append("saved postest helper differs")
                evidence["comparison"]["status"] = "failed"
            elif one["helper"].startswith("persisted physical"):
                evidence["comparison"]["postest_scope"] = one["helper"]
            else:
                # Reuse the same numeric/identity traversal for helper rows by
                # placing them into a result field whose complete tree is checked.
                helper_expected = {**expected, "tests": one}
                helper_observed = {**expected, "tests": two}
                checked = compare(
                    helper_expected, helper_observed, relative=relative, absolute=2e-6
                )
                evidence["comparison"]["postest_scope"] = one["helper"]
                evidence["comparison"]["postest_comparisons"] = checked["comparisons"]
                evidence["comparison"]["errors"].extend(checked["errors"])
                if checked["status"] != "passed":
                    evidence["comparison"]["status"] = "failed"
        else:
            evidence["comparison"] = {"status": "not_run"}
        evidence["status"] = (
            "passed"
            if evidence["comparison"]["status"] == "passed"
            and all(
                m.get("source_unchanged") and not m.get("scratch_remaining")
                for m in evidence["modes"].values()
            )
            else "failed"
        )
        rows.append(evidence)
        after = code_identity()
        if after["aggregate_sha256"] != identity["aggregate_sha256"]:
            evidence["status"] = "failed"
            evidence["code_changed_during_run"] = True
        write(base / "evidence.json", evidence)
        summary = {
            "schema": 1,
            "stage": "physical_source_fresh_process",
            "scope": "Deterministic development fixtures; bounded resident reference, not Stata parity or customer data",
            "code_before": identity,
            "code_after": after,
            "capture_manifest_sha256": digest(args.manifest),
            "cases": rows,
            "expected_base_models": sorted(manifest["models"]),
            "models_measured": sorted({r["case"]["model"] for r in rows}),
            "passed": sum(r["status"] == "passed" for r in rows),
            "failed": sum(r["status"] != "passed" for r in rows),
        }
        write(directory / "matrix.json", summary)
        print(
            json.dumps(
                {
                    "model": case["model"],
                    "status": evidence["status"],
                    "completed": len(rows),
                    "selected": len(chosen),
                }
            ),
            flush=True,
        )
    return 0 if all(r["status"] == "passed" for r in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    controller = sub.add_parser("run")
    controller.add_argument("--manifest", type=Path, required=True)
    controller.add_argument("--directory", type=Path, required=True)
    controller.add_argument("--models", nargs="*")
    controller.add_argument("--cases-per-model", type=int, default=1)
    controller.add_argument("--timeout", type=float, default=180)
    controller.add_argument("--skip-warm", action="store_true")
    child = sub.add_parser("worker")
    child.add_argument("--case", type=Path, required=True)
    child.add_argument("--directory", type=Path, required=True)
    child.add_argument("--mode", choices=["resident", "replay"], required=True)
    child.add_argument("--skip-warm", action="store_true")
    args = parser.parse_args()
    return worker(args) if args.command == "worker" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
