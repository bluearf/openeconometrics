"""Reproduce physical-file OLS, Poisson and histogram workloads in fresh processes.

Examples (from the repository, with the local development environment):
    python benchmarks/native_replay_probe.py write
    python benchmarks/native_replay_probe.py ols
    python benchmarks/native_replay_probe.py poisson
    python benchmarks/native_replay_probe.py hist

The default writer produces 10 million rows in bounded float64 Torch blocks.
Use --rows 1024 --directory artifacts/verification/my-small-probe for a smoke
check. Every command uses the same --rows and owned directory. Existing source
files are never replaced; repeat measurements receive separate JSON receipts.
Fit timings exclude explicit file/source-code hashing and result serialization.
No generated file, cloud resource or user project is installed or published.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "artifacts/verification/native-replay-probe"
SOURCE_NAME = "physical.parquet"
MARKER_NAME = ".native-replay-probe.json"
OWNER = "openecon-native-replay-probe-v1"
PREDICTORS = [f"x{index}" for index in range(4)]
TRUTH = {"ols": [1., .2, .5, -.1, .3], "poisson": [.2, .1, -.15, .08, .12]}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def code_snapshot() -> dict:
    """Pin working-tree Python bytes, without assigning them a Git commit."""
    paths = [Path(__file__).resolve()]
    for directory in (ROOT / "src/openecon", ROOT / "packages/openecon-charts/src"):
        paths.extend(directory.rglob("*.py"))
    files = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"scope": "this probe and all repository OpenEcon/chart production Python files",
            "file_count": len(files), "sha256": hashlib.sha256(encoded).hexdigest(),
            "script_sha256": files[Path(__file__).resolve().relative_to(ROOT).as_posix()]}


def peak_rss() -> dict:
    try:
        import resource
    except ImportError:
        return {"peak_process_rss_bytes": None, "native_unit": None,
                "scope": "Process peak RSS is unavailable on this platform."}
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        multiplier, unit = 1, "bytes"
    elif sys.platform.startswith("linux"):
        multiplier, unit = 1024, "KiB"
    else:
        return {"peak_process_rss_bytes": None, "native_value": raw,
                "native_unit": "unverified platform convention",
                "scope": "No conversion is claimed for this platform."}
    return {"peak_process_rss_bytes": int(raw * multiplier), "native_value": raw,
            "native_unit": unit,
            "scope": "Process high-water mark including imports, file hashing and operation; "
                     "not incremental fit memory or a workspace budget."}


def _exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _owned_directory(directory: Path, *, create: bool) -> None:
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / MARKER_NAME).exists():
            if any(directory.iterdir()):
                raise ValueError("The writer requires an empty directory or this probe's ownership marker.")
            _exclusive_json(directory / MARKER_NAME, {"owner": OWNER, "schema_version": 1})
    marker = json.loads((directory / MARKER_NAME).read_text())
    if marker != {"owner": OWNER, "schema_version": 1}:
        raise ValueError("The directory does not have this probe's valid ownership marker.")


def _positive(value: str) -> int:
    number = int(value)
    if number < 1 or number > 2**53 - 1:
        raise argparse.ArgumentTypeError("Choose an integer from 1 through 2**53 - 1.")
    return number


def _unchanged(before: dict, after: dict) -> None:
    if before != after:
        raise RuntimeError("Repository Python source changed during the operation; no passed receipt is produced.")


def write(args) -> dict:
    _owned_directory(args.directory, create=True)
    path, receipt = args.directory / SOURCE_NAME, args.directory / "source.json"
    if path.exists() or receipt.exists():
        raise FileExistsError("Source or source receipt already exists; choose another owned directory.")
    before = code_snapshot()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    torch.set_num_threads(args.threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.perf_counter()
    # Exclusive creation prevents replacing even a file created after the check.
    with path.open("xb") as handle, torch.no_grad(), torch.device("cpu"):
        writer = None
        try:
            for first in range(0, args.rows, args.block_rows):
                rows = min(args.block_rows, args.rows - first)
                x = torch.randn((rows, 4), generator=generator, dtype=torch.float64)
                y = (1. + x @ torch.tensor(TRUTH["ols"][1:], dtype=torch.float64)
                     + torch.randn(rows, generator=generator, dtype=torch.float64))
                count = torch.poisson(
                    (.2 + x @ torch.tensor(TRUTH["poisson"][1:], dtype=torch.float64)).exp(),
                    generator=generator,
                )
                # NumPy views bridge Torch to Arrow; no statistical estimator is imported.
                table = pa.table({**{name: x[:, j].numpy() for j, name in enumerate(PREDICTORS)},
                                  "y": y.numpy(), "count": count.numpy()})
                if writer is None:
                    writer = pq.ParquetWriter(handle, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=args.block_rows)
        finally:
            if writer is not None:
                writer.close()
    writer_seconds = time.perf_counter() - started
    hashing_started = time.perf_counter()
    source_sha256 = digest(path)
    after = code_snapshot()
    _unchanged(before, after)
    metadata = pq.read_metadata(path)
    if metadata.num_rows != args.rows or metadata.num_columns != 6:
        raise RuntimeError("The written Parquet metadata does not match the requested source.")
    record = {"schema_version": 1, "status": "passed", "rows": args.rows, "columns": 6,
              "row_group_size": args.block_rows, "generator": "bounded CPU float64 PyTorch blocks",
              "seed": args.seed, "source_name": SOURCE_NAME, "source_bytes": path.stat().st_size,
              "source_sha256": source_sha256, "writer_seconds": writer_seconds,
              "hashing_seconds": time.perf_counter() - hashing_started,
              "code_before": before, "code_after": after, "code_unchanged": True,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "pyarrow": pa.__version__,
                              "torch_threads": torch.get_num_threads()}, "rss": peak_rss()}
    _exclusive_json(receipt, record)
    return {"receipt": receipt.name, "rows": args.rows, "source_sha256": source_sha256,
            "writer_seconds": writer_seconds}


def _source_record(args) -> tuple[Path, dict, float]:
    _owned_directory(args.directory, create=False)
    path = args.directory / SOURCE_NAME
    record = json.loads((args.directory / "source.json").read_text())
    if (record.get("status") != "passed" or record.get("source_name") != SOURCE_NAME
            or record.get("rows") != args.rows or record.get("columns") != 6):
        raise ValueError("The source receipt does not match this probe and requested --rows.")
    started = time.perf_counter()
    if path.stat().st_size != record["source_bytes"] or digest(path) != record["source_sha256"]:
        raise RuntimeError("The physical source changed; it will not be silently replaced.")
    return path, record, time.perf_counter() - started


def run(args) -> dict:
    path, source_receipt, initial_hashing = _source_record(args)
    before = code_snapshot()
    start = time.perf_counter()
    import openecon as oe
    import openecon_charts
    import torch

    if Path(oe.__file__).resolve() != ROOT / "src/openecon/__init__.py":
        raise RuntimeError("Use this checkout's editable development environment for source-pinned measurements.")
    if (Path(openecon_charts.__file__).resolve()
            != ROOT / "packages/openecon-charts/src/openecon_charts/__init__.py"):
        raise RuntimeError("The loaded charts package does not belong to the hashed checkout.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    ready = time.perf_counter()
    # scan stays replayable even for a tiny smoke fixture; read may materialize
    # small files, which would test a different execution path.
    source = oe.scan(path)
    if not isinstance(source, oe.Dataset):
        raise RuntimeError("The physical source was not opened as a replayable Dataset.")
    begun = time.perf_counter()
    if args.command == "hist":
        result = oe.plot.hist(data=source, x="y", bins=40)
        counted = sum(row["count"] for row in result.data)
        if (result.total_n != args.rows or counted != args.rows or result.dropped_n != 0
                or len(result.data) != 40):
            raise RuntimeError("The histogram did not account for every physical observation in 40 bins.")
        summary = {"counted_observations": counted, "retained_bins": len(result.data),
                   "processing": result.config.get("processing")}
    else:
        result = getattr(oe, args.command)(data=source,
                    y="count" if args.command == "poisson" else "y", x=PREDICTORS,
                    covariance="robust" if args.command == "poisson" else "HC3")
        expected = dict(zip(["Intercept", *PREDICTORS], TRUTH[args.command], strict=True))
        if {row.term for row in result.coefficients} != set(expected):
            raise RuntimeError("The fit omitted an expected generated regressor.")
        error = max(abs(row.estimate - expected[row.term]) for row in result.coefficients)
        tolerance = max(.004, 8 / math.sqrt(args.rows))
        if (result.nobs != args.rows or result.nobs_original != args.rows
                or not math.isfinite(error) or error >= tolerance
                or any(not math.isfinite(row.std_error) or row.std_error <= 0.
                       for row in result.coefficients)):
            raise RuntimeError("Full-row count or finite coefficient/DGP sanity checks failed.")
        summary = {"nobs": result.nobs, "maximum_coefficient_error": error,
                   "coefficient_sanity_tolerance": tolerance,
                   "sanity_scope": "generated-signal check; not an independent numerical oracle or coverage test",
                   "streaming": result.provenance.get("streaming"),
                   "execution": result.provenance.get("execution")}
    finish = time.perf_counter()
    serialization_start = time.perf_counter()
    full_result = result.model_dump(mode="json")
    serialization_seconds = time.perf_counter() - serialization_start
    hashing_start = time.perf_counter()
    after_source_sha256 = digest(path)
    after = code_snapshot()
    _unchanged(before, after)
    if (after_source_sha256 != source_receipt["source_sha256"]
            or path.stat().st_size != source_receipt["source_bytes"]):
        raise RuntimeError("The physical source changed during the operation; no passed receipt is produced.")
    record = {"schema_version": 1, "status": "passed", "command": args.command,
              "physical_source": source_receipt,
              "source_sha256_before": source_receipt["source_sha256"],
              "source_sha256_after": after_source_sha256, "source_unchanged": True,
              "code_before": before, "code_after": after, "code_unchanged": True,
              "timing_seconds": {"imports": ready - start, "source_metadata": begun - ready,
                                 "run": finish - begun, "cold_total": finish - start,
                                 "result_serialization": serialization_seconds,
                                 "source_hash_before": initial_hashing,
                                 "source_and_code_hash_after": time.perf_counter() - hashing_start},
              "timing_scope": "run is the public API call plus bounded sanity checks; cold_total "
                              "also includes imports and source metadata, but excludes explicit "
                              "source/code hashing and result serialization; OS disk cache is uncontrolled",
              "rss": peak_rss(), "environment": {"platform": platform.platform(),
                  "python": platform.python_version(), "torch": torch.__version__,
                  "torch_threads": torch.get_num_threads(), "torch_interop_threads": 1},
              "summary": summary, "full_result": full_result,
              "limits": "One local CPU-oriented workload, possible concurrent development load; "
                        "no all-family, whole-model GPU, statistical calibration or 100B-row guarantee."}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    receipt = args.directory / f"{args.command}-{stamp}-{uuid.uuid4().hex[:8]}.json"
    _exclusive_json(receipt, record)
    return {"receipt": receipt.name, "rows": args.rows, "timing_seconds": record["timing_seconds"],
            "peak_process_rss_bytes": record["rss"]["peak_process_rss_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("write", "ols", "poisson", "hist"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--rows", type=_positive, default=10_000_000)
    parser.add_argument("--block-rows", type=_positive, default=65536)
    parser.add_argument("--threads", type=_positive, default=2)
    parser.add_argument("--seed", type=int, default=5631297)
    args = parser.parse_args()
    if args.rows < 32 or args.block_rows > 65536 or args.threads > 64 or not 0 <= args.seed < 2**63:
        parser.error("Use --rows >= 32, 1 <= --block-rows <= 65536, 1 <= --threads <= 64, and 0 <= --seed < 2**63.")
    args.directory = args.directory.resolve()
    print(json.dumps(write(args) if args.command == "write" else run(args)), flush=True)


if __name__ == "__main__":
    main()
