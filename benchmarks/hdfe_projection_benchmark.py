"""Source-pinned physical FE benchmark; run_case also runs in a frozen worker.

Timings are first/warm fits on a shared host, not cold OS-cache measurements.
The baseline and optimized workers must consume the same immutable manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import cProfile
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import subprocess
import sys
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_case(case, directory, *, profile=False):
    """Callable packaged-runtime entry point; no editable SDK import overrides."""
    import openecon as oe
    import torch
    from openecon.models import ModelSpec

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = Path(case["path"])
    if sha(path) != case["sha256"]:
        raise ValueError("Physical fixture hash differs from the frozen manifest")
    torch.set_num_threads(2)
    scratch = directory / "scratch"
    scratch.mkdir()
    os.environ["OPENECON_SCRATCH_DIRECTORY"] = str(scratch)
    stop, disk = threading.Event(), [0]

    def observe():
        while not stop.wait(0.02):
            try:
                disk[0] = max(
                    disk[0], sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())
                )
            except FileNotFoundError:
                pass

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    record = {
        "case": case,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "host": platform.platform(),
        "threads": 2,
        "load_at_start": os.getloadavg(),
        "sdk_path": oe.__file__,
        "scope": "First/warm physical fits; shared host and retained OS caches; parent process high-water RSS includes imports, both fits and optional profiler; scratch sampled20ms is a lower bound",
    }
    try:
        measurements = []
        for i in range(2):
            start = time.perf_counter()
            result = oe.fit(ModelSpec.model_validate(case["spec"]), data=oe.scan(path))
            elapsed = time.perf_counter() - start
            value = result.model_dump(mode="json")
            with gzip.open(directory / f"result-{i}.json.gz", "wt") as stream:
                json.dump(value, stream, allow_nan=False)
            measurements.append(
                {
                    "fit_seconds": elapsed,
                    "passes": result.provenance["streaming"]["passes"],
                    "nobs": result.nobs,
                    "nobs_original": result.nobs_original,
                    "result_sha256": sha(directory / f"result-{i}.json.gz"),
                }
            )
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
            oe.fit(ModelSpec.model_validate(case["spec"]), data=oe.scan(path))
            profiler.disable()
            profiler.dump_stats(str(directory / "profile.pstats"))
            with (directory / "profile.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
        record.update(status="passed", measurements=measurements)
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        stop.set()
        observer.join()
        record.update(
            peak_process_rss_bytes=int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024)
            ),
            sampled_scratch_peak_bytes=disk[0],
            scratch_clean=not any(scratch.iterdir()),
            source_unchanged=sha(path) == case["sha256"],
        )
        save(directory / "receipt.json", record)
    return record


def generate(directory):
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [
        str(root / "src"),
        str(root / "packages/openecon-charts/src"),
        str(root / "benchmarks"),
    ]
    from native_model_matrix_fixtures import fixture

    directory.mkdir(parents=True, exist_ok=False)
    cases = []
    for rows in (1201, 12001):
        for model in ("ppmlhdfe", "reghdfe"):
            frame, spec = fixture(model, rows)
            for suffix in ("csv", "parquet"):
                name = f"{model}-{rows}-{suffix}"
                path = directory / f"{name}.{suffix}"
                if suffix == "csv":
                    frame.to_csv(path, index=False)
                else:
                    frame.to_parquet(path, index=False, row_group_size=8192, compression="zstd")
                cases.append(
                    {
                        "id": name,
                        "path": str(path),
                        "rows": len(frame),
                        "sha256": sha(path),
                        "spec": spec.model_dump(mode="json"),
                        "fixture": "native_model_matrix_fixtures.fixture; original MARKET-107 seed/geometry",
                    }
                )
    save(directory / "manifest.json", {"schema": 1, "cases": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("directory", type=Path)
    for action in ("run", "worker"):
        p = sub.add_parser(action)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--directory", type=Path, required=True)
        p.add_argument("--source-root", type=Path, required=True)
        p.add_argument("--source-pin", required=True)
        p.add_argument("--profile", action="store_true")
        if action == "worker":
            p.add_argument("--case", required=True)
        else:
            p.add_argument("--jobs", type=int, default=2)
            p.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.directory.resolve())
        return
    cases = json.loads(args.manifest.read_text())["cases"]
    if args.action == "worker":
        sys.path[:0] = [
            str(args.source_root / "src"),
            str(args.source_root / "packages/openecon-charts/src"),
        ]
        case = next(c for c in cases if c["id"] == args.case)
        value = run_case(case, args.directory, profile=args.profile)
        value["source_pin"] = args.source_pin
        value["feature_sha256"] = {
            name: sha(args.source_root / name)
            for name in (
                "src/openecon/econometrics/streaming_hdfe.py",
                "src/openecon/econometrics/streaming_ppml.py",
            )
        }
        save(args.directory / "receipt.json", value)
        return 0 if value["status"] == "passed" else 1
    args.directory.mkdir(parents=True, exist_ok=False)

    def worker(case):
        directory = args.directory / case["id"]
        command = [
            args.python,
            str(Path(__file__).resolve()),
            "worker",
            "--manifest",
            str(args.manifest.resolve()),
            "--directory",
            str(directory.resolve()),
            "--source-root",
            str(args.source_root.resolve()),
            "--source-pin",
            args.source_pin,
            "--case",
            case["id"],
        ]
        if args.profile and case["id"] == "ppmlhdfe-1201-parquet":
            command.append("--profile")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=1800,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        (args.directory / f"{case['id']}.log").write_text(completed.stdout + completed.stderr)
        print(json.dumps({"case": case["id"], "exit_code": completed.returncode}), flush=True)
        return {"case": case["id"], "exit_code": completed.returncode}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        outcomes = list(pool.map(worker, cases))
    save(args.directory / "workers.json", outcomes)
    return int(any(x["exit_code"] for x in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
