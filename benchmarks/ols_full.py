"""Reproducible bounded-memory weighted OLS benchmark; no data files required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import pandas as pd
import torch

import openecon as oe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 10 or args.batch_rows < 1 or args.threads < 1:
        parser.error("Use at least ten observations and positive batch/thread counts.")
    torch.set_num_threads(args.threads)

    def batches():
        generator = torch.Generator().manual_seed(29173)
        for start in range(0, args.rows, args.batch_rows):
            size = min(args.batch_rows, args.rows-start)
            values = torch.randn((size, 4), dtype=torch.float64, generator=generator)
            y = 2 + .7*values[:, 0] - .3*values[:, 1] + .2*values[:, 2] + values[:, 3]
            yield pd.DataFrame({"y": y.numpy(), "x1": values[:, 0].numpy(),
                                "x2": values[:, 1].numpy(), "x3": values[:, 2].numpy(),
                                "w": (1+values[:, 0].abs()).numpy()})

    source = oe.Dataset.from_batches(batches, ["y", "x1", "x2", "x3", "w"],
                                    row_count=args.rows)
    started = time.perf_counter()
    result = oe.ols(data=source, y="y", x=["x1", "x2", "x3"],
                    weights="w", weight_type="aweight", covariance="HC3")
    elapsed = time.perf_counter()-started
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if platform.system() == "Darwin" else 1024
    except ImportError:
        rss = None
    assert result.nobs == args.rows and not result.sample_positions
    report = {
        "openecon_version": oe.__version__, "platform": platform.platform(),
        "processor": platform.machine(), "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "device": result.provenance["device"],
        "rows": args.rows, "batch_rows": args.batch_rows, "seed": 29173,
        "predictors": 3, "weight_type": "aweight", "covariance": "HC3",
        "elapsed_seconds": elapsed, "peak_process_rss_bytes": rss,
        "process_rss_includes_python_torch_and_pandas": True,
        "timing_includes_data_generation_and_model_passes": True,
        "timing_excludes_imports_disk_reads_and_ui": True,
        "streaming": result.provenance["streaming"],
        "coefficients": [coefficient.model_dump() for coefficient in result.coefficients],
        "known_population_coefficients": [2., .7, -.3, .2],
        "prediction_sample_rows": len(result.predictions),
        "all_sample_positions_retained": False, "cuda_hardware_validated": False,
        "hundred_billion_rows_hardware_validated": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"rows": args.rows, "seconds": elapsed,
                      "peak_process_rss_mib": rss/1024**2 if rss is not None else None,
                      "passes": report["streaming"]["passes"],
                      "output": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()
