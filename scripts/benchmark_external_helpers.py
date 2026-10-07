"""Physical million-row helper probes; each calculation runs in a fresh process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def probe(path, method, budget):
    import openecon as oe
    import torch
    from openecon.resources import use_workspace_budget
    torch.set_num_threads(2)
    source = oe.scan(path)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.monotonic()
    with use_workspace_budget(budget):
        if method in {"spearman", "kendall"}:
            result = oe.correlate(source, ["x", "z"], method=method)
        elif method == "describe":
            result = oe.describe(source, ["x", "y"], by="g", stats=["n", "mean", "p2.5", "p50", "p97.5", "sd"])
        elif method == "rm_anova":
            result = oe.rm_anova(source, "y", "subject", ["cell"], between=["g"])
        else:
            result = oe.manova(source, ["y", "z"], ["g"], covariates=["x"])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024**2 if sys.platform == "darwin" else 1024
    attrs = {key: value for key, value in result.attrs.items() if key in {
        "n", "n_complete", "n_subjects", "source_passes", "source_content_sha256", "batch_rows",
        "full_source_collected", "subject_matrix_collected", "resource_plan", "rank_resource_plan",
        "descriptive_resource_plan", "repeated_resource_plan", "tsqr_resource_plan", "manova_resource_plan"}}
    assert attrs["full_source_collected"] is False
    assert source.row_count == 1_000_000
    if method == "describe":
        assert result.n.sum() == 2_000_000
    elif method in {"spearman", "kendall"}:
        assert result["n"].iloc[0, 1] == 1_000_000
    else:
        assert attrs["n"] == 1_000_000
    return {"method": method, "seconds": round(time.monotonic()-start, 3),
            "workspace_budget_mib": budget, "peak_rss_mib": round(peak/scale, 2),
            "import_baseline_peak_rss_mib": round(baseline/scale, 2),
            "peak_increment_mib": round((peak-baseline)/scale, 2), "attrs": attrs}


def main(output):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="openecon-helper-million-") as directory:
        root = Path(directory)
        path = root / "million.parquet"
        generator = torch.Generator().manual_seed(100807)
        writer = None
        try:
            for start in range(0, 1_000_000, 10000):
                row = torch.arange(start, start+10000)
                x = (row % 127).to(torch.float64) / 11
                group = (row // 5) % 4
                frame = pd.DataFrame({"x": x.tolist(), "z": ((row*17) % 131).to(torch.float64).tolist(),
                    "y": (2+.3*x+.2*(row%5)+.5*group+torch.randn(len(row),generator=generator,dtype=torch.float64)).tolist(),
                    "g": group.tolist(), "subject": (row//5).tolist(), "cell": (row%5).tolist()})
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        scratch = root / "scratch"
        scratch.mkdir()
        environment = dict(os.environ, OPENECON_SCRATCH_DIRECTORY=str(scratch))
        records = []
        for method in ["spearman", "kendall", "describe", "rm_anova", "manova"]:
            completed = subprocess.run([sys.executable, __file__, "--probe", str(path), "--method", method,
                                       "--budget", "128"], env=environment, text=True, capture_output=True, check=True)
            record = json.loads(completed.stdout)
            assert list(scratch.iterdir()) == []
            record["scratch_removed"] = True
            records.append(record)
            print(json.dumps(record), flush=True)
        root_repo = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root_repo, text=True).strip()
        kernels = sorted((root_repo / "src/openecon/econometrics/stats").glob("*.py"))
        digest = hashlib.sha256()
        for file in kernels:
            digest.update(file.name.encode())
            digest.update(file.read_bytes())
        report = {"scope": "source kernels, CPU float64; RSS includes imports and Arrow reader; no frozen/UI/cloud claim",
                  "source_base_revision": revision, "stats_sources_sha256": digest.hexdigest(),
                  "physical_parquet": True, "rows": 1_000_000, "bytes": path.stat().st_size,
                  "fixture_seed": 100807, "results": records}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe(args.probe, args.method, args.budget)))
    else:
        main(args.output)
