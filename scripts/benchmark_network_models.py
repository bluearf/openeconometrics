"""Fresh-process native model scale receipts; modeled buffers and measured RSS differ."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(job):
    import pandas as pd
    import openecon as oe

    rng = random.Random(331)
    if job == "ergm":
        n = 6
        graph = oe.network(
            [
                {"source": i, "target": j}
                for i in range(n)
                for j in range(i + 1, n)
                if rng.random() < 0.45
            ],
            nodes=range(n),
        )
        result = oe.ergm(graph, terms=["edges", "twostars", "triangles"], max_work=500_000_000)
    elif job == "saom":
        graph = oe.network([], nodes=range(3), directed=True)
        times = [0.0, 0.5, 1.5]
        panels = [
            oe.simulate_saom(graph, times, rate=0.9, theta=[-0.5], seed=s)["panel"]
            for s in range(20)
        ]
        result = oe.saom(panels, times, max_iter=10, max_work=2_000_000_000)
    elif job in {"gaussian", "mixed", "embedding"}:
        n = 300 if job == "gaussian" else 1000
        pairs = set()
        while len(pairs) < 2000 if job == "gaussian" else len(pairs) < 10000:
            a, b = sorted(rng.sample(range(n), 2))
            pairs.add((a, b))
        rows = []
        for a, b in sorted(pairs):
            same = (a < n // 2) == (b < n // 2)
            y = (
                rng.gauss(-2 if same else 2, 0.5)
                if job == "gaussian"
                else int(rng.random() < (0.9 if same else 0.1))
            )
            rows.append((a, b, y, "test" if rng.random() < 0.15 else "train"))
        df = pd.DataFrame(rows, columns=["source", "target", "value", "split"])
        graph = oe.network([], nodes=range(n), max_memory_mb=256)
        if job == "gaussian":
            result = oe.gaussian_block_model(
                graph, 2, df, starts=1, max_iter=3, max_work=2_000_000_000
            )
        elif job == "mixed":
            result = oe.mixed_membership_block_model(
                graph, 2, df, starts=1, max_iter=10, batch_size=256, max_work=2_000_000_000
            )
        else:
            result = oe.network_embedding(
                graph, df, dimensions=8, max_iter=10, batch_size=256, max_work=2_000_000_000
            )
    else:
        n = 1000
        graph = oe.network([], nodes=range(n))
        f = pd.DataFrame(
            {
                "node": range(n),
                "x": [i % 2 * 2 - 1 for i in range(n)],
                "z": [rng.random() for i in range(n)],
            }
        )
        labels = pd.DataFrame(
            {
                "node": range(n),
                "label": [i % 2 for i in range(n)],
                "split": ["train" if i < 700 else "test" for i in range(n)],
            }
        )
        edges = pd.DataFrame(
            [
                (i, j, "train" if i < 700 and j < 700 else "test")
                for i in range(n)
                for j in [(i + 2) % n, (i + 4) % n]
            ],
            columns=["source", "target", "split"],
        )
        result = oe.network_gnn(
            graph,
            f,
            labels,
            edges,
            hidden=8,
            fanout=2,
            batch_size=64,
            max_iter=10,
            max_work=2_000_000_000,
        )
    metadata = result["metadata"]
    return {
        "status": "passed",
        "job": job,
        "metadata": metadata,
        "evaluation": result.get("evaluation", pd.DataFrame()).to_dict("records"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["ergm", "saom", "gaussian", "mixed", "embedding", "gnn"])
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.job:
        started = time.monotonic()
        result = run(a.job)
        result.update(
            seconds=time.monotonic() - started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rss_units="bytes on macOS",
            rss_scope="whole fresh Python process including imports; not incremental model bytes",
        )
        print(json.dumps(result, allow_nan=False))
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "packages/openecon-charts/src")
    results = []
    for job in ["ergm", "saom", "gaussian", "mixed", "embedding", "gnn"]:
        response = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--job", job],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        results.append(json.loads(response.stdout))
    receipt = {
        "status": "passed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "proof": "fresh source processes; not installed/frozen/GPU/public release evidence",
        "models": results,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "seconds": [r["seconds"] for r in results],
                "output": str(a.output),
            }
        )
    )


if __name__ == "__main__":
    main()
