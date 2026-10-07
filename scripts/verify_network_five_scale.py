"""Fresh-process synthetic scale measurements, distinct from correctness oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = ("hypergraph", "graphlets", "paths", "strong_cuts", "assignment", "general_matching", "dsp")


def measure(case):
    import openecon as oe
    import torch

    torch.set_num_threads(1)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()

    def graph(rows, **options):
        return oe.network(
            [dict(source=u, target=v, w=w) for u, v, w in rows], weight="w", **options
        )

    if case == "hypergraph":
        n = 8000
        h = oe.hypergraph([dict(id=e, members=[e * 4 + j for j in range(4)]) for e in range(n)])
        incidence, degrees, projected = h.incidence(), h.degree(), h.clique_projection()
        assert incidence._nnz() == 4 * n and projected.edge_count == 6 * n
        assert degrees.hyperdegree.eq(1).all()
        record = dict(
            nodes=4 * n,
            hyperedges=n,
            memberships=4 * n,
            incidence_sparse=True,
            projection_edges=projected.edge_count,
            work_used=projected.metadata["projection_work_used"],
            max_work=50_000_000,
        )
    elif case == "graphlets":
        n = 24
        g = graph(
            [(u, v, 1) for u in range(n) for v in range(n) if u != v and (u * 13 + v * 7) % 11 < 3],
            directed=True,
        )
        result = g.graphlets(4, connected=False)
        assert result["classes"]["count"].sum() == 10626
        record = dict(
            nodes=n,
            edges=g.edge_count,
            **{
                key: result["metadata"][key]
                for key in ("subsets_enumerated", "output_rows", "work_used", "max_work", "exact")
            },
        )
    elif case == "paths":
        layers = [list(range(1 + 3 * i, 4 + 3 * i)) for i in range(8)]
        rows = [(0, v, 1) for v in layers[0]] + [(u, 25, 1) for u in layers[-1]]
        rows += [(u, v, 1) for left, right in zip(layers, layers[1:]) for u in left for v in right]
        g = graph(rows, directed=True)
        result = g.k_shortest_paths(0, 25, k=50, max_path_length=9, max_output_nodes=500)
        assert result.attrs["returned"] == 50 and result.cost.eq(9).all()
        record = dict(
            nodes=g.node_count,
            edges=g.edge_count,
            output_rows=len(result),
            available_layered_paths=3**8,
            **{
                key: result.attrs[key]
                for key in (
                    "returned",
                    "work_used",
                    "max_work",
                    "max_frontier",
                    "exact",
                )
            },
        )
    elif case == "strong_cuts":
        n = 150
        g = graph(
            [(u, (u + delta) % n, 1) for u in range(n) for delta in (1, 5, 17)], directed=True
        )
        edges, nodes = g.strong_bridges(), g.strong_articulation_points()
        record = dict(
            nodes=n,
            edges=g.edge_count,
            strong_bridges=len(edges),
            strong_articulation_points=int(nodes.strong_articulation.sum()),
            work_used=edges.attrs["work_used"] + nodes.attrs["work_used"],
            max_work_per_call=50_000_000,
            exact=True,
        )
    elif case == "assignment":
        nleft, nright = 80, 120
        rows = [
            (u, nleft + (u * 11 + j * 13) % nright, 1 + (u * 7 + j * 3) % 100)
            for u in range(nleft)
            for j in range(8)
        ]
        g = graph(rows, nodes=range(nleft + nright))
        result = g.weighted_assignment(
            partition={u: int(u >= nleft) for u in range(nleft + nright)}
        )
        assert result["metadata"]["certificate"]["zero_duality_gap"]
        record = dict(
            nodes=g.node_count,
            edges=g.edge_count,
            pairs=len(result["pairs"]),
            weight=result["weight"],
            **{
                key: result["metadata"][key]
                for key in ("work_used", "max_work", "matrix_entries", "planned_work", "certified")
            },
        )
    elif case == "general_matching":
        n = 18
        g = graph([(u, v, 1 + (u * 7 + v * 13) % 100) for u in range(n) for v in range(u + 1, n)])
        result = g.general_matching(objective="cardinality_weight")
        assert len(result["pairs"]) == 9
        record = dict(
            nodes=n,
            edges=g.edge_count,
            pairs=9,
            weight=result["weight"],
            **{
                key: result["metadata"][key]
                for key in ("states", "work_used", "max_work", "certified")
            },
        )
    else:
        rng, n = random.Random(42), 80
        graphs = []
        for _ in range(3):
            pairs = rng.sample([(u, v) for u in range(n) for v in range(u + 1, n)], 180)
            graphs.append(graph([(u, v, rng.randint(1, 100)) for u, v in pairs], nodes=range(n)))
        result = graphs[0].qap_regression(
            dict(a=graphs[1], b=graphs[2]),
            method="dsp",
            joint={"both": ["a", "b"]},
            permutations=49,
            seed=42,
            adjustment="holm",
            max_work=150_000_000,
        )
        assert result.attrs["pvalue_resolution"] == 0.02
        record = dict(
            nodes=n,
            resident_edges=sum(g.edge_count for g in graphs),
            **{
                key: result.attrs[key]
                for key in (
                    "work_used",
                    "max_work",
                    "permutations",
                    "pvalue_resolution",
                    "test_method",
                )
            },
        )
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1024
    return dict(
        record,
        case=case,
        elapsed_seconds=elapsed,
        peak_rss_bytes=peak * multiplier,
        import_baseline_peak_rss_bytes=baseline * multiplier,
        incremental_peak_rss_bytes=max(0, peak - baseline) * multiplier,
        device="cpu",
        torch_version=torch.__version__,
        sdk_version=oe.__version__,
        geometry_limitations="one deterministic synthetic fixture; no general complexity/performance guarantee",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(measure(args.case), allow_nan=False))
        return
    if not args.output or args.output.exists():
        parser.error("Choose a new --output receipt path")
    records = []
    for case in CASES:
        run = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", case],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        record = json.loads(run.stdout)
        records.append(record)
        print(
            json.dumps({key: record[key] for key in ("case", "elapsed_seconds", "peak_rss_bytes")}),
            flush=True,
        )
    names = [
        "_network_hypergraph",
        "_network_graphlets",
        "_network_bounded_paths",
        "_network_matching",
        "_network_mrqap",
    ]
    report = dict(
        status="passed",
        measurements=records,
        system=platform.platform(),
        source_sha256={
            name: hashlib.sha256((ROOT / "src/openecon" / f"{name}.py").read_bytes()).hexdigest()
            for name in names
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rss_scope="fresh subprocess absolute high-water RSS; includes Python/Torch/imports and caller fixture; distinct from planned owned budgets",
        source_runtime_only=True,
        correctness_oracle=False,
        cuda_verified=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
