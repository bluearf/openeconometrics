"""Fresh CPU processes: coupled sparse rings, independent uniform equation, RSS."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def worker(n):
    import torch
    import openecon as oe
    torch.set_num_threads(1)
    def records():
        for layer in ["trade", "finance"]:
            for i in range(n):
                for offset in [1, 3, 11]:
                    yield dict(edge_id=f"{layer}/{i}/{offset}", source=i, source_layer=layer,
                               target=(i+offset) % n, target_layer=layer, weight=1.)
        for i in range(n):
            yield dict(edge_id=f"coupling/{i}", source=i, source_layer="trade", target=i,
                       target_layer="finance", weight=.5)
    start = time.perf_counter()
    g = oe.multilayer_network(records(), layers=["trade", "finance"], weight="weight", max_memory_mb=768)
    imported = time.perf_counter() - start
    start = time.perf_counter()
    result = g.matvec(dict.fromkeys(g._pairs, 1.))
    matvec = time.perf_counter() - start
    assert result.value.tolist() == [6.5] * (2*n)
    start = time.perf_counter()
    rank = g.pagerank(max_work=2_000_000_000)
    elapsed = time.perf_counter() - start
    error = max(abs(v - 1/(2*n)) for v in rank.pagerank)
    assert error < 1e-12 and g.node_count == 2*n and g.edge_count == 7*n
    return dict(physical_nodes=n, node_layer_states=g.node_count, layers=g.layer_count, separate_edges=g.edge_count,
        intra_layer_edges=6*n, inter_layer_edges=n, directed=False, coupling_weight=.5,
        topology="two undirected circulants, offsets 1/3/11 and explicit diagonal couplings",
        threads=torch.get_num_threads(), device="cpu", torch=torch.__version__, python=platform.python_version(),
        import_seconds=imported, matvec_seconds=matvec, pagerank_seconds=elapsed,
        pagerank_iterations=rank.attrs["iterations"], dense_adjacency=False, complete_result_rows=len(rank),
        oracle="every adjacency row sums to 6.5; regular undirected supra-graph has uniform PageRank",
        oracle_max_error=error, planned_owned_graph_bytes=g.metadata["estimated_owned_graph_bytes"],
        planned_peak_owned_bytes=g._graph._budget.peak, max_memory_bytes=g._graph._budget.limit,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope="macOS whole fresh process including imports/caller/output; separate from planned owned memory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker), allow_nan=False))
        return
    runs = []
    for n in [5000, 20000]:
        completed = subprocess.run([sys.executable, __file__, "--worker", str(n)], check=True, capture_output=True, text=True)
        runs.append(json.loads(completed.stdout))
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(), runs=runs,
                   scope="one regular workload; no out-of-core, GPU, all-topology or unlimited-scale claim")
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
