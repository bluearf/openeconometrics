"""Reproducible physical fixtures for sparse flow, matching and link workflows.

Default sizes are examples, not maximum supported sizes or runtime promises.
All API operations use complete resident sparse snapshots. Explicit candidates
are read from Parquet in batches; no implicit all-pair link list is generated.
Fixture creation, imports, independent readback checks and cleanup have separate
timing scope. Two Torch CPU threads are selected before fixture construction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import openecon as oe


SOURCE_PATHS = (
    "src/openecon/networks.py", "src/openecon/_network_sparse.py",
    "src/openecon/_network_flow.py", "src/openecon/_network_bipartite.py",
    "benchmarks/network_workflow_scale.py",
)


def fingerprint(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def cpu_name():
    if sys.platform == "darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor()


def physical(path):
    metadata = pq.ParquetFile(path).metadata
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(sha256=digest, bytes=path.stat().st_size, physical_rows=metadata.num_rows,
        row_groups=metadata.num_row_groups,
        largest_row_group=max(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)))


def write_records(path, records, *, weighted=True, batch_rows=65_536):
    """Bounded row batches; input records need not be sized or materialized."""
    writer = None
    sources, targets, weights = [], [], []

    def flush():
        nonlocal writer
        columns = {"source": pa.array(sources, type=pa.int64()), "target": pa.array(targets, type=pa.int64())}
        if weighted:
            columns["weight"] = pa.array(weights, type=pa.float64())
        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table, row_group_size=batch_rows)
        sources.clear()
        targets.clear()
        weights.clear()

    try:
        for u, v, w in records:
            sources.append(u)
            targets.append(v)
            if weighted:
                weights.append(w)
            if len(sources) == batch_rows:
                flush()
        if sources:
            flush()
    finally:
        if writer is not None:
            writer.close()
    return physical(path)


def flow_records(width, layers, seed):
    source, target = width * layers, width * layers + 1
    shift = 1 + seed % (width - 1)
    yield from ((source, i, 1.) for i in range(width))
    for layer in range(layers - 1):
        for i in range(width):
            yield layer * width + i, (layer + 1) * width + i, 1.
            yield layer * width + i, (layer + 1) * width + (i + shift) % width, 1.
    yield from (((layers - 1) * width + i, target, 1.) for i in range(width))


def matching_records(n):
    # Canonical sorted CSR gives the first n-1 left nodes their diagonal
    # neighbors. Last-left only has right0, forcing an n-long augmenting path
    # in phase two. Avoids merely timing a trivial greedy perfect matching.
    yield from ((i, n + i, 1.) for i in range(n - 1))
    yield from ((i, n + i + 1, 1.) for i in range(n - 1))
    yield n - 1, n, 1.


def bipartite_records(left_nodes, right_nodes, degree, seed):
    rng = random.Random(seed)
    for center in range(right_nodes):
        base = center % left_nodes
        neighbors = [base, *(x + (x >= base) for x in rng.sample(range(left_nodes - 1), degree - 1))]
        yield from ((left, left_nodes + center, 1.) for left in neighbors)


def candidate_records(nodes, rows, seed):
    rng = random.Random(seed)
    for _ in range(rows):
        u, v = rng.randrange(nodes), rng.randrange(nodes - 1)
        yield u, v + (v >= u), 1.


def graph_summary(graph):
    return dict(node_count=graph.node_count, edge_count=graph.edge_count,
        metadata=graph.metadata, lifetime_planned_owned_peak_bytes=graph._budget.peak,
        owned_memory_limit_bytes=graph._budget.limit)


def tensor_digest(graph):
    digest = hashlib.sha256()
    for tensor in (graph._edges.indices(), graph._edges.values()):
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()


def measure(phases, name, options, callback):
    start = time.perf_counter()
    try:
        result = callback()
    except Exception as error:
        phases[name] = dict(status="failed", seconds=time.perf_counter() - start, options=options,
            error_type=type(error).__name__, error_code=getattr(error, "code", None),
            error_message=str(error), peak_process_rss_bytes_after=peak_rss())
        print(json.dumps({"phase": name, "status": "failed", "error_code": getattr(error, "code", None)}), flush=True)
        raise
    phases[name] = dict(status="passed", seconds=time.perf_counter() - start,
        options=options, peak_process_rss_bytes_after=peak_rss())
    print(json.dumps({"phase": name, "status": "passed", "seconds": phases[name]["seconds"]}), flush=True)
    return result


def _import(path, nodes, args, *, directed=False, weighted=True):
    return oe.network(oe.scan(path), nodes=range(nodes), directed=directed,
        weight="weight" if weighted else None, batch_rows=4096, max_memory_mb=args.max_memory_mb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=716032)
    parser.add_argument("--flow-width", type=int, default=500)
    parser.add_argument("--flow-layers", type=int, default=40)
    parser.add_argument("--matching-size", type=int, default=20_000)
    parser.add_argument("--left-nodes", type=int, default=10_000)
    parser.add_argument("--right-nodes", type=int, default=20_000)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=100_000)
    parser.add_argument("--max-memory-mb", type=float, default=1024)
    parser.add_argument("--max-work", type=int, default=50_000_000)
    args = parser.parse_args()
    if (not 0 <= args.seed < 2**63 - 1 or args.flow_width < 2 or args.flow_layers < 2
            or args.matching_size < 2 or args.left_nodes < 2 or args.right_nodes < args.left_nodes
            or not 2 <= args.degree <= args.left_nodes or args.candidate_count < 1
            or args.max_work < 1 or not math.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0):
        parser.error("Use positive valid dimensions; right-nodes>=left-nodes,2<=degree<=left-nodes,finite memory.")
    if args.output.exists():
        parser.error("The output exists; use a fresh owned receipt path.")
    root = Path(__file__).resolve().parents[1]
    before = fingerprint(root)
    torch.set_num_threads(2)
    fixtures, phases, graphs, creation = {}, {}, {}, {}
    failure = None
    with tempfile.TemporaryDirectory(prefix="openecon-network-workflow-") as owned:
        flow_path, matching_path, bipartite_path, candidates_path = (
            Path(owned) / name for name in ("flow.parquet", "matching.parquet", "bipartite.parquet", "candidates.parquet"))
        try:
            builders = (
                ("layered_flow", flow_path, flow_records(args.flow_width, args.flow_layers, args.seed), True),
                ("alternating_matching", matching_path, matching_records(args.matching_size), False),
                ("bipartite", bipartite_path, bipartite_records(args.left_nodes, args.right_nodes, args.degree, args.seed + 1), True),
                ("explicit_candidates", candidates_path, candidate_records(args.left_nodes, args.candidate_count, args.seed + 2), False),
            )
            for name, path, records, weighted in builders:
                start = time.perf_counter()
                fixtures[name] = write_records(path, records, weighted=weighted)
                creation[name] = time.perf_counter() - start
            fixtures["layered_flow"].update(seed=args.seed, width=args.flow_width, layers=args.flow_layers,
                topology="unit-capacity layered two-out-neighbor grid; seed controls ring shift",
                expected_max_flow=args.flow_width)
            fixtures["alternating_matching"].update(left_nodes=args.matching_size, right_nodes=args.matching_size,
                topology="n-1 diagonal and shift links, final-left/right0 link; canonical ordering forces2phases",
                expected_cardinality=args.matching_size)
            fixtures["bipartite"].update(seed=args.seed + 1, left_nodes=args.left_nodes, right_nodes=args.right_nodes,
                right_degree=args.degree, topology="each rightnode has one cyclic anchor and distinct uniformly sampled other leftnodes",
                capacities="unit positive weights", expected_projection_weight_sum=args.right_nodes * args.degree * (args.degree - 1) // 2)
            fixtures["explicit_candidates"].update(seed=args.seed + 2, node_count=args.left_nodes,
                topology="uniform explicit ordered distinct-node pairs; duplicates retained; no filtering or sampling at scoring")

            flow_nodes = args.flow_width * args.flow_layers + 2
            graph = measure(phases, "flow_import", dict(nodes=flow_nodes, directed=True, weight="weight", batch_rows=4096),
                lambda: _import(flow_path, flow_nodes, args, directed=True))
            unchanged = tensor_digest(graph)
            result = measure(phases, "max_flow", dict(source=flow_nodes - 2, target=flow_nodes - 1, max_work=args.max_work),
                lambda: graph.max_flow(flow_nodes - 2, flow_nodes - 1, max_work=args.max_work))
            assert result["value"] == args.flow_width and result["metadata"]["certified"]
            assert result["metadata"]["cut_capacity"] == args.flow_width
            assert result["flows"].flow.between(0., result["flows"].capacity).all()
            assert tensor_digest(graph) == unchanged
            phases["max_flow"]["result"] = dict(value=result["value"], flow_rows=len(result["flows"]),
                cut_edges=len(result["cut_edges"]), source_partition_nodes=len(result["source_partition"]),
                target_partition_nodes=len(result["target_partition"]), metadata=result["metadata"],
                original_graph_unchanged=True)
            graphs["flow"] = graph_summary(graph)
            del result, graph
            gc.collect()

            n = args.matching_size
            graph = measure(phases, "matching_import", dict(nodes=2*n, directed=False, weight=None, batch_rows=4096),
                lambda: _import(matching_path, 2*n, args, weighted=False))
            unchanged = tensor_digest(graph)
            partition = {i: int(i >= n) for i in range(2*n)}
            result = measure(phases, "maximum_matching", dict(partition="explicitfullnode-to-side", max_work=args.max_work),
                lambda: graph.maximum_matching(partition, max_work=args.max_work))
            assert len(result) == n and result.attrs["certified"] and result.attrs["phases"] == 2
            assert result.source.nunique() == result.target.nunique() == n
            assert len(result.attrs["minimum_vertex_cover"]) == n and not result.attrs["unmatched_nodes"]
            assert tensor_digest(graph) == unchanged
            phases["maximum_matching"]["result"] = dict(matched_pairs=len(result),
                minimum_cover_nodes=len(result.attrs["minimum_vertex_cover"]), unmatched_nodes=0,
                phases=result.attrs["phases"], work_used=result.attrs["work_used"],
                algorithm=result.attrs["algorithm"], certified=True, original_graph_unchanged=True)
            graphs["matching"] = graph_summary(graph)
            del result, graph, partition
            gc.collect()

            total = args.left_nodes + args.right_nodes
            graph = measure(phases, "bipartite_import", dict(nodes=total, directed=False, weight="weight", batch_rows=4096),
                lambda: _import(bipartite_path, total, args))
            unchanged = tensor_digest(graph)
            partition = measure(phases, "bipartite_partition", dict(partition="inferred"), graph.bipartite)
            assert len(partition) == total
            assert partition.partition.iloc[:args.left_nodes].eq(0).all() and partition.partition.iloc[args.left_nodes:].eq(1).all()
            phases["bipartite_partition"]["result"] = dict(rows=len(partition), metadata=partition.attrs)
            projected = measure(phases, "bipartite_projection", dict(onto=0, weight="count", max_edges=1_000_000, max_work=args.max_work),
                lambda: graph.bipartite_projection(partition, onto=0, weight="count", max_work=args.max_work))
            assert projected.node_count == args.left_nodes
            weight_sum = float(projected._edges.values().sum())
            assert weight_sum == fixtures["bipartite"]["expected_projection_weight_sum"]
            assert tensor_digest(graph) == unchanged
            phases["bipartite_projection"]["result"] = dict(node_count=projected.node_count,
                edge_count=projected.edge_count, projected_weight_sum=weight_sum,
                independent_expected_wedge_sum=fixtures["bipartite"]["expected_projection_weight_sum"],
                metadata=projected.metadata, original_graph_unchanged=True)
            graphs["bipartite"] = graph_summary(graph)
            graphs["projection"] = graph_summary(projected)
            del graph, partition
            gc.collect()

            candidates = oe.scan(candidates_path)
            unchanged = tensor_digest(projected)
            for method in ("common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"):
                result = measure(phases, "links_" + method, dict(method=method, candidates=args.candidate_count,
                    max_pairs=args.candidate_count, max_work=args.max_work, batch_rows=4096),
                    lambda method=method: projected.link_prediction(candidates, method=method,
                        max_pairs=args.candidate_count, max_work=args.max_work, batch_rows=4096))
                assert len(result) == args.candidate_count and not result.attrs["sampled"]
                values = torch.as_tensor(result.score.to_numpy(), dtype=torch.float64)
                assert bool(torch.isfinite(values).all() & (values >= 0).all())
                if method == "jaccard":
                    assert bool((values <= 1).all())
                offset = 0
                iterator = candidates.iter_batches(columns=["source", "target"], batch_rows=4096)
                try:
                    for expected in iterator:
                        end = offset + len(expected)
                        assert result.source.iloc[offset:end].tolist() == expected.source.tolist()
                        assert result.target.iloc[offset:end].tolist() == expected.target.tolist()
                        offset = end
                finally:
                    iterator.close()
                assert offset == args.candidate_count
                phases["links_" + method]["result"] = dict(rows=len(result), minimum=float(values.min()),
                    maximum=float(values.max()), mean=float(values.mean()), metadata=result.attrs,
                    explicit_candidate_order_readback=True, original_graph_unchanged=tensor_digest(projected) == unchanged)
                assert phases["links_" + method]["result"]["original_graph_unchanged"]
                del result, values
                gc.collect()
            graphs["projection_final"] = graph_summary(projected)
            del projected, candidates
            gc.collect()
            for name, path, _, _ in builders:
                assert physical(path) == {key: fixtures[name][key] for key in physical(path)}
        except Exception as error:
            failure = dict(type=type(error).__name__, code=getattr(error, "code", None), message=str(error))
    after = fingerprint(root)
    stable = before == after
    receipt = dict(status="passed" if failure is None and stable else "failed",
        measured_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256_before=before,
        source_sha256_after=after, source_bytes_unchanged=stable,
        hardware=dict(system=platform.system(), release=platform.release(), machine=platform.machine(), cpu=cpu_name()),
        versions=dict(python=platform.python_version(), torch=torch.__version__, pandas=pd.__version__,
            pyarrow=pa.__version__, openecon=oe.__version__), torch_threads=torch.get_num_threads(),
        device="cpu", precision="float64", seed=args.seed,
        options=dict(max_memory_mb=args.max_memory_mb, max_work=args.max_work),
        fixtures=fixtures, fixture_creation_seconds_excluded=creation,
        graphs=graphs, phases=phases, peak_process_rss_bytes=peak_rss(),
        temporary_sources_removed=True, error=failure,
        timing_scope="Each complete public API call; fixture creation, independent checks and cleanup excluded. Imports timed separately.",
        memory_scope="RSS is whole-process lifetime peak including Python/Torch, fixtures, imports, checks and all phases. Owned memory guard excludes caller input/RSS.",
        scope=f"Resident sparse CPU graphs; one layered unit-capacity flow, one adversarial bipartite matching, one random bipartite count projection and {args.candidate_count:,} explicit ordered candidates scored with five methods. No all-pair link generation, arbitrary-precision, GPU, general matching, min-cost/multicommodity flow or universal performance claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=receipt["status"], source_bytes_unchanged=stable,
        peak_process_rss_bytes=receipt["peak_process_rss_bytes"], error=failure)), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
