"""Fresh-process network pipeline measurements, with physical input and plot readback.

python benchmarks/network_pipeline.py --output /tmp/owned-network-run --repeats 3
Add --large for matched full/selected 100k-node, 1m-edge display runs.
No timing thresholds, universal capacity claims, or external graph solvers.
"""
from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/openecon-charts/src")]
CASES = ("sparse", "directed", "disconnected", "hub", "dense", "temporal")
SEED = 716050


def peak_rss_bytes():
    """OS lifetime high-water mark: bytes on Darwin, KiB on Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def fingerprint():
    paths = [ROOT / "src/openecon/networks.py", ROOT / "src/openecon/plot_artifacts.py",
             ROOT / "uv.lock", Path(__file__)]
    paths += sorted((ROOT / "src/openecon").glob("_network*.py"))
    paths += sorted((ROOT / "packages/openecon-charts/src/openecon_charts").glob("*.py"))
    paths += sorted((ROOT / "packages/openecon-charts/src/openecon_charts/assets").glob("network*.js"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def hardware(torch):
    cpu = platform.processor()
    memory = None
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    elif hasattr(os, "sysconf"):
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return dict(platform=platform.platform(), cpu=cpu, logical_cpus=os.cpu_count(),
                physical_memory_bytes=memory, python=platform.python_version(),
                torch=torch.__version__, torch_threads=torch.get_num_threads(),
                seed=SEED, analytical_dtype="float64", analytical_device="cpu")


def fixture(case, nodes, frame=0):
    """Deterministic positive-weight topologies; disconnected includes 20% isolates."""
    import numpy as np
    import pandas as pd

    if case not in CASES or nodes < 10:
        raise ValueError("Use a known topology and at least ten nodes.")
    if case == "dense":
        source, target = np.triu_indices(nodes, k=1)
    elif case == "hub":
        leaf = np.arange(1, nodes, dtype=np.int64)
        source = np.concatenate([np.zeros(nodes - 1, dtype=np.int64), np.tile(leaf, 3)])
        target = np.concatenate([leaf, *[1 + (leaf - 1 + offset) % (nodes - 1) for offset in (1, 2, 3)]])
    elif case == "disconnected":
        active = nodes * 4 // 5
        block = max(10, active // 8)
        # Each contiguous block is disconnected from all others; trailing nodes are isolates.
        pairs = [(u, start + (u - start + offset) % width)
                 for start in range(0, active, block)
                 for width in [min(block, active - start)] if width > 4
                 for u in range(start, start + width) for offset in (1, 2, 3, 4)]
        source, target = np.array(pairs, dtype=np.int64).T
    else:
        degree = 10 if nodes >= 100_000 else 4
        source = np.tile(np.arange(nodes, dtype=np.int64), degree)
        target = np.concatenate([(np.arange(nodes) + offset + frame * degree) % nodes
                                 for offset in range(1, degree + 1)])
    generator = np.random.default_rng(SEED + frame)
    return pd.DataFrame({"source": source, "target": target,
                         "weight": generator.uniform(0.5, 1.5, len(source))})


def json_digest(value):
    digest, size = hashlib.sha256(), 0
    for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(value):
        encoded = chunk.encode("utf-8")
        size += len(encoded)
        digest.update(encoded)
    return dict(bytes=size, sha256=digest.hexdigest())


def fixed_coordinates(payload):
    """A deterministic grid isolates rendering from force-layout convergence."""
    width = math.ceil(math.sqrt(payload["node_count"]))
    for graph in [payload, *[frame["network"] for frame in payload.get("frames", [])]]:
        for node in graph["nodes"]:
            node["x"], node["y"] = node["id"] % width, node["id"] // width
    return payload


def measure_case(case, nodes, mode, directory, *, profile=False):
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    import openecon as oe
    from openecon.plot_artifacts import checked_plot_path, store_plot
    from openecon_charts import PlotSpec

    torch.set_num_threads(2)
    before = fingerprint()
    metadata = hardware(torch)
    phases = []

    def phase(name, function):
        start = time.perf_counter()
        value = function()
        phases.append(dict(name=name, seconds=time.perf_counter() - start,
                           cumulative_process_peak_rss_bytes=peak_rss_bytes()))
        return value

    baseline = peak_rss_bytes()
    profiler = cProfile.Profile() if profile else None
    if profiler:
        profiler.enable()
    frames = 4 if case == "temporal" else 1
    inputs = phase("synthetic_generation", lambda: [fixture(case, nodes, frame) for frame in range(frames)])
    paths = [directory / f"frame-{frame}.parquet" for frame in range(frames)]
    phase("physical_input_write", lambda inputs=inputs: [data.to_parquet(path, index=False) for data, path in zip(inputs, paths)])
    del inputs
    physical = [dict(bytes=path.stat().st_size, rows=pq.ParquetFile(path).metadata.num_rows,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    data = phase("physical_input_read", lambda: [pd.read_parquet(path) for path in paths])
    directed = case in ("directed", "temporal")
    graphs = phase("graph_construction", lambda data=data: [oe.network(value, weight="weight", directed=directed,
                    nodes=range(nodes), max_memory_mb=2048) for value in data])
    del data
    degrees = phase("degree_all_frames", lambda: [graph.degree() for graph in graphs])
    ranks = phase("pagerank_all_frames", lambda: [graph.pagerank(tol=1e-10, max_iter=200) for graph in graphs])
    components = phase("weak_components_all_frames", lambda: [graph.components() for graph in graphs])
    # Typed node identity, not row position, joins analytical outputs.
    tables = phase("result_table_join", lambda: [degree.merge(rank, on="node", validate="one_to_one")
                    .merge(component, on="node", validate="one_to_one")
                    for degree, rank, component in zip(degrees, ranks, components)])
    records = phase("result_table_records", lambda: [table.to_dict(orient="records") for table in tables])
    result_json = phase("result_table_json", lambda: json_digest(records))
    snapshot = phase("timeline_collection", lambda: oe.network_snapshots(
        {f"T{index}": graph for index, graph in enumerate(graphs)}, ordered=True)) if frames > 1 else graphs[0]
    transitions = phase("timeline_transitions", snapshot.transitions) if frames > 1 else None
    max_nodes, max_edges = (nodes, max(graph.edge_count for graph in graphs)) if mode == "full" else (2000, 10000)
    payload = phase("display_selection", lambda: snapshot.to_plot_data(max_nodes=max_nodes, max_edges=max_edges, seed=SEED))
    phase("fixed_coordinates", lambda: fixed_coordinates(payload))
    spec = phase("plot_validation", lambda: PlotSpec("network", f"{case} · {mode}", "", "", [],
        payload["shown_node_count"], payload["node_count"], 0,
        {"network": payload, "options": {"layout": "fixed", "labels": False, "height": 620}}))
    plot_public_json = phase("plot_public_json", lambda: json_digest(spec.model_dump()))
    plot = phase("plot_normalization", spec.transport_dump)
    plot_json = phase("plot_json", lambda: json_digest(plot))
    envelope = phase("production_artifact_store", lambda: store_plot(directory, plot))
    stored = phase("artifact_integrity_readback", lambda: checked_plot_path(directory, envelope["artifact"]))
    if profiler:
        profiler.disable()
        profiler.dump_stats(directory / "profile.pstats")
    if fingerprint() != before:
        raise RuntimeError("Measured sources changed during the run.")
    if plot_json != {"bytes": envelope["artifact"]["bytes"], "sha256": envelope["artifact"]["id"]}:
        raise RuntimeError("Artifact bytes differ from the measured serialization.")
    counts = [{key: frame.get(key) for key in ("node_count", "edge_count", "shown_node_count",
              "shown_edge_count", "sampled", "selection")} for frame in
              [payload, *[item["network"] for item in payload.get("frames", [])]]]
    result = dict(schema=1, case=case, mode=mode, metadata=metadata, directed=directed,
        weighted=True, frame_count=frames, input=physical, graphs=[graph.metadata for graph in graphs],
        algorithms=dict(scope="Complete graphs in every frame; no analytical sampling.",
            pagerank=[dict(rank.attrs) for rank in ranks],
            rank_sums=[float(rank["pagerank"].sum()) for rank in ranks],
            component_counts=[int(component["component"].nunique()) for component in components]),
        result_rows=[len(table) for table in tables], result_table_json=result_json,
        transition_rows=len(transitions) if transitions is not None else None,
        display=counts, plot_json=plot_json, plot_public_json=plot_public_json,
        transport_encoding=plot["config"]["network"].get("encoding", "expanded"),
        artifact_path=str(stored.relative_to(directory)),
        phases=phases, source_sha256=before, baseline_process_peak_rss_bytes=baseline,
        process_peak_rss_bytes=peak_rss_bytes(),
        estimated_retained_graph_buffers_bytes=sum(graph.metadata["estimated_owned_graph_bytes"] for graph in graphs),
        memory_scope="Fresh child lifetime OS peak RSS includes imports, generation, input, algorithms, Python tables, validation, JSON and storage. Phase marks are cumulative high-water marks, not isolated phase peaks. Graph estimates exclude caller input, Python/export objects and whole-process RSS; the 2 GiB graph budget is not a process memory guarantee.",
        timing_scope="Warm local file cache; generation/write are separate. Artifact storage repeats production validation and streaming JSON, then fsync/rename; it is not disk-only time. plot_json is a separate serialization probe, not an extra production pipeline step. Fixed coordinates isolate drawing from force convergence.",
        profiled=profile)
    if profiler:
        stats = pstats.Stats(profiler)
        result["profile"] = [dict(function=f"{Path(key[0]).name}:{key[1]}:{key[2]}", calls=value[1],
             self_seconds=value[2], cumulative_seconds=value[3]) for key, value in
             sorted(stats.stats.items(), key=lambda pair: pair[1][3], reverse=True)[:25]]
    return result


def summarize(runs):
    groups = {}
    for run in runs:
        key = f"{run['case']}/{run['mode']}/{run['graphs'][0]['node_count']}"
        groups.setdefault(key, []).append(run)
    return {key: dict(repeats=len(values), phases={phase["name"]: dict(
        median_seconds=statistics.median(samples), min_seconds=min(samples), max_seconds=max(samples))
        for phase in values[0]["phases"]
        for samples in [[next(item["seconds"] for item in value["phases"] if item["name"] == phase["name"])
                         for value in values]]},
        peak_rss_bytes=[value["process_peak_rss_bytes"] for value in values])
        for key, values in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New owned directory; existing paths are refused")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Separate profiled control, excluded from summaries")
    parser.add_argument("--child", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--nodes", type=int, default=10000, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("full", "selected"), default="full", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be between 1 and 20")
    if args.child and not 10 <= args.nodes <= (1000 if args.child == "dense" else 100000):
        parser.error("nodes must be 10..1000 for dense, or 10..100000 for other fixtures")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.child:
        report = measure_case(args.child, args.nodes, args.mode, args.output, profile=args.profile)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    start, before = time.perf_counter(), fingerprint()
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"),
        str(ROOT / "packages/openecon-charts/src")]))
    configurations = [(case, 600 if case == "dense" else 2500 if case == "temporal" else 10000, "full") for case in CASES]
    if args.large:
        configurations += [("sparse", 100000, mode) for mode in ("full", "selected")]
    runs, browser_cases = [], []
    for case, nodes, mode in configurations:
        for repeat in range(args.repeats):
            name = f"{case}-{nodes}-{mode}-{repeat + 1}"
            target = args.output / name
            subprocess.run([sys.executable, str(Path(__file__)), "--child", case, "--nodes", str(nodes),
                "--mode", mode, "--output", str(target)], env=environment, check=True)
            report = json.loads((target / "report.json").read_text())
            runs.append(report)
            print(f"{name}: peak RSS {report['process_peak_rss_bytes'] / 2**20:.1f} MiB", flush=True)
            if repeat == 0:
                browser_cases.append(dict(id=name, path=f"{name}/{report['artifact_path']}",
                    sha256=report["plot_json"]["sha256"], bytes=report["plot_json"]["bytes"],
                    metadata=report["metadata"], graphs=report["graphs"], display=report["display"]))
    profile_report = None
    if args.profile:
        target = args.output / "profile-control"
        subprocess.run([sys.executable, str(Path(__file__)), "--child", "sparse", "--nodes", "10000",
                        "--profile", "--output", str(target)], env=environment, check=True)
        profile_report = json.loads((target / "report.json").read_text())
    if fingerprint() != before:
        raise RuntimeError("Measured sources changed across repetitions.")
    manifest = dict(schema=1, measured_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=before, runs=runs, summary=summarize(runs),
                    profiled_control=profile_report, browser_cases=browser_cases,
                    suite_wall_seconds=time.perf_counter() - start)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
