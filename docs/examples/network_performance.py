"""Small installed-app companion to the reproducible network pipeline benchmark.

Import into a separate project and Run. Synthetic physical Parquet files live
only in an automatically removed temporary directory. This script reports CPU
phases and fixed-layout charts; browser first drawing is measured by the separate
headed-browser harness, not by the Python plot-construction clock.
"""
import json
import math
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
import openecon as oe
from openecon_charts import PlotSpec

SEED = 716050
torch.manual_seed(SEED)
torch.set_num_threads(2)
metadata = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "threads": torch.get_num_threads(),
            "seed": SEED, "dtype": "float64", "device": "cpu"}
print(json.dumps(metadata))
print("Complete CPU algorithms; fixed-coordinate displays. RSS is the cumulative process peak, including imports.")
rows, charts = [], []

with tempfile.TemporaryDirectory(prefix="openecon-network-performance-") as temporary:
    for case in ("sparse", "directed", "disconnected", "hub", "dense", "temporal"):
        n = 100 if case == "dense" else 250 if case == "temporal" else 1000
        frame_count = 4 if case == "temporal" else 1
        directed = case in ("directed", "temporal")
        times = {}

        def clock(name, operation):
            start = time.perf_counter()
            value = operation()
            times[name] = time.perf_counter() - start
            return value

        layers = {}
        owned = 0
        for frame_index in range(frame_count):
            if case == "dense":
                pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            elif case == "hub":
                pairs = [(0, v) for v in range(1, n)] + [(v, 1 + v % (n - 1)) for v in range(1, n)]
            elif case == "disconnected":
                pairs = [(u, (u // 100) * 100 + (u + 1) % 100) for u in range(800)]
            else:
                pairs = [(u, (u + offset + 4 * frame_index) % n) for u in range(n) for offset in (1, 2, 3, 4)]
            data = pd.DataFrame({"source": [u for u, _ in pairs], "target": [v for _, v in pairs],
                                 "weight": [1 + index % 7 / 10 for index in range(len(pairs))]})
            path = Path(temporary) / f"{case}-{frame_index}.parquet"
            clock(f"T{frame_index}.write", lambda: data.to_parquet(path, index=False))
            loaded = clock(f"T{frame_index}.read", lambda: pd.read_parquet(path))
            graph = clock(f"T{frame_index}.graph", lambda: oe.network(loaded, nodes=range(n),
                          weight="weight", directed=directed, max_memory_mb=64))
            degree = clock(f"T{frame_index}.degree", graph.degree)
            rank = clock(f"T{frame_index}.pagerank", graph.pagerank)
            component = clock(f"T{frame_index}.components", graph.components)
            clock(f"T{frame_index}.table", lambda: degree.merge(rank, on="node", validate="one_to_one")
                  .merge(component, on="node", validate="one_to_one"))
            layers[f"T{frame_index}"] = graph
            owned += graph.metadata["estimated_owned_graph_bytes"]
        source = oe.network_snapshots(layers, ordered=True) if frame_count > 1 else graph
        payload = clock("display_selection", lambda: source.to_plot_data(max_nodes=n, max_edges=10000, seed=SEED))
        width = math.ceil(math.sqrt(n))
        for value in [payload, *[item["network"] for item in payload.get("frames", [])]]:
            for node in value["nodes"]:
                node["x"], node["y"] = node["id"] % width, node["id"] // width
        chart = clock("plot_validation", lambda: PlotSpec("network", f"Performance · {case}", "", "", [],
                      payload["shown_node_count"], payload["node_count"], 0,
                      {"network": payload, "options": {"layout": "fixed", "height": 460,
                       "timeline": frame_count > 1, "labels": False}}))
        serialized = clock("plot_json", chart.model_dump_json)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
        rows.append({"Case": case, "Nodes/frame": n, "Edges/last frame": graph.edge_count,
                     "Frames": frame_count, "Directed": directed, "Shown nodes": payload["shown_node_count"],
                     "Shown edges": payload["shown_edge_count"], "Sampled": payload["sampled"],
                     "Graph buffers bytes": owned, "Process peak RSS bytes": peak_bytes,
                     "Plot JSON bytes": len(serialized.encode("utf-8"))})
        print(json.dumps({"case": case, "metadata": metadata, "seconds": times}))
        charts.append(chart)

print("Graph buffer estimates exclude caller input, Python/export objects and total RSS; a memory budget is not a process guarantee.")
display_callback = globals().get("display")
if display_callback is not None:
    display_callback(oe.DataFrame(pd.DataFrame(rows)))
    for chart in charts:
        display_callback(chart)
else:
    print(pd.DataFrame(rows).to_string(index=False))
