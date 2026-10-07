"""Synthetic worker timeline regression example for MARKET-49.

Import this file into a separate OpenEconometrics project and press Run.
The selected initial frame uses ForceAtlas2 in a worker, not fixed positions.
Switch snapshots rapidly, then wait at least four seconds after Ready. The
completed layout must remain Ready without an old timeout overwriting it.
The script writes no files and operates only on generated synthetic data.
"""

import openecon as oe

NODE_COUNT = 600
SEED = 49
nodes = [f"firm-{i:04d}" for i in range(NODE_COUNT)]
layers = {}
for month in range(4):
    edges = [
        (nodes[i], nodes[(i + offset + month) % NODE_COUNT])
        for i in range(NODE_COUNT)
        for offset in (1, 5, 19, 43)
    ]
    layers[f"2026-{month + 1:02d}"] = oe.network(
        {"source": [edge[0] for edge in edges], "target": [edge[1] for edge in edges]},
        nodes=nodes, directed=True, max_memory_mb=64,
    )

snapshots = oe.network_snapshots(layers, ordered=True)
timeline = oe.plot.network(
    snapshots, title="Worker timeline · synthetic firms · MARKET-49",
    timeline=True, frame_index=1, layout="forceatlas2", seed=SEED,
    layout_options={"iterations": 30, "work_limit": 2_000_000, "time_limit_ms": 1000},
    max_nodes=NODE_COUNT, max_edges=3000, width=1000, height=600,
    palette=["#14263d", "#4c78a8"],
    labels={"show": True, "min_zoom": 1, "max_count": 30},
)
print(f"Synthetic CPU graph: {NODE_COUNT} nodes, 2,400 directed edges per frame, four frames, seed={SEED}.")
display_callback = globals().get("display")
if display_callback is not None:
    display_callback(timeline)
