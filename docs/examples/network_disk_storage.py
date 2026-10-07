"""Synthetic disk snapshot example for source or installed Mac code panels."""
from pathlib import Path
import sys
import tempfile

import openecon as oe

show = globals().get("display", print)

def batches():
    for start in range(0, 30_000, 1000):
        yield oe.DataFrame({"source": range(start, start + 1000),
                            "target": range(start + 1, start + 1001),
                            "count": [2] * 1000})


with tempfile.TemporaryDirectory(prefix="openecon-disk-example-") as directory:
    destination = Path(directory) / "graph.sqlite"
    data = oe.Dataset.from_batches(batches, columns=["source", "target", "count"], row_count=30_000)
    graph = oe.network(data, weight="count", directed=True, nodes=["isolated"],
        store=destination, max_memory_mb=8, batch_rows=64,
        node_attributes={1: {"description": "synthetic chain node"}},
        edge_attributes={(1, 2): {"relation": "synthetic"}})
    show(graph.summary())
    reopened = oe.open_network(destination, max_memory_mb=8)
    neighbors = [row for block in reopened.neighbors(1) for row in block]
    show(oe.DataFrame(neighbors))
    assert neighbors == [{"node": 2, "weight": 2.0}]
    assert list(reopened.neighbors("isolated")) == []
    disk_storage_proof = dict(nodes=reopened.node_count, edges=reopened.edge_count,
        disk_bytes=reopened.metadata["disk_bytes"],
        bounded_rows=reopened.metadata["actual_peak_batch_rows"],
        max_memory_bytes=reopened.metadata["max_memory_bytes"],
        reopened=True, source_runtime_only=not getattr(sys, "frozen", False))
    print(disk_storage_proof)
    graph.close()
    reopened.close()
