"""Synthetic disk analysis for source or installed Mac code panels."""
from pathlib import Path
import sys
import tempfile

import openecon as oe
from openecon.analysis_contracts import AnalysisError

show = globals().get("display", print)

with tempfile.TemporaryDirectory(prefix="openecon-disk-analysis-") as directory:
    graph = oe.network((dict(source=i, target=(i + offset) % 200, count=2)
                        for i in range(200) for offset in range(50)),
        weight="count", directed=True, store=Path(directory) / "graph.sqlite",
        max_memory_mb=2, batch_rows=32)
    resident_guarded = False
    try:
        graph.materialize()
    except AnalysisError as exc:
        resident_guarded = exc.code == "network_memory_budget"
    assert resident_guarded
    reopened = oe.open_network(graph.path, max_memory_mb=2)
    degree = reopened.degree(batch_rows=32)
    rank = reopened.pagerank(batch_rows=32)
    components = reopened.components(batch_rows=32)
    show(reopened.summary())
    show(degree.head(10))
    show(rank.head(10))
    show(components.head(10))
    assert degree.out_degree.tolist() == [50] * 200
    assert abs(float(rank.pagerank.sum()) - 1) < 1e-12
    assert components.component.nunique() == 1
    disk_analysis_proof = dict(nodes=200, edges=10_000,
        resident_guarded=resident_guarded, reopened=True,
        pagerank_sum=float(rank.pagerank.sum()), components=int(components.component.nunique()),
        degree_metadata=degree.attrs, pagerank_metadata=rank.attrs,
        component_metadata=components.attrs, source_runtime_only=not getattr(sys, "frozen", False))
    print("Disk analysis complete: 200 nodes, 10,000 edges; resident import refused; exact CPU results.")
    graph.close()
    reopened.close()
