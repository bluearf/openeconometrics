"""Separate typed edges and owned interchange files; no human project data."""
from pathlib import Path
import json
import tempfile

import openecon as oe

show = globals().get("display", print)
graph = oe.multigraph([
    dict(edge_id=1, source="1", target=1, weight=2.),
    dict(edge_id="1", source="1", target=1, weight=3.),
    dict(edge_id="loop", source=1, target=1, weight=7.),
    dict(edge_id="zero", source="z", target="1", weight=0.),
], weight="weight", nodes=["isolated"],
    edge_attributes={1: {"relation": "trade"}, "1": {"relation": "loan"},
                     "loop": {"relation": "self"}, "zero": {"relation": "recorded zero"}})
original = graph.edges().to_dict("records")
show(graph.edges())
show(graph.degree())
changed = graph.edit_edges(weights={1: 9.}).filter(edge_ids=[1, "loop", "zero"])
show(changed.edges())
# Count preserves zero-edge topology; different attributes are explicitly dropped.
simple = graph.to_network(reducer="count", attributes="drop")
show(simple.degree())
with tempfile.TemporaryDirectory(prefix="openecon-multigraph-") as directory:
    roundtrips = []
    for format in ("graphml", "gexf"):
        path = Path(directory) / ("owned." + format)
        graph.write(path)
        reopened = oe.read_multigraph(path)
        assert reopened.edges().to_dict("records") == original
        roundtrips.append(format)
    show(reopened.edges())
assert graph.edges().to_dict("records") == original
multigraph_proof = dict(separate_edges=graph.edge_count, nodes=graph.node_count,
    roundtrip=roundtrips == ["graphml", "gexf"], source_unchanged=True,
    projection=simple.metadata["multigraph_projection"], typed_edge_ids_distinct=1 in graph.edge_attributes and "1" in graph.edge_attributes)
print("MULTIGRAPH_PROOF:" + json.dumps(multigraph_proof, allow_nan=False))
