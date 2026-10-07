"""Exact time boundaries and typed identities on synthetic, disposable data."""
from pathlib import Path
import json
import tempfile

import openecon as oe

show = globals().get("display", print)
graph = oe.dynamic_network([
    dict(node=1, dynamic_attributes={"phase": [dict(value="early", end=1), dict(value="late", startopen=1)]}),
    dict(node="1"), dict(node="c"), dict(node="isolated"),
], [
    dict(edge_id=1, source=1, target="1", weight=2., end=1),
    dict(edge_id="1", source=1, target="1", weight=3., end=1),
    dict(edge_id="zero", source=1, target=1, weight=0., end=1),
    dict(edge_id="second", source="1", target="c", weight=4., startopen=1),
], directed=True, version="1.2draft")
original = graph.edges().to_dict("records")
show(graph.edges())
first = graph.at(1)
second = graph.at(2)
assert first.edges().edge_id.tolist() == [1, "1", "zero"]
assert first.node_attributes[1]["phase"] == "early"
assert second.node_attributes[1]["phase"] == "late"
show(first.degree())
# A window needs an explicit decision about changing attribute values.
window = graph.window(0, 2, attributes="drop")
show(window.degree())
# Point projections explicitly retain zero topology via count and sum parallels.
layers = oe.network_snapshots({
    "first": first.to_network(reducer="count", attributes="drop"),
    "second": second.to_network(reducer="count", attributes="drop"),
}, ordered=True)
route = layers.temporal_path(1, "c")
assert route[["source", "target", "layer"]].to_dict("records") == [
    dict(source=1, target="1", layer="first"), dict(source="1", target="c", layer="second")]
show(route)
with tempfile.TemporaryDirectory(prefix="openecon-dynamic-") as directory:
    path = Path(directory) / "owned.gexf"
    graph.write(path)
    reopened = oe.read_dynamic_network(path)
    assert reopened.edges().to_dict("records") == original
    assert reopened.nodes().to_dict("records") == graph.nodes().to_dict("records")
    assert reopened.at(1).edges().to_dict("records") == first.edges().to_dict("records")
    show(reopened.edges())
assert graph.edges().to_dict("records") == original
dynamic_proof = dict(separate_edges=graph.edge_count, nodes=graph.node_count,
    roundtrip=True, source_unchanged=True, typed_edge_ids_distinct=type(first.edges().edge_id.iloc[0]) is int
        and type(first.edges().edge_id.iloc[1]) is str,
    boundary_verified=True, ordered_temporal_route_verified=True,
    explicit_window_loss=window.metadata["dynamic_selection"]["dropped_dynamic_attributes"],
    point_count=first.edge_count, temporal_path_model=window.metadata["dynamic_selection"]["temporal_path_model"])
print("DYNAMIC_NETWORK_PROOF:" + json.dumps(dynamic_proof, allow_nan=False))
