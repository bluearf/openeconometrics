"""Coupled trade/finance states; only owned synthetic data and temporary files."""
from fractions import Fraction
from pathlib import Path
import json
import tempfile

import openecon as oe

show = globals().get("display", print)
states = [(1, "trade"), ("1", "trade"), (1, "finance"), ("isolated", "finance")]
edges = [
    dict(edge_id=1, source=1, source_layer="trade", target="1", target_layer="trade", weight=2.),
    dict(edge_id="1", source=1, source_layer="trade", target="1", target_layer="trade", weight=3.),
    dict(edge_id="coupling", source=1, source_layer="trade", target=1, target_layer="finance", weight=.5),
    dict(edge_id="return", source=1, source_layer="finance", target=1, target_layer="trade", weight=4.),
    dict(edge_id="loop", source=1, source_layer="finance", target=1, target_layer="finance", weight=7.),
    dict(edge_id="zero", source="isolated", source_layer="finance", target="1", target_layer="trade", weight=0.),
]
graph = oe.multilayer_network(edges, weight="weight", directed=True, nodes=states,
                             layers=["trade", "finance", "empty"], max_memory_mb=128)
original = graph.edges().to_dict("records")
show(graph.layers())
show(graph.nodes())
show(graph.edges())
product = graph.matvec(dict(zip(states, [-1, 2, 3, 10])))
# Independent row equations: 5*2 + .5*3, 0, 4*(-1)+7*3, 0.
assert product.value.tolist() == [11.5, 0., 17., 0.]
show(product)
rank = graph.pagerank(tol=1e-12, max_iter=500)
# Fraction elimination of (I - .85 P.T) r = .15/4, including two dangling states.
a = [[Fraction(0), Fraction(5), Fraction(1, 2), Fraction(0)],
     [Fraction(0)] * 4, [Fraction(4), Fraction(0), Fraction(7), Fraction(0)], [Fraction(0)] * 4]
p = [[v / sum(row) if sum(row) else Fraction(1, 4) for v in row] for row in a]
system = [[Fraction(i == j) - Fraction(17, 20) * p[j][i] for j in range(4)] + [Fraction(3, 80)] for i in range(4)]
for i in range(4):
    div = system[i][i]
    system[i] = [v / div for v in system[i]]
    for j in range(4):
        if j != i:
            factor = system[j][i]
            system[j] = [v - factor * w for v, w in zip(system[j], system[i])]
expected_rank = [float(row[-1]) for row in system]
assert max(abs(a - b) for a, b in zip(expected_rank, rank.pagerank)) < 1e-12
show(rank)
show(graph.layer("trade").edges())
projection = graph.project(reducer="count", inter_layer="include", attributes="drop")
show(projection.degree())
edited = graph.edit_edges(weights={"coupling": 2.})
with tempfile.TemporaryDirectory(prefix="openecon-multilayer-") as directory:
    path = Path(directory) / "owned.ndjson"
    edited.write(path)
    reopened = oe.read_multilayer_network(path)
    assert reopened.edges().to_dict("records") == edited.edges().to_dict("records")
    assert reopened.nodes().to_dict("records") == edited.nodes().to_dict("records")
    assert reopened.layers().to_dict("records") == edited.layers().to_dict("records")
    show(reopened.edges())
assert graph.edges().to_dict("records") == original
unweighted = oe.multilayer_network([dict(edge_id="u", source="A", source_layer="trade",
                                        target="A", target_layer="finance")])
assert unweighted.edit_edges(attributes={"u": {"kind": "coupling"}}).weighted is False
multilayer_proof = dict(states=graph.node_count, separate_edges=graph.edge_count, layers=graph.layer_count,
    exact_identity_roundtrip=True, source_unchanged=True, unweighted_edit_preserved=True,
    oracle_max_error=max(abs(a-b) for a,b in zip(expected_rank, rank.pagerank)),
    adjacency_product=product.value.tolist(), pagerank=rank.pagerank.tolist(),
    projection=projection.metadata["multilayer_projection"], owned_memory=graph.metadata["estimated_import_peak_bytes"])
print("MULTILAYER_PROOF:" + json.dumps(multilayer_proof, allow_nan=False))
