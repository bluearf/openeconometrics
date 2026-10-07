"""Paste into the installed Mac code panel and Run with the signed-enabled SDK.

Integer 1 and string '1' remain separate. Repulsion stays negative in analysis;
PageRank/capacity/count-model calls fail explicitly. All tables are complete.
"""
import json

import openecon as oe
from openecon.analysis_contracts import AnalysisError


graph = oe.signed_network([
    {"source": 1, "target": "1", "weight": 2},
    {"source": "1", "target": "a", "weight": -3},
    {"source": 1, "target": "a", "weight": 1},
    {"source": "zero", "target": "a", "weight": 0},
], directed=True, nodes=[1, "1", "a", "isolate"])
strength = graph.signed_strength()
katz = graph.signed_katz(.1)
paths = graph.signed_shortest_paths(1)
communities = graph.signed_communities()
modularity = graph.signed_modularity(communities)
assert graph.node_count == 5 and graph.edge_count == 3
assert paths.distance.iloc[:3].tolist() == [0, 2, -1]
assert paths.reachable.tolist() == [True, True, True, False, False]
assert katz.signed_katz.iloc[:3].tolist() == [1, 1.2, .74]
rejected = []
for method, args in [("pagerank", ()), ("max_flow", (1, "a")), ("poisson_block_model", (2,))]:
    try:
        getattr(graph, method)(*args)
    except AnalysisError as error:
        assert error.code == "network_signed_method"
        rejected.append(method)
cycle = oe.signed_network([{"source": "a", "target": "b", "weight": -1}])
try:
    cycle.signed_shortest_paths("a")
except AnalysisError as error:
    assert error.code == "network_negative_cycle"
else:
    raise AssertionError("Undirected negative edge must be refused as a negative walk cycle")
for table in (strength, katz, paths, communities):
    render = globals().get("display")
    if render is not None:
        render(table)
    else:
        print(table.to_string(index=False))
print("SIGNED_NETWORK_RECEIPT:" + json.dumps({
    "node_count": graph.node_count, "edge_count": graph.edge_count,
    "typed_integer_and_string_preserved": type(strength.node.iloc[0]) is int and type(strength.node.iloc[1]) is str,
    "path_distances": paths.distance.iloc[:3].tolist(), "signed_katz": katz.signed_katz.iloc[:3].tolist(),
    "signed_modularity": modularity, "communities_converged": communities.attrs["converged"],
    "katz_converged": katz.attrs["converged"], "rejected_legacy_methods": rejected,
    "negative_cycle_rejected": True, "sampled": graph.metadata["sampled"],
    "sdk_version": oe.__version__, "dtype": "float64", "device": "cpu",
}, allow_nan=False))
