"""MARKET-60/67: direct incidence, explicit projections and bounded strong paths."""
import json
from fractions import Fraction

import openecon as oe
import torch

hyper = oe.hypergraph([
    {"id": 1, "members": [1, "1", "a"], "weight": 2},
    {"id": "1", "members": [1, "a"], "weight": 0},
    {"id": "solo", "members": ["a"], "weight": 3},
], nodes=[1, "1", "a", "isolated"])
degree = hyper.degree()
assert degree.hyperdegree.tolist() == [2, 1, 3, 0]
assert degree.strength.tolist() == [2, 2, 5, 0]
assert hyper.incidence().layout == torch.sparse_coo
assert hyper.incidence_matvec(torch.ones(3)).tolist() == [2, 1, 3, 0]
clique = hyper.clique_projection(max_edges=10)
star = hyper.star_projection(max_edges=10)
assert clique.node_count == 4 and clique.edge_count == 3
assert star.node_count == 7 and star.edge_count == 4
graph = oe.network([
    {"source": "s", "target": 1, "cost": .1},
    {"source": 1, "target": "t", "cost": .2},
    {"source": "s", "target": "1", "cost": .3},
    {"source": "1", "target": "t", "cost": .1},
    {"source": "t", "target": "s", "cost": 1},
], directed=True, weight="cost")
paths = graph.k_shortest_paths("s", "t", k=5, max_path_length=3,
    max_output_nodes=100, max_frontier=100, max_work=100_000)
assert paths.path.tolist() == [("s", 1, "t"), ("s", "1", "t")]
assert Fraction(paths.cost_numerator[0], paths.cost_denominator[0]) == Fraction(.1) + Fraction(.2)
bridges = graph.strong_bridges(max_work=100_000)
articulations = graph.strong_articulation_points(max_work=100_000)
assert len(bridges) == 5
assert set(articulations.loc[articulations.strong_articulation, "node"]) == {"s", "t"}
display(hyper.summary())
display(degree)
display(paths)
display(bridges)
display(articulations)
display(oe.plot.network(clique, title="Explicit hypergraph clique projection", layout="circular"))
print("HYPERGRAPH_PATHS_RECEIPT:" + json.dumps({
    "nodes": hyper.node_count, "hyperedges": hyper.edge_count,
    "memberships": hyper.metadata["memberships"], "sparse_incidence": True,
    "hyperdegree": degree.hyperdegree.tolist(), "strength": degree.strength.tolist(),
    "clique_edges": clique.edge_count, "star_nodes": star.node_count, "star_edges": star.edge_count,
    "paths": paths.path.tolist(), "strong_bridges": len(bridges),
    "strong_articulation_nodes": articulations.loc[articulations.strong_articulation, "node"].tolist(),
    "path_work": paths.attrs["work_used"], "path_max_work": paths.attrs["max_work"],
}, allow_nan=False))
