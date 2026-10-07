"""Run five native network extensions with synthetic data in the code panel.

No input files, installation, cloud services or optional scientific engines.
The assertions provide small hand-checkable examples, not scale guarantees.
"""

import json
import sys

import openecon as oe


def graph(rows, **options):
    return oe.network([dict(source=u, target=v, w=w) for u, v, w in rows], weight="w", **options)


hyper = oe.hypergraph(
    [dict(id=1, members=[1, "1", "a"], weight=2), dict(id="1", members=["a", "b"], weight=3)],
    nodes=[1, "1", "a", "b", "isolate"],
)
assert hyper.incidence().shape == (5, 2)
assert hyper.degree().hyperdegree.tolist() == [1, 1, 2, 1, 0]
assert hyper.degree().strength.tolist() == [2, 2, 5, 3, 0]
assert hyper.clique_projection().edge_count == 4
assert hyper.star_projection().edge_count == 5

cycle = graph([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)])
motif = cycle.graphlets()
assert motif["classes"]["count"].tolist() == [1]
assert len(motif["orbits"]) == 4 and motif["orbits"].orbit.nunique() == 1

routes = graph([(0, 1, 1), (1, 3, 1), (0, 2, 1), (2, 3, 2)], directed=True)
paths = routes.k_shortest_paths(0, 3, k=2, max_path_length=3, max_output_nodes=8)
assert paths.cost.tolist() == [2, 3]
directed_cycle = graph([(0, 1, 1), (1, 2, 1), (2, 0, 1)], directed=True)
strong_edges = directed_cycle.strong_bridges()
strong_nodes = directed_cycle.strong_articulation_points()
assert len(strong_edges) == len(strong_nodes) == 3
assert strong_nodes.strong_articulation.all()
assert len(directed_cycle.bridges()) == 0

assignment_graph = graph([(0, 2, 9), (0, 3, 8), (1, 2, 8), (1, 3, 1)])
assignment = assignment_graph.weighted_assignment(partition={0: 0, 1: 0, 2: 1, 3: 1})
assert assignment["weight"] == 16 and len(assignment["pairs"]) == 2
assert assignment["metadata"]["certificate"]["zero_duality_gap"] is True
odd = graph([(0, 1, 4), (1, 2, 7), (2, 0, 5)], nodes=[0, 1, 2, "isolate"])
matching = odd.general_matching()
assert matching["weight"] == 7 and len(matching["unmatched"]) == 2

y = graph([(0, 1, 2), (0, 2, 1), (1, 2, 4), (3, 1, 3)], nodes=range(4))
a = graph([(2, 3, 3), (1, 0, 1), (3, 1, 2), (1, 2, 5)], nodes=range(4))
b = graph([(2, 3, 1), (3, 0, 4), (1, 3, 5)], nodes=range(4))
regression = y.qap_regression(
    dict(a=a, b=b),
    method="dsp",
    permutations=99,
    seed=42,
    joint={"both": ["a", "b"]},
    adjustment="holm",
)
joint = oe.DataFrame(regression.attrs["joint_tests"])
assert regression.attrs["pvalue_resolution"] == 0.01
assert joint.hypothesis.tolist() == ["both"]

tables = [
    hyper.memberships(),
    hyper.degree(),
    motif["classes"],
    motif["orbits"],
    paths,
    strong_edges,
    strong_nodes,
    assignment["pairs"],
    matching["pairs"],
    regression,
    joint,
]
render = globals().get("display")
for table in tables:
    if render is not None:
        render(table)
    else:
        print(table.to_string(index=False))

print(
    "NETWORK_FIVE_RECEIPT:"
    + json.dumps(
        {
            "issues": ["MARKET-60", "MARKET-64", "MARKET-65", "MARKET-67", "MARKET-70"],
            "sdk_version": oe.__version__,
            "frozen": bool(getattr(sys, "frozen", False)),
            "hypergraph": {"nodes": 5, "edges": 2, "memberships": 5, "sparse_incidence": True},
            "graphlets": {"classes": 1, "orbit_rows": 4, "induced": True},
            "paths": {"costs": [2, 3], "strong_bridges": 3, "strong_articulation_points": 3},
            "matching": {"assignment_weight": 16, "general_weight": 7, "certified": True},
            "mrqap": {
                "method": "dsp",
                "permutations": 99,
                "resolution": 0.01,
                "joint_tests": len(joint),
                "adjustment": "holm",
            },
            "table_rows": [len(table) for table in tables],
        },
        allow_nan=False,
    )
)
