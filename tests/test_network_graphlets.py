import random

import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def graph(edges, directed=False, n=4):
    return oe.network(
        [dict(source=u, target=v) for u, v in edges], directed=directed, nodes=range(n)
    )


@pytest.mark.parametrize(
    "edges,expected",
    [
        ([(0, 1), (1, 2), (2, 3)], [2, 2]),
        ([(0, 1), (0, 2), (0, 3)], [1, 3]),
        ([(0, 1), (1, 2), (2, 3), (3, 0)], [4]),
        ([(u, v) for u in range(4) for v in range(u + 1, 4)], [4]),
    ],
)
def test_hand_graphlet_automorphism_orbits(edges, expected):
    result = graph(edges).graphlets()
    assert result["classes"]["count"].tolist() == [1]
    sizes = result["orbits"].groupby("orbit", sort=False).size().tolist()
    assert sorted(sizes) == sorted(expected)
    assert result["orbits"]["count"].sum() == 4


def test_complete_undirected_four_node_catalogue_has_eleven_classes_six_connected():
    pairs = [(u, v) for u in range(4) for v in range(u + 1, 4)]
    all_classes, connected = set(), set()
    for mask in range(64):
        g = graph([p for at, p in enumerate(pairs) if mask & (1 << at)])
        all_classes.update(g.graphlets(connected=False)["classes"].class_id)
        connected.update(g.graphlets()["classes"].class_id)
    assert len(all_classes) == 11 and len(connected) == 6


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_classes_and_orbit_equivalence_match_independent_graphmatcher(directed, seed):
    nx = pytest.importorskip("networkx")
    rng = random.Random(seed)
    edges = [
        (u, v)
        for u in range(4)
        for v in range(4)
        if u != v and (directed or u < v) and rng.random() < 0.5
    ]
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from(edges)
    cls = (
        nx.algorithms.isomorphism.DiGraphMatcher
        if directed
        else nx.algorithms.isomorphism.GraphMatcher
    )
    automorphisms = list(cls(G, G).isomorphisms_iter())
    expected = {u: frozenset(p[u] for p in automorphisms) for u in G}
    result = graph(edges, directed).graphlets(connected=False)
    actual = result["orbits"].set_index("node")["orbit"].to_dict()
    for u in G:
        assert {v for v in G if actual[v] == actual[u]} == set(expected[u])
    permutation = list(range(4))
    rng.shuffle(permutation)
    remapped = graph([(permutation[u], permutation[v]) for u, v in edges], directed).graphlets(
        connected=False
    )
    assert result["classes"].class_id.tolist() == remapped["classes"].class_id.tolist()


def test_directed_cycle_and_five_node_extension_and_induced_semantics():
    result = graph([(i, (i + 1) % 4) for i in range(4)], True).graphlets()
    assert result["orbits"].orbit.nunique() == 1
    g = graph([(u, v) for u in range(5) for v in range(u + 1, 5)], n=5)
    assert g.graphlets(size=5)["orbits"].orbit.nunique() == 1
    assert g.graphlets()["classes"]["count"].tolist() == [5]
    assert g.graphlets()["metadata"]["induced"] is True


@pytest.mark.parametrize(
    "options", [{"max_subgraphs": 0}, {"max_work": 10}, {"max_output_rows": 3}, {"size": 3}]
)
def test_full_enumeration_and_output_budgets_fail_explicitly(options):
    with pytest.raises(AnalysisError):
        graph([(0, 1), (1, 2), (2, 3)]).graphlets(**options)


def test_loops_multiedges_and_disconnected_classes_are_explicit():
    with pytest.raises(AnalysisError, match="Looped"):
        graph([(0, 0)]).graphlets(connected=False)
    from openecon._network_graphlets import graphlets

    m = oe.multigraph([dict(edge_id="a", source=0, target=1)], nodes=range(4))
    with pytest.raises(AnalysisError, match="projection"):
        graphlets(m)
    result = graph([]).graphlets(connected=False)
    assert result["classes"].class_id.tolist() == ["u4:0"]
    assert result["orbits"].orbit.nunique() == 1
    assert graph([]).graphlets()["classes"].empty
