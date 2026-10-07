from fractions import Fraction
import itertools
import random

import pytest
import torch

import openecon as oe


def path_oracle(edges, source, target, nodes, hops=None):
    adjacency = {n: [] for n in nodes}
    for (u, v), w in edges.items():
        adjacency[u].append((v, Fraction(w)))
    found = []
    def visit(path, cost):
        if path[-1] == target:
            found.append((cost, tuple(path)))
            return
        if len(path) - 1 >= (len(nodes) - 1 if hops is None else hops):
            return
        for v, w in adjacency[path[-1]]:
            if v not in path:
                visit(path + [v], cost + w)
    visit([source], Fraction())
    def key(n):
        return (0, n) if isinstance(n, int) else (1, n)
    return sorted(found, key=lambda item: (item[0], tuple(map(key, item[1]))))


def scc_oracle(edges, nodes):
    # Independent all-source reachability, intentionally dense only in tiny tests.
    reach = {}
    for s in nodes:
        seen, todo = {s}, [s]
        while todo:
            u = todo.pop()
            for a, b in edges:
                if a == u and b in nodes and b not in seen:
                    seen.add(b)
                    todo.append(b)
        reach[s] = seen
    classes = {frozenset(v for v in nodes if v in reach[u] and u in reach[v]) for u in nodes}
    return len(classes)


@pytest.mark.parametrize("mask", range(64))
def test_all_three_node_directed_topologies(mask):
    nodes = [0, 1, 2]
    candidates = list(itertools.permutations(nodes, 2))
    edges = {edge: 1 for i, edge in enumerate(candidates) if mask & (1 << i)}
    g = oe.network([dict(source=u, target=v) for u, v in edges], directed=True, nodes=nodes)
    expected = path_oracle(edges, 0, 2, nodes)
    paths = g.k_shortest_paths(0, 2, k=20)
    assert paths.path.tolist() == [p for _, p in expected]
    assert paths.cost.tolist() == [float(c) for c, _ in expected]
    base = scc_oracle(edges, nodes)
    expected_edges = {e for e in edges if scc_oracle(set(edges) - {e}, nodes) > base}
    assert {(r.source, r.target) for r in g.strong_bridges().itertuples()} == expected_edges
    assert g.strong_articulation_points().strong_articulation.tolist() == [
        scc_oracle([e for e in edges if u not in e], [v for v in nodes if v != u]) > base for u in nodes]


@pytest.mark.parametrize("seed", range(35))
def test_weighted_random_path_and_strong_cut_oracles(seed):
    rng = random.Random(seed)
    nodes = [1, "1", "a", 4, "z"]
    edges = {(u, v): rng.choice([.1, .2, 1., 2.]) for u in nodes for v in nodes if rng.random() < .3}
    g = oe.network([dict(source=u, target=v, w=w) for (u, v), w in edges.items()], directed=True, nodes=nodes, weight="w")
    for hops in [None, 2]:
        expected = path_oracle(edges, 1, "z", nodes, hops)[:7]
        result = g.k_shortest_paths(1, "z", k=7, max_path_length=hops)
        assert result.path.tolist() == [p for _, p in expected]
        assert [Fraction(int(r.cost_numerator), int(r.cost_denominator)) for r in result.itertuples()] == [c for c, _ in expected]
    base = scc_oracle(edges, nodes)
    assert {(r.source, r.target) for r in g.strong_bridges().itertuples()} == {
        e for e in edges if scc_oracle(set(edges) - {e}, nodes) > base}


def test_ties_loops_duplicates_undirected_and_identity():
    rows = [dict(source="s", target=v, w=1) for v in [1, "1"]] + [dict(source=v, target="t", w=1) for v in [1, "1"]]
    rows += [dict(source="s", target="s", w=100), dict(source="s", target=1, w=1)]
    g = oe.network(rows, directed=True, weight="w", nodes=["isolate"])
    assert g.k_shortest_paths("s", "t").path.tolist() == [("s", "1", "t"), ("s", 1, "t")]
    reverse = oe.network(list(reversed(rows)), directed=True, weight="w")
    assert g.k_shortest_paths("s", "t").path.tolist() == reverse.k_shortest_paths("s", "t").path.tolist()
    assert g.k_shortest_paths("s", "s").path.tolist() == [("s",)]
    assert len(g.k_shortest_paths("s", "isolate")) == 0
    u = oe.network([dict(source="a", target="b"), dict(source="b", target="c")])
    assert u.k_shortest_paths("c", "a").path.tolist() == [("c", "b", "a")]
    with pytest.raises(oe.AnalysisError, match="directed"):
        u.strong_bridges()
    assert len(g.bridges()) >= 0  # existing weak API remains callable


def test_explicit_budgets_and_no_partial_result():
    g = oe.network([dict(source=u, target=v) for u in range(6) for v in range(6) if u != v], directed=True)
    for options in [{"k": 0}, {"max_path_length": 6}, {"max_frontier": 1}, {"max_output_nodes": 1}, {"max_work": 1}]:
        with pytest.raises(oe.AnalysisError):
            g.k_shortest_paths(0, 5, **options)
    with pytest.raises(oe.AnalysisError, match="max_work"):
        g.strong_bridges(max_work=100)
    with pytest.raises(oe.AnalysisError, match="max_work"):
        g.strong_articulation_points(max_work=100)
    assert len(g.k_shortest_paths(0, 5, max_path_length=0)) == 0
    graph_copy = g._edges.clone()
    torch.set_default_device("meta")
    try:
        assert len(g.k_shortest_paths(0, 5, k=2)) == 2
        assert len(g.strong_bridges()) == 0
    finally:
        torch.set_default_device("cpu")
    assert torch.equal(g._edges.indices(), graph_copy.indices())
    assert torch.equal(g._edges.values(), graph_copy.values())
