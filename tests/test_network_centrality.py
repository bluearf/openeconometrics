"""Hand formulas and exhaustive simple-path oracles; no external graph solver."""
from fractions import Fraction
import itertools
import math
import random

import pytest
import torch

import openecon._network_centrality as centrality
from openecon.analysis_contracts import AnalysisError
from openecon.networks import network


def _values(frame, column):
    return dict(zip(frame.node, frame[column]))


def _graph(edges, directed=False, weighted=False, nodes=()):
    return network([dict(source=s, target=t, w=w) for s, t, w in edges],
                   weight="w" if weighted else None, directed=directed, nodes=nodes)


def _oracle(edges, nodes, directed, weighted, endpoints=False, normalized=True):
    """Enumerate all positive-cost simple paths with exact rational distances."""
    labels = list(nodes)
    adjacency = {node: {} for node in labels}
    for s, t, w in edges:
        adjacency[s][t] = adjacency[s].get(t, Fraction()) + Fraction(w)
        if not directed and s != t:
            adjacency[t][s] = adjacency[t].get(s, Fraction()) + Fraction(w)
    if not weighted:
        adjacency = {s: {t: Fraction(1) for t in targets} for s, targets in adjacency.items()}
    output = {node: Fraction() for node in labels}
    for source, target in itertools.permutations(labels, 2):
        found = []

        def walk(path, cost):
            if path[-1] == target:
                found.append((cost, path))
                return
            for following, distance in adjacency[path[-1]].items():
                if following not in path:
                    walk([*path, following], cost + distance)

        walk([source], Fraction())
        if found:
            shortest = min(cost for cost, _ in found)
            paths = [path for cost, path in found if cost == shortest]
            for node in labels:
                output[node] += Fraction(sum(node in (path if endpoints else path[1:-1]) for path in paths),
                                         len(paths))
    n = len(labels)
    scale = Fraction(1)
    if normalized:
        denominator = n * (n - 1) if endpoints else (n - 1) * (n - 2)
        scale = Fraction(1, denominator) if (n >= 2 if endpoints else n >= 3) else Fraction()
    elif not directed:
        scale /= 2
    return {node: float(value * scale) for node, value in output.items()}


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("normalized,endpoints", list(itertools.product([False, True], repeat=2)))
def test_brandes_matches_exhaustive_fraction_path_oracle(directed, weighted, normalized, endpoints):
    edges = [("a", "b", 1), ("a", "c", 1), ("b", "d", 1), ("c", "d", 1),
             ("a", "d", 3), ("d", "b", 2), ("b", "b", 7), ("c", "d", 1)]
    nodes = ["a", "b", "c", "d", "isolate"]
    graph = _graph(edges, directed, weighted, nodes)
    actual = centrality.betweenness(graph, normalized=normalized, endpoints=endpoints)
    assert _values(actual, "betweenness") == pytest.approx(
        _oracle(edges, nodes, directed, weighted, endpoints, normalized), abs=1e-12)
    assert actual.attrs["exact"] and not actual.attrs["sampled"]
    assert actual.attrs["shortest_paths"] == ("Dijkstra" if weighted else "BFS")


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("endpoints", [False, True])
def test_three_path_normalization_hand_values(directed, endpoints):
    graph = _graph([(0, 1, 1), (1, 2, 1)], directed=directed)
    expected = ([1 / 3, .5, 1 / 3] if endpoints else [0., .5, 0.]) if directed else (
        [2 / 3, 1., 2 / 3] if endpoints else [0., 1., 0.])
    assert centrality.betweenness(graph, endpoints=endpoints).betweenness.tolist() == pytest.approx(expected)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("normalized,endpoints", list(itertools.product([False, True], repeat=2)))
def test_sample_estimator_is_unbiased_over_all_source_subsets(monkeypatch, directed, normalized, endpoints):
    graph = _graph([(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 2, 1)], directed)
    expected = centrality.betweenness(graph, normalized=normalized, endpoints=endpoints).betweenness.tolist()
    for k in [1, 2, 3]:
        results = []
        for subset in itertools.combinations(range(4), k):
            class _Chosen:
                def __init__(self, seed):
                    pass

                def sample(self, population, count):
                    assert count == k and list(population) == list(range(4))
                    return subset

            monkeypatch.setattr(centrality.random, "Random", _Chosen)
            result = centrality.betweenness(graph, samples=k, normalized=normalized, endpoints=endpoints)
            assert result.attrs["exact"] is False and result.attrs["sampled"] is True
            assert result.attrs["source_sum_estimator_scale"] == 4 / k
            results.append(result.betweenness.tolist())
        average = [math.fsum(row[i] for row in results) / len(results) for i in range(4)]
        assert average == pytest.approx(expected, abs=1e-12)


def test_seed_reproducibility_and_all_sources_exact():
    graph = _graph([(i, i + 1, 1) for i in range(7)], directed=True)
    first = centrality.betweenness(graph, samples=3, seed=72)
    second = centrality.betweenness(graph, samples=3, seed=72)
    assert first.betweenness.tolist() == second.betweenness.tolist()
    assert first.attrs["source_indices_sha256"] == second.attrs["source_indices_sha256"]
    third = centrality.betweenness(graph, samples=3, seed=73)
    assert first.attrs["source_indices_sha256"] != third.attrs["source_indices_sha256"]
    assert centrality.betweenness(graph, samples=8).attrs["exact"]
    assert centrality.betweenness(graph, samples=8).betweenness.tolist() == centrality.betweenness(graph).betweenness.tolist()


def test_direction_disconnection_wasserman_faust_and_weighted_harmonic():
    graph = _graph([(0, 1, 2), (1, 2, 3)], directed=True, weighted=True, nodes=[0, 1, 2, 3])
    assert centrality.closeness(graph).closeness.tolist() == pytest.approx([0., 1 / 6, 1 / 6, 0.])
    assert centrality.closeness(graph, wf_improved=False).closeness.tolist() == pytest.approx([0., .5, 1 / 4, 0.])
    assert centrality.closeness(graph, direction="out").closeness.tolist() == pytest.approx([4 / 21, 1 / 9, 0., 0.])
    assert centrality.harmonic(graph).harmonic.tolist() == pytest.approx([0., .5, 1 / 5 + 1 / 3, 0.])
    assert centrality.harmonic(graph, direction="out").harmonic.tolist() == pytest.approx([.5 + .2, 1 / 3, 0., 0.])
    assert centrality.closeness(graph).attrs["direction"] == "in"
    undirected = _graph([(0, 1, 1), (1, 2, 1)], nodes=[3])
    assert centrality.closeness(undirected).closeness.tolist() == pytest.approx([0., 4 / 9, 2 / 3, 4 / 9])
    assert centrality.harmonic(undirected, direction="in").harmonic.tolist() == centrality.harmonic(
        undirected, direction="out").harmonic.tolist()


def test_indexed_heap_has_one_entry_per_node_and_decrease_order():
    distances = [100., 20., 30., 40.]
    queue = centrality._Heap(distances)
    for node in range(4):
        queue.decrease(node)
    distances[0] = 1.
    queue.decrease(0)
    assert len(queue.nodes) == 4
    assert [queue.pop() for _ in range(4)] == [0, 1, 2, 3]
    assert queue.positions == [-2] * 4


def test_log_counts_handle_more_than_float64_number_of_shortest_paths():
    # 2**1050 paths through 1,050 layers; no overflowing sigma vector or bigints.
    edges = [("s", (0, i), 1) for i in range(2)]
    # Graph labels only allow strings/ints, so encode the small layer tuples.
    edges = [(s, str(t), w) for s, t, w in edges]
    for layer in range(1, 1050):
        edges.extend((str((layer - 1, i)), str((layer, j)), 1) for i in range(2) for j in range(2))
    edges.extend((str((1049, i)), "t", 1) for i in range(2))
    graph = _graph(edges, directed=True)
    topology = centrality.csr(graph, loops=False)
    _, order, sigma = centrality._paths(topology, graph._index["s"], False, count_paths=True)
    assert len(order) == graph.node_count
    assert sigma[graph._index["t"]] == pytest.approx(1050 * math.log(2), rel=1e-13)
    assert sigma[graph._index["t"]] > math.log(torch.finfo(torch.float64).max)


@pytest.mark.parametrize("metric", [centrality.betweenness, centrality.closeness, centrality.harmonic])
def test_work_and_memory_guard_before_topology_allocation(monkeypatch, metric):
    graph = _graph([(i, i + 1, 1) for i in range(20)])
    monkeypatch.setattr(centrality, "csr", lambda *args, **kwargs: pytest.fail("Budget must run first"))
    with pytest.raises(AnalysisError) as caught:
        metric(graph, max_work=1)
    assert caught.value.code == "network_work_budget"
    graph._budget.limit = graph._base_bytes + 100
    with pytest.raises(AnalysisError) as caught:
        metric(graph)
    assert caught.value.code == "network_memory_budget"


@pytest.mark.parametrize("metric", [centrality.betweenness, centrality.closeness, centrality.harmonic])
def test_huge_distances_nonimproving_overflow_cycle_and_real_overflow(metric):
    cycle = _graph([(0, 1, 1e308), (1, 0, 1e308)], directed=True, weighted=True)
    result = metric(cycle)
    assert all(math.isfinite(value) for value in result.iloc[:, 1])
    chain = _graph([(0, 1, 1e308), (1, 2, 1e308)], directed=True, weighted=True)
    with pytest.raises(AnalysisError) as caught:
        metric(chain)
    assert caught.value.code == "network_precision"


def test_closeness_stable_when_sum_of_finite_distances_overflows():
    graph = _graph([(0, 1, 1e308), (0, 2, 1e308)], directed=True, weighted=True)
    result = centrality.closeness(graph, direction="out", wf_improved=False)
    assert _values(result, "closeness")[0] == pytest.approx(1e-308, rel=1e-13, abs=0.)
    assert _values(centrality.harmonic(graph, direction="out"), "harmonic")[0] == pytest.approx(2e-308, rel=1e-13, abs=0.)


@pytest.mark.parametrize("metric", [centrality.betweenness, centrality.closeness, centrality.harmonic])
def test_lost_positive_weight_distance_is_refused(metric):
    graph = _graph([(0, 1, 1e20), (1, 2, 1.)], directed=True, weighted=True)
    with pytest.raises(AnalysisError) as caught:
        metric(graph)
    assert caught.value.code == "network_precision"


@pytest.mark.parametrize("metric", [centrality.closeness, centrality.harmonic])
def test_reciprocal_overflow_refused(metric):
    graph = _graph([(0, 1, 5e-324)], directed=True, weighted=True)
    with pytest.raises(AnalysisError) as caught:
        metric(graph)
    assert caught.value.code == "network_precision"


def test_eigenvector_left_orientation_weight_strength_and_independent_two_node_formula():
    graph = _graph([(0, 1, 4.), (1, 0, 1.)], directed=True, weighted=True)
    result = centrality.eigenvector(graph, tol=1e-12)
    assert result.eigenvector.tolist() == pytest.approx([1 / math.sqrt(5), 2 / math.sqrt(5)], abs=1e-12)
    assert result.attrs["eigenvalue"] == pytest.approx(2., abs=1e-12)
    assert result.attrs["relative_eigen_residual"] <= 1e-12
    repeated = _graph([(0, 1, 1.)] * 4 + [(1, 0, 1.)], directed=True)
    assert repeated.weighted is False
    assert centrality.eigenvector(repeated, tol=1e-12).eigenvector.tolist() == pytest.approx(result.eigenvector.tolist(), abs=1e-12)
    assert result.attrs["weight_semantics"] == "aggregate edge strength"


def test_eigenvector_bipartite_star_known_values_and_sparse_no_square_allocation(monkeypatch):
    graph = _graph([(0, i, 1.) for i in range(1, 5)])
    original = torch.zeros

    def bounded(shape, *args, **kwargs):
        assert not isinstance(shape, tuple) or len(shape) != 2
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", bounded)
    result = centrality.eigenvector(graph, tol=1e-12)
    assert result.eigenvector.tolist() == pytest.approx([1 / math.sqrt(2)] + [1 / math.sqrt(8)] * 4, abs=1e-12)
    assert result.attrs["eigenvalue"] == pytest.approx(2.)
    assert result.attrs["iterations"] < 100


def test_eigenvalue_overflow_is_explicit_but_finite_centrality_survives():
    graph = _graph([(0, 1, 1e308), (1, 0, 1e308), (0, 0, 1e308), (1, 1, 1e308)],
                   directed=True, weighted=True)
    result = centrality.eigenvector(graph)
    assert result.eigenvector.tolist() == pytest.approx([1 / math.sqrt(2)] * 2)
    assert result.attrs["eigenvalue"] is None and result.attrs["eigenvalue_overflow"]
    assert result.attrs["eigenvalue_scaled"] == pytest.approx(2.)


def test_eigenvector_scaling_underflow_and_nonconvergence_refused():
    graph = _graph([(0, 1, 1e308), (1, 0, 5e-324)], directed=True, weighted=True)
    with pytest.raises(AnalysisError) as caught:
        centrality.eigenvector(graph)
    assert caught.value.code == "network_precision"
    star = _graph([(0, i, 1.) for i in range(1, 5)])
    with pytest.raises(AnalysisError) as caught:
        centrality.eigenvector(star, max_iter=1, tol=1e-15)
    assert caught.value.code == "network_nonconvergence"


def test_reducible_eigenvector_reports_nonuniqueness_and_selects_dominant_component():
    graph = _graph([(0, 1, 4), (1, 0, 4), (2, 3, 1), (3, 2, 1)], directed=True,
                   weighted=True, nodes=range(5))
    result = centrality.eigenvector(graph, tol=1e-12)
    assert result.eigenvector.tolist() == pytest.approx([1 / math.sqrt(2)] * 2 + [0.] * 3, abs=2e-12)
    assert result.attrs["eigenvalue"] == pytest.approx(4.)
    assert "not assumed" in result.attrs["uniqueness"]
    equal = _graph([(0, 1, 1), (2, 3, 1)], nodes=range(4))
    assert centrality.eigenvector(equal).eigenvector.tolist() == pytest.approx([.5] * 4)


@pytest.mark.parametrize("directed,expected", [(False, 1.), (True, .5)])
def test_two_node_endpoints_have_nonzero_normalization(directed, expected):
    graph = _graph([(0, 1, 1)], directed=directed)
    assert centrality.betweenness(graph, endpoints=True).betweenness.tolist() == [expected] * 2
    assert centrality.betweenness(graph).betweenness.tolist() == [0.] * 2


@pytest.mark.parametrize("n", [0, 1, 2])
def test_empty_isolate_and_normalization_zero(n):
    graph = _graph([], nodes=list(range(n)))
    for function, column in [(centrality.betweenness, "betweenness"),
                             (centrality.closeness, "closeness"), (centrality.harmonic, "harmonic")]:
        assert function(graph)[column].tolist() == [0.] * n
    assert centrality.betweenness(graph, endpoints=True).betweenness.tolist() == [0.] * n
    result = centrality.eigenvector(graph)
    assert result.eigenvector.tolist() == pytest.approx([1 / math.sqrt(n)] * n if n else [])
    assert result.attrs["iterations"] == 0 and result.attrs["eigenvalue"] == 0.


@pytest.mark.parametrize("function,kwargs", [
    (centrality.betweenness, {"samples": 0}), (centrality.betweenness, {"samples": 4}),
    (centrality.betweenness, {"samples": True}), (centrality.betweenness, {"seed": -1}),
    (centrality.betweenness, {"normalized": 1}), (centrality.betweenness, {"endpoints": "yes"}),
    (centrality.closeness, {"direction": "both"}), (centrality.harmonic, {"direction": []}),
    (centrality.closeness, {"wf_improved": 1}), (centrality.harmonic, {"max_work": True}),
    (centrality.eigenvector, {"max_iter": 0}), (centrality.eigenvector, {"tol": 0}),
    (centrality.eigenvector, {"tol": math.inf}), (centrality.eigenvector, {"tol": True}),
])
def test_option_guards(function, kwargs):
    graph = _graph([(0, 1, 1), (1, 2, 1)])
    with pytest.raises(AnalysisError) as caught:
        function(graph, **kwargs)
    assert caught.value.code == "network_invalid_option"


def test_algorithms_preserve_graph_and_caller_torch_defaults():
    graph = _graph([(0, 1, 1), (1, 2, 2), (2, 0, 3)], weighted=True)
    before_index = graph._edges.indices().clone()
    before_values = graph._edges.values().clone()
    original = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        for function in [centrality.betweenness, centrality.closeness, centrality.harmonic, centrality.eigenvector]:
            result = function(graph)
            assert result.attrs["device"] == "cpu"
            assert torch.get_default_dtype() == torch.float32
        assert torch.equal(before_index, graph._edges.indices())
        assert torch.equal(before_values, graph._edges.values())
    finally:
        torch.set_default_dtype(original)


def test_random_weighted_dag_matches_independent_fraction_oracle():
    rng = random.Random(329)
    for _ in range(8):
        edges = [(i, j, rng.randint(1, 5)) for i in range(5) for j in range(i + 1, 5) if rng.random() < .65]
        graph = _graph(edges, directed=True, weighted=True, nodes=range(5))
        assert _values(centrality.betweenness(graph), "betweenness") == pytest.approx(
            _oracle(edges, range(5), True, True), abs=1e-12)
