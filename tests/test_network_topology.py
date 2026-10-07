"""Independent bounded dense/brute references for native sparse topology."""
from itertools import combinations
import math
import random

import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.networks import network
from openecon import _network_topology as topology


def make(edges=(), *, nodes=(), directed=False, **options):
    return network([{"source": source, "target": target, "weight": weight}
                    for source, target, weight in edges],
                   nodes=nodes, weight="weight", directed=directed, **options)


def values(frame, column):
    return dict(zip(frame.node, frame[column]))


def small_oracle(edges, nodes, directed):
    """Dense scalar adjacency and cube diagonals only for tiny test fixtures."""
    nodes = list(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adjacency = [[0] * n for _ in range(n)]
    for source, target, weight in edges:
        if weight <= 0 or source == target:
            continue
        i, j = index[source], index[target]
        adjacency[i][j] = 1
        if not directed:
            adjacency[j][i] = 1
    incoming = [sum(row[i] for row in adjacency) for i in range(n)]
    outgoing = [sum(row) for row in adjacency]
    reciprocal = [sum(adjacency[i][j] * adjacency[j][i] for j in range(n)) for i in range(n)]
    symmetric = [[adjacency[i][j] + adjacency[j][i] if directed else adjacency[i][j]
                  for j in range(n)] for i in range(n)]
    counts, coefficients, denominators = [], [], []
    for i in range(n):
        count = sum(symmetric[i][j] * symmetric[j][k] * symmetric[k][i]
                    for j in range(n) for k in range(n)) // 2
        degree = incoming[i] + outgoing[i] if directed else outgoing[i]
        denominator = degree * (degree - 1) - 2 * reciprocal[i] if directed else degree * (degree - 1) // 2
        counts.append(count)
        coefficients.append(count / denominator if denominator else 0.0)
        denominators.append(denominator)
    return {"triangles": dict(zip(nodes, counts)), "clustering": dict(zip(nodes, coefficients)),
            "total": sum(counts) // 3,
            "transitivity": sum(counts) / sum(denominators) if sum(denominators) else 0.0,
            "adjacency": adjacency, "incoming": incoming, "outgoing": outgoing}


def core_oracle(edges, nodes):
    """Independent repeated threshold pruning, rather than bin peeling."""
    neighbors = {node: set() for node in nodes}
    for source, target, weight in edges:
        if weight > 0 and source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)
    result = dict.fromkeys(nodes, 0)
    for k in range(1, len(nodes)):
        live = set(nodes)
        while True:
            remove = {node for node in live if len(neighbors[node] & live) < k}
            if not remove:
                break
            live -= remove
        for node in live:
            result[node] = k
    return result


def strong_oracle(edges, nodes, directed):
    """Independent boolean transitive closure for small SCC fixtures."""
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    reachable = [[i == j for j in range(n)] for i in range(n)]
    for source, target, weight in edges:
        if weight <= 0:
            continue
        i, j = index[source], index[target]
        reachable[i][j] = True
        if not directed:
            reachable[j][i] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reachable[i][j] |= reachable[i][k] and reachable[k][j]
    def key(node):
        return (0, node) if isinstance(node, int) else (1, node)
    minima = {node: min((other for j, other in enumerate(nodes)
                        if reachable[i][j] and reachable[j][i]), key=key)
              for i, node in enumerate(nodes)}
    names = {node: rank for rank, node in enumerate(sorted(set(minima.values()), key=key))}
    return {node: names[minimum] for node, minimum in minima.items()}


def association_oracle(reference, directed):
    """Explicit tiny endpoint sample and centered Pearson moments."""
    adjacency = reference["adjacency"]
    samples = []
    for i, row in enumerate(adjacency):
        for j, present in enumerate(row):
            if not present:
                continue
            samples.append((reference["outgoing"][i], reference["incoming"][j])
                           if directed else (sum(row), sum(adjacency[j])))
    if not samples:
        return math.nan
    x = sum(a for a, _ in samples) / len(samples)
    y = sum(b for _, b in samples) / len(samples)
    xx = sum((a - x)**2 for a, _ in samples)
    yy = sum((b - y)**2 for _, b in samples)
    return sum((a - x) * (b - y) for a, b in samples) / math.sqrt(xx * yy) if xx and yy else math.nan


@pytest.mark.parametrize("directed", [False, True])
def test_random_small_graphs_against_cube_core_reachability_and_pearson_oracles(directed):
    generator = random.Random(9284)
    for _ in range(24):
        nodes = list(range(7))
        edges = [(i, j, generator.choice([0.0, .2, 2.0, 1e100]))
                 for i in nodes for j in nodes if generator.random() < .28]
        if edges:
            edges += [edges[0], edges[-1]]
        graph = make(edges, nodes=nodes, directed=directed, batch_rows=3)
        reference = small_oracle(edges, nodes, directed)
        actual = topology.clustering(graph)
        assert values(actual, "triangles") == reference["triangles"]
        assert values(actual, "clustering") == pytest.approx(reference["clustering"], abs=1e-15)
        assert actual.attrs["total_triangles"] == reference["total"]
        assert actual.attrs["transitivity"] == pytest.approx(reference["transitivity"], abs=1e-15)
        assert values(topology.k_core(graph), "core_number") == core_oracle(edges, nodes)
        assert values(topology.components_strong(graph), "component") == strong_oracle(edges, nodes, directed)
        expected = association_oracle(reference, directed)
        actual = topology.assortativity(graph)
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("mask", range(64))
def test_all_directed_three_node_orientations_have_exact_fagiolo_multiplicity(mask):
    arcs = [(i, j) for i in range(3) for j in range(3) if i != j]
    edges = [(source, target, 3.0) for bit, (source, target) in enumerate(arcs) if mask & (1 << bit)]
    graph = make(edges, nodes=[0, 1, 2], directed=True)
    reference = small_oracle(edges, [0, 1, 2], True)
    actual = topology.clustering(graph)
    assert values(actual, "triangles") == reference["triangles"]
    assert values(actual, "clustering") == pytest.approx(reference["clustering"])
    assert actual.attrs["triangle_convention"] == "Fagiolo binary directed, all orientations"
    assert actual.clustering.between(0, 1).all()


@pytest.mark.parametrize("reciprocal_pairs", [0, 1, 2, 3])
def test_reciprocal_triangle_counts_one_two_four_eight_without_weight_products(reciprocal_pairs):
    edges = [(0, 1, 1e100), (1, 2, .1), (2, 0, 100.0)]
    edges += [(target, source, 2.0) for source, target, _ in edges[:reciprocal_pairs]]
    edges += [(0, 0, 7.0), (0, 1, .5), (1, 1, 9.0)]
    graph = make(edges, nodes=[3], directed=True)
    counts = topology.triangles(graph)
    assert values(counts, "triangles") == {3: 0, 0: 2**reciprocal_pairs, 1: 2**reciprocal_pairs, 2: 2**reciprocal_pairs}
    assert counts.attrs["total_triangles"] == 2**reciprocal_pairs
    assert counts.attrs["self_loops_excluded"] == 2
    assert counts.attrs["weight_semantics"] == "unweighted positive-edge topology"
    assert topology.density(graph) == pytest.approx((3 + reciprocal_pairs) / 12)


def test_known_clique_star_path_isolates_loops_and_summary_conventions():
    clique = make([(a, b, 1) for a, b in combinations(range(5), 2)], nodes=range(5))
    counts = topology.triangles(clique)
    assert counts.triangles.tolist() == [6] * 5
    assert counts.attrs["total_triangles"] == 10
    assert topology.clustering(clique).clustering.tolist() == [1.0] * 5
    assert topology.k_core(clique).core_number.tolist() == [4] * 5
    assert topology.transitivity(clique) == topology.density(clique) == 1
    assert math.isnan(topology.assortativity(clique))
    summary = topology.topology_summary(clique)
    assert isinstance(summary, pd.DataFrame)
    assert summary.attrs["assortativity_undefined_reason"] == "zero endpoint degree variance"
    assert dict(zip(summary.Metric, summary.Value))["Degeneracy"] == 4
    assert isinstance(dict(zip(summary.Metric, summary.Value))["Triangles"], int)
    assert "\\begin{table}" in str(summary.to_latex(index=False, caption="Topology"))
    star = make([(0, node, 1) for node in range(1, 6)], nodes=range(7))
    assert topology.triangles(star, max_work=1).attrs["planned_work"] == 0
    assert topology.k_core(star).core_number.tolist() == [1] * 6 + [0]
    assert topology.assortativity(star) == -1
    path = make([(0, 1, 1), (1, 2, 1), (2, 3, 1)], nodes=range(4))
    assert topology.assortativity(path) == pytest.approx(-.5)
    cycle = make([(0, 1, 1), (1, 2, 1), (2, 0, 1)], nodes=[0, 1, 2], directed=True)
    assert topology.clustering(cycle).clustering.tolist() == [.5] * 3
    assert topology.transitivity(cycle) == .5
    assert topology.k_core(cycle).attrs["core_convention"] == "weak simple projection"


@pytest.mark.parametrize("directed", [False, True])
def test_empty_and_only_isolates_or_self_loops_have_typed_finite_zero_results(directed):
    empty = make(nodes=[], directed=directed)
    for operation in (topology.triangles, topology.clustering, topology.k_core, topology.components_strong):
        assert operation(empty).empty
    assert topology.triangles(empty).triangles.dtype == "int64"
    assert topology.k_core(empty).core_number.dtype == "int64"
    assert topology.components_strong(empty).component.dtype == "int64"
    assert topology.density(empty) == topology.transitivity(empty) == 0
    assert math.isnan(topology.assortativity(empty))
    assert topology.topology_summary(empty).attrs["assortativity_undefined_reason"] == "no loop-free edges"
    loops = make([(1, 1, 5), ("1", "1", 2), (1, "z", 0)], nodes=[2], directed=directed)
    assert topology.triangles(loops).triangles.eq(0).all()
    assert topology.clustering(loops).clustering.eq(0).all()
    assert topology.k_core(loops).core_number.eq(0).all()
    assert topology.density(loops) == 0
    assert values(topology.components_strong(loops), "component") == {2: 1, 1: 0, "1": 2, "z": 3}
    assert topology.triangles(loops).attrs["self_loops_excluded"] == 2


def test_strong_chain_of_ten_thousand_nodes_uses_iterative_traversal_and_canonical_ids():
    nodes = list(range(10000))
    edges = [(i, i + 1, 1) for i in range(9999)]
    graph = make(edges, nodes=nodes, directed=True, batch_rows=1024)
    result = topology.components_strong(graph)
    assert result.component.tolist() == nodes
    assert result.attrs["component_count"] == 10000
    closed = make([*edges, (9999, 0, 1)], nodes=reversed(nodes), directed=True, batch_rows=1024)
    assert topology.components_strong(closed).component.eq(0).all()


def test_results_and_component_ids_are_invariant_to_input_order_types_duplicates_and_chunks():
    nodes = [7, "7", "isolated", -2]
    edges = [(7, "7", 1), ("7", 7, 3), ("7", -2, 2), (-2, -2, 5), (7, "7", 4)]
    a = make(edges, nodes=nodes, directed=True, batch_rows=1)
    b = make(list(reversed(edges)), nodes=reversed(nodes), directed=True, batch_rows=4)
    for operation, columns in [(topology.components_strong, ["component"]), (topology.k_core, ["core_number"]),
                               (topology.clustering, ["clustering", "triangles"])]:
        left, right = operation(a), operation(b)
        for column in columns:
            assert values(left, column) == values(right, column)
    assert topology.triangles(a).attrs["planned_work"] == topology.triangles(b).attrs["planned_work"]
    assert values(topology.components_strong(a), "component") == {7: 1, "7": 1, "isolated": 2, -2: 0}


def test_extreme_reciprocal_weights_do_not_overflow_binary_topology_projection():
    graph = make([(a, b, 1e308) for a in range(3) for b in range(3) if a != b],
                 nodes=range(3), directed=True)
    assert topology.triangles(graph).triangles.tolist() == [8, 8, 8]
    assert topology.clustering(graph).clustering.tolist() == [1.0, 1.0, 1.0]
    assert topology.k_core(graph).core_number.tolist() == [2, 2, 2]
    assert topology.density(graph) == topology.transitivity(graph) == 1


def test_triangle_budget_rejects_before_forward_dictionary_workspace_is_admitted(monkeypatch):
    graph = make([(a, b, 1) for a, b in combinations(range(30), 2)], nodes=range(30))
    calls = []
    original = graph._guard
    monkeypatch.setattr(graph, "_guard", lambda workspace: (calls.append(workspace), original(workspace))[-1])
    with pytest.raises(AnalysisError) as error:
        topology.triangles(graph, max_work=1)
    assert error.value.code == "network_work_budget"
    assert len(calls) == 2, "Only initial topology and CSR guards occur before work rejection"
    accepted = topology.triangles(graph)
    work = accepted.attrs["planned_work"]
    assert work > 1 and accepted.attrs["actual_work"] == work
    assert topology.triangles(graph, max_work=work).attrs["total_triangles"] == math.comb(30, 3)
    with pytest.raises(AnalysisError) as error:
        topology.clustering(graph, max_work=work - 1)
    assert error.value.code == "network_work_budget"
    with pytest.raises(AnalysisError):
        topology.topology_summary(graph, max_work=work - 1)


@pytest.mark.parametrize("value", [0, -1, True, 2.5, math.inf, "100"])
def test_max_work_requires_explicit_positive_integer(value):
    graph = make(nodes=["isolate"])
    with pytest.raises(AnalysisError) as error:
        topology.triangles(graph, max_work=value)
    assert error.value.code == "network_invalid_option"


def test_memory_guard_precedes_projection_and_sparse_algorithms_never_create_dense_matrix(monkeypatch):
    graph = make([(0, 1, 1), (1, 2, 1), (2, 0, 1)], nodes=range(100))
    original_limit = graph._budget.limit
    graph._budget.limit = graph._base_bytes + 4096
    with pytest.raises(AnalysisError) as error:
        topology.k_core(graph)
    assert error.value.code == "network_memory_budget"
    graph._budget.limit = original_limit
    monkeypatch.setattr(torch.Tensor, "to_dense", lambda *args, **kwargs: pytest.fail("Topology cannot densify adjacency"))
    original_zeros = torch.zeros
    def bounded_zeros(size, *args, **kwargs):
        assert not isinstance(size, (tuple, list)) or len(size) <= 1, "No V-by-V numeric workspace"
        return original_zeros(size, *args, **kwargs)
    monkeypatch.setattr(torch, "zeros", bounded_zeros)
    before = graph._edges.indices().clone(), graph._edges.values().clone()
    for operation in (topology.triangles, topology.clustering, topology.k_core, topology.components_strong,
                      topology.topology_summary):
        operation(graph)
    assert torch.equal(graph._edges.indices(), before[0]) and torch.equal(graph._edges.values(), before[1])


def test_large_high_degree_star_needs_no_quadratic_neighbor_pair_work():
    graph = make([(0, node, 1) for node in range(1, 10001)], nodes=range(10001), batch_rows=1024)
    result = topology.triangles(graph, max_work=1)
    assert result.triangles.eq(0).all()
    assert result.attrs["planned_work"] == 0
    assert topology.k_core(graph).core_number.eq(1).all()


@pytest.mark.parametrize("directed", [False, True])
def test_density_uses_sparse_edge_counters_without_vertex_projection_workspace(monkeypatch, directed):
    graph = make([(0, 1, 1), (1, 2, 1), (2, 2, 1)], nodes=range(100), directed=directed)
    graph._budget.limit = graph._base_bytes + 8192
    monkeypatch.setattr(topology, "csr", lambda *args, **kwargs: pytest.fail("Density must not build CSR"))
    assert topology.density(graph) == pytest.approx((2 if directed else 4) / (100 * 99))
