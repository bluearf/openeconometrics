"""Independent scalar closure/deletion/subset oracles for sparse path methods."""
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import math
import random

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon import _network_paths as paths
from openecon.analysis_contracts import AnalysisError
from openecon.frame import DataFrame


def graph(edges=(), *, nodes=(), directed=False, weighted=True, **options):
    return oe.network([{"source": a, "target": b, "weight": w} for a, b, w in edges],
                      nodes=nodes, directed=directed,
                      weight="weight" if weighted else None, **options)


def oracle(edges, labels, directed, weighted):
    """Exact rational Floyd closure, independent of indexed shortest paths."""
    n, index = len(labels), {label: i for i, label in enumerate(labels)}
    weights = {}
    for a, b, w in edges:
        if not w:
            continue
        i, j = index[a], index[b]
        if not directed:
            i, j = min(i, j), max(i, j)
        weights[i, j] = weights.get((i, j), Fraction()) + Fraction(w)
    distance = [[Fraction() if i == j else math.inf for j in range(n)] for i in range(n)]
    for (i, j), w in weights.items():
        distance[i][j] = min(distance[i][j], w if weighted else Fraction(1))
        if not directed:
            distance[j][i] = min(distance[j][i], w if weighted else Fraction(1))
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("direction", ["out", "in"])
def test_selected_distances_eccentricities_summary_and_routes_match_fraction_closure(directed, weighted, direction):
    rng = random.Random(12139)
    for _ in range(10):
        labels = [0, 1, "1", "a", "isolated"]
        edges = [(a, b, rng.choice([1, 2, 4])) for a in labels[:-1] for b in labels[:-1]
                 if rng.random() < .35]
        if edges:
            edges.append(edges[0])
        network = graph(edges, nodes=labels, directed=directed, weighted=weighted)
        reference = oracle(edges, labels, directed, weighted)
        if direction == "in" and directed:
            reference = list(zip(*reference))
        result = paths.distances(network, direction=direction)
        assert isinstance(result, DataFrame)
        assert result.attrs["exact"] and result.attrs["pair_count"] == 25
        assert result.source.tolist() == [s for s in labels for _ in labels]
        assert result.target.tolist() == labels * 5
        assert result.distance.tolist() == pytest.approx([float(value) for row in reference for value in row])
        for disconnected in ["infinite", "reachable"]:
            expected_ecc = [max(row) if disconnected == "infinite" else
                            max(value for value in row if value != math.inf) for row in reference]
            eccentric = paths.eccentricity(network, direction=direction, disconnected=disconnected)
            assert eccentric.eccentricity.tolist() == pytest.approx([float(value) for value in expected_ecc])
            finite = [value for i, row in enumerate(reference) for j, value in enumerate(row)
                      if i != j and value != math.inf]
            average = math.inf if disconnected == "infinite" and len(finite) < 20 else (
                sum(finite, Fraction()) / len(finite) if finite else Fraction())
            efficiency = sum((1 / value for value in finite), Fraction()) / 20
            summary = paths.distance_summary(network, direction=direction, disconnected=disconnected)
            row = summary.iloc[0]
            assert row.average_distance == pytest.approx(float(average))
            assert row.global_efficiency == pytest.approx(float(efficiency))
            assert row.diameter == pytest.approx(float(max(expected_ecc)))
            assert row.radius == pytest.approx(float(min(expected_ecc)))
            assert row.reachable_pairs == len(finite)
            assert row.unreachable_pairs == 20 - len(finite)
            assert summary.attrs["ordered_pairs"] and summary.attrs["self_pairs_excluded"]
        if direction == "out":
            for i, source in enumerate(labels):
                for j, target in enumerate(labels):
                    route = paths.shortest_path(network, source, target)
                    assert route.attrs["distance"] == float(reference[i][j])
                    assert route.attrs["reachable"] == (reference[i][j] != math.inf)
                    if reference[i][j] == math.inf:
                        assert route.empty
                    else:
                        assert route.node.iloc[0] == source and route.node.iloc[-1] == target
                        assert route.distance.iloc[0] == 0
                        assert route.distance.iloc[-1] == float(reference[i][j])
                        assert route.step.tolist() == list(range(len(route)))
                        assert len(route.node) == len(set(route.node))
                        for a, b, previous, following in zip(route.node, route.node.iloc[1:],
                                                              route.distance, route.distance.iloc[1:]):
                            edge_cost = sum((Fraction(w) for s, t, w in edges
                                             if (s == a and t == b) or (not directed and s == b and t == a)),
                                            Fraction()) if weighted else Fraction(1)
                            assert following - previous == float(edge_cost)


def test_route_ties_follow_snapshot_indices_and_duplicate_records_have_explicit_cost_semantics():
    edges = [("a", "c", 1), ("a", "b", 1), ("b", "d", 1), ("c", "d", 1)]
    network = graph(edges, nodes=["a", "c", "b", "d"])
    assert paths.shortest_path(network, "a", "d").node.tolist() == ["a", "c", "d"]
    edges = [(0, 1, 1), (0, 1, 1), (1, 2, 1), (0, 2, 2)]
    weighted = paths.shortest_path(graph(edges, directed=True), 0, 2)
    unweighted = paths.shortest_path(graph(edges, directed=True, weighted=False), 0, 2)
    assert weighted.node.tolist() == unweighted.node.tolist() == [0, 2]
    assert weighted.distance.tolist() == [0, 2]
    assert unweighted.distance.tolist() == [0, 1]
    assert weighted.attrs["weight_semantics"] == "aggregate edge cost"
    assert unweighted.attrs["weight_semantics"] == "unique-edge hops"


@pytest.mark.parametrize("nodes", [[], ["one"], [0, 1]])
def test_empty_isolated_graph_distance_conventions(nodes):
    network = graph(nodes=nodes)
    result = paths.distances(network)
    assert len(result) == len(nodes)**2
    summary = paths.distance_summary(network)
    assert summary.global_efficiency.iloc[0] == 0
    assert summary.total_pairs.iloc[0] == len(nodes) * (len(nodes) - 1)
    if len(nodes) > 1:
        assert summary.average_distance.iloc[0] == summary.diameter.iloc[0] == math.inf
        assert summary.radius.iloc[0] == math.inf
    else:
        assert summary.average_distance.iloc[0] == summary.diameter.iloc[0] == summary.radius.iloc[0] == 0
    reachable = paths.distance_summary(network, disconnected="reachable")
    assert reachable.average_distance.iloc[0] == reachable.diameter.iloc[0] == reachable.radius.iloc[0] == 0


def test_selected_inputs_preserve_user_order_typed_identity_and_empty_selection():
    network = graph([(1, "1", 2), ("1", 3, 5)], nodes=[3, "1", 1], directed=True)
    result = paths.distances(network, sources=[1, "1"], targets=[3, 1])
    assert result.source.tolist() == [1, 1, "1", "1"]
    assert result.target.tolist() == [3, 1, 3, 1]
    assert result.distance.tolist() == [7, 0, 5, math.inf]
    empty = paths.distances(network, sources=[], targets=[1], max_pairs=0)
    assert empty.empty and empty.distance.dtype == "float64"
    assert list(empty.columns) == ["source", "target", "distance"]


@pytest.mark.parametrize("options", [
    {"sources": "a"}, {"targets": {1: 2}}, {"sources": [1, 1]}, {"targets": [True]},
    {"sources": [5]}, {"direction": "both"}, {"max_pairs": -1}, {"max_pairs": True},
    {"max_work": False}, {"max_work": 0},
])
def test_invalid_pair_options(options):
    network = graph([(1, 2, 1)])
    with pytest.raises(AnalysisError):
        paths.distances(network, **options)


@pytest.mark.parametrize("function,options", [
    (paths.eccentricity, {"disconnected": "ignore"}),
    (paths.distance_summary, {"disconnected": None}),
    (paths.eccentricity, {"direction": True}),
    (paths.shortest_path, {"source": 9, "target": 1}),
    (paths.shortest_path, {"source": 1, "target": 9}),
])
def test_invalid_distance_options(function, options):
    with pytest.raises(AnalysisError):
        function(graph([(1, 2, 1)]), **options)


def test_distance_output_pair_and_work_guards_precede_csr_or_path_allocation(monkeypatch):
    network = graph(nodes=range(1100), max_memory_mb=8)
    monkeypatch.setattr(paths, "csr", lambda *a, **k: pytest.fail("CSR ran before pair/work admission"))
    with pytest.raises(AnalysisError) as error:
        paths.distances(network)
    assert error.value.code == "network_pair_budget"
    for operation in [paths.eccentricity, paths.distance_summary]:
        with pytest.raises(AnalysisError) as error:
            operation(network, max_work=1)
        assert error.value.code == "network_work_budget"


@pytest.mark.parametrize("operation,args", [
    (paths.distances, {}), (paths.eccentricity, {}), (paths.distance_summary, {}),
    (paths.shortest_path, {"source": 1, "target": 2}),
    (paths.bridges, {}), (paths.articulation_points, {}), (paths.minimum_spanning_forest, {}),
])
def test_memory_guards_precede_csr_sort_and_output_allocations(monkeypatch, operation, args):
    network = graph([(1, 2, 1)])
    network._budget.limit = network._base_bytes + 4096
    monkeypatch.setattr(paths, "csr", lambda *a, **k: pytest.fail("CSR ran before memory admission"))
    monkeypatch.setattr(torch, "argsort", lambda *a, **k: pytest.fail("Sort ran before memory admission"))
    with pytest.raises(AnalysisError) as error:
        operation(network, **args)
    assert error.value.code == "network_memory_budget"


def test_average_distances_do_not_overflow_when_mean_is_representable():
    network = graph([(0, 1, 1e308), (0, 2, 1e308), (1, 2, 1e308)])
    summary = paths.distance_summary(network)
    assert summary.average_distance.iloc[0] == 1e308
    assert summary.diameter.iloc[0] == summary.radius.iloc[0] == 1e308
    assert summary.global_efficiency.iloc[0] == pytest.approx(1e-308, abs=0, rel=1e-14)


def test_efficiency_normalizes_before_reciprocal_overflow_and_true_overflow_is_explicit():
    # Two finite contributions diluted by 100*99 pairs have a finite mean,
    # despite either unnormalized reciprocal being above float64 range.
    network = graph([(0, 1, 1e-310)], nodes=range(100))
    summary = paths.distance_summary(network, disconnected="reachable")
    assert summary.global_efficiency.iloc[0] == pytest.approx(2 / 9900 / 1e-310)
    with pytest.raises(AnalysisError) as error:
        paths.distance_summary(graph([(0, 1, 1e-310)]))
    assert error.value.code == "network_precision"


def test_true_distance_overflow_and_lost_positive_cost_are_explicit():
    for network in [graph([(0, 1, 1e308), (1, 2, 1e308)], directed=True),
                    graph([(0, 1, 1e16), (1, 2, 1)], directed=True)]:
        with pytest.raises(AnalysisError) as error:
            paths.shortest_path(network, 0, 2)
        assert error.value.code == "network_precision"


def test_targeted_route_avoids_irrelevant_overflow_and_stops_at_finalized_target():
    network = graph([(0, 1, 1), (0, 2, 1e308), (2, 3, 1e308)],
                    nodes=["unreachable"], directed=True)
    route = paths.shortest_path(network, 0, 1)
    assert route.node.tolist() == [0, 1]
    assert route.attrs["distance"] == 1
    self_route = paths.shortest_path(network, 0, 0)
    assert self_route.node.tolist() == [0] and self_route.attrs["distance"] == 0
    unreachable = paths.shortest_path(network, 0, "unreachable")
    assert unreachable.empty and not unreachable.attrs["reachable"]
    assert unreachable.attrs["distance"] == math.inf
    with pytest.raises(AnalysisError) as error:
        paths.shortest_path(network, 0, 3)
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("weighted", [False, True])
def test_shared_target_option_returns_partial_distances_without_changing_all_source_result(weighted):
    from openecon._network_centrality import _paths
    from openecon._network_sparse import csr

    network = graph([(i, i + 1, 1) for i in range(100)], weighted=weighted)
    topology = csr(network)
    partial, order, counts = _paths(topology, network._index[0], weighted, target=network._index[1])
    assert order == [network._index[0], network._index[1]]
    assert partial[network._index[1]] == 1 and partial[network._index[100]] == math.inf
    assert counts is None
    complete, full_order, _ = _paths(topology, network._index[0], weighted)
    assert complete[network._index[100]] == 100 and len(full_order) == 101
    with pytest.raises(AnalysisError) as error:
        _paths(topology, network._index[0], weighted, target=network._index[1], count_paths=True)
    assert error.value.code == "network_invalid_option"


def test_targeted_route_ignores_overflow_walk_scanned_before_finite_destination():
    network = graph([(0, "branch", 1e308), ("branch", "huge", 1e308),
                     ("huge", "overflow", 1e308), (0, "target", 1e308)], directed=True)
    route = paths.shortest_path(network, 0, "target")
    assert route.node.tolist() == [0, "target"] and route.attrs["distance"] == 1e308


def test_targeted_dijkstra_finalizes_destination_before_irrelevant_equal_distance_precision_loss():
    from openecon._network_centrality import _paths
    from openecon._network_sparse import csr

    network = graph([("s", "branch", 1e16), ("branch", "dead", 1), ("s", "t", 1e16)],
                    nodes=["s", "branch", "dead", "t"], directed=True)
    result = paths.shortest_path(network, "s", "t")
    assert result.node.tolist() == ["s", "t"] and result.attrs["distance"] == 1e16
    partial, order, _ = _paths(csr(network), 0, True, target=3)
    assert order == [0, 3] and partial[3] == 1e16 and partial[2] == math.inf
    with pytest.raises(AnalysisError) as error:
        _paths(csr(network), 0, True)
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("target_cost", [None, 2e16])
def test_targeted_dijkstra_defers_unrelated_lost_cost_for_unreachable_or_larger_direct_target(target_cost):
    records = [("s", "branch", 1e16), ("branch", "dead", 1)]
    if target_cost is not None:
        records.append(("s", "t", target_cost))
    network = graph(records, nodes=["s", "branch", "dead", "t"], directed=True)
    result = paths.shortest_path(network, "s", "t")
    if target_cost is None:
        assert result.empty and not result.attrs["reachable"] and result.attrs["distance"] == math.inf
    else:
        assert result.node.tolist() == ["s", "t"] and result.attrs["distance"] == target_cost
    with pytest.raises(AnalysisError) as error:
        paths.distances(network, sources=["s"], targets=["t"])
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("direct_alternative", [False, True])
def test_target_relevant_ambiguous_branch_is_refused_without_approximate_relaxation(direct_alternative):
    records = [("s", "branch", 1e16), ("branch", "dead", 1), ("dead", "t", 10)]
    if direct_alternative:
        records.append(("s", "t", 2e16))
    network = graph(records, nodes=["s", "branch", "dead", "t"], directed=True)
    with pytest.raises(AnalysisError) as error:
        paths.shortest_path(network, "s", "t")
    assert error.value.code == "network_precision"
    assert "destination-relevant" in str(error.value)


@pytest.mark.parametrize("predecessors,target", [({}, 1), ((), 1), ([], 1), ([0], 1),
                                                  ([0, 0, 0], 1), ([0, 0], None)])
def test_private_predecessor_option_rejects_wrong_type_length_and_missing_target(predecessors, target):
    from openecon._network_centrality import _paths
    from openecon._network_sparse import csr

    with pytest.raises(AnalysisError) as error:
        _paths(csr(graph([(0, 1, 1)])), 0, True, target=target, predecessors=predecessors)
    assert error.value.code == "network_invalid_option"


@pytest.mark.parametrize("weighted", [False, True])
def test_route_reconstruction_needs_only_one_outgoing_csr_and_monotone_predecessors(monkeypatch, weighted):
    from openecon._network_centrality import _paths
    from openecon._network_sparse import csr

    network = graph([(0, 2, 1), (0, 1, 1), (1, 3, 1), (2, 3, 1), (3, 4, 1)],
                    nodes=[0, 1, 2, 3, 4], directed=True, weighted=weighted)
    calls, original = [], paths.csr
    def observed(*args, **options):
        calls.append(options)
        return original(*args, **options)
    monkeypatch.setattr(paths, "csr", observed)
    result = paths.shortest_path(network, 0, 4)
    assert result.node.tolist() == [0, 1, 3, 4]
    assert calls == [{"loops": False}]
    predecessors = [99] * 5
    distance, _, _ = _paths(csr(network), 0, weighted, target=4, predecessors=predecessors)
    assert predecessors == [0, 0, 0, 1, 3]
    assert all(distance[parent] < distance[node] for node, parent in enumerate(predecessors) if node)


def test_path_setup_budget_matches_forward_zero_sort_and_two_stable_reverse_sorts():
    directed = graph([(0, 1, 1), (1, 2, 1)], directed=True)
    out = paths.distances(directed, sources=[0], targets=[2])
    incoming = paths.distances(directed, sources=[0], targets=[2], direction="in")
    arcs = directed._arcs._nnz()
    assert incoming.attrs["planned_work"] - out.attrs["planned_work"] == 2 * arcs * arcs.bit_length()


def component_count(nodes, edges):
    """Independent deletion oracle using Python sets only in tiny fixtures."""
    unseen = set(nodes)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            a = stack.pop()
            for u, v in edges:
                if u == a and v in unseen:
                    unseen.remove(v)
                    stack.append(v)
                elif v == a and u in unseen:
                    unseen.remove(u)
                    stack.append(u)
    return count


@pytest.mark.parametrize("directed", [False, True])
def test_iterative_cuts_match_independent_vertex_and_connection_deletion(directed):
    rng = random.Random(2244)
    nodes = list(range(7))
    for _ in range(35):
        records = [(a, b, rng.choice([.1, 4, 1e100])) for a in nodes for b in nodes
                   if rng.random() < .2]
        # Avoid duplicate aggregate overflow in the graph import; topology
        # nevertheless includes reciprocal, loops and ordinary duplicates.
        records += [(0, 1, .1)] * 2
        network = graph(records, nodes=nodes, directed=directed)
        connections = {tuple(sorted((a, b))) for a, b, _ in records if a != b}
        initial = component_count(nodes, connections)
        expected_bridges = {pair for pair in connections
                            if component_count(nodes, connections - {pair}) > initial}
        expected_points = {node: component_count([n for n in nodes if n != node],
                                                 {pair for pair in connections if node not in pair}) > initial
                           for node in nodes}
        actual = paths.bridges(network)
        assert set(zip(actual.source, actual.target)) == expected_bridges
        points = paths.articulation_points(network)
        assert dict(zip(points.node, points.articulation)) == expected_points
        assert actual.attrs["connectivity"] == ("weak" if directed else "undirected")
        assert points.attrs["exact"]


def test_graph_cuts_ignore_loops_and_aggregate_strengths_with_typed_ids():
    network = graph([(1, "1", 1), ("1", 1, 2), ("1", "tail", 9),
                     ("1", "1", 100)], nodes=["isolate"], directed=True)
    assert paths.bridges(network).to_dict("records") == [{"source": 1, "target": "1"},
                                                         {"source": "1", "target": "tail"}]
    assert dict(zip(paths.articulation_points(network).node,
                    paths.articulation_points(network).articulation)) == {
                        "isolate": False, 1: False, "1": True, "tail": False}


def test_long_chain_iterative_cuts_have_no_python_recursion_limit():
    network = graph([(i, i + 1, 1) for i in range(9999)], nodes=[10000], max_memory_mb=64)
    bridges = paths.bridges(network)
    assert len(bridges) == 9999
    points = paths.articulation_points(network)
    assert points.articulation.sum() == 9998
    assert not points.loc[points.node.isin([0, 9999, 10000]), "articulation"].any()


def forest_oracle(edges, n):
    """Exhaustive acyclic spanning-forest subsets, independent of Kruskal."""
    aggregated = {}
    for a, b, weight in edges:
        if a == b or not weight:
            continue
        key = min(a, b), max(a, b)
        aggregated[key] = aggregated.get(key, Fraction()) + Fraction(weight)
    count = component_count(range(n), aggregated)
    candidates = list(aggregated)
    needed = n - count
    totals = []
    for chosen in combinations(candidates, needed):
        if component_count(range(n), chosen) == count:
            totals.append(sum((aggregated[pair] for pair in chosen), Fraction()))
    return count, needed, min(totals, default=Fraction())


def test_kruskal_matches_exhaustive_spanning_forest_subsets_and_preserves_snapshot():
    rng = random.Random(23435)
    for _ in range(30):
        edges = [(a, b, rng.choice([1, 2, 3, 5])) for a in range(5) for b in range(a, 5)
                 if rng.random() < .4]
        if edges:
            edges.append(edges[0])
        network = graph(edges, nodes=range(6))
        before = (network.metadata, network._edges.indices().clone(), network._edges.values().clone())
        expected_components, expected_edges, expected_cost = forest_oracle(edges, 6)
        forest = paths.minimum_spanning_forest(network)
        assert isinstance(forest, oe.Network)
        assert forest.node_count == network.node_count == 6
        assert forest.edge_count == expected_edges
        assert forest.metadata["component_count"] == expected_components
        assert forest.metadata["total_cost"] == float(expected_cost)
        assert len(set(forest.components().component)) == expected_components
        assert torch.equal(network._edges.indices(), before[1])
        assert torch.equal(network._edges.values(), before[2])
        assert network.metadata == before[0]
        second = paths.minimum_spanning_forest(network)
        assert torch.equal(forest._edges.indices(), second._edges.indices())
        assert torch.equal(forest._edges.values(), second._edges.values())


def test_unweighted_forest_uses_unique_edge_cost_and_retains_strengths_and_scalar_attributes():
    network = graph([(1, "1", 1), (1, "1", 1), ("1", "tail", 1), (1, "tail", 1),
                     ("tail", "tail", 1)], nodes=["isolate"], weighted=False)
    network = network.with_attributes(nodes={1: {"type": "int"}, "1": {"type": "string"},
                                             "isolate": {"label": "Alone"}},
                                      edges={(1, "1"): {"title": "duplicate aggregate"},
                                             ("tail", "tail"): {"title": "loop"}},
                                      graph_attributes={"title": "Native"})
    before = deepcopy((network.metadata, network.node_attributes, network.edge_attributes,
                       network.graph_attributes))
    forest = paths.minimum_spanning_forest(network)
    assert not forest.weighted and forest.node_count == 4
    assert forest.edge_count == 2 and forest.metadata["component_count"] == 2
    assert forest.metadata["total_cost"] == 2
    assert forest.metadata["cost_semantics"] == "unique-edge unit cost"
    assert forest._edges.values().tolist() == [2, 1]
    assert forest.node_attributes == network.node_attributes
    assert forest.graph_attributes == network.graph_attributes
    assert forest.edge_attributes == {(1, "1"): {"title": "duplicate aggregate"}}
    assert (network.metadata, network.node_attributes, network.edge_attributes,
            network.graph_attributes) == before
    altered = forest.node_attributes
    altered[1]["type"] = "changed"
    assert forest.node_attributes[1]["type"] == network.node_attributes[1]["type"] == "int"


@pytest.mark.parametrize("nodes", [[], [1], [0, 1, 2]])
def test_empty_forest_keeps_every_isolate(nodes):
    forest = paths.minimum_spanning_forest(graph(nodes=nodes))
    assert forest.node_count == len(nodes) and forest.edge_count == 0
    assert forest.metadata["component_count"] == len(nodes)
    assert forest.metadata["total_cost"] == 0


def test_kruskal_rejects_directed_graph_before_any_sort_and_reserves_work(monkeypatch):
    monkeypatch.setattr(torch, "argsort", lambda *a, **k: pytest.fail("sort ran before option/work admission"))
    with pytest.raises(AnalysisError) as error:
        paths.minimum_spanning_forest(graph([(1, 2, 1)], directed=True))
    assert error.value.code == "network_invalid_option"
    network = graph([(1, 2, 1)])
    for operation in [paths.minimum_spanning_forest, paths.bridges, paths.articulation_points]:
        with pytest.raises(AnalysisError) as error:
            operation(network, max_work=1)
        assert error.value.code == "network_work_budget"


def test_minimum_forest_total_cost_overflow_is_explicit():
    with pytest.raises(AnalysisError) as error:
        paths.minimum_spanning_forest(graph([(0, 1, 1e308), (1, 2, 1e308)]))
    assert error.value.code == "network_precision"


def test_actual_parquet_batch_import_has_full_graph_routes_cuts_and_forest(tmp_path):
    path = tmp_path / "physical-network.parquet"
    edges = [(0, 1, 2), (0, 1, 1), (1, 2, 4), (2, 0, 1),
             (2, 3, 2), (3, 4, 3), (4, 4, 9), (8, 8, 0)]
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_parquet(path)
    dataset = oe.scan(path)
    network = oe.network(dataset, weight="weight", batch_rows=2, nodes=["isolate"])
    assert network.metadata["input_rows"] == 8
    assert network.metadata["actual_peak_batch_rows"] == 2
    assert paths.shortest_path(network, 0, 4).node.tolist() == [0, 2, 3, 4]
    assert paths.shortest_path(network, 0, 4).distance.tolist() == [0, 1, 3, 6]
    assert paths.bridges(network).to_dict("records") == [{"source": 2, "target": 3},
                                                         {"source": 3, "target": 4}]
    points = paths.articulation_points(network)
    assert points.loc[points.articulation, "node"].tolist() == [2, 3]
    forest = paths.minimum_spanning_forest(network)
    assert forest.edge_count == 4 and forest.node_count == 7
    assert forest.metadata["component_count"] == 3
    assert forest.metadata["total_cost"] == 9
    assert paths.distance_summary(network, disconnected="reachable").reachable_pairs.iloc[0] == 20
    assert "\\begin{tabular}" in str(paths.distances(network, sources=[0], targets=[4]).to_latex())
    dataset.assert_unchanged()
