"""Independent exhaustive cuts and cardinality oracles, not another solver."""
from fractions import Fraction
import itertools
import math
import random

import pytest
import torch

from openecon import _network_flow as flow
from openecon.analysis_contracts import AnalysisError
from openecon.networks import network


def _graph(edges, *, directed=True, nodes=(), weighted=True, **options):
    return network([dict(source=u, target=v, capacity=c) for u, v, c in edges],
        weight="capacity" if weighted else None, directed=directed, nodes=nodes, **options)


def _all_cuts(edges, nodes, source, target, directed):
    candidates = [node for node in nodes if node not in (source, target)]
    optimum = None
    for bits in itertools.product((False, True), repeat=len(candidates)):
        left = {source, *(node for node, yes in zip(candidates, bits) if yes)}
        capacity = sum((Fraction(c) for u, v, c in edges
            if u in left and v not in left), Fraction()) if directed else sum(
            (Fraction(c) for u, v, c in edges if (u in left) != (v in left)), Fraction())
        if optimum is None or capacity < optimum:
            optimum = capacity
    return float(optimum)


def _assert_certificate(result, graph, expected):
    assert result["value"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result["metadata"]["certified"]
    assert result["metadata"]["exact"]
    assert not result["metadata"]["approximate"]
    assert set(result["source_partition"]) | set(result["target_partition"]) == set(graph._labels)
    assert not set(result["source_partition"]) & set(result["target_partition"])
    assert result["source"] in result["source_partition"]
    assert result["target"] in result["target_partition"]
    balances = dict.fromkeys(graph._labels, 0.)
    for row in result["flows"].itertuples(index=False):
        assert abs(row.flow) <= row.capacity + 1e-12
        assert not graph.directed or row.flow >= -1e-12
        balances[row.source] += row.flow
        balances[row.target] -= row.flow
        if row.source == row.target:
            assert row.flow == 0
    for node, balance in balances.items():
        assert balance == pytest.approx(expected if node == result["source"] else
            -expected if node == result["target"] else 0., abs=1e-12, rel=1e-12)
    assert result["cut_edges"].capacity.sum() == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert result["metadata"]["cut_capacity"] == pytest.approx(expected)
    assert result["metadata"]["work_used"] <= result["metadata"]["max_work"]
    assert "tabular" in result["flows"].to_latex(index=False)
    assert "tabular" in result["cut_edges"].to_latex(index=False)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(24))
def test_random_small_graph_matches_every_exact_fraction_cut(directed, seed):
    rng = random.Random(seed)
    nodes = [0, 1, "1", 2, "isolate"]
    edges = [(u, v, rng.randrange(1, 13) / 8)
        for u, v in itertools.product(nodes[:4], repeat=2) if rng.random() < .38]
    if edges:
        edges += [edges[0], edges[0]]
    graph = _graph(edges, nodes=nodes, directed=directed)
    for source, target in [(0, 2), (2, 0), (0, "isolate")]:
        expected = _all_cuts(edges, nodes, source, target, directed)
        _assert_certificate(flow.max_flow(graph, source, target), graph, expected)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(12))
def test_decimal_float_capacities_match_exact_fraction_cut_oracle(directed, seed):
    rng = random.Random(seed)
    nodes = list(range(6))
    edges = [(u, v, rng.random() * 1000 + .1)
        for u, v in itertools.product(nodes, repeat=2) if rng.random() < .4]
    graph = _graph(edges, nodes=nodes, directed=directed)
    expected = _all_cuts(edges, nodes, 0, 5, directed)
    _assert_certificate(flow.max_flow(graph, 0, 5), graph, expected)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("bits", range(64))
def test_every_three_node_binary_topology_has_correct_flow(directed, bits):
    arcs = list(itertools.permutations(range(3), 2))
    edges = [(u, v, 1) for i, (u, v) in enumerate(arcs) if bits & (1 << i)]
    graph = _graph(edges, nodes=range(3), directed=directed)
    expected = _all_cuts(edges, list(range(3)), 0, 2, directed)
    _assert_certificate(flow.min_cut(graph, 0, 2), graph, expected)
    assert flow.min_cut(graph, 0, 2)["metadata"]["kind"] == "network_min_cut"


def test_antiparallel_edges_duplicates_loops_and_typed_nodes():
    edges = [(1, "1", 3), (1, "1", 2), ("1", 1, 7), ("1", "t", 4),
        (1, "t", 1), (1, 1, 1000), ("t", "t", 5)]
    graph = _graph(edges, nodes=["isolate"])
    before = graph._edges.clone(), graph._arcs.clone(), graph.metadata
    result = flow.max_flow(graph, 1, "t")
    _assert_certificate(result, graph, 5)
    assert result["flows"].source.dtype == object
    assert result["flows"].target.dtype == object
    assert len(result["flows"]) == graph.edge_count
    assert torch.equal(graph._edges.indices(), before[0].indices())
    assert torch.equal(graph._edges.values(), before[0].values())
    assert torch.equal(graph._arcs.values(), before[1].values())
    assert graph.metadata == before[2]
    assert result["value"] == flow.min_cut(graph, 1, "t")["value"]


def test_flow_mapping_summary_latex_and_repr_are_bounded(tmp_path):
    graph = _graph([(i, i + 1, 1) for i in range(100)])
    result = flow.max_flow(graph, 0, 100)
    assert isinstance(result, dict) and isinstance(result, flow.NetworkFlowResult)
    summary = result.summary()
    assert len(summary) == 8 and list(summary.columns) == ["Metric", "Value"]
    assert summary.attrs["kind"] == "network_max_flow_summary"
    assert summary.loc[summary.Metric == "Source partition nodes", "Value"].item() == len(result["source_partition"])
    latex = result.to_latex(caption="Flow & cut")
    assert "tabular" in latex and "Flow value" in latex and "Flow \\& cut" in latex
    assert len(latex) < 2000 and len(repr(result)) < 160
    destination = tmp_path / "flow.tex"
    assert result.to_latex(destination) is None
    assert destination.read_text() == result.to_latex()
    assert len(result["flows"]) == 100
    cut = flow.min_cut(graph, 0, 100)
    assert isinstance(cut, flow.NetworkFlowResult)
    assert cut.summary().attrs["kind"] == "network_min_cut_summary"


def test_unweighted_parallel_records_are_aggregate_capacity():
    graph = _graph([(0, 1, 99), (0, 1, 99), (1, 2, 99), (1, 2, 99)], weighted=False)
    _assert_certificate(flow.max_flow(graph, 0, 2), graph, 2)
    assert not graph.weighted


def test_undirected_flow_is_signed_and_shared_not_two_independent_capacities():
    graph = _graph([("a", "b", 2), ("b", "c", 2)], directed=False)
    result = flow.max_flow(graph, "c", "a")
    _assert_certificate(result, graph, 2)
    assert result["flows"].flow.tolist() == [-2, -2]


@pytest.mark.parametrize("capacity", [1e-300, math.ldexp(1., -1074), 1e300, math.ldexp(1., 1023)])
def test_power_of_two_scaling_preserves_representable_capacity(capacity):
    graph = _graph([(0, 1, capacity), (1, 2, capacity)])
    result = flow.max_flow(graph, 0, 2)
    assert result["value"] == capacity
    assert result["metadata"]["cut_capacity"] == capacity
    assert result["flows"].flow.tolist() == [capacity, capacity]


def test_irrelevant_huge_loops_do_not_destroy_tiny_flow_scaling():
    graph = _graph([(0, 0, 1e308), (0, 1, 1e-300)])
    result = flow.max_flow(graph, 0, 1)
    assert result["value"] == 1e-300
    assert result["flows"].flow.tolist() == [0., 1e-300]


def test_float64_total_overflow_is_explicit():
    graph = _graph([(0, 1, 1e308), (1, 3, 1e308), (0, 2, 1e308), (2, 3, 1e308)])
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 3)
    assert error.value.code == "network_precision"


def test_capacity_rescaling_underflow_is_explicit():
    graph = _graph([(0, 1, 1e308), (1, 2, 1e-300)])
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 2)
    assert error.value.code == "network_precision"


def test_capacity_rescaling_to_inexact_positive_subnormal_is_explicit():
    graph = _graph([(0, 1, 1e308), (1, 2, .1)])
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 2)
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("directed", [False, True])
def test_lost_positive_residual_update_is_explicit(directed):
    graph = _graph([(0, 1, 1e16), (1, 2, 1)], directed=directed)
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 2)
    assert error.value.code == "network_precision"


def test_value_tiny_disjoint_path_can_round_without_losing_edge_flow():
    graph = _graph([(0, 1, 1.), (1, 3, 1.), (0, 2, 1e-20), (2, 3, 1e-20)])
    result = flow.max_flow(graph, 0, 3)
    assert result["value"] == 1.
    assert sorted(result["flows"].flow.tolist()) == [1e-20, 1e-20, 1., 1.]


def test_iterative_dinic_large_chain_has_no_python_recursion():
    n = 20_000
    graph = _graph(((i, i + 1, 3) for i in range(n - 1)))
    result = flow.max_flow(graph, 0, n - 1)
    assert result["value"] == 3
    assert result["metadata"]["augmentations"] == 1
    assert result["metadata"]["phases"] == 1
    assert len(result["flows"]) == n - 1


@pytest.mark.parametrize("source,target", [(None, 1), (True, 1), (1.0, 1), ("absent", 1), (0, 0)])
def test_invalid_endpoints_are_explicit(source, target):
    graph = _graph([(0, 1, 1)])
    with pytest.raises(AnalysisError):
        flow.max_flow(graph, source, target)


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, None, 2**64])
def test_invalid_flow_work_budgets(budget):
    graph = _graph([(0, 1, 1)])
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 1, max_work=budget)
    assert error.value.code == "network_invalid_option"


def test_work_preflight_before_residual_tensor_allocations(monkeypatch):
    graph = _graph([(0, 1, 1)])
    def denied(*args, **kwargs):
        raise AssertionError("allocated before work admission")
    monkeypatch.setattr(torch, "empty", denied)
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 1, max_work=1)
    assert error.value.code == "network_work_budget"


def test_flow_actual_traversal_budget_fails_without_partial_result():
    graph = _graph([(0, 1, 1), (1, 2, 1)])
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 2, max_work=37)
    assert error.value.code == "network_work_budget"


def test_memory_preflight_before_residual_tensor_allocations(monkeypatch):
    graph = _graph([(0, 1, 1)])
    graph._budget.limit = graph._base_bytes + 4096 + 100
    def denied(*args, **kwargs):
        raise AssertionError("allocated before memory admission")
    monkeypatch.setattr(torch, "empty", denied)
    with pytest.raises(AnalysisError) as error:
        flow.max_flow(graph, 0, 1)
    assert error.value.code == "network_memory_budget"


def _cardinality_oracle(edges, left, right):
    neighbors = {u: {v for a, v, _ in edges if a == u} for u in left}
    def search(at, used):
        if at == len(left):
            return 0
        return max([search(at + 1, used), *(1 + search(at + 1, used | {v})
            for v in neighbors[left[at]] - used)])
    return search(0, set())


def _assert_matching(result, edges, nodes, expected):
    assert len(result) == result.attrs["matched_pairs"] == expected
    assert result.attrs["certified"] and result.attrs["maximum"] and result.attrs["exact"]
    assert len(set(result.source)) == len(result)
    assert len(set(result.target)) == len(result)
    input_edges = {(u, v) for u, v, _ in edges}
    assert all((u, v) in input_edges or (v, u) in input_edges
        for u, v in zip(result.source, result.target))
    cover = set(result.attrs["minimum_vertex_cover"])
    assert len(cover) == expected
    assert all(u in cover or v in cover for u, v, _ in edges)
    assert set(result.attrs["unmatched_nodes"]) == set(nodes) - set(result.source) - set(result.target)
    assert result.attrs["work_used"] <= result.attrs["max_work"]
    assert "tabular" in result.to_latex(index=False)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(30))
def test_matching_matches_independent_exhaustive_cardinality(directed, seed):
    rng = random.Random(seed)
    left, right = [0, 1, "1", "L"], ["R0", "R1", "R2", "R3"]
    nodes = left + right + ["isolate"]
    edges = [(u, v, rng.randint(1, 8)) for u, v in itertools.product(left, right) if rng.random() < .4]
    if edges:
        edges += [edges[0]]
    graph = _graph(edges, directed=directed, nodes=nodes)
    partition = dict.fromkeys(left + ["isolate"], 0) | dict.fromkeys(right, 1)
    result = flow.maximum_matching(graph, partition)
    _assert_matching(result, edges, nodes, _cardinality_oracle(edges, left, right))


@pytest.mark.parametrize("bits", range(512))
def test_matching_every_three_by_three_binary_graph(bits):
    left, right = range(3), range(3, 6)
    edges = [(u, v, 1) for i, (u, v) in enumerate(itertools.product(left, right)) if bits & (1 << i)]
    graph = _graph(edges, nodes=range(6), directed=False)
    actual = flow.maximum_matching(graph, dict.fromkeys(left, 0) | dict.fromkeys(right, 1))
    _assert_matching(actual, edges, list(range(6)), _cardinality_oracle(edges, list(left), list(right)))


def test_matching_weak_directed_projection_ignores_capacity_and_duplicate_rows():
    edges = [("a", "x", 100), ("x", "a", 500), ("b", "x", .01), ("b", "y", 1)]
    graph = _graph(edges)
    result = flow.maximum_matching(graph, {"a": 0, "b": 0, "x": 1, "y": 1})
    _assert_matching(result, edges, graph._labels, 2)
    assert result.attrs["directed_projection"] == "weak binary"


def test_matching_handles_empty_graph_and_isolates():
    empty = flow.maximum_matching(_graph([]))
    assert list(empty.columns) == ["source", "target"] and empty.empty
    assert empty.attrs["matched_pairs"] == 0 and empty.attrs["minimum_vertex_cover"] == ()
    graph = _graph([], nodes=[1, "1", "isolate"])
    result = flow.maximum_matching(graph)
    assert result.empty and result.attrs["unmatched_nodes"] == graph._labels


@pytest.mark.parametrize("edges", [[(0, 0, 1)], [(0, 1, 1), (1, 2, 1), (2, 0, 1)]])
def test_matching_rejects_nonbipartite_graphs(edges):
    with pytest.raises(AnalysisError):
        flow.maximum_matching(_graph(edges))


def test_matching_partition_dataframe_preserves_typed_ids():
    import pandas as pd
    graph = _graph([(1, "x", 1), ("1", "y", 1)])
    partition = pd.DataFrame({"node": pd.Series([1, "1", "x", "y"], dtype=object), "partition": [0, 0, 1, 1]})
    result = flow.maximum_matching(graph, partition)
    assert set(result.source) == {1, "1"}
    assert set(result.target) == {"x", "y"}


def test_matching_iterative_long_augmenting_path():
    # Greedy first phase uses i -> i, leaving the last left vertex unmatched.
    # That vertex has only right0 and must traverse the full alternating chain
    # to reach the final unmatched right vertex. This exceeds recursion limits.
    n = 3000
    edges = [(i, n + i, 1) for i in range(n - 1)] + [(i, n + i + 1, 1) for i in range(n - 1)]
    edges.append((n - 1, n, 1))
    graph = _graph(edges, nodes=range(2 * n), directed=False)
    result = flow.maximum_matching(graph, {i: int(i >= n) for i in range(2 * n)})
    assert len(result) == n
    assert result.attrs["certified"]
    assert result.attrs["phases"] == 2


def test_matching_memory_guard_before_partition_or_tensor_copies(monkeypatch):
    graph = _graph([(0, 1, 1)])
    graph._budget.limit = graph._base_bytes + 4096 + 100
    def denied(*args, **kwargs):
        raise AssertionError("allocated before memory admission")
    monkeypatch.setattr(torch, "empty", denied)
    with pytest.raises(AnalysisError) as error:
        flow.maximum_matching(graph)
    assert error.value.code == "network_memory_budget"


def test_matching_work_guard_before_partition_validation(monkeypatch):
    graph = _graph([(0, 1, 1)])
    with pytest.raises(AnalysisError) as error:
        flow.maximum_matching(graph, max_work=1)
    assert error.value.code == "network_work_budget"


def test_matching_reuses_its_weak_projection_for_partition_validation(monkeypatch):
    import openecon._network_sparse as sparse
    graph = _graph([(0, 2, 1), (1, 2, 1), (1, 3, 1)])
    original, calls = sparse.csr, []
    def observed(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(sparse, "csr", observed)
    result = flow.maximum_matching(graph)
    assert len(result) == 2
    assert calls == [{"loops": False, "undirected": True}]
