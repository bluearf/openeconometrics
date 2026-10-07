"""Exact Fraction exhaustive bipartitions, independent of contractions."""
from fractions import Fraction
import itertools
import math
import random

import pytest
import torch

import openecon._network_cut as cut
from openecon.analysis_contracts import AnalysisError
from openecon.networks import network


def _graph(edges=(), *, nodes=(), weighted=True, directed=False, **options):
    return network([dict(source=u, target=v, w=w) for u, v, w in edges],
        nodes=nodes, weight="w" if weighted else None, directed=directed, **options)


def _oracle(graph):
    n, pairs, weights = graph.node_count, graph._edges.indices(), graph._edges.values()
    choices = []
    for mask in range(1, 2**n - 1):
        if not mask & 1:
            continue
        capacity = sum((Fraction(float(weights[i])) for i in range(graph.edge_count)
            if bool(mask & (1 << int(pairs[0, i]))) != bool(mask & (1 << int(pairs[1, i])))), Fraction())
        choices.append(capacity)
    return min(choices)


def _assert_result(graph, result, expected):
    assert isinstance(result, dict) and isinstance(result, cut.NetworkCutResult)
    assert result["value"] == float(expected)
    metadata = result["metadata"]
    assert Fraction(metadata["exact_capacity_numerator"], metadata["exact_capacity_denominator"]) == expected
    assert metadata["exact_optimum_for_stored_weights"] and metadata["certified"]
    assert metadata["exact"] and not metadata["sampled"]
    assert metadata["work_used"] <= metadata["max_work"]
    a, b = set(result["source_partition"]), set(result["target_partition"])
    assert a and b and a | b == set(graph._labels) and not a & b
    def key(label):
        return (0, label) if isinstance(label, int) else (1, label)
    assert min(graph._labels, key=key) in a
    assert tuple(sorted(a, key=key)) == result["source_partition"]
    assert tuple(sorted(b, key=key)) == result["target_partition"]
    pairs, weights = graph._edges.indices(), graph._edges.values()
    oracle_rows = []
    for i in range(graph.edge_count):
        u, v = graph._labels[int(pairs[0, i])], graph._labels[int(pairs[1, i])]
        if (u in a) != (v in a):
            oracle_rows.append((u, v, float(weights[i])) if key(u) <= key(v) else (v, u, float(weights[i])))
    oracle_rows.sort(key=lambda row: (key(row[0]), key(row[1])))
    actual = list(result["cut_edges"].itertuples(index=False, name=None))
    assert actual == oracle_rows
    assert sum((Fraction(w) for _, _, w in actual), Fraction()) == expected
    assert list(result["cut_edges"].columns) == ["source", "target", "capacity"]
    assert result["cut_edges"].source.dtype == object and result["cut_edges"].target.dtype == object
    assert "tabular" in result.to_latex() and "tabular" in result["cut_edges"].to_latex(index=False)


@pytest.mark.parametrize("mask", range(64))
@pytest.mark.parametrize("weighted", [False, True])
def test_every_four_node_topology_matches_exact_all_partition_oracle(mask, weighted):
    edges = [(u, v, (i + 1) / 8) for i, (u, v) in enumerate(itertools.combinations(range(4), 2)) if mask & (1 << i)]
    if edges:
        edges.extend([edges[0]] * 2)
    graph = _graph(edges, nodes=range(4), weighted=weighted)
    _assert_result(graph, cut.global_min_cut(graph), _oracle(graph))


@pytest.mark.parametrize("seed", range(70))
def test_random_weighted_typed_graphs_match_exact_stored_capacity_oracle(seed):
    rng = random.Random(seed)
    nodes = [1, "1", "b", "a", 2, "isolate"]
    edges = [(u, v, rng.choice([.1, .125, .3, 1., 5., 11.]))
        for u, v in itertools.combinations_with_replacement(nodes, 2) if rng.random() < .55]
    if edges:
        edges += [edges[0]] * 3
    graph = _graph(edges, nodes=nodes)
    _assert_result(graph, cut.global_min_cut(graph), _oracle(graph))


def test_deterministic_ties_ignore_input_order_and_preserve_typed_labels():
    nodes = ["z", "1", 1, 2, "a"]
    edges = [(u, v, 1) for u, v in itertools.combinations(nodes, 2)]
    reference = None
    for seed in range(12):
        rng = random.Random(seed)
        shuffled_nodes, shuffled_edges = nodes.copy(), edges.copy()
        rng.shuffle(shuffled_nodes)
        rng.shuffle(shuffled_edges)
        shuffled_edges = [(v, u, w) if rng.random() < .5 else (u, v, w) for u, v, w in shuffled_edges]
        graph = _graph(shuffled_edges, nodes=shuffled_nodes)
        result = cut.global_min_cut(graph)
        state = (result["source_partition"], result["target_partition"], result["cut_edges"].to_dict("records"))
        if reference is None:
            reference = state
        assert state == reference
        _assert_result(graph, result, Fraction(4))


def test_disconnected_canonical_component_zero_cut_ignores_huge_loops():
    graph = _graph([("a", "b", 1e308), (1, "1", 1e-300), (1, 1, 1e308)], nodes=["isolated", 8])
    result = cut.global_min_cut(graph)
    _assert_result(graph, result, Fraction())
    assert result["source_partition"] == (1, "1")
    assert result["metadata"]["phases"] == result["metadata"]["contractions"] == 0
    assert result["metadata"]["component_count"] == 4
    assert result["cut_edges"].empty


def test_zero_weight_endpoints_retain_isolates_and_choose_zero_cut():
    graph = _graph([(1, "1", 0), ("a", "b", 2)])
    result = cut.global_min_cut(graph)
    assert result["value"] == 0 and result["source_partition"] == (1,)


def test_unweighted_duplicate_records_are_aggregate_capacities():
    graph = _graph([(0, 1, 99), (0, 1, 99), (0, 2, 99), (1, 2, 99)], weighted=False)
    _assert_result(graph, cut.global_min_cut(graph), Fraction(2))
    assert not graph.weighted


def test_snapshot_tensors_metadata_and_attributes_are_unchanged():
    graph = _graph([(1, "1", 2), ("1", "t", 3), (1, "t", 4), (1, 1, 9)])
    graph = graph.with_attributes(nodes={1: {"name": "integer"}}, edges={(1, "1"): {"note": "original"}}, graph_attributes={"title": "Original"})
    before = graph.metadata, graph.node_attributes, graph.edge_attributes, graph.graph_attributes, graph._edges.indices().clone(), graph._edges.values().clone()
    result = cut.global_min_cut(graph)
    _assert_result(graph, result, _oracle(graph))
    assert (graph.metadata, graph.node_attributes, graph.edge_attributes, graph.graph_attributes) == before[:4]
    assert torch.equal(graph._edges.indices(), before[4]) and torch.equal(graph._edges.values(), before[5])


def test_exact_integer_near_ties_preserve_capacity_terms_lost_by_float_addition():
    graph = _graph([(0, 1, 1e16), (0, 2, 1e16), (1, 2, 1.)])
    result = cut.global_min_cut(graph)
    expected = Fraction(10**16 + 1)
    _assert_result(graph, result, expected)
    assert result["value"] == 1e16
    assert result["metadata"]["exact_capacity_numerator"] == 10**16 + 1


def test_intermediate_degree_can_overflow_float_without_corrupting_finite_optimum():
    graph = _graph([(0, 1, 1e308), (0, 2, 1e308), (1, 2, 1.)])
    expected = Fraction(1e308) + 1
    result = cut.global_min_cut(graph)
    _assert_result(graph, result, expected)
    assert result["value"] == 1e308


def test_entire_binary64_dynamic_range_is_exact_including_smallest_subnormal():
    tiny = math.ldexp(1., -1074)
    graph = _graph([(0, 1, 1e308), (1, 2, tiny)])
    result = cut.global_min_cut(graph)
    _assert_result(graph, result, Fraction(tiny))
    assert result["value"] == tiny
    assert result["metadata"]["maximum_integer_bits"] > 2000


def test_huge_loop_does_not_change_exact_scaling_of_tiny_cut():
    tiny = 1e-300
    graph = _graph([(0, 0, 1e308), (0, 1, tiny)])
    _assert_result(graph, cut.global_min_cut(graph), Fraction(tiny))


def test_final_exact_capacity_output_overflow_is_explicit():
    graph = _graph([(0, 1, 1e308), (0, 2, 1e308), (1, 2, 1e308)])
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(graph)
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("nodes", [[], [1]])
def test_global_cut_requires_nontrivial_partition(nodes):
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(_graph(nodes=nodes))
    assert error.value.code == "network_invalid_option"


def test_directed_budget_refused_before_algorithm_buffers(monkeypatch):
    graph = _graph([(0, 1, 1)], directed=True)
    monkeypatch.setattr(torch, "arange", lambda *a, **k: pytest.fail("allocated before direction validation"))
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(graph, max_work=1)
    assert error.value.code == "network_work_budget"


@pytest.mark.parametrize("maximum", [True, 0, -1, None, 1.5, 2**64])
def test_invalid_work_options(maximum):
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(_graph([(0, 1, 1)]), max_work=maximum)
    assert error.value.code == "network_invalid_option"


def test_initial_work_guard_before_union_find_buffers(monkeypatch):
    graph = _graph([(0, 1, 1)])
    monkeypatch.setattr(torch, "arange", lambda *a, **k: pytest.fail("allocated before work admission"))
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(graph, max_work=1)
    assert error.value.code == "network_work_budget"


def test_memory_guard_before_union_find_buffers(monkeypatch):
    graph = _graph([(0, 1, 1)])
    graph._budget.limit = graph._base_bytes + 4096
    monkeypatch.setattr(torch, "arange", lambda *a, **k: pytest.fail("allocated before memory admission"))
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(graph)
    assert error.value.code == "network_memory_budget"


def test_connected_phase_lower_bound_precedes_adjacency_allocation(monkeypatch):
    n = 1000
    graph = _graph([(i, i + 1, 1) for i in range(n - 1)])
    monkeypatch.setattr(cut, "_Adjacency", lambda *a, **k: pytest.fail("allocated before connected phase work admission"))
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(graph, max_work=100_000)
    assert error.value.code == "network_work_budget"


def test_actual_phase_work_guard_returns_no_incomplete_answer():
    graph = _graph([(i, j, 1) for i, j in itertools.combinations(range(12), 2)])
    with pytest.raises(AnalysisError) as error:
        cut.global_min_cut(graph, max_work=1200)
    assert error.value.code == "network_work_budget"


def test_sparse_dictionary_resizes_are_admitted_before_mutation():
    graph = _graph([(0, 1, 1)])
    work = cut._Work(10_000, 1)
    adjacency = cut._Adjacency(graph, 0, 36, work)
    graph._budget.limit = graph._base_bytes + 4096 + adjacency.table_bytes + 100
    with pytest.raises(AnalysisError) as error:
        adjacency.put_pair(0, 1, 1)
    assert error.value.code == "network_memory_budget"
    assert adjacency.rows == [{}, {}] and adjacency.arcs == 0


def test_dictionary_compaction_keeps_live_sparse_bound_and_symmetry():
    graph = _graph([(i, j, 1) for i, j in itertools.combinations(range(32), 2)], max_memory_mb=16)
    work = cut._Work(1_000_000, 1)
    adjacency = cut._Adjacency(graph, 0, 36, work)
    for i, j in itertools.combinations(range(32), 2):
        adjacency.put_pair(i, j, 1)
    for remove in range(31, 0, -1):
        adjacency.contract(0, remove, 1)
        assert adjacency.arcs == sum(map(len, adjacency.rows))
        assert adjacency.table_bytes == sum(map(__import__('sys').getsizeof, adjacency.rows))
        for i, row in enumerate(adjacency.rows):
            assert all(adjacency.rows[j][i] == value for j, value in row.items())
            assert __import__('sys').getsizeof(row) <= max(224, 128 * len(row))


def test_disconnected_large_isolate_graph_avoids_quadratic_phase_bound():
    graph = _graph([(0, 1, 1)], nodes=range(20_000), max_memory_mb=64)
    result = cut.global_min_cut(graph, max_work=1_000_000)
    assert result["value"] == 0 and result["metadata"]["phases"] == 0
    assert result["source_partition"] == (0, 1)


def test_summary_latex_file_export_is_bounded(tmp_path):
    graph = _graph([(0, 1, 3), (1, 2, 2), (0, 2, 5)])
    result = cut.global_min_cut(graph)
    assert len(result.summary()) == 8
    assert result.summary().attrs["exact_optimum_for_stored_weights"]
    assert len(result.to_latex()) < 2000 and len(repr(result)) < 160
    destination = tmp_path / "global-cut.tex"
    assert result.to_latex(destination) is None
    assert destination.read_text() == result.to_latex()


def test_default_meta_device_context_keeps_cut_buffers_cpu():
    graph = _graph([(0, 1, .5), (1, 2, .25), (0, 2, 1)])
    with torch.device("meta"):
        _assert_result(graph, cut.global_min_cut(graph), _oracle(graph))
