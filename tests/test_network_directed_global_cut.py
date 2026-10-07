"""All directed bipartitions and exact raw-record capacities, not a flow oracle."""
from fractions import Fraction
import itertools
import math
import random

import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def graph(records=(), *, nodes=(), **options):
    return oe.network([{"source": a, "target": b, "w": w} for a, b, w in records],
                      nodes=nodes, weight="w", directed=True, **options)


def key(label):
    return (0, label) if isinstance(label, int) else (1, label)


def oracle(records, nodes):
    candidates = []
    for bits in range(1, 2**len(nodes) - 1):
        selected = {node for i, node in enumerate(nodes) if bits & (1 << i)}
        candidates.append(sum((Fraction(w) for a, b, w in records
                               if a in selected and b not in selected), Fraction()))
    return min(candidates)


def certificate(result, records, nodes):
    selected, other = set(result["source_partition"]), set(result["target_partition"])
    assert selected and other and not selected & other and selected | other == set(nodes)
    expected = oracle(records, nodes)
    metadata = result["metadata"]
    assert result["value"] == float(expected)
    assert Fraction(metadata["exact_capacity_numerator"], metadata["exact_capacity_denominator"]) == expected
    assert metadata["directed"] and metadata["exact"] and not metadata["sampled"]
    assert metadata["exact_optimum_for_stored_weights"] and metadata["certified"]
    assert metadata["work_used"] <= metadata["max_work"]
    assert result["source_partition"] == tuple(sorted(selected, key=key))
    assert result["target_partition"] == tuple(sorted(other, key=key))
    expected_edges = {}
    for a, b, w in records:
        if w > 0 and a in selected and b in other:
            expected_edges[a, b] = expected_edges.get((a, b), Fraction()) + Fraction(w)
    actual = {(row.source, row.target): Fraction(row.capacity)
              for row in result["cut_edges"].itertuples(index=False)}
    assert len(actual) == len(result["cut_edges"])
    assert actual == expected_edges
    assert sum(actual.values(), Fraction()) == expected
    assert result["cut_edges"].source.dtype == result["cut_edges"].target.dtype == object
    assert "\\begin{tabular}" in result.to_latex()


@pytest.mark.parametrize("mask", range(64))
@pytest.mark.parametrize("weighted", [False, True])
def test_every_three_node_directed_topology_matches_all_bipartitions(mask, weighted):
    nodes = [1, "1", "b"]
    arcs = list(itertools.permutations(nodes, 2))
    records = [(a, b, (i + 1) / 8 if weighted else 1.)
               for i, (a, b) in enumerate(arcs) if mask & (1 << i)]
    # Loops and zero records cannot cross a cut. Duplicates sum capacities.
    records += [(nodes[0], nodes[0], 1e300), (nodes[0], nodes[1], 0.)]
    if mask & 1:
        records.append(records[0])
    certificate(graph(records, nodes=nodes).global_min_cut(), records, nodes)


@pytest.mark.parametrize("seed", range(80))
def test_weighted_directed_typed_graphs_match_raw_fraction_enumeration(seed):
    rng = random.Random(seed)
    nodes = [1, "1", "a", 2, "z"]
    records = [(a, b, rng.choice([.125, .25, 1., 8., 16.]))
               for a, b in itertools.product(nodes, repeat=2) if rng.random() < .5]
    if records:
        records += [records[0]] * 2
    rng.shuffle(records)
    certificate(graph(records, nodes=list(reversed(nodes))).global_min_cut(), records, nodes)


def test_directed_orientation_must_not_flip_to_include_smallest_label():
    records = [(1, "1", 9.), ("1", 1, 2.)]
    result = graph(records).global_min_cut()
    certificate(result, records, [1, "1"])
    assert result["source_partition"] == ("1",)
    assert list(result["cut_edges"].itertuples(index=False, name=None)) == [("1", 1, 2.)]


@pytest.mark.parametrize("records,nodes", [
    ([(0, 1, 5.)], [0, 1]),
    ([(0, 1, 5.), (1, 0, 4.), (1, 2, 9.)], [0, 1, 2]),
    ([(0, 1, 0.), (1, 0, 0.)], [0, 1]),
    ([(0, 0, 1e308), (1, 1, 1e-300)], [0, 1, "isolated"]),
])
def test_zero_cuts_for_weak_and_strong_disconnection(records, nodes):
    result = graph(records, nodes=nodes).global_min_cut()
    certificate(result, records, nodes)
    assert result["value"] == 0 and result["cut_edges"].empty
    assert result["metadata"]["flow_problems"] == 0


def test_subnormal_to_huge_capacities_do_not_lose_the_directed_optimum():
    tiny = math.ulp(0.)
    records = [(0, 1, 1e308), (1, 0, tiny), (0, 0, 1e308)]
    certificate(graph(records).global_min_cut(), records, [0, 1])


def test_exact_near_ties_and_sums_above_intermediate_float_range():
    nodes = [0, 1, 2]
    records = [(0, 1, 1e308), (0, 2, 1e308), (1, 0, .1), (2, 0, .1),
               (1, 2, 1.), (2, 1, 1.)]
    certificate(graph(records).global_min_cut(), records, nodes)


def test_ties_and_identity_alignment_are_import_order_invariant():
    nodes = ["b", 1, "1", "a"]
    records = [(a, b, 1.) for a, b in itertools.permutations(nodes, 2)]
    expected = None
    for seed in range(12):
        rng = random.Random(seed)
        rng.shuffle(nodes)
        rng.shuffle(records)
        result = graph(records, nodes=nodes).global_min_cut()
        certificate(result, records, nodes)
        value = result["source_partition"], result["target_partition"], result["cut_edges"].to_dict("records")
        expected = expected or value
        assert value == expected


def test_directed_work_and_memory_admission_precede_algorithm_buffers(monkeypatch):
    network = graph([(0, 1, 1.), (1, 0, 1.)])
    monkeypatch.setattr(torch, "full", lambda *a, **kw: pytest.fail("allocated before admission"))
    with pytest.raises(AnalysisError) as caught:
        network.global_min_cut(max_work=1)
    assert caught.value.code == "network_work_budget"
    network._budget.limit = network._base_bytes + 4096
    with pytest.raises(AnalysisError) as caught:
        network.global_min_cut()
    assert caught.value.code == "network_memory_budget"


def test_actual_traversal_exhaustion_returns_no_partial_global_cut(monkeypatch):
    from openecon._network_directed_cut import _Residual

    records = [(a, b, 1.) for a, b in itertools.permutations(range(8), 2)]
    started = []
    solve = _Residual.solve
    def tracking_solve(self, source, target):
        started.append((source, target))
        return solve(self, source, target)
    monkeypatch.setattr(_Residual, "solve", tracking_solve)
    with pytest.raises(AnalysisError) as caught:
        graph(records).global_min_cut(max_work=6500)
    assert caught.value.code == "network_work_budget"
    assert started, "The budget must pass admission and exhaust during actual traversal"


def test_original_snapshot_and_ambient_torch_context_are_preserved():
    records = [(0, 1, 2.), (1, 2, 4.), (2, 0, 8.), (0, 2, 1.)]
    network = graph(records)
    values, indices, metadata = network._edges.values().clone(), network._edges.indices().clone(), network.metadata
    with torch.device("meta"):
        result = network.global_min_cut()
    certificate(result, records, [0, 1, 2])
    assert torch.equal(values, network._edges.values()) and torch.equal(indices, network._edges.indices())
    assert metadata == network.metadata


@pytest.mark.parametrize("nodes", [[], [1], ["one"]])
def test_directed_nontrivial_partition_requirement(nodes):
    with pytest.raises(AnalysisError) as caught:
        graph(nodes=nodes).global_min_cut()
    assert caught.value.code == "network_invalid_option"


def test_true_export_overflow_is_explicit_after_exact_search():
    records = [(a, b, 1e308) for a, b in itertools.permutations(range(3), 2)]
    with pytest.raises(AnalysisError) as caught:
        graph(records).global_min_cut()
    assert caught.value.code == "network_precision"


def test_physical_parquet_batches_and_unweighted_duplicate_capacities(tmp_path):
    import pandas as pd

    records = [(0, 1, 1.), (0, 1, 1.), (1, 0, 1.), (1, 2, 1.), (2, 0, 1.)]
    path = tmp_path / "directed.parquet"
    pd.DataFrame(records, columns=["source", "target", "ignored"]).to_parquet(path, row_group_size=2)
    for batch in (1, 2, 4, 100):
        network = oe.network(oe.scan(path), directed=True, batch_rows=batch)
        result = network.global_min_cut()
        certificate(result, records, [0, 1, 2])
        assert network.metadata["input_rows"] == 5


def test_large_disconnected_graph_returns_zero_without_rooted_flow_plan():
    network = graph([(0, 1, 1.)], nodes=range(10_000), max_memory_mb=32)
    result = network.global_min_cut(max_work=1_000_000)
    assert result["value"] == 0 and result["metadata"]["flow_problems"] == 0
    assert len(result["source_partition"]) + len(result["target_partition"]) == 10_000
