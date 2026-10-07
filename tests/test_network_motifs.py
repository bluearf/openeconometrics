"""Independent labelled-subgraph isomorphism oracles for native triad census."""
from copy import deepcopy
from itertools import combinations, permutations
import random

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon import _network_motifs as motifs
from openecon.analysis_contracts import AnalysisError
from openecon.frame import DataFrame


# Canonical representatives read from the published sixteen-class triad table,
# independently of the implementation's degree and dyad classification rules.
REPRESENTATIVES = {
    "003": [], "012": [(0, 1)], "102": [(0, 1), (1, 0)],
    "021D": [(0, 1), (0, 2)], "021U": [(1, 0), (2, 0)],
    "021C": [(0, 1), (1, 2)],
    "111D": [(0, 1), (1, 0), (2, 0)],
    "111U": [(0, 1), (1, 0), (0, 2)],
    "030T": [(0, 1), (0, 2), (1, 2)],
    "030C": [(0, 1), (1, 2), (2, 0)],
    "201": [(0, 1), (1, 0), (0, 2), (2, 0)],
    "120D": [(0, 1), (1, 0), (2, 0), (2, 1)],
    "120U": [(0, 1), (1, 0), (0, 2), (1, 2)],
    "120C": [(0, 1), (1, 0), (1, 2), (2, 0)],
    "210": [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)],
    "300": [(a, b) for a in range(3) for b in range(3) if a != b],
}
ARCS = tuple((a, b) for a in range(3) for b in range(3) if a != b)


def isomorphism_lookup():
    table = {}
    for name, edges in REPRESENTATIVES.items():
        for permutation in permutations(range(3)):
            relabelled = frozenset((permutation[a], permutation[b]) for a, b in edges)
            if relabelled in table:
                assert table[relabelled] == name
            table[relabelled] = name
    assert len(table) == 64
    return table


ISO = isomorphism_lookup()


def graph(edges=(), *, nodes=(), directed=False, weighted=True, **options):
    records = [{"source": a, "target": b, "weight": w} for a, b, w in edges]
    return oe.network(records, nodes=nodes, directed=directed,
                      weight="weight" if weighted else None, **options)


def reference(edges, labels, directed):
    positive = {(a, b) for a, b, w in edges if w > 0 and a != b}
    if not directed:
        positive |= {(b, a) for a, b in positive}
    counts = dict.fromkeys(motifs.DIRECTED_TRIADS if directed else motifs.UNDIRECTED_TRIADS, 0)
    for triple in combinations(labels, 3):
        induced = frozenset((a, b) for a, label_a in enumerate(triple)
                            for b, label_b in enumerate(triple)
                            if a != b and (label_a, label_b) in positive)
        name = ISO[induced] if directed else motifs.UNDIRECTED_TRIADS[len(induced) // 2]
        counts[name] += 1
    return counts


def values(result):
    return dict(zip(result.triad, result["count"]))


@pytest.mark.parametrize("mask", range(64))
def test_all_64_directed_orientations_have_the_standard_induced_class(mask):
    selected = [edge for bit, edge in enumerate(ARCS) if mask & (1 << bit)]
    network = graph([(a, b, 1) for a, b in selected], nodes=range(3), directed=True)
    result = motifs.triad_census(network)
    expected = ISO[frozenset(selected)]
    assert isinstance(result, DataFrame)
    assert result.triad.tolist() == list(motifs.DIRECTED_TRIADS)
    assert values(result) == {name: int(name == expected) for name in motifs.DIRECTED_TRIADS}
    assert result.attrs["total_triads"] == 1
    assert result["count"].dtype == object
    assert all(type(value) is int for value in result["count"])


@pytest.mark.parametrize("mask", range(8))
def test_all_undirected_triples(mask):
    dyads = ((0, 1), (0, 2), (1, 2))
    selected = [(a, b, 7) for bit, (a, b) in enumerate(dyads) if mask & (1 << bit)]
    result = motifs.triad_census(graph(selected, nodes=range(3)))
    assert result.triad.tolist() == list(motifs.UNDIRECTED_TRIADS)
    assert values(result) == reference(selected, list(range(3)), False)


@pytest.mark.parametrize("name", list(REPRESENTATIVES))
def test_directed_permutation_and_typed_relabelling_invariance(name):
    labels = [2, "2", -4]
    expected = {kind: int(kind == name) for kind in motifs.DIRECTED_TRIADS}
    for order in permutations(labels):
        edges = [(order[a], order[b], 5) for a, b in REPRESENTATIVES[name]]
        # Snapshot order differs from label/permutation ordering.
        result = motifs.triad_census(graph(edges, nodes=list(reversed(order)), directed=True))
        assert values(result) == expected


@pytest.mark.parametrize("directed", [False, True])
def test_random_small_snapshots_match_independent_brute_induced_subgraphs(directed):
    rng = random.Random(219073)
    labels = [0, 1, "1", -2, "four", "isolate", 9]
    for _ in range(60):
        edges = [(a, b, rng.choice([1., 3., 17.])) for a in labels[:-1] for b in labels[:-1]
                 if rng.random() < .3]
        if edges:
            edges.extend([edges[0], (edges[0][0], edges[0][1], 0.)])
        network = graph(edges, nodes=labels, directed=directed)
        result = motifs.triad_census(network)
        assert values(result) == reference(edges, labels, directed)
        assert sum(result["count"]) == 35
        assert result.attrs["total_triads"] == 35
        assert result.attrs["exact"] and result.attrs["induced"]
        assert result.attrs["membership_probes"] >= result.attrs["weak_triangles"]
        assert result.attrs["actual_work"] <= result.attrs["max_work"]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("size", [0, 1, 2, 3, 19])
def test_empty_graphs_and_isolates_count_null_triples(directed, size):
    result = motifs.triad_census(graph(nodes=range(size), directed=directed))
    counts = values(result)
    null = "003" if directed else "empty"
    assert counts[null] == size * (size - 1) * (size - 2) // 6
    assert sum(counts.values()) == counts[null]
    assert result.attrs["topology_edges"] == result.attrs["weak_projection_edges"] == 0
    assert result.attrs["weak_triangles"] == 0


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_weights_loops_parallel_and_zero_rows_do_not_multiply_motifs(directed, weighted):
    edges = [(0, 0, 1e250), (1, 1, 3.), (2, 2, 7.),
             (0, 1, 1e200), (0, 1, 1e200), (1, 0, 2.),
             (1, 2, 1e-250), (2, 1, 0.)]
    network = graph(edges, nodes=[0, 1, 2, "isolate"], directed=directed, weighted=weighted)
    result = motifs.triad_census(network)
    expected_edges = edges if weighted else [(a, b, 1) for a, b, _ in edges]
    assert values(result) == reference(expected_edges, [0, 1, 2, "isolate"], directed)
    assert result.attrs["self_loops_excluded"] == 3
    assert result.attrs["weight_semantics"] == "binary unique positive-edge topology"
    assert result.attrs["parallel_edges"] == "coalesced input edges count once"


@pytest.mark.parametrize("directed", [False, True])
def test_disconnected_dyads_and_loops_only_vertices_have_exact_dyadic_counts(directed):
    edges = [(0, 1, 1), (1, 0, 1), (2, 3, 1), (4, 4, 9)]
    labels = list(range(8))
    result = motifs.triad_census(graph(edges, nodes=labels, directed=directed))
    assert values(result) == reference(edges, labels, directed)
    assert result.attrs["weak_projection_edges"] == 2
    assert result.attrs["weak_triangles"] == 0


@pytest.mark.parametrize("size,directed", [(25_000, True), (25_000, False)])
def test_large_star_uses_linear_work_without_enumerating_open_wedges(size, directed):
    network = graph([(0, i, 1) for i in range(1, size)], nodes=range(size), directed=directed,
                    max_memory_mb=96)
    result = motifs.triad_census(network, max_work=5_000_000)
    counts = values(result)
    expected_wedges = (size - 1) * (size - 2) // 2
    assert counts["021D" if directed else "two_edge_path"] == expected_wedges
    assert counts["012" if directed else "one_edge"] == 0
    assert counts["003" if directed else "empty"] == (size - 1) * (size - 2) * (size - 3) // 6
    assert result.attrs["weak_triangles"] == 0
    assert result.attrs["membership_probes"] == 0
    assert result.attrs["planned_work"] < 5_000_000
    assert all(type(value) is int for value in result["count"])


def test_combinatorial_counts_never_use_int64_or_float64():
    size = 10_000_000
    count = motifs._choose3(size)
    assert count == 166_666_616_666_670_000_000
    assert count > 2**63
    assert type(count) is int
    network = graph(nodes=range(3), directed=True)
    counts = dict.fromkeys(motifs.DIRECTED_TRIADS, 0)
    counts["003"] = count
    result = motifs._result(network, counts, maximum=1, work=0, probes=0,
        topology_edges=0, weak_edges=0, loops=0, triangles=0)
    assert result.loc[0, "count"] == count and type(result.loc[0, "count"]) is int
    assert str(count) in str(result.to_latex(index=False))


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "50", None])
def test_invalid_work_limits(limit):
    with pytest.raises(AnalysisError) as error:
        motifs.triad_census(graph(), max_work=limit)
    assert error.value.code == "network_invalid_option"


def test_setup_work_and_memory_are_admitted_before_csr_allocation(monkeypatch):
    network = graph([(0, 1, 1)], nodes=range(40), directed=True)
    monkeypatch.setattr(motifs, "csr", lambda *args, **kwargs: pytest.fail("CSR allocated before admission"))
    with pytest.raises(AnalysisError) as error:
        motifs.triad_census(network, max_work=1)
    assert error.value.code == "network_work_budget"
    network._budget.limit = network._base_bytes + 4096
    with pytest.raises(AnalysisError) as error:
        motifs.triad_census(network)
    assert error.value.code == "network_memory_budget"


def test_dense_triangle_work_refuses_before_classification_and_accepts_exact_bound(monkeypatch):
    n = 22
    network = graph([(a, b, 1) for a in range(n) for b in range(n) if a != b], directed=True)
    baseline = motifs.triad_census(network)
    planned = baseline.attrs["planned_work"]
    assert values(baseline)["300"] == n * (n - 1) * (n - 2) // 6
    assert baseline.attrs["weak_triangles"] == values(baseline)["300"]
    assert values(motifs.triad_census(network, max_work=planned)) == values(baseline)
    monkeypatch.setattr(motifs, "_forward_rows", lambda *args: pytest.fail("forward rows allocated before admission"))
    with pytest.raises(AnalysisError) as error:
        motifs.triad_census(network, max_work=planned - 1)
    assert error.value.code == "network_work_budget"


def test_forward_memory_guard_reserves_both_csr_directions_and_dictionary_rows(monkeypatch):
    network = graph([(0, 1, 1), (1, 2, 1), (2, 0, 1)], directed=True)
    calls = []
    original_guard = network._guard

    def guard(workspace):
        calls.append(workspace)
        original_guard(workspace)

    monkeypatch.setattr(network, "_guard", guard)
    original_forward = motifs._forward_rows

    def forward(topology):
        expected = topology.storage_bytes + 256 * topology.n + 256 * 3 + 8192
        assert calls[-1] == expected
        # Reject precisely this simultaneous workspace before any Python rows.
        network._budget.limit = network._base_bytes + expected + 4095
        with pytest.raises(AnalysisError) as error:
            original_guard(expected)
        assert error.value.code == "network_memory_budget"
        network._budget.limit = 512 * 1024**2
        return original_forward(topology)

    monkeypatch.setattr(motifs, "_forward_rows", forward)
    assert values(motifs.triad_census(network))["030C"] == 1


@pytest.mark.parametrize("directed", [False, True])
def test_physical_parquet_batches_are_complete_and_source_unchanged(tmp_path, directed):
    edges = [(0, 1, 2), (0, 1, 7), (1, 0, 1), (1, 2, 4),
             (2, 0, 1), (2, 3, 2), (3, 4, 3), (4, 4, 9), (8, 8, 0)]
    path = tmp_path / "triad-edges.parquet"
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_parquet(path)
    dataset = oe.scan(path)
    network = oe.network(dataset, weight="weight", batch_rows=2, nodes=["isolate"], directed=directed)
    result = motifs.triad_census(network)
    assert values(result) == reference(edges, list(network._labels), directed)
    assert result.attrs["network"]["input_rows"] == len(edges)
    assert result.attrs["network"]["actual_peak_batch_rows"] == 2
    assert result.attrs["total_triads"] == 35
    assert "\\begin{tabular}" in str(result.to_latex(index=False))
    dataset.assert_unchanged()


@pytest.mark.parametrize("directed", [False, True])
def test_snapshot_unchanged_and_native_cpu_ignores_default_device_dtype(directed):
    network = graph([(0, 1, 2), (1, 0, 3), (1, 2, 4), (0, 2, 5), (2, 2, 100)],
                    nodes=[0, "0", 1, 2], directed=directed)
    original_metadata = deepcopy(network.metadata)
    original_edges = network._edges._indices().clone(), network._edges._values().clone()
    original_arcs = network._arcs._indices().clone(), network._arcs._values().clone()
    before = values(motifs.triad_census(network))
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = motifs.triad_census(network)
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)
    assert values(result) == before
    assert result.attrs["computation_device"] == "cpu"
    assert network.metadata == original_metadata
    assert torch.equal(network._edges._indices(), original_edges[0])
    assert torch.equal(network._edges._values(), original_edges[1])
    assert torch.equal(network._arcs._indices(), original_arcs[0])
    assert torch.equal(network._arcs._values(), original_arcs[1])
