"""Independent small-graph oracles for native bipartite/link analysis."""
from fractions import Fraction
import itertools
import math
import random

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_bipartite import (_partition_ids, bipartite,
                                        bipartite_projection, link_prediction)


def graph_from(edges, *, nodes=None, directed=False, weight=None, **kwargs):
    records = [{"source": left, "target": right, "weight": value}
               for left, right, value in edges]
    return oe.network(records, nodes=nodes, directed=directed, weight=weight, **kwargs)


def scores(graph, pairs, method):
    return link_prediction(graph, pairs, method).score.tolist()


def edge_map(graph):
    pairs, values = graph._edges.indices(), graph._edges.values()
    return {frozenset((graph._labels[int(pairs[0, at])], graph._labels[int(pairs[1, at])])): float(values[at])
            for at in range(graph.edge_count)}


def test_typed_identity_disconnected_canonical_orientation_and_isolates():
    graph = graph_from([("z", "b", 1), ("1", 1, 1), ("a", "b", 1)], nodes=["alone", 5])
    result = bipartite(graph)
    actual = dict(zip(result.node, result.partition))
    assert actual == {"alone": 0, 5: 0, "z": 0, "b": 1, "1": 1, 1: 0, "a": 0}
    assert isinstance(result, oe.DataFrame)
    assert "\\begin{tabular}" in result.to_latex()
    assert result.attrs["inferred"]
    reverse = graph_from([("b", "a", 1), (1, "1", 1), ("b", "z", 1)], nodes=[5, "alone"])
    assert actual == dict(zip(bipartite(reverse).node, bipartite(reverse).partition))


@pytest.mark.parametrize("mask", range(64))
def test_all_four_node_simple_topologies_against_complete_color_assignment(mask):
    pairs = list(itertools.combinations(range(4), 2))
    edges = [pair for at, pair in enumerate(pairs) if mask & (1 << at)]
    oracle = any(all(side[left] != side[right] for left, right in edges)
                 for side in itertools.product((0, 1), repeat=4))
    graph = graph_from([(left, right, 1) for left, right in edges], nodes=range(4))
    if oracle:
        ids = _partition_ids(graph)
        assert all(ids[left] != ids[right] for left, right in edges)
    else:
        with pytest.raises(AnalysisError) as error:
            _partition_ids(graph)
        assert error.value.code == "network_not_bipartite"


def test_directed_partition_weak_semantics_no_strength_overflow():
    graph = graph_from([("a", "b", 1e308), ("b", "a", 1e308), ("c", "b", 1e308)],
                       directed=True, weight="weight")
    result = bipartite(graph)
    assert result.partition.tolist() == [0, 1, 0]
    assert result.attrs["directed_semantics"] == "weak binary projection"


@pytest.mark.parametrize("partition", [
    {"a": 0}, {"a": 0, "b": 0}, {"a": False, "b": 1}, {"a": .0, "b": 1},
    {"a": 0, "other": 1}, [0, 1], {"a": 0, "b": 2},
    pd.DataFrame({"node": ["a", "a"], "partition": [0, 1]}),
    pd.DataFrame({"wrong": ["a", "b"], "partition": [0, 1]}),
])
def test_rejects_incomplete_unknown_duplicate_and_invalid_partitions(partition):
    graph = graph_from([("a", "b", 1)])
    with pytest.raises(AnalysisError):
        bipartite(graph, partition)


def test_manual_partition_preserves_orientation_and_isolate_side():
    graph = graph_from([("a", "b", 1)], nodes=["iso"])
    supplied = pd.DataFrame({"node": ["b", "a", "iso"], "partition": [0, 1, 1]})
    result = bipartite(graph, supplied)
    assert dict(zip(result.node, result.partition)) == {"a": 1, "b": 0, "iso": 1}
    assert result.attrs["inferred"] is False


@pytest.mark.parametrize("directed", [False, True])
def test_positive_self_loop_rejected_but_zero_self_loop_only_retains_isolate(directed):
    graph = graph_from([("a", "a", 1)], directed=directed)
    with pytest.raises(AnalysisError) as error:
        bipartite(graph)
    assert error.value.code == "network_not_bipartite"
    zero = graph_from([("a", "a", 0)], weight="weight", directed=directed)
    assert bipartite(zero).partition.tolist() == [0]


def weighted_bipartite():
    edges = [(1, "x", 2), ("1", "x", 3), (1, "x", .5), (1, "y", 4),
             ("1", "y", 1), ("b", "x", 7), ("b", "z", 9), ("zero", "z", 0)]
    graph = graph_from(edges, nodes=["iso", "right_iso"], weight="weight")
    partition = {label: 1 if label in {"x", "y", "z", "right_iso"} else 0 for label in graph._labels}
    return graph, partition, edges


@pytest.mark.parametrize("onto", [0, 1])
@pytest.mark.parametrize("weight", ["count", "binary", "product"])
def test_weighted_projection_against_independent_fraction_oracle(onto, weight):
    graph, partition, edges = weighted_bipartite()
    strengths = {}
    for left, right, value in edges:
        if value:
            pair = frozenset((left, right))
            strengths[pair] = strengths.get(pair, Fraction()) + Fraction(value)
    chosen = [label for label in graph._labels if partition[label] == onto]
    opposite = [label for label in graph._labels if partition[label] != onto]
    expected = {}
    for left, right in itertools.combinations(chosen, 2):
        products = [strengths.get(frozenset((left, center)), 0) * strengths.get(frozenset((right, center)), 0)
                    for center in opposite]
        positive = [term for term in products if term]
        if positive:
            value = float(sum(positive)) if weight == "product" else 1. if weight == "binary" else float(len(positive))
            expected[frozenset((left, right))] = value
    before = graph._edges.clone()
    result = bipartite_projection(graph, partition, onto, weight)
    assert result._labels == tuple(chosen)
    assert result.weighted is (weight != "binary")
    assert edge_map(result) == expected
    assert result.metadata["projection_weight"] == weight
    assert torch.equal(graph._edges.values(), before.values())
    assert torch.equal(graph._edges.indices(), before.indices())


def test_projection_retains_selected_node_and_graph_attributes_explicitly_excludes_original_edges():
    graph, partition, _ = weighted_bipartite()
    graph = graph.with_attributes(nodes={1: {"label": "integer"}, "x": {"kind": "right"}, "iso": {"flag": True}},
                                  edges={(1, "x"): {"note": "original"}}, graph_attributes={"title": "Graph"})
    result = bipartite_projection(graph, partition)
    assert result.node_attributes == {1: {"label": "integer"}, "iso": {"flag": True}}
    assert result.graph_attributes == {"title": "Graph"}
    assert result.edge_attributes == {}
    assert result.metadata["edge_attributes"] == "not transferred to derived edges"
    result._node_attributes[1]["label"] = "modified"
    assert graph.node_attributes[1]["label"] == "integer"


@pytest.mark.parametrize("weight,value", [("product", 1e308), ("product", 1e-300)])
def test_projection_product_precision_errors(weight, value):
    graph = graph_from([("a", "x", value), ("b", "x", value)], weight="weight")
    with pytest.raises(AnalysisError) as error:
        bipartite_projection(graph, {"a": 0, "b": 0, "x": 1}, weight=weight)
    assert error.value.code == "network_precision"


def test_projection_product_preserves_small_terms_with_compensated_accumulation():
    graph = graph_from([("a", "x", 1e16), ("b", "x", 1),
                        ("a", "y", 1), ("b", "y", 1),
                        ("a", "z", 1), ("b", "z", 1)], weight="weight")
    partition = {"a": 0, "b": 0, "x": 1, "y": 1, "z": 1}
    assert edge_map(bipartite_projection(graph, partition, weight="product")) == {
        frozenset(("a", "b")): 1e16 + 2}


def test_projection_reuses_single_operation_owned_csr(monkeypatch):
    import openecon._network_bipartite as module
    original, calls = module.csr, []

    def wrapped(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "csr", wrapped)
    graph = graph_from([("a", "x", 1), ("b", "x", 1)])
    assert bipartite_projection(graph).edge_count == 1
    assert len(calls) == 1


def test_projection_known_work_lower_bound_before_sparse_allocation(monkeypatch):
    import openecon._network_bipartite as module
    graph = graph_from([("a", "x", 1), ("b", "x", 1)])

    def forbidden(*args, **kwargs):
        raise AssertionError("Known work lower bound must reject before CSR allocation")

    monkeypatch.setattr(module, "csr", forbidden)
    with pytest.raises(AnalysisError) as error:
        bipartite_projection(graph, {"a": 0, "b": 0, "x": 1}, max_work=1)
    assert error.value.code == "network_work_budget"


@pytest.mark.parametrize("options,code", [({"max_work": 4}, "network_work_budget"),
                                          ({"max_edges": 2}, "network_edge_budget"),
                                          ({"max_work": True}, "network_invalid_option"),
                                          ({"max_edges": 0}, "network_invalid_option"),
                                          ({"onto": True}, "network_invalid_partition"),
                                          ({"onto": 2}, "network_invalid_partition"),
                                          ({"weight": "unsupported"}, "network_invalid_option")])
def test_projection_explicit_guard_errors_preserve_parent(options, code):
    graph = graph_from([(node, "x", 1) for node in range(5)])
    before = graph._edges.clone()
    with pytest.raises(AnalysisError) as error:
        bipartite_projection(graph, {**{node: 0 for node in range(5)}, "x": 1}, **options)
    assert error.value.code == code
    assert torch.equal(graph._edges.values(), before.values())


def test_projection_directed_rejected_and_empty_graph_isolate_graph_supported():
    directed = graph_from([("a", "x", 1)], directed=True)
    with pytest.raises(AnalysisError) as error:
        bipartite_projection(directed)
    assert error.value.code == "network_directed_unsupported"
    empty = graph_from([], nodes=["a", 1])
    assert bipartite_projection(empty).node_count == 2
    assert bipartite_projection(empty, onto=1).node_count == 0


def oracle_score(neighbors, left, right, method):
    common = neighbors[left] & neighbors[right]
    if method == "common_neighbors":
        return len(common)
    if method == "jaccard":
        union = neighbors[left] | neighbors[right]
        return len(common) / len(union) if union else 0.
    if method == "adamic_adar":
        return math.fsum(1 / math.log(len(neighbors[node])) for node in common)
    if method == "resource_allocation":
        return math.fsum(1 / len(neighbors[node]) for node in common)
    return len(neighbors[left]) * len(neighbors[right])


@pytest.mark.parametrize("method", ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"])
@pytest.mark.parametrize("seed", range(12))
def test_all_explicit_pair_link_scores_against_set_oracle(method, seed):
    rng, nodes = random.Random(seed), [1, "1", "a", "b", "c", "d", "iso"]
    edges, neighbors = [], {label: set() for label in nodes}
    for left, right in itertools.combinations(nodes[:-1], 2):
        if rng.random() < .45:
            edges.extend([(left, right, .5), (right, left, 2)])
            neighbors[left].add(right)
            neighbors[right].add(left)
    edges.extend([(label, label, 50) for label in nodes[:3]])
    graph = graph_from(edges, nodes=nodes, weight="weight")
    candidates = list(itertools.combinations(nodes, 2)) + [("a", 1), (1, "a")]
    actual = link_prediction(graph, candidates, method)
    expected = [oracle_score(neighbors, left, right, method) for left, right in candidates]
    assert actual.score.tolist() == pytest.approx(expected, rel=1e-13, abs=1e-14)
    assert actual.source.tolist() == [left for left, _ in candidates]
    assert actual.target.tolist() == [right for _, right in candidates]
    assert actual.attrs["sampled"] is False
    assert actual.attrs["candidate_count"] == len(candidates)
    assert isinstance(actual, oe.DataFrame)


@pytest.mark.parametrize("adapter", ["columns", "frame", "records", "tuples", "generator"])
def test_candidate_adapters_custom_names_order_duplicate_pairs(adapter):
    graph = graph_from([("a", "x", 1), ("b", "x", 1)], nodes=["iso"])
    pairs = [("a", "b"), ("b", "a"), ("iso", "a"), ("a", "b")]
    columns = {"left": [left for left, _ in pairs], "right": [right for _, right in pairs]}
    data = columns if adapter == "columns" else pd.DataFrame(columns) if adapter == "frame" else (
        [{"left": left, "right": right} for left, right in pairs] if adapter == "records"
        else pairs if adapter == "tuples" else iter(pairs))
    result = link_prediction(graph, data, "common_neighbors", source="left", target="right", batch_rows=2)
    assert result.score.tolist() == [1, 1, 0, 1]
    assert result.attrs["batch_rows"] == 2


def test_parquet_candidate_dataset_is_batched_and_same_full_result(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"source": ["a", "b", "a", "b"] * 25,
                            "target": ["b", "a", "b", "a"] * 25}), path)
    source = oe.scan(path)
    graph = graph_from([("a", "x", 1), ("b", "x", 1)])
    result = link_prediction(graph, source, method="jaccard", batch_rows=3)
    assert result.score.tolist() == [1.] * 100
    assert result.attrs["batch_rows"] == 3
    assert source.row_count == 100


@pytest.mark.parametrize("pairs", [ [("a", "a")], [("a", "unknown")], [(True, "a")], [(1., "a")],
                                    [None], [("a",)], "a", {"source": ["a"], "target": []},
                                    {"source": "a", "target": "b"}, {"wrong": ["a"]}])
def test_invalid_pairs_rejected(pairs):
    graph = graph_from([("a", "b", 1)])
    with pytest.raises(AnalysisError):
        link_prediction(graph, pairs)


@pytest.mark.parametrize("options", [ {"method": "invalid"}, {"source": "target"}, {"source": ""},
                                      {"max_pairs": 0}, {"max_pairs": True}, {"batch_rows": 0},
                                      {"max_work": True}, {"max_work": 1} ])
def test_link_option_errors(options):
    graph = graph_from([("a", "b", 1)])
    with pytest.raises(AnalysisError):
        link_prediction(graph, [("a", "b")], **options)


def test_directed_link_prediction_explicit_rejection():
    graph = graph_from([("a", "b", 1)], directed=True)
    with pytest.raises(AnalysisError) as error:
        link_prediction(graph, [("a", "b")])
    assert error.value.code == "network_directed_unsupported"


def test_sized_candidate_budget_rejected_before_iterating_or_copying():
    class LargePairs:
        def __len__(self):
            return 100_000_000

        def __iter__(self):
            raise AssertionError("Must reject size before touching candidate rows")

    graph = graph_from([("a", "b", 1)])
    with pytest.raises(AnalysisError) as error:
        link_prediction(graph, LargePairs(), max_pairs=10)
    assert error.value.code == "network_pair_budget"


def test_sized_candidate_work_budget_rejected_before_sparse_allocation(monkeypatch):
    import openecon._network_bipartite as module
    graph = graph_from([("a", "b", 1)])

    def forbidden(*args, **kwargs):
        raise AssertionError("Work plan must reject before CSR allocation")

    monkeypatch.setattr(module, "csr", forbidden)
    with pytest.raises(AnalysisError) as error:
        link_prediction(graph, [("a", "b")] * 10, max_work=graph.node_count + graph._arcs._nnz() + 5)
    assert error.value.code == "network_work_budget"


def test_unknown_length_generator_closed_on_pair_and_work_guard():
    graph = graph_from([("a", "x", 1), ("b", "x", 1)])
    closed = []

    def source():
        try:
            while True:
                yield "a", "b"
        finally:
            closed.append(True)

    for options, code in [({"max_pairs": 2}, "network_pair_budget"),
                          ({"max_work": graph.node_count + graph._arcs._nnz() + 2}, "network_work_budget")]:
        with pytest.raises(AnalysisError) as error:
            link_prediction(graph, source(), batch_rows=1, **options)
        assert error.value.code == code
    assert closed == [True, True]


def test_memory_guard_admits_before_partition_and_candidate_records():
    class ForbiddenMapping(dict):
        def items(self):
            raise AssertionError("Must admit memory before copying partition")

    graph = graph_from([("a", "b", 1)])
    graph._budget.limit = graph._base_bytes + 4096 + 1
    with pytest.raises(AnalysisError) as error:
        _partition_ids(graph, ForbiddenMapping(a=0, b=1))
    assert error.value.code == "network_memory_budget"
    with pytest.raises(AnalysisError) as error:
        link_prediction(graph, [("a", "b")])
    assert error.value.code == "network_memory_budget"


def test_empty_link_output_has_typed_columns_and_zero_isolate_scores():
    graph = graph_from([], nodes=[1, "1"])
    empty = link_prediction(graph, [])
    assert list(empty.columns) == ["source", "target", "score"]
    assert str(empty.score.dtype) == "float64"
    for method in ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment"]:
        assert scores(graph, [(1, "1")], method) == [0.]


def test_torch_default_meta_context_does_not_redirect_cpu_algorithm_buffers():
    graph, partition, _ = weighted_bipartite()
    with torch.device("meta"):
        assert bipartite(graph, partition).partition.tolist() == _partition_ids(graph, partition)
        projected = bipartite_projection(graph, partition)
        assert projected._edges.device.type == "cpu"
        assert scores(graph, [(1, "1")], "common_neighbors") == [2.]
