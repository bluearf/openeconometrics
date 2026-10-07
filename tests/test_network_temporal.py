"""Independent set/Fraction/explicit-time-state oracles for sparse snapshots."""
from collections import OrderedDict
from fractions import Fraction
import math
import random

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon._network_temporal import NetworkSnapshots, network_snapshots
from openecon.analysis_contracts import AnalysisError


def graph(records=(), *, nodes=(), directed=False, reverse=False, memory=256):
    records, nodes = list(records), list(nodes)
    if reverse:
        records.reverse()
        nodes.reverse()
    return oe.network([{"source": u, "target": v, "weight": w} for u, v, w in records],
        weight="weight", nodes=nodes, directed=directed, batch_rows=2, max_memory_mb=memory)


def key(value):
    return (0, value) if isinstance(value, int) else (1, value)


def raw_edges(records, directed, loops):
    result = {}
    for u, v, weight in records:
        if not weight or (not loops and u == v):
            continue
        if not directed and key(u) > key(v):
            u, v = v, u
        result[u, v] = result.get((u, v), Fraction(0)) + Fraction(weight)
    return result


def oracle_path(records, nodes, directed, source, target, start=0):
    """Explicit small time-expanded state propagation, using full node loops."""
    active = {source}
    if source == target:
        return start
    for position in range(start, len(records)):
        active &= set(nodes[position])
        before = set(active)
        edges = raw_edges(records[position], directed, False)
        for u in nodes[position]:
            for v in nodes[position]:
                if u not in before:
                    continue
                pair = (u, v) if directed or key(u) <= key(v) else (v, u)
                if pair in edges:
                    active.add(v)
        if target in active:
            return position
    return None


def validate_path(result, records, nodes, directed, source, target, start):
    arrival = oracle_path(records, nodes, directed, source, target, start)
    assert result.attrs["reachable"] == (arrival is not None)
    assert result.attrs["arrival_position"] == arrival
    if arrival is None:
        assert result.empty and result.attrs["hops"] is None
        return
    assert result.attrs["hops"] == len(result)
    current, last = source, start - 1
    for row in result.itertuples(index=False):
        assert row.source == current
        assert last < row.position <= arrival
        for position in range(max(start, last + 1), row.position + 1):
            assert current in nodes[position]
        pair = ((row.source, row.target) if directed or key(row.source) <= key(row.target)
                else (row.target, row.source))
        assert pair in raw_edges(records[row.position], directed, False)
        current, last = row.target, row.position
    assert current == target
    if source != target:
        assert last == arrival


RECORDS = [
    [(1, "1", 2), ("1", "x", 3), (1, 1, 7), ("x", "1", 1)],
    [(1, "1", 4), (1, "x", 5), ("x", "x", 2)],
    [("1", "x", 8), ("x", 3, 2), ("1", "1", 11)],
    [(1, "1", 1), ("1", "x", 1)],
]
NODES = [[1, "1", "x", "isolate"], [1, "1", "x"],
         ["1", "x", 3, "new isolate"], [1, "1", "x", "isolate"]]
KEYS = [2026, "2026", "late", "early"]


def collection(*, directed=False, reverse=False, ordered=True):
    return network_snapshots(OrderedDict((name, graph(records, nodes=nodes, directed=directed,
        reverse=reverse)) for name, records, nodes in zip(KEYS, RECORDS, NODES)), ordered=ordered)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("loops", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_layer_summary_and_transitions_match_independent_sets(directed, loops, reverse):
    series = collection(directed=directed, reverse=reverse)
    summary = series.snapshot_summary(include_loops=loops)
    transitions = series.transitions(include_loops=loops)
    assert summary.layer.tolist() == KEYS
    assert isinstance(series, NetworkSnapshots)
    assert summary.position.tolist() == list(range(4))
    assert series.layer_keys == tuple(KEYS)
    assert series.node_count == len(set().union(*map(set, NODES)))
    sets = [set(raw_edges(records, directed, loops)) for records in RECORDS]
    for position, row in enumerate(summary.itertuples(index=False)):
        n = len(NODES[position])
        dyads = (n * (n - 1) if directed else n * (n - 1) // 2) + (n if loops else 0)
        assert row.nodes == n and row.edges == len(sets[position])
        assert row.eligible_dyads == dyads
        assert row.density == len(sets[position]) / dyads
    for position, row in enumerate(transitions.itertuples(index=False), 1):
        before, after = sets[position - 1], sets[position]
        previous_nodes, nodes = set(NODES[position - 1]), set(NODES[position])
        assert row.nodes_added == len(nodes - previous_nodes)
        assert row.nodes_removed == len(previous_nodes - nodes)
        assert row.nodes_persisted == len(nodes & previous_nodes)
        assert row.edges_added == len(after - before)
        assert row.edges_removed == len(before - after)
        assert row.edges_persisted == len(before & after)
        assert row.edge_jaccard == len(before & after) / len(before | after)
        assert row.edge_retention == len(before & after) / len(before)
        assert row.edge_turnover == len(before ^ after) / len(before | after)
    assert transitions.attrs["weight_changes"] == "ignored; presence only"
    assert "\\begin{tabular}" in transitions.to_latex(index=False)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("loops", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_edge_persistence_exact_union_counts_and_variable_node_eligibility(directed, loops, reverse):
    series = collection(directed=directed, reverse=reverse)
    result = series.edge_persistence(include_loops=loops)
    sets = [set(raw_edges(records, directed, loops)) for records in RECORDS]
    union = set().union(*sets)
    assert len(result) == len(union)
    assert list(zip(result.source, result.target)) == sorted(union, key=lambda pair: (key(pair[0]), key(pair[1])))
    for row in result.itertuples(index=False):
        pair = (row.source, row.target)
        positions = [i for i, edges in enumerate(sets) if pair in edges]
        eligible = sum(row.source in nodes and row.target in nodes for nodes in NODES)
        consecutive = longest = 0
        for edges in sets:
            consecutive = consecutive + 1 if pair in edges else 0
            longest = max(longest, consecutive)
        assert row.present_snapshots == len(positions)
        assert row.eligible_snapshots == eligible
        assert row.persistence == len(positions) / 4
        assert row.conditional_persistence == len(positions) / eligible
        assert row.longest_run == longest
        assert row.first_layer == KEYS[positions[0]] and row.last_layer == KEYS[positions[-1]]
        assert row.first_position == positions[0] and row.last_position == positions[-1]
    assert result.attrs["node_presence_storage"] == "compressed disjoint presence intervals"


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("loops", [False, True])
@pytest.mark.parametrize("reducer", ["sum", "mean", "max", "binary"])
@pytest.mark.parametrize("reverse", [False, True])
def test_union_aggregate_matches_independent_fraction_oracle(directed, loops, reducer, reverse):
    series = collection(directed=directed, reverse=reverse)
    result = series.aggregate(reducer=reducer, include_loops=loops)
    edges = [raw_edges(records, directed, loops) for records in RECORDS]
    pairs = set().union(*map(set, edges))
    indices, weights = result._edges.indices().tolist(), result._edges.values().tolist()
    actual = {(result._labels[u], result._labels[v]): weight
              for u, v, weight in zip(*indices, weights)}
    assert set(actual) == pairs
    for pair in pairs:
        values = [mapping.get(pair, Fraction(0)) for mapping in edges]
        expected = (sum(values) if reducer == "sum" else sum(values) / 4 if reducer == "mean"
                    else max(values) if reducer == "max" else 1.)
        assert actual[pair] == float(expected)
    assert result.directed == directed and result.weighted == (reducer != "binary")
    assert set(result._labels) == set().union(*map(set, NODES))
    assert result._edges.device.type == "cpu"
    assert result.metadata["mean_denominator"] == "all snapshots, including node/edge absences"
    assert result.metadata["attributes"].startswith("not merged")


def test_capture_order_and_mapping_edits_metadata_and_source_snapshots_remain_independent():
    first, second = graph([(1, "1", 2)], nodes=[1, "1", 3]), graph([(3, "1", 7)])
    mapping = OrderedDict([(5, first), ("5", second)])
    old = [(item._labels, item._edges.clone(), item.metadata) for item in (first, second)]
    series = network_snapshots(mapping)
    mapping.clear()
    mapping["replacement"] = graph([(0, 1, 1)])
    assert series.layer_keys == (5, "5") and series.snapshot_count == 2
    assert series.metadata["distinct_resident_snapshots"] == 2
    changed = series.metadata
    changed["ordered"] = True
    assert not series.ordered
    series.snapshot_summary()
    series.transitions()
    series.edge_persistence()
    series.aggregate()
    assert len(series.summary()) == 7 and "\\begin{tabular}" in series.to_latex()
    assert "NetworkSnapshots(" in repr(series)
    for item, (labels, edges, metadata) in zip((first, second), old):
        assert item._labels == labels and item.metadata == metadata
        assert torch.equal(item._edges.indices(), edges.indices())
        assert torch.equal(item._edges.values(), edges.values())


@pytest.mark.parametrize("directed", [False, True])
def test_temporal_paths_follow_insertion_positions_not_aggregate_reachability(directed):
    # The aggregate has A-B-C. B-C happened BEFORE A-B, so A cannot reach C.
    series = network_snapshots(OrderedDict([
        ("later looking key", graph([("b", "c", 1)], nodes=["a", "b", "c"], directed=directed)),
        ("earlier looking key", graph([("a", "b", 1)], nodes=["a", "b", "c"], directed=directed)),
    ]), ordered=True)
    unreachable = series.temporal_path("a", "c")
    assert unreachable.empty and not unreachable.attrs["reachable"]
    assert unreachable.attrs["arrival_layer"] is None and unreachable.attrs["hops"] is None
    assert series.aggregate().shortest_path("a", "c").node.tolist() == ["a", "b", "c"]
    reachable = series.temporal_path("a", "b")
    assert reachable.layer.tolist() == ["earlier looking key"]
    assert reachable.attrs["arrival_position"] == 1


@pytest.mark.parametrize("directed", [False, True])
def test_temporal_path_synchronous_contact_rule_waiting_and_zero_hop(directed):
    records = [[("a", "b", 1), ("b", "c", 1)], [], [("b", "c", 1)]]
    nodes = [["a", "b", "c"]] * 3
    series = network_snapshots({i: graph(edges, nodes=nodes[i], directed=directed)
                               for i, edges in enumerate(records)}, ordered=True)
    path = series.temporal_path("a", "c")
    validate_path(path, records, nodes, directed, "a", "c", 0)
    assert path.position.tolist() == [0, 2]
    identical = series.temporal_path("a", "a", start=1)
    assert identical.empty and identical.attrs["reachable"] and identical.attrs["hops"] == 0
    assert identical.attrs["arrival_position"] == 1


@pytest.mark.parametrize("directed", [False, True])
def test_temporal_path_does_not_wait_through_unobserved_node(directed):
    series = network_snapshots({
        0: graph([("a", "b", 1)], nodes=["c"], directed=directed),
        1: graph([], nodes=["a", "c"], directed=directed),
        2: graph([("b", "c", 1)], nodes=["a"], directed=directed),
    }, ordered=True)
    result = series.temporal_path("a", "c")
    assert not result.attrs["reachable"] and result.empty


def test_temporal_path_reappearing_node_can_be_reached_by_new_contact_and_witness_cycle():
    records = [[("a", "b", 1)], [("b", "c", 1)], [("c", "a", 1)], [("a", "z", 1)]]
    nodes = [["a", "b", "z"], ["b", "c", "z"], ["c", "a", "z"], ["a", "z"]]
    series = network_snapshots({i: graph(edges, nodes=nodes[i], directed=True)
                               for i, edges in enumerate(records)}, ordered=True)
    result = series.temporal_path("a", "z")
    validate_path(result, records, nodes, True, "a", "z", 0)
    assert result.source.tolist() == ["a", "b", "c", "a"]
    assert result.position.tolist() == list(range(4))


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(25))
def test_temporal_path_random_variable_nodes_matches_explicit_time_state_oracle(directed, seed):
    rng = random.Random(seed)
    labels = [1, "1", "a", 3, "z"]
    records, nodes = [], []
    for position in range(5):
        present = [label for label in labels if rng.random() < .8]
        if position == 0 and 1 not in present:
            present.append(1)
        nodes.append(present)
        records.append([(u, v, rng.randint(1, 9)) for u in present for v in present if rng.random() < .2])
    union = set().union(*map(set, nodes))
    series = network_snapshots({i: graph(edges, nodes=nodes[i], directed=directed)
                               for i, edges in enumerate(records)}, ordered=True)
    for target in union:
        for start in range(5):
            if 1 not in nodes[start]:
                continue
            result = series.temporal_path(1, target, start=start)
            validate_path(result, records, nodes, directed, 1, target, start)


def test_temporal_ties_are_exact_typed_label_and_import_order_invariant():
    records = [[("source", "1", 9), ("source", 1, 1)], [("1", "target", 1), (1, "target", 9)]]
    nodes = ["source", "target", 1, "1"]
    first = network_snapshots({i: graph(rows, nodes=nodes, directed=True) for i, rows in enumerate(records)}, ordered=True)
    reverse = network_snapshots({i: graph(rows, nodes=nodes, directed=True, reverse=True)
                                for i, rows in enumerate(records)}, ordered=True)
    expected, actual = first.temporal_path("source", "target"), reverse.temporal_path("source", "target")
    pd.testing.assert_frame_equal(expected, actual)
    assert expected.target.tolist() == [1, "target"]


@pytest.mark.parametrize("reducer", ["sum", "mean"])
@pytest.mark.parametrize("weights", [[.1, .2, .3], [1e308, 5e-324], [5e-324, 5e-324], [1., math.nextafter(1., 2.)]])
def test_aggregate_exact_binary64_accumulation_rounds_only_final_scalar(reducer, weights):
    series = network_snapshots({i: graph([(0, 1, value)]) for i, value in enumerate(weights)})
    expected = sum(map(Fraction, weights)) / (len(weights) if reducer == "mean" else 1)
    result = series.aggregate(reducer=reducer)
    assert result._edges.values()[0].item() == float(expected)
    assert "round only final scalar export" in result.metadata["arithmetic"]


def test_aggregate_true_overflow_is_refused_but_finite_mean_from_large_exact_sum_succeeds():
    first = graph([(0, 1, 1e308)])
    series = network_snapshots({0: first, 1: first})
    with pytest.raises(AnalysisError) as error:
        series.aggregate(reducer="sum")
    assert error.value.code == "network_precision"
    assert series.aggregate(reducer="mean")._edges.values()[0].item() == 1e308


def test_aggregate_positive_mean_underflow_is_not_silently_dropped():
    series = network_snapshots({0: graph([(0, 1, 5e-324)]), 1: graph(nodes=[0, 1])})
    with pytest.raises(AnalysisError) as error:
        series.aggregate(reducer="mean")
    assert error.value.code == "network_precision"
    assert series.aggregate(reducer="binary").edge_count == 1


@pytest.mark.parametrize("directed", [False, True])
def test_empty_snapshots_and_empty_turnover_conventions(directed):
    series = network_snapshots({0: graph(directed=directed), "0": graph(nodes=[1, "1"], directed=directed)}, ordered=True)
    result = series.transitions()
    assert result.edge_jaccard.tolist() == [1.] and result.edge_retention.tolist() == [1.]
    assert result.edge_turnover.tolist() == [0.]
    assert series.snapshot_summary().density.tolist() == [0., 0.]
    assert series.edge_persistence(max_edges=0).empty
    assert series.aggregate(max_edges=0).edge_count == 0
    assert set(series.aggregate()._labels) == {1, "1"}
    single = network_snapshots({"one": graph(directed=directed)})
    assert single.transitions().empty


@pytest.mark.parametrize("layers,options,code", [
    ({}, {}, "network_invalid_option"), ([], {}, "network_invalid_option"),
    ({"a": None}, {}, "network_invalid_option"), ({True: "dummy"}, {}, "network_invalid_label"),
    ({1.5: "dummy"}, {}, "network_invalid_label"), ({" ": "dummy"}, {}, "network_invalid_label"),
    ({"a": "dummy"}, {"ordered": 1}, "network_invalid_option"),
    ({"a": "dummy"}, {"max_work": 0}, "network_invalid_option"),
    ({"a": "dummy"}, {"max_work": 1}, "network_work_budget"),
])
def test_invalid_collection_input_is_explicit(layers, options, code):
    if isinstance(layers, dict):
        layers = {name: graph([(0, 1, 1)]) if value == "dummy" else value for name, value in layers.items()}
    with pytest.raises(AnalysisError) as error:
        network_snapshots(layers, **options)
    assert error.value.code == code


def test_mixed_directedness_and_unordered_causal_traversal_are_refused():
    with pytest.raises(AnalysisError) as error:
        network_snapshots({0: graph([(0, 1, 1)]), 1: graph([(0, 1, 1)], directed=True)})
    assert error.value.code == "network_invalid_option"
    with pytest.raises(AnalysisError) as error:
        network_snapshots({0: graph([(0, 1, 1)])}).temporal_path(0, 1)
    assert error.value.code == "network_invalid_option"


@pytest.mark.parametrize("method,options", [
    ("snapshot_summary", {"include_loops": 1}), ("transitions", {"include_loops": None}),
    ("edge_persistence", {"max_edges": True}), ("edge_persistence", {"max_edges": -1}),
    ("aggregate", {"reducer": "median"}), ("aggregate", {"reducer": []}),
    ("aggregate", {"include_loops": 1}), ("aggregate", {"max_edges": 1.5}),
    ("temporal_path", {"source": 0., "target": 1}), ("temporal_path", {"source": 0, "target": True}),
    ("temporal_path", {"source": 0, "target": 1, "start": False}),
])
def test_invalid_analysis_options_are_explicit(method, options):
    series = network_snapshots({0: graph([(0, 1, 1)])}, ordered=True)
    with pytest.raises(AnalysisError) as error:
        getattr(series, method)(**options)
    assert error.value.code in {"network_invalid_option", "network_invalid_label"}


@pytest.mark.parametrize("options", [{"source": 7, "target": 1}, {"source": 0, "target": 7},
    {"source": 0, "target": 1, "start": 9}, {"source": 0, "target": 1, "start": "0"},
    {"source": 2, "target": 1}])
def test_temporal_unknown_exact_node_or_layer_and_absent_starting_source(options):
    series = network_snapshots({0: graph([(0, 1, 1)]), 1: graph(nodes=[2])}, ordered=True)
    with pytest.raises(AnalysisError) as error:
        series.temporal_path(**options)
    assert error.value.code == "network_invalid_label"


@pytest.mark.parametrize("method,options", [
    ("snapshot_summary", {}), ("transitions", {}), ("edge_persistence", {}),
    ("aggregate", {}), ("temporal_path", {"source": 1, "target": "x"}),
])
def test_whole_operation_work_guard_precedes_output_and_tensor_allocations(method, options, monkeypatch):
    series = collection()
    def never(*args, **kwargs):
        pytest.fail("Work rejection must precede new output tensors/DataFrames.")
    monkeypatch.setattr(torch, "empty", never)
    monkeypatch.setattr(pd, "DataFrame", never)
    with pytest.raises(AnalysisError) as error:
        getattr(series, method)(max_work=1, **options)
    assert error.value.code == "network_work_budget"


@pytest.mark.parametrize("method", ["edge_persistence", "aggregate"])
def test_union_output_cardinality_limit_never_returns_partial_result(method):
    series = network_snapshots({0: graph([(0, 1, 1)]), 1: graph([(1, 2, 1)])})
    with pytest.raises(AnalysisError) as error:
        getattr(series, method)(max_edges=1)
    assert error.value.code == "network_output_budget"


def test_constructor_counts_distinct_resident_graphs_under_every_budget():
    first, second = graph([(0, 1, 1)]), graph([(1, 2, 1)])
    expected_peak = first._base_bytes + second._base_bytes + 1024 * 4 + 4096 + 2 * (256 + 2 * 28) + 512 * 2 + 8192
    first._budget.limit = expected_peak - 1
    with pytest.raises(AnalysisError) as error:
        network_snapshots({0: first, 1: second})
    assert error.value.code == "network_memory_budget"
    first._budget.limit = 256 * 1024**2
    second._budget.limit = expected_peak - 1
    with pytest.raises(AnalysisError) as error:
        network_snapshots({0: first, 1: second})
    assert error.value.code == "network_memory_budget"


def test_same_snapshot_reference_is_counted_once_not_once_per_layer():
    first = graph([(0, 1, 1)])
    peak = first._base_bytes + 1024 * 2 + 4096 + 2 * (256 + 2 * 28) + 512 * 2 + 8192
    first._budget.limit = peak
    series = network_snapshots({0: first, 1: first})
    assert series.metadata["estimated_resident_graph_bytes"] == first._base_bytes
    assert series.metadata["distinct_resident_snapshots"] == 1


@pytest.mark.parametrize("method,options", [
    ("snapshot_summary", {}), ("transitions", {}), ("edge_persistence", {}),
    ("aggregate", {}), ("temporal_path", {"source": 1, "target": "x"}),
])
def test_every_operation_counts_all_sources_against_smallest_budget(method, options):
    series = collection()
    graph = series._unique_graphs[-1]
    graph._budget.limit = series._resident_bytes + series._alignment_bytes + 4096
    with pytest.raises(AnalysisError) as error:
        getattr(series, method)(**options)
    assert error.value.code == "network_memory_budget"


def test_sparse_large_node_universe_default_meta_device_and_no_dense_time_arrays(monkeypatch):
    n = 10_000
    layers = {i: graph([(0, 1, i + 1), (1, 2, 1)], nodes=range(n)) for i in range(3)}
    series = network_snapshots(layers, ordered=True)
    original_empty, original_zeros = torch.empty, torch.zeros
    def bounded(constructor):
        def allocate(*args, **kwargs):
            shape = args[0] if args else kwargs.get("size", ())
            size = math.prod(shape) if isinstance(shape, (list, tuple)) else shape
            assert not isinstance(size, int) or size <= 4 * n
            return constructor(*args, **kwargs)
        return allocate
    monkeypatch.setattr(torch, "empty", bounded(original_empty))
    monkeypatch.setattr(torch, "zeros", bounded(original_zeros))
    with torch.device("meta"):
        assert series.snapshot_summary().edges.tolist() == [2, 2, 2]
        assert series.transitions().edges_persisted.tolist() == [2, 2]
        assert series.edge_persistence().present_snapshots.tolist() == [3, 3]
        assert series.aggregate()._edges.device.type == "cpu"
        assert series.temporal_path(0, 2).position.tolist() == [0, 1]


def test_physical_parquet_two_row_batches_snapshots_source_and_file_are_unchanged(tmp_path):
    path = tmp_path / "edges.parquet"
    pd.DataFrame(RECORDS[0], columns=["source", "target", "weight"]).assign(
        source=lambda frame: frame.source.astype(str), target=lambda frame: frame.target.astype(str)).to_parquet(path)
    before = path.read_bytes()
    dataset = oe.scan(path)
    snapshot = oe.network(dataset, weight="weight", directed=True, batch_rows=2)
    series = network_snapshots({"first": snapshot, "second": snapshot}, ordered=True)
    assert snapshot.metadata["actual_peak_batch_rows"] == 2
    assert series.edge_persistence(include_loops=True).present_snapshots.tolist() == [2] * snapshot.edge_count
    series.aggregate()
    dataset.assert_unchanged()
    assert path.read_bytes() == before


def test_stable_nodes_many_layers_compress_to_one_presence_interval_per_node():
    snapshot = graph([(i, i + 1, 1) for i in range(99)], nodes=range(100))
    series = network_snapshots({i: snapshot for i in range(200)})
    result = series.edge_persistence()
    assert result.eligible_snapshots.tolist() == [200] * 99
    assert result.longest_run.tolist() == [200] * 99
    assert series.metadata["distinct_resident_snapshots"] == 1
    # Eligibility is proportional to compressed runs, not time-edge cells.
    assert result.attrs["work_used"] < 4 * (200 * 100 + 200 * 99) + 3000
