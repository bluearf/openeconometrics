"""Code-first editing, typed identities, sparse budgets and saved timelines."""
from copy import deepcopy
import json

import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.networks import Network, network
from openecon._network_temporal import network_snapshots
import openecon_charts as charts


def graph(*, directed=False, max_memory_mb=256):
    return network({"source": [1, "1", "B", "B"], "target": ["1", "B", 1, "B"],
                    "w": [2., 3., 4., 5.]}, nodes=[2**80, "isolate"], weight="w",
                   directed=directed, max_memory_mb=max_memory_mb).with_attributes(
        nodes={1: {"country": "TR", "score": 5}, "1": {"country": "US", "score": 8}},
        edges={(1, "1"): {"year": 2020}, ("B", "B"): {"kind": "loop"}},
        graph_attributes={"study": "test"})


def dyads(value):
    return {(row.source, row.target): row.weight for row in value.edges().itertuples()}


@pytest.mark.parametrize("directed", [False, True])
def test_tables_preserve_typed_ids_isolates_loops_attributes_and_independence(directed):
    value = graph(directed=directed)
    nodes, edges = value.nodes(), value.edges()
    assert nodes.node.dtype == object and edges.source.dtype == object
    assert set(nodes.node) == {1, "1", "B", "isolate", 2**80}
    assert len(edges) == 4 and "attr.year" in edges and "attr.score" in nodes
    assert nodes.loc[nodes.node.map(lambda item: type(item) is int and item == 1), "attr.score"].item() == 5
    nodes.loc[:, "attr.score"] = 900
    edges.loc[:, "weight"] = 0
    assert value.node_attributes[1]["score"] == 5 and value.edge_count == 4
    assert not any(name.startswith("attr.") for name in value.nodes(attributes=False).columns)
    assert not any(name.startswith("attr.") for name in value.edges(attributes=False).columns)


@pytest.mark.parametrize("directed", [False, True])
def test_node_edits_are_simultaneous_and_preserve_attrs_and_source(directed):
    value = graph(directed=directed)
    before = value.to_plot_data()
    edited = value.edit_nodes(rename={1: "1", "1": 1}, add=["new"], remove=["isolate"])
    assert value.to_plot_data() == before
    assert edited.node_attributes["1"]["score"] == 5
    assert edited.node_attributes[1]["score"] == 8
    assert edited.graph_attributes == value.graph_attributes
    assert "new" in edited._index and "isolate" not in edited._index
    assert edited.edge_count == value.edge_count
    assert edited.metadata["derived_from"] == "immutable node edits"
    trimmed = value.edit_nodes(remove=["B"])
    assert trimmed.edge_count == 1 and trimmed.edge_attributes == {(1, "1"): {"year": 2020}}


@pytest.mark.parametrize("changes", [
    {"rename": {1: "B"}}, {"rename": {True: "C"}}, {"rename": {"missing": "C"}},
    {"remove": ["missing"]}, {"remove": [1], "rename": {1: "A"}},
    {"add": [1]}, {"add": ["C", "C"]}, {"add": "C"}, {"rename": []},
])
def test_invalid_node_edits_do_not_change_source(changes):
    value = graph()
    before = value.to_plot_data()
    with pytest.raises(AnalysisError):
        value.edit_nodes(**changes)
    assert value.to_plot_data() == before


@pytest.mark.parametrize("directed", [False, True])
def test_edge_edits_have_explicit_add_replace_remove_and_attribute_semantics(directed):
    value = graph(directed=directed)
    before = value.to_plot_data()
    edited = value.edit_edges(weights={(1, "1"): 10}, remove=[("B", "B")],
                             add=[{"source": "isolate", "target": "B", "weight": 2,
                                   "attrs": {"kind": "new"}},
                                  {"source": "isolate", "target": "B", "weight": 3},
                                  {"source": "1", "target": "B", "weight": 7}])
    assert value.to_plot_data() == before
    assert dyads(edited)[(1, "1")] == 10 and dyads(edited)[("1", "B")] == 10
    new_pair = ("isolate", "B") if directed else ("B", "isolate")
    assert dyads(edited)[new_pair] == 5
    assert edited.edge_attributes[new_pair] == {"kind": "new"}
    assert ("B", "B") not in dyads(edited)
    assert edited.edge_attributes[(1, "1")] == {"year": 2020}
    zeroed = value.edit_edges(weights={(1, "1"): 0})
    assert zeroed.edge_count == 3 and (1, "1") not in zeroed.edge_attributes


@pytest.mark.parametrize("changes", [
    {"remove": [("B", "isolate")]}, {"weights": {("B", "isolate"): 4}},
    {"weights": {(1, "1"): -1}}, {"weights": {(1, "1"): float("nan")}},
    {"remove": [(1, "1")], "weights": {(1, "1"): 1}},
    {"add": [{"source": "unknown", "target": 1}]},
    {"add": [{"source": 1, "target": "1", "unknown": 2}]},
    {"add": [{"source": 1, "target": "1", "attrs": {"weight": 2}}]},
    {"add": [{"source": 1, "target": "1", "attrs": {"x": []}}]},
    {"add": [{"source": 1, "target": "1", "attrs": {"x": 1}},
             {"source": 1, "target": "1", "attrs": {"x": 2}}]},
    {"remove": "bad"}, {"add": {"source": 1, "target": "1"}},
])
def test_invalid_edge_edits_are_rejected_atomically(changes):
    value = graph()
    before = value.to_plot_data()
    with pytest.raises(AnalysisError):
        value.edit_edges(**changes)
    assert value.to_plot_data() == before


def test_attribute_column_operations_merge_swap_drop_without_mutating_original():
    value = graph()
    edited = value.update_attributes(nodes={1: {"new": "value"}}, edges={("1", 1): {"note": "ok"}},
                                     graph_attributes={"version": 2})
    assert edited.node_attributes[1] == {"country": "TR", "score": 5, "new": "value"}
    assert edited.edge_attributes[(1, "1")] == {"year": 2020, "note": "ok"}
    assert edited.graph_attributes == {"study": "test", "version": 2}
    swapped = edited.rename_attributes({"score": "country", "country": "score"})
    assert swapped.node_attributes[1]["country"] == 5
    assert swapped.node_attributes[1]["score"] == "TR"
    dropped = swapped.drop_attributes(["new", "country"]).rename_attributes({"year": "time"}, scope="edges")
    assert dropped.node_attributes[1] == {"score": "TR"}
    assert dropped.edge_attributes[(1, "1")] == {"time": 2020, "note": "ok"}
    assert dropped.drop_attributes("study", scope="graph").graph_attributes == {"version": 2}
    assert value.node_attributes[1] == {"country": "TR", "score": 5}
    with pytest.raises(AnalysisError):
        value.rename_attributes({"score": "country"})
    with pytest.raises(AnalysisError):
        value.update_attributes(edges={("isolate", "B"): {"x": 1}})
    sparse_columns = value.update_attributes(nodes={1: {"one": 1}, "B": {"two": 2}})
    with pytest.raises(AnalysisError, match="merge"):
        sparse_columns.rename_attributes({"one": "two"})


def test_python_callable_filters_and_declarative_filters_have_induced_topology():
    value = graph()
    filtered = value.filter(nodes={"field": "node", "op": "in", "value": [1, "1", "B"]},
                            edges={"field": "weight", "op": "gte", "value": 3})
    assert filtered.node_count == 3 and filtered.edge_count == 3
    assert (1, "1") not in filtered.edge_attributes
    only_string = value.filter(nodes={"field": "node", "op": "eq", "value": "1"})
    assert only_string._labels == ("1",)
    only_integer = value.filter(nodes={"field": "node", "op": "eq", "value": 1})
    assert only_integer._labels == (1,)
    assert value.filter(nodes={"field": "node", "op": "eq", "value": True}).node_count == 0
    by_attr = value.filter(nodes={"field": "attrs.score", "op": "gt", "value": 6})
    assert by_attr._labels == ("1",)
    def mutating(row):
        row["attrs"]["country"] = "MUTATED"
        return row["node"] != "B"
    called = value.filter(nodes=mutating)
    assert called.edge_count == 1 and value.node_attributes[1]["country"] == "TR"
    assert called.node_attributes[1]["country"] == "TR"
    with pytest.raises(AnalysisError, match="boolean"):
        value.filter(nodes=lambda row: "true")


@pytest.mark.parametrize("predicate", [
    {"field": "weight", "op": "eval", "value": "code"},
    {"field": "degree", "op": "gte", "value": float("nan")},
    {"field": "degree", "op": "gt", "value": "3"},
    {"field": "node", "op": "in", "value": [1] * 10001},
    {"field": "node", "op": "eq", "value": {}},
])
def test_invalid_declarative_filters_are_cheap_and_finite(predicate):
    with pytest.raises(AnalysisError):
        graph().filter(nodes=predicate)


def test_saved_positions_and_native_view_rebuild_preserve_identity_and_subset_provenance():
    value = graph()
    positioned = value.with_positions({label: (index * 5, index * 2) for index, label in enumerate(value._labels)})
    plot = charts.network(positioned, layout="fixed", max_nodes=3, max_edges=2)
    rebuilt = Network.from_plot_data(plot.config["network"])
    assert rebuilt.node_count == 3 and rebuilt.edge_count == 2
    assert rebuilt.metadata["source_node_count"] == value.node_count
    assert rebuilt.metadata["source_edge_count"] == value.edge_count
    assert rebuilt.metadata["source_sampled"]
    assert {type(label) for label in rebuilt._labels} == {int, str}
    assert all(attrs["fixed"] for attrs in rebuilt.node_attributes.values())
    full = Network.from_plot_data(positioned.to_plot_data())
    assert set(full._labels) == set(value._labels)
    assert full.node_attributes[2**80]["x"] == 0
    assert dyads(full) == dyads(value)
    assert value.node_attributes.get(2**80, {}) == {}
    raw = value.to_plot_data()
    raw["nodes"][0].pop("identity")
    with pytest.raises(AnalysisError, match="typed identities"):
        Network.from_plot_data(raw)


@pytest.mark.parametrize("positions", [
    {1: (float("nan"), 2)}, {1: (1e10, 2)}, {"missing": (1, 2)},
    {True: (1, 2)}, {1: {"x": 1}}, {1: {"x": 1, "y": 2, "fixed": 1}},
])
def test_invalid_positions_do_not_mutate_source(positions):
    value = graph()
    before = value.node_attributes
    with pytest.raises(AnalysisError):
        value.with_positions(positions)
    assert value.node_attributes == before


def test_timeline_has_stable_union_ids_and_full_frame_counts():
    first = graph()
    second = first.edit_nodes(remove=[1], add=["new"])
    snapshots = network_snapshots({"before": first, "after": second}, ordered=True)
    payload = charts.network(snapshots, timeline=True).config["network"]
    assert [frame["label"] for frame in payload["frames"]] == ["before", "after"]
    ids = {}
    for frame in payload["frames"]:
        data = frame["network"]
        assert not data["sampled"]
        for node in data["nodes"]:
            identity = tuple(node["identity"].values())
            assert ids.setdefault(identity, node["id"]) == node["id"]
        assert len(data["edges"]) == data["edge_count"]
    assert tuple(first._labels) == tuple(graph()._labels)
    with pytest.raises(AnalysisError, match="frame"):
        Network.from_plot_data(payload)
    assert Network.from_plot_data(payload["frames"][1]["network"]).node_count == second.node_count
    json.dumps(payload, allow_nan=False)


def test_timeline_rejects_unbounded_frames_and_combined_display_overflow():
    small = network({"source": [], "target": []}, nodes=[1])
    many = network_snapshots({index: small for index in range(61)})
    with pytest.raises(AnalysisError, match="60"):
        many.to_plot_data()
    large = network({"source": [], "target": []}, nodes=range(50_001), max_memory_mb=512)
    collection = network_snapshots({"a": large, "b": large})
    with pytest.raises(AnalysisError, match="aggregate"):
        collection.to_plot_data(max_nodes=100_000)


def test_edit_and_attribute_table_memory_guards_fail_before_expensive_allocation():
    value = graph()
    before = deepcopy(value.metadata)
    value._budget.limit = value._base_bytes + 2048
    for operation in (lambda: value.edit_nodes(add=["a"]), lambda: value.edit_edges(weights={(1, "1"): 2}),
                      lambda: value.nodes(), lambda: value.edges(), lambda: value.filter()):
        with pytest.raises(AnalysisError, match="memory"):
            operation()
    assert value.metadata == before


def test_empty_tables_and_empty_filtered_graph_are_well_formed():
    value = network({"source": [], "target": []})
    assert list(value.nodes().columns) == ["node", "degree", "strength"]
    assert list(value.edges().columns) == ["source", "target", "weight"]
    filtered = graph().filter(nodes=lambda row: False)
    assert filtered.node_count == filtered.edge_count == 0
    assert not filtered.to_plot_data()["sampled"]
    binary = network({"source": [1], "target": [2]})
    untouched = binary.edit_edges(add=iter(()))
    assert not untouched.weighted and not untouched.metadata["weighted"]
