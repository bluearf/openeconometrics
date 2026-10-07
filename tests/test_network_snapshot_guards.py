"""Public snapshot ownership and planned-buffer regression checks."""
from copy import deepcopy

import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_sparse import csr


def _snapshot(graph):
    return (graph.metadata, graph.node_attributes, graph.edge_attributes,
            graph.graph_attributes, graph._edges.indices().clone(),
            graph._edges.values().clone(), graph._arcs._indices().clone(),
            graph._arcs._values().clone())


def _assert_snapshot(graph, before):
    after = _snapshot(graph)
    assert after[:4] == before[:4]
    for actual, expected in zip(after[4:], before[4:]):
        assert torch.equal(actual, expected)


def test_attribute_clone_owns_input_without_accumulating_budget_or_mutating_source():
    source = oe.network({"source": [1, "1"], "target": ["1", "tail"]},
                        nodes=["isolate"])
    before = _snapshot(source)
    nodes = {1: {"kind": "integer"}, "1": {"kind": "string"}}
    edges = {("1", 1): {"name": "typed link"}}
    graph_attrs = {"title": "Original"}
    first = source.with_attributes(nodes=nodes, edges=edges,
                                   graph_attributes=graph_attrs)
    owned_bytes = first.metadata["estimated_owned_graph_bytes"]
    for _ in range(5):
        first = first.with_attributes()
        assert first.metadata["estimated_owned_graph_bytes"] == owned_bytes
        assert first._edges.values().data_ptr() == source._edges.values().data_ptr()
    nodes[1]["kind"] = "changed"
    edges[("1", 1)]["name"] = "changed"
    graph_attrs["title"] = "Changed"
    assert first.node_attributes == {1: {"kind": "integer"}, "1": {"kind": "string"}}
    assert first.edge_attributes == {(1, "1"): {"name": "typed link"}}
    assert first.graph_attributes == {"title": "Original"}
    _assert_snapshot(source, before)
    cleared = first.with_attributes(nodes={}, edges={}, graph_attributes={})
    assert cleared.node_attributes == cleared.edge_attributes == cleared.graph_attributes == {}
    assert cleared.metadata["estimated_owned_graph_bytes"] < owned_bytes
    assert first.node_attributes[1]["kind"] == "integer"


def test_over_budget_attributes_fail_before_validating_and_copying_large_scalar_payload(monkeypatch):
    import openecon._network_io as graph_io

    graph = oe.network({"source": [1], "target": [2]}, max_memory_mb=.02)
    before = _snapshot(graph)
    copied = []
    original = graph_io._attributes

    def observe_copy(value):
        copied.append(value)
        return original(value)

    monkeypatch.setattr(graph_io, "_attributes", observe_copy)
    with pytest.raises(AnalysisError) as error:
        graph.with_attributes(nodes={1: {"large": "x" * 16_000}})
    assert error.value.code == "network_memory_budget"
    assert not copied, "Attribute size must be checked before scalar UTF-8 validation and copying."
    _assert_snapshot(graph, before)


def test_plot_custom_groups_preflights_full_edge_chunk_even_when_sample_is_tiny():
    rows = ({"source": u, "target": v}
            for u in range(100) for v in range(u + 1, 100))
    graph = oe.network(rows, nodes=range(100), max_memory_mb=1)
    graph = graph.with_attributes(nodes={i: {"value": "a" * 200} for i in range(100)})
    before = _snapshot(graph)
    groups = {i: 0 for i in range(100)}
    with pytest.raises(AnalysisError) as error:
        graph.to_plot_data(max_nodes=1, max_edges=1, groups=groups)
    assert error.value.code == "network_memory_budget"
    assert groups == {i: 0 for i in range(100)}
    _assert_snapshot(graph, before)


@pytest.mark.parametrize("options", [
    {}, {"reverse": True}, {"loops": False},
    {"undirected": True}, {"reverse": True, "loops": False, "undirected": True},
])
def test_csr_budget_rejection_precedes_tensor_sort_or_projection(monkeypatch, options):
    graph = oe.network({"source": [1, 2, 2], "target": [2, 1, 2]}, directed=True)
    before = _snapshot(graph)
    graph._budget.limit = graph._base_bytes + 4096

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("CSR allocation ran before the graph's memory preflight.")

    monkeypatch.setattr(torch, "cat", unexpected_allocation)
    monkeypatch.setattr(torch, "argsort", unexpected_allocation)
    with pytest.raises(AnalysisError) as error:
        csr(graph, **options)
    assert error.value.code == "network_memory_budget"
    _assert_snapshot(graph, before)


def test_derived_graph_retains_scalar_attributes_and_exact_weighted_typed_identity():
    graph = oe.network({"source": [1, 1, "1", 7], "target": ["1", "1", 7, 7],
                        "weight": [1., 2., 4., 5.]}, nodes=["isolate"], weight="weight")
    graph = graph.with_attributes(nodes={1: {"kind": "int"}, "1": {"kind": "str"},
                                         "isolate": {"kind": "isolated"}},
                                 edges={(1, "1"): {"name": "aggregate"}, (7, 7): {"name": "loop"}},
                                 graph_attributes={"title": "Complete graph"})
    before = _snapshot(graph)
    selection = ["isolate", "1", 1, 1]
    child = graph.subgraph(iter(selection))
    assert set(child._labels) == {1, "1", "isolate"}
    assert child.weighted and child.edge_count == 1
    assert child._edges.values().tolist() == [3.]
    assert child.node_attributes == graph.node_attributes
    assert child.edge_attributes == {(1, "1"): {"name": "aggregate"}}
    assert child.graph_attributes == {"title": "Complete graph"}
    assert dict(zip(child.shortest_paths(1).node, child.shortest_paths(1).distance)) == {
        1: 0., "1": 3., "isolate": float("inf")}
    groups = {1: 8, "1": "8", "isolate": 8}
    saved_groups = deepcopy(groups)
    data = child.to_plot_data(groups=groups)
    assert data["grouping"] == "User groups"
    by_id = {child._labels[node["id"]]: node["group"] for node in data["nodes"]}
    assert by_id[1] == by_id["isolate"] != by_id["1"]
    assert groups == saved_groups
    _assert_snapshot(graph, before)
