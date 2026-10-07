"""The public graph API shares one snapshot and the ordinary result protocol."""
import json

import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_sparse import csr


def fixture():
    return oe.network({'source': [1, '1', 'A', 'B', 'C', 'C'],
                       'target': ['1', 'A', 1, 'C', 'B', 'C']}, nodes=['isolate'])


@pytest.mark.parametrize('name', ['betweenness', 'closeness', 'harmonic', 'eigenvector',
                                 'triangles', 'clustering', 'core_numbers'])
def test_public_metrics_keep_identity_and_frame_protocol(name):
    graph = fixture()
    before = graph._edges.clone()
    result = getattr(graph, name)()
    assert result.node.tolist() == list(graph._labels)
    assert isinstance(result, oe.DataFrame)
    assert '\\begin{tabular}' in result.to_latex()
    assert torch.equal(before.indices(), graph._edges.indices())
    assert torch.equal(before.values(), graph._edges.values())
    assert 1 in result.node.tolist() and '1' in result.node.tolist()


@pytest.mark.parametrize('directed', [False, True])
@pytest.mark.parametrize('method', ['louvain', 'leiden'])
def test_public_communities_modularity_and_membership(directed, method):
    graph = oe.network({'source': ['a', 'b', 'c'], 'target': ['b', 'c', 'a']}, directed=directed)
    result = graph.communities(method=method, seed=19)
    assert result.attrs['modularity'] == pytest.approx(graph.modularity(result), abs=1e-12)
    assert result.attrs['converged']
    payload = graph.to_plot_data(groups=result)
    assert payload['grouping'] == method.title() + ' communities'
    assert len({node['group'] for node in payload['nodes']}) == result.community.nunique()


def test_public_strong_components_and_core_threshold_graph():
    graph = oe.network({'source': ['a', 'b', 'b'], 'target': ['b', 'a', 'c']}, directed=True, nodes=['iso'])
    assert graph.components().component.nunique() == 2
    assert graph.components('strong').component.nunique() == 3
    assert graph.k_core(2).node_count == 0
    assert graph.k_core(1).node_count == 3
    assert graph.k_core().equals(graph.core_numbers())
    assert graph.k_core(1).weighted is False
    assert isinstance(graph.topology_summary(), oe.DataFrame)
    assert graph.density() == .25


def test_subgraph_preserves_typed_identity_strength_semantics_attributes_and_isolates():
    graph = fixture().with_attributes(nodes={1: {'category': 'integer'}}, edges={(1, '1'): {'label': 'link'}})
    child = graph.subgraph([1, '1', 'isolate'])
    assert child.node_count == 3 and child.edge_count == 1
    assert child.weighted is False
    assert child.node_attributes == {1: {'category': 'integer'}}
    assert child.edge_attributes == {(1, '1'): {'label': 'link'}}
    assert child.graph_attributes == {}
    assert child.metadata['parent_node_count'] == 6
    attrs = child.node_attributes
    attrs[1]['category'] = 'changed'
    assert child.node_attributes[1]['category'] == 'integer'
    distances = child.shortest_paths(1)
    assert dict(zip(distances.node, distances.distance)) == {1: 0., '1': 1., 'isolate': float('inf')}
    with pytest.raises(AnalysisError, match='unknown'):
        graph.subgraph(['missing'])


@pytest.mark.parametrize('groups', [{1: 0}, ['a'], {1: True, '1': 0, 'A': 0, 'B': 1, 'C': 1, 'isolate': 2}])
def test_plot_rejects_incomplete_or_malformed_membership(groups):
    with pytest.raises(AnalysisError):
        fixture().to_plot_data(groups=groups)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('loops', [False, True])
def test_compact_csr_projection_and_tensor_context(reverse, loops):
    graph = oe.network({'source': ['a', 'b', 'b'], 'target': ['b', 'a', 'b'], 'weight': [1e308]*3},
                       directed=True)
    original = graph._edges.values().clone()
    with torch.device('meta'):
        view = csr(graph, reverse=reverse, loops=loops, undirected=True)
    assert view.node_count == 2 and view.arc_count == (3 if loops else 2)
    assert all(tensor.device.type == 'cpu' for tensor in view._tensors)
    assert list(view.row(0)) == [(1, 2.)]
    assert view.storage_bytes <= 8*3+16*3
    assert torch.equal(original, graph._edges.values())


def test_empty_csr_and_membership():
    graph = oe.network({'source': [], 'target': []})
    view = csr(graph)
    assert view.node_count == 0 and view.arc_count == 0 and list(view.offsets) == [0]
    result = graph.communities()
    assert len(result) == 0 and graph.modularity(result) == 0
    assert json.dumps(graph.to_plot_data(groups=result), allow_nan=False)


def test_display_labels_use_attributes_without_changing_graph_identity():
    graph = fixture().with_attributes(nodes={1: {"label": "Integer node"}, "1": {"label": "String node"}})
    payload = graph.to_plot_data()
    names = {graph._labels[node["id"]]: node["label"] for node in payload["nodes"]}
    assert names[1] == "Integer node" and names["1"] == "String node"
    assert 1 in graph._index and "1" in graph._index
    assert graph.node_attributes[1]["label"] == "Integer node"
