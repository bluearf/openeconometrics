"""Pooled timelines preserve the public graph protocol and expanded guards."""
from copy import deepcopy
import json

import pytest

from openecon_charts import PlotSpec, network
from openecon_charts.timeline import unpack


def fixture():
    class Graph:
        def to_plot_data(self, **kwargs):
            base = {'nodes': [{'id': 0, 'label': '1', 'degree': 1, 'group': 0,
                               'identity': {'type': 'integer', 'value': '1'}, 'attrs': {'a': 'long label' * 8}, 'x': 10, 'y': 0},
                              {'id': 1, 'label': '1', 'degree': 1, 'group': 0,
                               'identity': {'type': 'string', 'value': '1'}, 'attrs': {'a': 'long label' * 8}, 'x': 20, 'y': 1}],
                    'edges': [{'source': 0, 'target': 1, 'weight': 1, 'attrs': {'a': 'value' * 20}}],
                    'directed': True, 'node_count': 3, 'edge_count': 5, 'shown_node_count': 2,
                    'shown_edge_count': 1, 'sampled': True, 'selection': 'Explicit subset'}
            second = deepcopy(base)
            second['nodes'][1]['x'] = 30
            second['edges'][0]['weight'] = 2
            second['node_count'] = 4
            base['frames'] = [{'label': 'first', 'network': deepcopy(base)},
                              {'label': 'second', 'network': second},
                              {'label': 'third', 'network': deepcopy(second)}]
            return base
    return network(Graph(), frame_index=1, layout='fixed')


def test_wire_roundtrip_preserves_types_counts_order_positions_attrs_and_selected_frame(tmp_path):
    plot = fixture()
    public = plot.model_dump()
    wire = plot.transport_dump()
    assert wire['config']['network']['encoding'] == 'timeline-pool-v1'
    assert len(json.dumps(wire)) < len(json.dumps(public)) * .7
    assert unpack(wire) == public
    assert PlotSpec.load_view(plot.save_view(tmp_path / 'timeline.json')).model_dump() == public
    assert plot.model_dump() == public
    assert 'timeline-pool-v1' in plot.to_html()


@pytest.mark.parametrize('change', ['bool', 'negative', 'range', 'unused', 'version', 'nested', 'direction', 'identity'])
def test_invalid_pools_and_semantic_changes_are_refused(change):
    wire = fixture().transport_dump()
    graph = wire['config']['network']
    if change in ('bool', 'negative', 'range'):
        graph['base']['nodes'][0] = {'bool': True, 'negative': -1, 'range': 99999}[change]
    elif change == 'unused':
        graph['nodes'].append(deepcopy(graph['nodes'][0]))
    elif change == 'version':
        graph['encoding'] = 'future'
    elif change == 'nested':
        graph['frames'][0]['network']['frames'] = []
    elif change == 'direction':
        graph['frames'][0]['network']['directed'] = False
    elif change == 'identity':
        graph['nodes'][2]['identity'] = {'type': 'string', 'value': 'different'}
    with pytest.raises((TypeError, ValueError)):
        unpack(wire)


def test_reference_amplification_rejected_before_expansion():
    wire = fixture().transport_dump()
    wire['config']['network']['base']['nodes'] = [0] * 100001
    with pytest.raises(ValueError, match='aggregate'):
        unpack(wire)
