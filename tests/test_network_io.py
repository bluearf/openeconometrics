"""Static graph interoperability fixtures, integrity and resource-bound checks."""
from copy import deepcopy
import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

import openecon._network_io as io
from openecon.analysis_contracts import AnalysisError
from openecon.networks import _key, network


def _edges(graph):
    result = {}
    for u, v, weight in io._edge_records(graph):
        pair = graph._labels[u], graph._labels[v]
        if not graph.directed and _key(pair[0]) > _key(pair[1]):
            pair = pair[::-1]
        result[pair] = weight
    return result


def _fixture_graph(directed, weighted, graph_attrs=False):
    unusual = '<日本&"\'\\node>'
    graph = network({'source': [1, 1, "1", unusual], 'target': ["1", "1", unusual, unusual],
                     'w': [1.25, .5, 2.25, .25]}, nodes=[1, "1", unusual, 2**200, 'isolate'],
                    weight='w' if weighted else None, directed=directed)
    return graph.with_attributes(
        nodes={1: {'enabled': True, 'rank': -(2**63), 'score': .125, 'text': 'A & <B> "C"\nD'},
               '1': {'enabled': False, 'rank': 2**63 - 1, 'score': -1.5, 'text': 'Türkçe 日本語'},
               unusual: {'label': 'Custom display'}},
        edges={(1, '1'): {'tag': 'First & second', 'valid': True},
               ('1', unusual): {'tag': 'Route <x>', 'valid': False}},
        graph_attributes={'title': 'Study & sample', 'reviewed': True, 'year': 2026} if graph_attrs else {})


@pytest.mark.parametrize('format', ['graphml', 'gexf', 'pajek'])
@pytest.mark.parametrize('directed', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('batch_rows', [1, 3])
def test_native_exact_typed_identity_weights_isolates_and_scalar_attribute_roundtrip(
        tmp_path, format, directed, weighted, batch_rows):
    graph = _fixture_graph(directed, weighted, graph_attrs=format == 'graphml')
    path = tmp_path / ('graph.net' if format == 'pajek' else 'graph.' + format)
    assert graph.write(path) == path
    result = io.read_network(path, batch_rows=batch_rows)
    assert list(result._labels) == list(graph._labels)
    assert len({(type(label), label) for label in result._labels}) == 5
    assert _edges(result) == _edges(graph)
    assert result.directed == graph.directed and result.weighted == graph.weighted
    assert result.node_attributes == graph.node_attributes
    assert result.edge_attributes == graph.edge_attributes
    assert result.graph_attributes == graph.graph_attributes
    assert result.metadata['max_memory_bytes'] == 256 * 1024**2
    assert result.metadata['estimated_import_peak_bytes'] <= result.metadata['max_memory_bytes']
    assert result.metadata['file_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.metadata['file_passes'] == 3
    if format == 'gexf':
        root = ET.parse(path).getroot()
        element = next(child for child in root if child.tag.endswith('}graph'))
        assert set(element.attrib) == {'mode', 'defaultedgetype'}
        assert next(root.iter('{http://gexf.net/1.3}keywords')).text == 'openecon.weighted=' + str(weighted).lower()


@pytest.mark.parametrize('format', ['graphml', 'gexf', 'pajek'])
def test_empty_and_isolate_roundtrip(tmp_path, format):
    for nodes in [[], [1, '1', 'empty']]:
        graph = network([], nodes=nodes)
        path = tmp_path / ('graph.' + format)
        io.write_network(graph, path, format=format, overwrite=True)
        result = io.read_network(path)
        assert list(result._labels) == nodes and result.edge_count == 0 and not result.weighted
        assert result.node_attributes == {} and result.edge_attributes == {}


def test_external_graphml_defaults_overrides_and_edges_before_nodes(tmp_path):
    text = '''<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <key id="color" for="node" attr.name="color" attr.type="string"><default>blue</default></key>
      <key id="active" for="node" attr.name="active" attr.type="boolean"><default>true</default></key>
      <key id="w" for="edge" attr.name="weight" attr.type="double"><default>2.5</default></key>
      <key id="group" for="all" attr.name="cohort" attr.type="string"><default>A</default></key>
      <key id="study" for="graph" attr.name="study" attr.type="string"><default>Default study</default></key>
      <key id="year" for="graph" attr.name="year" attr.type="long"><default>2025</default></key>
      <graph edgedefault="directed"><data key="study">Actual study</data>
        <edge source="x" target="y"><data key="w">3.75</data></edge>
        <node id="x"><data key="color">red</data><data key="active">false</data></node>
        <edge source="y" target="x"/>
        <node id="y"/><node id="isolate"/>
      </graph></graphml>'''
    path = tmp_path / 'external.graphml'
    path.write_text(text)
    result = io.read_network(path, batch_rows=1)
    assert _edges(result) == {('x', 'y'): 3.75, ('y', 'x'): 2.5}
    assert result.node_attributes == {
        'x': {'color': 'red', 'active': False, 'cohort': 'A'},
        'y': {'color': 'blue', 'active': True, 'cohort': 'A'},
        'isolate': {'color': 'blue', 'active': True, 'cohort': 'A'}}
    assert result.edge_attributes == {('x', 'y'): {'cohort': 'A'}, ('y', 'x'): {'cohort': 'A'}}
    assert result.graph_attributes == {'study': 'Actual study', 'year': 2025, 'cohort': 'A'}
    assert result.weighted and result.directed


@pytest.mark.parametrize('namespace', ['http://gexf.net/1.3', 'http://gexf.net/1.2draft',
                                        'http://www.gexf.net/1.2draft'])
def test_external_gexf_defaults_labels_and_reversed_container_order(tmp_path, namespace):
    text = f'''<gexf xmlns="{namespace}"><graph mode="static" defaultedgetype="undirected">
      <attributes class="node"><attribute id="0" title="color" type="string"><default>blue</default></attribute></attributes>
      <attributes class="edge"><attribute id="0" title="reviewed" type="boolean"><default>false</default></attribute></attributes>
      <edges><edge source="a" target="b" weight="2.5"><attvalues><attvalue for="0" value="true"/></attvalues></edge></edges>
      <nodes><node id="a" label="Alpha"/><node id="b" label="Beta"/><node id="c" label="Isolate"/></nodes>
      </graph></gexf>'''
    path = tmp_path / 'external.gexf'
    path.write_text(text)
    result = io.read_network(path)
    assert _edges(result) == {('a', 'b'): 2.5}
    assert result.node_attributes == {'a': {'color': 'blue', 'label': 'Alpha'},
                                      'b': {'color': 'blue', 'label': 'Beta'},
                                      'c': {'color': 'blue', 'label': 'Isolate'}}
    assert result.edge_attributes == {('a', 'b'): {'reviewed': True}}
    assert result.weighted and not result.directed


@pytest.mark.parametrize('section,directed', [('*Edges', False), ('*Arcs', True),
                                             ('*Edgeslist', False), ('*Arcslist', True)])
def test_external_pajek_sections_and_isolates(tmp_path, section, directed):
    path = tmp_path / 'external.net'
    path.write_text('*Vertices 4\n1 "Alpha"\n2 "Beta"\n3 "Gamma"\n4 "Isolate"\n' + section +
                    ('\n1 2 3\n2 3\n' if section.lower().endswith('list') else '\n1 2\n2 3\n'))
    result = io.read_network(path)
    assert result.directed == directed and not result.weighted
    assert result.node_count == 4 and result.edge_count == (3 if section.lower().endswith('list') else 2)


def test_external_pajek_visual_properties_and_optional_weights(tmp_path):
    path = tmp_path / 'external.net'
    path.write_text('*Vertices 2\n1 "Alpha" 0.1 0.2 ellipse\n2 "Beta"\n*Arcs\n1 2 2.5 c Blue\n2 1 c Red\n')
    result = io.read_network(path)
    first, second = result._labels
    assert _edges(result) == {(first, second): 2.5, (second, first): 1.}
    assert result.edge_attributes == {(first, second): {'pajek_visual': 'c Blue'},
                                      (second, first): {'pajek_visual': 'c Red'}}
    assert result.node_attributes[first]['pajek_visual'] == '0.1 0.2 ellipse'
    assert result.weighted


@pytest.mark.parametrize('format,text', [
    ('graphml', '<!DOCTYPE graphml [<!ENTITY x "bad">]><graphml><graph edgedefault="undirected"/></graphml>'),
    ('graphml', '<graphml><graph edgedefault="undirected"><node id="x"><graph edgedefault="undirected"/></node></graph></graphml>'),
    ('graphml', '<graphml><graph edgedefault="undirected"><hyperedge/></graph></graphml>'),
    ('graphml', '<graphml><graph edgedefault="undirected"><node id="x"><port name="p"/></node></graph></graphml>'),
    ('graphml', '<graphml><graph edgedefault="undirected"><node id="x"/><edge source="x" target="x" directed="true"/></graph></graphml>'),
    ('gexf', '<gexf><graph mode="dynamic"><nodes/></graph></gexf>'),
    ('gexf', '<gexf><graph><nodes><node id="a"><nodes><node id="b"/></nodes></node></nodes></graph></gexf>'),
    ('gexf', '<gexf><graph><nodes><node id="a" pid="b"/><node id="b"/></nodes></graph></gexf>'),
    ('gexf', '<gexf><graph><nodes><node id="a"><parents><parent for="b"/></parents></node></nodes></graph></gexf>'),
    ('gexf', '<gexf xmlns:viz="http://gexf.net/1.3/viz"><graph><nodes><node id="a"><viz:color r="1" g="2" b="3"/></node></nodes></graph></gexf>'),
    ('gexf', '<gexf><graph><nodes><node id="a" start="0"/></nodes></graph></gexf>'),
    ('gexf', '<gexf><graph defaultedgetype="mutual"/></gexf>'),
    ('pajek', '*Vertices 2\n1 "A"\n2 "B"\n*Edges\n1 2\n*Arcs\n2 1\n'),
    ('pajek', '*Vertices 2 1\n1 "A"\n2 "B"\n*Edges\n1 2\n'),
    ('pajek', '*Vertices 2\n1 "A"\n2 "B"\n*Matrix\n0 1\n1 0\n'),
])
def test_unsafe_dynamic_hierarchical_mixed_or_unsupported_features_fail(tmp_path, format, text):
    path = tmp_path / ('unsafe.' + format)
    path.write_text(text)
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path)
    assert caught.value.code in {'network_unsafe_xml', 'network_file_feature'}


@pytest.mark.parametrize('text', [
    '*Vertices 2\n1 "A"\n01 "B"\n*Edges\n1 01\n',
    '*Vertices 2\n1 "A"\n2 "B"\n% openecon-node 1 []\n*Edges\n1 2\n',
    '*Vertices 2\n1 "A"\n2 "B"\n% openecon-node 1 {"identity":"A","identity":"B"}\n*Edges\n1 2\n',
    '% openecon-weighted false\n% openecon-weighted false\n*Vertices 0\n*Edges\n',
    '*Vertices 2\n1 "A"\n2 "B"\n*Edges\n% openecon-edge {"tag":"x"}\n',
    '*Vertices 2\n1 "A"\n2 "B"\n*Edges\n% openecon-edge {"tag":"x"}\n% openecon-edge {"tag":"y"}\n1 2\n',
    '*Vertices 2\n1 "A"\n2 "B"\n*Edges\n% openecon-edge {"weight":5}\n1 2\n',
])
def test_pajek_ambiguous_or_malformed_native_metadata_is_rejected(tmp_path, text):
    path = tmp_path / 'invalid.net'
    path.write_text(text)
    with pytest.raises(AnalysisError):
        io.read_network(path)


def test_edge_weight_is_reserved_and_attributes_are_immutable_owned_snapshots():
    graph = network({'source': ['a'], 'target': ['b'], 'w': [2.]}, weight='w')
    with pytest.raises(AnalysisError, match='weight'):
        graph.with_attributes(edges={('a', 'b'): {'weight': 9.}})
    source = {'a': {'note': 'original'}}
    copied = graph.with_attributes(nodes=source)
    source['a']['note'] = 'changed'
    readback = copied.node_attributes
    readback['a']['note'] = 'changed again'
    assert copied.node_attributes == {'a': {'note': 'original'}}
    assert graph.node_attributes == {}
    assert copied.with_attributes(edges={('a', 'b'): {'tag': 'edge'}}).node_attributes == copied.node_attributes


@pytest.mark.parametrize('attrs', [{'bad\x85name': 'x'}, {'bad\ud800name': 'x'}, {'value': '\ud800'},
                                   {'value': 'x' * 16385}, {'value': 2**63}, {'value': float('nan')},
                                   {'value': []}, {str(i): i for i in range(129)}])
def test_invalid_scalar_attributes_raise_structured_errors(attrs):
    graph = network([], nodes=['a'])
    with pytest.raises(AnalysisError) as caught:
        graph.with_attributes(nodes={'a': attrs})
    assert caught.value.code == 'network_file_attribute'


def test_attribute_copy_preflight_runs_before_encoded_serialization(monkeypatch):
    graph = network([], nodes=['a'])
    graph._budget.limit = graph._base_bytes + 5000
    monkeypatch.setattr(io, '_attributes', lambda *args: pytest.fail('Copy occurs after budget preflight'))
    with pytest.raises(AnalysisError) as caught:
        io.attach_attributes(graph, nodes={'a': {'text': 'x' * 16384}})
    assert caught.value.code == 'network_memory_budget'


@pytest.mark.parametrize('format', ['graphml', 'gexf', 'pajek'])
def test_atomic_overwrite_and_failed_export_preserve_existing_file(tmp_path, format, monkeypatch):
    path = tmp_path / ('graph.' + format)
    original = b'existing source remains intact'
    path.write_bytes(original)
    graph = network({'source': ['a'], 'target': ['b']})
    with pytest.raises(FileExistsError):
        io.write_network(graph, path)
    assert path.read_bytes() == original and not list(tmp_path.glob('.openecon-network-*'))
    real = io._write_pajek if format == 'pajek' else io._write_xml

    def partial(*args):
        args[1].write('partial\n')
        raise RuntimeError('injected write failure')

    monkeypatch.setattr(io, '_write_pajek' if format == 'pajek' else '_write_xml', partial)
    with pytest.raises(RuntimeError, match='injected'):
        io.write_network(graph, path, overwrite=True)
    assert path.read_bytes() == original and not list(tmp_path.glob('.openecon-network-*'))
    monkeypatch.setattr(io, '_write_pajek' if format == 'pajek' else '_write_xml', real)
    io.write_network(graph, path, overwrite=True)
    assert _edges(io.read_network(path)) == _edges(graph)


@pytest.mark.parametrize('format', ['gexf', 'pajek'])
def test_formats_without_scalar_graph_attrs_refuse_loss_atomically(tmp_path, format):
    graph = network([], nodes=['a']).with_attributes(graph_attributes={'study': 'preserve me'})
    path = tmp_path / ('graph.' + format)
    path.write_bytes(b'original')
    with pytest.raises(AnalysisError) as caught:
        io.write_network(graph, path, overwrite=True)
    assert caught.value.code == 'network_file_feature'
    assert path.read_bytes() == b'original' and not list(tmp_path.glob('.openecon-network-*'))


def test_global_xml_field_limit_refuses_unreadable_native_export(tmp_path):
    graph = network([], nodes=list(range(4))).with_attributes(
        nodes={i: {f'field_{i}_{j}': j for j in range(128)} for i in range(4)})
    with pytest.raises(AnalysisError, match='387'):
        io.write_network(graph, tmp_path / 'too_many.graphml')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('value', ['text', 'attribute'])
def test_huge_xml_parser_token_is_rejected_before_whole_file_read(tmp_path, monkeypatch, value):
    path = tmp_path / 'large.graphml'
    huge = 'x' * 2_000_000
    path.write_text('<graphml><graph edgedefault="undirected">' +
                    ('<node id="' + huge + '"/>' if value == 'attribute' else
                     '<node id="x"><data key="q">' + huge + '</data></node>') + '</graph></graphml>')
    captured = []
    real = io._Reader

    class Reader(real):
        def __init__(self, *args):
            super().__init__(*args)
            captured.append(self)

    monkeypatch.setattr(io, '_Reader', Reader)
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path)
    assert caught.value.code == 'network_file_attribute'
    assert captured[0].count <= 147456 < path.stat().st_size


@pytest.mark.parametrize('format', ['graphml', 'pajek'])
def test_edge_attribute_memory_limit_is_combined_with_complete_graph(tmp_path, monkeypatch, format):
    path = tmp_path / ('attrs.' + format)
    if format == 'graphml':
        text = '<graphml><key id="d" for="edge" attr.name="description" attr.type="string"/><graph edgedefault="undirected">'
        text += ''.join(f'<node id="n{i}"/>' for i in range(200))
        text += ''.join(f'<edge source="n{i}" target="n{i+1}"><data key="d">' + 'x' * 4096 + '</data></edge>' for i in range(199))
        text += '</graph></graphml>'
    else:
        text = '*Vertices 200\n' + ''.join(f'{i+1} "n{i}"\n' for i in range(200)) + '*Edges\n'
        text += ''.join('% openecon-edge ' + json.dumps({'description': 'x' * 4096}) + f'\n{i+1} {i+2}\n' for i in range(199))
    path.write_text(text)
    monkeypatch.setattr(io, 'attach_attributes', lambda *args, **kwargs: pytest.fail('Budget must stop retention before final attach'))
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path, max_memory_mb=2, batch_rows=2)
    assert caught.value.code == 'network_memory_budget'


@pytest.mark.parametrize('format', ['graphml', 'pajek'])
def test_changed_source_is_detected_across_passes(tmp_path, monkeypatch, format):
    graph = network({'source': ['a'], 'target': ['b']})
    path = tmp_path / ('changed.' + format)
    io.write_network(graph, path)
    real = io._xml_records if format == 'graphml' else io._pajek_lines
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield from real(*args, **kwargs)
        if calls == 1:
            # Semantics stay identical; content provenance nevertheless changed.
            path.write_bytes(path.read_bytes() + b'\n')

    monkeypatch.setattr(io, '_xml_records' if format == 'graphml' else '_pajek_lines', changed)
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path)
    assert caught.value.code == 'network_file_changed'


def test_declared_file_limit_and_bounded_pajek_line(tmp_path):
    path = tmp_path / 'big.net'
    path.write_text('%' + 'x' * 200000 + '\n*Vertices 0\n*Edges\n')
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path)
    assert caught.value.code == 'network_file_size'
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path, max_file_mb=.01)
    assert caught.value.code == 'network_file_size'


def test_io_preserves_original_graph_snapshot(tmp_path):
    graph = _fixture_graph(False, True)
    attrs = deepcopy((graph.node_attributes, graph.edge_attributes, graph.graph_attributes))
    indices, values = graph._edges.indices().clone(), graph._edges.values().clone()
    for format in ['graphml', 'gexf', 'pajek']:
        io.write_network(graph, tmp_path / ('preserved.' + format))
    assert attrs == (graph.node_attributes, graph.edge_attributes, graph.graph_attributes)
    assert indices.equal(graph._edges.indices()) and values.equal(graph._edges.values())


def test_foreign_pajek_identity_uses_vertex_numbers_and_preserves_repeated_labels(tmp_path):
    path = tmp_path / 'labels.net'
    path.write_text('*Vertices 3\n01 "Same label"\n2 "Same label"\n3 "Isolate"\n*Edges\n01 2\n')
    result = io.read_network(path)
    assert list(result._labels) == ['1', '2', '3']
    assert _edges(result) == {('1', '2'): 1.}
    assert result.node_attributes == {'1': {'label': 'Same label'}, '2': {'label': 'Same label'},
                                      '3': {'label': 'Isolate'}}


def test_pajek_export_refuses_unreadable_oversized_native_metadata_atomically(tmp_path):
    graph = network([], nodes=['a']).with_attributes(nodes={'a': {str(i): 'x' * 16384 for i in range(9)}})
    path = tmp_path / 'large.net'
    path.write_bytes(b'original')
    with pytest.raises(AnalysisError) as caught:
        io.write_network(graph, path, overwrite=True)
    assert caught.value.code == 'network_file_size'
    assert path.read_bytes() == b'original' and not list(tmp_path.glob('.openecon-network-*'))


@pytest.mark.parametrize('format', ['graphml', 'gexf'])
@pytest.mark.parametrize('character', ['&', '\r', '\n', '\t'])
def test_maximum_escaped_scalar_and_cr_are_exact_native_roundtrips(tmp_path, format, character):
    graph = network([], nodes=['a']).with_attributes(nodes={'a': {'text': character * 16384}})
    path = tmp_path / ('escaped.' + format)
    io.write_network(graph, path)
    assert io.read_network(path).node_attributes == graph.node_attributes


def test_xml_cdata_and_comments_do_not_confuse_quote_or_token_bounds(tmp_path):
    path = tmp_path / 'cdata.graphml'
    text = '''<graphml><key id="d" for="node" attr.name="text" attr.type="string"/><graph edgedefault="undirected">
      <node id="first"><data key="d"><![CDATA['unpaired literal quote > < text]]></data></node>'''
    text += ''.join(f'<node id="n{i}"/>' for i in range(10000))
    text += '</graph></graphml>'
    path.write_text(text)
    result = io.read_network(path)
    assert result.node_attributes['first']['text'] == "'unpaired literal quote > < text"
    assert result.node_count == 10001


def test_unsafe_dtd_detection_survives_feed_boundary(tmp_path):
    path = tmp_path / 'split.graphml'
    path.write_text('<!--' + 'x' * 16370 + '-->' + '<!DOCTYPE graphml [<!ENTITY x "bad">]><graphml><graph edgedefault="undirected"/></graphml>')
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path)
    assert caught.value.code == 'network_unsafe_xml'


@pytest.mark.parametrize('encoding', ['utf-16', 'latin-1'])
def test_non_utf8_xml_rejected_explicitly(tmp_path, encoding):
    path = tmp_path / 'encoding.graphml'
    text = '<?xml version="1.0" encoding="' + encoding + '"?><graphml><graph edgedefault="undirected"><node id="é"/></graph></graphml>'
    path.write_bytes(text.encode(encoding))
    with pytest.raises(AnalysisError) as caught:
        io.read_network(path)
    assert caught.value.code == 'network_unsafe_xml'
