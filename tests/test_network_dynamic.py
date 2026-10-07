"""Independent interval membership, typed identity and dynamic GEXF contract checks."""
from copy import deepcopy
from fractions import Fraction
from itertools import product

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def edge_rows(graph):
    return graph.edges(attributes=False).to_dict("records")


def alive(spells, time):
    # Independent scalar oracle uses only caller dictionaries, not production
    # interval normalization, intersections or cached presence indexes.
    time = Fraction(time)
    for spell in spells:
        low = spell.get("start", spell.get("startopen"))
        high = spell.get("end", spell.get("endopen"))
        if low is not None and (time < Fraction(low) or time == Fraction(low) and "startopen" in spell):
            continue
        if high is not None and (time > Fraction(high) or time == Fraction(high) and "endopen" in spell):
            continue
        return True
    return False


SPELLS = [[{}], [{"start": 0, "end": 1}], [{"startopen": 0, "endopen": 1}],
          [{"start": 1, "end": 2}], [{"start": 0, "endopen": 1}, {"start": 1, "end": 2}],
          [{"start": 0, "endopen": 1}, {"startopen": 1, "end": 2}]]


@pytest.mark.parametrize("source,target,edge", list(product(SPELLS, repeat=3)))
def test_all_small_spell_intersections_match_raw_scalar_membership(source, target, edge):
    graph = oe.dynamic_network([dict(node=1, spells=source), dict(node="1", spells=target)],
        [dict(edge_id=1, source=1, target="1", spells=edge, weight=0.),
         dict(edge_id="1", source="1", target=1, spells=edge, weight=2.)],
        version="1.2draft", interval={"start": 0, "end": 2})
    for time in [-1, 0, .5, 1, 1.5, 2, 3]:
        selected = graph.at(time)
        present = {value for value, spells in [(1, source), ("1", target)]
                   if 0 <= time <= 2 and alive(spells, time)}
        expected = 2 if present == {1, "1"} and alive(edge, time) else 0
        assert selected.node_count == len(present) and selected.edge_count == expected
        assert selected.nodes(attributes=False).node.tolist() == [value for value in [1, "1"] if value in present]
        if expected:
            assert [record['edge_id'] for record in edge_rows(selected)] == [1, "1"]
            degree = selected.degree().set_index('node')
            assert degree.degree.to_dict() == {1: 2, "1": 2}
            assert degree.strength.to_dict() == {1: 2., "1": 2.}


@pytest.mark.parametrize("spells", SPELLS)
@pytest.mark.parametrize("start,end,left,right", [(0, 2, False, False), (0, 1, True, False),
                                                 (1, 2, False, True), (1, 1, False, False)])
def test_window_overlap_and_cover_match_dense_rational_probe_oracle(spells, start, end, left, right):
    graph = oe.dynamic_network([dict(node="a", spells=spells), dict(node="b")],
        [dict(edge_id="e", source="a", target="b")], version="1.2draft")
    probes = [Fraction(i, 4) for i in range(start * 4, end * 4 + 1)
              if not (left and i == start * 4 or right and i == end * 4)]
    for selection, expected in [("overlap", any(alive(spells, t) for t in probes)),
                                ("cover", all(alive(spells, t) for t in probes))]:
        result = graph.window(start, end, start_open=left, end_open=right, selection=selection)
        assert result.edge_count == int(expected)


@pytest.mark.parametrize("timeformat,start,end,before,after", [
    ("double", .5, 1.5, 0., 2.), ("integer", 2**200, 2**200 + 2, 2**200 - 1, 2**200 + 3),
    ("date", "2026-10-07", "2026-10-08", "2026-10-06", "2026-10-09"),
    ("dateTime", "2026-10-07T00:00:00.000001Z", "2026-10-07T00:00:00.000003Z",
     "2026-10-07T00:00:00Z", "2026-10-07T00:00:00.000004Z")])
@pytest.mark.parametrize("directed", [False, True])
def test_time_formats_exact_boundaries_and_roundtrip(tmp_path, timeformat, start, end, before, after, directed):
    graph = oe.dynamic_network([dict(node=1), dict(node="1"), dict(node="isolate")],
        [dict(edge_id=2**200, source="1", target=1, start=start, end=end,
              attributes={"note": 'é CR\rLF\nTab\t & < > "'})], timeformat=timeformat, directed=directed)
    assert not graph.weighted
    for value, count in [(before, 0), (start, 1), (end, 1), (after, 0)]:
        assert graph.at(value).edge_count == count
    path = tmp_path / 'owned.gexf'
    graph.write(path)
    reopened = oe.read_dynamic_network(path)
    pd.testing.assert_frame_equal(reopened.nodes(), graph.nodes())
    pd.testing.assert_frame_equal(reopened.edges(), graph.edges())
    assert not reopened.weighted and reopened.directed == directed
    assert reopened.metadata['file_passes'] == 3
    assert reopened.at(start).node_count == 3


def test_datetime_offsets_are_exact_utc_instants_and_no_fraction_is_truncated():
    graph = oe.dynamic_network([dict(node="a", start="2026-10-07T03:00:00+03:00", end="2026-10-07T03:00:00+03:00")],
        [], timeformat="dateTime")
    assert graph.at("2026-10-07T00:00:00Z").node_count == 1
    assert graph.at("2026-10-07T00:00:00.000001Z").node_count == 0
    for value in ['2026-10-07T00:00:00.0000001Z', '2026-10-07T00:00', '2026-99-07T00:00:00Z']:
        with pytest.raises(AnalysisError) as error:
            graph.at(value)
        assert error.value.code == 'network_dynamic_time'


def timed_graph():
    return oe.dynamic_network([dict(node="a", dynamic_attributes={"status": [
        {"value": "old", "end": 1}, {"value": "new", "startopen": 1}]}) , dict(node="b")],
        [dict(edge_id=1, source="a", target="b", weight=1., dynamic_attributes={"weight": [
            {"value": 2., "start": 0, "end": 1}, {"value": 5., "startopen": 1, "end": 3}],
            "note": [{"value": "present", "start": 1, "end": 2}]})],
        version="1.2draft", node_defaults={"status": "unknown"})


def test_timed_attributes_defaults_weight_changes_and_explicit_window_loss_policy(tmp_path):
    graph = timed_graph()
    before = graph.edges().to_dict('records')
    assert graph.at(1).edges().weight.tolist() == [2.]
    assert graph.at(2).edges().weight.tolist() == [5.]
    assert graph.at(4).edges().weight.tolist() == [1.]
    assert graph.at(1).nodes().set_index('node')['attr.status'].to_dict() == {'a': 'old', 'b': 'unknown'}
    with pytest.raises(AnalysisError) as error:
        graph.window(0, 2)
    assert error.value.code == 'network_dynamic_ambiguity'
    with pytest.raises(AnalysisError):
        graph.window(0, 2, attributes='drop')
    result = graph.window(0, 2, attributes='drop', weight='max')
    assert result.edges().weight.tolist() == [5.]
    dropped = result.metadata['dynamic_selection']['dropped_dynamic_attributes']
    assert dropped == [{'scope': 'node', 'id': 'a', 'attribute': 'status'},
                       {'scope': 'edge', 'id': 1, 'attribute': 'note'}]
    assert graph.window(0, 2, attributes='drop', weight='min').edges().weight.tolist() == [2.]
    assert graph.edges().to_dict('records') == before
    path = tmp_path / 'timed.gexf'
    graph.write(path)
    reopened = oe.read_dynamic_network(path)
    pd.testing.assert_frame_equal(reopened.edges(), graph.edges())
    pd.testing.assert_frame_equal(reopened.nodes(), graph.nodes())
    assert reopened.defaults == graph.defaults
    assert reopened.window(0, 2, attributes='drop', weight='max').edges().weight.tolist() == [5.]


def test_inclusive_overlapping_values_are_ambiguous_even_at_one_point():
    graph = oe.dynamic_network([dict(node="a", dynamic_attributes={"x": [
        dict(value=1, start=0, end=1), dict(value=2, start=1, end=2)]})], [])
    assert graph.at(.5).nodes()['attr.x'].tolist() == [1]
    with pytest.raises(AnalysisError) as error:
        graph.at(1)
    assert error.value.code == 'network_dynamic_ambiguity'
    assert graph.window(1, 1, attributes='drop').nodes().columns.tolist() == ['node']


def test_equal_values_with_different_scalar_types_and_intermittent_presence_raise():
    graph = oe.dynamic_network([dict(node="a", dynamic_attributes={"x": [
        dict(value=True, start=0, end=1), dict(value=1, start=1, end=2)]})], [])
    with pytest.raises(AnalysisError):
        graph.at(1)
    gap = oe.dynamic_network([dict(node="a", dynamic_attributes={"x": [dict(value=1, start=0, end=1)]})], [])
    with pytest.raises(AnalysisError):
        gap.window(0, 2)
    assert 'attr.x' not in gap.window(0, 2, attributes='drop').nodes()


def test_source_mutation_export_overwrite_failure_and_unsupported_static_read_are_safe(tmp_path):
    nodes = [dict(node="a", attributes={"label": "source"}), dict(node="b")]
    edges = [dict(edge_id="one", source="a", target="b", start=0, end=2)]
    graph = oe.dynamic_network(nodes, edges)
    nodes[0]['attributes']['label'] = 'caller edit'
    edges[0]['end'] = 99
    table = graph.nodes()
    table.iloc[0]['attributes']['label'] = 'table edit'
    assert graph.nodes().iloc[0]['attributes'] == {'label': 'source'}
    assert graph.at(3).edge_count == 0
    path = tmp_path/'owned.gexf'
    graph.write(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        graph.write(path)
    assert path.read_bytes() == original
    for reader in [oe.read_network, oe.read_multigraph]:
        with pytest.raises(AnalysisError) as error:
            reader(path)
        assert error.value.code == 'network_file_feature'
    graph.write(path, overwrite=True)
    assert path.read_bytes() == original
    destination = tmp_path/'other.graphml'
    destination.write_text('original')
    with pytest.raises(AnalysisError):
        graph.write(destination, overwrite=True)
    assert destination.read_text() == 'original' and not list(tmp_path.glob('.openecon-dynamic-*'))


def external_xml(body, attrs='', version='1.3'):
    return f'<gexf xmlns="http://gexf.net/{version}" version="{version}"><graph mode="dynamic" defaultedgetype="directed" {attrs}>{body}</graph></gexf>'


def test_external_dynamic_defaults_static_weight_ids_and_forward_declarations(tmp_path):
    path = tmp_path/'external.gexf'
    path.write_text(external_xml('''<edges><edge id="first" source="a" target="b"><attvalues>
<attvalue for="w" value="4"/><attvalue for="x" value="9" start="1" end="2"/></attvalues></edge>
<edge id="second" source="a" target="b" start="0" end="3"/></edges>
<nodes><node id="a" label="Origin"/><node id="b"/></nodes>
<attributes class="edge" mode="static"><attribute id="w" title="weight" type="double"><default>2</default></attribute></attributes>
<attributes class="edge" mode="dynamic"><attribute id="x" title="x" type="integer"><default>7</default></attribute></attributes>''', 'timeformat="integer"'))
    graph = oe.read_dynamic_network(path)
    assert graph.at(1).edges().weight.tolist() == [4., 2.]
    assert graph.at(1).edges()['attr.x'].tolist() == [9, 7]
    assert graph.at(4).edges()['attr.x'].tolist() == [7]
    assert graph.at(4).nodes().set_index('node')['attr.label'].to_dict()['a'] == 'Origin'
    assert [row['edge_id'] for row in edge_rows(graph.at(1))] == ['first','second']


@pytest.mark.parametrize('body,attrs', [
    ('<nodes><node id="a"><node id="b"/></node></nodes>', ''),
    ('<nodes><node id="a" pid="parent"/></nodes>', ''),
    ('<nodes><node id="a" timestamps="[1,2]"/></nodes>', ''),
    ('<nodes><node id="a" startopen="0"/></nodes>', ''),
    ('<nodes><node id="a"/></nodes><edges><edge source="a" target="a"/></edges>', ''),
    ('<nodes><node id="a"/></nodes><edges><edge id="e" source="a" target="unknown"/></edges>', ''),
    ('<nodes><node id="a"/></nodes><edges><edge id="e" source="a" target="a" type="undirected"/></edges>', ''),
    ('<nodes><node id="a"/><node id="a"/></nodes>', ''),
    ('<nodes><node id="a"/></nodes><edges><edge id="e" source="a" target="a"/><edge id="e" source="a" target="a"/></edges>', ''),
    ('<attributes class="graph" mode="dynamic"/>', ''),
    ('<nodes><node id="a"><spells/></node></nodes>', ''),
    ('<nodes><node id="a" start="0"><spells><spell end="1"/></spells></node></nodes>', ''),
    ('<nodes><node id="a"/></nodes>', 'timezone="Europe/Istanbul"'),
    ('<nodes><node id="a"/></nodes>', 'timerepresentation="timestamp"'),
    ('<nodes><node id="a"/></nodes>', 'idtype="float"'),
])
def test_unsupported_or_malformed_xml_is_never_silently_flattened(tmp_path, body, attrs):
    path = tmp_path/'invalid.gexf'
    path.write_text(external_xml(body, attrs))
    before = path.read_bytes()
    with pytest.raises(AnalysisError):
        oe.read_dynamic_network(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('bounds', [dict(start=2,end=1),dict(startopen=1,end=1),dict(start=0,startopen=0),dict(startopen=None),dict(end=float('inf'))])
def test_invalid_intervals_fail_before_snapshot(bounds):
    with pytest.raises(AnalysisError) as error:
        oe.dynamic_network([dict(node="a",spells=[bounds])],[],version='1.2draft')
    assert error.value.code == 'network_dynamic_time'


def test_resource_preflight_iterator_cleanup_and_explicit_cpu(tmp_path, monkeypatch):
    graph = oe.dynamic_network([dict(node="a")],[dict(edge_id="loop",source="a",target="a")])
    with torch.device('meta'):
        selected = graph.at(0)
    assert selected._weights.device.type == 'cpu'
    with pytest.raises(AnalysisError) as error:
        graph.at(0,max_work=1)
    assert error.value.code == 'network_work_budget'
    with pytest.raises(AnalysisError) as error:
        oe.dynamic_network([dict(node="a")],[],max_memory_mb=1)
    assert error.value.code == 'network_memory_budget'
    closed = []
    def records():
        try:
            for i in range(100):
                yield dict(node=i,spells=[dict(start=0,end=1)])
        finally:
            closed.append(True)
    with pytest.raises(AnalysisError) as error:
        oe.dynamic_network(records(),[],max_events=3)
    assert error.value.code == 'network_event_budget' and closed == [True]
    path = tmp_path/'large.gexf'
    owned = oe.dynamic_network([dict(node=i,attributes={'text':'x'*4000}) for i in range(25)],[],max_memory_mb=64)
    owned.write(path)
    with pytest.raises(AnalysisError) as error:
        oe.read_dynamic_network(path,max_memory_mb=2)
    assert error.value.code == 'network_memory_budget'


def test_changed_file_between_passes_is_rejected(tmp_path,monkeypatch):
    import openecon._network_dynamic_io as io
    path=tmp_path/'changing.gexf'
    oe.dynamic_network([dict(node='a')],[]).write(path)
    original, calls=io._records, []
    def changed(*args,**kwargs):
        yield from original(*args,**kwargs)
        calls.append(True)
        if len(calls)==1:
            path.write_text(path.read_text().replace('>\n','>\n\n',1))
    monkeypatch.setattr(io,'_records',changed)
    with pytest.raises(AnalysisError) as error:
        oe.read_dynamic_network(path)
    assert error.value.code=='network_file_changed'


def test_static_projection_and_snapshot_order_require_explicit_choices():
    graph=oe.dynamic_network([dict(node='a'),dict(node='b'),dict(node='c')],
        [dict(edge_id='ab',source='a',target='b',start=0,end=1),
         dict(edge_id='bc',source='b',target='c',start=1,end=2)],directed=True)
    with pytest.raises(AnalysisError) as error:
        graph.pagerank()
    assert error.value.code=='network_dynamic_capacity'
    layers=oe.network_snapshots({'first':graph.at(0).to_network(reducer='count'),
        'second':graph.at(2).to_network(reducer='count')},ordered=True)
    route = layers.temporal_path('a', 'c')
    assert route[['source', 'target', 'layer']].to_dict('records') == [
        dict(source='a', target='b', layer='first'),
        dict(source='b', target='c', layer='second'),
    ]
    assert graph.window(0,2).metadata['dynamic_selection']['temporal_path_model'] is False


def test_bad_scalar_types_and_export_failure_preserve_destination(tmp_path):
    graph=oe.dynamic_network([dict(node='a',dynamic_attributes={'x':[dict(value=True,start=0,end=1),dict(value=1,start=2,end=3)]})],[])
    destination=tmp_path/'existing.gexf'
    destination.write_text('protected')
    with pytest.raises(AnalysisError) as error:
        graph.write(destination,overwrite=True)
    assert error.value.code=='network_file_attribute'
    assert destination.read_text()=='protected' and not list(tmp_path.glob('.openecon-dynamic-*'))
    before=deepcopy(graph.metadata)
    before['edge_count']=99
    assert graph.metadata['edge_count']==0


def test_double_integer_loss_and_invalid_options_do_not_leak_python_errors():
    graph=oe.dynamic_network([dict(node='a')],[])
    with pytest.raises(AnalysisError):
        graph.at(2**53+1)
    for option in [dict(timeformat=[]),dict(version=[]),dict(directed=1)]:
        with pytest.raises(AnalysisError):
            oe.dynamic_network([],[],**option)
    with pytest.raises(AnalysisError):
        graph.window(0,1,attributes=[])


@pytest.mark.parametrize("weight", [0., 2., "timed", "default"])
def test_conflicting_native_unweighted_metadata_is_refused(tmp_path, weight):
    options = dict(edge_defaults={"weight": 1.}) if weight == "default" else {}
    edge = dict(edge_id='e', source='a', target='b')
    if weight == "timed":
        edge['dynamic_attributes'] = {'weight': [dict(value=1., start=0, end=1)]}
    elif weight != "default":
        edge['weight'] = weight
    graph = oe.dynamic_network([dict(node='a'), dict(node='b')], [edge], **options)
    path = tmp_path / 'owned.gexf'
    graph.write(path)
    path.write_text(path.read_text().replace('openecon.weighted=true', 'openecon.weighted=false'))
    with pytest.raises(AnalysisError) as error:
        oe.read_dynamic_network(path)
    assert error.value.code == 'network_file_attribute'
