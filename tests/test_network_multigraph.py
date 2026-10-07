"""Typed parallel-edge identity, independent reductions and strict interchange."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_io import _edge_records


def rows(graph):
    return graph.edges(attributes=False).to_dict("records")


def fixture_graph(directed=False, graph_attrs=False):
    return oe.multigraph([
        dict(edge_id=1, source="1", target=1, w=2., attrs={"label": "first", "active": True}),
        dict(edge_id="1", source="1", target=1, w=3., attrs={"label": "second", "active": False}),
        dict(edge_id=2**200, source=1, target=1, w=7., attrs={"label": "loop", "active": True}),
        dict(edge_id="zero", source="z", target="1", w=0., attrs={"label": "zero", "active": True}),
    ], weight="w", attributes="attrs", directed=directed, nodes=["isolate"],
        node_attributes={1: {"label": "integer", "period": 2026}},
        graph_attributes={"title": "owned graph"} if graph_attrs else None)


@pytest.mark.parametrize("directed", [False, True])
def test_typed_edge_ids_parallel_attributes_zero_and_loop_degree(directed):
    graph = fixture_graph(directed)
    assert graph.edge_count == 4 and graph.node_count == 4
    assert rows(graph) == [dict(edge_id=1, source="1", target=1, weight=2.),
        dict(edge_id="1", source="1", target=1, weight=3.),
        dict(edge_id=2**200, source=1, target=1, weight=7.),
        dict(edge_id="zero", source="z", target="1", weight=0.)]
    assert graph.edge_attributes[1]["label"] == "first"
    assert graph.edge_attributes["1"]["label"] == "second"
    degree = graph.degree().set_index("node")
    if directed:
        assert degree.in_degree.to_dict() == {"isolate": 0, "1": 1, 1: 3, "z": 0}
        assert degree.out_degree.to_dict() == {"isolate": 0, "1": 2, 1: 1, "z": 1}
        assert degree.in_strength.to_dict() == {"isolate": 0., "1": 0., 1: 12., "z": 0.}
    else:
        assert degree.degree.to_dict() == {"isolate": 0, "1": 3, 1: 4, "z": 1}
        assert degree.strength.to_dict() == {"isolate": 0., "1": 5., 1: 19., "z": 0.}
    assert degree.attrs["exact"] and not degree.attrs["sampled"]
    assert graph.metadata["zero_weight_rows_dropped"] == graph.metadata["duplicate_edge_rows_aggregated"] == 0
    assert graph.summary().attrs["network"]["edge_count"] == 4


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("reducer", ["sum", "min", "max", "mean", "count", "binary"])
def test_reducers_match_independent_scalar_grouping(directed, reducer):
    data = [dict(edge_id="a", source=1, target="1", weight=2.),
            dict(edge_id="b", source=1, target="1", weight=3.),
            dict(edge_id="c", source="1", target=1, weight=4.),
            dict(edge_id="d", source="1", target="1", weight=5.)]
    graph = oe.multigraph(data, weight="weight", directed=directed, nodes=["isolate"])
    result = graph.to_network(reducer=reducer)
    grouped = {}
    for row in data:
        pair = (row["source"], row["target"])
        if not directed and isinstance(pair[0], str) and isinstance(pair[1], int):
            pair = pair[::-1]
        grouped.setdefault(pair, []).append(row["weight"])
    def reduce(values):
        return {"sum": sum(values), "min": min(values), "max": max(values),
                "mean": sum(values) / len(values), "count": float(len(values)), "binary": 1.}[reducer]
    expected = {pair: reduce(values) for pair, values in grouped.items()}
    actual = {}
    for i, j, value in _edge_records(result):
        pair = (result._labels[i], result._labels[j])
        if not directed and isinstance(pair[0], str) and isinstance(pair[1], int):
            pair = pair[::-1]
        actual[pair] = value
    assert actual == pytest.approx(expected)
    assert result.node_count == 3
    assert result.metadata["multigraph_projection"]["reducer"] == reducer
    assert not result.metadata["multigraph_projection"]["edge_ids_preserved"]


def test_projection_zero_and_conflicting_attributes_require_explicit_loss_policy():
    graph = fixture_graph()
    for options in [dict(reducer="sum"), dict(reducer="sum", zero="drop")]:
        with pytest.raises(AnalysisError) as error:
            graph.to_network(**options)
        assert error.value.code == "network_projection_loss"
    projected = graph.to_network(reducer="sum", zero="drop", attributes="drop")
    assert projected.edge_count == 2 and projected.node_count == 4
    assert projected.metadata["multigraph_projection"]["zero_pairs_dropped"] == 1
    assert projected._edge_attributes == {}
    binary = graph.to_network(reducer="binary", attributes="drop")
    assert binary.edge_count == 3
    assert binary.degree().set_index("node").degree.to_dict()["z"] == 1


def test_equal_attributes_survive_projection_with_input_order_opposite_typed_order():
    graph = oe.multigraph([dict(edge_id=i, source="z", target=-1, weight=2.) for i in range(2)],
        weight="weight", edge_attributes={0: {"label": "same"}, 1: {"label": "same"}},
        node_attributes={"z": {"zone": "east"}}, graph_attributes={"title": "sample"})
    projected = graph.to_network(reducer="sum")
    assert projected._edge_attributes == {(-1, "z"): {"label": "same"}}
    assert projected._node_attributes == {"z": {"zone": "east"}}
    assert projected._graph_attributes == {"title": "sample"}
    assert projected.edges().weight.tolist() == [4.]


def test_stable_mean_and_overflow_degree_sum_precision_errors():
    graph = oe.multigraph([dict(edge_id=i, source="a", target="b", w=1e308) for i in range(2)], weight="w")
    mean = graph.to_network(reducer="mean")
    assert mean.edges().weight.tolist() == [1e308]
    for method in [lambda: graph.degree(), lambda: graph.to_network(reducer="sum")]:
        with pytest.raises(AnalysisError) as error:
            method()
        assert error.value.code == "network_precision"


@pytest.mark.parametrize("values", [(True, 1), (1, 1.), (False, 0)])
def test_projection_refuses_equal_values_with_different_scalar_attribute_types(values):
    graph = oe.multigraph([dict(edge_id=i, source="a", target="b") for i in range(2)],
        edge_attributes={i: {"value": value} for i, value in enumerate(values)})
    with pytest.raises(AnalysisError) as error:
        graph.to_network(reducer="sum")
    assert error.value.code == "network_projection_loss"
    assert graph.to_network(reducer="sum", attributes="drop").edge_count == 1
    assert type(graph.edge_attributes[0]["value"]) is type(values[0])
    assert type(graph.edge_attributes[1]["value"]) is type(values[1])


def test_edit_and_filter_preserve_exact_ids_attributes_and_source_snapshot():
    graph = fixture_graph(True)
    before, attrs = rows(graph), graph.edge_attributes
    changed = graph.edit_edges(remove=["1"], weights={1: 9.},
        add=[dict(edge_id="new", source=1, target="z", weight=6., attributes={"label": "added"})],
        attributes={1: {"label": "updated"}})
    assert [row["edge_id"] for row in rows(changed)] == [1, 2**200, "zero", "new"]
    assert rows(changed)[0]["weight"] == 9.
    assert changed.edge_attributes["new"] == {"label": "added"}
    assert changed.edge_attributes[1] == {"label": "updated"}
    selected = changed.filter(nodes=["isolate", 1, "1"], edge_ids=[1, "new", 2**200])
    assert [row["edge_id"] for row in rows(selected)] == [1, 2**200]
    assert selected.nodes(attributes=False).node.tolist() == ["isolate", "1", 1]
    swapped = graph.edit_nodes(rename={1: "1", "1": 1}, add=["extra"])
    assert [row["edge_id"] for row in rows(swapped)] == [1, "1", 2**200, "zero"]
    assert rows(swapped)[0]["source"] == 1 and rows(swapped)[0]["target"] == "1"
    assert swapped.node_attributes == {"1": {"label": "integer", "period": 2026}}
    assert graph.edit_nodes(remove=[1]).edge_count == 1
    assert rows(graph) == before and graph.edge_attributes == attrs
    attrs[1]["label"] = "caller edit"
    assert graph.edge_attributes[1]["label"] == "first"


def test_attribute_replacement_reserves_retained_source_and_new_attributes():
    graph = oe.multigraph([dict(edge_id=i, source="a", target="b") for i in range(10)],
        edge_attributes={i: {"text": "x" * 12000} for i in range(10)}, max_memory_mb=3)
    with pytest.raises(AnalysisError) as error:
        graph.with_attributes()
    assert error.value.code == "network_memory_budget"
    cleared = graph.with_attributes(edges={})
    assert cleared.edge_attributes == {} and len(graph._edge_attributes) == 10
    changed = oe.multigraph([dict(edge_id=1, source="a", target="b")]).edit_edges(
        add=[dict(edge_id="new", source="a", target="b")], attributes={"new": {"text": "added"}})
    assert changed.edge_attributes == {"new": {"text": "added"}}
    with pytest.raises(AnalysisError) as error:
        changed.edit_edges(attributes={"absent": {"text": "invalid"}})
    assert error.value.code == "network_unknown_edge"
    with pytest.raises(AnalysisError) as error:
        changed.edit_edges(add=[dict(edge_id=[], source="a", target="b")])
    assert error.value.code == "network_invalid_label"


@pytest.mark.parametrize("extension", ["csv", "parquet"])
@pytest.mark.parametrize("batch", [1, 2, 65536])
def test_dataset_and_dataframe_import_preserve_separate_edges(tmp_path, extension, batch):
    frame = pd.DataFrame({"id": [9, 10, 11], "u": [1, 1, 1], "v": [2, 2, 1], "w": [2., 3., 0.]})
    path = tmp_path / ("edges." + extension)
    if extension == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, row_group_size=1)
    graph = oe.multigraph(oe.scan(path), edge_id="id", source="u", target="v", weight="w", batch_rows=batch)
    reference = oe.multigraph(frame, edge_id="id", source="u", target="v", weight="w")
    pd.testing.assert_frame_equal(graph.edges(), reference.edges())
    assert graph.edge_count == 3


@pytest.mark.parametrize("format", ["graphml", "gexf"])
@pytest.mark.parametrize("directed", [False, True])
def test_xml_roundtrip_typed_parallel_zero_loop_attributes_and_orientation(tmp_path, format, directed):
    graph = fixture_graph(directed, graph_attrs=format == "graphml")
    path = tmp_path / ("multi." + format)
    graph.write(path)
    reopened = oe.read_multigraph(path, batch_rows=2)
    pd.testing.assert_frame_equal(graph.edges(), reopened.edges())
    pd.testing.assert_frame_equal(graph.nodes(), reopened.nodes())
    pd.testing.assert_frame_equal(graph.degree(), reopened.degree())
    assert reopened.edge_attributes == graph.edge_attributes
    assert reopened.node_attributes == graph.node_attributes
    assert reopened.graph_attributes == graph.graph_attributes
    assert reopened.directed == directed and reopened.weighted
    assert reopened.metadata["file_passes"] == 2
    with pytest.raises(AnalysisError, match="read_multigraph"):
        oe.read_network(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        graph.write(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("format", ["graphml", "gexf"])
def test_unweighted_empty_and_utf8_attributes_roundtrip(tmp_path, format):
    for index, data in enumerate([[], [dict(edge_id="é-边", source=1, target="1")]]):
        graph = oe.multigraph(data, nodes=["isolate"])
        if data:
            graph = graph.with_attributes(edges={"é-边": {"note": "CR\rLF\nTab\t & < > \"", "ok": False}})
        path = tmp_path / (str(index) + "." + format)
        graph.write(path)
        reopened = oe.read_multigraph(path)
        assert not reopened.weighted
        pd.testing.assert_frame_equal(reopened.edges(), graph.edges())
        assert reopened.edge_attributes == graph.edge_attributes


def test_external_graphml_declared_defaults_and_edges_before_nodes(tmp_path):
    path = tmp_path / "external.graphml"
    path.write_text('''<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
<key id="w" for="edge" attr.name="weight" attr.type="double"><default>2</default></key>
<key id="r" for="edge" attr.name="relation" attr.type="string"><default>trade</default></key>
<graph edgedefault="directed"><edge id="alpha" source="n" target="m"/>
<edge id="beta" source="n" target="m"><data key="w">3</data><data key="r">loan</data></edge>
<edge id="zero" source="m" target="m"><data key="w">0</data></edge>
<node id="n"/><node id="m"/></graph></graphml>''')
    graph = oe.read_multigraph(path)
    assert [row["edge_id"] for row in rows(graph)] == ["alpha", "beta", "zero"]
    assert graph.edge_attributes == {"alpha": {"relation": "trade"}, "beta": {"relation": "loan"}, "zero": {"relation": "trade"}}
    assert graph.degree().set_index("node").in_degree.to_dict() == {"n": 0, "m": 3}


@pytest.mark.parametrize("edges", [
    '<edge source="a" target="b"/>',
    '<edge id="x" source="a" target="b"/><edge id="x" source="a" target="b"/>',
    '<edge id="x" source="a" target="missing"/>',
    '<edge id="x" source="a" target="b" directed="true"/>',
])
def test_invalid_xml_ids_endpoints_and_mixed_directions_reject(tmp_path, edges):
    path = tmp_path / "invalid.graphml"
    path.write_text('<graphml><graph edgedefault="undirected"><node id="a"/><node id="b"/>' + edges + '</graph></graphml>')
    with pytest.raises(AnalysisError):
        oe.read_multigraph(path)


def test_lossy_formats_and_unsupported_methods_never_project(tmp_path, monkeypatch):
    graph = fixture_graph(graph_attrs=True)
    monkeypatch.setattr(graph, "to_network", lambda **kwargs: pytest.fail("No implicit projection"))
    target = tmp_path / "user.net"
    target.write_bytes(b"user data")
    with pytest.raises(AnalysisError, match="Pajek"):
        graph.write(target, overwrite=True)
    assert target.read_bytes() == b"user data"
    with pytest.raises(AnalysisError, match="Pajek"):
        oe.read_multigraph(target)
    with pytest.raises(AnalysisError, match="graph attributes"):
        graph.write(tmp_path / "attrs.gexf")
    assert not (tmp_path / "attrs.gexf").exists()
    assert not list(tmp_path.glob(".openecon-network-*"))
    for method in ("pagerank", "components", "communities", "max_flow", "poisson_block_model", "to_plot_data"):
        with pytest.raises(AnalysisError) as error:
            getattr(graph, method)()
        assert error.value.code == "network_multigraph_capacity"


@pytest.mark.parametrize("record", [
    dict(edge_id=True, source=1, target=2), dict(edge_id=1.5, source=1, target=2),
    dict(edge_id=2**300, source=1, target=2), dict(edge_id="x", source=1, target=2, w=-1),
    dict(edge_id="x", source=1, target=2, w=math.inf), dict(edge_id="x", source=1, target=None, w=1),
])
def test_identity_and_weight_domains(record):
    with pytest.raises(AnalysisError):
        oe.multigraph([record], weight="w" if "w" in record else None)


def test_duplicate_ids_invalid_updates_and_missing_drop_are_explicit():
    data = [dict(edge_id=1, source="a", target="b")]
    with pytest.raises(AnalysisError) as error:
        oe.multigraph(data * 2)
    assert error.value.code == "network_duplicate_edge_id"
    assert oe.multigraph(data + [dict(edge_id=1, source=None, target="b")], missing="drop").edge_count == 1
    graph = oe.multigraph(data)
    for action in [lambda: graph.edit_edges(add=data), lambda: graph.edit_edges(remove=["1"]),
                   lambda: graph.edit_edges(weights={"x": 2}), lambda: graph.filter(edge_ids="1"),
                   lambda: graph.edit_nodes(rename={"a": "b"}),
                   lambda: graph.with_attributes(edges={1: {"openecon.edge_identity": "reserved"}})]:
        with pytest.raises(AnalysisError):
            action()
    assert graph.edit_edges(add=[dict(edge_id=2, source="a", target="b", weight=1)]).weighted


def test_budget_and_default_device_guard_before_allocations(monkeypatch):
    graph = fixture_graph()
    with torch.device("meta"):
        assert oe.multigraph([dict(edge_id=1, source="a", target="b")]).degree().degree.tolist() == [1, 1]
        assert graph.to_network(reducer="count", attributes="drop").node_count == 4
    def forbidden(*args, **kwargs):
        pytest.fail("Admission must precede tensor setup")
    monkeypatch.setattr(torch, "tensor", forbidden)
    with pytest.raises(AnalysisError) as error:
        oe.multigraph((dict(edge_id=i, source=1, target=2) for i in range(20_000)), max_memory_mb=1)
    assert error.value.code == "network_memory_budget"
    monkeypatch.setattr(torch, "unique", forbidden)
    with pytest.raises(AnalysisError) as error:
        graph.to_network(reducer="sum", max_work=1)
    assert error.value.code == "network_work_budget"
    monkeypatch.setattr(torch, "bincount", forbidden)
    with pytest.raises(AnalysisError) as error:
        graph.degree(max_work=1)
    assert error.value.code == "network_work_budget"


def test_memory_failure_closes_the_owned_iterator_and_empty_projection_is_valid():
    closed = []
    def stream():
        try:
            for identifier in range(5000):
                yield dict(edge_id=identifier, source="a", target="b")
        finally:
            closed.append(True)
    with pytest.raises(AnalysisError, match="max_memory_mb"):
        oe.multigraph(stream(), max_memory_mb=1)
    assert closed == [True]
    empty = oe.multigraph([], nodes=["isolate"])
    for reducer in ("sum", "mean", "min", "max", "count", "binary"):
        projected = empty.to_network(reducer=reducer)
        assert projected.node_count == 1 and projected.edge_count == 0


def test_xml_coupled_parser_and_graph_memory_guard_and_changed_file_reject(tmp_path, monkeypatch):
    graph = oe.multigraph([dict(edge_id=i, source="a", target="b", attrs={"note": "é" * 4096})
                          for i in range(15)], attributes="attrs")
    path = tmp_path / "large-attributes.graphml"
    graph.write(path)
    before = path.read_bytes()
    with pytest.raises(AnalysisError) as error:
        oe.read_multigraph(path, max_memory_mb=1)
    assert error.value.code == "network_memory_budget" and path.read_bytes() == before
    import openecon._network_io as io
    original = io._xml_records
    calls = 0
    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)
        if calls == 1:
            with path.open("a") as file:
                file.write("\n")
    monkeypatch.setattr(io, "_xml_records", changed)
    with pytest.raises(AnalysisError) as error:
        oe.read_multigraph(path)
    assert error.value.code == "network_file_changed"


def test_table_and_metadata_changes_do_not_mutate_source():
    graph = fixture_graph()
    frame = graph.edges()
    frame.loc[0, "weight"] = 999.
    frame.loc[0, "attr.label"] = "caller"
    metadata = graph.metadata
    metadata["edge_count"] = 0
    assert rows(graph)[0]["weight"] == 2.
    assert graph.edge_attributes[1]["label"] == "first" and graph.metadata["edge_count"] == 4


def test_new_public_api_remains_lazy_and_old_network_coalescing_unchanged():
    process = subprocess.run([sys.executable, "-c", """
import sys
import openecon as oe
assert {'MultiNetwork', 'multigraph', 'read_multigraph'} <= set(dir(oe))
assert 'torch' not in sys.modules
assert oe.capabilities()['network']['multigraph']['implicit_projection'] is False
assert 'torch' not in sys.modules
"""], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    data = dict(source=[1, 1], target=[2, 2], w=[2., 3.])
    graph = oe.network(data, weight="w")
    assert graph.edge_count == 1 and graph.edges().weight.tolist() == [5.]


def test_source_panel_example_has_ordered_outputs_and_saved_history(tmp_path):
    from openecon.console import ConsoleSession
    from openecon.workspace import Workspace
    code = (Path(__file__).resolve().parents[1] / "docs/examples/network_multigraph.py").read_text()
    workspace = Workspace(tmp_path / "owned-multigraph")
    session = ConsoleSession(workspace)
    try:
        run = session.execute(code, timeout_seconds=90)
        assert run["status"] == "ok", run.get("error")
        assert [item["type"] for item in run["outputs"]] == ["table"] * 5
        assert all("\\begin{tabular}" in item["latex"] for item in run["outputs"])
        probe = session.execute("print(json.dumps(multigraph_proof,allow_nan=False))")
        assert probe["status"] == "ok"
        proof = json.loads(probe["stdout"])
        assert proof["roundtrip"] and proof["source_unchanged"] and proof["separate_edges"] == 4
        saved = next(item for item in Workspace(workspace.path).console_history() if item["code"] == code)
        assert saved["outputs"] == run["outputs"]
    finally:
        session.close()
