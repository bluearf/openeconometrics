"""Portable graph-view protocol, layout/appearance schema and timeline guards."""
from copy import deepcopy
import json

import pytest

import openecon_charts as charts
from openecon_charts.network import network_options, validate_network, validate_network_presentation


def payload():
    return {"nodes": [
        {"id": 1, "label": "1", "degree": 1, "group": 0,
         "identity": {"type": "integer", "value": "1"}, "attrs": {"score": 5, "category": "A"},
         "x": 5, "y": 2, "fixed": True, "longitude": 28., "latitude": 41.},
        {"id": 2, "label": "1", "degree": 1, "group": 0,
         "identity": {"type": "string", "value": "1"}, "attrs": {"score": 8, "category": "B"},
         "x": 12, "y": 4, "fixed": False, "longitude": -73., "latitude": 40.},
    ], "edges": [{"source": 1, "target": 2, "weight": 3, "attrs": {"kind": "trade"}}],
        "directed": True, "node_count": 2, "edge_count": 1, "shown_node_count": 2,
        "shown_edge_count": 1, "sampled": False, "selection": "complete"}


class Graph:
    def __init__(self, data=None):
        self.data = payload() if data is None else data
        self.calls = []

    def to_plot_data(self, **options):
        self.calls.append(options)
        return self.data


@pytest.mark.parametrize("layout, options", [
    ("d3-force", {"charge": -80, "link_distance": 50, "theta": .7, "collision": False}),
    ("forceatlas2", {"scaling": 2, "gravity": 1, "strong_gravity": True, "linlog": True,
                   "outbound_attraction_distribution": True, "edge_weight_influence": 1,
                   "jitter_tolerance": .5, "slowdown": 2, "theta": 1}),
    ("circular", {"radius": 100, "angle": -90}),
    ("grid", {"spacing": 40, "columns": 2}),
    ("radial", {"spacing": 60, "root": 1, "angle": 30}),
    ("hierarchical", {"spacing": 30, "direction": "LR"}),
    ("geographic", {"projection": "mercator", "scale": 3}),
    ("community", {"radius": 200, "spacing": 30}),
    ("fixed", {}),
])
def test_every_layout_and_its_options_are_portable_independent_and_finite(layout, options):
    settings = {"iterations": 50, "work_limit": 10000, "time_limit_ms": 3000, **options}
    value = charts.network(Graph(), layout=layout, layout_options=settings)
    assert value.config["options"]["layout"] == layout
    assert value.config["options"]["layout_options"] == settings
    assert json.dumps(value.model_dump(), allow_nan=False)
    settings["iterations"] = 1
    assert value.config["options"]["layout_options"]["iterations"] == 50
    assert layout in value.to_html()


def test_every_option_roundtrips_with_typed_appearance_domains_and_saved_positions():
    options = dict(layout="fixed", layout_options={"iterations": 1},
        width=900, height=600, color="#123456", palette=["#123", "#abc"],
        point_size=12, line_width=3, opacity=.6,
        node_size={"field": "attrs.score", "range": [2, 24], "domain": [0, 10], "missing": 4},
        node_color={"field": "attrs.category", "scale": "categorical", "domain": [1, "1", True],
                    "range": ["#123", "#abc", "#fff"], "missing": "#999"},
        node_label={"field": "attrs.category", "domain": ["A", "B"], "range": ["Alpha", "Beta"], "missing": "Unknown"},
        edge_width={"field": "weight", "scale": "sqrt", "range": [.5, 8]},
        edge_color={"field": "attrs.kind", "scale": "categorical", "missing": "#111"},
        filters=[{"scope": "nodes", "field": "attrs.score", "op": "gte", "value": 5},
                 {"scope": "edges", "field": "attrs.kind", "op": "in", "value": ["trade"]}],
        annotations=[{"text": "Anchor", "node_id": 1, "color": "#123"},
                     {"text": "Note", "x": 0, "y": 2}], labels={"show": True, "min_zoom": .01, "max_count": 100},
        legend=True, timeline=False, view={"version": 1,
            "positions": [{"id": 1, "x": 30, "y": 50, "pinned": True,
                           "identity": {"type": "integer", "value": "1"}}],
            "transform": {"k": 2, "x": 10, "y": 12}, "selected": 1,
            "filters": [{"scope": "nodes", "field": "degree", "op": "gt", "value": 0}]})
    value = charts.network(Graph(), **options)
    assert value.config["options"]["node_size"]["scale"] == "linear"
    assert value.config["options"]["node_label"]["scale"] == "categorical"
    original = deepcopy(value.model_dump())
    options["view"]["positions"][0]["x"] = 999
    options["node_color"]["range"][0] = "#000"
    assert value.model_dump() == original
    assert value.config["network"]["nodes"][0]["identity"] != value.config["network"]["nodes"][1]["identity"]


@pytest.mark.parametrize("changes", [
    {"identity": {"type": "integer", "value": "01"}},
    {"identity": {"type": "integer", "value": "-0"}},
    {"identity": {"type": "integer", "value": str(2**256)}},
    {"identity": {"type": "boolean", "value": "true"}},
    {"identity": {"type": "string", "value": " "}},
    {"x": float("nan")}, {"x": 1e10}, {"fixed": 1}, {"pinned": False},
    {"longitude": 181}, {"latitude": 91}, {"attrs": {"nested": {}}},
    {"attrs": {"constructor": "x"}}, {"attrs": {"__proto__": "x"}},
    {"attrs": {"big": 2**53}}, {"attrs": {"notfinite": float("inf")}},
    {"attrs": {"control": "a\x00"}}, {"attrs": {"oversized": "a" * 4097}},
    {"attrs": {str(index): index for index in range(65)}},
])
def test_optional_node_fields_are_bounded_safe_and_strict(changes):
    data = payload()
    data["nodes"][0].update(changes)
    with pytest.raises((TypeError, ValueError)):
        validate_network(data)


def test_duplicate_identity_and_incomplete_coordinates_are_rejected():
    data = payload()
    data["nodes"][1]["identity"] = deepcopy(data["nodes"][0]["identity"])
    with pytest.raises(ValueError, match="identities"):
        validate_network(data)
    data = payload()
    data["nodes"][1].pop("y")
    with pytest.raises(ValueError, match="both"):
        validate_network(data)
    data = payload()
    data["nodes"][0].pop("x")
    data["nodes"][0].pop("y")
    with pytest.raises(ValueError, match="coordinates"):
        validate_network(data)


@pytest.mark.parametrize("options", [
    {"layout": "unsupported"}, {"layout": "grid", "layout_options": {"charge": -10}},
    {"layout_options": {"iterations": 0}}, {"layout_options": {"iterations": 2001}},
    {"layout_options": {"time_limit_ms": 120001}},
    {"layout": "forceatlas2", "layout_options": {"gravity": -1}},
    {"layout": "forceatlas2", "layout_options": {"linlog": 1}},
    {"layout": "hierarchical", "layout_options": {"direction": "diagonal"}},
    {"node_size": 257}, {"edge_width": .01}, {"node_color": "red"},
    {"node_size": {"field": "degree", "scale": "code"}},
    {"node_size": {"field": "degree", "range": [2, 4, 6]}},
    {"node_size": {"field": "degree", "scale": "log", "domain": [0, 10]}},
    {"node_size": {"field": "degree", "domain": [2, 1]}},
    {"node_size": {"field": "degree", "missing": {"field": "degree"}}},
    {"node_color": {"field": "group", "scale": "categorical", "domain": [1, 1.], "range": ["#fff", "#000"]}},
    {"node_label": {"field": "degree", "scale": "log"}},
    {"filters": [{"scope": "nodes", "field": "id", "op": "eval", "value": "1"}]},
    {"filters": [{"scope": "edges", "field": "weight", "op": "gt", "value": "1"}]},
    {"filters": [{"scope": "nodes", "field": "id", "op": "in", "value": [1] * 10001}]},
    {"annotations": [{"text": "x"}]}, {"annotations": [{"text": "x", "node_id": 1, "x": 1}]},
    {"labels": {"show": 1, "min_zoom": 1, "max_count": 1}},
    {"labels": {"show": True, "min_zoom": 1, "max_count": 10001}},
    {"timeline": 1}, {"legend": 1}, {"view": {"version": 2, "positions": [], "transform": {"k": 1, "x": 0, "y": 0}}},
    {"view": {"version": 1, "positions": [], "transform": {"k": 20, "x": 0, "y": 0}}},
])
def test_invalid_options_are_refused_before_graph_execution(options):
    graph = Graph()
    with pytest.raises((ValueError, TypeError)):
        charts.network(graph, **options)
    assert graph.calls == []


@pytest.mark.parametrize("options", [
    {"annotations": [{"text": "missing", "node_id": 3}]},
    {"layout": "radial", "layout_options": {"root": 3}},
    {"view": {"version": 1, "positions": [{"id": 3, "x": 0, "y": 0, "pinned": True}],
              "transform": {"k": 1, "x": 0, "y": 0}}},
    {"view": {"version": 1, "positions": [], "selected": 3, "transform": {"k": 1, "x": 0, "y": 0}}},
])
def test_presentation_references_must_exist_in_the_saved_graph(options):
    with pytest.raises(ValueError):
        charts.network(Graph(), **options)


def test_layout_prerequisites_apply_to_all_timeline_frames():
    data = payload()
    second = deepcopy(data)
    second["nodes"][0].pop("longitude")
    data["frames"] = [{"label": "before", "network": payload()}, {"label": "after", "network": second}]
    with pytest.raises(ValueError, match="longitude"):
        charts.network(Graph(data), layout="geographic")
    data = payload()
    data["nodes"][0].pop("x")
    data["nodes"][0].pop("y")
    data["nodes"][0]["fixed"] = False
    with pytest.raises(ValueError, match="coordinates"):
        charts.network(Graph(data), layout="fixed")
    data = payload()
    data["edges"][0]["weight"] = -1
    with pytest.raises(ValueError, match="negative"):
        charts.network(Graph(data), layout="forceatlas2")


def test_timeline_frame_schema_identity_and_directedness_are_strict():
    data = payload()
    data["frames"] = [{"label": "first", "network": payload()}, {"label": "second", "network": payload()}]
    copied = validate_network(data)
    assert len(copied["frames"]) == 2
    assert copied == data
    data["frames"][1]["network"]["nodes"][0]["identity"]["value"] = "7"
    with pytest.raises(ValueError, match="identity"):
        validate_network(data)
    data = payload()
    data["frames"] = [{"label": "second", "network": {**payload(), "directed": False}}]
    with pytest.raises(ValueError, match="directedness"):
        validate_network(data)
    data["frames"][0]["network"]["frames"] = []
    with pytest.raises(ValueError, match="recursively"):
        validate_network(data)


def test_timeline_aggregate_caps_and_label_memory_budget_are_enforced():
    data = payload()
    data["frames"] = [{"label": "x", "network": payload()}] * 61
    with pytest.raises(ValueError, match="60"):
        validate_network(data)
    count = 33_334
    base = {**payload(), "nodes": [{"id": index, "label": str(index), "degree": 0, "group": 0}
                                    for index in range(count)], "edges": [], "node_count": count,
            "edge_count": 0, "shown_node_count": count, "shown_edge_count": 0}
    base["frames"] = [{"label": "x", "network": {key: value for key, value in base.items() if key != "frames"}}] * 2
    with pytest.raises(ValueError, match="aggregate"):
        validate_network(base)
    data = {**payload(), "nodes": [{"id": index, "label": "x" * 4096, "degree": 0, "group": 0}
                                    for index in range(4097)], "edges": [], "node_count": 4097,
            "edge_count": 0, "shown_node_count": 4097, "shown_edge_count": 0}
    with pytest.raises(ValueError, match="16 MiB"):
        validate_network(data)


def test_legacy_protocol_still_works_without_optional_fields_or_new_options():
    data = payload()
    for row in data["nodes"]:
        for key in list(row):
            if key not in {"id", "label", "degree", "group"}:
                del row[key]
    data["edges"][0].pop("attrs")
    graph = Graph(data)
    value = charts.network(graph)
    assert graph.calls == [{"max_nodes": 1000, "max_edges": 5000, "seed": 0}]
    assert value.config["options"] == {} and value.config["network"] == data
    validate_network_presentation(value.config["network"], network_options({}))


def test_partial_label_options_zero_zoom_and_explicit_seed_are_reproducible():
    value = charts.network(Graph(), seed=37, labels={"min_zoom": 0, "max_count": 32})
    assert value.config["options"]["labels"] == {"show": True, "min_zoom": 0., "max_count": 32}
    assert value.config["options"]["seed"] == 37
    with pytest.raises(ValueError):
        charts.network(Graph(), seed=2**32)
    with pytest.raises(ValueError):
        network_options({"seed": True})


def test_saved_timeline_frame_index_and_positions_refer_to_active_frame():
    first, second = payload(), payload()
    second["nodes"][0]["id"] = 3
    second["nodes"][0]["identity"] = {"type": "integer", "value": "3"}
    second["edges"][0]["source"] = 3
    first["frames"] = [{"label": "before", "network": payload()}, {"label": "after", "network": second}]
    view = {"version": 1, "positions": [{"id": 3, "x": 1, "y": 2, "pinned": True,
                                        "identity": {"type": "integer", "value": "3"}}],
            "transform": {"k": 1, "x": 0, "y": 0}, "selected": 3}
    chart = charts.network(Graph(first), frame_index=1, view=view)
    assert chart.config["options"]["frame_index"] == 1
    with pytest.raises(ValueError, match="active"):
        charts.network(Graph(first), frame_index=0, view=view)
    with pytest.raises(ValueError, match="available"):
        charts.network(Graph(first), frame_index=2)
    with pytest.raises(ValueError, match="available"):
        charts.network(Graph(), frame_index=1)


def test_saved_position_identity_binding_refuses_reordered_entities_and_supports_legacy_ids():
    view = {"version": 1, "positions": [{"id": 1, "x": 1, "y": 2,
                                        "identity": {"type": "integer", "value": "1"}}],
            "transform": {"k": 1, "x": 0, "y": 0}}
    assert charts.network(Graph(), view=view).config["options"]["view"]["positions"][0]["pinned"] is False
    changed = payload()
    changed["nodes"][0]["identity"] = {"type": "integer", "value": "99"}
    with pytest.raises(ValueError, match="identity"):
        charts.network(Graph(changed), view=view)
    unbound = deepcopy(view)
    unbound["positions"][0].pop("identity")
    with pytest.raises(ValueError, match="binding"):
        charts.network(Graph(), view=unbound)
    legacy = payload()
    for node in legacy["nodes"]:
        node.pop("identity")
    assert charts.network(Graph(legacy), view=unbound)
    assert charts.network(Graph(legacy), view=view)
    view["positions"][0]["identity"] = {"type": "string", "value": "1"}
    with pytest.raises(ValueError, match="identity"):
        charts.network(Graph(legacy), view=view)
