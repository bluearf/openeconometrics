"""Code-first annotations and portable publication round trips."""
import html as html_module

import pytest

from openecon_charts import PlotSpec
from openecon_charts.network import network


def payload():
    return {"nodes": [{"id": 0, "label": "1", "degree": 1, "group": 0,
                       "identity": {"type": "integer", "value": "1"}},
                      {"id": 1, "label": "1", "degree": 1, "group": 0,
                       "identity": {"type": "string", "value": "1"}}],
            "edges": [{"source": 0, "target": 1, "weight": 1}], "directed": False,
            "node_count": 2, "edge_count": 1, "shown_node_count": 2,
            "shown_edge_count": 1, "sampled": False, "selection": "All nodes and edges"}


class Graph:
    def to_plot_data(self, **kwargs):
        return payload()


def graph():
    return Graph()


def test_node_annotations_bind_original_typed_identity_and_roundtrip(tmp_path):
    original = network(graph())
    plot = original.annotate_node("Number", node=1).annotate_node("Text", node="1", color="#123")
    notes = plot.model_dump()["config"]["options"]["annotations"]
    assert [item["node_id"] for item in notes] == [0, 1]
    assert "annotations" not in original.config["options"]
    path = plot.save_view(tmp_path / "network.json")
    assert PlotSpec.load_view(path).model_dump() == plot.model_dump()
    with pytest.raises(ValueError, match="absent"):
        plot.annotate_node("Missing", node="absent")
    with pytest.raises(TypeError, match="identity"):
        plot.annotate_node("Boolean", node=True)


def test_network_layout_coordinate_annotation_and_unsupported_cartesian_options():
    plot = network(graph()).annotate("Reference", x=10, y=-5, color="#315f91")
    assert plot.config["options"]["annotations"] == [
        {"text": "Reference", "x": 10.0, "y": -5.0, "color": "#315f91"}]
    for options in ({"arrow": True}, {"coords": "axes"}, {"dx": 5}):
        with pytest.raises(ValueError, match="layout coordinates"):
            plot.annotate("Reference", x=0, y=0, **options)


def test_portable_network_contains_offline_worker_webgl_and_true_type_pdf_font():
    html = network(graph(), layout="forceatlas2").to_html()
    assert "worker-src blob:" in html and "connect-src 'none'" in html_module.unescape(html)
    assert "OpenEconNetworkWorkerSource" in html
    assert "OpenEconNetworkPDF" in html and "OpenEconNetworkFont" in html
    assert "FontFile2 requires a TrueType outline font" in html
