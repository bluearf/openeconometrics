"""Network display semantics, bounded protocol and portable export safety."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

import openecon_charts as charts
from openecon_charts.network import network_options, validate_network


def payload(*, sampled=False):
    return {
        "nodes": [{"id": 5, "label": "A", "degree": 2, "group": 0},
                  {"id": 9, "label": "B", "degree": 1, "group": 0},
                  {"id": 11, "label": "Isolated", "degree": 0, "group": 1}],
        "edges": [{"source": 5, "target": 9, "weight": 2.5}],
        "directed": True, "node_count": 10 if sampled else 3,
        "edge_count": 20 if sampled else 1, "shown_node_count": 3,
        "shown_edge_count": 1, "sampled": sampled,
        "selection": "highest-degree induced subgraph",
    }


class Graph:
    def __init__(self, data=None):
        self.data = payload() if data is None else data
        self.calls = []

    def to_plot_data(self, **options):
        self.calls.append(options)
        return self.data


def spec(data=None, **changes):
    value = payload() if data is None else data
    fields = {"kind": "network", "title": "Network", "x_label": "", "y_label": "",
              "data": [], "sample_n": value["shown_node_count"], "total_n": value["node_count"],
              "dropped_n": 0, "config": {"network": value, "options": {}}}
    return charts.PlotSpec(**{**fields, **changes})


def test_protocol_preserves_full_counts_and_degree_while_displaying_a_subset():
    graph = Graph(payload(sampled=True))
    plot = charts.network(graph, title="Study network", max_nodes=10, max_edges=20, seed=17)
    assert graph.calls == [{"max_nodes": 10, "max_edges": 20, "seed": 17}]
    assert (plot.kind, plot.data, plot.sample_n, plot.total_n, plot.dropped_n) == ("network", [], 3, 10, 0)
    dumped = plot.model_dump()
    assert dumped["config"]["network"] == graph.data
    assert dumped["config"]["network"]["nodes"][0]["degree"] == 2  # original graph degree
    graph.data["nodes"][0]["label"] = "changed"
    dumped["config"]["network"]["edges"][0]["weight"] = 999
    assert plot.config["network"]["nodes"][0]["label"] == "A"
    assert plot.config["network"]["edges"][0]["weight"] == 2.5


def test_default_limits_empty_graph_and_independent_fluent_presentation():
    empty = {**payload(), "nodes": [], "edges": [], "node_count": 0, "edge_count": 0,
             "shown_node_count": 0, "shown_edge_count": 0}
    graph = Graph(empty)
    plot = charts.network(graph)
    assert graph.calls == [{"max_nodes": 1000, "max_edges": 5000, "seed": 0}]
    assert plot.title == "Network" and plot.sample_n == plot.total_n == 0
    palette = ["#123", "#ABCDEF"]
    styled = plot.with_options(title="Empty study", width=960, height=640, palette=palette,
                               point_size=8, line_width=2, opacity=.5)
    assert styled.config["options"] == {"width": 960, "height": 640, "point_size": 8.,
                                        "line_width": 2., "opacity": .5,
                                        "palette": ["#112233", "#abcdef"]}
    palette[0] = "#fff"
    styled.config["network"]["selection"] = "changed"
    assert plot.config["options"] == {}
    assert plot.config["network"]["selection"] == "highest-degree induced subgraph"


@pytest.mark.parametrize("changes", [
    {"directed": 1}, {"sampled": 0}, {"sampled": True}, {"node_count": 2},
    {"node_count": True}, {"edge_count": 0}, {"shown_node_count": 2},
    {"shown_edge_count": 2}, {"node_count": 3.0}, {"edge_count": 2**53},
    {"node_count": -1}, {"nodes": ()}, {"edges": ()}, {"selection": ""},
    {"selection": "x" * 1001}, {"selection": "a\n"}, {"extra": {}},
])
def test_metadata_and_schema_inconsistencies_are_refused(changes):
    with pytest.raises((ValueError, TypeError)):
        validate_network({**payload(), **changes})


def test_false_sampled_cannot_hide_omitted_nodes_or_edges():
    for name in ("node_count", "edge_count"):
        value = payload()
        value[name] += 1
        with pytest.raises(ValueError, match="sampled"):
            validate_network(value)
        value["sampled"] = True
        assert validate_network(value)["sampled"] is True
    zero = {**payload(), "nodes": [], "edges": [], "node_count": 0, "edge_count": 1,
            "shown_node_count": 0, "shown_edge_count": 0, "sampled": True}
    with pytest.raises(ValueError, match="without nodes"):
        validate_network(zero)


@pytest.mark.parametrize("changes", [
    {"id": True}, {"id": -1}, {"id": 2**53}, {"id": 5.0}, {"group": -1},
    {"group": True}, {"label": None}, {"label": "\x00"}, {"label": "\u0085"},
    {"label": "\ud800"}, {"label": "é" * 2049}, {"label": "x" * 4097},
    {"degree": float("nan")}, {"degree": float("inf")}, {"degree": -1},
    {"degree": True}, {"degree": 2**53}, {"unknown": "x"},
])
def test_node_ids_labels_and_degrees_are_safe(changes):
    value = payload()
    value["nodes"][0].update(changes)
    with pytest.raises((ValueError, TypeError)):
        validate_network(value)


@pytest.mark.parametrize("changes", [
    {"source": 100}, {"target": 0}, {"target": -1}, {"source": True}, {"source": 5.0},
    {"weight": float("inf")}, {"weight": float("nan")}, {"weight": True},
    {"weight": "1"}, {"weight": 2**53}, {"extra": None},
])
def test_edge_endpoints_weights_and_schema_are_safe(changes):
    value = payload()
    value["edges"][0].update(changes)
    with pytest.raises((ValueError, TypeError)):
        validate_network(value)


def test_duplicate_ids_bounded_lists_and_aggregate_utf8_label_budget():
    value = payload()
    value["nodes"][1]["id"] = 5
    with pytest.raises(ValueError, match="unique"):
        validate_network(value)
    for name, count in (("nodes", 100001), ("edges", 1000001)):
        value = payload()
        value[name] = [value[name][0]] * count
        with pytest.raises(ValueError, match="at most"):
            validate_network(value)
    def labeled(count):
        return {**payload(), "nodes": [{"id": i, "label": "é" * 2048, "degree": 0, "group": i}
                                       for i in range(count)], "edges": [], "node_count": count,
                "edge_count": 0, "shown_node_count": count, "shown_edge_count": 0}
    assert len(validate_network(labeled(4096))["nodes"]) == 4096
    with pytest.raises(ValueError, match="16 MiB"):
        validate_network(labeled(4097))


def test_payload_copy_preserves_labels_signed_weights_and_original_degree():
    value = payload()
    value["nodes"][0].update(label="1", degree=123.5)
    value["nodes"][1]["label"] = "1"  # labels are not IDs; equal display text is legal
    value["edges"][0]["weight"] = -2.5
    copied = validate_network(value)
    assert copied == value
    copied["nodes"][0]["group"] = 99
    assert value["nodes"][0]["group"] == 0


@pytest.mark.parametrize("options", [
    {"width": 319}, {"height": 1601}, {"width": True}, {"height": 500.5},
    {"point_size": 0}, {"point_size": 25}, {"line_width": .1}, {"line_width": 13},
    {"opacity": -1}, {"opacity": float("nan")}, {"opacity": True},
    {"color": "url(javascript:alert(1))"}, {"color": "red"}, {"palette": []},
    {"palette": ["#fff"] * 65}, {"palette": "#fff"}, {"palette": None},
    {"xlim": [0, 1]}, {"x_label": "x"}, {"x_scale": "log"}, {"grid": True},
    {"legend": "yes"}, {"annotations": [{"text": "missing position"}]}, {"__proto__": {}},
])
def test_inapplicable_or_invalid_options_are_refused_before_calling_the_graph(options):
    graph = Graph()
    with pytest.raises((TypeError, ValueError)):
        charts.network(graph, **options)
    assert graph.calls == []
    with pytest.raises((TypeError, ValueError)):
        charts.network(Graph()).with_options(**options)


@pytest.mark.parametrize("options", [
    {"max_nodes": 100001}, {"max_edges": 1000001}, {"max_nodes": 0}, {"max_edges": 0},
    {"max_nodes": True}, {"max_edges": 1.5}, {"seed": -1}, {"seed": True},
    {"seed": 1.5}, {"title": "bad\n"}, {"title": "\ud800"},
])
def test_display_limit_and_title_errors_are_cheap(options):
    graph = Graph()
    with pytest.raises((TypeError, ValueError)):
        charts.network(graph, **options)
    assert graph.calls == []
    with pytest.raises(TypeError, match="to_plot_data"):
        charts.network({})


def test_graph_must_honor_the_requested_display_caps():
    with pytest.raises(ValueError, match="honor"):
        charts.network(Graph(), max_nodes=2)
    value = payload()
    value["edges"] *= 2
    value["edge_count"] = value["shown_edge_count"] = 2
    with pytest.raises(ValueError, match="honor"):
        charts.network(Graph(value), max_edges=1)


@pytest.mark.parametrize("changes", [
    {"sample_n": 2}, {"total_n": 4}, {"dropped_n": 1}, {"x_label": "x"},
    {"y_label": "y"}, {"data": [{"x": 0}]}, {"config": {}}, {"config": None},
])
def test_raw_plotspec_envelope_cannot_disagree_with_the_network(changes):
    with pytest.raises((TypeError, ValueError)):
        spec(**changes)


def test_exports_reject_postcreation_mutations_before_writing(tmp_path):
    plot = charts.network(Graph())
    plot.config["network"]["edges"][0]["target"] = 999
    for export in (plot.model_dump, plot.to_html, plot.to_latex):
        with pytest.raises(ValueError, match="endpoint"):
            export()
    destination = tmp_path / "refused.tex"
    with pytest.raises(ValueError):
        plot.to_latex(destination)
    assert not destination.exists()
    plot = charts.network(Graph())
    plot.config["options"]["xlim"] = [0, 1]
    with pytest.raises(TypeError, match="Unsupported"):
        plot.model_dump()


def test_latex_is_a_bounded_publication_summary_with_honest_sample_metadata(tmp_path):
    value = payload(sampled=True)
    value["nodes"][0]["label"] = "NODE_MUST_NOT_BE_LISTED"
    value["selection"] = r"degree_1 & \input{secret}"
    plot = charts.network(Graph(value), title="Study_1 & network")
    source = plot.to_latex(label="tab:network")
    marker = json.loads(source.splitlines()[0].removeprefix("% OpenEcon chart: "))
    assert marker == {"kind": "network", "sample_n": 3, "total_n": 10, "dropped_n": 0}
    details = json.loads(source.splitlines()[1].removeprefix("% Network display metadata: "))
    assert details["node_count"] == 10 and details["edge_count"] == 20 and details["sampled"]
    assert r"\toprule" in source and r"\bottomrule" in source
    assert r"Nodes & 10 & 3 \\" in source and r"Edges & 20 & 1 \\" in source
    assert r"\caption{Study\_1 \& network}" in source
    assert r"\label{tab:network}" in source
    assert "subset of the original graph" in source
    assert r"degree\_1 \& \textbackslash{}input\{secret\}" in source
    assert "NODE_MUST_NOT_BE_LISTED" not in source
    assert r"\begin{tikzpicture}" not in source
    assert len(source) < 2500
    document = plot.to_latex(standalone=True, caption="Network summary")
    assert r"\usepackage{booktabs}" in document
    assert document.count(r"\begin{document}") == document.count(r"\end{document}") == 1
    assert r"\caption{Network summary}" in document
    output = StringIO()
    assert plot.to_latex(output) == output.getvalue()
    path = tmp_path / "network.tex"
    assert plot.to_latex(path) == path.read_text()
    assert "The complete graph is displayed." in charts.network(Graph()).to_latex()


def test_offline_html_contains_assets_safe_payload_and_no_remote_dependencies():
    graph = Graph()
    graph.data["nodes"][0]["label"] = "</script><script>window.NETWORK_INJECTION=1</script>"
    plot = charts.network(graph, color="#123", palette=["#fff", "#456"])
    document = plot.to_html()
    encoded = re.search(r'<script type="application/json" id="chart-data">(.*?)</script>', document, re.S).group(1)
    assert json.loads(encoded) == plot.model_dump()
    assert "\\u003c/script\\u003e" in encoded
    assert "<script>window.NETWORK_INJECTION" not in document
    assert "OpenEconNetwork" in document
    assert "connect-src &#x27;none&#x27;" in document
    assert "data:font/woff2;base64," in document
    assert not re.search(r'<(?:script|link)\b[^>]*(?:src|href)=[\"\']https?://', document)
    assert '<iframe' in plot._repr_html_() and 'sandbox="allow-scripts allow-downloads"' in plot._repr_html_()


def test_package_and_network_exports_need_only_the_standard_library():
    source = Path(__file__).resolve().parents[1] / "src"
    code = f'''
import sys
sys.path.insert(0, {str(source)!r})
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {{"torch", "numpy", "pandas", "scipy", "openecon", "networkx", "statsmodels"}}:
            raise AssertionError("unexpected optional dependency: " + fullname)
sys.meta_path.insert(0, Block())
import openecon_charts as charts
class Graph:
    def to_plot_data(self, **kwargs):
        return {payload()!r}
plot = charts.network(Graph())
assert plot.model_dump()["kind"] == "network"
assert "\\\\toprule" in plot.to_latex()
assert "OpenEconNetwork" in plot.to_html()
'''
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_normalized_options_are_independent_and_preserve_zero_opacity():
    values = {"color": "#ABC", "palette": ("#123", "#DEF"), "opacity": 0}
    original = deepcopy(values)
    normalized = network_options(values)
    assert normalized == {"color": "#aabbcc", "palette": ["#112233", "#ddeeff"], "opacity": 0.}
    assert values == original


def test_optional_grouping_preserves_old_persisted_payload_and_rejects_other_extensions():
    old = payload()
    copied = validate_network(old)
    assert copied == old and "grouping" not in copied
    restored = charts.PlotSpec(**spec(old).model_dump())
    assert restored.model_dump()["config"]["network"] == old
    for method in ("Weak components", "Leiden communities", "Louvain communities", "User groups"):
        value = {**payload(), "grouping": method}
        assert validate_network(value) == value
        assert charts.network(Graph(value)).model_dump()["config"]["network"]["grouping"] == method
    with pytest.raises(ValueError):
        validate_network({**payload(), "grouping": "User groups", "custom": "unsupported"})
    assert "Groups: Weak components." in restored.to_latex()
    assert "grouping" not in restored.config["network"], "Export does not mutate old saved graph metadata"


@pytest.mark.parametrize("grouping", [None, 1, True, "", "a\n", "a\u0085", "\ud800", "x" * 1001, "é" * 501])
def test_grouping_metadata_is_safe_nonempty_utf8_and_bounded(grouping):
    with pytest.raises((TypeError, ValueError)):
        validate_network({**payload(), "grouping": grouping})
    assert validate_network({**payload(), "grouping": "é" * 500})["grouping"] == "é" * 500


def test_groups_are_forwarded_only_when_explicit_and_keep_legacy_protocol_ducks_working():
    class LegacyGraph:
        def __init__(self):
            self.calls = []

        def to_plot_data(self, *, max_nodes, max_edges, seed):
            self.calls.append((max_nodes, max_edges, seed))
            return payload()

    legacy = LegacyGraph()
    assert charts.network(legacy).kind == "network"
    assert charts.network(legacy, groups=None).kind == "network"
    assert legacy.calls == [(1000, 5000, 0)] * 2
    groups = {5: 0, 9: 1, 11: 1}
    supplied = Graph({**payload(), "grouping": "User groups"})
    plot = charts.network(supplied, groups=groups)
    assert supplied.calls[0]["groups"] is groups
    assert plot.config["options"] == {}, "Groups are graph membership rather than presentation options"
    assert plot.config["network"]["grouping"] == "User groups"


def test_native_group_mapping_and_community_table_reach_the_chart_and_export_notes():
    oe = pytest.importorskip("openecon")
    graph = oe.network({"source": [0], "target": [1]}, nodes=[0, 1, 2])
    user = charts.network(graph, groups={0: "team_a", 1: "team_b", 2: "team_b"})
    assert user.config["network"]["grouping"] == "User groups"
    groups = {node["label"]: node["group"] for node in user.config["network"]["nodes"]}
    assert groups["0"] != groups["1"] == groups["2"]
    community = oe.DataFrame({"node": [2, 0, 1], "community": [8, 4, 4]})
    community.attrs["method"] = "leiden"
    plot = charts.network(graph, groups=community)
    assert plot.config["network"]["grouping"] == "Leiden communities"
    groups = {node["label"]: node["group"] for node in plot.config["network"]["nodes"]}
    assert groups["0"] == groups["1"] != groups["2"]
    source = plot.to_latex()
    metadata = json.loads(source.splitlines()[1].removeprefix("% Network display metadata: "))
    assert metadata["grouping"] == "Leiden communities"
    assert "Groups: Leiden communities." in source
    assert plot.model_dump()["config"]["network"]["grouping"] == "Leiden communities"


def test_grouping_latex_is_escaped_and_offline_html_keeps_metadata_in_safe_json():
    grouping = r"User_groups & \input{secret} </script>"
    plot = charts.network(Graph({**payload(sampled=True), "grouping": grouping}))
    source = plot.to_latex()
    assert r"Groups: User\_groups \& \textbackslash{}input\{secret\}" in source
    assert "\\input{secret}" not in source.split(r"\emph{Notes:}", 1)[1]
    document = plot.to_html()
    encoded = re.search(r'<script type="application/json" id="chart-data">(.*?)</script>', document, re.S).group(1)
    assert json.loads(encoded)["config"]["network"]["grouping"] == grouping
    assert "\\u003c/script\\u003e" in encoded
