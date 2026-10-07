"""Synthetic network workbench configured entirely from Python.

Paste this entire file into an OpenEconometrics Python editor and press Run.
The panel displays the tables, nine layouts and snapshot timeline directly.
No files are written unless an output folder is explicitly requested.

Or run from the repository with an environment containing openecon:
    python docs/examples/network_workbench.py --output /tmp/openeconometrics-workbench

Saving these specifications does not run the browser layouts. Open the generated
HTML to render them. PNG/SVG/PDF export uses the mounted chart's export menu.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import openecon as oe
from openecon_charts import PlotSpec


NAVY = "#14263d"
BLUE = "#4c78a8"
LAYOUT_SETTINGS = {
    "d3-force": {"charge": -55, "link_distance": 80, "collision": True},
    "forceatlas2": {"scaling": 2, "gravity": 1, "linlog": True,
                   "edge_weight_influence": 1, "jitter_tolerance": 1},
    "circular": {"radius": 180, "angle": -90},
    "grid": {"columns": 4, "spacing": 90},
    "radial": {"spacing": 90, "angle": -90},
    "hierarchical": {"spacing": 90, "direction": "LR"},
    "geographic": {"projection": "equirectangular", "scale": 8},
    "community": {"spacing": 50, "radius": 200},
    "fixed": {},
}


def run_example(output_dir: str | Path | None = None, *, display_callback=None) -> dict:
    """Build the synthetic example; optionally display it or save its artifacts."""
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    # Integer 1 and string "1" remain distinct node identities throughout.
    labels = [1, "1", "Ada", "Ben", "Cleo", "Dara", "Eli", "isolated"]
    graph = oe.network(
        {"source": [1, "1", "Ada", "Ada", "Ben", "Cleo", "Dara", "Eli"],
         "target": ["Ada", "Ada", "Ben", "Cleo", "Cleo", "Dara", "Eli", "Ada"],
         "weight": [1, 2, 4, 2, 3, 5, 2, 1]},
        nodes=labels, weight="weight", directed=True, max_memory_mb=128,
    )
    attributes = {
        node: {"label": "Integer one" if node == 1 else "String one" if node == "1" else str(node),
               "team": "lab" if index % 2 else "field", "score": 10 + index * 10,
               "longitude": 20.0 + index * 2, "latitude": 35.0 + index,
               "temporary": "remove this column"}
        for index, node in enumerate(labels)
    }
    graph = graph.with_attributes(
        nodes=attributes, edges={("Ada", "Ben"): {"year": 2024}},
        graph_attributes={"name": "Synthetic workbench", "synthetic": True},
    )
    initial = graph.rename_attributes({"team": "sector"}).drop_attributes("temporary")

    # Edits return new graphs. Renaming is simultaneous and does not merge nodes.
    edited = initial.edit_nodes(rename={"Ben": "Benoit"}, remove=["isolated"], add=["newcomer"])
    edited = edited.edit_edges(
        add=[{"source": "newcomer", "target": "Ada", "weight": 2,
              "attrs": {"year": 2026}}],
        remove=[("Eli", "Ada")], weights={("Ada", "Benoit"): 7},
    )
    edited = edited.update_attributes(
        nodes={"newcomer": {"label": "Newcomer", "sector": "field", "score": 30,
                            "longitude": 22.0, "latitude": 38.0}},
        edges={("newcomer", "Ada"): {"year": 2026}},
        graph_attributes={"reviewed": True},
    )
    coordinates = {node: {"x": (index % 4 - 1.5) * 100,
                          "y": (index // 4 - .5) * 100, "fixed": False}
                   for index, node in enumerate(edited.nodes().node)}
    edited = edited.with_positions(coordinates, fixed=False)
    edited = edited.update_attributes(nodes={1: {"fixed": True}})
    assert graph.node_count == initial.node_count == 8
    assert "Ben" in initial.nodes().node.tolist() and "Benoit" not in initial.nodes().node.tolist()
    assert "Ben" not in edited.nodes().node.tolist() and "Benoit" in edited.nodes().node.tolist()

    # Core filtering changes the graph; chart filters below change only its view.
    filtered = edited.filter(
        nodes={"field": "degree", "op": "gte", "value": 1},
        edges={"field": "weight", "op": "gte", "value": 2},
    )
    callable_selection = filtered.filter(nodes=lambda row: row["attrs"].get("score", 0) >= 20)
    assert callable_selection.node_count < filtered.node_count
    nodes, edges = filtered.nodes(), filtered.edges()
    assert "attr.sector" in nodes and "attr.year" in edges
    if output is not None:
        nodes.to_csv(output / "nodes.csv", index=False)
        edges.to_csv(output / "edges.csv", index=False)
        filtered.summary().to_latex(buf=output / "summary.tex", index=False)
    if display_callback is not None:
        display_callback(filtered.summary())
        display_callback(nodes)

    payload = filtered.to_plot_data(max_nodes=100, max_edges=100)
    ada_id = next(node["id"] for node in payload["nodes"]
                  if node["identity"] == {"type": "string", "value": "Ada"})
    sectors = dict(zip(nodes.node, nodes["attr.sector"]))
    groups = {node: 0 if sector == "lab" else 1 for node, sector in sectors.items()}
    appearance = {
        "width": 1000, "height": 600, "point_size": 5, "line_width": 1,
        "opacity": .9, "palette": [NAVY, BLUE], "legend": True,
        "node_size": {"field": "attrs.score", "domain": [0, 100], "range": [4, 14]},
        "node_color": {"field": "attrs.sector", "scale": "categorical",
                       "domain": ["lab", "field"], "range": [NAVY, BLUE]},
        "node_label": "label",
        "edge_width": {"field": "weight", "domain": [2, 7], "range": [.75, 3]},
        "edge_color": {"field": "attrs.year", "scale": "categorical",
                       "domain": [2024, 2026], "range": [NAVY, BLUE], "missing": BLUE},
        "labels": {"show": True, "min_zoom": .1, "max_count": 80},
        "filters": [{"scope": "edges", "field": "weight", "op": "gte", "value": 3}],
        "annotations": [{"text": "Ada", "node_id": ada_id, "color": NAVY},
                        {"text": "Synthetic example", "x": 0, "y": -150, "color": NAVY}],
    }
    charts = {}
    for layout, settings in LAYOUT_SETTINGS.items():
        options = {"iterations": 100, "work_limit": 2_000_000,
                   "time_limit_ms": 5000, **settings}
        if layout == "radial":
            options["root"] = ada_id
        chart = oe.plot.network(
            filtered, groups=groups, title=f"Network workbench: {layout}",
            max_nodes=100, max_edges=100, seed=7,
            layout=layout, layout_options=options, **appearance,
        )
        charts[layout] = chart
        if output is not None:
            chart.save_html(output / f"layout-{layout}.html")
            chart.save_view(output / f"layout-{layout}.json")
        if display_callback is not None:
            display_callback(chart)

    # Full PlotSpec JSON is portable. Typed identities bind positions to entities.
    fixed = charts["fixed"]
    fixed_payload = fixed.model_dump()
    view = {
        "version": 1,
        "positions": [{"id": row["id"], "identity": deepcopy(row["identity"]),
                       "x": row["x"], "y": row["y"], "pinned": row.get("fixed", False)}
                      for row in fixed_payload["config"]["network"]["nodes"]],
        "transform": {"k": 1, "x": 500, "y": 300}, "selected": ada_id,
        "filters": deepcopy(appearance["filters"]),
    }
    saved = fixed.with_options(view=view)
    if output is not None:
        saved.save_view(output / "workbench.json")
        saved.save_html(output / "workbench.html")
        reopened = PlotSpec.load_view(output / "workbench.json")
    else:
        reopened = PlotSpec(**saved.model_dump())
    assert reopened.model_dump() == saved.model_dump()
    restored = oe.Network.from_plot_data(reopened.config["network"])
    assert restored.node_count == filtered.node_count and restored.edge_count == filtered.edge_count
    identities = reopened.config["network"]["nodes"]
    assert any(row["identity"] == {"type": "integer", "value": "1"} for row in identities)
    assert any(row["identity"] == {"type": "string", "value": "1"} for row in identities)

    # Snapshot IDs and order are explicit. Colors come from each graph's attributes.
    layers = oe.network_snapshots({"before": initial, "after": edited}, ordered=True)
    timeline = oe.plot.network(
        layers, title="Synthetic snapshots", timeline=True, frame_index=1,
        layout="geographic", layout_options={"projection": "equirectangular", "scale": 8,
                                             "work_limit": 2_000_000, "time_limit_ms": 5000},
        max_nodes=100, max_edges=100, seed=7, width=1000, height=600,
        palette=[NAVY, BLUE], legend=True,
        node_color=appearance["node_color"], node_size=appearance["node_size"],
        labels=appearance["labels"],
    )
    if output is not None:
        timeline.save_html(output / "timeline.html")
        timeline.save_view(output / "timeline.json")
        assert PlotSpec.load_view(output / "timeline.json").model_dump() == timeline.model_dump()
    if display_callback is not None:
        display_callback(timeline)
    assert [frame["label"] for frame in timeline.config["network"]["frames"]] == ["before", "after"]
    assert timeline.config["options"]["frame_index"] == 1

    manifest = {
        "synthetic": True, "sdk_version": oe.__version__,
        "source_nodes": graph.node_count, "source_edges": graph.edge_count,
        "edited_nodes": edited.node_count, "edited_edges": edited.edge_count,
        "filtered_nodes": filtered.node_count, "filtered_edges": filtered.edge_count,
        "layouts": list(charts), "snapshots": layers.snapshot_count,
        "full_view_roundtrip": True, "typed_identity_roundtrip": True,
        "graph_from_display_roundtrip": True,
        "python_layout_executed": False, "python_pdf_api": False,
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}
                 if output is not None else {},
    }
    if output is not None:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/openeconometrics-workbench"))
    args = parser.parse_args(argv)
    print(json.dumps(run_example(args.output), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    panel_display = globals().get("display")
    if callable(panel_display):
        example_result = run_example(display_callback=panel_display)
        print(f"{len(example_result['layouts'])} layouts · {example_result['snapshots']} snapshots")
    else:
        raise SystemExit(main())
