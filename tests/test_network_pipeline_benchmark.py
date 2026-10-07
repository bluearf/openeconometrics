"""Measurement protocol checks: coverage, physical artifact integrity and scope."""
import importlib.util
import json
from pathlib import Path

import pytest
import openecon as oe

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_benchmark", ROOT / "benchmarks/network_pipeline.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_disconnected_fixture_has_eight_blocks_and_explicit_isolates():
    graph = oe.network(benchmark.fixture("disconnected", 200), nodes=range(200), weight="weight")
    degree = graph.degree()
    assert graph.node_count == 200 and graph.edge_count == 640
    assert (degree["degree"] == 0).sum() == 40
    assert graph.components()["component"].nunique() == 48


@pytest.mark.parametrize("case,edges,directed", [("dense", 4950, False), ("hub", 396, False), ("directed", 400, True)])
def test_named_fixtures_have_independent_topology_counts(case, edges, directed):
    graph = oe.network(benchmark.fixture(case, 100), nodes=range(100), weight="weight", directed=directed)
    assert graph.edge_count == edges and graph.node_count == 100
    assert graph.components()["component"].nunique() == 1
    if case == "hub":
        assert graph.degree().set_index("node").loc[0, "degree"] == 99


def test_selected_display_keeps_complete_analysis_and_verified_physical_artifact(tmp_path):
    report = benchmark.measure_case("sparse", 2500, "selected", tmp_path)
    assert report["result_rows"] == [2500]
    assert report["algorithms"]["rank_sums"] == pytest.approx([1.0])
    assert report["display"][0]["node_count"] == 2500
    assert report["display"][0]["shown_node_count"] == 2000
    assert report["display"][0]["sampled"] is True
    payload = json.loads((tmp_path / report["artifact_path"]).read_text())
    assert len(payload["config"]["network"]["nodes"]) == 2000
    assert report["input"][0]["rows"] == 10000
    assert report["process_peak_rss_bytes"] >= report["baseline_process_peak_rss_bytes"]
    assert "not isolated phase peaks" in report["memory_scope"]
    assert "not disk-only" in report["timing_scope"]


def test_temporal_measurement_preserves_all_four_frames(tmp_path):
    report = benchmark.measure_case("temporal", 100, "full", tmp_path)
    assert report["frame_count"] == 4 and report["transition_rows"] == 3
    assert report["result_rows"] == [100] * 4
    assert len(report["display"]) == 5  # Base plus the four saved frames.
    assert all(item["shown_edge_count"] == item["edge_count"] == 400 for item in report["display"])
    assert not any(item["sampled"] for item in report["display"])
    assert report["transport_encoding"] == "timeline-pool-v1"
    assert report["plot_json"]["bytes"] < report["plot_public_json"]["bytes"]
    payload = json.loads((tmp_path / report["artifact_path"]).read_text())
    from openecon_charts.timeline import unpack
    assert len(unpack(payload)["config"]["network"]["frames"]) == 4


@pytest.mark.parametrize("case,nodes", [("sparse", "0"), ("dense", "1001")])
def test_invalid_dimensions_refused_before_creating_output(tmp_path, monkeypatch, case, nodes):
    import sys
    output = tmp_path / "receipt"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--child", case, "--nodes", nodes, "--output", str(output)])
    with pytest.raises(SystemExit) as failure:
        benchmark.main()
    assert failure.value.code == 2
    assert not output.exists()


def test_existing_measurement_directory_is_never_overwritten(tmp_path, monkeypatch):
    import sys
    receipt = tmp_path / "report.json"
    receipt.write_text("previous measurement")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        benchmark.main()
    assert receipt.read_text() == "previous measurement"
