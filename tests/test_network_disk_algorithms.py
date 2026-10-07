"""Disk kernels against resident results and an independent Markov oracle."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_store import DiskNetwork
from test_network import stationary, values


def no_materialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A disk algorithm must not materialize the graph or its complete edge list")
    for name in ("materialize", "iter_nodes", "iter_edges"):
        monkeypatch.setattr(DiskNetwork, name, forbidden)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 65536])
def test_weighted_typed_identity_loops_duplicates_and_deterministic_components(tmp_path, monkeypatch, directed, batch):
    edges = [(1, "1", 2.), ("1", 1, 3.), ("1", "1", 7.), ("z", 9, 0.), (2**200, "x", 4.)]
    data = [dict(source=s, target=t, w=w) for s, t, w in edges]
    nodes = ["isolate", -2]
    disk = oe.network(data, directed=directed, weight="w", nodes=nodes, store=tmp_path / "typed")
    resident = oe.network(data, directed=directed, weight="w", nodes=nodes)
    no_materialization(monkeypatch)
    for method in ("degree", "components", "pagerank"):
        actual = getattr(disk, method)(batch_rows=batch)
        expected = getattr(resident, method)()
        pd.testing.assert_frame_equal(actual, expected, atol=2e-12, rtol=2e-12)
        assert actual.attrs["storage"] == "disk" and not actual.attrs["graph_materialized"]
        assert actual.attrs["exact"] and not actual.attrs["sampled"]
        assert actual.attrs["device"] == "cpu"
        assert actual.attrs["estimated_owned_peak_bytes"] <= actual.attrs["max_memory_bytes"]
        assert actual.attrs["actual_peak_batch_rows"] <= batch
        assert actual.attrs["edge_rows_read"] == actual.attrs["scan_passes"] * disk.edge_count
    assert values(disk.components(), "component") == {
        "isolate": 4, -2: 0, 1: 1, "1": 1, "z": 5, 9: 2, 2**200: 3, "x": 3}


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("damping", [0., .5, .85, .97])
def test_personalization_and_dangling_match_independent_gaussian_oracle(tmp_path, monkeypatch, directed, damping):
    edges = [("a", "b", 2.), ("a", "c", 1.), ("b", "c", 4.), ("c", "c", .5)]
    nodes = ["sink", "isolate"]
    p = {"a": 2., "c": 1., "sink": 4.}
    graph = oe.network([dict(source=s, target=t, w=w) for s, t, w in edges], weight="w",
                       directed=directed, nodes=nodes, store=tmp_path / "personalized")
    no_materialization(monkeypatch)
    result = graph.pagerank(damping=damping, personalization=p, max_iter=2000, tol=1e-12, batch_rows=1)
    assert values(result, "pagerank") == pytest.approx(stationary(edges, nodes, directed, damping, p), abs=2e-12)
    assert result.attrs["error_bound_estimate"] <= 1e-12
    assert result.pagerank.sum() == pytest.approx(1.)
    assert result.attrs["scan_passes"] == (0 if damping == 0 else 2 + result.attrs["iterations"])


@pytest.mark.parametrize("nodes", [[], ["b", "a"]])
def test_empty_and_all_isolates_preserve_output_dtypes(tmp_path, monkeypatch, nodes):
    graph = oe.network([], nodes=nodes, store=tmp_path / "empty")
    resident = oe.network([], nodes=nodes)
    no_materialization(monkeypatch)
    for method in ("degree", "pagerank", "components"):
        pd.testing.assert_frame_equal(getattr(graph, method)(), getattr(resident, method)())


def test_extreme_weights_use_stable_row_scaling_and_cpu_under_other_default_device(tmp_path, monkeypatch):
    graph = oe.network([dict(source="a", target="b", w=1e308), dict(source="a", target="c", w=1e308),
                        dict(source="b", target="c", w=1e-300)], weight="w", directed=True,
                        store=tmp_path / "extreme")
    no_materialization(monkeypatch)
    with torch.device("meta"):
        result = graph.pagerank(tol=1e-12, personalization={"a": 1e308, "c": 1e308})
        assert result.pagerank.sum() == pytest.approx(1.)
        assert graph.components().component.tolist() == [0, 0, 0]
    scaled = [("a", "b", 1.), ("a", "c", 1.), ("b", "c", 1.)]
    assert values(result, "pagerank") == pytest.approx(stationary(scaled, [], True, personalization={"a": 1., "c": 1.}), abs=2e-12)
    with pytest.raises(AnalysisError) as error:
        graph.degree()
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("method", ["degree", "pagerank", "components"])
def test_o_v_memory_and_minimum_work_admission_precede_tensor_allocation(tmp_path, monkeypatch, method):
    graph = oe.network([], nodes=range(5000), store=tmp_path / "too-many-nodes")
    graph = oe.open_network(graph.path, max_memory_mb=1)
    def forbidden(*args, **kwargs):
        pytest.fail("Admission must precede tensor construction and graph scans")
    monkeypatch.setattr(torch, "zeros", forbidden)
    monkeypatch.setattr(torch, "arange", forbidden)
    monkeypatch.setattr(graph, "_reader", forbidden)
    with pytest.raises(AnalysisError) as error:
        getattr(graph, method)()
    assert error.value.code == "network_memory_budget"


@pytest.mark.parametrize("method,options", [
    ("degree", dict(max_work=1)), ("components", dict(max_scan_bytes=1)),
    ("pagerank", dict(max_scan_bytes=71)), ("pagerank", dict(max_work=75)),
])
def test_minimum_scan_and_work_budget_fail_before_tensor_setup(tmp_path, monkeypatch, method, options):
    graph = oe.network([dict(source="a", target="b")], directed=True, store=tmp_path / "budget")
    def forbidden(*args, **kwargs):
        pytest.fail("Minimum budget failure must precede tensor setup")
    monkeypatch.setattr(torch, "zeros", forbidden)
    monkeypatch.setattr(torch, "arange", forbidden)
    with pytest.raises(AnalysisError) as error:
        getattr(graph, method)(**options)
    assert error.value.code == "network_work_budget"


def test_iteration_budgets_nonconvergence_and_unsupported_methods_are_explicit(tmp_path, monkeypatch):
    graph = oe.network([dict(source="a", target="b"), dict(source="b", target="c")],
                       directed=True, store=tmp_path / "iterations")
    no_materialization(monkeypatch)
    for options, code in [(dict(max_scan_bytes=144), "network_work_budget"),
                          (dict(max_work=180), "network_work_budget"),
                          (dict(max_iter=1), "network_nonconvergence"),
                          (dict(device="cuda"), "network_device"),
                          (dict(device="mps"), "network_device"),
                          (dict(personalization={"unknown": 1}), "network_unknown_node"),
                          (dict(personalization={"a": 0}), "network_invalid_option")]:
        with pytest.raises(AnalysisError) as error:
            graph.pagerank(**options)
        assert error.value.code == code
    for method, options in [("components", {"connectivity": "strong"}), ("communities", {}),
                            ("to_plot_data", {}), ("betweenness", {})]:
        with pytest.raises(AnalysisError) as error:
            getattr(graph, method)(**options)
        assert error.value.code == "network_capacity"


def test_dense_edges_exceed_resident_budget_but_disk_algorithms_run(tmp_path, monkeypatch):
    n = 200
    graph = oe.network((dict(source=i, target=(i + offset) % n, w=2)
                        for i in range(n) for offset in range(50)), weight="w", directed=True,
                       store=tmp_path / "dense", max_memory_mb=2, batch_rows=32)
    assert graph.resident_import_bound > 2 * 1024**2
    with pytest.raises(AnalysisError, match="max_memory_mb"):
        graph.materialize()
    no_materialization(monkeypatch)
    for result in (graph.degree(batch_rows=32), graph.pagerank(batch_rows=32), graph.components(batch_rows=32)):
        assert result.attrs["estimated_owned_peak_bytes"] <= 2 * 1024**2
    assert graph.degree().out_degree.tolist() == [50] * n
    assert graph.degree().out_strength.tolist() == [100.] * n
    assert graph.pagerank().pagerank.tolist() == pytest.approx([1 / n] * n, abs=1e-13)
    assert graph.components().component.tolist() == [0] * n


def test_store_changes_during_scan_never_return_a_result(tmp_path, monkeypatch):
    import openecon._network_disk_algorithms as algorithms
    graph = oe.network([dict(source="a", target="b")], store=tmp_path / "changed")
    original = algorithms._Scan.edges
    def changed(scan):
        for block in original(scan):
            yield block
            with graph.path.open("ab") as file:
                file.write(b"changed")
    monkeypatch.setattr(algorithms._Scan, "edges", changed)
    with pytest.raises(AnalysisError, match="changed"):
        graph.degree()


def test_source_panel_disk_analysis_example_has_ordered_latex_and_saved_readback(tmp_path):
    from openecon.console import ConsoleSession
    from openecon.workspace import Workspace
    code = (Path(__file__).resolve().parents[1] / "docs/examples/network_disk_analysis.py").read_text()
    workspace = Workspace(tmp_path / "owned-analysis-panel")
    session = ConsoleSession(workspace)
    try:
        run = session.execute(code, timeout_seconds=90)
        assert run["status"] == "ok", run.get("error")
        assert [output["type"] for output in run["outputs"]] == ["table"] * 4
        assert all("\\begin{tabular}" in item["latex"] for item in run["outputs"])
        probe = session.execute("import json\nprint(json.dumps(disk_analysis_proof, allow_nan=False))")
        assert probe["status"] == "ok"
        proof = json.loads(probe["stdout"])
        assert proof["nodes"] == 200 and proof["edges"] == 10_000
        assert proof["resident_guarded"] and proof["reopened"]
        assert proof["components"] == 1 and proof["pagerank_sum"] == pytest.approx(1.)
        saved = next(item for item in Workspace(workspace.path).console_history() if item["code"] == code)
        assert saved["outputs"] == run["outputs"]
    finally:
        session.close()
