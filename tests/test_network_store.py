"""Independent storage/identity, recovery and bounded-memory contracts."""
import math
import subprocess
import sqlite3
import sys

import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.data import DataError
from openecon._network_store import DiskNetwork


def records(graph):
    return {(row["source"], row["target"]): row["weight"]
            for block in graph.iter_edges(batch_rows=2) for row in block}


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 100])
def test_typed_identity_isolates_duplicates_loops_attributes_and_reopen(tmp_path, directed, rows):
    data = [dict(source=1, target="1", w=2), dict(source="1", target=1, w=3),
            dict(source="1", target="1", w=7), dict(source="zero", target=2**200, w=0),
            dict(source=None, target=1, w=3)]
    destination = tmp_path / "graph.sqlite"
    graph = oe.network(data, weight="w", directed=directed, nodes=["isolated"],
        missing="drop", store=destination, batch_rows=rows, max_memory_mb=8,
        node_attributes={1: {"label": "integer", "active": True}},
        edge_attributes={(1, "1"): {"relation": "trade"}}, graph_attributes={"period": 2026})
    assert isinstance(graph, DiskNetwork)
    assert graph.node_count == 5
    assert graph.metadata["missing_rows_dropped"] == graph.metadata["zero_weight_rows_dropped"] == 1
    assert graph.metadata["actual_peak_batch_rows"] <= rows
    expected = {(1, "1"): 2., ("1", 1): 3., ("1", "1"): 7.} if directed else {(1, "1"): 5., ("1", "1"): 7.}
    assert records(graph) == expected
    outgoing = {row["node"]: row["weight"] for block in graph.neighbors(1) for row in block}
    incoming = {row["node"]: row["weight"] for block in graph.neighbors(1, incoming=True) for row in block}
    assert outgoing == {"1": 2. if directed else 5.}
    assert incoming == {"1": 3. if directed else 5.}
    assert list(graph.neighbors("isolated")) == []
    reopened = oe.open_network(destination, max_memory_mb=8)
    assert records(reopened) == expected
    assert {row["node"]: row["attributes"] for block in reopened.iter_nodes() for row in block}[1] == {"label": "integer", "active": True}
    materialized = reopened.materialize()
    reference = oe.network(data, weight="w", nodes=["isolated"], directed=directed, missing="drop")
    pd.testing.assert_frame_equal(materialized.degree(), reference.degree())
    assert materialized._graph_attributes == {"period": 2026}
    assert materialized._edge_attributes[(1, "1")] == {"relation": "trade"}
    assert graph.metadata["input_sha256"] == reference.metadata["input_sha256"]
    assert graph.metadata["estimated_import_peak_bytes"] <= 8 * 1024**2


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_dataset_uses_disk_automatically_and_source_changes_are_detected(tmp_path, extension):
    path = tmp_path / ("edges." + extension)
    frame = pd.DataFrame({"source": [1, 2, 1], "target": [2, 3, 2], "w": [1., 2., 3.]})
    if extension == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, row_group_size=1)
    small = oe.network(oe.scan(path), weight="w", max_memory_mb=8, batch_rows=2)
    assert isinstance(small, oe.Network)
    graph = oe.network(oe.scan(path), weight="w", max_memory_mb=8, batch_rows=2, store=tmp_path / "graph.sqlite")
    assert isinstance(graph, DiskNetwork) and records(graph) == {(1, 2): 4., (2, 3): 2.}
    assert graph.pagerank().pagerank.sum() == pytest.approx(1.)
    store = graph.path
    with path.open("ab") as file:
        file.write(b"changed")
    with pytest.raises((AnalysisError, DataError), match="changed"):
        oe.open_network(store)
    # The saved graph is an independent snapshot only when explicitly requested.
    assert records(oe.open_network(store, verify_source=False)) == {(1, 2): 4., (2, 3): 2.}
    graph.close()
    assert store.exists()  # Closing an explicitly saved snapshot preserves it.
    with pytest.raises(AnalysisError, match="closed"):
        list(graph.iter_edges())


def test_large_dataset_automatically_retains_disk_storage_and_cleans_owned_temp():
    def batches():
        for start in range(0, 30_000, 1000):
            yield pd.DataFrame({"source": range(start, start + 1000), "target": range(start + 1, start + 1001)})
    source = oe.Dataset.from_batches(batches, columns=["source", "target"], row_count=30_000)
    graph = oe.network(source, max_memory_mb=8, batch_rows=64)
    assert isinstance(graph, DiskNetwork) and graph.edge_count == 30_000
    path = graph.path
    assert len(next(graph.neighbors(25_000))) == 2
    graph.close()
    assert not path.exists()


def test_stream_exceeds_resident_budget_but_disk_build_and_reads_stay_bounded(tmp_path):
    def edges():
        for node in range(30_000):
            yield dict(source=node, target=node + 1, w=1)

    with pytest.raises(AnalysisError) as error:
        oe.network(edges(), weight="w", max_memory_mb=8, batch_rows=64)
    assert error.value.code == "network_memory_budget"
    graph = oe.network(edges(), weight="w", store=tmp_path / "large.sqlite", max_memory_mb=8, batch_rows=64)
    assert graph.node_count == 30_001 and graph.edge_count == 30_000
    assert sum(len(block) for block in graph.iter_edges(batch_rows=100_000)) == 30_000
    assert max(len(block) for block in graph.iter_nodes(batch_rows=100_000)) <= 128
    assert graph.summary().attrs["network"]["edge_count"] == 30_000
    with pytest.raises(AnalysisError) as error:
        graph.pagerank()
    assert error.value.code == "network_memory_budget"


def test_cancellation_and_invalid_input_do_not_publish_or_replace_files(tmp_path):
    target = tmp_path / "cancel.sqlite"
    def data():
        for node in range(100):
            yield dict(source=node, target=node + 1)
    with pytest.raises(AnalysisError, match="cancelled"):
        oe.network(data(), store=target, cancelled=lambda: True)
    assert not target.exists() and not list(tmp_path.glob(".*building-*"))
    with pytest.raises(AnalysisError):
        oe.network([dict(source=True, target=1)], store=target)
    assert not target.exists() and not list(tmp_path.glob(".*building-*"))
    target.write_bytes(b"user-owned-data")
    with pytest.raises(AnalysisError, match="already exists"):
        oe.network(data(), store=target)
    assert target.read_bytes() == b"user-owned-data"


def test_source_changed_during_ingestion_is_never_published(tmp_path, monkeypatch):
    path = tmp_path / "edges.csv"
    pd.DataFrame({"source": [1, 2], "target": [2, 3]}).to_csv(path, index=False)
    source = oe.scan(path)
    original = source.iter_batches
    def changed(*args, **kwargs):
        iterator = original(*args, **kwargs)
        try:
            for batch in iterator:
                yield batch
                with path.open("ab") as file:
                    file.write(b"changed")
        finally:
            iterator.close()
    monkeypatch.setattr(source, "iter_batches", changed)
    with pytest.raises(DataError, match="changed"):
        oe.network(source, store=tmp_path / "graph.sqlite", batch_rows=1)
    assert not (tmp_path / "graph.sqlite").exists()
    assert not list(tmp_path.glob(".*building-*"))


def test_process_crash_leaves_unpublished_staging_file_that_cannot_open(tmp_path):
    target = tmp_path / "crashed.sqlite"
    code = """
import os
import openecon as oe
calls = 0
def cancelled():
    global calls
    calls += 1
    if calls == 3:
        os._exit(17)
    return False
oe.network((dict(source=i,target=i+1) for i in range(10000)),
    store=TARGET, batch_rows=32, cancelled=cancelled)
""".replace("TARGET", repr(str(target)))
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
    assert process.returncode == 17, process.stderr.decode()
    assert not target.exists()
    staging = list(tmp_path.glob(".*building-*"))
    assert staging
    for path in staging:
        with pytest.raises(AnalysisError, match="snapshot"):
            oe.open_network(path)


def test_float_overflow_disk_budget_and_read_identity_guards(tmp_path):
    with pytest.raises(AnalysisError, match="overflow"):
        oe.network([dict(source=1, target=2, w=1e308)] * 2, weight="w", store=tmp_path / "overflow")
    with pytest.raises(AnalysisError) as error:
        oe.network((dict(source=i,target=i+1) for i in range(20_000)), store=tmp_path / "full", max_disk_mb=1)
    assert error.value.code == "network_disk_budget"
    assert not (tmp_path / "full").exists()
    graph = oe.network([dict(source=1,target=2)], store=tmp_path / "edited")
    with graph.path.open("ab") as file:
        file.write(b"changed")
    with pytest.raises((DataError, AnalysisError), match="changed"):
        list(graph.iter_edges())


def test_fractional_coalescing_matches_independent_sum_and_large_attributes_are_bounded(tmp_path):
    graph = oe.network([dict(source="a",target="b",w=.1)] * 30, weight="w", store=tmp_path / "fractional",
        node_attributes={"a": {str(i): "é" * 8000 for i in range(8)}}, max_memory_mb=8)
    assert records(graph)[("a", "b")] == pytest.approx(math.fsum([.1] * 30), abs=2e-14)
    assert max(len(block) for block in graph.iter_nodes(batch_rows=100_000)) <= 8
    assert len(next(graph.iter_nodes())[0]["attributes"]) == 8
    assert graph.metadata["estimated_import_peak_bytes"] <= 8 * 1024**2


def test_concurrent_publication_does_not_replace_the_other_file(tmp_path, monkeypatch):
    import openecon._network_store as store
    original = store.os.link
    target = tmp_path / "concurrent"
    def concurrent(source, destination):
        target.write_bytes(b"another-job")
        original(source, destination)
    monkeypatch.setattr(store.os, "link", concurrent)
    with pytest.raises(AnalysisError, match="concurrently"):
        oe.network([dict(source=1, target=2)], store=target)
    assert target.read_bytes() == b"another-job"
    assert not list(tmp_path.glob(".*building-*"))


def test_reopen_rejects_a_dangling_adjacency_even_when_counts_match(tmp_path):
    graph = oe.network([dict(source=1, target=2)], store=tmp_path / "damaged")
    with sqlite3.connect(graph.path) as connection:
        connection.execute("UPDATE edges SET dst=1000")
    with pytest.raises(AnalysisError, match="missing node"):
        oe.open_network(graph.path)


def test_utf8_graph_manifest_can_reopen_at_maximum_scalar_attribute_sizes(tmp_path):
    attributes = {str(i): "é" * 8192 for i in range(128)}
    graph = oe.network([], store=tmp_path / "utf8", graph_attributes=attributes, max_memory_mb=64)
    assert oe.open_network(graph.path, max_memory_mb=64).metadata["graph_attributes"] == attributes


def test_complete_source_panel_example_persists_ordered_outputs(tmp_path):
    import json
    from pathlib import Path
    from openecon.console import ConsoleSession
    from openecon.workspace import Workspace

    code = (Path(__file__).resolve().parents[1] / "docs/examples/network_disk_storage.py").read_text()
    workspace = Workspace(tmp_path / "owned-disk-panel")
    session = ConsoleSession(workspace)
    try:
        run = session.execute(code, timeout_seconds=90)
        assert run["status"] == "ok", run.get("error")
        assert [output["type"] for output in run["outputs"]] == ["table", "table"]
        probe = session.execute("import json\nprint(json.dumps(disk_storage_proof, allow_nan=False))")
        assert probe["status"] == "ok"
        proof = json.loads(probe["stdout"])
        assert proof["nodes"] == 30_002 and proof["edges"] == 30_000 and proof["reopened"]
        assert proof["bounded_rows"] <= 64
        saved = next(item for item in Workspace(workspace.path).console_history() if item["code"] == code)
        assert saved["outputs"] == run["outputs"]
    finally:
        session.close()
