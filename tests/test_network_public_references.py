"""Offline public-data references and weighted/directed identity perturbations."""
import importlib.util
from pathlib import Path
import random

import pytest
import torch

import openecon as oe


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/network_reference"
spec = importlib.util.spec_from_file_location(
    "network_public_reference", ROOT / "docs/examples/network_reference_validation.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize("name", ["karate", "grqc"])
def test_full_public_network_matches_raw_file_and_published_references(name):
    report = reference.validate(FIXTURES, names=(name,))
    assert report["status"] == "passed"
    assert report["full_graph_analysis"] and not report["display_sampling"]
    assert not report["out_of_core_graph"]
    assert report["environment"]["device"] == "cpu"
    assert report["datasets"][name]["timing_seconds"]["complete_case"] > 0


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_public_topology_weight_direction_loops_isolates_and_typed_labels(directed, typed):
    nodes, pairs, _, _ = reference.load_fixture(FIXTURES, "karate")
    labels = {node: str(node) if typed and node % 2 else node for node in nodes}
    # Integer 1 and string "1" coexist; the unused counterpart is an isolate.
    isolate = 1 if typed else "1"
    nodes = [*(labels[node] for node in nodes), isolate, "zero endpoint"]
    records = [(labels[u], labels[v], float(1 + (u * 3 + v) % 5)) for u, v in pairs]
    records += [records[0], (labels[2], labels[2], .5),
                (isolate, "zero endpoint", 0.)]
    random.Random(54).shuffle(records)
    graph = oe.network([{"source": u, "target": v, "w": w} for u, v, w in records],
                       weight="w", directed=directed, nodes=list(reversed(nodes)), batch_rows=7)
    expected = reference.stationary(nodes, records, directed=directed)
    result = graph.pagerank(tol=1e-12, max_iter=1000)
    assert set(result.node) == set(nodes)
    assert dict(zip(result.node, result.pagerank)) == pytest.approx(expected, abs=2e-12)
    assert graph.node_count == 36
    assert graph.metadata["duplicate_edge_rows_aggregated"] == 1
    assert graph.metadata["zero_weight_rows_dropped"] == 1
    assert result.attrs["converged"]


def test_reference_fixture_checksum_is_checked_before_analysis(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_bytes((FIXTURES / "manifest.json").read_bytes())
    (tmp_path / "karate.csv").write_text("source,target\n1,2\n")
    monkeypatch.setattr(oe, "network", lambda *a, **kw: pytest.fail("Changed source reached analysis"))
    with pytest.raises(ValueError, match="pinned karate fixture changed"):
        reference.validate(tmp_path, names=("karate",))


def test_reference_runner_restores_thread_setting_after_failure(tmp_path):
    previous = torch.get_num_threads()
    with pytest.raises(FileNotFoundError):
        reference.validate(tmp_path)
    assert torch.get_num_threads() == previous
