"""Reproducible physical inputs and preallocation guards for measured workloads."""
import importlib.util
from pathlib import Path
import sys

import pyarrow.parquet as pq
import pytest


_PATH = Path(__file__).resolve().parents[1] / "benchmarks/network_advanced.py"
_SPEC = importlib.util.spec_from_file_location("network_advanced_benchmark", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize("planted", [False, True])
def test_fixture_is_physical_repeatable_and_seed_sensitive(tmp_path, planted):
    first, second, other = (tmp_path / name for name in ("first.parquet", "second.parquet", "other.parquet"))
    expected = benchmark.write_fixture(first, 1000, 100, 123, planted=planted)
    repeated = benchmark.write_fixture(second, 1000, 100, 123, planted=planted)
    changed = benchmark.write_fixture(other, 1000, 100, 124, planted=planted)
    assert expected == repeated
    assert first.read_bytes() == second.read_bytes()
    assert expected["sha256"] != changed["sha256"]
    assert expected["physical_rows"] == 1000 and expected["largest_row_group"] == 1000
    data = pq.read_table(first).to_pandas()
    assert len(data) == 1000
    assert data.source.between(0, 99).all() and data.target.between(0, 99).all()
    assert data.weight.between(.1, 3.1, inclusive="left").all()
    if planted:
        # The mandatory rings actually include every declared node, before
        # random edges; the planted comparison does not rely on unseen labels.
        ring = data.iloc[:100]
        assert ring.source.tolist() == list(range(100))
        assert ring.target.tolist() == [(node // 50) * 50 + (node + 1) % 50 for node in range(100)]


@pytest.mark.parametrize("option,value", [
    ("--rows", "0"), ("--nodes", "1"), ("--samples", "100001"),
    ("--community-nodes", "51"), ("--community-rows", "1"),
    ("--max-work", "0"), ("--max-memory-mb", "nan"),
])
def test_invalid_measurement_dimensions_rejected_before_creation(tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["benchmark", option, value, "--output", str(tmp_path / "receipt.json")])
    monkeypatch.setattr(benchmark, "write_fixture", lambda *a, **k: pytest.fail("Invalid input allocated a fixture."))
    with pytest.raises(SystemExit) as failure:
        benchmark.main()
    assert failure.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_existing_receipt_never_overwritten(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("previous evidence\n")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(receipt)])
    monkeypatch.setattr(benchmark, "write_fixture", lambda *a, **k: pytest.fail("Existing evidence permitted allocation."))
    with pytest.raises(SystemExit) as failure:
        benchmark.main()
    assert failure.value.code == 2
    assert receipt.read_text() == "previous evidence\n"
