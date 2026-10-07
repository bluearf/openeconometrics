"""Independent score aggregation and owned spill resource tests."""
from pathlib import Path
import sqlite3

import pytest
import torch

from openecon.engines.contracts import KernelError
from openecon.engines import streaming_groups
from openecon.engines.streaming_groups import ClusterAccumulator


@pytest.fixture(scope="module", autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("rows", [1, 7, 113, 9000])
def test_noncontiguous_keys_match_dense_sum_with_eviction(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(streaming_groups, "_CACHE_GROUPS", 3)
    generator = torch.Generator().manual_seed(816)
    scores = torch.randn((1057, 4), dtype=torch.float64, generator=generator)
    groups = torch.arange(len(scores)).remainder(41)
    labels = [b"key\0" + str(int(group)).encode() for group in groups]
    expected = torch.zeros((41, 4), dtype=torch.float64)
    expected.index_add_(0, groups, scores)
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("preserve")
    with ClusterAccumulator(4, tmp_path) as accumulator:
        scratch = Path(accumulator._scratch.name)
        assert scratch.parent == tmp_path and scratch.stat().st_mode & 0o777 == 0o700
        for start in range(0, len(scores), rows):
            accumulator.add(labels[start:start + rows], scores[start:start + rows])
        meat, count = accumulator.finish()
        torch.testing.assert_close(meat, expected.T @ expected, rtol=3e-13, atol=3e-12)
        assert type(count) is int and count == 41
        assert accumulator.diagnostics["cluster_peak_cached_groups"] <= 3
        assert accumulator.diagnostics["cluster_spill_writes"] >= 41
        if rows < len(scores):
            assert accumulator.diagnostics["cluster_spill_writes"] > 41
        assert accumulator.diagnostics["cluster_scratch_bytes"] > 0
    assert not scratch.exists()
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve"
    accumulator.close()


def test_many_groups_keep_fixed_cache_and_read_meat_in_small_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming_groups, "_CACHE_GROUPS", 7)
    monkeypatch.setattr(streaming_groups, "_READ_BYTES", 192)
    generator = torch.Generator().manual_seed(819)
    total = torch.zeros((3, 3), dtype=torch.float64)
    with ClusterAccumulator(3, tmp_path) as accumulator:
        for start in range(0, 20_003, 211):
            size = min(211, 20_003 - start)
            scores = torch.randn((size, 3), dtype=torch.float64, generator=generator)
            keys = [f"unique-{row}".encode() for row in range(start, start + size)]
            total += scores.T @ scores
            accumulator.add(keys, scores)
            assert accumulator._totals.shape == (7, 3)
            assert accumulator._corrections.shape == (7, 3)
            assert len(accumulator._cache) <= 7
            assert accumulator._totals.untyped_storage().nbytes() == 7 * 3 * 8
        meat, count = accumulator.finish()
        torch.testing.assert_close(meat, total, rtol=4e-13, atol=2e-10)
        assert count == 20_003
        assert accumulator.diagnostics["cluster_peak_cache_accounted_bytes"] <= streaming_groups._CACHE_BYTES
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("batch_rows", [1, 2, 3, 100])
def test_compensation_preserves_cancellation_with_spill(tmp_path, monkeypatch, batch_rows):
    monkeypatch.setattr(streaming_groups, "_CACHE_GROUPS", 1)
    scores = torch.tensor([[1e16, 1e16], [3, 4], [1, 2], [0, 0], [-1e16, -1e16]], dtype=torch.float64)
    keys = [b"a", b"b", b"a", b"b", b"a"]
    with ClusterAccumulator(2, tmp_path) as accumulator:
        for start in range(0, len(keys), batch_rows):
            accumulator.add(keys[start:start + batch_rows], scores[start:start + batch_rows])
        meat, count = accumulator.finish()
        torch.testing.assert_close(meat, torch.tensor([[10, 14], [14, 20]], dtype=torch.float64), rtol=0, atol=0)
        assert count == 2


def test_long_keys_bypass_cache_without_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming_groups, "_CACHE_BYTES", 4096)
    keys = [b"x" * 10000, b"x" * 10000 + b"\0", b"x" * 10000, b""]
    scores = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    with ClusterAccumulator(1, tmp_path) as accumulator:
        accumulator.add(keys, scores)
        meat, count = accumulator.finish()
        assert count == 3
        torch.testing.assert_close(meat, torch.tensor([[36.0]], dtype=torch.float64))
        assert accumulator.diagnostics["cluster_peak_cache_accounted_bytes"] <= 4096


@pytest.mark.parametrize("count", [0, 1])
def test_requires_two_groups_and_cleans_on_error(tmp_path, count):
    with pytest.raises(KernelError) as error:
        with ClusterAccumulator(2, tmp_path) as accumulator:
            accumulator.add([b"one"] * count, torch.ones((count, 2), dtype=torch.float64))
            accumulator.finish()
    assert error.value.code == "insufficient_clusters"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind,code", [
    ("keys", "invalid_clusters"), ("shape", "invalid_clusters"), ("dtype", "invalid_precision"),
    ("nonfinite", "non_finite_values"), ("device", "unsupported_device"),
])
def test_invalid_inputs_reject_and_remove_owned_database(tmp_path, kind, code):
    keys = [b"a", b"b"]
    scores = torch.ones((2, 2), dtype=torch.float64)
    if kind == "keys":
        keys[1] = "not encoded"
    elif kind == "shape":
        scores = scores[:, :1]
    elif kind == "dtype":
        scores = scores.float()
    elif kind == "nonfinite":
        scores[0, 0] = torch.inf
    else:
        scores = torch.empty((2, 2), dtype=torch.float64, device="meta")
    with pytest.raises(KernelError) as error:
        with ClusterAccumulator(2, tmp_path) as accumulator:
            accumulator.add(keys, scores)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("width", [0, -1, True, 1.5, 1025])
def test_invalid_width_does_not_create_scratch(tmp_path, width):
    with pytest.raises(KernelError) as error:
        ClusterAccumulator(width, tmp_path)
    assert error.value.code == "invalid_clusters"
    assert not list(tmp_path.iterdir())


def test_constructor_disk_failure_cleans_and_has_actionable_error(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")
    monkeypatch.setattr(streaming_groups.sqlite3, "connect", unavailable)
    with pytest.raises(KernelError) as error:
        ClusterAccumulator(2, tmp_path)
    assert error.value.code == "cluster_spill_failed"
    assert "disk space" in str(error.value)
    assert not list(tmp_path.iterdir())


def test_spill_write_failure_cleans_without_memory_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming_groups, "_CACHE_GROUPS", 1)
    class BrokenConnection:
        def __init__(self, connection):
            self.connection = connection
        def execute(self, query, *args):
            if query.startswith("INSERT"):
                raise sqlite3.OperationalError("disk full")
            return self.connection.execute(query, *args)
        def close(self):
            self.connection.close()
        def executemany(self, query, *args):
            if query.startswith("INSERT"):
                raise sqlite3.OperationalError("disk full")
            return self.connection.executemany(query, *args)
    with pytest.raises(KernelError) as error:
        with ClusterAccumulator(2, tmp_path) as accumulator:
            accumulator._connection = BrokenConnection(accumulator._connection)
            accumulator.add([b"a", b"b"], torch.ones((2, 2), dtype=torch.float64))
    assert error.value.code == "cluster_spill_failed"
    assert not list(tmp_path.iterdir())


def test_nonfinite_accumulation_rejects_and_cleans(tmp_path):
    with pytest.raises(KernelError) as error:
        with ClusterAccumulator(1, tmp_path) as accumulator:
            accumulator.add([b"a", b"b"], torch.tensor([[1e200], [1e200]], dtype=torch.float64))
            accumulator.finish()
    assert error.value.code == "numerical_failure"
    assert not list(tmp_path.iterdir())


def test_finished_and_closed_accumulator_cannot_be_reused(tmp_path):
    accumulator = ClusterAccumulator(1, tmp_path)
    accumulator.add([b"a", b"b"], torch.ones((2, 1), dtype=torch.float64, requires_grad=True))
    accumulator.finish()
    assert accumulator._totals.grad_fn is None
    with pytest.raises(KernelError):
        accumulator.add([b"a"], torch.ones((1, 1), dtype=torch.float64))
    with pytest.raises(KernelError):
        accumulator.finish()
    accumulator.close()
    accumulator.close()
    assert not list(tmp_path.iterdir())
