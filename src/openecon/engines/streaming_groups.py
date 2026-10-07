"""Bounded-memory, disk-backed one-way cluster score aggregation.

Only a fixed score cache and one small observation block live in memory. SQLite
holds every group outside that cache; neither the group count nor the observation
count determines a resident tensor's shape. The temporary database is owned by
this accumulator and is removed on success and ordinary exceptions. Supervised
workers also remove their own scratch directory after abrupt termination.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
import sqlite3
import tempfile

import torch
from torch import Tensor

from .contracts import KernelError

_CACHE_BYTES = 4 * 1024 * 1024
_CACHE_GROUPS = 4096
_SQLITE_CACHE_KIB = 2048
_REDUCTION_ROWS = 4096
_PENDING_ROWS = 256
_READ_BYTES = 1024 * 1024
_SLOT_OVERHEAD = 512


def _finite(value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise KernelError("numerical_failure", "Cluster score aggregation exceeded finite float64 precision.")


def _merge(total: Tensor, correction: Tensor, increment: Tensor, extra: Tensor) -> tuple[Tensor, Tensor]:
    """Neumaier addition preserves cancellation across reduction levels/blocks."""
    updated = total + increment
    error = torch.where(total.abs() >= increment.abs(),
                        (total - updated) + increment, (increment - updated) + total)
    adjusted = correction + extra + error
    _finite(updated)
    _finite(adjusted)
    return updated, adjusted


def _reduce(keys: Sequence[bytes], scores: Tensor) -> tuple[list[bytes], Tensor, Tensor]:
    """Pairwise compensated segments; the temporary key dictionary is block-sized."""
    labels: dict[bytes, int] = {}
    codes = []
    for key in keys:
        code = labels.get(key)
        if code is None:
            code = len(labels)
            labels[key] = code
        codes.append(code)
    unique = list(labels)
    if len(unique) == len(keys):
        return unique, scores, torch.zeros_like(scores)
    ids = torch.tensor(codes, dtype=torch.int64)
    order = torch.argsort(ids, stable=True)
    ids, totals = ids[order], scores[order]
    corrections = torch.zeros_like(totals)
    while len(ids) > len(unique):
        starts = torch.cat((torch.zeros(1, dtype=torch.int64),
                            torch.nonzero(ids[1:] != ids[:-1], as_tuple=False)[:, 0] + 1))
        lengths = torch.diff(torch.cat((starts, torch.tensor([len(ids)], dtype=torch.int64))))
        offsets = torch.arange(len(ids)) - torch.repeat_interleave(starts, lengths)
        ends = torch.repeat_interleave(starts + lengths, lengths)
        first = torch.nonzero(offsets.remainder(2) == 0, as_tuple=False)[:, 0]
        paired = first + 1 < ends[first]
        second = torch.where(paired, first + 1, first)
        right = torch.where(paired[:, None], totals[second], torch.zeros_like(totals[first]))
        right_correction = torch.where(paired[:, None], corrections[second],
                                       torch.zeros_like(corrections[first]))
        totals, corrections = _merge(totals[first], corrections[first], right, right_correction)
        ids = ids[first]
    return unique, totals, corrections


class ClusterAccumulator:
    """Accumulate raw sum_g(score_g score_g') with bounded RAM and SQLite spill.

    Keys must be canonical collision-free bytes supplied by the analysis layer.
    finish returns uncorrected meat and a Python integer group count; the caller
    applies CR1. At least two observed groups are required, matching dense kernels.
    """

    def __init__(self, width: int, scratch_directory: Path | None = None):
        if type(width) is not int or not 1 <= width <= 1024:
            raise KernelError("invalid_clusters", "Cluster score width must be an integer from 1 to 1024.")
        self.width = width
        self.capacity = max(1, min(_CACHE_GROUPS, _CACHE_BYTES // (16 * width + _SLOT_OVERHEAD + 64)))
        self._allocated_bytes = self.capacity * (16 * width + _SLOT_OVERHEAD)
        self._key_budget = max(0, _CACHE_BYTES - self._allocated_bytes)
        self._totals = torch.zeros((self.capacity, width), dtype=torch.float64)
        self._corrections = torch.zeros_like(self._totals)
        self._cache: OrderedDict[bytes, int] = OrderedDict()
        self._free = list(range(self.capacity))
        self._key_bytes = self._peak_groups = self._blocks = 0
        self._peak_bytes = self._allocated_bytes
        self._spill_writes = self._scratch_bytes = self._groups = 0
        self._closed = self._finished = self._failed = False
        self._connection = None
        self._scratch = None
        try:
            self._scratch = tempfile.TemporaryDirectory(prefix="openecon-clusters-", dir=scratch_directory)
            self._path = Path(self._scratch.name) / "groups.sqlite3"
            self._connection = sqlite3.connect(str(self._path), isolation_level=None)
            self._connection.execute("PRAGMA journal_mode=OFF")
            self._connection.execute("PRAGMA synchronous=OFF")
            self._connection.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_KIB}")
            self._connection.execute("PRAGMA mmap_size=0")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute("CREATE TABLE scores (key BLOB PRIMARY KEY, total BLOB NOT NULL, correction BLOB NOT NULL) WITHOUT ROWID")
            self._path.chmod(0o600)
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise self._storage_error() from exc

    @staticmethod
    def _storage_error() -> KernelError:
        return KernelError("cluster_spill_failed", "Cluster covariance needs writable temporary storage and enough free disk space; free disk space or select another scratch_directory.")

    @property
    def diagnostics(self) -> dict:
        return {
            "cluster_aggregation": "sqlite_spill",
            "cluster_cache_capacity": self.capacity,
            "cluster_peak_cached_groups": self._peak_groups,
            "cluster_cache_limit_bytes": _CACHE_BYTES,
            "cluster_peak_cache_accounted_bytes": self._peak_bytes,
            "cluster_sqlite_cache_bytes": _SQLITE_CACHE_KIB * 1024,
            "cluster_score_blocks": self._blocks,
            "cluster_spilled_groups": self._groups,
            "cluster_spill_writes": self._spill_writes,
            "cluster_scratch_bytes": self._scratch_bytes,
            "cluster_accumulation": "pairwise compensated block scores; Neumaier group and meat sums",
            "cluster_memory_complexity": "O(block_rows * design_columns + design_columns ** 2 + fixed cache)",
            "cluster_disk_complexity": "O(groups * (design_columns + key_bytes))",
            "cluster_reduction_block_rows": _REDUCTION_ROWS,
            "cluster_spill_io": "bulk SELECT/executemany; at most500 keys and1 MiB planned key/record payload per statement (one longer key handled alone)",
        }

    def _check_open(self) -> None:
        if self._closed or self._finished or self._failed:
            raise KernelError("invalid_clusters", "The cluster accumulator is closed, finished, or failed.")

    def _decode(self, value: bytes) -> Tensor:
        if len(value) != self.width * 8:
            raise KernelError("cluster_spill_failed", "The owned cluster spill contains an invalid score record.")
        result = torch.frombuffer(bytearray(value), dtype=torch.float64)
        _finite(result)
        return result

    @staticmethod
    def _encode(value: Tensor) -> bytes:
        return array("d", value.tolist()).tobytes()

    def _persist(self, key: bytes, slot: int) -> None:
        self._connection.execute("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                                 (key, self._encode(self._totals[slot]), self._encode(self._corrections[slot])))
        self._spill_writes += 1

    def _add_reduced(self, keys: list[bytes], totals: Tensor, corrections: Tensor) -> None:
        """Bulk spill I/O; identical vectorized Neumaier merge, bounded keys/blobs.

        A reduction block has at most4096 unique groups. No full group map is
        retained across blocks; SQLite owns all previous sums and corrections.
        SQL statements additionally bound key/record payload, so long byte keys
        cannot multiply a500-key query into an unexpectedly huge allocation.
        """
        if not keys:
            return
        previous = torch.zeros_like(totals)
        previous_correction = torch.zeros_like(corrections)
        mapping = {key: row for row, key in enumerate(keys)}
        parts = []
        start = 0
        while start < len(keys):
            stop, payload = start, 0
            while stop < len(keys) and stop-start < 500:
                extra = len(keys[stop])+16*self.width+64
                if stop > start and payload+extra > _READ_BYTES:
                    break
                payload += extra
                stop += 1
            parts.append((start, stop))
            part = keys[start:stop]
            statement = "SELECT key,total,correction FROM scores WHERE key IN ("+",".join("?" for _ in part)+")"
            records = self._connection.execute(statement, part).fetchall()
            if records:
                if any(len(record[1]) != self.width*8 or len(record[2]) != self.width*8 for record in records):
                    raise KernelError("cluster_spill_failed", "The owned cluster spill contains an invalid score record.")
                values = torch.frombuffer(bytearray(b"".join(record[1] for record in records)), dtype=torch.float64).reshape(-1, self.width)
                errors = torch.frombuffer(bytearray(b"".join(record[2] for record in records)), dtype=torch.float64).reshape(-1, self.width)
                indices = torch.tensor([mapping[record[0]] for record in records], dtype=torch.int64)
                previous.index_copy_(0, indices, values)
                previous_correction.index_copy_(0, indices, errors)
            start = stop
        updated, error = _merge(previous, previous_correction, totals, corrections)
        # Tensor->bytes is serialization only; no external numerical engine.
        values_buffer = memoryview(updated.contiguous().numpy().tobytes())
        errors_buffer = memoryview(error.contiguous().numpy().tobytes())
        stride = self.width*8
        for start, stop in parts:
            self._connection.executemany("INSERT OR REPLACE INTO scores VALUES(?,?,?)",
                ((keys[row], values_buffer[row*stride:(row+1)*stride],
                  errors_buffer[row*stride:(row+1)*stride]) for row in range(start, stop)))
        self._spill_writes += len(keys)

    @torch.inference_mode()
    def add(self, keys: Sequence[bytes], scores: Tensor) -> None:
        self._check_open()
        if (not isinstance(keys, Sequence) or isinstance(keys, (bytes, str))
                or not isinstance(scores, Tensor) or scores.ndim != 2
                or scores.shape != (len(keys), self.width) or any(not isinstance(key, bytes) for key in keys)):
            raise KernelError("invalid_clusters", "Cluster aggregation needs one byte key per score row and the configured score width.")
        if scores.device.type != "cpu":
            raise KernelError("unsupported_device", "Cluster score aggregation requires CPU tensors.")
        if scores.dtype != torch.float64:
            raise KernelError("invalid_precision", "Cluster score aggregation requires float64 tensors.")
        if not bool(torch.isfinite(scores).all()):
            raise KernelError("non_finite_values", "Cluster scores must be finite.")
        try:
            self._connection.execute("BEGIN")
            for start in range(0, len(keys), _REDUCTION_ROWS):
                stop = min(len(keys), start + _REDUCTION_ROWS)
                unique, totals, corrections = _reduce(keys[start:stop], scores[start:stop])
                self._add_reduced(unique, totals, corrections)
                self._blocks += 1
            self._connection.execute("COMMIT")
        except (OSError, sqlite3.Error) as exc:
            self._failed = True
            raise self._storage_error() from exc
        except Exception:
            self._failed = True
            raise

    @torch.inference_mode()
    def finish(self) -> tuple[Tensor, int]:
        self._check_open()
        try:
            self._connection.execute("BEGIN")
            for key, slot in self._cache.items():
                self._persist(key, slot)
            self._connection.execute("COMMIT")
            count = int(self._connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
            self._groups = count
            self._scratch_bytes = self._path.stat().st_size
            if count < 2:
                raise KernelError("insufficient_clusters", "Cluster covariance requires at least two observed groups.")
            total = torch.zeros((self.width, self.width), dtype=torch.float64)
            correction = torch.zeros_like(total)
            cursor = self._connection.execute("SELECT total, correction FROM scores")
            block_rows = max(1, min(1024, _READ_BYTES // (16 * self.width)))
            while rows := cursor.fetchmany(block_rows):
                values_buffer, errors_buffer = bytearray(), bytearray()
                for values, errors in rows:
                    if len(values) != self.width * 8 or len(errors) != self.width * 8:
                        raise KernelError("cluster_spill_failed", "The owned cluster spill contains an invalid score record.")
                    values_buffer.extend(values)
                    errors_buffer.extend(errors)
                values = torch.frombuffer(values_buffer, dtype=torch.float64).reshape(-1, self.width)
                errors = torch.frombuffer(errors_buffer, dtype=torch.float64).reshape(-1, self.width)
                scores = values + errors
                _finite(scores)
                total, correction = _merge(total, correction, scores.T @ scores, torch.zeros_like(total))
            meat = total + correction
            _finite(meat)
            self._finished = True
            return meat, count
        except (OSError, sqlite3.Error) as exc:
            self._failed = True
            raise self._storage_error() from exc
        except Exception:
            self._failed = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._scratch is not None:
                self._scratch.cleanup()
            self._cache.clear()
            self._free.clear()
            self._totals = torch.empty((0, self.width), dtype=torch.float64)
            self._corrections = torch.empty_like(self._totals)
            self._closed = True
        except (OSError, sqlite3.Error) as exc:
            raise self._storage_error() from exc

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except KernelError:
                # Preserve the original calculation error. The worker supervisor
                # owns this scratch parent and retries cleanup on worker exit.
                pass
        return False
