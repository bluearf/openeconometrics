"""Replayable, bounded-memory weighted OLS.

The design is centered and scaled before a balanced TSQR reduction.  Only
small factors, bounded reader/design batches and a 400-row reporting sample
remain in memory.  Cluster scores are reduced one subset at a time in SQLite;
finite-support HAC kernels retain a bounded window in physical time units.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
from itertools import combinations
import math
import os
from pathlib import Path
import struct
import sys
import sqlite3
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.execution import execution_scope, qr_factor
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _normalize, _TSQRTree
from openecon.streaming_design import (MAX_CATEGORY_BYTES, MAX_CLUSTER_BYTES,
                                      MAX_PARAMETERS, _label, numeric_values,
                                      row_hash_bytes)
from .design import OLSDesign
from .estimation import _ExactWeightSum
from .spec import validate_spec


_WORKING_BYTES = 128 * 1024 * 1024
_READER_ROWS = 65_536
_SAMPLE_ROWS = 400
_HAC_WINDOW_BYTES = 24 * 1024 * 1024


def _time_values(series, name):
    if getattr(series.dtype, "kind", None) in {"i", "u"} and not bool(series.between(-(2**53), 2**53).all()):
        raise AnalysisError("invalid_time", "Time periods exceed exact float64 integer precision; choose a coarser unit.")
    return numeric_values(series, name)


def _planned_rows(spec, width, column_count, lagged=False):
    reserve = 80 * (width + 1) ** 2 * 8
    if spec.cluster:
        reserve += 12 * 1024 * 1024
    if spec.covariance == "hac":
        reserve += _HAC_WINDOW_BYTES
    if lagged:
        reserve += 24 * 1024 * 1024
    remaining = _WORKING_BYTES - reserve
    per_row = 8 * max(width, 1) * 32 + 8 * max(column_count, 1) * 4
    if width > MAX_PARAMETERS or remaining < per_row:
        raise AnalysisError("model_too_wide", "The expanded model exceeds the bounded working-memory budget.")
    return max(1, min(_READER_ROWS, remaining // per_row))


def _anticipated_width(spec, design):
    if not design.categorical:
        return len(design.factor_specs) + int(spec.intercept)
    # Unknown expansion is planned against the largest width compatible with
    # fixed lag/HAC scratch. Actual width is still validated after discovery.
    extras = (12 * 1024 * 1024 if spec.cluster else 0)
    extras += _HAC_WINDOW_BYTES if spec.covariance == "hac" else 0
    extras += 24 * 1024 * 1024 if getattr(design, "lag_specs", None) else 0
    compatible = int(math.sqrt(max(1, (_WORKING_BYTES - extras) * .9) / 640)) - 1
    return min(MAX_PARAMETERS, max(1, compatible))


@dataclass
class _Block:
    frame: pd.DataFrame
    x: torch.Tensor
    y: torch.Tensor
    weights: torch.Tensor
    positions: torch.Tensor


class _Replay:
    """Projected raw hashing and physical positions are independent of width."""

    def __init__(self, spec, source: Dataset, design: OLSDesign):
        self.spec, self.source, self.design = spec, source, design
        self.clusters = [spec.cluster] if isinstance(spec.cluster, str) else list(spec.cluster or [])
        self.columns = list(dict.fromkeys([spec.outcome, *design.required,
                                          *([spec.weights] if spec.weights else []),
                                          *self.clusters,
                                          *([spec.time] if spec.time else []),
                                          *getattr(design, "lag_columns", [])]))
        self.sample_columns = list(dict.fromkeys([spec.outcome, *getattr(design, "current_required", design.required),
                                                 *([spec.weights] if spec.weights else []),
                                                 *self.clusters,
                                                 *([spec.time] if spec.time else []),
                                                 *getattr(design, "lag_columns", [])]))
        absent = set(self.columns) - set(source.columns)
        if absent:
            raise AnalysisError("missing_columns", f"Missing columns: {', '.join(sorted(absent))}.")
        self.baseline = None
        self.discovery_baseline = None
        self.passes = 0
        self.maximum_rows = 0
        self.sample_positions = []
        self.category_signature = None
        anticipated = _anticipated_width(spec, design)
        self.reader_rows = self._rows_for(anticipated)
        self.rows = self.reader_rows

    def _rows_for(self, width):
        # Reserve raw projected columns in addition to expanded numerical
        # vectors. Reader size is fixed across discovery and later CSV passes.
        return _planned_rows(self.spec, width, len(self.columns), bool(getattr(self.design, "lag_specs", None)))

    def plan(self):
        # Arrow, frames and producer batches preserve scalar dtypes across
        # reader partitions. Once the categorical width is known they can use
        # the actual model budget instead of retaining the discovery's
        # worst-case (384-column) reader size. CSV dtype inference depends on
        # chunk boundaries, so keep its partition stable for source digests.
        if self.source.provenance.get("kind") != "csv":
            self.reader_rows = self._rows_for(len(self.design.terms))
        self.rows = min(self.reader_rows, self._rows_for(len(self.design.terms)))

    def _retained(self, raw: pd.DataFrame, start: int):
        missing = raw.loc[:, self.sample_columns].isna().any(axis=1)
        if bool(missing.any()) and self.spec.missing == "raise":
            raise AnalysisError("missing_values", "Model columns contain missing values; use missing='drop'.")
        keep = ~missing
        retained = raw.loc[keep]
        if self.spec.weights:
            if (self.spec.weight_type == "fweight"
                and getattr(retained[self.spec.weights].dtype, "kind", None) in {"i", "u"}
                and not bool(retained[self.spec.weights].between(0, 2**53).all())):
                raise AnalysisError("invalid_weights", "Frequency weights must be nonnegative exactly representable integers.")
            weights = numeric_values(retained[self.spec.weights], self.spec.weights)
            if bool((weights < 0).any()):
                raise AnalysisError("invalid_weights", "Weights must be nonnegative.")
            if self.spec.weight_type == "fweight" and bool((weights != weights.round()).any()):
                raise AnalysisError("invalid_weights", "Frequency weights must be integers.")
            # Above 2**53 float64 cannot distinguish adjacent frequencies.
            if self.spec.weight_type == "fweight" and bool((weights > 2**53).any()):
                raise AnalysisError("invalid_weights", "Each frequency weight must be exactly representable in float64.")
            positive = weights > 0
            retained = retained.iloc[positive.nonzero().flatten().tolist()]
            local = torch.tensor(keep.to_numpy().nonzero()[0], dtype=torch.int64)[positive]
        else:
            local = torch.tensor(keep.to_numpy().nonzero()[0], dtype=torch.int64)
        return retained, local + start

    @contextmanager
    def frames(self, *, discovery=False):
        self.source.assert_unchanged()
        iterator = iter(self.source.iter_batches(self.columns, batch_rows=self.reader_rows))
        data_digest, position_digest = hashlib.sha256(), hashlib.sha256()
        counts = [0, 0]
        category_signature = {}

        def generate():
            for raw in iterator:
                if not isinstance(raw, pd.DataFrame):
                    raise AnalysisError("invalid_dataset", "A dataset batch must be a DataFrame.")
                projected = raw.loc[:, self.columns]
                for name in self.design.categorical:
                    column = projected[name]
                    if isinstance(column.dtype, pd.CategoricalDtype):
                        # The capped dictionary is small; equality/order are part
                        # of the fitted design, including unobserved levels.
                        if len(column.cat.categories) > MAX_PARAMETERS + 1:
                            raise AnalysisError("model_too_wide", "Categorical expansion exceeds 384 parameters.")
                        signature = (tuple(column.cat.categories), column.cat.ordered)
                        prior = category_signature.setdefault(name, signature)
                        if prior != signature:
                            raise AnalysisError("source_changed", "Categorical metadata changed between batches.")
                retained, positions = self._retained(projected, counts[0])
                data_digest.update(row_hash_bytes(projected))
                counts[0] += len(projected)
                yield projected, retained, positions

        def record_positions(positions):
            counts[1] += len(positions)
            position_digest.update(positions.numpy().tobytes())

        generator = generate()
        completed = False
        try:
            yield generator, record_positions
            # Every fitting caller exhausts the pass. Closing a prediction
            # iterator early cannot establish a complete source hash.
            completed = generator.gi_frame is None
            if not completed:
                raise AnalysisError("incomplete_replay", "The fitting pass did not exhaust the dataset.")
            self.source.assert_unchanged()
            record = {"original": counts[0], "used": counts[1],
                      "data_hash": data_digest.hexdigest(),
                      "positions_hash": position_digest.hexdigest()}
            raw_record = {"original": record["original"], "data_hash": record["data_hash"]}
            if self.discovery_baseline is not None and raw_record != self.discovery_baseline:
                raise AnalysisError("source_changed", "The model data changed after category discovery.")
            if discovery:
                self.discovery_baseline = raw_record
                self.category_signature = category_signature
            elif self.baseline is None:
                self.baseline = record
                if self.category_signature is None:
                    self.category_signature = category_signature
                elif category_signature != self.category_signature:
                    raise AnalysisError("source_changed", "Categorical metadata changed after discovery.")
            elif record != self.baseline or category_signature != self.category_signature:
                raise AnalysisError("source_changed", "The model data changed between replay passes.")
            self.passes += 1
        finally:
            generator.close()
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def batches(self):
        begin = getattr(self.design, "begin_streaming_pass", None)
        if begin is not None:
            begin()
        with self.frames() as (frames, record_positions):
            for _, retained, positions in frames:
                for start in range(0, len(retained), self.rows):
                    frame = retained.iloc[start:start + self.rows]
                    try:
                        x = self.design.encode(frame)
                    except AnalysisError as exc:
                        if exc.code == "unknown_category" and self.discovery_baseline is not None:
                            raise AnalysisError("source_changed", "A category changed after bounded design discovery.") from exc
                        raise
                    physical_positions = positions[start:start + self.rows]
                    finite = torch.isfinite(x).all(dim=1)
                    if not bool(finite.all()):
                        if self.spec.missing == "raise":
                            raise AnalysisError("missing_values", "A formula transform or lag produces missing/non-finite values.")
                        x = x[finite]
                        frame = frame.iloc[finite.nonzero().flatten().tolist()]
                        physical_positions = physical_positions[finite]
                    record_positions(physical_positions)
                    if not len(frame):
                        continue
                    y = numeric_values(frame[self.spec.outcome], self.spec.outcome)
                    weights = (numeric_values(frame[self.spec.weights], self.spec.weights)
                               if self.spec.weights else torch.ones(len(frame), dtype=torch.float64))
                    self.maximum_rows = max(self.maximum_rows, len(frame))
                    yield _Block(frame, x, y, weights, physical_positions)


def _discover(replay: _Replay):
    design = replay.design
    if not design.categorical:
        design._build_blocks()
        return
    levels = {name: {} for name in design.categorical}
    declared = {}
    accounted = 0

    def add(name, value):
        nonlocal accounted
        if not isinstance(value, (str, int, float, bool)):
            # NumPy numeric scalars from declared pandas dictionaries are
            # converted without coercing strings or dates.
            if hasattr(value, "item") and getattr(value, "dtype", None) is not None:
                value = value.item()
            if not isinstance(value, (str, int, float, bool)):
                raise AnalysisError("unsupported_category", "Category levels must be strings, numbers or booleans.")
        if isinstance(value, float) and not math.isfinite(value):
            raise AnalysisError("unsupported_category", "Category levels must be finite.")
        if sys.getsizeof(value) + 128 > MAX_CATEGORY_BYTES:
            raise AnalysisError("category_budget", "A category label exceeds the metadata budget.")
        key = _label(value, budget=MAX_CATEGORY_BYTES)
        if key in levels[name]:
            return
        accounted += sys.getsizeof(key) + sys.getsizeof(value) + 128
        if accounted > MAX_CATEGORY_BYTES:
            raise AnalysisError("category_budget", "Category labels exceed the bounded metadata budget.")
        levels[name][key] = value
        if len(levels[name]) > MAX_PARAMETERS + 1:
            raise AnalysisError("model_too_wide", "Categorical expansion exceeds 384 parameters.")

    with replay.frames(discovery=True) as (frames, _):
        for raw, _, _ in frames:
            for name in design.categorical:
                values = raw[name]
                if isinstance(values.dtype, pd.CategoricalDtype):
                    if name not in declared:
                        declared[name] = list(values.cat.categories)
                        for value in declared[name]:
                            add(name, value)
                else:
                    for value in values.array:
                        if not pd.isna(value):
                            add(name, value)
    for name in design.categorical:
        try:
            design.categories[name] = (declared[name] if name in declared
                                       else sorted(levels[name].values()))
        except TypeError as exc:
            raise AnalysisError("unsupported_category", "Category levels must have one sortable scalar type.") from exc
        if not design.categories[name]:
            raise AnalysisError("no_observations", "A categorical model column has no observed levels.")
    # Check expansion counts before OLSDesign materializes any Cartesian
    # product. The conservative count may include duplicate formula blocks.
    width = int(design.intercept)
    for factors in design.factor_specs:
        count = 1
        for name, kind, _ in factors:
            if kind == "categorical":
                count *= max(0, len(design.categories[name]) - int(design.intercept))
            if count > MAX_PARAMETERS:
                raise AnalysisError("model_too_wide", "Categorical interactions exceed 384 parameters.")
        width += count
        if width > MAX_PARAMETERS:
            raise AnalysisError("model_too_wide", "The expanded model exceeds 384 parameters.")
    design._build_blocks()


class _WeightedMoments:
    """Chan moments in maximum-normalized x/y/weight units."""

    def __init__(self, width, intercept=False, importance=False):
        self.n = 0
        self.weight_max = 0.0
        self.mass = 0.0
        self.magnitude = torch.zeros(width, dtype=torch.float64)
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.m2 = _CompensatedSum((width,))
        self.frequency_n = 0
        self.anchor = None
        self.actual_magnitude = torch.zeros(width, dtype=torch.float64)
        self.intercept = intercept
        self.importance_sum = _ExactWeightSum() if importance else None

    def add(self, values, weights, frequency):
        if self.anchor is None:
            self.anchor = values[0].clone() if self.intercept else torch.zeros(values.shape[1], dtype=torch.float64)
            if self.intercept:
                self.anchor[0] = 0
        weight_max = max(self.weight_max, float(weights.max()))
        weight_ratio = self.weight_max / weight_max
        self.mass *= weight_ratio
        self.m2.scale(torch.tensor(weight_ratio, dtype=torch.float64, device="cpu"))
        self.weight_max = weight_max
        normalized_weights = weights / weight_max
        block_mass = float(normalized_weights.sum())
        actual_magnitude = torch.maximum(self.actual_magnitude, values.abs().amax(dim=0))
        centered = values - self.anchor
        overflow = ~torch.isfinite(centered).all(dim=0)
        new_anchor = torch.where(overflow, torch.zeros_like(self.anchor), self.anchor)
        centered = values - new_anchor
        old_magnitude = torch.where(overflow, actual_magnitude, self.magnitude)
        magnitude = torch.maximum(old_magnitude, centered.abs().amax(dim=0))
        divisor = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
        ratio = self.magnitude / divisor
        self.mean = self.mean * ratio + (self.anchor / divisor - new_anchor / divisor)
        self.m2.scale(ratio.square())
        normalized = centered / divisor
        block_mean = (normalized * normalized_weights[:, None]).sum(dim=0) / block_mass
        block_m2 = ((normalized - block_mean).square() * normalized_weights[:, None]).sum(dim=0)
        total_mass = self.mass + block_mass
        delta = block_mean - self.mean
        self.mean += delta * (block_mass / total_mass)
        self.m2.add(block_m2 + delta.square() * (self.mass * (block_mass / total_mass)))
        self.mass = total_mass
        self.magnitude = magnitude
        self.actual_magnitude = actual_magnitude
        self.anchor = new_anchor
        self.n += len(values)
        if frequency:
            self.frequency_n += sum(int(value) for value in weights.tolist())
        if self.importance_sum is not None:
            self.importance_sum.add(weights)

    def normalization(self, spec, covariance):
        if spec.weight_type in {"aweight", "pweight"} or (spec.weight_type == "iweight" and covariance != "nonrobust"):
            multiplier = self.n / self.mass
            nobs = self.n
        else:
            multiplier = self.weight_max
            nobs = (self.frequency_n if spec.weight_type == "fweight" else
                    int(self.importance_sum.total()) if self.importance_sum is not None else self.n)
        if not math.isfinite(multiplier) or nobs <= 0:
            raise AnalysisError("invalid_weights", "The effective weight sum is not representable.")
        return multiplier, nobs


def _independent(factor, terms, depth):
    """Deterministic omission: intercept first, then the user's term order."""
    width = len(terms)
    largest = float(torch.linalg.svdvals(factor).max())
    tolerance = torch.finfo(torch.float64).eps * max(1, width) * max(1, depth) * largest
    basis, kept = [], []
    priority = sorted(range(width), key=lambda index: terms[index] != "Intercept")
    for index in priority:
        residual = factor[:, index].clone()
        # Reorthogonalization is necessary for near-dependent columns.
        for _ in range(2):
            for vector in basis:
                residual -= vector * torch.dot(vector, residual)
        norm = float(torch.linalg.vector_norm(residual))
        if norm > tolerance:
            basis.append(residual / norm)
            kept.append(index)
    kept.sort()
    if not kept:
        raise AnalysisError("singular_design", "No estimable design column remains.")
    return kept


def _cluster_keys(frame, columns):
    keys, size = [], sys.getsizeof([])
    typed = {name: getattr(frame[name].dtype, "kind", None) == "M" for name in columns}
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        pieces = []
        for name, value in zip(columns, row, strict=True):
            item = _label(value, budget=MAX_CLUSTER_BYTES, typed_temporal=typed[name])
            if sum(len(piece) for piece in pieces) + len(item) + 8 > MAX_CLUSTER_BYTES:
                raise AnalysisError("cluster_key_budget", "A combined cluster label exceeds the bounded key budget.")
            pieces.extend((struct.pack("!Q", len(item)), item))
        key = b"".join(pieces)
        size += sys.getsizeof(key) + 9
        if size > MAX_CLUSTER_BYTES:
            raise AnalysisError("cluster_key_budget", "Cluster labels exceed the bounded batch-key budget.")
        keys.append(key)
    return keys


def _add_cluster(accumulator, frame, scores, columns):
    try:
        keys = _cluster_keys(frame, columns)
    except AnalysisError as exc:
        if exc.code != "cluster_key_budget" or len(frame) <= 1:
            raise
        keys = None
    if keys is None:
        middle = len(frame) // 2
        _add_cluster(accumulator, frame.iloc[:middle], scores[:middle], columns)
        _add_cluster(accumulator, frame.iloc[middle:], scores[middle:], columns)
    else:
        accumulator.add(keys, scores)


class _GroupStore:
    """Disk-backed group Gram matrices and scores with a fixed 4 MiB cache.

    CR2/CR3 and cluster deletion need k-by-k information per group, rather
    than the k-vector used by CR1. No row-by-k-squared batch is created.
    """

    def __init__(self, width, directory):
        self.k = width
        self.width = width * width + width
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.maximum_cache_bytes = 0
        self.writes = 0
        self.closed = False
        self.scratch = tempfile.TemporaryDirectory(prefix="openecon-ols-groups-", dir=directory)
        self.path = Path(self.scratch.name) / "groups.sqlite3"
        self.connection = None
        try:
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA cache_size=-2048")
            self.connection.execute("PRAGMA mmap_size=0")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("CREATE TABLE groups (key BLOB PRIMARY KEY, total BLOB, correction BLOB) WITHOUT ROWID")
            self.path.chmod(0o600)
        except Exception:
            if self.connection is not None:
                self.connection.close()
            self.scratch.cleanup()
            raise

    def _write(self, key, value):
        self.connection.execute("INSERT OR REPLACE INTO groups VALUES (?, ?, ?)",
                                (key, value.value.numpy().tobytes(), value.correction.numpy().tobytes()))
        self.writes += 1

    def _evict(self):
        key, value = self.cache.popitem(last=False)
        self.cache_bytes -= 16 * self.width + sys.getsizeof(key) + 512
        self._write(key, value)

    def _add(self, key, increment):
        entry = self.cache.get(key)
        if entry is None:
            cost = 16 * self.width + sys.getsizeof(key) + 512
            while self.cache and self.cache_bytes + cost > 4 * 1024 * 1024:
                self._evict()
            row = self.connection.execute("SELECT total, correction FROM groups WHERE key=?", (key,)).fetchone()
            entry = _CompensatedSum((self.width,))
            if row:
                if len(row[0]) != self.width * 8 or len(row[1]) != self.width * 8:
                    raise AnalysisError("cluster_spill_failed", "An owned cluster Gram record has an invalid size.")
                entry.value.copy_(torch.frombuffer(bytearray(row[0]), dtype=torch.float64))
                entry.correction.copy_(torch.frombuffer(bytearray(row[1]), dtype=torch.float64))
            entry.add(increment)
            if cost > 4 * 1024 * 1024:
                self._write(key, entry)
                return
            self.cache[key] = entry
            self.cache_bytes += cost
            self.maximum_cache_bytes = max(self.maximum_cache_bytes, self.cache_bytes)
        else:
            self.cache.move_to_end(key)
            entry.add(increment)

    def add(self, frame, q, scores, columns):
        try:
            keys = _cluster_keys(frame, columns)
        except AnalysisError as exc:
            if exc.code != "cluster_key_budget" or len(frame) <= 1:
                raise
            keys = None
        if keys is None:
            middle = len(frame) // 2
            self.add(frame.iloc[:middle], q[:middle], scores[:middle], columns)
            self.add(frame.iloc[middle:], q[middle:], scores[middle:], columns)
            return
        groups = {}
        for row, key in enumerate(keys):
            groups.setdefault(key, []).append(row)
        for key, positions in groups.items():
            index = torch.tensor(positions, dtype=torch.int64)
            rows = q[index]
            gram = rows.T @ rows
            score = scores[index].sum(dim=0)
            self._add(key, torch.cat((score, gram.flatten())))

    def items(self):
        while self.cache:
            self._evict()
        cursor = self.connection.execute("SELECT total FROM groups")
        try:
            for (blob,) in cursor:
                value = torch.frombuffer(bytearray(blob), dtype=torch.float64)
                if len(value) != self.width or not bool(torch.isfinite(value).all()):
                    raise AnalysisError("cluster_spill_failed", "An owned cluster Gram record is invalid.")
                yield value[:self.k], value[self.k:].reshape(self.k, self.k)
        finally:
            cursor.close()

    @property
    def diagnostics(self):
        return {"cluster_aggregation": "sqlite_gram_spill", "cluster_cache_limit_bytes": 4 * 1024 * 1024,
                "cluster_peak_cache_accounted_bytes": self.maximum_cache_bytes,
                "cluster_sqlite_cache_bytes": 2 * 1024 * 1024,
                "cluster_spill_writes": self.writes, "cluster_scratch_bytes": self.path.stat().st_size,
                "cluster_gram_columns": self.width}

    def close(self):
        if not self.closed:
            self.connection.close()
            self.scratch.cleanup()
            self.cache.clear()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, exc, traceback):
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except (OSError, sqlite3.Error):
                pass
        return False


class _Multinomial:
    """Exact conditional-binomial multinomial counts, O(1) random state."""

    def __init__(self, total, population, generator):
        if total > 2**53 or population > 2**53:
            raise AnalysisError("invalid_resampling_options", "Bootstrap counts exceed exact float64 random-count precision.")
        self.remaining = torch.tensor(float(total), dtype=torch.float64)
        self.population = torch.tensor(float(population), dtype=torch.float64)
        self.generator = generator

    def draw(self, mass):
        value = float(mass)
        if value == float(self.population):
            count = self.remaining.clone()
        else:
            probability = torch.tensor(value, dtype=torch.float64) / self.population
            count = torch.binomial(self.remaining, probability.clamp(0, 1), generator=self.generator)
        self.remaining -= count
        self.population -= value
        return count


class _BootstrapGroups:
    """First-seen group ids and one replication's counts live on disk."""

    def __init__(self, replay, directory):
        self.scratch = tempfile.TemporaryDirectory(prefix="openecon-bootstrap-groups-", dir=directory)
        self.path = Path(self.scratch.name) / "groups.sqlite3"
        self.connection = None
        self.count = 0
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.columns = replay.clusters
        try:
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            self.path.chmod(0o600)
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA cache_size=-2048")
            self.connection.execute("PRAGMA mmap_size=0")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("CREATE TABLE groups (key BLOB PRIMARY KEY, id INTEGER UNIQUE, draw REAL NOT NULL) WITHOUT ROWID")
            for block in replay.batches():
                self._discover(block.frame)
        except Exception:
            self.close()
            raise
        if self.count < 2:
            self.close()
            raise AnalysisError("insufficient_clusters", "Cluster bootstrap requires at least two groups.")

    def _discover(self, frame):
        try:
            keys = _cluster_keys(frame, self.columns)
        except AnalysisError as exc:
            if exc.code != "cluster_key_budget" or len(frame) <= 1:
                raise
            keys = None
        if keys is None:
            middle = len(frame) // 2
            self._discover(frame.iloc[:middle])
            self._discover(frame.iloc[middle:])
            return
        for key in dict.fromkeys(keys):
            inserted = self.connection.execute("INSERT OR IGNORE INTO groups VALUES (?, ?, 0)", (key, self.count)).rowcount
            self.count += int(inserted != 0)

    def resample(self, generator):
        self.cache.clear()
        self.cache_bytes = 0
        draw = _Multinomial(self.count, self.count, generator)
        cursor = self.connection.execute("SELECT id FROM groups ORDER BY id")
        pending = []
        try:
            for (group,) in cursor:
                pending.append((float(draw.draw(1)), group))
                if len(pending) == 1024:
                    self.connection.executemany("UPDATE groups SET draw=? WHERE id=?", pending)
                    pending.clear()
            if pending:
                self.connection.executemany("UPDATE groups SET draw=? WHERE id=?", pending)
        finally:
            cursor.close()

    def weights(self, frame):
        try:
            keys = _cluster_keys(frame, self.columns)
        except AnalysisError as exc:
            if exc.code != "cluster_key_budget" or len(frame) <= 1:
                raise
            keys = None
        if keys is None:
            middle = len(frame) // 2
            return torch.cat((self.weights(frame.iloc[:middle]), self.weights(frame.iloc[middle:])))
        result = torch.empty(len(keys), dtype=torch.float64)
        for row, key in enumerate(keys):
            if key in self.cache:
                self.cache.move_to_end(key)
                value = self.cache[key]
            else:
                record = self.connection.execute("SELECT draw FROM groups WHERE key=?", (key,)).fetchone()
                if record is None:
                    raise AnalysisError("source_changed", "A clustering label changed after bootstrap discovery.")
                value = record[0]
                cost = sys.getsizeof(key) + 256
                while self.cache and self.cache_bytes + cost > 4 * 1024 * 1024:
                    old, _ = self.cache.popitem(last=False)
                    self.cache_bytes -= sys.getsizeof(old) + 256
                if cost <= 4 * 1024 * 1024:
                    self.cache[key] = value
                    self.cache_bytes += cost
            result[row] = value
        return result

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.scratch.cleanup()


def _bootstrap(replay, normalized, kept, original_theta, y_magnitude, transform, nobs):
    spec = replay.spec
    reps, seed = spec.options.get("reps", 199), spec.options.get("seed", 0)
    k = len(kept)
    generator = torch.Generator().manual_seed(seed)
    physical = replay.baseline["used"]
    units = nobs if spec.weight_type == "fweight" and not replay.clusters else physical
    scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
    groups = _BootstrapGroups(replay, Path(scratch) if scratch else None) if replay.clusters else None
    if groups is not None:
        units = groups.count
    mean = torch.zeros(k, dtype=torch.float64)
    cross = _CompensatedSum((k, k))
    success, failed = 0, 0
    try:
        for _ in range(reps):
            if groups is not None:
                groups.resample(generator)
            draw = _Multinomial(units, units, generator) if groups is None else None
            tree = _TSQRTree()
            selected_rows = 0
            for block in replay.batches():
                z, y, w = normalized(block)
                if groups is not None:
                    weights = w * groups.weights(block.frame)
                else:
                    counts = torch.empty(len(w), dtype=torch.float64)
                    mass = block.weights if spec.weight_type == "fweight" else torch.ones_like(w)
                    for row in range(len(w)):
                        counts[row] = draw.draw(mass[row])
                    weights = counts if spec.weight_type == "fweight" else w * counts
                positive = weights > 0
                selected_rows += int(positive.sum())
                if bool(positive.any()):
                    augmented = torch.cat((z[positive][:, kept], y[positive, None]), dim=1) * weights[positive].sqrt()[:, None]
                    tree.add(qr_factor(augmented))
            # The complete raw pass is verified even when the sampled rank is
            # deficient; failures do not conceal changing-source evidence.
            if selected_rows < k:
                failed += 1
                continue
            factor = tree.finish()
            r = factor[:k, :k]
            singular = torch.linalg.svdvals(r)
            threshold = torch.finfo(torch.float64).eps * k * max(1, tree.depth) * float(singular[0])
            if float(singular[-1]) <= threshold:
                failed += 1
                continue
            theta = torch.linalg.solve_triangular(r, factor[:k, k:k + 1], upper=True)[:, 0]
            delta = theta - original_theta
            success += 1
            difference = delta - mean
            mean += difference / success
            cross.add(torch.outer(difference, delta - mean))
    finally:
        if groups is not None:
            groups.close()
    if success < 2:
        raise AnalysisError("insufficient_resamples", "Fewer than two bootstrap replications identified the fitted model.")
    covariance = cross.value * (y_magnitude**2 / (success - 1))
    information = {"scheme": "cluster" if groups is not None else "pairs", "reps": reps,
                   "successful_reps": success, "failed_reps": failed, "seed": seed,
                   "resampling_units": units, "cluster_count": units if groups is not None else None,
                   "frequency_count_resampling": spec.weight_type == "fweight" and groups is None,
                   "random_count_algorithm": "exact conditional-binomial multinomial",
                   "bias_estimate": transform @ mean * y_magnitude,
                   "distribution": "normal", "correction": "Sample covariance across successful bootstrap estimates with B-1 divisor."}
    return covariance, information


class _Contrast:
    """Bounded design-only Satterthwaite/Hansen contrast moments.

    Cluster fits reuse the temporary Gram store during coefficient inference;
    later contrasts rebuild it inside a context manager. Row fits evaluate
    small blocks of contrasts per replay without N-by-N hat matrices.
    """

    def __init__(self, replay, residual_block, bridge, covariance, hansen):
        self.replay, self.residual_block, self.bridge = replay, residual_block, bridge
        self.power = .5 if covariance.endswith("2") else 1.
        self.hansen = hansen
        self.k = len(bridge)

    def _vectors(self, gradients):
        gradients = torch.as_tensor(gradients, dtype=torch.float64)
        if gradients.ndim != 2 or gradients.shape[1] != self.k or not bool(torch.isfinite(gradients).all()):
            raise AnalysisError("invalid_contrast", "Adjusted inference requires finite gradients for the kept coefficients.")
        vectors = gradients @ self.bridge
        scales = vectors.abs().amax(dim=1)
        vectors /= torch.where(scales > 0, scales, torch.ones_like(scales))[:, None]
        return vectors, scales

    def _finish(self, vectors, scales, totals, bbt, units):
        result = []
        for index, vector in enumerate(vectors):
            if float(scales[index]) == 0:
                result.append({"df": math.inf, "scale": 1.})
                continue
            alpha, alpha2, norm, cross = totals[index]
            trace = alpha - norm
            denominator = alpha2 - 2 * cross + bbt[index].square().sum()
            if float(trace) <= 0 or float(denominator) <= 0:
                raise AnalysisError("undefined_adjusted_inference", "The selected contrast has no identified adjusted reference variance.")
            df = float(trace.square() / denominator)
            scale = float((trace / torch.dot(vector, vector)).sqrt()) if self.hansen else 1.
            if not math.isfinite(df) or not math.isfinite(scale):
                raise AnalysisError("numerical_overflow", "Adjusted inference moments exceed float64 precision.")
            result.append({"df": min(float(units), max(1., df)), "scale": scale})
        return result

    def from_groups(self, gradients, store):
        vectors, scales = self._vectors(gradients)
        totals = _CompensatedSum((len(vectors), 4))
        bbt = _CompensatedSum((len(vectors), self.k, self.k))
        units = 0
        for _, gram in store.items():
            units += 1
            values, eigenvectors = torch.linalg.eigh(torch.eye(self.k, dtype=torch.float64) - (gram + gram.T) / 2)
            valid = values > 100 * torch.finfo(torch.float64).eps * self.k
            if self.hansen and not bool(valid.all()):
                raise AnalysisError("unsupported_inference", "Hansen inference with singular cluster annihilators requires a generalized jackknife implementation; ordinary cluster HC3 is insufficient.")
            inverse = torch.where(valid, values.clamp_min(torch.finfo(torch.float64).tiny).pow(-self.power), 0)
            adjusted = (vectors @ eigenvectors * inverse[None, :]) @ eigenvectors.T
            b = adjusted @ gram
            alpha = (b * adjusted).sum(dim=1)
            norm = b.square().sum(dim=1)
            totals.add(torch.stack((alpha, alpha.square(), norm, alpha * norm), dim=1))
            bbt.add(b[:, :, None] * b[:, None, :])
        return self._finish(vectors, scales, totals.value, bbt.value, units)

    def from_rows(self, gradients):
        vectors, scales = self._vectors(gradients)
        totals = _CompensatedSum((len(vectors), 4))
        bbt = _CompensatedSum((len(vectors), self.k, self.k))
        units = 0
        frequency = self.replay.spec.weight_type == "fweight"
        for block in self.replay.batches():
            _, weights, _, q, _ = self.residual_block(block)
            multiplicity = weights if frequency else torch.ones_like(weights)
            units += sum(int(value) for value in weights.tolist()) if frequency else len(weights)
            q = q if frequency else q * weights.sqrt()[:, None]
            complement = 1 - q.square().sum(dim=1)
            valid = complement > 100 * torch.finfo(torch.float64).eps * self.k
            if self.hansen and not bool(valid.all()):
                raise AnalysisError("unsupported_inference", "Hansen inference with unit-leverage rows requires a generalized jackknife implementation; ordinary HC3 is insufficient.")
            inverse = torch.where(valid, complement.clamp_min(torch.finfo(torch.float64).tiny).pow(-self.power), 0)
            increments = []
            for index, vector in enumerate(vectors):
                a = (q @ vector) * inverse
                alpha = a.square()
                b = q * a[:, None]
                norm = b.square().sum(dim=1)
                increments.append(torch.stack((torch.dot(multiplicity, alpha), torch.dot(multiplicity, alpha.square()),
                                               torch.dot(multiplicity, norm), torch.dot(multiplicity, alpha * norm))))
                weighted = b * multiplicity.sqrt()[:, None]
                increment = torch.zeros_like(bbt.value)
                increment[index] = weighted.T @ weighted
                bbt.add(increment)
            totals.add(torch.stack(increments))
        return self._finish(vectors, scales, totals.value, bbt.value, units)

    def controls(self, store=None):
        eye = torch.eye(self.k, dtype=torch.float64)
        # Two compensated k-by-k tensors per contrast stay below 16 MiB.
        size = max(1, min(8, (16 * 1024 * 1024) // (16 * self.k * self.k)))
        result = []
        for start in range(0, self.k, size):
            gradients = eye[start:start + size]
            result.extend(self.from_groups(gradients, store) if store is not None else self.from_rows(gradients))
        return result

    def __call__(self, gradient):
        gradient = torch.as_tensor(gradient, dtype=torch.float64)
        if gradient.ndim != 1:
            raise AnalysisError("invalid_contrast", "Adjusted inference requires one contrast gradient.")
        if not self.replay.clusters:
            return self.from_rows(gradient[None, :])[0]
        scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
        with _GroupStore(self.k, Path(scratch) if scratch else None) as store:
            for block in self.replay.batches():
                _, weights, resid, q, _ = self.residual_block(block)
                store.add(block.frame, q * weights.sqrt()[:, None], q * (weights * resid)[:, None], self.replay.clusters)
            return self.from_groups(gradient[None, :], store)[0]


class _HAC:
    def __init__(self, width, lags, kernel):
        if (lags + 1) * (width + 1) * 8 * 3 > _HAC_WINDOW_BYTES:
            raise AnalysisError("hac_window_budget", "The HAC lag window exceeds the bounded memory budget; reduce lags.")
        self.lags, self.kernel = lags, kernel
        self.meat = _CompensatedSum((width, width))
        self.times = torch.empty(0, dtype=torch.int64)
        self.scores = torch.empty((0, width), dtype=torch.float64)
        self.last_time = None

    def _periods(self, times):
        if bool((times != times.round()).any()) or bool((times.abs() > 2**53).any()):
            raise AnalysisError("invalid_time", "HAC time must contain exactly represented integer periods.")
        periods = times.to(torch.int64)
        if bool((periods[1:] <= periods[:-1]).any()) or (self.last_time is not None and int(periods[0]) <= self.last_time):
            raise AnalysisError("unordered_time", "Streaming HAC requires unique, strictly increasing time periods.")
        return periods

    def add(self, times, scores):
        if not len(times):
            return
        periods = self._periods(times)
        self.meat.add(scores.T @ scores)
        all_times = torch.cat((self.times, periods))
        all_scores = torch.cat((self.scores, scores))
        for lag in range(1, self.lags + 1):
            indices = torch.searchsorted(all_times, periods - lag)
            in_bounds = indices < len(all_times)
            indices = indices.clamp(max=len(all_times) - 1)
            valid = in_bounds & (all_times[indices] == periods - lag)
            if not bool(valid.any()):
                continue
            u = lag / (self.lags + 1)
            weight = (1 - u if self.kernel == "bartlett" else
                      (1 - 6 * u * u + 6 * u**3 if u <= .5 else 2 * (1 - u)**3)
                      if self.kernel == "parzen" else 1.0)
            cross = scores[valid].T @ all_scores[indices[valid]]
            self.meat.add((cross + cross.T) * weight)
        self.last_time = int(periods[-1])
        start = int(torch.searchsorted(all_times, torch.tensor(self.last_time - self.lags, dtype=torch.int64)))
        # clone releases the reader batch rather than retaining its whole storage.
        self.times = all_times[start:].clone()
        self.scores = all_scores[start:].clone()


class _Pilot(_HAC):
    """Newey-West 1994 pilot autocovariances in a bounded scalar window."""

    def __init__(self, n, span, kernel):
        order, exponent, constant = {"bartlett": (1, 2 / 9, 1.1447),
                                     "parzen": (2, 4 / 25, 2.6614),
                                     "quadratic_spectral": (2, 2 / 25, 1.3221)}[kernel]
        pilot = min(int(20 * (n / 100) ** exponent), span)
        super().__init__(1, pilot, kernel)
        self.n, self.order, self.constant = n, order, constant
        self.autocovariances = _CompensatedSum((pilot + 1,))

    def add(self, times, scores):
        if not len(times):
            return
        periods = self._periods(times)
        values = scores[:, 0]
        increment = torch.zeros(self.lags + 1, dtype=torch.float64)
        increment[0] = torch.dot(values, values)
        all_times = torch.cat((self.times, periods))
        all_values = torch.cat((self.scores[:, 0], values))
        for lag in range(1, self.lags + 1):
            indices = torch.searchsorted(all_times, periods - lag)
            in_bounds = indices < len(all_times)
            indices = indices.clamp(max=len(all_times) - 1)
            valid = in_bounds & (all_times[indices] == periods - lag)
            increment[lag] = torch.dot(values[valid], all_values[indices[valid]])
        self.autocovariances.add(increment)
        self.last_time = int(periods[-1])
        start = int(torch.searchsorted(all_times, torch.tensor(self.last_time - self.lags, dtype=torch.int64)))
        self.times = all_times[start:].clone()
        self.scores = all_values[start:, None].clone()

    def select(self):
        covariances = self.autocovariances.value / self.n
        sigma0 = float(covariances[0])
        sum_zero = sigma0 + 2 * float(covariances[1:].sum())
        powers = torch.arange(1, self.lags + 1, dtype=torch.float64).pow(self.order)
        sum_order = 2 * float(torch.dot(covariances[1:], powers))
        if not math.isfinite(sum_zero) or not math.isfinite(sum_order):
            raise AnalysisError("numerical_overflow", "Automatic HAC score moments exceed float64 precision; rescale predictors.")
        zero = abs(sum_zero) <= 100 * torch.finfo(torch.float64).eps * max(sigma0, torch.finfo(torch.float64).tiny)
        if zero:
            lags = 0
        else:
            gamma = self.constant * abs(sum_order / sum_zero) ** (2 / (2 * self.order + 1))
            selected = min(gamma * self.n ** (1 / (2 * self.order + 1)), self.lags)
            lags = selected if self.kernel == "quadratic_spectral" else math.floor(selected)
        return lags, {"lag_selection": "Newey-West 1994 plug-in", "automatic_lags": True,
                      "pilot_lags": self.lags, "lag_selection_order": self.order,
                      "lag_selection_constant": self.constant, "lag_selection_time_gaps": True}, zero


class _QuadraticSpectral(_HAC):
    """Exact all-pairs QS: disk scores, bounded pair tiles, quadratic work."""

    def __init__(self, width, lags, directory):
        super().__init__(width, 0, "quadratic_spectral")
        self.lags = lags
        self.scratch = tempfile.TemporaryDirectory(prefix="openecon-hac-", dir=directory)
        self.path = Path(self.scratch.name) / "scores.sqlite3"
        self.connection = None
        self.rows = 0
        self.closed = False
        try:
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA cache_size=-2048")
            self.connection.execute("PRAGMA mmap_size=0")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("CREATE TABLE scores (time INTEGER PRIMARY KEY, score BLOB NOT NULL)")
            self.path.chmod(0o600)
        except Exception:
            self.close()
            raise

    def add(self, times, scores):
        if not len(times):
            return
        periods = self._periods(times)
        self.last_time = int(periods[-1])
        self.meat.add(scores.T @ scores)
        for start in range(0, len(scores), 512):
            records = [(int(periods[index]), scores[index].numpy().tobytes())
                       for index in range(start, min(start + 512, len(scores)))]
            self.connection.executemany("INSERT INTO scores VALUES (?, ?)", records)
        self.rows += len(scores)

    @staticmethod
    def _decode(records):
        times = torch.tensor([record[0] for record in records], dtype=torch.int64)
        scores = torch.stack([torch.frombuffer(bytearray(record[1]), dtype=torch.float64) for record in records])
        if not bool(torch.isfinite(scores).all()):
            raise AnalysisError("hac_spill_failed", "An owned HAC score record is invalid.")
        return times, scores

    def finish(self):
        outer = self.connection.execute("SELECT time, score FROM scores ORDER BY time")
        try:
            while records := outer.fetchmany(512):
                times, scores = self._decode(records)
                inner = self.connection.execute("SELECT time, score FROM scores WHERE time>=? ORDER BY time", (int(times[0]),))
                try:
                    while future := inner.fetchmany(512):
                        future_times, future_scores = self._decode(future)
                        distance = future_times[None, :] - times[:, None]
                        theta = distance.to(torch.float64) * (6 * math.pi / 5 / (self.lags + 1))
                        small = theta.abs() < 1e-3
                        square = theta.square()
                        series = 1 - square / 10 + square.square() / 280 - square.pow(3) / 15120
                        safe = torch.where(small, torch.ones_like(theta), theta)
                        exact = 3 * (torch.sin(safe) / safe - torch.cos(safe)) / safe.square()
                        weights = torch.where(small, series, exact) * (distance > 0)
                        cross = scores.T @ (weights @ future_scores)
                        self.meat.add(cross + cross.T)
                finally:
                    inner.close()
        finally:
            outer.close()
        return self.meat.value

    @property
    def diagnostics(self):
        return {"hac_score_storage": "sqlite_spill", "hac_pair_block_rows": 512,
                "hac_sqlite_cache_bytes": 2 * 1024 * 1024,
                "hac_scratch_bytes": self.path.stat().st_size,
                "hac_score_rows": self.rows,
                "computational_complexity": "quadratic time; bounded RAM",
                "noncompact_support": True}

    def close(self):
        if not self.closed:
            if self.connection is not None:
                self.connection.close()
            self.scratch.cleanup()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, exc, traceback):
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except (OSError, sqlite3.Error):
                pass
        return False


def _unsupported(spec, design):
    for factors in design.factor_specs:
        for _, _, tree in factors:
            if (not getattr(design, "streaming_history", False)
                    and any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"L", "D"}
                            for node in ast.walk(tree))):
                raise AnalysisError("streaming_formula_boundary", "Streaming lag/difference formula terms require a cross-batch history and are not implemented.")
    if spec.covariance == "jackknife" and isinstance(spec.cluster, list) and len(spec.cluster) > 1:
        raise AnalysisError("unsupported_streaming_covariance", "Jackknife requires one-way cluster deletion or row deletion; multiway deletion needs an explicit resampling scheme.")
    if spec.covariance == "bootstrap" and isinstance(spec.cluster, list) and len(spec.cluster) > 1:
        raise AnalysisError("unsupported_streaming_covariance", "Bootstrap requires one-way cluster or pairs resampling; multiway resampling needs an explicit scheme.")


@torch.inference_mode()
def _fit_streaming(spec, source: Dataset, *, design=None, resources):
    """Fit a full OLS specification without an N-row tensor or position list.

    Returned x/y/fitted/resid/leverage are reporting samples, explicitly marked
    by samples_only.  The source, reusable design and kept indices are retained
    for the public result's bounded replay prediction iterator.
    """
    covariance_name = validate_spec(spec)
    design = design or OLSDesign(spec.options.get("terms", spec.predictors), spec.categorical,
                                 intercept=spec.intercept, time=spec.time)
    if getattr(design, "lag_specs", None):
        raw_columns = list(dict.fromkeys([spec.outcome, *design.required,
                                         *([spec.weights] if spec.weights else []),
                                         *([spec.cluster] if isinstance(spec.cluster, str) else spec.cluster or []),
                                         *([spec.time] if spec.time else [])]))
        rows = _planned_rows(spec, _anticipated_width(spec, design),
                             len(raw_columns) + len(design.lag_columns), True)
        source = design.with_streaming_history(source, batch_rows=rows, columns=raw_columns)
        design.streaming_history = True
    _unsupported(spec, design)
    if covariance_name == "hac" and not spec.time:
        raise AnalysisError("invalid_time", "Streaming HAC requires an explicit time column.")
    replay = _Replay(spec, source, design)

    def make_hac(lags):
        kernel = spec.options.get("kernel", "bartlett")
        if kernel == "quadratic_spectral":
            scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
            return resources.enter_context(_QuadraticSpectral(len(kept), lags, Path(scratch) if scratch else None))
        return _HAC(len(kept), lags, kernel)
    _discover(replay)
    replay.plan()
    width = len(design.terms)
    moments = _WeightedMoments(width + 1, spec.intercept,
                               spec.weight_type == "iweight" and covariance_name == "nonrobust")
    time_min, time_max = None, None
    for block in replay.batches():
        moments.add(torch.cat((block.x, block.y[:, None]), dim=1), block.weights,
                    spec.weight_type == "fweight")
        if covariance_name == "hac":
            periods = _time_values(block.frame[spec.time], spec.time)
            lower, upper = float(periods.min()), float(periods.max())
            time_min = lower if time_min is None else min(time_min, lower)
            time_max = upper if time_max is None else max(time_max, upper)
    if moments.n == 0:
        code = "empty_data" if replay.baseline["original"] == 0 else "empty_sample"
        raise AnalysisError(code, "No observations remain after missing values and zero weights.")
    multiplier, nobs = moments.normalization(spec, covariance_name)
    magnitude = torch.where(moments.magnitude[:width] > 0, moments.magnitude[:width],
                            torch.ones(width, dtype=torch.float64))
    centers = moments.mean[:width].clone() if spec.intercept else torch.zeros(width, dtype=torch.float64)
    variance = moments.m2.value[:width] / moments.mass
    if not spec.intercept:
        variance = variance + moments.mean[:width].square()
    rms = variance.clamp_min(0).sqrt()
    rms[rms == 0] = 1
    intercept_index = design.terms.index("Intercept") if "Intercept" in design.terms else None
    if intercept_index is not None:
        centers[intercept_index] = 0
        rms[intercept_index] = 1
        magnitude[intercept_index] = 1
    outcome_magnitude = moments.magnitude[-1:].clone()
    outcome_magnitude[outcome_magnitude == 0] = 1
    outcome_center = moments.mean[-1:].clone() if spec.intercept else torch.zeros(1, dtype=torch.float64)
    unit = torch.ones(1, dtype=torch.float64)

    def normalized(block):
        z = _normalize(block.x - moments.anchor[:width], magnitude, centers, rms)
        y = _normalize(block.y[:, None] - moments.anchor[-1], outcome_magnitude, outcome_center, unit)[:, 0]
        raw_weights = (spec.weight_type == "fweight"
                       or spec.weight_type == "iweight" and covariance_name == "nonrobust")
        w = block.weights if raw_weights else block.weights / moments.weight_max * multiplier
        if not bool(torch.isfinite(w).all()) or bool((w <= 0).any()):
            raise AnalysisError("invalid_weights", "Normalized weights are not positive finite float64 values.")
        return z, y, w

    tree = _TSQRTree()
    for block in replay.batches():
        z, y, w = normalized(block)
        augmented = torch.cat((z, y[:, None]), dim=1) * w.sqrt()[:, None]
        factor = qr_factor(augmented)
        tree.add(factor)
    augmented_factor = tree.finish()
    kept = _independent(augmented_factor[:, :width], design.terms, tree.depth)
    terms = [design.terms[index] for index in kept]
    omitted = [term for index, term in enumerate(design.terms) if index not in kept]
    k = len(kept)
    df_resid = nobs - k
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations", "OLS needs positive residual degrees of freedom.")
    reduced = torch.linalg.qr(augmented_factor[:, [*kept, width]], mode="r").R
    # Factor reduction is finished. Release its tree before postestimation
    # contrast blocks or disk-backed covariance work allocate scratch.
    tree.levels.clear()
    r = reduced[:k, :k]
    theta = torch.linalg.solve_triangular(r, reduced[:k, k:k + 1], upper=True)[:, 0]
    inverse_r = torch.linalg.solve_triangular(r, torch.eye(k, dtype=torch.float64), upper=True)
    transform = torch.diag(1 / magnitude[kept] / rms[kept])
    kept_intercept = terms.index("Intercept") if "Intercept" in terms else None
    if kept_intercept is not None:
        transform[kept_intercept, :] = -(moments.anchor[kept] / magnitude[kept] + centers[kept]) / rms[kept]
        transform[kept_intercept, kept_intercept] = 1
    original_theta = theta * outcome_magnitude[0]
    if kept_intercept is not None:
        original_theta[kept_intercept] += moments.anchor[-1] + outcome_center[0] * outcome_magnitude[0]
    params = transform @ original_theta
    bread_normalized = inverse_r @ inverse_r.T
    bread = transform @ bread_normalized @ transform.T
    rss, tss = _CompensatedSum(()), _CompensatedSum(())
    meat = _CompensatedSum((k, k))
    deleted_sum = _CompensatedSum((k,))
    deleted_cross = _CompensatedSum((k, k))
    sample = {name: [] for name in ("x", "y", "weights", "resid", "fitted", "leverage", "positions")}
    sample_count = 0
    unit_leverage = False
    log_weights = _CompensatedSum(())
    lags = None
    hac = None
    pilot = None
    selection = {}
    if covariance_name == "hac":
        lags = spec.options["lags"]
        if lags == "auto":
            kernel = spec.options.get("kernel", "bartlett")
            if kernel == "truncated":
                raise AnalysisError("invalid_covariance_options", "Automatic HAC lags require Bartlett, Parzen or quadratic spectral.")
            pilot = _Pilot(moments.n, int(time_max - time_min), kernel)
        else:
            hac = make_hac(lags)

    def residual_block(block):
        z, y, w = normalized(block)
        z = z[:, kept]
        resid = (y - z @ theta) * outcome_magnitude[0]
        q = torch.linalg.solve_triangular(r.T, z.T, upper=False).T
        leverage = w * q.square().sum(dim=1)
        return z, w, resid, q, leverage

    for block in replay.batches():
        _, w, resid, q, leverage = residual_block(block)
        rss.add((w * resid.square()).sum())
        centered_y = (((block.y - moments.anchor[-1]) / outcome_magnitude[0] - outcome_center[0])
                      * outcome_magnitude[0] if spec.intercept else block.y)
        tss.add((w * centered_y.square()).sum())
        if covariance_name in {"HC0", "HC1", "HC2", "HC3"}:
            row_leverage = q.square().sum(dim=1) if spec.weight_type == "fweight" else leverage
            adjustment = torch.ones_like(resid)
            if covariance_name in {"HC2", "HC3"}:
                complement = 1 - row_leverage
                valid = complement > torch.finfo(torch.float64).eps * 100 * k
                unit_leverage |= not bool(valid.all())
                adjustment = complement.clamp_min(torch.finfo(torch.float64).tiny)
                if covariance_name == "HC2":
                    adjustment = adjustment.sqrt()
            score_weight = w.sqrt() if spec.weight_type == "fweight" else w
            scalar = score_weight * resid / adjustment
            if covariance_name in {"HC2", "HC3"}:
                scalar = torch.where(valid, scalar, 0)
            scores = q * scalar[:, None]
            meat.add(scores.T @ scores)
        elif hac is not None:
            hac.add(_time_values(block.frame[spec.time], spec.time), q * (w * resid)[:, None])
        elif pilot is not None:
            slopes = [index for index in kept if index != intercept_index]
            factor_sum = block.x[:, slopes].sum(dim=1) if slopes else torch.zeros_like(resid)
            pilot.add(_time_values(block.frame[spec.time], spec.time), (factor_sum * w * resid)[:, None])
        elif covariance_name == "jackknife" and not replay.clusters:
            complement = 1 - (leverage / w if spec.weight_type == "fweight" else leverage)
            if bool((complement <= 100 * torch.finfo(torch.float64).eps * k).any()):
                raise AnalysisError("jackknife_rank_deficient", "Deleting a row removes an identified coefficient; jackknife covariance is undefined.")
            scalar = resid / complement if spec.weight_type == "fweight" else w * resid / complement
            deleted = q * scalar[:, None]
            multiplicity = w if spec.weight_type == "fweight" else torch.ones_like(w)
            deleted_sum.add((deleted * multiplicity[:, None]).sum(dim=0))
            weighted_deleted = deleted * multiplicity.sqrt()[:, None]
            deleted_cross.add(weighted_deleted.T @ weighted_deleted)
        if spec.weight_type == "aweight":
            log_weights.add(w.log().sum())
        take = min(_SAMPLE_ROWS - sample_count, len(block.y))
        if take:
            sample_count += take
            sample["x"].append(block.x[:take, kept].clone())
            sample["y"].append(block.y[:take].clone())
            sample["weights"].append(w[:take].clone())
            sample["resid"].append(resid[:take].clone())
            sample["fitted"].append((block.y[:take] - resid[:take]).clone())
            sample["leverage"].append(leverage[:take].clone())
            sample["positions"].append(block.positions[:take].clone())
    residual_ss, total_ss = float(rss.value), float(tss.value)
    sigma2 = residual_ss / df_resid
    warnings = []
    if pilot is not None:
        lags, selection, zero = pilot.select()
        if zero:
            warnings.append("Automatic HAC lag selection had zero long-run pilot variance and selected zero lags.")
        hac = make_hac(lags)
        for block in replay.batches():
            _, w, resid, q, _ = residual_block(block)
            hac.add(_time_values(block.frame[spec.time], spec.time), q * (w * resid)[:, None])
    if isinstance(hac, _QuadraticSpectral):
        hac.finish()
        warnings.append("Quadratic spectral HAC has noncompact support: all observed time differences are included; memory is bounded but work can be quadratic in rows.")
    if unit_leverage:
        warnings.append("Unit-leverage observations were excluded from HC2/HC3 covariance using the Moore-Penrose convention.")
    if omitted:
        warnings.append("Collinear terms omitted: " + ", ".join(omitted) + ".")
    group_counts, cluster_diagnostics = {}, []
    df_inference = df_resid
    extra_inference = {}
    adjustment = None
    controls = None
    if spec.options.get("dfadjust") or spec.options.get("hansen"):
        adjustment = _Contrast(replay, residual_block, transform @ inverse_r, covariance_name,
                               bool(spec.options.get("hansen")))
    if covariance_name == "nonrobust":
        covariance_normalized = bread_normalized * sigma2
    elif covariance_name == "bootstrap":
        covariance_normalized, extra_inference = _bootstrap(replay, normalized, kept, theta,
                                                            float(outcome_magnitude[0]), transform, nobs)
        df_inference = math.inf
        if extra_inference["failed_reps"]:
            warnings.append(f"{extra_inference['failed_reps']} bootstrap repetitions had insufficient observations or rank and were excluded; covariance uses {extra_inference['successful_reps']} successful repetitions.")
    elif covariance_name in {"cluster_hc2", "cluster_hc3"} or (covariance_name == "jackknife" and replay.clusters):
        scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
        directory = Path(scratch) if scratch else None
        groups = 0
        singular_groups = 0
        with _GroupStore(k, directory) as store:
            for block in replay.batches():
                _, w, resid, q, _ = residual_block(block)
                store.add(block.frame, q * w.sqrt()[:, None], q * (w * resid)[:, None], replay.clusters)
            for score, gram in store.items():
                groups += 1
                annihilator = torch.eye(k, dtype=torch.float64) - gram
                values, vectors = torch.linalg.eigh((annihilator + annihilator.T) / 2)
                valid = values > 100 * torch.finfo(torch.float64).eps * k
                if covariance_name == "jackknife":
                    if not bool(valid.all()):
                        raise AnalysisError("jackknife_rank_deficient", "Deleting a cluster removes an identified coefficient; jackknife covariance is undefined.")
                    deleted = torch.linalg.solve(annihilator, score)
                    deleted_sum.add(deleted)
                    deleted_cross.add(torch.outer(deleted, deleted))
                else:
                    singular_groups += int(not bool(valid.all()))
                    power = .5 if covariance_name == "cluster_hc2" else 1.
                    inverse = torch.where(valid, values.clamp_min(torch.finfo(torch.float64).tiny).pow(-power), 0)
                    adjusted_score = vectors @ (inverse * (vectors.T @ score))
                    meat.add(torch.outer(adjusted_score, adjusted_score))
            if groups < 2:
                raise AnalysisError("insufficient_clusters", "Cluster inference requires at least two groups.")
            cluster_diagnostics.append({"columns": replay.clusters, "groups": groups, **store.diagnostics})
            if adjustment is not None:
                controls = adjustment.controls(store)
        group_counts[(replay.clusters[0],)] = groups
        df_inference = groups - 1
        if covariance_name == "jackknife":
            average = deleted_sum.value / groups
            centered = deleted_cross.value - torch.outer(deleted_sum.value, deleted_sum.value) / groups
            covariance_normalized = inverse_r @ (centered * ((groups - 1) / groups)) @ inverse_r.T
            extra_inference = {"scheme": "cluster", "reps": groups, "successful_reps": groups,
                               "failed_reps": 0, "resampling_units": groups, "cluster_count": groups,
                               "bias_estimate": -(groups - 1) * (transform @ inverse_r @ average)}
        else:
            covariance_normalized = inverse_r @ meat.value @ inverse_r.T
            extra_inference = {"small_sample_correction": 1., "partial_annihilator": "Moore-Penrose low-rank inverse",
                               "singular_cluster_directions": singular_groups,
                               "cluster_counts": [groups], "cluster_count": groups, "cluster_dimensions": 1}
            if singular_groups:
                warnings.append("Cluster HC2/HC3 used a Moore-Penrose inverse for unit-leverage directions.")
            warnings.append("Cluster HC2/HC3 inference uses G-1; Bell-McCaffrey dfadjust and Hansen corrections are not enabled.")
    elif covariance_name == "jackknife":
        units = nobs if spec.weight_type == "fweight" else moments.n
        df_inference = units - 1
        average = deleted_sum.value / units
        centered = deleted_cross.value - torch.outer(deleted_sum.value, deleted_sum.value) / units
        covariance_normalized = inverse_r @ (centered * ((units - 1) / units)) @ inverse_r.T
        extra_inference = {"scheme": "delete-one", "reps": units, "successful_reps": units,
                           "failed_reps": 0, "resampling_units": units,
                           "bias_estimate": -(units - 1) * (transform @ inverse_r @ average),
                           "frequency_count_resampling": spec.weight_type == "fweight", "cluster_count": None}
    elif covariance_name == "cluster":
        scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
        directory = Path(scratch) if scratch else None
        subset_meat = _CompensatedSum((k, k))
        for count in range(1, len(replay.clusters) + 1):
            for subset in combinations(replay.clusters, count):
                with ClusterAccumulator(k, directory) as accumulator:
                    for block in replay.batches():
                        _, w, resid, q, _ = residual_block(block)
                        _add_cluster(accumulator, block.frame, q * (w * resid)[:, None], subset)
                    subset_value, groups = accumulator.finish()
                    group_counts[subset] = groups
                    factor = groups / (groups - 1) * (nobs - 1) / df_resid
                    subset_meat.add(subset_value * factor * (1 if count % 2 else -1))
                    cluster_diagnostics.append({"columns": list(subset), "groups": groups,
                                                **accumulator.diagnostics})
        df_inference = min(group_counts[(name,)] - 1 for name in replay.clusters)
        covariance_normalized = inverse_r @ subset_meat.value @ inverse_r.T
        marginal = [group_counts[(name,)] for name in replay.clusters]
        extra_inference = {"cluster_counts": marginal, "cluster_count": min(marginal),
                           "cluster_dimensions": len(marginal), "df_adjustment": "G-1",
                           "combination_counts": [{"dimensions": [replay.clusters.index(name) for name in subset],
                                                   "count": groups,
                                                   "correction": groups / (groups - 1) * (nobs - 1) / df_resid}
                                                  for subset, groups in group_counts.items()]}
        if df_inference < 30:
            warnings.append("Few clusters: cluster-robust inference can be unreliable.")
    else:
        covariance_normalized = inverse_r @ (hac.meat.value if hac is not None else meat.value) @ inverse_r.T
        if covariance_name in {"HC1", "hac"}:
            covariance_normalized *= nobs / df_resid
    covariance = transform @ covariance_normalized @ transform.T
    covariance = (covariance + covariance.T) / 2
    prediction_factor = None
    if covariance_name == "cluster" and len(replay.clusters) > 1:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        if bool((eigenvalues < 0).any()):
            prediction_factor = eigenvectors * eigenvalues.clamp_min(0).sqrt()[None, :]
            covariance = (eigenvectors * eigenvalues.clamp_min(0)[None, :]) @ eigenvectors.T
            covariance = (covariance + covariance.T) / 2
            extra_inference["psd_projection"] = "raw coefficient coordinates; not basis invariant"
            warnings.append("Multiway cluster covariance was not positive semidefinite; negative eigenvalues were set to zero.")
    if not bool(torch.isfinite(params).all()) or not bool(torch.isfinite(covariance).all()) or not math.isfinite(residual_ss):
        raise AnalysisError("numerical_overflow", "The model result is outside representable float64 units.")

    def prediction_basis(x):
        values = x.to(dtype=torch.float64)
        coefficient = values[:, kept_intercept:kept_intercept + 1] if kept_intercept is not None else 0.
        return _normalize(values - coefficient * moments.anchor[kept].to(values.device),
                          magnitude[kept].to(values.device), coefficient * centers[kept].to(values.device),
                          rms[kept].to(values.device))

    def contrast_covariance(rows):
        if prediction_factor is not None:
            values = rows.to(dtype=torch.float64)
            factor = values @ prediction_factor.to(values.device)
            return factor @ factor.T
        z = prediction_basis(rows)
        value = z @ covariance_normalized.to(z.device) @ z.T
        return (value + value.T) / 2

    def prediction_variance(x):
        if prediction_factor is not None:
            values = x.to(dtype=torch.float64)
            return (values @ prediction_factor.to(values.device)).square().sum(dim=1)
        z = prediction_basis(x)
        # Mapping covariance to large raw units can erase the small variance
        # of a prediction through cancellation. Keep its original small basis.
        return (z @ covariance_normalized.to(z.device) * z).sum(dim=1)

    def prediction_leverage(x):
        z = prediction_basis(x)
        q = torch.linalg.solve_triangular(r.T.to(z.device), z.T, upper=False).T
        return q.square().sum(dim=1)

    def prediction_fitted(x):
        z = prediction_basis(x)
        fitted = (z @ theta.to(z.device)) * outcome_magnitude[0].to(z.device)
        if kept_intercept is not None:
            offset = (moments.anchor[-1] + outcome_center[0] * outcome_magnitude[0]).to(z.device)
            fitted = fitted + x[:, kept_intercept].to(dtype=torch.float64, device=z.device) * offset
        return fitted

    if adjustment is not None:
        controls = controls if controls is not None else adjustment.controls()
        extra_inference.update(dfadjust=True, hansen=bool(spec.options.get("hansen")),
                               coefficient_df=[control["df"] for control in controls],
                               coefficient_scale=[control["scale"] for control in controls],
                               df_adjustment="Bell-McCaffrey/Satterthwaite contrast-specific",
                               scaling_semantics="scale multiplies t statistics and divides confidence interval width",
                               adjustment_source="Hansen 2025 A6/A7" if spec.options.get("hansen") else "Stata rregress adjusted degrees of freedom")
        warnings = [warning for warning in warnings if "Bell-McCaffrey dfadjust and Hansen corrections are not enabled" not in warning]
        warnings.append("Adjusted degrees of freedom apply to individual contrasts; joint model tests use conventional residual/cluster degrees of freedom.")
    r_squared = 1 - residual_ss / total_ss if total_ss > 0 else math.nan
    adjusted = 1 - (1 - r_squared) * (nobs - int(spec.intercept)) / df_resid
    ll = -.5 * nobs * (math.log(2 * math.pi) + 1 + math.log(residual_ss / nobs)) if residual_ss > 0 else math.inf
    if spec.weight_type == "aweight":
        ll += .5 * float(log_weights.value)
    kind = {"aweight": "aw", "fweight": "fw", "pweight": "pw", "iweight": "iw"}.get(spec.weight_type)
    likelihood = ("frequency_expanded_gaussian" if kind == "fw" else
                  "gaussian_wls" if kind in {None, "aw"} else "weighted_gaussian_pseudo_likelihood")
    slopes = [index for index in range(k) if index != kept_intercept]
    f_value, f_df_num = math.nan, k - int(kept_intercept is not None)
    if slopes:
        slope_covariance = covariance[slopes][:, slopes]
        f_df_num = int(torch.linalg.matrix_rank(slope_covariance))
        if f_df_num:
            slope_params = params[slopes]
            f_value = float(slope_params @ torch.linalg.pinv(slope_covariance, hermitian=True) @ slope_params / f_df_num)
    singular_values = torch.linalg.svdvals(r)
    metrics = {"r_squared": r_squared, "r_squared_adj": adjusted, "adjusted_r_squared": adjusted,
               "rmse": math.sqrt(sigma2), "root_mse": math.sqrt(sigma2),
               "prediction_rmse": math.sqrt(residual_ss / nobs),
               "sse": residual_ss, "rss": residual_ss, "tss": total_ss,
               "mss": total_ss - residual_ss, "sigma2": sigma2,
               "ss_resid": residual_ss, "ss_total": total_ss, "ss_model": total_ss - residual_ss,
               "nobs": nobs, "effective_nobs": nobs, "physical_nobs": moments.n, "nparams": k,
               "df_model": k - int(kept_intercept is not None), "df_resid": df_resid,
               "log_likelihood": ll, "likelihood_convention": likelihood, "weight_type": kind,
               "mean_y": float(moments.anchor[-1] + moments.mean[-1] * outcome_magnitude[0]),
               "sum_weights": moments.importance_sum.total() if moments.importance_sum is not None else float(nobs),
               "covariance": covariance_name,
               "condition_number": float(singular_values[0] / singular_values[-1]),
               "f_statistic": f_value, "f_df_num": f_df_num, "f_df_denom": df_inference}
    packed = {name: torch.cat(parts) if parts else torch.empty((0, k) if name == "x" else (0,),
                                                              dtype=torch.int64 if name == "positions" else torch.float64)
              for name, parts in sample.items()}
    metadata = {"algorithm": "weighted centered/scaled balanced TSQR",
                "passes": replay.passes, "reader_batch_rows": replay.reader_rows,
                "batch_rows": replay.rows, "row_limit": None,
                "numerical_batch_rows": replay.rows, "maximum_encoded_rows": replay.maximum_rows,
                "working_memory_budget_bytes": _WORKING_BYTES,
                "full_sample_positions_retained": False, "physical_nobs": moments.n,
                "weight_normalization": "raw" if spec.weight_type in {"fweight", "iweight"} and covariance_name == "nonrobust"
                else "raw" if spec.weight_type == "fweight" else "sum to physical rows",
                "tsqr_depth": tree.depth, "rank_diagnostic": "ordered reorthogonalized normalized TSQR columns",
                **replay.baseline}
    provenance = {"solver": "torch_tsqr", "sample_positions_omitted": True,
                  "sample_position_count": replay.baseline["used"],
                  "sample_positions_hash": replay.baseline["positions_hash"]}
    if cluster_diagnostics:
        metadata["cluster_subsets"] = cluster_diagnostics
    if hac is not None:
        metadata.update({"hac_lags": lags, "hac_kernel": hac.kernel,
                         "hac_time": "physical integer periods; gaps preserved", **selection})
        extra_inference.update({"kernel": hac.kernel, "lags": lags, "bandwidth": lags + 1,
                                "time_gaps": True, "small_sample_correction": nobs / df_resid,
                                "noncompact_support": isinstance(hac, _QuadraticSpectral), **selection})
        if isinstance(hac, _QuadraticSpectral):
            metadata.update(hac.diagnostics)
    return {"params": params, "covariance": covariance, "bread": bread,
            "terms": terms, "omitted_terms": omitted, "kept_indices": kept,
            "nobs": nobs, "df_resid": df_resid, "df_inference": df_inference,
            "sigma2": sigma2, "metrics": metrics, "warnings": warnings,
            "inference": {"distribution": "t", "df": df_inference,
                          "covariance": covariance_name, "weight_type": spec.weight_type,
                          "standard_errors": covariance.diagonal().clamp_min(0).sqrt(),
                          "dfadjust": False, "hansen": False,
                          "model_test": "F" if math.isfinite(df_inference) else "Wald chi-square",
                          "f_statistic": f_value, "f_df_num": f_df_num, "f_df_denom": df_inference,
                          **extra_inference},
            "x": packed["x"], "y": packed["y"], "weights": packed["weights"],
            "resid": packed["resid"], "fitted": packed["fitted"], "leverage": packed["leverage"],
            "sample_positions": packed["positions"], "samples_only": True,
            "nobs_original": replay.baseline["original"],
            "dropped_rows": replay.baseline["original"] - replay.baseline["used"],
            "data_hash": replay.baseline["data_hash"], "contrast_inference": adjustment,
            "state": {"residual_block": residual_block, "normalized_block": normalized,
                      "weight_scale": multiplier / moments.weight_max,
                      "prediction_variance": prediction_variance,
                      "prediction_leverage": prediction_leverage,
                      "prediction_fitted": prediction_fitted,
                      "contrast_covariance": contrast_covariance,
                      "contrast_basis": prediction_basis},
            "provenance": provenance, "streaming": metadata, "source": source,
            "design": design, "replay": replay}


def fit_streaming(spec, source: Dataset, *, design=None, device="auto"):
    """Fit OLS with bounded replay and guaranteed ordinary-exit spill cleanup."""
    with ExitStack() as resources, execution_scope(device) as execution:
        estimated = _fit_streaming(spec, source, design=design, resources=resources)
        estimated["provenance"].update(execution.metadata())
        return estimated
