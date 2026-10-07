"""Owned disk ordering and exact ordered/patterned native GEE equations.

AR(1) uses SQL neighbouring rows and bounded tensors, even for long panels.
Patterned structures inherently need a T-by-T correlation matrix. Its full
moment/factor and single-panel solve workspace is reserved before fetching a
panel; no total-sample or all-group numerical arrays are constructed.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum
from openecon.streaming_design import encode_cluster_labels

from .replay_sample import ReplaySample
from .streaming_linear import _finite, _period_keys, _scratch_directory

PATTERNED = frozenset({"stationary", "nonstationary", "unstructured"})
ORDERED = PATTERNED | {"ar1"}


class OrderedPanels:
    def __init__(self, sample):
        self.sample = sample
        self.scratch = self.db = None
        self.maximum_panel_rows = 0
        parent = _scratch_directory()
        estimate = sample.nrows * (128 * (len(sample.designs["mean"].terms) + 1) + 512)
        if estimate + 64 * 1024**2 > shutil.disk_usage(parent or tempfile.gettempdir()).free:
            raise AnalysisError("insufficient_scratch_space", "Ordered GEE needs more free owned temporary disk space.")
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-gee-", dir=parent)
            path = Path(self.scratch.name) / "panels.sqlite3"
            self.db = sqlite3.connect(path)
            for setting in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-2048", "temp_store=FILE", "mmap_size=0"):
                self.db.execute("PRAGMA " + setting)
            self.db.execute("CREATE TABLE raw(position INTEGER PRIMARY KEY,key BLOB,t INTEGER)")
            self.db.execute("CREATE UNIQUE INDEX panel_time ON raw(key,t)")
            path.chmod(0o600)
        except BaseException:
            self.close()
            raise

    def seed(self, notes, corr, force):
        dates = None
        for batch in self.sample.batches():
            values = batch.frame[self.sample.spec.time]
            is_date = pd.api.types.is_datetime64_any_dtype(values.dtype)
            if dates is not None and is_date != dates:
                raise AnalysisError("invalid_time", "The GEE time representation must be consistent across batches.")
            dates = is_date
            keys = encode_cluster_labels(batch.frame[self.sample.spec.panel])
            times = [int(value) for value in _period_keys(values)]
            try:
                self.db.executemany("INSERT INTO raw VALUES(?,?,?)", zip(batch.positions.tolist(), keys, times, strict=True))
            except sqlite3.IntegrityError as error:
                raise AnalysisError("repeated_time_values", "Time values are repeated within a GEE panel.") from error
        self.db.execute("CREATE TABLE metadata AS SELECT key,COUNT(*) AS n FROM raw GROUP BY key")
        self.db.execute("CREATE UNIQUE INDEX metadata_key ON metadata(key)")
        # Datetimes use the global dense rank, exactly as ModelFrame.time_index.
        if dates:
            self.db.execute("CREATE TABLE periods AS SELECT t,ROW_NUMBER() OVER(ORDER BY t)-1 AS rank FROM(SELECT DISTINCT t FROM raw)")
            self.db.execute("CREATE UNIQUE INDEX period_time ON periods(t)")
            period = "p.rank"
            join = " JOIN periods p USING(t)"
            notes.warn(f"Datetime column '{self.sample.spec.time}' is treated as consecutive periods in sorted order; gaps between dates are not detected.")
        else:
            period, join = "r.t", ""
        bad = self.db.execute("SELECT 1 FROM(SELECT " + period + " AS t,LAG(" + period + ") OVER(PARTITION BY key ORDER BY r.t) AS prev FROM raw r" + join + ") WHERE prev IS NOT NULL AND t-prev!=1 LIMIT 1").fetchone()
        if bad and not force:
            raise AnalysisError("unequal_spacing", f"corr='{corr}' needs equally spaced observations; pass force=True to treat each panel's observations as consecutive.")
        self.db.commit()

    def retained(self, frame, lag):
        keys = encode_cluster_labels(frame[self.sample.spec.panel])
        sizes = {}
        for key in set(keys):
            record = self.db.execute("SELECT n FROM metadata WHERE key=?", (key,)).fetchone()
            sizes[key] = record[0] if record else 0
        return torch.tensor([sizes[key] > lag for key in keys], dtype=torch.bool)

    def select(self, spec, source, batch_rows, notes, corr, order):
        lag = 1 if corr == "ar1" else order if corr in {"stationary", "nonstationary"} else 0
        short, observations = self.db.execute("SELECT COUNT(*),COALESCE(SUM(n),0) FROM metadata WHERE n<=?", (lag,)).fetchone()
        if short:
            total = self.db.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
            if short == total:
                raise AnalysisError("insufficient_panel_length", f"corr='{corr}' needs panels with more than {lag} observations.")
            sample = ReplaySample(spec, source, batch_rows=batch_rows, row_filter=lambda frame: self.retained(frame, lag))
            sample.add_design("mean")
            sample.prepare()
            if sample.baseline["data_hash"] != self.sample.baseline["data_hash"]:
                raise AnalysisError("source_changed", "The projected GEE inputs changed during short-panel selection.")
            # The re-prepared sample has different retained positions, but its
            # earlier discovery/rank/ordering passes were real work too.
            # Preserve measured peaks and completed full-source pass counts.
            sample.passes += self.sample.passes
            sample.maximum_rows = max(sample.maximum_rows, self.sample.maximum_rows)
            sample.actual_numeric_peak_rows = max(sample.actual_numeric_peak_rows, self.sample.actual_numeric_peak_rows)
            notes.warn(f"note: {short} panel(s) ({observations} obs) dropped because of too few observations (corr='{corr}' needs more than {lag} per panel).")
            self.sample = sample
            self.db.execute("DELETE FROM raw WHERE key IN(SELECT key FROM metadata WHERE n<=?)", (lag,))
            self.db.execute("DELETE FROM metadata WHERE n<=?", (lag,))
        self.db.execute("CREATE TABLE ordering AS SELECT position,key,ROW_NUMBER() OVER(PARTITION BY key ORDER BY t)-1 AS seq FROM raw")
        self.db.execute("CREATE UNIQUE INDEX ordered_position ON ordering(position)")
        self.db.execute("CREATE UNIQUE INDEX ordered_panel_seq ON ordering(key,seq)")
        groups, minimum, maximum, count = self.db.execute("SELECT COUNT(*),MIN(n),MAX(n),SUM(n) FROM metadata").fetchone()
        self.groups, self.longest = int(groups), int(maximum)
        self.structure = {"t_min": float(minimum), "t_avg": count / groups, "t_max": float(maximum)}
        self.db.commit()
        return self.sample

    def initialize(self, width):
        self.width = width
        self.names = [*(f"x{i}" for i in range(width)), "r"]
        self.db.execute("CREATE TABLE moments(position INTEGER PRIMARY KEY," + ",".join(name + " REAL" for name in self.names) + ",observed REAL,fitted REAL)")

    def rows(self, query, parameters=()):
        cursor = self.db.execute(query, parameters)
        try:
            while rows := cursor.fetchmany(self.sample.rows):
                yield rows
        finally:
            cursor.close()

    def panels(self):
        # The joint workspace reservation in the adapter MUST precede this
        # fetch. metadata is disk streamed; at most one panel is resident.
        cursor = self.db.execute("SELECT key,n FROM metadata ORDER BY key")
        try:
            while group := cursor.fetchone():
                key, size = group
                query = "SELECT " + ",".join("m." + name for name in self.names) + " FROM ordering o JOIN moments m USING(position) WHERE key=? ORDER BY seq"
                block = []
                for rows in self.rows(query, (key,)):
                    block.append(torch.tensor(rows, dtype=torch.float64))
                values = torch.cat(block)
                if len(values) != size:
                    raise AnalysisError("source_changed", "The GEE ordered panel differs from its selected metadata.")
                self.maximum_panel_rows = max(self.maximum_panel_rows, size)
                self.sample.actual_numeric_peak_rows = max(self.sample.actual_numeric_peak_rows, size)
                yield key, values
        finally:
            cursor.close()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


class OrderedCorrelation:
    def __init__(self, panels, kind, order):
        self.panels, self.kind, self.order = panels, kind, order
        self.alpha = torch.empty(0, dtype=torch.float64)
        self.matrix = self.chol = None

    def _estimate(self):
        store, periods = self.panels, self.panels.longest
        variance = _CompensatedSum(())
        query = "SELECT SUM(m.r*m.r)/n FROM moments m JOIN ordering o USING(position) JOIN metadata USING(key) GROUP BY key"
        for rows in store.rows(query):
            variance.add(torch.tensor([r[0] for r in rows], dtype=torch.float64).sum())
        if not float(variance.value) > 0:
            raise KernelError("perfect_fit", "The GEE Pearson residuals are all zero.")
        if self.kind in {"ar1", "stationary"}:
            estimates = []
            for lag in range(1, (1 if self.kind == "ar1" else self.order) + 1):
                value = _CompensatedSum(())
                query = "SELECT m.r*q.r/n FROM ordering o JOIN moments m USING(position) JOIN metadata USING(key) JOIN ordering p ON p.key=o.key AND p.seq=o.seq+? JOIN moments q ON q.position=p.position"
                for rows in store.rows(query, (lag,)):
                    value.add(torch.tensor([r[0] for r in rows], dtype=torch.float64).sum())
                estimates.append(value.value / variance.value)
            self.alpha = torch.stack(estimates)
            if self.kind == "ar1":
                if not -1 < float(self.alpha[0]) < 1:
                    raise KernelError("working_correlation_not_pd", "The AR(1) correlation is outside (-1, 1).")
                return
            matrix = torch.eye(periods, dtype=torch.float64)
            for lag, value in enumerate(self.alpha, 1):
                indices = torch.arange(periods - lag)
                matrix[indices, indices + lag] = value
                matrix[indices + lag, indices] = value
        else:
            cross = _CompensatedSum((periods, periods))
            counts = torch.zeros((periods, periods), dtype=torch.float64)
            for _, values in store.panels():
                size, r = len(values), values[:, -1]
                # Compensated accumulator without a second full T-by-T value.
                current = torch.outer(r, r) - cross.correction[:size, :size]
                total = cross.value[:size, :size] + current
                cross.correction[:size, :size] = (total - cross.value[:size, :size]) - current
                cross.value[:size, :size] = total
                counts[:size, :size] += 1
            matrix = cross.value / counts.clamp_min(1) / (variance.value / store.groups)
            matrix.diagonal().fill_(1)
            if self.kind == "nonstationary":
                for row in range(periods):
                    matrix[row, :max(0, row - self.order)] = 0
                    matrix[row, min(periods, row + self.order + 1):] = 0
            self.alpha = torch.empty(0, dtype=torch.float64)
        self.chol, info = torch.linalg.cholesky_ex(matrix)
        if int(info) != 0:
            raise KernelError("working_correlation_not_pd", "The estimated working correlation is not positive definite; choose a simpler structure or lower order.")
        self.matrix = matrix

    def moments(self, eq, beta, *, final=False):
        store, width = self.panels, len(beta)
        store.db.execute("DELETE FROM moments")
        pearson, deviance = _CompensatedSum(()), _CompensatedSum(())
        insert = "INSERT INTO moments VALUES(" + ",".join("?" for _ in range(width + 4)) + ")"
        for batch in eq.sample.batches():
            y, mu, residual, x = eq.at(batch, beta)
            pearson.add(residual.square().sum())
            deviance.add(eq.family.unit_deviance(y, mu).sum())
            values = torch.cat((x, residual[:, None], batch.numeric(eq.sample.spec.outcome)[:, None], (eq.mean + eq.scale * mu)[:, None]), 1)
            store.db.executemany(insert, ((int(position), *row) for position, row in zip(batch.positions.tolist(), values.tolist(), strict=True)))
        store.db.commit()
        if not float(pearson.value) > 1e-24 * eq.signal:
            raise KernelError("perfect_fit", "All GEE Pearson residuals are zero; working correlation and scale are not identified.")
        self._estimate()
        bread, gradient = _CompensatedSum((width, width)), _CompensatedSum((width,))
        meat, predictions = torch.zeros((width, width), dtype=torch.float64), []
        with ExitStack() as stack:
            accumulator = None
            if final:
                accumulator = ClusterAccumulator(width, scratch_directory=_scratch_directory())
                stack.callback(accumulator.close)
            if self.kind == "ar1":
                columns = [*("m." + name for name in store.names), *("COALESCE(a." + name + ",0)" for name in store.names), *("COALESCE(b." + name + ",0)" for name in store.names)]
                query = "SELECT o.key,o.seq,n," + ",".join(columns) + " FROM ordering o JOIN moments m USING(position) JOIN metadata USING(key) LEFT JOIN ordering prev ON prev.key=o.key AND prev.seq=o.seq-1 LEFT JOIN moments a ON a.position=prev.position LEFT JOIN ordering next ON next.key=o.key AND next.seq=o.seq+1 LEFT JOIN moments b ON b.position=next.position ORDER BY o.key,o.seq"
                alpha = self.alpha[0]
                for rows in store.rows(query):
                    keys = [row[0] for row in rows]
                    value = torch.tensor([row[1:] for row in rows], dtype=torch.float64)
                    position, size = value[:, 0], value[:, 1]
                    current, previous, following = value[:, 2:].split(width + 1, 1)
                    interior = ((position > 0) & (position < size - 1)).to(torch.float64)
                    solved = ((1 + alpha.square() * interior)[:, None] * current - alpha * (previous + following)) / (1 - alpha.square())
                    x = current[:, :width]
                    bread.add(x.T @ solved[:, :width])
                    gradient.add(x.T @ solved[:, width])
                    if final:
                        accumulator.add(keys, x * solved[:, width, None])
            else:
                for key, values in store.panels():
                    solved = torch.cholesky_solve(values, self.chol[:len(values), :len(values)])
                    x = values[:, :width]
                    bread.add(x.T @ solved[:, :width])
                    contribution = x.T @ solved[:, width]
                    gradient.add(contribution)
                    if final:
                        accumulator.add([key], contribution[None, :])
            if final:
                meat, groups = accumulator.finish()
                if groups != store.groups:
                    raise AnalysisError("source_changed", "The GEE final score panels changed.")
                cursor = store.db.execute("SELECT position,observed,fitted FROM moments ORDER BY position LIMIT 400")
                predictions = [{"row": pos, "observed": observed, "fitted": fitted, "residual": observed - fitted} for pos, observed, fitted in cursor]
                cursor.close()
        _finite(bread.value, gradient.value, meat)
        return bread.value, gradient.value, float(pearson.value), float(deviance.value), self.alpha, meat, predictions

    def full_matrix(self):
        if self.matrix is not None:
            return self.matrix
        n = self.panels.longest
        lag = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs()
        return self.alpha[0].pow(lag)
