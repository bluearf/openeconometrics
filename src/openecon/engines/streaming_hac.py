"""Exact HAC scores with disk sorting and bounded lag/pair state.

Scores are sorted by unit/period in an owned SQLite file. Only one read block
and the last L rows remain in RAM; gaps use actual period distances. This is
not a per-block HAC estimate, and boundary pairs are included once.
"""
from __future__ import annotations

from array import array
from collections.abc import Sequence
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import torch

from openecon.resources import plan_workspace
from .contracts import KernelError
from .covariance import _kernel, kernel_weights
from .streaming_ols import _CompensatedSum


class HACAccumulator:
    @staticmethod
    def workspace_buffers(width: int, lags: int, kernel: str = "bartlett") -> dict[str, int]:
        """Validate and report exact planned buffers before any allocation."""
        if type(width) is not int or not 1 <= width <= 1024:
            raise KernelError("invalid_scores", "HAC score width must be from1 to1024.")
        # Validate before allocating a lag-sized array.
        if type(lags) is not int or lags < 0:
            raise KernelError("invalid_lags", "HAC lags must be a nonnegative integer.")
        unbounded = kernel == "quadratic_spectral"
        # Validate the kernel without allocating a bandwidth-sized tensor. QS
        # bandwidth is not a truncation distance and never needs a lag tail.
        kernel_weights(0, kernel)
        tail = 0 if unbounded else lags
        return {
            "lag_weights": 8*tail,
            "score_blocks_and_lag_tail": 64*(4096+tail)*width,
            "period_unit_and_record_buffers": 256*(4096+tail),
            "all_lag_pair_tiles": 64*512**2 if unbounded and lags else 0,
            "moment_matrix": 64*width*width,
            "sqlite_cache": 2*1024**2,
        }

    def __init__(self, width: int, lags: int, kernel: str = "bartlett", *,
                 budget_bytes: int | None = None):
        self.plan = plan_workspace("disk-sorted HAC scores", self.workspace_buffers(width, lags, kernel), budget_bytes=budget_bytes)
        self.width, self.lags = width, lags
        unbounded = kernel == "quadratic_spectral"
        self.weights = kernel_weights(lags, kernel, count=0 if unbounded else lags)
        self._white = _CompensatedSum((width, width)) if lags == 0 else None
        self.kernel = kernel
        self.datetime = None
        self._connection = self._scratch = None
        self._closed = self._finished = False
        self.rows = 0
        try:
            self._scratch = tempfile.TemporaryDirectory(prefix="openecon-hac-")
            self.path = Path(self._scratch.name)/"scores.sqlite3"
            self._connection = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-2048", "temp_store=FILE", "mmap_size=0"):
                self._connection.execute("PRAGMA "+pragma)
            self._connection.execute("CREATE TABLE scores (panel BLOB, t INTEGER, score BLOB, PRIMARY KEY(panel,t)) WITHOUT ROWID")
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise KernelError("hac_spill_failed", "HAC needs writable temporary storage and enough disk space.") from exc

    def add(self, scores: torch.Tensor, positions: torch.Tensor, *,
            time: pd.Series | None = None, units: Sequence[bytes] | None = None):
        if self._closed or self._finished:
            raise KernelError("invalid_scores", "HAC accumulator is already closed or finished.")
        n = len(scores)
        if scores.shape != (n, self.width) or not bool(torch.isfinite(scores).all()) or len(positions) != n:
            raise KernelError("invalid_scores", "HAC needs finite aligned score rows.")
        date = time is not None and pd.api.types.is_datetime64_any_dtype(time.dtype)
        if self.datetime is not None and self.datetime != date:
            raise KernelError("source_changed", "HAC time type changed between batches.")
        self.datetime = date
        if time is None:
            # With no declared time dense HAC uses consecutive retained rows,
            # including after missing/zero-weight observations are dropped.
            periods = list(range(self.rows, self.rows+n))
        elif date:
            periods = time.dt.as_unit("ns").astype("int64").tolist()
        elif pd.api.types.is_integer_dtype(time.dtype) and not pd.api.types.is_bool_dtype(time.dtype):
            periods = [int(value) for value in time]
        else:
            values = torch.tensor(time.tolist(), dtype=torch.float64)
            if not bool(torch.isfinite(values).all()) or bool((values != values.round()).any()):
                raise KernelError("invalid_time", "HAC time must contain integer periods or datetimes.")
            periods = [int(value) for value in values.tolist()]
        if any(value < -(1<<63) or value >= 1<<63 for value in periods):
            raise KernelError("invalid_time", "HAC periods must fit exact signed64-bit integers.")
        keys = [b""]*n if units is None else units
        if len(keys) != n:
            raise KernelError("invalid_time", "HAC unit labels are not aligned.")
        if self.lags == 0:
            # Native lags=0 is White: repeated timestamps, ordering and
            # span cannot change it and need no source-sized spill table.
            self._white.add(scores.T@scores)
            self.rows += n
            return
        try:
            # All Python record objects are block-sized, never source-sized.
            records = ((key, period, array("d", row).tobytes())
                       for key, period, row in zip(keys, periods, scores.tolist(), strict=True))
            self._connection.executemany("INSERT INTO scores VALUES(?,?,?)", records)
            self.rows += n
        except sqlite3.IntegrityError as exc:
            raise KernelError("duplicate_time", "Periods must be distinct within each HAC series.") from exc
        except sqlite3.Error as exc:
            raise KernelError("hac_spill_failed", "Could not persist HAC scores; check free disk space.") from exc

    def finish(self):
        if self._closed or self._finished:
            raise KernelError("invalid_scores", "HAC accumulator is already closed or finished.")
        self._finished = True
        try:
            self._connection.commit()
            if self.lags == 0:
                self._pair_count = 0
                return (self._white.value+self._white.value.T)/2
            if not self.datetime:
                spans = self._connection.execute("SELECT MIN(t), MAX(t) FROM scores GROUP BY panel")
                if any(int(hi)-int(lo) >= 1<<62 for lo, hi in spans):
                    raise KernelError("invalid_time", "HAC period span is too wide; recode periods.")
                query = "SELECT panel,t,score FROM scores ORDER BY panel,t"
            else:
                # Dense semantics rank all distinct dates globally, even with panels.
                query = "WITH dates AS (SELECT t, DENSE_RANK() OVER(ORDER BY t)-1 AS period FROM (SELECT DISTINCT t FROM scores)) SELECT panel,period,score FROM scores JOIN dates USING(t) ORDER BY panel,period"
            if self.kernel == "quadratic_spectral":
                return self._finish_quadratic_spectral(query)
            cursor = self._connection.execute(query)
            total = _CompensatedSum((self.width, self.width))
            tail = torch.empty((0, self.width), dtype=torch.float64)
            tail_keys, tail_times = [], []
            while records := cursor.fetchmany(4096):
                block = torch.stack([torch.frombuffer(bytearray(row[2]), dtype=torch.float64) for row in records])
                values = torch.cat((tail, block))
                keys = tail_keys+[row[0] for row in records]
                times = tail_times+[int(row[1]) for row in records]
                start = len(tail)
                total.add(block.T@block)
                for offset in range(1, min(self.lags, len(values)-1)+1):
                    begin = max(offset, start)
                    keep = torch.tensor([keys[i] == keys[i-offset] and 0 < times[i]-times[i-offset] <= self.lags
                                         for i in range(begin, len(values))], dtype=torch.bool)
                    if not bool(keep.any()):
                        continue
                    distance = torch.tensor([times[i]-times[i-offset] if keys[i] == keys[i-offset] else 0
                                             for i in range(begin, len(values))], dtype=torch.int64)[keep]
                    left = values[begin:][keep]
                    right = values[begin-offset:len(values)-offset][keep]*self.weights[distance-1, None]
                    cross = left.T@right
                    total.add(cross+cross.T)
                retained = min(self.lags, len(values))
                tail = values[-retained:].clone() if retained else values[:0]
                tail_keys = keys[-retained:] if retained else []
                tail_times = times[-retained:] if retained else []
            return (total.value+total.value.T)/2
        except sqlite3.Error as exc:
            raise KernelError("hac_spill_failed", "Could not sort HAC scores; check free disk space.") from exc

    def _finish_quadratic_spectral(self, query):
        """Every positive within-series lag, with bounded exact pair tiles.

        Disk ordering also preserves global datetime ranks. Unlike a finite
        window, this can require quadratic work; preflight it before reading
        any score tiles rather than silently cutting off the kernel's tail.
        """
        counts = self._connection.execute("SELECT COUNT(*) FROM scores GROUP BY panel")
        pairs = sum(count*(count-1)//2 for (count,) in counts)
        self._pair_count = pairs if self.lags else 0
        if self._pair_count*self.width > 4_000_000_000:
            raise KernelError("hac_work_limit", "Exact quadratic spectral HAC exceeds the planned all-lag work budget; use a finite-support HAC kernel or a smaller series.")
        self._connection.execute("CREATE TEMP TABLE ordered_scores(panel BLOB, t INTEGER, score BLOB)")
        self._connection.execute("INSERT INTO ordered_scores "+query)
        self._connection.execute("CREATE INDEX ordered_score_key ON ordered_scores(panel,t)")
        total = _CompensatedSum((self.width, self.width))
        panels = self._connection.execute("SELECT DISTINCT panel FROM ordered_scores ORDER BY panel")
        for (panel,) in panels:
            outer = self._connection.execute("SELECT t,score FROM ordered_scores WHERE panel=? ORDER BY t", (panel,))
            while records := outer.fetchmany(512):
                times, block = self._decode_pairs(records)
                total.add(block.T@block)
                if self.lags == 0:
                    continue  # native dense lags=0 is White, including QS
                inner = self._connection.execute("SELECT t,score FROM ordered_scores WHERE panel=? AND t>=? ORDER BY t", (panel, int(times[0])))
                while future := inner.fetchmany(512):
                    future_times, future_block = self._decode_pairs(future)
                    distance = future_times[None, :]-times[:, None]
                    weights = _kernel(distance.to(torch.float64)/(self.lags+1), "quadratic_spectral")
                    weights = torch.where(distance > 0, weights, 0.)
                    cross = block.T@(weights@future_block)
                    total.add(cross+cross.T)
                inner.close()
            outer.close()
        panels.close()
        return (total.value+total.value.T)/2

    @staticmethod
    def _decode_pairs(records):
        times = torch.tensor([row[0] for row in records], dtype=torch.int64)
        scores = torch.stack([torch.frombuffer(bytearray(row[1]), dtype=torch.float64) for row in records])
        return times, scores

    @property
    def diagnostics(self):
        return {"hac_aggregation": "White scores without temporal pairing" if self.lags == 0 else "sqlite sorted scores with exact bounded all-lag pair tiles" if self.kernel == "quadratic_spectral" else "sqlite sorted scores with exact cross-block lag pairs",
                "hac_rows": self.rows, "hac_lags": self.lags, "hac_kernel": self.kernel,
                "hac_resource_plan": self.plan.record(),
                "hac_pair_count": getattr(self, "_pair_count", None),
                "hac_scratch_bytes": self.path.stat().st_size if self.path.exists() else 0}

    def close(self):
        self._closed = True
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
