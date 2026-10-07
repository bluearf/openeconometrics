"""Owned, bounded disk sorting for native time-series estimators.

One source pass writes projected rows into SQLite. All estimation replays read
the immutable sorted snapshot; a final raw-source pass checks that the source
did not change during the fit. Missing rows may only occur at series ends.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import pandas as pd

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.streaming_design import MAX_CATEGORY_BYTES, MAX_PARAMETERS, row_hash_bytes
from . import registry


class OrderedReplay:
    def __init__(self, spec, source):
        self.spec, self.original = spec, source
        self.columns = registry.spec_columns(spec)
        if set(self.columns)-set(source.columns):
            raise AnalysisError("missing_columns", "Time-series source is missing required columns.")
        self.rows = min(65536, max(1, 8*1024**2//max(512, 512*len(self.columns))))
        self.budget = min(128*1024**2, workspace_budget_bytes())
        self.plan = plan_workspace("sorted time-series replay", {
            "raw_reader_and_SQL_records": 256*self.rows*len(self.columns),
            "SQLite_cache_and_decode": 8*1024**2,
            "reporting_sample": 400*128*len(self.columns),
        }, budget_bytes=self.budget)
        self._scratch = self.db = None
        self.passes = 0
        self.dtypes = {}
        self.date = False
        self.category_dtypes = {}
        self.positions_column = "__openecon_physical_row__"
        if self.positions_column in source.columns:
            raise AnalysisError("reserved_column", "Rename the reserved physical-row metadata column before fitting this time-series model.")

    def __enter__(self):
        try:
            self._scratch = tempfile.TemporaryDirectory(prefix="openecon-time-series-",
                dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None)
            self.path = Path(self._scratch.name)/"series.sqlite3"
            self.db = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-4096", "temp_store=FILE", "mmap_size=0"):
                self.db.execute("PRAGMA "+pragma)
            fields = ",".join(f"c{i}" for i in range(len(self.columns)))
            self.db.execute(f"CREATE TABLE rows(pos INTEGER PRIMARY KEY,t INTEGER NOT NULL,valid INTEGER NOT NULL,{fields})")
            digest, original = hashlib.sha256(), 0
            self.original.assert_unchanged()
            for raw in self.original.iter_batches(self.columns, batch_rows=self.rows):
                if len(raw)*128*len(self.columns)+2*int(raw.memory_usage(index=False, deep=True).sum()) > self.budget:
                    raise AnalysisError("batch_too_large", "Raw time-series batch exceeds its replay budget.")
                for name in self.columns:
                    if name not in self.dtypes:
                        self.dtypes[name] = raw[name].dtype
                    elif str(raw[name].dtype) != str(self.dtypes[name]):
                        raise AnalysisError("source_changed", "Time-series column types changed between source batches.")
                digest.update(row_hash_bytes(raw.loc[:, self.columns]))
                missing = raw.isna().any(axis=1)
                if bool(missing.any()) and self.spec.missing == "raise":
                    raise AnalysisError("missing_values", "Time-series inputs contain missing observations; use missing='drop'.")
                periods = self._periods(raw, original)
                values = raw.copy()
                for name in self.columns:
                    if pd.api.types.is_datetime64_any_dtype(values[name].dtype):
                        values[name] = values[name].dt.as_unit("ns").astype("int64")
                sql = "INSERT INTO rows VALUES("+",".join("?" for _ in range(len(self.columns)+3))+")"
                records = ((original+i, period, int(not bad), *[_scalar(value) for value in row])
                           for i, (period, bad, row) in enumerate(zip(periods, missing.tolist(), values.itertuples(index=False, name=None), strict=True)))
                self.db.executemany(sql, records)
                original += len(raw)
            self.original.assert_unchanged()
            self.raw_hash, self.original_count = digest.hexdigest(), original
            self.db.execute("CREATE INDEX chronological ON rows(t,pos)")
            first, last, used = self.db.execute("SELECT MIN(t),MAX(t),COUNT(*) FROM rows WHERE valid=1").fetchone()
            if not used:
                raise AnalysisError("empty_sample", "No complete time-series rows remain.")
            if self.db.execute("SELECT 1 FROM rows WHERE valid=0 AND t BETWEEN ? AND ? LIMIT 1", (first, last)).fetchone():
                raise AnalysisError("time_gaps", "Missing observations occur inside the time series; restrict it to an uninterrupted sample.")
            if self.db.execute("SELECT 1 FROM rows WHERE valid=1 GROUP BY t HAVING COUNT(*)>1 LIMIT 1").fetchone():
                raise AnalysisError("duplicate_time", "Time-series periods must be distinct.")
            if not self.date and self.db.execute("SELECT 1 FROM (SELECT t-LAG(t) OVER(ORDER BY t) AS gap FROM rows WHERE valid=1) WHERE gap!=1 LIMIT 1").fetchone():
                raise AnalysisError("time_gaps", "Time-series integer periods must be consecutive.")
            self.count, self.last_period = used, None if self.date or self.spec.time is None else last
            category_bytes = 0
            for name in self.spec.categorical:
                dtype = self.dtypes[name]
                if isinstance(dtype, pd.CategoricalDtype):
                    levels = list(dtype.categories)
                    declared = dtype
                else:
                    index = self.columns.index(name)
                    values = [row[0] for row in self.db.execute(f"SELECT DISTINCT c{index} FROM rows WHERE c{index} IS NOT NULL LIMIT ?", (MAX_PARAMETERS+2,))]
                    try:
                        levels = sorted(values)
                    except TypeError as exc:
                        raise AnalysisError("ambiguous_categories", "Mixed categorical labels need an explicitly declared Categorical dtype.") from exc
                    declared = pd.CategoricalDtype(levels, ordered=False)
                category_bytes += sum(sys.getsizeof(value)+128 for value in levels)
                if len(levels) > MAX_PARAMETERS+1 or category_bytes > min(MAX_CATEGORY_BYTES, self.budget//8):
                    raise AnalysisError("category_budget", "Global pre-filter time-series categories exceed the bounded metadata budget.")
                self.category_dtypes[name] = declared
            self.db.commit()
            self.db.execute("PRAGMA query_only=ON")
            self.path.chmod(0o400)
            names = [*self.columns, self.positions_column]
            self.source = Dataset.from_batches(self._factory, names, row_count=self.count)
            return self
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise AnalysisError("time_series_spill_failed", "Time-series sorting needs writable temporary storage and enough disk space.") from exc
        except BaseException:
            self.close()
            raise

    def _periods(self, raw, offset):
        if self.spec.time is None:
            return list(range(offset, offset+len(raw)))
        values = raw[self.spec.time]
        if bool(values.isna().any()):
            raise AnalysisError("missing_time", "A declared time column cannot contain missing periods.")
        self.date = pd.api.types.is_datetime64_any_dtype(values.dtype)
        if self.date:
            return values.dt.as_unit("ns").astype("int64").tolist()
        output = []
        for value in values:
            if isinstance(value, bool) or not math.isfinite(value) or value != int(value):
                raise AnalysisError("invalid_time", "Declared time needs integer periods or datetimes.")
            value = int(value)
            if not -(1<<63) <= value < 1<<63:
                raise AnalysisError("invalid_time", "Declared periods must fit signed64-bit integers.")
            output.append(value)
        return output

    def _factory(self):
        cursor = self.db.execute("SELECT "+",".join(f"c{i}" for i in range(len(self.columns)))+",pos FROM rows WHERE valid=1 ORDER BY t,pos")
        while records := cursor.fetchmany(self.rows):
            frame = pd.DataFrame.from_records(records, columns=[*self.columns, self.positions_column])
            for name, dtype in self.dtypes.items():
                if name in self.category_dtypes:
                    frame[name] = frame[name].astype(self.category_dtypes[name])
                elif isinstance(dtype, pd.DatetimeTZDtype):
                    frame[name] = pd.to_datetime(frame[name], unit="ns", utc=True).dt.tz_convert(dtype.tz)
                else:
                    frame[name] = (pd.to_datetime(frame[name], unit="ns") if pd.api.types.is_datetime64_any_dtype(dtype)
                                   else frame[name].astype(dtype))
            yield frame
        self.passes += 1

    def verify_original(self):
        self.original.assert_unchanged()
        digest, count = hashlib.sha256(), 0
        for raw in self.original.iter_batches(self.columns, batch_rows=self.rows):
            digest.update(row_hash_bytes(raw.loc[:, self.columns]))
            count += len(raw)
        self.original.assert_unchanged()
        if count != self.original_count or digest.hexdigest() != self.raw_hash:
            raise AnalysisError("source_changed", "The source changed during time-series estimation; no result is accepted.")

    def provenance(self):
        return {"original_data_hash": self.raw_hash, "ordered_source_rows": self.count,
                "ordered_source_passes": self.passes, "ordered_scratch_bytes": self.path.stat().st_size,
                "ordered_resource_plan": self.plan.record(),
                "ordered_source": "immutable local SQLite snapshot; original reverified after fit"}

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def __exit__(self, *_):
        self.close()


def _scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float, str, bytes)):
        return value
    raise AnalysisError("invalid_column_type", "Time-series spill supports numeric/text scalar columns and declared datetimes.")
