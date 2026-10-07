"""Bounded statistical replay with selected-column content integrity checks.

Numeric kernels use anchored float64 moments. Source blocks and small result
geometry are resident; exact grouped order statistics use an owned SQLite file.
The workspace estimate excludes source-reader buffers and Python result objects.
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.resources import plan_workspace, workspace_budget_bytes
from . import common as c


class Replay:
    def __init__(self, data: Dataset, names: list[str], numeric: list[str], missing: str, *, check_small: bool = True):
        c.check_choice(missing, "missing", ("drop", "raise"))
        if len(set(names)) != len(names):
            raise AnalysisError("invalid_spec", "A column is used in more than one role.")
        absent = [name for name in names if name not in data.columns]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        self.data, self.names, self.numeric, self.missing = data, names, numeric, missing
        self.check_small = check_small
        self.digest: str | None = None
        self.n = self.dropped = self.raw_rows = self.passes = 0
        self.category_schema: dict[str, pd.Index | None] | None = None
        width = max(len(names), len(numeric), 1)
        fixed = 64 * (len(numeric) + 1) ** 2 + 4 * 1024**2
        self.rows = min(65536, max(1, (workspace_budget_bytes() - fixed) // (128 * width)))
        self.plan = plan_workspace("chunked statistical helper", {
            "moments_and_metadata": fixed, "selected_numeric_blocks": self.rows * 128 * width})

    def batches(self, *, listwise: bool = True) -> Iterator[pd.DataFrame]:
        digest = hashlib.sha256()
        n = dropped = raw_rows = 0
        largest = {name: 0.0 for name in self.numeric}
        for raw in self.data.iter_batches(self.names, batch_rows=self.rows):
            schema = {name: raw[name].dtype.categories if isinstance(raw[name].dtype, pd.CategoricalDtype) else None
                      for name in self.names}
            if self.category_schema is None:
                self.category_schema = schema
            else:
                for name, categories in schema.items():
                    original = self.category_schema[name]
                    if (categories is None) != (original is None) or (categories is not None and not categories.equals(original)):
                        raise AnalysisError("source_changed", "The declared category ordering changed between statistical source blocks.")
            digest.update(pd.util.hash_pandas_object(raw, index=False).to_numpy(dtype="uint64").tobytes())
            raw_rows += len(raw)
            # Validate dtypes even when a source block contains only missing rows.
            for name in self.numeric:
                dtype = raw[name].dtype
                from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
                if not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)) or is_complex_dtype(dtype):
                    raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric.")
            keep = ~raw.isna().any(axis=1)
            absent = int((~keep).sum())
            if absent and self.missing == "raise":
                raise AnalysisError("missing_values", "The selected columns contain missing values.")
            dropped += absent
            frame = raw.loc[keep] if listwise else raw
            if len(frame):
                for name in self.numeric:
                    values = c.values(frame, name, allow_missing=not listwise, check_scale=False)
                    finite = values[~torch.isnan(values)]
                    if len(finite):
                        largest[name] = max(largest[name], float(finite.abs().max()))
                n += len(frame)
                yield frame
        current = digest.hexdigest()
        if self.digest is not None and current != self.digest:
            raise AnalysisError("source_changed", "The selected source values changed between statistical passes.")
        if not raw_rows:
            raise AnalysisError("empty_data", "The dataset contains no observations.")
        if not n:
            raise AnalysisError("empty_sample", "No complete observations remain after excluding missing values.")
        if self.check_small:
            for name, magnitude in largest.items():
                if 0.0 < magnitude < c._SMALLEST:
                    raise AnalysisError("non_finite_values", f"Every value of column '{name}' is below {c._SMALLEST:g} in magnitude, too small for sums of squares in double precision; rescale it.")
        self.digest, self.n, self.dropped, self.raw_rows = current, n, dropped, raw_rows
        self.passes += 1

    def attrs(self) -> dict[str, Any]:
        return {"streaming": True, "batch_rows": self.rows, "source_passes": self.passes,
                "source_content_sha256": self.digest, "resource_plan": self.plan.record(),
                "precision": "float64"}


class Moments:
    """Chan moments of coordinates measured from the first finite observation."""
    def __init__(self, width: int):
        self.n = 0
        self.anchor = torch.zeros(width, dtype=torch.float64)
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.sscp = torch.zeros((width, width), dtype=torch.float64)
        self.raw_ss = torch.zeros(width, dtype=torch.float64)
        self.low = torch.full((width,), float("inf"), dtype=torch.float64)
        self.high = -self.low

    def add(self, x: Tensor, *, anchor: Tensor | None = None) -> None:
        if not len(x):
            return
        if not self.n:
            self.anchor = x[0].clone() if anchor is None else anchor.clone()
        centred = x - self.anchor
        size = len(x)
        mean = centred.mean(0)
        mean = mean + (centred - mean).mean(0)
        dev = centred - mean
        block = dev.T @ dev
        delta = mean - self.mean
        total = self.n + size
        self.sscp += block + torch.outer(delta, delta) * (self.n * size / total)
        self.mean += delta * (size / total)
        self.n = total
        self.raw_ss += x.square().sum(0)
        self.low = torch.minimum(self.low, x.min(0).values)
        self.high = torch.maximum(self.high, x.max(0).values)
        if not all(bool(torch.isfinite(v).all()) for v in
                   (self.mean, self.sscp, self.raw_ss)):
            raise AnalysisError("numerical_failure", "Statistical moments exceed float64 precision; rescale inputs.")

    @property
    def location(self) -> Tensor:
        return self.anchor + self.mean


def numeric_moments(sample: Replay, names: list[str], *, difference: bool = False) -> Moments:
    moments = Moments(len(names) + int(difference))
    for frame in sample.batches():
        x = c.matrix(frame, names, check_scale=False)
        if difference:
            x = torch.cat((x, (x[:, 0] - x[:, 1])[:, None]), 1)
        moments.add(x)
    return moments


class Grouped:
    """Finite group metadata; exact ranks spill to a disposable SQLite database."""
    LIMIT = 8192

    def __init__(self, sample: Replay, y: str, by: str, *, ordered: bool = False):
        self.sample, self.y, self.by = sample, y, by
        self.states: dict[Any, Moments] = {}
        self.overall = Moments(1)
        self.ordered = ordered
        self.temp, self.db = None, None
        try:
            if ordered:
                self.temp = tempfile.TemporaryDirectory(prefix="openecon-statistics-",
                                                       dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None)
                self.db = sqlite3.connect(str(Path(self.temp.name) / "order.sqlite"))
                self.db.execute("PRAGMA temp_store=FILE")
                self.db.execute("PRAGMA cache_size=-2048")
                self.db.execute("CREATE TABLE values_ (g INTEGER, v REAL)")
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise AnalysisError("statistics_spill_failed", "Exact grouped order statistics need writable local scratch storage.") from exc
        self.ids: dict[Any, int] = {}
        self.labels: list[Any] = []
        self.shift = 0.0
        self.collect()

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.temp is not None:
            self.temp.cleanup()
            self.temp = None

    def collect(self) -> None:
        try:
            for frame in self.sample.batches():
                values = c.values(frame, self.y, check_scale=False)
                if not self.overall.n:
                    self.shift = float(values[0])
                self.overall.add(values[:, None])
                codes, labels = c.group_codes(frame, self.by)
                for code, label in enumerate(labels):
                    try:
                        state = self.states.get(label)
                    except TypeError as exc:
                        raise AnalysisError("invalid_groups", "Group labels must be scalar values.") from exc
                    if state is None:
                        if len(self.states) >= self.LIMIT or sum(len(str(v)) for v in self.states) + len(str(label)) > 1024**2:
                            raise AnalysisError("workspace_limit", "Statistical group metadata exceeds its bounded workspace.")
                        self.ids[label] = len(self.states)
                        state = self.states[label] = Moments(1)
                    selected = values[codes == code]
                    state.add(selected[:, None], anchor=torch.tensor([self.shift], dtype=torch.float64))
                    if self.db is not None:
                        self.db.executemany("INSERT INTO values_ VALUES (?, ?)",
                                            ((self.ids[label], float(v) - self.shift) for v in selected.tolist()))
            # Same ordering semantics as the resident classical helper.
            categories = self.sample.category_schema[self.by]
            if categories is None:
                _, self.labels = c.group_codes(pd.DataFrame({self.by: list(self.states)}), self.by)
            else:
                order = {c._json_scalar(value): index for index, value in enumerate(categories)
                         if c._json_scalar(value) in self.states}
                self.labels = sorted(self.states, key=order.__getitem__)
            if self.db is not None:
                self.db.execute("CREATE INDEX ordered_values ON values_(g, v)")
                self.db.commit()
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise AnalysisError("statistics_spill_failed", "Exact grouped order statistics could not be written to local scratch storage.") from exc
        except BaseException:
            self.close()
            raise

    @property
    def moments(self) -> c.GroupMoments:
        states = [self.states[label] for label in self.labels]
        n = torch.tensor([state.n for state in states], dtype=torch.float64)
        mean = torch.tensor([float(state.mean[0]) for state in states], dtype=torch.float64)
        ss = torch.tensor([float(state.sscp[0, 0]) for state in states], dtype=torch.float64)
        var = torch.where(n > 1, ss / (n - 1).clamp_min(1), torch.full_like(ss, float("nan")))
        return c.GroupMoments(n, mean, ss, var)

    def centres(self, *, trimmed: bool = False) -> Tensor:
        assert self.db is not None
        out = []
        for label in self.labels:
            group, n = self.ids[label], self.states[label].n
            if trimmed:
                cut = math.floor(0.05 * n)
                # SQL AVG can lose low-order bits at a large level; values are
                # measured from a common anchor before insertion.
                value = self.db.execute("SELECT AVG(v) FROM (SELECT v FROM values_ WHERE g=? ORDER BY v LIMIT ? OFFSET ?)",
                                        (group, n - 2 * cut, cut)).fetchone()[0]
            else:
                rows = self.db.execute("SELECT v FROM values_ WHERE g=? ORDER BY v LIMIT ? OFFSET ?",
                                       (group, 2 if n % 2 == 0 else 1, (n - 1) // 2)).fetchall()
                value = sum(row[0] for row in rows) / len(rows)
            out.append(value)
        return torch.tensor(out, dtype=torch.float64)

    def levene(self, centers: Tensor) -> dict[str, Any]:
        states = [Moments(1) for _ in self.labels]
        positions = {label: index for index, label in enumerate(self.labels)}
        for frame in self.sample.batches():
            values = c.values(frame, self.y, check_scale=False) - self.shift
            codes, labels = c.group_codes(frame, self.by)
            for code, label in enumerate(labels):
                position = positions.get(label)
                if position is None:
                    raise AnalysisError("source_changed", "The grouping values changed between statistical passes.")
                states[position].add((values[codes == code] - centers[position]).abs()[:, None])
        n = torch.tensor([state.n for state in states], dtype=torch.float64)
        mean = torch.tensor([float(state.location[0]) for state in states], dtype=torch.float64)
        ss = torch.tensor([float(state.sscp[0, 0]) for state in states], dtype=torch.float64)
        grand = float((n * mean).sum() / n.sum())
        between = float((n * (mean - grand).square()).sum())
        within = float(ss.sum())
        df1, df2 = len(states) - 1, int(n.sum()) - len(states)
        statistic = c.ratio(between / df1, within / df2) if df1 > 0 and df2 > 0 else None
        adjusted = c.ratio(within**2, float((ss.square() / (n - 1).clamp_min(1))[n > 1].sum())) if within > 0 else None
        return {"statistic": statistic, "df1": df1, "df2": df2,
                "p_value": c.f_upper(statistic, df1, df2), "adjusted_df2": adjusted}
