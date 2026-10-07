"""Full observed-candidate threshold search, global TSQR and disk bootstrap.

The exact native sequential/Bai-refined search is preserved. Normal-equation
prefix moments screen every distinct observed split; the best five candidates
are refitted with global QR. Candidate SSR paths and Gaussian bootstrap draws
live on owned disk, never all N candidate matrices or N x repetitions RAM.
Hansen (2000), https://users.ssc.wisc.edu/~behansen/papers/ecnmt_00.html ;
[TS] threshold, https://www.stata.com/manuals/tsthreshold.pdf .
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.covariance import hc_residuals
from openecon.engines.execution import qr_factor
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from .core import kernel_call, wald_test
from .replay_sample import ReplaySample
from .streaming_linear import _Notes, _result, _solve
from .streaming_ucm import _bytes, _tensor
from .tsmodels import threshold as native

SUPPORTED = frozenset({"threshold"})


def supports_spec(spec):
    return spec.estimator in SUPPORTED


@contextmanager
def _disk():
    try:
        with tempfile.TemporaryDirectory(
            prefix="openecon-threshold-", dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None
        ) as name:
            path = Path(name) / "grid.sqlite"
            sql = sqlite3.connect(path)
            try:
                for pragma in (
                    "journal_mode=OFF",
                    "synchronous=OFF",
                    "temp_store=FILE",
                    "cache_size=-4096",
                    "mmap_size=0",
                ):
                    sql.execute("PRAGMA " + pragma)
                sql.execute(
                    "CREATE TABLE physical(pos INTEGER PRIMARY KEY,q REAL,t INTEGER,y REAL,xv BLOB,xw BLOB)"
                )
                sql.execute("CREATE TABLE search(pos INTEGER PRIMARY KEY,q REAL,ssr REAL)")
                sql.execute("CREATE TABLE first_path(pos INTEGER PRIMARY KEY,q REAL,ssr REAL)")
                sql.execute("CREATE TABLE draws(pos INTEGER PRIMARY KEY,value BLOB)")
                path.chmod(0o600)
                yield sql, path
            finally:
                sql.close()
    except (OSError, sqlite3.Error) as exc:
        raise AnalysisError(
            "threshold_spill_failed",
            "Threshold search needs writable local scratch storage and enough disk for sorted inputs/candidate paths/bootstrap draws.",
        ) from exc


def _time(values):
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values.dt.as_unit("ns").astype("int64").tolist()
    result = []
    for value in values:
        if (
            isinstance(value, bool)
            or not math.isfinite(value)
            or value != int(value)
            or not -(1 << 63) <= int(value) < 1 << 63
        ):
            raise AnalysisError(
                "invalid_time", "Declared time needs exact integer periods or datetimes."
            )
        result.append(int(value))
    return result


def _scratch_space(path, new_bytes, operation):
    # Conservatively reserve current pages plus row/index payloads and sorting
    # slack before an O(N) or O(N*R) write. An OS refusal is still translated by
    # _disk if another process consumes free space after this check.
    required = path.stat().st_size + 16 * 1024**2 + int(new_bytes)
    if required > shutil.disk_usage(path.parent).free:
        raise AnalysisError(
            "threshold_spill_failed",
            f"Not enough free scratch space for {operation}; at least {required:,} bytes are conservatively required. Choose OPENECON_SCRATCH_DIRECTORY on a larger disk or reduce the model/bootstrap repetitions.",
        )


class _Grid:
    def __init__(self, sample, notes, sql, path):
        self.sample, self.notes, self.sql, self.disk_path = sample, notes, sql, path
        self.n = sample.nrows
        design = sample.designs["mean"]
        self.design = design
        self.q_name = sample.spec.columns["threshold_var"]
        self.q_name = self.q_name if isinstance(self.q_name, str) else self.q_name[0]
        varying = notes.option("regions")
        varying = design.terms if varying is None else list(varying)
        unknown = set(varying) - set(design.terms)
        if unknown or not varying:
            raise AnalysisError(
                "invalid_option", "regions must name at least one present, non-omitted design term."
            )
        self.varying = [i for i, term in enumerate(design.terms) if term in varying]
        self.common = [i for i, term in enumerate(design.terms) if term not in varying]
        self.kv, self.kw = len(self.varying), len(self.common)
        trim = float(notes.option("trim"))
        self.m = int(notes.option("nthresholds"))
        if not 0.0 < trim < 0.5:
            raise AnalysisError("invalid_option", "trim must lie strictly between zero and .5.")
        self.minimum = max(math.ceil(trim * self.n), self.kv + 1)
        if (self.m + 1) * self.minimum > self.n:
            raise AnalysisError(
                "insufficient_observations",
                "The full sample cannot hold the requested trimmed regions.",
            )
        self.plan()
        _scratch_space(
            path,
            self.n * (2 * (8 * (len(design.terms) + 4) + 96) + 3 * 32 + 2 * 48),
            "sorted threshold records, lookup indexes and candidate paths",
        )
        squares = _CompensatedSum((1 + len(design.terms),))
        moments = _WeightedMoments(2, intercept=True)
        for batch in sample.batches():
            y = batch.numeric(sample.spec.outcome)
            x = sample.raw_design(batch, "mean")
            squares.add(torch.cat((y[:, None], x), 1).square().sum(0))
            moments.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
        if float(moments.m2.value[1]) <= 0.0:
            raise AnalysisError(
                "constant_outcome",
                "The outcome is constant; no threshold regression can be estimated.",
            )
        scales = (squares.value / self.n).sqrt()
        if not bool(torch.isfinite(scales).all()):
            raise AnalysisError(
                "numerical_failure", "Threshold RMS scales exceed finite precision; rescale inputs."
            )
        self.scale_y = float(scales[0]) or 1.0
        self.scales_v = scales[1:][self.varying].clamp_min(1e-300)
        self.scales_w = scales[1:][self.common].clamp_min(1e-300)
        self.total_centered = float(moments.m2.value[1] * moments.magnitude[1] ** 2)
        self.total_uncentered = float(squares.value[0])
        for batch in sample.batches():
            y = batch.numeric(sample.spec.outcome) / self.scale_y
            x = sample.raw_design(batch, "mean")
            q = batch.numeric(self.q_name)
            xv = x[:, self.varying] / self.scales_v
            xw = x[:, self.common] / self.scales_w
            time = (
                _time(batch.frame[sample.spec.time])
                if sample.spec.time
                else batch.positions.tolist()
            )
            sql.executemany(
                "INSERT INTO physical VALUES(?,?,?,?,?,?)",
                (
                    (int(pos), float(qi), int(t), float(yi), _bytes(v), _bytes(w))
                    for pos, qi, t, yi, v, w in zip(
                        batch.positions.tolist(), q.tolist(), time, y.tolist(), xv, xw, strict=True
                    )
                ),
            )
        if (
            sample.spec.time
            and sql.execute(
                "SELECT 1 FROM physical GROUP BY t HAVING COUNT(*)>1 LIMIT 1"
            ).fetchone()
        ):
            raise AnalysisError(
                "duplicate_time", "Threshold time periods must be distinct; gaps are allowed."
            )
        sql.execute(
            "CREATE TABLE rows AS SELECT ROW_NUMBER() OVER(ORDER BY q,t,pos)-1 AS rank,* FROM physical ORDER BY q,t,pos"
        )
        sql.execute("CREATE UNIQUE INDEX rank_lookup ON rows(rank)")
        sql.execute("CREATE INDEX q_lookup ON rows(q)")
        sql.execute("CREATE INDEX pos_lookup ON rows(pos)")
        first, last = sql.execute("SELECT MIN(q),MAX(q) FROM rows").fetchone()
        if first == last:
            raise AnalysisError(
                "constant_threshold_variable", "The threshold variable is constant."
            )
        sql.commit()
        self.numeric_hash = self.digest()
        self.draw_rep = None
        self.passes = 0
        self.base_total = self._totals()
        self.yy, self.ww, self.wy = self.base_total

    def plan(self, repetitions=0):
        width = self.kw + (self.m + 1) * self.kv
        if self.n <= width:
            raise AnalysisError(
                "insufficient_observations",
                "Threshold inference needs more full-sample observations than all common and region-specific coefficients.",
            )
        bytes_per_row = 256 * (width + 1) ** 2 + 64 * max(0, repetitions)
        self.resource = self.sample.plan_rows(
            "global observed-threshold prefix/QR search",
            {
                "candidate_region_gram_and_covariance": 512 * (width + 1) ** 2 * (self.m + 3),
                "SQLite_sorted_path_and_draw_cache": 8 * 1024**2,
                "minimum_native_normal_fill": 256 * max(1, repetitions),
            },
            bytes_per_row,
        )
        self.rows = self.sample.rows

    def digest(self):
        digest = hashlib.sha256()
        cursor = self.sql.execute("SELECT rank,pos,q,t,y,xv,xw FROM rows ORDER BY rank")
        while records := cursor.fetchmany(self.rows):
            for rank, pos, q, t, y, xv, xw in records:
                digest.update(str((rank, pos, q, t, y)).encode())
                digest.update(xv)
                digest.update(xw)
        return digest.hexdigest()

    def blocks(self, start=0, stop=None):
        stop = self.n if stop is None else stop
        cursor = self.sql.execute(
            "SELECT rank,q,y,xv,xw FROM rows WHERE rank>=? AND rank<? ORDER BY rank", (start, stop)
        )
        while records := cursor.fetchmany(self.rows):
            first = int(records[0][0])
            n = len(records)
            q = torch.tensor([row[1] for row in records], dtype=torch.float64)
            y = torch.tensor([row[2] for row in records], dtype=torch.float64)
            xv = torch.stack([_tensor(row[3], (self.kv,)) for row in records])
            xw = torch.stack([_tensor(row[4], (self.kw,)) for row in records])
            if self.draw_rep is not None:
                payload = self.sql.execute(
                    "SELECT pos,value FROM draws WHERE pos>=? AND pos<? ORDER BY pos",
                    (first, first + n),
                ).fetchall()
                if len(payload) != n:
                    raise AnalysisError(
                        "source_changed", "Threshold bootstrap draw rows changed or were truncated."
                    )
                y = torch.tensor(
                    [float(_tensor(row[1], (self.reps,))[self.draw_rep]) for row in payload],
                    dtype=torch.float64,
                )
            yield first, q, y, xv, xw
        self.passes += 1

    def _totals(self):
        yy, ww, wy = (
            _CompensatedSum(()),
            _CompensatedSum((self.kw, self.kw)),
            _CompensatedSum((self.kw,)),
        )
        for _, _, y, _, w in self.blocks():
            yy.add(y @ y)
            ww.add(w.T @ w)
            wy.add(w.T @ y)
        return float(yy.value), ww.value, wy.value

    def moment(self, start, stop):
        vv, wv, vy = (
            _CompensatedSum((self.kv, self.kv)),
            _CompensatedSum((self.kw, self.kv)),
            _CompensatedSum((self.kv,)),
        )
        for _, _, y, v, w in self.blocks(start, stop):
            vv.add(v.T @ v)
            wv.add(w.T @ v)
            vy.add(v.T @ y)
        return vv.value, wv.value, vy.value

    def threshold(self, position):
        return float(
            self.sql.execute("SELECT q FROM rows WHERE rank=?", (position - 1,)).fetchone()[0]
        )

    def design_blocks(self, splits):
        splits = torch.tensor(sorted(splits), dtype=torch.int64)
        for first, _, y, v, w in self.blocks():
            rank = torch.arange(first, first + len(y), dtype=torch.int64)
            region = torch.bucketize(rank, splits, right=True)
            x = torch.cat((w, *[v * (region == r)[:, None] for r in range(len(splits) + 1)]), 1)
            yield first, x, y

    def fit(self, splits):
        width = self.kw + (len(splits) + 1) * self.kv
        tree = _TSQRTree()
        for _, x, y in self.design_blocks(splits):
            tree.add(qr_factor(torch.cat((x, y[:, None]), 1)))
        factor = tree.finish()
        beta, bread, condition = _solve(factor, width)
        rss = _CompensatedSum(())
        for _, x, y in self.design_blocks(splits):
            rss.add((y - x @ beta).square().sum())
        return beta, bread, float(rss.value), condition

    def exact(self, splits):
        return self.fit(splits)[2]

    def search(self, fixed):
        bounds = sorted({0, self.n, *fixed})
        blocks = [self.moment(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        self.sql.execute("DELETE FROM search")
        for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            running_vv, running_wv, running_vy = (
                _CompensatedSum((self.kv, self.kv)),
                _CompensatedSum((self.kw, self.kv)),
                _CompensatedSum((self.kv,)),
            )
            total_vv, total_wv, total_vy = blocks[r]
            for first, q, y, v, w in self.blocks(a, b):
                cvv = torch.cumsum(v[:, :, None] * v[:, None, :], 0) + running_vv.value
                cwv = torch.cumsum(w[:, :, None] * v[:, None, :], 0) + running_wv.value
                cvy = torch.cumsum(v * y[:, None], 0) + running_vy.value
                positions = torch.arange(first + 1, first + len(y) + 1, dtype=torch.int64)
                distinct = torch.empty(len(y), dtype=torch.bool)
                distinct[:-1] = q[:-1] < q[1:]
                nextq = self.sql.execute(
                    "SELECT q FROM rows WHERE rank=?", (first + len(y),)
                ).fetchone()
                distinct[-1] = bool(nextq and float(q[-1]) < nextq[0])
                admissible = (
                    distinct & (positions - a >= self.minimum) & (b - positions >= self.minimum)
                )
                running_vv.add(v.T @ v)
                running_wv.add(w.T @ v)
                running_vy.add(v.T @ y)
                if not bool(admissible.any()):
                    continue
                index = admissible.nonzero().flatten()
                values = native._Grid._solve(
                    self,
                    blocks,
                    r,
                    cvv[index],
                    cwv[index],
                    cvy[index],
                    total_vv,
                    total_wv,
                    total_vy,
                )
                if bool(torch.isnan(values).any()):
                    raise AnalysisError(
                        "numerical_failure",
                        "Candidate threshold normal equations produced NaN; rescale or simplify region regressors.",
                    )
                self.sql.executemany(
                    "INSERT INTO search VALUES(?,?,?)",
                    (
                        (int(pos), float(gamma), float(value))
                        for pos, gamma, value in zip(
                            positions[index].tolist(),
                            q[index].tolist(),
                            values.tolist(),
                            strict=True,
                        )
                        if math.isfinite(value)
                    ),
                )
        self.sql.commit()
        count, minimum = self.sql.execute("SELECT COUNT(*),MIN(ssr) FROM search").fetchone()
        if not count:
            raise AnalysisError(
                "no_admissible_threshold",
                "No distinct threshold leaves identified regions of the required minimum size.",
            )
        return count, float(minimum)

    def best(self, fixed):
        count, _ = self.search(fixed)
        top = self.sql.execute("SELECT pos FROM search ORDER BY ssr,pos LIMIT 5").fetchall()
        exact = []
        for (pos,) in top:
            try:
                exact.append((self.exact([*fixed, pos]), pos))
            except AnalysisError as exc:
                if exc.code not in {"singular_design", "insufficient_observations"}:
                    raise
        if not exact:
            raise AnalysisError(
                "no_admissible_threshold",
                "Best screened regions are rank deficient under exact QR.",
            )
        value, position = min(exact)
        return position, value, count

    def path(self):
        count, minimum = self.sql.execute("SELECT COUNT(*),MIN(ssr) FROM first_path").fetchone()
        selected = (
            torch.linspace(0, count - 1, steps=min(200, count), dtype=torch.float64)
            .round()
            .to(torch.int64)
            .tolist()
        )
        output = []
        # One bounded ordered scan avoids repeatedly walking N candidate rows
        # for each of the200 retained preview offsets.
        cursor = self.sql.execute("SELECT q,ssr FROM first_path ORDER BY pos")
        offset, index = 0, 0
        while records := cursor.fetchmany(self.rows):
            end = offset + len(records)
            while index < len(selected) and selected[index] < end:
                output.append(records[selected[index] - offset])
                index += 1
            offset = end
        if index != len(selected):
            raise AnalysisError("source_changed", "The stored candidate path changed during its bounded scan.")
        return (
            count,
            float(minimum),
            {
                "threshold": [row[0] for row in output],
                "ssr": [row[1] * self.scale_y**2 for row in output],
                "note": "all admissible observed values screened globally; at most200 evenly spaced candidates",
            },
        )

    def gaussian_draws(self, residual_factory, reps, seed):
        # CPU normal_fill generates contiguous groups of16 and recomputes the
        # last16 when total%16!=0. Aligned calls and one sufficiently large last
        # call preserve the native N x reps row-major RNG law byte-for-byte.
        self.plan(reps)
        self.reps = reps
        _scratch_space(
            self.disk_path,
            self.n * (8 * reps + 96),
            "all full-sample bootstrap draws on disk",
        )
        unit = 16 // math.gcd(reps, 16)
        rows = max(unit, (self.rows // unit) * unit)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.sql.execute("DELETE FROM draws")
        pending = torch.empty(0, dtype=torch.float64)
        iterator = iter(residual_factory())
        offset = 0
        self.draw_maximum_rows = 0
        while offset < self.n:
            remaining = self.n - offset
            take = min(rows, remaining)
            if remaining > take and (remaining - take) * reps < 16:
                take = remaining
            while len(pending) < take:
                try:
                    pending = torch.cat((pending, next(iterator)))
                except StopIteration as exc:
                    raise AnalysisError(
                        "source_changed", "Threshold residual replay ended before its full sample."
                    ) from exc
            residual = pending[:take]
            pending = pending[take:].clone()
            self.draw_maximum_rows = max(self.draw_maximum_rows, take)
            draws = (
                torch.randn((take, reps), generator=generator, dtype=torch.float64)
                * residual[:, None]
            )
            self.sql.executemany(
                "INSERT INTO draws VALUES(?,?)",
                ((offset + i, _bytes(row)) for i, row in enumerate(draws)),
            )
            offset += take
        self.sql.commit()

    def bootstrap(self, single, initial_null, reps, seed):
        beta, _, single_ssr, _ = self.fit(single)
        statistic = self.n * (initial_null - single_ssr) / single_ssr
        self.gaussian_draws(
            lambda: (y - x @ beta for _, x, y in self.design_blocks(single)), reps, seed
        )
        exceed = 0
        try:
            for rep in range(reps):
                self.draw_rep = rep
                self.yy, self.ww, self.wy = self._totals()
                null = self.exact([])
                _, alternative = self.search([])
                if not alternative > 0.0:
                    raise AnalysisError(
                        "perfect_fit",
                        "A bootstrap threshold fits its pseudo-outcome exactly; F statistic is undefined.",
                    )
                exceed += self.n * (null - alternative) / alternative >= statistic
        finally:
            self.draw_rep = None
            self.yy, self.ww, self.wy = self.base_total
        return {
            "statistic": statistic,
            "df": None,
            "p_value": exceed / reps,
            "distribution": "bootstrap",
            "label": "Hansen fixed-regressor bootstrap test of no threshold effect",
            "reps": reps,
            "seed": seed,
            "draw_generation": "nativeCPU float64 row-major16-aligned calls with one final tail; draws on owned disk",
            "maximum_draw_batch_rows": self.draw_maximum_rows,
        }


def _covariance(grid, splits, beta, bread):
    sample = grid.sample
    kind = "HC1" if sample.spec.covariance == "robust" else sample.spec.covariance
    k = len(beta)
    df = grid.n - k
    thresholds = torch.tensor([grid.threshold(pos) for pos in splits], dtype=torch.float64)
    meat, rss = _CompensatedSum(bread.shape), _CompensatedSum(())
    predictions = []
    for batch in sample.batches():
        raw = sample.raw_design(batch, "mean")
        y = batch.numeric(sample.spec.outcome) / grid.scale_y
        v = raw[:, grid.varying] / grid.scales_v
        w = raw[:, grid.common] / grid.scales_w
        region = torch.bucketize(batch.numeric(grid.q_name), thresholds, right=False)
        x = torch.cat((w, *[v * (region == r)[:, None] for r in range(len(splits) + 1)]), 1)
        residual = y - x @ beta
        rss.add(residual.square().sum())
        if kind != "nonrobust":
            leverage = (x @ bread * x).sum(1) if kind in {"HC2", "HC3"} else None
            corrected = kernel_call(hc_residuals, residual, leverage, kind)
            scores = x * corrected[:, None]
            meat.add(scores.T @ scores)
        take = min(400 - len(predictions), len(y))
        for pos, observed, resid in zip(
            batch.positions[:take].tolist(),
            (y[:take] * grid.scale_y).tolist(),
            (residual[:take] * grid.scale_y).tolist(),
            strict=True,
        ):
            predictions.append(
                {"row": pos, "observed": observed, "fitted": observed - resid, "residual": resid}
            )
    ssr = float(rss.value)
    factor = grid.n / df if kind == "HC1" else 1.0
    covariance = (
        bread * (ssr / df) if kind == "nonrobust" else bread @ (meat.value * factor) @ bread
    )
    units = torch.cat(
        (
            grid.scale_y / grid.scales_w,
            *[grid.scale_y / grid.scales_v for _ in range(len(splits) + 1)],
        )
    )
    params = units * beta
    covariance = units[:, None] * covariance * units[None]
    info = {
        "covariance": kind,
        "df_resid": df,
        "df_inference": df,
        "correction": (
            "classical: SSR/(N-K)" if kind == "nonrobust" else f"{kind} full-row leverage sandwich"
        )
        + "; thresholds treated as known",
        "small_sample_correction": factor if kind == "HC1" else None,
    }
    return params, covariance, info, ssr * grid.scale_y**2, predictions


def fit_streaming_threshold(spec, source, *, batch_rows=None):
    if not isinstance(source, Dataset):
        raise AnalysisError("invalid_data", "Threshold replay needs a Dataset.")
    if spec.time == spec.outcome:
        raise AnalysisError("invalid_spec", "The time column must differ from outcome.")
    with torch.no_grad(), torch.device("cpu"), _disk() as (sql, path):
        sample = ReplaySample(spec, source, batch_rows=batch_rows)
        sample.add_design("mean")
        sample.prepare()
        notes = _Notes(spec)
        grid = _Grid(sample, notes, sql, path)
        null = grid.exact([])
        splits = []
        sequence = []
        single = None
        for step in range(grid.m):
            position, _, _ = grid.best(splits)
            if step == 0:
                sql.execute("INSERT INTO first_path SELECT * FROM search")
                sql.commit()
                single = [position]
            splits = sorted([*splits, position])
            for _ in range(10 if step else 0):
                changed = False
                for index in range(len(splits)):
                    others = splits[:index] + splits[index + 1 :]
                    new, _, _ = grid.best(others)
                    if new != splits[index]:
                        splits = sorted([*others, new])
                        changed = True
                if not changed:
                    break
            sequence.append(
                {
                    "thresholds": [grid.threshold(pos) for pos in splits],
                    "ssr": grid.exact(splits) * grid.scale_y**2,
                }
            )
        beta, bread, _, condition = grid.fit(splits)
        params, covariance, info, ssr, predictions = _covariance(grid, splits, beta, bread)
        k = len(params)
        df = grid.n - k
        hasconstant = "Intercept" in grid.design.terms
        total = grid.total_centered if hasconstant else grid.total_uncentered
        if (
            ssr < 1e-24 * grid.total_uncentered
            or null * grid.scale_y**2 < 1e-24 * grid.total_uncentered
        ):
            raise AnalysisError(
                "perfect_fit", "Threshold or linear null fits exactly; SSR inference is undefined."
            )

        def criteria(value, parameters):
            base = grid.n * math.log(value / grid.n)
            return {
                "aic": base + 2 * parameters,
                "bic": base + parameters * math.log(grid.n),
                "hqic": base + 2 * parameters * math.log(math.log(grid.n)),
            }

        thresholds = [grid.threshold(pos) for pos in splits]
        bounds = [0, *splits, grid.n]
        region_sizes = [b - a for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        terms = [grid.design.terms[i] for i in grid.common] + [
            f"region{r + 1}:{grid.design.terms[i]}"
            for r in range(len(splits) + 1)
            for i in grid.varying
        ]
        r2 = 1.0 - ssr / total
        metrics = {
            "r_squared": r2,
            "adjusted_r_squared": 1.0 - (1.0 - r2) * (grid.n - int(hasconstant)) / df,
            "rmse": math.sqrt(ssr / df),
            "ssr": ssr,
            **criteria(ssr, k),
            "df_model": k - int(hasconstant),
            "df_resid": df,
            **{f"threshold{j + 1}": value for j, value in enumerate(thresholds)},
        }
        count, minimum, curve = grid.path()
        by_count = [
            {
                "thresholds": 0,
                "ssr": null * grid.scale_y**2,
                **criteria(null * grid.scale_y**2, grid.kw + grid.kv),
            }
        ]
        for count_, record in enumerate(sequence, 1):
            by_count.append(
                {
                    "thresholds": count_,
                    "ssr": record["ssr"],
                    **criteria(record["ssr"], grid.kw + (count_ + 1) * grid.kv),
                }
            )
        extra = {
            "thresholds": thresholds,
            "threshold_variable": grid.q_name,
            "trim": float(notes.option("trim")),
            "minimum_region_size": grid.minimum,
            "region_sizes": region_sizes,
            "regions": [grid.design.terms[i] for i in grid.varying],
            "common": [grid.design.terms[i] for i in grid.common],
            "sequence": sequence,
            "by_number_of_thresholds": by_count,
            "candidates": count,
            "ssr_path": curve,
            "region_definition": "region1 q<=threshold1; subsequent regions lower<q<=upper",
        }
        if grid.m == 1:
            native._check_lr_precision(minimum, grid.yy)
            critical = -2.0 * math.log(1.0 - math.sqrt(1.0 - spec.alpha))
            cutoff = minimum * (1.0 + critical / grid.n)
            low, high = sql.execute(
                "SELECT MIN(q),MAX(q) FROM first_path WHERE ssr<=?", (cutoff,)
            ).fetchone()
            extra["threshold_confidence_set"] = {
                "low": low,
                "high": high,
                "level": 1.0 - spec.alpha,
                "critical_value": critical,
                "method": "Hansen (2000) LR inversion; homoskedastic asymptotic law; global screened SSR; hull of all accepted candidates",
            }
        tests = {}
        slopes = [i for i, term in enumerate(terms) if not term.endswith("Intercept")]
        if slopes and hasconstant:
            tests["model"] = wald_test(
                params,
                covariance,
                slopes,
                df_resid=df,
                label="F test that all coefficients except constants are zero",
            )
        reps = int(notes.option("bootstrap"))
        if reps:
            tests["threshold_effect"] = grid.bootstrap(
                single, null, reps, int(notes.option("seed"))
            )
        if grid.digest() != grid.numeric_hash:
            raise AnalysisError(
                "source_changed", "Sorted threshold input records changed during estimation."
            )
        for _ in sample.batches():
            pass
        bundle = _result(
            sample,
            terms=terms,
            beta=params,
            covariance=covariance,
            info=info,
            metrics=metrics,
            notes=notes,
            predictions=predictions,
            tests=tests,
            solver="global_observed_grid_tsqr",
            diagnostics={
                "candidates": count,
                "condition_number": condition,
                "sorted_passes": grid.passes,
            },
            extra=extra,
            resource=grid.resource.record(),
            title=f"Threshold regression ({grid.m} thresholds)",
            provenance_extra={
                "threshold_search": "every admissible observed split; bounded prefix screen then global top5 TSQR; sequential Bai refinement",
                "candidate_ties": "screenedSSR then sorted position; exactSSR then sorted position",
                "numeric_spool_hash": grid.numeric_hash,
                "threshold_scratch_bytes": path.stat().st_size,
                "bootstrap_draws_in_ram": False,
                "sorted_risk_or_candidate_matrix_in_ram": False,
            },
        )
        for coefficient in bundle.coefficients:
            if coefficient.term.startswith("region"):
                coefficient.equation = coefficient.term.split(":", 1)[0]
        return bundle
