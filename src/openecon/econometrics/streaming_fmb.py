"""Fama-MacBeth on bounded replays and owned disk-resident period factors.

Only linear cross-product geometry is stored in augmented QR factors. Each
period is solved separately; coefficient averages/covariances are accumulated
in standardized coordinates and mapped back once. HAC uses the actual global
period distances (or the declared datetime rank convention), never batch-local
lags. Neither observation matrices nor a T-by-K coefficient table are collected.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
import math
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.covariance import newey_west_lags
from openecon.engines.streaming_hac import HACAccumulator
from openecon.engines.streaming_ols import _CompensatedSum
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.resources import plan_workspace
from openecon.streaming_design import encode_cluster_labels

from .core import wald_test
from .panel.common import require_residual_variation
from .replay_sample import ReplaySample
from .streaming_linear import _CrossMoments, _Notes, _finite, _period_keys, _result, _scratch_directory, _solve


class _PeriodFactors:
    """Fixed-size LRU, O(T K²+N label bytes) disk, no global Python label map."""

    cache_bytes = 4*1024**2

    def __init__(self, width):
        self.width = width
        self.cache = OrderedDict()
        self.accounted = self.peak = self.writes = 0
        self.connection = self.scratch = None
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-fmb-", dir=_scratch_directory())
            self.path = Path(self.scratch.name)/"periods.sqlite3"
            self.connection = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-2048", "temp_store=FILE", "mmap_size=0"):
                self.connection.execute("PRAGMA "+pragma)
            self.connection.execute("CREATE TABLE periods(t INTEGER PRIMARY KEY,n INTEGER,r INTEGER,factor BLOB)")
            self.connection.execute("CREATE TABLE seen(k BLOB,t INTEGER,PRIMARY KEY(k,t)) WITHOUT ROWID")
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as error:
            self.close()
            raise self.failure() from error

    @staticmethod
    def failure():
        return AnalysisError("period_factor_spill_failed", "Fama-MacBeth replay needs writable temporary storage and enough disk space.")

    def _write(self, period, entry):
        n, factor = entry
        packed = memoryview(factor.contiguous().numpy().tobytes())
        self.connection.execute("INSERT OR REPLACE INTO periods VALUES(?,?,?,?)", (period, n, len(factor), packed))
        self.writes += 1

    def _get(self, period):
        if period in self.cache:
            self.cache.move_to_end(period)
            return self.cache[period]
        row = self.connection.execute("SELECT n,r,factor FROM periods WHERE t=?", (period,)).fetchone()
        if row is None:
            return None
        n, rows, buffer = row
        if not 1 <= rows <= self.width or len(buffer) != 8*rows*self.width:
            raise self.failure()
        factor = torch.frombuffer(bytearray(buffer), dtype=torch.float64).reshape(rows, self.width)
        _finite(factor)
        return n, factor

    def _put(self, period, entry):
        size = 256+entry[1].numel()*8
        if period in self.cache:
            self.accounted -= 256+self.cache.pop(period)[1].numel()*8
        while self.cache and self.accounted+size > self.cache_bytes:
            old_period, old_entry = self.cache.popitem(last=False)
            self.accounted -= 256+old_entry[1].numel()*8
            self._write(old_period, old_entry)
        if size > self.cache_bytes:
            self._write(period, entry)
        else:
            self.cache[period] = entry
            self.accounted += size
            self.peak = max(self.peak, self.accounted)

    def add(self, periods, units, values):
        """The Python loop is over periods; each numerical update is native QR."""
        try:
            self.connection.executemany("INSERT INTO seen VALUES(?,?)", zip(units, periods, strict=True))
            axis = torch.tensor(periods, dtype=torch.int64)
            distinct, index = torch.unique(axis, sorted=True, return_inverse=True)
            order = torch.argsort(index, stable=True)
            sizes = torch.bincount(index, minlength=len(distinct))
            starts = sizes.cumsum(0)-sizes
            for period, start, count in zip(distinct.tolist(), starts.tolist(), sizes.tolist(), strict=True):
                block = values[order[start:start+count]]
                previous = self._get(period)
                n = count
                if previous is not None:
                    n += previous[0]
                    block = torch.cat((previous[1], block))
                factor = torch.linalg.qr(block, mode="r")[1]
                _finite(factor)
                self._put(period, (n, factor))
        except sqlite3.IntegrityError as error:
            raise AnalysisError("repeated_time_values", "More than one retained observation has the same panel and time.") from error
        except sqlite3.Error as error:
            raise self.failure() from error

    def finish(self):
        try:
            for period, entry in self.cache.items():
                self._write(period, entry)
            self.cache.clear()
            self.accounted = 0
            self.connection.commit()
            self.count = self.connection.execute("SELECT COUNT(*) FROM periods").fetchone()[0]
            self.groups = self.connection.execute("SELECT COUNT(*) FROM (SELECT k FROM seen GROUP BY k)").fetchone()[0]
            self.scratch_bytes = self.path.stat().st_size
        except sqlite3.Error as error:
            raise self.failure() from error

    def factors(self):
        try:
            cursor = self.connection.execute("SELECT t,n,r,factor FROM periods ORDER BY t")
            for period, n, rows, buffer in cursor:
                if not 1 <= rows <= self.width or len(buffer) != 8*rows*self.width:
                    raise self.failure()
                factor = torch.frombuffer(bytearray(buffer), dtype=torch.float64).reshape(rows, self.width)
                _finite(factor)
                yield period, n, factor
        except sqlite3.Error as error:
            raise self.failure() from error

    def diagnostics(self):
        return {"period_factor_cache_bytes": self.cache_bytes, "peak_period_cache_bytes": self.peak,
                "period_factor_writes": self.writes, "period_factor_scratch_bytes": self.scratch_bytes,
                "period_factor_disk_complexity": "O(periods*parameters**2 + retained_rows*panel_label_bytes)",
                "period_factor_memory_complexity": "O(batch_rows*parameters + parameters**2 + fixed cache)",
                "period_estimates_materialized": False}

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def _period_fit(period, n, factor, k):
    if n <= k:
        raise AnalysisError("insufficient_observations", f"Period {period} has {n} observation(s) for {k} parameters; every period needs more observations.")
    try:
        beta, _, condition = _solve(factor, k)
    except AnalysisError as error:
        raise AnalysisError(error.code, f"Period {period}: {error}") from error
    column, response = factor[:, 0], factor[:, -1]
    centered = response-column*((column@response)/(column@column))
    tss = float(centered@centered)
    # The trailing augmented QR column retains all residual energy. Computing
    # y-Xb again on artificial rows would add avoidable rounding at exact fits.
    rss = float(response[k:]@response[k:])
    return beta, rss, tss, condition


def _fit(spec, source, batch_rows):
    notes = _Notes(spec)
    if spec.estimator != "xtfmb" or not spec.intercept or not spec.panel or not spec.time:
        raise AnalysisError("invalid_spec", "Fama-MacBeth replay requires estimator='xtfmb', panel, time and a constant.")
    if spec.covariance not in {"nonrobust", "hac"} or spec.weights or spec.cluster:
        raise AnalysisError("invalid_spec", "Fama-MacBeth replay supports nonrobust/Bartlett HAC covariance without observation weights or clusters.")
    if spec.covariance != "hac" and "lags" in spec.options:
        raise AnalysisError("invalid_spec", "lags applies to covariance='hac' only.")
    selected_lags = notes.option("lags")
    if selected_lags is not None and (type(selected_lags) is not int or selected_lags < 0):
        raise AnalysisError("invalid_lags", "Fama-MacBeth HAC lags must be a nonnegative integer.")
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean")
    sample.prepare()
    k = len(design.terms)
    if not k or design.terms[0] != "Intercept":
        raise AnalysisError("singular_design", "Fama-MacBeth needs a retained constant.")
    resource = sample.plan_rows("native Fama-MacBeth period replay", {
        "period_factor_LRU": _PeriodFactors.cache_bytes,
        "period_SQLite_cache": 2*1024**2,
        "period_factor_and_parameter_moments": 128*(k+1)**2,
    }, 128*(k+1)+512)
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        moments.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
    y_anchor, y_magnitude = float(moments.anchor[1]), float(moments.magnitude[1])
    y_center = float(moments.mean[1])
    y_mean = y_anchor+y_magnitude*y_center
    y_scale = y_magnitude*math.sqrt(float(moments.m2.value[1]/moments.mass))
    if not math.isfinite(y_scale) or y_scale <= 0:
        raise AnalysisError("constant_outcome", "The Fama-MacBeth outcome has no finite variation.")

    def y_work(batch):
        return ((batch.numeric(spec.outcome)-y_anchor)/y_magnitude-y_center)*(y_magnitude/y_scale)

    datetime = pd.api.types.is_datetime64_any_dtype(sample.sample[spec.time].dtype)
    if datetime:
        notes.warn(f"Datetime column '{spec.time}' is treated as consecutive periods in sorted order; gaps between dates are not detected.")
    with ExitStack() as stack:
        store = _PeriodFactors(k+1)
        stack.callback(store.close)
        for batch in sample.batches():
            try:
                periods = [int(value) for value in _period_keys(batch.frame[spec.time])]
            except (OverflowError, ValueError) as error:
                raise AnalysisError("invalid_time", "Fama-MacBeth periods must fit signed int64 nanosecond/integer coordinates.") from error
            units = encode_cluster_labels(batch.frame[spec.panel])
            store.add(periods, units, torch.cat((batch.designs["mean"], y_work(batch)[:, None]), 1))
        store.finish()
        count = store.count
        if count < 2:
            raise AnalysisError("insufficient_periods", "Fama-MacBeth needs at least two periods.")
        parameter_moments = _CrossMoments(k)
        rss, tss, r2_total = _CompensatedSum(()), _CompensatedSum(()), _CompensatedSum(())
        valid_r2 = 0
        minimum, maximum, worst_condition = sample.nrows, 0, 0.
        for period, n, factor in store.factors():
            beta, residual, centered, condition = _period_fit(period, n, factor, k)
            parameter_moments.add(beta[None, :], torch.ones(1, dtype=torch.float64))
            rss.add(torch.tensor(residual, dtype=torch.float64))
            tss.add(torch.tensor(centered, dtype=torch.float64))
            if centered > 0:
                r2_total.add(torch.tensor(1-residual/centered, dtype=torch.float64))
                valid_r2 += 1
            minimum, maximum = min(minimum, n), max(maximum, n)
            worst_condition = max(worst_condition, condition)
        scale_squared = y_scale*y_scale
        raw_scale = sample.nrows*(scale_squared+y_mean*y_mean)
        if not math.isfinite(raw_scale):
            raise AnalysisError("non_finite_result", "Fama-MacBeth outcome sums of squares exceed representable float64 units.")
        require_residual_variation(float(rss.value)*scale_squared, float(tss.value)*scale_squared,
                                   raw_scale, spec.outcome, "the period cross-sections")
        beta_work = parameter_moments.anchor+parameter_moments.mean
        spread_work = parameter_moments.m2.value
        nonrobust_work = spread_work/(count*(count-1))
        info = {"covariance": spec.covariance, "df_inference": count-1,
                "df_resid": count-1, "periods": count,
                "correction": "Fama-MacBeth: sd of the period estimates / sqrt(T)"}
        v_work = nonrobust_work
        if spec.covariance == "hac":
            lags = newey_west_lags(count) if selected_lags is None else selected_lags
            # Account for shared replay, SQLite/factor caches AND the helper's
            # lag state together. The helper's own limit is not a process cap.
            available = sample.workspace_budget-resource.estimated_bytes
            acc = HACAccumulator(k, lags, "bartlett", budget_bytes=available)
            stack.callback(acc.close)
            combined = plan_workspace("Fama-MacBeth shared replay and HAC", {
                "shared_replay_and_period_buffers": resource.estimated_bytes,
                "HAC_lag_and_SQLite_buffers": acc.plan.estimated_bytes,
            }, budget_bytes=sample.workspace_budget)
            # Bounded blocks only; no T-by-K estimates or period label vector.
            scores, periods = [], []

            def flush():
                if not scores:
                    return
                time = (pd.Series(pd.to_datetime(periods, utc=True)) if datetime
                        else pd.Series(periods, dtype="int64"))
                acc.add(torch.stack(scores), torch.arange(len(scores)), time=time)
                scores.clear()
                periods.clear()

            for period, n, factor in store.factors():
                estimate, _, _, _ = _period_fit(period, n, factor, k)
                scores.append(estimate-beta_work)
                periods.append(period)
                if len(scores) == 4096:
                    flush()
            flush()
            v_work = acc.finish()*(count/(count-1))/count**2
            info.update({"correction": f"Newey-West over periods (bartlett, {lags} lags): T/(T-1)",
                         "lags": lags, "kernel": "bartlett", "small_sample_correction": count/(count-1),
                         "HAC_resource_plan": combined.record(), "HAC_spill": acc.diagnostics})
        beta = y_scale*(design.transform@beta_work)
        beta[0] += y_mean
        covariance = scale_squared*(design.transform@v_work@design.transform.T)
        spread = scale_squared*(design.transform@spread_work@design.transform.T)
        _finite(beta, covariance, spread)
        predictions = []
        for batch in sample.batches():
            take = min(400-len(predictions), len(batch.frame))
            observed = batch.numeric(spec.outcome)
            fitted = y_mean+y_scale*(batch.designs["mean"]@beta_work)
            for row, outcome, value in zip(batch.positions[:take].tolist(), observed[:take].tolist(), fitted[:take].tolist(), strict=True):
                predictions.append({"row": row, "observed": outcome, "fitted": value, "residual": outcome-value})
        test = wald_test(beta, covariance, range(1, k), df_resid=count-1, label="F test of the slopes")
        info["period_factor_spill"] = store.diagnostics()
        return _result(sample, terms=design.terms, beta=beta, covariance=covariance, info=info,
                   metrics={"r_squared": float(r2_total.value)/valid_r2 if valid_r2 else None,
                            "n_periods": count, "n_groups": store.groups,
                            "obs_per_period_min": minimum, "obs_per_period_avg": sample.nrows/count,
                            "obs_per_period_max": maximum}, notes=notes, predictions=predictions,
                   tests={"model": test}, solver="native_period_augmented_TSQR_disk_replay",
                   diagnostics={"condition_number": worst_condition, "rank": k,
                                "condition_number_basis": "maximum unit-norm period QR design",
                                **store.diagnostics()},
                   extra={"period_estimates_sd": (spread.diagonal()/(count-1)).clamp_min(0).sqrt().tolist()},
                   resource=resource.record(), title="Fama-MacBeth regression")


def fit_streaming_fmb(spec, source, *, batch_rows=None):
    """Verified Dataset adapter; no collection fallback or arbitrary row cap."""
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("numerical_failure", "Native period QR could not produce a finite, full-rank Fama-MacBeth result.") from error
