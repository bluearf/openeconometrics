"""Native dynamic-panel equations from exact disk ordering and global factors.

Anderson--Hsiao first differences use SQL lag windows over all retained
panels, global TSQR, and full panel score sums. No whole panel or sample
needs to reside in memory. Further GMM adapters are registered separately.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .iv import kernels as iv
from .iv.common import check_fit
from .replay_sample import ReplaySample
from .streaming_linear import _Notes, _finite, _result, _scratch_directory

SUPPORTED = frozenset({"ahreg"})


def supports(spec):
    return spec.estimator in SUPPORTED


def _periods(values):
    if pd.api.types.is_datetime64_any_dtype(values.dtype) or pd.api.types.is_bool_dtype(
        values.dtype
    ):
        raise AnalysisError(
            "invalid_time",
            "Dynamic panel lag windows require exact integer periods; convert dates to explicit regular period codes first.",
        )
    output = []
    for value in values:
        if (
            not math.isfinite(value)
            or value != int(value)
            or not -(1 << 63) <= int(value) < (1 << 63)
        ):
            raise AnalysisError(
                "invalid_time", "Dynamic panel periods must be exact signed int64 integers."
            )
        output.append(int(value))
    return output


class _PanelStore:
    """Owned SQL ordering; native equations are read in bounded numerical blocks."""

    def __init__(self, sample, names):
        self.sample, self.names = sample, list(names)
        self.scratch = self.db = None
        self.width = len(names)
        parent = _scratch_directory()
        self.disk_estimate = sample.nrows * (self.width * 64 + 384) * 4
        if (
            self.disk_estimate + 64 * 1024**2
            > shutil.disk_usage(parent or tempfile.gettempdir()).free
        ):
            raise AnalysisError(
                "insufficient_scratch_space",
                "Full dynamic-panel ordering and equations need more free owned temporary disk space.",
            )
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-dpanel-", dir=parent)
            self.path = Path(self.scratch.name) / "panel.sqlite3"
            self.db = sqlite3.connect(self.path)
            for pragma in (
                "journal_mode=OFF",
                "synchronous=OFF",
                "cache_size=-4096",
                "temp_store=FILE",
                "mmap_size=0",
            ):
                self.db.execute("PRAGMA " + pragma)
            self.db.execute(
                "CREATE TABLE raw(position INTEGER PRIMARY KEY,key BLOB,t INTEGER,"
                + ",".join(f"v{i} REAL" for i in range(self.width))
                + ")"
            )
            self.db.execute("CREATE UNIQUE INDEX ordered_panel ON raw(key,t)")
            self.path.chmod(0o600)
        except BaseException:
            self.close()
            raise

    def seed(self):
        sample = self.sample
        for batch in sample.batches():
            keys = encode_cluster_labels(batch.frame[sample.spec.panel])
            times = _periods(batch.frame[sample.spec.time])
            values = torch.stack([batch.numeric(name) for name in self.names], 1)
            try:
                self.db.executemany(
                    "INSERT INTO raw VALUES(" + ",".join("?" for _ in range(3 + self.width)) + ")",
                    (
                        (int(position), key, time, *value)
                        for position, key, time, value in zip(
                            batch.positions.tolist(), keys, times, values.tolist(), strict=True
                        )
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise AnalysisError(
                    "repeated_time_values", "Dynamic-panel periods are repeated within a panel."
                ) from error
        self.db.commit()

    def ah_equations(self, instrument):
        lag = 2 if instrument == "levels" else 3
        windows = [
            *(f"LAG(t,{j}) OVER w AS t{j}" for j in range(1, lag + 1)),
            *(f"LAG(v{i},1) OVER w AS p{i}" for i in range(self.width)),
            "LAG(v0,2) OVER w AS y2",
            *(["LAG(v0,3) OVER w AS y3"] if lag == 3 else []),
        ]
        self.db.execute(
            "CREATE TABLE equations AS WITH windowed AS (SELECT *,"
            + ",".join(windows)
            + " FROM raw WINDOW w AS(PARTITION BY key ORDER BY t)) SELECT position,key,t,v0-p0 AS y,p0-y2 AS dlag,"
            + ("y2" if instrument == "levels" else "y2-y3")
            + " AS instrument"
            + "".join(f",v{i}-p{i} AS x{i - 1}" for i in range(1, self.width))
            + " FROM windowed WHERE "
            + " AND ".join(f"t-t{j}={j}" for j in range(1, lag + 1))
        )
        self.db.execute("CREATE UNIQUE INDEX ordered_equations ON equations(key,t)")
        self.db.commit()
        count = self.db.execute("SELECT COUNT(*) FROM equations").fetchone()[0]
        if count <= self.width:
            raise AnalysisError(
                "insufficient_observations",
                "Anderson--Hsiao needs more complete consecutive lag windows than coefficients.",
            )
        groups, minimum, maximum = self.db.execute(
            "SELECT COUNT(*),MIN(n),MAX(n) FROM(SELECT COUNT(*) AS n FROM equations GROUP BY key)"
        ).fetchone()
        if groups < 2:
            raise AnalysisError(
                "insufficient_clusters",
                "Anderson--Hsiao needs at least two panels with usable differenced equations.",
            )
        self.count, self.groups, self.minimum, self.maximum = count, groups, minimum, maximum

    def blocks(self):
        names = ["y", "dlag", "instrument", *(f"x{i}" for i in range(self.width - 1))]
        cursor = self.db.execute(
            "SELECT position,key,t," + ",".join(names) + " FROM equations ORDER BY key,t"
        )
        try:
            while rows := cursor.fetchmany(self.sample.rows):
                self.sample.actual_numeric_peak_rows = max(
                    self.sample.actual_numeric_peak_rows, len(rows)
                )
                value = torch.tensor([row[3:] for row in rows], dtype=torch.float64)
                _finite(value)
                yield (
                    [row[1] for row in rows],
                    torch.tensor([row[2] for row in rows], dtype=torch.int64),
                    value,
                    [row[0] for row in rows],
                )
        finally:
            cursor.close()

    def position_hash(self):
        digest = hashlib.sha256()
        cursor = self.db.execute("SELECT position FROM equations ORDER BY position")
        while rows := cursor.fetchmany(self.sample.rows):
            digest.update(
                torch.tensor([row[0] for row in rows], dtype=torch.int64)
                .contiguous()
                .numpy()
                .tobytes()
            )
        return digest.hexdigest()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def _ah_covariances(sample, store, projection, estimate):
    k, n = store.width, store.count
    first = projection.first
    ssr = _CompensatedSum(())
    tss = _CompensatedSum(())
    ordinary = _CompensatedSum((k, k))
    adjacent = _CompensatedSum((k, k))
    predictions, previous = [], None
    with ExitStack() as stack:
        score_groups = ClusterAccumulator(k, scratch_directory=_scratch_directory())
        first_groups = ClusterAccumulator(k, scratch_directory=_scratch_directory())
        stack.callback(score_groups.close)
        stack.callback(first_groups.close)
        for keys, periods, value, positions in store.blocks():
            y, lagged, excluded, x = value[:, 0], value[:, 1], value[:, 2], value[:, 3:]
            z = torch.cat((x, excluded[:, None]), 1)
            fitted_lag = z @ first.beta[:, 0]
            score_x = torch.cat((x, fitted_lag[:, None]), 1)
            mean = torch.cat((x, lagged[:, None]), 1) @ estimate.beta
            residual = y - mean
            ssr.add(residual @ residual)
            tss.add(y @ y)
            ordinary.add(score_x.T @ score_x)
            if len(keys) > 1:
                same = torch.tensor(
                    [a == b for a, b in zip(keys[1:], keys[:-1], strict=True)], dtype=torch.bool
                )
                same &= periods[1:] - periods[:-1] == 1
                adjacent.add(score_x[1:][same].T @ score_x[:-1][same])
            if (
                previous is not None
                and previous[0] == keys[0]
                and int(periods[0]) - previous[1] == 1
            ):
                adjacent.add(score_x[0][:, None] @ previous[2][None, :])
            previous = keys[-1], int(periods[-1]), score_x[-1].clone()
            score_groups.add(keys, score_x * residual[:, None])
            first_groups.add(keys, z * (lagged - fitted_lag)[:, None])
            take = min(400 - len(predictions), len(y))
            predictions.extend(
                {"row": int(position), "observed": float(a), "fitted": float(b),
                 "predicted": float(b), "residual": float(a - b)}
                for position, a, b in zip(positions[:take], y[:take], mean[:take], strict=True)
            )
        rss, total = float(ssr.value), float(tss.value)
        if not all(math.isfinite(value) for value in (rss, total)):
            raise AnalysisError(
                "precision_unsupported",
                "Differenced equation sums of squares exceed finite float64; rescale outcome units.",
            )
        check_fit(rss, total, n - k, total)
        meat, g = score_groups.finish()
        cr1 = g / (g - 1) * (n - 1) / (n - k)
        first_meat, _ = first_groups.finish()
        first_cov = first.xtx_inv @ (first_meat * cr1) @ first.xtx_inv
        first_test = wald_test(
            first.beta[:, 0],
            first_cov,
            [k - 1],
            df_resid=g - 1,
            label="Panel-clustered excluded-instrument F",
        )
        if sample.spec.covariance == "nonrobust":
            variance = rss / (2 * (n - k))
            matrix = 2 * ordinary.value - adjacent.value - adjacent.value.T
            covariance = estimate.bread @ (matrix * variance) @ estimate.bread
            info = {
                "covariance": "nonrobust",
                "df_inference": n - k,
                "correction": "homoskedastic level errors: RSS/[2(N-K)] * B Xhat' H Xhat B",
                "small_sample_correction": None,
                "level_error_variance": variance,
                "error_covariance": "H: diagonal2; -1 for consecutive equations in the same panel",
            }
        else:
            covariance = estimate.bread @ (meat * cr1) @ estimate.bread
            info = {
                "covariance": "cluster",
                "df_inference": g - 1,
                "cluster_count": g,
                "cluster_column": sample.spec.panel,
                "cluster_columns": [sample.spec.panel],
                "cluster_df": g - 1,
                "small_sample_correction": cr1,
                "correction": "cluster sandwich: G/(G-1)*(N-1)/(N-K)",
            }
        _finite(covariance, first_cov)
        return (covariance + covariance.T) / 2, info, rss, total, predictions, first_test


def _fit_ahreg(spec, source, *, batch_rows):
    declared = registry.cluster_columns(spec)
    if declared and declared != [spec.panel]:
        raise AnalysisError("invalid_spec", "Anderson--Hsiao clusters only on its model panel.")
    notes = _Notes(spec)
    sample = ReplaySample(spec, source, batch_rows=batch_rows).prepare()
    k = len(spec.predictors) + 1
    resource = sample.plan_rows(
        "Anderson--Hsiao sorted equations, TSQR and panel covariance",
        {
            "panel_SQLite_cache": 4 * 1024**2,
            "panel_score_caches": 12 * 1024**2,
            "global_IV_factors": 1024 * (k + 3) ** 2,
        },
        256 * (k + 6),
    )
    instrument = notes.option("instrument")
    with ExitStack() as stack:
        store = _PanelStore(sample, [spec.outcome, *spec.predictors])
        stack.callback(store.close)
        store.seed()
        store.ah_equations(instrument)
        tree = _TSQRTree()
        for _, _, values, _ in store.blocks():
            # Included exogenous x, excluded z, endogenous D.L.y and D.y.
            ordered = torch.cat((values[:, 3:], values[:, 2:3], values[:, 1:2], values[:, :1]), 1)
            tree.add(torch.linalg.qr(ordered, mode="r")[1])
        factor = tree.finish()
        x, z, lagged, y = (
            factor[:, : k - 1],
            factor[:, k - 1 : k],
            factor[:, k : k + 1],
            factor[:, k + 1],
        )
        _, omitted = collinear_columns(torch.cat((x, z), 1))
        if omitted:
            names = [spec.predictors[i] for i in omitted if i < k - 1]
            raise AnalysisError(
                "singular_design" if names else "underidentified",
                "Differenced predictors are time-invariant/collinear or the AH instrument carries no independent variation.",
            )
        projection = iv.project(y, x, lagged, z)
        estimate = iv.k_class(projection)
        covariance, info, rss, tss, predictions, first_test = _ah_covariances(
            sample, store, projection, estimate
        )
        term = f"L1.{spec.outcome}"
        excluded_term = f"L2.{spec.outcome}" if instrument == "levels" else f"D.L2.{spec.outcome}"
        first_rss, partial_rss = float(projection.first.ssr[0]), float(projection.partial.ssr[0])
        first_stage = {
            "term": term,
            "excluded_instrument": excluded_term,
            "partial_r_squared": 1 - first_rss / partial_rss if partial_rss > 0 else None,
            "excluded_coefficient": float(projection.first.beta[-1, 0]),
            "f_statistic": first_test["statistic"],
            "df": first_test["df"],
            "df2": first_test.get("df2"),
            "p_value": first_test["p_value"],
            "covariance": "cluster",
            "cluster_column": spec.panel,
            "note": "Relevance diagnostic; not an AR/CLR weak-instrument-robust inference test.",
        }
        order = torch.tensor([k - 1, *range(k - 1)], dtype=torch.int64)
        beta, covariance = estimate.beta[order], covariance[order][:, order]
        count = store.count
        omitted_rows = sample.nrows - count
        notes.warn(
            f"Excluded {omitted_rows} observation(s) without the required consecutive lag window for instrument='{instrument}'."
        )
        positions_hash = store.position_hash()
        for _ in sample.batches():
            pass
        info.update(
            {
                "covariance": spec.covariance,
                "df_resid": count - k,
                "effective_covariance": "panel_cluster"
                if spec.covariance != "nonrobust"
                else "homoskedastic_level_ma1",
                "residual_definition": "D.y minus rho * D.L.y minus D.x'b",
                "r_squared_definition": "uncentered, first-difference equation; can be negative",
                "error_variance": "RSS/[2(N-K)]" if spec.covariance == "nonrobust" else None,
            }
        )
        result = _result(
            sample,
            terms=[term, *spec.predictors],
            beta=beta,
            covariance=covariance,
            info=info,
            metrics={
                "r_squared": 1 - rss / tss,
                "rmse": math.sqrt(rss / (count - k)),
                "df_model": k,
                "df_resid": count - k,
                "n_groups": store.groups,
                "n_instruments": k,
                "n_endogenous": 1,
                "n_obs_transformed": count,
                "group_min": store.minimum,
                "group_max": store.maximum,
                "group_avg": count / store.groups,
            },
            notes=notes,
            predictions=predictions,
            tests={
                "model": wald_test(
                    beta,
                    covariance,
                    range(k),
                    df_resid=info["df_inference"],
                    label="Joint significance of the differenced equation",
                ),
                "overidentification": {
                    "statistic": None,
                    "df": 0,
                    "p_value": None,
                    "note": "Exactly identified; instrument validity is not tested.",
                },
            },
            solver="native_disk_lag_windows_and_global_TSQR_two_stage",
            diagnostics={
                "condition_number": estimate.condition_number,
                "panel_storage": "owned SQLite ordering; bounded row windows, no whole panel/sample tensor",
                "panel_disk_estimate_bytes": store.disk_estimate,
                "panel_disk_bytes": store.path.stat().st_size,
                "tsqr_reduction_depth": tree.depth,
            },
            extra={
                "instrument": instrument,
                "endogenous": [term],
                "instruments": [excluded_term, *(f"D.{name}" for name in spec.predictors)],
                "first_stage": [first_stage],
                "equations": "difference",
                "excluded_lag_window_rows": omitted_rows,
                "missing_rows": sample.original_count - sample.nrows,
            },
            resource=resource.record(),
            title="Anderson--Hsiao dynamic panel IV",
            use_t=True,
            provenance_extra={
                "transformation": "first differences",
                "equations": "difference",
                "dynamic_order": 1,
                "instrument": excluded_term,
                "predictor_exogeneity": "strictly exogenous with respect to all level errors",
                "instrument_validity_assumption": "level errors serially uncorrelated; lagged y uncorrelated with future errors; independent panels",
                "lag_semantics": "integer periods, step1; no cross-panel or across-gap lags",
                "derivatives": "closed form (linear IV)",
                "dense_observation_matrices": False,
                "sample_position_count": count,
                "source_complete_rows": sample.nrows,
                "effective_equation_rows": count,
                "prediction_sample": "first400 sorted complete differenced equations",
                "sample_positions_hash": positions_hash,
                "sample_hash": hashlib.sha256(
                    (sample.baseline["data_hash"] + positions_hash).encode()
                ).hexdigest(),
                "sample_order": "sorted panel key bytes then exact integer period; selection digest in physical input order",
            },
        )
        result.nobs = count
        result.dropped_rows = sample.original_count - count
        return result


def fit_streaming(spec, source, *, batch_rows=None):
    if not supports(spec):
        raise AnalysisError(
            "streaming_unsupported", "No full-source dynamic-panel replay adapter is registered."
        )
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit_ahreg(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError(
            "dynamic_panel_spill_failed",
            "Dynamic-panel replay needs writable temporary storage and enough free disk space.",
        ) from error
