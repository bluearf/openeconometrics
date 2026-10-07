"""Panel first-difference 2SLS with bounded owned chronological row storage.

Only genuinely consecutive retained observations are differenced; panel/time
keys and cross-block state are exact. Native global QR factors preserve linear
geometry, while covariance scores are replayed on the actual differenced rows.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
import shutil
import sqlite3
import struct
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .iv.common import check_fit
from .replay_sample import ReplayBatch
from .streaming_fe_iv import _ActualRows, _fit_geometry, _raw, _response, _sample, _screen
from .streaming_linear import _CrossMoments, _Notes, _covariance, _finite, _result, _scratch_directory
from .streaming_panelgls import _times


def _check_disk_space(sample, width, clusters, operation):
    """Conservative per-source-row spill estimate before creating any directory.

    Observed reporting key widths, a separate index/record reserve and a
    multiplier cover ordinary SQLite overhead. Wider later keys and concurrent
    external disk writes remain possible; this is a plan, not a guarantee.
    """
    columns = list(dict.fromkeys([sample.spec.panel, *clusters]))
    key_bytes = sum(max((len(key) for key in encode_cluster_labels(sample.sample[name])), default=0) for name in columns)
    required = 2*sample.nrows*(8*width+2*key_bytes+512)
    free = shutil.disk_usage(_scratch_directory() or tempfile.gettempdir()).free
    plan = {"estimated_scratch_bytes": required, "available_bytes": free,
            "source_row_upper_bound": sample.nrows, "numeric_columns": width,
            "observed_reporting_key_bytes": key_bytes, "reserve_bytes": 64*1024**2,
            "scope": "owned numeric/key records and index/page reserve; later wider keys, filesystem overhead and concurrent external writes can still fail"}
    if required+plan["reserve_bytes"]>free:
        error = AnalysisError("replay_disk_limit", f"{operation} needs an estimated {required:,} additional scratch bytes plus a reserve; available free disk is {free:,} bytes. Select a larger scratch volume. This is separate from the RAM workspace budget.")
        error.disk_plan = plan
        raise error
    return plan


class _Differences:
    """Immutable source snapshot plus one preceding row, never an entire panel."""
    def __init__(self, sample, response, k1, clusters):
        self.sample, self.clusters = sample, clusters
        self.width = len(sample.designs["regressors"].terms)+len(sample.designs["instruments"].terms)-k1
        self.scratch = self.db = None
        self.passes, self.kept, self.peak_rows = 0, 0, 0
        self.weight_scale, self.weight_mass = 1., 0.
        try:
            self.disk_plan = _check_disk_space(sample, self.width, clusters, "First-difference panel IV snapshot")
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-panel-differences-", dir=_scratch_directory())
            self.path = Path(self.scratch.name)/"rows.sqlite3"
            self.db = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-2048", "temp_store=FILE", "mmap_size=0"):
                self.db.execute("PRAGMA "+pragma)
            self.db.execute("CREATE TABLE rows(pos INTEGER PRIMARY KEY,g BLOB,t INTEGER,data BLOB,labels BLOB,weight REAL)")
            for batch in sample.batches():
                periods, self.date = _times(batch.frame[sample.spec.time])
                keys = encode_cluster_labels(batch.frame[sample.spec.panel])
                labels = [encode_cluster_labels(batch.frame[name]) for name in clusters]
                from .streaming_linear import _weights
                weights = _weights(sample, batch).tolist()
                values = _raw(batch, response, k1)
                _finite(values)
                packed = memoryview(values.contiguous().numpy().tobytes())
                stride = 8*self.width
                self.db.executemany("INSERT INTO rows VALUES(?,?,?,?,?,?)", ((int(pos), key, period, packed[i*stride:(i+1)*stride],
                    b"".join(len(column[i]).to_bytes(8, "big")+column[i] for column in labels), weight)
                    for i, (pos, key, period, weight) in enumerate(zip(batch.positions.tolist(), keys, periods, weights, strict=True))))
            try:
                self.db.execute("CREATE UNIQUE INDEX panel_time ON rows(g,t)")
            except sqlite3.IntegrityError as error:
                raise AnalysisError("repeated_time_values", "More than one retained row has the same panel and time.") from error
            self.db.execute("CREATE TABLE periods(t INTEGER PRIMARY KEY,rank INTEGER)")
            self.db.execute("INSERT INTO periods SELECT t,DENSE_RANK() OVER(ORDER BY t)-1 FROM rows GROUP BY t")
            self.db.execute("CREATE TABLE used_panels(g BLOB PRIMARY KEY,n INTEGER,mass REAL) WITHOUT ROWID")
            digest = hashlib.sha256()
            for batch, _ in self.batches():
                digest.update(b"".join(struct.pack("<q", int(pos)) for pos in batch.positions.tolist()))
                keys = batch.cluster_panel_keys
                counts = {}
                for key, weight in zip(keys, _weights(sample, batch).tolist(), strict=True):
                    old_n, old_mass = counts.get(key, (0, 0.))
                    counts[key] = old_n+1, old_mass+weight
                self.db.executemany("INSERT INTO used_panels VALUES(?,?,?) ON CONFLICT(g) DO UPDATE SET n=n+excluded.n,mass=mass+excluded.mass",
                                    ((key, count, mass) for key, (count, mass) in counts.items()))
                self.kept += len(batch.positions)
                self.weight_mass += float(_weights(sample, batch).sum())
            if not self.kept:
                raise AnalysisError("empty_sample", "No panel has two consecutive periods; first differences cannot be formed.")
            self.positions_hash = digest.hexdigest()
            self.nobs = int(round(self.weight_mass)) if sample.spec.weight_type == "fweight" else self.kept
            if sample.spec.weight_type in {"aweight", "pweight"}:
                self.weight_scale = self.kept/self.weight_mass
            size = "mass" if sample.spec.weight_type == "fweight" else "n"
            g, low, high, total = self.db.execute(f"SELECT COUNT(*),MIN({size}),MAX({size}),SUM({size}) FROM used_panels").fetchone()
            self.structure = {"n_groups": g, "t_min": float(low), "t_avg": total/g, "t_max": float(high)}
            self.db.commit()
            self.path.chmod(0o600)
        except BaseException:
            self.close()
            raise

    def batches(self):
        cursor = self.db.execute("SELECT pos,g,t,data,labels,weight FROM rows ORDER BY g,t")
        previous, previous_group, previous_time = None, None, None
        while records := cursor.fetchmany(self.sample.rows):
            data = torch.frombuffer(bytearray(b"".join(row[3] for row in records)), dtype=torch.float64).reshape(-1, self.width)
            prior = torch.cat((data[:1] if previous is None else previous[None], data[:-1]), 0)
            periods = {}
            if self.date:
                unique = list(dict.fromkeys([row[2] for row in records]+([previous_time] if previous_time is not None else [])))
                for start in range(0, len(unique), 500):
                    part = unique[start:start+500]
                    periods.update(self.db.execute("SELECT t,rank FROM periods WHERE t IN ("+",".join("?" for _ in part)+")", part))
            alive = []
            for row in records:
                consecutive = previous_group==row[1] and ((periods[row[2]]-periods[previous_time]) if self.date else row[2]-previous_time)==1
                alive.append(consecutive)
                previous_group, previous_time = row[1], row[2]
            valid = torch.tensor(alive, dtype=torch.bool)
            values = (data-prior)[valid]
            selected = [row for row, keep in zip(records, alive, strict=True) if keep]
            previous = data[-1].clone()
            self.peak_rows = max(self.peak_rows, len(records))
            if selected:
                weights = torch.tensor([row[5] for row in selected], dtype=torch.float64)*self.weight_scale
                # _weights applies pweight normalization at the facade level.
                if self.sample.spec.weight_type == "pweight":
                    weights *= self.sample.weight_mean
                batch = ReplayBatch(pd.DataFrame(index=range(len(values))), weights,
                                    torch.tensor([row[0] for row in selected], dtype=torch.int64), {})
                batch.cluster_panel_keys = [row[1] for row in selected]
                batch.level_response_work = data[valid, 0]
                batch.cluster_keys = [[] for _ in self.clusters]
                for row in selected:
                    blob, offset = row[4], 0
                    for column in batch.cluster_keys:
                        length = int.from_bytes(blob[offset:offset+8], "big")
                        column.append(blob[offset+8:offset+8+length])
                        offset += 8+length
                yield batch, values
        self.passes += 1

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def fit_streaming_fd_iv(spec, source, *, batch_rows=None):
    """Native global xtivreg FD, all current unweighted covariance/small options."""
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("singular_design", "The first-difference IV factors are numerically singular.") from error
    except (OSError, sqlite3.Error) as error:
        raise AnalysisError("panel_difference_spill_failed", "First-difference replay needs writable temporary storage and enough free disk space.") from error


def _fit(spec, source, *, batch_rows):
    notes = _Notes(spec)
    if spec.estimator!="xtivreg" or notes.option("model")!="fd":
        raise AnalysisError("invalid_spec", "This replay kernel requires xtivreg model='fd'.")
    if spec.time is None or notes.option("ec2sls"):
        raise AnalysisError("invalid_spec", "First-difference IV needs a time column; ec2sls applies only to RE.")
    if spec.weights is not None:
        raise AnalysisError("unsupported_weights", "xtivreg first differences currently do not accept observation weights.")
    if spec.covariance not in {"nonrobust", "robust", "cluster"}:
        raise AnalysisError("unsupported_covariance", "First-difference IV supports nonrobust, panel robust and one cluster column.")
    clusters = ([spec.panel] if spec.covariance=="robust" else registry.cluster_columns(spec) if spec.covariance=="cluster" else [])
    if len(clusters)>1:
        raise AnalysisError("unsupported_cluster_dimensions", "xtivreg currently supports one covariance cluster column.")
    sample, x, z, original_k1 = _sample(spec, source, batch_rows)
    width = len(x.terms)+len(z.terms)-original_k1
    resource = sample.plan_rows("global first-difference IV ordered rows and native TSQR", {
        "ordered_panel_SQLite_cache_and_decode": 4*1024**2,
        "bounded_actual_score_cluster_SQLite_cache": 6*1024**2 if clusters else 0,
        "global_joint_IV_factor_and_covariance": 4096*(width+2)**2,
    }, 768*(width+2))
    response, _, scale = _response(sample)
    if pd.api.types.is_datetime64_any_dtype(sample.sample[spec.time].dtype):
        notes.warn(f"Datetime column '{spec.time}' is treated as consecutive periods in sorted order; gaps between dates are not detected.")
    with ExitStack() as stack:
        states = _Differences(sample, response, original_k1, clusters)
        stack.callback(states.close)
        n, tree = states.kept, _TSQRTree()
        after = _CompensatedSum((width-1,))
        moments = _CrossMoments(1)
        for batch, values in states.batches():
            after.add(values[:, 1:].square().sum(0))
            moments.add(values[:, :1], batch.weights)
            augmented = torch.cat((torch.ones((len(values), 1), dtype=torch.float64), values[:, 1:], values[:, :1]), 1)
            tree.add(torch.linalg.qr(augmented, mode="r")[1])
        factor = tree.finish()
        sx, sz, k1, terms, zterms, omitted = _screen(sample, notes, factor, after.value, after.value,
                                                    x, z, original_k1, True)
        est, _, _, _ = _fit_geometry(factor, sx, sz, k1, terms, zterms, notes, n, 0, True, omitted)
        working, bread = est.beta, est.tsls.bread
        k, q = len(terms), len(terms)-k1
        df = n-k
        if df<=0:
            raise AnalysisError("insufficient_observations", "First-difference IV leaves no residual degrees of freedom.")
        rows = _ActualRows(sample, states.batches, sx, sz, True, n)
        scales = x.scales[x.kept][[i+1 for i in sx]]
        transform = torch.diag(torch.cat((torch.ones(1, dtype=torch.float64), 1/scales)))
        def scores():
            for batch in rows.batches():
                real_x, real_z = batch.designs["regressors"], batch.designs["instruments"]
                projected = torch.cat((real_x[:, :k1], real_z@est.proj.first.beta[:, :q]), 1)
                batch.residual_override = batch.response_work-real_x@working
                yield batch, projected, batch.response_work, batch.response_work*scale, scale
        covariance, info, rss, predictions = _covariance(rows, scores, working, bread, df=df, k=k,
            notes=notes, score_basis=torch.linalg.inv(transform), kind="cluster" if clusters else "nonrobust",
            cluster_columns=clusters, small=True, group_factor=True)
        tss = float(moments.m2.value[0, 0])
        check_fit(rss, tss, df, tss+float(moments.mass)*float(moments.anchor[0]+moments.mean[0])**2)
        beta, covariance = scale*(transform@working), scale*scale*(transform@covariance@transform.T)
        _finite(beta, covariance)
        small = bool(notes.option("small"))
        reference = info["df_inference"] if small else None
        info.update({"covariance": spec.covariance, "small": small, "df_inference": reference, "df_resid": df,
                     "residual_definition": "differenced outcome minus fitted"})
        if spec.covariance=="robust":
            info["correction"] = "vce(robust) = vce(cluster panel); "+info["correction"]
        r2 = 1-rss/tss
        metrics = {"r_squared": r2, "adjusted_r_squared": 1-(1-r2)*(n-1)/df, "rmse": scale*math.sqrt(rss/df),
            "df_model": k-1, "df_resid": df, **states.structure, "n_instruments": len(zterms)-k1, "n_endogenous": q}
        tests = {"model": wald_test(beta, covariance, range(1, k), df_resid=reference,
                    label="F test of the slopes" if small else "Wald chi2 test of the slopes")}
        notes.warn(f"Dropped {sample.nrows-n} observation(s) without an observation in the previous period: first differences use consecutive periods.")
        # Verify the complete raw source after using the immutable disk snapshot.
        for _ in sample.batches():
            pass
        extra = {"model": "fd", "endogenous": terms[k1:], "instruments": zterms[k1:], "omitted_instruments": omitted,
                 "ssr": scale*scale*rss, "differenced_observations": n, "dropped_for_differencing": sample.nrows-n}
        result = _result(sample, terms=terms, beta=beta, covariance=covariance, info=info, metrics=metrics, notes=notes,
            predictions=predictions, tests=tests, solver="native_ordered_first_difference_joint_TSQR_2SLS",
            diagnostics={"ordered_snapshot_passes": states.passes, "maximum_snapshot_rows": states.peak_rows,
                         "joint_TSQR_depth": tree.depth, "first_stage_condition_number": est.proj.first.condition_number,
                         "second_stage_condition_number": est.condition_number}, extra=extra, resource=resource.record(),
            title="First-difference IV regression", use_t=small,
            provenance_extra={"model": "fd", "sample_order": "panel key then exact time; physical source positions retained",
                "sample_position_count": n, "sample_positions_hash": states.positions_hash,
                "sample_hash": hashlib.sha256((sample.baseline["data_hash"]+states.positions_hash+"first_difference").encode()).hexdigest(),
                "hash_scope": "raw projected source plus exact chronological second-row positions; first-difference transformation",
                "source_retained_position_hash": sample.baseline["positions_hash"],
                "prediction_sample": "first400 chronological differenced observations",
                "scratch_disk_plan": states.disk_plan,
                "fd_time_contract": "exact numeric unit gaps; global retained datetime rank"})
        result.nobs = n
        result.dropped_rows = sample.original_count-n
        return result
