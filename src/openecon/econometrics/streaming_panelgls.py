"""Panel GLS/PCSE from owned ordered rows, never a dense periods×panels grid.

Independent/heteroskedastic panels use disk scalar group moments. Correlated
panels genuinely need quadratic covariance factors, checked against the live
workspace budget before allocation. Only one bounded period block is resident.
"""
from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path
import sqlite3
import struct
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from .core import wald_test
from .replay_sample import ReplaySample
from .streaming_linear import _GroupMeans, _Notes, _finite, _result, _scratch_directory, _solve
from .systems.kernels import cholesky_factor

_LOG_2PI = math.log(2*math.pi)


def _times(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        raise AnalysisError("invalid_time", "Boolean values do not define numeric panel periods.")
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return series.dt.as_unit("ns").astype("int64").tolist(), True
    out = []
    for value in series:
        if isinstance(value, bool) or not math.isfinite(value) or value!=int(value) or not -(1<<63)<=int(value)<1<<63:
            raise AnalysisError("invalid_time", "Panel time needs exact signed64 integer periods or declared datetimes.")
        out.append(int(value))
    return out, False


def _sort_key(value, dtype):
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return (int(value.value)+(1<<63)).to_bytes(8, "big")
    if pd.api.types.is_integer_dtype(dtype):
        return (int(value)+(1<<127)).to_bytes(16, "big")
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    bits = (~bits & ((1<<64)-1)) if bits>>63 else bits^(1<<63)
    return bits.to_bytes(8, "big")


class _Panels:
    def __init__(self, sample, scale, center):
        self.sample, self.scale, self.center = sample, scale, center
        self.width = len(sample.designs["mean"].terms)+1
        self.scratch = self.db = None
        self.passes = 0
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-panel-gls-", dir=_scratch_directory())
            self.path = Path(self.scratch.name)/"rows.sqlite3"
            self.db = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-4096", "temp_store=FILE", "mmap_size=0"):
                self.db.execute("PRAGMA "+pragma)
            self.db.execute("CREATE TABLE groups(id INTEGER PRIMARY KEY,key BLOB UNIQUE,sort BLOB,label,n INTEGER DEFAULT 0,rho REAL DEFAULT 0,var REAL DEFAULT 1,last_var REAL DEFAULT 1)")
            self.db.execute("CREATE TABLE rows(pos INTEGER PRIMARY KEY,g INTEGER,t INTEGER,data BLOB)")
            label_dtype = None
            for batch in sample.batches():
                labels = batch.frame[sample.spec.panel]
                if label_dtype is None:
                    label_dtype = labels.dtype
                keys = encode_cluster_labels(labels)
                unique = list(dict.fromkeys(keys))
                first = {key: i for i, key in reversed(list(enumerate(keys)))}
                numeric = pd.api.types.is_numeric_dtype(label_dtype) or pd.api.types.is_datetime64_any_dtype(label_dtype)
                # Group order matches dense sort_panel: numeric labels sorted;
                # text/object labels retain their original first appearance.
                entries = []
                for key in unique:
                    value = labels.iloc[first[key]]
                    if hasattr(value, "item"):
                        value = value.item()
                    sort = _sort_key(value, label_dtype) if numeric else None
                    if isinstance(value, pd.Timestamp):
                        value = value.isoformat()
                    if type(value) is int and not -(1<<63)<=value<1<<63:
                        value = str(value)
                    entries.append((key, sort, value))
                self.db.executemany("INSERT OR IGNORE INTO groups(key,sort,label) VALUES(?,?,?)", entries)
                mapping = {}
                for start in range(0, len(unique), 500):
                    part = unique[start:start+500]
                    query = "SELECT key,id FROM groups WHERE key IN ("+",".join("?" for _ in part)+")"
                    mapping.update(self.db.execute(query, part))
                periods, self.date = _times(batch.frame[sample.spec.time])
                data = torch.cat((batch.designs["mean"], ((batch.numeric(sample.spec.outcome)-center)/scale)[:, None]), 1)
                _finite(data)
                packed = memoryview(data.contiguous().numpy().tobytes())
                stride = 8*self.width
                self.db.executemany("INSERT INTO rows VALUES(?,?,?,?)", ((int(pos), mapping[key], t, packed[i*stride:(i+1)*stride]) for i, (pos, key, t) in enumerate(zip(batch.positions.tolist(), keys, periods, strict=True))))
            self.db.execute("CREATE TABLE sorted_groups AS SELECT ROW_NUMBER() OVER(ORDER BY COALESCE(sort,id),id)-1 AS new_id,id FROM groups")
            self.db.execute("CREATE UNIQUE INDEX sorted_original_id ON sorted_groups(id)")
            self.db.execute("UPDATE rows SET g=(SELECT new_id FROM sorted_groups WHERE id=rows.g)")
            self.db.execute("CREATE TABLE canonical_groups(id INTEGER PRIMARY KEY,key BLOB UNIQUE,sort BLOB,label,n INTEGER,rho REAL,var REAL,last_var REAL)")
            self.db.execute("INSERT INTO canonical_groups SELECT s.new_id,g.key,g.sort,g.label,0,0,1,1 FROM groups g JOIN sorted_groups s ON g.id=s.id")
            self.db.execute("DROP TABLE groups")
            self.db.execute("ALTER TABLE canonical_groups RENAME TO groups")
            self.db.execute("DROP TABLE sorted_groups")
            try:
                self.db.execute("CREATE UNIQUE INDEX panel_time ON rows(g,t)")
            except sqlite3.IntegrityError as error:
                raise AnalysisError("repeated_time_values", "More than one retained row has the same panel and time.") from error
            self.db.execute("CREATE INDEX period_panel ON rows(t,g)")
            self.db.execute("CREATE TABLE periods(t INTEGER PRIMARY KEY,n INTEGER,rank INTEGER)")
            self.db.execute("INSERT INTO periods SELECT t,COUNT(*),DENSE_RANK() OVER(ORDER BY t)-1 FROM rows GROUP BY t")
            self.db.execute("UPDATE groups SET n=(SELECT COUNT(*) FROM rows WHERE g=groups.id)")
            self.groups = int(self.db.execute("SELECT COUNT(*) FROM groups").fetchone()[0])
            self.periods = int(self.db.execute("SELECT COUNT(*) FROM periods").fetchone()[0])
            self.common = int(self.db.execute("SELECT COUNT(*) FROM periods WHERE n=?", (self.groups,)).fetchone()[0])
            smallest, largest = self.db.execute("SELECT MIN(n),MAX(n) FROM groups").fetchone()
            self.summary = {"n_groups": self.groups, "n_periods": self.periods, "obs_per_group_min": float(smallest), "obs_per_group_avg": sample.nrows/self.groups, "obs_per_group_max": float(largest)}
            self.balanced = smallest==self.periods
            self.db.commit()
            self.path.chmod(0o600)
        except BaseException:
            self.close()
            raise

    def lookup(self, ids, name):
        unique = list(dict.fromkeys(ids))
        values = {}
        for start in range(0, len(unique), 500):
            part = unique[start:start+500]
            values.update(self.db.execute(f"SELECT id,{name} FROM groups WHERE id IN ("+",".join("?" for _ in part)+")", part))
        return torch.tensor([values[gid] for gid in ids], dtype=torch.float64)

    def blocks(self, *, transformed=False):
        cursor = self.db.execute("SELECT pos,g,t,data FROM rows ORDER BY g,t")
        last = last_g = last_t = None
        while records := cursor.fetchmany(self.sample.rows):
            ids = [row[1] for row in records]
            data = torch.frombuffer(bytearray(b"".join(row[3] for row in records)), dtype=torch.float64).reshape(-1, self.width)
            previous = torch.cat((data[:1] if last is None else last[None], data[:-1]), 0)
            codes = torch.tensor(ids, dtype=torch.int64)
            first = torch.cat((torch.tensor([last_g is None or ids[0]!=last_g]), codes[1:]!=codes[:-1]))
            if transformed:
                ranks = {}
                if self.date:
                    periods = list(dict.fromkeys([row[2] for row in records]+([last_t] if last_t is not None else [])))
                    for start in range(0, len(periods), 500):
                        part = periods[start:start+500]
                        ranks.update(self.db.execute("SELECT t,rank FROM periods WHERE t IN ("+",".join("?" for _ in part)+")", part))
                # Python integer subtraction does not wrap at signed64 edges.
                for row, is_first in zip(records, first.tolist(), strict=True):
                    if not is_first:
                        step = ranks[row[2]]-ranks[last_t] if self.date else row[2]-last_t
                        if step!=1:
                            raise AnalysisError("time_gaps", "AR(1) disturbances need consecutive periods within each panel.")
                    last_t = row[2]
                rho = self.lookup(ids, "rho")
                transformed_data = data-rho[:, None]*previous
                transformed_data[first] = data[first]*(1-rho[first].square()).sqrt()[:, None]
            else:
                transformed_data = data
            yield records, data, transformed_data, previous, first
            last, last_g, last_t = data[-1].clone(), ids[-1], records[-1][2]
        self.passes += 1

    def time_batches(self, *, transformed=False):
        # Batch many small periods into one native matrix product/whitening.
        # At most max(group_count, configured row block) rows of a dense grid
        # are live, after the true G²/one-period resource check. Never T×G.
        query = "SELECT r.pos,r.g,r.t,r.data,p.data,g.rho FROM rows r JOIN groups g ON g.id=r.g LEFT JOIN rows p ON p.g=r.g AND p.t=(SELECT MAX(t) FROM rows WHERE g=r.g AND t<r.t) ORDER BY r.t,r.g"
        cursor = self.db.execute(query if transformed else "SELECT pos,g,t,data,NULL,0 FROM rows ORDER BY t,g")
        pending = cursor.fetchone()
        batches = []
        periods_per_batch = max(1, self.sample.rows//self.groups)
        while pending is not None:
            period, records = pending[2], []
            while pending is not None and pending[2]==period:
                records.append(pending)
                pending = cursor.fetchone()
            batches.append(records)
            if len(batches)==periods_per_batch:
                yield self._period_batch(batches, transformed)
                batches = []
        if batches:
            yield self._period_batch(batches, transformed)
        self.passes += 1

    def _period_batch(self, batches, transformed):
        records = [row for part in batches for row in part]
        data = torch.frombuffer(bytearray(b"".join(row[3] for row in records)), dtype=torch.float64).reshape(-1, self.width)
        if transformed:
            previous = torch.frombuffer(bytearray(b"".join(row[4] if row[4] is not None else row[3] for row in records)), dtype=torch.float64).reshape_as(data)
            rho = torch.tensor([row[5] for row in records], dtype=torch.float64)
            first = torch.tensor([row[4] is None for row in records])
            data -= rho[:, None]*previous
            data[first] = previous[first]*(1-rho[first].square()).sqrt()[:, None]
        rows = torch.tensor([i for i, part in enumerate(batches) for _ in part], dtype=torch.int64)
        cols = torch.tensor([row[1] for row in records], dtype=torch.int64)
        grid = torch.zeros((len(batches), self.groups, self.width), dtype=torch.float64)
        present = torch.zeros((len(batches), self.groups), dtype=torch.float64)
        grid[rows, cols], present[rows, cols] = data, 1.
        return grid, present

    def verify(self):
        for _ in self.sample.batches():
            pass

    def close(self):
        if self.db:
            self.db.close()
            self.db = None
        if self.scratch:
            self.scratch.cleanup()
            self.scratch = None


def _qr(panels, *, transformed=False, variances=False, sigma=None):
    tree = _TSQRTree()
    if sigma is not None:
        factor = cholesky_factor(sigma)
        for values, _ in panels.time_batches(transformed=transformed):
            rhs = values.permute(1, 0, 2).reshape(panels.groups, -1)
            block = torch.linalg.solve_triangular(factor, rhs, upper=False).reshape(panels.groups, -1, panels.width).permute(1, 0, 2).reshape(-1, panels.width)
            _, r = torch.linalg.qr(block, mode="r")
            tree.add(r)
    else:
        for records, _, values, _, _ in panels.blocks(transformed=transformed):
            if variances:
                values = values/panels.lookup([row[1] for row in records], "var").sqrt()[:, None]
            _finite(values)
            _, r = torch.linalg.qr(values, mode="r")
            tree.add(r)
    factor = tree.finish()
    beta, bread, condition = _solve(factor, panels.width-1)
    return beta, bread, condition


def _rho(panels, beta, kind, rhotype, *, np1=False):
    # Check observed period spacing before estimating a disturbance parameter.
    # Initial rho=0 is only a validation transformation, never a fitted value.
    for _ in panels.blocks(transformed=True):
        pass
    sums = _GroupMeans(5, 0, panel=True)
    try:
        for records, data, _, previous, first in panels.blocks():
            e = data[:, -1]-data[:, :-1]@beta
            lag = previous[:, -1]-previous[:, :-1]@beta
            pair = (~first).to(torch.float64)
            values = torch.stack((e*lag*pair, lag.square()*pair, e.square()*pair, (e-lag).square()*pair, e.square()), 1)
            sums.add([row[1].to_bytes(8, "big") for row in records], values, torch.ones_like(e), [])
        sums.finish()
        numerator, denominator = _CompensatedSum(()), _CompensatedSum(())
        cursor = sums.connection.execute("SELECT m.key,m.value,p.rows FROM means m JOIN panelmeta p USING(key)")
        while rows := cursor.fetchmany(panels.sample.rows):
            values = torch.frombuffer(bytearray(b"".join(row[1] for row in rows)), dtype=torch.float64).reshape(-1, 6)
            n = torch.tensor([row[2] for row in rows], dtype=torch.float64)
            cross, lagged, current, difference, every = (values[:, 1:]*n[:, None]).T
            if rhotype=="regress":
                rho = cross/lagged
            elif rhotype=="freg":
                rho = cross/current
            elif rhotype in {"tscorr", "theil"}:
                rho = cross/every
                if rhotype=="theil":
                    rho *= (n-len(beta))/n
            else:
                rho = 1-difference/every/2
                if rhotype=="nagar":
                    rho = (rho*n.square()+len(beta)**2)/(n.square()-len(beta)**2)
            valid = n>1
            if kind=="psar1" and not bool(valid.all()):
                raise AnalysisError("insufficient_observations", "Panel-specific AR(1) requires two observations in every panel.")
            if kind=="ar1":
                weight = n if np1 else n-1
                numerator.add((rho[valid]*weight[valid]).sum())
                denominator.add(weight[valid].sum())
            else:
                _check_rho(rho)
                panels.db.executemany("UPDATE groups SET rho=? WHERE id=?", ((float(value), int.from_bytes(row[0], "big")) for row, value in zip(rows, rho.tolist(), strict=True)))
        if kind=="ar1":
            if float(denominator.value)<=0:
                raise AnalysisError("insufficient_observations", "AR(1) requires panels with consecutive observation pairs.")
            common = numerator.value/denominator.value
            _check_rho(common)
            panels.db.execute("UPDATE groups SET rho=?", (float(common),))
    finally:
        sums.close()


def _check_rho(rho):
    if not bool(torch.isfinite(rho).all()) or bool((rho.abs()>=1).any()):
        raise AnalysisError("invalid_rho", "The estimated AR(1) coefficient is outside (-1,1); the Prais-Winsten transformation is undefined.")


def _variance(panels, beta, *, transformed=False, casewise=False, target="var", require_positive=False):
    if casewise and not panels.common:
        raise AnalysisError("no_common_periods", "No period is observed in every panel; use pairwise=True.")
    sums, rss = _GroupMeans(1, 0, panel=True), _CompensatedSum(())
    try:
        for records, _, data, _, _ in panels.blocks(transformed=transformed):
            residual = data[:, -1]-data[:, :-1]@beta
            rss.add(residual.square().sum())
            if casewise:
                periods = list(dict.fromkeys(row[2] for row in records))
                common = set()
                for start in range(0, len(periods), 500):
                    part = periods[start:start+500]
                    common.update(row[0] for row in panels.db.execute("SELECT t FROM periods WHERE n=? AND t IN ("+",".join("?" for _ in part)+")", [panels.groups, *part]))
                mask = torch.tensor([row[2] in common for row in records])
            else:
                mask = torch.ones(len(records), dtype=torch.bool)
            if bool(mask.any()):
                selected = [row[1].to_bytes(8, "big") for row, keep in zip(records, mask.tolist(), strict=True) if keep]
                values = residual[mask].square()[:, None]
                sums.add(selected, values, torch.ones(len(values), dtype=torch.float64), [])
        sums.finish()
        cursor = sums.connection.execute("SELECT key,value FROM means")
        while rows := cursor.fetchmany(panels.sample.rows):
            values = torch.frombuffer(bytearray(b"".join(row[1] for row in rows)), dtype=torch.float64).reshape(-1, 2)[:, 1]
            if (require_positive and not bool((values>0).all())) or not bool(torch.isfinite(values).all()):
                raise AnalysisError("perfect_fit", "Some panel has zero residual variance; its GLS weight is infinite.")
            panels.db.executemany(f"UPDATE groups SET {target}=? WHERE id=?", ((value, int.from_bytes(row[0], "big")) for row, value in zip(rows, values.tolist(), strict=True)))
        return float(rss.value)
    finally:
        sums.close()


def _sigma(panels, beta, *, transformed=False, pairwise=False):
    sigma, counts = _CompensatedSum((panels.groups, panels.groups)), _CompensatedSum((panels.groups, panels.groups))
    used = 0
    for data, present in panels.time_batches(transformed=transformed):
        if not pairwise:
            keep = present.sum(1)==panels.groups
            data, present = data[keep], present[keep]
        error = data[:, :, -1]-data[:, :, :-1]@beta
        sigma.add(error.T@error)
        if pairwise:
            counts.add(present.T@present)
        used += len(data)
    if not used:
        raise AnalysisError("no_common_periods", "No period is observed in every panel; use pairwise=True.")
    if pairwise:
        matrix = torch.where(counts.value>0, sigma.value/counts.value.clamp_min(1), torch.zeros_like(sigma.value))
        unpaired = int((counts.value==0).sum())//2
    else:
        matrix, unpaired = sigma.value/used, 0
    _finite(matrix)
    return matrix, used, unpaired


def fit_streaming_panelgls(spec, source, *, batch_rows=None):
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("singular_information", "The panel GLS factor is not identified.") from error
    except (OSError, sqlite3.Error) as error:
        raise AnalysisError("panel_spill_failed", "Panel GLS needs writable temporary storage and enough disk space.") from error


def _fit(spec, source, *, batch_rows):
    notes = _Notes(spec)
    pcse = spec.estimator=="xtpcse"
    if spec.estimator not in {"xtgls", "xtpcse"}:
        raise AnalysisError("invalid_spec", "This replay kernel requires xtgls or xtpcse.")
    corr = notes.option("correlation" if pcse else "corr")
    hetonly, independent, pairwise = (bool(notes.option(key)) for key in ("hetonly", "independent", "pairwise")) if pcse else (False, False, False)
    if hetonly and independent:
        raise AnalysisError("invalid_spec", "hetonly and independent are alternatives.")
    panels_kind = ("iid" if independent else "heteroskedastic" if hetonly else "correlated") if pcse else notes.option("panels")
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean")
    sample.prepare()
    p, n = len(design.terms), sample.nobs
    if not p or n<=p:
        raise AnalysisError("insufficient_observations", "Panel GLS needs a nonempty design and more observations than parameters.")
    resource = sample.plan_rows("ordered panel GLS/PCSE disk replay", {"ordered_panel_SQLite_cache_and_decode": 6*1024**2,
            "scalar_panel_moments_SQLite_cache": 6*1024**2, "global_TSQR_covariance": 2048*(p+1)**2}, 512*(p+1))
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        moments.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
    variation = float(moments.m2.value[1]/moments.mass)
    if variation<=0:
        raise AnalysisError("constant_outcome", "The panel outcome does not vary.")
    scale = float(moments.magnitude[1])*math.sqrt(variation)
    center = float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1]) if spec.intercept else 0.
    with ExitStack() as stack:
        panel = _Panels(sample, scale, center)
        stack.callback(panel.close)
        if panels_kind=="correlated":
            # Sigma/counts/compensation/cholesky/precision and row period copies
            # are charged together with existing replay/cache/factor reservations.
            resource = sample.plan_rows("quadratic contemporaneous panel covariance", {"ordered_and_group_SQLite_caches": 12*1024**2,
                     "global_TSQR_covariance": 2048*(p+1)**2,
                     "Sigma_counts_compensation_and_factor_scratch": 128*panel.groups**2,
                     "one_period_group_design_and_whitening": 128*panel.groups*(p+1)}, 512*(p+1))
            if not pcse:
                if not panel.balanced:
                    raise AnalysisError("unbalanced_panel", "Correlated GLS needs every panel observed in every period.")
                if panel.periods<panel.groups:
                    raise AnalysisError("insufficient_periods", "Correlated GLS needs at least as many periods as panels to estimate Sigma.")
        beta, bread, condition = _qr(panel)
        rss = _variance(panel, beta)
        if rss<=1e-24*n:
            raise AnalysisError("perfect_fit", "The regressors explain the panel outcome exactly.")
        rho, structure, sigma_record, converged, iterations = None, None, {}, True, 0
        if pcse:
            if corr!="independent":
                _rho(panel, beta, corr, notes.option("rhotype"), np1=bool(notes.option("np1")))
                beta, bread, condition = _qr(panel, transformed=True)
            transformed = corr!="independent"
            rss = _variance(panel, beta, transformed=transformed, casewise=hetonly and not pairwise)
            meat = _CompensatedSum((p, p))
            if panels_kind=="correlated":
                structure, common, unpaired = _sigma(panel, beta, transformed=transformed, pairwise=pairwise)
                for data, _ in panel.time_batches(transformed=transformed):
                    x = data[:, :, :p]
                    meat.add(torch.einsum("tgp,gh,thq->pq", x, structure, x))
                sigma_record = {"unpaired": unpaired} if pairwise else {"common_periods": common}
            else:
                structure = rss/n
                for records, _, data, _, _ in panel.blocks(transformed=transformed):
                    x = data[:, :p]
                    variance = torch.full((len(x),), structure, dtype=torch.float64) if independent else panel.lookup([row[1] for row in records], "var")
                    meat.add(x.T@(x*variance[:, None]))
            covariance = bread@meat.value@bread
        else:
            igls = bool(notes.option("igls"))
            converged = not igls
            while True:
                transformed = corr!="independent"
                if transformed:
                    _rho(panel, beta, corr, notes.option("rhotype"))
                reference = beta if iterations or not transformed else _qr(panel, transformed=True)[0]
                rss = _variance(panel, reference, transformed=transformed, require_positive=panels_kind=="heteroskedastic")
                if panels_kind=="correlated":
                    structure = _sigma(panel, reference, transformed=transformed)[0]
                    new_beta, covariance, condition = _qr(panel, transformed=transformed, sigma=structure)
                elif panels_kind=="heteroskedastic":
                    new_beta, covariance, condition = _qr(panel, transformed=transformed, variances=True)
                else:
                    structure = rss/n
                    new_beta, inverse, condition = _qr(panel, transformed=transformed)
                    covariance = inverse*structure
                old_raw, new_raw = scale*design.transform@beta, scale*design.transform@new_beta
                if spec.intercept:
                    old_raw[0] += center
                    new_raw[0] += center
                change = float(((new_raw-old_raw).abs()/(old_raw.abs()+1)).max())
                beta = new_beta
                iterations += 1
                if not igls:
                    break
                if iterations>1 and change<=notes.option("tolerance"):
                    converged = True
                    break
                if iterations>=notes.option("max_iterations"):
                    raise AnalysisError("nonconvergence", f"Iterated GLS did not converge in{iterations} iterations.")
        physical_transform = scale*design.transform
        raw_beta, raw_cov = physical_transform@beta, physical_transform@covariance@physical_transform.T
        if spec.intercept:
            raw_beta[0] += center
        raw_cov = (raw_cov+raw_cov.T)/2
        _finite(raw_beta, raw_cov)
        if pcse and bool((raw_cov.diagonal()<=0).any()):
            raise AnalysisError("invalid_covariance", "The panel-corrected covariance is not positive; use casewise Sigma or hetonly.")
        predictions = []
        for batch in sample.batches():
            take = min(400-len(predictions), len(batch.frame))
            expected = sample.raw_design(batch, "mean")@raw_beta
            for pos, actual, fitted in zip(batch.positions[:take].tolist(), batch.numeric(spec.outcome)[:take].tolist(), expected[:take].tolist(), strict=True):
                predictions.append({"row": pos, "observed": actual, "fitted": fitted, "residual": actual-fitted})
        slopes = list(range(int(spec.intercept), p))
        tests = {"model": wald_test(raw_beta, raw_cov, slopes, label="Wald chi2 test of the slopes")}
        covcount = 1 if panels_kind=="iid" else panel.groups if panels_kind=="heteroskedastic" else panel.groups*(panel.groups+1)//2
        metrics = {**panel.summary, "estimated_covariances": covcount, "estimated_autocorrelations": 0 if corr=="independent" else 1 if corr=="ar1" else panel.groups, "estimated_coefficients": p, "df_model": len(slopes)}
        extra = {"rhotype": notes.option("rhotype")}
        if corr!="independent":
            rho = [row[0] for row in panel.db.execute("SELECT rho FROM groups ORDER BY id LIMIT 201")]
            if panel.groups<=200:
                extra["rho"] = rho
            if corr=="ar1":
                metrics["rho"] = rho[0]
        if pcse:
            physical_y, final_rss = _WeightedMoments(2, intercept=True), _CompensatedSum(())
            for _, _, d, _, _ in panel.blocks(transformed=corr!="independent"):
                first_column = d[:, 0] if spec.intercept else torch.zeros(len(d), dtype=torch.float64)
                # Compute the exact transformed-response TSS in normalized
                # units; both RSS and TSS share scale², which may overflow
                # although their ratio remains representable.
                ys = d[:, -1]+(center/scale)*first_column
                physical_y.add(torch.stack((torch.ones_like(ys), ys), 1), torch.ones_like(ys), False)
                final_rss.add((d[:, -1]-d[:, :p]@beta).square().sum())
            root_tss = float(physical_y.magnitude[1])*math.sqrt(float(physical_y.m2.value[1]))
            metrics["r_squared"] = 1-(math.sqrt(float(final_rss.value))/root_tss)**2 if root_tss>0 else None
            sigma_kind = "independent" if independent else "hetonly" if hetonly else "pairwise" if pairwise else "casewise"
            extra.update({"correlation": corr, "sigma": sigma_kind, "np1": bool(notes.option("np1")), **sigma_record})
            if independent:
                extra["sigma2"] = structure*scale*scale
            elif panel.groups<=200:
                extra["sigma_variances" if hetonly else "sigma_matrix"] = ([row[0]*scale*scale for row in panel.db.execute("SELECT var FROM groups ORDER BY id")] if hetonly else (structure*scale*scale).tolist())
            correction = f"Beck-Katz panel-corrected covariance; Sigma{sigma_kind}, native period products without full period×panel grid"
        else:
            final_rss = _variance(panel, beta, transformed=corr!="independent", target="last_var", require_positive=panels_kind=="heteroskedastic")
            ll = None
            if corr=="independent":
                if panels_kind=="iid":
                    ll = -.5*n*(_LOG_2PI+math.log(final_rss/n)+1)-n*math.log(scale)
                elif panels_kind=="heteroskedastic":
                    ll = -.5*sum(count*(_LOG_2PI+math.log(var)+1) for count, var in panel.db.execute("SELECT n,last_var FROM groups"))-n*math.log(scale)
                else:
                    final_sigma = _sigma(panel, beta)[0]
                    sign, logdet = torch.linalg.slogdet(final_sigma)
                    ll = -.5*(n*(_LOG_2PI+1)+panel.periods*float(logdet))-n*math.log(scale) if sign>0 else None
            metrics["log_likelihood"] = ll
            extra.update({"panels": panels_kind, "corr": corr, "igls": bool(notes.option("igls")), "iterations": iterations,
                          "panel_levels": [row[0] for row in panel.db.execute("SELECT label FROM groups ORDER BY id LIMIT 200")]})
            if panels_kind=="iid":
                extra["sigma2"] = structure*scale*scale
            elif panel.groups<=200:
                extra["sigma"] = ([row[0]*scale*scale for row in panel.db.execute("SELECT var FROM groups ORDER BY id")] if panels_kind=="heteroskedastic" else (structure*scale*scale).tolist())
            correction = "GLS covariance from globally whitened TSQR; group variances or contemporaneous Sigma, original units"
        info = {"covariance": spec.covariance, "df_inference": None, "df_resid": None, "correction": correction, "nobs": n}
        panel.verify()
        return _result(sample, terms=design.terms, beta=raw_beta, covariance=raw_cov, info=info, metrics=metrics, notes=notes,
               predictions=predictions, tests=tests, solver="native_panel_GLS_PCSE_ordered_disk_replay", diagnostics={"converged": converged, "iterations": iterations,
                  "condition_number": condition, "ordered_replays": panel.passes, "ordered_scratch_bytes": panel.path.stat().st_size,
                  "matrix_scope": "correlated options retain checked G² factors and one G×P period block; no T×G×P tensor"}, extra=extra,
               resource=resource.record(), title="Linear regression, panel-corrected standard errors" if pcse else "Cross-sectional time-series FGLS regression", use_t=False)
