"""Native disk-backed high-dimensional fixed effects.

Group identities, iterative singleton selection and bipartite connectivity are
metadata on owned SQLite storage. Numerical blocks and global projections stay
in bounded float64 Torch batches. No observation/level table is collected.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .linear.common import check_fit, check_pweights, model_test
from .linear.reghdfe import _controls
from .replay_sample import ReplaySample
from .streaming_linear import _GroupMeans, _Notes, _covariance, _finite, _result, _scratch_directory, _solve, _weights


class _Selection:
    """Exact singleton fixed point and disk connected-component metadata."""

    def __init__(self, dimensions, clusters):
        self.dimensions, self.clusters = dimensions, clusters
        self.connection = self.scratch = None
        self.singleton_rounds = 0
        self.component_rounds = []
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-hd-selection-", dir=_scratch_directory())
            self.path = Path(self.scratch.name)/"selection.sqlite3"
            self.connection = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-2048", "temp_store=FILE", "mmap_size=0"):
                self.connection.execute("PRAGMA "+pragma)
            columns = [*(f"f{i} BLOB" for i in range(dimensions)), *(f"c{i} BLOB" for i in range(clusters))]
            self.connection.execute("CREATE TABLE rows(position INTEGER PRIMARY KEY,protected INTEGER,alive INTEGER,"+",".join(columns)+")")
            for i in range(dimensions):
                self.connection.execute(f"CREATE INDEX f{i}_rows ON rows(alive,f{i})")
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as error:
            self.close()
            raise self.failure() from error

    @staticmethod
    def failure():
        return AnalysisError("fixed_effect_spill_failed", "Replay fixed effects need writable temporary storage and enough free disk space.")

    def seed(self, sample, absorb, clusters, drop_singletons):
        try:
            for batch in sample.batches():
                labels = [encode_cluster_labels(batch.frame[name]) for name in [*absorb, *clusters]]
                protected = ((batch.weights >= 2).tolist() if sample.spec.weight_type == "fweight"
                             else [False]*len(batch.frame))
                records = ((position, int(repeated), 1, *keys) for position, repeated, keys in
                           zip(batch.positions.tolist(), protected, zip(*labels, strict=True), strict=True))
                self.connection.executemany("INSERT INTO rows VALUES("+",".join("?" for _ in range(3+len(labels)))+")", records)
            self.original_used = sample.nrows
            if drop_singletons:
                while True:
                    dropped = 0
                    for d in range(self.dimensions):
                        self.connection.execute("DROP TABLE IF EXISTS singleton")
                        self.connection.execute(f"CREATE TEMP TABLE singleton AS SELECT f{d} AS key FROM rows WHERE alive=1 GROUP BY f{d} HAVING COUNT(*)+SUM(protected)=1")
                        self.connection.execute("CREATE INDEX singleton_key ON singleton(key)")
                        self.connection.execute(f"UPDATE rows SET alive=0 WHERE alive=1 AND f{d} IN(SELECT key FROM singleton)")
                        dropped += self.connection.execute("SELECT changes()").fetchone()[0]
                    if not dropped:
                        break
                    self.singleton_rounds += 1
            self.used = self.connection.execute("SELECT COUNT(*) FROM rows WHERE alive=1").fetchone()[0]
            if not self.used:
                raise AnalysisError("empty_sample", "Every observation is a singleton in some absorbed dimension; absorb fewer dimensions or set drop_singletons=False.")
            self.levels = []
            for d in range(self.dimensions):
                self.connection.execute(f"CREATE TABLE keep{d}(key BLOB PRIMARY KEY) WITHOUT ROWID")
                self.connection.execute(f"INSERT INTO keep{d} SELECT DISTINCT f{d} FROM rows WHERE alive=1")
                self.levels.append(self.connection.execute(f"SELECT COUNT(*) FROM keep{d}").fetchone()[0])
            self.connection.commit()
        except sqlite3.Error as error:
            raise self.failure() from error

    def filter(self, frame, absorb):
        keep = torch.ones(len(frame), dtype=torch.bool)
        try:
            for d, name in enumerate(absorb):
                labels = encode_cluster_labels(frame[name])
                unique = list(dict.fromkeys(labels))
                found = set()
                for start in range(0, len(unique), 500):
                    part = unique[start:start+500]
                    query = f"SELECT key FROM keep{d} WHERE key IN("+",".join("?" for _ in part)+")"
                    found.update(row[0] for row in self.connection.execute(query, part))
                keep &= torch.tensor([key in found for key in labels], dtype=torch.bool)
            return keep
        except sqlite3.Error as error:
            raise self.failure() from error

    def _components(self, first, second):
        """Native SQLite min-label hooking/jumping; every graph table is on disk."""
        connection = self.connection
        try:
            for name in ("nodes", "edges", "proposals"):
                connection.execute(f"DROP TABLE IF EXISTS {name}")
            connection.execute("CREATE TABLE nodes(id INTEGER PRIMARY KEY,dim INTEGER,key BLOB,component INTEGER,UNIQUE(dim,key))")
            for d in (first, second):
                connection.execute(f"INSERT INTO nodes(dim,key) SELECT {d},key FROM keep{d}")
            connection.execute("UPDATE nodes SET component=id")
            connection.execute("CREATE TABLE edges(a INTEGER,b INTEGER,PRIMARY KEY(a,b)) WITHOUT ROWID")
            connection.execute(f"INSERT OR IGNORE INTO edges SELECT a.id,b.id FROM rows JOIN nodes a ON a.dim={first} AND a.key=rows.f{first} JOIN nodes b ON b.dim={second} AND b.key=rows.f{second} WHERE rows.alive=1")
            connection.execute("CREATE INDEX edges_b ON edges(b,a)")
            rounds = 0
            while True:
                connection.execute("DROP TABLE IF EXISTS proposals")
                connection.execute("CREATE TABLE proposals(id INTEGER PRIMARY KEY,value INTEGER)")
                connection.execute("INSERT INTO proposals SELECT id,MIN(component) FROM(SELECT id,component FROM nodes UNION ALL SELECT edges.a,nodes.component FROM edges JOIN nodes ON nodes.id=edges.b UNION ALL SELECT edges.b,nodes.component FROM edges JOIN nodes ON nodes.id=edges.a) GROUP BY id")
                connection.execute("UPDATE nodes SET component=(SELECT value FROM proposals WHERE proposals.id=nodes.id) WHERE component>(SELECT value FROM proposals WHERE proposals.id=nodes.id)")
                changed = connection.execute("SELECT changes()").fetchone()[0]
                connection.execute("UPDATE nodes SET component=(SELECT parent.component FROM nodes parent WHERE parent.id=nodes.component) WHERE component>(SELECT parent.component FROM nodes parent WHERE parent.id=nodes.component)")
                changed += connection.execute("SELECT changes()").fetchone()[0]
                rounds += 1
                if not changed:
                    break
            count = connection.execute("SELECT COUNT(DISTINCT component) FROM nodes").fetchone()[0]
            self.component_rounds.append({"first": first, "second": second, "rounds": rounds})
            return count
        except sqlite3.Error as error:
            raise self.failure() from error

    def degrees_of_freedom(self):
        try:
            nested = []
            for d in range(self.dimensions):
                inside = False
                for c in range(self.clusters):
                    bad = self.connection.execute(f"SELECT COUNT(*) FROM(SELECT f{d} FROM rows WHERE alive=1 GROUP BY f{d} HAVING MIN(c{c})<>MAX(c{c}))").fetchone()[0]
                    inside |= bad == 0
                nested.append(inside)
            redundant, earlier = [], []
            for d in range(self.dimensions):
                if nested[d]:
                    redundant.append(self.levels[d])
                elif not earlier:
                    redundant.append(0)
                    earlier.append(d)
                else:
                    redundant.append(max(self._components(other, d) for other in earlier))
                    earlier.append(d)
            return self.levels, redundant, nested, sum(self.levels)-sum(redundant)
        except sqlite3.Error as error:
            raise self.failure() from error

    def diagnostics(self):
        return {"singleton_metadata_rounds": self.singleton_rounds,
                "connected_component_rounds": self.component_rounds,
                "selection_disk_bytes": self.path.stat().st_size,
                "selection_memory": "bounded projected labels and SQLite page cache; no global group/index array"}

    def position_hash(self):
        digest = hashlib.sha256()
        cursor = self.connection.execute("SELECT position FROM rows WHERE alive=1 ORDER BY position")
        while rows := cursor.fetchmany(4096):
            block = torch.tensor([row[0] for row in rows], dtype=torch.int64)
            digest.update(block.numpy().astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def fit_streaming_hdfe(spec, source, *, batch_rows=None):
    """Bounded Dataset delegate; implementation is staged before registration."""
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("numerical_failure", "Native fixed-effect projection could not produce a finite full-rank result.") from error


def _fit(spec, source, batch_rows):
    notes = _Notes(spec)
    check_pweights(notes)
    tolerance, maximum_iterations = _controls(notes)
    absorb = spec.columns.get("absorb", [])
    absorb = [absorb] if isinstance(absorb, str) else list(absorb)
    if spec.estimator != "reghdfe" or not absorb:
        raise AnalysisError("invalid_spec", "Native high-dimensional FE replay requires reghdfe and at least one absorbed dimension.")
    clusters = registry.cluster_columns(spec) if spec.covariance == "cluster" else []
    first = ReplaySample(spec, source, batch_rows=batch_rows)
    first.add_design("mean", intercept=True)
    first.prepare()
    first.plan_rows("HD FE singleton discovery", {"selection_SQLite_cache": 2*1024**2},
                    128*len(absorb)+128*len(first.designs["mean"].terms)+512)
    with ExitStack() as stack:
        selection = _Selection(len(absorb), len(clusters))
        stack.callback(selection.close)
        selection.seed(first, absorb, clusters, notes.option("drop_singletons"))
        dropped = first.nrows-selection.used
        sample = ReplaySample(spec, source, batch_rows=batch_rows,
                              row_filter=(lambda frame: selection.filter(frame, absorb)) if dropped else None)
        design = sample.add_design("mean", intercept=True)
        sample.prepare()
        if (sample.baseline["data_hash"] != first.baseline["data_hash"] or
                sample.baseline["positions_hash"] != selection.position_hash()):
            raise AnalysisError("source_changed", "The original source or singleton-selected physical rows changed before FE replay.")
        if dropped:
            notes.warn(f"Dropped {dropped} singleton observation(s) alone in a level of an absorbed dimension (reghdfe default; drop_singletons=False keeps them).")
        levels, redundant, nested, total = selection.degrees_of_freedom()
        constant_added = all(nested)
        df_absorbed = total+int(constant_added)
        width = len(design.terms)
        resource = sample.plan_rows("native HD FE disk projections and covariance", {
            "initial_reporting_sample": first.reporting_bytes,
            "initial_design_metadata": 64*(width+1)**2,
            "initial_category_metadata": first._category_bytes,
            "selection_SQLite_cache": 2*1024**2,
            "projection_SQLite_cache": 2*1024**2,
            "cluster_accumulator_caches": 18*1024**2 if clusters else 0,
            "global_TSQR_and_covariance": 1024*(width+1)**2,
        }, 192*width+128*len(absorb)+512)
        moments = _WeightedMoments(2, intercept=True)
        for batch in sample.batches():
            y = batch.numeric(spec.outcome)
            moments.add(torch.stack((torch.ones_like(y), y), 1), _weights(sample, batch), False)
        y_anchor, y_magnitude = float(moments.anchor[1]), float(moments.magnitude[1])
        y_center = float(moments.mean[1])
        y_mean = y_anchor+y_magnitude*y_center
        y_scale = y_magnitude*math.sqrt(float(moments.m2.value[1]/moments.mass))
        if not math.isfinite(y_scale) or y_scale <= 0:
            raise AnalysisError("constant_outcome", "The absorbed outcome has no finite variation.")
        states = _Vectors(sample, width, len(absorb), selection.path.stat().st_size)
        stack.callback(states.close)
        before_ss, tss = _CompensatedSum((width-1,)), _CompensatedSum(())
        with states.writer("raw") as writer:
            for batch in sample.batches():
                outcome = ((batch.numeric(spec.outcome)-y_anchor)/y_magnitude-y_center)*(y_magnitude/y_scale)
                values = torch.cat((outcome[:, None], batch.designs["mean"][:, 1:]), 1)
                weights = _weights(sample, batch)
                before_ss.add((values[:, 1:].square()*weights[:, None]).sum(0))
                tss.add((outcome.square()*weights).sum())
                states.write(writer, values)
        iterations, maximum_update, method = _absorb(states, sample, absorb, tolerance, maximum_iterations)
        tree, within_ss, within_tss = _TSQRTree(), _CompensatedSum((width-1,)), _CompensatedSum(())
        for batch, (values,) in states.batches(sample, "y"):
            weights = _weights(sample, batch)
            within_ss.add((values[:, 1:].square()*weights[:, None]).sum(0))
            within_tss.add((values[:, 0].square()*weights).sum())
            tree.add(torch.linalg.qr(torch.cat((values[:, 1:], values[:, :1]), 1)*weights.sqrt()[:, None], mode="r")[1])
        all_factor = tree.finish()
        absorbed = (within_ss.value <= 1e-13*before_ss.value).nonzero().flatten().tolist()
        remaining = [i for i in range(width-1) if i not in absorbed]
        for i in absorbed:
            sample.notes["omitted_terms"].append(design.terms[i+1])
            notes.warn(f"Omitted {design.terms[i+1]}: collinearity with the absorbed fixed effects.")
        if not remaining:
            raise AnalysisError("empty_design", "Every regressor is collinear with the absorbed fixed effects; nothing is left to estimate.")
        kept, omitted = collinear_columns(all_factor[:, remaining])
        selected = [remaining[i] for i in kept]
        for i in omitted:
            sample.notes["omitted_terms"].append(design.terms[remaining[i]+1])
            notes.warn(f"Omitted {design.terms[remaining[i]+1]}: collinearity.")
        factor = all_factor[:, [*selected, all_factor.shape[1]-1]]
        beta, bread, condition = _solve(factor, len(selected))
        k, k_total = len(selected), len(selected)+df_absorbed
        df = sample.nobs-k_total
        if df <= 0:
            raise AnalysisError("insufficient_observations", "The absorbed regression has no residual degrees of freedom.")
        scales = design.scales[design.kept][[i+1 for i in selected]]

        def residual_blocks():
            for batch, (values,) in states.batches(sample, "y"):
                yield batch, values[:, [i+1 for i in selected]], values[:, 0], batch.numeric(spec.outcome), y_scale

        covariance, info, rss_work, predictions = _covariance(sample, residual_blocks, beta, bread,
                       df=df, k=k_total, notes=notes, score_basis=torch.diag(scales))
        scale_squared = y_scale*y_scale
        rss, total_tss, total_within = rss_work*scale_squared, float(tss.value)*scale_squared, float(within_tss.value)*scale_squared
        original_scale = total_tss+sample.nobs*y_mean*y_mean
        if not all(math.isfinite(value) for value in (rss, total_tss, total_within, original_scale)):
            raise AnalysisError("non_finite_result", "Absorbed outcome sums of squares exceed representable float64 units.")
        resolution = (32*tolerance)**2*total_within if len(absorb)>1 else 0.
        check_fit(rss, total_tss, df, original_scale, absorbed=True, resolution=resolution)
        transform = torch.diag(y_scale/scales)
        raw_beta, raw_covariance = transform@beta, transform@covariance@transform.T
        _finite(raw_beta, raw_covariance)
        terms = [design.terms[i+1] for i in selected]
        r2, r2within = 1-rss/total_tss, 1-rss/total_within
        metrics = {"r_squared": r2, "adjusted_r_squared": 1-(1-r2)*(sample.nobs-1)/df,
                   "r_squared_within": r2within,
                   "adjusted_r_squared_within": 1-(rss/df)/(total_within/(sample.nobs-df_absorbed)),
                   "rmse": math.sqrt(rss/df), "df_model": k, "df_resid": df,
                   "df_absorbed": df_absorbed, "n_singletons_dropped": dropped}
        test = model_test(notes, raw_beta, raw_covariance, range(k), df_inference=info["df_inference"],
                         classical=(total_within-rss, k, rss, df), label="Model F test (slopes)")
        info.update({"nobs": sample.nobs, "k_total": k_total, "df_resid": df,
                     "absorbed_degrees_of_freedom": df_absorbed,
                     "degrees_of_freedom_convention": "reghdfe: observed levels minus redundant (connected components); dimensions nested within any cluster column not counted, one degree of freedom added for the constant when every dimension is nested"})
        extra = {"absorbed": [{"column": name, "levels": count, "redundant": lost, "nested": inside}
                 for name, count, lost, inside in zip(absorb, levels, redundant, nested, strict=True)],
                 "constant_degree_of_freedom_added": constant_added,
                 "iterations": iterations, "converged": True, "method": method,
                 "max_update": maximum_update, "tolerance": tolerance, "singletons_dropped": dropped}
        diagnostics = {"condition_number": condition, "rank": k,
                       "condition_number_basis": "unit-norm standardized within TSQR design",
                       "tsqr_reduction_depth": tree.depth, **selection.diagnostics(), **states.diagnostics()}
        result = _result(sample, terms=terms, beta=raw_beta, covariance=raw_covariance, info=info,
                       metrics=metrics, notes=notes, predictions=predictions, tests={"model": test},
                       solver="native_disk_symmetric_Kaczmarz_CG_and_TSQR", diagnostics=diagnostics,
                       extra=extra, resource=resource.record())
        def fitted_blocks():
            for batch,(values,) in states.batches(sample,'y'):
                residual=values[:,0]-values[:,[i+1 for i in selected]]@beta
                yield batch.frame,batch.numeric(spec.outcome)-y_scale*residual
        from .postest.group_state import capture_fixed_replay
        result.extra['group_state']=capture_fixed_replay(result,fitted_blocks)
        return result


class _Vectors:
    """Owned sequential vector files; never an mmap or whole-table tensor."""

    def __init__(self, sample, width, dimensions, metadata_bytes):
        self.width, self.rows = width, sample.nrows
        self.connection = None
        self.scratch = None
        self.bytes_per_file = self.rows*width*8
        self.maximum_files = 3 if dimensions == 1 else 12
        self.required_bytes = self.bytes_per_file*self.maximum_files+metadata_bytes*2
        self.io_passes = 0
        try:
            parent = _scratch_directory()
            free = shutil.disk_usage(parent or tempfile.gettempdir()).free
            if self.required_bytes+64*1024**2 > free:
                error = AnalysisError("fixed_effect_disk_limit", f"Native fixed-effect projection needs an estimated {self.required_bytes:,} additional scratch bytes; available free disk is {free:,} bytes. Use a larger scratch volume. This is not a RAM budget.")
                error.disk_plan = {"estimated_scratch_bytes": self.required_bytes, "available_bytes": free,
                                   "state_vector_bytes": self.bytes_per_file*self.maximum_files,
                                   "scope": "owned vector files plus a metadata reserve; simultaneous external writes and filesystem overhead are not guaranteed"}
                raise error
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-hd-state-", dir=parent)
            self.path = Path(self.scratch.name)
        except OSError as error:
            self.close()
            raise _Selection.failure() from error

    def file(self, name):
        return self.path/(name+".bin")

    def writer(self, name):
        try:
            stream = self.file(name).open("wb")
            self.file(name).chmod(0o600)
            return stream
        except OSError as error:
            raise _Selection.failure() from error

    @staticmethod
    def write(stream, values):
        _finite(values)
        try:
            stream.write(memoryview(values.contiguous().numpy().tobytes()))
        except OSError as error:
            raise _Selection.failure() from error

    def copy(self, first, second):
        try:
            shutil.copyfile(self.file(first), self.file(second))
        except OSError as error:
            raise _Selection.failure() from error

    def swap(self, first, second):
        try:
            self.file(first).replace(self.file(second))
        except OSError as error:
            raise _Selection.failure() from error

    def batches(self, sample, *names):
        with ExitStack() as stack:
            try:
                streams = [stack.enter_context(self.file(name).open("rb")) for name in names]
                for batch in sample.batches():
                    byte_count = len(batch.frame)*self.width*8
                    values = []
                    for stream in streams:
                        block = bytearray(stream.read(byte_count))
                        if len(block) != byte_count:
                            raise AnalysisError("source_changed", "An owned FE vector is not aligned to retained replay rows.")
                        values.append(torch.frombuffer(block, dtype=torch.float64).reshape(-1, self.width))
                    yield batch, values
                if any(stream.read(1) for stream in streams):
                    raise AnalysisError("source_changed", "An owned FE vector has extra rows after replay.")
                self.io_passes += 1
            except OSError as error:
                raise _Selection.failure() from error

    def diagnostics(self):
        return {"FE_state_vector_IO_passes": self.io_passes,
                "FE_state_estimated_scratch_bytes": self.required_bytes,
                "FE_state_actual_scratch_bytes": sum(path.stat().st_size for path in self.path.iterdir()),
                "FE_state_storage": "owned sequential float64 files; bounded reads, no mmap/full table",
                "FE_state_disk_complexity": "O(retained_rows*design_columns + group moments and metadata)"}

    def close(self):
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def _project(states, sample, absorb, source, target, *, accumulated=None, added=None):
    """A single global M_d projection; optional accumulated removed means."""
    with ExitStack() as stack:
        groups = _GroupMeans(states.width, 0)
        stack.callback(groups.close)
        for batch, (values,) in states.batches(sample, source):
            groups.add(encode_cluster_labels(batch.frame[absorb]), values, _weights(sample, batch), [])
        groups.finish()
        output = stack.enter_context(states.writer(target))
        removed = stack.enter_context(states.writer(added)) if added else None
        inputs = (source, accumulated) if accumulated else (source,)
        for batch, blocks in states.batches(sample, *inputs):
            means = groups.lookup(encode_cluster_labels(batch.frame[absorb]))
            states.write(output, blocks[0]-means)
            if removed:
                states.write(removed, means+(blocks[1] if accumulated else 0.))


def _operator(states, sample, dimensions, source):
    """A=I−T on range(M_1), accumulated means avoid u−T*u cancellation."""
    order = [*range(1, len(dimensions)), *range(len(dimensions)-2, 0, -1)]
    current, accumulated = source, None
    for step, d in enumerate(order):
        following, added = f"work{step%2}", f"acc{step%2}"
        _project(states, sample, dimensions[d], current, following, accumulated=accumulated, added=added)
        current, accumulated = following, added
    _project(states, sample, dimensions[0], accumulated, "v")


def _dots(states, sample, first, second):
    total = _CompensatedSum((states.width,))
    inputs = (first,) if first == second else (first, second)
    for batch, values in states.batches(sample, *inputs):
        other = values[0] if first == second else values[1]
        total.add((values[0]*other*_weights(sample, batch)[:, None]).sum(0))
    _finite(total.value)
    return total.value


def _absorb(states, sample, dimensions, tolerance, max_iterations):
    _project(states, sample, dimensions[0], "raw", "y")
    if len(dimensions) == 1:
        return 1, 0., "within"
    norm = _dots(states, sample, "y", "y").sqrt()
    start = norm.clamp_min(torch.finfo(torch.float64).tiny).clone()
    floor = 8*torch.finfo(torch.float64).eps*norm
    active = norm > 0
    _operator(states, sample, dimensions, "y")
    states.copy("v", "r")
    states.copy("r", "u")
    ssr = _dots(states, sample, "r", "r")
    active &= ssr.sqrt() > floor
    relative = ssr.sqrt()/norm.clamp_min(torch.finfo(torch.float64).tiny)
    peak, cleaned = ssr.clone(), 0
    worst = 0.
    if not bool(active.any()):
        return 1, 0., "disk_symmetric_kaczmarz_cg"
    for iteration in range(1, max_iterations):
        _operator(states, sample, dimensions, "u")
        curvature = _dots(states, sample, "u", "v")
        alpha = torch.where((curvature > 0)&active, ssr/curvature, 0.)
        next_ssr, current_norm, direction_norm = _CompensatedSum((states.width,)), _CompensatedSum((states.width,)), _CompensatedSum((states.width,))
        with states.writer("next_y") as ywriter, states.writer("next_r") as rwriter:
            for batch, (y, r, u, v) in states.batches(sample, "y", "r", "u", "v"):
                weight = _weights(sample, batch)[:, None]
                y_new, r_new = y-u*alpha, r-v*alpha
                states.write(ywriter, y_new)
                states.write(rwriter, r_new)
                next_ssr.add((r_new.square()*weight).sum(0))
                current_norm.add((y_new.square()*weight).sum(0))
                direction_norm.add((u.square()*weight).sum(0))
        states.swap("next_y", "y")
        states.swap("next_r", "r")
        previous, ssr = ssr, next_ssr.value
        _finite(ssr, current_norm.value)
        norm = current_norm.value.sqrt()
        residual = ssr.sqrt()
        change = torch.maximum(direction_norm.value.sqrt()*alpha.abs(), residual)
        regular = change/norm.clamp_min(torch.finfo(torch.float64).tiny)
        relative = torch.where((residual <= floor)&(regular > tolerance), residual/start, regular)
        finished = (change <= tolerance*norm) | (residual <= floor)
        worst = max(worst, float(relative[active&finished].max()) if bool((active&finished).any()) else 0.)
        active &= ~finished
        if not bool(active.any()):
            _project(states, sample, dimensions[0], "y", "clean_y")
            states.swap("clean_y", "y")
            return iteration+1, worst, "disk_symmetric_kaczmarz_cg"
        beta = torch.where((curvature > 0)&active, ssr/previous.clamp_min(torch.finfo(torch.float64).tiny), 0.)
        with states.writer("next_u") as writer:
            for _, (r, u) in states.batches(sample, "r", "u"):
                states.write(writer, (r+u*beta)*active)
        states.swap("next_u", "u")
        peak = torch.maximum(peak, ssr)
        if iteration-cleaned >= 8 and bool(((ssr <= 1e-2*peak)&active).any()):
            _project(states, sample, dimensions[0], "r", "clean_r")
            states.swap("clean_r", "r")
            _project(states, sample, dimensions[0], "u", "clean_u")
            states.swap("clean_u", "u")
            ssr = _dots(states, sample, "r", "r")
            peak, cleaned = ssr.clone(), iteration
    raise AnalysisError("absorption_nonconvergence", f"Fixed-effect absorption did not converge in {max_iterations} sweeps (largest relative update {float(relative[active].max()):.3e}, tolerance {tolerance:.3e}).")
