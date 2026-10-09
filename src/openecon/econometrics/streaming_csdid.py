"""Balanced-panel CSDID with native replay nuisance fits and disk unit influences."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

import torch
from pandas.api.types import is_datetime64_any_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import collinear_columns, least_squares
from openecon.engines.separation import certify_separation
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .replay_sample import ReplaySample, MAX_PARAMETERS
from .streaming_did import _period_values
from .streaming_linear import _Notes, _finite, _result, _scratch_directory
from .teffects.csdid import _aggregations
from .teffects.index import IndexObjective

SUPPORTED = frozenset({"csdid"})


def supports(spec):
    return spec.estimator in SUPPORTED


class _Store:
    def __init__(self, sample, notes):
        self.sample, self.notes = sample, notes
        self.width = len(sample.designs["x"].terms) if sample.spec.predictors else 0
        self.directory = tempfile.TemporaryDirectory(
            prefix="openecon-csdid-", dir=_scratch_directory()
        )
        self.path = Path(self.directory.name) / "panel.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute(
            "CREATE TABLE units(key BLOB PRIMARY KEY,lo INTEGER,hi INTEGER,na INTEGER,yes INTEGER,gi INTEGER) WITHOUT ROWID"
        )
        self.db.execute(
            "CREATE TABLE records(key BLOB,time INTEGER,v BLOB,PRIMARY KEY(key,time)) WITHOUT ROWID"
        )
        self.db.execute("CREATE TABLE periods(time INTEGER PRIMARY KEY,value INTEGER,idx INTEGER)")
        self.db.execute(
            "CREATE TABLE phi(key BLOB,att INTEGER,value REAL,PRIMARY KEY(key,att)) WITHOUT ROWID"
        )
        self.timing = registry.role_columns(sample.spec, "treatment_time")[0]
        self.kind = None

    def disk_guard(self, required):
        free = shutil.disk_usage(self.path.parent).free
        if required > max(0, free - 64 * 1024**2):
            raise AnalysisError(
                "insufficient_scratch_space",
                "CSDID panel and joint influence records need more free owned temporary disk space.",
            )
        return required

    def seed(self):
        sample, spec = self.sample, self.sample.spec
        self.disk_guard(sample.nrows * (128 + 16 * (self.width + 1)))
        moment = _WeightedMoments(2, intercept=True)
        for batch in sample.batches():
            y = batch.numeric(spec.outcome)
            moment.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
        self.anchor, self.magnitude, self.center = (
            moment.anchor[1],
            moment.magnitude[1],
            moment.mean[1],
        )
        self.magnitude = (
            self.magnitude if self.magnitude > 0 else torch.tensor(1.0, dtype=torch.float64)
        )
        self.scale = self.magnitude * (moment.m2.value[1] / moment.mass).clamp_min(0).sqrt()
        self.scale = self.scale if self.scale > 0 else torch.tensor(1.0, dtype=torch.float64)
        for batch in sample.batches():
            series = batch.frame[spec.time]
            kind = "datetime" if is_datetime64_any_dtype(series.dtype) else "integer"
            if self.kind and self.kind != kind:
                raise AnalysisError(
                    "invalid_time", "CSDID time dtype changes between source partitions."
                )
            self.kind = kind
            time = (
                torch.as_tensor(series.astype("int64").to_numpy(copy=True), dtype=torch.int64)
                if kind == "datetime"
                else _period_values(series, integer=True)
            )
            first = _period_values(batch.frame[self.timing], integer=True)
            never = torch.as_tensor(batch.frame[self.timing].isna().to_numpy(), dtype=torch.bool)
            keys = encode_cluster_labels(batch.frame[spec.panel])
            unique = {}
            codes = []
            for key in keys:
                if key not in unique:
                    unique[key] = len(unique)
                codes.append(unique[key])
            code = torch.tensor(codes, dtype=torch.int64)
            n = len(unique)
            lo, hi = (
                torch.full((n,), torch.iinfo(torch.int64).max, dtype=torch.int64),
                torch.full((n,), torch.iinfo(torch.int64).min, dtype=torch.int64),
            )
            lo.scatter_reduce_(0, code[~never], first[~never], "amin")
            hi.scatter_reduce_(0, code[~never], first[~never], "amax")
            na, yes = torch.zeros(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64)
            na.scatter_reduce_(0, code, never.to(torch.int64), "amax")
            yes.scatter_reduce_(0, code, (~never).to(torch.int64), "amax")
            self.db.executemany(
                "INSERT INTO units VALUES(?,?,?,?,?,NULL) ON CONFLICT(key) DO UPDATE SET lo=MIN(lo,excluded.lo),hi=MAX(hi,excluded.hi),na=MAX(na,excluded.na),yes=MAX(yes,excluded.yes)",
                (
                    (key, int(lo[i]), int(hi[i]), int(na[i]), int(yes[i]))
                    for key, i in unique.items()
                ),
            )
            y = ((batch.numeric(spec.outcome) - self.anchor) / self.magnitude - self.center) * (
                self.magnitude / self.scale
            )
            x = batch.designs["x"] if self.width else torch.empty((len(y), 0), dtype=torch.float64)
            values = torch.cat((y[:, None], x), 1)
            _finite(values)
            packed = memoryview(values.contiguous().numpy().tobytes())
            stride = 8 * (self.width + 1)
            try:
                self.db.executemany(
                    "INSERT INTO records VALUES(?,?,?)",
                    (
                        (key, int(time[i]), packed[i * stride : (i + 1) * stride])
                        for i, key in enumerate(keys)
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise AnalysisError(
                    "unbalanced_panel", "CSDID needs exactly one observation per unit and period."
                ) from error
        bad = self.db.execute(
            "SELECT COUNT(*) FROM units WHERE (na=1 AND yes=1) OR (yes=1 AND lo<>hi)"
        ).fetchone()[0]
        if bad:
            raise AnalysisError(
                "invalid_treatment",
                "First treatment period must be constant within each unit, including never-treated status.",
            )
        self.periods()
        minimum = self.db.execute("SELECT MIN(value) FROM periods").fetchone()[0]
        early = self.db.execute(
            "SELECT COUNT(*) FROM units WHERE yes=1 AND lo<=?", (minimum,)
        ).fetchone()[0]
        if early:
            self.db.execute("DELETE FROM units WHERE yes=1 AND lo<=?", (minimum,))
            self.db.execute("DELETE FROM records WHERE key NOT IN(SELECT key FROM units)")
            self.notes.warn(
                f"Excluded {early} unit(s) treated in or before the first period (they have no pre-treatment period), as did does."
            )
            self.periods()
        self.n = self.db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        self.t = self.db.execute("SELECT COUNT(*) FROM periods").fetchone()[0]
        if not self.n or self.t < 2:
            raise AnalysisError(
                "invalid_treatment",
                "CSDID needs retained units and at least two observed periods after removing early treatment.",
            )
        if self.n > 2**53:
            raise AnalysisError(
                "precision_unsupported", "CSDID influence shares require exact float64 unit counts."
            )
        self.original_used = self.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        if self.original_used != self.n * self.t:
            raise AnalysisError(
                "unbalanced_panel", "CSDID needs every retained unit in every observed period."
            )
        # The composite record PK and global N*T count imply every unit has every period.
        bad = self.db.execute(
            "SELECT COUNT(*) FROM(SELECT key,COUNT(*) n FROM records GROUP BY key HAVING n<>?)",
            (self.t,),
        ).fetchone()[0]
        if bad:
            raise AnalysisError(
                "unbalanced_panel", "CSDID needs every retained unit in every observed period."
            )
        maximum = self.db.execute("SELECT MAX(value) FROM periods").fetchone()[0]
        late = self.db.execute(
            "SELECT COUNT(*) FROM units WHERE yes=1 AND lo>?", (maximum,)
        ).fetchone()[0]
        if late:
            self.notes.warn(
                "Units first treated after the last period are treated as never treated."
            )
        self.db.execute(
            "UPDATE units SET gi=CASE WHEN na=1 OR lo>? THEN ? ELSE (SELECT idx FROM periods WHERE value=units.lo) END",
            (maximum, self.t + 1),
        )
        if self.db.execute("SELECT COUNT(*) FROM units WHERE gi IS NULL").fetchone()[0]:
            raise AnalysisError(
                "invalid_treatment",
                "First treatment must be an observed period or missing for never-treated units.",
            )
        self.never = self.db.execute("SELECT COUNT(*) FROM units WHERE gi>?", (self.t,)).fetchone()[
            0
        ]
        if self.notes.option("control") == "never" and not self.never:
            raise AnalysisError(
                "invalid_treatment",
                "control='never' needs never-treated units; use control='notyet'.",
            )
        self.cohorts = self.db.execute(
            "SELECT gi,COUNT(*),MIN(lo) FROM units WHERE gi<=? GROUP BY gi ORDER BY gi", (self.t,)
        ).fetchmany(MAX_PARAMETERS + 1)
        if not self.cohorts:
            raise AnalysisError(
                "invalid_treatment", "No cohort is first treated after the first observed period."
            )
        if len(self.cohorts) > MAX_PARAMETERS:
            raise AnalysisError(
                "model_too_wide",
                "CSDID cohort output exceeds the bounded joint-influence model width.",
            )
        self.db.execute("CREATE INDEX unit_cohort ON units(gi,key)")
        self.db.commit()

    def periods(self):
        self.db.execute("DELETE FROM periods")
        self.db.execute(
            "INSERT INTO periods SELECT time,time,ROW_NUMBER() OVER(ORDER BY time)-1 FROM records GROUP BY time"
        )
        if self.kind == "datetime":
            self.db.execute("UPDATE periods SET value=idx")
            self.notes.warn(
                f"Datetime column '{self.sample.spec.time}' is treated as consecutive periods in sorted order; gaps between dates are not detected."
            )
        self.db.execute("CREATE INDEX IF NOT EXISTS period_index ON periods(idx)")
        self.db.execute("CREATE INDEX IF NOT EXISTS period_value ON periods(value)")

    def comparisons(self):
        rows = []
        base, control = self.notes.option("base"), self.notes.option("control")
        for gi, n1, g in self.cohorts:
            for ti in range(self.t):
                if base == "universal" and ti == gi - 1:
                    continue
                bi = gi - 1 if ti >= gi or base == "universal" else ti - 1
                if bi < 0:
                    continue
                limit = self.t if control == "never" else max(ti, bi)
                n0 = self.db.execute(
                    "SELECT COUNT(*) FROM units WHERE gi>? AND gi<>?", (limit, gi)
                ).fetchone()[0]
                t = self.db.execute("SELECT value FROM periods WHERE idx=?", (ti,)).fetchone()[0]
                if n0 < 2:
                    self.notes.warn(f"ATT({g:g},{t:g}) skipped: fewer than two control units.")
                    continue
                rows.append(_Pair(self, gi, ti, bi, g, t, n1, n0, limit))
                if len(rows) + len(self.cohorts) > MAX_PARAMETERS:
                    raise AnalysisError(
                        "model_too_wide",
                        "CSDID joint ATT and cohort-share influence output exceeds384 columns.",
                    )
        if not rows:
            raise AnalysisError("invalid_treatment", "No group-time effect is identified.")
        return rows

    def phi_add(self, index, keys, values):
        _finite(values)
        self.db.executemany(
            "INSERT INTO phi VALUES(?,?,?)",
            ((key, index, float(value)) for key, value in zip(keys, values, strict=True)),
        )

    def joint_factor(self, pairs, cohort_ids, shares):
        tree = _TSQRTree()
        cursor = self.db.execute("SELECT key,gi FROM units ORDER BY key")
        count = len(pairs)
        while block := cursor.fetchmany(self.sample.rows):
            keys = [row[0] for row in block]
            positions = {key: i for i, key in enumerate(keys)}
            columns, indices, values = [], [], []
            for start in range(0, len(keys), 500):
                part = keys[start : start + 500]
                for key, att, value in self.db.execute(
                    "SELECT key,att,value FROM phi WHERE key IN("
                    + ",".join("?" for _ in part)
                    + ")",
                    part,
                ):
                    indices.append(positions[key])
                    columns.append(att)
                    values.append(value)
            phi = torch.zeros((len(keys), count), dtype=torch.float64)
            if values:
                phi[torch.tensor(indices), torch.tensor(columns)] = torch.tensor(
                    values, dtype=torch.float64
                )
            gi = torch.tensor([row[1] for row in block], dtype=torch.int64)
            membership = torch.stack([(gi == c).to(torch.float64) for c in cohort_ids], 1)
            share = (membership - shares) / self.n
            matrix = torch.cat((phi, share), 1)
            _finite(matrix)
            self.sample.actual_numeric_peak_rows = max(
                self.sample.actual_numeric_peak_rows, len(keys)
            )
            tree.add(qr_factor(matrix))
        return tree.finish(), {
            "influence_TSQR_depth": tree.depth,
            "influence_TSQR_peak_factors": tree.peak_factors,
        }

    def close(self):
        self.db.close()
        self.directory.cleanup()


@dataclass
class _Pair:
    store: _Store
    gi: int
    ti: int
    bi: int
    g: int
    t: int
    n1: int
    n0: int
    limit: int

    def raw(self):
        db, width, sample = self.store.db, self.store.width, self.store.sample
        time = db.execute("SELECT time FROM periods WHERE idx=?", (self.ti,)).fetchone()[0]
        base = db.execute("SELECT time FROM periods WHERE idx=?", (self.bi,)).fetchone()[0]
        cursor = db.execute(
            "SELECT u.key,u.gi,a.v,b.v FROM units u JOIN records a ON a.key=u.key AND a.time=? JOIN records b ON b.key=u.key AND b.time=? WHERE u.gi=? OR (u.gi>? AND u.gi<>?) ORDER BY u.key",
            (time, base, self.gi, self.limit, self.gi),
        )
        while rows := cursor.fetchmany(sample.rows):
            a = torch.frombuffer(
                bytearray(b"".join(row[2] for row in rows)), dtype=torch.float64
            ).reshape(-1, width + 1)
            b = torch.frombuffer(
                bytearray(b"".join(row[3] for row in rows)), dtype=torch.float64
            ).reshape(-1, width + 1)
            d = torch.tensor([row[1] == self.gi for row in rows], dtype=torch.float64)
            sample.actual_numeric_peak_rows = max(sample.actual_numeric_peak_rows, len(rows))
            yield [row[0] for row in rows], a[:, 0] - b[:, 0], b[:, 1:], d

    def blocks(self):
        for keys, y, x, d in self.raw():
            if self.store.width:
                x = ((x - self.anchor) / self.magnitude - self.center) / self.rms
                x = x[:, self.kept]
            yield keys, y, x, d

    def prepare(self, method):
        if not self.store.width:
            self.kept, self.omitted = [], []
            return
        moment = _WeightedMoments(self.store.width, intercept=True)
        for _, _, x, d in self.raw():
            moment.add(x, torch.ones_like(d), False)
        self.anchor, self.magnitude, self.center = moment.anchor, moment.magnitude, moment.mean
        self.magnitude = torch.where(
            self.magnitude > 0, self.magnitude, torch.ones_like(self.magnitude)
        )
        self.rms = (moment.m2.value / moment.mass).clamp_min(0).sqrt()
        self.rms = torch.where(self.rms > 0, self.rms, torch.ones_like(self.rms))
        self.center[0], self.rms[0] = 0.0, 1.0
        alltree, controltree = _TSQRTree(), _TSQRTree()
        for _, _, x, d in self.raw():
            x = ((x - self.anchor) / self.magnitude - self.center) / self.rms
            alltree.add(qr_factor(x))
            controltree.add(qr_factor(x[d < 1]))
        kept, _ = collinear_columns(alltree.finish())
        local, _ = collinear_columns(controltree.finish()[:, kept])
        if len(local) < len(kept):
            if method != "reg":
                raise AnalysisError(
                    "overlap_violation",
                    "A covariate distinguishes the cohort but is constant among comparison controls; the propensity model separates them.",
                )
            kept = [kept[i] for i in local]
        if not kept or kept[0] != 0:
            raise AnalysisError(
                "invalid_spec", "CSDID covariates are collinear with the comparison constant."
            )
        self.kept = kept
        self.omitted = [
            self.store.sample.designs["x"].terms[i]
            for i in range(1, self.store.width)
            if i not in kept
        ]
        if method in {"reg", "dr"} and self.n0 <= len(kept):
            raise AnalysisError(
                "insufficient_observations",
                "CSDID needs more control units than outcome regression parameters in each comparison.",
            )

    def nuisance(self, method):
        self.prepare(method)
        self.gamma, self.beta = None, None
        k = len(self.kept)
        if not self.store.width:
            return
        if method in {"ipw", "dr"}:
            objective = _CompensatedSum((k,))

            def signed():
                for _, _, x, d in self.blocks():
                    yield x * (2 * d - 1)[:, None]

            for rows in signed():
                objective.add(rows.sum(0))
            try:
                self.separation = certify_separation(
                    signed, objective.value / (self.n0 + self.n1), k
                )
            except KernelError as error:
                if error.code == "separation_detected":
                    raise AnalysisError(
                        "overlap_violation",
                        "The CSDID propensity model separates the cohort from controls.",
                    ) from error
                raise
            start = torch.zeros(k, dtype=torch.float64)
            start[0] = math.log(self.n1 / self.n0)
            objective = _Objective(self, k)
            n = self.n0 + self.n1
            run = optimize.maximize_newton(
                lambda v: tuple(item / n for item in objective(v)),
                start,
                value_fn=lambda v: objective.value(v) / n,
                max_iter=200,
                step_tol=1e-10,
                scaled_gradient_tol=1e-10,
                raise_on_failure=False,
            )
            if not run.converged:
                raise AnalysisError(
                    "nonconvergence", "CSDID comparison propensity optimization did not converge."
                )
            self.gamma = run.theta
        if method in {"reg", "dr"}:
            tree = _TSQRTree()
            for _, y, x, d in self.blocks():
                mask = d < 1
                tree.add(qr_factor(torch.cat((x[mask], y[mask, None]), 1)))
            factor = tree.finish()
            self.beta = least_squares(
                factor[:, :k], factor[:, k], drop_collinear=False, tol=0.0
            ).beta

    def fit(self, method):
        self.nuisance(method)
        k = len(self.kept)
        moments = _CompensatedSum((4,))
        for _, y, x, d in self.blocks():
            resid = y - x @ self.beta if self.beta is not None else y
            odds = torch.exp(x @ self.gamma) if self.gamma is not None else torch.ones_like(y)
            if self.gamma is not None and bool(
                (torch.sigmoid(x @ self.gamma)[d < 1] > 1 - 1e-5).any()
            ):
                raise AnalysisError(
                    "overlap_violation", "A CSDID control propensity lies within1e-5 of one."
                )
            _finite(resid, odds)
            weight = (1 - d) * odds
            moments.add(
                torch.stack((d.sum(), (d * resid).sum(), weight.sum(), (weight * resid).sum()))
            )
        if not self.store.width:
            # All methods reduce to the unadjusted two means without covariates.
            self.eta1 = moments.value[1] / self.n1
            self.eta0 = moments.value[3] / self.n0
            self.att = self.eta1 - self.eta0
            self.direction = None
        else:
            self.eta1 = moments.value[1] / self.n1
            self.eta0 = (
                moments.value[3] / moments.value[2]
                if method != "reg"
                else torch.tensor(0.0, dtype=torch.float64)
            )
            self.att = self.eta1 - self.eta0
            gs = slice(0, k) if self.gamma is not None else None
            bs = slice(k if gs else 0, (k if gs else 0) + k) if self.beta is not None else None
            i1 = (k if gs else 0) + (k if bs else 0)
            i0 = i1 + 1 if method != "reg" else None
            size = i1 + 1 + (i0 is not None)
            derivative = _CompensatedSum((size, size))
            for _, y, x, d in self.blocks():
                resid = y - x @ self.beta if bs else y
                a = torch.zeros((size, size), dtype=torch.float64)
                if gs:
                    eta = x @ self.gamma
                    p, odds = torch.sigmoid(eta), torch.exp(eta)
                    weight = (1 - d) * odds
                    a[gs, gs] = -(x * (p * (1 - p))[:, None]).T @ x
                else:
                    weight = torch.zeros_like(y)
                if bs:
                    a[bs, bs] = -(x * (1 - d)[:, None]).T @ x
                    a[i1, bs] = -(d @ x)
                a[i1, i1] = -d.sum()
                if i0 is not None:
                    a[i0, i0] = -weight.sum()
                    a[i0, gs] = (weight * (resid - self.eta0)) @ x
                    if bs:
                        a[i0, bs] = -(weight @ x)
                derivative.add(a)
            contrast = torch.zeros(size, dtype=torch.float64)
            contrast[i1] = 1.0
            if i0 is not None:
                contrast[i0] = -1.0
            try:
                self.direction = torch.linalg.solve(derivative.value.T, contrast)
            except RuntimeError as error:
                raise AnalysisError(
                    "singular_jacobian",
                    "CSDID comparison has a singular estimating-equation derivative.",
                ) from error
            self.slices = gs, bs, i1, i0
            _finite(self.direction, self.att)
        return float(self.att * self.store.scale)

    def influence(self):
        for keys, y, x, d in self.blocks():
            if self.direction is None:
                phi = d * (y - self.eta1) / self.n1 - (1 - d) * (y - self.eta0) / self.n0
            else:
                gs, bs, i1, i0 = self.slices
                resid = y - x @ self.beta if bs else y
                blocks = []
                if gs:
                    eta = x @ self.gamma
                    blocks.append(x * (d - torch.sigmoid(eta))[:, None])
                if bs:
                    blocks.append(x * ((1 - d) * (y - x @ self.beta))[:, None])
                blocks.append((d * (resid - self.eta1))[:, None])
                if i0 is not None:
                    blocks.append(((1 - d) * torch.exp(eta) * (resid - self.eta0))[:, None])
                phi = -torch.cat(blocks, 1) @ self.direction
            yield keys, phi * self.store.scale


class _Objective:
    def __init__(self, pair, size):
        self.pair, self.size = pair, size

    def __call__(self, theta):
        value, gradient, hessian = (
            _CompensatedSum(()),
            _CompensatedSum((self.size,)),
            _CompensatedSum((self.size, self.size)),
        )
        for _, _, x, d in self.pair.blocks():
            v, g, h = IndexObjective("logit", x, d, torch.ones_like(d))(theta)
            value.add(v)
            gradient.add(g)
            hessian.add(h)
        return value.value, gradient.value, (hessian.value + hessian.value.T) / 2

    def value(self, theta):
        value = _CompensatedSum(())
        for _, _, x, d in self.pair.blocks():
            value.add(IndexObjective("logit", x, d, torch.ones_like(d)).value(theta))
        return value.value


def fit_streaming(spec, source, *, batch_rows=None):
    if not supports(spec):
        raise AnalysisError("streaming_unsupported", "No replay CSDID adapter is registered.")
    defaults = {"sample": "balanced", "anticipation": 0, "bootstrap_reps": 0, "seed": None, "uniform": False}
    if any(spec.options.get(name, default) != default for name, default in defaults.items()):
        raise AnalysisError("streaming_options_unsupported", "Additional sample domains and multiplier inference currently validate resident tables only.")
    if spec.panel is None:
        raise AnalysisError("streaming_options_unsupported", "Replay CSDID requires a balanced panel with group=.")
    if spec.weights or spec.covariance != "robust":
        raise AnalysisError(
            "streaming_options_unsupported",
            "CSDID implements analytic unit influence covariance without user weights.",
        )
    try:
        with torch.no_grad(), torch.device("cpu"), ExitStack() as stack:
            notes = _Notes(spec)
            timing = registry.role_columns(spec, "treatment_time")[0]
            sample = ReplaySample(spec, source, allow_missing=[timing], batch_rows=batch_rows)
            if spec.predictors:
                sample.add_design("x", intercept=True)
            else:
                sample.add_design("x", [], intercept=False)
            sample.prepare()
            all_terms = [
                term
                for name in spec.predictors
                for term in (
                    [f"{name}[{level}]" for level in sample.categories[name]["levels"][1:]]
                    if name in sample.categories
                    else [name]
                )
            ]
            for term in sample.notes["omitted_terms"]:
                notes.warn(
                    "Omitted because of collinearity in some group-time comparisons: " + term + "."
                )
            width = len(sample.designs["x"].terms) if spec.predictors else 0
            if 2 * width + 2 > MAX_PARAMETERS:
                raise AnalysisError(
                    "model_too_wide", "CSDID nuisance estimating equations exceed384 parameters."
                )
            comparison_buffers = {
                "panel_SQLite_cache": 4 * 1024**2,
                "comparison_derivatives": 128 * (2 * width + 2) ** 2,
                "comparison_nuisance_metadata": 128 * MAX_PARAMETERS * (width + 4),
            }
            sample.plan_rows(
                "CSDID panel storage and nuisance equations",
                comparison_buffers,
                256 * (width + 4),
            )
            store = _Store(sample, notes)
            stack.callback(store.close)
            store.seed()
            pairs = store.comparisons()
            cohort_ids = sorted({pair.gi for pair in pairs})
            joint = len(pairs) + len(cohort_ids)
            resource = sample.plan_rows(
                "CSDID full unit influence covariance",
                {
                    **comparison_buffers,
                    "joint_TSQR_and_covariance": 512 * (joint + 1) ** 2,
                },
                256 * (joint + width + 4),
            )
            disk = store.disk_guard(store.n * len(pairs) * 64)
            atts = []
            for index, pair in enumerate(pairs):
                atts.append(pair.fit(notes.option("method")))
                for term in pair.omitted:
                    notes.warn(
                        "Omitted because of collinearity in some group-time comparisons: "
                        + term
                        + "."
                    )
                for keys, phi in pair.influence():
                    store.phi_add(index, keys, phi)
            store.db.commit()
            att = torch.tensor(atts, dtype=torch.float64)
            shares = torch.tensor(
                [
                    store.db.execute("SELECT COUNT(*) FROM units WHERE gi=?", (gi,)).fetchone()[0]
                    / store.n
                    for gi in cohort_ids
                ],
                dtype=torch.float64,
            )
            factor, diagnostics = store.joint_factor(pairs, cohort_ids, shares)
            phi, share_phi = factor[:, : len(pairs)], factor[:, len(pairs) :]
            covariance = phi.T @ phi
            onehot = torch.tensor(
                [[float(pair.gi == gi) for gi in cohort_ids] for pair in pairs], dtype=torch.float64
            )
            pairmeta = [(pair.g, pair.t, pair.gi, pair.ti) for pair in pairs]
            cohorts = [
                store.db.execute("SELECT value FROM periods WHERE idx=?", (gi,)).fetchone()[0]
                for gi in cohort_ids
            ]
            extra = _aggregations(
                pairmeta, att, phi, onehot, shares, share_phi, spec.alpha, cohorts
            )
            extra.update(
                {
                    "method": notes.option("method"),
                    "control_group": notes.option("control"),
                    "base_period": notes.option("base"),
                    "cohorts": [row[2] for row in store.cohorts],
                    "never_treated_units": store.never,
                    "covariates": all_terms,
                }
            )
            pre = [i for i, pair in enumerate(pairs) if pair.ti < pair.gi]
            tests = (
                {
                    "pretrends": wald_test(
                        att,
                        covariance,
                        pre,
                        label="Pre-treatment ATT(g,t) = 0 (parallel trends before treatment)",
                    )
                }
                if pre
                else {}
            )
            # Verify original source and effective physical positions after every fit.
            positions = hashlib.sha256()
            retained = 0
            for batch in sample.batches():
                keys = encode_cluster_labels(batch.frame[spec.panel])
                found = set()
                for start in range(0, len(keys), 500):
                    part = keys[start : start + 500]
                    found.update(
                        row[0]
                        for row in store.db.execute(
                            "SELECT key FROM units WHERE key IN("
                            + ",".join("?" for _ in part)
                            + ")",
                            part,
                        )
                    )
                mask = torch.tensor([key in found for key in keys], dtype=torch.bool)
                positions.update(batch.positions[mask].contiguous().numpy().tobytes())
                retained += int(mask.sum())
            if retained != store.original_used:
                raise AnalysisError(
                    "source_changed", "CSDID effective panel row counts changed during replay."
                )
            diagnostics.update(
                {
                    "panel_disk_bytes": store.path.stat().st_size,
                    "influence_disk_estimate_bytes": disk,
                    "joint_influence_columns": joint,
                    "panel_storage": "owned SQLite unit-period records and sparse unit ATT influences; no N×ATT RAM matrix",
                    "maximum_pair_batch_rows": sample.actual_numeric_peak_rows,
                    "comparison_count": len(pairs),
                }
            )
            info = {
                "df_inference": None,
                "df_resid": None,
                "small_sample_correction": None,
                "bootstrap_reps": 0, "uniform": False, "multiplier": None,
                "se_method": "analytic shared influence functions", "families": {},
                "coefficient_intervals": "pointwise normal; simultaneous bands stored separately",
                "correction": "influence-function covariance of the group-time ATTs, clustered by unit (analytic; did's default multiplier bootstrap is not used)",
            }
            result = _result(
                sample,
                terms=[f"ATT({pair.g},{pair.t})" for pair in pairs],
                beta=att,
                covariance=covariance,
                info=info,
                metrics={
                    "n_groups": store.n,
                    "n_periods": store.t,
                    "n_cohorts": len(cohort_ids),
                    "simple_att": extra["simple"]["estimate"],
                    "simple_att_se": extra["simple"]["std_error"],
                },
                notes=notes,
                predictions=[],
                tests=tests,
                solver="native_replayed_group_time_estimating_equations",
                diagnostics=diagnostics,
                extra=extra,
                resource=resource.record(),
                title="Callaway-Sant'Anna group-time ATT",
                use_t=False,
                provenance_extra={
                    "units": store.n,
                    "effective_panel_rows": retained,
                    "complete_rows_before_early_treatment_filter": sample.nrows,
                    "sample_position_count": retained,
                    "sample_positions_hash": positions.hexdigest(),
                    "sample_hash": hashlib.sha256(
                        (sample.baseline["data_hash"] + positions.hexdigest()).encode()
                    ).hexdigest(),
                    "hash_scope": "projected raw model columns plus missing policy and whole-unit early-treatment exclusion; exact effective physical positions",
                    "effective_positions_hash": positions.hexdigest(),
                    "effective_position_hash_scope": "complete model rows excluding whole units already treated in the initial period",
                },
            )
            result.nobs = retained
            result.dropped_rows = result.nobs_original - retained
            for coefficient, pair in zip(result.coefficients, pairs, strict=True):
                coefficient.equation = f"Cohort {pair.g}"
            return result
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError(
            "csdid_spill_failed",
            "CSDID needs writable owned temporary storage and enough free disk space.",
        ) from error
