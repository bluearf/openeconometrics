"""Exact Hamilton likelihood/Kim smoothing with bounded global disk replay.

Expanded AR regime histories, the ergodic initial distribution and its score
are retained exactly. Numerical Hessian and per-period sandwich scores use
whole-series recursions. No independent chunk fit or recursion reset.
https://www.stata.com/manuals/tsmswitch.pdf
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.streaming_design import numeric_values
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares
from openecon.engines.optimize import maximize_bfgs, numerical_hessian
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from . import registry
from .core import build_result, information_criteria, kernel_call, ml_covariance
from .ordered_replay import OrderedReplay
from .replay_sample import ReplaySample
from .streaming_linear import _Notes
from .streaming_state_output import StateOutput
from .streaming_ucm import _bytes, _tensor
from .streaming_var import _Facade, _head
from .tsmodels import mswitch as native
from .tsmodels import mswitch_kernels as mk

SUPPORTED = frozenset({"mswitch"})


def supports_spec(spec):
    return spec.estimator in SUPPORTED


_CHAIN = """
def forward(logd: Tensor, a: Tensor, current: Tensor):
    n, k = logd.size(0), logd.size(1)
    predicted = torch.empty((n,k), dtype=torch.float64)
    filtered = torch.empty((n,k), dtype=torch.float64)
    ll = torch.empty((n,), dtype=torch.float64)
    for j in range(n):
        peak = torch.max(logd[j])
        pred = torch.matmul(current, a)
        joint = torch.exp(logd[j]-peak)*pred
        scale = torch.sum(joint)
        if not bool(torch.isfinite(scale)) or not bool(scale > 0.):
            raise RuntimeError("zero Hamilton likelihood")
        current = joint/scale
        predicted[j],filtered[j],ll[j] = pred,current,torch.log(scale)+peak
    return predicted,filtered,ll,current

def backward(filtered: Tensor,predicted: Tensor,a: Tensor,next: Tensor,nextpred: Tensor,hasnext: bool):
    n,k = filtered.size(0),filtered.size(1)
    smooth = torch.empty((n,k),dtype=torch.float64)
    for j in range(n-1,-1,-1):
        if hasnext:
            ratio = torch.where(nextpred > 0., next/torch.clamp(nextpred,min=1e-300), torch.zeros_like(nextpred))
            current = filtered[j]*torch.matmul(a,ratio)
            mass = torch.sum(current)
            if not bool(torch.isfinite(mass)) or not bool(mass > 0.):
                raise RuntimeError("zero Kim mass")
            current = current/mass
        else:
            current = filtered[j]
        smooth[j] = current
        next,nextpred,hasnext = current,predicted[j],True
    return smooth,next,nextpred
"""


@lru_cache(maxsize=1)
def _kernel():
    return torch.jit.CompilationUnit(_CHAIN)


def _forward(logd, a, current):
    try:
        return _kernel().forward(logd, a, current)
    except (RuntimeError, torch.jit.Error) as exc:
        raise AnalysisError(
            "numerical_failure",
            "Hamilton prediction mass is zero or nonfinite; rescale inputs or simplify regimes.",
        ) from exc


def _backward(filtered, predicted, a, next, nextpred, hasnext):
    try:
        return _kernel().backward(filtered, predicted, a, next, nextpred, hasnext)
    except (RuntimeError, torch.jit.Error) as exc:
        raise AnalysisError(
            "numerical_failure", "Kim smoothed state mass is outside finite precision."
        ) from exc


@contextmanager
def _disk():
    try:
        with tempfile.TemporaryDirectory(
            prefix="openecon-mswitch-", dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None
        ) as name:
            path = Path(name) / "chain.sqlite"
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
                    "CREATE TABLE blocks(id INTEGER PRIMARY KEY,n INTEGER,y BLOB,xs BLOB,xc BLOB)"
                )
                sql.execute(
                    "CREATE TABLE run(id INTEGER PRIMARY KEY,pred BLOB,filt BLOB,prev BLOB,ll BLOB)"
                )
                sql.execute("CREATE TABLE residuals(id INTEGER PRIMARY KEY,r REAL)")
                path.chmod(0o600)
                yield sql, path
            finally:
                sql.close()
    except (OSError, sqlite3.Error) as exc:
        raise AnalysisError(
            "state_space_spill_failed",
            "Hamilton filtering/Kim smoothing need writable local scratch storage and sufficient disk space.",
        ) from exc


class _Replay:
    def __init__(self, ordered, sql, path, batch_rows=None):
        self.ordered, self.spec, self.sql, self.path = ordered, ordered.spec, sql, path
        self.notes = _Notes(self.spec)
        self.sample = ReplaySample(self.spec, ordered.source, batch_rows=ordered.rows)
        self.design = self.sample.add_design("mean")
        self.sample.prepare()
        if batch_rows is not None:
            if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
                raise AnalysisError("invalid_batch_rows", "batch_rows must be a positive integer.")
            self.sample.rows = min(self.sample.rows, batch_rows)
        states = self.notes.option("states")
        switch = self.notes.option("switch")
        switch = (["Intercept"] if self.spec.intercept else []) if switch is None else list(switch)
        unknown = set(switch) - set(self.design.terms)
        if unknown:
            raise AnalysisError(
                "invalid_option",
                "switch names omitted or absent design terms: " + ", ".join(sorted(unknown)),
            )
        lags = sorted(set(self.notes.option("ar") or []))
        if any(lag < 1 for lag in lags):
            raise AnalysisError("invalid_option", "AR lags must be positive integers.")
        self.switching = [i for i, term in enumerate(self.design.terms) if term in switch]
        self.common = [i for i, term in enumerate(self.design.terms) if term not in switch]
        self.layout = mk.Layout(
            states,
            lags,
            len(self.switching),
            len(self.common),
            bool(self.notes.option("arswitch")),
            bool(self.notes.option("varswitch")),
        )
        layout = self.layout
        if not (layout.switching or layout.varswitch or layout.arswitch and lags):
            raise AnalysisError("invalid_option", "Nothing switches between states.")
        if layout.expanded > native.MAX_EXPANDED:
            raise AnalysisError(
                "model_too_large",
                f"{layout.expanded} expanded regime histories exceed the native512-state shape contract; simplify AR lags or state count.",
            )
        self.n = ordered.count - layout.order
        if self.n < 3 * layout.size + 5:
            raise AnalysisError(
                "insufficient_observations",
                "Too few retained periods for this Markov-switching specification.",
            )
        # Spool partitions must fit the largest later per-period score batch;
        # lowering a numerical plan cannot repartition immutable stored blobs.
        self.plan(max(1, 2 * layout.size))
        moments = _WeightedMoments(2, intercept=True)
        squares = _CompensatedSum((len(self.design.terms),))
        for batch in self.sample.batches():
            y = batch.numeric(self.spec.outcome)
            x = self.sample.raw_design(batch, "mean")
            moments.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
            squares.add(x.square().sum(0))
        self.scale_y = math.sqrt(
            float(moments.m2.value[1] * moments.magnitude[1] ** 2 / (moments.mass - 1))
        )
        if not math.isfinite(self.scale_y) or self.scale_y <= 0.0:
            raise AnalysisError(
                "constant_outcome", "The outcome has no finite variation for regime switching."
            )
        rms = (squares.value / ordered.count).sqrt()
        if not bool(torch.isfinite(rms).all()):
            raise AnalysisError(
                "numerical_failure", "Regressor RMS exceeds finite precision; rescale inputs."
            )
        rms = torch.where(rms > 0.0, rms, torch.ones_like(rms))
        self.scales = torch.cat((rms[self.switching], rms[self.common]))
        self.multiplier, self.offset = native._to_reported(layout, self.scale_y, self.scales)
        self._spool()
        self.numeric_hash = self.digest()
        self.report = (
            pd.concat(
                list(
                    _head(
                        ordered.source.iter_batches(batch_rows=ordered.rows),
                        min(ordered.count, 400 + layout.order),
                    )
                ),
                ignore_index=True,
            )
            .iloc[layout.order :]
            .copy()
        )
        self.positions = self.report.pop(ordered.positions_column).tolist()
        self.frame = SimpleNamespace(
            spec=self.spec,
            info=registry.get("mswitch"),
            n=len(self.report),
            original=self.report,
            sample=self.report,
            positions=self.positions,
            warnings=self.notes.warnings,
            notes=self.sample.notes,
            option=self.notes.option,
            warn=self.notes.warn,
            numeric=lambda name: numeric_values(self.report[name], name),
            _sorted_by=[self.spec.time] if self.spec.time else [],
            resource_plans=[ordered.plan.record(), self.last_plan.record()],
        )

    def plan(self, points):
        k, p, order = self.layout.expanded, self.layout.size, self.layout.order
        width = len(self.design.terms) + 1
        self.last_plan = self.sample.plan_rows(
            "global Hamilton/Kim disk replay",
            {
                "expanded_chain_and_joint_information": 128 * (points + 4) * k * k + 512 * p * p,
                "finite_AR_design_history": 128 * (order + 1) * width,
                "SQLite_order_and_state_caches": 12 * 1024**2,
            },
            8 * (points * (8 * k + p + 4) + 16 * (k + width + order + 1)),
        )
        if hasattr(self, "frame"):
            self.frame.resource_plans = [self.ordered.plan.record(), self.last_plan.record()]

    def _spool(self):
        order = self.layout.order
        history = torch.empty((0, 1 + len(self.design.terms)), dtype=torch.float64)
        seen = j = 0
        for batch in self.sample.batches():
            values = torch.cat(
                (
                    batch.numeric(self.spec.outcome)[:, None] / self.scale_y,
                    self.sample.raw_design(batch, "mean")[:, [*self.switching, *self.common]]
                    / self.scales,
                ),
                1,
            )
            joined = torch.cat((history, values), 0)
            skipped = min(len(values), max(0, order - seen))
            seen += len(values)
            if skipped < len(values):
                n = len(values) - skipped
                selected = joined[-(n + order) :].clone()
                xs = selected[:, 1 : 1 + self.layout.switching]
                xc = selected[:, 1 + self.layout.switching :]
                self.sql.execute(
                    "INSERT INTO blocks VALUES(?,?,?,?,?)",
                    (j, n, _bytes(selected[:, 0]), _bytes(xs), _bytes(xc)),
                )
                j += 1
            history = joined[-order:].clone() if order else joined[:0].clone()
        self.sql.commit()

    def digest(self):
        digest = hashlib.sha256()
        for _, n, y, xs, xc in self.sql.execute("SELECT * FROM blocks ORDER BY id"):
            digest.update(str(n).encode())
            digest.update(y)
            digest.update(xs)
            digest.update(xc)
        return digest.hexdigest()

    def blocks(self, reverse=False):
        layout = self.layout
        for j, n, y, xs, xc in self.sql.execute(
            "SELECT * FROM blocks ORDER BY id " + ("DESC" if reverse else "ASC")
        ):
            size = n + layout.order
            yield (
                j,
                n,
                _tensor(y, (size,)),
                _tensor(xs, (size, layout.switching)),
                _tensor(xc, (size, layout.common)),
            )

    def starts(self):
        layout = self.layout
        order = layout.order
        k = layout.states
        width = layout.switching + layout.common + len(layout.lags)
        tree = _TSQRTree()
        for _, n, y, xs, xc in self.blocks():
            columns = [
                xs[order:],
                xc[order:],
                *[y[order - lag : order - lag + n, None] for lag in layout.lags],
            ]
            x = torch.cat(columns, 1)
            tree.add(qr_factor(torch.cat((x, y[order:, None]), 1)))
        factor = tree.finish()
        fit = kernel_call(least_squares, factor[:, :width], factor[:, -1])
        beta = torch.zeros(width, dtype=torch.float64)
        beta[fit.kept] = fit.beta
        moments = _WeightedMoments(2, intercept=True)
        position = 0
        self.sql.execute("DELETE FROM residuals")
        for _, n, y, xs, xc in self.blocks():
            x = torch.cat(
                [
                    xs[order:],
                    xc[order:],
                    *[y[order - lag : order - lag + n, None] for lag in layout.lags],
                ],
                1,
            )
            residual = y[order:] - x @ beta
            moments.add(
                torch.stack((torch.ones_like(residual), residual), 1),
                torch.ones_like(residual),
                False,
            )
            self.sql.executemany(
                "INSERT INTO residuals VALUES(?,?)",
                ((position + i, value) for i, value in enumerate(residual.tolist())),
            )
            position += n
        self.sql.execute("CREATE INDEX IF NOT EXISTS residual_order ON residuals(r,id)")
        self.sql.commit()
        sd = math.sqrt(float(moments.m2.value[1] * moments.magnitude[1] ** 2 / (self.n - 1)))
        if not sd > 0.0 or not math.isfinite(sd):
            raise AnalysisError(
                "perfect_fit",
                "The linear starting model has no residual variation for regime switching.",
            )
        offsets = []
        spreads = []
        start = 0
        for group in range(k):
            length = self.n // k + int(group < self.n % k)
            accumulator = _WeightedMoments(2, intercept=True)
            cursor = self.sql.execute(
                "SELECT r FROM residuals ORDER BY r,id LIMIT ? OFFSET ?", (length, start)
            )
            while rows := cursor.fetchmany(self.sample.rows):
                residual = torch.tensor([item[0] for item in rows], dtype=torch.float64)
                accumulator.add(
                    torch.stack((torch.ones_like(residual), residual), 1),
                    torch.ones_like(residual),
                    False,
                )
            offsets.append(
                float(accumulator.anchor[1] + accumulator.magnitude[1] * accumulator.mean[1])
            )
            spreads.append(
                max(
                    math.sqrt(
                        float(
                            accumulator.m2.value[1] * accumulator.magnitude[1] ** 2 / (length - 1)
                        )
                    ),
                    0.1 * sd,
                )
                if length > 1
                else sd
            )
            start += length
        starts = []
        ks, kc = layout.switching, layout.common
        for persistence, shrink in ((0.9, 1.0), (0.7, 0.5), (0.95, 1.5)):
            b = beta[:ks].repeat(k, 1)
            if ks:
                b[:, 0] += shrink * torch.tensor(offsets, dtype=torch.float64)
            ar = beta[ks + kc :].repeat(k) if layout.arswitch else beta[ks + kc :]
            sigma = (
                torch.tensor([math.log(value) for value in spreads], dtype=torch.float64)
                if layout.varswitch
                else torch.tensor([math.log(sd)], dtype=torch.float64)
            )
            transition = torch.full((k, k), (1.0 - persistence) / (k - 1), dtype=torch.float64)
            transition.fill_diagonal_(persistence)
            logits = torch.log(transition[:, : k - 1] / transition[:, k - 1 :])
            starts.append(torch.cat((b.flatten(), beta[ks : ks + kc], ar, sigma, logits.flatten())))
        return starts

    def chain(self, theta):
        parts = self.layout.split(theta)
        p = mk.transition(parts["logit"])
        pi, fundamental = mk.ergodic(p)
        a, start = mk.expanded_chain(self.layout, p, pi)
        if not all(bool(torch.isfinite(v).all()) for v in (p, pi, fundamental, a, start)) or bool(
            (pi <= 0.0).any()
        ):
            raise AnalysisError(
                "numerical_failure", "The transition/ergodic system is outside finite precision."
            )
        return parts, p, pi, fundamental, a, start

    def score(self, theta, *, sink=None):
        self.plan(1)
        layout = self.layout
        k = layout.states
        K = layout.expanded
        order = layout.order
        parts, p, pi, fundamental, a, start = self.chain(theta)
        current = start / start.sum()
        ll = _CompensatedSum(())
        self.sql.execute("DELETE FROM run")
        fitted = []
        seen = 0
        for j, n, y, xs, xc in self.blocks():
            pieces = mk.densities(layout, theta, y, xs, xc)
            previous = current.clone()
            pred, filt, loglike, current = _forward(pieces.log_density, a, current)
            ll.add(loglike.sum())
            self.sql.execute(
                "INSERT INTO run VALUES(?,?,?,?,?)",
                (j, _bytes(pred), _bytes(filt), _bytes(previous), _bytes(loglike)),
            )
            take = min(n, 400 - seen)
            if take > 0:
                fitted.append(
                    (pred[:take] * (y[order : order + take, None] - pieces.residual[:take])).sum(1)
                    * self.scale_y
                )
                seen += take
        self.sql.commit()
        lag = layout.lag_state()
        indicator = [
            torch.nn.functional.one_hot(lag[level], k).to(torch.float64)
            for level in range(order + 1)
        ]
        joint = _CompensatedSum((K, K))
        gradient = _CompensatedSum((layout.size,))
        share = _CompensatedSum((k,))
        next = torch.zeros(K, dtype=torch.float64)
        nextpred = torch.zeros_like(next)
        hasnext = False
        initial = None
        for j, n, y, xs, xc in self.blocks(reverse=True):
            record = self.sql.execute("SELECT pred,filt,prev FROM run WHERE id=?", (j,)).fetchone()
            pred, filt, prev = (
                _tensor(record[0], (n, K)),
                _tensor(record[1], (n, K)),
                _tensor(record[2], (K,)),
            )
            smooth, next, nextpred = _kernel().backward(filt, pred, a, next, nextpred, hasnext)
            hasnext = True
            ratio = torch.where(pred > 0.0, smooth / pred.clamp_min(1e-300), torch.zeros_like(pred))
            previous = torch.cat((prev[None], filt[:-1]), 0)
            joint.add(a * (previous.T @ ratio))
            if j == 0:
                initial = start * (a @ ratio[0])
            share.add((smooth @ indicator[0]).sum(0))
            pieces = mk.densities(layout, theta, y, xs, xc)
            sigma = pieces.sigma[lag[0]]
            u = smooth * pieces.residual / sigma.square()
            phi = parts["phi"]
            g = {}
            if layout.switching:
                value = xs[order:].T @ (u @ indicator[0])
                for index, lag_ in enumerate(layout.lags):
                    coefficient = phi[:, index][lag[0]] if layout.arswitch else phi[0, index]
                    value -= xs[order - lag_ : order - lag_ + n].T @ (
                        (u * coefficient) @ indicator[lag_]
                    )
                g["beta"] = value.T.flatten()
            if layout.common:
                value = xc[order:].T @ u.sum(1)
                for index, lag_ in enumerate(layout.lags):
                    coefficient = phi[:, index][lag[0]] if layout.arswitch else phi[0, index]
                    value -= xc[order - lag_ : order - lag_ + n].T @ (u * coefficient).sum(1)
                g["alpha"] = value
            if layout.lags:
                values = []
                for lag_ in layout.lags:
                    term = u * pieces.deviation[lag_][:, lag[lag_]]
                    values.append(
                        (term @ indicator[0]).sum(0) if layout.arswitch else term.sum().reshape(1)
                    )
                g["phi"] = torch.stack(values, 1).flatten()
            standardized = smooth * ((pieces.residual / sigma).square() - 1.0)
            g["lnsigma"] = (
                (standardized @ indicator[0]).sum(0)
                if layout.varswitch
                else standardized.sum().reshape(1)
            )
            gradient.add(
                torch.cat(
                    [
                        g.get(name, torch.zeros(size, dtype=torch.float64))
                        for name, size in layout.sizes().items()
                    ]
                )
            )
            if sink is not None:
                sink(
                    j,
                    n,
                    pred @ indicator[0],
                    filt @ indicator[0],
                    smooth @ indicator[0],
                    y[order:] * self.scale_y,
                )
        counts = torch.zeros((k, k), dtype=torch.float64)
        counts.index_put_(
            (lag[0][:, None].expand(-1, K), lag[0][None].expand(K, -1)),
            joint.value,
            accumulate=True,
        )
        for level in range(1, order + 1):
            counts.index_put_((lag[level], lag[level - 1]), initial, accumulate=True)
        oldest = torch.zeros(k, dtype=torch.float64).index_add_(0, lag[order], initial)
        g_logit = counts[:, : k - 1] - counts.sum(1, keepdim=True) * p[:, : k - 1]
        weight = oldest / pi
        az = p @ fundamental
        for state in range(k):
            for col in range(k - 1):
                g_logit[state, col] += weight @ (
                    pi[state] * p[state, col] * (fundamental[col] - az[state])
                )
        final = gradient.value.clone()
        final[-k * (k - 1) :] = g_logit.flatten()
        if not bool(torch.isfinite(final).all()) or not bool(torch.isfinite(ll.value)):
            raise AnalysisError(
                "numerical_failure",
                "Global Hamilton likelihood or Fisher score exceeds finite precision.",
            )
        return {
            "loglik": ll.value,
            "gradient": final,
            "transition": p,
            "ergodic": pi,
            "fitted": torch.cat(fitted),
            "share": share.value / self.n,
        }

    def score_factor(self, theta):
        p = len(theta)
        self.plan(2 * p)
        eps = torch.finfo(torch.float64).eps ** (1.0 / 3.0)
        step = eps * theta.abs().clamp_min(1.0)
        points = []
        chains = []
        for j in range(p):
            for sign in (1.0, -1.0):
                point = theta.clone()
                point[j] += sign * step[j]
                points.append(point)
                _, _, _, _, a, start = self.chain(point)
                chains.append([a, start / start.sum()])
        tree = _TSQRTree()
        for _, n, y, xs, xc in self.blocks():
            values = []
            for index, point in enumerate(points):
                a, current = chains[index]
                pieces = mk.densities(self.layout, point, y, xs, xc)
                _, _, ll, current = _forward(pieces.log_density, a, current)
                chains[index][1] = current
                values.append(ll)
            scores = torch.stack(
                [(values[2 * j] - values[2 * j + 1]) / (2.0 * step[j]) for j in range(p)], 1
            )
            tree.add(qr_factor(scores))
        return tree.finish()

    def finalize(self, bundle):
        if self.digest() != self.numeric_hash:
            raise AnalysisError(
                "source_changed", "Stored state-space numeric inputs changed during estimation."
            )
        facade = _Facade(
            self.spec,
            self.ordered,
            self.report,
            self.positions,
            self.n,
            self.sample.categories,
            self.sample.provenance(),
        )
        bundle.provenance.update(facade.provenance())
        bundle.provenance.update(
            solver="global_hamilton_kim_disk_replay",
            filter_state_reset="only once at full series start",
            smoothing_storage="owned disk filtered/predicted blocks; global reverse Kim recursion",
            score_factor_rows_are_observations=False,
            numeric_spool_hash=self.numeric_hash,
            resource_plans=self.frame.resource_plans,
            state_space_scratch_bytes=self.path.stat().st_size,
        )
        bundle.nobs_original = self.ordered.original_count
        bundle.dropped_rows = self.ordered.original_count - self.n
        bundle.sample_positions = []
        return bundle


def fit_streaming_mswitch(spec, source, *, batch_rows=None):
    if not isinstance(source, Dataset):
        raise AnalysisError(
            "invalid_data", "The streaming Markov-switching adapter needs a Dataset."
        )
    with (
        torch.no_grad(),
        torch.device("cpu"),
        OrderedReplay(spec, source) as ordered,
        _disk() as (sql, path),
    ):
        replay = _Replay(ordered, sql, path, batch_rows)
        bundle = kernel_call(_fit, replay)
        ordered.verify_original()
        return replay.finalize(bundle)


def _fit(replay):
    spec, frame, layout = replay.spec, replay.frame, replay.layout
    maximum = int(frame.option("max_iterations"))

    def objective(theta):
        out = replay.score(theta)
        return out["loglik"], out["gradient"]

    def gradient(theta):
        return replay.score(theta)["gradient"]

    def hessian_fn(theta):
        return numerical_hessian(gradient, theta, levels=1)

    screened = []
    attempts = []
    for start in replay.starts():
        try:
            result = kernel_call(
                maximize_bfgs,
                objective,
                start,
                max_iter=min(40, maximum),
                hessian_fn=hessian_fn,
                raise_on_failure=False,
            )
        except AnalysisError as exc:
            if exc.code in {"state_space_spill_failed", "workspace_limit", "source_changed"}:
                raise
            attempts.append({"converged": False, "message": str(exc)[:120]})
            continue
        value = float(result.value)
        attempts.append(
            {
                "converged": bool(result.converged),
                "log_likelihood": value,
                "iterations": result.iterations,
            }
        )
        if math.isfinite(value):
            screened.append((value, len(screened), result))
    best = None
    for _, _, result in sorted(screened, key=lambda item: (-item[0], item[1])):
        if not result.converged:
            try:
                result = kernel_call(
                    maximize_bfgs,
                    objective,
                    result.theta,
                    max_iter=maximum,
                    hessian_fn=hessian_fn,
                    raise_on_failure=False,
                )
            except AnalysisError as exc:
                if exc.code in {"state_space_spill_failed", "workspace_limit", "source_changed"}:
                    raise
                continue
        if result.converged:
            best = result
            break
    if best is None:
        raise AnalysisError(
            "nonconvergence",
            "Hamilton likelihood did not converge from residual-quantile starts; increase iterations or simplify switching terms/states.",
        )
    theta, _ = native._relabel(layout, best.theta)
    out = replay.score(theta)
    hessian = kernel_call(numerical_hessian, gradient, theta)
    scores = replay.score_factor(theta) if spec.covariance in {"robust", "opg"} else None
    covariance, info = ml_covariance(frame, hessian=hessian, scores=scores, nobs=replay.n)
    params = replay.multiplier * theta + replay.offset
    covariance = replay.multiplier[:, None] * covariance * replay.multiplier[None]
    terms, equations = native._terms(
        layout, replay.design, replay.switching, replay.common, spec.outcome
    )
    ll = float(out["loglik"]) - replay.n * math.log(replay.scale_y)
    extra = native._transition_summary(
        layout, params, covariance, out["transition"], out["ergodic"]
    )
    extra.update(
        state_order="states sorted by first switching coefficient"
        if layout.switching
        else "states sorted by sigma"
        if layout.varswitch
        else "as estimated",
        starts=attempts,
        switching_terms=[replay.design.terms[i] for i in replay.switching],
        ar_lags=layout.lags,
        arswitch=layout.arswitch,
        varswitch=layout.varswitch,
        smoothed_share=out["share"].tolist(),
        estimation_units={
            "outcome_scale": replay.scale_y,
            "column_scales": replay.scales.tolist(),
            "note": "exact standardized likelihood; reported original units",
        },
    )
    metrics = {**information_criteria(ll, layout.size, replay.n), "states": layout.states}
    for j in range(layout.states):
        metrics[f"duration_state{j + 1}"] = (
            1.0 / (1.0 - float(out["transition"][j, j]))
            if float(out["transition"][j, j]) < 1.0
            else None
        )
    return build_result(
        frame,
        terms=terms,
        params=params,
        covariance=covariance,
        title=f"Markov-switching ({layout.states} states)",
        equations=equations,
        use_t=False,
        nobs=replay.n,
        metrics=metrics,
        fitted=out["fitted"],
        solver="global_hamilton_kim_disk_replay",
        solver_diagnostics={"converged": True, "iterations": best.iterations},
        optimizer={
            "method": "bfgs",
            "gradient": "analytic full-series Fisher identity, disk Kim smoother",
            "hessian": "Ridders derivative of global analytic score",
            "starts": len(attempts),
        },
        inference=info,
        extra=extra,
        categories=replay.sample.categories,
        provenance={
            "likelihood": "exact; conditional on first P observations for AR; full ergodic initial-state derivative",
            "fitted_values": "one-step E[y_t|Y_(t-1)]; first400 retained periods",
        },
    )


def mswitch_probabilities_streaming(result, source):
    """All filtered/predicted/smoothed base-regime probabilities as a Dataset."""
    if result.spec.estimator != "mswitch" or not isinstance(source, Dataset):
        raise AnalysisError(
            "invalid_result", "Probability replay needs a fitted mswitch result and Dataset."
        )
    with (
        torch.no_grad(),
        torch.device("cpu"),
        OrderedReplay(result.spec, source) as ordered,
        _disk() as (sql, path),
    ):
        replay = _Replay(ordered, sql, path)
        terms, _ = native._terms(
            replay.layout, replay.design, replay.switching, replay.common, result.spec.outcome
        )
        if terms != [c.term for c in result.coefficients]:
            raise AnalysisError(
                "invalid_data", "Evaluation columns do not reproduce saved regime parameters."
            )
        theta = (
            torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
            - replay.offset
        ) / replay.multiplier
        # Smoother traverses reverse blocks; persist only K base probabilities,
        # then emit one bounded block at a time in chronological order.
        sql.execute(
            "CREATE TABLE output(id INTEGER PRIMARY KEY,n INTEGER,pred BLOB,filt BLOB,smooth BLOB,y BLOB)"
        )

        def sink(j, n, pred, filt, smooth, y):
            sql.execute(
                "INSERT INTO output VALUES(?,?,?,?,?,?)",
                (j, n, _bytes(pred), _bytes(filt), _bytes(smooth), _bytes(y)),
            )

        replay.score(theta, sink=sink)
        with StateOutput(
            {
                "title": "Regime probabilities",
                "states": replay.layout.states,
                "smoother": "global Kim (1994)",
            }
        ) as output:
            periods = ordered.db.execute(
                "SELECT t,pos FROM rows WHERE valid=1 ORDER BY t,pos LIMIT -1 OFFSET ?",
                (replay.layout.order,),
            )
            for _, n, pred, filt, smooth, y in sql.execute("SELECT * FROM output ORDER BY id"):
                records = periods.fetchmany(n)
                time = [row[0] if result.spec.time else row[1] for row in records]
                if ordered.date:
                    time = pd.to_datetime(time, unit="ns", utc=True)
                columns = {"period": time, "observed": _tensor(y, (n,)).numpy()}
                for name, payload in (
                    ("filtered", filt),
                    ("smoothed", smooth),
                    ("predicted", pred),
                ):
                    values = _tensor(payload, (n, replay.layout.states))
                    for j in range(replay.layout.states):
                        columns[f"{name}_state{j + 1}"] = values[:, j].numpy()
                output.write(pd.DataFrame(columns))
            ordered.verify_original()
            if replay.digest() != replay.numeric_hash:
                raise AnalysisError(
                    "source_changed", "Regime numeric source changed during smoothing."
                )
            return output.finish()
