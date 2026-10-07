"""Exact global diffuse Kalman ML, with bounded blocks and disk smoothing.

The [TS] ucm state/observation equations and exact diffuse likelihood are
unchanged. Only filter state is carried between blocks. Per-period covariance
records for the backward diffuse smoother live in owned SQLite, never N x m²
RAM. Finite-difference likelihood points share an entire-source replay.
https://www.stata.com/manuals/tsucm.pdf
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
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
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from . import registry
from .core import build_result, information_criteria, kernel_call, ml_covariance
from .ordered_replay import OrderedReplay
from .replay_sample import ReplaySample
from .streaming_linear import _Notes
from .streaming_residual_diagnostics import diagnostics
from .streaming_var import _Facade, _head
from .tsmodels import statespace as ss
from .tsmodels import ucm as native

SUPPORTED = frozenset({"ucm"})


def supports_spec(spec):
    return spec.estimator in SUPPORTED


# Fixed, trusted source is independent of inspect.getsource / PyInstaller.
_FORWARD = """
def forward(data: Tensor, z: Tensor, t: Tensor, q: Tensor, h: Tensor,
            a: Tensor, ps: Tensor, pi: Tensor, store: bool):
    b, c, n = a.size(0), a.size(1), data.size(1)
    m = z.numel()
    vout = torch.zeros((b, c, n), dtype=torch.float64)
    fout = torch.ones((b, n), dtype=torch.float64)
    inf = torch.zeros((n,), dtype=torch.bool)
    loginf = torch.zeros((n,), dtype=torch.float64)
    length = n if store else 0
    pred = torch.zeros((length, m), dtype=torch.float64)
    pss = torch.zeros((length, m, m), dtype=torch.float64)
    pis = torch.zeros((length, m, m), dtype=torch.float64)
    gains = torch.zeros((length, m), dtype=torch.float64)
    gains1 = torch.zeros((length, m), dtype=torch.float64)
    vr = torch.zeros((length,), dtype=torch.float64)
    fr = torch.ones((length,), dtype=torch.float64)
    kinds = torch.zeros((length,), dtype=torch.int64)
    qm = torch.diag_embed(q)
    tt = t.transpose(1, 2)
    for j in range(n):
        if m == 0:
            v = data[:, j].unsqueeze(0).expand(b, c)
            f = h
            kind = 0
            k0 = torch.zeros((b, m), dtype=torch.float64)
            k1 = torch.zeros((b, m), dtype=torch.float64)
        else:
            v = data[:, j].unsqueeze(0) - torch.matmul(a, z)
            ms = torch.matmul(ps, z)
            f = torch.matmul(ms, z) + h
            mi = torch.matmul(pi, z)
            fi_tensor = torch.dot(z, mi)
            fi = float(fi_tensor)
            tms = torch.matmul(t, ms.unsqueeze(-1)).squeeze(-1)
            tmi = torch.matmul(t[0], mi)
            diffuse = float(torch.max(torch.abs(pi))) > 1e-10
            if diffuse and fi > 1e-10:
                kind = 2
                k0 = (tmi / fi_tensor).unsqueeze(0).expand(b, m)
                k1 = tms / fi_tensor - k0 * (f / fi_tensor).unsqueeze(1)
                newpi = torch.matmul(torch.matmul(t[0], pi), t[0].T) - torch.outer(tmi, k0[0])
                newps = torch.matmul(torch.matmul(t, ps), tt) - tms.unsqueeze(2)*k0.unsqueeze(1) - tmi.unsqueeze(0).unsqueeze(2)*k1.unsqueeze(1) + qm
                inf[j] = True
                loginf[j] = torch.log(fi_tensor)
            else:
                kind = 1 if diffuse else 0
                if not bool(torch.all(torch.isfinite(f))) or not bool(torch.all(f > 0.)):
                    raise RuntimeError("invalid Kalman variance")
                k0 = tms / f.unsqueeze(1)
                k1 = torch.zeros((b, m), dtype=torch.float64)
                newpi = torch.matmul(torch.matmul(t[0], pi), t[0].T)
                newps = torch.matmul(torch.matmul(t, ps), tt) - tms.unsqueeze(2)*k0.unsqueeze(1) + qm
            if store:
                pred[j], pss[j], pis[j] = a[0, 0], ps[0], pi
            a = torch.matmul(a, tt) + v.unsqueeze(2)*k0.unsqueeze(1)
            ps = (newps + newps.transpose(1, 2)) / 2.
            pi = (newpi + newpi.T) / 2.
            if float(torch.max(torch.abs(pi))) <= 1e-10:
                pi = torch.zeros_like(pi)
        if kind != 2:
            vout[:, :, j], fout[:, j] = v, f
        if store:
            gains[j], gains1[j], vr[j], fr[j], kinds[j] = k0[0], k1[0], v[0,0], (torch.dot(z, torch.matmul(pis[j], z)) if kind == 2 else f[0]), kind
    return vout, fout, inf, loginf, a, ps, pi, pred, pss, pis, gains, gains1, vr, fr, kinds

def backward(z: Tensor, t: Tensor, pred: Tensor, ps: Tensor, pi: Tensor,
             gains: Tensor, gains1: Tensor, v: Tensor, f: Tensor, kinds: Tensor,
             r: Tensor, r1: Tensor):
    n, m = pred.size(0), pred.size(1)
    states = torch.zeros((n, m), dtype=torch.float64)
    for j in range(n-1, -1, -1):
        old = r
        if int(kinds[j]) == 2:
            r = torch.matmul(t.T, old) - z*torch.dot(gains[j], old)
            r1 = z*(v[j]/f[j]) + torch.matmul(t.T, r1) - z*torch.dot(gains[j], r1) - z*torch.dot(gains1[j], old)
        else:
            r = z*(v[j]/f[j]) + torch.matmul(t.T, old) - z*torch.dot(gains[j], old)
            if int(kinds[j]) == 1:
                r1 = torch.matmul(t.T, r1)
        states[j] = pred[j] + torch.matmul(ps[j], r) + torch.matmul(pi[j], r1)
    return states, r, r1
"""


@lru_cache(maxsize=1)
def _kernel():
    return torch.jit.CompilationUnit(_FORWARD)


def _bytes(value):
    return value.contiguous().numpy().tobytes()


def _tensor(value, shape, dtype=torch.float64):
    return (
        torch.frombuffer(bytearray(value), dtype=dtype).reshape(shape)
        if value
        else torch.empty(shape, dtype=dtype)
    )


@contextmanager
def _disk():
    try:
        with tempfile.TemporaryDirectory(
            prefix="openecon-ucm-", dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None
        ) as name:
            path = Path(name) / "states.sqlite"
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
                    "CREATE TABLE blocks(id INTEGER PRIMARY KEY,n INTEGER,pred BLOB,ps BLOB,pi BLOB,k0 BLOB,k1 BLOB,v BLOB,f BLOB,kinds BLOB,y BLOB,reg BLOB,states BLOB)"
                )
                path.chmod(0o600)
                yield sql, path
            finally:
                sql.close()
    except (OSError, sqlite3.Error) as exc:
        raise AnalysisError(
            "state_space_spill_failed",
            "Diffuse smoothing needs writable scratch storage and sufficient disk space for global state records.",
        ) from exc


class _Problem:
    def __init__(self, replay, structure, scale):
        self.replay, self.structure, self.scale = replay, structure, scale
        self.n, self.kx = replay.sample.nrows, len(replay.design.terms)

    def _state(self, theta):
        values = ss.natural(self.structure, theta, self.scale)
        t, q, h, p0 = ss.system(self.structure, values, len(theta))
        if not all(bool(torch.isfinite(v).all()) for v in (t, q, h, p0)):
            raise AnalysisError(
                "degenerate_model", "Kalman parameters are outside finite float64 precision."
            )
        return (
            t,
            q,
            h,
            torch.zeros((len(theta), 1 + self.kx, self.structure.m), dtype=torch.float64),
            p0,
            torch.diag(self.structure.diffuse),
        )

    def blocks(self, theta):
        t, q, h, a, ps, pi = self._state(theta)
        for x, y in self.replay.numeric():
            data = torch.cat((y[None], x.T), 0).contiguous()
            try:
                out = _kernel().forward(data, self.structure.z, t, q, h, a, ps, pi, False)
            except (RuntimeError, torch.jit.Error) as exc:
                raise AnalysisError(
                    "degenerate_model",
                    "Kalman prediction-error variance became zero or nonfinite; rescale inputs or simplify the model.",
                ) from exc
            v, f, inf, loginf, a, ps, pi = out[:7]
            if not bool(torch.isfinite(v).all()) or not bool(torch.isfinite(f).all()):
                raise AnalysisError(
                    "numerical_failure",
                    "Global Kalman innovations exceed finite float64 precision.",
                )
            yield x, v, f, inf, loginf
        if pi.numel() and float(pi.abs().max()) > 1e-10:
            raise AnalysisError(
                "insufficient_observations",
                "The diffuse states are not identified from the complete series.",
            )

    def pieces(self, theta, *, reduce=True):
        # Full exact-diffuse filtering already profiles the deterministic initial
        # states. No per-block initial-state elimination is performed.
        b, c = len(theta), 1 + self.kx
        self.replay.plan(b, self.structure, False)
        cross, logdet, raw = (
            _CompensatedSum((b, c, c)),
            _CompensatedSum((b,)),
            _CompensatedSum((b, self.kx)),
        )
        for x, v, f, inf, loginf in self.blocks(theta):
            weight = torch.where(~inf, 1.0 / f, torch.zeros_like(f))
            cross.add(torch.einsum("bcn,bdn->bcd", v * weight[:, None], v))
            logdet.add(torch.where(~inf, torch.log(f), torch.zeros_like(f)).sum(1) + loginf.sum())
            raw.add((weight[:, :, None] * x[None].square()).sum(1))
        value = cross.value
        return {
            "syy": value[:, 0, 0],
            "sxy": value[:, 1:, 0],
            "sxx": value[:, 1:, 1:],
            "logdet": logdet.value,
            "raw": raw.value,
        }

    def beta(self, piece):
        if not self.kx:
            return torch.zeros((len(piece["syy"]), 0), dtype=torch.float64)
        root = piece["raw"].clamp_min(1e-300).sqrt()
        scaled = piece["sxx"] / (root[:, :, None] * root[:, None, :])
        chol, info = torch.linalg.cholesky_ex(piece["sxx"])
        if bool((info != 0).any()) or bool((torch.linalg.eigvalsh(scaled)[:, 0] <= 1e-10).any()):
            raise AnalysisError(
                "collinear_regressors",
                "Regressors are collinear with the unobserved components; remove a constant or trend duplicated by diffuse states.",
            )
        return torch.cholesky_solve(piece["sxy"][..., None], chol).squeeze(-1)

    def profile(self, theta):
        piece = self.pieces(theta)
        quad = piece["syy"] - (piece["sxy"] * self.beta(piece)).sum(1)
        return -0.5 * (self.n * math.log(2.0 * math.pi) + piece["logdet"] + quad)

    def full(self, piece, beta):
        quad = (
            piece["syy"]
            - 2.0 * (piece["sxy"] * beta).sum(1)
            + torch.einsum("k,bkl,l->b", beta, piece["sxx"], beta)
        )
        return -0.5 * (self.n * math.log(2.0 * math.pi) + piece["logdet"] + quad)

    def score_factor(self, theta, beta):
        p = len(theta)
        h = native._steps(theta, 1.0 / 3.0)
        points = theta.repeat(2 * p + 1, 1)
        for j in range(p):
            points[1 + j, j] += h[j]
            points[1 + p + j, j] -= h[j]
        self.replay.plan(len(points), self.structure, False)
        tree = _TSQRTree()
        for _, v, f, inf, _ in self.blocks(points):
            residual = v[:, 0] - (
                torch.einsum("bkn,k->bn", v[:, 1:], beta) if beta.numel() else 0.0
            )
            used = ~inf
            contributions = torch.where(
                used, -0.5 * (torch.log(f) + residual.square() / f), torch.zeros_like(f)
            )
            scores = ((contributions[1 : p + 1] - contributions[p + 1 :]) / (2.0 * h[:, None])).T
            if beta.numel():
                beta_scores = (
                    v[0, 1:]
                    * (torch.where(used, 1.0 / f[0], torch.zeros_like(f[0])) * residual[0])[None]
                ).T
                scores = torch.cat((scores, beta_scores), 1)
            if not bool(torch.isfinite(scores).all()):
                raise AnalysisError(
                    "numerical_failure", "Kalman per-period scores exceed finite float64 precision."
                )
            tree.add(qr_factor(scores))
        return tree.finish()


class _Replay:
    def __init__(self, ordered):
        self.ordered, self.spec = ordered, ordered.spec
        self.notes = _Notes(self.spec)
        self.sample = ReplaySample(self.spec, ordered.source, batch_rows=ordered.rows)
        self.design = self.sample.add_design("mean", intercept=False)
        self.sample.prepare()
        self.n = self.sample.nrows
        self.report = pd.concat(
            list(
                _head(ordered.source.iter_batches(batch_rows=ordered.rows), min(ordered.count, 400))
            ),
            ignore_index=True,
        )
        self.positions = self.report.pop(ordered.positions_column).tolist()
        self.frame = SimpleNamespace(
            spec=self.spec,
            info=registry.get("ucm"),
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
            resource_plans=[],
        )

    def plan(self, b, structure, store):
        m, c, p = (
            structure.m,
            1 + len(self.design.terms),
            len(structure.parameters) + len(self.design.terms),
        )
        perrow = 8 * (b * (4 * c + 4) + (6 * m * m + 8 * m + 16 if store else 0))
        plan = self.sample.plan_rows(
            "global exact diffuse Kalman replay",
            {
                "filter_state_and_joint_information": 512 * b * (m * m + c * m + c * c)
                + 512 * p * p,
                "SQLite_order_and_smoother_caches": 12 * 1024**2,
                "bounded_spectral_start": 512 * 4096,
            },
            max(8, perrow),
        )
        self.frame.resource_plans = [self.ordered.plan.record(), plan.record()]
        return plan

    def numeric(self):
        for batch in self.sample.batches():
            yield self.sample.raw_design(batch, "mean"), batch.numeric(self.spec.outcome)

    def scale_start(self, structure, frequency):
        self.plan(1, structure, False)
        sumsq = _CompensatedSum(())
        moments = _WeightedMoments(2, intercept=True)
        seen = 0
        previous = None
        spectral = torch.zeros(2049, dtype=torch.float64)
        buffer = torch.empty(0, dtype=torch.float64)
        spectral_n = 0
        for _, y in self.numeric():
            moments.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
            if structure.flags[0]:
                values = torch.cat((previous, y)) if previous is not None else y
                delta = values[1:] - values[:-1]
                sumsq.add(delta.square().sum())
                seen += len(delta)
                previous = y[-1:].clone()
            if structure.cycle and frequency is None:
                # Start selection only: bounded nonoverlapping Welch windows.
                buffer = torch.cat((buffer, y))
                while len(buffer) >= 4096:
                    window = buffer[:4096]
                    buffer = buffer[4096:].clone()
                    window = window - window.mean()
                    spectral += torch.fft.rfft(window).abs().square()
                    spectral_n += 1
        scale = (
            float(sumsq.value / seen)
            if structure.flags[0]
            else float(moments.m2.value[1] * moments.magnitude[1] ** 2 / moments.mass)
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise AnalysisError(
                "constant_outcome", "The outcome or its first difference has no finite variation."
            )
        if structure.cycle and frequency is None:
            if spectral_n:
                spectral[:2] = 0.0
                frequency = 2.0 * math.pi * int(spectral.argmax()) / 4096
            else:
                window = buffer - buffer.mean()
                power = torch.fft.rfft(window).abs().square()
                power[:2] = 0.0
                frequency = 2.0 * math.pi * int(power.argmax()) / len(window)
            frequency = min(max(frequency, 4.0 * math.pi / self.n), 0.9 * math.pi)
        start = {
            "var(level)": 0.2 * scale,
            "var(slope)": 0.01 * scale,
            "var(seasonal)": 0.05 * scale,
            "var(cycle)": 0.2 * scale,
            "var(e)": 0.5 * scale,
            "damping": 0.9,
        }
        if structure.cycle:
            start["frequency"] = frequency
        return scale, start

    def finalize(self, bundle):
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
            solver="global_exact_diffuse_kalman",
            filter_state_reset="only once at full series start",
            smoothing_storage="owned disk covariance/state records; bounded reverse blocks",
            resource_plans=self.frame.resource_plans,
        )
        bundle.nobs_original = self.ordered.original_count
        bundle.dropped_rows = self.ordered.original_count - self.n
        bundle.sample_positions = []
        return bundle


def _smooth(replay, problem, theta, beta, sink=None):
    structure = problem.structure
    m = structure.m
    replay.plan(1, structure, True)
    values = ss.natural(structure, theta[None], problem.scale)
    t, q, h, ps = ss.system(structure, values, 1)
    a = torch.zeros((1, 1, m), dtype=torch.float64)
    pi = torch.diag(structure.diffuse)
    diffuse = 0
    used = 0
    signal = []
    with _disk() as (sql, path):
        for j, (x, y) in enumerate(replay.numeric()):
            reg = x @ beta if len(beta) else torch.zeros_like(y)
            out = _kernel().forward((y - reg)[None], structure.z, t, q, h, a, ps, pi, True)
            v, f, inf, _, a, ps, pi = out[:7]
            diffuse += int(inf.sum())
            used += len(y)
            records = out[7:]
            sql.execute(
                "INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                (j, len(y), *[_bytes(value) for value in records], _bytes(y), _bytes(reg)),
            )
        if pi.numel() and float(pi.abs().max()) > 1e-10:
            raise AnalysisError(
                "insufficient_observations", "Diffuse state remains unidentified at the series end."
            )
        sql.commit()
        end_state = {
            "mean": a[0, 0].tolist(),
            "covariance": ps[0].tolist(),
            "last_period": replay.ordered.last_period,
        }
        r = torch.zeros(m, dtype=torch.float64)
        r1 = torch.zeros_like(r)
        cursor = sql.execute("SELECT id,n,pred,ps,pi,k0,k1,v,f,kinds FROM blocks ORDER BY id DESC")
        for row in cursor:
            j, n, *payload = row
            shapes = [(n, m), (n, m, m), (n, m, m), (n, m), (n, m), (n,), (n,), (n,)]
            tensors = [
                _tensor(value, shape, torch.int64 if i == 7 else torch.float64)
                for i, (value, shape) in enumerate(zip(payload, shapes, strict=True))
            ]
            states, r, r1 = _kernel().backward(structure.z, t[0], *tensors, r, r1)
            sql.execute("UPDATE blocks SET states=? WHERE id=?", (_bytes(states), j))
        sql.commit()
        count = 0
        for n, state, reg, observed in sql.execute("SELECT n,states,reg,y FROM blocks ORDER BY id"):
            if sink is not None:
                sink(_tensor(state, (n, m)), _tensor(observed, (n,)), _tensor(reg, (n,)))
            take = min(n, 400 - count)
            if take > 0:
                signal.append(
                    (
                        _tensor(state, (n, m))[:take] @ structure.z + _tensor(reg, (n,))[:take]
                    ).clone()
                )
                count += take

        def residuals():
            for n, vr, fr, kinds in sql.execute("SELECT n,v,f,kinds FROM blocks ORDER BY id"):
                v = _tensor(vr, (n,))
                f = _tensor(fr, (n,))
                kind = _tensor(kinds, (n,), torch.int64)
                values = v[kind != 2] / f[kind != 2].sqrt()
                if values.numel():
                    yield values

        tests = (
            diagnostics(residuals, used - diffuse, max(1, min((used - diffuse) // 2 - 2, 40)))
            if used - diffuse > 10
            else {}
        )
        tests.pop("arch_lm", None)
        bytes_on_disk = path.stat().st_size
    return torch.cat(signal), end_state, diffuse, tests, bytes_on_disk


def fit_streaming_ucm(spec, source, *, batch_rows=None):
    if not isinstance(source, Dataset):
        raise AnalysisError("invalid_data", "The replay UCM adapter needs a Dataset.")
    with torch.no_grad(), torch.device("cpu"), OrderedReplay(spec, source) as ordered:
        replay = _Replay(ordered)
        if batch_rows is not None:
            if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
                raise AnalysisError("invalid_batch_rows", "batch_rows must be a positive integer.")
            replay.sample.rows = min(replay.sample.rows, batch_rows)
        bundle = kernel_call(_fit, replay)
        ordered.verify_original()
        return replay.finalize(bundle)


def _fit(replay):
    spec, frame, n = replay.spec, replay.frame, replay.n
    structure = native._structure_from(spec)
    if structure.seasonal == 1:
        raise AnalysisError("invalid_option", "seasonal must be a period of at least 2.")
    if not structure.parameters:
        raise AnalysisError("invalid_model", "The UC model has no stochastic component.")
    if n < structure.m + len(structure.parameters) + len(replay.design.terms) + 3:
        raise AnalysisError(
            "insufficient_observations", "Too few periods for the complete UC specification."
        )
    frequency = frame.option("cycle_frequency")
    if frequency is not None and not 0.0 < frequency < math.pi:
        raise AnalysisError(
            "invalid_option", "cycle_frequency must lie strictly between zero and pi."
        )
    scale, start = replay.scale_start(structure, frequency)
    fixed = set()
    iterations = 0
    unconverged_value = None
    while True:
        structure = ss.Structure(
            structure.model, structure.seasonal, structure.cycle, frozenset(fixed)
        )
        if not any(name.startswith("var") for name in structure.parameters):
            raise AnalysisError("degenerate_model", "Every stochastic variance converged to zero.")
        problem = _Problem(replay, structure, scale)
        result = native._maximize(
            problem, ss.unconstrained(structure, start, scale), int(frame.option("max_iterations"))
        )
        iterations += result.iterations
        if unconverged_value is not None and float(result.value) < unconverged_value - 1e-6 * max(
            1.0, abs(unconverged_value)
        ):
            raise AnalysisError(
                "nonconvergence",
                "The constrained variance maximum is below the previous likelihood.",
            )
        values = {
            name: float(value[0])
            for name, value in ss.natural(structure, result.theta[None], scale).items()
        }
        small = [
            name
            for name in structure.parameters
            if name.startswith("var") and values[name] < native._ZERO * scale
        ]
        if not small and not result.converged:
            small = [
                name
                for name in structure.parameters
                if name.startswith("var") and values[name] < native._LOOSE_ZERO * scale
            ]
            unconverged_value = float(result.value) if small else None
        if structure.cycle and ("var(cycle)" in small or values["damping"] > 1.0 - 1e-6):
            raise AnalysisError(
                "degenerate_cycle",
                "The cycle converged to an undamped sinusoid or zero cycle variance; change frequency or drop the cycle.",
            )
        if not small:
            break
        fixed.update(small)
        start = {**start, **values}
    if not result.converged:
        raise AnalysisError(
            "nonconvergence",
            "UC likelihood did not converge; increase iterations or simplify components.",
        )
    for name in sorted(fixed):
        frame.warn(f"{name} converged to zero and is fixed at its boundary.")
    theta = result.theta
    p = len(theta)
    kx = len(replay.design.terms)
    points, h, pairs = native._hessian_points(theta)
    piece = problem.pieces(points)
    beta = problem.beta({name: value[:1] for name, value in piece.items()})[0]
    values_full = problem.full(piece, beta)
    ll = float(values_full[0])
    hessian = torch.zeros((p + kx, p + kx), dtype=torch.float64)
    hessian[:p, :p] = native._second_differences(values_full, h, pairs, p)
    if kx:
        sb = piece["sxy"] - torch.einsum("bkl,l->bk", piece["sxx"], beta)
        for j in range(p):
            hessian[j, p:] = hessian[p:, j] = (sb[1 + 2 * j] - sb[2 + 2 * j]) / (2.0 * h[j])
        hessian[p:, p:] = -piece["sxx"][0]
    scores = problem.score_factor(theta, beta) if spec.covariance in {"robust", "opg"} else None
    covariance, info = ml_covariance(frame, hessian=hessian, scores=scores, nobs=n)
    jac = torch.cat(
        (ss.jacobian_diagonal(structure, theta, scale), torch.ones(kx, dtype=torch.float64))
    )
    covariance = jac[:, None] * covariance * jac[None]
    order = [*range(p, p + kx), *range(p)]
    covariance = covariance[order][:, order]
    params = torch.tensor(
        [*beta.tolist(), *[values[name] for name in structure.parameters]], dtype=torch.float64
    )
    signal, end_state, diffuse, tests, disk_bytes = _smooth(replay, problem, theta, beta)
    terms = [*replay.design.terms, *[f"/{name}" for name in structure.parameters]]
    extra = {
        "model": structure.model,
        "seasonal": structure.seasonal,
        "cycle": structure.cycle,
        "variances": {
            name: 0.0 if name in fixed else values.get(name)
            for name in structure.all_parameters()
            if name.startswith("var")
        },
        "fixed_zero": sorted(fixed),
        "cycle_parameters": {
            "frequency": values["frequency"],
            "damping": values["damping"],
            "period": 2.0 * math.pi / values["frequency"],
        }
        if structure.cycle
        else None,
        "last_state": end_state,
        "state_names": native._state_names(structure),
        "scale": scale,
        "smoothing_scratch_bytes": disk_bytes,
    }
    bundle = build_result(
        frame,
        terms=terms,
        params=params,
        covariance=covariance,
        title=f"Unobserved-components model ({structure.model})",
        use_t=False,
        nobs=n,
        metrics={
            **information_criteria(ll, p + kx, n),
            "diffuse_periods": diffuse,
            "iterations": iterations,
        },
        fitted=signal,
        solver="global_exact_diffuse_kalman",
        solver_diagnostics={"converged": True, "iterations": iterations},
        optimizer={
            "method": result.method,
            "gradient": "central differences of full-series exact diffuse likelihood",
            "hessian": "central second differences, complete parameter cross-covariance",
        },
        inference=info,
        tests=tests,
        extra=extra,
        categories=replay.sample.categories,
        provenance={
            "score_factor_rows_are_observations": False,
            "likelihood": "exact diffuse Kalman filter, no chunk resets",
            "fitted_values": "smoothed full-series signal; first400 retained periods",
            "cycle_initial_frequency": "explicit cycle_frequency"
            if frequency is not None
            else "bounded Welch spectrum initialization only; exact likelihood maximized"
            if structure.cycle
            else None,
        },
    )
    for coefficient in bundle.coefficients:
        if coefficient.term.startswith("/var("):
            coefficient.p_value *= 0.5
            coefficient.ci_low = max(0.0, coefficient.ci_low)
    return bundle


def ucm_components_streaming(result, source):
    """All smoothed periods as an owned disk-backed Dataset (native components)."""
    from .streaming_state_output import StateOutput

    if not isinstance(source, Dataset):
        raise AnalysisError("invalid_data", "Streaming components need a Dataset.")
    with torch.no_grad(), torch.device("cpu"), OrderedReplay(result.spec, source) as ordered:
        structure, values, beta = native._fitted_structure(result)
        replay = _Replay(ordered)
        terms = [c.term for c in result.coefficients if not c.term.startswith("/")]
        if replay.design.terms != terms:
            raise AnalysisError(
                "invalid_data", "Evaluation columns do not reproduce the saved UC regressors."
            )
        theta = ss.unconstrained(
            structure,
            {name: float(value[0]) for name, value in values.items()},
            result.extra["scale"],
        )
        problem = _Problem(replay, structure, result.extra["scale"])
        periods = ordered.db.execute("SELECT t,pos FROM rows WHERE valid=1 ORDER BY t,pos")
        names = native._state_names(structure)
        with StateOutput(
            {
                "title": "Smoothed UC components",
                "model": structure.model,
                "smoother": "exact diffuse full-series fixed interval",
            }
        ) as output:

            def sink(states, observed, regression):
                records = periods.fetchmany(len(observed))
                time = [row[0] if result.spec.time else row[1] for row in records]
                if ordered.date:
                    time = pd.to_datetime(time, unit="ns", utc=True)
                columns = {"period": time, "observed": observed.numpy()}
                for name in ("level", "slope"):
                    if name in names:
                        columns[name] = states[:, names.index(name)].numpy()
                if structure.seasonal:
                    columns["seasonal"] = states[:, structure.season_index].numpy()
                if structure.cycle:
                    columns["cycle"] = states[:, structure.cycle_index].numpy()
                if beta.numel():
                    columns["regression"] = regression.numpy()
                fitted = states @ structure.z + regression
                columns.update(fitted=fitted.numpy(), irregular=(observed - fitted).numpy())
                output.write(pd.DataFrame(columns))

            _smooth(replay, problem, theta, beta, sink=sink)
            ordered.verify_original()
            return output.finish()


def _end_state(replay, structure, theta, beta, scale):
    replay.plan(1, structure, False)
    values = ss.natural(structure, theta[None], scale)
    t, q, h, ps = ss.system(structure, values, 1)
    a = torch.zeros((1, 1, structure.m), dtype=torch.float64)
    pi = torch.diag(structure.diffuse)
    for x, y in replay.numeric():
        adjusted = y - x @ beta if beta.numel() else y
        try:
            out = _kernel().forward(adjusted[None], structure.z, t, q, h, a, ps, pi, False)
        except (RuntimeError, torch.jit.Error) as exc:
            raise AnalysisError(
                "numerical_failure", "Global forecast Kalman state is outside finite precision."
            ) from exc
        a, ps, pi = out[4:7]
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(ps).all()):
        raise AnalysisError("numerical_failure", "Global forecast state/covariance is nonfinite.")
    if pi.numel() and float(pi.abs().max()) > 1e-10:
        raise AnalysisError(
            "insufficient_observations", "Diffuse state is unidentified in evaluation data."
        )
    return {
        "mean": a[0, 0].tolist(),
        "covariance": ps[0].tolist(),
        "last_period": replay.ordered.last_period,
    }


def ucm_forecast_streaming(result, steps, *, data, exog=None, alpha=0.05):
    """Reuse native forecast covariance after a global Dataset filter/smoother."""
    with torch.no_grad(), torch.device("cpu"), OrderedReplay(result.spec, data) as ordered:
        structure, values, beta = native._fitted_structure(result)
        replay = _Replay(ordered)
        terms = [c.term for c in result.coefficients if not c.term.startswith("/")]
        if replay.design.terms != terms:
            raise AnalysisError(
                "invalid_data", "Evaluation columns do not reproduce the saved UC regressors."
            )
        theta = ss.unconstrained(
            structure,
            {name: float(value[0]) for name, value in values.items()},
            result.extra["scale"],
        )
        state = _end_state(replay, structure, theta, beta, result.extra["scale"])
        ordered.verify_original()
        copy = result.model_copy(deep=True)
        copy.extra["last_state"] = state
        return native.ucm_forecast(copy, steps, exog=exog, alpha=alpha)
