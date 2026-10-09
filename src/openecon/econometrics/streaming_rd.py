"""Global native local-polynomial RD with bounded moments and disk tied-value metadata."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
import sqlite3

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .replay_sample import ReplaySample
from .streaming_linear import _GroupMeans, _Notes, _finite, _result, _scratch_directory
from .teffects.rd import _pair
from .teffects import rdkernels as rk, rdbandwidth as bw

SUPPORTED = frozenset({"rdrobust"})


def supports(spec):
    return spec.estimator in SUPPORTED


@dataclass
class _Polynomial:
    order: int
    bandwidth: float
    beta: torch.Tensor
    inverse: torch.Tensor
    count: int
    distinct: int


class _Store:
    """One owned tied-value table; no global running-value or group array."""

    def __init__(self, sample, width):
        self.sample, self.width = sample, width
        self.groups = _GroupMeans(width, 0)
        self.db = self.groups.connection
        self.db.execute(
            "CREATE TABLE running(dx REAL PRIMARY KEY,key BLOB,side INTEGER,n INTEGER,cumulative INTEGER) WITHOUT ROWID"
        )
        self.db.execute("CREATE INDEX running_side ON running(side,dx)")
        self.nn_cache = set()

    def seed(self, context):
        for batch in self.sample.batches():
            dx, response = context.values(batch)
            unique, codes, counts = torch.unique(
                dx, sorted=True, return_inverse=True, return_counts=True
            )
            keys = encode_cluster_labels(pd.Series(unique.numpy()))
            allkeys = [keys[i] for i in codes.tolist()]
            self.groups.add(allkeys, response, torch.ones(len(dx), dtype=torch.float64), [])
            self.db.executemany(
                "INSERT INTO running VALUES(?,?,?,?,NULL) ON CONFLICT(dx) DO UPDATE SET n=n+excluded.n",
                (
                    (float(value), key, int(value >= 0), int(count))
                    for value, key, count in zip(unique, keys, counts, strict=True)
                ),
            )
        self.groups.finish()
        self.db.execute(
            "CREATE TABLE cumulative AS SELECT dx,SUM(n) OVER(ORDER BY dx) AS value FROM running"
        )
        self.db.execute("CREATE UNIQUE INDEX cumulative_dx ON cumulative(dx)")
        self.db.execute(
            "UPDATE running SET cumulative=(SELECT value FROM cumulative WHERE cumulative.dx=running.dx)"
        )
        self.db.execute("DROP TABLE cumulative")
        self.db.execute("CREATE INDEX running_cumulative ON running(cumulative)")
        self.db.execute(
            "CREATE TABLE neighbors(side INTEGER,bound REAL,dx REAL,n INTEGER,totals BLOB,PRIMARY KEY(side,bound,dx)) WITHOUT ROWID"
        )
        self.db.commit()
        self.counts = []
        self.ranges = []
        for side in [0, 1]:
            n, low, high = self.db.execute(
                "SELECT SUM(n),MIN(dx),MAX(dx) FROM running WHERE side=?", (side,)
            ).fetchone()
            if not n or n < 2:
                raise AnalysisError("invalid_cutoff", "RD cutoff needs observations on both sides.")
            self.counts.append(int(n))
            self.ranges.append(max(abs(low), abs(high)))

    def _predicate(self, side, bound, kernel):
        relation = "<=" if kernel == "uniform" else "<"
        return "side=? AND ABS(dx)" + relation + "?", (side, bound)

    def count(self, side, bound, kernel):
        clause, args = self._predicate(side, bound, kernel)
        n, k = self.db.execute(
            "SELECT SUM(n),COUNT(*) FROM running WHERE " + clause, args
        ).fetchone()
        return int(n or 0), int(k)

    def quantile(self, prob):
        n = sum(self.counts)
        value = n * prob
        j = math.floor(value)
        indices = [min(j, n - 1)] if value - j > 1e-12 else [max(j - 1, 0), min(j, n - 1)]
        return sum(
            self.db.execute(
                "SELECT dx FROM running WHERE cumulative>? ORDER BY cumulative LIMIT 1", (i,)
            ).fetchone()[0]
            for i in indices
        ) / len(indices)

    def _neighbors(self, side, bound, kernel, matches):
        cache = (side, float(bound), kernel, matches)
        if cache in self.nn_cache:
            return
        n, _ = self.count(side, bound, kernel)
        target = min(matches, n - 1)
        if target < 1:
            raise AnalysisError(
                "insufficient_observations", "NN residuals need two local observations per side."
            )
        clause, args = self._predicate(side, bound, kernel)
        cursor = self.db.execute(
            "SELECT r.dx,r.n,m.value FROM running r JOIN means m ON m.key=r.key WHERE "
            + clause.replace("side", "r.side").replace("ABS(dx)", "ABS(r.dx)")
            + " ORDER BY r.dx",
            args,
        )
        step = max(1, min(4096, self.sample.rows))
        previous, pending = [], []
        while True:
            block = pending + cursor.fetchmany(max(0, step - len(pending)))
            if not block:
                break
            following = cursor.fetchmany(target)
            rows = previous + block + following
            values = torch.tensor([row[0] for row in rows], dtype=torch.float64)
            counts = torch.tensor([row[1] for row in rows], dtype=torch.int64)
            packed = torch.frombuffer(
                bytearray(b"".join(row[2] for row in rows)), dtype=torch.float64
            ).reshape(-1, self.width + 1)
            totals = packed[:, 1:] * counts[:, None]
            cumulative = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
            left, right = torch.arange(len(rows)), torch.arange(len(rows))
            for _ in range(target):
                active = cumulative[right + 1] - cumulative[left] - 1 < target
                hasleft, hasright = left > 0, right < len(rows) - 1
                dl = torch.where(
                    hasleft,
                    values - values[(left - 1).clamp_min(0)],
                    torch.full_like(values, math.inf),
                )
                dr = torch.where(
                    hasright,
                    values[(right + 1).clamp_max(len(rows) - 1)] - values,
                    torch.full_like(values, math.inf),
                )
                tied=hasleft & hasright & ((dl-dr).abs()<=torch.maximum(dl,dr)*math.sqrt(torch.finfo(torch.float64).eps))
                left -= (active & hasleft & ((dl < dr) | tied)).to(torch.int64)
                right += (active & hasright & ((dr < dl) | tied)).to(torch.int64)
            begin, end = len(previous), len(previous) + len(block)
            prefix = torch.cat(
                (torch.zeros((1, self.width), dtype=torch.float64), totals.cumsum(0))
            )
            windows = prefix[right[begin:end] + 1] - prefix[left[begin:end]]
            size = cumulative[right[begin:end] + 1] - cumulative[left[begin:end]]
            payload = memoryview(windows.contiguous().numpy().tobytes())
            stride = 8 * self.width
            self.db.executemany(
                "INSERT OR REPLACE INTO neighbors VALUES(?,?,?,?,?)",
                (
                    (side, bound, row[0], int(size[i]), payload[i * stride : (i + 1) * stride])
                    for i, row in enumerate(block)
                ),
            )
            previous = (previous + block)[-target:]
            pending = following
        self.db.commit()
        self.nn_cache.add(cache)

    def residuals(self, side, bound, kernel, matches, dx, response):
        self._neighbors(side, bound, kernel, matches)
        unique, codes = torch.unique(dx, sorted=True, return_inverse=True)
        found = {}
        values = unique.tolist()
        for start in range(0, len(values), 500):
            part = values[start : start + 500]
            found.update(
                (row[0], (row[1], row[2]))
                for row in self.db.execute(
                    "SELECT dx,n,totals FROM neighbors WHERE side=? AND bound=? AND dx IN("
                    + ",".join("?" for _ in part)
                    + ")",
                    [side, bound, *part],
                )
            )
        if len(found) != len(values):
            raise AnalysisError("source_changed", "Local running values changed during RD replay.")
        counts = torch.tensor([found[value][0] for value in values], dtype=torch.float64)[codes] - 1
        totals = torch.frombuffer(
            bytearray(b"".join(found[value][1] for value in values)), dtype=torch.float64
        ).reshape(-1, self.width)[codes]
        return (counts / (counts + 1)).sqrt()[:, None] * (
            response - (totals - response) / counts[:, None]
        )

    def diagnostics(self):
        return {
            "running_value_groups": self.groups.groups,
            "running_metadata_disk_bytes": self.groups.path.stat().st_size,
            "nearest_neighbor_storage": "complete tied-value window totals on owned SQLite; bounded vectorized halo blocks",
            "running_metadata_memory": "bounded decoded groups and SQLite cache; no N-vector",
        }

    def close(self):
        self.groups.close()


class _Context:
    def __init__(self, sample, notes):
        self.sample, self.notes = sample, notes
        self.running = registry.role_columns(sample.spec, "running")[0]
        self.fuzzy = registry.role_columns(sample.spec, "fuzzy")
        self.kernel = notes.option("kernel")
        self.vce = notes.option("vce")
        self.matches = int(notes.option("nnmatch"))
        self.cutoff = float(notes.option("cutoff"))
        self.outputs = [sample.spec.outcome, *self.fuzzy]
        self.width = len(self.outputs)
        self.clusters = (
            registry.cluster_columns(sample.spec) if sample.spec.covariance == "cluster" else []
        )
        if len(self.clusters) > 1:
            raise AnalysisError("cluster_dimensions", "RD supports one declared cluster column.")
        moments = _WeightedMoments(self.width + 2, intercept=True)
        for batch in sample.batches():
            cols = [batch.numeric(name) for name in [self.running, *self.outputs]]
            moments.add(
                torch.stack([torch.ones(len(batch.frame), dtype=torch.float64), *cols], 1),
                torch.ones(len(batch.frame), dtype=torch.float64),
                False,
            )
        self.anchor, self.magnitude, self.center = (
            moments.anchor[2:],
            moments.magnitude[2:],
            moments.mean[2:],
        )
        self.magnitude = torch.where(
            self.magnitude > 0, self.magnitude, torch.ones_like(self.magnitude)
        )
        self.scale = self.magnitude * (moments.m2.value[2:] / moments.mass).clamp_min(0).sqrt()
        self.scale = torch.where(self.scale > 0, self.scale, torch.ones_like(self.scale))
        self.mean = self.anchor + self.magnitude * self.center
        self.x_sd = float(
            moments.magnitude[1] * torch.sqrt(moments.m2.value[1] / (sample.nrows - 1))
        )

    def values(self, batch):
        dx = batch.numeric(self.running) - self.cutoff
        y = torch.stack([batch.numeric(name) for name in self.outputs], 1)
        response = ((y - self.anchor) / self.magnitude - self.center) * (
            self.magnitude / self.scale
        )
        _finite(dx, response)
        return dx, response

    def blocks(self, side, bound):
        for batch in self.sample.batches():
            dx, y = self.values(batch)
            mask = (dx >= 0) if side else (dx < 0)
            if bound is not None:
                mask &= rk.kernel_weights(dx, bound, self.kernel) > 0
            if not bool(mask.any()):
                continue
            labels = encode_cluster_labels(batch.frame[self.clusters[0]]) if self.clusters else None
            yield (
                dx[mask],
                y[mask],
                None
                if labels is None
                else [key for key, keep in zip(labels, mask.tolist(), strict=True) if keep],
            )

    def fit(self, side, bandwidth, order):
        n, distinct = self.store.count(side, bandwidth, self.kernel)
        if distinct <= order:
            raise AnalysisError(
                "insufficient_observations",
                "RD needs more distinct positive-weight running values than polynomial order on each side.",
            )
        tree = _TSQRTree()
        for dx, y, _ in self.blocks(side, bandwidth):
            design = rk.powers(dx / bandwidth, order)
            weights = rk.kernel_weights(dx, bandwidth, self.kernel)
            block = torch.cat((design, y), 1) * weights.sqrt()[:, None]
            _finite(block)
            tree.add(torch.linalg.qr(block, mode="r")[1])
        factor = tree.finish()
        fit = least_squares(
            factor[:, : order + 1], factor[:, order + 1 :], drop_collinear=False, tol=0.0
        )
        return _Polynomial(order, bandwidth, fit.beta, fit.xtx_inv, n, distinct)

    def raw_coefficient(self, fit, degree):
        result = fit.beta[degree] * self.scale / fit.bandwidth**degree
        if degree == 0:
            result += self.mean
        _finite(result)
        return result

    def covariance(self, side, main, bias=None, *, degree=0):
        """Coefficient response covariance (normalized running-variable basis)."""
        bound = max(main.bandwidth, bias.bandwidth) if bias else main.bandwidth
        n, _ = self.store.count(side, bound, self.kernel)
        moment = _CompensatedSum((main.order + 1,))
        if bias:
            for dx, _, _ in self.blocks(side, main.bandwidth):
                x = rk.powers(dx / main.bandwidth, main.order)
                w = rk.kernel_weights(dx, main.bandwidth, self.kernel)
                moment.add((x * w[:, None]).T @ (dx / main.bandwidth) ** (main.order + 1))
        cl, rb = (
            _CompensatedSum((self.width, self.width)),
            _CompensatedSum((self.width, self.width)),
        )
        with ExitStack() as stack:
            accumulators = []
            if self.clusters:
                for _ in range(2 if bias else 1):
                    acc = ClusterAccumulator(self.width, scratch_directory=_scratch_directory())
                    stack.callback(acc.close)
                    accumulators.append(acc)
            for dx, y, keys in self.blocks(side, bound):
                xp = rk.powers(dx / main.bandwidth, main.order)
                wp = rk.kernel_weights(dx, main.bandwidth, self.kernel)
                rows = xp * wp[:, None]
                qrows = rows
                residual = y - xp @ main.beta
                residb = residual
                if bias:
                    xq = rk.powers(dx / bias.bandwidth, bias.order)
                    wb = rk.kernel_weights(dx, bias.bandwidth, self.kernel)
                    projection = wb * (xq @ bias.inverse[:, main.order + 1])
                    qrows = (
                        rows
                        - (main.bandwidth / bias.bandwidth) ** (main.order + 1)
                        * projection[:, None]
                        * moment.value
                    )
                    residb = y - xq @ bias.beta
                if self.vce == "nn":
                    residual = self.store.residuals(side, bound, self.kernel, self.matches, dx, y)
                    residb = residual
                elif self.vce == "hc1":
                    if n <= main.order + 1 or bias and n <= bias.order + 1:
                        raise AnalysisError(
                            "insufficient_observations",
                            "Local HC1 needs residual degrees of freedom.",
                        )
                    residual *= math.sqrt(n / (n - main.order - 1))
                    residb = residb * math.sqrt(n / (n - bias.order - 1)) if bias else residual
                elif self.vce in {"hc2", "hc3"}:
                    leverage = (xp @ main.inverse * xp).sum(1) * wp
                    if bool((leverage >= 1).any()):
                        raise AnalysisError(
                            "numerical_failure", "Local HC2/HC3 is undefined at leverage one."
                        )
                    factor = (
                        (1 - leverage).rsqrt() if self.vce == "hc2" else (1 - leverage).reciprocal()
                    )
                    residual *= factor[:, None]
                    residb = residb * factor[:, None] if bias else residual
                elif self.vce != "hc0":
                    raise AnalysisError("invalid_spec", "Unknown RD variance residual rule.")
                scores = (rows @ main.inverse[degree])[:, None] * (residual * self.scale)
                scoresrb = (qrows @ main.inverse[degree])[:, None] * (residb * self.scale)
                _finite(scores, scoresrb)
                if accumulators:
                    accumulators[0].add(keys, scores)
                    if bias:
                        accumulators[1].add(keys, scoresrb)
                else:
                    cl.add(scores.T @ scores)
                    rb.add(scoresrb.T @ scoresrb)
            if accumulators:
                matrix, g = accumulators[0].finish()
                if not n > main.order + 1:
                    raise AnalysisError(
                        "insufficient_observations",
                        "Cluster RD needs local residual degrees of freedom.",
                    )
                factor = (n - 1) / (n - main.order - 1) * g / (g - 1)
                cl.add(matrix * factor)
                rb.add(accumulators[1].finish()[0] * factor if bias else matrix * factor)
        return cl.value, rb.value

    def bias_corrected(self, side, main, bias):
        total = _CompensatedSum((main.order + 1, self.width))
        moment = _CompensatedSum((main.order + 1,))
        for dx, _, _ in self.blocks(side, main.bandwidth):
            x = rk.powers(dx / main.bandwidth, main.order)
            w = rk.kernel_weights(dx, main.bandwidth, self.kernel)
            moment.add((x * w[:, None]).T @ (dx / main.bandwidth) ** (main.order + 1))
        for dx, y, _ in self.blocks(side, max(main.bandwidth, bias.bandwidth)):
            xp = rk.powers(dx / main.bandwidth, main.order)
            rows = xp * rk.kernel_weights(dx, main.bandwidth, self.kernel)[:, None]
            xq = rk.powers(dx / bias.bandwidth, bias.order)
            projection = rk.kernel_weights(dx, bias.bandwidth, self.kernel) * (
                xq @ bias.inverse[:, main.order + 1]
            )
            qrows = (
                rows
                - (main.bandwidth / bias.bandwidth) ** (main.order + 1)
                * projection[:, None]
                * moment.value
            )
            total.add(qrows.T @ y)
        return self.mean + (main.inverse @ total.value)[0] * self.scale

    def constants(self, side, o, nu, ob, hv, hb, scale, fuzzy):
        main = self.fit(side, hv, o)
        combine = torch.zeros(self.width, dtype=torch.float64)
        if fuzzy:
            ty, tt = self.raw_coefficient(main, nu).tolist()
            if tt == 0 or not math.isfinite(ty / tt**2):
                raise AnalysisError(
                    "bandwidth_selection_failed",
                    "Fuzzy pilot coefficient must identify the treatment ratio; give h or sharpbw.",
                )
            combine[0], combine[1] = 1 / tt, -ty / tt**2
        else:
            combine[0] = 1.0
        cov, _ = self.covariance(side, main, degree=nu)
        variance = (2 * nu + 1) * hv * float(combine @ cov @ combine)
        moment = _CompensatedSum((o + 1,))
        for dx, _, _ in self.blocks(side, hv):
            x = rk.powers(dx / hv, o)
            moment.add(
                (x * rk.kernel_weights(dx, hv, self.kernel)[:, None]).T @ (dx / hv) ** (o + 1)
            )
        bc = float((main.inverse @ moment.value)[nu])
        other = self.fit(side, hb, ob)
        bias = (
            math.sqrt(2 * (o + 1 - nu)) * bc * float(self.raw_coefficient(other, o + 1) @ combine)
        )
        regular = 0.0
        if scale > 0:
            cov, _ = self.covariance(side, other, degree=o + 1)
            vb = float(combine @ cov @ combine) / hb ** (2 * (o + 1))
            regular = 3 * bc * bc * vb
        return bw.Constants(variance, bias, scale * 2 * (o + 1 - nu) * regular, 1 / (2 * o + 3))

    def select(self, p, q):
        store, notes = self.store, self.notes
        maximum = max(store.ranges)
        iqr = store.quantile(0.75) - store.quantile(0.25)
        spread = min(self.x_sd, iqr / 1.349) if iqr > 0 else self.x_sd
        pilot = bw.PILOT_CONSTANTS[self.kernel] * spread * self.sample.nrows**-0.2
        unique=[int(store.db.execute("SELECT COUNT(*) FROM running WHERE side=?",(side,)).fetchone()[0]) for side in (0,1)]
        bwcheck=notes.option("bwcheck")
        if notes.option("masspoints")=="adjust":
            pilot *= (self.sample.nrows/sum(unique))**.2
            if bwcheck is None and any(1-unique[i]/store.counts[i]>=.2 for i in (0,1)):
                bwcheck=10
        floors=[0.,0.]
        if bwcheck is not None:
            floors=[float(store.db.execute("SELECT ABS(dx) FROM running WHERE side=? ORDER BY ABS(dx) LIMIT 1 OFFSET ?",(side,min(bwcheck,unique[side])-1)).fetchone()[0]) for side in (0,1)]
        restricted = bool(notes.option("bwrestrict"))
        pilot = min(pilot, maximum) if restricted else pilot
        pilot=max(pilot,*floors)
        method = notes.option("bwselect")
        base = method.replace("cer", "mse")
        fuzzy = bool(self.fuzzy) and not bool(notes.option("sharpbw"))
        # Perfect compliance on either side selects sharp bandwidths.
        if fuzzy:
            for side in [0, 1]:
                low, high = math.inf, -math.inf
                for _, y, _ in self.blocks(side, None):
                    low, high = min(low, float(y[:, 1].min())), max(high, float(y[:, 1].max()))
                if low == high:
                    fuzzy = False

        def cap(value, limit):
            return min(value, limit) if restricted else value

        def step(o, nu, ob, hb, regular):
            return [
                self.constants(side, o, nu, ob, pilot, hb[side], regular, fuzzy) for side in [0, 1]
            ]

        dl, dr = step(q + 1, q + 1, q + 2, [value + 1e-8 for value in store.ranges], 0.0)
        results = {}
        regular = float(notes.option("scaleregul"))

        def joint(sign, name):
            d = cap(bw._ratio(dl.V + dr.V, (dr.B + sign * dl.B) ** 2, 0.0, dl.rate, name), maximum)
            d=max(d,*floors)
            bl, br = step(q, p + 1, q + 1, [d, d], regular)
            b = cap(
                bw._ratio(
                    bl.V + br.V, (br.B + sign * bl.B) ** 2, regular * (bl.R + br.R), bl.rate, name
                ),
                maximum,
            )
            hl, hr = step(p, 0, q, [b, b], regular)
            h = cap(
                bw._ratio(
                    hl.V + hr.V, (hr.B + sign * hl.B) ** 2, regular * (hl.R + hr.R), hl.rate, name
                ),
                maximum,
            )
            results[name] = ((h, h), (b, b))

        if base in {"mserd", "msecomb1", "msecomb2"}:
            joint(-1.0, "mserd")
        if base in {"msesum", "msecomb1", "msecomb2"}:
            joint(1.0, "msesum")
        if base in {"msetwo", "msecomb2"}:
            d = [
                cap(bw._ratio(value.V, value.B**2, 0.0, value.rate, "msetwo"), store.ranges[i])
                for i, value in enumerate([dl, dr])
            ]
            d=[max(value,floors[i]) for i,value in enumerate(d)]
            bvals = step(q, p + 1, q + 1, d, regular)
            b = [
                cap(
                    bw._ratio(value.V, value.B**2, regular * value.R, value.rate, "msetwo"),
                    store.ranges[i],
                )
                for i, value in enumerate(bvals)
            ]
            hvals = step(p, 0, q, b, regular)
            h = [
                cap(
                    bw._ratio(value.V, value.B**2, regular * value.R, value.rate, "msetwo"),
                    store.ranges[i],
                )
                for i, value in enumerate(hvals)
            ]
            results["msetwo"] = (tuple(h), tuple(b))
        if base == "msecomb1":
            h = min(results["mserd"][0][0], results["msesum"][0][0])
            b = min(results["mserd"][1][0], results["msesum"][1][0])
            results[base] = ((h, h), (b, b))
        elif base == "msecomb2":
            results[base] = tuple(
                tuple(
                    sorted([results[key][which][side] for key in ["mserd", "msesum", "msetwo"]])[1]
                    for side in [0, 1]
                )
                for which in [0, 1]
            )
        h, b = results[base]
        if method.startswith("cer"):
            n = self.sample.nrows
            if self.clusters:
                # Exact group counts on each entire side, including nonlocal rows.
                with ExitStack() as stack:
                    n = 0
                    for side in [0, 1]:
                        acc = ClusterAccumulator(1, scratch_directory=_scratch_directory())
                        stack.callback(acc.close)
                        for dx, _, keys in self.blocks(side, None):
                            acc.add(keys, torch.ones((len(dx), 1), dtype=torch.float64))
                        n += acc.finish()[1]
            factor = n ** (-(p / ((3 + p) * (3 + 2 * p))))
            h = tuple(value * factor for value in h)
        return bw.Bandwidths(h, b, pilot, method, restricted), fuzzy


def fit_streaming(spec, source, *, batch_rows=None):
    if not supports(spec):
        raise AnalysisError(
            "streaming_unsupported", "No replay local-polynomial RD adapter is registered."
        )
    try:
        with torch.no_grad(), torch.device("cpu"), ExitStack() as stack:
            sample = ReplaySample(spec, source, batch_rows=batch_rows)
            sample.prepare()
            if sample.nrows > 2**53:
                raise AnalysisError(
                    "precision_unsupported",
                    "RD unit counts must be exactly representable in float64.",
                )
            notes = _Notes(spec)
            if spec.columns.get("covariates") or spec.weights is not None or notes.option("deriv") != 0:
                raise AnalysisError("streaming_unsupported", "Covariates, observation weights and derivative RD currently require a resident DataFrame; they are not silently ignored by Dataset replay.")
            p = int(notes.option("p"))
            q = p + 1 if notes.option("q") is None else int(notes.option("q"))
            if q <= p:
                raise AnalysisError("invalid_spec", "Bias-correction polynomial q must exceed p.")
            if spec.predictors:
                raise AnalysisError(
                    "not_implemented",
                    "Covariate-adjusted RD is not implemented by the native model.",
                )
            context = _Context(sample, notes)
            resource = sample.plan_rows(
                "native full-source RD local moments and NN halo",
                {
                    "global_polynomial_TSQR": 1024 * (q + 4 + context.width) ** 2,
                    "group_SQLite_and_cache": 8 * 1024**2,
                    "cluster_accumulator_caches": 12 * 1024**2 if context.clusters else 0,
                    "NN_halo_and_decode": 512 * context.matches * (context.width + 4),
                },
                256 * (q + context.width + 4),
            )
            store = _Store(sample, context.width)
            stack.callback(store.close)
            context.store = store
            store.seed(context)
            h, b = _pair(notes.option("h"), "h"), _pair(notes.option("b"), "b")
            rho = notes.option("rho")
            selection = None
            if h is None:
                if b is not None or rho is not None:
                    raise AnalysisError(
                        "invalid_spec", "b and rho require an explicit main bandwidth h."
                    )
                selection, fuzzy_bw = context.select(p, q)
                h, b = selection.h, selection.b
            elif b is None:
                factor = 1.0 if rho is None else float(rho)
                b = (h[0] / factor, h[1] / factor)
            elif rho is not None:
                raise AnalysisError("invalid_spec", "Give b or rho, not both.")
            fits = [context.fit(side, h[side], p) for side in [0, 1]]
            biases = [context.fit(side, b[side], q) for side in [0, 1]]
            jump = context.raw_coefficient(fits[1], 0) - context.raw_coefficient(fits[0], 0)
            corrected = context.bias_corrected(1, fits[1], biases[1]) - context.bias_corrected(
                0, fits[0], biases[0]
            )
            covariances = [context.covariance(side, fits[side], biases[side]) for side in [0, 1]]
            cl = sum(value[0] for value in covariances)
            rb = sum(value[1] for value in covariances)
            extra = {}
            if context.fuzzy:
                yjump, tjump = jump.tolist()
                if abs(tjump) < 1e-12:
                    raise AnalysisError(
                        "weak_first_stage", "The treatment probability does not jump at the cutoff."
                    )
                combine = torch.tensor([1 / tjump, -yjump / tjump**2], dtype=torch.float64)
                tau = yjump / tjump
                corrected_tau = tau - float(combine @ (jump - corrected))
                extra.update(
                    {
                        "first_stage": {
                            "conventional": tjump,
                            "bias_corrected": float(corrected[1]),
                            "std_error": math.sqrt(float(cl[1, 1])),
                            "robust_std_error": math.sqrt(float(rb[1, 1])),
                        },
                        "reduced_form": {
                            "conventional": yjump,
                            "bias_corrected": float(corrected[0]),
                        },
                    }
                )
            else:
                combine = torch.ones(1, dtype=torch.float64)
                tau, corrected_tau = float(jump[0]), float(corrected[0])
            variance, robust = float(combine @ cl @ combine), float(combine @ rb @ combine)
            if not variance > 0 or not robust > 0:
                raise AnalysisError(
                    "invalid_covariance", "The local-polynomial variance is not positive."
                )
            beta = torch.tensor([tau, corrected_tau, corrected_tau], dtype=torch.float64)
            covariance = torch.diag(torch.tensor([variance, variance, robust], dtype=torch.float64))
            metrics = {
                "h_left": h[0],
                "h_right": h[1],
                "b_left": b[0],
                "b_right": b[1],
                "n_left": store.counts[0],
                "n_right": store.counts[1],
                "n_h_left": fits[0].count,
                "n_h_right": fits[1].count,
                "n_b_left": biases[0].count,
                "n_b_right": biases[1].count,
                "p": p,
                "q": q,
                "cutoff": context.cutoff,
            }
            extra.update(
                {
                    "cutoff_evaluation": {"version":1,"point":"cutoff intercept, one-sided limit",
                        "derivative":0,
                        "sides": {name:{"conventional_variance":float(covariances[side][0][0,0]),
                                         "robust_variance":float(covariances[side][1][0,0])}
                                  for side,name in enumerate(["left","right"])}},
                    "kernel": context.kernel,
                    "vce": context.vce,
                    "nnmatch": context.matches if context.vce == "nn" else None,
                    "bandwidth_selection": None
                    if selection is None
                    else {
                        "method": selection.method,
                        "pilot": selection.pilot,
                        "bwrestrict": selection.restricted,
                    },
                    "rows": ["Conventional", "Bias-corrected", "Robust"],
                    "design": "fuzzy" if context.fuzzy else "sharp",
                    "confidence_level": 1 - spec.alpha,
                    "side_estimates": {
                        name: {
                            "conventional": float(context.raw_coefficient(fits[side], 0)[0]),
                            "bias_corrected": float(
                                context.bias_corrected(side, fits[side], biases[side])[0]
                            ),
                        }
                        for side, name in enumerate(["left", "right"])
                    },
                }
            )
            if selection is not None and context.fuzzy:
                extra["bandwidth_equation"] = "fuzzy" if fuzzy_bw else "outcome (sharp)"
            info = {
                "df_inference": None,
                "df_resid": None,
                "small_sample_correction": None,
                "variance_method": "rdrobust local-polynomial sandwich",
                "correction": context.vce
                + " residual local-polynomial sandwich; complete local sample and robust bias correction",
            }
            return _result(
                sample,
                terms=extra["rows"],
                beta=beta,
                covariance=covariance,
                info=info,
                metrics=metrics,
                notes=notes,
                predictions=[],
                tests={},
                solver="native_replayed_local_polynomial_TSQR",
                diagnostics=store.diagnostics(),
                extra=extra,
                resource=resource.record(),
                title="Regression discontinuity: " + extra["design"] + " local polynomial",
                use_t=False,
                provenance_extra={
                    "covariance_matrix": "diagonal: alternative estimates of one parameter; cross-row covariances are not reported",
                    "running_variable": context.running,
                },
            )
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except (OverflowError, ZeroDivisionError) as error:
        raise AnalysisError(
            "precision_unsupported",
            "RD polynomial or bandwidth operations exceed the float64 precision domain; rescale the running variable or provide stable explicit bandwidths.",
        ) from error
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError(
            "rd_spill_failed",
            "RD replay needs writable owned temporary storage and enough free disk space.",
        ) from error
