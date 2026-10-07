"""Full-source dynamic-panel GMM through bounded complete-panel native blocks.

Global instrument coordinates, TSQR factors and replayed per-panel moments
replace N-by-L arrays. Work on one complete panel is guarded before reading
it; no source truncation, estimator delegation or sampled diagnostics.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import math
import sqlite3
from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import normal_sf
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.resources import plan_workspace
from openecon.streaming_design import encode_cluster_labels

from .core import wald_test
from .dpanel import kernels, model, structure
from .dpanel.instruments import Instruments
from .replay_sample import ReplaySample
from .streaming_dpanel import _PanelStore, _periods
from .streaming_linear import _Notes, _finite, _result

SUPPORTED = frozenset({"xtdpd"})


def supports(spec):
    return spec.estimator in SUPPORTED


class _Store(_PanelStore):
    def seed(self):
        sample = self.sample
        self.date = False
        for batch in sample.batches():
            keys = encode_cluster_labels(batch.frame[sample.spec.panel])
            time = batch.frame[sample.spec.time]
            current_date = pd.api.types.is_datetime64_any_dtype(time.dtype)
            if (
                self.db.execute("SELECT 1 FROM raw LIMIT 1").fetchone()
                and current_date != self.date
            ):
                raise AnalysisError(
                    "source_changed", "Dynamic-panel time types changed between input blocks."
                )
            self.date = current_date
            times = time.dt.as_unit("ns").astype("int64").tolist() if self.date else _periods(time)
            values = torch.stack([batch.numeric(name) for name in self.names], 1)
            try:
                self.db.executemany(
                    "INSERT INTO raw VALUES(" + ",".join("?" for _ in range(3 + self.width)) + ")",
                    (
                        (int(position), key, int(t), *value)
                        for position, key, t, value in zip(
                            batch.positions.tolist(), keys, times, values.tolist(), strict=True
                        )
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise AnalysisError(
                    "repeated_time_values", "Time values are repeated within a dynamic panel."
                ) from error
        if self.date:
            self.db.execute("DROP INDEX ordered_panel")
            self.db.execute(
                "CREATE TABLE periods AS SELECT t,ROW_NUMBER() OVER(ORDER BY t)-1 AS idx FROM(SELECT DISTINCT t FROM raw)"
            )
            self.db.execute("CREATE UNIQUE INDEX period_identity ON periods(t)")
            self.db.execute("UPDATE raw SET t=(SELECT idx FROM periods WHERE periods.t=raw.t)")
            self.db.execute("CREATE UNIQUE INDEX ordered_panel ON raw(key,t)")
        self.minimum_time, self.maximum_time = self.db.execute(
            "SELECT MIN(t),MAX(t) FROM raw"
        ).fetchone()
        self.span = self.maximum_time - self.minimum_time + 1
        if self.span > structure.MAX_GRID_CELLS:
            raise AnalysisError(
                "time_span_too_large",
                "A dynamic-panel native local grid exceeds the guarded time span; use explicit regular period codes or a shorter lag span.",
            )
        self.db.execute("CREATE TABLE level_periods(period INTEGER PRIMARY KEY,n INTEGER)")
        self.db.execute("CREATE TABLE transformed_periods(period INTEGER PRIMARY KEY)")
        self.db.execute(
            "CREATE TABLE groups(key BLOB PRIMARY KEY,raw_n INTEGER,nt INTEGER,nl INTEGER,n INTEGER) WITHOUT ROWID"
        )
        self.db.execute("INSERT INTO groups(key,raw_n) SELECT key,COUNT(*) FROM raw GROUP BY key")
        self.db.commit()

    def panels(self):
        cursor = self.db.execute("SELECT key,raw_n FROM groups ORDER BY key")
        try:
            for key, n in cursor:
                numerical_width = getattr(self, "numerical_width", self.width + 8)
                per_row = 192 * (self.width + numerical_width + 4)
                reserved = dict(self.sample.adapter_plan.buffers)
                reserved.pop("native_row_block", None)
                panel_plan = plan_workspace(
                    "one complete native dynamic-panel block",
                    {
                        **reserved,
                        "complete_panel_values_and_transformation": n * per_row,
                        "native_panel_time_grid": 96 * self.span * (numerical_width + 4),
                    },
                    budget_bytes=self.sample.working_bytes,
                )
                previous = getattr(self, "peak_panel_plan", None)
                if previous is None or panel_plan.estimated_bytes > previous.estimated_bytes:
                    self.peak_panel_plan = panel_plan
                rows = self.db.execute(
                    "SELECT position,t,"
                    + ",".join(f"v{i}" for i in range(self.width))
                    + " FROM raw WHERE key=? ORDER BY t",
                    (key,),
                ).fetchall()
                values = torch.tensor([row[2:] for row in rows], dtype=torch.float64)
                period = torch.tensor(
                    [row[1] - self.minimum_time for row in rows], dtype=torch.int64
                )
                self.sample.actual_numeric_peak_rows = max(self.sample.actual_numeric_peak_rows, n)
                yield key, values, period, [row[0] for row in rows]
        finally:
            cursor.close()

    def geometry(self, period, opts):
        codes = torch.zeros(len(period), dtype=torch.int64)
        grid = structure.layout(codes, period, 1)
        if grid.span < self.span:
            grid.lookup = torch.cat(
                (grid.lookup, torch.full((self.span - grid.span,), -1, dtype=torch.int64))
            )
            grid.span = self.span
        obs = structure.levels(grid, opts.lags)
        transform = (
            structure.OrthogonalDeviations(obs)
            if opts.orthogonal
            else structure.Differences(grid, obs)
        )
        return grid, obs, transform

    def discover(self, opts):
        moments = _CompensatedSum(
            (1 + int(opts.constant) + opts.lags + len(self.sample.spec.predictors),)
        )
        count = 0
        for key, values, period, _ in self.panels():
            grid, obs, transform = self.geometry(period, opts)
            n = obs.n if opts.system else transform.n
            self.db.execute(
                "UPDATE groups SET nt=?,nl=?,n=? WHERE key=?", (transform.n, obs.n, n, key)
            )
            if obs.n:
                unique, counts = torch.unique(obs.period, return_counts=True)
                self.db.executemany(
                    "INSERT INTO level_periods VALUES(?,?) ON CONFLICT(period) DO UPDATE SET n=n+excluded.n",
                    ((int(t), int(n)) for t, n in zip(unique, counts, strict=True)),
                )
                self.db.executemany(
                    "INSERT OR IGNORE INTO transformed_periods VALUES(?)",
                    ((int(t),) for t in torch.unique(transform.period)),
                )
                if opts.constant:
                    levels = self.regressors(values, grid, obs, opts, [])
                    moments.add(torch.cat((levels, values[obs.rows, 0:1]), 1).sum(0))
                    count += obs.n
        self.db.commit()
        self.nt, self.nl, self.n, self.groups, self.smallest, self.largest = self.db.execute(
            "SELECT SUM(nt),SUM(nl),SUM(n),SUM(n>0),MIN(CASE WHEN n>0 THEN n END),MAX(n) FROM groups"
        ).fetchone()
        if not self.nt:
            raise AnalysisError(
                "insufficient_observations",
                "No complete transformed dynamic-panel equations remain after the declared lag windows.",
            )
        self.shift = None
        self.y_shift = 0.0
        if opts.constant:
            means = moments.value / count
            self.shift = means[:-1].clone()
            self.shift[0] = 0.0
            self.y_shift = float(means[-1])

    def regressors(self, values, grid, obs, opts, periods):
        parts = []
        if opts.constant:
            parts.append(torch.ones((obs.n, 1), dtype=torch.float64))
        for lag in range(1, opts.lags + 1):
            parts.append(values[grid.rows_at(obs.codes, obs.period - lag), :1])
        for name in self.sample.spec.predictors:
            parts.append(values[obs.rows, self.names.index(name) : self.names.index(name) + 1])
        for period in periods:
            parts.append((obs.period == period).to(torch.float64)[:, None])
        return torch.cat(parts, 1)


def _metadata(store, opts):
    # Coordinates are bounded before allocating Python labels or native arrays.
    periods_needed = opts.time_dummies or any(not group.collapse for group in opts.gmm)

    def period_list(table):
        if periods_needed:
            rows = store.db.execute(
                f"SELECT period FROM {table} ORDER BY period LIMIT 2001"
            ).fetchall()
            if len(rows) > 2000:
                raise AnalysisError(
                    "instrument_count_too_large",
                    "Uncollapsed/time-dummy period metadata exceeds the native width guard.",
                )
            return [row[0] for row in rows]
        return [store.db.execute(f"SELECT MAX(period) FROM {table}").fetchone()[0]]

    transformed, level = period_list("transformed_periods"), period_list("level_periods")
    columns, labels = [], {}

    def add(values):
        for value in values:
            if len(columns) >= 2000:
                raise AnalysisError(
                    "instrument_count_too_large",
                    "The complete model exceeds2000 planned instruments; collapse or cap GMM lags.",
                )
            columns.append(value)

    for number, group in enumerate(opts.gmm, 1):
        key = f"gmm{number}"
        labels[key] = group.label()
        if group.equation in ("diff", "both"):
            high = max(transformed) if group.hi is None else min(group.hi, max(transformed))
            for name in group.columns:
                if group.collapse:
                    add((key, "diff", name, lag, None) for lag in range(group.lo, high + 1))
                else:
                    add(
                        (key, "diff", name, lag, t)
                        for t in transformed
                        for lag in range(group.lo, high + 1)
                        if t - lag >= 0
                    )
        if opts.system and group.equation in ("level", "both"):
            high = max(level) - 1 if group.hi is None else min(group.hi, max(level) - 1)
            lags = (
                range(max(0, group.lo - 1), max(0, group.lo - 1) + 1)
                if group.equation == "both"
                else range(group.lo, high + 1)
            )
            for name in group.columns:
                if group.collapse:
                    add((key, "level", name, lag, None) for lag in lags)
                else:
                    add(
                        (key, "level", name, lag, t)
                        for t in level
                        for lag in lags
                        if t - lag - 1 >= 0
                    )
    for number, group in enumerate(opts.iv, 1):
        key = f"iv{number}"
        labels[key] = group.label()
        add((key, group.equation, name, group.passthru, None) for name in group.columns)
    dummies = level[1:] if opts.time_dummies else []
    if len(dummies) + opts.lags + len(store.sample.spec.predictors) + int(opts.constant) > 384:
        raise AnalysisError("model_too_wide", "Global dynamic-panel terms exceed384 columns.")

    def label(t):
        if not store.date:
            return str(store.minimum_time + t)
        nanoseconds = store.db.execute(
            "SELECT t FROM periods WHERE idx=?", (store.minimum_time + t,)
        ).fetchone()[0]
        value = pd.Timestamp(nanoseconds, unit="ns")
        dtype = store.sample.sample[store.sample.spec.time].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            value = value.tz_localize("UTC").tz_convert(dtype.tz)
        return str(value)

    terms = [
        *(["Intercept"] if opts.constant else []),
        *(f"L{lag}.{store.sample.spec.outcome}" for lag in range(1, opts.lags + 1)),
        *store.sample.spec.predictors,
        *(f"{store.sample.spec.time}[{label(t)}]" for t in dummies),
    ]
    if dummies:
        labels["time"] = "time dummies"
        add(("time", "level" if opts.system else "diff", t, None, None) for t in dummies)
    if opts.constant:
        labels["constant"] = "_cons"
        add([("constant", "level", None, None, None)])
    if not columns:
        raise AnalysisError(
            "underidentified", "The complete dynamic-panel model has no instruments."
        )
    store.dummy_means = torch.tensor(
        [
            store.db.execute("SELECT n FROM level_periods WHERE period=?", (t,)).fetchone()[0]
            / store.nl
            for t in dummies
        ],
        dtype=torch.float64,
    )
    return columns, labels, dummies, terms


def _panel(store, opts, metadata, values, period, *, keep_x=None, keep_z=None):
    columns, labels, dummies, terms = metadata
    grid, obs, transform = store.geometry(period, opts)
    if not obs.n:
        return None
    raw_levels = store.regressors(values, grid, obs, opts, dummies)
    x_t, y_t = transform.apply(raw_levels), transform.apply(values[obs.rows, 0])
    x_l, y_l = raw_levels, values[obs.rows, 0]
    if opts.constant:
        shift = torch.cat((store.shift, torch.zeros(len(dummies), dtype=torch.float64)))
        if dummies:
            # The dense native model centers these dummy columns at their global
            # level-equation means, just like the other nonconstant regressors.
            shift[-len(dummies) :] = store.dummy_means
        x_l, y_l = x_l - shift, y_l - store.y_shift
    else:
        shift = None
    x = torch.cat((x_t, x_l), 0) if opts.system else x_t
    y = torch.cat((y_t, y_l), 0) if opts.system else y_t
    z = torch.zeros((len(y), len(columns)), dtype=torch.float64)
    for index, (key, equation, name, lag, t) in enumerate(columns):
        if key.startswith("gmm"):
            cells = transform.period if equation == "diff" else obs.period
            codes = transform.codes if equation == "diff" else obs.codes
            found, present = grid.values_at(values[:, store.names.index(name)], codes, cells - lag)
            if equation == "level":
                before, earlier = grid.values_at(
                    values[:, store.names.index(name)], codes, cells - lag - 1
                )
                found, present = found - before, present & earlier
            if t is not None:
                present &= cells == t
            z[: transform.n, index] = (
                torch.where(present, found, torch.zeros_like(found))
                if equation == "diff"
                else z[: transform.n, index]
            )
            if equation == "level" and opts.system:
                z[transform.n :, index] = torch.where(present, found, torch.zeros_like(found))
        elif key.startswith("iv"):
            v = values[:, store.names.index(name)]
            if equation in ("diff", "both"):
                z[: transform.n, index] = (
                    grid.values_at(v, transform.codes, transform.period)[0]
                    if lag
                    else transform.apply(v[obs.rows])
                )
            if opts.system and equation in ("level", "both"):
                z[transform.n :, index] = v[obs.rows]
        else:
            v = (
                torch.ones(obs.n, dtype=torch.float64)
                if key == "constant"
                else (obs.period == name).to(torch.float64)
            )
            if equation == "diff":
                z[: transform.n, index] = transform.apply(v)
            elif opts.system:
                z[transform.n :, index] = v
    if keep_x is not None:
        x, x_l = x[:, keep_x], x_l[:, keep_x]
        terms = [terms[i] for i in keep_x]
        shift = None if shift is None else shift[keep_x]
    if keep_z is not None:
        z = z[:, keep_z]
        columns = [columns[i] for i in keep_z]
    inst = Instruments(
        z, [c[0] for c in columns], [c[1] for c in columns], labels, len(metadata[0]), 0
    )
    return model.Model(
        SimpleNamespace(spec=store.sample.spec),
        grid,
        obs,
        transform,
        x_l,
        y_l,
        terms,
        x,
        y,
        torch.zeros(len(y), dtype=torch.int64),
        inst,
        transform.n,
        opts.system,
        constant=terms.index("Intercept") if "Intercept" in terms else None,
        shift=shift,
        y_shift=store.y_shift,
    )


def _panels(store, opts, metadata, keep_x=None, keep_z=None):
    for key, values, period, positions in store.panels():
        current = _panel(store, opts, metadata, values, period, keep_x=keep_x, keep_z=keep_z)
        if current is not None:
            yield key, current, positions


def _screen(store, opts, metadata, notes):
    x_tree, z_tree = _TSQRTree(), _TSQRTree()
    for _, current, _ in _panels(store, opts, metadata):
        if len(current.y):
            x_tree.add(torch.linalg.qr(current.x, mode="r")[1])
            z_tree.add(torch.linalg.qr(current.instruments.z, mode="r")[1])
    rx, rz = x_tree.finish(), z_tree.finish()
    keep_x, omitted_x = collinear_columns(rx)
    for index in omitted_x:
        term = metadata[3][index]
        notes.warn("Omitted " + term + " because of global transformed-design collinearity.")
        store.sample.notes["omitted_terms"].append(term)
    if not keep_x:
        raise AnalysisError(
            "invalid_spec",
            "Every regressor is globally collinear after the dynamic-panel transformation.",
        )
    nonzero = (rz != 0).any(dim=0).nonzero().flatten().tolist()
    kept, omitted = kernels.instrument_rank(rz[:, nonzero])
    keep_z = [nonzero[i] for i in kept]
    if len(keep_z) < len(keep_x):
        raise AnalysisError(
            "underidentified",
            "The complete instrument matrix has fewer independent columns than coefficients.",
        )
    return keep_x, keep_z, len(metadata[0]) - len(nonzero), len(omitted)


def _global(store, opts, metadata, keep_x, keep_z):
    k, n_inst = len(keep_x), len(keep_z)
    a, b = _CompensatedSum((n_inst, k)), _CompensatedSum((n_inst,))
    y_t, y_l = _CompensatedSum(()), _CompensatedSum(())
    positive = _TSQRTree()
    for _, current, _ in _panels(store, opts, metadata, keep_x, keep_z):
        z = current.instruments.z
        a.add(z.T @ current.x)
        b.add(z.T @ current.y)
        y_t.add(current.y[: current.n_t] @ current.y[: current.n_t])
        y_l.add(current.y_levels @ current.y_levels)
        for block in model.one_step_factor(current, opts.h):
            if len(block):
                positive.add(torch.linalg.qr(block, mode="r")[1])
    return a.value, b.value, positive.finish(), float(y_t.value), float(y_l.value)


def _residual_moments(store, opts, metadata, keep_x, keep_z, beta):
    tree = _TSQRTree()
    ssr_t, ssr_l = _CompensatedSum(()), _CompensatedSum(())
    for _, current, _ in _panels(store, opts, metadata, keep_x, keep_z):
        residual = current.y - current.x @ beta
        g = current.instruments.z.T @ residual
        tree.add(g[None, :])
        ssr_t.add(residual[: current.n_t] @ residual[: current.n_t])
        if opts.system:
            ssr_l.add(residual[current.n_t :] @ residual[current.n_t :])
    return tree.finish(), float(ssr_t.value), float(ssr_l.value)


def _windmeijer(store, opts, metadata, keep_x, keep_z, one, two, a, b, v1):
    root = two.root
    q = root.T @ (root @ (b - a @ two.beta))
    m = _CompensatedSum(a.shape)
    for _, current, _ in _panels(store, opts, metadata, keep_x, keep_z):
        z, x = current.instruments.z, current.x
        gi = z.T @ (current.y - x @ one.beta)
        ai = z.T @ x
        m.add(ai * (gi @ q) + gi[:, None] * (ai.T @ q)[None, :])
    d = two.bread @ (root @ a).T @ (root @ m.value)
    covariance = two.bread + d @ two.bread + two.bread @ d.T + d @ v1 @ d.T
    return (covariance + covariance.T) / 2


def _ar_tests(store, opts, metadata, keep_x, keep_z, beta, covariance, p, sigma2):
    k, n_inst = len(keep_x), len(keep_z)
    totals = [
        {
            "xw": _CompensatedSum((k,)),
            "moment": _CompensatedSum((n_inst,)),
            "numerator": _CompensatedSum(()),
            "first": _CompensatedSum(()),
            "pairs": 0,
        }
        for _ in range(opts.artests)
    ]
    for _, current, _ in _panels(store, opts, metadata, keep_x, keep_z):
        residual = current.y - current.x @ beta
        inputs = model._ar_inputs(current, opts, beta, covariance, p, residual, sigma2)
        e, period = inputs.resid_diff, inputs.period_diff
        lookup = torch.full((store.span,), -1, dtype=torch.int64)
        lookup[period] = torch.arange(len(e), dtype=torch.int64)
        for order, total in enumerate(totals, 1):
            previous = period - order
            partner = torch.where(
                previous >= 0, lookup[previous.clamp(min=0)], torch.full_like(period, -1)
            )
            present = partner >= 0
            total["pairs"] += int(present.sum())
            w = (
                torch.where(present, e[partner.clamp(min=0)], torch.zeros_like(e))
                if len(e)
                else torch.empty(0, dtype=torch.float64)
            )
            ai = w @ e
            xw = inputs.x_diff.T @ w
            total["xw"].add(xw)
            total["numerator"].add(ai)
            if inputs.moments is not None:
                total["first"].add(ai.square())
                total["moment"].add(inputs.moments[0] * ai)
            elif inputs.align is not None:
                total["first"].add(sigma2 * (w @ w))
                dated = (
                    torch.where(
                        inputs.align >= 0,
                        w[inputs.align.clamp(min=0)],
                        torch.zeros(len(inputs.align), dtype=torch.float64),
                    )
                    if len(w)
                    else torch.zeros(len(inputs.align), dtype=torch.float64)
                )
                total["moment"].add(sigma2 * (inputs.cross.T @ dated))
            else:
                dw = inputs.adjoint.adjoint(w)
                total["first"].add(sigma2 * (dw @ dw))
                total["moment"].add(sigma2 * (inputs.cross.T @ dw))
    tests = {}
    for order, total in enumerate(totals, 1):
        label = f"Arellano-Bond test for AR({order}) in first differences"
        common = {"statistic": None, "p_value": None, "distribution": "normal", "label": label}
        if not total["pairs"]:
            tests[f"ar{order}"] = {
                **common,
                "note": f"no panel has differenced residuals {order} periods apart",
            }
            continue
        xw = total["xw"].value
        first = float(total["first"].value)
        middle = float(xw @ (p @ total["moment"].value))
        last = float(xw @ covariance @ xw)
        variance = first - 2 * middle + last
        if not math.isfinite(variance) or variance <= 1e-14 * max(
            abs(first), abs(last), torch.finfo(torch.float64).eps
        ):
            tests[f"ar{order}"] = {
                **common,
                "note": "the variance of the test statistic is not positive",
                "pairs": total["pairs"],
            }
        else:
            statistic = float(total["numerator"].value) / math.sqrt(variance)
            tests[f"ar{order}"] = {
                **common,
                "statistic": statistic,
                "p_value": 2 * normal_sf(abs(statistic)),
                "pairs": total["pairs"],
            }
    return tests


def _fit(spec, source, *, batch_rows):
    opts = model.read_options(spec)
    notes = _Notes(spec)
    private = spec.model_copy(
        update={"columns": {**spec.columns, "replay_instrument_inputs": opts.columns}}
    )
    sample = ReplaySample(private, source, batch_rows=batch_rows).prepare()
    sample.spec = spec
    sample.plan_rows(
        "dynamic-panel ordered discovery",
        {"panel_SQLite_cache": 4 * 1024**2},
        256 * (len(sample.columns) + 8),
    )
    names = list(dict.fromkeys([spec.outcome, *spec.predictors, *opts.columns]))
    with ExitStack() as stack:
        store = _Store(sample, names)
        stack.callback(store.close)
        store.seed()
        store.discover(opts)
        metadata = _metadata(store, opts)
        width = len(metadata[0]) + len(metadata[3]) + 2
        store.numerical_width = width
        resource = sample.plan_rows(
            "dynamic-panel global GMM and full diagnostics",
            {
                "panel_SQLite_cache": 4 * 1024**2,
                "global_GMM_factors_and_diagnostics": 1024 * width**2,
                "instrument_coordinate_metadata": 512 * width,
            },
            256 * (width + 4),
        )
        keep_x, keep_z, zero, omitted = _screen(store, opts, metadata, notes)
        k, n_inst = len(keep_x), len(keep_z)
        a, b, positive, yt, yl = _global(store, opts, metadata, keep_x, keep_z)
        root1, rank1 = kernels.weight_root([positive])
        one = kernels.gmm_step(a, b, root1, rank1)
        moments, ssr_t, ssr_l = _residual_moments(store, opts, metadata, keep_x, keep_z, one.beta)
        if ssr_t <= 1e-28 * max(yt, yl):
            raise AnalysisError(
                "perfect_fit",
                "The complete dynamic-panel transformed outcome is explained exactly or has no variation.",
            )
        level_errors = opts.system and opts.h == 1
        c = 2.0 if not opts.orthogonal and opts.h != 1 else 1.0
        wttot = store.nt if opts.system and opts.h > 1 else store.n
        sigma2 = (ssr_l if level_errors else ssr_t) / (c * wttot)
        if not math.isfinite(sigma2) or not sigma2 > 0:
            raise AnalysisError(
                "precision_unsupported",
                "The complete residual variance is not positive finite float64.",
            )
        n_effective = min(n_inst, rank1)
        p1 = kernels.projector(one, a)
        v1 = kernels.sandwich(p1, moments)
        two = None
        hansen_note = None
        onestep_nonrobust = not opts.twostep and not opts.robust
        if not onestep_nonrobust:
            if store.groups < 2:
                raise AnalysisError(
                    "insufficient_clusters",
                    "Robust covariance and two-step moment weighting need at least two contributing panels.",
                )
            root2, rank2 = kernels.weight_root([moments])
            if rank2 < n_inst:
                notes.warn(
                    f"The two-step weight matrix is singular (rank {rank2} of {n_inst}); a generalized inverse is used. Reduce instruments or cap lags."
                )
            if rank2 < k:
                hansen_note = f"The two-step weight matrix has rank {rank2} but {k} coefficients are estimated, so the two-step Hansen statistic is not available."
                if opts.twostep:
                    raise AnalysisError("underidentified", hansen_note)
                notes.warn(hansen_note)
            else:
                two = kernels.gmm_step(a, b, root2, rank2)
        if rank1 < n_inst:
            notes.warn(
                f"The one-step weight matrix Z'HZ is singular (rank {rank1} of {n_inst}); a generalized inverse is used."
            )
        if n_effective > store.groups:
            notes.warn(
                "The number of instruments exceeds the number of panels; Hansen tests may be weakened and two-step estimates biased. Collapse or cap GMM lags."
            )
        info = {"steps": 2 if opts.twostep else 1, "df_inference": None, "df_resid": None}
        if opts.twostep:
            final = two
            p_final = kernels.projector(two, a)
            covariance = (
                _windmeijer(store, opts, metadata, keep_x, keep_z, one, two, a, b, v1)
                if opts.robust
                else two.bread
            )
            info["correction"] = (
                "two-step robust: Windmeijer (2005) finite-sample correction"
                if opts.robust
                else "two-step: (X'Z W2 Z'X)^-1"
            )
            _, final_t, final_l = _residual_moments(
                store, opts, metadata, keep_x, keep_z, final.beta
            )
            sigma_final = (final_l if level_errors else final_t) / (c * wttot)
        else:
            final, p_final = one, p1
            covariance = v1 if opts.robust else sigma2 * one.bread
            info["correction"] = (
                "one-step robust sandwich clustered on the panel (no factor)"
                if opts.robust
                else f"one-step: sigma2 (X'Z W1 Z'X)^-1, sigma2=e'e/({c:g} N) from {'level' if level_errors else 'transformed'}-equation residuals (xtabond2)"
            )
            sigma_final = sigma2
        covariance_ar, sigma_ar = covariance, sigma2
        df = None
        if opts.small:
            if store.n <= k or wttot <= k:
                raise AnalysisError(
                    "insufficient_observations",
                    "Small-sample covariance needs more full equations than coefficients.",
                )
            tmp = wttot / (wttot - k)
            if onestep_nonrobust:
                factor, df = tmp, store.n - k
                info["correction"] += f"; small: N/(N-k) with N={wttot}"
            else:
                df = store.groups - int(opts.constant and 0 in keep_x)
                if store.groups < 2 or df < 1:
                    raise AnalysisError(
                        "insufficient_observations",
                        "Small-sample covariance needs at least two panels.",
                    )
                factor = store.groups / (store.groups - 1) * (store.n - 1) / (store.n - k)
                info["correction"] += "; small: G/(G-1) (N-1)/(N-k)"
            covariance = covariance * factor
            sigma_ar *= tmp
            sigma_final *= tmp
            info["small_sample_correction"] = factor
        info.update({"df_inference": df, "df_resid": df})
        if opts.robust:
            info.update(
                {
                    "cluster_count": store.groups,
                    "cluster_column": spec.panel,
                    "cluster_df": store.groups - 1 if df is None else df,
                }
            )
        tests = _ar_tests(
            store, opts, metadata, keep_x, keep_z, final.beta, covariance_ar, p_final, sigma_ar
        )
        columns = [metadata[0][i] for i in keep_z]
        inst = Instruments(
            torch.empty((0, n_inst), dtype=torch.float64),
            [c[0] for c in columns],
            [c[1] for c in columns],
            metadata[1],
            len(metadata[0]),
            zero,
        )
        template = SimpleNamespace(instruments=inst, system=opts.system)
        over = n_effective - k
        sargan = one.criterion / sigma2
        tests["sargan"] = model._overid(
            sargan, over, "Sargan test of overidentifying restrictions (not robust)"
        )
        if two is not None:
            tests["hansen"] = model._overid(
                two.criterion, over, "Hansen test of overidentifying restrictions (robust)"
            )
            if over > 0:
                tests.update(
                    model._difference_tests(
                        template, a, b, [moments], two.criterion, k, "hansen", 1.0
                    )
                )
        elif hansen_note is not None:
            tests["hansen"] = {
                "statistic": None,
                "df": over,
                "p_value": None,
                "distribution": "chi2",
                "label": "Hansen test of overidentifying restrictions (robust)",
                "note": hansen_note,
            }
        elif over > 0:
            tests.update(
                model._difference_tests(template, a, b, [positive], sargan, k, "sargan", sigma2)
            )
        beta = final.beta.clone()
        terms = [metadata[3][i] for i in keep_x]
        constant = terms.index("Intercept") if "Intercept" in terms else None
        if constant is not None:
            shift = torch.cat((store.shift, store.dummy_means))[keep_x]
            jac = torch.eye(k, dtype=torch.float64)
            jac[constant] -= shift
            beta = jac @ beta
            beta[constant] += store.y_shift
            covariance = jac @ covariance @ jac.T
        _finite(beta, covariance)
        slopes = [i for i in range(k) if i != constant]
        tests["model"] = wald_test(
            beta,
            covariance,
            slopes,
            df_resid=df,
            label="Wald test of all coefficients except the constant",
        )
        predictions = []
        store.db.execute("CREATE TABLE selected(position INTEGER PRIMARY KEY)")
        for _, current, positions in _panels(store, opts, metadata, keep_x, keep_z):
            rows = current.obs.rows if opts.system else current.obs.rows[current.transform.source]
            store.db.executemany(
                "INSERT INTO selected VALUES(?)", ((positions[int(i)],) for i in rows)
            )
            observed = current.y_levels + store.y_shift if opts.system else current.y
            fitted = (
                current.x_levels @ final.beta + store.y_shift
                if opts.system
                else current.x @ final.beta
            )
            take = min(400 - len(predictions), len(observed))
            predictions.extend(
                {"observed": float(y), "predicted": float(mu), "residual": float(y - mu)}
                for y, mu in zip(observed[:take], fitted[:take], strict=True)
            )
        store.db.commit()
        digest = hashlib.sha256()
        cursor = store.db.execute("SELECT position FROM selected ORDER BY position")
        while rows := cursor.fetchmany(sample.rows):
            digest.update(
                torch.tensor([row[0] for row in rows], dtype=torch.int64)
                .contiguous()
                .numpy()
                .tobytes()
            )
        positions_hash = digest.hexdigest()
        for _ in sample.batches():
            pass
        if store.date:
            notes.warn(
                f"Datetime column '{spec.time}' is treated as consecutive periods in sorted order; gaps between dates are not detected."
            )
        transformation = "forward orthogonal deviations" if opts.orthogonal else "first differences"
        info["residual_definition"] = (
            "level equation: y minus x'b (includes the panel effect)"
            if opts.system
            else f"transformed equation ({transformation})"
        )
        metrics = {
            "n_groups": store.groups,
            "n_instruments": n_effective,
            "obs_per_group_min": store.smallest,
            "obs_per_group_avg": store.n / store.groups,
            "obs_per_group_max": store.largest,
            "df_model": len(slopes),
            "sigma_e": math.sqrt(sigma_final),
        }
        if df is not None:
            metrics["df_resid"] = df
        extra = {
            "transformation": transformation,
            "equations": "system" if opts.system else "difference",
            "steps": 2 if opts.twostep else 1,
            "h": opts.h,
            "n_obs_transformed": store.nt,
            "n_obs_level": store.nl if opts.system else 0,
            "instrument_groups": {
                key: {"label": label, "columns": inst.groups.count(key)}
                for key, label in inst.labels.items()
            },
            "instrument_columns": n_inst,
            "instruments_planned": len(metadata[0]),
            "instruments_zero_dropped": zero,
            "instruments_collinear_dropped": omitted,
            "weight_matrix_rank": {"one_step": rank1, "two_step": two.rank if two else None},
            "sigma2_one_step": sigma2,
            "windmeijer": bool(opts.twostep and opts.robust),
            "constant": "level equation"
            if constant is not None
            else "differenced out"
            if not opts.system
            else "excluded",
            "rows_lag_only": sample.nrows - store.n,
        }
        result = _result(
            sample,
            terms=terms,
            beta=beta,
            covariance=covariance,
            info=info,
            metrics=metrics,
            notes=notes,
            predictions=predictions,
            tests=tests,
            solver="native_panel_replays_global_GMM_TSQR",
            diagnostics={
                "panel_storage": "owned sorted SQLite, one complete panel numerical block guarded before reading; no full sample or G×L matrix",
                "panel_disk_bytes": store.path.stat().st_size,
                "panel_disk_estimate_bytes": store.disk_estimate,
                "peak_complete_panel_resource_plan": store.peak_panel_plan.record(),
            },
            extra=extra,
            resource=max(
                (resource, store.peak_panel_plan), key=lambda plan: plan.estimated_bytes
            ).record(),
            title=f"{'Two-step' if opts.twostep else 'One-step'} {'system' if opts.system else 'difference'} GMM (dynamic panel data)",
            use_t=opts.small,
            provenance_extra={
                "transformation": transformation,
                "steps": 2 if opts.twostep else 1,
                "equations": "system" if opts.system else "difference",
                "h": opts.h,
                "weight_matrix_inverse": "Moore-Penrose inverse of the correlation-scale moment matrix (ordinary inverse when full rank)",
                "derivatives": "closed form (linear GMM)",
                "sample_position_count": store.n,
                "sample_positions_hash": positions_hash,
                "sample_hash": hashlib.sha256(
                    (sample.baseline["data_hash"] + positions_hash).encode()
                ).hexdigest(),
                "source_complete_rows": sample.nrows,
                "effective_equation_rows": store.n,
                "prediction_sample": "first400 sorted reported equations",
            },
        )
        result.nobs = store.n
        result.dropped_rows = sample.original_count - store.n
        return result


def fit_streaming(spec, source, *, batch_rows=None):
    if not supports(spec):
        raise AnalysisError(
            "streaming_unsupported", "No dynamic-panel GMM replay adapter is registered."
        )
    try:
        with torch.no_grad(), torch.device("cpu"):
            return _fit(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError(
            "dynamic_panel_spill_failed",
            "Dynamic-panel replay needs writable temporary storage and sufficient free disk.",
        ) from error
