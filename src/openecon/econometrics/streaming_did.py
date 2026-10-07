"""Full-sample DID/event-study with native, disk-backed two-way projection."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
import sqlite3

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
import torch

from openecon.analysis import _json_scalar
from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.inference import critical_value
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import wald_test
from .linear.common import check_fit, check_pweights
from .replay_sample import MAX_PARAMETERS, ReplaySample
from .streaming_hdfe import _Selection, _Vectors, _absorb
from .streaming_linear import _Notes, _covariance, _finite, _result, _solve, _weights
from .teffects.did import _TWFE_NOTE, _event_label

SUPPORTED = frozenset({"didregress", "eventstudy"})


def supports(spec):
    return spec.estimator in SUPPORTED


def _period_values(series, *, integer=False):
    if is_integer_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        present = series.dropna()
        if len(present) and (int(present.min()) < -(2**63) or int(present.max()) >= 2**63):
            raise AnalysisError(
                "invalid_time", "Integer periods must fit exact signed int64 time arithmetic."
            )
        return torch.as_tensor(
            series.fillna(0).to_numpy(dtype="int64", copy=True), dtype=torch.int64
        )
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise AnalysisError(
            "invalid_time",
            "Event-study time and treatment time need integer periods or datetime time.",
        )
    vals = torch.as_tensor(
        series.fillna(0).to_numpy(dtype="float64", copy=True), dtype=torch.float64
    )
    if (
        not bool(torch.isfinite(vals).all())
        or integer
        and bool(((vals != vals.round()) | (vals.abs() > 2**53)).any())
    ):
        raise AnalysisError(
            "invalid_time",
            "Floating event periods must be exact integers within float64 period precision.",
        )
    return vals.to(torch.int64) if integer else vals


class _Timing:
    """Sorted period identity and per-group adoption metadata live on owned SQLite."""

    def __init__(self, sample, selection):
        self.sample, self.db = sample, selection.connection
        self.name = sample.spec.time
        self.kind = None
        self.cohort_limit = max(1, sample.category_budget // 256)
        self.db.execute(
            "CREATE TABLE times(key BLOB PRIMARY KEY,value,label TEXT,idx INTEGER) WITHOUT ROWID"
        )
        self.db.execute(
            "CREATE TABLE adoption(key BLOB PRIMARY KEY,first INTEGER,lastoff INTEGER,lo,hi,never INTEGER,ever INTEGER) WITHOUT ROWID"
        )
        self.groups, self.periods = selection.levels

    def seed(self):
        sample, spec = self.sample, self.sample.spec
        timing = (
            registry.role_columns(spec, "treatment_time")[0]
            if spec.estimator == "eventstudy"
            else None
        )
        for batch in sample.batches():
            series = batch.frame[self.name]
            kind = (
                "datetime"
                if is_datetime64_any_dtype(series.dtype)
                else "numeric"
                if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)
                else "label"
            )
            if self.kind is not None and kind != self.kind:
                raise AnalysisError(
                    "invalid_time", "The time column changes type between replay partitions."
                )
            self.kind = kind
            if spec.estimator == "eventstudy" and kind == "label":
                raise AnalysisError(
                    "invalid_time", "Event-study time must contain integer periods or datetimes."
                )
            if kind == "datetime":
                values = series.astype("int64").tolist()
            elif kind == "numeric":
                values = _period_values(series, integer=spec.estimator == "eventstudy").tolist()
            else:
                if not all(isinstance(value, (str, bool)) for value in series):
                    raise AnalysisError(
                        "invalid_time",
                        "DID time labels must be sortable scalar values of one type.",
                    )
                values = series.tolist()
            keys = encode_cluster_labels(series)
            labels = (
                (value.isoformat() for value in series)
                if kind == "datetime"
                else (_json_scalar(value) for value in series)
            )
            records = dict(zip(keys, zip(values, labels), strict=True))
            self.db.executemany(
                "INSERT OR IGNORE INTO times VALUES(?,?,?,NULL)",
                (
                    (key, value, json.dumps(label, allow_nan=False))
                    for key, (value, label) in records.items()
                ),
            )
            if timing:
                _period_values(batch.frame[timing], integer=True)
        self.db.execute(
            "CREATE TABLE time_order AS SELECT key,ROW_NUMBER() OVER(ORDER BY value)-1 AS idx FROM times"
        )
        self.db.execute("CREATE UNIQUE INDEX time_order_key ON time_order(key)")
        self.db.execute(
            "UPDATE times SET idx=(SELECT idx FROM time_order WHERE time_order.key=times.key)"
        )
        self.db.execute("CREATE UNIQUE INDEX time_index ON times(idx)")
        self.db.execute("DROP TABLE time_order")
        self.periods = self.db.execute("SELECT COUNT(*) FROM times").fetchone()[0]
        if self.groups < 2 or self.periods < 2:
            raise AnalysisError(
                "insufficient_observations",
                "DID/event-study needs at least two groups and two periods.",
            )
        self.minimum = self.db.execute("SELECT MIN(value) FROM times").fetchone()[0]
        for batch in sample.batches():
            keys = encode_cluster_labels(batch.frame[spec.panel])
            unique, codes = {}, []
            for key in keys:
                if key not in unique:
                    unique[key] = len(unique)
                codes.append(unique[key])
            code = torch.tensor(codes, dtype=torch.int64)
            n = len(unique)
            if spec.estimator == "didregress":
                treatment = registry.role_columns(spec, "treatment")[0]
                d = batch.numeric(treatment)
                if bool(((d != 0) & (d != 1)).any()):
                    raise AnalysisError("invalid_treatment", "DID treatment must be coded0/1.")
                period = self.codes(batch)
                first = torch.full((n,), self.periods + 1, dtype=torch.int64)
                off = torch.full((n,), -1, dtype=torch.int64)
                first.scatter_reduce_(0, code[d == 1], period[d == 1], "amin")
                off.scatter_reduce_(0, code[d == 0], period[d == 0], "amax")
                records = (
                    (key, int(first[i]), int(off[i]), None, None, 0, 0) for key, i in unique.items()
                )
            else:
                never = torch.as_tensor(batch.frame[timing].isna().to_numpy(), dtype=torch.bool)
                start = _period_values(batch.frame[timing], integer=True)
                lo, hi = (
                    torch.full((n,), torch.iinfo(torch.int64).max, dtype=torch.int64),
                    torch.full((n,), torch.iinfo(torch.int64).min, dtype=torch.int64),
                )
                lo.scatter_reduce_(0, code[~never], start[~never], "amin")
                hi.scatter_reduce_(0, code[~never], start[~never], "amax")
                na, yes = torch.zeros(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64)
                na.scatter_reduce_(0, code, never.to(torch.int64), "amax")
                yes.scatter_reduce_(0, code, (~never).to(torch.int64), "amax")
                records = (
                    (key, self.periods + 1, -1, int(lo[i]), int(hi[i]), int(na[i]), int(yes[i]))
                    for key, i in unique.items()
                )
            self.db.executemany(
                "INSERT INTO adoption VALUES(?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET first=MIN(first,excluded.first),lastoff=MAX(lastoff,excluded.lastoff),lo=MIN(lo,excluded.lo),hi=MAX(hi,excluded.hi),never=MAX(never,excluded.never),ever=MAX(ever,excluded.ever)",
                records,
            )
        self.db.commit()
        if timing:
            bad = self.db.execute(
                "SELECT COUNT(*) FROM adoption WHERE (never=1 AND ever=1) OR (ever=1 AND lo<>hi)"
            ).fetchone()[0]
            if bad:
                raise AnalysisError(
                    "invalid_treatment",
                    "First treatment period must be constant within every group, including never-treated status.",
                )
            self.treated = self.db.execute("SELECT COUNT(*) FROM adoption WHERE ever=1").fetchone()[
                0
            ]
            values = self.db.execute(
                "SELECT DISTINCT lo FROM adoption WHERE ever=1 ORDER BY lo"
            ).fetchmany(self.cohort_limit + 1)
            if len(values) > self.cohort_limit:
                raise AnalysisError(
                    "model_metadata_budget",
                    "The event-study cohort output exceeds the bounded metadata plan.",
                )
            self.starts = [row[0] for row in values]
            if not self.treated:
                raise AnalysisError("invalid_treatment", "No group is ever treated.")
        else:
            self.treated = self.db.execute(
                "SELECT COUNT(*) FROM adoption WHERE first<=?", (self.periods,)
            ).fetchone()[0]
            self.absorbing = not self.db.execute(
                "SELECT COUNT(*) FROM adoption WHERE first<=lastoff"
            ).fetchone()[0]
            values = self.db.execute(
                "SELECT DISTINCT first FROM adoption WHERE first<=? ORDER BY first", (self.periods,)
            ).fetchmany(self.cohort_limit + 1)
            if len(values) > self.cohort_limit:
                raise AnalysisError(
                    "model_metadata_budget", "DID cohort output exceeds the bounded metadata plan."
                )
            self.starts = [row[0] for row in values]
            treated_rows = self.db.execute(
                "SELECT MAX(lastoff),MIN(first) FROM adoption"
            ).fetchone()
            if not self.treated or treated_rows[0] < 0:
                raise AnalysisError(
                    "invalid_treatment",
                    "DID treatment must contain both treated and untreated observations.",
                )

    def codes(self, batch):
        keys = encode_cluster_labels(batch.frame[self.name])
        unique, found = list(dict.fromkeys(keys)), {}
        for start in range(0, len(unique), 500):
            part = unique[start : start + 500]
            found.update(
                self.db.execute(
                    "SELECT key,idx FROM times WHERE key IN(" + ",".join("?" for _ in part) + ")",
                    part,
                )
            )
        if len(found) != len(unique):
            raise AnalysisError("source_changed", "Time identity changed between replay passes.")
        return torch.tensor([found[key] for key in keys], dtype=torch.int64)

    def group_treated(self, batch):
        keys = encode_cluster_labels(batch.frame[self.sample.spec.panel])
        found = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), 500):
            part = unique[start : start + 500]
            found.update(
                self.db.execute(
                    "SELECT key,first FROM adoption WHERE key IN("
                    + ",".join("?" for _ in part)
                    + ")",
                    part,
                )
            )
        return torch.tensor([found[key] <= self.periods for key in keys], dtype=torch.float64)

    def label(self, code):
        return json.loads(
            self.db.execute("SELECT label FROM times WHERE idx=?", (code,)).fetchone()[0]
        )

    def relative(self, batch):
        spec = self.sample.spec
        name = registry.role_columns(spec, "treatment_time")[0]
        treated = torch.as_tensor(~batch.frame[name].isna().to_numpy(), dtype=torch.bool)
        start = _period_values(batch.frame[name], integer=True)
        time = (
            self.codes(batch)
            if self.kind == "datetime"
            else _period_values(batch.frame[spec.time], integer=True)
        )
        relative = time - start
        if bool(
            (
                ((time >= 0) & (start < 0) & (relative < 0))
                | ((time < 0) & (start >= 0) & (relative > 0))
            )[treated].any()
        ):
            raise AnalysisError(
                "invalid_time", "Relative periods overflow exact int64 time arithmetic."
            )
        return relative, treated


@dataclass
class _Fit:
    terms: list
    beta: torch.Tensor
    covariance: torch.Tensor
    info: dict
    metrics: dict
    predictions: list
    diagnostics: dict
    extra: dict
    resource: dict


def _twfe(sample, selection, names, columns, notes):
    controls = sample.designs["controls"]
    terms = [*names, *controls.terms]
    if len(terms) != len(set(terms)):
        raise AnalysisError(
            "invalid_spec", "Generated event/treatment terms collide with a covariate name."
        )
    width = len(terms) + 1
    if width > MAX_PARAMETERS:
        raise AnalysisError(
            "model_too_wide",
            "DID/event-study projection exceeds the bounded global parameter plan.",
        )
    clusters = (
        [sample.spec.panel]
        if sample.spec.covariance == "robust"
        else registry.cluster_columns(sample.spec)
        if sample.spec.covariance == "cluster"
        else []
    )
    levels, redundant, nested, total = selection.degrees_of_freedom()
    absorbed = total + int(all(nested))
    resource = sample.plan_rows(
        "native DID/event-study two-way projection",
        {
            "selection_SQLite_cache": 2 * 1024**2,
            "projection_SQLite_cache": 2 * 1024**2,
            "cluster_accumulator_caches": 18 * 1024**2 if clusters else 0,
            "global_TSQR_covariance": 1024 * width**2,
            "period_and_cohort_metadata": sample.category_budget,
        },
        192 * width + 768,
    )
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(sample.spec.outcome)
        moments.add(torch.stack((torch.ones_like(y), y), 1), _weights(sample, batch), False)
    anchor, magnitude, center = (
        float(moments.anchor[1]),
        float(moments.magnitude[1]),
        float(moments.mean[1]),
    )
    y_scale = magnitude * math.sqrt(float(moments.m2.value[1] / moments.mass))
    if not math.isfinite(y_scale) or y_scale <= 0:
        raise AnalysisError(
            "constant_outcome", "DID/event-study outcome needs positive finite variation."
        )
    before, tss = _CompensatedSum((width - 1,)), _CompensatedSum(())
    with ExitStack() as stack:
        states = _Vectors(sample, width, 2, selection.path.stat().st_size)
        stack.callback(states.close)
        with states.writer("raw") as writer:
            for batch in sample.batches():
                y = ((batch.numeric(sample.spec.outcome) - anchor) / magnitude - center) * (
                    magnitude / y_scale
                )
                x = torch.cat((columns(batch), batch.designs["controls"]), 1)
                weights = _weights(sample, batch)
                before.add((x.square() * weights[:, None]).sum(0))
                tss.add((y.square() * weights).sum())
                states.write(writer, torch.cat((y[:, None], x), 1))
        iteration, update, method = _absorb(
            states, sample, [sample.spec.panel, sample.spec.time], 1e-10, 10000
        )
        tree, within, within_y = _TSQRTree(), _CompensatedSum((width - 1,)), _CompensatedSum(())
        for batch, (values,) in states.batches(sample, "y"):
            weights = _weights(sample, batch)
            within.add((values[:, 1:].square() * weights[:, None]).sum(0))
            within_y.add((values[:, 0].square() * weights).sum())
            tree.add(
                torch.linalg.qr(
                    torch.cat((values[:, 1:], values[:, :1]), 1) * weights.sqrt()[:, None], mode="r"
                )[1]
            )
        factor = tree.finish()
        removed = (within.value <= 1e-13 * before.value).nonzero().flatten().tolist()
        remaining = [i for i in range(len(terms)) if i not in removed]
        if not remaining:
            raise AnalysisError(
                "no_within_variation",
                "All DID/event-study regressors are absorbed by group and time effects.",
            )
        kept, omitted = collinear_columns(factor[:, remaining])
        selected = [remaining[i] for i in kept]
        for i in [*removed, *(remaining[i] for i in omitted)]:
            sample.notes["omitted_terms"].append(terms[i])
            notes.warn(
                f"Omitted {terms[i]}: collinearity with other regressors or group/time fixed effects."
            )
        beta, bread, condition = _solve(factor[:, [*selected, factor.shape[1] - 1]], len(selected))
        df = sample.nobs - len(selected) - absorbed
        if df <= 0:
            raise AnalysisError(
                "insufficient_observations",
                "Two-way fixed effects leave no residual degrees of freedom.",
            )
        # A control-free design has no scale vector; generated effects still
        # have unit scale before the common outcome transformation.
        control_scales = (
            controls.scales[controls.kept]
            if controls.kept
            else torch.empty(0, dtype=torch.float64)
        )
        all_scales = torch.cat((torch.ones(len(names), dtype=torch.float64), control_scales))
        scales = all_scales[selected]

        def factory():
            for batch, (values,) in states.batches(sample, "y"):
                yield (
                    batch,
                    values[:, [i + 1 for i in selected]],
                    values[:, 0],
                    batch.numeric(sample.spec.outcome),
                    y_scale,
                )

        covariance, info, rss_work, predictions = _covariance(
            sample,
            factory,
            beta,
            bread,
            df=df,
            k=len(selected) + absorbed,
            notes=notes,
            score_basis=torch.diag(scales),
            kind="cluster" if sample.spec.covariance == "robust" else sample.spec.covariance,
            cluster_columns=clusters,
        )
        if sample.spec.covariance == "robust":
            info["covariance"] = "robust"
            info["correction"] = "cluster-robust at the group level (" + info["correction"] + ")"
        rss, total, total_within = (
            rss_work * y_scale * y_scale,
            float(tss.value) * y_scale * y_scale,
            float(within_y.value) * y_scale * y_scale,
        )
        check_fit(
            rss,
            total,
            df,
            total + sample.nobs * (anchor + magnitude * center) ** 2,
            absorbed=True,
            resolution=(32e-10) ** 2 * total_within,
        )
        transform = torch.diag(y_scale / scales)
        beta, covariance = transform @ beta, transform @ covariance @ transform.T
        _finite(beta, covariance)
        info.update(
            {
                "df_resid": df,
                "k_small_sample": len(selected) + absorbed,
                "absorbed_degrees_of_freedom": absorbed,
                "degrees_of_freedom_convention": "non-nested observed fixed-effect levels minus connected-component redundancies; nesting in any declared cluster; constant counted when all dimensions nest",
            }
        )
        metrics = {
            "r_squared": 1 - rss / total if total > 0 else None,
            "r_squared_within": 1 - rss / total_within if total_within > 0 else None,
            "rmse": math.sqrt(rss / df),
            "df_resid": df,
            "n_groups": levels[0],
            "n_periods": levels[1],
        }
        if "cluster_count" in info:
            metrics["n_clusters"] = info["cluster_count"]
        diagnostics = {
            "condition_number": condition,
            "tsqr_reduction_depth": tree.depth,
            **selection.diagnostics(),
            **states.diagnostics(),
        }
        extra = {
            "absorbed_degrees_of_freedom": absorbed,
            "iterations": iteration,
            "max_update": update,
            "method": method,
            "absorbed": [
                {"column": name, "levels": n, "redundant": r, "nested": flag}
                for name, n, r, flag in zip(
                    [sample.spec.panel, sample.spec.time], levels, redundant, nested, strict=True
                )
            ],
        }
        return _Fit(
            [terms[i] for i in selected],
            beta,
            covariance,
            info,
            metrics,
            predictions,
            diagnostics,
            extra,
            resource.record(),
        )


def fit_streaming(spec, source, *, batch_rows=None):
    if not supports(spec):
        raise AnalysisError(
            "streaming_unsupported",
            "No replay DID/event-study adapter is registered for this model.",
        )
    try:
        with torch.no_grad(), torch.device("cpu"), ExitStack() as stack:
            timing_column = (
                registry.role_columns(spec, "treatment_time")[0]
                if spec.estimator == "eventstudy"
                else None
            )
            sample = ReplaySample(
                spec,
                source,
                batch_rows=batch_rows,
                allow_missing=[timing_column] if timing_column else [],
            )
            controls = sample.add_design("controls", intercept=False)
            sample.prepare()
            notes = _Notes(spec)
            check_pweights(notes)
            if spec.covariance not in {"nonrobust", "HC1", "robust", "cluster"}:
                raise AnalysisError(
                    "unsupported_streaming_covariance",
                    "DID/event-study replay supports classical, HC1 and one/two-way cluster covariance.",
                )
            clusters = (
                [spec.panel]
                if spec.covariance == "robust"
                else registry.cluster_columns(spec)
                if spec.covariance == "cluster"
                else []
            )
            if len(clusters) > 2:
                raise AnalysisError(
                    "unsupported_cluster_dimensions",
                    "DID/event-study supports one or two declared cluster dimensions.",
                )
            sample.plan_rows(
                "DID/event-study selection metadata", {"selection_SQLite_cache": 2 * 1024**2}, 768
            )
            selection = _Selection(2, len(clusters))
            stack.callback(selection.close)
            selection.seed(sample, [spec.panel, spec.time], clusters, False)
            timing = _Timing(sample, selection)
            timing.seed()
            if spec.covariance == "robust" and timing.groups < 30:
                notes.warn(
                    f"Only {timing.groups} groups (clusters); cluster-robust inference may be unreliable."
                )
            if spec.estimator == "didregress":
                treatment = registry.role_columns(spec, "treatment")[0]
                if treatment in spec.predictors:
                    raise AnalysisError(
                        "invalid_spec", "The treatment indicator must not also be a covariate."
                    )
                term = f"ATET:r1vs0.{treatment}"

                def columns(batch):
                    return batch.numeric(treatment)[:, None]

                fit = _twfe(sample, selection, [term], columns, notes)
                if not fit.terms or fit.terms[0] != term:
                    raise AnalysisError(
                        "no_within_variation",
                        "DID treatment is collinear with group and time fixed effects.",
                    )
                tests, extra = (
                    {},
                    {
                        "treatment": treatment,
                        "group": spec.panel,
                        "time": spec.time,
                        "treated_groups": timing.treated,
                        "control_groups": timing.groups - timing.treated,
                    },
                )
                if not timing.absorbing:
                    extra.update(
                        {
                            "adoption": "not absorbing",
                            "tests_note": "Parallel-trends and anticipation tests need absorbing group-level treatment.",
                        }
                    )
                else:
                    extra.update(
                        {
                            "adoption": "common" if len(timing.starts) == 1 else "staggered",
                            "first_treated_periods": [timing.label(i) for i in timing.starts],
                        }
                    )
                    if len(timing.starts) != 1 or timing.treated == timing.groups:
                        extra["tests_note"] = (
                            "Parallel-trends/anticipation tests need one common start and never-treated controls; use eventstudy."
                        )
                    elif timing.starts[0] < 2:
                        extra["tests_note"] = (
                            "At least two pre-treatment periods are needed for the tests."
                        )
                    else:
                        start = timing.starts[0]

                        def trend(batch):
                            period = timing.codes(batch)
                            clock = (
                                (_period_values(batch.frame[spec.time]) - timing.minimum).to(
                                    torch.float64
                                )
                                if timing.kind == "numeric"
                                else period.to(torch.float64)
                            )
                            return torch.cat(
                                (
                                    columns(batch),
                                    (timing.group_treated(batch) * clock * (period < start)).to(
                                        torch.float64
                                    )[:, None],
                                ),
                                1,
                            )

                        def leads(batch):
                            period = timing.codes(batch)
                            treated = timing.group_treated(batch)
                            return torch.cat(
                                (
                                    columns(batch),
                                    treated[:, None]
                                    * (period[:, None] == torch.arange(start - 1, 0, -1)).to(
                                        torch.float64
                                    ),
                                ),
                                1,
                            )

                        comparisons = [
                            (
                                "parallel_trends",
                                ["_pretrend"],
                                trend,
                                "Parallel-trends test: treated-group linear trend before treatment = 0 (estat ptrends)",
                            )
                        ]
                        if start + len(controls.terms) + 1 <= MAX_PARAMETERS:
                            comparisons.append(
                                (
                                    "granger",
                                    [f"_lead{i}" for i in range(1, start)],
                                    leads,
                                    "Anticipation (Granger) test: treatment leads = 0 (estat granger)",
                                )
                            )
                        else:
                            extra["granger_note"] = (
                                "Optional anticipation design exceeds the bounded parameter/workspace plan; no statistic or sampled substitute is reported."
                            )
                        for key, names, fn, label in comparisons:
                            augmented = _twfe(sample, selection, [term, *names], fn, notes)
                            fit.diagnostics.setdefault("auxiliary_projections", {})[key] = (
                                augmented.diagnostics
                            )
                            indices = [i for i, t in enumerate(augmented.terms) if t in names]
                            tests[key] = wald_test(
                                augmented.beta,
                                augmented.covariance,
                                indices,
                                df_resid=augmented.info["df_inference"],
                                label=label,
                            )
                if extra.get("adoption") == "staggered":
                    notes.warn(_TWFE_NOTE)
                title = "Difference-in-differences regression"
                equations = ["ATET" if t == term else "Controls" for t in fit.terms]
            else:
                relative_low, relative_high = (
                    torch.iinfo(torch.int64).max,
                    torch.iinfo(torch.int64).min,
                )
                for batch in sample.batches():
                    relative, treated = timing.relative(batch)
                    if bool(treated.any()):
                        relative_low, relative_high = (
                            min(relative_low, int(relative[treated].min())),
                            max(relative_high, int(relative[treated].max())),
                        )
                first = (
                    -int(notes.option("leads"))
                    if notes.option("leads") is not None
                    else relative_low
                )
                last = (
                    int(notes.option("lags")) if notes.option("lags") is not None else relative_high
                )
                reference = int(notes.option("reference"))
                if not first <= reference <= last:
                    raise AnalysisError(
                        "invalid_spec", "Reference period lies outside the event window."
                    )
                if last - first + 1 > MAX_PARAMETERS - len(controls.terms):
                    raise AnalysisError(
                        "model_too_wide",
                        "Event window exceeds the bounded parameter/output plan; give leads and lags.",
                    )
                counts = torch.zeros(last - first + 1, dtype=torch.int64)
                for batch in sample.batches():
                    relative, treated = timing.relative(batch)
                    counts += torch.bincount(
                        relative[treated].clamp(first, last) - first, minlength=len(counts)
                    )
                events = [
                    e for e in range(first, last + 1) if e != reference and counts[e - first] > 0
                ]
                names = [_event_label(e) for e in events]

                def columns(batch):
                    relative, treated = timing.relative(batch)
                    return (
                        treated[:, None]
                        & (
                            relative.clamp(first, last)[:, None]
                            == torch.tensor(events, dtype=torch.int64)
                        )
                    ).to(torch.float64)

                fit = _twfe(sample, selection, names, columns, notes)
                kept = {term: i for i, term in enumerate(fit.terms)}
                indices = [
                    kept[_event_label(e)] for e in events if e < 0 and _event_label(e) in kept
                ]
                tests = (
                    {
                        "pretrends": wald_test(
                            fit.beta,
                            fit.covariance,
                            indices,
                            df_resid=fit.info["df_inference"],
                            label="Joint pre-trend test: all leads = 0",
                        )
                    }
                    if indices
                    else {}
                )
                critical = critical_value(spec.alpha, fit.info["df_inference"])
                table = []
                for e in range(first, last + 1):
                    row = {
                        "relative_time": e,
                        "term": _event_label(e),
                        "reference": e == reference,
                        "binned": (
                            e == first
                            and notes.option("leads") is not None
                            and relative_low < first
                        )
                        or (
                            e == last and notes.option("lags") is not None and relative_high > last
                        ),
                        "observations": int(counts[e - first]),
                    }
                    if e == reference:
                        row.update(
                            {"estimate": 0.0, "std_error": 0.0, "ci_low": 0.0, "ci_high": 0.0}
                        )
                    elif _event_label(e) in kept:
                        i = kept[_event_label(e)]
                        value, se = float(fit.beta[i]), math.sqrt(float(fit.covariance[i, i]))
                        row.update(
                            {
                                "estimate": value,
                                "std_error": se,
                                "ci_low": value - critical * se,
                                "ci_high": value + critical * se,
                            }
                        )
                    else:
                        row.update(
                            {"estimate": None, "std_error": None, "ci_low": None, "ci_high": None}
                        )
                    table.append(row)
                extra = {
                    "event_table": table,
                    "reference": reference,
                    "window": [first, last],
                    "cohorts": timing.starts,
                    "never_treated_groups": timing.groups - timing.treated,
                    "treated_groups": timing.treated,
                }
                post = [kept[_event_label(e)] for e in events if e >= 0 and _event_label(e) in kept]
                if post:
                    vector = torch.zeros(len(fit.beta), dtype=torch.float64)
                    vector[post] = 1 / len(post)
                    value, se = (
                        float(vector @ fit.beta),
                        math.sqrt(float(vector @ fit.covariance @ vector)),
                    )
                    extra["average_post_effect"] = {
                        "estimate": value,
                        "std_error": se,
                        "ci_low": value - critical * se,
                        "ci_high": value + critical * se,
                        "periods": [e for e in events if e >= 0],
                    }
                if len(timing.starts) > 1:
                    notes.warn(_TWFE_NOTE)
                if timing.kind == "datetime":
                    notes.warn(
                        "Datetime time is treated as consecutive sorted periods; gaps between dates are not inferred."
                    )
                equations = ["Event" if t in names else "Controls" for t in fit.terms]
                title = "Event-study regression"
            fit.extra.update(extra)
            result = _result(
                sample,
                terms=fit.terms,
                beta=fit.beta,
                covariance=fit.covariance,
                info=fit.info,
                metrics=fit.metrics,
                notes=notes,
                predictions=fit.predictions,
                tests=tests,
                solver="native_disk_two_way_projection_TSQR",
                diagnostics=fit.diagnostics,
                extra=fit.extra,
                resource=fit.resource,
                title=title,
                provenance_extra={"absorbed": [spec.panel, spec.time], "singletons_dropped": 0},
            )
            for coefficient, equation in zip(result.coefficients, equations, strict=True):
                coefficient.equation = equation
            return result
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError(
            "fixed_effect_spill_failed",
            "DID/event-study needs writable owned temporary disk storage.",
        ) from error
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
