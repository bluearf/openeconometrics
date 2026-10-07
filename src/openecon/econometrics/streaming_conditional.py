"""Whole-group conditional logit replay; no global group/observation arrays."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import combinations
import hashlib
import math
import sqlite3

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import collinear_columns
from openecon.engines.optimize import information_inverse
from openecon.engines.covariance import nearest_psd
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.streaming_design import encode_cluster_labels
from openecon.resources import plan_workspace

from . import registry
from .core import information_criteria, lr_test, wald_test
from .discrete.common import (
    EXTREME,
    EXTREME_EARLY,
    Separated,
    check_pweights,
    maximize,
    raise_not_converged,
    separation_watch,
)
from .discrete.conditional import ConditionalLogitObjective
from .replay_sample import ReplaySample
from .streaming_linear import _GroupMeans, _Notes, _finite, _result, _scratch_directory

SUPPORTED = frozenset({"clogit"})


def supports(spec):
    return spec.estimator in SUPPORTED


class _Store:
    def __init__(self, sample, notes):
        self.sample, self.notes = sample, notes
        self.group = registry.role_columns(sample.spec, "group")[0]
        self.offset = registry.role_columns(sample.spec, "offset")
        self.clusters = (
            registry.cluster_columns(sample.spec) if sample.spec.covariance == "cluster" else []
        )
        self.design = sample.designs["x"]
        self.width = len(self.design.terms)
        self.means = _GroupMeans(self.width, len(self.clusters))
        self.db = self.means.connection
        self.db.execute(
            "CREATE TABLE groups(key BLOB PRIMARY KEY,n INTEGER,m INTEGER,low REAL,high REAL) WITHOUT ROWID"
        )
        self.db.execute(
            "CREATE TABLE records(position INTEGER PRIMARY KEY,key BLOB,v BLOB,"
            + ",".join(f"c{i} BLOB" for i in range(len(self.clusters)))
            + ")"
            if self.clusters
            else "CREATE TABLE records(position INTEGER PRIMARY KEY,key BLOB,v BLOB)"
        )
        self.db.execute("CREATE INDEX group_records ON records(key,position)")
        self.peak_rows = 0

    def seed(self):
        sample, spec = self.sample, self.sample.spec
        for batch in sample.batches():
            y = batch.numeric(spec.outcome)
            if bool(((y != 0) & (y != 1)).any()):
                raise AnalysisError(
                    "invalid_outcome", "Conditional logit requires outcomes coded0/1."
                )
            x = batch.designs["x"]
            keys = encode_cluster_labels(batch.frame[self.group])
            clusters = [encode_cluster_labels(batch.frame[name]) for name in self.clusters]
            self.means.add(keys, x, batch.weights, clusters)
            unique, codes = self.means._codes(keys)
            n = torch.bincount(codes, minlength=len(unique))
            m = torch.zeros(len(unique), dtype=torch.int64).index_add_(0, codes, y.to(torch.int64))
            low = torch.full((len(unique),), math.inf, dtype=torch.float64).scatter_reduce_(
                0, codes, batch.weights, "amin"
            )
            high = torch.zeros(len(unique), dtype=torch.float64).scatter_reduce_(
                0, codes, batch.weights, "amax"
            )
            self.db.executemany(
                "INSERT INTO groups VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET n=n+excluded.n,m=m+excluded.m,low=MIN(low,excluded.low),high=MAX(high,excluded.high)",
                (
                    (key, int(n[i]), int(m[i]), float(low[i]), float(high[i]))
                    for i, key in enumerate(unique)
                ),
            )
            offset = batch.numeric(self.offset[0]) if self.offset else torch.zeros_like(y)
            packed = memoryview(
                torch.cat((y[:, None], offset[:, None], x), 1).contiguous().numpy().tobytes()
            )
            stride = 8 * (self.width + 2)
            self.db.executemany(
                "INSERT INTO records VALUES("
                + ",".join("?" for _ in range(3 + len(clusters)))
                + ")",
                (
                    (
                        int(position),
                        key,
                        packed[i * stride : (i + 1) * stride],
                        *(col[i] for col in clusters),
                    )
                    for i, (position, key) in enumerate(
                        zip(batch.positions.tolist(), keys, strict=True)
                    )
                ),
            )
        self.means.finish()
        if self.db.execute("SELECT COUNT(*) FROM groups WHERE low<>high").fetchone()[0]:
            raise AnalysisError(
                "weights_not_constant_within_group",
                "Conditional likelihood weights must be constant within each entire group.",
            )
        dropped, nlost, flost, flostrows = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(SUM(low),0),COALESCE(SUM(low*n),0) FROM groups WHERE m=0 OR m=n"
        ).fetchone()
        self.dropped_groups = int(flost) if spec.weight_type == "fweight" else dropped
        self.dropped_observations = int(flostrows) if spec.weight_type == "fweight" else nlost
        if self.dropped_groups:
            self.notes.warn(
                f"note: {self.dropped_groups} group(s) ({self.dropped_observations} obs) dropped because of all positive or all negative outcomes."
            )
        self.db.execute("DELETE FROM records WHERE key IN(SELECT key FROM groups WHERE m=0 OR m=n)")
        self.db.execute("DELETE FROM groups WHERE m=0 OR m=n")
        row = self.db.execute(
            "SELECT COUNT(*),SUM(n),SUM(low),SUM(low*n),MAX(n),MAX(MIN(m,n-m)),MIN(n),MAX(m>1) FROM groups"
        ).fetchone()
        self.groups, self.rows = row[:2]
        if not self.groups:
            raise AnalysisError(
                "no_outcome_variation", "No group has both conditional-logit outcomes."
            )
        self.nobs = int(row[3]) if spec.weight_type == "fweight" else int(self.rows)
        self.effective_groups = int(row[2]) if spec.weight_type == "fweight" else int(self.groups)
        if self.nobs > 2**53 or self.effective_groups > 2**53:
            raise AnalysisError(
                "precision_unsupported",
                "Conditional logit frequency counts exceed exact float64 integers.",
            )
        self.largest, self.recursion_width, self.smallest, self.multiple = (
            int(row[4]),
            int(row[5]),
            int(row[6]),
            bool(row[7]),
        )
        self.mean_size = (
            float(row[3] / row[2])
            if spec.weight_type == "fweight"
            else float(self.rows / self.groups)
        )
        self.weight_mean = float(row[3] / self.rows)
        for index, name in enumerate(self.clusters):
            split = self.db.execute(
                "SELECT COUNT(*) FROM groups g JOIN means m ON m.key=g.key WHERE (m.nested&?)=0",
                (1 << index,),
            ).fetchone()[0]
            if split:
                self.notes.warn(
                    f"{split} group(s) of '{self.group}' are split across the clusters of '{name}'. Cluster covariance uses the within-group-centered per-observation score convention (Stata requires nonest for this design)."
                )
        self.db.commit()
        self.resource = sample.plan_rows(
            "whole-group conditional likelihood replay",
            {
                "conditional_group_SQLite_cache": 6 * 1024**2,
                "conditional_derivative_factors": 128 * (self.width + 1) ** 2,
                "conditional_cluster_caches": 12 * 1024**2 if self.clusters else 0,
            },
            128 * (self.width * self.width + self.recursion_width + len(self.clusters) + 8),
        )
        per_row = 128 * (self.width * self.width + self.recursion_width + len(self.clusters) + 8)
        reserve = dict(self.resource.buffers)
        reserve.pop("native_row_block")
        available = (sample.working_bytes - sum(reserve.values())) // per_row
        self.group_rows = max(sample.rows, min(self.largest, available))
        self.resource = plan_workspace(
            "whole-group conditional likelihood replay",
            {**reserve, "native_row_block": per_row * self.group_rows},
            budget_bytes=sample.working_bytes,
        )
        sample.adapter_plan = self.resource
        if self.largest > self.group_rows:
            raise AnalysisError(
                "conditional_group_workspace",
                "One complete group and its exact conditional recursion exceed the bounded workspace plan; reduce group/recursion width or increase the workspace budget.",
            )
        tree = _TSQRTree()
        for objective, _, _, _ in self.blocks(screening=True):
            tree.add(qr_factor(objective.x * objective.w_obs.sqrt()[:, None]))
        self.kept, self.omitted = collinear_columns(tree.finish())
        for index in self.omitted:
            self.notes.warn(
                f"Omitted because absorbed or collinear within groups: {self.design.terms[index]}."
            )
        if not self.kept:
            raise AnalysisError(
                "no_within_group_variation",
                "No regressor varies within informative conditional-logit groups.",
            )
        if self.effective_groups <= len(self.kept):
            raise AnalysisError(
                "insufficient_observations",
                "Conditional logit needs more informative groups than slope parameters.",
            )

    def blocks(self, screening=False):
        cursor = self.db.execute("SELECT key,n,low FROM groups ORDER BY key")
        pending = cursor.fetchone()
        while pending is not None:
            selected, used = [], 0
            while pending is not None and used + pending[1] <= self.group_rows:
                selected.append(pending)
                used += pending[1]
                pending = cursor.fetchone()
            if not selected:
                raise AnalysisError(
                    "conditional_group_workspace",
                    "A complete conditional-likelihood group exceeds the planned block.",
                )
            keys = [row[0] for row in selected]
            mapping = {key: index for index, key in enumerate(keys)}
            records = []
            for start in range(0, len(keys), 500):
                part = keys[start : start + 500]
                records.extend(
                    self.db.execute(
                        "SELECT key,v,"
                        + ",".join(f"c{i}" for i in range(len(self.clusters)))
                        + " FROM records WHERE key IN("
                        + ",".join("?" for _ in part)
                        + ") ORDER BY key,position",
                        part,
                    )
                    if self.clusters
                    else self.db.execute(
                        "SELECT key,v FROM records WHERE key IN("
                        + ",".join("?" for _ in part)
                        + ") ORDER BY key,position",
                        part,
                    )
                )
            packed = torch.frombuffer(
                bytearray(b"".join(row[1] for row in records)), dtype=torch.float64
            ).reshape(-1, self.width + 2)
            rowkeys = [row[0] for row in records]
            x = packed[:, 2:] - self.means.lookup(rowkeys)
            if not screening:
                x = x[:, self.kept]
            codes = torch.tensor([mapping[key] for key in rowkeys], dtype=torch.int64)
            weights = torch.tensor([row[2] for row in selected], dtype=torch.float64)
            objective = ConditionalLogitObjective(
                x, packed[:, 0], codes, len(keys), weights, packed[:, 1] if self.offset else None
            )
            self.peak_rows = max(self.peak_rows, len(records))
            self.sample.actual_numeric_peak_rows = max(
                self.sample.actual_numeric_peak_rows, len(records)
            )
            yield (
                objective,
                weights,
                keys,
                [[row[2 + i] for row in records] for i in range(len(self.clusters))],
            )

    def close(self):
        self.means.close()


class _Objective:
    def __init__(self, store):
        self.store, self.width = store, len(store.kept)

    def __call__(self, theta):
        value, gradient, hessian = (
            _CompensatedSum(()),
            _CompensatedSum((self.width,)),
            _CompensatedSum((self.width, self.width)),
        )
        for objective, _, _, _ in self.store.blocks():
            v, g, h = objective(theta)
            value.add(v)
            gradient.add(g)
            hessian.add(h)
        return value.value, gradient.value, (hessian.value + hessian.value.T) / 2

    def value(self, theta):
        value = _CompensatedSum(())
        for objective, _, _, _ in self.store.blocks():
            value.add(objective.value(theta))
        return value.value

    def separated(self, theta, threshold):
        total = 0
        for objective, _, _, _ in self.store.blocks():
            total += int((objective.group_log_likelihood(theta) > -threshold).sum())
        return total


def _covariance(store, theta, hessian):
    spec = store.sample.spec
    kind, k = spec.covariance, len(theta)
    bread = information_inverse(-hessian)
    info = {
        "covariance": kind,
        "df_inference": None,
        "df_resid": None,
        "small_sample_correction": None,
    }
    if kind == "nonrobust":
        info["correction"] = "observed information of the complete conditional likelihood"
        return bread, info
    meat = _CompensatedSum((k, k))
    with ExitStack() as stack:
        subsets = [
            combo
            for size in range(1, len(store.clusters) + 1)
            for combo in combinations(range(len(store.clusters)), size)
        ]
        accumulators = []
        for _ in subsets:
            acc = ClusterAccumulator(k, scratch_directory=_scratch_directory())
            stack.callback(acc.close)
            accumulators.append(acc)
        for objective, weights, _, labels in store.blocks():
            if kind in {"robust", "opg"}:
                scores = objective.group_scores(theta)
                multiplier = weights.sqrt() if spec.weight_type == "fweight" else weights
                scores = scores * multiplier[:, None]
                meat.add(scores.T @ scores)
            elif kind == "cluster":
                scores = objective.score_rows(theta) * weights[objective.codes, None]
                for indexes, acc in zip(subsets, accumulators, strict=True):
                    keys = (
                        labels[indexes[0]]
                        if len(indexes) == 1
                        else [
                            b"".join(len(key).to_bytes(8, "big") + key for key in cell)
                            for cell in zip(*(labels[i] for i in indexes), strict=True)
                        ]
                    )
                    acc.add(keys, scores)
            else:
                raise AnalysisError(
                    "unsupported_covariance",
                    "Conditional logit supports information, group OPG, group robust or declared cluster covariance.",
                )
        if kind == "opg":
            info["correction"] = "outer product of the group scores"
            return information_inverse(meat.value), info
        if kind == "robust":
            groups = store.effective_groups
            if groups < 2:
                raise AnalysisError(
                    "insufficient_clusters",
                    "Group robust covariance needs at least two informative groups.",
                )
            factor = groups / (groups - 1)
            matrix = meat.value * factor
            info.update(
                {
                    "correction": "robust = cluster sandwich on the group (Stata's clogit): G/(G-1)",
                    "cluster_count": groups,
                    "cluster_df": groups - 1,
                    "cluster_column": store.group,
                    "small_sample_correction": factor,
                }
            )
        else:
            counts = []
            for combo, acc in zip(subsets, accumulators, strict=True):
                value, count = acc.finish()
                meat.add(value * (1 if len(combo) % 2 else -1))
                if len(combo) == 1:
                    counts.append(count)
            matrix = meat.value
            adjusted = False
            if len(store.clusters) > 1:
                units = store.design.scales[store.design.kept][store.kept]
                matrix, adjusted = nearest_psd(matrix * units[:, None] * units[None, :])
                matrix = matrix / units[:, None] / units[None, :]
            groups = min(counts)
            factor = groups / (groups - 1)
            matrix *= factor
            info.update(
                {
                    "correction": "cluster sandwich: G/(G-1)"
                    if len(store.clusters) == 1
                    else "multiway cluster sandwich (inclusion-exclusion): G_min/(G_min-1)",
                    "cluster_count": groups,
                    "cluster_counts": counts,
                    "cluster_columns": store.clusters,
                    "cluster_column": store.clusters[0],
                    "cluster_df": groups - 1,
                    "small_sample_correction": factor,
                    "psd_adjusted": adjusted,
                }
            )
        covariance = bread @ matrix @ bread
        return (covariance + covariance.T) / 2, info


def fit_streaming(spec, source, *, batch_rows=None):
    if not supports(spec):
        raise AnalysisError(
            "streaming_unsupported", "No whole-group conditional likelihood adapter is registered."
        )
    try:
        with torch.no_grad(), torch.device("cpu"), ExitStack() as stack:
            notes = _Notes(spec)
            check_pweights(notes)
            group = registry.role_columns(spec, "group")[0]
            if group == spec.outcome or group in spec.predictors:
                raise AnalysisError(
                    "invalid_spec",
                    "Conditional-likelihood groups must differ from the outcome and regressors.",
                )
            sample = ReplaySample(spec, source, batch_rows=batch_rows)
            sample.add_design("x", intercept=False, implicit_constant=True)
            sample.prepare()
            if not sample.designs["x"].terms:
                raise AnalysisError(
                    "no_within_group_variation", "Conditional logit needs varying regressors."
                )
            sample.plan_rows(
                "conditional group metadata discovery",
                {"conditional_metadata_SQLite_cache": 6 * 1024**2},
                128 * (len(sample.designs["x"].terms) + len(registry.cluster_columns(spec)) + 8),
            )
            store = _Store(sample, notes)
            stack.callback(store.close)
            store.seed()
            objective = _Objective(store)
            start = torch.zeros(objective.width, dtype=torch.float64)
            scale = (
                1.0
                if spec.weight_type == "fweight" or 0.5 <= store.weight_mean <= 2
                else 1 / store.weight_mean
            )
            try:
                run = maximize(
                    objective,
                    start,
                    what="clogit",
                    callback=separation_watch(objective.separated),
                    scale=scale,
                )
            except Separated as stop:
                raise_not_converged(
                    "clogit",
                    objective.separated(stop.theta, EXTREME_EARLY),
                    "conditional likelihood is monotone",
                    unit="group",
                )
            if not run.converged:
                raise_not_converged(
                    "clogit",
                    objective.separated(run.theta, EXTREME),
                    run.diagnostics.get("message"),
                    unit="group",
                )
            covariance, info = _covariance(store, run.theta, run.hessian)
            transform = store.design.transform[store.kept][:, store.kept]
            beta = transform @ run.theta
            covariance = transform @ covariance @ transform.T
            _finite(beta, covariance)
            null = float(objective.value(start))
            ll = run.value
            criteria = information_criteria(ll, len(beta), store.nobs)
            tests = {
                "model": lr_test(ll, null, len(beta), label="LR chi2 test against b = 0")
                if spec.covariance == "nonrobust"
                else wald_test(
                    beta, covariance, range(len(beta)), label="Wald chi2 test of the coefficients"
                )
            }
            digest = hashlib.sha256()
            retained = 0
            for batch in sample.batches():
                keys = encode_cluster_labels(batch.frame[group])
                found = set()
                for first in range(0, len(keys), 500):
                    part = keys[first : first + 500]
                    found.update(
                        row[0]
                        for row in store.db.execute(
                            "SELECT key FROM groups WHERE key IN("
                            + ",".join("?" for _ in part)
                            + ")",
                            part,
                        )
                    )
                keep = torch.tensor([key in found for key in keys], dtype=torch.bool)
                digest.update(batch.positions[keep].contiguous().numpy().tobytes())
                retained += int(keep.sum())
            if retained != store.rows:
                raise AnalysisError(
                    "source_changed",
                    "Conditional-likelihood effective row counts changed during replay.",
                )
            result = _result(
                sample,
                terms=[store.design.terms[i] for i in store.kept],
                beta=beta,
                covariance=covariance,
                info=info,
                metrics={
                    "log_likelihood": ll,
                    "pseudo_r_squared": 1 - ll / null if null != 0 else None,
                    "aic": criteria["aic"],
                    "bic": criteria["bic"],
                    "n_groups": store.effective_groups,
                    "n_groups_dropped": store.dropped_groups,
                },
                notes=notes,
                predictions=[],
                tests=tests,
                solver="native_replayed_whole_group_conditional_likelihood",
                diagnostics={
                    "converged": True,
                    "iterations": run.iterations,
                    "largest_group": store.largest,
                    "largest_conditional_recursion_width": store.recursion_width,
                    "maximum_whole_group_block_rows": store.peak_rows,
                    "group_storage_disk_bytes": store.means.path.stat().st_size,
                    "group_storage": "owned SQLite complete informative groups; bounded exact conditional-kernel blocks",
                },
                extra={
                    "group": store.group,
                    "null_log_likelihood": null,
                    "group_sizes": {
                        "min": store.smallest,
                        "mean": store.mean_size,
                        "max": store.largest,
                    },
                    "multiple_positive_outcomes": store.multiple,
                    "observations_dropped": store.dropped_observations,
                    "weights_apply_to": "groups" if spec.weights else None,
                    "likelihood": "conditional on the number of positive outcomes in each group",
                },
                resource=store.resource.record(),
                use_t=False,
                provenance_extra={
                    "sample_position_count": retained,
                    "sample_positions_hash": digest.hexdigest(),
                    "sample_hash": hashlib.sha256(
                        (sample.baseline["data_hash"] + digest.hexdigest()).encode()
                    ).hexdigest(),
                    "hash_scope": "projected raw model columns and exact physical positions after missing/zero-weight and uninformative-group deletion",
                },
            )
            result.nobs = store.nobs
            result.dropped_rows = result.nobs_original - retained
            from .postest.group_state import capture_replayed_contract
            def keep_contract(frame):
                keys = encode_cluster_labels(frame[group])
                found = set()
                for start in range(0,len(keys),500):
                    part = keys[start:start+500]
                    found.update(row[0] for row in store.db.execute('SELECT key FROM groups WHERE key IN('+','.join('?' for _ in part)+')',part))
                return [key in found for key in keys]
            result.extra['group_state'] = capture_replayed_contract(result,sample,keep_contract)
            return result
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
    except (sqlite3.Error, OSError) as error:
        raise AnalysisError(
            "conditional_spill_failed",
            "Conditional likelihood replay needs writable owned temporary storage and enough free disk space.",
        ) from error
