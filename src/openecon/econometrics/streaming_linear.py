"""Native replay least squares with affine constraints or one absorbed effect.

Only observation blocks, TSQR factors and bounded score/mean caches are resident.
Group means and cluster scores spill to owned SQLite storage.  This module does
not collect a Dataset and does not promise replay support for other estimators.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
import math
import os
import sqlite3
import tempfile
from types import SimpleNamespace
from typing import Any, Callable

import torch
import pandas as pd
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.contracts import KernelError
from openecon.engines.covariance import nearest_psd
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_hac import HACAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import plan_workspace, tensor_bytes
from openecon.streaming_design import encode_cluster_labels, numeric_values

from . import registry
from .core import build_result, wald_test
from .linear.cnsreg import _FIXED_ROW, constraint_matrix, reduce_constraints
from .linear.common import check_fit, check_pweights, classical_f_test, model_test, robust_wald_test
from .replay_sample import ReplayBatch, ReplaySample


SUPPORTED = frozenset({"cnsreg", "areg", "xtreg", "ivregress"})


def _weights(sample: ReplaySample, batch: ReplayBatch) -> Tensor:
    # Linear a/p weights sum to physical N; likelihood pweights stay raw in
    # ReplaySample, hence this explicit estimator-specific conversion.
    return batch.weights / sample.weight_mean if sample.spec.weight_type == "pweight" else batch.weights


def _finite(*values: Tensor) -> None:
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise AnalysisError("numerical_failure", "Replay linear arithmetic exceeds finite float64 precision; rescale the model inputs.")


def _factor(factory: Callable, width: int, *, require_more: bool = True) -> tuple[Tensor, dict[str, Any]]:
    tree = _TSQRTree()
    for x, y, weights in factory():
        block = torch.cat((x, y[:, None]), 1) * weights.sqrt()[:, None]
        _finite(block)
        _, factor = torch.linalg.qr(block, mode="r")
        tree.add(factor)
    factor = tree.finish()
    if require_more and factor.shape[0] <= width:
        raise AnalysisError("insufficient_observations", "Replay regression needs more distinct observations than free parameters.")
    _finite(factor)
    return factor, {"tsqr_reduction_depth": tree.depth, "tsqr_peak_factors": tree.peak_factors}


def _solve(factor: Tensor, width: int) -> tuple[Tensor, Tensor, float]:
    # The factor is small; unit column scaling keeps the solve out of normal
    # equations while preserving a singular-design refusal.
    x, y = factor[:, :width], factor[:, width]
    scales = torch.linalg.vector_norm(x, dim=0)
    if bool((scales <= 0).any()):
        raise AnalysisError("singular_design", "A replay free design column has zero norm.")
    q, r = torch.linalg.qr(x / scales, mode="reduced")
    values = torch.linalg.svdvals(r)
    if values[-1] <= torch.finfo(torch.float64).eps * values[0] * max(1, width):
        raise AnalysisError("singular_design", "The replay free design is numerically singular; rescale or remove collinear terms.")
    params = torch.linalg.solve_triangular(r, (q.T @ y)[:, None], upper=True).flatten() / scales
    inverse = torch.linalg.solve_triangular(r, torch.eye(width, dtype=torch.float64), upper=True) / scales[:, None]
    bread = inverse @ inverse.T
    _finite(params, bread)
    return params, bread, float(values[0] / values[-1])


class _Notes:
    def __init__(self, spec: ModelSpec, warnings: list[str] | None = None):
        self.spec = spec
        self.warnings = warnings or []

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def option(self, name: str):
        if name in self.spec.options:
            return self.spec.options[name]
        option = registry.get(self.spec.estimator).option(name)
        return None if option is None else option.default


def _covariance(sample: ReplaySample, factory: Callable, beta: Tensor, bread: Tensor,
                *, df: float, k: int, notes: _Notes,
                score_basis: Tensor | None = None, kind: str | None = None,
                cluster_columns: list[str] | None = None, correction_df: float | None = None,
                nobs: int | None = None, collect_predictions: bool = True,
                small: bool = True, group_factor: bool = True) -> tuple[Tensor, dict, float, list[dict]]:
    kind = kind or ("HC1" if sample.spec.covariance == "robust" else sample.spec.covariance)
    if kind not in {"nonrobust", "HC1", "cluster", "hac", "driscoll_kraay"}:
        raise AnalysisError("unsupported_streaming_covariance", "Replay areg/cnsreg support nonrobust, HC1 and one/two-way cluster covariance.")
    n = sample.nobs if nobs is None else nobs
    correction_df = df if correction_df is None else correction_df
    info = {"covariance": kind, "df_inference": df}
    rss = _CompensatedSum(())
    meat = _CompensatedSum(bread.shape)
    predictions: list[dict] = []
    columns = registry.cluster_columns(sample.spec) if cluster_columns is None else cluster_columns
    ordered_plan = None
    ordered_budget = None
    if kind in {"hac", "driscoll_kraay"}:
        base = getattr(sample, "base", sample)
        lags = notes.option("lags")
        if kind == "hac" and (lags is None or sample.spec.panel is not None and sample.spec.time is None):
            raise AnalysisError("invalid_spec", "HAC requires lags and panel HAC requires a time column.")
        if lags is None:
            from openecon.engines.covariance import newey_west_lags
            # Distinct observed periods cannot outnumber retained physical rows.
            # This upper bound reserves default DK's complete possible lag state
            # before the first period cache/score database is allocated.
            lags = newey_west_lags(base.nrows)
        buffers = HACAccumulator.workspace_buffers(len(beta), int(lags), notes.option("kernel") or "bartlett")
        ordered_bytes = sum(buffers.values())
        previous = getattr(base, "adapter_plan", None)
        common = set(base.design_plan.record()["buffers"])
        reserve = ({name: size for name, size in previous.buffers
                    if name not in common and not name.startswith("ordered_covariance_")}
                   if previous is not None else {})
        width = sum(len(design.terms) for design in base.designs.values())
        reserve.update({"ordered_covariance_global_IV_geometry": 4096*(width+len(beta)+1)**2,
                        "ordered_covariance_engine_buffers": ordered_bytes,
                        "ordered_covariance_period_cache": 6*1024**2 if kind == "driscoll_kraay" else 0})
        ordered_plan = base.plan_rows("combined ordered linear covariance and replay buffers", reserve,
                                    256*(width+2*len(beta)+len(base.columns)+4))
        # Give the engine only its own reservation; source and factor buffers
        # remain charged separately in the combined plan above.
        ordered_budget = ordered_bytes
        info["combined_live_covariance_resource_plan"] = ordered_plan.record()
    with ExitStack() as stack:
        accumulators = []
        hac = None
        dk = None
        if kind == "driscoll_kraay":
            from .streaming_panel_options import DriscollKraayAccumulator
            dk = DriscollKraayAccumulator(len(beta), notes, _scratch_directory(), ordered_budget)
            stack.callback(dk.close)
        if kind == "hac":
            lags = notes.option("lags")
            hac = HACAccumulator(len(beta), int(lags), notes.option("kernel") or "bartlett", budget_bytes=ordered_budget)
            stack.callback(hac.close)
        if kind == "cluster":
            if not 1 <= len(columns) <= 2:
                raise AnalysisError("unsupported_cluster_dimensions", "Replay linear covariance supports one or two cluster columns.")
            for _ in range(2**len(columns)-1):
                acc = ClusterAccumulator(len(beta), scratch_directory=_scratch_directory())
                stack.callback(acc.close)
                accumulators.append(acc)
        for batch, x, y, observed, response_scale in factory():
            weights = _weights(sample, batch)
            resid = getattr(batch, "residual_override", None)
            if resid is None:
                resid = y - x @ beta
            _finite(resid)
            rss.add((weights * resid.square()).sum())
            if kind == "HC1":
                score_weight = weights.sqrt() if sample.spec.weight_type == "fweight" else weights
                scores = x * (resid*score_weight)[:, None]
                meat.add(scores.T @ scores)
            elif kind == "hac":
                score_weight = weights.sqrt() if sample.spec.weight_type == "fweight" else weights
                scores = x * (resid*score_weight)[:, None]
                hac.add(scores, batch.positions,
                        time=batch.frame[sample.spec.time] if sample.spec.time else None,
                        units=encode_cluster_labels(batch.frame[sample.spec.panel]) if sample.spec.panel else None)
            elif kind == "driscoll_kraay":
                dk.add(x*(resid*weights)[:, None], batch.frame[sample.spec.time])
            elif kind == "cluster":
                scores = x * (resid*weights)[:, None]
                labels = getattr(batch, "cluster_keys", None)
                if labels is None:
                    labels = [encode_cluster_labels(batch.frame[name]) for name in columns]
                for mask, acc in enumerate(accumulators, 1):
                    members = [labels[index] for index in range(len(labels)) if mask >> index & 1]
                    keys = (members[0] if len(members) == 1 else
                            [b"".join(len(key).to_bytes(8, "big")+key for key in cell)
                             for cell in zip(*members, strict=True)])
                    acc.add(keys, scores)
            take = min(400-len(predictions), len(y)) if collect_predictions else 0
            predicted = getattr(batch, "prediction_response", None)
            response_resid = (resid*response_scale if predicted is None else observed-predicted)
            for row, o, residual in zip(batch.positions[:take].tolist(), observed[:take].tolist(),
                                        response_resid[:take].tolist(), strict=True):
                predictions.append({"row": row, "observed": o, "fitted": o-residual, "residual": residual})
        ssr = float(rss.value)
        if kind == "nonrobust":
            covariance = bread*(ssr/(df if small else n))
            info["correction"] = "classical: SSR/(N-K)" if small else "classical: SSR/N"
        else:
            matrix = meat.value
            if kind == "HC1":
                factor = n/correction_df if small else 1.
                info.update({"correction": "HC1: N/(N-K)" if small else "HC1 without N/(N-K)", "small_sample_correction": factor})
            elif kind == "hac":
                matrix = hac.finish()
                factor = n/correction_df if small else 1.
                info.update({"correction": f"Newey-West HAC ({hac.kernel}, {hac.lags} lags)" + (": N/(N-K)" if small else ""),
                             "small_sample_correction": factor, "lags": hac.lags, "kernel": hac.kernel,
                             **hac.diagnostics})
            elif kind == "driscoll_kraay":
                matrix, dk_info = dk.finish()
                periods = dk_info["periods"]
                factor = periods/(periods-1)*(n-1)/correction_df
                info.update({**dk_info, "correction": "Driscoll-Kraay (xtscc): T/(T-1) * (N-1)/(N-K), K = slopes + constant",
                             "small_sample_correction": factor, "df_inference": periods-1,
                             "k_small_sample": n-correction_df})
            else:
                counts, diagnostics = [], []
                matrix = torch.zeros_like(bread)
                for mask, acc in enumerate(accumulators, 1):
                    block, groups = acc.finish()
                    if mask.bit_count() == 1:
                        counts.append(groups)
                    matrix += block if mask.bit_count() % 2 else -block
                    diagnostics.append(acc.diagnostics)
                groups = min(counts)
                factor = (groups/(groups-1) if group_factor else 1.)*((n-1)/correction_df if small else 1.)
                adjusted = False
                if len(columns) == 2:
                    # Eigenvalue clipping is NOT invariant to a change of
                    # score coordinates. Match the dense estimator's original
                    # score design, then map the adjusted meat back.
                    if score_basis is None:
                        matrix, adjusted = nearest_psd(matrix)
                    else:
                        original = score_basis.T@matrix@score_basis
                        original, adjusted = nearest_psd(original)
                        inverse = torch.linalg.inv(score_basis)
                        matrix = inverse.T@original@inverse
                    if adjusted:
                        notes.warn("The multiway cluster meat was not positive semidefinite; negative eigenvalues were replaced by zero.")
                correction = " * ".join(value for value, enabled in (("G/(G-1)", group_factor), ("(N-1)/(N-K)", small)) if enabled)
                info.update({"correction": "CR1: "+correction if correction else "cluster sandwich without finite-sample factors",
                             "small_sample_correction": factor, "cluster_count": groups,
                             "cluster_columns": columns, "cluster_column": columns[0],
                             "cluster_df": groups-1, "df_inference": groups-1,
                             "cluster_spill": diagnostics})
                if len(columns) == 2:
                    info.update({"cluster_counts": counts, "psd_adjusted": adjusted})
            covariance = bread @ (matrix*factor) @ bread
        covariance = (covariance+covariance.T)/2
        _finite(covariance)
        return covariance, info, ssr, predictions


def _scratch_directory() -> Path | None:
    value = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
    return Path(value) if value else None


def _result(sample: ReplaySample, *, terms, beta, covariance, info, metrics, notes,
            predictions, tests, solver, diagnostics, extra=None, resource=None,
            title=None, use_t=True, provenance_extra=None) -> ResultBundle:
    # Reuse the canonical finite-inference/result contract on the reporting
    # sample ONLY, then explicitly supply full count and hash provenance.  This
    # facade never presents a reporting sample as the fitted observation matrix.
    reporting = sample.sample
    frame = SimpleNamespace(spec=sample.spec, info=registry.get(sample.spec.estimator),
                            n=len(reporting), positions=sample.sample_positions,
                            original=reporting, sample=reporting, notes=sample.notes,
                            warnings=notes.warnings, _sorted_by=[], resource_plans=[resource] if resource else [],
                            numeric=lambda name: numeric_values(reporting[name], name))
    provenance = sample.provenance()
    provenance.update({"solver_diagnostics": diagnostics, "design_terms": list(terms),
                       "sample_position_base": 0, "sample_order": "input row order",
                       "prediction_limit": 400, "resource_plans": [resource] if resource else []})
    provenance.update(provenance_extra or {})
    result = build_result(frame, terms=terms, params=beta, covariance=covariance,
                          nobs=sample.nobs, df_inference=info["df_inference"],
                          df_resid=info["df_resid"], metrics=metrics, inference=info,
                          provenance=provenance, tests=tests, extra=extra,
                          categories=sample.categories, solver=solver, title=title, use_t=use_t)
    result.nobs_original = sample.original_count
    result.dropped_rows = sample.original_count-sample.nrows
    result.sample_positions = []
    result.predictions = predictions
    if sample.spec.estimator == 'mixed':
        from .postest.group_state import capture_replayed_contract
        result.extra['group_state'] = capture_replayed_contract(result, sample)
    return result


def _resource(sample: ReplaySample, width: int, *, groups=False) -> dict:
    return plan_workspace("native replay linear regression", {
        "row_blocks": tensor_bytes((sample.rows, width+4), itemsize=96),
        "TSQR_factors_and_covariance": tensor_bytes((width+1, width+1), itemsize=1024),
        "bounded_SQLite_caches": (24 if groups else 18)*1024*1024,
    }).record()


def _fit_cnsreg(spec: ModelSpec, source: Dataset, *, batch_rows=None) -> ResultBundle:
    notes = _Notes(spec)
    check_pweights(notes)
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean")
    sample.prepare()
    k, terms = len(design.terms), design.terms
    if sample.nrows <= len(design.kept)+len(sample.notes["omitted_terms"]):
        raise AnalysisError("insufficient_observations", "Constrained regression needs more distinct observations than design columns.")
    if not k:
        raise AnalysisError("empty_design", "All constrained regression design columns were omitted.")
    resource = _resource(sample, k)
    omitted = sample.notes["omitted_terms"]
    r, target = constraint_matrix(spec.options.get("constraints"), terms, omitted)
    r, target = reduce_constraints(notes, r, target)
    q = len(target)
    if q >= k:
        raise AnalysisError("invalid_constraint", "The constraints determine every coefficient; nothing is left to estimate.")
    transform = design.transform
    working_r = r @ transform
    row_scale = torch.linalg.vector_norm(working_r, dim=1)
    if bool((row_scale <= 0).any()):
        raise AnalysisError("invalid_constraint", "A constraint cannot be represented in standardized float64 units.")
    basis, triangle = torch.linalg.qr((working_r/row_scale[:, None]).T, mode="complete")
    particular = basis[:, :q] @ torch.linalg.solve_triangular(triangle[:q].T,
                        (target/row_scale)[:, None], upper=False).flatten()
    free = basis[:, q:]
    _finite(particular, free)
    raw_basis, _ = torch.linalg.qr(r.T, mode="complete")
    raw_free = raw_basis[:, q:]
    coefficient_basis = raw_free.T@transform@free
    score_basis = torch.linalg.inv(coefficient_basis)

    def blocks():
        for batch in sample.batches():
            x = batch.designs["mean"]
            yield x@free, batch.numeric(spec.outcome)-x@particular, _weights(sample, batch)

    factor, diagnostics = _factor(blocks, k-q)
    params, bread, condition = _solve(factor, k-q)
    df = sample.nobs-k+q
    if df <= 0:
        raise AnalysisError("insufficient_observations", "The constrained regression has no residual degrees of freedom.")

    def residual_blocks():
        for batch in sample.batches():
            x, y = batch.designs["mean"], batch.numeric(spec.outcome)
            yield batch, x@free, y-x@particular, y, 1.

    covariance, info, rss, predictions = _covariance(sample, residual_blocks, params, bread,
                                                  df=df, k=k-q, notes=notes, score_basis=score_basis)
    scale = _CompensatedSum(())
    for batch in sample.batches():
        scale.add((_weights(sample, batch)*batch.numeric(spec.outcome).square()).sum())
    check_fit(rss, None, df, float(scale.value))
    beta = transform @ (particular+free@params)
    covariance = transform @ free @ covariance @ free.T @ transform.T
    fixed = (torch.linalg.vector_norm(raw_free, dim=1) <= _FIXED_ROW*k).nonzero().flatten().tolist()
    reported = [index for index in range(k) if index not in fixed]
    if fixed:
        covariance[fixed, :], covariance[:, fixed] = 0., 0.
        notes.warn("Fixed by the constraints and not estimated: "+", ".join(terms[index] for index in fixed)+". These terms are reported in extra['constrained_terms'].")
    reported_terms = [terms[index] for index in reported]
    beta_reported, covariance_reported = beta[reported], covariance[reported][:, reported]
    slopes = [index for index, term in enumerate(reported_terms) if term != "Intercept"]
    rank = int(torch.linalg.matrix_rank(raw_free[[reported[index] for index in slopes]])) if slopes else 0
    test = robust_wald_test(notes, beta_reported, covariance_reported, slopes,
                           df_inference=info["df_inference"], label="Model F test (free parameters)", expected_rank=rank)
    info.update({"df_resid": df, "nobs": sample.nobs, "constraints": q,
                 "free_parameters": k-q, "degrees_of_freedom_convention": "cnsreg: N - K + q residual degrees of freedom"})
    diagnostics.update({"condition_number": condition, "rank": k-q,
                        "condition_number_basis": "unit-norm standardized free weighted TSQR design"})
    return _result(sample, terms=reported_terms, beta=beta_reported, covariance=covariance_reported,
                   info=info, metrics={"rmse": math.sqrt(rss/df), "df_model": test["df"],
                                       "df_resid": df, "df_constraints": q}, notes=notes,
                   predictions=predictions, tests={"model": test},
                   solver="constraint_reparameterization_native_torch_tsqr", diagnostics=diagnostics,
                   extra={"constrained_terms": {terms[index]: float(beta[index]) for index in fixed},
                          "constraint_matrix": {"terms": terms, "R": r.tolist(), "r": target.tolist()}}, resource=resource)


class _GroupMeans:
    """Weighted Chan group means, fixed RAM cache, O(G*K) owned disk storage."""

    cache_bytes = 4*1024*1024

    def __init__(self, width: int, cluster_dimensions: int, *, panel: bool = False):
        self.width, self.dimensions = width, cluster_dimensions
        self.panel = panel
        self.cache: OrderedDict[bytes, tuple[Tensor, tuple[bytes, ...], int]] = OrderedDict()
        self.accounted = self.peak_bytes = self.writes = 0
        self.connection = self.scratch = None
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-fe-", dir=_scratch_directory())
            self.path = Path(self.scratch.name)/"means.sqlite3"
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA cache_size=-2048")
            self.connection.execute("PRAGMA mmap_size=0")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("CREATE TABLE means(key BLOB PRIMARY KEY, value BLOB, cluster0 BLOB, cluster1 BLOB, nested INTEGER) WITHOUT ROWID")
            if panel:
                self.connection.execute("CREATE TABLE panelmeta(key BLOB PRIMARY KEY, rows INTEGER, mass REAL, low REAL, high REAL) WITHOUT ROWID")
                self.connection.execute("CREATE TABLE periods(key BLOB, period TEXT, PRIMARY KEY(key,period)) WITHOUT ROWID")
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as error:
            self.close()
            raise self.storage_error() from error

    @staticmethod
    def storage_error():
        return AnalysisError("fixed_effect_spill_failed", "Replay fixed effects need writable temporary storage and enough free disk space.")

    def _size(self, key, entry):
        return len(key)+8*(self.width+1)+sum(map(len, entry[1]))+512

    def _write(self, key, entry):
        value, first, mask = entry
        labels = [*first, *([None]*(2-len(first)))]
        self.connection.execute("INSERT OR REPLACE INTO means VALUES(?,?,?,?,?)",
                                (key, array("d", value.tolist()).tobytes(), *labels, mask))
        self.writes += 1

    def _get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        record = self.connection.execute("SELECT value,cluster0,cluster1,nested FROM means WHERE key=?", (key,)).fetchone()
        if record is None:
            return None
        if len(record[0]) != 8*(self.width+1):
            raise self.storage_error()
        value = torch.frombuffer(bytearray(record[0]), dtype=torch.float64)
        _finite(value)
        entry = value, tuple(record[1:1+self.dimensions]), record[3]
        self._put(key, entry)
        return entry

    def _put(self, key, entry):
        if key in self.cache:
            self.accounted -= self._size(key, self.cache.pop(key))
        size = self._size(key, entry)
        while self.cache and (self.accounted+size > self.cache_bytes or len(self.cache) >= 4096):
            old_key, old_entry = self.cache.popitem(last=False)
            self._write(old_key, old_entry)
            self.accounted -= self._size(old_key, old_entry)
        if size > self.cache_bytes:
            self._write(key, entry)
        else:
            self.cache[key] = entry
            self.accounted += size
            self.peak_bytes = max(self.peak_bytes, self.accounted)

    @staticmethod
    def _codes(keys):
        mapping: dict[bytes, int] = {}
        codes = []
        for key in keys:
            if key not in mapping:
                mapping[key] = len(mapping)
            codes.append(mapping[key])
        return list(mapping), torch.tensor(codes, dtype=torch.int64)

    def _load_block(self, keys):
        """One bounded group block; no per-group tensor or database operation."""
        mapping = {key: index for index, key in enumerate(keys)}
        values = torch.zeros((len(keys), self.width+1), dtype=torch.float64)
        labels = [None]*len(keys)
        masks = [(1 << self.dimensions)-1]*len(keys)
        for start in range(0, len(keys), 500):
            part = keys[start:start+500]
            query = "SELECT key,value,cluster0,cluster1,nested FROM means WHERE key IN ("+",".join("?" for _ in part)+")"
            records = self.connection.execute(query, part).fetchall()
            if not records:
                continue
            if any(len(record[1]) != 8*(self.width+1) for record in records):
                raise self.storage_error()
            block = torch.frombuffer(bytearray(b"".join(record[1] for record in records)), dtype=torch.float64).reshape(-1, self.width+1)
            indices = torch.tensor([mapping[record[0]] for record in records], dtype=torch.int64)
            values.index_copy_(0, indices, block)
            for index, record in zip(indices.tolist(), records, strict=True):
                labels[index] = tuple(record[2:2+self.dimensions])
                masks[index] = record[4]
        _finite(values)
        return values, labels, masks

    def add(self, keys, values: Tensor, weights: Tensor, clusters: list[list[bytes]], periods=None):
        unique, codes = self._codes(keys)
        # Anchored block means avoid cancellation between large group offsets.
        first_rows = torch.full((len(unique),), len(keys), dtype=torch.int64)
        first_rows.scatter_reduce_(0, codes, torch.arange(len(keys)), reduce="amin")
        anchors = values[first_rows]
        mass = torch.zeros(len(unique), dtype=torch.float64).index_add_(0, codes, weights)
        differences = torch.zeros_like(anchors).index_add_(0, codes, (values-anchors[codes])*weights[:, None])
        means = anchors+differences/mass[:, None]
        if self.panel:
            counts = torch.bincount(codes, minlength=len(unique))
            low = torch.full((len(unique),), math.inf, dtype=torch.float64).scatter_reduce_(0, codes, weights, "amin")
            high = torch.full((len(unique),), -math.inf, dtype=torch.float64).scatter_reduce_(0, codes, weights, "amax")
        first = [tuple(labels[int(row)] for labels in clusters) for row in first_rows]
        masks = [(1 << self.dimensions)-1]*len(unique)
        for row, group in enumerate(codes.tolist()):
            for dimension, labels in enumerate(clusters):
                if labels[row] != first[group][dimension]:
                    masks[group] &= ~(1 << dimension)
        try:
            self.connection.execute("BEGIN")
            old, original_labels, original_masks = self._load_block(unique)
            total = old[:, 0]+mass
            merged = torch.where((old[:, 0] > 0)[:, None], old[:, 1:]+(means-old[:, 1:])*(mass/total)[:, None], means)
            updated = torch.cat((total[:, None], merged), 1)
            _finite(updated)
            for index in range(len(unique)):
                if original_labels[index] is not None:
                    masks[index] &= original_masks[index]
                    for dimension in range(self.dimensions):
                        if original_labels[index][dimension] != first[index][dimension]:
                            masks[index] &= ~(1 << dimension)
                    first[index] = original_labels[index]
            packed = memoryview(updated.contiguous().numpy().tobytes())
            stride = 8*(self.width+1)
            self.connection.executemany("INSERT OR REPLACE INTO means VALUES(?,?,?,?,?)",
                ((key, packed[index*stride:(index+1)*stride],
                  first[index][0] if self.dimensions else None,
                  first[index][1] if self.dimensions == 2 else None, masks[index])
                 for index, key in enumerate(unique)))
            self.writes += len(unique)
            if self.panel:
                counts_list, mass_list = counts.tolist(), mass.tolist()
                low_list, high_list = low.tolist(), high.tolist()
                self.connection.executemany("INSERT INTO panelmeta VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET rows=rows+excluded.rows,mass=mass+excluded.mass,low=MIN(low,excluded.low),high=MAX(high,excluded.high)",
                    ((key, counts_list[index], mass_list[index], low_list[index], high_list[index])
                     for index, key in enumerate(unique)))
            if periods is not None:
                try:
                    self.connection.executemany("INSERT INTO periods VALUES(?,?)", zip(keys, periods, strict=True))
                except sqlite3.IntegrityError as error:
                    raise AnalysisError("repeated_time_values", "More than one retained observation has the same panel and time.") from error
            self.connection.execute("COMMIT")
        except sqlite3.Error as error:
            raise self.storage_error() from error

    def lookup(self, keys):
        unique, codes = self._codes(keys)
        try:
            values, _, _ = self._load_block(unique)
        except sqlite3.Error as error:
            raise self.storage_error() from error
        if bool((values[:, 0] <= 0).any()):
            raise AnalysisError("source_changed", "An absorbed level changed between replay passes.")
        return values[:, 1:][codes]

    def finish(self):
        try:
            for key, entry in self.cache.items():
                self._write(key, entry)
            self.cache.clear()
            self.accounted = 0
            self.groups = int(self.connection.execute("SELECT COUNT(*) FROM means").fetchone()[0])
            nested = False
            self.nested_flags = []
            for dimension in range(self.dimensions):
                missing = self.connection.execute("SELECT COUNT(*) FROM means WHERE (nested & ?) = 0", (1 << dimension,)).fetchone()[0]
                nested |= missing == 0
                self.nested_flags.append(missing == 0)
            self.nested = nested
            self.scratch_bytes = self.path.stat().st_size
        except (OSError, sqlite3.Error) as error:
            raise self.storage_error() from error

    def group_batches(self, rows: int):
        # Flush a bounded cache before a cursor reads the stable group table.
        for key, entry in self.cache.items():
            self._write(key, entry)
        self.cache.clear()
        self.accounted = 0
        try:
            cursor = self.connection.execute("SELECT m.key,m.value,m.cluster0,m.cluster1,p.rows,p.mass FROM means m JOIN panelmeta p USING(key)")
            while block := cursor.fetchmany(rows):
                means = torch.stack([torch.frombuffer(bytearray(record[1]), dtype=torch.float64)[1:] for record in block])
                sizes = torch.tensor([record[4] for record in block], dtype=torch.float64)
                mass = torch.tensor([record[5] for record in block], dtype=torch.float64)
                labels = [[record[2+dimension] for record in block] for dimension in range(self.dimensions)]
                yield means, sizes, mass, labels
        except sqlite3.Error as error:
            raise self.storage_error() from error

    def panel_structure(self, *, frequency=False, constant_weights=False):
        if constant_weights and self.connection.execute("SELECT 1 FROM panelmeta WHERE low!=high LIMIT 1").fetchone():
            raise AnalysisError("weights_vary_within_panel", "Weights must be constant within each panel for model='fe'.")
        column = "mass" if frequency else "rows"
        minimum, maximum, total = self.connection.execute(f"SELECT MIN({column}),MAX({column}),SUM({column}) FROM panelmeta").fetchone()
        return {"n_groups": self.groups, "t_min": float(minimum), "t_avg": total/self.groups, "t_max": float(maximum)}

    @property
    def diagnostics(self):
        return {"fixed_effect_aggregation": "weighted anchored Chan group means; SQLite spill",
                "fixed_effect_groups": self.groups, "fixed_effect_cache_limit_bytes": self.cache_bytes,
                "fixed_effect_peak_cache_accounted_bytes": self.peak_bytes,
                "fixed_effect_sqlite_cache_bytes": 2*1024*1024,
                "fixed_effect_scratch_bytes": self.scratch_bytes, "fixed_effect_spill_writes": self.writes,
                "fixed_effect_memory_complexity": "O(block_rows * design_columns + design_columns**2 + fixed cache)",
                "fixed_effect_disk_complexity": "O(groups * (design_columns + label_bytes))"}

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


class _CrossMoments:
    """Anchored weighted Chan second moments on bounded standardized paths."""

    def __init__(self, width=2):
        self.mass = 0.
        self.anchor = self.mean = None
        self.m2 = _CompensatedSum((width, width))

    def add(self, values: Tensor, weights: Tensor):
        if self.anchor is None:
            self.anchor, self.mean = values[0].clone(), torch.zeros(values.shape[1], dtype=torch.float64)
        shifted = values-self.anchor
        mass = float(weights.sum())
        mean = (shifted*weights[:, None]).sum(0)/mass
        centered = shifted-mean
        total = self.mass+mass
        delta = mean-self.mean
        self.m2.add(centered.T@(centered*weights[:, None])+torch.outer(delta, delta)*(self.mass*mass/total))
        self.mean += delta*(mass/total)
        self.mass = total
        _finite(self.m2.value, self.mean)

    def correlation(self, first=0, second=1):
        variance = self.m2.value.diagonal()
        if variance[first] <= 0 or variance[second] <= 0:
            return None
        return max(-1., min(1., float(self.m2.value[first, second]/variance[first].sqrt()/variance[second].sqrt())))

    def squared_correlation(self):
        result = self.correlation()
        return None if result is None else result*result


def _period_keys(series):
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return [str(value.value) for value in series]
    result = []
    for value in series.tolist():
        if isinstance(value, (int, bool)):
            integer = int(value)
        elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
            integer = int(value)
        else:
            raise AnalysisError("invalid_time", "Panel time must contain integer periods or datetimes.")
        if not -(1 << 63) <= integer < 1 << 63:
            raise AnalysisError("invalid_time", "Panel integer periods must be exactly representable signed int64 values.")
        result.append(str(integer))
    return result


def _fit_areg(spec: ModelSpec, source: Dataset, *, batch_rows=None, panel_model=None) -> ResultBundle:
    notes = _Notes(spec)
    check_pweights(notes)
    if panel_model:
        from .panel.xtreg import _validate
        _validate(notes, panel_model)
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean", intercept=True)
    sample.prepare()
    if not design.terms or design.terms[0] != "Intercept":
        raise AnalysisError("singular_design", "The absorbed regression needs a retained constant.")
    resource = _resource(sample, len(design.terms), groups=True)
    absorb = spec.panel if panel_model else spec.columns["absorb"]
    if not isinstance(absorb, str):
        raise AnalysisError("invalid_spec", "areg absorbs exactly one column.")
    columns = ([spec.panel] if panel_model == "fe" and spec.covariance == "robust" else
               registry.cluster_columns(spec) if spec.covariance == "cluster" else [])
    moments = _WeightedMoments(2, intercept=True)
    for batch in sample.batches():
        moments.add(torch.stack((torch.ones(len(batch.frame), dtype=torch.float64), batch.numeric(spec.outcome)), 1),
                    _weights(sample, batch), False)
    y_anchor = float(moments.anchor[1])
    y_magnitude = float(moments.magnitude[1])
    y_center = float(moments.mean[1])
    y_scale = y_magnitude*math.sqrt(float(moments.m2.value[1]/moments.mass))
    if not math.isfinite(y_scale) or y_scale <= 0:
        raise AnalysisError("constant_outcome", "The outcome has no finite representable variation.")
    y_mean = y_anchor+y_magnitude*y_center

    def centered_y(batch):
        return ((batch.numeric(spec.outcome)-y_anchor)/y_magnitude-y_center)*(y_magnitude/y_scale)

    with ExitStack() as stack:
        store = _GroupMeans(len(design.terms), len(columns), panel=bool(panel_model))
        stack.callback(store.close)
        for batch in sample.batches():
            keys = encode_cluster_labels(batch.frame[absorb])
            labels = [encode_cluster_labels(batch.frame[name]) for name in columns]
            values = torch.cat((centered_y(batch)[:, None], batch.designs["mean"][:, 1:]), 1)
            periods = _period_keys(batch.frame[spec.time]) if panel_model and spec.time else None
            store.add(keys, values, _weights(sample, batch), labels, periods)
        store.finish()
        groups = store.groups
        if groups < 2 and not panel_model:
            raise AnalysisError("insufficient_groups", f"The absorbed variable '{absorb}' has a single level; use regress instead.")
        if sample.nrows <= groups and panel_model != "be":
            raise AnalysisError("insufficient_observations", "The absorbed regression needs more observations than fixed-effect levels.")
        structure = None
        if panel_model:
            structure = store.panel_structure(frequency=spec.weight_type == "fweight", constant_weights=panel_model == "fe")
            if columns and not all(store.nested_flags):
                code = "cluster_varies_within_panel" if panel_model == "be" else "cluster_not_nested"
                raise AnalysisError(code, "Every declared cluster column must be constant within each panel for replay xtreg FE/BE.")
            if panel_model == "be":
                return _fit_between(spec, sample, design, store, notes, resource, structure,
                                    centered_y, y_mean, y_scale, columns)
        tree, pooled_tree = _TSQRTree(), _TSQRTree()
        within_ss = _CompensatedSum((len(design.terms)-1,))
        before_ss = _CompensatedSum((len(design.terms)-1,))
        tss, tss_within = _CompensatedSum(()), _CompensatedSum(())
        for batch in sample.batches():
            weights = _weights(sample, batch)
            raw = torch.cat((centered_y(batch)[:, None], batch.designs["mean"][:, 1:]), 1)
            within = raw-store.lookup(encode_cluster_labels(batch.frame[absorb]))
            before_ss.add((raw[:, 1:].square()*weights[:, None]).sum(0))
            within_ss.add((within[:, 1:].square()*weights[:, None]).sum(0))
            tss.add((raw[:, 0].square()*weights).sum())
            tss_within.add((within[:, 0].square()*weights).sum())
            block = torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), within[:, 1:], within[:, :1]), 1)
            _, factor = torch.linalg.qr(block*weights.sqrt()[:, None], mode="r")
            tree.add(factor)
            _, pooled = torch.linalg.qr(torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), raw[:, 1:], raw[:, :1]), 1)*weights.sqrt()[:, None], mode="r")
            pooled_tree.add(pooled)
        all_factor, pooled_factor = tree.finish(), pooled_tree.finish()
        absorbed = (within_ss.value <= 1e-13*before_ss.value).nonzero().flatten().tolist()
        remaining = [index for index in range(len(design.terms)-1) if index not in absorbed]
        candidate = all_factor[:, [0, *[index+1 for index in remaining]]]
        kept, collinear = collinear_columns(candidate)
        kept_slopes = [remaining[index-1] for index in kept if index]
        for index in absorbed:
            sample.notes["omitted_terms"].append(design.terms[index+1])
            notes.warn(f"Omitted {design.terms[index+1]}: collinearity with the absorbed fixed effects.")
        for index in collinear:
            sample.notes["omitted_terms"].append(design.terms[remaining[index-1]+1])
            notes.warn(f"Omitted {design.terms[remaining[index-1]+1]}: collinearity.")
        selected = [0, *[index+1 for index in kept_slopes]]
        factor = all_factor[:, [*selected, all_factor.shape[1]-1]]
        beta, bread, condition = _solve(factor, len(selected))
        k_slopes, k_total = len(kept_slopes), len(kept_slopes)+groups
        df = sample.nobs-k_total
        if df <= 0:
            raise AnalysisError("insufficient_observations", "The absorbed regression has no residual degrees of freedom.")
        outcome_level_scale = float(tss.value)*y_scale**2+sample.nobs*y_mean**2
        if panel_model == "fe" and float(tss_within.value)*y_scale**2 <= 1e-28*outcome_level_scale:
            raise AnalysisError("no_within_variation", "The outcome has no representable within-panel variation.")
        x_means, x_scales = design.means[design.kept][selected], design.scales[design.kept][selected]
        score_basis = torch.diag(x_scales)
        score_basis[0] = x_means
        score_basis[0, 0] = 1.

        def residual_blocks():
            for batch in sample.batches():
                raw = torch.cat((centered_y(batch)[:, None], batch.designs["mean"][:, 1:]), 1)
                within = raw-store.lookup(encode_cluster_labels(batch.frame[absorb]))
                x = torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), within[:, [index+1 for index in kept_slopes]]), 1)
                if panel_model == "fe":
                    batch.prediction_response = y_mean+y_scale*(beta[0]+raw[:, [index+1 for index in kept_slopes]]@beta[1:])
                yield batch, x, within[:, 0], batch.numeric(spec.outcome), y_scale

        covariance, info, rss_scaled, predictions = _covariance(sample, residual_blocks, beta, bread,
                                                               df=df, k=k_total, notes=notes,
                                                               score_basis=score_basis,
                                                               kind="cluster" if panel_model == "fe" and columns else None,
                                                               cluster_columns=columns,
                                                               correction_df=sample.nobs-len(selected) if panel_model else None)
        rss, total, total_within = rss_scaled*y_scale**2, float(tss.value)*y_scale**2, float(tss_within.value)*y_scale**2
        # The canonical perfect-fit threshold uses the original outcome level.
        uncentered = total+sample.nobs*y_mean**2
        if panel_model == "fe" and rss <= 1e-28*uncentered:
            raise AnalysisError("no_within_variation", "The regressors reproduce the within-panel outcome exactly; standard errors are undefined.")
        check_fit(rss, total, df, uncentered, absorbed=True)
        transform = torch.diag(y_scale/x_scales)
        transform[0] = -y_scale*x_means/x_scales
        transform[0, 0] = y_scale
        raw_beta = transform@beta
        raw_beta[0] += y_mean
        raw_covariance = transform@covariance@transform.T
        terms = [design.terms[index] for index in selected]
        if panel_model == "fe":
            return _finish_fixed_effects(spec, sample, design, selected, store, notes, resource,
                                        structure, beta, covariance, raw_beta, raw_covariance,
                                        info, rss, df, y_mean, y_scale, centered_y,
                                        predictions, condition, pooled_factor, bread)
        r_squared = 1-rss/total
        metrics = {"r_squared": r_squared, "adjusted_r_squared": 1-(1-r_squared)*(sample.nobs-1)/df,
                   "r_squared_within": 1-rss/total_within if total_within > 0 else None,
                   "rmse": math.sqrt(rss/df), "df_model": k_slopes, "df_resid": df,
                   "df_absorbed": groups-1, "n_groups": groups}
        tests = {"model": model_test(notes, raw_beta, raw_covariance, list(range(1, len(terms))),
                            df_inference=info["df_inference"], classical=(total_within-rss, k_slopes, rss, df),
                            label="Model F test (slopes)")}
        if spec.covariance == "nonrobust":
            pooled_x = pooled_factor[:, selected]
            pooled_y = pooled_factor[:, -1]
            q_pooled, _ = torch.linalg.qr(pooled_x, mode="reduced")
            pooled_rss = float((pooled_y-q_pooled@(q_pooled.T@pooled_y)).square().sum())*y_scale**2
            tests["absorbed"] = classical_f_test(pooled_rss-rss, groups-1, rss, df,
                                                label=f"F test of absorbed {absorb} effects")
        info.update({"df_resid": df, "nobs": sample.nobs, "k_total": k_total,
                     "absorbed_degrees_of_freedom": groups-1,
                     "degrees_of_freedom_convention": "areg: absorbed indicators count as parameters, also when nested within clusters"})
        diagnostics = {"condition_number": condition, "rank": len(selected), "absorbed_levels": groups,
                       "tsqr_reduction_depth": tree.depth, "tsqr_peak_factors": tree.peak_factors,
                       "condition_number_basis": "unit-norm standardized within weighted TSQR design", **store.diagnostics}
        result = _result(sample, terms=terms, beta=raw_beta, covariance=raw_covariance, info=info,
                       metrics=metrics, notes=notes, predictions=predictions, tests=tests,
                       solver="within_transform_native_torch_tsqr_sqlite", diagnostics=diagnostics,
                       extra={"absorbed": [{"column": absorb, "levels": groups, "redundant": 1, "nested": store.nested}],
                              "constant_degree_of_freedom_added": True}, resource=resource)
        def fitted_blocks():
            for batch in sample.batches():
                raw = torch.cat((centered_y(batch)[:,None],batch.designs['mean'][:,1:]),1)
                means = store.lookup(encode_cluster_labels(batch.frame[absorb]))
                within = raw-means
                x = torch.cat((torch.ones((len(raw),1),dtype=torch.float64),within[:,[i+1 for i in kept_slopes]]),1)
                effect = y_scale*(means[:,0]-beta[0]-means[:,[i+1 for i in kept_slopes]]@beta[1:])
                yield batch.frame,batch.numeric(spec.outcome)-y_scale*(within[:,0]-x@beta),effect
        from .postest.group_state import capture_fixed_replay
        result.extra['group_state'] = capture_fixed_replay(result,fitted_blocks)
        return result


def _panel_row_moments(spec, sample, selected, store, beta, centered_y):
    within, overall, effects = _CrossMoments(), _CrossMoments(), _CrossMoments()
    for batch in sample.batches():
        x = batch.designs["mean"][:, selected[1:]]
        y = centered_y(batch)
        means = store.lookup(encode_cluster_labels(batch.frame[spec.panel]))
        xb, xb_mean = x@beta[1:], means[:, selected[1:]]@beta[1:]
        weights = _weights(sample, batch)
        within.add(torch.stack((xb-xb_mean, y-means[:, 0]), 1), weights)
        overall.add(torch.stack((xb, y), 1), weights)
        effects.add(torch.stack((means[:, 0]-xb_mean, xb), 1), weights)
    return within, overall, effects


def _finish_fixed_effects(spec, sample, design, selected, store, notes, resource,
                          structure, beta, covariance, raw_beta, raw_covariance,
                          info, rss, df, y_mean, y_scale, centered_y,
                          predictions, condition, pooled_factor, bread):
    from .panel.xtreg import _check_constant_variance
    between = _CrossMoments(3)
    for means, _, _, _ in store.group_batches(sample.rows):
        xb_mean = means[:, selected[1:]]@beta[1:]
        between.add(torch.stack((xb_mean, means[:, 0], means[:, 0]-xb_mean), 1),
                    torch.ones(len(means), dtype=torch.float64))
    within, overall, effects = _panel_row_moments(spec, sample, selected, store, beta, centered_y)
    sigma_e = math.sqrt(rss/df)
    sigma_u = math.sqrt(max(0., float(between.m2.value[2, 2])/(store.groups-1)))*y_scale if store.groups > 1 else None
    if spec.covariance in {"robust", "cluster"}:
        means = design.means[design.kept][selected]
        scales = design.scales[design.kept][selected]
        row = torch.cat((torch.ones(1, dtype=torch.float64), -means[1:]/scales[1:]))
        classical = rss/df*float(row@bread@row)
        _check_constant_variance(notes, float(raw_covariance[0, 0]), classical)
    metrics = {"r_squared_within": within.squared_correlation(),
               "r_squared_between": (lambda r: None if r is None else r*r)(between.correlation()),
               "r_squared_overall": overall.squared_correlation(), "sigma_u": sigma_u,
               "sigma_e": sigma_e, "rho": None if sigma_u is None else sigma_u**2/(sigma_u**2+sigma_e**2),
               "corr_u_xb": effects.correlation(), "rmse": sigma_e, "df_resid": df, **structure}
    tests = {"model": wald_test(raw_beta, raw_covariance, range(1, len(selected)),
                                df_resid=info["df_inference"], label="F test of the slopes")}
    if spec.covariance == "nonrobust" and store.groups > 1:
        x, y = pooled_factor[:, selected], pooled_factor[:, -1]
        q, _ = torch.linalg.qr(x, mode="reduced")
        pooled_rss = float((y-q@(q.T@y)).square().sum())*y_scale**2
        tests["fixed_effects"] = classical_f_test(pooled_rss-rss, store.groups-1, rss, df,
                                                   label="F test that all u_i = 0")
    info.update({"df_resid": df, "k_small_sample": len(selected)})
    if spec.covariance == "robust":
        info["covariance"] = "robust"
        info["correction"] = "vce(robust) = vce(cluster panel); "+info["correction"]
        if store.groups < 30:
            notes.warn(f"Only {store.groups} panels; cluster-robust inference on the panel variable may be unreliable.")
    result = _result(sample, terms=[design.terms[index] for index in selected], beta=raw_beta,
                   covariance=raw_covariance, info=info, metrics=metrics, notes=notes,
                   predictions=predictions, tests=tests, solver="qr_within_native_replay_sqlite",
                   diagnostics={"condition_number": condition, "rank": len(selected), **store.diagnostics},
                   extra={"model": "fe", "ssr": rss, "absorbed_effects": store.groups,
                          "k_small_sample": len(selected)}, resource=resource,
                   title="Fixed-effects (within) regression", provenance_extra={"model": "fe"})
    def fitted_blocks():
        for batch in sample.batches():
            raw = torch.cat((centered_y(batch)[:,None],batch.designs['mean'][:,1:]),1)
            means = store.lookup(encode_cluster_labels(batch.frame[spec.panel]))
            within = raw-means
            x = torch.cat((torch.ones((len(raw),1),dtype=torch.float64),within[:,selected[1:]]),1)
            effect = y_scale*(means[:,0]-beta[0]-means[:,selected[1:]]@beta[1:])
            yield batch.frame,batch.numeric(spec.outcome)-y_scale*(within[:,0]-x@beta),effect
    from .postest.group_state import capture_fixed_replay
    result.extra['group_state'] = capture_fixed_replay(result,fitted_blocks)
    return result


def _fit_between(spec, sample, design, store, notes, resource, structure,
                  centered_y, y_mean, y_scale, columns):
    weighted = bool(spec.options.get("wls", False))
    outcome = _CrossMoments(1)

    def blocks():
        for means, sizes, _, _ in store.group_batches(sample.rows):
            weights = sizes if weighted else torch.ones(len(means), dtype=torch.float64)
            outcome.add(means[:, :1], weights)
            yield torch.cat((torch.ones((len(means), 1), dtype=torch.float64), means[:, 1:]), 1), means[:, 0], weights

    factor, diagnostics = _factor(blocks, len(design.terms), require_more=False)
    centered = factor[:, :-1].clone()
    centered[:, 1:] -= centered[:, :1]*((centered[:, 0]@centered[:, 1:])/(centered[:, 0]@centered[:, 0]))
    selected, omitted = collinear_columns(centered)
    for index in omitted:
        sample.notes["omitted_terms"].append(design.terms[index])
        notes.warn(f"Omitted {design.terms[index]}: collinearity in the panel-mean regression.")
    factor = factor[:, [*selected, factor.shape[1]-1]]
    beta, bread, condition = _solve(factor, len(selected))
    df = store.groups-len(selected)
    if df <= 0:
        raise AnalysisError("insufficient_observations", "The between regression needs more panels than estimated parameters.")
    x_means, x_scales = design.means[design.kept][selected], design.scales[design.kept][selected]
    score_basis = torch.diag(x_scales)
    score_basis[0], score_basis[0, 0] = x_means, 1.

    def residual_blocks():
        for means, sizes, _, labels in store.group_batches(sample.rows):
            weights = sizes if weighted else torch.ones(len(means), dtype=torch.float64)
            x = torch.cat((torch.ones((len(means), 1), dtype=torch.float64), means[:, 1:]), 1)[:, selected]
            batch = ReplayBatch(pd.DataFrame(index=range(len(means))), weights,
                                torch.arange(len(means)), {})
            batch.cluster_keys = labels
            yield batch, x, means[:, 0], y_mean+y_scale*means[:, 0], y_scale

    covariance, info, ssr, _ = _covariance(sample, residual_blocks, beta, bread,
                                df=df, k=len(selected), notes=notes, score_basis=score_basis,
                                kind="HC1" if spec.covariance == "robust" else None,
                                cluster_columns=columns, nobs=store.groups, collect_predictions=False)
    tss = float(outcome.m2.value[0, 0])*y_scale**2
    rss = ssr*y_scale**2
    mean_y = y_mean+y_scale*float(outcome.anchor[0]+outcome.mean[0])
    scale = tss+outcome.mass*mean_y**2
    from .panel.common import require_residual_variation
    require_residual_variation(rss, tss, scale, spec.outcome, "the panel means")
    transform = torch.diag(y_scale/x_scales)
    transform[0], transform[0, 0] = -y_scale*x_means/x_scales, y_scale
    raw_beta, raw_covariance = transform@beta, transform@covariance@transform.T
    raw_beta[0] += y_mean
    within, overall, _ = _panel_row_moments(spec, sample, selected, store, beta, centered_y)
    predictions = []
    for batch in sample.batches():
        take = min(400-len(predictions), len(batch.frame))
        fitted = y_mean+y_scale*(batch.designs["mean"][:, selected]@beta)
        for row, observed, predicted in zip(batch.positions[:take].tolist(), batch.numeric(spec.outcome)[:take].tolist(), fitted[:take].tolist(), strict=True):
            predictions.append({"row": row, "observed": observed, "fitted": predicted, "residual": observed-predicted})
    metrics = {"r_squared_between": 1-rss/tss, "r_squared_within": within.squared_correlation(),
               "r_squared_overall": overall.squared_correlation(), "rmse": math.sqrt(rss/df),
               "df_resid": df, **structure}
    info.update({"df_resid": df, "k_small_sample": len(selected)})
    if spec.covariance == "robust":
        info["covariance"] = "robust"
    tests = {"model": wald_test(raw_beta, raw_covariance, range(1, len(selected)),
                                df_resid=info["df_inference"], label="F test of the slopes")}
    return _result(sample, terms=[design.terms[index] for index in selected], beta=raw_beta,
                   covariance=raw_covariance, info=info, metrics=metrics, notes=notes,
                   predictions=predictions, tests=tests, solver="qr_between_native_replay_sqlite",
                   diagnostics={**diagnostics, "condition_number": condition, **store.diagnostics},
                   extra={"model": "be", "ssr": rss, "weighted_by_panel_length": weighted},
                   resource=resource, title="Between-effects regression", provenance_extra={"model": "be"})


def fit_streaming_linear(spec: ModelSpec, source: Dataset, *, batch_rows: int | None = None) -> ResultBundle:
    """Dataset delegate; unsupported families remain explicit capability errors."""
    if spec.estimator not in SUPPORTED:
        raise AnalysisError("unsupported_streaming_estimator", "Native replay linear support currently covers areg, cnsreg, xtreg FE/BE and ivregress 2SLS.")
    try:
        with torch.no_grad(), torch.device("cpu"):
            if spec.estimator == "cnsreg":
                return _fit_cnsreg(spec, source, batch_rows=batch_rows)
            if spec.estimator == "ivregress":
                from .streaming_iv import fit_iv_replay
                return fit_iv_replay(spec, source, batch_rows=batch_rows)
            if spec.estimator == "xtreg":
                model = spec.options.get("model", "fe")
                if model == "cre":
                    raise AnalysisError("streaming_unsupported", "CRE requires guarded in-memory actual-sample Mundlak means; Dataset is not collected.")
                if model == "re":
                    from .streaming_re import fit_streaming_re
                    return fit_streaming_re(spec, source, batch_rows=batch_rows)
                if model in {"pooled", "fd", "mle"}:
                    from .streaming_panel_options import fit_streaming_panel_options
                    return fit_streaming_panel_options(spec, source, batch_rows=batch_rows)
                if model not in {"fe", "be"}:
                    raise AnalysisError("invalid_spec", "Unknown panel regression model.")
                return _fit_areg(spec, source, batch_rows=batch_rows, panel_model=model)
            return _fit_areg(spec, source, batch_rows=batch_rows)
    except KernelError as error:
        raise AnalysisError(error.code, str(error)) from error
