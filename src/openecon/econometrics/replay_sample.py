"""Shared bounded, replayable sample/design contract for native estimators.

No full sample, design, weights, positions or group map is retained. Categories
are discovered globally before missing filtering; weights and rank are checked
globally. Every numerical pass verifies projected source hashes and positions.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
import hashlib
import json
import math
import sys
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.linalg import collinear_columns
from openecon.engines.execution import qr_factor
from openecon.engines.optimize import information_inverse
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.models import ModelSpec
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.streaming_design import (
    MAX_CATEGORY_BYTES, MAX_PARAMETERS, SAMPLE_LIMIT, _json_scalar, _label,
    encode_cluster_labels, numeric_values, row_hash_bytes,
)
from . import registry

WORKING_BYTES = 128 * 1024 * 1024


@dataclass
class ReplayDesign:
    name: str
    predictors: list[str]
    intercept: bool
    implicit_constant: bool = False
    prefix: str = ""
    terms: list[str] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)
    means: Tensor | None = None
    scales: Tensor | None = None
    anchor: Tensor | None = None
    magnitude: Tensor | None = None
    centered_mean: Tensor | None = None
    rms: Tensor | None = None
    transform: Tensor | None = None
    selector: Callable[[pd.DataFrame], Tensor] | None = None


@dataclass
class ReplayBatch:
    frame: pd.DataFrame
    weights: Tensor
    positions: Tensor
    designs: dict[str, Tensor]

    def numeric(self, name: str, *, allow_missing: bool = False) -> Tensor:
        if not allow_missing:
            return numeric_values(self.frame[name], name)
        values = self.frame[name].to_numpy(dtype="float64", na_value=math.nan)
        return torch.as_tensor(values.copy() if not values.flags.writeable else values,
                               dtype=torch.float64, device="cpu")


class ReplaySample:
    """Projected Dataset passes, global designs/weights, bounded reporting sample.

    Register all equation designs before ``prepare``. Implicit constants are
    used only for rank/centering (ordered cutpoints, normalized scale equations),
    and are removed from returned equation matrices. Designs always use the
    registry's treatment coding, including when no intercept is reported.
    ``allow_missing`` is reserved for estimator-specific censoring/selection
    domains, whose adapter must validate the remaining missing entries.
    """

    def __init__(self, spec: ModelSpec, source: Dataset, *, allow_missing: Sequence[str] = (),
                 outcome_categories: bool = False, ordered_outcome: bool = False,
                 batch_rows: int | None = None,
                 row_filter: Callable[[pd.DataFrame], Tensor] | None = None):
        if not isinstance(spec, ModelSpec) or not isinstance(source, Dataset):
            raise AnalysisError("invalid_dataset", "ReplaySample needs a validated spec and Dataset.")
        if batch_rows is not None and (type(batch_rows) is not int or not 1 <= batch_rows <= 65536):
            raise AnalysisError("invalid_batch_size", "batch_rows must be an integer from1 to65536.")
        self.spec, self.source = spec, source
        self.workspace_budget = workspace_budget_bytes()
        self.working_bytes = min(WORKING_BYTES, self.workspace_budget)
        self.category_budget = min(MAX_CATEGORY_BYTES, self.working_bytes//8)
        self.columns = registry.spec_columns(spec)
        absent = set(self.columns) - set(source.columns)
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(sorted(absent))}.")
        self.allow_missing = set(allow_missing)
        if self.allow_missing - set(self.columns):
            raise AnalysisError("invalid_spec", "Missing exemptions must be projected model columns.")
        self.outcome_categories, self.ordered_outcome = outcome_categories, ordered_outcome
        self.row_filter = row_filter
        self.labels: list[Any] = []
        self.categories: dict[str, dict[str, Any]] = {}
        self._levels: dict[str, dict[bytes, Any]] = {name: {} for name in spec.categorical}
        self._declarations: dict[str, tuple | None] = {}
        self._category_bytes = 0
        self.designs: dict[str, ReplayDesign] = {}
        # Category discovery needs projected raw columns, not an expanded
        # worst-case384-column numerical design. Plan these phases separately.
        self.rows = batch_rows or min(65536, max(1, self.working_bytes // max(512, 512*len(self.columns))))
        self.reporting_bytes = 64*len(self.columns)*SAMPLE_LIMIT
        self.discovery_plan = plan_workspace("replay projected discovery", {
            "projected_reader": 64*len(self.columns)*self.rows,
            "reporting_sample": self.reporting_bytes,
            "category_metadata": self.category_budget if spec.categorical or outcome_categories else 0,
        }, budget_bytes=self.working_bytes)
        self.reader_rows = self.rows
        self._specified_rows = batch_rows
        self.passes = self.maximum_rows = 0
        self.actual_numeric_peak_rows = 0
        self.baseline: dict[str, Any] | None = None
        self.original_count = self.nrows = self.nobs = 0
        self.weight_max = self.weight_mass = self.weight_mean = 1.0
        self.weight_multiplier = 1.0
        self.optimization_scale = 1.0
        self.sample = pd.DataFrame(columns=self.columns)
        self.sample_positions: list[int] = []
        self.notes: dict[str, Any] = {"omitted_terms": []}
        self._prepared = False

    def add_design(self, name: str, predictors: Sequence[str] | None = None, *,
                   intercept: bool | None = None, implicit_constant: bool = False,
                   prefix: str = "", selector: Callable[[pd.DataFrame], Tensor] | None = None) -> ReplayDesign:
        if self._prepared or name in self.designs:
            raise AnalysisError("invalid_spec", "Replay equation names must be unique and registered before preparation.")
        predictors = list(self.spec.predictors if predictors is None else predictors)
        if set(predictors) - set(self.columns):
            raise AnalysisError("invalid_spec", "Every replay regressor must be a projected model column.")
        result = ReplayDesign(name, predictors, self.spec.intercept if intercept is None else intercept,
                              implicit_constant, prefix)
        result.selector = selector
        self.designs[name] = result
        return result

    def _level(self, name: str, value: Any, collection: dict[bytes, Any]) -> None:
        safe = _json_scalar(value)
        key = _label(safe, budget=MAX_CATEGORY_BYTES)
        if key not in collection:
            self._category_bytes += sys.getsizeof(safe) + sys.getsizeof(key) + 128
            if self._category_bytes > self.category_budget:
                raise AnalysisError("category_budget", "Global category labels exceed their bounded metadata budget.")
            collection[key] = safe
        if len(collection) > MAX_PARAMETERS + 1:
            raise AnalysisError("model_too_wide", f"'{name}' exceeds the bounded category count.")

    def _raw(self, *, discovery: bool = False):
        self.source.assert_unchanged()
        raw_digest, positions_digest = hashlib.sha256(), hashlib.sha256()
        raw_digest.update(json.dumps({"columns": self.columns, "missing": self.spec.missing,
                                      "allowed_missing": sorted(self.allow_missing)},
                                     sort_keys=True).encode())
        original = used = frequencies = 0
        iterator = iter(self.source.iter_batches(self.columns, batch_rows=self.reader_rows))
        try:
            for raw in iterator:
                if not isinstance(raw, pd.DataFrame) or raw.columns.has_duplicates:
                    raise AnalysisError("invalid_dataset", "Dataset passes need unique-column DataFrame batches.")
                projected = raw.loc[:, self.columns]
                if sum(int(projected[name].memory_usage(index=False, deep=True)) for name in self.columns)*2 + self.reporting_bytes + self._category_bytes > self.working_bytes:
                    raise AnalysisError("batch_too_large", "Projected raw batch exceeds its snapshotted workspace budget; use a smaller Dataset reader batch.")
                for name in [*self._levels, *([self.spec.outcome] if self.outcome_categories else [])]:
                    values = projected[name]
                    if isinstance(values.dtype, pd.CategoricalDtype):
                        if len(values.cat.categories) > MAX_PARAMETERS + 1:
                            raise AnalysisError("model_too_wide", "Declared categorical dictionaries exceed the bounded budget.")
                        declaration = (tuple(_json_scalar(v) for v in values.cat.categories), bool(values.cat.ordered))
                        if name == self.spec.outcome and self.ordered_outcome and not values.cat.ordered:
                            raise AnalysisError("invalid_ordered_outcome", "An ordered outcome needs an ordered Categorical.")
                    else:
                        declaration = None
                        if name == self.spec.outcome and self.ordered_outcome and not (
                                pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype)):
                            raise AnalysisError("invalid_ordered_outcome", "Ordered outcomes must be numeric or an ordered Categorical.")
                    if name in self._declarations and self._declarations[name] != declaration:
                        raise AnalysisError("source_changed", "Categorical declarations changed between source batches/passes.")
                    self._declarations[name] = declaration
                    if discovery and name in self._levels:
                        for value in declaration[0] if declaration else values.dropna().unique():
                            self._level(name, value, self._levels[name])
                raw_digest.update(row_hash_bytes(projected))
                required = [name for name in self.columns if name not in self.allow_missing]
                missing = projected[required].isna().any(axis=1)
                if bool(missing.any()) and self.spec.missing == "raise":
                    raise AnalysisError("missing_values", "Model inputs contain missing observations; use missing='drop'.")
                local = torch.as_tensor((~missing).to_numpy().nonzero()[0], dtype=torch.int64, device="cpu")
                retained = projected.iloc[local.tolist()]
                if self.row_filter is not None and len(retained):
                    keep = self.row_filter(retained)
                    if not isinstance(keep, Tensor) or keep.dtype != torch.bool or keep.shape != (len(retained),):
                        raise AnalysisError("invalid_sample_filter", "A replay filter must return one boolean per retained row.")
                    local = local[keep]
                    retained = retained.iloc[keep.nonzero().flatten().tolist()]
                if (self.spec.weight_type == "fweight" and self.spec.weights
                        and getattr(retained[self.spec.weights].dtype, "kind", None) in {"i", "u"}
                        and not bool(retained[self.spec.weights].between(0, 2**53).all())):
                    raise AnalysisError("invalid_weights", "Frequency weights must be exact integers up to2^53.")
                weights = (numeric_values(retained[self.spec.weights], self.spec.weights)
                           if self.spec.weights else torch.ones(len(retained), dtype=torch.float64))
                if bool((weights < 0).any()):
                    raise AnalysisError("invalid_weights", "Likelihood weights must be nonnegative.")
                if self.spec.weight_type == "fweight":
                    if bool(((weights != weights.round()) | (weights > 2**53)).any()):
                        raise AnalysisError("invalid_weights", "Frequency weights must be exact nonnegative integers up to2^53.")
                    frequencies += int(float(weights.sum()))
                    if frequencies > 2**53:
                        raise AnalysisError("invalid_weights", "The total frequency exceeds exact float64 count precision.")
                positive = weights > 0
                retained, weights, local = retained.iloc[positive.nonzero().flatten().tolist()], weights[positive], local[positive]
                positions = local + original
                positions_digest.update(positions.numpy().astype("<i8", copy=False).tobytes())
                original += len(projected)
                used += len(retained)
                if len(retained):
                    self.maximum_rows = max(self.maximum_rows, len(retained))
                    yield retained, weights, positions
            record = {"original": original, "used": used, "frequency": frequencies,
                      "data_hash": raw_digest.hexdigest(), "positions_hash": positions_digest.hexdigest()}
            if self.baseline is None:
                self.baseline = record
            elif record != self.baseline:
                raise AnalysisError("source_changed", "The projected model sample changed between replay passes.")
            self.source.assert_unchanged()
            self.passes += 1
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def _encode(self, design: ReplayDesign, frame: pd.DataFrame) -> Tensor:
        pieces = [torch.ones((len(frame), 1), dtype=torch.float64)] if design.intercept or design.implicit_constant else []
        for name in design.predictors:
            if name in self.categories:
                levels = self.categories[name]["levels"]
                codes = torch.as_tensor(pd.Categorical(frame[name], categories=levels).codes.copy(), dtype=torch.int64)
                if bool((codes < 0).any()):
                    raise AnalysisError("source_changed", "A categorical value is absent from global discovery.")
                pieces.append((codes[:, None] == torch.arange(1, len(levels))).to(torch.float64))
            else:
                pieces.append(numeric_values(frame[name], name)[:, None])
        return torch.cat(pieces, 1) if pieces else torch.empty((len(frame), 0), dtype=torch.float64)

    def prepare(self) -> ReplaySample:
        if self._prepared:
            return self
        with torch.no_grad(), torch.device("cpu"):
            observed: dict[bytes, Any] = {}
            chunks = []
            for retained, _, positions in self._raw(discovery=True):
                take = min(SAMPLE_LIMIT-len(self.sample_positions), len(retained))
                if take:
                    chunks.append(retained.iloc[:take].copy())
                    self.sample_positions.extend(positions[:take].tolist())
                if self.outcome_categories:
                    for value in retained[self.spec.outcome].unique():
                        self._level(self.spec.outcome, value, observed)
            self.sample = pd.concat(chunks, ignore_index=True) if chunks else self.sample
            assert self.baseline is not None
            self.original_count, self.nrows = self.baseline["original"], self.baseline["used"]
            if not self.nrows:
                raise AnalysisError("empty_sample", "No positive-weight complete observations remain.")
            for name, collection in self._levels.items():
                declaration = self._declarations[name]
                try:
                    levels = list(declaration[0]) if declaration else sorted(collection.values())
                except TypeError as error:
                    raise AnalysisError("ambiguous_categories", "Mixed predictor labels need a declared Categorical.") from error
                if len(levels) < 2:
                    raise AnalysisError("constant_predictor", f"Categorical predictor '{name}' needs two levels.")
                self.categories[name] = {"levels": levels, "reference": levels[0],
                                         "ordered": declaration[1] if declaration else False,
                                         "coding": "treatment_drop_first",
                                         "levels_source": "original_model_inputs_before_missing_filter"}
            if self.outcome_categories:
                declaration = self._declarations[self.spec.outcome]
                if declaration:
                    self.labels = [label for label in declaration[0] if _label(label) in observed]
                else:
                    try:
                        self.labels = sorted(observed.values())
                    except TypeError:
                        self.labels = list(observed.values())
                if len(self.labels) < 2:
                    raise AnalysisError("constant_outcome", "At least two observed outcome categories are required.")
            moments = {}
            total_width = 0
            for name, design in self.designs.items():
                raw_terms = (["Intercept"] if design.intercept or design.implicit_constant else [])
                for column in design.predictors:
                    raw_terms.extend([f"{column}[{label}]" for label in self.categories[column]["levels"][1:]]
                                     if column in self.categories else [column])
                if len(raw_terms) > MAX_PARAMETERS or len(raw_terms) != len(set(raw_terms)):
                    raise AnalysisError("model_too_wide", "A replay design is too wide or has colliding term names.")
                design.terms = [design.prefix + term for term in raw_terms]
                total_width += len(raw_terms)
            if total_width > MAX_PARAMETERS:
                raise AnalysisError("model_too_wide", "Total likelihood equation width exceeds384 parameters.")
            reserve = {"rank_and_moment_factors": 640*(total_width+1)**2,
                       "projected_reader": 64*len(self.columns)*self.reader_rows,
                       "reporting_sample": self.reporting_bytes,
                       "category_metadata": self.category_budget if self._levels or self.outcome_categories else 0}
            per_row = max(8, 8*(32*total_width+4*len(self.columns)))
            plan_workspace("replay global design", {**reserve, "native_row_block": per_row},
                           budget_bytes=self.working_bytes)
            planned = min(65536, max(1, (self.working_bytes-sum(reserve.values()))//per_row))
            self.rows = min(self.reader_rows, planned)
            self.design_plan = plan_workspace("replay global design", {
                **reserve, "native_row_block": per_row*self.rows}, budget_bytes=self.working_bytes)
            moments = {name: _WeightedMoments(len(design.terms),
                       intercept=design.intercept or design.implicit_constant)
                       for name, design in self.designs.items()}
            mass = _WeightedMoments(1, intercept=True)
            for retained, raw_weights, _ in self._raw():
                for start in range(0, len(retained), self.rows):
                    frame = retained.iloc[start:start+self.rows]
                    weights = raw_weights[start:start+self.rows]
                    mass.add(torch.ones((len(frame), 1), dtype=torch.float64), weights, False)
                    for name, design in self.designs.items():
                        if len(design.terms):
                            selected = design.selector(frame) if design.selector else torch.ones(len(frame), dtype=torch.bool)
                            if bool(selected.any()):
                                moments[name].add(self._encode(design, frame)[selected], weights[selected], False)
            self.weight_max, self.weight_mass = mass.weight_max, mass.mass
            self.weight_mean = mass.weight_max * (mass.mass/self.nrows)
            self.nobs = self.baseline["frequency"] if self.spec.weight_type == "fweight" else self.nrows
            self.weight_multiplier = self.nrows/self.weight_mass if self.spec.weight_type == "aweight" else self.weight_max
            self.optimization_scale = (1/self.weight_mean if self.spec.weight_type in {"iweight", "pweight"}
                                       and not .5 <= self.weight_mean <= 2 else 1.)
            if not all(math.isfinite(value) and value > 0 for value in (self.weight_multiplier, self.optimization_scale)):
                raise AnalysisError("invalid_weights", "Global likelihood weights exceed float64 precision.")
            trees = {name: _TSQRTree() for name, design in self.designs.items() if design.terms}
            for name, design in self.designs.items():
                moment = moments[name]
                if not design.terms:
                    design.kept = []
                    design.transform = torch.empty((0, 0), dtype=torch.float64)
                    continue
                if not moment.n:
                    raise AnalysisError("empty_equation_sample", f"No observations identify equation'{name}'.")
                design.anchor, design.magnitude, design.centered_mean = moment.anchor, moment.magnitude, moment.mean
                if design.intercept or design.implicit_constant:
                    variance = moment.m2.value / moment.mass
                else:
                    variance = moment.m2.value / moment.mass + moment.mean.square()
                design.rms = variance.clamp_min(0).sqrt()
                if design.intercept or design.implicit_constant:
                    design.centered_mean[0], design.rms[0] = 0., 1.
                design.rms = torch.where(design.rms > 0, design.rms, torch.ones_like(design.rms))
                design.magnitude = torch.where(design.magnitude > 0, design.magnitude, torch.ones_like(design.magnitude))
                design.means = design.anchor + design.magnitude*design.centered_mean
                design.scales = design.magnitude*design.rms
            for batch in self.batches(preparing=True):
                for name, values in batch.designs.items():
                    if values.shape[1]:
                        selected = self.designs[name].selector(batch.frame) if self.designs[name].selector else torch.ones(len(values), dtype=torch.bool)
                        if not bool(selected.any()):
                            continue
                        factor = qr_factor(values[selected]*batch.weights[selected].sqrt()[:, None])
                        trees[name].add(factor)
            for name, design in self.designs.items():
                if not design.terms:
                    continue
                kept, omitted = collinear_columns(trees[name].finish())
                self.notes["omitted_terms"].extend(design.terms[index] for index in omitted)
                if design.implicit_constant:
                    kept = [index for index in kept if index != 0]
                design.kept = kept
                design.terms = [design.terms[index] for index in kept]
                if design.intercept:
                    transform = torch.diag(1/design.scales)
                    transform[0] = -design.means/design.scales
                    transform[0, 0] = 1.
                    design.transform = transform[kept][:, kept]
                else:
                    design.transform = torch.diag(1/design.scales[kept])
            self._prepared = True
        return self

    def plan_rows(self, operation: str, static_buffers: dict[str, int], bytes_per_row: int):
        """Reserve an adapter's buffers before constructing a native row objective.

        CSV reader partitions stay fixed; only numerical sub-blocks shrink.
        This is a live-buffer estimate, not an RSS or arbitrary total-row cap.
        """
        if not self._prepared or type(bytes_per_row) is not int or bytes_per_row < 1:
            raise AnalysisError("invalid_resource_budget", "Prepare replay data and supply a positive per-row buffer estimate.")
        reserve = dict(self.design_plan.buffers)
        reserve.pop("native_row_block", None)
        if set(reserve) & set(static_buffers):
            raise AnalysisError("invalid_resource_budget", "Adapter buffer names must not overwrite shared reservations.")
        reserve.update(static_buffers)
        plan_workspace(operation, {**reserve, "native_row_block": bytes_per_row},
                       budget_bytes=self.working_bytes)
        self.rows = min(self.rows, max(1, (self.working_bytes-sum(reserve.values()))//bytes_per_row))
        self.adapter_plan = plan_workspace(operation, {**reserve, "native_row_block": self.rows*bytes_per_row},
                                           budget_bytes=self.working_bytes)
        return self.adapter_plan

    def raw_design(self, batch: ReplayBatch, name: str) -> Tensor:
        """Encoded original-unit design for FE/IV adapters (bounded row block)."""
        design = self.designs[name]
        return self._encode(design, batch.frame)[:, design.kept]

    def batches(self, *, preparing: bool = False) -> Iterable[ReplayBatch]:
        if not preparing and not self._prepared:
            raise AnalysisError("invalid_spec", "Prepare the global replay sample before estimation.")
        with torch.no_grad(), torch.device("cpu"):
            for retained, raw_weights, positions in self._raw():
                for start in range(0, len(retained), self.rows):
                    frame = retained.iloc[start:start+self.rows]
                    weights = raw_weights[start:start+self.rows]/self.weight_max*self.weight_multiplier
                    matrices = {}
                    for name, design in self.designs.items():
                        x = self._encode(design, frame)
                        if x.shape[1]:
                            x = (x-design.anchor)/design.magnitude
                            if design.intercept or design.implicit_constant:
                                x = x-design.centered_mean
                            x = x/design.rms
                            if not preparing:
                                x = x[:, design.kept]
                        matrices[name] = x
                    self.actual_numeric_peak_rows = max(self.actual_numeric_peak_rows, len(frame))
                    yield ReplayBatch(frame, weights, positions[start:start+self.rows], matrices)

    def codes(self, frame: pd.DataFrame) -> Tensor:
        result = torch.as_tensor(pd.Categorical(frame[self.spec.outcome], categories=self.labels).codes.copy(), dtype=torch.int64, device="cpu")
        if bool((result < 0).any()):
            raise AnalysisError("source_changed", "An outcome label changed after global discovery.")
        return result

    def covariance(self, builder: Callable[[ReplayBatch], Any], theta: Tensor,
                   hessian: Tensor, *, canonical_units: Tensor | None = None) -> tuple[Tensor, dict[str, Any]]:
        kind, n = self.spec.covariance, self.nobs
        info: dict[str, Any] = {"covariance": kind}
        bread = None if kind == "opg" else information_inverse(-hessian)
        if kind == "nonrobust":
            return bread, {**info, "correction": "observed information"}
        if kind not in {"robust", "opg", "cluster"}:
            raise AnalysisError("unsupported_covariance", "This replay likelihood supports observed, OPG, robust and cluster covariance.")
        meat = _CompensatedSum((len(theta), len(theta)))
        columns = registry.cluster_columns(self.spec)
        if kind == "cluster" and not 1 <= len(columns) <= 2:
            raise AnalysisError("unsupported_cluster_dimensions", "Replay likelihood supports one or two cluster dimensions.")
        subsets = [part for width in range(1, len(columns)+1) for part in combinations(range(len(columns)), width)] if kind == "cluster" else []
        self.plan_rows("replay likelihood covariance", {"information_and_covariance": 128*len(theta)**2,
                       "cluster_cache_and_sqlite": len(subsets)*6*1024**2}, 128*(len(theta)+1))
        accumulators = []
        try:
            for _ in subsets:
                accumulators.append(ClusterAccumulator(len(theta)))
            for batch in self.batches():
                score = builder(batch).score_rows(theta)
                if score.shape != (len(batch.frame), len(theta)) or not bool(torch.isfinite(score).all()):
                    raise AnalysisError("numerical_failure", "Replay likelihood scores are not finite and aligned.")
                if accumulators:
                    keys = [encode_cluster_labels(batch.frame[column]) for column in columns]
                    for subset, accumulator in zip(subsets, accumulators, strict=True):
                        labels = keys[subset[0]] if len(subset) == 1 else [
                            b"".join(len(key).to_bytes(8, "little")+key for key in pair)
                            for pair in zip(*(keys[index] for index in subset), strict=True)]
                        accumulator.add(labels, score*batch.weights[:, None])
                else:
                    weight = batch.weights if self.spec.weight_type == "fweight" or kind == "opg" else batch.weights.square()
                    meat.add(score.T @ (score*weight[:, None]))
            if accumulators:
                combined, counts = _CompensatedSum((len(theta), len(theta))), []
                for subset, accumulator in zip(subsets, accumulators, strict=True):
                    matrix, groups = accumulator.finish()
                    combined.add(matrix*(1 if len(subset)%2 else -1))
                    if len(subset) == 1:
                        counts.append(groups)
                matrix, groups = combined.value, min(counts)
                psd_adjusted = False
                if len(columns) > 1:
                    from openecon.engines.covariance import nearest_psd
                    # Eigenvalue clipping is coordinate-dependent. Match the
                    # dense kernels' centred ORIGINAL-unit working parameters,
                    # then return to the standardized replay coordinates.
                    units = torch.ones(len(theta), dtype=torch.float64) if canonical_units is None else canonical_units
                    canonical_meat = matrix/units[:, None]/units[None, :]
                    canonical_meat, psd_adjusted = nearest_psd(canonical_meat)
                    matrix = canonical_meat*units[:, None]*units[None, :]
                factor = groups/(groups-1)
                info.update(accumulators[0].diagnostics)
                info.update({"correction": "cluster sandwich: G/(G-1)" if len(columns) == 1 else "multiway cluster sandwich (inclusion-exclusion): G_min/(G_min-1)",
                             "cluster_count": groups, "cluster_counts": counts,
                             "cluster_columns": columns, "cluster_column": columns[0],
                             "cluster_df": groups-1, "small_sample_correction": factor,
                             "psd_adjusted": psd_adjusted})
            else:
                matrix = meat.value
                factor = n/(n-1) if kind == "robust" else 1.
                info.update({"correction": "outer product of gradients" if kind == "opg" else "Huber-White sandwich: N/(N-1)",
                             "small_sample_correction": factor})
            covariance = information_inverse(matrix) if kind == "opg" else bread @ (matrix*factor) @ bread
            return (covariance+covariance.T)/2, info
        finally:
            for accumulator in accumulators:
                accumulator.close()

    def provenance(self) -> dict[str, Any]:
        return {"data_hash": self.baseline["data_hash"], "sample_hash": hashlib.sha256((self.baseline["data_hash"]+self.baseline["positions_hash"]).encode()).hexdigest(),
                "sample_positions_hash": self.baseline["positions_hash"],
                "hash_scope": "projected raw model columns plus missing policy; exact retained physical positions",
                "sample_positions_omitted": True, "sample_position_count": self.nrows,
                "prediction_sample": "first400 retained observations", "categorical_encoding": self.categories,
                "omitted_terms": list(dict.fromkeys(self.notes["omitted_terms"])),
                "streaming": {"passes": self.passes, "batch_rows": self.rows,
                              "reader_batch_rows": self.reader_rows, "maximum_batch_rows": self.actual_numeric_peak_rows,
                              "actual_numeric_peak_rows": self.actual_numeric_peak_rows,
                              "working_memory_budget_bytes": self.working_bytes,
                              "configured_workspace_budget_bytes": self.workspace_budget,
                              "resource_plan": getattr(self, "adapter_plan", self.design_plan).record(),
                              "dense_observation_matrix": False, "retained_reporting_rows": len(self.sample),
                              "source": {key: value for key, value in self.source.provenance.items() if key not in {"path"}}}}
