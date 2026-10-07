"""Saved-model evaluation on bounded projected replays, without a refit.

Observation predictions own a temporary Parquet file. Margins reduce effects
and parameter gradients globally, then apply the saved covariance once. MEM
uses the global weighted encoded design, never the average of batch MEMs.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from itertools import product
import os
from pathlib import Path
import tempfile

import pandas as pd
import torch

from openecon.analysis import _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.frame import as_frame
from openecon.resources import plan_workspace, workspace_budget_bytes


def _error(code, message):
    raise AnalysisError(code, message)


def _grids(model, at):
    from .prediction import _scalar
    at = {} if at is None else at
    if not isinstance(at, Mapping) or any(name not in model.intervention_variables() for name in at):
        _error("invalid_margins", "at must map original predictors to scalar values or finite grids.")
    grids = []
    count = 1
    for name, raw in at.items():
        points = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        if not points:
            _error("invalid_margins", "Evaluation grids must not be empty.")
        count *= len(points)
        if count > 1000:
            _error("margins_grid_limit", "Use at most 1000 evaluation-grid combinations.")
        for point in points:
            if name in model.categories:
                if point not in model.categories[name]:
                    _error("unknown_category", f"Column '{name}' has an unfitted at category.")
            else:
                _scalar(point, code="invalid_margins")
        grids.append((name, points))
    return [dict(zip((name for name, _ in grids), values, strict=True))
            for values in product(*(points for _, points in grids))]


class _Replay:
    def __init__(self, model, source, *, weights, batch_rows, state_bytes=0, kind="response"):
        self.model, self.source, self.weights = model, source, weights
        self.kind = kind
        self.columns = model.required(weights)
        missing = set(self.columns) - set(source.columns)
        if missing:
            _error("missing_columns", "Prediction data lack: " + ", ".join(sorted(missing)) + ".")
        # An intercept-only prediction still has one output for each input row.
        self.columns = self.columns or [source.columns[0]]
        k = len(model.state.terms)
        outcomes = len(model.choice.labels) if model.choice is not None else 1
        self.numeric_row_bytes = 8 * max(1, k) * max(24, 8 * outcomes)
        adapter = model.response_adapter
        self.adapter_planner = None if adapter is None else getattr(adapter, "workspace_row_bytes", None)
        if self.adapter_planner is not None and getattr(adapter, "truncation_column", None) is None:
            self.numeric_row_bytes += self.adapter_planner(kind=kind)
        per_row = 64 * len(self.columns) + self.numeric_row_bytes
        budget = self.budget = min(128 * 1024**2, workspace_budget_bytes())
        fixed = 32 * k * k + state_bytes
        self.fixed = fixed
        if batch_rows is not None and (type(batch_rows) is not int or not 1 <= batch_rows <= 65536):
            _error("invalid_batch_size", "Use batch_rows between 1 and 65,536.")
        self.rows = batch_rows or min(65536, max(1, (budget - fixed) // per_row))
        self.plan = plan_workspace("saved-model Dataset evaluation", {
            "parameter_covariance_and_validation": 32 * k * k,
            "global_reductions": state_bytes,
            "projected_rows_designs_and_gradients": self.rows * per_row,
        }, budget_bytes=budget)
        self.baseline, self.passes, self.maximum_rows = None, 0, 0
        self.maximum_reader_rows, self.adaptive_plan = 0, None

    def batches(self):
        from .prediction import _data, _check_categories
        digest = hashlib.sha256()
        rows = complete = positive = 0
        iterator = self.source.iter_batches(self.columns, batch_rows=self.rows)
        try:
            for raw in iterator:
                rows += len(raw)
                self.maximum_reader_rows = max(self.maximum_reader_rows, len(raw))
                hashes = pd.util.hash_pandas_object(raw, index=True, categorize=False)
                digest.update(hashes.to_numpy(dtype="<u8").tobytes())
                _, frame, _ = _data(self.model, raw, weights=self.weights)
                complete += len(frame)
                w = torch.ones(len(frame), dtype=torch.float64)
                if self.weights and self.model.result.spec.weights:
                    w = _numeric(frame[self.model.result.spec.weights], self.model.result.spec.weights)
                    if bool((w < 0).any()):
                        _error("invalid_weights", "Marginal-effect weights must be nonnegative.")
                    if self.model.result.spec.weight_type == "fweight" and bool((w != w.round()).any()):
                        _error("invalid_weights", "Frequency weights must be integers.")
                if self.weights:
                    keep = w > 0
                    frame = frame.iloc[keep.nonzero().flatten().tolist()].copy(deep=False)
                    w = w[keep]
                    _check_categories(self.model, frame)
                positive += len(frame)
                numerical_rows = len(raw)
                if self.adapter_planner is not None and getattr(self.model.response_adapter, "truncation_column", None) is not None:
                    per_row = self.numeric_row_bytes + self.adapter_planner(frame=frame, kind=self.kind)
                    reader_bytes = len(raw) * 64 * len(self.columns)
                    numerical_rows = min(len(raw), max(1, (self.budget - self.fixed - reader_bytes) // per_row))
                    plan = plan_workspace("count evaluation adaptive block", {
                        "parameter_covariance_and_global_reductions": self.fixed,
                        "retained_projected_reader": reader_bytes,
                        "numerical_design_and_tail_derivatives": numerical_rows * per_row,
                    }, budget_bytes=self.budget)
                    if self.adaptive_plan is None or plan.estimated_bytes > self.adaptive_plan.estimated_bytes:
                        self.adaptive_plan = plan
                for start in range(0, len(raw), numerical_rows):
                    part = raw.iloc[start:start + numerical_rows]
                    self.maximum_rows = max(self.maximum_rows, len(part))
                    if len(part) == len(raw):
                        yield raw, frame, w
                        continue
                    _, retained, _ = _data(self.model, part, weights=self.weights)
                    part_weights = torch.ones(len(retained), dtype=torch.float64)
                    if self.weights and self.model.result.spec.weights:
                        part_weights = _numeric(retained[self.model.result.spec.weights], self.model.result.spec.weights)
                    if self.weights:
                        keep = part_weights > 0
                        retained = retained.iloc[keep.nonzero().flatten().tolist()].copy(deep=False)
                        part_weights = part_weights[keep]
                    yield part, retained, part_weights
        finally:
            iterator.close()
        record = (rows, complete, positive, digest.hexdigest())
        if self.baseline is not None and record != self.baseline:
            _error("source_changed", "Evaluation rows or values changed between Dataset passes.")
        self.baseline = record
        self.passes += 1

    def metadata(self):
        return {"passes": self.passes, "batch_rows": self.rows,
                "maximum_batch_rows": self.maximum_rows, "maximum_reader_batch_rows": self.maximum_reader_rows,
                "source": self.source.provenance,
                "evaluation_digest": None if self.baseline is None else self.baseline[3],
                "resource_plan": self.plan.record(),
                "adaptive_resource_plan": None if self.adaptive_plan is None else self.adaptive_plan.record(),
                "full_source_collected": False}


class _Average:
    """Convex global mean with overflow-safe normalization of raw weights."""
    def __init__(self):
        self.scale = self.mass = 0.
        self.value = None

    def add(self, value, weights):
        local_scale = float(weights.max())
        scale = max(local_scale, self.scale)
        mass = float((weights / scale).sum())
        old_mass = self.mass * (self.scale / scale)
        total = old_mass + mass
        value = value.detach().clone()
        self.value = value if self.value is None else self.value * (old_mass / total) + value * (mass / total)
        self.scale, self.mass = scale, total
        if not bool(torch.isfinite(self.value).all()):
            _error("invalid_margins", "Global marginal effects or gradients exceed finite float64 range.")


def materialize_predictions(iterator, metadata):
    """Own bounded indexed Parquet output; cleanup on failure or object release."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .index_codec import IndexStorage
    scratch = tempfile.TemporaryDirectory(prefix="openecon-prediction-",
                                         dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None)
    path = Path(scratch.name) / "predictions.parquet"
    writer, rows, missing, columns, definition = None, 0, 0, None, None
    maximum_rows = 1
    response_metadata = {}
    index_storage = None
    try:
        for frame in iterator:
            rows += len(frame)
            maximum_rows = max(maximum_rows, len(frame))
            missing += len(frame.attrs.get("missing_row_positions", []))
            definition = frame.attrs.get("response_definition", definition)
            for name in ("outcome_labels", "outcome_columns", "outcome", "base_outcome"):
                if name in frame.attrs:
                    response_metadata[name] = frame.attrs[name]
            # Primitive indexes stay native and fast. Object labels use typed
            # scalar codes so a null-only or mixed-type block cannot change
            # the storage schema or silently merge integer/string identities.
            clean = pd.DataFrame(frame, copy=False)
            clean.attrs = {}
            if index_storage is None:
                index_storage = IndexStorage(frame.index, frame.columns)
            record = index_storage.append(pa.Table.from_pandas(clean, preserve_index=False), frame.index)
            if writer is None:
                columns = list(frame.columns)
                writer = pq.ParquetWriter(path, record.schema)
                path.chmod(0o600)
            writer.write_table(record)
        if writer is None:
            _error("empty_sample", "Prediction source has no rows.")
        writer.close()
        writer = None
        backing = scan(path)
        def factory():
            backing.assert_unchanged()
            with pq.ParquetFile(path) as reader:
                for block in reader.iter_batches(batch_size=min(65536, maximum_rows)):
                    yield index_storage.restore(block.to_pandas())
            backing.assert_unchanged()
        output = Dataset.from_batches(factory, columns, row_count=rows,
                                      metadata={"analysis": {**metadata, **response_metadata, "missing_prediction_rows": missing,
                                                              "response_definition": definition,
                                                              "storage": "owned indexed Parquet; one bounded block",
                                                              "index_preserved": True}})
        output._owned_prediction_output = scratch
        output._prediction_backing = backing
        return output
    except BaseException:
        if writer is not None:
            writer.close()
        scratch.cleanup()
        raise
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def predict_dataset(model, source, *, kind, alpha, interval, term, outcome, batch_rows):
    from .prediction import _prediction_frame
    replay = _Replay(model, source, weights=False, batch_rows=batch_rows, kind=kind)
    def frames():
        for raw, _, _ in replay.batches():
            yield _prediction_frame(model, raw, kind=kind, significance=alpha, interval=interval,
                                    term=term, outcome=outcome)
    output = materialize_predictions(frames(), {"estimator": model.result.spec.estimator,
                                                "kind": kind, "precision": "float64",
                                                "interval_method": "pointwise delta method" if interval else None})
    output._metadata["analysis"]["streaming"] = replay.metadata()
    return output


def margins_dataset(model, source, variables, *, method, at, kind, outcome, selected, batch_rows):
    from .prediction import (_Design, _encode, _effect_inference, _margins_setting)
    from .heteroskedastic_prediction import parameter_uncertainty
    from .ordinal_prediction import category_standard_error
    settings = _grids(model, at)
    contrasts = sum(len(model.categories[name]) - 1 if name in model.categories else 1 for name in variables)
    outcomes = len(selected) if selected is not None else 1
    count, k = len(settings) * contrasts * outcomes, len(model.state.terms)
    encoded_width = getattr(model.response_adapter,"encoded_width",k)
    replay = _Replay(model, source, weights=True, batch_rows=batch_rows, kind=kind,
                     state_bytes=64 * count * (k + 1) + 32 * (2 * contrasts + 1) * (k + 2))
    rows, gradients, scaled_gradients = [], [], []
    if method == "ame":
        reducer, template = _Average(), None
        for _, frame, raw_weights in replay.batches():
            if not len(frame):
                continue
            w = raw_weights / raw_weights.max()
            w = w / w.sum()
            batch_rows_out, batch_gradients, batch_scaled = [], [], []
            for setting in settings:
                def encoded(overrides):
                    scenario = frame.copy(deep=True)
                    for name, value in {**setting, **overrides}.items():
                        scenario[name] = value
                    return _encode(model, scenario)
                local = _margins_setting(model, variables, method, kind, selected, setting, encoded, w, inference=False)
                batch_rows_out.extend(local[0])
                batch_gradients.extend(local[1])
                batch_scaled.extend(local[2])
            template = batch_rows_out
            estimates = torch.tensor([item["estimate"] for item in template], dtype=torch.float64)[:, None]
            values = [estimates, torch.tensor(batch_gradients, dtype=torch.float64)]
            if model.heteroskedastic is not None:
                values.append(torch.tensor(batch_scaled, dtype=torch.float64))
            reducer.add(torch.cat(values, dim=1), raw_weights)
        if template is None:
            _error("empty_sample", "Marginal effects need complete, positive-weight evaluation rows.")
        correlation = parameter_uncertainty(model.state)[1] if model.heteroskedastic is not None else None
        for item, aggregate in zip(template, reducer.value, strict=True):
            gradient = aggregate[1:1 + k]
            scaled = aggregate[1 + k:] if correlation is not None else None
            rows.append({**item, **_effect_inference(model.state, float(aggregate[0]), gradient,
                         categorical=model.choice is not None or correlation is not None,
                         std_error=category_standard_error(scaled, correlation) if correlation is not None else None)})
            gradients.append(gradient.tolist())
            scaled_gradients.append(scaled.tolist() if scaled is not None else None)
    else:
        # One verified pass per at-grid, reducing all necessary contrasts in
        # that pass. No nonlinear mean is evaluated until global means exist.
        overrides = [{}]
        for name in variables:
            if name in model.categories:
                overrides.extend({name: level} for level in model.categories[name])
        for setting in settings:
            reducers = [_Average() for _ in overrides]
            for _, frame, raw_weights in replay.batches():
                if not len(frame):
                    continue
                w = raw_weights / raw_weights.max()
                w = w / w.sum()
                for override, reducer in zip(overrides, reducers, strict=True):
                    scenario = frame.copy(deep=True)
                    for name, value in {**setting, **override}.items():
                        scenario[name] = value
                    design = _encode(model, scenario)
                    mean = torch.cat([w @ design.x, (w @ design.deterministic)[None], (w @ design.scale)[None]])
                    reducer.add(mean, raw_weights)
            if reducers[0].value is None:
                _error("empty_sample", "Marginal effects need complete, positive-weight evaluation rows.")
            def encoded(override):
                index = overrides.index(override)
                average = reducers[index].value
                return _Design(average[:encoded_width][None], average[encoded_width:encoded_width + 1], average[encoded_width + 1:encoded_width + 2])
            local = _margins_setting(model, variables, method, kind, selected, setting, encoded,
                                     torch.ones(1, dtype=torch.float64))
            rows.extend(local[0])
            gradients.extend(local[1])
            scaled_gradients.extend(local[2])
    output = as_frame(pd.DataFrame(rows))
    n, complete, positive, _ = replay.baseline
    output.attrs.update(estimator=model.result.spec.estimator, kind=kind, evaluation_rows=positive,
                        input_evaluation_rows=n, complete_evaluation_rows=complete,
                        zero_weight_rows_excluded=complete - positive,
                        delta_gradients=gradients, delta_scaled_gradients=scaled_gradients,
                        parameter_terms=list(model.state.terms), covariance_source="saved fitted parameter covariance",
                        mem_definition="at weighted means of encoded design, offset and trials",
                        evaluation_sample="explicit supplied rows; outcomes not required",
                        streaming=replay.metadata())
    if model.choice is not None:
        output.attrs.update(outcome_labels=list(model.choice.labels),
                            outcome=model.choice.labels[selected[0]] if outcome is not None else None,
                            base_outcome=model.choice.labels[model.choice.base] if model.choice.base is not None else None)
    return output
