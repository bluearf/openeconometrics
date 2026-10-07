"""Exact replayed SUR, multivariate OLS and 3SLS using global TSQR factors.

The existing system kernels work solely on a joint R factor. Only construction
of that factor and the reporting sample differ here; the GLS, constraints,
iteration, residual covariance and instrument projection are unchanged.
"""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.execution import execution_scope, qr_factor
from openecon.engines.streaming_ols import _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.models import ModelSpec
from openecon.resources import plan_workspace, tensor_bytes
from .core import ModelFrame
from .replay_sample import ReplaySample
from .systems.common import SystemData

ESTIMATORS = ("sureg", "mvreg", "reg3")


class _Builder:
    def __init__(self, sample: ReplaySample):
        self.sample = sample

    def __call__(self, frame, *, first, rest, constant, outcomes):
        sample = self.sample
        names = list(dict.fromkeys([*first, *rest]))
        labels = ["Intercept"] if constant else []
        blocks, terms = {}, {}
        for name in names:
            design = sample.designs[name]
            if name in outcomes and (name in sample.categories or not design.terms):
                raise AnalysisError("constant_outcome", "System outcomes must be numeric and variable.")
            blocks[name] = list(range(len(labels), len(labels) + len(design.terms)))
            terms[name] = list(design.terms)
            labels.extend(design.terms)
        width = len(labels)
        if width > 384:
            raise AnalysisError("model_too_wide", "Combined streaming system width exceeds384 columns.")
        options = frame.spec.options
        equations = options.get("equations")
        count = len(equations) if equations else len(outcomes)
        parameters = (sum(int(eq.get("constant", frame.spec.intercept)) + sum(len(blocks[v]) for v in eq.get("x", []))
                          for eq in equations) if equations else count * (int(constant) + sum(len(blocks[v]) for v in frame.spec.predictors)))
        budget = plan_workspace("streaming combined linear system", {
            "weighted_block_and_normalization": tensor_bytes((sample.rows, width), itemsize=48),
            "compressed_system_and_factors": tensor_bytes((count * width, parameters), itemsize=24),
            "parameter_covariance_and_restrictions": tensor_bytes((parameters, parameters), itemsize=64),
            "global_factor_tree": tensor_bytes((width, width), itemsize=512),
        })
        frame.resource_plans.append(budget.record())

        def raw(batch):
            pieces = [torch.ones((len(batch.frame), 1), dtype=torch.float64)] if constant else []
            pieces.extend(sample.raw_design(batch, name) for name in names)
            return torch.cat(pieces, dim=1)

        moments = _WeightedMoments(width, intercept=constant)
        low, high = torch.full((width,), torch.inf), torch.full((width,), -torch.inf)
        report = []
        for batch in sample.batches():
            values = raw(batch)
            moments.add(values, batch.weights, False)
            low, high = torch.minimum(low, values.amin(0)), torch.maximum(high, values.amax(0))
            retained = sum(len(item) for item in report)
            if retained < len(sample.sample):
                report.append(values[:len(sample.sample) - retained].clone())
        flat = (low == high).nonzero().flatten().tolist()
        if any(blocks[name][0] in flat for name in outcomes):
            raise AnalysisError("constant_outcome", "A system outcome has no global sample variation.")
        magnitude = moments.magnitude.clone()
        magnitude = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
        variance = moments.m2.value / moments.mass
        rms = variance.clamp_min(0).sqrt()
        rms = torch.where(rms > 0, rms, torch.ones_like(rms))
        means = moments.anchor + magnitude * moments.mean
        scales = magnitude * rms
        if constant:
            means[0], scales[0] = 0., 1.
        tree = _TSQRTree()
        with execution_scope("auto") as trace:
            for batch in sample.batches():
                values = raw(batch)
                if constant:
                    values = ((values - moments.anchor) / magnitude - moments.mean) / rms
                    values[:, 0] = 1.
                else:
                    # An origin-free model must preserve its uncentered norm.
                    values = values / scales
                tree.add(qr_factor(values * batch.weights.sqrt()[:, None]))
            factor = tree.finish()
        mapping = torch.diag(scales)
        if constant:
            factor[0, 1:] = 0.
            mapping[0] = means
            mapping[0, 0] = 1.
        factor = factor @ mapping
        instruments = int(constant) + sum(len(blocks[name]) for name in dict.fromkeys(first))
        frame.notes["streaming_execution"] = trace.metadata()
        return SystemData(factor, labels, blocks, terms, constant, instruments, sample.nobs,
                          sample.nrows, None, torch.cat(report), sample.categories,
                          {index for index in flat if index > 0 or not constant})


def fit_streaming_systems(spec: ModelSpec, source: Dataset):
    if spec.estimator not in ESTIMATORS:
        raise AnalysisError("streaming_unsupported", "This estimator is not a streaming linear system.")
    sample = ReplaySample(spec, source)
    if spec.estimator == "mvreg":
        names = list(dict.fromkeys([*spec.predictors, *spec.columns["outcomes"]]))
        outcomes = list(spec.columns["outcomes"])
    else:
        names = list(spec.columns["system"])
        outcomes = [item["y"] for item in spec.options["equations"]]
    if set(outcomes) & set(spec.categorical):
        raise AnalysisError("invalid_spec", "System outcomes cannot be categorical.")
    for name in names:
        sample.add_design(name, [name], intercept=False)
    sample.prepare()
    # This is a bounded presentation frame, never an estimation approximation.
    # Every coefficient/covariance comes from the R factor of all source rows.
    report = ModelFrame(spec, sample.sample)
    builder = _Builder(sample)
    if spec.estimator == "reg3":
        from .systems.reg3 import fit_reg3
        function = fit_reg3
    else:
        from .systems.sureg import fit_mvreg, fit_sureg
        function = fit_sureg if spec.estimator == "sureg" else fit_mvreg
    result = function(spec, None, _frame=report, _system_factory=builder)
    result.nobs = sample.nobs
    result.nobs_original = sample.original_count
    result.dropped_rows = sample.original_count - sample.nrows
    result.sample_positions = []
    for record, physical in zip(result.predictions, sample.sample_positions, strict=False):
        record["row"] = physical
    omissions = list(dict.fromkeys([*result.provenance.get("omitted_terms", []), *sample.notes["omitted_terms"]]))
    result.provenance.update(sample.provenance())
    result.provenance["omitted_terms"] = omissions
    result.provenance.update(report.notes["streaming_execution"])
    result.provenance["solver"] = "global_centered_tsqr_" + spec.estimator
    result.provenance["system_compression"] = "One global weighted joint factor; unchanged native system kernels"
    return result
