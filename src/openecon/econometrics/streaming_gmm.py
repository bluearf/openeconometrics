"""Native global GMM: analytic moments/Jacobians and covariance replays.

Uses the same Gauss-Newton/LM solver and reporting code as dense GMM. Every
objective evaluation reads the entire sample. Only instrument factors,
moment sums, the parameter Jacobian and a reporting sample are retained.
"""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.execution import execution_scope, qr_factor
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_hac import HACAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.models import ModelSpec
from openecon.streaming_design import encode_cluster_labels
from .core import ModelFrame
from . import registry
from .replay_sample import ReplaySample
from .quantile import formula as formulas
from .systems.gmm import _instrument_lists, _check_exact_fit, parse_moments, fit_gmm


class _ReplayMoments:
    def __init__(self, sample: ReplaySample, frame: ModelFrame):
        self.sample, self.frame = sample, frame
        self.designs = list(sample.designs.values())
        self.widths = [len(design.terms) for design in self.designs]
        self.width = sum(self.widths)
        self.multiplier = (1. if not sample.spec.weights or sample.spec.weight_type in {"fweight", "aweight"}
                           else sample.nrows/(sample.weight_max*sample.weight_mass))
        self.mass = sample.nobs
        self.info = {}

    def initialize(self, parsed, index, parameters):
        self.formulas, self.index, self.parameters = list(parsed), index, parameters
        tape = max(len(formula.tape)*(1+len(formula.parameters)) for formula in parsed)
        buffers = {
            "moment_covariances_and_factors": 80*self.width**2,
            "moment_jacobians_and_lm_factors": 48*(self.width+parameters)*parameters,
        }
        if self.sample.spec.covariance == "cluster":
            buffers["fixed_cluster_spill_cache"] = 6*1024**2
        if self.sample.spec.covariance == "hac" or self.frame.option("wmatrix") == "hac":
            lags = int(self.frame.option("lags"))
            self.hac_buffers = HACAccumulator.workspace_buffers(self.width, lags, self.frame.option("kernel"))
            buffers.update({"hac_"+name: size for name, size in self.hac_buffers.items()})
        self.sample.plan_rows("global replay GMM objective", buffers,
                             8*(tape+8*self.width+8*len(parsed)*(parameters+1)))
        trees = [_TSQRTree() for _ in parsed]
        with execution_scope("auto") as trace:
            for batch in self.sample.batches():
                weight = self.weight(batch)
                for tree, design in zip(trees, self.designs, strict=True):
                    tree.add(qr_factor(batch.designs[design.name]*weight.sqrt()[:, None]))
        self.frame.notes["streaming_execution"] = trace.metadata()
        self.factors = [tree.finish() for tree in trees]
        if any(factor.shape[0] != factor.shape[1] or bool((factor.diagonal().abs() <= torch.finfo(torch.float64).eps*factor.diagonal().abs().max()).any())
               for factor in self.factors):
            raise AnalysisError("singular_instruments", "A moment equation has dependent global instruments.")
        originals = [torch.linalg.solve(design.transform.T, factor.T).T
                     for design, factor in zip(self.designs, self.factors, strict=True)]
        identity = torch.block_diag(*[factor.T for factor in originals])
        cross = _CompensatedSum((self.width, self.width))
        columns = sorted({name for formula in parsed for name in formula.columns})
        sums = _CompensatedSum((len(columns),))
        for batch in self.sample.batches():
            weight = self.weight(batch)
            z = torch.cat(self.bases(batch), dim=1)
            cross.add(z.T@(z*weight[:, None]))
            values = torch.stack([batch.numeric(name) for name in columns], dim=1)
            sums.add(weight@values.square())
        self.cross = cross.value
        by_column = dict(zip(columns, sums.value.tolist(), strict=True))
        self.floors = [1e-24*min((by_column[name] for name in formula.columns if by_column[name] > 0), default=0.)
                       for formula in parsed]
        return self, identity, [list(design.terms) for design in self.designs]

    def weight(self, batch):
        return batch.weights*self.multiplier

    def bases(self, batch):
        return [torch.linalg.solve_triangular(factor, batch.designs[design.name], upper=True, left=False)
                for design, factor in zip(self.designs, self.factors, strict=True)]

    def residuals(self, batch, theta, jacobian):
        columns = {name: batch.numeric(name) for formula in self.formulas for name in formula.columns}
        values, slopes = [], []
        for formula, index in zip(self.formulas, self.index, strict=True):
            value, local = formulas.evaluate(formula, columns, theta[index], len(batch.frame), jacobian=jacobian)
            values.append(value)
            if jacobian:
                full = torch.zeros((len(batch.frame), self.parameters), dtype=torch.float64)
                full[:, index] = local
                slopes.append(full)
        return torch.stack(values, dim=1), slopes

    def rows(self, bases, residuals):
        return torch.cat([z*residuals[:, j:j+1] for j, z in enumerate(bases)], dim=1)

    def moments(self, theta, jacobian=True):
        total = _CompensatedSum((self.width,))
        jac = _CompensatedSum((self.width, self.parameters)) if jacobian else None
        magnitude = _CompensatedSum((self.width, self.width)) if jacobian else None
        squares = _CompensatedSum((len(self.formulas),))
        report = []
        for batch in self.sample.batches():
            weight, bases = self.weight(batch), self.bases(batch)
            resid, slopes = self.residuals(batch, theta, jacobian)
            total.add(weight@self.rows(bases, resid))
            squares.add(weight@resid.square())
            retained = sum(len(item) for item in report)
            if retained < 400:
                report.append(resid[:400-retained].clone())
            if jacobian:
                jac.add(torch.cat([(z*weight[:, None]).T@slope for z, slope in zip(bases, slopes, strict=True)], dim=0))
                size = resid.abs()+torch.stack([slope.abs()@theta.abs() for slope in slopes], dim=1)
                rows = self.rows(bases, size)*weight[:, None]
                magnitude.add(rows.T@rows)
        if jacobian:
            self.theta, self.g = theta.clone(), total.value.clone()
            self._magnitude, self._squares = magnitude.value, squares.value
        return torch.cat(report), total.value, jac.value if jacobian else None

    def magnitude_meat(self):
        return self._magnitude

    def initial_s(self):
        # Within-equation bases are orthonormal; cross-equation blocks are0.
        return torch.block_diag(*[self.cross[start:start+width, start:start+width]
                                 for start, width in zip(self.starts, self.widths, strict=True)])

    @property
    def starts(self):
        return [sum(self.widths[:i]) for i in range(len(self.widths))]

    def check_exact(self, labels):
        # Existing exact-fit check uses weighted global sums; represent these
        # by one synthetic row solely for the check, never for estimation.
        _check_exact_fit(self._squares.clamp_min(0).sqrt()[None, :], None, self.floors, labels)

    def cluster_count(self):
        columns = registry.cluster_columns(self.sample.spec)
        if len(columns) != 1:
            raise AnalysisError("streaming_unsupported", "Replay GMM supports one cluster dimension.")
        if not hasattr(self, "_clusters"):
            with _cluster(1) as accumulator:
                for batch in self.sample.batches():
                    accumulator.add(encode_cluster_labels(batch.frame[columns[0]]), torch.zeros((len(batch.frame), 1), dtype=torch.float64))
                _, self._clusters = accumulator.finish()
        return self._clusters

    def moment_s(self, kind, center):
        if kind == "unadjusted":
            sigma = _CompensatedSum((len(self.formulas), len(self.formulas)))
            for batch in self.sample.batches():
                resid, _ = self.residuals(batch, self.theta, False)
                sigma.add((resid*self.weight(batch)[:, None]).T@resid)
            covariance = sigma.value/self.sample.nobs
            result = self.cross.clone()
            for j, (begin, width) in enumerate(zip(self.starts, self.widths, strict=True)):
                for k, (other, length) in enumerate(zip(self.starts, self.widths, strict=True)):
                    result[begin:begin+width, other:other+length] *= covariance[j, k]
            return (result+result.T)/2
        mean = self.g/self.mass if center else torch.zeros(self.width, dtype=torch.float64)
        total = _CompensatedSum((self.width, self.width))
        accumulator = ClusterAccumulator(self.width) if kind == "cluster" else None
        hac = HACAccumulator(self.width, int(self.frame.option("lags")), self.frame.option("kernel"),
                             budget_bytes=sum(self.hac_buffers.values())) if kind == "hac" else None
        try:
            for batch in self.sample.batches():
                weight = self.weight(batch)
                resid, _ = self.residuals(batch, self.theta, False)
                rows = self.rows(self.bases(batch), resid)-mean
                if accumulator is not None:
                    accumulator.add(encode_cluster_labels(batch.frame[registry.cluster_columns(self.sample.spec)[0]]), rows*weight[:, None])
                else:
                    rows = rows*(weight.sqrt() if self.sample.spec.weight_type == "fweight" else weight)[:, None]
                    if hac is not None:
                        hac.add(rows, batch.positions, time=batch.frame[self.sample.spec.time] if self.sample.spec.time else None,
                                units=encode_cluster_labels(batch.frame[self.sample.spec.panel]) if self.sample.spec.panel else None)
                    else:
                        total.add(rows.T@rows)
            if accumulator is not None:
                matrix, _ = accumulator.finish()
                self.info.update(accumulator.diagnostics)
            elif hac is not None:
                matrix = hac.finish()
                self.info.update(hac.diagnostics)
            else:
                matrix = total.value
            return matrix
        finally:
            if accumulator is not None:
                accumulator.close()
            if hac is not None:
                hac.close()


class _cluster(ClusterAccumulator):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def fit_streaming_gmm(spec: ModelSpec, source: Dataset, *, batch_rows: int | None = None):
    labels, parsed = parse_moments(spec.options["moments"])
    lists = _instrument_lists(spec.options.get("instruments"), len(parsed))
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    for label, instruments in zip(labels, lists, strict=True):
        if not instruments and not spec.options.get("instrument_constant", True):
            raise AnalysisError("invalid_spec", "A GMM equation needs at least one instrument.")
        sample.add_design(label, instruments, intercept=spec.options.get("instrument_constant", True), prefix=label+":")
    sample.prepare()
    report = ModelFrame(spec, sample.sample)
    context = _ReplayMoments(sample, report)
    result = fit_gmm(spec, None, _frame=report, _replay=context)
    result.nobs, result.nobs_original = sample.nobs, sample.original_count
    result.dropped_rows = sample.original_count-sample.nrows
    result.sample_positions = []
    result.inference.update(context.info)
    result.provenance.update(sample.provenance())
    result.provenance.update(report.notes["streaming_execution"])
    result.provenance["solver"] = "global_replay_gauss_newton_levenberg_marquardt_gmm"
    return result
