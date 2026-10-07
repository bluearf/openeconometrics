"""Full-source Johansen VEC from bounded, disk-ordered joint QR factors.

The factor preserves the complete joint product geometry, including constant
and trend vectors. Artificial factor rows are never counted as observations.
Higher residual moments, LM tests and predictions replay the actual periods.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares, symmetrize
from openecon.engines.streaming_ols import _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.resources import plan_workspace
from openecon.streaming_design import numeric_values
from . import registry
from .core import kernel_call
from .ordered_replay import OrderedReplay
from .streaming_linear import _Notes
from .streaming_var import _Facade, _diagnostics, _head
from .var import johansen as jo
from .var import kernels, vec
from .var.common import system_names
from .var.estimators import check_model_size


class _VECReplay:
    def __init__(self, ordered):
        self.ordered, self.spec = ordered, ordered.spec
        notes = _Notes(self.spec)
        self.names = system_names(self.spec)
        self.p, self.rank, self.trend = (notes.option(name) for name in ("lags", "rank", "trend"))
        self.p, self.rank = int(self.p), int(self.rank)
        self.k, self.t = len(self.names), ordered.count-self.p
        self.z1_labels = self.names+(["_cons"] if self.trend == "rconstant" else ["_trend"] if self.trend == "rtrend" else [])
        self.z2_labels = [("LD." if lag == 1 else f"L{lag}D.")+name for name in self.names for lag in range(1, self.p)]
        if self.trend == "trend":
            self.z2_labels.append("trend")
        self.constant = self.trend in {"constant", "rtrend", "trend"}
        if self.constant:
            self.z2_labels.append("Intercept")
        self.width = 2+self.k+len(self.z1_labels)+len(self.z2_labels)
        check_model_size(self.k*(self.rank+len(self.z2_labels)))
        if self.width > 384:
            raise AnalysisError("model_too_wide", "Combined Johansen lag/moment factor exceeds384 columns.")
        if self.t <= len(self.z2_labels)+len(self.z1_labels)+self.k:
            raise AnalysisError("insufficient_observations", "Johansen regression needs more usable periods than lag and level columns.")
        fixed = 512*self.width**2+128*(self.k*(self.rank+len(self.z2_labels)))**2+8*1024**2
        ordered.rows = min(ordered.rows, max(1, (ordered.budget-fixed)//(128*self.width)))
        self.plan = plan_workspace("bounded Johansen VEC factor and diagnostics", {
            "normalized_rows_and_lags": 128*ordered.rows*self.width,
            "joint_factor_tree": 512*self.width**2,
            "parameter_covariance": 128*(self.k*(self.rank+len(self.z2_labels)))**2,
            "reader_and_SQL_cache": 8*1024**2,
            "lag_history": 128*(max(self.p, int(notes.option("lm_lags")))+1)*self.width,
        }, budget_bytes=ordered.budget)
        moments = _WeightedMoments(self.k+1, intercept=True)
        for frame in ordered.source.iter_batches(batch_rows=ordered.rows):
            levels = self.values(frame)
            moments.add(torch.cat((torch.ones((len(levels), 1), dtype=torch.float64), levels), dim=1), torch.ones(len(levels), dtype=torch.float64), False)
            self.last_levels = levels[-self.p:].clone() if len(levels) >= self.p else levels.clone() if not hasattr(self, "last_levels") else torch.cat((self.last_levels, levels))[-self.p:].clone()
        self.shift = moments.anchor[1:]+moments.magnitude[1:]*moments.mean[1:] if self.trend != "none" else torch.zeros(self.k, dtype=torch.float64)
        self.report = pd.concat(list(_head(ordered.source.iter_batches(batch_rows=ordered.rows), min(ordered.count, 400+self.p))), ignore_index=True).iloc[self.p:].copy()
        self.positions = self.report.pop(ordered.positions_column).tolist()
        self.notes = notes
        self.frame = SimpleNamespace(spec=self.spec, info=registry.get("vec"), n=len(self.report),
                                    original=self.report, sample=self.report, positions=self.positions,
                                    notes={}, warnings=notes.warnings, _sorted_by=[self.spec.time] if self.spec.time else [],
                                    resource_plans=[self.plan.record()], option=notes.option, warn=notes.warn,
                                    numeric=lambda name: numeric_values(self.report[name], name))

    def values(self, frame):
        return torch.stack([numeric_values(frame[name], name) for name in self.names], dim=1)

    def sample(self):
        return SimpleNamespace(frame=self.frame, names=self.names, levels=self.last_levels,
                               last_period=self.ordered.last_period)

    def pieces(self):
        carry, seen = None, 0
        for frame in self.ordered.source.iter_batches(batch_rows=self.ordered.rows):
            values = self.values(frame)
            old = 0 if carry is None else len(carry)
            levels = values if carry is None else torch.cat((carry, values))
            first = max(self.p, old)
            if first < len(levels):
                differences = levels[1:]-levels[:-1]
                z0 = differences[first-1:]
                z1 = levels[first-1:-1]-self.shift
                z2 = torch.stack([differences[first-lag-1:len(levels)-lag-1, variable]
                                  for variable in range(self.k) for lag in range(1, self.p)], dim=1) if self.p > 1 else torch.empty((len(z0), 0), dtype=torch.float64)
                index = torch.arange(seen-old+first+1, seen+len(values)+1, dtype=torch.float64)
                ones = torch.ones((len(z0), 1), dtype=torch.float64)
                if self.trend == "rconstant":
                    z1 = torch.cat((z1, ones), dim=1)
                elif self.trend == "rtrend":
                    z1 = torch.cat((z1, index[:, None]), dim=1)
                if self.trend == "trend":
                    z2 = torch.cat((z2, index[:, None]), dim=1)
                if self.constant:
                    z2 = torch.cat((z2, ones), dim=1)
                yield z0, z1, z2, index, frame.iloc[first-old:].reset_index(drop=True)
            carry = levels[-self.p:].clone()
            seen += len(values)

    def joint(self):
        for z0, z1, z2, index, _ in self.pieces():
            yield torch.cat((torch.ones((len(z0), 1), dtype=torch.float64), index[:, None], z0, z1, z2), dim=1)

    def johansen(self):
        moments = _WeightedMoments(self.width, intercept=True)
        for block in self.joint():
            moments.add(block, torch.ones(len(block), dtype=torch.float64), False)
        means = moments.anchor+moments.magnitude*moments.mean
        scales = moments.magnitude*(moments.m2.value/moments.mass).clamp_min(0).sqrt()
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        means[0], scales[0] = 0., 1.
        tree = _TSQRTree()
        for block in self.joint():
            tree.add(qr_factor((block-means)/scales))
        factor = tree.finish()
        raw = factor*scales
        raw += factor[:, :1]*means
        self.ones, index = raw[:, 0], raw[:, 1]
        z0 = raw[:, 2:2+self.k]
        z1 = raw[:, 2+self.k:2+self.k+len(self.z1_labels)]
        z2 = raw[:, 2+self.k+len(self.z1_labels):]
        self.centered_tss = moments.m2.value[2:2+self.k]*moments.magnitude[2:2+self.k].square()
        both = torch.cat((z0, z1), dim=1)
        if len(self.z2_labels):
            fit = kernel_call(least_squares, z2, both, drop_collinear=True)
            if fit.omitted:
                raise AnalysisError("collinear_system", "The lagged differences are linearly dependent; remove a redundant variable or lag.")
            both = fit.resid
        r0, r1 = both[:, :self.k], both[:, self.k:]
        s00, s01, s11 = symmetrize(r0.T@r0/self.t), r0.T@r1/self.t, symmetrize(r1.T@r1/self.t)
        lower = kernel_call(kernels.spd_factor, s00, what="covariance matrix of the differences")
        chol = kernel_call(kernels.spd_factor, s11, code="collinear_system", what="moment matrix of the lagged levels")
        whitened = torch.linalg.solve_triangular(lower, s01, upper=False)
        whitened = torch.linalg.solve_triangular(chol, whitened.T, upper=False).T
        _, singular, vh = torch.linalg.svd(whitened, full_matrices=False)
        eigenvalues = singular[:self.k].square()
        if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues[0]) >= 1-1e-12:
            raise AnalysisError("collinear_system", "A Johansen canonical correlation equals one; the system is degenerate.")
        vectors = torch.linalg.solve_triangular(chol.T, vh[:self.k].T, upper=True)
        return jo.Johansen(self.t, self.k, self.p, self.trend, z0, z1, z2,
                           self.z1_labels, self.z2_labels, s00, s01, s11, eigenvalues,
                           vectors, kernels.logdet(lower), index, self.shift)

    def set_beta(self, beta):
        self.beta_centered = beta.clone()

    def fit_system(self, z, y, names):
        fit = kernels.fit_system(z, y, names)
        fit.sigma_ml *= len(y)/self.t
        fit.logdet_ml = kernels.logdet(kernels.spd_factor(fit.sigma_ml))
        return fit

    def design(self, z1, z2, index, centered_mu, rho):
        ce = z1@self.beta_centered
        if centered_mu is not None:
            ce += centered_mu
        if rho is not None:
            ce += index[:, None]*rho
        return torch.cat((ce, z2), dim=1)

    def diagnostic_source(self, centered_mu, rho):
        names = [f"D_{name}" for name in self.names]
        terms = [f"L._ce{i+1}" for i in range(self.rank)]+self.z2_labels
        def factory():
            for z0, z1, z2, index, _ in self.pieces():
                design = self.design(z1, z2, index, centered_mu, rho)
                yield pd.DataFrame(torch.cat((z0, design), dim=1).numpy(), columns=names+terms)
        def matrix(frame, selected):
            return torch.stack([numeric_values(frame[name], name) for name in selected], dim=1)
        return SimpleNamespace(ordered=self.ordered, p=self.p, names=names, constant=self.constant,
                               factory=factory, matrix=matrix), terms

    def diagnostics(self, beta, centered_mu, rho, fit, omega, lm_lags):
        lagged, terms = self.diagnostic_source(centered_mu, rho)
        record = _diagnostics(lagged, terms, fit.coef, fit.xtx_inv, omega, omega, lm_lags)
        return record["tests"], record["extra"]

    def preview(self, beta, centered_mu, rho, fit):
        lagged, terms = self.diagnostic_source(centered_mu, rho)
        report = pd.concat(list(_head(lagged.factory(), len(self.report))), ignore_index=True)
        return (lagged.matrix(report, terms)@fit.coef.T)[:, 0], numeric_values(report[lagged.names[0]], lagged.names[0])

    def finalize(self, result):
        facade = _Facade(self.spec, self.ordered, self.report, self.positions, self.t, {}, {})
        result.provenance.update(facade.provenance())
        result.provenance.update({"solver": "global_tsqr_disk_ordered_johansen",
                                  "sample_order": "sorted by "+self.spec.time if self.spec.time else "input row order",
                                  "moment_factor_memory_complexity": "O(batch_rows*lag_width + lag_width**2)",
                                  "factor_rows_are_observations": False,
                                  "resource_plans": [self.plan.record(), self.ordered.plan.record()]})
        result.nobs_original = self.ordered.original_count
        result.dropped_rows = self.ordered.original_count-self.t
        result.sample_positions = []
        return result


def fit_streaming_vec(spec, source):
    with OrderedReplay(spec, source) as ordered:
        replay = _VECReplay(ordered)
        result = kernel_call(vec.fit_vec, spec, None, _replay=replay)
        ordered.verify_original()
        return result
