"""Full-source ARCH-family inference using finite native variance state."""
from __future__ import annotations

import math
import sqlite3
from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import normal_isf
from openecon.streaming_design import numeric_values
from . import registry
from .arch import postfit
from .arch.estimators import _CORRECTIONS, _PRESAMPLE
from .arch.layout import read_layout
from .arch.maximize import maximize
from .arch.recursions import dense
from .arima.diagnostics import default_lags
from .arima.filters import spectral_radius
from .core import build_result, information_criteria, kernel_call, ml_covariance, wald_test
from .ordered_replay import OrderedReplay
from .replay_sample import ReplaySample
from .streaming_arch_kernel import StreamingArchLikelihood
from .streaming_linear import _Notes
from .streaming_residual_diagnostics import diagnostics
from .streaming_var import _Facade, _head


class _Replay:
    def __init__(self, ordered):
        self.ordered, self.spec, self.n = ordered, ordered.spec, ordered.count
        self.sample = ReplaySample(self.spec, ordered.source, batch_rows=ordered.rows)
        self.design = self.sample.add_design("mean")
        variance = registry.role_columns(self.spec, "variance_x")
        self.zdesign = self.sample.add_design("variance", variance, intercept=True) if variance else None
        self.sample.prepare()
        self.layout = read_layout(self.spec, len(self.design.terms), len(self.zdesign.terms)-1 if self.zdesign else 0)
        layout, ix = self.layout, self.layout.index
        if self.n<layout.k_free+layout.max_lag+5:
            raise AnalysisError("insufficient_observations", "ARCH needs more periods than parameters and lag state.")
        order = min(max(layout.max_lag+4, round(math.log(self.n)**2)), self.n//4) if layout.ma_lags else 0
        self.plan = self.sample.plan_rows("full ARCH analytic likelihood and finite variance state", {
            "native_state_and_tangent_ring": 256*layout.max_lag*(layout.k+1),
            "optimizer_Hessian_score_and_HR_factors": 2048*(layout.k+order+3)**2,
            "owned_time_and_sign_SQLite_cache": 12*1024**2,
            "retained_variance_preview_and_category_metadata": 128*400*(layout.k+4),
        }, 512*(layout.k+order+4))
        self.back = torch.eye(layout.k, dtype=torch.float64)
        self.center_x = self.design.means[self.design.kept].clone()
        if self.spec.intercept and layout.k_x>1:
            self.back[ix.x[0], ix.x[1:]] = -self.center_x[1:]
        self.center_z = self.zdesign.means[self.zdesign.kept][1:].clone() if self.zdesign else torch.empty(0, dtype=torch.float64)
        if self.zdesign:
            self.back[ix.c, ix.z] = -self.center_z
        self.notes = _Notes(self.spec)
        self.like = StreamingArchLikelihood(self.numeric, self.n, layout, budget_bytes=ordered.budget, scratch=ordered.path.parent)
        self.report = pd.concat(list(_head(ordered.source.iter_batches(batch_rows=ordered.rows), 400)), ignore_index=True)
        self.positions = self.report.pop(ordered.positions_column).tolist()
        self.frame = SimpleNamespace(spec=self.spec, info=registry.get("arch"), n=len(self.report), original=self.report, sample=self.report,
            positions=self.positions, warnings=self.notes.warnings, notes={"omitted_terms": self.sample.notes["omitted_terms"]},
            option=self.notes.option, warn=self.notes.warn, numeric=lambda name: numeric_values(self.report[name], name),
            _sorted_by=[self.spec.time] if self.spec.time else [], resource_plans=[self.plan.record(), ordered.plan.record()])

    def numeric(self):
        for batch in self.sample.batches():
            x = self.sample.raw_design(batch, "mean")
            if self.spec.intercept and self.layout.k_x>1:
                x[:, 1:] -= self.center_x[1:]
            z = self.sample.raw_design(batch, "variance")[:, 1:] - self.center_z if self.zdesign else torch.empty((len(x), 0), dtype=torch.float64)
            yield x, batch.numeric(self.spec.outcome), z

    def diagnostic_tests(self, theta, signs, flat):
        def factory():
            for row in self.like.blocks(theta, signs=signs, flat=flat):
                if row is None:
                    raise AnalysisError("numerical_failure", "ARCH diagnostics are undefined at estimates.")
                yield row[1]/row[3].sqrt()
        lags = min(self.notes.option("test_lags"), self.n-1) if self.notes.option("test_lags") is not None else default_lags(self.n)
        lm = self.notes.option("test_lags") or postfit.ARCH_LM_LAGS
        values = diagnostics(factory, self.n, lags, fitted_parameters=len(self.layout.ar_lags)+len(self.layout.ma_lags),
            arch_lags=min(lm, self.n-1), label_suffix="standardized residuals", allow_degenerate_aux=True)
        lm_result = values.pop("arch_lm", None)
        if lm_result:
            values["arch_lm_residuals"] = lm_result
        squares = diagnostics(lambda: (row.square() for row in factory()), self.n, lags, arch_lags=0, label_suffix="squared standardized residuals")
        if "ljung_box" in squares:
            values["ljung_box_squared"] = squares["ljung_box"]
        return values


def fit_streaming_arch(spec, source):
    try:
        with OrderedReplay(spec, source) as ordered:
            replay = _Replay(ordered)
            try:
                result = kernel_call(_fit, replay)
                ordered.verify_original()
                return result
            finally:
                replay.like.close()
    except (OSError, sqlite3.Error) as error:
        raise AnalysisError("state_spill_failed", "ARCH state/sign replay needs writable temporary disk space.") from error


def _fit(replay):
    spec, layout, frame, like, n = replay.spec, replay.layout, replay.frame, replay.like, replay.n
    ix = layout.index
    tolerance, iterations = float(frame.option("tolerance")), int(frame.option("max_iterations"))
    if tolerance<=0.:
        raise AnalysisError("invalid_option", "tolerance must be positive.")
    fit = maximize(like, tolerance, iterations)
    final = like.evaluate(fit.theta, scores=True, signs=fit.signs, flat=fit.flat)
    if final is None:
        raise AnalysisError("numerical_failure", "ARCH likelihood is undefined at estimates.")
    covariance, info = ml_covariance(frame, hessian=fit.hessian, scores=final.scores@fit.transform, nobs=n)
    transform, theta = replay.back@fit.transform, replay.back@fit.theta
    covariance = transform@covariance@transform.T
    info["correction"] = _CORRECTIONS[info["covariance"]]
    if layout.restricted:
        info["restriction"] = "IGARCH: the last GARCH coefficient is one minus the other ARCH and GARCH coefficients; its standard error follows from them"
    zterms = replay.zdesign.terms[1:] if replay.zdesign else []
    terms, equations = layout.terms(spec.outcome, replay.design.terms, zterms)
    persist = postfit.persistence(layout, theta)
    unconditional = postfit.unconditional_variance(layout, theta, persist)
    if not layout.restricted and persist is not None and persist>=1.:
        frame.warn(f"The estimated variance equation is not covariance stationary (persistence {persist:.4f} >=1).")
    if layout.kind != "egarch":
        negative = [terms[i] for i in [*ix.a, *ix.b] if float(theta[i])<0.]
        negative += [terms[j] for i, j in zip(ix.a, ix.g, strict=False) if float(theta[i]+theta[j])<0.]
        if not layout.k_z and float(theta[ix.c])<=0.:
            negative.append(terms[ix.c])
        if negative:
            frame.warn("The variance equation has negative coefficients ("+", ".join(negative)+"); the sample variance is positive but forecasts need not be.")
    if layout.ar_lags and spectral_radius(dense(theta[ix.ar].tolist(), layout.ar_lags))>=1.:
        frame.warn("The estimated disturbance AR polynomial is not stationary.")
    if layout.ma_lags and spectral_radius(dense(theta[ix.ma].tolist(), layout.ma_lags, -1.))>=1.:
        frame.warn("The estimated disturbance MA polynomial is not invertible.")
    tests = replay.diagnostic_tests(fit.theta, fit.signs, fit.flat)
    tested = [i for i in ix.x if terms[i] != "Intercept"]+ix.m+ix.ar+ix.ma
    if tested:
        tests["model"] = wald_test(theta, covariance, tested, label="Wald chi2 test that all mean-equation coefficients except the constant are zero")
    extra = {"model": layout.describe(), "persistence": persist, "unconditional_variance": unconditional,
        "distribution": postfit.distribution_record(layout, theta, covariance, normal_isf(.5*spec.alpha)),
        "state": {"residuals": final.tail_residual[-layout.max_lag:].tolist(), "disturbances": final.tail_disturbance[-layout.max_lag:].tolist(),
                  "variances": final.tail_variance[-layout.max_lag:].tolist(), "presample_variance": final.presample_variance, "last_period": replay.ordered.last_period},
        "conditional_variance_tail": final.tail_variance.tolist()}
    if layout.kind == "egarch" and persist is not None and abs(persist)<1. and not layout.k_z:
        extra["unconditional_log_variance"] = float(theta[ix.c])/(1.-persist)
    fit.record.update(gradient="analytic forward tangents in compiled float64 finite lag state", engine="bounded_native_state",
                      full_source_likelihood=True, sign_patterns="owned disk files" if layout.kind == "egarch" else None)
    result = build_result(frame, terms=terms, params=theta, covariance=covariance, title=layout.title(), equations=equations, use_t=False, nobs=n,
        metrics={**information_criteria(final.value, layout.k_free, n), "persistence": persist, "unconditional_variance": unconditional, "iterations": fit.record["iterations"]},
        fitted=frame.numeric(spec.outcome)-final.residual, solver="global_stateful_arch_likelihood",
        solver_diagnostics={"converged": True, "iterations": fit.record["iterations"]}, optimizer=fit.record, inference=info, tests=tests, extra=extra,
        categories=replay.sample.categories, provenance={"likelihood": f"conditional {layout.dist} full-source innovation likelihood", "presample": _PRESAMPLE})
    facade = _Facade(spec, replay.ordered, replay.report, replay.positions, n, replay.sample.categories, replay.sample.provenance())
    result.provenance.update(facade.provenance(), resource_plans=frame.resource_plans, score_factor_rows_are_observations=False)
    result.nobs, result.nobs_original, result.dropped_rows, result.sample_positions = n, replay.ordered.original_count, replay.ordered.original_count-n, []
    return result
