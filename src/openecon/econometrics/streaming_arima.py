"""Full-source ARIMA/SARIMA/ARMAX with bounded native filter state."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.execution import qr_factor
from openecon.engines.streaming_filter import PolynomialState
from openecon.engines.streaming_ols import _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import numeric_values
from . import registry
from .arima import estimators as arima
from .arima.diagnostics import default_lags
from .arima.maximize import maximize
from .core import build_result, information_criteria, kernel_call, wald_test
from .ordered_replay import OrderedReplay
from .replay_sample import ReplaySample
from .streaming_arma_kernel import StreamingArmaLikelihood
from .streaming_linear import _Notes
from .streaming_residual_diagnostics import diagnostics
from .streaming_var import _Facade, _head


class _ARIMAReplay:
    def __init__(self, ordered):
        self.ordered, self.spec = ordered, ordered.spec
        self.notes = _Notes(self.spec)
        self.orders = orders = arima.read_orders(self.spec)
        self.n = ordered.count-orders.lost
        self.constant = self.spec.intercept and bool(self.notes.option("constant"))
        if self.n < 2:
            raise AnalysisError("insufficient_observations", "Differencing leaves fewer than two observations.")
        self.level = ReplaySample(self.spec, ordered.source, batch_rows=ordered.rows)
        self.level.add_design("level", intercept=False)
        self.level.prepare()
        self.level_terms = self.level.designs["level"].terms
        self.difference_spec = self.spec.model_copy(update={"predictors": self.level_terms, "categorical": [], "intercept": self.constant})
        self.difference_source = Dataset.from_batches(self.differenced, [self.spec.outcome, *self.level_terms, *([self.spec.time] if self.spec.time else [])], row_count=self.n)
        self.sample = ReplaySample(self.difference_spec, self.difference_source, batch_rows=ordered.rows)
        self.design = self.sample.add_design("mean", intercept=self.constant)
        self.sample.prepare()
        self.k = len(self.design.terms)
        if self.n < max(self.k+orders.n_arma+3, orders.state+2):
            raise AnalysisError("insufficient_observations", "ARIMA needs more usable periods than its mean and ARMA parameters.")
        width = self.k+orders.n_arma+orders.state+3
        self.static_buffers = {
            "initial_level_sample_and_categories": self.level.reporting_bytes+self.level._category_bytes,
            "level_design_and_arma_covariance": 512*width**2,
            "filter_states_and_differences": 256*(orders.lost+orders.p_full+orders.q_full+41)*width,
            "SQL_reader_and_sort_cache": 8*1024**2,
        }
        self.bytes_per_row = 512*width
        self.plan = self.sample.plan_rows("stateful ARIMA full-source likelihood", self.static_buffers, self.bytes_per_row)
        # The shared initial projected reader also lives during differencing.
        self.level.rows = min(self.level.rows, self.sample.rows)
        moments = _WeightedMoments(2, intercept=True)
        for batch in self.sample.batches():
            y = batch.numeric(self.spec.outcome)
            moments.add(torch.stack((torch.ones_like(y), y), 1), torch.ones_like(y), False)
        self.y_anchor, self.y_magnitude, self.y_center = [float(v[1]) for v in (moments.anchor, moments.magnitude, moments.mean)]
        self.scale_y = self.y_magnitude*math.sqrt(float(moments.m2.value[1]/moments.mass))
        if not math.isfinite(self.scale_y) or self.scale_y <= 0.:
            raise AnalysisError("constant_outcome", "The differenced ARIMA outcome has no finite variation.")
        self.mean_y = self.y_anchor+self.y_magnitude*self.y_center if self.constant else 0.
        self.scale_x, self.mean_x = self.design.scales[self.design.kept], self.design.means[self.design.kept]
        options = {"p": orders.p, "q": orders.q, "seasonal_p": orders.seasonal_p, "seasonal_q": orders.seasonal_q,
                   "period": orders.period, "exact": self.notes.option("method") == "ml"}
        self.like = StreamingArmaLikelihood(self.numeric, self.n, self.k, budget_bytes=ordered.budget, workspace_reserver=self.reserve_state, **options)
        self.report = pd.concat(list(_head(ordered.source.iter_batches(batch_rows=ordered.rows), min(ordered.count, 400+orders.lost))), ignore_index=True).iloc[orders.lost:].copy()
        self.positions = self.report.pop(ordered.positions_column).tolist()
        self.frame = SimpleNamespace(spec=self.spec, info=registry.get("arima"), n=len(self.report), original=self.report, sample=self.report,
                                     positions=self.positions, warnings=self.notes.warnings, notes={"omitted_terms": self.sample.notes["omitted_terms"]},
                                     option=self.notes.option, warn=self.notes.warn, numeric=lambda name: numeric_values(self.report[name], name),
                                     _sorted_by=[self.spec.time] if self.spec.time else [], resource_plans=[self.plan.record(), ordered.plan.record()])

    def differenced(self):
        orders = self.orders
        # Match the native sequential differences, including their floating
        # point order, while retaining only finite filter histories.
        filters = [PolynomialState([1., *([0.]*(orders.period-1)), -1.]) for _ in range(orders.seasonal_d)]
        filters += [PolynomialState([1., -1.]) for _ in range(orders.d)]
        seen = 0
        for batch in self.level.batches():
            values = torch.cat((batch.numeric(self.spec.outcome)[:, None], self.level.raw_design(batch, "level")), 1).T
            for state in filters:
                values = state(values)
            skipped = min(values.shape[1], max(0, orders.lost-seen))
            seen += values.shape[1]
            if skipped < values.shape[1]:
                frame = pd.DataFrame(values[:, skipped:].T.numpy(), columns=[self.spec.outcome, *self.level_terms])
                if self.spec.time:
                    frame[self.spec.time] = batch.frame[self.spec.time].iloc[skipped:].reset_index(drop=True)
                yield frame

    def reserve_state(self, operation, buffers):
        plan = self.sample.plan_rows(operation, {**self.static_buffers, **buffers}, self.bytes_per_row)
        self.level.rows = min(self.level.rows, self.sample.rows)
        return plan

    def numeric(self):
        for batch in self.sample.batches():
            y = batch.numeric(self.spec.outcome)
            y = ((y-self.y_anchor)/self.y_magnitude-self.y_center)*(self.y_magnitude/self.scale_y) if self.constant else y/self.scale_y
            yield batch.designs["mean"], y

    def residual_factory(self, theta):
        return lambda: (row[0]*self.scale_y for row in self.like.decomposition_blocks(theta, scores=False))

    def state_record(self, theta, decomposition):
        orders, beta = self.orders, theta[:self.k]
        process, level = torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
        # Outcome net of nonconstant regressors in levels, before differencing.
        slope_terms = [term for term in self.design.terms if term != "Intercept"]
        slope_positions = [self.level_terms.index(term) for term in slope_terms]
        slope_beta = beta[1:] if self.constant else beta
        for batch in self.level.batches():
            values = batch.numeric(self.spec.outcome)
            if slope_terms:
                values = values-self.level.raw_design(batch, "level")[:, slope_positions]@slope_beta
            level = torch.cat((level, values))[-orders.lost:].clone() if orders.lost else level
        for x, y in self.numeric():
            values = (y-x@decomposition.psi[:self.k])*self.scale_y
            process = torch.cat((process, values))[-orders.p_full:].clone() if orders.p_full else process
        disturbance_covariance = decomposition.disturbance_covariance
        negligible = disturbance_covariance.numel() == 0 or float(disturbance_covariance.abs().max()) <= 1e-14
        return {"process": process.tolist(), "disturbances": (decomposition.disturbances*self.scale_y).tolist(),
                "disturbance_covariance": None if negligible else disturbance_covariance.tolist(),
                "levels": level.tolist(), "last_period": self.ordered.last_period}

    def decomposition(self, theta):
        tree, innovations, disturbances, count = _TSQRTree(), [], torch.empty(0, dtype=torch.float64), 0
        scores = self.spec.covariance != "nonrobust"
        for v, _, score, _, _, conditional in self.like.decomposition_blocks(theta, scores=scores):
            if scores:
                tree.add(qr_factor(score))
            used = min(len(v), max(0, 400-count))
            if used:
                innovations.append(v[:used].clone())
            count += len(v)
            disturbances = torch.cat((disturbances, conditional))[-self.orders.q_full:].clone() if self.orders.q_full else disturbances
        disturbances, covariance = self.like.end_disturbances, self.like.end_disturbance_covariance
        return SimpleNamespace(innovations=torch.cat(innovations), scores=tree.finish() if scores else None,
                               disturbances=disturbances, disturbance_covariance=covariance, psi=theta[:-1])

    def finalize(self, result):
        facade = _Facade(self.spec, self.ordered, self.report, self.positions, self.n, self.level.categories, self.sample.provenance())
        result.provenance.update(facade.provenance())
        result.provenance.update(solver="global_stateful_arma_likelihood", sample_order="sorted by "+self.spec.time if self.spec.time else "input row order",
                                 score_factor_rows_are_observations=False, resource_plans=self.frame.resource_plans,
                                 categorical_encoding=self.level.categories,
                                 retained_arma_innovation_rows=len(result.predictions), presample_plan=None if self.like.last_plan is None else self.like.last_plan.record())
        result.nobs_original, result.dropped_rows, result.sample_positions = self.ordered.original_count, self.ordered.original_count-self.n, []
        return result


def fit_streaming_arima(spec, source):
    with OrderedReplay(spec, source) as ordered:
        replay = _ARIMAReplay(ordered)
        result = kernel_call(_fit, replay)
        ordered.verify_original()
        return result


def _fit(replay):
    spec, frame, orders, like, n, k = replay.spec, replay.frame, replay.orders, replay.like, replay.n, replay.k
    tolerance, iterations = float(frame.option("tolerance")), int(frame.option("max_iterations"))
    if tolerance <= 0.:
        raise AnalysisError("invalid_option", "tolerance must be positive.")
    psi, hessian_c, optimizer = maximize(like, None, None, tolerance, iterations)
    pieces = like.evaluate(psi, derivatives=False)
    if pieces is None or not pieces.ss > 0.:
        raise AnalysisError("numerical_failure", "ARIMA likelihood is undefined at estimates.")
    sigma_std = math.sqrt(pieces.ss/n)
    theta_std = torch.cat((psi, torch.tensor([sigma_std], dtype=torch.float64)))
    ll = -.5*n*(math.log(2.*math.pi)+1.+math.log(pieces.ss/n))-.5*pieces.logdet-n*math.log(replay.scale_y)
    if optimizer.get("starts"):
        optimizer["starts"] = {name: None if value is None else value-n*math.log(replay.scale_y) for name, value in optimizer["starts"].items()}
    hessian = like.full_hessian(psi, hessian_c)
    decomposition = replay.decomposition(theta_std)
    units = torch.diag(torch.cat((replay.scale_y/replay.scale_x, torch.ones(orders.n_arma, dtype=torch.float64), torch.tensor([replay.scale_y], dtype=torch.float64))))
    if replay.constant:
        units[0, 1:k] = -replay.scale_y*replay.mean_x[1:]/replay.scale_x[1:]
    theta = units@theta_std
    if replay.constant:
        theta[0] += replay.mean_y
    phi, ma, sphi, sma = like.split(psi)
    roots = {"ar": arima._roots(phi, moving_average=False), "ma": arima._roots(ma, moving_average=True),
             "seasonal_ar": arima._roots(sphi, moving_average=False), "seasonal_ma": arima._roots(sma, moving_average=True),
             "convention": "inverse roots (eigenvalues of the companion matrix); stationary / invertible when every modulus is below 1; seasonal roots are those of the polynomial in L^s"}
    largest_ar = max((root["modulus"] for root in roots["ar"]+roots["seasonal_ar"]), default=0.)
    largest_ma = max((root["modulus"] for root in roots["ma"]+roots["seasonal_ma"]), default=0.)
    if largest_ar >= arima._NEAR_UNIT:
        frame.warn("The estimated AR polynomial is on or close to a unit root; consider differencing the series.")
    if largest_ma >= arima._NEAR_UNIT:
        frame.warn("The estimated MA polynomial is on or near the invertibility boundary; standard errors may be unreliable.")
    covariance, info = arima._covariance(frame, hessian, decomposition.scores, n, largest_ma)
    covariance = units@covariance@units.T
    terms, equations = list(replay.design.terms), [arima._differenced_label(spec.outcome, orders)]*k
    for label, ar, moving in (("ARMA", phi, ma), (f"ARMA{orders.period}", sphi, sma)):
        terms += [f"{label}:L{i}.ar" for i in range(1, len(ar)+1)]+[f"{label}:L{i}.ma" for i in range(1, len(moving)+1)]
        equations += [label]*(len(ar)+len(moving))
    terms.append("/sigma")
    equations.append(None)
    lags = default_lags(n) if frame.option("ljung_lags") is None else int(frame.option("ljung_lags"))
    tests = diagnostics(replay.residual_factory(theta_std), n, lags, fitted_parameters=orders.n_arma)
    tested = [i for i, term in enumerate(terms[:-1]) if term != "Intercept"]
    if tested:
        tests["model"] = wald_test(theta, covariance, tested, label="Wald chi2 test that all coefficients except the constant are zero")
    metrics = {**information_criteria(ll, k+orders.n_arma+1, n), "sigma": float(theta[-1]), "n_differenced": n, "iterations": optimizer["iterations"]}
    extra = {"model_order": {"p": orders.p, "d": orders.d, "q": orders.q, "P": orders.seasonal_p, "D": orders.seasonal_d, "Q": orders.seasonal_q, "period": orders.period},
             "method": frame.option("method"), "ar": phi, "ma": ma, "seasonal": {"ar": sphi, "ma": sma, "period": orders.period}, "roots": roots,
             "last_state": replay.state_record(theta, decomposition)}
    observed = frame.numeric(spec.outcome)
    result = build_result(frame, terms=terms, params=theta, covariance=covariance, equations=equations, use_t=False, metrics=metrics, nobs=n,
                           fitted=observed-decomposition.innovations*replay.scale_y, observed=observed, optimizer=optimizer, inference=info, tests=tests, extra=extra,
                           categories=replay.level.categories, title=orders.label()+" regression"+(" (conditional)" if frame.option("method") == "css" else ""),
                           solver="global_stateful_arma_likelihood", provenance={"method": frame.option("method"), "likelihood": "global exact stationary initial-state likelihood" if like.exact else "global conditional likelihood, only series presample at zero",
                               "differencing": {"d": orders.d, "D": orders.seasonal_d, "period": orders.period, "observations_lost": orders.lost},
                               "fitted_values": "one-step-ahead predictions in levels; first400 real retained observations"})
    result.coefficients[-1].p_value *= .5
    result.coefficients[-1].ci_low = max(0., result.coefficients[-1].ci_low)
    return replay.finalize(result)
