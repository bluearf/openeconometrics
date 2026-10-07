"""Censored Poisson/NB regression, using observed inclusive count events.

Auto mode follows cpoisson's ll/ul classification. An explicit status column
also admits genuinely coarsened intervals [ll,ul], distinct from two-sided
top/bottom coding. Limits never condition on selection into the sample.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import ModelFrame, build_result, column_list, information_criteria, kernel_call
from openecon.econometrics.count import common
from openecon.econometrics.count.censored_kernels import CensoredPieces
from openecon.econometrics.count.kernels import NegBinDensity, PoissonDensity
from openecon.econometrics.glm.common import (
    build_spec, optimizer_record, require_observations, require_terms, resolve_covariance,
    safe_exp, slope_indices,
)
from openecon.models import ModelSpec, ResultBundle

_MAX_COUNT = 2**53


@dataclass
class _Sample:
    frame: ModelFrame
    y: Tensor
    weights: Any
    offset: Tensor | None
    design: Any
    count: common.Block
    lower: Tensor
    upper: Tensor
    statuses: Tensor
    record: dict[str, Any]


def _bound(frame, option, role):
    columns = frame.role(role)
    if columns and option in frame.spec.options:
        raise AnalysisError("invalid_spec", f"Give {option} as a fixed point or a column, not both.")
    if columns:
        raw = frame.series(columns[0])
        if not pd.api.types.is_numeric_dtype(raw.dtype) or pd.api.types.is_bool_dtype(raw.dtype):
            raise AnalysisError("invalid_censoring_limit", f"{columns[0]} must contain integer count limits or missing values.")
        if bool((raw.dropna().abs() > _MAX_COUNT).any()):
            raise AnalysisError("invalid_censoring_limit", "Censoring limits must not exceed exact float64 count precision (2^53).")
        values = frame.numeric(columns[0], allow_missing=True)
        record = {f"{option}_column": columns[0]}
    else:
        value = frame.option(option)
        if value is not None and value > _MAX_COUNT:
            raise AnalysisError("invalid_censoring_limit", "Censoring limits must not exceed 2^53.")
        values = torch.full((frame.n,), math.nan if value is None else value, dtype=torch.float64)
        record = {option: value}
    finite = ~torch.isnan(values)
    if bool(((values[finite] < 0) | (values[finite] != values[finite].round())).any()):
        raise AnalysisError("invalid_censoring_limit", "Censoring limits must be nonnegative integers.")
    return values, record


def _prepare(spec, data):
    missing_limits = [*registry.role_columns(spec, "left_limit"), *registry.role_columns(spec, "right_limit")]
    # Irrelevant missing limits are
    # unbounded, not a reason to drop an otherwise observed model row.
    frame = ModelFrame(spec, data, allow_missing=missing_limits)
    y, weights, offset = common.count_sample(frame, spec.estimator, integer=True)
    if bool((pd.to_numeric(frame.series(spec.outcome)).abs() > _MAX_COUNT).any()):
        raise AnalysisError("invalid_count_outcome", "Observed counts must not exceed 2^53.")
    ll, left_record = _bound(frame, "ll", "left_limit")
    ul, right_record = _bound(frame, "ul", "right_limit")
    both = torch.isfinite(ll) & torch.isfinite(ul)
    columns = frame.role("censoring")
    if columns:
        values = frame.series(columns[0])
        mapping = {"exact": 0, "left": 1, "right": 2, "interval": 3}
        if not bool(values.isin(mapping).all()):
            raise AnalysisError("invalid_censoring_status", "censoring must contain only exact, left, right or interval.")
        statuses = torch.tensor(values.map(mapping).tolist(), dtype=torch.int64)
        needed_left = (statuses == 1) | (statuses == 3)
        needed_right = (statuses == 2) | (statuses == 3)
        if bool((needed_left & ~torch.isfinite(ll)).any()) or bool((needed_right & ~torch.isfinite(ul)).any()):
            raise AnalysisError("missing_censoring_limit", "Every censored event needs its applicable finite limit(s).")
        if bool((both & ((ll > ul) | ((ll == ul) & (statuses != 3)))).any()):
            raise AnalysisError("invalid_censoring_limits", "Limits must satisfy ll<ul, or ll<=ul for an explicit interval.")
        contradictory = ((statuses == 1) & (y > ll)) | ((statuses == 2) & (y < ul)) \
            | ((statuses == 3) & ((y < ll) | (y > ul)))
        if bool(contradictory.any()):
            raise AnalysisError("inconsistent_censoring", "The recorded count contradicts its declared censoring event.")
    else:
        if bool((both & (ll >= ul)).any()):
            raise AnalysisError("invalid_censoring_limits", "Auto-classified censoring requires ll<ul in every bounded row.")
        statuses = torch.zeros(frame.n, dtype=torch.int64)
        statuses[torch.isfinite(ll) & (y <= ll)] = 1
        statuses[torch.isfinite(ul) & (y >= ul)] = 2
    lower, upper = y.clone(), y.clone()
    lower[statuses == 1], upper[statuses == 1] = 0., ll[statuses == 1]
    lower[statuses == 2], upper[statuses == 2] = ul[statuses == 2], math.inf
    lower[statuses == 3], upper[statuses == 3] = ll[statuses == 3], ul[statuses == 3]
    if bool(((lower == 0) & torch.isinf(upper)).all()):
        raise AnalysisError("unidentified_censoring", "Every recorded event covers the entire nonnegative count support; the likelihood contains no parameter information.")
    if spec.intercept and (bool((lower == 0).all()) or bool(torch.isinf(upper).all())):
        raise AnalysisError("separation_detected", "The censored-count likelihood has no finite maximum: every event includes zero or every event is right-unbounded.")
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    require_terms(design)
    require_observations(weights.nobs, len(design.terms) + int(spec.estimator == "cnbreg"))
    count = common.design_block(design, weights.user, spec.outcome, offset)
    record = {**left_record, **right_record, "censoring_column": columns[0] if columns else None,
              "censoring_counts": {name: common.count_total(statuses == code, weights)
                                    for name, code in (("exact", 0), ("left", 1), ("right", 2), ("interval", 3))},
              "event_endpoints": "inclusive integer count bounds",
              "fitted_values": "latent unconditional mean E[Y|X]",
              "stata_parity_validated": False}
    return _Sample(frame, y, weights, offset, design, count, lower, upper, statuses, record)


def _flat(s, model, theta, hessian):
    k = s.count.size
    weak = common.flattest_information(-hessian[:k, :k], s.count.x, s.weights)
    if weak < common.FLAT_INFORMATION:
        log_events = model.objective.observations(theta)
        near_certain = (s.lower != s.upper) & (log_events > -1e-9)
        if bool(near_certain.any()):
            return "mean"
    if model.size > k and float(theta[k]) < math.log(common.ALPHA_BOUNDARY) \
            and common.flattest_information(-hessian[k:, k:], None, s.weights) < common.FLAT_INFORMATION:
        return "alpha"
    return None


def _run(s, model, starts):
    candidates, reasons, precision_error = [], set(), None
    pieces = model.objective.pieces
    def watch(iteration, theta, value):
        if iteration >= 15 and iteration % 5 == 0:
            kind = _flat(s, model, theta, model.objective(theta)[2])
            if kind:
                raise common.Boundary(theta.clone(), kind)
    for start in starts:
        try:
            kernel_call(pieces.check_precision, model.objective.indices(start))
        except AnalysisError as exc:
            if exc.code != "precision_unsupported":
                raise
            precision_error = exc
            continue
        pieces.precision_rejected = False
        pieces.reject_precision_trials = True
        try:
            attempt, stop = common.run(model, start, what=s.frame.spec.estimator, weights=s.weights, callback=watch)
        finally:
            pieces.reject_precision_trials = False
        if pieces.precision_rejected or model.objective.precision_rejected:
            precision_error = AnalysisError("precision_unsupported", "The NB iteration reached a count/shape scale outside the validated numerical domain; no unverified maximum is reported.")
        last = stop.theta if stop else attempt.theta
        kernel_call(pieces.check_precision, model.objective.indices(last))
        reason = stop.kind if stop else _flat(s, model, attempt.theta, attempt.hessian)
        reasons.add(reason)
        if attempt is not None and attempt.converged and reason is None:
            candidates.append(attempt)
    if candidates:
        return max(candidates, key=lambda result: result.value)
    if "alpha" in reasons:
        raise AnalysisError("boundary_solution", "cnbreg dispersion tends to zero and has no interior maximum; use oe.cpoisson.")
    if "mean" in reasons:
        raise AnalysisError("separation_detected", "A regressor isolates count events whose probability tends to one; the censored likelihood has no finite coefficient estimate.")
    if precision_error is not None:
        raise precision_error
    raise common.not_converged(s.frame.spec.estimator, "the Newton iteration did not identify a finite interior maximum")


def _estimate(s, dispersion=None):
    # Starting values use the observed proxy only; the fitted likelihood always
    # uses the full event. Centering is exactly the same as ordinary count ML.
    poisson_pieces = kernel_call(CensoredPieces, PoissonDensity(), s.lower, s.upper)
    # Validate static NB endpoint precision before attempting a starting fit.
    nb_pieces = kernel_call(CensoredPieces, NegBinDensity(dispersion), s.lower, s.upper) if dispersion else None
    initial = common.poisson_start(s.frame, s.design, s.y, s.weights, s.offset, s.frame.spec.estimator)
    poisson_model = common.Model(poisson_pieces, [s.count], s.weights.user)
    poisson = _run(s, poisson_model, [initial.state.beta])
    if dispersion is None:
        return poisson_model, poisson, poisson
    ancillary = common.scalar_block("/lnalpha" if dispersion == "mean" else "/lndelta")
    model = common.Model(nb_pieces, [s.count, ancillary], s.weights.user)
    starts = [torch.cat([poisson.theta, torch.tensor([math.log(value)], dtype=torch.float64)])
              for value in (.2, 1., 3.)]
    return model, _run(s, model, starts), poisson


def _report(s, model, fitted, poisson, dispersion):
    frame, spec, k = s.frame, s.frame.spec, s.count.size
    theta, covariance, inference = common.covariance_of(frame, model, fitted.theta, fitted.hessian, s.weights)
    null = None
    if spec.intercept:
        if k == 1:
            null = fitted.value
        else:
            constant = common.constant_block(frame.n, spec.outcome, s.offset)
            blocks = [constant, *model.blocks[1:]]
            reduced = common.Model(model.objective.pieces, blocks, s.weights.user)
            start = torch.cat([fitted.theta[:1], fitted.theta[k:]])
            try:
                kernel_call(reduced.objective.pieces.check_precision, reduced.objective.indices(start))
                attempt, stop = common.run(reduced, start, what=f"{spec.estimator} null model", weights=s.weights)
                if attempt is not None and attempt.converged:
                    kernel_call(reduced.objective.pieces.check_precision, reduced.objective.indices(attempt.theta))
                    null = attempt.value
                else:
                    frame.warn("The constant-only comparison model did not converge; the model Wald test is reported.")
            except AnalysisError as exc:
                if exc.code != "precision_unsupported":
                    raise
                frame.warn("The constant-only comparison model exceeded the validated count/shape precision domain; the model Wald test is reported.")
    tests = {"model": common.model_test(frame, theta, covariance, slope_indices(s.design.terms),
                                       fitted.value, null, "count")}
    metrics = {"log_likelihood": fitted.value, "pseudo_r_squared": common.pseudo_r_squared(fitted.value, null),
               **information_criteria(fitted.value, model.size, s.weights.nobs)}
    record = {**s.record, "null_log_likelihood": null, "tail_evaluation": "log-domain gamma / analytic incomplete-beta continued fraction"}
    if dispersion is not None:
        name = "alpha" if dispersion == "mean" else "delta"
        metrics[name] = safe_exp(float(theta[k]))
        record.update({"dispersion": dispersion, "variance_function": "mu+alpha*mu^2" if dispersion == "mean" else "mu*(1+delta)",
                       "alpha": common.exponentiated(float(theta[k]), math.sqrt(float(covariance[k, k])), spec.alpha, name),
                       "cpoisson_log_likelihood": poisson.value})
        if spec.covariance in {"nonrobust", "opg"}:
            tests["alpha"] = common.boundary_test(fitted.value, poisson.value, name, "censored Poisson")
    mu = torch.exp(model.objective.indices(fitted.theta)[0])
    return build_result(frame, terms=model.terms, params=theta, covariance=covariance,
                        equations=model.equations if dispersion else None, use_t=False,
                        metrics=metrics, fitted=mu, nobs=s.weights.nobs, inference=inference, tests=tests,
                        extra=record, categories=s.design.categories, solver="newton_analytic_observed_hessian",
                        solver_diagnostics={"converged": True, "iterations": fitted.iterations},
                        optimizer=optimizer_record(fitted), df_resid=s.weights.nobs - model.size)


def fit_cpoisson(spec: ModelSpec, data: Any) -> ResultBundle:
    s = _prepare(spec, data)
    return _report(s, *_estimate(s), None)


def fit_cnbreg(spec: ModelSpec, data: Any) -> ResultBundle:
    s = _prepare(spec, data)
    dispersion = s.frame.option("dispersion")
    return _report(s, *_estimate(s, dispersion), dispersion)


def _limits(ll, ul):
    columns, options = {}, {}
    for name, value, role in (("ll", ll, "left_limit"), ("ul", ul, "right_limit")):
        if value is None:
            continue
        if isinstance(value, str):
            columns[role] = value
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) \
                or value < 0 or value != int(value) or value > _MAX_COUNT:
            raise AnalysisError("invalid_censoring_limit", f"{name} must be a nonnegative integer <=2^53 or an endpoint column name.")
        else:
            options[name] = int(value)
    return columns, options


def _spec(command, data, y, x, *, ll, ul, censoring, covariance, cluster, weights, weight_type,
          offset, exposure, categorical, intercept, missing, alpha, dispersion=None):
    columns, options = _limits(ll, ul)
    spec = build_spec(command, outcome=y, predictors=column_list(x, "x"),
                      categorical=column_list(categorical, "categorical"), intercept=intercept,
                      covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
                      weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
                      columns={**columns, "censoring": censoring, "offset": offset, "exposure": exposure},
                      options={**options, **({"dispersion": dispersion} if dispersion else {})})
    return fit(spec, data=data)


def cpoisson(*, data: Any, y: str, x: Sequence[str], ll: int | str | None = None,
             ul: int | str | None = None, censoring: str | None = None,
             covariance: str | None = None, cluster: str | Sequence[str] | None = None,
             weights: str | None = None, weight_type: str | None = None,
             offset: str | None = None, exposure: str | None = None,
             categorical: Sequence[str] | None = None, intercept: bool = True,
             missing: str = "raise", alpha: float = .05) -> ResultBundle:
    """Poisson ML for exact/left/right/interval count events; ll/ul accept numbers or columns.

    Omit censoring to classify y<=ll as left and y>=ul as right. A status column
    contains exact, left, right or interval; interval denotes ll<=Y<=ul.
    Fitted values are the latent mean, not the recorded top-coded count.
    """
    return _spec("cpoisson", data, y, x, ll=ll, ul=ul, censoring=censoring,
                 covariance=covariance, cluster=cluster, weights=weights, weight_type=weight_type,
                 offset=offset, exposure=exposure, categorical=categorical, intercept=intercept,
                 missing=missing, alpha=alpha)


def cnbreg(*, data: Any, y: str, x: Sequence[str], ll: int | str | None = None,
           ul: int | str | None = None, censoring: str | None = None, dispersion: str = "mean",
           covariance: str | None = None, cluster: str | Sequence[str] | None = None,
           weights: str | None = None, weight_type: str | None = None,
           offset: str | None = None, exposure: str | None = None,
           categorical: Sequence[str] | None = None, intercept: bool = True,
           missing: str = "raise", alpha: float = .05) -> ResultBundle:
    """Censored NB2 (mean) or NB1 (constant) ML; zero dispersion raises boundary_solution."""
    return _spec("cnbreg", data, y, x, ll=ll, ul=ul, censoring=censoring, dispersion=dispersion,
                 covariance=covariance, cluster=cluster, weights=weights, weight_type=weight_type,
                 offset=offset, exposure=exposure, categorical=categorical, intercept=intercept,
                 missing=missing, alpha=alpha)
