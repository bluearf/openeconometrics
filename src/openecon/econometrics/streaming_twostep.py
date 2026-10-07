"""Exact native selection and endogenous two-step estimators over replayed rows."""
from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import cholesky_inverse, least_squares
from openecon.engines.optimize import information_inverse, maximize_newton
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.streaming_design import MAX_PARAMETERS
from .core import ModelFrame, build_result, wald_test
from .streaming_likelihood import (
    ReplayObjective, _binary_y, _censored_bounds, _linear_start, _option,
    _require_variance, _role,
)


def _fit(sample, builder, start):
    objective = ReplayObjective(sample, builder, len(start))
    scale = sample.optimization_scale
    run = maximize_newton(lambda theta: tuple(piece*scale for piece in objective(theta)), start,
                          value_fn=lambda theta: objective.value(theta)*scale,
                          max_iter=200, step_tol=1e-10, scaled_gradient_tol=1e-10,
                          raise_on_failure=False)
    if not run.converged:
        raise AnalysisError("nonconvergence", f"The global two-step likelihood did not converge: {run.diagnostics.get('message')}")
    run.value /= scale
    run.hessian /= scale
    return run


def _binary_certificate(sample, matrix, response, width):
    from openecon.engines.separation import certify_separation

    def replay():
        for batch in sample.batches():
            yield matrix(batch)*torch.where(response(batch) == 1, 1., -1.)[:, None]
    total, count = _CompensatedSum((width,)), 0
    for values in replay():
        total.add(values.sum(0))
        count += len(values)
    record = certify_separation(replay, total.value/count, width)
    sample.notes.setdefault("separation_certificates", {})["twostep_control"] = record


def _inverse_design(sample, matrix, width, selector=None):
    tree = _TSQRTree()
    for batch in sample.batches():
        x = matrix(batch)
        selected = selector(batch) if selector else torch.ones(len(x), dtype=torch.bool)
        if bool(selected.any()):
            tree.add(qr_factor(x[selected]*batch.weights[selected].sqrt()[:, None]))
    factor = tree.finish()
    if factor.shape[1] != width:
        raise AnalysisError("invalid_design", "The replayed two-step design changed width.")
    return least_squares(factor, torch.zeros(len(factor), dtype=torch.float64), drop_collinear=False).xtx_inv


def _finish(sample, terms, params, covariance, *, metrics, extra, inference, tests,
            matrix, theta, response, prediction, solver, iterations=0, equations=None, title=None):
    fitted, observed, remaining = [], [], 400
    for batch in sample.batches():
        values = prediction(matrix(batch)@theta)
        actual = response(batch)
        take = min(remaining, len(batch.frame))
        if take:
            fitted.append(values[:take].clone())
            observed.append(actual[:take].clone())
            remaining -= take
    frame = ModelFrame(sample.requested_spec, sample.sample, allow_missing=sample.allow_missing)
    result = build_result(frame, terms=terms, params=params, covariance=covariance, equations=equations,
                          nobs=sample.nobs, use_t=False, title=title, metrics=metrics, tests=tests,
                          extra=extra, inference=inference, categories=sample.categories,
                          warnings=sample.notes.get("warnings", ()),
                          fitted=torch.cat(fitted), observed=torch.cat(observed), solver=solver,
                          solver_diagnostics={"converged": True, "iterations": iterations,
                                              "dense_observation_matrix": False},
                          provenance={**sample.provenance(), "prediction_sample": "first400 retained observations",
                                      "prediction_definition": "selection probability" if sample.spec.estimator == "heckman"
                                      else "structural outcome prediction from the global Newey coefficients",
                                      "separation_certificates": sample.notes.get("separation_certificates", {})})
    for row in result.predictions:
        row["row"] = sample.sample_positions[int(row["row"])]
    result.nobs_original, result.dropped_rows = sample.original_count, sample.original_count-sample.nrows
    result.sample_positions = []
    return result


def _heckman(sample):
    from .discrete.kernels import HetprobitObjective, mills_ratio

    spec = sample.spec
    mean, selection = sample.designs["mean"], sample.designs["selection"]
    k, q = len(mean.terms), len(selection.terms)
    if k+q+1 > MAX_PARAMETERS:
        raise AnalysisError("model_too_wide", "Heckman two-step exceeds the bounded total parameter width.")
    sample.plan_rows("replay Heckman two-step", {"global_covariance_factors": 128*(k+q+1)**2},
                     256*(k+q+5))
    def chosen(batch):
        return _binary_y(batch, _role(spec, "select")[0]) == 1
    def probit_builder(batch):
        x = batch.designs["selection"]
        return HetprobitObjective(x, torch.empty((len(x), 0), dtype=torch.float64),
                                 chosen(batch).to(torch.float64), batch.weights)
    probit = _fit(sample, probit_builder, torch.zeros(q, dtype=torch.float64))
    v_probit = information_inverse(-probit.hessian)
    def augmented(batch):
        mills = mills_ratio(batch.designs["selection"]@probit.theta)[1]
        return torch.cat((batch.designs["mean"], mills[:, None]), 1)
    def target(batch):
        return batch.numeric(spec.outcome, allow_missing=True)
    try:
        beta, variance = _linear_start(sample, "mean", target, chosen, matrix=augmented)
        inverse = _inverse_design(sample, augmented, k+1, chosen)
    except Exception as error:
        if getattr(error, "code", None) != "singular_design":
            raise
        raise AnalysisError("collinear_mills_ratio", "The global inverse Mills ratio is collinear with the selected outcome regressors.") from error
    _require_variance(sample, variance, target, chosen)
    mass, delta_total = _CompensatedSum(()), _CompensatedSum(())
    for batch in sample.batches():
        on = chosen(batch)
        _, _, delta = mills_ratio(batch.designs["selection"][on]@probit.theta)
        mass.add(batch.weights[on].sum())
        delta_total.add(batch.weights[on]@delta)
    n_selected = float(mass.value)
    if n_selected <= k+1:
        raise AnalysisError("insufficient_observations", "Heckman two-step needs more selected observations than outcome and Mills parameters.")
    slope = float(beta[k])
    sigma = math.sqrt(variance+slope*slope*float(delta_total.value)/n_selected)
    rho = slope/sigma
    truncated = abs(rho) > 1
    if truncated:
        rho, sigma = math.copysign(1., rho), abs(slope)
    inner, link = _CompensatedSum((k+1, k+1)), _CompensatedSum((k+1, q))
    for batch in sample.batches():
        on = chosen(batch)
        x, z, weights = augmented(batch)[on], batch.designs["selection"][on], batch.weights[on]
        _, _, delta = mills_ratio(z@probit.theta)
        inner.add(x.T@(x*(weights*(1-rho*rho*delta))[:, None]))
        link.add(x.T@(z*(weights*delta)[:, None]))
    outcome = sigma*sigma*(inverse@(inner.value+rho*rho*link.value@v_probit@link.value.T)@inverse)
    cross = slope*inverse@link.value@v_probit
    size = k+q+1
    params = torch.cat((beta[:k], probit.theta, beta[k:]))
    covariance = torch.zeros((size, size), dtype=torch.float64)
    order = torch.tensor([*range(k), size-1], dtype=torch.int64)
    covariance[order[:, None], order] = (outcome+outcome.T)/2
    covariance[k:k+q, k:k+q] = v_probit
    covariance[order[:, None], torch.arange(k, k+q)] = cross
    covariance[torch.arange(k, k+q)[:, None], order] = cross.T
    transform = torch.block_diag(mean.transform, selection.transform, torch.ones((1, 1), dtype=torch.float64))
    params, covariance = transform@params, transform@covariance@transform.T
    n_selected = int(n_selected) if spec.weight_type == "fweight" else int(round(n_selected))
    n_other = sample.nobs-n_selected
    if truncated:
        sample.notes["warnings"] = ["The two-step rho was outside [-1, 1]; rho is truncated and sigma is set to |lambda| (rhosigma)."]
    slopes = [i for i, term in enumerate(mean.terms) if term != "Intercept"]
    tests = {"model": wald_test(params, covariance, slopes, label="Wald chi2 test of the outcome slopes")} if slopes else {}
    extra = {"select": spec.columns["select"], "selection_terms": selection.terms,
             "n_selected": n_selected, "n_nonselected": n_other, "method": "twostep",
             "rho_truncated": truncated, "probit_log_likelihood": probit.value,
             "covariance": "Heckman (1979) two-step covariance; probit covariance for the selection equation"}
    return _finish(sample, [*mean.terms, *selection.terms, "mills:lambda"], params, covariance,
                   metrics={"rho": rho, "sigma": sigma, "lambda": slope, "n_selected": n_selected, "n_censored": n_other},
                   extra=extra, inference={"covariance": "nonrobust", "df_inference": None,
                   "correction": "Heckman two-step: sigma^2 (X*'X*)^-1 [X*'(I-rho^2 D)X*+Q] (X*'X*)^-1"}, tests=tests,
                   matrix=lambda b: b.designs["selection"], theta=probit.theta,
                   response=lambda b: chosen(b).to(torch.float64), prediction=torch.special.ndtr,
                   solver="torch_replayed_probit_then_tsqr_heckman", iterations=probit.iterations,
                   equations=[spec.outcome]*k+["select"]*q+["mills"], title="Heckman selection model (two-step)")


def _newey(sample):
    from .discrete.kernels import HetprobitObjective
    from .limited.kernels import CensoredObjective
    from .streaming_likelihood_diagnostics import _first_stage

    spec, name = sample.spec, sample.spec.estimator
    mean, instruments, endogenous = sample.designs["mean"], sample.designs["instruments"], sample.designs["endogenous"]
    k, kz, p = len(mean.terms), len(instruments.terms), len(endogenous.terms)
    if p != len(_role(spec, "endogenous")) or kz-k < p or instruments.terms[:k] != mean.terms:
        raise AnalysisError("underidentified", "The globally retained instruments must identify every endogenous regressor.")
    if sample.nobs <= kz+p+1:
        raise AnalysisError("insufficient_observations", "Newey two-step needs more observations than first-stage and control-function parameters.")
    width = kz+p+int(name == "ivtobit")
    if width > MAX_PARAMETERS:
        raise AnalysisError("model_too_wide", "The Newey auxiliary likelihood exceeds the bounded parameter width.")
    sample.plan_rows("replay Newey minimum chi-squared", {"global_information_factors": 256*(width+1)**2}, 512*(width+1))
    centres = endogenous.means[endogenous.kept]/endogenous.scales[endogenous.kept] if mean.intercept else torch.zeros(p, dtype=torch.float64)
    def y2(batch):
        return batch.designs["endogenous"]-centres
    first = []
    for j in range(p):
        beta, variance = _linear_start(sample, "instruments", lambda b, j=j: y2(b)[:, j])
        _require_variance(sample, variance, lambda b, j=j: y2(b)[:, j])
        first.append(beta)
    first = torch.stack(first, 1)
    def residuals(batch):
        return y2(batch)-batch.designs["instruments"]@first
    def structural(batch):
        return torch.cat((batch.designs["mean"], y2(batch)), 1)
    def reduced_matrix(batch):
        return torch.cat((batch.designs["instruments"], residuals(batch)), 1)
    def conditional_matrix(batch):
        return torch.cat((structural(batch), residuals(batch)), 1)

    def outcome_fit(matrix, columns):
        if name == "ivprobit":
            def response(batch):
                return _binary_y(batch, spec.outcome)
            _binary_certificate(sample, matrix, response, columns)
            def builder(batch):
                x = matrix(batch)
                return HetprobitObjective(x, torch.empty((len(x), 0), dtype=torch.float64), response(batch), batch.weights)
            start = torch.zeros(columns, dtype=torch.float64)
        else:
            def builder(batch):
                return CensoredObjective(matrix(batch), *_censored_bounds(batch, spec), batch.weights)
            beta, variance = _linear_start(sample, "instruments", lambda b: b.numeric(spec.outcome), matrix=matrix)
            _require_variance(sample, variance, lambda b: b.numeric(spec.outcome))
            start = torch.cat((beta, torch.tensor([.5*math.log(variance)], dtype=torch.float64)))
        run = _fit(sample, builder, start)
        return run.theta, information_inverse(-run.hessian), run.value

    reduced, reduced_cov, _ = outcome_fit(reduced_matrix, kz+p)
    conditional, conditional_cov, _ = outcome_fit(conditional_matrix, k+2*p)
    alpha, lam, beta = reduced[:kz], reduced[kz:kz+p], conditional[k:k+p]
    _, auxiliary_variance = _linear_start(sample, "instruments", lambda b: y2(b)@(lam-beta))
    inverse = _inverse_design(sample, lambda b: b.designs["instruments"], kz)
    mass = sample.nobs if spec.weight_type == "fweight" else sample.nrows
    omega = reduced_cov[:kz, :kz]+inverse*(auxiliary_variance*mass/(sample.nobs-kz))
    distance = torch.zeros((kz, k+p), dtype=torch.float64)
    distance[:k, :k] = torch.eye(k, dtype=torch.float64)
    distance[:, k:] = first
    try:
        omega_inverse = cholesky_inverse(omega)
        covariance = cholesky_inverse(distance.T@omega_inverse@distance)
    except Exception as error:
        if getattr(error, "code", None) not in {"singular_design", "singular_covariance", "singular_matrix", "not_positive_definite"}:
            raise
        raise AnalysisError("singular_covariance", "The global minimum chi-squared weighting matrix does not identify the endogenous coefficients.") from error
    theta = covariance@(distance.T@(omega_inverse@alpha))
    transform = torch.block_diag(mean.transform, endogenous.transform)
    if mean.intercept:
        transform[0, k:] = -centres
    params, covariance = transform@theta, transform@covariance@transform.T
    terms = [*mean.terms, *endogenous.terms]
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    tests = {"model": wald_test(params, covariance, slopes, label="Wald chi2 test of the slopes"),
             "exogeneity": wald_test(conditional, conditional_cov, range(k+p, k+2*p),
                                     label="Wald test of exogeneity (first-stage residual coefficients = 0)")}
    extra = {"method": "twostep", "endogenous": _role(spec, "endogenous"),
             "instruments": instruments.terms[k:], "n_endogenous": p,
             "covariance": "Newey (1987) minimum chi-squared: (D' Omega^-1 D)^-1"}
    _first_stage(sample, extra)
    metrics = {}
    if name == "ivprobit":
        positives, total = _CompensatedSum(()), _CompensatedSum(())
        for batch in sample.batches():
            positives.add(batch.weights@_binary_y(batch, spec.outcome))
            total.add(batch.weights.sum())
        extra.update({"zero_outcomes": float(total.value-positives.value), "nonzero_outcomes": float(positives.value),
                      "normalization": "coefficients of the index conditional on the first-stage errors; ML coefficients differ by 1 / sd(u | v)"})
    else:
        counts = torch.zeros(3, dtype=torch.float64)
        for batch in sample.batches():
            low, high = _censored_bounds(batch, spec)
            frequency = batch.weights if spec.weight_type == "fweight" else torch.ones_like(low)
            left, right = torch.isneginf(low), torch.isposinf(high)
            counts += torch.stack((frequency[left].sum(), frequency[~left&~right].sum(), frequency[right].sum()))
        if counts[1] == 0:
            raise AnalysisError("no_uncensored_observations", "The global endogenous Tobit sample has no uncensored observations.")
        metrics.update(zip(("n_left_censored", "n_uncensored", "n_right_censored"), counts.tolist(), strict=True))
        extra["limits"] = {"lower": _option(spec, "ll"), "upper": _option(spec, "ul")}
    return _finish(sample, terms, params, covariance, metrics=metrics, extra=extra,
                   inference={"covariance": "nonrobust", "df_inference": None,
                              "correction": "Newey minimum chi-squared two-step: (D' Omega^-1 D)^-1 with Omega = J_aa^-1 + s^2 (Z'Z)^-1"},
                   tests=tests, matrix=structural, theta=theta,
                   response=lambda b: b.numeric(spec.outcome),
                   prediction=torch.special.ndtr if name == "ivprobit" else lambda value: value,
                   solver="torch_replayed_newey_minimum_chi_squared", title=f"{name} (two-step)")


def fit_twostep(sample):
    spec = sample.spec
    if spec.covariance != "nonrobust":
        raise AnalysisError("unsupported_covariance", "Native two-step estimators have their own consistent covariance; use nonrobust or ML for robust/cluster covariance.")
    if spec.weight_type not in {None, "fweight"}:
        raise AnalysisError("unsupported_weights", "Native two-step estimators accept frequency weights only.")
    return _heckman(sample) if spec.estimator == "heckman" else _newey(sample)
