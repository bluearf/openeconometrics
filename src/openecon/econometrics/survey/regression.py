"""Weighted survey regressions with complete-design Taylor score covariance.

Fits use resident CPU float64. Normalizing sampling weights inside the equations
makes coefficients and design covariance invariant to a common weight scale.
Complete-design PSU totals retain domain-excluded and missing-row zero scores.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.separation import certify_separation
from .common import FLOAT, finite, number
from .regression_common import (
    MAX_WORK, SurveyRegressionResult, admit, build_result, check_work, cpu_call,
)

_MAX_ITERATIONS = 1000
_LOG_NORMAL_CONSTANT = 0.5 * math.log(2.0 * math.pi)
_EPSILON = torch.finfo(FLOAT).eps


def _options(max_iter, tolerance):
    if type(max_iter) is not int or not 1 <= max_iter <= _MAX_ITERATIONS:
        raise AnalysisError(
            "invalid_survey_option", "max_iter must be an integer from 1 to 1000."
        )
    return max_iter, number(tolerance, "tolerance", 1e-14, 1e-3)


def _work_budget(sample, family, max_iter):
    width = len(sample.labels)
    n = int(sample.selected.sum())
    lp_constraints = max(8, 4 * width) + 2 * width
    diagnostic = 0 if family == "linear" else (
        64 * n * width * 3 + 64 * 100 * lp_constraints * width * width
    )
    overhead = check_work(sample, 0)
    planned = check_work(sample, 1 if family == "linear" else max_iter)
    if planned + diagnostic > MAX_WORK:
        raise AnalysisError(
            "survey_regression_budget",
            "Fit and bounded separation diagnostics exceed 50 million work units before execution.",
        )
    return overhead, diagnostic, n * width * width


def _weights(sample):
    selected = sample.weights[sample.selected]
    # Divide before summation to avoid overflow from a representable common scale.
    relative = selected / selected.max()
    weights = finite(relative * (len(selected) / relative.sum()), "Normalized survey weights")
    if bool((weights <= 0).any()):
        raise AnalysisError(
            "survey_numerical_failure",
            "Sampling weight ratios underflowed; narrow their numerical range explicitly.",
        )
    return weights


def _scaled_design(sample, weights):
    original = sample.X[sample.selected]
    scales = original.abs().amax(0)
    if bool((scales == 0).any()):
        raise AnalysisError(
            "survey_regression_rank", "Every coefficient needs an identified nonzero design column."
        )
    design = finite(original / scales, "Scaled survey regression design")
    singular = torch.linalg.svdvals(design * torch.sqrt(weights)[:, None])
    threshold = 10 * _EPSILON * max(design.shape) * float(singular[0])
    if len(singular) != design.shape[1] or float(singular[-1]) <= threshold:
        raise AnalysisError(
            "survey_regression_rank",
            "The weighted design is rank deficient; remove redundant columns explicitly.",
        )
    return design, scales


def _solve(sensitivity, score):
    sensitivity = finite((sensitivity + sensitivity.T) * 0.5, "Survey regression sensitivity")
    factor, info = torch.linalg.cholesky_ex(sensitivity)
    if int(info) != 0:
        raise AnalysisError(
            "survey_regression_rank",
            "The weighted sensitivity is not positive definite; no ridge or pseudoinverse was used.",
        )
    return finite(
        torch.cholesky_solve(score[:, None], factor).squeeze(1), "Survey regression Newton step"
    )


def _separation_error(family):
    raise AnalysisError(
        "survey_regression_separation",
        f"The {family} equations have a separating direction and no identified finite fit.",
    )


def _certify(replay, objective, width, family):
    try:
        return certify_separation(replay, objective, width, max_passes=64)
    except KernelError as error:
        if error.code == "separation_detected":
            _separation_error(family)
        raise AnalysisError("survey_separation_check_failed", str(error)) from error


def _binary_check(design, response, family):
    signed = design * (2 * response - 1)[:, None]
    signed = signed / signed.abs().amax(1).clamp_min(torch.finfo(FLOAT).tiny)[:, None]

    def replay():
        for start in range(0, len(signed), 4096):
            yield signed[start : start + 4096]

    return _certify(replay, signed.mean(0), design.shape[1], family)


def _poisson_check(design, response):
    if not bool((response == 0).any()):
        return {
            "separation_diagnostic": "all count responses positive; full weighted design rank",
        }
    normalized = design / design.abs().amax(1).clamp_min(torch.finfo(FLOAT).tiny)[:, None]
    positive, negative_zero = normalized[response > 0], -normalized[response == 0]

    def replay():
        # The paired positive-response constraints require Xd=0; zero-count
        # constraints require Xd<=0. A positive objective witnesses strict
        # reduction of at least one zero-count mean with no positive-count loss.
        for start in range(0, len(positive), 4096):
            block = positive[start : start + 4096]
            yield block
            yield -block
        for start in range(0, len(negative_zero), 4096):
            yield negative_zero[start : start + 4096]

    count = 2 * len(positive) + len(negative_zero)
    objective = negative_zero.sum(0) / count
    metadata = _certify(replay, objective, design.shape[1], "poisson")
    metadata["separation_constraints"] = (
        "Xd=0 on positive responses; Xd<=0 on zero responses; paired positive constraints"
    )
    return metadata


def _components(design, response, weights, coefficient, family):
    """Exact weighted objective, score factor, and observed positive sensitivity."""
    index = finite(design @ coefficient, "Survey regression linear predictor")
    if family == "logit":
        row_objective = torch.nn.functional.logsigmoid((2 * response - 1) * index)
        score_factor = torch.where(
            response == 1, torch.sigmoid(-index), -torch.sigmoid(index)
        )
        # This product retains tiny curvature on either side of the logit link.
        curvature = torch.sigmoid(index) * torch.sigmoid(-index)
    elif family == "probit":
        sign = 2 * response - 1
        signed_index = sign * index
        if bool((signed_index < -1e4).any()):
            raise AnalysisError(
                "survey_numerical_failure",
                "The probit lower-tail index exceeds stable observed-curvature precision.",
            )
        row_objective = torch.special.log_ndtr(signed_index)
        lower_mills = math.sqrt(2 / math.pi) / torch.special.erfcx(
            -signed_index.clamp_max(0) / math.sqrt(2)
        )
        upper_mills = torch.exp(
            -0.5 * signed_index.square() - _LOG_NORMAL_CONSTANT - row_objective
        )
        mills = torch.where(signed_index < 0, lower_mills, upper_mills)
        score_factor = sign * mills
        curvature = mills * (mills + signed_index)
        if bool((curvature < 0).any()):
            raise AnalysisError(
                "survey_numerical_failure", "The probit observed curvature lost numerical precision."
            )
    else:
        mean = finite(torch.exp(index), "Poisson mean")
        if bool((mean <= 0).any()):
            raise AnalysisError(
                "survey_numerical_failure",
                "The Poisson mean underflowed; rescale the numerical input explicitly.",
            )
        # Remove the parameter-independent saturated likelihood. This avoids
        # cancellation of factorial-sized terms for representable large counts.
        delta = index - torch.log(response.clamp_min(1))
        row_objective = torch.where(
            response > 0, response * (delta - torch.expm1(delta)), -mean
        )
        score_factor = response - mean
        curvature = mean
    objective = finite((weights * row_objective).sum(), "Weighted survey regression objective")
    score_factor = finite(score_factor, "Survey regression score factor")
    curvature = finite(curvature, "Survey regression observed curvature")
    score = finite(design.T @ (weights * score_factor), "Survey regression score")
    sensitivity = finite(
        (design.T * (weights * curvature)) @ design, "Survey regression observed sensitivity"
    )
    return float(objective), score, sensitivity, score_factor


def _finish(
    sample, coefficient, scales, sensitivity, score_factor, weights, convergence,
):
    beta = finite(coefficient / scales, "Survey regression coefficients")
    bread = finite(
        scales[:, None] * sensitivity * scales[None, :],
        "Survey regression original-unit sensitivity",
    )
    _, info = torch.linalg.cholesky_ex((bread + bread.T) * 0.5)
    if int(info) != 0:
        raise AnalysisError(
            "survey_regression_rank",
            "Original-unit sensitivity lost positive definiteness; rescale columns explicitly.",
        )
    rows = torch.zeros_like(sample.X)
    rows[sample.selected] = finite(
        sample.X[sample.selected] * (weights * score_factor)[:, None],
        "Full-row weighted survey regression score",
    )
    return {"beta": beta, "bread": bread, "score_rows": rows, "convergence": convergence}


def _inference_options(sample, alpha, null):
    alpha = number(alpha, "alpha", 1e-8, 1-1e-8)
    if isinstance(null, (list, tuple)):
        if len(null) != len(sample.labels):
            raise AnalysisError(
                "invalid_survey_option", "One null value per coefficient is required."
            )
        null = [number(value, "null") for value in null]
    else:
        null = number(null, "null")
    return alpha, null


@cpu_call
def fit_sample(sample, family, *, max_iter=100, tolerance=1e-9, budget=None):
    """Solve coefficients on an admitted physical sample, without covariance.

    Zero sampling weights may be represented by deselected original rows.
    The caller retains the design and parameter order. An optional cumulative
    ledger exposes reserve(units), charged before diagnostics and every full
    score/Hessian evaluation, including rejected step-halving candidates.
    """
    max_iter, tolerance = _options(max_iter, tolerance)
    if family not in {"linear", "logit", "probit", "poisson"}:
        raise AnalysisError("invalid_survey_option", "Unknown survey regression family.")
    if int(sample.selected.sum()) <= len(sample.labels):
        raise AnalysisError(
            "survey_regression_rank", "More positive-weight admitted rows than coefficients are required."
        )
    if bool((sample.weights[sample.selected] <= 0).any()):
        raise AnalysisError("invalid_survey_replicates", "Selected fit rows require positive weights.")
    overhead, diagnostic_work, evaluation_cost = _work_budget(sample, family, max_iter)
    if budget is not None:
        budget.reserve(overhead + diagnostic_work)
    weights = _weights(sample)
    matrix, scales = _scaled_design(sample, weights)
    response = sample.y[sample.selected]
    n = len(response)
    if family == "linear":
        if budget is not None:
            budget.reserve(evaluation_cost)
        sensitivity = finite((matrix.T * weights) @ matrix, "Weighted linear sensitivity")
        coefficient = _solve(sensitivity, matrix.T @ (weights * response))
        residual = finite(response - matrix @ coefficient, "Survey regression residual")
        objective = float(
            finite((weights * residual.square()).sum(), "Weighted residual sum of squares")
        )
        score = finite(matrix.T @ (weights * residual), "Weighted linear score")
        convergence = dict(
            converged=True, iterations=1, tolerance=tolerance, max_iter=max_iter,
            score_max_abs=float(score.abs().max()) / n, objective=objective,
            newton_steps=0, score_evaluations=1,
            work_units=overhead+evaluation_cost, max_work_units=MAX_WORK,
            diagnostic_work_allowance=0,
            algorithm="column-scaled weighted normal equations; strict Cholesky solve",
        )
        return _finish(
            sample, coefficient, scales, sensitivity, residual, weights, convergence,
        )
    if family in ("logit", "probit"):
        if bool(((response != 0) & (response != 1)).any()):
            raise AnalysisError(
                "invalid_survey_response", "Binary survey regressions require exact 0/1 responses."
            )
        if sample.intercept and bool((response == response[0]).all()):
            _separation_error(family)
    elif sample.metadata.get("raw_outcome_count_admissible") is not True or bool(
        ((response < 0) | (response != torch.floor(response)) | (response > 2**53)).any()
    ):
        raise AnalysisError(
            "invalid_survey_response", "Poisson responses must be exact nonnegative counts at most 2^53."
        )
    elif sample.intercept and not bool((response > 0).any()):
        _separation_error(family)
    separation = (
        _binary_check(matrix, response, family) if family in ("logit", "probit")
        else _poisson_check(matrix, response)
    )
    evaluations = 0

    def evaluate(coefficient):
        nonlocal evaluations
        work = overhead + diagnostic_work + (evaluations + 1) * evaluation_cost
        if work > MAX_WORK:
            raise AnalysisError(
                "survey_regression_budget",
                "Actual score/Hessian evaluations including step halving exceed 50 million work units.",
            )
        if budget is not None:
            budget.reserve(evaluation_cost)
        evaluations += 1
        return _components(matrix, response, weights, coefficient, family)

    coefficient = torch.zeros(matrix.shape[1], dtype=FLOAT)
    if sample.intercept:
        if family == "logit":
            coefficient[0] = (
                math.log(float(weights[response == 1].sum()))
                - math.log(float(weights[response == 0].sum()))
            )
        elif family == "probit":
            fraction = weights[response == 1].sum() / weights.sum()
            other = weights[response == 0].sum() / weights.sum()
            coefficient[0] = (
                torch.special.ndtri(fraction) if float(fraction) <= 0.5
                else -torch.special.ndtri(other)
            )
        else:
            mean = float((weights * response).sum() / weights.sum())
            coefficient[0] = math.log(mean)
    iterations = 0
    for iteration in range(max_iter + 1):
        objective, score, sensitivity, factor = evaluate(coefficient)
        direction = _solve(sensitivity, score)
        gradient = float(score.abs().max()) / n
        gradient_scale = max(1.0, float((weights * response.abs()).sum()) / n)
        step_small = float(direction.abs().max()) <= math.sqrt(tolerance) * (
            1 + float(coefficient.abs().max())
        )
        if gradient <= tolerance * gradient_scale and step_small:
            convergence = dict(
                converged=True, iterations=max(1, iterations), newton_steps=iterations,
                score_evaluations=evaluations, tolerance=tolerance, max_iter=max_iter,
                score_max_abs=gradient, objective=objective,
                objective_reference=(
                    "weighted Poisson log pseudo-likelihood relative to saturated row means"
                    if family == "poisson" else "weighted binary log pseudo-likelihood"
                ),
                work_units=overhead+diagnostic_work+evaluations*evaluation_cost,
                max_work_units=MAX_WORK, diagnostic_work_allowance=diagnostic_work,
                work_accounting="full-design overhead + reserved bounded separation work + actual score/Hessian evaluations",
                algorithm="column-scaled observed Newton score with Armijo step halving",
                **separation,
            )
            return _finish(
                sample, coefficient, scales, sensitivity, factor, weights, convergence,
            )
        if iteration == max_iter:
            break
        ascent = float(score @ direction)
        if not math.isfinite(ascent) or ascent <= 0:
            raise AnalysisError(
                "survey_regression_nonconvergence",
                "The survey Newton direction has no finite objective ascent.",
            )
        accepted, step = False, 1.0
        roundoff = 64 * _EPSILON * max(1.0, abs(objective))
        for _ in range(40):
            candidate = coefficient + step * direction
            try:
                candidate_objective, _, _, _ = evaluate(candidate)
            except AnalysisError as error:
                if error.code != "survey_numerical_failure":
                    raise
                step *= 0.5
                continue
            target = objective + 1e-4 * step * ascent
            if candidate_objective >= target or (
                candidate_objective >= objective - roundoff and target <= objective + roundoff
            ):
                coefficient, accepted = candidate, True
                iterations += 1
                break
            step *= 0.5
        if not accepted:
            raise AnalysisError(
                "survey_regression_nonconvergence",
                "Step halving found no finite improving survey likelihood.",
            )
    raise AnalysisError(
        "survey_regression_nonconvergence",
        f"The {family} score did not converge within max_iter; no partial fit was returned.",
    )


@cpu_call
def _fit(
    data, design, outcome, regressors, *, family, intercept, domain, missing,
    alpha, null, max_iter, tolerance,
):
    max_iter, tolerance = _options(max_iter, tolerance)
    sample = admit(
        data, design, outcome, regressors, intercept=intercept, domain=domain, missing=missing
    )
    alpha, null = _inference_options(sample, alpha, null)
    fitted = fit_sample(sample, family, max_iter=max_iter, tolerance=tolerance)
    return build_result(sample, family, **fitted, alpha=alpha, null=null)


def survey_regress(
    data, design, outcome, regressors, *, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9,
) -> SurveyRegressionResult:
    """Weighted linear regression with full-design PSU/FPC Taylor covariance.

    Regressors declare numeric columns without implicit formula expansion.
    Excluded domain/missing rows retain their zero-score PSUs and design df.
    No HC1 correction, model covariance, or rank repair is used.
    """
    return _fit(
        data, design, outcome, regressors, family="linear", intercept=intercept, domain=domain,
        missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
    )


def survey_logit(
    data, design, outcome, regressors, *, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9,
) -> SurveyRegressionResult:
    """Sampling-weighted binary logit with exact score and design covariance.

    Responses must be exact 0/1. A native bounded replay diagnostic certifies
    absence of complete or quasi separation. Rank loss, overflow and failed
    convergence raise an error instead of yielding a penalized partial fit.
    """
    return _fit(
        data, design, outcome, regressors, family="logit", intercept=intercept, domain=domain,
        missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
    )


def survey_probit(
    data, design, outcome, regressors, *, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9,
) -> SurveyRegressionResult:
    """Weighted binary probit using its exact observed score sensitivity.

    Stable log-normal-CDF scores are linearized by complete-design PSU totals.
    The observed Bernoulli Hessian replaces no part with Fisher scoring.
    Separated and nonconvergent fits are refused.
    """
    return _fit(
        data, design, outcome, regressors, family="probit", intercept=intercept, domain=domain,
        missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
    )


def survey_poisson(
    data, design, outcome, regressors, *, intercept=True, domain=None, missing="raise",
    alpha=0.05, null=0.0, max_iter=100, tolerance=1e-9,
) -> SurveyRegressionResult:
    """Weighted count Poisson with full-design Taylor covariance of its scores.

    Exact nonnegative counts at most 2^53 are supported without offsets or
    exposures. A native bounded cone certificate checks general zero-count
    separation. Overflow, rank loss and nonconvergence refuse a result.
    """
    return _fit(
        data, design, outcome, regressors, family="poisson", intercept=intercept, domain=domain,
        missing=missing, alpha=alpha, null=null, max_iter=max_iter, tolerance=tolerance,
    )
