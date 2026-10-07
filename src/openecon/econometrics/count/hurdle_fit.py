"""Two-part (hurdle) models: Cragg's ``churdle`` and the count ``hurdle``.

Both models split the outcome into a *participation* decision and an *amount*:

    Pr(y > ll) = F(z'g)                              (selection equation)
    y | y > ll ~ a distribution truncated at ll      (outcome equation)

and observations at or below the limit contribute ``ln{1 - F(z'g)}`` only.
Because the two parts share no parameter the log likelihood is the sum of a
binary likelihood on all observations and a truncated likelihood on the
observations above the limit. The parts are maximized separately (each by
Newton-Raphson with analytic derivatives), which gives exactly the joint
maximum; the joint Hessian is block diagonal and the per-observation scores of
both parts are stacked, so robust and cluster covariances are those of the
joint estimator (they include the covariance between the two equations).

``churdle`` (Stata's ``churdle linear`` / ``churdle exponential``, Cragg 1971)
---------------------------------------------------------------------------
Probit selection (``select_link='logit'`` is an OpenEconometrics extension) and, for
``y > ll``,

* ``model='linear'``: ``y = x'b + e``, ``e ~ N(0, sigma^2)`` truncated to
  ``y > ll``:  ``ln f = ln phi((y - x'b)/sigma) - ln sigma - ln Phi((x'b - ll)/sigma)``;
* ``model='exponential'``: ``ln y = x'b + e`` truncated to ``y > ll``:
  ``ln f = ln phi((ln y - x'b)/sigma) - ln sigma - ln y - ln Phi((x'b - ln ll)/sigma)``,
  where the last term vanishes for ``ll = 0`` (the lognormal hurdle model).

Parameters are reported in Stata's order: outcome equation, selection
equation (``select:<term>``, Stata's ``selection_ll``), ``/lnsigma``.

``hurdle`` (Mullahy 1986; community commands ``hplogit`` / ``hnblogit``)
---------------------------------------------------------------------
``Pr(y > 0) = F(z'g)`` with a logit, probit or complementary log-log link
(the last is Mullahy's Poisson hurdle, ``Pr(y = 0) = exp(-exp(z'g))``) and a
zero-truncated Poisson or negative binomial (NB2) for the positive counts.

Model test (both): LR chi2 that the slopes of the outcome equation are zero
(the selection part is common to both models and cancels) under the
conventional covariance, Wald otherwise; ``pseudo_r_squared = 1 - ll/ll_0``
with ``ll_0`` the log likelihood of that comparison model, as Stata's churdle.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, information_criteria, kernel_call,
)
from openecon.econometrics.count import common
from openecon.econometrics.count.common import Model
from openecon.econometrics.count.kernels import BinaryPieces, TruncatedNormalPieces, link_pieces
from openecon.econometrics.count.truncated import fit_truncated, truncated_sample
from openecon.econometrics.glm.common import (
    Weights, check_pweights, likelihood_covariance, likelihood_weights, optimizer_record,
    require_terms, safe_exp, slope_indices, uncenter,
)
from openecon.engines import optimize
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

_EXTREME = 1e-6
_EXTREME_EARLY = 1e-10
_WATCH_FROM, _WATCH_EVERY = 20, 5


class _Separated(Exception):
    def __init__(self, theta: Tensor):
        super().__init__("separated")
        self.theta = theta


def _perfect(model: Model, theta: Tensor, positive: Tensor, link: str, threshold: float) -> int:
    """Observations whose observed selection status is predicted with probability ~1."""
    a, b = link_pieces(link, model.objective.indices(theta)[0], False)[:2]
    probability = torch.exp(torch.where(positive, a, b))
    high = probability > 1 - threshold
    if not bool(high.any()) or bool((probability < threshold).any()):
        return 0
    return int(high.sum())


def _selection(frame: ModelFrame, positive: Tensor, design: Design, weights: Weights,
               link: str, command: str) -> tuple[Model, optimize.OptimResult]:
    """The binary selection equation ``Pr(positive) = F(z'g)`` by Newton-Raphson."""
    w = weights.user
    model = Model(BinaryPieces(positive, link), [common.design_block(design, w, "select")], w)
    start = torch.zeros(model.size, dtype=torch.float64)
    start[0] = common.link_inverse(link, float((w * positive).sum() / w.sum()))

    def watch(iteration: int, theta: Tensor, value: float) -> None:
        if iteration >= _WATCH_FROM and iteration % _WATCH_EVERY == 0 \
                and _perfect(model, theta, positive, link, _EXTREME_EARLY):
            raise _Separated(theta.clone())

    try:
        result, _ = common.run(model, start, what=f"{command} (selection equation)",
                               weights=weights, callback=watch)
        theta = result.theta
    except _Separated as stop:
        result, theta = None, stop.theta
    flat = result is not None and result.converged \
        and _perfect(model, theta, positive, link, _EXTREME) \
        and common.flattest_information(-result.hessian, model.blocks[0].x,
                                        weights) < common.FLAT_INFORMATION
    if result is None or not result.converged or flat:
        if _perfect(model, theta, positive, link, _EXTREME):
            raise AnalysisError(
                "separation_detected",
                f"{command}: the selection equation predicts some observations perfectly "
                "(fitted probabilities numerically 0 or 1) and its likelihood keeps "
                "increasing: complete or quasi-complete separation, so finite estimates do "
                "not exist. Remove or combine the selection regressor that separates the "
                "observations above the limit from the others.")
        message = "the likelihood is monotone" if result is None \
            else result.diagnostics.get("message")
        raise common.not_converged(f"{command} (selection equation)", message)
    return model, result


def _subsample(frame: ModelFrame, rows: Tensor, weights: Weights) -> tuple[Design, Weights, Design]:
    """Outcome design, screened on the selected rows, and their weights."""
    full = frame.design()
    w = weights.user[rows]
    frequency = None if weights.frequency is None else weights.frequency[rows]
    nobs = len(rows) if frequency is None else int(round(float(frequency.sum())))
    sub = Design(full.x[rows], full.terms, full.categories, full.intercept)
    sub = frame.drop_collinear(sub, w if weights.weighted else None)
    require_terms(sub)
    return sub, Weights(w, frequency, nobs, weights.weighted), full


def _require_above(nobs: int, parameters: int, command: str, where: str, equation: str) -> None:
    """The second part is estimated from the observations above the limit only."""
    if nobs <= parameters:
        raise AnalysisError(
            "insufficient_observations",
            f"{command}: the {equation} equation is estimated from the {nobs} observation(s) "
            f"{where}, which is not more than its {parameters} parameter(s). Use fewer "
            f"regressors in the {equation} equation or more data.")


def _assemble(frame: ModelFrame, weights: Weights, outcome: Model,
              outcome_result: optimize.OptimResult, rows: Tensor, select: Model,
              select_result: optimize.OptimResult) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Joint estimates and covariance in the order (outcome, selection, ancillary)."""
    k, q = outcome.blocks[0].size, select.size
    size = outcome.size + q
    first = torch.tensor([*range(k), *range(k + q, size)], dtype=torch.int64)
    second = torch.arange(k, k + q)
    theta = torch.zeros(size, dtype=torch.float64)
    theta[first] = outcome_result.theta
    theta[second] = select_result.theta
    hessian = torch.zeros((size, size), dtype=torch.float64)
    hessian[first[:, None], first] = outcome_result.hessian
    hessian[second[:, None], second] = select_result.hessian

    def scores() -> Tensor:
        table = torch.zeros((frame.n, size), dtype=torch.float64)
        table[rows[:, None], first] = outcome.objective.score_rows(outcome_result.theta)
        table[:, k:k + q] = select.objective.score_rows(select_result.theta)
        return table

    covariance, info = likelihood_covariance(frame, hessian, scores, weights)
    theta, covariance = uncenter(theta, covariance,
                                 [(0, outcome.blocks[0].means), (k, select.blocks[0].means)])
    info["df_inference"] = None
    return theta, covariance, info


def _finish(frame: ModelFrame, weights: Weights, outcome: Model,
            outcome_result: optimize.OptimResult, rows: Tensor, select: Model,
            select_result: optimize.OptimResult, null: float | None, sub: Design,
            fitted: Callable[[Tensor, Tensor], Tensor], metrics: dict[str, Any],
            tests: dict[str, Any], extra: dict[str, Any], *, equation: str, link: str,
            ancillary: str | None) -> ResultBundle:
    """Covariance, model test, fit statistics and result assembly of a two-part model.

    ``ancillary`` names the positive parameter estimated in logs after the
    selection equation (``sigma`` or ``alpha``), reported in ``metrics`` and,
    with its delta-method standard error and exponentiated interval, in ``extra``.
    """
    spec = frame.spec
    k, q = outcome.blocks[0].size, select.size
    theta, covariance, info = _assemble(frame, weights, outcome, outcome_result, rows, select,
                                        select_result)
    log_likelihood = outcome_result.value + select_result.value
    null_total = None if null is None else null + select_result.value
    terms = [*outcome.blocks[0].terms, *select.terms, *outcome.terms[k:]]
    equations = [spec.outcome] * k + ["select"] * q + [None] * (outcome.size - k)
    slopes = slope_indices(sub.terms)
    tests = {"model": common.model_test(frame, theta, covariance, slopes, log_likelihood,
                                        null_total, equation), **tests}
    nobs = weights.nobs
    criteria = information_criteria(log_likelihood, len(terms), nobs)
    metrics = {"log_likelihood": log_likelihood,
               "pseudo_r_squared": common.pseudo_r_squared(log_likelihood, null_total),
               "aic": criteria["aic"], "bic": criteria["bic"], **metrics}
    if ancillary is not None:
        log_value = float(theta[k + q])
        metrics[ancillary] = safe_exp(log_value)
        extra = {**extra, ancillary: common.exponentiated(
            log_value, math.sqrt(float(covariance[k + q, k + q])), spec.alpha, ancillary)}
    probability = torch.exp(link_pieces(
        link, select.objective.indices(select_result.theta)[0], False)[0])
    mean = fitted(theta, probability)
    if not bool(torch.isfinite(mean).all()):
        frame.warn("Some fitted means are not finite; the chart sample of fitted values is "
                   "omitted.")
        mean = None
    record = {"null_log_likelihood": null_total,
              "null_model": None if null is None else
              f"constant-only {equation} equation, full selection equation",
              "selection_log_likelihood": select_result.value,
              "outcome_log_likelihood": outcome_result.value,
              "mean_selection_probability":
                  float((weights.user * probability).sum() / weights.user.sum()), **extra}
    iterations = outcome_result.iterations + select_result.iterations
    return build_result(
        frame, terms=terms, params=theta, covariance=covariance, equations=equations,
        use_t=False, metrics=metrics, fitted=mean, nobs=nobs, inference=info, tests=tests,
        extra=record, categories={**sub.categories, **select.blocks[0].design.categories},
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": iterations},
        optimizer={**optimizer_record(outcome_result), "iterations": iterations,
                   "selection_iterations": select_result.iterations,
                   "outcome_iterations": outcome_result.iterations,
                   "note": "the two parts share no parameter and are maximized separately"},
        df_resid=nobs - len(terms))


def _columns(full: Design, sub: Design) -> Tensor:
    """The full-sample design restricted to the columns kept on the subsample."""
    return full.x[:, [full.terms.index(term) for term in sub.terms]]


# ---- churdle -----------------------------------------------------------------------------


def fit_churdle(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``churdle``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    weights = likelihood_weights(frame)
    y = frame.numeric(spec.outcome)
    kind, link = frame.option("model"), frame.option("select_link")
    limit = float(frame.option("ll"))
    if kind == "exponential" and limit < 0:
        raise AnalysisError("invalid_option", "churdle exponential models ln(y) above the "
                            "limit, so ll must be zero or positive.")
    selected = y > limit
    above = int(selected.sum())
    if above == 0 or above == frame.n:
        raise AnalysisError(
            "no_selection_variation",
            f"churdle needs outcomes on both sides of the lower limit ll = {limit:g}: "
            + ("no outcome exceeds it." if above == 0 else
               "every outcome exceeds it, so the selection equation is not identified; fit "
               "the outcome equation alone with oe.truncreg."))
    under = int((y < limit).sum())
    if under:
        frame.warn(f"{under} observation(s) of '{spec.outcome}' lie below the lower limit "
                   f"{limit:g}; they are treated as bounded observations at the limit.")
    select_design = frame.drop_collinear(
        frame.design(frame.role("select_x"), intercept=True, prefix="select:"),
        weights.for_screen())
    select, select_result = _selection(frame, selected, select_design, weights, link, "churdle")
    rows = selected.nonzero().flatten()
    sub, sub_weights, full = _subsample(frame, rows, weights)
    _require_above(sub_weights.nobs, len(sub.terms) + 1, "churdle", f"above the limit {limit:g}",
                   "outcome")
    w = sub_weights.user
    observed = y[rows]
    if kind == "linear":
        t, cut, jacobian = observed, limit, None
    else:
        t = torch.log(observed)
        cut, jacobian = (math.log(limit) if limit > 0 else None), -t
    pieces = TruncatedNormalPieces(t, cut, jacobian)
    block = common.design_block(sub, w, spec.outcome)
    outcome = Model(pieces, [block, common.scalar_block("/lnsigma")], w)
    k = block.size
    ols = kernel_call(least_squares, block.x, t, w, drop_collinear=True, tol=0.0)
    spread = math.sqrt(float(ols.ssr) / float(w.sum()))
    center = float((w * t).sum() / w.sum())
    scale = math.sqrt(float((w * (t - center).square()).sum() / w.sum()))
    if not scale > 1e-12 * max(1.0, abs(center)):
        raise AnalysisError("constant_outcome", f"'{spec.outcome}' does not vary above the "
                            "lower limit; sigma is not identified.")
    if spread <= 1e-7 * scale:
        raise AnalysisError(
            "perfect_fit", "churdle: the outcome equation fits the outcomes above the limit "
            "exactly, so sigma is zero and the likelihood is unbounded. Remove regressors "
            "that determine the outcome exactly.")
    start = torch.cat([ols.beta, torch.tensor([math.log(spread)], dtype=torch.float64)])
    result, _ = common.run(outcome, start, what="churdle (outcome equation)",
                           weights=sub_weights)
    if not result.converged:
        raise common.not_converged(
            "churdle (outcome equation)", result.diagnostics.get("message"),
            "The truncated normal likelihood may have no interior maximum when the outcomes "
            "above the limit pile up next to it (the fitted mean runs below the limit and "
            "sigma grows). Consider model='exponential' or a different limit.")
    null = None
    if spec.intercept:
        if k == 1:
            null = result.value
        else:
            constant = common.constant_block(len(rows), spec.outcome)
            null_model = Model(pieces, [constant, common.scalar_block("/lnsigma")], w)
            fitted, _ = common.run(
                null_model, torch.tensor([center, math.log(scale)], dtype=torch.float64),
                what="churdle (constant-only outcome equation)", weights=sub_weights)
            if fitted.converged:
                null = fitted.value
            else:
                frame.warn("The constant-only comparison model did not converge; the Wald "
                           "test of the outcome equation's slopes is reported instead of the "
                           "LR test.")
    x_all = _columns(full, sub)

    def fitted_mean(theta: Tensor, probability: Tensor) -> Tensor:
        xb, sigma = x_all @ theta[:k], safe_exp(float(theta[-1]))
        if kind == "linear":
            ratio = torch.exp(-0.5 * ((xb - limit) / sigma).square() - 0.5 * math.log(
                2 * math.pi) - torch.special.log_ndtr((xb - limit) / sigma))
            conditional = xb + sigma * ratio
        elif limit > 0:
            shift = (xb - math.log(limit)) / sigma
            conditional = torch.exp(xb + 0.5 * sigma ** 2 + torch.special.log_ndtr(
                shift + sigma) - torch.special.log_ndtr(shift))
        else:
            conditional = torch.exp(xb + 0.5 * sigma ** 2)
        return (1 - probability) * limit + probability * conditional

    extra = {"model": kind, "ll": limit, "select_link": link,
             "n_bounded_observations": common.count_total(~selected, weights),
             "fitted_values": "E[y] = (1 - P) ll + P E[y | y > ll], P = Pr(y > ll)",
             "selection_equation": "Pr(y > ll); terms prefixed select: (Stata's selection_ll)"}
    return _finish(frame, weights, outcome, result, rows, select, select_result, null, sub,
                   fitted_mean, {}, {}, extra, equation="outcome", link=link,
                   ancillary="sigma")


# ---- hurdle (counts) -----------------------------------------------------------------------


def fit_hurdle(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``hurdle``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    y, weights, offset = common.count_sample(frame, "hurdle", integer=True)
    dist, link = frame.option("dist"), frame.option("zero_link")
    positive = y > 0
    count = int(positive.sum())
    if count == 0 or count == frame.n:
        raise AnalysisError(
            "no_selection_variation",
            "hurdle needs both zero and positive outcomes: "
            + (f"'{spec.outcome}' is zero in every observation." if count == 0 else
               f"'{spec.outcome}' has no zeros, so the participation equation is not "
               "identified; fit the positive counts with oe.tpoisson or oe.tnbreg."))
    names = frame.role("select_x") or list(spec.predictors)
    select_design = frame.drop_collinear(
        frame.design(names, intercept=True, prefix="select:"), weights.for_screen())
    select, select_result = _selection(frame, positive, select_design, weights, link, "hurdle")
    rows = positive.nonzero().flatten()
    sub, sub_weights, full = _subsample(frame, rows, weights)
    negbin = dist == "nbinomial"
    _require_above(sub_weights.nobs, len(sub.terms) + int(negbin), "hurdle", "above zero", "count")
    part_offset = None if offset is None else offset[rows]
    sample = truncated_sample(frame, "hurdle", y[rows], sub_weights, part_offset, 0, sub)
    part = fit_truncated(sample, "mean" if negbin else None, intercept=spec.intercept,
                         poisson_command="dist='poisson'")
    k = sample.count.size
    x_all = _columns(full, sub)

    def fitted_mean(theta: Tensor, probability: Tensor) -> Tensor:
        eta = x_all @ theta[:k]
        mu = torch.exp(eta if offset is None else eta + offset)
        if negbin:
            a = safe_exp(float(theta[-1]))
            zero = torch.exp(-torch.log1p(a * mu) / a)
        else:
            zero = torch.exp(-mu)
        return probability * mu / (1 - zero)

    metrics: dict[str, Any] = {"n_zero_observations": common.count_total(~positive, weights)}
    tests: dict[str, Any] = {}
    extra: dict[str, Any] = {
        "dist": dist, "zero_link": link,
        "fitted_values": "E[y] = Pr(y > 0) E[y | y > 0]",
        "selection_equation": "Pr(y > 0); terms prefixed select:"}
    if negbin:
        extra["tpoisson_log_likelihood"] = part.poisson.value
        if spec.covariance in {"nonrobust", "opg"}:
            tests["alpha"] = common.boundary_test(
                part.result.value, part.poisson.value, "alpha",
                "the Poisson hurdle model (zero-truncated Poisson counts)")
        else:
            extra["alpha_test_note"] = (
                "The LR test of alpha = 0 is not reported with a robust or cluster "
                "covariance (it is not a likelihood-based test).")
    return _finish(frame, weights, part.model, part.result, rows, select, select_result,
                   part.null, sub, fitted_mean, metrics, tests, extra, equation="count",
                   link=link, ancillary="alpha" if negbin else None)
