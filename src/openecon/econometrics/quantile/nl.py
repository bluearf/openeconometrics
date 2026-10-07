"""Stata's nl: nonlinear least squares for a regression function written as a formula.

Model.  y_i = f(x_i; b) + e_i with f given by ``options['formula']`` (see
``quantile.formula`` for the grammar).  The estimator minimizes the weighted residual
sum of squares S(b) = sum_i w_i (y_i - f(x_i; b))^2.

Solver.  Gauss-Newton with step halving and Levenberg-Marquardt damping.  With the
analytic Jacobian J = df/db', residuals r and G = J'WJ, g = J'W r, a step solves

    (G + mu diag(G)) d = g

by Cholesky on the unit-diagonal matrix (Marquardt's scaling, so the damping is
invariant to the units of the parameters).  mu starts at 0 (pure Gauss-Newton, as
Stata's modified Gauss-Newton); a step that does not reduce S is halved up to four
times, then mu is raised tenfold until S falls, and lowered again after every success.
Convergence requires, after an accepted step,

    |d_j| <= tol (|b_j| + 1e-3) for all j   and   S_old - S_new <= tol S_new,

or a Gauss-Newton step whose predicted reduction g'd is below tol^2 S (a stationary
point to rounding).  The covariance uses one Householder QR of sqrt(W) J at the solution.

Reported (Methods and formulas of [R] nl).  With K parameters and N observations:
s^2 = S / (N - K); nonrobust V = s^2 (J'WJ)^-1; robust is the HC1 sandwich
N/(N-K) (J'J)^-1 [sum_i r_i^2 J_i J_i'] (J'J)^-1, HC2/HC3 its leverage-adjusted
variants and cluster the CR1 sandwich G/(G-1) (N-1)/(N-K) with G - 1 degrees of freedom,
exactly as ``regress`` treats J as the regressor matrix.  If some parameter enters as an
additive constant (its Jacobian column is a nonzero constant) the total sum of squares
is centered and one model degree of freedom is given to the constant, as Stata does when
it reports "Parameter ... taken as constant term in model"; otherwise it is uncentered:

    R^2 = 1 - S / TSS,    adjusted R^2 = 1 - (1 - R^2) (N - c) / (N - K),    c = 1 or 0.

Weights: aweights and pweights are rescaled to sum to N, fweights replicate observations
(N = their sum); pweights need a robust or cluster covariance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, kernel_call, linear_covariance, make_spec,
)
from openecon.econometrics.linear.common import estimation_weights, weighted_mean
from openecon.econometrics.quantile import formula as formulas
from openecon.engines import linalg
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec, ResultBundle

_HALVINGS = 4
_MAX_DAMPING = 1e12
_STATIONARY_DAMPING = 1e-6  # damping up to which g'd certifies a stationary point
_PARAMETER_FLOOR = 1e-3     # tau of |d_j| <= tol (|b_j| + tau)
_EXACT_FIT = 1e-28          # S / sum w y^2 below which the fit is exact (rounding only)

Model = Callable[[Tensor, bool], tuple[Tensor, Tensor | None]]


@dataclass
class NonlinearFit:
    theta: Tensor
    fitted: Tensor
    resid: Tensor
    jacobian: Tensor
    rss: float
    iterations: int
    evaluations: int
    damping: float               # Levenberg-Marquardt parameter of the last accepted step
    predicted_reduction: float   # g'd of the Gauss-Newton step at the solution, relative to S
    history: list[float] = field(default_factory=list)


def _weighted_sum(values: Tensor, weights: Tensor | None) -> float:
    return float(values.sum() if weights is None else (values * weights).sum())


def _step(gram: Tensor, gradient: Tensor, damping: float) -> Tensor | None:
    """Solve (G + mu diag(G)) d = g on the unit-diagonal scale; None if not positive definite."""
    scale = gram.diagonal().sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    matrix = gram / scale[:, None] / scale
    if damping:
        matrix = matrix + damping * torch.eye(matrix.shape[0], dtype=matrix.dtype)
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info) != 0:
        return None
    step = torch.cholesky_solve((gradient / scale)[:, None], factor)[:, 0] / scale
    return step if bool(torch.isfinite(step).all()) else None


@torch.no_grad()
def nonlinear_least_squares(model: Model, start: Tensor, y: Tensor,
                            weights: Tensor | None = None, *, tolerance: float = 1e-8,
                            max_iterations: int = 1000) -> NonlinearFit:
    """Minimize sum_i w_i (y_i - f_i(theta))^2; ``model(theta, need_jacobian) -> (f, J)``."""
    theta = start.clone()
    fitted, jacobian = model(theta, True)
    resid = y - fitted
    rss = _weighted_sum(resid.square(), weights)
    if not math.isfinite(rss) or not bool(torch.isfinite(jacobian).all()):
        raise KernelError(
            "invalid_start",
            "The regression function or its derivatives are not finite at the starting values "
            "(for example the log of a nonpositive number, a division by zero or an overflow). "
            "Give starting values with start={...} or {name=value} in the formula.")
    floor = _EXACT_FIT * _weighted_sum(y.square(), weights)
    evaluations, damping, history = 1, 0.0, [rss]
    predicted = math.inf
    for iteration in range(1, max_iterations + 1):
        weighted = jacobian if weights is None else jacobian * weights[:, None]
        gram = weighted.T @ jacobian
        gradient = weighted.T @ resid
        accepted = None
        while accepted is None:
            step = _step(gram, gradient, damping)
            if step is None:
                damping = max(10 * damping, 1e-6)
            else:
                if damping <= _STATIONARY_DAMPING:
                    # The gradient lies in the range of G, so g'(G + mu D)^-1 g with a tiny mu
                    # is the Gauss-Newton predicted reduction even when G is singular.
                    predicted = float(gradient @ step)
                    if predicted <= tolerance ** 2 * rss + floor:
                        # Stationary to rounding: no step can reduce S any further.
                        return NonlinearFit(theta, fitted, resid, jacobian, rss, iteration - 1,
                                            evaluations, damping, predicted / max(rss, 1e-300),
                                            history)
                for halving in range(_HALVINGS + 1 if damping == 0.0 else 1):
                    trial = theta + step * 0.5 ** halving
                    trial_fitted, _ = model(trial, False)
                    evaluations += 1
                    if bool(torch.isfinite(trial_fitted).all()):
                        trial_rss = _weighted_sum((y - trial_fitted).square(), weights)
                        if trial_rss <= rss:
                            accepted = (trial, step * 0.5 ** halving, trial_rss)
                            break
                if accepted is None:
                    damping = max(10 * damping, 1e-4)
            if accepted is None and damping > _MAX_DAMPING:
                raise KernelError(
                    "nonconvergence",
                    "No step reduces the residual sum of squares: the model is not identified "
                    "near the current parameter values or the starting values are poor. Give "
                    "better starting values or simplify the regression function.")
        trial, step, trial_rss = accepted
        small = bool((step.abs() <= tolerance * (trial.abs() + _PARAMETER_FLOOR)).all())
        settled = rss - trial_rss <= tolerance * trial_rss + floor
        theta, previous = trial, rss
        fitted, jacobian = model(theta, True)
        evaluations += 1
        if not bool(torch.isfinite(jacobian).all()):
            raise KernelError("numerical_failure", "The derivatives of the regression function "
                              "are not finite at the current parameter values; give other "
                              "starting values.")
        resid = y - fitted
        rss = trial_rss
        history.append(rss)
        del history[:-50]
        if small and settled and damping <= 1.0:
            return NonlinearFit(theta, fitted, resid, jacobian, rss, iteration, evaluations,
                                damping, max(previous - rss, 0.0) / max(rss, 1e-300), history)
        damping = damping / 10 if damping > 1e-10 else 0.0
    raise KernelError("nonconvergence", f"Nonlinear least squares did not converge in "
                      f"{max_iterations} iterations. Give better starting values, rescale the "
                      "variables or raise max_iterations.")


def _start_values(frame: ModelFrame, formula: formulas.Formula) -> Tensor:
    given = frame.spec.options.get("start")
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise AnalysisError("invalid_start", "start must map parameter names to numbers, for "
                            "example start={'b0': 1, 'b1': 0.5}.")
    unknown = [name for name in given if name not in formula.parameters]
    if unknown:
        raise AnalysisError("invalid_start", f"start names parameters that are not in the "
                            f"formula: {', '.join(map(str, unknown))}. The formula has: "
                            f"{', '.join(formula.parameters)}.")
    values = {**formula.start, **given}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise AnalysisError("invalid_start", f"The starting value of '{name}' must be a "
                                "finite number.")
    missing = [name for name in formula.parameters if name not in values]
    if missing:
        frame.warn(f"No starting value given for {', '.join(missing)}; started at 0 (as "
                   "Stata's nl does). Nonlinear models usually need informed starting values.")
    return torch.tensor([float(values.get(name, 0.0)) for name in formula.parameters],
                        dtype=torch.float64)


def _constant_term(jacobian: Tensor, names: Sequence[str]) -> str | None:
    """The first parameter whose Jacobian column is a nonzero constant (an intercept)."""
    low, high = jacobian.min(dim=0).values, jacobian.max(dim=0).values
    for j, name in enumerate(names):
        centre = float(high[j] + low[j]) / 2
        if centre != 0 and float(high[j] - low[j]) <= 1e-12 * abs(centre):
            return name
    return None


def fit_nl(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``nl``."""
    formula = formulas.parse(spec.options.get("formula"))
    if spec.outcome in formula.columns:
        raise AnalysisError("invalid_formula", "The outcome must not appear in the regression "
                            "function.")
    stray = [name for name in spec.predictors if name not in formula.columns]
    if stray:
        raise AnalysisError("invalid_spec", f"Predictors not used in the formula: "
                            f"{', '.join(stray)}. List only the columns the formula reads (or "
                            "none: they are taken from the formula).")
    frame = ModelFrame(spec, data, extra_columns=formula.columns)
    if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance", "pweights are sampling weights and need "
                            "covariance='robust' (Stata's vce(robust)) or a cluster column.")
    names = list(formula.parameters)
    k = len(names)
    weights, nobs = estimation_weights(frame)
    if frame.n <= k:
        raise AnalysisError("insufficient_observations", f"The model has {k} parameters but only "
                            f"{frame.n} observation(s); it needs more observations than "
                            "parameters.")
    y = frame.numeric(spec.outcome)
    columns = {name: frame.numeric(name) for name in formula.columns}
    n = frame.n

    def model(theta: Tensor, need_jacobian: bool) -> tuple[Tensor, Tensor | None]:
        return formulas.evaluate(formula, columns, theta, n, jacobian=need_jacobian)

    start = _start_values(frame, formula)
    tolerance = float(frame.option("tolerance"))
    if not 0 < tolerance < 1:
        raise AnalysisError("invalid_option", "tolerance must lie strictly between 0 and 1.")
    max_iterations = int(frame.option("max_iterations"))
    fit = kernel_call(nonlinear_least_squares, model, start, y, weights, tolerance=tolerance,
                      max_iterations=max_iterations)
    jacobian, resid = fit.jacobian, fit.resid
    constant = _constant_term(jacobian, names)
    centered = y - weighted_mean(y, weights) if constant is not None else y
    tss = _weighted_sum(centered.square(), weights)
    if fit.rss <= _EXACT_FIT * _weighted_sum(y.square(), weights):
        raise AnalysisError("perfect_fit", "The regression function fits the outcome exactly, so "
                            "standard errors are undefined.")
    try:
        solved = kernel_call(linalg.least_squares, jacobian, resid, weights, drop_collinear=False)
    except AnalysisError as exc:
        if exc.code != "singular_design":
            raise
        kept, omitted = kernel_call(linalg.collinear_columns, jacobian, weights)
        unidentified = ", ".join(names[j] for j in omitted) or "some parameters"
        raise AnalysisError(
            "not_identified",
            f"The Jacobian is rank deficient at the solution: {unidentified} cannot be estimated "
            "separately from the other parameters. Remove or fix redundant parameters.") from exc
    df_resid = nobs - k
    v, info = linear_covariance(frame, x=jacobian, resid=resid, bread=solved.xtx_inv, n=nobs, k=k,
                                df_resid=df_resid, weights=weights, ssr=fit.rss)
    c = int(constant is not None)
    r_squared = 1 - fit.rss / tss if tss > 0 else None
    metrics: dict[str, Any] = {
        "r_squared": r_squared,
        "adjusted_r_squared": None if r_squared is None
        else 1 - (1 - r_squared) * (nobs - c) / df_resid,
        "rmse": math.sqrt(fit.rss / df_resid), "rss": fit.rss, "tss": tss,
    }
    if spec.weights is None or spec.weight_type == "fweight":
        log_likelihood = -0.5 * nobs * (math.log(2 * math.pi) + math.log(fit.rss / nobs) + 1)
        metrics["log_likelihood"] = log_likelihood
    metrics.update({"iterations": fit.iterations, "df_model": k - c, "df_resid": df_resid})
    extra = {"formula": formula.source, "parameters": names, "columns": list(formula.columns),
             "start": dict(zip(names, start.tolist(), strict=True)), "constant_term": constant,
             "total_sum_of_squares": "centered" if constant is not None else "uncentered"}
    diagnostics = {
        "converged": True, "iterations": fit.iterations, "function_evaluations": fit.evaluations,
        "final_damping": fit.damping, "relative_predicted_reduction": fit.predicted_reduction,
        "rss_history": fit.history[-10:], "jacobian": "analytic (forward-mode over the formula)",
        "jacobian_condition_number": solved.condition_number,
    }
    return build_result(
        frame, terms=names, params=fit.theta, covariance=v,
        df_inference=info.get("df_inference", df_resid), df_resid=df_resid, metrics=metrics,
        fitted=fit.fitted, solver="gauss_newton_levenberg_marquardt",
        solver_diagnostics=diagnostics,
        optimizer={"method": "Gauss-Newton with step halving and Levenberg-Marquardt damping",
                   "tolerance": tolerance, "max_iterations": max_iterations},
        inference=info, extra=extra, nobs=nobs,
    )


def nl(*, data: Any, y: str, formula: str, start: dict[str, float] | None = None,
       covariance: str | None = None, cluster: str | None = None, weights: str | None = None,
       weight_type: str | None = None, max_iterations: int = 1000, tolerance: float = 1e-8,
       missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Nonlinear least squares (Stata's ``nl``; SPSS ``NLR``; EViews nonlinear LS).

    Model: y_i = f(x_i; b) + e_i, where the regression function f is written as
    ``formula`` in the style of Stata's substitutable expressions. The
    estimator minimizes sum_i w_i (y_i - f(x_i; b))^2.

    Formula grammar
    ---------------
    - parameters in braces: ``{b0}``, or ``{b0=1.5}`` to give a starting value;
    - data columns by name (names must be valid identifiers);
    - numbers, ``+ - * / ^`` (``**`` also means power), parentheses;
    - functions ``exp``, ``ln``/``log`` (natural log), ``sqrt``, ``abs``,
      ``sin``, ``cos``, ``tan``, ``expit``/``invlogit``, ``normal`` (standard
      normal cdf) and ``normalden`` (its density).

    ``-x^2`` means ``-(x^2)`` as in Stata; a chain ``a^b^c`` must be
    parenthesized. Give only the right-hand side (``y`` is passed separately).
    The formula is parsed into a syntax tree and checked node by node; it is
    never executed as code, and anything outside the grammar raises
    ``invalid_formula``.

    Parameters
    ----------
    data : DataFrame, dict of columns or list of records.
    y : outcome column.
    formula : the regression function, e.g. ``"{b0} + {b1} * exp(-{b2} * x)"``.
    start : ``{parameter: value}`` starting values. They override ``{b=value}``
        in the formula; parameters with neither start at 0 with a warning (as in
        Stata). Good starting values matter for nonlinear models.
    covariance : ``'nonrobust'`` (default), ``'robust'`` (Stata's
        ``vce(robust)``), ``'HC2'``, ``'HC3'`` or ``'cluster'`` (implied by
        ``cluster``).
    weights, weight_type : ``'aweight'``, ``'fweight'`` or ``'pweight'``
        (pweights need robust/cluster and select robust by default).
    max_iterations, tolerance : iteration limit (1000) and the relative
        convergence tolerance on the parameters and the residual sum of
        squares (1e-8; Stata's ``eps()`` is 1e-5).
    missing : ``'raise'`` (default) or ``'drop'``.  alpha : test size.

    Estimation
    ----------
    Gauss-Newton with step halving and Levenberg-Marquardt damping (Stata uses
    a modified Gauss-Newton). The Jacobian df/db' is analytic: the parsed
    formula is differentiated by forward-mode propagation of exact partial
    derivatives (no numerical differencing, no autograd).

    Covariance and reported statistics (J the Jacobian at the solution, K
    parameters, N observations): nonrobust V = s^2 (J'J)^-1 with
    s^2 = RSS/(N - K); robust the HC1 sandwich N/(N-K) (J'J)^-1 (sum r_i^2 J_i
    J_i') (J'J)^-1; cluster the CR1 sandwich with G/(G-1) (N-1)/(N-K) and G - 1
    degrees of freedom. Student t inference with N - K degrees of freedom.

    ``metrics``: ``r_squared`` = 1 - RSS/TSS and ``adjusted_r_squared``, where
    the total sum of squares is centered when a parameter enters as an additive
    constant (``extra['constant_term']``) and uncentered otherwise, as in
    Stata; ``rmse``, ``rss``, ``tss``, ``log_likelihood`` (Gaussian; unweighted
    or frequency-weighted fits), ``iterations``, ``df_model``, ``df_resid``.
    Terms are the parameter names (Stata prints them as ``/b0``).
    ``provenance['solver_diagnostics']`` reports convergence. Stata's nl prints
    no model test, so ``tests`` is empty.

    Errors: ``invalid_formula``, ``invalid_start`` (also: the function is not
    finite at the starting values), ``nonconvergence``, ``not_identified``
    (rank-deficient Jacobian), ``perfect_fit``, ``insufficient_observations``.

    Stata: ``nl (y = {b0} + {b1}*exp(-{b2}*x)), initial(b0 1 b1 2 b2 0.1)``.

    Example::

        import openecon as oe
        fit = oe.nl(data=df, y="y", formula="{b0} + {b1} * exp(-{b2} * x)",
                    start={"b0": 1, "b1": 2, "b2": 0.1})
        print(fit.summary())
        print(fit.provenance["solver_diagnostics"]["iterations"])
    """
    from openecon.analysis import fit

    parsed = formulas.parse(formula)
    if covariance is None and weight_type == "pweight" and not cluster:
        covariance = "robust"
    spec = make_spec(
        "nl", outcome=y, predictors=list(parsed.columns), covariance=covariance, cluster=cluster,
        intercept=False, weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        options={"formula": formula, "start": start,
                 "max_iterations": None if max_iterations == 1000 else max_iterations,
                 "tolerance": None if tolerance == 1e-8 else tolerance},
    )
    return fit(spec, data=data)
