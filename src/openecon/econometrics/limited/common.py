"""Conventions shared by the limited-dependent-variable family's analysis layer.

* **Weights** (Stata's semantics for likelihood estimators, shared with the
  discrete family). Every likelihood, score and Hessian sum is
  ``sum_i w_i (.)`` with ``fweight``: integer frequencies, ``N = sum w_i``,
  results identical to the duplicated-row data set; ``aweight``: rescaled to
  sum to the number of rows; ``iweight``: used as given; ``pweight``: as
  iweight for the point estimates, and a robust or cluster covariance is
  required (``robust`` is the convenience default).
* **Covariance.** ``core.ml_covariance``: ``nonrobust`` = inverse observed
  information, ``opg`` = ``(sum_i w_i s_i s_i')^-1``, ``robust`` = ``N/(N-1)``
  sandwich, ``cluster`` = ``G/(G-1)`` sandwich (meats built from ``w_i s_i``).
* **Scale parameters.** Likelihoods are maximized in ``ln sigma``; a reported
  ``/sigma`` and its covariance come from the exact delta method
  ``V(b, sigma) = J V(b, ln sigma) J'`` with ``J = diag(I, sigma)``.
* **Limits.** ``ll`` / ``ul`` are numbers; ``ll_at_min`` / ``ul_at_max`` use
  the smallest / largest outcome of the estimation sample (Stata's ``ll`` and
  ``ul`` without a value).
* **Counts** (censored, selected, ...) are frequency-weighted under fweights,
  like ``N``, and row counts otherwise.
* **Centring.** In an equation with a constant the index is
  ``a + x'b = (a + m'b) + (x - m)'b``, so every likelihood is maximized on
  regressors centred at their (weighted) means ``m`` and the constant is
  mapped back exactly, ``a = a_c - m'b``, with covariance ``J V J'`` for the
  Jacobian ``J`` of that linear map (scores and Hessians transform the same
  way, so this holds for every covariance estimator). The reported model is
  the one specified; the arithmetic no longer loses ``(mean / spread)^2``
  digits when a regressor has a large level (a date, an identifier-like
  number). Equations without a constant are not centred.
* **Degenerate scale.** A regression that fits the outcome exactly has no
  error variance: the likelihood is unbounded as ``sigma -> 0``. This is
  reported as ``perfect_fit`` before any iteration starts.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, ModelFrame, kernel_call
from openecon.econometrics.discrete import common as _shared
from openecon.econometrics.discrete.common import (  # noqa: F401  (re-exported for the family)
    Weights, binary_outcome, build_spec, check_pweights, likelihood_weights, maximize,
    optimizer_record, probit_fit, require_observations, resolve_covariance,
)
from openecon.engines import optimize
from openecon.engines.inference import critical_value
from openecon.engines.linalg import least_squares

LIKELIHOOD_KINDS = {"nonrobust", "opg"}


def likelihood_covariance(frame: ModelFrame, hessian: Tensor, rows: Callable[[], Tensor],
                          weights: Weights, **overrides: Any) -> tuple[Tensor, dict[str, Any]]:
    """``core.ml_covariance`` with the family's weight semantics.

    ``rows()`` returns the per-observation score rows before user weights. The
    sandwich meats of ``robust`` and ``cluster`` use the weighted scores
    ``w_i s_i`` (``sum f_i s_i s_i'`` under fweights). The outer product of
    gradients estimates the information of ``sum_i w_i l_i``, which is linear
    in the weights, so ``opg`` is ``(sum_i w_i s_i s_i')^-1`` for every weight
    type: an integer iweight then gives the same ``nonrobust`` and ``opg``
    covariance as the equal fweight, as in Stata's ``ml``.
    """
    kind = overrides.get("kind") or frame.spec.covariance
    if kind == "opg" and weights.weighted and weights.frequency is None:
        weights = Weights(weights.user, weights.user, weights.nobs, True)
    return _shared.likelihood_covariance(frame, hessian, rows, weights, **overrides)


def offset_column(frame: ModelFrame) -> Tensor | None:
    """The ``offset`` role of the current sample (coefficient fixed at one), or None."""
    names = frame.role("offset")
    if names and names[0] == frame.spec.outcome:
        raise AnalysisError("invalid_spec", "The outcome cannot be its own offset.")
    return frame.numeric(names[0]) if names else None


def count(mask: Tensor, weights: Weights) -> int:
    """Observations flagged by ``mask`` as Stata counts them (frequencies under fweights)."""
    if weights.frequency is not None:
        return int(round(float(weights.frequency[mask].sum())))
    return int(mask.sum())


def resolve_limits(frame: ModelFrame, y: Tensor, command: str) -> tuple[float | None, float | None]:
    """Censoring limits from the options ``ll``, ``ul``, ``ll_at_min`` and ``ul_at_max``."""
    options = frame.spec.options
    lower, upper = options.get("ll"), options.get("ul")
    if options.get("ll_at_min"):
        if lower is not None:
            raise AnalysisError("invalid_spec", f"{command}: give either ll or ll='min', not "
                                "both.")
        lower = float(y.min())
    if options.get("ul_at_max"):
        if upper is not None:
            raise AnalysisError("invalid_spec", f"{command}: give either ul or ul='max', not "
                                "both.")
        upper = float(y.max())
    lower = None if lower is None else float(lower)
    upper = None if upper is None else float(upper)
    if lower is not None and upper is not None and not lower < upper:
        raise AnalysisError("invalid_limits", f"{command}: the lower limit ({lower:g}) must be "
                            f"below the upper limit ({upper:g}).")
    return lower, upper


def limit_options(ll: Any, ul: Any, command: str) -> dict[str, Any]:
    """Options of a convenience function's ``ll`` / ``ul`` (numbers, ``'min'`` / ``'max'``)."""
    options: dict[str, Any] = {}
    for name, value, keyword, flag in (("ll", ll, "min", "ll_at_min"),
                                       ("ul", ul, "max", "ul_at_max")):
        if value is None:
            continue
        if isinstance(value, str):
            if value != keyword:
                raise AnalysisError("invalid_spec", f"{command}: {name} must be a number or "
                                    f"'{keyword}' (the {keyword}imum outcome).")
            options[flag] = True
        elif isinstance(value, bool) or not isinstance(value, numbers.Real) \
                or not math.isfinite(value):
            # numbers.Real covers Python and NumPy integers and floats (not booleans).
            raise AnalysisError("invalid_spec", f"{command}: {name} must be a finite number or "
                                f"'{keyword}'.")
        else:
            options[name] = float(value)
    return options


def centre(design: Design, weights: Tensor | None = None) -> Tensor:
    """Centre the regressors of an equation whose first column is the constant.

    ``design.x`` is replaced by ``[1, x - m]`` with ``m`` the (weighted) column
    means; the returned vector holds ``m`` (zero for the constant, and all
    zeros for a design without a constant, which is left untouched). See
    "Centring" in the module notes.
    """
    columns = design.x.shape[1]
    means = torch.zeros(columns, dtype=torch.float64)
    if design.intercept and columns > 1:
        block = design.x[:, 1:]
        if weights is None:
            means[1:] = block.mean(dim=0)
        else:
            means[1:] = (weights[:, None] * block).sum(dim=0) / weights.sum()
        centred = design.x.clone()
        centred[:, 1:] -= means[1:]
        design.x = centred
    return means


def constant_shift(blocks: Sequence[Tensor], extra: int = 0) -> Tensor:
    """Jacobian of ``(a_c, b) -> (a, b) = (a_c - m'b, b)`` for consecutive equations.

    ``blocks`` holds the centres ``m`` of each equation as returned by
    ``centre`` (constant first); ``extra`` trailing parameters (ancillary
    ones) are left unchanged.
    """
    size = sum(len(block) for block in blocks) + extra
    jacobian = torch.eye(size, dtype=torch.float64)
    start = 0
    for block in blocks:
        if len(block) > 1:
            jacobian[start, start + 1:start + len(block)] = -block[1:]
        start += len(block)
    return jacobian


def uncentre(theta: Tensor, covariance: Tensor, jacobian: Tensor) -> tuple[Tensor, Tensor]:
    """The linear change of parameters ``J theta`` and its covariance ``J V J'``."""
    return jacobian @ theta, jacobian @ covariance @ jacobian.T


def require_error_variance(ssr: float, target: Tensor, weights: Tensor, command: str,
                           what: str = "the outcome") -> None:
    """Refuse a regression whose residuals are zero to rounding (``sigma`` not identified)."""
    size = float((weights * target.square()).sum())
    if not ssr > 1e-24 * size:
        raise AnalysisError(
            "perfect_fit",
            f"{command}: the regressors fit {what} exactly (it is constant or an exact linear "
            "function of them), so the error variance is zero and the likelihood has no "
            "maximum. Check the outcome column and remove regressors that determine it.")


def ols_start(x: Tensor, y: Tensor, weights: Tensor, offset: Tensor | None = None, *,
              command: str = "The model", what: str = "the outcome") -> tuple[Tensor, float]:
    """Weighted least-squares coefficients and ``ln`` of the ML residual scale.

    Starting values of every normal likelihood in the family: QR least squares
    of ``y - offset`` on ``x`` and ``sigma^2 = sum w e^2 / sum w``. An exact
    fit raises ``perfect_fit``: the likelihood is then unbounded in ``sigma``.
    """
    target = y if offset is None else y - offset
    solved = kernel_call(least_squares, x, target, weights, drop_collinear=False)
    ssr = float(solved.ssr)
    require_error_variance(ssr, target, weights, command, what)
    return solved.beta, 0.5 * math.log(ssr / float(weights.sum()))


def weight_scale(weights: Weights) -> float:
    """Objective scale of the iteration under iweights / pweights of arbitrary magnitude.

    The discrete family's ``Weights.scale``: the optimizer runs on
    ``ll / mean(w)`` so that its convergence rules do not depend on the unit
    of importance or sampling weights; results are scaled back.
    """
    return float(getattr(weights, "scale", 1.0))


def run_newton(objective: Any, start: Tensor, *, command: str, advice: str,
               max_iter: int = 200, scale: float = 1.0) -> optimize.OptimResult:
    """Newton-Raphson maximization; a failed run becomes ``AnalysisError('nonconvergence')``."""
    try:
        result = maximize(objective, start, what=command, max_iter=max_iter, scale=scale)
    except AnalysisError as exc:
        if exc.code != "numerical_failure":
            raise
        # The iteration ran into a region without a finite ascent step: a likelihood that
        # increases without bound (for example a scale parameter running to zero).
        raise AnalysisError("nonconvergence", f"{command} did not converge: the likelihood "
                            f"has no interior maximum along the iteration path. {advice}"
                            ) from exc
    if not result.converged:
        raise AnalysisError("nonconvergence", f"{command} did not converge: "
                            f"{result.diagnostics.get('message')} {advice}")
    return result


def scale_to_sigma(theta: Tensor, covariance: Tensor, index: int) -> tuple[Tensor, Tensor]:
    """Replace ``ln sigma`` at ``index`` by ``sigma`` with its delta-method covariance."""
    sigma = math.exp(float(theta[index]))
    params = theta.clone()
    params[index] = sigma
    jacobian = torch.ones(len(theta), dtype=torch.float64)
    jacobian[index] = sigma
    return params, covariance * jacobian[:, None] * jacobian


def sigma_record(log_sigma: float, variance: float, alpha: float) -> dict[str, Any]:
    """``sigma = exp(ln sigma)`` with Stata's delta-method standard error and log-scale interval."""
    error = math.sqrt(max(variance, 0.0))
    z = critical_value(alpha)
    sigma = math.exp(log_sigma)
    return {"estimate": sigma, "std_error": sigma * error,
            "ci_low": math.exp(log_sigma - z * error), "ci_high": math.exp(log_sigma + z * error),
            "method": "sigma = exp(lnsigma); delta-method standard error sigma * se(lnsigma); "
                      "confidence interval = exp of the lnsigma interval"}


def rho_record(athrho: float, variance: float, alpha: float) -> dict[str, Any]:
    """``rho = tanh(athrho)`` with its delta-method standard error and transformed interval."""
    error = math.sqrt(max(variance, 0.0))
    z = critical_value(alpha)
    return {"estimate": math.tanh(athrho), "std_error": error / math.cosh(athrho) ** 2,
            "ci_low": math.tanh(athrho - z * error), "ci_high": math.tanh(athrho + z * error),
            "athrho": athrho, "athrho_std_error": error,
            "method": "rho = tanh(athrho); delta-method standard error (1 - rho^2) se(athrho); "
                      "confidence interval = tanh of the athrho interval"}


def slope_indices(design: Design, start: int = 0) -> list[int]:
    """Positions (shifted by ``start``) of the non-constant terms of a design block."""
    return [start + i for i in range(int(design.intercept), len(design.terms))]


def solver_fields(result: optimize.OptimResult) -> dict[str, Any]:
    """The solver keywords of ``core.build_result`` for a converged Newton run."""
    return {"solver": "newton_observed_hessian",
            "solver_diagnostics": {"converged": True, "iterations": result.iterations},
            "optimizer": optimizer_record(result)}
