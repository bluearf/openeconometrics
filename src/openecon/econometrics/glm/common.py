"""Conventions shared by the glm family's analysis layer.

* **Weights** (Stata's semantics for likelihood estimators). Every likelihood,
  score and Hessian sum is ``sum_i w_i (.)`` with

  - ``fweight``: ``w_i`` the integer frequency, ``N = sum w_i``; results equal
    those of the duplicated-row data set (scores are passed unweighted with
    ``frequency=`` to ``core.ml_covariance``);
  - ``aweight``: ``w_i`` rescaled to sum to the number of rows, ``N`` = rows;
  - ``iweight``: ``w_i`` used as given (not rescaled), ``N`` = rows;
  - ``pweight``: as iweight for the point estimates and the log
    pseudolikelihood (Stata does not rescale sampling weights); they need a
    robust or cluster covariance (``robust`` is the convenience default),
    which is invariant to the scale of the weights.

* **Offset / exposure.** ``offset`` enters the linear predictor with
  coefficient one; ``exposure`` enters as ``ln(exposure)`` and must be
  positive. As in Stata the two are mutually exclusive.
* **Covariance.** ``core.ml_covariance``: ``nonrobust`` = inverse information,
  ``opg``, ``robust`` = ``N/(N-1)`` sandwich, ``cluster`` = ``G/(G-1)``
  sandwich (two cluster columns: inclusion-exclusion with the smallest G).
  With weights the likelihood is ``sum_i w_i l_i``, so the information and
  its outer-product estimate are linear in the weights,
  ``OPG = (sum_i w_i s_i s_i')^-1`` for every weight type, while the
  sandwich meat of non-frequency weights is ``sum_i (w_i s_i)(w_i s_i)'``
  (frequency weights replicate rows: ``sum_i f_i s_i s_i'``).
* **Centering.** A regressor whose mean is large relative to its spread
  (a calendar year, an identifier, a level in raw units) makes ``X'WX``
  nearly singular next to the constant: its condition number grows with
  ``(mean/sd)^2`` and the inverse information loses that many digits. With a
  constant in the model every likelihood is therefore maximized on the
  design with its other columns centered at their weighted means, which
  changes nothing but the meaning of the constant; estimates and covariance
  are mapped back exactly, ``b0 = b0_c - m'b`` and ``V = T V_c T'``
  (:func:`center_design`, :func:`uncenter`).
* **Newton-Raphson.** ``engines.optimize.maximize_newton`` never declares
  convergence in a non-concave region or on a monotone likelihood; a failed
  run is returned (not raised) so that each estimator can say what happened
  (separation, a boundary solution) in its own words.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, ModelFrame, kernel_call, make_spec, ml_covariance
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec


@dataclass
class Weights:
    user: Tensor               # weights of every likelihood sum (ones when unweighted)
    frequency: Tensor | None   # the integer frequencies under fweights
    nobs: int                  # N as Stata counts it
    weighted: bool

    def for_screen(self) -> Tensor | None:
        return self.user if self.weighted else None


def likelihood_weights(frame: ModelFrame) -> Weights:
    """Weights as the likelihood sums receive them and N as Stata counts it."""
    raw = frame.weights()
    if raw is None:
        return Weights(torch.ones(frame.n, dtype=torch.float64), None, frame.n, False)
    kind = frame.spec.weight_type
    if kind == "fweight":
        return Weights(raw, raw, int(round(float(raw.sum()))), True)
    if kind == "aweight":
        return Weights(raw * (frame.n / float(raw.sum())), None, frame.n, True)
    if bool((raw < 0).any()):
        raise AnalysisError("negative_weights", "Negative iweights are not supported by this "
                            "estimator: every weight must be positive.")
    return Weights(raw, None, frame.n, True)


def build_spec(estimator: str, **fields: Any) -> ModelSpec:
    """``core.make_spec`` for convenience functions: every rejection is an AnalysisError.

    The registry reports an invalid option, role, weight type or covariance as
    a validation ``ValueError``; a convenience function turns it into
    ``AnalysisError('invalid_spec', ...)`` with the registry's own message.
    """
    try:
        return make_spec(estimator, **fields)
    except AnalysisError:
        raise
    except ValueError as exc:
        errors = exc.errors() if hasattr(exc, "errors") else []
        message = "; ".join(str(error.get("msg", "")).removeprefix("Value error, ")
                            for error in errors) or str(exc)
        raise AnalysisError("invalid_spec", message) from exc


def resolve_covariance(covariance: str | None, cluster: Any, weight_type: str | None) -> str | None:
    """Convenience-function default: pweights imply the robust covariance."""
    if covariance is None and weight_type == "pweight" and not cluster:
        return "robust"
    return covariance


def check_pweights(frame: ModelFrame) -> None:
    if frame.spec.weight_type == "pweight" and frame.spec.covariance in {"nonrobust", "opg"}:
        raise AnalysisError(
            "unsupported_covariance",
            "pweights are sampling weights and need a robust covariance: choose "
            "covariance='robust' (Stata's vce(robust)) or give a cluster column.")


def linear_offset(frame: ModelFrame) -> Tensor | None:
    """``offset`` or ``ln(exposure)`` of the current sample, or None."""
    offset, exposure = frame.role("offset"), frame.role("exposure")
    if offset and exposure:
        raise AnalysisError("invalid_spec", "Give either offset or exposure, not both "
                            "(exposure is ln(exposure) used as an offset).")
    if offset:
        return frame.numeric(offset[0])
    if exposure:
        values = frame.numeric(exposure[0])
        if bool((values <= 0).any()):
            raise AnalysisError("invalid_exposure", f"Exposure column '{exposure[0]}' must be "
                                "strictly positive (its logarithm enters the linear predictor).")
        return torch.log(values)
    return None


def count_outcome(frame: ModelFrame, y: Tensor, command: str, *, note: bool = True) -> None:
    """Stata's rule for count models: negative values are an error, non-integers a note.

    ``note=False`` is for pseudo-likelihood estimators whose outcome is not
    meant to be a count (``ppmlhdfe``: trade flows, expenditures).
    """
    if bool((y < 0).any()):
        raise AnalysisError("invalid_count_outcome", f"{command} needs a nonnegative outcome; "
                            f"'{frame.spec.outcome}' contains negative values.")
    if note and bool((y != y.round()).any()):
        frame.warn(f"The outcome '{frame.spec.outcome}' has non-integer values; the estimates "
                   "are quasi-likelihood estimates and you are responsible for interpreting "
                   "a noncount outcome.")


def center_design(design: Design, weights: Tensor) -> tuple[Tensor, Tensor | None]:
    """The design with its non-constant columns centered at their weighted means.

    Returns ``(x, means)``; ``means`` is None (and ``x`` the design itself)
    when there is no constant to absorb the shift or nothing to center.
    """
    x = design.x
    if not design.intercept or x.shape[1] < 2:
        return x, None
    means = (weights[:, None] * x[:, 1:]).sum(dim=0) / weights.sum()
    centered = x.clone()
    centered[:, 1:] -= means
    return centered, means


def to_centered(beta: Tensor, means: Tensor | None) -> Tensor:
    """Coefficients of the centered design: the constant becomes ``b0 + m'b``."""
    if means is None:
        return beta
    centered = beta.clone()
    centered[0] = beta[0] + means @ beta[1:1 + len(means)]
    return centered


def uncenter(theta: Tensor, covariance: Tensor | None,
             blocks: list[tuple[int, Tensor | None]]) -> tuple[Tensor, Tensor | None]:
    """Map estimates of centered designs back to the original columns.

    ``blocks`` lists ``(position of the constant in theta, means)`` for every
    centered design block. With ``T`` the identity except
    ``T[constant, slopes] = -means``: ``theta = T theta_c`` and
    ``V = T V_c T'``. Slope estimates and their covariance are unchanged.
    """
    if all(means is None for _, means in blocks):
        return theta, covariance
    transform = torch.eye(len(theta), dtype=torch.float64)
    for start, means in blocks:
        if means is not None:
            transform[start, start + 1:start + 1 + len(means)] = -means
    moved = None if covariance is None else transform @ covariance @ transform.T
    return transform @ theta, moved


def safe_exp(value: float) -> float:
    """``exp`` that overflows to infinity instead of raising (reported as missing)."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def require_terms(design: Design) -> None:
    """A model needs at least one term (the constant counts)."""
    if not design.terms:
        raise AnalysisError("empty_design", "The model has no terms to estimate: give at least "
                            "one regressor that varies or keep the constant (intercept=True).")


def require_observations(nobs: int, parameters: int) -> None:
    if nobs <= parameters:
        raise AnalysisError("insufficient_observations", f"Estimation needs more observations "
                            f"({nobs}) than parameters ({parameters}).")


def likelihood_covariance(frame: ModelFrame, hessian: Tensor, rows: Callable[[], Tensor],
                          weights: Weights) -> tuple[Tensor, dict[str, Any]]:
    """``core.ml_covariance`` with the family's weight semantics.

    ``rows()`` returns the per-observation score rows before user weights; it
    is only called when the covariance needs them, so the conventional
    covariance never allocates an n-by-k score matrix. A singular information
    matrix is reported with advice instead of the kernel's code.
    """
    try:
        if frame.spec.covariance == "nonrobust":
            return ml_covariance(frame, hessian=hessian, nobs=weights.nobs)
        scores = kernel_call(rows)
        if weights.frequency is not None:
            return ml_covariance(frame, hessian=hessian, scores=scores, nobs=weights.nobs,
                                 frequency=weights.frequency)
        if weights.weighted and frame.spec.covariance == "opg":
            # The OPG estimates the information of sum_i w_i l_i, which is linear in w_i.
            return ml_covariance(frame, hessian=hessian, scores=scores, nobs=weights.nobs,
                                 frequency=weights.user)
        if weights.weighted:
            scores = scores * weights.user[:, None]
        return ml_covariance(frame, hessian=hessian, scores=scores, nobs=weights.nobs)
    except AnalysisError as exc:
        if exc.code != "singular_information":
            raise
        raise AnalysisError(
            "singular_information",
            "The information matrix is not positive definite at the estimates, so standard "
            "errors are undefined. A parameter is not identified: check for regressors that "
            "predict the outcome perfectly or that vary in very few observations.") from exc


def maximize(objective: Callable[[Tensor], Any], start: Tensor, *, what: str,
             max_iter: int = 200, tolerance: float = 1e-10,
             callback: Callable[[int, Tensor, float], Any] | None = None,
             ) -> optimize.OptimResult:
    """Newton-Raphson that returns a non-converged run instead of raising."""
    precision_check = getattr(objective, "check_precision", None)
    try:
        if precision_check is not None:
            precision_check(start)
            objective.precision_rejected = False
            objective.reject_precision_trials = True
        try:
            result = optimize.maximize_newton(
                objective, start, value_fn=getattr(objective, "value", None), max_iter=max_iter,
                step_tol=tolerance, scaled_gradient_tol=tolerance, callback=callback,
                raise_on_failure=False)
        finally:
            if precision_check is not None:
                objective.reject_precision_trials = False
        if precision_check is not None:
            precision_check(result.theta)
            if objective.precision_rejected:
                result.diagnostics["precision_rejected"] = True
        return result
    except KernelError as exc:
        if exc.code == "invalid_start":
            raise AnalysisError("invalid_start", f"The {what} likelihood is not finite at the "
                                "starting values; check the outcome's range and the scale of "
                                "the regressors.") from exc
        raise AnalysisError(exc.code,
                            f"The {what} likelihood could not be maximized: {exc}") from exc
    except (FloatingPointError, OverflowError, RuntimeError) as exc:
        raise AnalysisError("numerical_failure", f"The {what} likelihood could not be "
                            f"evaluated: {exc}") from exc


def optimizer_record(result: optimize.OptimResult) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {"method": result.method, "iterations": result.iterations,
            "converged": result.converged, "gradient_max": diagnostics.get("gradient_max"),
            "backtracks": diagnostics.get("backtracks"),
            "nonconcave_iterations": diagnostics.get("nonconcave_iterations"),
            "message": diagnostics.get("message")}


def slope_indices(terms: list[str]) -> list[int]:
    return [i for i, term in enumerate(terms) if term != "Intercept"]


def weighted_mean(values: Tensor, weights: Tensor) -> float:
    return float((weights * values).sum() / weights.sum())


def solve_weighted(x: Tensor, y: Tensor, weights: Tensor):
    """QR least squares on an already screened design (kernel errors translated)."""
    from openecon.engines.linalg import least_squares

    return kernel_call(least_squares, x, y, weights, drop_collinear=True, tol=0.0)
