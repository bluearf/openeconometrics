"""Maximization of the concentrated ARMA likelihood (``kernels.ArmaLikelihood``).

sigma^2 is concentrated out; the regression and ARMA parameters psi are found by BFGS
(``engines.optimize.maximize_bfgs``) with the analytic gradient. ``maximize`` chooses
the starting values, guards against local maxima on small samples by screening several
starts, and returns the maximizer with the Hessian of the concentrated log likelihood.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.kernels import ArmaLikelihood, shrink, starting_values
from openecon.econometrics.core import kernel_call
from openecon.engines import optimize
from openecon.engines.contracts import KernelError

_MULTISTART_ROWS = 20_000   # largest sample maximized from every starting value
_DISTINCT_MAXIMUM = 1e-7    # log-likelihood gap that separates two local maxima
_SCREENING_ITERATIONS = 100
_SCREENING_TOLERANCE = 1e-6
_SPLIT_HESSIAN = 3          # regressors from which the Hessian is differentiated in two parts


def _ascend(like: ArmaLikelihood, start: Tensor, tolerance: float, max_iterations: int,
            curvature: bool = True, screening: bool = False):
    """BFGS on the concentrated likelihood from one start.

    ``curvature`` seeds the inverse Hessian with the Gauss-Newton information (long first
    steps); without it BFGS starts from the identity (short, cautious first steps).
    ``screening`` judges convergence by the Gauss-Newton curvature instead of the numerical
    Hessian, which makes a run several times cheaper; such a run only ranks starting
    values and is always followed by a full run from its end point.
    """
    inverse = None
    information = like.gauss_newton(start) if curvature else None
    if information is not None:
        try:
            inverse = optimize.information_inverse(information)
        except KernelError:
            inverse = None

    def gauss_newton(theta: Tensor) -> Tensor:
        value = like.gauss_newton(theta)
        return torch.full((like.n_psi, like.n_psi), math.nan, dtype=torch.float64) \
            if value is None else -value

    if screening:
        result = optimize.maximize_bfgs(
            like.concentrated, start, max_iter=min(max_iterations, _SCREENING_ITERATIONS),
            gradient_tol=max(tolerance, _SCREENING_TOLERANCE), initial_inverse_hessian=inverse,
            raise_on_failure=False, hessian_fn=gauss_newton)
    else:
        result = optimize.maximize_bfgs(
            like.concentrated, start, max_iter=max_iterations, gradient_tol=tolerance,
            initial_inverse_hessian=inverse, raise_on_failure=False,
            hessian_fn=_split_hessian(like))
    return result, "Gauss-Newton" if inverse is not None else "identity"


def _split_hessian(like: ArmaLikelihood):
    """Hessian of the concentrated likelihood for a model with several regressors.

    The engine's default differentiates the analytic gradient along every coordinate
    with Ridders extrapolation (six gradients per coordinate). The gradient is close to
    linear in the regression coefficients (the sum of squares is exactly quadratic in
    them), so with _SPLIT_HESSIAN or more regressors plain central differences
    (``numerical_hessian(levels=1)``, two gradients per coordinate) are used for the
    whole matrix and only the ARMA block is recomputed with the extrapolation. Returns
    None (the engine's default) for small models.
    """
    k = like.k
    if k < _SPLIT_HESSIAN or not like.n_arma:
        return None

    def hessian(theta: Tensor) -> Tensor:
        full = optimize.numerical_hessian(lambda point: like.concentrated(point)[1], theta,
                                          levels=1)
        head = theta[:k]
        full[k:, k:] = optimize.numerical_hessian(
            lambda tail: like.concentrated(torch.cat([head, tail]))[1][k:], theta[k:])
        return full

    return hessian


def _conditional_start(like: ArmaLikelihood, x: Tensor, start: Tensor, tolerance: float,
                       max_iterations: int) -> Tensor | None:
    """The conditional (CSS) estimate pulled into the stationary and invertible region."""
    p, q, sp, sq = like.sizes
    conditional = (like.replay_conditional_clone() if hasattr(like, "replay_conditional_clone") else
                   ArmaLikelihood(like.y, x, p=p, q=q, seasonal_p=sp, seasonal_q=sq,
                                  period=like.period if sp or sq else 0, exact=False))
    try:
        result, _ = _ascend(conditional, start, tolerance, max_iterations)
    except KernelError:
        return None
    if not result.converged:
        return None
    phi, theta, sphi, stheta = like.split(result.theta)
    blocks = (shrink(phi, moving_average=False), shrink(theta, moving_average=True),
              shrink(sphi, moving_average=False), shrink(stheta, moving_average=True))
    candidate = result.theta.clone()
    candidate[like.k:] = torch.tensor([value for block in blocks for value in block],
                                      dtype=torch.float64)
    return candidate


def maximize(like: ArmaLikelihood, y: Tensor, x: Tensor, tolerance: float,
              max_iterations: int) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Maximize the concentrated likelihood; returns psi, its Hessian and an optimizer record.

    Starting values: Hannan-Rissanen and OLS with white-noise errors. A large sample is
    maximized from the better of the two. The likelihood of a mixed or over-fitted ARMA
    model can have several local maxima, so up to _MULTISTART_ROWS observations a model
    with both AR and MA terms (or three or more ARMA terms) is tried from each start (and,
    for exact ML,
    from the conditional estimate) with the Gauss-Newton initial curvature, and from the
    Hannan-Rissanen start also with the identity, in cheap screening runs; the full
    maximization then starts from the end point of the best screening run (the next best
    if it fails). ``record["iterations"]`` counts the BFGS iterations of that screening
    run and of the final run together.
    """
    if like.n_psi == 0:
        return (torch.zeros(0, dtype=torch.float64), torch.zeros((0, 0), dtype=torch.float64),
                {"method": "closed_form", "iterations": 0, "converged": True})
    if hasattr(like, "replay_starting_values"):
        start = kernel_call(like.replay_starting_values)
        residual_ss, response_ss = like.start_residual_ss, like.response_ss
    else:
        start = kernel_call(starting_values, y, x, like.sizes, like.period)
        residual = y - x @ start[:like.k] if like.k else y
        residual_ss, response_ss = float(torch.dot(residual, residual)), float(torch.dot(y, y))
    if residual_ss <= 1e-20 * max(response_ss, 1e-300):
        raise AnalysisError("perfect_fit", "The regressors reproduce the (differenced) outcome "
                            "exactly; there is no disturbance left to model. Remove regressors "
                            "that determine the outcome.")
    candidates = [("hannan_rissanen", start)]
    if like.n_arma:
        white = start.clone()
        white[like.k:] = 0.0
        candidates.append(("ols_white_noise", white))
    p, q, seasonal_p, seasonal_q = like.sizes
    several = (p + seasonal_p) * (q + seasonal_q) > 0 and like.n <= _MULTISTART_ROWS \
        or like.n_arma >= 3 and like.n <= _MULTISTART_ROWS
    if several and like.exact:
        conditional = _conditional_start(like, x, start, tolerance, max_iterations)
        if conditional is not None:
            candidates.append(("conditional_estimate", conditional))
    feasible = [(like.concentrated(candidate)[0], name, candidate)
                for name, candidate in candidates]
    feasible = sorted((item for item in feasible if math.isfinite(item[0])),
                      key=lambda item: -item[0])
    if not feasible:
        raise AnalysisError("invalid_start", "The likelihood could not be evaluated at the "
                            "starting values; check the series for extreme values.")
    best = None
    failure = "no starting value could be improved."
    tried: dict[str, float | None] = {}
    # (label, starting point, BFGS iterations already spent on it by its screening run)
    finalists = [(name, candidate, 0) for _, name, candidate in feasible[:1]]
    if several:
        # Rank the starts by cheap screening runs, with long (Gauss-Newton) and with
        # cautious (identity) first steps: they can end in different basins.
        ranked = []
        for seeded in (True, False):
            for _, name, candidate in feasible:
                if not seeded and name != "hannan_rissanen":
                    continue
                label = name if seeded else f"{name}_identity"
                try:
                    trial, _ = _ascend(like, candidate, tolerance, max_iterations, seeded, True)
                except KernelError:
                    tried[label] = None
                    continue
                tried[label] = trial.value if math.isfinite(trial.value) else None
                if math.isfinite(trial.value):
                    ranked.append((trial.value, label, trial.theta, trial.iterations))
        ranked.sort(key=lambda item: -item[0])
        distinct = [item for i, item in enumerate(ranked)
                    if i == 0 or item[0] < ranked[i - 1][0] - _DISTINCT_MAXIMUM]
        finalists = [(label, theta, spent) for _, label, theta, spent in distinct] + finalists
    for name, candidate, spent in finalists:
        try:
            result, curvature = _ascend(like, candidate, tolerance, max_iterations)
        except KernelError as exc:
            failure = str(exc)
            continue
        if result.converged:
            best = (result, name, curvature, spent)
            break
        failure = str(result.diagnostics.get("message"))
    if best is None:
        raise AnalysisError(
            "nonconvergence",
            f"The ARIMA likelihood maximization did not converge: {failure} "
            "Try a simpler model (fewer AR/MA terms), check for over-differencing or common AR "
            "and MA factors, raise max_iterations, or use method='css'.")
    result, name, curvature, spent = best
    # The reported count is the whole path to the maximum: the screening run that
    # produced the starting point of the final run plus the final run itself (which
    # often needs no further step, because screening already converged).
    record = {
        "method": result.method, "iterations": result.iterations + spent, "converged": True,
        "screening_iterations": spent, "final_iterations": result.iterations,
        "gradient_max": result.diagnostics.get("gradient_max"),
        "scaled_gradient": result.diagnostics.get("scaled_gradient"),
        "function_evaluations": result.diagnostics.get("function_evaluations"),
        "polish_iterations": result.diagnostics.get("polish_iterations"),
        "message": result.diagnostics.get("message"), "starting_values": name,
        "starts": tried,
        "gradient": "analytic",
        "hessian": "numerical (Ridders-extrapolated central differences of the analytic "
                   "gradient of the concentrated log likelihood); sigma row analytic"
                   if _split_hessian(like) is None else
                   "numerical (central differences of the analytic gradient of the concentrated "
                   "log likelihood, Ridders-extrapolated in the ARMA block); sigma row analytic",
        "initial_curvature": curvature,
    }
    return result.theta, result.hessian, record
