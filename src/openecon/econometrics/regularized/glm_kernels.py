"""Bounded CPU float64 elastic-net binomial/Poisson optimization.

The objective and proximal Newton structure follow Friedman, Hastie and
Tibshirani (2010), https://doi.org/10.18637/jss.v033.i01, and the authors'
https://glmnet.stanford.edu/articles/glmnet.html.  Observation weights sum to
one here; penalty factors are literal multipliers of BOTH penalty components,
and are deliberately not rescaled as they are by glmnet.  No solver library,
screening rule, clipping of fitted means, or approximate MLE is used.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError

INNER_SWEEPS = 100
BACKTRACKS = 40
_EPS = torch.finfo(torch.float64).eps
_POISSON_ETA_LIMIT = 700.0


def solver_work(n: int, p: int, iterations: int, path_count: int) -> int:
    """Conservative scalar arithmetic bound; q includes a possible intercept.

    Includes weighted rank/SVD admission, each dense Hessian, all 100 inner
    quadratic sweeps, all 40 objective backtracks, and a free-subspace
    eigensystem at every outer iteration.  Roots sum this for every CV fit.
    Exp/log evaluations count as scalar work, rather than measured CPU time.
    """
    q = p + 1
    admission = 12 * n * q * q + 40 * q**3 + 20 * n * q
    outer = 8 * n * q * q + 8 * INNER_SWEEPS * q * q + 20 * BACKTRACKS * n + 40 * n * q + 40 * q**3
    return admission + path_count * (iterations + 1) * outer


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{name} exceeds finite float64 arithmetic.")


def _tensor(value: Tensor, dimension: int, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.ndim != dimension
    ):
        raise AnalysisError("invalid_design", f"{name} must be a CPU float64 {dimension}D tensor.")
    _finite(value, name)
    return value.detach()


def _likelihood(eta: Tensor, y: Tensor, w: Tensor, family: str):
    if not bool(torch.isfinite(eta).all()):
        return None
    if family == "binomial":
        # Binary-specific forms retain small residuals/curvature even when
        # sigmoid(eta) rounds to one and avoid softplus(eta)-eta cancellation.
        positive = torch.sigmoid(eta)
        negative = torch.sigmoid(-eta)
        losses = torch.nn.functional.softplus(torch.where(y == 1, -eta, eta))
        score = torch.where(y == 1, -negative, positive)
        curvature = positive * negative
    else:
        # Exp under/overflow would fabricate zero/inf means.  Reject the
        # numerical domain instead of clipping either likelihood or score.
        if bool((eta.abs() > _POISSON_ETA_LIMIT).any()):
            return None
        mean = eta.exp()
        losses, score, curvature = mean - y * eta, mean - y, mean
    loss = torch.dot(w, losses)
    if not all(bool(torch.isfinite(v).all()) for v in (loss, score, curvature)):
        return None
    return loss, w * score, w * curvature


def _objective(loss: Tensor, b: Tensor, l1: Tensor, ridge: Tensor) -> float:
    return float(loss + torch.dot(l1, b.abs()) + 0.5 * torch.dot(ridge, b.square()))


def _kkt(gradient: Tensor, b: Tensor, l1: Tensor) -> float:
    residual = torch.where(
        b != 0,
        (gradient + l1 * b.sign()).abs(),
        (gradient.abs() - l1).clamp_min(0),
    )
    return float(residual.max()) if len(b) else 0.0


def _check_free_rank(x: Tensor, w: Tensor, free: Tensor) -> None:
    if not bool(free.any()):
        return
    xf = x[:, free] * w.sqrt()[:, None]
    try:
        singular = torch.linalg.svdvals(xf)
    except RuntimeError as exc:
        raise AnalysisError("numerical_failure", "Unpenalized design rank audit failed.") from exc
    if len(singular) < int(free.sum()) or float(singular[-1]) <= (
        max(xf.shape) * _EPS * max(float(singular[0]), 1.0)
    ):
        raise AnalysisError(
            "unidentified_model",
            "Unpenalized columns (including the intercept) are not numerically identified.",
        )


def _free_correction(hessian: Tensor, gradient: Tensor, free: Tensor) -> float:
    """Require an actual stationary free solution, not a vanishing score at infinity.

    Along a separating logistic or zero-mean Poisson ray both score and
    curvature approach zero, but their undamped Newton ratio does not.  The
    absolute free Newton correction must therefore also vanish.  Numerically
    singular free information is refused, not interpreted as convergence.
    This is a numerical convergence safeguard, not a separation classifier.
    """
    if not bool(free.any()):
        return 0.0
    h = hessian[free][:, free]
    try:
        values, vectors = torch.linalg.eigh(h)
    except RuntimeError as exc:
        raise AnalysisError("numerical_failure", "Unpenalized information audit failed.") from exc
    _finite(values, "Unpenalized information")
    if float(values[0]) <= max(1, len(values)) * _EPS * float(values[-1]):
        raise AnalysisError(
            "unidentified_model",
            "Unpenalized information is numerically singular; a finite identified solution cannot be certified.",
        )
    correction = vectors @ ((vectors.T @ gradient[free]) / values)
    _finite(correction, "Unpenalized Newton correction")
    return float(correction.abs().max())


def _quadratic_direction(b: Tensor, gradient: Tensor, hessian: Tensor, l1: Tensor, limit: float):
    # Damping regularizes only the local proposal, never the fitted objective.
    # Free convergence certification always uses the undamped information.
    q = len(b)
    diagonal = hessian.diagonal()
    damp = max(float(diagonal.abs().max()), 1.0) * 1e-12
    h = hessian + damp * torch.eye(q, dtype=torch.float64)
    if not bool((l1 > 0).any()):
        # Smooth ridge and zero-lambda limits need no CD approximation.  The
        # exact bounded dense Newton solve also handles strongly correlated
        # but identified unpenalized designs without slow coordinate mixing.
        try:
            direction = torch.linalg.solve(h, -gradient)
        except RuntimeError as exc:
            raise AnalysisError(
                "numerical_failure", "Dense GLM Newton factorization failed."
            ) from exc
        _finite(direction, "Dense GLM Newton direction")
        return direction, 1
    candidate, score = b.clone(), gradient.clone()
    target = max(limit * 0.1, min(0.1, _kkt(gradient, b, l1)) * _kkt(gradient, b, l1))
    for sweeps in range(1, INNER_SWEEPS + 1):
        for j in range(q):
            old = candidate[j].clone()
            partial = h[j, j] * old - score[j]
            candidate[j] = partial.sign() * (partial.abs() - l1[j]).clamp_min(0) / h[j, j]
            score += h[:, j] * (candidate[j] - old)
        if _kkt(score, candidate, l1) <= target:
            break
    # One bounded exact solve on the discovered support prevents correlated
    # columns from requiring thousands of additional coordinate sweeps.  It
    # is accepted only inside that orthant and when it improves the complete
    # quadratic objective; omitted-coordinate KKT is still checked outside.
    active = (candidate != 0) | (l1 == 0)
    if bool(active.any()):
        try:
            polished = torch.zeros_like(b)
            right = h @ b - gradient - l1 * candidate.sign()
            polished[active] = torch.linalg.solve(h[active][:, active], right[active])
            constrained = active & (l1 > 0)
            same_sign = bool((polished[constrained] * candidate[constrained] > 0).all())
            if same_sign and bool(torch.isfinite(polished).all()):

                def quadratic(point):
                    delta = point - b
                    return float(
                        torch.dot(gradient, delta)
                        + 0.5 * torch.dot(delta, h @ delta)
                        + torch.dot(l1, point.abs() - b.abs())
                    )

                if quadratic(polished) <= quadratic(candidate):
                    candidate = polished
        except RuntimeError:
            # The CD proposal remains a valid bounded descent direction.
            pass
    _finite(candidate, "Proximal Newton proposal")
    return candidate - b, sweeps


def solve_path(
    z: Tensor,
    y: Tensor,
    weights: Tensor,
    lambdas: list[float],
    *,
    family: str,
    l1_ratio: float,
    penalty_factors: Tensor,
    intercept: bool,
    max_iterations: int,
    tolerance: float,
) -> list[dict]:
    """Fit each supplied strictly decreasing lambda; never return unconverged fits.

    ``z`` is the caller's standardized slope design.  ``weights`` are positive
    raw observation weights; max-first normalization prevents sum overflow.
    KKT convergence is absolute in this mean-loss objective, ``<=tolerance``.
    For free directions, an undamped Newton correction <=sqrt(tolerance) is
    additionally required.  Poisson |eta|>700 is outside this numerical domain.
    """
    z, y, weights = _tensor(z, 2, "z"), _tensor(y, 1, "y"), _tensor(weights, 1, "weights")
    factors = _tensor(penalty_factors, 1, "penalty_factors")
    n, p = z.shape
    if n < 1 or n > 5000 or p > 64 or len(y) != n or len(weights) != n or len(factors) != p:
        raise AnalysisError(
            "invalid_design",
            "Design requires 1..5000 rows, <=64 slopes, and aligned outcome/weights/factors.",
        )
    if family not in {"binomial", "poisson"}:
        raise AnalysisError("invalid_family", "family must be 'binomial' or 'poisson'.")
    if (family == "binomial" and not bool(((y == 0) | (y == 1)).all())) or (
        family == "poisson" and not bool((y >= 0).all())
    ):
        raise AnalysisError(
            "invalid_outcome", "Binomial y must be binary; Poisson y must be nonnegative."
        )
    if not bool((weights > 0).all()) or not bool((factors >= 0).all()):
        raise AnalysisError(
            "invalid_weights", "Weights must be positive and penalty factors nonnegative."
        )
    if (
        isinstance(l1_ratio, bool)
        or not isinstance(l1_ratio, (int, float))
        or not math.isfinite(l1_ratio)
        or not 0 <= l1_ratio <= 1
    ):
        raise AnalysisError("invalid_penalty", "l1_ratio must be finite in [0,1].")
    if not isinstance(intercept, bool):
        raise AnalysisError("invalid_intercept", "intercept must be boolean.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or not 1 <= max_iterations <= 10000
    ):
        raise AnalysisError("invalid_iterations", "max_iterations must be an integer in 1..10000.")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or not 1e-12 <= tolerance <= 1e-3
    ):
        raise AnalysisError("invalid_tolerance", "tolerance must be finite in 1e-12..1e-3.")
    if (
        not isinstance(lambdas, (list, tuple))
        or not 1 <= len(lambdas) <= 100
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in lambdas
        )
        or any(left <= right for left, right in zip(lambdas, lambdas[1:]))
    ):
        raise AnalysisError(
            "invalid_penalty",
            "lambdas must contain 1..100 strictly decreasing finite nonnegative values.",
        )
    scaled = weights / weights.max()
    w = scaled / scaled.sum()
    if not bool((w > 0).all()):
        raise AnalysisError(
            "numerical_failure",
            "Relative observation weights underflow float64; rescale their range.",
        )
    x = torch.cat((torch.ones((n, 1), dtype=torch.float64), z), dim=1) if intercept else z
    loads = torch.cat((torch.zeros(1, dtype=torch.float64), factors)) if intercept else factors
    b = torch.zeros(x.shape[1], dtype=torch.float64)
    if intercept:
        if family == "binomial":
            yes, no = w[y == 1].sum(), w[y == 0].sum()
            if float(yes) <= 0 or float(no) <= 0:
                raise AnalysisError(
                    "nonfinite_mle",
                    "An unpenalized binomial intercept requires both outcome classes.",
                )
            b[0] = yes.log() - no.log()
        else:
            mean = torch.dot(w, y)
            if float(mean) <= 0:
                raise AnalysisError(
                    "nonfinite_mle",
                    "All-zero Poisson outcomes have no finite unpenalized intercept.",
                )
            b[0] = mean.log()
    # A zero lambda frees every slope, including previously penalized columns.
    _check_free_rank(
        x, w, torch.ones_like(loads, dtype=torch.bool) if lambdas[-1] == 0 else loads == 0
    )
    records = []
    for value in lambdas:
        lam = float(value)
        l1, ridge = lam * float(l1_ratio) * loads, lam * (1 - float(l1_ratio)) * loads
        _finite(l1, "L1 penalty")
        _finite(ridge, "L2 penalty")
        free = loads == 0 if lam > 0 else torch.ones_like(loads, dtype=torch.bool)
        eta = x @ b
        inner_total, backtrack_total = 0, 0
        free_step = math.inf if bool(free.any()) else 0.0
        for iteration in range(max_iterations + 1):
            likelihood = _likelihood(eta, y, w, family)
            if likelihood is None:
                raise AnalysisError(
                    "numerical_failure",
                    "Current GLM likelihood is outside finite float64 arithmetic (Poisson requires |eta|<=700).",
                )
            loss, score, curvature = likelihood
            gradient = x.T @ score + ridge * b
            hessian = x.T @ (curvature[:, None] * x) + torch.diag(ridge)
            _finite(gradient, "GLM score")
            _finite(hessian, "GLM information")
            violation = _kkt(gradient, b, l1)
            objective = _objective(loss, b, l1, ridge)
            if not math.isfinite(objective):
                raise AnalysisError(
                    "numerical_failure", "GLM objective exceeds finite float64 arithmetic."
                )
            if violation <= tolerance:
                free_step = _free_correction(hessian, gradient, free)
                if free_step <= math.sqrt(tolerance):
                    records.append(
                        {
                            "penalty": lam,
                            "coefficients": b[int(intercept) :].tolist(),
                            "constant": float(b[0]) if intercept else 0.0,
                            "objective": objective,
                            "iterations": iteration,
                            "kkt_max": violation,
                            "kkt_limit": float(tolerance),
                            "free_newton_correction": free_step,
                            "inner_sweeps": inner_total,
                            "backtracks": backtrack_total,
                            "converged": True,
                        }
                    )
                    break
            if iteration == max_iterations:
                raise AnalysisError(
                    "nonconvergence",
                    f"GLM failed KKT/finite free-solution checks after {max_iterations} Newton iterations; separation or a nonfinite/unidentified MLE is not accepted as convergence.",
                )
            direction, inner = _quadratic_direction(b, gradient, hessian, l1, tolerance)
            inner_total += inner
            descent = float(
                torch.dot(gradient, direction) + torch.dot(l1, (b + direction).abs() - b.abs())
            )
            if not math.isfinite(descent) or descent >= 0:
                raise AnalysisError(
                    "nonconvergence",
                    "GLM cannot find a finite descending Newton proposal; no approximate fit is returned.",
                )
            eta_direction = x @ direction
            step = 1.0
            for backtrack in range(BACKTRACKS):
                proposal = b + step * direction
                trial_eta = eta + step * eta_direction
                trial = _likelihood(trial_eta, y, w, family)
                trial_objective = (
                    _objective(trial[0], proposal, l1, ridge) if trial is not None else math.inf
                )
                rounding = 32 * _EPS * max(1.0, abs(objective))
                if (
                    math.isfinite(trial_objective)
                    and trial_objective <= objective + 1e-4 * step * descent + rounding
                ):
                    b, eta = proposal, trial_eta
                    backtrack_total += backtrack
                    break
                step *= 0.5
            else:
                raise AnalysisError(
                    "nonconvergence",
                    "GLM objective line search exhausted 40 bounded backtracks; no unconverged output is returned.",
                )
        else:  # pragma: no cover - terminal iteration raises explicitly
            raise AssertionError("bounded GLM loop escaped")
    return records
