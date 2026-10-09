"""CPU float64 stacked control-function estimating equations.

The covariance uses the observed derivative of the full two-stage score.
Gamma and inverse Gaussian use a fixed working dispersion of one; the
optimizer uses minus half the GLM deviance. Reported criteria omit only the
explicit outcome-only constants specified for each working score; they do
not imply a distributional fit claim for the positive-mean QMLE models.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor

from openecon.econometrics.glm.families import (
    Binomial, Cloglog, Gamma, Gaussian, Identity, InverseGaussian, Log, Logit,
    Poisson, Probit,
)
from openecon.econometrics.glm.kernels import GlmObjective
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares
from openecon.engines.optimize import maximize_newton
from openecon.engines.separation import certify_separation
from openecon.resources import plan_workspace

MAX_ROWS = 5000
MAX_EXOG = 16
MAX_INSTRUMENTS = 16
MAX_JOINT_PARAMETERS = 36
MAX_ITERATIONS = 1000
DEFAULT_MAX_WORK = 10_000_000_000
CERTIFICATE_MAX_PASSES = 64
MAX_CONDITION = 1e8
KINDS = frozenset({"gaussian", "logit", "probit", "cloglog", "poisson",
                   "gamma", "inverse_gaussian", "fractional_logit"})
_EPS = torch.finfo(torch.float64).eps


def certificate_work(n: int, width: int, kind: str) -> int:
    """Preallocation work bound for the binary/fractional separation proof.

    Each pass replays at most 2n signed rows, then solves a box LP with at
    most max(8,4*width) retained constraints and 100 primal-dual steps.
    This includes normalization, cut generation, face SVD and both Newton
    directions; exhausted or ambiguous certificates refuse estimation.
    """
    if kind not in {"logit", "probit", "cloglog", "fractional_logit"}:
        return 0
    n, width = _positive_integer(n, "n"), _positive_integer(width, "width")
    return CERTIFICATE_MAX_PASSES * (64 * n * width + 100 * (64 * width**3 + 256 * width**2))


def _family_link(kind: str):
    if not isinstance(kind, str) or kind not in KINDS:
        raise KernelError("unsupported_model", "Unknown control-function outcome kind.")
    return {
        "gaussian": (Gaussian, Identity), "logit": (Binomial, Logit),
        "probit": (Binomial, Probit), "cloglog": (Binomial, Cloglog),
        "poisson": (Poisson, Log), "gamma": (Gamma, Log),
        "inverse_gaussian": (InverseGaussian, Log),
        "fractional_logit": (lambda: Binomial(combinatorial=False), Logit),
    }[kind]


def _tensor(value: Tensor, name: str, shape: tuple[int, ...] | None = None) -> None:
    if not isinstance(value, Tensor) or value.device.type != "cpu" \
            or value.dtype != torch.float64 or value.layout != torch.strided \
            or (shape is not None and value.shape != shape):
        raise KernelError("invalid_design", f"{name} must be a CPU float64 tensor with the declared shape.")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise KernelError("invalid_options", f"{name} must be a positive integer.")
    return int(value)


def _admit(z: Tensor, x: Tensor, d: Tensor, y: Tensor, kind: str,
           cluster_codes: Tensor | None, iterations: int, max_work: int):
    _family_link(kind)
    _tensor(z, "z")
    _tensor(x, "x")
    if z.ndim != 2 or x.ndim != 2:
        raise KernelError("invalid_design", "z and x must be matrices.")
    n, kz = z.shape
    nx, kx = x.shape
    joint = kz + kx + 1
    excluded = kz - (kx - 1)
    # A shared prefix may contain an intercept plus exogenous columns, or
    # only exogenous columns. The public layer counts its declared intercept.
    if nx != n or not 1 <= n <= MAX_ROWS or kx < 1 or kx - 1 > MAX_EXOG + 1 \
            or not 1 <= excluded <= MAX_INSTRUMENTS or joint > MAX_JOINT_PARAMETERS \
            or n <= joint:
        raise KernelError("unsupported_dimensions", "Require at most 5000 rows, 16 exogenous and 16 excluded-instrument columns, 36 joint parameters, and n greater than the joint width.")
    _tensor(d, "d", (n,))
    _tensor(y, "y", (n,))
    max_work = _positive_integer(max_work, "max_work")
    # At most 50 line-search halvings and 120 ridge attempts per Newton step.
    # 128 derivative evaluations is a conservative bound including value trials.
    kq = kx + 1
    work = 20 * n * (kz * kz + kq * kq + joint * joint) + 20 * joint**3
    work += iterations * (128 * n * kq * kq + 256 * kq**3)
    separation_work = certificate_work(n, kq, kind)
    work += separation_work
    if work > max_work:
        raise KernelError("work_limit", f"Conservative control-function work estimate {work} exceeds max_work={max_work}.")
    capacity = max(8, 4 * kq)
    separation_workspace = 8 * (3 * (capacity + 2 * kq) * kq + 12 * kq**2
                                + 24 * (capacity + 2 * kq))
    plan = plan_workspace("control_function_joint", {
        "model_inputs": 8 * n * (kz + kx + 2),
        "first_stage_qr": 8 * (4 * n * kz + 8 * kz**2),
        "outcome_design_and_qr": 8 * (5 * n * kq + 8 * kq**2),
        "objective_vectors_and_line_search": 8 * (24 * n + 16 * kq**2),
        "joint_scores_and_cluster_sums": 8 * 5 * n * joint,
        "stacked_derivatives_and_solves": 8 * 18 * joint**2,
        "optimizer_history": 8 * 256 + 8192,
        "separation_scaled_design_and_replay": 8 * (3 * n * kq + 4 * n) if separation_work else 0,
        "separation_primal_dual_workspace": separation_workspace if separation_work else 0,
    }).record()
    for name, value in (("z", z), ("x", x), ("d", d), ("y", y)):
        if not bool(torch.isfinite(value).all()):
            raise KernelError("non_finite_values", f"{name} must be finite.")
    if not torch.equal(z[:, :kx - 1], x[:, :kx - 1]) \
            or not torch.equal(x[:, -1], d):
        raise KernelError("invalid_design", "Designs must share the intercept/exogenous prefix and x must end in d.")
    if kind in {"logit", "probit", "cloglog"}:
        if not bool(((y == 0) | (y == 1)).all()):
            raise KernelError("invalid_outcome", "Binary outcomes must be exactly zero or one.")
    elif kind == "fractional_logit":
        if not bool(((y >= 0) & (y <= 1)).all()):
            raise KernelError("invalid_outcome", "Fractional outcomes must lie in [0, 1].")
    elif kind == "poisson":
        if not bool(((y >= 0) & (y == torch.floor(y))).all()):
            raise KernelError("invalid_outcome", "Poisson outcomes must be nonnegative integers.")
    elif kind in {"gamma", "inverse_gaussian"} and not bool((y > 0).all()):
        raise KernelError("invalid_outcome", "Positive-mean outcomes must be strictly positive.")
    has_intercept = kx > 1 and bool((x[:, 0] == 1).all())
    if has_intercept and kind in {"logit", "probit", "cloglog", "fractional_logit"} \
            and (bool((y == 0).all()) or bool((y == 1).all())):
        raise KernelError("separation_detected", "A constant boundary outcome has no finite intercept estimate.")
    if has_intercept and kind == "poisson" and bool((y == 0).all()):
        raise KernelError("separation_detected", "An all-zero Poisson outcome has no finite intercept estimate.")
    groups = None
    if cluster_codes is not None:
        if not isinstance(cluster_codes, Tensor) or cluster_codes.device.type != "cpu" \
                or cluster_codes.dtype != torch.int64 or cluster_codes.shape != (n,) \
                or cluster_codes.layout != torch.strided:
            raise KernelError("invalid_clusters", "cluster_codes must be a CPU int64 vector [n].")
        if int(cluster_codes.min()) < 0 or int(cluster_codes.max()) >= n:
            raise KernelError("invalid_clusters", "Cluster codes must be bounded dense nonnegative integers.")
        groups = int(cluster_codes.max()) + 1
        if torch.unique(cluster_codes).numel() != groups:
            raise KernelError("invalid_clusters", "Cluster codes must cover exactly 0..G-1.")
        if groups <= joint:
            raise KernelError("singular_covariance", "A positive full joint CR0 covariance requires more clusters than joint parameters.")
    return n, kz, kq, joint, groups, work, plan


def _rank(matrix: Tensor, name: str) -> None:
    scale = matrix.abs().amax(dim=0)
    if not bool((scale > 0).all()):
        raise KernelError("rank_deficient", f"{name} has a zero column.")
    singular = torch.linalg.svdvals(matrix / scale)
    if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= 0 \
            or float(singular[0] / singular[-1]) > MAX_CONDITION:
        raise KernelError("rank_deficient", f"{name} is singular or numerically ill-conditioned after column scaling.")


def _positive_matrix(matrix: Tensor, name: str) -> None:
    if not bool(torch.isfinite(matrix).all()) or not bool((matrix.diag() > 0).all()):
        raise KernelError("singular_covariance", f"{name} must have finite positive diagonal entries.")
    scale = matrix.diag().sqrt()
    corr = matrix / scale[:, None] / scale[None, :]
    if not bool(torch.isfinite(corr).all()):
        raise KernelError("singular_covariance", f"{name} cannot be equilibrated within float64 precision.")
    eigen = torch.linalg.eigvalsh((corr + corr.T) * 0.5)
    if not bool(torch.isfinite(eigen).all()) or float(eigen[0]) <= 64 * matrix.shape[0] * _EPS:
        raise KernelError("singular_covariance", f"{name} is not positive definite to float64 precision.")


def _objective(q: Tensor, y: Tensor, kind: str) -> GlmObjective:
    family, link = _family_link(kind)
    return GlmObjective(q, y, torch.ones(y.shape, dtype=torch.float64, device="cpu"),
                        None, family(), link())


def _residual(z: Tensor, d: Tensor, gamma: Tensor) -> Tensor:
    fitted = z @ gamma
    residual = d - fitted
    scale = max(float(d.abs().max()), float(fitted.abs().max()))
    if not bool(torch.isfinite(residual).all()) or float(residual.abs().max()) <= 64 * _EPS * scale:
        raise KernelError("rank_deficient", "The first-stage residual has zero variation to float64 precision.")
    return residual


def _state(objective: GlmObjective, beta: Tensor):
    state = objective.state(beta)
    if state is None:
        raise KernelError("numerical_domain", "The fitted mean or working criterion is outside the finite numerical domain.")
    score, derivative, _ = objective.pieces(state)
    if not bool(torch.isfinite(score).all()) or not bool(torch.isfinite(derivative).all()):
        raise KernelError("numerical_domain", "Observed score derivatives are not finite; no clipping is applied.")
    return state, score, derivative


def _separation(q: Tensor, y: Tensor, kind: str) -> dict[str, Any]:
    if kind not in {"logit", "probit", "cloglog", "fractional_logit"}:
        return {}
    scaled = q / q.abs().amax(dim=0)
    if bool((q[:, 0] == 1).all()):
        scaled[:, 1:] -= scaled[:, 1:].mean(dim=0)
    rms = scaled.square().mean(dim=0).sqrt()
    if not bool((rms > 0).all()):
        raise KernelError("separation_check_failed", "The signed design cannot be standardized at float64 precision.")
    scaled /= rms
    endpoint = (y == 0) | (y == 1)
    interior = ~endpoint
    sign = 2 * (y == 1).to(torch.float64) - 1
    # For an interior fractional response, recession requires q_i'd=0.
    # Its two opposite halfspaces cancel exactly in the mean objective.
    count = len(y) + int(interior.sum())
    objective = (scaled[endpoint] * sign[endpoint, None]).sum(0) / count

    def replay():
        for start in range(0, len(y), 512):
            stop = start + 512
            yield scaled[start:stop] * sign[start:stop, None]
            chosen = interior[start:stop]
            if bool(chosen.any()):
                yield scaled[start:stop][chosen]

    return certify_separation(replay, objective, q.shape[1],
                              max_passes=CERTIFICATE_MAX_PASSES)


def _criterion(kind: str, y: Tensor, state) -> Tensor:
    eta, mu = state.eta, state.mu
    if kind == "gaussian":
        rows = -0.5 * (y - mu).square()
    elif kind in {"logit", "fractional_logit"}:
        rows = y * eta - torch.nn.functional.softplus(eta)
    elif kind in {"probit", "cloglog"}:
        rows = torch.xlogy(y, mu) + torch.xlogy(1 - y, state.comp)
    elif kind == "poisson":
        rows = y * eta - mu - torch.lgamma(y + 1)
    elif kind == "gamma":
        rows = -y / mu - eta
    else:
        rows = -0.5 * y / mu.square() + mu.reciprocal()
    value = rows.sum()
    if not bool(torch.isfinite(value)):
        raise KernelError("numerical_domain", "The reported working criterion is not finite.")
    return value


def _evaluate(z: Tensor, x: Tensor, d: Tensor, y: Tensor, kind: str,
              gamma: Tensor, beta: Tensor, cluster_codes: Tensor | None,
              groups: int | None, work: int, plan: dict[str, Any],
              certificate: dict[str, Any] | None = None) -> dict[str, Any]:
    _rank(z, "First-stage design")
    residual = _residual(z, d, gamma)
    q = torch.cat((x, residual[:, None]), dim=1)
    _rank(q, "Residual-inclusion design")
    if certificate is None:
        certificate = _separation(q, y, kind)
    objective = _objective(q, y, kind)
    state, s, h = _state(objective, beta)
    if kind == "gaussian" and float(s.abs().max()) <= 64 * _EPS * max(
            float(y.abs().max()), float(state.mu.abs().max())):
        raise KernelError("singular_covariance", "The outcome residual variance is zero to float64 precision.")
    kz, kq = z.shape[1], q.shape[1]
    scores = torch.cat((z * residual[:, None], q * s[:, None]), dim=1)
    bread = torch.zeros((kz + kq, kz + kq), dtype=torch.float64, device="cpu")
    bread[:kz, :kz] = z.T @ z
    cross = beta[-1] * (q * h[:, None]).T @ z
    cross[-1] += s @ z  # d(q_i)/d(gamma)' contributes the residual-coordinate score.
    bread[kz:, :kz] = cross
    bread[kz:, kz:] = -(q * h[:, None]).T @ q
    _positive_matrix(bread[kz:, kz:], "Observed outcome information")
    if not bool(torch.isfinite(bread).all()) or not bool(torch.isfinite(scores).all()):
        raise KernelError("numerical_domain", "The full stacked derivative or score is not finite.")
    # Row/column equilibration preserves the nonsymmetric estimating equations.
    # Solve A V A' = M with general solves; neither B nor its cross block is symmetrized.
    row_scale = bread.abs().amax(dim=1).reciprocal()
    a = row_scale[:, None] * bread
    col_scale = a.abs().amax(dim=0).reciprocal()
    a = a * col_scale[None, :]
    if not bool(torch.isfinite(a).all()):
        raise KernelError("singular_bread", "The stacked derivative cannot be equilibrated within float64 precision.")
    singular = torch.linalg.svdvals(a)
    if float(singular[-1]) <= 0 or float(singular[0] / singular[-1]) > MAX_CONDITION:
        raise KernelError("singular_bread", "The stacked observed derivative is numerically singular.")
    if groups is None:
        meat = scores.T @ scores
    else:
        summed = torch.zeros((groups, scores.shape[1]), dtype=torch.float64, device="cpu")
        summed.index_add_(0, cluster_codes, scores)
        meat = summed.T @ summed
    if not bool(torch.isfinite(meat).all()):
        raise KernelError("numerical_domain", "The stacked score meat overflows float64.")
    scaled_meat = row_scale[:, None] * meat * row_scale[None, :]
    left = torch.linalg.solve(a, scaled_meat)
    covariance = torch.linalg.solve(a, left.T).T
    covariance = col_scale[:, None] * covariance * col_scale[None, :]
    covariance = (covariance + covariance.T) * 0.5
    _positive_matrix(covariance, "Full joint sandwich covariance")
    return {
        "gamma": gamma.clone(), "beta": beta.clone(), "residual": residual,
        "design": q, "fitted": state.mu, "row_scores": scores,
        "bread": bread, "meat": meat, "joint_covariance": covariance,
        "scalar_score": s, "scalar_derivative": h,
        "criterion": _criterion(kind, y, state),
        "optimizer": {"method": "evaluation", "iterations": 0, "converged": False, **certificate},
        "cluster_count": groups, "resource_plan": plan, "work_estimate": work,
    }


@torch.no_grad()
def evaluate_joint(z: Tensor, x: Tensor, d: Tensor, y: Tensor, kind: str,
                   gamma: Tensor, beta: Tensor, *, cluster_codes: Tensor | None = None,
                   max_work: int = DEFAULT_MAX_WORK) -> dict[str, Any]:
    """Evaluate the exact stacked equations at supplied parameters; do not refit."""
    with torch.device("cpu"):
        _, kz, kq, _, groups, work, plan = _admit(z, x, d, y, kind, cluster_codes,
                                                0, max_work)
        _tensor(gamma, "gamma", (kz,))
        _tensor(beta, "beta", (kq,))
        if not bool(torch.isfinite(gamma).all()) or not bool(torch.isfinite(beta).all()):
            raise KernelError("non_finite_values", "Parameters must be finite.")
        return _evaluate(z, x, d, y, kind, gamma, beta, cluster_codes, groups, work, plan)


@torch.no_grad()
def fit_joint(z: Tensor, x: Tensor, d: Tensor, y: Tensor, kind: str, *,
              cluster_codes: Tensor | None = None, max_iterations: int = 100,
              tolerance: float = 1e-9, max_work: int = DEFAULT_MAX_WORK) -> dict[str, Any]:
    """OLS first stage, residual-inclusion outcome, and full HC0/CR0 sandwich."""
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    if max_iterations > MAX_ITERATIONS or isinstance(tolerance, bool) \
            or not isinstance(tolerance, Real) or not math.isfinite(tolerance) \
            or not 0 < tolerance < 1:
        raise KernelError("invalid_options", "Use at most 1000 iterations and a finite tolerance in (0, 1).")
    with torch.device("cpu"):
        _, _, _, _, groups, work, plan = _admit(z, x, d, y, kind, cluster_codes,
                                               max_iterations, max_work)
        _rank(z, "First-stage design")
        gamma = least_squares(z, d, drop_collinear=False).beta
        q = torch.cat((x, _residual(z, d, gamma)[:, None]), dim=1)
        _rank(q, "Residual-inclusion design")
        certificate = _separation(q, y, kind)
        objective = _objective(q, y, kind)
        if kind == "gaussian":
            beta = least_squares(q, y, drop_collinear=False).beta
            optimizer = {"method": "ols_qr", "iterations": 0, "converged": True,
                         "gradient_max": float((q.T @ (y - q @ beta)).abs().max()),
                         "function_evaluations": 1, "derivative_evaluations": 1,
                         "backtracks": 0, "nonconcave_iterations": 0,
                         "message": "Closed-form OLS residual-inclusion fit."}
        else:
            intercept = bool((q[:, 0] == 1).all())
            result = maximize_newton(objective, objective.start(intercept=intercept),
                                     max_iter=max_iterations, step_tol=float(tolerance),
                                     scaled_gradient_tol=float(tolerance),
                                     value_fn=objective.value,
                                     raise_on_failure=False)
            if not result.converged:
                raise KernelError("nonconvergence", "The outcome optimizer did not converge within its explicit iteration budget.")
            beta = result.theta
            optimizer = {"method": result.method, "iterations": result.iterations,
                         "converged": True, **result.diagnostics}
        answer = _evaluate(z, x, d, y, kind, gamma, beta, cluster_codes, groups, work, plan,
                           certificate)
        answer["optimizer"] = {**optimizer, **certificate, "max_iterations": max_iterations,
                               "tolerance": float(tolerance), "working_dispersion": 1.0,
                               "criterion": "canonical_working_score_sum",
                               "optimization_criterion": "negative_half_glm_deviance"}
        return answer


@torch.no_grad()
def mean_link(kind: str, eta: Tensor) -> tuple[Tensor, Tensor]:
    """Conditional mean and its derivative, without clipping or device fallback."""
    _tensor(eta, "eta")
    if eta.ndim != 1 or not 1 <= eta.numel() <= MAX_ROWS or not bool(torch.isfinite(eta).all()):
        raise KernelError("invalid_design", "eta must be a finite vector of at most 5000 rows.")
    with torch.device("cpu"):
        family, link = _family_link(kind)
        family, link = family(), link()
        mu = link.inverse(eta)
        derivative = link.derivative(eta, mu)
        if not bool(family.valid_mu(mu).all()) or not bool(torch.isfinite(derivative).all()) \
                or not bool((derivative > 0).all()):
            raise KernelError("numerical_domain", "Conditional means or derivatives leave the finite link domain.")
        if family.binomial and not bool((link.complement(eta, mu) > 0).all()):
            raise KernelError("numerical_domain", "The conditional probability reaches a numerical boundary.")
        return mu, derivative
