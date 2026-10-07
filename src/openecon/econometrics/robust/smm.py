"""Bisquare S and S-initialized MM regression, entirely float64 PyTorch.

Normalized rho_c(u)=1-(1-(u/c)^2)^3 inside c, 1 outside.
S minimizes s(beta), with mean rho_c(r/s)=b. MM fixes this S scale,
uses c1>=c0 (rho1<=rho0), and descends the final rho objective from S.
Finite elemental sampling approximates, and never certifies, the S minimum.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, kernel_call, make_spec
from openecon.engines import linalg
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import tensor_bytes


def rho(u: Tensor, c: float) -> Tensor:
    z = (u / c).square().clamp(max=1)
    # Expanded polynomial avoids cancellation when u is close to zero.
    return z * (3 - z * (3 - z))


def weight(u: Tensor, c: float) -> Tensor:
    return (1 - (u / c).square()).clamp_min(0).square()


def psi(u: Tensor, c: float) -> Tensor:
    return u * weight(u, c)


def psi_prime(u: Tensor, c: float) -> Tensor:
    z = (u / c).square()
    return torch.where(z < 1, (1 - z) * (1 - 5 * z), torch.zeros_like(u))


def _normal_expect(function, *, limit: float = 10, points: int = 4001) -> float:
    grid = torch.linspace(-limit, limit, points, dtype=torch.float64)
    density = torch.exp(-grid.square() / 2) / math.sqrt(2 * math.pi)
    return float(torch.trapezoid(function(grid) * density, grid))


def calibrated_tune(b: float) -> float:
    lo, hi = 0.01, 20.0
    for _ in range(55):
        mid = (lo + hi) / 2
        if _normal_expect(lambda z: rho(z, mid)) > b:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gaussian_efficiency(c: float) -> float:
    """Asymptotic efficiency (E psi')^2/E psi^2 under N(0,1)."""
    # Align the quadrature endpoints with the compact support of psi. A
    # whole-line grid straddling the derivative's kink loses precision for S c.
    return _normal_expect(lambda z: psi_prime(z, c), limit=c, points=8001) ** 2 / _normal_expect(
        lambda z: psi(z, c).square(), limit=c, points=8001
    )


def m_scale(residual: Tensor, c: float, b: float) -> float:
    """Unique positive M-scale root by safeguarded log bisection."""
    values = residual.abs()
    floor = max(float(values.max()) * 1e-14, 1e-300)
    # Too many exact fits force an imploded scale; never replace with MAD or OLS.
    if float((values > floor).to(torch.float64).mean()) <= b:
        raise KernelError("zero_scale", "The S scale implodes because too many residuals are zero.")
    lo = floor
    hi = max(float(torch.quantile(values, 0.5)), floor * 2)
    while float(rho(residual / hi, c).mean()) > b:
        hi *= 2
        if not math.isfinite(hi):
            raise KernelError("numerical_failure", "The S scale could not be bracketed.")
    for _ in range(60):
        mid = math.exp((math.log(lo) + math.log(hi)) / 2)
        if float(rho(residual / mid, c).mean()) > b:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1 < 1e-11:
            break
    return (lo + hi) / 2


@dataclass
class SFit:
    beta: Tensor
    scale: float
    iterations: int
    valid_starts: int
    converged_starts: int
    scale_equation_error: float
    score_error: float


def _score_error(x: Tensor, residual: Tensor, scale: float, c: float) -> float:
    score = x.T @ psi(residual / scale, c)
    divisor = x.abs().sum(dim=0).clamp_min(1)
    return float((score / divisor).abs().max())


def _refine_s(x, y, beta, c, b, max_iterations, tolerance):
    scale = m_scale(y - x @ beta, c, b)
    for iteration in range(1, max_iterations + 1):
        candidate = linalg.least_squares(
            x, y, weight((y - x @ beta) / scale, c), drop_collinear=False
        ).beta
        movement = candidate - beta
        new_scale = m_scale(y - x @ candidate, c, b)
        step = 1.0
        while new_scale > scale * (1 + 1e-12) and step > 2**-30:
            step *= 0.5
            candidate = beta + step * movement
            new_scale = m_scale(y - x @ candidate, c, b)
        if new_scale > scale * (1 + 1e-10):
            raise KernelError("nonconvergence", "S refinement could not decrease the robust scale.")
        change = float((candidate - beta).abs().max()) / max(1.0, float(beta.abs().max()))
        beta, scale = candidate, new_scale
        error = _score_error(x, y - x @ beta, scale, c)
        if change <= tolerance and error <= math.sqrt(tolerance):
            return beta, scale, iteration, error
    raise KernelError("nonconvergence", "S refinement reached max_iterations without convergence.")


@torch.no_grad()
def s_estimate(
    x: Tensor,
    y: Tensor,
    *,
    c: float,
    b: float,
    starts: int,
    seed: int,
    max_iterations: int,
    tolerance: float,
) -> SFit:
    n, k = x.shape
    if n <= 2 * k:
        raise KernelError(
            "insufficient_observations", "S regression needs N>2K for a high-breakdown start."
        )
    ordinary = linalg.least_squares(x, y, drop_collinear=False)
    if float(ordinary.ssr) <= 1e-28 * max(float(y.square().sum()), 1e-300):
        raise KernelError(
            "perfect_fit", "The outcome is fitted exactly; the positive S scale is undefined."
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates: list[tuple[float, Tensor]] = []
    valid = 0
    # Include OLS as a clean-data candidate; elemental starts supply robustness.
    for index in range(starts + 1):
        rows = torch.randperm(n, generator=generator)[:k] if index else None
        try:
            beta = (
                ordinary.beta
                if rows is None
                else linalg.least_squares(x[rows], y[rows], drop_collinear=False).beta
            )
            scale = m_scale(y - x @ beta, c, b)
        except KernelError as exc:
            if exc.code in {"singular_design", "zero_scale"}:
                continue
            raise
        valid += int(index > 0)
        candidates.append((scale, beta))
        candidates.sort(key=lambda item: item[0])
        del candidates[10:]
    if not valid:
        raise KernelError(
            "no_valid_initialization", "No nonsingular elemental S start has positive scale."
        )
    solutions = []
    for _, beta in candidates:
        try:
            solutions.append(_refine_s(x, y, beta, c, b, max_iterations, tolerance))
        except KernelError as exc:
            if exc.code not in {"singular_design", "nonconvergence", "zero_scale"}:
                raise
    if not solutions:
        raise KernelError(
            "nonconvergence",
            "None of the retained S starts converged; increase starts/max_iterations.",
        )
    beta, scale, iterations, error = min(solutions, key=lambda item: item[1])
    return SFit(
        beta,
        scale,
        iterations,
        valid,
        len(solutions),
        abs(float(rho((y - x @ beta) / scale, c).mean()) - b),
        error,
    )


@torch.no_grad()
def mm_estimate(x, y, initial: SFit, c: float, max_iterations: int, tolerance: float):
    beta, scale = initial.beta.clone(), initial.scale
    initial_objective = float(rho((y - x @ beta) / scale, c).sum())
    objective = initial_objective
    for iteration in range(1, max_iterations + 1):
        candidate = linalg.least_squares(
            x, y, weight((y - x @ beta) / scale, c), drop_collinear=False
        ).beta
        movement, step = candidate - beta, 1.0
        candidate_objective = float(rho((y - x @ candidate) / scale, c).sum())
        while candidate_objective > objective + 1e-12 and step > 2**-30:
            step *= 0.5
            candidate = beta + step * movement
            candidate_objective = float(rho((y - x @ candidate) / scale, c).sum())
        if candidate_objective > objective + 1e-10:
            raise KernelError(
                "nonconvergence", "MM refinement could not decrease its fixed-scale objective."
            )
        change = float((candidate - beta).abs().max()) / max(1.0, float(beta.abs().max()))
        beta, objective = candidate, candidate_objective
        if change <= tolerance and _score_error(x, y - x @ beta, scale, c) <= math.sqrt(tolerance):
            return beta, iteration, initial_objective, objective
    raise KernelError("nonconvergence", "MM refinement reached max_iterations without convergence.")


def estimating_equations(x, y, sfit: SFit, beta, c0, c1, b, mm):
    """Joint scores and minus derivative, including uncertainty of the S scale."""
    n, k = x.shape
    u0 = (y - x @ sfit.beta) / sfit.scale
    p0, d0 = psi(u0, c0), psi_prime(u0, c0)
    scores0 = torch.cat([x * p0[:, None], (rho(u0, c0) - b)[:, None]], dim=1)
    a0 = torch.zeros((k + 1, k + 1), dtype=torch.float64)
    a0[:k, :k] = x.T @ (x * d0[:, None]) / sfit.scale
    a0[:k, k] = x.T @ (d0 * u0)
    a0[k, :k] = (6 / c0**2) * (p0 @ x) / sfit.scale
    a0[k, k] = (6 / c0**2) * torch.sum(p0 * u0)
    if not mm:
        return scores0, a0
    u1 = (y - x @ beta) / sfit.scale
    d1 = psi_prime(u1, c1)
    scores = torch.cat([x * psi(u1, c1)[:, None], scores0], dim=1)
    a = torch.zeros((2 * k + 1, 2 * k + 1), dtype=torch.float64)
    a[:k, :k] = x.T @ (x * d1[:, None]) / sfit.scale
    a[:k, -1] = x.T @ (d1 * u1)
    a[k:, k:] = a0
    return scores, a


def _fit(spec: ModelSpec, data: Any, *, mm: bool) -> ResultBundle:
    frame = ModelFrame(spec, data)
    width = frame.design_width()
    frame.workspace_plan(
        "S/MM elemental search and joint sandwich",
        {
            "design_and_residual_buffers": tensor_bytes((frame.n, width + 12), itemsize=32),
            "joint_scores_and_covariance": tensor_bytes((frame.n, 2 * width + 1), itemsize=16)
            + tensor_bytes((2 * width + 1, 2 * width + 1), itemsize=48),
            "retained_elemental_starts": tensor_bytes((10, width), itemsize=16),
        },
    )
    design = frame.drop_collinear(frame.design())
    x, y = design.x.clone(), frame.numeric(spec.outcome)
    n, k = x.shape
    if not k:
        raise AnalysisError("no_regressors", "S/MM regression needs at least one estimable term.")
    # Affine normalization improves elemental fits without changing the estimator.
    transform = torch.eye(k, dtype=torch.float64)
    if design.intercept and k > 1:
        centers = torch.quantile(x[:, 1:], 0.5, dim=0)
        spread = torch.quantile((x[:, 1:] - centers).abs(), 0.5, dim=0)
        spread = torch.where(spread > 0, spread, x[:, 1:].std(dim=0)).clamp_min(1e-300)
        x[:, 1:] = (x[:, 1:] - centers) / spread
        transform[0, 1:] = -centers / spread
        transform[1:, 1:] = torch.diag(1 / spread)
    b = float(frame.option("breakdown"))
    explicit_tune = frame.option("scale_tune")
    c0 = calibrated_tune(b) if explicit_tune is None else float(explicit_tune)
    c1 = float(frame.option("efficiency_tune")) if mm else c0
    if c1 < c0:
        raise AnalysisError(
            "invalid_option", "MM efficiency_tune must be at least scale_tune so rho1<=rho0."
        )
    starts, seed = int(frame.option("starts")), int(frame.option("seed"))
    limit, tolerance = int(frame.option("max_iterations")), float(frame.option("tolerance"))
    sfit = kernel_call(
        s_estimate,
        x,
        y,
        c=c0,
        b=b,
        starts=starts,
        seed=seed,
        max_iterations=limit,
        tolerance=tolerance,
    )
    beta, iterations, initial_objective, objective = sfit.beta, 0, None, None
    if mm:
        beta, iterations, initial_objective, objective = kernel_call(
            mm_estimate, x, y, sfit, c1, limit, tolerance
        )
    scores, a = estimating_equations(x, y, sfit, beta, c0, c1, b, mm)
    try:
        inverse = torch.linalg.inv(a)
    except RuntimeError as exc:
        raise AnalysisError(
            "singular_information", "The S/MM estimating-equation derivative is singular."
        ) from exc
    influence = scores @ inverse.T
    kind = spec.covariance
    infer = {"covariance": kind, "scale_uncertainty": "joint S coefficient/scale scores"}
    df = n - k
    if kind == "nonrobust":
        u = (y - x @ beta) / sfit.scale
        derivative = float(psi_prime(u, c1).mean())
        if derivative <= 0:
            raise AnalysisError("singular_information", "The mean psi derivative is not positive.")
        scalar = sfit.scale**2 * float(psi(u, c1).square().mean()) / derivative**2
        v = linalg.least_squares(x, y, drop_collinear=False).xtx_inv * scalar * n / (n - k)
        infer.update(
            {
                "correction": "homoskedastic symmetric-error M covariance; N/(N-K)",
                "scale_uncertainty": "orthogonal under symmetric errors",
            }
        )
    elif kind == "cluster":
        codes, count = frame.cluster_dimensions()[0]
        sums = torch.zeros((count, influence.shape[1]), dtype=torch.float64)
        sums.index_add_(0, codes, influence)
        factor = count / (count - 1) * (n - 1) / (n - k)
        v = (sums.T @ sums)[:k, :k] * factor
        df = count - 1
        infer.update(
            {
                "cluster_count": count,
                "cluster_df": df,
                "cluster_column": spec.cluster
                if isinstance(spec.cluster, str)
                else spec.cluster[0],
                "correction": "joint-score CR1: G/(G-1)*(N-1)/(N-K)",
                "small_sample_correction": factor,
            }
        )
    else:
        factor = n / (n - k)
        v = (influence.T @ influence)[:k, :k] * factor
        infer.update({"correction": "joint-score HC1: N/(N-K)", "small_sample_correction": factor})
    final = transform @ beta
    v = transform @ v @ transform.T
    theoretical = gaussian_efficiency(c1)
    frame.warn(
        "The seeded elemental S search approximates a nonconvex minimum; finite starts do not certify the theoretical breakdown point."
    )
    if explicit_tune is not None:
        frame.warn(
            "An explicit scale_tune overrides Gaussian consistency calibration; the S scale need not estimate Gaussian sigma consistently."
        )
    diagnostics = {
        "converged": True,
        "s_iterations": sfit.iterations,
        "mm_iterations": iterations,
        "valid_elemental_starts": sfit.valid_starts,
        "converged_retained_starts": sfit.converged_starts,
        "scale_equation_error": sfit.scale_equation_error,
        "s_score_error": sfit.score_error,
        "final_score_error": _score_error(x, y - x @ beta, sfit.scale, c1),
        "s_minimum": "approximate seeded multistart",
        "mm_initial_objective": initial_objective,
        "mm_final_objective": objective,
        "seed": seed,
    }
    return build_result(
        frame,
        terms=design.terms,
        params=final,
        covariance=v,
        use_t=True,
        df_inference=df,
        df_resid=n - k,
        fitted=design.x @ final,
        solver="seeded bisquare S; fixed-scale MM IRLS"
        if mm
        else "seeded bisquare S scale minimization",
        solver_diagnostics=diagnostics,
        inference=infer,
        categories=design.categories,
        metrics={
            "scale": sfit.scale,
            "scale_tune": c0,
            "efficiency_tune": c1,
            "nominal_asymptotic_breakdown": b,
            "gaussian_asymptotic_efficiency": theoretical,
            "s_iterations": sfit.iterations,
            "mm_iterations": iterations,
        },
        extra={
            "s_coefficients": (transform @ sfit.beta).tolist(),
            "case_weights": weight((y - x @ beta) / sfit.scale, c1).tolist(),
            "case_weight_sample_positions": frame.positions,
            "scale_equation_target": b,
            "loss": "normalized Tukey bisquare; rho=1 outside c",
            "scale_fixed_during_mm": mm,
            "joint_estimating_equation_order": (["MM coefficients"] if mm else [])
            + ["S coefficients", "log S scale"],
        },
    )


def fit_sreg(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit(spec, data, mm=False)


def fit_mmreg(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit(spec, data, mm=True)


def _convenience(
    name,
    *,
    data,
    y,
    x,
    covariance,
    cluster,
    categorical,
    intercept,
    missing,
    alpha,
    breakdown,
    scale_tune,
    starts,
    seed,
    max_iterations,
    tolerance,
    efficiency_tune=None,
):
    from openecon.analysis import fit

    spec = make_spec(
        name,
        outcome=y,
        predictors=column_list(x, "x"),
        covariance=covariance,
        cluster=cluster,
        categorical=column_list(categorical, "categorical"),
        intercept=intercept,
        missing=missing,
        alpha=alpha,
        options={
            "breakdown": breakdown,
            "scale_tune": scale_tune,
            "starts": starts,
            "seed": seed,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "efficiency_tune": efficiency_tune,
        },
    )
    return fit(spec, data=data)


def sreg(
    *,
    data: Any,
    y: str,
    x: Sequence[str] | None = None,
    covariance: str | None = None,
    cluster: str | None = None,
    categorical: Sequence[str] | None = None,
    intercept: bool = True,
    missing: str = "raise",
    alpha: float = 0.05,
    breakdown: float = 0.5,
    scale_tune: float | None = None,
    starts: int = 500,
    seed: int = 0,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> ResultBundle:
    """Minimize the bisquare M-scale with deterministic elemental initialization."""
    return _convenience("sreg", **locals())


def mmreg(
    *,
    data: Any,
    y: str,
    x: Sequence[str] | None = None,
    covariance: str | None = None,
    cluster: str | None = None,
    categorical: Sequence[str] | None = None,
    intercept: bool = True,
    missing: str = "raise",
    alpha: float = 0.05,
    breakdown: float = 0.5,
    scale_tune: float | None = None,
    efficiency_tune: float = 4.685061,
    starts: int = 500,
    seed: int = 0,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> ResultBundle:
    """S-initialized efficient bisquare MM regression at the fixed S scale."""
    return _convenience("mmreg", **locals())
