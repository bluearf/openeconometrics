"""Explicit GLS and Kaiser generalized-image factor functionals.

The GLS criterion is the uniqueness-scaled concentrated criterion in IBM's
FACTOR Algorithms, pp. 3--6, not psych's fixed sample-inverse GLS variant.
Kaiser image extraction follows pp. 8--9 of that primary algorithm document:
https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/factor.pdf
These implementations do not establish execution equivalence to a vendor.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines import optimize
from openecon.engines.contracts import KernelError

from . import extraction as ex
from .extraction_extensions import ExtensionExtraction, _factor_count, _validated_correlation

MAX_GLS_VARIABLES = 32
MAX_GLS_WORK = 2_000_000_000
GEOMETRY_TOLERANCE = 1e-8
GRADIENT_TOLERANCE = 1e-7


def _separated(values: Tensor, m: int, *, above: float) -> None:
    scale = max(1.0, float(values.abs().max()))
    floor = GEOMETRY_TOLERANCE * scale
    if float(values[m - 1]) <= above + floor:
        raise KernelError("unidentified_factor", "Requested factors have a nonpositive or "
                          "numerically unidentified retained root; retain fewer factors.")
    if m < len(values) and float(values[m - 1] - values[m]) <= floor:
        raise KernelError("unidentified_factor", "Retained and discarded roots are not "
                          "separated; the requested factor subspace is unidentified.")


def gls_discrepancy(theta: Tensor, r: Tensor, m: int) -> tuple[Tensor, Tensor]:
    """F=.5 sum_tail(gamma-1)^2 and its exact log-free-uniqueness gradient.

    psi = .005 + exp(theta), gamma are roots of psi^-1/2 R psi^-1/2.
    The factor of gamma in the derivative is essential; this is not the ML
    discrepancy or an off-diagonal unweighted residual criterion.
    """
    free = torch.exp(theta)
    psi = ex.PSI_FLOOR + free
    scale = psi.rsqrt()
    values, vectors = torch.linalg.eigh(r * torch.outer(scale, scale))
    tail = values[:len(r) - m]
    if not bool(torch.isfinite(values).all()) or not bool((tail > 0).all()):
        return torch.tensor(float("inf"), dtype=r.dtype), torch.full_like(theta, float("nan"))
    tail_vectors = vectors[:, :len(r) - m]
    value = (tail - 1).square().sum() / 2
    gradient = -(free / psi) * (tail_vectors.square() @ (tail * (tail - 1)))
    return value, gradient


def generalized_least_squares(r: Tensor, smc: Tensor, m: int, *, max_iter: int,
                              tol: float = GRADIENT_TOLERANCE) -> ExtensionExtraction:
    """Three deterministic starts; an interior stationary local minimum only.

    The .005 uniqueness bound is declared, not an estimated natural boundary.
    Solutions at that bound, unresolved curvature or roots, failed iterations,
    and unsupported matrix work refuse the analysis. No sampling covariance or
    model chi-square is inferred from the optimization Hessian.
    """
    r, _ = _validated_correlation(r)
    p = len(r)
    _factor_count(m, p)
    if (p - m) ** 2 < p + m:
        raise KernelError("too_many_factors", "GLS extraction needs nonnegative model degrees of freedom.")
    if p > MAX_GLS_VARIABLES:
        raise KernelError("workspace_limit", f"GLS extraction is bounded to {MAX_GLS_VARIABLES} variables.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or not 1 <= max_iter <= 10000:
        raise KernelError("invalid_option", "GLS max_iter must be an integer from 1 through 10000.")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) \
            or not 1e-12 <= tol <= GRADIENT_TOLERANCE:
        raise KernelError("invalid_option", "GLS gradient tolerance must be from 1e-12 through 1e-7.")
    if 3 * max_iter * p ** 4 > MAX_GLS_WORK:
        raise KernelError("work_limit", "Requested GLS starts, iterations and numerical curvature exceed the work budget.")
    if not isinstance(smc, Tensor) or smc.shape != (p,) or smc.dtype != torch.float64 \
            or smc.device.type != "cpu" or not bool(torch.isfinite(smc).all()) \
            or not bool(((smc >= 0) & (smc <= 1)).all()):
        raise KernelError("invalid_spec", "GLS starting SMCs must be a CPU float64 vector in [0,1].")
    base = ((1 - .5 * m / p) * (1 - smc)).clamp_min(2 * ex.PSI_FLOOR)
    starts = [base, (base * .7).clamp_min(2 * ex.PSI_FLOOR),
              (base * 1.3).clamp_min(2 * ex.PSI_FLOOR)]

    def objective(theta):
        value, gradient = gls_discrepancy(theta, r, m)
        return -value, -gradient

    candidates, attempts = [], []
    for start in starts:
        solution = optimize.maximize_bfgs(objective, torch.log(start - ex.PSI_FLOOR),
                                          max_iter=max_iter, raise_on_failure=False)
        value, gradient = gls_discrepancy(solution.theta, r, m)
        size = float(gradient.abs().max())
        free = torch.exp(solution.theta)
        curvature = torch.linalg.eigvalsh(-(solution.hessian + solution.hessian.T) / 2)
        interior = bool((free > 1e-6).all())
        isolated = bool(torch.isfinite(curvature).all()) and float(curvature.min()) \
            > GEOMETRY_TOLERANCE * max(1.0, float(curvature.abs().max()))
        accepted = solution.converged and math.isfinite(float(value)) and size <= tol \
            and interior and isolated
        attempts.append({"converged": solution.converged, "iterations": solution.iterations,
                         "discrepancy": float(value), "gradient_max": size,
                         "interior": interior, "isolated_local_minimum": isolated})
        if accepted:
            candidates.append((float(value), solution, curvature))
    if not candidates:
        raise KernelError("no_convergence", "GLS extraction did not certify an interior stationary "
                          "local minimum within the declared iterations and uniqueness bound.")
    _, solution, curvature = min(candidates, key=lambda item: item[0])
    psi = ex.PSI_FLOOR + torch.exp(solution.theta)
    scale = psi.rsqrt()
    roots, vectors = ex._descending(r * torch.outer(scale, scale))
    _separated(roots, m, above=1.0)
    loadings = psi.sqrt()[:, None] * vectors[:, :m] * (roots[:m] - 1).sqrt()
    reproduced = loadings @ loadings.T + torch.diag(psi)
    value, gradient = gls_discrepancy(solution.theta, r, m)
    return ExtensionExtraction(
        loadings, loadings.square().sum(0), psi, solution.iterations, True, float(value),
        diagnostics={"extraction_objective": "half_squared_uniqueness_scaled_tail_root_discrepancy",
                     "extraction_variant": "IBM concentrated uniqueness-scaled GLS",
                     "converged": True, "gradient_max": float(gradient.abs().max()),
                     "gradient_tolerance": float(tol), "local_minimum_certified": True,
                     "minimum_objective_curvature": float(curvature.min()),
                     "extraction_optimum": "best certified local minimum among three deterministic starts",
                     "uniqueness_floor": ex.PSI_FLOOR, "uniqueness_convention": "optimized GLS variance; not forced to one minus communality",
                     "generalized_roots": roots.tolist(), "optimizer_attempts": attempts,
                     "loading_inference": "not provided", "model_fit_test": "not provided"},
        matrices={"uniqueness_scaled_correlation": r * torch.outer(scale, scale),
                  "reproduced_correlation": reproduced,
                  "optimized_uniqueness": torch.diag(psi)})


def generalized_image_factors(r: Tensor, m: int) -> ExtensionExtraction:
    """Kaiser image: S V_m (gamma-1)/sqrt(gamma), S²=diag(1/diag(R^-1)).

    This is a generalized eigensystem of R relative to S², rather than ordinary
    PCA of the image covariance. Only roots above one enter the pattern.
    """
    r, factor = _validated_correlation(r)
    p = len(r)
    _factor_count(m, p)
    if m >= p:
        raise KernelError("too_many_factors", "Generalized-image extraction needs fewer factors than variables.")
    inverse = torch.cholesky_inverse(factor)
    conditional = inverse.diagonal().reciprocal()
    s = conditional.sqrt()
    scaled = r / torch.outer(s, s)
    values, vectors = ex._descending(scaled)
    _separated(values, m, above=1.0)
    loadings = s[:, None] * vectors[:, :m] * ((values[:m] - 1) / values[:m].sqrt())
    anti = inverse * torch.outer(conditional, conditional)
    image = r - 2 * torch.diag(conditional) + anti
    uniqueness = 1 - loadings.square().sum(1)
    if not bool(torch.isfinite(loadings).all()) or not bool(torch.isfinite(image).all()):
        raise KernelError("numerical_failure", "Generalized-image extraction exceeded float64 precision.")
    return ExtensionExtraction(
        loadings, values, uniqueness,
        heywood=torch.nonzero(uniqueness <= 0).flatten().tolist(),
        diagnostics={"extraction_objective": "Kaiser_generalized_image_eigensystem",
                     "extraction_variant": "Kaiser generalized image (IBM published algorithm)",
                     "extraction_eigenvalues": "S^-1 R S^-1 generalized-image roots",
                     "retained_roots_above_one": True, "converged": True,
                     "loading_inference": "not provided", "model_fit_test": "not provided"},
        matrices={"generalized_image_matrix": scaled,
                  "conditional_variance": torch.diag(conditional),
                  "image_covariance": (image + image.T) / 2,
                  "anti_image_covariance": (anti + anti.T) / 2})
