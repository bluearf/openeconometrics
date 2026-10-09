"""Bounded alpha and image-covariance extraction on correlation matrices.

Alpha uses the Kaiser--Caffrey iteration documented in the SAS/IML sample
library. Image-covariance extraction is PCA of the covariance of each
variable's linear prediction from the remaining variables (SAS METHOD=IMAGE).
It is not the differently defined generalized image extraction in SPSS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from . import extraction as ex


@dataclass
class ExtensionExtraction(ex.Extraction):
    diagnostics: dict[str, Any] = field(default_factory=dict)
    matrices: dict[str, Tensor] = field(default_factory=dict)


def _validated_correlation(r: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(r, Tensor) or r.ndim != 2 or r.shape[0] != r.shape[1] \
            or r.shape[0] < 3:
        raise KernelError("invalid_matrix", "Extraction requires a square correlation matrix with at least three variables.")
    if r.device.type != "cpu" or r.dtype != torch.float64:
        raise KernelError("unsupported_device", "Alpha and image-covariance extraction require CPU float64 tensors.")
    if r.shape[0] > 256:
        raise KernelError("workspace_limit", "Alpha and image-covariance extraction support at most 256 variables.")
    if not bool(torch.isfinite(r).all()):
        raise KernelError("non_finite_values", "Correlation values must be finite.")
    if not torch.allclose(r, r.T, rtol=0, atol=1e-12) or not torch.allclose(
            r.diagonal(), torch.ones(r.shape[0], dtype=r.dtype), rtol=0, atol=1e-12):
        raise KernelError("invalid_matrix", "Extraction requires a symmetric correlation matrix with a unit diagonal; it is not repaired.")
    r = (r + r.T) / 2
    factor, info = torch.linalg.cholesky_ex(r, check_errors=False)
    lost = factor.diagonal().square() <= 8 * r.shape[0] * torch.finfo(r.dtype).eps
    if int(info) != 0 or bool(lost.any()):
        raise KernelError("singular_matrix", "Alpha and image-covariance extraction require a positive-definite correlation matrix; remove redundant variables.")
    return r, factor


def _factor_count(m: int, p: int) -> None:
    if isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= p:
        raise KernelError("too_many_factors", "Supply an explicit integer factor count between one and the number of variables.")


def alpha_factors(r: Tensor, communalities: Tensor, m: int, *, max_iter: int,
                  tol: float) -> ExtensionExtraction:
    """Iterate H^{-1/2}(R-I)H^{-1/2}+I, without a likelihood or uniqueness bound.

    For retained eigenvectors V and positive eigenvalues gamma, the current
    pattern is sqrt(H) V sqrt(gamma). The next communalities are its row sums
    of squares. Convergence means max(abs(next-current)) <= tol; the returned
    pattern and eigenvalues belong to the last actual iteration, so its row
    sums agree exactly with the reported final communalities.
    """
    r, _ = _validated_correlation(r)
    p = r.shape[0]
    _factor_count(m, p)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise KernelError("invalid_option", "max_iter must be a positive integer.")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0:
        raise KernelError("invalid_option", "tol must be finite and positive.")
    if max_iter * p ** 3 > 2_000_000_000:
        raise KernelError("work_limit", "Alpha extraction iterations exceed the matrix work budget.")
    if not isinstance(communalities, Tensor) or communalities.shape != (p,) \
            or communalities.dtype != torch.float64 or communalities.device.type != "cpu":
        raise KernelError("invalid_spec", "Initial communalities must be a CPU float64 vector with one value per variable.")
    if not bool(torch.isfinite(communalities).all()) or not bool((communalities > 0).all()) \
            or not bool((communalities <= 1).all()):
        raise KernelError("unidentified_factor", "Alpha extraction requires positive finite initial SMC communalities no greater than one.")
    current = communalities.clone()
    identity = torch.eye(p, dtype=r.dtype)
    off_diagonal = r - identity
    for iteration in range(1, max_iter + 1):
        scale = current.rsqrt()
        weighted = off_diagonal * torch.outer(scale, scale) + identity
        if not bool(torch.isfinite(weighted).all()):
            raise KernelError("numerical_failure", "Alpha extraction exceeded float64 precision; rescale or remove weak variables.")
        values, vectors = ex._descending(weighted)
        root_floor = 64 * p * torch.finfo(r.dtype).eps * max(1.0, float(values.abs().max()))
        if not bool((values[:m] > root_floor).all()):
            raise KernelError("unidentified_factor", "The requested alpha factors have nonpositive or numerically unidentified roots; retain fewer factors.")
        loadings = current.sqrt()[:, None] * vectors[:, :m] * values[:m].sqrt()
        updated = loadings.square().sum(1)
        if not bool(torch.isfinite(updated).all()) or not bool((updated > 0).all()):
            raise KernelError("unidentified_factor", "Alpha extraction produced non-finite or zero communalities.")
        change = float((updated - current).abs().max())
        if change <= tol:
            uniqueness = 1 - updated
            return ExtensionExtraction(
                loadings, values, uniqueness, iteration, True,
                heywood=torch.nonzero(uniqueness <= 0).flatten().tolist(),
                diagnostics={"extraction_objective": "alpha_generalizability_iteration",
                             "converged": True, "communality_max_change": change,
                             "communality_tolerance": float(tol), "extraction_start": "SMC",
                             "last_iteration_communalities": current.tolist(),
                             "extraction_eigenvalues": "last H^-1/2 (R-I) H^-1/2 + I iteration",
                             "retained_roots_above_one": bool((values[:m] > 1).all()),
                             "loading_inference": "not provided", "model_fit_test": "not provided"})
        current = updated
    raise KernelError("no_convergence", f"Alpha extraction did not converge in {max_iter} iterations (last communality change {change:.2e}, tolerance {tol:g}). Raise max_iterations or retain fewer factors.")


def image_covariance_factors(r: Tensor, m: int) -> ExtensionExtraction:
    """PCA of C=B'RB, B=I-R^{-1}D and D=diag(1/diag(R^{-1})).

    Column j of B predicts standardized variable j from the remaining variables.
    Thus C=R-2D+D R^{-1}D is the image covariance. This routine decomposes C
    directly, without the alternative generalized SPSS image eigensystem.
    """
    r, factor = _validated_correlation(r)
    p = r.shape[0]
    _factor_count(m, p)
    inverse = torch.cholesky_inverse(factor)
    conditional_variance = inverse.diagonal().reciprocal()
    coefficients = torch.eye(p, dtype=r.dtype) - inverse * conditional_variance[None, :]
    coefficients.diagonal().zero_()
    covariance = coefficients.T @ r @ coefficients
    covariance = (covariance + covariance.T) / 2
    if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(covariance).all()):
        raise KernelError("numerical_failure", "Image coefficients or covariance exceeded float64 precision.")
    values, vectors = ex._descending(covariance)
    floor = 64 * p * torch.finfo(r.dtype).eps * max(1.0, float(values.abs().max()))
    if float(values.min()) < -floor:
        raise KernelError("numerical_failure", "Computed image covariance is not positive semidefinite; it is not repaired.")
    if not bool((values[:m] > floor).all()):
        raise KernelError("unidentified_factor", "The requested image-covariance factors have zero or numerically unidentified image variance; retain fewer factors.")
    values = values.clamp_min(0)
    loadings = vectors[:, :m] * values[:m].sqrt()
    uniqueness = 1 - loadings.square().sum(1)
    return ExtensionExtraction(
        loadings, values, uniqueness,
        heywood=torch.nonzero(uniqueness <= 0).flatten().tolist(),
        diagnostics={"extraction_objective": "PCA_of_linear_prediction_image_covariance",
                     "extraction_variant": "SAS/Guttman image covariance PCA",
                     "extraction_eigenvalues": "image covariance",
                     "image_coefficient_orientation": "rows predict columns; diagonal zero",
                     "converged": True, "loading_inference": "not provided",
                     "model_fit_test": "not provided"},
        matrices={"image_coefficients": coefficients, "image_covariance": covariance})
