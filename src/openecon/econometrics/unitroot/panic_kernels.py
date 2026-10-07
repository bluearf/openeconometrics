"""Float64 kernels for the fixed-factor Bai--Ng (2004) PANIC procedure.

PCA uses raw differences for the intercept case, time-demeaned differences
for the trend case. No cross-sectional centering or per-unit standardization
is performed. The estimated level series consist of t=2,...,T, not an
artificial extra zero observation in the ADF regression.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot.panel_kernels import adf_blocks, batch_ols
from openecon.econometrics.unitroot import tables

SOURCE = "https://doi.org/10.1111/j.1468-0262.2004.00528.x"
# Bai and Ng (2004), Table I, p. 1136. Columns: 1%, 5%, 10%.
# The trend/m=5/10% cell really prints -55.286; it is not silently corrected.
MQ_CRITICAL = {
    "constant": (
        (-20.151, -13.730, -11.022),
        (-31.621, -23.535, -19.923),
        (-41.064, -32.296, -28.399),
        (-48.501, -40.442, -36.592),
        (-58.383, -48.617, -44.111),
        (-66.978, -57.040, -52.312),
    ),
    "trend": (
        (-29.246, -21.313, -17.829),
        (-38.619, -31.356, -27.435),
        (-50.019, -40.180, -35.685),
        (-58.140, -48.421, -44.079),
        (-64.729, -55.818, -55.286),
        (-74.251, -64.393, -59.555),
    ),
}


@torch.no_grad()
def decompose(values: Tensor, factors: int, trend: str):
    """(differenced scores, loadings, cumulated idiosyncratic levels, spectrum).

    A single global scale avoids overflow without changing PCA's relative
    unit weights. Residual levels/loadings are returned in that scaled unit;
    the public optional component tables restore the original measurement.
    """
    raw_differences = values[:, 1:] - values[:, :-1]
    if not bool(torch.isfinite(raw_differences).all()):
        raise AnalysisError("numerical_failure", "Panel differences overflow float64; rescale the input.")
    scale = float(raw_differences.abs().amax())
    if not scale > 0:
        raise AnalysisError("constant_series", "The panel has no nonzero differences.")
    differences = (raw_differences / scale).T.contiguous()
    if trend == "trend":
        differences = differences - differences.mean(dim=0)
    if not bool(torch.isfinite(differences).all()):
        raise AnalysisError("numerical_failure", "Panel differences are not finite in float64.")
    u, spectrum, vh = torch.linalg.svd(differences, full_matrices=False)
    if not float(spectrum[0]) > 0 or (factors and float(spectrum[factors - 1]) <= 1e-10 * float(spectrum[0])):
        raise AnalysisError("factor_rank", "The requested factors exceed the numerical rank of the differenced panel.")
    if 0 < factors < len(spectrum) and float(spectrum[factors - 1] - spectrum[factors]) <= 1e-10 * float(spectrum[0]):
        raise AnalysisError("factor_boundary", "The requested PCA cut splits an unresolved repeated singular value; choose a separated factor count.")
    length = differences.shape[0]
    scores = u[:, :factors] * math.sqrt(length)
    loadings = vh[:factors].T * (spectrum[:factors] / math.sqrt(length))
    # Canonical signs are presentation only; projections/tests do not depend on them.
    pivots = scores.abs().argmax(dim=0)
    signs = scores[pivots, torch.arange(factors)].sign()
    scores, loadings = scores * signs, loadings * signs
    residuals = differences - scores @ loadings.T
    unit_norm = torch.linalg.vector_norm(differences, dim=0)
    residual_norm = torch.linalg.vector_norm(residuals, dim=0)
    if bool((residual_norm <= 1e-10 * unit_norm).any()) or bool((unit_norm == 0).any()):
        raise AnalysisError("degenerate_idiosyncratic", "At least one unit has no resolvable idiosyncratic differences after factor removal.")
    return scores, loadings, residuals.cumsum(dim=0).T, spectrum, scale


def idiosyncratic_probabilities(statistics: Tensor):
    """Stable log-CDF for the published no-constant MacKinnon response surface.

    Pooled Fisher must use log probabilities, not clipped or cancellation-prone
    ``log(ndtr(z))``. Outside the published lower boundary the response-surface
    probability is zero and its log is unavailable for finite pooled inference.
    """
    small = [v * s for v, s in zip(tables.TAU_SMALL_P["n"][0], tables.TAU_SMALL_SCALE, strict=True)]
    large = [v * s for v, s in zip(tables.TAU_LARGE_P["n"][0], tables.TAU_LARGE_SCALE, strict=True)]
    value = statistics
    left = small[0] + small[1] * value + small[2] * value.square()
    right = large[0] + large[1] * value + large[2] * value.square() + large[3] * value.pow(3)
    transformed = torch.where(value <= tables.TAU_STAR["n"][0], left, right)
    logp = torch.special.log_ndtr(transformed)
    logp = torch.where(value > tables.TAU_MAX["n"][0], torch.zeros_like(logp), logp)
    logp = torch.where(value < tables.TAU_MIN["n"][0], torch.full_like(logp, -math.inf), logp)
    return logp.exp(), logp


@torch.no_grad()
def adf_statistics(levels: Tensor, lags: int, trend: str):
    """Batched, column-scaled ADF t statistics with classical residual variance."""
    target, level, others = adf_blocks(levels, lags, trend)
    design = torch.cat([level[:, :, None], others], dim=2)
    scales = design.abs().amax(dim=1)
    target_scale = target.abs().amax(dim=1)
    if bool((scales == 0).any()) or bool((target_scale == 0).any()):
        raise AnalysisError("collinear_regressors", "An estimated component cannot identify every requested ADF term.")
    fit = batch_ols(design / scales[:, None, :], target / target_scale[:, None], "PANIC component ADF")
    if bool((fit.ssr <= 1e-24 * (target / target_scale[:, None]).square().sum(dim=1)).any()):
        raise AnalysisError("perfect_fit", "A component ADF fits exactly or at float64 rounding level; its t statistic is undefined.")
    statistic = fit.beta[:, 0] / fit.se[:, 0]
    if not bool(torch.isfinite(statistic).all()):
        raise AnalysisError("numerical_failure", "A component ADF statistic is not finite in float64.")
    return statistic, fit.nobs


@torch.no_grad()
def _least_squares(x: Tensor, target: Tensor, what: str):
    """Multiresponse QR; all columns must be identified, no pseudo-inverse."""
    rows, width = x.shape
    if rows <= width:
        raise AnalysisError("insufficient_observations", f"{what} needs more than {width} usable observations.")
    scales = x.abs().amax(dim=0)
    if bool((scales == 0).any()):
        raise AnalysisError("collinear_regressors", f"{what} contains a zero regressor.")
    normalized = x / scales
    q, triangle = torch.linalg.qr(normalized, mode="reduced")
    if bool((triangle.diagonal().abs() <= 1e-10 * torch.linalg.vector_norm(normalized, dim=0)).any()):
        raise AnalysisError("collinear_regressors", f"{what} is rank deficient.")
    rhs = q.T @ target
    coefficients = torch.linalg.solve_triangular(triangle, rhs, upper=True) / scales[:, None]
    return coefficients, target - q @ rhs


@torch.no_grad()
def deterministic_residuals(levels: Tensor, trend: str) -> Tensor:
    """Project t=2..T levels; extrapolate the same deterministic fit to F_1=0.

    This supplies the first lag pair in the paper's sums without estimating a
    deterministic regression on a fictitious first factor observation.
    """
    length = levels.shape[0]
    columns = [torch.ones(length + 1, dtype=torch.float64)]
    if trend == "trend":
        columns.append(torch.linspace(-1, 1, length + 1, dtype=torch.float64))
    full_design = torch.stack(columns, dim=1)
    coefficients, _ = _least_squares(full_design[1:], levels, "Common-factor deterministic projection")
    anchored = torch.cat([torch.zeros((1, levels.shape[1]), dtype=torch.float64), levels])
    return anchored - full_design @ coefficients


@torch.no_grad()
def mq_statistic(projected: Tensor, total: int, method: str, bandwidth: int, var_lags: int):
    """Symmetric generalized-eigenvalue MQ_c or finite-VAR filtered MQ_f."""
    if method == "mqc":
        lagged, current = projected[:-1], projected[1:]
        _, innovations = _least_squares(lagged, current, "Common-factor VAR(1)")
        if bandwidth >= innovations.shape[0]:
            raise AnalysisError("invalid_lags", "Bartlett bandwidth must be smaller than the common VAR residual sample.")
        correction = torch.zeros((projected.shape[1], projected.shape[1]), dtype=torch.float64)
        for lag in range(1, bandwidth + 1):
            one_sided = innovations[lag:].T @ innovations[:-lag]
            correction += (1 - lag / (bandwidth + 1)) * one_sided
        # T * Sigma_1: Sigma_1 is defined with denominator original T, not
        # the number of residual pairs at a particular lag.
        numerator = .5 * (current.T @ lagged + lagged.T @ current - correction - correction.T)
    else:
        differences = projected[1:] - projected[:-1]
        if var_lags:
            design = torch.cat([differences[var_lags - lag:-lag] for lag in range(1, var_lags + 1)], dim=1)
            coefficients, _ = _least_squares(design, differences[var_lags:], "Common-factor difference VAR")
            level_lags = torch.cat([projected[var_lags - lag:projected.shape[0] - lag] for lag in range(1, var_lags + 1)], dim=1)
            filtered = projected[var_lags:] - level_lags @ coefficients
        else:
            filtered = projected
        lagged, current = filtered[:-1], filtered[1:]
        numerator = .5 * (current.T @ lagged + lagged.T @ current)
    gram = lagged.T @ lagged
    if lagged.shape[0] <= projected.shape[1]:
        raise AnalysisError("insufficient_observations", "The common-factor eigenvalue test has too few valid lag pairs.")
    norms = torch.linalg.vector_norm(lagged, dim=0)
    if bool((norms == 0).any()):
        raise AnalysisError("common_rank", "The lagged common-factor space is rank deficient.")
    normalized = lagged / norms
    pivots = torch.linalg.qr(normalized, mode="r").R.diagonal().abs()
    if bool((pivots <= 1e-10).any()):
        raise AnalysisError("common_rank", "The lagged common-factor space is numerically rank deficient.")
    triangle = torch.linalg.cholesky(gram)
    left = torch.linalg.solve_triangular(triangle, numerator, upper=False)
    whitened = torch.linalg.solve_triangular(triangle, left.T, upper=False).T
    eigenvalue = float(torch.linalg.eigvalsh(.5 * (whitened + whitened.T))[0])
    statistic = total * (eigenvalue - 1)
    if not math.isfinite(statistic):
        raise AnalysisError("numerical_failure", "The common MQ statistic is not finite in float64.")
    return statistic, lagged.shape[0]


@torch.no_grad()
def common_sequence(levels: Tensor, total: int, trend: str, method: str,
                    bandwidth: int, var_lags: int, inference: str, level: float):
    """Largest-m PCA spaces, tested r,r-1,... with the published stopping rule."""
    projected = deterministic_residuals(levels, trend)
    # Same eigenspaces as T^-2 sum F F'; no unstable square-matrix inverse.
    _, _, vh = torch.linalg.svd(projected[1:], full_matrices=False)
    rows = []
    selected = None
    for m in range(levels.shape[1], 0, -1):
        space = projected @ vh[:m].T
        statistic, nobs = mq_statistic(space, total, method, bandwidth, var_lags)
        critical = MQ_CRITICAL[trend][m - 1] if inference == "asymptotic" else None
        reject = statistic < critical[(.01, .05, .1).index(level)] if critical else None
        rows.append({"test": method.upper(), "stochastic_trends_null": m,
                     "statistic": statistic, "p_value": None, "nobs": nobs,
                     "critical_1%": critical[0] if critical else None,
                     "critical_5%": critical[1] if critical else None,
                     "critical_10%": critical[2] if critical else None,
                     "reject": reject})
        if reject is False:
            selected = m
            break
        if reject is True:
            selected = 0
    return rows, selected
