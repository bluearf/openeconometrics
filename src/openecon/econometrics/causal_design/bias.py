"""Fixed externally supplied bias boxes and Gaussian contrast interval unions.

For beta_hat ~ N(theta + bias, covariance), each declared bias lies in a fixed
box. A linear support function gives exact identification endpoints; the union
of ordinary Gaussian intervals covers every admissible fixed bias. This does
not estimate a restriction from noisy pretrends or implement HonestDiD.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.resources import plan_workspace

from . import common as m


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{label} exceeds finite float64 range.")
    return value


def _product(left, right, label):
    value = _finite(left * right, label)
    if bool(((left != 0) & (right != 0) & (value == 0)).any()):
        raise AnalysisError("numerical_failure", f"A nonzero {label} contribution underflows float64.")
    return value


def _sum(values):
    """Native Neumaier sum preserves small terms across severe cancellation."""
    total = torch.zeros((), dtype=torch.float64, device="cpu")
    correction = total.clone()
    for value in values:
        updated = total + value
        correction = correction + torch.where(
            total.abs() >= value.abs(), (total - updated) + value, (value - updated) + total
        )
        total = updated
    return _finite(total + correction, "compensated sum")


def _ratio(numerator, denominator):
    value = numerator / denominator
    if not math.isfinite(value) or (numerator != 0 and value == 0):
        raise AnalysisError("numerical_failure", "A nonzero tipping scale is not representable in float64.")
    return value


def _shift(base, change):
    value = base + change
    if not math.isfinite(value) or (change != 0 and value == base):
        raise AnalysisError("numerical_failure", "A nonzero bias or uncertainty endpoint shift is not representable in float64.")
    return value


def _endpoint(center, bias, radius, identified):
    value = float(_sum(torch.tensor([center, bias, radius], dtype=torch.float64, device="cpu")))
    if radius != 0 and value == identified:
        raise AnalysisError("numerical_failure", "A nonzero uncertainty endpoint shift is not representable in float64.")
    return value


@m.procedure
def bias_sensitivity(
    data: Any,
    estimate: str,
    contrast: str,
    bias_lower: str,
    bias_upper: str,
    *,
    covariance: Sequence[Sequence[float]],
    restriction: str | None = None,
    scales: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    level: float = 0.95,
    missing: str = "raise",
    device: str = "cpu",
    weights=None,
    max_work: int = 100_000_000,
):
    """Union Gaussian intervals over a fixed external coordinatewise bias box.

    Rows give ordered estimates, contrast coefficients and fixed bias limits;
    ``covariance`` is their complete joint covariance in that exact row order.
    Declare ``restriction='fixed_external'``: limits and nonnegative increasing
    ``scales`` must be specified independently of the estimation noise. The
    target is contrast'*(true reduced-form mean - bias), with any bias in the
    scaled box. Off-diagonal covariance is retained. Returned intervals are
    conservative pointwise Gaussian unions, not simultaneous, data-adaptive
    pretrend inference or HonestDiD. Missing rows, weights and Dataset inputs
    are refused. Zero variance is allowed for a genuinely deterministic
    contrast; nonrepresentable nonzero uncertainty is refused.
    """
    m.options(device, weights, max_work, level)
    c.check_choice(restriction, "restriction", ("fixed_external",))
    c.check_choice(missing, "missing", ("raise",))
    names = [c.check_name(name, role) for name, role in (
        (estimate, "estimate"), (contrast, "contrast"),
        (bias_lower, "bias_lower"), (bias_upper, "bias_upper"),
    )]
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "Estimate, contrast and bias bounds need distinct columns.")
    if not isinstance(scales, (list, tuple)) or not 1 <= len(scales) <= 256:
        raise AnalysisError("invalid_option", "scales must contain 1..256 increasing nonnegative values.")
    grid = [c.check_number(value, "scale", minimum=0) for value in scales]
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise AnalysisError("invalid_option", "scales must increase strictly without duplicates.")
    selected, metadata = m.sample(data, names, numeric=names, missing=missing,
                                  max_work=max_work, minimum=1)
    p = len(selected)
    if p > 64:
        raise AnalysisError("resource_limit", "Fixed bias sensitivity accepts at most 64 ordered terms.")
    planned = metadata["n_input"] * 8 + p ** 3 * 16 + len(grid) * p * 8
    m.work(planned, max_work, "complete bias support functions and joint covariance validation")
    plan = plan_workspace("fixed external bias sensitivity", {
        "joint_covariance_eigendecomposition": p * p * 128,
        "complete_tables_and_state": (len(grid) * 16 + p * p + p * 8) * 128,
    })
    if not isinstance(covariance, (list, tuple)) or len(covariance) != p:
        raise AnalysisError("invalid_covariance", "covariance needs one ordered row per estimate.")
    rows = []
    for row in covariance:
        if not isinstance(row, (list, tuple)) or len(row) != p:
            raise AnalysisError("invalid_covariance", "covariance must be a complete square matrix.")
        rows.append([c.check_number(value, "covariance cell") for value in row])
    cov = torch.tensor(rows, dtype=torch.float64, device="cpu")
    if not torch.equal(cov, cov.T):
        raise AnalysisError("invalid_covariance", "Joint covariance must be exactly symmetric; it is not repaired.")
    magnitude = float(cov.abs().max())
    if magnitude:
        normalized = cov / magnitude
        if bool(((cov != 0) & (normalized == 0)).any()):
            raise AnalysisError("numerical_failure", "Nonzero covariance cells underflow during scale normalization.")
        eigen = torch.linalg.eigvalsh(normalized)
        if float(eigen.min()) < 0:
            raise AnalysisError("invalid_covariance", "Joint covariance must be positive semidefinite; no projection is used.")
    else:
        normalized = cov.clone()
        eigen = torch.zeros(p, dtype=torch.float64, device="cpu")
    values, coefficients, lower, upper = [m.tensor(selected, name) for name in names]
    if not bool((lower <= upper).all()):
        raise AnalysisError("invalid_option", "Every fixed lower bias limit must be <= its upper limit.")
    if not bool((coefficients != 0).any()):
        raise AnalysisError("invalid_contrast", "At least one contrast coefficient must be nonzero.")
    coefficient_scale = float(coefficients.abs().max())
    unit = coefficients / coefficient_scale
    if bool(((coefficients != 0) & (unit == 0)).any()):
        raise AnalysisError("numerical_failure", "Nonzero contrast coefficients underflow during scale normalization.")
    projected = torch.stack([_sum(_product(row, unit, "covariance projection")) for row in normalized])
    quadratic = float(_sum(_product(unit, projected, "contrast variance")))
    if quadratic < 0:
        raise AnalysisError("numerical_failure", "Contrast variance became negative; no clipping is performed.")
    if quadratic == 0 and bool((projected != 0).any()):
        raise AnalysisError("numerical_failure", "A nonzero covariance projection has zero rounded contrast variance.")
    se = math.sqrt(quadratic) * math.sqrt(magnitude) * coefficient_scale
    variance = se * se
    if not math.isfinite(variance) or (quadratic > 0 and variance == 0):
        raise AnalysisError("numerical_failure", "Nonzero contrast variance is not representable in float64.")
    center = float(_sum(_product(coefficients, values, "contrast mean")))
    low_terms = _product(coefficients, torch.where(coefficients >= 0, lower, upper), "bias support")
    high_terms = _product(coefficients, torch.where(coefficients >= 0, upper, lower), "bias support")
    low_support = float(_sum(low_terms))
    high_support = float(_sum(high_terms))
    critical = normal_isf((1 - level) / 2)
    if not math.isfinite(critical) or critical <= 0:
        raise AnalysisError("numerical_failure", "The requested Gaussian critical value is not representable as positive float64.")
    radius = critical * se
    if not math.isfinite(radius):
        raise AnalysisError("numerical_failure", "Gaussian interval radius overflows float64.")
    sensitivity = []
    for scale in grid:
        low = scale * low_support
        high = scale * high_support
        if (scale and low_support and low == 0) or (scale and high_support and high == 0):
            raise AnalysisError("numerical_failure", "Scaled nonzero bias underflows float64.")
        identified_lower, identified_upper = _shift(center, -high), _shift(center, -low)
        row = [scale, low, high, identified_lower, identified_upper,
               _endpoint(center, -high, -radius, identified_lower),
               _endpoint(center, -low, radius, identified_upper)]
        if not all(math.isfinite(value) for value in row):
            raise AnalysisError("numerical_failure", "A bias support endpoint overflows float64.")
        sensitivity.append(row)
    # No infinite sentinel: a None threshold means this direction cannot reach
    # zero at any finite nonnegative scale under the supplied box.
    id_tip = ci_tip = None
    if center == 0:
        id_tip = 0.0
    elif center > 0 and high_support > 0:
        id_tip = _ratio(center, high_support)
    elif center < 0 and low_support < 0:
        id_tip = _ratio(center, low_support)
    if abs(center) <= radius:
        ci_tip = 0.0
    elif center > 0 and high_support > 0:
        ci_tip = _ratio(center - radius, high_support)
    elif center < 0 and low_support < 0:
        ci_tip = _ratio(center + radius, low_support)
    if any(value is not None and not math.isfinite(value) for value in (id_tip, ci_tip)):
        raise AnalysisError("numerical_failure", "A nonzero tipping scale exceeds finite float64 range.")
    z = center / se if se else None
    if z is not None and not math.isfinite(z):
        raise AnalysisError("numerical_failure", "The zero-bias z statistic overflows float64.")
    pvalue = 2 * normal_sf(abs(z)) if z is not None else None
    settings = dict(estimate=estimate, contrast=contrast, bias_lower=bias_lower,
                    bias_upper=bias_upper, covariance=rows, restriction=restriction,
                    scales=grid, level=level, missing=missing, device=device,
                    weights=None, max_work=max_work)
    state = dict(target="fixed-bias linear contrast with pointwise Gaussian union intervals",
                 covariance=rows, covariance_order=metadata["positions"],
                 normalized_covariance_eigenvalues=eigen.tolist(),
                 contrast=coefficients.tolist(), estimates=values.tolist(),
                 bias_lower=lower.tolist(), bias_upper=upper.tolist(),
                 lower_support_contributions=low_terms.tolist(),
                 upper_support_contributions=high_terms.tolist(),
                 contrast_estimate=center, contrast_variance=variance,
                 standard_error=se, df=None, critical=critical, sensitivity=sensitivity,
                 identification_tipping_scale=id_tip, interval_tipping_scale=ci_tip,
                 planned_work=planned, inference="fixed externally supplied bounds; conservative pointwise Gaussian interval union",
                 assumptions=["Complete supplied joint covariance and Gaussian/asymptotic Gaussian reduced-form estimates",
                              "Bias limits, contrast and scales fixed independently of estimation noise",
                              "True bias belongs to the supplied scaled coordinatewise box"],
                 unsupported=["estimated/noisy pretrend restriction", "HonestDiD", "simultaneous inference", "estimated bias-limit uncertainty"])
    metadata["procedure_resource_plan"] = plan.record()
    return m.result("bias_sensitivity", {
        "sensitivity": m.frame(sensitivity, columns=["scale", "bias_lower", "bias_upper", "identified_lower", "identified_upper", "ci_lower", "ci_upper"]),
        "contrast": m.frame([[center, variance, se, z, pvalue, _shift(center, -radius), _shift(center, radius), id_tip, ci_tip]],
                            columns=["estimate", "variance", "se", "z", "p_zero_bias", "ci_lower_zero_bias", "ci_upper_zero_bias", "identification_tipping_scale", "interval_tipping_scale"]),
        "terms": m.frame(list(zip(metadata["positions"], values.tolist(), coefficients.tolist(), lower.tolist(), upper.tolist(), low_terms.tolist(), high_terms.tolist(), strict=True)), columns=["position", "estimate", "contrast", "bias_lower", "bias_upper", "support_lower", "support_upper"]),
        "covariance": m.frame(rows, columns=[f"position_{i}" for i in metadata["positions"]]),
    }, metadata, settings, state, notes=["Identification endpoints are plug-in support bounds; Gaussian interval unions add pointwise sampling uncertainty."])
