"""Single-confounder, classical linear omitted-variable sensitivity.

The bias and adjusted standard error are the exact partial-R2 identities in
Cinelli and Hazlett (2020), equations 12 and 14. The point-estimate robustness
value solves equation 18 for equal confounder strengths. Primary reference:
https://doi.org/10.1111/rssb.12348; author manuscript:
https://carloscinelli.com/files/Cinelli%20and%20Hazlett%20-%20Making%20Sense%20of%20Sensitivity.pdf.
These identities do not establish causal identification or validate a chosen
confounder-strength grid. They require the classical homoskedastic OLS SE and
one omitted scalar regressor, with one additional degree of freedom consumed.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.engines.distributions import t_ppf, t_sf
from openecon.engines.linalg import least_squares
from openecon.resources import plan_workspace

from . import common as m


def _failure(message):
    raise AnalysisError("numerical_failure", message + " Rescale the input or strength grid.")


def _grid(value, name):
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 64:
        raise AnalysisError("invalid_option", f"{name} must be a list of 1 to 64 strengths.")
    values = [c.check_number(item, name, minimum=0, maximum=1) for item in value]
    if any(item >= 1 for item in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise AnalysisError(
            "invalid_option", f"{name} must be strictly increasing and lie in [0, 1)."
        )
    return values


def _scaled_feature(value):
    magnitude = float(value.abs().max())
    if magnitude == 0:
        raise AnalysisError("singular_design", "A selected regressor is constant.")
    normalized = value / magnitude
    center = float(normalized.mean())
    centered = normalized - center
    correction = float(centered.mean())
    center += correction
    centered -= correction
    spread = float(torch.sqrt(centered.square().mean()))
    if spread == 0:
        raise AnalysisError("singular_design", "A selected regressor is constant.")
    scale = magnitude * spread
    if not math.isfinite(scale) or scale == 0:
        _failure("Regressor variation is not representable in float64.")
    return centered / spread, center * magnitude, scale


def _infer(estimate, variance, df, critical):
    if not math.isfinite(variance) or variance <= 0:
        _failure("A nonzero coefficient variance is not representable in float64.")
    se = math.sqrt(variance)
    statistic = estimate / se
    radius = critical * se
    lower, upper = estimate - radius, estimate + radius
    if not all(math.isfinite(value) for value in (estimate, se, statistic, lower, upper)):
        _failure("Coefficient inference exceeds finite float64 range.")
    if radius <= 0 or lower == estimate or upper == estimate:
        _failure("The positive confidence radius cannot be resolved at the coefficient's scale.")
    return [estimate, se, statistic, df, 2 * t_sf(abs(statistic), df), lower, upper]


def _scenario(estimate, se, df, rd, ry, direction, orientation, critical):
    # Taking roots separately preserves representable biases even when rd*ry
    # underflows. Multiplying by the original SE first also avoids a needless
    # overflow of a dimensionless intermediate.
    bias = se * math.sqrt(df) * math.sqrt(ry) * math.sqrt(rd) / math.sqrt(1 - rd)
    adjusted_se = se * math.sqrt(1 - ry) / math.sqrt(1 - rd) * math.sqrt(df / (df - 1))
    if not math.isfinite(bias) or (rd > 0 and ry > 0 and bias == 0):
        _failure("The nonzero omitted-variable bias is not representable in float64.")
    adjusted = estimate + (1 if direction == "increase" else -1) * orientation * bias
    if bias > 0 and adjusted == estimate:
        _failure("The nonzero bias adjustment cannot be resolved at the coefficient's scale.")
    inference = _infer(adjusted, adjusted_se * adjusted_se, df - 1, critical)
    return [rd, ry, direction, bias, adjusted_se * adjusted_se, *inference]


@m.procedure
def ovb_sensitivity(
    data,
    y,
    treatment,
    controls,
    *,
    r2_treatment=(0.0, 0.1),
    r2_outcome=(0.0, 0.1),
    direction="both",
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Classical OLS sensitivity to one hypothetical omitted scalar regressor.

    ``r2_treatment`` is R2(D ~ Z | controls); ``r2_outcome`` is
    R2(Y ~ Z | D, controls). ``reduce`` subtracts signed bias from the original
    treatment coefficient, including crossing zero; ``increase`` adds it.
    An exactly zero coefficient uses the positive orientation. All Cartesian
    grid cells are returned. Controls must be an explicit list, possibly [].
    """
    m.options(device, weights, max_work, level)
    y, treatment = c.check_name(y, "y"), c.check_name(treatment, "treatment")
    if controls is None:
        raise AnalysisError("invalid_spec", "controls must be an explicit list, possibly [].")
    controls = c.name_list(controls, "controls", minimum=0)
    names = [y, treatment, *controls]
    if len(set(names)) != len(names) or "_cons" in names:
        raise AnalysisError(
            "invalid_spec", "Outcome and regressors must be distinct names other than '_cons'."
        )
    rd_grid, ry_grid = _grid(r2_treatment, "r2_treatment"), _grid(r2_outcome, "r2_outcome")
    c.check_choice(direction, "direction", ("both", "reduce", "increase"))
    directions = ["reduce", "increase"] if direction == "both" else [direction]
    cells, p = len(rd_grid) * len(ry_grid) * len(directions), len(controls) + 2
    # The declared quadratic QR cost is checked before selecting numerical
    # copies; the complete grid, including inference, is admitted separately.
    m.work(64 * cells + p**3, max_work, "OVB grid and model factorization")
    selected, metadata = m.sample(
        data, names, numeric=names, missing=missing, max_work=max_work,
        cost=4 * p * p, minimum=p + 2,
    )
    if any(pd.api.types.is_bool_dtype(selected[name].dtype) for name in names):
        raise AnalysisError("non_numeric_column", "All OLS roles require numeric values, not Boolean labels.")
    n, df = len(selected), len(selected) - p
    work_units = 4 * n * p * p + p**3 + 64 * cells
    m.work(work_units, max_work, "complete OVB model and sensitivity grid")
    plan = plan_workspace(
        "complete OVB sensitivity",
        {
            "sample_and_qr_arrays": 128 * n * p,
            "coefficient_covariance_transforms": 128 * p * p,
            "sensitivity_grid_and_saved_numeric_state": 256 * cells,
        },
    )
    values = [m.tensor(selected, name) for name in [treatment, *controls]]
    standardized, centers, scales = [], [], []
    for value in values:
        scaled, center, scale = _scaled_feature(value)
        standardized.append(scaled)
        centers.append(center)
        scales.append(scale)
    x = torch.stack([torch.ones(n, dtype=torch.float64, device="cpu"), *standardized], dim=1)
    outcome = m.tensor(selected, y)
    outcome_scale = float(outcome.abs().max())
    if outcome_scale == 0:
        raise AnalysisError("degenerate_outcome", "A positive OLS residual variance is required.")
    outcome_scaled = outcome / outcome_scale
    fit = least_squares(x, outcome_scaled, drop_collinear=False)
    ssr = float(fit.ssr)
    roundoff_floor = 256 * torch.finfo(torch.float64).eps**2 * float(outcome_scaled.square().sum())
    if ssr <= roundoff_floor:
        raise AnalysisError(
            "degenerate_outcome", "The residual variance is zero or indistinguishable from QR rounding noise."
        )
    transform = torch.zeros((p, p), dtype=torch.float64, device="cpu")
    transform[0, 0] = outcome_scale
    for j, (center, scale) in enumerate(zip(centers, scales), start=1):
        transform[j, j] = outcome_scale / scale
        transform[0, j] = -(center / scale) * outcome_scale
    coefficients = transform @ fit.beta
    covariance_scaled = fit.xtx_inv * (ssr / df)
    covariance = transform @ covariance_scaled @ transform.T
    covariance = covariance * 0.5 + covariance.T * 0.5
    if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(covariance).all()):
        _failure("The original-unit coefficient vector or covariance is not representable.")
    critical = -t_ppf((1 - level) / 2, df)
    adjusted_critical = -t_ppf((1 - level) / 2, df - 1)
    coefficient_rows = [
        _infer(float(coefficients[j]), float(covariance[j, j]), df, critical)
        for j in range(p)
    ]
    estimate, se, statistic = coefficient_rows[1][:3]
    orientation = 1 if estimate >= 0 else -1
    rows = [
        _scenario(estimate, se, df, rd, ry, direct, orientation, adjusted_critical)
        for rd in rd_grid for ry in ry_grid for direct in directions
    ]
    f = abs(statistic) / math.sqrt(df)
    robustness = 0.0 if f == 0 else f / (0.5 * math.hypot(f, 2) + 0.5 * f)
    if robustness >= 1 or (estimate != 0 and robustness == 0):
        _failure("The equal-strength point-estimate tipping value is not representable in [0, 1).")
    tipping = _scenario(
        estimate, se, df, robustness, robustness, "reduce", orientation, adjusted_critical
    )
    terms = ["_cons", treatment, *controls]
    raw_ssr = (ssr * outcome_scale) * outcome_scale
    if not math.isfinite(raw_ssr) or raw_ssr == 0:
        _failure("The positive original-unit residual sum of squares is not representable.")
    raw_variance = raw_ssr / df
    if not math.isfinite(raw_variance) or raw_variance == 0:
        _failure("The positive original-unit residual variance is not representable.")
    settings = dict(
        y=y, treatment=treatment, controls=controls, r2_treatment=rd_grid,
        r2_outcome=ry_grid, direction=direction, level=level, missing=missing,
        device="cpu", weights=None, max_work=max_work,
        confounder_dimension=1, covariance_type="classical_homoskedastic",
    )
    metadata["model_resource_plan"] = plan.record()
    metadata["work_units"] = work_units
    state = dict(
        terms=terms, outcome=outcome.tolist(), regressors=torch.stack(values, dim=1).tolist(),
        standardized_design=x.tolist(), regressor_centers=centers, regressor_scales=scales,
        outcome_scale=outcome_scale, coefficient_transform=transform.tolist(),
        scaled_coefficients=fit.beta.tolist(), coefficients=coefficients.tolist(),
        covariance=covariance.tolist(), scaled_covariance=covariance_scaled.tolist(),
        fitted=(fit.fitted * outcome_scale).tolist(), residuals=(fit.resid * outcome_scale).tolist(),
        scaled_residual_sum_of_squares=ssr, residual_sum_of_squares=raw_ssr,
        residual_variance=raw_variance, df=df, adjusted_df=df - 1, rank=fit.rank,
        standardized_condition_number=fit.condition_number,
        critical_value=critical, adjusted_critical_value=adjusted_critical,
        bias_orientation=orientation, sensitivity_rows=rows,
        equal_strength_point_estimate_robustness_value=robustness,
        tipping_adjusted_estimate=tipping[5], tipping_bias=tipping[3],
        treatment_outcome_partial_r2=(abs(statistic) / math.hypot(abs(statistic), math.sqrt(df)))**2,
        adjusted_joint_covariance=None,
    )
    inference_columns = ["estimate", "std_error", "t", "df", "p_value", "ci_lower", "ci_upper"]
    return m.result(
        "ovb_sensitivity",
        {
            "coefficients": m.frame(coefficient_rows, columns=inference_columns, index=terms),
            "covariance": m.frame(covariance, columns=terms, index=terms),
            "sensitivity": m.frame(
                rows, columns=["r2_treatment", "r2_outcome", "direction", "bias_magnitude", "variance", *inference_columns]
            ),
            "robustness": m.frame(
                [[robustness, robustness, tipping[3], tipping[5], abs(statistic), df]],
                columns=["r2_treatment", "r2_outcome", "bias_magnitude", "tipping_estimate", "original_abs_t", "original_df"],
            ),
        },
        metadata, settings, state,
        notes=[
            "Exact partial-R2 identities for one omitted scalar regressor under classical homoskedastic OLS.",
            "Student t inference is exact under independent homoskedastic normal errors in the linear model; otherwise it is an approximation.",
            "The strength grid and linear adjustment do not establish causal identification.",
            "Reduce follows the original coefficient sign through zero; an exactly zero coefficient uses positive orientation.",
            "Robustness is the equal-strength point-estimate zero threshold, not a significance threshold.",
            "Only the original full model covariance and adjusted target variances are identified; joint augmented or cross-scenario covariance is not supplied.",
        ],
    )
