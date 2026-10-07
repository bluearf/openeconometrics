"""Machado--Santos Silva panel quantiles via location/scale moments.

Y_it = alpha_i + X_it beta + (delta_i + X_it gamma) U_it;
E U = 0, E |U| = 1, common continuous U distribution independent of X.
Scale uses 2r(I[r>=0]-lambda_hat), the authors' centered-sign transformation.
Joint inference profiles all individual effects analytically, rather than
pretending the estimated standardized residuals are observed disturbances.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, kernel_call, make_spec
from openecon.engines import linalg
from openecon.engines.absorb import group_means
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import tensor_bytes


def sample_quantile(values: Tensor, tau: float) -> float:
    """Empirical inverse CDF, midpoint at an exact n*tau (Hyndman-Fan type 2)."""
    ordered = torch.sort(values).values
    target = len(values) * tau
    nearest = round(target)
    if abs(target - nearest) <= 1e-10 and 0 < nearest < len(values):
        return float((ordered[nearest - 1] + ordered[nearest]) / 2)
    return float(ordered[max(0, min(len(values) - 1, math.ceil(target) - 1))])


def density_at(values: Tensor, quantiles: Tensor, bandwidth: float | None):
    spread = min(
        float(values.std()), (sample_quantile(values, 0.75) - sample_quantile(values, 0.25)) / 1.349
    )
    if not spread > 0:
        raise AnalysisError(
            "degenerate_sparsity", "Standardized residuals do not have positive continuous spread."
        )
    bandwidth = 0.9 * spread * len(values) ** (-0.2) if bandwidth is None else bandwidth
    densities = []
    # No N by Q density grid: quantiles can be numerous.
    for q in quantiles:
        z = (values - q) / bandwidth
        densities.append(torch.exp(-z.square() / 2).mean() / (bandwidth * math.sqrt(2 * math.pi)))
    density = torch.stack(densities)
    if not bool(torch.isfinite(density).all()) or bool((density <= 1e-12).any()):
        raise AnalysisError(
            "degenerate_sparsity",
            "The quantile-process density is not positive at every requested quantile.",
        )
    return density, float(bandwidth)


def profile_influence(
    x: Tensor,
    residual: Tensor,
    scale: Tensor,
    scale_residual: Tensor,
    codes: Tensor,
    groups: int,
    quantiles: Tensor,
    probabilities: Tensor,
    density: Tensor,
    sign_center: float,
    bread: Tensor,
):
    """Rows of IF for (beta, gamma, q), equal to the full dummy-moment inverse.

    The full system has location/scale dummy scores plus quantile CDF scores.
    Its minus Jacobian blocks are L'L, L'diag(h)L, L'L, and
    n*f(q)*[mean(L/s), q*mean(L/s), 1].  Profiling the two dummy
    sets avoids O(G^2) covariance and O(NG) dummy construction.
    The scale derivative h=2(I[r>=0]-lambda) includes the location
    stage. The lambda derivative vanishes because L'r=0 (same X).
    """
    n, p = x.shape
    sizes = torch.bincount(codes, minlength=groups).to(torch.float64)
    h = 2 * ((residual >= 0).to(torch.float64) - sign_center)
    hbar = group_means(h, codes, groups)
    hxbar = group_means(h[:, None] * x, codes, groups)
    beta_if = (x * residual[:, None]) @ bread
    cross = x.T @ (h[:, None] * x)
    gamma_if = (x * scale_residual[:, None]) @ bread - beta_if @ cross @ bread
    # Direct influence of each individual location effect in scale estimation.
    gamma_if -= residual[:, None] * (hxbar[codes] @ bread)
    invscale_sum = torch.zeros(groups, dtype=torch.float64)
    invscale_sum.index_add_(0, codes, 1 / scale)
    weighted_x = (x / scale[:, None]).mean(dim=0)
    alpha_direct = residual / sizes[codes]
    location_part = beta_if @ weighted_x + invscale_sum[codes] * alpha_direct / n
    # Scale fixed effects include the first-stage location effect and slope.
    scale_beta = (invscale_sum[:, None] * hxbar).sum(dim=0) / n
    scale_part = gamma_if @ weighted_x - beta_if @ scale_beta
    scale_part += (
        invscale_sum[codes] / n * (scale_residual / sizes[codes] - hbar[codes] * alpha_direct)
    )
    u = residual / scale
    q_if = []
    for tau, q, f in zip(probabilities, quantiles, density, strict=True):
        q_if.append((tau - (u <= q).to(torch.float64)) / (n * f) - location_part - q * scale_part)
    return torch.cat([beta_if, gamma_if, torch.stack(q_if, dim=1)], dim=1)


def fit_panel_mmqr(spec: ModelSpec, data: Any) -> ResultBundle:
    frame = ModelFrame(spec, data)
    taus = list(frame.option("quantiles"))
    if not taus or len(set(taus)) != len(taus) or any(not 0.05 <= tau <= 0.95 for tau in taus):
        raise AnalysisError(
            "invalid_quantile", "panel_mmqr needs distinct quantiles in [0.05,0.95]."
        )
    # Preserve caller's quantile order; the process covariance follows this order.
    frame.sort_panel()
    codes, groups = frame.codes(spec.panel)
    sizes = torch.bincount(codes, minlength=groups)
    if groups < 2:
        raise AnalysisError("insufficient_panels", "Panel MM-QR requires at least two individuals.")
    if bool((sizes < 3).any()):
        raise AnalysisError(
            "insufficient_periods",
            "Each retained panel needs at least three observations; large-T inference still applies.",
        )
    width = frame.design_width(intercept=False)
    dimension = 2 * width + len(taus)
    frame.workspace_plan(
        "panel MM-QR profiled moments and quantile process",
        {
            "within_design_and_moment_buffers": tensor_bytes((frame.n, width + 12), itemsize=64),
            "joint_influence_rows": tensor_bytes((frame.n, dimension), itemsize=24),
            "panel_means": tensor_bytes((groups, width + 5), itemsize=32),
            "joint_and_reported_covariance": tensor_bytes((dimension, dimension), itemsize=24)
            + tensor_bytes((len(taus) * width, len(taus) * width), itemsize=24),
        },
    )
    design = frame.design(intercept=False)
    means = group_means(design.x, codes, groups)
    within = design.x - means[codes]
    design, within = frame.drop_absorbed(design, within)
    if not design.terms:
        raise AnalysisError(
            "no_within_variation", "No predictor has estimable within-panel variation."
        )
    x = design.x
    p = x.shape[1]
    if frame.n <= groups + p:
        raise AnalysisError(
            "insufficient_observations",
            "Panel MM-QR needs residual degrees of freedom after all individual effects.",
        )
    means = group_means(x, codes, groups)
    y = frame.numeric(spec.outcome)
    ybar = group_means(y, codes, groups)
    first = kernel_call(linalg.least_squares, within, y - ybar[codes], drop_collinear=False)
    beta = first.beta
    alpha = ybar - means @ beta
    residual = y - alpha[codes] - x @ beta
    if float(residual.square().sum()) <= 1e-28 * float(y.square().sum()):
        raise AnalysisError(
            "perfect_fit", "The panel location equation is fitted exactly; the scale is undefined."
        )
    sign_center = float((residual >= 0).to(torch.float64).mean())
    transformed = 2 * residual * ((residual >= 0).to(torch.float64) - sign_center)
    abar = group_means(transformed, codes, groups)
    second = kernel_call(
        linalg.least_squares, within, transformed - abar[codes], drop_collinear=False
    )
    gamma = second.beta
    delta = abar - means @ gamma
    scale = delta[codes] + x @ gamma
    floor = max(float(transformed.abs().max()) * 1e-12, 1e-300)
    if bool((scale <= floor).any()):
        raise AnalysisError(
            "nonpositive_scale",
            f"{int((scale <= floor).sum())} fitted scales are nonpositive or numerically zero; the linear location-scale model is outside its supported domain.",
        )
    u = residual / scale
    probabilities = torch.tensor(taus, dtype=torch.float64)
    quantiles = torch.tensor([sample_quantile(u, tau) for tau in taus], dtype=torch.float64)
    density, bandwidth = density_at(u, quantiles, frame.option("density_bandwidth"))
    scale_residual = transformed - scale
    influence = profile_influence(
        within,
        residual,
        scale,
        scale_residual,
        codes,
        groups,
        quantiles,
        probabilities,
        density,
        sign_center,
        first.xtx_inv,
    )
    n = frame.n
    df_resid = n - groups - p
    kind = spec.covariance
    infer = {
        "covariance": kind,
        "quantile_density": "Gaussian KDE of standardized residuals",
        "density_bandwidth": bandwidth,
        "scale_uncertainty": "joint profile location/scale/CDF moments",
        "individual_effects": "absorbed in both location and scale",
        "k_small_sample": groups + p,
    }
    if kind == "HC0":
        joint = influence.T @ influence
        df = df_resid
        infer.update(
            {"correction": "HC0 joint profile-moment sandwich", "small_sample_correction": 1.0}
        )
    else:
        if kind == "robust":
            cluster_codes, count = codes, groups
            column = spec.panel
        else:
            cluster_codes, count = frame.cluster_dimensions()[0]
            column = spec.cluster if isinstance(spec.cluster, str) else spec.cluster[0]
            minimum = torch.full((groups,), count, dtype=torch.int64)
            maximum = torch.full((groups,), -1, dtype=torch.int64)
            minimum.scatter_reduce_(0, codes, cluster_codes, reduce="amin", include_self=True)
            maximum.scatter_reduce_(0, codes, cluster_codes, reduce="amax", include_self=True)
            if not bool((minimum == maximum).all()):
                raise AnalysisError(
                    "cluster_not_nested", "Each panel must lie inside one cluster for panel MM-QR."
                )
        if count < 30:
            frame.warn(f"Only {count} independent clusters; cluster inference may be unreliable.")
        sums = torch.zeros(
            (count, dimension if p == width else 2 * p + len(taus)), dtype=torch.float64
        )
        sums.index_add_(0, cluster_codes, influence)
        factor = count / (count - 1) * (n - 1) / df_resid
        joint = (sums.T @ sums) * factor
        df = count - 1
        infer.update(
            {
                "correction": "joint profile-moment CR1: G/(G-1)*(N-1)/(N-I-K)",
                "cluster_count": count,
                "cluster_df": df,
                "cluster_column": column,
                "small_sample_correction": factor,
            }
        )
    # Delta method for the joint quantile process, not independent per-quantile SEs.
    transform = torch.zeros((len(taus) * p, 2 * p + len(taus)), dtype=torch.float64)
    parameters = []
    terms = []
    equations = []
    for index, (tau, q) in enumerate(zip(taus, quantiles, strict=True)):
        block = slice(index * p, (index + 1) * p)
        transform[block, :p] = torch.eye(p, dtype=torch.float64)
        transform[block, p : 2 * p] = q * torch.eye(p, dtype=torch.float64)
        transform[block, 2 * p + index] = gamma
        parameters.append(beta + q * gamma)
        terms.extend(f"q{tau:g}:{term}" for term in design.terms)
        equations.extend([f"q{tau:g}"] * p)
    covariance = transform @ joint @ transform.T
    frame.warn(
        "Panel MM-QR uses a common strictly exogenous location-scale error distribution and large-T asymptotics; its scale and quantile estimates have incidental-parameter bias for fixed T. No split-panel bias correction is applied."
    )
    if groups / float(sizes.min()) >= 1:
        frame.warn(
            "The number of individuals is not small relative to the shortest panel; n=o(T) is not an appropriate finite-sample approximation and nominal confidence intervals may be biased."
        )
    levels = frame.levels(spec.panel)
    effects = [
        {
            "panel": level,
            "location": float(alpha[i]),
            "scale": float(delta[i]),
            "quantiles": {
                f"{tau:g}": float(alpha[i] + q * delta[i])
                for tau, q in zip(taus, quantiles, strict=True)
            },
        }
        for i, level in enumerate(levels)
    ]
    fitted = y - residual + scale * quantiles[int(torch.argmin((probabilities - 0.5).abs()))]
    moment_error = (
        max(
            float((within.T @ residual).abs().max()), float((within.T @ scale_residual).abs().max())
        )
        / n
    )
    return build_result(
        frame,
        terms=terms,
        params=torch.cat(parameters),
        covariance=covariance,
        equations=equations,
        use_t=True,
        df_inference=df,
        df_resid=df_resid,
        fitted=fitted,
        solver="absorbed location/scale least squares and standardized empirical quantiles",
        solver_diagnostics={
            "converged": True,
            "location_rank": p,
            "scale_rank": p,
            "moment_error": moment_error,
            "cdf_moment_errors": [
                abs(float((u <= q).to(torch.float64).mean()) - tau)
                for tau, q in zip(taus, quantiles, strict=True)
            ],
        },
        inference=infer,
        categories=design.categories,
        metrics={
            "n_groups": groups,
            "t_min": int(sizes.min()),
            "t_max": int(sizes.max()),
            "t_avg": n / groups,
            "minimum_scale": float(scale.min()),
            "density_bandwidth": bandwidth,
            "sign_probability": sign_center,
        },
        provenance={
            "published_reference": "Machado and Santos Silva (2019), DOI 10.1016/j.jeconom.2019.04.009, section 3.1",
            "covariance_reference": "joint triangular profile-moment delta sandwich; not xtqreg V parity",
            "quantile_domain": [0.05, 0.95],
            "scale_design": "same encoded X as location",
            "distribution_assumptions": "common continuous standardized U independent of X; E U=0, E|U|=1; independent clusters; growing panel lengths",
            "prediction_sample_quantile": taus[int(torch.argmin((probabilities - 0.5).abs()))],
        },
        extra={
            "location_terms": design.terms,
            "location_coefficients": beta.tolist(),
            "scale_coefficients": gamma.tolist(),
            "error_quantiles": quantiles.tolist(),
            "quantiles": taus,
            "quantile_density": density.tolist(),
            "joint_moment_covariance": joint.tolist(),
            "joint_moment_order": [
                *[f"location:{t}" for t in design.terms],
                *[f"scale:{t}" for t in design.terms],
                *[f"error_quantile:{tau:g}" for tau in taus],
            ],
            "individual_effects": effects,
            "fitted_scale": scale.tolist(),
            "fitted_scale_sample_positions": frame.positions,
            "scale_transform": "2r*(I[r>=0]-mean(I[r>=0]))",
            "quantile_convention": "Hyndman-Fan type 2 empirical inverse CDF",
            "bias_correction": "none",
        },
    )


def panel_mmqr(
    *,
    data: Any,
    y: str,
    x: Sequence[str],
    panel: str,
    time: str | None = None,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    covariance: str | None = None,
    cluster: str | None = None,
    categorical: Sequence[str] | None = None,
    density_bandwidth: float | None = None,
    missing: str = "raise",
    alpha: float = 0.05,
) -> ResultBundle:
    """Linear panel location/scale quantiles with distributional individual effects.

    ``robust`` clusters on panel, ``cluster`` on a column nesting panels, and
    ``HC0`` assumes independent observations. Quantiles must be in [.05,.95].
    Same X in both stages, no weights, no finite-T bias correction.
    """
    from openecon.analysis import fit

    if isinstance(quantiles, (str, bytes)) or not isinstance(quantiles, Sequence):
        raise AnalysisError("invalid_spec", "quantiles must be a sequence of probabilities.")
    spec = make_spec(
        "panel_mmqr",
        outcome=y,
        predictors=column_list(x, "x"),
        panel=panel,
        time=time,
        intercept=False,
        covariance=covariance,
        cluster=cluster,
        categorical=column_list(categorical, "categorical"),
        missing=missing,
        alpha=alpha,
        options={"quantiles": list(quantiles), "density_bandwidth": density_bandwidth},
    )
    return fit(spec, data=data)
