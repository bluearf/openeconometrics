"""Declared Gaussian finite-sample joint inference; native float64 CPU Torch.

Known covariance shape with one independent residual scale (joint t) and an
unknown multivariate covariance (Hotelling) have different sampling laws.
Neither law applies automatically to a robust or arbitrary fitted covariance.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.postest.multiple import (
    _labels, _level, _matrix, _shape, _vector, stepdown,
)
from openecon.engines.distributions import f_isf
from openecon.engines.inference import student_t_two_sided
from openecon.models import ResultBundle
from openecon.resources import plan_workspace

MAX_DIMENSION = 384
MAX_CELLS = 8_000_000
MAX_WORK = 100_000_000


def _df(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not 1 <= value <= 1_000_000:
        raise AnalysisError("invalid_df", "Supply one finite common df between 1 and 1,000,000.")
    return float(value)


def _description(value, name):
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError("missing_pivot_design", f"Describe {name} explicitly.")
    return value.strip()


def _admit(n, draws, seed, *, stored_dimensions=0, sample_count=0):
    if isinstance(draws, bool) or not isinstance(draws, int) or not 1000 <= draws <= 200_000:
        raise AnalysisError("invalid_draws", "draws must be an integer between 1000 and 200000.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise AnalysisError("invalid_seed", "seed must be a nonnegative signed 64-bit integer.")
    work = draws * n * n + n**3
    if not 1 <= n <= MAX_DIMENSION or n * draws > MAX_CELLS or work > MAX_WORK:
        raise AnalysisError("work_budget", "The joint-t family exceeds its declared work budget.")
    plan = plan_workspace("finite-sample joint t", {
        "noise, transformed draws and stepdown selection": 8 * draws * n * 6,
        "scale sampler, masks and critical-value sorting": 8 * draws * 32,
        "covariance, correlation, spectral factors and result vectors": 8 * (n * n * 12 + n * 16),
        "stored model covariance and sample identity validation": 8 * stored_dimensions**2 * 12 + 64 * sample_count,
    })
    return plan.record(), work


def _geometry(covariance, n, labels):
    cov = _matrix(covariance, "covariance", columns=n, rows=n)
    if (cov.diag() <= 0).any():
        raise AnalysisError("invalid_covariance", "Every member needs a positive variance.")
    se = cov.diag().sqrt()
    corr = cov / se[:, None] / se[None, :]
    if not torch.isfinite(corr).all() or not torch.allclose(corr, corr.T, atol=1e-12, rtol=1e-10):
        raise AnalysisError("invalid_covariance", "Covariance must be symmetric in marginal-SE units.")
    corr = (corr + corr.T) * 0.5
    order = sorted(range(n), key=lambda i: str(labels[i]))
    eigenvalues, vectors = torch.linalg.eigh(corr[order][:, order])
    if eigenvalues.min() < -1e-12:
        raise AnalysisError("invalid_covariance", "Covariance must be positive semidefinite.")
    return cov, se, corr, order, eigenvalues, vectors


def _chi_square(draws, df, generator):
    """Marsaglia-Tsang gamma sampler with local RNG and bounded rejection.

    Generate chi-square(df) as 2*Gamma(df/2,1). For shape < 1 use the
    Gamma(a+1)*U**(1/a) identity. No private Torch API or global RNG mutation.
    Source: Marsaglia & Tsang (2000), doi:10.1145/358407.358414.
    """
    shape = df / 2
    augmented = shape + 1 if shape < 1 else shape
    d = augmented - 1 / 3
    c = 1 / math.sqrt(9 * d)
    values = torch.empty(draws, dtype=torch.float64)
    pending = torch.arange(draws)
    attempts = 0
    while pending.numel() and attempts < 128:
        attempts += 1
        z = torch.randn(len(pending), dtype=torch.float64, generator=generator)
        u = torch.rand(len(pending), dtype=torch.float64, generator=generator)
        v = (1 + c * z).pow(3)
        positive = v > 0
        safe_v = v.clamp_min(torch.finfo(torch.float64).tiny)
        accept = positive & ((u < 1 - 0.0331 * z.pow(4)) |
                             (u.log() < 0.5 * z.square() + d * (1 - v + safe_v.log())))
        values[pending[accept]] = 2 * d * v[accept]
        pending = pending[~accept]
    if pending.numel():
        raise AnalysisError("sampling_failure", "The bounded chi-square sampler did not complete.")
    if shape < 1:
        values *= torch.rand(draws, dtype=torch.float64, generator=generator).pow(1 / shape)
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise AnalysisError("sampling_failure", "The common chi-square scale is unresolved.")
    return values, attempts


def _joint_t(covariance, n, labels, df, draws, seed):
    cov, se, corr, order, eigenvalues, vectors = _geometry(covariance, n, labels)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(draws, n, dtype=torch.float64, generator=generator)
    canonical = noise @ (vectors * eigenvalues.clamp_min(0).sqrt()).T
    chi_square, attempts = _chi_square(draws, df, generator)
    canonical /= (chi_square / df).sqrt()[:, None]
    if not torch.isfinite(canonical).all():
        raise AnalysisError("sampling_failure", "The joint-t simulation produced nonfinite pivots.")
    inverse = torch.argsort(torch.tensor(order))
    return canonical[:, inverse], cov, se, corr, order, eigenvalues, attempts


@torch.no_grad()
def simultaneous_t_ci(
    estimates, covariance, *, df, pivot_description, family_description,
    labels=None, alpha=0.05, draws=50_000, seed=1729,
):
    """Max-|t| intervals for a known joint shape and ONE independent common scale.

    Caller must justify (estimate-true)/SE = correlated N(0,R)/sqrt(chi2(df)/df),
    with known R and an independent common chi-square denominator. A generic
    robust covariance or separate coefficient dfs do not satisfy this contract.
    Singular PSD shapes are permitted; zero marginal variances are refused.
    Critical values have recorded Monte Carlo uncertainty, not exact precision.
    """
    _level(alpha)
    df = _df(df)
    pivot_description = _description(pivot_description, "the known-shape/common-scale pivot")
    family_description = _description(family_description, "the compatible estimate family")
    (n,) = _shape(estimates, "estimates", 1)
    plan, work = _admit(n, draws, seed)
    estimates = _vector(estimates, "estimates", maximum=MAX_DIMENSION)
    names = _labels(labels, n)
    simulated, cov, se, corr, order, eig, attempts = _joint_t(
        covariance, n, names, df, draws, seed,
    )
    critical = float(torch.quantile(simulated.abs().amax(1), 1 - alpha, interpolation="higher"))
    low, high = estimates - critical * se, estimates + critical * se
    if not math.isfinite(critical) or not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise AnalysisError("nonfinite_result", "The joint-t interval endpoints are unresolved.")
    return table({
        "hypothesis": names, "estimate": estimates.tolist(), "std_error": se.tolist(),
        "ci_low": low.tolist(), "ci_high": high.tolist(),
    }, title="Common-scale joint-t simultaneous confidence intervals",
        distribution="Finite-sample joint t with known correlation and one common independent scale",
        df=df, degrees_of_freedom=df, alpha=alpha, family_size=n, family_members=names,
        family_description=family_description, pivot_description=pivot_description,
        critical_value=critical, joint_covariance=cov.tolist(), joint_correlation=corr.tolist(),
        draws=draws, seed=seed, simulation_order_positions=order,
        simulation_hypotheses=[names[i] for i in order],
        seed_order_policy="Canonical unique labels", smallest_correlation_eigenvalue=float(eig.min()),
        clipped_eigenvalues=int((eig < 0).sum()), psd_tolerance=1e-12,
        cdf_monte_carlo_std_error=math.sqrt(alpha * (1 - alpha) / draws),
        chi_square_sampler="Marsaglia-Tsang with shape augmentation below one",
        sampler_max_iterations=128, sampler_iterations=attempts, failures=0,
        assumptions="Known covariance shape and independent common chi-square residual scale; no arbitrary robust covariance",
        pivot_design_verified=False, missing_policy="raise", precision="float64", device="cpu",
        dataset_support=False, resource_plan=plan, estimated_work=work, stata_parity_validated=False,
    )


@torch.no_grad()
def ols_stepdown(
    result, *, error_model, terms=None, null_values=None, tail="two-sided",
    alpha=0.05, draws=50_000, seed=1729,
):
    """Finite-sample maxT stepdown for one classical fixed-design OLS/WLS fit.

    Explicit error_model='iid_gaussian' (unweighted OLS) or
    'known_precision_gaussian' (aweights treated as KNOWN relative precision).
    Targets are stored coefficients against declared constants. All true-null
    subsets have the same joint-t law irrespective of false-null coefficients:
    subset pivotality provides strong FWER control, conditional on fixed design.
    Robust/cluster/HAC/absorbed/model mixtures and data-dependent selection of
    the reported family are unsupported. No fitting data or stored fit is changed.
    """
    _level(alpha)
    if tail not in ("two-sided", "less", "greater"):
        raise AnalysisError("invalid_method", "Choose a two-sided, less or greater alternative.")
    if not isinstance(result, ResultBundle) or result.spec.estimator != "ols":
        raise AnalysisError("unsupported_model", "Supply one restored ordinary OLS ResultBundle.")
    spec = result.spec
    if (spec.covariance != "nonrobust" or result.inference.get("covariance") != "nonrobust"
            or spec.cluster is not None or spec.panel is not None
            or result.inference.get("distribution") != "t"
            or result.inference.get("dfadjust") or result.inference.get("hansen")):
        raise AnalysisError("unsupported_covariance", "Only classical nonrobust fixed-design OLS is supported.")
    if result.provenance.get("device") != "cpu" or result.provenance.get("sample_positions_omitted"):
        raise AnalysisError("unsupported_domain", "Use an ordinary resident CPU fit with complete sample identities.")
    weighted = spec.weights is not None
    if (not weighted and error_model != "iid_gaussian") or (
        weighted and (spec.weight_type != "aweight" or error_model != "known_precision_gaussian")
    ):
        raise AnalysisError("unsupported_error_model", "Declare iid Gaussian OLS or known-relative-precision Gaussian aweights.")
    coefficient_names = [c.term for c in result.coefficients]
    k = len(coefficient_names)
    if not 1 <= k <= MAX_DIMENSION or len(set(coefficient_names)) != k:
        raise AnalysisError("invalid_family", "The stored coefficient identities are not a bounded unique family.")
    if terms is not None and not hasattr(terms, "__len__"):
        raise AnalysisError("invalid_family", "Choose unique stored coefficient identities.")
    names = coefficient_names if terms is None else _labels(terms, len(terms))
    if not names or any(name not in coefficient_names for name in names):
        raise AnalysisError("invalid_family", "Choose unique stored coefficient identities.")
    indices = [coefficient_names.index(name) for name in names]
    df = _df(result.metrics.get("df_resid"))
    if result.nobs > 1_000_000:
        raise AnalysisError("work_budget", "Stored sample identity validation is bounded to 1,000,000 rows.")
    plan, work = _admit(len(names), draws, seed, stored_dimensions=k, sample_count=result.nobs)
    if (df != result.nobs - k or result.inference.get("df_inference") != df
            or len(result.sample_positions) != result.nobs
            or len(set(result.sample_positions)) != result.nobs
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < result.nobs_original
                   for i in result.sample_positions)
            or not result.provenance.get("sample_hash")):
        raise AnalysisError("invalid_result_state", "The stored df/sample identities are incompatible with ordinary OLS.")
    full_cov = _matrix(result.covariance_matrix, "stored covariance", columns=k, rows=k)
    stored_se = torch.tensor([c.std_error for c in result.coefficients], dtype=torch.float64)
    if not torch.allclose(full_cov.diag(), stored_se.square(), rtol=1e-9, atol=0):
        raise AnalysisError("invalid_result_state", "Stored coefficient SEs and full covariance disagree.")
    selected_cov = full_cov[indices][:, indices]
    estimates = torch.tensor([result.coefficients[i].estimate for i in indices], dtype=torch.float64)
    null = torch.zeros(len(names), dtype=torch.float64) if null_values is None else _vector(null_values, "null_values", maximum=MAX_DIMENSION)
    if len(null) != len(names):
        raise AnalysisError("invalid_family", "Supply one null value in the requested coefficient order.")
    simulated, cov, se, corr, order, _, attempts = _joint_t(
        selected_cov, len(names), names, df, draws, seed,
    )
    observed = (estimates - null) / se
    if not torch.isfinite(observed).all():
        raise AnalysisError("nonfinite_result", "The coefficient null statistics are unresolved.")
    output = stepdown(observed, simulated, labels=names, tail=tail, alpha=alpha,
        null_description="Exact Gaussian fixed-design coefficient joint-t pivot; one independent residual scale; subset pivotality",
    )
    analytic_two_sided = [student_t_two_sided(float(t), df) for t in observed]
    if tail == "two-sided":
        analytic = analytic_two_sided
    else:
        greater = [p / 2 if t >= 0 else 1 - p / 2 for t, p in zip(observed, analytic_two_sided, strict=True)]
        analytic = greater if tail == "greater" else [1 - p for p in greater]
    output["p_value"] = analytic
    output["estimate"] = estimates.tolist()
    output["null_value"] = null.tolist()
    output["std_error"] = se.tolist()
    count_correction = 1
    output["adjusted_exceedances"] = [int(round(p * (draws + 1) - count_correction)) for p in output.adjusted_p_value]
    output.attrs.update(
        title="Gaussian fixed-design OLS/WLS joint-t stepdown", source_model_id=result.id,
        source_estimator="ols", error_model=error_model, null_design_verified=True,
        p_value_source="Analytic marginal Student-t; adjusted p-values use Monte Carlo plus-one maxT",
        assumptions="Fixed full-rank design and sample selection independent of errors; declared Gaussian errors, known relative precision if weighted; common independent residual scale; subset pivotality; family/model not selected using outcomes",
        df=df, degrees_of_freedom=df, seed=seed, null_values=null.tolist(),
        joint_covariance=cov.tolist(), joint_correlation=corr.tolist(),
        simulation_order_positions=order, sampler_iterations=attempts, sampler_max_iterations=128,
        chi_square_sampler="Marsaglia-Tsang", failures=0, nobs=result.nobs,
        nobs_original=result.nobs_original, dropped_rows=result.dropped_rows,
        sample_hash=result.provenance["sample_hash"], data_hash=result.provenance.get("data_hash"),
        sample_positions_sha256=hashlib.sha256(json.dumps(result.sample_positions, separators=(",", ":")).encode()).hexdigest(),
        missing_policy=spec.missing, weights=spec.weights, weight_type=spec.weight_type,
        covariance="nonrobust", resource_plan=plan, estimated_work=work,
        null_generation="Parametric pivotal joint t; no nuisance refit or model bootstrap",
    )
    return output


@torch.no_grad()
def hotelling_region(data, columns, *, sampling_model, alpha=0.05, missing="raise"):
    """Hotelling T² confidence ellipsoid for an iid multivariate-normal mean.

    sampling_model='iid_multivariate_normal' must be declared explicitly.
    n>p and full-rank covariance are required. Coordinate intervals are
    conservative projections of the EXACT ellipsoid, not an exact rectangle.
    Complete rows are shared across all columns; missing='drop' is explicit.
    Input must be a resident DataFrame. No weights, clusters or Dataset route.
    """
    _level(alpha)
    if sampling_model != "iid_multivariate_normal":
        raise AnalysisError("unsupported_sampling_model", "Declare iid_multivariate_normal sampling explicitly.")
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError("unsupported_domain", "Hotelling accepts a resident pandas DataFrame.")
    if isinstance(columns, (str, bytes)) or not isinstance(columns, (list, tuple)):
        raise AnalysisError("invalid_columns", "Supply a unique list of numeric column names.")
    p = len(columns)
    if not 1 <= p <= MAX_DIMENSION or any(not isinstance(c, str) for c in columns) or len(set(columns)) != p:
        raise AnalysisError("invalid_columns", "Supply a bounded unique family of column names.")
    if data.columns.has_duplicates or any(c not in data for c in columns):
        raise AnalysisError("invalid_columns", "Required columns must each exist exactly once.")
    if missing not in ("raise", "drop"):
        raise AnalysisError("invalid_missing", "Choose missing='raise' or missing='drop'.")
    if any(not pd.api.types.is_numeric_dtype(data[c].dtype) or pd.api.types.is_bool_dtype(data[c].dtype)
           or pd.api.types.is_complex_dtype(data[c].dtype) for c in columns):
        raise AnalysisError("invalid_values", "Every selected column must contain real numeric values.")
    original_n = len(data)
    work = original_n * p * p + p**3
    if original_n * p > MAX_CELLS or work > MAX_WORK:
        raise AnalysisError("work_budget", "The resident Hotelling sample exceeds its work budget.")
    plan = plan_workspace("Hotelling mean region", {
        "selected numeric sample and centered/normalized buffers": 8 * original_n * p * 8,
        "sample masks and positions": 8 * original_n * 8,
        "covariance, correlation and normalized region geometry": 8 * p * p * 12,
    }).record()
    selected = data.loc[:, list(columns)]
    complete = ~selected.isna().any(axis=1)
    if not complete.all() and missing == "raise":
        raise AnalysisError("missing_values", "All variables require the same complete sample; choose missing='drop' explicitly.")
    positions = complete.to_numpy().nonzero()[0].tolist()
    n = len(positions)
    if n <= p:
        raise AnalysisError("insufficient_observations", "Hotelling needs more complete rows than variables.")
    try:
        x = torch.as_tensor(selected.loc[complete].to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise AnalysisError("invalid_values", "The selected sample must contain finite real values.") from exc
    if not torch.isfinite(x).all():
        raise AnalysisError("nonfinite_values", "The selected sample must contain finite values.")
    # Translate before mean calculation to reduce loss at large origins.
    origin = x[0]
    shifted = x - origin
    mean_shift = shifted.mean(0)
    center = origin + mean_shift
    centered = shifted - mean_shift
    scale = centered.abs().amax(0)
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise AnalysisError("singular_covariance", "Each variable needs finite resolved variation.")
    normalized = centered / scale
    normalized_mean_cov = normalized.T @ normalized / ((n - 1) * n)
    standardized_se = normalized_mean_cov.diag().sqrt()
    corr = normalized_mean_cov / standardized_se[:, None] / standardized_se[None, :]
    corr = (corr + corr.T) * 0.5
    eig = torch.linalg.eigvalsh(corr)
    if eig.min() <= 1e-12 * eig.max():
        raise AnalysisError("singular_covariance", "The joint covariance is singular or numerically unresolved.")
    normalized_precision = torch.linalg.solve(normalized_mean_cov, torch.eye(p, dtype=torch.float64))
    se = scale * standardized_se
    cov = normalized_mean_cov * scale[:, None] * scale[None, :]
    radius_sq = p * (n - 1) / (n - p) * f_isf(alpha, p, n - p)
    width = math.sqrt(radius_sq) * se
    low, high = center - width, center + width
    if not math.isfinite(radius_sq) or not torch.isfinite(cov).all() or (cov.diag() <= 0).any() or not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise AnalysisError("nonfinite_result", "Hotelling region/endpoints are outside float64 support.")
    from openecon.analysis import _frame_hasher

    return table({"variable": list(columns), "estimate": center.tolist(), "std_error": se.tolist(),
                  "projected_ci_low": low.tolist(), "projected_ci_high": high.tolist()},
        title="Hotelling mean confidence ellipsoid and coordinate projections",
        alpha=alpha, nobs=n, nobs_original=original_n, dropped_rows=original_n - n,
        dimensions=p, df_numerator=p, df_denominator=n - p,
        distribution="Hotelling T-squared: p*(n-1)/(n-p) times F(p,n-p)",
        region_center=center.tolist(), region_scale=scale.tolist(),
        normalized_region_precision=normalized_precision.tolist(), region_radius_squared=radius_sq,
        region_definition="((theta-center)/region_scale)' normalized_region_precision ((theta-center)/region_scale) <= region_radius_squared",
        coordinate_projection_coverage="At least 1-alpha; the ellipsoid itself has exact coverage under the declared model",
        joint_covariance=cov.tolist(), joint_correlation=corr.tolist(), sample_positions=positions,
        sample_hash=_frame_hasher(selected.loc[complete].reset_index(drop=True)).hexdigest(),
        sample_position_base=0, missing_policy=missing, sampling_model=sampling_model,
        assumptions="One iid multivariate normal sample, unknown positive-definite covariance, n>p; no weights or dependent rows",
        source="https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm",
        precision="float64", device="cpu", dataset_support=False, resource_plan=plan,
        estimated_work=work, stata_parity_validated=False,
    )
