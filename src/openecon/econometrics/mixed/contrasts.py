"""Scalar model-based Satterthwaite contrasts for saved Gaussian mixed fits."""

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, table
from openecon.econometrics.mixed.lmm import _Setup, _likelihood
from openecon.econometrics.postest.common import matched_frame, require_result
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines import distributions, optimize
from openecon.engines.contracts import KernelError
from openecon.resources import plan_workspace


@resident_cpu
def mixed_satterthwaite(result, *, data, contrast, null=0.0, alpha=0.05,
                        max_work=200_000_000):
    """Model-based scalar Satterthwaite t approximation for saved Gaussian mixed ML/REML.

    df=2*v^2/(gradient(v)' Cov(theta) gradient(v)); theta information is
    recomputed at the saved estimates, without optimization or refitting.
    This model-based approximation is not Kenward-Roger, cluster-robust or
    a universal small-cluster coverage guarantee. Known-population fixed
    design contrasts are marginal over the zero-mean random effects.
    """
    result = require_result(result, "mixed result")
    spec = result.spec
    if (spec.estimator != "mixed" or spec.covariance != "nonrobust"
            or spec.weights or spec.categorical or spec.panel):
        raise AnalysisError("unsupported_spec", "Use numeric unweighted model-based mixed ML/REML.")
    if (not isinstance(contrast, dict) or not contrast
            or any(not isinstance(k, str) or isinstance(v, bool)
                   or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for k, v in contrast.items())
            or not any(v != 0 for v in contrast.values())):
        raise AnalysisError("invalid_contrast", "Give finite nonzero named fixed-effect weights.")
    if (isinstance(null, bool) or not isinstance(null, (int, float)) or not math.isfinite(null)
            or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1):
        raise AnalysisError("invalid_option", "Use a finite null and alpha in (0,1).")
    if result.nobs_original > 8192 or result.nobs > 8192 or len(result.coefficients) > 64:
        raise AnalysisError("work_budget", "Saved contrast geometry is bounded to 8192 rows/64 parameters.")
    p_requested = len(spec.predictors) + int(spec.intercept)
    random = spec.columns.get("random", [])
    q_requested = len(random if isinstance(random, list) else [random]) + int(spec.options.get("random_intercept", True))
    saved_theta = result.extra.get("theta", [])
    if (not isinstance(saved_theta, list) or not 1 <= len(saved_theta) <= 32
            or p_requested > 32 or q_requested > 8):
        raise AnalysisError("invalid_result_state", "Use bounded complete variance coordinates and fixed/random geometry.")
    n, p0, q0, nt = result.nobs_original, p_requested, q_requested, len(saved_theta)
    plan = plan_workspace("saved scalar mixed Satterthwaite contrast", {
        "resident designs and identities": 96*n*(p0+q0+4),
        "maximal group sufficient statistics, Woodbury and derivative scratch": 128*n*(q0*q0+q0*(p0+2)+p0+2),
        "information, derivative and inverse factors": 512*(nt*nt+p0*p0),
        "output coordinates and sample positions": 64*(n+nt*nt),
    }).record()
    original, positions = matched_frame(result, data)
    setup = _Setup(ModelFrame(spec, original))
    x, p = setup.design.x, setup.p
    if p > 32 or setup.z.shape[1] > 8 or setup.n_variance + 1 > 32:
        raise AnalysisError("work_budget", "Use at most 32 fixed/variance parameters and 8 random columns.")
    terms = setup.design.terms
    if isinstance(max_work, bool) or not isinstance(max_work, int) or max_work < 1:
        raise AnalysisError("invalid_option", "max_work must be a positive integer.")
    planned_work = 64*(setup.n_variance+1)*(setup.n_top*(p+setup.z.shape[1])**3+p**3)
    if planned_work > max_work:
        raise AnalysisError("work_budget", "The variance derivative plan exceeds max_work.")
    if [c.term for c in result.coefficients[:p]] != terms or any(k not in terms for k in contrast):
        raise AnalysisError("invalid_contrast", "Choose only stored fixed-effect terms in their saved geometry.")
    try:
        theta = torch.tensor(saved_theta, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_result_state", "Variance coordinates must be numeric.") from exc
    if theta.shape != (setup.n_variance + 1,) or not torch.isfinite(theta).all():
        raise AnalysisError("invalid_result_state", "The complete saved variance coordinates are required.")
    if result.extra.get("method") != ("reml" if setup.reml else "ml"):
        raise AnalysisError("invalid_result_state", "Saved likelihood criterion disagrees with the specification.")
    c = torch.tensor([contrast.get(name, 0.0) for name in terms], dtype=torch.float64)
    like = _likelihood(setup, x)

    def variance(point):
        fitted = like.evaluate(point)
        covariance = optimize.information_inverse(fitted.xvx / fitted.sigma2)
        return c @ covariance @ c

    try:
        v = variance(theta)
        hessian = optimize.numerical_hessian(like.gradient, theta)
        theta_covariance = optimize.information_inverse(-hessian)
        gradient = optimize.numerical_gradient(variance, theta)
    except KernelError as exc:
        raise AnalysisError("numerical_failure", str(exc)) from exc
    try:
        full_cov = torch.tensor(result.covariance_matrix, dtype=torch.float64)
        saved_se = torch.tensor(result.extra.get("theta_std_error", []), dtype=torch.float64)
        fixed_se = torch.tensor([item.std_error for item in result.coefficients[:p]], dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_result_state", "Saved covariance and standard errors must be numeric.") from exc
    evaluated = like.evaluate(theta)
    fixed_cov = optimize.information_inverse(evaluated.xvx / evaluated.sigma2)
    if (full_cov.shape != (len(result.coefficients), len(result.coefficients))
            or not torch.isfinite(full_cov).all() or not torch.isfinite(fixed_se).all()
            or (fixed_se < 0).any()
            or not torch.allclose(full_cov[:p, :p], fixed_cov, rtol=1e-7, atol=1e-12)
            or not torch.allclose(fixed_se.square(), fixed_cov.diag(), rtol=1e-7, atol=1e-12)):
        raise AnalysisError("invalid_result_state", "Saved fixed covariance disagrees with the saved model.")
    if (saved_se.shape != theta.shape or not torch.isfinite(saved_se).all() or (saved_se <= 0).any()
            or not torch.allclose(saved_se.square(), theta_covariance.diag(), rtol=1e-5, atol=1e-12)):
        raise AnalysisError("invalid_result_state", "Saved variance-information diagonal disagrees with the model.")
    variability = gradient @ theta_covariance @ gradient
    if not torch.isfinite(v) or v <= 0 or not torch.isfinite(variability) or variability <= 0:
        raise AnalysisError("invalid_covariance", "Positive estimable contrast and variance uncertainty required.")
    df = float(2 * v.square() / variability)
    if not math.isfinite(df) or df <= 0:
        raise AnalysisError("invalid_degrees_of_freedom", "The Satterthwaite df must be finite and positive.")
    beta = torch.tensor([v.estimate for v in result.coefficients[:p]], dtype=torch.float64)
    if not torch.allclose(beta, evaluated.beta, rtol=1e-7, atol=1e-9):
        raise AnalysisError("invalid_result_state", "Saved fixed estimates disagree with the saved model.")
    estimate, se = float(c @ beta), math.sqrt(float(v))
    statistic = (estimate - null) / se
    critical = distributions.t_isf(alpha / 2, df)
    return saved_summary(TableSet(
        {"contrast": table([dict(estimate=estimate, null=float(null), std_error=se,
                                 t=statistic, df=df, p_value=2 * distributions.t_sf(abs(statistic), df),
                                 ci_low=estimate-critical*se, ci_high=estimate+critical*se)]),
         "variance_coordinates": table({"coordinate": list(range(len(theta))),
                                        "theta": theta.tolist(), "gradient": gradient.tolist()}),
         "variance_covariance": table(theta_covariance.tolist())},
        source_result_id=result.id, source_data_hash=result.provenance["data_hash"],
        source_sample_positions=positions, contrast=contrast, fixed_terms=terms,
        alpha=alpha, criterion=result.extra["method"], df_method="scalar Satterthwaite",
        variance_variability=float(variability), refitted=False,
        planned_work=planned_work, resource_plan=plan,
        assumptions="Gaussian independent top-level groups, correctly specified random covariance; interior information; model-based ML/REML approximation",
        exclusions=["Kenward-Roger", "mixedflex", "robust/cluster", "weights", "categories", "joint multi-df contrast"],
        precision="float64", device="cpu", stata_parity_validated=False,
    ))
