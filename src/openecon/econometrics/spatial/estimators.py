"""Public spatial ML estimators and publication impact tables."""

from __future__ import annotations

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame,
    TableSet,
    build_result,
    column_list,
    kernel_call,
    make_spec,
)
from openecon.models import ResultBundle

from . import kernels
from .weights import SpatialWeights, _keys


def _payload(weights):
    return SpatialWeights.from_payload(weights).to_payload()


def _estimate(
    model,
    data,
    outcome,
    predictors,
    *,
    key,
    spatial_weights,
    intercept=True,
    missing="raise",
    alpha=0.05,
    max_n=512,
    max_iterations=100,
    tolerance=1e-9,
    error_weights=None,
):
    options = {
        "spatial_weights": _payload(spatial_weights),
        "max_n": max_n,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
    }
    if error_weights is not None:
        options["error_weights"] = _payload(error_weights)
    spec = make_spec(
        model,
        outcome=outcome,
        predictors=column_list(predictors, "predictors"),
        intercept=intercept,
        missing=missing,
        alpha=alpha,
        columns={"key": key},
        options=options,
    )
    return fit_spatial(spec, data)


def sar(data, outcome, predictors, *, key, spatial_weights, **options):
    """Gaussian SAR: y=rho W y+X beta+epsilon; see spatial.md for domain."""
    return _estimate(
        "sar", data, outcome, predictors, key=key, spatial_weights=spatial_weights, **options
    )


def sem(data, outcome, predictors, *, key, spatial_weights, **options):
    """Gaussian SEM: y=X beta+u, u=lambda W u+epsilon."""
    return _estimate(
        "sem", data, outcome, predictors, key=key, spatial_weights=spatial_weights, **options
    )


def sac(data, outcome, predictors, *, key, spatial_weights, error_weights=None, **options):
    """Gaussian SAC: SAR mean and SEM innovations; separate error W is optional."""
    return _estimate(
        "sac",
        data,
        outcome,
        predictors,
        key=key,
        spatial_weights=spatial_weights,
        error_weights=error_weights,
        **options,
    )


def sdm(data, outcome, predictors, *, key, spatial_weights, **options):
    """Gaussian SDM: SAR with every numeric predictor's W lag (no lagged constant)."""
    return _estimate(
        "sdm", data, outcome, predictors, key=key, spatial_weights=spatial_weights, **options
    )


def fit_spatial(spec, data):
    frame = ModelFrame(spec, data)
    key = frame.role("key")[0]
    if key == spec.outcome or key in spec.predictors:
        raise AnalysisError(
            "invalid_spatial_keys",
            "Spatial key identity must be separate from the model's numeric outcome and predictors.",
        )
    weights = SpatialWeights.from_payload(frame.option("spatial_weights"))
    # Validate the full identity domain before applying a missing-data induced subgraph.
    weights.align(_keys(frame.original[key].tolist()))
    weights = weights.align(_keys(frame.series(key).tolist()), subset=True)
    error_weights = None
    if spec.estimator == "sac":
        payload = frame.option("error_weights")
        error_weights = weights if payload is None else SpatialWeights.from_payload(payload)
        if payload is not None:
            error_weights.align(_keys(frame.original[key].tolist()))
            error_weights = error_weights.align(_keys(frame.series(key).tolist()), subset=True)
    n = frame.n
    max_n = frame.option("max_n")
    if n > max_n:
        raise AnalysisError(
            "spatial_dense_limit",
            f"Exact dense spatial ML permits at most max_n={max_n} complete observations; supplied {n}.",
        )
    width = frame.design_width() + (len(spec.predictors) if spec.estimator == "sdm" else 0)
    p = width + len(kernels.spatial_names(spec.estimator)) + 1
    frame.workspace_plan(
        "exact dense spatial likelihood and full information",
        {
            "dense_filters_factorizations_autograd_and_multiplier": 8 * 96 * n**2,
            "design_QR_and_sample_vectors": 8 * (24 * n * max(width, 1) + 20 * n),
            "full_parameter_information": 8 * 12 * p**2,
            "sparse_weight_indices": 48
            * (weights.nnz + (error_weights.nnz if error_weights else 0)),
        },
    )
    w = weights.dense(max_n=max_n)
    we = (
        w if error_weights is None or error_weights is weights else error_weights.dense(max_n=max_n)
    )
    design = frame.design()
    terms = list(design.terms)
    if any(term in {"rho", "lambda", "ln_sigma2"} or term.startswith("W:") for term in terms):
        raise AnalysisError(
            "duplicate_terms",
            "Spatial predictor names cannot be rho, lambda, ln_sigma2 or begin with W:.",
        )
    x = design.x
    if spec.estimator == "sdm":
        # No WX intercept: this is an explicitly stated Durbin specification,
        # including for unnormalized W and isolates where W1 is not constant.
        x = torch.cat((x, w @ frame.matrix(spec.predictors)), dim=1)
        terms.extend("W:" + name for name in spec.predictors)
    y = frame.numeric(spec.outcome)
    fit = kernel_call(
        kernels.fit_ml,
        y,
        x,
        w,
        model=spec.estimator,
        w_error=we,
        max_iterations=frame.option("max_iterations"),
        tolerance=frame.option("tolerance"),
    )
    terms.extend((*kernels.spatial_names(spec.estimator), "ln_sigma2"))
    summary = weights.summary()
    summary["spectral_radius"] = float(torch.linalg.eigvals(w).abs().max())
    if summary["isolate_keys"]:
        frame.warn(
            "Zero-row isolates are retained in the estimation sample; their spatial lag is zero."
        )
    if frame.dropped_missing:
        frame.warn(
            "Missing-data filtering induced a spatial subgraph; row-normalized weights were re-normalized on retained keys."
        )
    extra = {
        "spatial_weights": weights.to_payload(),
        "spatial_summary": summary,
        "innovation_variance": float(torch.exp(fit.params[-1])),
        "innovation_residuals": fit.innovations.tolist(),
        "structural_residuals": fit.structural_residuals.tolist(),
        "model_equation": {
            "sar": "y=rho W y+X beta+epsilon",
            "sem": "y=X beta+u; u=lambda W u+epsilon",
            "sac": "y=rho W y+X beta+u; u=lambda W_error u+epsilon",
            "sdm": "y=rho W y+X beta+W X_slopes theta+epsilon",
        }[spec.estimator],
    }
    if error_weights is not None:
        extra["error_weights"] = error_weights.to_payload()
        extra["error_weight_summary"] = error_weights.summary()
        extra["error_weight_summary"]["spectral_radius"] = float(
            torch.linalg.eigvals(we).abs().max()
        )
    if spec.estimator in {"sar", "sdm", "sac"}:
        extra["impacts"] = kernel_call(
            kernels.multiplier_impacts,
            fit.params,
            fit.covariance,
            w,
            terms=terms,
            predictors=spec.predictors,
            model=spec.estimator,
            alpha=spec.alpha,
        )
    return build_result(
        frame,
        terms=terms,
        params=fit.params,
        covariance=fit.covariance,
        fitted=fit.fitted,
        df_resid=n - p,
        metrics={
            "log_likelihood": fit.log_likelihood,
            "aic": 2 * p - 2 * fit.log_likelihood,
            "bic": p * float(torch.log(torch.tensor(n, dtype=torch.float64)))
            - 2 * fit.log_likelihood,
            "sigma2": extra["innovation_variance"],
        },
        solver="profile Gaussian spatial ML; multi-start Newton; exact dense logdet",
        solver_diagnostics=fit.diagnostics,
        optimizer={
            "converged": True,
            "iterations": fit.diagnostics["iterations"],
            "max_iterations": frame.option("max_iterations"),
            "tolerance": frame.option("tolerance"),
        },
        inference={
            "correction": "inverse full observed information; Gaussian ML variance divisor N",
            "fitted_definition": "unconditional reduced-form mean (I-rho W)^-1 X_aug beta",
            "innovation_definition": "(I-lambda W_error)[(I-rho W)y-X_aug beta]",
            "weights_conditioning": "spatial W fixed and exogenous; no observation weights",
        },
        provenance={
            "spatial_weight_hash": summary["sha256"],
            "spatial_weight_hash_scope": "effective sample-aligned keyed COO W including normalization and policies",
            "spatial_execution_domain": "cross-sectional CPU exact dense likelihood; explicitly budgeted; no Dataset route",
        },
        extra=extra,
    )


def spatial_impacts(result):
    """Publication tables of persisted direct/indirect/total effects and delta CIs."""
    if not isinstance(result, ResultBundle) or result.spec.estimator not in {"sar", "sdm", "sac", "sar_iv"}:
        raise AnalysisError(
            "unsupported_spatial_impacts", "Pass a fitted SAR, SDM or SAC ResultBundle."
        )
    impact = result.extra.get("impacts")
    if not isinstance(impact, dict) or not isinstance(impact.get("rows"), list):
        raise AnalysisError(
            "invalid_result", "The fitted result has no persisted spatial impact inference."
        )
    table = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "gradient"} for row in impact["rows"]]
    )
    return TableSet({"Spatial multiplier impacts (delta method)": table})
