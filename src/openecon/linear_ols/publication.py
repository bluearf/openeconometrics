"""Plain, escaped-by-the-caller publication disclosures for native OLS results."""
from __future__ import annotations

import math


_LABELS = {
    "nonrobust": "conventional standard errors",
    "HC0": "HC0 heteroskedasticity-robust standard errors",
    "HC1": "HC1 heteroskedasticity-robust standard errors",
    "HC2": "HC2 heteroskedasticity-robust standard errors",
    "HC3": "HC3 heteroskedasticity-robust standard errors",
    "cluster": "cluster-robust standard errors",
    "cluster_hc2": "CR2 cluster leverage-adjusted standard errors",
    "cluster_hc3": "CR3 cluster leverage-adjusted standard errors",
    "hac": "heteroskedasticity and autocorrelation consistent (HAC) standard errors",
    "bootstrap": "bootstrap standard errors",
    "jackknife": "jackknife standard errors",
}
_WEIGHTS = {"aweight": "analytic", "fweight": "frequency",
            "pweight": "sampling", "iweight": "importance"}


def _number(value):
    return f"{value:g}" if isinstance(value, (int, float)) and math.isfinite(value) else str(value)


def model_notes(model, *, number=1) -> list[str]:
    """Describe the inference actually used; never recompute table statistics."""
    if model.spec.estimator != "ols":
        raise ValueError("OLS publication notes require an OLS result.")
    inference, spec = model.inference, model.spec
    covariance = str(inference.get("covariance", spec.covariance)).lower()
    key = covariance.upper() if covariance.startswith("hc") else covariance
    details = [f"({number}) OLS: " + _LABELS.get(key, covariance)]
    clusters = [spec.cluster] if isinstance(spec.cluster, str) else list(spec.cluster or [])
    if clusters:
        counts = inference.get("cluster_counts") or [inference.get("cluster_count")]
        grouping = [f"{name} ({counts[i]} clusters)" if i < len(counts) and counts[i] is not None
                    else f"{name} (cluster count unavailable)" for i, name in enumerate(clusters)]
        details.append("clustered by " + ", ".join(grouping))
        if covariance == "cluster" and len(clusters) > 1:
            details.append("CGM inclusion-exclusion with intersection-specific finite-sample corrections")
    if covariance == "hac":
        kernel = inference.get("kernel", spec.options.get("kernel", "bartlett"))
        details.append(f"{str(kernel).replace('_', ' ')} kernel, lags = {_number(inference.get('lags', spec.options.get('lags')))}")
        if inference.get("bandwidth") is not None:
            details.append("bandwidth = " + _number(inference["bandwidth"]))
        if inference.get("lag_selection"):
            details.append("lag selection: " + str(inference["lag_selection"]))
        if inference.get("time_gaps"):
            details.append("actual calendar gaps retained")
    if inference.get("small_sample_correction") is not None:
        details.append("finite-sample covariance factor = " + _number(inference["small_sample_correction"]))
    if covariance in {"bootstrap", "jackknife"}:
        scheme = inference.get("scheme", "unspecified")
        requested = inference.get("reps", spec.options.get("reps"))
        successful = inference.get("successful_reps", requested)
        details.append(f"{scheme} resampling, {_number(successful)} of {_number(requested)} repetitions used")
        if inference.get("failed_reps"):
            details.append(f"{inference['failed_reps']} repetitions excluded for insufficient rank or observations")
        if inference.get("seed") is not None:
            details.append("seed = " + str(inference["seed"]))
        if inference.get("frequency_count_resampling"):
            details.append("frequency-expanded observations are the resampling units")
    coefficient_df = inference.get("coefficient_df")
    if inference.get("use_t", False):
        if coefficient_df:
            details.append("Student t inference with coefficient-specific Bell-McCaffrey/Satterthwaite degrees of freedom")
            details.append("coefficient df: " + ", ".join(
                f"{term.term} = {_number(degrees)}" for term, degrees in zip(model.coefficients, coefficient_df, strict=True)))
        else:
            degrees = inference.get("df_inference")
            details.append("Student t inference" + (f" (df = {_number(degrees)})" if degrees is not None else ""))
    else:
        details.append("normal z inference")
    if inference.get("hansen"):
        scales = inference.get("coefficient_scale")
        details.append("Hansen (2025) scaling multiplies t statistics and divides confidence interval widths; printed standard errors are unchanged")
        if scales:
            details.append("coefficient scales: " + ", ".join(
                f"{term.term} = {_number(scale)}" for term, scale in zip(model.coefficients, scales, strict=True)))
    weight_type = spec.weight_type
    if weight_type:
        description = _WEIGHTS.get(weight_type, weight_type)
        normalization = ("normalized to mean one" if weight_type in {"aweight", "pweight"}
                         or weight_type == "iweight" and covariance != "nonrobust" else "raw weights")
        details.append(f"{description} weights: {spec.weights} ({normalization})")
        details.append(f"effective observations = {model.nobs}")
    alpha = inference.get("alpha", spec.alpha)
    details.append(f"{100 * inference.get('confidence_level', 1 - alpha):g}% confidence level (alpha = {alpha:g})")
    physical = model.provenance.get("sample_position_count", model.nobs_original - model.dropped_rows)
    details.append(f"{'physical ' if weight_type else ''}estimation sample {physical} of {model.nobs_original} observations")
    if model.dropped_rows:
        details.append(f"{model.dropped_rows} observations excluded from estimation")
    if not spec.intercept:
        details.append("no intercept; R-squared is uncentered")
    test = model.tests.get("model")
    if test and test.get("distribution") == "F" and test.get("statistic") is not None:
        details.append(f"model F({_number(test['df'])}, {_number(test['df2'])}) = {_number(test['statistic'])}, p = {_number(test['p_value'])}")
    elif test and test.get("distribution") == "chi2" and test.get("statistic") is not None:
        details.append(f"model Wald chi-square({_number(test['df'])}) = {_number(test['statistic'])}, p = {_number(test['p_value'])}")
    return ["; ".join(details) + ".", *(f"({number}) Warning: {warning}" for warning in model.warnings)]
