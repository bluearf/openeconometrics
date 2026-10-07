"""Plain publication disclosures from persisted inference, without estimation."""
from __future__ import annotations


def _number(value):
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def model_notes(model, *, number=1) -> list[str]:
    if model.spec.estimator == "ols":
        from openecon.linear_ols.publication import model_notes as ols_notes
        return ols_notes(model, number=number)
    spec, info = model.spec, model.inference
    if info.get("available") is False:
        target = model.extra.get("target") or info.get("target") or "fixed-parameter evaluation"
        details = [f"({number}) {spec.estimator.upper()}: {target}",
                   "coefficient inference unavailable; no standard errors, tests or confidence intervals are supplied",
                   f"estimation sample {model.nobs} of {model.nobs_original} observations"]
        notes = [note for container in (info, model.extra)
                 if isinstance(container.get("notes"), (list, tuple))
                 for note in container["notes"] if isinstance(note, str)]
        return ["; ".join(details) + ".", *(f"({number}) {note}" for note in notes),
                *(f"({number}) Warning: {warning}" for warning in model.warnings)]
    covariance = info.get("covariance") or spec.covariance
    descriptions = {
        "nonrobust": "conventional standard errors", "robust": "Huber-White robust standard errors",
        "cluster": "cluster-robust standard errors", "opg": "outer-product-of-gradients standard errors",
        "hac": "heteroskedasticity and autocorrelation consistent (HAC) standard errors",
        "driscoll_kraay": "Driscoll-Kraay standard errors",
        "bootstrap": "bootstrap standard errors", "jackknife": "jackknife standard errors",
        **{f"HC{i}": f"HC{i} heteroskedasticity-robust standard errors" for i in range(4)},
    }
    details = [f"({number}) {spec.estimator.upper()}: " + descriptions.get(covariance, str(covariance))]
    # Robust aliases can actually cluster on the panel or subject id, even when
    # spec.cluster is None. The fitted inference record takes precedence.
    columns = info.get("cluster_columns") or info.get("cluster_column") or spec.cluster
    columns = [columns] if isinstance(columns, str) else list(columns or [])
    if columns:
        counts = info.get("cluster_counts") or ([info.get("cluster_count")] if len(columns) == 1 else [])
        details.append("clustered by " + ", ".join(
            f"{column} ({counts[i]} clusters)" if i < len(counts) and counts[i] is not None
            else f"{column} (cluster count unavailable)" for i, column in enumerate(columns)))
    elif covariance == "cluster":
        details.append("cluster grouping unavailable")
    if info.get("use_t") is True:
        degrees = info.get("df_inference")
        details.append("Student t inference" + (f" (df = {_number(degrees)})" if degrees is not None
                                                else " (degrees of freedom unavailable)"))
    elif info.get("use_t") is False:
        details.append("normal z inference")
    else:
        details.append("inference distribution unavailable")
    alpha = info.get("alpha", spec.alpha)
    details.append(f"{100 * info.get('confidence_level', 1 - alpha):g}% confidence level (alpha = {alpha:g})")
    physical = model.provenance.get("sample_position_count", model.nobs_original - model.dropped_rows)
    if spec.weight_type:
        details.append(f"{spec.weight_type} weights: {spec.weights}; effective observations = {model.nobs}")
    details.append(f"{'physical ' if spec.weight_type else ''}estimation sample {physical} of {model.nobs_original} observations")
    if model.dropped_rows:
        details.append(f"{model.dropped_rows} observations excluded from estimation")
    # These are recorded scalars and method notes, not guesses from estimator
    # titles. In particular /sigma may use one-sided tests while slopes do not.
    fields = {
        "correction": "covariance correction", "small_sample_correction": "finite-sample covariance factor",
        "df_resid": "residual df", "kernel": "kernel", "lags": "lags", "bandwidth": "bandwidth",
        "periods": "periods", "effective_covariance": "effective covariance",
        "absorbed_degrees_of_freedom": "absorbed df", "degrees_of_freedom_convention": "df convention",
        "sigma": "scale-parameter inference", "reference": "reference", "r_squared_definition": "R-squared definition",
        "scheme": "resampling scheme", "reps": "requested repetitions", "successful_reps": "used repetitions",
        "failed_reps": "excluded repetitions", "seed": "seed",
    }
    for key, label in fields.items():
        value = info.get(key)
        if value is not None and isinstance(value, (str, int, float)):
            details.append(f"{label}: {_number(value)}")
    extra_fields = {
        "model": "model form", "absorbed_effects": "absorbed effects",
        "absorbed_degrees_of_freedom": "absorbed df", "family": "family", "link": "link",
        "distribution": "outcome distribution", "metric": "parameter metric", "method": "method",
        "likelihood": "likelihood", "partial_likelihood": "partial likelihood", "ties": "ties",
        "log_likelihood_scale": "log-likelihood scale", "parameterization": "parameterization",
        "alpha_test_note": "dispersion test", "vuong_note": "Vuong test",
        "excluded_lag_window_rows": "excluded lag-window rows", "missing_rows": "missing rows",
    }
    for key, label in extra_fields.items():
        value = model.extra.get(key)
        if value is not None and isinstance(value, (str, int, float)):
            details.append(f"{label}: {_number(value)}")
    extra_notes = [note for container in (info, model.extra)
                   if isinstance(container.get("notes"), (list, tuple))
                   for note in container["notes"] if isinstance(note, str)]
    return ["; ".join(details) + ".", *(f"({number}) {note}" for note in extra_notes),
            *(f"({number}) Warning: {warning}" for warning in model.warnings)]
