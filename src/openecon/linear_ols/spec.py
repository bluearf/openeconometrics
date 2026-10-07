"""Lightweight OLS contracts; importing capability metadata never loads Torch."""
from __future__ import annotations

import math

COVARIANCES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster",
               "cluster_hc2", "cluster_hc3", "hac", "bootstrap", "jackknife")
WEIGHTS = ("aweight", "fweight", "pweight", "iweight")
KERNELS = ("bartlett", "parzen", "quadratic_spectral", "truncated")
OPTIONS = {"terms", "lags", "kernel", "reps", "seed", "dfadjust", "hansen"}


def validate_spec(spec) -> str:
    """Validate statistical choices before data access, including direct ModelSpec use."""
    covariance = spec.covariance
    if covariance == "robust":
        covariance = "HC1"
    clusters = [spec.cluster] if isinstance(spec.cluster, str) else list(spec.cluster or [])
    if len(clusters) != len(set(clusters)) or len(clusters) > 4:
        raise ValueError("Specify up to four distinct cluster columns.")
    if covariance is None:
        covariance = "cluster" if clusters else ("HC1" if spec.weight_type == "pweight" else "nonrobust")
    if covariance not in COVARIANCES:
        raise ValueError(f"OLS covariance must be one of {', '.join(COVARIANCES)}.")
    clustered = covariance in {"cluster", "cluster_hc2", "cluster_hc3"}
    if clustered and not clusters:
        raise ValueError("Cluster covariance requires cluster columns.")
    if clusters and not (clustered or covariance in {"bootstrap", "jackknife"}):
        raise ValueError("Cluster columns apply to cluster covariance or cluster resampling.")
    if covariance in {"cluster_hc2", "cluster_hc3"} and len(clusters) != 1:
        raise ValueError("Cluster HC2/HC3 require one cluster column.")
    if getattr(spec, "weights", None) is None and getattr(spec, "weight_type", None) is not None:
        raise ValueError("weight_type requires a weight column.")
    if getattr(spec, "weights", None) is not None and getattr(spec, "weight_type", None) not in WEIGHTS:
        raise ValueError("A weight column requires weight_type=aweight, fweight, pweight or iweight.")
    if getattr(spec, "weight_type", None) == "pweight" and covariance == "nonrobust":
        raise ValueError("Sampling weights require robust or cluster covariance.")
    if getattr(spec, "panel", None) is not None:
        raise ValueError("OLS uses time or cluster columns; panel estimators have separate specifications.")
    options = getattr(spec, "options", {})
    if set(options) - OPTIONS:
        raise ValueError(f"Unknown OLS option(s): {', '.join(sorted(set(options) - OPTIONS))}.")
    if "terms" in options and (not isinstance(options["terms"], list)
                              or not all(isinstance(term, str) and term.strip() for term in options["terms"])):
        raise ValueError("OLS design terms must be a list of nonempty expressions.")
    if covariance == "hac":
        lags = options.get("lags")
        if lags != "auto" and (isinstance(lags, bool) or not isinstance(lags, int) or lags < 0):
            raise ValueError("HAC requires nonnegative lags, or lags='auto'.")
        if options.get("kernel", "bartlett") not in KERNELS:
            raise ValueError(f"HAC kernel must be one of {', '.join(KERNELS)}.")
    elif "lags" in options or "kernel" in options:
        raise ValueError("lags and kernel apply only to HAC covariance.")
    if covariance == "bootstrap":
        reps = options.get("reps", 199)
        if isinstance(reps, bool) or not isinstance(reps, int) or not 2 <= reps <= 100_000:
            raise ValueError("Bootstrap reps must be an integer between 2 and 100,000.")
    if covariance not in {"bootstrap", "jackknife"} and ("reps" in options or "seed" in options):
        raise ValueError("reps and seed apply only to resampling covariance.")
    if "seed" in options and (isinstance(options["seed"], bool) or not isinstance(options["seed"], int)
                              or not 0 <= options["seed"] < 2**63):
        raise ValueError("seed must be an integer between zero and 2**63-1.")
    for flag in ("dfadjust", "hansen"):
        if flag in options and not isinstance(options[flag], bool):
            raise ValueError(f"{flag} must be a boolean.")
    if options.get("dfadjust") and covariance not in {"HC2", "HC3", "cluster_hc2", "cluster_hc3"}:
        raise ValueError("dfadjust applies to HC2/HC3 or cluster HC2/HC3.")
    if options.get("hansen") and covariance not in {"HC3", "cluster_hc3"}:
        raise ValueError("hansen applies to HC3 or cluster HC3.")
    if not math.isfinite(spec.alpha) or not 0 < spec.alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    if not getattr(spec, "predictors", []) and not spec.intercept:
        raise ValueError("OLS without an intercept needs at least one predictor.")
    if spec.outcome in spec.predictors or spec.outcome in clusters:
        raise ValueError("The outcome cannot be a predictor or a cluster column.")
    if not set(spec.categorical).issubset(spec.predictors):
        raise ValueError("Every categorical column must be a predictor.")
    return covariance


def describe() -> dict:
    return {
        "covariances": list(COVARIANCES), "default_covariance": "nonrobust",
        "weights": list(WEIGHTS), "cluster_dimensions": 4,
        "formula": True, "collinearity": "omit", "default_missing": "drop",
        "postestimation": ["test", "testparm", "lincom", "nlcom", "predict", "iter_predict",
                           "iter_influence", "margins", "vif", "hettest", "white_test",
                           "reset_test", "breusch_godfrey", "diagnostics"],
        "stata_parity_validated": False,
    }
