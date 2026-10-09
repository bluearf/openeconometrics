"""Training-only, weighted design state for bounded penalized GLM prediction.

The solver owns the intercept.  The returned matrix contains only slopes; a
categorical predictor's penalty factor applies unchanged to its complete block.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.streaming_design import numeric_values

MAX_PREDICTORS = 16
MAX_LEVELS = 16
MAX_SLOPES = 64
MAX_LABEL_BYTES = 4096


def _names(value, label, *, optional=False):
    if optional and value is None:
        return []
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip() or len(name.encode("utf-8")) > MAX_LABEL_BYTES
        for name in value
    ):
        raise AnalysisError("invalid_design", f"{label} must be a list of bounded nonempty column names.")
    result = list(value)
    if len(set(result)) != len(result):
        raise AnalysisError("invalid_design", f"{label} must not contain duplicate names.")
    return result


def _frame(df, predictors):
    if not isinstance(df, pd.DataFrame):
        raise AnalysisError("invalid_data", "Penalized GLM design requires a resident DataFrame.")
    if df.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Design data must not contain duplicate column names.")
    absent = [name for name in predictors if name not in df.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Design data is missing predictor columns: {absent}.")
    if len(df) == 0:
        raise AnalysisError("empty_sample", "Design data must contain at least one row.")


def _level(value):
    # Object columns can retain NumPy scalar objects.  Convert only scalar
    # wrappers, never dates, arrays, custom objects or arbitrary string casts.
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        raise AnalysisError("missing_data", "Categorical predictors must not contain missing values.")
    kind = {bool: "bool", int: "int", float: "float", str: "str"}.get(type(value))
    if kind is None:
        raise AnalysisError("unsupported_category", "Categorical levels must be typed JSON strings, numbers or booleans.")
    if kind == "float" and not math.isfinite(value):
        raise AnalysisError("non_finite_values", "Categorical numeric levels must be finite.")
    if kind == "str" and len(value.encode("utf-8")) > MAX_LABEL_BYTES:
        raise AnalysisError("category_metadata_too_large", "A categorical label exceeds 4096 UTF-8 bytes.")
    return {"type": kind, "value": value}


def _key(level):
    return level["type"], level["value"]


def _schema(predictors, categorical, levels, intercept):
    terms, owners, term_levels = [], [], []
    for name in predictors:
        if name not in categorical:
            terms.append(name)
            owners.append(name)
            term_levels.append(None)
            continue
        for level in levels[name][int(intercept):]:
            label = json.dumps(level["value"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            terms.append(f"C({name})[{level['type']}:{label}]")
            owners.append(name)
            term_levels.append(dict(level))
    if len(terms) > MAX_SLOPES:
        raise AnalysisError("model_too_wide", "Penalized GLM categorical expansion exceeds 64 slope columns.")
    if len(set(terms)) != len(terms) or (intercept and "Intercept" in terms):
        raise AnalysisError("duplicate_terms", "Numeric names collide with generated intercept or categorical term names.")
    return terms, owners, term_levels


def _encode(df, predictors, categorical, levels, intercept):
    columns = []
    for name in predictors:
        if name not in categorical:
            columns.append(numeric_values(df[name], name))
            continue
        keys = [_key(_level(value)) for value in df[name].to_numpy(dtype=object)]
        known = {_key(level) for level in levels[name]}
        if any(key not in known for key in keys):
            raise AnalysisError("unseen_category", f"Predictor '{name}' contains a level absent from its training schema.")
        for level in levels[name][int(intercept):]:
            target = _key(level)
            columns.append(torch.tensor([key == target for key in keys], dtype=torch.float64))
    return torch.stack(columns, dim=1) if columns else torch.empty((len(df), 0), dtype=torch.float64)


def _factors(predictors, penalty_factors, forced_controls):
    if penalty_factors is None:
        penalty_factors = {}
    if not isinstance(penalty_factors, Mapping) or any(key not in predictors for key in penalty_factors):
        raise AnalysisError("invalid_penalty_factors", "Penalty factors must map existing predictor names to finite nonnegative numbers.")
    factors = {}
    for name in predictors:
        value = penalty_factors.get(name, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalysisError("invalid_penalty_factors", "Each penalty factor must be a finite nonnegative number.")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise AnalysisError("invalid_penalty_factors", "Each penalty factor must be finite.") from exc
        if not math.isfinite(value) or value < 0:
            raise AnalysisError("invalid_penalty_factors", "Each penalty factor must be a finite nonnegative number.")
        factors[name] = value
    try:
        forced = _names(forced_controls, "forced_controls", optional=True)
    except AnalysisError as exc:
        raise AnalysisError("invalid_forced_controls", str(exc)) from exc
    if any(name not in predictors for name in forced):
        raise AnalysisError("invalid_forced_controls", "Every forced control must be an existing predictor.")
    for name in forced:
        factors[name] = 0.0
    return factors, forced


def _normalize_weights(weights, n):
    if not isinstance(weights, torch.Tensor) or weights.ndim != 1 or weights.numel() != n:
        raise AnalysisError("invalid_weights", "Design weights must be a one-dimensional Tensor aligned to training rows.")
    if weights.is_complex() or weights.dtype == torch.bool:
        raise AnalysisError("invalid_weights", "Design weights must contain positive real numbers.")
    weights = weights.to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise AnalysisError("invalid_weights", "Design weights must be finite and strictly positive.")
    scaled = weights / weights.max()
    normalized = scaled / scaled.sum()
    if not bool((normalized > 0).all()):
        raise AnalysisError("numerical_failure", "Weight ratios exceed representable float64 precision.")
    return normalized


def _standardize(x, weights, *, intercept, standardize):
    p = x.shape[1]
    if not p:
        return x, torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
    # Bounded coordinates avoid overflowing a finite weighted mean/RMS by
    # squaring a large raw predictor or summing unnormalized raw weights.
    magnitude = x.abs().amax(dim=0)
    magnitude = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    bounded = x / magnitude
    bounded_center = weights @ bounded if intercept else torch.zeros(p, dtype=torch.float64)
    center = bounded_center * magnitude
    if intercept:
        # A constant predictor must center to exact zero, including weights
        # whose floating sum differs slightly from one.
        constant = (x == x[0]).all(dim=0)
        center = torch.where(constant, x[0], center)
        bounded_center = torch.where(constant, x[0] / magnitude, bounded_center)
    residual = bounded - bounded_center
    if standardize:
        bounded_rms = (weights @ residual.square()).sqrt()
        scale = magnitude * bounded_rms
        zero = bounded_rms == 0
        scale = torch.where(zero, torch.ones_like(scale), scale)
        z = residual / torch.where(zero, 1 / magnitude, bounded_rms)
    else:
        scale = torch.ones(p, dtype=torch.float64)
        z = x - center
    if not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()) or not bool(torch.isfinite(z).all()):
        raise AnalysisError("numerical_failure", "Weighted design moments exceed finite float64 arithmetic.")
    return z, center, scale


def fit_design(df, predictors, categorical, weights, *, intercept, standardize, penalty_factors, forced_controls):
    """Fit a JSON-safe weighted schema using exactly the supplied training rows."""
    if type(intercept) is not bool or type(standardize) is not bool:
        raise AnalysisError("invalid_design", "intercept and standardize must be booleans.")
    predictors = _names(predictors, "predictors")
    categorical = _names(categorical, "categorical", optional=True)
    if len(predictors) > MAX_PREDICTORS:
        raise AnalysisError("model_too_wide", "Penalized GLM design supports at most 16 predictors.")
    if any(name not in predictors for name in categorical):
        raise AnalysisError("invalid_design", "Every categorical variable must be a predictor.")
    _frame(df, predictors)
    normalized_weights = _normalize_weights(weights, len(df))
    factors, forced = _factors(predictors, penalty_factors, forced_controls)
    levels = {}
    for name in categorical:
        observed, keys = [], set()
        for value in df[name].to_numpy(dtype=object):
            level = _level(value)
            key = _key(level)
            if key not in keys:
                keys.add(key)
                observed.append(level)
                if len(observed) > MAX_LEVELS:
                    raise AnalysisError("model_too_wide", f"Categorical predictor '{name}' exceeds 16 observed training levels.")
        levels[name] = observed
    terms, owners, term_levels = _schema(predictors, categorical, levels, intercept)
    x = _encode(df, predictors, categorical, levels, intercept)
    z, centers, scales = _standardize(x, normalized_weights, intercept=intercept, standardize=standardize)
    effective = [factors[name] for name in owners]
    state = {
        "version": 1,
        "predictors": predictors,
        "categorical": categorical,
        "levels": levels,
        "categorical_encoding": "treatment" if intercept else "one_hot",
        "terms": terms,
        "term_predictors": owners,
        "term_levels": term_levels,
        "centers": centers.tolist(),
        "scales": scales.tolist(),
        "intercept": intercept,
        "standardize": standardize,
        "penalty_factors": factors,
        "forced_controls": forced,
        "effective_factors": effective,
        "level_order": "first appearance in supplied training rows",
        "weight_moments": "positive weights divided by maximum then sum; weighted mean and population RMS",
    }
    return z, state, torch.tensor(effective, dtype=torch.float64)


def _validate_state(state):
    if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
        raise AnalysisError("invalid_prediction_state", "Unsupported or missing penalized GLM design state version.")
    try:
        predictors = _names(state.get("predictors"), "predictors")
        categorical = _names(state.get("categorical"), "categorical")
        if len(predictors) > MAX_PREDICTORS or any(name not in predictors for name in categorical):
            raise ValueError("invalid predictor schema")
        intercept, standardize = state.get("intercept"), state.get("standardize")
        if type(intercept) is not bool or type(standardize) is not bool:
            raise ValueError("invalid preprocessing flags")
        if state.get("categorical_encoding") != ("treatment" if intercept else "one_hot"):
            raise ValueError("inconsistent category encoding")
        levels = state.get("levels")
        if not isinstance(levels, dict) or set(levels) != set(categorical):
            raise ValueError("invalid category schema")
        for name in categorical:
            entries = levels[name]
            if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_LEVELS:
                raise ValueError("invalid training levels")
            keys = []
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"type", "value"} or _level(entry["value"]) != entry:
                    raise ValueError("invalid typed training level")
                keys.append(_key(entry))
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate training levels")
        terms, owners, term_levels = _schema(predictors, categorical, levels, intercept)
        if state.get("terms") != terms or state.get("term_predictors") != owners or state.get("term_levels") != term_levels:
            raise ValueError("inconsistent expanded slope schema")
        for entry in state["term_levels"]:
            if entry is not None and (not isinstance(entry, dict) or set(entry) != {"type", "value"} or _level(entry["value"]) != entry):
                raise ValueError("invalid typed slope level")
        factors, forced = _factors(predictors, state.get("penalty_factors"), state.get("forced_controls"))
        if state.get("penalty_factors") != factors or state.get("forced_controls") != forced or state.get("effective_factors") != [factors[name] for name in owners]:
            raise ValueError("inconsistent penalty factor schema")
        if not isinstance(state["effective_factors"], list) or any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in state["effective_factors"]):
            raise ValueError("invalid effective penalty factors")
        vectors = []
        for key in ("centers", "scales"):
            values = state.get(key)
            if not isinstance(values, list) or len(values) != len(terms) or any(type(value) not in {int, float} or not math.isfinite(value) for value in values):
                raise ValueError("invalid preprocessing vector")
            vectors.append(torch.tensor(values, dtype=torch.float64))
        centers, scales = vectors
        if not bool((scales > 0).all()) or (not intercept and bool((centers != 0).any())) or (not standardize and bool((scales != 1).any())):
            raise ValueError("inconsistent preprocessing vectors")
        return predictors, categorical, levels, intercept, centers, scales
    except (AnalysisError, ValueError, TypeError, KeyError, OverflowError, RuntimeError) as exc:
        raise AnalysisError("invalid_prediction_state", "Saved penalized GLM design schema is invalid or inconsistent.") from exc


def transform_design(df, state):
    """Replay the immutable training schema; reject unseen levels and invalid rows."""
    predictors, categorical, levels, intercept, centers, scales = _validate_state(state)
    _frame(df, predictors)
    x = _encode(df, predictors, categorical, levels, intercept)
    difference = x - centers
    z = difference / scales
    # Opposite signed finite extremes can overflow their difference while the
    # standardized result remains representable.  Divide first only there.
    overflow = ~torch.isfinite(difference)
    if bool(overflow.any()):
        alternative = x / scales - centers / scales
        z = torch.where(overflow, alternative, z)
    if not bool(torch.isfinite(z).all()):
        raise AnalysisError("numerical_failure", "Prediction design exceeds finite float64 arithmetic.")
    return z
