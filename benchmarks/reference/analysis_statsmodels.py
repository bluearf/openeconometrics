"""Validated statsmodels adapters with explicit sample and inference semantics."""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from scipy.optimize import linprog
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from .models import Coefficient, ModelSpec, ResultBundle


class AnalysisError(ValueError):
    """An actionable failure that clients can render without parsing messages."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


_SUPPORTED_COVARIANCES = {
    "ols": ("nonrobust", "HC1", "HC3", "cluster"),
    "logit": ("nonrobust", "cluster"),
    "probit": ("nonrobust", "cluster"),
}
_MAX_BINARY_ITERATIONS = 100
_MAX_DESIGN_BYTES = 256 * 1024 * 1024


def capabilities() -> dict[str, Any]:
    """Describe implemented capabilities, without implying Stata parity."""
    return {
        "schema_version": "1",
        "estimators": {
            name: {
                "covariances": list(covariances),
                "binary_outcome": name != "ols",
                "categorical_predictors": True,
                "intercept": True,
                "missing": ["raise", "drop"],
                "inference": "Student t" if name == "ols" else "Normal z",
            }
            for name, covariances in _SUPPORTED_COVARIANCES.items()
        },
        "cluster_dimensions": 1,
        "weights": False,
        "fixed_effect_absorption": False,
        "out_of_core_estimation": False,
        "stata_parity_validated": False,
        "max_chart_predictions": 400,
        "max_design_matrix_bytes": _MAX_DESIGN_BYTES,
        "limitations": [
            "Logit and probit support nonrobust and one-way cluster covariance only.",
            "Categorical predictors use explicit treatment coding with the first level omitted.",
            "Singular designs, constant columns, and separated binary models are rejected.",
            "Estimation materializes a float64 design matrix in memory.",
            "Cluster inference uses CR1; few clusters can make inference unreliable.",
        ],
    }


def _json_scalar(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise AnalysisError(
        "unsupported_category",
        "Categorical levels must be finite strings, numbers, or booleans.",
    )


def _numeric(series: pd.Series, name: str) -> np.ndarray:
    if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
        raise AnalysisError(
            "non_numeric_column",
            f"Column '{name}' must be numeric; declare a categorical predictor explicitly.",
        )
    if is_complex_dtype(series.dtype):
        raise AnalysisError("complex_values", f"Column '{name}' must contain real numbers.")
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all():
        raise AnalysisError("non_finite_values", f"Column '{name}' contains non-finite values.")
    return values


def _hash_frame(frame: pd.DataFrame, positions: np.ndarray | None = None) -> str:
    """Hash model inputs in row order; the pandas version is recorded alongside it."""
    digest = hashlib.sha256()
    schema: list[dict[str, Any]] = []
    for name in frame.columns:
        item: dict[str, Any] = {"name": name, "dtype": str(frame[name].dtype)}
        if isinstance(frame[name].dtype, pd.CategoricalDtype):
            item["categories"] = [_json_scalar(v) for v in frame[name].cat.categories]
            item["ordered"] = frame[name].cat.ordered
        schema.append(item)
    digest.update(json.dumps(schema, ensure_ascii=False, sort_keys=True).encode())
    try:
        hashed = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("unsupported_values", "Model inputs must contain scalar values.") from exc
    digest.update(hashed.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    if positions is not None:
        digest.update(positions.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _prepare_data(
    spec: ModelSpec, data: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError("invalid_data", "Data must be a pandas DataFrame.")
    if data.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    if not len(data):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    columns = list(dict.fromkeys([spec.outcome, *spec.predictors, *([spec.cluster] if spec.cluster else [])]))
    absent = [name for name in columns if name not in data.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    original = data.loc[:, columns].reset_index(drop=True)
    missing = original.isna().any(axis=1).to_numpy()
    if missing.any() and spec.missing == "raise":
        raise AnalysisError(
            "missing_values",
            f"{int(missing.sum())} observation(s) contain missing model inputs. Choose missing='drop' explicitly to exclude them.",
        )
    positions = np.flatnonzero(~missing)
    sample = original.iloc[positions].reset_index(drop=True)
    if not len(sample):
        raise AnalysisError("empty_sample", "No complete observations remain for estimation.")
    y = _numeric(sample[spec.outcome], spec.outcome)
    if np.unique(y).size < 2:
        raise AnalysisError("constant_outcome", "The outcome must contain at least two distinct values.")
    if spec.estimator != "ols" and not np.isin(y, [0.0, 1.0]).all():
        raise AnalysisError("invalid_binary_outcome", "Logit and probit require an outcome containing both 0 and 1 (or booleans).")

    pieces: list[pd.DataFrame] = []
    category_metadata: dict[str, Any] = {}
    design_columns = int(spec.intercept) + len(spec.predictors) - len(spec.categorical)
    if len(sample) * design_columns * 8 > _MAX_DESIGN_BYTES:
        raise AnalysisError("design_too_large", "The alpha limits materialized float64 design matrices to 256 MiB. Reduce predictors or use an explicit smaller dataset.")
    if spec.intercept:
        pieces.append(pd.DataFrame({"Intercept": np.ones(len(sample))}))
    for name in spec.predictors:
        values = sample[name]
        if name in spec.categorical:
            original_values = original[name]
            if isinstance(original_values.dtype, pd.CategoricalDtype):
                levels = list(original_values.cat.categories)
                ordered = bool(original_values.cat.ordered)
            else:
                try:
                    levels = sorted(original_values.dropna().unique().tolist())
                except (TypeError, ValueError) as exc:
                    raise AnalysisError(
                        "ambiguous_categories", f"Column '{name}' has mixed or non-scalar category types; use a pandas Categorical with explicit levels."
                    ) from exc
                ordered = False
            safe_levels = [_json_scalar(value) for value in levels]
            if len(levels) < 2 or values.nunique() < 2:
                raise AnalysisError("constant_predictor", f"Categorical predictor '{name}' has fewer than two observed levels.")
            design_columns += len(levels) - 1
            if len(sample) * design_columns * 8 > _MAX_DESIGN_BYTES:
                raise AnalysisError("design_too_large", f"Categorical expansion of '{name}' exceeds the alpha's 256 MiB design-matrix limit. Reduce levels or choose another model.")
            encoded = pd.get_dummies(
                pd.Series(pd.Categorical(values, categories=levels, ordered=ordered)),
                drop_first=True,
                dtype=np.float64,
            )
            encoded.columns = [f"{name}[{value}]" for value in safe_levels[1:]]
            pieces.append(encoded)
            category_metadata[name] = {
                "levels": safe_levels,
                "reference": safe_levels[0],
                "ordered": ordered,
                "coding": "treatment_drop_first",
                "levels_source": "original_model_inputs_before_missing_filter",
            }
        else:
            numeric = _numeric(values, name)
            if np.unique(numeric).size < 2:
                raise AnalysisError("constant_predictor", f"Predictor '{name}' is constant; use the intercept option instead.")
            pieces.append(pd.DataFrame({name: numeric}))
    design = pd.concat(pieces, axis=1)
    if design.columns.has_duplicates:
        raise AnalysisError("duplicate_terms", "Predictor names collide with generated design terms. Rename the affected columns.")
    x = design.to_numpy(dtype=np.float64)
    n, k = x.shape
    if n <= k:
        raise AnalysisError("insufficient_observations", f"Estimation requires more observations ({n}) than design columns ({k}).")
    # Scale for rank diagnostics only; estimates retain the original units.
    scales = np.max(np.abs(x), axis=0)
    if np.any(scales == 0):
        raise AnalysisError("singular_design", "The design contains an unobserved category or an all-zero column.")
    scaled = x / scales
    if np.linalg.matrix_rank(scaled) < k:
        raise AnalysisError("singular_design", "The design matrix is rank deficient. Remove collinear predictors or empty category levels.")
    return original, positions, y, design, category_metadata


def _check_binary_separation(y: np.ndarray, x: np.ndarray) -> None:
    """Detect complete and quasi separation through a bounded linear program."""
    scaled = x / np.maximum(np.max(np.abs(x), axis=0), np.finfo(float).tiny)
    signed = (2.0 * y - 1.0)[:, None] * scaled
    # A feasible beta with all signed margins nonnegative and a positive total
    # margin is a separating direction. Bounds remove the arbitrary scale.
    check = linprog(
        -signed.mean(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(len(y)),
        bounds=[(-1.0, 1.0)] * x.shape[1],
        method="highs",
    )
    if not check.success:
        raise AnalysisError("separation_check_failed", "The binary-model separation diagnostic did not complete successfully.")
    if -float(check.fun) > 1e-8:
        raise AnalysisError(
            "separation_detected",
            "Complete or quasi-complete separation was detected. A finite unpenalized binary-model estimate does not exist.",
        )


def _safe_metric(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def fit(spec: ModelSpec, *, data: pd.DataFrame) -> ResultBundle:
    """Fit one validated model with no silent sample or method substitutions."""
    if not isinstance(spec, ModelSpec):
        raise AnalysisError("invalid_spec", "Spec must be a validated ModelSpec instance.")
    # Revalidate mutated instances as well as objects constructed through clients.
    spec = ModelSpec.model_validate(spec.model_dump())
    if spec.covariance not in _SUPPORTED_COVARIANCES[spec.estimator]:
        raise AnalysisError(
            "unsupported_covariance",
            f"{spec.estimator} supports only {', '.join(_SUPPORTED_COVARIANCES[spec.estimator])} covariance in this alpha.",
        )
    original, positions, y, design, categories = _prepare_data(spec, data)
    x = design.to_numpy(dtype=np.float64)
    n, k = x.shape
    recorded_warnings: list[str] = []
    if len(original) > n:
        recorded_warnings.append(f"Excluded {len(original) - n} observation(s) with missing model inputs.")
    condition = float(np.linalg.cond(x / np.max(np.abs(x), axis=0)))
    if condition > 1e8:
        recorded_warnings.append("The scaled design is ill-conditioned; coefficient and inference sensitivity should be reviewed.")
    group_count: int | None = None
    cov_kwds: dict[str, Any] = {}
    correction_factor: float | None = None
    if spec.covariance == "cluster":
        groups = original.iloc[positions][spec.cluster]
        if is_numeric_dtype(groups.dtype):
            _numeric(groups, spec.cluster)
        try:
            codes, unique_groups = pd.factorize(groups, sort=False)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_clusters", "Cluster labels must be scalar values.") from exc
        group_count = len(unique_groups)
        if group_count < 2:
            raise AnalysisError("insufficient_clusters", "Cluster covariance requires at least two observed clusters.")
        cov_kwds = {"groups": codes, "use_correction": True, "df_correction": True}
        correction_factor = group_count / (group_count - 1) * (n - 1) / (n - k)
        if group_count < 30:
            recorded_warnings.append(f"Only {group_count} clusters are available; cluster-robust inference may be unreliable.")
    elif spec.covariance == "HC1":
        correction_factor = n / (n - k)

    if spec.estimator != "ols":
        _check_binary_separation(y, x)
    use_t = spec.estimator == "ols"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", HessianInversionWarning)
            if spec.estimator == "ols":
                model = sm.OLS(y, design, missing="raise", hasconst=spec.intercept)
                result = model.fit(cov_type=spec.covariance, cov_kwds=cov_kwds, use_t=use_t)
                if model.rank != k:
                    raise AnalysisError("singular_design", "The estimator found a numerically rank-deficient design. Rescale or remove problematic predictors.")
            else:
                model_class = sm.Logit if spec.estimator == "logit" else sm.Probit
                model = model_class(y, design, missing="raise", check_rank=True)
                fit_covariance_options = {"cov_kwds": cov_kwds} if cov_kwds else {}
                result = model.fit(
                    method="newton", maxiter=_MAX_BINARY_ITERATIONS, disp=False,
                    full_output=True, warn_convergence=True, tol=1e-8,
                    cov_type=spec.covariance, use_t=use_t, **fit_covariance_options,
                )
                if not result.mle_retvals.get("converged", False):
                    raise AnalysisError("nonconvergence", "The optimizer did not converge; no valid result was produced.")
            covariance = np.asarray(result.cov_params(), dtype=np.float64)
            parameters = np.asarray(result.params, dtype=np.float64)
            standard_errors = np.asarray(result.bse, dtype=np.float64)
            statistics = np.asarray(result.tvalues, dtype=np.float64)
            p_values = np.asarray(result.pvalues, dtype=np.float64)
            intervals = np.asarray(result.conf_int(alpha=spec.alpha), dtype=np.float64)
            fitted = np.asarray(result.predict(), dtype=np.float64)
            metric_values = {
                "r_squared": result.rsquared if spec.estimator == "ols" else None,
                "adjusted_r_squared": result.rsquared_adj if spec.estimator == "ols" else None,
                "pseudo_r_squared": result.prsquared if spec.estimator != "ols" else None,
                "aic": result.aic,
                "bic": result.bic,
                "log_likelihood": result.llf,
                "rmse": np.sqrt(np.mean((y - fitted) ** 2)),
                "df_resid": n - k,
                "condition_number_scaled": condition,
            }
            for item in caught:
                recorded_warnings.append(f"{item.category.__name__}: {item.message}")
    except (PerfectSeparationError, PerfectSeparationWarning) as exc:
        raise AnalysisError("separation_detected", "Separation prevents a finite unpenalized binary-model estimate.") from exc
    except ConvergenceWarning as exc:
        raise AnalysisError("nonconvergence", "The optimizer did not converge; no valid result was produced.") from exc
    except (HessianInversionWarning, np.linalg.LinAlgError) as exc:
        raise AnalysisError("singular_information", "The model information or covariance matrix is singular.") from exc
    except (FloatingPointError, OverflowError) as exc:
        raise AnalysisError("numerical_failure", "Numerical overflow prevented a valid model fit.") from exc

    arrays = [parameters, covariance, standard_errors, statistics, p_values, intervals, fitted]
    if any(not np.isfinite(array).all() for array in arrays):
        raise AnalysisError("non_finite_result", "The estimator produced non-finite coefficients, inference, or predictions.")
    if np.any(standard_errors <= 0):
        raise AnalysisError("invalid_covariance", "The fitted covariance does not provide strictly positive standard errors.")
    if spec.estimator != "ols" and np.any((fitted < 0) | (fitted > 1)):
        raise AnalysisError("invalid_predictions", "The binary estimator produced predictions outside [0, 1].")
    sample = original.iloc[positions].reset_index(drop=True)
    try:
        package_version = version("openecon")
    except PackageNotFoundError:
        package_version = "development"
    chart_rows = np.linspace(0, n - 1, num=min(n, 400), dtype=int)
    coefficients = [
        Coefficient(
            term=str(term), estimate=float(parameters[i]), std_error=float(standard_errors[i]),
            statistic=float(statistics[i]), p_value=float(p_values[i]),
            ci_low=float(intervals[i, 0]), ci_high=float(intervals[i, 1]),
        )
        for i, term in enumerate(design.columns)
    ]
    inference_df = (group_count - 1 if group_count is not None else n - k) if use_t else None
    return ResultBundle(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), spec=spec,
        nobs=n, nobs_original=len(original), dropped_rows=len(original) - n,
        coefficients=coefficients, covariance_matrix=covariance.tolist(),
        metrics={name: _safe_metric(value) for name, value in metric_values.items()},
        warnings=list(dict.fromkeys(recorded_warnings)),
        predictions=[
            {"row": int(positions[i]), "observed": float(y[i]), "fitted": float(fitted[i]), "residual": float(y[i] - fitted[i])}
            for i in chart_rows
        ],
        sample_positions=positions.tolist(),
        provenance={
            "schema_version": "1", "backend": "statsmodels", "estimator": spec.estimator,
            "versions": {
                "openecon": package_version, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            },
            "data_hash": _hash_frame(original), "sample_hash": _hash_frame(sample, positions),
            "hash_algorithm": "sha256 over schema and pandas row hashes (pandas version recorded)",
            "hash_scope": "model-input columns in positional row order; index labels excluded",
            "input_columns": list(original.columns), "design_terms": list(design.columns),
            "categorical_encoding": categories, "precision": "float64",
            "sample_position_base": 0, "prediction_sample": "evenly spaced positional observations",
            "prediction_limit": 400, "stata_parity_validated": False,
            "optimizer": None if spec.estimator == "ols" else {"method": "newton", "maxiter": _MAX_BINARY_ITERATIONS, "tolerance": 1e-8, "converged": True},
        },
        inference={
            "covariance": spec.covariance, "use_t": use_t,
            "distribution": "t" if use_t else "normal", "alpha": spec.alpha,
            "confidence_level": 1 - spec.alpha, "df_resid": n - k,
            "df_inference": inference_df, "cluster_count": group_count,
            "cluster_df": group_count - 1 if group_count is not None else None,
            "cluster_column": spec.cluster, "small_sample_correction": correction_factor,
            "correction": "CR1: G/(G-1) * (N-1)/(N-K)" if group_count else ("HC1: N/(N-K)" if spec.covariance == "HC1" else spec.covariance),
            "intercept": spec.intercept, "n_parameters": k,
            "residual_definition": "observed minus fitted response",
        },
    )
