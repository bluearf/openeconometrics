"""Framework-independent analysis API backed by one float64 tensor core."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from array import array
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from typing import Any
from uuid import uuid4

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from .engines.contracts import KernelError
from .engines.inference import critical_value, two_sided_p_values
from .engines.torch_engine import solve
from .models import Coefficient, ModelSpec, ResultBundle
from .analysis_contracts import (
    AnalysisError, _MAX_DESIGN_BYTES, _SUPPORTED_COVARIANCES, capabilities as capabilities,
)

_MAX_BINARY_ITERATIONS = 100


def _json_scalar(value: Any) -> str | int | float | bool:
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    raise AnalysisError(
        "unsupported_category", "Categorical levels must be finite strings, numbers, or booleans."
    )


def _coerce_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, Mapping) or (
        isinstance(data, Sequence) and not isinstance(data, (str, bytes))
        and all(isinstance(row, Mapping) for row in data)
    ):
        try:
            return pd.DataFrame(data)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_data", "Data columns must have consistent lengths and scalar observations.") from exc
    raise AnalysisError("invalid_data", "Data must be a DataFrame, a mapping of columns, or a list of row records.")


def _numeric(series: pd.Series, name: str) -> torch.Tensor:
    if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
        raise AnalysisError(
            "non_numeric_column",
            f"Column '{name}' must be numeric; declare a categorical predictor explicitly.",
        )
    if is_complex_dtype(series.dtype):
        raise AnalysisError("complex_values", f"Column '{name}' must contain real numbers.")
    # A zero-copy bridge where pandas storage allows it. All numerical work
    # after this boundary uses tensors; pandas retains ownership of tabular I/O.
    buffer = series.to_numpy(dtype="float64", na_value=float("nan"))
    if not buffer.flags.writeable:
        # Arrow can expose read-only buffers. Give Torch owned writable storage;
        # streaming sources bound this copy to one projected batch.
        buffer = buffer.copy()
    values = torch.as_tensor(buffer, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains non-finite values.")
    return values


def _frame_hasher(frame: pd.DataFrame):
    """Hash model inputs in row order, retaining the previous hash contract."""
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
    return digest


def _position_bytes(positions: list[int]) -> bytes:
    packed = array("q", positions)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _hash_frame(frame: pd.DataFrame, positions: list[int] | None = None) -> str:
    digest = _frame_hasher(frame)
    if positions is not None:
        digest.update(_position_bytes(positions))
    return digest.hexdigest()


def _prepare_data(spec: ModelSpec, data: Any) -> tuple[pd.DataFrame, list[int], torch.Tensor, torch.Tensor, list[str], dict[str, Any]]:
    data = _coerce_frame(data)
    if data.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    if not len(data):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    columns = list(dict.fromkeys([spec.outcome, *spec.predictors, *([spec.cluster] if spec.cluster else [])]))
    absent = [name for name in columns if name not in data.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    original = data.loc[:, columns].reset_index(drop=True)
    missing = original.isna().any(axis=1)
    missing_count = int(missing.sum())
    if missing_count and spec.missing == "raise":
        raise AnalysisError(
            "missing_values",
            f"{missing_count} observation(s) contain missing model inputs. Choose missing='drop' explicitly to exclude them.",
        )
    positions = original.index[~missing].tolist()
    sample = original.iloc[positions].reset_index(drop=True) if missing_count else original
    if not len(sample):
        raise AnalysisError("empty_sample", "No complete observations remain for estimation.")
    y = _numeric(sample[spec.outcome], spec.outcome)
    if not bool(torch.any(y != y[0])):
        raise AnalysisError("constant_outcome", "The outcome must contain at least two distinct values.")
    if spec.estimator != "ols" and not bool(((y == 0) | (y == 1)).all()):
        raise AnalysisError("invalid_binary_outcome", "Logit and probit require an outcome containing both 0 and 1 (or booleans).")
    pieces: list[torch.Tensor] = []
    terms: list[str] = []
    categories: dict[str, Any] = {}
    design_columns = int(spec.intercept) + len(spec.predictors) - len(spec.categorical)
    if len(sample) * design_columns * 8 > _MAX_DESIGN_BYTES:
        raise AnalysisError("design_too_large", "The alpha limits materialized float64 design matrices to 256 MiB. Reduce predictors or use an explicit smaller dataset.")
    if spec.intercept:
        pieces.append(torch.ones((len(sample), 1), dtype=torch.float64))
        terms.append("Intercept")
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
                    raise AnalysisError("ambiguous_categories", f"Column '{name}' has mixed or non-scalar category types; use a pandas Categorical with explicit levels.") from exc
                ordered = False
            safe_levels = [_json_scalar(value) for value in levels]
            if len(levels) < 2 or values.nunique() < 2:
                raise AnalysisError("constant_predictor", f"Categorical predictor '{name}' has fewer than two observed levels.")
            design_columns += len(levels) - 1
            if len(sample) * design_columns * 8 > _MAX_DESIGN_BYTES:
                raise AnalysisError("design_too_large", f"Categorical expansion of '{name}' exceeds the alpha's 256 MiB design-matrix limit. Reduce levels or choose another model.")
            encoded = pd.get_dummies(pd.Series(pd.Categorical(values, categories=levels, ordered=ordered)), drop_first=True, dtype="float64")
            pieces.append(torch.as_tensor(encoded.to_numpy(), dtype=torch.float64))
            terms.extend(f"{name}[{value}]" for value in safe_levels[1:])
            categories[name] = {
                "levels": safe_levels, "reference": safe_levels[0], "ordered": ordered,
                "coding": "treatment_drop_first", "levels_source": "original_model_inputs_before_missing_filter",
            }
        else:
            numeric = _numeric(values, name)
            if not bool(torch.any(numeric != numeric[0])):
                raise AnalysisError("constant_predictor", f"Predictor '{name}' is constant; use the intercept option instead.")
            pieces.append(numeric[:, None])
            terms.append(name)
    if len(terms) != len(set(terms)):
        raise AnalysisError("duplicate_terms", "Predictor names collide with generated design terms. Rename the affected columns.")
    x = torch.cat(pieces, dim=1)
    n, k = x.shape
    if n <= k:
        raise AnalysisError("insufficient_observations", f"Estimation requires more observations ({n}) than design columns ({k}).")
    return original, positions, y, x, terms, categories


def _check_binary_separation(y: torch.Tensor, x: torch.Tensor, *, intercept: bool = True) -> None:
    """Native Torch dual bounds and globally replayed separation witnesses."""
    from .engines.separation import certify_separation

    with torch.no_grad(), torch.device("cpu"):
        scaled = x / x.abs().amax(dim=0).clamp_min(torch.finfo(torch.float64).tiny)
        if intercept:
            scaled[:, 1:] -= scaled[:, 1:].mean(dim=0)
        scale = scaled.square().mean(dim=0).sqrt()
        scaled /= torch.where(scale > 0, scale, torch.ones_like(scale))
        # Signed rows are constructed per replay block, not as a full extra copy.
        objective = scaled.T @ (2 * y - 1) / len(y)

        def replay():
            for start in range(0, len(y), 65536):
                yield (2 * y[start:start + 65536] - 1)[:, None] * scaled[start:start + 65536]

        try:
            certify_separation(replay, objective, x.shape[1])
        except KernelError as error:
            raise AnalysisError(error.code, str(error)) from error


def _safe_metric(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def fit(spec: ModelSpec, *, data: Any) -> ResultBundle:
    """Fit a statistical specification from a DataFrame, columns, or row records.

    The public API contains statistical choices only. OpenEconometrics executes its own
    float64 OLS and likelihood solvers on a shared tensor core with bounded
    replay adapters for supported estimators.
    """
    if not isinstance(spec, ModelSpec):
        raise AnalysisError("invalid_spec", "Spec must be a validated ModelSpec instance.")
    spec = ModelSpec.model_validate(spec.model_dump())
    if spec.estimator == "ols":
        from openecon.linear_ols import fit_ols
        return fit_ols(spec, data=data)
    from openecon.econometrics import registry
    estimator = registry.get(spec.estimator)
    from openecon.dataset import Dataset
    if not estimator.legacy:
        from openecon.econometrics import streaming_registry
        if isinstance(data, Dataset):
            return streaming_registry.fit(spec, data)
        if streaming_registry.supports_spec(spec):
            frame = _coerce_frame(data)
            if len(frame) > 100_000:
                return streaming_registry.fit(spec, Dataset.from_frame(frame))
        # Registry estimators own their sample, kernel and result assembly.
        return registry.load_entry(estimator)(spec, data)
    if spec.covariance not in _SUPPORTED_COVARIANCES[spec.estimator]:
        raise AnalysisError("unsupported_covariance", f"{spec.estimator} supports only {', '.join(_SUPPORTED_COVARIANCES[spec.estimator])} covariance in this alpha.")
    if isinstance(data, Dataset):
        from openecon.streaming_analysis import fit_streaming
        return fit_streaming(spec, data)
    # All shipped estimators use replayed batches for a large design. Existing
    # frames belong to the caller; the solver avoids a second full design copy.
    frame = _coerce_frame(data)
    design_columns = len(spec.predictors) + int(spec.intercept)
    unknown_category_width = False
    for name in spec.categorical:
        if name in frame and isinstance(frame[name].dtype, pd.CategoricalDtype):
            design_columns += len(frame[name].cat.categories) - 2
        else:
            unknown_category_width = True
    # Discover unknown category levels in bounded batches before dense
    # preparation can allocate a full positional vector or dummy matrices.
    if unknown_category_width:
        from openecon.streaming_analysis import MAX_PARAMETERS
        design_columns = max(design_columns, MAX_PARAMETERS)
    if len(frame) * design_columns * 8 > _MAX_DESIGN_BYTES:
        from openecon.streaming_analysis import fit_streaming
        return fit_streaming(spec, Dataset.from_frame(frame))
    try:
        original, positions, y, x, terms, categories = _prepare_data(spec, frame)
    except AnalysisError as exc:
        # Ordinary string columns can expand beyond the initial width estimate.
        # Dense preparation checks the budget before allocating their dummies.
        if exc.code != "design_too_large":
            raise
        from openecon.streaming_analysis import fit_streaming
        return fit_streaming(spec, Dataset.from_frame(frame))
    n, k = x.shape
    recorded_warnings = []
    if len(original) > n:
        recorded_warnings.append(f"Excluded {len(original) - n} observation(s) with missing model inputs.")
    group_count = None
    groups = None
    correction_factor = None
    if spec.covariance == "cluster":
        labels = original[spec.cluster]
        if len(original) != n:
            labels = labels.iloc[positions]
        if is_numeric_dtype(labels.dtype):
            _numeric(labels, spec.cluster)
        try:
            codes, unique_groups = pd.factorize(labels, sort=False)
            groups = torch.as_tensor(codes, dtype=torch.int64)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_clusters", "Cluster labels must be scalar values.") from exc
        group_count = len(unique_groups)
        if group_count < 2:
            raise AnalysisError("insufficient_clusters", "Cluster covariance requires at least two observed clusters.")
        correction_factor = group_count / (group_count - 1) * (n - 1) / (n - k)
        if group_count < 30:
            recorded_warnings.append(f"Only {group_count} clusters are available; cluster-robust inference may be unreliable.")
    elif spec.covariance == "HC1":
        correction_factor = n / (n - k)
    if spec.estimator != "ols":
        _check_binary_separation(y, x, intercept=spec.intercept)
    try:
        solution = solve(spec.estimator, x, y, spec.covariance, groups, intercept=spec.intercept,
                         max_iter=_MAX_BINARY_ITERATIONS, tolerance=1e-10, device="cpu")
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    except (FloatingPointError, OverflowError) as exc:
        raise AnalysisError("numerical_failure", "The numerical solver could not produce a valid model fit.") from exc
    parameters, covariance, fitted = solution.parameters, solution.covariance, solution.fitted
    condition = float(solution.condition_number)
    if condition > 1e8:
        recorded_warnings.append("The centered/scaled design is ill-conditioned; coefficient and inference sensitivity should be reviewed.")
    if not all(bool(torch.isfinite(v).all()) for v in (parameters, covariance, fitted)):
        raise AnalysisError("non_finite_result", "The solver produced non-finite coefficients, covariance or predictions.")
    if bool((covariance.diag() <= 0).any()):
        raise AnalysisError("invalid_covariance", "The fitted covariance does not provide strictly positive standard errors.")
    standard_errors = covariance.diag().sqrt()
    statistics = parameters / standard_errors
    use_t = spec.estimator == "ols"
    inference_df = (group_count - 1 if group_count is not None else n - k) if use_t else None
    try:
        p_values = two_sided_p_values(statistics, inference_df)
        critical = critical_value(spec.alpha, inference_df)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    intervals = torch.stack((parameters - critical * standard_errors, parameters + critical * standard_errors), dim=1)
    if not all(bool(torch.isfinite(v).all()) for v in (standard_errors, statistics, p_values, intervals)):
        raise AnalysisError("non_finite_result", "The fitted model did not provide finite statistical inference.")
    if spec.estimator != "ols" and bool(((fitted < 0) | (fitted > 1)).any()):
        raise AnalysisError("invalid_predictions", "The binary estimator produced predictions outside [0, 1].")
    residuals = y - fitted
    sse = float(torch.dot(residuals, residuals))
    r_squared = adjusted_r_squared = pseudo_r_squared = None
    if spec.estimator == "ols":
        centered_y = y - y.mean() if spec.intercept else y
        total = float(torch.dot(centered_y, centered_y))
        r_squared = 1 - sse / total if total > 0 else None
        adjusted_r_squared = 1 - ((n - int(spec.intercept)) / (n - k)) * (1 - r_squared) if r_squared is not None else None
    else:
        mean_y = float(y.mean())
        null_ll = n * (mean_y * math.log(mean_y) + (1 - mean_y) * math.log1p(-mean_y))
        pseudo_r_squared = 1 - solution.log_likelihood / null_ll
    metric_values = {
        "r_squared": r_squared, "adjusted_r_squared": adjusted_r_squared, "pseudo_r_squared": pseudo_r_squared,
        "aic": -2 * solution.log_likelihood + 2 * k,
        "bic": -2 * solution.log_likelihood + math.log(n) * k,
        "log_likelihood": solution.log_likelihood, "rmse": math.sqrt(sse / n),
        "df_resid": n - k, "condition_number_scaled": condition,
    }
    sample = original.iloc[positions].reset_index(drop=True) if len(original) != n else original
    input_hasher = _frame_hasher(original)
    data_hash = input_hasher.hexdigest()
    sample_hasher = input_hasher.copy() if sample is original else _frame_hasher(sample)
    sample_hasher.update(_position_bytes(positions))
    try:
        package_version = version("openecon")
    except PackageNotFoundError:
        package_version = "development"
    versions = {"openecon": package_version, "python": platform.python_version(), "pandas": pd.__version__, "torch": torch.__version__}
    chart_rows = torch.linspace(0, n - 1, steps=min(n, 400), dtype=torch.float64).to(dtype=torch.int64).tolist()
    coefficients = [Coefficient(term=term, estimate=float(parameters[i]), std_error=float(standard_errors[i]),
                                statistic=float(statistics[i]), p_value=float(p_values[i]),
                                ci_low=float(intervals[i, 0]), ci_high=float(intervals[i, 1]))
                    for i, term in enumerate(terms)]
    return ResultBundle(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), spec=spec,
        nobs=n, nobs_original=len(original), dropped_rows=len(original) - n,
        coefficients=coefficients, covariance_matrix=covariance.tolist(),
        metrics={name: _safe_metric(value) for name, value in metric_values.items()}, warnings=recorded_warnings,
        predictions=[{"row": positions[i], "observed": float(y[i]), "fitted": float(fitted[i]), "residual": float(residuals[i])} for i in chart_rows],
        sample_positions=positions,
        provenance={
            "schema_version": "3", "backend": "openecon.torch", "engine": "openecon", "device": "cpu",
            "estimator": spec.estimator, "versions": versions, "data_hash": data_hash,
            "sample_hash": sample_hasher.hexdigest(),
            "hash_algorithm": "sha256 over schema and pandas row hashes (pandas version recorded)",
            "hash_scope": "model-input columns in positional row order; index labels excluded",
            "input_columns": list(original.columns), "design_terms": terms, "categorical_encoding": categories,
            "precision": "float64", "sample_position_base": 0,
            "prediction_sample": "evenly spaced positional observations", "prediction_limit": 400,
            "stata_parity_validated": False, "solver": solution.solver,
            "solver_diagnostics": solution.diagnostics,
            "condition_number_basis": solution.diagnostics.get("condition_number_basis", "scaled_design"),
            "optimizer": None if spec.estimator == "ols" else {
                "method": solution.solver, "maxiter": _MAX_BINARY_ITERATIONS, "tolerance": 1e-10,
                "iterations": solution.iterations, "converged": True,
            },
            "separation_diagnostic": "native Torch primal-dual constraint generation with global replay" if spec.estimator != "ols" else None,
        },
        inference={
            "covariance": spec.covariance, "use_t": use_t, "distribution": "t" if use_t else "normal",
            "alpha": spec.alpha, "confidence_level": 1 - spec.alpha, "df_resid": n - k,
            "df_inference": inference_df, "cluster_count": group_count,
            "cluster_df": group_count - 1 if group_count is not None else None, "cluster_column": spec.cluster,
            "small_sample_correction": correction_factor,
            "correction": "CR1: G/(G-1) * (N-1)/(N-K)" if group_count else ("HC1: N/(N-K)" if spec.covariance == "HC1" else spec.covariance),
            "intercept": spec.intercept, "n_parameters": k,
            "residual_definition": "observed minus fitted response",
        },
    )


def _convenience(estimator: str, data: Any, y: str, x: Sequence[str], covariance: str | None,
                 categorical: Sequence[str] | None, intercept: bool, cluster: str | None,
                 missing: str, alpha: float) -> ResultBundle:
    if isinstance(x, (str, bytes)):
        raise AnalysisError("invalid_spec", "x must be a list of predictor column names, for example x=['education'].")
    if isinstance(categorical, (str, bytes)):
        raise AnalysisError("invalid_spec", "categorical must be a list of column names.")
    selected = covariance if covariance is not None else ("cluster" if cluster else ("HC3" if estimator == "ols" else "nonrobust"))
    spec = ModelSpec(estimator=estimator, outcome=y, predictors=list(x), covariance=selected,
                     categorical=list(categorical or []), intercept=intercept, cluster=cluster,
                     missing=missing, alpha=alpha)
    return fit(spec, data=data)


def ols(*, data: Any, y: str | None = None, x: Sequence[str] | None = None,
        formula: str | None = None, covariance: str | None = None,
        categorical: Sequence[str] | None = None, intercept: bool = True,
        cluster: str | Sequence[str] | None = None, weights: str | None = None,
        weight_type: str | None = None, missing: str = "drop", alpha: float = 0.05,
        time: str | None = None, lags: int | str | None = None,
        kernel: str | None = None, reps: int | None = None,
        seed: int | None = None, dfadjust: bool = False,
        hansen: bool = False, device: str = "auto") -> ResultBundle:
    """Full OLS/WLS interface with reusable prediction and postestimation state.

    Default: classical covariance and listwise deletion, like Stata regress.
    Select HC3 explicitly to preserve the earlier alpha's covariance convention.
    Formulae support C(), arithmetic I(), log(), interactions and time lags.
    """
    from openecon.linear_ols import ols as comprehensive_ols
    return comprehensive_ols(data=data, y=y, x=x, formula=formula, covariance=covariance,
                             categorical=categorical, intercept=intercept, cluster=cluster,
                             weights=weights, weight_type=weight_type, missing=missing,
                             alpha=alpha, time=time, lags=lags, kernel=kernel, reps=reps,
                             seed=seed, dfadjust=dfadjust, hansen=hansen, device=device)


def logit(*, data: Any, y: str, x: Sequence[str], covariance: str | None = None,
          categorical: Sequence[str] | None = None, intercept: bool = True,
          cluster: str | None = None, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Binary logistic regression; default covariance is observed information."""
    return _convenience("logit", data, y, x, covariance, categorical, intercept, cluster, missing, alpha)


def probit(*, data: Any, y: str, x: Sequence[str], covariance: str | None = None,
           categorical: Sequence[str] | None = None, intercept: bool = True,
           cluster: str | None = None, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Binary probit regression; default covariance is observed information."""
    return _convenience("probit", data, y, x, covariance, categorical, intercept, cluster, missing, alpha)


def _coefficient_inference(result, method, *args, **kwargs):
    # Retain OLS's fitted basis, Hansen adjustments and streamed contrast state.
    operation = getattr(result, method, None)
    if callable(operation):
        return operation(*args, **kwargs)
    from openecon.econometrics.postest import inference
    return getattr(inference, method)(result, *args, **kwargs)


def test(result, *args, **kwargs):
    """Wald restrictions for OLS or a persisted registry ResultBundle."""
    return _coefficient_inference(result, "test", *args, **kwargs)


def testparm(result, *args, **kwargs):
    """Joint coefficient zero tests by name, factor or wildcard pattern."""
    return _coefficient_inference(result, "testparm", *args, **kwargs)


def lincom(result, *args, **kwargs):
    """Linear-combination inference with the model's recorded covariance."""
    return _coefficient_inference(result, "lincom", *args, **kwargs)


def nlcom(result, *args, **kwargs):
    """Scalar nonlinear delta-method inference from fitted coefficients."""
    return _coefficient_inference(result, "nlcom", *args, **kwargs)


def _prediction_operation(result, operation_name, *args, **kwargs):
    operation = getattr(result, operation_name, None)
    if callable(operation):
        return operation(*args, **kwargs)
    from openecon.econometrics.postest import prediction
    return getattr(prediction, operation_name)(result, *args, **kwargs)


def predict(result, *args, **kwargs):
    """Fitted OLS prediction methods or supported saved scalar-mean adapters."""
    return _prediction_operation(result, "predict", *args, **kwargs)


def margins(result, *args, **kwargs):
    """OLS marginal effects or effects from supported saved mean parameters."""
    return _prediction_operation(result, "margins", *args, **kwargs)
