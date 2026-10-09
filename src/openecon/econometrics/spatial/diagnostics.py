"""Spatial specification tests from verified, persisted resident OLS geometry.

The source OLS result did not bind a spatial key or a graph.  This procedure
records that association as supplied by the caller now, after verifying the
original numeric model inputs and physical estimation positions.  No estimator
or saved private fit state is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_integer_dtype, is_numeric_dtype
from pydantic import BaseModel, ValidationError

from openecon.analysis import _frame_hasher, _position_bytes
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.common import matched_frame
from openecon.econometrics.postest.index_codec import encode as encode_index
from openecon.econometrics.postest.inference import _parameters
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.models import Coefficient, ModelSpec, ResultBundle
from openecon.resources import plan_workspace

from .weights import SpatialWeights, _keys

TESTS = (
    "moran_normal", "moran_gaussian_mc", "lm_error", "lm_lag",
    "robust_lm_error", "robust_lm_lag", "lm_joint", "wx_f",
)
_MAX_N = 512
_MAX_P = 32
_MAX_WORK = 250_000_000
_MAX_DRAWS = 100_000
_MAX_SOURCE_BYTES = 16 * 1024**2
_MAX_JSON_BYTES = 32 * 1024**2
_MAX_PHYSICAL_BYTES = 16 * 1024**2
_TEST_COLUMNS = [
    "test", "statistic", "distribution", "df_num", "df_denom", "p_value",
    "expected", "variance", "z", "alternative", "extreme_count", "draws", "seed",
]


def _error(code, message):
    raise AnalysisError(code, message)


def _count(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        _error("invalid_option", f"{name} must be an integer in {low}..{high}.")
    return int(value)


def _json_size(value, *, limit, role):
    """Conservative JSON bytes checked before serialization or owned copies."""
    entries, size = 0, 0

    def visit(item, depth):
        nonlocal entries, size
        entries += 1
        if depth > 32 or entries > 1_000_000:
            _error("invalid_result", f"{role} exceeds the bounded JSON identity domain.")
        if isinstance(item, BaseModel):
            for name in type(item).model_fields:
                visit(getattr(item, name), depth + 1)
        elif isinstance(item, dict):
            if len(item) > 1_000_000 - entries or any(not isinstance(k, str) for k in item):
                _error("invalid_result", f"{role} requires bounded string metadata keys.")
            for key, nested in item.items():
                size += 6 * len(key) + 4
                visit(nested, depth + 1)
        elif isinstance(item, (tuple, list)):
            if len(item) > 1_000_000 - entries:
                _error("invalid_result", f"{role} sequences exceed the bounded JSON domain.")
            size += 2 * len(item) + 2
            for nested in item:
                visit(nested, depth + 1)
        elif isinstance(item, str):
            size += 6 * len(item) + 2
        elif item is None or isinstance(item, (bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                _error("invalid_result", f"{role} may not contain nonfinite numbers.")
            if isinstance(item, int) and item.bit_length() > 4096:
                _error("invalid_result", f"{role} integer identity exceeds the bounded JSON domain.")
            size += len(str(item)) + 2
        else:
            _error("invalid_result", f"{role} must consist of ordinary finite JSON data.")
        if size > limit:
            _error("work_budget_exceeded", f"{role} exceeds its {limit:,}-byte JSON budget.")

    # Depth-limited traversal never materializes a second million-entry worklist.
    visit(value, 0)
    return size


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _source(result):
    if not isinstance(result, ResultBundle) or not isinstance(result.spec, ModelSpec):
        _error("invalid_result", "Pass a fitted or JSON-restored OLS ResultBundle.")
    spec = result.spec
    if spec.estimator != "ols":
        _error("unsupported_spatial_diagnostics", "Spatial diagnostics require a saved OLS result.")
    if (
        not isinstance(spec.options, dict) or not isinstance(spec.predictors, list)
        or not isinstance(spec.columns, dict) or not isinstance(spec.categorical, list)
        or not isinstance(result.extra, dict) or not isinstance(result.metrics, dict)
    ):
        _error("invalid_result", "Saved specification, metric and metadata records must have their original typed shapes.")
    if (
        spec.categorical or spec.weights or spec.weight_type or spec.cluster or spec.time
        or spec.panel or spec.columns or spec.covariance != "nonrobust"
        or set(spec.options) - {"terms", "dfadjust", "hansen"}
        or spec.options.get("dfadjust") or spec.options.get("hansen")
    ):
        _error(
            "unsupported_spatial_diagnostics",
            "Use unweighted, nonrobust, numeric cross-sectional OLS with literal predictors.",
        )
    if (
        not isinstance(spec.intercept, bool)
        or not isinstance(spec.predictors, list)
        or any(not isinstance(name, str) or not name.strip() or len(name) > 256 for name in spec.predictors)
        or len(set(spec.predictors)) != len(spec.predictors)
        or not isinstance(spec.outcome, str) or not spec.outcome.strip() or len(spec.outcome) > 256
        or spec.outcome in spec.predictors
    ):
        _error("invalid_result", "Saved outcome and predictor roles must be distinct literal column names.")
    if "terms" in spec.options and spec.options["terms"] != spec.predictors:
        _error("unsupported_spatial_diagnostics", "Formula transforms and interactions are outside this numeric saved-design domain.")
    if len(spec.predictors) + int(spec.intercept) > _MAX_P:
        _error("work_budget_exceeded", "Spatial OLS diagnostics permit at most 32 original design columns.")
    if not isinstance(result.coefficients, list) or not 1 <= len(result.coefficients) <= _MAX_P:
        _error("invalid_result", "Saved OLS must contain 1..32 coefficients.")
    if any(not isinstance(c, Coefficient) or c.equation is not None for c in result.coefficients):
        _error("invalid_result", "Saved OLS requires complete single-equation coefficient records.")
    for coefficient in result.coefficients:
        if any(isinstance(getattr(coefficient, name), bool) or not isinstance(getattr(coefficient, name), Real)
               or not math.isfinite(getattr(coefficient, name)) for name in
               ("estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high")):
            _error("invalid_result", "Saved coefficient numbers must be finite real scalars, excluding booleans.")
    p = len(result.coefficients)
    if (
        isinstance(result.nobs, bool) or not isinstance(result.nobs, int)
        or not max(4, p + 1) <= result.nobs <= _MAX_N
        or isinstance(result.nobs_original, bool) or not isinstance(result.nobs_original, int)
        or not result.nobs <= result.nobs_original <= _MAX_N
        or isinstance(result.dropped_rows, bool) or not isinstance(result.dropped_rows, int)
        or result.dropped_rows != result.nobs_original - result.nobs
    ):
        _error("invalid_result", "Saved physical original and retained samples must agree and contain at most 512 rows.")
    positions = result.sample_positions
    if (
        not isinstance(positions, list) or len(positions) != result.nobs
        or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < result.nobs_original for i in positions)
        or any(a >= b for a, b in zip(positions, positions[1:]))
    ):
        _error("invalid_result", "Saved sample positions must be unique, increasing, zero-based physical row integers.")
    provenance, inference = result.provenance, result.inference
    if not isinstance(provenance, dict) or not isinstance(inference, dict):
        _error("invalid_result", "Saved OLS requires complete provenance and inference records.")
    if (
        provenance.get("engine") != "openecon" or provenance.get("estimator") != "ols"
        or provenance.get("backend") != "openecon.torch" or provenance.get("precision") != "float64"
        or provenance.get("solver") != "torch_qr" or provenance.get("device") != "cpu"
        or provenance.get("streaming") or provenance.get("sample_positions_omitted") is not False
        or provenance.get("postestimation") or result.extra.get("constraints")
    ):
        _error("unsupported_spatial_diagnostics", "Use the original resident CPU float64 unconstrained OLS covariance and complete saved sample.")
    if (
        provenance.get("sample_position_base") != 0
        or isinstance(provenance.get("sample_position_count"), bool)
        or provenance.get("sample_position_count") != result.nobs
        or provenance.get("categories") != {} or provenance.get("categorical_encoding") != {}
    ):
        _error("invalid_result", "Saved numeric OLS coding and positional sample records disagree.")
    columns = [spec.outcome, *spec.predictors]
    if provenance.get("input_columns") != columns:
        _error("invalid_result", "Saved input columns must exactly match the literal numeric OLS roles and order.")
    terms = [c.term for c in result.coefficients]
    original_terms = [*( ["Intercept"] if spec.intercept else []), *spec.predictors]
    omitted = provenance.get("omitted_terms")
    if (
        not isinstance(omitted, list) or any(not isinstance(t, str) for t in omitted)
        or len(set(omitted)) != len(omitted) or not set(omitted).issubset(original_terms)
        or terms != [t for t in original_terms if t not in omitted]
        or len(set(terms)) != p or provenance.get("design_terms") != terms
        or (spec.intercept and "Intercept" not in terms)
    ):
        _error("invalid_result", "Saved coefficients, omissions, intercept and original numeric design order disagree.")
    for name in ("data_hash", "sample_hash"):
        value = provenance.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            _error("invalid_result", "Saved OLS requires complete SHA-256 original and retained-sample hashes.")
    if (
        inference.get("covariance") != "nonrobust" or inference.get("correction") != "nonrobust"
        or inference.get("use_t") is not True or inference.get("distribution") != "t"
        or isinstance(inference.get("df_inference"), bool)
        or inference.get("df_inference") != result.nobs - p
        or inference.get("weight_type") is not None or inference.get("dfadjust") is not False
        or inference.get("hansen") is not False
    ):
        _error("invalid_result", "Saved inference must record the original nonrobust OLS residual degrees of freedom.")
    details = provenance.get("inference_details")
    if (
        not isinstance(details, dict) or details.get("covariance") != "nonrobust"
        or details.get("distribution") != "t" or details.get("df") != result.nobs - p
        or details.get("dfadjust") is not False or details.get("hansen") is not False
        or inference.get("confidence_level") != 1 - spec.alpha
    ):
        _error("invalid_result", "Saved original covariance, confidence level and provenance inference identities disagree.")
    source_bytes = _json_size(result, limit=_MAX_SOURCE_BYTES, role="Saved OLS state")
    try:
        ModelSpec.model_validate(spec.model_dump())
    except (ValidationError, ValueError, TypeError) as exc:
        raise AnalysisError("invalid_result", "The saved OLS specification is malformed.") from exc
    return columns, terms, list(positions), source_bytes


def _frame(data, *, key, columns, original_n):
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        _error("streaming_unsupported", "Spatial saved OLS diagnostics require the original resident DataFrame.")
    if not isinstance(data, pd.DataFrame):
        _error("invalid_data", "Pass the original resident pandas or OpenEconometrics DataFrame.")
    if len(data) != original_n or len(data) > _MAX_N:
        _error("data_mismatch", "The complete original physical OLS table is required, with at most 512 rows.")
    if data.columns.has_duplicates:
        _error("duplicate_columns", "Original data columns must be unique.")
    if not isinstance(key, str) or not key.strip() or len(key) > 256:
        _error("invalid_spatial_keys", "key must name one literal spatial identity column.")
    if key in columns:
        _error("invalid_spatial_keys", "The declared spatial identity must be separate from model input columns.")
    if any(name not in data.columns for name in [*columns, key]):
        _error("data_mismatch", "The original model inputs and declared spatial key column are required.")
    # Do not inspect or copy unrelated user columns.  Deep accounting below is
    # limited to the declared model/key/index receipt, whose physical size is bounded.
    physical_bytes = int(data.index.memory_usage(deep=True)) + sum(int(data[name].memory_usage(index=False, deep=True)) for name in [*columns, key])
    if physical_bytes > _MAX_PHYSICAL_BYTES:
        _error("work_budget_exceeded", "Declared numeric inputs/key/index exceed the 16 MiB physical input budget.")
    plan_workspace("saved spatial OLS input admission", {
        "projected_input_hash_and_receipt": 3 * physical_bytes + 16 * original_n * (len(columns) + 1),
    })
    for name in columns:
        values = data[name]
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype) or is_complex_dtype(values.dtype):
            _error("non_numeric_column", f"Model column '{name}' requires real numeric values, excluding booleans.")
        numpy_dtype = getattr(values.dtype, "numpy_dtype", values.dtype)
        if getattr(numpy_dtype, "itemsize", 8) > 8:
            _error("unsafe_numeric_values", f"Model column '{name}' exceeds the admitted float64 input precision.")
        for value in values:
            if pd.isna(value):
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                _error("non_finite_values", f"Model column '{name}' contains nonfinite real values.")
            if (is_integer_dtype(values.dtype) or isinstance(value, Integral)) and abs(value) > 2**53:
                _error("unsafe_numeric_values", f"Model column '{name}' contains integers beyond the exact float64 domain.")
    keys = _keys(data[key].tolist())
    index_codes = [encode_index(v) for v in data.index]
    index_names = [encode_index(v) for v in data.index.names]
    _json_size([list(keys), index_codes, index_names], limit=_MAX_PHYSICAL_BYTES, role="Declared key/index receipt")
    return keys, index_codes, index_names, physical_bytes


def _weights_input(value, *, n):
    if isinstance(value, SpatialWeights):
        sizes = (value.n, value.nnz)
        # Immutable constructor validation is replayed even for an object whose
        # frozen fields were altered outside the public constructor.
        raw = {"schema": "openecon.spatial_weights.v1", "keys": value.keys,
               "rows": value.rows, "cols": value.cols, "values": value.values,
               "normalization": value.normalization, "diagonal": value.diagonal, "isolates": value.isolates}
    elif isinstance(value, dict):
        raw = value
        if not isinstance(raw.get("keys"), (list, tuple)) or not isinstance(raw.get("values"), (list, tuple)):
            _error("invalid_spatial_weights", "Supply a complete keyed SpatialWeights payload.")
        sizes = (len(raw["keys"]), len(raw["values"]))
    else:
        _error("invalid_spatial_weights", "Supply SpatialWeights or its complete JSON payload.")
    if sizes[0] != n or sizes[1] > n * (n - 1):
        _error("spatial_key_mismatch", "The full original graph must have exactly the original physical key domain.")
    graph_bytes = _json_size(raw, limit=_MAX_PHYSICAL_BYTES, role="Original graph state")
    return raw, sizes[1], graph_bytes


def _verify_geometry(result, frame, columns, terms, positions):
    used = frame.iloc[positions]
    expected_positions = frame.loc[:, columns].notna().all(axis=1).to_numpy().nonzero()[0].tolist()
    if positions != expected_positions or (result.spec.missing == "raise" and len(positions) != len(frame)):
        _error("invalid_result", "Saved positions must be the exact original complete-case numeric OLS sample.")
    sample_hasher = _frame_hasher(used.loc[:, columns].reset_index(drop=True))
    sample_hasher.update(_position_bytes(positions))
    if sample_hasher.hexdigest() != result.provenance["sample_hash"]:
        _error("data_mismatch", "The retained OLS sample hash does not match its original inputs and physical positions.")
    vectors = []
    for term in terms:
        if term == "Intercept" and result.spec.intercept:
            vectors.append(torch.ones(len(positions), dtype=torch.float64, device="cpu"))
        else:
            vectors.append(torch.as_tensor(used[term].to_numpy(dtype="float64", na_value=float("nan")), dtype=torch.float64, device="cpu").clone())
    x = torch.stack(vectors, dim=1)
    y = torch.as_tensor(used[result.spec.outcome].to_numpy(dtype="float64", na_value=float("nan")), dtype=torch.float64, device="cpu").clone()
    parameters = _parameters(result)
    beta, covariance = parameters.beta, parameters.covariance
    scales = x.abs().amax(dim=0)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()) or bool((scales == 0).any()):
        _error("invalid_result", "The retained saved design must have finite nonzero column scales.")
    scaled = x / scales
    try:
        q, r = torch.linalg.qr(scaled, mode="reduced")
        singular = torch.linalg.svdvals(r)
    except RuntimeError as exc:
        raise AnalysisError("invalid_result", "The saved OLS design could not be validated.") from exc
    threshold = 64 * torch.finfo(torch.float64).eps * max(x.shape) * float(singular[0])
    if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= threshold:
        _error("invalid_result", "The declared retained OLS columns must have their full recorded rank.")
    fitted = x @ beta
    residuals = y - fitted
    rss = float(residuals @ residuals)
    if not math.isfinite(rss) or rss <= 0 or not bool(torch.isfinite(fitted).all()):
        _error("degenerate_residuals", "Spatial inference requires positive finite saved OLS residual sum of squares.")
    residual_scale = float(residuals.abs().max())
    outcome_scale = max(float(y.abs().max()), float(fitted.abs().max()), residual_scale)
    projection = q.T @ (residuals / outcome_scale)
    tolerance = 2e-8 * math.sqrt(len(y))
    if float(projection.abs().max()) > tolerance:
        _error("invalid_result", "Saved coefficients do not satisfy OLS residual orthogonality on the matched original design.")
    df = len(y) - len(terms)
    for name, expected in (("df_resid", df), ("df_model", len(terms) - int(result.spec.intercept)), ("ss_resid", rss), ("rmse", math.sqrt(rss / df))):
        observed = result.metrics.get(name)
        if isinstance(observed, bool) or not isinstance(observed, Real) or not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=2e-8, abs_tol=0):
            _error("invalid_result", f"Saved {name} does not agree with the matched OLS sample and coefficients.")
    try:
        inv_r = torch.linalg.solve_triangular(r, torch.eye(len(terms), dtype=torch.float64, device="cpu"), upper=True)
        expected_v = (inv_r @ inv_r.T) * (rss / df)
        expected_v = expected_v / scales[:, None] / scales[None, :]
    except RuntimeError as exc:
        raise AnalysisError("invalid_result", "The original full OLS covariance could not be validated.") from exc
    variances = expected_v.diagonal()
    if not bool(torch.isfinite(expected_v).all()) or bool((variances <= 0).any()):
        _error("invalid_result", "The declared OLS covariance is outside finite float64 inference.")
    std = variances.sqrt()
    relative_v = covariance / std[:, None] / std[None, :]
    reference_v = expected_v / std[:, None] / std[None, :]
    if not torch.allclose(relative_v, reference_v, rtol=2e-8, atol=2e-8):
        _error("invalid_result", "The full saved covariance does not equal RSS/(N-P) times the matched OLS information inverse.")
    for index, coefficient in enumerate(result.coefficients):
        if not math.isclose(coefficient.std_error, float(covariance[index, index].sqrt()), rel_tol=2e-8, abs_tol=0):
            _error("invalid_result", "Saved coefficient standard errors disagree with the full covariance.")
        if not math.isclose(coefficient.statistic, coefficient.estimate / coefficient.std_error, rel_tol=2e-8, abs_tol=2e-8):
            _error("invalid_result", "Saved coefficient t statistics disagree with the original standard errors.")
        p_value = student_t_two_sided(coefficient.statistic, df)
        margin = critical_value(result.spec.alpha, df) * coefficient.std_error
        if (
            not math.isclose(coefficient.p_value, p_value, rel_tol=2e-8, abs_tol=2e-10)
            or not math.isclose(coefficient.ci_low, coefficient.estimate - margin, rel_tol=2e-8, abs_tol=2e-8 * coefficient.std_error)
            or not math.isclose(coefficient.ci_high, coefficient.estimate + margin, rel_tol=2e-8, abs_tol=2e-8 * coefficient.std_error)
        ):
            _error("invalid_result", "Saved coefficient tails and intervals disagree with the original OLS inference.")
    saved_se = result.provenance["inference_details"].get("standard_errors")
    if (
        not isinstance(saved_se, list) or len(saved_se) != len(terms)
        or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isclose(v, result.coefficients[i].std_error, rel_tol=2e-8, abs_tol=0) for i, v in enumerate(saved_se))
    ):
        _error("invalid_result", "Original provenance standard errors disagree with the stored coefficient covariance.")
    return x, y, beta, {"n": len(y), "p": len(terms), "df_resid": df, "rss": rss,
                        "sigma2_unbiased": rss / df, "sigma2_ml": rss / len(y),
                        "rank_tolerance": threshold, "orthogonality_tolerance": tolerance,
                        "max_scaled_residual_projection": float(projection.abs().max()),
                        "covariance_identity_rtol": 2e-8, "covariance_identity_atol_standardized": 2e-8}


@resident_cpu
def spatial_diagnostics(result, *, data, key, spatial_weights, tests=None, wx_predictors=None,
                        simulations=999, seed=0, alternative="two-sided", max_n=512,
                        max_work=250_000_000):
    """Eight bounded saved-OLS spatial tests under fixed-design iid Gaussian errors.

    ``tests`` selects a nonempty ordered subset; the default requests all eight.
    Every requested target must be identified.  The two Moran methods use the
    regression residual projector, LM references are asymptotic chi-squared,
    and WX is a finite-sample nested F test of exogenous spatial covariates.
    Robust LM means robustness to the competing local spatial alternative.
    """
    from .diagnostic_kernels import diagnose, resource_estimates

    max_n = _count(max_n, "max_n", 4, _MAX_N)
    max_work = _count(max_work, "max_work", 1, _MAX_WORK)
    simulations = _count(simulations, "simulations", 1, _MAX_DRAWS)
    seed = _count(seed, "seed", 0, 2**63 - 1)
    if not isinstance(alternative, str) or alternative not in {"two-sided", "greater", "less"}:
        _error("invalid_option", "alternative must be two-sided, greater or less for Moran inference.")
    requested = list(TESTS) if tests is None else tests
    if (
        not isinstance(requested, (list, tuple)) or not requested
        or any(not isinstance(t, str) or t not in TESTS for t in requested)
        or len(set(requested)) != len(requested)
    ):
        _error("invalid_option", "tests must select a nonempty ordered subset of distinct supported spatial diagnostic names.")
    requested = list(requested)
    columns, terms, positions, source_bytes = _source(result)
    n, p = result.nobs, len(terms)
    if n > max_n:
        _error("work_budget_exceeded", f"The retained sample exceeds max_n={max_n} before dense geometry allocation.")
    keys, index_codes, index_names, physical_bytes = _frame(data, key=key, columns=columns, original_n=result.nobs_original)
    raw_weights, original_nnz, graph_bytes = _weights_input(spatial_weights, n=len(data))
    if wx_predictors is None:
        wx_names = [term for term in terms if term != "Intercept"]
    else:
        if not isinstance(wx_predictors, (list, tuple)) or not wx_predictors or any(not isinstance(t, str) for t in wx_predictors):
            _error("invalid_option", "wx_predictors must name distinct retained numeric original slopes.")
        wx_names = list(wx_predictors)
    if len(set(wx_names)) != len(wx_names) or any(t not in terms or t == "Intercept" for t in wx_names):
        _error("invalid_option", "WX targets must be distinct retained original numeric slopes, excluding the intercept.")
    if "wx_f" in requested and (not wx_names or n - p - len(wx_names) <= 0):
        _error("wx_f_unidentified", "The declared WX target needs slopes and positive unrestricted residual degrees of freedom.")
    wx_indices = [terms.index(t) for t in wx_names]
    estimates = resource_estimates(n, p, draws=simulations, mc="moran_gaussian_mc" in requested, wx="wx_f" in requested)
    preparation_work = 8 * n * p**2 + 4 * p**3 + 8 * n * p
    total_work = estimates["work"] + preparation_work
    if total_work > max_work:
        _error("work_budget_exceeded", f"Saved OLS geometry and requested tests need {total_work:,} work units, exceeding max_work={max_work:,}.")
    null_count = simulations if "moran_gaussian_mc" in requested else 0
    # Finite JSON float scalars need at most 32 ASCII bytes.  COO key/position
    # receipts and output objects get separate conservative named proxies.
    json_upper = source_bytes + 2 * graph_bytes + 64 * n**2 + 256 * n * (p + 8) + 64 * null_count + physical_bytes + 65_536
    if json_upper > _MAX_JSON_BYTES:
        _error("work_budget_exceeded", "Complete spatial diagnostic state exceeds the 32 MiB JSON output budget before allocation.")
    plan = plan_workspace("saved OLS spatial diagnostics", {
        **estimates["buffers"], "original_input_projection_and_numeric_copy": physical_bytes + 16 * len(data) * (len(columns) + 1),
        "saved_source_and_graph_JSON_owned_copies": 3 * source_bytes + 6 * graph_bytes,
        "full_original_and_induced_COO_copies": 768 * original_nnz + 320 * len(data),
        "saved_design_beta_V_and_validation_QR": 8 * (6 * n * p + 8 * p * p + 4 * n),
        "complete_output_numeric_table_copies": 24 * (n * n + n * (p + 5) + null_count + 16),
        "complete_output_JSON_proxy": json_upper,
    })
    frame, matched_positions = matched_frame(result, data, role="saved OLS result")
    if matched_positions != positions:
        _error("invalid_result", "Saved physical estimation rows changed during data admission.")
    full_weights = SpatialWeights.from_payload(raw_weights).align(keys)
    retained_keys = [keys[i] for i in positions]
    effective_weights = full_weights.align(retained_keys, subset=len(positions) != len(keys))
    w = effective_weights.dense(max_n=max_n)
    x, y, beta, verification = _verify_geometry(result, frame, columns, terms, positions)
    kernel = diagnose(x, y, beta, w, tests=requested, wx_indices=wx_indices,
                      alternative=alternative, draws=simulations, seed=seed,
                      max_work=max_work - preparation_work, max_n=max_n)
    source_model = _canonical(result.model_dump(mode="json"))
    original_payload, effective_payload = full_weights.to_payload(), effective_weights.to_payload()
    association = {"source": "caller-declared at diagnosis; source OLS did not bind spatial keys, W or index labels",
                   "key_column": key, "original_positions": list(range(len(frame))),
                   "original_keys": list(keys), "original_index_codes": index_codes,
                   "original_index_names": index_names, "sample_positions": positions,
                   "retained_keys": retained_keys}
    sample = table({"key": retained_keys, "original_index_code": [index_codes[i] for i in positions],
                    "observed": y.tolist(), "fitted": kernel["fitted"].tolist(),
                    "residual": kernel["residuals"].tolist()}, index=positions)
    design = table(x.tolist(), columns=terms, index=positions)
    weights = table(w.tolist(), columns=[encode_index(v) for v in retained_keys], index=positions)
    for value in (sample, design, weights):
        value.index.name = "sample_position"
    scores = table([] if kernel["scores"] is None else kernel["scores"].reshape(-1, 1).tolist(),
                   columns=["score"], index=[] if kernel["scores"] is None else ["lag", "error"])
    information = table([] if kernel["information"] is None else kernel["information"].tolist(),
                        columns=["lag", "error"], index=[] if kernel["information"] is None else ["lag", "error"])
    scores.index.name = information.index.name = "parameter"
    null_statistics = kernel["null_statistics"]
    null_simulation = table([[float(v)] for v in null_statistics], columns=["statistic"])
    null_simulation.index.name = "draw"
    test_rows = [{name: row.get(name) for name in _TEST_COLUMNS} for row in kernel["tests"]]
    test_table = table(test_rows, columns=_TEST_COLUMNS).astype(object)
    test_table = test_table.where(pd.notna(test_table), None)
    output = TableSet({"tests": test_table, "sample": sample,
                       "design": design, "weights": weights, "scores": scores,
                       "information": information, "null_simulation": null_simulation},
                      title="Saved OLS spatial diagnostics", schema="openecon.spatial_diagnostics.v1",
                      estimator="ols", sampling_model="fixed_design_iid_gaussian", tests=requested,
                      test_metadata=kernel["tests"],
                      source_model=source_model, source_model_sha256=hashlib.sha256(source_model.encode()).hexdigest(),
                      source_data_hash=result.provenance["data_hash"], source_sample_hash=result.provenance["sample_hash"],
                      source_verified_geometry=verification, geometry=kernel["geometry"],
                      original_spatial_weights=original_payload, effective_spatial_weights=effective_payload,
                      original_spatial_weights_sha256=_digest(original_payload),
                      effective_spatial_weights_sha256=_digest(effective_payload),
                      association=association, association_sha256=_digest(association),
                      graph_subsetting="induced retained graph; row normalization reapplied if normalization='row'; zero isolates retained",
                      wx_predictors=wx_names, alternative=alternative,
                      simulations=simulations if null_count else 0, seed=seed if null_count else None,
                      precision="float64", device="cpu", refitted=False,
                      resource_plan=plan.record(), kernel_resource_plan=kernel["resource_plan"],
                      workspace_observed=kernel["workspace_observed"],
                      work_plan={"preparation": preparation_work, "kernel": estimates["work"], "total": total_work,
                                 "max_work": max_work, "original_n": len(frame), "n": n, "p": p,
                                 "max_n": max_n, "hard_original_n": _MAX_N, "hard_p": _MAX_P,
                                 "json_upper_bound_bytes": json_upper, "json_budget_bytes": _MAX_JSON_BYTES},
                      notes=["Fixed-design unweighted OLS with iid Gaussian errors is the inference model.",
                             "Moran moments use the fitted residual projector; its normal reference is approximate.",
                             "Gaussian Monte Carlo uses inclusive tails and the plus-one rule with a local generator.",
                             "LM references are asymptotic; robust LM adjusts for the competing local spatial alternative.",
                             "WX F tests additional exogenous spatial covariates using the complete declared target rank.",
                             "Source input hashes exclude historical index labels and spatial identities; their current association is caller-declared.",
                             "Coherently replaced saved numbers cannot be authenticated without a signed original fit record."])
    # The shared finite JSON schema sorts mapping keys. Canonical table order
    # therefore survives a generic restore without requiring a serializer change.
    output["settings"] = table([], columns=["setting", "json"])
    output = TableSet(dict(sorted(output.items())), title=output.title, **output.attrs)
    output.attrs["table_index_names"] = {name: list(value.index.names) for name, value in output.items()}
    output.attrs["table_order"] = list(output)
    return saved_summary(output)
