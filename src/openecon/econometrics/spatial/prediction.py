"""Restored fixed-network ML mean projections and full joint delta state.

The target integrates over spatial innovations. No realized residual is used,
no estimator is called, and the effective fitted network cannot be replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_integer_dtype, is_numeric_dtype
from pydantic import BaseModel

from openecon.analysis import _frame_hasher
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.index_codec import encode as encode_index
from openecon.econometrics.postest.inference import _parameters
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import normal_isf
from openecon.models import Coefficient, ResultBundle
from openecon.resources import plan_workspace

from .kernels import stable_bound
from .weights import SpatialWeights, _keys

_MODELS = {"sar", "sem", "sac", "sdm"}
_MAX_N = 512
_MAX_PARAMETERS = 128
_MAX_SOURCE_BYTES = 16 * 1024**2


def _error(code, message):
    raise AnalysisError(code, message)


def _count(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        _error("invalid_option", f"{name} must be an integer in {low}..{high}.")
    return int(value)


def _source_size(result):
    """Admit JSON hashing before allocating a complete serialized result copy."""
    pending = [(result, 0)]
    entries, total = 0, 0
    while pending:
        item, depth = pending.pop()
        entries += 1
        if depth > 32 or entries > 1_000_000:
            _error("invalid_result", "Saved state exceeds the bounded JSON identity domain.")
        if isinstance(item, BaseModel):
            pending.extend((getattr(item, name), depth + 1) for name in type(item).model_fields)
        elif isinstance(item, dict):
            if len(item) > 1_000_000 - entries:
                _error("invalid_result", "Saved metadata exceeds the bounded JSON identity domain.")
            if any(not isinstance(key, str) for key in item):
                _error("invalid_result", "Saved JSON state must have string metadata keys.")
            total += sum(6 * len(key) + 4 for key in item)
            pending.extend((value, depth + 1) for value in item.values())
        elif isinstance(item, (list, tuple)):
            if len(item) > 1_000_000 - entries:
                _error("invalid_result", "Saved sequences exceed the bounded JSON identity domain.")
            total += 2 * len(item) + 2
            pending.extend((value, depth + 1) for value in item)
        elif isinstance(item, str):
            total += 6 * len(item) + 2
        elif item is None or isinstance(item, (bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                _error("invalid_result", "Saved state may not contain nonfinite numbers.")
            if isinstance(item, int) and item.bit_length() > 4096:
                _error("invalid_result", "Saved integer identities exceed the bounded JSON domain.")
            total += len(str(item)) + 2
        else:
            _error("invalid_result", "Saved state must be ordinary finite JSON data.")
        if total > _MAX_SOURCE_BYTES:
            _error(
                "work_budget_exceeded", "Canonical saved-result identity exceeds the 16 MiB budget."
            )
    return total


def _result(result):
    if not isinstance(result, ResultBundle) or result.spec.estimator not in _MODELS:
        _error(
            "unsupported_spatial_prediction",
            "Pass a restored SAR, SEM, SAC or SDM Gaussian ML ResultBundle.",
        )
    spec = result.spec
    if (
        result.provenance.get("estimator") != spec.estimator
        or result.provenance.get("family") != "spatial"
    ):
        _error(
            "invalid_result",
            "Saved fitted estimator identity must agree with the declared spatial ML model.",
        )
    if (
        spec.covariance != "nonrobust"
        or spec.weights
        or spec.weight_type
        or spec.cluster
        or spec.panel
        or spec.time
        or spec.categorical
        or spec.options.get("formula")
    ):
        _error(
            "unsupported_spatial_prediction",
            "Use the saved unweighted numeric cross-sectional Gaussian ML design.",
        )
    if (
        not isinstance(spec.intercept, bool)
        or not isinstance(spec.predictors, list)
        or any(not isinstance(name, str) or not name or len(name) > 256 for name in spec.predictors)
        or len(set(spec.predictors)) != len(spec.predictors)
    ):
        _error("invalid_result", "Saved predictor roles must be distinct numeric column names.")
    key = spec.columns.get("key")
    if not isinstance(key, str) or not key or key in [spec.outcome, *spec.predictors]:
        _error(
            "invalid_result",
            "Saved spatial key role must be separate from outcomes and predictors.",
        )
    if set(spec.columns) != {"key"}:
        _error("invalid_result", "Saved ML spatial roles must contain exactly the key identity.")
    k = len(result.coefficients)
    if not 1 <= k <= _MAX_PARAMETERS:
        _error(
            "work_budget_exceeded", "Saved network mean inference permits at most 128 parameters."
        )
    if any(not isinstance(item, Coefficient) for item in result.coefficients):
        _error(
            "invalid_result",
            "Saved coefficient records must be complete typed coefficient objects.",
        )
    if (
        not isinstance(result.nobs, int)
        or isinstance(result.nobs, bool)
        or not k < result.nobs <= _MAX_N
    ):
        _error(
            "invalid_result",
            "Saved sample must have more observations than parameters and at most 512 units.",
        )
    if (
        not isinstance(result.nobs_original, int)
        or isinstance(result.nobs_original, bool)
        or result.nobs_original < result.nobs
        or len(result.sample_positions) != result.nobs
        or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < result.nobs_original
            for i in result.sample_positions
        )
        or len(set(result.sample_positions)) != result.nobs
    ):
        _error(
            "invalid_result",
            "Saved effective-sample positions must be complete, distinct and within the original rows.",
        )
    expected = (["Intercept"] if spec.intercept else []) + spec.predictors
    if spec.estimator == "sdm":
        expected += ["W:" + name for name in spec.predictors]
    expected += (
        ["lambda"]
        if spec.estimator == "sem"
        else ["rho", "lambda"]
        if spec.estimator == "sac"
        else ["rho"]
    )
    expected += ["ln_sigma2"]
    if len(expected) != len(set(expected)) or set(expected) != {
        c.term for c in result.coefficients
    }:
        _error(
            "invalid_result",
            "Saved coefficient names must match all regression, spatial and variance roles exactly.",
        )
    design_terms = result.provenance.get("design_terms")
    if (
        not isinstance(design_terms, list)
        or any(not isinstance(term, str) for term in design_terms)
        or len(design_terms) != len(expected)
        or set(design_terms) != set(expected)
        or result.inference.get("n_parameters") != k
        or result.inference.get("covariance") != spec.covariance
    ):
        _error(
            "invalid_result",
            "Saved fitted design and covariance identity must match every declared parameter role.",
        )
    if (
        result.inference.get("use_t") is not False
        or result.inference.get("distribution") != "normal"
    ):
        _error(
            "invalid_result",
            "Gaussian ML network means require the saved asymptotic normal inference law.",
        )
    optimizer = result.provenance.get("optimizer")
    diagnostics = result.provenance.get("solver_diagnostics")
    if (
        not isinstance(optimizer, dict)
        or optimizer.get("converged") is not True
        or not isinstance(diagnostics, dict)
        or diagnostics.get("converged") is not True
    ):
        _error("invalid_result", "The saved spatial maximum must record completed convergence.")
    if result.provenance.get("postestimation"):
        _error(
            "unsupported_spatial_prediction",
            "Post-estimation covariance replacements require their own verified spatial adapter.",
        )
    return key, expected


def _inputs(data, key, predictors, n):
    if not isinstance(data, pd.DataFrame):
        _error(
            "unsupported_data", "Saved network mean inference requires a resident pandas DataFrame."
        )
    if len(data) != n:
        _error(
            "spatial_key_mismatch",
            "Provide every effective fitted-network key exactly once; new or partial networks are unsupported.",
        )
    names = [key, *predictors]
    if data.columns.has_duplicates or any(name not in data for name in names):
        _error(
            "prediction_domain", "All key and saved numeric predictor columns must exist uniquely."
        )
    for name in predictors:
        series = data[name]
        if (
            not is_numeric_dtype(series.dtype)
            or is_bool_dtype(series.dtype)
            or is_complex_dtype(series.dtype)
        ):
            _error(
                "invalid_values",
                "Query predictors must be real numeric values, excluding booleans and strings.",
            )
        if is_integer_dtype(series.dtype):
            minimum, maximum = series.min(skipna=True), series.max(skipna=True)
            if not pd.isna(minimum) and (int(minimum) < -(2**53) or int(maximum) > 2**53):
                _error(
                    "numerical_range",
                    "Integer query predictors must lie within the exact float64 domain [-2**53,2**53].",
                )
        elif getattr(series.dtype, "itemsize", 8) > 8:
            _error(
                "numerical_range",
                "Wider-than-float64 query predictors require explicit user conversion.",
            )
    if data[names].isna().any().any():
        _error(
            "missing_values",
            "The complete fixed network needs complete query keys and predictors; no rows are dropped.",
        )
    keys = _keys(data[key].tolist())
    labels = [encode_index(label) for label in data.index]
    if any(len(value) > 4096 for value in labels) or sum(map(len, labels)) > 2 * 1024**2:
        _error(
            "work_budget_exceeded",
            "Encoded query row identities exceed their bounded storage domain.",
        )
    if any(len(name) > 256 for name in predictors):
        _error("work_budget_exceeded", "Query feature names are limited to 256 characters.")
    projected = data.loc[:, names].copy()
    try:
        x = (
            torch.stack(
                [
                    torch.as_tensor(
                        projected[name].to_numpy(dtype="float64", copy=True), dtype=torch.float64
                    )
                    for name in predictors
                ],
                dim=1,
            )
            if predictors
            else torch.empty((n, 0), dtype=torch.float64)
        )
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        raise AnalysisError(
            "invalid_values", "Query values must be representable as finite float64 numbers."
        ) from exc
    if not bool(torch.isfinite(x).all()):
        _error("non_finite_values", "The complete query design must contain finite values.")
    return projected, x, keys, labels


def _weights(result):
    weights = SpatialWeights.from_payload(result.extra.get("spatial_weights"))
    summary = weights.summary()
    saved_summary = result.extra.get("spatial_summary")
    if (
        summary["sha256"] != result.provenance.get("spatial_weight_hash")
        or not isinstance(saved_summary, dict)
        or saved_summary.get("sha256") != summary["sha256"]
        or weights.n != result.nobs
    ):
        _error(
            "invalid_result",
            "Saved effective spatial weights differ from their hash or effective sample size.",
        )
    errors = None
    if result.spec.estimator == "sac":
        errors = SpatialWeights.from_payload(result.extra.get("error_weights"))
        error_summary = result.extra.get("error_weight_summary")
        if (
            not isinstance(error_summary, dict)
            or error_summary.get("sha256") != errors.summary()["sha256"]
            or set(errors.keys) != set(weights.keys)
        ):
            _error(
                "invalid_result",
                "Saved SAC error graph must match its recorded hash and effective key domain.",
            )
    return weights, errors


def _law(result, state, weights, errors):
    for term, graph in (("rho", weights), ("lambda", errors or weights)):
        if term not in state.terms:
            continue
        value = float(state.beta[state.terms.index(term)])
        norm = graph.summary()["row_sum_max"]
        if not math.isfinite(norm) or norm <= 0 or abs(value) >= (1 - 1e-6) / norm:
            _error(
                "invalid_result",
                f"Saved {term} lies outside the recorded finite stable spatial law.",
            )
    variance_log = float(state.beta[state.terms.index("ln_sigma2")])
    if not -700 <= variance_log <= 700:
        _error(
            "invalid_result",
            "Saved innovation variance lies outside resolved finite float64 units.",
        )
    variance = math.exp(variance_log)
    recorded = result.extra.get("innovation_variance")
    try:
        recorded_float = float(recorded)
    except (OverflowError, ValueError, TypeError) as exc:
        raise AnalysisError(
            "invalid_result", "Saved innovation variance must be finite and resolved."
        ) from exc
    if (
        not isinstance(recorded, Real)
        or isinstance(recorded, bool)
        or not math.isfinite(recorded_float)
        or not math.isclose(recorded_float, variance, rel_tol=1e-10)
    ):
        _error("invalid_result", "Saved innovation variance and ln_sigma2 disagree.")


def _solve(a, rhs):
    try:
        value = torch.linalg.solve(a, rhs)
    except RuntimeError as exc:
        raise AnalysisError(
            "spatial_solve_failure", "The saved spatial mean filter could not be solved."
        ) from exc
    residual = (a @ value - rhs).abs().amax(0)
    norm = float(a.abs().sum(1).amax())
    denominator = norm * value.abs().amax(0) + rhs.abs().amax(0)
    relative = torch.where(denominator > 0, residual / denominator, residual)
    if (
        not bool(torch.isfinite(value).all())
        or not bool(torch.isfinite(relative).all())
        or float(relative.max()) > 1e-10
    ):
        _error(
            "spatial_solve_failure",
            "The saved spatial mean solve has an unresolved backward residual.",
        )
    return value, float(relative.max())


@resident_cpu
@torch.no_grad()
def spatial_predict(result, *, data, alpha=0.05, max_n=512, max_work=250_000_000):
    """Unconditional ML mean/delta inference on the complete saved fixed graph.

    SAR/SAC/SDM use (I-rho W)^-1 times the declared query mean index; SEM
    uses X beta. SDM recomputes WX on all query nodes. Full parameter and
    query-mean covariance, gradients and keyed rows survive summary_state.
    Normal mean confidence intervals exclude innovation/observation error.
    Resident N<=512 and P<=128; no network replacement, new nodes, row drop,
    Dataset batching, fixed/realized errors, or IV results. No refit occurs.
    """
    max_n = _count(max_n, "max_n", 4, _MAX_N)
    max_work = _count(max_work, "max_work", 1, 1_000_000_000)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or getattr(getattr(alpha, "dtype", None), "itemsize", 8) > 8
        or not 1e-12 <= alpha < 1
    ):
        _error("invalid_alpha", "alpha must be a finite float64-compatible number in [1e-12,1).")
    alpha = float(alpha)
    key, _ = _result(result)
    n, p = result.nobs, len(result.coefficients)
    if n > max_n:
        _error("spatial_dense_limit", "The complete effective network exceeds the requested max_n.")
    saved_max = result.spec.options.get("max_n")
    if isinstance(saved_max, bool) or not isinstance(saved_max, int) or not n <= saved_max <= 2048:
        _error("invalid_result", "The saved dense-network ceiling is invalid.")
    work = 2 * n**3 + 2 * n**2 * p + 2 * n * p**2 + 2 * p**3
    if work > max_work:
        _error(
            "work_budget_exceeded", "Dense solves/covariance exceed the declared operation budget."
        )
    source_bytes = _source_size(result)
    plan = plan_workspace(
        "saved spatial full-network mean and joint delta inference",
        {
            "filters, solve, full mean covariance and temporary products": 8 * 24 * n**2,
            "query design, derivative solves and full Jacobian": 8 * 16 * n * p,
            "saved parameter covariance, validation and output copies": 8 * 16 * p**2,
            "canonical full-result serialization and hashing": 2 * source_bytes,
            "positions, keys and admitted encoded row identities": (512 + 2 * 4096) * n,
        },
    ).record()
    state = _parameters(result)
    weights, errors = _weights(result)
    _law(result, state, weights, errors)
    projected, numeric, keys, labels = _inputs(data, key, result.spec.predictors, n)
    if set(keys) != set(weights.keys):
        _error(
            "spatial_key_mismatch",
            "Query identities must match the complete saved effective graph.",
        )
    aligned = weights.align(keys)
    w = aligned.dense(max_n=max_n)
    x = (
        torch.cat((torch.ones((n, 1), dtype=torch.float64), numeric), 1)
        if result.spec.intercept
        else numeric
    )
    base_terms = (["Intercept"] if result.spec.intercept else []) + result.spec.predictors
    mean_design = x
    mean_terms = list(base_terms)
    if result.spec.estimator == "sdm":
        mean_design = torch.cat((x, w @ numeric), 1)
        mean_terms += ["W:" + term for term in result.spec.predictors]
    indices = [state.terms.index(term) for term in mean_terms]
    rhs = mean_design @ state.beta[indices]
    if not bool(torch.isfinite(rhs).all()):
        _error("non_finite_prediction", "The saved coefficients exceed finite query mean units.")
    jacobian = torch.zeros((n, p), dtype=torch.float64)
    backward_error = 0.0
    if result.spec.estimator == "sem":
        mean = rhs
        jacobian[:, indices] = mean_design
    else:
        rho_index = state.terms.index("rho")
        try:
            bound = stable_bound(w)
        except KernelError as exc:
            raise AnalysisError(exc.code, str(exc)) from exc
        if abs(float(state.beta[rho_index])) >= bound:
            _error("invalid_result", "Aligned saved rho lies outside the stable mean law.")
        a = torch.eye(n, dtype=torch.float64) - state.beta[rho_index] * w
        mean_matrix, backward_error = _solve(a, rhs[:, None])
        mean = mean_matrix[:, 0]
        solved, derivative_error = _solve(a, torch.cat((mean_design, (w @ mean)[:, None]), 1))
        jacobian[:, indices] = solved[:, :-1]
        jacobian[:, rho_index] = solved[:, -1]
        backward_error = max(backward_error, derivative_error)
    covariance = jacobian @ state.covariance @ jacobian.T
    absolute = jacobian.abs() @ state.covariance.abs() @ jacobian.abs().T
    if (
        not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(jacobian).all())
        or not bool(torch.isfinite(covariance).all())
        or not bool(torch.isfinite(absolute).all())
    ):
        _error(
            "non_finite_prediction",
            "Saved mean/gradient/covariance is outside finite float64 units.",
        )
    tolerance = 128 * p * torch.finfo(torch.float64).eps * absolute.diagonal().clamp_min(1e-300)
    if bool((covariance.diagonal() < -tolerance).any()):
        _error("invalid_covariance", "Saved full covariance produces negative mean variances.")
    covariance = covariance / 2 + covariance.T / 2
    zero_variance = covariance.diagonal() == 0
    gradient_scale = jacobian.abs().amax(1)
    covariance_scale = float(state.covariance.abs().max())
    if bool((zero_variance & (gradient_scale > 0)).any()) and covariance_scale > 0:
        normalized_covariance = state.covariance / covariance_scale
        normalized_gradient = (
            jacobian / gradient_scale.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
        )
        relative_variance = (
            (normalized_gradient @ normalized_covariance) * normalized_gradient
        ).sum(1)
        relative_absolute = (
            (normalized_gradient.abs() @ normalized_covariance.abs()) * normalized_gradient.abs()
        ).sum(1)
        guard = 128 * p * torch.finfo(torch.float64).eps * relative_absolute
        # A positive-definite saved covariance has strictly positive variance
        # for every nonzero gradient. For PSD state, retain genuine null-space
        # and zero-J cases; only a resolved positive quadratic form is refused.
        cholesky = torch.linalg.cholesky_ex(normalized_covariance, check_errors=False)
        nonzero_resolved = (relative_variance > guard) | (
            (cholesky.info == 0) & (gradient_scale > 0)
        )
        if bool((zero_variance & nonzero_resolved & (gradient_scale > 0)).any()):
            _error(
                "numerical_range",
                "Nonzero projected mean variance is not representable in float64; no false zero standard error is returned.",
            )
    standard_error = covariance.diagonal().clamp_min(0).sqrt()
    critical = normal_isf(alpha / 2)
    low, high = mean - critical * standard_error, mean + critical * standard_error
    if not bool(torch.isfinite(low).all()) or not bool(torch.isfinite(high).all()):
        _error("non_finite_prediction", "Mean confidence endpoints exceed finite float64 units.")
    source_json = json.dumps(
        result.model_dump(mode="json"), allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    if len(source_json.encode()) > _MAX_SOURCE_BYTES:
        _error("work_budget_exceeded", "Canonical saved-result identity exceeds the 16 MiB budget.")
    columns = ["parameter:" + term for term in state.terms]
    sample_rows = [
        [i, k, label, *row] for i, (k, label, row) in enumerate(zip(keys, labels, numeric.tolist()))
    ]
    output = TableSet(
        {
            "means": table(
                {
                    "position": list(range(n)),
                    "key": list(keys),
                    "mean": mean.tolist(),
                    "std_error": standard_error.tolist(),
                    "ci_low": low.tolist(),
                    "ci_high": high.tolist(),
                }
            ),
            "sample": table(
                sample_rows,
                columns=[
                    "position",
                    "key",
                    "index_json",
                    *["input:" + name for name in result.spec.predictors],
                ],
            ),
            "jacobian": table(jacobian.tolist(), columns=columns),
            "parameter_covariance": table(state.covariance.tolist(), columns=columns),
            "mean_covariance": table(
                covariance.tolist(), columns=["row:" + str(i) for i in range(n)]
            ),
        },
        title="Saved " + result.spec.estimator.upper() + " complete-network mean inference",
        method="spatial_predict",
        estimator=result.spec.estimator,
        source_result_id=result.id,
        source_result_sha256=hashlib.sha256(source_json.encode()).hexdigest(),
        source_hash_scope="SHA256 of complete canonical finite JSON ResultBundle, including all covariance and keyed spatial state",
        alpha=alpha,
        confidence_level=1 - alpha,
        critical_value=critical,
        reference="asymptotic normal full joint parameter delta method",
        target="unconditional reduced-form mean given fixed X and complete effective keyed graph",
        conditioning="Fixed exogenous graph and query X; no realized error or innovation/observation uncertainty",
        refitted=False,
        mean_formula="X beta"
        if result.spec.estimator == "sem"
        else "solve(I-rho W, X beta + WX_slopes theta [SDM only])",
        parameter_order=list(state.terms),
        coefficients=state.beta.tolist(),
        covariance_key_order=list(keys),
        saved_graph_sha256=weights.summary()["sha256"],
        aligned_graph_sha256=aligned.summary()["sha256"],
        saved_error_graph_sha256=errors.summary()["sha256"] if errors else None,
        saved_effective_weights=weights.to_payload(),
        saved_error_weights=errors.to_payload() if errors else None,
        query_sha256=_frame_hasher(projected).hexdigest(),
        query_hash_scope="Complete key/X schema and row order; index stored separately",
        row_index_codec="openecon.econometrics.postest.index_codec.decode(sample.index_json)",
        n=n,
        parameter_count=p,
        max_n=max_n,
        max_work=max_work,
        estimated_work=work,
        backward_solve_relative_error=backward_error,
        solve_error_limit=1e-10,
        resource_plan=plan,
        workspace_observed={
            "aligned_dense_W_bytes": w.numel() * w.element_size(),
            "query_numeric_bytes": numeric.numel() * numeric.element_size(),
            "augmented_mean_design_bytes": mean_design.numel() * mean_design.element_size(),
            "full_jacobian_bytes": jacobian.numel() * jacobian.element_size(),
            "full_parameter_covariance_bytes": state.covariance.numel()
            * state.covariance.element_size(),
            "full_mean_covariance_bytes": covariance.numel() * covariance.element_size(),
            "scope": "Named tensor sizes; not peak RSS, allocator or private factorization workspace",
        },
        precision="float64",
        device="cpu",
        dataset_support=False,
        weights_support=False,
        mean_covariance_scope="Full N-by-N joint covariance in query key order; arbitrary saved coefficient permutations matched by names",
        nuisance_mean_gradients="lambda and ln_sigma2 are zero; beta covariance remains full jointly estimated covariance",
        uncertainty_scope="Mean confidence intervals; no future-observation interval, graph uncertainty or ML global-maximum guarantee",
        persistence="All covariance/gradient rows and indexed query inputs via oe.summary_state; console previews may truncate",
    )
    return saved_summary(output)
