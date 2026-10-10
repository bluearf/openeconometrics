"""Resident input admission and native float64 conjugate posterior algebra."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from pydantic import BaseModel

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes

MAX_ROWS = 10_000
MAX_PREDICTORS = 32
MAX_DRAWS = 10_000
MAX_DRAW_ELEMENTS = 2_000_000
DEFAULT_MAX_WORK = 100_000_000
DEFAULT_MAX_BYTES = 128 * 1024**2
MAX_JSON_BYTES = 32 * 1024**2


def integer(value, name, *, low=1, high=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise AnalysisError("invalid_option", f"{name} must be an integer in [{low}, {high}].")
    return int(value)


def real(value, name, *, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise AnalysisError("invalid_option", f"{name} must be a finite real number.")
    value = float(value)
    if (low is not None and value <= low) or (high is not None and value >= high):
        raise AnalysisError("invalid_option", f"{name} is outside its open supported interval.")
    return value


def names(values, *, name="x"):
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise AnalysisError("invalid_spec", f"{name} must be an ordered sequence of column names.")
    if len(values) > MAX_PREDICTORS:
        raise AnalysisError("dimension_limit", "Bayesian regression supports at most 32 predictors.")
    result = list(values)
    if any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in result) or len(set(result)) != len(result):
        raise AnalysisError("invalid_spec", "Column names must be nonempty and unique.")
    return result


def admit(n, k, *, operation="bayesian_conjugate", draws=0, queries=0,
          max_work=DEFAULT_MAX_WORK, max_bytes=DEFAULT_MAX_BYTES, index_bytes=0):
    n = integer(n, "rows", low=0, high=MAX_ROWS)
    k = integer(k, "parameters", high=MAX_PREDICTORS + 1)
    draws = integer(draws, "draws", low=0, high=MAX_DRAWS)
    queries = integer(queries, "prediction rows", low=0, high=MAX_ROWS)
    max_work = integer(max_work, "max_work")
    max_bytes = integer(max_bytes, "max_bytes")
    index_bytes = integer(index_bytes, "index_bytes", low=0)
    if draws * (k + 1 + 2 * queries) > MAX_DRAW_ELEMENTS:
        raise AnalysisError("dimension_limit", "Joint parameter and prediction draws exceed 2,000,000 values.")
    work = 16 * n * k**2 + 80 * k**3 + 8 * queries * k**2 + 8 * draws * k * (k + queries + 1)
    if work > max_work:
        raise AnalysisError("work_limit", f"This posterior operation needs {work:,} work units, exceeding max_work.")
    plan = plan_workspace(operation, {
        "source_sample_and_typed_saved_state": 256 * n * (k + 3),
        "source_index_and_descriptors": 1024 * (n + queries) + 8 * index_bytes,
        "design_prior_posterior_factorizations": 256 * k**2 + 48 * n * (k + 1),
        "query_design_scales_and_output": 256 * queries * (k + 4),
        "joint_parameter_and_predictive_draws": 96 * draws * (k + 1 + 2 * queries),
    }, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    return plan.record() | {"estimated_work": work, "max_work": max_work, "max_bytes": max_bytes}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metadata_bytes(value):
    """Bounded iterator traversal; never build a stack proportional to an input array."""
    stack, size, nodes = [(iter((value,)), 0)], 0, 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if depth > 64 or nodes > 2_000_000:
            raise ValueError("Bayesian saved metadata exceeds bounded size/nesting.")
        if isinstance(item, BaseModel):
            item = item.__dict__
        if isinstance(item, Mapping):
            if len(item) > 100 or nodes + len(item) > 2_000_000:
                raise ValueError("Bayesian saved metadata mapping is over-sized.")
            size += 256 + 96 * len(item)
            stack.append((iter(item.values()), depth+1))
            size += sum(64+4*len(key) for key in item if isinstance(key, str))
        elif isinstance(item, (list, tuple)):
            if nodes + len(item) > 2_000_000:
                raise ValueError("Bayesian saved metadata array is over-sized.")
            size += 64 + 16 * len(item)
            stack.append((iter(item), depth+1))
        elif isinstance(item, str):
            size += 64 + 4 * len(item)
        else:
            size += 32
    return size


def resident(data, columns):
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "Bayesian conjugate regression needs resident data; Dataset is not collected.")
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError("unsupported_data", "Supply a resident pandas/OpenEconometrics DataFrame.")
    if data.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Input columns must be unique.")
    if len(data) > MAX_ROWS:
        raise AnalysisError("dimension_limit", "Bayesian conjugate input supports at most 10,000 original rows.")
    if any(column not in data for column in columns):
        raise AnalysisError("missing_columns", "A declared Bayesian input column is absent.")
    return len(data)


def source_values(data, columns):
    result, dtypes = [], []
    for name in columns:
        column = data[name]
        if not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype) or is_complex_dtype(column.dtype):
            raise AnalysisError("unsupported_column", "Bayesian model columns must be real numeric, excluding booleans and complex values.")
        dtypes.append(str(column.dtype))
        values = []
        for value in column:
            if pd.isna(value):
                values.append(None)
                continue
            value = value.item() if hasattr(value, "item") else value
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise AnalysisError("nonfinite_data", "Bayesian data must contain finite real numbers or explicitly handled missing values.")
            if isinstance(value, int) and int(float(value)) != value:
                raise AnalysisError("unsupported_precision", "An integer input is not exactly representable in float64.")
            values.append(value)
        result.append(tuple(values))
    return tuple(result), tuple(dtypes)


def matrices(values, positions, *, intercept):
    block = torch.tensor([[column[i] for column in values] for i in positions],
                         dtype=torch.float64, device="cpu")
    y, x = block[:, 0], block[:, 1:]
    if intercept:
        x = torch.cat((torch.ones((len(positions), 1), dtype=torch.float64, device="cpu"), x), dim=1)
    return y, x


def _log_ratio(numerator, denominator):
    ratio = numerator / denominator
    if 0.5 <= ratio <= 2:
        return math.log1p((numerator - denominator) / denominator)
    return math.log(numerator) - math.log(denominator)


def _log_half_gamma_ratio(shape, *, normalizer=None):
    """log Gamma(a+1/2)/Gamma(a), without subtracting large log-gammas.

    Shift to a>=16 by the exact Gamma recurrence. The Bernoulli-polynomial
    expansion (DLMF 5.11.8) then has first omitted magnitude <3e-18.
    """
    shift = max(0, math.ceil(16 - shape))
    value = shape + shift
    inverse = 1 / value
    square = inverse * inverse
    correction = inverse * (-1/8 + square * (1/192 + square * (-1/640 + square *
        (17/14336 + square * (-31/18432 + square * (691/180224))))))
    recurrences = (
        -math.log1p(0.5 / (shape + i)) if shape + i >= 0.5
        else math.log(shape + i) - math.log(shape + i + 0.5)
        for i in range(shift)
    )
    leading = 0.5 * (math.log(value) if normalizer is None else _log_ratio(value, normalizer))
    return math.fsum((leading, correction, *recurrences))


def _log_gamma_increment(shape, periods, *, normalizer=None):
    """Exact integer/half-integer sample increment, bounded by N<=10000."""
    half = periods % 2
    start = shape + half / 2
    terms = (math.log(start + i) if normalizer is None else _log_ratio(start + i, normalizer)
             for i in range(periods // 2))
    return math.fsum((_log_half_gamma_ratio(shape, normalizer=normalizer) if half else 0.0, *terms))


def _log_scale_increment(increment, original):
    # Keep a small observed-data penalty even when original+increment rounds
    # back to original. The other branch avoids overflow in increment/original.
    if increment <= original:
        return math.log1p(increment / original)
    return math.fsum((math.log(increment), -math.log(original), math.log1p(original / increment)))


def posterior_algebra(y, x, prior):
    """Stable complete proper NIG posterior, retaining scale/covariance distinction."""
    k = x.shape[1]
    mean0 = torch.tensor(prior.mean, dtype=torch.float64, device="cpu")
    v0 = torch.tensor(prior.scale_matrix, dtype=torch.float64, device="cpu")
    if mean0.shape != (k,) or v0.shape != (k, k):
        raise AnalysisError("invalid_prior", "Prior dimensions must match the complete ordered coefficient design.")
    if not bool(torch.isfinite(v0).all()) or not torch.allclose(v0, v0.T, rtol=0, atol=0):
        raise AnalysisError("invalid_prior", "Prior scale_matrix must be finite and exactly symmetric.")
    l0, info = torch.linalg.cholesky_ex(v0)
    if int(info) or not bool(torch.isfinite(l0).all()):
        raise AnalysisError("invalid_prior", "Prior scale_matrix must be positive definite.")
    p0 = torch.cholesky_inverse(l0)
    pn = p0 + x.T @ x
    ln, info = torch.linalg.cholesky_ex(pn)
    if int(info) or not bool(torch.isfinite(ln).all()):
        raise AnalysisError("numerical_failure", "Posterior precision is not representably positive definite.")
    vn = torch.cholesky_inverse(ln)
    mn = torch.cholesky_solve((p0 @ mean0 + x.T @ y)[:, None], ln).flatten()
    residual, deviation = y - x @ mn, mn - mean0
    an = prior.shape + len(y) / 2
    increment = 0.5 * float(residual @ residual + deviation @ p0 @ deviation)
    bn = prior.scale + increment
    scale = vn * (bn / an)
    # The stored shape can round to 1 even though the mathematical increment
    # is strictly above 1 (e.g. two rows and an arbitrarily small proper shape).
    # Evaluate the moment boundary from the original increments, never by
    # subtracting 1 from the rounded posterior shape.
    moment_denominator = math.fsum((prior.shape, (len(y) - 2) / 2))
    variance_mean = bn / moment_denominator if moment_denominator > 0 else None
    covariance = vn * variance_mean if variance_mean is not None else None
    log_evidence = math.fsum((
        -len(y) * math.log(2 * math.pi) / 2,
        -float(torch.log(torch.diag(ln)).sum()),
        -float(torch.log(torch.diag(l0)).sum()),
        _log_gamma_increment(prior.shape, len(y), normalizer=prior.scale),
        -an * _log_scale_increment(increment, prior.scale),
    ))
    blocks = (vn, mn, scale) + (() if covariance is None else (covariance,))
    if not all(bool(torch.isfinite(block).all()) for block in blocks) or not all(math.isfinite(v) for v in (an, bn, log_evidence)) or bn <= 0:
        raise AnalysisError("numerical_failure", "The analytic posterior exceeds supported float64 range.")
    if any(bool(torch.diag(block).le(0).any()) for block in (vn, scale)):
        raise AnalysisError("numerical_failure", "Posterior scales must be representably positive.")
    return {"mean": mn.tolist(), "conditional_scale_matrix": vn.tolist(),
            "shape": an, "scale": bn, "degrees_of_freedom": 2 * an,
            "coefficient_scale_matrix": scale.tolist(),
            "coefficient_covariance": None if covariance is None else covariance.tolist(),
            "variance_mean": variance_mean,
            "log_marginal_likelihood": log_evidence}


def close(actual, expected, name):
    """Unit-normalized replay comparison, including small/zero-valued fields."""
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError(f"Saved {name} availability disagrees with the analytic posterior.")
        return
    a = torch.tensor(actual, dtype=torch.float64, device="cpu")
    b = torch.tensor(expected, dtype=torch.float64, device="cpu")
    if a.shape != b.shape or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError(f"Saved {name} dimensions or finiteness are invalid.")
    # Structural zeros must remain zero. Normalize before subtraction so a
    # subnormal value cannot disappear in the difference or a tolerance floor.
    scale = torch.maximum(torch.abs(a), torch.abs(b))
    denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
    if bool(((b == 0) & (a != 0)).any()) or not bool((
        (a / denominator - b / denominator).abs() <= 2e-11
    ).all()):
        raise ValueError(f"Saved {name} disagrees with the analytic posterior replay.")


def load_mapping(value):
    if isinstance(value, str):
        if len(value) > MAX_JSON_BYTES:
            raise AnalysisError("state_limit", "Saved posterior JSON exceeds 32 MiB.")
        # The decoder's object graph can greatly exceed the input text. Admit
        # before UTF-8 encoding too; encoding itself uses at most four bytes
        # per character and remains inside this conservative allocation plan.
        plan_workspace("bayesian_json_decode", {"decoded_state_and_parser": 64 * len(value)})
        encoded_size = len(value.encode("utf-8"))
        if encoded_size > MAX_JSON_BYTES:
            raise AnalysisError("state_limit", "Saved posterior JSON exceeds 32 MiB.")
        plan_workspace("bayesian_json_decode", {"decoded_state_and_parser": 64 * encoded_size})
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise AnalysisError("invalid_state", "Supply a PosteriorBundle, saved mapping or its JSON string.")
    return value
