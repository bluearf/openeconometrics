"""Bound shapes, metadata, work and copies before any source/tensor/RNG effect."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes

MAX_ROWS = 10_000
MAX_DRAWS = 10_000
MAX_HORIZON = 24
MAX_ELEMENTS = 2_000_000
MAX_JSON_BYTES = 32 * 1024**2
DEFAULT_MAX_BYTES = 128 * 1024**2
DEFAULT_MAX_WORK = 100_000_000


def integer(value, name, *, low=0, high=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise AnalysisError("invalid_option", f"{name} must be an integer in [{low}, {high}].")
    return int(value)


def real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise AnalysisError(
            "invalid_option", f"{name} must be finite and real, excluding booleans."
        )
    return float(value)


def names(value):
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not 2 <= len(value) <= 4
    ):
        raise AnalysisError("invalid_spec", "Declare an ordered sequence of 2 to 4 series names.")
    if any(not isinstance(v, str) or not v or len(v) > 1000 for v in value) or len(
        set(value)
    ) != len(value):
        raise AnalysisError("invalid_spec", "Series names must be nonempty, bounded and distinct.")
    return tuple(value)


def metadata_size(value):
    """Iterator traversal has bounded auxiliary space, including oversized integers."""
    stack, count, size = [(iter((value,)), 0)], 0, 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        count += 1
        if depth > 32 or count > MAX_ELEMENTS:
            raise AnalysisError(
                "metadata_limit", "Saved BVAR metadata exceeds its depth/value limit."
            )
        if hasattr(item, "__pydantic_fields__"):
            item = item.__dict__
        if isinstance(item, Mapping):
            if len(item) > 100 or any(not isinstance(k, str) or len(k) > 1000 for k in item):
                raise AnalysisError("metadata_limit", "Saved mappings require bounded string keys.")
            size += 256 + sum(96 + 4 * len(k) for k in item)
            stack.append((iter(item.values()), depth + 1))
        elif isinstance(item, (tuple, list)):
            if len(item) > MAX_ELEMENTS or count + len(item) > MAX_ELEMENTS:
                raise AnalysisError("metadata_limit", "Saved BVAR array is oversized.")
            size += 64 + 16 * len(item)
            stack.append((iter(item), depth + 1))
        elif isinstance(item, str):
            if len(item) > 10_000:
                raise AnalysisError("metadata_limit", "Individual saved strings are oversized.")
            size += 64 + 4 * len(item)
        elif isinstance(item, int) and not isinstance(item, bool):
            if item.bit_length() > 128:
                raise AnalysisError(
                    "metadata_limit", "Saved integers are limited to 128 magnitude bits."
                )
            size += 48
        elif item is None or isinstance(item, bool):
            size += 32
        elif isinstance(item, float) and math.isfinite(item):
            size += 32
        else:
            raise AnalysisError(
                "invalid_state", "Saved BVAR metadata contains an unsupported primitive."
            )
    return size


def metadata_admit(value, *, max_bytes=DEFAULT_MAX_BYTES, operation="bvar_metadata"):
    size = metadata_size(value)
    plan_workspace(
        operation,
        {"typed_metadata_validation_serialization_copies": 4 * size},
        budget_bytes=min(integer(max_bytes, "max_bytes", low=1), workspace_budget_bytes()),
    )
    return size


def admit(
    n,
    m,
    p,
    intercept,
    *,
    draws=0,
    horizon=0,
    queries=0,
    irf=False,
    max_work=DEFAULT_MAX_WORK,
    max_bytes=DEFAULT_MAX_BYTES,
    index_bytes=0,
):
    n = integer(n, "original rows", high=MAX_ROWS)
    m = integer(m, "series", low=2, high=4)
    p = integer(p, "lags", low=1, high=4)
    if type(intercept) is not bool or type(irf) is not bool:
        raise AnalysisError("invalid_option", "intercept and irf must be boolean.")
    k = m * p + int(intercept)
    draws = integer(draws, "draws", high=MAX_DRAWS)
    horizon = integer(horizon, "horizon", high=MAX_HORIZON)
    queries = integer(queries, "query rows", high=MAX_ROWS)
    max_work = integer(max_work, "max_work", low=1)
    max_bytes = integer(max_bytes, "max_bytes", low=1)
    index_bytes = integer(index_bytes, "index bytes")
    d = (horizon + 1) * m**2 if irf else horizon * m
    # Stored primitives, coefficients/Sigma, summaries and both mean/outcome paths.
    values = draws * (2 * k * m + 2 * m**2 + (d if irf else 3 * d)) + 2 * d**2
    values += queries * (k + 3 * m + 4 * m**2) + 2 * (queries * m) ** 2
    if values > MAX_ELEMENTS:
        raise AnalysisError(
            "dimension_limit", "Joint BVAR draw/output state exceeds 2,000,000 values."
        )
    work = 16 * n * k**2 + 80 * k**3 + 16 * queries * (k**2 + m**2) + 16 * queries**2 * (k + m**2)
    work += draws * (32 * (k * m**2 + k**2 * m + (m * p) ** 3) + 16 * d * (k + d))
    if work > max_work:
        raise AnalysisError(
            "work_limit", f"BVAR operation needs {work:,} work units, exceeding max_work."
        )
    plan = plan_workspace(
        "bayesian_var",
        {
            "source_values_index_and_typed_copies": 320 * n * (m + 3) + 8 * index_bytes,
            "design_prior_factorizations": 128 * n * (k + m) + 512 * k**2,
            "full_joint_parameter_covariance": 128 * (k * m + m * (m + 1) // 2) ** 2,
            "query_states": 256 * queries * (k + m + m**2),
            "primitive_draw_output_and_full_covariance_copies": 128 * values,
        },
        budget_bytes=min(max_bytes, workspace_budget_bytes()),
    )
    return plan.record() | {"estimated_work": work, "stored_numeric_values": values}


def digest(value):
    metadata_size(value)
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    ).hexdigest()


def load_mapping(value):
    if isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_JSON_BYTES:
            raise AnalysisError(
                "metadata_limit", "Encoded BVAR state exceeds 32 MiB before JSON decoding."
            )
        if isinstance(value, str) and len(value.encode()) > MAX_JSON_BYTES:
            raise AnalysisError("metadata_limit", "Encoded BVAR UTF-8 state exceeds 32 MiB.")
        try:
            value = json.loads(
                value, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON"))
            )
        except (ValueError, RecursionError) as exc:
            raise AnalysisError("invalid_state", "Saved BVAR JSON is invalid.") from exc
    if not isinstance(value, Mapping):
        raise AnalysisError(
            "invalid_state", "Supply a complete saved BVAR mapping or encoded JSON."
        )
    metadata_admit(value, max_bytes=value.get("max_bytes", DEFAULT_MAX_BYTES))
    return value
