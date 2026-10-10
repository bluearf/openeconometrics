"""Complete, replayable proper-prior point-null comparisons and affine predictions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype, pandas_dtype
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.mi.common import (
    _decode_index,
    _encode_index,
    _freeze,
    _index_envelope,
    _thaw,
)
from openecon.engines.distributions import t_isf
from openecon.resources import plan_workspace, workspace_budget_bytes

from .core import (
    MAX_JSON_BYTES,
    MAX_ROWS,
    admit,
    digest,
    integer,
    matrices,
    metadata_bytes as _base_metadata_bytes,
    resident,
    source_values,
)
from .hypothesis_kernels import draw_arrays, restricted_algebra
from .posterior import PosteriorBundle

SCHEMA = "openecon.posterior_hypothesis_comparison.v1"
DRAW_SCHEMA = "openecon.posterior_hypothesis_draws.v1"
MAX_QUERIES = 512
_NULL_MATRICES = {
    "prior_conditional_scale_matrix",
    "conditional_scale_matrix",
    "coefficient_scale_matrix",
    "coefficient_covariance",
}
_NULL_SCALARS = {
    "prior_shape",
    "prior_scale",
    "prior_scale_increment",
    "shape",
    "scale",
    "data_scale_increment",
    "variance_moment_denominator",
    "degrees_of_freedom",
    "variance_mean",
    "alpha",
    "log_marginal_likelihood",
}
_NULL_KEYS = (
    _NULL_MATRICES
    | _NULL_SCALARS
    | {
        "schema_version",
        "terms",
        "constraint_rank",
        "free_dimension",
        "prior_mean",
        "basis",
        "free_mean",
        "free_conditional_scale_matrix",
        "mean",
        "credible_intervals",
    }
)
_RESULT_KEYS = {
    "null_posterior",
    "log_bayes_factor_null_alternative",
    "bayes_factor_null_alternative",
    "bayes_factor_status",
    "constraint_prior_log_density",
    "constraint_posterior_log_density",
    "density_coordinate_convention",
    "log_posterior_model_odds_null_alternative",
    "posterior_probability_null",
    "posterior_probability_alternative",
}
_PAYLOAD_KEYS = {
    "schema_version",
    "method",
    "alternative",
    "constraints",
    "values",
    "labels",
    "model_prior_odds_null_alternative",
    "alpha",
    "results",
    "resource_admission",
    "digest",
}


def _error(message, code="invalid_state"):
    raise AnalysisError(code, message)


def metadata_bytes(value):
    """Include arbitrary-size integer storage/encoded digits in bounded metadata admission."""
    size = _base_metadata_bytes(value)
    stack = [iter((value,))]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(item, BaseModel):
            item = item.__dict__
        if isinstance(item, Mapping):
            stack.append(iter(item.values()))
        elif isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif type(item) is int:
            bits = item.bit_length()
            size += (bits + 7) // 8 + 4 * (bits * 30103 // 100000 + 2)
    return size


def _json_input_admission(value):
    """Reserve bounded encoded and dense decoded objects before either JSON decoder."""
    if not isinstance(value, (str, bytes, bytearray)) or len(value) > MAX_JSON_BYTES:
        _error("Saved posterior JSON exceeds 32 MiB.", "state_limit")
    if isinstance(value, str):
        size = 0
        for character in value:
            code = ord(character)
            size += 1 if code < 128 else 2 if code < 2048 else 3 if code < 65536 else 4
            if size > MAX_JSON_BYTES:
                _error("Saved posterior JSON exceeds 32 MiB.", "state_limit")
    else:
        size = len(value)
    # Dense arrays of empty objects can expand far beyond encoded byte count;
    # include decoder objects, Pydantic validation and retained copies.
    plan_workspace(
        "hypothesis JSON decoding and typed validation", {"encoded and decoded state": 64 * size}
    )


def load_mapping(value):
    if isinstance(value, (str, bytes, bytearray)):
        _json_input_admission(value)
        try:
            value = json.loads(value)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise AnalysisError(
                "invalid_state", "Saved hypothesis state needs valid bounded JSON."
            ) from exc
    if not isinstance(value, Mapping):
        _error("Supply a complete posterior model, resident mapping or bounded JSON.")
    return value


def _deep_copy_admission(model):
    plan_workspace(
        "hypothesis typed model deep copy",
        {"complete model object and copy buffers": 8 * metadata_bytes(model.__dict__)},
    )


def _serialization_admission(model, indent):
    if indent is not None and (type(indent) is not int or indent < 0):
        _error("JSON indent must be a nonnegative integer or None.", "invalid_option")
    if indent is not None and indent > MAX_JSON_BYTES:
        _error("JSON indent exceeds the complete 32 MiB encoded envelope.", "state_limit")
    metadata = metadata_bytes(model.__dict__)
    padding = 0
    if indent:
        stack = [(iter((model.__dict__,)), 0)]
        while stack:
            try:
                item = next(stack[-1][0])
            except StopIteration:
                stack.pop()
                continue
            depth = stack[-1][1]
            # A container emits indentation on opening and closing lines.
            padding += 2 * indent * (depth + 1)
            if isinstance(item, Mapping):
                stack.append((iter(item.values()), depth + 1))
            elif isinstance(item, (list, tuple)):
                stack.append((iter(item), depth + 1))
    encoded_bound = metadata + padding
    if encoded_bound > MAX_JSON_BYTES:
        _error("Complete indented JSON exceeds the 32 MiB encoded envelope.", "state_limit")
    plan_workspace(
        "hypothesis complete JSON serialization",
        {"complete retained metadata and output buffers": 4 * metadata + 4 * encoded_bound},
    )


def _real(value, name, *, positive=False, probability=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        _error(f"{name} must be a finite real number, excluding booleans.", "invalid_option")
    original = value
    try:
        value = float(value)
    except (OverflowError, ValueError):
        _error(f"{name} must be representable in float64.", "invalid_option")
    if not math.isfinite(value) or (positive and value <= 0) or (probability and not 0 < value < 1):
        _error(f"{name} lies outside its finite supported interval.", "invalid_option")
    if isinstance(original, Integral) and int(value) != original:
        _error(f"{name} is not exactly representable in float64.", "unsupported_precision")
    return value


def _sequence(value, size, name):
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or len(value) != size
    ):
        _error(f"{name} must be a resident array with exactly {size} entries.")
    return value


def _vector(value, size, name):
    _sequence(value, size, name)
    for cell in value:
        _real(cell, name)


def _matrix(value, rows, columns, name):
    _sequence(value, rows, name)
    for row in value:
        _vector(row, columns, name)


def _keys(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != expected:
        _error(f"{name} must contain exactly its declared versioned fields.")


def _posterior_raw(value):
    """Inspect a typed result without serializing an unadmitted copied payload."""
    if isinstance(value, PosteriorBundle):
        value = value.__dict__
    if not isinstance(value, Mapping):
        _error("The alternative must be a complete PosteriorBundle or its saved mapping.")
    state = value.get("state")
    if isinstance(state, BaseModel):
        state = state.__dict__
    if not isinstance(state, Mapping):
        _error("The alternative source state must be a resident mapping.")
    columns = state.get("source_values")
    if not isinstance(columns, (list, tuple)) or not 1 <= len(columns) <= 33:
        _error("The alternative source must have 1 to 33 resident numeric columns.")
    first = columns[0]
    if not isinstance(first, (list, tuple)) or not 1 <= len(first) <= MAX_ROWS:
        _error("The alternative source row count is outside 1 to 10,000.")
    n = len(first)
    for column in columns:
        _sequence(column, n, "alternative source column")
        for cell in column:
            if cell is not None:
                _real(cell, "alternative source cell")
    terms = value.get("terms")
    if not isinstance(terms, (list, tuple)) or not 1 <= len(terms) <= 33:
        _error("The alternative coefficient dimension is outside 1 to 33.")
    k = len(terms)
    if any(not isinstance(term, str) or not term.strip() or len(term) > 1000 for term in terms):
        _error(
            "Alternative coefficient names must be nonempty strings of at most 1,000 characters."
        )
    max_work = integer(value.get("max_work"), "max_work")
    max_bytes = integer(value.get("max_bytes"), "max_bytes")
    admit(n, k, max_work=max_work, max_bytes=max_bytes)
    for key in (
        "mean",
        "conditional_scale_matrix",
        "coefficient_scale_matrix",
        "coefficient_covariance",
        "credible_intervals",
    ):
        block = value.get(key)
        if key == "coefficient_covariance" and block is None:
            continue
        if key == "mean":
            _vector(block, k, "alternative " + key)
        else:
            _matrix(block, k, 2 if key == "credible_intervals" else k, "alternative " + key)
    for key in ("shape", "scale", "degrees_of_freedom", "log_marginal_likelihood", "alpha"):
        _real(value.get(key), "alternative " + key)
    if value.get("variance_mean") is not None:
        _real(value["variance_mean"], "alternative variance_mean")
    prior = value.get("prior")
    if isinstance(prior, BaseModel):
        prior = prior.__dict__
    if not isinstance(prior, Mapping):
        _error("The alternative needs a complete proper prior mapping.")
    _vector(prior.get("mean"), k, "alternative prior mean")
    _matrix(prior.get("scale_matrix"), k, k, "alternative prior scale_matrix")
    _real(prior.get("shape"), "alternative prior shape", positive=True)
    _real(prior.get("scale"), "alternative prior scale", positive=True)
    plan_workspace(
        "bayesian_hypothesis_alternative_admission",
        {"complete_source_and_typed_validation_copy": 4 * metadata_bytes(value)},
        budget_bytes=min(max_bytes, workspace_budget_bytes()),
    )
    return n, k, max_work, max_bytes


def _alternative(value):
    if isinstance(value, str):
        value = load_mapping(value)
    _posterior_raw(value)
    if isinstance(value, PosteriorBundle):
        value = value.model_dump(mode="json")
    try:
        return PosteriorBundle.model_validate(dict(value))
    except (OverflowError, TypeError, KeyError, AttributeError) as exc:
        raise AnalysisError(
            "invalid_state", "The alternative posterior is not a complete finite typed state."
        ) from exc


def _admit_comparison(alternative, rank, *, draws=0, queries=0, metadata=0):
    n, k, work, limit = _posterior_raw(alternative)
    rank = integer(rank, "constraint rank", high=k)
    queries = integer(queries, "joint query rows", low=0, high=MAX_QUERIES)
    estimated = (
        32 * n * k * k
        + 240 * k**3
        + 16 * queries * queries * k
        + 16 * draws * k * (k + queries + 1)
    )
    if estimated > work:
        _error(
            f"This complete restricted posterior replay needs {estimated:,} work units, exceeding max_work.",
            "work_limit",
        )
    admit(n, k, draws=draws, queries=queries, max_work=work, max_bytes=limit)
    plan = plan_workspace(
        "bayesian_hypothesis_complete_replay",
        {
            "complete_alternative_state_replay_and_copy": 8 * metadata_bytes(alternative),
            "comparison_metadata_replay_and_copy": 4 * metadata,
            "prior_constraint_and_free_geometry": 768 * k * k,
            "restricted_design_and_analytic_factorizations": 128 * n * (k + 1),
            "joint_prediction_scale_and_covariance": 256 * queries * queries
            + 256 * queries * (k + 4),
            "seeded_exact_joint_draws_and_saved_copy": 128 * draws * (k + 1 + 2 * queries),
        },
        budget_bytes=min(limit, workspace_budget_bytes()),
    )
    return plan.record() | {
        "estimated_work": estimated,
        "max_work": work,
        "max_bytes": limit,
        "constraint_rank": rank,
        "free_dimension": k - rank,
        "joint_query_limit": MAX_QUERIES,
    }


def _cache_envelope(results, k, r):
    _keys(results, _RESULT_KEYS, "comparison results")
    null = results["null_posterior"]
    _keys(null, _NULL_KEYS, "restricted posterior")
    if null["schema_version"] != "openecon.restricted_nig_posterior.v1":
        _error("Unknown restricted posterior schema version.")
    if not isinstance(results["bayes_factor_status"], str) or results[
        "bayes_factor_status"
    ] not in {"finite", "overflow", "underflow"}:
        _error("Unknown Bayes-factor representability status.")
    if (
        results["density_coordinate_convention"]
        != "prior-equilibrated constraint rows; common row Jacobian cancels"
    ):
        _error("Unknown exact constraint-density coordinate convention.")
    free = k - r
    for key in _NULL_MATRICES:
        _matrix(null[key], k, k, key)
    for key in ("prior_mean", "mean"):
        _vector(null[key], k, key)
    _vector(null["free_mean"], free, "free_mean")
    _matrix(null["basis"], k, free, "basis")
    _matrix(null["free_conditional_scale_matrix"], free, free, "free_conditional_scale_matrix")
    _matrix(null["credible_intervals"], k, 2, "credible_intervals")
    _sequence(null["terms"], k, "restricted terms")
    if any(
        not isinstance(term, str) or not term.strip() or len(term) > 1000 for term in null["terms"]
    ):
        _error("Restricted coefficient names must be nonempty strings of at most 1,000 characters.")
    for key in _NULL_SCALARS:
        _real(null[key], key)
    if (
        type(null["constraint_rank"]) is not int
        or type(null["free_dimension"]) is not int
        or null["constraint_rank"] != r
        or null["free_dimension"] != free
    ):
        _error(
            "Restricted model dimensions must be strict integers matching admitted original/free dimensions."
        )
    for key in _RESULT_KEYS - {
        "null_posterior",
        "bayes_factor_status",
        "density_coordinate_convention",
    }:
        if results[key] is not None:
            _real(results[key], key)


def _compare(actual, expected, name):
    if isinstance(expected, Mapping):
        _keys(actual, set(expected), name)
        for key, value in expected.items():
            _compare(actual[key], value, name + "." + key)
    elif isinstance(expected, str) or expected is None or type(expected) in (int, bool):
        if type(actual) is not type(expected) or actual != expected:
            _error(f"Saved {name} differs from complete analytic replay.")
    elif isinstance(expected, (list, tuple)):
        _sequence(actual, len(expected), name)
        for index, (saved, value) in enumerate(zip(actual, expected)):
            _compare(saved, value, f"{name}[{index}]")
    else:
        saved = _real(actual, name)
        # Affine point constraints contain structural zeros. No absolute
        # floor may turn their support into even subnormal uncertainty.
        if expected == 0.0:
            agrees = saved == 0.0
        else:
            scale = max(abs(saved), abs(expected))
            agrees = abs(saved - expected) / scale <= 2e-11
        if not agrees:
            _error(f"Saved {name} disagrees with complete analytic replay.")


def _comparison_envelope(payload):
    _keys(payload, _PAYLOAD_KEYS, "hypothesis state")
    if (
        payload["schema_version"] != SCHEMA
        or payload["method"] != "conditional_proper_nig_linear_equalities"
    ):
        _error("Unknown proper-prior hypothesis schema or method.")
    c = payload["constraints"]
    if not isinstance(c, (list, tuple)) or not 1 <= len(c) <= 33:
        _error("Declare 1 to 33 resident constraint rows.")
    if not isinstance(payload["alternative"], Mapping) or not isinstance(
        payload["alternative"].get("state"), Mapping
    ):
        _error("The saved alternative and source must be portable resident mappings.")
    _, k, _, _ = _posterior_raw(payload["alternative"])
    r = len(c)
    if r > k:
        _error("The constraint count cannot exceed the original coefficient dimension.")
    _matrix(c, r, k, "constraints")
    _vector(payload["values"], r, "constraint values")
    _sequence(payload["labels"], r, "constraint labels")
    labels = payload["labels"]
    if (
        any(
            not isinstance(label, str) or not label.strip() or len(label) > 1000 for label in labels
        )
        or len(set(labels)) != r
    ):
        _error("Constraint labels must be distinct, nonempty strings of at most 1,000 characters.")
    alpha = _real(payload["alpha"], "alpha", probability=True)
    odds = payload["model_prior_odds_null_alternative"]
    if odds is not None:
        odds = _real(odds, "model prior odds", positive=True)
    _cache_envelope(payload["results"], k, r)
    _admit_comparison(payload["alternative"], r, metadata=metadata_bytes(payload))
    return k, r, alpha, odds


def _replay(payload):
    _, r, alpha, odds = _comparison_envelope(payload)
    alternative = _alternative(payload["alternative"])
    expected = restricted_algebra(
        alternative, payload["constraints"], payload["values"], odds, alpha
    )
    _compare(payload["results"], expected, "results")
    # Resource receipts describe the source operation, not variable serialized
    # object overhead during later validations. Recompute its stable dimensions.
    original_record = _admit_comparison(payload["alternative"], r)
    receipt = payload["resource_admission"]
    _keys(receipt, set(original_record), "resource_admission")
    recorded_budget = integer(receipt["budget_bytes"], "recorded workspace budget")
    if not original_record["estimated_workspace_bytes"] <= recorded_budget <= alternative.max_bytes:
        _error("Recorded workspace admission is inconsistent with its complete source operation.")
    original_record["budget_bytes"] = recorded_budget
    _compare(payload["resource_admission"], original_record, "resource_admission")
    if payload["digest"] != digest(
        {key: value for key, value in payload.items() if key != "digest"}
    ):
        _error("Hypothesis digest disagrees with its complete source, model and posterior state.")
    return payload


class PosteriorHypothesisComparison(BaseModel):
    """Immutable complete alternative/null posterior state with exact point-null evidence."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True, revalidate_instances="always"
    )
    schema_version: Literal["openecon.posterior_hypothesis_comparison.v1"] = SCHEMA
    payload: Any

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        """Admit encoded input before decoding complete posterior state."""
        _json_input_admission(json_data)
        return super().model_validate_json(json_data, **kwargs)

    def __deepcopy__(self, memo=None):
        _deep_copy_admission(self)
        _checked(self)
        return super().__deepcopy__(memo)

    def model_dump(self, **kwargs):
        """Validate complete semantics before any portable copy, including field exclusions."""
        _checked(self)
        return super().model_dump(**kwargs)

    def model_dump_json(self, *, indent=None, **kwargs):
        """Admit complete encoded output and replay before indented JSON allocation."""
        _serialization_admission(self, indent)
        _checked(self)
        return super().model_dump_json(indent=indent, **kwargs)

    @model_validator(mode="after")
    def validate_semantics(self):
        """Replay conditional-prior geometry, all uncertainty and both model evidences."""
        if self.schema_version != SCHEMA:
            _error("Unknown hypothesis schema version.")
        object.__setattr__(self, "payload", _freeze(_replay(self.payload)))
        return self

    @field_serializer("payload")
    def serialize_payload(self, value):
        """Serialize admitted, replayed source and posterior fields without a trusted-cache bypass."""
        if self.schema_version != SCHEMA:
            _error("Unknown hypothesis schema version.")
        return _thaw(_replay(value))

    def summary(self) -> TableSet:
        """Return separately labelled model evidence and full original-unit null coefficient uncertainty."""
        state = _checked(self)
        results, null = state["results"], state["results"]["null_posterior"]
        coefficients = pd.DataFrame(
            {
                "Term": null["terms"],
                "Posterior mean": null["mean"],
                "Posterior std. dev.": [
                    math.sqrt(row[i]) for i, row in enumerate(null["coefficient_covariance"])
                ],
                "Student-t scale": [
                    math.sqrt(row[i]) for i, row in enumerate(null["coefficient_scale_matrix"])
                ],
                "Credible lower": [row[0] for row in null["credible_intervals"]],
                "Credible upper": [row[1] for row in null["credible_intervals"]],
            }
        )
        models = pd.DataFrame(
            {
                "Model": ["null", "alternative"],
                "Log marginal likelihood": [
                    null["log_marginal_likelihood"],
                    state["alternative"]["log_marginal_likelihood"],
                ],
                "Posterior model probability": [
                    results["posterior_probability_null"],
                    results["posterior_probability_alternative"],
                ],
            }
        )
        models.attrs.update(
            {
                "log_bayes_factor_null_alternative": results["log_bayes_factor_null_alternative"],
                "model_prior_odds_null_alternative": state["model_prior_odds_null_alternative"],
                "null_prior": "alternative joint prior conditioned on C beta = d",
            }
        )
        tables = {"models": models, "null_coefficients": coefficients}
        for key in (
            "prior_conditional_scale_matrix",
            "conditional_scale_matrix",
            "coefficient_scale_matrix",
            "coefficient_covariance",
        ):
            tables["null_" + key] = pd.DataFrame(
                null[key], index=null["terms"], columns=null["terms"]
            )
        return TableSet(
            tables,
            title="Proper-prior linear-equality model comparison",
            schema_version=SCHEMA,
            comparison_sha256=state["digest"],
            log_bayes_factor_null_alternative=results["log_bayes_factor_null_alternative"],
            model_prior_odds_null_alternative=state["model_prior_odds_null_alternative"],
            null_prior="alternative joint NIG prior conditioned on C beta = values",
            constraint_rank=null["constraint_rank"],
            free_dimension=null["free_dimension"],
            credible_probability=1 - state["alpha"],
        )

    def to_tables(self) -> TableSet:
        """Render fully replayed native model/coefficient tables and full original-unit null matrices."""
        return self.summary()


def _raw_state(value):
    if isinstance(value, PosteriorHypothesisComparison):
        if value.schema_version != SCHEMA:
            _error("Unknown hypothesis schema version.")
        return value.payload
    value = load_mapping(value)
    if set(value) == {"schema_version", "payload"}:
        if value["schema_version"] != SCHEMA:
            _error("Unknown hypothesis schema version.")
        value = value["payload"]
    return value


def _checked(value):
    return _replay(_raw_state(value))


def bayes_hypothesis(
    *,
    result: PosteriorBundle | Mapping | str,
    constraints: Sequence[Sequence[float]],
    values: Sequence[float],
    labels: Sequence[str] | None = None,
    model_prior_odds: float | None = None,
    alpha: float | None = None,
) -> PosteriorHypothesisComparison:
    """Compare the proper alternative with exact full-row-rank equalities C beta = values.

    The null prior is the complete alternative joint prior conditioned on the
    equalities, including sigma². Explicit prior odds are required for posterior
    model probabilities; an absent prior-odds argument leaves them unavailable.
    Constraints follow the complete original coefficient order. Redundant or
    numerically unresolved rows are refused, including rank changes by tolerance.
    """
    if isinstance(result, str):
        result = load_mapping(result)
    _, k, _, _ = _posterior_raw(result)
    if isinstance(constraints, (str, bytes)) or not isinstance(constraints, (list, tuple)):
        _error("constraints must be a resident list or tuple of rows.", "invalid_hypothesis")
    r = len(constraints)
    if not 1 <= r <= k:
        _error(
            "Declare between 1 and the original coefficient count constraint rows.",
            "invalid_hypothesis",
        )
    _matrix(constraints, r, k, "constraints")
    _vector(values, r, "values")
    labels = [f"constraint_{i + 1}" for i in range(r)] if labels is None else labels
    _sequence(labels, r, "labels")
    if (
        any(
            not isinstance(label, str) or not label.strip() or len(label) > 1000 for label in labels
        )
        or len(set(labels)) != r
    ):
        _error("Constraint labels must be distinct, nonempty strings of at most 1,000 characters.")
    if alpha is not None:
        alpha = _real(alpha, "alpha", probability=True)
    odds = (
        None
        if model_prior_odds is None
        else _real(model_prior_odds, "model_prior_odds", positive=True)
    )
    _admit_comparison(
        result,
        r,
        metadata=metadata_bytes({"constraints": constraints, "values": values, "labels": labels}),
    )
    alternative = _alternative(result)
    alpha = alternative.alpha if alpha is None else _real(alpha, "alpha", probability=True)
    alternative_map = alternative.model_dump(mode="json")
    resource = _admit_comparison(alternative_map, r)
    solved = restricted_algebra(alternative, constraints, values, odds, alpha)
    payload = {
        "schema_version": SCHEMA,
        "method": "conditional_proper_nig_linear_equalities",
        "alternative": alternative_map,
        "constraints": [list(row) for row in constraints],
        "values": list(values),
        "labels": list(labels),
        "model_prior_odds_null_alternative": odds,
        "alpha": alpha,
        "results": solved,
        "resource_admission": resource,
    }
    return PosteriorHypothesisComparison(payload=payload | {"digest": digest(payload)})


def bayes_hypothesis_restore(
    *, saved: PosteriorHypothesisComparison | Mapping | str
) -> PosteriorHypothesisComparison:
    """Restore complete analytic model comparison by semantic replay, without refitting or sampling."""
    return PosteriorHypothesisComparison(payload=_checked(saved))


def _query(state, data, missing):
    if not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("Query missing policy must be raise or drop.", "invalid_option")
    alternative = state["alternative"]
    source = alternative["state"]
    predictors = source["predictors"]
    n = resident(data, list(predictors))
    _admit_comparison(alternative, len(state["constraints"]), queries=n)
    index = _encode_index(data.index)
    size = metadata_bytes(index)
    plan_workspace(
        "bayesian_hypothesis_query_index",
        {"complete_query_index": 4 * size},
        budget_bytes=min(alternative["max_bytes"], workspace_budget_bytes()),
    )
    values, dtypes = source_values(data, list(predictors))
    positions = tuple(i for i in range(n) if all(column[i] is not None for column in values))
    if not positions or (missing == "raise" and len(positions) != n):
        _error(
            "Query complete-case sample is empty or violates its declared missing policy.",
            "missing_data",
        )
    dummy = (tuple(0 for _ in range(n)), *values)
    _, design = matrices(dummy, positions, intercept=source["intercept"])
    query = {
        "columns": list(predictors),
        "dtypes": list(dtypes),
        "values": [list(column) for column in values],
        "index": index,
        "sample_positions": list(positions),
        "missing": missing,
        "rows_original": n,
    }
    return design, data.index[list(positions)], query


def bayes_hypothesis_predict(
    *,
    result: PosteriorHypothesisComparison | Mapping | str,
    data: Any,
    target: str = "null",
    missing: str = "raise",
    alpha: float | None = None,
) -> pd.DataFrame:
    """Predict conditional means and new outcomes with complete joint Student-t scale/covariance.

    target is the declared null or alternative model. A singular constrained
    coefficient scale and deterministic all-fixed mean target are valid. Every
    new outcome retains its independent Gaussian observation uncertainty.
    """
    if not isinstance(target, str) or target not in {"null", "alternative"}:
        _error("target must be null or alternative.", "invalid_option")
    if not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("Query missing policy must be raise or drop.", "invalid_option")
    if alpha is not None:
        alpha = _real(alpha, "alpha", probability=True)
    raw = _raw_state(result)
    _comparison_envelope(raw)
    q = resident(data, list(raw["alternative"]["state"]["predictors"]))
    _admit_comparison(raw["alternative"], len(raw["constraints"]), queries=q)
    state = _checked(result)
    alpha = state["alpha"] if alpha is None else _real(alpha, "alpha", probability=True)
    selected = state["results"]["null_posterior"] if target == "null" else state["alternative"]
    with torch.device("cpu"):
        design, index, query = _query(state, data, missing)
        mean = design @ torch.tensor(selected["mean"], dtype=torch.float64, device="cpu")
        if target == "null":
            if selected["free_dimension"]:
                basis = torch.tensor(selected["basis"], dtype=torch.float64, device="cpu")
                free_factor = torch.linalg.cholesky(
                    torch.tensor(
                        selected["free_conditional_scale_matrix"], dtype=torch.float64, device="cpu"
                    )
                )
                loading = design @ basis @ free_factor
            else:
                loading = torch.zeros((len(design), 0), dtype=torch.float64, device="cpu")
        else:
            factor_v = torch.linalg.cholesky(
                torch.tensor(
                    selected["conditional_scale_matrix"], dtype=torch.float64, device="cpu"
                )
            )
            loading = design @ factor_v
        joint_mean = (loading @ loading.T) * (selected["scale"] / selected["shape"])
        joint_outcome = joint_mean + torch.eye(len(design), dtype=torch.float64, device="cpu") * (
            selected["scale"] / selected["shape"]
        )
        variance_mean = selected["variance_mean"]
        mean_covariance = None if variance_mean is None else (loading @ loading.T) * variance_mean
        outcome_covariance = (
            None
            if variance_mean is None
            else mean_covariance
            + torch.eye(len(design), dtype=torch.float64, device="cpu") * variance_mean
        )
        smean, soutcome = torch.diag(joint_mean).sqrt(), torch.diag(joint_outcome).sqrt()
        critical = t_isf(alpha / 2, selected["degrees_of_freedom"])
        values = torch.stack(
            (
                mean,
                smean,
                soutcome,
                mean - critical * smean,
                mean + critical * smean,
                mean - critical * soutcome,
                mean + critical * soutcome,
            ),
            dim=1,
        )
        finite_blocks = (values, joint_mean, joint_outcome) + (
            () if mean_covariance is None else (mean_covariance, outcome_covariance)
        )
        if not all(bool(torch.isfinite(block).all()) for block in finite_blocks):
            _error("Predictions exceed representable float64 range.", "numerical_failure")
        frame = pd.DataFrame(
            values.tolist(),
            index=index,
            columns=[
                "mean",
                "mean_student_t_scale",
                "outcome_student_t_scale",
                "mean_credible_lower",
                "mean_credible_upper",
                "outcome_predictive_lower",
                "outcome_predictive_upper",
            ],
        )
        frame.attrs.update(
            {
                "schema_version": "openecon.hypothesis_prediction.v1",
                "comparison_sha256": state["digest"],
                "target": target,
                "query_state": query,
                "credible_probability": 1 - alpha,
                "degrees_of_freedom": selected["degrees_of_freedom"],
                "joint_mean_student_t_scale": joint_mean.tolist(),
                "joint_outcome_student_t_scale": joint_outcome.tolist(),
                "joint_mean_covariance": None
                if mean_covariance is None
                else mean_covariance.tolist(),
                "joint_outcome_covariance": None
                if outcome_covariance is None
                else outcome_covariance.tolist(),
            }
        )
        return frame


def _query_envelope(query, state):
    if query is None:
        return None
    _keys(
        query,
        {"columns", "dtypes", "values", "index", "sample_positions", "missing", "rows_original"},
        "saved query",
    )
    n = integer(query["rows_original"], "query original rows", high=MAX_QUERIES)
    source = state["alternative"]["state"]
    k = len(source["predictors"])
    _sequence(query["columns"], k, "saved query columns")
    if list(query["columns"]) != list(source["predictors"]):
        _error("Saved query columns disagree with the complete model predictor order.")
    _sequence(query["dtypes"], k, "saved query dtypes")
    for dtype in query["dtypes"]:
        if not isinstance(dtype, str) or not dtype or len(dtype) > 1000:
            _error("Saved query dtypes must be bounded numeric dtype strings.")
        try:
            parsed = pandas_dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_state", "A saved query dtype is unsupported.") from exc
        if not is_numeric_dtype(parsed) or is_bool_dtype(parsed) or is_complex_dtype(parsed):
            _error("Saved query dtypes must be real numeric, excluding booleans/complex types.")
    _sequence(query["values"], k, "saved query values")
    for column in query["values"]:
        _sequence(column, n, "saved query column")
        for value in column:
            if value is not None:
                _real(value, "saved query cell")
    _index_envelope(query["index"], n)
    positions = tuple(
        i for i in range(n) if all(column[i] is not None for column in query["values"])
    )
    if (
        not positions
        or not isinstance(query["missing"], str)
        or query["missing"] not in {"raise", "drop"}
        or (query["missing"] == "raise" and len(positions) != n)
    ):
        _error("Saved query missing policy disagrees with its original complete-case sample.")
    if (
        not isinstance(query["sample_positions"], (list, tuple))
        or any(type(value) is not int for value in query["sample_positions"])
        or tuple(query["sample_positions"]) != positions
    ):
        _error("Saved query positions disagree with the original source.")
    return n, source, positions


def _query_replay(query, state):
    if query is None:
        return None
    n, source, positions = _query_envelope(query, state)
    index = _decode_index(query["index"])
    if len(index) != n or _encode_index(index) != _thaw(query["index"]):
        _error("Saved query index does not round trip exactly.")
    for values, dtype in zip(query["values"], query["dtypes"], strict=True):
        frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
        captured, _ = source_values(frame, ["x"])
        if tuple(values) != captured[0]:
            _error("Saved query values do not round trip in their original declared dtype.")
    _, design = matrices(
        (tuple(0 for _ in range(n)), *query["values"]), positions, intercept=source["intercept"]
    )
    return design.tolist()


def _replay_draws(payload):
    _keys(
        payload,
        {"schema_version", "comparison", "target", "count", "seed", "query", "arrays", "digest"},
        "hypothesis draws",
    )
    if (
        payload["schema_version"] != DRAW_SCHEMA
        or not isinstance(payload["target"], str)
        or payload["target"] not in {"null", "alternative"}
    ):
        _error("Unknown hypothesis draw schema or target.")
    count = integer(payload["count"], "draw count", high=10000)
    seed = integer(payload["seed"], "seed", low=0)
    state = payload["comparison"]
    _comparison_envelope(state)
    _, k, _, _ = _posterior_raw(state["alternative"])
    if payload["query"] is not None:
        _keys(
            payload["query"],
            {
                "columns",
                "dtypes",
                "values",
                "index",
                "sample_positions",
                "missing",
                "rows_original",
            },
            "saved query",
        )
    q = (
        0
        if payload["query"] is None
        else integer(payload["query"]["rows_original"], "query original rows", high=MAX_QUERIES)
    )
    _admit_comparison(
        state["alternative"],
        len(state["constraints"]),
        draws=count,
        queries=q,
        metadata=metadata_bytes(payload),
    )
    _query_envelope(payload["query"], state)
    arrays = payload["arrays"]
    _keys(arrays, {"beta", "sigma_squared", "mean_draws", "outcome_draws"}, "joint draw arrays")
    _matrix(arrays["beta"], count, k, "beta draws")
    _vector(arrays["sigma_squared"], count, "variance draws")
    if payload["query"] is not None and not isinstance(
        payload["query"]["sample_positions"], (list, tuple)
    ):
        _error("Saved query positions must be a resident integer array.")
    positions = 0 if payload["query"] is None else len(payload["query"]["sample_positions"])
    if not 0 <= positions <= q:
        _error("Saved query sample count exceeds its original row admission.")
    for key in ("mean_draws", "outcome_draws"):
        if payload["query"] is None:
            if arrays[key] is not None:
                _error("Draw outcomes require a complete saved query source.")
        else:
            _matrix(arrays[key], count, positions, key)
    state = _replay(state)
    query = _query_replay(payload["query"], state)
    computed = draw_arrays(
        state["results"]["null_posterior"],
        state["alternative"],
        payload["target"],
        count,
        seed,
        query,
    )
    if digest(arrays) != digest(computed):
        _error("Saved draws disagree with their complete posterior, query, count and local seed.")
    if payload["digest"] != digest(
        {key: value for key, value in payload.items() if key != "digest"}
    ):
        _error("Saved joint draw digest disagrees with its complete state.")
    return payload


class PosteriorHypothesisDraws(BaseModel):
    """Immutable bounded exact draws, including zero-dimensional constrained coefficients."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True, revalidate_instances="always"
    )
    schema_version: Literal["openecon.posterior_hypothesis_draws.v1"] = DRAW_SCHEMA
    payload: Any

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        """Admit the complete encoded draw record before decoding or local seeded replay."""
        _json_input_admission(json_data)
        return super().model_validate_json(json_data, **kwargs)

    def __deepcopy__(self, memo=None):
        _deep_copy_admission(self)
        if self.schema_version != DRAW_SCHEMA:
            _error("Unknown hypothesis draw schema version.")
        _replay_draws(self.payload)
        return super().__deepcopy__(memo)

    def model_dump(self, **kwargs):
        """Validate exact source/target draws before any portable copy or field exclusion."""
        if self.schema_version != DRAW_SCHEMA:
            _error("Unknown hypothesis draw schema version.")
        _replay_draws(self.payload)
        return super().model_dump(**kwargs)

    def model_dump_json(self, *, indent=None, **kwargs):
        """Budget complete indented output before exact draw replay and serialization."""
        _serialization_admission(self, indent)
        if self.schema_version != DRAW_SCHEMA:
            _error("Unknown hypothesis draw schema version.")
        _replay_draws(self.payload)
        return super().model_dump_json(indent=indent, **kwargs)

    @model_validator(mode="after")
    def validate_semantics(self):
        """Replay exact target/seed draws and freeze all original source and query metadata."""
        if self.schema_version != DRAW_SCHEMA:
            _error("Unknown hypothesis draw schema version.")
        object.__setattr__(self, "payload", _freeze(_replay_draws(self.payload)))
        return self

    @field_serializer("payload")
    def serialize_payload(self, value):
        """Replay before copying any draw cache into portable output."""
        if self.schema_version != DRAW_SCHEMA:
            _error("Unknown hypothesis draw schema version.")
        return _thaw(_replay_draws(value))

    def summary(self) -> pd.DataFrame:
        """Report exact-draw counts and constrained dimensions without MCMC convergence claims."""
        if self.schema_version != DRAW_SCHEMA:
            _error("Unknown hypothesis draw schema version.")
        state = _replay_draws(self.payload)
        return pd.DataFrame(
            {
                "Target": [state["target"]],
                "Exact joint draws": [state["count"]],
                "Seed": [state["seed"]],
                "Constraint rank": [len(state["comparison"]["constraints"])],
                "Free null coefficients": [
                    state["comparison"]["results"]["null_posterior"]["free_dimension"]
                ],
            }
        )


def bayes_hypothesis_draws(
    *,
    result: PosteriorHypothesisComparison | Mapping | str,
    target: str = "null",
    draws: int = 1000,
    seed: int = 0,
    data: Any = None,
    missing: str = "raise",
) -> PosteriorHypothesisDraws:
    """Draw exact joint beta/sigma² and optional outcomes under the specified proper null/alternative.

    A local CPU generator preserves ambient RNG. All coefficients fixed is a
    valid zero-free-dimensional affine posterior; variance and outcome draws
    remain stochastic. Each complete saved draw record replays its target/seed.
    """
    if not isinstance(target, str) or target not in {"null", "alternative"}:
        _error("target must be null or alternative.", "invalid_option")
    count, seed = integer(draws, "draws", high=10000), integer(seed, "seed", low=0)
    if not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("Query missing policy must be raise or drop.", "invalid_option")
    raw = _raw_state(result)
    _comparison_envelope(raw)
    q = 0 if data is None else resident(data, list(raw["alternative"]["state"]["predictors"]))
    _admit_comparison(raw["alternative"], len(raw["constraints"]), draws=count, queries=q)
    state = _checked(result)
    design, query = None, None
    if data is not None:
        matrix, _, query = _query(state, data, missing)
        design = matrix.tolist()
    arrays = draw_arrays(
        state["results"]["null_posterior"], state["alternative"], target, count, seed, design
    )
    payload = {
        "schema_version": DRAW_SCHEMA,
        "comparison": state,
        "target": target,
        "count": count,
        "seed": seed,
        "query": query,
        "arrays": arrays,
    }
    return PosteriorHypothesisDraws(payload=payload | {"digest": digest(payload)})


def bayes_hypothesis_draws_restore(
    *, saved: PosteriorHypothesisDraws | Mapping | str
) -> PosteriorHypothesisDraws:
    """Restore an exact-draw record by complete source, affine posterior and seed replay."""
    if isinstance(saved, PosteriorHypothesisDraws):
        if saved.schema_version != DRAW_SCHEMA:
            _error("Unknown hypothesis draw schema version.")
        payload = saved.payload
    else:
        payload = load_mapping(saved)
        if set(payload) == {"schema_version", "payload"}:
            if payload["schema_version"] != DRAW_SCHEMA:
                _error("Unknown hypothesis draw schema version.")
            payload = payload["payload"]
    return PosteriorHypothesisDraws(payload=_replay_draws(payload))
