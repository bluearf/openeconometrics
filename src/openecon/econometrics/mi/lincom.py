"""Saved, marginal Rubin inference for declared linear MI contrasts.

Every imputation's complete covariance is projected before pooling. Nulls enter
only the t statistic; they never change between-imputation variance or the
unshifted estimate and confidence interval. These are unadjusted marginal tests,
not D1 joint inference or simultaneous confidence intervals.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Integral
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call
from openecon.econometrics.mi.joint import (
    FLOAT, MAX_M, MAX_P, MIPoolResult, _shape, digest, mi_pool, tensors,
)
from openecon.econometrics.postest.multiple import _level
from openecon.engines.distributions import normal_sf, t_sf
from openecon.resources import plan_workspace

DEFAULT_MAX_WORK = 10_000_000
MAX_WORK = 1_000_000_000
MAX_METADATA = 131_072
MAX_LABEL = 256


def _pool_envelope(pool):
    """Read bounded dimensions/strings without dumping or coercing a pool."""
    if isinstance(pool, MIPoolResult):
        state = pool.__dict__
    elif isinstance(pool, Mapping):
        state = pool
    else:
        raise AnalysisError("invalid_pool", "Use a validated MIPoolResult, restored explicitly if saved.")
    qs, us = _shape(state.get("estimates")), _shape(state.get("covariances"))
    if (len(qs) != 2 or not 2 <= qs[0] <= MAX_M or not 1 <= qs[1] <= MAX_P
            or us != (*qs, qs[1])):
        raise AnalysisError("invalid_pool", "The source needs 2..100 imputations and 1..32 common terms.")
    for key, size in (("terms", qs[1]), ("imputation_ids", qs[0])):
        _names(state.get(key), size, key)
    for key, limit in (("metadata_json", MAX_METADATA), ("imputation_description", 8192)):
        text = state.get(key)
        if not isinstance(text, str) or not 1 <= len(text) <= limit:
            raise AnalysisError("invalid_pool", f"Source {key} exceeds its declared string budget.")
    return state, qs


def _names(names, count, label="names"):
    if (not isinstance(names, (list, tuple)) or len(names) != count
            or any(not isinstance(name, str) or not name.strip() or len(name) > MAX_LABEL
                   for name in names) or len(set(names)) != count):
        raise AnalysisError("invalid_labels", f"{label} needs {count} unique nonempty labels of at most 256 characters.")
    return tuple(names)


def _work_limit(max_work):
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or not 1 <= max_work <= MAX_WORK:
        raise AnalysisError("invalid_spec", "max_work must be an integer in [1, 1000000000].")
    return int(max_work)


def _admit(pool, weights, values, names, max_work):
    """Shape/type/work admission precedes every source copy and tensor buffer."""
    _, (m, p) = _pool_envelope(pool)
    shape = _shape(weights)
    if len(shape) != 2 or not 1 <= shape[0] <= p or shape[1] != p:
        raise AnalysisError("invalid_contrasts", "R must be a full-row-rank k-by-p matrix with 1<=k<=p<=32.")
    k = shape[0]
    if _shape(values) != (k,):
        raise AnalysisError("invalid_contrasts", "Supply one finite numeric null value per contrast.")
    _names(names, k)
    max_work = _work_limit(max_work)
    work = m * (k * p * p + k * k * p + k * p + k * k + p**3 + k**3) + p**3 + k**3
    if work > max_work:
        raise AnalysisError("resource_limit", f"Linear MI contrast work {work:,} exceeds max_work={max_work:,}.")
    plan = plan_workspace("multiple-imputation linear contrasts", {
        "source tensors, validated copies and serialized state": 192 * m * p * (p + 1),
        "projected tensors, pooled results and serialized state": 192 * m * k * (k + 1),
        "restriction and projection workspace": 512 * (p * p + k * k + k * p),
        # Canonical JSON escapes a supplementary Unicode character with two
        # six-byte surrogate escapes. Reserve both nested metadata envelopes.
        "bounded metadata, labels and inferential tables": 24 * MAX_METADATA + 8192 * (m + p + k),
    })
    return m, p, k, work, plan.record()


def _validated_pool(pool):
    # The surrounding admission has already checked dimensions and string sizes.
    # Explicit restoration also refuses unchecked model_construct/copy shortcuts.
    state = dict(pool.__dict__) if isinstance(pool, MIPoolResult) else dict(pool)
    return MIPoolResult.model_validate(state)


@torch.no_grad()
def _project(pool, weights, values):
    q, u, *_ = tensors(pool.estimates, pool.covariances)
    r = torch.as_tensor(weights, dtype=FLOAT, device="cpu")
    null = torch.as_tensor(values, dtype=FLOAT, device="cpu")
    if not torch.isfinite(r).all() or not torch.isfinite(null).all():
        raise AnalysisError("invalid_contrasts", "Contrast weights and null values must be finite.")
    scale = float(r.abs().max())
    if scale == 0:
        raise AnalysisError("singular_contrasts", "Contrast rows must be linearly independent and nonzero.")
    try:
        singular = torch.linalg.svdvals(r / scale)
        tolerance = torch.finfo(FLOAT).eps * max(r.shape) * singular[0]
        if not torch.isfinite(singular).all() or bool(singular[-1] <= tolerance):
            raise AnalysisError("singular_contrasts", "Contrast rows must be linearly independent and numerically resolved.")
        transformed_q = q @ r.T
        transformed_u = torch.matmul(torch.matmul(r.unsqueeze(0), u), r.T)
    except torch.linalg.LinAlgError as exc:
        raise AnalysisError("numerical_failure", "The linear contrast projection failed.") from exc
    if not torch.isfinite(transformed_q).all() or not torch.isfinite(transformed_u).all():
        raise AnalysisError("numerical_failure", "Contrast projection overflowed; rescale the declared estimands.")
    # Positive marginal variances and PSD checks use the existing pooling guard.
    # No diagonal-only approximation, covariance repair, or ridge is substituted.
    tensors(transformed_q, transformed_u)
    return transformed_q, transformed_u


def _metadata(source, work, max_work, workspace):
    return {
        "precision": "float64", "device": "cpu",
        "source_pool_integrity_sha256": source.integrity_sha256,
        "source_contract_preserved": True,
        "complete_df": source.complete_df,
        "covariance_projection": "R U_i R' for every imputation before Rubin pooling",
        "null_contract": "Nulls affect t statistics and p values only; estimates and intervals target R Q",
        "inference": "Marginal Rubin or Barnard-Rubin t inference with contrast-specific missing information",
        "multiple_testing_adjustment": "none", "simultaneous_intervals": False,
        "joint_test": False, "stata_parity_validated": False,
        "work_estimate": work, "max_work": max_work, "workspace": workspace,
    }


class MILincomResult(BaseModel):
    """Immutable source, projection and null-aware marginal MI inference state."""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["mi-lincom-v1"] = "mi-lincom-v1"
    source_pool: MIPoolResult
    weights: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    names: tuple[str, ...]
    transformed_pool: MIPoolResult
    max_work: int = Field(default=DEFAULT_MAX_WORK, ge=1, le=MAX_WORK)
    work_estimate: int = Field(gt=0)
    metadata_json: str = Field(min_length=1, max_length=MAX_METADATA)
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def raw_admission(cls, state):
        if not isinstance(state, Mapping):
            return state
        metadata = state.get("metadata_json")
        if not isinstance(metadata, str) or not 1 <= len(metadata) <= MAX_METADATA:
            raise AnalysisError("invalid_state", "Saved contrast metadata exceeds its bounded string budget.")
        m, _, k, work, _ = _admit(
            state.get("source_pool"), state.get("weights"), state.get("values"),
            state.get("names"), state.get("max_work", DEFAULT_MAX_WORK),
        )
        _, projected_shape = _pool_envelope(state.get("transformed_pool"))
        if projected_shape != (m, k):
            raise AnalysisError("invalid_state", "Saved projection dimensions disagree with the declared source and contrasts.")
        if type(state.get("work_estimate")) is not int or state.get("work_estimate") != work:
            raise AnalysisError("invalid_state", "Saved contrast work estimate disagrees with its dimensions.")
        admitted = dict(state)
        admitted["source_pool"] = _validated_pool(state.get("source_pool"))
        admitted["transformed_pool"] = _validated_pool(state.get("transformed_pool"))
        return admitted

    @model_validator(mode="after")
    def valid(self):
        _, _, _, work, workspace = _admit(
            self.source_pool, self.weights, self.values, self.names, self.max_work,
        )
        q, u = _project(self.source_pool, self.weights, self.values)
        target = self.transformed_pool
        tq, tu, *_ = tensors(target.estimates, target.covariances)
        if not torch.equal(q, tq) or not torch.equal(u, tu):
            raise ValueError("Saved contrast projection disagrees with the source's full per-imputation covariance.")
        if (target.terms != self.names or target.imputation_ids != self.source_pool.imputation_ids
                or target.complete_df != self.source_pool.complete_df
                or target.imputation_description != self.source_pool.imputation_description):
            raise ValueError("Saved contrast projection changed labels, imputation IDs or the source inference contract.")
        metadata = json.loads(self.metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("Saved contrast metadata must be an object.")
        recorded = metadata.get("workspace", {})
        budget = recorded.get("budget_bytes") if isinstance(recorded, dict) else None
        if type(budget) is not int or budget < workspace["estimated_workspace_bytes"]:
            raise ValueError("Saved contrast workspace budget does not cover its declared buffers.")
        workspace["budget_bytes"] = budget
        if metadata != _metadata(self.source_pool, work, self.max_work, workspace):
            raise ValueError("Saved contrast metadata disagrees with its scientific and resource contracts.")
        if digest(self.model_dump(mode="json", exclude={"integrity_sha256"})) != self.integrity_sha256:
            raise ValueError("MI linear contrast integrity mismatch.")
        # Validate null-aware inference as well as unshifted pooled inference.
        self.table
        return self

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    @property
    def alpha(self):
        return self.transformed_pool.alpha

    @property
    def table(self):
        """Unshifted estimates/intervals, with t and p for each declared null."""
        m, _, k, _, _ = _admit(self.source_pool, self.weights, self.values, self.names, self.max_work)
        _, projected_shape = _pool_envelope(self.transformed_pool)
        if projected_shape != (m, k):
            raise AnalysisError("invalid_state", "Saved projection dimensions disagree with the source.")
        result = _validated_pool(self.transformed_pool).table.copy()
        mean = torch.tensor(result["estimate"].tolist(), dtype=FLOAT, device="cpu")
        se = torch.tensor(result["std_error"].tolist(), dtype=FLOAT, device="cpu")
        null = torch.tensor(self.values, dtype=FLOAT, device="cpu")
        statistic = (mean - null) / se
        if not torch.isfinite(statistic).all():
            raise AnalysisError("numerical_failure", "The null-aware contrast statistic overflowed; rescale the estimand and null.")
        p_values = []
        for value, df in zip(statistic.tolist(), result["df"].tolist(), strict=True):
            p = 2 * (normal_sf(abs(value)) if df is None or math.isnan(df)
                     else kernel_call(t_sf, abs(value), df))
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise AnalysisError("numerical_failure", "Contrast tail probabilities are not finite probabilities.")
            p_values.append(p)
        result["null_value"] = list(self.values)
        result["statistic"] = statistic.tolist()
        result["p_value"] = p_values
        result.attrs.update({
            "source_pool_integrity_sha256": self.source_pool.integrity_sha256,
            "multiple_testing_adjustment": "none", "simultaneous_intervals": False,
            "joint_test": False, "stata_parity_validated": False,
        })
        return result

    @property
    def tables(self):
        contrasts = self.table
        projected = self.transformed_pool.tables
        return TableSet({"contrasts": contrasts,
                         **{key: value for key, value in projected.items() if key != "coefficients"}},
                        title="Multiple-imputation linear contrasts", **self.metadata)

    @property
    def latex(self):
        return self.tables.to_latex()

    def to_latex(self, **options):
        return self.tables.to_latex(**options)

    def model_copy(self, *, update=None, deep=False):
        """Revalidate source, projection and checksum instead of unchecked copies."""
        _admit(self.source_pool, self.weights, self.values, self.names, self.max_work)
        _pool_envelope(self.transformed_pool)
        state = self.model_dump(mode="json")
        state.update(update or {})
        return type(self).model_validate(state)


@torch.no_grad()
def mi_lincom(pool, R, *, values=None, names=None, alpha=None, max_work=DEFAULT_MAX_WORK):
    """Pool declared linear contrasts with full covariance and null-aware t tests.

    Pass a validated MIPoolResult and full-row-rank k-by-p numeric CPU matrix R,
    1<=k<=p<=32. Nulls default to zero; alpha defaults to the source pool's level.
    Each imputation is projected before pooling, preserving imputation IDs,
    complete_df and the source's declared scientific contract. Marginal intervals
    describe R Q; tests describe R Q=values. No multiplicity correction or joint
    test is implied. Bool/complex inputs and resource-limit bypasses are refused.
    """
    if not isinstance(pool, MIPoolResult):
        raise AnalysisError("invalid_pool", "mi_lincom requires a validated MIPoolResult.")
    _, (_, p) = _pool_envelope(pool)
    shape = _shape(R)
    if len(shape) != 2 or not 1 <= shape[0] <= p or shape[1] != p:
        raise AnalysisError("invalid_contrasts", "R must have k-by-p shape, 1<=k<=p<=32.")
    k = shape[0]
    values = [0.0] * k if values is None else values
    names = [f"contrast-{j + 1}" for j in range(k)] if names is None else names
    _, _, _, work, workspace = _admit(pool, R, values, names, max_work)
    source = _validated_pool(pool)
    alpha = source.alpha if alpha is None else alpha
    _level(alpha)
    q, u = _project(source, R, values)
    projected = mi_pool(q, u, terms=list(names), imputation_ids=list(source.imputation_ids),
                        complete_df=source.complete_df, alpha=alpha,
                        imputation_description=source.imputation_description)
    state = {
        "schema_version": "mi-lincom-v1", "source_pool": source.model_dump(mode="json"),
        "weights": torch.as_tensor(R, dtype=FLOAT, device="cpu").tolist(),
        "values": torch.as_tensor(values, dtype=FLOAT, device="cpu").tolist(),
        "names": list(names), "transformed_pool": projected.model_dump(mode="json"),
        "max_work": int(max_work), "work_estimate": work,
        "metadata_json": json.dumps(_metadata(source, work, int(max_work), workspace),
                                    sort_keys=True, allow_nan=False, separators=(",", ":")),
    }
    return MILincomResult(**state, integrity_sha256=digest(state))
