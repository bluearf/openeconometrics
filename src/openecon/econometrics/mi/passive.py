"""Declared deterministic derivation of saved MI completions, without FCS feedback."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from numbers import Real
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import (
    MIResult, MAX_COLUMNS, MAX_IMPUTATIONS, MAX_ROWS, _decode_index, _digest,
    _metadata_bytes, admit, integer, make_result,
)
from openecon.resources import plan_workspace


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError("invalid_passive_recipe", "Recipe constants must be finite real numbers, not booleans.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise AnalysisError("invalid_passive_recipe", "Recipe constants must fit finite float64 values.") from error
    if not math.isfinite(number):
        raise AnalysisError("invalid_passive_recipe", "Recipe constants must be finite real numbers, not booleans.")
    return number


def _json_mapping(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 16384:
        raise ValueError(f"Passive {name} must be a bounded JSON string.")
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Passive {name} must encode a JSON object.")
    return parsed


def _recipes(value, columns):
    if (not isinstance(columns, (list, tuple)) or not 1 <= len(columns) <= MAX_COLUMNS
            or any(not isinstance(name, str) or not name for name in columns)
            or len(set(columns)) != len(columns)):
        raise AnalysisError("invalid_imputations", "Source columns must be distinct nonempty names.")
    if not isinstance(value, Mapping) or not 1 <= len(value) <= MAX_COLUMNS:
        raise AnalysisError("invalid_passive_recipe", "Declare 1..16 named recipes in dependency order.")
    if len(columns) + len(value) > MAX_COLUMNS:
        raise AnalysisError("mi_shape_limit", "Source and derived columns together must not exceed 16.")
    available = set(columns)
    output = {}
    for name, recipe in value.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 256 or name in available:
            raise AnalysisError("invalid_passive_recipe", "Derived names must be new, nonempty labels of at most 256 characters.")
        if not isinstance(recipe, Mapping):
            raise AnalysisError("invalid_passive_recipe", "Each recipe must be an explicit mapping, not an expression.")
        operation = recipe.get("operation")
        keys = {"affine": {"operation", "inputs", "coefficients", "intercept"},
                "product": {"operation", "inputs"},
                "power": {"operation", "inputs", "exponent"}}
        if not isinstance(operation, str) or operation not in keys or set(recipe) != keys[operation]:
            raise AnalysisError("invalid_passive_recipe", "Use exactly the documented affine/product/power recipe fields.")
        inputs = recipe["inputs"]
        if (isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence)
                or not 1 <= len(inputs) <= MAX_COLUMNS
                or any(not isinstance(v, str) or v not in available for v in inputs)):
            raise AnalysisError("invalid_passive_recipe", "Inputs must refer to source or previously derived columns; cycles/forward references are unsupported.")
        normalized = {"operation": operation, "inputs": list(inputs)}
        if operation == "affine":
            coefficients = recipe["coefficients"]
            if (isinstance(coefficients, (str, bytes)) or not isinstance(coefficients, Sequence)
                    or len(coefficients) != len(inputs)):
                raise AnalysisError("invalid_passive_recipe", "Affine coefficients must match the declared inputs.")
            normalized.update(coefficients=[_number(v) for v in coefficients],
                              intercept=_number(recipe["intercept"]))
        elif operation == "power":
            if len(inputs) != 1:
                raise AnalysisError("invalid_passive_recipe", "A power recipe needs exactly one input.")
            normalized["exponent"] = integer(recipe["exponent"], "exponent", minimum=1, maximum=8)
        else:
            if len(inputs) < 2:
                raise AnalysisError("invalid_passive_recipe", "A product needs at least two explicit factors.")
        output[name] = normalized
        available.add(name)
    return output


def _plan(source, count, max_work):
    if isinstance(source, MIResult):
        source_state = source.__dict__
    elif isinstance(source, Mapping):
        source_state = source
    else:
        raise AnalysisError("invalid_imputations", "Supply a saved MIResult.")
    arrays = tuple(source_state.get(key) for key in ("original", "columns", "completed_matrices"))
    if any(not isinstance(value, (list, tuple)) for value in arrays):
        raise AnalysisError("invalid_imputations", "Source MI dimensions need bounded saved sequences.")
    original, columns, completions = arrays
    n, p, m = map(len, arrays)
    if not 1 <= n <= MAX_ROWS or not 1 <= p < p + count <= MAX_COLUMNS or not 1 <= m <= MAX_IMPUTATIONS:
        raise AnalysisError("mi_shape_limit", "Passive MI dimensions exceed their declared limits.")
    if (any(not isinstance(row, (list, tuple)) or len(row) != p for row in original)
            or any(not isinstance(matrix, (list, tuple)) or len(matrix) != n
                   or any(not isinstance(row, (list, tuple)) or len(row) != p for row in matrix)
                   for matrix in completions)):
        raise AnalysisError("invalid_imputations", "Source MI matrix shapes disagree.")
    metadata = source_state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise AnalysisError("invalid_imputations", "Source MI metadata must be a saved mapping.")
    max_work = integer(max_work, "max_work", minimum=1)
    work = (m + 2) * n * (p + count) * (16 * count + 4)
    if work > max_work:
        raise AnalysisError("mi_work_limit", f"Passive derivation work {work:,} exceeds max_work={max_work:,}.")
    plan = plan_workspace("post-MI passive derivation", {
        "source and extended numeric work": 8 * (m + 3) * n * (p + count) * 8,
        "source and extended immutable state/JSON": (m + 2) * n * (p + count) * 640 + n * 16384,
        "source metadata validation and serialized copies": 8 * _metadata_bytes(metadata),
        "recipe and mask work": n * (p + count) * 16 + 16384,
    })
    return work, plan.record()


def _expand(matrix, columns, recipes):
    """All numerical transformations use explicit CPU float64 Torch buffers."""
    current = matrix
    names = list(columns)
    for name, recipe in recipes.items():
        selected = current[..., [names.index(v) for v in recipe["inputs"]]]
        present = ~torch.isnan(selected).any(-1)
        if recipe["operation"] == "affine":
            coefficients = torch.tensor(recipe["coefficients"], dtype=torch.float64, device="cpu")
            terms = selected * coefficients
            lost = (terms == 0) & (selected != 0) & (coefficients != 0)
            if bool((lost.any(-1) & present).any()):
                raise AnalysisError("numerical_failure", "A passive affine term underflowed to zero; rescale the declared formula.")
            derived = terms.sum(-1) + recipe["intercept"]
        elif recipe["operation"] == "product":
            derived = selected.prod(-1)
            if bool(((derived == 0) & (selected != 0).all(-1) & present).any()):
                raise AnalysisError("numerical_failure", "A passive product underflowed to zero; rescale the declared formula.")
        else:
            derived = selected[..., 0].pow(recipe["exponent"])
            if bool(((derived == 0) & (selected[..., 0] != 0) & present).any()):
                raise AnalysisError("numerical_failure", "A passive power underflowed to zero; rescale the declared formula.")
        # Original nulls propagate through every declared dependency. No clipping.
        if not bool(torch.isfinite(derived[present]).all()):
            raise AnalysisError("numerical_failure", "A passive recipe overflowed; rescale the declared formula.")
        derived = torch.where(present, derived, torch.full_like(derived, float("nan")))
        current = torch.cat((current, derived.unsqueeze(-1)), dim=-1)
        names.append(name)
    return current


def _original_tensor(source):
    return torch.tensor([[float("nan") if v is None else v for v in row]
                         for row in source.original], dtype=torch.float64, device="cpu")


def _validated_source(source):
    if not isinstance(source.metadata, Mapping):
        raise AnalysisError("invalid_imputations", "Source MI metadata must be a saved mapping.")
    state = source.model_dump()
    operation = source.metadata.get("operation")
    if operation == "fixed_single_target_pattern_mixture":
        from openecon.econometrics.mi.sensitivity import MIDeltaResult
        return MIDeltaResult.model_validate(state)
    if source.metadata.get("api") == "mi_discrete" or operation == "discrete_fcs":
        from openecon.econometrics.mi.discrete import MIDiscreteResult
        return MIDiscreteResult.model_validate(state)
    return MIResult.model_validate(state)


class MIPassiveResult(BaseModel):
    """Source, formula DAG and derived full state; restoration recomputes every cell."""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["mi-passive-v1"] = "mi-passive-v1"
    source: MIResult
    result: MIResult
    recipes_json: str = Field(max_length=16384)
    metadata_json: str = Field(max_length=16384)
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def admission(cls, state):
        if not isinstance(state, Mapping):
            return state
        recipes_state = _json_mapping(state.get("recipes_json"), "recipes")
        meta = _json_mapping(state.get("metadata_json"), "metadata")
        if not isinstance(meta.get("resource_plan"), Mapping):
            raise ValueError("Passive resource plan must be a saved mapping.")
        source = state.get("source")
        columns = source.columns if isinstance(source, MIResult) else source.get("columns", ()) if isinstance(source, Mapping) else ()
        recipes = _recipes(recipes_state, columns)
        _plan(source, len(recipes), meta.get("max_work"))
        return state

    @model_validator(mode="after")
    def scientific_identity(self):
        # Explicit dump/reload also refuses forged model_construct instances.
        source = _validated_source(self.source)
        # Extension validators must survive nesting through the MIResult field.
        # This changes only the validated Python type; persisted fields are equal.
        object.__setattr__(self, "source", source)
        result = MIResult.model_validate(self.result.model_dump())
        recipes = _recipes(json.loads(self.recipes_json), source.columns)
        meta = json.loads(self.metadata_json)
        work, plan = _plan(source, len(recipes), meta.get("max_work"))
        if (meta.get("operation") != "post_MI_passive_transform" or meta.get("fcs_feedback") is not False
                or meta.get("substantive_model_compatibility_claim") is not False
                or meta.get("new_stochastic_draws") != 0 or type(meta.get("new_stochastic_draws")) is not int
                or meta.get("source_integrity_sha256") != source.integrity_sha256
                or meta.get("source_result_class") != type(source).__name__
                or meta.get("work_estimate") != work):
            raise ValueError("Passive operation, source, resource or scientific-claim metadata disagrees.")
        saved_plan = meta.get("resource_plan", {})
        if (any(saved_plan.get(k) != v for k, v in plan.items() if k != "budget_bytes")
                or type(saved_plan.get("budget_bytes")) is not int
                or saved_plan["budget_bytes"] < plan["estimated_workspace_bytes"]):
            raise ValueError("Passive resource plan differs from the admitted dimensions.")
        if (result.columns != (*source.columns, *recipes)
                or result.method != source.method or result.seed != source.seed
                or result.metadata["index"] != source.metadata["index"]
                or result.metadata["imputation_seeds"] != source.metadata["imputation_seeds"]
                or result.metadata.get("operation") != "post_MI_passive_transform"
                or result.metadata.get("source_integrity_sha256") != source.integrity_sha256
                or result.metadata.get("fcs_feedback") is not False
                or result.metadata.get("substantive_model_compatibility_claim") is not False):
            raise ValueError("Passive derivation changed source columns, identity or imputation seeds.")
        expected_original = _expand(_original_tensor(source), source.columns, recipes)
        expected_completed = _expand(torch.tensor(source.completed_matrices, dtype=torch.float64, device="cpu"), source.columns, recipes)
        original = [[None if math.isnan(v) else v for v in row] for row in expected_original.tolist()]
        if result.original != tuple(tuple(row) for row in original) or result.completed_matrices != tuple(tuple(tuple(row) for row in matrix) for matrix in expected_completed.tolist()):
            raise ValueError("Passive derived cells disagree with the saved source/formula DAG.")
        if _digest(self.model_dump(mode="json", exclude={"integrity_sha256"})) != self.integrity_sha256:
            raise ValueError("Passive full-state integrity mismatch.")
        return self

    @property
    def recipes(self):
        return json.loads(self.recipes_json)

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    def model_copy(self, *, update=None, deep=False):
        self._replay_admission()
        state = self.model_dump(mode="json")
        state.update(update or {})
        return type(self).model_validate(state)

    def dataset(self, imputation: int = 1):
        """Return a fresh expanded completion, retaining the original physical index."""
        self._replay_admission()
        return self.result.dataset(imputation)

    def _replay_admission(self):
        recipes = _recipes(_json_mapping(self.recipes_json, "recipes"), self.source.columns)
        meta = _json_mapping(self.metadata_json, "metadata")
        _plan(self.source, len(recipes), meta.get("max_work"))

    @property
    def table(self):
        self._replay_admission()
        from openecon.econometrics.core import table
        n = len(self.source.original)
        missing = dict(zip(self.result.columns, self.result.metadata["missing_counts"]))
        return table([{"column": name, "operation": recipe["operation"],
                       "observed": n - missing[name], "missing": missing[name],
                       "imputations": len(self.source.completed_matrices)}
                      for name, recipe in self.recipes.items()],
                     method="mi_passive", nobs=n, inference="deterministic derivation; no new stochastic draws")

    @property
    def latex(self):
        return self.table.latex

    def to_latex(self, **options):
        return self.table.to_latex(**options)


def mi_passive(source, recipes, *, max_work: int = 100_000_000):
    """Derive affine/product/integer-power columns from a saved MIResult.

    Named recipe mappings must be in dependency order, with exactly documented
    fields. At most16 total columns,10000 rows and100 imputations are admitted.
    This is deterministic post-MI derivation; it does not feed values back into
    FCS or establish imputation/analysis-model compatibility. Observed nulls
    propagate through all declared inputs, including zero affine coefficients.
    Nonzero multiplicative terms/results that underflow to zero are refused.
    """
    if not isinstance(source, MIResult):
        raise AnalysisError("invalid_imputations", "Supply a saved MIResult, not a fitted model or Dataset.")
    recipes = _recipes(recipes, source.columns)
    work, plan = _plan(source, len(recipes), max_work)
    source = _validated_source(source)
    original = _expand(_original_tensor(source), source.columns, recipes)
    completed = _expand(torch.tensor(source.completed_matrices, dtype=torch.float64, device="cpu"), source.columns, recipes)
    names = [*source.columns, *recipes]
    frame = pd.DataFrame(original.tolist(), columns=names, index=_decode_index(source.metadata["index"]))
    selected, values, missing, metadata = admit(frame, names, m=len(completed), iterations=1, max_work=max_work)
    metadata.update(operation="post_MI_passive_transform", source_integrity_sha256=source.integrity_sha256,
                    fcs_feedback=False, substantive_model_compatibility_claim=False)
    result = make_result(source.method, selected, values, missing, completed, source.seed,
                         metadata, source.metadata["imputation_seeds"])
    meta = {"operation": "post_MI_passive_transform", "source_integrity_sha256": source.integrity_sha256,
            "source_result_class": type(source).__name__,
            "fcs_feedback": False, "substantive_model_compatibility_claim": False,
            "new_stochastic_draws": 0, "work_estimate": work, "max_work": max_work,
            "resource_plan": plan}
    # Recipe insertion order is the declared DAG; sorting its keys changes meaning.
    state = {"source": source.model_dump(), "result": result.model_dump(),
             "recipes_json": json.dumps(recipes, allow_nan=False, separators=(",", ":")),
             "metadata_json": json.dumps(meta, sort_keys=True, allow_nan=False, separators=(",", ":")),
             "schema_version": "mi-passive-v1"}
    return MIPassiveResult(**state, integrity_sha256=_digest(state))
