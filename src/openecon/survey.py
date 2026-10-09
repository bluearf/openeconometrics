"""Bounded single-stage survey declarations; no survey estimator is implied."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from openecon.analysis_contracts import AnalysisError


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SurveyStratum(_Record):
    """Geometry only: labels and row-level weights are not stored."""

    nobs: StrictInt = Field(gt=0)
    n_psu: StrictInt = Field(gt=0)
    population_psu: StrictInt | None = Field(default=None, gt=0)
    certainty: bool = Field(strict=True)


class SurveyValidation(_Record):
    nobs: StrictInt = Field(gt=0)
    n_strata: StrictInt = Field(gt=0)
    n_psu: StrictInt = Field(gt=0)
    design_df: StrictInt = Field(ge=0)
    sum_weights: float = Field(gt=0, allow_inf_nan=False, strict=True)
    design_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[SurveyStratum, ...]

    @model_validator(mode="after")
    def geometry(self):
        if (len(self.strata) != self.n_strata
                or sum(s.nobs for s in self.strata) != self.nobs
                or sum(s.n_psu for s in self.strata) != self.n_psu
                or self.design_df != self.n_psu - self.n_strata):
            raise ValueError("Survey validation geometry is inconsistent.")
        for s in self.strata:
            if (s.n_psu > s.nobs or s.population_psu is not None
                    and s.population_psu < s.n_psu
                    or s.certainty != (s.population_psu == s.n_psu)):
                raise ValueError("Survey stratum geometry is inconsistent.")
        return self


class SurveyDesign(_Record):
    """Immutable JSON declaration bound to all selected design rows in order.

    Restore with ``SurveyDesign.model_validate_json`` and call ``revalidate``
    before use. A serialized record is not an authenticated proof of sampling.
    """

    schema_version: Literal["survey-design-v1"] = "survey-design-v1"
    stages: Literal[1] = 1
    weights: str
    psu: str
    strata: str | None = None
    fpc: str | None = None
    singleton: Literal["raise", "certainty"] = "raise"
    max_rows: StrictInt = Field(default=1_000_000, ge=1, le=1_000_000)
    max_memory_mb: StrictInt = Field(default=64, ge=1, le=512)
    validation: SurveyValidation

    @field_validator("weights", "psu", "strata", "fpc", mode="before")
    @classmethod
    def column_name(cls, value):
        if value is not None and (not isinstance(value, str)
                                  or not value.strip() or len(value) > 200):
            raise ValueError("Design roles need nonempty column names of at most 200 characters.")
        return value

    @field_validator("stages", mode="before")
    @classmethod
    def one_stage(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Only explicit single-stage survey declarations are supported.")
        return value

    @model_validator(mode="after")
    def distinct_roles(self):
        roles = [v for v in (self.weights, self.psu, self.strata, self.fpc) if v is not None]
        if len(roles) != len(set(roles)):
            raise ValueError("Design roles must use distinct columns.")
        return self

    def revalidate(self, data: Any) -> SurveyValidation:
        """Verify declaration, geometry and exact design inputs without dropping rows."""
        current = _validate(data, **self.model_dump(exclude={"validation", "schema_version", "stages"}))
        if current != self.validation:
            raise AnalysisError("survey_design_changed", "Design inputs or saved geometry changed; declare the survey again.")
        return current


def _scalar(value):
    if (type(value).__module__.startswith("numpy") and type(value).__name__ != "ndarray"
            and hasattr(value, "item") and callable(value.item)):
        value = value.item()
    return value


def _identity(value):
    value = _scalar(value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Integral):
        if int(value).bit_length() > 800:
            raise AnalysisError("invalid_survey_id", "Integer PSU/strata IDs exceed the bounded scalar encoding.")
        return ("int", int(value))
    if isinstance(value, Real) and math.isfinite(value):
        return ("float", float(value).hex())
    if isinstance(value, str) and value and len(value) <= 256:
        return ("str", value)
    raise AnalysisError("invalid_survey_id", "PSU/strata IDs must be nonmissing finite scalar strings (<=256 characters), numbers or booleans.")


def _number(value, *, population=False):
    value = _scalar(value)
    code = "invalid_survey_fpc" if population else "invalid_survey_weights"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError(code, "FPC counts and sampling weights must be positive finite real numbers.")
    try:
        finite = math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite or value <= 0:
        raise AnalysisError(code,
                            "FPC counts and sampling weights must be positive finite real numbers.")
    if population:
        if int(value) != value or value > 2**53:
            raise AnalysisError("invalid_survey_fpc", "FPC must be an exact population PSU count, at most 2**53; fractions are unsupported.")
        return int(value)
    return float(value)


def _validate(data, *, weights, psu, strata, fpc, singleton, max_rows, max_memory_mb):
    import pandas as pd

    # Validate scalar options before converting user input or allocating group maps.
    if type(max_rows) is not int or not 1 <= max_rows <= 1_000_000:
        raise AnalysisError("survey_budget", "max_rows must be an integer in 1..1,000,000.")
    if type(max_memory_mb) is not int or not 1 <= max_memory_mb <= 512:
        raise AnalysisError("survey_budget", "max_memory_mb must be an integer in 1..512.")
    if not isinstance(singleton, str) or singleton not in {"raise", "certainty"}:
        raise AnalysisError("invalid_survey_design", "singleton must be 'raise' or 'certainty'.")
    roles = [v for v in (weights, psu, strata, fpc) if v is not None]
    if (weights is None or psu is None or any(not isinstance(v, str) or not v.strip()
                                             or len(v) > 200 for v in roles)
            or len(roles) != len(set(roles))):
        raise AnalysisError("invalid_survey_design", "Design roles need distinct nonempty column names of at most 200 characters.")
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, Mapping):
        if any(type(v).__module__.startswith("torch") for v in data.values()):
            raise AnalysisError("unsupported_survey_input", "Tensor/device columns are unsupported; provide resident tabular columns.")
        if any(v not in data for v in roles):
            raise AnalysisError("missing_column", "Each declared design column must exist.")
        selected = {v: data[v] for v in roles}
        if any(hasattr(v, "__len__") and len(v) > max_rows for v in selected.values()):
            raise AnalysisError("survey_budget", "The complete declaration exceeds max_rows.")
        if any(hasattr(v, "__len__") and len(v) * 1024 > max_memory_mb * 1024**2
               for v in selected.values()):
            raise AnalysisError("survey_budget", "Design rows exceed the indexing admission budget.")
        try:
            frame = pd.DataFrame(selected)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_data", "Survey columns need consistent lengths and scalar values.") from exc
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) > max_rows:
            raise AnalysisError("survey_budget", "The complete declaration exceeds max_rows.")
        if len(data) * 1024 > max_memory_mb * 1024**2:
            raise AnalysisError("survey_budget", "Design rows exceed the indexing admission budget.")
        if not all(isinstance(row, Mapping) for row in data):
            raise AnalysisError("invalid_data", "Survey row records must be mappings.")
        if any(v not in row for row in data for v in roles):
            raise AnalysisError("missing_column", "Each row must contain every design column.")
        try:
            frame = pd.DataFrame([{v: row[v] for v in roles} for row in data])
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_data", "Survey columns need scalar values.") from exc
    else:
        raise AnalysisError("unsupported_survey_input", "Use a resident DataFrame or column/row mapping; Dataset/device inputs are unsupported.")
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Survey data needs unique column names.")
    if not len(frame):
        raise AnalysisError("empty_data", "A survey declaration needs observations.")
    if len(frame) > max_rows:
        raise AnalysisError("survey_budget", "The complete declaration exceeds max_rows.")
    if any(v not in frame for v in roles):
        raise AnalysisError("missing_column", "Each declared design column must exist.")
    # Bound projected storage plus conservative per-row group/hash admission.
    planned = sum(int(frame[v].memory_usage(index=False, deep=True)) for v in roles) + len(frame) * 1024
    if planned > max_memory_mb * 1024**2:
        raise AnalysisError("survey_budget", "Complete design validation exceeds max_memory_mb; increase the explicit budget or use a smaller declared sample.")
    digest = hashlib.sha256()
    digest.update(json.dumps([(v, str(frame[v].dtype)) for v in roles], ensure_ascii=False).encode())
    for v in roles:
        dtype = frame[v].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            digest.update(json.dumps(["categories", v, dtype.ordered]).encode())
            for label in dtype.categories:
                digest.update(json.dumps(_identity(label), ensure_ascii=False).encode())
                digest.update(b"\n")
    groups = {}
    weight_values = []
    for row in frame[roles].itertuples(index=False, name=None):
        values = dict(zip(roles, row))
        w = _number(values[weights])
        h = _identity(values[strata]) if strata else ("unstratified",)
        p = _identity(values[psu])
        population = _number(values[fpc], population=True) if fpc else None
        record = groups.setdefault(h, {"psus": set(), "nobs": 0, "population": population})
        if record["population"] != population:
            raise AnalysisError("invalid_survey_fpc", "Population PSU counts must be constant within every stratum.")
        record["psus"].add(p)
        record["nobs"] += 1
        weight_values.append(w)
        digest.update(json.dumps([h, p, w.hex(), population], ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    records = []
    for record in groups.values():
        n = len(record["psus"])
        population = record["population"]
        if population is not None and population < n:
            raise AnalysisError("invalid_survey_fpc", "Population PSU counts cannot be smaller than sampled PSU counts.")
        certainty = population == n
        if n == 1 and (singleton != "certainty" or not certainty):
            raise AnalysisError("singleton_survey_stratum", "Singleton strata require singleton='certainty' and an explicit census FPC count of one.")
        records.append(SurveyStratum(nobs=record["nobs"], n_psu=n, population_psu=population, certainty=certainty))
    try:
        sum_weights = math.fsum(weight_values)
    except OverflowError as exc:
        raise AnalysisError("invalid_survey_weights", "Sampling weight sum overflows float64.") from exc
    if not math.isfinite(sum_weights):
        raise AnalysisError("invalid_survey_weights", "Sampling weight sum overflows float64.")
    n_psu = sum(r.n_psu for r in records)
    return SurveyValidation(nobs=len(frame), n_strata=len(records), n_psu=n_psu,
                            design_df=n_psu - len(records), sum_weights=sum_weights,
                            design_input_sha256=digest.hexdigest(), strata=tuple(records))


def survey_design(data: Any, *, weights: str, psu: str, strata: str | None = None,
                  fpc: str | None = None, singleton: str = "raise",
                  max_rows: int = 1_000_000, max_memory_mb: int = 64) -> SurveyDesign:
    """Declare and validate a bounded resident single-stage sampling design.

    All design rows remain in the declaration. PSU labels are nested in strata
    and scalar types stay distinct. No design-based estimates/SEs are produced.

    Parameters
    ----------
    data : resident DataFrame, column mapping or list of row records.
    weights : column of strictly positive finite sampling weights.
    psu : column of primary sampling-unit IDs, nested within strata.
    strata : optional stratum column; None declares one unstratified design.
    fpc : optional stratum-constant integer population PSU counts, not fractions.
    singleton : 'raise' or 'certainty'; certainty requires explicit census FPC=1.
    max_rows : complete-input admission limit, at most 1,000,000 rows.
    max_memory_mb : projected-storage and conservative indexing/hash admission
        budget in 1..512 MiB; excludes caller input and total process RSS.

    Returns
    -------
    SurveyDesign with immutable declaration and geometry. JSON restore must be
    followed by revalidate(data). Hash covers ordered design columns and dtypes,
    not unrelated outcome values, index labels or an authenticated survey design.
    """
    options = dict(weights=weights, psu=psu, strata=strata, fpc=fpc, singleton=singleton,
                   max_rows=max_rows, max_memory_mb=max_memory_mb)
    validation = _validate(data, **options)
    return SurveyDesign(**options, validation=validation)
