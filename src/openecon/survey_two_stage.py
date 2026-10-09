"""Exactly two sequential SRSWOR stages with explicit nested sampling geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.survey import _Record, _identity, _number


class TwoStageStratum(_Record):
    n_psu: StrictInt = Field(gt=0)
    population_psu: StrictInt = Field(gt=0, le=2**53)
    psu_indices: tuple[StrictInt, ...]


class TwoStagePSU(_Record):
    stratum_index: StrictInt = Field(ge=0)
    n_ssu: StrictInt = Field(gt=0)
    population_ssu: StrictInt = Field(gt=0, le=2**53)
    row_positions: tuple[StrictInt, ...]


class TwoStageValidation(_Record):
    nobs: StrictInt = Field(gt=0, le=1_000_000)
    n_strata: StrictInt = Field(gt=0, le=1_000_000)
    n_psu: StrictInt = Field(gt=0, le=1_000_000)
    design_df: StrictInt = Field(ge=0)
    sum_weights: float = Field(gt=0, allow_inf_nan=False, strict=True)
    design_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[TwoStageStratum, ...]
    psus: tuple[TwoStagePSU, ...]

    @model_validator(mode="before")
    @classmethod
    def bounded_geometry(cls, value):
        if isinstance(value, Mapping):
            n = value.get("nobs")
            if type(n) is not int or not 1 <= n <= 1_000_000:
                raise ValueError("Complete row count is outside the bounded design.")
            for key in ("strata", "psus"):
                records = value.get(key)
                if not isinstance(records, (tuple, list)) or not 1 <= len(records) <= n:
                    raise ValueError("Nested record dimensions exceed the complete sample.")
            count = 0
            for record in value["psus"]:
                rows = (
                    record.get("row_positions")
                    if isinstance(record, Mapping)
                    else getattr(record, "row_positions", None)
                )
                if not isinstance(rows, (tuple, list)):
                    raise ValueError("Each PSU requires bounded physical row positions.")
                count += len(rows)
                if count > n:
                    raise ValueError("Nested row records exceed the complete sample.")
        return value

    @model_validator(mode="after")
    def geometry(self):
        if (
            len(self.strata) != self.n_strata
            or len(self.psus) != self.n_psu
            or self.design_df != self.n_psu - self.n_strata
        ):
            raise ValueError("Two-stage dimension/df geometry is inconsistent.")
        seen_psus, seen_rows = [], []
        for h, s in enumerate(self.strata):
            if (
                len(s.psu_indices) != s.n_psu
                or s.population_psu < s.n_psu
                or s.n_psu == 1
                and s.population_psu != 1
                or list(s.psu_indices) != sorted(set(s.psu_indices))
            ):
                raise ValueError("Stage-one counts or census singleton are inconsistent.")
            for i in s.psu_indices:
                if not 0 <= i < self.n_psu or self.psus[i].stratum_index != h:
                    raise ValueError("Nested PSU assignment is inconsistent.")
            seen_psus.extend(s.psu_indices)
        if sorted(seen_psus) != list(range(self.n_psu)):
            raise ValueError("Each PSU must belong to exactly one stratum.")
        for p in self.psus:
            if (
                p.stratum_index >= self.n_strata
                or len(p.row_positions) != p.n_ssu
                or p.population_ssu < p.n_ssu
                or p.n_ssu == 1
                and p.population_ssu != 1
                or list(p.row_positions) != sorted(set(p.row_positions))
                or any(not 0 <= i < self.nobs for i in p.row_positions)
            ):
                raise ValueError(
                    "Stage-two counts, row assignment or census singleton are inconsistent."
                )
            seen_rows.extend(p.row_positions)
        if sorted(seen_rows) != list(range(self.nobs)):
            raise ValueError("Every physical row must be a distinct nested sampled SSU.")
        if not math.isclose(math.fsum(self.weights), self.sum_weights, rel_tol=1e-12):
            raise ValueError("Derived stage weight sum is inconsistent.")
        return self

    @property
    def weights(self):
        values = [0.0] * self.nobs
        for p in self.psus:
            s = self.strata[p.stratum_index]
            w = s.population_psu / s.n_psu * p.population_ssu / p.n_ssu
            for row in p.row_positions:
                values[row] = w
        return tuple(values)


class SurveyTwoStageDesign(_Record):
    """Frozen geometry; restore and revalidate before using original data.

    A checksum records inputs, rather than authenticating the sampling process.
    One physical row represents one unique sampled SSU within its PSU/stratum.
    """

    schema_version: Literal["survey-two-stage-design-v1"] = "survey-two-stage-design-v1"
    stages: Literal[2] = 2
    sampling: Literal["sequential-srswor"] = "sequential-srswor"
    psu: str
    ssu: str
    population_psu: str
    population_ssu: str
    strata: str | None = None
    max_rows: StrictInt = Field(default=1_000_000, ge=1, le=1_000_000)
    max_memory_mb: StrictInt = Field(default=64, ge=1, le=512)
    validation: TwoStageValidation

    @model_validator(mode="before")
    @classmethod
    def admit_before_nested_records(cls, raw):
        if isinstance(raw, Mapping):
            validation = raw.get("validation")
            n = (
                validation.nobs
                if isinstance(validation, TwoStageValidation)
                else (validation.get("nobs") if isinstance(validation, Mapping) else None)
            )
            memory = raw.get("max_memory_mb", 64)
            rows = raw.get("max_rows", 1_000_000)
            if (
                type(n) is not int
                or not 1 <= n <= 1_000_000
                or type(memory) is not int
                or not 1 <= memory <= 512
                or type(rows) is not int
                or not 1 <= rows <= 1_000_000
                or n > rows
            ):
                raise ValueError("Saved design dimensions exceed explicit admission limits.")
            admission(n, memory)
        return raw

    @field_validator("psu", "ssu", "population_psu", "population_ssu", "strata", mode="before")
    @classmethod
    def name(cls, value):
        if value is not None and (
            not isinstance(value, str) or not value.strip() or len(value) > 200
        ):
            raise ValueError(
                "Design roles require nonempty column names of at most 200 characters."
            )
        return value

    @field_validator("stages", mode="before")
    @classmethod
    def exactly_two(cls, value):
        if type(value) is not int or value != 2:
            raise ValueError("Exactly two SRSWOR stages are supported.")
        return value

    @model_validator(mode="after")
    def roles_and_budget(self):
        roles = [
            v
            for v in (self.psu, self.ssu, self.population_psu, self.population_ssu, self.strata)
            if v is not None
        ]
        if len(set(roles)) != len(roles) or self.validation.nobs > self.max_rows:
            raise ValueError("Design roles or row budget are inconsistent.")
        admission(self.validation.nobs, self.max_memory_mb)
        return self

    def revalidate(self, data: Any):
        current = _validate(
            data, **self.model_dump(exclude={"validation", "schema_version", "stages", "sampling"})
        )
        if current != self.validation:
            raise AnalysisError(
                "survey_design_changed",
                "Two-stage inputs or ordered geometry changed; declare again.",
            )
        return current


def admission(n, memory_mb, *, k=0):
    return plan_workspace(
        "two-stage survey inference",
        {
            "complete nested identities, indices and JSON replay": n * 1536,
            "target/score and serialization buffers": (n * k + k * k) * 192,
        },
        budget_bytes=min(memory_mb * 1024**2, workspace_budget_bytes()),
    )


def _validate(data, *, psu, ssu, population_psu, population_ssu, strata, max_rows, max_memory_mb):
    import pandas as pd

    if type(max_rows) is not int or not 1 <= max_rows <= 1_000_000:
        raise AnalysisError("survey_budget", "max_rows must be an integer in 1..1,000,000.")
    if type(max_memory_mb) is not int or not 1 <= max_memory_mb <= 512:
        raise AnalysisError("survey_budget", "max_memory_mb must be an integer in 1..512.")
    roles = [v for v in (psu, ssu, population_psu, population_ssu, strata) if v is not None]
    if (
        len(roles) != (5 if strata is not None else 4)
        or any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in roles)
        or len(set(roles)) != len(roles)
    ):
        raise AnalysisError(
            "invalid_survey_design", "Two-stage roles need distinct nonempty column names."
        )
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, Mapping):
        if any(type(v).__module__.startswith("torch") for v in data.values()):
            raise AnalysisError("unsupported_survey_input", "Use resident tabular columns.")
        if any(v not in data for v in roles):
            raise AnalysisError("missing_column", "Every design role must exist.")
        selected = {v: data[v] for v in roles}
        lengths = [len(v) for v in selected.values() if hasattr(v, "__len__")]
        if lengths and max(lengths) > max_rows:
            raise AnalysisError("survey_budget", "Complete geometry exceeds max_rows.")
        if lengths:
            admission(max(lengths), max_memory_mb)
        try:
            frame = pd.DataFrame(selected)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError(
                "invalid_data", "Design columns need consistent resident scalar rows."
            ) from exc
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) > max_rows:
            raise AnalysisError("survey_budget", "Complete geometry exceeds max_rows.")
        admission(len(data), max_memory_mb)
        if any(not isinstance(row, Mapping) or any(v not in row for v in roles) for row in data):
            raise AnalysisError("invalid_data", "Every row must contain all design roles.")
        frame = pd.DataFrame([{v: row[v] for v in roles} for row in data])
    else:
        raise AnalysisError(
            "unsupported_survey_input", "Use a resident DataFrame or column/row mapping."
        )
    if frame.columns.has_duplicates or any(v not in frame for v in roles):
        raise AnalysisError("missing_column", "Design roles require existing unique columns.")
    if not len(frame):
        raise AnalysisError("empty_data", "A design requires observations.")
    if len(frame) > max_rows:
        raise AnalysisError("survey_budget", "Complete geometry exceeds max_rows.")
    admission(len(frame), max_memory_mb)
    if sum(int(frame[v].memory_usage(index=False, deep=True)) for v in roles) + len(
        frame
    ) * 1536 > min(max_memory_mb * 1024**2, workspace_budget_bytes()):
        raise AnalysisError(
            "survey_budget", "Complete nested geometry exceeds the resident budget."
        )
    checksum = hashlib.sha256()
    checksum.update(
        json.dumps([(v, str(frame[v].dtype)) for v in roles], ensure_ascii=False).encode()
    )
    for v in roles:
        dtype = frame[v].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            checksum.update(
                json.dumps(
                    [v, dtype.ordered, [_identity(c) for c in dtype.categories]], ensure_ascii=False
                ).encode()
            )
    hs, ps, strata_records, psu_records, ssus = {}, {}, [], [], set()
    for position, row in enumerate(frame[roles].itertuples(index=False, name=None)):
        values = dict(zip(roles, row))
        h = _identity(values[strata]) if strata else ("unstratified",)
        p = _identity(values[psu])
        q = _identity(values[ssu])
        N = _number(values[population_psu], population=True)
        M = _number(values[population_ssu], population=True)
        if (h, p, q) in ssus:
            raise AnalysisError(
                "invalid_survey_nesting",
                "Each physical row must be a unique SSU within its PSU/stratum.",
            )
        ssus.add((h, p, q))
        if h not in hs:
            hs[h] = len(strata_records)
            strata_records.append({"population_psu": N, "psu_indices": []})
        hi = hs[h]
        sr = strata_records[hi]
        if sr["population_psu"] != N:
            raise AnalysisError(
                "invalid_survey_fpc", "PSU population counts must be constant within each stratum."
            )
        if (h, p) not in ps:
            ps[h, p] = len(psu_records)
            sr["psu_indices"].append(ps[h, p])
            psu_records.append({"stratum_index": hi, "population_ssu": M, "row_positions": []})
        pr = psu_records[ps[h, p]]
        if pr["population_ssu"] != M:
            raise AnalysisError(
                "invalid_survey_fpc",
                "SSU population counts must be constant within each nested PSU.",
            )
        pr["row_positions"].append(position)
        checksum.update(
            json.dumps([h, p, q, N, M], ensure_ascii=False, separators=(",", ":")).encode()
        )
        checksum.update(b"\n")
    for s in strata_records:
        s["n_psu"] = len(s["psu_indices"])
        if s["population_psu"] < s["n_psu"] or s["n_psu"] == 1 and s["population_psu"] != 1:
            raise AnalysisError(
                "invalid_survey_fpc",
                "Stage-one counts must dominate the sample; singleton requires census.",
            )
    for p in psu_records:
        p["n_ssu"] = len(p["row_positions"])
        if p["population_ssu"] < p["n_ssu"] or p["n_ssu"] == 1 and p["population_ssu"] != 1:
            raise AnalysisError(
                "invalid_survey_fpc",
                "Stage-two counts must dominate the sample; singleton requires census.",
            )
    weights = [0.0] * len(frame)
    for p in psu_records:
        s = strata_records[p["stratum_index"]]
        w = s["population_psu"] / s["n_psu"] * p["population_ssu"] / p["n_ssu"]
        for i in p["row_positions"]:
            weights[i] = w
    return TwoStageValidation(
        nobs=len(frame),
        n_strata=len(strata_records),
        n_psu=len(psu_records),
        design_df=len(psu_records) - len(strata_records),
        sum_weights=math.fsum(weights),
        design_input_sha256=checksum.hexdigest(),
        strata=strata_records,
        psus=psu_records,
    )


def survey_two_stage_design(
    data: Any,
    *,
    psu: str,
    ssu: str,
    population_psu: str,
    population_ssu: str,
    strata: str | None = None,
    max_rows: int = 1_000_000,
    max_memory_mb: int = 64,
) -> SurveyTwoStageDesign:
    """Declare two nested SRSWOR stages; derive N/n × M/m weights from complete counts."""
    options = dict(
        psu=psu,
        ssu=ssu,
        population_psu=population_psu,
        population_ssu=population_ssu,
        strata=strata,
        max_rows=max_rows,
        max_memory_mb=max_memory_mb,
    )
    return SurveyTwoStageDesign(**options, validation=_validate(data, **options))
