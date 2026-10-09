"""Exactly three nested sequential SRSWOR stages with explicit finite populations."""

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


class ThreeStageStratum(_Record):
    n_psu: StrictInt = Field(gt=0)
    population_psu: StrictInt = Field(gt=0, le=2**53)
    psu_indices: tuple[StrictInt, ...]


class ThreeStagePSU(_Record):
    stratum_index: StrictInt = Field(ge=0)
    n_ssu: StrictInt = Field(gt=0)
    population_ssu: StrictInt = Field(gt=0, le=2**53)
    ssu_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class ThreeStageSSU(_Record):
    psu_index: StrictInt = Field(ge=0)
    n_tsu: StrictInt = Field(gt=0)
    population_tsu: StrictInt = Field(gt=0, le=2**53)
    row_positions: tuple[StrictInt, ...]


def _indices(values, size):
    return (
        bool(values) and list(values) == sorted(set(values)) and all(0 <= i < size for i in values)
    )


class ThreeStageValidation(_Record):
    nobs: StrictInt = Field(gt=0, le=1_000_000)
    n_strata: StrictInt = Field(gt=0, le=1_000_000)
    n_psu: StrictInt = Field(gt=0, le=1_000_000)
    n_ssu: StrictInt = Field(gt=0, le=1_000_000)
    design_df: StrictInt = Field(ge=0)
    sum_weights: float = Field(gt=0, allow_inf_nan=False, strict=True)
    design_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[ThreeStageStratum, ...]
    psus: tuple[ThreeStagePSU, ...]
    ssus: tuple[ThreeStageSSU, ...]

    @model_validator(mode="before")
    @classmethod
    def bounded_geometry(cls, raw):
        if isinstance(raw, Mapping):
            n = raw.get("nobs")
            if type(n) is not int or not 1 <= n <= 1_000_000:
                raise ValueError("Complete row count is outside the bounded design.")
            for key in ("strata", "psus", "ssus"):
                records = raw.get(key)
                if not isinstance(records, (tuple, list)) or not 1 <= len(records) <= n:
                    raise ValueError("Nested record dimensions exceed the complete sample.")
                total_rows, total_children = 0, 0
                for record in records:
                    get = (
                        record.get
                        if isinstance(record, Mapping)
                        else lambda k: getattr(record, k, None)
                    )
                    for field in ("row_positions", "psu_indices", "ssu_indices"):
                        values = get(field)
                        if values is None:
                            continue
                        if not isinstance(values, (tuple, list)) or len(values) > n:
                            raise ValueError("Nested index lists exceed the complete sample.")
                        if field == "row_positions":
                            total_rows += len(values)
                        else:
                            total_children += len(values)
                    if total_rows > n or total_children > n:
                        raise ValueError("Nested index storage exceeds the complete sample.")
        return raw

    @model_validator(mode="after")
    def geometry(self):
        if (len(self.strata), len(self.psus), len(self.ssus), self.design_df) != (
            self.n_strata,
            self.n_psu,
            self.n_ssu,
            self.n_psu - self.n_strata,
        ):
            raise ValueError("Three-stage dimensions or first-stage df are inconsistent.")
        seen_psus, seen_ssus, seen_rows = [], [], []
        for h, s in enumerate(self.strata):
            if (
                len(s.psu_indices) != s.n_psu
                or not _indices(s.psu_indices, self.n_psu)
                or s.population_psu < s.n_psu
                or s.n_psu == 1
                and s.population_psu != 1
            ):
                raise ValueError("Stage-one counts or census singleton are inconsistent.")
            for i in s.psu_indices:
                if self.psus[i].stratum_index != h:
                    raise ValueError("PSU parent assignment is inconsistent.")
            seen_psus.extend(s.psu_indices)
        for i, p in enumerate(self.psus):
            if (
                p.stratum_index >= self.n_strata
                or len(p.ssu_indices) != p.n_ssu
                or not _indices(p.ssu_indices, self.n_ssu)
                or not _indices(p.row_positions, self.nobs)
                or p.population_ssu < p.n_ssu
                or p.n_ssu == 1
                and p.population_ssu != 1
            ):
                raise ValueError(
                    "Stage-two counts, membership or census singleton are inconsistent."
                )
            descendants = []
            for j in p.ssu_indices:
                if self.ssus[j].psu_index != i:
                    raise ValueError("SSU parent assignment is inconsistent.")
                descendants.extend(self.ssus[j].row_positions)
            if sorted(descendants) != list(p.row_positions):
                raise ValueError("PSU physical rows must equal its complete SSU descendants.")
            seen_ssus.extend(p.ssu_indices)
        for q in self.ssus:
            if (
                q.psu_index >= self.n_psu
                or len(q.row_positions) != q.n_tsu
                or not _indices(q.row_positions, self.nobs)
                or q.population_tsu < q.n_tsu
                or q.n_tsu == 1
                and q.population_tsu != 1
            ):
                raise ValueError(
                    "Stage-three counts, physical rows or census singleton are inconsistent."
                )
            seen_rows.extend(q.row_positions)
        if sorted(seen_psus) != list(range(self.n_psu)) or sorted(seen_ssus) != list(
            range(self.n_ssu)
        ):
            raise ValueError("Every sampled unit must have exactly one parent.")
        if sorted(seen_rows) != list(range(self.nobs)):
            raise ValueError("Every physical row must be a distinct nested terminal TSU.")
        if not math.isclose(math.fsum(self.weights), self.sum_weights, rel_tol=1e-12):
            raise ValueError("Derived three-stage weight sum is inconsistent.")
        return self

    @property
    def weights(self):
        values = [0.0] * self.nobs
        for q in self.ssus:
            p = self.psus[q.psu_index]
            s = self.strata[p.stratum_index]
            w = s.population_psu / s.n_psu * p.population_ssu / p.n_ssu * q.population_tsu / q.n_tsu
            for i in q.row_positions:
                values[i] = w
        return tuple(values)


def admission(n, memory_mb, *, k=0):
    return plan_workspace(
        "three-stage survey inference",
        {
            "complete three-level identities, indices and JSON replay": n * 3072,
            "target, nested totals and three covariance serialization buffers": (n * k + k * k)
            * 384,
        },
        budget_bytes=min(memory_mb * 1024**2, workspace_budget_bytes()),
    )


class SurveyThreeStageDesign(_Record):
    """Frozen three-stage geometry; provenance hashes do not authenticate sampling."""

    schema_version: Literal["survey-three-stage-design-v1"] = "survey-three-stage-design-v1"
    stages: Literal[3] = 3
    sampling: Literal["sequential-srswor"] = "sequential-srswor"
    psu: str
    ssu: str
    tsu: str
    population_psu: str
    population_ssu: str
    population_tsu: str
    strata: str | None = None
    max_rows: StrictInt = Field(default=1_000_000, ge=1, le=1_000_000)
    max_memory_mb: StrictInt = Field(default=64, ge=1, le=512)
    validation: ThreeStageValidation

    @model_validator(mode="before")
    @classmethod
    def admit_before_nested_records(cls, raw):
        if isinstance(raw, Mapping):
            validation = raw.get("validation")
            n = (
                validation.nobs
                if isinstance(validation, ThreeStageValidation)
                else (validation.get("nobs") if isinstance(validation, Mapping) else None)
            )
            memory, rows = raw.get("max_memory_mb", 64), raw.get("max_rows", 1_000_000)
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

    @field_validator(
        "psu",
        "ssu",
        "tsu",
        "population_psu",
        "population_ssu",
        "population_tsu",
        "strata",
        mode="before",
    )
    @classmethod
    def name(cls, value):
        if value is not None and (
            not isinstance(value, str) or not value.strip() or len(value) > 200
        ):
            raise ValueError("Design roles require nonempty names of at most 200 characters.")
        return value

    @field_validator("stages", mode="before")
    @classmethod
    def exactly_three(cls, value):
        if type(value) is not int or value != 3:
            raise ValueError("Exactly three SRSWOR stages are supported.")
        return value

    @model_validator(mode="after")
    def roles_and_budget(self):
        roles = [
            v
            for v in (
                self.psu,
                self.ssu,
                self.tsu,
                self.population_psu,
                self.population_ssu,
                self.population_tsu,
                self.strata,
            )
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
                "Three-stage inputs or ordered geometry changed; declare again.",
            )
        return current


def _validate(
    data,
    *,
    psu,
    ssu,
    tsu,
    population_psu,
    population_ssu,
    population_tsu,
    strata,
    max_rows,
    max_memory_mb,
):
    import pandas as pd

    if type(max_rows) is not int or not 1 <= max_rows <= 1_000_000:
        raise AnalysisError("survey_budget", "max_rows must be an integer in 1..1,000,000.")
    if type(max_memory_mb) is not int or not 1 <= max_memory_mb <= 512:
        raise AnalysisError("survey_budget", "max_memory_mb must be an integer in 1..512.")
    roles = [
        v
        for v in (psu, ssu, tsu, population_psu, population_ssu, population_tsu, strata)
        if v is not None
    ]
    if (
        len(roles) != (7 if strata is not None else 6)
        or any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in roles)
        or len(set(roles)) != len(roles)
    ):
        raise AnalysisError(
            "invalid_survey_design", "Three-stage roles need distinct nonempty names."
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
        if lengths:
            if max(lengths) > max_rows:
                raise AnalysisError("survey_budget", "Complete geometry exceeds max_rows.")
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
            raise AnalysisError("invalid_data", "Every row must contain every design role.")
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
    ) * 3072 > min(max_memory_mb * 1024**2, workspace_budget_bytes()):
        raise AnalysisError(
            "survey_budget", "Complete three-level geometry exceeds the resident budget."
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
    hs, ps, qs, terminals = {}, {}, {}, set()
    sr, pr, qr = [], [], []
    for position, row in enumerate(frame[roles].itertuples(index=False, name=None)):
        values = dict(zip(roles, row))
        h = _identity(values[strata]) if strata else ("unstratified",)
        p, q, t = (_identity(values[v]) for v in (psu, ssu, tsu))
        N, M, L = (
            _number(values[v], population=True)
            for v in (population_psu, population_ssu, population_tsu)
        )
        if (h, p, q, t) in terminals:
            raise AnalysisError(
                "invalid_survey_nesting",
                "Each physical row must be a unique TSU within its SSU/PSU/stratum.",
            )
        terminals.add((h, p, q, t))
        if h not in hs:
            hs[h] = len(sr)
            sr.append({"population_psu": N, "psu_indices": []})
        s = sr[hs[h]]
        if s["population_psu"] != N:
            raise AnalysisError(
                "invalid_survey_fpc", "PSU population counts must be constant within each stratum."
            )
        if (h, p) not in ps:
            ps[h, p] = len(pr)
            s["psu_indices"].append(ps[h, p])
            pr.append(
                {
                    "stratum_index": hs[h],
                    "population_ssu": M,
                    "ssu_indices": [],
                    "row_positions": [],
                }
            )
        parent = pr[ps[h, p]]
        if parent["population_ssu"] != M:
            raise AnalysisError(
                "invalid_survey_fpc",
                "SSU population counts must be constant within each nested PSU.",
            )
        if (h, p, q) not in qs:
            qs[h, p, q] = len(qr)
            parent["ssu_indices"].append(qs[h, p, q])
            qr.append({"psu_index": ps[h, p], "population_tsu": L, "row_positions": []})
        child = qr[qs[h, p, q]]
        if child["population_tsu"] != L:
            raise AnalysisError(
                "invalid_survey_fpc",
                "TSU population counts must be constant within each nested SSU.",
            )
        parent["row_positions"].append(position)
        child["row_positions"].append(position)
        checksum.update(
            json.dumps([h, p, q, t, N, M, L], ensure_ascii=False, separators=(",", ":")).encode()
        )
        checksum.update(b"\n")
    for records, size, population, name in (
        (sr, "psu_indices", "population_psu", "n_psu"),
        (pr, "ssu_indices", "population_ssu", "n_ssu"),
        (qr, "row_positions", "population_tsu", "n_tsu"),
    ):
        for record in records:
            record[name] = len(record[size])
            if record[population] < record[name] or record[name] == 1 and record[population] != 1:
                raise AnalysisError(
                    "invalid_survey_fpc",
                    "Each stage population must dominate its sample; singleton requires census.",
                )
    weights = [0.0] * len(frame)
    for q in qr:
        p = pr[q["psu_index"]]
        s = sr[p["stratum_index"]]
        w = (
            s["population_psu"]
            / s["n_psu"]
            * p["population_ssu"]
            / p["n_ssu"]
            * q["population_tsu"]
            / q["n_tsu"]
        )
        for i in q["row_positions"]:
            weights[i] = w
    return ThreeStageValidation(
        nobs=len(frame),
        n_strata=len(sr),
        n_psu=len(pr),
        n_ssu=len(qr),
        design_df=len(pr) - len(sr),
        sum_weights=math.fsum(weights),
        design_input_sha256=checksum.hexdigest(),
        strata=sr,
        psus=pr,
        ssus=qr,
    )


def survey_three_stage_design(
    data: Any,
    *,
    psu: str,
    ssu: str,
    tsu: str,
    population_psu: str,
    population_ssu: str,
    population_tsu: str,
    strata: str | None = None,
    max_rows: int = 1_000_000,
    max_memory_mb: int = 64,
) -> SurveyThreeStageDesign:
    """Declare three SRSWOR stages with N/n × M/m × L/l inverse-probability weights."""
    options = dict(
        psu=psu,
        ssu=ssu,
        tsu=tsu,
        population_psu=population_psu,
        population_ssu=population_ssu,
        population_tsu=population_tsu,
        strata=strata,
        max_rows=max_rows,
        max_memory_mb=max_memory_mb,
    )
    return SurveyThreeStageDesign(**options, validation=_validate(data, **options))
