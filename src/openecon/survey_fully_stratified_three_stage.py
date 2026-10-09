"""Three SRSWOR selections with explicit SSU and TSU strata within sampled parents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, Literal

from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.survey import _Record as _SurveyRecord, _identity, _number


class _Record(_SurveyRecord):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class FullyStratifiedThreeStageStratum(_Record):
    n_psu: StrictInt = Field(gt=0)
    population_psu: StrictInt = Field(gt=0, le=2**53)
    psu_indices: tuple[StrictInt, ...]


class FullyStratifiedThreeStagePSU(_Record):
    stratum_index: StrictInt = Field(ge=0)
    n_ssu: StrictInt = Field(gt=0)
    cell_indices: tuple[StrictInt, ...]
    ssu_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class FullyStratifiedThreeStageCell(_Record):
    psu_index: StrictInt = Field(ge=0)
    frame_index: StrictInt = Field(ge=0)
    n_ssu: StrictInt = Field(gt=0)
    population_ssu: StrictInt = Field(gt=0, le=2**53)
    ssu_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class FullyStratifiedThreeStageSSU(_Record):
    cell_index: StrictInt = Field(ge=0)
    psu_index: StrictInt = Field(ge=0)
    n_tsu: StrictInt = Field(gt=0)
    tsu_cell_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class FullyStratifiedThreeStageTSUCell(_Record):
    ssu_index: StrictInt = Field(ge=0)
    frame_index: StrictInt = Field(ge=0)
    n_tsu: StrictInt = Field(gt=0)
    population_tsu: StrictInt = Field(gt=0, le=2**53)
    row_positions: tuple[StrictInt, ...]


def _indices(values, size):
    return (
        bool(values) and list(values) == sorted(set(values)) and all(0 <= i < size for i in values)
    )


def _decoded_identity(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Frame identities require an exact typed scalar pair.")
    tag, encoded = value
    if tag == "float" and isinstance(encoded, str) and len(encoded) <= 32:
        try:
            decoded = float.fromhex(encoded)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Invalid frame float identity.") from exc
    elif tag in ("bool", "int", "str"):
        decoded = encoded
    else:
        raise ValueError("Invalid frame identity kind.")
    try:
        identity = _identity(decoded)
    except AnalysisError as exc:
        raise ValueError("Frame identity exceeds canonical scalar bounds.") from exc
    if tuple(value) != identity or type(encoded) is not type(identity[1]):
        raise ValueError("Frame identities must use the exact canonical scalar encoding.")
    return decoded


class FullyStratifiedThreeStageFrameCell(_Record):
    """Caller-declared positive lower-stratum population, using canonical identities."""

    stratum: tuple[str, Any] | None
    psu: tuple[str, Any]
    ssu_stratum: tuple[str, Any]
    population_ssu: StrictInt = Field(gt=0, le=2**53)

    @model_validator(mode="before")
    @classmethod
    def bounded_record(cls, value):
        if isinstance(value, cls):
            value = vars(value)
        if isinstance(value, Mapping) and (
            len(value) != 4 or set(value) != {"stratum", "psu", "ssu_stratum", "population_ssu"}
        ):
            raise ValueError("Saved frame records require exactly four bounded canonical fields.")
        return value

    @field_validator("stratum", "psu", "ssu_stratum", mode="before")
    @classmethod
    def canonical_identity(cls, value):
        if value is not None:
            _decoded_identity(value)
        return value


class FullyStratifiedThreeStageTSUFrameCell(_Record):
    """Explicit positive terminal stratum within a sampled SSU."""

    stratum: tuple[str, Any] | None
    psu: tuple[str, Any]
    ssu_stratum: tuple[str, Any]
    ssu: tuple[str, Any]
    tsu_stratum: tuple[str, Any]
    population_tsu: StrictInt = Field(gt=0, le=2**53)

    @model_validator(mode="before")
    @classmethod
    def bounded_record(cls, value):
        if isinstance(value, cls):
            value = vars(value)
        if isinstance(value, Mapping) and set(value) != {
            "stratum",
            "psu",
            "ssu_stratum",
            "ssu",
            "tsu_stratum",
            "population_tsu",
        }:
            raise ValueError("Saved terminal frame records require exactly six canonical fields.")
        return value

    @field_validator("stratum", "psu", "ssu_stratum", "ssu", "tsu_stratum", mode="before")
    @classmethod
    def canonical_identity(cls, value):
        if value is not None:
            _decoded_identity(value)
        return value


def frame_digest(records):
    return hashlib.sha256(
        json.dumps(
            [r.model_dump(mode="json") for r in records],
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _frame_sort_key(record):
    return json.dumps(
        [record.stratum, record.psu, record.ssu_stratum]
        + (
            [record.ssu, record.tsu_stratum]
            if isinstance(record, FullyStratifiedThreeStageTSUFrameCell)
            else []
        ),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


class FullyStratifiedThreeStageValidation(_Record):
    nobs: StrictInt = Field(gt=0, le=1_000_000)
    n_strata: StrictInt = Field(gt=0, le=1_000_000)
    n_psu: StrictInt = Field(gt=0, le=1_000_000)
    n_cells: StrictInt = Field(gt=0, le=1_000_000)
    n_ssu: StrictInt = Field(gt=0, le=1_000_000)
    n_tsu_cells: StrictInt = Field(gt=0, le=1_000_000)
    design_df: StrictInt = Field(ge=0)
    sum_weights: float = Field(gt=0, allow_inf_nan=False, strict=True)
    design_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tsu_frame_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[FullyStratifiedThreeStageStratum, ...]
    psus: tuple[FullyStratifiedThreeStagePSU, ...]
    cells: tuple[FullyStratifiedThreeStageCell, ...]
    ssus: tuple[FullyStratifiedThreeStageSSU, ...]
    tsu_cells: tuple[FullyStratifiedThreeStageTSUCell, ...]

    @model_validator(mode="before")
    @classmethod
    def bounded_geometry(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            n = raw.get("nobs")
            if type(n) is not int or not 1 <= n <= 1_000_000:
                raise ValueError("Complete row count is outside the bounded design.")
            for key in ("strata", "psus", "cells", "ssus", "tsu_cells"):
                records = raw.get(key)
                if not isinstance(records, (tuple, list)) or not 1 <= len(records) <= n:
                    raise ValueError("Nested record dimensions exceed the complete sample.")
                totals = dict.fromkeys(
                    (
                        "row_positions",
                        "psu_indices",
                        "cell_indices",
                        "ssu_indices",
                        "tsu_cell_indices",
                    ),
                    0,
                )
                for record in records:
                    get = (
                        record.get
                        if isinstance(record, Mapping)
                        else lambda k: getattr(record, k, None)
                    )
                    for field in totals:
                        indices = get(field)
                        if indices is None:
                            continue
                        if not isinstance(indices, (tuple, list)) or len(indices) > n:
                            raise ValueError("Nested index lists exceed the complete sample.")
                        totals[field] += len(indices)
                        if totals[field] > n:
                            raise ValueError("Nested index storage exceeds the complete sample.")
        return raw

    @model_validator(mode="after")
    def geometry(self):
        if (
            len(self.strata),
            len(self.psus),
            len(self.cells),
            len(self.ssus),
            len(self.tsu_cells),
            self.design_df,
        ) != (
            self.n_strata,
            self.n_psu,
            self.n_cells,
            self.n_ssu,
            self.n_tsu_cells,
            self.n_psu - self.n_strata,
        ) or not self.n_psu <= self.n_cells <= self.n_ssu <= self.n_tsu_cells <= self.nobs:
            raise ValueError("Three-stage cell dimensions or first-stage df are inconsistent.")
        seen_psus, seen_cells, seen_ssus, seen_rows, seen_frames = [], [], [], [], []
        for h, stratum in enumerate(self.strata):
            if (
                len(stratum.psu_indices) != stratum.n_psu
                or not _indices(stratum.psu_indices, self.n_psu)
                or stratum.population_psu < stratum.n_psu
                or stratum.n_psu == 1
                and stratum.population_psu != 1
            ):
                raise ValueError("Stage-one counts or census singleton are inconsistent.")
            if any(self.psus[i].stratum_index != h for i in stratum.psu_indices):
                raise ValueError("PSU parent assignment is inconsistent.")
            seen_psus.extend(stratum.psu_indices)
        for i, psu in enumerate(self.psus):
            if (
                psu.stratum_index >= self.n_strata
                or len(psu.ssu_indices) != psu.n_ssu
                or not _indices(psu.ssu_indices, self.n_ssu)
                or not _indices(psu.cell_indices, self.n_cells)
                or not _indices(psu.row_positions, self.nobs)
            ):
                raise ValueError("PSU membership is inconsistent.")
            descendants, children = [], []
            for c in psu.cell_indices:
                cell = self.cells[c]
                if cell.psu_index != i:
                    raise ValueError("Lower stratum parent assignment is inconsistent.")
                descendants.extend(cell.row_positions)
                children.extend(cell.ssu_indices)
            if sorted(descendants) != list(psu.row_positions) or sorted(children) != list(
                psu.ssu_indices
            ):
                raise ValueError("PSU rows/SSUs must equal its complete lower-stratum descendants.")
            seen_cells.extend(psu.cell_indices)
        for c, cell in enumerate(self.cells):
            if (
                cell.psu_index >= self.n_psu
                or cell.frame_index >= self.n_cells
                or len(cell.ssu_indices) != cell.n_ssu
                or not _indices(cell.ssu_indices, self.n_ssu)
                or not _indices(cell.row_positions, self.nobs)
                or cell.population_ssu < cell.n_ssu
                or cell.n_ssu == 1
                and cell.population_ssu != 1
            ):
                raise ValueError(
                    "Lower-stratum counts, membership or census singleton are inconsistent."
                )
            descendants = []
            for j in cell.ssu_indices:
                child = self.ssus[j]
                if child.cell_index != c or child.psu_index != cell.psu_index:
                    raise ValueError("SSU lower-stratum/PSU parent assignment is inconsistent.")
                descendants.extend(child.row_positions)
            if sorted(descendants) != list(cell.row_positions):
                raise ValueError("Lower-stratum rows must equal its complete SSU descendants.")
            seen_ssus.extend(cell.ssu_indices)
            seen_frames.append(cell.frame_index)
        seen_terminal_cells, seen_terminal_frames = [], []
        for i, child in enumerate(self.ssus):
            if (
                child.psu_index >= self.n_psu
                or child.cell_index >= self.n_cells
                or len(child.row_positions) != child.n_tsu
                or not _indices(child.row_positions, self.nobs)
                or not _indices(child.tsu_cell_indices, self.n_tsu_cells)
            ):
                raise ValueError(
                    "Stage-three counts, physical rows or census singleton are inconsistent."
                )
            descendants = []
            for j in child.tsu_cell_indices:
                terminal = self.tsu_cells[j]
                if terminal.ssu_index != i:
                    raise ValueError("Terminal stratum parent assignment is inconsistent.")
                descendants.extend(terminal.row_positions)
            if sorted(descendants) != list(child.row_positions):
                raise ValueError("SSU rows must equal its complete terminal-stratum descendants.")
            seen_terminal_cells.extend(child.tsu_cell_indices)
        for terminal in self.tsu_cells:
            if (
                terminal.ssu_index >= self.n_ssu
                or terminal.frame_index >= self.n_tsu_cells
                or len(terminal.row_positions) != terminal.n_tsu
                or not _indices(terminal.row_positions, self.nobs)
                or terminal.population_tsu < terminal.n_tsu
                or terminal.n_tsu == 1
                and terminal.population_tsu != 1
            ):
                raise ValueError("Terminal-cell counts, physical rows or census singleton differ.")
            seen_rows.extend(terminal.row_positions)
            seen_terminal_frames.append(terminal.frame_index)
        for observed, size in (
            (seen_psus, self.n_psu),
            (seen_cells, self.n_cells),
            (seen_ssus, self.n_ssu),
            (seen_frames, self.n_cells),
            (seen_terminal_cells, self.n_tsu_cells),
            (seen_terminal_frames, self.n_tsu_cells),
            (seen_rows, self.nobs),
        ):
            if sorted(observed) != list(range(size)):
                raise ValueError(
                    "Every cell, sampled unit and physical row must have exactly one parent."
                )
        if not math.isclose(math.fsum(self.weights), self.sum_weights, rel_tol=1e-12):
            raise ValueError("Derived stratified three-stage weight sum is inconsistent.")
        return self

    @property
    def weights(self):
        values = [0.0] * self.nobs
        for terminal in self.tsu_cells:
            child = self.ssus[terminal.ssu_index]
            cell = self.cells[child.cell_index]
            stratum = self.strata[self.psus[child.psu_index].stratum_index]
            weight = (
                stratum.population_psu
                / stratum.n_psu
                * cell.population_ssu
                / cell.n_ssu
                * terminal.population_tsu
                / terminal.n_tsu
            )
            for position in terminal.row_positions:
                values[position] = weight
        return tuple(values)


def admission(n, memory_mb, *, k=0):
    return plan_workspace(
        "fully stratified three-stage survey inference",
        {
            "complete typed SSU/TSU frames, three-level identities, indices and JSON replay": n
            * 16384,
            "target, nested totals and three covariance serialization buffers": (n * k + k * k)
            * 384,
        },
        budget_bytes=min(memory_mb * 1024**2, workspace_budget_bytes()),
    )


class SurveyFullyStratifiedThreeStageDesign(_Record):
    """Frozen three-stage geometry; provenance hashes do not authenticate sampling."""

    schema_version: Literal["survey-fully-stratified-three-stage-design-v1"] = (
        "survey-fully-stratified-three-stage-design-v1"
    )
    stages: Literal[3] = 3
    sampling: Literal["sequential-srswor"] = "sequential-srswor"
    psu: str
    ssu_strata: str
    ssu: str
    tsu: str
    tsu_strata: str
    population_psu: str
    population_ssu: str
    population_tsu: str
    strata: str | None = None
    max_rows: StrictInt = Field(default=1_000_000, ge=1, le=1_000_000)
    max_memory_mb: StrictInt = Field(default=64, ge=1, le=512)
    ssu_frame: tuple[FullyStratifiedThreeStageFrameCell, ...]
    tsu_frame: tuple[FullyStratifiedThreeStageTSUFrameCell, ...]
    validation: FullyStratifiedThreeStageValidation

    @model_validator(mode="before")
    @classmethod
    def admit_before_nested_records(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            validation = raw.get("validation")
            n = (
                validation.nobs
                if isinstance(validation, FullyStratifiedThreeStageValidation)
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
            for name in ("ssu_frame", "tsu_frame"):
                frame = raw.get(name)
                if not isinstance(frame, (list, tuple)) or not 1 <= len(frame) <= n:
                    raise ValueError("Saved explicit frame exceeds complete row dimensions.")
        return raw

    @field_validator(
        "psu",
        "ssu_strata",
        "ssu",
        "tsu",
        "tsu_strata",
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
                self.ssu_strata,
                self.ssu,
                self.tsu,
                self.tsu_strata,
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
        if (
            len(self.ssu_frame) != self.validation.n_cells
            or sorted(self.ssu_frame, key=_frame_sort_key) != list(self.ssu_frame)
            or frame_digest(self.ssu_frame) != self.validation.frame_input_sha256
        ):
            raise ValueError("Canonical lower-stratum frame and its fingerprint are inconsistent.")
        seen, parent_ids, stratum_ids = set(), {}, {}
        for cell in self.validation.cells:
            record = self.ssu_frame[cell.frame_index]
            key = (record.stratum, record.psu, record.ssu_stratum)
            if key in seen or (record.stratum is None) != (self.strata is None):
                raise ValueError(
                    "Frame cells require unique typed nested identities and stratum roles."
                )
            seen.add(key)
            if record.population_ssu != cell.population_ssu:
                raise ValueError("Frame population count differs from its lower-stratum geometry.")
            parent_key = (record.stratum, record.psu)
            if cell.psu_index in parent_ids and parent_ids[cell.psu_index] != parent_key:
                raise ValueError("Frame cells assigned to one PSU must retain its typed identity.")
            parent_ids[cell.psu_index] = parent_key
            h = self.validation.psus[cell.psu_index].stratum_index
            if h in stratum_ids and stratum_ids[h] != record.stratum:
                raise ValueError(
                    "Frame PSU identities disagree with their declared first-stage stratum."
                )
            stratum_ids[h] = record.stratum
        if (
            len(set(parent_ids.values())) != self.validation.n_psu
            or len(set(stratum_ids.values())) != self.validation.n_strata
        ):
            raise ValueError(
                "Typed frame PSU/stratum identities must form a complete unique partition."
            )
        if (
            len(self.tsu_frame) != self.validation.n_tsu_cells
            or sorted(self.tsu_frame, key=_frame_sort_key) != list(self.tsu_frame)
            or frame_digest(self.tsu_frame) != self.validation.tsu_frame_input_sha256
        ):
            raise ValueError("Canonical terminal frame and its fingerprint are inconsistent.")
        terminal_keys, ssu_ids = set(), {}
        for terminal in self.validation.tsu_cells:
            record = self.tsu_frame[terminal.frame_index]
            child = self.validation.ssus[terminal.ssu_index]
            cell = self.validation.cells[child.cell_index]
            lower = self.ssu_frame[cell.frame_index]
            if (record.stratum, record.psu, record.ssu_stratum) != (
                lower.stratum,
                lower.psu,
                lower.ssu_stratum,
            ) or record.population_tsu != terminal.population_tsu:
                raise ValueError(
                    "Terminal frame ancestry/population differs from its sampling cell."
                )
            parent_key = (record.stratum, record.psu, record.ssu_stratum, record.ssu)
            key = (*parent_key, record.tsu_stratum)
            if key in terminal_keys:
                raise ValueError("Terminal frame requires unique nested typed cells.")
            terminal_keys.add(key)
            if terminal.ssu_index in ssu_ids and ssu_ids[terminal.ssu_index] != parent_key:
                raise ValueError("Terminal cells of one SSU must retain its typed identity.")
            ssu_ids[terminal.ssu_index] = parent_key
        if len(set(ssu_ids.values())) != self.validation.n_ssu:
            raise ValueError("Terminal frame SSU identities must form a complete unique partition.")
        return self

    def revalidate(self, data: Any):
        try:
            validated = type(self).model_validate(self)
        except ValueError as exc:
            raise AnalysisError(
                "invalid_survey_design", "Saved three-stage design is invalid."
            ) from exc
        options = validated.model_dump(
            exclude={"validation", "schema_version", "stages", "sampling", "ssu_frame", "tsu_frame"}
        )
        raw_frame = []
        for record in validated.ssu_frame:
            row = {
                validated.psu: _decoded_identity(record.psu),
                validated.ssu_strata: _decoded_identity(record.ssu_stratum),
                validated.population_ssu: record.population_ssu,
            }
            if validated.strata:
                row[validated.strata] = _decoded_identity(record.stratum)
            raw_frame.append(row)
        raw_terminal = []
        for record in validated.tsu_frame:
            row = {
                validated.psu: _decoded_identity(record.psu),
                validated.ssu_strata: _decoded_identity(record.ssu_stratum),
                validated.ssu: _decoded_identity(record.ssu),
                validated.tsu_strata: _decoded_identity(record.tsu_stratum),
                validated.population_tsu: record.population_tsu,
            }
            if validated.strata:
                row[validated.strata] = _decoded_identity(record.stratum)
            raw_terminal.append(row)
        current, normalized, terminal = _validate(
            data, ssu_frame=raw_frame, tsu_frame=raw_terminal, **options
        )
        if normalized != validated.ssu_frame or terminal != validated.tsu_frame:
            raise AnalysisError(
                "survey_design_changed", "Lower-stratum frame changed; declare again."
            )
        if current != validated.validation:
            raise AnalysisError(
                "survey_design_changed",
                "Three-stage inputs or ordered geometry changed; declare again.",
            )
        return current


def _validate(
    data,
    *,
    psu,
    ssu_strata,
    ssu_frame,
    tsu_frame,
    ssu,
    tsu,
    tsu_strata,
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
        for v in (
            psu,
            ssu_strata,
            ssu,
            tsu_strata,
            tsu,
            population_psu,
            population_ssu,
            population_tsu,
            strata,
        )
        if v is not None
    ]
    if (
        len(roles) != (9 if strata is not None else 8)
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
    ) * 16384 > min(max_memory_mb * 1024**2, workspace_budget_bytes()):
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

    def normalize(records, terminal=False):
        if (
            not isinstance(records, Sequence)
            or isinstance(records, (str, bytes))
            or not 1 <= len(records) <= len(frame)
        ):
            raise AnalysisError(
                "invalid_survey_frame", "Declare both bounded explicit positive-cell frames."
            )
        ids = [psu, ssu_strata] + ([ssu, tsu_strata] if terminal else [])
        population_role = population_tsu if terminal else population_ssu
        exact_roles = ids + [population_role] + ([strata] if strata else [])
        lookup, normalized = {}, []
        for raw in records:
            if not isinstance(raw, Mapping) or set(raw) != set(exact_roles):
                raise AnalysisError(
                    "invalid_survey_frame",
                    "Frame records need exactly their declared identity and population roles.",
                )
            h = _identity(raw[strata]) if strata else None
            identities = [_identity(raw[v]) for v in ids]
            key = (h, *identities)
            population = _number(raw[population_role], population=True)
            if key in lookup:
                raise AnalysisError("invalid_survey_frame", "Duplicate typed frame cell.")
            lookup[key] = population
            values = dict(stratum=h, psu=identities[0], ssu_stratum=identities[1])
            if terminal:
                normalized.append(
                    FullyStratifiedThreeStageTSUFrameCell(
                        **values,
                        ssu=identities[2],
                        tsu_stratum=identities[3],
                        population_tsu=population,
                    )
                )
            else:
                normalized.append(
                    FullyStratifiedThreeStageFrameCell(**values, population_ssu=population)
                )
        normalized = tuple(sorted(normalized, key=_frame_sort_key))
        indices = {
            (r.stratum, r.psu, r.ssu_stratum, *((r.ssu, r.tsu_stratum) if terminal else ())): i
            for i, r in enumerate(normalized)
        }
        return normalized, lookup, indices

    lower, lower_lookup, lower_indices = normalize(ssu_frame)
    terminal_frame, terminal_lookup, terminal_indices = normalize(tsu_frame, True)
    lower_sha, terminal_sha = frame_digest(lower), frame_digest(terminal_frame)
    checksum.update(lower_sha.encode())
    checksum.update(terminal_sha.encode())
    hs, ps, cs, ss, ts, leaves = {}, {}, {}, {}, {}, set()
    sr, pr, cr, qr, tr = [], [], [], [], []
    for position, row in enumerate(frame[roles].itertuples(index=False, name=None)):
        values = dict(zip(roles, row))
        h = _identity(values[strata]) if strata else None
        p, g, u, q, t = (_identity(values[v]) for v in (psu, ssu_strata, ssu, tsu_strata, tsu))
        N, M, L = (
            _number(values[v], population=True)
            for v in (population_psu, population_ssu, population_tsu)
        )
        ck, sk, tk, leaf = (h, p, g), (h, p, g, u), (h, p, g, u, q), (h, p, g, u, q, t)
        if lower_lookup.get(ck) != M or terminal_lookup.get(tk) != L:
            raise AnalysisError(
                "invalid_survey_frame",
                "Observed cells/counts must match both complete declared frames.",
            )
        if leaf in leaves:
            raise AnalysisError(
                "invalid_survey_nesting",
                "Each physical row must be a unique nested TSU within its terminal stratum.",
            )
        leaves.add(leaf)
        if h not in hs:
            hs[h] = len(sr)
            sr.append(dict(population_psu=N, psu_indices=[]))
        stratum = sr[hs[h]]
        if stratum["population_psu"] != N:
            raise AnalysisError(
                "invalid_survey_fpc", "PSU population counts must be constant within each stratum."
            )
        if (h, p) not in ps:
            ps[h, p] = len(pr)
            stratum["psu_indices"].append(ps[h, p])
            pr.append(dict(stratum_index=hs[h], cell_indices=[], ssu_indices=[], row_positions=[]))
        parent = pr[ps[h, p]]
        if ck not in cs:
            cs[ck] = len(cr)
            parent["cell_indices"].append(cs[ck])
            cr.append(
                dict(
                    psu_index=ps[h, p],
                    frame_index=lower_indices[ck],
                    population_ssu=M,
                    ssu_indices=[],
                    row_positions=[],
                )
            )
        cell = cr[cs[ck]]
        if sk not in ss:
            ss[sk] = len(qr)
            cell["ssu_indices"].append(ss[sk])
            parent["ssu_indices"].append(ss[sk])
            qr.append(
                dict(psu_index=ps[h, p], cell_index=cs[ck], tsu_cell_indices=[], row_positions=[])
            )
        child = qr[ss[sk]]
        if tk not in ts:
            ts[tk] = len(tr)
            child["tsu_cell_indices"].append(ts[tk])
            tr.append(
                dict(
                    ssu_index=ss[sk],
                    frame_index=terminal_indices[tk],
                    population_tsu=L,
                    row_positions=[],
                )
            )
        for record in (parent, cell, child, tr[ts[tk]]):
            record["row_positions"].append(position)
        checksum.update(
            json.dumps(
                [h, p, g, u, q, t, N, M, L], ensure_ascii=False, separators=(",", ":")
            ).encode()
        )
        checksum.update(b"\n")
    if set(cs) != set(lower_lookup) or set(ts) != set(terminal_lookup):
        raise AnalysisError(
            "invalid_survey_frame",
            "Every positive declared cell must be sampled inside its observed parent.",
        )
    for records, field, population, count in (
        (sr, "psu_indices", "population_psu", "n_psu"),
        (cr, "ssu_indices", "population_ssu", "n_ssu"),
        (tr, "row_positions", "population_tsu", "n_tsu"),
    ):
        for record in records:
            record[count] = len(record[field])
            if record[population] < record[count] or record[count] == 1 and record[population] != 1:
                raise AnalysisError(
                    "invalid_survey_fpc",
                    "Each selection-cell population must dominate its sample; singleton requires census.",
                )
    for record in pr:
        record["n_ssu"] = len(record["ssu_indices"])
    for record in qr:
        record["n_tsu"] = len(record["row_positions"])
    weights = [0.0] * len(frame)
    for terminal in tr:
        child = qr[terminal["ssu_index"]]
        cell = cr[child["cell_index"]]
        stratum = sr[pr[child["psu_index"]]["stratum_index"]]
        weight = (
            stratum["population_psu"]
            / stratum["n_psu"]
            * cell["population_ssu"]
            / cell["n_ssu"]
            * terminal["population_tsu"]
            / terminal["n_tsu"]
        )
        for position in terminal["row_positions"]:
            weights[position] = weight
    validation = FullyStratifiedThreeStageValidation(
        nobs=len(frame),
        n_strata=len(sr),
        n_psu=len(pr),
        n_cells=len(cr),
        n_ssu=len(qr),
        n_tsu_cells=len(tr),
        design_df=len(pr) - len(sr),
        sum_weights=math.fsum(weights),
        design_input_sha256=checksum.hexdigest(),
        frame_input_sha256=lower_sha,
        tsu_frame_input_sha256=terminal_sha,
        strata=sr,
        psus=pr,
        cells=cr,
        ssus=qr,
        tsu_cells=tr,
    )
    return validation, lower, terminal_frame


def survey_fully_stratified_three_stage_design(
    data: Any,
    *,
    psu: str,
    ssu_strata: str,
    ssu_frame: Sequence[Mapping[str, Any]],
    ssu: str,
    tsu_strata: str,
    tsu_frame: Sequence[Mapping[str, Any]],
    tsu: str,
    population_psu: str,
    population_ssu: str,
    population_tsu: str,
    strata: str | None = None,
    max_rows: int = 1_000_000,
    max_memory_mb: int = 64,
) -> SurveyFullyStratifiedThreeStageDesign:
    """Declare exactly three SRSWOR selections with two explicit positive-cell frames.

    SSUs are selected within SSU strata inside each sampled PSU; TSUs are
    selected within TSU strata inside each sampled SSU. Completely omitted
    external cells cannot be authenticated from caller-provided inputs.
    """
    options = dict(
        psu=psu,
        ssu_strata=ssu_strata,
        ssu=ssu,
        tsu_strata=tsu_strata,
        tsu=tsu,
        population_psu=population_psu,
        population_ssu=population_ssu,
        population_tsu=population_tsu,
        strata=strata,
        max_rows=max_rows,
        max_memory_mb=max_memory_mb,
    )
    validation, lower, terminal = _validate(
        data, ssu_frame=ssu_frame, tsu_frame=tsu_frame, **options
    )
    return SurveyFullyStratifiedThreeStageDesign(
        **options, ssu_frame=lower, tsu_frame=terminal, validation=validation
    )
