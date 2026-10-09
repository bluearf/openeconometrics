"""Three real SRSWOR selections with declared SSU strata inside each sampled PSU."""

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


class StratifiedThreeStageStratum(_Record):
    n_psu: StrictInt = Field(gt=0)
    population_psu: StrictInt = Field(gt=0, le=2**53)
    psu_indices: tuple[StrictInt, ...]


class StratifiedThreeStagePSU(_Record):
    stratum_index: StrictInt = Field(ge=0)
    n_ssu: StrictInt = Field(gt=0)
    cell_indices: tuple[StrictInt, ...]
    ssu_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class StratifiedThreeStageCell(_Record):
    psu_index: StrictInt = Field(ge=0)
    frame_index: StrictInt = Field(ge=0)
    n_ssu: StrictInt = Field(gt=0)
    population_ssu: StrictInt = Field(gt=0, le=2**53)
    ssu_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class StratifiedThreeStageSSU(_Record):
    cell_index: StrictInt = Field(ge=0)
    psu_index: StrictInt = Field(ge=0)
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


class StratifiedThreeStageFrameCell(_Record):
    """Caller-declared positive lower-stratum population, using canonical identities."""

    stratum: tuple[str, Any] | None
    psu: tuple[str, Any]
    ssu_stratum: tuple[str, Any]
    population_ssu: StrictInt = Field(gt=0, le=2**53)

    @model_validator(mode="before")
    @classmethod
    def bounded_record(cls, value):
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
        [record.stratum, record.psu, record.ssu_stratum],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


class StratifiedThreeStageValidation(_Record):
    nobs: StrictInt = Field(gt=0, le=1_000_000)
    n_strata: StrictInt = Field(gt=0, le=1_000_000)
    n_psu: StrictInt = Field(gt=0, le=1_000_000)
    n_cells: StrictInt = Field(gt=0, le=1_000_000)
    n_ssu: StrictInt = Field(gt=0, le=1_000_000)
    design_df: StrictInt = Field(ge=0)
    sum_weights: float = Field(gt=0, allow_inf_nan=False, strict=True)
    design_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[StratifiedThreeStageStratum, ...]
    psus: tuple[StratifiedThreeStagePSU, ...]
    cells: tuple[StratifiedThreeStageCell, ...]
    ssus: tuple[StratifiedThreeStageSSU, ...]

    @model_validator(mode="before")
    @classmethod
    def bounded_geometry(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            n = raw.get("nobs")
            if type(n) is not int or not 1 <= n <= 1_000_000:
                raise ValueError("Complete row count is outside the bounded design.")
            for key in ("strata", "psus", "cells", "ssus"):
                records = raw.get(key)
                if not isinstance(records, (tuple, list)) or not 1 <= len(records) <= n:
                    raise ValueError("Nested record dimensions exceed the complete sample.")
                totals = dict.fromkeys(
                    ("row_positions", "psu_indices", "cell_indices", "ssu_indices"), 0
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
        if (len(self.strata), len(self.psus), len(self.cells), len(self.ssus), self.design_df) != (
            self.n_strata,
            self.n_psu,
            self.n_cells,
            self.n_ssu,
            self.n_psu - self.n_strata,
        ) or not self.n_psu <= self.n_cells <= self.n_ssu <= self.nobs:
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
        for child in self.ssus:
            if (
                child.psu_index >= self.n_psu
                or child.cell_index >= self.n_cells
                or len(child.row_positions) != child.n_tsu
                or not _indices(child.row_positions, self.nobs)
                or child.population_tsu < child.n_tsu
                or child.n_tsu == 1
                and child.population_tsu != 1
            ):
                raise ValueError(
                    "Stage-three counts, physical rows or census singleton are inconsistent."
                )
            seen_rows.extend(child.row_positions)
        for observed, size in (
            (seen_psus, self.n_psu),
            (seen_cells, self.n_cells),
            (seen_ssus, self.n_ssu),
            (seen_frames, self.n_cells),
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
        for child in self.ssus:
            cell = self.cells[child.cell_index]
            stratum = self.strata[self.psus[child.psu_index].stratum_index]
            weight = (
                stratum.population_psu
                / stratum.n_psu
                * cell.population_ssu
                / cell.n_ssu
                * child.population_tsu
                / child.n_tsu
            )
            for position in child.row_positions:
                values[position] = weight
        return tuple(values)


def admission(n, memory_mb, *, k=0):
    return plan_workspace(
        "lower-stratified three-stage survey inference",
        {
            "complete typed lower-cell frame, three-level identities, indices and JSON replay": n
            * 8192,
            "target, nested totals and three covariance serialization buffers": (n * k + k * k)
            * 384,
        },
        budget_bytes=min(memory_mb * 1024**2, workspace_budget_bytes()),
    )


class SurveyStratifiedThreeStageDesign(_Record):
    """Frozen three-stage geometry; provenance hashes do not authenticate sampling."""

    schema_version: Literal["survey-stratified-three-stage-design-v1"] = (
        "survey-stratified-three-stage-design-v1"
    )
    stages: Literal[3] = 3
    sampling: Literal["sequential-srswor"] = "sequential-srswor"
    psu: str
    ssu_strata: str
    ssu: str
    tsu: str
    population_psu: str
    population_ssu: str
    population_tsu: str
    strata: str | None = None
    max_rows: StrictInt = Field(default=1_000_000, ge=1, le=1_000_000)
    max_memory_mb: StrictInt = Field(default=64, ge=1, le=512)
    ssu_frame: tuple[StratifiedThreeStageFrameCell, ...]
    validation: StratifiedThreeStageValidation

    @model_validator(mode="before")
    @classmethod
    def admit_before_nested_records(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            validation = raw.get("validation")
            n = (
                validation.nobs
                if isinstance(validation, StratifiedThreeStageValidation)
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
            frame = raw.get("ssu_frame")
            if not isinstance(frame, (list, tuple)) or not 1 <= len(frame) <= n:
                raise ValueError("Saved lower-stratum frame exceeds complete row dimensions.")
        return raw

    @field_validator(
        "psu",
        "ssu_strata",
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
                self.ssu_strata,
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
        return self

    def revalidate(self, data: Any):
        try:
            validated = type(self).model_validate(self)
        except ValueError as exc:
            raise AnalysisError(
                "invalid_survey_design", "Saved three-stage design is invalid."
            ) from exc
        options = validated.model_dump(
            exclude={"validation", "schema_version", "stages", "sampling", "ssu_frame"}
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
        current, normalized = _validate(data, ssu_frame=raw_frame, **options)
        if normalized != validated.ssu_frame:
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
        for v in (psu, ssu_strata, ssu, tsu, population_psu, population_ssu, population_tsu, strata)
        if v is not None
    ]
    if (
        len(roles) != (8 if strata is not None else 7)
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
    ) * 8192 > min(max_memory_mb * 1024**2, workspace_budget_bytes()):
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
    if (
        not isinstance(ssu_frame, Sequence)
        or isinstance(ssu_frame, (str, bytes))
        or not 1 <= len(ssu_frame) <= len(frame)
    ):
        raise AnalysisError(
            "invalid_survey_frame", "Declare a bounded explicit positive lower-stratum frame."
        )
    frame_roles = [psu, ssu_strata, population_ssu] + ([strata] if strata else [])
    normalized, population_lookup = [], {}
    for raw in ssu_frame:
        if (
            not isinstance(raw, Mapping)
            or len(raw) != len(frame_roles)
            or set(raw) != set(frame_roles)
        ):
            raise AnalysisError(
                "invalid_survey_frame",
                "Each frame record needs exactly the declared PSU/lower-stratum/population roles.",
            )
        h = _identity(raw[strata]) if strata else None
        p, g = _identity(raw[psu]), _identity(raw[ssu_strata])
        population = _number(raw[population_ssu], population=True)
        key = (h, p, g)
        if key in population_lookup:
            raise AnalysisError("invalid_survey_frame", "Duplicate typed lower-stratum frame cell.")
        population_lookup[key] = population
        normalized.append(
            StratifiedThreeStageFrameCell(
                stratum=h, psu=p, ssu_stratum=g, population_ssu=population
            )
        )
    normalized = tuple(sorted(normalized, key=_frame_sort_key))
    frame_sha = frame_digest(normalized)
    frame_indices = {(r.stratum, r.psu, r.ssu_stratum): i for i, r in enumerate(normalized)}
    checksum.update(frame_sha.encode())
    hs, ps, cs, qs, terminals = {}, {}, {}, {}, set()
    sr, pr, cr, qr = [], [], [], []
    for position, row in enumerate(frame[roles].itertuples(index=False, name=None)):
        values = dict(zip(roles, row))
        h = _identity(values[strata]) if strata else None
        p, g, q, t = (_identity(values[v]) for v in (psu, ssu_strata, ssu, tsu))
        N, M, L = (
            _number(values[v], population=True)
            for v in (population_psu, population_ssu, population_tsu)
        )
        cell_key = (h, p, g)
        if cell_key not in population_lookup or population_lookup[cell_key] != M:
            raise AnalysisError(
                "invalid_survey_frame",
                "Observed lower-stratum cells/counts must match the complete declared frame.",
            )
        if (h, p, g, q, t) in terminals:
            raise AnalysisError(
                "invalid_survey_nesting",
                "Each physical row must be a unique nested TSU within lower stratum/SSU/PSU.",
            )
        terminals.add((h, p, g, q, t))
        if h not in hs:
            hs[h] = len(sr)
            sr.append({"population_psu": N, "psu_indices": []})
        stratum = sr[hs[h]]
        if stratum["population_psu"] != N:
            raise AnalysisError(
                "invalid_survey_fpc", "PSU population counts must be constant within each stratum."
            )
        if (h, p) not in ps:
            ps[h, p] = len(pr)
            stratum["psu_indices"].append(ps[h, p])
            pr.append(
                {"stratum_index": hs[h], "cell_indices": [], "ssu_indices": [], "row_positions": []}
            )
        parent = pr[ps[h, p]]
        if cell_key not in cs:
            cs[cell_key] = len(cr)
            parent["cell_indices"].append(cs[cell_key])
            cr.append(
                {
                    "psu_index": ps[h, p],
                    "frame_index": frame_indices[cell_key],
                    "population_ssu": M,
                    "ssu_indices": [],
                    "row_positions": [],
                }
            )
        cell = cr[cs[cell_key]]
        if (h, p, g, q) not in qs:
            qs[h, p, g, q] = len(qr)
            cell["ssu_indices"].append(qs[h, p, g, q])
            parent["ssu_indices"].append(qs[h, p, g, q])
            qr.append(
                {
                    "psu_index": ps[h, p],
                    "cell_index": cs[cell_key],
                    "population_tsu": L,
                    "row_positions": [],
                }
            )
        child = qr[qs[h, p, g, q]]
        if child["population_tsu"] != L:
            raise AnalysisError(
                "invalid_survey_fpc",
                "TSU population counts must be constant within each nested SSU.",
            )
        parent["row_positions"].append(position)
        cell["row_positions"].append(position)
        child["row_positions"].append(position)
        checksum.update(
            json.dumps([h, p, g, q, t, N, M, L], ensure_ascii=False, separators=(",", ":")).encode()
        )
        checksum.update(b"\n")
    if set(cs) != set(population_lookup):
        raise AnalysisError(
            "invalid_survey_frame",
            "Every positive declared lower stratum must have sampled SSUs in an observed PSU.",
        )
    for records, size, population, name in (
        (sr, "psu_indices", "population_psu", "n_psu"),
        (cr, "ssu_indices", "population_ssu", "n_ssu"),
        (qr, "row_positions", "population_tsu", "n_tsu"),
    ):
        for record in records:
            record[name] = len(record[size])
            if record[population] < record[name] or record[name] == 1 and record[population] != 1:
                raise AnalysisError(
                    "invalid_survey_fpc",
                    "Each stage/cell population must dominate its sample; singleton requires census.",
                )
    for record in pr:
        record["n_ssu"] = len(record["ssu_indices"])
    weights = [0.0] * len(frame)
    for child in qr:
        cell = cr[child["cell_index"]]
        stratum = sr[pr[child["psu_index"]]["stratum_index"]]
        weight = (
            stratum["population_psu"]
            / stratum["n_psu"]
            * cell["population_ssu"]
            / cell["n_ssu"]
            * child["population_tsu"]
            / child["n_tsu"]
        )
        for position in child["row_positions"]:
            weights[position] = weight
    validation = StratifiedThreeStageValidation(
        nobs=len(frame),
        n_strata=len(sr),
        n_psu=len(pr),
        n_cells=len(cr),
        n_ssu=len(qr),
        design_df=len(pr) - len(sr),
        sum_weights=math.fsum(weights),
        design_input_sha256=checksum.hexdigest(),
        frame_input_sha256=frame_sha,
        strata=sr,
        psus=pr,
        cells=cr,
        ssus=qr,
    )
    return validation, normalized


def survey_stratified_three_stage_design(
    data: Any,
    *,
    psu: str,
    ssu_strata: str,
    ssu_frame: Sequence[Mapping[str, Any]],
    ssu: str,
    tsu: str,
    population_psu: str,
    population_ssu: str,
    population_tsu: str,
    strata: str | None = None,
    max_rows: int = 1_000_000,
    max_memory_mb: int = 64,
) -> SurveyStratifiedThreeStageDesign:
    """Declare three selections with per-lower-stratum M/m and explicit positive cell frame.

    Completely omitted external strata cannot be authenticated from caller inputs.
    """
    options = dict(
        psu=psu,
        ssu_strata=ssu_strata,
        ssu=ssu,
        tsu=tsu,
        population_psu=population_psu,
        population_ssu=population_ssu,
        population_tsu=population_tsu,
        strata=strata,
        max_rows=max_rows,
        max_memory_mb=max_memory_mb,
    )
    validation, normalized = _validate(data, ssu_frame=ssu_frame, **options)
    return SurveyStratifiedThreeStageDesign(**options, ssu_frame=normalized, validation=validation)
