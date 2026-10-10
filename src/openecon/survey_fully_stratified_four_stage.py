"""Four independent nested SRSWOR selections with explicit strata at every level."""

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
from openecon.survey_four_stage import FourStageStratum
from openecon.survey_fully_stratified_three_stage import _decoded_identity


class _Record(_SurveyRecord):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


def _path(raw):
    if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= 8:
        raise ValueError("A bounded canonical sampling path is required.")
    for i, value in enumerate(raw):
        if value is not None or i != 0:
            _decoded_identity(value)
    return tuple(None if v is None else tuple(v) for v in raw)


def _key(path):
    return tuple(None if v is None else tuple(v) for v in path)


def _json(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


class FullyStratifiedFourStageFrame(_Record):
    """Complete caller-declared positive stratum within an observed sampled parent."""

    path: tuple[Any, ...]
    population: StrictInt = Field(gt=0, le=2**53)

    _canonical = field_validator("path", mode="before")(_path)


class FullyStratifiedFourStageCell(_Record):
    path: tuple[Any, ...]
    parent_unit: StrictInt | None = Field(default=None, ge=0)
    population: StrictInt = Field(gt=0, le=2**53)
    sampled_indices: tuple[StrictInt, ...] = Field(min_length=1, max_length=1_000_000)

    _canonical = field_validator("path", mode="before")(_path)


class FullyStratifiedFourStageUnit(_Record):
    path: tuple[Any, ...]
    cell_index: StrictInt = Field(ge=0)
    row_positions: tuple[StrictInt, ...] = Field(min_length=1, max_length=1_000_000)

    _canonical = field_validator("path", mode="before")(_path)


def _indices(values, size):
    return (
        bool(values) and list(values) == sorted(set(values)) and all(0 <= i < size for i in values)
    )


class FullyStratifiedFourStageValidation(_Record):
    nobs: StrictInt = Field(gt=0, le=1_000_000)
    n_strata: StrictInt = Field(gt=0, le=1_000_000)
    n_psu: StrictInt = Field(gt=0, le=1_000_000)
    n_ssu: StrictInt = Field(gt=0, le=1_000_000)
    n_tsu: StrictInt = Field(gt=0, le=1_000_000)
    design_df: StrictInt = Field(ge=0)
    sum_weights: float = Field(gt=0, allow_inf_nan=False, strict=True)
    design_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: tuple[tuple[FullyStratifiedFourStageCell, ...], ...]
    psus: tuple[FullyStratifiedFourStageUnit, ...]
    ssus: tuple[FullyStratifiedFourStageUnit, ...]
    tsus: tuple[FullyStratifiedFourStageUnit, ...]
    row_paths: tuple[tuple[Any, ...], ...]

    @model_validator(mode="before")
    @classmethod
    def bounded_geometry(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            n = raw.get("nobs")
            if type(n) is not int or not 1 <= n <= 1_000_000:
                raise ValueError("Complete row count exceeds bounded admission.")
            levels = raw.get("cells")
            if not isinstance(levels, (list, tuple)) or len(levels) != 4:
                raise ValueError("Exactly four complete stratum levels are required.")
            for records, index_field in [
                *((v, "sampled_indices") for v in levels),
                *((raw.get(k), "row_positions") for k in ("psus", "ssus", "tsus")),
            ]:
                if not isinstance(records, (list, tuple)) or not 1 <= len(records) <= n:
                    raise ValueError("Complete sampling records exceed the row budget.")
                count = 0
                for record in records:
                    values = (
                        record.get(index_field)
                        if isinstance(record, Mapping)
                        else getattr(record, index_field, None)
                    )
                    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= n:
                        raise ValueError("Sampling index lists exceed the row budget.")
                    count += len(values)
                    if count > n:
                        raise ValueError("Sampling index storage exceeds the row budget.")
            rows = raw.get("row_paths")
            if not isinstance(rows, (list, tuple)) or len(rows) != n:
                raise ValueError("Every physical FSU needs one complete bounded path.")
            for path in rows:
                _path(path)
                if len(path) != 8:
                    raise ValueError("Physical FSU paths require all four stratum/unit pairs.")
        return raw

    @field_validator("row_paths", mode="before")
    @classmethod
    def canonical_rows(cls, rows):
        return tuple(_path(p) for p in rows)

    @model_validator(mode="after")
    def geometry(self):
        units = (self.psus, self.ssus, self.tsus)
        sizes = (self.n_psu, self.n_ssu, self.n_tsu, self.nobs)
        if (
            tuple(map(len, units)) != sizes[:3]
            or list(sizes) != sorted(sizes)
            or len(self.cells[0]) != self.n_strata
            or self.design_df != self.n_psu - self.n_strata
        ):
            raise ValueError("Four-level dimensions or first-stage reference df differ.")
        rows = tuple(_key(p) for p in self.row_paths)
        if len(set(rows)) != self.nobs:
            raise ValueError("Each physical FSU must have a unique complete typed path.")
        for stage, cells in enumerate(self.cells):
            seen, paths = [], set()
            descendants = [[] for _ in range(sizes[stage - 1])] if stage else None
            for ci, cell in enumerate(cells):
                path, indices = _key(cell.path), cell.sampled_indices
                n = len(indices)
                if (
                    len(path) != 2 * stage + 1
                    or path in paths
                    or not _indices(indices, sizes[stage])
                    or cell.population < n
                    or n == 1
                    and cell.population != 1
                ):
                    raise ValueError(
                        "Stratum path, population, membership or census singleton differs."
                    )
                paths.add(path)
                if stage == 0:
                    if cell.parent_unit is not None:
                        raise ValueError("First-stage strata cannot have sampled parents.")
                elif (
                    cell.parent_unit is None
                    or cell.parent_unit >= sizes[stage - 1]
                    or path[:-1] != _key(units[stage - 1][cell.parent_unit].path)
                ):
                    raise ValueError(
                        "A lower stratum must belong to its exact complete sampled parent."
                    )
                for index in indices:
                    if stage < 3:
                        unit = units[stage][index]
                        if (
                            unit.cell_index != ci
                            or len(unit.path) != 2 * stage + 2
                            or _key(unit.path)[:-1] != path
                            or not _indices(unit.row_positions, self.nobs)
                            or any(
                                rows[i][: 2 * stage + 2] != _key(unit.path)
                                for i in unit.row_positions
                            )
                        ):
                            raise ValueError(
                                "Sampled-unit path, stratum or physical descendants differ."
                            )
                        physical = unit.row_positions
                    else:
                        if rows[index][:-1] != path:
                            raise ValueError(
                                "An FSU belongs to a different declared terminal stratum."
                            )
                        physical = (index,)
                    if stage:
                        descendants[cell.parent_unit].extend(physical)
                seen.extend(indices)
            if sorted(seen) != list(range(sizes[stage])):
                raise ValueError("Each sampled unit must occur in exactly one complete stratum.")
            if stage:
                for unit, positions in zip(units[stage - 1], descendants):
                    if sorted(positions) != list(unit.row_positions):
                        raise ValueError(
                            "Sampled-parent rows must equal all lower-stratum descendants."
                        )
        if not math.isclose(math.fsum(self.weights), self.sum_weights, rel_tol=1e-12):
            raise ValueError("Derived inverse-probability weight sum differs.")
        return self

    @property
    def strata(self):
        return tuple(
            FourStageStratum(
                n_psu=len(c.sampled_indices),
                population_psu=c.population,
                psu_indices=c.sampled_indices,
            )
            for c in self.cells[0]
        )

    @property
    def weights(self):
        units = (self.psus, self.ssus, self.tsus)
        values = [0.0] * self.nobs
        for cell in self.cells[3]:
            weight = cell.population / len(cell.sampled_indices)
            parent = cell.parent_unit
            for stage in (2, 1, 0):
                ancestor = self.cells[stage][units[stage][parent].cell_index]
                weight *= ancestor.population / len(ancestor.sampled_indices)
                parent = ancestor.parent_unit
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Four-stage weights must be positive finite float64 values.")
            for i in cell.sampled_indices:
                values[i] = weight
        return tuple(values)


def admission(n, memory_mb, *, k=0):
    return plan_workspace(
        "fully stratified four-stage survey inference",
        {
            "all four stratum/unit paths, declarations and JSON replay": n * 32768,
            "target, complete nested totals and four covariance buffers": (n * k + k * k) * 384,
        },
        budget_bytes=min(memory_mb * 1024**2, workspace_budget_bytes()),
    )


class SurveyFullyStratifiedFourStageDesign(_Record):
    """Exactly four selections; declared frames are provenance, not authenticated evidence."""

    schema_version: Literal["survey-fully-stratified-four-stage-design-v1"] = (
        "survey-fully-stratified-four-stage-design-v1"
    )
    stages: Literal[4] = 4
    sampling: Literal["sequential-srswor"] = "sequential-srswor"
    psu: str
    ssu: str
    tsu: str
    fsu: str
    ssu_strata: str
    tsu_strata: str
    fsu_strata: str
    population_psu: str
    strata: str | None = None
    ssu_frame: tuple[FullyStratifiedFourStageFrame, ...]
    tsu_frame: tuple[FullyStratifiedFourStageFrame, ...]
    fsu_frame: tuple[FullyStratifiedFourStageFrame, ...]
    max_rows: StrictInt = Field(default=1_000_000, ge=1, le=1_000_000)
    max_memory_mb: StrictInt = Field(default=64, ge=1, le=512)
    validation: FullyStratifiedFourStageValidation

    @model_validator(mode="before")
    @classmethod
    def admit(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            v = raw.get("validation")
            n = v.get("nobs") if isinstance(v, Mapping) else getattr(v, "nobs", None)
            memory, rows = raw.get("max_memory_mb", 64), raw.get("max_rows", 1_000_000)
            if (
                type(n) is not int
                or type(rows) is not int
                or not 1 <= n <= rows <= 1_000_000
                or type(memory) is not int
                or not 1 <= memory <= 512
            ):
                raise ValueError("Saved geometry exceeds explicit admission limits.")
            admission(n, memory)
            for name in ("ssu_frame", "tsu_frame", "fsu_frame"):
                frame = raw.get(name)
                if not isinstance(frame, (tuple, list)) or not 1 <= len(frame) <= n:
                    raise ValueError("Explicit complete frames exceed the admitted sample.")
        return raw

    @field_validator(
        "psu",
        "ssu",
        "tsu",
        "fsu",
        "ssu_strata",
        "tsu_strata",
        "fsu_strata",
        "population_psu",
        "strata",
        mode="before",
    )
    @classmethod
    def name(cls, value):
        if value is not None and (
            not isinstance(value, str) or not value.strip() or len(value) > 200
        ):
            raise ValueError("Use nonempty role names of at most 200 characters.")
        return value

    @field_validator("stages", mode="before")
    @classmethod
    def four(cls, value):
        if type(value) is not int or value != 4:
            raise ValueError("Exactly four real sampling stages are required.")
        return value

    @model_validator(mode="after")
    def roles_and_frames(self):
        roles = [
            self.psu,
            self.ssu,
            self.tsu,
            self.fsu,
            self.ssu_strata,
            self.tsu_strata,
            self.fsu_strata,
            self.population_psu,
        ]
        if self.strata is not None:
            roles.append(self.strata)
        if len(set(roles)) != len(roles):
            raise ValueError("All design roles must have distinct names.")
        for stage, frames in enumerate((self.ssu_frame, self.tsu_frame, self.fsu_frame), 1):
            pairs = [(_key(f.path), f.population) for f in frames]
            actual = [(_key(c.path), c.population) for c in self.validation.cells[stage]]
            if (
                any(len(path) != 2 * stage + 1 for path, _ in pairs)
                or len(set(pairs)) != len(pairs)
                or sorted(pairs, key=_json) != pairs
                or set(pairs) != set(actual)
            ):
                raise ValueError(
                    "Every declared positive frame stratum must have sampled coverage and matching geometry."
                )
        return self

    def revalidate(self, data):
        try:
            current = type(self).model_validate(self)
            options = current.model_dump(
                exclude={"validation", "schema_version", "stages", "sampling"}
            )
            validation, _ = _validate(data, **options)
        except (ValueError, TypeError) as exc:
            raise AnalysisError(
                "invalid_survey_design", "Complete typed geometry failed revalidation."
            ) from exc
        if validation != current.validation:
            raise AnalysisError(
                "survey_design_changed",
                "Ordered physical sample or complete four-stage declaration changed.",
            )
        return validation


def _frames(raw, stage, n):
    if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= n:
        raise AnalysisError(
            "invalid_survey_frame", "Supply a nonempty bounded complete lower-stratum frame."
        )
    names = ("stratum", "psu", "ssu_stratum", "ssu", "tsu_stratum", "tsu", "fsu_stratum")[
        : 2 * stage + 1
    ]
    population = ("population_ssu", "population_tsu", "population_fsu")[stage - 1]
    parsed = []
    for record in raw:
        try:
            if (
                isinstance(record, FullyStratifiedFourStageFrame)
                or isinstance(record, Mapping)
                and set(record) == {"path", "population"}
            ):
                item = FullyStratifiedFourStageFrame.model_validate(record)
            else:
                if not isinstance(record, Mapping) or set(record) != set(names) | {population}:
                    raise ValueError(
                        "Declare exactly the complete parent/stratum identities and population."
                    )
                path = tuple(
                    None if name == "stratum" and record[name] is None else _identity(record[name])
                    for name in names
                )
                item = FullyStratifiedFourStageFrame(
                    path=path, population=_number(record[population], population=True)
                )
            if len(item.path) != 2 * stage + 1:
                raise ValueError("The frame path belongs to another sampling stage.")
            parsed.append(item)
        except (ValueError, TypeError) as exc:
            raise AnalysisError(
                "invalid_survey_frame", "Invalid bounded lower-stratum frame record."
            ) from exc
    paths = [_key(f.path) for f in parsed]
    if len(set(paths)) != len(paths):
        raise AnalysisError(
            "invalid_survey_frame", "Each lower-stratum frame path must occur exactly once."
        )
    return tuple(sorted(parsed, key=lambda f: _json((_key(f.path), f.population))))


def _validate(
    data,
    *,
    psu,
    ssu,
    tsu,
    fsu,
    ssu_strata,
    tsu_strata,
    fsu_strata,
    population_psu,
    strata,
    ssu_frame,
    tsu_frame,
    fsu_frame,
    max_rows,
    max_memory_mb,
):
    import pandas as pd

    if (
        type(max_rows) is not int
        or not 1 <= max_rows <= 1_000_000
        or type(max_memory_mb) is not int
        or not 1 <= max_memory_mb <= 512
    ):
        raise AnalysisError(
            "survey_budget", "Declare integer row and memory budgets in supported bounds."
        )
    roles = [psu, ssu_strata, ssu, tsu_strata, tsu, fsu_strata, fsu, population_psu] + (
        [strata] if strata is not None else []
    )
    if any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in roles) or len(
        set(roles)
    ) != len(roles):
        raise AnalysisError("invalid_survey_design", "Design roles require distinct bounded names.")
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, Mapping):
        if any(type(v).__module__.startswith("torch") for v in data.values()):
            raise AnalysisError("unsupported_survey_input", "Use resident scalar tabular columns.")
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
                "invalid_data", "Design columns require consistent resident rows."
            ) from exc
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) > max_rows:
            raise AnalysisError("survey_budget", "Complete geometry exceeds max_rows.")
        admission(len(data), max_memory_mb)
        if any(not isinstance(row, Mapping) or any(v not in row for v in roles) for row in data):
            raise AnalysisError("invalid_data", "Each row must contain all design roles.")
        frame = pd.DataFrame([{v: row[v] for v in roles} for row in data])
    else:
        raise AnalysisError(
            "unsupported_survey_input", "Use a resident table or scalar row/column mapping."
        )
    if frame.columns.has_duplicates or any(v not in frame for v in roles):
        raise AnalysisError("missing_column", "Design roles must exist in a unique-column table.")
    if not 1 <= len(frame) <= max_rows:
        raise AnalysisError("survey_budget", "Complete nonempty sample must fit max_rows.")
    admission(len(frame), max_memory_mb)
    if sum(int(frame[v].memory_usage(index=False, deep=True)) for v in roles) + len(
        frame
    ) * 32768 > min(max_memory_mb * 1024**2, workspace_budget_bytes()):
        raise AnalysisError("survey_budget", "Full typed geometry exceeds the resident budget.")
    frames = tuple(
        _frames(raw, stage, len(frame))
        for stage, raw in enumerate((ssu_frame, tsu_frame, fsu_frame), 1)
    )
    populations = [{_key(f.path): f.population for f in records} for records in frames]
    cell_ids = [{} for _ in range(4)]
    unit_ids = [{} for _ in range(3)]
    cells = [[] for _ in range(4)]
    units = [[] for _ in range(3)]
    paths = []
    checksum = hashlib.sha256(_json([(v, str(frame[v].dtype)) for v in roles]).encode())
    for role in roles:
        dtype = frame[role].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            checksum.update(
                _json([role, dtype.ordered, [_identity(c) for c in dtype.categories]]).encode()
            )
    for position, row in enumerate(frame[roles].itertuples(index=False, name=None)):
        values = dict(zip(roles, row))
        path = (
            None if strata is None else _identity(values[strata]),
            *(
                _identity(values[v])
                for v in (psu, ssu_strata, ssu, tsu_strata, tsu, fsu_strata, fsu)
            ),
        )
        paths.append(path)
        N = _number(values[population_psu], population=True)
        parent = None
        for stage in range(4):
            cp = path[: 2 * stage + 1]
            population = N if stage == 0 else populations[stage - 1].get(cp)
            if population is None:
                raise AnalysisError(
                    "invalid_survey_frame",
                    "Every observed lower stratum needs an explicit positive population frame.",
                )
            if cp not in cell_ids[stage]:
                cell_ids[stage][cp] = len(cells[stage])
                cells[stage].append(
                    dict(path=cp, parent_unit=parent, population=population, sampled_indices=[])
                )
            ci = cell_ids[stage][cp]
            cell = cells[stage][ci]
            if cell["population"] != population or cell["parent_unit"] != parent:
                raise AnalysisError(
                    "invalid_survey_fpc", "Stratum population or complete parent must be constant."
                )
            if stage < 3:
                up = path[: 2 * stage + 2]
                if up not in unit_ids[stage]:
                    unit_ids[stage][up] = len(units[stage])
                    cell["sampled_indices"].append(unit_ids[stage][up])
                    units[stage].append(dict(path=up, cell_index=ci, row_positions=[]))
                parent = unit_ids[stage][up]
                units[stage][parent]["row_positions"].append(position)
            else:
                cell["sampled_indices"].append(position)
        checksum.update(_json([path, N]).encode())
        checksum.update(b"\n")
    for stage, declared in enumerate(populations, 1):
        if set(declared) != set(cell_ids[stage]):
            raise AnalysisError(
                "invalid_survey_frame",
                "Every declared positive lower stratum requires sampled coverage within an observed parent.",
            )
    checksum.update(
        _json([[f.model_dump(mode="json") for f in records] for records in frames]).encode()
    )
    # Construct only after every complete stratum and physical path is admitted.
    records = tuple(tuple(FullyStratifiedFourStageCell(**c) for c in level) for level in cells)
    unit_records = tuple(tuple(FullyStratifiedFourStageUnit(**u) for u in level) for level in units)
    weights = []
    for path in paths:
        w = 1.0
        for stage in range(4):
            c = records[stage][cell_ids[stage][path[: 2 * stage + 1]]]
            w *= c.population / len(c.sampled_indices)
        weights.append(w)
    try:
        validation = FullyStratifiedFourStageValidation(
            nobs=len(frame),
            n_strata=len(cells[0]),
            n_psu=len(units[0]),
            n_ssu=len(units[1]),
            n_tsu=len(units[2]),
            design_df=len(units[0]) - len(cells[0]),
            sum_weights=math.fsum(weights),
            design_input_sha256=checksum.hexdigest(),
            cells=records,
            psus=unit_records[0],
            ssus=unit_records[1],
            tsus=unit_records[2],
            row_paths=paths,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        raise AnalysisError(
            "invalid_survey_fpc",
            "Every stage needs lawful populations, complete nesting and census-only singletons.",
        ) from exc
    return validation, frames


def survey_fully_stratified_four_stage_design(
    data: Any,
    *,
    psu: str,
    ssu: str,
    tsu: str,
    fsu: str,
    ssu_strata: str,
    tsu_strata: str,
    fsu_strata: str,
    population_psu: str,
    ssu_frame,
    tsu_frame,
    fsu_frame,
    strata: str | None = None,
    max_rows: int = 1_000_000,
    max_memory_mb: int = 64,
) -> SurveyFullyStratifiedFourStageDesign:
    """Declare complete positive lower strata and four sequential SRSWOR selections."""
    options = dict(
        psu=psu,
        ssu=ssu,
        tsu=tsu,
        fsu=fsu,
        ssu_strata=ssu_strata,
        tsu_strata=tsu_strata,
        fsu_strata=fsu_strata,
        population_psu=population_psu,
        strata=strata,
        ssu_frame=ssu_frame,
        tsu_frame=tsu_frame,
        fsu_frame=fsu_frame,
        max_rows=max_rows,
        max_memory_mb=max_memory_mb,
    )
    validation, frames = _validate(data, **options)
    options.update(zip(("ssu_frame", "tsu_frame", "fsu_frame"), frames))
    return SurveyFullyStratifiedFourStageDesign(**options, validation=validation)
