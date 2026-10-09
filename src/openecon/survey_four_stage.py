"""Exactly four nested sequential SRSWOR stages with explicit finite populations."""

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


class FourStageStratum(_Record):
    n_psu: StrictInt = Field(gt=0)
    population_psu: StrictInt = Field(gt=0, le=2**53)
    psu_indices: tuple[StrictInt, ...]


class FourStagePSU(_Record):
    stratum_index: StrictInt = Field(ge=0)
    n_ssu: StrictInt = Field(gt=0)
    population_ssu: StrictInt = Field(gt=0, le=2**53)
    ssu_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class FourStageSSU(_Record):
    psu_index: StrictInt = Field(ge=0)
    n_tsu: StrictInt = Field(gt=0)
    population_tsu: StrictInt = Field(gt=0, le=2**53)
    tsu_indices: tuple[StrictInt, ...]
    row_positions: tuple[StrictInt, ...]


class FourStageTSU(_Record):
    ssu_index: StrictInt = Field(ge=0)
    n_fsu: StrictInt = Field(gt=0)
    population_fsu: StrictInt = Field(gt=0, le=2**53)
    row_positions: tuple[StrictInt, ...]


def _indices(values, size):
    return (
        bool(values) and list(values) == sorted(set(values)) and all(0 <= i < size for i in values)
    )


class FourStageValidation(_Record):
    nobs: StrictInt = Field(gt=0, le=1_000_000)
    n_strata: StrictInt = Field(gt=0, le=1_000_000)
    n_psu: StrictInt = Field(gt=0, le=1_000_000)
    n_ssu: StrictInt = Field(gt=0, le=1_000_000)
    n_tsu: StrictInt = Field(gt=0, le=1_000_000)
    design_df: StrictInt = Field(ge=0)
    sum_weights: float = Field(gt=0, allow_inf_nan=False, strict=True)
    design_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: tuple[FourStageStratum, ...]
    psus: tuple[FourStagePSU, ...]
    ssus: tuple[FourStageSSU, ...]
    tsus: tuple[FourStageTSU, ...]

    @model_validator(mode="before")
    @classmethod
    def bounded_geometry(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            n = raw.get("nobs")
            if type(n) is not int or not 1 <= n <= 1_000_000:
                raise ValueError("Complete row count is outside the bounded design.")
            for key in ("strata", "psus", "ssus", "tsus"):
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
                    for field in ("row_positions", "psu_indices", "ssu_indices", "tsu_indices"):
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
        sizes = (self.n_strata, self.n_psu, self.n_ssu, self.n_tsu, self.nobs)
        if (
            tuple(map(len, (self.strata, self.psus, self.ssus, self.tsus))) != sizes[:4]
            or list(sizes) != sorted(sizes)
            or self.design_df != self.n_psu - self.n_strata
        ):
            raise ValueError("Four-stage dimensions or first-stage df are inconsistent.")
        seen_psus, seen_ssus, seen_tsus, seen_rows = [], [], [], []
        for h, stratum in enumerate(self.strata):
            if (
                len(stratum.psu_indices) != stratum.n_psu
                or not _indices(stratum.psu_indices, self.n_psu)
                or stratum.population_psu < stratum.n_psu
                or stratum.n_psu == 1
                and stratum.population_psu != 1
                or any(self.psus[i].stratum_index != h for i in stratum.psu_indices)
            ):
                raise ValueError("Stage-one counts, parent assignment or census singleton differ.")
            seen_psus.extend(stratum.psu_indices)
        for (
            records,
            children,
            parent_field,
            child_field,
            child_parent,
            count,
            population,
            size,
            parent_size,
            seen,
        ) in (
            (
                self.psus,
                self.ssus,
                "stratum_index",
                "ssu_indices",
                "psu_index",
                "n_ssu",
                "population_ssu",
                self.n_ssu,
                self.n_strata,
                seen_ssus,
            ),
            (
                self.ssus,
                self.tsus,
                "psu_index",
                "tsu_indices",
                "ssu_index",
                "n_tsu",
                "population_tsu",
                self.n_tsu,
                self.n_psu,
                seen_tsus,
            ),
        ):
            for i, record in enumerate(records):
                indices = getattr(record, child_field)
                n, N = getattr(record, count), getattr(record, population)
                if (
                    getattr(record, parent_field) >= parent_size
                    or len(indices) != n
                    or not _indices(indices, size)
                    or not _indices(record.row_positions, self.nobs)
                    or N < n
                    or n == 1
                    and N != 1
                ):
                    raise ValueError("Intermediate counts, membership or census singleton differ.")
                descendants = []
                for j in indices:
                    if getattr(children[j], child_parent) != i:
                        raise ValueError("Nested sampled-unit parent assignment differs.")
                    descendants.extend(children[j].row_positions)
                if sorted(descendants) != list(record.row_positions):
                    raise ValueError("Physical rows must equal complete nested descendants.")
                seen.extend(indices)
        for terminal in self.tsus:
            if (
                terminal.ssu_index >= self.n_ssu
                or len(terminal.row_positions) != terminal.n_fsu
                or not _indices(terminal.row_positions, self.nobs)
                or terminal.population_fsu < terminal.n_fsu
                or terminal.n_fsu == 1
                and terminal.population_fsu != 1
            ):
                raise ValueError("Fourth-stage rows, population or census singleton differ.")
            seen_rows.extend(terminal.row_positions)
        for seen, size in (
            (seen_psus, self.n_psu),
            (seen_ssus, self.n_ssu),
            (seen_tsus, self.n_tsu),
            (seen_rows, self.nobs),
        ):
            if sorted(seen) != list(range(size)):
                raise ValueError(
                    "Every sampled unit and physical FSU must have exactly one parent."
                )
        if not math.isclose(math.fsum(self.weights), self.sum_weights, rel_tol=1e-12):
            raise ValueError("Derived four-stage weight sum differs from the complete geometry.")
        return self

    @property
    def weights(self):
        values = [0.0] * self.nobs
        for terminal in self.tsus:
            child = self.ssus[terminal.ssu_index]
            parent = self.psus[child.psu_index]
            stratum = self.strata[parent.stratum_index]
            w = (
                stratum.population_psu
                / stratum.n_psu
                * parent.population_ssu
                / parent.n_ssu
                * child.population_tsu
                / child.n_tsu
                * terminal.population_fsu
                / terminal.n_fsu
            )
            for position in terminal.row_positions:
                values[position] = w
        return tuple(values)


def admission(n, memory_mb, *, k=0):
    return plan_workspace(
        "four-stage survey inference",
        {
            "complete four-level identities, indices and JSON replay": n * 24576,
            "target, nested totals and four covariance serialization buffers": (n * k + k * k)
            * 384,
        },
        budget_bytes=min(memory_mb * 1024**2, workspace_budget_bytes()),
    )


class SurveyFourStageDesign(_Record):
    """Frozen four-stage geometry; provenance hashes do not authenticate sampling."""

    schema_version: Literal["survey-four-stage-design-v1"] = "survey-four-stage-design-v1"
    stages: Literal[4] = 4
    sampling: Literal["sequential-srswor"] = "sequential-srswor"
    psu: str
    ssu: str
    tsu: str
    fsu: str
    population_psu: str
    population_ssu: str
    population_tsu: str
    population_fsu: str
    strata: str | None = None
    max_rows: StrictInt = Field(default=1_000_000, ge=1, le=1_000_000)
    max_memory_mb: StrictInt = Field(default=64, ge=1, le=512)
    validation: FourStageValidation

    @model_validator(mode="before")
    @classmethod
    def admit_before_nested_records(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            validation = raw.get("validation")
            n = (
                validation.nobs
                if isinstance(validation, FourStageValidation)
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
        "fsu",
        "population_psu",
        "population_ssu",
        "population_tsu",
        "population_fsu",
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
    def exactly_four(cls, value):
        if type(value) is not int or value != 4:
            raise ValueError("Exactly four SRSWOR stages are supported.")
        return value

    @model_validator(mode="after")
    def roles_and_budget(self):
        roles = [
            v
            for v in (
                self.psu,
                self.ssu,
                self.tsu,
                self.fsu,
                self.population_psu,
                self.population_ssu,
                self.population_tsu,
                self.population_fsu,
                self.strata,
            )
            if v is not None
        ]
        if len(set(roles)) != len(roles) or self.validation.nobs > self.max_rows:
            raise ValueError("Design roles or row budget are inconsistent.")
        admission(self.validation.nobs, self.max_memory_mb)
        return self

    def revalidate(self, data: Any):
        try:
            validated = type(self).model_validate(self)
        except (ValueError, TypeError) as exc:
            raise AnalysisError(
                "invalid_survey_design", "Saved four-stage declaration is invalid."
            ) from exc
        current = _validate(
            data,
            **validated.model_dump(exclude={"validation", "schema_version", "stages", "sampling"}),
        )
        if current != validated.validation:
            raise AnalysisError(
                "survey_design_changed",
                "Four-stage inputs or ordered geometry changed; declare again.",
            )
        return current


def _validate(
    data,
    *,
    psu,
    ssu,
    tsu,
    fsu,
    population_psu,
    population_ssu,
    population_tsu,
    population_fsu,
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
            ssu,
            tsu,
            fsu,
            population_psu,
            population_ssu,
            population_tsu,
            population_fsu,
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
            "invalid_survey_design", "Four-stage roles need distinct nonempty names."
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
    ) * 24576 > min(max_memory_mb * 1024**2, workspace_budget_bytes()):
        raise AnalysisError(
            "survey_budget", "Complete four-level geometry exceeds the resident budget."
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
    hs, ps, qs, ts, terminals = {}, {}, {}, {}, set()
    sr, pr, qr, tr = [], [], [], []
    for position, row in enumerate(frame[roles].itertuples(index=False, name=None)):
        values = dict(zip(roles, row))
        h = _identity(values[strata]) if strata else ("unstratified",)
        p, q, t, u = (_identity(values[v]) for v in (psu, ssu, tsu, fsu))
        N, M, L, K = (
            _number(values[v], population=True)
            for v in (population_psu, population_ssu, population_tsu, population_fsu)
        )
        if (h, p, q, t, u) in terminals:
            raise AnalysisError(
                "invalid_survey_nesting",
                "Each physical row must be a unique FSU within its nested TSU/SSU/PSU/stratum.",
            )
        terminals.add((h, p, q, t, u))
        if h not in hs:
            hs[h] = len(sr)
            sr.append({"population_psu": N, "psu_indices": []})
        stratum = sr[hs[h]]
        if stratum["population_psu"] != N:
            raise AnalysisError(
                "invalid_survey_fpc", "PSU population must be constant within each stratum."
            )
        if (h, p) not in ps:
            ps[h, p] = len(pr)
            stratum["psu_indices"].append(ps[h, p])
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
                "invalid_survey_fpc", "SSU population must be constant within each nested PSU."
            )
        if (h, p, q) not in qs:
            qs[h, p, q] = len(qr)
            parent["ssu_indices"].append(qs[h, p, q])
            qr.append(
                {"psu_index": ps[h, p], "population_tsu": L, "tsu_indices": [], "row_positions": []}
            )
        child = qr[qs[h, p, q]]
        if child["population_tsu"] != L:
            raise AnalysisError(
                "invalid_survey_fpc", "TSU population must be constant within each nested SSU."
            )
        if (h, p, q, t) not in ts:
            ts[h, p, q, t] = len(tr)
            child["tsu_indices"].append(ts[h, p, q, t])
            tr.append({"ssu_index": qs[h, p, q], "population_fsu": K, "row_positions": []})
        terminal = tr[ts[h, p, q, t]]
        if terminal["population_fsu"] != K:
            raise AnalysisError(
                "invalid_survey_fpc", "FSU population must be constant within each nested TSU."
            )
        for record in (parent, child, terminal):
            record["row_positions"].append(position)
        checksum.update(
            json.dumps(
                [h, p, q, t, u, N, M, L, K], ensure_ascii=False, separators=(",", ":")
            ).encode()
        )
        checksum.update(b"\n")
    for records, indices, population, count in (
        (sr, "psu_indices", "population_psu", "n_psu"),
        (pr, "ssu_indices", "population_ssu", "n_ssu"),
        (qr, "tsu_indices", "population_tsu", "n_tsu"),
        (tr, "row_positions", "population_fsu", "n_fsu"),
    ):
        for record in records:
            record[count] = len(record[indices])
            if record[population] < record[count] or record[count] == 1 and record[population] != 1:
                raise AnalysisError(
                    "invalid_survey_fpc",
                    "Each stage population must dominate its sample; singleton requires census.",
                )
    weights = [0.0] * len(frame)
    for terminal in tr:
        child = qr[terminal["ssu_index"]]
        parent = pr[child["psu_index"]]
        stratum = sr[parent["stratum_index"]]
        w = (
            stratum["population_psu"]
            / stratum["n_psu"]
            * parent["population_ssu"]
            / parent["n_ssu"]
            * child["population_tsu"]
            / child["n_tsu"]
            * terminal["population_fsu"]
            / terminal["n_fsu"]
        )
        for i in terminal["row_positions"]:
            weights[i] = w
    return FourStageValidation(
        nobs=len(frame),
        n_strata=len(sr),
        n_psu=len(pr),
        n_ssu=len(qr),
        n_tsu=len(tr),
        design_df=len(pr) - len(sr),
        sum_weights=math.fsum(weights),
        design_input_sha256=checksum.hexdigest(),
        strata=sr,
        psus=pr,
        ssus=qr,
        tsus=tr,
    )


def survey_four_stage_design(
    data: Any,
    *,
    psu: str,
    ssu: str,
    tsu: str,
    fsu: str,
    population_psu: str,
    population_ssu: str,
    population_tsu: str,
    population_fsu: str,
    strata: str | None = None,
    max_rows: int = 1_000_000,
    max_memory_mb: int = 64,
) -> SurveyFourStageDesign:
    """Declare four SRSWOR stages with N/n × M/m × L/l × K/k inverse-probability weights."""
    options = dict(
        psu=psu,
        ssu=ssu,
        tsu=tsu,
        fsu=fsu,
        population_psu=population_psu,
        population_ssu=population_ssu,
        population_tsu=population_tsu,
        population_fsu=population_fsu,
        strata=strata,
        max_rows=max_rows,
        max_memory_mb=max_memory_mb,
    )
    return SurveyFourStageDesign(**options, validation=_validate(data, **options))
