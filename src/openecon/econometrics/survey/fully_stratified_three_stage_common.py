"""Complete three-stage admission, covariance and explicit reference-df inference."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
from numbers import Real
import pandas as pd
import torch
from openecon.analysis_contracts import AnalysisError
from openecon.survey import _identity, _scalar
from openecon.survey_fully_stratified_three_stage import (
    SurveyFullyStratifiedThreeStageDesign,
    admission,
)
from .regression_common import cpu_call
from .common import FLOAT, MAX_TARGETS, Target, columns, digest, finite, number
from .two_stage_common import (
    DF_CONVENTION,
    validate_covariance as validate_covariance,
    validate_identity as validate_identity,
    validate_sample_metadata as _validate_sample_metadata,
    inference_frame as _reference_frame,
)

DESCRIPTIVE_WEIGHT_SEMANTICS = (
    "derived inverse probabilities N/n * M_cell/m_cell * L_cell/l_cell; sequential SRSWOR"
)


@cpu_call
def stage_covariances(weighted_rows, design):
    """Three SRSWOR components of complete full-weighted linearized rows.

    Stage two centers SSU totals independently within declared SSU strata;
    stage three centers rows independently within declared TSU strata.
    The lower-stage multipliers f1 and f1*f2_cell remove the sampling noise already
    represented by the higher-stage between-unit components.
    """
    n, k = weighted_rows.shape
    if n != design.validation.nobs or not 1 <= k <= MAX_TARGETS:
        raise AnalysisError(
            "invalid_survey_target", "Weighted scores differ from the complete three-level design."
        )
    admission(n, design.max_memory_mb, k=k)
    finite(weighted_rows, "Three-stage weighted linearized rows")
    v = design.validation
    psu_totals = torch.stack([weighted_rows[list(p.row_positions)].sum(0) for p in v.psus])
    ssu_totals = torch.stack([weighted_rows[list(q.row_positions)].sum(0) for q in v.ssus])
    parts = [torch.zeros((k, k), dtype=FLOAT) for _ in range(3)]

    def add(stage, block, f, prefix):
        if f < 1:
            centered = block - block.mean(0)
            parts[stage] += (
                prefix * (1 - f) * len(block) / (len(block) - 1) * (centered.T @ centered)
            )

    for s in v.strata:
        f1 = s.n_psu / s.population_psu
        add(0, psu_totals[list(s.psu_indices)], f1, 1.0)
        for i in s.psu_indices:
            p = v.psus[i]
            for c in p.cell_indices:
                cell = v.cells[c]
                f2 = cell.n_ssu / cell.population_ssu
                add(1, ssu_totals[list(cell.ssu_indices)], f2, f1)
                for j in cell.ssu_indices:
                    for r in v.ssus[j].tsu_cell_indices:
                        terminal = v.tsu_cells[r]
                        f3 = terminal.n_tsu / terminal.population_tsu
                        add(2, weighted_rows[list(terminal.row_positions)], f3, f1 * f2)
    return tuple(finite((part + part.T) / 2, "Stage-specific covariance") for part in parts)


@cpu_call
def fully_stratified_three_stage_covariance(weighted_rows, design):
    a, b, c = stage_covariances(weighted_rows, design)
    return finite(a + b + c, "Three-stage covariance")


def validate_sample_metadata(metadata, design, *, weight_semantics=DESCRIPTIVE_WEIGHT_SEMANTICS):
    positions = _validate_sample_metadata(metadata, design)
    domain = metadata.get("domain_column")
    if (
        "domain_column" not in metadata
        or domain is not None
        and (not isinstance(domain, str) or not domain.strip() or len(domain) > 200)
        or domain is None
        and metadata.get("out_of_domain_positions")
    ):
        raise ValueError("Saved domain declaration differs from complete sample partition.")
    if (
        any(
            type(metadata.get(key)) is not int
            for key in ("n_design", "n_domain", "n_used", "n_lower_strata", "n_terminal_strata")
        )
        or metadata.get("ssu_frame_sha256") != design.validation.frame_input_sha256
        or metadata.get("n_lower_strata") != design.validation.n_cells
        or metadata.get("tsu_frame_sha256") != design.validation.tsu_frame_input_sha256
        or metadata.get("n_terminal_strata") != design.validation.n_tsu_cells
        or metadata.get("lower_stage_stratification")
        != "stage2 within sampled PSU and stage3 within sampled SSU"
        or metadata.get("weight_semantics") != weight_semantics
    ):
        raise ValueError("Saved lower-stratum frame/geometry metadata is inconsistent.")
    return positions


@cpu_call
def inference_frame(state, *, estimates=None, covariance=None, labels=None, null=None):
    frame = _reference_frame(
        state, estimates=estimates, covariance=covariance, labels=labels, null=null
    )
    frame.attrs.update(
        inference="fully stratified three-stage SRSWOR linearization; reference first-stage Student t",
        uncertainty="all three FPC terms; stage-two and stage-three centering within declared strata; df0 positive variance has no tests/CI",
    )
    return frame


@cpu_call
def prepare_fully_stratified_three_stage(
    data,
    design,
    kind,
    outcomes,
    *,
    denominators=None,
    categories=None,
    domain=None,
    missing="raise",
):
    if not isinstance(design, SurveyFullyStratifiedThreeStageDesign):
        raise AnalysisError(
            "invalid_survey_design", "Supply a validated SurveyFullyStratifiedThreeStageDesign."
        )
    validation = design.revalidate(data)
    if (
        not isinstance(kind, str)
        or kind not in {"mean", "total", "ratio", "proportion"}
        or not isinstance(missing, str)
        or missing not in {"raise", "drop"}
    ):
        raise AnalysisError(
            "invalid_survey_target", "Declare mean/total/ratio/proportion and missing raise/drop."
        )
    outcomes = columns(outcomes, "outcomes")
    if kind != "ratio" and denominators is not None:
        raise AnalysisError(
            "invalid_survey_target", "Denominators are only used for ratio targets."
        )
    denominators = columns(denominators, "denominators") if kind == "ratio" else []
    if kind == "ratio" and len(outcomes) != len(denominators):
        raise AnalysisError("invalid_survey_target", "Declare one denominator per ratio numerator.")
    if kind != "ratio" and denominators:
        raise AnalysisError(
            "invalid_survey_target", "Denominators are only used for ratio targets."
        )
    if (
        kind == "proportion"
        and len(outcomes) != 1
        or kind != "proportion"
        and categories is not None
    ):
        raise AnalysisError("invalid_survey_target", "Categories require one proportion outcome.")
    if domain is not None and (
        not isinstance(domain, str) or not domain.strip() or len(domain) > 200
    ):
        raise AnalysisError(
            "invalid_survey_domain", "domain must name a complete Boolean/0-1 indicator column."
        )
    if isinstance(data, Mapping) and any(
        type(v).__module__.startswith("torch") for v in data.values()
    ):
        raise AnalysisError(
            "unsupported_survey_input",
            "Device/tensor columns require explicit resident table conversion.",
        )
    if not isinstance(data, (pd.DataFrame, Mapping, Sequence)) or isinstance(data, (str, bytes)):
        raise AnalysisError("unsupported_survey_input", "Use complete resident survey columns.")
    roles = list(dict.fromkeys(outcomes + denominators + ([domain] if domain else [])))
    projection = list(
        dict.fromkeys(roles + [design.psu] + ([design.strata] if design.strata else []))
    )
    k = MAX_TARGETS if kind == "proportion" else len(outcomes)
    admission(validation.nobs, design.max_memory_mb, k=k)
    # Project only admitted columns before constructing a resident table.
    if isinstance(data, pd.DataFrame):
        frame = data
    else:
        if isinstance(data, Mapping):
            if any(v not in data for v in projection):
                raise AnalysisError("invalid_survey_columns", "Every target column must exist.")
            projected = {v: data[v] for v in projection}
            if any(
                not hasattr(v, "__len__") or len(v) != validation.nobs for v in projected.values()
            ):
                raise AnalysisError(
                    "invalid_data", "Target columns must match the complete design row count."
                )
        else:
            if any(any(v not in row for v in projection) for row in data):
                raise AnalysisError(
                    "invalid_survey_columns", "Every row must contain target columns."
                )
            projected = [{v: row[v] for v in projection} for row in data]
        try:
            frame = pd.DataFrame(projected)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError(
                "invalid_data", "Target columns require consistent resident scalar rows."
            ) from exc
    if frame.columns.has_duplicates or any(v not in frame for v in roles):
        raise AnalysisError(
            "invalid_survey_columns", "Survey target roles must exist in a unique-column table."
        )
    planned = sum(int(frame[v].memory_usage(index=False, deep=True)) for v in roles)
    if planned > design.max_memory_mb * 1024**2:
        raise AnalysisError(
            "survey_budget", "Target columns exceed the declared resident admission budget."
        )
    members = [True] * len(frame)
    if domain:
        members = []
        for value in frame[domain]:
            value = _scalar(value)
            if (
                not isinstance(value, (bool, Real))
                or not math.isfinite(value)
                or value not in (0, 1)
            ):
                raise AnalysisError(
                    "invalid_survey_domain", "Domain membership must be complete Boolean/0-1."
                )
            members.append(bool(value))
    if not any(members):
        raise AnalysisError("empty_survey_domain", "The declared domain has no observations.")
    selected = members.copy()
    excluded = []
    for i, row in enumerate(frame[outcomes + denominators].itertuples(index=False, name=None)):
        if members[i] and any(not pd.api.types.is_scalar(v) for v in row):
            raise AnalysisError(
                "invalid_survey_target", "Target observations must be scalar values."
            )
        if members[i] and any(pd.isna(v) for v in row):
            if missing == "raise":
                raise AnalysisError(
                    "survey_missing", "Missing in-domain target values require missing='drop'."
                )
            selected[i] = False
            excluded.append(i)
    if not any(selected):
        raise AnalysisError(
            "empty_survey_domain", "No complete in-domain target observations remain."
        )
    cat = None
    if kind == "proportion":
        series = frame[outcomes[0]]
        if categories is None and isinstance(series.dtype, pd.CategoricalDtype):
            categories = series.cat.categories.tolist()
        if not isinstance(categories, (list, tuple)) or not 1 <= len(categories) <= MAX_TARGETS:
            raise AnalysisError(
                "survey_categories",
                "Declare 1..32 category levels, or use a categorical dictionary.",
            )
        cat = [_identity(v) for v in categories]
        if len(set(cat)) != len(cat):
            raise AnalysisError("survey_categories", "Typed category levels must be unique.")
        labels = [f"{outcomes[0]}[{i}]={v[0]}:{v[1]}" for i, v in enumerate(cat)]
        values = torch.zeros((len(frame), len(cat)), dtype=FLOAT)
        for i, value in enumerate(series):
            if selected[i]:
                ident = _identity(value)
                if ident not in cat:
                    raise AnalysisError(
                        "survey_categories",
                        "An in-domain value is outside the declared category dictionary.",
                    )
                values[i, cat.index(ident)] = 1.0
        denominator = torch.ones_like(values)
    else:
        labels = (
            [f"{y}/{x}" for y, x in zip(outcomes, denominators)] if kind == "ratio" else outcomes
        )

        def numeric(names):
            data_values = [
                [number(v, name) for name, v in zip(names, row)]
                if selected[i]
                else [0.0] * len(names)
                for i, row in enumerate(frame[names].itertuples(index=False, name=None))
            ]
            return torch.tensor(data_values, dtype=FLOAT)

        values = numeric(outcomes)
        denominator = numeric(denominators) if kind == "ratio" else torch.ones_like(values)
    weights = torch.tensor(validation.weights, dtype=FLOAT)
    ids, strata_ids, strata_groups, groups = {}, {}, [], []
    group_columns = [design.psu] + ([design.strata] if design.strata else [])
    for row in frame[group_columns].itertuples(index=False, name=None):
        h = _identity(row[1]) if design.strata else ("unstratified",)
        p = _identity(row[0])
        if h not in strata_ids:
            strata_ids[h] = len(strata_groups)
            strata_groups.append([])
        if (h, p) not in ids:
            ids[h, p] = len(ids)
            strata_groups[strata_ids[h]].append(ids[h, p])
        groups.append(ids[h, p])
    fpc = [1.0 - s.n_psu / s.population_psu if s.population_psu else 1.0 for s in validation.strata]
    metadata = {
        "n_design": len(frame),
        "n_domain": sum(members),
        "n_used": sum(selected),
        "sample_positions": [i for i, use in enumerate(selected) if use],
        "outcome_exclusions": excluded,
        "out_of_domain_positions": [i for i, use in enumerate(members) if not use],
        "domain_column": domain,
        "missing": missing,
        "outcomes": outcomes,
        "denominators": denominators,
        "categories": [list(v) for v in cat] if cat is not None else None,
        "sum_design_weights": validation.sum_weights,
        "sum_target_weights": float(weights[torch.tensor(selected)].sum()),
        "sample_input_sha256": digest(
            [design.validation.design_input_sha256, selected, values.tolist(), denominator.tolist()]
        ),
        "design_df_convention": DF_CONVENTION,
        "multistage_support": True,
        "weight_semantics": DESCRIPTIVE_WEIGHT_SEMANTICS,
        "sampling": "sequential-srswor",
        "ssu_frame_sha256": validation.frame_input_sha256,
        "n_lower_strata": validation.n_cells,
        "tsu_frame_sha256": validation.tsu_frame_input_sha256,
        "n_terminal_strata": validation.n_tsu_cells,
        "lower_stage_stratification": "stage2 within sampled PSU and stage3 within sampled SSU",
        "precision": "float64",
        "device": "cpu",
        "dataset_support": False,
        "stata_parity_validated": False,
        "workspace": admission(len(frame), design.max_memory_mb, k=values.shape[1]).record(),
    }
    return Target(
        design,
        kind,
        labels,
        weights,
        torch.tensor(selected, dtype=FLOAT),
        values,
        denominator,
        torch.tensor(groups),
        strata_groups,
        fpc,
        metadata,
    )
