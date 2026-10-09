"""Complete two-stage admission, recursive covariance and reference-df inference."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
from numbers import Real
import pandas as pd
import torch
from openecon.analysis_contracts import AnalysisError
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.survey import _identity, _scalar
from openecon.survey_two_stage import SurveyTwoStageDesign, admission
from .regression_common import cpu_call
from .common import FLOAT, MAX_TARGETS, Target, columns, digest, finite, number

DF_CONVENTION = "complete first-stage PSUs minus strata; reference convention, not exact or guaranteed conservative coverage"


@cpu_call
def stage_covariances(weighted_rows, design):
    """Recursive SRSWOR covariance of full-row weighted linearized variables.

    The within-PSU term is multiplied by first-stage f1, since the observed
    between-PSU variability already contains second-stage sampling noise.
    """
    n, k = weighted_rows.shape
    if n != design.validation.nobs or not 1 <= k <= MAX_TARGETS:
        raise AnalysisError(
            "invalid_survey_target", "Weighted score rows differ from complete nested geometry."
        )
    admission(n, design.max_memory_mb, k=k)
    finite(weighted_rows, "Two-stage weighted linearized rows")
    psus = design.validation.psus
    totals = torch.stack([weighted_rows[list(p.row_positions)].sum(0) for p in psus])
    between = torch.zeros((k, k), dtype=FLOAT)
    within = torch.zeros_like(between)
    for s in design.validation.strata:
        f1 = s.n_psu / s.population_psu
        if f1 < 1:
            block = totals[list(s.psu_indices)]
            centered = block - block.mean(0)
            between += (1 - f1) * s.n_psu / (s.n_psu - 1) * (centered.T @ centered)
        for i in s.psu_indices:
            p = psus[i]
            f2 = p.n_ssu / p.population_ssu
            if f2 < 1:
                block = weighted_rows[list(p.row_positions)]
                centered = block - block.mean(0)
                within += f1 * (1 - f2) * p.n_ssu / (p.n_ssu - 1) * (centered.T @ centered)
    return finite((between + between.T) / 2, "Stage-one covariance"), finite(
        (within + within.T) / 2, "Stage-two covariance"
    )


@cpu_call
def two_stage_covariance(weighted_rows, design):
    a, b = stage_covariances(weighted_rows, design)
    return finite(a + b, "Two-stage covariance")


@cpu_call
def validate_covariance(covariance, k):
    if (
        not isinstance(covariance, (list, tuple))
        or len(covariance) != k
        or any(not isinstance(row, (list, tuple)) or len(row) != k for row in covariance)
        or any(
            type(v) not in (int, float) or not math.isfinite(v) for row in covariance for v in row
        )
    ):
        raise ValueError("Two-stage covariance dimensions differ from labels.")
    cov = torch.tensor(covariance, dtype=FLOAT)
    scale = float(cov.abs().max()) or 1.0
    if (
        not bool(torch.isfinite(cov).all())
        or not torch.allclose(cov, cov.T, atol=0.0, rtol=1e-12)
        or float(torch.linalg.eigvalsh(cov / scale).min()) < -1e-10
    ):
        raise ValueError("Covariance must be finite symmetric positive semidefinite.")
    return cov


def validate_identity(value):
    """Require the exact canonical typed scalar encoding saved by admission."""
    if (
        not isinstance(value, list)
        or len(value) != 2
        or value[0] not in ("bool", "int", "float", "str")
    ):
        raise ValueError("Saved category identity is invalid.")
    try:
        raw = (
            float.fromhex(value[1])
            if value[0] == "float" and isinstance(value[1], str)
            else value[1]
        )
        valid = tuple(value) == _identity(raw)
    except (AnalysisError, TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("Saved category identity is not canonical.")


def validate_sample_metadata(metadata, design):
    n = design.validation.nobs
    sets = []
    for key in ("sample_positions", "outcome_exclusions", "out_of_domain_positions"):
        positions = metadata.get(key)
        if (
            not isinstance(positions, list)
            or len(positions) > n
            or any(type(i) is not int or not 0 <= i < n for i in positions)
            or positions != sorted(set(positions))
        ):
            raise ValueError("Saved physical sample positions are inconsistent.")
        sets.append(set(positions))
    if (
        not sets[0]
        or any(sets[i] & sets[j] for i in range(3) for j in range(i))
        or set.union(*sets) != set(range(n))
        or metadata.get("n_design") != n
        or metadata.get("n_used") != len(sets[0])
        or metadata.get("n_domain") != n - len(sets[2])
        or metadata.get("missing") not in ("raise", "drop")
        or metadata.get("missing") == "raise"
        and sets[1]
        or metadata.get("design_df_convention") != DF_CONVENTION
        or metadata.get("precision") != "float64"
        or metadata.get("device") != "cpu"
        or metadata.get("multistage_support") is not True
        or metadata.get("dataset_support") is not False
        or metadata.get("stata_parity_validated") is not False
        or metadata.get("sampling") != "sequential-srswor"
    ):
        raise ValueError("Saved sample partition or method metadata is inconsistent.")
    weights = design.validation.weights
    for key, value in (
        ("sum_design_weights", math.fsum(weights)),
        ("sum_target_weights", math.fsum(weights[i] for i in sets[0])),
    ):
        actual = metadata.get(key)
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isclose(actual, value, rel_tol=1e-12)
        ):
            raise ValueError("Saved derived sampling-weight sums are inconsistent.")
    return sorted(sets[0])


@cpu_call
def inference_frame(state, *, estimates=None, covariance=None, labels=None, null=None):
    estimates = state.estimates if estimates is None else estimates
    covariance = state.covariance if covariance is None else covariance
    labels = state.labels if labels is None else labels
    null = state.null if null is None else null
    critical = critical_value(state.alpha, state.df) if state.df else None
    rows = []
    for i, value in enumerate(estimates):
        se = math.sqrt(max(covariance[i][i], 0.0))
        statistic = (value - null[i]) / se if se > 0 and state.df else None
        p = student_t_two_sided(statistic, state.df) if statistic is not None else None
        width = critical * se if critical is not None else (0.0 if se == 0 else None)
        rows.append(
            [
                value,
                se,
                statistic,
                p,
                None if width is None else value - width,
                None if width is None else value + width,
            ]
        )
    frame = pd.DataFrame(
        rows,
        index=list(labels),
        columns=["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"],
    )
    frame.attrs.update(
        survey_state=state.model_dump(mode="json"),
        covariance_matrix=[list(r) for r in covariance],
        inference="two-stage SRSWOR linearization; reference first-stage Student t",
        df=state.df,
        design_df_convention=DF_CONVENTION,
        method=getattr(state, "method", "two-stage-taylor"),
        target=getattr(state, "target", getattr(state, "family", None)),
        alpha=state.alpha,
        uncertainty="both stage FPC terms; df0 positive variance has no tests/CI",
    )
    return frame


@cpu_call
def prepare_two_stage(
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
    if not isinstance(design, SurveyTwoStageDesign):
        raise AnalysisError("invalid_survey_design", "Supply a validated SurveyTwoStageDesign.")
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
    if domain is not None and (not isinstance(domain, str) or not domain.strip()):
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
        "weight_semantics": "derived inverse probabilities N/n * M/m; sequential SRSWOR",
        "sampling": "sequential-srswor",
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
