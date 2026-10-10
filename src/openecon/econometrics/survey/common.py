"""Shared immutable result and complete-design target admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.survey import SurveyDesign, _identity, _scalar

FLOAT = torch.float64
MAX_TARGETS = 32
MAX_DEFF_WORK = 50_000_000
SRSWR_WEIGHTED_REFERENCE = (
    "unequal-weight weighted-population SRSWR plug-in, full original row geometry, "
    "fixed eligibility, no FPC"
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def number(value, name, low=None, high=None):
    value = _scalar(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError("invalid_survey_option", f"{name} needs a finite real number.")
    try:
        valid = math.isfinite(value)
    except (ValueError, OverflowError):
        valid = False
    if not valid or low is not None and value < low or high is not None and value > high:
        raise AnalysisError(
            "invalid_survey_option", f"{name} is outside its finite supported range."
        )
    return float(value)


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError(
            "survey_numerical_failure", f"{name} overflowed; rescale the input explicitly."
        )
    return value


def allocation(n, k, groups, replicates=0, supplied=False):
    cells = n * k + groups * k + k * k + replicates * k
    if supplied:
        cells += n * replicates
    return plan_workspace(
        "single-stage survey inference",
        {
            "tensor/result/temporary serialization buffers": cells * 96,
            "physical-row identities and sample indices": n * 256,
        },
        budget_bytes=workspace_budget_bytes(),
    )


class SurveyResult(BaseModel):
    """Complete, integrity-checked statistical envelope; its digest is not authentication."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["survey-result-v1"] = "survey-result-v1"
    method: Literal["taylor", "brr", "fay", "jackknife", "bootstrap"]
    target: Literal["mean", "total", "ratio", "proportion"]
    labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_TARGETS)
    estimates: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    df: StrictInt = Field(ge=0)
    alpha: float = Field(gt=0, lt=1)
    null: tuple[float, ...]
    design: SurveyDesign
    metadata: dict[str, Any]
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_state(self):
        size = len(self.labels)
        if (
            len(set(self.labels)) != size
            or len(self.estimates) != size
            or len(self.null) != size
            or len(self.covariance) != size
            or any(len(r) != size for r in self.covariance)
            or any(not math.isfinite(v) for v in (*self.estimates, *self.null))
        ):
            raise ValueError("Survey result dimensions/labels/values are inconsistent.")
        cov = torch.tensor(self.covariance, dtype=FLOAT)
        scale = max(float(cov.abs().max()), 1e-300)
        if (
            not torch.isfinite(cov).all()
            or not torch.allclose(cov, cov.T, atol=scale * 1e-12, rtol=1e-12)
            or float(torch.linalg.eigvalsh(cov / scale).min()) < -1e-10
        ):
            raise ValueError("Survey covariance must be finite symmetric positive semidefinite.")
        if self.metadata.get("n_design") != self.design.validation.nobs:
            raise ValueError("Survey result design/sample geometry is inconsistent.")
        positions = self.metadata.get("sample_positions")
        if (
            not isinstance(positions, list)
            or not positions
            or len(positions) > self.design.validation.nobs
            or any(
                type(v) is not int or not 0 <= v < self.design.validation.nobs for v in positions
            )
            or positions != sorted(set(positions))
        ):
            raise ValueError("Survey result original positions are inconsistent.")
        if self.metadata.get("n_used") != len(positions):
            raise ValueError("Saved sample count differs from its original positions.")
        k = len(self.labels)
        if self.method != "taylor":
            count = self.metadata.get("replicate_count")
            ids = self.metadata.get("replicate_ids")
            values = self.metadata.get("replicate_estimates")
            multipliers = self.metadata.get("variance_multipliers")
            if (
                type(count) is not int
                or not 2 <= count <= 4096
                or not isinstance(ids, list)
                or len(ids) != count
                or len(set(ids)) != count
                or any(not isinstance(v, str) or not v for v in ids)
                or not isinstance(values, list)
                or len(values) != count
                or any(not isinstance(row, list) or len(row) != k for row in values)
                or not isinstance(multipliers, list)
                or len(multipliers) != count
                or self.metadata.get("failed_replicates") != []
                or self.df > count - 1
            ):
                raise ValueError("Saved replicate geometry is inconsistent.")
            replicas = torch.tensor(values, dtype=FLOAT)
            scales = torch.tensor(multipliers, dtype=FLOAT)
            if (
                not torch.isfinite(replicas).all()
                or not torch.isfinite(scales).all()
                or (scales < 0).any()
            ):
                raise ValueError("Saved replicate estimates/scales must be finite and nonnegative.")
            centering = self.metadata.get("centering")
            if centering == "original":
                difference = replicas - torch.tensor(self.estimates, dtype=FLOAT)
            elif centering == "replicate_mean":
                difference = replicas - replicas.mean(0)
            elif centering == "stratum_mean" and self.method == "jackknife":
                strata = self.metadata.get("replicate_stratum_indices")
                if (
                    not isinstance(strata, list)
                    or len(strata) != count
                    or any(type(v) is not int for v in strata)
                ):
                    raise ValueError("Saved jackknife strata are inconsistent.")
                difference = torch.empty_like(replicas)
                for h in sorted(set(strata)):
                    rows = [i for i, value in enumerate(strata) if value == h]
                    difference[rows] = replicas[rows] - replicas[rows].mean(0)
            else:
                raise ValueError("Unsupported saved replicate centering.")
            replay = (difference.T * scales) @ difference
            if not torch.allclose(cov, replay, atol=scale * 1e-10, rtol=1e-10):
                raise ValueError("Saved covariance differs from complete replicate inference.")
        if self.method == "taylor" and self.df != self.design.validation.design_df:
            raise ValueError("Taylor inference df must retain the complete design geometry.")
        if (
            "srs_reference_state" in self.metadata
            or self.metadata.get("srs_reference") == SRSWR_WEIGHTED_REFERENCE
        ):
            validate_srswr_reference(self, cov)
        if (
            digest(self.model_dump(mode="json", exclude={"integrity_sha256"}))
            != self.integrity_sha256
        ):
            raise ValueError("Survey state integrity changed.")
        return self

    def to_frame(self):
        """Restore inference and an exportable table without refitting or original data."""
        state = SurveyResult.model_validate_json(self.model_dump_json())
        return inference_frame(state)

    def contrast(self, coefficients, *, null=0.0):
        """One declared linear contrast using the saved complete joint covariance."""
        state = SurveyResult.model_validate_json(self.model_dump_json())
        if not isinstance(coefficients, (list, tuple)) or len(coefficients) != len(state.labels):
            raise AnalysisError(
                "invalid_survey_contrast", "Declare one coefficient per saved target."
            )
        a = torch.tensor([number(v, "contrast coefficient") for v in coefficients], dtype=FLOAT)
        estimate = finite(a @ torch.tensor(state.estimates, dtype=FLOAT), "Contrast")
        variance = finite(
            a @ torch.tensor(state.covariance, dtype=FLOAT) @ a, "Contrast covariance"
        )
        return inference_frame(
            state,
            estimates=[float(estimate)],
            covariance=[[max(float(variance), 0.0)]],
            labels=["linear contrast"],
            null=[number(null, "null")],
        )


def validate_srswr_reference(state, covariance):
    """Replay the named weighted-population comparator from bounded saved moments."""
    metadata = state.metadata

    def scalar(value):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except (OverflowError, ValueError):
            return False

    record = metadata.get("srs_reference_state")
    expected = {
        "schema_version", "law", "n_design", "sum_design_weights",
        "weighted_influence_mean", "weighted_centered_crossproducts", "fpc_applied",
        "eligibility",
    }
    if (
        state.method != "taylor"
        or metadata.get("srs_reference") != SRSWR_WEIGHTED_REFERENCE
        or not isinstance(record, dict)
        or set(record) != expected
        or record.get("schema_version") != "survey-srswr-weighted-v1"
        or record.get("law") != "full-design-weighted-population-srswr-linearized"
        or record.get("fpc_applied") is not False
        or record.get("eligibility") != "fixed-domain-and-joint-complete-case-indicator"
    ):
        raise ValueError("Saved SRSWR reference law is unsupported or incomplete.")
    n, k = record.get("n_design"), len(state.labels)
    population = record.get("sum_design_weights")
    if (
        type(n) is not int
        or n < 2
        or n != state.design.validation.nobs
        or n * k * (k + 2) > MAX_DEFF_WORK
        or not scalar(population)
        or population <= 0
        or not math.isclose(
            population, state.design.validation.sum_weights, rel_tol=1e-12, abs_tol=0.0
        )
    ):
        raise ValueError("Saved SRSWR full-design sample/weight geometry is inconsistent.")

    def vector(value, width):
        if (
            not isinstance(value, list)
            or len(value) != width
            or any(not scalar(v) for v in value)
        ):
            raise ValueError("Saved SRSWR moments must be finite and correctly dimensioned.")
        return value

    mean = torch.tensor(vector(record.get("weighted_influence_mean"), k), dtype=FLOAT)

    def matrix(value):
        if not isinstance(value, list) or len(value) != k:
            raise ValueError("Saved SRSWR covariance/moments dimensions are inconsistent.")
        tensor = torch.tensor([vector(row, k) for row in value], dtype=FLOAT)
        scale = max(float(tensor.abs().max()), 1e-300)
        if (
            not torch.allclose(tensor, tensor.T, atol=scale * 1e-12, rtol=1e-12)
            or (tensor.diagonal() < 0).any()
            or float(torch.linalg.eigvalsh(tensor / scale).min()) < -1e-10
        ):
            raise ValueError("Saved SRSWR covariance/moments must be symmetric positive semidefinite.")
        return tensor

    moments = matrix(record.get("weighted_centered_crossproducts"))
    reference = matrix(metadata.get("srs_covariance"))
    replay = population / (n - 1) * moments
    scale = max(float(reference.abs().max()), 1e-300)
    if (
        not torch.isfinite(replay).all()
        or not torch.allclose(reference, replay, atol=scale * 1e-12, rtol=1e-12)
    ):
        raise ValueError("Saved SRSWR covariance differs from its complete moment replay.")
    psus = metadata.get("psu_influence_sums")
    if not isinstance(psus, list) or len(psus) != state.design.validation.n_psu:
        raise ValueError("Saved SRSWR weighted influence geometry is inconsistent.")
    psu_values = torch.tensor([vector(row, k) for row in psus], dtype=FLOAT)
    summed = psu_values.sum(0) / population
    mean_scale = max(
        float(psu_values.abs().sum(0).max()) / population,
        math.sqrt(float(moments.diagonal().max())) / math.sqrt(population),
        1e-300,
    )
    if not math.isfinite(mean_scale):
        raise ValueError("Saved SRSWR moment scale is not finite.")
    if not torch.allclose(mean, summed, atol=mean_scale * 1e-10, rtol=1e-10):
        raise ValueError("Saved SRSWR mean differs from the complete PSU influence sums.")
    target_mean = (
        torch.tensor(state.estimates, dtype=FLOAT) / population
        if state.target == "total" else torch.zeros(k, dtype=FLOAT)
    )
    if not torch.allclose(mean, target_mean, atol=mean_scale * 1e-10, rtol=1e-10):
        raise ValueError("Saved SRSWR mean is inconsistent with the target estimating equation.")
    groups, corrections = metadata.get("stratum_psu_indices"), metadata.get("fpc_multipliers")
    strata = state.design.validation.strata
    if (
        not isinstance(groups, list)
        or len(groups) != len(strata)
        or any(
            not isinstance(rows, list)
            or len(rows) != stratum.n_psu
            or any(type(index) is not int for index in rows)
            for rows, stratum in zip(groups, strata)
        )
        or sorted(index for rows in groups for index in rows) != list(range(len(psus)))
        or not isinstance(corrections, list)
        or len(corrections) != len(strata)
        or any(
            type(value) not in (int, float)
            or value != (1.0 - stratum.n_psu / stratum.population_psu
                         if stratum.population_psu else 1.0)
            for value, stratum in zip(corrections, strata)
        )
    ):
        raise ValueError("Saved SRSWR Taylor strata/FPC geometry is inconsistent.")
    design_replay = torch.zeros_like(covariance)
    for rows, correction in zip(groups, corrections):
        if correction:
            block = psu_values[rows]
            centered = block - block.mean(0)
            design_replay += correction * len(rows) / (len(rows) - 1) * (centered.T @ centered)
    cov_scale = max(float(covariance.abs().max()), 1e-300)
    if not torch.allclose(covariance, design_replay, atol=cov_scale * 1e-12, rtol=1e-12):
        raise ValueError("Saved SRSWR numerator differs from complete Taylor covariance replay.")
    effects = metadata.get("design_effect")
    if not isinstance(effects, list) or len(effects) != k:
        raise ValueError("Saved SRSWR design-effect dimensions are inconsistent.")
    for i, effect in enumerate(effects):
        denominator = float(reference[i, i])
        if denominator <= 0:
            if effect is not None:
                raise ValueError("Zero SRSWR variance has an undefined design effect.")
        else:
            ratio = float(covariance[i, i]) / denominator
            if (
                not scalar(effect)
                or effect < 0
                or not math.isfinite(ratio)
                or not math.isclose(effect, ratio, rel_tol=1e-12, abs_tol=0.0)
            ):
                raise ValueError("Saved SRSWR design effect differs from its covariance ratio.")


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
        width = critical * se if critical is not None else 0.0
        rows.append([value, se, statistic, p, value - width, value + width])
    frame = pd.DataFrame(
        rows,
        index=list(labels),
        columns=["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"],
    )
    frame.attrs.update(
        survey_state=state.model_dump(mode="json"),
        covariance_matrix=[list(r) for r in covariance],
        inference="design Student t; zero-variance tests undefined",
        df=state.df,
        method=state.method,
        target=state.target,
        alpha=state.alpha,
        uncertainty="design-based; no model/regression covariance substitution",
    )
    return frame


@dataclass
class Target:
    design: SurveyDesign
    kind: str
    labels: list[str]
    weights: torch.Tensor
    domain: torch.Tensor
    values: torch.Tensor
    denominator: torch.Tensor
    groups: torch.Tensor
    strata_groups: list[list[int]]
    fpc: list[float]
    metadata: dict

    def evaluate(self, weights):
        effective = weights * self.domain
        numerator = finite((effective[:, None] * self.values).sum(0), "Survey numerator")
        if self.kind == "total":
            return numerator, self.values * self.domain[:, None]
        denominator = finite((effective[:, None] * self.denominator).sum(0), "Survey denominator")
        absolute = (effective[:, None] * self.denominator.abs()).sum(0)
        if bool((denominator.abs() <= absolute * 1e-12).any()) or bool((absolute == 0).any()):
            raise AnalysisError(
                "survey_denominator", "A survey/replicate denominator is zero or unstable."
            )
        theta = finite(numerator / denominator, "Survey estimate")
        influence = finite(
            self.domain[:, None] * (self.values - self.denominator * theta) / denominator,
            "Survey influence",
        )
        return theta, influence


def columns(value, name):
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, (list, tuple))
        or not 1 <= len(value) <= MAX_TARGETS
        or any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in value)
        or len(set(value)) != len(value)
    ):
        raise AnalysisError(
            "invalid_survey_target", f"{name} needs 1..{MAX_TARGETS} distinct column names."
        )
    return list(value)


def prepare(
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
    if not isinstance(design, SurveyDesign):
        raise AnalysisError(
            "invalid_survey_design", "Supply a validated single-stage SurveyDesign."
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
    # Design revalidation already admits complete row/hash geometry before copying target columns.
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    roles = list(dict.fromkeys(outcomes + denominators + ([domain] if domain else [])))
    if frame.columns.has_duplicates or any(v not in frame for v in roles):
        raise AnalysisError(
            "invalid_survey_columns", "Survey target roles must exist in a unique-column table."
        )
    k = MAX_TARGETS if kind == "proportion" else len(outcomes)
    allocation(len(frame), k, validation.n_psu)
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
    weights = torch.tensor(frame[design.weights].tolist(), dtype=FLOAT)
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
        "design_df_convention": "all declared PSUs minus all declared strata; zero-domain PSUs retained",
        "precision": "float64",
        "device": "cpu",
        "dataset_support": False,
        "stata_parity_validated": False,
        "workspace": allocation(len(frame), values.shape[1], validation.n_psu).record(),
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


def result(target, method, estimate, covariance, *, alpha=0.05, null=0.0, df=None, metadata=None):
    alpha = number(alpha, "alpha", 1e-8, 1.0 - 1e-8)
    if isinstance(null, (list, tuple)):
        null = [number(v, "null") for v in null]
        if len(null) != len(target.labels):
            raise AnalysisError("invalid_survey_option", "One null value per target is required.")
    else:
        null = [number(null, "null")] * len(target.labels)
    finite(estimate, "Survey estimate")
    finite(covariance, "Survey covariance")
    payload = {
        "schema_version": "survey-result-v1",
        "method": method,
        "target": target.kind,
        "labels": target.labels,
        "estimates": estimate.tolist(),
        "covariance": ((covariance + covariance.T) / 2).tolist(),
        "df": target.design.validation.design_df if df is None else df,
        "alpha": alpha,
        "null": null,
        "design": target.design.model_dump(mode="json"),
        "metadata": {**target.metadata, **(metadata or {})},
    }
    return SurveyResult(**payload, integrity_sha256=digest(payload))
