"""Joint four-stage SRSWOR descriptive estimates and replayable inference."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import math
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.survey_four_stage import (
    SurveyFourStageDesign,
    admission,
)
from .regression_common import cpu_call
from .common import FLOAT, MAX_TARGETS, Target, columns, digest, finite, number
from .four_stage_common import (
    inference_frame,
    prepare_four_stage,
    stage_covariances,
    validate_covariance,
    validate_identity,
    validate_sample_metadata,
)

VARIANCE_FORMULA = "sum_h (1-f1) n/(n-1) centered PSU totals outer + sum_hi f1 (1-f2) m/(m-1) centered SSU totals outer + sum_hij f1*f2 (1-f3) l/(l-1) centered TSU totals outer + sum_hijt f1*f2*f3 (1-f4) k/(k-1) centered weighted FSU rows outer"
CONFIDENCE_INTERVAL = "unclipped reference first-stage design-t; df0 positive variance unavailable; zero variance point interval"


def _saved_arithmetic(method):
    @wraps(method)
    def checked(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except OverflowError as exc:
            raise ValueError(
                "Saved four-stage primitives exceed finite numeric bounds."
            ) from exc

    return checked


class SurveyFourStageResult(BaseModel):
    """Saved primitives replay all four stages; a checksum is not authentication."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")
    schema_version: Literal["survey-four-stage-result-v1"] = (
        "survey-four-stage-result-v1"
    )
    method: Literal["four-stage-taylor"] = "four-stage-taylor"
    target: Literal["mean", "total", "ratio", "proportion"]
    labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_TARGETS)
    estimates: tuple[StrictFloat, ...]
    covariance: tuple[tuple[StrictFloat, ...], ...]
    df: StrictInt = Field(ge=0)
    alpha: float = Field(ge=1e-8, le=1 - 1e-8, allow_inf_nan=False, strict=True)
    null: tuple[StrictFloat, ...]
    design: SurveyFourStageDesign
    metadata: dict[str, Any]
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def bounded_state(cls, raw):
        if isinstance(raw, cls):
            raw = vars(raw)
        if isinstance(raw, Mapping):
            design = raw.get("design")
            if isinstance(design, SurveyFourStageDesign):
                design = vars(design)
            validation = design.get("validation") if isinstance(design, Mapping) else None
            if isinstance(validation, BaseModel):
                validation = vars(validation)
            if not isinstance(validation, Mapping):
                raise ValueError(
                    "Saved target requires bounded four-stage geometry."
                )
            n = validation.get("nobs")
            memory = design.get("max_memory_mb", 64)
            labels = raw.get("labels")
            if (
                type(n) is not int
                or not 1 <= n <= 1_000_000
                or type(memory) is not int
                or not 1 <= memory <= 512
                or not isinstance(labels, (list, tuple))
                or not 1 <= len(labels) <= MAX_TARGETS
                or not isinstance(raw.get("metadata"), Mapping)
            ):
                raise ValueError("Saved target dimensions exceed bounded admission.")
            admission(n, memory, k=len(labels))
            k = len(labels)
            for key in ("estimates", "null"):
                values = raw.get(key)
                if not isinstance(values, (list, tuple)) or len(values) != k:
                    raise ValueError(
                        "Saved target numeric vectors must match bounded labels before parsing."
                    )
            covariance = raw.get("covariance")
            if (
                not isinstance(covariance, (list, tuple))
                or len(covariance) != k
                or any(not isinstance(row, (list, tuple)) or len(row) != k for row in covariance)
            ):
                raise ValueError("Saved covariance must be bounded k by k before numeric parsing.")
        return raw

    @model_validator(mode="after")
    @cpu_call
    @_saved_arithmetic
    def replay(self):
        n = self.design.validation.nobs
        k = len(self.labels)
        admission(n, self.design.max_memory_mb, k=k)
        if (
            len(set(self.labels)) != k
            or len(self.estimates) != k
            or len(self.null) != k
            or any(not math.isfinite(v) for v in (*self.estimates, *self.null))
            or self.df != self.design.validation.design_df
        ):
            raise ValueError("Saved target dimensions or complete-design df are inconsistent.")
        positions = validate_sample_metadata(self.metadata, self.design)
        if (
            self.metadata.get("variance_formula") != VARIANCE_FORMULA
            or self.metadata.get("confidence_interval") != CONFIDENCE_INTERVAL
            or self.metadata.get("calibration_support") is not False
            or "design_effect" not in self.metadata
            or self.metadata["design_effect"] is not None
        ):
            raise ValueError("Saved target scientific classification is inconsistent.")
        cov = validate_covariance(self.covariance, k)
        tensors = []
        for key in ("primitive_values", "primitive_denominators", "row_influences"):
            rows = self.metadata.get(key)
            if (
                not isinstance(rows, list)
                or len(rows) != n
                or any(not isinstance(row, list) or len(row) != k for row in rows)
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for row in rows for v in row
                )
            ):
                raise ValueError("Saved target primitive dimensions/values are invalid.")
            tensors.append(torch.tensor(rows, dtype=FLOAT))
        values, denominators, saved_rows = tensors
        selected = torch.zeros(n, dtype=FLOAT)
        selected[positions] = 1.0
        if bool((values[~selected.bool()] != 0).any()):
            raise ValueError("Excluded rows must retain zero target values.")
        if self.target != "ratio" and not bool((denominators == 1).all()):
            raise ValueError("Nonratio targets require constant-one denominators.")
        if self.target == "ratio" and bool((denominators[~selected.bool()] != 0).any()):
            raise ValueError("Excluded ratio denominators must be zero.")
        outcomes = self.metadata.get("outcomes")
        names = self.metadata.get("denominators")
        if not isinstance(outcomes, list) or not outcomes or not isinstance(names, list):
            raise ValueError("Saved target roles are invalid.")
        try:
            columns(outcomes, "saved outcomes")
            if self.target == "ratio":
                columns(names, "saved denominators")
            elif names:
                raise ValueError("Only saved ratio targets may declare denominator names.")
        except AnalysisError as exc:
            raise ValueError("Saved target names differ from live column admission.") from exc
        if "categories" not in self.metadata or (
            self.target != "proportion" and self.metadata["categories"] is not None
        ):
            raise ValueError("Only proportion targets may declare saved category roles.")
        if self.target == "proportion":
            categories = self.metadata.get("categories")
            if (
                len(outcomes) != 1
                or not isinstance(categories, list)
                or len(categories) != k
                or any(not isinstance(c, list) or len(c) != 2 for c in categories)
                or not bool(((values == 0) | (values == 1)).all())
                or not bool((values.sum(1) == selected).all())
            ):
                raise ValueError("Saved category dictionary/one-hot rows are invalid.")
            for category in categories:
                validate_identity(category)
            if len({tuple(category) for category in categories}) != k:
                raise ValueError("Saved category dictionary requires distinct typed levels.")
            labels = [f"{outcomes[0]}[{i}]={v[0]}:{v[1]}" for i, v in enumerate(categories)]
        elif self.target == "ratio":
            if len(outcomes) != k or len(names) != k:
                raise ValueError("Ratio roles differ from joint target dimensions.")
            labels = [f"{y}/{x}" for y, x in zip(outcomes, names)]
        else:
            if len(outcomes) != k or names:
                raise ValueError("Target roles differ from joint target dimensions.")
            labels = outcomes
        if list(self.labels) != labels:
            raise ValueError("Saved target order differs from declared roles.")
        expected_sha = digest(
            [
                self.design.validation.design_input_sha256,
                selected.bool().tolist(),
                values.tolist(),
                denominators.tolist(),
            ]
        )
        if self.metadata.get("sample_input_sha256") != expected_sha:
            raise ValueError("Saved sample primitives differ from their ordered fingerprint.")
        weights = torch.tensor(self.design.validation.weights, dtype=FLOAT)
        target = Target(
            self.design,
            self.target,
            labels,
            weights,
            selected,
            values,
            denominators,
            None,
            None,
            None,
            self.metadata,
        )
        estimate, influence = target.evaluate(weights)
        weighted = weights[:, None] * influence
        if not torch.allclose(
            estimate, torch.tensor(self.estimates, dtype=FLOAT), rtol=1e-11, atol=0.0
        ) or not torch.allclose(weighted, saved_rows, rtol=1e-11, atol=0.0):
            raise ValueError("Saved estimates/linearized rows differ from primitive targets.")
        a, b, c, d = stage_covariances(weighted, self.design)
        if not torch.allclose(cov, a + b + c + d, rtol=1e-10, atol=0.0):
            raise ValueError("Saved covariance differs from all four recursive sampling stages.")
        for key, expected in (
            ("stage1_covariance", a),
            ("stage2_covariance", b),
            ("stage3_covariance", c),
            ("stage4_covariance", d),
        ):
            stored = validate_covariance(self.metadata.get(key, []), k)
            if not torch.allclose(stored, expected, rtol=1e-10, atol=0.0):
                raise ValueError("Saved stage-specific covariance differs from primitive replay.")
        if (
            digest(self.model_dump(mode="json", exclude={"integrity_sha256"}))
            != self.integrity_sha256
        ):
            raise ValueError("Saved four-stage integrity changed.")
        return self

    def to_frame(self):
        return inference_frame(type(self).model_validate_json(self.model_dump_json()))

    @cpu_call
    def contrast(self, coefficients, *, null=0.0):
        state = type(self).model_validate_json(self.model_dump_json())
        if not isinstance(coefficients, (list, tuple)) or len(coefficients) != len(state.labels):
            raise AnalysisError(
                "invalid_survey_contrast", "Declare one coefficient per saved target."
            )
        a = torch.tensor([number(v, "contrast") for v in coefficients], dtype=FLOAT)
        estimate = finite(a @ torch.tensor(state.estimates, dtype=FLOAT), "Contrast")
        variance = finite(a @ torch.tensor(state.covariance, dtype=FLOAT) @ a, "Contrast variance")
        return inference_frame(
            state,
            estimates=[float(estimate)],
            covariance=[[max(float(variance), 0.0)]],
            labels=["linear contrast"],
            null=[number(null, "null")],
        )


@cpu_call
def _estimate(target, *, alpha, null):
    alpha = number(alpha, "alpha", 1e-8, 1 - 1e-8)
    nulls = (
        [number(v, "null") for v in null]
        if isinstance(null, (list, tuple))
        else [number(null, "null")] * len(target.labels)
    )
    if len(nulls) != len(target.labels):
        raise AnalysisError("invalid_survey_option", "One null value per joint target is required.")
    estimate, influence = target.evaluate(target.weights)
    weighted = finite(
        target.weights[:, None] * influence, "Weighted four-stage influence"
    )
    a, b, c, d = stage_covariances(weighted, target.design)
    metadata = {
        **target.metadata,
        "primitive_values": target.values.tolist(),
        "primitive_denominators": target.denominator.tolist(),
        "row_influences": weighted.tolist(),
        "stage1_covariance": a.tolist(),
        "stage2_covariance": b.tolist(),
        "stage3_covariance": c.tolist(),
        "stage4_covariance": d.tolist(),
        "variance_formula": VARIANCE_FORMULA,
        "confidence_interval": CONFIDENCE_INTERVAL,
        "calibration_support": False,
        "design_effect": None,
    }
    payload = dict(
        schema_version="survey-four-stage-result-v1",
        method="four-stage-taylor",
        target=target.kind,
        labels=target.labels,
        estimates=estimate.tolist(),
        covariance=(a + b + c + d).tolist(),
        df=target.design.validation.design_df,
        alpha=alpha,
        null=nulls,
        design=target.design.model_dump(mode="json"),
        metadata=metadata,
    )
    return SurveyFourStageResult(**payload, integrity_sha256=digest(payload))


@cpu_call
def survey_four_stage_mean(
    data, design, outcomes, *, domain=None, missing="raise", alpha=0.05, null=0.0
) -> SurveyFourStageResult:
    """Joint Hájek means with four SRSWOR contributions and nested PSU/SSU/TSU/FSU selection."""
    return _estimate(
        prepare_four_stage(
            data, design, "mean", outcomes, domain=domain, missing=missing
        ),
        alpha=alpha,
        null=null,
    )


@cpu_call
def survey_four_stage_total(
    data, design, outcomes, *, domain=None, missing="raise", alpha=0.05, null=0.0
) -> SurveyFourStageResult:
    """Joint Horvitz–Thompson totals with full four-stage SRSWOR covariance."""
    return _estimate(
        prepare_four_stage(
            data, design, "total", outcomes, domain=domain, missing=missing
        ),
        alpha=alpha,
        null=null,
    )


@cpu_call
def survey_four_stage_ratio(
    data, design, numerators, denominators, *, domain=None, missing="raise", alpha=0.05, null=0.0
) -> SurveyFourStageResult:
    """Paired total ratios with complete-domain four-stage delta covariance."""
    return _estimate(
        prepare_four_stage(
            data,
            design,
            "ratio",
            numerators,
            denominators=denominators,
            domain=domain,
            missing=missing,
        ),
        alpha=alpha,
        null=null,
    )


@cpu_call
def survey_four_stage_proportion(
    data, design, outcome, *, categories=None, domain=None, missing="raise", alpha=0.05, null=0.0
) -> SurveyFourStageResult:
    """Declared category shares, absent levels and full joint four-stage covariance."""
    return _estimate(
        prepare_four_stage(
            data,
            design,
            "proportion",
            outcome,
            categories=categories,
            domain=domain,
            missing=missing,
        ),
        alpha=alpha,
        null=null,
    )
