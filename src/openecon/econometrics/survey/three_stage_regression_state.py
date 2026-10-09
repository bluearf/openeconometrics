"""Replayable coefficient inference for explicit three-stage SRSWOR sampling.

Saved rows bind the numeric fit, normalized pseudo-score, observed sensitivity
and all three stage variance terms. They do not claim authentication of source data.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import math
from types import SimpleNamespace
from typing import Any, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import f_sf
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.survey_three_stage import SurveyThreeStageDesign
from .common import FLOAT, digest, finite, number
from .regression import _binary_check, _components, _poisson_check, _solve
from .regression_common import MAX_WORK, cpu_call
from .two_stage_regression_state import _close, _fit_evidence, _matrix
from .three_stage_common import (
    inference_frame,
    stage_covariances,
    validate_covariance,
    validate_sample_metadata,
)


def replay_work(n, used, k, groups, ssus, family="linear"):
    """Conservative three-level covariance, rank and separation replay work."""
    constraints = max(8, 4 * k) + 2 * k
    diagnostic = 0 if family == "linear" else 64 * used * k * 3 + 64 * 100 * constraints * k * k
    return (
        used * k * k * 12 + (n + groups + ssus) * k * k * 12 + k**3 * 10 + n * k * 12 + diagnostic
    )


def regression_workspace(n, used, k, groups, ssus, memory_mb):
    """Admit all three physical levels, full scores and serialized replay."""
    return plan_workspace(
        "three-stage survey regression and saved-state replay",
        {
            "three-level physical indexing": n * 3072,
            "fit, score rows, influences and serialization": n * k * 640,
            "selected primitive and scaled designs": used * (k + 1) * 224,
            "PSU/SSU totals and three covariance components": (groups + ssus) * k * 160
            + k * k * 2560,
        },
        budget_bytes=min(workspace_budget_bytes(), memory_mb * 1024**2),
    )


@cpu_call
def replay_fit(design, X, y, positions, beta, family, *, certify=True):
    """Evaluate at saved raw coefficients, without optimizing or readmitting rows."""
    n, k = design.validation.nobs, X.shape[1]
    if len(X) <= k:
        raise ValueError("More admitted rows than coefficients are required.")
    complete_weights = design.validation.weights
    weights = torch.tensor([complete_weights[i] for i in positions], dtype=FLOAT)
    relative = weights / weights.max()
    normalized = finite(relative * (len(X) / relative.sum()), "Normalized three-stage weights")
    if bool((normalized <= 0).any()):
        raise ValueError("Saved positive sampling weights underflow after normalization.")
    scales = X.abs().amax(0)
    if bool((scales == 0).any()):
        raise ValueError("Saved design has an unidentified zero column.")
    scaled = finite(X / scales, "Scaled saved three-stage design")
    singular = torch.linalg.svdvals(scaled * normalized.sqrt()[:, None])
    threshold = 10 * torch.finfo(FLOAT).eps * max(scaled.shape) * float(singular[0])
    if len(singular) != k or float(singular[-1]) <= threshold:
        raise ValueError("Saved weighted design must have full column rank.")
    coefficient = finite(beta * scales, "Scaled saved coefficients")
    linear_expected = None
    if family in ("logit", "probit") and bool(((y != 0) & (y != 1)).any()):
        raise ValueError("Saved binary responses must be exact zero or one.")
    if family == "poisson" and bool(((y < 0) | (y != y.floor()) | (y > 2**53)).any()):
        raise ValueError(
            "Saved Poisson responses must be exactly representable nonnegative counts."
        )
    if certify and family in ("logit", "probit"):
        _binary_check(scaled, y, family)
    elif certify and family == "poisson":
        _poisson_check(scaled, y)
    if family == "linear":
        factor = finite(y - scaled @ coefficient, "Saved weighted linear residual")
        objective = float(finite((normalized * factor.square()).sum(), "Saved weighted RSS"))
        score = finite(scaled.T @ (normalized * factor), "Saved weighted linear score")
        sensitivity = finite((scaled.T * normalized) @ scaled, "Saved linear sensitivity")
        # Replay the analytic solution itself: inverse-amplified residuals do
        # not give a conditioning-independent coefficient accuracy threshold.
        linear_expected = _solve(sensitivity, scaled.T @ (normalized * y)) / scales
    else:
        objective, score, sensitivity, factor = _components(
            scaled, y, normalized, coefficient, family
        )
    direction = _solve(sensitivity, score)
    bread = finite(
        scales[:, None] * sensitivity * scales[None, :], "Saved original-scale sensitivity"
    )
    rows = torch.zeros((n, k), dtype=FLOAT)
    rows[positions] = finite(X * (normalized * factor)[:, None], "Saved weighted score rows")
    sensitivity_scale = bread.diag().sqrt()
    correlation = bread / sensitivity_scale[:, None] / sensitivity_scale[None, :]
    influences = finite(
        torch.linalg.solve(correlation, (rows / sensitivity_scale).T).T / sensitivity_scale,
        "Three-stage row coefficient influences",
    )
    stage1, stage2, stage3 = stage_covariances(influences, design)
    covariance = finite(stage1 + stage2 + stage3, "Three-stage coefficient covariance")
    return dict(
        bread=bread,
        row_scores=rows,
        stage1_covariance=stage1,
        stage2_covariance=stage2,
        stage3_covariance=stage3,
        covariance=covariance,
        score=score,
        normalized_weights=normalized,
        scaled_X=scaled,
        scaled_beta=coefficient,
        factor=factor,
        direction=direction,
        objective=objective,
        raw_beta=beta,
        linear_expected=linear_expected,
    )


def _checked_replay(function):
    """Turn out-of-range serialized arithmetic into typed validation rejection."""

    @wraps(function)
    def checked(self):
        try:
            return function(self)
        except OverflowError as error:
            raise ValueError(
                "Saved regression numeric fields must be bounded finite float64 values."
            ) from error

    return checked


class SurveyThreeStageRegressionResult(BaseModel):
    """Bounded saved regression with all three SRSWOR stage covariance components."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["survey-three-stage-regression-result-v1"] = (
        "survey-three-stage-regression-result-v1"
    )
    method: Literal["three-stage-taylor"] = "three-stage-taylor"
    family: Literal["linear", "logit", "probit", "poisson"]
    outcome: str = Field(min_length=1, max_length=200)
    regressors: tuple[str, ...] = Field(max_length=31)
    intercept: StrictBool
    labels: tuple[str, ...] = Field(min_length=1, max_length=32)
    coefficients: tuple[StrictFloat, ...] = Field(min_length=1, max_length=32)
    covariance: tuple[tuple[StrictFloat, ...], ...] = Field(min_length=1, max_length=32)
    df: StrictInt = Field(ge=0)
    alpha: StrictFloat = Field(ge=1e-8, le=1 - 1e-8)
    null: tuple[StrictFloat, ...] = Field(min_length=1, max_length=32)
    design: SurveyThreeStageDesign
    metadata: dict[str, Any]
    integrity_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _bounded(cls, raw):
        if isinstance(raw, Mapping):
            design, metadata = raw.get("design"), raw.get("metadata")
            validation = design.get("validation") if isinstance(design, dict) else None
            if not isinstance(design, SurveyThreeStageDesign) and not isinstance(
                validation, Mapping
            ):
                raise ValueError(
                    "Saved regression needs a complete typed design validation record."
                )
            n = (
                design.validation.nobs
                if isinstance(design, SurveyThreeStageDesign)
                else validation.get("nobs")
            )
            k = len(raw.get("labels", [])) if isinstance(raw.get("labels"), (list, tuple)) else 0
            if (
                type(n) is not int
                or not 1 <= n <= 1_000_000
                or not 1 <= k <= 32
                or not isinstance(metadata, dict)
            ):
                raise ValueError("Saved three-stage regression dimensions are inadmissible.")
            for key in ("coefficients", "null"):
                values = raw.get(key)
                if not isinstance(values, (list, tuple)) or len(values) != k:
                    raise ValueError(
                        "Saved coefficient numeric vectors must match bounded labels before parsing."
                    )
            covariance = raw.get("covariance")
            if (
                not isinstance(covariance, (list, tuple))
                or len(covariance) != k
                or any(not isinstance(row, (list, tuple)) or len(row) != k for row in covariance)
            ):
                raise ValueError("Saved covariance must be bounded k by k before numeric parsing.")
            used = metadata.get("n_used")
            groups = (
                design.validation.n_psu
                if isinstance(design, SurveyThreeStageDesign)
                else validation.get("n_psu")
            )
            ssus = (
                design.validation.n_ssu
                if isinstance(design, SurveyThreeStageDesign)
                else validation.get("n_ssu")
            )
            memory = (
                design.max_memory_mb
                if isinstance(design, SurveyThreeStageDesign)
                else design.get("max_memory_mb", 64)
            )
            if (
                type(used) is not int
                or not k < used <= n
                or type(groups) is not int
                or not 1 <= groups <= n
                or type(ssus) is not int
                or not groups <= ssus <= n
                or type(memory) is not int
                or not 1 <= memory <= 512
                or raw.get("family") not in ("linear", "logit", "probit", "poisson")
                or replay_work(n, used, k, groups, ssus, raw.get("family")) * 2 > MAX_WORK
            ):
                raise ValueError("Saved three-stage regression exceeds bounded replay dimensions.")
            regression_workspace(n, n, k, groups, ssus, memory)
        return raw

    @model_validator(mode="after")
    @cpu_call
    @_checked_replay
    def _replay(self):
        k, n = len(self.labels), self.design.validation.nobs
        expected_labels = (("_cons",) if self.intercept else ()) + self.regressors
        if (
            self.labels != expected_labels
            or len(set(self.regressors)) != len(self.regressors)
            or any(not v.strip() or len(v) > 200 for v in self.regressors)
            or not self.outcome.strip()
            or self.outcome in self.regressors
            or self.intercept
            and "_cons" in self.regressors
            or len(self.coefficients) != k
            or len(self.null) != k
            or any(not math.isfinite(v) for v in (*self.coefficients, *self.null))
            or self.df != self.design.validation.design_df
        ):
            raise ValueError(
                "Saved coefficient order, finite values or first-stage reference df differ."
            )
        positions = validate_sample_metadata(self.metadata, self.design)
        used = len(positions)
        if (
            self.metadata.get("outcome") != self.outcome
            or self.metadata.get("regressors") != list(self.regressors)
            or self.metadata.get("parameter_order") != list(self.labels)
            or self.metadata.get("outcomes") != [self.outcome, *self.regressors]
            or self.metadata.get("denominators") != []
            or self.metadata.get("categories") is not None
            or self.metadata.get("model_exclusions") != self.metadata.get("outcome_exclusions")
            or self.metadata.get("calibration_support") is not False
            or self.family == "poisson"
            and self.metadata.get("raw_outcome_count_admissible") is not True
        ):
            raise ValueError("Saved model roles and admission metadata differ.")
        required = regression_workspace(
            n,
            n,
            k,
            self.design.validation.n_psu,
            self.design.validation.n_ssu,
            self.design.max_memory_mb,
        ).record()
        if self.metadata.get("workspace") != required:
            raise ValueError("Saved regression workspace does not match complete dimensions.")
        cost = replay_work(
            n, used, k, self.design.validation.n_psu, self.design.validation.n_ssu, self.family
        )
        work, convergence = self.metadata.get("work"), self.metadata.get("convergence")
        if (
            not isinstance(work, dict)
            or work.get("max_work_units") != MAX_WORK
            or work.get("replay_work_units") != cost * 2
            or type(work.get("planned_work_units")) is not int
            or not 1 <= work["planned_work_units"] <= MAX_WORK
            or type(work.get("actual_work_units")) is not int
            or not isinstance(convergence, dict)
            or type(convergence.get("work_units")) is not int
            or work["actual_work_units"] != convergence["work_units"] + cost * 2
            or work["actual_work_units"] > MAX_WORK
        ):
            raise ValueError("Saved cumulative fit and replay work is incomplete.")
        max_iter = convergence.get("max_iter")
        if type(max_iter) is not int or not 1 <= max_iter <= 1000:
            raise ValueError("Saved fit needs bounded declared iterations.")
        overhead = n * k * 4 + self.design.validation.n_psu * k * k * 4
        lp_constraints = max(8, 4 * k) + 2 * k
        diagnostic = (
            0
            if self.family == "linear"
            else (64 * used * k * 3 + 64 * 100 * lp_constraints * k * k)
        )
        planned = (
            overhead
            + diagnostic
            + used * k * k * (1 if self.family == "linear" else max_iter)
            + cost * 2
        )
        if work["planned_work_units"] != planned:
            raise ValueError(
                "Saved cumulative work plan differs from declared fit and replay dimensions."
            )
        X = _matrix(self.metadata.get("primitive_X"), used, k, "primitive design")
        raw_y = self.metadata.get("primitive_y")
        if not isinstance(raw_y, (list, tuple)) or len(raw_y) != used:
            raise ValueError("Saved primitive responses need exactly the admitted row count.")
        y_matrix = _matrix([[v] for v in raw_y], used, 1, "primitive responses")
        y = y_matrix[:, 0]
        if self.intercept and not bool((X[:, 0] == 1).all()):
            raise ValueError("Saved intercept column must be exactly one.")
        values = torch.zeros((n, 1 + len(self.regressors)), dtype=FLOAT)
        values[positions, 0] = y
        values[positions, 1:] = X[:, int(self.intercept) :]
        selected = [False] * n
        for i in positions:
            selected[i] = True
        if self.metadata.get("sample_input_sha256") != digest(
            [
                self.design.validation.design_input_sha256,
                selected,
                values.tolist(),
                torch.ones_like(values).tolist(),
            ]
        ):
            raise ValueError(
                "Saved primitive rows differ from the admitted numeric sample fingerprint."
            )
        beta = torch.tensor(self.coefficients, dtype=FLOAT)
        replay = replay_fit(self.design, X, y, positions, beta, self.family)
        _fit_evidence(convergence, replay, y, self.family, used, k, n, self.design.validation.n_psu)
        for name in (
            "bread",
            "row_scores",
            "stage1_covariance",
            "stage2_covariance",
            "stage3_covariance",
        ):
            shape = (n, k) if name == "row_scores" else (k, k)
            stored = _matrix(self.metadata.get(name), *shape, name)
            _close(stored, replay[name], name)
            if name.startswith("stage"):
                validate_covariance(self.metadata[name], k)
        covariance = validate_covariance(self.covariance, k)
        _close(covariance, replay["covariance"], "full three-stage covariance")
        payload = self.model_dump(mode="json", exclude={"integrity_sha256"})
        if self.integrity_sha256 != digest(payload):
            raise ValueError("Saved three-stage regression integrity fingerprint differs.")
        return self

    @property
    def estimates(self):
        return self.coefficients

    @property
    def target(self):
        return "coefficients"

    def to_frame(self) -> pd.DataFrame:
        """Coefficient inference with explicit first-stage reference degrees of freedom."""
        state = restore(self)
        frame = inference_frame(state)
        frame.attrs["survey_three_stage_regression_state"] = state.model_dump(mode="json")
        return frame

    def predict(self, data, *, kind="response", missing="raise", alpha=None) -> pd.DataFrame:
        """Conditional fitted means on up to 256 rows, with joint coefficient covariance."""
        return predict(self, data, kind=kind, missing=missing, alpha=alpha)

    def lincom(self, coefficients, *, null=0.0, alpha=None) -> pd.DataFrame:
        """A coefficient linear combination using all three sampling-stage variance terms."""
        return lincom(self, coefficients, null=null, alpha=alpha)

    def test(self, restrictions, *, null=None) -> pd.DataFrame:
        """Adjusted joint Wald F under the declared first-stage reference convention."""
        return test(self, restrictions, null=null)


def restore(result):
    if not isinstance(result, SurveyThreeStageRegressionResult):
        raise AnalysisError(
            "invalid_survey_result", "Supply a saved SurveyThreeStageRegressionResult."
        )
    try:
        return SurveyThreeStageRegressionResult.model_validate_json(result.model_dump_json())
    except (ValueError, TypeError, AnalysisError, RuntimeError) as error:
        raise AnalysisError(
            "invalid_survey_result", "Saved three-stage coefficient state failed replay."
        ) from error


def _target(state, estimates, covariance, labels, *, alpha, null, target, metadata=None):
    proxy = SimpleNamespace(
        estimates=estimates,
        covariance=covariance,
        labels=labels,
        null=null,
        alpha=alpha,
        df=state.df,
        method=state.method,
        target=target,
        model_dump=state.model_dump,
    )
    frame = inference_frame(proxy)
    frame.attrs.update(
        {
            "survey_three_stage_regression_state": state.model_dump(mode="json"),
            "conditional_fixed_covariates": True,
            "population_distribution_uncertainty": False,
            **(metadata or {}),
        }
    )
    return frame


@cpu_call
def lincom(result, coefficients, *, null=0.0, alpha=None):
    state = restore(result)
    if isinstance(coefficients, Mapping):
        if any(v not in state.labels for v in coefficients):
            raise AnalysisError(
                "invalid_survey_option", "Linear-combination names must be fitted coefficients."
            )
        coefficients = [coefficients.get(v, 0.0) for v in state.labels]
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != len(state.labels):
        raise AnalysisError(
            "invalid_survey_option",
            "Declare one finite linear-combination coefficient per parameter.",
        )
    vector = torch.tensor([number(v, "linear combination") for v in coefficients], dtype=FLOAT)
    beta, covariance = (
        torch.tensor(state.coefficients, dtype=FLOAT),
        torch.tensor(state.covariance, dtype=FLOAT),
    )
    estimate = finite((vector @ beta).reshape(1), "Linear combination")
    variance = finite((vector @ covariance @ vector).reshape(1, 1), "Linear-combination variance")
    alpha = state.alpha if alpha is None else number(alpha, "alpha", 1e-8, 1 - 1e-8)
    return _target(
        state,
        estimate.tolist(),
        variance.tolist(),
        ["lincom"],
        alpha=alpha,
        null=[number(null, "null")],
        target="lincom",
        metadata={"jacobian": [vector.tolist()]},
    )


@cpu_call
def predict(result, data, *, kind="response", missing="raise", alpha=None):
    from .regression_postest import _data, _link

    state = restore(result)
    if not isinstance(kind, str) or kind not in {"linear", "response"}:
        raise AnalysisError("invalid_survey_option", "Prediction kind must be linear or response.")
    labels, X, _, metadata = _data(state, data, missing=missing, max_rows=256)
    workspace = plan_workspace(
        "three-stage conditional mean inference",
        {
            "conditional rows and Jacobians": len(X) * (len(state.labels) + 3) * 192,
            "joint target covariance and serialization": len(X) ** 2 * 192,
        },
        budget_bytes=min(workspace_budget_bytes(), state.design.max_memory_mb * 1024**2),
    )
    metadata["workspace"] = workspace.record()
    eta = finite(X @ torch.tensor(state.coefficients, dtype=FLOAT), "Conditional linear prediction")
    if kind == "linear":
        means, jacobian = eta, X
    else:
        means, derivative, _ = _link(state.family, eta)
        if state.family == "poisson" and bool((means <= 0).any()):
            raise AnalysisError(
                "survey_numerical_failure", "Poisson conditional means underflowed."
            )
        jacobian = finite(X * derivative[:, None], "Conditional prediction Jacobian")
    covariance = finite(
        jacobian @ torch.tensor(state.covariance, dtype=FLOAT) @ jacobian.T,
        "Joint conditional mean covariance",
    )
    covariance = (covariance + covariance.T) / 2
    alpha = state.alpha if alpha is None else number(alpha, "alpha", 1e-8, 1 - 1e-8)
    return _target(
        state,
        means.tolist(),
        covariance.tolist(),
        labels.tolist(),
        alpha=alpha,
        null=[0.0] * len(means),
        target=f"predict:{kind}",
        metadata={
            **metadata,
            "jacobian": jacobian.tolist(),
            "uncertainty": "three-stage coefficient uncertainty; fixed covariates; mean intervals",
        },
    )


@cpu_call
def test(result, restrictions, *, null=None):
    state = restore(result)
    k = len(state.labels)
    if (
        not isinstance(restrictions, (list, tuple))
        or not 1 <= len(restrictions) <= k
        or any(not isinstance(row, (list, tuple)) or len(row) != k for row in restrictions)
    ):
        raise AnalysisError(
            "invalid_survey_option", "Use 1..K finite, independent restriction rows of width K."
        )
    q = len(restrictions)
    if q > state.df:
        raise AnalysisError(
            "survey_test_unavailable",
            "Joint testing requires first-stage reference df at least restriction rank.",
        )
    R = torch.tensor([[number(v, "restriction") for v in row] for row in restrictions], dtype=FLOAT)
    if null is None:
        null = [0.0] * q
    if not isinstance(null, (list, tuple)) or len(null) != q:
        raise AnalysisError("invalid_survey_option", "Declare one finite null per restriction.")
    target = torch.tensor([number(v, "null") for v in null], dtype=FLOAT)
    row_scale = R.abs().amax(1)
    if bool((row_scale == 0).any()):
        raise AnalysisError(
            "survey_regression_rank", "Restrictions must be nonzero and independent."
        )
    R, target = R / row_scale[:, None], target / row_scale
    singular = torch.linalg.svdvals(R)
    if float(singular[-1]) <= 10 * torch.finfo(FLOAT).eps * max(R.shape) * float(singular[0]):
        raise AnalysisError("survey_regression_rank", "Restriction rows must have full rank.")
    covariance = finite(
        R @ torch.tensor(state.covariance, dtype=FLOAT) @ R.T, "Restriction covariance"
    )
    scale = covariance.diag().sqrt()
    if bool((scale <= 0).any()):
        raise AnalysisError(
            "survey_test_unavailable", "The restriction covariance must be positive definite."
        )
    correlation = covariance / scale[:, None] / scale[None, :]
    difference = finite(
        (R @ torch.tensor(state.coefficients, dtype=FLOAT) - target) / scale,
        "Scaled restriction difference",
    )
    wald = float(finite(difference @ _solve(correlation, difference), "Joint Wald statistic"))
    denominator_df = state.df - q + 1
    statistic = denominator_df / state.df * wald / q
    frame = pd.DataFrame(
        [[wald, statistic, q, denominator_df, f_sf(statistic, q, denominator_df)]],
        columns=["wald", "statistic", "numerator_df", "denominator_df", "p_value"],
    )
    frame.attrs.update(
        survey_three_stage_regression_state=state.model_dump(mode="json"),
        restrictions=[list(row) for row in restrictions],
        null=list(null),
        covariance_matrix=covariance.tolist(),
        design_df=state.df,
        design_df_convention=state.metadata["design_df_convention"],
        inference="adjusted joint Wald F under first-stage reference df",
    )
    return frame
