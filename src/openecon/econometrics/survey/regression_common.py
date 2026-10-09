"""Single-stage regression admission, complete-design covariance and saved state."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from functools import wraps
import math
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.survey import SurveyDesign
from openecon.survey import _scalar
from .common import FLOAT, digest, finite, number, prepare

MAX_PARAMETERS = 32
MAX_WORK = 50_000_000


def cpu_call(function):
    """Keep this explicitly resident CPU API independent of ambient Torch defaults."""
    @wraps(function)
    def execute(*args, **kwargs):
        with torch.no_grad(), torch.device("cpu"):
            return function(*args, **kwargs)
    return execute


def covariance_from_scores(bread, scores, strata, corrections):
    """Full PSU covariance, without an ordinary-regression residual-df multiplier."""
    scale = bread.diag().sqrt()
    scaled = bread / scale[:, None] / scale[None, :]
    influence = finite(torch.linalg.solve(scaled, (scores / scale).T).T / scale,
                       "PSU regression influence")
    covariance = torch.zeros_like(bread)
    for groups, correction in zip(strata, corrections):
        if correction == 0:
            continue
        block = influence[groups]
        centered = block - block.mean(0)
        covariance += correction * len(groups) / (len(groups) - 1) * (centered.T @ centered)
    return finite((covariance + covariance.T) / 2, "Survey regression covariance")


def covariance_from_replicates(coefficients, record, design, positions, *, original_work=0):
    """Validate saved full coefficient replicas before replaying their covariance."""
    if not isinstance(record, dict):
        raise ValueError("Saved regression replication must be a complete record.")
    count = record.get("replicate_count")
    width = len(coefficients)
    if type(count) is not int or not 2 <= count <= 4096:
        raise ValueError("Saved regression needs 2..4096 complete coefficient replicas.")
    if (count * width * width > MAX_WORK
            or design.validation.nobs * count > 8_000_000
            or design.validation.nobs * width * count > MAX_WORK):
        raise ValueError("Saved coefficient replica replay exceeds its work budget.")
    plan_workspace("saved survey coefficient replica replay",
                   {"replicas, covariance and validation": count * width * 192 + width * width * 512},
                   budget_bytes=workspace_budget_bytes())
    method = record.get("method")
    fingerprint = record.get("replicate_weight_sha256")
    generation = record.get("generation")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(v not in "0123456789abcdef" for v in fingerprint)
            or not isinstance(generation, str) or not generation or len(generation) > 2000):
        raise ValueError("Saved coefficient replicas need complete weight-plan fingerprints and provenance.")
    ids = record.get("replicate_ids")
    estimates = record.get("replicate_estimates")
    multipliers = record.get("variance_multipliers")
    convergence = record.get("replicate_convergence")
    if (not isinstance(method, str) or method not in {"brr", "fay", "jackknife", "bootstrap"}
            or not isinstance(ids, list) or len(ids) != count
            or any(not isinstance(v, str) or not v or len(v) > 200 for v in ids)
            or len(set(ids)) != count
            or not isinstance(estimates, list) or len(estimates) != count
            or any(not isinstance(row, list) or len(row) != width for row in estimates)
            or not isinstance(multipliers, list) or len(multipliers) != count
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in multipliers)
            or not isinstance(convergence, list) or len(convergence) != count
            or record.get("failed_replicates") != []):
        raise ValueError("Saved coefficient replica IDs/estimates/convergence are incomplete.")
    allowed_positions = set(positions)
    for identity, fit in zip(ids, convergence):
        if (not isinstance(fit, dict) or fit.get("converged") is not True
                or fit.get("replicate_id") != identity
                or type(fit.get("max_iter")) is not int or not 1 <= fit["max_iter"] <= 1000
                or type(fit.get("iterations")) is not int or not 1 <= fit["iterations"] <= fit["max_iter"]
                or type(fit.get("n_used")) is not int or not width < fit["n_used"] <= len(positions)
                or type(fit.get("work_units")) is not int or not 1 <= fit["work_units"] <= MAX_WORK):
            raise ValueError("Every saved coefficient replica needs accepted finite fit evidence.")
        fitted = fit.get("sample_positions")
        if (not isinstance(fitted, list) or len(fitted) != fit["n_used"]
                or any(type(v) is not int for v in fitted) or fitted != sorted(set(fitted))
                or any(v not in allowed_positions for v in fitted)):
            raise ValueError("Saved replica positions differ from the original admitted sample.")
    if (type(original_work) is not int or not 0 <= original_work <= MAX_WORK
            or original_work + sum(fit["work_units"] for fit in convergence) > MAX_WORK):
        raise ValueError("Saved original and replica fits exceed the cumulative work budget.")
    work = record.get("work")
    if (not isinstance(work, dict) or work.get("max_work_units") != MAX_WORK
            or type(work.get("planned_work_units")) is not int
            or not 1 <= work["planned_work_units"] <= MAX_WORK
            or type(work.get("actual_work_units")) is not int
            or not original_work + sum(fit["work_units"] for fit in convergence)
            <= work["actual_work_units"] <= MAX_WORK):
        raise ValueError("Saved coefficient replica cumulative work evidence is incomplete.")
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in estimates for v in row):
        raise ValueError("Saved coefficient replicas must contain finite real numbers.")
    centering = record.get("centering")
    if not isinstance(centering, str) or centering not in {"original", "replicate_mean", "stratum_mean"}:
        raise ValueError("Saved coefficient replica centering is invalid.")
    strata = record.get("replicate_stratum_indices")
    fpc = [1 - s.n_psu / s.population_psu if s.population_psu else 1.
           for s in design.validation.strata]
    stochastic = [h for h, correction in enumerate(fpc) if correction]
    if method in {"brr", "fay"}:
        rho = record.get("fay_rho")
        if (type(rho) not in (int, float) or not math.isfinite(rho) or not 0 <= rho <= 1 - 1e-6
                or method == "brr" and rho != 0
                or not stochastic or count <= len(stochastic)
                or any(fpc[h] != 1 or design.validation.strata[h].n_psu != 2 for h in stochastic)
                or centering == "stratum_mean"):
            raise ValueError("Saved BRR/Fay replica geometry or rho is incompatible.")
        expected = [1 / (count * (1 - rho) ** 2)] * count
        raw_signs = record.get("balanced_signs")
        width_signs = len(stochastic)
        if (count * width_signs ** 2 > MAX_WORK
                or not isinstance(raw_signs, list) or len(raw_signs) != count
                or any(not isinstance(row, list) or len(row) != width_signs for row in raw_signs)
                or any(type(v) not in (int, float) or v not in (-1., 1.)
                       for row in raw_signs for v in row)
                or record.get("balanced_signs_sha256") != digest(raw_signs)):
            raise ValueError("Saved BRR/Fay needs bounded complete balanced selection signs.")
        plan_workspace("saved BRR/Fay balance replay", {"signs and validation": count * width_signs * 192},
                       budget_bytes=workspace_budget_bytes())
        signs = torch.tensor(raw_signs, dtype=FLOAT)
        if not torch.equal(signs.sum(0), torch.zeros(width_signs, dtype=FLOAT)):
            raise ValueError("Saved BRR/Fay stratum signs are unbalanced.")
        for column in range(1, width_signs):
            if not bool(((signs[:, :column].T @ signs[:, column]) == 0).all()):
                raise ValueError("Saved BRR/Fay stratum signs must be orthogonal.")
        if "Sylvester" in generation and (
                count & (count - 1) or ids != [f"sylvester-{i}" for i in range(count)]
                or any(row != [(-1.) ** ((i & j).bit_count()) for j in range(1, width_signs + 1)]
                       for i, row in enumerate(raw_signs))):
            raise ValueError("Saved generated BRR/Fay differs from its declared Sylvester plan.")
    elif method == "jackknife":
        expected_strata = [h for h in stochastic for _ in range(design.validation.strata[h].n_psu)]
        if (not isinstance(strata, list) or len(strata) != count
                or any(type(h) is not int for h in strata)
                or strata != expected_strata or centering == "replicate_mean"):
            raise ValueError("Saved jackknife needs every noncensus stratum's delete-one PSU replicas.")
        expected = [fpc[h] * (design.validation.strata[h].n_psu - 1)
                    / design.validation.strata[h].n_psu for h in strata]
    else:
        scale = record.get("scale")
        rscales = record.get("rscales")
        justification = record.get("justification")
        if (type(scale) not in (int, float) or not math.isfinite(scale) or not 1e-15 <= scale <= 1e15
                or not isinstance(rscales, list) or len(rscales) != count
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 1e-15 <= v <= 1e15
                       for v in rscales)
                or not isinstance(justification, str) or not 1 <= len(justification.strip()) <= 2000
                or centering == "stratum_mean"):
            raise ValueError("Saved design-bootstrap needs explicit scales and sampling provenance.")
        expected = [scale * value for value in rscales]
    if len(expected) != count or any(not math.isclose(v, e, rel_tol=1e-12, abs_tol=0.)
                                     for v, e in zip(multipliers, expected)):
        raise ValueError("Saved coefficient replica variance multipliers disagree with their method.")
    replicas = torch.tensor(estimates, dtype=FLOAT)
    if centering == "original":
        difference = replicas - torch.tensor(coefficients, dtype=FLOAT)
    elif centering == "replicate_mean":
        difference = replicas - replicas.mean(0)
    else:
        difference = torch.empty_like(replicas)
        for h in sorted(set(strata)):
            selection = [i for i, value in enumerate(strata) if value == h]
            difference[selection] = replicas[selection] - replicas[selection].mean(0)
    return finite((difference.T * torch.tensor(multipliers, dtype=FLOAT)) @ difference,
                  "Saved coefficient replica covariance")


class SurveyRegressionResult(BaseModel):
    """Integrity-checked complete coefficient state; its digest is not authentication."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["survey-regression-result-v1"] = "survey-regression-result-v1"
    family: Literal["linear", "logit", "probit", "poisson"]
    outcome: str
    regressors: tuple[str, ...]
    intercept: bool = Field(strict=True)
    labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_PARAMETERS)
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    df: StrictInt = Field(ge=0)
    alpha: float = Field(gt=0, lt=1)
    null: tuple[float, ...]
    design: SurveyDesign
    metadata: dict[str, Any]
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    @cpu_call
    def validate_state(self):
        k = len(self.labels)
        if self.design.validation.nobs > self.design.max_rows:
            raise ValueError("Saved regression exceeds declared row limits.")
        plan_workspace("saved survey regression score replay",
                       {"score/replay/serialization": self.design.validation.n_psu*k*192 + k*k*512},
                       budget_bytes=workspace_budget_bytes())
        if self.design.validation.n_psu*k*k*4 > MAX_WORK:
            raise ValueError("Saved regression PSU replay exceeds its work budget.")
        expected = (("_cons",) if self.intercept else ()) + self.regressors
        if (not self.outcome or self.outcome in self.regressors
                or self.labels != expected or len(set(self.labels)) != k
                or any(not v or len(v) > 200 for v in self.regressors)
                or len(self.coefficients) != k or len(self.null) != k
                or len(self.covariance) != k or any(len(r) != k for r in self.covariance)
                or any(not math.isfinite(v) for v in (*self.coefficients, *self.null))
                or "replication" not in self.metadata and self.df != self.design.validation.design_df):
            raise ValueError("Survey regression parameter/df geometry is inconsistent.")
        cov = torch.tensor(self.covariance, dtype=FLOAT)
        scale = max(float(cov.abs().max()), 1e-300)
        if (not torch.isfinite(cov).all()
                or not torch.allclose(cov, cov.T, atol=scale * 1e-12, rtol=1e-12)
                or float(torch.linalg.eigvalsh(cov / scale).min()) < -1e-10):
            raise ValueError("Survey regression covariance must be finite symmetric PSD.")
        positions = self.metadata.get("sample_positions")
        if (not isinstance(positions, list) or not positions
                or any(type(i) is not int or not 0 <= i < self.design.validation.nobs
                       for i in positions)
                or positions != sorted(set(positions))
                or self.metadata.get("n_used") != len(positions)
                or self.metadata.get("n_design") != self.design.validation.nobs
                or self.metadata.get("parameter_order") != list(self.labels)):
            raise ValueError("Survey regression original sample/order is inconsistent.")
        convergence = self.metadata.get("convergence")
        if (not isinstance(convergence, dict) or convergence.get("converged") is not True
                or type(convergence.get("iterations")) is not int
                or convergence["iterations"] < 0):
            raise ValueError("Saved survey regression needs accepted actual convergence.")
        groups = self.metadata.get("stratum_psu_indices")
        corrections = self.metadata.get("fpc_multipliers")
        expected_fpc = [1 - s.n_psu / s.population_psu if s.population_psu else 1.
                        for s in self.design.validation.strata]
        if (not isinstance(groups, list) or len(groups) != len(expected_fpc)
                or any(not isinstance(g, list) for g in groups)
                or any(any(type(i) is not int for i in g) for g in groups)
                or [len(g) for g in groups] != [s.n_psu for s in self.design.validation.strata]
                or sorted(i for g in groups for i in g) != list(range(self.design.validation.n_psu))
                or corrections != expected_fpc):
            raise ValueError("Saved complete PSU/FPC geometry differs from its declaration.")
        scores = self.metadata.get("psu_score_sums")
        bread = self.metadata.get("sensitivity")
        if (not isinstance(scores, list) or len(scores) != self.design.validation.n_psu
                or any(not isinstance(r, list) or len(r) != k for r in scores)
                or not isinstance(bread, list) or len(bread) != k
                or any(not isinstance(r, list) or len(r) != k for r in bread)):
            raise ValueError("Saved regression score/sensitivity dimensions are inconsistent.")
        a, s = torch.tensor(bread, dtype=FLOAT), torch.tensor(scores, dtype=FLOAT)
        if not torch.isfinite(a).all() or not torch.isfinite(s).all():
            raise ValueError("Saved regression score/sensitivity must be finite.")
        if bool((a.diag() <= 0).any()):
            raise ValueError("Saved regression sensitivity needs positive identified diagonals.")
        a_scale = max(float(a.abs().max()), 1e-300)
        scaled = a / a.diag().sqrt()[:, None] / a.diag().sqrt()[None, :]
        if (not torch.allclose(a, a.T, atol=a_scale * 1e-12, rtol=1e-12)
                or float(torch.linalg.eigvalsh(scaled).min()) <= 1e-12):
            raise ValueError("Saved regression sensitivity must have identified full rank.")
        if "replication" in self.metadata:
            if self.family not in {"linear", "logit", "probit", "poisson"}:
                raise ValueError("Coefficient replication requires a supported numeric regression family.")
            replication = self.metadata["replication"]
            replay = covariance_from_replicates(self.coefficients, replication, self.design, positions,
                                               original_work=convergence.get("work_units"))
            explicit = replication.get("df_explicit")
            count = replication["replicate_count"]
            if (type(explicit) is not bool
                    or replication.get("replicate_df_convention") !=
                    "minimum of complete-design df and R-1 unless explicitly declared"
                    or explicit and (replication["method"] != "bootstrap" or not 1 <= self.df < count)
                    or not explicit and self.df != min(self.design.validation.design_df, count - 1)):
                raise ValueError("Saved coefficient replica degrees of freedom are inconsistent.")
        else:
            replay = covariance_from_scores(a, s, groups, corrections)
        if not torch.allclose(cov, replay, atol=scale * 1e-10, rtol=1e-10):
            raise ValueError("Saved coefficient covariance differs from complete variance replay.")
        if digest(self.model_dump(mode="json", exclude={"integrity_sha256"})) != self.integrity_sha256:
            raise ValueError("Survey regression state integrity changed.")
        return self

    def to_frame(self):
        """Restore coefficient design-t inference without refitting or original observations."""
        state = restore(self)
        return inference_frame(state, state.coefficients, state.covariance, state.labels,
                               null=state.null, target="coefficients")


def restore(result):
    if not isinstance(result, SurveyRegressionResult):
        raise AnalysisError("invalid_survey_result", "Supply a saved SurveyRegressionResult.")
    return SurveyRegressionResult.model_validate_json(result.model_dump_json())


@cpu_call
def inference_frame(state, estimates, covariance, labels, *, null=0.0, alpha=None, target="contrast"):
    alpha = state.alpha if alpha is None else number(alpha, "alpha", 1e-8, 1-1e-8)
    null = list(null) if isinstance(null, (list, tuple)) else [number(null, "null")] * len(estimates)
    if len(null) != len(estimates):
        raise AnalysisError("invalid_survey_option", "One null per requested target is required.")
    critical = critical_value(alpha, state.df) if state.df else None
    rows = []
    for i, estimate in enumerate(estimates):
        estimate = number(estimate, "estimate")
        se = math.sqrt(max(float(covariance[i][i]), 0.0))
        statistic = (estimate-null[i])/se if se > 0 and state.df else None
        p = student_t_two_sided(statistic, state.df) if statistic is not None else None
        width = critical * se if critical is not None else 0.
        rows.append([estimate, se, statistic, p, estimate-width, estimate+width])
    frame = pd.DataFrame(rows, index=list(labels),
                         columns=["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"])
    frame.attrs.update(survey_regression_state=state.model_dump(mode="json"),
                       covariance_matrix=[list(map(float, row)) for row in covariance],
                       target=target, method=("single-stage " + state.metadata["replication"]["method"]
                                              + " coefficient delta" if "replication" in state.metadata
                                              else "single-stage Taylor coefficient delta"),
                       design_df=state.df, alpha=alpha,
                       uncertainty="design coefficient uncertainty; fixed supplied covariates, no future-outcome interval")
    return frame


@dataclass
class RegressionSample:
    design: SurveyDesign
    outcome: str
    regressors: list[str]
    intercept: bool
    labels: list[str]
    X: torch.Tensor
    y: torch.Tensor
    weights: torch.Tensor
    selected: torch.Tensor
    groups: torch.Tensor
    strata_groups: list[list[int]]
    fpc: list[float]
    metadata: dict


def admit(data, design, outcome, regressors, *, intercept=True, domain=None, missing="raise"):
    if (type(intercept) is not bool or not isinstance(outcome, str) or not outcome.strip()
            or len(outcome) > 200 or not isinstance(regressors, (str, list, tuple))):
        raise AnalysisError("invalid_survey_option", "Declare one outcome, numeric regressors and Boolean intercept.")
    regressors = [regressors] if isinstance(regressors, str) else list(regressors)
    if (len(regressors) > 31
            or any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in regressors)
            or len(set(regressors)) != len(regressors)
            or outcome in regressors or intercept and "_cons" in regressors
            or not regressors and not intercept):
        raise AnalysisError("invalid_survey_option", "Use up to 31 distinct regressors excluding outcome/intercept label.")
    if not isinstance(design, SurveyDesign):
        raise AnalysisError("invalid_survey_design", "Supply a validated one-stage SurveyDesign.")
    if domain is not None and (not isinstance(domain, str) or not domain.strip()):
        raise AnalysisError(
            "invalid_survey_domain", "domain must name a complete Boolean/0-1 indicator column."
        )
    if isinstance(data, Mapping) or isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        design.revalidate(data)
        names = list(dict.fromkeys([design.weights, design.psu, design.strata, design.fpc,
                                   outcome, *regressors, domain]))
        names = [v for v in names if v is not None]
        try:
            if isinstance(data, Mapping):
                if any(v not in data for v in names):
                    raise AnalysisError("missing_column", "Every declared regression role must exist.")
                data = pd.DataFrame({v: data[v] for v in names})
            else:
                data = pd.DataFrame([{v: row.get(v) for v in names} for row in data])
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_data", "Regression columns need consistent resident scalar rows.") from exc
    target = prepare(data, design, "mean", [outcome, *regressors], domain=domain, missing=missing)
    selected = target.domain.bool()
    source_frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    raw_counts = [_scalar(source_frame[outcome].iloc[i]) for i in target.metadata["sample_positions"]]
    count_admissible = all(0 <= v <= 2**53 and int(v) == v for v in raw_counts)
    X = target.values[:, 1:]
    if intercept:
        X = torch.cat([target.domain[:, None], X], dim=1)
    labels = (["_cons"] if intercept else []) + regressors
    if int(selected.sum()) <= len(labels):
        raise AnalysisError("survey_regression_rank", "More complete in-domain rows than coefficients are required.")
    workspace = plan_workspace(
        "single-stage survey regression",
        {"fit, score, replay and serialization": len(X)*(X.shape[1]+6)*160
         + design.validation.n_psu*X.shape[1]*192 + X.shape[1]**2*512,
         "physical row/sample indexing": len(X)*256},
        budget_bytes=workspace_budget_bytes())
    metadata = {**target.metadata, "outcome": outcome, "regressors": regressors,
                "parameter_order": labels, "workspace": workspace.record(),
                "weight_semantics": "positive sampling weights; weighted pseudo-score normalized to mean one in fit sample",
                "covariance_correction": "PSU/stratum finite-sample and declared first-stage FPC only; no regression HC1 factor",
                "multistage_support": False, "calibration_support": False,
                "design_effect": None, "raw_outcome_count_admissible": count_admissible,
                "model_exclusions": target.metadata["outcome_exclusions"]}
    return RegressionSample(design, outcome, regressors, intercept, labels, X, target.values[:, 0],
                            target.weights, selected, target.groups, target.strata_groups, target.fpc, metadata)


def check_work(sample, iterations):
    k = len(sample.labels)
    work = int(sample.selected.sum()) * k*k * iterations + len(sample.X)*k*4 + sample.design.validation.n_psu*k*k*4
    if work > MAX_WORK:
        raise AnalysisError("survey_regression_budget", "Declared fit iterations and N*K*K exceed 50 million work units.")
    return work


def build_result(sample, family, beta, bread, score_rows, *, alpha, null, convergence,
                 covariance=None, df=None, replication=None):
    finite(beta, "Survey coefficients")
    finite(bread, "Survey sensitivity")
    finite(score_rows, "Survey score rows")
    if beta.shape != (len(sample.labels),) or bread.shape != (len(sample.labels), len(sample.labels)):
        raise AnalysisError("survey_regression_rank", "Coefficient/sensitivity dimensions differ from admitted regressors.")
    scores = torch.zeros((sample.design.validation.n_psu, len(sample.labels)), dtype=FLOAT)
    scores.index_add_(0, sample.groups, score_rows)
    cov = (covariance_from_scores(bread, scores, sample.strata_groups, sample.fpc)
           if covariance is None else finite(covariance, "Coefficient replica covariance"))
    if (covariance is None) != (replication is None) or (replication is None and df is not None):
        raise AnalysisError("invalid_survey_replicates", "Replica covariance needs a complete method record.")
    alpha = number(alpha, "alpha", 1e-8, 1-1e-8)
    values = [number(v, "null") for v in null] if isinstance(null, (list, tuple)) else [number(null, "null")] * len(beta)
    if len(values) != len(beta):
        raise AnalysisError("invalid_survey_option", "One null value per coefficient is required.")
    payload = dict(schema_version="survey-regression-result-v1", family=family, outcome=sample.outcome,
                   regressors=sample.regressors, intercept=sample.intercept, labels=sample.labels,
                   coefficients=beta.tolist(), covariance=cov.tolist(),
                   df=sample.design.validation.design_df if df is None else df,
                   alpha=alpha, null=values, design=sample.design.model_dump(mode="json"),
                   metadata={**sample.metadata, "convergence": convergence, "sensitivity": bread.tolist(),
                             "psu_score_sums": scores.tolist(), "stratum_psu_indices": sample.strata_groups,
                             "fpc_multipliers": sample.fpc,
                             "variance_formula": "A^-1 [sum_h (1-f_h)*m_h/(m_h-1)*sum centered PSU score outer products] A^-T"})
    if replication is not None:
        payload["metadata"].update(
            replication=replication,
            covariance_correction="complete coefficient refits with declared replicate variance multipliers",
            variance_formula="sum_r multiplier[r] * (beta[r]-center[r]) outer product",
        )
    return SurveyRegressionResult(**payload, integrity_sha256=digest(payload))
