"""Native coefficient fitting and recursive two-stage SRSWOR linearization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.survey import _scalar
from .common import digest
from .regression import _inference_options, _options, _work_budget, fit_sample
from .regression_common import MAX_WORK, RegressionSample, cpu_call
from .two_stage_common import prepare_two_stage
from .two_stage_regression_state import (
    SurveyTwoStageRegressionResult,
    regression_workspace,
    replay_fit,
    replay_work,
)


class _Ledger:
    def __init__(self):
        self.used = 0

    def reserve(self, units):
        if type(units) is not int or units < 0 or self.used + units > MAX_WORK:
            raise AnalysisError(
                "survey_regression_budget",
                "Cumulative two-stage fit and replay exceed 50 million work units.",
            )
        self.used += units


def _admit(data, design, outcome, regressors, *, intercept, domain, missing):
    if (
        type(intercept) is not bool
        or not isinstance(outcome, str)
        or not outcome.strip()
        or len(outcome) > 200
        or not isinstance(regressors, (str, list, tuple))
    ):
        raise AnalysisError(
            "invalid_survey_option",
            "Declare a numeric outcome, numeric regressors and Boolean intercept.",
        )
    regressors = [regressors] if isinstance(regressors, str) else list(regressors)
    if (
        len(regressors) > 31
        or any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in regressors)
        or len(set(regressors)) != len(regressors)
        or outcome in regressors
        or intercept
        and "_cons" in regressors
        or not regressors
        and not intercept
    ):
        raise AnalysisError(
            "invalid_survey_option",
            "Use up to 31 distinct regressors excluding the outcome and intercept label.",
        )
    target = prepare_two_stage(
        data, design, "mean", [outcome, *regressors], domain=domain, missing=missing
    )
    selected = target.domain.bool()
    positions = target.metadata["sample_positions"]
    k = len(regressors) + int(intercept)
    if len(positions) <= k:
        raise AnalysisError(
            "survey_regression_rank", "More complete in-domain rows than coefficients are required."
        )
    workspace = regression_workspace(
        len(target.weights), len(positions), k, design.validation.n_psu, design.max_memory_mb
    )

    # Preserve exact raw integer identity before float64 can round a large count.
    def raw_outcome(i):
        if isinstance(data, pd.DataFrame):
            return _scalar(data[outcome].iloc[i])
        if isinstance(data, Mapping):
            column = data[outcome]
            return _scalar(column.iloc[i] if isinstance(column, pd.Series) else column[i])
        if isinstance(data, Sequence):
            return _scalar(data[i][outcome])
        raise AnalysisError("unsupported_survey_input", "Use complete resident scalar rows.")

    count_admissible = all(
        0 <= raw_outcome(i) <= 2**53 and int(raw_outcome(i)) == raw_outcome(i) for i in positions
    )
    X = target.values[:, 1:]
    if intercept:
        X = torch.cat((target.domain[:, None], X), 1)
    labels = (["_cons"] if intercept else []) + regressors
    metadata = {
        **target.metadata,
        "outcome": outcome,
        "regressors": regressors,
        "parameter_order": labels,
        "workspace": workspace.record(),
        "raw_outcome_count_admissible": count_admissible,
        "model_exclusions": target.metadata["outcome_exclusions"],
        "calibration_support": False,
        "weight_semantics": "derived inverse probabilities N/n * M/m; pseudo-score normalized to active mean one",
        "covariance_correction": "recursive SRSWOR between-PSU and within-SSU FPC; no regression HC1 factor",
    }
    return RegressionSample(
        design,
        outcome,
        regressors,
        intercept,
        labels,
        X,
        target.values[:, 0],
        target.weights,
        selected,
        target.groups,
        target.strata_groups,
        target.fpc,
        metadata,
    )


@cpu_call
def _fit(
    data,
    design,
    outcome,
    regressors,
    *,
    family,
    intercept,
    domain,
    missing,
    alpha,
    null,
    max_iter,
    tolerance,
):
    max_iter, tolerance = _options(max_iter, tolerance)
    sample = _admit(
        data, design, outcome, regressors, intercept=intercept, domain=domain, missing=missing
    )
    alpha, null = _inference_options(sample, alpha, null)
    overhead, diagnostic, evaluation = _work_budget(sample, family, max_iter)
    cost = replay_work(
        len(sample.X),
        int(sample.selected.sum()),
        len(sample.labels),
        design.validation.n_psu,
        family,
    )
    planned = (
        overhead + diagnostic + evaluation * (1 if family == "linear" else max_iter) + cost * 2
    )
    if planned > MAX_WORK:
        raise AnalysisError(
            "survey_regression_budget",
            "Declared iterations, bounded separation and full two-stage state replay exceed 50 million work units.",
        )
    ledger = _Ledger()
    ledger.reserve(cost * 2)
    fitted = fit_sample(sample, family, max_iter=max_iter, tolerance=tolerance, budget=ledger)
    positions = sample.metadata["sample_positions"]
    replay = replay_fit(
        design,
        sample.X[sample.selected],
        sample.y[sample.selected],
        positions,
        fitted["beta"],
        family,
    )
    metadata = {
        **sample.metadata,
        "convergence": fitted["convergence"],
        "primitive_X": sample.X[sample.selected].tolist(),
        "primitive_y": sample.y[sample.selected].tolist(),
        **{
            name: replay[name].tolist()
            for name in ("bread", "row_scores", "stage1_covariance", "stage2_covariance")
        },
        "work": {
            "max_work_units": MAX_WORK,
            "planned_work_units": planned,
            "actual_work_units": ledger.used,
            "replay_work_units": cost * 2,
        },
    }
    values = {
        "schema_version": "survey-two-stage-regression-result-v1",
        "method": "two-stage-taylor",
        "family": family,
        "outcome": outcome,
        "regressors": tuple(sample.regressors),
        "intercept": intercept,
        "labels": tuple(sample.labels),
        "coefficients": tuple(fitted["beta"].tolist()),
        "covariance": tuple(tuple(row) for row in replay["covariance"].tolist()),
        "df": design.validation.design_df,
        "alpha": alpha,
        "null": tuple(null)
        if isinstance(null, (list, tuple))
        else tuple([null] * len(sample.labels)),
        "design": design.model_dump(mode="json"),
        "metadata": metadata,
    }
    values["integrity_sha256"] = digest(values)
    return SurveyTwoStageRegressionResult.model_validate(values)


def survey_two_stage_regress(
    data,
    design,
    outcome,
    regressors,
    *,
    intercept=True,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    max_iter=100,
    tolerance=1e-9,
) -> SurveyTwoStageRegressionResult:
    """Weighted linear coefficients with complete recursive two-stage SRSWOR covariance.

    Sampling weights are derived from declared stage population counts. Domain
    and missing exclusions retain zero-score physical SSUs and all nested PSUs.
    The reference df are complete first-stage PSUs minus strata.
    """
    return _fit(
        data,
        design,
        outcome,
        regressors,
        family="linear",
        intercept=intercept,
        domain=domain,
        missing=missing,
        alpha=alpha,
        null=null,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def survey_two_stage_logit(
    data,
    design,
    outcome,
    regressors,
    *,
    intercept=True,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    max_iter=100,
    tolerance=1e-9,
) -> SurveyTwoStageRegressionResult:
    """Native binary logit with both SRSWOR stage FPC terms and strict finite fitting.

    Exact 0/1 responses, full weighted rank and bounded separation certification
    are required. No penalization or ordinary-regression HC1 correction is used.
    """
    return _fit(
        data,
        design,
        outcome,
        regressors,
        family="logit",
        intercept=intercept,
        domain=domain,
        missing=missing,
        alpha=alpha,
        null=null,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def survey_two_stage_probit(
    data,
    design,
    outcome,
    regressors,
    *,
    intercept=True,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    max_iter=100,
    tolerance=1e-9,
) -> SurveyTwoStageRegressionResult:
    """Native probit using observed score sensitivity and both sampling stages.

    The stable Bernoulli score and exact observed Hessian are linearized at the
    saved fit; excluded physical SSUs retain zero scores in nested geometry.
    """
    return _fit(
        data,
        design,
        outcome,
        regressors,
        family="probit",
        intercept=intercept,
        domain=domain,
        missing=missing,
        alpha=alpha,
        null=null,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def survey_two_stage_poisson(
    data,
    design,
    outcome,
    regressors,
    *,
    intercept=True,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    max_iter=100,
    tolerance=1e-9,
) -> SurveyTwoStageRegressionResult:
    """Native count Poisson with recursive two-stage SRSWOR score covariance.

    Exact nonnegative counts at most 2^53 are supported. No offset/exposure,
    arbitrary weights, separation repair or partial nonconvergent fit is used.
    """
    return _fit(
        data,
        design,
        outcome,
        regressors,
        family="poisson",
        intercept=intercept,
        domain=domain,
        missing=missing,
        alpha=alpha,
        null=null,
        max_iter=max_iter,
        tolerance=tolerance,
    )
