"""Design-linearized means, totals, ratios and category proportions."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from .common import (
    FLOAT,
    MAX_DEFF_WORK,
    SRSWR_WEIGHTED_REFERENCE,
    SurveyResult,
    finite,
    prepare,
    result,
)


def taylor(target, *, alpha, null, deff):
    if type(deff) is not bool:
        raise AnalysisError("invalid_survey_option", "deff must be Boolean.")
    estimate, influence = target.evaluate(target.weights)
    weighted = finite(target.weights[:, None] * influence, "Weighted PSU influence")
    aggregated = torch.zeros((target.design.validation.n_psu, len(target.labels)), dtype=FLOAT)
    aggregated.index_add_(0, target.groups, weighted)
    covariance = torch.zeros((len(target.labels), len(target.labels)), dtype=FLOAT)
    for psus, correction in zip(target.strata_groups, target.fpc):
        if not correction:
            continue
        block = aggregated[psus]
        centred = block - block.mean(0)
        covariance += correction * len(psus) / (len(psus) - 1) * (centred.T @ centred)
    metadata = {
        "variance_formula": "sum_h (1-f_h) m_h/(m_h-1) sum_j centered weighted PSU influences outer products",
        "fpc_multipliers": target.fpc,
        "psu_influence_sums": aggregated.tolist(),
        "stratum_psu_indices": target.strata_groups,
        "confidence_interval": "unclipped design-t Wald interval; census point interval, undefined zero-SE tests",
    }
    if deff:
        n = len(target.weights)
        if n < 2:
            raise AnalysisError(
                "unsupported_survey_deff", "SRSWR reference needs at least two design rows."
            )
        if bool((target.weights == target.weights[0]).all()):
            # Preserve the accepted equal-weight result, including its metadata.
            centred = weighted - weighted.mean(0)
            reference = finite(n / (n - 1) * (centred.T @ centred), "SRSWR covariance")
            reference_name = "equal-weight independent row PSUs, full population/domain geometry, with replacement, no FPC"
        else:
            width = len(target.labels)
            if n * width * (width + 2) > MAX_DEFF_WORK:
                raise AnalysisError(
                    "survey_budget",
                    "Unequal-weight SRSWR reference exceeds 50 million full-design moment work units.",
                )
            population = finite(target.weights.sum(), "SRSWR population weight sum")
            mean = finite(weighted.sum(0) / population, "SRSWR weighted influence mean")
            centred = finite(influence - mean, "SRSWR centered row influences")
            moments = finite(
                (centred.T * target.weights) @ centred,
                "SRSWR weighted centered crossproducts",
            )
            reference = finite(population / (n - 1) * moments, "SRSWR covariance")
            positive = reference.diagonal() > 0
            finite(
                covariance.diagonal()[positive] / reference.diagonal()[positive],
                "SRSWR design-effect ratios",
            )
            reference_name = SRSWR_WEIGHTED_REFERENCE
            metadata["srs_reference_state"] = {
                "schema_version": "survey-srswr-weighted-v1",
                "law": "full-design-weighted-population-srswr-linearized",
                "n_design": n,
                "sum_design_weights": float(population),
                "weighted_influence_mean": mean.tolist(),
                "weighted_centered_crossproducts": moments.tolist(),
                "fpc_applied": False,
                "eligibility": "fixed-domain-and-joint-complete-case-indicator",
            }
        metadata.update(
            srs_reference=reference_name,
            srs_covariance=reference.tolist(),
            design_effect=[
                float(covariance[i, i] / reference[i, i]) if reference[i, i] > 0 else None
                for i in range(len(target.labels))
            ],
        )
    return result(target, "taylor", estimate, covariance, alpha=alpha, null=null, metadata=metadata)


def survey_mean(
    data, design, outcomes, *, domain=None, missing="raise", alpha=0.05, null=0.0, deff=False
) -> SurveyResult:
    """Joint Hájek means with complete-design Taylor covariance and design-t inference."""
    return taylor(
        prepare(data, design, "mean", outcomes, domain=domain, missing=missing),
        alpha=alpha,
        null=null,
        deff=deff,
    )


def survey_total(
    data, design, outcomes, *, domain=None, missing="raise", alpha=0.05, null=0.0, deff=False
) -> SurveyResult:
    """Unnormalized weighted totals; full joint Taylor covariance retains zero-domain PSUs."""
    return taylor(
        prepare(data, design, "total", outcomes, domain=domain, missing=missing),
        alpha=alpha,
        null=null,
        deff=deff,
    )


def survey_ratio(
    data,
    design,
    numerators,
    denominators,
    *,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    deff=False,
) -> SurveyResult:
    """Paired weighted-total ratios with full joint delta/score covariance."""
    return taylor(
        prepare(
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
        deff=deff,
    )


def survey_proportion(
    data,
    design,
    outcome,
    *,
    categories=None,
    domain=None,
    missing="raise",
    alpha=0.05,
    null=0.0,
    deff=False,
) -> SurveyResult:
    """Declared category proportions, including absent levels and singular joint covariance."""
    return taylor(
        prepare(
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
        deff=deff,
    )
