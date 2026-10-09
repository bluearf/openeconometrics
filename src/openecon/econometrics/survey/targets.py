"""Design-linearized means, totals, ratios and category proportions."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from .common import FLOAT, SurveyResult, finite, prepare, result


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
        if not bool((target.weights == target.weights[0]).all()):
            raise AnalysisError(
                "unsupported_survey_deff",
                "SRSWR design effects currently require equal sampling weights.",
            )
        n = len(target.weights)
        if n < 2:
            raise AnalysisError(
                "unsupported_survey_deff", "SRSWR reference needs at least two design rows."
            )
        centred = weighted - weighted.mean(0)
        reference = finite(n / (n - 1) * (centred.T @ centred), "SRSWR covariance")
        metadata.update(
            srs_reference="equal-weight independent row PSUs, full population/domain geometry, with replacement, no FPC",
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
