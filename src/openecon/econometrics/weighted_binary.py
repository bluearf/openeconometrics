"""Direct weighted logit/probit on the accepted resident Bernoulli ML engine.

The unweighted legacy solver and replayable Dataset route stay independent.
These admission bounds describe implementation resources, not identification.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.resident_cpu import resident_cpu

MAX_ROWS = 100_000
MAX_PARAMETERS = 64
MAX_ITERATIONS = 100
MAX_WORK = 5_000_000_000


@resident_cpu
def fit_weighted_binary(spec, data):
    from openecon.analysis import _check_binary_separation, _coerce_frame
    from openecon.dataset import Dataset
    from openecon.econometrics.core import ModelFrame
    from openecon.econometrics.glm.commands import _report
    from openecon.econometrics.glm.common import likelihood_weights, require_terms
    from openecon.econometrics.glm.families import Binomial, make_link
    from openecon.econometrics.glm.glm import binomial_outcome, estimate

    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported",
            "Weighted binary models require a resident "
            "table; the unweighted Dataset route does not accept weights.",
        )
    data = _coerce_frame(data)
    if len(data) > MAX_ROWS:
        raise AnalysisError(
            "weighted_binary_row_limit",
            "Weighted binary original input exceeds "
            "the 100,000-row resident resource bound; no rows were trimmed.",
        )
    frame = ModelFrame(spec, data)
    raw = frame.weights()
    if bool((raw <= 0).any()):
        raise AnalysisError(
            "negative_weights", "Retained binary likelihood weights must be positive."
        )
    try:
        total = math.fsum(raw.tolist())
    except OverflowError as exc:
        raise AnalysisError(
            "weight_precision_unsupported",
            "The weight sum overflows float64; rescale supplied weights.",
        ) from exc
    if (
        not math.isfinite(total)
        or not total > 0
        or (spec.weight_type == "fweight" and total > 2**53 - 1)
    ):
        raise AnalysisError(
            "weight_precision_unsupported",
            "The weight sum must be finite and "
            "frequency totals must be exactly representable in float64.",
        )
    width = int(spec.intercept)
    for name in spec.predictors:
        if name not in spec.categorical:
            width += 1
        else:
            column = frame.original[name]
            width += max(
                0,
                (
                    len(column.cat.categories)
                    if isinstance(column.dtype, pd.CategoricalDtype)
                    else column.nunique()
                )
                - 1,
            )
    work = MAX_ITERATIONS * (frame.n * width * width + width**3)
    if width > MAX_PARAMETERS or work > MAX_WORK:
        raise AnalysisError(
            "weighted_binary_work_limit",
            "Weighted binary expanded parameters "
            "or 100-iteration dense work exceed the 64-parameter/5e9 resource bound.",
        )
    frame.workspace_plan(
        "weighted binary likelihood",
        {
            "design_and_score_copies": 8 * frame.n * width * 6,
            "likelihood_vectors": 8 * frame.n * 32,
            "information_and_optimizer": 8 * width * width * 16,
        },
    )
    weights = likelihood_weights(frame)
    if not bool(torch.isfinite(weights.user).all()) or not bool((weights.user > 0).all()):
        raise AnalysisError(
            "weight_precision_unsupported",
            "Weight normalization is outside "
            "the positive finite float64 domain; rescale supplied weights.",
        )
    y = binomial_outcome(frame, frame.numeric(spec.outcome), None, spec.estimator)
    design = frame.design()
    require_terms(design)
    screened = frame.drop_collinear(design, weights.for_screen())
    if len(screened.terms) != len(design.terms):
        raise AnalysisError(
            "rank_deficient",
            "Weighted binary design is rank deficient; "
            "remove redundant predictors or unused categorical levels.",
        )
    _check_binary_separation(y, design.x, intercept=spec.intercept)
    fitted = estimate(
        frame,
        design,
        y,
        Binomial(),
        make_link(spec.estimator),
        weights=weights,
        offset=None,
        command=spec.estimator,
        max_iterations=MAX_ITERATIONS,
    )
    result = _report(
        frame,
        fitted,
        y,
        None,
        command=spec.estimator,
        extra={
            "weighted_binary_contract": "resident-ml-v1",
            "weight_semantics": {
                "fweight": "exact row frequencies",
                "aweight": "normalized to retained physical row count",
                "iweight": "supplied positive likelihood weights",
                "pweight": "supplied positive sampling pseudo-likelihood weights",
            }[spec.weight_type],
            "physical_rows": frame.n,
            "weight_sum": float(weights.user.sum()),
            "resource_limits": {
                "original_rows": MAX_ROWS,
                "parameters": MAX_PARAMETERS,
                "iterations": MAX_ITERATIONS,
                "work": MAX_WORK,
            },
            "admitted_work": work,
        },
    )
    result.provenance["separation_diagnostic"] = (
        "native Torch primal-dual constraint generation with global replay"
    )
    return result
