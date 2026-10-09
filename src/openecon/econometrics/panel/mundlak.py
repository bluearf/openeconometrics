"""Numeric Mundlak CRE: actual-sample means and a covariance-matched joint test."""

from __future__ import annotations

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, table, wald_test
from openecon.econometrics.postest.group_state import keys
from openecon.engines.absorb import group_means
from openecon.models import ResultBundle
from openecon.resources import tensor_bytes


def fit_cre(sample):
    from .xtreg import _fit_re

    frame, spec = sample.frame, sample.spec
    if spec.categorical:
        raise AnalysisError(
            "unsupported_categorical", "CRE currently accepts numeric predictors only."
        )
    frame.workspace_plan(
        "Mundlak means and saved panel state",
        {
            "augmented_design_and_means": tensor_bytes(
                (frame.n, 2 * len(spec.predictors) + 1), itemsize=48
            ),
            "saved_group_means": tensor_bytes((sample.n, len(spec.predictors)), itemsize=40),
        },
    )
    design = frame.design()
    means = group_means(design.x[:, 1:], sample.codes, sample.n)
    within = design.x[:, 1:] - means[sample.codes]
    varying = [
        i
        for i in range(len(spec.predictors))
        if float(within[:, i].square().sum())
        > 1e-13 * max(float((design.x[:, i + 1] - design.x[:, i + 1].mean()).square().sum()), 1e-28)
    ]
    if not varying:
        raise AnalysisError(
            "no_within_variation", "Mundlak testing requires a time-varying predictor."
        )
    names = ["mean(" + spec.predictors[i] + ")" for i in varying]
    augmented = Design(
        torch.cat([design.x, means[sample.codes][:, varying]], 1), [*design.terms, *names], {}, True
    )
    if len(augmented.terms) != len(set(augmented.terms)):
        raise AnalysisError(
            "duplicate_terms", "Predictor names collide with generated Mundlak mean terms."
        )
    # Fail when the declared Mundlak test cannot identify every included mean.
    screened = frame.drop_collinear(augmented)
    if any(name not in screened.terms for name in names):
        raise AnalysisError(
            "underidentified",
            "Panel means are collinear; the declared joint Mundlak test is unidentified.",
        )
    result = _fit_re(sample, screened)
    result.title = "Correlated random-effects (Mundlak) regression"
    result.extra["model"] = result.provenance["model"] = "cre"
    terms = [c.term for c in result.coefficients]
    beta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    covariance = torch.tensor(result.covariance_matrix, dtype=torch.float64)
    result.tests["mundlak"] = wald_test(
        beta,
        covariance,
        [terms.index(name) for name in names],
        label="Joint Mundlak test: actual-sample panel means are zero",
    )
    result.tests["model"] = wald_test(
        beta,
        covariance,
        [i for i, term in enumerate(terms) if term != "Intercept" and term not in names],
        label="Wald chi2 test of original covariates (excluding Mundlak means)",
    )
    panel_keys = [
        key.hex() for key in keys(frame.sample.iloc[sample.first_rows().tolist()], [spec.panel])
    ]
    result.extra["cre_state"] = {
        "mean_terms": names,
        "mean_predictors": [spec.predictors[i] for i in varying],
        "panel_keys": panel_keys,
        "panel_means": means[:, varying].tolist(),
        "mean_scope": "actual retained estimation sample, separately for each panel",
        "prediction": "saved panel means; unknown panels rejected; no random-effect BLUP",
    }
    return result


def mundlak_test(result):
    """Return the saved joint Wald test of retained-sample Mundlak mean terms."""
    if not isinstance(result, ResultBundle) or "mundlak" not in result.tests:
        raise AnalysisError("invalid_result", "mundlak_test requires a saved CRE result.")
    return table(
        [result.tests["mundlak"]],
        title="Mundlak specification test",
        covariance=result.inference,
        mean_scope=result.extra["cre_state"]["mean_scope"],
    )


def cre_predict(result, data):
    """Population means conditional on *saved* Mundlak means; no mean recomputation."""
    from openecon.analysis import _coerce_frame, _numeric
    from openecon.dataset import Dataset
    from openecon.resources import plan_workspace

    if (
        not isinstance(result, ResultBundle)
        or result.spec.options.get("model") != "cre"
        or "cre_state" not in result.extra
    ):
        raise AnalysisError("invalid_result", "cre_predict requires a saved CRE result.")
    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported",
            "CRE prediction needs in-memory rows; Dataset is not collected.",
        )
    data = _coerce_frame(data)
    required = [result.spec.panel, *result.spec.predictors]
    if data.columns.has_duplicates or any(name not in data for name in required) or not len(data):
        raise AnalysisError(
            "invalid_predictors",
            "CRE prediction requires unique complete predictor and panel columns.",
        )
    if data[required].isna().any().any():
        raise AnalysisError("missing_values", "CRE prediction inputs must be complete.")
    plan_workspace(
        "saved Mundlak prediction",
        {"design": tensor_bytes((len(data), len(result.coefficients)), itemsize=40)},
    )
    state = result.extra["cre_state"]
    lookup = dict(zip(state["panel_keys"], state["panel_means"], strict=True))
    query_keys = [key.hex() for key in keys(data, [result.spec.panel])]
    if any(key not in lookup for key in query_keys):
        raise AnalysisError(
            "unknown_group", "CRE prediction cannot supply fitted Mundlak means for a new panel."
        )
    means = torch.tensor([lookup[key] for key in query_keys], dtype=torch.float64)
    columns = {name: _numeric(data[name], name) for name in result.spec.predictors}
    columns.update({name: means[:, i] for i, name in enumerate(state["mean_terms"])})
    columns["Intercept"] = torch.ones(len(data), dtype=torch.float64)
    predicted = sum(columns[c.term] * c.estimate for c in result.coefficients)
    if not bool(torch.isfinite(predicted).all()):
        raise AnalysisError(
            "non_finite_prediction", "CRE prediction exceeds finite float64 arithmetic."
        )
    return pd.Series(predicted.tolist(), index=data.index, name="predicted")
