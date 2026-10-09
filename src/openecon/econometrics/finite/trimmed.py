"""Complete h-subset enumeration for bounded small-sample LTS; descriptive prediction."""

from __future__ import annotations

import itertools
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from . import common as c


@resident_cpu
def lts(
    data,
    y,
    x=None,
    *,
    h=None,
    intercept=True,
    missing="raise",
    max_subsets=20000,
    max_work=100_000_000,
    device="cpu",
    weights=None,
):
    """Enumerate EVERY full-rank h-subset QR fit and minimize computed trimmed SSE.

    <=24 resident input rows, <=3 coefficients, <=20000 h-subsets; h defaults
    to floor((n+p+1)/2), with h>=p+1. Any singular/ill-conditioned candidate
    refuses the whole fit. All objectives/subsets are retained; no sampling
    inference is fabricated from the selected residuals.
    """
    c.domain(device, weights)
    max_subsets = c.c.check_count(max_subsets, "max_subsets", maximum=20000)
    names = c.c.name_list(x, "x", minimum=0)
    c.c.check_flag(intercept, "intercept")
    p = len(names) + int(intercept)
    # Inspect resident row count/missing policy before materializing all subsets.
    frame, _ = c.sample(data, [y] + names, missing, 24)
    n = len(frame)
    if not 1 <= p <= 3:
        raise AnalysisError(
            "resource_limit", "LTS needs 1..3 numeric coefficients including any intercept."
        )
    h = (n + p + 1) // 2 if h is None else c.c.check_count(h, "h")
    if not p + 1 <= h <= n:
        raise AnalysisError("invalid_option", "LTS requires p+1 <= h <= n.")
    subsets = math.comb(n, h)
    if subsets > max_subsets:
        raise AnalysisError(
            "resource_limit",
            f"Complete LTS requires {subsets} subsets, exceeding max_subsets={max_subsets}.",
        )
    matrix, response, transform, state, settings = c.design(
        data, y, names, intercept, missing, 24, 3, "lts", 32 * subsets, max_work, records=subsets
    )
    if float(response.abs().max()) > 1e8:
        raise AnalysisError(
            "non_finite_values", "LTS response must lie within +/-1e8; rescale explicitly."
        )
    origin = float(response.mean()) if intercept else 0.0
    centered = response - origin
    candidates, best, best_theta, best_subset = [], math.inf, None, None
    for indices in itertools.combinations(range(n), h):
        idx = torch.tensor(indices, dtype=torch.int64)
        design, values = matrix[idx], centered[idx]
        cond = c.condition(design)
        q, r = torch.linalg.qr(design, mode="reduced")
        theta = torch.linalg.solve_triangular(r, (q.T @ values)[:, None], upper=True).flatten()
        residual = values - design @ theta
        sse = float(residual @ residual)
        candidates.append(
            {
                "sample_indices": list(indices),
                "positions": [state["sample_positions"][i] for i in indices],
                "sse": sse,
                "condition": cond,
            }
        )
        if sse < best:
            best, best_theta, best_subset = sse, theta, list(indices)
    residual = centered - matrix @ best_theta
    order = sorted(range(n), key=lambda i: (float(residual[i] ** 2), i))
    selected = order[:h]
    trimmed_sse = float((residual[selected] ** 2).sum())
    if abs(trimmed_sse - best) > 1e-8 * max(1.0, best) or not math.isfinite(best):
        raise AnalysisError(
            "numerical_failure", "Exhaustive LTS objective failed complete-enumeration consistency."
        )
    if intercept:
        best_theta[0] += origin
    raw = transform @ best_theta
    state.update(
        h=h,
        subset_count=subsets,
        max_subsets=max_subsets,
        candidates=candidates,
        best_subset=best_subset,
        trimmed_sample_indices=selected,
        parameters_scaled=best_theta.tolist(),
        parameters=raw.tolist(),
        trimmed_sse=trimmed_sse,
        best_subset_sse=best,
        settings=settings,
        solver="complete h-subset normalized float64 QR; computed global minimum in declared full-rank domain",
        inference="descriptive only; no selected-model OLS covariance or CI",
    )
    return c.seal(
        "lts",
        {
            "coefficients": table(
                [[t, float(b)] for t, b in zip(state["terms"], raw)], columns=["term", "estimate"]
            ),
            "summary": table(
                [[n, p, h, subsets, trimmed_sse, math.sqrt(trimmed_sse / h)]],
                columns=[
                    "n",
                    "p",
                    "h",
                    "enumerated_subsets",
                    "trimmed_sse",
                    "descriptive_trimmed_rms",
                ],
            ),
            "residuals": table(
                [
                    [pos, lab, float(v), float(v - r), float(r), i in selected]
                    for i, (pos, lab, v, r) in enumerate(
                        zip(state["sample_positions"], state["sample_labels"], response, residual)
                    )
                ],
                columns=["position", "label", "observed", "fitted", "residual", "trimmed_inlier"],
            ),
            "candidates": table(
                [
                    [
                        i,
                        json_indices(v["sample_indices"]),
                        json_indices(v["positions"]),
                        v["sse"],
                        v["condition"],
                    ]
                    for i, v in enumerate(candidates)
                ],
                columns=[
                    "candidate",
                    "sample_indices",
                    "source_positions",
                    "subset_sse",
                    "condition",
                ],
            ),
        },
        state,
        inference="descriptive exhaustive small-sample LTS; no sampling covariance",
        **settings,
    )


def json_indices(values):
    return c.canonical(values)


@resident_cpu
def lts_predict(result, data, *, missing="raise", max_work=100_000_000, device="cpu", weights=None):
    """Predict saved LTS coefficients descriptively; retain full fit and new-row identities."""
    c.domain(device, weights)
    state = c.intact(result, "lts")
    matrix, sample, settings = c.prediction(data, state, missing, "lts_predict", max_work)
    predicted = matrix @ torch.tensor(state["parameters_scaled"], dtype=torch.float64)
    if not bool(torch.isfinite(predicted).all()):
        raise AnalysisError("numerical_failure", "Saved LTS prediction is nonfinite.")
    return c.seal(
        "lts_predict",
        {
            "predictions": table(
                [
                    [pos, lab, float(v)]
                    for pos, lab, v in zip(
                        sample["sample_positions"], sample["sample_labels"], predicted
                    )
                ],
                columns=["position", "label", "predicted"],
            ),
        },
        {
            "fit": state,
            "prediction_sample": sample,
            "design_scaled": matrix.tolist(),
            "settings": settings,
        },
        inference="descriptive saved LTS mean; no SE/CI",
        **settings,
    )
