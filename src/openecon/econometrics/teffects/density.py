"""Cattaneo--Jansson--Ma local polynomial density discontinuity test.

The validated domain uses explicit bandwidths, unrestricted left/right local
polynomials, and the jackknife covariance. It estimates the empirical CDF,
not a regression outcome, and retains the left/right covariance.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, make_spec, table, kernel_call
from openecon.econometrics.teffects.rd import _pair
from openecon.econometrics.teffects.rdkernels import kernel_weights, powers
from openecon.engines.linalg import least_squares
from openecon.engines.distributions import normal_sf


def rddensity(
    *,
    data: Any,
    running: str,
    h: float | Sequence[float],
    cutoff: float = 0.0,
    p: int = 2,
    q: int | None = None,
    kernel: str = "triangular",
    masspoints: bool = True,
    missing: str = "raise",
) -> TableSet:
    """Manipulation test at a cutoff with explicit bandwidths and jackknife VCE.

    The order-p estimate is conventional; inference uses order q (default
    p+1), as in community rddensity. Automatic density bandwidth selection,
    restricted fits and plug-in VCE are outside this domain. Mass points use
    tied empirical ranks and shared leave-one-out scores. No outcome or RD
    treatment effect is inferred by this diagnostic.
    """
    q = p + 1 if q is None else q
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (p, q)) or not 1 <= p < q <= 8:
        raise AnalysisError("invalid_option", "rddensity requires integer 1 <= p < q <= 8.")
    if kernel not in ("triangular", "uniform", "epanechnikov") or not isinstance(masspoints, bool):
        raise AnalysisError("invalid_option", "Use a supported kernel and boolean masspoints.")
    if (
        not isinstance(cutoff, (int, float))
        or isinstance(cutoff, bool)
        or not math.isfinite(cutoff)
    ):
        raise AnalysisError("invalid_cutoff", "cutoff must be finite.")
    bands = _pair(list(h) if isinstance(h, (list, tuple)) else h, "h")
    if bands is None:
        raise AnalysisError("invalid_option", "rddensity requires explicit h bandwidths.")
    # Use the shared running-variable/missing-data contract without fitting RD.
    frame = ModelFrame(
        make_spec("rdrobust", outcome=running, columns={"running": running}, missing=missing), data
    )
    x = frame.numeric(running).sort().values - cutoff
    if len(x) < 2 or not bool((x < 0).any()) or not bool((x >= 0).any()):
        raise AnalysisError("invalid_cutoff", "The cutoff must leave observations on both sides.")
    unique, blocks, counts = torch.unique_consecutive(x, return_inverse=True, return_counts=True)
    rank = torch.arange(len(x), dtype=torch.float64)
    if masspoints:
        rank = (counts.cumsum(0) - 1)[blocks].to(torch.float64)
    cdf = rank / (len(x) - 1)
    mask = (x >= -bands[0]) & (x <= bands[1])
    local = x[mask]
    targets = cdf[mask]
    left = local < 0
    right = ~left
    local_unique, local_blocks, local_counts = torch.unique_consecutive(
        local, return_inverse=True, return_counts=True
    )
    first = local_counts.cumsum(0) - local_counts

    def estimate(order):
        dual = torch.zeros((len(local), 2), dtype=torch.float64)
        n_effective = []
        for side, part in enumerate((left, right)):
            dx = local[part]
            weights = kernel_weights(dx, bands[side], kernel)
            if int(torch.unique(dx[weights > 0]).numel()) <= order:
                raise AnalysisError(
                    "insufficient_observations",
                    "Each density bandwidth needs more distinct positive-weight values than the polynomial order.",
                )
            design = powers(dx / bands[side], order)
            fit = kernel_call(
                least_squares, design, targets[part], weights, drop_collinear=False, tol=0.0
            )
            dual[part, side] = (design @ fit.xtx_inv[:, 1]) * weights / bands[side]
            n_effective.append(int((weights > 0).sum()))
        value = dual.T @ targets
        # Leave-one-out CDF score: sum of kernel design rows strictly after
        # each sorted local observation. Tied values share the first score.
        tails = dual.flip(0).cumsum(0).flip(0) - dual
        if masspoints:
            tails = tails[first[local_blocks]]
        scores = tails / (len(x) - 1)
        covariance = scores.T @ scores
        difference = float(value[1] - value[0])
        variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
        if variance <= 0:
            raise AnalysisError(
                "invalid_covariance", "Density difference jackknife variance must be positive."
            )
        statistic = difference / math.sqrt(variance)
        return (
            {
                "left": float(value[0]),
                "right": float(value[1]),
                "difference": difference,
                "std_error": math.sqrt(variance),
                "statistic": statistic,
                "p_value": 2 * normal_sf(abs(statistic)),
                "order": order,
            },
            covariance.tolist(),
            n_effective,
        )

    conventional, _, _ = estimate(p)
    robust, covariance, effective = estimate(q)
    return TableSet(
        {
            "density": table(
                [{"method": "Conventional", **conventional}, {"method": "Robust", **robust}]
            )
        },
        procedure="rddensity",
        p=p,
        q=q,
        h=list(bands),
        cutoff=cutoff,
        kernel=kernel,
        covariance=covariance,
        nobs=len(x),
        n_left=int((x < 0).sum()),
        n_right=int((x >= 0).sum()),
        n_effective=effective,
        masspoints=masspoints,
        n_unique=len(unique),
        inference="jackknife normal",
        fitselect="unrestricted",
        bandwidth_selection="explicit",
        source="Cattaneo, Jansson and Ma (2020), Simple Local Polynomial Density Estimators",
        source_url="https://rdpackages.github.io/rddensity/",
    )
