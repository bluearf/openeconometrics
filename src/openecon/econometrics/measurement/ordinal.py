"""Two-step ordered latent-normal pair correlations with subject refit uncertainty."""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.discrete.bivariate import bvn_cdf
from openecon.econometrics.multivariate import common as c
from openecon.engines import distributions as dist
from . import common as m


def _thresholds(counts):
    if bool((counts <= 0).any()):
        raise AnalysisError(
            "empty_category",
            "Every declared ordinal level must be observed; empty categories are not silently removed.",
        )
    probabilities = counts.cumsum(0)[:-1] / counts.sum()
    return torch.tensor([dist.normal_ppf(float(x)) for x in probabilities], dtype=c.FLOAT)


def _minimize(objective, bound, iterations, tolerance):
    grid = torch.linspace(-bound, bound, 33, dtype=c.FLOAT).tolist()
    values = [objective(x) for x in grid]
    best = min(range(len(grid)), key=values.__getitem__)
    candidates = [
        i for i in range(1, len(grid) - 1) if values[i] <= min(values[i - 1], values[i + 1])
    ]
    if best in (0, len(grid) - 1):
        candidates.append(best)
    roots = []
    phi = (math.sqrt(5) - 1) / 2
    for i in candidates:
        left, right = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
        a, b = right - phi * (right - left), left + phi * (right - left)
        fa, fb = objective(a), objective(b)
        for step in range(iterations):
            if right - left <= tolerance:
                break
            if fa < fb:
                right, b, fb = b, a, fa
                a = right - phi * (right - left)
                fa = objective(a)
            else:
                left, a, fa = a, b, fb
                b = left + phi * (right - left)
                fb = objective(b)
        else:
            raise AnalysisError(
                "no_convergence",
                "Latent correlation search exceeded max_iterations before tolerance.",
            )
        root = (left + right) / 2
        roots.append((objective(root), root, step + 1, right - left))
    if not roots:
        raise AnalysisError(
            "numerical_failure", "No finite interior latent correlation minimum was bracketed."
        )
    value, root, steps, width = min(roots)
    if not math.isfinite(value):
        raise AnalysisError(
            "numerical_failure", "Occupied ordinal probabilities could not be evaluated reliably."
        )
    if abs(root) >= bound - tolerance * 4:
        raise AnalysisError(
            "correlation_boundary", "Latent correlation optimum reached its declared bound."
        )
    return root, dict(
        negative_log_likelihood=value,
        iterations=steps,
        bracket_width=width,
        max_correlation=bound,
        tolerance=tolerance,
        candidate_minima=len(roots),
        evaluations_lower_bound=33 + steps,
    )


def _search_options(bound, iterations, tolerance):
    c.check_number(bound, "max_correlation", minimum=0.1, maximum=0.995)
    c.check_count(iterations, "max_iterations", minimum=20, maximum=300)
    c.check_number(tolerance, "tolerance", minimum=1e-12, maximum=1e-3)


def _rectangle(tx, ty, rho):
    # Native Genz CDF with exact normal marginal boundary values. No smoothing of
    # the observed table; an unresolvable occupied cell makes the fit fail.
    x = torch.cat((torch.tensor([-40.0], dtype=c.FLOAT), tx, torch.tensor([40.0], dtype=c.FLOAT)))
    y = torch.cat((torch.tensor([-40.0], dtype=c.FLOAT), ty, torch.tensor([40.0], dtype=c.FLOAT)))
    a, b = torch.meshgrid(x, y, indexing="ij")
    cumulative = bvn_cdf(a.flatten(), b.flatten(), rho).reshape(len(x), len(y))
    cumulative[0, :] = 0
    cumulative[:, 0] = 0
    cumulative[-1, :] = torch.special.ndtr(y)
    cumulative[:, -1] = torch.special.ndtr(x)
    return cumulative[1:, 1:] - cumulative[:-1, 1:] - cumulative[1:, :-1] + cumulative[:-1, :-1]


@c.procedure
def polychoric(
    data: Any,
    columns: list[str],
    *,
    categories: list[list],
    max_correlation: float = 0.98,
    max_iterations: int = 80,
    tolerance: float = 1e-9,
    inference: str = "none",
    level: float = 0.95,
    missing: str = "drop",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """Two-step polychoric correlation of exactly two ordered categorical variables.

    Each marginal threshold is the normal quantile of cumulative observed counts;
    rho minimizes the multinomial rectangle likelihood within the declared bound.
    Explicit category orders, no smoothing/empty-level removal. Assumes bivariate
    latent normal thresholds. The full subject jackknife re-estimates thresholds,
    unlike a fixed-threshold Hessian. Genz bivariate probabilities are native Torch.
    """
    m.options(inference, level, max_fits, max_work, device, weights)
    _search_options(max_correlation, max_iterations, tolerance)
    names = c.name_list(columns, "columns", minimum=2)
    if len(names) != 2 or not isinstance(categories, (list, tuple)) or len(categories) != 2:
        raise AnalysisError(
            "invalid_spec", "Polychoric requires two columns and two declared category lists."
        )
    cx, cy = m.levels(categories[0]), m.levels(categories[1])
    selected, metadata = m.sample(
        data,
        names,
        missing=missing,
        q=max(len(cx), len(cy)),
        inference=inference,
        max_fits=max_fits,
        max_work=max_work,
        fit_cost=(33 + 33 * (max_iterations + 2)) * 20 * len(cx) * len(cy),
    )

    def fit(frame):
        code_x, code_y = (
            m.codes(frame[[names[0]]], cx).flatten(),
            m.codes(frame[[names[1]]], cy).flatten(),
        )
        count = (
            torch.bincount(code_x * len(cy) + code_y, minlength=len(cx) * len(cy))
            .reshape(len(cx), len(cy))
            .to(c.FLOAT)
        )
        tx, ty = _thresholds(count.sum(1)), _thresholds(count.sum(0))
        occupied = count > 0

        def objective(rho):
            probability = _rectangle(tx, ty, rho)
            if (
                bool((probability[occupied] <= 0).any())
                or abs(float(probability.sum()) - 1) > 1e-10
                or float(probability.min()) < -1e-12
            ):
                return float("inf")
            return -float((count[occupied] * probability[occupied].log()).sum())

        rho, details = _minimize(objective, max_correlation, max_iterations, tolerance)
        return (
            rho,
            {
                "counts": c.frame(count, columns=cy, index=cx),
                "probabilities": c.frame(_rectangle(tx, ty, rho), columns=cy, index=cx),
                "thresholds_x": c.frame(tx[:, None], columns=["threshold"], index=cx[:-1]),
                "thresholds_y": c.frame(ty[:, None], columns=["threshold"], index=cy[:-1]),
            },
            {**details, "method": "two-step marginal thresholds; interior multinomial rho fit"},
        )

    return m.result(
        "polychoric",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(
            categories=[cx, cy],
            max_correlation=max_correlation,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
    )


@c.procedure
def polyserial(
    data: Any,
    continuous: str,
    ordinal: str,
    *,
    categories: list,
    max_correlation: float = 0.98,
    max_iterations: int = 80,
    tolerance: float = 1e-9,
    inference: str = "none",
    level: float = 0.95,
    missing: str = "drop",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """Two-step polyserial correlation with one continuous and one ordinal variable.

    Continuous observations are standardized with sample (n-1) SD, as polycor;
    marginal ordinal thresholds are estimated from counts. Conditional normal
    probabilities optimize rho on the declared interior domain. Full subject
    jackknife refits continuous mean/SD and every ordinal threshold. No joint ML.
    """
    m.options(inference, level, max_fits, max_work, device, weights)
    _search_options(max_correlation, max_iterations, tolerance)
    names = c.name_list([continuous, ordinal], "columns", minimum=2)
    cats = m.levels(categories)
    selected, metadata = m.sample(
        data,
        names,
        numeric=[continuous],
        missing=missing,
        q=len(cats),
        inference=inference,
        max_fits=max_fits,
        max_work=max_work,
        fit_cost=33 + 33 * (max_iterations + 2),
    )

    def fit(frame):
        x = c.column(frame, continuous)
        mean = float(x.mean())
        sd = float(x.std(correction=1))
        if not sd > 1e-12 * max(float(x.abs().max()), 1e-300):
            raise AnalysisError(
                "constant_column", "Continuous polyserial variable must have nonzero variance."
            )
        z = (x - mean) / sd
        code = m.codes(frame[[ordinal]], cats).flatten()
        cuts = _thresholds(torch.bincount(code, minlength=len(cats)).to(c.FLOAT))
        cuts = torch.cat(
            (
                torch.tensor([-float("inf")], dtype=c.FLOAT),
                cuts,
                torch.tensor([float("inf")], dtype=c.FLOAT),
            )
        )

        def objective(rho):
            scale = math.sqrt(1 - rho * rho)
            lower, upper = (cuts[code] - rho * z) / scale, (cuts[code + 1] - rho * z) / scale
            # Compute survival differences in the right tail, CDF differences elsewhere.
            probability = torch.where(
                lower > 0,
                torch.special.ndtr(-lower) - torch.special.ndtr(-upper),
                torch.special.ndtr(upper) - torch.special.ndtr(lower),
            )
            if bool((probability <= 0).any()):
                return float("inf")
            return -float(probability.log().sum())

        rho, details = _minimize(objective, max_correlation, max_iterations, tolerance)
        return (
            rho,
            {"thresholds": c.frame(cuts[1:-1, None], columns=["threshold"], index=cats[:-1])},
            {
                **details,
                "continuous_mean": mean,
                "continuous_sample_sd": sd,
                "method": "two-step sample-SD continuous standardization; conditional-normal rho fit",
                "likelihood": "conditional ordinal likelihood; rho-independent continuous density omitted",
            },
        )

    return m.result(
        "polyserial",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(
            categories=cats,
            continuous=continuous,
            ordinal=ordinal,
            max_correlation=max_correlation,
            max_iterations=max_iterations,
            tolerance=tolerance,
        ),
    )
