"""Balanced complete ANOVA ICC and explicitly one-factor model omega-total."""

from __future__ import annotations

from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c, extraction as ex
from . import common as m


@c.procedure
def icc(
    data: Any,
    raters: list[str],
    *,
    model: str = "ICC2",
    average: bool = False,
    inference: str = "none",
    level: float = 0.95,
    missing: str = "drop",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """Shrout-Fleiss ICC1/ICC2/ICC3, single or average ratings, balanced complete units.

    ICC1: one-way random absolute agreement. ICC2: two-way random absolute
    agreement. ICC3: two-way fixed consistency. Subject rows are independent,
    and each column is a rater; missing='drop' removes an entire incomplete unit.
    Negative estimates are retained. Optional delete-one subject t inference
    is approximate and does not implement the classical exact F intervals.
    """
    m.options(inference, level, max_fits, max_work, device, weights)
    c.check_choice(model, "model", ("ICC1", "ICC2", "ICC3"))
    c.check_flag(average, "average")
    names = c.name_list(raters, "raters", minimum=2)
    selected, metadata = m.sample(
        data,
        names,
        numeric=names,
        missing=missing,
        inference=inference,
        max_fits=max_fits,
        max_work=max_work,
    )

    def fit(frame):
        x = c.matrix(frame, names)
        n, k = x.shape
        x = x - x.mean()
        scale = float(x.abs().max())
        if not scale > 0:
            raise AnalysisError("undefined_coefficient", "All ratings are identical.")
        x /= scale
        row, col = x.mean(1), x.mean(0)
        msb = float(k * row.square().sum() / (n - 1))
        msj = float(n * col.square().sum() / (k - 1))
        msw = float((x - row[:, None]).square().sum() / (n * (k - 1)))
        mse = float((x - row[:, None] - col[None, :]).square().sum() / ((n - 1) * (k - 1)))
        if model == "ICC1":
            numerator, denominator = msb - msw, msb if average else msb + (k - 1) * msw
        elif model == "ICC2":
            numerator = msb - mse
            denominator = (
                msb + (msj - mse) / n if average else msb + (k - 1) * mse + k * (msj - mse) / n
            )
        else:
            numerator, denominator = msb - mse, msb if average else msb + (k - 1) * mse
        rows = [[msb, n - 1], [msj, k - 1], [msw, n * (k - 1)], [mse, (n - 1) * (k - 1)]]
        return (
            m.ratio(numerator, denominator),
            {
                "mean_squares": c.frame(
                    rows,
                    columns=["normalized_mean_square", "df"],
                    index=["subjects", "raters", "within", "error"],
                ),
            },
            dict(
                normalization_scale=scale,
                normalization="ratings centered and divided by scale before mean squares",
                generalize_raters=model != "ICC3",
                agreement="consistency" if model == "ICC3" else "absolute",
            ),
        )

    return m.result(
        "icc",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(model=model, average=average, raters=names),
    )


@c.procedure
def omega_total(
    data: Any,
    items: list[str],
    *,
    standardized: bool = False,
    reverse: list[str] | None = None,
    max_iterations: int = 1000,
    inference: str = "none",
    level: float = 0.95,
    missing: str = "drop",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """One-factor congeneric McDonald omega-total using native ML factor extraction.

    Omega = (sum(loadings))² / ((sum(loadings))² + sum(uniqueness)); raw items
    rescale standardized loadings and unique variances into original item units.
    No automatic item reversal. Reports the implied covariance and fit discrepancy;
    this is a model-based variance ratio, not empirical denominator/hierarchical omega.
    Unidentified/Heywood solutions fail. Optional subject jackknife refits all ML
    nuisance parameters; no item/row subsampling or fixed-loading uncertainty.
    """
    m.options(inference, level, max_fits, max_work, device, weights)
    c.check_flag(standardized, "standardized")
    iterations = c.check_count(max_iterations, "max_iterations", maximum=10000)
    names = c.name_list(items, "items", minimum=3)
    flipped = c.name_list(reverse, "reverse", minimum=0)
    if any(name not in names for name in flipped):
        raise AnalysisError("invalid_spec", "Every reverse item must be in items.")
    selected, metadata = m.sample(
        data,
        names,
        numeric=names,
        missing=missing,
        inference=inference,
        max_fits=max_fits,
        max_work=max_work,
        fit_cost=iterations * len(names) ** 3,
    )

    def fit(frame):
        x = c.matrix(frame, names)
        if flipped:
            x *= torch.tensor([-1 if name in flipped else 1 for name in names], dtype=c.FLOAT)
        _, sscp, _ = c.moments(x, names)
        r = c.correlation(sscp)
        smc = ex.squared_multiple_correlations(r)
        extracted = ex.maximum_likelihood(r, smc, 1, max_iter=iterations)
        if extracted.heywood:
            raise AnalysisError(
                "heywood_case",
                "One-factor omega requires an interior identified ML solution; uniqueness reached its lower bound.",
            )
        loading = c.fix_signs(extracted.loadings).flatten()
        std = (
            torch.ones(len(names), dtype=c.FLOAT)
            if standardized
            else (sscp.diagonal() / (len(frame) - 1)).sqrt()
        )
        loading, unique = loading * std, extracted.uniqueness * std.square()
        common, specific = float(loading.sum().square()), float(unique.sum())
        implied = torch.outer(loading, loading) + torch.diag(unique)
        if not bool(torch.isfinite(implied).all()):
            raise AnalysisError(
                "non_finite_result", "Implied covariance overflowed; rescale the items."
            )
        return (
            m.ratio(common, common + specific),
            {
                "items": c.frame(
                    torch.column_stack((loading, unique)),
                    columns=["loading", "unique_variance"],
                    index=names,
                ),
                "implied_covariance": c.frame(implied, columns=names, index=names),
            },
            dict(
                common_sum_variance=common,
                unique_sum_variance=specific,
                discrepancy=extracted.discrepancy,
                optimizer_iterations=extracted.iterations,
                factors=1,
                heywood=[],
                fit="Gaussian one-factor ML",
            ),
        )

    return m.result(
        "omega_total",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(
            standardized=standardized,
            reverse=flipped,
            max_iterations=iterations,
            variance_denominator="one-factor implied covariance sum",
        ),
    )
