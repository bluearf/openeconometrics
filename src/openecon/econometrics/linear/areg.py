"""Linear regression with one absorbed categorical variable: Stata's ``areg``.

Model and estimator
-------------------
``y_ig = x_ig'b + alpha_g + u_ig`` with ``G`` levels of the absorbed variable.
The level effects are partialled out by the (weighted) within transform
``M_g x = x - mean_g(x)`` (``engines.absorb.demean`` with one dimension, one
``index_add_`` pass): by Frisch-Waugh-Lovell the least-squares slopes on the
demeaned data equal those of the dummy-variable regression. A regressor
without within-level variation is omitted (``ModelFrame.drop_absorbed``,
which compares the within sum of squares with the mean-deviated original one
at 1e-13, so a regressor with a large offset or between variation is kept).

Stata's areg reports a constant. Following its Methods and formulas, the
grand (weighted) means are added back to the demeaned variables and the
regression ``(y~ + ybar) on [1, X~ + xbar]`` is solved, so that
``_cons = ybar - xbar'b`` and its standard error come from the same
least-squares problem. Because ``X~`` has weighted mean zero, the slope block
of every covariance equals the within-regression covariance exactly.

Degrees of freedom (the areg convention)
----------------------------------------
The absorbed indicators count as parameters: ``K_total = k_slopes + G`` (the
constant plus ``G - 1`` indicators), ``df_resid = N - K_total``,
``sigma^2 = RSS / df_resid``. HC1 (``vce(robust)``) uses ``N/(N - K_total)``
and cluster CR1 uses ``G_c/(G_c - 1) (N - 1)/(N - K_total)`` with the same
``K_total`` even when the absorbed levels are nested inside the clusters.
This is the documented difference from ``xtreg, fe`` (which does not count
the absorbed effects when they are nested in the clusters) and makes areg's
clustered standard errors larger by ``sqrt((N - k - 1)/(N - k - G))``.

Reported
--------
``r_squared`` (overall, including the absorbed effects: ``1 - RSS/TSS`` about
the grand mean), ``adjusted_r_squared`` with ``K_total``, ``r_squared_within``
(``1 - RSS/TSS_within``), ``rmse``, ``df_model`` (slopes), ``df_resid``,
``df_absorbed = G - 1``, ``n_groups``. ``tests['model']`` is the F test of the
slopes (classical ``(MSS_within/k)/(RSS/df_resid)`` or Wald-robust) and
``tests['absorbed']`` the F test that all level effects are zero,
``((RSS_pooled - RSS)/(G - 1)) / (RSS/df_resid)``, conventional covariance only.
``extra['absorbed']`` has the same shape as reghdfe's: one record
``{column, levels, redundant, nested}`` where ``redundant = 1`` is the level
absorbed into the reported constant (``levels - redundant = df_absorbed``) and
``nested`` says whether the levels nest within a cluster column (recorded for
information; areg counts them regardless).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, linear_covariance, make_spec,
)
from openecon.econometrics.linear.common import (
    check_fit, check_observations, check_pweights, classical_f_test, covariance_kind,
    estimation_weights, model_test, nested_in_clusters, ones, resolve_covariance,
    solve_least_squares, solver_diagnostics, sum_of_squares, weighted_mean,
)
from openecon.engines.absorb import demean
from openecon.models import ModelSpec, ResultBundle


def fit_areg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``areg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    absorb = frame.role("absorb")[0]
    codes, groups = frame.codes(absorb)
    if groups < 2:
        raise AnalysisError("insufficient_groups", f"The absorbed variable '{absorb}' has a single "
                            "level; use regress instead.")
    check_observations(frame, groups, what=f"absorbed levels of '{absorb}'")
    weights, nobs = estimation_weights(frame)
    kind = covariance_kind(frame)
    clustered = kind == "cluster"
    clusters = frame.cluster_dimensions() if clustered else []
    y = frame.numeric(spec.outcome)
    design = frame.design(intercept=False)
    block = torch.cat([y[:, None], design.x], dim=1)
    within = kernel_call(demean, block, [(codes, groups)], weights)
    y_within, x_within = within.values[:, 0], within.values[:, 1:]
    design, x_within = frame.drop_absorbed(design, x_within, weights)
    k_slopes = len(design.terms)
    k_total = k_slopes + groups
    df_resid = nobs - k_total
    # Stata's areg: grand means added back and a constant column, so that _cons and its
    # standard error come out of the same least-squares problem as the slopes.
    y_mean, x_mean = weighted_mean(y, weights), weighted_mean(design.x, weights)
    # Keep the constant orthogonal to within slopes. Adding large grand means
    # before the sandwich makes bread/meat products cancel catastrophically.
    # The reporting constant and covariance are mapped back only after fitting.
    x_tilde = torch.cat([ones(frame.n), x_within], dim=1)
    fitted = solve_least_squares(x_tilde, y_within, weights)
    rss = float(fitted.ssr)
    tss = sum_of_squares(y, weights, centered=True)
    tss_within = sum_of_squares(y_within, weights, centered=False)
    check_fit(rss, tss, df_resid, sum_of_squares(y, weights, centered=False), absorbed=True)
    covariance, info = linear_covariance(
        frame, x=x_tilde, resid=fitted.resid, bread=fitted.xtx_inv, n=nobs, k=k_total,
        df_resid=df_resid, weights=weights, ssr=rss, kind=kind, clusters=clusters or None,
        cluster_names=registry.cluster_columns(spec) if clustered else None)
    transform = torch.eye(k_slopes + 1, dtype=torch.float64)
    transform[0, 1:] = -x_mean
    if info.get("psd_adjusted"):
        # CGM eigenvalue clipping is not equivariant under centering. Its
        # original-coordinate meat must be adjusted before mapping it back to
        # the numerically stable centered sandwich (not an unstable raw bread).
        original = torch.cat([ones(frame.n), x_within + x_mean], dim=1)
        original_meat, info = linear_covariance(
            frame, x=original, resid=fitted.resid,
            bread=torch.eye(k_slopes + 1, dtype=torch.float64), n=nobs, k=k_total,
            df_resid=df_resid, weights=weights, ssr=rss, kind=kind,
            clusters=clusters or None,
            cluster_names=registry.cluster_columns(spec) if clustered else None)
        centered_meat = transform.T @ original_meat @ transform
        covariance = fitted.xtx_inv @ centered_meat @ fitted.xtx_inv
    covariance = transform @ covariance @ transform.T
    params = transform @ fitted.beta
    params[0] += y_mean
    terms = ["Intercept", *design.terms]
    r_squared = 1 - rss / tss
    metrics = {
        "r_squared": r_squared,
        "adjusted_r_squared": 1 - (1 - r_squared) * (nobs - 1) / df_resid,
        "r_squared_within": 1 - rss / tss_within if tss_within > 0 else None,
        "rmse": (rss / df_resid) ** 0.5, "df_model": k_slopes, "df_resid": df_resid,
        "df_absorbed": groups - 1, "n_groups": groups,
    }
    tests = {"model": model_test(
        frame, params, covariance, list(range(1, k_slopes + 1)),
        df_inference=info["df_inference"], classical=(tss_within - rss, k_slopes, rss, df_resid),
        label="Model F test (slopes)")}
    if spec.covariance == "nonrobust":
        # F test of the absorbed indicators: pooled regression on [1, X] = the restricted model.
        pooled = solve_least_squares(torch.cat([ones(frame.n), design.x], dim=1), y, weights)
        tests["absorbed"] = classical_f_test(float(pooled.ssr) - rss, groups - 1, rss, df_resid,
                                             label=f"F test of absorbed {absorb} effects")
    nested = nested_in_clusters([(codes, groups)], clusters)[0] if clustered else False
    info.update({"nobs": nobs, "k_total": k_total, "absorbed_degrees_of_freedom": groups - 1,
                 "degrees_of_freedom_convention": "areg: absorbed indicators count as parameters, "
                 "also when nested within clusters"})
    return build_result(
        frame, terms=terms, params=params, covariance=covariance, nobs=nobs,
        df_inference=info["df_inference"], df_resid=df_resid, metrics=metrics,
        fitted=y - fitted.resid, inference=info, tests=tests, categories=design.categories,
        solver="within_transform_householder_qr",
        solver_diagnostics=solver_diagnostics(fitted, absorbed_levels=groups),
        extra={"absorbed": [{"column": absorb, "levels": groups, "redundant": 1, "nested": nested}],
               "constant_degree_of_freedom_added": True})


def areg(*, data: Any, y: str, x: Sequence[str], absorb: str, covariance: str | None = None,
         cluster: str | Sequence[str] | None = None, weights: str | None = None,
         weight_type: str | None = None, categorical: Sequence[str] | None = None,
         missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Linear regression absorbing one categorical variable, Stata's ``areg``.

    Model
        ``y = x'b + alpha[absorb] + u``: one fixed effect per level of ``absorb``,
        removed by the within transform ``x - mean_level(x)`` (weighted when weights
        are given). Slopes equal those of the dummy-variable regression; a regressor
        constant within levels is omitted and recorded. The constant ``Intercept`` is
        ``ybar - xbar'b`` (grand means) with its standard error from the regression of
        the demeaned-plus-grand-mean variables on a column of ones, as Stata computes it.

    Degrees of freedom (areg convention)
        ``K_total = k + G`` (``k`` slopes, the constant and ``G - 1`` absorbed
        indicators), ``df_resid = N - K_total``. ``nonrobust`` = ``RSS/df_resid (X~'X~)^-1``;
        ``'HC1'`` (alias ``'robust'``) = White sandwich times ``N/(N - K_total)``;
        ``'cluster'`` = CR1 times ``G_c/(G_c-1) (N-1)/(N - K_total)`` with the SAME
        ``K_total`` even when the absorbed levels are nested within the clusters. That
        is the documented difference from ``xtreg, fe`` and from ``reghdfe``, which do
        not count nested effects. t tests use ``df_resid`` (``G_c - 1`` with clusters;
        ``G_min - 1`` with two cluster columns, whose Cameron-Gelbach-Miller meat is
        made positive semidefinite by zeroing negative eigenvalues when needed, with
        ``inference['psd_adjusted']`` and a warning).

    Parameters
        data: DataFrame, mapping of columns or list of row records.
        y: outcome column. x: list of regressor columns (a bare string is an error).
        absorb: column whose levels are absorbed (any scalar labels).
        covariance: ``'nonrobust'`` (default), ``'HC1'``/``'robust'``, ``'cluster'``
            (one or two cluster columns; implied by ``cluster``).
        cluster: one cluster column, or two for two-way clustering.
        weights, weight_type: ``'aweight'`` (rescaled to sum to N), ``'fweight'``
            (replicates observations; N = sum of weights) or ``'pweight'`` (sampling
            weights: need HC1 or cluster; HC1 is their default).
        categorical: regressors expanded as ``name[level]`` dummies (first level omitted).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing model inputs.
        alpha: significance level of the confidence intervals.

    Result
        ``metrics``: r_squared (overall, with the fixed effects), adjusted_r_squared
        (with ``K_total``), r_squared_within, rmse, df_model, df_resid, df_absorbed
        (``G - 1``), n_groups. ``tests['model']``: F test of the slopes;
        ``tests['absorbed']``: F test that all absorbed effects are zero (conventional
        covariance only, as Stata prints it). ``extra['absorbed']``: one record
        ``{column, levels, redundant, nested}`` (reghdfe's shape).

    Errors
        ``insufficient_groups`` (one level), ``insufficient_observations`` (no more
        observations than absorbed levels, or no residual degrees of freedom),
        ``perfect_fit``, ``constant_outcome``, ``unsupported_covariance`` (pweights with
        the conventional covariance), ``missing_values``, ``invalid_spec``.

    Stata
        ``areg y x1 x2, absorb(firm) vce(cluster firm)`` is
        ``oe.areg(data=df, y='y', x=['x1', 'x2'], absorb='firm', cluster='firm')``.

    Example
        >>> import pandas as pd, numpy as np, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"firm": np.repeat(range(40), 5), "x": rng.normal(size=200)})
        >>> df["y"] = df.firm * 0.1 + 1.5 * df.x + rng.normal(size=200)
        >>> print(oe.areg(data=df, y="y", x=["x"], absorb="firm").summary())
    """
    if not isinstance(absorb, str):
        raise AnalysisError("invalid_spec", "areg absorbs exactly one column: pass absorb='name' "
                            "(use reghdfe for several fixed-effect dimensions).")
    spec = make_spec(
        "areg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=True,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"absorb": absorb})
    return fit(spec, data=data)
