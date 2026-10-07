"""Multinomial (polytomous) logistic regression: Stata's ``mlogit`` (SPSS NOMREG).

Model
-----
For an outcome with ``J`` unordered categories and a base category ``B``,

    Pr(y = j | x) = exp(x'b_j) / sum_l exp(x'b_l),      b_B = 0,

so each of the ``J - 1`` other categories has its own coefficient vector and
``exp(b_j)`` is a relative-risk ratio against the base. The log likelihood
``sum_i w_i [x_i'b_{y_i} - ln sum_l exp(x_i'b_l)]`` is globally concave.

Categories and base
-------------------
Categories are the distinct outcome values of the estimation sample, listed in
sorted order (the category order of a pandas ``Categorical``; first appearance
when the values cannot be sorted). The base is the category with the largest
(weighted) frequency, as in Stata, with ties going to the first category in
that order; ``base=`` selects another one.

Estimation
----------
Newton-Raphson from zero with the analytic score and the full
``k (J - 1)``-square Hessian of ``kernels.MultinomialObjective`` (softmax by a
stable log-sum-exp; Hessian accumulated over row blocks). With a constant the
iteration runs on regressors centred at their means and the constants are
mapped back exactly (``a_j = a_cj - m'b_j``).

Reported
--------
One equation per non-base category, terms ``<category>:<term>``. ``metrics``:
log_likelihood, pseudo_r_squared (McFadden, against the constant-only model,
whose probabilities are the weighted category shares), aic, bic
(``k (J - 1)`` parameters), n_categories. ``tests['model']``: LR chi2 against
the constant-only model (``nonrobust``) or the Wald chi2 of all slopes.
Without a constant the null model is ``b = 0`` (equal probabilities).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call,
)
from openecon.econometrics.discrete.common import (
    EXTREME, EXTREME_EARLY, Separated, build_spec, centre_regressors, check_pweights,
    constant_shift, json_label, label_text, likelihood_covariance, likelihood_weights, maximize,
    model_test, optimizer_record, perfectly_predicted, pseudo_r_squared, raise_not_converged,
    reparameterize, require_observations, resolve_covariance, separation_watch,
)
from openecon.econometrics.discrete.kernels import MultinomialObjective
from openecon.engines.covariance import group_sums
from openecon.models import ModelSpec, ResultBundle

MAX_CATEGORIES = 300


def multinomial_outcome(frame: ModelFrame, command: str) -> tuple[Tensor, list[Any]]:
    """Category codes ``0..J-1`` in display order and the category labels."""
    name = frame.spec.outcome
    series = frame.series(name)
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = series.cat.remove_unused_categories()
        values, uniques = observed.cat.codes.to_numpy(dtype="int64", copy=True), \
            list(observed.cat.categories)
    else:
        if is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
            frame.numeric(name)                       # finite real values only
        try:
            values, uniques = pd.factorize(series, sort=True)
        except TypeError:
            # Mixed labels cannot be sorted: keep the order of first appearance.
            values, uniques = pd.factorize(series, sort=False)
        uniques = list(uniques)
    labels = [json_label(value) for value in uniques]
    if len(labels) < 2:
        raise AnalysisError("constant_outcome", f"{command}: the outcome '{name}' takes a single "
                            "value in the estimation sample; at least two categories are needed.")
    if len(labels) > MAX_CATEGORIES:
        raise AnalysisError(
            "too_many_categories",
            f"{command}: the outcome '{name}' has {len(labels)} distinct values (limit "
            f"{MAX_CATEGORIES}). A multinomial model needs a small set of categories.")
    names = [label_text(label) for label in labels]
    if len(set(names)) != len(names):
        raise AnalysisError("ambiguous_categories", f"{command}: outcome categories of '{name}' "
                            "have identical labels after conversion to text; recode them.")
    return torch.from_numpy(values.astype("int64")), labels


def _base_category(frame: ModelFrame, labels: list[Any], totals: Tensor, command: str) -> int:
    requested = frame.option("base")
    if requested is None:
        return int(torch.argmax(totals))              # first of the most frequent categories
    for position, label in enumerate(labels):
        if type(label) is bool or type(requested) is bool:
            match = label is requested
        else:
            match = label == requested
        if match:
            return position
    raise AnalysisError(
        "invalid_base_category",
        f"{command}: base={requested!r} is not an outcome category of the estimation sample "
        f"(categories: {', '.join(label_text(label) for label in labels)}).")


def fit_mlogit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``mlogit``: ``(spec, data) -> ResultBundle``."""
    command = "mlogit"
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    codes, labels = multinomial_outcome(frame, command)
    categories = len(labels)
    weights = likelihood_weights(frame)
    totals = group_sums(weights.user, codes, categories)
    base = _base_category(frame, labels, totals, command)
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    centre = centre_regressors(design, weights)
    k, equations = len(design.terms), categories - 1
    if k == 0:
        raise AnalysisError("no_parameters", f"{command}: no regressor is left to estimate; "
                            "include a constant or a regressor that varies.")
    size = k * equations
    require_observations(weights.nobs, size)
    objective = kernel_call(MultinomialObjective, design.x, codes, categories, base, weights.user)

    def separated(theta: Tensor, threshold: float) -> int:
        # A category predicted perfectly, or ruled out for part of the sample.
        return perfectly_predicted(objective.probabilities(theta), threshold) \
            or objective.excluded(theta, threshold)

    start = torch.zeros(size, dtype=torch.float64)
    try:
        result = maximize(objective, start, what=command, callback=separation_watch(separated),
                          scale=weights.scale)
    except Separated as stop:
        raise_not_converged(command, separated(stop.theta, EXTREME)
                            or separated(stop.theta, EXTREME_EARLY), "the likelihood is monotone.")
    if not result.converged:
        raise_not_converged(command, separated(result.theta, EXTREME),
                            result.diagnostics.get("message"))
    centred, log_likelihood = result.theta, result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    # Fitted on x - m: every equation's constant is a = a_c - m'b_j.
    jacobian = torch.kron(torch.eye(equations, dtype=torch.float64), constant_shift(centre))
    theta, covariance = reparameterize(centred, covariance, jacobian)
    total = float(totals.sum())
    if design.intercept:
        null = float((totals * torch.log(totals / total)).sum())
        null_model = "the constant-only model"
    else:
        null, null_model = -total * math.log(categories), "the equal-probability model (b = 0)"
    slopes = [j * k + i for j in range(equations) for i, term in enumerate(design.terms)
              if term != "Intercept"]
    criteria = information_criteria(log_likelihood, size, weights.nobs)
    metrics = {"log_likelihood": log_likelihood,
               "pseudo_r_squared": pseudo_r_squared(log_likelihood, null),
               "aic": criteria["aic"], "bic": criteria["bic"], "n_categories": categories}
    tests = {"model": model_test(frame, theta, covariance, slopes, log_likelihood, null,
                                 null_model=null_model)}
    names = [label_text(label) for position, label in enumerate(labels) if position != base]
    extra = {
        "categories": labels, "base": labels[base], "equations": names,
        "category_counts": totals.tolist(), "null_log_likelihood": null,
        "base_rule": "option base" if frame.option("base") is not None
        else "most frequent category (first in category order on ties)",
    }
    return build_result(
        frame, terms=[f"{name}:{term}" for name in names for term in design.terms],
        params=theta, covariance=covariance,
        equations=[name for name in names for _ in design.terms], use_t=False,
        metrics=metrics, nobs=weights.nobs, inference=info, tests=tests, extra=extra,
        categories=design.categories, solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def mlogit(*, data: Any, y: str, x: Sequence[str], base: Any = None,
           covariance: str | None = None, cluster: str | Sequence[str] | None = None,
           weights: str | None = None, weight_type: str | None = None,
           categorical: Sequence[str] | None = None, intercept: bool = True,
           missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Multinomial logistic regression, Stata's ``mlogit`` (SPSS NOMREG).

    Model
        For ``J`` unordered outcome categories with base category ``B``,

            ``Pr(y = j | x) = exp(x'b_j) / sum_l exp(x'b_l)``,  ``b_B = 0``.

        ``b_j`` is the effect of ``x`` on the log odds of category ``j`` against the
        base; ``exp(b_j)`` is the relative-risk ratio (Stata's ``rrr``).

    Estimator
        Maximum likelihood of ``sum_i w_i ln Pr(y_i | x_i)`` over the ``k (J - 1)``
        coefficients by Newton-Raphson from zero, with the analytic score
        ``X'(w (d_j - p_j))`` and Hessian blocks ``-X' diag(w p_j (1[j=l] - p_l)) X``.
        The likelihood is globally concave, so the iteration converges unless a
        regressor separates some categories (``separation_detected``).

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome with ``J >= 2`` categories: any scalar labels (numbers, strings,
            booleans or a pandas ``Categorical``). Categories are reported in sorted
            order (``Categorical`` order).
        x: list of regressors; collinear columns are omitted with a warning.
        base: the base category (a value of ``y``). Default: the most frequent category
            (weighted frequency; the first category on ties), as in Stata.
        covariance: ``'nonrobust'`` (default; inverse observed information), ``'opg'``,
            ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``; implied by
            ``cluster``; two columns give two-way clustering).
        cluster: cluster column, or a list of two columns.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        categorical: regressors to expand into treatment-coded indicators.
        intercept: include a constant in every equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Result
        z statistics. Terms are ``<category>:<term>`` grouped in one equation per
        non-base category (equation = the category label). ``metrics``: log_likelihood,
        pseudo_r_squared (``1 - ll/ll_0`` with the constant-only model), aic, bic
        (``k (J - 1)`` parameters), n_categories. ``tests['model']``: LR chi2 with
        ``(k - 1)(J - 1)`` degrees of freedom (``nonrobust``) or the Wald chi2 of all
        slopes. ``extra``: ``categories``, ``base``, ``equations``, ``category_counts``,
        ``null_log_likelihood``.

    Stata
        ``mlogit insure age male nonwhite, baseoutcome(1) vce(robust)`` is
        ``oe.mlogit(data=df, y='insure', x=['age', 'male', 'nonwhite'], base=1,
        covariance='robust')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=900)})
        >>> utility = np.column_stack([np.zeros(900), 0.5 + df.x, -0.3 - 0.7 * df.x])
        >>> df["choice"] = (utility + rng.gumbel(size=(900, 3))).argmax(axis=1)
        >>> print(oe.mlogit(data=df, y="choice", x=["x"], base=0).summary())
    """
    spec = build_spec(
        "mlogit", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        options={"base": base})
    return fit(spec, data=data)
