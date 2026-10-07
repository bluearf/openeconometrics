"""Conditional (fixed-effects) logistic regression: Stata's ``clogit`` and ``xtlogit, fe``.

Model
-----
A binary outcome with a group effect ``a_g`` (a matched set, a stratum, a
person observed over time),

    Pr(y_i = 1 | x_i) = L(a_g + x_i'b + offset_i),   i in group g.

The ``a_g`` are not estimated. Conditioning on the number of positive outcomes
of each group, ``m_g = sum_{i in g} y_i``, removes them: the conditional
likelihood of group g is

    L_g = exp(sum_i y_i x_i'b) / sum_{d: sum d = m_g} exp(sum_i d_i x_i'b),

where the denominator runs over all ways of placing ``m_g`` positives among the
``T_g`` rows of the group. It is an elementary symmetric function, evaluated
together with its first two derivatives by the recursive algorithm of
``conditional.ConditionalLogitObjective`` (no enumeration; groups with one
positive or one negative use a plain multinomial-logit formula).

Sample
------
Groups whose outcomes are all positive or all negative have ``L_g = 1`` and
carry no information; they are dropped and counted, as Stata does. Regressors
that do not vary within any group are not identified and are omitted with a
warning. There is no constant.

Weights apply to groups as a whole (Stata's rule) and must be constant within
a group: ``fweight`` replicates groups (group and observation counts, dropped
ones included, are then frequency-weighted), ``iweight`` and ``pweight`` scale
each group's contribution.

Centring
--------
``L_g`` depends on the regressors only through their differences within the
group (``sum_i (y_i - d_i) = 0`` for every admissible ``d``), so the likelihood
is evaluated on ``x_i - xbar_g``. This is the same function of ``b``, but its
Hessian no longer loses digits to the level of a regressor (a year, an income
in currency units), and the per-observation scores below do not depend on the
origin of the regressors.

Covariance
----------
``nonrobust``: inverse observed information. ``opg``: outer product of the
GROUP scores. ``robust``: the sandwich clustered on the group,
``G/(G-1) (-H)^-1 [sum_g s_g s_g'] (-H)^-1`` (Stata: ``vce(robust)`` of clogit
is ``vce(cluster groupvar)``, because rows of one group are not independent
contributions). ``cluster``: per-observation scores
``(y_i - Pr(y_i = 1 | m_g)) (x_i - xbar_g)`` summed within the cluster
column(s), with ``G_c/(G_c-1)``. When every group lies inside one cluster the
cluster sums are sums of group scores (Stata's ``vce(cluster)``; the centring
is then immaterial). Groups that are split across clusters are accepted with a
warning: Stata refuses them unless ``nonest`` is given, and how a group's score
is divided among its rows is then a convention (see ``docs``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, ml_covariance,
)
from openecon.econometrics.discrete.common import (
    EXTREME, EXTREME_EARLY, Separated, Weights, binary_outcome, build_spec, check_pweights,
    likelihood_covariance, likelihood_weights, maximize, model_test, offset_column,
    optimizer_record, pseudo_r_squared, raise_not_converged, resolve_covariance,
    separation_watch,
)
from openecon.econometrics.discrete.conditional import ConditionalLogitObjective
from openecon.engines.absorb import demean
from openecon.engines.covariance import group_counts, group_sums
from openecon.models import ModelSpec, ResultBundle


def _informative_groups(frame: ModelFrame, group: str, command: str) -> tuple[Tensor, Tensor, int]:
    """Drop groups without outcome variation; return ``(y, codes, G)`` of the rest.

    The dropped groups and their observations are counted as Stata's note
    reports them; with fweights a group counts ``f_g`` times.
    """
    y = binary_outcome(frame, frame.spec.outcome, command)
    codes, count = frame.codes(group)
    sizes = group_counts(codes, count)
    positives = group_sums(y, codes, count)
    informative = (positives > 0) & (positives < sizes)
    dropped = int((~informative).sum())
    observations = int(sizes[~informative].sum())
    if dropped and frame.spec.weight_type == "fweight":
        raw = frame.weights()
        lost = ~informative[codes]
        replicas = torch.zeros(count, dtype=torch.float64).scatter_reduce(
            0, codes, raw, "amax", include_self=False)
        dropped = int(round(float(replicas[~informative].sum())))
        observations = int(round(float(raw[lost].sum())))
    frame.notes["groups_dropped"] = dropped
    frame.notes["observations_dropped"] = observations
    if bool((~informative).all()):
        raise AnalysisError(
            "no_outcome_variation",
            f"{command}: every group of '{group}' has all positive or all negative outcomes, so "
            "the conditional likelihood carries no information. The outcome must vary within "
            "at least one group.")
    if dropped:
        frame.restrict(informative[codes],
                       f"note: {dropped} group(s) ({observations} obs) "
                       "dropped because of all positive or all negative outcomes.")
        y = binary_outcome(frame, frame.spec.outcome, command)
        codes, count = frame.codes(group)
    return y, codes, count


def _group_weights(weights: Weights, codes: Tensor, count: int, group: str,
                   command: str) -> Tensor:
    if not weights.weighted:
        return torch.ones(count, dtype=torch.float64)
    empty = torch.full((count,), -torch.inf, dtype=torch.float64)
    highest = empty.scatter_reduce(0, codes, weights.user, "amax")
    lowest = -empty.clone().scatter_reduce(0, codes, -weights.user, "amax")
    if bool((highest != lowest).any()):
        raise AnalysisError(
            "weights_not_constant_within_group",
            f"{command}: weights apply to groups as a whole and must be constant within each "
            f"group of '{group}' (Stata's rule for clogit).")
    return highest


def _note_split_groups(frame: ModelFrame, codes: Tensor, count: int, group: str) -> None:
    """Warn when a cluster column cuts through the groups (Stata's ``nonest`` situation)."""
    for name in registry.cluster_columns(frame.spec):
        cluster = frame.codes(name)[0].to(torch.float64)
        empty = torch.zeros(count, dtype=torch.float64)
        highest = empty.scatter_reduce(0, codes, cluster, "amax", include_self=False)
        lowest = empty.scatter_reduce(0, codes, cluster, "amin", include_self=False)
        split = int((highest != lowest).sum())
        if split:
            frame.warn(
                f"{split} group(s) of '{group}' are split across the clusters of '{name}'. "
                "The cluster sums then use the per-observation scores "
                "(y - Pr(y = 1 | m)) (x - group mean of x), a convention chosen so that the "
                "covariance does not depend on the origin of the regressors; Stata refuses "
                "this design unless nonest is specified.")


def _covariance(frame: ModelFrame, objective: ConditionalLogitObjective, theta: Tensor,
                hessian: Tensor, weights: Weights, group: str, groups: int,
                group_w: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Covariance with the group as the unit of the OPG and robust estimators."""
    kind = frame.spec.covariance
    if kind == "cluster":
        _note_split_groups(frame, objective.codes, objective.groups, group)
    if kind in {"nonrobust", "cluster"}:
        return likelihood_covariance(frame, hessian, lambda: objective.score_rows(theta), weights)
    scores = kernel_call(objective.group_scores, theta)
    frequency = group_w if weights.frequency is not None else None
    if frequency is None and weights.weighted:
        scores = scores * group_w[:, None]
    try:
        covariance, info = ml_covariance(frame, hessian=hessian, scores=scores, nobs=groups,
                                         frequency=frequency, kind=kind)
    except AnalysisError as exc:
        if exc.code != "singular_information":
            raise
        raise AnalysisError(
            "singular_information",
            "The information matrix is not positive definite at the estimates, so standard "
            "errors are undefined: a regressor has too little within-group variation.") from exc
    info["df_inference"] = None
    if kind == "robust":
        info.update({"correction": "robust = cluster sandwich on the group (Stata's clogit): "
                                   "G/(G-1)",
                     "cluster_count": groups, "cluster_df": groups - 1, "cluster_column": group})
    else:
        info["correction"] = "outer product of the group scores"
    return covariance, info


def fit_clogit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``clogit``: ``(spec, data) -> ResultBundle``."""
    command = "clogit"
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    group = frame.role("group")[0]
    if group == spec.outcome or group in spec.predictors:
        raise AnalysisError("invalid_spec", f"{command}: the group column '{group}' must differ "
                            "from the outcome and the regressors.")
    y, codes, count = _informative_groups(frame, group, command)
    weights = likelihood_weights(frame)
    group_w = _group_weights(weights, codes, count, group, command)
    groups = int(round(float(group_w.sum()))) if weights.frequency is not None else count
    offset = offset_column(frame)
    design = frame.design(intercept=False)
    screen = weights.for_screen()
    within = kernel_call(demean, design.x, [(codes, count)], screen).values
    design, within = frame.drop_absorbed(design, within, screen)
    k = len(design.terms)
    if k == 0:
        raise AnalysisError(
            "no_within_group_variation",
            f"{command}: no regressor varies within the groups of '{group}', so nothing is "
            "identified by the conditional likelihood.")
    if groups <= k:
        raise AnalysisError("insufficient_observations", f"{command} needs more informative "
                            f"groups ({groups}) than parameters ({k}).")
    # The conditional likelihood is a function of x_i - xbar_g only (see "Centring").
    objective = kernel_call(ConditionalLogitObjective, within, y, codes, count, group_w, offset)

    def separated(theta: Tensor, threshold: float) -> int:
        # A group whose observed outcomes have conditional probability one.
        log_likelihood = objective.group_log_likelihood(theta)
        return int((log_likelihood > -threshold).sum())

    start = torch.zeros(k, dtype=torch.float64)
    try:
        result = maximize(objective, start, what=command, callback=separation_watch(separated),
                          scale=weights.scale)
    except Separated as stop:
        raise_not_converged(command, separated(stop.theta, EXTREME_EARLY),
                            "the likelihood is monotone.", unit="group")
    if not result.converged:
        raise_not_converged(command, separated(result.theta, EXTREME),
                            result.diagnostics.get("message"), unit="group")
    theta, log_likelihood = result.theta, result.value
    covariance, info = _covariance(frame, objective, theta, result.hessian, weights, group,
                                   groups, group_w)
    null = float(objective.value(start))
    criteria = information_criteria(log_likelihood, k, weights.nobs)
    metrics = {"log_likelihood": log_likelihood,
               "pseudo_r_squared": pseudo_r_squared(log_likelihood, null),
               "aic": criteria["aic"], "bic": criteria["bic"], "n_groups": groups,
               "n_groups_dropped": frame.notes["groups_dropped"]}
    tests = {"model": model_test(frame, theta, covariance, range(k), log_likelihood, null,
                                 null_model="b = 0", what="the coefficients")}
    sizes = objective.sizes.to(torch.float64)
    replicas = group_w if weights.frequency is not None else torch.ones_like(sizes)
    extra = {
        "group": group, "null_log_likelihood": null,
        "group_sizes": {"min": int(sizes.min()),
                        "mean": float((sizes * replicas).sum() / replicas.sum()),
                        "max": int(sizes.max())},
        "multiple_positive_outcomes": bool((objective.positives > 1).any()),
        "observations_dropped": frame.notes["observations_dropped"],
        "weights_apply_to": "groups" if weights.weighted else None,
        "likelihood": "conditional on the number of positive outcomes in each group",
    }
    return build_result(
        frame, terms=design.terms, params=theta, covariance=covariance, use_t=False,
        metrics=metrics, nobs=weights.nobs, inference=info, tests=tests, extra=extra,
        categories=design.categories, solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def clogit(*, data: Any, y: str, x: Sequence[str], group: str, covariance: str | None = None,
           cluster: str | Sequence[str] | None = None, weights: str | None = None,
           weight_type: str | None = None, offset: str | None = None,
           categorical: Sequence[str] | None = None, missing: str = "raise",
           alpha: float = 0.05) -> ResultBundle:
    """Conditional (fixed-effects) logistic regression, Stata's ``clogit`` / ``xtlogit, fe``.

    Model
        ``Pr(y_i = 1) = L(a_g + x_i'b)`` for row i of group g, with an unrestricted group
        effect ``a_g``. Conditioning on the number of positives ``m_g`` in each group
        eliminates ``a_g``:

            ``ln L_g = sum_i y_i x_i'b - ln f_g(T_g, m_g)``,

        where ``f_g`` is the sum of ``exp(sum_i d_i x_i'b)`` over all outcome vectors d of
        the group with ``m_g`` ones. ``exp(b)`` is an odds ratio within the group. The same
        likelihood is McFadden's choice model when each group is a choice set with one
        chosen alternative, and ``xtlogit, fe`` when the group is the panel unit.

    Estimator
        Newton-Raphson from ``b = 0`` on the conditional log likelihood. ``f_g`` and its
        first two derivatives come from the recursion
        ``f(t, j) = f(t-1, j) + exp(x_t'b) f(t-1, j-1)`` carried in a normalized
        (overflow-free) form for all groups at once; the cost is
        ``O(N min(m, T - m) k^2)`` and ``O(N k^2)`` for groups with one positive.
        Regressors are centred within groups first, which leaves the likelihood unchanged
        and keeps the Hessian accurate when a regressor has a large level.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome coded 0/1. Several positives per group are allowed.
        x: list of regressors. A regressor without within-group variation is omitted
            with a warning; there is no constant.
        group: column identifying the groups (matched sets, strata, panel units); any
            scalar labels. Groups with all positive or all negative outcomes are dropped
            and counted in ``metrics['n_groups_dropped']`` and a warning.
        covariance: ``'nonrobust'`` (default; inverse observed information), ``'opg'``
            (outer product of group scores), ``'robust'`` (sandwich clustered on the
            group, ``G/(G-1)``; Stata's ``vce(robust)`` of clogit), ``'cluster'``
            (implied by ``cluster``; ``G_c/(G_c-1)``).
        cluster: cluster column, or a list of two columns. Clusters normally contain whole
            groups (Stata's requirement). Groups split across clusters are accepted with a
            warning (Stata's ``nonest``); a group's score is then divided among its rows as
            ``(y_i - Pr(y_i = 1 | m_g)) (x_i - xbar_g)``.
        weights, weight_type: ``'fweight'``, ``'iweight'`` or ``'pweight'`` (robust by
            default); constant within group, applied to the group as a whole.
        offset: column added to ``x'b`` with coefficient 1.
        categorical: regressors to expand into treatment-coded indicators.
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Result
        z statistics for the slopes. ``metrics``: log_likelihood, pseudo_r_squared
        (``1 - ll/ll_0`` with ``ll_0`` the conditional likelihood at ``b = 0``), aic, bic,
        n_groups, n_groups_dropped. ``tests['model']``: LR chi2(k) against ``b = 0``
        (``nonrobust``) or Wald chi2(k). ``extra``: ``group``, ``group_sizes``,
        ``multiple_positive_outcomes``, ``observations_dropped``, ``null_log_likelihood``.
        ``nobs`` counts the rows of the informative groups. A regressor that predicts the
        outcome perfectly within groups raises ``separation_detected``.

    Stata
        ``clogit low lwt smoke ptd ht ui i.race, group(pairid)`` is
        ``oe.clogit(data=df, y='low', x=['lwt', 'smoke', 'ptd', 'ht', 'ui', 'race'],
        categorical=['race'], group='pairid')``; ``xtlogit union age grade, fe`` (after
        ``xtset idcode``) is ``oe.clogit(data=df, y='union', x=['age', 'grade'],
        group='idcode')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"id": np.repeat(np.arange(300), 5), "x": rng.normal(size=1500)})
        >>> effect = rng.normal(size=300)[df.id]
        >>> df["y"] = (rng.random(1500) < 1 / (1 + np.exp(-(effect + 0.8 * df.x)))) * 1.0
        >>> print(oe.clogit(data=df, y="y", x=["x"], group="id").summary())
    """
    spec = build_spec(
        "clogit", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=False,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"group": group, "offset": offset})
    return fit(spec, data=data)
