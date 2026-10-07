"""Entry point and convenience function of Stata's ``teffects`` (all six estimators)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, column_list, make_spec
from openecon.models import ModelSpec, ResultBundle

_MATCHING = {"nnmatch", "psmatch"}
# Options and roles that only some methods read; giving them elsewhere is a mistake.
_OWNERS = {
    "omodel": {"ra", "ipwra", "aipw"}, "tmodel": {"ipw", "ipwra", "aipw", "psmatch"},
    "pstolerance": {"ipw", "ipwra", "aipw", "psmatch"}, "neighbors": _MATCHING,
    "metric": {"nnmatch"}, "caliper": _MATCHING, "vce_neighbors": _MATCHING,
    "matching_vce": {"nnmatch"},
}
_ROLE_OWNERS = {"tx": {"ipw", "ipwra", "aipw", "psmatch"}, "biasadj": _MATCHING,
                "ematch": {"nnmatch"}}


def fit_teffects(spec: ModelSpec, data: Any) -> ResultBundle:
    """Validate the method-specific contract and dispatch to the estimator."""
    frame = ModelFrame(spec, data)
    method = frame.option("method")
    for option, owners in _OWNERS.items():
        if option in spec.options and method not in owners:
            raise AnalysisError("invalid_spec", f"Option '{option}' does not apply to "
                                f"method='{method}' (it is used by {', '.join(sorted(owners))}).")
    for role, owners in _ROLE_OWNERS.items():
        if frame.role(role) and method not in owners:
            raise AnalysisError("invalid_spec", f"'{role}' columns do not apply to "
                                f"method='{method}'.")
    treatment = frame.role("treatment")[0]
    if frame.option("tlevel") is not None and frame.option("estimand") != "atet":
        raise AnalysisError("invalid_spec", "tlevel specifies the ATET target population only.")
    if treatment in spec.predictors or treatment in frame.role("tx"):
        raise AnalysisError("invalid_spec", f"The treatment '{treatment}' must not also be a "
                            "covariate.")
    if treatment == spec.outcome:
        raise AnalysisError("invalid_spec", "The treatment must differ from the outcome.")
    if method in _MATCHING:
        from openecon.econometrics.teffects.matching import fit_matching

        return fit_matching(frame, method)
    from openecon.econometrics.teffects.stacked import fit_stacked

    return fit_stacked(frame, method)


def teffects(*, data: Any, y: str, treatment: str, x: Sequence[str] | None = None,
             tx: Sequence[str] | None = None, method: str = "ra", estimand: str = "ate",
             omodel: str | None = None, tmodel: str | None = None, control: Any = None,
             neighbors: int | None = None, metric: str | None = None,
             biasadj: Sequence[str] | None = None, caliper: float | None = None,
             ematch: Sequence[str] | None = None, tlevel: Any = None,
             matching_vce: str | None = None,
             vce_neighbors: int | None = None, pstolerance: float | None = None,
             categorical: Sequence[str] | None = None, intercept: bool = True,
             covariance: str | None = None, cluster: str | None = None,
             weights: str | None = None, weight_type: str | None = None,
             missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Treatment effects under selection on observables (Stata's ``teffects``).

    Potential outcomes y(l) for treatment levels l = 0 (control) .. L-1; the
    estimands are the potential-outcome means POM_l = E[y(l)], the average
    treatment effects ATE_l = POM_l - POM_0 and the effects in the population
    ATET_l = E[y(l) - y(0) | D = tlevel]. Identification rests on
    unconfoundedness given the covariates and on overlap (every observation
    has a positive probability of every level).

    ``method`` selects the estimator (``D`` observed level, ``p_l(x)`` the
    treatment-model probability, ``mu_l(x)`` the outcome model of level l
    fitted on the observations with D = l):

    - ``'ra'`` regression adjustment (``teffects ra``): POM_l = mean of mu_l(x_i)
      over all observations (atet: over the treated).
    - ``'ipw'`` inverse-probability weighting (``teffects ipw``): POM_l is the
      weighted mean of y among D = l with normalized weights 1/p_l(x) (atet:
      weight 1 for the treated and p_1/(1 - p_1) for the controls).
    - ``'ipwra'`` (``teffects ipwra``): regression adjustment whose outcome
      models are fitted by weighted (quasi-)ML with the IPW weights; doubly
      robust (Wooldridge 2007).
    - ``'aipw'`` (``teffects aipw``): POM_l = mean of
      1{D=l}(y - mu_l(x))/p_l(x) + mu_l(x), the efficient-influence-function
      estimator (Robins, Rotnitzky and Zhao 1994). ATET uses target-population
      augmentation and p_target/p_l weighting; all contrasts share tlevel.
    - ``'nnmatch'`` / ``'psmatch'``: nearest-neighbour matching with
      replacement on the covariates ``x`` or on the estimated propensity score,
      with Abadie-Imbens standard errors (see ``docs/econometrics/teffects.md``).

    Parameters: ``data`` the table; ``y`` the outcome; ``treatment`` the
    treatment column (levels sorted, ``control`` names the control level,
    default the lowest); ``x`` the outcome-model covariates (the matching
    covariates for nnmatch); ``tx`` the treatment-model covariates (default:
    ``x``); ``estimand`` ``'ate'``, ``'atet'`` or ``'pomeans'``; ``omodel``
    ``'linear'`` (default), ``'logit'``, ``'probit'`` (outcome in [0, 1];
    fractional values give the quasi-likelihood model, Stata's flogit/fprobit)
    or ``'poisson'``; ``tmodel`` ``'logit'`` (default; multinomial logit for
    more than two levels) or ``'probit'``; ``pstolerance`` the overlap tolerance
    (default 1e-5: an estimated probability below it raises
    ``overlap_violation``, as Stata does); ``categorical`` covariates to
    treatment-code; ``intercept`` the constant of the outcome model; ``weights``
    with ``weight_type`` ``'fweight'`` or ``'pweight'`` (ra/ipw/ipwra/aipw);
    ``missing`` ``'raise'`` or ``'drop'``; ``alpha`` the test size. Matching
    options: ``neighbors`` (matches per observation, default 1), ``metric``
    (``'mahalanobis'`` default, ``'ivariance'``, ``'euclidean'``), ``biasadj``
    (covariates of the linear bias correction), ``caliper`` and
    ``vce_neighbors`` (default 2), ``ematch`` (exact-match columns),
    ``matching_vce='iid'`` (unadjusted nnmatch homoskedastic variance).
    ``tlevel`` chooses the ATET population, default the first non-control level.
    Matching accepts bounded fweight duplication (100,000 rows/2,000,000 entries).

    Covariance (z inference): ``'robust'`` (default) is the sandwich of the
    stacked estimating equations of the treatment model, the outcome models
    and the potential-outcome means, which accounts for the estimated nuisance
    models; ``'cluster'`` with ``cluster`` sums the scores by cluster. Neither
    applies a small-sample factor (Stata's teffects/gmm convention). For
    matching, ``'robust'`` is the Abadie-Imbens heteroskedasticity-robust
    estimator, with the Abadie-Imbens (2016) adjustment for the estimated
    propensity score under psmatch.

    The result's terms follow Stata: ``ATE:r1vs0.treat`` (or ``ATET:...``) and
    ``POmean:0.treat``, or ``POmeans:l.treat`` for every level; ``extra`` holds
    the auxiliary equations (``TME1``, ``OME0``, ``OME1`` with coefficients and
    standard errors), the potential-outcome means, the observations by level
    and the overlap diagnostics (estimated propensity scores by group);
    ``metrics`` the group sizes and the propensity-score range.

    Stata: ``teffects ra (y x1 x2) (treat)``, ``teffects ipw (y) (treat x1 x2,
    probit)``, ``teffects ipwra (y x1 x2, poisson) (treat x1 x2), atet``,
    ``teffects aipw (y x1 x2) (treat x1 x2)``, ``teffects nnmatch (y x1 x2)
    (treat), nneighbor(2) biasadj(x1)``, ``teffects psmatch (y) (treat x1 x2)``.

    Example::

        import openecon as oe
        ate = oe.teffects(data=df, y="bweight", treatment="mbsmoke",
                          x=["mage", "prenatal1", "mmarried"], method="aipw")
        print(ate.summary())
        att = oe.teffects(data=df, y="bweight", treatment="mbsmoke",
                          x=["mage", "prenatal1"], method="ipwra", estimand="atet")
        matched = oe.teffects(data=df, y="bweight", treatment="mbsmoke",
                              x=["mage", "prenatal1"], method="nnmatch", biasadj=["mage"])
    """
    from openecon.analysis import fit

    spec = make_spec(
        "teffects", outcome=y, predictors=column_list(x, "x"), intercept=intercept,
        categorical=column_list(categorical, "categorical"), covariance=covariance,
        cluster=cluster, weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"treatment": treatment, "tx": column_list(tx, "tx"),
                 "biasadj": column_list(biasadj, "biasadj"), "ematch": column_list(ematch, "ematch")},
        options={"method": method, "estimand": estimand, "omodel": omodel, "tmodel": tmodel,
                 "control": control, "neighbors": neighbors, "metric": metric, "caliper": caliper,
                 "vce_neighbors": vce_neighbors, "pstolerance": pstolerance,
                 "matching_vce": matching_vce, "tlevel": tlevel},
    )
    return fit(spec, data=data)
