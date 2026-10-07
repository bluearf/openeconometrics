"""Likelihood-ratio tests and information-criterion tables (Stata's lrtest and estat ic).

Both procedures read what a fitted result records: the maximized log
likelihood ``metrics['log_likelihood']``, the number of estimated parameters
(the reported coefficients, Stata's ``e(rank)``), the number of observations
``nobs`` (Stata's ``e(N)``) and the estimation rows ``sample_positions``.
Nothing is refitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import table
from openecon.econometrics.postest.common import (
    LIKELIHOOD_COVARIANCES, is_suest, label, log_likelihood, null_log_likelihood,
    parameter_count, require_result, result_covariance, sample_key,
)
from openecon.engines.distributions import chi2_sf
from openecon.models import ResultBundle

# Roles holding further columns that define the outcome (second outcomes, the selection
# indicator, interval upper bounds, failure and entry of survival data, binomial trials).
# The systems family's 'system' role lists regressors too: its outcomes are read from the
# equations option; a VAR's 'system' role lists its endogenous variables.
_OUTCOME_ROLES = ("outcome2", "outcomes", "select", "upper", "failure", "entry", "trials")
# Fragments of ancillary parameter names that measure a variance or overdispersion,
# whose null value (zero) lies on the boundary of the parameter space.
_BOUNDARY_MARKERS = ("alpha", "sigma", "sig2", "var", "lnsig", "delta", "theta")


def _incompatible(message: str) -> AnalysisError:
    return AnalysisError("incompatible_models", message)


def _outcomes(result: ResultBundle) -> tuple[str, ...]:
    """The columns that define the modelled outcome(s) of a fit."""
    spec = result.spec
    found = [spec.outcome]
    found += [f"{role}={','.join(registry.role_columns(spec, role))}" for role in _OUTCOME_ROLES
              if spec.columns.get(role) is not None]
    equations = spec.options.get("equations")
    if isinstance(equations, list):
        found += [f"equation={item.get('y')}" for item in equations if isinstance(item, dict)]
    elif registry.get(spec.estimator).family == "var":
        found += [f"system={name}" for name in registry.role_columns(spec, "system")]
    return tuple(found)


def _unreported_scale(full: ResultBundle, restricted: ResultBundle) -> bool:
    """Whether the restricted fit hides a residual-variance parameter the full fit reports."""
    if restricted.spec.estimator not in _UNREPORTED_SCALE:
        return False
    names = [c.term.split(":")[-1].lower() for c in full.coefficients]
    return any(name.startswith("/") and ("sig" in name or "residual" in name)
               for name in names)


def _reml(result: ResultBundle) -> bool:
    """Whether the log likelihood is a restricted (REML) likelihood (``mixed, reml``)."""
    return result.extra.get("method") == "reml"


def _fixed_terms(result: ResultBundle) -> list[str]:
    return [c.term for c in result.coefficients if not c.term.split(":")[-1].startswith("/")]


# Least-squares fits whose error variance is estimated but not reported as a coefficient.
_UNREPORTED_SCALE = ("ols", "glm")
# Residual-variance terms: present (unreported) in a least-squares restricted fit too.
_RESIDUAL_TERMS = ("/sigma", "/lnsigma", "/sigma_e", "/lnsig_e", "/var(residual)")


def _boundary_terms(full: ResultBundle, restricted: ResultBundle) -> list[str]:
    restricted_terms = {c.term for c in restricted.coefficients}
    hidden_scale = restricted.spec.estimator in _UNREPORTED_SCALE
    found = []
    for c in full.coefficients:
        name = c.term.split(":")[-1].lower()
        if hidden_scale and name in _RESIDUAL_TERMS:
            continue
        if c.term not in restricted_terms and name.startswith("/") \
                and any(marker in name for marker in _BOUNDARY_MARKERS):
            found.append(c.term)
    return found


def lrtest(full: ResultBundle, restricted: ResultBundle, *, df: int | None = None,
           force: bool = False):
    """Likelihood-ratio test of a restricted model nested in a full model (Stata's ``lrtest``).

    Model and statistic
    -------------------
    Both models are fitted by maximum likelihood on the same observations; the
    restricted model is the full model with ``df`` parameters constrained
    (usually set to zero). Under the null hypothesis that the restrictions hold

        LR = 2 (ll_full - ll_restricted)  ~  chi2(df),

    with ``df`` the difference in the number of estimated parameters (the
    reported coefficients, Stata's ``e(rank)``; terms omitted for collinearity
    do not count) unless ``df`` is given. Nothing is refitted: the function
    reads ``metrics['log_likelihood']``, the coefficient lists and the
    estimation rows of the two results.

    Parameters
    ----------
    full, restricted : ResultBundle
        The unrestricted and the restricted fit, in that order.
    df : int, optional
        Degrees of freedom to use instead of the parameter-count difference
        (Stata's ``df(#)``), e.g. when restrictions are imposed by constraints
        rather than by dropping terms.
    force : bool
        Stata's ``force``: compute the test although (a) a fit carries a robust,
        cluster or other non-likelihood covariance or sampling weights (its
        log likelihood is then a pseudo-likelihood and LR is not chi2), or (b)
        the two results come from different estimators (e.g. ``poisson``
        against ``nbreg``). The note records that the test was forced.

    Refusals (``AnalysisError('incompatible_models')``)
    ---------------------------------------------------
    - either result has no log likelihood (least-squares, GMM or
      nonparametric fits) or is a combined ``oe.suest`` result;
    - the estimation samples differ (number of observations, number of rows in
      the data or the set of estimation rows): refit both models on the same
      rows; ``force`` does not override this;
    - the outcomes differ (the outcome column and the columns that define it:
      a second outcome, the selection indicator, an interval upper bound,
      failure/entry times, binomial trials, the dependent variable of every
      system equation; regressors may differ);
    - a fit uses a robust/cluster/HAC covariance or pweights, or the
      estimators differ, without ``force``;
    - the full model does not have more parameters than the restricted one
      (swap the arguments or give ``df``);
    - the restricted model has the larger log likelihood beyond rounding
      (relative tolerance 1e-7): the models are not nested as given;
    - one model is a REML fit (``mixed(method='reml')``) and the other is not
      (cannot be forced), or both are REML fits with different fixed effects
      (the REML criterion depends on the fixed-effects design; ``force``
      computes it anyway).

    Returns
    -------
    A one-row table (``openecon.frame.DataFrame``) with columns
    ``statistic``, ``df``, ``p_value``, ``ll_full`` and ``ll_restricted``.
    ``attrs`` carry ``distribution`` ('chi2'), ``label``, ``k_full``,
    ``k_restricted``, ``nobs``, ``df_source``, ``forced``, ``boundary_terms``,
    ``p_value_chibar2`` (half the chi2(1) tail when exactly one boundary
    parameter is tested, as Stata reports for ``alpha = 0`` after ``nbreg``)
    and ``notes`` (the nesting assumption and every caveat).

    Boundary note: when the full model has extra variance or overdispersion
    parameters (``/lnalpha``, ``/sigma_u`` ...) their null value lies on the
    boundary of the parameter space; the chi2 reference is then conservative
    and the mixture chibar2 applies (Self and Liang 1987). When the restricted
    model is an ``ols``/``glm`` fit (error variance estimated but not reported)
    and the full model reports a residual-variance term (``mixed`` against
    ``ols``), the parameter-count difference overstates df by one, as Stata's
    e(rank) rule does; a note says so and ``df=`` should be given.

    Stata: ``lrtest full restricted [, df(#) force]``. EViews: the
    "Redundant Variables - Likelihood Ratio" test. SPSS: the change in -2 log
    likelihood between blocks.

    Example::

        full = oe.logit(data=df, y="union", x=["age", "grade", "south"])
        small = oe.logit(data=df, y="union", x=["age"])
        oe.lrtest(full, small)
    """
    full = require_result(full, "full model")
    restricted = require_result(restricted, "restricted model")
    if not isinstance(force, bool):
        raise AnalysisError("invalid_spec", "force must be True or False.")
    notes: list[str] = []
    for role, bundle in (("full", full), ("restricted", restricted)):
        if is_suest(bundle):
            raise _incompatible(f"The {role} model is a combined suest result; lrtest needs two "
                                "individual likelihood fits.")
        if log_likelihood(bundle) is None:
            raise _incompatible(f"The {role} model ({label(bundle)}) reports no log likelihood, "
                                "so a likelihood-ratio test is not defined. Use a Wald test of its "
                                "coefficients instead.")
    if _outcomes(full) != _outcomes(restricted):
        raise _incompatible(f"The models explain different outcomes ({', '.join(_outcomes(full))} "
                            f"and {', '.join(_outcomes(restricted))}); an LR test compares fits of "
                            "the same outcome.")
    if (full.nobs != restricted.nobs or full.nobs_original != restricted.nobs_original
            or sample_key(full) != sample_key(restricted)):
        raise _incompatible(
            f"The models were fitted on different estimation samples (N = {full.nobs} and "
            f"N = {restricted.nobs}, or different rows). Refit both on the same observations, "
            "for example by dropping rows with missing values in any variable of the full model "
            "before fitting either.")
    forced = False
    for role, bundle in (("full", full), ("restricted", restricted)):
        covariance = result_covariance(bundle)
        pweighted = bundle.spec.weight_type == "pweight"
        if covariance in LIKELIHOOD_COVARIANCES and not pweighted:
            continue
        reason = "sampling weights (pweights)" if pweighted else f"covariance='{covariance}'"
        if not force:
            raise _incompatible(
                f"The {role} model uses {reason}: its log likelihood is a pseudo-likelihood and "
                "the LR statistic is not chi2 (Stata's lrtest refuses vce(robust), vce(cluster) "
                "and pweights). Refit with the conventional covariance, test the coefficients "
                "with a Wald test, or pass force=True.")
        forced = True
        notes.append(f"forced: the {role} model uses {reason}; the chi2 reference distribution "
                     "is not justified.")
    if _reml(full) != _reml(restricted):
        raise _incompatible("One model was fitted by REML and the other by ML; their log "
                            "likelihoods are different criteria. Refit both with the same method.")
    if _reml(full) and _fixed_terms(full) != _fixed_terms(restricted):
        if not force:
            raise _incompatible(
                "Both models were fitted by REML but their fixed effects differ. The REML "
                "likelihood depends on the fixed-effects design, so LR tests of fixed effects are "
                "not valid under REML (Stata refuses them as well); refit both with "
                "method='ml'.")
        forced = True
        notes.append("forced: REML fits with different fixed effects; the REML criteria are not "
                     "comparable.")
    if full.spec.estimator != restricted.spec.estimator:
        if not force:
            raise _incompatible(
                f"The models come from different estimators ({full.spec.estimator} and "
                f"{restricted.spec.estimator}). The test is valid only when one likelihood nests "
                "the other (e.g. poisson in nbreg); pass force=True to compute it.")
        forced = True
        notes.append(f"forced: different estimators ({full.spec.estimator}, "
                     f"{restricted.spec.estimator}); nesting of the likelihoods is assumed.")
    k_full, k_restricted = parameter_count(full), parameter_count(restricted)
    if df is None:
        df_value, df_source = k_full - k_restricted, "difference in the number of parameters"
        if df_value <= 0:
            raise _incompatible(
                f"The full model has {k_full} parameters and the restricted model {k_restricted}; "
                "the full model must have more. Swap the arguments, or give df= when the "
                "restrictions are imposed without dropping parameters.")
    else:
        if isinstance(df, bool) or not isinstance(df, int) or df < 1:
            raise AnalysisError("invalid_spec", "df must be a positive integer.")
        df_value, df_source = df, "given"
        if df != k_full - k_restricted:
            notes.append(f"df = {df} was given; the parameter counts differ by "
                         f"{k_full - k_restricted}.")
    ll_full, ll_restricted = log_likelihood(full), log_likelihood(restricted)
    assert ll_full is not None and ll_restricted is not None
    tolerance = 1e-7 * max(1.0, abs(ll_full))
    if ll_restricted - ll_full > tolerance:
        raise _incompatible(
            f"The restricted model has the larger log likelihood ({ll_restricted:.6g} > "
            f"{ll_full:.6g}), so it is not nested in the full model as given. Check the order of "
            "the arguments and that both fits converged.")
    statistic = max(0.0, 2.0 * (ll_full - ll_restricted))
    p_value = chi2_sf(statistic, df_value)
    restricted_terms = {c.term for c in restricted.coefficients}
    full_terms = {c.term for c in full.coefficients}
    notes.insert(0, "Assumption: the restricted model is nested in the full model.")
    if not restricted_terms <= full_terms:
        notes.append("The restricted model has terms the full model lacks; the test is valid "
                     "only if it is nevertheless nested (e.g. a reparameterization), and df "
                     "should be checked.")
    boundary = _boundary_terms(full, restricted)
    chibar = None
    if df is None and _unreported_scale(full, restricted):
        notes.append(f"{restricted.spec.estimator} estimates its error variance without "
                     "reporting it as a coefficient, while the full model reports one "
                     "(e.g. /sigma_e or /var(Residual)); the parameter-count difference (Stata's "
                     "e(rank) rule) then overstates df by one. Give df= explicitly.")
    if boundary:
        notes.append(f"The test involves {', '.join(boundary)}, whose null value lies on the "
                     "boundary of the parameter space; the chi2 reference is conservative "
                     "(chibar2 mixture).")
        if df_value == 1:
            chibar = 0.5 * p_value
    if forced:
        notes.append("The test was forced; interpret the p-value with care.")
    return table(
        [[statistic, df_value, p_value, ll_full, ll_restricted]],
        columns=["statistic", "df", "p_value", "ll_full", "ll_restricted"],
        index=["LR chi2"], distribution="chi2", label="Likelihood-ratio test",
        k_full=k_full, k_restricted=k_restricted, nobs=full.nobs, df_source=df_source,
        forced=forced, boundary_terms=boundary, p_value_chibar2=chibar,
        full_model=label(full), restricted_model=label(restricted), notes=notes,
        note=" ".join(notes),
    )


def estat_ic(*results: ResultBundle, names: Sequence[str] | None = None):
    """Information criteria of fitted models (Stata's ``estat ic`` / ``estimates stats``).

    For each result

        AIC = -2 ll + 2 k,        BIC = -2 ll + k ln N,

    with ``ll`` the maximized log likelihood (``metrics['log_likelihood']``),
    ``k`` the number of estimated parameters (the reported coefficients,
    Stata's ``e(rank)``: ancillary parameters such as ``/lnalpha`` or
    ``/sigma`` count, terms omitted for collinearity do not) and ``N`` the
    number of observations ``nobs`` (Stata's ``e(N)``; the sum of frequency
    weights for fweighted fits). Smaller values are better; compare models
    only on the same estimation sample and the same outcome.

    Parameters
    ----------
    *results : ResultBundle
        One or more fitted models that report a log likelihood.
    names : sequence of str, optional
        Row labels; default ``"1: logit"``, ``"2: probit"`` ...

    Returns
    -------
    A table with one row per model and columns ``model``, ``nobs``,
    ``ll_null`` (the constant-only log likelihood when the result reports it,
    Stata's ``ll(null)``; otherwise empty), ``ll``, ``df``, ``aic`` and
    ``bic``. ``attrs['notes']`` warns when the samples differ.

    Raises ``AnalysisError('missing_log_likelihood')`` for a result without a
    log likelihood (least squares with a non-likelihood objective, GMM,
    nonparametric fits) and ``invalid_spec`` for bad ``names``.

    Stata: ``estat ic`` after one model, ``estimates stats m1 m2`` for several.
    EViews reports the same criteria divided by N (Akaike info criterion
    ``-2 ll/N + 2k/N``); multiply EViews' values by N to compare.

    Example::

        m1 = oe.poisson(data=df, y="visits", x=["age", "income"])
        m2 = oe.nbreg(data=df, y="visits", x=["age", "income"])
        oe.estat_ic(m1, m2, names=["poisson", "nbreg"])
    """
    if not results:
        raise AnalysisError("invalid_spec", "estat_ic needs at least one fitted result.")
    bundles = [require_result(value, f"result {i + 1}") for i, value in enumerate(results)]
    if names is None:
        labels = [f"{i + 1}: {bundle.spec.estimator}" for i, bundle in enumerate(bundles)]
    else:
        names = None if isinstance(names, str) else list(names)
        if names is None or len(names) != len(bundles):
            raise AnalysisError("invalid_spec", "names must be a list with one label per result.")
        labels = [str(name) for name in names]
        if len(set(labels)) != len(labels):
            raise AnalysisError("invalid_spec", "names must be distinct.")
    rows = []
    for name, bundle in zip(labels, bundles, strict=True):
        ll = log_likelihood(bundle)
        if ll is None or is_suest(bundle):
            raise AnalysisError("missing_log_likelihood", f"Model {name!r} ({label(bundle)}) "
                                "reports no log likelihood, so AIC and BIC are not defined for it.")
        k, n = parameter_count(bundle), bundle.nobs
        rows.append([name, n, null_log_likelihood(bundle), ll, k, -2 * ll + 2 * k,
                     -2 * ll + k * math.log(n)])
    notes = ["AIC = -2 ll + 2 k and BIC = -2 ll + k ln(N) (Stata's convention); N is the "
             "number of observations."]
    if len({bundle.nobs for bundle in bundles}) > 1 or \
            len({sample_key(bundle) for bundle in bundles}) > 1:
        notes.append("The models use different estimation samples; their information criteria "
                     "are not comparable.")
    if len({_outcomes(bundle) for bundle in bundles}) > 1:
        notes.append("The models explain different outcomes; their information criteria are not "
                     "comparable.")
    if any(_reml(bundle) for bundle in bundles):
        notes.append("REML log likelihoods are comparable only between models with the same "
                     "fixed effects (and are never comparable with ML fits).")
    return table(rows, columns=["model", "nobs", "ll_null", "ll", "df", "aic", "bic"],
                 title="Akaike and Bayesian information criteria", notes=notes)
