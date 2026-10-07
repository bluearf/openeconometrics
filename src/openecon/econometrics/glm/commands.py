"""Stata's ``poisson``, ``cloglog`` and ``fracreg`` on top of the GLM core.

All three are single-index likelihood models with unit dispersion, fitted by
Newton-Raphson with the analytic observed Hessian (``glm.estimate``):

* ``poisson``: ``y ~ Poisson(mu)``, ``mu = exp(x'b + offset)``;
  ``ll = sum w [y ln mu - mu - lnG(y+1)]``.
* ``cloglog``: ``Pr(y = 1) = 1 - exp(-exp(x'b + offset))``.
* ``fracreg``: ``E[y|x] = G(x'b)`` for ``y`` in [0, 1] with ``G`` logistic or
  standard normal, estimated by Bernoulli quasi-maximum likelihood
  (Papke and Wooldridge 1996): ``ll = sum w [y ln G + (1-y) ln(1-G)]``. The
  quasi-likelihood is consistent for the mean whatever the distribution of
  ``y``, so the default covariance is the robust sandwich (Stata's default).

Reported as Stata does: z statistics; the model test is the likelihood-ratio
chi2 against the constant-only model (same offset and weights) under the
conventional covariance and the Wald chi2 of the slopes otherwise (``fracreg``
always reports Wald); McFadden's pseudo R-squared ``1 - ll/ll_0``; ``estat
ic`` information criteria. ``poisson`` adds the deviance and Pearson
goodness-of-fit tests of ``estat gof`` (chi2 with ``N - K`` degrees of
freedom).

Without a constant (``intercept=False``) there is no constant-only model: the
comparison model then has every coefficient at zero (``eta = offset``), which
is nested in the fitted one, and the test and ``extra['null_model']`` say so.
(Stata prints a Wald test and no pseudo R-squared in that case.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, lr_test, wald_test,
)
from openecon.econometrics.glm.common import (
    build_spec, check_pweights, count_outcome, likelihood_weights, linear_offset,
    require_terms, resolve_covariance, slope_indices,
)
from openecon.econometrics.glm.families import Binomial, Poisson, make_link
from openecon.econometrics.glm.glm import GlmFit, binomial_outcome, estimate, null_log_likelihood
from openecon.engines.distributions import chi2_sf
from openecon.models import ModelSpec, ResultBundle


def _fit(frame: ModelFrame, y: Tensor, family: Any, link_name: str, command: str,
         ) -> tuple[GlmFit, Tensor | None]:
    check_pweights(frame)
    weights = likelihood_weights(frame)
    offset = linear_offset(frame)
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    require_terms(design)
    link = kernel_call(make_link, link_name)
    return estimate(frame, design, y, family, link, weights=weights, offset=offset,
                    command=command), offset


def _report(frame: ModelFrame, fitted: GlmFit, y: Tensor, offset: Tensor | None, *,
            command: str, wald_only: bool = False, gof: bool = False,
            extra: dict[str, Any] | None = None) -> ResultBundle:
    """Covariance, model test, pseudo R-squared and result assembly shared by the commands."""
    spec, design = frame.spec, fitted.design
    k, nobs = len(design.terms), fitted.nobs
    covariance, info = fitted.covariance(frame)
    log_likelihood = fitted.log_likelihood()
    null = null_log_likelihood(fitted, frame, y, offset, intercept=spec.intercept,
                               command=command)
    slopes = slope_indices(design.terms)
    if spec.covariance == "nonrobust" and not wald_only:
        against = "the constant-only model" if spec.intercept \
            else "the model with every coefficient zero"
        model = lr_test(log_likelihood, null, len(slopes), label=f"LR chi2 test against {against}")
    else:
        model = wald_test(fitted.beta, covariance, slopes, label="Wald chi2 test of the slopes")
    criteria = information_criteria(log_likelihood, k, nobs)
    metrics: dict[str, Any] = {
        "log_likelihood": log_likelihood,
        "pseudo_r_squared": 1 - log_likelihood / null if null != 0 else None,
        "aic": criteria["aic"], "bic": criteria["bic"],
    }
    tests: dict[str, Any] = {"model": model}
    if gof:
        metrics.update({"deviance": fitted.deviance, "pearson": fitted.pearson})
        for name, value, label in (("gof_deviance", fitted.deviance, "Deviance goodness of fit"),
                                   ("gof_pearson", fitted.pearson, "Pearson goodness of fit")):
            df = fitted.df_resid
            tests[name] = {"statistic": value, "df": df,
                           "p_value": kernel_call(chi2_sf, value, df) if df > 0 else None,
                           "distribution": "chi2", "label": label}
    metrics["df_resid"] = fitted.df_resid
    record = {"null_log_likelihood": null, "link": fitted.link.name,
              "null_model": "constant only, same offset and weights" if spec.intercept
              else "all coefficients zero (no constant in the model)", **(extra or {})}
    return build_result(
        frame, terms=design.terms, params=fitted.beta, covariance=covariance, use_t=False,
        df_resid=fitted.df_resid, metrics=metrics, fitted=fitted.fitted(), nobs=nobs,
        inference=info, tests=tests, extra=record, categories=design.categories,
        solver=fitted.solver,
        solver_diagnostics={"converged": True, "iterations": fitted.iterations},
        optimizer=fitted.optimizer)


# ---- poisson -------------------------------------------------------------------------------


def fit_poisson(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``poisson``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    y = frame.numeric(spec.outcome)
    count_outcome(frame, y, "poisson")
    if not bool((y > 0).any()):
        raise AnalysisError("constant_outcome", f"'{spec.outcome}' is zero in every observation; "
                            "the Poisson mean is not identified.")
    fitted, offset = _fit(frame, y, Poisson(), "log", "poisson")
    return _report(frame, fitted, y, offset, command="poisson", gof=True)


def poisson(*, data: Any, y: str, x: Sequence[str], covariance: str | None = None,
            cluster: str | Sequence[str] | None = None, weights: str | None = None,
            weight_type: str | None = None, offset: str | None = None,
            exposure: str | None = None, categorical: Sequence[str] | None = None,
            intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Poisson regression, Stata's ``poisson``.

    Model
        ``y ~ Poisson(mu)`` with ``mu = exp(x'b + offset)``. The log likelihood
        ``sum w [y ln mu - mu - lnG(y + 1)]`` is globally concave and is maximized by
        Newton-Raphson with the analytic score ``X'w(y - mu)`` and Hessian
        ``-X' diag(w mu) X``. Because the estimator only needs ``E[y|x] = mu`` to be
        right, it is also the Poisson pseudo-maximum-likelihood estimator of an
        exponential mean for any nonnegative outcome (use a robust covariance then).

    Parameters
        data, y, x: data, outcome (nonnegative; non-integer values are allowed with a
            recorded note, as in Stata) and the list of regressors (``x=[]`` fits the
            constant-only model, e.g. an overall rate with ``exposure``).
        covariance: ``'nonrobust'`` (default, inverse observed information),
            ``'opg'``, ``'robust'`` (sandwich times ``N/(N-1)``), ``'cluster'``
            (``G/(G-1)``; one or two cluster columns).
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        offset / exposure: ``offset`` enters with coefficient 1; ``exposure`` enters as
            ``ln(exposure)`` (rates per unit of exposure). Mutually exclusive.
        categorical, intercept, missing, alpha: as in :func:`glm`.

    Result
        z statistics. ``metrics``: log_likelihood, pseudo_r_squared (McFadden,
        ``1 - ll/ll_0`` with ``ll_0`` the constant-only model with the same offset and
        weights), aic, bic, deviance, pearson, df_resid. ``tests['model']``: LR
        chi2(k) against the constant-only model (``nonrobust``) or the Wald chi2(k)
        of the slopes (other covariances), as Stata prints. ``tests['gof_deviance']``
        and ``tests['gof_pearson']``: ``estat gof`` (chi2 with ``N - K`` df; a small
        p-value signals overdispersion or misspecification - consider
        :func:`nbreg`). Incidence-rate ratios are ``exp(estimate)``.

    Stata
        ``poisson deaths smokes i.agecat, exposure(pyears) vce(robust)`` is
        ``oe.poisson(data=df, y='deaths', x=['smokes', 'agecat'], categorical=['agecat'],
        exposure='pyears', covariance='robust')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=500)})
        >>> df["y"] = rng.poisson(np.exp(0.5 + 0.4 * df.x))
        >>> print(oe.poisson(data=df, y="y", x=["x"]).summary())
    """
    spec = build_spec(
        "poisson", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"offset": offset, "exposure": exposure})
    return fit(spec, data=data)


# ---- cloglog -------------------------------------------------------------------------------


def fit_cloglog(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``cloglog``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    y = binomial_outcome(frame, frame.numeric(spec.outcome), None, "cloglog")
    fitted, offset = _fit(frame, y, Binomial(), "cloglog", "cloglog")
    user = fitted.weights.user
    counts = {"zero_outcomes": float((user * (y == 0)).sum()),
              "nonzero_outcomes": float((user * (y == 1)).sum())}
    return _report(frame, fitted, y, offset, command="cloglog", extra=counts)


def cloglog(*, data: Any, y: str, x: Sequence[str], covariance: str | None = None,
            cluster: str | Sequence[str] | None = None, weights: str | None = None,
            weight_type: str | None = None, offset: str | None = None,
            categorical: Sequence[str] | None = None, intercept: bool = True,
            missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Complementary log-log regression for a binary outcome, Stata's ``cloglog``.

    Model
        ``Pr(y = 1 | x) = 1 - exp(-exp(x'b + offset))``: the binary model implied by a
        proportional-hazards (grouped duration) process, asymmetric around 1/2 unlike
        logit and probit. ``exp(b)`` is a hazard ratio. Maximum likelihood by
        Newton-Raphson with the analytic observed Hessian (the link is not canonical,
        so the observed and expected information differ).

    Parameters
        data, y, x: data, 0/1 outcome and the list of regressors (may be empty).
        covariance: ``'nonrobust'`` (default, OIM), ``'opg'``, ``'robust'``
            (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``).
        weights, weight_type: ``'fweight'``, ``'pweight'``, ``'iweight'`` (as Stata).
        offset: column added to the linear predictor.
        categorical, intercept, cluster, missing, alpha: as in :func:`glm`.

    Result
        z statistics. ``metrics``: log_likelihood, pseudo_r_squared (McFadden), aic,
        bic, df_resid. ``tests['model']``: LR chi2 (``nonrobust``) or Wald chi2.
        ``extra``: zero_outcomes, nonzero_outcomes, null_log_likelihood. Complete or
        quasi-complete separation raises ``separation_detected``.

    Stata
        ``cloglog died age i.drug, vce(cluster clinic)`` is
        ``oe.cloglog(data=df, y='died', x=['age', 'drug'], categorical=['drug'],
        cluster='clinic')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=500)})
        >>> df["y"] = rng.binomial(1, 1 - np.exp(-np.exp(-0.5 + 0.8 * df.x)))
        >>> print(oe.cloglog(data=df, y="y", x=["x"]).summary())
    """
    spec = build_spec(
        "cloglog", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"offset": offset})
    return fit(spec, data=data)


# ---- fracreg -------------------------------------------------------------------------------


def fit_fracreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``fracreg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    y = frame.numeric(spec.outcome)
    if bool((y < 0).any()) or bool((y > 1).any()):
        raise AnalysisError("invalid_fractional_outcome", f"fracreg needs '{spec.outcome}' in "
                            "[0, 1]; rescale percentages by 100 or use another model.")
    if not bool((y != y[0]).any()):
        raise AnalysisError("constant_outcome", f"'{spec.outcome}' does not vary; the model "
                            "cannot be estimated.")
    link = frame.option("link")
    fitted, offset = _fit(frame, y, Binomial(combinatorial=False), link, "fracreg")
    return _report(frame, fitted, y, offset, command="fracreg", wald_only=True,
                   extra={"quasi_likelihood": True,
                          "log_likelihood_note": "Bernoulli log pseudolikelihood"})


def fracreg(*, data: Any, y: str, x: Sequence[str], link: str = "logit",
            covariance: str | None = None, cluster: str | Sequence[str] | None = None,
            weights: str | None = None, weight_type: str | None = None,
            categorical: Sequence[str] | None = None, intercept: bool = True,
            missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Fractional response regression, Stata's ``fracreg logit`` / ``fracreg probit``.

    Model
        ``E[y | x] = G(x'b)`` for an outcome in ``[0, 1]`` (0 and 1 included), with
        ``G`` the logistic (``link='logit'``) or standard normal (``link='probit'``)
        cdf. The coefficients maximize the Bernoulli quasi-log-likelihood
        ``sum w [y ln G + (1 - y) ln(1 - G)]`` (Papke and Wooldridge 1996), which is
        consistent for the conditional mean without any distributional assumption.
        Newton-Raphson with analytic derivatives.

    Parameters
        data, y, x: data, fractional outcome and the list of regressors (may be empty).
        link: ``'logit'`` (default) or ``'probit'``.
        covariance: ``'robust'`` (DEFAULT, as Stata: sandwich times ``N/(N-1)``; the
            information-matrix equality does not hold for a quasi-likelihood),
            ``'cluster'`` (``G/(G-1)``), ``'nonrobust'`` (OIM), ``'opg'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'``, ``'iweight'``.
        categorical, intercept, cluster, missing, alpha: as in :func:`glm`.

    Result
        z statistics. ``metrics``: log_likelihood (the log PSEUDOlikelihood; see
        ``extra['quasi_likelihood']``), pseudo_r_squared (``1 - ll/ll_0``), aic, bic,
        df_resid. ``tests['model']``: Wald chi2(k) of the slopes. For an outcome
        strictly inside (0, 1) with a modelled precision see :func:`betareg`.

    Stata
        ``fracreg probit prate mrate c.age, vce(cluster firm)`` is
        ``oe.fracreg(data=df, y='prate', x=['mrate', 'age'], link='probit', cluster='firm')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=500)})
        >>> df["y"] = np.clip(1 / (1 + np.exp(-0.3 - 0.6 * df.x)) + rng.normal(0, 0.2, 500), 0, 1)
        >>> print(oe.fracreg(data=df, y="y", x=["x"]).summary())
    """
    spec = build_spec(
        "fracreg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=covariance, cluster=cluster, weights=weights, weight_type=weight_type,
        missing=missing, alpha=alpha, options={"link": link})
    return fit(spec, data=data)
