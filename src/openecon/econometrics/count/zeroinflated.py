"""Public commands ``oe.zip`` and ``oe.zinb`` (zero-inflated count regression).

The estimation code is in ``zeroinflated_fit``; this module holds the registry
entry points and the Stata-style convenience functions. (The module defines a
function named ``zip``, so the builtin of that name is not used here.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openecon.analysis import fit
from openecon.econometrics.core import column_list
from openecon.econometrics.count.zeroinflated_fit import fit_zinb, fit_zip
from openecon.econometrics.glm.common import build_spec, resolve_covariance
from openecon.models import ModelSpec, ResultBundle

__all__ = ["fit_zinb", "fit_zip", "zinb", "zip"]


def _spec(estimator: str, *, y: str, x: Sequence[str], inflate: Sequence[str],
          inflate_link: str, covariance: str | None, cluster: Any, weights: str | None,
          weight_type: str | None, offset: str | None, exposure: str | None,
          categorical: Sequence[str] | None, intercept: bool, missing: str,
          alpha: float) -> ModelSpec:
    return build_spec(
        estimator, outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"inflate": column_list(inflate, "inflate"), "offset": offset,
                 "exposure": exposure},
        options={"inflate_link": inflate_link})


def zip(*, data: Any, y: str, x: Sequence[str], inflate: Sequence[str],  # noqa: A001
        inflate_link: str = "logit", covariance: str | None = None,
        cluster: str | Sequence[str] | None = None, weights: str | None = None,
        weight_type: str | None = None, offset: str | None = None,
        exposure: str | None = None, categorical: Sequence[str] | None = None,
        intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Zero-inflated Poisson regression, Stata's ``zip``.

    Model
        With probability ``F(z'g)`` the outcome is a structural ("excess") zero;
        otherwise it is Poisson with mean ``mu = exp(x'b + offset)``:

            Pr(y = 0) = F + (1 - F) exp(-mu),
            Pr(y = k) = (1 - F) exp(-mu) mu^k / k!,   k = 1, 2, ...

        ``F`` is the logistic cdf (default) or the standard normal cdf. The log
        likelihood ``sum w ln Pr(y_i)`` is maximized over ``(b, g)`` jointly by
        Newton-Raphson with analytic score and Hessian (the zero mixture is a
        log-sum-exp), starting from the Poisson estimates and a binary fit of
        ``1[y = 0]``. ``E[y|x, z] = (1 - F) mu``.

    Parameters
        data: the data set (pandas DataFrame, dict of columns or row records).
        y: nonnegative count outcome with at least one zero.
        x: list of regressors of the count equation (``[]``: constant only).
        inflate: list of regressors of the inflation equation, which always has a
            constant; ``[]`` is Stata's ``inflate(_cons)`` (one inflation probability
            for everyone). Terms are reported as ``inflate:<name>``.
        inflate_link: ``'logit'`` (default) or ``'probit'`` (Stata's ``probit`` option).
        covariance: ``'nonrobust'`` (default: inverse observed information),
            ``'opg'``, ``'robust'`` (sandwich times ``N/(N-1)``), ``'cluster'``
            (``G/(G-1)``; one or two cluster columns).
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'`` (replicates rows), ``'aweight'`` (rescaled
            to sum to N), ``'iweight'``, ``'pweight'`` (robust by default).
        offset / exposure: ``offset`` enters the count equation with coefficient 1;
            ``exposure`` enters as ``ln(exposure)``. Mutually exclusive.
        categorical: columns of ``x`` or ``inflate`` to expand into indicator terms.
        intercept: include the constant of the count equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing inputs.
        alpha: significance level of the confidence intervals (default 0.05).

    Result
        z statistics. Terms: the count equation (equation = the outcome), then
        ``inflate:<name>`` (equation ``inflate``): a positive inflation coefficient
        raises the probability of an excess zero. ``metrics``: log_likelihood, aic,
        bic, n_zero_observations. ``tests['model']``: LR chi2(k) that the count
        equation's slopes are zero (the comparison model keeps the inflation equation)
        under ``nonrobust``, the Wald chi2 otherwise. ``tests['vuong']``: Vuong test
        against the Poisson model, ``V = sqrt(N) mean(m)/sd(m)`` with
        ``m_i = ln f_zip(y_i) - ln f_poisson(y_i)`` and the upper-tail p-value;
        ``extra['vuong']`` holds the AIC- and BIC-corrected versions. ``extra`` also
        has the Poisson and constant-only log likelihoods and the mean fitted
        inflation probability. The chart sample (``predictions``) uses
        ``E[y] = (1 - F) mu``.

        Data without excess zeros have no interior maximum (``boundary_solution``:
        use :func:`poisson`); an outcome without zeros raises ``no_zero_outcomes``;
        a regressor that predicts zeros perfectly - in the inflation equation, or
        a count regressor that singles out observations whose outcomes are all
        zero - raises ``separation_detected``.

    Stata
        ``zip count persons livebait, inflate(child camper) vuong`` is
        ``oe.zip(data=df, y='count', x=['persons', 'livebait'], inflate=['child', 'camper'])``;
        ``inflate(_cons)`` is ``inflate=[]`` and the ``probit`` option is
        ``inflate_link='probit'``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=2000), "z": rng.normal(size=2000)})
        >>> excess = rng.uniform(size=2000) < 1 / (1 + np.exp(0.5 - df.z))
        >>> df["y"] = np.where(excess, 0, rng.poisson(np.exp(0.8 + 0.4 * df.x)))
        >>> print(oe.zip(data=df, y="y", x=["x"], inflate=["z"]).summary())
    """
    spec = _spec("zip", y=y, x=x, inflate=inflate, inflate_link=inflate_link,
                 covariance=covariance, cluster=cluster, weights=weights,
                 weight_type=weight_type, offset=offset, exposure=exposure,
                 categorical=categorical, intercept=intercept, missing=missing, alpha=alpha)
    return fit(spec, data=data)


def zinb(*, data: Any, y: str, x: Sequence[str], inflate: Sequence[str],
         inflate_link: str = "logit", covariance: str | None = None,
         cluster: str | Sequence[str] | None = None, weights: str | None = None,
         weight_type: str | None = None, offset: str | None = None,
         exposure: str | None = None, categorical: Sequence[str] | None = None,
         intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Zero-inflated negative binomial regression, Stata's ``zinb``.

    Model
        As :func:`zip` with the Poisson replaced by the negative binomial (NB2)
        distribution with mean ``mu = exp(x'b + offset)`` and variance
        ``mu (1 + alpha mu)``; with ``m = 1/alpha``,

            f(k) = G(k+m) / (G(m) k!) (1 + alpha mu)^-m (alpha mu / (1 + alpha mu))^k,
            Pr(y = 0) = F + (1 - F) (1 + alpha mu)^-m,   Pr(y = k) = (1 - F) f(k).

        ``(b, g, ln alpha)`` are estimated jointly by Newton-Raphson with analytic
        derivatives, starting from the zero-inflated Poisson estimates and a
        dispersion taken from the plain negative binomial fit.

    Parameters
        data: the data set (pandas DataFrame, dict of columns or row records).
        y: nonnegative count outcome with at least one zero.
        x: list of regressors of the count equation (``[]``: constant only).
        inflate: list of regressors of the inflation equation, which always has a
            constant; ``[]`` is Stata's ``inflate(_cons)``.
        inflate_link: ``'logit'`` (default) or ``'probit'``.
        covariance: ``'nonrobust'`` (default: inverse observed information of all
            parameters), ``'opg'``, ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'``
            (``G/(G-1)``; one or two cluster columns).
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'iweight'``,
            ``'pweight'`` (robust by default).
        offset / exposure: offset of the count equation, or exposure entering as
            ``ln(exposure)``. Mutually exclusive.
        categorical: columns of ``x`` or ``inflate`` to expand into indicator terms.
        intercept: include the constant of the count equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing inputs.
        alpha: significance level of the confidence intervals (default 0.05).

    Result
        z statistics. Terms: the count equation, ``inflate:<name>`` and the ancillary
        ``/lnalpha``. ``metrics``: log_likelihood, aic, bic, n_zero_observations,
        alpha. ``extra['alpha']``: alpha with its delta-method standard error and the
        confidence interval exponentiated from ``ln alpha``. ``tests['model']``: LR
        chi2 of the count equation's slopes (``nonrobust``) or Wald chi2.
        ``tests['alpha']``: LR test of ``alpha = 0`` against :func:`zip`
        (``chibar2(01)``: half the chi2(1) tail; likelihood-based covariances only).
        ``tests['vuong']``: Vuong test against the negative binomial model without
        inflation, with the AIC/BIC-corrected statistics in ``extra['vuong']``.

        ``boundary_solution`` is raised when ``alpha`` (use :func:`zip`) or the
        inflation probability (use :func:`nbreg`) is estimated at zero;
        ``separation_detected`` when a regressor predicts zeros perfectly.

    Stata
        ``zinb count persons livebait, inflate(child camper) zip`` is
        ``oe.zinb(data=df, y='count', x=['persons', 'livebait'], inflate=['child', 'camper'])``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=3000), "z": rng.normal(size=3000)})
        >>> mu = np.exp(0.8 + 0.4 * df.x)
        >>> counts = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))          # alpha = 0.5
        >>> df["y"] = np.where(rng.uniform(size=3000) < 1 / (1 + np.exp(0.5 - df.z)), 0, counts)
        >>> print(oe.zinb(data=df, y="y", x=["x"], inflate=["z"]).summary())
    """
    spec = _spec("zinb", y=y, x=x, inflate=inflate, inflate_link=inflate_link,
                 covariance=covariance, cluster=cluster, weights=weights,
                 weight_type=weight_type, offset=offset, exposure=exposure,
                 categorical=categorical, intercept=intercept, missing=missing, alpha=alpha)
    return fit(spec, data=data)
