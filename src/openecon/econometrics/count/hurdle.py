"""Public commands ``oe.churdle`` and ``oe.hurdle`` (two-part / hurdle models).

The estimation code is in ``hurdle_fit``; this module holds the registry entry
points and the Stata-style convenience functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import column_list
from openecon.econometrics.count.hurdle_fit import fit_churdle, fit_hurdle
from openecon.econometrics.glm.common import build_spec, resolve_covariance
from openecon.models import ResultBundle

__all__ = ["churdle", "fit_churdle", "fit_hurdle", "hurdle"]


def churdle(*, data: Any, y: str, x: Sequence[str], select_x: Sequence[str] | None = None,
            select: Sequence[str] | None = None, model: str = "exponential",
            ll: float = 0.0, select_link: str = "probit", covariance: str | None = None,
            cluster: str | Sequence[str] | None = None, weights: str | None = None,
            weight_type: str | None = None, categorical: Sequence[str] | None = None,
            intercept: bool = True, missing: str = "raise", alpha: float = 0.05,
            ) -> ResultBundle:
    """Cragg hurdle regression, Stata's ``churdle linear`` / ``churdle exponential``.

    Model
        A continuous outcome with a mass of observations at a lower limit ``ll``
        (expenditure, hours, amounts: many zeros and positive values otherwise) is
        generated in two steps (Cragg 1971):

            selection:  Pr(y > ll | z) = Phi(z'g)                     (probit)
            outcome:    y | y > ll  follows, with e ~ N(0, sigma^2),
                model='linear':       y    = x'b + e  truncated to y > ll
                model='exponential':  ln y = x'b + e  truncated to y > ll

        Log likelihood per observation (``d = 1[y > ll]``):

            (1 - d) ln{1 - Phi(z'g)} + d [ln Phi(z'g) + ln f(y)],
            linear:       ln f = ln phi((y - x'b)/sigma) - ln sigma - ln Phi((x'b - ll)/sigma)
            exponential:  ln f = ln phi((ln y - x'b)/sigma) - ln sigma - ln y
                                 - ln Phi((x'b - ln ll)/sigma)      [last term 0 if ll = 0]

        Unlike tobit, different variables and coefficients may drive participation
        and the amount. The two parts share no parameter, so each is maximized by
        Newton-Raphson with analytic derivatives (the outcome equation in
        ``(b, ln sigma)`` from least-squares starting values); robust and cluster
        covariances are those of the joint estimator.

    Parameters
        data: the data set (pandas DataFrame, dict of columns or row records).
        y: outcome; observations with ``y <= ll`` are the bounded ones.
        x: list of regressors of the outcome equation.
        select_x: list of regressors of the selection equation (required), which
            always has a constant. ``select`` is accepted as an alias, after Stata's
            ``select()`` option.
        model: ``'exponential'`` (default) or ``'linear'``.
        ll: lower limit (default 0). ``model='exponential'`` needs ``ll >= 0``.
        select_link: ``'probit'`` (default, Stata) or ``'logit'``.
        covariance: ``'nonrobust'`` (default: inverse observed information),
            ``'opg'``, ``'robust'`` (sandwich times ``N/(N-1)``), ``'cluster'``
            (``G/(G-1)``; one or two cluster columns).
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'iweight'``,
            ``'pweight'`` (robust by default).
        categorical: columns of ``x`` or ``select_x`` to expand into indicator terms.
        intercept: include the constant of the outcome equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing inputs.
        alpha: significance level of the confidence intervals (default 0.05).

    Result
        z statistics. Terms in Stata's order: the outcome equation (equation = the
        outcome), ``select:<name>`` (equation ``select``, Stata's ``selection_ll``) and
        ``/lnsigma``. ``metrics``: log_likelihood, pseudo_r_squared (``1 - ll/ll_0``),
        aic, bic, sigma. ``extra['sigma']``: sigma with its delta-method standard error
        and the interval exponentiated from ``ln sigma``. ``tests['model']``: LR chi2
        that the slopes of the outcome equation are zero (``nonrobust``; the comparison
        model keeps the selection equation) or their Wald chi2. ``extra`` also has
        the log likelihoods of the two parts, the number of bounded observations and
        the mean selection probability. The chart sample uses
        ``E[y] = (1 - P) ll + P E[y | y > ll]``.

        ``no_selection_variation`` is raised when no (or every) outcome exceeds
        ``ll``; ``separation_detected`` when the selection equation predicts
        perfectly.

    Stata
        ``churdle linear hours age i.smoke, select(commute whours) ll(0)`` is
        ``oe.churdle(data=df, y='hours', x=['age', 'smoke'], categorical=['smoke'],
        select_x=['commute', 'whours'], model='linear', ll=0)``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=3000), "z": rng.normal(size=3000)})
        >>> takes_part = 0.3 + 0.8 * df.z + rng.normal(size=3000) > 0
        >>> amount = np.exp(1.0 + 0.5 * df.x + 0.6 * rng.normal(size=3000))
        >>> df["y"] = np.where(takes_part, amount, 0.0)
        >>> print(oe.churdle(data=df, y="y", x=["x"], select_x=["z"]).summary())
    """
    if select is not None and select_x is not None:
        raise AnalysisError("invalid_spec", "Give the selection regressors once: select_x "
                            "(select is an alias of the same argument).")
    chosen = column_list(select_x if select_x is not None else select, "select_x")
    spec = build_spec(
        "churdle", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"select_x": chosen},
        options={"model": model, "ll": ll, "select_link": select_link})
    return fit(spec, data=data)


def hurdle(*, data: Any, y: str, x: Sequence[str], select_x: Sequence[str] | None = None,
           dist: str = "poisson", zero_link: str = "logit", covariance: str | None = None,
           cluster: str | Sequence[str] | None = None, weights: str | None = None,
           weight_type: str | None = None, offset: str | None = None,
           exposure: str | None = None, categorical: Sequence[str] | None = None,
           intercept: bool = True, missing: str = "raise", alpha: float = 0.05,
           ) -> ResultBundle:
    """Hurdle regression for counts (Mullahy 1986), as ``hplogit`` / ``hnblogit``.

    Model
        Zeros and positive counts come from two separate processes:

            participation:  Pr(y > 0 | z) = F(z'g)
            positive count: Pr(y = k | y > 0) = f(k) / {1 - f(0)},   k = 1, 2, ...

        ``F`` is logistic (default), standard normal (``zero_link='probit'``) or
        ``1 - exp(-exp(z'g))`` (``'cloglog'``, Mullahy's Poisson hurdle);
        ``f`` is Poisson (``dist='poisson'``) or negative binomial with
        ``Var = mu (1 + alpha mu)`` (``dist='nbinomial'``), ``mu = exp(x'b + offset)``.
        Log likelihood: ``sum_{y=0} ln(1 - F) + sum_{y>0} [ln F + ln f(y) - ln{1 - f(0)}]``.
        In contrast to :func:`zip`, every zero comes from the participation
        equation, so the model also fits data with *fewer* zeros than the count
        distribution implies.

        The two parts share no parameter and are maximized separately by
        Newton-Raphson with analytic derivatives (the count part is
        :func:`tpoisson` / :func:`tnbreg` on the positive counts); robust and
        cluster covariances are those of the joint estimator.

    Parameters
        data: the data set (pandas DataFrame, dict of columns or row records).
        y: nonnegative integer count with zeros and positive values.
        x: list of regressors of the count equation (``[]``: constant only).
        select_x: list of regressors of the participation equation; default: the
            regressors ``x``. Its constant is always included.
        dist: ``'poisson'`` (default) or ``'nbinomial'`` (NB2, adds ``/lnalpha``).
        zero_link: ``'logit'`` (default), ``'probit'`` or ``'cloglog'``.
        covariance: ``'nonrobust'`` (default), ``'opg'``, ``'robust'``, ``'cluster'``.
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'iweight'``,
            ``'pweight'`` (robust by default).
        offset / exposure: offset of the count equation, or exposure entering as
            ``ln(exposure)``. Mutually exclusive.
        categorical: columns of ``x`` or ``select_x`` to expand into indicator terms.
        intercept: include the constant of the count equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing inputs.
        alpha: significance level of the confidence intervals (default 0.05).

    Result
        z statistics. Terms: the count equation (equation = the outcome),
        ``select:<name>`` (equation ``select``: a positive coefficient raises
        ``Pr(y > 0)``) and, for ``dist='nbinomial'``, ``/lnalpha``. ``metrics``:
        log_likelihood, pseudo_r_squared, aic, bic, n_zero_observations, alpha.
        ``tests['model']``: LR chi2 that the slopes of the count equation are zero
        (``nonrobust``) or their Wald chi2. ``tests['alpha']`` (negative binomial):
        LR test of ``alpha = 0`` against the Poisson hurdle (``chibar2(01)``).
        ``extra``: the log likelihoods of the two parts, ``extra['alpha']`` and the
        mean participation probability. The chart sample uses
        ``E[y] = Pr(y > 0) mu / {1 - f(0)}``.

        ``no_selection_variation`` is raised without zeros or without positive
        counts; ``separation_detected`` when a participation regressor predicts
        ``y > 0`` perfectly, or when a count regressor singles out observations whose
        positive counts all equal 1 (its coefficient diverges to minus infinity);
        ``boundary_solution`` when ``alpha`` is estimated at zero (use
        ``dist='poisson'``).

    Stata
        The community commands ``hplogit y x1 x2`` and ``hnblogit y x1 x2`` (Hilbe)
        are ``oe.hurdle(data=df, y='y', x=['x1', 'x2'])`` and
        ``oe.hurdle(..., dist='nbinomial')``. The same estimates come from
        ``logit`` on ``y > 0`` plus ``tpoisson`` / ``tnbreg`` on the positive counts.
        Note that ``hplogit`` reports the logit equation first.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=4000)})
        >>> visits = rng.poisson(np.exp(0.6 + 0.4 * df.x))
        >>> any_visit = rng.uniform(size=4000) < 1 / (1 + np.exp(-0.2 - 0.7 * df.x))
        >>> df["y"] = np.where(any_visit, np.maximum(visits, 1), 0)
        >>> print(oe.hurdle(data=df, y="y", x=["x"]).summary())
    """
    spec = build_spec(
        "hurdle", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"select_x": column_list(select_x, "select_x"), "offset": offset,
                 "exposure": exposure},
        options={"dist": dist, "zero_link": zero_link})
    return fit(spec, data=data)
