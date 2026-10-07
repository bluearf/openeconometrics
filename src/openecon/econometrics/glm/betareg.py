"""Beta regression for outcomes strictly inside (0, 1): Stata's ``betareg``.

Model (Ferrari and Cribari-Neto 2004; Smithson and Verkuilen 2006)
-----------------------------------------------------------------
``y_i ~ Beta(mu_i phi_i, (1 - mu_i) phi_i)``, so ``E[y_i] = mu_i`` and
``Var(y_i) = mu_i (1 - mu_i) / (1 + phi_i)``: ``phi`` is a precision. Two
equations are estimated jointly,

    mean       g(mu_i)  = x_i'b     g: logit (default), probit, cloglog, loglog
    precision  s(phi_i) = z_i'c     s: log (default), identity, sqrt

with log likelihood

    l_i = lnG(phi) - lnG(mu phi) - lnG((1-mu) phi)
          + (mu phi - 1) ln y + ((1-mu) phi - 1) ln(1 - y).

The precision equation always contains a constant; the columns of the role
``scale`` add regressors to it. Maximum likelihood by Newton-Raphson with the
analytic score and Hessian of ``kernels.BetaObjective`` (digamma / trigamma).

Starting values (Ferrari and Cribari-Neto)
------------------------------------------
``b_0`` is the (weighted) least-squares regression of ``g(y)`` on X; with its
residual variance ``s^2`` and ``mu_0 = g^-1(X b_0)``,
``phi_0 = mean(mu_0 (1 - mu_0) / (s^2 (dmu/deta)^2)) - 1`` (1 if not
positive); the precision equation starts at ``s(phi_0)`` for its constant and
zero slopes.

Reported
--------
z statistics; terms of the precision equation are prefixed ``scale:`` with
equation ``scale``. ``metrics``: log_likelihood, aic, bic, pseudo_r_squared
(the squared correlation between ``x'b`` and ``g(y)``, Ferrari and
Cribari-Neto's measure; Stata's betareg prints none), df_resid.
``tests['model']``: Wald chi2 of the mean-equation slopes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, wald_test,
)
from openecon.econometrics.glm.common import (
    build_spec, center_design, check_pweights, likelihood_covariance, likelihood_weights,
    maximize, optimizer_record, require_observations, resolve_covariance, solve_weighted,
    uncenter,
)
from openecon.econometrics.glm.families import make_link
from openecon.econometrics.glm.kernels import BetaObjective, ScaleLink
from openecon.engines.inference import critical_value
from openecon.models import ModelSpec, ResultBundle


def fit_betareg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``betareg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    weights = likelihood_weights(frame)
    w, nobs = weights.user, weights.nobs
    y = frame.numeric(spec.outcome)
    if bool((y <= 0).any()) or bool((y >= 1).any()):
        raise AnalysisError(
            "invalid_fractional_outcome",
            f"betareg needs '{spec.outcome}' strictly between 0 and 1. For an outcome that "
            "takes the values 0 or 1 use oe.fracreg (fractional logit/probit).")
    if not bool((y != y[0]).any()):
        raise AnalysisError("constant_outcome", f"'{spec.outcome}' does not vary: a degenerate "
                            "outcome has infinite precision and the beta likelihood is "
                            "unbounded.")
    link = kernel_call(make_link, frame.option("link"))
    scale_link = kernel_call(ScaleLink, frame.option("scale_link"))
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    scale = frame.drop_collinear(
        frame.design(frame.role("scale"), intercept=True, prefix="scale:"), weights.for_screen())
    k, q = len(design.terms), len(scale.terms)
    require_observations(nobs, k + q)
    if k == 0:
        raise AnalysisError("empty_design", "The mean equation has no terms; include a constant "
                            "or at least one regressor.")
    # Both equations are fitted on centered designs (see common.center_design).
    x, means = center_design(design, w)
    z, scale_means = center_design(scale, w)
    # Ferrari and Cribari-Neto starting values.
    transformed = link.link(y)
    ols = solve_weighted(x, transformed, w)
    if ols.omitted:
        raise AnalysisError("singular_design", "The mean-equation design is rank deficient.")
    eta = x @ ols.beta
    mu = link.inverse(eta)
    variance = float(ols.ssr) / max(nobs - k, 1) * link.derivative(eta, mu).square()
    ratio = mu * link.complement(eta, mu) / variance
    phi = float((w * ratio).sum() / w.sum()) - 1 if bool(torch.isfinite(ratio).all()) else 1.0
    if not (math.isfinite(phi) and phi > 0):
        phi = 1.0
    gamma = torch.zeros(q, dtype=torch.float64)
    gamma[0] = float(scale_link.link(torch.tensor(phi, dtype=torch.float64)))
    objective = kernel_call(BetaObjective, x, z, y, w, link, scale_link)
    result = maximize(objective, torch.cat([ols.beta, gamma]), what="beta regression")
    if not result.converged:
        raise AnalysisError(
            "nonconvergence",
            f"The beta regression likelihood did not converge: "
            f"{result.diagnostics.get('message')} Outcomes extremely close to 0 or 1, or a "
            "precision equation that is not identified, are the usual causes.")
    centered, log_likelihood = result.theta, result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centered), weights)
    theta, covariance = uncenter(centered, covariance, [(0, means), (k, scale_means)])
    info["df_inference"] = None
    eta, mu, phi_fitted = kernel_call(objective.fitted, centered)
    # Squared (weighted) correlation between the linear predictor and g(y).
    total = w.sum()
    eta_c, g_c = eta - (w * eta).sum() / total, transformed - (w * transformed).sum() / total
    denominator = float((w * eta_c.square()).sum() * (w * g_c.square()).sum())
    pseudo = float((w * eta_c * g_c).sum()) ** 2 / denominator if denominator > 0 else None
    criteria = information_criteria(log_likelihood, k + q, nobs)
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
               "pseudo_r_squared": pseudo, "df_resid": nobs - k - q}
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests = {"model": wald_test(theta, covariance, slopes,
                                label="Wald chi2 test of the mean-equation slopes")}
    extra: dict[str, Any] = {
        "link": link.name, "scale_link": scale_link.name,
        "pseudo_r_squared_definition": "squared correlation between x'b and link(y)",
        "precision_range": [float(phi_fitted.min()), float(phi_fitted.max())],
        "starting_values": {"phi": phi, "method": "ferrari_cribari_neto_ols"},
    }
    if q == 1:
        # Constant precision: report phi itself by the delta method.
        constant = theta[k].reshape(1)
        estimate = float(scale_link.inverse(constant))
        jacobian = float(scale_link.derivatives(constant, scale_link.inverse(constant))[0])
        se_link = math.sqrt(float(covariance[k, k]))
        critical = critical_value(spec.alpha)
        bounds = [float(scale_link.inverse(constant + sign * critical * se_link))
                  for sign in (-1, 1)] if scale_link.name == "log" else \
            [estimate - critical * abs(jacobian) * se_link,
             estimate + critical * abs(jacobian) * se_link]
        extra["precision"] = {"estimate": estimate, "std_error": abs(jacobian) * se_link,
                              "ci_low": bounds[0], "ci_high": bounds[1],
                              "method": "delta method from the scale-equation constant"}
    return build_result(
        frame, terms=[*design.terms, *scale.terms], params=theta, covariance=covariance,
        equations=[spec.outcome] * k + ["scale"] * q, use_t=False, df_resid=nobs - k - q,
        metrics=metrics, fitted=mu, nobs=nobs, inference=info, tests=tests, extra=extra,
        categories={**design.categories, **scale.categories}, solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def betareg(*, data: Any, y: str, x: Sequence[str], link: str = "logit",
            scale_link: str = "log", scale: Sequence[str] | None = None,
            covariance: str | None = None, cluster: str | Sequence[str] | None = None,
            weights: str | None = None, weight_type: str | None = None,
            categorical: Sequence[str] | None = None, intercept: bool = True,
            missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Beta regression for a proportion strictly inside (0, 1), Stata's ``betareg``.

    Model
        ``y ~ Beta(mu phi, (1 - mu) phi)``: ``E[y] = mu`` and
        ``Var(y) = mu (1 - mu) / (1 + phi)``. The mean equation is ``g(mu) = x'b`` and
        the precision equation ``s(phi) = z'c`` (a larger ``phi`` means less
        dispersion). Both are estimated jointly by maximum likelihood (Newton-Raphson,
        analytic gradient and Hessian) from Ferrari and Cribari-Neto's starting values.

    Parameters
        data, y, x: data, outcome with ``0 < y < 1`` (use :func:`fracreg` when 0 or 1
            occur) and the list of mean-equation regressors (may be empty).
        link: mean link ``'logit'`` (default), ``'probit'``, ``'cloglog'``, ``'loglog'``.
        scale_link: precision link ``'log'`` (default), ``'identity'``, ``'sqrt'``.
        scale: list of regressors of the precision equation (its constant is always
            included). Default: a constant precision.
        covariance: ``'nonrobust'`` (default, inverse observed information),
            ``'opg'``, ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``).
        weights, weight_type: ``'fweight'``, ``'pweight'`` (robust by default),
            ``'iweight'`` (as Stata's betareg).
        categorical: columns of ``x`` or ``scale`` expanded as ``name[level]`` dummies.
        intercept: include the mean-equation constant.
        cluster, missing, alpha: as in :func:`glm`.

    Result
        z statistics. Terms: mean equation (equation = the outcome), then
        ``scale:Intercept`` and ``scale:<column>`` (equation ``scale``). ``metrics``:
        log_likelihood, aic, bic, pseudo_r_squared (squared correlation between
        ``x'b`` and ``g(y)``), df_resid. ``tests['model']``: Wald chi2 of the
        mean-equation slopes. ``extra['precision']`` (constant precision only): ``phi``
        with its delta-method standard error and interval; ``extra['precision_range']``.

    Stata
        ``betareg prate mrate i.sole, scale(totemp) link(probit)`` is
        ``oe.betareg(data=df, y='prate', x=['mrate', 'sole'], categorical=['sole'],
        scale=['totemp'], link='probit')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=500)})
        >>> mu = 1 / (1 + np.exp(-0.2 - 0.5 * df.x))
        >>> df["y"] = rng.beta(mu * 8, (1 - mu) * 8)
        >>> print(oe.betareg(data=df, y="y", x=["x"]).summary())
    """
    spec = build_spec(
        "betareg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"scale": column_list(scale, "scale")},
        options={"link": link, "scale_link": scale_link})
    return fit(spec, data=data)
