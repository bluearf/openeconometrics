"""Probit and tobit with continuous endogenous regressors: Stata's ``ivprobit`` and ``ivtobit``.

Model
-----
    y1* = x1'g + y2'b + u              (outcome equation; d = (g, b))
    y2  = Pi [x1, x2] + v              (one reduced form per endogenous regressor)
    (u, v) jointly normal,

``y1 = 1[y1* > 0]`` with ``var(u) = 1`` for ``ivprobit``; ``y1*`` censored at
``ll`` / ``ul`` for ``ivtobit``. ``x2`` are the excluded instruments. The
endogenous regressors are correlated with ``u`` through ``Cov(v, u)``;
exogeneity is ``Cov(v, u) = 0``.

``method='ml'`` (Stata's default) maximizes the joint likelihood
``f(y1 | y2, x) f(y2 | x)`` by Newton-Raphson with the analytic score and
Hessian of ``iv_kernels.IVObjective``, in a recursive working
parameterization that is unconstrained for any number of endogenous
regressors, from control-function starting values. The estimates are then
mapped exactly to Stata's parameters -- the coefficients of the outcome
equation, the reduced-form coefficients, ``/athrho{i}_{j}`` (``atanh`` of the
error correlations) and ``/lnsigma{i}`` -- with the delta-method covariance
``J V J'`` (analytic Jacobian). ``method='twostep'`` is Newey's (1987)
minimum chi-squared estimator (``ivsample.newey_two_step``). Both run on
centred exogenous variables and endogenous regressors when the model has a
constant; ``EndogenousSample.reported`` maps every constant back exactly.

The Wald test of exogeneity is the joint test that the correlations between
``u`` and every ``v_j`` are zero (``/athrho{j+1}_1 = 0``) for ML, and the test
that the first-stage residuals have zero coefficients in the two-stage
conditional model for the two-step estimator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    build_result, column_list, information_criteria, kernel_call, wald_test,
)
from openecon.econometrics.limited.censored import censoring_bounds
from openecon.econometrics.limited.common import (
    binary_outcome, build_spec, check_pweights, likelihood_covariance, limit_options, ols_start,
    probit_fit, require_observations, resolve_covariance, resolve_limits, rho_record,
    run_newton, sigma_record, solver_fields, weight_scale,
)
from openecon.econometrics.limited.iv_kernels import IVObjective, ProbitRows
from openecon.econometrics.limited.ivsample import (
    EndogenousSample, OutcomeFit, newey_two_step, structural_index,
)
from openecon.econometrics.limited.kernels import CensoredObjective, CensoredRows
from openecon.engines.linalg import collinear_columns, least_squares
from openecon.engines.optimize import information_inverse
from openecon.models import ModelSpec, ResultBundle

_ADVICE = ("Check that the excluded instruments are relevant for the endogenous regressors "
           "(first-stage F in a two-step fit) and the scale of the regressors; "
           "method='twostep' does not need the joint likelihood.")


def _check_method(sample: EndogenousSample, method: str) -> None:
    spec, command = sample.frame.spec, sample.command
    if method != "twostep":
        check_pweights(sample.frame)
        return
    if spec.covariance != "nonrobust":
        raise AnalysisError(
            "unsupported_covariance",
            f"{command}, method='twostep' has its own minimum chi-squared covariance; choose "
            "covariance='nonrobust' (the default) or method='ml' for robust and cluster "
            "covariances.")
    if spec.weight_type not in (None, "fweight"):
        raise AnalysisError("unsupported_weights", f"{command}, method='twostep' accepts "
                            "frequency weights only; use method='ml' for other weight types.")


def _control_design(sample: EndogenousSample, resid: Tensor) -> Tensor:
    """``[w, residuals]``, refusing a residual block that is collinear with ``w``."""
    design = torch.cat([sample.w, resid], dim=1)
    _, dependent = kernel_call(collinear_columns, design, sample.weights.user)
    if dependent:
        raise AnalysisError(
            "underidentified",
            f"{sample.command}: the first-stage residuals are collinear with the regressors of "
            "the outcome equation, so the excluded instruments do not explain the endogenous "
            "regressors (rank condition). Use instruments that are relevant.")
    return design


def _probit_outcome(sample: EndogenousSample, y: Tensor) -> OutcomeFit:
    weights, command = sample.weights, sample.command

    def outcome_fit(design: Tensor, what: str) -> tuple[Tensor, Tensor, float]:
        _, result = probit_fit(design, y, weights.user, command=command, what=what,
                               scale=weight_scale(weights))
        return result.theta, kernel_call(information_inverse, -result.hessian), result.value

    return outcome_fit


def _tobit_outcome(sample: EndogenousSample, lower: Tensor, upper: Tensor, y: Tensor
                   ) -> OutcomeFit:
    weights, command = sample.weights, sample.command

    def outcome_fit(design: Tensor, what: str) -> tuple[Tensor, Tensor, float]:
        objective = kernel_call(CensoredObjective, design, lower, upper, weights.user)
        beta, tau = ols_start(design, y, weights.user, command=command)
        start = torch.cat([beta, torch.tensor([tau], dtype=torch.float64)])
        result = run_newton(objective, start, command=f"{command} ({what})", advice=_ADVICE,
                            scale=weight_scale(weights))
        return result.theta, kernel_call(information_inverse, -result.hessian), result.value

    return outcome_fit


def _ml(sample: EndogenousSample, rows: Any, outcome_fit: OutcomeFit) -> dict[str, Any]:
    """Joint maximum likelihood; estimates and covariance in Stata's parameterization."""
    command, frame, weights = sample.command, sample.frame, sample.weights
    w, kz, p = weights.user, sample.kz, sample.p
    kw = sample.k1 + p
    objective = kernel_call(IVObjective, sample.z.x, sample.y2, sample.k1, rows, w)
    require_observations(weights.nobs, objective.size)
    start = torch.zeros(objective.size, dtype=torch.float64)
    resid = torch.empty_like(sample.y2)
    total = float(w.sum())
    for j in range(p):
        width = kz + j
        solved = kernel_call(least_squares, objective.base[:, :width], sample.y2[:, j], w,
                             drop_collinear=False)
        start[objective.o_phi[j]:objective.o_phi[j] + width] = solved.beta
        start[objective.o_tau + j] = 0.5 * math.log(float(solved.ssr) / total)
        resid[:, j] = solved.resid
    control, _, _ = outcome_fit(_control_design(sample, resid),
                                "control-function starting values")
    start[:kw + p] = control[:kw + p]
    if objective.scale:
        start[-1] = control[-1]
    result = run_newton(objective, start, command=command, advice=_ADVICE,
                        scale=weight_scale(weights))
    theta = result.theta
    internal, info = likelihood_covariance(frame, result.hessian,
                                           lambda: objective.score_rows(theta), weights)
    params, jacobian = objective.structural(theta)
    if not bool(torch.isfinite(params).all()) or not bool(torch.isfinite(jacobian).all()):
        raise AnalysisError(
            "boundary_solution",
            f"{command}: an error correlation runs to +-1, so the likelihood has no interior "
            "maximum. The endogenous regressor is (almost) a deterministic function of the "
            "outcome error; check the model, or use method='twostep'.")
    index = structural_index(sample, params)
    params, covariance = sample.reported(params, jacobian @ internal @ jacobian.T,
                                         reduced_forms=True)
    info["correction"] = (f"{info['correction']}; estimated in a recursive parameterization "
                          "and mapped to the reported parameters by the delta method (analytic "
                          "Jacobian)")
    names = objective.ancillary_names()
    first = kw + p * kz
    correlations, deviations = {}, {}
    labels = [frame.spec.outcome, *sample.endogenous]
    for position, name in enumerate(names, start=first):
        estimate, variance = float(params[position]), float(covariance[position, position])
        if name.startswith("/athrho"):
            i, j = (int(part) for part in name.removeprefix("/athrho").split("_"))
            correlations[name] = {**rho_record(estimate, variance, frame.spec.alpha),
                                  "label": f"corr(e.{labels[i - 1]}, e.{labels[j - 1]})"}
        else:
            i = int(name.removeprefix("/lnsigma"))
            deviations[name] = {**sigma_record(estimate, variance, frame.spec.alpha),
                                "label": f"sd(e.{labels[i - 1]})"}
    exogeneity = [first + i * (i + 1) // 2 for i in range(p)]
    terms = [*sample.terms,
             *(f"{name}:{term}" for name in sample.endogenous for term in sample.z.terms), *names]
    equations = [frame.spec.outcome] * kw \
        + [name for name in sample.endogenous for _ in range(kz)] + [None] * len(names)
    return {"params": params, "covariance": covariance, "info": info, "result": result,
            "terms": terms, "equations": equations, "exogeneity": exogeneity,
            "correlations": correlations, "deviations": deviations, "size": objective.size,
            "index": index}


def _slopes(sample: EndogenousSample) -> list[int]:
    return [i for i, term in enumerate(sample.terms) if term != "Intercept"]


def _finish(sample: EndogenousSample, method: str, rows: Any, outcome_fit: OutcomeFit, *,
            fitted, metrics: dict[str, Any], extra: dict[str, Any], prediction: str
            ) -> ResultBundle:
    """Run the chosen estimator and assemble the result."""
    frame, weights = sample.frame, sample.weights
    slopes = _slopes(sample)
    record = {**sample.record(), "method": method, **extra}
    if method == "twostep":
        step = newey_two_step(sample, outcome_fit)
        index = structural_index(sample, step["delta"])
        delta, covariance = sample.reported(step["delta"], step["covariance"],
                                            reduced_forms=False)
        tests = {"model": wald_test(delta, covariance, slopes,
                                    label="Wald chi2 test of the slopes"),
                 "exogeneity": step["exogeneity"]}
        record["covariance"] = "Newey (1987) minimum chi-squared: (D' Omega^-1 D)^-1"
        info = {"covariance": "nonrobust", "df_inference": None,
                "correction": "Newey minimum chi-squared two-step: (D' Omega^-1 D)^-1 with "
                              "Omega = J_aa^-1 + s^2 (Z'Z)^-1"}
        return build_result(
            frame, terms=sample.terms, params=delta, covariance=covariance, use_t=False,
            title=f"{frame.info.title} (two-step)", metrics=metrics,
            fitted=fitted(index), nobs=weights.nobs, inference=info,
            tests=tests, extra=record, categories=sample.categories,
            provenance={"prediction_definition": prediction},
            solver="newey_minimum_chi_squared",
            solver_diagnostics={"converged": True, "iterations": 0})
    ml = _ml(sample, rows, outcome_fit)
    params, covariance, result = ml["params"], ml["covariance"], ml["result"]
    tests = {"model": wald_test(params, covariance, slopes,
                                label="Wald chi2 test of the slopes"),
             "exogeneity": wald_test(params, covariance, ml["exogeneity"],
                                     label="Wald test of exogeneity (corr = 0)")}
    criteria = information_criteria(result.value, ml["size"], weights.nobs)
    record.update({"correlations": ml["correlations"], "standard_deviations": ml["deviations"],
                   "hessian": "analytic",
                   "starting_values": "recursive first-stage least squares and the "
                                      "control-function fit of the outcome equation"})
    metrics = {"log_likelihood": result.value, "aic": criteria["aic"], "bic": criteria["bic"],
               **metrics}
    if "/lnsigma1" in ml["deviations"]:
        metrics["sigma"] = ml["deviations"]["/lnsigma1"]["estimate"]
    return build_result(
        frame, terms=ml["terms"], params=params, covariance=covariance,
        equations=ml["equations"], use_t=False, metrics=metrics,
        fitted=fitted(ml["index"]), nobs=weights.nobs, inference=ml["info"],
        tests=tests, extra=record, categories=sample.categories,
        provenance={"prediction_definition": prediction}, **solver_fields(result))


def fit_ivprobit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``ivprobit``: ``(spec, data) -> ResultBundle``."""
    command, method = "ivprobit", spec.options.get("method", "ml")
    sample = EndogenousSample(spec, data, command)
    _check_method(sample, method)
    y = binary_outcome(sample.frame, spec.outcome, command)
    w = sample.weights.user
    positives, total = float((w * y).sum()), float(w.sum())
    if positives == 0 or positives == total:
        raise AnalysisError("constant_outcome", f"{command}: the outcome '{spec.outcome}' does "
                            "not vary in the estimation sample.")
    extra = {"zero_outcomes": total - positives, "nonzero_outcomes": positives}
    if method == "twostep":
        extra["normalization"] = ("coefficients of the index conditional on the first-stage "
                                  "errors; they differ from the ML coefficients by the factor "
                                  "1 / sd(u | v)")
    return _finish(sample, method, kernel_call(ProbitRows, y), _probit_outcome(sample, y),
                   fitted=torch.special.ndtr, metrics={}, extra=extra,
                   prediction="Phi(w'd): probability of a positive outcome at the estimated "
                              "coefficients of the outcome equation")


def fit_ivtobit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``ivtobit``: ``(spec, data) -> ResultBundle``."""
    command, method = "ivtobit", spec.options.get("method", "ml")
    sample = EndogenousSample(spec, data, command)
    _check_method(sample, method)
    y = sample.frame.numeric(spec.outcome)
    ll, ul = resolve_limits(sample.frame, y, command)
    lower, upper, n_left, n_uncensored, n_right = censoring_bounds(
        y, ll, ul, sample.weights, command)
    metrics = {"n_left_censored": n_left, "n_uncensored": n_uncensored,
               "n_right_censored": n_right}
    return _finish(sample, method, kernel_call(CensoredRows, lower, upper),
                   _tobit_outcome(sample, lower, upper, y), fitted=lambda index: index,
                   metrics=metrics, extra={"limits": {"lower": ll, "upper": ul}},
                   prediction="linear prediction w'd of the latent outcome")


def _spec(estimator: str, *, y, x, endog, instruments, method, covariance, cluster, weights,
          weight_type, categorical, intercept, missing, alpha, options) -> ModelSpec:
    resolved = covariance if method == "twostep" else resolve_covariance(
        covariance, cluster, weight_type)
    return build_spec(
        estimator, outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolved, cluster=cluster, weights=weights, weight_type=weight_type,
        missing=missing, alpha=alpha,
        columns={"endogenous": column_list(endog, "endog"),
                 "instruments": column_list(instruments, "instruments")},
        options={"method": method, **options})


def ivprobit(*, data: Any, y: str, x: Sequence[str] | None = None, endog: Sequence[str],
             instruments: Sequence[str], method: str = "ml", covariance: str | None = None,
             cluster: str | Sequence[str] | None = None, weights: str | None = None,
             weight_type: str | None = None, categorical: Sequence[str] | None = None,
             intercept: bool = True, missing: str = "raise", alpha: float = 0.05
             ) -> ResultBundle:
    """Probit with continuous endogenous regressors, Stata's ``ivprobit``.

    Model
        ``y = 1[x'g + y2'b + u > 0]`` and ``y2 = Pi [x, instruments] + v`` with
        ``(u, v)`` jointly normal and ``var(u) = 1``. The endogenous regressors ``y2``
        are continuous; they are correlated with ``u`` unless ``Cov(v, u) = 0``.

    Estimator
        ``method='ml'`` (default): full maximum likelihood of the outcome equation and
        the reduced forms,

            ``ln L = sum w {ln Phi[q (w'd + a'v) / omega] + ln f(y2 | x)}``,
            ``q = 2y - 1``, ``v = y2 - Pi z``, ``u | v ~ N(a'v, omega^2)``,

        by Newton-Raphson with the analytic score and Hessian, in a recursive
        parameterization that is unconstrained for any number of endogenous regressors;
        the estimates are mapped exactly to Stata's parameters with the delta-method
        covariance. ``method='twostep'``: Newey's (1987) minimum chi-squared estimator
        from the first-stage residuals, a reduced-form probit and the two-stage
        conditional (Rivers-Vuong) probit. Its coefficients are normalized by
        ``sd(u | v)`` and are not directly comparable with the ML coefficients. With a
        constant, exogenous variables and endogenous regressors are centred during the
        computation and every constant is mapped back exactly.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome coded 0/1.
        x: exogenous regressors of the outcome equation (optional).
        endog: continuous endogenous regressors.
        instruments: excluded instruments, at least one per endogenous regressor; the
            exogenous regressors are instruments automatically.
        method: ``'ml'`` or ``'twostep'``.
        covariance: ML: ``'nonrobust'`` (default; inverse observed information),
            ``'opg'``, ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``;
            implied by ``cluster``). Two-step: only ``'nonrobust'`` (its own covariance).
        cluster: cluster column, or a list of two columns (ML only).
        weights, weight_type: ML: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``. Two-step: ``'fweight'`` only.
        categorical: columns of ``x`` or ``instruments`` to expand into indicators.
        intercept: include a constant in the outcome equation and the reduced forms.
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Result
        z statistics. ML terms: the outcome equation (exogenous terms, then the
        endogenous regressors; equation = the outcome), the reduced forms
        ``<endog>:<term>`` (one equation per endogenous regressor) and the ancillary
        ``/athrho2_1`` (atanh of ``corr(v, u)``) and ``/lnsigma2`` (``ln sd(v)``); with
        several endogenous regressors ``/athrho{i}_{j}`` for every pair of equations
        (1 = outcome) and ``/lnsigma{i}``. Two-step terms: the outcome equation only.
        ``metrics`` (ML): log_likelihood, aic, bic. ``tests['model']``: Wald chi2 of the
        slopes; ``tests['exogeneity']``: Wald chi2(p) of exogeneity. ``extra``:
        exogenous / endogenous / instruments, ``first_stage`` (R-squared, partial
        R-squared and F of the excluded instruments, rmse), ``correlations`` and
        ``standard_deviations`` on the natural scale with delta-method standard errors
        (ML). The chart sample holds ``Phi(w'd)``.

    Stata
        ``ivprobit fem_work fem_educ kids (other_inc = male_educ)`` is
        ``oe.ivprobit(data=df, y='fem_work', x=['fem_educ', 'kids'], endog=['other_inc'],
        instruments=['male_educ'])``; add ``method='twostep'`` for ``ivprobit ..., twostep``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> n = 3000
        >>> df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        >>> e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
        >>> df["y2"] = 0.5 * df.x + 0.8 * df.z + e[:, 1]
        >>> df["y"] = (0.2 + 0.6 * df.x - 0.7 * df.y2 + e[:, 0] > 0) * 1.0
        >>> print(oe.ivprobit(data=df, y="y", x=["x"], endog=["y2"],
        ...                   instruments=["z"]).summary())
    """
    spec = _spec("ivprobit", y=y, x=x, endog=endog, instruments=instruments, method=method,
                 covariance=covariance, cluster=cluster, weights=weights,
                 weight_type=weight_type, categorical=categorical, intercept=intercept,
                 missing=missing, alpha=alpha, options={})
    return fit(spec, data=data)


def ivtobit(*, data: Any, y: str, x: Sequence[str] | None = None, endog: Sequence[str],
            instruments: Sequence[str], ll: float | str | None = None,
            ul: float | str | None = None, method: str = "ml", covariance: str | None = None,
            cluster: str | Sequence[str] | None = None, weights: str | None = None,
            weight_type: str | None = None, categorical: Sequence[str] | None = None,
            intercept: bool = True, missing: str = "raise", alpha: float = 0.05
            ) -> ResultBundle:
    """Tobit with continuous endogenous regressors, Stata's ``ivtobit``.

    Model
        ``y* = x'g + y2'b + u`` observed as ``y = max(ll, min(y*, ul))`` and
        ``y2 = Pi [x, instruments] + v`` with ``(u, v)`` jointly normal. The endogenous
        regressors are continuous; exogeneity is ``Cov(v, u) = 0``.

    Estimator
        ``method='ml'`` (default): full maximum likelihood of the censored outcome
        equation conditional on the reduced-form errors, ``y* | v ~ N(w'd + a'v,
        omega^2)``, times the normal density of ``y2``, by Newton-Raphson with the
        analytic score and Hessian; the estimates are mapped exactly to Stata's
        parameters with the delta-method covariance. ``method='twostep'``: Newey's
        (1987) minimum chi-squared estimator built from the first-stage residuals, a
        reduced-form tobit and the two-stage conditional tobit. With a constant,
        exogenous variables and endogenous regressors are centred during the
        computation and every constant is mapped back exactly.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome; values ``<= ll`` are left-censored, values ``>= ul`` right-censored.
        x: exogenous regressors of the outcome equation (optional).
        endog: continuous endogenous regressors.
        instruments: excluded instruments, at least one per endogenous regressor.
        ll, ul: censoring limits: a number, ``'min'`` / ``'max'`` (the smallest / largest
            outcome) or None.
        method: ``'ml'`` or ``'twostep'``.
        covariance: ML: ``'nonrobust'`` (default), ``'opg'``, ``'robust'``, ``'cluster'``
            (implied by ``cluster``). Two-step: only ``'nonrobust'``.
        cluster: cluster column, or a list of two columns (ML only).
        weights, weight_type: ML: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``. Two-step: ``'fweight'`` only.
        categorical: columns of ``x`` or ``instruments`` to expand into indicators.
        intercept: include a constant in the outcome equation and the reduced forms.
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Result
        z statistics (Stata's ivtobit, unlike tobit, reports z). ML terms: the outcome
        equation (equation = the outcome), the reduced forms ``<endog>:<term>`` and the
        ancillary ``/athrho2_1`` (atanh of ``corr(v, u)``), ``/lnsigma1`` (``ln sd(u)``)
        and ``/lnsigma2`` (``ln sd(v)``); more pairs with several endogenous regressors.
        Two-step terms: the outcome equation only. ``metrics``: [log_likelihood, aic,
        bic,] n_left_censored, n_uncensored, n_right_censored. ``tests['model']``: Wald
        chi2 of the slopes; ``tests['exogeneity']``: Wald chi2(p). ``extra``: limits,
        exogenous / endogenous / instruments, ``first_stage``, ``correlations`` and
        ``standard_deviations`` (ML). The chart sample holds the linear prediction.

    Stata
        ``ivtobit fem_inc fem_educ kids (other_inc = male_educ), ll(10)`` is
        ``oe.ivtobit(data=df, y='fem_inc', x=['fem_educ', 'kids'], endog=['other_inc'],
        instruments=['male_educ'], ll=10)``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> n = 3000
        >>> df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        >>> e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
        >>> df["y2"] = 0.5 * df.x + 0.8 * df.z + e[:, 1]
        >>> df["y"] = np.maximum(0.0, 0.5 + 0.6 * df.x - 0.7 * df.y2 + e[:, 0])
        >>> print(oe.ivtobit(data=df, y="y", x=["x"], endog=["y2"], instruments=["z"],
        ...                  ll=0).summary())
    """
    spec = _spec("ivtobit", y=y, x=x, endog=endog, instruments=instruments, method=method,
                 covariance=covariance, cluster=cluster, weights=weights,
                 weight_type=weight_type, categorical=categorical, intercept=intercept,
                 missing=missing, alpha=alpha, options=limit_options(ll, ul, "ivtobit"))
    return fit(spec, data=data)
