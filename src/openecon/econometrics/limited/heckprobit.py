"""Probit model with sample selection by full maximum likelihood: Stata's ``heckprobit``.

Model
-----
    outcome:    y = 1[x'b + u1 > 0]      (observed only when selected)
    selection:  s = 1[z'g + u2 > 0]
    (u1, u2) standard bivariate normal with correlation rho.

    ln L = sum_{s=1, y=1} w ln Phi2(x'b, z'g; rho)
         + sum_{s=1, y=0} w ln Phi2(-x'b, z'g; -rho)
         + sum_{s=0}      w ln Phi(-z'g).

Estimation
----------
Newton-Raphson over ``(b, g, athrho)``, ``rho = tanh(athrho)``, with the
analytic score and analytic Hessian of
``selection_kernels.HeckprobitObjective``: the selected rows are
bivariate-probit observations whose second outcome is one (the discrete
family's Genz bivariate normal distribution function with a tail-accurate
logarithm), the nonselected rows a probit term. Starting values: the probit of
the outcome on the selected sample, the probit of the selection equation and
``rho = 0``.

Reported (Stata's conventions)
------------------------------
z tests; terms of the outcome equation, ``select:<term>`` and ``/athrho``;
``rho`` with its delta-method standard error; ``tests['model']`` the Wald chi2
of the outcome slopes; ``tests['rho']`` the likelihood-ratio test of
independent equations against the two separate probits (``nonrobust``,
``opg``) or the Wald test of ``athrho = 0`` (``robust``, ``cluster``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    build_result, column_list, information_criteria, kernel_call, lr_test, wald_test,
)
from openecon.econometrics.limited.common import (
    LIKELIHOOD_KINDS, build_spec, likelihood_covariance, maximize, probit_fit,
    require_observations, resolve_covariance, rho_record, slope_indices, solver_fields,
    weight_scale,
)
from openecon.econometrics.limited.heckman import BOUNDARY, SelectionSample
from openecon.econometrics.limited.selection_kernels import HeckprobitObjective
from openecon.models import ModelSpec, ResultBundle


def _raise_failure(command: str, athrho: float, message: Any) -> None:
    if abs(athrho) > BOUNDARY:
        sign = "+1" if athrho > 0 else "-1"
        raise AnalysisError(
            "boundary_solution",
            f"{command}: the correlation between the outcome and selection errors runs to "
            f"{sign} (athrho = {athrho:.3g}) and the likelihood keeps increasing, so no "
            "interior maximum exists. Either the two errors are (almost) perfectly correlated "
            "or the model is weakly identified because the selection equation has no regressor "
            "that is excluded from the outcome equation; add one, or fit a probit on the "
            "selected sample.")
    raise AnalysisError(
        "nonconvergence",
        f"{command} did not converge: {message} The likelihood of the probit model with "
        "selection is not globally concave; check that the selection equation contains a "
        "regressor excluded from the outcome equation and that neither outcome is predicted "
        "perfectly.")


def fit_heckprobit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``heckprobit``: ``(spec, data) -> ResultBundle``."""
    command = "heckprobit"
    sample = SelectionSample(spec, data, command)
    frame, weights, on = sample.frame, sample.weights, sample.selected
    y1, w1 = sample.y[on], weights.user[on]
    if bool(((y1 != 0) & (y1 != 1)).any()):
        raise AnalysisError("invalid_binary_outcome", f"{command} needs '{spec.outcome}' coded "
                            "0/1 where it is observed; recode the positive outcome as 1 and the "
                            "negative outcome as 0.")
    positives, total = float((w1 * y1).sum()), float(w1.sum())
    if positives == 0 or positives == total:
        raise AnalysisError("constant_outcome", f"{command}: the outcome '{spec.outcome}' does "
                            "not vary in the selected sample.")
    k, q = len(sample.design_x.terms), len(sample.design_z.terms)
    size = k + q + 1
    require_observations(sample.n_selected, k + 1)
    require_observations(weights.nobs, size)
    selection = sample.selection_probit()
    _, outcome = probit_fit(sample.design_x.x[on], y1, w1, command=command,
                            what="probit of the outcome on the selected sample",
                            scale=weight_scale(weights))
    objective = kernel_call(HeckprobitObjective, sample.design_x.x, sample.design_z.x, sample.y,
                            on, weights.user)
    start = torch.cat([outcome.theta, selection.theta, torch.zeros(1, dtype=torch.float64)])
    try:
        result = maximize(objective, start, what=command, scale=weight_scale(weights))
    except AnalysisError as exc:
        if exc.code != "numerical_failure":
            raise
        _raise_failure(command, 0.0, "the likelihood has no interior maximum along the "
                                     "iteration path.")
    last = size - 1
    centred = result.theta
    if not result.converged or abs(float(centred[last])) > BOUNDARY:
        _raise_failure(command, float(centred[last]), result.diagnostics.get("message"))
    log_likelihood = result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    theta, covariance = sample.reported(centred, covariance, 1)
    rho = rho_record(float(theta[last]), float(covariance[last, last]), spec.alpha)
    slopes = slope_indices(sample.design_x)
    tests: dict[str, Any] = {}
    if slopes:
        tests["model"] = wald_test(theta, covariance, slopes,
                                   label="Wald chi2 test of the outcome slopes")
    comparison = selection.value + outcome.value
    if spec.covariance in LIKELIHOOD_KINDS:
        tests["rho"] = lr_test(log_likelihood, comparison, 1,
                               label="LR test of independent equations (rho = 0)")
    else:
        tests["rho"] = wald_test(theta, covariance, [last],
                                 label="Wald test of independent equations (rho = 0)")
    criteria = information_criteria(log_likelihood, size, weights.nobs)
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
               "rho": rho["estimate"], "n_selected": sample.n_selected,
               "n_censored": sample.n_nonselected}
    extra = {
        "select": spec.columns["select"], "selection_terms": sample.design_z.terms,
        "n_selected": sample.n_selected, "n_nonselected": sample.n_nonselected, "rho": rho,
        "comparison_log_likelihood": comparison,
        "probit_log_likelihoods": {"outcome": outcome.value, "selection": selection.value},
        "outcome_counts": {"selected_zero": total - positives, "selected_one": positives},
        "hessian": "analytic",
        "starting_values": "probit of the outcome on the selected sample, probit of the "
                           "selection equation and rho = 0",
    }
    terms = [*sample.design_x.terms, *sample.design_z.terms, "/athrho"]
    return build_result(
        frame, terms=terms, params=theta, covariance=covariance,
        equations=[spec.outcome] * k + ["select"] * q + [None], use_t=False, metrics=metrics,
        fitted=torch.special.ndtr(sample.design_z.x @ centred[k:k + q]),
        observed=on.to(torch.float64), nobs=weights.nobs, inference=info, tests=tests,
        extra=extra, categories={**sample.design_x.categories, **sample.design_z.categories},
        provenance={"prediction_definition": "fitted selection probability Phi(z'g) against "
                                             "the selection indicator"},
        **solver_fields(result))


def heckprobit(*, data: Any, y: str, x: Sequence[str], select: str, select_x: Sequence[str],
               covariance: str | None = None, cluster: str | Sequence[str] | None = None,
               weights: str | None = None, weight_type: str | None = None,
               categorical: Sequence[str] | None = None, intercept: bool = True,
               missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Probit model with sample selection, Stata's ``heckprobit``.

    Model
        ``y = 1[x'b + u1 > 0]`` is observed only when ``select = 1[z'g + u2 > 0]`` equals
        one; ``(u1, u2)`` are standard bivariate normal with correlation ``rho``. With
        ``rho != 0`` a probit on the selected sample is inconsistent. The model is best
        identified when ``select_x`` contains a regressor that is not in ``x``.

    Estimator
        Full maximum likelihood over ``(b, g, athrho)``, ``rho = tanh(athrho)``:

            ``ln L = sum_{s=1,y=1} w ln Phi2(x'b, z'g; rho)
                     + sum_{s=1,y=0} w ln Phi2(-x'b, z'g; -rho)
                     + sum_{s=0} w ln Phi(-z'g)``,

        by Newton-Raphson with the analytic score and analytic Hessian. ``Phi2`` is
        Genz's bivariate normal algorithm (absolute error about 1e-15) with a
        tail-accurate logarithm. Starting values: the two separate probits and
        ``rho = 0``.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome coded 0/1; it may be missing where ``select`` is 0 (values there are
            ignored).
        x: regressors of the outcome equation; they must be complete in every row.
        select: selection indicator coded 0/1.
        select_x: regressors of the selection equation (a constant is always included).
        covariance: ``'nonrobust'`` (default; inverse observed information), ``'opg'``,
            ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``; implied by
            ``cluster``; two columns give two-way clustering).
        cluster: cluster column, or a list of two columns.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        categorical: columns of ``x`` or ``select_x`` to expand into indicators.
        intercept: include a constant in the outcome equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows; a selected row with a
            missing outcome is incomplete.
        alpha: significance level of the confidence intervals.

    Result
        z statistics. Terms: the outcome equation (equation = the outcome),
        ``select:<term>`` (equation ``select``) and the ancillary ``/athrho``.
        ``metrics``: log_likelihood, aic, bic, rho, n_selected, n_censored (nonselected).
        ``tests['model']``: Wald chi2 of the outcome slopes. ``tests['rho']``: LR chi2(1)
        of ``rho = 0`` against the two separate probits (``nonrobust``, ``opg``) or the
        Wald chi2(1) (``robust``, ``cluster``). ``extra['rho']``: estimate, delta-method
        ``std_error`` and the interval ``tanh`` of the athrho interval. The chart sample
        holds the fitted selection probability. A correlation running to +-1 raises
        ``boundary_solution``.

    Stata
        ``heckprobit private years logptax, select(vote = years loginc logptax)`` is
        ``oe.heckprobit(data=df, y='private', x=['years', 'logptax'], select='vote',
        select_x=['years', 'loginc', 'logptax'])``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> n = 4000
        >>> df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        >>> u = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
        >>> df["s"] = (0.5 + 0.6 * df.x + 0.9 * df.z + u[:, 1] > 0) * 1.0
        >>> df["y"] = np.where(df.s == 1, (0.2 + 0.8 * df.x + u[:, 0] > 0) * 1.0, np.nan)
        >>> print(oe.heckprobit(data=df, y="y", x=["x"], select="s",
        ...                     select_x=["x", "z"]).summary())
    """
    spec = build_spec(
        "heckprobit", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"select": select, "select_x": column_list(select_x, "select_x")})
    return fit(spec, data=data)
