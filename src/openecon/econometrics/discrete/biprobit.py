"""Bivariate probit by full maximum likelihood: Stata's ``biprobit``.

Model
-----
Two binary outcomes with latent indices and jointly normal errors,

    y1* = x1'b1 + e1,   y2* = x2'b2 + e2,   y_j = 1[y_j* > 0],
    (e1, e2) ~ N(0, [[1, rho], [rho, 1]]).

With ``q_j = 2 y_j - 1`` the probability of an observed pair is

    Pr(y1, y2 | x) = Phi2(q1 x1'b1, q2 x2'b2; q1 q2 rho),

``Phi2`` being the standard bivariate normal distribution function
(``bivariate.bvn_cdf``, Genz's algorithm). When both equations have the same
regressors this is Stata's ``biprobit y1 y2 x``; with ``x2`` given it is the
seemingly unrelated form ``biprobit (y1 = x) (y2 = x2)``. One outcome may enter
the other equation as a regressor (the recursive bivariate probit with an
endogenous dummy), whose likelihood is the same expression.

Estimation
----------
Newton-Raphson on ``(b1, b2, athrho)`` with ``athrho = atanh(rho)``, using the
analytic score and the analytic Hessian of ``bivariate.BiprobitObjective``.
With constants in the equations the iteration runs on regressors centred at
their means and the constants are mapped back exactly. Starting values are the
two univariate probit fits and ``rho = 0`` (Stata's "comparison" models). A correlation that runs to +-1 (one outcome is a
deterministic function of the other given the regressors, or a cell of the 2x2
table of outcomes is empty) has no interior maximum and is rejected with
``boundary_solution``: the iteration is stopped when it fails or goes flat with
``|rho| > 0.995``.

Reported
--------
Terms ``<y1>:<term>`` and ``<y2>:<term>`` (equations named after the outcomes),
then the ancillary ``/athrho``. ``metrics``: log_likelihood, aic, bic
(``k1 + k2 + 1`` parameters), rho. ``extra['rho']``: the correlation with its
delta-method standard error ``(1 - rho^2) se(athrho)`` and the confidence
interval ``tanh(athrho -+ z se)``. ``tests['model']``: Wald chi2 of all slopes
of both equations (what Stata's header prints). ``tests['rho']``: the
likelihood-ratio chi2(1) of ``rho = 0`` against the two separate probits under
a likelihood-based covariance (``nonrobust``, ``opg``), the Wald chi2(1) of
``athrho = 0`` under ``robust`` / ``cluster``.
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
    ModelFrame, build_result, column_list, information_criteria, kernel_call, lr_test,
    wald_test,
)
from openecon.econometrics.discrete.bivariate import BiprobitObjective
from openecon.econometrics.discrete.common import (
    EXTREME, EXTREME_EARLY, WATCH_EVERY, WATCH_FROM, Separated, binary_outcome, build_spec,
    centre_regressors, check_pweights, constant_shift, likelihood_covariance,
    likelihood_weights, maximize, optimizer_record, perfectly_predicted, probit_fit,
    raise_not_converged, reparameterize, require_observations, resolve_covariance,
)
from openecon.engines.inference import critical_value
from openecon.models import ModelSpec, ResultBundle

# A FAILED run whose |athrho| exceeds this value is a correlation running to the boundary:
# |rho| > 0.995. (A converged fit with such a correlation is reported as usual.)
BOUNDARY = 3.0
# A run whose |athrho| passes this value is stopped early: |rho| > 1 - 1.1e-8.
BOUNDARY_EARLY = 9.5
# The likelihood counts as flat when a round of WATCH_EVERY iterations gains less than this
# relative amount; with |athrho| > BOUNDARY the run is then stopped (see ``fit_biprobit``).
FLAT = 1e-11


def _equations(frame: ModelFrame, command: str) -> tuple[str, list[str]]:
    """The second outcome and the regressors of its equation, validated."""
    spec = frame.spec
    second = frame.role("outcome2")[0]
    regressors = frame.role("predictors2") or list(spec.predictors)
    if second == spec.outcome:
        raise AnalysisError("invalid_spec", f"{command}: y1 and y2 must be two different "
                            "outcome columns.")
    if second in regressors:
        raise AnalysisError("invalid_spec", f"{command}: the outcome '{second}' cannot be a "
                            "regressor of its own equation.")
    if second in spec.predictors and spec.outcome in regressors:
        raise AnalysisError(
            "invalid_spec",
            f"{command}: each outcome is a regressor of the other equation. Such a simultaneous "
            "model is not coherent; at most one outcome may enter the other equation "
            "(recursive bivariate probit).")
    return second, regressors


def _varying_outcome(y: Tensor, w: Tensor, name: str, command: str) -> None:
    positives, total = float((w * y).sum()), float(w.sum())
    if positives == 0 or positives == total:
        raise AnalysisError("constant_outcome", f"{command}: the outcome '{name}' does not vary "
                            "in the estimation sample.")


def _raise_failure(command: str, objective: BiprobitObjective, theta: Tensor, message: Any,
                   limit: float, threshold: float) -> None:
    """Explain a bivariate probit iteration that did not converge."""
    athrho = float(theta[-1])
    if abs(athrho) > limit:
        sign = "+1" if athrho > 0 else "-1"
        raise AnalysisError(
            "boundary_solution",
            f"{command}: the correlation of the two equations runs to {sign} (athrho = "
            f"{athrho:.3g}) and the likelihood keeps increasing, so no interior maximum exists. "
            "Given the regressors one outcome is (almost) a deterministic function of the "
            "other: the two outcomes are identical or exact complements, or one cell of their "
            "2x2 table is empty or nearly so. Use a single probit, or check how the outcomes "
            "are defined.")
    count = perfectly_predicted(objective.outcome_probabilities(theta), threshold)
    raise_not_converged(
        command, count, message,
        advice="The bivariate probit likelihood is not globally concave; check that both "
               "equations are identified (an exclusion restriction helps when one outcome "
               "enters the other equation) and the scale of the regressors.")


def fit_biprobit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``biprobit``: ``(spec, data) -> ResultBundle``."""
    command = "biprobit"
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    first = spec.outcome
    second, regressors = _equations(frame, command)
    y1 = binary_outcome(frame, first, command)
    y2 = binary_outcome(frame, second, command)
    weights = likelihood_weights(frame)
    w, screen = weights.user, weights.for_screen()
    _varying_outcome(y1, w, first, command)
    _varying_outcome(y2, w, second, command)
    design1 = frame.drop_collinear(frame.design(prefix=f"{first}:"), screen)
    design2 = frame.drop_collinear(frame.design(regressors, prefix=f"{second}:"), screen)
    centre1, centre2 = centre_regressors(design1, weights), centre_regressors(design2, weights)
    k1, k2 = len(design1.terms), len(design2.terms)
    if k1 == 0 or k2 == 0:
        raise AnalysisError("no_parameters", f"{command}: an equation has no regressor left; "
                            "include a constant or a regressor that varies.")
    size = k1 + k2 + 1
    require_observations(weights.nobs, size)
    _, probit1 = probit_fit(design1.x, y1, w, command=command, what=f"probit for {first}",
                            scale=weights.scale)
    _, probit2 = probit_fit(design2.x, y2, w, command=command, what=f"probit for {second}",
                            scale=weights.scale)
    objective = kernel_call(BiprobitObjective, design1.x, design2.x, y1, y2, w)

    previous = [-math.inf]

    def watch(iteration: int, theta: Tensor, value: float) -> None:
        if iteration < WATCH_FROM or iteration % WATCH_EVERY:
            return
        athrho = abs(float(theta[-1]))
        # With an empty cell in the 2x2 table of outcomes the likelihood increases in
        # |athrho| for ever but ever more slowly: it is flat to rounding long before
        # |athrho| is large, and Newton-Raphson would creep on for hundreds of iterations.
        flat = value - previous[0] <= FLAT * max(1.0, abs(value))
        previous[0] = value
        if athrho > BOUNDARY_EARLY or (flat and athrho > BOUNDARY) or perfectly_predicted(
                objective.outcome_probabilities(theta), EXTREME_EARLY):
            raise Separated(theta.clone())

    start = torch.cat([probit1.theta, probit2.theta, torch.zeros(1, dtype=torch.float64)])
    try:
        result = maximize(objective, start, what=command, callback=watch, scale=weights.scale)
    except Separated as stop:
        _raise_failure(command, objective, stop.theta, "the likelihood is monotone.",
                       BOUNDARY, EXTREME_EARLY)
    if not result.converged:
        _raise_failure(command, objective, result.theta, result.diagnostics.get("message"),
                       BOUNDARY, EXTREME)
    centred, log_likelihood = result.theta, result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    # Fitted on centred regressors: each equation's constant is a = a_c - m'b.
    jacobian = torch.block_diag(constant_shift(centre1), constant_shift(centre2),
                                torch.ones((1, 1), dtype=torch.float64))
    theta, covariance = reparameterize(centred, covariance, jacobian)
    last = size - 1
    athrho = float(theta[last])
    rho = math.tanh(athrho)
    se_athrho = math.sqrt(max(float(covariance[last, last]), 0.0))
    z = critical_value(spec.alpha)
    rho_record = {
        "estimate": rho, "std_error": se_athrho / math.cosh(athrho) ** 2,
        "ci_low": math.tanh(athrho - z * se_athrho), "ci_high": math.tanh(athrho + z * se_athrho),
        "athrho": athrho, "athrho_std_error": se_athrho,
        "method": "rho = tanh(athrho); delta-method standard error (1 - rho^2) se(athrho); "
                  "confidence interval = tanh of the athrho interval",
    }
    slopes = [offset + i for offset, design in ((0, design1), (k1, design2))
              for i in range(int(design.intercept), len(design.terms))]
    comparison = probit1.value + probit2.value
    tests: dict[str, Any] = {
        "model": wald_test(theta, covariance, slopes,
                           label="Wald chi2 test of the slopes of both equations")}
    if spec.covariance in {"nonrobust", "opg"}:
        tests["rho"] = lr_test(log_likelihood, comparison, 1,
                               label="LR test of rho = 0 (two independent probits)")
    else:
        tests["rho"] = wald_test(theta, covariance, [last], label="Wald test of rho = 0")
    criteria = information_criteria(log_likelihood, size, weights.nobs)
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
               "rho": rho}
    cells = {f"{a}{b}": float((w * (y1 == a) * (y2 == b)).sum()) for a in (0, 1) for b in (0, 1)}
    seemingly_unrelated = [term.removeprefix(f"{first}:") for term in design1.terms] \
        != [term.removeprefix(f"{second}:") for term in design2.terms]
    extra = {
        "rho": rho_record, "equations": [first, second],
        "comparison_log_likelihood": comparison,
        "probit_log_likelihoods": [probit1.value, probit2.value],
        "outcome_counts": cells,
        "outcome_counts_key": "first digit = " + first + ", second digit = " + second,
        "seemingly_unrelated": seemingly_unrelated,
        "recursive": first in regressors or second in spec.predictors,
        "hessian": "analytic",
        "starting_values": "univariate probit estimates and rho = 0",
    }
    title = "Seemingly unrelated bivariate probit" if seemingly_unrelated \
        else "Bivariate probit regression"
    return build_result(
        frame, terms=[*design1.terms, *design2.terms, "/athrho"], params=theta,
        covariance=covariance, equations=[first] * k1 + [second] * k2 + [None], use_t=False,
        title=title, metrics=metrics, fitted=objective.marginal_probabilities(centred)[0],
        nobs=weights.nobs, inference=info, tests=tests, extra=extra,
        categories={**design1.categories, **design2.categories},
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def biprobit(*, data: Any, y1: str, y2: str, x: Sequence[str], x2: Sequence[str] | None = None,
             covariance: str | None = None, cluster: str | Sequence[str] | None = None,
             weights: str | None = None, weight_type: str | None = None,
             categorical: Sequence[str] | None = None, intercept: bool = True,
             missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Bivariate probit regression, Stata's ``biprobit``.

    Model
        Two binary outcomes whose latent errors are correlated:

            ``y1 = 1[x'b1 + e1 > 0]``,  ``y2 = 1[x2'b2 + e2 > 0]``,
            ``(e1, e2)`` standard bivariate normal with correlation ``rho``.

        ``Pr(y1, y2 | x) = Phi2(q1 x'b1, q2 x2'b2; q1 q2 rho)`` with ``q_j = 2 y_j - 1``.
        ``rho = 0`` gives two independent probits. With ``x2=None`` both equations use
        ``x`` (``biprobit y1 y2 x``); otherwise the model is the seemingly unrelated
        bivariate probit ``biprobit (y1 = x) (y2 = x2)``. ``y1`` may be listed in ``x2``
        (or ``y2`` in ``x``, not both): the recursive bivariate probit with an endogenous
        binary regressor has the same likelihood.

    Estimator
        Full maximum likelihood over ``(b1, b2, athrho)``, ``athrho = atanh(rho)``, by
        Newton-Raphson with the analytic score and analytic Hessian. ``Phi2`` is computed
        by Genz's algorithm (20-point Gauss-Legendre quadrature of Plackett's identity,
        absolute error about 1e-15); joint probabilities below 1e-5 are recomputed from a
        positive one-dimensional integrand so that ``ln Phi2`` keeps its relative accuracy
        for outlying observations. Starting values: the two univariate probits and
        ``rho = 0``.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y1, y2: the two outcomes, each coded 0/1. Rows with a missing value in either
            outcome follow ``missing``.
        x: regressors of the first equation (and of the second when ``x2`` is None);
            collinear columns are omitted with a warning, equation by equation.
        x2: regressors of the second equation (default: ``x``).
        covariance: ``'nonrobust'`` (default; inverse observed information), ``'opg'``,
            ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``; implied by
            ``cluster``; two columns give two-way clustering).
        cluster: cluster column, or a list of two columns.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        categorical: columns of ``x`` or ``x2`` to expand into treatment-coded indicators.
        intercept: include a constant in both equations (default True).
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Result
        z statistics. Terms ``<y1>:<term>`` (equation ``y1``), ``<y2>:<term>`` (equation
        ``y2``) and the ancillary ``/athrho``. ``metrics``: log_likelihood, aic, bic, rho.
        ``extra['rho']``: estimate, delta-method ``std_error``, ``ci_low`` / ``ci_high``
        (``tanh`` of the athrho interval). ``tests['model']``: Wald chi2 of all slopes.
        ``tests['rho']``: LR chi2(1) of ``rho = 0`` against the two probits
        (``nonrobust``, ``opg``) or the Wald chi2(1) (``robust``, ``cluster``).
        ``extra`` also holds ``comparison_log_likelihood``, ``probit_log_likelihoods``
        and the 2x2 ``outcome_counts``. The chart sample holds the fitted ``Pr(y1 = 1)``.
        A correlation running to +-1 raises ``boundary_solution``.

    Stata
        ``biprobit private vote logptax loginc years`` is
        ``oe.biprobit(data=df, y1='private', y2='vote', x=['logptax', 'loginc', 'years'])``;
        ``biprobit (private = logptax loginc) (vote = years), vce(robust)`` is
        ``oe.biprobit(data=df, y1='private', y2='vote', x=['logptax', 'loginc'],
        x2=['years'], covariance='robust')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=2000), "z": rng.normal(size=2000)})
        >>> e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=2000)
        >>> df["work"] = (0.3 + 0.8 * df.x + e[:, 0] > 0) * 1.0
        >>> df["insured"] = (-0.2 + 0.5 * df.x - 0.6 * df.z + e[:, 1] > 0) * 1.0
        >>> print(oe.biprobit(data=df, y1="work", y2="insured", x=["x"], x2=["x", "z"]).summary())
    """
    second = None if x2 is None else column_list(x2, "x2")
    if second is not None and not second:
        raise AnalysisError("invalid_spec", "x2 must name at least one regressor; pass x2=None "
                            "to use the regressors of the first equation.")
    spec = build_spec(
        "biprobit", outcome=y1, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"outcome2": y2, "predictors2": second})
    return fit(spec, data=data)
