"""Generalized negative binomial regression: Stata's ``gnbreg``.

Model
-----
``y`` is negative binomial (NB2) with mean ``mu_i = exp(x_i'b + offset_i)`` and
an overdispersion parameter that varies with covariates,

    ln(alpha_i) = z_i'd,        Var(y_i | x_i, z_i) = mu_i (1 + alpha_i mu_i),

so that ``nbreg`` is the special case in which the ``lnalpha`` equation holds a
constant only. With ``m_i = 1/alpha_i`` the log likelihood of one observation is

    lnG(y+m) - lnG(m) - lnG(y+1) - (y+m) ln(1 + alpha mu) + y ln(alpha mu).

Estimation
----------
``(b, d)`` jointly by Newton-Raphson with the analytic score and Hessian of the
NB2 density in the two indices ``(x'b, z'd)`` (``kernels.NegBinDensity``),
starting from the constant-dispersion negative binomial fit (itself started
from Poisson and a moment estimate of alpha).

Reported (as Stata)
-------------------
z statistics; terms of the dispersion equation are ``lnalpha:<name>``
(equation ``lnalpha``). ``tests['model']``: LR chi2 that the slopes of the
mean equation are zero - the comparison model has a constant-only mean and the
full ``lnalpha`` equation - under the conventional covariance, Wald otherwise;
McFadden's pseudo R-squared from the same comparison model.
``tests['lnalpha']`` (an OpenEconometrics addition) tests that the slopes of the
``lnalpha`` equation are zero, i.e. ``nbreg`` against ``gnbreg``: LR under the
conventional covariance, Wald otherwise.

Boundaries
----------
The limit ``alpha -> 0`` is the Poisson model and lies outside the parameter
space. Data without overdispersion - in the whole sample, or in the part of it
that a regressor of the ``lnalpha`` equation singles out - drive ``ln(alpha)``
to minus infinity and are reported as ``boundary_solution``. Because
``lnG(y + 1/alpha) - lnG(1/alpha)`` is a difference of numbers of order
``1/alpha``, the density is not evaluated outside the validated count plus shape region
(``y_i + 1/alpha_i + 1 <= 1e7``). Such trial points are tracked and rejected
instead of feeding rounding noise to the line search. A genuine Poisson limit
requires an independent dispersion-score and weak-information certificate.
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
    ModelFrame, build_result, column_list, information_criteria, lr_test, wald_test,
)
from openecon.econometrics.count import common
from openecon.econometrics.count.common import Boundary, Model
from openecon.econometrics.count.kernels import CountPieces, NegBinDensity
from openecon.econometrics.glm.common import (
    build_spec, optimizer_record, require_observations, require_terms, resolve_covariance,
    slope_indices,
)
from openecon.engines.count_numeric import poisson_logmass
from openecon.models import ModelSpec, ResultBundle

_WATCH_FROM, _WATCH_EVERY = 15, 5
# Below this dispersion ``lnG(y + 1/alpha) - lnG(1/alpha)`` is a difference of numbers of
# order 1e10 and the density has lost the digits a line search needs: such a trial point
# is outside the numerical domain (the limit alpha -> 0 is the Poisson model).
_LOG_ALPHA_FLOOR = math.log(1e-9)


class _Guarded:
    """Per-observation pieces that reject a point where some ``ln(alpha_i)`` is below the floor."""

    def __init__(self, pieces: CountPieces):
        self.pieces = pieces

    def __call__(self, index: Sequence[Tensor], derivatives: bool = True):
        self.check_precision(index)
        if float(index[1].min()) < _LOG_ALPHA_FLOOR:
            return None
        return self.pieces(index, derivatives)

    def check_precision(self, index=None):
        self.pieces.check_precision(index)


def fit_gnbreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``gnbreg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    y, weights, offset = common.count_sample(frame, "gnbreg")
    if not bool((y > 0).any()):
        raise AnalysisError("constant_outcome", f"'{spec.outcome}' is zero in every observation; "
                            "the model is not identified.")
    w = weights.user
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    require_terms(design)
    shape = frame.drop_collinear(
        frame.design(frame.role("lnalpha"), intercept=True, prefix="lnalpha:"),
        weights.for_screen())
    k, q, nobs = len(design.terms), len(shape.terms), weights.nobs
    require_observations(nobs, k + q)
    poisson = common.poisson_start(frame, design, y, weights, offset, "gnbreg")
    count = common.design_block(design, w, spec.outcome, offset)
    dispersion = common.design_block(shape, w, "lnalpha")
    pieces = _Guarded(CountPieces(NegBinDensity("mean"), y))
    model = Model(pieces, [count, dispersion], w)

    # Constant-dispersion negative binomial model: starting values and the nbreg comparison.
    mu = poisson.state.mu
    moment = float((w * ((y - mu).square() - mu)).sum() / (w * mu.square()).sum())
    moment = min(max(moment if math.isfinite(moment) else 1.0, 0.05), 50.0)
    plain = Model(pieces, [count, common.scalar_block("/lnalpha")], w)
    plain_start = torch.cat([poisson.state.beta,
                             torch.tensor([math.log(moment)], dtype=torch.float64)])
    try:
        nbreg, _ = common.run(plain, plain_start, what="negative binomial", weights=weights,
                              callback=common.alpha_watch(k))
    except AnalysisError as exc:
        if exc.code != "precision_unsupported":
            raise
        nbreg = None
    nbreg = nbreg if nbreg is not None and nbreg.converged else None

    def start_from(beta: Tensor, log_alpha: float) -> Tensor:
        tail = torch.zeros(q, dtype=torch.float64)
        tail[0] = log_alpha
        return torch.cat([beta, tail])

    starts = []
    if nbreg is not None:
        starts.append(start_from(nbreg.theta[:k], float(nbreg.theta[k])))
    starts.extend([start_from(poisson.state.beta, math.log(moment)),
                   start_from(poisson.state.beta, math.log(0.1))])

    def alphas(theta: Tensor) -> Tensor:
        return torch.exp(model.objective.indices(theta)[1])

    def watch(iteration: int, theta: Tensor, value: float) -> None:
        if iteration >= _WATCH_FROM and iteration % _WATCH_EVERY == 0 \
                and float(alphas(theta).max()) < 1e-9:
            raise Boundary(theta.clone(), "alpha")

    result, theta, message = None, starts[0], ""
    precision_error = None
    attempted = False
    for start in starts:
        try:
            attempt, stop = common.run(model, start, what="generalized negative binomial",
                                       weights=weights, callback=watch)
        except AnalysisError as exc:
            if exc.code != "precision_unsupported":
                raise
            precision_error = exc
            continue
        attempted = True
        if attempt is not None and attempt.converged:
            block = slice(k, k + q)
            if float(alphas(attempt.theta).min()) < common.ALPHA_BOUNDARY \
                    and common.flattest_information(-attempt.hessian[block, block],
                                                    dispersion.x, weights) \
                    < common.FLAT_INFORMATION:
                # "Converged" where gradient and curvature of ln(alpha) have vanished for
                # the observations whose dispersion ran to zero: a boundary, not a maximum.
                theta, message = attempt.theta, "the dispersion runs to zero"
                continue
            result = attempt
            break
        theta = stop.theta if stop is not None else attempt.theta
        message = "the dispersion runs to zero" if stop is not None \
            else str(attempt.diagnostics.get("message"))
    if result is None:
        if precision_error is not None and not attempted:
            raise precision_error
        fitted_alpha = alphas(theta)
        if model.objective.precision_rejected or precision_error is not None:
            block = slice(k, k + q)
            hessian = model.objective(theta)[2]
            collapsed_mask = fitted_alpha < common.ALPHA_BOUNDARY
            certified = False
            if bool(collapsed_mask.any()):
                # A feasible direction changes only the collapsed observations' index.
                # This independently distinguishes a genuine groupwise Poisson limit
                # from a generic trial that exceeded the gamma precision region.
                target = -collapsed_mask.to(torch.float64)
                direction = torch.linalg.lstsq(dispersion.x, target).solution
                projection = dispersion.x @ direction
                feasible = float((projection - target).abs().max()) < 1e-8
                denominator = (w * projection.square()).sum()
                weak = float(direction @ (-hessian[block, block]) @ direction) * nobs / float(denominator)
                eta = model.objective.indices(theta)[0]
                mu_limit = torch.exp(eta)
                subset = collapsed_mask
                relative_alpha = fitted_alpha[subset] / fitted_alpha[subset].max()
                score = (w[subset] * relative_alpha * ((y[subset] - mu_limit[subset]).square() - y[subset]) / 2).sum()
                rows = model.objective.observations(theta)
                limit_rows = poisson_logmass(y, mu_limit, eta=eta)
                difference = (w[subset] * (rows[subset] - limit_rows[subset])).sum()
                reference = (w[subset] * limit_rows[subset]).sum()
                certified = (feasible and float(score) <= 0 and abs(weak) < common.FLAT_INFORMATION
                             and abs(float(difference)) <= 1e-6 * max(1., abs(float(reference))))
            if not certified:
                raise AnalysisError("precision_unsupported", "gnbreg: numerical count/shape precision prevented a certified maximum; no rounded likelihood is reported.")
        if float(fitted_alpha.max()) < common.ALPHA_BOUNDARY:
            raise AnalysisError(
                "boundary_solution",
                "gnbreg: the estimated overdispersion lies at zero for every observation: the "
                "data are not overdispersed relative to Poisson, so the likelihood has no "
                "interior maximum. Use oe.poisson, with covariance='robust' if the variance "
                "may differ from the mean.")
        collapsed = int((fitted_alpha < common.ALPHA_BOUNDARY).sum())
        if collapsed and bool(torch.isfinite(fitted_alpha).all()) \
                and float(fitted_alpha.max()) <= 1e6:
            raise AnalysisError(
                "boundary_solution",
                f"gnbreg: the estimated overdispersion runs to zero for {collapsed} of the "
                f"{frame.n} observations: the lnalpha equation singles out observations whose "
                "counts are not overdispersed relative to Poisson, so its coefficients diverge "
                "and the likelihood has no interior maximum. Remove the lnalpha regressor that "
                "identifies them or merge that category with another one.")
        hint = ""
        if collapsed or not bool(torch.isfinite(fitted_alpha).all()) \
                or float(fitted_alpha.max()) > 1e6:
            hint = ("ln(alpha) diverges for part of the sample: a regressor of the lnalpha "
                    "equation may identify observations without (or with unbounded) "
                    "overdispersion; remove or combine it. ")
        raise common.not_converged("gnbreg", message, hint + "Check the scale of the "
                                   "regressors of both equations.")

    log_likelihood = result.value
    theta, covariance, info = common.covariance_of(frame, model, result.theta, result.hessian,
                                                   weights)
    null = None
    if spec.intercept:
        if k == 1:
            null = log_likelihood
        else:
            constant = common.constant_block(frame.n, spec.outcome, offset)
            try:
                fitted, _ = common.run(Model(pieces, [constant, dispersion], w),
                                       torch.cat([result.theta[:1], result.theta[k:]]),
                                       what="gnbreg (constant-only model)", weights=weights)
            except AnalysisError as exc:
                if exc.code != "precision_unsupported":
                    raise
                fitted = None
                frame.warn("The optional constant-only comparison exceeded count/shape precision; the model Wald test is reported.")
            if fitted is not None and fitted.converged:
                null = fitted.value
            else:
                frame.warn("The constant-only comparison model did not converge; the Wald "
                           "test of the mean equation's slopes is reported instead of the LR "
                           "test.")
    slopes = slope_indices(design.terms)
    tests: dict[str, Any] = {"model": common.model_test(frame, theta, covariance, slopes,
                                                        log_likelihood, null, "mean")}
    if q > 1:
        if spec.covariance == "nonrobust" and nbreg is not None:
            tests["lnalpha"] = lr_test(
                log_likelihood, nbreg.value, q - 1,
                label="LR chi2 test that the slopes of the lnalpha equation are zero "
                      "(constant alpha: nbreg)")
        else:
            tests["lnalpha"] = wald_test(
                theta, covariance, list(range(k + 1, k + q)),
                label="Wald chi2 test that the slopes of the lnalpha equation are zero "
                      "(constant alpha: nbreg)")
    criteria = information_criteria(log_likelihood, k + q, nobs)
    metrics = {"log_likelihood": log_likelihood,
               "pseudo_r_squared": common.pseudo_r_squared(log_likelihood, null),
               "aic": criteria["aic"], "bic": criteria["bic"]}
    fitted_alpha = alphas(result.theta)
    extra = {
        "null_log_likelihood": null,
        "null_model": None if null is None else
        "constant-only mean equation, full lnalpha equation",
        "nbreg_log_likelihood": None if nbreg is None else nbreg.value,
        "poisson_log_likelihood": poisson.log_likelihood(),
        "alpha": {"mean": float((w * fitted_alpha).sum() / w.sum()),
                  "min": float(fitted_alpha.min()), "max": float(fitted_alpha.max()),
                  "definition": "alpha_i = exp(z_i'd) over the estimation sample"},
        "variance_function": "mu (1 + alpha_i mu)",
        "starting_values": "nbreg estimates" if nbreg is not None else "poisson_and_moments"}
    return build_result(
        frame, terms=model.terms, params=theta, covariance=covariance,
        equations=model.equations, use_t=False, metrics=metrics,
        fitted=torch.exp(model.objective.indices(result.theta)[0]), nobs=nobs, inference=info,
        tests=tests, extra=extra, categories={**design.categories, **shape.categories},
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result), df_resid=nobs - k - q)


def gnbreg(*, data: Any, y: str, x: Sequence[str], lnalpha: Sequence[str] | None = None,
           covariance: str | None = None, cluster: str | Sequence[str] | None = None,
           weights: str | None = None, weight_type: str | None = None,
           offset: str | None = None, exposure: str | None = None,
           categorical: Sequence[str] | None = None, intercept: bool = True,
           missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Generalized negative binomial regression, Stata's ``gnbreg``.

    Model
        ``E[y|x] = mu = exp(x'b + offset)`` and ``Var(y|x, z) = mu (1 + alpha mu)``
        with an overdispersion that depends on covariates, ``ln(alpha) = z'd``. The
        ``lnalpha`` equation always has a constant; without regressors the model is
        :func:`nbreg`. With ``m = 1/alpha`` the log likelihood of an observation is

            lnG(y+m) - lnG(m) - lnG(y+1) - (y+m) ln(1 + alpha mu) + y ln(alpha mu).

        ``(b, d)`` are estimated jointly by Newton-Raphson with analytic score and
        Hessian, starting from the constant-alpha negative binomial fit.

    Parameters
        data: the data set (pandas DataFrame, dict of columns or row records).
        y: nonnegative count outcome.
        x: list of regressors of the mean equation (``[]``: constant only).
        lnalpha: list of regressors of ``ln(alpha)`` (default: constant only). Terms
            are reported as ``lnalpha:<name>``; a positive coefficient means more
            overdispersion.
        covariance: ``'nonrobust'`` (default: inverse observed information),
            ``'opg'``, ``'robust'`` (sandwich times ``N/(N-1)``), ``'cluster'``
            (``G/(G-1)``; one or two cluster columns).
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'iweight'``,
            ``'pweight'`` (robust by default).
        offset / exposure: offset of the mean equation, or exposure entering as
            ``ln(exposure)``. Mutually exclusive.
        categorical: columns of ``x`` or ``lnalpha`` to expand into indicator terms.
        intercept: include the constant of the mean equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing inputs.
        alpha: significance level of the confidence intervals (default 0.05).

    Result
        z statistics. Terms: the mean equation (equation = the outcome) and
        ``lnalpha:<name>`` (equation ``lnalpha``). ``metrics``: log_likelihood,
        pseudo_r_squared, aic, bic. ``tests['model']``: LR chi2 that the slopes of the
        mean equation are zero (``nonrobust``; the comparison model keeps the
        ``lnalpha`` equation) or their Wald chi2. ``tests['lnalpha']``: test that the
        ``lnalpha`` slopes are zero (constant alpha, i.e. :func:`nbreg`): LR under
        ``nonrobust``, Wald otherwise. ``extra['alpha']``: mean, minimum and maximum of
        the fitted ``alpha_i``; ``extra`` also holds the nbreg, Poisson and
        constant-only log likelihoods.

        Data without overdispersion raise ``boundary_solution`` (use :func:`poisson`),
        also when only the observations singled out by a regressor of the ``lnalpha``
        equation lack it (remove that regressor). A regressor of the mean equation
        that singles out observations whose outcomes are all zero raises
        ``separation_detected``.

    Stata
        ``gnbreg deaths age_mos, lnalpha(smokes) exposure(pyears)`` is
        ``oe.gnbreg(data=df, y='deaths', x=['age_mos'], lnalpha=['smokes'],
        exposure='pyears')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=3000), "z": rng.integers(0, 2, size=3000)})
        >>> mu, a = np.exp(0.5 + 0.4 * df.x), np.exp(-1.0 + 1.2 * df.z)
        >>> df["y"] = rng.negative_binomial(1 / a, 1 / (1 + a * mu))
        >>> print(oe.gnbreg(data=df, y="y", x=["x"], lnalpha=["z"]).summary())
    """
    spec = build_spec(
        "gnbreg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"lnalpha": column_list(lnalpha, "lnalpha"), "offset": offset,
                 "exposure": exposure})
    return fit(spec, data=data)
