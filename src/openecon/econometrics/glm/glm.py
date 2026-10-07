"""Generalized linear models: Stata's ``glm`` and the core shared by its relatives.

Model
-----
``g(E[y_i]) = eta_i = x_i'b + offset_i`` and ``Var(y_i) = phi V(mu_i)/w_i``
for a family (variance function ``V``) and a link ``g``; see ``families.py``
for every formula. The coefficients maximize the quasi-log-likelihood, i.e.
minimize the deviance ``D = sum_i w_i d(y_i, mu_i)``; they do not depend on
the dispersion ``phi``.

Estimation
----------
``optimizer='ml'`` (Stata's default) is Newton-Raphson on ``-D/2`` with the
analytic OBSERVED Hessian (``engines.optimize.maximize_newton``: Marquardt
steps where the Hessian is not negative definite, step halving, convergence
only at a concave point). ``optimizer='irls'`` is Fisher scoring: weighted QR
least squares of the working response on X with the EXPECTED information
weights, step halving while the deviance rises, convergence on the relative
change in deviance. Both coincide for canonical links. Starting values are
one weighted least-squares step from ``mu_0``: ``y`` (gaussian, gamma, inverse
Gaussian), ``(y + 0.5)/(n + 1)`` (binomial) or ``y + 0.1`` (poisson, negative
binomial).

Dispersion and covariance (Stata's conventions)
-----------------------------------------------
``phi`` is 1 for binomial, poisson and nbinomial and the Pearson statistic
over the residual degrees of freedom (``scale='x2'``) for gaussian, gamma and
inverse Gaussian; ``scale='dev'`` uses the deviance instead and a number fixes
it. With ``I`` the unit-dispersion information (observed for ``ml``, expected
for ``irls``) and ``S`` the unit-dispersion score rows:

    nonrobust   phi I^-1                    (vce(oim) / vce(eim))
    opg         phi^2 (S'S)^-1
    robust      N/(N-1)  I^-1 S'S I^-1      (not scaled by phi)
    cluster     G/(G-1)  I^-1 (sum_g s_g s_g') I^-1

The scores of the likelihood with dispersion ``phi`` are ``S/phi`` and its
information is ``I/phi``, which is where the powers of ``phi`` come from: the
OPG ``((S/phi)'(S/phi))^-1`` is ``phi^2 (S'S)^-1`` and the sandwich is free of
``phi``. Coefficient tests are z tests. ``df_resid = N - K`` enters only ``phi``.

Log likelihood (as Stata's glm reports it)
------------------------------------------
The reported log likelihood is the full family likelihood, normalizing terms
included, and never depends on ``scale`` (which only multiplies the
covariance). Binomial, poisson and nbinomial have no dispersion. For the
Gaussian family the dispersion is concentrated out, ``phi = deviance/N``,
which reproduces the log likelihood of ``regress``
(``-N/2 (1 + ln 2pi + ln(RSS/N))``). For gamma and inverse Gaussian Stata
evaluates the likelihood at ``phi = 1``:

    gamma             sum w [-(y/mu + ln mu)]
    inverse Gaussian  sum w [-(y - mu)^2 / (2 y mu^2) - ln(2 pi y^3)/2]

(checked against Stata's e(ll), e(aic) and e(bic) in the tests). ``aic``/
``bic`` follow ``estat ic`` and ``aic_glm``/``bic_glm`` are the two
statistics Stata's glm header prints: ``(-2 ll + 2K)/N`` and
``deviance - df_resid ln N``.

Separation
----------
A binomial model whose iteration does not converge while observations with
fitted probabilities numerically 0 or 1 are all perfectly predicted has a
monotone likelihood (complete or quasi-complete separation): the fit is
rejected with ``separation_detected`` instead of reporting huge coefficients.

Boundary solutions
------------------
Links that do not keep the mean inside the family's support (binomial with
``log`` or ``identity``; poisson and nbinomial with ``identity`` or a power
link with exponent >= 1) can have their maximum where a fitted mean sits on
the edge of the link's domain (a probability of 1, a count mean of 0) with
finite coefficients. The score is not zero there and the information weights
of the edge observations are unbounded, so neither optimizer yields valid
standard errors: a fit that ends within 1e-6 of such an edge, converged or
not, is rejected with ``boundary_solution`` and the advice to change the link.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call, wald_test,
)
from openecon.econometrics.glm.common import (
    Weights, build_spec, center_design, check_pweights, count_outcome, likelihood_covariance,
    likelihood_weights, linear_offset, maximize, optimizer_record, require_observations,
    require_terms, resolve_covariance, slope_indices, uncenter,
)
from openecon.econometrics.glm.families import (
    CANONICAL_LINK, FAMILY_LINKS, Family, Link, make_family, make_link,
)
from openecon.econometrics.glm.kernels import GlmObjective, GlmState
from openecon.models import ModelSpec, ResultBundle

# Fitted probabilities closer than this to 0 or 1 count as numerically degenerate.
_EXTREME = 1e-6
# Early exit of a separated Newton run: checked every _WATCH_EVERY iterations from
# iteration _WATCH_FROM on, with a much stricter degeneracy threshold.
_EXTREME_EARLY = 1e-10
_WATCH_FROM, _WATCH_EVERY = 20, 5
# A Pearson/deviance statistic below this fraction of its natural scale is an exact fit.
_EXACT_FIT = 1e-24


@dataclass
class GlmFit:
    """A fitted GLM before dispersion, covariance and reporting choices.

    ``objective``, ``state`` and ``hessian`` live in the coordinates of the
    centered design (``common.center_design``); ``beta`` and ``covariance``
    are mapped back to the original columns.
    """

    design: Design
    objective: GlmObjective
    state: GlmState
    weights: Weights
    family: Family
    link: Link
    hessian: Tensor            # unit-dispersion Hessian (minus the Fisher information for irls)
    information: str           # "observed" or "expected"
    iterations: int
    optimizer: dict[str, Any]
    solver: str
    df_resid: int
    pearson: float
    means: Tensor | None = None    # centering of the non-constant columns, if any

    @property
    def beta(self) -> Tensor:
        return uncenter(self.state.beta, None, [(0, self.means)])[0]

    @property
    def deviance(self) -> float:
        # An exact fit sums to -0.0 (or to rounding below zero); report a clean zero.
        return self.state.deviance if self.state.deviance > 0 else 0.0

    @property
    def nobs(self) -> int:
        return self.weights.nobs

    def fitted(self) -> Tensor:
        """Fitted means on the outcome's own scale (counts for binomial trials)."""
        trials = getattr(self.family, "trials", None)
        return self.state.mu if trials is None else self.state.mu * trials

    def log_likelihood(self, phi: float = 1.0) -> float:
        return self.objective.log_likelihood(self.state, phi, self.weights.user)

    def covariance(self, frame: ModelFrame, phi: float = 1.0) -> tuple[Tensor, dict[str, Any]]:
        """Covariance by ``spec.covariance``: ``phi I^-1``, ``phi^2 (S'S)^-1`` or a sandwich."""
        covariance, info = likelihood_covariance(
            frame, self.hessian, lambda: self.objective.score_rows(self.state), self.weights)
        covariance = uncenter(self.state.beta, covariance, [(0, self.means)])[1]
        kind = frame.spec.covariance
        if kind == "nonrobust":
            info["correction"] = f"{self.information} information"
        elif kind in {"robust", "cluster"}:
            info["correction"] += f"; bread: {self.information} information"
        if kind == "nonrobust" and phi != 1.0:
            covariance = covariance * phi
            info["correction"] += f" times dispersion {phi:.6g}"
        elif kind == "opg" and phi != 1.0:
            # Scores of the dispersion-phi likelihood are S/phi: ((S/phi)'(S/phi))^-1.
            covariance = covariance * phi ** 2
            info["correction"] += f" times squared dispersion {phi:.6g}^2"
        info["dispersion"] = phi
        info["df_inference"] = None
        return covariance, info


class _Separated(Exception):
    """Raised from the Newton callback to stop a separated binomial fit early."""

    def __init__(self, theta: Tensor):
        super().__init__("separated")
        self.theta = theta


def _separated(objective: GlmObjective, state: GlmState | None, threshold: float) -> int:
    """Number of perfectly predicted observations with degenerate fitted probabilities.

    Zero unless such observations exist AND each of them is predicted
    correctly, the signature of a likelihood that keeps rising along a
    separating direction.
    """
    if state is None or not objective.family.binomial:
        return 0
    low, high = state.mu < threshold, state.comp < threshold
    extreme = low | high
    if not bool(extreme.any()):
        return 0
    perfect = torch.where(low, objective.y == 0, objective.y == 1)
    return int(extreme.sum()) if bool(perfect[extreme].all()) else 0


def _domain_edge(objective: GlmObjective, state: GlmState) -> int:
    """Observations whose fitted mean sits at a FINITE edge of the link's domain.

    Binomial: the log link reaches ``mu = 1`` at ``eta = 0`` and the identity
    link reaches ``mu = 0`` and ``mu = 1`` at ``eta = 0`` and ``eta = 1``.
    Poisson / negative binomial: the identity link (and a power link with
    exponent >= 1) reaches ``mu = 0`` at ``eta = 0``. None of these keeps the
    mean inside the family's support, so the likelihood can be maximized ON
    that edge with finite coefficients: a constrained maximum where the score
    is not zero and where the information weights ``(dmu/deta)^2 / V`` of the
    edge observations are unbounded, so neither Newton nor IRLS yields valid
    standard errors there. The other links only reach the edge as ``eta``
    diverges (separation). Gamma and inverse Gaussian likelihoods tend to
    minus infinity at ``mu = 0`` and never have such a maximum.
    """
    link, family = objective.link, objective.family
    if family.binomial:
        if link.name == "log":
            return int((state.comp < _EXTREME).sum())
        if link.name == "identity":
            return int(((state.mu < _EXTREME) | (state.comp < _EXTREME)).sum())
        return 0
    if family.name in {"poisson", "nbinomial"} and (
            link.name == "identity" or getattr(link, "exponent", 0.0) >= 1):
        prior = objective.prior
        level = float((prior * objective.y).sum() / prior.sum())
        return int((state.mu < _EXTREME * level).sum())
    return 0


def _check_edge(objective: GlmObjective, state: GlmState | None, command: str) -> None:
    """Refuse a fit that sits on a finite edge of the link's domain (see _domain_edge)."""
    edge = 0 if state is None else _domain_edge(objective, state)
    if not edge:
        return
    link = objective.link.name
    if objective.family.binomial:
        what = f"fitted probabilit{'y' if edge == 1 else 'ies'}"
        where = "1" if link == "log" else "0 or 1"
        keeps = "probabilities inside (0, 1)"
        advice = ("Use link='logit', 'probit' or 'cloglog'"
                  + ("; for risk ratios, oe.poisson with covariance='robust' estimates the same "
                     "log-linear mean without the constraint." if link == "log" else
                     "; for risk differences, least squares with a robust covariance estimates "
                     "the same linear mean without the constraint."))
    else:
        what = f"fitted mean{'' if edge == 1 else 's'}"
        where, keeps = "0", "means positive"
        advice = ("Use link='log' (the family's default), or least squares with a robust "
                  "covariance for a linear mean.")
    raise AnalysisError(
        "boundary_solution",
        f"{command}: the likelihood is maximized on the boundary of the {link} link's domain, "
        f"where {edge} {what} reach{'es' if edge == 1 else ''} {where}: the {link} link does "
        f"not keep {keeps}, the score is not zero at such a maximum and the usual standard "
        f"errors do not apply. {advice}")


def _diagnose(objective: GlmObjective, beta: Tensor, message: str, command: str) -> None:
    """Raise the most informative error for a GLM iteration that did not converge."""
    state = objective.state(beta) if bool(torch.isfinite(beta).all()) else None
    _check_edge(objective, state, command)
    count = _separated(objective, state, _EXTREME)
    if count:
        raise AnalysisError(
            "separation_detected",
            f"{command}: {count} observation(s) are predicted perfectly (fitted probabilities "
            "numerically 0 or 1) and the likelihood keeps increasing: complete or "
            "quasi-complete separation, so finite maximum likelihood estimates do not exist. "
            "Remove or combine the regressor (or category) that separates the outcome.")
    hint = ""
    if state is not None and not objective.family.binomial and bool((state.mu < 1e-10).any()):
        hint = (" Some fitted means are numerically zero: a regressor may predict zero "
                "outcomes perfectly (separation).")
    raise AnalysisError("nonconvergence", f"{command} did not converge: {message}{hint} "
                        "Check the scale of the regressors, the link and the starting sample.")


def estimate(frame: ModelFrame, design: Design, y: Tensor, family: Family, link: Link, *,
             weights: Weights, offset: Tensor | None, optimizer: str = "ml",
             tolerance: float = 1e-10, max_iterations: int = 100,
             command: str = "glm") -> GlmFit:
    """Fit one GLM on a screened design (``y`` on the working scale of ``family``)."""
    trials = getattr(family, "trials", None)
    prior = weights.user if trials is None else weights.user * trials
    x, means = center_design(design, prior)
    objective = kernel_call(GlmObjective, x, y, prior, offset, family, link, trials)
    k = design.x.shape[1]
    require_observations(weights.nobs, k)
    if k == 0:
        state = objective.state(torch.zeros(0, dtype=torch.float64))
        if state is None:
            raise AnalysisError("invalid_start", f"{command}: the offset alone is outside the "
                                "link's domain.")
        return GlmFit(design, objective, state, weights, family, link,
                      torch.zeros((0, 0), dtype=torch.float64), "observed", 0,
                      {"method": "none", "iterations": 0, "converged": True}, "none",
                      weights.nobs, objective.pearson(state))
    start = kernel_call(objective.start, design.intercept)
    if optimizer == "irls":
        run = kernel_call(objective.irls, start, tol=tolerance, max_iter=max_iterations)
        if not run.converged:
            _diagnose(objective, run.state.beta, run.message, command)
        state = run.state
        _check_edge(objective, state, command)
        hessian, information = -objective.fisher_information(state), "expected"
        record = {"method": "irls_fisher_scoring_qr", "iterations": run.iterations,
                  "converged": True, "gradient_max": objective.gradient_max(state),
                  "message": run.message, "tolerance": tolerance, "maxiter": max_iterations}
        iterations, solver = run.iterations, "irls_fisher_scoring_qr"
    else:
        def watch(iteration: int, theta: Tensor, value: float) -> None:
            # A well-posed binomial fit converges long before this; one that still moves
            # with perfectly predicted observations at 1e-10 is separated, and iterating
            # to the limit would only waste O(n) passes.
            if family.binomial and iteration >= _WATCH_FROM and iteration % _WATCH_EVERY == 0 \
                    and _separated(objective, objective.state(theta), _EXTREME_EARLY):
                raise _Separated(theta.clone())

        try:
            result = maximize(objective, start, what=command, max_iter=max_iterations,
                              tolerance=tolerance, callback=watch)
        except _Separated as stop:
            _diagnose(objective, stop.theta, "the likelihood is monotone", command)
        if not result.converged:
            _diagnose(objective, result.theta, str(result.diagnostics.get("message")), command)
        state = objective.state(result.theta)
        if state is None:
            raise AnalysisError("numerical_failure",
                                f"{command}: the fitted values are not finite.")
        _check_edge(objective, state, command)
        hessian, information = result.hessian, "observed"
        record = {**optimizer_record(result), "tolerance": tolerance, "maxiter": max_iterations}
        iterations, solver = result.iterations, "newton_observed_hessian"
    return GlmFit(design, objective, state, weights, family, link, hessian, information,
                  iterations, record, solver, weights.nobs - k, objective.pearson(state), means)


def null_log_likelihood(fit_: GlmFit, frame: ModelFrame, y: Tensor, offset: Tensor | None,
                        *, intercept: bool, command: str) -> float:
    """Log likelihood of the constant-only model with the same offset and weights.

    Without a constant in the model the comparison point is ``b = 0``
    (``eta = offset``). Without an offset the constant-only fit is closed
    form: every fitted mean equals the weighted mean of the outcome.
    """
    family, link, weights = fit_.family, fit_.link, fit_.weights
    objective = fit_.objective
    if not intercept:
        empty = Design(torch.empty((frame.n, 0), dtype=torch.float64), [], {}, False)
        return estimate(frame, empty, y, family, link, weights=weights, offset=offset,
                        command=command).log_likelihood()
    ones = Design(torch.ones((frame.n, 1), dtype=torch.float64), ["Intercept"], {}, True)
    if offset is None:
        prior = objective.prior
        mean = (prior * y).sum() / prior.sum()
        null = GlmObjective(ones.x, y, prior, None, family, link, objective.mult)
        state = null.state(link.link(mean).reshape(1))
        if state is None:
            raise AnalysisError("constant_outcome", f"{command}: the constant-only model has no "
                                "finite likelihood; the outcome does not vary.")
        return null.log_likelihood(state, 1.0, weights.user)
    return estimate(frame, ones, y, family, link, weights=weights, offset=offset,
                    command=command).log_likelihood()


# ---- glm: options, outcome, dispersion ---------------------------------------------------


def _settings(frame: ModelFrame) -> dict[str, Any]:
    spec = frame.spec
    family = frame.option("family")
    link = frame.option("link") or CANONICAL_LINK[family]
    power, k = frame.option("power"), frame.option("dispersion")
    if (link == "power") != (power is not None):
        raise AnalysisError("invalid_option", "power is the exponent of link='power': give both "
                            "(for example link='power', power=0.5) or neither.")
    if "dispersion" in spec.options and family != "nbinomial":
        raise AnalysisError("invalid_option", "dispersion is the fixed overdispersion of "
                            "family='nbinomial'; use scale for the covariance multiplier.")
    if not k > 0:
        raise AnalysisError("invalid_option", "dispersion must be a positive number (use "
                            "oe.nbreg to estimate it).")
    allowed = FAMILY_LINKS[family]
    if link not in allowed:
        raise AnalysisError("invalid_option", f"link='{link}' is not available for "
                            f"family='{family}' (allowed: {', '.join(allowed)}).")
    scale = frame.option("scale")
    if scale is not None and scale not in ("x2", "dev"):
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) \
                or not math.isfinite(scale) or scale <= 0:
            raise AnalysisError("invalid_option", "scale must be 'x2', 'dev' or a positive "
                                "number.")
    tolerance = frame.option("tolerance")
    if not 0 < tolerance < 1:
        raise AnalysisError("invalid_option", "tolerance must be a number in (0, 1).")
    return {"family": family, "link": link, "power": power, "k": float(k), "scale": scale,
            "optimizer": frame.option("optimizer"), "tolerance": float(tolerance),
            "max_iterations": frame.option("max_iterations")}


def binomial_outcome(frame: ModelFrame, y: Tensor, trials: Tensor | None,
                     command: str) -> Tensor:
    """Validate a binary / binomial outcome and return it on the proportion scale."""
    name = frame.spec.outcome
    if trials is None:
        if not bool(((y == 0) | (y == 1)).all()):
            raise AnalysisError("invalid_binary_outcome", f"{command} needs '{name}' coded 0/1 "
                                "(give trials= for binomial counts, or use oe.fracreg for "
                                "proportions).")
        proportion = y
    else:
        if bool((trials < 1).any()) or bool((trials != trials.round()).any()):
            raise AnalysisError("invalid_trials", "The binomial trials must be positive integers.")
        if bool((y < 0).any()) or bool((y > trials).any()) or bool((y != y.round()).any()):
            raise AnalysisError("invalid_binomial_outcome", f"With trials, '{name}' must be an "
                                "integer count of successes between 0 and the trials.")
        proportion = y / trials
    if bool((proportion == 0).all()) or bool((proportion == 1).all()):
        raise AnalysisError("constant_outcome", f"'{name}' does not vary (all failures or all "
                            "successes); the model cannot be estimated.")
    return proportion


def _outcome(frame: ModelFrame, family: str) -> tuple[Tensor, Tensor | None]:
    y = frame.numeric(frame.spec.outcome)
    trials_role = frame.role("trials")
    if trials_role and family != "binomial":
        raise AnalysisError("invalid_spec", "trials is the binomial denominator and needs "
                            "family='binomial'.")
    if family == "binomial":
        trials = frame.numeric(trials_role[0]) if trials_role else None
        return binomial_outcome(frame, y, trials, "glm (binomial)"), trials
    if family in {"poisson", "nbinomial"}:
        count_outcome(frame, y, f"glm ({family})")
    elif family in {"gamma", "inverse_gaussian"} and bool((y <= 0).any()):
        raise AnalysisError("invalid_positive_outcome", f"family='{family}' needs a strictly "
                            f"positive outcome; '{frame.spec.outcome}' has values <= 0.")
    return y, None


def _statistic_scale(family: Family, y: Tensor, prior: Tensor) -> float:
    """The size a Pearson/deviance statistic has when the fit is NOT exact.

    ``sum w y^2 / V(y)`` up to constants: an estimated dispersion below
    ``_EXACT_FIT`` times this is rounding noise (float64 cannot resolve a fit
    that close to exact), as in ``regress``.
    """
    if family.name == "gaussian":
        return float((prior * y.square()).sum())
    if family.name == "inverse_gaussian":
        return float((prior / y).sum())
    if family.name in {"poisson", "nbinomial"}:
        return float((prior * (1 + y)).sum())
    return float(prior.sum())


def dispersion(rule: Any, family: Family, deviance: float, pearson: float, df_resid: int,
               reference: float = 0.0) -> tuple[float, str]:
    """The dispersion ``phi`` and the rule that produced it."""
    rule = family.default_scale if rule is None else rule
    if rule in ("x2", "dev"):
        if df_resid <= 0:
            raise AnalysisError("insufficient_observations", "The dispersion needs positive "
                                "residual degrees of freedom (more observations than "
                                "parameters).")
        statistic = pearson if rule == "x2" else deviance
        if not statistic > _EXACT_FIT * reference:
            raise AnalysisError("perfect_fit", "The model fits the outcome exactly, so the "
                                "estimated dispersion and every standard error are zero. Remove "
                                "the regressor that reproduces the outcome, or fix the scale.")
        return statistic / df_resid, "Pearson chi2 / df" if rule == "x2" else "deviance / df"
    return float(rule), "fixed"


def fit_glm(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``glm``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    settings = _settings(frame)
    weights = likelihood_weights(frame)
    offset = linear_offset(frame)
    y, trials = _outcome(frame, settings["family"])
    family = kernel_call(make_family, settings["family"], k=settings["k"], trials=trials)
    link = kernel_call(make_link, settings["link"], power=settings["power"], k=settings["k"])
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    require_terms(design)
    fitted = estimate(frame, design, y, family, link, weights=weights, offset=offset,
                      optimizer=settings["optimizer"], tolerance=settings["tolerance"],
                      max_iterations=settings["max_iterations"])
    k, nobs, df_resid = len(design.terms), fitted.nobs, fitted.df_resid
    reference = _statistic_scale(family, y, fitted.objective.prior)
    phi, rule = dispersion(settings["scale"], family, fitted.deviance, fitted.pearson, df_resid,
                           reference)
    covariance, info = fitted.covariance(frame, phi)
    # Stata's glm: the likelihood does not use the covariance scale. Gaussian: the dispersion
    # is concentrated out (phi = deviance/N, regress's log likelihood); gamma and inverse
    # Gaussian: phi = 1; the other families have no dispersion.
    concentrated = family.name == "gaussian"
    phi_ll = fitted.deviance / nobs if concentrated else 1.0
    if concentrated and not fitted.deviance > _EXACT_FIT * reference:
        raise AnalysisError("perfect_fit", "The model fits the outcome exactly, so the Gaussian "
                            "log likelihood is unbounded. Remove the regressor that reproduces "
                            "the outcome.")
    log_likelihood = fitted.log_likelihood(phi_ll)
    criteria = information_criteria(log_likelihood, k, nobs)
    metrics = {
        "deviance": fitted.deviance, "pearson": fitted.pearson,
        "dispersion_deviance": fitted.deviance / df_resid if df_resid > 0 else None,
        "dispersion_pearson": fitted.pearson / df_resid if df_resid > 0 else None,
        "scale": phi, **criteria, "aic_glm": criteria["aic"] / nobs,
        "bic_glm": fitted.deviance - df_resid * math.log(nobs), "df_resid": df_resid,
        "iterations": fitted.iterations,
    }
    info["scale_rule"] = rule
    tests = {"model": wald_test(fitted.beta, covariance, slope_indices(design.terms),
                                label="Wald chi2 test of the slopes")}
    extra = {"family": family.name, "link": link.name, "variance_function": family.text,
             "link_function": _LINK_TEXT.get(settings["link"], link.name),
             "scale_rule": rule, "optimizer": settings["optimizer"],
             "information": fitted.information, "canonical_link": CANONICAL_LINK[family.name],
             "log_likelihood_dispersion": None if family.default_scale == 1.0 else phi_ll,
             "log_likelihood_dispersion_rule":
                 "deviance / N (concentrated)" if concentrated
                 else "none (the family's likelihood has no dispersion)"
                 if family.default_scale == 1.0 else "1 (as Stata's glm)"}
    if family.name == "nbinomial":
        extra["nbinomial_dispersion"] = settings["k"]
    return build_result(
        frame, terms=design.terms, params=fitted.beta, covariance=covariance, use_t=False,
        df_resid=df_resid, metrics=metrics, fitted=fitted.fitted(), nobs=nobs, inference=info,
        tests=tests, extra=extra, categories=design.categories, solver=fitted.solver,
        solver_diagnostics={"converged": True, "iterations": fitted.iterations},
        optimizer=fitted.optimizer)


_LINK_TEXT = {
    "identity": "eta = mu", "log": "eta = ln(mu)", "logit": "eta = ln(mu / (1 - mu))",
    "probit": "eta = invnormal(mu)", "cloglog": "eta = ln(-ln(1 - mu))",
    "loglog": "eta = -ln(-ln(mu))", "power": "eta = mu^power", "reciprocal": "eta = 1 / mu",
    "inverse_squared": "eta = 1 / mu^2", "nbinomial": "eta = ln(k mu / (1 + k mu))",
}


def glm(*, data: Any, y: str, x: Sequence[str], family: str = "gaussian",
        link: str | None = None, covariance: str | None = None,
        cluster: str | Sequence[str] | None = None, weights: str | None = None,
        weight_type: str | None = None, offset: str | None = None, exposure: str | None = None,
        trials: str | None = None, scale: str | float | None = None,
        dispersion: float | None = None, power: float | None = None, optimizer: str = "ml",
        max_iterations: int = 100, tolerance: float = 1e-10,
        categorical: Sequence[str] | None = None, intercept: bool = True,
        missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Generalized linear model, Stata's ``glm``.

    Model
        ``g(E[y]) = x'b + offset`` with ``Var(y) = phi V(mu) / w``. The coefficients
        minimize the deviance ``D = sum w d(y, mu)`` (maximum quasi-likelihood).

        ============================ ============= ==================================
        family                       V(mu)         default link
        ============================ ============= ==================================
        ``gaussian``                 1             ``identity``
        ``binomial`` (``trials=n``)  mu (1 - mu/n) ``logit``
        ``poisson``                  mu            ``log``
        ``gamma``                    mu^2          ``reciprocal`` (1/mu)
        ``inverse_gaussian``         mu^3          ``inverse_squared`` (1/mu^2)
        ``nbinomial``                mu + k mu^2   ``log`` (k = ``dispersion``, fixed)
        ============================ ============= ==================================

        Links: ``identity``, ``log``, ``logit``, ``probit``, ``cloglog``
        (``ln(-ln(1-mu))``), ``loglog`` (``-ln(-ln(mu))``), ``power`` (``mu^power``),
        ``reciprocal`` (power -1), ``inverse_squared`` (power -2) and ``nbinomial``
        (``ln(k mu/(1 + k mu))``, family nbinomial only). The binomial family takes
        logit, probit, cloglog, loglog, log and identity; the positive families take
        log, identity, power, reciprocal and inverse_squared.

    Estimator
        ``optimizer='ml'``: Newton-Raphson with the analytic observed Hessian (Stata's
        default; reports the OIM covariance). ``optimizer='irls'``: Fisher scoring by
        weighted QR least squares with the expected information (Stata's ``irls``;
        reports the EIM covariance). Identical for canonical links. Collinear terms are
        omitted and recorded. A separated binomial model raises
        ``separation_detected``. A link that does not keep the mean inside the family's
        support (binomial with ``log`` or ``identity``; poisson / nbinomial with
        ``identity``) can have its maximum on the edge of the link's domain, where the
        score is not zero and no valid standard errors exist: ``boundary_solution``.

    Parameters
        data: DataFrame, mapping of columns or list of records.
        y, x: outcome and regressor columns (``x`` is a list; ``x=[]`` fits the
            constant-only model).
        family, link, power: see above; ``power`` only with ``link='power'``.
        dispersion: fixed overdispersion ``k`` of ``family='nbinomial'`` (default 1;
            Stata's ``family(nbinomial #)``). To ESTIMATE it use :func:`nbreg`.
        scale: dispersion ``phi`` multiplying the conventional covariance: ``'x2'``
            (Pearson chi2 / df), ``'dev'`` (deviance / df) or a positive number.
            Default 1 for binomial, poisson, nbinomial and ``'x2'`` otherwise (Stata).
        covariance: ``'nonrobust'`` (default: ``phi`` times the inverse OIM, or EIM
            with irls), ``'opg'`` (``phi^2 (S'S)^-1``, the outer product of the scores
            ``S/phi`` of the dispersion-``phi`` likelihood), ``'robust'`` (Huber-White
            sandwich times ``N/(N-1)``), ``'cluster'`` (``G/(G-1)``; one or two columns,
            two use Cameron-Gelbach-Miller inclusion-exclusion). Robust and cluster
            are not scaled by ``phi``.
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'`` (replicates observations, N = sum of
            weights), ``'aweight'`` (rescaled to sum to N), ``'pweight'`` (sampling
            weights; robust covariance by default and required), ``'iweight'`` (used
            as given).
        offset, exposure: ``offset`` is added to the linear predictor; ``exposure``
            adds ``ln(exposure)`` (mutually exclusive).
        trials: binomial denominator column; ``y`` is then a count ``0..n``. Without
            it the binomial outcome must be 0/1.
        max_iterations, tolerance: iteration limit and convergence tolerance (scaled
            gradient and relative step for ml, relative deviance change for irls).
        categorical: regressors expanded as ``name[level]`` dummies (first level omitted).
        intercept: include the constant (named ``Intercept``).
        missing: ``'raise'`` (default) or ``'drop'``. alpha: 1 - confidence level.

    Result
        z statistics. ``metrics``: deviance, pearson, dispersion_deviance (deviance/df),
        dispersion_pearson (pearson/df), scale (the ``phi`` used), log_likelihood (full
        family log likelihood as Stata reports it: ``phi`` concentrated out,
        ``deviance/N``, for gaussian so that it equals regress's; ``phi = 1`` for gamma
        and inverse Gaussian; never affected by ``scale``),
        aic and bic (``-2ll + 2K``, ``-2ll + K ln N``),
        aic_glm (``(-2ll + 2K)/N``) and bic_glm (``deviance - df_resid ln N``) as Stata's
        glm prints them, df_resid, iterations. ``tests['model']``: Wald chi2 of the
        slopes. ``extra``: family, link, variance and link formulas, scale rule,
        optimizer, information type.

    Stata
        ``glm y x1 x2, family(gamma) link(log) vce(robust)`` is
        ``oe.glm(data=df, y='y', x=['x1', 'x2'], family='gamma', link='log',
        covariance='robust')``; ``glm d x, family(binomial n) link(probit) irls`` is
        ``oe.glm(..., y='d', family='binomial', trials='n', link='probit',
        optimizer='irls')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=500)})
        >>> df["y"] = rng.gamma(2.0, np.exp(0.5 + 0.3 * df.x) / 2.0)
        >>> print(oe.glm(data=df, y="y", x=["x"], family="gamma", link="log").summary())
    """
    spec = build_spec(
        "glm", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"offset": offset, "exposure": exposure, "trials": trials},
        options={"family": family, "link": link, "power": power, "dispersion": dispersion,
                 "scale": scale, "optimizer": optimizer, "max_iterations": max_iterations,
                 "tolerance": tolerance})
    return fit(spec, data=data)
