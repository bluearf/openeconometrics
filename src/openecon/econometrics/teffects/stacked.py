"""Regression adjustment, IPW, IPWRA and AIPW (Stata's teffects ra / ipw / ipwra / aipw).

Notation: treatment levels l = 0..L-1 (0 = control), D_i the observed level,
p_l(x) the treatment-model probability, mu_l(x) = m(x'b_l) the outcome model of
level l (fitted on the observations with D = l), w_i the user weight and
POM_l the potential-outcome mean of level l. Each estimator is the solution
of stacked estimating equations ``sum_i w_i psi_i(theta) = 0`` in
theta = (gamma, b_0..b_{L-1}, POM_0..POM_{L-1}):

treatment   x_t u_s(gamma)                       (logit/probit/mlogit score)
outcome     x_o 1{D=l} omega_l r_l(b_l)          (omega_l = 1 for ra and aipw)
POM, ra     mu_l - POM_l                         (atet: 1{D=t}(mu_l - POM_l))
POM, ipw    1{D=l} omega_l (y - POM_l)           (normalized weights, as Stata)
POM, ipwra  as ra, with outcome models weighted by omega_l
POM, aipw   1{D=l} (y - mu_l) / p_l + mu_l - POM_l

with omega_l = 1/p_l for ate/pomeans and, for atet (binary treatment, treated
level t), omega_t = 1 and omega_0 = p_t / p_0. The reported effects are
ATE_l = POM_l - POM_0 (or ATET), with POmean = POM_0, a linear transformation
of theta. Standard errors come from the sandwich ``A^-1 B A^-T`` of all the
equations (``common.stacked_covariance``), so the estimation of the nuisance
models is accounted for; ``A`` is assembled analytically from the derivative
scalars of ``index.py`` (no autograd, no numerical Jacobian).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, ModelFrame, build_result
from openecon.econometrics.discrete.common import constant_shift
from openecon.econometrics.teffects.common import (
    StudyWeights, Treatment, check_overlap, coefficient_table, propensity_summary,
    stacked_covariance, study_weights, treatment_coding,
)
from openecon.econometrics.teffects.index import (
    TreatmentFit, fit_index, fit_treatment, mean_pieces, residual_pieces,
)
from openecon.models import ResultBundle

METHOD_TITLES = {
    "ra": "regression adjustment", "ipw": "inverse-probability weights",
    "ipwra": "IPW regression adjustment", "aipw": "augmented IPW",
}
_TREATMENT_MODEL = {"ipw", "ipwra", "aipw"}
_OUTCOME_MODEL = {"ra", "ipwra", "aipw"}


@dataclass
class _Equation:
    """One fitted outcome model: kept design columns, coefficients and pieces."""

    terms: list[str]
    means: Tensor              # centring offsets of the kept columns (0 for the constant)
    x: Tensor                  # [n, k] centred design over all rows
    beta: Tensor
    mu: Tensor
    dmu: Tensor
    r: Tensor
    dr: Tensor


def _centred_design(frame: ModelFrame, predictors: Sequence[str], weights: StudyWeights, *,
                    intercept: bool) -> tuple[Design, Tensor]:
    design = frame.design(predictors, intercept=intercept)
    means = torch.zeros(design.x.shape[1], dtype=torch.float64)
    # Centring is an exact reparameterization only when the constant absorbs the shift.
    if design.intercept and design.x.shape[1] > 1:
        first = 1
        w = weights.user
        means[first:] = (design.x[:, first:] * w[:, None]).sum(dim=0) / w.sum()
        design.x = design.x - means
    return design, means


def _screen(frame: ModelFrame, design: Design, rows: Tensor | None, weights: Tensor,
            prefix: str) -> list[int]:
    """Kept columns of ``design`` (screened on ``rows``), omissions recorded with ``prefix``."""
    x = design.x if rows is None else design.x[rows]
    w = weights if rows is None else weights[rows]
    labelled = Design(x, [prefix + term for term in design.terms], design.categories,
                      design.intercept)
    kept = frame.drop_collinear(labelled, w)
    index = {term: i for i, term in enumerate(labelled.terms)}
    return [index[term] for term in kept.terms]


def _outcome_values(frame: ModelFrame, model: str) -> Tensor:
    y = frame.numeric(frame.spec.outcome)
    if model in {"logit", "probit"} and bool(((y < 0) | (y > 1)).any()):
        raise AnalysisError("invalid_outcome", f"omodel='{model}' needs an outcome in [0, 1] "
                            "(0/1, or a fraction for the quasi-likelihood model).")
    if model == "poisson" and bool((y < 0).any()):
        raise AnalysisError("invalid_outcome", "omodel='poisson' needs a nonnegative outcome.")
    return y


def _ipw_weights(fit: TreatmentFit, estimand: str, levels: int, target: int = 1) -> tuple[Tensor, Tensor]:
    """omega [n, L] and d omega_l / d eta_s [n, L, S] (the multiplier of x_t)."""
    p, a = fit.probabilities, fit.slopes
    if estimand == "atet":
        treated = p[:, target:target + 1]
        omega = treated / p
        derivative = (a[:, target:target + 1, :] * p[:, :, None] - treated[:, :, None] * a) \
            / p.square()[:, :, None]
        return omega, derivative
    return p.reciprocal(), -a / p.square()[:, :, None]


def fit_stacked(frame: ModelFrame, method: str) -> ResultBundle:
    """ra / ipw / ipwra / aipw on the prepared frame."""
    spec = frame.spec
    estimand, omodel, tmodel = (frame.option(name) for name in ("estimand", "omodel", "tmodel"))
    if spec.covariance not in {"robust", "cluster"}:
        raise AnalysisError("unsupported_covariance", f"teffects {method} supports covariance="
                            "'robust' (default) or 'cluster', as Stata's vce().")
    weights = study_weights(frame)
    treatment = treatment_coding(frame, frame.role("treatment")[0], frame.option("control"))
    levels, codes, w = treatment.levels, treatment.codes, weights.user
    target_label = frame.option("tlevel")
    target = 1 if target_label is None else treatment.labels.index(target_label) if target_label in treatment.labels else -1
    if target <= 0:
        raise AnalysisError("invalid_treatment", "ATET tlevel must name an observed non-control treatment level.")
    y = _outcome_values(frame, omodel)
    masks = [codes == level for level in range(levels)]
    indicators = [mask.to(torch.float64) for mask in masks]

    fit: TreatmentFit | None = None
    categories = {}
    xt = None
    t_terms: list[str] = []
    t_means = torch.zeros(0, dtype=torch.float64)
    if method in _TREATMENT_MODEL:
        tx = frame.role("tx") or list(spec.predictors)
        design, t_means = _centred_design(frame, tx, weights, intercept=True)
        categories.update(design.categories)
        kept = _screen(frame, design, None, w, "TME:")
        xt, t_terms, t_means = design.x[:, kept], [design.terms[i] for i in kept], t_means[kept]
        if frame.n <= xt.shape[1] * (levels - 1):
            raise AnalysisError("insufficient_observations", "The treatment model has more "
                                "parameters than observations.")
        fit = fit_treatment(tmodel, xt, codes, levels, w)
        check_overlap(fit.probabilities, float(frame.option("pstolerance")), treatment)
        omega, omega_slope = _ipw_weights(fit, estimand, levels, target)
    else:
        omega = torch.ones((frame.n, levels), dtype=torch.float64)
        omega_slope = None

    equations: list[_Equation] = []
    if method in _OUTCOME_MODEL:
        base, o_means = _centred_design(frame, spec.predictors, weights, intercept=spec.intercept)
        categories.update(base.categories)
        if base.x.shape[1] == 0:
            raise AnalysisError("invalid_spec", "The outcome model has no terms: give covariates "
                                "or keep the intercept.")
        for level in range(levels):
            kept = _screen(frame, base, masks[level], w, f"OME{treatment.text(level)}:")
            x = base.x[:, kept]
            if treatment.counts[level] <= x.shape[1]:
                raise AnalysisError("insufficient_observations", f"Treatment level "
                                    f"{treatment.text(level)} has {treatment.counts[level]} "
                                    f"observation(s) for {x.shape[1]} outcome-model parameters.")
            sample_weights = (w * omega[:, level]) if method == "ipwra" else w
            beta = fit_index(omodel, x[masks[level]], y[masks[level]],
                             sample_weights[masks[level]],
                             what=f"outcome model of level {treatment.text(level)}")
            eta = x @ beta
            mu, dmu = mean_pieces(omodel, eta)
            r, dr = residual_pieces(omodel, y, eta)
            equations.append(_Equation([base.terms[i] for i in kept], o_means[kept], x, beta,
                                       mu, dmu, r, dr))

    selection = indicators[target] if estimand == "atet" else torch.ones(frame.n, dtype=torch.float64)
    tau = torch.empty(levels, dtype=torch.float64)
    for level in range(levels):
        if method in {"ra", "ipwra"}:
            tau[level] = (w * selection * equations[level].mu).sum() / (w * selection).sum()
        elif method == "ipw":
            mass = w * indicators[level] * omega[:, level]
            tau[level] = (mass * y).sum() / mass.sum()
        else:
            eq = equations[level]
            augmented = indicators[level] * omega[:, level] * (y - eq.mu) + selection * eq.mu
            tau[level] = (w * augmented).sum() / (w * selection).sum()

    layout = _Layout(xt, fit, equations, levels)
    jacobian = _jacobian(method, layout, y, w, indicators, omega, omega_slope, tau, selection)

    def scores(start: int, stop: int) -> Tensor:
        return _scores(method, layout, y, indicators, omega, tau, selection, start, stop)

    covariance, info = stacked_covariance(frame, jacobian, scores, weights)
    result = _result(frame, method, treatment, weights, layout, tau, covariance, info, t_terms,
                     t_means, estimand, omodel, tmodel, categories)
    if estimand == "atet":
        result.extra["target_population"] = {"treatment_level": treatment.labels[target],
                                              "definition": "E[Y(l)-Y(control) | D=tlevel]"}
    return result


class _Layout:
    """Positions of the parameter blocks in theta = (gamma, b_0..b_{L-1}, POM)."""

    def __init__(self, xt: Tensor | None, fit: TreatmentFit | None,
                 equations: list[_Equation], levels: int):
        self.xt, self.fit, self.equations, self.levels = xt, fit, equations, levels
        self.kt = 0 if xt is None else xt.shape[1]
        self.equations_t = 0 if fit is None else levels - 1
        offset = self.kt * self.equations_t
        self.beta: list[slice] = []
        for eq in equations:
            self.beta.append(slice(offset, offset + len(eq.beta)))
            offset += len(eq.beta)
        self.tau = offset
        self.size = offset + levels

    def gamma(self, s: int) -> slice:
        return slice(s * self.kt, (s + 1) * self.kt)


def _jacobian(method: str, layout: _Layout, y: Tensor, w: Tensor, indicators: list[Tensor],
              omega: Tensor, omega_slope: Tensor | None, tau: Tensor, selection: Tensor
              ) -> Tensor:
    """A = sum_i w_i d psi_i / d theta' (analytic; see the module notes)."""
    jacobian = torch.zeros((layout.size, layout.size), dtype=torch.float64)
    xt, fit = layout.xt, layout.fit

    def cross(z1: Tensor, c: Tensor, z2: Tensor) -> Tensor:
        return (z1 * (w * c)[:, None]).T @ z2

    for s in range(layout.equations_t):
        for r in range(layout.equations_t):
            jacobian[layout.gamma(s), layout.gamma(r)] = cross(xt, fit.curvature[:, s, r], xt)
    for level, eq in enumerate(layout.equations):
        rows = layout.beta[level]
        ind = indicators[level]
        weight = omega[:, level] if method == "ipwra" else 1.0
        jacobian[rows, rows] = cross(eq.x, ind * weight * eq.dr, eq.x)
        if method == "ipwra":
            for s in range(layout.equations_t):
                jacobian[rows, layout.gamma(s)] = cross(eq.x, ind * eq.r * omega_slope[:, level, s],
                                                        xt)
    for level in range(layout.levels):
        row = layout.tau + level
        ind = indicators[level]
        if method in {"ra", "ipwra"}:
            eq = layout.equations[level]
            jacobian[row, layout.beta[level]] = (w * selection * eq.dmu) @ eq.x
            jacobian[row, row] = -(w * selection).sum()
        elif method == "ipw":
            for s in range(layout.equations_t):
                jacobian[row, layout.gamma(s)] = (w * ind * (y - tau[level])
                                                  * omega_slope[:, level, s]) @ xt
            jacobian[row, row] = -(w * ind * omega[:, level]).sum()
        else:
            eq = layout.equations[level]
            jacobian[row, layout.beta[level]] = (w * eq.dmu * (selection - ind * omega[:, level])) @ eq.x
            for s in range(layout.equations_t):
                jacobian[row, layout.gamma(s)] = (w * ind * (y - eq.mu)
                                                  * omega_slope[:, level, s]) @ xt
            jacobian[row, row] = -(w * selection).sum()
    return jacobian


def _scores(method: str, layout: _Layout, y: Tensor, indicators: list[Tensor], omega: Tensor,
            tau: Tensor, selection: Tensor, start: int, stop: int) -> Tensor:
    """Unweighted estimating-function rows psi_i of observations start:stop."""
    part = slice(start, stop)
    blocks: list[Tensor] = []
    if layout.fit is not None:
        xt = layout.xt[part]
        blocks.extend(xt * layout.fit.residual[part, s, None] for s in range(layout.equations_t))
    for level, eq in enumerate(layout.equations):
        weight = omega[part, level] if method == "ipwra" else 1.0
        blocks.append(eq.x[part] * (indicators[level][part] * weight * eq.r[part])[:, None])
    columns = []
    for level in range(layout.levels):
        ind = indicators[level][part]
        if method in {"ra", "ipwra"}:
            columns.append(selection[part] * (layout.equations[level].mu[part] - tau[level]))
        elif method == "ipw":
            columns.append(ind * omega[part, level] * (y[part] - tau[level]))
        else:
            eq = layout.equations[level]
            columns.append(ind * omega[part, level] * (y[part] - eq.mu[part]) +
                           selection[part] * (eq.mu[part] - tau[level]))
    blocks.append(torch.stack(columns, dim=1))
    return torch.cat(blocks, dim=1)


def _result(frame: ModelFrame, method: str, treatment: Treatment, weights: StudyWeights,
            layout: _Layout, tau: Tensor, covariance: Tensor, info: dict[str, Any],
            t_terms: list[str], t_means: Tensor, estimand: str, omodel: str,
            tmodel: str, categories: dict[str, Any]) -> ResultBundle:
    levels = treatment.levels
    pom = covariance[layout.tau:, layout.tau:]
    if estimand == "pomeans":
        transform = torch.eye(levels, dtype=torch.float64)
        terms = [treatment.mean_term("POmeans", level) for level in range(levels)]
        equations = ["POmeans"] * levels
    else:
        name = "ATET" if estimand == "atet" else "ATE"
        transform = torch.zeros((levels, levels), dtype=torch.float64)
        for level in range(1, levels):
            transform[level - 1, level], transform[level - 1, 0] = 1.0, -1.0
        transform[levels - 1, 0] = 1.0
        terms = [treatment.effect_term(name, level) for level in range(1, levels)]
        terms.append(treatment.mean_term("POmean", 0))
        equations = [name] * (levels - 1) + ["POmean"]
    params, reported = transform @ tau, transform @ pom @ transform.T

    auxiliary: dict[str, Any] = {}
    if layout.fit is not None:
        shift = constant_shift(t_means)
        for s in range(layout.equations_t):
            block = layout.gamma(s)
            auxiliary[f"TME{treatment.text(s + 1)}"] = coefficient_table(
                t_terms, shift @ layout.fit.gamma[s], shift @ covariance[block, block] @ shift.T)
    for level, eq in enumerate(layout.equations):
        block = layout.beta[level]
        shift = constant_shift(eq.means) if eq.terms and eq.terms[0] == "Intercept" \
            else torch.eye(len(eq.beta), dtype=torch.float64)
        auxiliary[f"OME{treatment.text(level)}"] = coefficient_table(
            eq.terms, shift @ eq.beta, shift @ covariance[block, block] @ shift.T)

    counts = {treatment.text(level): treatment.counts[level] for level in range(levels)}
    extra: dict[str, Any] = {
        "method": method, "estimand": estimand, "treatment": treatment.column,
        "levels": [treatment.text(level) for level in range(levels)],
        "control": treatment.text(0), "observations_by_level": counts,
        "outcome_model": omodel if method in _OUTCOME_MODEL else None,
        "treatment_model": (("mlogit" if levels > 2 else tmodel) if layout.fit is not None
                            else None),
        "auxiliary_equations": auxiliary,
        "potential_outcome_means": {treatment.text(level): float(tau[level])
                                    for level in range(levels)},
    }
    metrics: dict[str, Any] = {"n_levels": levels}
    from openecon.econometrics.postest.causal_state import stacked_state
    blocks = []
    if layout.fit is not None:
        shift = constant_shift(t_means)
        for s in range(layout.equations_t):
            blocks.append((layout.gamma(s),f'TME{treatment.text(s+1)}',t_terms,layout.fit.gamma[s],shift))
    for level, eq in enumerate(layout.equations):
        shift = constant_shift(eq.means) if eq.terms and eq.terms[0] == 'Intercept' else torch.eye(len(eq.beta),dtype=torch.float64)
        blocks.append((layout.beta[level],f'OME{treatment.text(level)}',eq.terms,eq.beta,shift))
    extra['evaluation_state'] = stacked_state(len(covariance),covariance,tau,layout.tau,
                                              [treatment.text(i) for i in range(levels)],blocks)
    if levels == 2:
        metrics.update({"n_control": treatment.counts[0], "n_treated": treatment.counts[1]})
    if layout.fit is not None:
        extra["treatment_covariates"] = t_terms
        extra["overlap"] = {"pstolerance": float(frame.option("pstolerance")),
                            "propensity": propensity_summary(layout.fit.probabilities, treatment)}
        metrics["treatment_log_likelihood"] = layout.fit.log_likelihood
        if levels == 2:
            p = layout.fit.probabilities[:, 1]
            metrics.update({"propensity_min": float(p.min()), "propensity_max": float(p.max())})
    text = METHOD_TITLES[method]
    info["correction"] = info["correction"] + " (Stata teffects vce(robust) / vce(cluster))"
    return build_result(
        frame, terms=terms, params=params, covariance=reported, equations=equations,
        title=f"Treatment-effects estimation: {text}", use_t=False, metrics=metrics,
        solver="stacked_estimating_equations", inference=info, extra=extra,
        nobs=weights.nobs, categories=categories,
        provenance={"method": method, "estimand": estimand, "outcome_model": extra["outcome_model"],
                    "treatment_model": extra["treatment_model"],
                    "parameters_in_stacked_system": layout.size},
    )
