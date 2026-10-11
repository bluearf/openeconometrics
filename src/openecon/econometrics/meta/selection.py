"""Vevea-Hedges (1995) step-function selection model for independent effects.

Each study's one-sided p-value ``p_i = 1 - Phi(y_i/sqrt(v_i))`` (alternative
"greater") falls in an interval ``(a_{j-1}, a_j]`` of declared cutpoints; the
observed density is the normal random-effects density times the interval
weight ``omega_j`` divided by ``A_i = sum_j omega_j B_ij`` (Vevea and Hedges
1995, Psychometrika 60:419-435; Hedges 1992, Statistical Science 7:246-255).
``omega_1 = 1``; other weights are estimated by maximum likelihood or fixed
a priori (Vevea and Woods 2005, Psychological Methods 10:428-443). The
likelihood, interval convention, natural-scale Hessian, Wald tests of
``omega_j = 1`` and the likelihood-ratio test against the unadjusted ML model
follow https://wviechtb.github.io/metafor/reference/selmodel.html and the
weightr reference implementation https://github.com/cran/weightr.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_sf, normal_isf, normal_sf
from openecon.engines.optimize import information_inverse, maximize_bfgs, numerical_hessian

from .common import critical, level_check, sample
from .models import meta_regress

SOURCE = "https://wviechtb.github.io/metafor/reference/selmodel.html"
WEIGHTR_SOURCE = "https://github.com/cran/weightr"
LOG_2PI = math.log(2*math.pi)
WEIGHT_RANGE = (1e-6, 1e6)


def _interval_mass(t_upper, t_lower):
    """Phi(t_upper) - Phi(t_lower), computed in the tail that avoids cancellation."""
    lower_tail = torch.special.ndtr(t_upper)-torch.special.ndtr(t_lower)
    upper_tail = torch.special.ndtr(-t_lower)-torch.special.ndtr(-t_upper)
    return torch.where(t_lower > 0, upper_tail, lower_tail)


class StepLikelihood:
    """Analytic log likelihood and gradient in natural parameters (beta, tau2, omega)."""

    def __init__(self, y, v, x, cuts, groups, intervals):
        self.y, self.v, self.x = y, v, x
        self.k, self.p = x.shape
        self.intervals = intervals
        se = v.sqrt()
        # Effect-scale cutpoints c_ij, j = 0..J: +inf, interior, -inf.
        interior = [se*normal_isf(a) for a in cuts[:-1]]
        self.c = torch.stack([torch.full_like(y, math.inf), *interior, torch.full_like(y, -math.inf)], 1)
        self.groups = groups
        self.counts = torch.bincount(groups, minlength=intervals).to(torch.float64)

    def value_gradient(self, beta, tau2, omega):
        mu = self.x@beta
        eta = self.v+tau2
        s = eta.sqrt()
        t = (self.c-mu[:, None])/s[:, None]
        mass = _interval_mass(t[:, :-1], t[:, 1:]).clamp_min(0)
        area = mass@omega
        if bool((area <= 0).any()) or not bool(torch.isfinite(area).all()):
            raise AnalysisError("numerical_failure", "A selection normalizing constant underflowed; rescale effects.")
        finite = torch.isfinite(t)
        pdf = torch.where(finite, torch.exp(-.5*torch.where(finite, t, 0)**2)/math.sqrt(2*math.pi), 0)
        pdf_t = torch.where(finite, pdf*torch.where(finite, t, 0), 0)
        d_mass_mu = -(pdf[:, :-1]-pdf[:, 1:])/s[:, None]
        d_mass_eta = -(pdf_t[:, :-1]-pdf_t[:, 1:])/(2*eta[:, None])
        residual = self.y-mu
        value = (omega.log()[self.groups].sum()-.5*(self.k*LOG_2PI+eta.log().sum()+(residual.square()/eta).sum())
                 - area.log().sum())
        d_mu = residual/eta-(d_mass_mu@omega)/area
        d_eta = -.5/eta+.5*residual.square()/eta.square()-(d_mass_eta@omega)/area
        d_omega = self.counts/omega-(mass/area[:, None]).sum(0)
        return float(value), self.x.T@d_mu, float(d_eta.sum()), d_omega


def _fit(lik, start_beta, start_tau2, free_weights, fixed_weights, tau2_free):
    """Maximize on (beta, log tau2, log omega_free); natural-scale Hessian at the optimum."""
    p, intervals = lik.p, lik.intervals

    def unpack(theta):
        beta = theta[:p]
        tau2 = float(theta[p].exp()) if tau2_free else 0.0
        omega = fixed_weights.clone()
        if free_weights:
            omega[1:] = theta[p+int(tau2_free):].exp()
        return beta, tau2, omega

    def objective(theta):
        beta, tau2, omega = unpack(theta)
        value, d_beta, d_tau2, d_omega = lik.value_gradient(beta, tau2, omega)
        parts = [d_beta]
        if tau2_free:
            parts.append(torch.tensor([d_tau2*tau2], dtype=torch.float64))
        if free_weights:
            parts.append(d_omega[1:]*omega[1:])
        return torch.tensor(value, dtype=torch.float64), torch.cat(parts)

    start = [start_beta]
    if tau2_free:
        start.append(torch.tensor([math.log(start_tau2)], dtype=torch.float64))
    if free_weights:
        start.append(torch.zeros(intervals-1, dtype=torch.float64))
    try:
        result = maximize_bfgs(objective, torch.cat(start), max_iter=2000, raise_on_failure=False)
    except (KernelError, AnalysisError):
        return None
    if not result.converged:
        return None
    theta, value = result.theta, float(result.value)
    # Newton polish with the numerical Hessian of the analytic gradient.
    for _ in range(10):
        try:
            hessian = numerical_hessian(lambda point: objective(point)[1], theta)
            gradient = objective(theta)[1]
            step = torch.linalg.solve(-hessian, gradient)
            trial_value, _ = objective(theta+step)
        except (KernelError, AnalysisError, RuntimeError):
            break
        if not math.isfinite(float(trial_value)) or float(trial_value) < value-1e-12*max(1.0, abs(value)):
            break
        theta, value = theta+step, max(value, float(trial_value))
        if float((step.abs()/theta.abs().clamp_min(1.0)).max()) <= 1e-12:
            break
    beta, tau2, omega = unpack(theta)
    return {"beta": beta, "tau2": tau2, "omega": omega, "loglik": value,
            "iterations": result.iterations, "gradient": objective(theta)[1]}


def _natural_covariance(lik, fit, free_weights, tau2_free):
    """Inverse observed information over free natural parameters (beta, tau2, omega_2..J)."""
    p = lik.p

    def gradient(theta):
        beta = theta[:p]
        tau2 = float(theta[p]) if tau2_free else 0.0
        omega = fit["omega"].clone()
        if free_weights:
            omega[1:] = theta[p+int(tau2_free):]
        _, d_beta, d_tau2, d_omega = lik.value_gradient(beta, tau2, omega)
        parts = [d_beta]
        if tau2_free:
            parts.append(torch.tensor([d_tau2], dtype=torch.float64))
        if free_weights:
            parts.append(d_omega[1:])
        return torch.cat(parts)

    point = [fit["beta"]]
    if tau2_free:
        point.append(torch.tensor([fit["tau2"]], dtype=torch.float64))
    if free_weights:
        point.append(fit["omega"][1:])
    try:
        hessian = numerical_hessian(gradient, torch.cat(point))
        return information_inverse(-hessian)
    except KernelError as exc:
        raise AnalysisError("singular_information", "The selection-model information matrix is not positive definite; parameters are not identified.") from exc


def _labels_for(cuts):
    lower = [0.0, *cuts[:-1]]
    return [f"{a:g} < p <= {b:g}" for a, b in zip(lower, cuts)], lower


@torch.no_grad()
def meta_selection(*, data, yi: str = "yi", vi: str = "vi", study: str | None = None,
                   moderators: list[str] | None = None, steps: list[float] | None = None,
                   weights: list[float] | None = None, heterogeneity: bool = True,
                   alternative: str = "greater", level: float = .95,
                   dependence: str = "independent") -> TableSet:
    """Fit the Vevea-Hedges step-function selection model by maximum likelihood.

    One-sided p-value cutpoints (steps, 1 appended) define intervals; weights
    are estimated (omega_1=1) or fixed (Vevea-Woods sensitivity). Reports
    adjusted coefficients, tau², weights with natural-scale observed-information
    covariance, and the LRT against the unadjusted ML model. Not a bias proof.
    """
    moderators = [] if moderators is None else moderators
    if not isinstance(moderators, list) or len(moderators) > 10:
        raise AnalysisError("invalid_spec", "moderators must be an explicit list of <=10 numeric columns.")
    if alternative not in ("greater", "less"):
        raise AnalysisError("invalid_spec", "alternative must be 'greater' or 'less' (one-sided p-values).")
    if type(heterogeneity) is not bool:
        raise AnalysisError("invalid_spec", "heterogeneity must be a boolean.")
    steps = [.025, 1.0] if steps is None else steps
    if not isinstance(steps, (list, tuple)) or not steps or any(isinstance(a, bool) or not isinstance(a, (int, float)) for a in steps):
        raise AnalysisError("invalid_spec", "steps must be a list of p-value cutpoints in (0, 1].")
    cuts = [float(a) for a in steps]
    if cuts[-1] != 1.0:
        cuts.append(1.0)
    if any(not 0 < a <= 1 for a in cuts) or any(b <= a for a, b in zip(cuts, cuts[1:])) or len(cuts) < 2 or len(cuts) > 10:
        raise AnalysisError("invalid_spec", "steps must be 1..9 strictly increasing cutpoints in (0,1); 1 closes the last interval.")
    intervals = len(cuts)
    if weights is not None:
        if not isinstance(weights, (list, tuple)) or len(weights) != intervals or any(
                isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights):
            raise AnalysisError("invalid_spec", "Fixed weights need one finite positive value per p-value interval.")
    level = level_check(level)
    frame, values, ids, digest = sample(data, [yi, vi, *moderators], study, dependence=dependence, minimum=3, maximum=2000)
    y, v = values[yi], values[vi]
    if bool(((v < 1e-12) | (v > 1e12)).any()) or float(v.max()/v.min()) > 1e12 or bool((y.abs() > 1e8).any()):
        raise AnalysisError("invalid_variance", "Sampling variances must lie in [1e-12,1e12] with span <=1e12 and |yi|<=1e8.")
    k = len(ids)
    terms = ["Intercept", *moderators]
    p = len(terms)
    x = torch.stack([torch.ones(k, dtype=torch.float64), *[values[name] for name in moderators]], 1)
    free_weights = weights is None
    n_free = p+int(heterogeneity)+(intervals-1 if free_weights else 0)
    if k <= n_free:
        raise AnalysisError("insufficient_studies", "The selection model needs more studies than free parameters.")
    sign = -1.0 if alternative == "less" else 1.0
    pvalues = torch.special.ndtr(-sign*y/v.sqrt())
    cut_tensor = torch.tensor(cuts, dtype=torch.float64)
    groups = torch.searchsorted(cut_tensor, pvalues, right=False)
    counts = torch.bincount(groups, minlength=intervals)
    if free_weights and bool((counts == 0).any()):
        raise AnalysisError("empty_interval", "Every p-value interval must contain at least one study when its weight is estimated.")
    lik = StepLikelihood(sign*y, v, sign*x, cuts, groups, intervals)
    unadjusted = meta_regress(data=pd.DataFrame({"__y__": y.tolist(), "__v__": v.tolist(),
                                                **{f"__m{i}__": values[name].tolist() for i, name in enumerate(moderators)}}),
                              yi="__y__", vi="__v__", moderators=[f"__m{i}__" for i in range(len(moderators))],
                              method="ML" if heterogeneity else "common", level=level)
    start_beta = torch.tensor(unadjusted["coefficients"].estimate.tolist(), dtype=torch.float64)
    start_tau2 = float(unadjusted["heterogeneity"].iloc[0].tau2)
    fixed = torch.ones(intervals, dtype=torch.float64) if free_weights else torch.tensor(weights, dtype=torch.float64)
    candidates = []
    zero = _fit(lik, start_beta, 0.0, free_weights, fixed, False)
    if zero is not None:
        candidates.append(("boundary", zero))
    if heterogeneity:
        typical = float(v.median())
        for start in (start_tau2+1e-3*typical, max(start_tau2, typical)):
            interior = _fit(lik, start_beta, start, free_weights, fixed, True)
            if interior is not None and interior["tau2"] > 1e-10*typical:
                candidates.append(("interior", interior))
    if not candidates:
        raise AnalysisError("selection_not_converged", "The selection-model likelihood did not converge; weights may be unidentified.")
    kind, best = max(candidates, key=lambda item: item[1]["loglik"])
    if heterogeneity and kind == "boundary":
        _, _, d_tau2, _ = lik.value_gradient(best["beta"], 0.0, best["omega"])
        if d_tau2*float(v.median())/k > 1e-7:
            raise AnalysisError("selection_not_converged", "The zero-heterogeneity boundary fails the score (KKT) check.")
    omega = best["omega"]
    if free_weights and bool(((omega[1:] < WEIGHT_RANGE[0]) | (omega[1:] > WEIGHT_RANGE[1])).any()):
        raise AnalysisError("selection_not_identified", "An estimated selection weight runs to its boundary; the step function is not identified by these data.")
    tau2_free = heterogeneity and kind == "interior"
    covariance = _natural_covariance(lik, best, free_weights, tau2_free)
    names = [*terms, *(["tau2"] if tau2_free else []), *([f"omega{j+1}" for j in range(1, intervals)] if free_weights else [])]
    beta = best["beta"]
    beta_cov = covariance[:p, :p]
    se = beta_cov.diagonal().sqrt()
    c = critical(level, "z", None)
    coefficients = table({"term": terms, "estimate": beta.tolist(), "std_error": se.tolist(),
                          "statistic": (beta/se).tolist(),
                          "p_value": [2*normal_sf(abs(float(z))) for z in beta/se],
                          "ci_low": (beta-c*se).tolist(), "ci_high": (beta+c*se).tolist()})
    labels, lower = _labels_for(cuts)
    weight_rows = []
    for j in range(intervals):
        row = {"interval": labels[j], "p_lower": lower[j], "p_upper": cuts[j], "k": int(counts[j]),
               "estimate": float(omega[j]), "fixed": (j == 0) or not free_weights,
               "std_error": None, "statistic": None, "p_value": None, "ci_low": None, "ci_high": None}
        if free_weights and j > 0:
            position = p+int(tau2_free)+j-1
            standard = float(covariance[position, position].sqrt())
            statistic = (float(omega[j])-1)/standard
            row.update(std_error=standard, statistic=statistic, p_value=2*normal_sf(abs(statistic)),
                       ci_low=max(0.0, float(omega[j])-c*standard), ci_high=float(omega[j])+c*standard)
        weight_rows.append(row)
    tau2_se = float(covariance[p, p].sqrt()) if tau2_free else None
    loglik0 = float(unadjusted["heterogeneity"].iloc[0].loglik_ml)
    lrt = max(0.0, 2*(best["loglik"]-loglik0))
    lrt_df = intervals-1 if free_weights else 0
    zero_loglik = zero["loglik"] if zero is not None else None
    heterogeneity_lrt = max(0.0, 2*(best["loglik"]-zero_loglik)) if heterogeneity and zero_loglik is not None else None
    tests = [{"test": "LRT selection weights (vs unadjusted ML model)", "statistic": lrt if lrt_df else None,
              "df": lrt_df if lrt_df else None, "p_value": chi2_sf(lrt, lrt_df) if lrt_df and lrt > 0 else (1.0 if lrt_df else None)},
             {"test": "LRT tau2 = 0 within selection model", "statistic": heterogeneity_lrt,
              "df": 1 if heterogeneity_lrt is not None else None,
              "p_value": (chi2_sf(heterogeneity_lrt, 1) if heterogeneity_lrt > 0 else 1.0) if heterogeneity_lrt is not None else None}]
    unadjusted_rows = unadjusted["coefficients"].copy()
    unadjusted_rows["term"] = terms
    frame_tests = table(tests)
    frame_weights = table(weight_rows)
    return TableSet({"coefficients": coefficients,
                     "covariance": table(beta_cov.tolist(), columns=terms, index=terms),
                     "parameter_covariance": table(covariance.tolist(), columns=names, index=names),
                     "selection_weights": frame_weights.astype(object).where(frame_weights.notna(), None),
                     "heterogeneity": table([{"tau2": best["tau2"], "std_error": tau2_se, "boundary": heterogeneity and not tau2_free,
                                              "estimated": heterogeneity}]).astype(object),
                     "tests": frame_tests.astype(object).where(frame_tests.notna(), None),
                     "unadjusted": unadjusted_rows,
                     "fit": table([{"loglik": best["loglik"], "loglik_unadjusted": loglik0, "k": k,
                                    "n_free_parameters": n_free-(int(heterogeneity) if heterogeneity and not tau2_free else 0),
                                    "unadjusted_tau2": start_tau2}])},
                    title="Step-function selection model", procedure="meta_selection", sample_hash=digest,
                    level=level, steps=cuts, alternative=alternative, weights_fixed=not free_weights,
                    heterogeneity=heterogeneity, n_studies=k, source=SOURCE, reference_implementation=WEIGHTR_SOURCE,
                    optimizer={"kind": kind, "candidates": [{"kind": name, "loglik": item["loglik"], "tau2": item["tau2"]} for name, item in candidates]},
                    notes=["Selection weights are relative to the first interval (omega_1 = 1); Wald tests compare each weight with 1.",
                           "Inference conditions on the declared step function; the model is a sensitivity analysis, not proof of publication bias.",
                           "Natural-scale observed information; a zero tau2 boundary is fixed and excluded from the covariance."])
