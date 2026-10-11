"""Survival treatment effects on mean survival time: Stata's ``stteffects ra/ipw/ipwra``.

Binary treatment A (control = code 0), failure indicator delta, observed time t.
Potential-outcome means (POM) of the survival time T(j) are estimated by

* ``ra`` (regression adjustment): a Weibull PH model fitted separately in each arm
  by maximum likelihood (censoring in the likelihood); ``POM_j`` is the sample mean
  (ATE, pomeans) or treated mean (ATET) of the model mean
  ``m_j(x) = Gamma(1 + 1/p_j) exp(-x'b_j / p_j)``.
* ``ipw``: a logit treatment model and a Weibull PH censoring model (event =
  censoring, covariates ``cx``) fitted separately in each arm; ``POM_j`` is the
  normalized weighted mean of the failure times of arm j with weights
  ``delta / (P(A = j | z) G_j(t | c))``, ``G_j`` the censoring survivor function
  (ATET: weight ``P(A=1|z) / P(A=0|z)`` instead of ``1 / P(A=0|z)`` for controls and 1
  for the treated).
* ``ipwra``: the logit treatment model and the arm-specific Weibull outcome models
  fitted with the inverse-probability weights (ATE: ``1/P(A=j|z)``; ATET: 1 and
  ``P(A=1|z)/P(A=0|z)``); POMs as in ``ra``.

All nuisance models and POMs solve one stacked system of estimating equations
``sum_i psi_i(theta) = 0`` (scores of the logit, Weibull outcome and censoring
likelihoods; POM moments). The covariance is the M-estimation sandwich
``A^-1 B A^-T`` with the analytic Jacobian ``A`` and ``B = sum psi psi'`` (or cluster
sums), without small-sample factor, as ``teffects`` (``teffects.common``). The ATE
(ATET) and the control POM are the linear transform ``(POM_1 - POM_0, POM_0)``.

The mean survival time integrates the fitted Weibull survivor function to infinity:
it extrapolates beyond follow-up and relies on the Weibull form (Stata's default
outcome and censoring models). Restricted means, other distributions, multivalued
treatments and weights are not supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import Design, ModelFrame, build_result, column_list
from openecon.econometrics.discrete.common import build_spec, constant_shift, maximize
from openecon.econometrics.survival.cox import _survival_columns
from openecon.econometrics.survival.data import FLOAT, check_times, end_before_entry_note
from openecon.econometrics.teffects.common import (
    check_overlap, coefficient_table, propensity_summary, stacked_covariance, study_weights,
    treatment_coding,
)
from openecon.econometrics.teffects.index import fit_treatment
from openecon.models import ModelSpec, ResultBundle

_TITLES = {"ra": "regression adjustment (Weibull)",
           "ipw": "inverse-probability weighting (logit treatment, Weibull censoring)",
           "ipwra": "IPW regression adjustment (logit treatment, Weibull outcome)"}


class WeibullArm:
    """Weighted Weibull PH log likelihood ``sum w [d (a + mu + (p-1) ln t) - exp(mu) t^p]``."""

    def __init__(self, x: Tensor, log_t: Tensor, event: Tensor, weights: Tensor):
        self.x, self.log_t, self.event, self.w = x, log_t, event, weights
        self.k = x.shape[1]

    def pieces(self, theta: Tensor):
        mu = self.x @ theta[:self.k]
        p = torch.exp(theta[self.k])
        plt = p * self.log_t
        big_h = torch.exp(mu + plt)
        return mu, p, plt, big_h

    def value(self, theta: Tensor) -> Tensor:
        mu, p, plt, big_h = self.pieces(theta)
        return (self.w * (self.event * (theta[self.k] + mu + plt - self.log_t) - big_h)).sum()

    def scores(self, theta: Tensor) -> Tensor:
        """Per-observation scores (unweighted) [n, k + 1]."""
        _, _, plt, big_h = self.pieces(theta)
        s_mu = self.event - big_h
        s_a = self.event * (1 + plt) - big_h * plt
        return torch.cat([self.x * s_mu[:, None], s_a[:, None]], dim=1)

    def hessian_rows(self, theta: Tensor, weights: Tensor) -> Tensor:
        """``sum_i weights_i d s_i / d theta'`` [k + 1, k + 1]."""
        _, _, plt, big_h = self.pieces(theta)
        k = self.k
        out = torch.zeros((k + 1, k + 1), dtype=FLOAT)
        out[:k, :k] = -(self.x * (weights * big_h)[:, None]).T @ self.x
        cross = -self.x.T @ (weights * big_h * plt)
        out[:k, k] = out[k, :k] = cross
        out[k, k] = (weights * (self.event * plt - big_h * (plt + plt * plt))).sum()
        return out

    def __call__(self, theta: Tensor):
        value = self.value(theta)
        gradient = (self.scores(theta) * self.w[:, None]).sum(0)
        return value, gradient, self.hessian_rows(theta, self.w)


def _weibull_fit(x: Tensor, log_t: Tensor, event: Tensor, weights: Tensor, what: str) -> Tensor:
    if float((weights * event).sum()) <= 0:
        raise AnalysisError("no_failures", f"{what}: no events, so the Weibull model is not "
                            "identified.")
    objective = WeibullArm(x, log_t, event, weights)
    start = torch.zeros(x.shape[1] + 1, dtype=FLOAT)
    rate = float((weights * event).sum() / (weights * torch.exp(log_t)).sum())
    start[0] = torch.log(torch.tensor(rate, dtype=FLOAT))
    result = maximize(objective, start, what=what)
    if not result.converged:
        raise AnalysisError("nonconvergence", f"{what} did not converge "
                            f"({result.diagnostics.get('message')}); check for covariates that "
                            "separate events from censoring within the arm.")
    return result.theta


@dataclass
class _Block:
    name: str
    terms: list[str]
    means: Tensor
    rows: slice


def _design(frame: ModelFrame, names: Sequence[str], prefix: str) -> tuple[Design, Tensor]:
    design = frame.drop_collinear(frame.design(list(names), intercept=True, prefix=prefix))
    means = torch.zeros(design.x.shape[1], dtype=FLOAT)
    if design.x.shape[1] > 1:
        means[1:] = design.x[:, 1:].mean(0)
        design.x = design.x - means
    return design, means


def _mean_survival(x: Tensor, theta: Tensor) -> tuple[Tensor, Tensor]:
    """Weibull PH mean ``Gamma(1 + 1/p) exp(-x'b/p)`` and its derivative rows [n, k + 1]."""
    k = x.shape[1]
    mu = x @ theta[:k]
    p = torch.exp(theta[k])
    inv = 1.0 / p
    m = torch.exp(torch.lgamma(1 + inv) - mu * inv)
    d_beta = -(m * inv)[:, None] * x
    d_a = m * (mu - torch.digamma(1 + inv)) * inv
    return m, torch.cat([d_beta, d_a[:, None]], dim=1)


def fit_stteffects(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``stteffects``: ``(spec, data) -> ResultBundle`` (module notes)."""
    command = "stteffects"
    frame = ModelFrame(spec, data)
    method, estimand = frame.option("method"), frame.option("estimand")
    treatment_name = frame.role("treatment")[0]
    if treatment_name in spec.predictors or treatment_name in frame.role("tx") \
            or treatment_name in frame.role("cx"):
        raise AnalysisError("invalid_spec", f"The treatment '{treatment_name}' must not also be "
                            "a covariate.")
    if frame.role("cx") and method != "ipw":
        raise AnalysisError("invalid_spec", "'cx' (censoring-model covariates) applies to "
                            "method='ipw' only.")
    if frame.role("tx") and method == "ra":
        raise AnalysisError("invalid_spec", "'tx' (treatment-model covariates) does not apply "
                            "to method='ra'.")
    t, _, delta = _survival_columns(frame)
    usable = check_times(t, None, spec.outcome, None)
    if not bool(usable.all()):
        frame.restrict(usable, end_before_entry_note(int((~usable).sum())))
        t, _, delta = _survival_columns(frame)
    treatment = treatment_coding(frame, treatment_name, frame.option("control"))
    if treatment.levels != 2:
        raise AnalysisError("invalid_treatment", f"{command} supports a binary treatment; "
                            f"'{treatment_name}' has {treatment.levels} levels.")
    weights = study_weights(frame)
    n = frame.n
    a = treatment.codes
    arm = [(a == 0).to(FLOAT), (a == 1).to(FLOAT)]
    log_t = torch.log(t)
    ones = torch.ones(n, dtype=FLOAT)
    target = ones if estimand != "atet" else arm[1]
    blocks: list[_Block] = []
    size = 2
    # ---- treatment model -------------------------------------------------------------
    treat_fit = None
    if method in {"ipw", "ipwra"}:
        tz, t_means = _design(frame, frame.role("tx") or spec.predictors, "")
        treat_fit = fit_treatment("logit", tz.x, a, 2, weights.user)
        check_overlap(treat_fit.probabilities, float(frame.option("pstolerance")), treatment)
        gamma = treat_fit.gamma[0]
        p1 = treat_fit.probabilities[:, 1]
        blocks.append(_Block(f"TME{treatment.text(1)}", tz.terms, t_means,
                             slice(size, size + len(tz.terms))))
        size += len(tz.terms)
        if estimand == "atet":
            omega = [p1 / (1 - p1), ones]
            d_omega = [(p1 / (1 - p1))[:, None] * tz.x, torch.zeros_like(tz.x)]
        else:
            omega = [1 / (1 - p1), 1 / p1]
            d_omega = [(p1 / (1 - p1))[:, None] * tz.x, (-(1 - p1) / p1)[:, None] * tz.x]
    # ---- outcome or censoring models per arm ------------------------------------------
    equations: list[tuple[Tensor, Tensor, WeibullArm]] = []
    if method in {"ra", "ipwra"}:
        ox, o_means = _design(frame, spec.predictors, "")
        for j in (0, 1):
            w = arm[j] if method == "ra" else arm[j] * omega[j]
            theta_j = _weibull_fit(ox.x, log_t, delta, w, f"{command} outcome model "
                                   f"({treatment.text(j)})")
            equations.append((theta_j, w, WeibullArm(ox.x, log_t, delta, w)))
            blocks.append(_Block(f"OME{treatment.text(j)}", ox.terms + ["/ln_p"],
                                 torch.cat([o_means, torch.zeros(1, dtype=FLOAT)]),
                                 slice(size, size + len(ox.terms) + 1)))
            size += len(ox.terms) + 1
    else:
        cx, c_means = _design(frame, frame.role("cx") or spec.predictors, "")
        censored = 1 - delta
        for j in (0, 1):
            if float((arm[j] * censored).sum()) == 0:
                raise AnalysisError("insufficient_censoring", f"{command}: arm "
                                    f"{treatment.text(j)} has no censored observations, so the "
                                    "Weibull censoring model is not identified.")
            theta_j = _weibull_fit(cx.x, log_t, censored, arm[j], f"{command} censoring model "
                                   f"({treatment.text(j)})")
            equations.append((theta_j, arm[j], WeibullArm(cx.x, log_t, censored, arm[j])))
            blocks.append(_Block(f"CME{treatment.text(j)}", cx.terms + ["/ln_p"],
                                 torch.cat([c_means, torch.zeros(1, dtype=FLOAT)]),
                                 slice(size, size + len(cx.terms) + 1)))
            size += len(cx.terms) + 1
    frame.workspace_plan("stacked survival treatment-effect equations",
                         {"estimating_function_rows": 8 * 2 * n * size})
    # ---- potential-outcome means, moments and Jacobian --------------------------------
    psi = torch.zeros((n, size), dtype=FLOAT)
    jacobian = torch.zeros((size, size), dtype=FLOAT)
    pom = torch.zeros(2, dtype=FLOAT)
    outcome_blocks = [b for b in blocks if not b.name.startswith("TME")]
    if treat_fit is not None:
        tb = blocks[0].rows
        tzx = tz.x
        psi[:, tb] = (arm[1] - p1)[:, None] * tzx
        jacobian[tb, tb] = -(tzx * (p1 * (1 - p1))[:, None]).T @ tzx
    for j, (theta_j, w, model) in enumerate(equations):
        rows = outcome_blocks[j].rows
        scores = model.scores(theta_j)
        psi[:, rows] = scores * w[:, None]
        jacobian[rows, rows] = model.hessian_rows(theta_j, w)
        if method == "ipwra":
            jacobian[rows, blocks[0].rows] = (scores * arm[j][:, None]).T @ d_omega[j]
        if method in {"ra", "ipwra"}:
            m, dm = _mean_survival(ox.x, theta_j)
            pom[j] = (target * m).sum() / target.sum()
            psi[:, j] = target * (m - pom[j])
            jacobian[j, j] = -target.sum()
            jacobian[j, rows] = (target[:, None] * dm).sum(0)
        else:
            _, _, plt, big_c = model.pieces(theta_j)
            base = arm[j] * delta * torch.exp(big_c)
            weight = base * omega[j]
            pom[j] = (weight * t).sum() / weight.sum()
            residual = t - pom[j]
            psi[:, j] = weight * residual
            jacobian[j, j] = -weight.sum()
            # d exp(H_c)/d(kappa, ln p_c) = exp(H_c) H_c (c, p_c ln t)
            design_c = torch.cat([cx.x, plt[:, None]], dim=1)
            jacobian[j, rows] = ((weight * residual * big_c)[:, None] * design_c).sum(0)
            jacobian[j, blocks[0].rows] = ((base * residual)[:, None] * d_omega[j]).sum(0)
    covariance, info = stacked_covariance(frame, jacobian, lambda lo, hi: psi[lo:hi], weights)
    # ---- reporting ----------------------------------------------------------------
    pom_cov = covariance[:2, :2]
    if estimand == "pomeans":
        transform = torch.eye(2, dtype=FLOAT)
        terms = [treatment.mean_term("POmeans", level) for level in (0, 1)]
        equation_names = ["POmeans", "POmeans"]
    else:
        name = "ATET" if estimand == "atet" else "ATE"
        transform = torch.tensor([[-1.0, 1.0], [1.0, 0.0]], dtype=FLOAT)
        terms = [treatment.effect_term(name, 1), treatment.mean_term("POmean", 0)]
        equation_names = [name, "POmean"]
    params, reported = transform @ pom, transform @ pom_cov @ transform.T
    auxiliary: dict[str, Any] = {}
    stacked = torch.zeros(size, dtype=FLOAT)
    stacked[:2] = pom
    if treat_fit is not None:
        stacked[blocks[0].rows] = gamma
    for j, (theta_j, _, _) in enumerate(equations):
        stacked[outcome_blocks[j].rows] = theta_j
    for block in blocks:
        width = block.rows.stop - block.rows.start
        shift = torch.eye(width, dtype=FLOAT)
        main = width - (0 if block.name.startswith("TME") else 1)
        shift[:main, :main] = constant_shift(block.means[:main])
        estimates = shift @ stacked[block.rows]
        auxiliary[block.name] = coefficient_table(
            block.terms, estimates, shift @ covariance[block.rows, block.rows] @ shift.T)
    counts = {treatment.text(level): treatment.counts[level] for level in (0, 1)}
    censored_counts = {treatment.text(j): float((arm[j] * (1 - delta)).sum()) for j in (0, 1)}
    extra: dict[str, Any] = {
        "method": method, "estimand": estimand, "treatment": treatment.column,
        "levels": [treatment.text(level) for level in (0, 1)], "control": treatment.text(0),
        "observations_by_level": counts, "censored_by_level": censored_counts,
        "outcome_model": "weibull" if method != "ipw" else None,
        "censoring_model": "weibull" if method == "ipw" else None,
        "treatment_model": "logit" if treat_fit is not None else None,
        "target": "mean survival time (Weibull extrapolation to infinity)",
        "potential_outcome_means": {treatment.text(level): float(pom[level]) for level in (0, 1)},
        "potential_outcome_covariance": pom_cov.tolist(),
        "auxiliary_equations": auxiliary,
    }
    metrics: dict[str, Any] = {"n_control": treatment.counts[0], "n_treated": treatment.counts[1],
                               "n_failures": float(delta.sum()),
                               "parameters_in_stacked_system": size}
    if treat_fit is not None:
        extra["overlap"] = {"pstolerance": float(frame.option("pstolerance")),
                            "propensity": propensity_summary(treat_fit.probabilities, treatment)}
        metrics.update({"propensity_min": float(p1.min()), "propensity_max": float(p1.max()),
                        "treatment_log_likelihood": treat_fit.log_likelihood})
    info["correction"] = info["correction"] + " (Stata stteffects vce(robust) / vce(cluster))"
    return build_result(
        frame, terms=terms, params=params, covariance=reported, equations=equation_names,
        title=f"Survival treatment effects: {_TITLES[method]}", use_t=False, metrics=metrics,
        solver="stacked_estimating_equations", inference=info, extra=extra, nobs=weights.nobs,
        provenance={"method": method, "estimand": estimand,
                    "parameters_in_stacked_system": size,
                    "jacobian": "analytic", "target": extra["target"]})


def stteffects(*, data: Any, time: str, failure: str, treatment: str,
               x: Sequence[str] | None = None, method: str = "ra", estimand: str = "ate",
               tx: Sequence[str] | None = None, cx: Sequence[str] | None = None,
               control: Any = None, pstolerance: float | None = None,
               covariance: str | None = None, cluster: str | None = None,
               missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Treatment effects on mean survival time: RA, IPW or IPWRA (Stata ``stteffects``).

    Model
        Binary treatment; potential mean survival times from arm-specific Weibull PH
        outcome models (``ra``; ``ipwra`` with inverse-probability weights from a logit
        treatment model) or from censoring-adjusted inverse-probability weighting with a
        logit treatment model and arm-specific Weibull censoring models (``ipw``).

    Estimator
        One stacked system of estimating equations (nuisance scores and potential-outcome
        mean moments) with the analytic Jacobian; robust (or cluster) M-estimation
        sandwich without small-sample factor.

    Parameters
        data: DataFrame. time: observed time (> 0). failure: 0/1 event indicator.
        treatment: binary treatment column; control: its control level (default lowest).
        x: outcome-model covariates (ra, ipwra); also the default of tx and cx.
        method: ``'ra'`` (default), ``'ipw'`` or ``'ipwra'``.
        estimand: ``'ate'`` (default), ``'atet'`` or ``'pomeans'``.
        tx: treatment-model covariates (ipw, ipwra). cx: censoring-model covariates (ipw).
        pstolerance: smallest admissible estimated treatment probability (default 1e-5).
        covariance: ``'robust'`` (default) or ``'cluster'`` with ``cluster``.
        missing, alpha: as elsewhere.

    Result
        ``ATE:r1vs0.<treat>`` (or ``ATET``) and ``POmean:0.<treat>``, or ``POmeans`` for
        both levels, with z tests. ``extra``: potential_outcome_means, auxiliary_equations
        (TME1, OME0/OME1 or CME0/CME1 with standard errors), overlap diagnostics.

    Stata
        ``stteffects ra (x1 x2) (treat)``, ``stteffects ipw (treat x1 x2) (x1 x2)``,
        ``stteffects ipwra (x1 x2) (treat x1 x2), atet``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(3); n = 500
        >>> x = rng.normal(size=n); a = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
        >>> t = rng.weibull(1.3, n) * np.exp(0.3 * a - 0.2 * x); c = rng.exponential(3, n)
        >>> df = pd.DataFrame({"t": np.minimum(t, c), "d": (t <= c) * 1, "a": a, "x": x})
        >>> print(oe.stteffects(data=df, time="t", failure="d", treatment="a", x=["x"]).summary())
    """
    spec = build_spec(
        "stteffects", outcome=time, predictors=column_list(x, "x"), covariance=covariance,
        cluster=cluster, missing=missing, alpha=alpha,
        columns={"failure": failure, "treatment": treatment, "tx": column_list(tx, "tx"),
                 "cx": column_list(cx, "cx")},
        options={"method": method, "estimand": estimand, "control": control,
                 "pstolerance": pstolerance})
    return fit(spec, data=data)
