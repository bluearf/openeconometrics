"""Kline--Santos (2012) score bootstrap for Wald and LM tests after binary logit/probit ML.

Kline and Santos, "A Score Based Approach to Wild Bootstrap Inference",
Journal of Econometric Methods 1 (2012), equations (12)-(14) and Examples
4.1-4.2: with score contributions s_c of independent units c (observations,
or clusters with their contributions summed) evaluated at an estimate theta,
A = R J(theta)^-1 (J = minus the Hessian of the log likelihood) and Sigma the
recentred sample covariance of the s_c,

    T = (A Sigma A')^-1/2 A sum_c s_c,     T* = (A Sigma* A')^-1/2 A sum_c s_c W_c,

where Sigma* is the recentred sample covariance of the perturbed s_c W_c and
W_c are independent wild multipliers. No model is re-estimated in the draws.
The LM route evaluates everything at the null-restricted MLE (T is the robust
score statistic); the Wald route evaluates the scores at the unrestricted MLE
and compares the robust Wald statistic of R theta^ - r with the same
perturbed-score distribution. Sample covariances use the G/(G-1) factor.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, column_list, make_spec, table
from openecon.econometrics.postest import wild_multipliers as wm
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines.distributions import chi2_sf

SOURCE = "https://eml.berkeley.edu/~pkline/papers/ScoreFinal_web.pdf"
WORKING_PAPER = "https://doi.org/10.3386/w16127"
DEFAULT_MAX_WORK = 20_000_000_000
_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
_BLOCK_BYTES = 64 * 1024**2


def link_terms(model: str, eta: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Log likelihood contributions, d ll / d eta and -d2 ll / d eta2 for logit or probit."""
    if model == "logit":
        p = torch.sigmoid(eta)
        loglik = y * eta - torch.nn.functional.softplus(eta)
        # sigmoid(eta) sigmoid(-eta) avoids the cancellation of p (1 - p) in the tails.
        return loglik, y - p, p * torch.sigmoid(-eta)
    log_phi = -0.5 * eta.square() - _LOG_SQRT_2PI
    log_upper = torch.special.log_ndtr(eta)          # log Phi(eta)
    log_lower = torch.special.log_ndtr(-eta)         # log (1 - Phi(eta))
    lam1 = torch.exp(log_phi - log_upper)            # phi / Phi
    lam0 = torch.exp(log_phi - log_lower)            # phi / (1 - Phi)
    loglik = y * log_upper + (1 - y) * log_lower
    gradient = y * lam1 - (1 - y) * lam0
    weight = y * lam1 * (lam1 + eta) + (1 - y) * lam0 * (lam0 - eta)
    return loglik, gradient, weight


def newton(model: str, x: Tensor, y: Tensor, free: list[int], offset: Tensor,
           start: Tensor | None = None) -> Tensor:
    """MLE of the free coefficients with the others fixed (``offset`` = X_fixed b_fixed).

    Newton steps with step halving on the log likelihood, analytic gradient
    and Hessian, convergence when the scaled step and gradient are below
    1e-10. Raises ``convergence_failure`` (including complete separation).
    """
    z = x[:, free]
    beta = torch.zeros(len(free), dtype=torch.float64) if start is None else start.clone()
    loglik = link_terms(model, offset + z @ beta, y)[0].sum()
    for _ in range(200):
        _, gradient, weight = link_terms(model, offset + z @ beta, y)
        score = z.T @ gradient
        information = (z * weight[:, None]).T @ z
        factor, info = torch.linalg.cholesky_ex((information + information.T) / 2)
        if int(info) != 0:
            raise AnalysisError("convergence_failure", "The information matrix is singular; the "
                                "model is not identified (collinearity or separation).")
        step = torch.cholesky_solve(score[:, None], factor)[:, 0]
        accepted = False
        for _ in range(60):
            candidate = beta + step
            value = link_terms(model, offset + z @ candidate, y)[0].sum()
            if bool(torch.isfinite(value)) and float(value) >= float(loglik) - 1e-12 * abs(float(loglik)):
                accepted = True
                break
            step = step / 2
        if not accepted:
            raise AnalysisError("convergence_failure", "No ascent step was found.")
        beta, loglik = candidate, value
        if float(step.abs().max()) < 1e-10 * (1 + float(beta.abs().max())) \
                and float(score.abs().max()) < 1e-6 * (1 + float(information.diagonal().max())):
            if float(beta.abs().max()) > 1e6:
                break
            return beta
    raise AnalysisError("convergence_failure", "Newton iterations did not converge; the data may "
                        "be (quasi-)completely separated.")


def _restrictions(terms: list[str], null: Any) -> tuple[list[int], Tensor]:
    if not isinstance(null, dict) or not 1 <= len(null) <= 24:
        raise AnalysisError("invalid_restrictions", "Supply 1..24 term-to-null-value restrictions.")
    fixed, values = [], []
    for term, value in null.items():
        if term not in terms:
            raise AnalysisError("invalid_restrictions", f"Unknown term {term!r}.")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AnalysisError("invalid_restrictions", "Null values must be finite real numbers.")
        fixed.append(terms.index(term))
        values.append(float(value))
    return fixed, torch.tensor(values, dtype=torch.float64)


def unit_scores(model: str, x: Tensor, y: Tensor, theta: Tensor, codes: Tensor, groups: int
                ) -> tuple[Tensor, Tensor]:
    """Unit score sums [G, k] and the information matrix J = -H [k, k] at ``theta``."""
    _, gradient, weight = link_terms(model, x @ theta, y)
    contributions = x * gradient[:, None]
    sums = contributions.new_zeros((groups, x.shape[1])).index_add_(0, codes, contributions)
    information = (x * weight[:, None]).T @ x
    return sums, (information + information.T) / 2


def score_statistics(scores: Tensor, a: Tensor, multipliers: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """Kline--Santos studentized statistics for unit scores [G, k] and A [q, k].

    ``multipliers`` [B, G] (None gives the unperturbed statistic). Returns
    (quadratic form G, signed T for q = 1 else NaN, ok), batched over B.
    """
    units = scores.shape[0]
    projected = scores @ a.T                                     # [G, q]
    perturbed = projected[None] if multipliers is None else multipliers[:, :, None] * projected[None]
    total = perturbed.sum(1)                                     # [B, q]
    centred = perturbed - perturbed.mean(1, keepdim=True)
    cov = units / (units - 1) * centred.transpose(1, 2) @ centred
    factor, info = torch.linalg.cholesky_ex((cov + cov.transpose(1, 2)) / 2)
    quad = (total[:, None, :] @ torch.cholesky_solve(total[:, :, None], factor))[:, 0, 0]
    if a.shape[0] == 1:
        signed = total[:, 0] / cov[:, 0, 0].clamp_min(0).sqrt()
    else:
        signed = torch.full_like(quad, float("nan"))
    ok = (info == 0) & torch.isfinite(quad) & (quad >= 0)
    return quad, signed, ok


@resident_cpu
def score_bootstrap_test(*, data: Any, y: str, x: Sequence[str] | None, null: dict,
                         model: str = "logit", method: str = "lm", cluster: str | None = None,
                         wild: str = "rademacher", reps: int = 999, seed: int = 0,
                         tail: str = "symmetric", categorical: Sequence[str] | None = None,
                         intercept: bool = True, missing: str = "raise",
                         max_work: int = DEFAULT_MAX_WORK) -> TableSet:
    """Kline--Santos score bootstrap test of coefficient restrictions after binary logit or probit.

    ``null`` maps 1..24 coefficient terms to hypothesised values. The model
    is fitted by Newton ML with analytic derivatives, unrestricted and with
    the restricted coefficients fixed. ``method="lm"`` (default) evaluates the
    score contributions and Hessian at the restricted MLE and reports the
    robust score (LM) statistic; ``method="wald"`` uses the unrestricted MLE
    and the robust Wald statistic. In both, the bootstrap distribution is that
    of the studentized perturbed score sum (Kline and Santos 2012, eq. 14) with
    independent multipliers per observation or per ``cluster`` (Rademacher,
    Webb or Mammen) and the recentred sample covariance of the perturbed unit
    scores with the G/(G-1) factor; no model is re-estimated in the draws.
    One restriction: ``tail`` symmetric |T| (default), equal_tail, upper or
    lower on the signed T; several: the quadratic form, upper tail. Plus-one
    p-values. Unweighted binary 0/1 outcomes only; failed draws (singular
    perturbed covariance) raise ``bootstrap_failure``.
    """
    if model not in ("logit", "probit"):
        raise AnalysisError("invalid_option", "model must be 'logit' or 'probit'.")
    if method not in ("lm", "wald"):
        raise AnalysisError("invalid_option", "method must be 'lm' or 'wald'.")
    if tail not in ("symmetric", "equal_tail", "upper", "lower"):
        raise AnalysisError("invalid_option", "tail must be symmetric, equal_tail, upper or lower.")
    wm.check_kind(wild)
    reps = wm.check_count(reps, "reps", 99, 99_999)
    seed = wm.check_count(seed, "seed", 0, 2**63 - 1)
    max_work = wm.check_count(max_work, "max_work", 1, 10**14)
    if cluster is not None and not isinstance(cluster, str):
        raise AnalysisError("invalid_cluster", "cluster must name one column.")
    spec = make_spec(model, outcome=y, predictors=column_list(x, "x"),
                     categorical=column_list(categorical, "categorical"),
                     covariance="cluster" if cluster else "nonrobust", cluster=cluster,
                     intercept=intercept, missing=missing)
    frame = ModelFrame(spec, data)
    design = frame.drop_collinear(frame.design(), None)
    matrix, terms = design.x, list(design.terms)
    outcome = frame.numeric(y)
    if not bool(((outcome == 0) | (outcome == 1)).all()):
        raise AnalysisError("invalid_outcome", "The outcome must be coded 0/1.")
    n, k = matrix.shape
    if cluster is not None:
        codes, groups = frame.codes([cluster])
    else:
        codes, groups = torch.arange(n), n
    if groups < 3:
        raise AnalysisError("insufficient_clusters", "At least three independent units are required.")
    fixed, values = _restrictions(terms, null)
    q = len(fixed)
    if q > 1 and tail != "symmetric":
        raise AnalysisError("invalid_option", "Joint tests use the upper tail of the quadratic form.")
    work = (reps + 1) * (groups * q * 6 + q ** 3) + 400 * n * k * k
    if work > max_work:
        raise AnalysisError("work_budget_exceeded", f"The score bootstrap needs about {work:.3g} "
                            f"floating-point operations, above max_work={max_work}.")
    unrestricted = newton(model, matrix, outcome, list(range(k)), torch.zeros(n, dtype=torch.float64))
    free = [j for j in range(k) if j not in fixed]
    offset = matrix[:, fixed] @ values
    restricted = torch.zeros(k, dtype=torch.float64)
    restricted[fixed] = values
    if free:
        restricted[free] = newton(model, matrix, outcome, free, offset)
    selector = torch.zeros((q, k), dtype=torch.float64)
    selector[torch.arange(q), torch.tensor(fixed)] = 1.0
    theta = restricted if method == "lm" else unrestricted
    scores, information = unit_scores(model, matrix, outcome, theta, codes.to(torch.int64), groups)
    factor, info = torch.linalg.cholesky_ex(information)
    if int(info) != 0:
        raise AnalysisError("singular_information", "The information matrix is singular at the "
                            "evaluation point.")
    a = torch.cholesky_solve(selector.T.contiguous(), factor).T          # R J^-1
    if method == "lm":
        quad, signed, ok = score_statistics(scores, a, None)
    else:
        projected = scores @ a.T
        centred = projected - projected.mean(0, keepdim=True)
        cov = groups / (groups - 1) * centred.T @ centred
        delta = unrestricted[fixed] - values
        chol, good = torch.linalg.cholesky_ex(cov)
        quad = (delta[None, :] @ torch.cholesky_solve(delta[:, None], chol))[0]
        signed = delta[:1] / cov[0, 0].sqrt() if q == 1 else torch.tensor([float("nan")])
        ok = (good == 0).reshape(1) & torch.isfinite(quad)
    if not bool(ok.all()):
        raise AnalysisError("singular_covariance", "The score covariance of the tested "
                            "restrictions is singular.")
    observed_quad, observed_signed = float(quad[0]), float(signed[0])
    per_draw = 8 * groups * (3 * q + 2)
    block = max(1, _BLOCK_BYTES // per_draw)
    quads, signs = [], []
    for multipliers in wm.blocks(wild, reps, groups, seed, block):
        b_quad, b_signed, good = score_statistics(scores, a, multipliers)
        failed = int((~good).sum())
        if failed:
            raise AnalysisError("bootstrap_failure", f"{failed} score-bootstrap draws had a singular "
                                "perturbed score covariance; no draw is dropped.")
        quads.append(b_quad)
        signs.append(b_signed)
    quads, signs = torch.cat(quads), torch.cat(signs)
    if q == 1:
        # With J = -H, R J^-1 sum s(theta~) is the one-step estimate of R theta^ - r,
        # so the signed LM statistic has the orientation of the Wald statistic.
        statistic = observed_signed
        probability = wm.tail_p_values(statistic, signs, tail)
        rule = tail
    else:
        statistic = observed_quad
        probability = wm.tail_p_values(observed_quad, quads, "upper")
        rule = "upper"
    row = {"model": model, "method": method, "statistic": statistic, "quadratic_form": observed_quad,
           "df": q, "asymptotic_p_value": chi2_sf(observed_quad, q), "bootstrap_p_value": probability,
           "mc_std_error": wm.mc_error(probability, reps), "tail": rule}
    estimates = [{"term": term, "unrestricted": float(unrestricted[j]), "restricted": float(restricted[j])}
                 for j, term in enumerate(terms)]
    return TableSet(
        {"test": table([row], title="Kline-Santos score bootstrap"),
         "estimates": table(estimates, title="Unrestricted and null-restricted ML estimates")},
        title="Score bootstrap test", source=SOURCE, working_paper=WORKING_PAPER, model=model,
        method=method, null=null, observations=n, units=groups, cluster=cluster, multiplier=wild,
        draws=reps, failed_draws=0, seed=seed, p_value_rule="plus-one (1 + #exceedances)/(B + 1)",
        studentization="recentred sample covariance of (perturbed) unit scores, factor G/(G-1)",
        evaluation_point="null-restricted MLE" if method == "lm" else "unrestricted MLE",
        restricted_log_likelihood=float(link_terms(model, matrix @ restricted, outcome)[0].sum()),
        unrestricted_log_likelihood=float(link_terms(model, matrix @ unrestricted, outcome)[0].sum()),
        terms=terms, work=work, max_work=max_work, device="cpu", dtype="float64",
        assumptions="correctly specified binary index model; independent observations or clusters; "
                    "no re-estimation in bootstrap draws (score bootstrap)")
