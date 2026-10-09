"""Native conditional ML kernels for eight binary-mediator model domains.

The likelihood factorizes into a Bernoulli mediator and a conditional outcome.
Observed information therefore has exact zero cross-equation blocks. HC0 uses
the complete paired row scores and retains their empirical cross-equation meat.
All returned coordinates use the supplied controls' original units.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F

from openecon.analysis_contracts import AnalysisError
from openecon.resources import workspace_budget_bytes

DT = torch.float64
MAX_N = 4096
MAX_C = 10
WORK_LIMIT = 2_000_000_000
LOG_2PI = math.log(2 * math.pi)


def _options(mediator_link, outcome_model, interaction, covariance):
    if mediator_link not in {"logit", "probit"}:
        raise AnalysisError("invalid_option", "mediator_link must be logit or probit.")
    if outcome_model not in {"gaussian", "logit", "probit", "poisson"}:
        raise AnalysisError("invalid_option", "Unsupported binary mediation outcome model.")
    if type(interaction) is not bool:
        raise AnalysisError("invalid_option", "interaction must be a boolean.")
    if covariance not in {"OIM", "HC0"}:
        raise AnalysisError("invalid_covariance", "covariance must be OIM or HC0.")


def _data(y, m, a, c, outcome_model):
    values = []
    for value, name, dimensions in ((y, "y", 1), (m, "m", 1), (a, "a", 1), (c, "c", 2)):
        if (
            not isinstance(value, torch.Tensor) or value.ndim != dimensions
            or value.device.type != "cpu" or value.is_complex()
            or value.requires_grad or value.dtype == torch.bool
        ):
            raise AnalysisError("unsupported_input", f"{name} needs a real resident CPU tensor without gradients.")
        if value.shape[0] > MAX_N or (dimensions == 2 and value.shape[1] > MAX_C):
            raise AnalysisError("resource_limit", "Binary mediation supports at most 4096 rows and 10 controls.")
        values.append(value.detach().to(dtype=DT, device="cpu"))
    y, m, a, c = values
    n = len(y)
    if not 1 <= n <= MAX_N or c.shape[1] > MAX_C:
        raise AnalysisError("resource_limit", "Binary mediation supports at most 4096 rows and 10 controls.")
    if len(m) != n or len(a) != n or c.shape[0] != n:
        raise AnalysisError("shape_mismatch", "All equation inputs must retain the same complete rows.")
    if any(not bool(torch.isfinite(value).all()) or bool((value.abs() > 1e12).any()) for value in values):
        raise AnalysisError("invalid_input", "Complete finite numeric inputs must have magnitude at most 1e12.")
    for value, name in ((m, "mediator"), (a, "treatment")):
        if not bool(((value == 0) | (value == 1)).all()) or len(torch.unique(value)) != 2:
            raise AnalysisError("invalid_binary", f"{name} must contain exactly numeric 0 and 1, both observed.")
    if outcome_model in {"logit", "probit"}:
        if not bool(((y == 0) | (y == 1)).all()) or len(torch.unique(y)) != 2:
            raise AnalysisError("invalid_binary", "Binary outcome must contain numeric 0 and 1, both observed.")
    elif outcome_model == "poisson":
        if bool((y < 0).any()) or not bool((y == y.round()).all()):
            raise AnalysisError("invalid_count", "Poisson outcome requires nonnegative integer counts.")
        if not bool((y > 0).any()):
            raise AnalysisError("no_finite_mle", "An all-zero Poisson outcome has no finite intercept ML fit.")
    return y, m, a, c


def _designs(m, a, c, interaction):
    one = torch.ones((len(a), 1), dtype=DT, device="cpu")
    xm = torch.cat((one, a[:, None], c), dim=1)
    parts = (one, a[:, None], m[:, None])
    if interaction:
        parts = (*parts, (a * m)[:, None])
    xy = torch.cat((*parts, c), dim=1)
    return xm, xy


def _geometry(m, a, c, interaction, outcome_model):
    k = c.shape[1]
    center = c.mean(0)
    centered = c - center
    if k:
        largest = centered.abs().amax(0)
        if bool((largest == 0).any()):
            raise AnalysisError("rank_deficient", "Controls cannot be constant or duplicate the intercept.")
        scale = largest * (centered / largest).square().mean(0).sqrt()
        z = centered / scale
    else:
        scale, z = center.clone(), centered
    xm, xy = _designs(m, a, z, interaction)
    for x, name in ((xm, "mediator"), (xy, "outcome")):
        if len(a) <= x.shape[1] + 1:
            raise AnalysisError("insufficient_data", f"The {name} equation requires residual information beyond its regressors.")
        singular = torch.linalg.svdvals(x)
        if float(singular[-1]) <= float(singular[0]) * 1e-10:
            raise AnalysisError("rank_deficient", f"The centered/scaled {name} design is rank deficient or ill-conditioned.")
    qm, qy = xm.shape[1], xy.shape[1]
    p = qm + qy + int(outcome_model == "gaussian")
    transform = torch.eye(p, dtype=DT, device="cpu")
    for start, q, control_start in ((0, qm, 2), (qm, qy, 3 + int(interaction))):
        if k:
            transform[start, start + control_start:start + q] = -center / scale
            indices = torch.arange(start + control_start, start + q, device="cpu")
            transform[indices, indices] = scale.reciprocal()
    inverse = torch.eye(p, dtype=DT, device="cpu")
    for start, q, control_start in ((0, qm, 2), (qm, qy, 3 + int(interaction))):
        if k:
            inverse[start, start + control_start:start + q] = center
            indices = torch.arange(start + control_start, start + q, device="cpu")
            inverse[indices, indices] = scale
    return xm, xy, transform, inverse, center, scale


def _glmlog(eta, y, kind):
    if kind == "logit":
        return torch.where(y == 1, -F.softplus(-eta), -F.softplus(eta))
    if kind == "probit":
        return torch.special.log_ndtr(torch.where(y == 1, eta, -eta))
    return y * eta - eta.exp() - torch.lgamma(y + 1)


def _glm_parts(x, y, beta, kind):
    eta = x @ beta
    ll = _glmlog(eta, y, kind)
    if kind == "logit":
        prob = torch.sigmoid(eta)
        derivative = y - prob
        curvature = prob * torch.sigmoid(-eta)
    elif kind == "probit":
        sign = 2 * y - 1
        signed = sign * eta
        ratio = (-eta.square() / 2 - LOG_2PI / 2 - torch.special.log_ndtr(signed)).exp()
        derivative = sign * ratio
        curvature = ratio * (ratio + signed)
    else:
        mean = eta.exp()
        derivative = y - mean
        curvature = mean
    scores = x * derivative[:, None]
    information = x.T @ (x * curvature[:, None])
    return ll, scores, (information + information.T) / 2


def _inverse_information(information, label):
    if not bool(torch.isfinite(information).all()) or bool((information.diag() <= 0).any()):
        raise AnalysisError("unidentified_model", f"Nonpositive/nonfinite {label} observed information; no ridge.")
    factor = information.diag().sqrt().reciprocal()
    normalized = information * factor[:, None] * factor[None, :]
    eigen = torch.linalg.eigvalsh(normalized)
    if float(eigen[0]) <= max(1e-12, float(eigen[-1]) * 1e-10):
        raise AnalysisError("unidentified_model", f"Ill-conditioned {label} observed information; no ridge.")
    inverse = torch.linalg.inv(normalized) * factor[:, None] * factor[None, :]
    if not bool(torch.isfinite(inverse).all()):
        raise AnalysisError("numerical_failure", f"{label} information inverse exceeds float64 representation.")
    return (inverse + inverse.T) / 2


def _glm_fit(x, y, kind, max_iterations, tolerance):
    beta = torch.zeros(x.shape[1], dtype=DT, device="cpu")
    average = y.mean()
    beta[0] = (
        torch.logit(average) if kind == "logit"
        else torch.special.ndtri(average) if kind == "probit"
        else average.log()
    )
    ll, scores, information = _glm_parts(x, y, beta, kind)
    backtracks = 0
    for iteration in range(1, max_iterations + 1):
        bread = _inverse_information(information, kind)
        score = scores.sum(0)
        step = bread @ score
        gate = float(score @ step)
        # A separated likelihood can have a tiny score and Newton decrement
        # while its parameter step stays O(1). Both gates are essential.
        if gate <= max(tolerance * tolerance * len(y), 1e-18) and float(step.abs().max()) <= tolerance * (1 + float(beta.abs().max())):
            return beta, dict(iterations=iteration, backtracks=backtracks,
                              score_decrement=gate, parameter_step=float(step.abs().max()), converged=True)
        if not bool(torch.isfinite(step).all()) or gate < -1e-10:
            raise AnalysisError("numerical_failure", "Invalid native Newton direction.")
        accepted = False
        for halving in range(50):
            amount = 0.5 ** halving
            candidate = beta + amount * step
            if not bool(torch.isfinite(candidate).all()):
                continue
            candidate_ll = _glmlog(x @ candidate, y, kind)
            if not bool(torch.isfinite(candidate_ll).all()):
                continue
            old, new = float(ll.sum()), float(candidate_ll.sum())
            roundoff = 32 * torch.finfo(DT).eps * max(1.0, abs(old))
            if new + roundoff >= old + 1e-4 * amount * max(0.0, gate):
                beta = candidate
                ll, scores, information = _glm_parts(x, y, beta, kind)
                backtracks += halving
                accepted = True
                break
        if not accepted:
            raise AnalysisError("nonconvergence", "Native likelihood line search failed; no fallback.")
    raise AnalysisError("no_finite_mle", "No finite stationary GLM ML fit within the iteration/identification budget; separation is not approximated.")


def _gaussian_parts(x, y, theta):
    beta, logsigma = theta[:-1], theta[-1]
    residual = y - x @ beta
    inverse_variance = (-2 * logsigma).exp()
    scaled_square = residual.square() * inverse_variance
    ll = -LOG_2PI / 2 - logsigma - scaled_square / 2
    beta_score = x * (residual * inverse_variance)[:, None]
    scores = torch.cat((beta_score, (scaled_square - 1)[:, None]), 1)
    q = len(beta)
    information = torch.zeros((q + 1, q + 1), dtype=DT, device="cpu")
    information[:q, :q] = x.T @ x * inverse_variance
    cross = 2 * beta_score.sum(0)
    information[:q, q] = cross
    information[q, :q] = cross
    information[q, q] = 2 * scaled_square.sum()
    return ll, scores, information


def _gaussian_fit(x, y):
    center = y.mean()
    centered = y - center
    beta = torch.linalg.lstsq(x, centered, rcond=1e-12, driver="gelsd").solution
    beta[0] += center
    residual = y - x @ beta
    sigma2 = residual.square().mean()
    energy = centered.square().mean()
    if float(energy) == 0 or not math.isfinite(float(sigma2)) or float(sigma2) <= max(torch.finfo(DT).tiny, float(energy) * 1e-24):
        raise AnalysisError("no_finite_mle", "Zero/numerically unidentified Gaussian residual variance has no finite log-sigma ML fit.")
    theta = torch.cat((beta, sigma2.log()[None] / 2))
    _, scores, information = _gaussian_parts(x, y, theta)
    bread = _inverse_information(information, "Gaussian")
    score = scores.sum(0)
    gate = float(score @ bread @ score)
    if not math.isfinite(gate) or gate > 1e-10:
        raise AnalysisError("nonconvergence", "Gaussian normalized ML score gate failed.")
    return theta, dict(iterations=1, backtracks=0, score_decrement=gate,
                      parameter_step=0.0, converged=True)


def _joint_parts(theta, xm, xy, y, m, mediator_link, outcome_model):
    qm = xm.shape[1]
    mediator_theta, outcome_theta = theta[:qm], theta[qm:]
    lm, sm, im = _glm_parts(xm, m, mediator_theta, mediator_link)
    ly, sy, iy = (
        _gaussian_parts(xy, y, outcome_theta) if outcome_model == "gaussian"
        else _glm_parts(xy, y, outcome_theta, outcome_model)
    )
    p = len(theta)
    information = torch.zeros((p, p), dtype=DT, device="cpu")
    information[:qm, :qm], information[qm:, qm:] = im, iy
    return torch.stack((lm, ly), 1), torch.cat((sm, sy), 1), information


def row_loglikelihood(theta, y, m, a, c, *, mediator_link="logit", outcome_model="gaussian", interaction=True):
    """Differentiable complete n×2 likelihood in original parameter coordinates."""
    xm, xy = _designs(m, a, c, interaction)
    qm = xm.shape[1]
    lm = _glmlog(xm @ theta[:qm], m, mediator_link)
    if outcome_model == "gaussian":
        residual = y - xy @ theta[qm:-1]
        ly = -LOG_2PI / 2 - theta[-1] - residual.square() * (-2 * theta[-1]).exp() / 2
    else:
        ly = _glmlog(xy @ theta[qm:], y, outcome_model)
    return torch.stack((lm, ly), 1)


def _parameter_names(k, interaction, outcome_model):
    return ["mediator:_cons", "mediator:a", *[f"mediator:c{i}" for i in range(k)],
            "outcome:_cons", "outcome:a", "outcome:m",
            *(["outcome:a:m"] if interaction else []),
            *[f"outcome:c{i}" for i in range(k)],
            *(["outcome:log_sigma"] if outcome_model == "gaussian" else [])]


def _evaluate(theta, xm, xy, transform, inverse, y, m, a, c, mediator_link, outcome_model, interaction, covariance):
    # Compute at centered controls, then transform score covectors and the
    # complete inverse information rather than invert an ill-scaled matrix.
    standardized = inverse @ theta
    ll, score_standardized, information_standardized = _joint_parts(
        standardized, xm, xy, y, m, mediator_link, outcome_model
    )
    bread_standardized = _inverse_information(information_standardized, "joint")
    scores = score_standardized @ inverse
    information = inverse.T @ information_standardized @ inverse
    bread = transform @ bread_standardized @ transform.T
    bread = (bread + bread.T) / 2
    if covariance == "OIM":
        joint = bread
    else:
        impact = score_standardized @ bread_standardized @ transform.T
        joint = impact.T @ impact
        joint = (joint + joint.T) / 2
    if any(not bool(torch.isfinite(value).all()) for value in (ll, scores, information, bread, joint)):
        raise AnalysisError("numerical_failure", "Complete original-unit likelihood/covariance is nonfinite.")
    original_xm, original_xy = _designs(m, a, c, interaction)
    qm = xm.shape[1]
    stop = len(theta) - int(outcome_model == "gaussian")
    for original, standardized_value in (
        (original_xm @ theta[:qm], xm @ standardized[:qm]),
        (original_xy @ theta[qm:stop], xy @ standardized[qm:stop]),
    ):
        if not bool(torch.allclose(original, standardized_value, rtol=1e-9, atol=1e-9)):
            raise AnalysisError("numerical_failure", "Original-unit linear predictor loses information through control cancellation.")
    return ll, scores, information, bread, joint


def evaluate_joint(theta, y, m, a, c, *, mediator_link="logit", outcome_model="gaussian", interaction=True, covariance="HC0"):
    """Validate and recompute complete saved equation state without fitting."""
    _options(mediator_link, outcome_model, interaction, covariance)
    y, m, a, c = _data(y, m, a, c, outcome_model)
    xm, xy, transform, inverse, center, scale = _geometry(m, a, c, interaction, outcome_model)
    p = len(transform)
    if not isinstance(theta, torch.Tensor) or theta.ndim != 1 or len(theta) != p or theta.device.type != "cpu" or theta.is_complex():
        raise AnalysisError("invalid_state", "Saved theta does not match complete native equation coordinates.")
    theta = theta.detach().to(dtype=DT, device="cpu")
    if not bool(torch.isfinite(theta).all()):
        raise AnalysisError("invalid_state", "Saved theta must be finite.")
    ll, scores, information, bread, joint = _evaluate(theta, xm, xy, transform, inverse, y, m, a, c, mediator_link, outcome_model, interaction, covariance)
    return dict(theta=theta.tolist(), covariance=joint.tolist(), information=information.tolist(),
                bread=bread.tolist(), scores=scores.tolist(), row_loglikelihood=ll.tolist(),
                log_likelihood=float(ll.sum()), equation_log_likelihood=ll.sum(0).tolist(),
                parameter_names=_parameter_names(c.shape[1], interaction, outcome_model),
                parameter_slices={"mediator": [0, xm.shape[1]], "outcome": [xm.shape[1], p]},
                n=len(y), k_controls=c.shape[1], mediator_link=mediator_link,
                outcome_model=outcome_model, interaction=interaction, covariance_type=covariance,
                control_center=center.tolist(), control_scale=scale.tolist())


def stationarity_check(theta, y, m, a, c, *, mediator_link="logit", outcome_model="gaussian", interaction=True, tolerance=1e-9):
    """Certify saved ML stationarity without fitting or trusting fit metadata.

    A small likelihood score alone cannot certify a finite Bernoulli/Poisson
    maximum: scores also vanish along separated diverging sequences. The
    centered-control Newton parameter step must therefore be small as well.
    Gaussian coefficient steps are measured in residual-standard-deviation
    units, together with the log-sigma step, so outcome units do not affect the
    gate. This is a numerical finite-stationary certificate, not a signature.
    """
    _options(mediator_link, outcome_model, interaction, "OIM")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not 1e-12 <= tolerance <= 1e-5:
        raise AnalysisError("invalid_option", "tolerance must be finite in [1e-12,1e-5].")
    y, m, a, c = _data(y, m, a, c, outcome_model)
    xm, xy, _, inverse, _, _ = _geometry(m, a, c, interaction, outcome_model)
    if not isinstance(theta, torch.Tensor) or theta.ndim != 1 or len(theta) != len(inverse) or theta.device.type != "cpu" or theta.is_complex():
        raise AnalysisError("invalid_state", "Saved theta does not match complete native equation coordinates.")
    theta = theta.detach().to(dtype=DT, device="cpu")
    if not bool(torch.isfinite(theta).all()):
        raise AnalysisError("invalid_state", "Saved theta must be finite.")
    standardized = inverse @ theta
    qm = xm.shape[1]
    result = {}
    for name, x, response, parameter, kind in (
        ("mediator", xm, m, standardized[:qm], mediator_link),
        ("outcome", xy, y, standardized[qm:], outcome_model),
    ):
        _, scores, information = (
            _gaussian_parts(x, response, parameter) if kind == "gaussian"
            else _glm_parts(x, response, parameter, kind)
        )
        bread = _inverse_information(information, name)
        score = scores.sum(0)
        step = bread @ score
        decrement = float(score @ step)
        absolute_step = float(step.abs().max())
        if kind == "gaussian":
            normalized_step = max(float(step[:-1].abs().max() * torch.exp(-parameter[-1])), abs(float(step[-1])))
        else:
            normalized_step = absolute_step / (1 + float(parameter.abs().max()))
        limit = max(float(tolerance) ** 2 * len(y), 1e-18)
        if (
            not math.isfinite(decrement) or decrement < -1e-12
            or not math.isfinite(normalized_step) or decrement > limit
            or normalized_step > tolerance
        ):
            raise AnalysisError("invalid_result", f"Saved {name} parameters fail the actual normalized ML score/step stationarity gates.")
        result[name] = dict(score_decrement=decrement, parameter_step=absolute_step,
                            normalized_parameter_step=normalized_step,
                            score_decrement_limit=limit, parameter_step_limit=float(tolerance),
                            stationary=True)
    return result


def fit_joint(y, m, a, c, *, mediator_link="logit", outcome_model="gaussian", interaction=True, covariance="HC0", max_iterations=200, tolerance=1e-9):
    """Fit complete conditional ML equations and joint OIM/HC0 covariance."""
    _options(mediator_link, outcome_model, interaction, covariance)
    if type(max_iterations) is not int or not 1 <= max_iterations <= 1000:
        raise AnalysisError("invalid_option", "max_iterations must be an integer in 1..1000.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not 1e-12 <= tolerance <= 1e-5:
        raise AnalysisError("invalid_option", "tolerance must be finite in [1e-12,1e-5].")
    y, m, a, c = _data(y, m, a, c, outcome_model)
    qm, qy = 2 + c.shape[1], 3 + int(interaction) + c.shape[1]
    p = qm + qy + int(outcome_model == "gaussian")
    # The joint-coordinate square is a conservative ceiling for the two
    # separate Newton Hessians and complete paired-score covariance work.
    work = len(y) * p * p * max_iterations
    memory = 8 * (len(y) * (8 * p + 16) + 20 * p * p)
    if work > WORK_LIMIT or memory > workspace_budget_bytes():
        raise AnalysisError("resource_limit", "Complete binary mediation Newton/score workspace exceeds the declared budget.")
    try:
        xm, xy, transform, inverse, _, _ = _geometry(m, a, c, interaction, outcome_model)
        bm, dm = _glm_fit(xm, m, mediator_link, max_iterations, float(tolerance))
        by, dy = (
            _gaussian_fit(xy, y) if outcome_model == "gaussian"
            else _glm_fit(xy, y, outcome_model, max_iterations, float(tolerance))
        )
        standardized = torch.cat((bm, by))
        theta = transform @ standardized
        # Restoring the original-unit vector must reproduce its fitted state.
        if not bool(torch.allclose(inverse @ theta, standardized, rtol=1e-9, atol=1e-9)):
            raise AnalysisError("numerical_failure", "Original-unit parameters cannot retain their complete fitted precision.")
        result = evaluate_joint(theta, y, m, a, c, mediator_link=mediator_link,
                                outcome_model=outcome_model, interaction=interaction, covariance=covariance)
        try:
            stationarity_check(theta, y, m, a, c, mediator_link=mediator_link,
                               outcome_model=outcome_model, interaction=interaction,
                               tolerance=float(tolerance))
        except AnalysisError as error:
            if error.code != "invalid_result":
                raise
            raise AnalysisError(
                "numerical_failure",
                "Original-unit parameter representation fails the final normalized ML score/step gates; no unrestorable fit is returned.",
            ) from error
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("numerical_failure", "Native equation factorization failed; no ridge or fallback.") from error
    result.update(equations={
        "mediator": {**dm, "parameter_start": 0, "parameter_stop": qm,
                     "log_likelihood": result["equation_log_likelihood"][0]},
        "outcome": {**dy, "parameter_start": qm, "parameter_stop": p,
                    "log_likelihood": result["equation_log_likelihood"][1]},
    }, convergence={"converged": True, "mediator": dm, "outcome": dy},
                  solver="native centered/scaled Newton ML; closed Gaussian normalized ML",
                  score_convention="complete uncorrected per-subject conditional likelihood scores",
                  information_convention="full observed Hessian; exact factorized cross-equation zero blocks",
                  estimated_work=work, work_limit=WORK_LIMIT, workspace_bytes=memory,
                  max_iterations=max_iterations, tolerance=float(tolerance))
    return result
