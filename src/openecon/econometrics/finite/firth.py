"""Full-model Jeffreys logistic fits and constrained penalized-likelihood inference."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines import distributions as dist
from . import common as c


def options(max_iter, tol):
    return c.c.check_count(max_iter, "max_iter", maximum=200), c.c.check_number(
        tol, "tol", minimum=1e-10, maximum=1e-6
    )


def objective(theta, x, y):
    eta = x @ theta
    weight = torch.sigmoid(eta) * torch.sigmoid(-eta)
    info = x.T @ (weight[:, None] * x)
    sign, logdet = torch.linalg.slogdet(info)
    if (
        float(sign.detach()) <= 0
        or not bool(torch.isfinite(logdet))
        or float(eta.detach().abs().max()) > 100
    ):
        raise AnalysisError(
            "numerical_failure",
            "Firth information is singular/nonfinite or logits exceed the supported +/-100 domain.",
        )
    return (y * eta - torch.nn.functional.softplus(eta)).sum() + 0.5 * logdet


def solve(x, y, *, start=None, restriction=None, target=None, max_iter=100, tol=1e-8):
    """Newton ascent in the free subspace, keeping the FULL Jeffreys determinant."""
    p = x.shape[1]
    initial = torch.zeros(p, dtype=torch.float64) if start is None else start.clone()
    if restriction is None:
        origin, basis = torch.zeros(p, dtype=torch.float64), torch.eye(p, dtype=torch.float64)
    else:
        # Full QR constructs an orthonormal nullspace, unlike fixing a scaled
        # coordinate (a raw intercept restriction involves normalized slopes).
        c.condition(restriction.T)
        q, r = torch.linalg.qr(restriction.T, mode="complete")
        rank = restriction.shape[0]
        origin = (
            q[:, :rank]
            @ torch.linalg.solve_triangular(r[:rank].T, target[:, None], upper=False).flatten()
        )
        basis = q[:, rank:]
    free = basis.T @ (initial - origin)
    trace = []
    if not len(free):
        theta = origin
        value = float(objective(theta, x, y))
        return theta, {
            "converged": True,
            "iterations": 0,
            "free_score_max": 0.0,
            "penalized_loglik": value,
            "free_dimensions": 0,
            "trace": [],
            "constraint_residual": float((restriction @ theta - target).abs().max()),
        }
    for iteration in range(max_iter + 1):
        v = free.detach().requires_grad_(True)

        def function(a):
            return objective(origin + basis @ a, x, y)

        value = function(v)
        gradient = torch.autograd.grad(value, v)[0].detach()
        hessian = torch.autograd.functional.hessian(function, v).detach()
        score = float(gradient.abs().max())
        curvature = -0.5 * (hessian + hessian.T)
        eigen = torch.linalg.eigvalsh(curvature)
        trace.append(
            {
                "iteration": iteration,
                "penalized_loglik": float(value.detach()),
                "free_score_max": score,
            }
        )
        if score <= tol and float(eigen[0]) > 1e-12:
            theta = origin + basis @ free
            return theta, {
                "converged": True,
                "iterations": iteration,
                "free_score_max": score,
                "penalized_loglik": float(value.detach()),
                "free_dimensions": len(free),
                "trace": trace,
                "constraint_residual": 0.0
                if restriction is None
                else float((restriction @ theta - target).abs().max()),
            }
        if iteration == max_iter:
            break
        damping = max(0.0, 1e-6 - float(eigen[0]))
        direction = torch.linalg.solve(
            curvature + damping * torch.eye(len(free), dtype=torch.float64), gradient
        )
        # Bound logits along a trial direction before objective evaluation.
        if float(direction.abs().max()) > 10:
            direction *= 10 / float(direction.abs().max())
        gain = float(gradient @ direction)
        accepted = False
        for backtrack in range(30):
            step = 2.0 ** (-backtrack)
            candidate = free + step * direction
            try:
                trial = float(function(candidate).detach())
            except AnalysisError:
                continue
            rounding = 2e-14 * max(1.0, abs(float(value.detach())))
            if trial >= float(value.detach()) + 1e-4 * step * gain - rounding:
                free = candidate.detach()
                trace[-1].update(damping=damping, step=step, backtracks=backtrack)
                accepted = True
                break
        if not accepted:
            raise AnalysisError(
                "nonconvergence",
                "Firth safeguarded ascent failed; no fit or partial inference is returned.",
            )
    raise AnalysisError(
        "nonconvergence", "Firth score/curvature convergence gates failed at max_iter."
    )


def fisher(x, theta):
    eta = x @ theta
    info = x.T @ ((torch.sigmoid(eta) * torch.sigmoid(-eta))[:, None] * x)
    eigen = torch.linalg.eigvalsh(info)
    if float(eigen[0]) <= 0 or float(eigen[-1] / eigen[0]) > 1e12:
        raise AnalysisError(
            "numerical_failure", "Expected Fisher information is singular or too ill-conditioned."
        )
    return torch.linalg.inv(info)


def fit_state(result):
    state = c.intact(result, "firth_logit")
    x, y = (
        torch.tensor(state["design_scaled"], dtype=torch.float64),
        torch.tensor(state["response"], dtype=torch.float64),
    )
    theta = torch.tensor(state["parameters_scaled"], dtype=torch.float64)
    transform = torch.tensor(state["transform"], dtype=torch.float64)
    return state, x, y, theta, transform


@resident_cpu
def firth_logit(
    data,
    y,
    x=None,
    *,
    intercept=True,
    missing="raise",
    covariance="fisher",
    level=0.95,
    max_iter=100,
    tol=1e-8,
    max_work=100_000_000,
    device="cpu",
    weights=None,
):
    """Fit resident numeric Bernoulli Firth logistic with full expected-Fisher Wald covariance.

    CPU float64, <=2000 input rows and <=8 full-rank coefficients. Jeffreys
    likelihood uses the full design. Wald normal inference is asymptotic;
    profile/test APIs separately refit nuisance parameters. No weighted route.
    """
    c.domain(device, weights)
    level, (max_iter, tol) = c.confidence(level), options(max_iter, tol)
    if covariance != "fisher":
        raise AnalysisError(
            "unsupported_covariance", "Only full inverse expected-Fisher covariance is supported."
        )
    matrix, response, transform, state, settings = c.design(
        data,
        y,
        x,
        intercept,
        missing,
        2000,
        8,
        "firth_logit",
        max_iter * 32,
        max_work,
        records=max_iter + 1,
    )
    if not bool(((response == 0) | (response == 1)).all()):
        raise AnalysisError(
            "invalid_response", "Firth logistic outcome must be numeric Bernoulli 0/1."
        )
    theta, convergence = solve(matrix, response, max_iter=max_iter, tol=tol)
    cov = fisher(matrix, theta)
    raw, raw_cov = transform @ theta, transform @ cov @ transform.T
    critical = dist.normal_isf((1 - level) / 2)
    rows = []
    for term, estimate, variance in zip(state["terms"], raw, raw_cov.diag()):
        se = math.sqrt(float(variance))
        z = float(estimate) / se
        rows.append(
            [
                term,
                float(estimate),
                se,
                z,
                2 * dist.normal_sf(abs(z)),
                float(estimate) - critical * se,
                float(estimate) + critical * se,
            ]
        )
    state.update(
        parameters_scaled=theta.tolist(),
        covariance_scaled=cov.tolist(),
        parameters=raw.tolist(),
        covariance_raw=raw_cov.tolist(),
        convergence=convergence,
        max_iter=max_iter,
        tol=tol,
        covariance="expected_fisher",
        level=level,
        settings=settings,
    )
    fitted = torch.sigmoid(matrix @ theta)
    return c.seal(
        "firth_logit",
        {
            "coefficients": table(
                rows, columns=["term", "estimate", "std_error", "z", "p_value", "ci_low", "ci_high"]
            ),
            "covariance": table(raw_cov.tolist(), columns=state["terms"], index=state["terms"]),
            "fitted": table(
                [
                    [pos, lab, float(yv), float(pv)]
                    for pos, lab, yv, pv in zip(
                        state["sample_positions"], state["sample_labels"], response, fitted
                    )
                ],
                columns=["position", "label", "observed", "probability"],
            ),
            "convergence": table(
                [
                    [
                        convergence["iterations"],
                        convergence["free_score_max"],
                        convergence["penalized_loglik"],
                        True,
                    ]
                ],
                columns=["iterations", "score_max", "penalized_loglik", "converged"],
            ),
        },
        state,
        inference="asymptotic normal Wald; full inverse expected Fisher",
        **settings,
    )


@resident_cpu
def firth_predict(
    result, data, *, level=0.95, missing="raise", max_work=100_000_000, device="cpu", weights=None
):
    """Predict saved links/probabilities with full-covariance delta SE and link-normal mean CI."""
    c.domain(device, weights)
    level = c.confidence(level)
    state = c.intact(result, "firth_logit")
    matrix, sample, settings = c.prediction(data, state, missing, "firth_predict", max_work)
    theta, cov = (
        torch.tensor(state["parameters_scaled"], dtype=torch.float64),
        torch.tensor(state["covariance_scaled"], dtype=torch.float64),
    )
    link = matrix @ theta
    variances = torch.sum((matrix @ cov) * matrix, dim=1)
    if not bool(torch.isfinite(variances).all()) or bool((variances < -1e-12).any()):
        raise AnalysisError("numerical_failure", "Prediction covariance is nonfinite or negative.")
    se = variances.clamp_min(0).sqrt()
    critical = dist.normal_isf((1 - level) / 2)
    probability = torch.sigmoid(link)
    rows = [
        [
            pos,
            lab,
            float(link_value),
            float(s),
            float(prob),
            float(prob * (1 - prob) * s),
            float(lo),
            float(hi),
        ]
        for pos, lab, link_value, s, prob, lo, hi in zip(
            sample["sample_positions"],
            sample["sample_labels"],
            link,
            se,
            probability,
            torch.sigmoid(link - critical * se),
            torch.sigmoid(link + critical * se),
        )
    ]
    return c.seal(
        "firth_predict",
        {
            "predictions": table(
                rows,
                columns=[
                    "position",
                    "label",
                    "link",
                    "link_std_error",
                    "probability",
                    "probability_std_error",
                    "ci_low",
                    "ci_high",
                ],
            )
        },
        {
            "fit": state,
            "prediction_sample": sample,
            "design_scaled": matrix.tolist(),
            "level": level,
            "settings": settings,
        },
        inference="asymptotic link-normal conditional mean CI; delta probability SE",
        **settings,
    )


def lr(full, constrained):
    value = 2 * (full - constrained)
    if value < -1e-7:
        raise AnalysisError(
            "numerical_failure",
            "Constrained penalized likelihood exceeds saved unrestricted optimum.",
        )
    return max(0.0, value)


@resident_cpu
def firth_test(
    result,
    restrictions,
    values=None,
    *,
    max_iter=100,
    tol=1e-8,
    max_work=100_000_000,
    device="cpu",
    weights=None,
):
    """Test full-row-rank raw linear restrictions by same-full-model penalized LR and chi-square limit."""
    c.domain(device, weights)
    max_iter, tol = options(max_iter, tol)
    state = c.intact(result, "firth_logit")
    p, n = len(state["terms"]), len(state["response"])
    settings = c.guard(
        "firth_test", n, p, work=n * p**3 * max_iter * 32, max_work=max_work, records=max_iter + 1
    )
    try:
        r = torch.tensor(restrictions, dtype=torch.float64)
        target = (
            torch.zeros(r.shape[0], dtype=torch.float64)
            if values is None
            else torch.tensor(values, dtype=torch.float64)
        )
    except (TypeError, ValueError, RuntimeError, IndexError) as exc:
        raise AnalysisError(
            "invalid_spec", "Declare a finite rectangular restriction matrix and matching targets."
        ) from exc
    if (
        r.ndim != 2
        or not 1 <= r.shape[0] <= p
        or r.shape[1] != p
        or target.shape != (r.shape[0],)
        or (
            not bool(torch.isfinite(r).all())
            or not bool(torch.isfinite(target).all())
            or float(r.abs().max()) > 1e6
            or float(target.abs().max()) > 1e6
        )
    ):
        raise AnalysisError(
            "invalid_spec",
            "Restrictions must be finite <=1e6, shape [rank,p], with matching targets.",
        )
    c.condition(r.T)
    state, x, y, theta, transform = fit_state(result)
    constrained, convergence = solve(
        x, y, start=theta, restriction=r @ transform, target=target, max_iter=max_iter, tol=tol
    )
    statistic = lr(state["convergence"]["penalized_loglik"], convergence["penalized_loglik"])
    raw_cov = torch.tensor(state["covariance_raw"], dtype=torch.float64)
    contrast = r @ torch.tensor(state["parameters"], dtype=torch.float64)
    contrast_cov = r @ raw_cov @ r.T
    return c.seal(
        "firth_test",
        {
            "test": table(
                [[statistic, r.shape[0], dist.chi2_sf(statistic, r.shape[0])]],
                columns=["penalized_lr", "df", "p_value"],
            ),
            "contrasts": table(
                [[j, float(v), float(q)] for j, (v, q) in enumerate(zip(contrast, target))],
                columns=["restriction", "estimate", "null_value"],
            ),
            "contrast_covariance": table(
                contrast_cov.tolist(), columns=[f"R{i}" for i in range(len(target))]
            ),
            "constrained_coefficients": table(
                [[t, float(b)] for t, b in zip(state["terms"], transform @ constrained)],
                columns=["term", "estimate"],
            ),
        },
        {
            "fit": state,
            "restrictions": r.tolist(),
            "values": target.tolist(),
            "constrained_parameters_scaled": constrained.tolist(),
            "constrained_convergence": convergence,
            "max_iter": max_iter,
            "tol": tol,
            "settings": settings,
        },
        inference="asymptotic chi-square penalized LR; full-model Jeffreys determinant",
        **settings,
    )


@resident_cpu
def firth_profile(
    result,
    terms=None,
    *,
    level=0.95,
    max_iter=100,
    tol=1e-8,
    bracket_steps=12,
    root_steps=40,
    max_work=100_000_000,
    device="cpu",
    weights=None,
):
    """Refit nuisance parameters at each raw coefficient value and invert asymptotic penalized LR."""
    c.domain(device, weights)
    level, (max_iter, tol) = c.confidence(level), options(max_iter, tol)
    c.c.check_count(bracket_steps, "bracket_steps", maximum=16)
    c.c.check_count(root_steps, "root_steps", minimum=24, maximum=48)
    state = c.intact(result, "firth_logit")
    names = state["terms"] if terms is None else c.c.name_list(terms, "terms")
    if not names or set(names) - set(state["terms"]):
        raise AnalysisError("invalid_spec", "Profile terms must name saved raw coefficients.")
    p, n = len(state["terms"]), len(state["response"])
    fits = 2 * len(names) * (bracket_steps + root_steps + 1)
    settings = c.guard(
        "firth_profile",
        n,
        p,
        work=n * p**3 * max_iter * 32 * fits,
        max_work=max_work,
        records=fits * (max_iter + 1),
    )
    state, x, y, theta, transform = fit_state(result)
    raw = torch.tensor(state["parameters"], dtype=torch.float64)
    raw_cov = torch.tensor(state["covariance_raw"], dtype=torch.float64)
    cutoff = dist.chi2_isf(1 - level, 1)
    rows, evaluations = [], []
    for name in names:
        j = state["terms"].index(name)
        estimate, se = float(raw[j]), math.sqrt(float(raw_cov[j, j]))
        restriction = transform[j : j + 1]
        endpoints = []
        for side in (-1, 1):

            def evaluate(value):
                constrained, convergence = solve(
                    x,
                    y,
                    start=theta,
                    restriction=restriction,
                    target=torch.tensor([value], dtype=torch.float64),
                    max_iter=max_iter,
                    tol=tol,
                )
                statistic = lr(
                    state["convergence"]["penalized_loglik"], convergence["penalized_loglik"]
                )
                evaluations.append(
                    {
                        "term": name,
                        "side": side,
                        "value": value,
                        "penalized_lr": statistic,
                        "parameters_scaled": constrained.tolist(),
                        "convergence": convergence,
                    }
                )
                return statistic

            outside = None
            for k in range(bracket_steps):
                candidate = estimate + side * se * 2.0**k
                if evaluate(candidate) >= cutoff:
                    outside = candidate
                    break
            if outside is None:
                raise AnalysisError(
                    "unbracketed_interval",
                    "Profile endpoint was not bracketed; no clipped/Wald replacement is returned.",
                )
            inside = estimate
            for _ in range(root_steps):
                midpoint = (inside + outside) / 2
                if evaluate(midpoint) >= cutoff:
                    outside = midpoint
                else:
                    inside = midpoint
            endpoint = (inside + outside) / 2
            # Confirm both likelihood residual and raw endpoint width, not only iteration count.
            value = evaluate(endpoint)
            if abs(value - cutoff) > 1e-5 or abs(outside - inside) > 1e-6 * max(1.0, abs(endpoint)):
                raise AnalysisError(
                    "nonconvergence", "Profile endpoint failed likelihood/width convergence checks."
                )
            endpoints.append(endpoint)
        rows.append([name, estimate, endpoints[0], endpoints[1], cutoff, level])
    return c.seal(
        "firth_profile",
        {
            "intervals": table(
                rows, columns=["term", "estimate", "ci_low", "ci_high", "lr_cutoff", "level"]
            ),
            "evaluations": table(
                [
                    [
                        v["term"],
                        v["side"],
                        v["value"],
                        v["penalized_lr"],
                        v["convergence"]["iterations"],
                        v["convergence"]["free_score_max"],
                    ]
                    for v in evaluations
                ],
                columns=["term", "side", "value", "penalized_lr", "iterations", "free_score_max"],
            ),
        },
        {
            "fit": state,
            "profile_evaluations": evaluations,
            "terms": names,
            "level": level,
            "cutoff": cutoff,
            "max_iter": max_iter,
            "tol": tol,
            "bracket_steps": bracket_steps,
            "root_steps": root_steps,
            "settings": settings,
        },
        inference="asymptotic one-df chi-square penalized profile LR; nuisance refitted",
        **settings,
    )
