"""Normal random-parameter choice logit and NB2 GLMM with checked quadrature."""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.econometrics.mixed.extended_common import (
    Objective,
    check_factor,
    covariance_report,
    factor,
    factor_size,
    factor_start,
    maximize,
    optimizer_record,
    quadrature,
    transformed,
)
from openecon.engines.linalg import collinear_columns, least_squares


def _fit(spec, data, choice):
    frame = ModelFrame(spec, data)
    if choice and frame.role("availability"):
        available = frame.numeric(frame.role("availability")[0])
        y = frame.numeric(spec.outcome)
        if not bool(((available == 0) | (available == 1)).all()) or bool(
            ((available == 0) & (y != 0)).any()
        ):
            raise AnalysisError(
                "invalid_availability",
                "Availability must be binary; unavailable alternatives cannot be chosen.",
            )
        frame.restrict((available == 1).numpy(), reason="declared unavailable choice alternatives")
    group = frame.role("group")[0]
    codes, groups = frame.codes(group)
    if groups < 2:
        raise AnalysisError(
            "insufficient_groups", "At least two random-effect subjects are required."
        )
    design = frame.design()
    x, y, p = design.x, frame.numeric(spec.outcome), len(design.terms)
    if collinear_columns(x)[1]:
        raise AnalysisError("rank_deficient", "Fixed-effect design must have full rank.")
    if choice:
        random = frame.role("random")
        if not 1 <= len(random) <= 2 or not set(random).issubset(spec.predictors):
            raise AnalysisError(
                "invalid_spec", "Declare one or two random coefficients, each also in x."
            )
        case, alternative = frame.role("case")[0], frame.role("alternative")[0]
        cc, cases = frame.codes([group, case])
        if frame.sample.duplicated([group, case, alternative]).any():
            raise AnalysisError(
                "duplicate_alternative", "Each subject/case/alternative key must occur once."
            )
        if not bool(((y == 0) | (y == 1)).all()):
            raise AnalysisError("invalid_choice", "Chosen outcome must be binary.")
        selected = torch.zeros(cases, dtype=torch.float64).index_add(0, cc, y)
        if not bool((selected == 1).all()) or int(torch.bincount(cc).min()) < 2:
            raise AnalysisError(
                "incomplete_choice",
                "Every available case needs at least two alternatives and exactly one chosen alternative.",
            )
        z = torch.stack([frame.numeric(name) for name in random], dim=1)
        znames = random
        # A variable constant in every case has no utility contrast and cannot be identified.
        for j in range(p):
            maxima = torch.full((cases,), -math.inf, dtype=torch.float64).scatter_reduce(
                0, cc, x[:, j], "amax"
            )
            minima = torch.full((cases,), math.inf, dtype=torch.float64).scatter_reduce(
                0, cc, x[:, j], "amin"
            )
            if not bool((maxima != minima).any()):
                raise AnalysisError(
                    "unidentified_choice",
                    "Choice predictors must vary within at least one available alternative set; use explicit alternative indicators for constants.",
                )
        offset = torch.zeros(frame.n, dtype=torch.float64)
    else:
        random = frame.role("random")
        if len(random) > 1:
            raise AnalysisError(
                "invalid_spec", "menbreg supports a random intercept and at most one random slope."
            )
        znames = [*random, "_cons"]
        z = torch.stack(
            [*[frame.numeric(name) for name in random], torch.ones(frame.n, dtype=torch.float64)],
            dim=1,
        )
        if bool((y < 0).any()) or bool((y != y.round()).any()) or not bool((y > 0).any()):
            raise AnalysisError(
                "invalid_count_outcome",
                "NB2 GLMM needs nonnegative integer counts with positive observations.",
            )
        offset = torch.zeros(frame.n, dtype=torch.float64)
        if frame.role("offset"):
            offset += frame.numeric(frame.role("offset")[0])
        if frame.role("exposure"):
            exposure = frame.numeric(frame.role("exposure")[0])
            if bool((exposure <= 0).any()):
                raise AnalysisError("invalid_exposure", "Exposure must be positive.")
            offset += exposure.log()
    if collinear_columns(z)[1]:
        raise AnalysisError("rank_deficient", "Random-effect design must have full rank.")
    q, structure = z.shape[1], frame.option("covstructure")
    size = factor_size(q, structure)
    k = p + size + int(not choice)
    points, maximum = frame.option("intpoints"), frame.option("max_intpoints")
    if maximum < 2 * points:
        raise AnalysisError(
            "invalid_quadrature",
            "max_intpoints must allow at least one doubled-order convergence check.",
        )
    frame.workspace_plan(
        "checked product quadrature and full information",
        {
            "integration_and_derivative_tape": 32 * (frame.n + groups) * maximum**q * (k + 2),
            "full_information": 64 * k * k,
        },
    )
    if maximum**q > 4096:
        raise AnalysisError("quadrature_budget", "Product quadrature is bounded to 4096 nodes.")

    def make_value(order):
        nodes, logweights = quadrature(order, q)

        def probabilities(theta):
            lower = factor(theta[p : p + size], q, structure)
            eta = (x @ theta[:p] + offset)[:, None] + z @ lower @ nodes.T
            high = torch.full((cases, len(nodes)), -math.inf, dtype=torch.float64).scatter_reduce(
                0, cc[:, None].expand_as(eta), eta, "amax"
            )
            exponent = torch.exp(eta - high[cc])
            denom = torch.zeros_like(high).index_add(0, cc, exponent)
            return eta - high[cc] - denom[cc].log()

        def value(theta):
            lower = factor(theta[p : p + size], q, structure)
            eta = (x @ theta[:p] + offset)[:, None] + z @ lower @ nodes.T
            if choice:
                logrow = probabilities(theta) * y[:, None]
            else:
                r = torch.exp(-theta[-1])
                total = torch.logaddexp(r.log(), eta)
                logrow = (
                    torch.lgamma(y[:, None] + r)
                    - torch.lgamma(r)
                    - torch.lgamma(y[:, None] + 1)
                    + r * (r.log() - total)
                    + y[:, None] * (eta - total)
                )
            panel = torch.zeros((groups, len(nodes)), dtype=torch.float64).index_add(
                0, codes, logrow
            )
            return torch.logsumexp(panel + logweights, dim=1).sum()

        return value, probabilities, nodes, logweights

    # Fit then double the integration order; likelihood, standardized score and
    # ALL covariance entries must settle. No RNG or silent approximate success.
    if choice:
        beta = torch.zeros(p, dtype=torch.float64)
    else:
        beta = least_squares(x, torch.log(y + 0.5) - offset, drop_collinear=False).beta
    starts = [
        torch.cat(
            (
                beta,
                factor_start(q, structure, sd),
                torch.tensor([math.log(0.5)], dtype=torch.float64) if not choice else beta[:0],
            )
        )
        for sd in (0.25, 0.6, 1.0)
    ]
    previous, previous_v, checks, used = None, None, [], 0
    while True:
        value, probabilities, nodes, logweights = make_value(points)
        objective = Objective(value, 4 * frame.n * points**q * k, frame.option("max_work") - used)
        fit, raw_v = maximize(objective, starts)
        used += objective.used
        if previous is not None:
            delta_ll = abs(fit.value - previous.value)
            shift = fit.theta - previous.theta
            standardized_shift = float(shift @ torch.linalg.solve(raw_v, shift))
            normalization = raw_v.diagonal().sqrt()
            v_error = float(
                ((raw_v - previous_v) / normalization[:, None] / normalization).abs().max()
            )
            checks.append(
                {
                    "points": points,
                    "log_likelihood_change": delta_ll,
                    "parameter_shift_squared_se": standardized_shift,
                    "normalized_covariance_change": v_error,
                }
            )
            if delta_ll < 1e-5 * max(1, groups) and standardized_shift < 1e-6 and v_error < 1e-3:
                break
        if points >= maximum:
            raise AnalysisError(
                "integration_not_converged",
                "Quadrature order did not stabilize likelihood, parameters and full covariance; increase max_intpoints within the declared node/work/memory budgets.",
            )
        previous, previous_v = fit, raw_v
        starts = [fit.theta]
        points = min(2 * points, maximum)
    check_factor(fit.theta[p : p + size], q, structure)
    if not choice and float(fit.theta[-1].exp()) < 1e-6:
        raise AnalysisError("boundary_solution", "NB2 dispersion is at the Poisson boundary.")

    def report(theta):
        return torch.cat(
            (
                theta[:p],
                covariance_report(theta[p : p + size], q, structure),
                theta[-1:] if not choice else theta[:0],
            )
        )

    parameters, vc = transformed(fit.theta, raw_v, report)
    terms = [*design.terms, *[f"/var({name}[{group}])" for name in znames]]
    if structure == "unstructured":
        terms += [f"/cov({znames[i]},{znames[j]}[{group}])" for i in range(q) for j in range(i)]
    if not choice:
        terms += ["/lnalpha"]
    if choice:

        def response(theta):
            return (probabilities(theta).exp() * logweights.exp()).sum(1)

        target = "normal-population alternative probabilities conditional on the declared available case set"
    else:

        def response(theta):
            lower = factor(theta[p : p + size], q, structure)
            variance = (z @ lower).square().sum(1)
            return (x @ theta[:p] + offset + 0.5 * variance).exp()

        target = "NB2 population mean integrated over normal subject effects"
    with torch.enable_grad():
        jac = torch.autograd.functional.jacobian(response, fit.theta)
    mean = response(fit.theta)
    se = ((jac @ raw_v) * jac).sum(1).clamp_min(0).sqrt()
    return build_result(
        frame,
        terms=terms,
        params=parameters,
        covariance=vc,
        use_t=False,
        metrics={"log_likelihood": fit.value, "n_groups": groups},
        fitted=mean,
        solver="native checked normal product Gauss-Hermite likelihood",
        optimizer={**optimizer_record(fit, objective), "total_work_used": used},
        inference={
            "correction": "full observed information; all mean/variance/covariance/dispersion cross terms"
        },
        extra={
            "group": group,
            "random_terms": znames,
            "covstructure": structure,
            "latent_distribution": "subject normal coefficients"
            if choice
            else "normal group effects; conditional NB2 mean dispersion",
            "intpoints_used": points,
            "quadrature_checks": checks,
            "rng": "none; deterministic product quadrature",
            "response_target": target,
            "response": [
                {"row": row, "mean": float(m), "std_error": float(s)}
                for row, m, s in zip(frame.positions, mean, se, strict=True)
            ],
            "likelihood_theta": fit.theta.tolist(),
            "likelihood_covariance": raw_v.tolist(),
            "case_columns": [group, case] if choice else None,
            "alternative": alternative if choice else None,
        },
    )


def fit_mixedlogit(spec, data):
    with torch.device("cpu"), torch.no_grad():
        return _fit(spec, data, True)


def fit_menbreg(spec, data):
    with torch.device("cpu"), torch.no_grad():
        return _fit(spec, data, False)


def mixedlogit(
    *,
    data: Any,
    y: str,
    x,
    group: str,
    case: str,
    alternative: str,
    random,
    availability: str | None = None,
    covstructure: str = "independent",
    intpoints: int = 16,
    max_intpoints: int = 64,
    max_work: int = 2_000_000_000,
    missing: str = "raise",
    alpha: float = 0.05,
):
    from openecon.analysis import fit

    columns = {
        "group": group,
        "case": case,
        "alternative": alternative,
        "random": column_list(random, "random"),
    }
    if availability:
        columns["availability"] = availability
    return fit(
        make_spec(
            "mixedlogit",
            outcome=y,
            predictors=column_list(x, "x"),
            intercept=False,
            columns=columns,
            missing=missing,
            alpha=alpha,
            options={
                "covstructure": covstructure,
                "intpoints": intpoints,
                "max_intpoints": max_intpoints,
                "max_work": max_work,
            },
        ),
        data=data,
    )


def menbreg(
    *,
    data: Any,
    y: str,
    x=None,
    group: str,
    random=None,
    covstructure: str = "independent",
    offset: str | None = None,
    exposure: str | None = None,
    intpoints: int = 16,
    max_intpoints: int = 64,
    max_work: int = 2_000_000_000,
    missing: str = "raise",
    alpha: float = 0.05,
):
    from openecon.analysis import fit

    columns = {"group": group, "random": column_list(random or [], "random")}
    columns.update(
        {key: val for key, val in {"offset": offset, "exposure": exposure}.items() if val}
    )
    return fit(
        make_spec(
            "menbreg",
            outcome=y,
            predictors=column_list(x or [], "x"),
            columns=columns,
            missing=missing,
            alpha=alpha,
            options={
                "covstructure": covstructure,
                "intpoints": intpoints,
                "max_intpoints": max_intpoints,
                "max_work": max_work,
            },
        ),
        data=data,
    )
