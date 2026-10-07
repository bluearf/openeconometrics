"""HHG panel negative binomial and joint panel stochastic-frontier likelihoods."""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.econometrics.mixed.extended_common import (
    Objective,
    maximize,
    optimizer_record,
    transformed,
)
from openecon.engines.linalg import least_squares


def _sample(spec, data):
    frame = ModelFrame(spec, data)
    if spec.time:
        frame.sort_panel()
    codes, groups = frame.codes(spec.panel)
    if groups < 2:
        raise AnalysisError("insufficient_groups", "At least two panels are required.")
    design = frame.design()
    if not design.terms:
        raise AnalysisError("invalid_spec", "At least one design term is required.")
    # Reject rather than silently changing the likelihood parameterization.
    from openecon.engines.linalg import collinear_columns

    if collinear_columns(design.x)[1]:
        raise AnalysisError("rank_deficient", "The fixed-effect design must have full rank.")
    y = frame.numeric(spec.outcome)
    frame.workspace_plan(
        "panel likelihood and full information",
        {
            "design_derivative_buffers": 128 * frame.n * (len(design.terms) + 5) ** 2,
            "panel_moments": 128 * groups * (len(design.terms) + 5) ** 2,
        },
    )
    return frame, design, y, codes, groups


def _sum(values, codes, groups):
    return torch.zeros(groups, dtype=torch.float64).index_add(0, codes, values)


def fit_xtnbreg(spec, data):
    with torch.device("cpu"), torch.no_grad():
        frame, design, y, codes, groups = _sample(spec, data)
        if bool((y < 0).any()) or bool((y != y.round()).any()) or not bool((y > 0).any()):
            raise AnalysisError(
                "invalid_count_outcome", "Panel NB needs varying nonnegative integer counts."
            )
        fe = frame.option("model") == "fe"
        totals = _sum(y, codes, groups)
        if fe and bool((totals == 0).any()):
            frame.restrict(
                (totals[codes] > 0).numpy(),
                reason="all-zero panels have no conditional count likelihood",
            )
            design = frame.design()
            y = frame.numeric(spec.outcome)
            codes, groups = frame.codes(spec.panel)
            totals = _sum(y, codes, groups)
        counts = torch.bincount(codes, minlength=groups)
        if groups < 2 or int(counts.min()) < 2:
            raise AnalysisError(
                "insufficient_group_size",
                "HHG panel NB needs at least two observations in every retained panel.",
            )
        offset = torch.zeros(frame.n, dtype=torch.float64)
        if frame.role("offset"):
            offset += frame.numeric(frame.role("offset")[0])
        if frame.role("exposure"):
            exposure = frame.numeric(frame.role("exposure")[0])
            if bool((exposure <= 0).any()):
                raise AnalysisError("invalid_exposure", "Exposure must be positive.")
            offset += exposure.log()
        x, p = design.x, len(design.terms)

        def value(theta):
            lam = (x @ theta[:p] + offset).exp()
            total_lam = _sum(lam, codes, groups)
            row = (torch.lgamma(lam + y) - torch.lgamma(lam) - torch.lgamma(y + 1)).sum()
            if fe:
                panel = (
                    torch.lgamma(total_lam)
                    + torch.lgamma(totals + 1)
                    - torch.lgamma(total_lam + totals)
                )
            else:
                r, s = theta[p:].exp().unbind()
                panel = (
                    torch.lgamma(r + s)
                    - torch.lgamma(r)
                    - torch.lgamma(s)
                    + torch.lgamma(r + total_lam)
                    + torch.lgamma(s + totals)
                    - torch.lgamma(r + s + total_lam + totals)
                )
            return row + panel.sum()

        objective = Objective(value, 8 * frame.n * (p + 2) ** 2, frame.option("max_work"))
        initial = least_squares(x, torch.log(y + 0.5) - offset, drop_collinear=False).beta
        starts = []
        for concentration in (1.0, 10.0, 100.0):
            starts.append(
                initial
                if fe
                else torch.cat(
                    (
                        initial,
                        torch.tensor(
                            [math.log(concentration), math.log(concentration)], dtype=torch.float64
                        ),
                    )
                )
            )
        fit, vc = maximize(objective, starts)
        if not fe and bool((fit.theta[p:].abs() > 20).any()):
            raise AnalysisError(
                "boundary_solution",
                "HHG beta concentration is outside the validated interior domain.",
            )
        return build_result(
            frame,
            terms=[*design.terms, *([] if fe else ["/ln_r", "/ln_s"])],
            params=fit.theta,
            covariance=vc,
            use_t=False,
            metrics={"log_likelihood": fit.value, "n_groups": groups},
            solver="native HHG closed-form group likelihood",
            optimizer=optimizer_record(fit, objective),
            inference={
                "correction": "full observed information, including regression/concentration cross terms"
            },
            extra={
                "model": "hhg_fe" if fe else "hhg_re",
                "group": spec.panel,
                "latent_distribution": "condition on each panel total"
                if fe
                else "1/(1+dispersion) ~ Beta(r,s)",
                "fixed_effect_semantics": "FE conditions out group dispersion; it does not remove an additive log-mean intercept",
                "r": None if fe else float(fit.theta[p].exp()),
                "s": None if fe else float(fit.theta[p + 1].exp()),
                "population_mean_exists": None if fe else bool(fit.theta[p].exp() > 1),
                "offset": frame.role("offset"),
                "exposure": frame.role("exposure"),
            },
        )


def fit_xtfrontier(spec, data):
    with torch.device("cpu"), torch.no_grad():
        frame, design, y, codes, groups = _sample(spec, data)
        x, p = design.x, len(design.terms)
        sizes = torch.bincount(codes, minlength=groups).to(torch.float64)
        if int(sizes.min()) < 2:
            raise AnalysisError(
                "insufficient_group_size",
                "Joint panel inefficiency needs at least two observations per firm.",
            )
        sign = 1.0 if frame.option("cost") else -1.0
        normal = frame.option("distribution") == "truncated_normal"
        varying = frame.option("time_varying")
        if varying and not spec.time:
            raise AnalysisError(
                "missing_time", "Time-varying inefficiency requires a declared calendar."
            )
        age = torch.zeros(frame.n, dtype=torch.float64)
        if spec.time:
            time = frame.numeric(spec.time)
            last = torch.full((groups,), -math.inf, dtype=torch.float64).scatter_reduce(
                0, codes, time, "amax"
            )
            age = time - last[codes]

        def moments(theta):
            mu = theta[p] if normal else torch.zeros((), dtype=torch.float64)
            start = p + int(normal)
            su, sv = theta[start : start + 2].exp().unbind()
            residual = y - x @ theta[:p]
            weights = torch.exp(-theta[-1] * age) if varying else torch.ones_like(y)
            a = 1 / su.square() + _sum(weights.square(), codes, groups) / sv.square()
            b = mu / su.square() + sign * _sum(weights * residual, codes, groups) / sv.square()
            return mu, su, sv, residual, weights, a, b

        def value(theta):
            mu, su, sv, residual, weights, a, b = moments(theta)
            panel = (
                -sizes / 2 * math.log(2 * math.pi)
                - sizes * sv.log()
                - su.log()
                - 0.5 * a.log()
                - 0.5 * _sum(residual.square(), codes, groups) / sv.square()
                - 0.5 * mu.square() / su.square()
                + 0.5 * b.square() / a
                + torch.special.log_ndtr(b / a.sqrt())
                - torch.special.log_ndtr(mu / su)
            )
            return panel.sum()

        objective = Objective(value, 12 * frame.n * (p + 4) ** 2, frame.option("max_work"))
        beta = least_squares(x, y, drop_collinear=False).beta
        sd = float((y - x @ beta).square().mean().sqrt())
        if sd < 1e-8:
            raise AnalysisError("exact_fit", "Panel frontier needs nonzero residual variation.")
        starts = []
        for ratio in (0.3, 1.0, 2.0):
            shifted = beta.clone()
            if "Intercept" in design.terms:
                shifted[design.terms.index("Intercept")] -= sign * sd * ratio
            values = [
                *([sd * ratio] if normal else []),
                math.log(sd * ratio),
                math.log(sd * 0.7),
                *([0.0] if varying else []),
            ]
            starts.append(torch.cat((shifted, torch.tensor(values, dtype=torch.float64))))
        fit, raw_v = maximize(objective, starts)
        mu, su, sv, _, weights, a, b = moments(fit.theta)
        if min(float(su), float(sv)) < 1e-4 or not bool(torch.isfinite(weights).all()):
            raise AnalysisError(
                "boundary_solution",
                "Panel frontier has a zero variance or nonfinite time multiplier.",
            )

        def report(theta):
            start = p + int(normal)
            logs = theta[start : start + 2]
            return torch.cat(
                (
                    theta[:start],
                    torch.stack((torch.logsumexp(2 * logs, dim=0), 2 * (logs[0] - logs[1]))),
                    theta[-1:] if varying else theta[:0],
                )
            )

        reported, vc = transformed(fit.theta, raw_v, report)
        terms = [
            *design.terms,
            *(["/mu"] if normal else []),
            "/lnsigma2",
            "/lgtgamma",
            *(["/eta"] if varying else []),
        ]

        def efficiency(theta):
            _, _, _, _, w, aa, bb = moments(theta)
            m, s = bb / aa, aa.rsqrt()
            z = m / s
            return torch.exp(
                -w * m[codes]
                + 0.5 * w.square() * s[codes].square()
                + torch.special.log_ndtr(z[codes] - w * s[codes])
                - torch.special.log_ndtr(z[codes])
            )

        # Never construct an N x N prediction covariance.
        with torch.enable_grad():
            jac = torch.autograd.functional.jacobian(efficiency, fit.theta)
        eff = efficiency(fit.theta)
        eff_se = ((jac @ raw_v) * jac).sum(1).clamp_min(0).sqrt()
        m, s = b / a, a.rsqrt()
        log_ratio = (
            -0.5 * (m / s).square() - 0.5 * math.log(2 * math.pi) - torch.special.log_ndtr(m / s)
        )
        inefficiency = weights * (m + s * log_ratio.exp())[codes]
        return build_result(
            frame,
            terms=terms,
            params=reported,
            covariance=vc,
            use_t=False,
            metrics={
                "log_likelihood": fit.value,
                "n_groups": groups,
                "sigma_u2": float(su.square()),
                "sigma_v2": float(sv.square()),
            },
            solver="native joint firm likelihood; truncated-normal posterior",
            optimizer=optimizer_record(fit, objective),
            inference={
                "correction": "full observed information and full-parameter delta efficiency uncertainty"
            },
            extra={
                "distribution": frame.option("distribution"),
                "inefficiency": "Battese-Coelli exponential calendar multiplier"
                if varying
                else "shared time-invariant firm effect",
                "cost": frame.option("cost"),
                "calendar_column": spec.time,
                "efficiency_target": "E[exp(-u_it) | complete firm residual vector]",
                "efficiency": [
                    {
                        "row": row,
                        "mean": float(e),
                        "std_error": float(se),
                        "inefficiency_mean": float(u),
                    }
                    for row, e, se, u in zip(
                        frame.positions, eff, eff_se, inefficiency, strict=True
                    )
                ],
                "likelihood_theta": fit.theta.tolist(),
                "likelihood_covariance": raw_v.tolist(),
            },
        )


def xtnbreg(
    *,
    data: Any,
    y: str,
    x=None,
    panel: str,
    time: str | None = None,
    model: str = "re",
    offset: str | None = None,
    exposure: str | None = None,
    missing: str = "raise",
    max_work: int = 2_000_000_000,
    alpha: float = 0.05,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            "xtnbreg",
            outcome=y,
            predictors=column_list(x or [], "x"),
            panel=panel,
            time=time,
            columns={
                key: val for key, val in {"offset": offset, "exposure": exposure}.items() if val
            },
            missing=missing,
            alpha=alpha,
            options={"model": model, "max_work": max_work},
        ),
        data=data,
    )


def xtfrontier(
    *,
    data: Any,
    y: str,
    x,
    panel: str,
    time: str | None = None,
    distribution: str = "truncated_normal",
    time_varying: bool = False,
    cost: bool = False,
    missing: str = "raise",
    max_work: int = 2_000_000_000,
    alpha: float = 0.05,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            "xtfrontier",
            outcome=y,
            predictors=column_list(x, "x"),
            panel=panel,
            time=time,
            missing=missing,
            alpha=alpha,
            options={
                "distribution": distribution,
                "time_varying": time_varying,
                "cost": cost,
                "max_work": max_work,
            },
        ),
        data=data,
    )
