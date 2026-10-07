"""Resident crossed or arbitrary-depth nested Gaussian ML, with full information."""

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
    transformed,
)
from openecon.engines.linalg import collinear_columns, least_squares


def fit_mixedflex(spec, data):
    with torch.device("cpu"), torch.no_grad():
        frame = ModelFrame(spec, data)
        groups, random = frame.role("group"), frame.role("random")
        if not 1 <= len(groups) <= 8 or len(random) > 2:
            raise AnalysisError(
                "invalid_spec",
                "mixedflex supports one to eight grouping levels and at most two lowest-level random slopes.",
            )
        if len(set(groups)) != len(groups) or set(groups) & {
            spec.outcome,
            *spec.predictors,
            *random,
        }:
            raise AnalysisError(
                "invalid_spec",
                "Grouping columns must be distinct from one another and the numeric model columns.",
            )
        design = frame.design()
        x, y = design.x, frame.numeric(spec.outcome)
        if collinear_columns(x)[1]:
            raise AnalysisError("rank_deficient", "Fixed-effect design must have full rank.")
        p, n = x.shape[1], frame.n
        structure = frame.option("covstructure")
        z_names = [*random, "_cons"]
        q = len(z_names)
        size = factor_size(q, structure)
        k = p + size + len(groups)
        # Full dense likelihood is deliberately bounded; no hidden sparse fallback.
        frame.workspace_plan(
            "dense crossed/nested joint Gaussian likelihood",
            {
                "covariance_factors_and_derivative_tape": 96 * n * n * (k + 3),
                "group_masks": 16 * n * n * len(groups),
                "information": 64 * k * k,
            },
        )
        if n <= k or float(y.std()) < 1e-8:
            raise AnalysisError(
                "insufficient_observations",
                "Need a varying outcome and more observations than parameters.",
            )
        nested = frame.option("grouping") == "nested"
        levels, masks, labels = [], [], []
        for depth, group in enumerate(groups):
            names = groups[: depth + 1] if nested else [group]
            codes, count = frame.codes(names)
            if count < 2:
                raise AnalysisError(
                    "insufficient_groups", "Every random grouping factor needs at least two levels."
                )
            masks.append((codes[:, None] == codes[None, :]).to(torch.float64))
            levels.append({"columns": names, "n_groups": count})
            labels.append(frame.sample[names].drop_duplicates().to_dict("records"))
        z = torch.stack(
            [*[frame.numeric(name) for name in random], torch.ones(n, dtype=torch.float64)], dim=1
        )
        if collinear_columns(z)[1] or int(torch.diagonal(masks[-1]).sum()) == 0:
            raise AnalysisError("rank_deficient", "Random-effect columns must be independent.")
        if bool((masks[-1].sum(1) == 1).all()):
            raise AnalysisError(
                "insufficient_group_size",
                "Lowest grouping level has one observation per group; its intercept variance is confounded with residual variance.",
            )
        identity = torch.eye(n, dtype=torch.float64)

        def matrix(theta):
            lower = factor(theta[p : p + size], q, structure)
            covariance = theta[-1].mul(2).exp() * identity + masks[-1] * (z @ lower @ lower.T @ z.T)
            for j, mask in enumerate(masks[:-1]):
                covariance = covariance + theta[p + size + j].mul(2).exp() * mask
            return covariance

        def value(theta):
            covariance = matrix(theta)
            chol = torch.linalg.cholesky(covariance)
            residual = y - x @ theta[:p]
            solve = torch.cholesky_solve(residual[:, None], chol)[:, 0]
            return -0.5 * (
                n * math.log(2 * math.pi) + 2 * chol.diagonal().log().sum() + residual @ solve
            )

        beta = least_squares(x, y, drop_collinear=False).beta
        sd = float((y - x @ beta).square().mean().sqrt())
        starts = [
            torch.cat(
                (
                    beta,
                    factor_start(q, structure, sd * fraction),
                    torch.full((len(groups),), math.log(sd * fraction), dtype=torch.float64),
                )
            )
            for fraction in (0.25, 0.6, 1.0)
        ]
        objective = Objective(value, n**3 + 4 * n * n * k, frame.option("max_work"))
        fit, raw_v = maximize(objective, starts)
        check_factor(fit.theta[p : p + size], q, structure)
        residual_var = float(fit.theta[-1].mul(2).exp())
        if residual_var < 1e-8 or bool(
            (fit.theta[p + size : -1].mul(2).exp() / residual_var < 1e-8).any()
        ):
            raise AnalysisError(
                "boundary_solution", "A Gaussian variance component is at the boundary."
            )

        def report(theta):
            return torch.cat(
                (
                    theta[:p],
                    covariance_report(theta[p : p + size], q, structure),
                    theta[p + size :].mul(2).exp(),
                )
            )

        parameters, vc = transformed(fit.theta, raw_v, report)
        terms = [*design.terms, *[f"/var({name}[{groups[-1]}])" for name in z_names]]
        if structure == "unstructured":
            terms += [
                f"/cov({z_names[i]},{z_names[j]}[{groups[-1]}])" for i in range(q) for j in range(i)
            ]
        terms += [*[f"/var(_cons[{name}])" for name in groups[:-1]], "/var(Residual)"]
        return build_result(
            frame,
            terms=terms,
            params=parameters,
            covariance=vc,
            use_t=False,
            metrics={"log_likelihood": fit.value},
            solver="native dense Gaussian joint ML",
            optimizer=optimizer_record(fit, objective),
            inference={
                "correction": "full joint observed information; fixed/variance cross terms retained"
            },
            extra={
                "grouping": frame.option("grouping"),
                "levels": levels,
                "group_labels": labels,
                "covstructure": structure,
                "random_terms": z_names,
                "method": "ml",
                "likelihood_theta": fit.theta.tolist(),
                "likelihood_covariance": raw_v.tolist(),
                "prediction_domain": "saved general crossed BLUP adapter is separate; no zero-effect fallback",
            },
        )


def mixedflex(
    *,
    data: Any,
    y: str,
    x=None,
    group,
    random=None,
    grouping: str = "crossed",
    covstructure: str = "independent",
    missing: str = "raise",
    max_work: int = 2_000_000_000,
    alpha: float = 0.05,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            "mixedflex",
            outcome=y,
            predictors=column_list(x or [], "x"),
            columns={
                "group": column_list([group] if isinstance(group, str) else group, "group"),
                "random": column_list(random or [], "random"),
            },
            missing=missing,
            alpha=alpha,
            options={"grouping": grouping, "covstructure": covstructure, "max_work": max_work},
        ),
        data=data,
    )
