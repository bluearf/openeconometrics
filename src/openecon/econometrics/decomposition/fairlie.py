"""Explicit binary-outcome Fairlie rank-matched decomposition."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.econometrics.decomposition.mediation import _equation


def fit_fairlie(spec, data):
    frame = ModelFrame(spec, data)
    design = frame.design()
    x, y = design.x, frame.numeric(spec.outcome)
    if not bool(((y == 0) | (y == 1)).all()):
        raise AnalysisError("invalid_outcome", "Fairlie decomposition needs binary 0/1 outcomes.")
    codes, count = frame.codes(frame.role("group")[0])
    groups = frame.option("groups")
    labels = frame.sample[frame.role("group")[0]].tolist()
    if (
        count != 2
        or not isinstance(groups, list)
        or len(groups) != 2
        or groups[0] == groups[1]
        or set(groups) != set(labels)
    ):
        raise AnalysisError(
            "invalid_groups", "Declare the exact two ordered group labels in groups=[A,B]."
        )
    # Use declared label ordering, independently of factorization/input order.
    group_codes = torch.tensor(
        [0 if label == groups[0] else 1 for label in labels], dtype=torch.int64
    )
    members = [torch.where(group_codes == i)[0] for i in (0, 1)]
    reps, matching = frame.option("reps"), frame.option("matching_reps")
    k = x.shape[1]
    if min(len(m) for m in members) <= k + 1:
        raise AnalysisError(
            "insufficient_observations",
            "Each Fairlie group must have more rows than design width plus one.",
        )
    work = (reps + 1) * matching * min(len(m) for m in members) * k * k
    if work > frame.option("max_work") or k > 24:
        raise AnalysisError(
            "work_budget_exceeded", "Fairlie decomposition exceeds its matching/fit budget."
        )
    frame.workspace_plan(
        "Fairlie resampling and rank matching",
        {"designs": len(y) * k * 40, "replicates": (reps + 1) * (k + 3) * 32},
    )
    generator = torch.Generator(device="cpu").manual_seed(frame.option("seed"))
    kind = frame.option("link")
    reference = frame.option("reference")
    probability = torch.sigmoid if kind == "logit" else torch.special.ndtr

    def compute(rows):
        xx, yy = x[rows], y[rows]
        gg = group_codes[rows]
        ref = (
            torch.ones(len(rows), dtype=torch.bool)
            if reference == "pooled"
            else gg == (0 if reference == "a" else 1)
        )
        beta, influence, _ = _equation(xx[ref], yy[ref], None, kind)
        units = [torch.where(gg == i)[0] for i in (0, 1)]
        predictions = probability(xx @ beta)
        gap = yy[units[0]].mean() - yy[units[1]].mean()
        full_explained = predictions[units[0]].mean() - predictions[units[1]].mean()
        detail = []
        for _ in range(matching):
            paired = []
            size = min(len(u) for u in units)
            for u in units:
                selected = (
                    u[torch.randperm(len(u), generator=generator)[:size]] if len(u) > size else u
                )
                order = torch.argsort(predictions[selected], stable=True)
                paired.append(xx[selected[order]].clone())
            a, b = paired
            running = b.clone()
            previous = probability(running @ beta).mean()
            contributions = beta.new_zeros(k - 1)
            # Randomized switching order removes any single arbitrary path choice.
            for column in torch.randperm(k - 1, generator=generator).tolist():
                running[:, column + 1] = a[:, column + 1]
                current = probability(running @ beta).mean()
                contributions[column] = current - previous
                previous = current
            detail.append(contributions)
        contributions = torch.stack(detail).mean(0)
        # Report exact full-group explained mean; preserve the finite matching
        # Monte Carlo remainder explicitly rather than rescaling contributions.
        remainder = full_explained - contributions.sum()
        values = torch.cat(
            (
                torch.stack((gap, full_explained, gap - full_explained)),
                contributions,
                remainder[None],
            )
        )
        return values, beta, influence.T @ influence

    positions = torch.arange(frame.n)
    values, beta, model_covariance = compute(positions)
    draws = []
    first_positions = None
    for replication in range(reps):
        selected = torch.cat(
            [u[torch.randint(len(u), (len(u),), generator=generator)] for u in members]
        )
        if replication == 0:
            first_positions = selected.tolist()
        try:
            draws.append(compute(selected)[0])
        except AnalysisError as exc:
            raise AnalysisError(
                "bootstrap_failure", f"Fairlie replicate {replication} failed; no draw is omitted."
            ) from exc
    simulated = torch.stack(draws)
    centered = simulated - simulated.mean(0)
    v = centered.T @ centered / (reps - 1)
    # Equal-size rank matching makes its remainder exactly zero. It is metadata,
    # not a coefficient with a fictional standard error.
    report = list(range(len(values) - 1))
    terms = [
        "gap",
        "explained",
        "unexplained",
        *[f"contribution:{name}" for name in design.terms[1:]],
    ]
    quantiles = torch.quantile(
        simulated[:, report],
        torch.tensor([spec.alpha / 2, 1 - spec.alpha / 2], dtype=torch.float64),
        dim=0,
    )
    return build_result(
        frame,
        terms=terms,
        params=values[report],
        covariance=v[report][:, report],
        title="Fairlie binary-outcome decomposition",
        use_t=False,
        solver="native binary ML, probability-rank matching and stratified-pairs bootstrap",
        inference={
            "covariance": "bootstrap",
            "correction": "sample covariance across all declared bootstrap draws",
            "distribution": "normal",
            "df_inference": None,
            "pointwise_percentile_intervals": "stored in extra; normal intervals in coefficient table",
        },
        extra={
            "groups": groups,
            "link": kind,
            "reference": reference,
            "seed": frame.option("seed"),
            "reps": reps,
            "matching_reps": matching,
            "matching_mc_remainder": float(values[-1]),
            "matching_remainder_bootstrap": simulated[:, -1].tolist(),
            "group_sizes": [len(u) for u in members],
            "reference_terms": design.terms,
            "reference_parameters": beta.tolist(),
            "reference_covariance": model_covariance.tolist(),
            "bootstrap_percentile_intervals": quantiles.T.tolist(),
            "bootstrap_estimates": simulated[:, report].tolist(),
            "bootstrap_first_positions": first_positions,
            "reference_covariance_kind": "HC0 outer product of binary-ML influence rows",
            "conventions": "A-B; unweighted; pooled reference has no group dummy; stable probability ranks; random larger-group subsamples and randomized switching order",
            "identities": "gap=explained+unexplained; explained=sum(contributions)+matching_mc_remainder",
        },
        warnings=[
            "This is a descriptive counterfactual decomposition, not a causal effect. Matching Monte Carlo remainder is reported explicitly; weights/categorical expansions are outside this domain."
        ],
    )


def fairlie(
    *,
    data,
    y,
    x,
    group,
    groups,
    link="logit",
    reference="pooled",
    reps=99,
    matching_reps=20,
    seed=0,
    max_work=100_000_000,
    missing="raise",
    alpha=0.05,
):
    """Fairlie binary logit/probit decomposition with ordered groups and rank matching.

    Full-group explained differences and per-variable contributions are separate
    from the finite matching remainder. Stratified-pairs bootstrap refits ML and
    matching every time; stores full joint covariance, percentile intervals and
    seeds. Unweighted numeric resident domain, without a causal interpretation.
    """
    from openecon.analysis import fit

    return fit(
        make_spec(
            "fairlie",
            outcome=y,
            predictors=column_list(x, "x"),
            columns={"group": group},
            covariance="bootstrap",
            missing=missing,
            alpha=alpha,
            options={
                "groups": groups,
                "link": link,
                "reference": reference,
                "reps": reps,
                "matching_reps": matching_reps,
                "seed": seed,
                "max_work": max_work,
            },
        ),
        data=data,
    )
