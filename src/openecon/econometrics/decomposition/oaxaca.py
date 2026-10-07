"""Two/three-fold decompositions with normalized categorical detail and bootstrap."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.advanced_common import ls
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec, table
from openecon.models import ResultBundle


def _normalization(design):
    """Linear map from treatment-coded slopes to sum-to-zero category effects."""
    terms = list(design.terms)
    matrix = torch.eye(len(terms), dtype=torch.float64)
    for name, record in design.categories.items():
        columns = [terms.index(f"{name}[{level}]") for level in record["levels"][1:]]
        reference = torch.zeros(matrix.shape[1], dtype=torch.float64)
        for j in columns:
            reference += matrix[j] / len(record["levels"])
        matrix[0] += reference
        for j in columns:
            matrix[j] -= reference
        matrix = torch.cat((matrix, -reference[None, :]))
        terms.append(f"{name}[{record['levels'][0]}]")
    return terms, matrix


def _means_in_normalized_basis(design, rows, weight, terms):
    x = design.x[rows]
    w = weight[rows]
    means = (x * w[:, None]).sum(0) / w.sum()
    result = means.tolist()
    for name, record in design.categories.items():
        columns = [design.terms.index(f"{name}[{level}]") for level in record["levels"][1:]]
        result.append(1 - float(means[columns].sum()))
    return torch.tensor(result, dtype=torch.float64)


def _decompose(frame, design, normalized_terms, change, rows_a, rows_b, reference, fold):
    y = frame.numeric(frame.spec.outcome)
    weight = frame.weights()
    weight = weight if weight is not None else torch.ones(frame.n, dtype=torch.float64)
    estimates = []
    for rows in (rows_a, rows_b):
        fit = ls(design.x[rows] * weight[rows].sqrt()[:, None], y[rows] * weight[rows].sqrt())
        estimates.append(fit.beta)
    a, b = (change @ coef for coef in estimates)
    ma = _means_in_normalized_basis(design, rows_a, weight, normalized_terms)
    mb = _means_in_normalized_basis(design, rows_b, weight, normalized_terms)
    dx, db = ma - mb, a - b
    if fold == 3:
        if reference not in ("a", "b"):
            raise AnalysisError(
                "invalid_reference", "Three-fold decompositions use reference='a' or 'b'."
            )
        components = [dx * b, mb * db, dx * db] if reference == "b" else [dx * a, ma * db, -dx * db]
        labels = ["endowments", "coefficients", "interaction"]
    else:
        if reference in ("a", "b"):
            benchmark = a if reference == "a" else b
        elif reference == "reimers":
            benchmark = (a + b) / 2
        elif reference == "cotton":
            benchmark = (
                len(rows_a) / (len(rows_a) + len(rows_b)) * a
                + len(rows_b) / (len(rows_a) + len(rows_b)) * b
            )
        else:
            rows = torch.cat((rows_a, rows_b))
            x = design.x[rows]
            if reference == "pooled":
                group = torch.cat((torch.ones(len(rows_a)), torch.zeros(len(rows_b)))).to(
                    torch.float64
                )
                x = torch.cat((x, group[:, None]), dim=1)
            fit = ls(x * weight[rows].sqrt()[:, None], y[rows] * weight[rows].sqrt())
            benchmark = change @ fit.beta[: design.x.shape[1]]
        components = [dx * benchmark, ma * (a - benchmark) + mb * (benchmark - b)]
        labels = ["explained", "unexplained"]
    detail = torch.stack(components)
    aggregates = detail.sum(1)
    gap = ma @ a - mb @ b
    return (
        torch.cat((aggregates, gap.reshape(1))),
        detail,
        labels,
        {
            "a_coefficients": a.tolist(),
            "b_coefficients": b.tolist(),
            "a_means": ma.tolist(),
            "b_means": mb.tolist(),
            "a_nobs": len(rows_a),
            "b_nobs": len(rows_b),
            "a_weight_sum": float(weight[rows_a].sum()),
            "b_weight_sum": float(weight[rows_b].sum()),
        },
    )


def fit_oaxaca(spec, data):
    frame = ModelFrame(spec, data)
    group = frame.role("group")[0]
    levels = frame.levels(group)
    chosen = frame.option("groups") or levels
    if (
        not isinstance(chosen, list)
        or len(chosen) != 2
        or chosen[0] == chosen[1]
        or len(levels) != 2
        or set(map(str, chosen)) != set(map(str, levels))
    ):
        raise AnalysisError(
            "invalid_groups",
            "Use exactly two observed groups in declared [a,b] order; no automatic gap/sign swapping.",
        )
    # Compare original scalar labels; stringification is not a label conversion.
    if not all(value in levels for value in chosen):
        raise AnalysisError(
            "invalid_groups", "The declared group labels must match the original scalar values."
        )
    codes, _ = frame.codes(group)
    index_a = levels.index(chosen[0])
    index_b = levels.index(chosen[1])
    rows_a, rows_b = torch.where(codes == index_a)[0], torch.where(codes == index_b)[0]
    design = frame.design()
    terms, change = _normalization(design)
    reps, seed, fold, reference = (
        frame.option(key) for key in ("reps", "seed", "fold", "reference")
    )
    frame.workspace_plan(
        "Oaxaca bootstrap and normalized detail",
        {
            "bootstrap_estimates_and_detail": 16 * reps * (4 + 4 * len(terms)),
            "parameter_covariance": 64 * len(terms) ** 2,
        },
    )
    point, detail, labels, state = _decompose(
        frame, design, terms, change, rows_a, rows_b, reference, fold
    )
    generator = torch.Generator().manual_seed(seed)
    estimates, details = [], []
    for repetition in range(reps):
        ia = rows_a[torch.randint(len(rows_a), (len(rows_a),), generator=generator)]
        ib = rows_b[torch.randint(len(rows_b), (len(rows_b),), generator=generator)]
        try:
            aggregate, components, _, _ = _decompose(
                frame, design, terms, change, ia, ib, reference, fold
            )
        except AnalysisError as error:
            raise AnalysisError(
                "degenerate_bootstrap_draw",
                f"Oaxaca bootstrap draw {repetition} failed ({error.code}); no silent deletion/resampling of unsuccessful draws.",
            ) from error
        estimates.append(aggregate)
        details.append(components)
    estimates, details = torch.stack(estimates), torch.stack(details)
    centered = estimates - estimates.mean(0)
    covariance = centered.T @ centered / (reps - 1)
    detail_se = details.std(0, unbiased=True)
    interval = torch.quantile(
        details, torch.tensor([spec.alpha / 2, 1 - spec.alpha / 2], dtype=torch.float64), dim=0
    )
    rows = [
        [
            labels[i],
            term,
            float(detail[i, j]),
            float(detail_se[i, j]),
            float(interval[0, i, j]),
            float(interval[1, i, j]),
        ]
        for i in range(len(labels))
        for j, term in enumerate(terms)
    ]
    state.update(
        group_column=group,
        group_order=chosen,
        reference=reference,
        fold=fold,
        estimand="weighted mean outcome of group a minus group b; descriptive decomposition",
        categorical_normalization="sum-to-zero full set of category effects; intercept shifted by category average",
        normalized_terms=terms,
        detail=rows,
        detail_columns=["component", "term", "estimate", "std_error", "ci_low", "ci_high"],
        uncertainty="stratified pairs bootstrap; fixed observed sample counts in each group; normal aggregate intervals and percentile detail intervals",
        seed=seed,
        reps=reps,
        successful_draws=reps,
        weights="analytic weights retained with each resampled row"
        if spec.weights
        else "equal rows",
        reference_population_share="observed group counts" if reference == "cotton" else None,
    )
    return build_result(
        frame,
        terms=[*labels, "gap"],
        params=point,
        covariance=covariance,
        use_t=False,
        solver="float64_weighted_group_QR_and_stratified_bootstrap",
        categories=design.categories,
        extra=state,
        inference={"correction": "stratified pairs bootstrap; all aggregate covariance retained"},
    )


def oaxaca(
    *,
    data,
    y,
    x,
    group,
    groups=None,
    fold=2,
    reference="pooled",
    categorical=None,
    weights=None,
    reps=200,
    seed=0,
    missing="raise",
    alpha=0.05,
):
    """Two/three-fold Oaxaca-Blinder; group order and reference are never inferred from gap sign."""
    from openecon.analysis import fit

    return fit(
        make_spec(
            "oaxaca",
            outcome=y,
            predictors=column_list(x, "x"),
            columns={"group": group},
            categorical=column_list(categorical, "categorical"),
            weights=weights,
            weight_type="aweight",
            covariance="bootstrap",
            missing=missing,
            alpha=alpha,
            options={
                "groups": groups,
                "fold": fold,
                "reference": reference,
                "reps": reps,
                "seed": seed,
            },
        ),
        data=data,
    )


def oaxaca_details(result):
    """Detailed normalized components and their declared bootstrap uncertainty."""
    if not isinstance(result, ResultBundle) or result.spec.estimator != "oaxaca":
        raise AnalysisError("invalid_result", "oaxaca_details needs a saved Oaxaca-Blinder result.")
    return table(
        result.extra["detail"],
        columns=result.extra["detail_columns"],
        reference=result.extra["reference"],
        group_order=result.extra["group_order"],
        categorical_normalization=result.extra["categorical_normalization"],
        uncertainty=result.extra["uncertainty"],
    )
