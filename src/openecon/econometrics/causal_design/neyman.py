"""Finite-population ATEs under four explicitly declared assignment designs.

Neyman variance estimates target conservative bounds, not the unidentified true
randomization covariance. Normal inference is asymptotic and concerns a weak
average-effect null, not a Fisher constant-effect sharp null.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import common as c
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.resources import plan_workspace

from . import common as m
from .observational_survival import _SUM_CAPACITY, _divide, _finite, _multiply, _sum_rows
from .paired import _pair_key

_TERMS = ["mean_control", "mean_treated", "difference"]


def _label_bytes(value):
    value = m._label(value)
    if isinstance(value, str):
        if len(value) > 1024 or len(value.encode("utf-8")) > 1024:
            raise AnalysisError(
                "resource_limit",
                "Neyman identifier/index/column labels are limited to1024 UTF-8 bytes.",
            )
    elif isinstance(value, int) and value.bit_length() > 3400:
        raise AnalysisError(
            "resource_limit", "A Neyman numeric label exceeds the1024-byte textual domain."
        )
    encoded = m.canonical(value)
    if len(encoded) > 8192:
        raise AnalysisError(
            "resource_limit", "A Neyman serialized label exceeds the declared textual domain."
        )
    return len(encoded)


def _add(left, right, label):
    value = _finite(left + right, label)
    if bool(((left != 0) & (right != 0) & ((value == left) | (value == right))).any()):
        raise AnalysisError("numerical_failure", f"A nonzero {label} contribution is absorbed.")
    return value


def _difference(left, right, label):
    return _add(left, -right, label)


def _anchored(values):
    """Use a common small location; fall back to zero if subtraction loses it.

    Constant columns retain their exact location, while widely separated scales
    remain in the expansion rather than losing a small term to preprocessing.
    """
    columns = torch.arange(values.shape[1], device="cpu")
    anchor = values[values.abs().argmin(0), columns]
    residuals = _finite(values - anchor, "anchored outcomes")
    absorbed = (
        (values != 0) & (anchor != 0) & ((residuals == values) | (residuals == -anchor))
    ).any(0)
    anchor = torch.where(absorbed, torch.zeros_like(anchor), anchor)
    return anchor, _finite(values - anchor, "anchored outcomes")


def _prepare(data, y, treatment, identity, *, design, required, missing, max_work):
    if not isinstance(design, str) or design != required:
        raise AnalysisError("unsupported_design", f"Explicitly declare design='{required}'.")
    if missing != "raise":
        raise AnalysisError(
            "unsupported_missing",
            "Neyman design inference requires missing='raise'; deleting outcomes changes the fixed population and assignment topology.",
        )
    names = [c.check_name(y, "y"), c.check_name(treatment, "treatment")]
    if identity is not None:
        names.append(c.check_name(identity, "design identifier"))
    if len(set(names)) != len(names):
        raise AnalysisError(
            "invalid_spec", "Outcome, treatment and design identifier must be distinct columns."
        )
    for name in names:
        _label_bytes(name)
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset",
            "Neyman design inference requires a resident table, not Dataset replay.",
        )
    source = c.source(data)
    n = len(source)
    if n > 100000:
        raise AnalysisError("resource_limit", "Neyman input is limited to100000 original rows.")
    absent = [name for name in names if name not in source]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {absent}.")
    label_bytes = sum(_label_bytes(value) for value in source.index)
    if identity is not None:
        label_bytes += sum(_label_bytes(value) for value in source[identity])
    label_bytes += sum(_label_bytes(name) for name in names)
    # Includes worst-case bounded expansion arithmetic for aggregation, means,
    # all nine covariance entries, group summaries and complete saved state.
    cost = 128 + 48 * _SUM_CAPACITY * 12
    planned = n * cost + 4096 + 32 * label_bytes
    m.work(
        planned,
        max_work,
        "complete original topology, bounded means, covariance bound and saved state",
    )
    plan = plan_workspace(
        "Neyman complete design and inference",
        {
            "selected_inputs_original_topology_and_provenance": 2048 * n * len(names),
            "all_group_unit_outcomes_and_centered_covariance_products": 4096 * n,
            "complete_tables_and_typed_scientific_state": 4096 * n,
            "full_floating_expansion_buffers": 128 * _SUM_CAPACITY * 12,
            "escaped_labels_full_metadata_tables_and_json": 16 * label_bytes,
        },
    )
    # Validate assignment/identity before even selecting the outcome column.
    topology, topology_metadata = m.sample(
        source,
        names[1:],
        numeric=[treatment],
        missing="raise",
        max_work=max_work,
        cost=cost,
        minimum=4,
    )
    assignment = m.binary(topology, treatment)
    groups = []
    if identity is None:
        groups = [dict(kind=None, label=None, positions=list(range(n)))]
    else:
        found = {}
        for position, value in enumerate(topology[identity]):
            try:
                key, label = _pair_key(value)
            except AnalysisError as exc:
                raise AnalysisError(
                    "invalid_design_identifier",
                    "Design identifiers must be nonmissing finite typed numeric, boolean or text scalars.",
                ) from exc
            found.setdefault(key, [label, []])[1].append(position)
        groups = [
            dict(kind=key[0], label=label, positions=positions)
            for key, (label, positions) in found.items()
        ]
    for group in groups:
        positions = group["positions"]
        count = int(assignment[positions].sum())
        group.update(n=len(positions), n_treated=count, n_control=len(positions) - count)
        if required == "paired_randomized":
            if len(positions) != 2 or count != 1:
                raise AnalysisError(
                    "invalid_pair",
                    "Every original pair must have exactly two rows, one treated and one control.",
                )
        elif required == "cluster_randomized":
            if count not in (0, len(positions)):
                raise AnalysisError(
                    "invalid_cluster", "Treatment must be constant within every original cluster."
                )
            group["assignment"] = int(count > 0)
        elif min(count, len(positions) - count) < 2:
            raise AnalysisError(
                "insufficient_observations",
                "Every original complete/stratified population requires at least two units per arm.",
            )
    if (
        required == "cluster_randomized"
        and min(sum(group["assignment"] == arm for group in groups) for arm in (0, 1)) < 2
    ):
        raise AnalysisError(
            "insufficient_clusters", "At least two original clusters per arm are required."
        )
    if required == "paired_randomized" and len(groups) < 2:
        raise AnalysisError("insufficient_pairs", "At least two original pairs are required.")
    selected, metadata = m.sample(
        source,
        names,
        numeric=[y, treatment],
        missing="raise",
        max_work=max_work,
        cost=cost,
        minimum=4,
    )
    metadata.update(
        design_sample_sha256=topology_metadata["sample_sha256"],
        computation_resource_plan=plan.record(),
        planned_work=planned,
        population="all original supplied units, fixed independently of assignment",
        n_groups=len(groups),
        original_escaped_label_bytes=label_bytes,
    )
    return m.tensor(selected, y), assignment, groups, metadata


def _moments(values):
    """Mean and unbiased sample covariance, with all signed products retained."""
    n = len(values)
    anchor, residuals = _anchored(values)
    residual_means = _divide(_sum_rows(residuals), n, "anchored design-unit mean")
    means = _add(anchor, residual_means, "design-unit mean location")
    centered = _finite(residuals - residual_means, "centered design-unit outcomes")
    scales = centered.abs().max(0).values
    divisors = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = _divide(centered, divisors, "design-unit covariance normalization")
    products = _multiply(
        normalized[:, :, None], normalized[:, None, :], "normalized covariance product"
    )
    covariance = _divide(
        _sum_rows(products.reshape(n, -1)).reshape(values.shape[1], values.shape[1]),
        n - 1,
        "sample covariance normalization",
    )
    large = torch.maximum(scales[:, None], scales[None, :])
    small = torch.minimum(scales[:, None], scales[None, :])
    covariance = _multiply(
        _multiply(covariance, large, "sample covariance rescaling"), small, "sample covariance"
    )
    if bool(((covariance.diag() == 0) & (centered != 0).any(0)).any()):
        raise AnalysisError(
            "numerical_failure", "A nonconstant design-unit variance rounded to zero."
        )
    return means, covariance, centered


def _independent_arms(values, assignment):
    variances, centered = [], []
    for arm in (0, 1):
        positions = assignment == arm
        _, covariance, residuals = _moments(values[positions, None])
        variances.append(_divide(covariance[0, 0], int(positions.sum()), "arm mean variance bound"))
        centered.append(residuals[:, 0].tolist())
    # Saved group points must use the same original-row location calculation as
    # public points. Raw Y/n contributions can invent a difference for constant
    # outcomes when the two arm counts differ, even with compensated summation.
    estimates, scores, anchor, coefficients, location = _original_contributions(
        values,
        assignment,
        [dict(positions=list(range(len(values))))],
        "complete_randomized",
    )
    v0, v1 = variances
    total = _add(v0, v1, "contrast variance bound")
    zero = torch.tensor(0.0, dtype=torch.float64, device="cpu")
    bound = torch.stack(
        [torch.stack([v0, zero, -v0]), torch.stack([zero, v1, v1]), torch.stack([-v0, v1, total])]
    )
    return (
        estimates,
        bound,
        dict(
            arm_centered_outcomes=centered,
            mean_contributions=scores.tolist(),
            mean_anchor=float(anchor),
            mean_anchor_coefficients=coefficients.tolist(),
            mean_anchor_contributions=location.tolist(),
            mean_contribution_method="bounded expansion of original design-unit common-location residual contributions plus analytic arm/contrast anchor contributions",
            observed_arm_variance_bounds=[float(v0), float(v1)],
        ),
    )


def _original_contributions(outcome, assignment, groups, required):
    """Aggregate points before rounding group totals/means or pair contrasts."""
    anchors, residual_matrix = _anchored(outcome[:, None])
    anchor, residuals = anchors[0], residual_matrix[:, 0]
    scores = torch.zeros((len(outcome), 3), dtype=torch.float64, device="cpu")
    coefficients = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64, device="cpu")
    if required == "cluster_randomized":
        counts = [sum(group["assignment"] == arm for group in groups) for arm in (0, 1)]
        individual_counts = [int((assignment == arm).sum()) for arm in (0, 1)]
        for arm in (0, 1):
            used = assignment == arm
            count = counts[arm]
            scores[used, arm] = _divide(
                _multiply(residuals[used], len(groups), "original-row HT numerator"),
                len(outcome) * count,
                "original-row HT contribution",
            )
            coefficients[arm] = len(groups) * individual_counts[arm] / (len(outcome) * count)
        coefficients[2] = (
            len(groups)
            * (individual_counts[1] * counts[0] - individual_counts[0] * counts[1])
            / (len(outcome) * counts[0] * counts[1])
        )
    elif required == "paired_randomized":
        for arm in (0, 1):
            used = assignment == arm
            scores[used, arm] = _divide(
                residuals[used], len(groups), "original-row paired mean contribution"
            )
    else:
        for group in groups:
            for arm in (0, 1):
                used = torch.tensor(
                    [i for i in group["positions"] if assignment[i] == arm],
                    dtype=torch.int64,
                    device="cpu",
                )
                if required == "complete_randomized":
                    scores[used, arm] = _divide(
                        residuals[used], len(used), "original-row complete mean contribution"
                    )
                else:
                    scores[used, arm] = _divide(
                        _multiply(residuals[used], group["n"], "original-row stratum numerator"),
                        len(outcome) * len(used),
                        "original-row stratum mean contribution",
                    )
    scores[:, 2] = _difference(scores[:, 1], scores[:, 0], "original-row contrast contribution")
    location = _multiply(anchor, coefficients, "point-estimate anchor contribution")
    estimates = _add(_sum_rows(scores), location, "point-estimate location restoration")
    return estimates, scores, anchor, coefficients, location


def _cluster_references(outcome, groups):
    """Total differences from original rows avoid rounded-total degeneracy.

    A smallest same-arm cluster is reused as reference. Its total repetition
    covers at most the arm's original row count, keeping the full plan linear.
    """
    relative = torch.zeros(len(groups), dtype=torch.float64, device="cpu")
    references = []
    for arm in (0, 1):
        used = [i for i, group in enumerate(groups) if group["assignment"] == arm]
        reference = min(used, key=lambda i: groups[i]["n"])
        reference_positions = groups[reference]["positions"]
        references.append(reference)
        for i in used:
            signed = torch.cat((outcome[groups[i]["positions"]], -outcome[reference_positions]))
            relative[i] = _multiply(
                _sum_rows(signed[:, None])[0],
                len(groups) / len(outcome),
                "reference-centered scaled cluster total",
            )
    return relative, references


def _typed_groups(groups, extra):
    rows = [
        [
            i,
            group["kind"],
            group["label"],
            group["n"],
            group["n_control"],
            group["n_treated"],
            m.canonical(group["positions"]),
            *extra[i],
        ]
        for i, group in enumerate(groups)
    ]
    table = m.frame(
        rows,
        columns=[
            "group_index",
            "identifier_type",
            "identifier",
            "n",
            "n_control",
            "n_treated",
            "positions",
            "mean_control",
            "mean_treated",
            "difference",
            "variance_bound_control",
            "variance_bound_treated",
            "variance_bound_difference",
        ],
    )
    table["identifier"] = pd.Series([group["label"] for group in groups], dtype=object)
    return table


def _finish(
    name, metadata, settings, groups, outcomes, assignment, estimates, bound, tables, state
):
    level, null_effect, alternative = (
        settings["level"],
        settings["null_effect"],
        settings["alternative"],
    )
    _finite(bound, "complete covariance bound")
    scale = float(bound.abs().max())
    if scale > 0:
        normalized = _divide(bound, scale, "covariance bound PSD check")
        if float(torch.linalg.eigvalsh(normalized).min()) < -256 * torch.finfo(torch.float64).eps:
            raise AnalysisError(
                "numerical_failure",
                "The unrepaired covariance bound is outside its PSD roundoff domain.",
            )
    if not bool((bound.diag() >= 0).all()) or not bool((bound == bound.T).all()):
        raise AnalysisError(
            "numerical_failure", "The covariance bound is not symmetric with nonnegative variances."
        )
    errors = bound.diag().sqrt()
    critical = normal_isf((1 - level) / 2)
    if not math.isfinite(critical) or critical <= 0:
        raise AnalysisError(
            "numerical_failure", "The Gaussian critical value is not representable."
        )
    rows = []
    for j, (term, estimate, error) in enumerate(
        zip(_TERMS, estimates.tolist(), errors.tolist(), strict=True)
    ):
        radius = critical * error
        low, high = estimate - radius, estimate + radius
        if not all(math.isfinite(value) for value in (radius, low, high)) or (
            error != 0 and (radius == 0 or low == estimate or high == estimate)
        ):
            raise AnalysisError(
                "numerical_failure", "A nonzero normal confidence shift is not representable."
            )
        z = p = None
        if j == 2 and error > 0:
            difference = float(
                _difference(
                    torch.tensor(estimate, dtype=torch.float64),
                    torch.tensor(null_effect, dtype=torch.float64),
                    "weak-null contrast",
                )
            )
            z = difference / error
            if not math.isfinite(z) or (difference != 0 and z == 0):
                raise AnalysisError(
                    "numerical_failure", "The weak-null normal statistic is not representable."
                )
            p = (
                2 * normal_sf(abs(z))
                if alternative == "two-sided"
                else normal_sf(z if alternative == "greater" else -z)
            )
        rows.append(
            [
                term,
                estimate,
                error,
                null_effect if j == 2 else None,
                z,
                p,
                low,
                high,
                None,
                error == 0,
            ]
        )
    subjects = m.frame(
        list(zip(metadata["positions"], assignment.tolist(), outcomes.tolist(), strict=True)),
        columns=["position", "treatment", "outcome"],
    )
    state.update(
        target="finite-population average treatment effect over all original supplied individuals",
        potential_outcomes_fixed=True,
        estimates=estimates.tolist(),
        estimated_covariance_bound=bound.tolist(),
        true_randomization_covariance=None,
        covariance_identified=False,
        covariance_method="estimated conservative Neyman covariance bound; its expectation dominates the true design covariance in PSD order",
        inference="pointwise asymptotic normal weak-null inference under appropriate design CLT and variance consistency; no exact t/permutation law or finite-sample coverage guarantee",
        design_groups=groups,
        original_outcomes=outcomes.tolist(),
        original_assignment=assignment.tolist(),
        critical_value=critical,
        df=None,
        sum_method="native bounded64-part FastTwoSum expansions with final float64 rounding",
        sum_capacity=_SUM_CAPACITY,
        no_generic_weights=True,
        missing_policy="raise only; original fixed population and topology retained",
        assumptions=[
            "consistency and no interference between assignment units",
            "original population and design groups fixed before assignment",
            "declared assignment law is the actual experiment",
            "appropriate nondegenerate design CLT and covariance-bound consistency for normal inference",
        ],
    )
    return m.result(
        name,
        {
            "effects": m.frame(
                rows,
                columns=[
                    "term",
                    "estimate",
                    "std_error",
                    "null_value",
                    "z",
                    "p_value",
                    "ci_low",
                    "ci_high",
                    "df",
                    "degenerate_variance",
                ],
            ),
            "covariance_bound": m.frame(bound.tolist(), columns=_TERMS, index=_TERMS),
            "subjects": subjects,
            **tables,
        },
        metadata,
        settings,
        state,
        notes=[
            "The covariance table estimates a conservative bound, not the unidentified true randomization covariance; conservatism is in expectation/asymptotically, not a sample-by-sample guarantee.",
            "Normal p-values test only the finite-population average-effect difference; heterogeneous individual effects are allowed and no Fisher sharp null is imposed.",
            "Intervals are pointwise asymptotic and two-sided even when the weak-null p-value is one-sided; no exact t or finite-sample coverage is asserted.",
            "Zero empirical variance gives a point interval and undefined z/p; it does not establish population certainty.",
        ],
    )


def _run(
    name,
    data,
    y,
    treatment,
    identity,
    *,
    design,
    required,
    null_effect,
    alternative,
    level,
    missing,
    device,
    weights,
    max_work,
):
    m.options(device, weights, max_work, level)
    null_effect = c.check_number(null_effect, "null_effect")
    c.check_choice(alternative, "alternative", ("two-sided", "greater", "less"))
    if required != "complete_randomized":
        identity = c.check_name(identity, "design identifier")
    outcome, assignment, groups, metadata = _prepare(
        data,
        y,
        treatment,
        identity,
        design=design,
        required=required,
        missing=missing,
        max_work=max_work,
    )
    settings = dict(
        y=y,
        treatment=treatment,
        design=design,
        null_effect=null_effect,
        alternative=alternative,
        level=level,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    if identity is not None:
        settings[
            {
                "stratified_randomized": "strata",
                "cluster_randomized": "cluster",
                "paired_randomized": "pair",
            }[required]
        ] = identity
    state, tables = {}, {}
    if required in ("complete_randomized", "stratified_randomized"):
        matrices, extra, summaries = [], [], []
        for group in groups:
            positions = group["positions"]
            estimates, bound, moments = _independent_arms(outcome[positions], assignment[positions])
            weight = group["n"] / len(outcome)
            matrices.append(
                _multiply(
                    _multiply(bound, weight, "stratum covariance weight"),
                    weight,
                    "stratum covariance contribution",
                )
            )
            extra.append([*estimates.tolist(), *bound.diag().tolist()])
            summaries.append(
                dict(
                    weight=weight,
                    estimates=estimates.tolist(),
                    covariance_bound=bound.tolist(),
                    **moments,
                )
            )
        bound = _sum_rows(torch.stack(matrices).reshape(len(groups), 9)).reshape(3, 3)
        tables["groups"] = _typed_groups(groups, extra)
        state.update(
            group_summaries=summaries,
            fixed_treatment_counts=[group["n_treated"] for group in groups],
            assignment_law="uniform fixed-treated-count assignments, independent across predeclared strata"
            if identity is not None
            else "uniform complete fixed-treated-count assignments",
            variance_bound_formula="sum_h (N_h/N)^2 * diag(s0_h^2/n0_h,s1_h^2/n1_h), transformed to both arm means and their difference; unidentified covariance correction omitted",
        )
    elif required == "cluster_randomized":
        counts, unit_rows = [], []
        scale = len(groups) / len(outcome)
        for group in groups:
            total = _sum_rows(outcome[group["positions"], None])[0]
            scaled_total = _multiply(total, scale, "scaled cluster total")
            counts.append(group["assignment"])
            unit_rows.append([group["n"], float(total), float(scaled_total), group["assignment"]])
        cluster_assignment = torch.tensor(counts, dtype=torch.float64, device="cpu")
        relative_clusters, references = _cluster_references(outcome, groups)
        estimates, bound, moments = _independent_arms(relative_clusters, cluster_assignment)
        tables["clusters"] = m.frame(
            [
                [i, group["kind"], group["label"], m.canonical(group["positions"]), *row]
                for i, (group, row) in enumerate(zip(groups, unit_rows, strict=True))
            ],
            columns=[
                "cluster_index",
                "identifier_type",
                "identifier",
                "positions",
                "cluster_size",
                "observed_total",
                "scaled_total",
                "treatment",
            ],
        )
        tables["clusters"]["identifier"] = pd.Series(
            [group["label"] for group in groups], dtype=object
        )
        state.update(
            **moments,
            cluster_units=unit_rows,
            n_clusters=len(groups),
            n_treated_clusters=sum(counts),
            n_control_clusters=len(groups) - sum(counts),
            cluster_total_scale=scale,
            variance_reference_clusters=references,
            reference_centered_scaled_cluster_totals=relative_clusters.tolist(),
            cluster_covariance_method="sample covariance of original-row total differences from a smallest same-arm reference cluster; rounded raw totals are display summaries only",
            assignment_law="uniform fixed-treated-cluster-count complete randomization, all individuals observed",
            variance_bound_formula="s_scaled_total,0^2/G0+s_scaled_total,1^2/G1 for individual-average ATE; each scaled total is G/N times the cluster outcome sum",
            unequal_cluster_size_target="individual-average finite-population ATE, not unweighted average of cluster means",
            maximum_cluster_population_fraction=max(group["n"] for group in groups) / len(outcome),
        )
    else:
        values, pairs = [], []
        for i, group in enumerate(groups):
            treated, control = sorted(group["positions"], key=lambda j: -float(assignment[j]))
            y0, y1 = outcome[control], outcome[treated]
            difference = _difference(y1, y0, "pair contrast")
            values.append(torch.stack([y0, y1, difference]))
            pairs.append(
                [
                    i,
                    group["kind"],
                    group["label"],
                    control,
                    treated,
                    float(y0),
                    float(y1),
                    float(difference),
                ]
            )
        pair_outcomes = torch.stack(values)
        estimates, sample_covariance, centered = _moments(pair_outcomes)
        bound = _divide(sample_covariance, len(groups), "pair-mean covariance bound")
        tables["pairs"] = m.frame(
            pairs,
            columns=[
                "pair_index",
                "identifier_type",
                "identifier",
                "control_position",
                "treated_position",
                "control_outcome",
                "treated_outcome",
                "difference",
            ],
        )
        tables["pairs"]["identifier"] = pd.Series(
            [group["label"] for group in groups], dtype=object
        )
        state.update(
            pair_outcomes=pair_outcomes.tolist(),
            centered_pair_outcomes=centered.tolist(),
            n_pairs=len(groups),
            assignment_law="independent fair one-treated-of-two assignments in each predeclared pair",
            variance_bound_formula="full unbiased sample covariance of observed [control,treated,difference] pair vectors divided by M; unidentified between-pair mean covariance adds a PSD expectation gap",
        )
    estimates, original_contributions, anchor, coefficients, location = _original_contributions(
        outcome, assignment, groups, required
    )
    state.update(
        original_row_mean_contributions=original_contributions.tolist(),
        point_anchor=float(anchor),
        point_anchor_coefficients=coefficients.tolist(),
        point_anchor_contributions=location.tolist(),
        point_estimate_method="bounded expansion of all original-row common-location residual weighted contributions, plus analytic design anchor contributions; group means/totals and pair contrasts are rounded summaries, never the point-estimate aggregation path",
    )
    return _finish(
        name, metadata, settings, groups, outcome, assignment, estimates, bound, tables, state
    )


@m.procedure
def neyman_ate(
    data,
    y,
    treatment,
    *,
    design,
    null_effect=0.0,
    alternative="two-sided",
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Complete fixed-count randomized finite-population ATE and Neyman bound."""
    return _run(
        "neyman_ate",
        data,
        y,
        treatment,
        None,
        design=design,
        required="complete_randomized",
        null_effect=null_effect,
        alternative=alternative,
        level=level,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )


@m.procedure
def stratified_neyman_ate(
    data,
    y,
    treatment,
    strata,
    *,
    design,
    null_effect=0.0,
    alternative="two-sided",
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Independent fixed-count strata, size-weighted SATE and Neyman bound."""
    return _run(
        "stratified_neyman_ate",
        data,
        y,
        treatment,
        strata,
        design=design,
        required="stratified_randomized",
        null_effect=null_effect,
        alternative=alternative,
        level=level,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )


@m.procedure
def cluster_neyman_ate(
    data,
    y,
    treatment,
    cluster,
    *,
    design,
    null_effect=0.0,
    alternative="two-sided",
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Fixed-count cluster HT individual-average SATE using G/N scaled totals."""
    return _run(
        "cluster_neyman_ate",
        data,
        y,
        treatment,
        cluster,
        design=design,
        required="cluster_randomized",
        null_effect=null_effect,
        alternative=alternative,
        level=level,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )


@m.procedure
def paired_neyman_ate(
    data,
    y,
    treatment,
    pair,
    *,
    design,
    null_effect=0.0,
    alternative="two-sided",
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Independent fair two-unit pairs, SATE and full pair covariance bound."""
    return _run(
        "paired_neyman_ate",
        data,
        y,
        treatment,
        pair,
        design=design,
        required="paired_randomized",
        null_effect=null_effect,
        alternative=alternative,
        level=level,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )
