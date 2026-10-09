"""Fixed observational distribution targets with complete nuisance inference."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.resources import plan_workspace

from . import common as m

_HALVINGS = 12
_ASSUMPTIONS = (
    "Consistency/SUTVA, independent identically distributed rows, pre-treatment numeric controls, "
    "conditional unconfoundedness and population propensity bounded away from zero and one. "
    "These are declared assumptions, not established by fitted overlap diagnostics."
)


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{label} is not finite in float64.")


def _grid(values, name, *, quantile=False):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 16:
        raise AnalysisError("invalid_targets", f"{name} must be a fixed list of 1..16 targets.")
    result = []
    for value in values:
        m.c.check_number(value, name)
        value = float(value)
        if abs(value) > 1e150 or (quantile and not 0 < value < 1):
            raise AnalysisError(
                "invalid_targets", f"{name} is outside the supported finite domain."
            )
        result.append(value)
    if len(set(result)) != len(result):
        raise AnalysisError("invalid_targets", f"{name} must contain unique predeclared targets.")
    return result


def _options(design, overlap, max_iterations, device, weights, max_work, level):
    m.options(device, weights, max_work, level)
    m.c.check_choice(design, "design", ("unconfounded",))
    m.c.check_number(overlap, "overlap", minimum=0, maximum=0.5, exclusive=True)
    if overlap >= 0.5:
        raise AnalysisError("invalid_option", "overlap must be strictly less than one half.")
    m.c.check_count(max_iterations, "max_iterations", minimum=1, maximum=200)


def _fit_work(n, p, iterations):
    # Each derivative call includes Hessian/score and an undamped Cholesky solve.
    # Backtracking is a separate value-only calculation; there is no ridge search.
    derivative = 2 * n * p * p + 12 * n * p + 20 * n + 8 * p**3 + 20 * p * p
    value = 2 * n * p + 12 * n + 4 * p
    return (iterations + 1) * derivative + iterations * _HALVINGS * value


def _prepare(data, y, treatment, x, missing, max_work, k, iterations, *, reps=0, folds=0):
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset", "Observational targets require a resident table."
        )
    controls = m.c.name_list(x, "x", minimum=0)
    m.c.check_name(y, "y")
    m.c.check_name(treatment, "treatment")
    if len(controls) > 16:
        raise AnalysisError("resource_limit", "At most 16 fixed numeric controls are supported.")
    names = [y, treatment, *controls]
    if len(names) != len(set(names)):
        raise AnalysisError("invalid_roles", "Outcome, treatment and controls must be distinct.")
    source = m.c.source(data)
    n, p = len(source), len(controls) + 1
    if n > 10000:
        raise AnalysisError("resource_limit", "At most 10000 resident input rows are supported.")
    fits = reps + 1 if reps else folds * (1 + 2 * k) if folds else 1
    dimension = p + 2 * k
    units = fits * _fit_work(n, p, iterations)
    units += n * (18 * k + 6 * dimension**2) + 8 * dimension**3
    units += reps * n * (4 * p + 4 * k + 4 * math.ceil(math.log2(max(n, 2))))
    units += reps * (18 * k * k + 12 * k)
    m.work(units, max_work, "all observational nuisance fits, line searches and targets")
    plan = plan_workspace(
        "observational distribution complete fits and artifacts",
        {
            "source_sample_design_and_python_json": 512 * n * (p + 3),
            "scores_jacobian_covariances_and_table_json": 384 * (n * dimension + dimension**2),
            "all_fit_certificates_probabilities_weights_and_json": fits
            * (512 * n * (p + 2) + 512 * p**2),
            "bootstrap_indices_refitted_designs_targets_and_json": reps * n * (256 * p + 512),
            "bootstrap_joint_targets_covariance_and_json": 512 * (reps * 3 * k + 9 * k**2),
            "crossfit_models_predictions_and_json": folds * 512 * n * k,
        },
    )
    selected, metadata = m.sample(
        data, names, numeric=names, missing=missing, max_work=max_work, minimum=30
    )
    outcome, assignment = m.tensor(selected, y), m.binary(selected, treatment)
    if min(int((assignment == arm).sum()) for arm in (0, 1)) < 6:
        raise AnalysisError(
            "insufficient_arm_sample", "Each original treatment arm needs at least six rows."
        )
    raw = (
        torch.stack([m.tensor(selected, name) for name in controls], 1)
        if controls
        else torch.empty((len(selected), 0), dtype=torch.float64)
    )
    metadata.update(
        observational_resource_plan=plan.record(),
        planned_work=units,
        control_columns=controls,
        roles={"y": y, "treatment": treatment, "x": controls},
    )
    return outcome, assignment, raw, metadata


def _standardize(raw):
    n, p = raw.shape
    center = raw.mean(0)
    scale = (raw - center).square().mean(0).sqrt()
    _finite(center, "Training centering")
    _finite(scale, "Training scale")
    if p and bool((scale == 0).any()):
        raise AnalysisError(
            "singular_design", "A nuisance control is constant; no column is dropped."
        )
    z = torch.cat((torch.ones((n, 1), dtype=torch.float64), (raw - center) / scale), 1)
    _finite(z, "Standardized design")
    singular = torch.linalg.svdvals(z)
    if singular[-1] <= singular[0] * max(z.shape) * torch.finfo(torch.float64).eps:
        raise AnalysisError("singular_design", "The nuisance design is not full rank.")
    return z, center, scale


def _logit(raw, response, max_iterations, label):
    """Unpenalized finite logit MLE; no curvature/objective regularization."""
    n = len(response)
    if n < max(6, raw.shape[1] + 3):
        raise AnalysisError("insufficient_training_sample", f"{label} needs more training rows.")
    if response.min() == response.max():
        raise AnalysisError(
            "degenerate_training_class", f"{label} must have both response classes."
        )
    z, center, scale = _standardize(raw)
    beta = torch.zeros(z.shape[1], dtype=torch.float64)
    beta[0] = torch.logit(response.mean())
    history, searches = [], []
    converged = False
    for iteration in range(max_iterations + 1):
        eta = z @ beta
        _finite(eta, label + " index")
        probability = torch.sigmoid(eta)
        value = (response * eta - torch.nn.functional.softplus(eta)).sum()
        score = z.T @ (response - probability)
        information = z.T @ (z * (probability * (1 - probability))[:, None])
        _finite(information, label + " information")
        _finite(score, label + " score")
        _finite(value, label + " likelihood")
        factor, status = torch.linalg.cholesky_ex(information)
        if int(status) != 0:
            raise AnalysisError(
                "separation_or_singular_curvature", f"{label} has no admitted finite curvature."
            )
        eigenvalues = torch.linalg.eigvalsh(information)
        if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
            raise AnalysisError(
                "separation_or_singular_curvature",
                f"{label} curvature exceeds the admitted condition limit.",
            )
        direction = torch.cholesky_solve(score[:, None], factor).flatten()
        decrement = float(score @ direction)
        relative = float((direction.abs() / beta.abs().clamp_min(1)).max())
        history.append(float(value))
        # Both curvature and the UNDAMPED Newton step are required. Small score
        # alone would falsely certify a separating likelihood at infinite beta.
        if decrement <= 1e-12 and float(score.abs().max()) <= 1e-9 * n and relative <= 1e-9:
            converged = True
            break
        if iteration == max_iterations:
            break
        accepted = False
        for halving in range(_HALVINGS):
            fraction = 2.0 ** (-halving)
            candidate = beta + fraction * direction
            candidate_eta = z @ candidate
            candidate_value = (
                response * candidate_eta - torch.nn.functional.softplus(candidate_eta)
            ).sum()
            allowance = 16 * torch.finfo(torch.float64).eps * max(1.0, abs(float(value)))
            if (
                torch.isfinite(candidate_value)
                and float(candidate_value) >= float(value) + 1e-4 * fraction * decrement - allowance
            ):
                beta = candidate
                searches.append(
                    {"iteration": iteration + 1, "halvings": halving, "fraction": fraction}
                )
                accepted = True
                break
        if not accepted:
            raise AnalysisError(
                "nonconvergence", f"{label} failed the bounded unpenalized line search."
            )
    if not converged:
        raise AnalysisError(
            "nonconvergence", f"{label} has no certified finite MLE within max_iterations."
        )
    if bool(((probability <= 1e-10) | (probability >= 1 - 1e-10)).any()):
        raise AnalysisError(
            "separation_or_extreme_fit", f"{label} predicts a boundary probability; fit refused."
        )
    score_rows = z * (response - probability)[:, None]
    fit_contribution = torch.cholesky_solve(score_rows.T, factor).T
    coefficient_covariance = fit_contribution.T @ fit_contribution
    _finite(coefficient_covariance, label + " coefficient covariance")
    record = dict(
        kind="unpenalized_logit",
        training_n=n,
        coefficients=beta.tolist(),
        center=center.tolist(),
        scale=scale.tolist(),
        design=z.tolist(),
        response=response.tolist(),
        probabilities=probability.tolist(),
        log_likelihood=float(value),
        score=score.tolist(),
        hessian=(-information).tolist(),
        score_rows=score_rows.tolist(),
        coefficient_covariance=coefficient_covariance.tolist(),
        coefficient_influence=fit_contribution.tolist(),
        curvature_eigenvalues=eigenvalues.tolist(),
        max_curvature_condition=1e12,
        iterations=iteration,
        converged=True,
        scaled_gradient=decrement,
        undamped_relative_newton_step=relative,
        line_search=searches,
        value_history=history,
        objective_penalty=0.0,
        curvature_ridge=0.0,
        max_halvings=_HALVINGS,
        boundary_probability_guard=1e-10,
    )
    return probability, z, record


def _predict(fit, raw):
    center, scale, beta = [
        torch.tensor(fit[key], dtype=torch.float64) for key in ("center", "scale", "coefficients")
    ]
    z = torch.cat((torch.ones((len(raw), 1), dtype=torch.float64), (raw - center) / scale), 1)
    _finite(z, "Held-out standardized design")
    eta = z @ beta
    _finite(eta, "Held-out logit index")
    result = torch.sigmoid(eta)
    _finite(result, "Held-out nuisance predictions")
    return result


def _overlap(e, overlap):
    if bool(((e < overlap) | (e > 1 - overlap)).any()):
        raise AnalysisError(
            "overlap_violation",
            "Fitted propensity violates declared overlap; no clipping or row trimming.",
        )


def _weights(d, e):
    return torch.stack(((1 - d) / (1 - e), d / e), 1)


def _joint(arms, contributions):
    k = arms.shape[1]
    values = torch.cat((arms[0], arms[1], arms[1] - arms[0]))
    joint = torch.cat(
        (contributions[:, :k], contributions[:, k:], contributions[:, k:] - contributions[:, :k]), 1
    )
    covariance = joint.T @ joint
    _finite(covariance, "Joint covariance")
    if bool(((covariance.diag() == 0) & (joint != 0).any(0)).any()):
        raise AnalysisError(
            "numerical_failure",
            "A nonconstant variance underflowed; zero uncertainty is not asserted.",
        )
    return values, covariance, joint


def _names(k):
    return [f"{arm}[{j}]" for arm in ("cdf0", "cdf1", "effect") for j in range(k)]


def _wald(value, variance, level):
    se = math.sqrt(float(variance))
    critical = math.sqrt(2) * float(torch.erfinv(torch.tensor(level, dtype=torch.float64)))
    if se == 0:
        return [0.0, None, None, value, value, "zero_empirical_variance"]
    z = value / se
    radius = critical * se
    low, high = value - radius, value + radius
    if (
        not all(math.isfinite(item) for item in (z, radius, low, high))
        or radius <= 0
        or low == value
        or high == value
    ):
        raise AnalysisError(
            "numerical_failure",
            "A nonzero normal statistic or confidence shift is not representable.",
        )
    p = float(torch.special.erfc(torch.tensor(abs(z) / math.sqrt(2), dtype=torch.float64)))
    return [se, z, p, low, high, "pointwise_asymptotic_normal"]


def _cdf_tables(grid, values, covariance, level):
    k = len(grid)
    effects = [
        [
            grid[j],
            float(values[j]),
            float(values[k + j]),
            float(values[2 * k + j]),
            *_wald(float(values[2 * k + j]), covariance[2 * k + j, 2 * k + j], level),
        ]
        for j in range(k)
    ]
    arms = [
        [
            arm,
            grid[j],
            float(values[arm * k + j]),
            *_wald(float(values[arm * k + j]), covariance[arm * k + j, arm * k + j], level),
        ]
        for arm in (0, 1)
        for j in range(k)
    ]
    names = _names(k)
    return dict(
        effects=m.frame(
            effects,
            columns=[
                "threshold",
                "cdf0",
                "cdf1",
                "estimate",
                "std_error",
                "z",
                "p_value",
                "ci_low",
                "ci_high",
                "inference_status",
            ],
        ),
        arm_targets=m.frame(
            arms,
            columns=[
                "arm",
                "threshold",
                "estimate",
                "std_error",
                "z",
                "p_value",
                "ci_low",
                "ci_high",
                "inference_status",
            ],
        ),
        joint_covariance=m.frame(covariance.tolist(), columns=names, index=names),
        covariance=m.frame(
            covariance[2 * k :, 2 * k :].tolist(),
            columns=[f"effect[{j}]" for j in range(k)],
            index=[f"effect[{j}]" for j in range(k)],
        ),
    )


def _settings(
    y, treatment, x, design, targets, overlap, level, missing, max_iterations, max_work, **extra
):
    return dict(
        y=y,
        treatment=treatment,
        x=m.c.name_list(x, "x", minimum=0),
        design=design,
        **targets,
        overlap=overlap,
        level=level,
        missing=missing,
        max_iterations=max_iterations,
        max_halvings=_HALVINGS,
        device="cpu",
        weights=None,
        max_work=max_work,
        **extra,
    )


def _source(y, d, raw):
    return dict(outcome=y.tolist(), treatment=d.tolist(), controls=raw.tolist())


@m.procedure
def treatment_cdf_ipw(
    data,
    y,
    treatment,
    *,
    x,
    design,
    thresholds,
    overlap=0.01,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_iterations=40,
    max_work=100_000_000,
):
    """Fixed-grid marginal CDF contrast with fitted-logit stacked sandwich inference."""
    _options(design, overlap, max_iterations, device, weights, max_work, level)
    grid = _grid(thresholds, "thresholds")
    outcome, d, raw, metadata = _prepare(
        data, y, treatment, x, missing, max_work, len(grid), max_iterations
    )
    n, k = len(outcome), len(grid)
    e, z, fit = _logit(raw, d, max_iterations, "Propensity model")
    _overlap(e, overlap)
    mass = _weights(d, e)
    indicator = (outcome[:, None] <= torch.tensor(grid, dtype=torch.float64)).to(torch.float64)
    arms = mass.T @ indicator / mass.sum(0)[:, None]
    # Constant empirical arm indicators have exact normalized mean 0/1. Preserve
    # that algebraic identity rather than converting summation roundoff to an SE.
    for arm in (0, 1):
        values = indicator[d == arm]
        constant = values.min(0).values == values.max(0).values
        arms[arm, constant] = values[0, constant]
    p = z.shape[1]
    score = torch.cat(
        (
            z * (d - e)[:, None],
            mass[:, 0, None] * (indicator - arms[0]),
            mass[:, 1, None] * (indicator - arms[1]),
        ),
        1,
    )
    jacobian = torch.zeros((p + 2 * k, p + 2 * k), dtype=torch.float64)
    jacobian[:p, :p] = -(z.T @ (z * (e * (1 - e))[:, None])) / n
    for arm in (0, 1):
        start = p + arm * k
        derivative = mass[:, arm] * (e if arm == 0 else -(1 - e))
        jacobian[start : start + k, :p] = ((indicator - arms[arm]) * derivative[:, None]).T @ z / n
        jacobian[start : start + k, start : start + k] = (
            -torch.eye(k, dtype=torch.float64) * mass[:, arm].mean()
        )
    influence = -torch.linalg.solve(jacobian, score.T).T / n
    full_covariance = influence.T @ influence
    values, covariance, target_contributions = _joint(arms, influence[:, p:])
    tables = _cdf_tables(grid, values, covariance, level)
    full_names = [f"propensity_beta[{j}]" for j in range(p)] + _names(k)[: 2 * k]
    tables["nuisance_covariance"] = m.frame(
        full_covariance.tolist(), columns=full_names, index=full_names
    )
    state = dict(
        assumptions=_ASSUMPTIONS
        + " Correct finite-dimensional logistic propensity specification and regular finite MLE.",
        inference="HC0 stacked estimating equations; includes estimated propensity and Hájek denominators; pointwise normal intervals, no simultaneous band",
        covariance_type="HC0",
        df=None,
        reference_distribution="standard normal",
        source=_source(outcome, d, raw),
        propensity_fit=fit,
        propensity=e.tolist(),
        inverse_weights=mass.tolist(),
        indicators=indicator.tolist(),
        arm_targets=arms.tolist(),
        score_parameters=full_names,
        scores=score.tolist(),
        jacobian=jacobian.tolist(),
        normalized_influence=influence.tolist(),
        full_parameter_covariance=full_covariance.tolist(),
        joint_parameters=_names(k),
        joint_targets=values.tolist(),
        normalized_target_influence=target_contributions.tolist(),
        joint_covariance=covariance.tolist(),
        overlap_diagnostics=dict(
            min=float(e.min()),
            max=float(e.max()),
            weight_sums=mass.sum(0).tolist(),
            effective_arm_n=(mass.sum(0).square() / mass.square().sum(0)).tolist(),
        ),
    )
    return m.result(
        "treatment_cdf_ipw",
        tables,
        metadata,
        _settings(
            y,
            treatment,
            x,
            design,
            {"thresholds": grid},
            overlap,
            level,
            missing,
            max_iterations,
            max_work,
        ),
        state,
        notes=[
            "Fitted overlap does not establish unconfoundedness or population positivity.",
            "Zero empirical variance does not prove unsampled-tail certainty.",
        ],
    )


def _quantiles(outcome, d, e, q):
    mass = _weights(d, e)
    values, ordered = [], []
    for arm in (0, 1):
        rows = (d == arm).nonzero().flatten()
        if len(rows) < 6:
            raise AnalysisError(
                "insufficient_arm_sample",
                "Every quantile arm, including bootstrap arms, needs six rows.",
            )
        order = rows[torch.argsort(outcome[rows], stable=True)]
        cumulative = mass[order, arm].cumsum(0)
        index = torch.searchsorted(cumulative, q * cumulative[-1], right=False)
        values.append(outcome[order[index]])
        ordered.append(
            dict(
                sample_positions=order.tolist(),
                outcomes=outcome[order].tolist(),
                weights=mass[order, arm].tolist(),
                cumulative_weight=cumulative.tolist(),
                quantile_order_indices=index.tolist(),
            )
        )
    return torch.stack(values), mass, ordered


@m.procedure
def treatment_quantile_ipw(
    data,
    y,
    treatment,
    *,
    x,
    design,
    quantiles,
    reps=199,
    seed=1729,
    overlap=0.01,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_iterations=40,
    max_work=1_000_000_000,
):
    """Weighted inverse-CDF QTE; complete IID row bootstrap refits every propensity."""
    _options(design, overlap, max_iterations, device, weights, max_work, level)
    grid = _grid(quantiles, "quantiles", quantile=True)
    m.c.check_count(reps, "reps", minimum=20, maximum=999)
    m.c.check_count(seed, "seed", minimum=0, maximum=2**63 - 1)
    outcome, d, raw, metadata = _prepare(
        data, y, treatment, x, missing, max_work, len(grid), max_iterations, reps=reps
    )
    n, k = len(outcome), len(grid)
    q = torch.tensor(grid, dtype=torch.float64)
    e, _, fit = _logit(raw, d, max_iterations, "Propensity model")
    _overlap(e, overlap)
    arms, mass, ordered = _quantiles(outcome, d, e, q)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    before = generator.get_state().tolist()
    draws = torch.randint(n, (reps, n), generator=generator, device="cpu")
    records, replicates = [], []
    for rep, draw in enumerate(draws):
        rd, ry, rx = d[draw], outcome[draw], raw[draw]
        re, _, rf = _logit(rx, rd, max_iterations, f"Bootstrap propensity {rep}")
        _overlap(re, overlap)
        targets, rmass, rordered = _quantiles(ry, rd, re, q)
        joint = torch.cat((targets[0], targets[1], targets[1] - targets[0]))
        replicates.append(joint)
        records.append(
            dict(
                replicate=rep,
                original_positions=[metadata["positions"][j] for j in draw.tolist()],
                propensity_fit=rf,
                inverse_weights=rmass.tolist(),
                ordered_arms=rordered,
                joint_targets=joint.tolist(),
            )
        )
    replicate = torch.stack(replicates)
    center = replicate - replicate.mean(0)
    constant = replicate.min(0).values == replicate.max(0).values
    center[:, constant] = 0.0
    covariance = center.T @ center / (reps - 1)
    _finite(covariance, "Bootstrap joint covariance")
    if bool(((covariance.diag() == 0) & (center != 0).any(0)).any()):
        raise AnalysisError("numerical_failure", "A nonconstant bootstrap variance underflowed.")
    values = torch.cat((arms[0], arms[1], arms[1] - arms[0]))
    alpha = (1 - level) / 2
    intervals = torch.quantile(
        replicate, torch.tensor([alpha, 1 - alpha], dtype=torch.float64), dim=0
    )
    _finite(intervals, "Percentile intervals")
    names = [f"{arm}[{j}]" for arm in ("quantile0", "quantile1", "effect") for j in range(k)]

    def row(at):
        return [
            float(values[at]),
            float(covariance[at, at].sqrt()),
            None,
            None,
            float(intervals[0, at]),
            float(intervals[1, at]),
            "zero_empirical_bootstrap_variance" if constant[at] else "bootstrap_percentile",
        ]

    tables = dict(
        effects=m.frame(
            [[grid[j], float(arms[0, j]), float(arms[1, j]), *row(2 * k + j)] for j in range(k)],
            columns=[
                "quantile",
                "quantile0",
                "quantile1",
                "estimate",
                "std_error",
                "z",
                "p_value",
                "ci_low",
                "ci_high",
                "inference_status",
            ],
        ),
        arm_targets=m.frame(
            [[arm, grid[j], *row(arm * k + j)] for arm in (0, 1) for j in range(k)],
            columns=[
                "arm",
                "quantile",
                "estimate",
                "std_error",
                "z",
                "p_value",
                "ci_low",
                "ci_high",
                "inference_status",
            ],
        ),
        joint_covariance=m.frame(covariance.tolist(), columns=names, index=names),
        covariance=m.frame(
            covariance[2 * k :, 2 * k :].tolist(), columns=names[2 * k :], index=names[2 * k :]
        ),
    )
    state = dict(
        assumptions=_ASSUMPTIONS
        + " Correct finite-dimensional logistic propensity; regular finite MLE; continuous marginal outcomes with unique quantiles and positive finite density at chosen quantiles. Ties may be processed, but do not establish this population regularity.",
        inference="IID whole-row empirical bootstrap, every nuisance refitted; percentile pointwise intervals; no normal-bootstrap p value",
        inverse_cdf="smallest observed arm outcome with sorted cumulative inverse weight >= q times total sorted inverse weight; inclusive ties; no interpolation",
        source=_source(outcome, d, raw),
        propensity_fit=fit,
        propensity=e.tolist(),
        inverse_weights=mass.tolist(),
        ordered_arms=ordered,
        joint_targets=values.tolist(),
        joint_parameters=names,
        bootstrap_sample_indices=draws.tolist(),
        bootstrap_records=records,
        bootstrap_joint_targets=replicate.tolist(),
        joint_covariance=covariance.tolist(),
        joint_percentile_intervals=intervals.tolist(),
        rng=dict(
            algorithm="Torch CPU randint with local generator",
            seed=seed,
            before=before,
            after=generator.get_state().tolist(),
        ),
        completed_replicates=reps,
        failed_replicates=0,
        degenerate_joint_indices=constant.nonzero().flatten().tolist(),
        df=None,
    )
    return m.result(
        "treatment_quantile_ipw",
        tables,
        metadata,
        _settings(
            y,
            treatment,
            x,
            design,
            {"quantiles": grid},
            overlap,
            level,
            missing,
            max_iterations,
            max_work,
            reps=reps,
            seed=seed,
        ),
        state,
        notes=[
            "Quantile treatment effects are differences of marginal quantiles, not quantiles of individual effects.",
            "No replicate is skipped, redrawn or repaired; a failed fit rejects the complete bootstrap.",
        ],
    )


@m.procedure
def treatment_cdf_aipw(
    data,
    y,
    treatment,
    *,
    x,
    design,
    thresholds,
    folds=3,
    seed=1729,
    overlap=0.01,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_iterations=40,
    max_work=100_000_000,
):
    """Cross-fitted threshold-logit AIPW CDF targets and complete joint score covariance."""
    _options(design, overlap, max_iterations, device, weights, max_work, level)
    grid = _grid(thresholds, "thresholds")
    m.c.check_count(folds, "folds", minimum=2, maximum=5)
    m.c.check_count(seed, "seed", minimum=0, maximum=2**63 - 1)
    outcome, d, raw, metadata = _prepare(
        data, y, treatment, x, missing, max_work, len(grid), max_iterations, folds=folds
    )
    n, k = len(outcome), len(grid)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(n, generator=generator, device="cpu")
    assignment = torch.empty(n, dtype=torch.int64)
    assignment[order] = torch.arange(n) % folds
    e = torch.empty(n, dtype=torch.float64)
    predicted = torch.empty((n, 2, k), dtype=torch.float64)
    indicator = (outcome[:, None] <= torch.tensor(grid, dtype=torch.float64)).to(torch.float64)
    records = []
    for fold in range(folds):
        train, test = (
            (assignment != fold).nonzero().flatten(),
            (assignment == fold).nonzero().flatten(),
        )
        _, _, pf = _logit(raw[train], d[train], max_iterations, f"Fold {fold} propensity")
        fitted_training = torch.tensor(pf["probabilities"], dtype=torch.float64)
        _overlap(fitted_training, overlap)
        e[test] = _predict(pf, raw[test])
        _overlap(e[test], overlap)
        models = []
        for arm in (0, 1):
            arm_train = train[d[train] == arm]
            arm_models = []
            for j in range(k):
                _, _, of = _logit(
                    raw[arm_train],
                    indicator[arm_train, j],
                    max_iterations,
                    f"Fold {fold} arm {arm} threshold {j}",
                )
                predicted[test, arm, j] = _predict(of, raw[test])
                arm_models.append(of)
            models.append(arm_models)
        records.append(
            dict(
                fold=fold,
                train_sample_indices=train.tolist(),
                test_sample_indices=test.tolist(),
                train_original_positions=[metadata["positions"][j] for j in train.tolist()],
                test_original_positions=[metadata["positions"][j] for j in test.tolist()],
                propensity_fit=pf,
                outcome_fits=models,
            )
        )
    mass = _weights(d, e)
    score = predicted + mass[:, :, None] * (indicator[:, None, :] - predicted)
    arm_targets = score.mean(0)
    centered = torch.cat((score[:, 0] - arm_targets[0], score[:, 1] - arm_targets[1]), 1) / n
    values, covariance, contribution = _joint(arm_targets, centered)
    state = dict(
        assumptions=_ASSUMPTIONS,
        point_consistency="At least one nuisance component consistently estimated (propensity, or both arm conditional indicator means); finite moments and positivity.",
        inference="Both propensity and arm conditional-indicator nuisances consistently estimated, sufficient product rates and regularity for cross-fitted orthogonal inference; HC0 held-out score covariance, pointwise normal intervals only.",
        covariance_type="HC0",
        df=None,
        reference_distribution="standard normal",
        limitations="One-correct-model point consistency does not alone validate the displayed confidence intervals. Raw AIPW CDF estimates may be outside [0,1] or nonmonotone; no clipping/rearrangement or quantile inversion.",
        source=_source(outcome, d, raw),
        indicators=indicator.tolist(),
        fold_assignment=assignment.tolist(),
        fold_records=records,
        rng=dict(
            algorithm="Torch CPU randperm local generator", seed=seed, permutation=order.tolist()
        ),
        propensity=e.tolist(),
        outcome_predictions=predicted.tolist(),
        inverse_weights=mass.tolist(),
        potential_scores=score.tolist(),
        arm_targets=arm_targets.tolist(),
        normalized_target_influence=contribution.tolist(),
        joint_parameters=_names(k),
        joint_targets=values.tolist(),
        joint_covariance=covariance.tolist(),
        overlap_diagnostics=dict(min=float(e.min()), max=float(e.max())),
        cross_fitted=True,
        fold_unit="IID row",
    )
    return m.result(
        "treatment_cdf_aipw",
        _cdf_tables(grid, values, covariance, level),
        metadata,
        _settings(
            y,
            treatment,
            x,
            design,
            {"thresholds": grid},
            overlap,
            level,
            missing,
            max_iterations,
            max_work,
            folds=folds,
            seed=seed,
        ),
        state,
        notes=[
            "Every nuisance prediction uses a held-out row; no empty arm/class fold is repaired.",
            "One-correct-model consistency is distinct from the both-consistent/product-rate CI assumptions.",
        ],
    )
