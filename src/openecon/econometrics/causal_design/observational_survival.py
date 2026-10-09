"""Known-nuisance subject-score survival and restricted-mean causal targets.

These are Horvitz--Thompson/AIPW score means, not weighted Kaplan--Meier.
No nuisance models are fitted or treated as known merely because predictions
were supplied. Identification requires the caller's true known propensity and
censoring laws; outcome augmentation is an externally prespecified function.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import common as c
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.resources import plan_workspace

from . import common as m

_SUM_CAPACITY = 64


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{label} exceeds finite float64 range.")
    return value


def _multiply(left, right, label):
    value = _finite(left * right, label)
    if bool(((left != 0) & (right != 0) & (value == 0)).any()):
        raise AnalysisError("numerical_failure", f"A nonzero {label} contribution underflows.")
    return value


def _divide(numerator, denominator, label):
    value = _finite(numerator / denominator, label)
    if bool(((numerator != 0) & (value == 0)).any()):
        raise AnalysisError("numerical_failure", f"A nonzero {label} contribution underflows.")
    return value


def _sum_rows(values):
    """Native bounded expansions retain nested cancellation remainders.

    FastTwoSum splits additions into high values and residuals; a single compensation
    variable can itself lose a small term when several scales cancel. Each vector
    coordinate retains its expansion, including harmless zero placeholders.
    """
    partials = []
    for value in values:
        x = value
        keep = 0
        for y in partials:
            larger = x.abs() >= y.abs()
            a, b = torch.where(larger, x, y), torch.where(larger, y, x)
            high = _finite(a + b, "score expansion sum")
            low = b - (high - a)
            if bool((low != 0).any()):
                partials[keep] = low
                keep += 1
            x = high
        partials[keep:] = [x]
        if len(partials) > _SUM_CAPACITY:
            raise AnalysisError(
                "numerical_failure",
                "Score sum exceeds the declared64-part floating-expansion capacity.",
            )
    total = torch.zeros(values.shape[1:], dtype=torch.float64, device="cpu")
    for value in reversed(partials):
        total = total + value
    return _finite(total, "expanded score sum")


def _declarations(design, nuisance, missing, positivity):
    if not isinstance(design, str) or design != "unconfounded":
        raise AnalysisError("unsupported_design", "Explicitly declare design='unconfounded'.")
    if not isinstance(nuisance, str) or nuisance != "known":
        raise AnalysisError(
            "unsupported_nuisance",
            "Declare nuisance='known' for true externally known treatment/censoring probabilities; fitted predictions are unsupported.",
        )
    if missing != "raise":
        raise AnalysisError(
            "unsupported_missing",
            "Known-nuisance causal targets require missing='raise'; row deletion may change the target and nuisance alignment.",
        )
    value = c.check_number(positivity, "positivity", minimum=0, maximum=0.5, exclusive=True)
    if value >= 0.5:
        raise AnalysisError("invalid_option", "positivity must be strictly between0 and.5.")
    return value


def _grid(thresholds):
    if not isinstance(thresholds, (list, tuple)) or not 1 <= len(thresholds) <= 32:
        raise AnalysisError(
            "invalid_option", "thresholds must contain 1..32 fixed increasing times."
        )
    values = [c.check_number(value, "threshold", minimum=0) for value in thresholds]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise AnalysisError("invalid_option", "Survival thresholds must increase strictly.")
    return values


def _sample(data, time, event, treatment, propensity, extra, missing, max_work, dimension):
    names = [
        c.check_name(name, role)
        for name, role in [
            (time, "time"),
            (event, "event"),
            (treatment, "treatment"),
            (propensity, "propensity"),
            *extra,
        ]
    ]
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "Survival input roles must name distinct columns.")
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset",
            "Known-nuisance causal targets require a resident table; Dataset replay is unsupported.",
        )
    source = c.source(data)
    n = len(source)
    if n > 100000 or len(names) > 64:
        raise AnalysisError(
            "resource_limit",
            "Causal survival inputs are limited to100000 rows and64 selected columns.",
        )
    cost = 64 + 16 * _SUM_CAPACITY * dimension + 8 * dimension * dimension
    planned = n * cost + 32 * dimension * dimension
    m.work(
        planned, max_work, "complete subject scores, bounded expansion means and joint covariance"
    )
    plan = plan_workspace(
        "known-nuisance causal survival full inference and state",
        {
            "selected_subjects_numeric_copies_and_provenance": 768 * n * len(names),
            "subject_scores_centered_covariance_and_python_state": 1024 * n * dimension,
            "complete_joint_covariance_tables_and_json": 1024 * dimension * dimension,
            "full_floating_expansion_capacity_and_live_error_vectors": 64
            * _SUM_CAPACITY
            * dimension,
        },
    )
    # missing='raise' keeps the original n: preflight the entire declared
    # computation before selecting/copying or validating numeric subject data.
    selected, metadata = m.sample(
        source, names, numeric=names, missing=missing, max_work=max_work, cost=cost, minimum=6
    )
    metadata["computation_resource_plan"] = plan.record()
    metadata["planned_work"] = planned
    times, events, assignment, p = [m.tensor(selected, name) for name in names[:4]]
    assignment = m.binary(selected, treatment)
    if bool((times < 0).any()):
        raise AnalysisError("invalid_time", "Observed time U=min(T,C) must be nonnegative.")
    if not bool(((events == 0) | (events == 1)).all()):
        raise AnalysisError("invalid_event", "event must be numeric0/1 for censoring/failure.")
    if min(int((assignment == arm).sum()) for arm in (0, 1)) < 3:
        raise AnalysisError(
            "insufficient_observations", "Each arm requires at least three subjects."
        )
    return selected, metadata, times, events, assignment, p


def _propensity(p, assignment, positivity):
    if bool(((p <= 0) | (p >= 1) | (p < positivity) | (p > 1 - positivity)).any()):
        raise AnalysisError(
            "positivity_violation",
            "Known propensity must lie within the declared positivity bounds; no trimming/clipping is performed.",
        )
    probability = torch.where(assignment == 1, p, 1 - p)
    weights = _divide(torch.ones_like(p), probability, "inverse treatment probability")
    return probability, weights


def _infer(scores, labels, level):
    n = len(scores)
    means = _divide(_sum_rows(scores), n, "subject-score mean")
    centered = _finite(scores - means, "centered subject scores")
    scales = centered.abs().max(0).values
    divisors = torch.where(scales == 0, torch.ones_like(scales), scales)
    standardized = _divide(centered, divisors, "centered-score normalization")
    normalized_covariance = standardized.T @ standardized / (n * (n - 1))
    if bool(((normalized_covariance.diag() == 0) & (centered != 0).any(0)).any()):
        raise AnalysisError(
            "numerical_failure", "A nonconstant subject-score variance rounded to zero."
        )
    # Multiply the larger scale first: a tiny intermediate must not lose a
    # covariance that the other, large scale would have made representable.
    large = torch.maximum(scales[:, None], scales[None, :])
    small = torch.minimum(scales[:, None], scales[None, :])
    covariance = _multiply(
        _multiply(normalized_covariance, large, "covariance rescaling"), small, "joint covariance"
    )
    errors = covariance.diag().sqrt()
    critical = normal_isf((1 - level) / 2)
    if not math.isfinite(critical) or critical <= 0:
        raise AnalysisError(
            "numerical_failure", "The Gaussian critical value is not representable."
        )
    rows = []
    for label, estimate, error in zip(labels, means.tolist(), errors.tolist(), strict=True):
        if error == 0:
            z = p_value = None
        else:
            z = estimate / error
            if not math.isfinite(z) or (estimate != 0 and z == 0):
                raise AnalysisError(
                    "numerical_failure", "The nonzero normal statistic is not representable."
                )
            p_value = 2 * normal_sf(abs(z))
        radius = critical * error
        low, high = estimate - radius, estimate + radius
        if not all(math.isfinite(value) for value in (radius, low, high)) or (
            error != 0 and (radius == 0 or low == estimate or high == estimate)
        ):
            raise AnalysisError(
                "numerical_failure", "A nonzero confidence endpoint shift is not representable."
            )
        rows.append([label, estimate, error, z, p_value, low, high, None, error == 0])
    table = m.frame(
        rows,
        columns=[
            "term",
            "estimate",
            "std_error",
            "z",
            "p_value",
            "ci_low",
            "ci_high",
            "df",
            "degenerate_variance",
        ],
    )
    return means, covariance, centered, critical, table


def _finish(name, metadata, settings, state, scores, labels, level, subjects, curve=None):
    means, covariance, centered, critical, effects = _infer(scores, labels, level)
    state.update(
        score_columns=labels,
        subject_scores=scores.tolist(),
        centered_subject_scores=centered.tolist(),
        estimates=means.tolist(),
        covariance_terms=labels,
        covariance_matrix=covariance.tolist(),
        score_sum_method="native Torch64-part bounded FastTwoSum floating expansion; final float64 rounding",
        score_sum_capacity=_SUM_CAPACITY,
        critical_value=critical,
        df=None,
        covariance_method="unbiased full subject-score sample covariance/n; HC1 factor n/(n-1)",
        inference="pointwise asymptotic normal for independent sampled subjects; no estimated-nuisance uncertainty, simultaneous bands, or finite-sample coverage claim",
        nuisance_source="caller-declared true externally known propensity/censoring law; augmentation functions fixed independently of analysis data",
        estimated_nuisance_supported=False,
        assumptions=[
            "independent identically distributed population sample and consistency/no interference",
            "potential event times conditionally exchangeable given pretreatment covariates",
            "the supplied propensity is the true conditional treatment probability",
            "all probability bounds and targets prespecified independently of analysis outcomes",
        ]
        + state.get("assumptions", []),
    )
    tables = {
        "effects": effects,
        "covariance": m.frame(covariance.tolist(), columns=labels, index=labels),
        "subjects": subjects,
        "scores": m.frame(scores.tolist(), columns=labels, index=metadata["positions"]),
    }
    if curve is not None:
        rows = [
            [t, float(means[3 * j]), float(means[3 * j + 1]), float(means[3 * j + 2])]
            for j, t in enumerate(curve)
        ]
        tables["curve"] = m.frame(
            rows, columns=["threshold", "survival_control", "survival_treated", "difference"]
        )
        state["sample_curve_outside_probability_range"] = bool(
            ((means[0::3] < 0) | (means[0::3] > 1) | (means[1::3] < 0) | (means[1::3] > 1)).any()
        )
        state["sample_curve_increases"] = [
            bool((values[1:] > values[:-1]).any()) for values in (means[0::3], means[1::3])
        ]
    return m.result(
        name,
        tables,
        metadata,
        settings,
        state,
        notes=[
            "Known-nuisance declarations cannot be established from predictions or balance diagnostics; in-sample estimated models are unsupported.",
            "Complete full-cohort scores retain cross-arm covariance; inference does not condition on observed arm counts.",
            "HT/AIPW score means and normal intervals are not clipped to survival/RMST bounds; a score curve need not be monotone.",
            "Zero empirical variance yields a point interval and undefined z/p; it does not establish population certainty.",
        ],
    )


def _subjects(metadata, times, events, assignment, p, probability, weights):
    return m.frame(
        list(
            zip(
                metadata["positions"],
                times.tolist(),
                events.tolist(),
                assignment.tolist(),
                p.tolist(),
                probability.tolist(),
                weights.tolist(),
                strict=True,
            )
        ),
        columns=[
            "position",
            "time",
            "event",
            "treatment",
            "propensity",
            "observed_treatment_probability",
            "treatment_weight",
        ],
    )


@m.procedure
def treatment_survival_ipcw(
    data,
    time,
    event,
    treatment,
    propensity,
    censor_survival,
    *,
    design,
    nuisance,
    thresholds,
    positivity=1e-6,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Fixed-grid causal survival score means with true known propensity and G.

    ``censor_survival`` is one distinct numeric column per fixed threshold, giving
    P(C>t|A,X) in that exact order. The score is I(A=a)*I(U>t)/(p_a*G).
    All cross-time and cross-arm covariance is retained. This is HT, not KM.
    """
    m.options(device, weights, max_work, level)
    positivity = _declarations(design, nuisance, missing, positivity)
    grid = _grid(thresholds)
    g_names = c.name_list(censor_survival, "censor_survival", minimum=1)
    if len(g_names) != len(grid):
        raise AnalysisError("invalid_spec", "One censor-survival column is required per threshold.")
    selected, metadata, times, events, assignment, p = _sample(
        data,
        time,
        event,
        treatment,
        propensity,
        [(name, "censor_survival") for name in g_names],
        missing,
        max_work,
        3 * len(grid),
    )
    probability, weights = _propensity(p, assignment, positivity)
    g = torch.stack([m.tensor(selected, name) for name in g_names], 1)
    if bool(((g < positivity) | (g > 1)).any()) or bool((g[:, 1:] > g[:, :-1]).any()):
        raise AnalysisError(
            "invalid_censor_survival",
            "Known G must lie in [positivity,1] and be nonincreasing within each subject.",
        )
    grid_tensor = torch.tensor(grid, dtype=torch.float64, device="cpu")
    indicator = (times[:, None] > grid_tensor).to(torch.float64)
    inverse_g = _divide(torch.ones_like(g), g, "inverse censor probability")
    adjusted = _multiply(
        _multiply(indicator, inverse_g, "censor-adjusted survival"),
        weights[:, None],
        "treatment-adjusted survival",
    )
    score0 = torch.where(assignment[:, None] == 0, adjusted, 0.0)
    score1 = torch.where(assignment[:, None] == 1, adjusted, 0.0)
    scores = torch.stack((score0, score1, score1 - score0), 2).reshape(len(times), -1)
    labels = [
        f"{term}@{t!r}"
        for t in grid
        for term in ("survival_control", "survival_treated", "difference")
    ]
    subjects = _subjects(metadata, times, events, assignment, p, probability, weights)
    for j, t in enumerate(grid):
        subjects[f"G@{t!r}"] = g[:, j].tolist()
    state = dict(
        target="S_a(t)=P(T(a)>t), with treated-minus-control survival contrast on a fixed grid",
        thresholds=grid,
        censor_survival=g.tolist(),
        inverse_censor_survival=inverse_g.tolist(),
        observed_survival_indicators=indicator.tolist(),
        strict_time_rule="U>t and G=P(C>t|A,X); time-zero events and tied censoring are excluded at that threshold",
        event_role="validated and retained; the IPCW survival indicator uses U=min(T,C), not event",
        assumptions=[
            "event and censor times independent conditional on pretreatment covariates and received treatment",
            "supplied G is the true conditional censor-survival probability at each fixed threshold; censor positivity at every grid time",
        ],
        estimator="Horvitz-Thompson score mean, not Kaplan-Meier; no monotonicity/probability projection",
    )
    settings = dict(
        time=time,
        event=event,
        treatment=treatment,
        propensity=propensity,
        censor_survival=g_names,
        thresholds=grid,
        design=design,
        nuisance=nuisance,
        positivity=positivity,
        level=level,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    return _finish(
        "treatment_survival_ipcw",
        metadata,
        settings,
        state,
        scores,
        labels,
        level,
        subjects,
        curve=grid,
    )


@m.procedure
def treatment_rmst_ipcw(
    data,
    time,
    event,
    treatment,
    propensity,
    censor_rate,
    *,
    design,
    nuisance,
    tau,
    positivity=1e-6,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Causal RMST HT means under a known conditional exponential censor law.

    For rate lambda>=0, G(t|A,X)=exp(-lambda*t) for t<tau. Left-limit positivity at
    tau allows administrative censoring at tau. The subject
        integral is expm1(lambda*min(U,tau))/lambda, with the exact lambda=0 limit.
        No event-time distribution or censoring model is fitted.
    """
    m.options(device, weights, max_work, level)
    positivity = _declarations(design, nuisance, missing, positivity)
    tau = c.check_number(tau, "tau", minimum=0, exclusive=True)
    selected, metadata, times, events, assignment, p = _sample(
        data,
        time,
        event,
        treatment,
        propensity,
        [(censor_rate, "censor_rate")],
        missing,
        max_work,
        3,
    )
    probability, weights = _propensity(p, assignment, positivity)
    rate = m.tensor(selected, censor_rate)
    if bool((rate < 0).any()):
        raise AnalysisError(
            "invalid_censor_rate", "Known conditional censor rates must be nonnegative."
        )
    if bool(((rate == 0) & (events == 0) & (times < tau)).any()):
        raise AnalysisError(
            "invalid_censor_rate",
            "A zero censor hazard through tau cannot produce censoring before tau.",
        )
    hazard = _multiply(rate, tau, "censor hazard at tau")
    g_tau = torch.exp(-hazard)
    if bool((g_tau < positivity).any()):
        raise AnalysisError(
            "positivity_violation",
            "Known censor survival at tau violates positivity; no rate clipping is performed.",
        )
    capped = times.clamp(max=tau)
    exposure = _multiply(rate, capped, "observed censor hazard")
    integral = capped.clone()
    positive = rate > 0
    integral[positive] = _divide(
        torch.expm1(exposure[positive]), rate[positive], "censor-adjusted time integral"
    )
    adjusted = _multiply(integral, weights, "treatment-adjusted RMST")
    score0, score1 = (
        torch.where(assignment == 0, adjusted, 0.0),
        torch.where(assignment == 1, adjusted, 0.0),
    )
    scores = torch.stack((score0, score1, score1 - score0), 1)
    subjects = _subjects(metadata, times, events, assignment, p, probability, weights)
    for name, values in [
        ("censor_rate", rate),
        ("censor_survival_tau_left", g_tau),
        ("capped_observed_time", capped),
        ("censor_adjusted_integral", integral),
    ]:
        subjects[name] = values.tolist()
    state = dict(
        target="E[min(T(1),tau)]-E[min(T(0),tau)] in the sampled superpopulation",
        tau=tau,
        conditional_censor_rates=rate.tolist(),
        censor_survival_tau_left=g_tau.tolist(),
        capped_observed_times=capped.tolist(),
        subject_integrals=integral.tolist(),
        censor_model="true known G(t|A,X)=exp(-lambda*t) for0<=t<tau; G(tau-)=exp(-lambda*tau); administrative censoring at tau allowed; zero rate means no censoring before tau",
        integral_method="native expm1(lambda*min(U,tau))/lambda; exact zero-rate min(U,tau) limit",
        event_role="validated and retained; the integral of I(U>t)/G(t) uses U=min(T,C), not event",
        assumptions=[
            "event and censor times independent conditional on pretreatment covariates and treatment",
            "conditional censor survival follows the supplied true known exponential law through tau",
        ],
        estimator="Horvitz-Thompson mean of analytically integrated known-IPCW subject scores; no Kaplan-Meier/extrapolated event model",
    )
    settings = dict(
        time=time,
        event=event,
        treatment=treatment,
        propensity=propensity,
        censor_rate=censor_rate,
        tau=tau,
        design=design,
        nuisance=nuisance,
        positivity=positivity,
        level=level,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    return _finish(
        "treatment_rmst_ipcw",
        metadata,
        settings,
        state,
        scores,
        ["rmst_control", "rmst_treated", "difference"],
        level,
        subjects,
    )


@m.procedure
def treatment_rmst_aipw(
    data,
    time,
    event,
    treatment,
    propensity,
    m0,
    m1,
    *,
    design,
    nuisance,
    tau,
    positivity=1e-6,
    level=0.95,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """AIPW capped survival means with known p and fixed external augmentation.

    Every capped event time must be fully observed: event=1 or U>=tau. m0/m1 are
    externally prespecified functions in [0,tau], not models fitted on these data;
    they may be misspecified because the true known propensity identifies the target.
    """
    m.options(device, weights, max_work, level)
    positivity = _declarations(design, nuisance, missing, positivity)
    tau = c.check_number(tau, "tau", minimum=0, exclusive=True)
    selected, metadata, times, events, assignment, p = _sample(
        data, time, event, treatment, propensity, [(m0, "m0"), (m1, "m1")], missing, max_work, 3
    )
    probability, weights = _propensity(p, assignment, positivity)
    if bool(((events == 0) & (times < tau)).any()):
        raise AnalysisError(
            "incomplete_horizon",
            "AIPW RMST requires event=1 or observed follow-up>=tau for every subject; early censoring is unsupported.",
        )
    predictions = torch.stack([m.tensor(selected, name) for name in (m0, m1)], 1)
    if bool(((predictions < 0) | (predictions > tau)).any()):
        raise AnalysisError(
            "invalid_augmentation",
            "Fixed external capped-time predictions must lie within [0,tau].",
        )
    capped = times.clamp(max=tau)
    arm_scores = []
    for arm in (0, 1):
        residual = torch.where(assignment == arm, capped - predictions[:, arm], 0.0)
        correction = _multiply(residual, weights, "AIPW residual correction")
        values = _finite(predictions[:, arm] + correction, "augmented RMST scores")
        if bool(((correction != 0) & (values == predictions[:, arm])).any()):
            raise AnalysisError(
                "numerical_failure",
                "A nonzero augmentation correction was absorbed by its prediction.",
            )
        arm_scores.append(values)
    scores = torch.stack(
        (*arm_scores, _finite(arm_scores[1] - arm_scores[0], "AIPW contrast scores")), 1
    )
    subjects = _subjects(metadata, times, events, assignment, p, probability, weights)
    subjects["capped_event_time"] = capped.tolist()
    subjects["m0"] = predictions[:, 0].tolist()
    subjects["m1"] = predictions[:, 1].tolist()
    state = dict(
        target="E[min(T(1),tau)]-E[min(T(0),tau)] under complete capped-time observation",
        tau=tau,
        capped_event_times=capped.tolist(),
        fixed_augmentation=predictions.tolist(),
        augmentation_source="functions fixed externally and independently of analysis data; correctness of m0/m1 is not required with true known p",
        assumptions=[
            "every restricted event time is observed: event=1 or U>=tau",
            "m0/m1 are external prespecified functions, held fixed in this population-sampling inference",
        ],
        estimator="mean of m_a(X)+I(A=a)/p_a(X)*(min(T,tau)-m_a(X))",
        estimated_nuisance_double_robust_inference=False,
    )
    settings = dict(
        time=time,
        event=event,
        treatment=treatment,
        propensity=propensity,
        m0=m0,
        m1=m1,
        tau=tau,
        design=design,
        nuisance=nuisance,
        positivity=positivity,
        level=level,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    return _finish(
        "treatment_rmst_aipw",
        metadata,
        settings,
        state,
        scores,
        ["rmst_control", "rmst_treated", "difference"],
        level,
        subjects,
    )
