"""Full finite conditional exponential families; no asymptotic or sampled fallback."""

from __future__ import annotations

from itertools import combinations
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.finite import common as f
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace
from .codec import intact, seal


def number(value, name):
    return c.check_number(value, name, minimum=-40, maximum=40)


def plan(operation, n, candidates, max_work):
    c.check_count(max_work, "max_work")
    # Includes enumeration/constraint evaluation, 3 root searches and full response covariance.
    work = candidates * (n * n + 10 * n + 300)
    if work > max_work:
        raise AnalysisError("work_limit", f"Structural work {work} exceeds max_work={max_work}.")
    resource = plan_workspace(
        operation,
        {
            "allocation_lists_tensors_and_complete_json": candidates * (512 * n + 1024),
            "grouped_support_and_root_traces": 512 * candidates + 512 * 300,
            "full_response_covariance_and_sample": 512 * n * n + 1024 * n,
        },
    ).record()
    return {"resource_plan": resource, "planned_work": work, "max_work": int(max_work)}


def compositions(total, n):
    if n == 1:
        yield (total,)
    else:
        for count in range(total + 1):
            for rest in compositions(total - count, n - 1):
                yield (count, *rest)


def tensor(values):
    return torch.tensor(values, dtype=torch.float64, device="cpu")


def distribution(state, beta):
    support, base = tensor(state["support"]), tensor(state["log_base"])
    logp = base + beta * support
    logp -= torch.logsumexp(logp, 0)
    return support, logp, logp.exp()


def root(function, target, increasing, what):
    lo, hi = -40.0, 40.0
    a, b = function(lo), function(hi)
    if not (a <= target <= b if increasing else b <= target <= a):
        raise AnalysisError(
            "unbracketed_root", f"{what} is outside the declared coefficient [-40,40] domain."
        )
    trace = []
    for iteration in range(50):
        mid = (lo + hi) / 2
        value = function(mid)
        trace.append([iteration, lo, hi, mid, value])
        if (value < target) == increasing:
            lo = mid
        else:
            hi = mid
    value = (lo + hi) / 2
    if abs(function(value) - target) > 1e-8 * max(1, abs(target)):
        raise AnalysisError("nonconvergence", f"{what} root failed its residual check.")
    return value, {
        "target": target,
        "increasing": increasing,
        "trace": trace,
        "residual": function(value) - target,
        "bracket": [lo, hi],
    }


def fit(family, data, y, x, nuisance, exposure, missing, max_states, max_work, device, weights):
    f.domain(device, weights)
    c.check_name(y, "y")
    c.check_name(x, "x")
    nuisance = c.name_list(nuisance, "nuisance", minimum=0)
    if len(nuisance) > 3:
        raise AnalysisError(
            "resource_limit", "At most three integer nuisance columns are supported."
        )
    names = [y, x, *nuisance]
    if exposure is not None:
        c.check_name(exposure, "exposure")
        names.append(exposure)
    if len(set(names)) != len(names):
        raise AnalysisError(
            "invalid_spec", "Outcome, target, nuisance and exposure roles must be distinct."
        )
    c.check_count(max_states, "max_states")
    if max_states > 20000:
        raise AnalysisError(
            "resource_limit", "max_states cannot exceed 20000 structural allocations."
        )
    frame, state = f.sample(data, names, missing, 16 if family == "logit" else 8)
    raw = c.matrix(frame, names)
    n = len(frame)
    yy, xx = raw[:, 0], raw[:, 1]
    zz = raw[:, 2 : 2 + len(nuisance)]
    integers = raw[:, : 2 + len(nuisance)]
    if not bool(torch.isfinite(raw).all()):
        raise AnalysisError("non_finite_values", "All analysed values must be finite.")
    if not bool((integers == integers.round()).all()):
        raise AnalysisError(
            "invalid_integer", "Outcome and target/nuisance columns must be explicit integers."
        )
    if bool((raw[:, 1 : 2 + len(nuisance)].abs() > 8).any()):
        raise AnalysisError(
            "resource_limit", "Target/nuisance integer values must lie within +/-8."
        )
    if family == "logit":
        if not bool(((yy == 0) | (yy == 1)).all()):
            raise AnalysisError("invalid_outcome", "Bernoulli outcomes must be numeric zero/one.")
    elif bool((yy < 0).any()) or float(yy.sum()) > 24:
        raise AnalysisError(
            "invalid_outcome", "Poisson counts must be nonnegative with total <=24."
        )
    ee = raw[:, -1] if exposure is not None else torch.ones(n, dtype=torch.float64, device="cpu")
    if bool(((ee < 1e-12) | (ee > 1e12)).any()):
        raise AnalysisError(
            "invalid_exposure", "Known positive exposures must lie in [1e-12,1e12]."
        )
    total = int(yy.sum())
    count = math.comb(n, total) if family == "logit" else math.comb(total + n - 1, n - 1)
    if count > max_states:
        raise AnalysisError(
            "state_limit",
            f"Pre-filter structural allocations {count} exceed max_states={max_states}.",
        )
    settings = plan(f"exact_{family}_fit", n, count, max_work)
    design = torch.cat(
        [torch.ones((n, 1), dtype=torch.float64, device="cpu"), xx[:, None], zz], dim=1
    )
    f.condition(design)
    response = [int(v) for v in yy.tolist()]
    target = [int(v) for v in xx.tolist()]
    nuisance_values = [[int(v) for v in row] for row in zz.tolist()]
    constraint = [
        sum(response[i] * nuisance_values[i][j] for i in range(n)) for j in range(len(nuisance))
    ]
    observed = sum(response[i] * target[i] for i in range(n))
    if family == "logit":

        def generator():
            for active in combinations(range(n), total):
                allocation = [0] * n
                for i in active:
                    allocation[i] = 1
                yield tuple(allocation)

        candidates = generator()
    else:
        candidates = compositions(total, n)
    allocations, stats, logs = [], [], []
    log_exposure = ee.log().tolist()
    for allocation in candidates:
        if any(
            sum(allocation[i] * nuisance_values[i][j] for i in range(n)) != constraint[j]
            for j in range(len(nuisance))
        ):
            continue
        allocations.append(list(allocation))
        stats.append(sum(allocation[i] * target[i] for i in range(n)))
        logs.append(
            0.0
            if family == "logit"
            else sum(
                allocation[i] * log_exposure[i] - math.lgamma(allocation[i] + 1) for i in range(n)
            )
        )
    support = sorted(set(stats))
    if len(support) < 2:
        raise AnalysisError(
            "unidentified_conditional", "Conditioned sufficient statistic has no variation."
        )
    # Complete aggregated weights; allocations sharing S are generally not equiprobable.
    stats_tensor, log_tensor = tensor(stats), tensor(logs)
    base = [float(torch.logsumexp(log_tensor[stats_tensor == s], 0)) for s in support]
    state.update(
        family=family,
        y=y,
        x=x,
        nuisance=nuisance,
        exposure=exposure,
        response=response,
        target=target,
        nuisance_values=nuisance_values,
        exposures=ee.tolist(),
        conditioned_total=total,
        conditioned_nuisance=constraint,
        observed_statistic=observed,
        support=support,
        log_base=base,
        allocations=allocations,
        allocation_statistics=stats,
        allocation_log_base=logs,
        structural_allocations=count,
        feasible_allocations=len(allocations),
        settings=settings,
        numerical_domain=[-40, 40],
        max_states=max_states,
    )
    boundary = (
        "lower" if observed == support[0] else "upper" if observed == support[-1] else "interior"
    )
    if boundary == "interior":
        beta, trace = root(
            lambda b: float(distribution(state, b)[0] @ distribution(state, b)[2]),
            observed,
            True,
            "CMLE",
        )
        ss, logp, pp = distribution(state, beta)
        information = float(((ss - observed) ** 2) @ pp)
        ll = float(logp[support.index(observed)])
    else:
        beta, trace, information, ll = (
            None,
            {"reason": "infinite CMLE on support boundary", "trace": []},
            0.0,
            0.0,
        )
    state.update(
        coefficient=beta, boundary=boundary, information=information, cmle_diagnostics=trace
    )
    return seal(
        f"exact_{family}_fit",
        {
            "coefficients": table(
                [[x, beta, boundary, information, ll]],
                columns=[
                    "term",
                    "coefficient",
                    "cmle_boundary",
                    "conditional_information",
                    "statistic_loglikelihood",
                ],
            ),
            "support": table(
                [[s, b] for s, b in zip(support, base)], columns=["statistic", "log_base_measure"]
            ),
            "conditioning": table(
                [["total", total]] + [[name, v] for name, v in zip(nuisance, constraint)],
                columns=["constraint", "observed"],
            ),
        },
        state,
        inference="Single target conditional CMLE; no Wald SE/covariance/p/CI or nuisance estimates.",
        **settings,
    )


def checked(result, family, max_work, operation):
    state = intact(result, f"exact_{family}_fit")
    settings = plan(operation, len(state["response"]), state["structural_allocations"], max_work)
    return state, settings


def intervals(result, family, level, max_work):
    level = f.confidence(level)
    state, settings = checked(result, family, max_work, f"exact_{family}_ci")
    observed, support = state["observed_statistic"], tensor(state["support"])
    target = math.log((1 - level) / 2)
    roots = {}
    bounds = []
    for which, mask, increasing, extreme in [
        ("lower", support >= observed, True, observed == state["support"][0]),
        ("upper", support <= observed, False, observed == state["support"][-1]),
    ]:
        if extreme:
            bounds.append(None)
            roots[which] = {
                "boundary": "negative_infinity" if which == "lower" else "positive_infinity",
                "trace": [],
            }
        else:
            value, trace = root(
                lambda b: float(torch.logsumexp(distribution(state, b)[1][mask], 0)),
                target,
                increasing,
                f"{which} confidence bound",
            )
            bounds.append(value)
            roots[which] = trace
    output_state = {"fit_state": state, "level": level, "root_diagnostics": roots}
    return seal(
        f"exact_{family}_ci",
        {
            "intervals": table(
                [
                    [
                        state["x"],
                        level,
                        *bounds,
                        roots["lower"].get("boundary", "finite"),
                        roots["upper"].get("boundary", "finite"),
                    ]
                ],
                columns=["term", "level", "lower", "upper", "lower_boundary", "upper_boundary"],
            )
        },
        output_state,
        inference="Inclusive central equal-tail exact conditional coefficient interval; no mid-p.",
        **settings,
    )


def test(result, family, null, max_work):
    null = number(null, "null")
    state, settings = checked(result, family, max_work, f"exact_{family}_test")
    support, logp, pp = distribution(state, null)
    observed_index = state["support"].index(state["observed_statistic"])
    critical = logp <= logp[observed_index] + 1e-12
    p = min(1.0, float(pp[critical].sum()))
    return seal(
        f"exact_{family}_test",
        {
            "test": table(
                [[state["x"], null, state["observed_statistic"], float(pp[observed_index]), p]],
                columns=[
                    "term",
                    "null_coefficient",
                    "observed_statistic",
                    "observed_probability",
                    "p_value",
                ],
            ),
            "null_distribution": table(
                [
                    [s, float(lp), float(pr), bool(k)]
                    for s, lp, pr, k in zip(state["support"], logp, pp, critical)
                ],
                columns=["statistic", "log_probability", "probability", "critical_region"],
            ),
        },
        {"fit_state": state, "null": null, "log_probability_tie_tolerance": 1e-12},
        inference="Probability-ordered two-sided conditional test of grouped sufficient statistic, inclusive ties.",
        **settings,
    )


def moments(result, family, coefficient, max_work):
    state, settings = checked(result, family, max_work, f"exact_{family}_moments")
    plugin = coefficient is None
    beta = state["coefficient"] if plugin else number(coefficient, "coefficient")
    allocations = tensor(state["allocations"])
    logbase = tensor(state["allocation_log_base"])
    stats = tensor(state["allocation_statistics"])
    boundary = state["boundary"] if plugin else "interior"
    if beta is None:
        face = stats == (state["support"][0] if boundary == "lower" else state["support"][-1])
        logbase = logbase[face]
        active = allocations[face]
        pp = torch.softmax(logbase, 0)
        full = torch.zeros(len(allocations), dtype=torch.float64, device="cpu")
        full[face] = pp
    else:
        active = allocations
        pp = torch.softmax(logbase + beta * stats, 0)
        full = pp
    mean = pp @ active
    centered = active - mean
    covariance = centered.T @ (centered * pp[:, None])
    sd = covariance.diag().clamp_min(0).sqrt()
    aligned = {pos: i for i, pos in enumerate(state["sample_positions"])}
    labels = dict(zip(state["sample_positions"], state["sample_labels"]))
    labels.update(zip(state["dropped_positions"], state["dropped_labels"]))
    rows = [
        [
            pos,
            labels[pos],
            pos in aligned,
            float(mean[aligned[pos]]) if pos in aligned else None,
            float(sd[aligned[pos]]) if pos in aligned else None,
        ]
        for pos in range(state["original_n"])
    ]
    return seal(
        f"exact_{family}_moments",
        {
            "response_moments": table(
                rows,
                columns=[
                    "original_position",
                    "row_label",
                    "in_sample",
                    "conditional_mean",
                    "conditional_response_sd",
                ],
            ),
            "response_covariance": table(
                covariance.tolist(),
                columns=[f"position_{p}" for p in state["sample_positions"]],
                index=[f"position_{p}" for p in state["sample_positions"]],
            ),
        },
        {
            "fit_state": state,
            "coefficient": beta,
            "boundary": boundary,
            "evaluation": "CMLE plugin" if plugin else "known coefficient",
            "allocation_probabilities": full.tolist(),
            "full_response_covariance": covariance.tolist(),
        },
        inference="Conditional training response distribution; SD is response spread, not parameter/mean estimation SE. No new-row prediction.",
        **settings,
    )
