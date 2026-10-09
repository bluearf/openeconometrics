"""Explicit hierarchy-preserving backward Poisson interaction selection."""

from itertools import combinations
import math

from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import summary_state
from openecon.resources import plan_workspace, workspace_budget_bytes
from .loglinear import _digest, _error, _integer, _names, _options, loglinear_ipf
from .sampling import _saved


def _maximal(terms, dimensions):
    return [
        list(term)
        for term in sorted(terms, key=lambda t: (len(t), tuple(dimensions.index(d) for d in t)))
        if not any(set(term) < set(other) for other in terms)
    ]


def _generators(dimensions, margins):
    # Parse at most 32 tiny dimension sets without expanding cell contrasts.
    if not isinstance(margins, (list, tuple)) or not 1 <= len(margins) <= 32:
        _error("margins must declare 1..32 generating dimension sets.")
    groups = []
    for margin in margins:
        group = _names(margin, "generating margin", 1, len(dimensions))
        if any(d not in dimensions for d in group):
            _error("Margins must use declared dimensions.")
        canonical = tuple(d for d in dimensions if d in group)
        if canonical in groups:
            _error("Repeated generating margin.")
        groups.append(canonical)
    if set().union(*map(set, groups)) != set(dimensions):
        _error("Every dimension must occur in generating margins.")
    return [g for g in groups if not any(set(g) < set(h) for h in groups)]


@resident_cpu
def loglinear_select(
    data,
    dimensions,
    count,
    *,
    levels,
    margins,
    criterion="bic",
    structural=None,
    offset=None,
    max_iter=1000,
    tol=1e-9,
    level=0.95,
    max_models=64,
    max_work=300_000_000,
    max_bytes=128 * 1024**2,
):
    """Backward AIC/BIC search dropping only maximal hierarchical interactions.

    Starts at explicit generating margins; retains all main effects. Every
    eligible deletion is fitted and retained, including nonselected candidates.
    Failures refuse the search. BIC uses total independently sampled persons
    (sum of Poisson counts) as its declared sample convention. The returned
    selected fit is complete; its Wald intervals are not selection-adjusted.
    """
    dimensions = _names(dimensions, "dimensions", 2, 4)
    max_iter, tol, level, max_work, budget = _options(max_iter, tol, level, max_work, max_bytes)
    max_models = _integer(max_models, "max_models", 2, 256)
    if criterion not in ("aic", "bic"):
        _error("criterion must be 'aic' or 'bic'.")
    if not isinstance(levels, dict) or set(levels) != set(dimensions):
        _error("Declare all dimension levels.")
    # Bound category counts/grid before enumerating any cell contrasts.
    cells = 1
    for d in dimensions:
        seq = levels[d]
        if not isinstance(seq, (list, tuple)) or not 2 <= len(seq) <= 16:
            _error("Each dimension requires 2..16 levels.")
        cells *= len(seq)
    if cells > 4096:
        _error("Complete selection grid exceeds 4096 cells.", "resource_limit")
    generators = _generators(dimensions, margins)
    terms = {s for g in generators for k in range(1, len(g) + 1) for s in combinations(g, k)}
    removable = sum(len(t) > 1 for t in terms)
    # Admit a conservative whole-search plan before fitting any candidate.
    # The candidate count is bounded by the initial term count at every step.
    planned_models = 1 + removable * (removable + 1) // 2
    if planned_models > max_models:
        _error(
            f"Complete worst-case search requires {planned_models} models; max_models={max_models}.",
            "resource_limit",
        )
    p = 1 + sum(math.prod(len(levels[d]) - 1 for d in term) for term in terms)
    if p > 128:
        _error("Initial hierarchical design exceeds 128 parameters.", "resource_limit")
    single_work = (
        cells * p * p * (max_iter + 8) + (max_iter + 4) * p**3 + cells * len(generators) * max_iter
    )
    search_work = planned_models * single_work
    if search_work > max_work:
        _error(f"Complete planned selection work {search_work} exceeds max_work.", "resource_limit")
    resource = plan_workspace(
        "loglinear complete backward selection",
        {
            "retained complete candidate states": planned_models
            * (8 * cells * (p + 24) + 32 * p * p + 64 * max_iter + 4096),
            "largest active candidate buffers": 8 * cells * (8 * p + 24)
            + 8 * 12 * p * p
            + 8 * 6 * max_iter,
        },
        budget_bytes=min(budget, workspace_budget_bytes()),
    ).record()
    candidate_states = []
    trace = []
    decisions = []

    def fit(model_terms, step, deleted):
        generating = _maximal(model_terms, dimensions)
        result = loglinear_ipf(
            data,
            dimensions,
            count,
            levels=levels,
            margins=generating,
            structural=structural,
            offset=offset,
            max_iter=max_iter,
            tol=tol,
            level=level,
            max_work=max_work,
            max_bytes=budget,
        )
        rank = len(result.attrs["state"]["terms"])
        ll = result.attrs["log_likelihood"]
        n = result.attrs["total_count"]
        penalty = 2.0 if criterion == "aic" else math.log(n)
        value = -2 * ll + penalty * rank
        idx = len(candidate_states)
        candidate_states.append(summary_state(result))
        trace.append(
            [
                float(idx),
                float(step),
                ":".join(deleted) if deleted else "initial",
                float(rank),
                ll,
                value,
                False,
            ]
        )
        return result, value, idx

    current, current_value, current_id = fit(terms, 0, None)
    trace[current_id][-1] = True
    step = 0
    while True:
        eligible = [tuple(t) for t in _maximal(terms, dimensions) if len(t) > 1]
        if not eligible:
            termination = "main_effects_only"
            break
        step += 1
        candidates = []
        for deleted in eligible:
            remaining = terms - {deleted}
            result, value, idx = fit(remaining, step, deleted)
            candidates.append((value, idx, deleted, result, remaining))
        winner = min(candidates, key=lambda c: (c[0], c[1]))
        value, idx, deleted, result, remaining = winner
        if value >= current_value - 1e-8:
            decisions.append(
                [float(step), float(current_id), float(idx), current_value, value, False]
            )
            termination = "no_criterion_improvement"
            break
        decisions.append([float(step), float(current_id), float(idx), current_value, value, True])
        trace[idx][-1] = True
        current, current_value, current_id, terms = result, value, idx, remaining
    state = dict(
        schema="openecon.loglinear-selection.v1",
        dimensions=dimensions,
        criterion=criterion,
        start_margins=[list(g) for g in generators],
        selected_margins=_maximal(terms, dimensions),
        selected_candidate=current_id,
        selected_summary=summary_state(current),
        candidates=candidate_states,
        trace=trace,
        decisions=decisions,
        termination=termination,
        planned_models=planned_models,
        estimated_work=search_work,
    )
    state["sha256"] = _digest(state)
    return _saved(
        TableSet(
            {
                "candidates": table(
                    trace,
                    columns=[
                        "candidate",
                        "step",
                        "removed_interaction",
                        "rank",
                        "log_likelihood",
                        criterion,
                        "accepted_on_path",
                    ],
                ),
                "decisions": table(
                    decisions,
                    columns=[
                        "step",
                        "from_candidate",
                        "best_candidate",
                        "from_criterion",
                        "best_criterion",
                        "accepted",
                    ],
                ),
                "selected_parameters": current["parameters"].copy(),
                "selected_covariance": current["covariance"].copy(),
                "selected_cells": current["cells"].copy(),
                "selected_goodness_of_fit": current["goodness_of_fit"].copy(),
            },
            title="Hierarchical Poisson backward interaction selection",
            criterion=criterion,
            selected_margins=state["selected_margins"],
            selected_candidate=current_id,
            selected_criterion=current_value,
            evaluated_models=len(candidate_states),
            termination=termination,
            state=state,
            resource={**resource, "estimated_work": search_work, "planned_models": planned_models},
            sample_positions=current.attrs["sample_positions"],
            nobs=current.attrs["nobs"],
            device="cpu",
            precision="float64",
            sampling="independent_poisson_cells",
            inference="Selected-model inference is not post-selection calibrated; AIC/BIC search is greedy, not exhaustive-global selection",
            bic_sample_convention="total Poisson count (persons); conditions outside this convention are unsupported",
        )
    )
