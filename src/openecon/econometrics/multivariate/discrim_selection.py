"""Bounded singleton Wilks LDA selection; screening is not selective inference.

References: SAS/STAT STEPDISC (2024.12), pp. 9725–9739,
https://documentation.sas.com/api/docsets/statug/v_035/content/stepdisc.pdf?locale=en
and https://www.stata.com/support/faqs/statistics/stepwise-regression-problems/ .
The candidate test is the prespecified ANCOVA reference F, not a valid
post-selection p-value. No cross-validation or selected-model tests are reported.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.resources import plan_workspace

from . import common as c
from . import discrim_options as o
from .discrim import _priors, _softmax_rows

MAX_ROWS = 100_000
MAX_VARIABLES = 32
MAX_GROUPS = 32
MAX_STEPS = 128
MAX_WORK = 2_000_000_000
MAX_CONDITION = 1e10
_CANDIDATE_COLUMNS = ["step", "action", "variable", "model_size", "partial_lambda",
                      "partial_r_squared", "statistic", "df1", "df2", "reference_p"]
_HISTORY_COLUMNS = ["step", "action", "variable", "model_size", "wilks_lambda",
                    "statistic", "df1", "df2", "reference_p"]


def _plan(p: int, groups: int, rows: int, steps: int, *, prediction: bool = False) -> dict:
    if p > MAX_VARIABLES or groups > MAX_GROUPS or rows > MAX_ROWS:
        raise AnalysisError("workspace_limit", "Stepwise LDA supports at most 32 candidate "
                            "variables, 32 groups and 100000 resident rows.")
    # Every candidate assessment includes a selected-matrix solve and group SVD.
    evaluations = 0 if prediction else (steps+1)*2*p
    work = rows*(4*p*p+groups*p*4) + evaluations*(8*p**3+8*groups**3)
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Stepwise candidate evaluation exceeds the work "
                            "budget; reduce variables, groups, rows or max_steps.")
    return plan_workspace("stepwise discriminant selection", {
        "resident_rows_and_moments": rows*(p+3)*128,
        "group_matrices": 32*(groups+2)*p*p*8,
        "candidate_trace_and_saved_state": evaluations*1024,
        "prediction_and_classification": rows*groups*64,
    }).record() | {"candidate_evaluations_upper_bound": evaluations, "work_upper_bound": work}


def _checked_within(within: Tensor) -> tuple[Tensor, Tensor]:
    if not bool(torch.isfinite(within).all()) or bool((within.diagonal() <= 0).any()):
        raise AnalysisError("singular_matrix", "All candidate variables need positive pooled "
                            "within-group variance and finite moments.")
    scale = within.diagonal().sqrt()
    correlation = within/torch.outer(scale, scale)
    c.positive_definite(correlation, "full candidate within-group correlation", "Remove redundant candidates.")
    eigenvalues = torch.linalg.eigvalsh(correlation)
    if float(eigenvalues[-1]/eigenvalues[0]) > MAX_CONDITION:
        raise AnalysisError("singular_matrix", "Full candidate within-group correlation is "
                            "too ill-conditioned for stepwise screening.")
    return correlation, scale


def _center_means(means: Tensor, counts: Tensor) -> Tensor:
    grand = (counts[:, None]*means).sum(0)/sum(int(v) for v in counts.tolist())
    grand += (counts[:, None]*(means-grand)).sum(0)/sum(int(v) for v in counts.tolist())
    return means-grand


def _candidate(within: Tensor, means: Tensor, counts: Tensor,
               retained: list[int], candidate: int) -> list[float]:
    """Independent conditional contribution of a singleton; G-1 and N-G-k df.

    Residualize on the *within* regression. The class residual SSCP is evaluated
    through (I+A Wss^-1 A')^-1 with a complete group SVD, avoiding subtraction
    of two almost equal total/error determinants or squared residual sums.
    """
    groups = len(counts)
    n = sum(int(v) for v in counts.tolist())
    df1, df2 = groups-1, n-groups-len(retained)
    if df2 < 1:
        raise AnalysisError("insufficient_observations", "Conditional ANCOVA requires N-G-k > 0.")
    centered = _center_means(means, counts)
    response = centered[:, candidate]
    error = within[candidate, candidate]
    if retained:
        ss = within[retained][:, retained]
        cross = within[retained, candidate]
        factor = torch.linalg.cholesky(ss)
        slope = torch.cholesky_solve(cross[:, None], factor).flatten()
        error = error-cross.dot(slope)
        response = response-centered[:, retained]@slope
        a = counts.sqrt()[:, None]*centered[:, retained]
        whitened = torch.linalg.solve_triangular(factor, a.T, upper=False).T
        left, singular, _ = torch.linalg.svd(whitened, full_matrices=True)
        transformed = left.T@(counts.sqrt()*response)
        denominator = torch.ones(groups, dtype=c.FLOAT)
        denominator[:len(singular)] += singular.square()
        hypothesis = (transformed.square()/denominator).sum()
    else:
        hypothesis = (counts*response.square()).sum()
    ratio = float(hypothesis/error)
    statistic = ratio*df2/df1
    if float(error) <= 0 or not all(math.isfinite(v) for v in (ratio, statistic)) or ratio < 0:
        raise AnalysisError("numerical_failure", "Conditional screening moments are unresolved; rescale inputs.")
    log_ratio = math.log1p(ratio)
    partial = math.exp(-log_ratio)
    r_squared = -math.expm1(-log_ratio)
    probability = c.f_upper(statistic, df1, df2)
    if probability is None:
        raise AnalysisError("numerical_failure", "Conditional reference probability is undefined.")
    return [partial, r_squared, statistic, df1, df2, probability]


def _wilks(within: Tensor, means: Tensor, counts: Tensor, selected: list[int]) -> float:
    if not selected:
        return 1.0
    factor = torch.linalg.cholesky(within[selected][:, selected])
    a = counts.sqrt()[:, None]*_center_means(means, counts)[:, selected]
    whitened = torch.linalg.solve_triangular(factor, a.T, upper=False).T
    singular = torch.linalg.svdvals(whitened)
    return math.exp(-float(torch.log1p(singular.square()).sum()))


def _selection(within: Tensor, means: Tensor, counts: Tensor, names: list[str],
               method: str, include: list[str], p_enter: float, p_remove: float,
               max_steps: int) -> tuple[list[int], list[list], list[list]]:
    forced = {names.index(name) for name in include}
    selected = list(range(len(names))) if method == "backward" else sorted(forced)
    visited = {tuple(selected)}
    candidates, history = [], []
    while True:
        step, chosen, action = len(history)+1, None, None
        if method != "forward":
            removal = []
            for j in selected:
                if j in forced:
                    continue
                values = _candidate(within, means, counts, [i for i in selected if i != j], j)
                candidates.append([step, "remove", names[j], len(selected), *values])
                removal.append((j, values))
            # Order by the statistic, preserving strength when tail probabilities
            # underflow together. Strict comparisons retain first input order.
            if removal:
                chosen = removal[0]
                for item in removal[1:]:
                    if item[1][2] < chosen[1][2]:
                        chosen = item
                if chosen[1][-1] > p_remove:
                    action = "remove"
                else:
                    chosen = None
        if chosen is None and method != "backward":
            entry = []
            for j in range(len(names)):
                if j in selected:
                    continue
                values = _candidate(within, means, counts, selected, j)
                candidates.append([step, "enter", names[j], len(selected), *values])
                entry.append((j, values))
            if entry:
                chosen = entry[0]
                for item in entry[1:]:
                    if item[1][2] > chosen[1][2]:
                        chosen = item
                if chosen[1][-1] < p_enter:
                    action = "enter"
                else:
                    chosen = None
        if chosen is None:
            return selected, candidates, history
        if len(history) >= max_steps:
            raise AnalysisError("non_convergence", "Selection reached max_steps before satisfying its stopping criteria.")
        j, values = chosen
        selected = sorted([*selected, j]) if action == "enter" else [i for i in selected if i != j]
        if tuple(selected) in visited:
            raise AnalysisError("non_convergence", "Stepwise selection revisited a model; cycling is refused.")
        visited.add(tuple(selected))
        history.append([step, action, names[j], len(selected),
                        _wilks(within, means, counts, selected), values[2], *values[3:]])


def _moments(data: Any, group: str, names: list[str], weights: str | None,
             missing: str, steps: int) -> tuple[dict, Tensor, Tensor, Tensor, dict]:
    used = [group, *names, *([weights] if weights is not None else [])]
    if len(set(used)) != len(used):
        raise AnalysisError("invalid_spec", "Group, candidate and weight roles must be distinct.")
    data = o._project_columns(data, used)
    rows = o._resident_rows(data)
    _plan(len(names), 0, rows, steps)
    sample, _, dropped = c.select(o._project_records(data, used), used,
                                  numeric=[*names, *([weights] if weights is not None else [])], missing=missing)
    if weights is None:
        w = torch.ones(len(sample), dtype=c.FLOAT)
    else:
        if bool((sample[weights] > o.EXACT_COUNT).any()):
            raise AnalysisError("invalid_weights", "Frequency weights exceed exact float64 integer precision.")
        w = c.column(sample, weights)
        if bool(((w < 0) | (w != w.round())).any()):
            raise AnalysisError("invalid_weights", "Frequency weights must be nonnegative exact integers.")
    n = sum(int(v) for v in w.tolist())
    if n > o.EXACT_COUNT:
        raise AnalysisError("invalid_weights", "Frequency weight total exceeds exact float64 integer precision.")
    positive = w > 0
    zeros = int((~positive).sum())
    raw = c.matrix(sample, names)[positive]
    sample = sample.loc[positive.numpy()].reset_index(drop=True)
    w = w[positive]
    if not len(sample):
        raise AnalysisError("empty_sample", "No positive-frequency rows remain.")
    labels = o._labels(pd.unique(sample[group]))
    if len(labels) < 2:
        raise AnalysisError("invalid_groups", "Stepwise LDA needs at least two positive-frequency groups.")
    if set(map(str, labels)) & {"n", "prior", "n_physical", "percent_correct"}:
        raise AnalysisError("invalid_groups", "Group labels conflict with classification result columns.")
    plan = _plan(len(names), len(labels), rows, steps)
    codes, labels = c.group_codes(sample[group], group)
    counts = torch.zeros(len(labels), dtype=c.FLOAT).index_add_(0, codes, w)
    if n-len(labels) < len(names):
        raise AnalysisError("insufficient_observations", "Full candidate within covariance needs N-G >= candidate count.")
    origin = raw[0].clone()
    x = raw-origin
    offsets = torch.zeros((len(labels), len(names)), dtype=c.FLOAT).index_add_(0, codes, w[:, None]*x)/counts[:, None]
    offsets += torch.zeros_like(offsets).index_add_(0, codes, w[:, None]*(x-offsets[codes]))/counts[:, None]
    ss = []
    for g in range(len(labels)):
        keep = codes == g
        dev = x[keep]-offsets[g]
        block = dev.T@(w[keep, None]*dev)
        ss.append((block+block.T)/2)
    state = {"version": 1, "group": group, "candidates": names, "groups": labels,
             "counts": [int(v) for v in counts.tolist()], "origin": origin.tolist(),
             "mean_offsets": offsets.tolist(), "group_sscp": [v.tolist() for v in ss],
             "physical_counts": torch.bincount(codes, minlength=len(labels)).tolist(),
             "n_input": rows, "n_missing": dropped, "n_zero_weight": zeros,
             "weights": weights, "weight_type": "fweight" if weights is not None else None,
             "missing": missing}
    return state, x, codes, w, plan


def _geometry(state: dict) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    names, labels = state["candidates"], state["groups"]
    counts = torch.tensor(state["counts"], dtype=c.FLOAT)
    means = torch.tensor(state["mean_offsets"], dtype=c.FLOAT)
    blocks = torch.tensor(state["group_sscp"], dtype=c.FLOAT)
    within = blocks.sum(0)
    _, scale = _checked_within(within)
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(blocks).all()):
        raise AnalysisError("numerical_failure", "Group moments overflow float64; rescale inputs.")
    centered = _center_means(means, counts)
    total = within+centered.T@(counts[:, None]*centered)
    if not bool(torch.isfinite(total).all()):
        raise AnalysisError("numerical_failure", "Total moments overflow float64; rescale inputs.")
    assert means.shape == (len(labels), len(names))
    return counts, means, within, total, scale


def _classifier(state: dict) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    counts, means, within, _, _ = _geometry(state)
    selected = [state["candidates"].index(name) for name in state["selected"]]
    prior = torch.tensor(state["priors"], dtype=c.FLOAT)
    factor = torch.linalg.cholesky(within[selected][:, selected]) if selected else None
    return counts, means[:, selected], factor, prior


@c.procedure
def discrim_stepwise(data: Any, group: str, columns: list[str], *, method: str = "stepwise",
                     include: list[str] | None = None, p_enter: float = .05, p_remove: float = .1,
                     max_steps: int | None = None, weights: str | None = None,
                     weight_type: str = "fweight", priors: Any = "equal", missing: str = "drop") -> TableSet:
    """Forward/backward/removal-first singleton LDA screening on a fixed sample.

    All candidates must have a well-conditioned full pooled within covariance.
    Entry uses reference_p < p_enter; removal uses reference_p > p_remove.
    Input column order breaks exact ties. Include variables are never removed.
    Frequency weights represent literal replicated independent observations;
    correlated copies, survey weights and analytic weights are outside this model.
    Screening probabilities do not provide post-selection inference. Training
    confusion is descriptive resubstitution; selection is not cross-validated.
    """
    group = c.check_name(group, "group")
    names = c.name_list(columns, "columns")
    if set(names) & {"group", "variable", "Intercept"}:
        raise AnalysisError("invalid_spec", "Candidate names conflict with discriminant result columns.")
    c.check_choice(method, "method", ("forward", "backward", "stepwise"))
    c.check_choice(weight_type, "weight_type", ("fweight",))
    c.check_choice(missing, "missing", ("drop", "raise"))
    if weights is not None:
        weights = c.check_name(weights, "weights")
    forced = [] if include is None else c.name_list(include, "include", minimum=0)
    if not set(forced) <= set(names):
        raise AnalysisError("invalid_spec", "include must contain candidate names only.")
    forced = [name for name in names if name in forced]
    p_enter = c.check_number(p_enter, "p_enter", minimum=0, maximum=1, exclusive=True)
    p_remove = c.check_number(p_remove, "p_remove", minimum=0, maximum=1, exclusive=True)
    if p_enter >= 1 or p_remove >= 1:
        raise AnalysisError("invalid_option", "Screening thresholds must be strictly between zero and one.")
    steps = 2*len(names) if max_steps is None else c.check_count(max_steps, "max_steps", maximum=MAX_STEPS)
    _plan(len(names), 0, 0, steps)
    state, x, codes, w, plan = _moments(data, group, names, weights, missing, steps)
    counts, means, within, total, scale = _geometry(state)
    if isinstance(priors, Tensor) and (priors.device.type != "cpu" or priors.is_complex()) \
            or getattr(getattr(priors, "dtype", None), "kind", None) == "c":
        raise AnalysisError("invalid_option", "Priors require real CPU probabilities.")
    prior = _priors(priors, counts, state["groups"])
    selected, candidates, history = _selection(within/torch.outer(scale, scale), means/scale,
        counts, names, method, forced, p_enter, p_remove, steps)
    chosen = [names[j] for j in selected]
    state.update({"selected": chosen, "include": forced, "selection_method": method,
                  "p_enter": p_enter, "p_remove": p_remove, "max_steps": steps,
                  "priors": prior.tolist(), "history": history, "stop_reason": "criteria_met"})
    factor = torch.linalg.cholesky(within[selected][:, selected]) if selected else None
    df = sum(state["counts"])-len(counts)
    scores = (o._centered_linear_scores(x[:, selected], means[:, selected], factor, df, torch.log(prior))
              if selected else torch.log(prior).expand(len(x), -1))
    if not bool(torch.isfinite(scores).all()):
        raise AnalysisError("numerical_failure", "Selected classification scores overflow; rescale inputs.")
    labels = state["groups"]
    confusion, rate = o._weighted_classification(scores, codes, counts, labels, w)
    physical = torch.tensor(state["physical_counts"], dtype=c.FLOAT)
    physical_confusion, _ = o._weighted_classification(scores, codes, physical, labels, torch.ones_like(w))
    origin = torch.tensor(state["origin"], dtype=c.FLOAT)
    raw_means = origin+means
    coefficient = (torch.cholesky_solve(raw_means[:, selected].T, factor)*df
                   if selected else torch.empty((0, len(labels)), dtype=c.FLOAT))
    intercept = -.5*(raw_means[:, selected].T*coefficient).sum(0)+torch.log(prior)
    if not bool(torch.isfinite(coefficient).all()) or not bool(torch.isfinite(intercept).all()):
        raise AnalysisError("numerical_failure", "Selected raw classification functions overflow; rescale inputs.")
    tables = {
        "candidate_tests": c.frame(candidates, columns=_CANDIDATE_COLUMNS),
        "selection_history": c.frame(history, columns=_HISTORY_COLUMNS),
        "groups": c.frame([[int(count), float(pr), int(phys)] for count, pr, phys in
                            zip(counts, prior, physical, strict=True)], columns=["n", "prior", "n_physical"], index=labels),
        "candidate_group_means": c.frame(raw_means, columns=names, index=labels),
        "candidate_within_sscp": c.frame(within, columns=names, index=names),
        "candidate_total_sscp": c.frame(total, columns=names, index=names),
        "group_means": c.frame(raw_means[:, selected], columns=chosen, index=labels),
        "pooled_covariance": c.frame(within[selected][:, selected]/df, columns=chosen, index=chosen),
        "classification_functions": c.frame(torch.cat([coefficient, intercept[None, :]]),
                                             columns=list(map(str, labels)), index=[*chosen, "Intercept"]),
        "classification_table": confusion, "classification_table_physical": physical_confusion,
    }
    return TableSet(tables, title="Stepwise linear discriminant screening", procedure="discrim_stepwise",
        method=method, classifier_method="lda", group=group, candidates=names, variables=chosen,
        groups=labels, n=sum(state["counts"]), n_physical=len(x), n_missing=state["n_missing"],
        n_zero_weight=state["n_zero_weight"], weight_sum=sum(state["counts"]), weights=weights,
        weight_type=state["weight_type"], missing="fixed listwise sample across all candidates and weights",
        include=forced, priors=prior.tolist(), stop_reason="criteria_met", steps=len(history),
        percent_correct_training=rate, resource_plan=plan, precision="float64", device="cpu",
        selection_state=state, selection_state_sha256=o._hash(state),
        reference_p_role="Unadjusted prespecified ANCOVA reference distribution used only for screening; not selective inference.",
        post_selection_inference=False, cross_validation=False,
        tie_policy="first input column on exact conditional F ties; order by F even if reference probabilities underflow", removal_priority=True,
        inferential_assumptions="Independent multivariate-normal observations with common group covariance; frequency counts represent literal independent replication. Selection probabilities are not inferential p-values.",
        notes=["No selected-model equality, canonical significance or Box M tests are reported.",
               "Training accuracy is descriptive resubstitution and does not account for selection."])


def _saved(result: TableSet) -> dict:
    if not isinstance(result, TableSet):
        raise AnalysisError("invalid_result", "Supply a saved stepwise discriminant TableSet.")
    state = result.attrs.get("selection_state")
    if not isinstance(state, dict) or o._hash(state) != result.attrs.get("selection_state_sha256"):
        raise AnalysisError("invalid_result", "Saved selection state is missing or changed.")
    expected = {"version", "group", "candidates", "groups", "counts", "origin", "mean_offsets", "group_sscp",
                "physical_counts", "n_input", "n_missing", "n_zero_weight", "weights", "weight_type",
                "missing", "selected", "include", "selection_method", "p_enter", "p_remove",
                "max_steps", "priors", "history", "stop_reason"}
    try:
        if set(state) != expected or state["version"] != 1 or state["stop_reason"] != "criteria_met":
            raise ValueError("Unsupported state fields")
        names = c.name_list(state["candidates"], "saved candidates")
        group = c.check_name(state["group"], "saved group")
        if group in names or set(names) & {"group", "variable", "Intercept"}:
            raise ValueError("Invalid saved candidate roles")
        chosen = c.name_list(state["selected"], "saved selected variables", minimum=0)
        forced = c.name_list(state["include"], "saved include", minimum=0)
        if chosen != [name for name in names if name in chosen] or not set(forced) <= set(chosen):
            raise ValueError("Invalid selected subset")
        labels = o._labels(state["groups"])
        count_list, counts = o._counts(state["counts"], labels, minimum=1)
        rows = c.check_count(state["n_input"], "saved input rows", maximum=MAX_ROWS)
        steps = c.check_count(state["max_steps"], "saved max_steps", maximum=MAX_STEPS)
        _plan(len(names), len(labels), rows, steps)
        if len(labels) < 2 or sum(count_list)-len(labels) < len(names):
            raise ValueError("Invalid sample rank")
        physical = [c.check_count(v, "saved physical count", maximum=rows) for v in state["physical_counts"]]
        if len(physical) != len(labels) or any(p > n for p, n in zip(physical, count_list, strict=True)):
            raise ValueError("Invalid physical counts")
        dropped = c.check_count(state["n_missing"], "saved missing", minimum=0, maximum=rows)
        zeros = c.check_count(state["n_zero_weight"], "saved zeros", minimum=0, maximum=rows)
        if sum(physical)+dropped+zeros != rows:
            raise ValueError("Invalid sample accounting")
        if state["weights"] is None:
            if state["weight_type"] is not None or zeros or physical != count_list:
                raise ValueError("Invalid unit frequencies")
        else:
            c.check_name(state["weights"], "saved weights")
            if state["weight_type"] != "fweight" or state["weights"] in [group, *names]:
                raise ValueError("Invalid weight type")
        c.check_choice(state["missing"], "saved missing", ("drop", "raise"))
        c.check_choice(state["selection_method"], "saved method", ("forward", "backward", "stepwise"))
        for key in ("p_enter", "p_remove"):
            if not 0 < c.check_number(state[key], key) < 1:
                raise ValueError("Invalid screening threshold")
        prior = _priors(state["priors"], counts, labels)
        if not torch.allclose(prior, torch.tensor(state["priors"], dtype=c.FLOAT), rtol=0, atol=1e-15):
            raise ValueError("Saved prior normalization changed")
        origin = o._matrix(torch.tensor(state["origin"], dtype=c.FLOAT), (len(names),), "Saved origin")
        means = o._matrix(state["mean_offsets"], (len(labels), len(names)), "Saved means")
        if float(origin.abs().max()) > c.LARGEST or float((means+origin).abs().max()) > c.LARGEST:
            raise ValueError("Saved means exceed admitted input domain")
        if len(state["group_sscp"]) != len(labels):
            raise ValueError("Invalid group matrix count")
        for block, count, phys in zip(state["group_sscp"], count_list, physical, strict=True):
            matrix = o._matrix(block, (len(names), len(names)), "Saved group SSCP")
            if count == 1:
                if bool((matrix != 0).any()):
                    raise ValueError("Invalid singleton SSCP")
            else:
                o._covariance(matrix/(count-1), names, count)
                positive = matrix.diagonal() > 0
                if bool(positive.any()):
                    scale_g = matrix.diagonal()[positive].sqrt()
                    corr_g = matrix[positive][:, positive]/torch.outer(scale_g, scale_g)
                    rank = int((torch.linalg.eigvalsh(corr_g) > 8*len(names)*torch.finfo(c.FLOAT).eps).sum())
                    if rank > phys-1:
                        raise ValueError("SSCP rank exceeds positive physical rows minus one")
        _, _, within, total, scale = _geometry(state)
        selected, candidates, history = _selection(within/torch.outer(scale, scale), means/scale, counts,
            names, state["selection_method"], forced, state["p_enter"], state["p_remove"], steps)
        # Validate the saved decisions from declared moments; no data refit is performed.
        if [names[i] for i in selected] != chosen or history != state["history"]:
            raise ValueError("Saved selection decisions disagree with source moments")
        metadata = {"procedure": "discrim_stepwise", "classifier_method": "lda", "group": group,
                    "variables": chosen, "candidates": names, "groups": labels, "n": sum(count_list),
                    "n_physical": sum(physical), "n_missing": dropped, "n_zero_weight": zeros,
                    "weight_sum": sum(count_list), "weights": state["weights"],
                    "weight_type": state["weight_type"], "include": forced, "priors": state["priors"],
                    "method": state["selection_method"], "steps": len(history), "stop_reason": "criteria_met",
                    "post_selection_inference": False, "cross_validation": False}
        if any(result.attrs.get(key) != value for key, value in metadata.items()):
            raise ValueError("Result metadata disagrees with state")
        df = sum(count_list)-len(labels)
        raw_means = means+origin
        selected_factor = torch.linalg.cholesky(within[selected][:, selected]) if selected else None
        coefficient = (torch.cholesky_solve(raw_means[:, selected].T, selected_factor)*df
                       if selected else torch.empty((0, len(labels)), dtype=c.FLOAT))
        intercept = -.5*(raw_means[:, selected].T*coefficient).sum(0)+torch.log(prior)
        for key, values, columns, index in (
            ("candidate_within_sscp", within, names, names),
            ("candidate_total_sscp", total, names, names),
            ("candidate_group_means", raw_means, names, labels),
            ("group_means", raw_means[:, selected], chosen, labels),
            ("pooled_covariance", within[selected][:, selected]/df, chosen, chosen),
            ("classification_functions", torch.cat([coefficient, intercept[None, :]]),
             list(map(str, labels)), [*chosen, "Intercept"]),
            ("groups", torch.tensor([[count, float(pr), phys] for count, pr, phys in
                                      zip(count_list, prior, physical, strict=True)], dtype=c.FLOAT),
             ["n", "prior", "n_physical"], labels),
        ):
            frame = result[key]
            if list(frame.columns) != columns or list(frame.index) != index \
                    or not torch.allclose(torch.tensor(frame.to_numpy(dtype="float64")), values,
                                          rtol=1e-12, atol=1e-14):
                raise ValueError("Saved tables disagree with source state")
        for key, columns, values in (("selection_history", _HISTORY_COLUMNS, history),
                                     ("candidate_tests", _CANDIDATE_COLUMNS, candidates)):
            if list(result[key].columns) != columns or result[key].to_numpy(dtype=object).tolist() != values:
                raise ValueError("Saved screening tables disagree with source state")
    except (AnalysisError, ValueError, TypeError, KeyError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_result", "Saved selection state is inconsistent or outside its domain.") from exc
    return state


@c.procedure
def discrim_stepwise_predict(result: TableSet, data: Any) -> pd.DataFrame:
    """Saved selected LDA posterior projection, including a prior-only empty model."""
    state = _saved(result)
    names, labels = state["selected"], state["groups"]
    data = o._project_columns(data, names) if names else data
    rows = o._resident_rows(data)
    plan = _plan(len(state["candidates"]), len(labels), rows, state["max_steps"], prediction=True)
    if names:
        frame = c.source(o._project_records(data, names))
        c.require_numeric(frame, names)
        frame = frame.loc[:, names]
        keep = ~frame.isna().any(axis=1)
        selected = [state["candidates"].index(name) for name in names]
        origin = torch.tensor(state["origin"], dtype=c.FLOAT)[selected]
        x = c.matrix(frame.loc[keep], names)-origin
        counts, means, factor, prior = _classifier(state)
        scores = o._centered_linear_scores(x, means, factor, sum(state["counts"])-len(labels), torch.log(prior))
        posterior = _softmax_rows(scores)
    else:
        frame = pd.DataFrame(index=data.index if isinstance(data, pd.DataFrame) else pd.RangeIndex(rows))
        keep = pd.Series(True, index=frame.index)
        prior = torch.tensor(state["priors"], dtype=c.FLOAT)
        scores = torch.log(prior).expand(rows, -1)
        posterior = prior.expand(rows, -1)
    if not bool(torch.isfinite(posterior).all()):
        raise AnalysisError("numerical_failure", "Saved prediction distances overflow float64.")
    columns = [f"posterior_{label}" for label in labels]
    out = table(pd.DataFrame(index=frame.index, columns=["predicted", *columns], dtype=object))
    mask = keep.to_numpy()
    if bool(keep.any()):
        out.loc[mask, "predicted"] = [labels[i] for i in scores.argmax(1).tolist()]
    for j, name in enumerate(columns):
        values = pd.Series(float("nan"), index=frame.index, dtype="float64")
        values.loc[mask] = posterior[:, j].numpy()
        out[name] = values
    out.attrs.update({"resource_plan": plan, "prediction_origin": "saved stepwise centered LDA state",
                      "selection_refit": False, "post_selection_inference": False})
    return out
