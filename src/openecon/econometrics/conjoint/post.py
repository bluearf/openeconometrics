"""Held-out validation, individual-normalized importance and conditional shares."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from . import common as c
from .model import evaluated, scored


@resident_cpu
def conjoint_holdout(result, plan, responses, *, subject="subject", profile="profile_id", score="score",
                     missing="raise", max_work=100_000_000, device="cpu", weights=None):
    """Validate separately held-out scores without fitting them or assigning correlation CI.

    Reject training IDs AND reused training factor combinations. Require every
    retained subject to score every holdout. Report descriptive Pearson, Kendall
    tau-b, MAE and RMSE. Whole-subject dropping is explicit; no silent exclusion.
    """
    state, data, x, predictions, _ = evaluated(result, plan, profile, device, weights, max_work)
    training_ids = {c.key(v) for v in state["training_profile_ids"]}
    training_combinations = {c.canonical([c.key(v) for v in row]) for row in state["training_profiles"]}
    order = state["attribute_order"]
    if any(c.key(v) in training_ids for v in data[profile]) or any(
        c.canonical([c.key(v) for v in row]) in training_combinations
        for row in data[order].itertuples(index=False, name=None)
    ):
        raise AnalysisError("holdout_leakage", "Holdouts must have new profile IDs and genuinely held-out factor combinations.")
    groups, dropped, original_n = scored(responses, data, subject, profile, score, missing)
    lookup = {c.key(record["subject"]): i for i, record in enumerate(state["subjects"])}
    observed = {c.key(g["subject"]) for g in groups} | {c.key(g["subject"]) for g in dropped}
    if observed - set(lookup):
        raise AnalysisError("unknown_subject", "Holdouts cannot validate a subject without saved individual utilities.")
    absent = set(lookup) - observed
    if absent and missing == "raise":
        raise AnalysisError("incomplete_subject", "Every fitted subject must supply the complete holdout responses.")
    dropped += [{"subject": state["subjects"][lookup[k]]["subject"], "reason": "no holdout responses"} for k in sorted(absent)]
    n, p, m = len(data), x.shape[1], len(groups)
    if n < 2:
        raise AnalysisError("insufficient_observations", "Holdout validation needs at least two profiles.")
    settings = c.guard("conjoint_holdout", n, p, m, device=device, weights=weights, max_work=max_work,
                       work=m*(n*n + n*p*p), extra_buffers={"tau_pair_indices_and_differences": 64*n*n})
    indices = torch.triu_indices(n, n, 1)
    rows, details, retained = [], [], []
    for group in groups:
        predicted = predictions[lookup[c.key(group["subject"])]]
        actual = torch.tensor(group["scores"], dtype=torch.float64)
        a, b = actual-actual.mean(), predicted-predicted.mean()
        denominator = float(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))
        pearson = float((a@b)/denominator) if denominator else None
        dx = torch.sign(actual[indices[0]]-actual[indices[1]])
        dy = torch.sign(predicted[indices[0]]-predicted[indices[1]])
        tau_denominator = math.sqrt(int((dx != 0).sum())*int((dy != 0).sum()))
        tau = float((dx*dy).sum())/tau_denominator if tau_denominator else None
        error = actual-predicted
        rows.append([group["subject"], n, pearson, tau, float(torch.mean(error.abs())),
                     float(torch.sqrt(torch.mean(error**2))), denominator == 0, tau_denominator == 0])
        details.extend([group["subject"], c.label(card), float(a), float(b), pos, label]
                       for card, a, b, pos, label in zip(data[profile], actual, predicted, group["positions"], group["labels"]))
        retained.append({"subject": group["subject"], "positions": group["positions"], "labels": group["labels"]})
    return c.seal("conjoint_holdout", {
        "validation": table(rows, columns=["subject", "profiles", "pearson", "kendall_tau_b", "mae", "rmse", "pearson_undefined", "tau_undefined"]),
        "holdout_predictions": table(details, columns=["subject", "profile_id", "observed", "predicted", "source_position", "source_label"]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"], "holdout_profile_ids": [c.label(v) for v in data[profile]],
        "original_response_rows": original_n, "subjects": retained, "dropped_subjects": dropped, "missing": missing,
        "settings": settings}, inference="descriptive validation; no correlation significance/CI", **settings)


@resident_cpu
def conjoint_importance(result, *, max_work=100_000_000, device="cpu", weights=None):
    """Normalize utility ranges per individual, then average importance equally over subjects.

    Ranges use the finite declared level universe. Importance is descriptive;
    the importance of mean utilities is separately labelled. A zero total
    utility range is explicitly undefined and prevents a complete group average.
    """
    state = c.intact(result, "conjoint_fit")
    attributes, order = state["attributes"], state["attribute_order"]
    p, m = len(state["terms"]), len(state["subjects"])
    level_count = sum(len(attributes[a]) for a in order)
    settings = c.guard("conjoint_importance", level_count, p, m, device=device, weights=weights,
                       max_work=max_work, work=m*level_count*p*p)
    transform = torch.tensor(state["transform"], dtype=torch.float64)
    ranges, utility_by_subject, individual = [], [], []
    for record in state["subjects"]:
        beta = torch.tensor(record["parameters_scaled"], dtype=torch.float64)
        utilities, span = [], []
        for item in state["mapping"]:
            values = torch.stack([c.level_vector(item, value, attributes[item["attribute"]], p)@transform@beta
                                  for value in attributes[item["attribute"]]])
            utilities.append(values)
            span.append(float(values.max()-values.min()))
        total = sum(span)
        for attr, value in zip(order, span):
            individual.append([record["subject"], attr, value, None if total == 0 else 100*value/total])
        utility_by_subject.append(utilities)
        ranges.append(span)
    ranges = torch.tensor(ranges, dtype=torch.float64)
    sums = ranges.sum(1)
    all_defined = bool((sums>0).all())
    average = (100*ranges/sums[:, None]).mean(0) if all_defined else None
    group_spans = [float(values.max()-values.min()) for values in (
        torch.stack([utilities[j] for utilities in utility_by_subject]).mean(0) for j in range(len(order))
    )]
    group_total = sum(group_spans)
    grouped = [[attr, None if average is None else float(average[j]), group_spans[j],
                None if group_total == 0 else 100*group_spans[j]/group_total]
               for j, attr in enumerate(order)]
    return c.seal("conjoint_importance", {
        "individual": table(individual, columns=["subject", "attribute", "utility_range", "importance_percent"]),
        "group": table(grouped, columns=["attribute", "mean_individual_importance_percent", "range_of_mean_utilities", "importance_of_mean_utilities_percent"]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"], "level_universe": attributes,
        "undefined_subjects": [r["subject"] for r, total in zip(state["subjects"], sums) if float(total) == 0],
        "settings": settings}, inference="descriptive finite-level importance; no sampling CI",
       complete_group_average_defined=all_defined, **settings)


@resident_cpu
def conjoint_simulate(result, plan, *, method="first_choice", temperature=1.0, tie_tolerance=0.0,
                     profile="profile_id", max_work=100_000_000, device="cpu", weights=None):
    """Calculate conditional first-choice, positive-score BTL or stable logit preference shares.

    Exact first-choice ties split equally unless a positive tie tolerance is
    declared. BTL rejects nonpositive scores without shifts/dropped respondents.
    Logit accepts all finite scores, at an explicit positive temperature.
    Equal-subject shares are not population estimates or CBC-fitted probabilities.
    """
    c.c.check_choice(method, "method", ("first_choice", "btl", "logit"))
    temperature = c.c.check_number(temperature, "temperature", minimum=0, exclusive=True)
    tie_tolerance = c.c.check_number(tie_tolerance, "tie_tolerance", minimum=0)
    if method != "logit" and temperature != 1 or method != "first_choice" and tie_tolerance != 0:
        raise AnalysisError("inapplicable_option", "Temperature applies only to logit; tie tolerance only to first-choice.")
    state, data, _, scores, settings = evaluated(result, plan, profile, device, weights, max_work)
    if method == "first_choice":
        winners = scores.max(1, keepdim=True).values-scores <= tie_tolerance
        probabilities = winners.to(torch.float64)/winners.sum(1, keepdim=True)
    elif method == "btl":
        if not bool((scores>0).all()):
            raise AnalysisError("nonpositive_btl_score", "BTL requires every subject's score for every alternative to be positive.")
        scaled = scores/scores.max(1, keepdim=True).values
        probabilities = scaled/scaled.sum(1, keepdim=True)
    else:
        probabilities = torch.softmax((scores-scores.max(1, keepdim=True).values)/temperature, dim=1)
    if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(1), torch.ones(scores.shape[0], dtype=torch.float64), atol=1e-12, rtol=0):
        raise AnalysisError("numerical_failure", "Preference probabilities are not finite and normalized.")
    rows = [[record["subject"], c.label(card), float(value), float(probability)]
            for record, values, probs in zip(state["subjects"], scores, probabilities)
            for card, value, probability in zip(data[profile], values, probs)]
    average = probabilities.mean(0)
    return c.seal("conjoint_simulate", {
        "individual": table(rows, columns=["subject", "profile_id", "predicted_score", "conditional_share"]),
        "group": table([[c.label(card), float(value)] for card, value in zip(data[profile], average)],
                       columns=["profile_id", "equal_subject_share"]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"], "profile_ids": [c.label(v) for v in data[profile]],
        "subjects": [r["subject"] for r in state["subjects"]], "method": method, "temperature": temperature,
        "tie_tolerance": tie_tolerance, "settings": settings},
       inference="conditional descriptive shares; no market sampling CI", **settings)
