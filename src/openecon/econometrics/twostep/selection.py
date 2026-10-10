"""IBM Statistics14 change/jump rule with explicit finite boundary policies."""
from __future__ import annotations

import math


def select_two_stage(criteria, merges, criterion, max_clusters):
    index = 3 if criterion == "bic" else 4
    values = {int(row[0]): row[index] for row in criteria}
    leaves = len(merges) + 1
    trace = {"mode": "two_stage", "criterion": criterion, "change_threshold": 0.04,
             "dominance_threshold": 1.15, "changes": [], "jumps": [],
             "initial_k": None, "selected_k": None, "reason": None}

    def finish(k, reason):
        trace.update(selected_k=k, reason=reason)
        return k, trace

    if leaves == 1 or max_clusters == 1:
        return finish(1, "one_feasible_cut" if leaves == 1 else "declared_one_cluster_cap")
    first = values[1] - values[2]
    if first < 0:
        return finish(1, "negative_initial_change")
    if first == 0:
        return finish(1, "zero_initial_change_extension")
    if leaves == 2:
        trace["initial_k"] = 2
        return finish(2, "two_cut_no_jump_extension")
    upper = min(max_clusters, leaves - 1)
    initial = upper
    crossed = False
    for k in range(1, upper + 1):
        change = values[k] - values[k+1]
        ratio = change / first
        if not math.isfinite(ratio):
            # The comparison can be performed without an overflowing division.
            crossed_here = change < 0.04 * first
            kind, recorded = "overflow", None
        else:
            crossed_here = ratio < 0.04
            kind, recorded = "finite", ratio
        trace["changes"].append({"clusters": k, "change": change, "ratio": recorded, "kind": kind})
        if crossed_here:
            initial, crossed = k, True
            break
    trace["initial_k"] = initial
    losses = {entry["clusters_after"] + 1: entry["distance"] for entry in merges}
    ranked = []
    for k in range(initial, 1, -1):
        numerator, denominator = losses[k], losses[k+1]
        if denominator == 0:
            ratio = math.inf if numerator > 0 else 1.0
            kind = "positive_over_zero_extension" if numerator > 0 else "zero_over_zero_neutral_extension"
        else:
            ratio = numerator / denominator
            kind = "finite" if math.isfinite(ratio) else "overflow"
        trace["jumps"].append({"clusters": k, "numerator": numerator, "denominator": denominator,
                                "ratio": ratio if math.isfinite(ratio) else None, "kind": kind})
        ranked.append((ratio, k))
    ranked.sort(reverse=True)
    if len(ranked) == 1:
        return finish(ranked[0][1], "single_jump_extension")
    first_jump, second_jump = ranked[:2]
    dominates = first_jump[0] > 1.15 * second_jump[0]
    chosen = first_jump[1] if dominates else max(first_jump[1], second_jump[1])
    trace["change_crossed"] = crossed
    return finish(chosen, "dominant_jump" if dominates else "top_two_larger_count")
