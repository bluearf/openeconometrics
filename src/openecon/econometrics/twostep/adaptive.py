"""Bounded CF-summary rebuilds and sparse-record reinsertion.

IBM Statistics 14 TwoStep guide pp2,6 supplies the lifecycle and log-volume
query cutoff. Deterministic threshold growth, ties and finite retry limits are
declared local policies, not claims about a licensed executable's internals.
"""
from __future__ import annotations

from dataclasses import replace
import math

from openecon.analysis_contracts import AnalysisError
from . import kernel


class Budget:
    def __init__(self, features, max_work):
        self.unit = 16 * (features + 8)
        self.max_work = max_work
        self.cost = self.distances = self.clamps = 0

    def charge(self, count=1):
        if self.cost + self.unit * count > self.max_work:
            raise AnalysisError("resource_limit", "Cumulative TwoStep rebuild/reinsertion work exceeds max_work; no partial model.")
        self.cost += self.unit * count

    def charge_distance(self):
        self.charge()
        self.distances += 1

    def loss(self, a, b, variance):
        self.charge_distance()
        loss, clamp = kernel._distance(a, b, variance)
        self.clamps += clamp
        return loss


def _leaves(tree):
    result = []
    def visit(node):
        for entry in node.entries:
            if node.leaf:
                result.append(entry.cf)
            else:
                visit(entry.child)
    visit(tree.root)
    return sorted(result, key=kernel._key)


def _absorb(tree, incoming):
    """Route a record and absorb only; never allocate a leaf or node."""
    def descend(node):
        if not node.entries:
            return False
        closest = tree._closest(node.entries, incoming)
        if node.leaf:
            if tree._loss(closest.cf, incoming) > tree.threshold:
                return False
            closest.cf = kernel.merge(closest.cf, incoming)
            return True
        accepted = descend(closest.child)
        if accepted:
            closest.cf = kernel._summary(closest.child)
        return accepted
    return descend(tree.root)


def noise_cutoff(X, levels):
    """log(product training ranges * product categorical level counts)."""
    ranges = (X.max(0).values - X.min(0).values).tolist() if X.shape[1] else []
    if any(not math.isfinite(value) or value <= 0 for value in ranges):
        raise AnalysisError("degenerate_continuous", "TwoStep noise volume needs positive finite training ranges.")
    return math.fsum([math.log(value) for value in ranges] + [math.log(value) for value in levels])


def build_adaptive(X, codes, levels, globalvar, row_order, *, threshold,
                   branch_factor, max_preclusters, max_nodes, max_rebuilds,
                   noise_fraction, max_work):
    budget = Budget(X.shape[1] + sum(levels), max_work)
    options = (globalvar, threshold, branch_factor, max_preclusters, max_nodes)
    tree = kernel._Tree(*options, budget=budget)
    noise = set()
    trace = []
    rebuild_count = 0

    def record(row):
        budget.charge()
        return kernel.singleton(X[row], codes[row], levels, row)

    def new_tree(pool, target):
        rebuilt = kernel._Tree(globalvar, target, branch_factor, max_preclusters, max_nodes, budget)
        for cf in sorted(pool, key=kernel._key):
            budget.charge(cf.count)
            rebuilt.insert(cf)
        return rebuilt

    def transition(pool, incoming, final=False):
        nonlocal tree, noise, rebuild_count
        before = tree.threshold
        largest = max(cf.count for cf in pool)
        sparse = [cf for cf in pool if cf.count < noise_fraction * largest]
        retained = [cf for cf in pool if cf.count >= noise_fraction * largest]
        deferred = noise | {row for cf in sparse for row in cf.rows}
        if final and not sparse:
            return
        cost_before, distances_before = budget.cost, budget.distances
        target = before
        attempted = []
        while True:
            if not final:
                if rebuild_count >= max_rebuilds:
                    raise AnalysisError("resource_limit", "TwoStep max_rebuilds exhausted; no records are silently dropped.")
                rebuild_count += 1
                losses = [budget.loss(a, b, globalvar) for i, a in enumerate(retained) for b in retained[i+1:]]
                closest = min(losses) if losses else before
                # Strict progress even at an exact zero/tied distance.
                target = math.nextafter(max(target * 2, closest), math.inf)
                if not math.isfinite(target):
                    raise AnalysisError("numerical_failure", "TwoStep adaptive threshold cannot grow finitely.")
                attempted.append(target)
            try:
                rebuilt = new_tree(retained, target)
                break
            except AnalysisError as exc:
                if final or exc.code != "resource_limit" or not any(text in str(exc) for text in ("exceeds max_nodes=", "exceeds max_preclusters=")):
                    raise
        reinserted = []
        for row in sorted(deferred):
            if _absorb(rebuilt, record(row)):
                reinserted.append(row)
        noise = deferred - set(reinserted)
        tree = rebuilt
        trace.append({
            "phase": "final_noise" if final else "fullness_rebuild",
            "incoming": kernel._cf_dict(incoming) if incoming is not None else None,
            "before": [kernel._cf_dict(cf) for cf in pool],
            "sparse_rows": sorted(row for cf in sparse for row in cf.rows),
            "deferred_rows": sorted(deferred), "reinserted_rows": reinserted,
            "noise_rows": sorted(noise),
            "after": [kernel._cf_dict(cf) for cf in _leaves(tree)],
            "threshold_before": before, "threshold_after": target,
            "attempted_thresholds": attempted,
            "cost_before": cost_before, "cost_after": budget.cost,
            "distances_before": distances_before, "distances_after": budget.distances,
        })

    for row in row_order.tolist():
        incoming = record(row)
        prior = _leaves(tree)
        budget.charge(len(prior))
        try:
            tree.insert(incoming)
        except AnalysisError as exc:
            if exc.code != "resource_limit" or not any(text in str(exc) for text in ("exceeds max_nodes=", "exceeds max_preclusters=")):
                raise
            # The failed split may have mutated nodes. Immutable CFs from before
            # insertion plus the incoming record conserve all accepted records.
            transition([*prior, incoming], incoming)
    if noise_fraction:
        transition(_leaves(tree), None, final=True)
    result = tree.result()
    noise_cf = None
    for row in sorted(noise):
        cf = record(row)
        noise_cf = cf if noise_cf is None else kernel.merge(noise_cf, cf)
    kept_rows = sorted(row for cf in result.preclusters for row in cf.rows)
    if len(kept_rows) + len(noise) != len(X) or sorted(kept_rows + list(noise)) != list(range(len(X))):
        raise AnalysisError("invalid_cf", "Adaptive TwoStep failed complete physical-row conservation.")
    result.tree.update({
        "automatic_rebuild": True, "threshold_initial": threshold,
        "threshold_final": tree.threshold, "max_rebuilds": max_rebuilds,
        "rebuild_count": rebuild_count, "rebuild_trace": trace,
        "noise_fraction": noise_fraction,
        "noise_cutoff": noise_cutoff(X, levels) if noise_fraction else None,
        "operation_cost": budget.cost, "max_work": max_work,
    })
    return replace(result, distance_evaluations=budget.distances,
                   roundoff_clamps=budget.clamps, noise_cf=noise_cf)
