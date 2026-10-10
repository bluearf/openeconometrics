"""Bounded CF tree and mixed Gaussian/multinomial merge-loss hierarchy.

The score and tree routing/splitting follow IBM's TwoStep algorithm guide:
https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/twostep_cluster.pdf
Adaptive rebuilding is explicit and bounded; legacy insertion still refuses
tree fullness. The hierarchy retains every declared non-noise cut.
All continuous summaries use CPU float64 parallel Welford updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError

ROUNDING_MULTIPLIER = 64
ROUNDING_RULE = (
    "clamp negative merge loss only within 64*float64_eps*max(1,abs(xi_a)+abs(xi_b)+abs(xi_merged))"
)
_EPS = torch.finfo(torch.float64).eps


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError(
            "numerical_failure", f"TwoStep {name} exceeds finite float64 arithmetic."
        )


def _continuous(value: Tensor, name: str) -> None:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.ndim != 1
    ):
        raise AnalysisError(
            "invalid_design", f"TwoStep {name} must be a one-dimensional CPU float64 tensor."
        )
    _finite(value, name)


@dataclass(frozen=True, eq=False)
class CF:
    """Pure cluster summary; M2 is centered squared deviations, not variance."""

    count: int
    mean: Tensor
    m2: Tensor
    categorical_counts: tuple[Tensor, ...]
    rows: tuple[int, ...]

    def __post_init__(self):
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or not 1 <= self.count <= 2000
        ):
            raise AnalysisError("invalid_cf", "TwoStep CF count must be an integer in 1..2000.")
        _continuous(self.mean, "CF mean")
        _continuous(self.m2, "CF M2")
        if (
            self.mean.shape != self.m2.shape
            or len(self.mean) + len(self.categorical_counts) > 16
            or not bool((self.m2 >= 0).all())
        ):
            raise AnalysisError(
                "invalid_cf", "TwoStep CF dimensions/M2 are inconsistent or exceed 16 features."
            )
        if not isinstance(self.categorical_counts, tuple) or any(
            not isinstance(c, Tensor)
            or c.device.type != "cpu"
            or c.dtype != torch.int64
            or c.ndim != 1
            or not 1 <= len(c) <= 2000
            or not bool((c >= 0).all())
            or int(c.sum()) != self.count
            for c in self.categorical_counts
        ):
            raise AnalysisError(
                "invalid_cf",
                "TwoStep categorical CF counts must be nonnegative CPU int64 counts summing to cluster size.",
            )
        if (
            not isinstance(self.rows, tuple)
            or len(self.rows) != self.count
            or any(
                isinstance(row, bool) or not isinstance(row, int) or row < 0 for row in self.rows
            )
            or len(set(self.rows)) != self.count
            or self.rows != tuple(sorted(self.rows))
        ):
            raise AnalysisError(
                "invalid_cf",
                "TwoStep CF rows must be distinct sorted nonnegative physical positions.",
            )


def singleton(numeric: Tensor, codes: Tensor, levels: tuple[int, ...], row: int) -> CF:
    """Create one physical-record CF; numerical/category arrays are copied."""
    _continuous(numeric, "record")
    if (
        not isinstance(codes, Tensor)
        or codes.dtype != torch.int64
        or codes.device.type != "cpu"
        or codes.ndim != 1
        or len(codes) != len(levels)
    ):
        raise AnalysisError(
            "invalid_design", "TwoStep categorical record must have aligned CPU int64 codes."
        )
    counts = []
    for code, count in zip(codes.tolist(), levels):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 2000
            or not 0 <= code < count
        ):
            raise AnalysisError(
                "invalid_design", "TwoStep categorical levels/codes are out of range."
            )
        histogram = torch.zeros(count, dtype=torch.int64)
        histogram[code] = 1
        counts.append(histogram)
    return CF(1, numeric.detach().clone(), torch.zeros_like(numeric), tuple(counts), (row,))


def _combined(a: CF, b: CF):
    if (
        a.mean.shape != b.mean.shape
        or len(a.categorical_counts) != len(b.categorical_counts)
        or any(ac.shape != bc.shape for ac, bc in zip(a.categorical_counts, b.categorical_counts))
    ):
        raise AnalysisError(
            "invalid_cf", "TwoStep CF merge requires matching feature/level dimensions."
        )
    count = a.count + b.count
    delta = b.mean - a.mean
    mean = a.mean + delta * (b.count / count)
    m2 = a.m2 + b.m2 + delta.square() * (a.count * b.count / count)
    _finite(mean, "merged CF mean")
    _finite(m2, "merged CF M2")
    categorical = tuple(ac + bc for ac, bc in zip(a.categorical_counts, b.categorical_counts))
    return count, mean, m2, categorical


def merge(a: CF, b: CF) -> CF:
    """Combine disjoint summaries without mutating either input (Welford)."""
    if set(a.rows).intersection(b.rows):
        raise AnalysisError("invalid_cf", "TwoStep CFs to merge overlap physical rows.")
    count, mean, m2, categorical = _combined(a, b)
    return CF(count, mean, m2, categorical, tuple(sorted((*a.rows, *b.rows))))


def _global(globalvar: Tensor, p: int) -> None:
    _continuous(globalvar, "global variance")
    if len(globalvar) != p or not bool((globalvar > 0).all()):
        raise AnalysisError(
            "degenerate_continuous",
            "TwoStep requires finite positive global population variance for every continuous feature.",
        )


def _score(count: int, m2: Tensor, categorical: tuple[Tensor, ...], globalvar: Tensor) -> float:
    # IBM's within-cluster estimate is the population/MLE variance M2/N.
    variance = globalvar + m2 / count
    _finite(variance, "adjusted variance")
    if not bool((variance > 0).all()):
        raise AnalysisError(
            "numerical_failure", "TwoStep adjusted continuous variance is not positive."
        )
    components = 0.5 * variance.log().sum()
    for histogram in categorical:
        positive = histogram[histogram > 0].to(dtype=torch.float64) / count
        components -= torch.dot(positive, positive.log())
    score = -count * float(components)
    if not math.isfinite(score):
        raise AnalysisError(
            "numerical_failure", "TwoStep xi score exceeds finite float64 arithmetic."
        )
    return score


def xi(cf: CF, globalvar: Tensor) -> float:
    """xi=-N*(sum .5log(globalvar+M2/N)+sum multinomial entropy)."""
    _global(globalvar, len(cf.mean))
    return _score(cf.count, cf.m2, cf.categorical_counts, globalvar)


def _distance(a: CF, b: CF, globalvar: Tensor, scores: tuple[float, float] | None = None):
    # Distance evaluation needs only sufficient statistics, so it does not
    # copy physical row lists or allocate an unnecessary merged CF.
    count, _, m2, categorical = _combined(a, b)
    left, right = (
        scores
        if scores is not None
        else (
            _score(a.count, a.m2, a.categorical_counts, globalvar),
            _score(b.count, b.m2, b.categorical_counts, globalvar),
        )
    )
    combined = _score(count, m2, categorical, globalvar)
    # Algebraically xi_a+xi_b-xi_ab, expressed as log ratios rather than
    # subtracting three large xi values.  Exact duplicate records then have
    # distance exactly zero, including at a strict zero absorption threshold.
    va, vb, vm = globalvar + a.m2 / a.count, globalvar + b.m2 / b.count, globalvar + m2 / count

    def log_ratio(numerator, denominator):
        change = (numerator - denominator) / denominator
        return torch.where(
            torch.isfinite(change) & (change > -1),
            torch.log1p(change),
            numerator.log() - denominator.log(),
        )

    contributions = [
        0.5 * float(a.count * log_ratio(vm, va).sum()),
        0.5 * float(b.count * log_ratio(vm, vb).sum()),
    ]
    for ha, hb, hm in zip(a.categorical_counts, b.categorical_counts, categorical):
        for histogram, size in ((ha, a.count), (hb, b.count)):
            mask = histogram > 0
            observed = histogram[mask].to(dtype=torch.float64)
            # These integer-sized products are exact below the record bound;
            # proportional category distributions give ratio exactly one.
            ratio = observed * count / (hm[mask].to(dtype=torch.float64) * size)
            contributions.append(float(torch.dot(observed, ratio.log())))
    distance = math.fsum(contributions)
    if not math.isfinite(distance):
        raise AnalysisError(
            "numerical_failure", "TwoStep merge loss exceeds finite float64 arithmetic."
        )
    bound = ROUNDING_MULTIPLIER * _EPS * max(1.0, abs(left) + abs(right) + abs(combined))
    if distance < -bound:
        raise AnalysisError(
            "numerical_failure",
            "TwoStep merge loss is materially negative; no sign reversal or arbitrary clipping is applied.",
        )
    return max(0.0, distance), int(distance < 0)


def merge_loss(a: CF, b: CF, globalvar: Tensor) -> float:
    """Nonnegative xi(a)+xi(b)-xi(a+b); tiny roundoff follows ROUNDING_RULE."""
    _global(globalvar, len(a.mean))
    return _distance(a, b, globalvar)[0]


@dataclass(frozen=True)
class TreeResult:
    preclusters: tuple[CF, ...]
    tree: dict
    node_count: int
    depth: int
    split_count: int
    distance_evaluations: int
    roundoff_clamps: int = 0
    noise_cf: CF | None = None


@dataclass(frozen=True)
class Hierarchy:
    cuts: dict[int, tuple[CF, ...]]
    merges: tuple[dict, ...]
    distance_evaluations: int
    roundoff_clamps: int = 0


@dataclass
class _Entry:
    id: int
    cf: CF
    child: _Node | None = None


@dataclass
class _Node:
    id: int
    leaf: bool
    entries: list[_Entry] = field(default_factory=list)


def _key(cf: CF):
    return cf.rows[0]


def _summary(node: _Node) -> CF:
    summaries = sorted((entry.cf for entry in node.entries), key=_key)
    if not summaries:
        raise AnalysisError("invalid_cf", "TwoStep cannot summarize an empty CF node.")
    result = summaries[0]
    for other in summaries[1:]:
        result = merge(result, other)
    return result


def _cf_dict(cf: CF) -> dict:
    return {
        "count": cf.count,
        "mean": cf.mean.tolist(),
        "m2": cf.m2.tolist(),
        "categorical_counts": [counts.tolist() for counts in cf.categorical_counts],
        "rows": list(cf.rows),
    }


class _Tree:
    def __init__(self, globalvar, threshold, branch_factor, max_preclusters, max_nodes, budget=None):
        self.globalvar, self.threshold, self.branch_factor = globalvar, threshold, branch_factor
        self.max_preclusters, self.max_nodes = max_preclusters, max_nodes
        self.budget = budget
        self.node_count = self.entry_count = self.precluster_count = 0
        self.split_count = self.distance_evaluations = self.roundoff_clamps = 0
        self.root = self._node(True)

    def _node(self, leaf):
        if self.node_count == self.max_nodes:
            raise AnalysisError(
                "resource_limit",
                f"TwoStep CF tree exceeds max_nodes={self.max_nodes}; no threshold rebuild, truncation, or partial model is returned.",
            )
        self.node_count += 1
        return _Node(self.node_count - 1, leaf)

    def _entry(self, cf, child=None):
        self.entry_count += 1
        return _Entry(self.entry_count - 1, cf, child)

    def _loss(self, a, b):
        if self.budget is not None:
            self.budget.charge_distance()
        distance, clamp = _distance(a, b, self.globalvar)
        self.distance_evaluations += 1
        self.roundoff_clamps += clamp
        if self.budget is not None:
            self.budget.clamps += clamp
        return distance

    def _closest(self, entries, incoming):
        return min(
            entries, key=lambda entry: (self._loss(entry.cf, incoming), _key(entry.cf), entry.id)
        )

    def _split(self, node):
        entries = sorted(node.entries, key=lambda entry: (_key(entry.cf), entry.id))
        # Lexicographically first farthest pair resolves exact distance ties.
        pairs = (
            (self._loss(a.cf, b.cf), _key(a.cf), _key(b.cf), a.id, b.id, a, b)
            for i, a in enumerate(entries)
            for b in entries[i + 1 :]
        )
        _, _, _, _, _, first, second = min(pairs, key=lambda pair: (-pair[0], *pair[1:5]))
        left, right = [first], [second]
        for entry in entries:
            if entry is first or entry is second:
                continue
            dl, dr = self._loss(entry.cf, first.cf), self._loss(entry.cf, second.cf)
            if dl < dr or (
                dl == dr
                and (len(left), _key(first.cf), first.id)
                <= (len(right), _key(second.cf), second.id)
            ):
                left.append(entry)
            else:
                right.append(entry)
        sibling = self._node(node.leaf)
        node.entries = sorted(left, key=lambda entry: (_key(entry.cf), entry.id))
        sibling.entries = sorted(right, key=lambda entry: (_key(entry.cf), entry.id))
        self.split_count += 1
        return sibling

    def _insert(self, node, incoming):
        if node.leaf:
            closest = self._closest(node.entries, incoming) if node.entries else None
            if closest is not None and self._loss(closest.cf, incoming) <= self.threshold:
                closest.cf = merge(closest.cf, incoming)
            else:
                if self.precluster_count == self.max_preclusters:
                    raise AnalysisError(
                        "resource_limit",
                        f"TwoStep CF tree exceeds max_preclusters={self.max_preclusters}; increase the limit or choose an explicit larger threshold. No records are silently discarded or rebuilt.",
                    )
                node.entries.append(self._entry(incoming))
                self.precluster_count += 1
        else:
            closest = self._closest(node.entries, incoming)
            sibling = self._insert(closest.child, incoming)
            closest.cf = _summary(closest.child)
            if sibling is not None:
                node.entries.append(self._entry(_summary(sibling), sibling))
        return self._split(node) if len(node.entries) > self.branch_factor else None

    def insert(self, incoming):
        sibling = self._insert(self.root, incoming)
        if sibling is not None:
            left = self.root
            self.root = self._node(False)
            self.root.entries = [
                self._entry(_summary(left), left),
                self._entry(_summary(sibling), sibling),
            ]

    def result(self):
        leaves, depths = [], set()

        def visit(node, depth):
            if node.leaf:
                depths.add(depth)
                leaves.extend(entry.cf for entry in node.entries)
            return {
                "id": node.id,
                "leaf": node.leaf,
                "entries": [
                    {
                        "id": entry.id,
                        "cf": _cf_dict(entry.cf),
                        "child": visit(entry.child, depth + 1) if entry.child is not None else None,
                    }
                    for entry in node.entries
                ],
            }

        serialized = visit(self.root, 1)
        if len(depths) != 1:
            raise AnalysisError(
                "invalid_cf", "TwoStep CF tree failed its balanced leaf-depth audit."
            )
        serialized.update(
            {
                "distance_roundoff_rule": ROUNDING_RULE,
                "automatic_rebuild": False,
                "threshold": self.threshold,
                "branch_factor": self.branch_factor,
            }
        )
        return TreeResult(
            tuple(sorted(leaves, key=_key)),
            serialized,
            self.node_count,
            next(iter(depths)),
            self.split_count,
            self.distance_evaluations,
            self.roundoff_clamps,
        )


def build_tree(
    X: Tensor,
    codes: Tensor,
    levels: tuple[int, ...] | list[int],
    globalvar: Tensor,
    row_order: Tensor,
    *,
    threshold: float,
    branch_factor: int,
    max_preclusters: int,
    max_nodes: int,
    rebuild: bool = False,
    max_rebuilds: int = 16,
    noise_fraction: float = 0.0,
    max_work: int = 300_000_000,
) -> TreeResult:
    """Sequential balanced CF-tree insertion with absolute merge-loss absorption."""
    if (
        not isinstance(X, Tensor)
        or X.device.type != "cpu"
        or X.dtype != torch.float64
        or X.ndim != 2
    ):
        raise AnalysisError(
            "invalid_design", "TwoStep numeric design must be a CPU float64 matrix."
        )
    _finite(X, "numeric input")
    if (
        not isinstance(codes, Tensor)
        or codes.device.type != "cpu"
        or codes.dtype != torch.int64
        or codes.ndim != 2
    ):
        raise AnalysisError(
            "invalid_design", "TwoStep categorical design must be a CPU int64 matrix."
        )
    n, p = X.shape
    q = codes.shape[1]
    if not 1 <= n <= 2000 or codes.shape[0] != n or not 1 <= p + q <= 16:
        raise AnalysisError(
            "invalid_design",
            "TwoStep requires 1..2000 aligned records and 1..16 continuous/categorical features.",
        )
    if (
        not isinstance(levels, (tuple, list))
        or len(levels) != q
        or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= n for v in levels)
    ):
        raise AnalysisError(
            "invalid_design",
            "TwoStep categorical levels must be aligned positive integers no larger than record count.",
        )
    if any(
        not bool(((codes[:, j] >= 0) & (codes[:, j] < length)).all())
        for j, length in enumerate(levels)
    ):
        raise AnalysisError(
            "invalid_design", "TwoStep categorical codes are outside declared levels."
        )
    _global(globalvar, p)
    if (
        not isinstance(row_order, Tensor)
        or row_order.device.type != "cpu"
        or row_order.dtype != torch.int64
        or row_order.ndim != 1
        or len(row_order) != n
        or sorted(row_order.tolist()) != list(range(n))
    ):
        raise AnalysisError(
            "invalid_order",
            "TwoStep row_order must be a complete CPU int64 permutation of physical record positions.",
        )
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise AnalysisError(
            "invalid_threshold",
            "TwoStep threshold must be a finite nonnegative absolute merge loss.",
        )
    for name, value, minimum, maximum in (
        ("branch_factor", branch_factor, 2, 128),
        ("max_preclusters", max_preclusters, 1, 128),
        ("max_nodes", max_nodes, 1, 4096),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise AnalysisError(
                "invalid_tree_limit", f"TwoStep {name} must be an integer in {minimum}..{maximum}."
            )
    if type(rebuild) is not bool or type(max_rebuilds) is not int or not 0 <= max_rebuilds <= 64:
        raise AnalysisError("invalid_tree_limit", "rebuild must be bool and max_rebuilds an integer in 0..64.")
    if type(max_work) is not int or max_work < 1:
        raise AnalysisError("invalid_tree_limit", "max_work must be a positive integer.")
    if isinstance(noise_fraction, bool) or not isinstance(noise_fraction, (int, float)) or not math.isfinite(noise_fraction) or not 0 <= noise_fraction <= 1 or noise_fraction > 0 and not rebuild:
        raise AnalysisError("invalid_tree_limit", "noise_fraction must be finite in 0..1 and requires adaptive rebuilding.")
    if rebuild:
        from .adaptive import build_adaptive
        return build_adaptive(X, codes, tuple(levels), globalvar.detach(), row_order,
                              threshold=float(threshold), branch_factor=branch_factor,
                              max_preclusters=max_preclusters, max_nodes=max_nodes,
                              max_rebuilds=max_rebuilds, noise_fraction=float(noise_fraction), max_work=max_work)
    tree = _Tree(globalvar.detach(), float(threshold), branch_factor, max_preclusters, max_nodes)
    for row in row_order.tolist():
        tree.insert(singleton(X[row], codes[row], tuple(levels), row))
    return tree.result()


def agglomerate(preclusters: tuple[CF, ...] | list[CF], globalvar: Tensor) -> Hierarchy:
    """Complete deterministic closest-pair sequence; cached O(M²) distances.

    Cuts map every k=M..1 to CFs sorted by first physical row.  Merge IDs use
    canonical initial leaves 0..M-1 and sequential new nodes M..2M-2.
    Ties compare cluster minimum rows and then cluster IDs.  Heap entries for
    removed clusters are ignored without recomputing unchanged distances.
    """
    if (
        not isinstance(preclusters, (list, tuple))
        or not 1 <= len(preclusters) <= 128
        or any(not isinstance(cf, CF) for cf in preclusters)
    ):
        raise AnalysisError("invalid_cf", "TwoStep hierarchy requires 1..128 valid precluster CFs.")
    ordered = sorted(preclusters, key=_key)
    _global(globalvar, len(ordered[0].mean))
    rows = [row for cf in ordered for row in cf.rows]
    if len(rows) > 2000 or len(set(rows)) != len(rows):
        raise AnalysisError(
            "invalid_cf", "TwoStep preclusters must partition at most 2000 distinct physical rows."
        )
    active = dict(enumerate(ordered))
    scores = {
        key: _score(cf.count, cf.m2, cf.categorical_counts, globalvar) for key, cf in active.items()
    }
    heap, merges = [], []
    evaluations = clamps = 0

    def push(left, right):
        nonlocal evaluations, clamps
        if (_key(active[left]), left) > (_key(active[right]), right):
            left, right = right, left
        distance, clamp = _distance(
            active[left], active[right], globalvar, (scores[left], scores[right])
        )
        evaluations += 1
        clamps += clamp
        heapq.heappush(heap, (distance, _key(active[left]), _key(active[right]), left, right))

    for left in active:
        for right in active:
            if left < right:
                push(left, right)
    cuts = {len(active): tuple(ordered)}
    next_id = len(active)
    while len(active) > 1:
        while heap:
            distance, _, _, left, right = heapq.heappop(heap)
            if left in active and right in active:
                break
        else:  # pragma: no cover - cache invariant
            raise AnalysisError("invalid_cf", "TwoStep closest-pair distance cache is incomplete.")
        combined = merge(active[left], active[right])
        del active[left], active[right]
        del scores[left], scores[right]
        active[next_id] = combined
        scores[next_id] = _score(
            combined.count, combined.m2, combined.categorical_counts, globalvar
        )
        merges.append(
            {
                "left": left,
                "right": right,
                "merged": next_id,
                "distance": distance,
                "count": combined.count,
                "clusters_after": len(active),
            }
        )
        for other in active:
            if other != next_id:
                push(other, next_id)
        cuts[len(active)] = tuple(sorted(active.values(), key=_key))
        next_id += 1
    return Hierarchy(cuts, tuple(merges), evaluations, clamps)
