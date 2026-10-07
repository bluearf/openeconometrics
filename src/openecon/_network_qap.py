"""Sparse bivariate QAP correlation with an explicit node-exchangeability null.

One uniform node permutation acts on both endpoints of the second network.
Absent dyads are zeros, not missing observations. No dense V-by-V matrix is
constructed, and no independent-dyad standard error is reported.

Dekker, Krackhardt and Snijders (2007), sections 2.2 and 3:
https://www.stats.ox.ac.uk/~snijders/DekkerKrackhardtSnijders.pdf
This is bivariate QAP, not partial correlation or MRQAP regression.
"""
from __future__ import annotations

import math
import sys

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import Network, _error, _integer, _key


class _Work:
    def __init__(self, limit):
        self.limit = _integer(limit, "max_work", 2**63 - 1)
        self.used = 0

    def add(self, count):
        self.used += count
        if self.used > self.limit:
            _error("work_budget", "QAP permutations exceed max_work; increase the explicit "
                   "work budget or request fewer permutations. No partial test is returned.")


def _edge_map(graph, canonical, values, include_loops):
    """Scale positive strengths by an exact power of two before centering."""
    pairs, weights = graph._edges.indices(), graph._edges.values()
    left, right, weight = (memoryview(item.numpy()) for item in (pairs[0], pairs[1], weights))
    maximum = max((weight[i] for i in range(len(weight))
                   if include_loops or left[i] != right[i]), default=0.)
    exponent = math.frexp(maximum)[1] if maximum and values == "weight" else 0
    result = {}
    for i in range(len(weight)):
        if not include_loops and left[i] == right[i]:
            continue
        u, v = canonical[graph._labels[left[i]]], canonical[graph._labels[right[i]]]
        if not graph.directed and u > v:
            u, v = v, u
        value = math.ldexp(weight[i], -exponent) if values == "weight" else 1.
        if (value <= 0 or not math.isfinite(value)
                or (values == "weight" and math.ldexp(value, exponent) != weight[i])):
            _error("precision", "QAP edge scaling loses a positive weight; reduce the "
                   "weight dynamic range or use values='binary' explicitly.")
        result[u, v] = value
    return result, exponent


def _moments(edges, dyads):
    if not edges:
        _error("undefined_statistic", "QAP correlation is undefined for a constant zero network.")
    # For a complete weighted graph, represent the mean as anchor + offset.
    # Computing a rounded mean directly loses sub-ulp variation, e.g. the mean
    # of [1, 1+ulp, 1] rounds to 1 although its true displacement is ulp/3.
    # Sparse graphs use zero as the anchor, naturally including absent dyads.
    anchor = edges[min(edges)] if len(edges) == dyads else 0.
    offset = math.fsum(value - anchor for value in edges.values()) / dyads
    # Centering before squaring avoids subtracting two nearly equal, large
    # moments when a dense weighted network is almost constant.
    terms = (((value - anchor) - offset) ** 2 for value in edges.values())
    zero_centered = -anchor - offset
    variance = math.fsum(terms) + (dyads - len(edges)) * zero_centered * zero_centered
    if not math.isfinite(variance) or variance <= 0:
        _error("undefined_statistic", "QAP correlation requires variation over all included dyads; "
               "a constant network has no defined correlation.")
    return (anchor, offset), math.sqrt(variance)


def _correlation(first, second, first_mean, second_mean, first_norm, second_norm,
                 dyads, directed, permutation=None, inverse=None):
    """Centered products over the sparse union, plus the absent/absent count.

Unlike sum(x*y)-sum(x)*sum(y)/D, these centered products retain precision for
nearly constant dense networks. Both dictionaries are scanned, but storage and
iteration remain O(V+E), even when the dyad universe is enormous.
"""
    intersection = 0
    first_anchor, first_offset = first_mean
    second_anchor, second_offset = second_mean
    first_zero = -first_anchor - first_offset
    second_zero = -second_anchor - second_offset

    def terms():
        nonlocal intersection
        for (u, v), a in first.items():
            x, y = (inverse[u], inverse[v]) if inverse is not None else (u, v)
            if not directed and x > y:
                x, y = y, x
            b = second.get((x, y), 0.)
            intersection += b > 0
            yield ((a - first_anchor) - first_offset) * ((b - second_anchor) - second_offset)
        for (u, v), b in second.items():
            x, y = (permutation[u], permutation[v]) if permutation is not None else (u, v)
            if not directed and x > y:
                x, y = y, x
            if (x, y) not in first:
                yield first_zero * ((b - second_anchor) - second_offset)
        missing = dyads - len(first) - len(second) + intersection
        if missing:
            yield missing * first_zero * second_zero

    # Sequential division avoids overflowing a product of two norms. All
    # products involve centered independently scaled values, never raw weights.
    result = (math.fsum(terms()) / first_norm) / second_norm
    tolerance = 64 * sys.float_info.epsilon
    if not math.isfinite(result) or abs(result) > 1 + tolerance:
        _error("precision", "QAP correlation cannot be represented reliably in float64.")
    return max(-1., min(1., result))


def _presence_correlation(first, second, dyads, directed, permutation=None, inverse=None):
    """Uniform positive weights reduce to binary overlap, with exact centering.

    The integer numerator D*overlap-E1*E2 preserves rare-edge precision without
    constructing any absent dyad, and only the smaller edge set is traversed.
    """
    if len(first) <= len(second):
        selected, lookup, relabel = first, second, inverse
    else:
        selected, lookup, relabel = second, first, permutation
    overlap = 0
    for u, v in selected:
        if relabel is not None:
            u, v = relabel[u], relabel[v]
        if not directed and u > v:
            u, v = v, u
        overlap += (u, v) in lookup
    a, b = len(first), len(second)
    numerator = dyads * overlap - a * b
    result = (numerator / math.sqrt(a * (dyads - a))) / math.sqrt(b * (dyads - b))
    return max(-1., min(1., result))


def qap_correlation(graph, other, *, values="weight", include_loops=False,
                    permutations=999, seed=0, alternative="two-sided", max_work=50_000_000):
    """Pearson dyad correlation and a Monte Carlo QAP randomization p-value.

    The graphs must have identical exact node IDs and directedness. ``weight``
    uses stored aggregate strengths; ``binary`` uses positive-edge presence.
    All eligible absent dyads count as zero. Undirected dyads are counted once.
    Self-loops are excluded unless explicitly included. Uniform node-label
    permutations of ``other`` preserve its complete relational structure.

    This test assumes node exchangeability under the randomization null. It
    does not establish causality, adjust for confounders, or provide a generic
    test of zero population correlation under arbitrary network dependence.
    """
    if not isinstance(other, Network):
        _error("invalid_option", "other must be a Network snapshot.")
    if not isinstance(values, str) or values not in ("weight", "binary"):
        _error("invalid_option", "values must be 'weight' or 'binary'.")
    if not isinstance(include_loops, bool):
        _error("invalid_option", "include_loops must be boolean.")
    if not isinstance(alternative, str) or alternative not in ("two-sided", "greater", "less"):
        _error("invalid_option", "alternative must be 'two-sided', 'greater', or 'less'.")
    permutations = _integer(permutations, "permutations", 1_000_000)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    work = _Work(max_work)
    if graph.directed != other.directed:
        _error("invalid_option", "QAP requires graphs with the same directedness.")
    n = graph.node_count
    if n != other.node_count:
        _error("invalid_label", "QAP requires identical exact node-label sets, including isolates.")
    dyads = n * (n - 1) if graph.directed else n * (n - 1) // 2
    if include_loops:
        dyads += n
    if dyads < 2:
        _error("undefined_statistic", "QAP correlation needs at least two included dyads.")
    if dyads > 2**53:
        _error("precision", "The included dyad count exceeds exact float64 integer range.")
    # Both resident graph snapshots count against BOTH callers' memory budgets;
    # the same object is counted only once. Estimates cover hash-table growth,
    # label alignment, permutation tensors and the compact result frame.
    e = graph.edge_count + other.edge_count
    work.add(3 * n + 4 * e + 32)
    workspace = 512 * n + 512 * e + 8192
    graph._guard(workspace + (0 if other is graph else other._base_bytes))
    if other is not graph:
        other._guard(workspace + graph._base_bytes)
    if any(label not in other._index for label in graph._labels):
        _error("invalid_label", "QAP requires identical exact node-label sets, including isolates.")
    canonical = {label: i for i, label in enumerate(sorted(graph._labels, key=_key))}
    with torch.device("cpu"), torch.no_grad():
        first, first_exponent = _edge_map(graph, canonical, values, include_loops)
        second, second_exponent = _edge_map(other, canonical, values, include_loops)
        first_mean, first_norm = _moments(first, dyads)
        second_mean, second_norm = _moments(second, dyads)
        # Positive constant weights cancel from a Pearson correlation exactly;
        # use the overlap shortcut even for weight mode when both supports are
        # uniform. Complete or empty constant dyad vectors were rejected above.
        first_value, second_value = next(iter(first.values())), next(iter(second.values()))
        uniform = all(value == first_value for value in first.values()) and all(
            value == second_value for value in second.values())
        work.add(2 * (len(first) + len(second)) + 1)
        def correlation(permutation=None, inverse=None):
            if uniform:
                return _presence_correlation(first, second, dyads, graph.directed, permutation, inverse)
            return _correlation(first, second, first_mean, second_mean, first_norm,
                                second_norm, dyads, graph.directed, permutation, inverse)

        observed = correlation()
        # Preflight the complete requested test before allocating permutation
        # tensors. There is no early-stop/sampled-dyad fallback and no incomplete
        # p-value if the requested number of permutations exceeds its budget.
        per_permutation = 3 * n + (min(len(first), len(second)) if uniform else
                                   2 * (len(first) + len(second))) + 1
        work.add(permutations * per_permutation)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        identity = torch.arange(n, dtype=torch.int64, device="cpu")
        inverse_tensor = torch.empty(n, dtype=torch.int64, device="cpu")
        inverse = memoryview(inverse_tensor.numpy())
        extreme = 0
        comparison_tolerance = 64 * sys.float_info.epsilon
        for _ in range(permutations):
            permutation_tensor = torch.randperm(n, dtype=torch.int64, device="cpu", generator=generator)
            inverse_tensor[permutation_tensor] = identity
            permutation = memoryview(permutation_tensor.numpy())
            statistic = correlation(permutation, inverse)
            if alternative == "two-sided":
                extreme += abs(statistic) >= abs(observed) - comparison_tolerance
            elif alternative == "greater":
                extreme += statistic >= observed - comparison_tolerance
            else:
                extreme += statistic <= observed + comparison_tolerance
    pvalue = (extreme + 1) / (permutations + 1)
    result = as_frame(pd.DataFrame({"correlation": [observed], "pvalue": [pvalue],
        "permutations": [permutations], "extreme_permutations": [extreme],
        "dyads": [dyads], "nodes": [n]}))
    result.attrs.update(kind="network_qap_correlation", algorithm="sparse centered bivariate QAP",
        values=values, weight_semantics="aggregate positive strength" if values == "weight" else "positive-edge presence",
        directed=graph.directed, include_loops=include_loops, absent_dyads="zero",
        undirected_dyads="unordered; each pair counted once", alternative=alternative,
        seed=seed, rng="private CPU Torch generator; canonical typed-label order", torch_version=torch.__version__,
        permutation="uniform with replacement; same node permutation applied to both endpoints of other",
        pvalue_method="(1 + extreme_permutations) / (1 + permutations)",
        pvalue_resolution=1 / (permutations + 1), comparison_tolerance=comparison_tolerance,
        two_sided_definition="absolute correlation at least as large as the observed absolute correlation",
        null="Node-label exchangeability of the second network relative to the first",
        inference_scope="bivariate randomization association; no causal claim, confounder adjustment or MRQAP",
        memory_scope="O(V+E) workspace plus both resident snapshots under both memory budgets",
        scaled_weight_exponents=[first_exponent, second_exponent],
        correlation_algorithm="exact integer overlap numerator" if uniform else "centered sparse union",
        device="cpu", dtype="float64", work_used=work.used, max_work=work.limit)
    return result
