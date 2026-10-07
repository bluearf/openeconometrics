"""Native sparse degree-corrected Poisson hard-label block modelling.

Karrer and Newman (2011), https://arxiv.org/abs/1008.3926, use an undirected
multigraph with eligible self-loops. Stored loop strengths here are *actual*
loop counts, not twice those counts. All self-loop dyads remain eligible even
when their observed count is zero. Directed graphs use separate normalized
out-/in-degree parameters and the analogous ordered Poisson model.

Integer sufficient statistics and CSR buffers are CPU Torch. The implementation
uses O(V+E+K^2) storage and O(E+VK^2) work per coordinate-ascent sweep. It fits
a real profile likelihood, not modularity, and certifies only a single-node
local optimum within the stated numerical tolerance. No SciPy is used.
"""
from __future__ import annotations

import math
import sys

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key, _real
from openecon._network_sbm import NetworkBlockResult, _initial, _structural_start, _view, _Work
from openecon._network_sparse import csr
from openecon._network_count_diagnostics import restart_seed, run_diagnostics


_MAX_COUNT = 2**53


def _log_ratio(numerator, denominator):
    """Log an exact positive integer ratio without losing its near-one delta."""
    if numerator <= 0 or denominator <= 0:
        _error("precision", "Positive observed counts require positive fitted means.")
    difference = numerator - denominator
    if abs(difference) <= denominator // 2:
        return math.log1p(difference / denominator)
    return math.log(numerator) - math.log(denominator)


def _saturated(count):
    """log Pois(count; count), retaining precision at large integer counts.

    Stirling's log-factorial correction avoids cancellation between O(c log c)
    terms. The alternating expansion through c^-11 has error below 2e-18 at
    c=32; small counts use the native log-gamma function directly.
    """
    if count < 32:
        return count * math.log(count) - count - math.lgamma(count + 1)
    inverse = 1. / count
    square = inverse * inverse
    correction = inverse * (1. / 12 + square * (-1. / 360 + square *
        (1. / 1260 + square * (-1. / 1680 + square *
        (1. / 1188 - square * 691. / 360360)))))
    return -.5 * math.log(2 * math.pi * count) - correction


def _entropy_delta(old, new):
    """Difference x log x, using exact integer differences and stable ratios."""
    if old < 0 or new < 0:
        _error("precision", "Poisson block sufficient counts became negative.")
    if old == new:
        return 0.
    if not old:
        return new * math.log(new)
    if not new:
        return -old * math.log(old)
    # Choosing the larger anchor keeps the ratio bounded away from underflow;
    # the exact integer delta retains unit differences at the 2^53 boundary.
    delta = new - old
    if new >= old:
        return delta * math.log(new) + old * _log_ratio(new, old)
    return delta * math.log(old) + new * _log_ratio(new, old)


def _counts(graph, work):
    """Validate the complete stored count graph before any iterative fitting."""
    work.add(graph.node_count + 2 * graph.edge_count + 8)
    n = graph.node_count
    outgoing_t = torch.zeros(n, dtype=torch.int64, device="cpu")
    incoming_t = torch.zeros(n, dtype=torch.int64, device="cpu")
    loops_t = torch.zeros(n, dtype=torch.int64, device="cpu")
    outgoing, incoming, loops = _view(outgoing_t), _view(incoming_t), _view(loops_t)
    pairs = graph._edges.indices()
    sources, targets, weights = _view(pairs[0]), _view(pairs[1]), _view(graph._edges.values())
    total, loop_total = 0, 0
    for source, target, raw in zip(sources, targets, weights):
        if not raw.is_integer() or not 0 < raw <= _MAX_COUNT:
            _error("invalid_weight", "Degree-corrected Poisson SBM requires positive exact integer "
                   "stored count strengths no larger than 2^53; continuous weights are not counts.")
        count = int(raw)
        total += count
        if total > _MAX_COUNT or (not graph.directed and 2 * total > _MAX_COUNT):
            _error("precision", "Total counts and undirected total stubs must remain within 2^53.")
        outgoing[source] += count
        if graph.directed:
            incoming[target] += count
        elif source == target:
            outgoing[source] += count
        else:
            outgoing[target] += count
        if source == target:
            loops[source] = count
            loop_total += count
    return outgoing_t, incoming_t, loops_t, total, loop_total


class _State:
    """Exact counts for a single partition, with sparse single-node updates."""

    def __init__(self, graph, membership, groups, outgoing_degree, incoming_degree, total, work):
        self.graph, self.groups, self.work, self.total = graph, groups, work, total
        self.degree_out, self.degree_in = _view(outgoing_degree), _view(incoming_degree)
        self.membership_t = torch.tensor(membership, dtype=torch.int64, device="cpu")
        self.sizes_t = torch.bincount(self.membership_t, minlength=groups)
        self.counts_t = torch.zeros(groups * groups, dtype=torch.int64, device="cpu")
        self.out_stubs_t = torch.zeros(groups, dtype=torch.int64, device="cpu")
        self.in_stubs_t = torch.zeros(groups, dtype=torch.int64, device="cpu")
        self.membership, self.sizes = _view(self.membership_t), _view(self.sizes_t)
        self.counts = _view(self.counts_t)
        self.out_stubs, self.in_stubs = _view(self.out_stubs_t), _view(self.in_stubs_t)
        work.add(graph.node_count + graph.edge_count + groups * groups + 2 * groups)
        for node, block in enumerate(self.membership):
            self.out_stubs[block] += self.degree_out[node]
            self.in_stubs[block] += self.degree_in[node]
        pairs = graph._edges.indices()
        for source, target, weight in zip(_view(pairs[0]), _view(pairs[1]), _view(graph._edges.values())):
            self.counts[self.index(self.membership[source], self.membership[target])] += int(weight)

    def index(self, source, target):
        if not self.graph.directed and source > target:
            source, target = target, source
        return source * self.groups + target

    def omega(self, source, target):
        count = self.counts[self.index(source, target)]
        return 2 * count if not self.graph.directed and source == target else count

    def likelihood(self):
        """Complete profile Poisson LL, including factorials and eligible zeros.

        Total fitted expected multiplicity equals total observed multiplicity,
        so -sum(lambda) cancels the +sum(count) in the saturated log masses.
        The remaining sparse terms use exact integer mean/count ratios. This
        cancellation includes all absent dyads; they have not been omitted.
        """
        self.work.add(4 * self.graph.edge_count + 1)
        pairs = self.graph._edges.indices()
        sources, targets, weights = _view(pairs[0]), _view(pairs[1]), _view(self.graph._edges.values())
        def terms():
            for source, target, raw in zip(sources, targets, weights):
                count = int(raw)
                a, b = self.membership[source], self.membership[target]
                numerator = self.degree_out[source] * self.omega(a, b)
                if self.graph.directed:
                    numerator *= self.degree_in[target]
                    denominator = self.out_stubs[a] * self.in_stubs[b] * count
                else:
                    numerator *= self.degree_out[target]
                    denominator = self.out_stubs[a] * self.out_stubs[b] * count
                    if source == target:
                        denominator *= 2
                yield count * _log_ratio(numerator, denominator)
                yield _saturated(count)
        result = math.fsum(terms())
        if not math.isfinite(result):
            _error("precision", "Poisson profile likelihood became nonfinite.")
        return result

    def move(self, node, target, outgoing, incoming, loop_count, *, apply=False):
        """Evaluate incident block count and group stub changes in O(K)."""
        old, groups = self.membership[node], self.groups
        self.work.add(24 * groups + 16)
        deltas = {}
        def add(source, target, delta):
            at = self.index(source, target)
            deltas[at] = deltas.get(at, 0) + delta
        for other in range(groups):
            add(old, other, -outgoing[other])
            add(target, other, outgoing[other])
            if self.graph.directed:
                add(other, old, -incoming[other])
                add(other, target, incoming[other])
        add(old, old, -loop_count)
        add(target, target, loop_count)
        changes, terms = [], []
        for at in sorted(deltas):
            before, after = self.counts[at], self.counts[at] + deltas[at]
            source, destination = divmod(at, groups)
            if not self.graph.directed and source == destination:
                value = .5 * _entropy_delta(2 * before, 2 * after)
            else:
                value = _entropy_delta(before, after)
            terms.append(value)
            changes.append((at, after))
        out_degree, in_degree = self.degree_out[node], self.degree_in[node]
        terms.extend([-_entropy_delta(self.out_stubs[old], self.out_stubs[old] - out_degree),
                      -_entropy_delta(self.out_stubs[target], self.out_stubs[target] + out_degree)])
        if self.graph.directed:
            terms.extend([-_entropy_delta(self.in_stubs[old], self.in_stubs[old] - in_degree),
                          -_entropy_delta(self.in_stubs[target], self.in_stubs[target] + in_degree)])
        gain = math.fsum(terms)
        rounding = 128 * sys.float_info.epsilon * max(1., math.fsum(map(abs, terms)))
        if apply:
            for at, count in changes:
                self.counts[at] = count
            self.out_stubs[old] -= out_degree
            self.out_stubs[target] += out_degree
            self.in_stubs[old] -= in_degree
            self.in_stubs[target] += in_degree
            self.sizes[old] -= 1
            self.sizes[target] += 1
            self.membership[node] = target
        return gain, rounding


def degree_corrected_block_model(graph, groups, *, initial=None, seed=0, starts=4,
                                 max_iter=100, tol=1e-9, max_work=50_000_000):
    """Fit a fixed-K degree-corrected Poisson multigraph SBM.

    Every self-loop dyad is eligible. Undirected stored loops are actual loop
    counts; their fitted mean is theta_i^2 * omega_rr / 2. Off-diagonal fitted
    means are theta_i * theta_j * omega_rs. Directed means instead use
    theta_out_i * theta_in_j * omega_rs, including loops without a half factor.

    Strengths must be exact nonnegative integer counts after graph coalescing,
    with total counts (and undirected total stubs) at most 2^53. Memberships
    remain nonempty. The first non-user start uses sparse binary adjacency
    profiles when edges exist; other starts are balanced private-CPU-Torch
    random partitions. Empty graphs use balanced starts. max_iter=0 requires a
    supplied initial partition and starts=1. A finite iteration limit does not
    certify convergence. No automatic K selection or membership uncertainty is
    provided. Zero-stub groups export theta=0 as an unidentified convention.
    """
    groups = _integer(groups, "groups", 128)
    starts = _integer(starts, "starts", 64)
    max_iter = _integer(max_iter, "max_iter", 10_000, zero=True)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    tol = _real(tol, "invalid_option", "tol must be finite and nonnegative.")
    if tol < 0:
        _error("invalid_option", "tol must be finite and nonnegative.")
    n, e, a = graph.node_count, graph.edge_count, graph._arcs._nnz()
    if groups > n:
        _error("invalid_option", "groups cannot exceed nodes; every requested group must be nonempty.")
    if not max_iter and (initial is None or starts != 1):
        _error("invalid_option", "max_iter=0 requires initial membership and starts=1.")
    dyads = n * n if graph.directed else n * (n + 1) // 2
    if dyads > _MAX_COUNT:
        _error("precision", "Eligible self-loop-inclusive dyads exceed exact float64 integer range.")
    work = _Work(max_work)
    work.add(n * (3 + n.bit_length()) + 4 * a + 4 * e + 32)
    minimum = n + 2 * e + 8 + starts * (2 * n + 5 * e + groups * groups + 2 * groups + 1)
    minimum += 4 * n + 12 * groups * groups
    structural = groups > 1 and e > 0 and (initial is None or starts > 1)
    if structural:
        # The per-start n charge above already covers the initialization's
        # label vector. Admit the remaining structural work before CSR setup.
        minimum += groups * (4 * n + 2 * a + groups)
    if work.used + minimum > work.limit:
        _error("work_budget", "Requested degree-corrected block-model setup exceeds max_work; no fit is returned.")
    needs_moves = groups > 1 and max_iter > 0
    workspace = (768 * n + (192 * a if needs_moves else 0) + 768 * groups * groups
                 + 768 * groups + starts * (1024 + 160 * (max_iter + 1)) + 32768)
    graph._guard(workspace)
    with torch.device("cpu"), torch.no_grad():
        initial_labels = _initial(graph, initial, groups) if initial is not None else None
        canonical = sorted(range(n), key=lambda node: _key(graph._labels[node]))
        out_degree, in_degree, loop_t, total, loop_total = _counts(graph, work)
        loops = _view(loop_t)
        outgoing_csr = csr(graph, loops=False) if needs_moves else None
        incoming_csr = csr(graph, reverse=True, loops=False) if needs_moves and graph.directed else None
        outgoing_t = torch.zeros(groups, dtype=torch.int64, device="cpu")
        incoming_t = torch.zeros(groups, dtype=torch.int64, device="cpu")
        outgoing, incoming = _view(outgoing_t), _view(incoming_t)
        best, best_likelihood, runs = None, -math.inf, []
        for start in range(starts):
            generator = torch.Generator(device="cpu").manual_seed(restart_seed(seed, start))
            if start == 0 and initial_labels is not None:
                labels, initialization = initial_labels, "user"
            elif groups == 1:
                labels, initialization = [0] * n, "unique one-group partition"
            elif structural and start == (1 if initial_labels is not None else 0):
                labels = _structural_start(graph, groups, canonical, outgoing_csr, incoming_csr, work)
                initialization = "sparse binary structural-distance seeds"
            else:
                work.add(n)
                order_t = torch.randperm(n, generator=generator, device="cpu")
                labels = [0] * n
                for position, rank in enumerate(_view(order_t)):
                    labels[canonical[rank]] = position % groups
                initialization = "balanced random"
            state = _State(graph, labels, groups, out_degree, in_degree, total, work)
            history, converged, iterations, moves = [state.likelihood()], groups == 1, 0, 0
            membership_changes = []
            if groups > 1:
                for iteration in range(max_iter):
                    work.add(n)
                    order_t = torch.randperm(n, generator=generator, device="cpu")
                    sweep_moves = 0
                    for rank in _view(order_t):
                        node = canonical[rank]
                        old = state.membership[node]
                        if state.sizes[old] <= 1 or not (state.degree_out[node] or state.degree_in[node]):
                            # A zero-stub node changes neither the block counts
                            # nor degree-normalization denominators. Every
                            # candidate has exactly zero profile gain.
                            work.add(1)
                            continue
                        degree = len(outgoing_csr.neighbors(node))
                        if incoming_csr is not None:
                            degree += len(incoming_csr.neighbors(node))
                        work.add(degree + 2 * groups + 1)
                        outgoing_t.zero_()
                        incoming_t.zero_()
                        for neighbor, raw in outgoing_csr.row(node):
                            outgoing[state.membership[neighbor]] += int(raw)
                        if incoming_csr is not None:
                            for neighbor, raw in incoming_csr.row(node):
                                incoming[state.membership[neighbor]] += int(raw)
                        target, largest = old, 0.
                        for candidate in range(groups):
                            if candidate == old:
                                continue
                            gain, rounding = state.move(node, candidate, outgoing, incoming, loops[node])
                            if gain > tol + rounding and gain > largest:
                                target, largest = candidate, gain
                        if target != old:
                            state.move(node, target, outgoing, incoming, loops[node], apply=True)
                            sweep_moves += 1
                    iterations, moves = iteration + 1, moves + sweep_moves
                    membership_changes.append(sweep_moves)
                    likelihood = state.likelihood()
                    if likelihood < history[-1] - 256 * sys.float_info.epsilon * max(1., total, abs(history[-1])):
                        _error("precision", "Degree-corrected Poisson ascent lost likelihood beyond rounding tolerance.")
                    history.append(likelihood)
                    if not sweep_moves:
                        converged = True
                        break
            likelihood = history[-1]
            runs.append(dict(start=start, initialization=initialization, iterations=iterations, moves=moves,
                converged=converged, log_likelihood=likelihood, likelihood_history=history,
                **run_diagnostics(seed, start, groups, max_iter, history, membership_changes)))
            if likelihood > best_likelihood or (likelihood == best_likelihood and converged
                    and best is not None and not runs[best[1]]["converged"]):
                best_likelihood, best = likelihood, (state, start)
        state, selected_start = best
        work.add(4 * n + 12 * groups * groups)
        group_order, seen = [], set()
        for node in canonical:
            block = state.membership[node]
            if block not in seen:
                seen.add(block)
                group_order.append(block)
        remap = {old: new for new, old in enumerate(group_order)}
        columns = {"block": [remap[state.membership[node]] for node in range(n)]}
        if graph.directed:
            columns.update(theta_out=[state.degree_out[node] / state.out_stubs[state.membership[node]]
                if state.out_stubs[state.membership[node]] else 0. for node in range(n)],
                theta_in=[state.degree_in[node] / state.in_stubs[state.membership[node]]
                if state.in_stubs[state.membership[node]] else 0. for node in range(n)],
                theta_out_identified=[bool(state.out_stubs[state.membership[node]]) for node in range(n)],
                theta_in_identified=[bool(state.in_stubs[state.membership[node]]) for node in range(n)])
        else:
            columns.update(theta=[state.degree_out[node] / state.out_stubs[state.membership[node]]
                if state.out_stubs[state.membership[node]] else 0. for node in range(n)],
                theta_identified=[bool(state.out_stubs[state.membership[node]]) for node in range(n)])
        membership = graph._frame(columns, kind="network_block_membership", fixed_groups=True,
            method="degree_corrected_poisson_sbm", model="degree_corrected_poisson")
        rows = []
        for source in range(groups):
            for target in range(0 if graph.directed else source, groups):
                old_source, old_target = group_order[source], group_order[target]
                count = state.counts[state.index(old_source, old_target)]
                omega = state.omega(old_source, old_target)
                stub_source = state.out_stubs[old_source]
                stub_target = (state.in_stubs if graph.directed else state.out_stubs)[old_target]
                expected = omega if graph.directed or source != target else omega / 2
                rows.append((source, target, state.sizes[old_source], state.sizes[old_target],
                             stub_source, stub_target, count, omega, expected, bool(stub_source and stub_target)))
        blocks = as_frame(pd.DataFrame(rows, columns=["source_block", "target_block", "nodes_source",
            "nodes_target", "stubs_source", "stubs_target", "count_strength", "omega", "expected_count", "identified"]))
        blocks.attrs.update(kind="network_degree_corrected_poisson_blocks", model="degree_corrected_poisson",
            directed=graph.directed, self_loops=True, omega_semantics=("ordered fitted block count" if graph.directed
            else "off-diagonal fitted count; diagonal twice actual within-block count"))
        selected = runs[selected_start]
        metadata = dict(kind="network_block_model", model="degree_corrected_poisson", degree_corrected=True,
            algorithm="sparse hard-label profile ascent", nodes=n, groups=groups, fixed_groups=True,
            directed=graph.directed, dyads=dyads, topology_edges=e, count_strength=total,
            observed_loop_count=loop_total, self_loops_excluded=0, self_loops=True,
            self_loops_eligible=True, values="integer counts", device="cpu", dtype="float64",
            weight_semantics="stored coalesced positive integer multiplicity; undirected loop is actual count",
            parallel_edges="coalesced multiplicities add", log_likelihood=best_likelihood,
            likelihood_scope="complete conditional Poisson multigraph likelihood including eligible zero dyads, factorials and loop factors",
            converged=selected["converged"], iterations=selected["iterations"], starts=starts,
            stopping_reason=selected["stopping_reason"], selected_seed=selected["seed"],
            selected_start=selected_start, start_fits=runs, seed=seed, max_iter=max_iter, tol=tol,
            initialization_scope="optional first user membership; first non-user start uses sparse binary adjacency profiles when edges exist; others balanced private Torch random",
            restart_rng="independent private CPU Torch generator; seed=(seed+start) modulo 2**63",
            parameter_change_scope="hard memberships changed per sweep; conditional node/block rates reprofiled exactly",
            convergence_scope=("unique one-group partition; no admissible move" if groups == 1 else
                "completed sweep with no admissible single-node gain exceeding tol plus numerical guard"),
            numerical_tolerance="128*float64_epsilon*max(1,sum(abs(changed entropy contributions)))",
            global_optimum_certified=False, membership_uncertainty="not estimated",
            model_selection="fixed K supplied by caller; no automatic selection or likelihood-ratio p-value",
            block_label_semantics="canonical smallest typed node label; labels interchangeable",
            zero_stub_blocks="theta=0 is an unidentified convention; identification flags exported explicitly",
            degree_parameter_scope=("separate theta_out/theta_in normalized in each positive-stub group"
                if graph.directed else "theta normalized in each positive-stub group"),
            sampled=False, memory_scope="estimated owned graph and O(V+E+K^2) workspace; excludes process RSS and caller input",
            estimated_workspace_bytes=workspace, work_used=work.used, max_work=work.limit,
            work_scope="charged node/edge and candidate block-update units; not CPU instruction count",
            torch_version=torch.__version__)
        return NetworkBlockResult(membership=membership, blocks=blocks, metadata=metadata)
