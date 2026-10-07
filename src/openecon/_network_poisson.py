"""Native fixed-K hard-label count-valued Poisson stochastic block models.

Each eligible off-diagonal dyad has an independent Poisson count with a mean
shared by its hard-membership block. Absent dyads are zeros; directed dyads are
ordered, undirected dyads are unordered and counted once. The MLE block mean
is count sum / eligible dyads. This is the ordinary Poisson SBM, restricted to
self-loop-free observations, rather than a degree-corrected, continuous-weight
or mixed-membership model. Reference: Karrer and Newman (2011), section II,
https://arxiv.org/abs/1008.3926 . Their self-loop convention is not used here.

Sparse Torch CSR traversal and exact int64 sufficient counts use O(V+E+K^2)
storage and O(E+VK^2) work per sweep. Hard-label coordinate ascent certifies
only a tolerance-based single-node local optimum after a complete no-move
sweep. Full likelihoods include Poisson factorial constants and are evaluated
as saturated log masses minus nonnegative deviances to avoid cancellation.
"""
from __future__ import annotations

import math
import sys

import pandas as pd
import torch

from openecon._network_sbm import (
    NetworkBlockResult, _Work, _initial, _potential, _structural_start, _view,
)
from openecon._network_sparse import csr
from openecon._network_count_diagnostics import restart_seed, run_diagnostics
from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key, _real

_EXACT_COUNT = 2**53
_LOG_TWO_PI = math.log(2 * math.pi)


def _profile(count, dyads):
    """Parameter-dependent profile term; the omitted -total count is fixed."""
    if count < 0 or dyads < 0 or (count and not dyads):
        _error("precision", "Poisson block counts and possible dyads became inconsistent.")
    return count * math.log(count / dyads) if count else 0.


def _saturated(count):
    """Poisson log mass at its own integer count, including log-factorial.

    Direct lgamma is stable for small counts. At count>=32, use the Stirling
    log-factorial remainder through 1/(1680 count^7), whose next omitted term
    has absolute magnitude below 1/(1188 count^9) (<2.4e-17 at32). Computing
    count*log(count)-count-lgamma(count+1) there would lose the small log mass.
    """
    if not count:
        return 0.
    if count < 32:
        return math.fsum([count * math.log(count), -count, -math.lgamma(count + 1)])
    inverse = 1. / count
    square = inverse * inverse
    correction = inverse * (1 / 12 + square * (-1 / 360 + square * (1 / 1260 - square / 1680)))
    return -.5 * (_LOG_TWO_PI + math.log(count)) - correction


def _deviance(count, block_count, dyads):
    """Half the Poisson deviance at an exact sufficient-count MLE rate.

    The rational numerator measures closeness before float conversion. A
    convergent series evaluates (1+r)*log(1+r)-r near r=0; direct subtraction
    would erase tiny positive deviations at large counts.
    """
    difference = count * dyads - block_count
    if not difference:
        return 0.
    rate = block_count / dyads
    relative = difference / block_count
    if abs(relative) < .125:
        power = relative * relative
        terms = []
        for order in range(2, 34):
            terms.append(power * (1 if order % 2 == 0 else -1) / (order * (order - 1)))
            power *= relative
        value = rate * math.fsum(terms)
    else:
        value = math.fsum([count * math.log(count * dyads / block_count), -difference / dyads])
    if value < 0 or not math.isfinite(value):
        _error("precision", "Poisson deviance could not be evaluated as a finite nonnegative value.")
    return value


class _State:
    """One sparse partition with exact count and topology sufficient buffers."""

    def __init__(self, graph, membership, groups, work, saturated):
        self.graph, self.groups, self.work, self.saturated = graph, groups, work, saturated
        self.membership_t = torch.tensor(membership, dtype=torch.int64, device="cpu")
        self.sizes_t = torch.bincount(self.membership_t, minlength=groups)
        self.counts_t = torch.zeros(groups * groups, dtype=torch.int64, device="cpu")
        self.topology_t = torch.zeros(groups * groups, dtype=torch.int64, device="cpu")
        self.terms_t = torch.zeros(groups * groups, dtype=torch.float64, device="cpu")
        self.membership, self.sizes = _view(self.membership_t), _view(self.sizes_t)
        self.counts, self.topology, self.terms = _view(self.counts_t), _view(self.topology_t), _view(self.terms_t)
        work.add(graph.node_count + graph.edge_count + groups * groups)
        pairs, weights = graph._edges.indices(), _view(graph._edges.values())
        source, target = _view(pairs[0]), _view(pairs[1])
        for u, v, weight in zip(source, target, weights):
            if u != v:
                at = self.index(self.membership[u], self.membership[v])
                self.counts[at] += int(weight)
                self.topology[at] += 1
        for a, b in self.cells():
            at = self.index(a, b)
            self.terms[at] = _profile(self.counts[at], _potential(a, b, self.sizes, graph.directed))

    def index(self, a, b):
        if not self.graph.directed and a > b:
            a, b = b, a
        return a * self.groups + b

    def cells(self):
        for a in range(self.groups):
            for b in range(0 if self.graph.directed else a, self.groups):
                yield a, b

    def likelihood(self):
        """Actual Poisson likelihood, not merely an unnormalized objective."""
        self.work.add(self.graph.edge_count + self.groups * self.groups)
        pairs, weights = self.graph._edges.indices(), _view(self.graph._edges.values())
        source, target = _view(pairs[0]), _view(pairs[1])
        def contributions():
            for a, b in self.cells():
                at = self.index(a, b)
                count, possible = self.counts[at], _potential(a, b, self.sizes, self.graph.directed)
                if count:
                    yield (possible - self.topology[at]) * (count / possible)
            for u, v, weight in zip(source, target, weights):
                if u != v:
                    a, b = self.membership[u], self.membership[v]
                    at = self.index(a, b)
                    yield _deviance(int(weight), self.counts[at], _potential(a, b, self.sizes, self.graph.directed))
        value = self.saturated - math.fsum(contributions())
        if not math.isfinite(value) or value > 0:
            _error("precision", "Poisson profile likelihood became nonfinite or positive.")
        return value

    def move(self, node, target, outgoing, incoming, *, apply=False, outgoing_topology=None,
             incoming_topology=None):
        """Update only blocks touching the old or target group, including zeros."""
        old = self.membership[node]
        groups, directed = self.groups, self.graph.directed
        self.work.add(24 * groups + 8)
        deltas, topology_deltas = {}, {}
        def add(a, b, delta, topology_delta):
            at = self.index(a, b)
            deltas[at] = deltas.get(at, 0) + delta
            topology_deltas[at] = topology_deltas.get(at, 0) + topology_delta
        for other in range(groups):
            out_top = outgoing_topology[other] if outgoing_topology is not None else 0
            add(old, other, -outgoing[other], -out_top)
            add(target, other, outgoing[other], out_top)
            if directed:
                in_top = incoming_topology[other] if incoming_topology is not None else 0
                add(other, old, -incoming[other], -in_top)
                add(other, target, incoming[other], in_top)
        sizes = list(self.sizes)
        sizes[old] -= 1
        sizes[target] += 1
        changes = []
        for at in sorted(deltas):
            a, b = divmod(at, groups)
            count = self.counts[at] + deltas[at]
            value = _profile(count, _potential(a, b, sizes, directed))
            changes.append((at, count, self.topology[at] + topology_deltas[at], value))
        gain = math.fsum(value - self.terms[at] for at, _, _, value in changes)
        rounding = 64 * sys.float_info.epsilon * max(1.,
            math.fsum(abs(self.terms[at]) + abs(value) for at, _, _, value in changes))
        if apply:
            if outgoing_topology is None or (directed and incoming_topology is None):
                raise ValueError("Applying a Poisson move requires incident topology counts.")
            for at, count, topology, value in changes:
                self.counts[at], self.topology[at], self.terms[at] = count, topology, value
            self.sizes[old] -= 1
            self.sizes[target] += 1
            self.membership[node] = target
        return gain, rounding


def poisson_block_model(graph, groups, *, initial=None, seed=0, starts=4, max_iter=100,
                        tol=1e-9, max_work=50_000_000):
    """Fit a fixed-K independent count-valued Poisson SBM by sparse ascent.

    All stored weights, including excluded self-loops, must be positive exact
    integers <=2**53. The off-diagonal total must not exceed 2**53; arbitrary
    continuous strengths are rejected. Input
    duplicates are already coalesced by Network and represent aggregate counts.
    Validation applies to the stored float64 snapshot, not original input
    provenance: fractional inputs whose coalesced sum is an integer are accepted,
    and original integers rounded during import cannot be reconstructed here.
    Self-loops are excluded. Block `edges` is the count total, `mean_count` its
    count-per-eligible-dyad MLE; unobserved zero-dyad means are NaN. Absent dyads
    are zero observations. No Bernoulli probability or standard error is implied.

    All fixed groups remain nonempty. Initial mappings/tables use exact node
    labels; vectors use snapshot node order. The first non-user start uses the
    native binary adjacency-profile structural initializer (only an initializer
    for the count objective); other starts are balanced random. A private CPU
    Torch generator controls all random starts and node sweep order. Exported
    block IDs are canonicalized by their smallest typed node label.

    max_iter=0 requires a full initial partition and starts=1. A truncated fit
    has converged=False. A complete no-move sweep certifies only a single-node
    local optimum within tol plus conservative float64 profile-term rounding;
    it never certifies global optimality, selects K, or estimates uncertainty.
    """
    groups = _integer(groups, "groups", 128)
    starts = _integer(starts, "starts", 64)
    max_iter = _integer(max_iter, "max_iter", 10_000, zero=True)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    tol = _real(tol, "invalid_option", "tol must be finite and nonnegative.")
    if tol < 0:
        _error("invalid_option", "Poisson block modelling requires nonnegative tol.")
    n, e, a = graph.node_count, graph.edge_count, graph._arcs._nnz()
    if groups > n:
        _error("invalid_option", "groups cannot exceed node count; every group must be nonempty.")
    if not max_iter and (initial is None or starts != 1):
        _error("invalid_option", "max_iter=0 requires initial membership and starts=1.")
    dyads = n * (n - 1) if graph.directed else n * (n - 1) // 2
    if dyads > _EXACT_COUNT:
        _error("precision", "Eligible dyad count exceeds exact float64 integer range.")
    work = _Work(max_work)
    work.add(n * (3 + n.bit_length()) + 4 * a + 8 * e + 32)
    minimum = starts * (n + 3 * e + 3 * groups * groups) + 2 * n + 8 * groups * groups
    if groups > 1 and (initial is None or starts > 1):
        minimum += n + groups * (4 * n + 2 * a + groups)
    if groups > 1:
        structured = int(initial is None or starts > 1)
        minimum += (starts - int(initial is not None) - structured) * n
    if work.used + minimum > work.limit:
        _error("work_budget", "Requested Poisson start setup exceeds max_work; no fit is returned.")
    needs_moves = groups > 1 and max_iter > 0
    workspace = (512 * n + (192 * a if needs_moves else 0) + 640 * groups * groups + 640 * groups
                 + starts * (1024 + 160 * (max_iter + 1)) + 32768)
    graph._guard(workspace)
    with torch.device("cpu"), torch.no_grad():
        user_initial = _initial(graph, initial, groups) if initial is not None else None
        pairs, weights = graph._edges.indices(), _view(graph._edges.values())
        source, target = _view(pairs[0]), _view(pairs[1])
        total, topology_edges = 0, 0
        def saturated_masses():
            nonlocal total, topology_edges
            for u, v, weight in zip(source, target, weights):
                if not weight.is_integer() or weight > _EXACT_COUNT:
                    _error("count_weights", "Poisson SBM requires exact nonnegative integer counts <=2**53.")
                if u == v:
                    continue
                count = int(weight)
                total += count
                topology_edges += 1
                if total > _EXACT_COUNT:
                    _error("precision", "Total off-diagonal count exceeds exact float64 integer range (2**53).")
                yield _saturated(count)
        saturated = math.fsum(saturated_masses())
        canonical = sorted(range(n), key=lambda node: _key(graph._labels[node]))
        outgoing_csr = csr(graph, loops=False) if needs_moves else None
        incoming_csr = csr(graph, reverse=True, loops=False) if needs_moves and graph.directed else None
        buffers = [torch.zeros(groups, dtype=torch.int64, device="cpu") for _ in range(4)]
        outgoing_t, incoming_t, out_top_t, in_top_t = buffers
        outgoing, incoming, out_top, in_top = [_view(tensor) for tensor in buffers]
        best, best_likelihood, runs = None, -math.inf, []
        for start in range(starts):
            generator = torch.Generator(device="cpu").manual_seed(restart_seed(seed, start))
            if start == 0 and user_initial is not None:
                membership, initialization = user_initial, "user"
            elif groups == 1:
                membership, initialization = [0] * n, "unique one-group partition"
            elif start == (1 if user_initial is not None else 0):
                membership = _structural_start(graph, groups, canonical, outgoing_csr, incoming_csr, work)
                initialization = "sparse binary structural-distance seeds"
            else:
                work.add(n)
                permutation = _view(torch.randperm(n, generator=generator, device="cpu"))
                membership = [0] * n
                for position, rank in enumerate(permutation):
                    membership[canonical[rank]] = position % groups
                initialization = "balanced random"
            state = _State(graph, membership, groups, work, saturated)
            history, moves, converged = [state.likelihood()], 0, groups == 1
            membership_changes = []
            iterations = 0
            if groups > 1:
                for iteration in range(max_iter):
                    work.add(n)
                    order = _view(torch.randperm(n, generator=generator, device="cpu"))
                    sweep_moves = 0
                    for rank in order:
                        node = canonical[rank]
                        old = state.membership[node]
                        if state.sizes[old] <= 1:
                            work.add(1)
                            continue
                        degree = len(outgoing_csr.neighbors(node))
                        if incoming_csr is not None:
                            degree += len(incoming_csr.neighbors(node))
                        work.add(4 * degree + 4 * groups + 1)
                        for tensor in buffers:
                            tensor.zero_()
                        for neighbor, weight in outgoing_csr.row(node):
                            outgoing[state.membership[neighbor]] += int(weight)
                            out_top[state.membership[neighbor]] += 1
                        if incoming_csr is not None:
                            for neighbor, weight in incoming_csr.row(node):
                                incoming[state.membership[neighbor]] += int(weight)
                                in_top[state.membership[neighbor]] += 1
                        selected, largest = old, 0.
                        for target in range(groups):
                            if target == old:
                                continue
                            gain, rounding = state.move(node, target, outgoing, incoming)
                            if gain > tol + rounding and gain > largest:
                                selected, largest = target, gain
                        if selected != old:
                            state.move(node, selected, outgoing, incoming, apply=True,
                                       outgoing_topology=out_top, incoming_topology=in_top)
                            sweep_moves += 1
                    iterations = iteration + 1
                    moves += sweep_moves
                    membership_changes.append(sweep_moves)
                    likelihood = state.likelihood()
                    if likelihood < history[-1] - 128 * sys.float_info.epsilon * max(1., abs(history[-1])):
                        _error("precision", "Poisson block ascent lost likelihood beyond rounding tolerance.")
                    history.append(likelihood)
                    if not sweep_moves:
                        converged = True
                        break
            likelihood = state.likelihood()
            runs.append(dict(start=start, initialization=initialization, iterations=iterations,
                             moves=moves, converged=converged, log_likelihood=likelihood,
                             likelihood_history=history,
                             **run_diagnostics(seed, start, groups, max_iter, history, membership_changes)))
            if likelihood > best_likelihood or (likelihood == best_likelihood and converged
                                                 and best is not None and not runs[best[1]]["converged"]):
                best_likelihood, best = likelihood, (state, start)
        state, selected_start = best
        work.add(2 * n + 8 * groups * groups)
        group_order, seen = [], set()
        for node in canonical:
            group = state.membership[node]
            if group not in seen:
                seen.add(group)
                group_order.append(group)
        remap = {old: new for new, old in enumerate(group_order)}
        membership_frame = graph._frame({"block": [remap[state.membership[node]] for node in range(n)]},
            kind="network_block_membership", fixed_groups=True, method="poisson_sbm", model="poisson")
        rows = []
        for source in range(groups):
            for target in range(0 if graph.directed else source, groups):
                old_source, old_target = group_order[source], group_order[target]
                count = state.counts[state.index(old_source, old_target)]
                possible = _potential(old_source, old_target, state.sizes, graph.directed)
                rows.append((source, target, state.sizes[old_source], state.sizes[old_target],
                             count, possible, count / possible if possible else math.nan, bool(possible)))
        blocks = as_frame(pd.DataFrame(rows, columns=["source_block", "target_block", "nodes_source",
            "nodes_target", "edges", "dyads", "mean_count", "identified"]))
        blocks.attrs.update(kind="network_block_means", model="poisson", directed=graph.directed,
                            self_loops=False, edges_semantics="aggregate count multiplicity")
        selected = runs[selected_start]
        metadata = dict(kind="network_block_model", model="poisson", algorithm="sparse hard-label profile ascent",
            nodes=n, groups=groups, fixed_groups=True, directed=graph.directed, dyads=dyads,
            self_loops=False, device="cpu", dtype="float64",
            topology_edges=topology_edges, total_edge_count_strength=total, self_loops_excluded=e - topology_edges,
            values="count", weight_semantics="off-diagonal positive exact integer counts; aggregate multiplicities",
            count_precision_scope="stored coalesced float64 snapshot; original input count provenance is not reconstructed",
            parallel_edges="coalesced counts are added", degree_corrected=False,
            log_likelihood=best_likelihood, converged=selected["converged"], iterations=selected["iterations"],
            stopping_reason=selected["stopping_reason"], selected_seed=selected["seed"],
            likelihood_scope="conditional independent Poisson dyad likelihood given hard memberships, including factorial constants",
            factorial_scope="lgamma for counts<32; stable saturated Stirling remainder through count^-7 otherwise (next term <2.4e-17)",
            starts=starts, selected_start=selected_start, start_fits=runs, seed=seed, max_iter=max_iter, tol=tol,
            initialization_scope="first non-user start uses sparse binary adjacency profiles; others balanced random",
            restart_rng="independent private CPU Torch generator; seed=(seed+start) modulo 2**63",
            parameter_change_scope="hard memberships changed per sweep; conditional block means reprofiled exactly",
            convergence_scope=("unique one-group partition; no admissible move" if groups == 1 else
                               "complete sweep with no admissible single-node gain exceeding tol plus rounding"),
            numerical_tolerance="64*float64_epsilon*max(1,sum(abs(old/new affected parameter-dependent block terms)))",
            global_optimum_certified=False, membership_uncertainty="not estimated",
            model_selection="fixed groups supplied by caller; no automatic selection or likelihood-ratio p-value",
            block_label_semantics="canonical smallest typed node label; labels interchangeable",
            zero_dyad_blocks="unidentified mean counts exported as NaN", sampled=False,
            memory_scope="estimated owned graph and O(V+E+K^2) workspace; excludes process RSS and caller input",
            estimated_workspace_bytes=workspace, work_used=work.used, max_work=work.limit,
            work_scope="charged node/edge and candidate block-update units; not CPU instruction count",
            torch_version=torch.__version__)
        return NetworkBlockResult(membership=membership_frame, blocks=blocks, metadata=metadata)
