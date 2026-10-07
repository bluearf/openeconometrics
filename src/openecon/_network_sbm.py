"""Sparse fixed-order hard-label Bernoulli stochastic block models.

For a given partition, each eligible off-diagonal dyad is an independent
Bernoulli observation and p_rs = edges_rs / dyads_rs maximizes its block
likelihood. Hard-label coordinate ascent optimizes that profile likelihood,
not modularity. This is an ordinary, not degree-corrected, SBM. Reference:
Choi, Wolfe and Airoldi (2012), section 2, https://arxiv.org/abs/1011.4644 .

Only O(V + E + K^2) storage is used; absent dyads are counted algebraically.
Each sweep uses O(E + V K^2) work. Counts and CSR buffers are CPU Torch,
with zero-copy scalar traversal. No dense V-by-V array or external solver is
used. A completed no-move sweep certifies a single-node local optimum within
the stated tolerance, never a global optimum or a selected number of groups.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral
import sys

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key, _label, _real
from openecon._network_sparse import csr


def _view(tensor):
    return memoryview(tensor.numpy())


class _Work:
    def __init__(self, limit):
        self.limit = _integer(limit, "max_work", 2**63 - 1)
        self.used = 0

    def add(self, count):
        if self.used + count > self.limit:
            _error("work_budget", "Block modelling exceeds max_work; increase the explicit "
                   "budget or request fewer starts/iterations/groups. No partial fit is returned.")
        self.used += count


class NetworkBlockResult(dict):
    """Fixed-K SBM mapping; display and LaTeX use a bounded fit summary.

    Membership and model-specific block parameters are explicit mapping
    entries. A Bernoulli zero-dyad block has probability NaN, not zero.
    Block labels are canonical identifiers, not ordered economic categories.
    """

    def summary(self):
        """Return eleven fit metrics without serializing every node or block."""
        meta = self["metadata"]
        model = meta["model"]
        title = {"bernoulli": "Bernoulli SBM", "poisson": "Poisson SBM",
                 "degree_corrected_poisson": "Degree-corrected Poisson SBM"}[model]
        frame = as_frame(pd.DataFrame([
            ("Model", title), ("Nodes", meta["nodes"]),
            ("Groups (fixed)", meta["groups"]), ("Eligible dyads", meta["dyads"]),
            ("Positive dyads (including loops)" if meta.get("self_loops") else
             "Positive off-diagonal edges", meta["topology_edges"]),
            ("Profile log likelihood", meta["log_likelihood"]),
            ("Converged", "Yes" if meta["converged"] else "No"),
            ("Selected start", meta["selected_start"]),
            ("Starts", meta["starts"]), ("Selected-start sweeps", meta["iterations"]),
            ("Optimality", "Single-node local (tolerance)" if meta["converged"] else "Uncertified")
        ], columns=["Metric", "Value"]))
        frame.attrs.update(kind="network_block_model_summary", model=model,
                           fixed_groups=True, global_optimum_certified=False)
        return frame

    def expected_edges(self, pairs, *, max_pairs=100_000, max_memory_mb=256):
        """Predict expected counts and presence probabilities for explicit node pairs.

        Input pair order and duplicates are preserved. Poisson presence is
        1-exp(-mean_count). Only the degree-corrected model includes loops;
        its undirected raw loop mean is theta_i squared times omega_rr / 2.
        Identification flags expose unidentified zero-stub conventions.
        No dense all-pairs table or parameter-uncertainty interval is created.
        """
        from openecon._network_block_prediction import expected_edges
        return expected_edges(self, pairs, max_pairs=max_pairs, max_memory_mb=max_memory_mb)

    def to_latex(self, buf=None, **kwargs):
        """Export the bounded fit summary, forwarding Frame table options."""
        kwargs.setdefault("index", False)
        return self.summary().to_latex(buf=buf, **kwargs)

    def __repr__(self):
        meta = self["metadata"]
        return (f"NetworkBlockResult(groups={meta['groups']}, "
                f"log_likelihood={meta['log_likelihood']!r}, converged={meta['converged']})")


def _potential(a, b, sizes, directed):
    if a == b:
        value = sizes[a] * (sizes[a] - 1)
        return value if directed else value // 2
    return sizes[a] * sizes[b]


def _term(edges, dyads):
    """Bernoulli profile log likelihood, stable at 0/1 and rare nonedges."""
    if not 0 <= edges <= dyads:
        _error("precision", "Block edge/dyad counts became inconsistent.")
    if edges == 0 or edges == dyads:
        return 0.
    missing = dyads - edges
    if edges <= missing:
        return edges * (math.log(edges) - math.log(dyads)) + missing * math.log1p(-edges / dyads)
    return missing * (math.log(missing) - math.log(dyads)) + edges * math.log1p(-missing / dyads)


def _initial(graph, initial, groups):
    """Validate a full user partition before encoding it in bounded buffers."""
    n = graph.node_count
    if isinstance(initial, pd.DataFrame):
        if (list(initial.columns).count("node") != 1 or list(initial.columns).count("block") != 1
                or len(initial) != n):
            _error("invalid_membership", "Initial membership needs exactly one node/block row per node.")
        mapping = {}
        for label, block in initial.loc[:, ["node", "block"]].itertuples(index=False, name=None):
            label = _label(label)
            if label not in graph._index or label in mapping:
                _error("invalid_membership", "Initial membership contains an unknown or duplicate node.")
            mapping[label] = block
        initial = mapping
    if isinstance(initial, Mapping):
        if len(initial) != n or any(_label(label) not in graph._index for label in initial):
            _error("invalid_membership", "Initial membership must cover every exact node ID once.")
        initial = [initial[label] for label in graph._labels]
    elif isinstance(initial, torch.Tensor):
        if (initial.ndim != 1 or initial.layout != torch.strided or initial.device.type == "meta"
                or initial.dtype == torch.bool or initial.dtype.is_floating_point or initial.dtype.is_complex
                or initial.numel() != n):
            _error("invalid_membership", "Initial tensor must be a materialized length-V integer vector.")
        initial = initial.cpu().tolist()
    elif (isinstance(initial, (str, bytes)) or not isinstance(initial, (Sequence, pd.Series))
          or len(initial) != n):
        _error("invalid_membership", "Initial membership needs a full mapping, node/block table or sized vector.")
    result = []
    for value in initial:
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < groups:
            _error("invalid_membership", "Initial blocks must be integer identifiers from 0 to groups-1.")
        result.append(int(value))
    if len(set(result)) != groups:
        _error("invalid_membership", "Every requested group must be nonempty in initial membership.")
    return result


def _structural_start(graph, groups, canonical, outgoing, incoming, work):
    """Farthest-first adjacency-profile seeds in O(K(V+E)), not dense distance.

    Binary row Hamming distance uses degrees and shared-neighbor counts.
    Incoming CSR propagates each seed's neighbor indicator to its predecessors;
    a seed touches at most E arcs. Directed profiles include both row and column
    distance. This is only an initialization for hard-label block models and can
    represent disassortative as well as assortative block structure.
    """
    n = graph.node_count
    reverse = incoming if incoming is not None else outgoing
    work.add(n + groups * (4 * n + 2 * graph._arcs._nnz() + groups))
    degree_t = torch.empty(n, dtype=torch.int64, device="cpu")
    degree = _view(degree_t)
    for node in range(n):
        degree[node] = len(outgoing.neighbors(node))
        if incoming is not None:
            degree[node] += len(incoming.neighbors(node))
    best_t = torch.full((n,), 2 * n + 1, dtype=torch.int64, device="cpu")
    block_t = torch.zeros(n, dtype=torch.int64, device="cpu")
    overlap_t = torch.zeros(n, dtype=torch.int64, device="cpu")
    best, block, overlap = _view(best_t), _view(block_t), _view(overlap_t)
    seeds, selected = [], set()
    first = max(canonical, key=lambda node: degree[node])
    for group in range(groups):
        seed = first if group == 0 else max((node for node in canonical if node not in selected),
                                             key=lambda node: best[node])
        seeds.append(seed)
        selected.add(seed)
        overlap_t.zero_()
        for neighbor in outgoing.neighbors(seed):
            for other in reverse.neighbors(neighbor):
                overlap[other] += 1
        if incoming is not None:
            for neighbor in incoming.neighbors(seed):
                for other in outgoing.neighbors(neighbor):
                    overlap[other] += 1
        for node in canonical:
            distance = degree[node] + degree[seed] - 2 * overlap[node]
            if distance < best[node]:
                best[node], block[node] = distance, group
    # Identical rows may produce duplicate seeds; explicitly keep every chosen
    # representative in its own block rather than silently collapsing K.
    for group, seed in enumerate(seeds):
        block[seed] = group
    return block_t.tolist()


class _State:
    """Exact sufficient counts and stable profile contributions for one start."""

    def __init__(self, graph, membership, groups, work):
        n = graph.node_count
        self.graph, self.groups, self.work = graph, groups, work
        self.membership_t = torch.tensor(membership, dtype=torch.int64, device="cpu")
        self.sizes_t = torch.bincount(self.membership_t, minlength=groups)
        self.counts_t = torch.zeros(groups * groups, dtype=torch.int64, device="cpu")
        self.terms_t = torch.zeros(groups * groups, dtype=torch.float64, device="cpu")
        self.membership, self.sizes = _view(self.membership_t), _view(self.sizes_t)
        self.counts, self.terms = _view(self.counts_t), _view(self.terms_t)
        work.add(n + graph.edge_count + groups * groups)
        pairs = graph._edges.indices()
        source, target = _view(pairs[0]), _view(pairs[1])
        for u, v in zip(source, target):
            if u == v:
                continue
            a, b = self.membership[u], self.membership[v]
            self.counts[self.index(a, b)] += 1
        for a, b in self.cells():
            at = self.index(a, b)
            self.terms[at] = _term(self.counts[at], _potential(a, b, self.sizes, graph.directed))

    def index(self, a, b):
        if not self.graph.directed and a > b:
            a, b = b, a
        return a * self.groups + b

    def cells(self):
        for a in range(self.groups):
            for b in range(0 if self.graph.directed else a, self.groups):
                yield a, b

    def likelihood(self):
        return math.fsum(self.terms[self.index(a, b)] for a, b in self.cells())

    def move(self, node, target, outgoing, incoming, *, apply=False):
        """Evaluate only blocks incident on either changed group, then apply."""
        old = self.membership[node]
        groups, directed = self.groups, self.graph.directed
        self.work.add(16 * groups + 8)
        deltas = {}
        def add(a, b, delta):
            at = self.index(a, b)
            deltas[at] = deltas.get(at, 0) + delta
        for other in range(groups):
            add(old, other, -outgoing[other])
            add(target, other, outgoing[other])
            if directed:
                add(other, old, -incoming[other])
                add(other, target, incoming[other])
        # Zero deltas are retained: changing group sizes changes absent dyads,
        # including blocks with no incident edge at all.
        sizes = list(self.sizes)
        sizes[old] -= 1
        sizes[target] += 1
        changes = []
        for at in sorted(deltas):
            a, b = divmod(at, groups)
            count = self.counts[at] + deltas[at]
            value = _term(count, _potential(a, b, sizes, directed))
            changes.append((at, count, value))
        gain = math.fsum(value - self.terms[at] for at, _, value in changes)
        rounding = 64 * sys.float_info.epsilon * max(1.,
            math.fsum(abs(self.terms[at]) + abs(value) for at, _, value in changes))
        if apply:
            for at, count, value in changes:
                self.counts[at], self.terms[at] = count, value
            self.sizes[old] -= 1
            self.sizes[target] += 1
            self.membership[node] = target
        return gain, rounding


def block_model(graph, groups, *, values="binary", initial=None, seed=0, starts=4,
                max_iter=100, tol=1e-9, max_work=50_000_000):
    """Fit a fixed-K hard-label Bernoulli SBM by sparse profile ascent.

    ``binary`` explicitly uses positive-edge presence, ignoring stored strength,
    duplicate multiplicity and self-loops. Absent eligible dyads count as zero.
    All groups remain nonempty. Initial vectors follow snapshot node order;
    mappings or node/block tables use exact node IDs. Block IDs are canonicalized
    by the smallest typed node label in each group at export.

    The first non-user start uses sparse adjacency-profile Hamming seeds; other
    starts are balanced random. Random starts and sweep order use a private CPU
    Torch RNG. A start
    ends after max_iter sweeps or a complete sweep with no admissible improvement
    exceeding tol plus its documented rounding tolerance. Truncated starts are
    returned with converged=False, never labeled locally/globally optimal. K=1
    has no admissible move. max_iter=0 requires initial and starts=1.
    """
    groups = _integer(groups, "groups", 128)
    starts = _integer(starts, "starts", 64)
    max_iter = _integer(max_iter, "max_iter", 10_000, zero=True)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    tol = _real(tol, "invalid_option", "tol must be finite and nonnegative.")
    if tol < 0 or not isinstance(values, str) or values != "binary":
        _error("invalid_option", "Block modelling requires values='binary' and nonnegative tol.")
    n, e, a = graph.node_count, graph.edge_count, graph._arcs._nnz()
    if groups > n:
        _error("invalid_option", "groups cannot exceed the number of nodes; every group must be nonempty.")
    if not max_iter and (initial is None or starts != 1):
        _error("invalid_option", "max_iter=0 requires initial membership and starts=1.")
    dyads = n * (n - 1) if graph.directed else n * (n - 1) // 2
    if dyads > 2**53:
        _error("precision", "Eligible dyad count exceeds exact float64 integer range.")
    work = _Work(max_work)
    # Admit all requested start setup and result serialization before the first
    # CSR/buffer allocation. Iterative work is admitted before each node/move.
    work.add(n * (3 + n.bit_length()) + 4 * a + 4 * e + 32)
    minimum = starts * (n + e + groups * groups) + 2 * n + 8 * groups * groups
    if groups > 1 and (initial is None or starts > 1):
        minimum += n + groups * (4 * n + 2 * a + groups)
    if groups > 1:
        structured = int(initial is None or starts > 1)
        minimum += (starts - int(initial is not None) - structured) * n
    if work.used + minimum > work.limit:
        _error("work_budget", "Requested block-model start setup exceeds max_work; no fit is returned.")
    needs_moves = groups > 1 and max_iter > 0
    workspace = (512 * n + (192 * a if needs_moves else 0) + 512 * groups * groups + 512 * groups
                 + starts * (512 + 64 * (max_iter + 1)) + 32768)
    graph._guard(workspace)
    with torch.device("cpu"), torch.no_grad():
        user_initial = _initial(graph, initial, groups) if initial is not None else None
        canonical = sorted(range(n), key=lambda node: _key(graph._labels[node]))
        outgoing_csr = csr(graph, loops=False) if needs_moves else None
        incoming_csr = csr(graph, reverse=True, loops=False) if needs_moves and graph.directed else None
        generator = torch.Generator(device="cpu").manual_seed(seed)
        outgoing_t = torch.zeros(groups, dtype=torch.int64, device="cpu")
        incoming_t = torch.zeros(groups, dtype=torch.int64, device="cpu")
        outgoing, incoming = _view(outgoing_t), _view(incoming_t)
        best, best_likelihood, runs = None, -math.inf, []
        for start in range(starts):
            if start == 0 and user_initial is not None:
                membership = user_initial
                initialization = "user"
            elif groups == 1:
                membership = [0] * n
                initialization = "unique one-group partition"
            elif start == (1 if user_initial is not None else 0):
                membership = _structural_start(graph, groups, canonical, outgoing_csr, incoming_csr, work)
                initialization = "sparse structural-distance seeds"
            else:
                work.add(n)
                permutation = _view(torch.randperm(n, generator=generator, device="cpu"))
                membership = [0] * n
                for position, rank in enumerate(permutation):
                    membership[canonical[rank]] = position % groups
                initialization = "balanced random"
            state = _State(graph, membership, groups, work)
            history, moves, converged = [state.likelihood()], 0, groups == 1
            iterations = 0
            if groups > 1:
                for iteration in range(max_iter):
                    work.add(n)
                    order_t = torch.randperm(n, generator=generator, device="cpu")
                    order, sweep_moves = _view(order_t), 0
                    for rank in order:
                        node = canonical[rank]
                        old = state.membership[node]
                        if state.sizes[old] <= 1:
                            work.add(1)
                            continue
                        degree = len(outgoing_csr.neighbors(node))
                        if incoming_csr is not None:
                            degree += len(incoming_csr.neighbors(node))
                        work.add(degree + 2 * groups + 1)
                        outgoing_t.zero_()
                        incoming_t.zero_()
                        for neighbor in outgoing_csr.neighbors(node):
                            outgoing[state.membership[neighbor]] += 1
                        if incoming_csr is not None:
                            for neighbor in incoming_csr.neighbors(node):
                                incoming[state.membership[neighbor]] += 1
                        selected, largest = old, 0.
                        for target in range(groups):
                            if target == old:
                                continue
                            gain, rounding = state.move(node, target, outgoing, incoming)
                            if gain > tol + rounding and gain > largest:
                                selected, largest = target, gain
                        if selected != old:
                            state.move(node, selected, outgoing, incoming, apply=True)
                            sweep_moves += 1
                    iterations = iteration + 1
                    moves += sweep_moves
                    work.add(groups * groups)
                    likelihood = state.likelihood()
                    if likelihood < history[-1] - 128 * sys.float_info.epsilon * max(1., abs(history[-1])):
                        _error("precision", "Block-model ascent lost likelihood beyond rounding tolerance.")
                    history.append(likelihood)
                    if not sweep_moves:
                        converged = True
                        break
            likelihood = state.likelihood()
            runs.append(dict(start=start, initialization=initialization, iterations=iterations,
                             moves=moves, converged=converged, log_likelihood=likelihood,
                             likelihood_history=history))
            # Prefer a certified no-move result when a tied truncated start has
            # exactly the same objective; otherwise select highest fitted LL.
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
                                        kind="network_block_membership", fixed_groups=True,
                                        method="bernoulli_sbm", model="bernoulli")
        rows = []
        for source in range(groups):
            for target in range(0 if graph.directed else source, groups):
                old_source, old_target = group_order[source], group_order[target]
                count = state.counts[state.index(old_source, old_target)]
                possible = _potential(old_source, old_target, state.sizes, graph.directed)
                rows.append((source, target, state.sizes[old_source], state.sizes[old_target],
                             count, possible, count / possible if possible else math.nan, bool(possible)))
        blocks = as_frame(pd.DataFrame(rows, columns=["source_block", "target_block", "nodes_source",
            "nodes_target", "edges", "dyads", "probability", "identified"]))
        blocks.attrs.update(kind="network_block_probabilities", model="bernoulli",
                            directed=graph.directed, self_loops=False)
        topology_edges = int(sum(state.counts[state.index(x, y)] for x, y in state.cells()))
        selected = runs[selected_start]
        metadata = dict(kind="network_block_model", model="bernoulli", algorithm="sparse hard-label profile ascent",
            nodes=n, groups=groups, fixed_groups=True, directed=graph.directed, dyads=dyads,
            topology_edges=topology_edges, self_loops_excluded=e - topology_edges,
            values="binary", weight_semantics="unique positive-edge presence; stored strengths ignored",
            parallel_edges="coalesced input edges count once", degree_corrected=False,
            log_likelihood=best_likelihood, converged=selected["converged"], iterations=selected["iterations"],
            likelihood_scope="conditional Bernoulli graph likelihood given hard memberships; no class-proportion term",
            starts=starts, selected_start=selected_start, start_fits=runs, seed=seed, max_iter=max_iter, tol=tol,
            initialization_scope="first non-user start uses O(K(V+E)) adjacency-profile Hamming seeds; others balanced random",
            convergence_scope=("unique one-group partition; no admissible move" if groups == 1 else
                               "completed sweep with no admissible single-node gain exceeding tol plus rounding"),
            numerical_tolerance="64*float64_epsilon*max(1,sum(abs(old/new affected block contributions)))",
            global_optimum_certified=False, membership_uncertainty="not estimated",
            model_selection="fixed groups supplied by caller; no automatic selection or likelihood-ratio p-value",
            block_label_semantics="canonical smallest typed node label; labels interchangeable",
            zero_dyad_blocks="unidentified probabilities exported as NaN", sampled=False,
            memory_scope="estimated owned graph and O(V+E+K^2) workspace; excludes process RSS and caller input",
            estimated_workspace_bytes=workspace, work_used=work.used, max_work=work.limit,
            work_scope="charged node/edge and candidate block-update units; not CPU instruction count",
            torch_version=torch.__version__)
        return NetworkBlockResult(membership=membership_frame, blocks=blocks, metadata=metadata)
