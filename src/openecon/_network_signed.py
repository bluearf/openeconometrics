"""Bounded CPU signed adjacency analysis; no nonnegative solver is reused.

Signed modularity follows Gomez, Jensen and Arenas (2009), eqs. 18--20:
https://arxiv.org/abs/0812.3030 . Diagonal adjacency entries count once.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
import hashlib
import math
import sys

import torch

from openecon._network_communities import _membership
from openecon.networks import (Network, _Budget, _coalesce, _error, _integer, _key,
                              _label, _missing, _real, _rows, _tensor_bytes)


def signed_network(data, source="source", target="target", weight="weight", directed=False,
                   nodes=None, missing="raise", batch_rows=65536, max_memory_mb=256):
    """Build a resident signed graph; duplicate weights sum algebraically.

    Zero/cancelled pairs retain their endpoints. No magnitude conversion,
    truncation, disk-engine fallback or probabilistic interpretation is implied.
    """
    if not isinstance(directed, bool) or not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("invalid_option", "directed must be boolean; missing must be 'raise' or 'drop'.")
    columns = [source, target] + ([weight] if weight is not None else [])
    if (any(not isinstance(name, str) or not name.strip() or len(name) > 200 for name in columns)
            or len(set(columns)) != len(columns)):
        _error("invalid_columns", "Source, target and optional weight must be distinct valid column names.")
    requested_rows = _integer(batch_rows, "batch_rows", 1_000_000)
    budget = _Budget(max_memory_mb)
    rows = min(requested_rows, max(1, (budget.limit - 4096) // 2048))
    labels, label_index = [], {}
    label_bytes = 4096
    chunks = []
    input_rows = missing_rows = zero_rows = positive_rows = negative_rows = 0
    peak_rows = 0
    digest = hashlib.sha256()

    def retained():
        return sum(_tensor_bytes(item) for item in chunks if item is not None)

    def intern(raw):
        nonlocal label_bytes
        value = _label(raw)
        if value not in label_index:
            # Includes hash-table growth, index integers, references and UTF-8
            # label storage. Reserve before inserting a new node.
            addition = 512 + 2 * sys.getsizeof(value) + (len(value.encode("utf-8")) if isinstance(value, str) else 32)
            budget.check(label_bytes + addition + retained() + 1024 * rows)
            label_index[value] = len(labels)
            labels.append(value)
            label_bytes += addition
        return label_index[value]

    if nodes is not None:
        if not isinstance(nodes, Iterable) or isinstance(nodes, (str, bytes, Mapping)):
            _error("invalid_label", "nodes must be an iterable of string/integer isolate labels.")
        for label in nodes:
            intern(label)
    with torch.device("cpu"), torch.no_grad():
        for block, size in _rows(data, columns, rows):
            budget.check(label_bytes + retained() + 1024 * size)
            left, right, weights = [], [], []
            peak_rows = max(peak_rows, size)
            for record in block:
                record = tuple(record)
                input_rows += 1
                if any(_missing(value) for value in record):
                    if missing == "raise":
                        _error("missing_data", "Edge endpoints/weights contain missing values; choose missing='drop' explicitly.")
                    missing_rows += 1
                    continue
                i, j = intern(record[0]), intern(record[1])
                value = _real(record[2], "invalid_weight", "Signed weights must be finite real numbers.") if weight is not None else 1.0
                digest.update(repr((_key(labels[i]), _key(labels[j]), value)).encode("utf-8") + b"\n")
                if value == 0:
                    zero_rows += 1
                    continue
                positive_rows += value > 0
                negative_rows += value < 0
                left.append(i if directed else min(i, j))
                right.append(j if directed else max(i, j))
                weights.append(value)
            if not weights:
                continue
            budget.check(label_bytes + retained() + 1024 * size + 120 * len(weights))
            index = torch.tensor([left, right], dtype=torch.int64)
            values = torch.tensor(weights, dtype=torch.float64)
            level = 0
            chunk = _coalesce(index, values, len(labels), budget, label_bytes + retained() + 1024 * size)
            del index, values, left, right, weights
            while level < len(chunks) and chunks[level] is not None:
                previous = chunks[level]
                total = previous._nnz() + chunk._nnz()
                budget.check(label_bytes + retained() + _tensor_bytes(chunk) + 120 * total)
                index = torch.cat([previous.indices(), chunk.indices()], dim=1)
                values = torch.cat([previous.values(), chunk.values()])
                chunk = _coalesce(index, values, len(labels), budget,
                                  label_bytes + retained() + _tensor_bytes(chunk))
                chunks[level] = None
                del index, values, previous
                level += 1
            if level == len(chunks):
                chunks.append(chunk)
            else:
                chunks[level] = chunk
        pieces = [part for part in chunks if part is not None]
        total = sum(part._nnz() for part in pieces)
        budget.check(label_bytes + retained() + 120 * total + 4096)
        if pieces:
            index = torch.cat([part.indices() for part in pieces], dim=1)
            values = torch.cat([part.values() for part in pieces])
        else:
            index = torch.empty((2, 0), dtype=torch.int64)
            values = torch.empty(0, dtype=torch.float64)
        edges = _coalesce(index, values, len(labels), budget, label_bytes + retained())
        del index, values, pieces
        chunks.clear()
        unique = edges._nnz()
        nonzero = edges.values() != 0
        cancelled = int((~nonzero).sum())
        budget.check(label_bytes + 80 * unique + 4096)
        edges = torch.sparse_coo_tensor(edges.indices()[:, nonzero], edges.values()[nonzero],
            (len(labels), len(labels)), dtype=torch.float64, device="cpu", is_coalesced=True, check_invariants=True)
        metadata = {"input_rows": input_rows, "missing_rows_dropped": missing_rows,
                    "zero_weight_rows_dropped": zero_rows, "positive_edge_rows": positive_rows, "negative_edge_rows": negative_rows,
                    "cancelled_edge_pairs_dropped": cancelled,
                    "duplicate_edge_rows_aggregated": positive_rows + negative_rows - unique,
                    "requested_batch_rows": requested_rows, "effective_batch_rows": rows,
                    "actual_peak_batch_rows": peak_rows, "input_sha256": digest.hexdigest(),
                    "import_algorithm": "bounded COO coalescing with binary-level chunk merges",
                    "memory_scope": "estimated owned graph and planned buffers; excludes caller input and total process RSS"}
        result = Network(tuple(labels), label_index, edges, directed, weight is not None,
                         budget, label_bytes, metadata)
        return SignedNetwork(result)


def _cpu(function):
    @wraps(function)
    def call(*args, **kwargs):
        with torch.device("cpu"), torch.no_grad():
            return function(*args, **kwargs)
    return call


def _finite(value):
    if not bool(torch.isfinite(value).all()):
        _error("precision", "Signed arithmetic exceeds float64 range; no partial result is returned.")
    return value


def _positive(value, name):
    value = _real(value, "invalid_option", f"{name} must be positive and finite.")
    if value <= 0:
        _error("invalid_option", f"{name} must be positive and finite.")
    return value


class _Work:
    def __init__(self, maximum):
        self.maximum = _integer(maximum, "max_work", 2**63 - 1)
        self.used = 0

    def add(self, amount):
        if self.used + amount > self.maximum:
            _error("work_budget", "Signed analysis exceeds max_work; no incomplete result is returned.")
        self.used += amount


class SignedNetwork:
    """Signed adjacency snapshot with explicitly named signed methods.

    CPU float64 only. Diagonal entries count once in strengths and modularity;
    an undirected off-diagonal entry represents two oppositely oriented arcs.
    """

    def __init__(self, graph):
        self._graph = graph
        graph._metadata.update(signed=True, device="cpu", dtype="float64", sampled=False,
            weight_semantics="algebraic duplicate sum; positive attraction, negative repulsion",
            loop_semantics="adjacency diagonal counted once",
            storage_scope="resident sparse CPU graph; bounded import, not disk-native analysis")

    def weighted_assignment(self, partition=None, *, objective="weight", max_matrix_entries=1_000_000,
                            max_work=50_000_000):
        """Exact signed rewards; negative edges remain eligible for cardinality objectives."""
        from openecon._network_matching import weighted_assignment
        return weighted_assignment(self._graph, partition, objective=objective,
            max_matrix_entries=max_matrix_entries, max_work=max_work)

    def general_matching(self, *, objective="weight", max_component_nodes=24,
                         max_states=1_000_000, max_work=50_000_000):
        """Bounded general matching with original signed edge rewards."""
        from openecon._network_matching import general_matching
        return general_matching(self._graph, objective=objective, max_component_nodes=max_component_nodes,
            max_states=max_states, max_work=max_work)

    @property
    def node_count(self):
        return self._graph.node_count

    @property
    def edge_count(self):
        return self._graph.edge_count

    @property
    def directed(self):
        return self._graph.directed

    @property
    def metadata(self):
        return self._graph.metadata

    def __repr__(self):
        return f"SignedNetwork(nodes={self.node_count}, edges={self.edge_count}, directed={self.directed})"

    def __getattr__(self, name):
        if not name.startswith("_") and callable(getattr(Network, name, None)):
            def refuse(*args, **kwargs):
                _error("signed_method", f"{name} is not defined on SignedNetwork. Use an explicitly named "
                       "signed method; PageRank probabilities, capacities and count models cannot use signed weights.")
            return refuse
        raise AttributeError(name)

    def _frame(self, columns, **metadata):
        self._graph._guard(512 * self.node_count)
        return self._graph._frame(columns, **metadata)

    def _channels(self, *, normalized=False):
        graph = self._graph
        n, a = self.node_count, graph._arcs._nnz()
        graph._guard(512 * n + 128 * a)
        u, v = graph._arcs._indices()
        weights = graph._arcs._values()
        if normalized and a:
            scale = weights.abs().max()
            scaled = weights / scale
            if bool(((scaled == 0) & (weights != 0)).any()):
                _error("precision", "Signed weight normalization would erase a nonzero edge.")
            weights = scaled
        p, m = weights.clamp_min(0), (-weights).clamp_min(0)
        channels = []
        for indices, values in ((u, p), (v, p), (u, m), (v, m)):
            total = torch.zeros(n, dtype=torch.float64)
            total.index_add_(0, indices, values)
            channels.append(_finite(total))
        return u, v, p, m, channels

    @_cpu
    def signed_strength(self):
        """Positive, negative-magnitude, net and absolute adjacency strengths."""
        _, _, _, _, (po, pi, mo, mi) = self._channels()
        columns = {}
        for suffix, positive, negative in (("out", po, mo), ("in", pi, mi)):
            if not self.directed and suffix == "in":
                continue
            tail = f"_{suffix}_strength" if self.directed else "_strength"
            for name, values in (("positive", positive), ("negative", negative),
                                 ("net", positive - negative), ("absolute", positive + negative)):
                columns[name + tail] = _finite(values).numpy()
        return self._frame(columns, kind="signed_network_strength")

    @_cpu
    def signed_katz(self, alpha, beta=1., *, tol=1e-10, max_iter=1000, max_work=50_000_000):
        """Incoming signed resolvent x = beta*1 + alpha*A.T*x; not a probability."""
        alpha = _real(alpha, "invalid_option", "alpha must be nonnegative and finite.")
        if alpha < 0:
            _error("invalid_option", "alpha must be nonnegative and finite.")
        beta, tol = _positive(beta, "beta"), _positive(tol, "tol")
        maximum, work = _integer(max_iter, "max_iter", 1_000_000), _Work(max_work)
        u, v, p, m, (_, pi, _, mi) = self._channels()
        bound = float(_finite(pi + mi).max()) if self.node_count else 0.
        contraction = alpha * bound
        if not math.isfinite(contraction) or contraction >= 1:
            _error("signed_contraction", "signed_katz requires alpha * max absolute incoming strength < 1.")
        x = torch.full((self.node_count,), beta, dtype=torch.float64)
        weights = p - m
        if alpha and bool(((weights != 0) & (alpha * weights == 0)).any()):
            _error("precision", "Katz scaling would erase a nonzero signed edge.")

        def step(current):
            work.add(self.node_count + len(weights))
            result = torch.full_like(current, beta)
            result.index_add_(0, v, alpha * weights * current[u])
            return _finite(result)

        for iteration in range(1, maximum + 1):
            updated = step(x)
            # The returned iterate has its OWN fixed-point residual checked.
            residual = float(_finite(step(updated) - updated).abs().max()) if self.node_count else 0.
            x = updated
            scale = 1 + (float(x.abs().max()) if self.node_count else 0.)
            if residual <= tol * scale:
                error_bound = residual / (1 - contraction)
                if not math.isfinite(error_bound):
                    _error("precision", "The signed Katz error bound exceeds float64 range.")
                return self._frame({"signed_katz": x.numpy()}, kind="signed_network_katz",
                    alpha=alpha, beta=beta, tol=tol, iterations=iteration, converged=True,
                    residual_linf=residual, contraction_bound=contraction,
                    error_bound_linf=error_bound, work_used=work.used,
                    max_work=work.maximum, normalization="none; signed incoming resolvent")
        _error("not_converged", "signed_katz did not converge within max_iter; no partial centrality is returned.")

    def _model(self, work):
        work.add(self.node_count + self._graph._arcs._nnz())
        u, v, p, m, strengths = self._channels(normalized=True)
        positive, negative = float(p.sum()), float(m.sum())
        return u, v, p, m, strengths, positive, negative

    def _objective(self, membership, model, resolution, work):
        u, v, p, m, strengths, positive, negative = model
        n = self.node_count
        work.add(n + len(p))
        if positive + negative == 0:
            return 0.
        labels = torch.tensor(membership, dtype=torch.int64)
        intra = labels[u] == labels[v]
        score = float(p[intra].sum() - m[intra].sum())
        po, pi, mo, mi = strengths
        for outgoing, incoming, mass, sign in ((po, pi, positive, -1), (mo, mi, negative, 1)):
            if mass:
                left, right = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
                left.index_add_(0, labels, outgoing)
                right.index_add_(0, labels, incoming)
                score += sign * resolution * float((left * (right / mass)).sum())
        result = score / (positive + negative)
        if not math.isfinite(result):
            _error("precision", "Signed modularity exceeds float64 range.")
        return result

    @_cpu
    def signed_modularity(self, membership, resolution=1., *, max_work=50_000_000):
        """Gomez signed modularity, directed out/in null models, zero channels omitted."""
        resolution, work = _positive(resolution, "resolution"), _Work(max_work)
        work.add(self.node_count)
        labels = _membership(self._graph, membership)
        return self._objective(labels, self._model(work), resolution, work)

    @_cpu
    def signed_communities(self, resolution=1., *, tol=1e-12, max_sweeps=100,
                           max_work=50_000_000):
        """Deterministic all-community greedy moves; a single-node stationary point.

        Every existing community and one empty singleton are candidates. This
        optimizes signed modularity, not unsigned Louvain or global optimality.
        """
        resolution, tol = _positive(resolution, "resolution"), _positive(tol, "tol")
        maximum, work = _integer(max_sweeps, "max_sweeps", 1_000_000), _Work(max_work)
        graph, n = self._graph, self.node_count
        graph._guard(1536 * n + 512 * graph._arcs._nnz() + 32 * maximum)
        model = self._model(work)
        u, v, p, m, strengths, positive, negative = model
        membership = list(range(n))
        order = sorted(range(n), key=lambda i: _key(graph._labels[i]))
        neighbors = [[] for _ in range(n)]
        for i, j, pos, neg in zip(u.tolist(), v.tolist(), p.tolist(), m.tolist()):
            if i != j:
                neighbors[i].append((j, pos, neg))
                neighbors[j].append((i, pos, neg))
        totals = [values.tolist() for values in strengths]
        individual = [values[:] for values in totals]
        counts = [1] * n
        trace = [self._objective(membership, model, resolution, work)]
        for sweep in range(1, maximum + 1):
            moves = 0
            if positive + negative:
                for i in order:
                    old = membership[i]
                    incident = {}
                    work.add(len(neighbors[i]) + n)
                    for j, pos, neg in neighbors[i]:
                        values = incident.setdefault(membership[j], [0., 0.])
                        values[0] += pos
                        values[1] += neg
                    empty = next((c for c, count in enumerate(counts) if count == 0), None)
                    old_p, old_m = incident.get(old, (0., 0.))
                    best, best_gain = old, tol
                    for candidate, count in enumerate(counts):
                        if candidate == old or (not count and candidate != empty):
                            continue
                        work.add(1)
                        new_p, new_m = incident.get(candidate, (0., 0.))
                        gain = new_p - old_p - new_m + old_m
                        for left, right, mass, sign in ((0, 1, positive, -1), (2, 3, negative, 1)):
                            if mass:
                                oi, ii = individual[left][i], individual[right][i]
                                delta = ((totals[left][candidate] - totals[left][old]) * (ii / mass)
                                    + (totals[right][candidate] - totals[right][old]) * (oi / mass)
                                    + 2 * oi * (ii / mass))
                                gain += sign * resolution * delta
                        gain /= positive + negative
                        if not math.isfinite(gain):
                            _error("precision", "Signed community move exceeds float64 range.")
                        if gain > best_gain:
                            best, best_gain = candidate, gain
                    if best != old:
                        for channel in range(4):
                            totals[channel][old] -= individual[channel][i]
                            totals[channel][best] += individual[channel][i]
                        counts[old] -= 1
                        counts[best] += 1
                        membership[i] = best
                        moves += 1
            score = self._objective(membership, model, resolution, work)
            if score + tol < trace[-1]:
                _error("precision", "Signed objective decreased unexpectedly; no partition is returned.")
            trace.append(score)
            if not moves:
                # Stable IDs are assigned in typed-label order, independent of input order.
                identifiers = {}
                for i in order:
                    identifiers.setdefault(membership[i], len(identifiers))
                return self._frame({"community": [identifiers[c] for c in membership]},
                    kind="signed_network_communities", algorithm="all-community signed greedy moves",
                    optimality="single-node stationary point within tol; not global or multilevel",
                    resolution=resolution, tol=tol, sweeps=sweep, converged=True, signed_modularity=score,
                    objective_trace=trace, work_used=work.used, max_work=work.maximum)
        _error("not_converged", "Signed communities did not stabilize within max_sweeps; no partial partition is returned.")

    @_cpu
    def signed_shortest_paths(self, source, *, max_work=50_000_000):
        """Bellman--Ford minimum walk costs; reachable negative cycles are errors."""
        source = _label(source)
        graph, n, work = self._graph, self.node_count, _Work(max_work)
        if source not in graph._index:
            _error("invalid_source", "The source must be a graph node with the same typed identity.")
        u, v = graph._arcs._indices()
        weights = graph._arcs._values()
        graph._guard(512 * n + 128 * len(weights))
        distances = torch.full((n,), float("inf"), dtype=torch.float64)
        distances[graph._index[source]] = 0.
        iterations = 0
        for iteration in range(1, n + 1):
            work.add(n + len(weights))
            reachable = torch.isfinite(distances[u])
            previous = distances[u[reachable]]
            costs = weights[reachable]
            candidate = _finite(previous + costs)
            if bool(((costs != 0) & (candidate == previous)).any()):
                _error("precision", "Float64 path addition would erase a nonzero cost; negative-cycle checks cannot be certified.")
            updated = distances.clone()
            updated.scatter_reduce_(0, v[reachable], candidate, reduce="amin", include_self=True)
            changed = bool((updated < distances).any())
            iterations = iteration
            if not changed:
                break
            if iteration == n:
                _error("negative_cycle", "A negative cycle is reachable from the source. Minimum walk costs are undefined; "
                       "an undirected negative edge also forms a two-arc negative cycle.")
            distances = updated
        reachable = torch.isfinite(distances)
        output = distances.clone()
        output[~reachable] = float("nan")
        return self._frame({"distance": output.numpy(), "reachable": reachable.numpy()},
            kind="signed_network_shortest_paths", algorithm="synchronous sparse Bellman-Ford",
            source=source, iterations=iterations, reachable_negative_cycle=False,
            path_semantics="minimum walk cost; unreachable nodes have null distance",
            work_used=work.used, max_work=work.maximum)
