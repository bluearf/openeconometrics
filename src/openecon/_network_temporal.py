"""Bounded sparse snapshot collections, turnover and strictly temporal paths.

Layer keys have no inferred chronological meaning. Mapping insertion order is
captured; causal traversal additionally requires ``ordered=True``. A temporal
edge consumes one snapshot, and a node can wait only while continuously present.
This is a snapshot foundation, not a supra-adjacency, SAOM or dynamic ERGM model.

Holme and Saramaki (2012), Temporal Networks:
https://arxiv.org/abs/1108.1780
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import math
import sys

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import Network, _Budget, _error, _integer, _key, _label


class _Work:
    def __init__(self, maximum):
        self.maximum = _integer(maximum, "max_work", 2**63 - 1)
        self.used = 0

    def add(self, count):
        self.used += count
        if self.used > self.maximum:
            _error("work_budget", "Snapshot analysis exceeds max_work; increase the explicit "
                   "structural/word-operation budget. No partial result is returned.")


def _boolean(value, name):
    if not isinstance(value, bool):
        _error("invalid_option", f"{name} must be boolean.")
    return value


def _parts(value):
    numerator, denominator = value.as_integer_ratio()
    trailing = (numerator & -numerator).bit_length() - 1
    return numerator >> trailing, trailing - denominator.bit_length() + 1


def _export(units, exponent, divisor):
    # Divide the exact sum BEFORE rounding. A sum outside float64 may still
    # have a finite mean; all arithmetic up to scalar export is exact.
    exact = (Fraction(units << exponent, divisor) if exponent >= 0 else
             Fraction(units, divisor << -exponent))
    try:
        value = float(exact)
    except OverflowError:
        _error("precision", "An aggregate edge strength exceeds binary64 output range.")
    if not math.isfinite(value) or value <= 0:
        _error("precision", "An aggregate positive strength would overflow or round to zero.")
    return value


def _table(columns):
    # Typed identities remain objects: integer 1 and string '1' are distinct.
    identities = {"layer", "from_layer", "to_layer", "source", "target",
                  "first_layer", "last_layer"}
    return as_frame(pd.DataFrame({name: pd.Series(values, dtype=object)
                                 if name in identities else pd.Series(values)
                                 for name, values in columns.items()}))


def _interval_overlap(first, second):
    if first is second:
        return sum(end - start + 1 for start, end in first)
    left = right = total = 0
    while left < len(first) and right < len(second):
        a, b = first[left], second[right]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
        if a[1] <= b[1]:
            left += 1
        else:
            right += 1
    return total


def network_snapshots(layers, *, ordered=False, max_work=50_000_000):
    """Capture a nonempty ordered mapping of exact layer IDs to sparse Networks.

    String/integer layer keys retain their exact type and insertion order.
    ``ordered=True`` explicitly asserts that this order represents time; keys
    are never sorted, parsed as timestamps, or assigned elapsed-time durations.
    All snapshots must have the same directedness. Node sets can vary, including
    isolates. Resident snapshots are retained by reference, without edge copies.
    """
    return NetworkSnapshots(layers, ordered=ordered, max_work=max_work)


class NetworkSnapshots:
    """An immutable captured sequence of sparse layers with bounded analyses.

    The source mapping is copied into tuples, so later mapping edits have no
    effect. Network snapshots themselves retain their immutable snapshot API.
    Union labels and a small alignment index are retained, never a T-by-N-by-N
    array. All resident graphs count against every source memory allowance.
    """

    def __init__(self, layers, *, ordered=False, max_work=50_000_000):
        ordered = _boolean(ordered, "ordered")
        work = _Work(max_work)
        if not isinstance(layers, Mapping) or not layers:
            _error("invalid_option", "layers must be a nonempty mapping of string/integer IDs to Networks.")
        work.add(4 * len(layers))
        keys, graphs, unique, key_bytes = [], [], {}, 4096
        resident, limiting = 0, None
        for raw_key, graph in layers.items():
            key = _label(raw_key)
            if not isinstance(graph, Network):
                _error("invalid_option", "Every layer must contain a Network snapshot.")
            if graphs and graph.directed != graphs[0].directed:
                _error("invalid_option", "All snapshots must have the same directedness.")
            prospective = resident + (0 if id(graph) in unique else graph._base_bytes)
            added_key_bytes = 256 + 2 * sys.getsizeof(key)
            if limiting is None or graph._budget.limit < limiting._budget.limit:
                limiting = graph
            # The smallest seen allowance enforces ALL seen graph budgets.
            # Reserve the complete mapping-reference capacity before growing
            # capture lists, rather than first allocating an unbounded list.
            limiting._budget.check(prospective + key_bytes + added_key_bytes
                                   + 512 * len(layers) + 8192)
            keys.append(key)
            graphs.append(graph)
            unique[id(graph)] = graph
            key_bytes += added_key_bytes
            resident = prospective
        if len(set(keys)) != len(keys):
            _error("invalid_label", "Layer IDs must be unique after exact integer-label normalization.")
        unique_graphs = tuple(unique.values())
        resident = sum(graph._base_bytes for graph in unique_graphs)
        node_upper = sum(graph.node_count for graph in unique_graphs)
        work.add(node_upper)
        # Reserve union set, canonical sort, alignment map and captured keys
        # before collecting labels. Shared snapshot objects count only once.
        peak = resident + 1024 * node_upper + key_bytes + 512 * len(keys) + 8192
        for graph in unique_graphs:
            graph._budget.check(peak)
        labels = set()
        for graph in unique_graphs:
            labels.update(graph._labels)
        work.add(len(labels) * max(1, len(labels).bit_length()))
        self._labels = tuple(sorted(labels, key=_key))
        self._index = {label: index for index, label in enumerate(self._labels)}
        self._keys, self._graphs = tuple(keys), tuple(graphs)
        self._layer_index = {key: index for index, key in enumerate(keys)}
        self._unique_graphs, self._resident_bytes = unique_graphs, resident
        self._alignment_bytes = 512 * len(labels) + key_bytes + 512 * len(keys)
        self._ordered, self._directed = ordered, graphs[0].directed
        self._node_observations = sum(graph.node_count for graph in graphs)
        self._edge_observations = sum(graph.edge_count for graph in graphs)
        self._metadata = dict(snapshot_count=len(keys), node_count=len(labels),
            stored_edge_observations=self._edge_observations, directed=self.directed,
            ordered=self.ordered, order="captured mapping insertion order; no inferred timestamps",
            distinct_resident_snapshots=len(unique_graphs), estimated_resident_graph_bytes=resident,
            estimated_alignment_bytes=self._alignment_bytes, device="cpu", dtype="float64",
            max_work=work.maximum, work_used=work.used,
            memory_scope="all unique resident snapshots plus sparse alignment and operation/output buffers; "
                         "under every source memory budget; excludes caller inputs and whole-process RSS",
            scope="sparse snapshots, turnover, persistence, aggregate graph and strict temporal paths")
        self._guard(4096, work)
        self._metadata["work_used"] = work.used

    @property
    def layer_keys(self):
        """Return captured exact IDs in their explicit insertion order."""
        return self._keys

    @property
    def snapshot_count(self):
        return len(self._keys)

    @property
    def node_count(self):
        return len(self._labels)

    @property
    def directed(self):
        return self._directed

    @property
    def ordered(self):
        return self._ordered

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def __repr__(self):
        return (f"NetworkSnapshots(snapshots={self.snapshot_count}, nodes={self.node_count}, "
                f"directed={self.directed}, ordered={self.ordered})")

    def _guard(self, workspace, work):
        work.add(len(self._unique_graphs))
        total = self._resident_bytes + self._alignment_bytes + int(workspace) + 4096
        for graph in self._unique_graphs:
            graph._budget.check(total)

    def _edges(self, graph, include_loops):
        pairs, values = graph._edges.indices(), graph._edges.values()
        left, right, weights = (memoryview(tensor.numpy()) for tensor in (pairs[0], pairs[1], values))
        for i in range(len(weights)):
            if not include_loops and left[i] == right[i]:
                continue
            u, v = self._index[graph._labels[left[i]]], self._index[graph._labels[right[i]]]
            if not self.directed and u > v:
                u, v = v, u
            yield (u, v), weights[i]

    def _result(self, result, kind, work, **metadata):
        result.attrs.update(kind=kind, snapshot_count=self.snapshot_count, ordered=self.ordered,
            directed=self.directed, order=self._metadata["order"], device="cpu", dtype="float64",
            memory_scope=self._metadata["memory_scope"], work_used=work.used,
            max_work=work.maximum, **metadata)
        return result

    def summary(self):
        """Return seven bounded collection metrics, excluding layer/edge histories."""
        result = as_frame(pd.DataFrame([
            ("Snapshots", self.snapshot_count), ("Union nodes", self.node_count),
            ("Stored edge observations", self._edge_observations),
            ("Directed", "Yes" if self.directed else "No"),
            ("Explicit time order", "Yes" if self.ordered else "No"),
            ("Node sets", "May vary; isolates retained"),
            ("Storage", "Sparse snapshots; no dense time-dyad array")], columns=["Metric", "Value"]))
        result.attrs.update(kind="network_snapshots_summary", snapshot_count=self.snapshot_count,
                            ordered=self.ordered, directed=self.directed)
        return result

    def to_latex(self, buf=None, **kwargs):
        """Export the bounded summary table, forwarding Frame export options."""
        kwargs.setdefault("index", False)
        return self.summary().to_latex(buf=buf, **kwargs)

    def to_plot_data(self, max_nodes=1000, max_edges=5000, seed=0, groups=None):
        """Save an explicit timeline using stable union IDs and honest frame counts.

        At most 60 frames are exported. The base plus all frames together must
        fit the chart's aggregate entry and display-memory budgets. Defaults
        remain conservative; they do not sample statistical calculations.
        Temporal layout does not infer timestamps or elapsed-time durations.
        """
        from openecon_charts.network import MAX_NODES, MAX_EDGES, validate_network
        max_nodes = _integer(max_nodes, "max_nodes", MAX_NODES)
        max_edges = _integer(max_edges, "max_edges", MAX_EDGES)
        _integer(seed, "seed", 2**32 - 1, zero=True)
        if self.snapshot_count > 60:
            _error("invalid_option", "A saved timeline supports at most 60 snapshots; select a smaller collection explicitly.")
        if groups is not None:
            _error("invalid_option", "Timeline groups are unsupported; use node_color={'field': 'attrs.NAME'} for explicit attribute colors.")
        upper_nodes = sum(min(max_nodes, graph.node_count) for graph in self._graphs)
        upper_edges = sum(min(max_edges, graph.edge_count) for graph in self._graphs)
        upper_nodes += min(max_nodes, self._graphs[0].node_count)
        upper_edges += min(max_edges, self._graphs[0].edge_count)
        if upper_nodes > MAX_NODES or upper_edges > MAX_EDGES:
            _error("memory_budget", "Timeline base and frames exceed the aggregate display cap; reduce max_nodes/max_edges explicitly.")
        work = _Work(50_000_000)
        self._guard(1024 * upper_nodes + 768 * upper_edges + 4096, work)
        frames = []
        for key, graph in zip(self._keys, self._graphs):
            payload = graph.to_plot_data(max_nodes=max_nodes, max_edges=max_edges, seed=seed)
            local = {node["id"]: self._index[graph._labels[node["id"]]] for node in payload["nodes"]}
            for node in payload["nodes"]:
                node["id"] = local[node["id"]]
            for edge in payload["edges"]:
                edge["source"], edge["target"] = local[edge["source"]], local[edge["target"]]
            frames.append({"label": str(key), "network": payload})
        result = deepcopy(frames[0]["network"])
        result["frames"] = frames
        return validate_network(result)

    def snapshot_summary(self, *, include_loops=False, max_work=50_000_000):
        """Return binary edge counts and density for every explicitly requested layer."""
        include_loops = _boolean(include_loops, "include_loops")
        work = _Work(max_work)
        work.add(4 * self.snapshot_count + self._edge_observations)
        self._guard(1024 * self.snapshot_count + 4096, work)
        columns = {name: [] for name in ("layer", "position", "nodes", "edges", "eligible_dyads", "density")}
        for position, (key, graph) in enumerate(zip(self._keys, self._graphs)):
            edges = sum(1 for _ in self._edges(graph, include_loops))
            n = graph.node_count
            dyads = n * (n - 1) if self.directed else n * (n - 1) // 2
            dyads += n if include_loops else 0
            for name, value in zip(columns, (key, position, n, edges, dyads, edges / dyads if dyads else 0.)):
                columns[name].append(value)
        return self._result(_table(columns), "network_snapshot_summary", work,
            include_loops=include_loops, values="binary", density_empty_dyad_universe=0.,
            undirected_dyads="unordered; each edge counted once")

    def transitions(self, *, include_loops=False, max_work=50_000_000):
        """Adjacent-layer binary edge turnover and node entry/exit, ignoring strengths.

        Edge Jaccard is persisted/union; empty/empty is one. Retention is
        persisted/previous edges; an empty previous set has retention one.
        Turnover is (added+removed)/union, with zero for an empty union. Node
        disappearance removes its edges; comparisons are not conditional on a
        stable node sample. For unordered layers these are descriptive neighbors
        in the captured mapping, not causal time transitions.
        """
        include_loops = _boolean(include_loops, "include_loops")
        work = _Work(max_work)
        work.add(8 * self.snapshot_count + 4 * (self._node_observations + self._edge_observations))
        largest_n = max(graph.node_count for graph in self._graphs)
        largest_e = max(graph.edge_count for graph in self._graphs)
        self._guard(1024 * largest_n + 1024 * largest_e + 2048 * self.snapshot_count + 4096, work)
        columns = {name: [] for name in ("from_layer", "to_layer", "from_position", "to_position",
            "nodes_added", "nodes_removed", "nodes_persisted", "edges_added", "edges_removed",
            "edges_persisted", "edge_jaccard", "edge_retention", "edge_turnover")}
        previous_nodes = set(self._graphs[0]._labels)
        previous_edges = {pair for pair, _ in self._edges(self._graphs[0], include_loops)}
        for position in range(1, self.snapshot_count):
            graph = self._graphs[position]
            nodes = set(graph._labels)
            edges = {pair for pair, _ in self._edges(graph, include_loops)}
            persisted = sum(pair in previous_edges for pair in edges)
            node_persisted = sum(node in previous_nodes for node in nodes)
            added, removed = len(edges) - persisted, len(previous_edges) - persisted
            union = persisted + added + removed
            values = (self._keys[position - 1], self._keys[position], position - 1, position,
                len(nodes) - node_persisted, len(previous_nodes) - node_persisted, node_persisted,
                added, removed, persisted, persisted / union if union else 1.,
                persisted / len(previous_edges) if previous_edges else 1.,
                (added + removed) / union if union else 0.)
            for name, value in zip(columns, values):
                columns[name].append(value)
            previous_nodes, previous_edges = nodes, edges
        return self._result(_table(columns), "network_snapshot_transitions", work,
            include_loops=include_loops, values="binary", weight_changes="ignored; presence only",
            node_set_scope="all observed endpoints; node disappearance removes its edges",
            empty_union_jaccard=1., empty_union_turnover=0., empty_previous_retention=1.)

    def edge_persistence(self, *, include_loops=False, max_edges=100_000, max_work=50_000_000):
        """Return one bounded row per union edge, not a snapshot-by-edge history.

        ``persistence`` divides present snapshots by ALL snapshots.
        ``conditional_persistence`` divides by snapshots where both endpoints
        were observed, including isolates. Longest runs refer to consecutive
        captured positions, even for explicitly unordered multilayer input.
        Node presence is compressed into disjoint intervals; eligibility never
        enumerates all absent dyads or every time-edge cell.
        """
        include_loops = _boolean(include_loops, "include_loops")
        maximum_edges = _integer(max_edges, "max_edges", 2**63 - 1, zero=True)
        work = _Work(max_work)
        work.add(6 * self.snapshot_count + 4 * self._node_observations + 4 * self._edge_observations)
        rows, intervals, interval_count = {}, {}, 0
        self._guard(768 * self.node_count + 4096, work)
        for position, graph in enumerate(self._graphs):
            prospective = min(maximum_edges, len(rows) + graph.edge_count)
            self._guard(768 * self.node_count + 128 * (interval_count + graph.node_count)
                        + 768 * prospective + 4096, work)
            for label in graph._labels:
                node = self._index[label]
                spans = intervals.setdefault(node, [])
                if spans and spans[-1][1] == position - 1:
                    spans[-1][1] = position
                else:
                    spans.append([position, position])
                    interval_count += 1
            for pair, _ in self._edges(graph, include_loops):
                state = rows.get(pair)
                if state is None:
                    if len(rows) >= maximum_edges:
                        _error("output_budget", "Union edge persistence exceeds max_edges; no partial table is returned.")
                    # count, last position, current run, longest run, first position
                    rows[pair] = [1, position, 1, 1, position]
                else:
                    state[0] += 1
                    state[2] = state[2] + 1 if state[1] == position - 1 else 1
                    state[1], state[3] = position, max(state[3], state[2])
        # Admit the COMPLETE eligibility pass and canonical sort before the
        # requested output columns. Stable nodes each require just one interval.
        work.add(len(rows) * max(1, len(rows).bit_length()) + sum(
            len(intervals[u]) + len(intervals[v]) for u, v in rows))
        self._guard(768 * self.node_count + 128 * interval_count + 2048 * len(rows) + 4096, work)
        columns = {name: [] for name in ("source", "target", "present_snapshots", "eligible_snapshots",
            "persistence", "conditional_persistence", "longest_run", "first_layer", "last_layer",
            "first_position", "last_position")}
        for u, v in sorted(rows):
            count, last, _, longest, first = rows[u, v]
            eligible = _interval_overlap(intervals[u], intervals[v])
            if not count <= eligible <= self.snapshot_count:
                _error("precision", "Observed edge presence disagrees with node-presence eligibility.")
            values = (self._labels[u], self._labels[v], count, eligible,
                count / self.snapshot_count, count / eligible, longest,
                self._keys[first], self._keys[last], first, last)
            for name, value in zip(columns, values):
                columns[name].append(value)
        return self._result(_table(columns), "network_edge_persistence", work,
            include_loops=include_loops, values="binary", max_edges=maximum_edges,
            denominator="all snapshots; node absence counts as zero",
            conditional_denominator="snapshots in which both endpoints are observed",
            run_semantics="consecutive captured positions; causal time interpretation only if ordered=True",
            node_presence_storage="compressed disjoint presence intervals")

    def aggregate(self, *, reducer="sum", include_loops=False, max_edges=1_000_000,
                  max_work=50_000_000):
        """Return a union-node sparse aggregate with an explicit strength reducer.

        ``sum``/``mean`` use exact stored binary64 strengths until final scalar
        rounding. Mean divides by ALL snapshots, including node/edge absences.
        ``max`` retains the greatest strength; ``binary`` uses unit presence.
        Overflow or rounding a positive aggregate to zero is rejected. Attribute
        dictionaries are not merged; they remain on the source snapshots.
        """
        if not isinstance(reducer, str) or reducer not in {"sum", "mean", "max", "binary"}:
            _error("invalid_option", "reducer must be 'sum', 'mean', 'max', or 'binary'.")
        include_loops = _boolean(include_loops, "include_loops")
        maximum_edges = _integer(max_edges, "max_edges", 2**63 - 1, zero=True)
        work = _Work(max_work)
        work.add(8 * self.snapshot_count + 8 * self._edge_observations)
        self._guard(768 * self.node_count + 4096, work)
        exponent, maximum_power = 0, None
        if reducer in {"sum", "mean"}:
            minimum_power = None
            for graph in self._graphs:
                for _, value in self._edges(graph, include_loops):
                    mantissa, power = _parts(value)
                    minimum_power = power if minimum_power is None else min(minimum_power, power)
                    maximum_power = (power + mantissa.bit_length() if maximum_power is None else
                                     max(maximum_power, power + mantissa.bit_length()))
            exponent = 0 if minimum_power is None else minimum_power
        bits = max(1, (0 if maximum_power is None else maximum_power) - exponent
                   + self.snapshot_count.bit_length())
        words = max(1, (bits + 29) // 30)
        work.add(self._edge_observations * (2 * words if reducer in {"sum", "mean"} else 1))
        integer_bytes = 32 + 4 * words
        totals = {}
        for graph in self._graphs:
            prospective = min(maximum_edges, len(totals) + graph.edge_count)
            self._guard(768 * self.node_count + (512 + integer_bytes) * prospective + 4096, work)
            for pair, value in self._edges(graph, include_loops):
                if pair not in totals and len(totals) >= maximum_edges:
                    _error("output_budget", "Aggregate union edges exceed max_edges; no partial graph is returned.")
                if reducer == "binary":
                    totals[pair] = 1.
                elif reducer == "max":
                    totals[pair] = max(totals.get(pair, 0.), value)
                else:
                    mantissa, power = _parts(value)
                    totals[pair] = totals.get(pair, 0) + (mantissa << (power - exponent))
        e, n = len(totals), self.node_count
        work.add(e * (max(1, e.bit_length()) + 4 * words) + n)
        label_bytes = 4096 + sum(512 + 2 * sys.getsizeof(label)
            + (len(label.encode("utf-8")) if isinstance(label, str) else 32) for label in self._labels)
        # Output dictionaries/COO/arcs/constructor buffers coexist with exact
        # accumulation and all source snapshots. Admit them before allocation.
        self._guard(label_bytes + 1024 * n + (1024 + integer_bytes) * e + 8192, work)
        divisor = self.snapshot_count if reducer == "mean" else 1
        with torch.device("cpu"), torch.no_grad():
            index = torch.empty((2, e), dtype=torch.int64, device="cpu")
            values = torch.empty(e, dtype=torch.float64, device="cpu")
            left, right, weight = (memoryview(tensor.numpy()) for tensor in (index[0], index[1], values))
            for i, pair in enumerate(sorted(totals)):
                left[i], right[i] = pair
                value = totals[pair]
                weight[i] = _export(value, exponent, divisor) if reducer in {"sum", "mean"} else value
            edges = torch.sparse_coo_tensor(index, values, (n, n), dtype=torch.float64,
                device="cpu", is_coalesced=True, check_invariants=True)
            allowance = min(graph._budget.limit for graph in self._unique_graphs)
            metadata = dict(derived_from="network snapshot aggregation", snapshot_count=self.snapshot_count,
                reducer=reducer, include_loops=include_loops, ordered=self.ordered,
                mean_denominator="all snapshots, including node/edge absences",
                input_rows=e, positive_edge_rows=e, missing_rows_dropped=0, zero_weight_rows_dropped=0,
                duplicate_edge_rows_aggregated=0, max_edges=maximum_edges, max_work=work.maximum,
                work_used=work.used, device="cpu", dtype="float64",
                arithmetic="exact stored binary64 integer units; round only final scalar export"
                    if reducer in {"sum", "mean"} else "stored positive strengths" if reducer == "max" else "binary presence",
                maximum_integer_bits=bits if reducer in {"sum", "mean"} else None,
                weight_semantics="positive aggregate strength" if reducer != "binary" else "unit edge presence",
                attributes="not merged; original snapshot attributes retained",
                order=self._metadata["order"], memory_scope=self._metadata["memory_scope"])
            return Network(self._labels, self._index.copy(), edges, self.directed, reducer != "binary",
                           _Budget(allowance / 1024**2), label_bytes, metadata)

    def temporal_path(self, source, target, *, start=None, max_work=50_000_000):
        """Earliest-arrival strict time-respecting path; one edge per snapshot.

        Requires ``ordered=True``. The source must be observed at ``start``;
        ``None`` means the first captured layer. Both exact node IDs must occur
        in the union. Waiting is allowed only while that node stays observed in
        every intervening snapshot. Strengths are ignored; loops never advance
        a path. Arrival is an ordered POSITION, not elapsed time, fastest duration
        or minimum total hops. Ties use the smallest canonical predecessor ID.

        Return only traversed edge rows. ``attrs['reachable']`` distinguishes an
        unreachable empty table from the zero-hop source-equals-target path.
        """
        if not self.ordered:
            _error("invalid_option", "temporal_path requires ordered=True; general layers have no implied time order.")
        source, target = _label(source), _label(target)
        if source not in self._index or target not in self._index:
            _error("invalid_label", "source and target must be exact node IDs in the snapshot union.")
        first = 0 if start is None else self._layer_index.get(_label(start))
        if first is None:
            _error("invalid_label", "start must be an exact captured layer ID.")
        if source not in self._graphs[first]._index:
            _error("invalid_label", "The source must be observed in the starting snapshot.")
        work = _Work(max_work)
        selected = self._graphs[first:]
        work.add(16 * len(selected) + 4 * sum(graph.node_count for graph in selected)
                 + 8 * sum(graph.edge_count for graph in selected))
        self._guard(1024 * self.node_count + 1024 * len(selected) + 4096, work)
        source_index, target_index = self._index[source], self._index[target]
        active, witnesses = {source_index: -1}, []
        arrival = first if source_index == target_index else None
        for position in range(first, self.snapshot_count):
            if arrival is not None:
                break
            graph = self._graphs[position]
            present = {self._index[label] for label in graph._labels}
            active = {node: record for node, record in active.items() if node in present}
            candidates = {}
            for (u, v), _ in self._edges(graph, False):
                if u in active and v not in active:
                    if v not in candidates or u < candidates[v][0]:
                        candidates[v] = (u, active[u])
                if not self.directed and v in active and u not in active:
                    if u not in candidates or v < candidates[u][0]:
                        candidates[u] = (v, active[v])
            self._guard(1024 * self.node_count + 192 * (len(witnesses) + len(candidates))
                        + 1024 * len(selected) + 4096, work)
            # Synchronous update: arrivals at this layer cannot emit another
            # edge until the next layer. A missing node discards its reachability
            # state; reappearing later requires a new contact-based witness.
            for v, (u, previous) in candidates.items():
                active[v] = len(witnesses)
                witnesses.append((previous, u, v, position))
            if target_index in active:
                arrival = position
            if not active:
                break
        path = []
        if arrival is not None and source_index != target_index:
            record = active[target_index]
            while record >= 0:
                previous, u, v, position = witnesses[record]
                path.append((self._labels[u], self._labels[v], self._keys[position], position))
                record = previous
            path.reverse()
        columns = {name: [] for name in ("source", "target", "layer", "position")}
        for row in path:
            for name, value in zip(columns, row):
                columns[name].append(value)
        return self._result(_table(columns), "network_temporal_path", work,
            source=source, target=target, start_layer=self._keys[first], start_position=first,
            reachable=arrival is not None, arrival_layer=None if arrival is None else self._keys[arrival],
            arrival_position=arrival, hops=len(path) if arrival is not None else None,
            traversal="strictly increasing snapshot positions; at most one edge per layer",
            waiting="allowed only while the node is observed in every intervening snapshot",
            objective="earliest arrival position; not shortest hops, fastest duration or elapsed time",
            values="binary; positive strengths ignored", tie_break="smallest canonical typed predecessor ID",
            reachable_state_storage="current sparse node states plus contact witnesses; no time-node matrix")
