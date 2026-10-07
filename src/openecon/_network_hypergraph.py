"""Typed hyperedges and sparse Torch incidence; projections are explicit graphs."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _Budget, _error, _integer, _label, _weight, network
from openecon._network_flow import _Work


def _sequence(value, name):
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        _error("invalid_data", f"{name} must be an iterable of typed node IDs.")
    return value


def hypergraph(records, *, nodes=None, directed=False, max_memberships=1_000_000,
               max_memory_mb=256, max_work=50_000_000):
    """Build resident typed hyperedges; repeated IDs/members are errors, zero weights stay.

    Undirected records contain id/members/weight. Directed records contain
    id/tail/head/weight, with nonempty tail and head; their overlap is allowed.
    Distinct IDs may describe parallel hyperedges. Singleton edges are valid.
    No pairwise network is inferred until an explicit projection is requested.
    """
    return Hypergraph(records, nodes=nodes, directed=directed, max_memberships=max_memberships,
                      max_memory_mb=max_memory_mb, max_work=max_work)


class Hypergraph:
    """Resident typed hyperedges with sparse incidence and explicit budgeted projections."""
    def __init__(self, records, *, nodes=None, directed=False, max_memberships=1_000_000,
                 max_memory_mb=256, max_work=50_000_000):
        if not isinstance(directed, bool):
            _error("invalid_option", "directed must be a boolean.")
        self.directed = directed
        self._budget = _Budget(max_memory_mb)
        maximum = _integer(max_memberships, "max_memberships", 2**63 - 1)
        work = _Work(max_work, 1)
        labels, index, ids, seen_ids, groups, weights = [], {}, [], set(), [], []
        count, text_bytes = 0, 0

        def guard(extra=0):
            self._budget.check(4096 + text_bytes + 512 * (len(labels) + len(ids)) + 256 * count + extra)

        def add_node(raw):
            nonlocal text_bytes
            work.add()
            label = _label(raw)
            if label not in index:
                text_bytes += len(label.encode()) if isinstance(label, str) else 64
                guard(1024)
                index[label] = len(labels)
                labels.append(label)
            return index[label]

        if nodes is not None:
            for raw in _sequence(nodes, "nodes"):
                add_node(raw)
        for record in _sequence(records, "records"):
            work.add()
            allowed = {"id", "weight", "tail", "head"} if directed else {"id", "weight", "members"}
            if not isinstance(record, Mapping) or "id" not in record or set(record) - allowed:
                _error("invalid_data", "Each hyperedge requires id, its memberships, and optional weight only.")
            edge_id = _label(record["id"])
            if edge_id in seen_ids:
                _error("duplicate_id", "Hyperedge IDs must be unique; parallel edges need separate IDs.")
            value = _weight(record.get("weight", 1.))
            text_bytes += len(edge_id.encode()) if isinstance(edge_id, str) else 64
            guard(1024)
            ids.append(edge_id)
            seen_ids.add(edge_id)
            memberships = []
            for role in (("tail", "head") if directed else ("members",)):
                if role not in record:
                    _error("invalid_data", f"Each hyperedge requires {role}.")
                selected, seen = [], set()
                for raw in _sequence(record[role], role):
                    work.add()
                    if count >= maximum:
                        _error("output_budget", "Hypergraph memberships exceed max_memberships.")
                    node = add_node(raw)
                    if node in seen:
                        _error("duplicate_membership", "Membership may appear once per hyperedge role.")
                    count += 1
                    guard()
                    seen.add(node)
                    selected.append(node)
                if not selected:
                    _error("invalid_data", "Each hyperedge role must contain at least one node.")
                memberships.append(tuple(selected))
            groups.append(tuple(memberships))
            weights.append(value)
        self._labels, self._index = tuple(labels), index
        self._ids, self._groups = tuple(ids), tuple(groups)
        self._membership_count = count
        self._base_bytes = 4096 + text_bytes + 512 * (len(labels) + len(ids)) + 256 * count
        self._guard(64 * count + 64 * len(ids))
        with torch.device("cpu"), torch.no_grad():
            self._weights = torch.tensor(weights, dtype=torch.float64)
            matrices = []
            for role in range(2 if directed else 1):
                pairs = [(node, edge) for edge, group in enumerate(groups) for node in group[role]]
                indices = torch.tensor(pairs, dtype=torch.int64).T.contiguous() if pairs else torch.empty((2, 0), dtype=torch.int64)
                matrices.append(torch.sparse_coo_tensor(indices, torch.ones(len(pairs), dtype=torch.float64),
                    (len(labels), len(ids)), dtype=torch.float64, device="cpu", check_invariants=True).coalesce())
            self._matrices = tuple(matrices)
        self._metadata = dict(kind="hypergraph", nodes=len(labels), hyperedges=len(ids),
            memberships=count, directed=directed, weighted=True, device="cpu", dtype="float64",
            storage="sparse incidence O(V+H+I)", max_memory_bytes=self._budget.limit,
            owned_memory_estimate_bytes=self._base_bytes, max_work=work.maximum, work_used=work.used,
            duplicate_membership="error within each role", parallel_hyperedges="distinct typed IDs",
            zero_weight="retained membership; contributes degree but zero strength",
            direct_algorithms=["incidence", "incidence_matvec", "degree", "strength", "summary"],
            projection_algorithms="explicit clique/star Network methods; not direct hypergraph methods")

    @property
    def metadata(self):
        return deepcopy({**self._metadata, "owned_peak_estimate_bytes": self._budget.peak})

    @property
    def node_count(self):
        return len(self._labels)

    @property
    def edge_count(self):
        return len(self._ids)

    def _guard(self, extra):
        self._budget.check(self._base_bytes + extra + 4096)

    def _role(self, role):
        allowed = ("tail", "head") if self.directed else ("members",)
        if role not in allowed:
            _error("invalid_option", f"role must be one of {allowed}; directed incidence requires tail/head.")
        return allowed.index(role)

    def incidence(self, *, role="members", weighted=False):
        """Copy sparse V-by-H binary incidence; weighted=True multiplies each column by edge weight."""
        if not isinstance(weighted, bool):
            _error("invalid_option", "weighted must be boolean.")
        matrix = self._matrices[self._role(role)]
        self._guard(96 * matrix._nnz() + 4096)
        with torch.device("cpu"), torch.no_grad():
            values = self._weights[matrix.indices()[1]] if weighted else matrix.values().clone()
            return torch.sparse_coo_tensor(matrix.indices().clone(), values, matrix.shape,
                dtype=torch.float64, device="cpu", check_invariants=True).coalesce()

    def incidence_matvec(self, values, *, role="members", weighted=False, transpose=False):
        """Sparse incidence/vector product in CPU float64; no dense V-by-H matrix."""
        if not isinstance(transpose, bool):
            _error("invalid_option", "transpose must be boolean.")
        count = self.node_count if transpose else self.edge_count
        if (not isinstance(values, torch.Tensor) or values.ndim != 1 or values.numel() != count
                or values.layout != torch.strided or values.is_complex()):
            _error("invalid_option", "values must be a one-dimensional Torch tensor aligned to incidence.")
        self._guard(128 * self._membership_count + 64 * (self.node_count + self.edge_count))
        with torch.device("cpu"), torch.no_grad():
            vector = values.detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(vector).all()):
                _error("invalid_data", "Incidence vector must be finite.")
            matrix = self.incidence(role=role, weighted=weighted)
            result = torch.sparse.mm(matrix.T if transpose else matrix, vector[:, None]).flatten()
            if not bool(torch.isfinite(result).all()):
                _error("precision", "Incidence product exceeds float64 range.")
            return result

    def degree(self):
        """Hyperedge membership degree and sum of incident weights, separately per directed role."""
        self._guard(128 * self._membership_count + 512 * self.node_count)
        columns = {"node": pd.Series(self._labels, dtype=object)}
        with torch.device("cpu"), torch.no_grad():
            for role, matrix in zip(("out", "in") if self.directed else ("",), self._matrices):
                indices = matrix.indices()
                degree = torch.bincount(indices[0], minlength=self.node_count)
                strength = torch.zeros(self.node_count, dtype=torch.float64)
                strength.index_add_(0, indices[0], self._weights[indices[1]])
                if not bool(torch.isfinite(strength).all()):
                    _error("precision", "Hypergraph node strength exceeds float64 range.")
                prefix = role + "_" if role else ""
                columns[prefix + "hyperdegree"] = degree.numpy()
                columns[prefix + "strength"] = strength.numpy()
        result = as_frame(pd.DataFrame(columns))
        result.attrs.update(kind="hypergraph_degree", hypergraph=self.metadata, direct=True, projected=False)
        return result

    def memberships(self):
        """Full typed node/hyperedge/role table; each role membership occurs once."""
        self._guard(512 * self._membership_count)
        rows = [(self._labels[node], self._ids[edge], role, float(self._weights[edge]))
                for edge, groups in enumerate(self._groups)
                for role, members in zip(("tail", "head") if self.directed else ("members",), groups)
                for node in members]
        result = as_frame(pd.DataFrame(rows, columns=["node", "hyperedge", "role", "weight"]))
        result["node"] = pd.Series([row[0] for row in rows], dtype=object)
        result["hyperedge"] = pd.Series([row[1] for row in rows], dtype=object)
        result.attrs.update(kind="hypergraph_memberships", hypergraph=self.metadata, direct=True)
        return result

    def strength(self):
        """Sum of incident hyperedge weights per node, separately for directed tail/head."""
        result = self.degree()
        return as_frame(result[[name for name in result.columns if name == "node" or name.endswith("strength")]])

    def summary(self):
        """Bounded hypergraph counts and support; never expands into pairwise edges."""
        rows = [("Nodes", self.node_count), ("Hyperedges", self.edge_count),
                ("Memberships", self._membership_count), ("Directed", self.directed),
                ("Storage", "Sparse incidence"), ("Device", "CPU float64"),
                ("Owned memory estimate", self._base_bytes), ("Projection", "None")]
        result = as_frame(pd.DataFrame(rows, columns=["Metric", "Value"]))
        result.attrs.update(kind="hypergraph_summary", hypergraph=self.metadata, direct=True)
        return result

    def clique_projection(self, *, reducer="sum", max_edges=100_000, max_work=50_000_000):
        """Explicit pair graph; sum/count/normalized edge contribution, with pre-expansion admission.

        Directed tails connect to heads. Undirected members form unordered pairs.
        normalized divides each hyperedge weight by its nonloop candidate pair count.
        max_edges conservatively limits candidate pairs before duplicate aggregation.
        Singleton hyperedges remain nodes but cannot create a clique edge.
        """
        if not isinstance(reducer, str) or reducer not in {"sum", "count", "normalized"}:
            _error("invalid_option", "reducer must be sum, count or normalized.")
        maximum = _integer(max_edges, "max_edges", 2**63 - 1, zero=True)
        planned = sum(len(g[0]) * len(g[1]) if self.directed else len(g[0]) * (len(g[0]) - 1) // 2
                      for g in self._groups)
        if planned > maximum:
            _error("output_budget", "Clique candidate pairs exceed max_edges before expansion.")
        work = _Work(max_work, self._membership_count + planned)
        self._guard(768 * planned + 512 * self.node_count)
        pairs = {}
        for edge, groups in enumerate(self._groups):
            candidates = ((a, b) for a in groups[0] for b in groups[1] if a != b) if self.directed else (
                (a, b) if a < b else (b, a) for i, a in enumerate(groups[0]) for b in groups[0][i + 1:])
            divisor = (len(groups[0]) * len(groups[1]) - len(set(groups[0]) & set(groups[1]))
                       if self.directed else len(groups[0]) * (len(groups[0]) - 1) // 2)
            value = 1. if reducer == "count" else float(self._weights[edge]) / (divisor or 1) if reducer == "normalized" else float(self._weights[edge])
            for pair in candidates:
                total = pairs.get(pair, 0.) + value
                if not math.isfinite(total):
                    _error("precision", "Projected strengths exceed float64 range.")
                if reducer == "normalized" and value == 0 and self._weights[edge] > 0:
                    _error("precision", "Normalized projection loses a positive weight.")
                pairs[pair] = total
        reserved = self._base_bytes + 768 * planned + 512 * self.node_count + 4096
        result = network(({"source": self._labels[a], "target": self._labels[b], "weight": value}
                          for (a, b), value in pairs.items()), nodes=self._labels, directed=self.directed,
                         weight="weight", max_memory_mb=(self._budget.limit - reserved) / 1024**2)
        self._guard(768 * planned + result._base_bytes + 512 * self.node_count)
        result._metadata.update(hypergraph_projection="clique", reducer=reducer,
            candidate_pairs=planned, hypergraph=self.metadata, direct_hypergraph_analysis=False,
            projection_work_used=work.used, projection_max_work=work.maximum)
        return result

    def star_projection(self, *, max_edges=100_000, max_work=50_000_000):
        """Explicit node/hyperedge bipartite graph with typed original IDs in node attributes.

        Proxy IDs are disjoint integer ranges; each spoke carries hyperedge weight.
        Directed tail -> hyperedge -> head. Graph paths sum spoke costs, not an
        inferred hyperedge cost. Zero-weight spokes follow Network zero-edge policy.
        """
        maximum = _integer(max_edges, "max_edges", 2**63 - 1, zero=True)
        if self._membership_count > maximum:
            _error("output_budget", "Star incidence edges exceed max_edges before expansion.")
        work = _Work(max_work, self.node_count + self.edge_count + self._membership_count)
        self._guard(1024 * (self.node_count + self.edge_count) + 768 * self._membership_count)
        n = self.node_count
        attrs = {i: {"kind": "node", "original_id": label} for i, label in enumerate(self._labels)}
        attrs.update({n + i: {"kind": "hyperedge", "original_id": label} for i, label in enumerate(self._ids)})
        def records():
            for edge, groups in enumerate(self._groups):
                for role, group in enumerate(groups):
                    for node in group:
                        a, b = (n + edge, node) if self.directed and role == 1 else (node, n + edge)
                        yield {"source": a, "target": b, "weight": float(self._weights[edge])}
        reserved = self._base_bytes + 1024 * (n + self.edge_count) + 768 * self._membership_count + 4096
        result = network(records(), nodes=range(n + self.edge_count), directed=self.directed,
                         weight="weight", node_attributes=attrs, max_memory_mb=(self._budget.limit - reserved) / 1024**2)
        self._guard(result._base_bytes + 1024 * (n + self.edge_count) + 768 * self._membership_count)
        result._metadata.update(hypergraph_projection="star", hypergraph=self.metadata,
            proxy_identity="node attributes kind/original_id; nodes then hyperedges", weight_semantics="hyperedge weight per spoke",
            direct_hypergraph_analysis=False, projection_work_used=work.used, projection_max_work=work.maximum)
        return result

    def to_dict(self):
        """Inert typed JSON records preserving parallel, singleton and zero-weight hyperedges."""
        self._guard(1024 * (self.node_count + self.edge_count) + 512 * self._membership_count)
        records = []
        for edge, groups in enumerate(self._groups):
            record = {"id": self._ids[edge], "weight": float(self._weights[edge])}
            for role, members in zip(("tail", "head") if self.directed else ("members",), groups):
                record[role] = [self._labels[node] for node in members]
            records.append(record)
        return {"format": "openecon-hypergraph", "version": 1, "directed": self.directed,
                "nodes": list(self._labels), "hyperedges": records}

    def write(self, path):
        """Write a new inert JSON file; existing files are never overwritten."""
        # Worst-case JSON escapes, per-membership repeated labels and all
        # simultaneous dict/text/UTF-8 buffers are admitted before serialization.
        lengths = [len(str(label).encode("utf-8")) for label in self._labels]
        size = 1024 + sum(lengths) * 6 + 512 * self.edge_count
        for edge, groups in enumerate(self._groups):
            size += 6 * len(str(self._ids[edge]).encode("utf-8"))
            size += sum(6 * lengths[node] + 32 for group in groups for node in group)
        self._guard(8 * size + 1024 * (self.node_count + self.edge_count))
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
        self._guard(3 * len(encoded.encode("utf-8")))
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        return Path(path)


def read_hypergraph(path, *, max_memberships=1_000_000, max_memory_mb=256, max_work=50_000_000):
    """Read bounded inert hypergraph JSON, then revalidate every typed membership."""
    budget = _Budget(max_memory_mb)
    path = Path(path)
    maximum = max(0, (budget.limit - 4096) // 16)
    if path.stat().st_size > maximum:
        _error("memory_budget", "Hypergraph JSON exceeds the pre-parse memory bound.")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        _error("memory_budget", "Hypergraph JSON grew past the pre-parse memory bound.")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        _error("invalid_data", f"Invalid hypergraph JSON: {type(exc).__name__}.")
    if (not isinstance(data, dict) or set(data) != {"format", "version", "directed", "nodes", "hyperedges"}
            or data["format"] != "openecon-hypergraph" or type(data["version"]) is not int or data["version"] != 1
            or not isinstance(data["nodes"], list) or not isinstance(data["hyperedges"], list)):
        _error("invalid_data", "Unsupported hypergraph format or version.")
    result = hypergraph(data["hyperedges"], nodes=data["nodes"], directed=data["directed"],
                        max_memberships=max_memberships, max_memory_mb=(budget.limit - 16 * len(raw)) / 1024**2,
                        max_work=max_work)
    result._metadata["read_parse_reserve_bytes"] = 16 * len(raw)
    result._metadata["max_memory_bytes"] = budget.limit
    result._budget.limit = budget.limit
    return result
