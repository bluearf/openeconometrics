"""Two-phase native Torch simplex for bounded directed cost-flow LPs.

No graph/LP library is a runtime dependency. Conservation equations and shared
capacity slacks form Ax=b,x>=0. Bland's rule prevents combinatorial cycling;
iteration/work/owned-memory guards bound the admitted dense tableau. Artificial
columns retain the row transformation, including redundant conservation rows,
so every returned dual/Farkas certificate refers to the original constraints.

https://courses.csail.mit.edu/6.854/19/Notes/n14-LP_algorithms.html
https://courses.csail.mit.edu/6.854/18/Notes/n12-duality.html
"""

from collections.abc import Mapping
import math

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon._network_flow import _Work, _endpoint
from openecon.networks import Network, _error, _integer

EPS = 1e-10


class CostFlowResult(dict):
    def summary(self):
        return as_frame(
            pd.DataFrame(
                {
                    "Metric": [
                        "Status",
                        "Objective",
                        "Commodities",
                        "Domain",
                        "Certified",
                        "Pivots",
                    ],
                    "Value": [
                        self["status"],
                        self.get("objective"),
                        self["metadata"]["commodities"],
                        self["metadata"]["domain"],
                        self["metadata"]["certified"],
                        self["metadata"]["pivots"],
                    ],
                }
            )
        )

    def to_latex(self, buf=None, **kwargs):
        kwargs.setdefault("index", False)
        return self.summary().to_latex(buf=buf, **kwargs)

    def __repr__(self):
        return f"CostFlowResult(status={self['status']!r}, objective={self.get('objective')!r})"


class _Simplex:
    def __init__(self, a, b, work, max_pivots):
        self.a, self.b, self.work = a, b, work
        self.max_pivots, self.pivots = max_pivots, 0
        self.m, self.n = a.shape
        self.sign = torch.where(b < 0, -1.0, 1.0).to(torch.float64)
        self.table = torch.cat(
            [
                a * self.sign[:, None],
                torch.eye(self.m, dtype=torch.float64),
                (b * self.sign)[:, None],
            ],
            dim=1,
        )
        self.table = torch.cat(
            [self.table, torch.zeros((1, self.n + self.m + 1), dtype=torch.float64)], dim=0
        )
        self.basis = torch.arange(self.n, self.n + self.m)

    def objective(self, c):
        self.table[-1] = torch.cat([c, torch.zeros(1, dtype=torch.float64)]) - (
            c[self.basis, None] * self.table[:-1]
        ).sum(dim=0)

    def pivot(self, row, column):
        self.work.add(2 * self.table.numel())
        self.pivots += 1
        if self.pivots > self.max_pivots:
            _error("work_budget", "Cost flow exceeds max_pivots; no partial solution is returned.")
        self.table[row] /= self.table[row, column].clone()
        factors = self.table[:, column].clone()
        factors[row] = 0
        self.table -= factors[:, None] * self.table[row].clone()[None, :]
        self.basis[row] = column
        if not bool(torch.isfinite(self.table).all()):
            _error(
                "precision_unsupported",
                "Nonfinite simplex tableau; this numerical domain is unsupported.",
            )

    def iterate(self, allowed):
        while True:
            candidates = torch.nonzero(self.table[-1, :allowed] < -EPS).flatten()
            if not len(candidates):
                return None
            entering = int(candidates[0])
            column, rhs = self.table[:-1, entering], self.table[:-1, -1]
            possible = torch.nonzero(column > EPS).flatten()
            if not len(possible):
                ray = torch.zeros(allowed, dtype=torch.float64)
                ray[entering] = 1
                ray[self.basis] = -column
                return ray
            ratios = rhs[possible].clamp_min(0) / column[possible]
            minimum = ratios.min()
            tied = possible[ratios <= minimum + EPS * max(1.0, abs(float(minimum)))]
            leaving = int(tied[torch.argmin(self.basis[tied])])
            self.pivot(leaving, entering)

    def solve(self, costs):
        phase1 = torch.cat([torch.zeros(self.n), torch.ones(self.m)]).to(torch.float64)
        self.objective(phase1)
        if self.iterate(self.n + self.m) is not None:
            _error(
                "precision_unsupported",
                "Unexpected unbounded artificial-variable phase; no certificate is returned.",
            )
        if -float(self.table[-1, -1]) > EPS:
            dual = (1 - self.table[-1, self.n : self.n + self.m]) * self.sign
            if float((self.a.T @ dual).max()) > 1e-7 or not float(self.b @ dual) > EPS:
                _error(
                    "precision_unsupported",
                    "Infeasibility certificate failed numerical verification.",
                )
            return "infeasible", None, dual, None
        # Remove zero artificials. Redundant *transformed* rows are discarded;
        # artificial columns continue to map retained equations to original rows.
        row = 0
        while row < len(self.basis):
            if int(self.basis[row]) >= self.n:
                free = torch.ones(self.n, dtype=torch.bool)
                free[self.basis[self.basis < self.n]] = False
                candidates = torch.nonzero(free & (self.table[row, : self.n].abs() > EPS)).flatten()
                if len(candidates):
                    self.pivot(row, int(candidates[0]))
                else:
                    keep = torch.arange(len(self.table)) != row
                    self.table = self.table[keep]
                    self.basis = self.basis[torch.arange(len(self.basis)) != row]
                    continue
            row += 1
        cost_scale = max(1.0, float(costs.abs().max()))
        self.objective(torch.cat([costs / cost_scale, torch.zeros(self.m, dtype=torch.float64)]))
        ray = self.iterate(self.n)
        point = torch.zeros(self.n, dtype=torch.float64)
        point[self.basis] = self.table[:-1, -1]
        dual = -self.table[-1, self.n : self.n + self.m] * self.sign * cost_scale
        tolerance = 1e-7
        if (
            float((self.a @ point - self.b).abs().max()) > tolerance
            or float(point.min()) < -tolerance
        ):
            _error("precision_unsupported", "Primal conservation/capacity certificate failed.")
        if ray is not None:
            if (
                float((self.a @ ray).abs().max()) > 1e-7
                or float(ray.min()) < -1e-7
                or not float(costs @ ray) < -EPS
            ):
                _error("precision_unsupported", "Unbounded improving-ray certificate failed.")
            return "unbounded", point, None, ray
        reduced = costs - self.a.T @ dual
        gap = float(costs @ point - self.b @ dual)
        objective_tolerance = 1e-7 * max(1.0, abs(float(costs @ point)), abs(float(self.b @ dual)))
        if float(reduced.min()) < -1e-7 * cost_scale or abs(gap) > objective_tolerance:
            _error("precision_unsupported", "Dual reduced-cost/optimality certificate failed.")
        return "optimal", point, dual, None


def _edge_values(graph, values, name, *, infinite=False):
    edges = graph._edges.indices()
    if isinstance(values, Mapping):
        keys = [(graph._labels[int(u)], graph._labels[int(v)]) for u, v in edges.T]
        if len(values) != len(keys) or any(key not in values for key in keys):
            _error(
                "invalid_option",
                f"{name} must supply exactly one value per coalesced directed edge.",
            )
        values = [values[key] for key in keys]
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu").clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        _error("invalid_option", f"{name} must contain numeric edge values: {exc}")
    if (
        tensor.shape != (graph.edge_count,)
        or bool(torch.isnan(tensor).any())
        or bool(torch.isneginf(tensor).any())
        or (not infinite and not bool(torch.isfinite(tensor).all()))
    ):
        _error("invalid_option", f"{name} must have one valid value per directed edge.")
    return tensor


def cost_flow(graph, commodities, costs, *, capacities, domain, max_work, max_entries, max_pivots):
    if type(graph) is not Network or not graph.directed:
        _error(
            "unsupported_graph",
            "Cost flow currently validates coalesced directed Network snapshots.",
        )
    if (
        not isinstance(commodities, Mapping)
        or not commodities
        or any(not isinstance(key, str) or not key for key in commodities)
    ):
        _error("invalid_option", "commodities must map nonempty names to node-demand mappings.")
    k, nodes, edges = len(commodities), graph.node_count, graph.edge_count
    if not edges or not nodes:
        _error("invalid_option", "Cost flow needs at least one retained edge.")
    if domain not in {"fractional", "integral"} or (k > 1 and domain == "integral"):
        _error(
            "unsupported_flow_domain",
            "Integral mode is validated for single-commodity flow only; multiple commodities are fractional and splittable.",
        )
    max_entries = _integer(max_entries, "max_entries", 2**63 - 1)
    max_pivots = _integer(max_pivots, "max_pivots", 2**63 - 1)
    # Conservative admission before edge vectors, demand matrices or tableau allocation.
    rows_max, variables_max = k * nodes + edges, k * edges + edges
    planned = (rows_max + 1) * (variables_max + rows_max + 1)
    if planned > max_entries:
        _error(
            "memory_budget",
            "Cost-flow dense tableau exceeds max_entries; choose an explicit larger budget or a smaller problem.",
        )
    graph._guard(8 * (4 * planned + 8 * (k * nodes + k * edges + edges)))
    work = _Work(max_work, planned)
    cap = _edge_values(
        graph,
        graph._edges.values() if capacities is None else capacities,
        "capacities",
        infinite=True,
    )
    if bool((cap < 0).any()):
        _error(
            "invalid_option",
            "Capacities must be nonnegative; +inf explicitly denotes unlimited capacity.",
        )
    commodity_costs = isinstance(costs, Mapping) and set(costs) == set(commodities)
    c = torch.stack(
        [
            _edge_values(graph, costs[name] if commodity_costs else costs, "costs")
            for name in commodities
        ]
    )
    nonzero_cost = c.abs()[c != 0]
    if len(nonzero_cost) and float(nonzero_cost.max() / nonzero_cost.min()) > 1e8:
        _error(
            "precision_unsupported",
            "Cost dynamic range exceeds the validated 1e8 ratio; small reduced costs would be lost by normalization.",
        )
    demands = torch.zeros((k, nodes), dtype=torch.float64)
    for index, values in enumerate(commodities.values()):
        if not isinstance(values, Mapping):
            _error("invalid_option", "Each commodity must map node labels to net-inflow demands.")
        for node, value in values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                _error("invalid_option", "Demands must be finite numeric values.")
            if not math.isfinite(numeric):
                _error("invalid_option", "Demands must be finite.")
            demands[index, _endpoint(graph, node, "demand node")] = numeric
    if domain == "integral":
        finite = cap[torch.isfinite(cap)]
        if (
            bool((demands != demands.round()).any())
            or bool((finite != finite.round()).any())
            or float(demands.abs().max()) > 2**40
            or (len(finite) and float(finite.abs().max()) > 2**40)
        ):
            _error(
                "unsupported_flow_domain",
                "Integral flow requires integral demands/capacities within the exact validated 2^40 domain.",
            )
    finite_edges = torch.nonzero(torch.isfinite(cap)).flatten()
    m, nf = k * nodes + len(finite_edges), k * edges
    a = torch.zeros((m, nf + len(finite_edges)), dtype=torch.float64)
    uv = graph._edges.indices()
    edge_ids = torch.arange(edges)
    for commodity in range(k):
        # index_add handles self-loops: incoming and outgoing cancel exactly.
        block = a[
            commodity * nodes : (commodity + 1) * nodes, commodity * edges : (commodity + 1) * edges
        ]
        block.index_put_((uv[1], edge_ids), torch.ones(edges, dtype=torch.float64), accumulate=True)
        block.index_put_(
            (uv[0], edge_ids), -torch.ones(edges, dtype=torch.float64), accumulate=True
        )
        a[k * nodes + torch.arange(len(finite_edges)), commodity * edges + finite_edges] = 1
    a[k * nodes + torch.arange(len(finite_edges)), nf + torch.arange(len(finite_edges))] = 1
    b = torch.cat([demands.flatten(), cap[finite_edges]])
    objective = torch.cat([c.flatten(), torch.zeros(len(finite_edges), dtype=torch.float64)])
    solver = _Simplex(a, b, work, max_pivots)
    status, point, dual, ray = solver.solve(objective)
    if point is not None and domain == "integral":
        rounded = point.round()
        if (
            float((point - rounded).abs().max()) > 1e-6
            or float((a @ rounded - b).abs().max()) > 1e-7
        ):
            _error(
                "precision_unsupported",
                "Integral-flow certificate failed; no rounded partial answer is supplied.",
            )
        point = rounded
        if status == "optimal" and abs(float(objective @ point - b @ dual)) > 1e-7 * max(
            1.0, abs(float(objective @ point))
        ):
            _error(
                "precision_unsupported",
                "Rounded integral objective does not match its dual certificate.",
            )
    names = list(commodities)

    def flow_frame(vector):
        data = {
            "commodity": [],
            "edge_id": [],
            "source": [],
            "target": [],
            "flow": [],
            "unit_cost": [],
        }
        for commodity, name in enumerate(names):
            for edge, (u, v) in enumerate(uv.T.tolist()):
                for column, value in zip(
                    data,
                    [
                        name,
                        edge,
                        graph._labels[u],
                        graph._labels[v],
                        float(vector[commodity * edges + edge]),
                        float(c[commodity, edge]),
                    ],
                    strict=True,
                ):
                    data[column].append(value)
        frame = as_frame(pd.DataFrame(data))
        frame.attrs.update(kind="cost_flow", device="cpu", precision="float64", sampled=False)
        return frame

    certificate = {
        "type": "primal-dual optimality"
        if status == "optimal"
        else "Farkas infeasibility"
        if status == "infeasible"
        else "feasible point and improving ray",
        "constraint_order": "commodity-major node balances, then finite-edge capacities in coalesced order",
        "node_order": list(graph._labels),
        "commodity_order": names,
        "finite_capacity_edge_ids": finite_edges.tolist(),
        "verified": True,
        "tolerance": "1e-7 absolute primal/ray residual; 1e-7 relative objective/reduced-cost scale",
    }
    if dual is not None:
        certificate["dual"] = dual.tolist()
        certificate["dual_objective"] = float(b @ dual)
    if status == "optimal":
        certificate["primal_residual_max"] = float((a @ point - b).abs().max())
        certificate["reduced_cost_min"] = float((objective - a.T @ dual).min())
        certificate["duality_gap"] = float(objective @ point - b @ dual)
    if ray is not None:
        certificate["improving_ray"] = flow_frame(ray)
        certificate["objective_direction"] = float(objective @ ray)
    metadata = {
        "algorithm": "native two-phase simplex with Bland pivots",
        "device": "cpu",
        "precision": "float64",
        "sampled": False,
        "domain": domain,
        "commodities": k,
        "certified": True,
        "work_used": work.used,
        "work_unit": "admitted tableau cells plus twice current tableau cells per pivot",
        "max_work": max_work,
        "pivots": solver.pivots,
        "max_pivots": max_pivots,
        "tableau_entries_admitted": planned,
        "max_entries": max_entries,
        "capacity_semantics": "sum of commodity flows per directed coalesced edge",
        "negative_cycles": "finite-capacity circulation optimized; unlimited improving circulation reported unbounded",
    }
    return CostFlowResult(
        status=status,
        objective=float(objective @ point) if status == "optimal" else None,
        flows=flow_frame(point) if point is not None else None,
        certificate=certificate,
        metadata=metadata,
    )
