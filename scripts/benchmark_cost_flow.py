"""Bounded 100-node/180-arc/three-commodity native LP scale + HiGHS oracle.

SciPy is a development oracle only. Reports the admitted domain and budgets,
not a general performance comparison against R/Stata or integer MCF solvers.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import openecon as oe
from scipy.optimize import linprog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    size = 10
    edges = []
    for row in range(size):
        for col in range(size):
            source = row * size + col
            for target in ([source + 1] if col + 1 < size else []) + (
                [source + size] if row + 1 < size else []
            ):
                edges.append(dict(source=source, target=target, capacity=10.0))
    graph = oe.network(edges, directed=True, weight="capacity")
    costs = {
        (e["source"], e["target"]): float(1 + ((e["source"] * 7 + e["target"]) % 13) / 10)
        for e in edges
    }
    demand = {"a": {0: -3.0, 99: 3.0}, "b": {1: -2.0, 99: 2.0}, "c": {10: -2.0, 99: 2.0}}
    budgets = dict(max_entries=1_000_000, max_work=20_000_000_000, max_pivots=10000)
    start = time.perf_counter()
    result = graph.multicommodity_flow(demand, costs, **budgets)
    elapsed = time.perf_counter() - start
    labels = result["certificate"]["node_order"]
    index = {label: i for i, label in enumerate(labels)}
    uv = graph._edges.indices().numpy()
    k, n, m = 3, len(labels), len(edges)
    a = np.zeros((k * n + m, k * m + m))
    for commodity in range(k):
        for edge, (u, v) in enumerate(uv.T):
            a[commodity * n + u, commodity * m + edge] -= 1
            a[commodity * n + v, commodity * m + edge] += 1
        a[k * n + np.arange(m), commodity * m + np.arange(m)] = 1
    a[k * n + np.arange(m), k * m + np.arange(m)] = 1
    b = np.zeros(k * n + m)
    b[k * n :] = 10
    for commodity, demands in enumerate(demand.values()):
        for node, value in demands.items():
            b[commodity * n + index[node]] = value
    c = np.r_[np.tile([costs[labels[u], labels[v]] for u, v in uv.T], k), np.zeros(m)]
    oracle = linprog(c, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    assert oracle.success and result["status"] == "optimal"
    assert abs(result["objective"] - oracle.fun) < 1e-7
    flows = result["flows"].flow.to_numpy().reshape(k, m)
    point = np.r_[flows.ravel(), 10 - flows.sum(0)]
    dual = np.asarray(result["certificate"]["dual"])
    assert np.max(np.abs(a @ point - b)) < 1e-7
    assert np.min(point) >= -1e-7 and np.min(c - a.T @ dual) >= -1e-7
    assert abs(c @ point - b @ dual) < 1e-7
    receipt = dict(
        platform=platform.platform(),
        nodes=n,
        arcs=m,
        commodities=k,
        domain="fractional splittable",
        native_seconds=elapsed,
        objective=result["objective"],
        independent_oracle="SciPy HiGHS development only",
        oracle_objective=oracle.fun,
        max_primal_residual=float(np.max(np.abs(a @ point - b))),
        duality_gap=float(c @ point - b @ dual),
        metadata=result["metadata"],
        thread_environment="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        no_general_R_Stata_or_integer_MCF_speed_claim=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
