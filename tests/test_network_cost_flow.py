import itertools
import random

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import linprog

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def graph(edges):
    return oe.network(
        pd.DataFrame(edges, columns=["source", "target", "cap"]), directed=True, weight="cap"
    )


def standard_form(result, g, demands, costs, capacities=None):
    names = list(demands)
    labels = result["certificate"]["node_order"]
    uv = g._edges.indices().numpy()
    cap = g._edges.values().numpy() if capacities is None else np.asarray(capacities)
    k, n, m = len(names), len(labels), len(cap)
    finite = np.flatnonzero(np.isfinite(cap))
    a = np.zeros((k * n + len(finite), k * m + len(finite)))
    for commodity in range(k):
        for edge, (u, v) in enumerate(uv.T):
            a[commodity * n + u, commodity * m + edge] -= 1
            a[commodity * n + v, commodity * m + edge] += 1
        a[k * n + np.arange(len(finite)), commodity * m + finite] = 1
    a[k * n + np.arange(len(finite)), k * m + np.arange(len(finite))] = 1
    b = np.r_[[demands[name].get(label, 0) for name in names for label in labels], cap[finite]]
    edge_cost = np.array(
        [
            [((costs[name] if name in costs else costs)[(labels[u], labels[v])]) for u, v in uv.T]
            for name in names
        ]
    )
    c = np.r_[edge_cost.ravel(), np.zeros(len(finite))]
    return a, b, c, finite


def verify(result, g, demands, costs, capacities=None):
    a, b, c, finite = standard_form(result, g, demands, costs, capacities)
    k, m = len(demands), g.edge_count
    if result["status"] == "infeasible":
        dual = np.asarray(result["certificate"]["dual"])
        assert (a.T @ dual).max() <= 1e-7
        assert b @ dual > 0
        return
    flows = result["flows"].flow.to_numpy().reshape(k, m)
    cap = g._edges.values().numpy() if capacities is None else np.asarray(capacities)
    point = np.r_[flows.ravel(), cap[finite] - flows.sum(0)[finite]]
    assert point.min() >= -1e-7
    assert_allclose(a @ point, b, atol=1e-7)
    if result["status"] == "unbounded":
        ray = result["certificate"]["improving_ray"].flow.to_numpy().reshape(k, m)
        full_ray = np.r_[ray.ravel(), -ray.sum(0)[finite]]
        assert full_ray.min() >= -1e-7
        assert_allclose(a @ full_ray, 0, atol=1e-7)
        assert c @ full_ray < 0
        return
    dual = np.asarray(result["certificate"]["dual"])
    assert (c - a.T @ dual).min() >= -1e-7
    assert_allclose(c @ point, b @ dual, atol=1e-7)
    assert_allclose(result["objective"], c @ point, atol=1e-7)


@pytest.mark.parametrize("seed", range(24))
def test_integer_min_cost_against_every_feasible_small_flow(seed):
    rng = random.Random(seed)
    edges = [(0, 1, rng.randint(1, 2)), (0, 2, 2), (1, 2, 2), (2, 0, 1), (1, 1, 1)]
    costs = {(u, v): rng.randint(-3, 5) for u, v, _ in edges}
    demands = {0: -2, 2: 2}
    g = graph(edges)
    expected = []
    for flow in itertools.product(*[range(cap + 1) for _, _, cap in edges]):
        balance = np.zeros(3)
        for (u, v, _), value in zip(edges, flow, strict=True):
            balance[u] -= value
            balance[v] += value
        if np.array_equal(balance, [-2, 0, 2]):
            expected.append(
                sum(value * costs[u, v] for (u, v, _), value in zip(edges, flow, strict=True))
            )
    result = g.min_cost_flow(demands, costs, domain="integral")
    assert result["status"] == "optimal" and result["objective"] == min(expected)
    assert_allclose(result["flows"].flow, result["flows"].flow.round(), atol=1e-12)
    verify(result, g, {"flow": demands}, costs)


@pytest.mark.parametrize("seed", range(15))
def test_fractional_multicommodity_full_lp_against_independent_highs(seed):
    rng = np.random.default_rng(seed)
    edges = [(0, 1, 1.5), (0, 2, 2.0), (1, 2, 2.0), (1, 3, 1.2), (2, 3, 2.3), (3, 0, 0.7)]
    g = graph(edges)
    costs = {(u, v): float(rng.uniform(-2, 4)) for u, v, _ in edges}
    demands = {"a": {0: -1.6, 3: 1.6}, "b": {1: -0.8, 3: 0.8}}
    result = g.multicommodity_flow(demands, costs)
    a, b, c, _ = standard_form(result, g, demands, costs)
    expected = linprog(c, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    assert expected.success
    assert_allclose(result["objective"], expected.fun, atol=1e-7)
    verify(result, g, demands, costs)


def test_disconnected_farkas_imbalanced_demand_and_shared_capacity_infeasibility():
    g = graph([(0, 1, 1), (2, 3, 2)])
    costs = {(0, 1): 1.0, (2, 3): 1.0}
    for demand in [{0: -1, 3: 1}, {0: -1, 1: 2}, {0: -2, 1: 2}]:
        result = g.min_cost_flow(demand, costs)
        assert result["status"] == "infeasible"
        verify(result, g, {"flow": demand}, costs)
    demand = {"a": {0: -0.6, 1: 0.6}, "b": {0: -0.6, 1: 0.6}}
    result = g.multicommodity_flow(demand, costs)
    assert result["status"] == "infeasible"
    verify(result, g, demand, costs)


def test_finite_negative_cycle_optimum_and_unlimited_negative_cycle_ray():
    g = graph([(0, 1, 2), (1, 0, 2), (1, 2, 1)])
    costs = {(0, 1): -3.0, (1, 0): 1.0, (1, 2): 0.0}
    demand = {0: -1, 2: 1}
    finite = g.min_cost_flow(demand, costs)
    assert finite["objective"] == -5
    verify(finite, g, {"flow": demand}, costs)
    unlimited = g.min_cost_flow(demand, costs, capacities=[np.inf, np.inf, 1])
    # Coalesced order is 0->1, 1->0, 1->2.
    assert unlimited["status"] == "unbounded"
    verify(unlimited, g, {"flow": demand}, costs, [np.inf, np.inf, 1])


def test_flow_domains_budgets_and_existing_max_flow_preserved():
    g = graph([(0, 1, 2), (1, 2, 2), (0, 2, 1)])
    costs = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 4.0}
    maximum = g.max_flow(0, 2)
    for kw in [
        dict(max_work=1),
        dict(max_entries=1),
        dict(max_pivots=1),
        dict(domain="integral", capacities=[0.5, 1, 2]),
    ]:
        with pytest.raises(AnalysisError):
            g.min_cost_flow({0: -2, 2: 2}, costs, **kw)
    with pytest.raises(AnalysisError):
        g.multicommodity_flow({"a": {0: -1, 2: 1}, "b": {}}, costs, domain="integral")
    assert g.max_flow(0, 2)["value"] == maximum["value"] == 3
    assert "tabular" in g.min_cost_flow({0: -2, 2: 2}, costs).to_latex()


def test_commodity_specific_costs_and_distinct_integer_string_node_ids():
    g = graph([(1, "1", 2), (1, "x", 2), ("x", "1", 2)])
    demand = {"cheap_direct": {1: -1, "1": 1}, "cheap_via": {1: -1, "1": 1}}
    costs = {
        "cheap_direct": {(1, "1"): 1.0, (1, "x"): 3.0, ("x", "1"): 3.0},
        "cheap_via": {(1, "1"): 6.0, (1, "x"): 1.0, ("x", "1"): 1.0},
    }
    result = g.multicommodity_flow(demand, costs)
    assert result["objective"] == 3
    verify(result, g, demand, costs)
    with pytest.raises(AnalysisError) as exc:
        g.min_cost_flow({1: -1, "1": 1}, {(1, "1"): 1e-12, (1, "x"): 1.0, ("x", "1"): 1.0})
    assert exc.value.code == "network_precision_unsupported"
