"""Independent Fraction, elimination and min-plus oracles for signed graphs."""
from fractions import Fraction
from itertools import product
import math
import random

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def records(edges):
    return [{"source": u, "target": v, "weight": w} for u, v, w in edges]


def exact_modularity(edges, groups, directed, resolution=1):
    """Definition evaluated with exact rational arithmetic, no production helper."""
    matrix = {}
    for u, v, w in edges:
        key = (u, v) if directed else tuple(sorted((u, v)))
        matrix[key] = matrix.get(key, Fraction()) + Fraction(w)
    if not directed:
        matrix.update({(v, u): w for (u, v), w in list(matrix.items()) if u != v})
    po, pi, mo, mi = ({node: Fraction() for node in groups} for _ in range(4))
    inside = Fraction()
    for (u, v), w in matrix.items():
        p, m = max(w, 0), max(-w, 0)
        po[u] += p
        pi[v] += p
        mo[u] += m
        mi[v] += m
        if groups[u] == groups[v]:
            inside += w
    positive, negative = sum(po.values()), sum(mo.values())
    if not positive + negative:
        return Fraction()
    for out, inc, total, sign in ((po, pi, positive, -1), (mo, mi, negative, 1)):
        if total:
            inside += sign * resolution * sum(out[u] * inc[v] / total
                for u, v in product(groups, repeat=2) if groups[u] == groups[v])
    return inside / (positive + negative)


def eliminate(matrix, rhs):
    augmented = [list(map(Fraction, row)) + [Fraction(value)] for row, value in zip(matrix, rhs)]
    n = len(rhs)
    for k in range(n):
        pivot = next(i for i in range(k, n) if augmented[i][k])
        augmented[k], augmented[pivot] = augmented[pivot], augmented[k]
        divisor = augmented[k][k]
        augmented[k] = [value / divisor for value in augmented[k]]
        for i in range(n):
            if i != k:
                factor = augmented[i][k]
                augmented[i] = [a - factor*b for a, b in zip(augmented[i], augmented[k])]
    return [float(row[-1]) for row in augmented]


def floyd(edges, n):
    costs = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        costs[i][i] = 0
    for i, j, weight in edges:
        costs[i][j] = min(costs[i][j], weight)
    for k, i, j in product(range(n), repeat=3):
        costs[i][j] = min(costs[i][j], costs[i][k] + costs[k][j])
    return costs


def test_typed_identity_cancellation_zero_missing_and_channel_strength():
    edges = [(1, "1", 3), (1, "1", -1), ("1", 1, -4), (1, 1, -2),
             ("cancel", 1, 5), ("cancel", 1, -5), ("zero", 1, 0)]
    g = oe.signed_network(records(edges), directed=True, nodes=["isolate"], batch_rows=2)
    result = g.signed_strength().set_index("node")
    assert g.node_count == 5 and g.edge_count == 3
    assert result.loc[1, "positive_out_strength"] == 2
    assert result.loc[1, "negative_out_strength"] == 2
    assert result.loc[1, "negative_in_strength"] == 6
    assert result.loc["1", "negative_out_strength"] == 4
    assert result.loc["1", "positive_in_strength"] == 2
    assert result.loc["cancel", "absolute_out_strength"] == 0
    assert result.loc["zero", "absolute_in_strength"] == 0
    assert g.metadata["cancelled_edge_pairs_dropped"] == 1
    assert g.metadata["duplicate_edge_rows_aggregated"] == 2
    assert g.metadata["negative_edge_rows"] == 4
    assert g.metadata["zero_weight_rows_dropped"] == 1
    assert g.metadata["sampled"] is False
    assert result.attrs["kind"] == "signed_network_strength"
    with pytest.raises(AnalysisError, match="missing"):
        oe.signed_network(records([(None, 1, 2)]))
    dropped = oe.signed_network(records([(None, 1, 2), ("x", "y", 0)]), missing="drop")
    assert dropped.node_count == 2 and dropped.metadata["missing_rows_dropped"] == 1


@pytest.mark.parametrize("directed", [False, True])
def test_strength_self_loops_once_and_modularity_fraction_definition(directed):
    edges = [(0, 1, 3), (1, 2, -2), (2, 3, 4), (3, 0, -1), (0, 0, -3), (3, 3, 2)]
    g = oe.signed_network(records(edges), directed=directed, nodes=range(5))
    strength = g.signed_strength().set_index("node")
    assert strength.loc[0, "negative_out_strength" if directed else "negative_strength"] == (3 if directed else 4)
    for labels in product(range(2), repeat=5):
        groups = dict(enumerate(labels))
        assert g.signed_modularity(groups) == pytest.approx(float(exact_modularity(edges, groups, directed)), abs=1e-14)
    assert g.signed_modularity([0] * 5) == pytest.approx(0, abs=1e-14)
    reverse = oe.signed_network(records([(u, v, -w) for u, v, w in edges]), directed=directed, nodes=range(5))
    assert reverse.signed_modularity([0, 0, 1, 1, 2]) == pytest.approx(-g.signed_modularity([0, 0, 1, 1, 2]))


@pytest.mark.parametrize("weights", [[1, 2, 3], [-1, -2, -3], [0, 0, 0]])
def test_one_or_zero_channels(weights):
    edges = [(0, 1, weights[0]), (1, 2, weights[1]), (0, 0, weights[2])]
    g = oe.signed_network(records(edges), nodes=range(4))
    groups = {0: 0, 1: 0, 2: 1, 3: 2}
    assert g.signed_modularity(groups) == pytest.approx(float(exact_modularity(edges, groups, False)))
    assert g.signed_communities().attrs["converged"]


@pytest.mark.parametrize("directed", [False, True])
def test_katz_against_independent_fraction_elimination(directed):
    edges = [(0, 1, -6), (1, 2, 2), (2, 0, 1), (2, 2, -1)]
    g = oe.signed_network(records(edges), directed=directed, nodes=range(4))
    alpha = Fraction(1, 10)
    matrix = [[Fraction(i == j) for j in range(4)] for i in range(4)]
    for i, j, weight in edges:
        matrix[j][i] -= alpha * weight
        if not directed and i != j:
            matrix[i][j] -= alpha * weight
    expected = eliminate(matrix, [1]*4)
    result = g.signed_katz(float(alpha), tol=1e-13)
    assert result.signed_katz.tolist() == pytest.approx(expected, abs=2e-12)
    actual = result.signed_katz.tolist()
    residual = max(abs(sum(float(matrix[i][j])*actual[j] for j in range(4)) - 1) for i in range(4))
    assert residual == pytest.approx(result.attrs["residual_linf"], abs=1e-14)
    assert result.attrs["error_bound_linf"] >= max(abs(a-b) for a, b in zip(actual, expected)) - 1e-14
    # Strong repulsion may produce a negative score; it must never be clipped.
    repulsive = oe.signed_network(records([(i, 10, 1) for i in range(10)] + [(10, 11, -10)]), directed=True)
    assert repulsive.signed_katz(.09).signed_katz.iloc[-1] == pytest.approx(-.71)
    assert g.signed_katz(0).signed_katz.tolist() == [1]*4
    with pytest.raises(AnalysisError) as error:
        g.signed_katz(1)
    assert error.value.code == "network_signed_contraction"
    with pytest.raises(AnalysisError) as error:
        g.signed_katz(.1, tol=1e-16, max_iter=1)
    assert error.value.code == "network_not_converged"


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_communities_stationary_against_independent_modularity(directed, seed):
    rng = random.Random(seed)
    edges = [(u, v, rng.choice([-3, -1, 0, 2, 4])) for u in range(5) for v in range(u if not directed else 0, 5)]
    g = oe.signed_network(records(edges), directed=directed, nodes=range(6))
    result = g.signed_communities(tol=1e-13)
    groups = dict(zip(result.node, result.community))
    expected = float(exact_modularity(edges, groups, directed))
    assert result.attrs["signed_modularity"] == pytest.approx(expected, abs=1e-13)
    assert all(b + 1e-13 >= a for a, b in zip(result.attrs["objective_trace"], result.attrs["objective_trace"][1:]))
    for node in groups:
        for group in range(7):
            moved = dict(groups)
            moved[node] = group
            assert float(exact_modularity(edges, moved, directed)) <= expected + 1e-12


def test_planted_communities_global_small_fixture_and_determinism():
    edges = [(i, j, 2 if i//2 == j//2 else -1) for i in range(4) for j in range(i+1, 4)]
    g = oe.signed_network(records(edges), nodes=range(4))
    result = g.signed_communities()
    groups = dict(zip(result.node, result.community))
    optimum = max(float(exact_modularity(edges, dict(enumerate(labels)), False)) for labels in product(range(4), repeat=4))
    assert result.attrs["signed_modularity"] == pytest.approx(optimum)
    reverse = oe.signed_network(records(list(reversed(edges))), nodes=list(reversed(range(4))))
    assert dict(zip(reverse.signed_communities().node, reverse.signed_communities().community)) == groups
    with pytest.raises(AnalysisError) as error:
        g.signed_communities(max_sweeps=1)
    assert error.value.code == "network_not_converged"


@pytest.mark.parametrize("seed", range(15))
def test_negative_dag_paths_against_min_plus_oracle(seed):
    rng = random.Random(seed)
    edges = [(i, j, rng.randrange(-8, 9)) for i in range(7) for j in range(i+1, 7) if rng.random() < .5]
    g = oe.signed_network(records(edges), directed=True, nodes=range(8))
    expected = floyd([(i, j, w) for i, j, w in edges if w], 8)[0]
    result = g.signed_shortest_paths(0)
    for actual, reachable, value in zip(result.distance, result.reachable, expected):
        assert reachable == math.isfinite(value)
        assert actual == value if reachable else math.isnan(actual)
    assert result.attrs["reachable_negative_cycle"] is False


@pytest.mark.parametrize("edges,directed", [([(0, 1, -1)], False), ([(0, 0, -1)], True),
    ([(0, 1, 3), (1, 2, -4), (2, 0, 0.5)], True)])
def test_reachable_negative_cycles_refused(edges, directed):
    g = oe.signed_network(records(edges), directed=directed, nodes=[0, 1, 2])
    with pytest.raises(AnalysisError) as error:
        g.signed_shortest_paths(0)
    assert error.value.code == "network_negative_cycle"


def test_unreachable_negative_cycle_and_typed_source():
    g = oe.signed_network(records([(1, "1", -3), ("x", "y", -1), ("y", "x", -2)]), directed=True)
    assert g.signed_shortest_paths(1).distance.tolist()[:2] == [0, -3]
    assert g.signed_shortest_paths("1").reachable.tolist() == [False, True, False, False]
    with pytest.raises(AnalysisError):
        g.signed_shortest_paths(1.)


@pytest.mark.parametrize("method,args", [("pagerank", ()), ("max_flow", (0, 1)), ("min_cut", (0, 1)),
    ("poisson_block_model", (2,)), ("degree_corrected_block_model", (2,)), ("block_model", (2,)),
    ("communities", ()), ("katz", ()), ("shortest_paths", (0,))])
def test_legacy_nonnegative_methods_explicitly_refused(method, args):
    g = oe.signed_network(records([(0, 1, -1)]))
    with pytest.raises(AnalysisError) as error:
        getattr(g, method)(*args)
    assert error.value.code == "network_signed_method"
    # SignedNetwork is deliberately not a Network subtype passed into other solvers.
    assert not isinstance(g, oe.Network)


@pytest.mark.parametrize("method,args", [("signed_katz", (.1,)), ("signed_communities", ()),
    ("signed_modularity", ([0, 1, 2],)), ("signed_shortest_paths", (0,))])
def test_work_budget_refuses_incomplete_results(method, args):
    g = oe.signed_network(records([(0, 1, 1), (1, 2, -2)]), directed=True)
    with pytest.raises(AnalysisError) as error:
        getattr(g, method)(*args, max_work=1)
    assert error.value.code == "network_work_budget"


def test_memory_budget_and_generator_closed_without_truncation():
    closed = []
    def rows():
        try:
            for i in range(10000):
                yield {"source": i, "target": i+1, "weight": -1}
        finally:
            closed.append(True)
    with pytest.raises(AnalysisError) as error:
        oe.signed_network(rows(), max_memory_mb=.02, batch_rows=1)
    assert error.value.code == "network_memory_budget"
    assert closed == [True]
    g = oe.signed_network(records([(0, 1, -1)]), max_memory_mb=.02, batch_rows=1, nodes=range(5))
    g._graph._budget.limit = g._graph._base_bytes + 4096
    with pytest.raises(AnalysisError) as error:
        g.signed_communities()
    assert error.value.code == "network_memory_budget"


@pytest.mark.parametrize("value", [True, float("inf"), -float("inf"), "-1"])
def test_invalid_weights(value):
    with pytest.raises(AnalysisError):
        oe.signed_network(records([(0, 1, value)]))


def test_precision_overflow_normalization_and_path_addition_refused():
    with pytest.raises(AnalysisError) as error:
        oe.signed_network(records([(0, 1, -1e308), (0, 1, -1e308)]))
    assert error.value.code == "network_precision"
    g = oe.signed_network(records([(0, 1, 1e308), (0, 2, -1e308)]), directed=True)
    with pytest.raises(AnalysisError):
        g.signed_strength()
    tiny = oe.signed_network(records([(0, 1, 1e308), (1, 2, -1e-300)]), directed=True)
    with pytest.raises(AnalysisError):
        tiny.signed_modularity([0, 1, 2])
    with pytest.raises(AnalysisError):
        tiny.signed_shortest_paths(0)


def test_empty_unweighted_input_membership_and_cpu_context():
    empty = oe.signed_network([], nodes=[1, "1"])
    assert empty.signed_strength().absolute_strength.tolist() == [0, 0]
    assert empty.signed_modularity({1: "a", "1": "b"}) == 0
    assert empty.signed_katz(.2).signed_katz.tolist() == [1, 1]
    assert empty.signed_shortest_paths(1).reachable.tolist() == [True, False]
    assert oe.signed_network([], nodes=[]).signed_katz(0).empty
    with pytest.raises(AnalysisError):
        empty.signed_modularity({1: 0})
    with torch.device("meta"):
        g = oe.signed_network([{"source": 0, "target": 1}], weight=None)
        assert g.signed_strength().positive_strength.tolist() == [1, 1]
        assert g.signed_katz(.1).attrs["network"]["device"] == "cpu"


def test_dataset_snapshot_and_nonnegative_api_regression(tmp_path):
    frame = pd.DataFrame(records([(0, 1, -1), (1, 2, 2)]))
    path = tmp_path/'signed.parquet'
    frame.to_parquet(path)
    graph = oe.signed_network(oe.scan(path), batch_rows=1, directed=True)
    assert graph.signed_shortest_paths(0).distance.tolist() == [0, -1, 1]
    frame.to_parquet(path)
    assert graph.signed_strength().negative_out_strength.tolist() == [1, 0, 0]
    with pytest.raises(AnalysisError):
        oe.network(frame, weight="weight")
    positive = oe.network(records([(0, 1, 3), (1, 2, 2)]), weight="weight")
    assert positive.degree().strength.tolist() == [3, 5, 2]
    assert positive.shortest_paths(0).distance.tolist() == [0, 3, 5]


def test_katz_never_publishes_nonfinite_diagnostics_under_extreme_options():
    graph = oe.signed_network(records([(0, 0, -1)]), directed=True)
    with pytest.raises(AnalysisError) as error:
        graph.signed_katz(math.nextafter(1., 0.), beta=1e307, tol=1e308, max_iter=1)
    assert error.value.code == "network_precision"
