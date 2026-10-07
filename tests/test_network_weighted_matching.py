from fractions import Fraction
from itertools import combinations
import random

import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def exhaustive(edges, objective):
    best = None
    for mask in range(1 << len(edges)):
        chosen = [e for at, e in enumerate(edges) if mask & (1 << at)]
        ends = [u for u, v, w in chosen] + [v for u, v, w in chosen]
        if len(ends) != len(set(ends)):
            continue
        count, weight = (
            len(chosen),
            sum((Fraction.from_float(w) for u, v, w in chosen), Fraction(0)),
        )
        score = (
            (weight,)
            if objective == "weight"
            else ((count,) if objective == "cardinality" else (count, weight))
        )
        if best is None or score > best:
            best = score
    return best


def snapshot(edges, nodes=range(6)):
    return oe.signed_network([dict(source=u, target=v, weight=w) for u, v, w in edges], nodes=nodes)


def result_score(result, objective):
    count, weight = len(result["pairs"]), Fraction(result["metadata"]["exact_weight"])
    return (
        (weight,)
        if objective == "weight"
        else ((count,) if objective == "cardinality" else (count, weight))
    )


@pytest.mark.parametrize("objective", ["weight", "cardinality", "cardinality_weight"])
@pytest.mark.parametrize("seed", range(16))
def test_general_matching_matches_exhaustive_edge_subset_optimum(objective, seed):
    rng = random.Random(seed)
    edges = [
        (u, v, float(rng.choice([-4, -1, 1, 2, 7])))
        for u, v in combinations(range(6), 2)
        if rng.random() < 0.5
    ]
    result = snapshot(edges).general_matching(objective=objective)
    assert result_score(result, objective) == exhaustive(edges, objective)
    matched = [*result["pairs"].source, *result["pairs"].target]
    assert len(matched) == len(set(matched))
    assert set(matched) | set(result["unmatched"].node) == set(range(6))
    assert result["metadata"]["certificate"]["exhaustive_recurrence"] is True
    assert result["metadata"]["work_used"] <= result["metadata"]["max_work"]


@pytest.mark.parametrize("objective", ["weight", "cardinality", "cardinality_weight"])
@pytest.mark.parametrize("seed", range(16))
def test_weighted_bipartite_assignment_matches_exhaustive_optimum_and_dual(objective, seed):
    rng = random.Random(seed)
    edges = [
        (u, v, float(rng.choice([-5, -2, 1, 3, 9])))
        for u in range(3)
        for v in range(3, 6)
        if rng.random() < 0.7
    ]
    result = snapshot(edges).weighted_assignment(
        {u: int(u >= 3) for u in range(6)}, objective=objective
    )
    assert result_score(result, objective) == exhaustive(edges, objective)
    certificate = result["metadata"]["certificate"]
    pu, pv = (
        [Fraction(v) for v in certificate["row_potentials"]],
        [Fraction(v) for v in certificate["column_potentials"]],
    )
    values = {(u, v): Fraction.from_float(w) for u, v, w in edges}
    bound = sum((abs(w) for w in values.values()), Fraction(0))
    bonus = 2 * bound + 1
    forbidden = 5 * bonus + 2 * bound + 1
    for u in range(3):
        for j in range(6):
            weight = values.get((u, j + 3)) if j < 3 else None
            cost = (
                Fraction(0)
                if j >= 3
                else (
                    forbidden
                    if weight is None
                    else -(
                        weight
                        if objective == "weight"
                        else (Fraction(1) if objective == "cardinality" else bonus + weight)
                    )
                )
            )
            assert pu[u] + pv[j] <= cost
    assert (
        sum(pu) + sum(pv)
        == Fraction(certificate["primal_exact"])
        == Fraction(certificate["dual_exact"])
    )


def test_signed_objectives_zero_unmatched_odd_cycle_and_parallel_edge_ids():
    triangle = snapshot([(0, 1, -2.0), (1, 2, -3.0), (0, 2, -1.0)], range(4))
    assert triangle.general_matching()["pairs"].empty
    assert triangle.general_matching(objective="cardinality_weight")["weight"] == -1
    with pytest.raises(AnalysisError, match="not bipartite"):
        triangle.weighted_assignment()
    m = oe.multigraph(
        [
            dict(edge_id=1, source=0, target=1, w=0.0),
            dict(edge_id="1", source=0, target=1, w=4.0),
            dict(edge_id="other", source=2, target=3, w=2.0),
        ],
        weight="w",
        nodes=range(5),
    )
    for method in (m.general_matching, m.weighted_assignment):
        result = method(objective="cardinality_weight")
        assert set(result["pairs"].edge_id) == {"1", "other"}
        assert result["weight"] == 6 and result["unmatched"].node.tolist() == [4]
    zero = oe.multigraph([dict(edge_id="zero", source=0, target=1, w=0.0)], weight="w")
    assert zero.general_matching(objective="cardinality")["pairs"].edge_id.tolist() == ["zero"]


def test_close_float_weights_have_exact_objective_certificate_and_typed_unmatched():
    g = snapshot(
        [(1, "a", 1e16), (1, "b", 1e16), ("a", "c", 1.0), ("b", "c", 2.0)], [1, "1", "a", "b", "c"]
    )
    result = g.general_matching()
    assert Fraction(result["metadata"]["exact_weight"]) == Fraction(10**16 + 2)
    assert "1" in result["unmatched"].node.tolist()


@pytest.mark.parametrize(
    "method,options",
    [
        ("general_matching", {"max_states": 1}),
        ("general_matching", {"max_component_nodes": 2}),
        ("general_matching", {"max_work": 1}),
        ("weighted_assignment", {"max_matrix_entries": 1}),
        ("weighted_assignment", {"max_work": 1}),
    ],
)
def test_matching_budgets_do_not_return_partial_optimum(method, options):
    g = snapshot([(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)], range(4))
    with pytest.raises(AnalysisError):
        getattr(g, method)(**options)


@pytest.mark.parametrize("method", ["weighted_assignment", "general_matching"])
def test_loops_and_directed_are_explicit_and_original_cardinality_matching_unchanged(method):
    with pytest.raises(AnalysisError, match="Self-loops"):
        getattr(oe.network([dict(source=0, target=0)]), method)()
    with pytest.raises(AnalysisError, match="undirected"):
        getattr(oe.network([dict(source=0, target=1)], directed=True), method)()
    old = oe.network([dict(source=0, target=2), dict(source=1, target=2), dict(source=1, target=3)])
    before = old.maximum_matching()
    getattr(old, method)(objective="cardinality")
    after = old.maximum_matching()
    assert before.to_dict("records") == after.to_dict("records")
