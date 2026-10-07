"""Enumeration and conditional-probability oracles independent of the estimator."""

from itertools import combinations
import math

import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_ergm import _change, _enumerate, statistics
from openecon._network_model_common import Work, adjacency


def graph(edges=(), n=4, **kwargs):
    return oe.network([{"source": i, "target": j} for i, j in edges], nodes=range(n), **kwargs)


def oracle_stats(n, edges):
    edges = set(map(frozenset, edges))
    degree = [sum(i in edge for edge in edges) for i in range(n)]
    triangles = sum(
        all(frozenset(pair) in edges for pair in combinations(triple, 2))
        for triple in combinations(range(n), 3)
    )
    return len(edges), sum(d * (d - 1) // 2 for d in degree), triangles


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_exact_enumeration_and_change_statistics(n):
    pairs = list(combinations(range(n), 2))
    g = graph(n=n)
    work = Work(g, workspace=100000, max_work=10_000_000, timeout=30)
    actual = _enumerate(n, ["edges", "twostars", "triangles"], work, 65536)
    expected = []
    for mask in range(2 ** len(pairs)):
        edges = [pair for bit, pair in enumerate(pairs) if mask & (1 << bit)]
        expected.append(oracle_stats(n, edges))
        a = adjacency(graph(edges, n=n))
        assert statistics(a, ["edges", "twostars", "triangles"]).tolist() == list(expected[-1])
        for i, j in pairs:
            absent = [pair for pair in edges if pair != (i, j)]
            before = oracle_stats(n, absent)
            after = oracle_stats(n, [*absent, (i, j)])
            assert _change(a, i, j, ["edges", "twostars", "triangles"]).tolist() == [
                b - a for a, b in zip(before, after)
            ]
    assert sorted(map(tuple, actual.tolist())) == sorted(expected)


def test_edges_only_exact_and_mple_match_closed_form_and_fisher():
    g = graph([(0, 1), (1, 2)], n=4)
    expected = math.log(2 / 4)
    for method in ["exact", "mple"]:
        fit = oe.ergm(g, method=method)
        meta = fit["metadata"]
        assert meta["converged"]
        assert fit["coefficients"].coefficient[0] == pytest.approx(expected, abs=1e-6)
        assert meta["objective"] == pytest.approx(2 * math.log(1 / 3) + 4 * math.log(2 / 3))
        assert "tabular" in fit.to_latex()
        if method == "exact":
            assert meta["log_partition"] == pytest.approx(6 * math.log1p(math.exp(expected)))
            assert fit["covariance"].iloc[0, 0] == pytest.approx(1 / (6 * (1 / 3) * (2 / 3)))
        else:
            assert fit["covariance"] is None


def test_dependent_exact_log_partition_and_score_from_direct_probabilities():
    g = graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    theta = [0.1, -0.15, 0.2]
    fit = oe.ergm(g, terms=["edges", "twostars", "triangles"], initial=theta, max_iter=1)
    estimated = fit["coefficients"].coefficient.tolist()
    pairs = list(combinations(range(4), 2))
    rows = [
        oracle_stats(4, [pair for i, pair in enumerate(pairs) if mask & (1 << i)])
        for mask in range(64)
    ]
    unnormalized = [math.exp(sum(a * b for a, b in zip(row, estimated))) for row in rows]
    z = sum(unnormalized)
    expectation = [sum(w * row[j] for w, row in zip(unnormalized, rows)) / z for j in range(3)]
    observed = oracle_stats(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    assert fit["metadata"]["log_partition"] == pytest.approx(math.log(z))
    assert fit["metadata"]["score"] == pytest.approx([a - b for a, b in zip(observed, expectation)])


def test_seeded_gibbs_moments_and_global_rng_preserved():
    g = graph(n=4)
    torch.manual_seed(819)
    state = torch.random.get_rng_state().clone()
    a = oe.simulate_ergm(g, [math.log(0.3 / 0.7)], draws=1500, burn_in=300, thin=8, seed=9)
    assert torch.equal(state, torch.random.get_rng_state())
    assert a["statistics"].edges.mean() == pytest.approx(6 * 0.3, abs=0.12)
    b = oe.simulate_ergm(g, [math.log(0.3 / 0.7)], draws=1500, burn_in=300, thin=8, seed=9)
    assert a["statistics"].equals(b["statistics"])
    assert a["metadata"]["chain"] == 1
    assert len(a["diagnostics"]) == 1


def test_boundary_not_reported_as_finite_converged_mle():
    assert not oe.ergm(graph())["metadata"]["converged"]
    assert not oe.ergm(graph(combinations(range(4), 2)))["metadata"]["converged"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "mcmc"},
        {"terms": ["unknown"]},
        {"terms": []},
        {"terms": ["edges", "edges"]},
        {"max_states": 1},
        {"max_work": 1},
        {"tol": -1},
        {"initial": [float("nan")]},
        {"initial": [1, 2]},
        {"cancelled": lambda: True},
    ],
)
def test_fail_closed_options_and_resources(kwargs):
    with pytest.raises(AnalysisError):
        oe.ergm(graph([(0, 1)]), **kwargs)


def test_graph_domain_and_memory_admission():
    for g in [graph([(0, 0)]), graph([(0, 1)], directed=True)]:
        with pytest.raises(AnalysisError):
            oe.ergm(g)
    with pytest.raises(AnalysisError, match="memory"):
        oe.ergm(graph(n=5, max_memory_mb=0.2))


def test_dependent_support_certificate_matches_independent_convex_hull():
    from scipy.spatial import ConvexHull
    from openecon._network_ergm import _support_boundary

    g = graph(n=5)
    work = Work(g, workspace=2_000_000, max_work=50_000_000, timeout=60)
    data = _enumerate(5, ["edges", "twostars", "triangles"], work, 65536)
    unique = torch.unique(data, dim=0)
    hull = ConvexHull(unique.numpy())
    for row in unique:
        expected = bool(
            (abs(hull.equations[:, :3] @ row.numpy() + hull.equations[:, 3]) < 1e-10).any()
        )
        assert _support_boundary(data, row, work) == expected
