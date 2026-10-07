"""Independent actor-choice, Kolmogorov transition and recovery tests."""

import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_saom import _scaffold, choices, generator_matrix
from openecon._network_model_common import Work


def graph(edges=(), nodes=(0, 1, 2)):
    return oe.network([{"source": a, "target": b} for a, b in edges], nodes=nodes, directed=True)


def oracle_actor(a, i):
    n = len(a)
    return [
        sum(a[i]),
        sum(a[i][j] * a[j][i] for j in range(n)),
        sum(a[i][j] * a[j][k] * a[i][k] for j in range(n) for k in range(n)),
    ]


@pytest.mark.parametrize("state", range(64))
def test_actor_choices_and_generator_match_independent_python(state):
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    a = [[0.0] * 3 for _ in range(3)]
    for bit, (i, j) in enumerate(pairs):
        a[i][j] = float(bool(state & (1 << bit)))
    for actor in range(3):
        values, targets = choices(
            torch.tensor(a, dtype=torch.float64),
            actor,
            ["outdegree", "reciprocity", "transitive_triplets"],
        )
        expected = [oracle_actor(a, actor)]
        for j in targets:
            b = [row.copy() for row in a]
            b[actor][j] = 1 - b[actor][j]
            expected.append(oracle_actor(b, actor))
        assert values.tolist() == expected


def test_exact_transition_nonnegative_stochastic_and_independent_expm():
    g = graph()
    work = Work(g, workspace=1000000, max_work=10000000, timeout=20)
    features, destinations, pairs = _scaffold(3, ["outdegree", "reciprocity"], work)
    theta = torch.tensor([math.log(0.7), -0.8, 0.5], dtype=torch.float64)
    q = generator_matrix(theta, features, destinations)
    expected = np.zeros((64, 64))
    for state in range(64):
        a = [[0.0] * 3 for _ in range(3)]
        for bit, (i, j) in enumerate(pairs):
            a[i][j] = float(bool(state & (1 << bit)))
        for actor in range(3):
            scores = [sum(x * t for x, t in zip(oracle_actor(a, actor)[:2], theta[1:].tolist()))]
            for j in range(3):
                if j == actor:
                    continue
                b = [row.copy() for row in a]
                b[actor][j] = 1 - b[actor][j]
                scores.append(
                    sum(x * t for x, t in zip(oracle_actor(b, actor)[:2], theta[1:].tolist()))
                )
            probabilities = np.exp(scores) / np.exp(scores).sum()
            for choice, j in enumerate(k for k in range(3) if k != actor):
                expected[state, state ^ (1 << pairs.index((actor, j)))] = (
                    0.7 * probabilities[choice + 1]
                )
        expected[state, state] = -expected[state].sum()
    assert q.numpy() == pytest.approx(expected)
    transition = torch.matrix_exp(q * 1.3)
    assert transition.numpy() == pytest.approx(expm(expected * 1.3), abs=2e-13)
    assert transition.sum(1).tolist() == pytest.approx([1] * 64)
    assert float(transition.min()) >= 0


def test_two_actor_parameter_recovery_seed_and_typed_identity():
    g = graph(nodes=(1, "1"))
    times = [0.0, 0.6, 1.8, 3.0]
    panels = [
        oe.simulate_saom(g, times, rate=1.1, theta=[-0.7], seed=seed)["panel"]
        for seed in range(240)
    ]
    fit = oe.saom(panels, times, max_iter=70)
    assert fit["metadata"]["converged"]
    assert fit["metadata"]["opportunity_rate"] == pytest.approx(1.1, abs=0.25)
    assert fit["coefficients"].coefficient.iloc[1] == pytest.approx(-0.7, abs=0.3)
    assert fit["metadata"]["actor_ids"] == [1, "1"]
    assert fit["covariance"] is not None
    assert "tabular" in fit.to_latex()
    a = oe.simulate_saom(g, times, seed=13)
    b = oe.simulate_saom(g, times, seed=13)
    assert a["diagnostics"].equals(b["diagnostics"])
    assert [x.edges().to_dict("records") for x in a["panel"]._graphs] == [
        x.edges().to_dict("records") for x in b["panel"]._graphs
    ]


def test_missing_actor_and_panel_order_rejected():
    changed = oe.network_snapshots({0: graph(), 1: graph(nodes=(0, 1))}, ordered=True)
    with pytest.raises(AnalysisError, match="same fully observed"):
        oe.saom(changed, [0, 1])
    unordered = oe.network_snapshots({0: graph(), 1: graph()})
    with pytest.raises(AnalysisError, match="ordered"):
        oe.saom(unordered, [0, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"times": [0, 0]},
        {"terms": ["behavior"]},
        {"rate": 0},
        {"max_work": 1},
        {"max_events": 1, "rate": 100},
        {"cancelled": lambda: True},
    ],
)
def test_simulation_fail_closed(kwargs):
    kwargs = {"times": [0, 1], **kwargs}
    with pytest.raises(AnalysisError):
        oe.simulate_saom(graph(), **kwargs)


def test_exact_state_budget_does_not_silently_substitute_mom_estimator():
    panel = oe.network_snapshots({0: graph(nodes=range(4)), 1: graph(nodes=range(4))}, ordered=True)
    with pytest.raises(AnalysisError, match="max_states"):
        oe.saom(panel, [0, 1])
