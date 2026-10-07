"""Independent likelihood, simplex, sample-space and holdout leakage checks."""

from itertools import combinations
import math
import random

import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def graph(n=4):
    return oe.network([], nodes=range(n))


def test_gaussian_profile_oracle_negative_zero_and_variance_floor():
    pairs = list(combinations(range(4), 2))
    y = [-2.0, 0.0, 0.5, 1.0, 1.5, 0.0]
    df = pd.DataFrame(
        [(a, b, value) for (a, b), value in zip(pairs, y)], columns=["source", "target", "value"]
    )
    fit = oe.gaussian_block_model(
        graph(), 2, df, initial=[0, 0, 1, 1], starts=1, max_iter=0, variance_floor=0.2
    )
    groups = [0, 0, 1, 1]
    cells = {}
    for (a, b), value in zip(pairs, y):
        cells.setdefault(tuple(sorted((groups[a], groups[b]))), []).append(value)
    likelihood = 0
    for row in fit["blocks"].itertuples(index=False):
        values = cells[row.source_block, row.target_block]
        mean = sum(values) / len(values)
        variance = max(0.2, sum((v - mean) ** 2 for v in values) / len(values))
        assert row.mean == pytest.approx(mean)
        assert row.variance == pytest.approx(variance)
        likelihood += sum(
            -0.5 * (math.log(2 * math.pi * variance) + (v - mean) ** 2 / variance) for v in values
        )
    assert fit["metadata"]["objective"] == pytest.approx(likelihood)
    assert fit["evaluation"].log_likelihood[0] == pytest.approx(likelihood)
    assert fit.predict([(0, 1)])["mean"][0] == -2
    assert "tabular" in fit.to_latex()


def test_gaussian_holdout_does_not_train_or_identify_empty_block():
    df = pd.DataFrame(
        [(0, 1, -2, "train"), (0, 2, 1, "train"), (1, 2, 0, "train"), (2, 3, 10, "test")],
        columns=["source", "target", "value", "split"],
    )
    options = dict(initial=[0, 0, 1, 1], starts=1, max_iter=0)
    fit = oe.gaussian_block_model(graph(), 2, df, **options)
    changed = df.copy()
    changed.loc[changed.split == "test", "value"] = -9000
    other = oe.gaussian_block_model(graph(), 2, changed, **options)
    assert fit["membership"].equals(other["membership"])
    assert fit["blocks"].equals(other["blocks"])
    assert fit["evaluation"].status.iloc[-1] == "unidentified training block"
    with pytest.raises(AnalysisError, match="no training"):
        fit.predict([(2, 3)])


def fixture(n=28):
    rng = random.Random(184)
    rows = []
    for a, b in combinations(range(n), 2):
        p = 0.94 if (a < n // 2) == (b < n // 2) else 0.06
        value = int(rng.random() < p)
        rows.append((a, b, value, "train" if rng.random() < 0.75 else "test"))
    return pd.DataFrame(rows, columns=["source", "target", "value", "split"])


def test_mixed_membership_likelihood_simplex_recovery_and_baseline():
    df = fixture()
    fit = oe.mixed_membership_block_model(
        graph(28), 2, df, seed=8, max_iter=250, batch_size=64, max_work=100_000_000
    )
    member = fit["membership"].drop(columns="node")
    assert member.sum(axis=1).tolist() == pytest.approx([1] * 28)
    assert member.min().min() >= 0
    assert fit["blocks"].to_numpy() == pytest.approx(fit["blocks"].to_numpy().T)
    train = df.loc[df.split == "train"]
    predicted = fit.predict(list(train[["source", "target"]].itertuples(index=False, name=None)))
    ll = sum(
        y * math.log(p) + (1 - y) * math.log1p(-p) for y, p in zip(train.value, predicted["mean"])
    )
    assert fit["metadata"]["objective"] == pytest.approx(ll, abs=1e-8)
    test = fit["evaluation"].loc[fit["evaluation"].split == "test"].iloc[0]
    assert test.log_likelihood > test.baseline_log_likelihood
    labels = member.to_numpy().argmax(1)
    correct = sum((labels[i] == labels[0]) == (i < 14) for i in range(28))
    assert correct >= 26


def test_mixed_holdout_and_topology_independent_training():
    df = fixture(8)
    changed = df.copy()
    changed.loc[changed.split == "test", "value"] = (
        1 - changed.loc[changed.split == "test", "value"]
    )
    dense = oe.network(
        [{"source": a, "target": b} for a, b in combinations(range(8), 2)], nodes=range(8)
    )
    a = oe.mixed_membership_block_model(graph(8), 2, df, max_iter=15, seed=2)
    b = oe.mixed_membership_block_model(dense, 2, changed, max_iter=15, seed=2)
    assert a["membership"].equals(b["membership"])
    assert a["blocks"].equals(b["blocks"])


@pytest.mark.parametrize(
    "rows", [[(0, 0, 1)], [(0, 1, 1), (1, 0, 0)], [(0, 4, 1)], [(0, 1, float("nan"))]]
)
def test_invalid_dyads_fail_closed(rows):
    df = pd.DataFrame(rows, columns=["source", "target", "value"])
    for function in [oe.gaussian_block_model, oe.mixed_membership_block_model]:
        with pytest.raises(AnalysisError):
            function(graph(), 2, df)


def test_binary_no_rounding_and_work_memory_stop_limits():
    df = pd.DataFrame([(0, 1, 0.5)], columns=["source", "target", "value"])
    with pytest.raises(AnalysisError, match="exactly 0 or 1"):
        oe.mixed_membership_block_model(graph(), 2, df)
    df.value = 1
    for function in [oe.gaussian_block_model, oe.mixed_membership_block_model]:
        for options in [{"max_work": 1}, {"max_observations": 0}, {"cancelled": lambda: True}]:
            with pytest.raises(AnalysisError):
                function(graph(), 2, df, **options)


def test_gaussian_group_recovery_and_holdout_baseline():
    rng = random.Random(719)
    n = 20
    df = pd.DataFrame(
        [
            (
                a,
                b,
                rng.gauss(-3 if (a < 10) == (b < 10) else 3, 0.4),
                "train" if rng.random() < 0.8 else "test",
            )
            for a, b in combinations(range(n), 2)
        ],
        columns=["source", "target", "value", "split"],
    )
    fit = oe.gaussian_block_model(
        graph(n), 2, df, seed=2, starts=12, max_iter=50, max_work=500_000_000
    )
    labels = fit["membership"].block.tolist()
    assert sum((labels[i] == labels[0]) == (i < 10) for i in range(n)) >= 19
    heldout = fit["evaluation"].loc[fit["evaluation"].split == "test"].iloc[0]
    assert heldout.log_likelihood > heldout.baseline_log_likelihood
