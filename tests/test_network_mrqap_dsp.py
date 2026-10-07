from itertools import permutations

import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon._network_mrqap import _adjust
from test_network_mrqap import A, B, Y, graph, full_vectors


def oracle(records, draws, *, directed, include_loops, intercept, method, tested):
    """Reconstruct dense columns, project by SVD and permute the selected null side.

    This uses neither sparse residual helpers nor sufficient-product Cholesky.
    For joint tests the reference statistic comes from full vs reduced SSE.
    """
    pairs, columns = full_vectors(records, directed=directed, include_loops=include_loops)
    y = torch.tensor([float(v) for v in columns[0]], dtype=torch.float64)
    X = torch.tensor([[float(v) for v in row] for row in zip(*columns[1:])], dtype=torch.float64)
    Z = X[:, [i for i in range(X.shape[1]) if i not in tested]]
    if intercept:
        Z = torch.cat([Z, torch.ones((len(y), 1), dtype=torch.float64)], dim=1)

    def residual(value):
        return (
            value - Z @ torch.linalg.lstsq(Z, value, driver="gelsd").solution
            if Z.shape[1]
            else value
        )

    yr, xr = residual(y), residual(X[:, tested])

    def statistic(x, response):
        if len(tested) == 1:
            return float(
                torch.dot(x[:, 0], response)
                / torch.linalg.vector_norm(x[:, 0])
                / torch.linalg.vector_norm(response)
            )
        fitted = x @ torch.linalg.lstsq(x, response, driver="gelsd").solution
        return 1 - float(torch.sum((response - fitted) ** 2) / torch.sum(response**2))

    observed = statistic(xr, yr)
    values = []
    for order in draws:
        source = yr[:, None] if method == "freedman_lane" else xr
        lookup = {}
        for (u, v), row in zip(pairs, source):
            u, v = order[u], order[v]
            if not directed and u > v:
                u, v = v, u
            lookup[u, v] = row
        permuted = residual(torch.stack([lookup[pair] for pair in pairs]))
        values.append(
            statistic(xr, permuted[:, 0]) if method == "freedman_lane" else statistic(permuted, yr)
        )
    return observed, values


def fixed_draws(monkeypatch):
    draws = list(permutations(range(4)))
    iterator = iter(draws)
    monkeypatch.setattr(
        torch, "randperm", lambda n, **kw: torch.tensor(next(iterator), dtype=torch.int64)
    )
    return draws


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("loops", [False, True])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_dsp_coefficient_permutations_match_independent_dense_oracle(
    directed, loops, intercept, alternative, monkeypatch
):
    draws = fixed_draws(monkeypatch)
    graphs = [graph(rows, directed=directed) for rows in [Y, A, B]]
    actual = graphs[0].qap_regression(
        {"a": graphs[1], "b": graphs[2]},
        method="dsp",
        permutations=24,
        include_loops=loops,
        intercept=intercept,
        alternative=alternative,
    )
    for j in range(2):
        observed, values = oracle(
            [Y, A, B],
            draws,
            directed=directed,
            include_loops=loops,
            intercept=intercept,
            method="dsp",
            tested=[j],
        )
        count = sum(
            (abs(v) >= abs(observed) - 1e-12)
            if alternative == "two-sided"
            else (v >= observed - 1e-12 if alternative == "greater" else v <= observed + 1e-12)
            for v in values
        )
        assert actual.iloc[j].statistic == pytest.approx(observed, abs=1e-12)
        assert actual.iloc[j].extreme_permutations == count
        assert actual.iloc[j].pvalue == (count + 1) / 25
    assert actual.attrs["test_method"] == "dsp"
    assert "focal-residual" in actual.attrs["null"]


@pytest.mark.parametrize("method", ["freedman_lane", "dsp"])
@pytest.mark.parametrize("directed", [False, True])
def test_joint_null_matches_dense_full_vs_reduced_model_oracle(method, directed, monkeypatch):
    draws = fixed_draws(monkeypatch)
    graphs = [graph(rows, directed=directed) for rows in [Y, A, B]]
    actual = graphs[0].qap_regression(
        {"a": graphs[1], "b": graphs[2]},
        method=method,
        permutations=24,
        joint={"both": ["b", "a"]},
        adjustment="holm",
    )
    observed, values = oracle(
        [Y, A, B],
        draws,
        directed=directed,
        include_loops=False,
        intercept=True,
        method=method,
        tested=[0, 1],
    )
    joint = actual.attrs["joint_tests"][0]
    assert joint["statistic"] == pytest.approx(observed, abs=1e-12)
    count = sum(v >= observed - 1e-12 for v in values)
    assert joint["extreme_permutations"] == count and joint["pvalue"] == (count + 1) / 25
    raw = [*actual.pvalue.iloc[:2], joint["pvalue"]]
    order = sorted(range(3), key=lambda i: raw[i])
    expected = [0.0] * 3
    for at, i in enumerate(order):
        expected[i] = min(1.0, max(raw[order[j]] * (3 - j) for j in range(at + 1)))
    assert actual.adjusted_pvalue.iloc[:2].tolist() == pytest.approx(expected[:2])
    assert joint["adjusted_pvalue"] == pytest.approx(expected[2])
    assert (
        actual.adjusted_pvalue.iloc[2] != actual.adjusted_pvalue.iloc[2]
    )  # Intercept is untested.
    assert joint["terms"] == ["a", "b"] and joint["alternative"] == "greater"


@pytest.mark.parametrize(
    "method,expected",
    [
        ("none", [0.01, 0.04, 0.03]),
        ("bonferroni", [0.03, 0.12, 0.09]),
        ("holm", [0.03, 0.06, 0.06]),
        ("bh", [0.03, 0.04, 0.04]),
    ],
)
def test_adjustments_known_values(method, expected):
    assert _adjust([0.01, 0.04, 0.03], method) == pytest.approx(expected)


@pytest.mark.parametrize("method", ["freedman_lane", "dsp"])
@pytest.mark.parametrize("directed", [False, True])
def test_joint_conditioning_on_another_predictor_matches_dense_oracle(
    method, directed, monkeypatch
):
    c = [(0, 3, 2), (1, 2, 1), (1, 3, 3), (3, 0, 1)]
    draws = fixed_draws(monkeypatch)
    graphs = [graph(rows, directed=directed) for rows in [Y, A, B, c]]
    actual = graphs[0].qap_regression(
        dict(a=graphs[1], b=graphs[2], c=graphs[3]),
        method=method,
        permutations=24,
        joint={"a_b_given_c": ["a", "b"]},
    )
    observed, values = oracle(
        [Y, A, B, c],
        draws,
        directed=directed,
        include_loops=False,
        intercept=True,
        method=method,
        tested=[0, 1],
    )
    joint = actual.attrs["joint_tests"][0]
    assert joint["statistic"] == pytest.approx(observed, abs=1e-12)
    assert joint["extreme_permutations"] == sum(v >= observed - 1e-12 for v in values)


@pytest.mark.parametrize("name", ["bad\x7f", "bad\ud800"])
def test_joint_names_are_serializable_and_bounded(name):
    with pytest.raises(AnalysisError, match="hypothesis names"):
        graph(Y).qap_regression({"a": graph(A), "b": graph(B)}, joint={name: ["a"]})


@pytest.mark.parametrize(
    "options",
    [
        {"method": "raw_x"},
        {"adjustment": "auto"},
        {"joint": {"empty": []}},
        {"joint": {"unknown": ["nope"]}},
        {"joint": {"duplicate": ["a", "a"]}},
        {"joint": {"intercept": ["Constant"]}},
    ],
)
def test_invalid_joint_tests_methods_and_adjustments_are_explicit(options):
    with pytest.raises(AnalysisError):
        graph(Y).qap_regression({"a": graph(A), "b": graph(B)}, permutations=2, **options)


def test_full_joint_work_is_admitted_before_permutation_and_rank_deficiency_is_explicit(
    monkeypatch,
):
    monkeypatch.setattr(
        torch, "randperm", lambda *a, **k: pytest.fail("RNG allocated before admission")
    )
    with pytest.raises(AnalysisError, match="work plan"):
        graph(Y).qap_regression(
            {"a": graph(A), "b": graph(B)}, joint={"all": ["a", "b"]}, max_work=10
        )
    with pytest.raises(AnalysisError, match="rank deficient"):
        graph(Y).qap_regression(
            {"a": graph(A), "b": graph(A)}, method="dsp", joint={"all": ["a", "b"]}, permutations=2
        )
