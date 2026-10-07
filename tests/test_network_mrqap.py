"""Independent dense Freedman--Lane and rational coefficient oracles."""
from fractions import Fraction
from itertools import permutations as all_permutations
import math

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
import openecon._network_mrqap as mrqap
from openecon._network_mrqap import qap_regression


Y = [(0, 1, 2), (0, 2, 1), (1, 2, 4), (3, 1, 3), (2, 2, 7)]
A = [(2, 3, 3), (1, 0, 1), (3, 1, 2), (1, 2, 5), (0, 0, 4)]
B = [(2, 3, 1), (3, 0, 4), (1, 3, 5), (0, 0, 2)]


def graph(records, *, directed=False, nodes=range(4), reverse=False):
    records, nodes = list(records), list(nodes)
    if reverse:
        records.reverse()
        nodes.reverse()
    return oe.network([{"source": u, "target": v, "weight": w} for u, v, w in records],
                      weight="weight", nodes=nodes, directed=directed)


def full_vectors(records, *, nodes=range(4), directed=False, include_loops=False, values="weight"):
    """Enumerate the tiny dyad universe without any production vector helpers."""
    labels = sorted(nodes, key=lambda value: (isinstance(value, str), value))
    index, n = {label: i for i, label in enumerate(labels)}, len(labels)
    pairs = [(u, v) for u in range(n) for v in range(n)
             if (include_loops or u != v) and (directed or u <= v)]
    columns = []
    for column in records:
        edges = {}
        for u, v, weight in column:
            u, v = index[u], index[v]
            if not directed and u > v:
                u, v = v, u
            edges[u, v] = edges.get((u, v), Fraction(0)) + Fraction(weight)
        columns.append([Fraction(bool(edges.get(pair, 0))) if values == "binary"
                        else edges.get(pair, Fraction(0)) for pair in pairs])
    return pairs, columns


def dense_oracle(records, *, nodes=range(4), directed=False, include_loops=False,
                 values="weight", intercept=True, draws=()):
    """Explicitly build Y*=reduced fit + BOTH-endpoint permuted residual.

    Dense SVD least squares is independent of the production sufficient-product
    Cholesky fits; the nuisance projection is recomputed after reconstruction.
    """
    pairs, columns = full_vectors(records, nodes=nodes, directed=directed,
                                  include_loops=include_loops, values=values)
    y = torch.tensor([float(v) for v in columns[0]], dtype=torch.float64, device="cpu")
    features = torch.tensor([[float(v) for v in row] for row in zip(*columns[1:])],
                            dtype=torch.float64, device="cpu")
    p, d = features.shape[1], len(pairs)
    constant = torch.ones((d, 1), dtype=torch.float64, device="cpu")
    full = torch.cat([features, constant], dim=1) if intercept else features

    def fit(design, target):
        return design @ torch.linalg.lstsq(design, target, driver="gelsd").solution if design.shape[1] else torch.zeros_like(target)

    coefficients = torch.linalg.lstsq(full, y, driver="gelsd").solution.tolist()
    observed, statistics = [], [[] for _ in range(p)]
    for j in range(p):
        nuisance = features[:, [i for i in range(p) if i != j]]
        if intercept:
            nuisance = torch.cat([nuisance, constant], dim=1)
        reduced_fit = fit(nuisance, y)
        y_residual = y - reduced_fit
        x_residual = features[:, j] - fit(nuisance, features[:, j])

        def statistic(response):
            response = response - fit(nuisance, response)
            return float(torch.dot(x_residual, response) / torch.linalg.vector_norm(x_residual)
                         / torch.linalg.vector_norm(response))

        observed.append(statistic(y))
        for permutation in draws:
            lookup = {}
            for (u, v), value in zip(pairs, y_residual):
                u, v = permutation[u], permutation[v]
                if not directed and u > v:
                    u, v = v, u
                lookup[u, v] = value
            permuted = torch.stack([lookup[pair] for pair in pairs])
            reconstructed = reduced_fit + permuted
            statistics[j].append(statistic(reconstructed))
    return coefficients, observed, statistics


def rational_coefficients(columns, intercept=True):
    y, features = columns[0], list(zip(*columns[1:]))
    design = [[*row, Fraction(1)] if intercept else list(row) for row in features]
    p = len(design[0])
    augmented = [[sum(row[i] * row[j] for row in design) for j in range(p)]
                 + [sum(row[i] * value for row, value in zip(design, y))] for i in range(p)]
    for at in range(p):
        pivot = next(i for i in range(at, p) if augmented[i][at])
        augmented[at], augmented[pivot] = augmented[pivot], augmented[at]
        divisor = augmented[at][at]
        augmented[at] = [v / divisor for v in augmented[at]]
        for i in range(p):
            if i != at:
                multiplier = augmented[i][at]
                augmented[i] = [a - multiplier * b for a, b in zip(augmented[i], augmented[at])]
    return [row[-1] for row in augmented]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("include_loops", [False, True])
@pytest.mark.parametrize("values", ["weight", "binary"])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_mrqap_matches_dense_reconstructed_freedman_lane_oracle(
        directed, include_loops, values, intercept, alternative):
    count, seed = 37, 813
    rng = torch.Generator(device="cpu").manual_seed(seed)
    draws = [torch.randperm(4, generator=rng, device="cpu").tolist() for _ in range(count)]
    expected, observed, statistics = dense_oracle([Y, A, B], directed=directed,
        include_loops=include_loops, values=values, intercept=intercept, draws=draws)
    _, columns = full_vectors([Y, A, B], directed=directed, include_loops=include_loops, values=values)
    exact = rational_coefficients(columns, intercept=intercept)
    result = qap_regression(graph(Y, directed=directed),
        {"b": graph(B, directed=directed), "a": graph(A, directed=directed, reverse=True)},
        permutations=count, seed=seed, include_loops=include_loops, values=values,
        intercept=intercept, alternative=alternative)
    assert result.term.tolist() == ["a", "b", *(["Constant"] if intercept else [])]
    assert result.coefficient.tolist() == pytest.approx(expected, abs=3e-13)
    assert result.coefficient.tolist() == pytest.approx([float(v) for v in exact], abs=3e-13)
    assert result.statistic.iloc[:2].tolist() == pytest.approx(observed, abs=2e-14)
    tolerance = result.attrs["comparison_tolerance"]
    for j in range(2):
        if alternative == "two-sided":
            extreme = sum(abs(v) >= abs(observed[j]) - tolerance for v in statistics[j])
        elif alternative == "greater":
            extreme = sum(v >= observed[j] - tolerance for v in statistics[j])
        else:
            extreme = sum(v <= observed[j] + tolerance for v in statistics[j])
        assert result.extreme_permutations.iloc[j] == extreme
        assert result.pvalue.iloc[j] == (extreme + 1) / (count + 1)
    if intercept:
        assert pd.isna(result.statistic.iloc[-1]) and pd.isna(result.pvalue.iloc[-1])
        assert pd.isna(result.permutations.iloc[-1]) and pd.isna(result.extreme_permutations.iloc[-1])
    assert result.attrs["statistic"] == "partial correlation"
    assert result.attrs["rank"] == 2 + int(intercept)
    assert "residual node-exchangeability" in result.attrs["null"]
    assert "approximate" in result.attrs["inference_scope"]
    assert result.attrs["multiplicity_adjustment"] == "none; per-coefficient p-values"
    assert "\\begin{tabular}" in result.to_latex(index=False)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("include_loops", [False, True])
def test_every_node_permutation_statistics_match_dense_oracle(directed, include_loops, monkeypatch):
    draws = list(all_permutations(range(4)))
    _, observed, expected = dense_oracle([Y, A, B], directed=directed,
                                        include_loops=include_loops, draws=draws)
    iterator, captured = iter(draws), []
    original = mrqap._statistic

    def capture(*args):
        result = original(*args)
        captured.append(result)
        return result

    def prescribed(n, **kwargs):
        assert n == 4 and kwargs["device"] == "cpu"
        return torch.tensor(next(iterator), dtype=torch.int64, device="cpu")

    monkeypatch.setattr(mrqap, "_statistic", capture)
    monkeypatch.setattr(torch, "randperm", prescribed)
    qap_regression(graph(Y, directed=directed), {"a": graph(A, directed=directed),
        "b": graph(B, directed=directed)}, permutations=24, include_loops=include_loops)
    assert captured == pytest.approx([*observed, *[v for row in zip(*expected) for v in row]], abs=2e-14)


def test_mapping_import_order_typed_labels_private_rng_and_snapshot_immutability():
    nodes = [1, "1", "a", 2, "isolated"]
    records = [[(nodes[u], nodes[v], w) for u, v, w in column] for column in [Y, A, B]]
    snapshots = [graph(column, directed=True, nodes=nodes) for column in records]
    before = [(item._labels, item._edges.clone(), item.metadata) for item in snapshots]
    rng = torch.random.get_rng_state().clone()
    first = qap_regression(snapshots[0], {"b": snapshots[2], "a": snapshots[1]}, permutations=31, seed=517)
    shuffled = [graph(column, directed=True, nodes=nodes, reverse=True) for column in records]
    second = qap_regression(shuffled[0], {"a": shuffled[1], "b": shuffled[2]}, permutations=31, seed=517)
    assert first.equals(second)
    assert first.attrs == second.attrs
    assert torch.equal(rng, torch.random.get_rng_state())
    for snapshot, (labels, edges, metadata) in zip(snapshots, before):
        assert snapshot._labels == labels and snapshot.metadata == metadata
        assert torch.equal(snapshot._edges.indices(), edges.indices())
        assert torch.equal(snapshot._edges.values(), edges.values())


@pytest.mark.parametrize("yscale,xscale", [(1e-300, 1e-300), (1e300, 1e300),
                                         (1e-150, 1e150), (1e150, 1e-150)])
@pytest.mark.parametrize("intercept", [False, True])
def test_independent_weight_scaling_restores_raw_coefficients(yscale, xscale, intercept):
    def scaled(records, scale):
        return [(u, v, w * scale) for u, v, w in records]
    base = qap_regression(graph(Y), {"a": graph(A), "b": graph(B)}, permutations=23, seed=52, intercept=intercept)
    result = qap_regression(graph(scaled(Y, yscale)),
        {"a": graph(scaled(A, xscale)), "b": graph(scaled(B, xscale))},
        permutations=23, seed=52, intercept=intercept)
    for j in range(2):
        assert result.coefficient.iloc[j] / (yscale / xscale) == pytest.approx(base.coefficient.iloc[j], rel=2e-13)
    if intercept:
        assert result.coefficient.iloc[-1] / yscale == pytest.approx(base.coefficient.iloc[-1], abs=2e-14)
    assert result.statistic.iloc[:2].tolist() == pytest.approx(base.statistic.iloc[:2].tolist(), abs=1e-14)
    assert result.extreme_permutations.iloc[:2].tolist() == base.extreme_permutations.iloc[:2].tolist()


@pytest.mark.parametrize("scale", [1e-300, 1., 1e300])
def test_complete_almost_constant_weights_retain_sub_ulp_mean_variation(scale):
    next_value = math.nextafter(scale, math.inf)
    a = [(0, 1, scale), (0, 2, next_value), (1, 2, scale)]
    y = [(0, 1, next_value), (0, 2, scale), (1, 2, scale)]
    result = qap_regression(graph(y, nodes=range(3)), {"x": graph(a, nodes=range(3))}, permutations=29, seed=35)
    _, columns = full_vectors([y, a], nodes=range(3))
    exact = rational_coefficients(columns)
    assert result.coefficient.iloc[0] == pytest.approx(float(exact[0]), abs=2e-15)
    assert result.coefficient.iloc[-1] / scale == pytest.approx(float(exact[-1]) / scale, abs=2e-15)
    assert result.statistic.iloc[0] == pytest.approx(-.5, abs=2e-15)
    reordered = qap_regression(graph(y, nodes=range(3), reverse=True),
        {"x": graph(a, nodes=range(3), reverse=True)}, permutations=29, seed=35)
    assert result.equals(reordered)


def test_single_predictor_freedman_lane_matches_bivariate_qap():
    first, second = graph(Y), graph(A)
    fitted = qap_regression(first, {"x": second}, permutations=71, seed=993)
    correlation = first.qap_correlation(second, permutations=71, seed=993)
    assert fitted.statistic.iloc[0] == pytest.approx(correlation.correlation.iloc[0], abs=2e-15)
    # Permuting Y rather than X gives the same population distribution; using
    # inverse node permutations explicitly reproduces the identical seeded draws.
    rng = torch.Generator(device="cpu").manual_seed(993)
    draws = [torch.randperm(4, generator=rng, device="cpu").tolist() for _ in range(71)]
    _, observed, statistics = dense_oracle([Y, A], draws=draws)
    extreme = sum(abs(value) >= abs(observed[0]) - fitted.attrs["comparison_tolerance"]
                  for value in statistics[0])
    assert fitted.pvalue.iloc[0] == (extreme + 1) / 72


@pytest.mark.parametrize("options", [
    {"permutations": 0}, {"permutations": True}, {"permutations": 1.5}, {"permutations": 1_000_001},
    {"seed": -1}, {"seed": True}, {"seed": 2**63}, {"values": "signed"}, {"values": []},
    {"include_loops": 1}, {"intercept": 1}, {"alternative": "bad"}, {"alternative": []}, {"max_work": 0},
])
def test_invalid_options_are_explicit(options):
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(Y), {"a": graph(A)}, **options)
    assert error.value.code == "network_invalid_option"


@pytest.mark.parametrize("predictors", [[], {}, {"x": []}, {"": None}, {"Constant": None},
                                      {"x\n": None}, {1: None}, {"x" * 129: None}, {"\ud800": None}])
def test_predictor_mapping_and_names_are_validated(predictors):
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(Y), predictors)
    assert error.value.code == "network_invalid_option"


def test_no_more_than_64_predictors_are_admitted():
    a = graph(A)
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(Y), {f"x{i}": a for i in range(65)}, permutations=1)
    assert error.value.code == "network_invalid_option"


@pytest.mark.parametrize("other,code", [
    (lambda: graph(A, directed=True), "network_invalid_option"),
    (lambda: graph(A, nodes=range(5)), "network_invalid_label"),
    (lambda: graph([(0, "1", 1)], nodes=[0, "1", 2, 3]), "network_invalid_label"),
])
def test_exact_nodes_isolates_and_directedness_are_required(other, code):
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(Y), {"x": other()}, permutations=1)
    assert error.value.code == code


@pytest.mark.parametrize("predictors,intercept", [
    ([A, A], True), ([A, [(u, v, 2*w) for u, v, w in A]], False),
    ([[]], True), ([[]], False),
    ([[(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)]], True),
])
def test_rank_deficient_models_reject_without_silent_dropping(predictors, intercept):
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(Y), {f"x{i}": graph(records) for i, records in enumerate(predictors)},
                       permutations=1, intercept=intercept)
    assert error.value.code == "network_rank_deficient"


def test_near_singular_correlation_factorization_is_rejected_explicitly():
    nearly = [(u, v, w + (1e-10 if at == 0 else 0)) for at, (u, v, w) in enumerate(A)]
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(Y), {"x": graph(A), "almost_x": graph(nearly)}, permutations=1)
    assert error.value.code == "network_rank_deficient"


def test_zero_centered_response_is_undefined_but_no_intercept_constant_can_fit():
    complete = [(u, v, 1) for u in range(4) for v in range(u+1, 4)]
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(complete), {"x": graph(A)}, permutations=1)
    assert error.value.code == "network_undefined_statistic"
    no_intercept = qap_regression(graph(complete), {"x": graph(A)}, permutations=5, intercept=False)
    assert len(no_intercept) == 1 and no_intercept.coefficient.iloc[0] > 0


def test_response_explained_by_nuisance_is_not_given_an_arbitrary_partial_correlation():
    a, b = graph(A), graph(B)
    with pytest.raises(AnalysisError) as error:
        qap_regression(a, {"a": a, "b": b}, permutations=1)
    assert error.value.code in ("network_undefined_statistic", "network_precision")
    nearly = [(u, v, w + (1e-15 if at == 1 else 0)) for at, (u, v, w) in enumerate(A)]
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(nearly), {"a": a, "b": b}, permutations=1)
    assert error.value.code in ("network_undefined_statistic", "network_precision")


def test_excluded_loops_do_not_affect_scaling_but_extreme_included_range_rejects():
    tiny = [(u, v, w*1e-300) for u, v, w in A if u != v]
    a = graph(tiny)
    loop = graph([*tiny, (0, 0, 1e308)])
    first = qap_regression(graph(Y), {"x": a}, permutations=13)
    second = qap_regression(graph(Y), {"x": loop}, permutations=13)
    assert first.equals(second)
    with pytest.raises(AnalysisError) as error:
        qap_regression(graph(Y), {"x": loop}, permutations=1, include_loops=True)
    assert error.value.code == "network_precision"
    assert len(qap_regression(graph(Y), {"x": loop}, permutations=1, include_loops=True, values="binary")) == 2


@pytest.mark.parametrize("yscale,xscale", [(1e300, 1e-300), (1e-300, 1e300)])
def test_unrepresentable_raw_coefficients_reject_without_silent_infinity_or_zero(yscale, xscale):
    y = graph([(u, v, w*yscale) for u, v, w in Y])
    x = graph([(u, v, w*xscale) for u, v, w in A])
    with pytest.raises(AnalysisError) as error:
        qap_regression(y, {"x": x}, permutations=1)
    assert error.value.code == "network_precision"


def test_full_permutation_work_preflight_precedes_factor_and_rng_allocations(monkeypatch):
    y, a = graph(Y), graph(A)
    expected = qap_regression(y, {"x": a}, permutations=19)

    def never(*args, **kwargs):
        pytest.fail("Rejected work must not allocate factors or draw permutations.")

    monkeypatch.setattr(torch, "randperm", never)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", never)
    with pytest.raises(AnalysisError) as error:
        qap_regression(y, {"x": a}, permutations=19, max_work=expected.attrs["work_estimate"] - 1)
    assert error.value.code == "network_work_budget"


def test_all_resident_snapshots_count_against_every_distinct_owned_budget(monkeypatch):
    y, a, b = graph(Y), graph(A), graph(B)
    graphs, observed = [y, a, b], []
    for item in graphs:
        original = item._guard

        def capture(workspace, item=item, original=original):
            observed.append((item, workspace))
            return original(workspace)

        monkeypatch.setattr(item, "_guard", capture)
    qap_regression(y, {"a": a, "b": b}, permutations=1)
    assert len(observed) == 3
    totals = [item._base_bytes + workspace + 4096 for item, workspace in observed]
    assert len(set(totals)) == 1

    def never(*args, **kwargs):
        pytest.fail("Memory rejection must precede randomization.")

    monkeypatch.setattr(torch, "randperm", never)
    for item in graphs:
        original = item._budget.limit
        item._budget.limit = totals[0] - 1
        with pytest.raises(AnalysisError) as error:
            qap_regression(y, {"a": a, "b": b}, permutations=1)
        assert error.value.code == "network_memory_budget"
        item._budget.limit = original


def test_shared_response_predictor_snapshot_deduplicates_resident_memory(monkeypatch):
    y, seen = graph(Y), []
    original = y._guard

    def capture(workspace):
        seen.append(workspace)
        return original(workspace)

    monkeypatch.setattr(y, "_guard", capture)
    result = qap_regression(y, {"itself": y}, permutations=1)
    assert len(seen) == 1 and result.coefficient.iloc[0] == pytest.approx(1.)
    y._budget.limit = y._base_bytes + seen[0] + 4096
    assert qap_regression(y, {"itself": y}, permutations=1).statistic.iloc[0] == pytest.approx(1.)


def test_sparse_large_node_universe_never_materializes_a_square_or_dyad_vector(monkeypatch):
    n = 10_000
    y, a, b = [graph(column, nodes=range(n), directed=True) for column in [Y, A, B]]
    original_tensor = torch.tensor

    def bounded_tensor(data, **kwargs):
        result = original_tensor(data, **kwargs)
        assert result.numel() <= 2 * n
        assert result.device.type == "cpu"
        return result

    monkeypatch.setattr(torch, "tensor", bounded_tensor)
    with torch.device("meta"):
        result = qap_regression(y, {"a": a, "b": b}, permutations=2)
    assert result.dyads.iloc[0] == n * (n - 1)
    assert result.attrs["device"] == "cpu" and "no dense dyad" in result.attrs["memory_scope"]


def test_parquet_chunked_input_and_export_match_in_memory(tmp_path):
    paths = []
    for at, records in enumerate([Y, A, B]):
        path = tmp_path / f"relations-{at}.parquet"
        pd.DataFrame(records, columns=["source", "target", "weight"]).to_parquet(path, index=False)
        paths.append(path)
    snapshots = [oe.network(oe.scan(path), weight="weight", nodes=range(4), directed=True, batch_rows=2)
                 for path in paths]
    result = qap_regression(snapshots[0], {"a": snapshots[1], "b": snapshots[2]}, permutations=17, seed=82)
    expected = qap_regression(graph(Y, directed=True),
        {"a": graph(A, directed=True), "b": graph(B, directed=True)}, permutations=17, seed=82)
    assert result.equals(expected)
    assert all(snapshot.metadata["actual_peak_batch_rows"] == 2 for snapshot in snapshots)
    assert "\\begin{tabular}" in result.to_latex(index=False)
    assert "pvalue" in result.to_latex(index=False)
