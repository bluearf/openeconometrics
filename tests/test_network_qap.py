"""Independent dense dyad/Fraction oracles for sparse bivariate QAP inference."""
from fractions import Fraction
from itertools import permutations as all_permutations
import math

import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_qap import qap_correlation
import openecon._network_qap as qap_module


def graph(records, *, directed=False, nodes=range(4), reverse=False):
    records = list(records)
    if reverse:
        records.reverse()
        nodes = list(reversed(list(nodes)))
    return oe.network([{"source": u, "target": v, "weight": w} for u, v, w in records],
                      weight="weight", directed=directed, nodes=nodes)


def oracle(first, second, nodes, directed, include_loops, values, permutation=None):
    """Build small full vectors from records, with exact rational centering.

    This deliberately does not use production edge maps, moments, or sparse
    correlation. The second graph is explicitly relabeled at BOTH endpoints.
    """
    labels = sorted(nodes, key=lambda value: (isinstance(value, str), value))
    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    dyads = [(u, v) for u in range(n) for v in range(n)
             if (include_loops or u != v) and (directed or u <= v)]

    def dense(records, relabel=None):
        lookup = {}
        for u, v, w in records:
            u, v = index[u], index[v]
            if relabel is not None:
                u, v = relabel[u], relabel[v]
            if not directed and u > v:
                u, v = v, u
            lookup[u, v] = lookup.get((u, v), Fraction(0)) + Fraction(w)
        return [Fraction(bool(lookup.get(pair, 0))) if values == "binary"
                else lookup.get(pair, Fraction(0)) for pair in dyads]

    a, b = dense(first), dense(second, permutation)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    ac, bc = [value - ma for value in a], [value - mb for value in b]
    cov = sum(x * y for x, y in zip(ac, bc))
    va, vb = sum(x * x for x in ac), sum(x * x for x in bc)
    if not va or not vb:
        raise ValueError("constant dyad vector")
    return math.sqrt(float(cov * cov / (va * vb))) * (-1 if cov < 0 else 1)


FIRST = [(0, 1, 2), (0, 2, 1), (1, 2, 4), (3, 1, 3), (2, 2, 7)]
SECOND = [(2, 0, 3), (1, 0, 1), (3, 1, 2), (1, 2, 5), (0, 0, 4)]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("include_loops", [False, True])
@pytest.mark.parametrize("values", ["weight", "binary"])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_qap_matches_independent_seeded_dense_monte_carlo_oracle(
        directed, include_loops, values, alternative):
    count, seed = 71, 90210
    first, second = graph(FIRST, directed=directed), graph(SECOND, directed=directed, reverse=True)
    result = qap_correlation(first, second, permutations=count, seed=seed, values=values,
                             include_loops=include_loops, alternative=alternative)
    observed = oracle(FIRST, SECOND, range(4), directed, include_loops, values)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    statistics = [oracle(FIRST, SECOND, range(4), directed, include_loops, values,
                        torch.randperm(4, generator=generator, device="cpu").tolist()) for _ in range(count)]
    tolerance = result.attrs["comparison_tolerance"]
    if alternative == "two-sided":
        extreme = sum(abs(value) >= abs(observed) - tolerance for value in statistics)
    elif alternative == "greater":
        extreme = sum(value >= observed - tolerance for value in statistics)
    else:
        extreme = sum(value <= observed + tolerance for value in statistics)
    assert result.correlation.iloc[0] == pytest.approx(observed, abs=2e-14)
    assert result.extreme_permutations.iloc[0] == extreme
    assert result.pvalue.iloc[0] == (extreme + 1) / (count + 1)
    assert result.dyads.iloc[0] == (12 if directed else 6) + (4 if include_loops else 0)
    assert result.attrs["absent_dyads"] == "zero"
    assert result.attrs["inference_scope"].endswith("or MRQAP")
    assert "exchangeability" in result.attrs["null"]
    assert "\\begin{tabular}" in result.to_latex(index=False)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("include_loops", [False, True])
def test_qap_every_node_permutation_matches_dense_exact_enumeration(directed, include_loops, monkeypatch):
    enumerated = list(all_permutations(range(4)))
    iterator = iter(enumerated)
    actual = []
    original = qap_module._correlation

    def capture(*args, **kwargs):
        statistic = original(*args, **kwargs)
        actual.append(statistic)
        return statistic

    def prescribed(n, **kwargs):
        assert n == 4 and kwargs["device"] == "cpu"
        return torch.tensor(next(iterator), dtype=torch.int64, device="cpu")

    monkeypatch.setattr(qap_module, "_correlation", capture)
    monkeypatch.setattr(torch, "randperm", prescribed)
    result = qap_correlation(graph(FIRST, directed=directed), graph(SECOND, directed=directed),
                             permutations=24, include_loops=include_loops)
    observed = oracle(FIRST, SECOND, range(4), directed, include_loops, "weight")
    expected = [oracle(FIRST, SECOND, range(4), directed, include_loops, "weight", perm)
                for perm in enumerated]
    assert actual == pytest.approx([observed, *expected], abs=2e-14)
    extreme = sum(abs(value) >= abs(observed) - result.attrs["comparison_tolerance"]
                  for value in expected)
    assert result.pvalue.iloc[0] == (extreme + 1) / 25


def test_qap_exact_typed_labels_import_order_seed_and_global_rng_are_preserved():
    nodes = [1, "1", "a", 2, "isolated"]
    first = [(1, "1", 1), ("1", "a", 2), (2, 1, 3)]
    second = [("a", 2, 4), (1, "a", 1), (2, "1", 2)]
    a, b = graph(first, nodes=nodes, directed=True), graph(second, nodes=nodes, directed=True)
    labels, edges, attrs = a._labels, a._edges.clone(), a.metadata
    rng = torch.random.get_rng_state().clone()
    expected = qap_correlation(a, b, permutations=37, seed=845)
    shuffled = qap_correlation(graph(first, nodes=nodes, directed=True, reverse=True),
                               graph(second, nodes=nodes, directed=True, reverse=True),
                               permutations=37, seed=845)
    assert expected.correlation.tolist() == pytest.approx(shuffled.correlation.tolist(), abs=1e-15)
    assert expected.pvalue.tolist() == shuffled.pvalue.tolist()
    assert expected.extreme_permutations.tolist() == shuffled.extreme_permutations.tolist()
    assert expected.correlation.iloc[0] == pytest.approx(oracle(first, second, nodes, True, False, "weight"))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert a._labels == labels and a.metadata == attrs
    assert torch.equal(a._edges.indices(), edges.indices())
    assert torch.equal(a._edges.values(), edges.values())


@pytest.mark.parametrize("scale", [1e-300, 1., 1e300])
@pytest.mark.parametrize("directed", [False, True])
def test_qap_weight_scaling_preserves_correlation_and_randomization_distribution(scale, directed):
    first = [(u, v, w * scale) for u, v, w in FIRST]
    second = [(u, v, w * scale) for u, v, w in SECOND]
    result = qap_correlation(graph(first, directed=directed), graph(second, directed=directed),
                             permutations=73, seed=71, include_loops=True)
    base = qap_correlation(graph(FIRST, directed=directed), graph(SECOND, directed=directed),
                           permutations=73, seed=71, include_loops=True)
    assert result.correlation.iloc[0] == pytest.approx(base.correlation.iloc[0], abs=2e-15)
    assert result.pvalue.iloc[0] == base.pvalue.iloc[0]
    assert result.correlation.iloc[0] == pytest.approx(oracle(first, second, range(4), directed, True, "weight"), abs=2e-15)


@pytest.mark.parametrize("scale", [1e-300, 1., 1e300])
def test_qap_nearly_constant_complete_weights_do_not_cancel_variance_or_correlation(scale):
    next_value = math.nextafter(scale, math.inf)
    first = [(0, 1, scale), (0, 2, next_value), (1, 2, scale)]
    second = [(0, 1, next_value), (0, 2, scale), (1, 2, scale)]
    result = qap_correlation(graph(first, nodes=range(3)), graph(second, nodes=range(3)),
                             permutations=29, seed=35)
    assert result.correlation.iloc[0] == pytest.approx(-.5, abs=2e-15)
    assert result.correlation.iloc[0] == pytest.approx(oracle(first, second, range(3), False, False, "weight"), abs=2e-15)
    identical = qap_correlation(graph(first, nodes=range(3)), graph(first, nodes=range(3)), permutations=29)
    assert identical.correlation.iloc[0] == pytest.approx(1., abs=2e-15)
    shuffled = qap_correlation(graph(first, nodes=range(3), reverse=True),
                               graph(second, nodes=range(3), reverse=True), permutations=29, seed=35)
    assert result.equals(shuffled)


def test_qap_zero_absent_dyads_and_isolates_enter_the_correlation():
    first, second = [(0, 1, 1)], [(0, 2, 1)]
    result = qap_correlation(graph(first, nodes=range(100)), graph(second, nodes=range(100)), permutations=9)
    assert result.dyads.iloc[0] == 4950
    assert result.correlation.iloc[0] == pytest.approx(-1 / 4949, abs=1e-17)


def test_qap_duplicates_aggregate_before_weight_comparison_and_binary_is_presence():
    first = [(0, 1, 1), (1, 0, 2), (1, 2, 4), (2, 3, 2)]
    second = [(0, 1, 1), (1, 2, 4), (2, 3, 2)]
    a, b = graph(first), graph(second)
    weighted = qap_correlation(a, b, permutations=11)
    binary = qap_correlation(a, b, permutations=11, values="binary")
    assert weighted.correlation.iloc[0] == pytest.approx(oracle(first, second, range(4), False, False, "weight"))
    assert weighted.correlation.iloc[0] < 1
    assert binary.correlation.iloc[0] == pytest.approx(1.)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_qap_uniform_weight_overlap_shortcut_all_permutations_and_smaller_edge_side(directed, swap, monkeypatch):
    first = [(0, 1, 8), (1, 2, 8), (2, 3, 8), (3, 0, 8), (0, 2, 8)]
    second = [(0, 1, .25), (1, 3, .25)]
    if swap:
        first, second = second, first
    enumerated = list(all_permutations(range(4)))
    iterator, actual = iter(enumerated), []
    original = qap_module._presence_correlation

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        actual.append(result)
        return result

    monkeypatch.setattr(qap_module, "_presence_correlation", capture)
    monkeypatch.setattr(torch, "randperm", lambda n, **kwargs: torch.tensor(next(iterator), dtype=torch.int64, device="cpu"))
    result = qap_correlation(graph(first, directed=directed), graph(second, directed=directed), permutations=24)
    expected = [oracle(first, second, range(4), directed, False, "weight", perm) for perm in enumerated]
    observed = oracle(first, second, range(4), directed, False, "weight")
    assert actual == pytest.approx([observed, *expected], abs=2e-14)
    assert result.attrs["correlation_algorithm"] == "exact integer overlap numerator"
    assert result.attrs["work_used"] < 24 * (3 * 4 + 2 * (len(first) + len(second))) + 200


def test_qap_excluded_extreme_loops_do_not_affect_scaling_or_inference():
    base = graph([(0, 1, 1e-300), (1, 2, 2e-300)])
    huge_loop = graph([(0, 1, 1e-300), (1, 2, 2e-300), (2, 2, 1e308)])
    expected, actual = qap_correlation(base, base, permutations=31), qap_correlation(base, huge_loop, permutations=31)
    assert expected.correlation.tolist() == actual.correlation.tolist()
    assert expected.pvalue.tolist() == actual.pvalue.tolist()
    with pytest.raises(AnalysisError) as error:
        qap_correlation(base, huge_loop, permutations=31, include_loops=True)
    assert error.value.code == "network_precision"


def test_qap_extreme_weight_dynamic_range_is_explicit_but_binary_can_run():
    first = graph([(0, 1, 1e308), (1, 2, 5e-324)])
    second = graph([(0, 1, 1), (1, 2, 1)])
    with pytest.raises(AnalysisError) as error:
        qap_correlation(first, second, permutations=7)
    assert error.value.code == "network_precision"
    assert qap_correlation(first, second, values="binary", permutations=7).correlation.iloc[0] == pytest.approx(1.)


@pytest.mark.parametrize("first,second,options,code", [
    ([], [], {}, "network_undefined_statistic"),
    ([(0, 1, 1)], [], {}, "network_undefined_statistic"),
    ([(0, 1, 1), (0, 2, 1), (1, 2, 1)], [(0, 1, 2)], {"nodes": range(3)}, "network_undefined_statistic"),
    ([(0, 0, 1)], [(0, 0, 1)], {"nodes": [0], "include_loops": True}, "network_undefined_statistic"),
])
def test_qap_undefined_constant_or_too_small_dyad_universe(first, second, options, code):
    options = dict(options)
    nodes = options.pop("nodes", range(4))
    with pytest.raises(AnalysisError) as error:
        qap_correlation(graph(first, nodes=nodes), graph(second, nodes=nodes), permutations=5, **options)
    assert error.value.code == code


@pytest.mark.parametrize("options", [
    {"permutations": 0}, {"permutations": True}, {"permutations": 1.5}, {"permutations": 1_000_001},
    {"seed": -1}, {"seed": True}, {"seed": 2**63}, {"values": "signed"}, {"values": []},
    {"include_loops": 1}, {"alternative": "bad"}, {"alternative": []}, {"max_work": 0},
])
def test_qap_invalid_options_are_explicit(options):
    a, b = graph(FIRST), graph(SECOND)
    with pytest.raises(AnalysisError) as error:
        qap_correlation(a, b, **options)
    assert error.value.code == "network_invalid_option"


def test_qap_requires_matching_directedness_exact_node_ids_and_network_snapshot():
    a = graph([(1, "1", 1)], nodes=[1, "1", "a"])
    b = graph([(1, "1", 1)], nodes=[1, "1", "b"])
    for other, code in [(b, "network_invalid_label"),
                        (graph([(1, "1", 1)], nodes=[1, "1"]), "network_invalid_label"),
                        (graph([(1, "1", 1)], nodes=[1, "1", "a"], directed=True), "network_invalid_option"),
                        ({"source": [1]}, "network_invalid_option")]:
        with pytest.raises(AnalysisError) as error:
            qap_correlation(a, other, permutations=1)
        assert error.value.code == code


def test_qap_both_graph_memory_budgets_include_both_resident_snapshots(monkeypatch):
    first, second = graph(FIRST), graph(SECOND)

    def never(*args, **kwargs):
        pytest.fail("Memory rejection must happen before a permutation allocation.")

    monkeypatch.setattr(torch, "randperm", never)
    workspace = 512 * first.node_count + 512 * (first.edge_count + second.edge_count) + 8192
    second._budget.limit = second._base_bytes + workspace + 4096 + first._base_bytes - 1
    with pytest.raises(AnalysisError) as error:
        qap_correlation(first, second, permutations=1)
    assert error.value.code == "network_memory_budget"
    second._budget.limit += 1
    first._budget.limit = first._base_bytes + workspace + 4096 + second._base_bytes - 1
    with pytest.raises(AnalysisError) as error:
        qap_correlation(first, second, permutations=1)
    assert error.value.code == "network_memory_budget"


def test_qap_same_snapshot_is_not_counted_twice_in_memory_budget():
    first = graph(FIRST)
    workspace = 512 * first.node_count + 512 * (2 * first.edge_count) + 8192
    first._budget.limit = first._base_bytes + workspace + 4096
    assert qap_correlation(first, first, permutations=1).correlation.iloc[0] == pytest.approx(1.)


def test_qap_whole_permutation_work_is_guarded_before_rng_allocation(monkeypatch):
    first, second = graph(FIRST), graph(SECOND)
    expected = qap_correlation(first, second, permutations=17)

    def never(*args, **kwargs):
        pytest.fail("Work rejection must precede permutation allocation.")

    monkeypatch.setattr(torch, "randperm", never)
    with pytest.raises(AnalysisError) as error:
        qap_correlation(first, second, permutations=17, max_work=expected.attrs["work_used"] - 1)
    assert error.value.code == "network_work_budget"


def test_qap_sparse_large_node_universe_no_dense_adjacency_allocation(monkeypatch):
    n = 10_000
    first, second = graph([(0, 1, 1), (1, 2, 1)], nodes=range(n)), graph([(0, 2, 1), (2, 3, 1)], nodes=range(n))
    original_empty, original_zeros = torch.empty, torch.zeros

    def bounded(constructor):
        def allocate(*args, **kwargs):
            shape = args[0] if args else kwargs.get("size", ())
            size = math.prod(shape) if isinstance(shape, (list, tuple)) else shape
            assert not isinstance(size, int) or size <= 4 * n
            return constructor(*args, **kwargs)
        return allocate

    monkeypatch.setattr(torch, "empty", bounded(original_empty))
    monkeypatch.setattr(torch, "zeros", bounded(original_zeros))
    with torch.device("meta"):
        result = qap_correlation(first, second, permutations=2)
    assert result.dyads.iloc[0] == n * (n - 1) // 2
    assert result.attrs["device"] == "cpu"
    assert "O(V+E)" in result.attrs["memory_scope"]


def test_qap_permutation_comparison_is_conservative_at_float64_ties():
    first = graph([(0, 1, 1), (1, 2, 1), (2, 0, 1)], nodes=range(4))
    result = qap_correlation(first, first, permutations=99, seed=79)
    assert result.pvalue.iloc[0] >= result.attrs["pvalue_resolution"]
    assert result.pvalue.iloc[0] <= 1
    assert result.correlation.iloc[0] == pytest.approx(1.)
