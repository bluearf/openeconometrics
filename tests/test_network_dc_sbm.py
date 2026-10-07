"""Independent exact-rate, full-Poisson and move-delta degree-corrected oracles."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations_with_replacement, product
import math
import random

import pandas as pd
import pytest
import torch

from openecon import _network_dc_sbm as dc
from openecon.analysis_contracts import AnalysisError
from openecon.networks import network


def graph(edges=(), *, nodes=(), directed=False, **kwargs):
    return network([{"source": a, "target": b, "count": w} for a, b, w in edges],
                   nodes=nodes, directed=directed, weight="count", **kwargs)


def counts(edges, nodes, directed):
    order = {node: position for position, node in enumerate(nodes)}
    result = {}
    for source, target, weight in edges:
        if not weight:
            continue
        if not directed and order[source] > order[target]:
            source, target = target, source
        result[source, target] = result.get((source, target), 0) + int(weight)
    return result


def rates(edges, nodes, membership, directed):
    """Enumerate every eligible dyad, computing exact rational fitted means."""
    observed = counts(edges, nodes, directed)
    degree_out, degree_in, blocks = dict.fromkeys(nodes, 0), dict.fromkeys(nodes, 0), {}
    for (source, target), count in observed.items():
        degree_out[source] += count
        if directed:
            degree_in[target] += count
        else:
            degree_out[target] += count
        a, b = membership[source], membership[target]
        if not directed and a > b:
            a, b = b, a
        multiplier = 2 if not directed and a == b else 1
        blocks[a, b] = blocks.get((a, b), 0) + multiplier * count
    groups = set(membership.values())
    stubs_out = {a: sum(degree_out[node] for node in nodes if membership[node] == a) for a in groups}
    stubs_in = {a: sum(degree_in[node] for node in nodes if membership[node] == a) for a in groups}
    eligible = product(nodes, repeat=2) if directed else combinations_with_replacement(nodes, 2)
    means = {}
    for source, target in eligible:
        a, b = membership[source], membership[target]
        pair = (a, b) if directed or a <= b else (b, a)
        numerator = degree_out[source] * (degree_in if directed else degree_out)[target] * blocks.get(pair, 0)
        denominator = stubs_out[a] * (stubs_in if directed else stubs_out)[b]
        if not directed and source == target:
            denominator *= 2
        means[source, target] = Fraction(numerator, denominator) if denominator else Fraction(0)
    return observed, means, degree_out, degree_in, stubs_out, stubs_in, blocks


def likelihood(edges, nodes, membership, directed):
    observed, means, *_ = rates(edges, nodes, membership, directed)
    with localcontext() as context:
        context.prec = 80
        value = Decimal(0)
        for pair, rate in means.items():
            count = observed.get(pair, 0)
            mean = Decimal(rate.numerator) / Decimal(rate.denominator)
            value -= mean
            if count:
                value += Decimal(count) * mean.ln() - Decimal(math.factorial(count)).ln()
        return float(value)


def labels(result):
    return dict(zip(result["membership"].node, result["membership"].block))


def check_result(result, edges, nodes, directed, groups, *, check_local=True):
    meta, partition = result["metadata"], labels(result)
    assert len(partition) == len(nodes) and set(partition) == set(nodes)
    assert set(partition.values()) == set(range(groups))
    assert meta["model"] == "degree_corrected_poisson" and meta["degree_corrected"]
    assert meta["self_loops"] and meta["self_loops_eligible"] and not meta["sampled"]
    assert meta["dtype"] == "float64" and meta["device"] == "cpu"
    assert meta["dyads"] == (len(nodes)**2 if directed else len(nodes) * (len(nodes) + 1) // 2)
    assert meta["work_used"] <= meta["max_work"]
    assert not meta["global_optimum_certified"] and meta["fixed_groups"]
    expected = likelihood(edges, nodes, partition, directed)
    assert meta["log_likelihood"] == pytest.approx(expected, abs=2e-10)
    observed, means, degree_out, degree_in, stubs_out, stubs_in, mixing = rates(edges, nodes, partition, directed)
    assert sum(means.values()) == sum(observed.values())
    assert meta["count_strength"] == sum(observed.values())
    assert meta["observed_loop_count"] == sum(count for (a, b), count in observed.items() if a == b)
    for row in result["membership"].itertuples(index=False):
        node, block = row.node, row.block
        if directed:
            assert row.theta_out == (degree_out[node] / stubs_out[block] if stubs_out[block] else 0.)
            assert row.theta_in == (degree_in[node] / stubs_in[block] if stubs_in[block] else 0.)
            assert row.theta_out_identified == bool(stubs_out[block])
            assert row.theta_in_identified == bool(stubs_in[block])
            assert sum(rate for (a, b), rate in means.items() if a == node) == degree_out[node]
            assert sum(rate for (a, b), rate in means.items() if b == node) == degree_in[node]
        else:
            assert row.theta == (degree_out[node] / stubs_out[block] if stubs_out[block] else 0.)
            assert row.theta_identified == bool(stubs_out[block])
            fitted_degree = sum(rate * (2 if a == b else 1) for (a, b), rate in means.items() if node in (a, b))
            assert fitted_degree == degree_out[node]
    for row in result["blocks"].itertuples(index=False):
        pair = row.source_block, row.target_block
        multiplier = 2 if not directed and pair[0] == pair[1] else 1
        assert row.omega == mixing.get(pair, 0)
        assert row.count_strength * multiplier == row.omega
        assert row.expected_count == row.count_strength
        assert row.stubs_source == stubs_out[pair[0]]
        assert row.stubs_target == (stubs_in if directed else stubs_out)[pair[1]]
        assert row.identified == bool(row.stubs_source and row.stubs_target)
    assert len(result["blocks"]) == (groups**2 if directed else groups * (groups + 1) // 2)
    for run in meta["start_fits"]:
        assert len(run["likelihood_history"]) == run["iterations"] + 1
        assert run["log_likelihood"] == run["likelihood_history"][-1]
        assert all(a <= b + 1e-9 for a, b in zip(run["likelihood_history"], run["likelihood_history"][1:]))
    assert meta["selected_start"] == max(range(len(meta["start_fits"])),
        key=lambda i: (meta["start_fits"][i]["log_likelihood"], meta["start_fits"][i]["converged"], -i))
    if check_local and meta["converged"]:
        for node in nodes:
            if sum(group == partition[node] for group in partition.values()) <= 1:
                continue
            for target in range(groups):
                changed = dict(partition, **{})
                changed[node] = target
                assert likelihood(edges, nodes, changed, directed) <= expected + meta["tol"] + 1e-8


@pytest.mark.parametrize("mask", range(64))
def test_all_three_node_undirected_loop_topologies_fixed_profiles(mask):
    dyads = list(combinations_with_replacement(range(3), 2))
    edges = [(a, b, 1 + bit % 3) for bit, (a, b) in enumerate(dyads) if mask & (1 << bit)]
    snapshot = graph(edges, nodes=range(3))
    for initial in [[0, 0, 1], [0, 1, 0], [0, 1, 1]]:
        result = dc.degree_corrected_block_model(snapshot, 2, initial=initial, starts=1, max_iter=0)
        check_result(result, edges, list(range(3)), False, 2)
        assert not result["metadata"]["converged"]


@pytest.mark.parametrize("mask", range(128))
def test_directed_with_loops_independent_poisson_dyad_profiles(mask):
    dyads = list(product(range(3), repeat=2))
    edges = [(a, b, 1 + bit % 4) for bit, (a, b) in enumerate(dyads) if mask & (1 << bit)]
    snapshot = graph(edges, nodes=range(3), directed=True)
    for initial in [[0, 0, 1], [0, 1, 0], [0, 1, 1]]:
        result = dc.degree_corrected_block_model(snapshot, 2, initial=initial, starts=1, max_iter=0)
        check_result(result, edges, list(range(3)), True, 2)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_every_single_node_gain_matches_independent_full_likelihood(directed, seed):
    rng = random.Random(seed)
    nodes = list(range(6))
    dyads = list(product(nodes, repeat=2)) if directed else list(combinations_with_replacement(nodes, 2))
    edges = [(a, b, rng.randrange(1, 10)) for a, b in dyads if rng.random() < .5]
    snapshot = graph(edges, nodes=nodes, directed=directed)
    partition = [0, 0, 1, 1, 2, 2]
    work = dc._Work(10_000_000)
    out_degree, in_degree, loops_t, total, _ = dc._counts(snapshot, work)
    state = dc._State(snapshot, partition, 3, out_degree, in_degree, total, work)
    base = likelihood(edges, nodes, dict(enumerate(partition)), directed)
    observed = counts(edges, nodes, directed)
    for node in nodes:
        outgoing, incoming = [0] * 3, [0] * 3
        for (a, b), count in observed.items():
            if a == b:
                continue
            if a == node:
                outgoing[partition[b]] += count
            if b == node:
                (incoming if directed else outgoing)[partition[a]] += count
        for target in range(3):
            if target == partition[node]:
                continue
            changed = dict(enumerate(partition))
            changed[node] = target
            expected_gain = likelihood(edges, nodes, changed, directed) - base
            gain, rounding = state.move(node, target, outgoing, incoming, int(loops_t[node]))
            assert gain == pytest.approx(expected_gain, abs=5e-10)
            assert rounding >= 0
            fresh = dc._State(snapshot, partition, 3, out_degree, in_degree, total, work)
            fresh.move(node, target, outgoing, incoming, int(loops_t[node]), apply=True)
            assert fresh.likelihood() == pytest.approx(base + expected_gain, abs=5e-10)
            assert fresh.counts_t.min() >= 0


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_random_count_graphs_ascent_monotonicity_and_local_certification(directed, seed):
    rng = random.Random(seed + 123)
    nodes = list(range(7))
    dyads = product(nodes, repeat=2) if directed else combinations_with_replacement(nodes, 2)
    edges = [(a, b, rng.randrange(1, 8)) for a, b in dyads if rng.random() < .3]
    result = dc.degree_corrected_block_model(graph(edges, nodes=nodes, directed=directed),
        3, starts=4, max_iter=50, seed=seed)
    check_result(result, edges, nodes, directed, 3)
    assert result["metadata"]["converged"]


@pytest.mark.parametrize("directed", [False, True])
def test_one_node_three_actual_loops_has_correct_factor_and_absolute_likelihood(directed):
    result = dc.degree_corrected_block_model(graph([(9, 9, 3)], nodes=[9], directed=directed), 1)
    check_result(result, [(9, 9, 3)], [9], directed, 1)
    assert result["blocks"].omega.iloc[0] == (3 if directed else 6)
    assert result["blocks"].expected_count.iloc[0] == 3
    assert result["metadata"]["log_likelihood"] == pytest.approx(3 * math.log(3) - 3 - math.log(6))


def test_two_node_single_edge_fit_includes_absent_loop_exposure():
    result = dc.degree_corrected_block_model(graph([(0, 1, 1)], nodes=[0, 1]), 1)
    assert result["metadata"]["log_likelihood"] == pytest.approx(math.log(.5) - 1)
    assert list(result["membership"].theta) == [.5, .5]


@pytest.mark.parametrize("directed", [False, True])
def test_isolates_and_entire_zero_stub_groups_are_explicitly_unidentified(directed):
    edges, nodes = [(0, 1, 2)], list(range(5))
    result = dc.degree_corrected_block_model(graph(edges, nodes=nodes, directed=directed),
        3, initial=[0, 0, 0, 1, 2], starts=1, max_iter=0)
    check_result(result, edges, nodes, directed, 3)
    membership = result["membership"].set_index("node")
    if directed:
        assert membership.loc[2, "theta_out"] == membership.loc[2, "theta_in"] == 0
        assert membership.loc[2, "theta_out_identified"]
        assert not membership.loc[3, "theta_out_identified"]
    else:
        assert membership.loc[2, "theta"] == 0 and membership.loc[2, "theta_identified"]
        assert membership.loc[3, "theta"] == 0 and not membership.loc[3, "theta_identified"]
    assert not result["blocks"].iloc[-1].identified


@pytest.mark.parametrize("directed", [False, True])
def test_no_edges_is_a_flat_likelihood_with_honest_nonidentification(directed):
    result = dc.degree_corrected_block_model(graph(nodes=range(8), directed=directed), 3, starts=4)
    assert result["metadata"]["log_likelihood"] == 0
    assert result["metadata"]["converged"] and result["metadata"]["iterations"] == 1
    assert not result["blocks"].identified.any()
    check_result(result, [], list(range(8)), directed, 3)


@pytest.mark.parametrize("directed", [False, True])
def test_large_flat_graph_does_not_spend_candidate_budget_on_zero_stub_nodes(directed):
    # Every possible partition has likelihood zero. Trying 63 alternative
    # blocks for each isolate previously exhausted the budget on this graph.
    result = dc.degree_corrected_block_model(graph(nodes=range(4000), directed=directed),
        64, starts=1, max_iter=10, max_work=1_000_000)
    meta = result["metadata"]
    assert meta["converged"] and meta["iterations"] == 1
    assert meta["log_likelihood"] == 0 and meta["work_used"] < 1_000_000
    assert len(result["membership"]) == 4000 and result["membership"].block.nunique() == 64
    assert not result["blocks"].identified.any()


@pytest.mark.parametrize("directed", [False, True])
def test_zero_stub_nodes_never_enter_profile_move_evaluation(directed, monkeypatch):
    original = dc._State.move
    considered = set()

    def counted(self, node, *args, **kwargs):
        assert self.degree_out[node] or self.degree_in[node], "zero-stub node entered a flat candidate search"
        considered.add(self.graph._labels[node])
        return original(self, node, *args, **kwargs)

    monkeypatch.setattr(dc._State, "move", counted)
    edges = [(0, 1, 3), (2, 0, 2)]
    result = dc.degree_corrected_block_model(graph(edges, nodes=range(8), directed=directed),
        2, initial=[0, 1, 0, 1, 0, 1, 0, 1], starts=1, max_iter=10)
    check_result(result, edges, list(range(8)), directed, 2)
    assert considered <= {0, 1, 2}
    if directed:
        # Sources and sinks with only one nonzero degree still affect likelihood.
        assert {1, 2} <= considered


def test_structural_initialization_budget_is_admitted_before_csr(monkeypatch):
    snapshot = graph([(0, 1, 1)], nodes=range(4000))

    def forbidden(*args, **kwargs):
        raise AssertionError("CSR allocated before structural setup budget was admitted")

    monkeypatch.setattr(dc, "csr", forbidden)
    with pytest.raises(AnalysisError, match="setup exceeds max_work"):
        dc.degree_corrected_block_model(snapshot, 64, starts=1, max_work=1_000_000)


def test_user_structural_and_random_initializations_are_retained():
    result = dc.degree_corrected_block_model(graph([(0, 1, 3), (2, 3, 5)], nodes=range(8)),
        2, initial=[0, 1, 0, 1, 0, 1, 0, 1], starts=4)
    assert [run["initialization"] for run in result["metadata"]["start_fits"]] == [
        "user", "sparse binary structural-distance seeds", "balanced random", "balanced random"]
    check_result(result, [(0, 1, 3), (2, 3, 5)], list(range(8)), False, 2)


@pytest.mark.parametrize("directed", [False, True])
def test_structural_start_recovers_heterogeneous_exact_count_blocks_without_planted_initial(directed):
    factors, incoming = [1, 2, 3, 4] * 2, [4, 3, 2, 1] * 2
    nodes = list(range(8))
    dyads = product(nodes, repeat=2) if directed else combinations_with_replacement(nodes, 2)
    edges = [(a, b, factors[a] * incoming[b] if directed else
              factors[a] * factors[b] * (1 if a == b else 2))
             for a, b in dyads if a // 4 == b // 4]
    result = dc.degree_corrected_block_model(graph(edges, nodes=nodes, directed=directed),
        2, starts=1, seed=13, max_iter=20)
    assert labels(result) == {node: node // 4 for node in nodes}
    assert result["metadata"]["converged"]
    assert result["metadata"]["start_fits"][0]["initialization"] == "sparse binary structural-distance seeds"
    check_result(result, edges, nodes, directed, 2)
    observed, means, *_ = rates(edges, nodes, labels(result), directed)
    assert all(mean == observed.get(pair, 0) for pair, mean in means.items())


@pytest.mark.parametrize("directed", [False, True])
def test_exact_typed_ids_order_independence_and_private_torch_rng(directed):
    nodes = [1, "1", "é", -1, "isolated"]
    edges = [(1, "1", 3), ("1", "é", 2), ("é", 1, 4), (-1, -1, 2)]
    first_graph = graph(edges, nodes=nodes, directed=directed)
    before = torch.get_rng_state().clone()
    first = dc.degree_corrected_block_model(first_graph, 2, seed=912, starts=5)
    assert torch.equal(torch.get_rng_state(), before)
    second = dc.degree_corrected_block_model(graph(list(reversed(edges)), nodes=list(reversed(nodes)), directed=directed),
        2, seed=912, starts=5)
    assert labels(first) == labels(second)
    assert first["metadata"]["start_fits"] == second["metadata"]["start_fits"]
    check_result(first, edges, nodes, directed, 2)


@pytest.mark.parametrize("directed", [False, True])
def test_duplicate_multiplicity_and_reciprocal_counts_do_not_collapse_to_binary(directed):
    edges = [(0, 1, 2), (0, 1, 3), (1, 0, 4), (1, 1, 2), (1, 1, 1)]
    result = dc.degree_corrected_block_model(graph(edges, nodes=range(3), directed=directed),
        2, initial=[0, 0, 1], starts=1, max_iter=0)
    check_result(result, edges, list(range(3)), directed, 2)
    assert result["metadata"]["count_strength"] == 12
    assert result["metadata"]["topology_edges"] == (3 if directed else 2)


@pytest.mark.parametrize("count", [1, 2, 7, 31, 32, 33, 100, 1000, 10_000, 2**52, 2**53])
def test_native_saturated_poisson_mass_retains_large_count_precision(count):
    with localcontext() as context:
        context.prec = 90
        value = Decimal(count)
        if count < 1000:
            expected = value * value.ln() - value - Decimal(math.factorial(count)).ln()
        else:
            # Independent high-order Stirling oracle with Decimal constants.
            pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286")
            expected = -(Decimal(2) * pi * value).ln() / 2
            for numerator, denominator, power in [(1, 12, 1), (-1, 360, 3), (1, 1260, 5),
                    (-1, 1680, 7), (1, 1188, 9), (-691, 360360, 11), (1, 156, 13), (-3617, 122400, 15)]:
                expected -= Decimal(numerator) / (Decimal(denominator) * value**power)
        assert dc._saturated(count) == pytest.approx(float(expected), abs=2e-13)


@pytest.mark.parametrize("directed", [False, True])
def test_large_exact_loop_count_fit_avoids_log_gamma_cancellation(directed):
    count = 2**52 if not directed else 2**53
    result = dc.degree_corrected_block_model(graph([(0, 0, count)], nodes=[0], directed=directed), 1)
    assert result["metadata"]["log_likelihood"] == dc._saturated(count)
    assert result["metadata"]["log_likelihood"] < -15
    assert result["blocks"].omega.iloc[0] == (count if directed else 2 * count)


@pytest.mark.parametrize("pair", [(2**53, 2**53 - 1), (2**53 - 1, 2**53),
    (2**106 + 1, 2**106), (2**106, 2**106 + 1), (3, 1), (1, 3)])
def test_exact_integer_log_ratio_retains_small_relative_differences(pair):
    with localcontext() as context:
        context.prec = 100
        expected = (Decimal(pair[0]) / Decimal(pair[1])).ln()
        assert dc._log_ratio(*pair) == pytest.approx(float(expected), rel=2e-15, abs=1e-40)


@pytest.mark.parametrize("value", [.5, 1.5, 10.25, 2**53 + 2])
def test_non_count_strengths_are_rejected(value):
    with pytest.raises(AnalysisError, match="integer.*count|counts"):
        dc.degree_corrected_block_model(graph([(0, 1, value)], nodes=[0, 1]), 1)


@pytest.mark.parametrize("directed", [False, True])
def test_total_strength_precision_guard(directed):
    edges = [(0, 0, 2**52), (1, 1, 2**52 + 1 if directed else 1)]
    with pytest.raises(AnalysisError, match="Total counts"):
        dc.degree_corrected_block_model(graph(edges, nodes=[0, 1], directed=directed), 1)


@pytest.mark.parametrize("option,value", [("groups", 0), ("groups", True), ("groups", 129),
    ("groups", 4), ("starts", 0), ("starts", True), ("starts", 65), ("max_iter", -1),
    ("max_iter", True), ("max_iter", 10001), ("seed", -1), ("seed", True),
    ("seed", 2**63), ("tol", -1), ("tol", math.nan), ("tol", math.inf), ("tol", True),
    ("max_work", 0), ("max_work", True)])
def test_invalid_fit_options_are_structured_errors(option, value):
    options = {"groups": 2, option: value}
    with pytest.raises(AnalysisError):
        dc.degree_corrected_block_model(graph([(0, 1, 1)], nodes=range(3)), **options)


@pytest.mark.parametrize("initial", [[0, 0, 0], [0, 1], [0, 1, 2], [False, 1, 1],
    {0: 0, 1: 1}, {0: 0, 1: 1, "2": 1}, torch.tensor([0., 1., 1.]),
    pd.DataFrame({"node": [0, 0, 2], "block": [0, 1, 1]})])
def test_invalid_initial_membership_is_rejected(initial):
    with pytest.raises(AnalysisError):
        dc.degree_corrected_block_model(graph([(0, 1, 1)], nodes=range(3)), 2,
            initial=initial, starts=1, max_iter=0)


@pytest.mark.parametrize("kind", ["mapping", "frame", "vector", "tensor"])
def test_initial_membership_representations_and_canonical_group_export(kind):
    values = {"a": 1, "b": 1, "c": 0}
    initial = {"mapping": values, "frame": pd.DataFrame({"node": list(values), "block": list(values.values())}),
               "vector": [1, 1, 0], "tensor": torch.tensor([1, 1, 0])}[kind]
    result = dc.degree_corrected_block_model(graph([("a", "b", 2)], nodes=list(values)),
        2, initial=initial, starts=1, max_iter=0)
    assert labels(result) == {"a": 0, "b": 0, "c": 1}
    check_result(result, [("a", "b", 2)], list(values), False, 2)


@pytest.mark.parametrize("starts,initial", [(1, None), (2, [0, 0, 1])])
def test_zero_iterations_requires_exactly_one_supplied_partition(starts, initial):
    with pytest.raises(AnalysisError, match="max_iter=0"):
        dc.degree_corrected_block_model(graph(nodes=range(3)), 2, max_iter=0, starts=starts, initial=initial)


def test_iteration_limit_is_not_mislabeled_converged():
    result = dc.degree_corrected_block_model(graph([(0, 0, 1), (0, 1, 4), (0, 3, 4),
        (1, 2, 1), (2, 2, 5), (3, 3, 4)], nodes=range(4)),
        2, initial=[0, 1, 0, 1], starts=1, max_iter=1)
    assert result["metadata"]["start_fits"][0]["moves"] > 0
    assert not result["metadata"]["converged"]


def test_work_and_memory_preflight_before_csr_and_no_partial_fit(monkeypatch):
    snapshot = graph([(0, 1, 1)], nodes=range(10))
    original = dc.csr
    def forbidden(*args, **kwargs):
        raise AssertionError("CSR allocated before budget admission")
    monkeypatch.setattr(dc, "csr", forbidden)
    with pytest.raises(AnalysisError, match="max_work"):
        dc.degree_corrected_block_model(snapshot, 3, starts=4, max_work=100)
    limited = graph([(0, 1, 1)], nodes=range(10), max_memory_mb=.04)
    with pytest.raises(AnalysisError, match="max_memory_mb"):
        dc.degree_corrected_block_model(limited, 3)
    monkeypatch.setattr(dc, "csr", original)
    complete = dc.degree_corrected_block_model(snapshot, 3, starts=1)
    with pytest.raises(AnalysisError, match="max_work"):
        dc.degree_corrected_block_model(snapshot, 3, starts=1, max_work=complete["metadata"]["work_used"] - 1)


@pytest.mark.parametrize("groups,max_iter", [(1, 100), (2, 0)])
def test_fixed_profiles_do_not_allocate_unused_csr(groups, max_iter, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fixed profile does not need CSR")
    monkeypatch.setattr(dc, "csr", forbidden)
    initial = [0, 0, 1] if groups == 2 else None
    result = dc.degree_corrected_block_model(graph([(0, 1, 3), (0, 0, 2)], nodes=range(3)),
        groups, max_iter=max_iter, initial=initial, starts=1)
    check_result(result, [(0, 1, 3), (0, 0, 2)], list(range(3)), False, groups)


@pytest.mark.parametrize("directed", [False, True])
def test_large_star_remains_sparse_and_complete_including_loop_exposure(directed, monkeypatch):
    n = 4000
    edges = [(0, node, 1 + node % 3) for node in range(1, n)]
    snapshot = graph(edges, nodes=range(n), directed=directed, max_memory_mb=128)
    zeros = torch.zeros
    def bounded(shape, *args, **kwargs):
        if isinstance(shape, (list, tuple)) and len(shape) == 2 and min(shape) > 128:
            raise AssertionError("dense node adjacency allocation")
        return zeros(shape, *args, **kwargs)
    monkeypatch.setattr(torch, "zeros", bounded)
    result = dc.degree_corrected_block_model(snapshot, 2, starts=1, max_iter=4, max_work=5_000_000)
    assert len(result["membership"]) == n
    assert result["metadata"]["dyads"] == (n**2 if directed else n * (n + 1) // 2)
    assert result["metadata"]["estimated_workspace_bytes"] < 20 * 1024**2
    assert sum(result["blocks"].count_strength) == sum(count for _, _, count in edges)


def test_cpu_buffers_ignore_ambient_meta_device_and_do_not_mutate_graph():
    snapshot = graph([(0, 1, 2), (0, 0, 1)], nodes=range(4))
    original = torch.get_default_device()
    indices, values, metadata = snapshot._edges.indices().clone(), snapshot._edges.values().clone(), deepcopy(snapshot.metadata)
    try:
        torch.set_default_device("meta")
        result = dc.degree_corrected_block_model(snapshot, 2, starts=2)
    finally:
        torch.set_default_device(original)
    assert torch.equal(snapshot._edges.indices(), indices)
    assert torch.equal(snapshot._edges.values(), values)
    assert snapshot.metadata == metadata
    check_result(result, [(0, 1, 2), (0, 0, 1)], list(range(4)), False, 2)


@pytest.mark.parametrize("directed", [False, True])
def test_physical_parquet_count_batches_preserve_complete_fit(tmp_path, directed):
    import openecon as oe
    rows = pd.DataFrame({"source": [0, 0, 1, 1, 2, 3], "target": [0, 1, 0, 2, 2, 3],
                         "count": [2, 5, 3, 7, 4, 1]})
    path = tmp_path / "count-multigraph.parquet"
    rows.to_parquet(path, row_group_size=2)
    snapshot = oe.network(oe.scan(path), weight="count", nodes=range(5), directed=directed, batch_rows=2)
    result = dc.degree_corrected_block_model(snapshot, 2, starts=3, seed=19)
    edges = list(rows.itertuples(index=False, name=None))
    check_result(result, edges, list(range(5)), directed, 2)


def test_membership_plot_groups_are_usable_without_solver_dependency():
    snapshot = graph([(0, 1, 3), (1, 2, 1)], nodes=range(4))
    result = dc.degree_corrected_block_model(snapshot, 2, starts=2)
    plot = snapshot.to_plot_data(groups=result["membership"])
    assert len(plot["nodes"]) == 4
    assert result["membership"].attrs["method"] == "degree_corrected_poisson_sbm"
    assert "tabular" in result["membership"].to_latex(index=False)
    assert "tabular" in result["blocks"].to_latex(index=False)
