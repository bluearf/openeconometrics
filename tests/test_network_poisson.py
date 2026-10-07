"""Independent count-dyad likelihood and exhaustive-move Poisson SBM oracles."""
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import combinations, product
import math
import random

import pandas as pd
import pytest
import torch

from openecon import _network_poisson as poisson
from openecon.analysis_contracts import AnalysisError
from openecon.frame import DataFrame
from openecon.networks import network


def graph(edges=(), *, nodes=(), directed=False, **kwargs):
    return network([{"source": a, "target": b, "w": w} for a, b, w in edges],
                   nodes=nodes, directed=directed, weight="w", **kwargs)


def observed_counts(edges, directed):
    result = {}
    for a, b, count in edges:
        if a != b and count:
            key = (a, b) if directed else frozenset((a, b))
            result[key] = result.get(key, 0) + count
    return result


def oracle(edges, nodes, membership, directed):
    """Enumerate all absent/present dyads without implementation buffers."""
    observed = observed_counts(edges, directed)
    observations = {}
    dyads = ((a, b) for a in nodes for b in nodes if a != b) if directed else combinations(nodes, 2)
    for a, b in dyads:
        x, y = membership[a], membership[b]
        cell = (x, y) if directed or x <= y else (y, x)
        observations.setdefault(cell, []).append(observed.get((a, b) if directed else frozenset((a, b)), 0))
    terms, counts = [], {}
    for cell, values in observations.items():
        total, possible = sum(values), len(values)
        mean = total / possible
        counts[cell] = total, possible
        for value in values:
            terms.append(value * math.log(mean) - mean - math.lgamma(value + 1) if mean else 0.)
    return math.fsum(terms), counts


def membership(result):
    return dict(zip(result["membership"].node, result["membership"].block))


def assert_result(result, edges, nodes, directed, groups, *, local=True):
    meta, labels = result["metadata"], membership(result)
    assert isinstance(result, poisson.NetworkBlockResult)
    assert isinstance(result["membership"], DataFrame)
    assert isinstance(result["blocks"], DataFrame)
    assert set(labels) == set(nodes) and sorted(set(labels.values())) == list(range(groups))
    expected, counts = oracle(edges, nodes, labels, directed)
    assert meta["log_likelihood"] == pytest.approx(expected, abs=2e-10)
    assert meta["model"] == "poisson" and meta["values"] == "count"
    assert meta["self_loops"] is False and meta["device"] == "cpu" and meta["dtype"] == "float64"
    assert meta["fixed_groups"] and not meta["global_optimum_certified"]
    assert not meta["degree_corrected"] and not meta["sampled"]
    assert meta["work_used"] <= meta["max_work"]
    assert meta["topology_edges"] == len(observed_counts(edges, directed))
    assert meta["total_edge_count_strength"] == sum(observed_counts(edges, directed).values())
    assert result["membership"].attrs["method"] == "poisson_sbm"
    assert "probability" not in result["blocks"].columns
    assert len(result["blocks"]) == (groups**2 if directed else groups * (groups + 1) // 2)
    for x, y, nx, ny, actual, possible, mean, identified in result["blocks"].itertuples(index=False, name=None):
        assert nx == sum(group == x for group in labels.values())
        assert ny == sum(group == y for group in labels.values())
        assert (actual, possible) == counts.get((x, y), (0, 0))
        assert identified == bool(possible)
        if possible:
            assert mean == actual / possible
        else:
            assert math.isnan(mean)
    assert sum(result["blocks"].edges) == meta["total_edge_count_strength"]
    assert sum(result["blocks"].dyads) == len(nodes) * (len(nodes) - 1) // (1 if directed else 2)
    for run in meta["start_fits"]:
        assert len(run["likelihood_history"]) == run["iterations"] + 1
        assert run["log_likelihood"] == run["likelihood_history"][-1]
        assert all(a <= b + 1e-9 for a, b in zip(run["likelihood_history"], run["likelihood_history"][1:]))
    assert meta["selected_start"] == max(range(len(meta["start_fits"])),
        key=lambda i: (meta["start_fits"][i]["log_likelihood"], meta["start_fits"][i]["converged"], -i))
    if local and meta["converged"]:
        for node in nodes:
            original = labels[node]
            if sum(block == original for block in labels.values()) <= 1:
                continue
            for target in range(groups):
                changed = dict(labels)
                changed[node] = target
                candidate, _ = oracle(edges, nodes, changed, directed)
                assert candidate <= expected + meta["tol"] + 1e-8
    assert result.summary().shape == (11, 2)
    assert "tabular" in result.to_latex()


@pytest.mark.parametrize("mask", range(64))
def test_all_four_node_undirected_count_topologies_and_fixed_partitions(mask):
    dyads = list(combinations(range(4), 2))
    edges = [(a, b, bit + 1) for bit, (a, b) in enumerate(dyads) if mask & (1 << bit)]
    snapshot = graph(edges, nodes=range(4))
    for tail in product(range(2), repeat=3):
        initial = [0, *tail]
        if max(initial) == 0:
            continue
        result = poisson.poisson_block_model(snapshot, 2, initial=initial, starts=1, max_iter=0)
        assert_result(result, edges, list(range(4)), False, 2)
        assert not result["metadata"]["converged"]


@pytest.mark.parametrize("mask", range(64))
def test_all_three_node_directed_count_topologies_and_fixed_partitions(mask):
    dyads = [(a, b) for a in range(3) for b in range(3) if a != b]
    edges = [(a, b, bit + 2) for bit, (a, b) in enumerate(dyads) if mask & (1 << bit)]
    snapshot = graph(edges, nodes=range(3), directed=True)
    for initial in [[0, 0, 1], [0, 1, 0], [0, 1, 1]]:
        result = poisson.poisson_block_model(snapshot, 2, initial=initial, starts=1, max_iter=0)
        assert_result(result, edges, list(range(3)), True, 2)


@pytest.mark.parametrize("directed", [False, True])
def test_every_count_move_delta_and_applied_topology_match_dense_oracle(directed):
    rng = random.Random(834)
    nodes = list(range(7))
    for _ in range(30):
        edges = [(a, b, rng.randrange(1, 8)) for a in nodes for b in nodes if rng.random() < .3]
        snapshot = graph(edges, nodes=nodes, directed=directed)
        observed = observed_counts(edges, directed)
        initial = [0, 1, 2, *[rng.randrange(3) for _ in range(4)]]
        original = dict(zip(nodes, initial))
        baseline, _ = oracle(edges, nodes, original, directed)
        saturated = math.fsum(poisson._saturated(count) for count in observed.values())
        for node in nodes:
            if initial.count(initial[node]) < 2:
                continue
            outgoing, incoming, out_top, in_top = [0] * 3, [0] * 3, [0] * 3, [0] * 3
            for other in nodes:
                if other == node:
                    continue
                out_count = observed.get((node, other) if directed else frozenset((node, other)), 0)
                in_count = observed.get((other, node) if directed else frozenset((node, other)), 0)
                outgoing[initial[other]] += out_count
                incoming[initial[other]] += in_count
                out_top[initial[other]] += bool(out_count)
                in_top[initial[other]] += bool(in_count)
            for target in range(3):
                if target == initial[node]:
                    continue
                state = poisson._State(snapshot, initial, 3, poisson._Work(1_000_000), saturated)
                gain, rounding = state.move(node, target, outgoing, incoming)
                changed = dict(original)
                changed[node] = target
                candidate, _ = oracle(edges, nodes, changed, directed)
                assert gain == pytest.approx(candidate - baseline, abs=3e-12)
                assert rounding >= 0
                state.move(node, target, outgoing, incoming, apply=True,
                           outgoing_topology=out_top, incoming_topology=in_top)
                assert state.likelihood() == pytest.approx(candidate, abs=2e-12)
                assert sum(state.counts[at] for at in range(9)) == sum(observed.values())
                assert sum(state.topology[at] for at in range(9)) == len(observed)


@pytest.mark.parametrize("directed", [False, True])
def test_multistart_ascent_is_local_deterministic_and_preserves_rng_and_snapshot(directed):
    rng = random.Random(591)
    nodes = list(range(14))
    edges = [(a, b, rng.randrange(1, 9)) for a, b in product(nodes, repeat=2) if rng.random() < .2]
    snapshot = graph(edges, nodes=nodes, directed=directed)
    before = torch.random.get_rng_state().clone(), random.getstate()
    indices, weights = snapshot._edges.indices().clone(), snapshot._edges.values().clone()
    attrs = deepcopy(snapshot._node_attributes), deepcopy(snapshot._edge_attributes), deepcopy(snapshot._graph_attributes)
    result = poisson.poisson_block_model(snapshot, 3, starts=4, seed=71)
    second = poisson.poisson_block_model(snapshot, 3, starts=4, seed=71)
    assert_result(result, edges, nodes, directed, 3)
    pd.testing.assert_frame_equal(result["membership"], second["membership"])
    assert result["metadata"] == second["metadata"]
    assert torch.equal(before[0], torch.random.get_rng_state()) and before[1] == random.getstate()
    assert torch.equal(indices, snapshot._edges.indices()) and torch.equal(weights, snapshot._edges.values())
    assert attrs == (snapshot._node_attributes, snapshot._edge_attributes, snapshot._graph_attributes)


@pytest.mark.parametrize("directed", [False, True])
def test_node_and_edge_order_do_not_change_canonical_starts_and_membership(directed):
    nodes = ["a", 1, "b", 7, "c", 9, "isolate"]
    edges = [("a", 1, 3), ("b", 7, 4), ("c", 9, 2), (1, 7, 1), (7, "a", 3)]
    first = poisson.poisson_block_model(graph(edges, nodes=nodes, directed=directed), 3, seed=119)
    second = poisson.poisson_block_model(graph(edges[::-1], nodes=nodes[::-1], directed=directed), 3, seed=119)
    assert membership(first) == membership(second)
    assert first["metadata"]["start_fits"] == second["metadata"]["start_fits"]
    pd.testing.assert_frame_equal(first["blocks"], second["blocks"])


@pytest.mark.parametrize("initial_type", ["mapping", "frame", "tensor", "vector"])
def test_initial_forms_canonical_group_labels_and_singleton_unidentified_blocks(initial_type):
    nodes = ["b", 5, "a"]
    labels = [2, 0, 1]
    initial = labels
    if initial_type == "mapping":
        initial = dict(zip(nodes, labels))
    elif initial_type == "frame":
        initial = pd.DataFrame({"node": pd.Series(nodes, dtype=object), "block": labels})
    elif initial_type == "tensor":
        initial = torch.tensor(labels)
    edges = [("b", 5, 2), ("a", "a", 5)]
    result = poisson.poisson_block_model(graph(edges, nodes=nodes), 3, initial=initial, starts=1, max_iter=0)
    assert_result(result, edges, nodes, False, 3)
    assert result["blocks"].mean_count.isna().sum() == 3
    assert result["metadata"]["self_loops_excluded"] == 1


@pytest.mark.parametrize("directed", [False, True])
def test_zeros_duplicate_count_multiplicities_and_excluded_loops_are_unambiguous(directed):
    edges = [(0, 1, 2), (0, 1, 3), (1, 0, 7), (1, 2, 0), (2, 2, 19)]
    result = poisson.poisson_block_model(graph(edges, nodes=range(4), directed=directed), 2, starts=1)
    assert_result(result, edges, list(range(4)), directed, 2)
    assert result["metadata"]["total_edge_count_strength"] == 12
    unweighted = network([{"source": 0, "target": 1}] * 3, nodes=range(2))
    fitted = poisson.poisson_block_model(unweighted, 1, starts=1)
    assert fitted["blocks"].edges.tolist() == [3]
    assert fitted["blocks"].mean_count.tolist() == [3.]
    assert fitted["metadata"]["log_likelihood"] == pytest.approx(3 * math.log(3) - 3 - math.lgamma(4))


@pytest.mark.parametrize("directed", [False, True])
def test_one_group_and_edgeless_models_have_correct_zero_rates(directed):
    result = poisson.poisson_block_model(graph(nodes=range(7), directed=directed), 3)
    assert_result(result, [], list(range(7)), directed, 3)
    assert result["metadata"]["log_likelihood"] == 0
    assert result["metadata"]["converged"]
    single = poisson.poisson_block_model(graph([(0, 0, 5)], nodes=[0], directed=directed), 1, starts=1)
    assert_result(single, [(0, 0, 5)], [0], directed, 1)
    assert single["blocks"].identified.tolist() == [False]


@pytest.mark.parametrize("count", [*range(1, 258), 1_000, 1_000_000, 2**53 - 1, 2**53])
def test_saturated_count_log_mass_including_factorial_constant_matches_decimal(count):
    with localcontext() as context:
        context.prec = 90
        value = Decimal(count)
        if count <= 1_000:
            log_factorial = Decimal(math.factorial(count)).ln()
        else:
            # Independent high-precision Euler-Maclaurin log-factorial expression
            # with two more correction terms than the implementation.
            pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286")
            log_factorial = ((value + Decimal(".5")) * value.ln() - value + (2 * pi).ln() / 2
                + 1 / (12 * value) - 1 / (360 * value**3) + 1 / (1260 * value**5)
                - 1 / (1680 * value**7) + 1 / (1188 * value**9) - Decimal(691) / (360360 * value**11))
        expected = float(value * value.ln() - value - log_factorial)
    assert poisson._saturated(count) == pytest.approx(expected, abs=1.5e-13)
    snapshot = graph([(0, 1, count)], nodes=range(2))
    result = poisson.poisson_block_model(snapshot, 1, starts=1)
    assert result["metadata"]["log_likelihood"] == pytest.approx(expected, abs=1.5e-13)


@pytest.mark.parametrize("count,total,dyads", [
    (1, 2**53, 1), (2**53 - 1, 2**53, 1), (1, 1, 2**53),
    (127, 255, 2), (128, 255, 2), (256, 768, 3),
    (9, 100, 11), (13, 100, 11), (1_000_000, 2_000_003, 2),
    (2**52, 2**53 - 1, 2), (2**52 - 1, 2**53 - 1, 2),
])
def test_nonnegative_deviance_near_equal_rates_and_extremes_matches_decimal(count, total, dyads):
    with localcontext() as context:
        context.prec = 85
        rate = Decimal(total) / Decimal(dyads)
        expected = float(Decimal(count) * (Decimal(count) / rate).ln() + rate - Decimal(count))
    assert poisson._deviance(count, total, dyads) == pytest.approx(expected, rel=6e-15, abs=2e-16)


@pytest.mark.parametrize("edges,code", [
    ([(0, 1, .5)], "network_count_weights"),
    ([(0, 0, .5)], "network_count_weights"),
    ([(0, 1, 2**53 + 2)], "network_count_weights"),
    ([(0, 1, 2**53), (1, 2, 1)], "network_precision"),
])
def test_fractional_and_unsafe_count_precision_rejected_before_adjacency(edges, code, monkeypatch):
    snapshot = graph(edges, nodes=range(3))
    def forbidden(*args, **kwargs):
        raise AssertionError("Count validation was deferred until after CSR allocation")
    monkeypatch.setattr(poisson, "csr", forbidden)
    with pytest.raises(AnalysisError) as error:
        poisson.poisson_block_model(snapshot, 2)
    assert error.value.code == code


def test_work_start_setup_and_full_trace_memory_preflight_reject_before_csr(monkeypatch):
    snapshot = graph([(0, 1, 3)], nodes=range(10), max_memory_mb=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("CSR allocated before preflight admission")
    monkeypatch.setattr(poisson, "csr", forbidden)
    with pytest.raises(AnalysisError) as error:
        poisson.poisson_block_model(snapshot, 3, max_work=50)
    assert error.value.code == "network_work_budget"
    with pytest.raises(AnalysisError) as error:
        poisson.poisson_block_model(snapshot, 3, starts=64, max_iter=10_000)
    assert error.value.code == "network_memory_budget"
    n, a, e, groups, starts = 10, snapshot._arcs._nnz(), 1, 3, 4
    base = n * (3 + n.bit_length()) + 4 * a + 8 * e + 32
    setup = starts * (n + 3 * e + 3 * groups**2) + 2 * n + 8 * groups**2
    setup += n + groups * (4 * n + 2 * a + groups) + (starts - 1) * n
    with pytest.raises(AnalysisError) as error:
        poisson.poisson_block_model(snapshot, groups, starts=starts, max_work=base + setup - 1)
    assert error.value.code == "network_work_budget"


def test_iteration_budget_exhaustion_returns_no_partial_fit():
    snapshot = graph([(0, 1, 3), (2, 3, 7), (5, 6, 4)], nodes=range(8))
    full = poisson.poisson_block_model(snapshot, 2, starts=1)
    with pytest.raises(AnalysisError) as error:
        poisson.poisson_block_model(snapshot, 2, starts=1, max_work=full["metadata"]["work_used"] - 1)
    assert error.value.code == "network_work_budget"


@pytest.mark.parametrize("groups,max_iter", [(1, 100), (3, 0)])
def test_fixed_partition_avoids_adjacency_and_overrides_meta_default_device(groups, max_iter, monkeypatch):
    snapshot = graph([(0, 1, 3)], nodes=range(8))
    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed count partition allocated unused CSR")
    monkeypatch.setattr(poisson, "csr", forbidden)
    initial = [i % groups for i in range(8)] if max_iter == 0 else None
    original = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        result = poisson.poisson_block_model(snapshot, groups, starts=1, initial=initial, max_iter=max_iter)
    finally:
        torch.set_default_device(original)
    assert_result(result, [(0, 1, 3)], list(range(8)), False, groups)


@pytest.mark.parametrize("directed", [False, True])
def test_sparse_ten_thousand_node_count_star_has_no_dense_adjacency(directed, monkeypatch):
    size = 10_000
    edges = [(0, node, 3) for node in range(1, size)]
    snapshot = graph(edges, nodes=range(size), directed=directed, max_memory_mb=64)
    original = torch.zeros
    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            assert shape != (size, size)
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(torch, "zeros", zeros)
    result = poisson.poisson_block_model(snapshot, 2, starts=1, max_work=10_000_000)
    assert result["metadata"]["converged"]
    assert result["membership"].block.value_counts().sort_values().tolist() == [1, size - 1]
    assert result["metadata"]["topology_edges"] == size - 1
    assert result["metadata"]["total_edge_count_strength"] == 3 * (size - 1)
    assert result["metadata"]["log_likelihood"] == pytest.approx((size - 1) * (3 * math.log(3) - 3 - math.lgamma(4)))
    assert result["metadata"]["estimated_workspace_bytes"] < 20 * 1024**2


@pytest.mark.parametrize("options", [
    {"groups": 0}, {"groups": True}, {"groups": 129}, {"groups": 9},
    {"groups": 2, "starts": 0}, {"groups": 2, "starts": 65},
    {"groups": 2, "max_iter": -1}, {"groups": 2, "max_iter": 0},
    {"groups": 2, "tol": -1}, {"groups": 2, "tol": math.nan},
    {"groups": 2, "seed": True}, {"groups": 2, "max_work": True},
    {"groups": 2, "initial": [0] * 8},
    {"groups": 2, "initial": [0, 1, 0, 1, 0, 1, 0, True]},
    {"groups": 2, "initial": [0, 1]}, {"groups": 2, "initial": "00110011"},
    {"groups": 2, "initial": {i: i % 2 for i in range(1, 9)}},
    {"groups": 2, "initial": torch.ones((8, 1), dtype=torch.int64)},
])
def test_invalid_options_and_initial_memberships_are_rejected(options):
    with pytest.raises(AnalysisError):
        poisson.poisson_block_model(graph(nodes=range(8)), **options)


def test_empty_nodes_and_unsafe_eligible_dyads_rejected():
    with pytest.raises(AnalysisError):
        poisson.poisson_block_model(graph(), 1)
    class FakeNetwork:
        node_count = 10**9
        edge_count = 0
        directed = True
        _arcs = graph(nodes=range(3))._arcs
    with pytest.raises(AnalysisError) as error:
        poisson.poisson_block_model(FakeNetwork(), 1)
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("directed", [False, True])
def test_physical_parquet_batches_counts_and_initial_block_rotation(tmp_path, directed):
    import openecon as oe
    edges = [(0, 1, 2), (0, 1, 7), (1, 0, 1), (1, 2, 4),
             (2, 0, 1), (2, 3, 2), (3, 4, 3), (4, 4, 9), (8, 8, 0)]
    path = tmp_path / "count-edges.parquet"
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_parquet(path)
    dataset = oe.scan(path)
    snapshot = oe.network(dataset, weight="weight", batch_rows=2, nodes=[9], directed=directed)
    initial = {node: i % 3 for i, node in enumerate(snapshot._labels)}
    first = poisson.poisson_block_model(snapshot, 3, initial=initial, starts=1, max_iter=0)
    second = poisson.poisson_block_model(snapshot, 3, initial={node: (block + 1) % 3 for node, block in initial.items()},
                                         starts=1, max_iter=0)
    assert_result(first, edges, list(snapshot._labels), directed, 3)
    assert membership(first) == membership(second)
    pd.testing.assert_frame_equal(first["blocks"], second["blocks"])
    assert first["membership"].attrs["network"]["actual_peak_batch_rows"] == 2
    dataset.assert_unchanged()


@pytest.mark.parametrize("directed", [False, True])
def test_complete_topology_recovers_count_blocks_and_reports_truncated_ascent(directed):
    nodes = list(range(6))
    edges = [(a, b, 11 if a // 3 == b // 3 else 1) for a in nodes for b in nodes
             if (a != b if directed else a < b)]
    snapshot = graph(edges, nodes=nodes, directed=directed)
    result = poisson.poisson_block_model(snapshot, 2, seed=0)
    assert_result(result, edges, nodes, directed, 2)
    assert membership(result) == {node: node // 3 for node in nodes}
    assert set(result["blocks"].mean_count) == {1., 11.}
    initial = [0, 1, 0, 1, 0, 1]
    truncated = poisson.poisson_block_model(snapshot, 2, initial=initial, starts=1, max_iter=1)
    assert_result(truncated, edges, nodes, directed, 2)
    assert not truncated["metadata"]["converged"]
    assert truncated["metadata"]["start_fits"][0]["moves"] > 0
    assert truncated["metadata"]["iterations"] == 1
    assert truncated.summary().Value.iloc[-1] == "Uncertified"


def test_near_exact_huge_count_rates_preserve_small_actual_likelihood():
    counts = [2**51 - 1, 2**51, 2**51 + 1]
    edges = [(a, b, count) for (a, b), count in zip(combinations(range(3), 2), counts)]
    result = poisson.poisson_block_model(graph(edges, nodes=range(3)), 1, starts=1)
    with localcontext() as context:
        context.prec = 90
        mean = sum(map(Decimal, counts)) / 3
        pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286")
        terms = []
        for count in counts:
            value = Decimal(count)
            log_factorial = ((value + Decimal(".5")) * value.ln() - value + (2 * pi).ln() / 2
                             + 1 / (12 * value) - 1 / (360 * value**3))
            terms.append(value * mean.ln() - mean - log_factorial)
        expected = float(sum(terms))
    assert result["metadata"]["log_likelihood"] == pytest.approx(expected, abs=2e-14)
    assert -60 < expected < -50
    assert result["blocks"].edges.tolist() == [sum(counts)]


def test_user_structural_and_random_starts_are_all_charged_and_retained():
    edges = [(0, 1, 3), (2, 3, 5)]
    result = poisson.poisson_block_model(graph(edges, nodes=range(8)), 2,
        initial=[0, 1, 0, 1, 0, 1, 0, 1], starts=4)
    assert [row["initialization"] for row in result["metadata"]["start_fits"]] == [
        "user", "sparse binary structural-distance seeds", "balanced random", "balanced random"]
    assert_result(result, edges, list(range(8)), False, 2)


def test_stored_count_snapshot_is_authority_not_unrecoverable_original_input_provenance():
    # Existing Network import converts weights to float64 before this method.
    # Explicitly document the authority boundary rather than claiming that this
    # method can reject an original integer whose low bit was already rounded.
    snapshot = graph([(0, 1, 2**53 + 1)], nodes=range(2))
    assert snapshot._edges.values().tolist() == [float(2**53)]
    fitted = poisson.poisson_block_model(snapshot, 1, starts=1)
    assert fitted["metadata"]["total_edge_count_strength"] == 2**53
    assert "original input count provenance" in fitted["metadata"]["count_precision_scope"]
    # Fractional rows may coalesce into a valid stored count dyad; a fractional
    # stored dyad itself is still rejected by the separate validation cases.
    aggregate = graph([(0, 1, .5), (0, 1, .5)], nodes=range(2))
    accepted = poisson.poisson_block_model(aggregate, 1, starts=1)
    assert accepted["blocks"].edges.tolist() == [1]
    assert accepted["metadata"]["log_likelihood"] == -1.
