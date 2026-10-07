"""Independent dyad likelihood and exhaustive single-move SBM oracles."""
from copy import deepcopy
from itertools import combinations, product
import math
import random

import pandas as pd
import pytest
import torch

from openecon import _network_sbm as sbm
from openecon.analysis_contracts import AnalysisError
from openecon.frame import DataFrame
from openecon.networks import network


def graph(edges=(), *, nodes=(), directed=False, **kwargs):
    return network([{"source": a, "target": b, "w": w} for a, b, w in edges],
                   nodes=nodes, directed=directed, weight="w", **kwargs)


def oracle(edges, nodes, membership, directed):
    """Enumerate eligible dyads independently; never use implementation counts."""
    presence = {(a, b) for a, b, w in edges if w > 0 and a != b}
    if not directed:
        presence |= {(b, a) for a, b in presence}
    counts = {}
    dyads = ((a, b) for a in nodes for b in nodes if a != b) if directed else combinations(nodes, 2)
    for a, b in dyads:
        x, y = membership[a], membership[b]
        cell = (x, y) if directed or x <= y else (y, x)
        actual, possible = counts.get(cell, (0, 0))
        counts[cell] = actual + int((a, b) in presence), possible + 1
    terms = []
    for count, possible in counts.values():
        probability = count / possible
        if count:
            terms.append(count * math.log(probability))
        if count < possible:
            terms.append((possible - count) * math.log1p(-probability))
    return math.fsum(terms), counts


def membership(result):
    return dict(zip(result["membership"].node, result["membership"].block))


def assert_result(result, edges, nodes, directed, groups):
    meta, labels = result["metadata"], membership(result)
    assert isinstance(result, sbm.NetworkBlockResult)
    assert isinstance(result["membership"], DataFrame)
    assert isinstance(result["blocks"], DataFrame)
    assert len(labels) == len(nodes) and set(labels) == set(nodes)
    assert sorted(set(labels.values())) == list(range(groups))
    expected, counts = oracle(edges, nodes, labels, directed)
    assert meta["log_likelihood"] == pytest.approx(expected, abs=5e-10)
    assert meta["work_used"] <= meta["max_work"]
    assert meta["fixed_groups"] and not meta["global_optimum_certified"]
    assert not meta["degree_corrected"] and not meta["sampled"]
    block_rows = list(result["blocks"].itertuples(index=False, name=None))
    assert len(block_rows) == groups**2 if directed else len(block_rows) == groups * (groups + 1) // 2
    for x, y, nx, ny, actual, possible, probability, identified in block_rows:
        assert nx == sum(group == x for group in labels.values())
        assert ny == sum(group == y for group in labels.values())
        assert (actual, possible) == counts.get((x, y), (0, 0))
        assert identified == bool(possible)
        if possible:
            assert probability == actual / possible
        else:
            assert math.isnan(probability)
    assert sum(result["blocks"].dyads) == len(nodes) * (len(nodes) - 1) // (1 if directed else 2)
    for run in meta["start_fits"]:
        assert len(run["likelihood_history"]) == run["iterations"] + 1
        assert run["log_likelihood"] == run["likelihood_history"][-1]
        assert all(a <= b + 1e-9 for a, b in zip(run["likelihood_history"], run["likelihood_history"][1:]))
    assert meta["selected_start"] == max(range(len(meta["start_fits"])),
        key=lambda i: (meta["start_fits"][i]["log_likelihood"], meta["start_fits"][i]["converged"], -i))
    if meta["converged"]:
        for node in nodes:
            original = labels[node]
            if sum(block == original for block in labels.values()) <= 1:
                continue
            for target in range(groups):
                changed = dict(labels)
                changed[node] = target
                candidate, _ = oracle(edges, nodes, changed, directed)
                assert candidate <= expected + meta["tol"] + 1e-8
    summary = result.summary()
    assert summary.shape == (11, 2)
    assert "tabular" in result.to_latex()
    assert "groups=" in repr(result)


@pytest.mark.parametrize("mask", range(64))
def test_all_four_node_undirected_topologies_and_every_fixed_two_group_partition(mask):
    dyads = list(combinations(range(4), 2))
    edges = [(a, b, 1) for bit, (a, b) in enumerate(dyads) if mask & (1 << bit)]
    snapshot = graph(edges, nodes=range(4))
    for tail in product(range(2), repeat=3):
        initial = [0, *tail]
        if max(initial) == 0:
            continue
        result = sbm.block_model(snapshot, 2, initial=initial, starts=1, max_iter=0)
        assert_result(result, edges, list(range(4)), False, 2)
        assert not result["metadata"]["converged"]


@pytest.mark.parametrize("mask", range(64))
def test_all_three_node_directed_topologies_have_correct_offdiagonal_block_counts(mask):
    dyads = [(a, b) for a in range(3) for b in range(3) if a != b]
    edges = [(a, b, 1) for bit, (a, b) in enumerate(dyads) if mask & (1 << bit)]
    snapshot = graph(edges, nodes=range(3), directed=True)
    for initial in [[0, 0, 1], [0, 1, 0], [0, 1, 1]]:
        result = sbm.block_model(snapshot, 2, initial=initial, starts=1, max_iter=0)
        assert_result(result, edges, list(range(3)), True, 2)


@pytest.mark.parametrize("directed", [False, True])
def test_every_moving_delta_matches_independent_dyad_enumeration(directed):
    rng = random.Random(11391)
    nodes = list(range(7))
    for _ in range(35):
        edges = [(a, b, 1) for a in nodes for b in nodes if rng.random() < .3]
        snapshot = graph(edges, nodes=nodes, directed=directed)
        initial = [0, 1, 2, *[rng.randrange(3) for _ in range(4)]]
        original = dict(zip(nodes, initial))
        baseline, _ = oracle(edges, nodes, original, directed)
        for node in nodes:
            old = initial[node]
            if initial.count(old) < 2:
                continue
            outgoing, incoming = [0] * 3, [0] * 3
            presence = {(a, b) for a, b, w in edges if w and a != b}
            if not directed:
                presence |= {(b, a) for a, b in presence}
            for a, b in presence:
                if a == node:
                    outgoing[initial[b]] += 1
                if b == node:
                    incoming[initial[a]] += 1
            for target in range(3):
                if target == old:
                    continue
                state = sbm._State(snapshot, initial, 3, sbm._Work(100_000))
                gain, rounding = state.move(node, target, outgoing, incoming)
                changed = dict(original)
                changed[node] = target
                candidate, _ = oracle(edges, nodes, changed, directed)
                assert gain == pytest.approx(candidate - baseline, abs=1e-12)
                assert rounding >= 0
                state.move(node, target, outgoing, incoming, apply=True)
                assert state.likelihood() == pytest.approx(candidate, abs=1e-12)
                assert list(state.membership) == [changed[label] for label in snapshot._labels]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_random_sparse_fits_increase_profile_likelihood_and_certify_only_local_optima(directed, seed):
    rng = random.Random(seed)
    nodes = [2, "2", "a", 4, -3, "b", "isolate"]
    edges = [(a, b, rng.choice([1e-240, 1., 1e240])) for a in nodes[:-1] for b in nodes[:-1]
             if rng.random() < .3]
    snapshot = graph(edges, nodes=nodes, directed=directed)
    result = sbm.block_model(snapshot, 3, starts=4, seed=seed)
    assert_result(result, edges, nodes, directed, 3)
    assert result["metadata"]["converged"]


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("size,groups", [(1, 1), (4, 1), (4, 2), (4, 4), (30, 3)])
def test_empty_topology_fits_boundary_probabilities_and_nonempty_groups(directed, size, groups):
    snapshot = graph(nodes=range(size), directed=directed)
    result = sbm.block_model(snapshot, groups, starts=2)
    assert_result(result, [], list(range(size)), directed, groups)
    assert result["metadata"]["log_likelihood"] == 0.
    assert result["metadata"]["converged"]
    assert all(value == 0 or math.isnan(value) for value in result["blocks"].probability)


@pytest.mark.parametrize("directed", [False, True])
def test_complete_topology_fits_one_probabilities_and_excludes_self_loops(directed):
    nodes = list(range(8))
    edges = [(a, b, 7) for a in nodes for b in nodes]
    result = sbm.block_model(graph(edges, nodes=nodes, directed=directed), 3)
    assert_result(result, edges, nodes, directed, 3)
    assert result["metadata"]["self_loops_excluded"] == 8
    assert all(value == 1 or math.isnan(value) for value in result["blocks"].probability)


def test_boundary_singletons_export_unidentified_probability_as_nan_not_zero():
    edges = [(0, 1, 1)]
    result = sbm.block_model(graph(edges, nodes=range(2)), 2)
    assert result["blocks"].identified.tolist() == [False, True, False]
    assert result["blocks"].probability.tolist()[1] == 1.
    assert math.isnan(result["blocks"].probability.iloc[0])
    assert result["metadata"]["zero_dyad_blocks"].startswith("unidentified")


@pytest.mark.parametrize("directed", [False, True])
def test_duplicate_weights_and_large_loops_collapse_to_explicit_binary_topology(directed):
    edges = [(0, 1, 1e200), (0, 1, 1e200), (1, 0, 1e-200),
             (2, 0, 5), (2, 3, 0), (0, 0, 1e308), (4, 4, 9)]
    nodes = list(range(6))
    result = sbm.block_model(graph(edges, nodes=nodes, directed=directed), 2, values="binary")
    assert_result(result, edges, nodes, directed, 2)
    assert result["metadata"]["self_loops_excluded"] == 2
    assert result["metadata"]["weight_semantics"].startswith("unique positive-edge")
    with pytest.raises(AnalysisError, match="binary"):
        sbm.block_model(graph(edges, nodes=nodes, directed=directed), 2, values="weight")


@pytest.mark.parametrize("directed", [False, True])
def test_import_order_invariance_exact_typed_labels_and_private_rng(directed):
    nodes = [2, "2", "a", 4, -3, "b", "isolate"]
    edges = [(2, "2", 1), ("2", "a", 1), ("a", 2, 1), (4, -3, 1), (-3, "b", 1)]
    rng = random.Random(857)
    before = torch.get_rng_state().clone()
    expected = None
    for _ in range(10):
        rng.shuffle(nodes)
        rng.shuffle(edges)
        result = sbm.block_model(graph(edges, nodes=nodes, directed=directed), 3, seed=231)
        current = membership(result), result["blocks"].to_dict("records"), result["metadata"]["start_fits"]
        if expected is None:
            expected = current
        else:
            assert current[0] == expected[0]
            pd.testing.assert_frame_equal(result["blocks"], pd.DataFrame(expected[1]), check_frame_type=False)
            assert current[2] == expected[2]
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("initial_type", ["mapping", "frame", "list", "tensor"])
def test_initial_partition_forms_recover_exact_boundary_block_model(initial_type):
    nodes = [0, "0", 2, "2", "isolate"]
    initial = [0, 0, 1, 1, 2]
    mapping = dict(zip(nodes, initial))
    edges = [(0, "0", 1), (2, "2", 1)]
    if initial_type == "mapping":
        initial = mapping
    elif initial_type == "frame":
        initial = pd.DataFrame({"node": pd.Series(nodes, dtype=object), "block": initial})
    elif initial_type == "tensor":
        initial = torch.tensor(initial)
    result = sbm.block_model(graph(edges, nodes=nodes), 3, initial=initial, starts=1)
    assert_result(result, edges, nodes, False, 3)
    assert result["metadata"]["log_likelihood"] == 0.
    assert result["metadata"]["start_fits"][0]["initialization"] == "user"


def test_max_iteration_truncation_is_honest_and_preserves_all_trace_steps():
    rng = random.Random(791)
    nodes = list(range(15))
    edges = [(a, b, 1) for a, b in combinations(nodes, 2) if rng.random() < .3]
    result = sbm.block_model(graph(edges, nodes=nodes), 3, max_iter=1, starts=1, seed=21)
    assert_result(result, edges, nodes, False, 3)
    assert result["metadata"]["start_fits"][0]["moves"] > 0
    assert not result["metadata"]["converged"]
    assert result["metadata"]["iterations"] == 1
    assert result.summary().Value.iloc[-1] == "Uncertified"


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("assortative", [False, True])
def test_sparse_structural_seed_recovers_assortative_and_disassortative_block_models(directed, assortative):
    nodes = list(range(18))
    edges = [(a, b, 1) for a in nodes for b in nodes
             if (a != b if directed else a < b) and ((a // 6 == b // 6) == assortative)]
    snapshot = graph(edges, nodes=nodes, directed=directed)
    result = sbm.block_model(snapshot, 3, starts=1, seed=0)
    assert_result(result, edges, nodes, directed, 3)
    assert membership(result) == {node: node // 6 for node in nodes}
    assert result["metadata"]["log_likelihood"] == 0.
    assert result["metadata"]["start_fits"][0]["initialization"] == "sparse structural-distance seeds"
    assert result["metadata"]["converged"]


def test_user_initial_remains_first_and_structural_seed_precedes_random_other_starts():
    edges = [(0, 1, 1), (2, 3, 1)]
    result = sbm.block_model(graph(edges, nodes=range(8)), 2,
                             initial=[0, 1, 0, 1, 0, 1, 0, 1], starts=4)
    assert [row["initialization"] for row in result["metadata"]["start_fits"]] == [
        "user", "sparse structural-distance seeds", "balanced random", "balanced random"]


def test_structural_seed_work_is_admitted_before_csr_buffers(monkeypatch):
    snapshot = graph([(0, 1, 1)], nodes=range(8))
    n, a, e, groups = snapshot.node_count, snapshot._arcs._nnz(), snapshot.edge_count, 3
    base = n * (3 + n.bit_length()) + 4 * a + 4 * e + 32
    minimum = n + e + groups * groups + 2 * n + 8 * groups * groups
    structure = n + groups * (4 * n + 2 * a + groups)
    def forbidden(*args, **kwargs):
        raise AssertionError("CSR allocated before structural start admission")
    monkeypatch.setattr(sbm, "csr", forbidden)
    with pytest.raises(AnalysisError) as error:
        sbm.block_model(snapshot, groups, starts=1, max_work=base + minimum + structure - 1)
    assert error.value.code == "network_work_budget"


@pytest.mark.parametrize("directed", [False, True])
def test_large_star_uses_sparse_storage_and_all_absent_dyads(directed, monkeypatch):
    size = 10_000
    edges = [(0, node, 1) for node in range(1, size)]
    snapshot = graph(edges, nodes=range(size), directed=directed, max_memory_mb=64)
    original = torch.zeros
    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            assert shape != (size, size)
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(torch, "zeros", zeros)
    result = sbm.block_model(snapshot, 2, starts=1, seed=11, max_work=10_000_000)
    assert result["metadata"]["topology_edges"] == size - 1
    assert result["metadata"]["dyads"] == size * (size - 1) // (1 if directed else 2)
    assert result["metadata"]["log_likelihood"] == 0.
    assert result["metadata"]["converged"]
    assert result["membership"].block.value_counts().sort_values().tolist() == [1, size - 1]
    assert result["metadata"]["estimated_workspace_bytes"] < 20 * 1024**2


def test_work_preflight_rejects_before_csr_and_partial_iteration_raises_without_result(monkeypatch):
    snapshot = graph([(0, 1, 1), (2, 3, 1)], nodes=range(8))
    original = sbm.csr
    def forbidden(*args, **kwargs):
        raise AssertionError("CSR allocated before budget rejection")
    monkeypatch.setattr(sbm, "csr", forbidden)
    with pytest.raises(AnalysisError) as error:
        sbm.block_model(snapshot, 2, max_work=50)
    assert error.value.code == "network_work_budget"
    monkeypatch.setattr(sbm, "csr", original)
    full = sbm.block_model(snapshot, 2, starts=1)
    with pytest.raises(AnalysisError) as error:
        sbm.block_model(snapshot, 2, starts=1, max_work=full["metadata"]["work_used"] - 1)
    assert error.value.code == "network_work_budget"


def test_memory_preflight_accounts_for_csr_labels_groups_and_full_requested_trace(monkeypatch):
    snapshot = graph([(0, 1, 1)], nodes=range(10), max_memory_mb=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("CSR allocated before memory rejection")
    monkeypatch.setattr(sbm, "csr", forbidden)
    with pytest.raises(AnalysisError) as error:
        sbm.block_model(snapshot, 3, starts=64, max_iter=10_000)
    assert error.value.code == "network_memory_budget"


def test_cpu_buffers_override_meta_default_and_snapshot_stays_immutable():
    snapshot = graph([(0, 1, 2), (1, 2, 4)], nodes=range(8))
    labels, indices, weights = snapshot._labels, snapshot._edges.indices().clone(), snapshot._edges.values().clone()
    attrs = deepcopy(snapshot._node_attributes), deepcopy(snapshot._edge_attributes), deepcopy(snapshot._graph_attributes)
    original = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        result = sbm.block_model(snapshot, 2, seed=12)
    finally:
        torch.set_default_device(original)
    assert result["membership"].shape == (8, 2)
    assert snapshot._labels == labels
    assert torch.equal(snapshot._edges.indices(), indices)
    assert torch.equal(snapshot._edges.values(), weights)
    assert (snapshot._node_attributes, snapshot._edge_attributes, snapshot._graph_attributes) == attrs


@pytest.mark.parametrize("options", [
    {"groups": 0}, {"groups": True}, {"groups": 129}, {"groups": 9},
    {"groups": 2, "starts": 0}, {"groups": 2, "starts": 65},
    {"groups": 2, "max_iter": -1}, {"groups": 2, "max_iter": 0},
    {"groups": 2, "tol": -1}, {"groups": 2, "tol": math.nan},
    {"groups": 2, "seed": True}, {"groups": 2, "max_work": True},
    {"groups": 2, "values": "weight"}, {"groups": 2, "initial": [0] * 8},
    {"groups": 2, "initial": [0, 1, 0, 1, 0, 1, 0, True]},
    {"groups": 2, "initial": [0, 1]}, {"groups": 2, "initial": "00110011"},
    {"groups": 2, "initial": {i: i % 2 for i in range(1, 9)}},
    {"groups": 2, "initial": torch.ones((8, 1), dtype=torch.int64)},
])
def test_invalid_options_are_rejected(options):
    with pytest.raises(AnalysisError):
        sbm.block_model(graph(nodes=range(8)), **options)


def test_empty_node_set_and_ineligible_probabilistic_precision_are_rejected():
    with pytest.raises(AnalysisError):
        sbm.block_model(graph(), 1)
    snapshot = graph(nodes=range(3), directed=True)
    class FakeNetwork:
        node_count = 10**9
        edge_count = 0
        directed = True
        _arcs = snapshot._arcs
    with pytest.raises(AnalysisError) as error:
        sbm.block_model(FakeNetwork(), 1)
    assert error.value.code == "network_precision"


@pytest.mark.parametrize("directed", [False, True])
def test_physical_parquet_batches_and_given_block_permutations_preserve_profile(tmp_path, directed):
    import openecon as oe
    edges = [(0, 1, 2), (0, 1, 7), (1, 0, 1), (1, 2, 4),
             (2, 0, 1), (2, 3, 2), (3, 4, 3), (4, 4, 9), (8, 8, 0)]
    path = tmp_path / "block-edges.parquet"
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_parquet(path)
    dataset = oe.scan(path)
    snapshot = oe.network(dataset, weight="weight", batch_rows=2, nodes=[9], directed=directed)
    initial = {node: i % 3 for i, node in enumerate(snapshot._labels)}
    first = sbm.block_model(snapshot, 3, initial=initial, starts=1, max_iter=0)
    rotated = {node: (block + 1) % 3 for node, block in initial.items()}
    second = sbm.block_model(snapshot, 3, initial=rotated, starts=1, max_iter=0)
    assert_result(first, edges, list(snapshot._labels), directed, 3)
    assert membership(first) == membership(second)
    pd.testing.assert_frame_equal(first["blocks"], second["blocks"])
    assert first["metadata"]["log_likelihood"] == pytest.approx(second["metadata"]["log_likelihood"], abs=1e-14)
    assert first["membership"].attrs["network"]["actual_peak_batch_rows"] == 2
    assert first["membership"].attrs["method"] == "bernoulli_sbm"
    dataset.assert_unchanged()


@pytest.mark.parametrize("groups,max_iter", [(1, 100), (3, 0)])
def test_fixed_partition_likelihood_avoids_unneeded_csr(groups, max_iter, monkeypatch):
    snapshot = graph([(0, 1, 1)], nodes=range(8), directed=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed partition profile allocated unused adjacency")
    monkeypatch.setattr(sbm, "csr", forbidden)
    initial = [i % groups for i in range(8)] if max_iter == 0 else None
    result = sbm.block_model(snapshot, groups, initial=initial, starts=1, max_iter=max_iter)
    assert_result(result, [(0, 1, 1)], list(range(8)), True, groups)


@pytest.mark.parametrize("edges,dyads", [(1, 2**53), (2**53 - 1, 2**53), (7, 17), (0, 0), (0, 9), (9, 9)])
def test_profile_likelihood_is_stable_at_rare_edges_and_probability_boundaries(edges, dyads):
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 70
        if not dyads or edges in (0, dyads):
            expected = 0.
        else:
            probability = Decimal(edges) / Decimal(dyads)
            expected = float(Decimal(edges) * probability.ln() + Decimal(dyads - edges) * (1 - probability).ln())
    assert sbm._term(edges, dyads) == pytest.approx(expected, abs=2e-13)
