"""Independent dense small-graph objective and refinement/aggregation oracles."""
import itertools
import json
import math
import random

import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import scan
from openecon.networks import network
from openecon._network_communities import (
    _Level, _Work, _local_move, _one_iteration, _refine, communities, modularity,
)


def build(edges, *, directed=False, nodes=None, batch_rows=2, scale=1.):
    return network([{"source": source, "target": target, "weight": weight * scale}
                    for source, target, weight in edges], weight="weight", directed=directed,
                   nodes=nodes, batch_rows=batch_rows)


def objective(edges, nodes, membership, directed, resolution=1.):
    """Independent quadratic definition, including diagonal/self-loop terms."""
    size = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = [[0.] * size for _ in range(size)]
    scale = max((weight for _, _, weight in edges), default=1.)
    for source, target, weight in edges:
        i, j = index[source], index[target]
        adjacency[i][j] += weight / scale
        if not directed:
            adjacency[j][i] += weight / scale
    total = math.fsum(value for row in adjacency for value in row)
    if not total:
        return 0.
    outgoing = [math.fsum(row) / total for row in adjacency]
    incoming = [math.fsum(adjacency[i][j] for i in range(size)) / total for j in range(size)]
    return math.fsum(adjacency[i][j] / total - resolution * outgoing[i] * incoming[j]
                     for i in range(size) for j in range(size) if membership[i] == membership[j])


def partitions(n):
    """Restricted-growth labels enumerate every disjoint partition once."""
    if not n:
        yield []
        return
    for values in itertools.product(range(n), repeat=n - 1):
        membership = [0, *values]
        if all(value <= 1 + max(membership[:i]) for i, value in enumerate(membership[1:], 1)):
            yield membership


def grouped(frame):
    return {frozenset(frame.node[frame.community == group]) for group in frame.community.unique()}


def connected(edges, group):
    if not group:
        return False
    reached = {next(iter(group))}
    while True:
        previous = set(reached)
        for source, target, weight in edges:
            if weight and source in group and target in group and (source in reached or target in reached):
                reached.update([source, target])
        if reached == previous:
            return reached == set(group)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("resolution", [.2, 1., 3.])
def test_all_small_partitions_independent_objective_with_loops_duplicates_and_isolate(directed, resolution):
    edges = [(1, "1", 2.), (1, "1", 1.), ("1", 1, 4.), (1, 1, .5),
             ("1", "b", 3.), ("b", "b", 2.)]
    nodes = [1, "1", "b", "isolate"]
    graph = build(edges, directed=directed, nodes=nodes)
    for membership in partitions(len(nodes)):
        expected = objective(edges, nodes, membership, directed, resolution)
        assert modularity(graph, membership, resolution) == pytest.approx(expected, abs=2e-15)
        mapping = dict(zip(nodes, [str(value) for value in membership]))
        assert modularity(graph, mapping, resolution) == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("directed", [False, True])
def test_aggregation_preserves_all_partitions_and_self_loop_null_weights(directed):
    edges = [(0, 0, 3.), (0, 1, 2.), (1, 0, 1.), (1, 2, 4.), (2, 3, 5.), (3, 3, 7.)]
    graph = build(edges, directed=directed, nodes=range(4))
    work = _Work(10_000_000)
    level = _Level.from_graph(graph, work)
    refinement = [0, 0, 1, 2]
    aggregate, _ = level.aggregate(refinement, work)
    assert aggregate.n == 3
    for membership in partitions(3):
        original = [membership[group] for group in refinement]
        expected = objective(edges, list(range(4)), original, directed, .7)
        assert aggregate.quality(membership, .7, work) == pytest.approx(expected, abs=2e-15)
        assert level.quality(original, .7, work) == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("directed", [False, True])
def test_local_move_gain_matches_brute_node_and_empty_community_oracle(directed):
    rng = random.Random(991)
    edges = [(i, j, rng.uniform(.1, 2.)) for i in range(5) for j in range(5) if rng.random() < .35]
    graph = build(edges, directed=directed, nodes=range(5))
    for initial in [[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 1, 2, 3, 4]]:
        work = _Work(10_000_000)
        level = _Level.from_graph(graph, work)
        for fast in [True, False]:
            result, _ = _local_move(level, initial, 1., 1e-12, random.Random(55), work, fast)
            final = objective(edges, list(range(5)), result, directed)
            assert final >= objective(edges, list(range(5)), initial, directed) - 2e-15
            for node in range(5):
                for target in range(6):
                    candidate = result.copy()
                    candidate[node] = target
                    assert objective(edges, list(range(5)), candidate, directed) <= final + 1e-12


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("bridge", [0., .005])
def test_real_refinement_escapes_node_local_trap_even_with_connected_weak_bridge(directed, bridge):
    edges = [(0, 1, 1.), (1, 2, 1.), (2, 0, 1.), (3, 4, 1.), (4, 5, 1.), (5, 3, 1.)]
    if bridge:
        edges.append((2, 3, bridge))
    graph = build(edges, directed=directed, nodes=range(6))
    initial = [0] * 6
    louvain, louvain_info = _one_iteration(graph, initial, "louvain", 1., 1e-12,
                                          random.Random(4), _Work(100000))
    leiden, detail = _one_iteration(graph, initial, "leiden", 1., 1e-12,
                                    random.Random(4), _Work(100000))
    assert louvain == initial and louvain_info["refinement_merges"] == 0
    assert len(set(leiden)) == 2
    assert leiden[:3] == [leiden[0]] * 3 and leiden[3:] == [leiden[3]] * 3
    assert detail["refinement_merges"] > 0 and detail["aggregations"] > 0
    assert objective(edges, list(range(6)), leiden, directed) > .49
    assert all(b >= a - 2e-15 for a, b in zip(detail["level_objectives"], detail["level_objectives"][1:]))


@pytest.mark.parametrize("directed", [False, True])
def test_refinement_retains_coarse_partition_and_eligible_groups_stay_connected(directed):
    edges = [(0, 1, 3.), (1, 2, 2.), (2, 0, 1.), (2, 3, .001), (3, 4, 3.), (4, 5, 2.),
             (5, 3, 1.), (0, 6, 7.), (6, 7, 8.)]
    graph = build(edges, directed=directed, nodes=range(8))
    coarse = [0, 0, 0, 0, 0, 0, 1, 1]
    for seed in range(12):
        work = _Work(100000)
        level = _Level.from_graph(graph, work)
        refined, merges = _refine(level, coarse, 1., random.Random(seed), work)
        assert merges >= 0
        for group in set(refined):
            vertices = {i for i in range(8) if refined[i] == group}
            assert len({coarse[i] for i in vertices}) == 1
            assert connected(edges, vertices)
        assert objective(edges, list(range(8)), refined, directed) >= objective(edges, list(range(8)), list(range(8)), directed) - 2e-15


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("method", ["louvain", "leiden"])
def test_final_determinism_seed_metadata_and_weight_scaling_batched_invariance(directed, method):
    edges = [(0, 1, 2.), (1, 2, 2.), (2, 0, 2.), (3, 4, 3.), (4, 5, 3.), (5, 3, 3.),
             (2, 3, .01), (5, 6, .01), (6, 6, .5)]
    reference = None
    for scale, rows in [(1., 1), (1e-100, 2), (1e100, 65536)]:
        graph = build(edges, directed=directed, nodes=range(8), batch_rows=rows, scale=scale)
        result = communities(graph, method=method, seed=717, tol=1e-12)
        assert result.attrs["converged"] and result.attrs["stable_iteration"]
        assert result.attrs["seed"] == 717 and result.attrs["method"] == method
        assert result.attrs["asymptotic_optimality_claimed"] is False
        assert 0 < result.attrs["work_used"] <= result.attrs["max_work"]
        assert result.attrs["modularity"] == pytest.approx(modularity(graph, result), abs=2e-15)
        history = result.attrs["objective_history"]
        assert all(b >= a - 2e-15 for a, b in zip(history, history[1:]))
        if method == "leiden":
            assert result.attrs["refinement_theta"] == .01
            assert all(connected(edges, group) for group in grouped(result))
        if reference is None:
            reference = grouped(result), result.attrs["modularity"]
        else:
            assert grouped(result) == reference[0]
            assert result.attrs["modularity"] == pytest.approx(reference[1], abs=2e-15)
        assert communities(graph, method=method, seed=717, tol=1e-12).equals(result)
        assert "\\begin{tabular}" in result.to_latex()
        assert json.loads(result.to_json())["community"]


@pytest.mark.parametrize("directed", [False, True])
def test_two_separated_triangles_matches_exhaustive_global_optimum_on_reference_only(directed):
    edges = [(0, 1, 1.), (1, 2, 1.), (2, 0, 1.), (3, 4, 1.), (4, 5, 1.), (5, 3, 1.), (2, 3, .005)]
    graph = build(edges, directed=directed, nodes=range(6))
    maximum = max(objective(edges, list(range(6)), membership, directed) for membership in partitions(6))
    for method in ["louvain", "leiden"]:
        result = communities(graph, method=method, seed=21)
        assert result.attrs["modularity"] == pytest.approx(maximum, abs=2e-15)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("method", ["louvain", "leiden"])
def test_seeded_small_sparse_partitions_stable_node_moves_and_leiden_connectivity(directed, method):
    for seed in range(12):
        rng = random.Random(4519 + seed)
        edges = [(i, j, rng.uniform(.05, 4.)) for i in range(8) for j in range(8)
                 if rng.random() < .19]
        graph = build(edges, directed=directed, nodes=range(8))
        result = communities(graph, method=method, seed=seed, tol=1e-12)
        membership = result.community.tolist()
        final = objective(edges, list(range(8)), membership, directed)
        assert result.attrs["modularity"] == pytest.approx(final, abs=2e-15)
        assert final >= objective(edges, list(range(8)), list(range(8)), directed) - 2e-15
        for node in range(8):
            for target in range(9):
                candidate = membership.copy()
                candidate[node] = target
                assert objective(edges, list(range(8)), candidate, directed) <= final + 1e-12
        if method == "leiden":
            assert all(connected(edges, group) for group in grouped(result))


def test_parquet_multibatch_partition_is_full_graph_not_head(tmp_path, monkeypatch):
    data = pd.DataFrame({"source": [0, 1, 2, 3, 4, 5, 2], "target": [1, 2, 0, 4, 5, 3, 3],
                         "weight": [2., 2., 2., 2., 2., 2., .01]})
    path = tmp_path / "edges.parquet"
    data.to_parquet(path, index=False, row_group_size=2)
    source = scan(path)
    monkeypatch.setattr(type(source), "head", lambda *args, **kwargs: pytest.fail("No head/collect path."))
    graph = network(source, weight="weight", batch_rows=1, nodes=range(7))
    result = communities(graph, seed=0)
    resident = communities(network(data, weight="weight", nodes=range(7)), seed=0)
    assert grouped(result) == grouped(resident)
    assert graph.metadata["input_rows"] == len(data) and graph.metadata["actual_peak_batch_rows"] == 1


def test_empty_isolates_and_self_loop_conventions():
    empty = network({"source": [], "target": []})
    assert modularity(empty, []) == 0.
    assert communities(empty).empty
    graph = network({"source": [], "target": []}, nodes=["b", 1, "1"])
    result = communities(graph)
    assert result.attrs["iterations"] == 0 and result.attrs["modularity"] == 0.
    assert dict(zip(result.node, result.community)) == {1: 0, "1": 1, "b": 2}
    loops = build([(0, 0, 2.), (1, 1, 3.)], nodes=range(2))
    assert modularity(loops, [0, 1]) == pytest.approx(1 - .4**2 - .6**2)
    assert modularity(loops, [0, 0]) == pytest.approx(0.)


def test_sum_overflow_safe_modularity_and_explicit_underflow_domain():
    edges = [(0, 1, 1e308), (1, 0, 1e308), (2, 2, 1e308)]
    graph = build(edges, directed=True, nodes=range(3))
    assert modularity(graph, [0, 0, 1]) == pytest.approx(4/9)
    assert communities(graph).attrs["modularity"] == pytest.approx(4/9)
    underflow = build([(0, 1, 1e308), (2, 3, 1e-308)])
    for operation in [lambda: modularity(underflow, [0, 0, 1, 1]), lambda: communities(underflow)]:
        with pytest.raises(AnalysisError) as error:
            operation()
        assert error.value.code == "network_precision"


@pytest.mark.parametrize("membership", [[0], [0, True], [0, math.nan], {"a": 0}, {"a": 0, "bad": 1},
                                        "00", {"a": 0, True: 1}, pd.DataFrame({"node": ["a", "a"], "community": [0, 1]})])
def test_strict_complete_membership_validation(membership):
    graph = build([("a", "b", 1.)])
    with pytest.raises(AnalysisError):
        modularity(graph, membership)


@pytest.mark.parametrize("membership", [torch.tensor(1), torch.tensor([[0, 1]]),
                                        torch.tensor([0., 1.]), torch.tensor([False, True]),
                                        torch.tensor([0j, 1j]), torch.empty(2, dtype=torch.int64, device="meta")])
def test_materialized_integer_tensor_membership_domain(membership):
    with pytest.raises(AnalysisError) as error:
        modularity(build([("a", "b", 1.)]), membership)
    assert error.value.code == "network_invalid_membership"


def test_numpy_buffer_membership_sequence_uses_native_objective_and_no_external_algorithm():
    graph = build([("a", "b", 1.), ("b", "b", 2.)])
    buffer = pd.Series([0, 1], dtype="int64").to_numpy()
    assert modularity(graph, buffer) == pytest.approx(modularity(graph, [0, 1]), abs=2e-15)
    assert modularity(graph, torch.tensor([0, 1])) == pytest.approx(modularity(graph, [0, 1]), abs=2e-15)


@pytest.mark.parametrize("options", [{"method": "label-propagation"}, {"seed": -1}, {"seed": True},
                                     {"max_iter": 0}, {"tol": 0}, {"tol": math.inf},
                                     {"resolution": 0}, {"resolution": math.nan}, {"max_work": True}])
def test_public_options_rejected(options):
    with pytest.raises(AnalysisError):
        communities(build([(0, 1, 1.)]), **options)


def test_work_iteration_and_preallocation_memory_guards(monkeypatch):
    graph = build([(0, 1, 1.), (1, 2, 1.), (2, 0, 1.)])
    for operation in [lambda: modularity(graph, [0, 0, 0], max_work=1),
                      lambda: communities(graph, max_work=1)]:
        with pytest.raises(AnalysisError) as error:
            operation()
        assert error.value.code == "network_work_budget"
    with pytest.raises(AnalysisError) as error:
        communities(graph, max_iter=1)
    assert error.value.code == "network_nonconvergence"
    graph._budget.limit = graph._base_bytes + 50
    def forbidden(*args, **kwargs):
        pytest.fail("Memory refusal must precede sparse allocation.")
    monkeypatch.setattr(torch, "sparse_coo_tensor", forbidden)
    for operation in [lambda: modularity(graph, [0, 0, 0]), lambda: communities(graph)]:
        with pytest.raises(AnalysisError) as error:
            operation()
        assert error.value.code == "network_memory_budget"


def test_public_cpu_scope_restores_meta_and_inference_context():
    graph = build([(0, 1, 1.), (1, 2, 1.)])
    with torch.device("meta"), torch.inference_mode():
        result = communities(graph)
        assert math.isfinite(modularity(graph, result))
        assert torch.empty(0).device.type == "meta" and torch.is_inference_mode_enabled()
