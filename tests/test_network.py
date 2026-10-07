"""Independent small-graph references; no NetworkX or statistical solver oracle."""
import json
import math

import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import scan
from openecon.networks import network


def values(frame, column):
    return dict(zip(frame.node, frame[column]))


def stationary(edges, nodes, directed, damping=.85, personalization=None):
    """Independent dense Gaussian elimination of the small Markov system."""
    labels = list(dict.fromkeys([*nodes, *(x for edge in edges for x in edge[:2])]))
    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    weights = [[0.] * n for _ in range(n)]
    for source, target, weight in edges:
        i, j = index[source], index[target]
        weights[i][j] += weight
        if not directed and i != j:
            weights[j][i] += weight
    p = [personalization.get(label, 0.) if personalization is not None else 1. for label in labels]
    p = [value / sum(p) for value in p]
    transition = [[value / sum(row) for value in row] if sum(row) else p[:] for row in weights]
    matrix = [[float(i == j) - damping * transition[j][i] for j in range(n)]
              + [(1 - damping) * p[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(matrix[i][col]))
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        scale = matrix[col][col]
        matrix[col] = [x / scale for x in matrix[col]]
        for row in range(n):
            if row != col:
                scale = matrix[row][col]
                matrix[row] = [a - scale * b for a, b in zip(matrix[row], matrix[col])]
    return dict(zip(labels, (row[-1] for row in matrix)))


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("batch_rows", [1, 2, 3, 65536])
def test_weighted_duplicates_reverse_loops_zero_and_missing(directed, batch_rows):
    data = {"source": ["a", "a", "b", "b", "c", "a"],
            "target": ["b", "b", "a", "b", "d", None], "w": [2., 3., 4., 7., 0., 2.]}
    graph = network(data, weight="w", directed=directed, nodes=["isolate"],
                    missing="drop", batch_rows=batch_rows)
    assert graph.node_count == 5
    assert graph.edge_count == (3 if directed else 2)
    assert graph.metadata["input_rows"] == 6
    assert graph.metadata["missing_rows_dropped"] == 1
    assert graph.metadata["zero_weight_rows_dropped"] == 1
    assert graph.metadata["duplicate_edge_rows_aggregated"] == (1 if directed else 2)
    degree = graph.degree()
    if directed:
        assert values(degree, "in_degree") == {"isolate": 0, "a": 1, "b": 2, "c": 0, "d": 0}
        assert values(degree, "out_strength")["b"] == 11
        assert values(degree, "in_strength")["b"] == 12
    else:
        assert values(degree, "degree") == {"isolate": 0, "a": 1, "b": 3, "c": 0, "d": 0}
        assert values(degree, "strength")["b"] == 23
    component = values(graph.components(), "component")
    assert component["a"] == component["b"]
    assert len(set(component.values())) == 4
    result = graph.pagerank(tol=1e-12)
    expected = stationary([("a", "b", 2), ("a", "b", 3), ("b", "a", 4), ("b", "b", 7)],
                          ["isolate", "c", "d"], directed)
    assert values(result, "pagerank") == pytest.approx(expected, abs=2e-12)
    assert result.attrs["converged"] and result.attrs["error_bound_estimate"] <= 1e-12
    assert result.pagerank.sum() == pytest.approx(1.)


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("damping", [0., .5, .85, .97])
def test_personalization_weighted_and_dangling_stationary_oracle(directed, damping):
    edges = [("a", "b", 2.), ("a", "c", 1.), ("b", "c", 4.), ("c", "c", .5)]
    graph = network([dict(source=s, target=t, w=w) for s, t, w in edges],
                    weight="w", directed=directed, nodes=["sink", "isolate"])
    p = {"a": 2., "c": 1., "sink": 4.}
    expected = stationary(edges, ["sink", "isolate"], directed, damping, p)
    assert values(graph.pagerank(damping=damping, personalization=p, max_iter=2000, tol=1e-12),
                  "pagerank") == pytest.approx(expected, abs=2e-12)


def test_bfs_vs_aggregate_weight_dijkstra_direction_and_infinity():
    edges = {"source": ["a", "a", "b", "b", "c"], "target": ["b", "c", "c", "d", "d"],
             "w": [1., 10., 2., 12., 3.]}
    graph = network(edges, weight="w", directed=True, nodes=["i"])
    assert values(graph.shortest_paths("a"), "distance") == {"i": math.inf, "a": 0., "b": 1., "c": 3., "d": 6.}
    assert math.isinf(values(graph.shortest_paths("d"), "distance")["a"])
    unweighted = network(edges, directed=True, nodes=["i"])
    assert values(unweighted.shortest_paths("a"), "distance")["d"] == 2
    undirected = network(edges, weight="w")
    assert values(undirected.shortest_paths("d"), "distance")["a"] == 6
    assert graph.shortest_paths("a").attrs["algorithm"] == "Dijkstra"
    assert unweighted.shortest_paths("a").attrs["algorithm"] == "BFS"


def test_cycle_isolates_empty_type_identity_and_publication_output():
    cycle = network({"source": [0, 1, 2, 3], "target": [1, 2, 3, 0]}, directed=True)
    assert cycle.pagerank().pagerank.tolist() == pytest.approx([.25] * 4)
    graph = network({"source": [1], "target": ["1"]}, nodes=["isolated"])
    assert graph.node_count == 3 and values(graph.degree(), "degree")[1] == 1
    assert values(graph.degree(), "degree")["1"] == 1
    assert "\\begin{table}" in str(graph.summary().to_latex(index=False, caption="Network summary"))
    empty = network({"source": [], "target": []})
    assert empty.degree().empty and empty.components().empty and empty.pagerank().empty
    assert empty.to_plot_data()["sampled"] is False
    isolated = network([], nodes=["a", "b"])
    assert isolated.pagerank().pagerank.tolist() == pytest.approx([.5, .5])


def test_physical_parquet_batch_invariance_and_no_collect(tmp_path, monkeypatch):
    frame = pd.DataFrame({"source": ["a", "b", "a", "c", "d", "b"],
                          "target": ["b", "a", "c", "c", "e", "d"], "w": [1., 2., 3., 4., 0., 2.]})
    path = tmp_path / "edges.parquet"
    frame.to_parquet(path, row_group_size=2)
    source = scan(path)
    monkeypatch.setattr(source, "head", lambda *args, **kwargs: pytest.fail("No resident preview is used"))
    reference = network(frame, weight="w", nodes=["z"], batch_rows=65536)
    for rows in [1, 2, 3, 10]:
        actual = network(source, weight="w", nodes=["z"], batch_rows=rows)
        assert actual.metadata["actual_peak_batch_rows"] <= rows
        assert actual.metadata["input_sha256"] == reference.metadata["input_sha256"]
        pd.testing.assert_frame_equal(actual.degree(), reference.degree(), check_flags=False)
        assert values(actual.pagerank(), "pagerank") == pytest.approx(values(reference.pagerank(), "pagerank"))


def test_plot_caps_are_induced_and_deterministic_not_analysis_sampling():
    edges = [{"source": 0, "target": i} for i in range(1, 12)] + [{"source": 2, "target": 3}]
    a = network(edges, nodes=[20])
    b = network(list(reversed(edges)), nodes=[20])
    before = a.pagerank()
    plot = a.to_plot_data(max_nodes=3, max_edges=1, seed=7)
    assert plot["node_count"] == 13 and plot["edge_count"] == 12
    assert plot["shown_node_count"] == 3 and plot["shown_edge_count"] == 1 and plot["sampled"]
    assert [item["label"] for item in plot["nodes"]] == ["0", "2", "3"]
    selected = {item["id"] for item in plot["nodes"]}
    assert all(edge["source"] in selected and edge["target"] in selected for edge in plot["edges"])
    assert [item["label"] for item in b.to_plot_data(max_nodes=3, max_edges=1)["nodes"]] == ["0", "2", "3"]
    assert a.to_plot_data(max_nodes=3, max_edges=1, seed=0) == plot
    assert values(a.pagerank(), "pagerank") == values(before, "pagerank")
    json.dumps(plot, allow_nan=False)


@pytest.mark.parametrize("label", [True, False, 1.0, [], {}, "", " ", "a\x00b", "a\x85b", "x" * 4097, "é" * 2049, 1 << 256])
def test_label_guards(label):
    with pytest.raises(AnalysisError, match="[Nn]ode|[Ii]nteger"):
        network({"source": [label], "target": ["b"]})


@pytest.mark.parametrize("weight", [-1., math.inf, -math.inf, True, "1", 1j, 10**400])
def test_invalid_weights(weight):
    with pytest.raises(AnalysisError) as error:
        network({"source": ["a"], "target": ["b"], "w": [weight]}, weight="w")
    assert error.value.code == "network_invalid_weight"


def test_missing_modes_zero_nodes_and_overflow_refusal():
    with pytest.raises(AnalysisError, match="missing"):
        network({"source": [None], "target": ["b"]})
    with pytest.raises(AnalysisError, match="missing"):
        network({"source": ["a"], "target": ["b"], "w": [math.nan]}, weight="w")
    graph = network({"source": [None, "a"], "target": ["b", "z"], "w": [1., 0.]}, weight="w", missing="drop")
    assert graph.node_count == 2 and graph.edge_count == 0
    assert graph.metadata["missing_rows_dropped"] == graph.metadata["zero_weight_rows_dropped"] == 1
    with pytest.raises(AnalysisError, match="overflow"):
        network({"source": ["a", "a"], "target": ["b", "b"], "w": [1e308, 1e308]}, weight="w")


def test_shortest_path_overflow_walks_are_distinct_from_overflow_shortest_distance():
    cycle = network({"source": ["a", "b"], "target": ["b", "a"], "w": [1e308, 1e308]},
                    weight="w", directed=True)
    assert values(cycle.shortest_paths("a"), "distance") == {"a": 0., "b": 1e308}
    alternate = network({"source": ["a", "b", "a"], "target": ["b", "c", "c"],
                         "w": [1e308, 1e308, 1.5e308]}, weight="w", directed=True)
    assert values(alternate.shortest_paths("a"), "distance")["c"] == 1.5e308
    overflow_only = network({"source": ["a", "b"], "target": ["b", "c"], "w": [1e308, 1e308]},
                           weight="w", directed=True)
    with pytest.raises(AnalysisError) as error:
        overflow_only.shortest_paths("a")
    assert error.value.code == "network_precision"


def test_count_plot_and_scaled_pagerank_do_not_need_representable_strength():
    graph = network({"source": ["a", "a"], "target": ["b", "c"], "w": [1e308, 1e308]},
                    weight="w", directed=True)
    with pytest.raises(AnalysisError) as error:
        graph.degree()
    assert error.value.code == "network_precision"
    assert [node["degree"] for node in graph.to_plot_data()["nodes"]] == [2, 1, 1]
    assert graph.pagerank().pagerank.sum() == pytest.approx(1.)


def test_column_series_use_position_and_records_are_lazy():
    graph = network({"source": pd.Series(["a", "b"], index=[10, 20]),
                     "target": pd.Series(["b", "c"], index=[30, 40])}, batch_rows=1)
    streamed = network(({"source": source, "target": target} for source, target in [("a", "b"), ("b", "c")]),
                       batch_rows=1)
    assert values(graph.degree(), "degree") == values(streamed.degree(), "degree") == {"a": 1, "b": 2, "c": 1}


@pytest.mark.parametrize("options", [{"batch_rows": True}, {"batch_rows": 0}, {"max_memory_mb": 0},
                                     {"max_memory_mb": math.inf}, {"directed": 1}, {"missing": []},
                                     {"source": "target"}])
def test_constructor_options(options):
    with pytest.raises(AnalysisError):
        network({"source": ["a"], "target": ["b"]}, **options)


def test_memory_guard_precedes_tensor_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A refused import must not create a sparse tensor")
    monkeypatch.setattr(torch, "sparse_coo_tensor", forbidden)
    with pytest.raises(AnalysisError) as error:
        network({"source": ["a"], "target": ["b"]}, max_memory_mb=.001)
    assert error.value.code == "network_memory_budget"


def test_operation_budget_guard_and_nonconvergence_are_explicit():
    graph = network([], nodes=list(range(200)), max_memory_mb=.25)
    with pytest.raises(AnalysisError) as error:
        graph.to_plot_data()
    assert error.value.code == "network_memory_budget"
    graph = network({"source": ["a", "a", "b"], "target": ["b", "c", "c"]}, directed=True)
    with pytest.raises(AnalysisError) as error:
        graph.pagerank(max_iter=1, tol=1e-15)
    assert error.value.code == "network_nonconvergence"
    for options in [{"damping": 1}, {"tol": 0}, {"max_iter": False}, {"personalization": {"q": 1}},
                    {"personalization": {"a": 0}}, {"device": "mps"}]:
        with pytest.raises(AnalysisError):
            graph.pagerank(**options)
    with pytest.raises(AnalysisError):
        graph.shortest_paths("unknown")


def test_cpu_scope_restores_meta_and_inference_context():
    with torch.device("meta"), torch.inference_mode():
        graph = network({"source": ["a"], "target": ["b"]})
        assert graph._edges.device.type == "cpu"
        assert graph.pagerank().pagerank.sum() == pytest.approx(1.)
        assert values(graph.shortest_paths("a"), "distance")["b"] == 1.
        assert torch.empty(0).device.type == "meta" and torch.is_inference_mode_enabled()
