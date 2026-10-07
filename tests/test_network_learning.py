"""Split leakage, independent aggregation, fitted inference and identity exports."""

from itertools import combinations
import json
import random

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon._network_learning import mean_features


def graph(n=24):
    return oe.network([], nodes=range(n))


def embedding_sample(n=24):
    rng = random.Random(127)
    return pd.DataFrame(
        [
            (i, j, int((i < n // 2) == (j < n // 2)), "train" if rng.random() < 0.8 else "test")
            for i, j in combinations(range(n), 2)
        ],
        columns=["source", "target", "value", "split"],
    )


def test_embedding_holdout_baseline_prediction_and_saved_identity(tmp_path):
    df = embedding_sample()
    fit = oe.network_embedding(graph(), df, dimensions=2, seed=9, max_iter=80, batch_size=64)
    test = fit["evaluation"].loc[fit["evaluation"].split == "test"].iloc[0]
    assert test.log_loss < test.baseline_log_loss * 0.25
    assert test.accuracy > 0.95
    pairs = [(0, 1), (0, 13), (4, 5), (2, 1)]
    path = tmp_path / "embedding.json"
    fit.save_json(path)
    restored = oe.EmbeddingResult.load_json(path)
    assert restored.predict(pairs).equals(fit.predict(pairs))
    assert fit["embedding"].node.tolist() == list(range(24))
    assert "tabular" in fit.to_latex()
    changed = df.copy()
    changed.loc[changed.split == "test", "value"] = (
        1 - changed.loc[changed.split == "test", "value"]
    )
    other = oe.network_embedding(graph(), changed, dimensions=2, seed=9, max_iter=80, batch_size=64)
    assert fit["embedding"].equals(other["embedding"])
    assert fit["metadata"]["bias"] == other["metadata"]["bias"]


def test_independent_mean_aggregation_isolates_and_uniform_fanout():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]], dtype=torch.float64)
    neighbors = [[1, 2], [0], []]
    actual = mean_features(x, [0, 1, 2], neighbors)
    assert actual.tolist() == [[1, 2, 5.5, 7], [3, 4, 1, 2], [8, 10, 0, 0]]
    a = mean_features(x, [0], neighbors, fanout=1, generator=torch.Generator().manual_seed(8))
    b = mean_features(x, [0], neighbors, fanout=1, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert a[0, 2:].tolist() in [[3, 4], [8, 10]]


def supervised_fixture():
    n = 40
    features = pd.DataFrame(
        {
            "node": range(n),
            "x": [-1.0 if i % 2 else 1.0 for i in range(n)],
            "z": [(i % 7) / 10 for i in range(n)],
        }
    )
    labels = pd.DataFrame(
        {
            "node": range(n),
            "label": ["negative" if i % 2 else "positive" for i in range(n)],
            "split": ["train" if i < 28 else "validation" if i < 34 else "test" for i in range(n)],
        }
    )
    edges = pd.DataFrame(
        [
            (i, i + 2, "train" if i + 2 < 28 else "validation" if i + 2 < 34 else "test")
            for i in range(38)
        ],
        columns=["source", "target", "split"],
    )
    return graph(n), features, labels, edges


def test_gnn_holdout_no_gradient_leakage_and_saved_inductive_inference(tmp_path):
    g, features, labels, edges = supervised_fixture()
    fit = oe.network_gnn(g, features, labels, edges, seed=3, max_iter=35, batch_size=8)
    heldout = fit["evaluation"].loc[fit["evaluation"].split != "train"]
    assert heldout.accuracy.tolist() == [1.0, 1.0]
    assert (heldout.accuracy > heldout.baseline_accuracy).all()
    path = tmp_path / "gnn.json"
    fit.save_json(path)
    restored = oe.GNNResult.load_json(path)
    inference_graph = oe.network(edges[["source", "target"]], nodes=range(40))
    assert restored.predict_graph(inference_graph, features).equals(
        fit.predict_graph(inference_graph, features)
    )
    assert fit.predict_graph(inference_graph, features).prediction.tolist() == labels.label.tolist()
    changed_features = features.copy()
    changed_features.loc[labels.split != "train", "z"] += 10
    changed_labels = labels.copy()
    changed_labels.loc[labels.split != "train", "label"] = [
        "positive" if s == "negative" else "negative"
        for s in changed_labels.loc[labels.split != "train", "label"]
    ]
    changed_edges = edges.loc[edges.split == "train"].copy()
    other = oe.network_gnn(
        g, changed_features, changed_labels, changed_edges, seed=3, max_iter=35, batch_size=8
    )
    assert fit["state"] == other["state"]


def test_future_training_edge_guard_and_unsupported_device():
    g, features, labels, edges = supervised_fixture()
    edges.loc[edges.target >= 28, "split"] = "train"
    with pytest.raises(AnalysisError, match="held-out node"):
        oe.network_gnn(g, features, labels, edges)
    for device in ["cuda", "mps", "auto"]:
        with pytest.raises(AnalysisError, match="CPU float64"):
            oe.network_embedding(graph(), embedding_sample(), device=device)


def test_budget_stop_and_invalid_saved_geometry(tmp_path):
    g, features, labels, edges = supervised_fixture()
    for options in [{"max_work": 1}, {"cancelled": lambda: True}]:
        with pytest.raises(AnalysisError):
            oe.network_gnn(g, features, labels, edges, **options)
        with pytest.raises(AnalysisError):
            oe.network_embedding(graph(), embedding_sample(), **options)
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"schema": "openecon-mean-graphsage-v1", "features": [], "classes": []}),
        encoding="utf-8",
    )
    with pytest.raises(AnalysisError):
        oe.GNNResult.load_json(path)
    with pytest.raises(AnalysisError, match="before parsing"):
        oe.GNNResult.load_json(path, max_bytes=2)


def test_typed_node_ids_survive_embedding_export(tmp_path):
    labels = [1, "1", 2, "2"]
    g = oe.network([], nodes=labels)
    df = pd.DataFrame(
        [(1, "1", 1), (2, "2", 1), (1, 2, 0), ("1", "2", 0)], columns=["source", "target", "value"]
    )
    fit = oe.network_embedding(g, df, dimensions=2, max_iter=2)
    path = tmp_path / "typed.json"
    fit.save_json(path)
    restored = oe.EmbeddingResult.load_json(path)
    assert restored["embedding"].node.tolist() == labels
    assert restored.predict([(1, "1"), ("1", 2)]).source.tolist() == [1, "1"]
