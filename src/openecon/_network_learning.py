"""CPU float64 mini-batch link embeddings and one-layer mean GraphSAGE.

Graph topology is never implicitly split: explicit dyad splits and node-label
splits define the training information. Explicit labeled zero dyads are the
negative observations; no unknown/held-out/future edge is sampled as negative.
Hamilton et al. (2017), https://arxiv.org/abs/1706.02216, Algorithm 1 (mean).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _label, _real
from openecon._network_model_common import ModelResult, Work, binary_graph
from openecon._network_latent_blocks import _sample


def _options(graph, device, seed, max_iter, batch_size, learning_rate):
    binary_graph(graph)
    if device != "cpu":
        _error(
            "unsupported_device",
            "Network learning currently supports tested CPU float64 only; CUDA/MPS are not substituted or silently emulated.",
        )
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    iterations = _integer(max_iter, "max_iter", 10_000)
    batch_size = _integer(batch_size, "batch_size", 100_000)
    rate = _real(learning_rate, "invalid_option", "learning_rate must be positive and finite.")
    if rate <= 0:
        _error("invalid_option", "learning_rate must be positive.")
    return seed, iterations, batch_size, rate


class EmbeddingResult(ModelResult):
    def predict(self, pairs, *, max_pairs=100_000):
        maximum = _integer(max_pairs, "max_pairs", 1_000_000)
        if not isinstance(pairs, (list, tuple)) or len(pairs) > maximum:
            _error("output_budget", "Provide a bounded list of typed source/target pairs.")
        meta = self["metadata"]
        labels = self["embedding"].node.tolist()
        index = {label: i for i, label in enumerate(labels)}
        dimensions = self["embedding"].shape[1] - 1
        if (
            4096 + 1024 * len(labels) * dimensions + 1024 * len(pairs)
            > meta["prediction_memory_budget"]
        ):
            _error("memory_budget", "Embedding prediction exceeds fitted memory allowance.")
        left = torch.tensor(self["embedding"].drop(columns="node").to_numpy(), dtype=torch.float64)
        right = torch.tensor(
            self["target_embedding"].drop(columns="node").to_numpy(), dtype=torch.float64
        )
        rows = []
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                _error("invalid_pairs", "Each dyad needs exactly two typed IDs.")
            a, b = _label(pair[0]), _label(pair[1])
            if a not in index or b not in index or a == b:
                _error("invalid_pairs", "Embedding dyads need distinct fitted nodes.")
            eta = float(left[index[a]] @ right[index[b]]) / math.sqrt(dimensions) + meta["bias"]
            p = float(torch.tensor(eta, dtype=torch.float64).sigmoid())
            rows.append((a, b, p))
        result = as_frame(pd.DataFrame(rows, columns=["source", "target", "probability"]))
        if len(rows):
            result.source, result.target = (
                pd.Series([r[0] for r in rows], dtype=object),
                pd.Series([r[1] for r in rows], dtype=object),
            )
        return result

    def save_json(self, path):
        """Export inference state with typed labels; opening a model never trains it."""
        payload = {
            "schema": "openecon-link-embedding-v1",
            "metadata": self["metadata"],
            "nodes": self["embedding"].node.tolist(),
            "source": self["embedding"].drop(columns="node").to_numpy().tolist(),
            "target": self["target_embedding"].drop(columns="node").to_numpy().tolist(),
        }
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path, *, max_bytes=16 * 1024**2):
        path = Path(path)
        limit = _integer(max_bytes, "max_bytes", 256 * 1024**2)
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            _error("output_budget", "Embedding state exceeds max_bytes before parsing.")
        try:
            payload = json.loads(raw)
            if payload["schema"] != "openecon-link-embedding-v1":
                raise ValueError()
            labels = [_label(x) for x in payload["nodes"]]
            left, right = (
                torch.tensor(payload["source"], dtype=torch.float64),
                torch.tensor(payload["target"], dtype=torch.float64),
            )
            meta = payload["metadata"]
            if (
                left.ndim != 2
                or left.shape != right.shape
                or left.shape[0] != len(labels)
                or len(set(labels)) != len(labels)
                or not bool(torch.isfinite(left).all())
                or not bool(torch.isfinite(right).all())
                or not 1 <= left.shape[1] <= 256
            ):
                raise ValueError()
            _real(meta["bias"], "invalid_result", "Saved bias must be finite.")
            _integer(meta["prediction_memory_budget"], "prediction_memory_budget", 2**63 - 1)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            from openecon.analysis_contracts import AnalysisError

            raise AnalysisError(
                "network_invalid_result", "Saved embedding state is invalid."
            ) from exc

        def frame(values):
            result = as_frame(
                pd.DataFrame(
                    values.numpy(), columns=[f"dimension_{k}" for k in range(left.shape[1])]
                )
            )
            result.insert(0, "node", pd.Series(labels, dtype=object))
            return result

        return cls(embedding=frame(left), target_embedding=frame(right), metadata=meta)


def network_embedding(
    graph,
    observations,
    *,
    dimensions=8,
    seed=0,
    max_iter=100,
    batch_size=1024,
    learning_rate=0.03,
    device="cpu",
    max_observations=100_000,
    max_work=500_000_000,
    timeout=120.0,
    cancelled=None,
):
    """Logistic source/target matrix-factorization embeddings on explicit labeled dyads.

    Every node identity is retained. Undirected source/target weights are tied.
    All labeled zero TRAIN dyads are used (explicit negative observations);
    no unlabeled edge, validation/test response, or graph topology enters the
    loss. This is link embedding, separate from supervised node GNN learning.
    """
    seed, iterations, batch_size, rate = _options(
        graph, device, seed, max_iter, batch_size, learning_rate
    )
    dimensions = _integer(dimensions, "dimensions", 256)
    maximum = _integer(max_observations, "max_observations", 1_000_000)
    count = len(observations) if isinstance(observations, pd.DataFrame) else 0
    if count > maximum:
        _error("observation_budget", "Embedding observations exceed max_observations.")
    n = graph.node_count
    work = Work(
        graph,
        workspace=32768 + 2048 * count + 2048 * n * dimensions + 2048 * batch_size * dimensions,
        max_work=max_work,
        timeout=timeout,
        cancelled=cancelled,
    )
    pairs, y, splits = _sample(graph, observations, maximum, work, binary=True)
    train = torch.tensor([i for i, s in enumerate(splits) if s == "train"], dtype=torch.int64)
    train_mean = float(y[train].mean())
    if train_mean in {0.0, 1.0}:
        _error(
            "unidentified_model",
            "Link embedding training needs both positive and explicit negative dyads.",
        )
    rng = torch.Generator(device="cpu").manual_seed(seed)
    left = (torch.randn((n, dimensions), dtype=torch.float64, generator=rng) * 0.1).requires_grad_()
    right = (
        (torch.randn((n, dimensions), dtype=torch.float64, generator=rng) * 0.1).requires_grad_()
        if graph.directed
        else left
    )
    bias = torch.tensor(
        math.log(train_mean / (1 - train_mean)), dtype=torch.float64, requires_grad=True
    )
    parameters = [left, *([right] if graph.directed else []), bias]
    optimizer = torch.optim.Adam(parameters, lr=rate)
    history = []
    for iteration in range(iterations):
        order = train[torch.randperm(len(train), generator=rng)]
        total = 0.0
        for start in range(0, len(order), batch_size):
            work.add(8 * (n * dimensions + batch_size * dimensions))
            ids = order[start : start + batch_size]
            optimizer.zero_grad()
            selected = pairs[ids]
            eta = (left[selected[:, 0]] * right[selected[:, 1]]).sum(1) / math.sqrt(
                dimensions
            ) + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                eta, y[ids], reduction="mean"
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append(total / len(train))
    frame = graph._frame(
        {f"dimension_{k}": left[:, k].detach().tolist() for k in range(dimensions)}
    )
    target = graph._frame(
        {f"dimension_{k}": right[:, k].detach().tolist() for k in range(dimensions)}
    )
    seen = set(pairs[train].reshape(-1).tolist())
    result = EmbeddingResult(
        embedding=frame,
        target_embedding=target,
        metadata={
            "model": "Logistic link embedding",
            "method": "mini-batch Adam",
            "nodes": n,
            "observations": len(train),
            "dimensions": dimensions,
            "directed": graph.directed,
            "bias": float(bias.detach()),
            "seed": seed,
            "iterations": iterations,
            "converged": False,
            "convergence": "fixed training budget; no convergence guarantee",
            "objective": history[-1],
            "loss_history": history,
            "batch_size": batch_size,
            "learning_rate": rate,
            "negative_sampling": "all explicit labeled zero training dyads; no unobserved or held-out dyads sampled",
            "split_counts": {s: splits.count(s) for s in set(splits)},
            "training_topology": "explicit training observations only; input graph is identity registry",
            "unobserved_train_nodes": [graph._labels[i] for i in range(n) if i not in seen],
            "uncertainty": "not estimated",
            "supported_devices": {"cpu": "float64", "cuda": "unsupported", "mps": "unsupported"},
            "prediction_memory_budget": graph._budget.limit,
            **work.metadata(),
        },
    )
    rows = []
    for split in ["train", "validation", "test"]:
        ids = [i for i, s in enumerate(splits) if s == split]
        if ids:
            selected = [
                (graph._labels[int(pairs[i, 0])], graph._labels[int(pairs[i, 1])]) for i in ids
            ]
            predicted = torch.tensor(
                result.predict(selected, max_pairs=len(ids)).probability.to_numpy(),
                dtype=torch.float64,
            ).clamp(1e-12, 1 - 1e-12)
            loss = float(
                -(y[ids] * predicted.log() + (1 - y[ids]) * torch.log1p(-predicted)).mean()
            )
            base = float(
                -(y[ids] * math.log(train_mean) + (1 - y[ids]) * math.log1p(-train_mean)).mean()
            )
            rows.append(
                (
                    split,
                    len(ids),
                    loss,
                    base,
                    float(((predicted >= 0.5) == y[ids]).to(torch.float64).mean()),
                )
            )
    result["evaluation"] = as_frame(
        pd.DataFrame(
            rows, columns=["split", "observations", "log_loss", "baseline_log_loss", "accuracy"]
        )
    )
    return result


def _node_table(graph, frame, required):
    if (
        not isinstance(frame, pd.DataFrame)
        or len(frame) != graph.node_count
        or any(list(frame.columns).count(c) != 1 for c in ["node", *required])
    ):
        _error(
            "invalid_nodes",
            "Node tables must cover every typed ID exactly once with unique required columns.",
        )
    indices = {}
    for position, raw in enumerate(frame.node):
        node = _label(raw)
        if node not in graph._index or node in indices:
            _error("invalid_nodes", "Unknown or duplicate node identity in node table.")
        indices[node] = position
    return frame.iloc[[indices[label] for label in graph._labels]]


def mean_features(x, nodes, neighbors, *, fanout=None, generator=None):
    """Self features concatenated with outgoing-neighbor mean; isolates get zero."""
    rows = []
    for node in nodes:
        selected = neighbors[node]
        if fanout is not None and len(selected) > fanout:
            selected = [
                selected[i]
                for i in torch.randperm(len(selected), generator=generator)[:fanout].tolist()
            ]
        mean = x[selected].mean(0) if selected else torch.zeros(x.shape[1], dtype=torch.float64)
        rows.append(torch.cat((x[node], mean)))
    return torch.stack(rows)


class GNNResult(ModelResult):
    def save_json(self, path):
        Path(path).write_text(json.dumps(self["state"], allow_nan=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path, *, max_bytes=16 * 1024**2):
        limit = _integer(max_bytes, "max_bytes", 256 * 1024**2)
        with Path(path).open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            _error("output_budget", "GNN state exceeds max_bytes before parsing.")
        try:
            state = json.loads(raw)
            cls._state(state)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            from openecon.analysis_contracts import AnalysisError

            raise AnalysisError("network_invalid_result", "Saved GNN state is invalid.") from exc
        return cls(
            state=state,
            metadata={"model": "One-layer mean GraphSAGE classifier", "method": "saved inference"},
        )

    @staticmethod
    def _state(state):
        if state["schema"] != "openecon-mean-graphsage-v1":
            raise ValueError("Unsupported GNN schema")
        features, classes = state["features"], state["classes"]
        if (
            not isinstance(features, list)
            or not 1 <= len(features) <= 256
            or any(not isinstance(x, str) or not x for x in features)
            or len(set(features)) != len(features)
            or not isinstance(classes, list)
            or not 2 <= len(classes) <= 128
            or len(set(_label(x) for x in classes)) != len(classes)
        ):
            raise ValueError("Invalid GNN feature/class identities")
        values = [
            torch.tensor(state[name], dtype=torch.float64)
            for name in ["location", "scale", "w1", "b1", "w2", "b2"]
        ]
        location, scale, w1, b1, w2, b2 = values
        d, c = len(features), len(classes)
        if (
            location.shape != (d,)
            or scale.shape != (d,)
            or bool((scale <= 0).any())
            or w1.ndim != 2
            or w1.shape[0] != 2 * d
            or not 1 <= w1.shape[1] <= 256
            or b1.shape != (w1.shape[1],)
            or w2.shape != (w1.shape[1], c)
            or b2.shape != (c,)
            or any(not bool(torch.isfinite(v).all()) for v in values)
        ):
            raise ValueError("Invalid GNN tensor geometry")
        return values

    def predict_graph(
        self, graph, features, *, batch_size=128, max_work=50_000_000, timeout=60.0, cancelled=None
    ):
        """Inductive full-neighbor inference on an explicitly supplied graph; no training."""
        binary_graph(graph)
        batch = _integer(batch_size, "batch_size", 100_000)
        try:
            location, scale, w1, b1, w2, b2 = self._state(self["state"])
        except (ValueError, KeyError, TypeError, RuntimeError):
            _error("invalid_result", "GNN inference state is invalid.")
        columns, classes = self["state"]["features"], self["state"]["classes"]
        features = _node_table(graph, features, columns)
        if set(features.columns) != {"node", *columns}:
            _error(
                "invalid_features", "Inference features must match the saved feature names exactly."
            )
        n, d = graph.node_count, len(columns)
        work = Work(
            graph,
            workspace=32768 + 2048 * n * (d + len(classes)) + graph.edge_count * (2048 + 16 * d),
            max_work=max_work,
            timeout=timeout,
            cancelled=cancelled,
        )
        try:
            x = torch.tensor(
                features.loc[:, columns].to_numpy(dtype="float64"), dtype=torch.float64
            )
        except (TypeError, ValueError):
            _error("invalid_features", "Inference features must be finite numeric columns.")
        x = (x - location) / scale
        if not bool(torch.isfinite(x).all()):
            _error("invalid_features", "Inference features/normalization exceed float64 range.")
        neighbors = [[] for _ in range(n)]
        for a, b in graph._edges.indices().T.tolist():
            neighbors[a].append(b)
            if not graph.directed:
                neighbors[b].append(a)
        rows = []
        with torch.no_grad():
            for start in range(0, n, batch):
                ids = list(range(start, min(start + batch, n)))
                work.add(
                    sum(len(neighbors[i]) for i in ids) * d
                    + len(ids) * w1.shape[1] * (d + len(classes))
                )
                design = mean_features(x, ids, neighbors)
                probabilities = (torch.relu(design @ w1 + b1) @ w2 + b2).softmax(1)
                for i, row in zip(ids, probabilities.tolist()):
                    rows.append(
                        (graph._labels[i], classes[max(range(len(row)), key=row.__getitem__)], *row)
                    )
        result = as_frame(
            pd.DataFrame(
                rows,
                columns=["node", "prediction", *[f"probability_{i}" for i in range(len(classes))]],
            )
        )
        result.node, result.prediction = (
            pd.Series([r[0] for r in rows], dtype=object),
            pd.Series([r[1] for r in rows], dtype=object),
        )
        result.attrs.update(
            classes=classes, inference="saved inductive full-neighbor mean", **work.metadata()
        )
        return result


def network_gnn(
    graph,
    features,
    labels,
    edges,
    *,
    hidden=16,
    fanout=16,
    seed=0,
    max_iter=100,
    batch_size=128,
    learning_rate=0.03,
    device="cpu",
    max_edges=100_000,
    max_work=500_000_000,
    timeout=120.0,
    cancelled=None,
):
    """One-layer mean GraphSAGE node classification with explicit information splits.

    Node table: node/label/split (train/validation/test). Feature table: node and
    finite numeric columns. Edge table: source/target/split; training messages
    require train edges with BOTH endpoints in train nodes. Validation inference
    uses train+validation edges; test inference uses train+test edges. Feature
    normalization uses training nodes only. Explicit edge splits encode temporal
    availability; the caller must not label future edges as train.
    CPU float64 only; fanout and minibatch bound aggregation buffers. Full
    neighbor means are used in reported inference (no stochastic dropout).
    """
    seed, iterations, batch_size, rate = _options(
        graph, device, seed, max_iter, batch_size, learning_rate
    )
    hidden = _integer(hidden, "hidden", 256)
    fanout = _integer(fanout, "fanout", 100_000)
    maximum = _integer(max_edges, "max_edges", 1_000_000)
    features = _node_table(graph, features, [])
    labels = _node_table(graph, labels, ["label", "split"])
    columns = [name for name in features.columns if name != "node"]
    if not columns or len(columns) > 256 or len(set(columns)) != len(columns):
        _error("invalid_features", "Use 1..256 unique numeric feature columns.")
    count = len(edges) if isinstance(edges, pd.DataFrame) else 0
    if count > maximum:
        _error("observation_budget", "GNN edges exceed max_edges.")
    n, d = graph.node_count, len(columns)
    work = Work(
        graph,
        workspace=32768
        + 2048 * n * (d + hidden)
        + count * (2048 + 16 * d)
        + 2048 * batch_size * (fanout + 1) * d,
        max_work=max_work,
        timeout=timeout,
        cancelled=cancelled,
    )
    try:
        x = torch.tensor(features.loc[:, columns].to_numpy(dtype="float64"), dtype=torch.float64)
    except (TypeError, ValueError):
        _error("invalid_features", "Node features must be finite numeric values.")
    if not bool(torch.isfinite(x).all()):
        _error("invalid_features", "Node features must be finite, with no missing covariates.")
    splits = labels.split.tolist()
    if any(not isinstance(s, str) or s not in {"train", "validation", "test"} for s in splits):
        _error("invalid_split", "Node splits are train, validation or test.")
    classes = []
    encoded = []
    train = [i for i, s in enumerate(splits) if s == "train"]
    for i in train:
        label = _label(labels.label.iloc[i])
        if label not in classes:
            classes.append(label)
    if not 2 <= len(classes) <= 128:
        _error("invalid_labels", "Train nodes must contain 2..128 observed classes.")
    for label in labels.label:
        label = _label(label)
        if label not in classes:
            _error("unsupported_class", "Held-out classes absent from training cannot be inferred.")
        encoded.append(classes.index(label))
    target = torch.tensor(encoded, dtype=torch.int64)
    # Reuse exact dyad/split validation with a dummy value: zero is structural,
    # not a supervised outcome and never enters a negative-sampling procedure.
    if not isinstance(edges, pd.DataFrame) or "split" not in edges:
        _error("invalid_edges", "GNN requires explicit source/target/split edge rows.")
    edge_frame = edges.loc[:, ["source", "target", "split"]].assign(value=0.0)
    pairs, _, edge_splits = _sample(graph, edge_frame, maximum, work)
    neighbor_sets = {split: [set() for _ in range(n)] for split in ["train", "validation", "test"]}
    for pair, edge_split in zip(pairs.tolist(), edge_splits):
        a, b = pair
        if edge_split == "train" and (splits[a] != "train" or splits[b] != "train"):
            _error(
                "topology_leakage",
                "A training edge touches a held-out node; label its availability explicitly.",
            )
        for split in neighbor_sets:
            if edge_split == "train" or edge_split == split:
                neighbor_sets[split][a].add(b)
                if not graph.directed:
                    neighbor_sets[split][b].add(a)
    neighbors = {s: [sorted(nodes) for nodes in rows] for s, rows in neighbor_sets.items()}
    location, scale = x[train].mean(0), x[train].std(0, unbiased=False).clamp_min(1e-12)
    x = (x - location) / scale
    if not bool(torch.isfinite(x).all()):
        _error("invalid_features", "Training feature normalization exceeds float64 range.")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    w1 = (
        torch.randn((2 * d, hidden), dtype=torch.float64, generator=rng) / math.sqrt(2 * d)
    ).requires_grad_()
    b1 = torch.zeros(hidden, dtype=torch.float64, requires_grad=True)
    w2 = (
        torch.randn((hidden, len(classes)), dtype=torch.float64, generator=rng) / math.sqrt(hidden)
    ).requires_grad_()
    b2 = torch.zeros(len(classes), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=rate)
    history = []
    for _ in range(iterations):
        order = [train[i] for i in torch.randperm(len(train), generator=rng).tolist()]
        total = 0.0
        for start in range(0, len(order), batch_size):
            work.add(8 * batch_size * ((fanout + hidden) * d + hidden * len(classes)))
            ids = order[start : start + batch_size]
            design = mean_features(x, ids, neighbors["train"], fanout=fanout, generator=rng)
            optimizer.zero_grad()
            logits = torch.relu(design @ w1 + b1) @ w2 + b2
            loss = torch.nn.functional.cross_entropy(logits, target[ids])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append(total / len(train))
    predictions, metrics = [], []
    majority = int(torch.bincount(target[train]).argmax())
    with torch.no_grad():
        for split in ["train", "validation", "test"]:
            ids = [i for i, s in enumerate(splits) if s == split]
            if not ids:
                continue
            probabilities = []
            for start in range(0, len(ids), batch_size):
                selected = ids[start : start + batch_size]
                work.add(
                    sum(len(neighbors[split][i]) for i in selected) * d
                    + len(selected) * hidden * (d + len(classes))
                )
                design = mean_features(x, selected, neighbors[split])
                p = (torch.relu(design @ w1 + b1) @ w2 + b2).softmax(1)
                probabilities.append(p)
                for i, row in zip(selected, p.tolist()):
                    predictions.append(
                        (
                            graph._labels[i],
                            split,
                            classes[max(range(len(row)), key=row.__getitem__)],
                            *row,
                        )
                    )
            p = torch.cat(probabilities)
            actual = target[ids]
            metrics.append(
                (
                    split,
                    len(ids),
                    float((p.argmax(1) == actual).to(torch.float64).mean()),
                    float((actual == majority).to(torch.float64).mean()),
                    float(-p[torch.arange(len(ids)), actual].clamp_min(1e-300).log().mean()),
                )
            )
    prediction_frame = as_frame(
        pd.DataFrame(
            predictions,
            columns=[
                "node",
                "split",
                "prediction",
                *[f"probability_{i}" for i in range(len(classes))],
            ],
        )
    )
    prediction_frame.node = pd.Series([r[0] for r in predictions], dtype=object)
    prediction_frame.prediction = pd.Series([r[2] for r in predictions], dtype=object)
    return GNNResult(
        predictions=prediction_frame,
        evaluation=as_frame(
            pd.DataFrame(
                metrics,
                columns=["split", "observations", "accuracy", "baseline_accuracy", "log_loss"],
            )
        ),
        state={
            "schema": "openecon-mean-graphsage-v1",
            "features": columns,
            "classes": classes,
            "location": location.tolist(),
            "scale": scale.tolist(),
            "w1": w1.detach().tolist(),
            "b1": b1.detach().tolist(),
            "w2": w2.detach().tolist(),
            "b2": b2.detach().tolist(),
        },
        metadata={
            "model": "One-layer mean GraphSAGE classifier",
            "method": "mini-batch supervised Adam",
            "nodes": n,
            "observations": len(train),
            "seed": seed,
            "iterations": iterations,
            "hidden": hidden,
            "fanout": fanout,
            "batch_size": batch_size,
            "learning_rate": rate,
            "loss_history": history,
            "objective": history[-1],
            "converged": False,
            "convergence": "fixed epoch budget; no convergence guarantee",
            "uncertainty": "not estimated",
            "split_counts": {s: splits.count(s) for s in set(splits)},
            "edge_split_counts": {s: edge_splits.count(s) for s in set(edge_splits)},
            "aggregation": "self + outgoing mean; isolated neighbor vector zero; training uniform fanout without replacement",
            "topology": "train edges restricted to train nodes; full per-split means only at inference",
            "normalization": "training-node features only",
            "negative_sampling": "not applicable to supervised node labels",
            "supported_devices": {"cpu": "float64", "cuda": "unsupported", "mps": "unsupported"},
            **work.metadata(),
        },
    )
