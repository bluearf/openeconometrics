"""Native code-panel example for MARKET-61/62/63/66; synthetic data only."""

from itertools import combinations
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import pandas as pd
import openecon as oe

display = globals().get("display", print)

# Exact independent-edge ERGM; integer 1 and string "1" are distinct nodes.
g = oe.network(
    [{"source": 1, "target": "1"}, {"source": "1", "target": "b"}], nodes=[1, "1", "b", "c"]
)
erg = oe.ergm(g)
assert erg["metadata"]["converged"]
assert abs(erg["coefficients"].coefficient.iloc[0] - math.log(0.5)) < 1e-6
chain = erg.simulate(draws=150, burn_in=100, thin=5, seed=12)
assert len(chain["graphs"]) == 150

# Fully observed directed two-actor independent panel replicates with explicit times.
actors = oe.network([], nodes=[1, "1"], directed=True)
times = [0.0, 0.6, 1.8, 3.0]
panels = [
    oe.simulate_saom(actors, times, rate=1.1, theta=[-0.7], seed=s)["panel"] for s in range(240)
]
actor_fit = oe.saom(panels, times, max_iter=70)
assert actor_fit["metadata"]["converged"]
assert abs(actor_fit["metadata"]["opportunity_rate"] - 1.1) < 0.25

# Separate real-valued Gaussian and conditional Bernoulli-simplex models.
n = 20
rng = random.Random(719)
registry = oe.network([], nodes=range(n))
real = pd.DataFrame(
    [
        (
            i,
            j,
            rng.gauss(-3 if (i < 10) == (j < 10) else 3, 0.4),
            "train" if rng.random() < 0.8 else "test",
        )
        for i, j in combinations(range(n), 2)
    ],
    columns=["source", "target", "value", "split"],
)
gaussian = oe.gaussian_block_model(
    registry, 2, real, seed=2, starts=12, max_iter=50, max_work=500_000_000
)
assert (
    gaussian["evaluation"].iloc[-1].log_likelihood
    > gaussian["evaluation"].iloc[-1].baseline_log_likelihood
)
binary = pd.DataFrame(
    [
        (i, j, int((i < 10) == (j < 10)), "train" if rng.random() < 0.8 else "test")
        for i, j in combinations(range(n), 2)
    ],
    columns=real.columns,
)
mixed = oe.mixed_membership_block_model(registry, 2, binary, seed=8, max_iter=150, batch_size=64)
assert mixed["membership"].drop(columns="node").sum(axis=1).sub(1).abs().max() < 1e-12
embedding = oe.network_embedding(registry, binary, dimensions=2, seed=9, max_iter=80, batch_size=64)
assert embedding["evaluation"].iloc[-1].accuracy > 0.95

# One-layer sampled mean-GraphSAGE. Held-out topology has an explicit mask.
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
classifier = oe.network_gnn(
    oe.network([], nodes=range(n)), features, labels, edges, seed=3, max_iter=35, batch_size=8
)
assert classifier["evaluation"].accuracy.tolist() == [1.0, 1.0, 1.0]
with tempfile.TemporaryDirectory(prefix="openecon-model-example-") as temporary:
    embedding_path = Path(temporary) / "embedding.json"
    embedding.save_json(embedding_path)
    restored_embedding = oe.EmbeddingResult.load_json(embedding_path)
    assert restored_embedding.predict([(0, 1), (0, 11)]).equals(
        embedding.predict([(0, 1), (0, 11)])
    )
    gnn_path = Path(temporary) / "gnn.json"
    classifier.save_json(gnn_path)
    restored_gnn = oe.GNNResult.load_json(gnn_path)
    inference = oe.network(edges[["source", "target"]], nodes=range(n))
    assert restored_gnn.predict_graph(inference, features).equals(
        classifier.predict_graph(inference, features)
    )
for result in [erg, actor_fit, gaussian, mixed, embedding, classifier]:
    display(result)
display(embedding["evaluation"])
display(classifier["evaluation"])
print(
    "NETWORK_MODELS:"
    + json.dumps(
        dict(
            exact_ergm=True,
            ergm_gibbs_seeded=True,
            saom_parameter_recovery=True,
            gaussian_holdout_beats_baseline=True,
            mixed_simplex=True,
            embedding_holdout_accuracy=float(embedding["evaluation"].iloc[-1].accuracy),
            gnn_holdout_accuracy=float(classifier["evaluation"].iloc[-1].accuracy),
            saved_inference_equal=True,
            typed_actor_ids=actor_fit["metadata"]["actor_ids"],
            frozen=bool(getattr(sys, "frozen", False)),
            sdk_from_bundle=bool(
                getattr(sys, "frozen", False)
                and Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
            ),
            device="cpu",
            dtype="float64",
            cuda_verified=False,
        ),
        allow_nan=False,
    )
)
