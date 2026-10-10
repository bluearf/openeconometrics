"""Scientific and adversarial tests for the declared resident CART stage."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from openecon.econometrics.supervised import common as c
from openecon.econometrics.supervised import cart as module
from openecon.econometrics.supervised.cart import (
    CartState,
    CartQueryState,
    cart,
    cart_predict,
    cart_restore,
    cart_prediction_restore,
)
from openecon.econometrics.supervised.split import prediction_split, split_restore


def frame(task="regression", n=70):
    rng = np.random.default_rng(683)
    x = rng.normal(size=(n, 3))
    f = pd.DataFrame(
        x,
        columns=["a", "b", "c"],
        index=pd.Index([f"r{i // 2}" for i in range(n)], name="physical label"),
    )
    f["y"] = (
        x[:, 0] + 0.3 * rng.normal(size=n)
        if task == "regression"
        else np.where(x[:, 0] > 0.4, "C", np.where(x[:, 1] > 0, "B", "A"))
    )
    f["w"] = rng.uniform(0.3, 3, size=n)
    return f


def fit(task="regression", **kwargs):
    f = frame(task)
    split = prediction_split(f, roles=["train"] * 50 + ["validation"] * 10 + ["test"] * 10)
    return cart(
        f,
        outcome="y",
        features=["a", "b", "c"],
        weights="w",
        split=split,
        task=task,
        min_leaf=3,
        max_depth=4,
        max_nodes=31,
        **kwargs,
    )


def rehash(state):
    state.pop("digest", None)
    c.seal(state)
    return state


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_exhaustive_node_objectives_and_canonical_pruning(task):
    result = fit(task)
    state = result.attrs["state"]
    x, y, w = (
        np.array(state["source"]["x"]),
        np.array(state["source"]["y"]),
        np.array(state["source"]["weights"]),
    )
    classes = state["classes"]
    total = sum(w[state["split"]["positions"]["train"]])
    unit = state["normalization"]["outcome_scale"]

    def impurity(rows):
        if classes:
            mass = sum(w[rows])
            counts = np.array([sum(w[rows][y[rows] == v]) for v in classes])
            return (mass - np.dot(counts, counts) / mass) / total
        mean = np.dot(w[rows], y[rows]) / sum(w[rows])
        return np.dot(w[rows], (y[rows] - mean) ** 2) / total / unit**2

    for node in state["nodes"]:
        rows = node["positions"]
        assert node["impurity"] == pytest.approx(impurity(rows), rel=2e-12, abs=1e-15)
        if node["feature"] is None:
            continue
        candidates = []
        for j in range(3):
            values = sorted(set(x[rows, j]))
            for left, right in zip(values, values[1:]):
                threshold = left / 2 + right / 2
                a = [i for i in rows if x[i, j] <= threshold]
                b = [i for i in rows if x[i, j] > threshold]
                if min(len(a), len(b)) >= 3:
                    candidates.append((impurity(rows) - impurity(a) - impurity(b), j, threshold))
        best = max(candidates, key=lambda v: (v[0], -v[1], -v[2]))
        assert node["split_gain"] == pytest.approx(best[0], rel=3e-11, abs=2e-15)
        assert node["feature"] == best[1]
        assert node["threshold"] == best[2]
    path = state["derived"]["path"]
    assert path[-1]["leaf_count"] == 1
    assert all(a["leaf_count"] > b["leaf_count"] for a, b in zip(path, path[1:]))
    assert all(a["alpha"] <= b["alpha"] + 1e-14 for a, b in zip(path, path[1:]))
    assert path[-1]["risk"] == state["nodes"][0]["risk"]
    if classes:
        assert state["nodes"][0]["risk"] != pytest.approx(state["nodes"][0]["impurity"])


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_restore_predict_never_calls_grow(task, monkeypatch):
    result = fit(task, select="validation")
    monkeypatch.setattr(module, "_grow", lambda *a: pytest.fail("saved state refit called"))
    saved = result.to_json()
    restored = cart_restore(saved)
    assert restored.attrs == result.attrs
    assert restored["predictions"].index.equals(frame(task).index)
    query = cart_predict(restored, frame(task), outcome="y", weights="w")
    assert query.attrs["prediction"] == result.attrs["state"]["derived"]["prediction"]
    typed = CartState(payload=result.attrs["state"])
    assert CartState.model_validate_json(typed.model_dump_json()).payload == typed.payload
    assert copy.deepcopy(typed).payload == typed.payload
    assert typed.model_copy(deep=True).payload == typed.payload


def test_training_only_transforms_and_test_labels():
    f = frame()
    f.loc[f.index[0], "a"] = np.nan
    split = prediction_split(f, roles=["train"] * 50 + ["validation"] * 10 + ["test"] * 10)
    kwargs = dict(
        outcome="y",
        features=["a", "b", "c"],
        split=split,
        weights="w",
        min_leaf=3,
        max_depth=4,
        max_nodes=31,
        impute="mean",
        standardize="population",
        select="validation",
    )
    original = cart(f, **kwargs).attrs["state"]
    changed = f.copy()
    changed.iloc[60:, changed.columns.get_loc("y")] = 9e7
    changed.iloc[60:, changed.columns.get_loc("a")] = -9e8
    alternate = cart(changed, **kwargs).attrs["state"]
    assert original["nodes"] == alternate["nodes"]
    assert original["transform"] == alternate["transform"]
    assert original["derived"]["selected_stage"] == alternate["derived"]["selected_stage"]
    observed = f["a"].iloc[:50].notna()
    mean = np.average(f["a"].iloc[:50][observed], weights=f["w"].iloc[:50][observed])
    assert original["transform"]["imputation_means"][0] == pytest.approx(mean)


def test_weight_scale_and_tiny_feature_response_units():
    f = frame()
    split = prediction_split(f, roles=["train"] * len(f))
    kwargs = dict(
        outcome="y",
        features=["a", "b", "c"],
        split=split,
        weights="w",
        min_leaf=3,
        max_depth=4,
        max_nodes=31,
    )
    original = cart(f, **kwargs).attrs["state"]
    scaled = f.copy()
    scaled["w"] *= 1e-120
    scaled["a"] *= 1e-120
    scaled["y"] *= 1e-120
    alternate = cart(scaled, **kwargs).attrs["state"]
    assert [(n["feature"], n["positions"]) for n in original["nodes"]] == [
        (n["feature"], n["positions"]) for n in alternate["nodes"]
    ]
    assert [p["cp"] for p in original["derived"]["path"]] == pytest.approx(
        [p["cp"] for p in alternate["derived"]["path"]], rel=2e-12
    )
    cart_restore(alternate)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(70, name="range"),
        pd.date_range("2024-01-01", periods=70, tz="Europe/Istanbul", name="date"),
        pd.CategoricalIndex(
            ["a", "b"] * 35, categories=["b", "a", "unused"], ordered=True, name="category"
        ),
        pd.MultiIndex.from_arrays([["a", "b"] * 35, np.arange(70)], names=["group", "physical"]),
    ],
)
def test_typed_index_roundtrip(index):
    f = frame()
    f.index = index
    s = prediction_split(f)
    assert c.restore_index(split_restore(s).payload["index"], 70).equals(index)
    r = cart(f, outcome="y", features=["a"], split=s)
    pd.testing.assert_index_equal(cart_restore(r.to_json())["predictions"].index, index, exact=True)


@pytest.mark.parametrize("strategy", ["iid", "group", "chronological"])
def test_split_determinism_atomicity_and_rng(strategy):
    f = frame()
    f["group"] = [j // 3 for j in range(70)]
    f["time"] = [2**53 + j // 3 for j in range(70)]
    kwargs = (
        {"group": "group"}
        if strategy == "group"
        else {"time": "time"}
        if strategy == "chronological"
        else {}
    )
    before = torch.random.get_rng_state().clone()
    a = prediction_split(f, strategy=strategy, seed=19, **kwargs)
    b = prediction_split(f, strategy=strategy, seed=19, **kwargs)
    assert torch.equal(before, torch.random.get_rng_state())
    assert a.payload == b.payload
    assert sorted(i for rows in a.payload["positions"].values() for i in rows) == list(range(70))
    if strategy != "iid":
        units = f[kwargs.get("group", kwargs.get("time"))].tolist()
        assert all(
            len({a.payload["roles"][i] for i, actual in enumerate(units) if actual == unit}) == 1
            for unit in set(units)
        )


def test_split_explicit_leakage_and_changed_source_role():
    f = pd.DataFrame(
        {
            "x": range(6),
            "y": range(6),
            "g": [0, 0, 1, 1, 2, 2],
            "t": [0, 0, 2**53, 2**53, 2**53 + 1, 2**53 + 1],
        }
    )
    with pytest.raises(AnalysisError, match="atomic group"):
        prediction_split(f, roles=["train", "test"] * 3, group="g")
    with pytest.raises(AnalysisError, match="look ahead"):
        prediction_split(f, roles=["test"] * 2 + ["train"] * 2 + ["validation"] * 2, time="t")
    split = prediction_split(f, roles=["train"] * 4 + ["test"] * 2, group="g")
    f["g"] += 9
    with pytest.raises(AnalysisError, match="source role identity"):
        cart(f, outcome="y", features=["x"], split=split, min_leaf=1)


def test_unknown_test_classes_do_not_change_model():
    f = frame("classification")
    s = prediction_split(f, roles=["train"] * 50 + ["validation"] * 10 + ["test"] * 10)
    f.iloc[60:, f.columns.get_loc("y")] = "NEW"
    r = cart(f, outcome="y", features=["a", "b"], split=s, task="classification", min_leaf=3)
    assert "NEW" not in r.attrs["state"]["classes"]
    assert r.attrs["metrics"]["test"]["available"] is False
    with pytest.raises(AnalysisError, match="absent from training"):
        cart_predict(r, f, outcome="y")
    cart_predict(r, f[["a", "b"]])


def test_metrics_zero_probability_and_single_class_auc():
    from openecon.econometrics.supervised.metrics import classification, regression

    y = torch.tensor([0, 1, 1], dtype=torch.int64)
    p = c.tensor([[1, 0], [1, 0], [0.2, 0.8]])
    m = classification(y, p, c.tensor([1, 2, 3]), 2)
    assert m["log_loss"] is None and m["log_loss_unbounded"]
    # Pairwise weighted concordance: positive weight3 beats all negatives;
    # positive weight2 ties with negative weight1.
    assert m["auc"] == pytest.approx((3 + 1) / 5)
    assert (
        classification(torch.zeros(3, dtype=torch.int64), p, c.tensor([1, 2, 3]), 2)["auc"] is None
    )
    assert regression(c.tensor([1, 1]), c.tensor([1, 2]), c.tensor([1, 2]))["r2"] is None


@pytest.mark.parametrize(
    "path,value",
    [
        (("nodes", 0, "risk"), 9.0),
        (("nodes", 0, "n"), True),
        (("derived", "selected_stage"), True),
        (("derived", "prediction", "predictions", 0), 9.0),
        (("derived", "importance", "raw", 0), 1e-320),
        (("transform", "imputation_means", 0), 9.0),
        (("source", "dtypes", "features", 0), "object"),
        (("derived", "path", 0, "cp"), 9.0),
    ],
)
def test_rehashed_cache_forgery_refused(path, value):
    state = copy.deepcopy(fit().attrs["state"])
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    rehash(state)
    with pytest.raises(AnalysisError):
        cart_restore(state)


@pytest.mark.parametrize("bad", [[], [[1.0]], None])
def test_malformed_cache_shape_refuses_before_numeric_replay(bad, monkeypatch):
    state = copy.deepcopy(fit().attrs["state"])
    state["derived"]["prediction"]["predictions"] = bad
    rehash(state)
    monkeypatch.setattr(
        module, "_context", lambda *a: pytest.fail("numeric replay before shape admission")
    )
    with pytest.raises(AnalysisError):
        cart_restore(state)


def test_serializer_semantic_replay_and_structural_zero():
    r = fit()
    r.attrs["state"]["nodes"][0]["prediction"] *= 4
    rehash(r.attrs["state"])
    with pytest.raises(AnalysisError):
        r.to_json()
    state = copy.deepcopy(fit().attrs["state"])
    state["derived"]["importance"]["uncertainty_available"] = 0
    rehash(state)
    with pytest.raises(ValueError):
        CartState(payload=state)


def test_shape_budget_precedes_source_copy_and_scan(monkeypatch):
    large = pd.DataFrame(np.empty((20_001, 2)), columns=["x", "y"])
    monkeypatch.setattr(c, "resident", lambda *a: pytest.fail("copy before resident admission"))
    with pytest.raises(AnalysisError):
        cart(large, outcome="y", features=["x"])
    with pytest.raises(AnalysisError):
        cart(frame(), outcome="y", features=["a"], max_work=1)


def test_json_copy_dump_budget_gates():
    dense = '{"payload": {"nodes": [' + ",".join(["{}"] * 40_000) + "]}}"
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        CartState.model_validate_json(dense)
    forged = CartState.model_construct(payload={"nodes": [{}] * 40_000})
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        forged.model_copy(deep=True)
    valid = CartState(payload=fit().attrs["state"])
    with pytest.raises(AnalysisError):
        valid.model_dump_json(indent=10_000)


@pytest.mark.parametrize("operation", ["metadata", "deep copy", "dump"])
def test_dense_metadata_refuses_without_materializing_children(operation):
    # Caller input already exists; the admission walker must not copy 30k child pointers.
    dense = {"dense": [{} for _ in range(30_000)]}
    typed = CartState.model_construct(payload=dense)
    action = {
        "metadata": lambda: c.metadata(dense),
        "deep copy": lambda: typed.model_copy(deep=True),
        "dump": lambda: typed.model_dump_json(),
    }[operation]
    tracemalloc.start()
    try:
        with use_workspace_budget(1), pytest.raises(AnalysisError):
            action()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 128 * 1024


@pytest.mark.parametrize("indent", [None, 0, 2, 64])
def test_escaped_json_and_indent_bound_covers_native_encoder(indent):
    value = {"astral": ["😀" * 16_384] * 5, "controls": ['\\"\n\t\x00é'], "nested": [[1]]}
    _, encoded, indentation, lines = c._metadata_geometry(value)
    bound = encoded + (0 if indent is None else indent * indentation + lines)
    actual = json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)
    assert len(actual.encode()) <= bound
    assert c.json_output(value, indent=indent) == actual


@pytest.mark.parametrize("case", ["non-BMP", "nested indent"])
def test_json_size_refuses_before_native_serialization(monkeypatch, case):
    if case == "non-BMP":
        # Old size*2 planned ~65.6MB although ensure_ascii produced ~98.3MB.
        value, indent = {"astral": ["😀" * 16_384] * 500}, None
    else:
        value, indent = list(range(30_000)), 64
        for _ in range(39):
            value = [value]
    monkeypatch.setattr(
        c.json, "dumps", lambda *a, **k: pytest.fail("encoder before output admission")
    )
    with pytest.raises(AnalysisError, match="64 MiB"):
        c.json_output(value, indent=indent)


@pytest.mark.parametrize(
    "task,where,field,value",
    [
        ("regression", "root", "probabilities", [0.5]),
        ("regression", "root", "predicted_class", 0),
        ("regression", "root", "prediction", None),
        ("classification", "root", "prediction", 0.5),
        ("classification", "root", "probabilities", [False, 0.5, 0.5]),
        ("classification", "root", "probabilities", [0, 0.5, 0.5]),
        ("classification", "root", "predicted_class", None),
        ("regression", "leaf", "threshold", 0.0),
        ("regression", "leaf", "left", 2),
        ("classification", "leaf", "right", 3),
        ("classification", "leaf", "stop_reason", None),
        ("regression", "leaf", "stop_reason", True),
        ("regression", "leaf", "split_gain", 1e-320),
        ("classification", "root", "stop_reason", "pure"),
    ],
)
def test_conditional_node_cache_refuses_before_context(monkeypatch, task, where, field, value):
    state = copy.deepcopy(fit(task).attrs["state"])
    node = (
        state["nodes"][0]
        if where == "root"
        else next(node for node in state["nodes"] if node["feature"] is None)
    )
    node[field] = value
    rehash(state)
    monkeypatch.setattr(module, "_context", lambda *a, **k: pytest.fail("unadmitted cache context"))
    with pytest.raises(AnalysisError):
        cart_restore(state)


@pytest.mark.parametrize("task,classes", [("regression", [0, 1]), ("classification", [])])
def test_cached_class_shape_binds_task_before_context(monkeypatch, task, classes):
    state = copy.deepcopy(fit(task).attrs["state"])
    state["classes"] = classes
    rehash(state)
    monkeypatch.setattr(module, "_context", lambda *a, **k: pytest.fail("unadmitted task context"))
    with pytest.raises(AnalysisError):
        cart_restore(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_depth", True),
        ("max_nodes", 2),
        ("min_leaf", False),
        ("cp", True),
        ("min_weight_fraction", float("nan")),
    ],
)
def test_invalid_options(field, value):
    kwargs = {"max_depth": 4, "max_nodes": 31, "min_leaf": 3}
    kwargs[field] = value
    with pytest.raises(AnalysisError):
        cart(frame(), outcome="y", features=["a"], **kwargs)


def test_integer_representation_generator_and_node_limit():
    f = pd.DataFrame({"x": [0, 1, 2**53 + 1], "y": [1, 2, 3]})
    with pytest.raises(AnalysisError, match="represented exactly"):
        cart(f, outcome="y", features=["x"], min_leaf=1)
    with pytest.raises(AnalysisError, match="resident"):
        cart({"x": (i for i in range(9)), "y": list(range(9))}, outcome="y", features=["x"])
    f = frame()
    with pytest.raises(AnalysisError, match="no truncated"):
        cart(
            f,
            outcome="y",
            features=["a"],
            split=prediction_split(f, roles=["train"] * 70),
            min_leaf=1,
            max_nodes=1,
        )


def test_ambient_dtype_device_rng_preserved():
    rng = torch.random.get_rng_state().clone()
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        r = fit()
        cart_restore(r.to_json())
        assert torch.get_default_dtype() == torch.float32
        assert torch.get_default_device().type == "meta"
        assert torch.equal(rng, torch.random.get_rng_state())
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_complete_query_source_roundtrip_and_forgery(task, monkeypatch):
    model = fit(task)
    data = frame(task).iloc[[7, 0, 3, 3]].copy()
    query = cart_predict(model, data, outcome="y", weights="w")
    monkeypatch.setattr(module, "_grow", lambda *a: pytest.fail("query state refit"))
    restored = cart_prediction_restore(query.to_json())
    assert restored.attrs == query.attrs
    assert restored["predictions"].index.equals(data.index)
    typed = CartQueryState(payload=query.attrs)
    assert CartQueryState.model_validate_json(typed.model_dump_json()).payload == typed.payload
    assert copy.deepcopy(typed).payload == typed.payload
    damaged = copy.deepcopy(query.attrs)
    damaged["prediction"]["leaf_ids"][0] = 999
    rehash(damaged)
    with pytest.raises(AnalysisError):
        cart_prediction_restore(damaged)


def test_query_shape_admission_before_model_replay(monkeypatch):
    query = cart_predict(fit(), frame()).attrs
    query["prediction"]["predictions"] = []
    rehash(query)
    monkeypatch.setattr(
        module, "cart_restore", lambda *a: pytest.fail("model replay before query shape admission")
    )
    with pytest.raises(AnalysisError):
        cart_prediction_restore(query)


def test_zero_importance_subnormal_forgery():
    f = pd.DataFrame(
        {"a": list(range(30)), "b": [1.0] * 30, "y": [float(i // 5) for i in range(30)]}
    )
    split = prediction_split(f, roles=["train"] * 30)
    r = cart(f, outcome="y", features=["a", "b"], split=split, min_leaf=2)
    state = copy.deepcopy(r.attrs["state"])
    assert state["derived"]["importance"]["raw"][1] == 0.0
    state["derived"]["importance"]["raw"][1] = 1e-320
    rehash(state)
    with pytest.raises(AnalysisError):
        cart_restore(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("classes", None),
        ("settings", []),
        ("features", None),
        ("source", []),
        ("normalization", []),
        ("transform", None),
        ("nodes", {}),
        ("derived", None),
    ],
)
def test_malformed_state_typed_refusal(field, value):
    state = copy.deepcopy(fit().attrs["state"])
    state[field] = value
    rehash(state)
    with pytest.raises(AnalysisError):
        cart_restore(state)


@pytest.mark.parametrize(
    "path,value",
    [
        (("confusion",), []),
        (("classwise",), []),
        (("calibration",), []),
        (("auc_defined",), 1),
        (("n",), True),
        (("averages",), []),
    ],
)
def test_malformed_metric_cache_before_math(path, value, monkeypatch):
    state = copy.deepcopy(fit("classification").attrs["state"])
    state["derived"]["metrics"]["train"][path[0]] = value
    rehash(state)
    monkeypatch.setattr(
        module, "_context", lambda *a: pytest.fail("numeric replay before metric admission")
    )
    with pytest.raises(AnalysisError):
        cart_restore(state)


def test_result_deepcopy_replay_and_no_shared_state():
    result = fit()
    cloned = copy.deepcopy(result)
    cloned.attrs["state"]["nodes"][0]["risk"] = 999.0
    assert result.attrs["state"]["nodes"][0]["risk"] != 999.0
    with pytest.raises(AnalysisError):
        copy.deepcopy(cloned)


def test_constant_response_and_population_missing_refusal():
    f = pd.DataFrame({"x": list(range(12)), "y": [3.0] * 12, "w": [0.5, 1.1, 0.7] * 4})
    split = prediction_split(f, roles=["train"] * 12)
    result = cart(f, outcome="y", features=["x"], weights="w", split=split, min_leaf=1)
    assert len(result.attrs["state"]["nodes"]) == 1
    assert result.attrs["metrics"]["train"]["rmse"] == 0.0
    assert result.attrs["metrics"]["train"]["r2"] is None
    cart_restore(result.to_json())
    f["x"] = np.nan
    with pytest.raises(AnalysisError, match="observed training"):
        cart(f, outcome="y", features=["x"], weights="w", split=split, impute="mean")


def test_full_zero_class_probability_forgery():
    result = fit("classification")
    state = copy.deepcopy(result.attrs["state"])
    leaf = next(
        node for node in state["nodes"] if node["feature"] is None and 0.0 in node["probabilities"]
    )
    leaf["probabilities"][leaf["probabilities"].index(0.0)] = 1e-320
    rehash(state)
    with pytest.raises(AnalysisError):
        cart_restore(state)


def test_cp_closed_event_and_adjacent_floats():
    f = pd.DataFrame({"x": [float(i) for i in range(30)], "y": [float(i // 5) for i in range(30)]})
    split = prediction_split(f, roles=["train"] * 30)
    options = dict(outcome="y", features=["x"], split=split, min_leaf=2)
    maximal = cart(f, **options)
    path = maximal.attrs["state"]["derived"]["path"]
    event = next(point for point in path if point["cp"] > 0)
    below = cart(f, cp=math.nextafter(event["cp"], -math.inf), **options)
    exact = cart(f, cp=event["cp"], **options)
    above = cart(f, cp=math.nextafter(event["cp"], math.inf), **options)
    assert below.attrs["selected_stage"] == event["stage"] - 1
    assert exact.attrs["selected_stage"] == event["stage"]
    assert above.attrs["selected_stage"] == event["stage"]
    cart_restore(exact.to_json())


def test_query_threshold_adjacent_floats_and_equality_left():
    f = pd.DataFrame({"x": [float(i) for i in range(30)], "y": [float(i // 5) for i in range(30)]})
    result = cart(
        f, outcome="y", features=["x"], split=prediction_split(f, roles=["train"] * 30), min_leaf=2
    )
    root = result.attrs["state"]["nodes"][0]
    threshold = root["threshold"]
    query = cart_predict(
        result,
        pd.DataFrame(
            {
                "x": [
                    math.nextafter(threshold, -math.inf),
                    threshold,
                    math.nextafter(threshold, math.inf),
                ]
            }
        ),
    )
    ids = query.attrs["prediction"]["leaf_ids"]

    def ancestor(leaf):
        while leaf > 3:
            leaf //= 2
        return leaf

    assert [ancestor(v) for v in ids] == [2, 2, 3]


def test_original_synthetic_full_native_reference_fixture():
    base = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "cart_oracle", base / "scripts/verify_supervised_cart_oracles.py"
    )
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    fixture = json.loads(
        (base / "tests/fixtures/supervised-cart-native-reference.json").read_text()
    )
    assert fixture["author_archive_sha256"] == reference_module.ARCHIVE_SHA
    for options, frozen in zip(fixture["metadata"]["synthetic_cases"], fixture["cases"]):
        state = reference_module.fixed_original_source_fixture(options["task"], options["seed"]).attrs["state"]
        source_identity = {key: state[key] for key in fixture["source_identity_fields"]}
        encoded_source = json.dumps(source_identity, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()
        assert hashlib.sha256(encoded_source).hexdigest() == frozen["source_input_sha256"]
        native = frozen["native_reference"]
        nodes = {tuple(node["positions"]): node for node in state["nodes"]}
        assert set(nodes) == {tuple(node["rows"]) for node in native["nodes"]}
        for ref in native["nodes"]:
            node = nodes[tuple(ref["rows"])]
            assert node["n"] == ref["n"] and node["feature"] == ref["feature"]
            assert node["weight_sum"] == pytest.approx(ref["weight"], rel=2e-10, abs=2e-12)
        actual = list(reversed(state["derived"]["path"]))
        cp_table = np.asarray(native["cptable"])
        np.testing.assert_allclose(
            [point["cp"] for point in actual], cp_table[:, 0], rtol=2e-10, atol=2e-12
        )
        assert [point["leaf_count"] - 1 for point in actual] == list(cp_table[:, 1])


def test_complete_metrics_against_original_response_not_reconstructed_units():
    result = fit()
    source = result.attrs["state"]["source"]
    prediction = result.attrs["state"]["derived"]["prediction"]["predictions"]
    for role, rows in result.attrs["state"]["split"]["positions"].items():
        y = np.array(source["y"])[rows]
        p = np.array(prediction)[rows]
        w = np.array(source["weights"])[rows]
        metric = result.attrs["metrics"][role]
        assert metric["rmse"] == pytest.approx(
            np.sqrt(np.average((y - p) ** 2, weights=w)), rel=1e-13
        )
        assert metric["mae"] == pytest.approx(np.average(np.abs(y - p), weights=w), rel=1e-13)
        assert metric["r2"] == pytest.approx(
            1 - np.dot(w, (y - p) ** 2) / np.dot(w, (y - np.average(y, weights=w)) ** 2), rel=1e-13
        )


def test_zero_alpha_classification_prunes_at_cp_zero():
    f = pd.DataFrame(
        {
            "x": [float(i) for i in range(12)],
            "y": ["A", "B", "A", "A", "B", "A", "A", "A", "B", "A", "A", "A"],
        }
    )
    result = cart(
        f,
        outcome="y",
        features=["x"],
        task="classification",
        split=prediction_split(f, roles=["train"] * 12),
        min_leaf=2,
        max_depth=1,
    )
    state = result.attrs["state"]
    assert len(state["nodes"]) == 3
    assert state["derived"]["path"][1]["alpha"] == 0.0
    assert result.attrs["selected_stage"] == 1
    assert result.attrs["leaf_count"] == 1
    cart_restore(result.to_json())


def test_large_case_weight_units_preserve_classification_and_restore():
    f = frame("classification")
    split = prediction_split(f, roles=["train"] * len(f))
    kwargs = dict(
        outcome="y",
        features=["a", "b"],
        task="classification",
        weights="w",
        split=split,
        min_leaf=3,
    )
    original = cart(f, **kwargs)
    f["w"] *= 1e99
    scaled = cart(f, **kwargs)
    assert (
        scaled.attrs["state"]["derived"]["prediction"]
        == original.attrs["state"]["derived"]["prediction"]
    )
    assert scaled.attrs["metrics"]["train"]["weight_sum"] > 1e100
    assert cart_restore(scaled.to_json()).attrs == scaled.attrs


def loader():
    path = Path(__file__).parent / "reference/supervised_cart_fixed_inputs.py"
    spec = importlib.util.spec_from_file_location("cart_original_input_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_packet(module):
    return json.loads((module.FIXTURES / "supervised-cart-native-source-inputs.json").read_bytes())


def resealed_decode(module, packet):
    body = json.dumps(packet, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return module._checked_packet(body, _expected_packet_sha256=hashlib.sha256(body).hexdigest())


def test_all_original_complete_input_fields_and_both_minor_packets_are_present():
    module = loader()
    packet = original_packet(module)
    for case in packet["cases"]:
        identity = module.load_original_case(case["options"]["task"], case["options"]["seed"])
        assert set(identity) == set(module.INPUT_FIELDS)
        assert identity == case["complete_physical_identity_by_original_minor"]["3.11"]
        assert identity == case["complete_physical_identity_by_original_minor"]["3.13"]
        assert hashlib.sha256(module._canonical(identity)).hexdigest() == case["source_input_sha256"]
        assert identity["n"] == 96
        assert [len(identity["split"]["positions"][role]) for role in ("train", "validation", "test")] == [72, 12, 12]


@pytest.mark.parametrize("field", ("cell", "dtype", "index", "role", "setting", "class"))
@pytest.mark.parametrize("reseal_identity", (False, True))
def test_resealed_complete_physical_field_mutations_still_fail_original_native_identity(field, reseal_identity):
    module = loader()
    packet = copy.deepcopy(original_packet(module))
    case = packet["cases"][0]
    for identity in case["complete_physical_identity_by_original_minor"].values():
        if field == "cell":
            identity["source"]["x"][0][0] += .125
        elif field == "dtype":
            identity["source"]["dtypes"]["features"][0] = "float32"
        elif field == "index":
            identity["source"]["index"]["start"] += 1
        elif field == "role":
            identity["split"]["explicit_roles"][0] = "test"
        elif field == "setting":
            identity["settings"]["min_leaf"] += 1
        else:
            identity["classes"] = ["unexpected"]
    # Updating the candidate's own input hash cannot replace original R/native
    # fixture bindings; the whole original reference hash remains literal pinned.
    if reseal_identity:
        case["source_input_sha256"] = hashlib.sha256(module._canonical(case["complete_physical_identity_by_original_minor"]["3.13"])).hexdigest()
    with pytest.raises(ValueError):
        resealed_decode(module, packet)


@pytest.mark.parametrize("change", ("omit_field", "add_field", "omit_minor", "reorder_cases", "wrong_git", "wrong_raw", "wrong_fitted"))
def test_resealed_provenance_structure_and_case_transport_cannot_hide_changes(change):
    module = loader()
    packet = copy.deepcopy(original_packet(module))
    case = packet["cases"][0]
    if change == "omit_field":
        case["complete_physical_identity_by_original_minor"]["3.11"].pop("source")
    elif change == "add_field":
        case["complete_physical_identity_by_original_minor"]["3.11"]["hidden"] = "must never disappear"
    elif change == "omit_minor":
        case["complete_physical_identity_by_original_minor"].pop("3.11")
    elif change == "reorder_cases":
        packet["cases"].reverse()
    elif change == "wrong_git":
        packet["provenance"]["original_git_head"] = "0" * 40
    elif change == "wrong_raw":
        packet["provenance"]["original_full_raw_state_artifact_sha256_by_minor"]["3.11"] = "0" * 64
    else:
        case["original_full_fitted_state_digest_by_minor"]["3.13"] = "0" * 64
    with pytest.raises(ValueError):
        resealed_decode(module, packet)


def test_complete_packet_hash_is_mandatory_in_public_loader(tmp_path):
    module = loader()
    packet = original_packet(module)
    packet["cases"][0]["options"]["seed"] = 1
    changed = tmp_path / "changed-original-inputs.json"
    changed.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="complete CART input packet hash"):
        module.load_original_case("regression", 8813, path=changed)


def test_duplicate_nonfinite_and_byte_admission_guards_remain_active(tmp_path, monkeypatch):
    module = loader()
    for body in (b'{"same": 1, "same": 2}', b'{"value": NaN}', b'{"value": Infinity}'):
        with pytest.raises(ValueError):
            module._json(body)
    oversized = tmp_path / "oversized-original-inputs.json"
    oversized.write_bytes(b" " * (module.MAX_PACKET_BYTES + 1))
    def forbidden(*args, **kwargs):
        raise AssertionError("Byte admission happened after unbounded read or JSON parsing.")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(module, "_json", forbidden)
    with pytest.raises(ValueError, match="fixed byte bound"):
        module._read_bounded(oversized)
    with pytest.raises(ValueError, match="complete CART input packet hash"):
        module._checked_packet(b" " * (module.MAX_PACKET_BYTES + 1))


@pytest.mark.parametrize("task,seed", (("unknown", 8813), ("regression", True), ("regression", 8813.0), ("regression", 0)))
def test_only_exact_four_declared_case_labels_are_admitted(task, seed):
    module = loader()
    with pytest.raises(ValueError):
        module.load_original_case(task, seed)


def test_fixed_reference_preparation_does_not_generate_new_inputs_or_use_cached_fitted_tree(monkeypatch):
    module = loader()
    base = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("cart_reference_proposal", base / "scripts/verify_supervised_cart_oracles.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    import numpy as np
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed original CART reference input attempted seed generation/cached restore.")
    monkeypatch.setattr(reference, "source_fixture", forbidden)
    monkeypatch.setattr(reference, "cart_restore", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for name in ("rand", "randn", "randint", "manual_seed"):
        monkeypatch.setattr(torch, name, forbidden)
    called = []
    original_fit = reference.cart
    def counted(data, **kwargs):
        called.append(kwargs)
        return original_fit(data, **kwargs)
    monkeypatch.setattr(reference, "cart", counted)
    for task, seed in (("regression", 8813), ("regression", 19423), ("classification", 8813), ("classification", 19423)):
        original = module.load_original_case(task, seed)
        state = reference.fixed_original_source_fixture(task, seed).attrs["state"]
        assert {key: state[key] for key in module.INPUT_FIELDS} == original
        assert state["source"]["dtypes"] == original["source"]["dtypes"]
        assert state["source"]["index"] == original["source"]["index"]
    assert len(called) == 4
