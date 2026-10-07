import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.spatial.weights import SpatialWeights
from openecon.resources import use_workspace_budget


def small_weights():
    keys = ["a", "b", "c", "d", "e"]
    edges = [
        ("a", "b", 2),
        ("a", "c", 1),
        ("b", "a", 1),
        ("b", "e", 2),
        ("c", "a", 1),
        ("c", "d", 2),
        ("d", "e", 1),
        ("e", "b", 3),
    ]
    return oe.spatial_weights(keys, edges)


def test_normalization_key_reorder_and_payload_roundtrip():
    w = small_weights()
    matrix = w.dense().numpy()
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    np.testing.assert_array_equal(np.diag(matrix), 0)
    reordered = w.align(list(reversed(w.keys)))
    np.testing.assert_allclose(reordered.dense().numpy(), matrix[::-1, ::-1])
    restored = SpatialWeights.from_payload(json.loads(json.dumps(w.to_payload(), allow_nan=False)))
    assert restored.to_payload() == w.to_payload()
    assert restored.summary()["sha256"] == w.summary()["sha256"]
    assert w.sparse_tensor().layout == torch.sparse_coo
    with pytest.raises(AnalysisError, match="key domains"):
        w.align(["a", "b"])
    sub = w.align(["a", "b", "c"], subset=True)
    assert sub.keys == ("a", "b", "c")
    np.testing.assert_allclose(sub.dense().numpy().sum(axis=1), 1)


@pytest.mark.parametrize(
    "keys,edges,options,code",
    [
        (["a", "a"], [], {}, "invalid_spatial_keys"),
        ([True, "b"], [], {}, "invalid_spatial_keys"),
        ([1.0, "b"], [], {}, "invalid_spatial_keys"),
        (["a", "b"], [("a", "c", 1)], {}, "spatial_key_mismatch"),
        (["a", "b"], [("a", "a", 1)], {}, "spatial_diagonal"),
        (["a", "b"], [("a", "b", -1)], {}, "invalid_spatial_weights"),
        (["a", "b"], [("a", "b", float("nan"))], {}, "invalid_spatial_weights"),
        (["a", "b"], [("a", "b", 1), ("a", "b", 1)], {}, "duplicate_spatial_edges"),
        (["a", "b"], [("a", "b", 1)], {}, "spatial_isolates"),
        (["a", "b"], [("a", "b", 1)], {"normalization": "spectral"}, "invalid_spatial_weights"),
    ],
)
def test_weight_guards(keys, edges, options, code):
    with pytest.raises(AnalysisError) as error:
        oe.spatial_weights(keys, edges, **options)
    assert error.value.code == code


def test_sparse_weight_construction_payload_alignment_and_iterables_preflight():
    class Oversized:
        def __len__(self):
            return 10000

        def __iter__(self):
            raise AssertionError("Oversized input must be rejected before traversal")

    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.spatial_weights(Oversized(), [])
    keys = list(range(64))
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.spatial_weights(keys, Oversized())
    consumed = 0

    def edges():
        nonlocal consumed
        for i in keys:
            for j in keys:
                if i != j:
                    consumed += 1
                    yield i, j, 1.0

    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.spatial_weights(keys, edges())
    assert consumed < 3000
    w = oe.spatial_weights(keys, list(edges())[:3000], isolates="zero")
    payload = w.to_payload()
    for operation in (
        lambda: SpatialWeights.from_payload(payload),
        w.to_payload,
        w.summary,
        lambda: w.align(list(reversed(keys))),
    ):
        with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
            operation()


def test_explicit_diagonal_drop_isolate_zero_and_payload_tampering():
    w = oe.spatial_weights(
        ["a", "b", "c"],
        [("a", "a", 4), ("a", "b", 2), ("b", "a", 1)],
        normalization="none",
        diagonal="drop",
        isolates="zero",
    )
    assert w.summary()["isolate_keys"] == ["c"]
    assert w.values == (2.0, 1.0)
    bad = w.to_payload()
    bad["rows"].append(2)
    bad["cols"].append(2)
    bad["values"].append(2)
    with pytest.raises(AnalysisError, match="zero diagonal"):
        SpatialWeights.from_payload(bad)
    bad = w.to_payload()
    bad["normalization"] = "row"
    with pytest.raises(AnalysisError, match="unit nonzero row sums"):
        SpatialWeights.from_payload(bad)
    bad = w.to_payload()
    bad["cols"][0] = 9
    with pytest.raises(AnalysisError, match="key domain"):
        SpatialWeights.from_payload(bad)


def test_moran_randomization_moments_equal_all_120_assignments():
    w = small_weights()
    matrix = w.dense().numpy()
    values = np.array([1.0, 2.0, -1.0, 4.0, 7.0])
    data = pd.DataFrame({"id": w.keys, "y": values})
    result = oe.moran(data, "y", key="id", spatial_weights=w, null="randomization")

    def statistic(values):
        centered = values - values.mean()
        return len(values) / matrix.sum() * (centered @ matrix @ centered) / (centered @ centered)

    all_null = np.array(
        [statistic(values[np.array(order)]) for order in itertools.permutations(range(5))]
    )
    assert result["statistic"] == pytest.approx(statistic(values), abs=1e-14)
    assert result["expected"] == pytest.approx(all_null.mean(), abs=1e-14)
    assert result["variance"] == pytest.approx(all_null.var(ddof=0), abs=1e-14)
    json.dumps(result, allow_nan=False)
    normal = oe.moran(data, "y", key="id", spatial_weights=w, null="normality")
    s1 = ((matrix + matrix.T) ** 2).sum() / 2
    s2 = ((matrix.sum(axis=0) + matrix.sum(axis=1)) ** 2).sum()
    variance = (25 * s1 - 5 * s2 + 3 * matrix.sum() ** 2) / (24 * matrix.sum() ** 2) - 1 / 16
    assert normal["variance"] == pytest.approx(variance, abs=1e-14)


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_seeded_permutations_match_independent_numpy_statistic_and_plusone(alternative):
    w = small_weights()
    matrix = w.dense().numpy()
    values = np.array([1.0, 2.0, -1.0, 4.0, 7.0])
    data = pd.DataFrame({"id": w.keys, "y": values})
    result = oe.moran(
        data, "y", key="id", spatial_weights=w, permutations=149, seed=38, alternative=alternative
    )
    # Reproduce only Torch's index RNG; statistic and tail computations are NumPy.
    generator = torch.Generator().manual_seed(38)
    centered = values - values.mean()
    null = []
    for _ in range(149):
        draw = centered[torch.randperm(5, generator=generator).numpy()]
        null.append(5 / matrix.sum() * (draw @ matrix @ draw) / (draw @ draw))
    null = np.array(null)
    observed = result["statistic"]
    extreme = (
        np.abs(null + 1 / 4) >= abs(observed + 1 / 4) - 1e-12
        if alternative == "two-sided"
        else null >= observed - 1e-12
        if alternative == "greater"
        else null <= observed + 1e-12
    )
    assert result["permutation"]["extreme_count"] == int(extreme.sum())
    assert result["permutation"]["p_value"] == (int(extreme.sum()) + 1) / 150
    assert result["permutation"]["variance_population"] == pytest.approx(null.var(), abs=1e-14)
    state = torch.random.get_rng_state().clone()
    again = oe.moran(
        data, "y", key="id", spatial_weights=w, permutations=149, seed=38, alternative=alternative
    )
    assert again == result and torch.equal(state, torch.random.get_rng_state())


def test_isolate_null_retains_full_n_and_missing_induced_restandardization():
    keys = ["a", "b", "c", "d", "e"]
    w = oe.spatial_weights(keys, [("a", "b", 1), ("b", "c", 1), ("c", "a", 1)], isolates="zero")
    y = np.array([1.0, 2.0, -1.0, 4.0, 7.0])
    data = pd.DataFrame({"id": keys, "y": y})
    result = oe.moran(data, "y", key="id", spatial_weights=w)
    matrix = w.dense().numpy()
    centered = y - y.mean()
    null = np.array(
        [
            5 / matrix.sum() * (z @ matrix @ z) / (z @ z)
            for order in itertools.permutations(range(5))
            for z in [centered[list(order)]]
        ]
    )
    assert result["nobs"] == 5 and result["expected"] == pytest.approx(-0.25)
    assert result["variance"] == pytest.approx(null.var(), abs=1e-14)
    data.loc[4, "y"] = np.nan
    dropped = oe.moran(data, "y", key="id", spatial_weights=w, missing="drop")
    assert dropped["sample_positions"] == [0, 1, 2, 3] and dropped["nobs"] == 4
    assert dropped["spatial_summary"]["isolate_keys"] == ["d"]


def test_moran_work_missing_sample_constant_and_budget_guards():
    w = small_weights()
    data = pd.DataFrame({"id": w.keys, "y": [1, 2, -1, 4, 7]})
    with pytest.raises(AnalysisError, match="max_permutation_work"):
        oe.moran(data, "y", key="id", spatial_weights=w, permutations=99, max_permutation_work=1)
    data.loc[0, "y"] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        oe.moran(data, "y", key="id", spatial_weights=w)
    data["y"] = 1
    with pytest.raises(AnalysisError, match="constant"):
        oe.moran(data, "y", key="id", spatial_weights=w)
    n = 8000
    big = oe.spatial_weights(range(n), [(i, (i + 1) % n, 1) for i in range(n)])
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.moran(pd.DataFrame({"id": range(n), "y": range(n)}), "y", key="id", spatial_weights=big)
