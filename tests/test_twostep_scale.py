"""Regression for large offsets: stable centered moments and saved replay."""

import math

import pandas as pd
import pytest
import torch

from openecon.econometrics.twostep.helpers import twostep_profiles
from openecon.econometrics.twostep.public import (
    twostep,
    twostep_assign,
    twostep_cut,
    twostep_load,
    twostep_save,
)


def fit(frame, **kwargs):
    options = dict(continuous=["x"], n_clusters=2, threshold=0.0, max_preclusters=16, max_nodes=64)
    options.update(kwargs)
    return twostep(data=frame, **options)


@pytest.mark.parametrize(
    "offsets",
    [
        [18.0, 14.0, 10.0, 12.0],
        [0.0, 0.0, 2.0, 2.0],
        [0.0, 2.0, 4.0, 6.0, 8.0],
        [0.0, 0.0, 0.0, 8.0, 10.0],
        [float(2 * i) for i in range(11)],
    ],
)
def test_large_offset_population_scaling_and_complete_replay_profiles(offsets):
    values = [1e16 + delta for delta in offsets]
    frame = pd.DataFrame({"x": values}, index=[7] * len(values))
    result = fit(frame)
    state = result.attrs["twostep_state"]
    scaling = state["scaling"][0]
    origin = values[0]
    shifted = [value - origin for value in values]
    center = math.fsum(shifted) / len(values)
    expected_sd = math.sqrt(math.fsum((delta - center) ** 2 for delta in shifted) / len(values))
    assert scaling["origin"] == origin
    assert scaling["center"] == pytest.approx(center, abs=1e-13)
    assert scaling["sd"] == pytest.approx(expected_sd, rel=1e-13)
    assert scaling["mean"] == origin + center  # Display-only fused value.
    numeric = torch.tensor(state["training_numeric"], dtype=torch.float64)
    torch.testing.assert_close(
        numeric.mean(0), torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0.0
    )
    torch.testing.assert_close(
        numeric.square().mean(0), torch.ones(1, dtype=torch.float64), atol=1e-14, rtol=0.0
    )
    assert state["global_variance"] == pytest.approx([1.0], rel=1e-13)
    serialized = twostep_save(result)
    loaded = twostep_load(serialized)
    assert twostep_save(loaded) == serialized
    assert loaded.attrs["twostep_state"]["scaling"] == state["scaling"]
    profiles = twostep_profiles(loaded)["continuous_profiles"]
    cut = state["cuts"]["2"]
    for label, cf in enumerate(cut, 1):
        group_offsets = [values[row] - origin for row in cf["rows"]]
        mean_offset = math.fsum(group_offsets) / len(group_offsets)
        sd = math.sqrt(
            math.fsum((delta - mean_offset) ** 2 for delta in group_offsets) / len(group_offsets)
        )
        record = profiles.iloc[label - 1]
        assert int(record["count"]) == cf["count"]
        assert float(record["mean"]) == pytest.approx(origin + mean_offset, rel=0.0, abs=2.0)
        assert float(record["population_sd"]) == pytest.approx(sd, rel=1e-12, abs=1e-12)
    replay = twostep_cut(loaded, 1)
    assert replay.attrs["n_clusters"] == 1
    assert len(replay["assignments"]) == len(values)


@pytest.mark.parametrize("translation", [1e16, -1e16])
def test_representable_translation_preserves_hierarchy_ic_and_query_assignment(translation):
    offsets = [0.0, 2.0, 4.0, 6.0, 8.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    original = pd.DataFrame({"x": offsets, "category": ["a"] * 5 + ["b"] * 5})
    original.index = [8] * len(original)
    translated = original.copy()
    translated["x"] += translation
    first, second = (
        fit(original, categorical=["category"]),
        fit(translated, categorical=["category"]),
    )
    a, b = first.attrs["twostep_state"], second.attrs["twostep_state"]
    assert a["training_numeric"] == b["training_numeric"]
    assert a["training_codes"] == b["training_codes"]
    assert a["cuts"] == b["cuts"]
    assert a["merges"] == b["merges"]
    assert a["criteria"] == b["criteria"]
    assert a["sample_sha256"] != b["sample_sha256"]
    for name in ("assignments", "criteria", "merges", "sizes", "categorical_profiles"):
        assert first[name].equals(second[name])
    query = original.iloc[[7, 0, 4]].copy()
    query_translated = query.copy()
    query_translated["x"] += translation
    left, right = twostep_assign(first, data=query), twostep_assign(second, data=query_translated)
    assert left["assignments"].equals(right["assignments"])
    assert left["assignments"]["position"].tolist() == [0, 1, 2]


def test_two_part_center_keeps_missing_physical_positions_and_pinned_defaults():
    frame = pd.DataFrame(
        {
            "x": [1e16, math.nan, 1e16 + 2.0, 1e16 + 6.0, 1e16 + 8.0, 1e16 + 10.0],
            "category": ["a", "a", "a", None, "b", "b"],
        },
        index=[4, 4, 4, 4, 4, 4],
    )
    fitted = fit(frame, categorical=["category"], missing="drop")
    saved = twostep_load(twostep_save(fitted))
    state = saved.attrs["twostep_state"]
    assert state["positions"] == [0, 2, 4, 5]
    assert state["missing_positions"] == [1, 3]
    assert state["scaling"][0]["origin"] == 1e16
    assert state["scaling"][0]["center"] == 5.0
    assert state["controls"]["order"] == "input" and state["controls"]["seed"] is None
    assert state["controls"]["criterion"] == "bic"
    assert state["controls"]["min_precluster_size"] == 1
    assert saved["assignments"]["status"].tolist() == [
        "cluster",
        "missing",
        "cluster",
        "missing",
        "cluster",
        "cluster",
    ]
    assigned = twostep_assign(saved, data=frame, missing="drop")
    assert assigned["assignments"]["position"].tolist() == list(range(6))
    assert assigned["assignments"]["status"].tolist() == [
        "cluster",
        "missing",
        "cluster",
        "missing",
        "cluster",
        "cluster",
    ]
