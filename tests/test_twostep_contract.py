"""Public sample, persistence, resource and descriptive TwoStep contracts."""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.twostep import kernel, public
from openecon.econometrics.twostep.helpers import adjusted_rand
from openecon.resources import use_workspace_budget


def data():
    return pd.DataFrame({"x": [0., .1, .2, 5., 5.1, 5.2], "c": ["a"] * 3 + ["b"] * 3},
                        index=["duplicate"] * 6)


def fit(frame=None, **options):
    return oe.twostep(data=data() if frame is None else frame, continuous=["x"], categorical=["c"],
                      **(dict(n_clusters=2, max_preclusters=8, max_nodes=32) | options))


@pytest.fixture
def fitted():
    return fit()


def saved(result):
    return result.attrs["twostep_state"]


def resign(result, mutate, *, rebuild=False):
    copied = copy.deepcopy(result)
    state = saved(copied)
    mutate(state)
    if rebuild:
        return public._output(state)
    copied.attrs["state_sha256"] = public._seal(state)
    return copied


def test_public_names_are_registered_and_descriptive(fitted):
    for name in ("twostep", "twostep_assign", "twostep_cut", "twostep_profiles", "twostep_quality",
                 "twostep_stability", "twostep_save", "twostep_load"):
        assert callable(getattr(oe, name))
    for result in (fitted, oe.twostep_profiles(fitted), oe.twostep_quality(fitted),
                   oe.twostep_stability([fitted, fitted])):
        assert result.attrs["device"] == "cpu"
        assert result.attrs["dtype"] == "float64"
        assert result.attrs["inference"].startswith("descriptive")
        for frame in result.values():
            assert not set(frame.columns) & {"standard_error", "p_value", "df", "ci_lower", "ci_upper"}


def test_physical_missing_alignment_input_unchanged():
    frame = data()
    frame.loc[frame.index, "unused"] = np.nan
    frame.iloc[1, frame.columns.get_loc("x")] = np.nan
    frame.iloc[4, frame.columns.get_loc("c")] = None
    before = frame.copy(deep=True)
    with pytest.raises(AnalysisError, match="Missing"):
        fit(frame)
    result = fit(frame, missing="drop")
    assert saved(result)["positions"] == [0, 2, 3, 5]
    assert saved(result)["missing_positions"] == [1, 4]
    assert result["assignments"]["position"].tolist() == list(range(6))
    assert result["assignments"]["status"].tolist() == ["cluster", "missing", "cluster", "cluster", "missing", "cluster"]
    assert oe.twostep_load(oe.twostep_save(result))["assignments"].equals(result["assignments"])
    pd.testing.assert_frame_equal(frame, before)


def test_column_mapping_series_are_physical_sequences_without_implicit_index_join():
    mapping = {"x": pd.Series([0., .1, 5., 5.1], index=[7, 8, 9, 10]),
               "c": pd.Series([1, 1., 1, 1.], dtype=object, index=[11, 12, 13, 14])}
    result = fit(mapping)
    assert result.attrs["n_complete"] == result.attrs["n_input"] == 4
    assert saved(result)["positions"] == [0, 1, 2, 3]
    assert saved(result)["categorical"][0]["levels"] == [["int", 1], ["float", 1.]]
    restored = oe.twostep_load(oe.twostep_save(result))
    query = oe.twostep_assign(restored, data=mapping)
    assert query.attrs["n_complete"] == 4


def test_complex_continuous_training_and_query_refuse_without_discarding_imaginary_part(fitted):
    frame = data()
    frame["x"] = frame["x"].astype(complex)+100j
    with pytest.raises(AnalysisError, match="real numeric dtype"):
        fit(frame)
    with pytest.raises(AnalysisError, match="real numeric dtype"):
        oe.twostep_assign(fitted, data=frame)


def test_large_integer_category_identity_refused_without_lossy_float_fallback():
    with pytest.raises(AnalysisError, match="Categorical integers"):
        oe.twostep(data={"c": [2**53+1, 2**53+2]}, categorical=["c"], max_preclusters=8)
    result = oe.twostep(data={"c": [float(2**53), float(2**53+2)]}, categorical=["c"], max_preclusters=8)
    with pytest.raises(AnalysisError, match="Categorical integers"):
        oe.twostep_assign(result, data={"c": [2**53+1]})


def test_query_alignment_training_maps_scaling_frozen(fitted):
    before = copy.deepcopy(saved(fitted))
    query = pd.DataFrame({"x": [10., np.nan, .05], "c": ["b", "a", "a"]}, index=[7, 7, 7])
    out = oe.twostep_assign(fitted, data=query, missing="drop")
    assert out["assignments"]["position"].tolist() == [0, 1, 2]
    assert out["assignments"]["status"].tolist() == ["cluster", "missing", "cluster"]
    assert out.attrs["n_missing"] == 1
    assert saved(fitted) == before
    with pytest.raises(AnalysisError, match="Unknown category"):
        oe.twostep_assign(fitted, data=pd.DataFrame({"x": [0., 1.], "c": ["a", "unseen"]}))


def test_typed_categories_distinguish_bool_integer_float_and_string():
    frame = pd.DataFrame({"x": np.arange(8, dtype=float),
                          "c": pd.Series([True, 1, 1., "1"] * 2, dtype=object)})
    out = fit(frame, n_clusters=4)
    assert saved(out)["categorical"][0]["levels"] == [["bool", True], ["int", 1], ["float", 1.], ["str", "1"]]
    profiles = oe.twostep_profiles(out)["categorical_profiles"]
    assert set(profiles["level_type"]) == {"bool", "int", "float", "str"}
    restored = oe.twostep_load(oe.twostep_save(out))
    assert saved(restored)["categorical"] == saved(out)["categorical"]
    assert oe.twostep_assign(restored, data=frame)["assignments"].equals(oe.twostep_assign(out, data=frame)["assignments"])


def test_mapping_integer_float_categories_and_unseen_typed_query():
    result = fit({"x": [0., 1., 5., 6.], "c": [1, 1, 1., 1.]})
    assert saved(result)["categorical"][0]["levels"] == [["int", 1], ["float", 1.]]
    integers = fit({"x": [0., .1, 5., 5.1], "c": [1, 1, 2, 2]})
    with pytest.raises(AnalysisError, match="Unknown category"):
        oe.twostep_assign(integers, data={"x": [0.], "c": [1.]})


def test_stable_centering_large_offset_original_unit_profiles():
    shifts = np.array([0., 2., 4., 10., 12., 14.])
    result = fit(pd.DataFrame({"x": 1e16 + shifts, "c": ["a"] * 3 + ["b"] * 3}))
    scale = saved(result)["scaling"][0]
    expected = (shifts - shifts.mean()) / shifts.std(ddof=0)
    np.testing.assert_allclose(np.array(saved(result)["training_numeric"])[:, 0], expected, atol=1e-15)
    assert scale["origin"] == 1e16
    assert scale["center"] == 7.
    assert oe.twostep_profiles(result)["continuous_profiles"]["mean"].tolist() == [1e16 + 2., 1e16 + 12.]
    np.testing.assert_allclose(oe.twostep_profiles(result)["continuous_profiles"]["population_sd"],
                               [math.sqrt(8 / 3)] * 2, rtol=1e-15)


def test_profiles_population_moments_original_units(fitted):
    profiles = oe.twostep_profiles(fitted)
    selected = saved(fitted)["cuts"]["2"]
    for label, cf in enumerate(selected, 1):
        values = data().iloc[cf["rows"]]["x"].to_numpy()
        row = profiles["continuous_profiles"].query("cluster == @label").iloc[0]
        assert row["mean"] == pytest.approx(float(values.mean()), abs=1e-13)
        assert row["population_sd"] == pytest.approx(float(values.std(ddof=0)), abs=1e-13)
        cats = profiles["categorical_profiles"].query("cluster == @label")
        assert cats["count"].sum() == len(values)
        assert cats["proportion"].sum() == pytest.approx(1.)
    assert profiles["sizes"]["proportion"].sum() == pytest.approx(1.)
    assert profiles.attrs["n_clustered"] == 6


def test_exact_silhouette_independent_singleton_loss(fitted):
    state = saved(fitted)
    numeric = np.asarray(state["training_numeric"])
    codes = np.asarray(state["training_codes"])
    variance = np.asarray(state["global_variance"])
    # Singleton xi difference: log(1 + squared separation/(4 globalvar)),
    # plus 2 log(2) for each differing categorical coordinate.
    distance = np.zeros((len(numeric), len(numeric)))
    for i in range(len(numeric)):
        for j in range(i):
            delta = numeric[i] - numeric[j]
            loss = float(np.log1p(delta * delta / (4 * variance)).sum())
            loss += 2 * math.log(2) * int((codes[i] != codes[j]).sum())
            distance[i, j] = distance[j, i] = loss
    labels = {row: label for label, cf in enumerate(state["cuts"]["2"], 1) for row in cf["rows"]}
    expected = []
    for i in range(len(numeric)):
        own = [j for j in labels if labels[j] == labels[i] and i != j]
        other = [j for j in labels if labels[j] != labels[i]]
        within, between = float(distance[i, own].mean()), float(distance[i, other].mean())
        expected.append((between - within) / max(within, between))
    quality = oe.twostep_quality(fitted)
    np.testing.assert_allclose(quality["silhouettes"]["silhouette"], expected, rtol=1e-12, atol=1e-13)
    assert quality["quality"].iloc[0]["mean_silhouette"] == pytest.approx(np.mean(expected), abs=1e-13)


def test_quality_singletons_one_cluster_and_zero_distances(fitted):
    singletons = oe.twostep_cut(fitted, 6)
    assert oe.twostep_quality(singletons)["silhouettes"]["silhouette"].tolist() == [0.] * 6
    with pytest.raises(AnalysisError, match="one-cluster"):
        oe.twostep_quality(oe.twostep_cut(fitted, 1))
    assert adjusted_rand([1, 1], [8, 8])[0] == 1.
    assert adjusted_rand([1, 2], [8, 9])[0] == 1.


def noisy():
    return pd.DataFrame({"x": [0.] * 3 + [5.] * 3 + [20., np.nan], "c": ["a"] * 3 + ["b"] * 3 + ["z", "a"]})


def test_noise_excluded_quality_profiles_stability():
    frame = noisy()
    a = fit(frame, missing="drop", min_precluster_size=2)
    b = fit(frame, missing="drop", min_precluster_size=1)
    assert saved(a)["noise_positions"] == [6]
    assert a["assignments"]["status"].tolist()[-2:] == ["noise", "missing"]
    assert oe.twostep_profiles(a).attrs["n_clustered"] == 6
    assert oe.twostep_quality(a)["silhouettes"]["position"].tolist() == list(range(6))
    row = oe.twostep_stability([a, b])["pairs"].iloc[0]
    assert row["n_common"] == 6
    assert row["left_noise"] == 1
    assert row["right_noise"] == 0
    assert row["excluded_union"] == 1
    assert row["adjusted_rand"] == 1.


@pytest.mark.parametrize("left,right,expected", [
    ([1, 1, 2, 2], [7, 7, 9, 9], 1.),
    ([1, 1, 2, 2], [7, 9, 7, 9], -.5),
    ([1, 1, 2, 2, 2, 2], [7, 7, 9, 9, 9, 10], 64 / 109),
    ([1, 1, 1, 1], [7, 9, 7, 9], 0.),
])
def test_ari_exact_known_partitions_label_invariant(left, right, expected):
    score, cells = adjusted_rand(left, right)
    assert score == pytest.approx(expected, abs=1e-15)
    assert sum(cells.values()) == len(left)
    assert adjusted_rand(right, left)[0] == score
    assert adjusted_rand([x * 43 + 9 for x in left], [x * 13 - 5 for x in right])[0] == score


def test_stability_all_pairs_and_corpus_binding(fitted):
    cuts = [fitted, oe.twostep_cut(fitted, 3), oe.twostep_cut(fitted, 1)]
    output = oe.twostep_stability(cuts)
    assert output["pairs"][["left", "right"]].to_numpy().tolist() == [[1., 2.], [1., 3.], [2., 3.]]
    other = data()
    other.iloc[0, 0] = .3
    with pytest.raises(AnalysisError, match="same typed raw input corpus"):
        oe.twostep_stability([fitted, fit(other)])


def test_local_rng_and_user_default_device_restored():
    torch.manual_seed(674)
    np.random.seed(342)
    torch_before, numpy_before = torch.random.get_rng_state().clone(), np.random.get_state()
    previous_device = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        first = fit(order="random", seed=894)
        second = fit(order="random", seed=894)
        assert torch.get_default_device().type == "meta"
        assert saved(first)["order"] == saved(second)["order"]
        assert oe.twostep_stability([first, second])["pairs"].iloc[0]["adjusted_rand"] == 1.
        assert torch.equal(torch_before, torch.random.get_rng_state())
        after = np.random.get_state()
        assert after[0] == numpy_before[0]
        np.testing.assert_array_equal(after[1], numpy_before[1])
        assert after[2:] == numpy_before[2:]
    finally:
        torch.set_default_device(previous_device)


@pytest.mark.parametrize("options", [{"device": "cuda"}, {"device": "mps"}, {"weights": "w"},
                                      {"max_work": True}, {"max_work": 1.5}, {"max_work": 0},
                                      {"max_bytes": True}, {"max_bytes": 1.5}, {"max_bytes": 0}])
@pytest.mark.parametrize("operation", ["fit", "cut", "assign", "profiles", "quality", "stability"])
def test_unsupported_device_weight_and_invalid_budgets(fitted, options, operation):
    call = {"fit": lambda: fit(**options), "cut": lambda: oe.twostep_cut(fitted, 2, **options),
            "assign": lambda: oe.twostep_assign(fitted, data=data(), **options),
            "profiles": lambda: oe.twostep_profiles(fitted, **options),
            "quality": lambda: oe.twostep_quality(fitted, **options),
            "stability": lambda: oe.twostep_stability([fitted, fitted], **options)}[operation]
    with pytest.raises(AnalysisError):
        call()


@pytest.mark.parametrize("options", [{"n_clusters": True}, {"n_clusters": 0}, {"max_clusters": 0},
                                      {"threshold": True}, {"threshold": float("inf")}, {"threshold": -1},
                                      {"branch_factor": 1}, {"max_preclusters": 129}, {"max_nodes": 513},
                                      {"min_precluster_size": 0}, {"order": "random"}, {"order": "reverse"},
                                      {"seed": 5}, {"missing": "omit"}, {"criterion": "jump"}])
def test_fit_controls_rejected(options):
    with pytest.raises(AnalysisError):
        fit(**options)


def test_dataset_never_collected(monkeypatch):
    called = []
    monkeypatch.setattr(Dataset, "collect", lambda *a, **kw: called.append("collect"), raising=False)
    with pytest.raises(AnalysisError, match="Dataset cannot be collected"):
        fit(object.__new__(Dataset))
    assert called == []


@pytest.mark.parametrize("options", [{"max_work": 1}, {"max_bytes": 1}])
def test_fit_refuses_before_input_collection(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("Input conversion happened before admission")
    monkeypatch.setattr(public, "_coerce_frame", forbidden)
    with pytest.raises(AnalysisError):
        fit({"x": [0., 1.], "c": ["a", "b"]}, **options)


def test_row_limit_refuses_before_input_collection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Input conversion happened before row cap")
    monkeypatch.setattr(public, "_coerce_frame", forbidden)
    with pytest.raises(AnalysisError, match="physical input rows"):
        fit({"x": [0.] * 2001, "c": ["a"] * 2001})


@pytest.mark.parametrize("options", [{"max_work": 1}, {"max_bytes": 1}])
def test_quality_refuses_before_pair_allocations(fitted, monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("Singleton quality allocation happened before admission")
    monkeypatch.setattr(kernel, "singleton", forbidden)
    with pytest.raises(AnalysisError):
        oe.twostep_quality(fitted, **options)


@pytest.mark.parametrize("helper", ["profiles", "quality", "stability"])
def test_combined_validation_and_operation_budget_precedes_full_validation(fitted, monkeypatch, helper):
    validation_work = public._validation_cost(fitted)[0]
    def forbidden(*args, **kwargs):
        raise AssertionError("Full state validation happened before total-work admission")
    monkeypatch.setattr(public, "_state", forbidden)
    with pytest.raises(AnalysisError):
        if helper == "stability":
            oe.twostep_stability([fitted, fitted], max_work=2 * validation_work)
        else:
            getattr(oe, "twostep_" + helper)(fitted, max_work=validation_work)


def test_quality_row_cap_even_when_budget_large(monkeypatch):
    frame = pd.DataFrame({"x": [0.] * 250 + [5.] * 251, "c": ["a"] * 250 + ["b"] * 251})
    result = fit(frame, max_preclusters=4)
    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized quality singleton allocation")
    monkeypatch.setattr(kernel, "singleton", forbidden)
    with pytest.raises(AnalysisError, match="at most 500"):
        oe.twostep_quality(result, max_work=2**50, max_bytes=2**50)


def test_task_workspace_override_applies(fitted):
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError, match="workspace bytes"):
            fit()
        assert oe.twostep_profiles(fitted).attrs["resources"]["workspace"]["budget_bytes"] == 1024**2


@pytest.mark.parametrize("results", [None, [], [None], tuple([None, None]), [None] * 11])
def test_stability_list_bounds(results):
    with pytest.raises(AnalysisError):
        oe.twostep_stability(results)


MUTATIONS = [
    ("selected bool", lambda s: s.__setitem__("selected_k", True)),
    ("selected out of range", lambda s: s.__setitem__("selected_k", 100)),
    ("physical bool", lambda s: s["positions"].__setitem__(0, False)),
    ("duplicate physical", lambda s: s["positions"].__setitem__(1, 0)),
    ("missing alignment", lambda s: s["missing_positions"].append(0)),
    ("noise alignment", lambda s: s["noise_positions"].append(0)),
    ("order duplicate", lambda s: s["order"].__setitem__(1, s["order"][0])),
    ("scaling zero", lambda s: s["scaling"][0].__setitem__("sd", 0)),
    ("scaling bool", lambda s: s["scaling"][0].__setitem__("sd", True)),
    ("scaling changed", lambda s: s["scaling"][0].__setitem__("mean", 100)),
    ("global variance", lambda s: s["global_variance"].__setitem__(0, 2.)),
    ("category duplicate", lambda s: s["categorical"][0]["levels"].__setitem__(1, ["str", "a"])),
    ("category malformed type", lambda s: s["categorical"][0]["levels"].__setitem__(0, ["int", "a"])),
    ("category total bool", lambda s: s["categorical"][0]["counts"].__setitem__(0, True)),
    ("training numeric", lambda s: s["training_numeric"][0].__setitem__(0, 3.)),
    ("training codes", lambda s: s["training_codes"][0].__setitem__(0, 1)),
    ("raw sample changed", lambda s: s["raw_sample"][0][0].__setitem__(1, 12.)),
    ("raw typed bool", lambda s: s["raw_sample"][0].__setitem__(0, ["bool", True])),
    ("sample digest forged", lambda s: s.__setitem__("sample_sha256", "f" * 64)),
    ("CF mean changed", lambda s: s["cuts"]["2"][0]["mean"].__setitem__(0, 3.)),
    ("CF M2 changed", lambda s: s["cuts"]["2"][0]["m2"].__setitem__(0, 3.)),
    ("CF M2 negative", lambda s: s["cuts"]["2"][0]["m2"].__setitem__(0, -1.)),
    ("CF count bool", lambda s: s["cuts"]["2"][0].__setitem__("count", True)),
    ("CF rows overlap", lambda s: s["cuts"]["2"][1]["rows"].__setitem__(0, 0)),
    ("CF cat histogram", lambda s: s["cuts"]["2"][0]["categorical_counts"].__setitem__(0, [2, 1])),
    ("precluster moments", lambda s: s["preclusters"][0]["mean"].__setitem__(0, 3.)),
    ("criteria score", lambda s: s["criteria"][0].__setitem__(1, 100.)),
    ("criteria parameter count", lambda s: s["criteria"][0].__setitem__(2, 100)),
    ("criterion option", lambda s: s["controls"].__setitem__("criterion", "jump")),
    ("noise policy bool", lambda s: s["controls"].__setitem__("min_precluster_size", True)),
    ("fixed selected mismatch", lambda s: s["controls"].__setitem__("n_clusters", 1)),
]


@pytest.mark.parametrize("name,mutation", MUTATIONS, ids=[name for name, _ in MUTATIONS])
def test_resigned_semantic_corruption_rejected(fitted, name, mutation):
    corrupted = resign(fitted, mutation)
    with pytest.raises(AnalysisError):
        oe.twostep_assign(corrupted, data=data())


def test_resigned_fake_tree_moments_rejected(fitted):
    corrupted = resign(fitted, lambda s: s["tree"]["entries"][0]["cf"]["mean"].__setitem__(0, 3.), rebuild=True)
    with pytest.raises(AnalysisError):
        oe.twostep_load(oe.summary_state(corrupted))


def test_resigned_merge_wrong_distance_rejected_even_when_display_rebuilt(fitted):
    corrupted = resign(fitted, lambda s: s["merges"][0].__setitem__("distance", 3.), rebuild=True)
    with pytest.raises(AnalysisError):
        oe.twostep_load(oe.summary_state(corrupted))


@pytest.mark.parametrize("name", ["assignments", "criteria", "merges", "sizes", "continuous_profiles", "categorical_profiles"])
def test_displayed_tables_bound_to_state(fitted, name):
    corrupted = copy.deepcopy(fitted)
    corrupted[name].iloc[0, 0] = 99
    with pytest.raises(AnalysisError):
        oe.twostep_save(corrupted)


def test_checksum_and_full_json_restart(fitted, tmp_path):
    encoded = oe.twostep_save(fitted)
    restored = oe.twostep_load(encoded)
    assert restored.attrs == fitted.attrs
    assert list(restored) == list(fitted)
    for name in fitted:
        pd.testing.assert_frame_equal(restored[name], fitted[name])
    assert saved(restored)["training_numeric"] == saved(fitted)["training_numeric"]
    payload = json.loads(encoded)
    payload["attrs"]["twostep_state"]["scaling"][0]["sd"] = 9.
    with pytest.raises(AnalysisError):
        oe.twostep_load(json.dumps(payload))
    path = tmp_path / "twostep.json"
    path.write_text(encoded)
    script = """import json,sys,openecon as oe
r=oe.twostep_load(open(sys.argv[1]).read())
p=oe.twostep_profiles(r)
q=oe.twostep_quality(r)
a=oe.twostep_assign(r,data={'x':[0.05,5.05],'c':['a','b']})
print(json.dumps({'state':r.attrs['state_sha256'],'tables':list(r),'mean':q['quality'].iloc[0]['mean_silhouette'],'labels':a['assignments']['cluster'].tolist(),'count':p.attrs['n_clustered']}))
"""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=f"{root / 'src'}:{root / 'packages/openecon-charts/src'}")
    process = subprocess.run([sys.executable, "-c", script, str(path)], cwd=root, env=environment,
                             check=True, capture_output=True, text=True)
    receipt = json.loads(process.stdout)
    assert receipt["state"] == fitted.attrs["state_sha256"]
    assert receipt["tables"] == list(fitted)
    assert receipt["count"] == 6
    assert receipt["labels"] == [1, 2]
    assert receipt["mean"] == pytest.approx(oe.twostep_quality(fitted)["quality"].iloc[0]["mean_silhouette"])


def test_settings_preview_is_not_complete_state(fitted):
    setting = fitted["settings"].set_index("setting").loc["twostep_state", "json"]
    assert "full_state" in json.loads(setting)
    assert len(oe.twostep_save(fitted)) > len(setting)
    assert len(json.loads(oe.twostep_save(fitted))["attrs"]["twostep_state"]["cuts"]) == 6
