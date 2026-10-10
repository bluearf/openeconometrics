"""Independent frequency geometry, replication and saved-state certification."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.torch_test_state import preserve_torch_default_device

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical import frequency_scaling as fs
from openecon.econometrics.categorical.frequency import _seal
from openecon.econometrics.categorical.scaling import mca, overals
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def categorical_sample():
    rng = np.random.default_rng(211)
    n = 48
    return pd.DataFrame(dict(a=rng.choice(["red", "blue", "green"], n),
                             b=rng.choice(["low", "mid", "high"], n),
                             c=rng.choice([False, True], n),
                             z=rng.normal(size=n), q=rng.normal(size=n),
                             w=rng.integers(1, 5, n)))


def matrix(result, name):
    return result[name].to_numpy(dtype=float)


def mca_fit(frame=None):
    return fs.mca_fweight(categorical_sample() if frame is None else frame,
                          ["a", "b", "c"], frequency="w", n_components=2)


def numeric_sample():
    rng = np.random.default_rng(97)
    z = rng.normal(size=37)
    q = -.63*z+rng.normal(size=37)*.51
    return pd.DataFrame(dict(z=z, q=q, w=rng.integers(1, 7, len(z))))


def numeric_fit(frame=None, **kwargs):
    return fs.overals_fweight(numeric_sample() if frame is None else frame,
                             [["z"], ["q"]], n_components=1, frequency="w",
                             scales={"z": "numeric", "q": "numeric"},
                             n_starts=1, max_iter=300, tol=1e-11, **kwargs)


def mixed_fit(frame=None):
    return fs.overals_fweight(categorical_sample() if frame is None else frame,
                             [["a", "z"], ["b", "q"]], frequency="w", n_components=2,
                             scales={"a": "nominal", "z": "numeric", "b": "ordinal", "q": "numeric"},
                             orders={"b": ["low", "mid", "high"]}, n_starts=2,
                             max_iter=300, tol=1e-9)


def test_mca_independent_numpy_disjunctive_svd_and_contributions():
    frame = categorical_sample()
    fit = mca_fit(frame)
    # Construct indicator columns independently in first physical appearance order.
    levels = [list(dict.fromkeys(frame[name])) for name in ["a", "b", "c"]]
    z = np.column_stack([frame[name].to_numpy() == category
                         for name, cats in zip(["a", "b", "c"], levels) for category in cats]).astype(float)
    weight = frame.w.to_numpy()
    mass = weight@z/(weight.sum()*3)
    s = np.sqrt(weight/weight.sum())[:, None]*(z/3-mass)/np.sqrt(mass)
    _, singular, vt = np.linalg.svd(s, full_matrices=False)
    standard = vt[:2].T/np.sqrt(mass)[:, None]
    rows = (z/3-mass)@standard
    for j in range(2):
        if rows[np.argmax(np.abs(rows[:, j])), j] < 0:
            rows[:, j] *= -1
            standard[:, j] *= -1
    eigenvalues = singular[singular > singular[0]*1e-12]**2
    np.testing.assert_allclose(fit["inertia"].raw_inertia, eigenvalues, atol=3e-15)
    np.testing.assert_allclose(matrix(fit, "row_coordinates"), rows, atol=2e-14)
    np.testing.assert_allclose(matrix(fit, "category_standard"), standard, atol=2e-14)
    np.testing.assert_allclose(fit["category_coordinates"].mass, mass, atol=2e-16)
    np.testing.assert_allclose(matrix(fit, "row_contributions"), rows**2*weight[:, None]/weight.sum()/eigenvalues[:2], atol=3e-15)
    np.testing.assert_allclose(matrix(fit, "category_contributions").sum(0), 1, atol=2e-15)
    np.testing.assert_allclose(matrix(fit, "row_contributions").sum(0), 1, atol=2e-15)
    assert eigenvalues.sum() == pytest.approx((z.shape[1]-3)/3, abs=2e-15)
    fs._checked(fit, "mca_fweight")


def test_mca_literal_replication_and_weighted_barycentres():
    frame = categorical_sample()
    fitted = mca_fit(frame)
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    oracle = mca(expanded, ["a", "b", "c"], n_components=2)
    np.testing.assert_allclose(np.repeat(matrix(fitted, "row_coordinates"), frame.w, axis=0), matrix(oracle, "row_coordinates"), atol=3e-14)
    np.testing.assert_allclose(matrix(fitted, "category_standard"), matrix(oracle, "category_standard"), atol=3e-14)
    weight, scores = frame.w.to_numpy(), matrix(fitted, "row_coordinates")
    np.testing.assert_allclose(np.average(scores, axis=0, weights=weight), 0, atol=1e-15)
    np.testing.assert_allclose(scores.T@(weight[:, None]*scores)/weight.sum(),
                               np.diag(fitted["inertia"].raw_inertia.iloc[:2]), atol=3e-15)


def test_two_numeric_overals_matches_independent_weighted_projection_eigenproblem():
    frame = numeric_sample()
    fit = numeric_fit(frame)
    weight = frame.w.to_numpy()
    raw = frame[["z", "q"]].to_numpy()
    z = (raw-np.average(raw, axis=0, weights=weight))/np.sqrt(np.average((raw-np.average(raw, axis=0, weights=weight))**2, axis=0, weights=weight))
    weighted = np.sqrt(weight/weight.sum())[:, None]*z
    operator = (weighted[:, :1]@weighted[:, :1].T+weighted[:, 1:]@weighted[:, 1:].T)/2
    eigen, vectors = np.linalg.eigh(operator)
    x = vectors[:, -1]/np.sqrt(weight/weight.sum())
    if x[np.argmax(np.abs(x))] < 0:
        x *= -1
    rho = np.sum(weight*z[:, 0]*z[:, 1])/weight.sum()
    assert fit.attrs["objective"] == pytest.approx((1-abs(rho))/2, abs=3e-11)
    assert fit.attrs["objective"] == pytest.approx(1-eigen[-1], abs=3e-11)
    np.testing.assert_allclose(matrix(fit, "object_scores")[:, 0], x, atol=4e-6)
    scores = matrix(fit, "object_scores")
    np.testing.assert_allclose(scores.T@(weight[:, None]*scores)/weight.sum(), [[1]], atol=3e-15)
    assert fit["set_fit"].loss_per_person.mean() == pytest.approx(fit.attrs["objective"], abs=1e-15)
    fs._checked(fit, "overals_fweight")


def test_mixed_overals_weighted_geometry_monotone_maps_and_complete_verification():
    frame = categorical_sample()
    fit = mixed_fit(frame)
    scores, weight = matrix(fit, "object_scores"), frame.w.to_numpy()
    np.testing.assert_allclose(np.average(scores, axis=0, weights=weight), 0, atol=3e-15)
    np.testing.assert_allclose(scores.T@(weight[:, None]*scores)/weight.sum(), np.eye(2), atol=3e-15)
    quant = fit["category_quantifications"].query("variable == 'b'").quantification.astype(float).to_numpy()
    assert np.all(np.diff(quant) >= -1e-13)
    for _, trace in fit["iterations"].groupby("start"):
        assert np.all(np.diff(trace.loss.astype(float)) <= 1e-9)
    assert fit.attrs["frequency_total"] == int(weight.sum())
    assert not fit.attrs["inference"] and not fit.attrs["global_optimum"]
    fs._checked(fit, "overals_fweight")


def test_multiple_nominal_two_sets_has_independent_weighted_projection_loss():
    frame = categorical_sample()
    fit = fs.overals_fweight(frame, [["a"], ["b"]], n_components=1, frequency="w",
                             scales={"a": "multiple_nominal", "b": "multiple_nominal"}, n_starts=1,
                             max_iter=300, tol=1e-11)
    w = frame.w.to_numpy()
    v = np.sqrt(w/w.sum())
    projectors = []
    for name in ("a", "b"):
        z = np.column_stack([frame[name].to_numpy() == value for value in dict.fromkeys(frame[name])]).astype(float)
        u = v[:, None]*z
        projectors.append(u@np.linalg.inv(u.T@u)@u.T-np.outer(v, v))
    leading = np.linalg.eigvalsh((projectors[0]+projectors[1])/2)[-1]
    assert fit.attrs["objective"] == pytest.approx(1-leading, abs=3e-10)
    fs._checked(fit, "overals_fweight")


@pytest.mark.parametrize("helper", [mca_fit, numeric_fit])
def test_weight_scale_invariance_large_counts_never_expand_rows(helper, monkeypatch):
    frame = categorical_sample() if helper is mca_fit else numeric_sample()
    expected = helper(frame)
    frame.w *= 1_000_000
    def forbidden(*args, **kwargs):
        raise AssertionError("Replicated numerical rows are forbidden")
    monkeypatch.setattr(torch, "repeat_interleave", forbidden)
    actual = helper(frame)
    name = "row_coordinates" if helper is mca_fit else "object_scores"
    np.testing.assert_allclose(matrix(actual, name), matrix(expected, name), atol=2e-12)
    assert actual.attrs["declared_work"] == expected.attrs["declared_work"]
    assert len(actual[name]) == len(frame)
    assert actual.attrs["frequency_total"] == int(frame.w.sum())


def test_numeric_overals_literal_replication_and_allones_existing_behavior():
    frame = numeric_sample()
    actual = numeric_fit(frame)
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    oracle = overals(expanded, [["z"], ["q"]], n_components=1,
                     scales={"z": "numeric", "q": "numeric"}, n_starts=1, max_iter=300, tol=1e-11)
    assert actual.attrs["objective"] == pytest.approx(oracle.attrs["objective"], abs=3e-11)
    assert np.max(np.abs(np.repeat(matrix(actual, "object_scores"), frame.w, axis=0)-matrix(oracle, "object_scores"))) < 1e-5
    ones = categorical_sample().assign(w=1)
    old = mca(ones, ["a", "b", "c"], n_components=2)
    np.testing.assert_allclose(matrix(mca_fit(ones), "row_coordinates"), matrix(old, "row_coordinates"), atol=2e-14)


@pytest.mark.parametrize("helper", [mca_fit, numeric_fit, mixed_fit])
def test_complete_generic_json_latex_tables_and_saved_model_verification(helper):
    fit = helper()
    encoded = summary_state(fit)
    restored = restore_summary(encoded)
    assert summary_state(restored) == encoded
    assert restored.to_latex() == fit.to_latex()
    for name in fit:
        assert fit[name].equals(restored[name]), name
    assert "frequency_state" in json.loads(encoded)["attrs"]
    fs._checked(restored, fit.attrs["method"])


def test_projection_training_reproduction_subset_missing_unknown_typed_categories():
    frame = categorical_sample()
    fit = mca_fit(frame)
    projected = fs.mca_fweight_project(restore_summary(summary_state(fit)), frame.drop(columns="w"))
    np.testing.assert_allclose(matrix(projected, "row_coordinates"), matrix(fit, "row_coordinates"), atol=1e-15)
    query = frame.iloc[[7, 2, 7]].drop(columns="w").copy()
    query.index = [3, 3, 2]
    query.iloc[1, query.columns.get_loc("a")] = None
    subset = fs.mca_fweight_project(fit, query, missing="drop")
    assert subset.attrs["sample_positions"] == [0, 2]
    assert list(subset["row_coordinates"].index) == [3, 2]
    np.testing.assert_allclose(matrix(subset, "row_coordinates"), matrix(fit, "row_coordinates")[[7, 7]], atol=1e-15)
    with pytest.raises(AnalysisError, match="missing"):
        fs.mca_fweight_project(fit, query)
    query.iloc[0, query.columns.get_loc("a")] = "unseen"
    with pytest.raises(AnalysisError) as caught:
        fs.mca_fweight_project(fit, query, missing="drop")
    assert caught.value.code == "unknown_category"
    typed = pd.DataFrame({"a": pd.Series([True, 1, "1", False, True, 1, "1", False], dtype=object),
                          "b": list("aabbabab"), "w": [1]*8})
    saved = fs.mca_fweight(typed, ["a", "b"], frequency="w", n_components=1)
    assert len(saved.attrs["frequency_state"]["levels"][0]) == 4
    fs._checked(restore_summary(summary_state(saved)), "mca_fweight")


@pytest.mark.parametrize("helper", [mca_fit, numeric_fit])
def test_zero_missing_counts_and_duplicate_indices_preserve_physical_positions(helper):
    frame = categorical_sample() if helper is mca_fit else numeric_sample()
    frame.index = [8]*len(frame)
    frame.w = frame.w.astype(float)
    frame.iloc[1, frame.columns.get_loc("w")] = 0
    name = "a" if helper is mca_fit else "z"
    frame.iloc[1, frame.columns.get_loc(name)] = None
    frame.iloc[3, frame.columns.get_loc("w")] = np.nan
    frame.iloc[5, frame.columns.get_loc(name)] = None
    fit = helper(frame)
    assert fit.attrs["zero_positions"] == [1]
    assert fit.attrs["missing_positions"] == [3, 5]
    assert fit.attrs["sample_positions"] == [i for i in range(len(frame)) if i not in (1, 3, 5)]
    assert list(fit["row_coordinates" if helper is mca_fit else "object_scores"].index) == [8]*(len(frame)-3)
    fs._checked(fit, fit.attrs["method"])


@pytest.mark.parametrize("mutation", ["counts", "mass", "scores", "standard", "table", "settings", "partition", "inertia"])
def test_resealed_mca_semantic_corruption_is_rejected(mutation):
    fit = deepcopy(mca_fit())
    state = fit.attrs["frequency_state"]
    if mutation == "counts":
        state["frequencies"][0] += 1
        state["frequency_total"] += 1
        fit.attrs["frequency_total"] += 1
    elif mutation == "mass":
        state["category_mass"][0] += .001
    elif mutation == "scores":
        state["scores"][0][0] += .1
    elif mutation == "standard":
        state["category_standard"][0][0] += .1
    elif mutation == "table":
        fit["row_coordinates"].iat[0, 0] += .1
    elif mutation == "settings":
        fit["settings"].iat[0, 1] = "wrong"
    elif mutation == "partition":
        state["zero_positions"] = [0]
        fit.attrs["zero_positions"] = [0]
    else:
        state["eigenvalues"][0] *= 1.1
    fit.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError) as caught:
        fs.mca_fweight_project(fit, categorical_sample())
    assert caught.value.code == "invalid_state"


@pytest.mark.parametrize("mutation", ["scores", "loading", "set_scores", "norm", "numeric", "trace", "choice", "table"])
def test_resealed_overals_geometry_and_trace_corruption_is_rejected(mutation):
    fit = deepcopy(numeric_fit())
    state = fit.attrs["frequency_state"]
    if mutation == "scores":
        state["scores"][0][0] += .1
    elif mutation == "loading":
        state["loadings"][0][0][0] += .1
    elif mutation == "set_scores":
        state["set_scores"][0][0][0] += .1
    elif mutation == "norm":
        state["quantifications"][0][0][0] += .1
    elif mutation == "numeric":
        state["levels"][0][0] += 1
    elif mutation == "trace":
        state["trace"][-1][3] += .1
    elif mutation == "choice":
        state["selected_start"] = 1
        fit.attrs["selected_start"] = 1
    else:
        fit["set_fit"].iat[0, 1] += .1
    fit.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError) as caught:
        fs._checked(fit, "overals_fweight")
    assert caught.value.code == "invalid_state"


@pytest.mark.parametrize("helper", [mca_fit, numeric_fit])
@pytest.mark.parametrize("bad", [-1, .5, float("inf"), True, 10**1000])
def test_invalid_counts_rejected_even_on_feature_missing_rows(helper, bad):
    frame = categorical_sample() if helper is mca_fit else numeric_sample()
    frame.w = frame.w.astype(object)
    frame.iloc[0, frame.columns.get_loc("w")] = bad
    frame.iloc[0, frame.columns.get_loc("a" if helper is mca_fit else "z")] = None
    with pytest.raises(AnalysisError):
        helper(frame)


@pytest.mark.parametrize("kind", ["mca_fweight", "overals_fweight"])
def test_saved_nested_tensor_refusal_before_hash_or_tensor_expansion(kind, monkeypatch):
    fit = deepcopy(mca_fit() if kind == "mca_fweight" else numeric_fit())
    key = "row_coordinates" if kind == "mca_fweight" else "object_scores"
    fit[key] = fit[key].astype(object)
    fit[key].iat[0, 0] = torch.arange(100)
    def forbidden(*args, **kwargs):
        raise AssertionError("Hashing/converting malformed state is forbidden")
    monkeypatch.setattr(fs.f, "_seal", forbidden)
    with pytest.raises(AnalysisError) as caught:
        fs._checked(fit, kind)
    assert caught.value.code == "invalid_state"


def test_combined_projection_admission_before_saved_numerical_reconstruction(monkeypatch):
    fit = mca_fit()
    state, plan = fs._preflight(fit, "mca_fweight", fs.BYTES, fs.WORK)
    query = pd.concat([categorical_sample()]*50, ignore_index=True)
    budget = plan["estimated_workspace_bytes"]
    # Standalone verification fits this exact limit; the query plus verification does not.
    fs._checked(fit, "mca_fweight", max_bytes=budget)
    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical reconstruction before combined admission")
    monkeypatch.setattr(fs, "_indicator", forbidden)
    with pytest.raises(AnalysisError) as caught:
        fs.mca_fweight_project(fit, query, max_bytes=budget)
    assert caught.value.code == "workspace_limit"


@pytest.mark.parametrize("method", [fs.mca_fweight, fs.overals_fweight])
def test_invalid_device_and_tiny_work_budget_before_tensors(method, monkeypatch):
    frame = categorical_sample()
    args, extra = (["a", "b"], {}) if method is fs.mca_fweight else ([["a"], ["b"]], {"scales": {"a": "nominal", "b": "nominal"}, "n_components": 1})
    with pytest.raises(AnalysisError) as caught:
        method(frame, args, frequency="w", device="cuda", **extra)
    assert caught.value.code == "unsupported_option"
    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor allocation before work admission")
    monkeypatch.setattr(torch, "tensor", forbidden)
    with pytest.raises(AnalysisError) as caught:
        method(frame, args, frequency="w", max_work=1, **extra)
    assert caught.value.code == "resource_limit"


def test_global_workspace_refusal_and_default_device_pinning():
    large = pd.concat([categorical_sample()]*20, ignore_index=True)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        mca_fit(large)
    assert caught.value.code == "workspace_limit"
    with preserve_torch_default_device():
        torch.set_default_device("meta")
        fit = mca_fit()
        query = fs.mca_fweight_project(fit, categorical_sample())
        assert fit.attrs["device"] == query.attrs["device"] == "cpu"
        assert torch.get_default_device().type == "meta"


def test_private_overals_generator_preserves_global_random_state():
    before = torch.random.get_rng_state().clone()
    mixed_fit()
    assert torch.equal(before, torch.random.get_rng_state())


def test_nonconvergence_rank_and_structural_controls_refuse():
    with pytest.raises(AnalysisError):
        fs.overals_fweight(numeric_sample(), [["z"], ["q"]], n_components=2,
                           frequency="w", scales={"z": "numeric", "q": "numeric"})
    with pytest.raises(AnalysisError) as caught:
        fs.overals_fweight(numeric_sample(), [["z"], ["q"]], n_components=1,
                           frequency="w", scales={"z": "numeric", "q": "numeric"}, n_starts=1, max_iter=1, tol=1e-12)
    assert caught.value.code == "nonconvergence"
    for extra in ({"frequency": "a"}, {"n_components": True}, {"max_bytes": True}):
        kwargs = dict(frequency="w", **({} if "frequency" in extra else extra))
        if "frequency" in extra:
            kwargs["frequency"] = extra["frequency"]
        with pytest.raises(AnalysisError):
            fs.mca_fweight(categorical_sample(), ["a", "b"], **kwargs)


def test_multiple_nominal_dimensions_can_exceed_number_of_variables():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(dict(a=rng.integers(0, 5, 50), b=rng.integers(0, 5, 50), w=np.ones(50, dtype=int)))
    fit = fs.overals_fweight(frame, [["a"], ["b"]], n_components=3, frequency="w",
                             scales={"a": "multiple_nominal", "b": "multiple_nominal"},
                             max_iter=200, n_starts=2, tol=1e-9)
    assert fit["object_scores"].shape == (50, 3)
    fs._checked(fit, "overals_fweight")


def test_tiny_numeric_variance_raises_analysis_error_instead_of_type_error():
    frame = numeric_sample()
    frame.z *= 1e-14
    with pytest.raises(AnalysisError) as caught:
        numeric_fit(frame)
    assert caught.value.code == "nonconvergence"


def test_coherently_shrunk_loadings_fail_block_normal_equations():
    fit = deepcopy(numeric_fit())
    state = fit.attrs["frequency_state"]
    for loading in state["loadings"]:
        for row in loading:
            for j in range(len(row)):
                row[j] *= 1e-8
    for values in state["set_scores"]:
        for row in values:
            for j in range(len(row)):
                row[j] *= 1e-8
    w, x = np.asarray(state["frequencies"]), np.asarray(state["scores"])
    objective = float(sum(np.sum(w[:, None]*(x-np.asarray(values))**2) for values in state["set_scores"])/w.sum()/len(state["sets"]))
    state["objective"] = fit.attrs["objective"] = objective
    state["trace"] = [[0, 1, objective, 1-objective, True]]
    state["starts"] = [[0, objective, 1, True, "accepted"]]
    state["settings"]["tol"] = 1e-7
    fit.attrs["settings"] = state["settings"]
    fit.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError, match="normal equations"):
        fs._checked(fit, "overals_fweight")


def test_matching_saved_cell_numpy_array_refuses_without_scalar_conversion():
    fit = deepcopy(mca_fit())
    fit["row_coordinates"] = fit["row_coordinates"].astype(object)
    fit["row_coordinates"].iat[0, 0] = np.arange(100)
    with pytest.raises(AnalysisError) as caught:
        fs._checked(fit, "mca_fweight")
    assert caught.value.code == "invalid_state"


def test_saved_integer_index_cannot_be_changed_to_boolean_by_equality_alias():
    frame = categorical_sample()
    frame.index = np.arange(len(frame))
    fit = deepcopy(mca_fit(frame))
    fit["row_coordinates"].index = [False, *range(1, len(frame))]
    with pytest.raises(AnalysisError) as caught:
        fs._checked(fit, "mca_fweight")
    assert caught.value.code == "invalid_state"


def test_saved_dimension_boolean_cannot_alias_integer_metadata():
    fit = deepcopy(numeric_fit())
    fit.attrs["n_components"] = True
    with pytest.raises(AnalysisError) as caught:
        fs._checked(fit, "overals_fweight")
    assert caught.value.code == "invalid_state"


def test_compressed_and_expanded_multinomial_overals_same_deterministic_start():
    frame = categorical_sample()
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True).assign(w=1)
    options = dict(frequency="w", n_components=1, scales={"a": "multiple_nominal", "b": "multiple_nominal"},
                   n_starts=1, max_iter=300, tol=1e-11)
    actual = fs.overals_fweight(frame, [["a"], ["b"]], **options)
    oracle = fs.overals_fweight(expanded, [["a"], ["b"]], **options)
    assert actual.attrs["objective"] == pytest.approx(oracle.attrs["objective"], abs=5e-15)
    np.testing.assert_allclose(np.repeat(matrix(actual, "object_scores"), frame.w, axis=0),
                               matrix(oracle, "object_scores"), atol=1e-13)
