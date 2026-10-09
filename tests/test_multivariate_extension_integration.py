"""Public option integration, literal frequency replication and persisted scoring."""
import json

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet


def fixture(seed=321, n=211):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 2))
    loadings = np.array([[.8, .1], [.7, .1], [.75, .2], [.1, .8], [.1, .7], [.2, .75]])
    x = f @ loadings.T + rng.normal(scale=.55, size=(n, 6))
    frame = pd.DataFrame(x * np.arange(1, 7) + np.arange(6), columns=list("abcdef"))
    return frame.assign(w=np.arange(n) % 4)


@pytest.mark.parametrize("method", ["pf", "ipf", "pcf", "ml", "minres"])
@pytest.mark.parametrize("rotation", [None, "varimax", "oblimin"])
def test_frequency_factor_complete_tables_match_literal_expansion(method, rotation):
    data = fixture()
    names = list("abcdef")
    expanded = data.loc[data.index.repeat(data.w), names]
    options = dict(method=method, factors=2, rotate=rotation, max_iterations=2000)
    result = oe.factor(data, names, weights="w", **options)
    oracle = oe.factor(expanded, names, **options)
    assert result.attrs["n"] == int(data.w.sum())
    assert result.attrs["physical_rows"] == int((data.w > 0).sum())
    assert result.attrs["n_zero_weight"] == int((data.w == 0).sum())
    assert result.attrs["covariance_divisor"] == result.attrs["n"] - 1
    assert result.attrs["factors"] == oracle.attrs["factors"] == 2
    assert set(result) == set(oracle)
    for key in result:
        assert list(result[key].index) == list(oracle[key].index)
        assert list(result[key].columns) == list(oracle[key].columns)
        np.testing.assert_allclose(result[key], oracle[key], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(oe.factor_scores(result, data), oe.factor_scores(oracle, data), atol=2e-6, rtol=2e-6)


def test_frequency_factor_replay_missing_zero_and_saved_scores():
    data = fixture()
    data.index = pd.Index([f"subject-{i}" for i in range(len(data))])
    data.iloc[2, 0] = np.nan
    data.iloc[4, -1] = np.nan
    names = list("abcdef")
    result = oe.factor(data, names, weights="w", factors=2, scores="anderson_rubin")
    replay = oe.factor(oe.Dataset.from_frame(data), names, weights="w", factors=2, scores="anderson_rubin")
    for key in result:
        np.testing.assert_allclose(result[key], replay[key], atol=1e-9, rtol=1e-9)
    assert result.attrs["n_missing"] == replay.attrs["n_missing"] == 2
    assert result.attrs["n_zero_weight"] == 52
    assert result.attrs["physical_rows"] == 157
    payload = json.loads(json.dumps({"attrs": result.attrs, "tables": {
        key: value.astype(object).where(value.notna(), None).to_dict(orient="split") for key, value in result.items()}}))
    restored = TableSet({key: pd.DataFrame(**value) for key, value in payload["tables"].items()}, **payload["attrs"])
    original = oe.factor_scores(result, data)
    saved = oe.factor_scores(restored, data)
    assert original.index.equals(data.index)
    assert original.iloc[2].isna().all()
    np.testing.assert_allclose(original, saved, atol=1e-12, rtol=0, equal_nan=True)


@pytest.mark.parametrize("weight", [-1, .5, 2**53 + 1])
def test_frequency_factor_rejects_invalid_weight(weight):
    data = fixture()
    data["w"] = pd.Series([1]*len(data), dtype="uint64" if weight > 2**53 else "float64")
    data.loc[0, "w"] = weight
    with pytest.raises(AnalysisError):
        oe.factor(data, list("abcdef"), factors=2, weights="w")


def test_frequency_factor_weight_roles_and_summary_refusal():
    data = fixture()
    with pytest.raises(AnalysisError, match="distinct"):
        oe.factor(data, list("abcdef"), weights="a", factors=2)
    with pytest.raises(AnalysisError, match="weight_type"):
        oe.factor(data, list("abcdef"), weights="w", weight_type="pweight", factors=2)
    with pytest.raises(AnalysisError, match="Summary matrices"):
        oe.factor_matrix(data[list("abcdef")].corr(), n=len(data), weights="w", factors=2)
    with pytest.raises(AnalysisError, match="at least three"):
        oe.factor(data.assign(w=[1, 1]+[0]*(len(data)-2)), list("abcdef"), weights="w", factors=2)


def test_frequency_factor_large_offset_anchored_moments():
    data = fixture(n=401)
    names = list("abcdef")
    for col in names:
        data[col] += 1e10
    expanded = data.loc[data.index.repeat(data.w), names]
    fit = oe.factor(data, names, weights="w", factors=2)
    oracle = oe.factor(expanded, names, factors=2)
    np.testing.assert_allclose(fit["loadings"], oracle["loadings"], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(fit["descriptives"], oracle["descriptives"], atol=2e-6, rtol=1e-7)


@pytest.mark.parametrize("method", ["alpha", "image_covariance"])
@pytest.mark.parametrize("seed", [17, 412])
def test_new_extraction_raw_summary_replay_and_full_score_state(method, seed):
    data = fixture(seed, n=331).drop(columns="w")
    names = list(data)
    result = oe.factor(data, names, method=method, factors=2)
    summary = oe.factor_matrix(data.corr(), n=len(data), means=data.mean(), sds=data.std(), method=method, factors=2)
    replay = oe.factor(oe.Dataset.from_frame(data), names, method=method, factors=2)
    for key in result:
        np.testing.assert_allclose(result[key], summary[key], atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(result[key], replay[key], atol=2e-7, rtol=2e-7)
    assert list(result["fit"].index) == ["independence"]
    assert result.attrs["discrepancy"] is None
    assert result.attrs["mineigen"] is None
    assert result.attrs["converged"]
    encoded = json.dumps({"attrs": result.attrs, "tables": {
        key: table.astype(object).where(table.notna(), None).to_dict(orient="split")
        for key, table in result.items()}}, allow_nan=False)
    payload = json.loads(encoded)
    restored = TableSet({key: pd.DataFrame(**table) for key, table in payload["tables"].items()}, **payload["attrs"])
    np.testing.assert_allclose(oe.factor_scores(restored, data), oe.factor_scores(result, data), atol=1e-12, rtol=0)
    if method == "image_covariance":
        z = (data.to_numpy() - data.mean().to_numpy()) / data.std().to_numpy()
        predictions = np.column_stack([z[:, np.arange(6) != j] @ np.linalg.lstsq(z[:, np.arange(6) != j], z[:, j], rcond=None)[0] for j in range(6)])
        np.testing.assert_allclose(result["image_covariance"], np.cov(predictions, rowvar=False), atol=1e-12, rtol=1e-12)
        assert any("not exact principal-component scores" in note for note in result.attrs["notes"])


@pytest.mark.parametrize("method", ["alpha", "image_covariance"])
def test_new_extraction_weighted_rotated_replication_and_option_gates(method):
    data = fixture(12, n=211)
    names = list("abcdef")
    options = dict(method=method, factors=2, rotate="varimax")
    actual = oe.factor(data, names, weights="w", **options)
    expected = oe.factor(data.loc[data.index.repeat(data.w), names], names, **options)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-7, rtol=2e-7)
    for kwargs in ({}, {"factors": 2, "mineigen": .1}, {"factors": 6}):
        with pytest.raises(AnalysisError):
            oe.factor(data, names, method=method, **kwargs)


def test_alpha_nonpositive_generalizability_note_is_explicit():
    r = [[1, .24119734366209722, .5570435639339154],
         [.24119734366209722, 1, .4452959397905106],
         [.5570435639339154, .4452959397905106, 1]]
    result = oe.factor_matrix(r, n=400, columns=list("abc"), method="alpha", factors=2)
    assert result.attrs["retained_roots_above_one"] is False
    assert any("generalizability is nonpositive" in note for note in result.attrs["notes"])


def test_image_closed_form_options_and_weight_iteration_budgets():
    data = fixture()
    for kwargs in ({"tolerance": 1e-8}, {"max_iterations": 1000}):
        with pytest.raises(AnalysisError, match="closed form"):
            oe.factor(data, list("abcdef"), method="image_covariance", factors=2, **kwargs)
    # Declared iterative fit work is admitted before reading/converting the columns.
    large = pd.DataFrame({f"v{i}": [0., 1., 2.] for i in range(130)}).assign(w=1)
    with pytest.raises(AnalysisError, match="work budget"):
        oe.factor(large, list(large.columns[:-1]), method="ml", factors=2, weights="w")


@pytest.mark.parametrize("rotation", ["cf", "partial_target"])
@pytest.mark.parametrize("oblique", [False, True])
def test_public_extension_rotation_covariance_and_complete_saved_state(rotation, oblique):
    data = fixture(62, n=331).drop(columns="w")
    names = list(data)
    base = oe.factor(data, names, factors=2)
    loading = base["loadings"].to_numpy()
    opts = {"rotate": rotation, "factors": 2, "rotation_tolerance": 1e-8}
    if rotation == "cf":
        opts.update(cf_kappa=1/len(names), cf_oblique=oblique)
    else:
        transform = np.array([[1., .15], [.1, 1.]]) if oblique else np.array([[np.cos(.12), -np.sin(.12)], [np.sin(.12), np.cos(.12)]])
        if oblique:
            transform /= np.linalg.norm(transform, axis=0)
            transform = np.linalg.inv(transform).T
        target = loading @ transform
        mask = np.ones_like(target, dtype=bool)
        mask[0, 1] = mask[3, 0] = False
        target[~mask] = np.nan
        opts.update(target=pd.DataFrame(target, index=names, columns=["first", "second"]),
                    target_mask=pd.DataFrame(mask, index=names, columns=["first", "second"]), target_oblique=oblique)
    result = oe.factor(data, names, **opts)
    assert result.attrs["rotation_optimum"] == "local stationary point; minimum/global uniqueness not established"
    assert result.attrs["rotation_tolerance"] == 1e-8
    pattern = result["rotated_loadings"].to_numpy()
    matrix = result["rotation_matrix"].to_numpy()
    phi = result["factor_correlations"].to_numpy() if oblique else np.eye(2)
    np.testing.assert_allclose(pattern, loading @ matrix, atol=1e-10)
    np.testing.assert_allclose(pattern @ phi @ pattern.T, loading @ loading.T, atol=2e-10)
    np.testing.assert_allclose(phi, np.linalg.inv(matrix.T @ matrix), atol=2e-10)
    if oblique:
        np.testing.assert_allclose(phi.diagonal(), 1., atol=1e-10)
        np.testing.assert_allclose(result["structure"], pattern @ phi, atol=1e-10)
    if rotation == "partial_target":
        np.testing.assert_allclose(result["rotation_target"], target, equal_nan=True)
        np.testing.assert_array_equal(result["rotation_target_mask"], mask)
        np.testing.assert_allclose(pattern[mask], target[mask], atol=2e-7)
    encoded = json.dumps({"attrs": result.attrs, "tables": {key: value.astype(object).where(value.notna(), None).to_dict(orient="split") for key, value in result.items()}}, allow_nan=False)
    payload = json.loads(encoded)
    restored = TableSet({key: pd.DataFrame(**value) for key, value in payload["tables"].items()}, **payload["attrs"])
    np.testing.assert_allclose(oe.factor_scores(restored, data), oe.factor_scores(result, data), atol=1e-12)
    replay = oe.factor(oe.Dataset.from_frame(data), names, **opts)
    for key in result:
        np.testing.assert_allclose(result[key], replay[key], atol=2e-7, equal_nan=True)
    if oblique:
        with pytest.raises(AnalysisError, match="orthogonal"):
            oe.factor(data, names, scores="anderson_rubin", **opts)


def test_extension_rotation_admission_precedes_input_read_and_mask_labels_match():
    for kwargs in ({"factors": 1}, {"factors": 17}, {"factors": 2, "max_iterations": 10001}):
        with pytest.raises(AnalysisError, match="CF and partial-target"):
            oe.factor(None, list("abcdef"), rotate="cf", **kwargs)
    data = fixture().drop(columns="w")
    target = oe.factor(data, list(data), factors=2)["loadings"]
    mask = pd.DataFrame(True, index=list(data), columns=["wrong", "labels"])
    with pytest.raises(AnalysisError, match="mask labels"):
        oe.factor(data, list(data), factors=2, rotate="partial_target", target=target, target_mask=mask)


def test_public_bootstrap_large_integer_index_and_population_assumptions():
    rng = np.random.default_rng(71)
    latent = rng.normal(size=90)
    data = pd.DataFrame(.7*latent[:, None] + rng.normal(scale=.6, size=(90, 4)), columns=list("abcd"))
    data.index = pd.Index([10**400+i for i in range(90)], dtype=object)
    result = oe.factor_bootstrap(data, list(data), replications=39, seed=5)
    assert result.attrs["original_index"] == list(data.index)
    assert result.attrs["sample_positions"] == list(range(90))
    assert "finite fourth moments" in result.attrs["inferential_assumptions"]
    assert "not empirically verified" in result.attrs["inferential_assumptions"]
    json.dumps(result.attrs, allow_nan=False)
