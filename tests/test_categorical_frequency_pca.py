"""Independent replication/correlation-PCA and retained membership contracts."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.frequency_pca import (
    catpca_category_centroids, catpca_fweight, catpca_fweight_predict,
)
from openecon.econometrics.categorical.frequency import _seal
from openecon.econometrics.categorical.optimal import catpca
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


VARIABLES = ["a", "b", "z", "q"]
OPTIONS = dict(scales={"a": "nominal", "b": "ordinal", "z": "numeric", "q": "numeric"},
               orders={"b": ["low", "mid", "high"]}, maxiter=300, n_starts=2, tol=1e-11)


def frequency_pca_sample():
    """Deterministic 48-row example; ordinal low/mid pool but retain membership."""
    rng = np.random.default_rng(413)
    n = 48
    x = rng.normal(size=n)
    return pd.DataFrame(dict(a=rng.choice(["red", "blue", "green"], n),
                             b=rng.choice(["low", "mid", "high"], n), z=x,
                             q=.7*x+rng.normal(size=n), w=rng.integers(1, 5, n)))


def mixed_fit(frame=None):
    frame = frequency_pca_sample() if frame is None else frame
    return catpca_fweight(frame, VARIABLES, frequency="w", **OPTIONS)


def matrix(result, name):
    return result[name].to_numpy(dtype=float)


def test_numeric_fit_matches_independent_numpy_weighted_correlation_pca():
    rng = np.random.default_rng(97)
    raw = rng.normal(size=(37, 4))
    raw[:, 2] += 1.3*raw[:, 0]
    weight = rng.integers(1, 7, len(raw))
    frame = pd.DataFrame(raw, columns=list("abcd")).assign(w=weight)
    fit = catpca_fweight(frame, list("abcd"), frequency="w", scales=dict.fromkeys("abcd", "numeric"), components=2)
    mean = np.average(raw, axis=0, weights=weight)
    variance = np.average((raw-mean)**2, axis=0, weights=weight)
    standardized = (raw-mean)/np.sqrt(variance)
    covariance = standardized.T@(weight[:, None]*standardized)/weight.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues, axes = eigenvalues[::-1], eigenvectors[:, ::-1][:, :2]
    for column in range(2):
        if axes[np.argmax(np.abs(axes[:, column])), column] < 0:
            axes[:, column] *= -1
    scores = standardized@axes/np.sqrt(eigenvalues[:2])
    np.testing.assert_allclose(matrix(fit, "transformed")[:, 1:], standardized, atol=2e-14)
    np.testing.assert_allclose(fit["eigenvalues"].eigenvalue, eigenvalues, atol=2e-14)
    np.testing.assert_allclose(matrix(fit, "scores")[:, 1:], scores, atol=3e-14)
    np.testing.assert_allclose(matrix(fit, "loadings"), axes*np.sqrt(eigenvalues[:2]), atol=3e-14)
    assert fit["fit"].reconstruction_loss.iloc[0] == pytest.approx(eigenvalues[2:].sum(), abs=2e-14)
    assert fit.attrs["weight_type"] == "frequency"
    assert "descriptive" in fit.attrs["inference"]


def test_mixed_fit_matches_literal_replication_and_monotone_objective():
    frame = frequency_pca_sample()
    fit = mixed_fit(frame)
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    oracle = catpca(expanded, VARIABLES, **OPTIONS)
    np.testing.assert_allclose(matrix(fit, "loadings"), matrix(oracle, "loadings"), atol=3e-12)
    np.testing.assert_allclose(np.repeat(matrix(fit, "scores")[:, 1:], frame.w, axis=0), matrix(oracle, "scores")[:, 1:], atol=3e-12)
    np.testing.assert_allclose(fit["quantifications"].quantification.astype(float), oracle["quantifications"].quantification.astype(float), atol=3e-12)
    scores, transformed = matrix(fit, "scores")[:, 1:], matrix(fit, "transformed")[:, 1:]
    weight = frame.w.to_numpy()
    np.testing.assert_allclose(scores.T@(weight[:, None]*scores)/weight.sum(), np.eye(2), atol=3e-14)
    np.testing.assert_allclose(np.average(transformed, axis=0, weights=weight), 0, atol=1e-14)
    np.testing.assert_allclose(np.average(transformed**2, axis=0, weights=weight), 1, atol=1e-14)
    for _, iterations in fit["iterations"].groupby("start"):
        assert np.all(np.diff(iterations.objective.to_numpy()) <= 1e-9)
    assert fit.attrs["frequency_total"] == int(weight.sum())
    assert len(fit["scores"]) == len(frame)


def test_weight_scale_invariance_large_counts_never_expand_rows(monkeypatch):
    frame = frequency_pca_sample()
    expected = mixed_fit(frame)
    # Expanding the valid frequency total would require almost a billion rows.
    # The numerical work and table domain must remain the original 48 rows.
    frame.w *= 7_000_000
    def refuse_expansion(*args, **kwargs):
        raise AssertionError("runtime row expansion is forbidden")
    monkeypatch.setattr(torch, "repeat_interleave", refuse_expansion)
    result = mixed_fit(frame)
    assert result.attrs["frequency_total"] == 840_000_000
    assert result.attrs["declared_work"] == expected.attrs["declared_work"]
    assert len(result["scores"]) == 48
    np.testing.assert_allclose(matrix(result, "scores"), matrix(expected, "scores"), atol=5e-12)


def test_original_category_centroids_keep_pooled_ordinal_categories_distinct():
    frame = frequency_pca_sample()
    fit = mixed_fit(frame)
    centers = catpca_category_centroids(fit)
    mapping = fit["quantifications"].set_index(["variable", "category"])
    assert mapping.loc[("b", "low"), "quantification"] == pytest.approx(mapping.loc[("b", "mid"), "quantification"], abs=1e-13)
    scores = matrix(fit, "scores")[:, 1:]
    for _, row in centers["centroids"].iterrows():
        membership = frame[row.variable] == row.category
        expected = np.average(scores[membership], axis=0, weights=frame.w[membership])
        np.testing.assert_allclose(row[["component_1", "component_2"]].to_numpy(dtype=float), expected, atol=2e-14)
        assert row.frequency_count == int(frame.w[membership].sum())
    b = centers["centroids"].query("variable == 'b'").set_index("category")
    assert np.linalg.norm(b.loc["low", ["component_1", "component_2"]].to_numpy(dtype=float)-b.loc["mid", ["component_1", "component_2"]].to_numpy(dtype=float)) > .1
    assert fit.attrs["frequency_state"]["membership"]["b"] == [OPTIONS["orders"]["b"].index(x) for x in frame.b]


@pytest.mark.parametrize("basis", [[[.8, -.6], [.6, .8]], [[1.2, .35], [.1, .8]]])
def test_centroid_declared_basis_transport_preserves_reconstruction_and_metric(basis):
    fit = mixed_fit()
    base = catpca_category_centroids(fit)
    result = catpca_category_centroids(fit, transform=basis)
    inverse = np.linalg.inv(basis)
    np.testing.assert_allclose(result["centroids"].iloc[:, 4:].to_numpy(dtype=float), base["centroids"].iloc[:, 4:].to_numpy(dtype=float)@inverse.T, atol=2e-14)
    np.testing.assert_allclose(matrix(result, "scores")[:, 1:], matrix(fit, "scores")[:, 1:]@inverse.T, atol=2e-14)
    np.testing.assert_allclose(matrix(result, "loadings"), matrix(fit, "loadings")@basis, atol=2e-14)
    np.testing.assert_allclose(matrix(result, "metric"), inverse@inverse.T, atol=2e-14)
    np.testing.assert_allclose(matrix(result, "scores")[:, 1:]@matrix(result, "loadings").T, matrix(fit, "scores")[:, 1:]@matrix(fit, "loadings").T, atol=3e-14)
    weight = np.asarray(fit.attrs["frequency_state"]["frequencies"])
    new_scores = matrix(result, "scores")[:, 1:]
    np.testing.assert_allclose(new_scores.T@(weight[:, None]*new_scores)/weight.sum(), matrix(result, "metric"), atol=3e-14)


def test_complete_json_latex_projection_and_centroid_roundtrip():
    frame = frequency_pca_sample()
    fit = mixed_fit(frame)
    encoded = summary_state(fit)
    restored = restore_summary(encoded)
    assert summary_state(restored) == encoded
    assert restored.to_latex() == fit.to_latex()
    projection = catpca_fweight_predict(restored, frame.drop(columns="w"))
    np.testing.assert_allclose(matrix(projection, "scores"), matrix(fit, "scores"), atol=2e-14)
    for output in (projection, catpca_category_centroids(restored, transform=[[1.2, .3], [.2, .9]])):
        json_value = summary_state(output)
        portable = restore_summary(json_value)
        assert json_value == summary_state(portable)
        assert output.to_latex() == portable.to_latex()
    assert "membership" in json.loads(encoded)["attrs"]["frequency_state"]


def test_count_missing_zero_and_duplicate_index_preserve_physical_positions():
    frame = frequency_pca_sample()
    frame.index = [3]*len(frame)
    frame.iloc[1, frame.columns.get_loc("w")] = 0
    frame.iloc[1, frame.columns.get_loc("a")] = None
    frame.iloc[3, frame.columns.get_loc("w")] = np.nan
    frame.iloc[5, frame.columns.get_loc("z")] = np.nan
    fit = mixed_fit(frame)
    assert fit.attrs["zero_positions"] == [1]
    assert fit.attrs["missing_positions"] == [3, 5]
    assert fit.attrs["sample_positions"] == [x for x in range(48) if x not in (1, 3, 5)]
    query = frame.iloc[[2, 5, 8]].drop(columns="w")
    projection = catpca_fweight_predict(fit, query, missing="drop")
    assert projection.attrs["sample_positions"] == [0, 2]
    with pytest.raises(AnalysisError, match="missing"):
        catpca_fweight_predict(fit, query)
    zero_only = frequency_pca_sample()
    zero_only.loc[1, ["w", "a"]] = [0, None]
    accepted = catpca_fweight(zero_only, VARIABLES, frequency="w", missing="raise", **OPTIONS)
    assert accepted.attrs["zero_positions"] == [1]


def test_unknown_categories_and_typed_numeric_category_identity():
    frame = frequency_pca_sample()
    frame["a"] = pd.Series([True, 1, 1.0, "1"]*12, dtype=object)
    result = mixed_fit(frame)
    types = result["quantifications"].query("variable == 'a'").category_type.tolist()
    assert types == ["bool", "int", "float", "str"]
    centers = catpca_category_centroids(result)
    assert centers["centroids"].query("variable == 'a'").category_type.tolist() == types
    query = frame.iloc[:2].copy()
    query.loc[0, "a"] = "unseen"
    with pytest.raises(AnalysisError, match="Unknown typed category"):
        catpca_fweight_predict(result, query)


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("inf"), 1_000_000_001, "2"])
def test_invalid_frequencies_refused_even_when_feature_missing(bad):
    frame = frequency_pca_sample()
    frame["w"] = frame.w.astype(object)
    frame.loc[0, "w"] = bad
    frame.loc[0, "z"] = np.nan
    with pytest.raises(AnalysisError):
        mixed_fit(frame)


@pytest.mark.parametrize("basis", [[[1, 0], [0, 0]], [[1, 0], [0, 1e-7]], [[1, float("nan")], [0, 1]], [[True, 0], [0, 1]], [[1, 0, 0], [0, 1, 0]], [[1e7, 0], [0, 1e7]]])
def test_invalid_basis_is_refused(basis):
    with pytest.raises(AnalysisError):
        catpca_category_centroids(mixed_fit(), transform=basis)


@pytest.mark.parametrize("table,column", [("scores", "component_1"), ("transformed", "a"), ("sample", "frequency"), ("fit", "reconstruction_loss"), ("eigenvalues", "eigenvalue"), ("quantifications", "quantification")])
def test_specialized_reuse_rejects_tampered_tables(table, column):
    restored = restore_summary(summary_state(mixed_fit()))
    restored[table][column] = restored[table][column].astype(float)
    restored[table].loc[restored[table].index[0], column] += .3
    with pytest.raises(AnalysisError) as error:
        catpca_category_centroids(restored)
    assert error.value.code == "invalid_state"
    with pytest.raises(AnalysisError):
        catpca_fweight_predict(restored, frequency_pca_sample())


def test_membership_tampering_and_legacy_map_only_state_refused():
    result = mixed_fit()
    broken = restore_summary(summary_state(result))
    broken.attrs["frequency_state"]["membership"]["b"][0] = 42
    broken.attrs["state_sha256"] = _seal(broken.attrs["frequency_state"])
    with pytest.raises(AnalysisError):
        catpca_category_centroids(broken)
    missing = restore_summary(summary_state(result))
    del missing.attrs["frequency_state"]["membership"]
    missing.attrs["state_sha256"] = _seal(missing.attrs["frequency_state"])
    with pytest.raises(AnalysisError):
        catpca_category_centroids(missing)
    legacy = catpca(frequency_pca_sample(), VARIABLES, **OPTIONS)
    with pytest.raises(AnalysisError, match="retaining original memberships"):
        catpca_category_centroids(legacy)


def test_budget_failure_precedes_numerical_conversion_or_saved_hash(monkeypatch):
    frame = frequency_pca_sample()
    fit = mixed_fit(frame)
    larger = mixed_fit(pd.concat([frame]*3, ignore_index=True))
    def fail(*args, **kwargs):
        raise AssertionError("numerical conversion happened before admission")
    monkeypatch.setattr(torch, "tensor", fail)
    with pytest.raises(AnalysisError) as error:
        catpca_fweight(frame, VARIABLES, frequency="w", max_bytes=1, **OPTIONS)
    assert error.value.code in ("resource_limit", "workspace_limit")
    with pytest.raises(AnalysisError) as error:
        catpca_category_centroids(fit, max_work=1)
    assert error.value.code in ("resource_limit", "workspace_limit")
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            catpca_category_centroids(larger)
        assert error.value.code in ("resource_limit", "workspace_limit")
    oversized = restore_summary(summary_state(fit))
    oversized.attrs["frequency_state"]["membership"]["b"] = [0]*12013
    with pytest.raises(AnalysisError) as error:
        catpca_category_centroids(oversized)
    assert error.value.code in ("resource_limit", "workspace_limit")


def test_explicit_nonconvergence_is_not_published_as_success():
    with pytest.raises(AnalysisError) as error:
        catpca_fweight(frequency_pca_sample(), VARIABLES, frequency="w", scales=OPTIONS["scales"], orders=OPTIONS["orders"], n_starts=1, maxiter=1, tol=1e-12)
    assert error.value.code == "nonconvergence"


def test_raw_numeric_state_prevents_resealed_standardization_forgery():
    fit = restore_summary(summary_state(mixed_fit()))
    descriptor = next(x for x in fit.attrs["frequency_state"]["descriptors"] if x["name"] == "z")
    descriptor["mean"] += .4
    fit["numeric_scaling"].loc[fit["numeric_scaling"].variable == "z", "mean"] = descriptor["mean"]
    fit.attrs["state_sha256"] = _seal(fit.attrs["frequency_state"])
    with pytest.raises(AnalysisError, match="original numeric scaling"):
        catpca_fweight_predict(fit, frequency_pca_sample())


@pytest.mark.parametrize("kind", ["improvement", "convergence", "chosen", "settings"])
def test_saved_trace_and_settings_must_match_selected_converged_fit(kind):
    fit = restore_summary(summary_state(mixed_fit()))
    if kind == "improvement":
        fit["iterations"].loc[1, "improvement"] += .01
    elif kind == "convergence":
        fit["starts"].loc[0, "converged"] = False
    elif kind == "chosen":
        fit.attrs["chosen_start"] = 42
    else:
        fit["settings"].loc[0, "json"] = "null"
    with pytest.raises(AnalysisError, match="convergence trace"):
        catpca_category_centroids(fit)


def test_saved_objects_and_nonfinite_metadata_refused_before_hash(monkeypatch):
    import openecon.econometrics.categorical.frequency_pca as module
    fit = mixed_fit()
    def fail(*args, **kwargs):
        raise AssertionError("hashing was attempted before bounded primitive checks")
    monkeypatch.setattr(module, "_seal", fail)
    for invalid in (object(), 2**100, float("nan")):
        broken = restore_summary(summary_state(fit))
        broken.attrs["frequency_state"]["numeric_values"]["z"][0] = invalid
        with pytest.raises(AnalysisError) as error:
            catpca_category_centroids(broken)
        assert error.value.code == "invalid_state"


@pytest.mark.parametrize("basis", [np.eye(2, dtype=bool), np.eye(2, dtype=complex), torch.eye(2).to_sparse()])
def test_boolean_complex_and_sparse_array_bases_refused(basis):
    with pytest.raises(AnalysisError) as error:
        catpca_category_centroids(mixed_fit(), transform=basis)
    assert error.value.code == "invalid_transform"


@pytest.mark.parametrize("kind", ["version_bool", "long_variable", "long_frequency"])
def test_saved_state_scalar_and_name_domains_match_fit(kind):
    fit = restore_summary(summary_state(mixed_fit()))
    state = fit.attrs["frequency_state"]
    if kind == "version_bool":
        state["version"] = True
    elif kind == "long_variable":
        state["variables"][0] = "x"*129
    else:
        state["frequency"] = "w"*129
    fit.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError) as error:
        catpca_category_centroids(fit)
    assert error.value.code == "invalid_state"



def test_complex_numeric_query_is_refused_with_analysis_error():
    fit = mixed_fit()
    query = frequency_pca_sample().iloc[:2].copy()
    query["z"] = [1+2j, 3+4j]
    with pytest.raises(AnalysisError):
        catpca_fweight_predict(fit, query)


def test_fit_rejects_unreusable_large_state_before_tensor_allocation(monkeypatch):
    frame = pd.DataFrame(np.random.default_rng(8).normal(size=(3000, 12)), columns=list("abcdefghijkl")).assign(w=1)
    def fail(*args, **kwargs):
        raise AssertionError("fit allocation preceded reusable-state admission")
    monkeypatch.setattr(torch, "tensor", fail)
    with pytest.raises(AnalysisError, match="reusable domain") as error:
        catpca_fweight(frame, list("abcdefghijkl"), frequency="w", scales=dict.fromkeys("abcdefghijkl", "numeric"), components=11, n_starts=1, maxiter=2)
    assert error.value.code == "resource_limit"


def test_centroid_aggregate_work_is_admitted_before_source_tensor_copies(monkeypatch):
    fit = mixed_fit()
    n, p, d = 48, 4, 2
    validation_work = 64*n*p*(p+d)
    def fail(*args, **kwargs):
        raise AssertionError("tensor conversion preceded aggregate centroid admission")
    monkeypatch.setattr(torch, "tensor", fail)
    with pytest.raises(AnalysisError) as error:
        catpca_category_centroids(fit, max_work=validation_work+1)
    assert error.value.code == "resource_limit"



def test_numpy_integral_seed_and_controls_are_normalized_before_torch():
    frame = frequency_pca_sample()
    options = dict(frequency="w", scales=dict.fromkeys(["z", "q"], "numeric"), components=1, n_starts=1, maxiter=2, seed=0)
    expected = catpca_fweight(frame, ["z", "q"], **options)
    options.update(seed=np.int64(0), n_starts=np.int64(1), maxiter=np.int64(2), tol=np.float64(1e-8))
    result = catpca_fweight(frame, ["z", "q"], **options)
    assert summary_state(result) == summary_state(expected)
