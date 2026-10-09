"""Independent weighted rotation geometry and complete reusable state."""
import copy

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.frequency_pca import catpca_fweight, catpca_fweight_predict
from openecon.econometrics.categorical.optimal import catpca
from openecon.econometrics.categorical.rotation import catpca_promax, catpca_varimax
from openecon.econometrics.categorical.frequency_rotation import (
    _checked, _plan, _seal, catpca_promax_fweight, catpca_rotated_predict_fweight,
    catpca_varimax_fweight,
)
from openecon.econometrics.summary_state import restore_summary, summary_state


def rotation_sample():
    rng = np.random.default_rng(719)
    latent = rng.normal(size=(64, 2))
    matrix = np.array([[.9, .1], [.8, .3], [.1, .9], [.2, .8]])
    frame = pd.DataFrame(latent@matrix.T+.25*rng.normal(size=(64, 4)), columns=list("abcd"))
    frame["count"] = rng.integers(1, 8, len(frame))
    return frame


def fit_sample():
    frame = rotation_sample()
    return frame, catpca_fweight(frame, list("abcd"), frequency="count", components=2,
                                scales=dict.fromkeys("abcd", "numeric"), n_starts=1)


def identify(loadings, transform):
    pattern = loadings@transform
    transform = transform[:, sorted(range(2), key=lambda k: -sum(pattern[:, k]**2))]
    pattern = loadings@transform
    return transform*np.sign(pattern[np.argmax(abs(pattern), axis=0), range(2)])[None, :]


def oracle(loadings, normalize):
    matrix = loadings/np.linalg.norm(loadings, axis=1)[:, None] if normalize else loadings
    def rot(angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    def criterion(angle):
        values = matrix@rot(angle)
        return -np.sum(np.sum(values**4, axis=0)-np.sum(values**2, axis=0)**2/len(values))
    grid = np.linspace(-np.pi/4, np.pi/4, 1001)
    angle = grid[np.argmin([criterion(x) for x in grid])]
    optimal = minimize_scalar(criterion, bounds=(angle-.003, angle+.003), method="bounded", options={"xatol": 1e-14})
    return identify(loadings, rot(optimal.x))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("method", [catpca_varimax_fweight, catpca_promax_fweight])
def test_independent_periodic_and_lstsq_oracle_weighted_geometry(method, normalize):
    frame, base = fit_sample()
    result = method(base, normalize=normalize, tol=1e-12)
    loadings = base["loadings"].to_numpy(float)
    expected = oracle(loadings, normalize)
    if method is catpca_promax_fweight:
        rotated = loadings@expected
        normalized = rotated/np.linalg.norm(loadings, axis=1)[:, None] if normalize else rotated
        target = np.sign(normalized)*abs(normalized)**4
        raw = expected@np.linalg.lstsq(rotated, target, rcond=None)[0]
        inverse = np.linalg.inv(raw)
        expected = identify(loadings, raw*np.sqrt(np.diag(inverse@inverse.T))[None, :])
    transform = result["transformation"].to_numpy(float)
    np.testing.assert_allclose(transform, expected, atol=2e-7)
    pattern, structure = result["pattern"].to_numpy(float), result["structure"].to_numpy(float)
    scores = result["scores"].iloc[:, 1:].to_numpy(float)
    phi = result["component_correlations"].to_numpy(float)
    np.testing.assert_allclose(scores.T@(frame["count"].to_numpy()[:, None]*scores)/frame["count"].sum(), phi, atol=1e-12)
    np.testing.assert_allclose(structure, pattern@phi, atol=1e-12)
    np.testing.assert_allclose(pattern@phi@pattern.T, loadings@loadings.T, atol=1e-12)
    np.testing.assert_allclose(scores@pattern.T, base["scores"].iloc[:, 1:].to_numpy(float)@loadings.T, atol=1e-12)


@pytest.mark.parametrize("method", [catpca_varimax_fweight, catpca_promax_fweight])
def test_complete_roundtrip_reprojection_and_no_catpca_refit(method, monkeypatch):
    frame, base = fit_sample()
    fitted = method(base)
    restored = restore_summary(summary_state(fitted))
    assert summary_state(restored) == summary_state(fitted)
    assert restored.to_latex() == fitted.to_latex()
    monkeypatch.setattr("openecon.econometrics.categorical.frequency_pca.catpca_fweight", lambda *a, **kw: pytest.fail("Refit"))
    monkeypatch.setattr("openecon.econometrics.categorical.frequency_rotation._varimax", lambda *a, **kw: pytest.fail("Rotation optimizer"))
    result = catpca_rotated_predict_fweight(restored, frame.iloc[[9, 2, 13]])
    expected = fitted["scores"].iloc[[9, 2, 13], 1:].to_numpy(float)
    np.testing.assert_allclose(result["scores"].iloc[:, 1:].to_numpy(float), expected, atol=1e-12)
    raw = catpca_fweight_predict(base, frame.iloc[[9, 2, 13]])
    np.testing.assert_allclose(result["reconstruction"].iloc[:, 1:].to_numpy(float), raw["scores"].iloc[:, 1:].to_numpy(float)@base["loadings"].to_numpy(float).T, atol=1e-12)


@pytest.mark.parametrize("mutation", ["transform", "trace", "frequency", "table", "extra", "version"])
def test_coherent_resealing_does_not_forge_rotation(mutation):
    _, base = fit_sample()
    result = copy.deepcopy(catpca_promax_fweight(base))
    state = result.attrs["rotation_state"]
    if mutation == "transform":
        state["transform"][0][0] += .1
    elif mutation == "trace":
        state["trace"][-1][3] += .1
        result["rotation_iterations"].iloc[-1, -1] += .1
    elif mutation == "frequency":
        result.attrs["frequency_total"] += 1
    elif mutation == "table":
        result["structure"].iloc[0, 0] += .1
    elif mutation == "extra":
        state["extra"] = 0
    else:
        state["version"] = True
    result.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError):
        _checked(result, 128*1024**2, 300_000_000, "cpu")


def test_budget_refusal_precedes_tensor_copy(monkeypatch):
    _, base = fit_sample()
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Copied tensor before admission"))
    with pytest.raises(AnalysisError, match="work"):
        catpca_varimax_fweight(base, max_work=1)


@pytest.mark.parametrize("weighted,unweighted", [(catpca_varimax_fweight, catpca_varimax), (catpca_promax_fweight, catpca_promax)])
def test_literal_replicated_population_rotation_oracle(weighted, unweighted):
    frame, base = fit_sample()
    expanded = frame.loc[frame.index.repeat(frame["count"])].reset_index(drop=True)
    expanded_base = catpca(expanded, list("abcd"), scales=dict.fromkeys("abcd", "numeric"), components=2, n_starts=1)
    actual, expected = weighted(base), unweighted(expanded_base)
    for name in ("pattern", "structure", "component_correlations", "transformation"):
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-10)


@pytest.mark.parametrize("method", [catpca_varimax_fweight, catpca_promax_fweight])
def test_original_category_centroids_keep_ordinal_ties_and_exact_counts(method):
    rng = np.random.default_rng(413)
    x = rng.normal(size=48)
    frame = pd.DataFrame(dict(a=rng.choice(["red", "blue", "green"], 48),
                             b=rng.choice(["low", "mid", "high"], 48), z=x,
                             q=.7*x+rng.normal(size=48), w=rng.integers(1, 5, 48)))
    base = catpca_fweight(frame, ["a", "b", "z", "q"], frequency="w", components=2,
                          scales={"a": "nominal", "b": "ordinal", "z": "numeric", "q": "numeric"},
                          orders={"b": ["low", "mid", "high"]}, n_starts=2, maxiter=300, tol=1e-11)
    mapping = base["quantifications"].query("variable == 'b'").set_index("category")
    assert mapping.loc["low", "quantification"] == pytest.approx(mapping.loc["mid", "quantification"], abs=1e-13)
    result = method(base)
    scores = result["scores"].iloc[:, 1:].to_numpy(float)
    for row in result["category_centroids"].itertuples(index=False):
        mask = frame[row.variable] == row.category
        expected = np.average(scores[mask], axis=0, weights=frame.w[mask])
        np.testing.assert_allclose([row.component_1, row.component_2], expected, atol=1e-12)
        assert row.frequency_count == frame.w[mask].sum()
    centers = result["category_centroids"].query("variable == 'b'").set_index("category")
    assert np.linalg.norm(centers.loc["low"].iloc[-2:].to_numpy(float)-centers.loc["mid"].iloc[-2:].to_numpy(float)) > .1
    restored = restore_summary(summary_state(result))
    _checked(restored, 128*1024**2, 300_000_000, "cpu")


def test_empty_query_and_large_counts_do_not_expand_rows(monkeypatch):
    frame = rotation_sample()
    frame["count"] = 10_000_000
    base = catpca_fweight(frame, list("abcd"), frequency="count", components=2,
                          scales=dict.fromkeys("abcd", "numeric"), n_starts=1)
    result = catpca_varimax_fweight(base)
    assert result.attrs["frequency_total"] == 640_000_000
    assert len(result["scores"]) == 64
    with pytest.raises(AnalysisError):
        catpca_rotated_predict_fweight(result, frame.iloc[:0])
    query = frame.iloc[:1].copy()
    query["a"] = np.nan
    projected = catpca_rotated_predict_fweight(result, query, missing="drop")
    assert projected["scores"].shape == (0, 3)


def test_reuse_and_query_work_are_admitted_together_before_tensor(monkeypatch):
    frame, base = fit_sample()
    result = catpca_varimax_fweight(base)
    state = result.attrs["rotation_state"]
    fit_plan = _plan(base, state["controls"], source_bytes=len(state["source_summary"].encode()))
    allowance = fit_plan["planned_work"]+64*len(frame)*4*2//2
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Tensor before aggregate budget"))
    with pytest.raises(AnalysisError, match="work"):
        catpca_rotated_predict_fweight(result, frame, max_work=allowance)


def test_embedded_json_nesting_is_refused_before_restore(monkeypatch):
    _, base = fit_sample()
    result = catpca_varimax_fweight(base)
    raw = "["*100+"0"+"]"*100
    import hashlib
    state = result.attrs["rotation_state"]
    state["source_summary"] = raw
    state["source_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    result.attrs["state_sha256"] = _seal(state)
    monkeypatch.setattr("openecon.econometrics.categorical.frequency_rotation.restore_summary", lambda *a: pytest.fail("Parsed nested JSON"))
    with pytest.raises(AnalysisError, match="bounded structure"):
        _checked(result, 128*1024**2, 300_000_000, "cpu")
