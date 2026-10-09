"""Independent optimal-scaling oracles and complete saved-map contracts."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.optimal import (
    _pava, _pca_update, catpca, catpca_predict, catreg_nominal,
    catreg_ordinal, catreg_predict,
)
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def sample(n=180):
    rng = np.random.default_rng(44)
    frame = pd.DataFrame({
        "a": rng.choice(["red", "blue", "green"], n),
        "b": rng.choice(["low", "mid", "high"], n),
        "z": rng.normal(size=n),
    })
    frame["y"] = (frame.a.map({"red": 2, "blue": -1, "green": .2})
                  + frame.b.map({"low": 2, "mid": 1, "high": -2})
                  + .8*frame.z + rng.normal(scale=.3, size=n))
    return frame


def dummy_oracle(frame):
    columns = [np.ones(len(frame)), frame.z.to_numpy()]
    for name in ("a", "b"):
        levels = sorted(frame[name].unique())
        columns.extend((frame[name] == level).to_numpy().astype(float) for level in levels[1:])
    matrix = np.column_stack(columns)
    beta = np.linalg.lstsq(matrix, frame.y, rcond=None)[0]
    return matrix@beta


def ordinal_oracle(frame):
    # Independent bounded quadratic least squares in cumulative category
    # increments. Enumerate effect direction; nominal contrasts are unrestricted.
    base = [np.ones(len(frame)), frame.z.to_numpy()]
    for level in ("blue", "green"):
        base.append((frame.a == level).to_numpy().astype(float))
    positions = frame.b.map({"low": 0, "mid": 1, "high": 2}).to_numpy()
    fits = []
    for direction in (1, -1):
        matrix = np.column_stack(base+[direction*(positions >= k) for k in (1, 2)])
        lower = np.array([-np.inf]*len(base)+[0, 0])
        solution = lsq_linear(matrix, frame.y, bounds=(lower, np.full(matrix.shape[1], np.inf)), tol=1e-13, lsmr_tol=1e-13)
        fits.append((np.sum((frame.y-matrix@solution.x)**2), matrix@solution.x))
    return min(fits, key=lambda value: value[0])[1]


def enumerate_isotonic(values, weights):
    # Exhaust every contiguous partition rather than implementing PAVA.
    best = None
    for breaks in itertools.product((False, True), repeat=len(values)-1):
        stops = [i+1 for i, cut in enumerate(breaks) if cut]+[len(values)]
        score, start = np.empty(len(values)), 0
        for stop in stops:
            score[start:stop] = np.average(values[start:stop], weights=weights[start:stop])
            start = stop
        if np.any(np.diff(score) < -1e-14):
            continue
        objective = np.sum(weights*(score-values)**2)
        if best is None or objective < best[0]:
            best = (objective, score.copy())
    return best[1]


def check_maps(result):
    transformed = result["transformed"].iloc[:, 1:].to_numpy(dtype=float)
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=3e-14)
    np.testing.assert_allclose(np.mean(transformed**2, axis=0), 1, atol=3e-14)
    for _, rows in result["quantifications"].groupby("variable", sort=False):
        if rows.iloc[0]["scale"] == "ordinal":
            assert np.all(np.diff(rows.quantification.to_numpy()) >= -1e-12)
    for _, rows in result["iterations"].groupby("start"):
        assert np.all(np.diff(rows.objective.to_numpy()) <= 2e-9)


def test_nominal_matches_independent_dummy_ols_and_label_permutation():
    frame = sample()
    scales = {"a": "nominal", "b": "nominal", "z": "numeric"}
    result = catreg_nominal(frame, "y", ["a", "b", "z"], scales=scales, tol=1e-12)
    np.testing.assert_allclose(result["fitted"].fitted, dummy_oracle(frame), atol=3e-7, rtol=1e-7)
    renamed = frame.assign(a=frame.a.map({"red": "Z", "blue": "A", "green": "B"}))
    again = catreg_nominal(renamed, "y", ["a", "b", "z"], scales=scales, tol=1e-12)
    np.testing.assert_allclose(result["fitted"].fitted, again["fitted"].fitted, atol=1e-12)
    check_maps(result)
    assert len(result["starts"]) == 3
    assert result.attrs["converged"]
    assert "descriptive" in result.attrs["inference"]
    assert not {"p", "std_error", "ci_lower", "covariance"}.intersection(result["coefficients"].columns)


def test_ordinal_matches_independent_constrained_quadratic_fit():
    frame = sample()
    result = catreg_ordinal(frame, "y", ["a", "b", "z"],
                            scales={"a": "nominal", "b": "ordinal", "z": "numeric"},
                            orders={"b": ["low", "mid", "high"]}, tol=1e-12)
    np.testing.assert_allclose(result["fitted"].fitted, ordinal_oracle(frame), atol=3e-7, rtol=1e-7)
    assert result["coefficients"].set_index("term").loc["b", "beta_descriptive"] < 0
    check_maps(result)


def test_negative_ordinal_ties_enumerated_partition_oracle():
    counts, means = np.array([1, 3, 2, 4]), np.array([5, 3, 4, 1.])
    frame = pd.DataFrame({"x": np.repeat(["a", "b", "c", "d"], counts), "y": np.repeat(means, counts)})
    result = catreg_ordinal(frame, "y", ["x"], orders={"x": ["a", "b", "c", "d"]})
    oracle = -enumerate_isotonic(-means, counts)
    np.testing.assert_allclose(result["fitted"].fitted, np.repeat(oracle, counts), atol=1e-12)
    assert result["fit"].sse.iloc[0] == pytest.approx(1.2)
    assert result["fit"].r_squared.iloc[0] == pytest.approx(16/17)
    assert result["coefficients"].beta_descriptive.iloc[1] < 0
    check_maps(result)


@pytest.mark.parametrize("values,weights", [([3, 0, 2, 1], [1, 4, 2, 3]), ([2, 2, 1], [2, 3, 1]), ([4, 3, 2, 1], [1, 3, 5, 2])])
def test_weighted_pava_matches_all_contiguous_partitions(values, weights):
    got = _pava(torch.tensor(values, dtype=torch.float64), torch.tensor(weights, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, enumerate_isotonic(np.array(values, float), np.array(weights, float)))


def test_numeric_catreg_reduces_to_ordinary_regression_in_original_units():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(90, 3)), columns=["x", "z", "y"])
    frame.y += 4+2*frame.x-.5*frame.z
    result = catreg_nominal(frame, "y", ["x", "z"], scales={"x": "numeric", "z": "numeric"})
    matrix = np.column_stack([np.ones(len(frame)), frame.x, frame.z])
    fitted = matrix@np.linalg.lstsq(matrix, frame.y, rcond=None)[0]
    np.testing.assert_allclose(result["fitted"].fitted, fitted, atol=1e-12)
    check_maps(result)


def test_catpca_numeric_limit_scores_projection_and_reconstruction():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(120, 4))@np.array([[1, .7, .2, .1], [.1, 1, -.3, .6], [.2, .1, 1, .3], [.6, .1, .2, 1.]])
    frame = pd.DataFrame(values, columns=list("abcd"))
    result = catpca(frame, list("abcd"), scales=dict.fromkeys(list("abcd"), "numeric"), components=2)
    standardized = (values-values.mean(axis=0))/values.std(axis=0)
    u, singular, vh = np.linalg.svd(standardized, full_matrices=False)
    np.testing.assert_allclose(result["eigenvalues"].eigenvalue, singular**2/len(frame), atol=1e-12)
    loadings = result["loadings"].to_numpy(float)
    scores = result["scores"].iloc[:, 1:].to_numpy(float)
    np.testing.assert_allclose(scores.T@scores/len(frame), np.eye(2), atol=1e-12)
    np.testing.assert_allclose(scores@loadings.T, (u[:, :2]*singular[:2])@vh[:2], atol=1e-12)
    np.testing.assert_allclose(scores@scores.T/len(frame), u[:, :2]@u[:, :2].T, atol=1e-12)
    assert result["fit"].reconstruction_loss.iloc[0] == pytest.approx(np.sum(singular[2:]**2)/len(frame))
    prediction = catpca_predict(restore_summary(summary_state(result)), frame)
    np.testing.assert_allclose(prediction["scores"].iloc[:, 1:], scores, atol=1e-12)
    np.testing.assert_allclose(prediction["reconstruction"].iloc[:, 1:], scores@loadings.T, atol=1e-12)
    check_maps(result)


@pytest.mark.parametrize("scale", ["nominal", "ordinal"])
def test_catpca_binary_categorical_limit_equals_pca(scale):
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"a": rng.integers(0, 2, 100), "b": rng.normal(size=100), "c": rng.normal(size=100)})
    kwargs = {"orders": {"a": [0, 1]}} if scale == "ordinal" else {}
    result = catpca(frame, ["a", "b", "c"], components=2,
                    scales={"a": scale, "b": "numeric", "c": "numeric"}, **kwargs)
    z = (frame.to_numpy()-frame.to_numpy().mean(axis=0))/frame.to_numpy().std(axis=0)
    singular = np.linalg.svd(z, compute_uv=False)
    np.testing.assert_allclose(result["eigenvalues"].eigenvalue, singular**2/len(z), atol=1e-11)
    check_maps(result)


def test_catpca_nominal_category_block_independent_weighted_svd():
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(30, 3))
    raw -= raw.mean(axis=0)
    scores, _ = np.linalg.qr(raw)
    scores *= np.sqrt(30)
    codes = np.repeat(np.arange(4), [3, 7, 12, 8])
    means = np.stack([scores[codes == k].mean(axis=0) for k in range(4)])
    counts = np.bincount(codes)
    u, singular, vh = np.linalg.svd(means*np.sqrt(counts)[:, None], full_matrices=False)
    expected_q = np.sqrt(30)*u[:, 0]/np.sqrt(counts)
    expected_loss = 30-singular[0]**2
    descriptor = {"scale": "nominal", "levels": list(range(4))}
    got = _pca_update(torch.tensor(scores, dtype=torch.float64), torch.tensor(codes), descriptor, torch.ones(30, dtype=torch.float64)).numpy()
    loading = scores.T@got/30
    np.testing.assert_allclose(got[:, None]@loading[None, :], expected_q[codes, None]@(singular[0]/np.sqrt(30)*vh[0])[None, :], atol=1e-12)
    assert np.sum((got[:, None]-scores@loading[:, None])**2) == pytest.approx(expected_loss)


def test_catpca_genuine_multilevel_solution_monotone_and_reproducible():
    frame = sample(100)
    kwargs = dict(components=1, scales={"a": "nominal", "b": "ordinal", "z": "numeric"}, orders={"b": ["low", "mid", "high"]}, tol=1e-9)
    result = catpca(frame, ["a", "b", "z"], **kwargs)
    again = catpca(frame, ["a", "b", "z"], **kwargs)
    assert summary_state(result) == summary_state(again)
    check_maps(result)
    scores = result["scores"].iloc[:, 1:].to_numpy(float)
    transformed = result["transformed"].iloc[:, 1:].to_numpy(float)
    loading = result["loadings"].to_numpy(float)
    assert result["fit"].reconstruction_loss.iloc[0] == pytest.approx(np.sum((transformed-scores@loading.T)**2)/len(frame))
    assert result.attrs["iterations"] > 1


@pytest.mark.parametrize("kind", ["nominal", "ordinal", "pca"])
def test_complete_portable_restore_predictions_and_latex(kind):
    frame = sample(70)
    if kind == "nominal":
        result = catreg_nominal(frame, "y", ["a", "b"])
        predict = catreg_predict
    elif kind == "ordinal":
        result = catreg_ordinal(frame, "y", ["b"], orders={"b": ["low", "mid", "high"]})
        predict = catreg_predict
    else:
        result = catpca(frame, ["a", "b", "z"], components=1, scales={"a": "nominal", "b": "nominal", "z": "numeric"})
        predict = catpca_predict
    restored = restore_summary(summary_state(result))
    assert restored.attrs == result.attrs
    for name in result:
        pd.testing.assert_frame_equal(restored[name], result[name], check_dtype=False, check_index_type=False)
    assert summary_state(predict(restored, frame)) == summary_state(predict(result, frame))
    assert "\\begin{tabular}" in result.to_latex()
    assert "quantifications" in result.to_latex()
    assert result.to_latex() == restored.to_latex()
    state = json.loads(summary_state(result))
    state["attrs"]["optimal_state"]["descriptors"][0]["quantifications"][0] += .1
    with pytest.raises(AnalysisError, match="integrity"):
        predict(restore_summary(json.dumps(state)), frame)


def test_missing_sample_positions_unknown_levels_and_empty_prediction():
    frame = sample(40)
    frame.loc[1, "y"] = np.nan
    frame.loc[5, "a"] = None
    result = catreg_nominal(frame, "y", ["a", "b"])
    assert result.attrs["sample_positions"] == [i for i in range(40) if i not in (1, 5)]
    assert result.attrs["n_missing"] == 2
    with pytest.raises(AnalysisError, match="Missing"):
        catreg_nominal(frame, "y", ["a", "b"], missing="raise")
    with pytest.raises(AnalysisError, match="Unknown category"):
        catreg_predict(result, pd.DataFrame({"a": ["unseen"], "b": ["low"]}))
    prediction = catreg_predict(result, pd.DataFrame({"a": [None], "b": ["low"]}), missing="drop")
    assert len(prediction["predictions"]) == 0


@pytest.mark.parametrize("kwargs", [{"orders": {}}, {"orders": {"b": ["low", "high"]}}, {"orders": {"b": ["low", "mid", "mid"]}}])
def test_ordinal_order_refuses_incomplete_or_duplicate_geometry(kwargs):
    with pytest.raises(AnalysisError):
        catreg_ordinal(sample(30), "y", ["b"], **kwargs)


@pytest.mark.parametrize("kwargs", [{"weights": "z"}, {"device": "cuda"}, {"n_starts": 0}, {"seed": True}, {"tol": float("nan")}, {"max_bytes": 0}, {"scales": {"a": "nominal"}}, {"scales": "nominal"}])
def test_unsupported_options_and_invalid_controls(kwargs):
    with pytest.raises(AnalysisError):
        catreg_nominal(sample(30), "y", ["a", "b"], **kwargs)


def test_work_memory_global_budget_and_row_category_bounds():
    frame = sample(30)
    for kwargs in ({"max_work": 1}, {"max_bytes": 1}):
        with pytest.raises(AnalysisError):
            catreg_nominal(frame, "y", ["a", "b"], **kwargs)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError, match="workspace"):
            catreg_nominal(sample(3000), "y", ["a", "b", "z"], scales={"a": "nominal", "b": "nominal", "z": "numeric"}, max_work=10**12)
    with pytest.raises(AnalysisError, match="3000"):
        catreg_nominal(sample(3001), "y", ["a"])
    with pytest.raises(AnalysisError, match="32"):
        catreg_nominal(pd.DataFrame({"x": range(33), "y": range(33)}), "y", ["x"])


def test_selection_budget_precedes_copy_and_wide_mapping_is_projected(monkeypatch):
    import openecon.econometrics.categorical.optimal as optimal

    frame = sample(30)
    original = optimal._coerce_frame
    selected_columns = []

    def inspect(data):
        selected_columns.append(list(data))
        return original(data)

    monkeypatch.setattr(optimal, "_coerce_frame", inspect)
    with pytest.raises(AnalysisError, match="workspace"):
        catreg_nominal(frame.to_dict("list"), "y", ["a", "b"], max_bytes=1)
    assert selected_columns == []
    wide = frame.to_dict("list")
    wide.update({f"unused_{i}": [object()]*len(frame) for i in range(400)})
    wide["unused_scalar"] = object()
    result = catreg_nominal(wide, "y", ["a", "b"])
    assert selected_columns == [["y", "a", "b"]]
    assert result.attrs["n"] == len(frame)
    selected_columns.clear()
    records = [{**row, "unused": object()} for row in frame.to_dict("records")]
    catreg_nominal(records, "y", ["a", "b"])
    assert all(list(row) == ["y", "a", "b"] for row in selected_columns[0])


def test_nonconvergence_degenerate_and_rank_failures_are_explicit():
    with pytest.raises(AnalysisError, match="converged"):
        catreg_nominal(sample(), "y", ["a", "b"], maxiter=1, tol=1e-12)
    with pytest.raises(AnalysisError, match="converged"):
        catpca(sample(70), ["a", "b", "z"], components=1, scales={"a": "nominal", "b": "nominal", "z": "numeric"}, maxiter=1, tol=1e-12)
    with pytest.raises(AnalysisError, match="levels"):
        catreg_nominal(sample(30).assign(a="one"), "y", ["a"])
    frame = sample(30).assign(z2=lambda value: value.z)
    with pytest.raises(AnalysisError):
        catreg_nominal(frame, "y", ["z", "z2"], scales={"z": "numeric", "z2": "numeric"})


def test_resident_cpu_ignores_unrelated_default_device():
    frame = sample(40)
    try:
        torch.set_default_device("meta")
        result = catreg_nominal(frame, "y", ["a", "b"])
        prediction = catreg_predict(result, frame)
        pca = catpca(frame, ["a", "b", "z"], components=1, scales={"a": "nominal", "b": "nominal", "z": "numeric"})
        assert result.attrs["device"] == prediction.attrs["device"] == pca.attrs["device"] == "cpu"
    finally:
        torch.set_default_device("cpu")
