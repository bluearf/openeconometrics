"""Independent rotation oracles, coherent geometry and portable state checks."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.optimal import catpca, catpca_predict
from openecon.econometrics.categorical.rotation import (
    _seal, catpca_promax, catpca_rotated_predict, catpca_varimax,
)
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def numeric_sample(n=170, dimensions=2):
    rng = np.random.default_rng(61)
    latent = rng.normal(size=(n, dimensions))
    latent[:, -1] += .45*latent[:, 0]
    loadings = np.array([[.9, .1, .2], [.8, .2, .15], [.15, .95, .1], [.3, .8, .15], [.4, .2, .9], [.1, .3, .8]])[:, :dimensions]
    values = latent@loadings.T+.3*rng.normal(size=(n, 6))
    return pd.DataFrame(values, columns=list("abcdef"))


def fit_numeric(dimensions=2):
    frame = numeric_sample(dimensions=dimensions)
    return frame, catpca(frame, list(frame), scales=dict.fromkeys(frame, "numeric"), components=dimensions)


def geometry(result, name):
    return result[name].to_numpy(dtype=float)


def identify(loadings, transform):
    pattern = loadings@transform
    order = sorted(range(pattern.shape[1]), key=lambda j: (-np.sum(pattern[:, j]**2), j))
    transform = transform[:, order].copy()
    pattern = loadings@transform
    for j in range(pattern.shape[1]):
        if pattern[np.argmax(np.abs(pattern[:, j])), j] < 0:
            transform[:, j] *= -1
    return transform


def criterion(loadings):
    return np.sum(np.sum(loadings**4, axis=0)-np.sum(loadings**2, axis=0)**2/len(loadings))


def independent_varimax_2d(loadings, normalize):
    # Globally bracket the periodic one-dimensional criterion, then a scalar
    # optimizer. This is independent of the production planar angle formula.
    base = loadings/np.linalg.norm(loadings, axis=1)[:, None] if normalize else loadings

    def rotation(angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def objective(angle):
        return -criterion(base@rotation(angle))

    grid = np.linspace(-np.pi/4, np.pi/4, 1001)
    best = int(np.argmin([objective(angle) for angle in grid]))
    step = grid[1]-grid[0]
    solution = minimize_scalar(objective, bounds=(grid[best]-step, grid[best]+step), method="bounded", options={"xatol": 1e-14})
    return identify(loadings, rotation(solution.x)), -solution.fun


@pytest.mark.parametrize("normalize", [False, True])
def test_varimax_independent_global_2d_oracle_and_reconstruction(normalize):
    _, base = fit_numeric()
    original = geometry(base, "loadings")
    result = catpca_varimax(base, normalize=normalize, tol=1e-12)
    oracle, objective = independent_varimax_2d(original, normalize)
    transform = geometry(result, "transformation")
    np.testing.assert_allclose(transform, oracle, atol=3e-8)
    scaled = original/np.linalg.norm(original, axis=1)[:, None] if normalize else original
    assert criterion(scaled@transform) == pytest.approx(objective, abs=2e-12)
    np.testing.assert_allclose(transform.T@transform, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(geometry(result, "pattern"), original@transform, atol=1e-14)
    np.testing.assert_allclose(geometry(result, "structure"), geometry(result, "pattern"), atol=1e-14)
    np.testing.assert_allclose(geometry(result, "scores")[:, 1:], geometry(base, "scores")[:, 1:]@transform, atol=1e-14)
    np.testing.assert_allclose(geometry(result, "reconstruction")[:, 1:], geometry(base, "scores")[:, 1:]@original.T, atol=1e-14)
    assert result["fit"].reconstruction_loss.iloc[0] == pytest.approx(base["fit"].reconstruction_loss.iloc[0], abs=1e-14)
    assert np.all(result["rotation_iterations"].improvement.to_numpy()[1:] >= -1e-13)


@pytest.mark.parametrize("normalize,power", [(False, 2), (True, 4), (True, 1), (False, 10)])
def test_promax_independent_powered_target_lstsq_and_oblique_geometry(normalize, power):
    _, base = fit_numeric()
    loadings = geometry(base, "loadings")
    result = catpca_promax(base, normalize=normalize, power=power, tol=1e-12)
    # Use independently optimized varimax, then NumPy least squares and matrix
    # inverses. Signed/permuted varimax columns give the same identified result.
    varimax_matrix, _ = independent_varimax_2d(loadings, normalize)
    varimax = loadings@varimax_matrix
    target_base = varimax/np.linalg.norm(loadings, axis=1)[:, None] if normalize else varimax
    target = np.sign(target_base)*np.abs(target_base)**power
    least_squares = np.linalg.lstsq(varimax, target, rcond=None)[0]
    raw = varimax_matrix@least_squares
    raw_phi = np.linalg.inv(raw)@np.linalg.inv(raw).T
    expected = identify(loadings, raw*np.sqrt(np.diag(raw_phi))[None, :])
    transform = geometry(result, "transformation")
    np.testing.assert_allclose(transform, expected, atol=1e-7)
    phi = geometry(result, "component_correlations")
    pattern, structure = geometry(result, "pattern"), geometry(result, "structure")
    scores = geometry(result, "scores")[:, 1:]
    np.testing.assert_allclose(phi, np.linalg.inv(transform.T@transform), atol=1e-13)
    np.testing.assert_allclose(np.diag(phi), 1, atol=1e-14)
    np.testing.assert_allclose(scores.T@scores/len(scores), phi, atol=1e-13)
    np.testing.assert_allclose(pattern, loadings@transform, atol=1e-13)
    np.testing.assert_allclose(structure, pattern@phi, atol=1e-13)
    np.testing.assert_allclose(pattern@phi@pattern.T, loadings@loadings.T, atol=1e-13)
    np.testing.assert_allclose(scores@pattern.T, geometry(base, "scores")[:, 1:]@loadings.T, atol=1e-13)
    if power == 1 and not normalize:
        np.testing.assert_allclose(phi, np.eye(2), atol=1e-13)
    elif power == 4:
        assert abs(phi[0, 1]) > .1
        assert np.max(np.abs(pattern-structure)) > .1


@pytest.mark.parametrize("method", [catpca_varimax, catpca_promax])
def test_3d_stationary_geometry_and_new_person_projection(method):
    frame, base = fit_numeric(3)
    rotated = method(base, tol=1e-12)
    new = frame.iloc[[7, 2, 45, 16]].copy()
    new.iloc[0] += .7
    restored = restore_summary(summary_state(rotated))
    predicted = catpca_rotated_predict(restored, new)
    base_predicted = catpca_predict(base, new)
    transform = geometry(rotated, "transformation")
    np.testing.assert_allclose(geometry(predicted, "scores")[:, 1:], geometry(base_predicted, "scores")[:, 1:]@np.linalg.inv(transform).T, atol=1e-12)
    np.testing.assert_allclose(geometry(predicted, "reconstruction"), geometry(base_predicted, "reconstruction"), atol=1e-12)
    assert rotated["rotation_iterations"].relative_stationarity.iloc[-1] <= 1e-12
    assert np.all(np.diff(rotated["rotation_iterations"].varimax_criterion) >= -1e-12)
    assert summary_state(restored) == summary_state(rotated)
    assert restored.to_latex() == rotated.to_latex()
    saved_prediction = restore_summary(summary_state(predicted))
    assert summary_state(saved_prediction) == summary_state(predicted)
    assert saved_prediction.to_latex() == predicted.to_latex()
    assert all(name not in table.columns for name in ("std_error", "p_value", "ci_lower") for table in rotated.values())


def mixed_sample():
    frame = numeric_sample()
    frame["a"] = pd.cut(frame.a, [-np.inf, -.6, .6, np.inf], labels=["red", "green", "blue"]).astype(str)
    frame["b"] = pd.cut(frame.b, [-np.inf, -.5, .5, np.inf], labels=["low", "mid", "high"]).astype(str)
    return frame


def test_mixed_maps_missing_unknown_levels_and_full_restoration():
    frame = mixed_sample()
    frame.loc[6, "c"] = np.nan
    scales = dict.fromkeys(frame, "numeric")
    scales.update(a="nominal", b="ordinal")
    base = catpca(frame, list(frame), scales=scales, orders={"b": ["low", "mid", "high"]}, tol=1e-10, maxiter=200)
    rotated = catpca_promax(restore_summary(summary_state(base)))
    assert rotated.attrs["sample_positions"] == base.attrs["sample_positions"]
    assert 6 not in rotated.attrs["sample_positions"]
    assert rotated["quantifications"].to_numpy().tolist() == base["quantifications"].to_numpy().tolist()
    assert "centroids" not in rotated
    assert "omitted" in rotated.attrs["category_coordinates"]
    restored = restore_summary(summary_state(rotated))
    predicted = catpca_rotated_predict(restored, frame, missing="drop")
    np.testing.assert_allclose(geometry(predicted, "scores"), geometry(rotated, "scores"), atol=3e-12)
    assert predicted.attrs["sample_positions"] == base.attrs["sample_positions"]
    with pytest.raises(AnalysisError, match="Unknown category"):
        catpca_rotated_predict(restored, frame.fillna(0).assign(a="unobserved"))
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(restored, frame)
    assert error.value.code == "missing_values"


@pytest.mark.parametrize("key", ["transform", "pattern", "structure", "phi", "scores", "reconstruction", "varimax_transform"])
def test_rehashed_rotation_numeric_tampering_is_refused(key):
    frame, base = fit_numeric()
    rotated = catpca_promax(base)
    restored = restore_summary(summary_state(rotated))
    restored.attrs["rotation_state"][key][0][0] += .03
    restored.attrs["state_sha256"] = _seal(restored.attrs["rotation_state"])
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(restored, frame.iloc[:3])
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("key", ["scores", "transformed", "pattern", "component_correlations", "rotation_iterations", "fit", "quantifications"])
def test_display_table_tampering_is_refused(key):
    frame = mixed_sample()
    base = catpca(frame, list(frame), scales={**dict.fromkeys(frame, "numeric"), "a": "nominal", "b": "nominal"})
    rotated = catpca_varimax(base)
    last = rotated[key].shape[1]-1
    rotated[key].iloc[0, last] += .03
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(rotated, frame.iloc[:2])
    assert error.value.code == "invalid_state"


def tampered_embedded_base(rotated, mutate):
    result = restore_summary(summary_state(rotated))
    rotation = result.attrs["rotation_state"]
    base_payload = json.loads(rotation["base_summary"])
    mutate(base_payload)
    base_payload["attrs"]["state_sha256"] = _seal(base_payload["attrs"]["optimal_state"])
    source = json.dumps(base_payload, sort_keys=True)
    rotation["base_summary"] = source
    rotation["base_summary_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    result.attrs["state_sha256"] = _seal(rotation)
    return result


@pytest.mark.parametrize("where", ["eigenvalues", "axes", "scores", "positions", "map"])
def test_rehashed_embedded_source_tampering_is_refused(where):
    frame = mixed_sample()
    base = catpca(frame, list(frame), scales={**dict.fromkeys(frame, "numeric"), "a": "nominal", "b": "nominal"})
    rotated = catpca_promax(base)

    def mutate(payload):
        state = payload["attrs"]["optimal_state"]
        if where == "eigenvalues":
            state["eigenvalues"][0] *= 1.1
        elif where == "axes":
            state["axes"][0][0] += .1
        elif where == "scores":
            payload["tables"]["scores"]["data"][0][1] += .1
        elif where == "positions":
            payload["attrs"]["sample_positions"][0] = 1
        else:
            state["descriptors"][0]["counts"][0] += 1

    altered = tampered_embedded_base(rotated, mutate)
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(altered, frame.iloc[:2])
    assert error.value.code == "invalid_state"


def test_stationary_but_wrong_varimax_transform_is_refused():
    frame, base = fit_numeric()
    rotated = catpca_varimax(base)
    altered = restore_summary(summary_state(rotated))
    # Integrity alone cannot establish the claimed estimator. A changed power
    # must fail the method-specific powered-target checks even with a new hash.
    altered.attrs["rotation_state"]["power"] = 4
    altered.attrs["power"] = 4
    altered.attrs["state_sha256"] = _seal(altered.attrs["rotation_state"])
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(altered, frame.iloc[:2])
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("problem", ["text_matrix", "extra_field", "oversized_trace"])
def test_forged_state_is_refused_before_checksum_serialization(monkeypatch, problem):
    import openecon.econometrics.categorical.rotation as module

    frame, base = fit_numeric()
    result = catpca_promax(base)
    state = result.attrs["rotation_state"]
    if problem == "text_matrix":
        state["pattern"][0][0] = "9"*100_000
    elif problem == "extra_field":
        state["extra"] = [1]*100_000
    else:
        state["trace"] *= 1000

    def no_serialization(*args, **kwargs):
        raise AssertionError("Forged payload was serialized before admission")

    monkeypatch.setattr(module, "_seal", no_serialization)
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(result, frame.iloc[:2])
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("options", [{"max_iter": 0}, {"tol": float("nan")}, {"normalize": 1}, {"device": "mps"}, {"max_bytes": False}])
def test_invalid_options(options):
    _, base = fit_numeric()
    with pytest.raises(AnalysisError):
        catpca_varimax(base, **options)


@pytest.mark.parametrize("power", [0, 11, True, float("inf")])
def test_invalid_promax_power(power):
    _, base = fit_numeric()
    with pytest.raises(AnalysisError):
        catpca_promax(base, power=power)


def test_nonconvergence_and_before_allocation_budget(monkeypatch):
    _, base = fit_numeric(3)
    with pytest.raises(AnalysisError) as error:
        catpca_varimax(base, max_iter=1, tol=1e-12)
    assert error.value.code == "nonconvergence"
    rotated = catpca_promax(base)
    source_calls = []

    def no_tensor(*args, **kwargs):
        source_calls.append(1)
        raise AssertionError("Numerical copies happened before resource refusal")

    monkeypatch.setattr(torch, "tensor", no_tensor)
    for options in ({"max_bytes": 1}, {"max_work": 1}):
        with pytest.raises(AnalysisError) as error:
            catpca_varimax(base, **options)
        assert error.value.code in ("workspace_limit", "resource_limit")
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            catpca_varimax(base)
        assert error.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(rotated, numeric_sample(), max_bytes=1)
    assert error.value.code == "workspace_limit"
    assert not source_calls


def test_default_meta_device_cannot_move_rotation_or_projection():
    frame, base = fit_numeric()
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        rotated = catpca_promax(base)
        projected = catpca_rotated_predict(rotated, frame.iloc[:3])
    finally:
        torch.set_default_device(previous)
    assert rotated.attrs["device"] == projected.attrs["device"] == "cpu"
    assert rotated.attrs["dtype"] == projected.attrs["dtype"] == "float64"


def test_single_component_and_wrong_method_are_refused():
    frame = numeric_sample()
    base = catpca(frame, list(frame), scales=dict.fromkeys(frame, "numeric"), components=1)
    with pytest.raises(AnalysisError):
        catpca_varimax(base)
    with pytest.raises(AnalysisError) as error:
        catpca_rotated_predict(base, frame)
    assert error.value.code == "invalid_result"
