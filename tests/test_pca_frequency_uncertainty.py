"""Literal-expanded IID-unit oracles for count-weighted PCA uncertainty."""
from __future__ import annotations

from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import frequency_bootstrap as f
from openecon.econometrics.multivariate import pca_frequency_uncertainty as p
from openecon.econometrics.multivariate import pca_subspace as s
from openecon.econometrics.multivariate import pca_uncertainty as u
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget

NAMES = ["a", "b", "c", "d", "e", "f"]


def domain(distribution="gaussian", *, n=603, seed=913):
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        latent, errors = rng.normal(size=(n, 2)), rng.normal(size=(n, 6))
    else:
        latent = rng.lognormal(sigma=.45, size=(n, 2))-np.exp(.45**2/2)
        latent /= np.sqrt(np.expm1(.45**2)*np.exp(.45**2))
        errors = rng.standard_t(8, size=(n, 6))/np.sqrt(8/6)
    loading = np.array([.95, .93, .9, .72, .7, .68])
    values = latent[:, [0, 0, 0, 1, 1, 1]]*loading
    values += errors*np.sqrt(1-loading**2)
    values = values*[3., 2.5, 2., 1.5, 1.3, 1.1]+[5., -2., 12., .5, 9., -3.]
    frame = pd.DataFrame(values, columns=NAMES)
    frame["freq"] = rng.integers(0, 5, n)
    return frame


def call(frame, *, subspace=False, matrix="correlation", components=2,
         replications=39, confidence=.9, seed=124, columns=NAMES, **kwargs):
    function = p.pca_subspace_fweight_bootstrap if subspace else p.pca_fweight_bootstrap
    return function(frame, columns, weights="freq", components=components,
                    matrix=matrix, replications=replications, confidence=confidence,
                    seed=seed, **kwargs)


def oracle(values, *, matrix, components, subspace, anchors=None):
    # Rectangular literal data, independently centered and decomposed by SVD.
    shifted = np.asarray(values)-values[0]
    mean = shifted.mean(0)
    centered = shifted-mean
    correction = centered.mean(0)
    centered -= correction
    covariance = centered.T@centered/(len(values)-1)
    sd = np.sqrt(covariance.diagonal())
    z = centered if matrix == "covariance" else centered/sd
    _, singular, right = np.linalg.svd(z, full_matrices=False)
    roots = singular**2/(len(values)-1)
    vectors = right[:components].T
    target = covariance if matrix == "covariance" else covariance/np.outer(sd, sd)
    if subspace:
        projector = vectors@vectors.T
        captured = np.trace(projector@target)
        residual = np.trace((np.eye(len(sd))-projector)@target)
        parameters = np.r_[projector[np.triu_indices(len(sd))], captured, residual,
                            captured/np.trace(target)]
        return dict(parameters=parameters, covariance=covariance, target=target,
                    sd=sd, mean=values[0]+mean+correction, projector=projector, roots=roots)
    anchors = np.abs(vectors).argmax(0) if anchors is None else np.asarray(anchors)
    vectors *= np.sign(vectors[anchors, np.arange(components)])
    loadings = vectors*np.sqrt(roots[:components])
    parameters = np.concatenate([np.r_[roots[j], vectors[:, j], loadings[:, j]]
                                 for j in range(components)])
    return dict(parameters=parameters, covariance=covariance, target=target,
                sd=sd, mean=values[0]+mean+correction, roots=roots,
                vectors=vectors, loadings=loadings, anchors=anchors)


def expanded_oracles(frame, *, matrix, subspace, components=2, seed=124, replications=39):
    weights = frame.freq.to_numpy(dtype=np.int64)
    values = frame[NAMES].to_numpy()
    expanded = np.repeat(values, weights, axis=0)
    source_rows = np.repeat(np.arange(len(values)), weights)
    point = oracle(expanded, matrix=matrix, components=components, subspace=subspace)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts, fits = [], []
    for _ in range(replications):
        indices = torch.randint(len(expanded), (len(expanded),), generator=generator,
                                device="cpu").numpy()
        counts.append(np.bincount(source_rows[indices], minlength=len(values)))
        fits.append(oracle(expanded[indices], matrix=matrix, components=components,
                           subspace=subspace, anchors=point.get("anchors")))
    return point, np.asarray(counts), fits


@pytest.mark.parametrize("distribution", ["gaussian", "skewed_t8"])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
@pytest.mark.parametrize("subspace", [False, True])
def test_two_domains_all_count_vectors_targets_joint_inference_and_moments(distribution, matrix, subspace):
    frame = domain(distribution)
    result = call(frame, matrix=matrix, subspace=subspace)
    point, counts, fits = expanded_oracles(frame, matrix=matrix, subspace=subspace)
    draws = np.array([value["parameters"] for value in fits])
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    np.testing.assert_array_equal(result["replicate_counts"].sum(1), int(frame.freq.sum()))
    np.testing.assert_array_equal(result["replicate_counts"].loc[:, frame.index[frame.freq == 0]], 0)
    np.testing.assert_allclose(result["replicates"], draws, atol=3e-11, rtol=3e-12)
    np.testing.assert_allclose(result["estimates"].estimate, point["parameters"], atol=3e-11)
    covariance = np.cov(draws, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], covariance, atol=2e-12, rtol=3e-10)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=3e-12)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(draws, [.05, .95], axis=0, method="linear").T, atol=3e-11)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias, draws.mean(0)-point["parameters"], atol=3e-11)
    np.testing.assert_allclose(result["point_covariance"], point["covariance"], atol=2e-12)
    np.testing.assert_allclose(result["point_matrix"], point["target"], atol=2e-12)
    np.testing.assert_allclose(result["point_eigenvalues"].eigenvalue, point["roots"], atol=2e-11)
    np.testing.assert_allclose(result["replicate_means"], [value["mean"] for value in fits], atol=3e-12)
    np.testing.assert_allclose(result["replicate_standard_deviations"], [value["sd"] for value in fits], atol=3e-12)
    np.testing.assert_allclose(result["replicate_origins"]+result["replicate_mean_offsets"], result["replicate_means"], atol=0)
    if subspace:
        np.testing.assert_allclose(result["point_projector"], point["projector"], atol=2e-12)
        np.testing.assert_allclose(result["replicate_diagnostics"].frobenius_distance_to_point_projector,
                                   [np.linalg.norm(v["projector"]-point["projector"]) for v in fits], atol=3e-12)
        assert not result.attrs["individual_eigenvalue_inference_available"]
        assert "point_eigenvectors" not in result
    else:
        np.testing.assert_allclose(result["point_eigenvectors"], point["vectors"], atol=2e-12)
        np.testing.assert_allclose(result["point_loadings"], point["loadings"], atol=2e-12)
        assert result.attrs["sign_anchors"] == [NAMES[i] for i in point["anchors"]]
    assert result["estimates"][["df", "p_value"]].isna().all().all()
    assert result.attrs["successful_replications"] == 39
    assert result.attrs["failed_replications"] == []
    assert not result.attrs["covariance_inverse_available"]
    assert not result.attrs["familywise_intervals"]
    f.validate_saved(result)


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
@pytest.mark.parametrize("subspace", [False, True])
def test_all_one_frequencies_equal_existing_iid_draws(matrix, subspace):
    frame = domain()
    frame["freq"] = 1
    weighted = call(frame, matrix=matrix, subspace=subspace)
    function = s.pca_subspace_bootstrap if subspace else u.pca_bootstrap
    ordinary = function(frame, NAMES, matrix=matrix, components=2, replications=39,
                        confidence=.9, seed=124)
    for key in ["estimates", "covariance", "replicates", "point_covariance", "point_matrix"]:
        np.testing.assert_allclose(weighted[key], ordinary[key], atol=3e-11, rtol=2e-11, equal_nan=True)
    expected = np.array([np.bincount(row, minlength=len(frame)) for row in ordinary["resample_indices"].to_numpy()])
    np.testing.assert_array_equal(weighted["replicate_counts"], expected)


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
@pytest.mark.parametrize("subspace", [False, True])
def test_large_location_against_represented_literal_oracle(matrix, subspace):
    frame = domain()
    frame[NAMES] += 1e12
    result = call(frame, matrix=matrix, subspace=subspace)
    point, counts, fits = expanded_oracles(frame, matrix=matrix, subspace=subspace)
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    np.testing.assert_allclose(result["point_covariance"], point["covariance"], atol=3e-12)
    np.testing.assert_allclose(result["replicates"], [v["parameters"] for v in fits], atol=3e-11, rtol=3e-12)


@pytest.mark.parametrize("subspace", [False, True])
def test_correlation_positive_units_invariance_and_covariance_scaling(subspace):
    frame = domain()
    result = call(frame, subspace=subspace)
    transformed = frame.copy()
    transformed[NAMES] = transformed[NAMES]*[.5, 2., 1.5, .25, 3., 4.]+[7., -8., 2., 3., -1., 11.]
    other = call(transformed, subspace=subspace)
    np.testing.assert_allclose(result["replicates"], other["replicates"], atol=3e-11)
    covariance = call(frame, matrix="covariance", subspace=subspace)
    scaled = frame.copy()
    scaled[NAMES] *= 3
    scaled = call(scaled, matrix="covariance", subspace=subspace)
    expected = covariance["replicates"].to_numpy().copy()
    if subspace:
        expected[:, -3:-1] *= 9
    else:
        for j in range(2):
            expected[:, j*13] *= 9
            expected[:, j*13+7:j*13+13] *= 3
    np.testing.assert_allclose(scaled["replicates"], expected, atol=3e-10, rtol=2e-12)


def represented(covariance, *, n=240):
    rng = np.random.default_rng(317)
    x = rng.normal(size=(n, len(covariance)))
    x -= x.mean(0)
    q, _ = np.linalg.qr(x)
    roots, vectors = np.linalg.eigh(covariance)
    x = q@(vectors*np.sqrt(roots))@vectors.T*np.sqrt(n-1)
    frame = pd.DataFrame(x, columns=NAMES[:len(covariance)])
    frame["freq"] = 1
    return frame


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_internal_ties_projector_rotation_invariance_and_axes_refusal(matrix, monkeypatch):
    covariance = np.diag([4., 4., 1., .5]) if matrix == "covariance" else np.array(
        [[1., .75, 0., 0.], [.75, 1., 0., 0.], [0., 0., 1., .75], [0., 0., .75, 1.]])
    frame = represented(covariance)
    columns = NAMES[:4]
    baseline = call(frame, matrix=matrix, subspace=True, columns=columns)
    with pytest.raises(AnalysisError, match="simple"):
        call(frame, matrix=matrix, columns=columns)
    original = torch.linalg.eigh
    def rotated(value):
        roots, vectors = original(value)
        turn = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
        vectors[:, -2:] = vectors[:, -2:]@turn
        return roots, vectors
    monkeypatch.setattr(torch.linalg, "eigh", rotated)
    rotated_result = call(frame, matrix=matrix, subspace=True, columns=columns)
    np.testing.assert_allclose(rotated_result["replicates"], baseline["replicates"], atol=8e-13)
    np.testing.assert_allclose(rotated_result["point_projector"], baseline["point_projector"], atol=5e-13)


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_boundary_tie_refuses_projector(matrix):
    covariance = np.diag([4., 4., 1., .5]) if matrix == "covariance" else np.array(
        [[1., .75, 0., 0.], [.75, 1., 0., 0.], [0., 0., 1., .75], [0., 0., .75, 1.]])
    with pytest.raises(AnalysisError, match="boundary"):
        call(represented(covariance), matrix=matrix, subspace=True, components=1, columns=NAMES[:4])


@pytest.mark.parametrize("subspace", [False, True])
def test_missing_zero_counts_typed_identity_complete_json_and_integrity(subspace):
    frame = domain(n=303)
    frame.index = pd.Index([1, True, "1", ("x", 2), *range(4, len(frame))], name="case")
    frame.iloc[1, 0] = np.nan
    frame.iloc[2, -1] = np.nan
    frame.iloc[3, -1] = 0
    copy = frame.copy(deep=True)
    result = call(frame, subspace=subspace)
    assert list(result["source_index"].source_position) == [0, *range(3, len(frame))]
    assert list(result["replicate_counts"].columns) == list(result["source_index"].source_position)
    assert (result["replicate_counts"][3] == 0).all()
    assert result.attrs["n_missing"] == 2
    saved = json.loads(json.dumps(summary_state(result), allow_nan=False))
    restored = restore_summary(saved)
    for key in result:
        left = pd.DataFrame(result[key]).astype(object).where(pd.notna(result[key]), None)
        right = pd.DataFrame(restored[key]).astype(object).where(pd.notna(restored[key]), None)
        pd.testing.assert_frame_equal(left, right)
    f.validate_saved(restored)
    restored["replicate_counts"].iloc[0, 0] += 1
    with pytest.raises(AnalysisError):
        f.validate_saved(restored)
    pd.testing.assert_frame_equal(frame, copy)
    with pytest.raises(AnalysisError) as caught:
        call(frame, subspace=subspace, missing="raise")
    assert caught.value.code == "missing_values"


@pytest.mark.parametrize("subspace", [False, True])
def test_one_failed_count_draw_refuses_whole_result_after_all_attempts(subspace, monkeypatch):
    draw = f.draw_counts
    attempts = []
    def invalid_once(sample, generator):
        counts = draw(sample, generator)
        attempts.append(1)
        if len(attempts) == 7:
            counts.zero_()
            counts[0] = sample.total
        return counts
    monkeypatch.setattr(f, "draw_counts", invalid_once)
    with pytest.raises(AnalysisError) as caught:
        call(domain(), subspace=subspace)
    assert caught.value.code == "bootstrap_failure"
    assert caught.value.replications_attempted == len(attempts) == 39
    assert caught.value.successful_replications == 38
    assert len(caught.value.failures) == 1 and caught.value.failures[0]["replication"] == 7


@pytest.mark.parametrize("bad", [-1., .5, np.inf, True, 2**53+2])
def test_invalid_integer_frequency_domain(bad):
    frame = domain()
    if isinstance(bad, bool):
        frame["freq"] = True
    else:
        frame["freq"] = frame.freq.astype(float)
        frame.loc[0, "freq"] = bad
    with pytest.raises(AnalysisError):
        call(frame)


@pytest.mark.parametrize("settings", [{"weight_type": "aweight"}, {"components": True},
    {"matrix": "precision"}, {"replications": 18}, {"replications": 2000},
    {"confidence": .99}, {"seed": -1}, {"seed": 2**63}, {"seed": True},
    {"missing": "pairwise"}, {"anchors": ["unknown", "b"]}])
def test_invalid_settings(settings):
    with pytest.raises(AnalysisError):
        call(domain(), **settings)


@pytest.mark.parametrize("subspace", [False, True])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_raw_subnormal_variance_refused_before_restandardization(subspace, matrix):
    frame = domain()
    frame[NAMES] *= 1e-157
    with pytest.raises(AnalysisError, match="subnormal|precision|rescale"):
        call(frame, subspace=subspace, matrix=matrix)


@pytest.mark.parametrize("subspace", [False, True])
def test_limits_before_sampler_and_without_expansion(subspace, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("sampler/expansion must not run")
    frame = domain()
    monkeypatch.setattr(torch, "repeat_interleave", fail)
    result = call(frame, subspace=subspace)
    assert len(result["sample"]) == len(frame)
    monkeypatch.setattr(f, "draw_counts", fail)
    frame["freq"] = 1000
    with pytest.raises(AnalysisError):
        call(frame, subspace=subspace)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError):
            call(domain(), subspace=subspace)


@pytest.mark.parametrize("subspace", [False, True])
def test_measurement_graph_and_oversized_numeric_identity_refused(subspace):
    frame = domain()
    mapping = {name: torch.tensor(frame[name].to_numpy(), dtype=torch.float64) for name in [*NAMES, "freq"]}
    mapping[NAMES[0]].requires_grad_(True)
    with pytest.raises(AnalysisError):
        call(mapping, subspace=subspace)
    frame.index = pd.Index([Decimal("1"*10000), *range(1, len(frame))], dtype=object)
    with pytest.raises(AnalysisError):
        call(frame, subspace=subspace)


@pytest.mark.parametrize("subspace", [False, True])
def test_zero_frequency_extreme_rows_do_not_enter_origin_or_arithmetic(subspace):
    frame = domain()
    frame.loc[0, "freq"] = 0
    frame.loc[0, NAMES] = 1e300
    result = call(frame, subspace=subspace)
    point, counts, fits = expanded_oracles(frame, matrix="correlation", subspace=subspace)
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    np.testing.assert_allclose(result["estimates"].estimate, point["parameters"], atol=3e-11)
    np.testing.assert_allclose(result["replicates"], [v["parameters"] for v in fits], atol=3e-11)
    assert np.all(result["point_origin"].to_numpy() < 1e100)


def test_invalid_weight_on_missing_measurement_row_is_still_refused():
    frame = domain()
    frame.loc[0, "a"] = np.nan
    frame["freq"] = frame.freq.astype(float)
    frame.loc[0, "freq"] = .5
    with pytest.raises(AnalysisError):
        call(frame, missing="drop")


@pytest.mark.parametrize("kind", ["rows", "work", "portable", "parameters"])
def test_preallocation_admission_sentinels(kind, monkeypatch):
    frame = domain(n=10001 if kind == "rows" else 1000)
    options = {}
    if kind == "work":
        frame["freq"] = 100
        options.update(columns=NAMES[:2], components=1, replications=199)
    elif kind in ("portable", "parameters"):
        for i in range(6, 16):
            frame[f"x{i}"] = np.arange(len(frame))
        options.update(columns=[*NAMES, *[f"x{i}" for i in range(6, 16)]],
                       components=4 if kind == "parameters" else 1,
                       replications=39 if kind == "parameters" else 1999)
    def fail(*args, **kwargs):
        raise AssertionError("weighted fit/sampler must not run")
    monkeypatch.setattr(f, "moments", fail)
    monkeypatch.setattr(f, "draw_counts", fail)
    with pytest.raises(AnalysisError) as caught:
        call(frame, **options)
    if kind == "work":
        assert caught.value.code == "work_limit"


def test_maximum_bounded_units_use_single_rank_vector_and_compressed_counts(monkeypatch):
    frame = domain(n=200)
    frame["freq"] = 500
    original = torch.randint
    shapes = []
    def recording(*args, **kwargs):
        shapes.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "randint", recording)
    result = call(frame, columns=NAMES[:2], components=1, replications=19, confidence=.8)
    assert shapes == [(100000,)]*19
    assert result["replicate_counts"].shape == (19, 200)
    assert result.attrs["n_units"] == 100000
    assert result.attrs["expanded_measurements_materialized"] is False


@pytest.mark.parametrize("seed", [0, 2**63-1])
def test_seed_extremes_and_deterministic_complete_state(seed):
    a = call(domain(), seed=seed)
    b = call(domain(), seed=seed)
    assert summary_state(a) == summary_state(b)
