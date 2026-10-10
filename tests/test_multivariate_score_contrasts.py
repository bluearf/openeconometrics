"""Independent complete-refit references for eight simultaneous score domains."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import score_contrasts as u
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget

import test_multivariate_score_uncertainty as sr

DOMAINS = sr.DOMAINS


def contrast_matrix(domain, rows=3):
    k = 4 if domain.startswith("cca") else 2
    matrix = np.random.default_rng(188).normal(size=(3, rows * k)) / 3
    matrix[0] = 0
    matrix[0, 0], matrix[0, -1] = 1, -1  # crosses query positions and CCA X/Y blocks
    matrix[1] = 0
    matrix[1, k - 1], matrix[1, k] = .5, 1
    return matrix


def call(domain, fit, query, matrix, **kwargs):
    function = oe.pca_fweight_bootstrap_score_contrasts if domain.startswith("pca") else \
        oe.canon_bootstrap_score_contrasts if domain.startswith("cca") else oe.factor_bootstrap_score_contrasts
    return function(fit, query, matrix, **kwargs)


def reference(point, draws, matrix, level):
    point, draws = matrix @ point, draws @ matrix.T
    covariance = np.cov(draws, rowvar=False, ddof=1)
    se = np.sqrt(np.diag(covariance))
    deviations = (draws - point) / se
    maxima = np.max(np.abs(deviations), axis=1)
    critical = np.quantile(maxima, level, method="linear")
    return point, draws, covariance, se, deviations, maxima, critical


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("distribution", ["normal", "skew_t8"])
def test_all_eight_complete_independent_refits_and_joint_bands(domain, distribution):
    frame, names, weight, target, fit, query = sr.cached(domain, distribution)
    before = summary_state(fit)
    matrix = contrast_matrix(domain)
    out = call(domain, fit, query, matrix, labels=["difference", "paired sum", "fixed combination"])
    point, draws, *_ = sr.oracle(domain, frame, names, weight, target, fit, query)
    point, draws, covariance, se, deviations, maxima, critical = reference(point, draws, matrix, .9)
    np.testing.assert_allclose(out["estimates"].estimate, point, atol=5e-11)
    np.testing.assert_allclose(out["replicates"], draws, atol=5e-11)
    np.testing.assert_allclose(out["covariance"], covariance, atol=5e-11)
    np.testing.assert_allclose(out["estimates"].std_error, se, atol=5e-11)
    np.testing.assert_allclose(out["standardized_deviations"], deviations, atol=5e-10)
    np.testing.assert_allclose(out["joint_maxima"].absolute_maximum, maxima, atol=5e-10)
    np.testing.assert_allclose(out.attrs["critical_value"], critical, atol=5e-10)
    np.testing.assert_allclose(out["estimates"][["ci_lower", "ci_upper"]],
                               np.stack((point - critical * se, point + critical * se), axis=1), atol=5e-10)
    np.testing.assert_allclose(out["estimates"].bootstrap_bias, draws.mean(0) - point, atol=5e-11)
    np.testing.assert_allclose(out["estimates"][["marginal_percentile_lower", "marginal_percentile_upper"]],
                               np.quantile(draws, [.05, .95], axis=0).T, atol=5e-11)
    source_cov = out["score__covariance"].to_numpy()
    np.testing.assert_allclose(covariance, matrix @ source_cov @ matrix.T, atol=5e-11)
    assert out["estimates"][["p_value", "df"]].isna().all().all()
    assert out.attrs["familywise_intervals"] and out.attrs["covariance_divisor"] == 38
    assert "fixed" in out.attrs["standardization"] and "first-order" in out.attrs["uncertainty"]
    assert summary_state(fit) == before
    for name, frame in fit.items():
        pd.testing.assert_frame_equal(out["score__fit__" + name], frame)


@pytest.mark.parametrize("domain", DOMAINS)
def test_sign_scale_permutation_and_duplicate_family_invariants(domain):
    *_, fit, query = sr.cached(domain)
    matrix = contrast_matrix(domain)
    base = call(domain, fit, query, matrix)
    scale = np.array([2., -3., .25])
    changed = call(domain, fit, query, matrix * scale[:, None])
    np.testing.assert_allclose(changed.attrs["critical_value"], base.attrs["critical_value"], atol=1e-12)
    np.testing.assert_allclose(changed["replicates"], base["replicates"] * scale, atol=1e-12)
    np.testing.assert_allclose(changed["covariance"], base["covariance"] * np.outer(scale, scale), atol=1e-12)
    original_bounds = base["estimates"][["ci_lower", "ci_upper"]].to_numpy()
    expected = np.sort(original_bounds * scale[:, None], axis=1)
    np.testing.assert_allclose(changed["estimates"][["ci_lower", "ci_upper"]], expected, atol=1e-12)
    permutation = [2, 0, 1, 2]
    redundant = call(domain, fit, query, matrix[permutation])
    np.testing.assert_allclose(redundant.attrs["critical_value"], base.attrs["critical_value"], atol=1e-12)
    np.testing.assert_allclose(redundant["replicates"], base["replicates"].to_numpy()[:, permutation], atol=1e-12)
    assert np.linalg.matrix_rank(redundant["covariance"]) < 4


@pytest.mark.parametrize("domain", DOMAINS)
def test_missing_physical_coordinates_typed_identity_and_complete_restore(domain):
    *_, fit, query = sr.setup(domain, missing=True)
    query.index = pd.Index([("duplicate", 1), ("duplicate", 1), pd.Timestamp("2026-01-04", tz="UTC")],
                           tupleize_cols=False, name="query")
    query.columns.name = "measurements"
    query.iloc[1, 0] = np.nan
    k = 4 if domain.startswith("cca") else 2
    matrix = contrast_matrix(domain)
    matrix[:, k:2 * k] = 0
    out = call(domain, fit, query, matrix)
    assert out.attrs["score_attrs"]["query_positions"] == [0, 2]
    assert out["contrasts"].shape[1] == 3 * k and out["retained_contrasts"].shape[1] == 2 * k
    assert summary_state(restore_summary(summary_state(out))) == summary_state(out)
    restored_fit = restore_summary(summary_state(fit))
    replay = call(domain, restored_fit, query.copy(), out["contrasts"].copy())
    assert summary_state(replay) == summary_state(out)
    bad = matrix.copy()
    bad[0, k] = 1
    with pytest.raises(AnalysisError, match="missing physical"):
        call(domain, fit, query, bad)
    with pytest.raises(AnalysisError):
        call(domain, fit, query, matrix, missing="raise")


@pytest.mark.parametrize("domain", DOMAINS)
def test_large_location_and_unit_scaling_independent_refits(domain):
    frame, names, weight, target, fit, query = sr.setup(domain, shift=1e12, scale=3.)
    matrix = contrast_matrix(domain)
    out = call(domain, fit, query, matrix, confidence=.95)
    point, draws, *_ = sr.oracle(domain, frame, names, weight, target, fit, query)
    point, draws, covariance, _, _, _, critical = reference(point, draws, matrix, .95)
    np.testing.assert_allclose(out["replicates"], draws, atol=1e-10)
    np.testing.assert_allclose(out["estimates"].estimate, point, atol=1e-10)
    np.testing.assert_allclose(out["covariance"], covariance, atol=1e-10)
    np.testing.assert_allclose(out.attrs["critical_value"], critical, atol=1e-9)


@pytest.mark.parametrize("domain", DOMAINS)
def test_zero_variance_repeated_query_difference_refuses_entire_family(domain):
    *_, fit, query = sr.cached(domain)
    query = pd.concat([query.iloc[:1]] * 2)
    k = 4 if domain.startswith("cca") else 2
    matrix = np.zeros((2, 2 * k))
    matrix[0, 0] = 1
    matrix[1, 0], matrix[1, k] = 1, -1
    with pytest.raises(AnalysisError, match="cancelled bootstrap variance"):
        call(domain, fit, query, matrix)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("mutation", ["draw", "sample", "seed"])
def test_resealed_semantic_training_forgery_rejected(domain, mutation):
    *_, original, query = sr.cached(domain)
    fit = restore_summary(summary_state(original))
    if mutation == "seed":
        fit.attrs["seed"] += 1
    else:
        fit["replicates" if mutation == "draw" else "sample"].iloc[0, 0] += .001
    sr.reseal(fit)
    with pytest.raises(AnalysisError, match="canonical|replay"):
        call(domain, fit, query, contrast_matrix(domain))


@pytest.mark.parametrize("bad", [None, [], [1, 2], [[1], [1, 2]], np.ones((2, 7)),
                                 np.ones((65, 6)), np.ones((1, 6, 1)), np.zeros((1, 6)),
                                 np.ones((1, 6), dtype=bool), np.ones((1, 6), dtype=complex),
                                 [[True, 0, 0, 0, 0, 0]], [[1j, 0, 0, 0, 0, 0]],
                                 np.full((1, 6), np.nan), np.full((1, 6), np.inf),
                                 np.ones((1, 6), dtype=object)])
def test_invalid_contrasts_refuse_before_canonical_refit(monkeypatch, bad):
    *_, fit, query = sr.cached("pca_covariance")
    monkeypatch.setattr(u.s, "_replay", lambda *a: pytest.fail("invalid contrast executed a training replay"))
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, bad)


@pytest.mark.parametrize("bad", [["x", "x", "z"], ["x"], ["x", "y", "z" * 5000], [1, 2, 3]])
def test_invalid_labels_refuse_before_canonical_refit(monkeypatch, bad):
    *_, fit, query = sr.cached("pca_covariance")
    monkeypatch.setattr(u.s, "_replay", lambda *a: pytest.fail("invalid labels replayed"))
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, contrast_matrix("pca_covariance"), labels=bad)


@pytest.mark.parametrize("bad", [0, 1, .999, True, float("nan"), "0.9"])
def test_invalid_confidence_or_tail_resolution_refuses_before_replay(monkeypatch, bad):
    *_, fit, query = sr.cached("cca_iid")
    monkeypatch.setattr(u.s, "_replay", lambda *a: pytest.fail("invalid confidence replayed"))
    with pytest.raises(AnalysisError):
        call("cca_iid", fit, query, contrast_matrix("cca_iid"), confidence=bad)


def test_labelled_geometry_exact_alignment_and_numeric_tensor_contract(monkeypatch):
    *_, fit, query = sr.cached("pca_covariance")
    matrix = contrast_matrix("pca_covariance")
    plain = call("pca_covariance", fit, query, matrix)
    labelled = plain["contrasts"].copy()
    assert summary_state(call("pca_covariance", fit, query, labelled)) == summary_state(plain)
    tensor = call("pca_covariance", fit, query, torch.tensor(matrix))
    np.testing.assert_array_equal(tensor["replicates"], plain["replicates"])
    monkeypatch.setattr(u.s, "_replay", lambda *a: pytest.fail("invalid labelled geometry replayed"))
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, labelled.iloc[:, ::-1])
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, torch.tensor(matrix, requires_grad=True))
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, labelled, labels=list(labelled.index))


def test_budget_and_missing_hypothesis_refuse_before_conversion_or_refit(monkeypatch):
    *_, fit, query = sr.cached("pca_covariance")
    matrix = contrast_matrix("pca_covariance")
    monkeypatch.setattr(u, "_matrix", lambda *a: pytest.fail("budget denial converted contrast matrix"))
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, matrix)


def test_nonzero_missing_coordinate_refuses_before_training_replay(monkeypatch):
    *_, fit, query = sr.cached("cca_iid")
    query = query.copy()
    query.iloc[0, 0] = np.nan
    monkeypatch.setattr(u.s, "_replay", lambda *a: pytest.fail("missing hypothesis replayed"))
    with pytest.raises(AnalysisError, match="missing physical"):
        call("cca_iid", fit, query, contrast_matrix("cca_iid"))


@pytest.mark.parametrize("domain", ["pf_iid", "target_iid", "pf_frequency", "target_frequency"])
@pytest.mark.parametrize("factors", [3, 4])
def test_three_and_four_factor_complete_independent_joint_reference(domain, factors):
    frame, names, weight, target, fit, query = sr.setup(domain, factors=factors)
    matrix = np.random.default_rng(184).normal(size=(3, 3 * factors))
    out = call(domain, fit, query, matrix)
    point, draws, *_ = sr.oracle(domain, frame, names, weight, target, fit, query)
    point, draws, covariance, _, _, _, critical = reference(point, draws, matrix, .9)
    np.testing.assert_allclose(out["replicates"], draws, atol=1e-10)
    np.testing.assert_allclose(out["estimates"].estimate, point, atol=1e-10)
    np.testing.assert_allclose(out["covariance"], covariance, atol=1e-10)
    np.testing.assert_allclose(out.attrs["critical_value"], critical, atol=1e-9)


def test_resident_matrix_boundaries_and_tail_rank(monkeypatch):
    *_, fit, query = sr.cached("pca_covariance")
    matrix = contrast_matrix("pca_covariance")
    boundary = call("pca_covariance", fit, query, np.tile(matrix[0], (64, 1)), confidence=.975)
    assert boundary["covariance"].shape == (64, 64)
    np.testing.assert_allclose(boundary["joint_maxima"].absolute_maximum,
                               boundary["standardized_deviations"].iloc[:, 0].abs(), atol=1e-12)
    monkeypatch.setattr(u.s, "_replay", lambda *a: pytest.fail("invalid resident input replayed"))
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, torch.empty((3, 6), device="meta"))
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query.iloc[:0], np.empty((1, 0)))
    with pytest.raises(AnalysisError):
        call("pca_covariance", fit, query, matrix, confidence=.976)
