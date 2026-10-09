"""Literal-expanded NumPy PF and SciPy full-target bootstrap oracles."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.linalg import orthogonal_procrustes

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import factor_frequency_uncertainty as u
from openecon.econometrics.multivariate import factor_uncertainty as old_u
from openecon.econometrics.multivariate import frequency_bootstrap as f
from openecon.econometrics.multivariate.factor import factor
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def fixture(domain="normal", *, n=603, factors=2, seed=623):
    rng = np.random.default_rng(seed)
    if factors == 2:
        target = np.array([[.8, 0], [.75, .1], [.7, -.05], [0, .65], [.1, .6], [-.05, .55]])
    else:
        target = np.zeros((3 * factors, factors))
        for j, strength in enumerate([.85, .72, .6, .5][:factors]):
            target[3 * j:3 * j + 3, j] = [strength, strength - .04, strength - .08]
    p = len(target)
    if domain == "normal":
        latent, errors = rng.normal(size=(n, factors)), rng.normal(size=(n, p))
    else:
        latent = (rng.lognormal(sigma=.5, size=(n, factors)) - np.exp(.125)) / np.sqrt(np.expm1(.25) * np.exp(.25))
        errors = rng.standard_t(8, size=(n, p)) / np.sqrt(8 / 6)
    values = latent @ target.T + errors * np.sqrt(1 - (target * target).sum(1))
    scales = np.array([2, 3, 1, 4, .5, 1.8]) if p == 6 else np.linspace(.5, 3, p)
    means = np.array([5, -2, 12, .5, 3, 8]) if p == 6 else np.arange(p) - 3
    frame = pd.DataFrame(values * scales + means, columns=[f"x{i + 1}" for i in range(p)])
    frame["frequency"] = 2 + np.arange(n) % 3
    frame.loc[np.arange(n) % 17 == 0, "frequency"] = 0
    return frame, target


def oracle(values, factors, anchors=None, target=None):
    shifted = values - values[0]
    offset = shifted.mean(0)
    centered = shifted - offset
    correction = centered.mean(0)
    centered -= correction
    covariance = centered.T @ centered / (len(values) - 1)
    sd = np.sqrt(covariance.diagonal())
    correlation = covariance / np.outer(sd, sd)
    np.fill_diagonal(correlation, 1.)
    smc = 1 - 1 / np.linalg.inv(correlation).diagonal()
    reduced = correlation.copy()
    np.fill_diagonal(reduced, smc)
    roots, vectors = np.linalg.eigh(reduced)
    roots, vectors = roots[::-1], vectors[:, ::-1]
    loadings = vectors[:, :factors] * np.sqrt(roots[:factors])
    if target is None:
        anchors = list(np.abs(loadings).argmax(0)) if anchors is None else anchors
        loadings *= np.where(loadings[anchors, np.arange(factors)] < 0, -1, 1)
    else:
        loadings = loadings @ orthogonal_procrustes(loadings, target)[0]
    uniqueness = 1 - (loadings * loadings).sum(1)
    return dict(vector=np.r_[loadings.ravel(), uniqueness], loadings=loadings,
                uniqueness=uniqueness, roots=roots, smc=smc, covariance=covariance,
                correlation=correlation, mean=values[0] + offset + correction, sd=sd)


def fit(frame, *, target=None, factors=2, **options):
    names = [name for name in frame if name != "frequency"]
    defaults = dict(weights="frequency", factors=factors, target=target, replications=199, seed=19)
    defaults.update(options)
    return u.factor_multifactor_fweight_bootstrap(frame, names, **defaults)


def literal_draws(frame, factors, *, target=None, anchors=None, b=199, seed=19):
    values = frame.drop(columns="frequency").to_numpy()
    frequencies = frame.frequency.to_numpy(dtype=np.int64)
    ordinals = np.repeat(np.arange(len(frame)), frequencies)
    expanded = values[ordinals]
    point = oracle(expanded, factors, anchors, target)
    if target is None and anchors is None:
        # Fix point anchors once, independently of every replicate.
        anchors = np.abs(point["loadings"]).argmax(0).tolist()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts, refs = [], []
    for _ in range(b):
        ranks = torch.randint(len(expanded), (len(expanded),), generator=generator).numpy()
        counts.append(np.bincount(ordinals[ranks], minlength=len(frame)))
        refs.append(oracle(expanded[ranks], factors, anchors, target))
    return point, np.array(counts), refs


@pytest.mark.parametrize("domain", ["normal", "skew_t8"])
@pytest.mark.parametrize("rotated", [False, True])
def test_full_199_literal_expansion_numpy_scipy_vector_covariance_percentile_oracle(domain, rotated):
    frame, target = fixture(domain)
    target = target if rotated else None
    result = fit(frame, target=target)
    point, counts, refs = literal_draws(frame, 2, target=target)
    vectors = np.array([ref["vector"] for ref in refs])
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    assert np.all(counts.sum(1) == frame.frequency.sum())
    assert not counts[:, frame.frequency.to_numpy() == 0].any()
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=2e-11)
    np.testing.assert_allclose(result["replicates"], vectors, atol=2e-11, rtol=2e-11)
    cov = np.cov(vectors, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], cov, atol=3e-13, rtol=4e-9)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(cov.diagonal()), atol=2e-11)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]], np.quantile(vectors, [.025, .975], axis=0).T, atol=2e-11)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias, vectors.mean(0) - point["vector"], atol=2e-11)
    for key, table in (("mean", "replicate_means"), ("sd", "replicate_standard_deviations"),
                       ("roots", "replicate_eigenvalues"), ("smc", "replicate_smc")):
        np.testing.assert_allclose(result[table], [r[key] for r in refs], atol=2e-11)
    np.testing.assert_allclose(result["replicate_covariances"], [r["covariance"].ravel() for r in refs], atol=5e-11)
    for key, table in (("loadings", "point_loadings"), ("covariance", "point_covariance"),
                       ("correlation", "point_correlation")):
        np.testing.assert_allclose(result[table], point[key], atol=2e-11)
    np.testing.assert_allclose(result["point_uniqueness"].uniqueness, point["uniqueness"], atol=2e-11)
    np.testing.assert_allclose(result["replicate_origins"] + result["replicate_mean_offsets"], result["replicate_means"], atol=0)
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert result.attrs["moment_covariance_divisor"] == int(frame.frequency.sum()) - 1
    assert result.attrs["covariance_divisor"] == 198
    if rotated:
        rotation = result["point_rotation_matrix"].to_numpy()
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(2), atol=2e-12)


@pytest.mark.parametrize("factors", [3, 4])
@pytest.mark.parametrize("rotated", [False, True])
def test_three_four_factor_full_literal_reference(factors, rotated):
    frame, target = fixture(factors=factors)
    target = target if rotated else None
    result = fit(frame, factors=factors, target=target, replications=39, confidence=.90)
    point, counts, refs = literal_draws(frame, factors, target=target, b=39)
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    np.testing.assert_allclose(result["replicates"], [r["vector"] for r in refs], atol=3e-11)
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=3e-11)


def test_point_frequency_pf_agreement_and_loading_covariance_invariants():
    frame, target = fixture()
    names = list(frame.drop(columns="frequency"))
    result = fit(frame, target=target, replications=39, confidence=.90)
    old = factor(frame, names, factors=2, method="pf", rotate="target", target=target, kaiser=False, weights="frequency")
    np.testing.assert_allclose(result["point_loadings"], old["rotated_loadings"], atol=2e-12)
    np.testing.assert_allclose(result["point_uniqueness"].uniqueness, old["uniqueness"].uniqueness, atol=2e-12)
    for vector in result["replicates"].to_numpy():
        loadings, uniqueness = vector[:12].reshape(6, 2), vector[12:]
        np.testing.assert_allclose(np.diag(loadings @ loadings.T) + uniqueness, np.ones(6), atol=2e-12)


@pytest.mark.parametrize("rotated", [False, True])
def test_json_saved_state_typed_index_missing_accounting_and_semantic_count_replay(rotated):
    frame, target = fixture(n=403)
    target = target if rotated else None
    frame.index = pd.Index([("unit", i % 5) for i in range(len(frame))], tupleize_cols=False, name="duplicated")
    frame.iloc[1, 0] = np.nan
    frame.iloc[4, -1] = np.nan
    complete = frame.dropna()
    result = fit(frame, target=target, replications=39, confidence=.90)
    restored = restore_summary(json.loads(json.dumps(summary_state(result), allow_nan=False)))
    f.validate_saved(restored)
    names = list(complete.drop(columns="frequency"))
    anchors = [names.index(name) for name in result.attrs["sign_anchors"]] if not rotated else None
    for counts, vector in zip(restored["replicate_counts"].to_numpy(dtype=np.int64), restored["replicates"].to_numpy(), strict=True):
        values = np.repeat(complete[names].to_numpy(), counts, axis=0)
        np.testing.assert_allclose(oracle(values, 2, anchors, target)["vector"], vector, atol=3e-11)
    rerun = fit(complete.reset_index(drop=True), target=target, replications=39, confidence=.90)
    np.testing.assert_array_equal(rerun["replicate_counts"], result["replicate_counts"])
    np.testing.assert_allclose(rerun["replicates"], result["replicates"], atol=0)


@pytest.mark.parametrize("field", ["replicate_counts", "source_frequencies", "replicate_means", "point_eigenvalues", "covariance"])
def test_saved_integrity_binds_complete_counts_and_scientific_tables(field):
    result = fit(fixture(n=303)[0], replications=39, confidence=.90)
    changed = copy.deepcopy(result)
    changed[field].iloc[0, 0] += 1
    with pytest.raises(AnalysisError):
        f.validate_saved(changed)


@pytest.mark.parametrize("rotated", [False, True])
def test_translation_units_and_zero_frequency_outliers_match_literal_oracle(rotated):
    frame, target = fixture(n=403)
    target = target if rotated else None
    names = list(frame.drop(columns="frequency"))
    frame[names] = frame[names] * [.2, 2, 1, 3, .5, 1] + 1e12
    frame.loc[frame.frequency == 0, names] = -1e145
    result = fit(frame, target=target, replications=39, confidence=.90)
    point, counts, refs = literal_draws(frame, 2, target=target, b=39)
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=7e-11)
    np.testing.assert_allclose(result["replicates"], [r["vector"] for r in refs], atol=7e-11)


@pytest.mark.parametrize("fail_all", [False, True])
def test_every_replication_attempted_any_failure_refuses_all_uncertainty(monkeypatch, fail_all):
    original, calls = u._parameters, []
    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1 and (fail_all or len(calls) in (4, 8)):
            raise AnalysisError("heywood_case", "forced draw-only failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(u, "_parameters", failing)
    with pytest.raises(AnalysisError, match="frequency PF") as caught:
        fit(fixture(n=303)[0], replications=39, confidence=.90)
    assert len(calls) == 40 and caught.value.replications_attempted == 39
    assert len(caught.value.failures) == (39 if fail_all else 2)


@pytest.mark.parametrize("target", [True, [[True, False]] * 6, [["1", "0"]] * 6,
                                    [[1 + 1j, 0]] * 6, np.zeros((6, 2)), np.ones((6, 1)),
                                    np.full((6, 2), np.inf)])
def test_full_target_refuses_nonreal_shape_nonfinite_and_unidentified_polar(target):
    with pytest.raises(AnalysisError):
        fit(fixture(n=303)[0], target=target, replications=39, confidence=.90)


@pytest.mark.parametrize("kind", ["rank", "constant", "subnormal", "zero", "huge_total"])
def test_rank_moment_precision_and_frequency_resource_refusals(kind):
    frame, _ = fixture(n=103)
    if kind == "rank":
        frame.x2 = frame.x1
    elif kind == "constant":
        frame.x1 = 1.
    elif kind == "subnormal":
        frame.x1 *= 1e-156
    elif kind == "zero":
        frame.frequency = 0
    else:
        frame.frequency = 100001
    with pytest.raises(AnalysisError):
        fit(frame, replications=39, confidence=.90)


@pytest.mark.parametrize("options", [dict(factors=1), dict(factors=True), dict(weight_type="pweight"),
                                     dict(anchors=["wrong", "x2"]), dict(target=np.ones((6, 2)), anchors=["x1", "x4"]),
                                     dict(replications=18), dict(seed=True), dict(confidence=1), dict(missing="pairwise")])
def test_strict_fixed_options(options):
    with pytest.raises(AnalysisError):
        fit(fixture(n=103)[0], **options)


def test_zero_frequency_raw_rows_are_retained_but_never_materialized_as_units(monkeypatch):
    frame, target = fixture(n=303)
    def forbidden(*args, **kwargs):
        raise AssertionError("expanded measurements or unweighted raw refit")
    monkeypatch.setattr(torch, "repeat_interleave", forbidden)
    monkeypatch.setattr(old_u, "_parameters", forbidden)
    result = fit(frame, target=target, replications=39, confidence=.90)
    assert len(result["sample"]) == len(frame)
    assert not result["replicate_counts"].to_numpy()[:, frame.frequency == 0].any()


def test_workspace_refuses_before_numeric_tensor_allocation(monkeypatch):
    frame, _ = fixture(n=103)
    def forbidden(*args, **kwargs):
        raise AssertionError("numeric tensor copy before admission")
    monkeypatch.setattr(torch, "tensor", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        fit(frame, replications=39, confidence=.90)


def block_frame(strengths=(.65, .65), n=120):
    correlation = np.eye(6)
    for j, strength in enumerate(strengths):
        block = slice(3 * j, 3 * j + 3)
        correlation[block, block] = strength
        np.fill_diagonal(correlation[block, block], 1.)
    rng = np.random.default_rng(924)
    values = rng.normal(size=(n, 6))
    values -= values.mean(0)
    values = np.linalg.qr(values)[0] @ np.linalg.cholesky(correlation).T * np.sqrt(n - 1)
    frame = pd.DataFrame(values, columns=[f"x{i+1}" for i in range(6)])
    frame["frequency"] = 1
    target = np.array([[.8, 0]] * 3 + [[0, .8]] * 3)
    return frame, target


def test_internal_repeated_roots_identify_fixed_target_but_not_unrotated_axes():
    frame, target = block_frame()
    names = list(frame.drop(columns="frequency"))
    sample = f.prepare(frame, names, weights="frequency", weight_type="fweight", replications=39,
                       confidence=.90, seed=19, missing="drop", parameters=18)
    moment = f.moments(sample.values, sample.counts)
    with pytest.raises(AnalysisError, match="eigenvalue gap"):
        u._parameters(moment, names, 2, None, None)
    fitted = u._parameters(moment, names, 2, torch.tensor(target), None)
    expected = oracle(frame[names].to_numpy(), 2, target=target)
    np.testing.assert_allclose(fitted.vector(), expected["vector"], atol=2e-12)
    result = fit(frame, target=target, replications=39, confidence=.90)
    assert result.attrs["internal_repeated_roots_allowed"]
    assert not result.attrs["point_unrotated_axes_identified"]


def test_full_target_reflections_and_labelled_target_order():
    frame, target = fixture(n=303)
    names = list(frame.drop(columns="frequency"))
    reflection = target * [-1, 1]
    labelled = pd.DataFrame(reflection, index=names, columns=["Factor1", "Factor2"])
    result = fit(frame, target=labelled, replications=39, confidence=.90)
    point, _, refs = literal_draws(frame, 2, target=reflection, b=39)
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=2e-11)
    np.testing.assert_allclose(result["replicates"], [r["vector"] for r in refs], atol=2e-11)
    with pytest.raises(AnalysisError, match="in order"):
        fit(frame, target=labelled.iloc[::-1], replications=39, confidence=.90)


def test_uniqueness_boundary_is_refused_without_clipping(monkeypatch):
    original = u.ex.principal_factors
    def boundary(*args):
        extracted = original(*args)
        extracted.loadings[0] = torch.tensor([1.1, 0.], dtype=torch.float64)
        return extracted
    monkeypatch.setattr(u.ex, "principal_factors", boundary)
    with pytest.raises(AnalysisError, match="uniqueness"):
        fit(fixture(n=303)[0], replications=39, confidence=.90)


@pytest.mark.parametrize("measurement", [True, 1 + 2j])
def test_nonreal_source_measurements_are_rejected_before_coercion(measurement):
    frame, _ = fixture(n=103)
    frame["x1"] = frame.x1.astype(object)
    frame.iloc[1, 0] = measurement
    with pytest.raises(AnalysisError):
        fit(frame, replications=39, confidence=.90)


def test_weights_are_not_silently_rescaled_and_global_rng_is_preserved():
    frame, target = fixture(n=303)
    torch.manual_seed(424)
    before = torch.get_rng_state().clone()
    result = fit(frame, target=target, replications=39, confidence=.90)
    assert torch.equal(torch.get_rng_state(), before)
    doubled = frame.copy()
    doubled.frequency *= 2
    second = fit(doubled, target=target, replications=39, confidence=.90)
    assert second.attrs["n"] == 2 * result.attrs["n"]
    assert not np.allclose(second["covariance"], result["covariance"], atol=1e-8)
