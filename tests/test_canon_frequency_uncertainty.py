"""Frequency CCA checked by literal-expanded covariance-pencil fits.

Only the seeded integer ranks share the production RNG. Literal data expansion,
NumPy moments and SciPy generalized-eigen CCA are independent of production
frequency moments, compressed QR, root charts and output summaries.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import linalg

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import canon_frequency_uncertainty as u
from openecon.econometrics.multivariate import frequency_bootstrap as f
from openecon.econometrics.multivariate.canon import canon
from openecon.econometrics.postest.index_codec import decode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget

X, Y = ["x1", "x2", "x3"], ["y1", "y2", "y3"]
NAMES = X + Y


def fixture(domain="normal", n=603, seed=621):
    rng = np.random.default_rng(seed)
    if domain == "normal":
        latent, errors = rng.normal(size=(n, 3)), rng.normal(size=(n, 6))
    else:
        latent = (rng.lognormal(sigma=.35, size=(n, 3)) - np.exp(.35**2 / 2)) \
            / np.sqrt(np.expm1(.35**2) * np.exp(.35**2))
        errors = rng.standard_t(8, size=(n, 6)) / np.sqrt(8 / 6)
    values = np.c_[latent + .35 * errors[:, :3],
                   latent * [.9, .5, .18] + errors[:, 3:] * [.35, .7, 1.]]
    values = values * [3, 1.7, 2.2, 4, .8, 1.4] + [5, -9, 13, .5, 8, -1]
    frame = pd.DataFrame(values, columns=NAMES)
    frame["frequency"] = 2 + np.arange(n) % 3
    frame.loc[np.arange(n) % 17 == 0, "frequency"] = 0
    return frame


def oracle(values, components, target, *, p=3, anchors=(0, 1)):
    shifted = values - values[0]
    offset = shifted.mean(0)
    centered = shifted - offset
    correction = centered.mean(0)
    centered -= correction
    covariance = centered.T @ centered / (len(values) - 1)
    sd = np.sqrt(covariance.diagonal())
    xx, yy, xy = covariance[:p, :p], covariance[p:, p:], covariance[:p, p:]
    square, a = linalg.eigh(xy @ linalg.solve(yy, xy.T, assume_a="pos"), xx)
    order = np.argsort(square)[::-1]
    roots = np.sqrt(np.maximum(0, square[order[:min(p, values.shape[1] - p)]]))
    a = a[:, order[:components]]
    b = linalg.solve(yy, xy.T @ a, assume_a="pos") / roots[:components]
    for j, anchor in enumerate(anchors[:components]):
        sign = 1 if a[anchor, j] > 0 else -1
        a[:, j] *= sign
        b[:, j] *= sign
    matrices = {"x_coefficients": a, "y_coefficients": b,
                "x_standardized_coefficients": sd[:p, None] * a,
                "y_standardized_coefficients": sd[p:, None] * b,
                "x_loadings": xx @ a / sd[:p, None], "y_loadings": yy @ b / sd[p:, None],
                "x_cross_loadings": xy @ b / sd[:p, None], "y_cross_loadings": xy.T @ a / sd[p:, None]}
    vector = roots[:components] if target == "correlations" else np.concatenate([
        np.r_[roots[j], *(matrix[:, j] for matrix in matrices.values())] for j in range(components)])
    return dict(vector=vector, roots=roots, matrices=matrices, mean=values[0] + offset + correction,
                sd=sd, covariance=covariance, correlation=covariance / np.outer(sd, sd))


def fit(frame, target="correlations", **options):
    defaults = dict(weights="frequency", components=2, target=target, replications=199, seed=19)
    if target == "coefficients":
        defaults["anchors"] = X[:2]
    defaults.update(options)
    return u.canon_fweight_bootstrap(frame, X, Y, **defaults)


def literal_draws(frame, *, components=2, target="correlations", b=199, seed=19):
    values = frame[NAMES].to_numpy()
    frequencies = frame.frequency.to_numpy(dtype=np.int64)
    expanded_ordinals = np.repeat(np.arange(len(frame)), frequencies)
    expanded = values[expanded_ordinals]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    counts, references = [], []
    for _ in range(b):
        ranks = torch.randint(len(expanded), (len(expanded),), generator=generator).numpy()
        counts.append(np.bincount(expanded_ordinals[ranks], minlength=len(frame)))
        references.append(oracle(expanded[ranks], components, target))
    return oracle(expanded, components, target), np.array(counts), references


@pytest.mark.parametrize("domain", ["normal", "skew_t8"])
@pytest.mark.parametrize("target", ["correlations", "coefficients"])
def test_complete_199_draw_literal_expansion_generalized_eigen_oracle(domain, target):
    frame = fixture(domain)
    result = fit(frame, target)
    point, counts, refs = literal_draws(frame, target=target)
    vectors = np.array([r["vector"] for r in refs])
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    assert counts.dtype == np.int64 and np.all(counts.sum(1) == frame.frequency.sum())
    assert not counts[:, frame.frequency.to_numpy() == 0].any()
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=2e-11)
    np.testing.assert_allclose(result["replicates"], vectors, atol=2e-11, rtol=2e-11)
    cov = np.cov(vectors, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], cov, atol=5e-13, rtol=4e-9)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(cov.diagonal()), atol=2e-11)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(vectors, [.025, .975], axis=0).T, atol=2e-11)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias, vectors.mean(0) - point["vector"], atol=2e-11)
    for key, table in (("mean", "replicate_means"), ("sd", "replicate_standard_deviations"),
                       ("roots", "replicate_correlations")):
        np.testing.assert_allclose(result[table], [r[key] for r in refs], atol=2e-11)
    np.testing.assert_allclose(result["replicate_covariances"], [r["covariance"].ravel() for r in refs], atol=4e-11)
    np.testing.assert_allclose(result["point_covariance"], point["covariance"], atol=2e-11)
    np.testing.assert_allclose(result["point_correlation"], point["correlation"], atol=2e-11)
    np.testing.assert_allclose(result["replicate_origins"] + result["replicate_mean_offsets"], result["replicate_means"], atol=0)
    if target == "coefficients":
        for key, matrix in point["matrices"].items():
            np.testing.assert_allclose(result["point_" + key], matrix, atol=2e-11)
        assert result.attrs["parameter_dimension"] == 50
        assert (result["replicate_diagnostics"].minimum_x_signed_direction_cosine > np.sqrt(.5)).all()
        assert (result["replicate_diagnostics"].minimum_y_signed_direction_cosine > np.sqrt(.5)).all()
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert result.attrs["moment_covariance_divisor"] == int(frame.frequency.sum()) - 1
    assert result.attrs["covariance_divisor"] == 198


def test_point_matches_existing_frequency_cca_and_every_draw_normalization():
    frame = fixture()
    result = fit(frame, "coefficients", replications=39, confidence=.90)
    old = canon(frame, X, Y, weights="frequency")
    np.testing.assert_allclose(result["point_correlations"].correlation, old["correlations"].correlation, atol=2e-12)
    p = len(X)
    for vector, flattened in zip(result["replicates"].to_numpy(), result["replicate_covariances"].to_numpy(), strict=True):
        covariance = flattened.reshape(6, 6)
        for j in range(2):
            v = vector[j * 25:(j + 1) * 25]
            a, b = v[1:4], v[4:7]
            np.testing.assert_allclose(a @ covariance[:p, :p] @ a, 1, atol=2e-12)
            np.testing.assert_allclose(b @ covariance[p:, p:] @ b, 1, atol=2e-12)
            np.testing.assert_allclose(a @ covariance[:p, p:] @ b, v[0], atol=2e-12)


@pytest.mark.parametrize("target", ["correlations", "coefficients"])
def test_saved_json_typed_identity_missing_zeros_and_full_count_refit(target):
    frame = fixture(n=403)
    frame.index = pd.MultiIndex.from_tuples([(i % 2, "same" if i % 3 else ("nested", i)) for i in range(len(frame))], names=["group", "unit"])
    frame.iloc[1, frame.columns.get_loc("x1")] = np.nan
    frame.iloc[3, frame.columns.get_loc("frequency")] = np.nan
    complete = frame.dropna(subset=NAMES + ["frequency"])
    original = frame.copy(deep=True)
    result = fit(frame, target, replications=39, confidence=.90)
    restored = restore_summary(json.loads(json.dumps(summary_state(result), allow_nan=False)))
    f.validate_saved(restored)
    pd.testing.assert_frame_equal(frame, original)
    np.testing.assert_allclose(restored["sample"], complete[NAMES])
    positions = list(result["source_index"].source_position)
    assert positions == [i for i, keep in enumerate(frame[NAMES + ["frequency"]].notna().all(1)) if keep]
    assert [decode(code) for code in result["source_index"].index_code] == list(complete.index)
    for counts, vector in zip(restored["replicate_counts"].to_numpy(dtype=np.int64), restored["replicates"].to_numpy(), strict=True):
        literal = np.repeat(complete[NAMES].to_numpy(), counts, axis=0)
        np.testing.assert_allclose(oracle(literal, 2, target)["vector"], vector, atol=2e-11)
    rerun = fit(complete.reset_index(drop=True), target, replications=39, confidence=.90)
    np.testing.assert_array_equal(rerun["replicate_counts"], result["replicate_counts"])
    np.testing.assert_allclose(rerun["replicates"], result["replicates"], atol=0)


@pytest.mark.parametrize("table_name", ["replicate_counts", "source_frequencies", "replicates", "replicate_covariances", "estimates"])
def test_complete_saved_integrity_refuses_changed_fields(table_name):
    result = fit(fixture(n=303), replications=39, confidence=.90)
    changed = copy.deepcopy(result)
    changed[table_name].iloc[0, 0] += 1
    with pytest.raises(AnalysisError):
        f.validate_saved(changed)


@pytest.mark.parametrize("target", ["correlations", "coefficients"])
def test_large_origin_and_units_agree_with_literal_oracle(target):
    frame = fixture(n=403)
    frame[NAMES] = frame[NAMES] * np.array([.5, 2, .2, 3, 1, .7]) + 1e12
    result = fit(frame, target, replications=39, confidence=.90)
    point, counts, refs = literal_draws(frame, target=target, b=39)
    np.testing.assert_array_equal(result["replicate_counts"], counts)
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=8e-11)
    np.testing.assert_allclose(result["replicates"], [r["vector"] for r in refs], atol=8e-11)
    np.testing.assert_allclose(result["replicate_mean_offsets"],
                               np.asarray([r["mean"] for r in refs]) - result["replicate_origins"].to_numpy(), atol=3e-4)


@pytest.mark.parametrize("weight", [-1, .5, float("inf"), True, complex(1, 1), 2**53 + 1])
def test_invalid_frequency_admission(weight):
    frame = fixture(n=103)
    frame["frequency"] = frame.frequency.astype(object)
    frame.iloc[1, -1] = weight
    with pytest.raises(AnalysisError):
        fit(frame, replications=39, confidence=.90)


@pytest.mark.parametrize("change", [dict(weight_type="aweight"), dict(components=True), dict(target="loadings"),
                                    dict(target="correlations", anchors=["x1", "x2"]), dict(seed=True),
                                    dict(replications=18), dict(confidence=1), dict(missing="pairwise")])
def test_strict_settings(change):
    with pytest.raises(AnalysisError):
        fit(fixture(n=103), **change)


@pytest.mark.parametrize("kind", ["constant", "rank", "subnormal", "zero_counts", "too_few_units", "huge_total"])
def test_rank_precision_and_frequency_unit_admission(kind):
    frame = fixture(n=103)
    if kind == "constant":
        frame.x1 = 1.
    elif kind == "rank":
        frame.x2 = frame.x1
    elif kind == "subnormal":
        frame.x1 *= 1e-156
    elif kind == "zero_counts":
        frame.frequency = 0
    elif kind == "too_few_units":
        frame.frequency = 0
        frame.iloc[1:5, -1] = 1
    else:
        frame.frequency = 100001
    with pytest.raises(AnalysisError):
        fit(frame, replications=39, confidence=.90)


@pytest.mark.parametrize("fail_all", [False, True])
def test_every_failed_draw_is_attempted_collected_without_replacement(monkeypatch, fail_all):
    original, calls = u._parameters, []
    def changed(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1 and (fail_all or len(calls) in (3, 7)):
            raise AnalysisError("unidentified_correlation", "forced draw-only failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(u, "_parameters", changed)
    with pytest.raises(AnalysisError, match="frequency CCA") as caught:
        fit(fixture(n=303), replications=39, confidence=.90)
    assert len(calls) == 40 and caught.value.replications_attempted == 39
    assert len(caught.value.failures) == (39 if fail_all else 2)


def test_zero_frequency_large_values_never_enter_moment_arithmetic():
    frame = fixture(n=303)
    base = fit(frame, replications=39, confidence=.90)
    frame.loc[frame.frequency == 0, NAMES] = 1e145
    result = fit(frame, replications=39, confidence=.90)
    np.testing.assert_array_equal(result["replicate_counts"], base["replicate_counts"])
    np.testing.assert_allclose(result["replicates"], base["replicates"], atol=0)


def test_dimension_and_workspace_admission_precede_numeric_copy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("numeric tensor allocation before admission")
    frame = fixture(n=103)
    monkeypatch.setattr(torch, "tensor", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        fit(frame, replications=39, confidence=.90)
    with pytest.raises(AnalysisError):
        u.canon_fweight_bootstrap(frame, [f"a{i}" for i in range(17)], Y, weights="frequency", components=2)


def test_dataset_iterator_and_autograd_measurements_are_not_silently_coerced():
    with pytest.raises(AnalysisError):
        fit(Dataset.from_frame(fixture(n=103)), replications=39, confidence=.90)
    mapping = {name: torch.tensor(fixture(n=103)[name].to_numpy(), dtype=torch.float64) for name in NAMES}
    mapping["frequency"] = [1] * 103
    mapping["x1"].requires_grad_()
    with pytest.raises(AnalysisError):
        fit(mapping, replications=39, confidence=.90)


def controlled_frame(roots=(.8, .5, .2), n=120):
    rng = np.random.default_rng(921)
    values = rng.normal(size=(n, 6))
    values -= values.mean(0)
    basis = np.linalg.qr(values)[0] * np.sqrt(n - 1)
    x = basis[:, :3]
    y = x * roots + basis[:, 3:] * np.sqrt(1 - np.square(roots))
    frame = pd.DataFrame(np.c_[x, y], columns=NAMES)
    frame["frequency"] = 1
    return frame


@pytest.mark.parametrize("roots", [(.8, .8, .2), (.8, .5, .5), (.8, 0, 0), (1, .5, .2)])
def test_simple_positive_interior_and_retained_boundary_gate(roots):
    with pytest.raises(AnalysisError, match="retained|interior|simple|positive"):
        fit(controlled_frame(roots), replications=39, confidence=.90)


def test_same_normalization_but_signed_axis_exits_fixed_point_chart():
    frame = controlled_frame()
    sample = f.prepare(frame, NAMES, weights="frequency", weight_type="fweight", replications=39,
                       confidence=.90, seed=19, missing="drop", parameters=50)
    point = u._parameters(f.moments(sample.values, sample.counts), X, Y, 2, "coefficients", X[:2])
    values = sample.values.clone()
    angle = np.pi / 3
    rotation = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=torch.float64)
    values[:, :2] = values[:, :2] @ rotation
    with pytest.raises(AnalysisError, match="point-covariance chart"):
        u._parameters(f.moments(values, sample.counts), X, Y, 2, "coefficients", point.anchors, point)


@pytest.mark.parametrize("kind", ["changed_total", "zero_support", "integer_overflow"])
def test_rehashed_invalid_count_structure_refused_beyond_integrity(kind):
    result = fit(fixture(n=303), replications=39, confidence=.90)
    changed = copy.deepcopy(result)
    if kind == "changed_total":
        changed["replicate_counts"].iloc[0, 1] += 1
    elif kind == "zero_support":
        changed["replicate_counts"].iloc[0, 0] = 1
        positive = np.flatnonzero(changed["replicate_counts"].iloc[0].to_numpy() > 1)[0]
        changed["replicate_counts"].iloc[0, positive] -= 1
    else:
        changed["replicate_counts"].iloc[0, 1] = np.iinfo(np.int64).max
    f.seal(changed)
    with pytest.raises(AnalysisError, match="count-structure"):
        f.validate_saved(changed)


def test_literal_unit_rng_does_not_mutate_global_rng_or_expand_data(monkeypatch):
    frame = fixture(n=303)
    torch.manual_seed(918)
    before = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("materialized replicated measurements")
    monkeypatch.setattr(torch, "repeat_interleave", forbidden)
    result = fit(frame, "coefficients", replications=39, confidence=.90)
    assert torch.equal(torch.get_rng_state(), before)
    assert result.attrs["n"] == int(frame.frequency.sum())
    assert result.attrs["n_complete"] == len(frame)


@pytest.mark.parametrize("column", ["x1", "frequency"])
def test_missing_raise_includes_zero_frequency_source_rows(column):
    frame = fixture(n=103)
    frame.loc[0, column] = np.nan
    with pytest.raises(AnalysisError, match="listwise"):
        fit(frame, missing="raise", replications=39, confidence=.90)
