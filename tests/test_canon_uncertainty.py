"""CCA bootstrap checked by independent symmetric generalized-eigen fits.

The oracle estimates raw covariance and solves the covariance pencil with
SciPy; it does not use the production standardized QR/SVD decomposition.
Every requested bootstrap row draw and complete parameter vector is checked.
"""
from __future__ import annotations

import copy
import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import linalg

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import canon_uncertainty as u
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget

X = ["x1", "x2", "x3"]
Y = ["y1", "y2", "y3"]
NAMES = X + Y


def data(domain="normal", *, n=603, seed=178):
    rng = np.random.default_rng(seed)
    if domain == "normal":
        latent = rng.normal(size=(n, 3))
        errors = rng.normal(size=(n, 6))
    else:
        latent = rng.lognormal(sigma=.35, size=(n, 3)) - np.exp(.35**2 / 2)
        latent /= np.sqrt(np.expm1(.35**2) * np.exp(.35**2))
        errors = rng.standard_t(8, size=(n, 6)) / np.sqrt(8 / 6)
    x = latent + .35 * errors[:, :3]
    y = latent * [.9, .5, .18] + errors[:, 3:] * [.35, .7, 1.0]
    values = np.c_[x, y] * [3, 1.7, 2.2, 4, .8, 1.4] + [5, -9, 13, .5, 8, -1]
    return pd.DataFrame(values, columns=NAMES)


def independent_fit(values, components, target, anchors, *, p=3):
    values = np.asarray(values, dtype=np.float64)
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
    for j, anchor in enumerate(anchors or [0] * components):
        sign = 1 if a[anchor, j] > 0 else -1
        a[:, j] *= sign
        b[:, j] *= sign
    matrices = {
        "x_coefficients": a, "y_coefficients": b,
        "x_standardized_coefficients": sd[:p, None] * a,
        "y_standardized_coefficients": sd[p:, None] * b,
        "x_loadings": (xx @ a) / sd[:p, None],
        "y_loadings": (yy @ b) / sd[p:, None],
        "x_cross_loadings": (xy @ b) / sd[:p, None],
        "y_cross_loadings": (xy.T @ a) / sd[p:, None],
    }
    vector = roots[:components] if target == "correlations" else np.concatenate([
        np.r_[roots[j], *(matrix[:, j] for matrix in matrices.values())]
        for j in range(components)])
    return {"vector": vector, "roots": roots, "matrices": matrices,
            "mean": values[0] + offset + correction, "sd": sd,
            "covariance": covariance, "correlation": covariance / np.outer(sd, sd)}


def independent_draws(values, components, target, anchors, replications, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = np.array([torch.randint(len(values), (len(values),), generator=generator,
                                     device="cpu").numpy() for _ in range(replications)])
    fitted = [independent_fit(values[index], components, target, anchors) for index in indices]
    return indices, fitted


@pytest.mark.parametrize("domain", ["normal", "skew_t8"])
@pytest.mark.parametrize("target", ["correlations", "coefficients"])
def test_two_domains_full_199_draw_generalized_eigen_oracle(domain, target):
    frame = data(domain)
    result = u.canon_bootstrap(frame, X, Y, components=2, target=target,
                               anchors=["x1", "x2"] if target == "coefficients" else None,
                               replications=199, seed=19)
    point = independent_fit(frame.to_numpy(), 2, target, [0, 1])
    indices, fitted = independent_draws(frame.to_numpy(), 2, target, [0, 1], 199, 19)
    expected = np.array([fit["vector"] for fit in fitted])
    np.testing.assert_array_equal(result["resample_indices"], indices)
    np.testing.assert_allclose(result["replicates"], expected, atol=8e-12, rtol=8e-12)
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=8e-12)
    covariance = np.cov(expected, rowvar=False, ddof=1)
    if target == "correlations":
        assert covariance.shape == (2, 2)
    np.testing.assert_allclose(result["covariance"], covariance, atol=2e-13, rtol=4e-10)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=8e-12)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(expected, [.025, .975], axis=0, method="linear").T,
                               atol=8e-12, rtol=8e-12)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias,
                               expected.mean(0) - point["vector"], atol=8e-12)
    np.testing.assert_allclose(result["replicate_means"], [fit["mean"] for fit in fitted], atol=8e-12)
    np.testing.assert_allclose(result["replicate_standard_deviations"], [fit["sd"] for fit in fitted], atol=8e-12)
    np.testing.assert_allclose(result["replicate_correlations"], [fit["roots"] for fit in fitted], atol=8e-12)
    np.testing.assert_allclose(result["point_correlations"].correlation, point["roots"], atol=8e-12)
    for name in ["covariance", "correlation"]:
        np.testing.assert_allclose(result["point_" + name], point[name], atol=8e-12)
    if target == "coefficients":
        for name, matrix in point["matrices"].items():
            np.testing.assert_allclose(result["point_" + name], matrix, atol=8e-12)
        assert (result["replicates"][["standardized_coefficient:Can1:X:x1",
                                      "standardized_coefficient:Can2:X:x2"]] > 0).all().all()
        assert (result["replicate_diagnostics"].minimum_x_signed_direction_cosine > np.sqrt(.5)).all()
        assert (result["replicate_diagnostics"].minimum_y_signed_direction_cosine > np.sqrt(.5)).all()
        assert result.attrs["parameter_dimension"] == 50
    else:
        assert result.attrs["parameter_dimension"] == 2 and result.attrs["sign_anchors"] is None
        assert "point_x_coefficients" not in result
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert not result.attrs["p_values_available"] and not result.attrs["inference_df_available"]
    assert result.attrs["failed_replications"] == [] and result.attrs["successful_replications"] == 199
    assert result.attrs["reestimate_moments_every_draw"] and result.attrs["restandardize_every_draw"]
    assert result.attrs["parameter_order"] == result["parameter_order"].values.tolist()
    assert not result.attrs["familywise_intervals"]
    u._validate_saved(restore_summary(summary_state(result)))


def test_canonical_normalization_and_cross_loading_identities_every_draw():
    result = u.canon_bootstrap(data(), X, Y, components=2, target="coefficients", anchors=["x1", "x2"], replications=39, confidence=.9)
    values = result["sample"].to_numpy()
    for index, vector in zip(result["resample_indices"].to_numpy(), result["replicates"].to_numpy(), strict=True):
        covariance = np.cov(values[index], rowvar=False)
        sd = np.sqrt(covariance.diagonal())
        unpacked = vector.reshape(2, 25)
        a, b = unpacked[:, 1:4].T, unpacked[:, 4:7].T
        rho = unpacked[:, 0]
        np.testing.assert_allclose(a.T @ covariance[:3, :3] @ a, np.eye(2), atol=3e-12)
        np.testing.assert_allclose(b.T @ covariance[3:, 3:] @ b, np.eye(2), atol=3e-12)
        np.testing.assert_allclose(a.T @ covariance[:3, 3:] @ b, np.diag(rho), atol=3e-12)
        np.testing.assert_allclose(unpacked[:, 7:10], (a * sd[:3, None]).T, atol=3e-12)
        np.testing.assert_allclose(unpacked[:, 10:13], (b * sd[3:, None]).T, atol=3e-12)
        np.testing.assert_allclose(unpacked[:, 19:22], unpacked[:, 13:16] * rho[:, None], atol=3e-12)
        np.testing.assert_allclose(unpacked[:, 22:25], unpacked[:, 16:19] * rho[:, None], atol=3e-12)


@pytest.mark.parametrize("target", ["correlations", "coefficients"])
def test_units_translation_large_origin_and_independent_oracle(target):
    frame = data(n=403)
    scale = np.array([1e-30, 7e14, 2e-10, 3e5, .07, 1e22])
    shifted = frame * scale
    shifted += [2e-28, -2e16, 3e-8, -1e7, 7, 1e24]
    options = dict(components=2, target=target, replications=39, confidence=.9, seed=26)
    if target == "coefficients":
        options["anchors"] = ["x1", "x2"]
    original = u.canon_bootstrap(frame, X, Y, **options)
    changed = u.canon_bootstrap(shifted, X, Y, **options)
    np.testing.assert_allclose(changed["replicate_correlations"], original["replicate_correlations"], atol=2e-12)
    if target == "coefficients":
        dimensionless = [i for i, row in enumerate(changed.attrs["parameter_order"]) if row[0] != "raw_coefficient"]
        np.testing.assert_allclose(changed["replicates"].iloc[:, dimensionless], original["replicates"].iloc[:, dimensionless], atol=3e-12)
        for name, factors in (("x_coefficients", scale[:3]), ("y_coefficients", scale[3:])):
            np.testing.assert_allclose(changed["point_" + name].to_numpy() * factors[:, None], original["point_" + name], atol=3e-12)
    large = frame + 1e12
    high = u.canon_bootstrap(large, X, Y, **options)
    for index, actual in zip(high["resample_indices"].to_numpy(), high["replicates"].to_numpy(), strict=True):
        expected = independent_fit(large.to_numpy()[index], 2, target, [0, 1])
        np.testing.assert_allclose(actual, expected["vector"], atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("target", ["correlations", "coefficients"])
def test_complete_json_replay_typed_identities_missing_alignment_and_seed(target):
    frame = data(n=310)
    frame.index = pd.MultiIndex.from_tuples([(Decimal(i), "same" if i % 2 else None) for i in range(len(frame))], names=["serial", 4])
    frame.columns.name = Decimal("17")
    frame.iloc[8, 1] = np.nan
    frame.iloc[19, 4] = np.nan
    frame["unused"] = [object() for _ in range(len(frame))]
    before = frame.copy(deep=True)
    options = dict(components=2, target=target, replications=39, confidence=.9, seed=91)
    if target == "coefficients":
        options["anchors"] = ["x1", "x2"]
    result = u.canon_bootstrap(frame, X, Y, **options)
    pd.testing.assert_frame_equal(frame, before)
    assert result.attrs["physical_rows"] == 310 and result.attrs["n"] == 308 and result.attrs["n_missing"] == 2
    assert result.attrs["sample_positions"] == [i for i in range(310) if i not in (8, 19)]
    encoded = summary_state(result)
    assert "NaN" not in encoded and "Infinity" not in encoded
    saved = restore_summary(encoded)
    u._validate_saved(saved)
    assert saved.attrs == result.attrs
    for name in result:
        left = saved[name].astype(object).where(saved[name].notna(), None)
        right = result[name].astype(object).where(result[name].notna(), None)
        pd.testing.assert_frame_equal(left, right)
    assert [encode(decode(code)) for code in saved.attrs["sample_index_codes"]] == saved.attrs["sample_index_codes"]
    assert [decode(code) for code in saved.attrs["source_index_names"]] == ["serial", 4]
    assert [decode(code) for code in saved.attrs["source_column_names"]] == [Decimal("17")]
    assert saved.attrs["source_index_nlevels"] == 2
    replay = u.canon_bootstrap(saved["sample"], X, Y, **options)
    repeat = u.canon_bootstrap(frame, X, Y, **options)
    for name in ["estimates", "covariance", "replicates", "resample_indices", "replicate_means", "replicate_standard_deviations", "replicate_diagnostics"]:
        pd.testing.assert_frame_equal(replay[name], result[name])
        pd.testing.assert_frame_equal(repeat[name], result[name])
    changed = u.canon_bootstrap(frame, X, Y, **(options | {"seed": 92}))
    assert not changed["resample_indices"].equals(result["resample_indices"])
    with pytest.raises(AnalysisError, match="missing"):
        u.canon_bootstrap(frame, X, Y, **options, missing="raise")


@pytest.mark.parametrize("mutation", ["sample", "replicates", "covariance", "indices", "means", "scales", "anchors", "labels", "axis_names", "settings"])
def test_full_saved_integrity_refuses_tampering(mutation):
    result = u.canon_bootstrap(data(n=303), X, Y, components=2, target="coefficients", anchors=["x1", "x2"], replications=39, confidence=.9)
    saved = restore_summary(summary_state(result))
    table_names = {"sample": "sample", "replicates": "replicates", "covariance": "covariance", "indices": "resample_indices", "means": "replicate_means", "scales": "replicate_standard_deviations"}
    if mutation in table_names:
        saved[table_names[mutation]].iloc[0, 0] += 1
    elif mutation == "anchors":
        saved.attrs["sign_anchors"][0] = "x3"
    elif mutation == "labels":
        saved.attrs["sample_index_codes"][0] = encode("changed")
    elif mutation == "axis_names":
        saved["sample"].index.name = "changed"
    else:
        saved["settings"].iloc[0, 1] = "false"
    with pytest.raises(AnalysisError) as error:
        u._validate_saved(saved)
    assert error.value.code == "invalid_state"


def controlled_moments(rhos=(.85, .5, .15), *, n=80):
    rng = np.random.default_rng(11)
    block = rng.normal(size=(n, 6))
    block -= block.mean(0)
    q, _ = np.linalg.qr(block)
    x = q[:, :3] * np.sqrt(n - 1)
    y = (q[:, :3] * rhos + q[:, 3:] * np.sqrt(1 - np.array(rhos)**2)) * np.sqrt(n - 1)
    return torch.as_tensor(np.c_[x, y], dtype=torch.float64)


@pytest.mark.parametrize("rhos,components", [((.8, .8, .1), 1), ((.8, .4, .4), 2), ((.8, .3, 0), 3), ((1, .4, .1), 1), ((1 - 5e-9, .4, .1), 1)])
def test_positive_simple_interior_and_omitted_boundary_refusals(rhos, components):
    with pytest.raises(AnalysisError) as error:
        u._parameters(controlled_moments(rhos), X, Y, components, "correlations", None)
    assert error.value.code == "unidentified_correlation"


def test_default_unique_anchor_fixed_sign_and_wrong_anchor_refusal():
    frame = data()
    result = u.canon_bootstrap(frame, X, Y, components=2, target="coefficients", replications=39, confidence=.9)
    assert result.attrs["sign_anchors"] == ["x1", "x2"]
    assert result.attrs["anchor_selection"].startswith("unique strongest")
    with pytest.raises(AnalysisError) as error:
        u._parameters(controlled_moments(), X, Y, 2, "coefficients", ["x2", "x2"])
    assert error.value.code == "unidentified_anchor"
    values = controlled_moments()
    rotation = torch.tensor([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]], dtype=torch.float64) / np.sqrt(2)
    values[:, :3] = values[:, :3] @ rotation
    with pytest.raises(AnalysisError) as error:
        u._parameters(values, X, Y, 2, "coefficients", None)
    assert error.value.code == "unidentified_anchor"
    # The caller can choose a valid chart even when automatic point pivots tie.
    fitted = u._parameters(values, X, Y, 2, "coefficients", ["x1", "x1"])
    assert fitted.anchors == ["x1", "x1"]


def test_coefficient_chart_crossing_refuses_without_permutation():
    point_values = controlled_moments()
    point = u._parameters(point_values, X, Y, 2, "coefficients", ["x1", "x2"])
    altered = point_values.clone()
    angle = np.pi / 3
    rotation = torch.tensor([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=torch.float64)
    altered[:, :3] = altered[:, :3] @ rotation
    with pytest.raises(AnalysisError) as error:
        u._parameters(altered, X, Y, 2, "coefficients", ["x1", "x2"], point)
    assert error.value.code == "component_crossing"
    roots = u._parameters(altered, X, Y, 2, "correlations", None)
    np.testing.assert_allclose(roots.roots.numpy(), point.roots.numpy(), atol=3e-12)


@pytest.mark.parametrize("side", ["x", "y"])
@pytest.mark.parametrize("noise", [0, 1e-6])
def test_rank_and_condition_refuse_without_deleting_variables(side, noise):
    frame = data()
    names = X if side == "x" else Y
    frame[names[1]] = frame[names[0]] + noise * frame[names[2]]
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(frame, X, Y, components=2, replications=39, confidence=.9)
    assert error.value.code == "singular_matrix"


@pytest.mark.parametrize("kind", ["bool", "complex", "object", "infinite", "huge", "constant", "subnormal"])
def test_source_measurement_admission(kind):
    frame = data()
    if kind == "bool":
        frame["x1"] = frame.x1 > 5
    elif kind == "complex":
        frame["x1"] = frame.x1.astype(complex) + 1j
    elif kind == "object":
        frame["x1"] = frame.x1.astype(str)
    elif kind == "infinite":
        frame.loc[0, "x1"] = np.inf
    elif kind == "huge":
        frame["x1"] *= 1e151
    elif kind == "constant":
        frame["x1"] = 1
    else:
        frame["x1"] *= 1e-155
    with pytest.raises(AnalysisError):
        u.canon_bootstrap(frame, X, Y, components=2, replications=39, confidence=.9)


@pytest.mark.parametrize("changes", [
    {"components": True}, {"components": 0}, {"components": 4},
    {"target": "selected"}, {"anchors": ["x1"]}, {"anchors": ["y1", "x2"]},
    {"replications": True}, {"replications": 18}, {"replications": 2000},
    {"confidence": 0}, {"confidence": 1}, {"confidence": .999},
    {"seed": True}, {"seed": -1}, {"seed": 2**63}, {"missing": "mean"},
])
def test_strict_option_refusals(changes):
    options = dict(components=2, target="coefficients", anchors=["x1", "x2"], replications=39, confidence=.9)
    with pytest.raises(AnalysisError):
        u.canon_bootstrap(data(), X, Y, **(options | changes))


def test_correlations_reject_anchors_and_disjoint_missing_duplicate_specs():
    with pytest.raises(AnalysisError):
        u.canon_bootstrap(data(), X, Y, components=2, anchors=["x1", "x2"])
    for x, y in [(X, X), (["x1", "x1"], Y), (["absent"], Y), ("x1", Y)]:
        with pytest.raises(AnalysisError):
            u.canon_bootstrap(data(), x, y, components=1, replications=39, confidence=.9)
    duplicate = pd.concat([data(), data().x1], axis=1)
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(duplicate, X, Y, components=2, replications=39, confidence=.9)
    assert error.value.code == "duplicate_columns"


def test_resident_mapping_records_and_small_complete_sample():
    frame = data(n=303)
    options = dict(components=2, target="coefficients", anchors=["x1", "x2"], replications=39, confidence=.9)
    fitted = u.canon_bootstrap(frame, X, Y, **options)
    for source in (frame.to_dict("list"), frame.to_dict("records")):
        other = u.canon_bootstrap(source, X, Y, **options)
        pd.testing.assert_frame_equal(other["replicates"], fitted["replicates"])
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(frame.iloc[:19], X, Y, **options)
    assert error.value.code == "insufficient_observations"
    missing = frame.copy()
    missing.loc[19:, "x1"] = np.nan
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(missing, X, Y, **options)
    assert error.value.code == "insufficient_observations"


@pytest.mark.parametrize("form", ["mapping", "records"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1 + 0j])
def test_mixed_boolean_complex_measurements_refused_before_coercion(form, value):
    frame = data(n=303)
    source = frame.to_dict("list" if form == "mapping" else "records")
    if form == "mapping":
        source["x1"][0] = value
    else:
        source[0]["x1"] = value
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(source, X, Y, components=2, replications=39, confidence=.9)
    assert error.value.code == "non_numeric_column"


@pytest.mark.parametrize("representation", ["autograd", "matrix", "sparse"])
def test_plain_tensor_mapping_admission_is_predictable(representation):
    frame = data(n=303)
    source = {name: torch.tensor(frame[name].to_numpy(), dtype=torch.float64) for name in NAMES}
    if representation == "autograd":
        source["x1"].requires_grad_(True)
    elif representation == "matrix":
        source["x1"] = source["x1"][:, None]
    else:
        source["x1"] = source["x1"].to_sparse()
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(source, X, Y, components=2, replications=39, confidence=.9)
    assert error.value.code == "invalid_data"
    source["x1"] = torch.tensor(frame.x1.to_numpy(), dtype=torch.float64)
    actual = u.canon_bootstrap(source, X, Y, components=2, replications=39, confidence=.9)
    expected = u.canon_bootstrap(frame, X, Y, components=2, replications=39, confidence=.9)
    pd.testing.assert_frame_equal(actual["replicates"], expected["replicates"])


def test_all_failed_and_partial_failed_draws_attempted_and_refused(monkeypatch):
    original = u._parameters
    calls = []
    failures = {1, 3, 39}

    def forced(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) - 1 in failures:
            raise AnalysisError("unidentified_correlation", "deliberate failed draw")
        return original(*args, **kwargs)

    monkeypatch.setattr(u, "_parameters", forced)
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(data(), X, Y, components=2, replications=39, confidence=.9)
    assert len(calls) == 40 and error.value.code == "bootstrap_failure"
    assert error.value.replications_attempted == 39 and error.value.successful_replications == 36
    assert [entry["replication"] for entry in error.value.failures] == [1, 3, 39]
    failures.update(range(1, 40))
    calls.clear()
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(data(), X, Y, components=2, replications=39, confidence=.9)
    assert len(calls) == 40 and error.value.successful_replications == 0
    assert len(error.value.failures) == 39


def test_real_draw_subnormal_variance_is_collected_without_replacement(monkeypatch):
    # Point variation is normal; deliberately fixed legitimate draw indices
    # omit the rare full-scale observation and leave only subnormal variation.
    rng = np.random.default_rng(0)
    frame = data(n=100)
    frame["x3"] = rng.normal(size=100) * 1e-157
    frame.loc[99, "x3"] = 1e-151
    draw = torch.arange(100, dtype=torch.int64) % 99
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: draw.clone())
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(frame, X, Y, components=1, replications=39, confidence=.9)
    assert error.value.code == "bootstrap_failure"
    assert len(error.value.failures) == 39
    assert {entry["code"] for entry in error.value.failures} == {"numerical_failure"}


def test_resolved_roots_but_subnormal_raw_coefficient_covariance_refused():
    # Both raw sample variances are normal and finite, but the scale of raw
    # coefficient uncertainty is too small to represent a normal variance.
    first = np.r_[np.ones(5000), -np.ones(5000)]
    second = first.copy()
    second[np.r_[np.arange(500), np.arange(5000, 5500)]] *= -1
    frame = pd.DataFrame({"a": first * 1e150, "b": second * 1e150})
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(frame, ["a"], ["b"], components=1, target="coefficients", anchors=["a"], replications=39, confidence=.9, seed=73)
    assert error.value.code == "numerical_failure" and "bootstrap variance" in str(error.value)
    roots = u.canon_bootstrap(frame, ["a"], ["b"], components=1, replications=39, confidence=.9, seed=73)
    assert np.isfinite(roots["covariance"].to_numpy()).all()


def test_early_dimension_work_export_and_workspace_before_coercion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("input must not be coerced")

    monkeypatch.setattr(u, "_selected", forbidden)
    for frame, options, expected in [
        (data(n=10_001), {}, "workspace_limit"),
        (data(n=1000), {"replications": 1999}, "resource_limit"),
        (pd.DataFrame(np.zeros((9999, 16)), columns=[f"v{i}" for i in range(16)]), {"x": [f"v{i}" for i in range(8)], "y": [f"v{i}" for i in range(8, 16)], "components": 4, "replications": 1999}, "work_limit"),
    ]:
        x, y = options.get("x", X), options.get("y", Y)
        kwargs = {key: value for key, value in options.items() if key not in ("x", "y")}
        with pytest.raises(AnalysisError) as error:
            u.canon_bootstrap(frame, x, y, **({"components": 2, "replications": 39, "confidence": .9} | kwargs))
        assert error.value.code == expected
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(data(), X, Y, components=2, replications=39, confidence=.9)
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("kind", ["text", "encoded_text", "depth", "unsupported", "numeric", "total"])
def test_bounded_lossless_source_identities_before_numeric_tensor(kind, monkeypatch):
    frame = data(n=303)
    labels = list(range(len(frame)))
    if kind == "text":
        labels[0] = "a" * 4097
    elif kind == "encoded_text":
        labels[0] = "ü" * 3000
    elif kind == "depth":
        value = "bottom"
        for _ in range(40):
            value = (value,)
        labels[0] = value
    elif kind == "unsupported":
        labels[0] = object()
    elif kind == "numeric":
        labels[0] = 1 << 20000
    else:
        labels = ["a" * 4000 for _ in range(603)]
        frame = data(n=603)
    frame.index = pd.Index(labels, dtype=object)
    monkeypatch.setattr(u.c, "matrix", lambda *args, **kwargs: pytest.fail("identity refusal must precede numeric tensor"))
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(frame, X, Y, components=2, replications=39, confidence=.9)
    assert error.value.code == ("invalid_index" if kind == "unsupported" else "resource_limit")


def test_long_parameter_labels_admission_and_no_mutation():
    frame = data(n=303)
    renames = {name: name + "z" * 3500 for name in NAMES}
    frame.rename(columns=renames, inplace=True)
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(frame, [renames[name] for name in X], [renames[name] for name in Y], components=2, target="coefficients", replications=39, confidence=.9)
    assert error.value.code == "resource_limit"


def test_json_nan_and_bad_schema_refuse_restoration():
    result = u.canon_bootstrap(data(n=303), X, Y, components=2, replications=39, confidence=.9)
    state = json.loads(summary_state(result))
    bad = copy.deepcopy(state)
    bad["schema"] = "unknown"
    with pytest.raises(AnalysisError):
        restore_summary(json.dumps(bad))
    bad = copy.deepcopy(state)
    bad["tables"]["replicates"]["data"][0][0] = float("nan")
    with pytest.raises(AnalysisError):
        restore_summary(json.dumps(bad))


@pytest.mark.parametrize("p,q,components", [(1, 1, 1), (1, 2, 1), (2, 3, 2), (3, 2, 2)])
def test_unequal_block_width_and_single_correlation_oracles(p, q, components):
    frame = data()
    x, y = X[:p], Y[:q]
    for target in ("correlations", "coefficients"):
        result = u.canon_bootstrap(frame, x, y, components=components, target=target,
                                   anchors=x[:components] if target == "coefficients" else None,
                                   replications=39, confidence=.9, seed=128)
        values = frame[x + y].to_numpy()
        for indices, actual in zip(result["resample_indices"].to_numpy(), result["replicates"].to_numpy(), strict=True):
            oracle = independent_fit(values[indices], components, target, list(range(components)), p=p)
            np.testing.assert_allclose(actual, oracle["vector"], atol=8e-12)
        expected_dimension = components * (1 + 4 * (p + q)) if target == "coefficients" else components
        assert result.attrs["parameter_dimension"] == expected_dimension


def test_stream_summary_iterator_and_non_cpu_columns_rejected_before_consumption():
    calls = []

    def batches():
        calls.append(True)
        yield data()

    stream = Dataset.from_batches(batches, columns=NAMES, row_count=603)
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap(stream, X, Y, components=2)
    assert error.value.code == "unsupported_input" and not calls
    with pytest.raises(AnalysisError) as error:
        u.canon_bootstrap({name: torch.empty(70, device="meta") for name in NAMES}, X, Y, components=2)
    assert error.value.code == "unsupported_device"
    with pytest.raises(AnalysisError):
        u.canon_bootstrap(iter(data().to_dict("records")), X, Y, components=2)


def test_unrelated_default_device_and_dtype_do_not_change_results():
    frame = data(n=303)
    options = dict(components=2, target="coefficients", anchors=["x1", "x2"], replications=39, confidence=.9, seed=71)
    expected = u.canon_bootstrap(frame, X, Y, **options)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            actual = u.canon_bootstrap(frame, X, Y, **options)
    finally:
        torch.set_default_dtype(previous)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])
    assert actual.attrs == expected.attrs
