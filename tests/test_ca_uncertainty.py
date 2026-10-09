"""Independent full-refit CA and sampling-law uncertainty acceptance.

NumPy SVD supplies every point/draw functional without a production fit
helper. The literal declared Torch CPU RNG independently reconstructs draws.
"""
from __future__ import annotations

import json
from decimal import Decimal
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import ca_uncertainty as u
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def domain(which=0):
    values = ([[1300, 150, 120, 80], [100, 1000, 190, 110], [90, 140, 600, 400]]
              if which == 0 else
              [[1100, 160, 100, 70, 110], [150, 850, 150, 130, 100],
               [110, 130, 550, 140, 90], [90, 100, 180, 400, 280]])
    return pd.DataFrame(values, index=[f"r{i}" for i in range(len(values))],
                        columns=[f"c{i}" for i in range(len(values[0]))])


def numpy_fit(counts, dimensions, anchors=None):
    values = np.asarray(counts, dtype=np.float64)
    probability = values/values.sum()
    rows, columns = values.sum(1)/values.sum(), values.sum(0)/values.sum()
    residual = (probability-np.outer(rows, columns))/np.sqrt(np.outer(rows, columns))
    left, roots, right_t = np.linalg.svd(residual, full_matrices=False)
    left, right = left[:, :dimensions].copy(), right_t.T[:, :dimensions].copy()
    if anchors is None:
        anchors = np.argmax(np.abs(left), axis=0)
    for j, i in enumerate(anchors):
        sign = 1 if left[i, j] > 0 else -1
        left[:, j] *= sign
        right[:, j] *= sign
    row_standard, column_standard = left/np.sqrt(rows[:, None]), right/np.sqrt(columns[:, None])
    row_principal, column_principal = row_standard*roots[:dimensions], column_standard*roots[:dimensions]
    # Direct chi-square/N independently checks total inertia normalization.
    expected = np.outer(values.sum(1), values.sum(0))/values.sum()
    inertia = np.sum((values-expected)**2/expected)/values.sum()
    vector = np.r_[inertia, rows, columns,
                   *[np.r_[roots[j], roots[j]**2, row_standard[:, j], row_principal[:, j],
                            column_standard[:, j], column_principal[:, j]]
                     for j in range(dimensions)]]
    return dict(vector=vector, roots=roots, anchors=list(anchors), probability=probability,
                residual=residual, row_mass=rows, column_mass=columns, left=left, right=right,
                row_standard=row_standard, column_standard=column_standard,
                row_principal=row_principal, column_principal=column_principal)


def literal_draws(values, sampling, *, seed, replications):
    values = np.asarray(values, dtype=np.int64)
    r, c = values.shape
    total, row_totals = int(values.sum()), values.sum(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = []
    for _ in range(replications):
        if sampling == "multinomial":
            probabilities = torch.tensor(values.flatten()/total, dtype=torch.float64)
            indices = torch.multinomial(probabilities, total, replacement=True, generator=generator)
            draw = torch.bincount(indices, minlength=r*c).numpy().reshape(r, c)
        else:
            draw = []
            for i in range(r):
                probabilities = torch.tensor(values[i]/row_totals[i], dtype=torch.float64)
                indices = torch.multinomial(probabilities, int(row_totals[i]),
                                            replacement=True, generator=generator)
                draw.append(torch.bincount(indices, minlength=c).numpy())
            draw = np.array(draw)
        draws.append(draw)
    return np.array(draws)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("sampling", ["multinomial", "row_multinomial"])
def test_two_domains_every_draw_joint_covariance_percentiles_and_normalization(which, sampling):
    frame = domain(which)
    result = u.ca_bootstrap(frame, dimensions=2, sampling=sampling, replications=199, seed=317+which)
    point = numpy_fit(frame, 2)
    draws = literal_draws(frame, sampling, seed=317+which, replications=199)
    fits = [numpy_fit(draw, 2, point["anchors"]) for draw in draws]
    vectors = np.array([fit["vector"] for fit in fits])
    np.testing.assert_array_equal(result["resample_counts"], draws.reshape(199, -1))
    np.testing.assert_allclose(result["replicates"], vectors, rtol=8e-13, atol=8e-13)
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=8e-13)
    covariance = np.cov(vectors, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], covariance, atol=5e-15, rtol=8e-11)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=8e-13)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(vectors, [.025, .975], axis=0, method="linear").T, atol=8e-13)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias,
                               vectors.mean(0)-point["vector"], atol=8e-13)
    for table, name in [("point_probabilities", "probability"), ("point_residuals", "residual")]:
        np.testing.assert_allclose(result[table], point[name], atol=8e-13)
    np.testing.assert_allclose(result["point_spectrum"].rho, point["roots"], atol=8e-13)
    np.testing.assert_allclose(result["replicate_spectrum"], [fit["roots"] for fit in fits], atol=8e-13)
    for axis, name in [("row", "rows"), ("column", "columns")]:
        np.testing.assert_allclose(result[name].mass, point[f"{axis}_mass"], atol=8e-13)
        standard = result[name][["Dim1_standard", "Dim2_standard"]].to_numpy()
        principal = result[name][["Dim1_principal", "Dim2_principal"]].to_numpy()
        np.testing.assert_allclose(standard, point[f"{axis}_standard"], atol=8e-13)
        np.testing.assert_allclose(principal, point[f"{axis}_principal"], atol=8e-13)
        np.testing.assert_allclose(standard.T@np.diag(point[f"{axis}_mass"])@standard, np.eye(2), atol=8e-13)
        np.testing.assert_allclose(point[f"{axis}_mass"]@standard, np.zeros(2), atol=8e-13)
    approximation = (point["left"]*point["roots"][:2])@point["right"].T
    # A 3-row table has exactly two nonzero axes; the 4-row domain retains two.
    if which == 0:
        np.testing.assert_allclose(approximation, point["residual"], atol=8e-13)
    assert (result["replicate_diagnostics"].minimum_signed_direction_cosine > np.sqrt(.5)).all()
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert result.attrs["parameter_order"] == result["parameter_order"].values.tolist()
    assert result.attrs["successful_replications"] == 199 and result.attrs["failed_replications"] == []
    assert result.attrs["parameter_dimension"] == len(point["vector"])
    assert result.attrs["sign_anchor_positions"] == point["anchors"]
    assert result.attrs["p_values_available"] is False and result.attrs["familywise_intervals"] is False
    assert result.attrs["reestimate_masses_and_axes_every_draw"] is True


@pytest.mark.parametrize("sampling", ["multinomial", "row_multinomial"])
def test_count_sampling_exact_constraints_and_independent_probability_law(sampling):
    frame = domain()
    result = u.ca_bootstrap(frame, dimensions=2, sampling=sampling, replications=1999, seed=719)
    counts = result["resample_counts"].to_numpy().reshape(1999, *frame.shape)
    total, row_totals = frame.to_numpy().sum(), frame.to_numpy().sum(1)
    np.testing.assert_array_equal(counts.sum((1, 2)), np.full(1999, total))
    if sampling == "row_multinomial":
        np.testing.assert_array_equal(counts.sum(2), np.tile(row_totals, (1999, 1)))
        np.testing.assert_allclose(result["sampling_probabilities"].sum(1), 1, atol=2e-16)
        row_mass_columns = [f"row_mass:{i+1}" for i in range(len(frame))]
        assert np.count_nonzero(result["covariance"].loc[row_mass_columns].to_numpy()) == 0
        np.testing.assert_array_equal(result["estimates"].loc[row_mass_columns, "std_error"], 0)
        covariance = np.zeros((frame.size, frame.size))
        for i, total_i in enumerate(row_totals):
            p = frame.iloc[i].to_numpy()/total_i
            covariance[i*frame.shape[1]:(i+1)*frame.shape[1], i*frame.shape[1]:(i+1)*frame.shape[1]] = total_i*(np.diag(p)-np.outer(p, p))
    else:
        assert (counts.sum(2).std(0) > 10).all()
        p = frame.to_numpy().flatten()/total
        np.testing.assert_allclose(result["sampling_probabilities"].to_numpy().sum(), 1, atol=2e-16)
        covariance = total*(np.diag(p)-np.outer(p, p))
    samples = counts.reshape(len(counts), -1)
    # These are sampling-law diagnostics, with broad Monte Carlo bounds, not
    # a claimed confidence-interval coverage experiment.
    expected = frame.to_numpy().flatten()
    se_mean = np.sqrt(covariance.diagonal()/len(samples))
    assert np.max(np.abs(samples.mean(0)-expected)/se_mean) < 4.5
    observed = np.cov(samples, rowvar=False, ddof=1)
    standard = np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))
    assert np.max(np.abs(observed-covariance)/standard) < .12


def test_sampling_models_have_same_point_and_distinct_full_uncertainty():
    a = u.ca_bootstrap(domain(), dimensions=2, seed=91, replications=39)
    b = u.ca_bootstrap(domain(), dimensions=2, sampling="row_multinomial", seed=91, replications=39)
    np.testing.assert_array_equal(a["estimates"].estimate, b["estimates"].estimate)
    assert not a["resample_counts"].equals(b["resample_counts"])
    assert not a["covariance"].equals(b["covariance"])
    assert a.attrs["inference_target"] != b.attrs["inference_target"]


@pytest.mark.parametrize("sampling", ["multinomial", "row_multinomial"])
def test_full_json_restore_typed_source_axes_replay_and_no_input_or_global_rng_mutation(sampling):
    frame = domain()
    frame.index = pd.MultiIndex.from_tuples([(Decimal("1.25"), "x"), (7, None),
                                           (UUID(int=17), "z")], names=[Decimal("4"), 19])
    frame.columns = pd.Index([1, "1", pd.Timestamp("2024-01-01", tz="UTC"), UUID(int=12)], dtype=object)
    frame.columns.name = ("axis", Decimal("7.5"))
    before = frame.copy(deep=True)
    torch.manual_seed(123)
    global_rng = torch.random.get_rng_state().clone()
    result = u.ca_bootstrap(frame, dimensions=2, sampling=sampling, seed=191, replications=39)
    pd.testing.assert_frame_equal(frame, before)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    state = summary_state(result)
    assert "NaN" not in state and "Infinity" not in state
    payload = json.loads(state)
    assert len(payload["tables"]["resample_counts"]["data"]) == 39
    saved = restore_summary(state)
    assert saved.attrs == result.attrs
    for name in result:
        left = saved[name].astype(object).where(saved[name].notna(), None)
        right = result[name].astype(object).where(result[name].notna(), None)
        pd.testing.assert_frame_equal(left, right)
    row_labels = [decode(code) for code in saved.attrs["row_label_codes"]]
    col_labels = [decode(code) for code in saved.attrs["column_label_codes"]]
    row_names = [decode(code) for code in saved.attrs["source_row_axis_names"]]
    column_names = [decode(code) for code in saved.attrs["source_column_axis_names"]]
    rebuilt = saved["point_counts"].copy()
    rebuilt.index = pd.MultiIndex.from_tuples(row_labels, names=row_names)
    rebuilt.columns = pd.Index(col_labels, dtype=object, name=column_names[0])
    assert [encode(value) for value in rebuilt.index] == result.attrs["row_label_codes"]
    assert [encode(value) for value in rebuilt.columns] == result.attrs["column_label_codes"]
    anchors = [decode(code) for code in saved.attrs["sign_anchor_codes"]]
    replay = u.ca_bootstrap(rebuilt, dimensions=2, sampling=sampling, anchors=anchors,
                           seed=191, replications=39)
    for name in ["replicates", "resample_counts", "covariance", "estimates", "replicate_spectrum"]:
        pd.testing.assert_frame_equal(replay[name], result[name])
    changed = u.ca_bootstrap(frame, dimensions=2, sampling=sampling, seed=192, replications=39)
    assert not changed["resample_counts"].equals(result["resample_counts"])
    assert result.attrs["source_content_sha256"] == replay.attrs["source_content_sha256"]
    assert result.attrs["source_row_nlevels"] == 2 and result.attrs["source_column_nlevels"] == 1


def test_caller_anchors_are_fixed_and_exact_typed_labels():
    frame = domain()
    frame.index = pd.Index([1, "1", 3], dtype=object)
    point = numpy_fit(frame, 2)
    anchors = [frame.index[i] for i in point["anchors"]]
    result = u.ca_bootstrap(frame, dimensions=2, anchors=anchors, replications=39, seed=317)
    assert result.attrs["anchor_selection"] == "caller declared"
    assert result.attrs["sign_anchor_codes"] == [encode(value) for value in anchors]
    for j, i in enumerate(point["anchors"]):
        assert (result["replicates"][f"row_standard:Dim{j+1}:{i+1}"] > 0).all()
    with pytest.raises(AnalysisError, match="exact declared"):
        u.ca_bootstrap(frame, dimensions=2, anchors=[1.0, "1"], replications=39)


@pytest.mark.parametrize("sampling", ["multinomial", "row_multinomial"])
def test_near_axis_crossing_refuses_every_requested_refit_without_partial_output(sampling):
    counts = pd.DataFrame([[20, 2, 2], [2, 19, 2], [2, 2, 18]])
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(counts, dimensions=2, sampling=sampling, replications=39, seed=11)
    error = caught.value
    assert error.code == "bootstrap_failure" and error.replications_attempted == 39
    assert any(item["code"] == "axis_crossing" for item in error.failures)
    assert error.successful_replications+len(error.failures) == 39
    assert [item["replication"] for item in error.failures] == sorted({item["replication"] for item in error.failures})
    assert max(item["replication"] for item in error.failures) > 30


def test_zero_margin_draw_refuses_instead_of_removing_support():
    counts = pd.DataFrame([[18, 0], [0, 2]])
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(counts, dimensions=1, replications=39, seed=11)
    assert caught.value.code == "bootstrap_failure"
    assert caught.value.replications_attempted == 39
    assert caught.value.failures == [{"replication": 12, "code": "zero_margin",
                                    "message": "Every declared category must have positive mass in every fit; no category is removed."}]
    # Conditional deterministic rows preserve their support, giving a truly
    # deterministic target and zero covariance without an artificial ridge.
    result = u.ca_bootstrap(counts, dimensions=1, sampling="row_multinomial", replications=39, seed=11)
    assert np.count_nonzero(result["covariance"]) == 0


@pytest.mark.parametrize("counts,dimensions,code", [
    ([[12, 8], [0, 0]], 1, "zero_margin"),
    ([[12, 0], [8, 0]], 1, "zero_margin"),
    ([[10, 10], [20, 20]], 1, "unidentified_axis"),
    ([[30, 3, 3], [3, 30, 3], [3, 3, 30]], 2, "unidentified_axis"),
    ([[30, 3, 3], [3, 30, 3], [3, 3, 30]], 1, "unidentified_axis"),
    ([[10, 10], [10, 10]], 2, "invalid_spec"),
])
def test_point_rank_positive_mass_and_boundary_gap_guards(counts, dimensions, code):
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(pd.DataFrame(counts), dimensions=dimensions, replications=39)
    assert caught.value.code == code


@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.5, -1, np.nan, np.inf, -np.inf,
                                 3+0j, "3", None, pd.NA])
def test_no_count_coercion_fractional_missing_complex_or_negative(bad):
    frame = domain().astype(object)
    frame.iat[0, 0] = bad
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(frame, dimensions=2, replications=39)
    assert caught.value.code == "invalid_counts"


@pytest.mark.parametrize("options", [
    {"dimensions": True}, {"dimensions": 0}, {"dimensions": 5}, {"dimensions": 1.5},
    {"replications": 18}, {"replications": 2000}, {"replications": True},
    {"confidence": .99, "replications": 39}, {"confidence": 1}, {"confidence": 0},
    {"confidence": np.nan}, {"seed": True}, {"seed": -1}, {"seed": 2**63},
    {"sampling": "poisson"}, {"anchors": ["r0"]}, {"anchors": ["r0", "absent"]},
])
def test_invalid_fixed_specifications_and_percentile_tail_admission(options):
    settings = dict(dimensions=2, replications=39)
    settings.update(options)
    with pytest.raises(AnalysisError):
        u.ca_bootstrap(domain(), **settings)


@pytest.mark.parametrize("counts", [np.ones((3, 3)), [[1, 2], [3, 4]], {"x": [1, 2]}])
def test_only_labelled_resident_count_dataframe(counts):
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(counts, dimensions=1, replications=39)
    assert caught.value.code == "unsupported_input"


@pytest.mark.parametrize("axis", ["row", "column", "name"])
def test_unportable_source_identities_refuse(axis):
    frame = domain()
    if axis == "row":
        frame.index = [object(), "a", "b"]
    elif axis == "column":
        frame.columns = [object(), "a", "b", "c"]
    else:
        frame.index.name = object()
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(frame, dimensions=2, replications=39)
    assert caught.value.code == "invalid_label"


def test_duplicate_labels_refuse_and_oversized_identity_is_bounded():
    frame = domain()
    frame.index = ["same", "same", "third"]
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(frame, dimensions=2, replications=39)
    assert caught.value.code == "duplicate_labels"
    frame = domain()
    frame.columns = ["ü"*3000, "a", "b", "c"]
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(frame, dimensions=2, replications=39)
    assert caught.value.code == "resource_limit"
    nested = "leaf"
    for _ in range(18):
        nested = (nested,)
    frame = domain()
    frame.index.name = nested
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(frame, dimensions=2, replications=39)
    assert caught.value.code == "resource_limit"


def test_decimal_storage_is_admitted_before_shared_identity_encoding(monkeypatch):
    frame = domain()
    huge = Decimal("9"*100000)
    frame.index = pd.Index([huge, "a", "b"], dtype=object)
    original = u.encode

    def guarded(value):
        assert value is not huge, "Oversized Decimal reached the shared encoder."
        return original(value)

    monkeypatch.setattr(u, "encode", guarded)
    with pytest.raises(AnalysisError) as caught:
        u.ca_bootstrap(frame, dimensions=2, replications=39)
    assert caught.value.code == "resource_limit"


@pytest.mark.parametrize("kind", ["count", "shape", "parameters", "work", "workspace"])
def test_resource_gates_precede_any_model_tensor_sampling_or_svd(monkeypatch, kind):
    def forbidden(*args, **kwargs):
        raise AssertionError("A bounded-resource rejection allocated a model tensor.")
    frame, options = domain(), dict(dimensions=2, replications=39)
    if kind == "count":
        frame.iat[0, 0] = 100001
    elif kind == "shape":
        frame = pd.DataFrame(np.ones((17, 3), dtype=int)*20)
    elif kind == "parameters":
        frame = pd.DataFrame(np.eye(16, dtype=int)*100+10)
        options["dimensions"] = 4
    elif kind == "work":
        frame *= 20
        options["replications"] = 1999
    else:
        options["replications"] = 1999
    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch, "multinomial", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    with use_workspace_budget(1 if kind == "workspace" else 512):
        with pytest.raises(AnalysisError) as caught:
            u.ca_bootstrap(frame, **options)
    assert caught.value.code in {"workspace_limit", "resource_limit", "work_limit"}


def test_resource_plan_and_export_estimate_cover_actual_full_state():
    result = u.ca_bootstrap(domain(1), dimensions=2, replications=199, seed=701)
    encoded_bytes = len(summary_state(result).encode())
    assert encoded_bytes < result.attrs["estimated_complete_export_bytes"] < 32*1024**2
    buffers = result.attrs["resource_plan"]["buffers"]
    assert buffers["all_sampled_count_tables"] >= 199*domain(1).size*8
    assert buffers["all_joint_vectors_and_diagnostics"] >= 199*result.attrs["parameter_dimension"]*8
    assert "not a process-RSS" in result.attrs["resource_plan"]["scope"]


def test_default_device_and_dtype_cannot_change_native_cpu_float64():
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            result = u.ca_bootstrap(domain(), dimensions=2, replications=39, seed=317)
    finally:
        torch.set_default_dtype(torch.float64)
    assert result.attrs["device"] == "cpu" and result.attrs["precision"] == "float64"
    assert result["replicates"].to_numpy().dtype == np.float64
    point = numpy_fit(domain(), 2)
    np.testing.assert_allclose(result["estimates"].estimate, point["vector"], atol=8e-13)


def test_exact_integer_float_counts_match_integer_source_and_resolved_point_scale_invariance():
    frame = domain()
    a = u.ca_bootstrap(frame, dimensions=2, replications=39, seed=317)
    b = u.ca_bootstrap(frame.astype(float), dimensions=2, replications=39, seed=317)
    pd.testing.assert_frame_equal(a["replicates"], b["replicates"])
    scaled = u.ca_bootstrap(frame*4, dimensions=2, replications=39, seed=317)
    np.testing.assert_allclose(a["estimates"].estimate, scaled["estimates"].estimate, atol=8e-13)
    assert scaled["estimates"].loc["rho:Dim1", "std_error"] < a["estimates"].loc["rho:Dim1", "std_error"]
