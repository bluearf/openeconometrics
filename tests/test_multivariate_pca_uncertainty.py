"""Independent PCA IID bootstrap oracles and numerical/persistence boundaries.

The fitting oracle uses NumPy only, with literal row draws shared through the
declared CPU Torch RNG. It does not call a production moment/PCA helper.
"""
from __future__ import annotations

import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import pca_uncertainty as u
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


NAMES = ["a", "b", "c", "d", "e", "f"]


def data(distribution="normal", *, n=603, seed=891):
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        latent = rng.normal(size=(n, 2))
        errors = rng.normal(size=(n, 6))
    else:
        latent = (rng.lognormal(sigma=.45, size=(n, 2))-np.exp(.45**2/2))
        latent /= np.sqrt(np.expm1(.45**2)*np.exp(.45**2))
        errors = rng.standard_t(7, size=(n, 6))/np.sqrt(7/5)
    latent[:, 1] = .15*latent[:, 0]+np.sqrt(1-.15**2)*latent[:, 1]
    loading = np.array([.95, .93, .9, .8, .78, .76])
    measured = latent[:, [0, 0, 0, 1, 1, 1]]*loading
    measured += errors*np.sqrt(1-loading**2)
    measured = measured*[3, 2.5, 2, 1.5, 1.3, 1.1]+[5, -2, 12, .5, 9, -3]
    return pd.DataFrame(measured, columns=NAMES)


def independent_fit(values, *, matrix, components, anchors):
    values = np.asarray(values, dtype=np.float64)
    origin = values[0]
    shift = values-origin
    offset = shift.mean(axis=0)
    centered = shift-offset
    correction = centered.mean(axis=0)
    centered -= correction
    covariance = centered.T@centered/(len(values)-1)
    standard_deviations = np.sqrt(covariance.diagonal())
    target = covariance.copy()
    if matrix == "correlation":
        target /= np.outer(standard_deviations, standard_deviations)
        np.fill_diagonal(target, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(target)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    vectors = eigenvectors[:, :components].copy()
    for j, anchor in enumerate(anchors):
        vectors[:, j] *= 1 if vectors[anchor, j] > 0 else -1
    loadings = vectors*np.sqrt(eigenvalues[:components])
    parameters = np.concatenate([np.r_[eigenvalues[j], vectors[:, j], loadings[:, j]]
                                 for j in range(components)])
    return {"parameters": parameters, "mean": origin+offset+correction,
            "sd": standard_deviations, "covariance": covariance,
            "target": target, "roots": eigenvalues, "vectors": vectors,
            "loadings": loadings}


def independent_draws(values, *, matrix, components, anchors, seed, replications):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = np.array([torch.randint(len(values), (len(values),), generator=generator,
                                     device="cpu").numpy()
                        for _ in range(replications)])
    fitted = [independent_fit(values[index], matrix=matrix, components=components,
                             anchors=anchors) for index in indices]
    return indices, fitted


@pytest.mark.parametrize("distribution", ["normal", "skewed_heavy_tail"])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_two_domains_every_draw_full_joint_covariance_and_percentiles(distribution, matrix):
    frame = data(distribution)
    result = u.pca_bootstrap(frame, NAMES, components=2, matrix=matrix,
                             anchors=["a", "d"], replications=199, seed=913)
    point = independent_fit(frame.to_numpy(), matrix=matrix, components=2, anchors=[0, 3])
    indices, fitted = independent_draws(frame.to_numpy(), matrix=matrix, components=2,
                                       anchors=[0, 3], replications=199, seed=913)
    expected = np.array([fit["parameters"] for fit in fitted])
    np.testing.assert_array_equal(result["resample_indices"], indices)
    np.testing.assert_allclose(result["replicates"], expected, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(result["estimates"].estimate, point["parameters"], atol=3e-12)
    covariance = np.cov(expected, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], covariance, atol=2e-13, rtol=3e-11)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=3e-12)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(expected, [.025, .975], axis=0, method="linear").T, atol=3e-12)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias,
                               expected.mean(0)-point["parameters"], atol=3e-12)
    for name, oracle_key in [("point_covariance", "covariance"), ("point_matrix", "target"),
                              ("point_eigenvectors", "vectors"), ("point_loadings", "loadings")]:
        np.testing.assert_allclose(result[name], point[oracle_key], atol=3e-12)
    np.testing.assert_allclose(result["point_eigenvalues"].eigenvalue, point["roots"], atol=3e-12)
    np.testing.assert_allclose(result["replicate_means"], [fit["mean"] for fit in fitted], atol=3e-12)
    np.testing.assert_allclose(result["replicate_standard_deviations"],
                               [fit["sd"] for fit in fitted], atol=3e-12)
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert not result.attrs["p_values_available"] and not result.attrs["inference_df_available"]
    assert result.attrs["successful_replications"] == 199 and result.attrs["failed_replications"] == []
    assert result.attrs["restandardize_every_draw"] == (matrix == "correlation")
    assert result.attrs["parameter_dimension"] == 26
    assert result.attrs["parameter_order"] == result["parameter_order"].values.tolist()
    assert (result["replicates"][["eigenvector:Comp1:a", "eigenvector:Comp2:d"]] > 0).all().all()
    assert (result["replicate_diagnostics"].minimum_signed_direction_cosine > np.sqrt(.5)).all()


def test_correlation_restandardizes_instead_of_using_point_scale():
    frame = data("skewed_heavy_tail")
    result = u.pca_bootstrap(frame, NAMES, components=2, replications=59, confidence=.9, seed=815)
    fixed_scale = result["descriptives"].std_dev.to_numpy()
    values = frame.to_numpy()
    incorrect = []
    for index in result["resample_indices"].to_numpy():
        standardized = values[index]/fixed_scale
        covariance = np.cov(standardized, rowvar=False, ddof=1)
        incorrect.append(np.linalg.eigvalsh(covariance)[-1])
    assert np.max(np.abs(result["replicates"]["eigenvalue:Comp1"]-incorrect)) > .05
    assert np.std(result["replicate_standard_deviations"].to_numpy(), axis=0).min() > .01
    assert result.attrs["reestimate_moments_every_draw"]


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_full_finite_saved_state_replay_and_seed(matrix):
    frame = data(n=310)
    frame.index = pd.MultiIndex.from_tuples([(Decimal(i), "same" if i % 2 else None)
                                            for i in range(len(frame))], names=["serial", 4])
    frame.columns.name = Decimal("17")
    result = u.pca_bootstrap(frame, NAMES, components=2, matrix=matrix,
                             replications=59, confidence=.9, seed=19)
    encoded = summary_state(result)
    assert "NaN" not in encoded and "Infinity" not in encoded
    saved = restore_summary(encoded)
    assert saved.attrs == result.attrs
    for name in result:
        # Finite JSON represents the undefined eigenvalue variable label as
        # null; Pandas may materialize the same missing cell as None or NaN.
        left = saved[name].astype(object).where(saved[name].notna(), None)
        right = result[name].astype(object).where(result[name].notna(), None)
        pd.testing.assert_frame_equal(left, right)
    original = [decode(code) for code in saved.attrs["sample_index_codes"]]
    assert [encode(value) for value in original] == saved.attrs["sample_index_codes"]
    assert [decode(code) for code in saved.attrs["source_index_names"]] == ["serial", 4]
    assert saved.attrs["source_index_nlevels"] == 2
    assert [decode(code) for code in saved.attrs["source_column_names"]] == [Decimal("17")]
    assert saved.attrs["source_column_nlevels"] == 1
    replay = u.pca_bootstrap(saved["sample"], saved.attrs["variables"],
                             components=saved.attrs["components"], matrix=saved.attrs["matrix"],
                             anchors=saved.attrs["sign_anchors"], replications=59,
                             confidence=.9, seed=saved.attrs["seed"])
    repeat = u.pca_bootstrap(frame, NAMES, components=2, matrix=matrix,
                             replications=59, confidence=.9, seed=19)
    changed = u.pca_bootstrap(frame, NAMES, components=2, matrix=matrix,
                              replications=59, confidence=.9, seed=20)
    for name in ["replicates", "covariance", "estimates", "resample_indices", "replicate_diagnostics"]:
        pd.testing.assert_frame_equal(replay[name], result[name])
        pd.testing.assert_frame_equal(repeat[name], result[name])
    assert not changed["resample_indices"].equals(result["resample_indices"])
    assert not changed["replicates"].equals(result["replicates"])
    assert len(saved["replicates"]) == 59 and len(saved["resample_indices"].columns) == len(frame)


def test_missing_rows_duplicate_typed_labels_and_input_preservation():
    frame = data(n=310)
    frame.index = pd.Index([None, np.nan, pd.NA, True, 1, Decimal("1"), *range(304)], dtype=object)
    frame.iloc[8, 1] = np.nan
    frame.iloc[19, 4] = np.nan
    frame["ignored"] = [object() for _ in range(len(frame))]
    before = frame.copy(deep=True)
    result = u.pca_bootstrap(frame, NAMES, components=2, replications=39, seed=212)
    keep = frame[NAMES].notna().all(axis=1)
    clean = frame.loc[keep, NAMES]
    reference = u.pca_bootstrap(clean, NAMES, components=2, anchors=result.attrs["sign_anchors"],
                                replications=39, seed=212)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result["replicates"], reference["replicates"])
    assert result.attrs["n_missing"] == 2 and result.attrs["n"] == 308
    assert result.attrs["sample_positions"] == np.flatnonzero(keep).tolist()
    assert result.attrs["sample_index_codes"] == [encode(value) for value in frame.index[keep]]
    assert result.attrs["sample_index_codes"][3] != result.attrs["sample_index_codes"][4]
    assert result.attrs["sample_index_codes"][4] != result.attrs["sample_index_codes"][5]
    assert result.attrs["sample_positions"] == list(result["sample"].index)
    json.dumps(result.attrs, allow_nan=False)
    with pytest.raises(AnalysisError, match="missing"):
        u.pca_bootstrap(frame, NAMES, components=2, missing="raise")


def test_mapping_records_and_unused_unsized_column():
    frame = data(n=310)
    direct = u.pca_bootstrap(frame, NAMES, components=2, replications=39, seed=715)
    mapping = frame.to_dict(orient="list")
    mapping["ignored"] = object()
    mapped = u.pca_bootstrap(mapping, NAMES, components=2, replications=39, seed=715)
    records = u.pca_bootstrap(frame.to_dict(orient="records"), NAMES, components=2, replications=39, seed=715)
    pd.testing.assert_frame_equal(mapped["replicates"], direct["replicates"])
    pd.testing.assert_frame_equal(records["replicates"], direct["replicates"])


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_represented_large_origin_agrees_with_independent_draws(matrix):
    frame = data(n=603)+1e12
    result = u.pca_bootstrap(frame, NAMES, components=2, matrix=matrix, anchors=["a", "d"],
                             replications=39, seed=815)
    _, fitted = independent_draws(frame.to_numpy(), matrix=matrix, components=2,
                                  anchors=[0, 3], replications=39, seed=815)
    np.testing.assert_allclose(result["replicates"], [fit["parameters"] for fit in fitted], atol=3e-12)
    np.testing.assert_allclose(result["replicate_means"], [fit["mean"] for fit in fitted], atol=.00025)


def test_correlation_units_invariance_and_covariance_units_effect():
    frame = data(n=603)
    changed = frame*[.01, 100, .2, 7, 11, 3]+[600, -900, 1, 2, 3, 4]
    first = u.pca_bootstrap(frame, NAMES, components=2, replications=39, seed=713)
    second = u.pca_bootstrap(changed, NAMES, components=2,
                             anchors=first.attrs["sign_anchors"], replications=39, seed=713)
    np.testing.assert_allclose(first["replicates"], second["replicates"], atol=8e-12)
    first_cov = u.pca_bootstrap(frame, NAMES, components=1, matrix="covariance", replications=39, seed=713)
    second_cov = u.pca_bootstrap(changed, NAMES, components=1, matrix="covariance", replications=39, seed=713)
    assert np.max(np.abs(first_cov["replicates"].iloc[:, 0]-second_cov["replicates"].iloc[:, 0])) > 1e4


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_unrepresentable_joint_covariance_refuses_instead_of_zero_or_infinite_se(scale):
    frame = data(n=603)*scale
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, NAMES, components=2, matrix="covariance",
                        replications=39, seed=713)
    assert caught.value.code == "numerical_failure"
    # Correlation's target is invariant to the same representable unit changes.
    actual = u.pca_bootstrap(frame, NAMES, components=2, replications=39, seed=713)
    expected = u.pca_bootstrap(data(n=603), NAMES, components=2,
                              anchors=actual.attrs["sign_anchors"], replications=39, seed=713)
    np.testing.assert_allclose(actual["replicates"], expected["replicates"], atol=8e-12)


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
@pytest.mark.parametrize("scale", [1e-158, 1e-160, 1e-161])
def test_subnormal_raw_moments_refuse_before_quantized_restandardization(matrix, scale):
    with pytest.raises(AnalysisError, match="raw sample variance is subnormal") as caught:
        u.pca_bootstrap(data(n=603)*scale, NAMES, components=2, matrix=matrix,
                        replications=39, seed=713)
    assert caught.value.code == "numerical_failure"


def test_replicated_subnormal_variances_trigger_exhaustive_failure():
    rng = np.random.default_rng(713)
    latent = rng.normal(size=603)
    values = latent[:, None]+rng.normal(size=(603, 2))*[.3, .4]
    # Both point variances are normal; genuine row draws cross the floor.
    values *= np.sqrt(np.finfo(np.float64).tiny*1.02/values.var(axis=0, ddof=1))
    assert (values.var(axis=0, ddof=1) > np.finfo(np.float64).tiny).all()
    frame = pd.DataFrame(values, columns=["a", "b"])
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, ["a", "b"], components=1, replications=39, seed=713)
    error = caught.value
    assert error.code == "bootstrap_failure" and error.replications_attempted == 39
    assert 0 < len(error.failures) < 39
    assert all(failure["code"] == "numerical_failure" and "subnormal" in failure["message"]
               for failure in error.failures)


def test_full_component_count_and_fixed_declared_negative_anchor():
    frame = data(n=603)[["a", "b"]].rename(columns={"b": "d"})
    frame["d"] = -frame.d
    result = u.pca_bootstrap(frame, ["a", "d"], components=2, matrix="covariance",
                             anchors=["d", "a"], replications=39, seed=77)
    assert result.attrs["parameter_dimension"] == 10
    assert result.attrs["sign_anchors"] == ["d", "a"]
    assert (result["replicates"]["eigenvector:Comp1:d"] > 0).all()
    expected = independent_fit(frame.to_numpy(), matrix="covariance", components=2, anchors=[1, 0])
    np.testing.assert_allclose(result["estimates"].estimate, expected["parameters"], atol=3e-12)


def test_every_requested_failure_is_reported_without_subset(monkeypatch):
    original = u._parameters
    calls = []

    def injected(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) in (3, 12):
            raise AnalysisError("singular_matrix", "controlled refit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(u, "_parameters", injected)
    with pytest.raises(AnalysisError, match="2 of 39") as caught:
        u.pca_bootstrap(data(n=310), NAMES, components=2, replications=39, seed=7)
    error = caught.value
    assert error.code == "bootstrap_failure" and len(calls) == 40
    assert error.replications_attempted == 39 and error.successful_replications == 37
    assert [entry["replication"] for entry in error.failures] == [2, 11]
    assert all(entry["code"] == "singular_matrix" for entry in error.failures)


def diagonal_sample(variances):
    p = len(variances)
    # Repeated signed coordinate rows have exactly zero means and diagonal
    # covariance, letting degeneracy/anchor tests avoid a random near-tie.
    rows = np.vstack([np.eye(p), -np.eye(p)])*np.sqrt(variances)
    return pd.DataFrame(np.tile(rows, (8, 1)), columns=list("abcdef")[:p])


@pytest.mark.parametrize("variances,components", [([1, 1], 1), ([9, 1, 1], 2),
                                                 ([1, 1-5e-9], 1)])
def test_retained_internal_and_boundary_ties_refuse(variances, components):
    frame = diagonal_sample(variances)
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, list(frame), components=components, matrix="covariance")
    assert caught.value.code == "unidentified_component"


def test_zero_declared_anchor_refuses_at_point():
    frame = diagonal_sample([9, 1])
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, list(frame), components=1, matrix="covariance", anchors=["b"])
    assert caught.value.code == "unidentified_anchor"


def test_actual_singular_draw_refuses_all_and_never_discards_it():
    # Only one row carries variation in b. Some genuine iid draws omit it.
    frame = pd.DataFrame({"a": np.arange(20, dtype=float), "b": [1.]+[0.]*19})
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, ["a", "b"], components=1, matrix="covariance", replications=39, seed=28)
    error = caught.value
    assert error.code == "bootstrap_failure" and error.replications_attempted == 39
    assert error.successful_replications+len(error.failures) == 39
    assert any(entry["code"] == "zero_variance" for entry in error.failures)
    assert [entry["replication"] for entry in error.failures] == sorted(entry["replication"] for entry in error.failures)


def test_actual_axis_crossing_not_permuted_or_procrustes_aligned():
    frame = diagonal_sample([1.2, 1])
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, ["a", "b"], components=1, matrix="covariance", replications=39, seed=29)
    assert caught.value.code == "bootstrap_failure"
    assert any(failure["code"] in ("unidentified_anchor", "component_crossing")
               for failure in caught.value.failures)


def test_actual_nonzero_anchor_direction_crossing_is_explicit():
    values = np.random.default_rng(0).normal(size=(50, 2))*[1.1, 1]
    frame = pd.DataFrame(values, columns=["a", "b"])
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, ["a", "b"], components=1, matrix="covariance",
                        replications=39, seed=29)
    assert caught.value.code == "bootstrap_failure"
    assert [failure["replication"] for failure in caught.value.failures
            if failure["code"] == "component_crossing"] == [8, 14, 28]


@pytest.mark.parametrize("options", [{"matrix": "raw"}, {"components": True}, {"components": 0},
                                    {"components": 7}, {"components": 1.5}, {"replications": True},
                                    {"replications": 18}, {"replications": 2000}, {"confidence": 1},
                                    {"confidence": 0}, {"confidence": .999}, {"seed": -1},
                                    {"seed": True}, {"seed": 2**63}, {"missing": "pairwise"},
                                    {"anchors": ["a"]}, {"anchors": ["a", "absent"]}])
def test_invalid_options(options):
    with pytest.raises(AnalysisError):
        u.pca_bootstrap(data(n=70), NAMES, **({"components": 2} | options))


@pytest.mark.parametrize("options", [{"weights": "w"}, {"cluster": "group"},
                                    {"device": "mps"}, {"mineigen": 1}, {"rotate": "varimax"}])
def test_no_implicit_weight_cluster_device_rotation_or_count_selection(options):
    with pytest.raises(TypeError):
        u.pca_bootstrap(data(), NAMES, components=2, **options)


@pytest.mark.parametrize("kind", ["boolean", "complex", "text", "infinity", "huge", "constant", "aliased"])
def test_nonnumeric_or_unresolved_sample_refused(kind):
    frame = data(n=70)
    if kind == "boolean":
        frame["a"] = frame.a > frame.a.median()
    elif kind == "complex":
        frame["a"] = frame.a+1j
    elif kind == "text":
        frame["a"] = "text"
    elif kind == "infinity":
        frame.loc[0, "a"] = np.inf
    elif kind == "huge":
        frame.loc[0, "a"] = 1e151
    elif kind == "constant":
        frame["a"] = 1.
    else:
        frame["a"] = frame.b
    with pytest.raises(AnalysisError):
        u.pca_bootstrap(frame, NAMES, components=2)


def test_unsupported_index_identity_refused_before_refit(monkeypatch):
    frame = data(n=70)
    frame.index = pd.Index([object() for _ in range(70)])

    def forbidden(*args, **kwargs):
        raise AssertionError("PCA fit happened before identity validation")

    monkeypatch.setattr(u, "_parameters", forbidden)
    with pytest.raises(AnalysisError):
        u.pca_bootstrap(frame, NAMES, components=2)


def test_hierarchical_source_columns_and_lossy_axis_names_refuse():
    frame = data(n=70)
    frame.columns = pd.MultiIndex.from_tuples([(name, "measurement") for name in NAMES])
    with pytest.raises(AnalysisError, match="one source column level"):
        u.pca_bootstrap(frame, NAMES, components=2)
    frame = data(n=70)
    frame.columns.name = object()
    with pytest.raises(AnalysisError, match="lossless"):
        u.pca_bootstrap(frame, NAMES, components=2)


def test_dataset_summary_non_cpu_and_tiny_sample_refusals():
    from openecon.econometrics.multivariate.summary import prepare

    calls = []

    def batches():
        calls.append(True)
        yield data()

    stream = Dataset.from_batches(batches, columns=NAMES, row_count=603)
    with pytest.raises(AnalysisError, match="resident"):
        u.pca_bootstrap(stream, NAMES, components=2)
    assert not calls
    summary = prepare([[1, .6], [.6, 1]], n=100, columns=["a", "b"], matrix="correlation")
    with pytest.raises(AnalysisError, match="resident"):
        u.pca_bootstrap(summary, ["a", "b"], components=1)
    with pytest.raises(AnalysisError, match="CPU"):
        u.pca_bootstrap({name: torch.empty(70, device="meta") for name in NAMES}, NAMES, components=2)
    with pytest.raises(AnalysisError, match="complete rows"):
        u.pca_bootstrap(data(n=19), NAMES, components=2)


def test_row_variable_parameter_work_export_and_workspace_admission_before_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("rows were materialized before resource admission")

    monkeypatch.setattr(u, "_selected", forbidden)
    with pytest.raises(AnalysisError, match="10000"):
        u.pca_bootstrap({name: range(10001) for name in NAMES}, NAMES, components=2)
    names = [f"x{i}" for i in range(17)]
    with pytest.raises(AnalysisError, match="16 variables"):
        u.pca_bootstrap({name: range(50) for name in names}, names, components=2)
    names = names[:16]
    with pytest.raises(AnalysisError, match="128 joint"):
        u.pca_bootstrap({name: range(50) for name in names}, names, components=4)
    with pytest.raises(AnalysisError, match="work"):
        u.pca_bootstrap({name: range(10000) for name in names}, names, components=2, replications=1999)
    with pytest.raises(AnalysisError, match="export"):
        u.pca_bootstrap({name: range(10000) for name in ["a", "b"]}, ["a", "b"],
                        components=1, replications=1999)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        u.pca_bootstrap({name: range(603) for name in NAMES}, NAMES, components=2)
    label = "x"*4097
    with pytest.raises(AnalysisError, match="labels"):
        u.pca_bootstrap({label: range(100), "a": range(100)}, [label, "a"], components=1)


def test_source_index_admission_before_fit(monkeypatch):
    frame = data(n=310)
    frame.index = ["x"*5000+str(i) for i in range(len(frame))]

    def forbidden(*args, **kwargs):
        raise AssertionError("PCA fit happened before index admission")

    monkeypatch.setattr(u, "_parameters", forbidden)
    with pytest.raises(AnalysisError, match="one MiB"):
        u.pca_bootstrap(frame, NAMES, components=2, replications=39)


def test_index_text_and_depth_refuse_before_encoding_or_refit(monkeypatch):
    frame = data(n=70)
    frame.index = ["x"*64001, *range(69)]

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized label reached encoding")

    monkeypatch.setattr(u, "encode", forbidden)
    with pytest.raises(AnalysisError, match="64000"):
        u.pca_bootstrap(frame, NAMES, components=2, replications=39)
    value = "x"
    for _ in range(34):
        value = (value,)
    frame.index = pd.Index([value, *range(69)], tupleize_cols=False)
    with pytest.raises(AnalysisError, match="depth"):
        u.pca_bootstrap(frame, NAMES, components=2, replications=39)


def test_unrelated_default_device_and_dtype_do_not_change_results():
    frame = data(n=310)
    expected = u.pca_bootstrap(frame, NAMES, components=2, replications=39, seed=541)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            actual = u.pca_bootstrap(frame, NAMES, components=2, replications=39, seed=541)
    finally:
        torch.set_default_dtype(previous)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])
    assert actual.attrs["device"] == "cpu" and actual.attrs["precision"] == "float64"
