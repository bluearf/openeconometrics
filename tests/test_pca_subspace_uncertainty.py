"""Independent invariant-projector IID oracles and admission/persistence checks."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import pca_subspace as s
from openecon.econometrics.multivariate import pca_uncertainty as u
from openecon.econometrics.multivariate.summary import prepare
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget

NAMES = ["a", "b", "c", "d", "e", "f"]


def domain(distribution="gaussian", *, n=603, seed=891):
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        latent, errors = rng.normal(size=(n, 2)), rng.normal(size=(n, 6))
    else:
        latent = rng.lognormal(sigma=.45, size=(n, 2))-np.exp(.45**2/2)
        latent /= np.sqrt(np.expm1(.45**2)*np.exp(.45**2))
        errors = rng.standard_t(8, size=(n, 6))/np.sqrt(8/6)
    loading = np.array([.95, .93, .9, .8, .78, .76])
    values = latent[:, [0, 0, 0, 1, 1, 1]]*loading
    values += errors*np.sqrt(1-loading**2)
    values = values*[3, 2.5, 2, 1.5, 1.3, 1.1]+[5, -2, 12, .5, 9, -3]
    return pd.DataFrame(values, columns=NAMES)


def oracle(values, *, matrix, components):
    values = np.asarray(values, dtype=np.float64)
    origin = values[0]
    shifted = values-origin
    offset = shifted.mean(0)
    centred = shifted-offset
    correction = centred.mean(0)
    centred -= correction
    covariance = centred.T@centred/(len(values)-1)
    sd = np.sqrt(covariance.diagonal())
    target = covariance.copy()
    if matrix == "correlation":
        target /= np.outer(sd, sd)
        np.fill_diagonal(target, 1)
    roots, vectors = np.linalg.eigh(target)
    roots, vectors = roots[::-1], vectors[:, ::-1]
    projector = vectors[:, :components]@vectors[:, :components].T
    captured = np.trace(projector@target)
    residual = np.trace((np.eye(len(sd))-projector)@target)
    parameters = np.r_[projector[np.triu_indices(len(sd))], captured, residual,
                        captured/np.trace(target)]
    # The second oracle uses a rectangular data SVD rather than covariance EVD.
    standardized = centred if matrix == "covariance" else centred/sd
    _, _, right = np.linalg.svd(standardized, full_matrices=False)
    svd_projector = right[:components].T@right[:components]
    return {"parameters": parameters, "projector": projector, "roots": roots,
            "covariance": covariance, "target": target, "mean": origin+offset+correction,
            "sd": sd, "svd_projector": svd_projector}


def literal_draws(values, *, matrix, components, seed, replications):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = np.array([torch.randint(len(values), (len(values),), generator=generator,
                                     device="cpu").numpy() for _ in range(replications)])
    fits = [oracle(values[index], matrix=matrix, components=components) for index in indices]
    return indices, fits


@pytest.mark.parametrize("distribution", ["gaussian", "skewed_t8"])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_two_domains_all_draws_joint_covariance_intervals_and_moments(distribution, matrix):
    frame = domain(distribution)
    result = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix=matrix,
                                      replications=199, seed=913)
    point = oracle(frame.to_numpy(), matrix=matrix, components=2)
    indices, fits = literal_draws(frame.to_numpy(), matrix=matrix, components=2,
                                  replications=199, seed=913)
    expected = np.array([fit["parameters"] for fit in fits])
    np.testing.assert_array_equal(result["resample_indices"], indices)
    np.testing.assert_allclose(result["replicates"], expected, atol=2e-11, rtol=3e-12)
    np.testing.assert_allclose(result["estimates"].estimate, point["parameters"], atol=2e-11)
    joint = np.cov(expected, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], joint, atol=2e-12, rtol=3e-10)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(joint.diagonal()), atol=4e-12)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(expected, [.025, .975], axis=0, method="linear").T,
                               atol=2e-11, rtol=3e-12)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias,
                               expected.mean(0)-point["parameters"], atol=2e-11)
    for table_name, key in [("point_projector", "projector"), ("point_covariance", "covariance"),
                             ("point_matrix", "target")]:
        np.testing.assert_allclose(result[table_name], point[key], atol=8e-12)
    np.testing.assert_allclose(result["point_eigenvalues"].eigenvalue, point["roots"], atol=8e-12)
    np.testing.assert_allclose(result["replicate_means"], [fit["mean"] for fit in fits], atol=3e-12)
    np.testing.assert_allclose(result["replicate_standard_deviations"], [fit["sd"] for fit in fits], atol=3e-12)
    for fit in fits:
        np.testing.assert_allclose(fit["projector"], fit["svd_projector"], atol=6e-13)
    diagnostics = result["replicate_diagnostics"]
    np.testing.assert_allclose(diagnostics.frobenius_distance_to_point_projector,
                               [np.linalg.norm(fit["projector"]-point["projector"]) for fit in fits], atol=4e-13)
    np.testing.assert_allclose(diagnostics.relative_boundary_eigenvalue_gap,
                               [(fit["roots"][1]-fit["roots"][2])/fit["roots"][0] for fit in fits], atol=5e-14)
    assert result["estimates"][["df", "p_value"]].isna().all().all()
    assert result.attrs["successful_replications"] == 199 and result.attrs["failed_replications"] == []
    assert result.attrs["parameter_dimension"] == 24
    assert result.attrs["parameter_order"] == result["parameter_order"].values.tolist()
    assert result.attrs["covariance_may_be_singular"] and not result.attrs["covariance_inverse_available"]
    assert not result.attrs["individual_eigenvalue_inference_available"]
    assert not result.attrs["eigenvector_inference_available"]
    assert not result.attrs["simultaneous_region_available"]
    assert "point_eigenvectors" not in result
    assert "nonzero functional derivative variance" in result.attrs["inferential_assumptions"]


def represented_sample(covariance, *, n=240, seed=492):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, len(covariance)))
    values -= values.mean(0)
    q, _ = np.linalg.qr(values)
    roots, vectors = np.linalg.eigh(covariance)
    factor = (vectors*np.sqrt(roots))@vectors.T
    return pd.DataFrame(q@factor*np.sqrt(n-1), columns=NAMES[:len(covariance)])


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_internal_repeated_roots_admitted_with_no_individual_axis_or_root_ci(matrix):
    if matrix == "covariance":
        rotation, _ = np.linalg.qr(np.random.default_rng(612).normal(size=(6, 6)))
        target = (rotation*np.array([8., 8., 2., 2., .5, .5]))@rotation.T
    else:
        target = np.zeros((6, 6))
        target[:3, :3] = target[3:, 3:] = np.full((3, 3), .8)+.2*np.eye(3)
    frame = represented_sample(target)
    result = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix=matrix,
                                      replications=39, seed=492)
    roots = result["point_eigenvalues"].eigenvalue.to_numpy()
    np.testing.assert_allclose(roots[0], roots[1], atol=3e-14)
    assert result.attrs["point_diagnostics"][0] > .7
    np.testing.assert_allclose(result["point_projector"],
                               oracle(frame.to_numpy(), matrix=matrix, components=2)["projector"], atol=1e-13)
    assert set(result["parameter_order"].kind) == {"projector", "captured_variance",
                                                   "residual_variance", "captured_trace_share"}
    # The existing eigenpair method deliberately refuses this same point fit.
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, NAMES, components=2, matrix=matrix, replications=39)
    assert caught.value.code == "unidentified_component"


def test_rotating_each_retained_and_discarded_basis_changes_no_target(monkeypatch):
    frame = domain(n=310)
    reference = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix="covariance",
                                         replications=39, seed=73)
    original = torch.linalg.eigh
    left, _ = np.linalg.qr(np.random.default_rng(93).normal(size=(4, 4)))
    right, _ = np.linalg.qr(np.random.default_rng(94).normal(size=(2, 2)))
    rotation = np.zeros((6, 6))
    rotation[:4, :4], rotation[4:, 4:] = left, right
    basis_change = torch.tensor(rotation, dtype=torch.float64)

    def rotated(*args, **kwargs):
        roots, vectors = original(*args, **kwargs)
        return roots, vectors@basis_change

    monkeypatch.setattr(torch.linalg, "eigh", rotated)
    actual = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix="covariance",
                                      replications=39, seed=73)
    for name in ["replicates", "estimates", "covariance", "point_projector"]:
        np.testing.assert_allclose(actual[name], reference[name], atol=4e-12, equal_nan=True)


def test_finite_difference_matches_cross_boundary_projector_derivative_with_internal_ties():
    rng = np.random.default_rng(691)
    vectors, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    roots = np.array([8., 8., 2., 2., .5, .5])
    target = (vectors*roots)@vectors.T
    perturbation = rng.normal(size=(6, 6))
    perturbation = (perturbation+perturbation.T)/2
    derivative = np.zeros((6, 6))
    for i in range(2):
        for j in range(2, 6):
            weight = vectors[:, j]@perturbation@vectors[:, i]/(roots[i]-roots[j])
            derivative += weight*(np.outer(vectors[:, j], vectors[:, i])
                                  +np.outer(vectors[:, i], vectors[:, j]))
    epsilon = 1e-5
    plus = s._parameters(torch.tensor(represented_sample(target+epsilon*perturbation).to_numpy()),
                          "covariance", 2)[1].numpy()
    minus = s._parameters(torch.tensor(represented_sample(target-epsilon*perturbation).to_numpy()),
                           "covariance", 2)[1].numpy()
    np.testing.assert_allclose((plus-minus)/(2*epsilon), derivative, atol=2e-10, rtol=2e-8)


def test_structurally_singular_covariance_and_trace_identities_are_not_inverted():
    result = s.pca_subspace_bootstrap(domain(), NAMES, components=2, replications=39, seed=53)
    draws = result["replicates"]
    np.testing.assert_allclose(draws[[f"P[{i},{i}]" for i in range(1, 7)]].sum(axis=1), 2, atol=3e-14)
    np.testing.assert_allclose(draws.captured_variance+draws.residual_variance, 6, atol=1e-14)
    np.testing.assert_allclose(draws.captured_trace_share, draws.captured_variance/6, atol=1e-15)
    assert np.linalg.matrix_rank(result["covariance"], tol=1e-10) < 24
    assert result.attrs["covariance_inverse_available"] is False
    assert result.attrs["familywise_intervals"] is False


def test_correlation_restandardizes_each_draw_and_is_unit_invariant():
    frame = domain("skewed_t8")
    result = s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=59, confidence=.9, seed=815)
    fixed_sd = result["descriptives"].std_dev.to_numpy()
    incorrect = []
    for index in result["resample_indices"].to_numpy():
        covariance = np.cov(frame.to_numpy()[index]/fixed_sd, rowvar=False, ddof=1)
        incorrect.append(np.linalg.eigvalsh(covariance)[-2:].sum())
    assert np.max(np.abs(result["replicates"].captured_variance-incorrect)) > .05
    assert np.std(result["replicate_standard_deviations"], axis=0).min() > .01
    changed = frame*[.01, 100, .2, 7, 11, 3]+[600, -900, 1, 2, 3, 4]
    other = s.pca_subspace_bootstrap(changed, NAMES, components=2,
                                     replications=59, confidence=.9, seed=815)
    np.testing.assert_allclose(other["replicates"], result["replicates"], atol=8e-12)
    assert result.attrs["restandardize_every_draw"] and result.attrs["reestimate_moments_every_draw"]
    covariance = s.pca_subspace_bootstrap(changed, NAMES, components=2, matrix="covariance",
                                          replications=39, seed=815)
    assert np.max(covariance["replicates"].captured_variance) > 1000
    assert not covariance.attrs["restandardize_every_draw"]


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_large_represented_origin_agrees_with_stable_independent_moments(matrix):
    frame = domain()+1e12
    result = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix=matrix,
                                      replications=39, seed=815)
    _, fitted = literal_draws(frame.to_numpy(), matrix=matrix, components=2, replications=39, seed=815)
    np.testing.assert_allclose(result["replicates"], [fit["parameters"] for fit in fitted], atol=2e-11)
    np.testing.assert_allclose(result["replicate_means"], [fit["mean"] for fit in fitted], atol=.00025)


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_finite_json_full_state_replay_seed_and_typed_identity(matrix):
    frame = domain(n=310)
    frame.index = pd.MultiIndex.from_tuples([(Decimal(i), None if i % 2 else "same")
                                            for i in range(310)], names=["serial", 4])
    frame.columns.name = Decimal("17")
    result = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix=matrix,
                                      replications=59, confidence=.9, seed=19)
    encoded = summary_state(result)
    assert "NaN" not in encoded and "Infinity" not in encoded
    saved = restore_summary(encoded)
    assert saved.attrs == result.attrs
    for name in result:
        left = saved[name].astype(object).where(saved[name].notna(), None)
        right = result[name].astype(object).where(result[name].notna(), None)
        pd.testing.assert_frame_equal(left, right)
    assert [encode(decode(code)) for code in saved.attrs["sample_index_codes"]] == saved.attrs["sample_index_codes"]
    assert [decode(code) for code in saved.attrs["source_index_names"]] == ["serial", 4]
    assert [decode(code) for code in saved.attrs["source_column_names"]] == [Decimal("17")]
    replay = s.pca_subspace_bootstrap(saved["sample"], NAMES, components=2, matrix=matrix,
                                      replications=59, confidence=.9, seed=19)
    repeated = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix=matrix,
                                        replications=59, confidence=.9, seed=19)
    changed = s.pca_subspace_bootstrap(frame, NAMES, components=2, matrix=matrix,
                                       replications=59, confidence=.9, seed=20)
    for name in ["replicates", "covariance", "estimates", "resample_indices"]:
        pd.testing.assert_frame_equal(replay[name], result[name])
        pd.testing.assert_frame_equal(repeated[name], result[name])
    assert not changed["resample_indices"].equals(result["resample_indices"])
    assert not changed["replicates"].equals(result["replicates"])
    identity = {"variables": NAMES, "positions": result.attrs["sample_positions"],
                "index_codes": result.attrs["sample_index_codes"],
                "index_names": result.attrs["source_index_names"], "index_nlevels": 2,
                "column_names": result.attrs["source_column_names"], "column_nlevels": 1,
                "matrix": matrix, "components": 2}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    digest.update(frame.to_numpy().tobytes())
    assert digest.hexdigest() == result.attrs["source_content_sha256"]


def test_missing_alignment_duplicate_typed_labels_and_input_preservation():
    frame = domain(n=310)
    frame.index = pd.Index([None, np.nan, pd.NA, True, 1, Decimal("1"), *range(304)], dtype=object)
    frame.iloc[8, 1] = np.nan
    frame.iloc[19, 4] = np.nan
    frame["ignored"] = [object() for _ in range(310)]
    before = frame.copy(deep=True)
    result = s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=39, seed=212)
    keep = frame[NAMES].notna().all(axis=1)
    reference = s.pca_subspace_bootstrap(frame.loc[keep, NAMES], NAMES, components=2,
                                         replications=39, seed=212)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result["replicates"], reference["replicates"])
    assert result.attrs["n_missing"] == 2 and result.attrs["n"] == 308
    assert result.attrs["sample_positions"] == np.flatnonzero(keep).tolist()
    assert result.attrs["sample_index_codes"] == [encode(value) for value in frame.index[keep]]
    assert len(set(result.attrs["sample_index_codes"][:6])) == 6
    assert result.attrs["sample_positions"] == result["sample"].index.tolist()
    json.dumps(result.attrs, allow_nan=False)
    with pytest.raises(AnalysisError, match="missing"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2, missing="raise")


def test_mapping_records_ordinal_labels_and_unused_unsized_column():
    frame = domain(n=310)
    names = ["captured_variance", "P[1,1]", "a:b", "a", "b:a", "P[2,2]"]
    frame.columns = names
    direct = s.pca_subspace_bootstrap(frame, names, components=2, replications=39, seed=715)
    mapping = frame.to_dict(orient="list")
    mapping["ignored"] = object()
    mapped = s.pca_subspace_bootstrap(mapping, names, components=2, replications=39, seed=715)
    records = s.pca_subspace_bootstrap(frame.to_dict(orient="records"), names,
                                       components=2, replications=39, seed=715)
    pd.testing.assert_frame_equal(mapped["replicates"], direct["replicates"])
    pd.testing.assert_frame_equal(records["replicates"], direct["replicates"])
    assert direct["replicates"].columns.is_unique
    assert direct["parameter_order"].row_variable.iloc[0] == "captured_variance"


@pytest.mark.parametrize("roots,components", [([1, 1], 1), ([9, 1, 1], 2), ([1, 1-5e-9], 1)])
def test_only_boundary_ties_refuse(roots, components):
    frame = represented_sample(np.diag(roots))
    with pytest.raises(AnalysisError) as caught:
        s.pca_subspace_bootstrap(frame, list(frame), components=components, matrix="covariance")
    assert caught.value.code == "unidentified_subspace"


def test_every_injected_failure_attempts_and_reports_all_requested_draws(monkeypatch):
    original, calls = s._parameters, []

    def injected(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) in (3, 12):
            raise AnalysisError("unidentified_subspace", "controlled boundary tie")
        return original(*args, **kwargs)

    monkeypatch.setattr(s, "_parameters", injected)
    with pytest.raises(AnalysisError, match="2 of 39") as caught:
        s.pca_subspace_bootstrap(domain(n=310), NAMES, components=2, replications=39, seed=7)
    error = caught.value
    assert error.code == "bootstrap_failure" and len(calls) == 40
    assert error.replications_attempted == 39 and error.successful_replications == 37
    assert [entry["replication"] for entry in error.failures] == [2, 11]
    assert all(entry["code"] == "unidentified_subspace" for entry in error.failures)


def test_real_singular_draw_refuses_full_result_instead_of_discarding_draws():
    frame = pd.DataFrame({"a": np.arange(20, dtype=float), "b": [1.]+[0.]*19})
    with pytest.raises(AnalysisError) as caught:
        s.pca_subspace_bootstrap(frame, ["a", "b"], components=1, matrix="covariance",
                                 replications=39, seed=28)
    error = caught.value
    assert error.code == "bootstrap_failure" and error.replications_attempted == 39
    assert error.successful_replications+len(error.failures) == 39
    assert any(entry["code"] == "zero_variance" for entry in error.failures)


def test_subspace_does_not_inherit_axis_anchor_or_direction_chart_refusals():
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 2))*[1.1, 1], columns=["a", "b"])
    with pytest.raises(AnalysisError) as caught:
        u.pca_bootstrap(frame, ["a", "b"], components=1, matrix="covariance", replications=39, seed=29)
    assert any(failure["code"] == "component_crossing" for failure in caught.value.failures)
    result = s.pca_subspace_bootstrap(frame, ["a", "b"], components=1, matrix="covariance",
                                      replications=39, seed=29)
    assert result.attrs["successful_replications"] == 39
    assert result["replicate_diagnostics"].frobenius_distance_to_point_projector.max() > 1


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
@pytest.mark.parametrize("scale", [1e-158, 1e-160])
def test_subnormal_raw_moments_refuse_before_restandardization(matrix, scale):
    with pytest.raises(AnalysisError, match="raw sample variance is subnormal") as caught:
        s.pca_subspace_bootstrap(domain()*scale, NAMES, components=2, matrix=matrix,
                                 replications=39, seed=713)
    assert caught.value.code == "numerical_failure"


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_nonconstant_covariance_underflow_or_overflow_refuses(scale):
    with pytest.raises(AnalysisError) as caught:
        s.pca_subspace_bootstrap(domain()*scale, NAMES, components=2, matrix="covariance",
                                 replications=39, seed=713)
    assert caught.value.code == "numerical_failure"
    actual = s.pca_subspace_bootstrap(domain()*scale, NAMES, components=2, replications=39, seed=713)
    reference = s.pca_subspace_bootstrap(domain(), NAMES, components=2, replications=39, seed=713)
    np.testing.assert_allclose(actual["replicates"], reference["replicates"], atol=8e-12)


@pytest.mark.parametrize("options", [{"matrix": "raw"}, {"components": True}, {"components": 0},
                                    {"components": 6}, {"components": 1.5}, {"replications": True},
                                    {"replications": 18}, {"replications": 2000}, {"confidence": 1},
                                    {"confidence": 0}, {"confidence": .999}, {"seed": -1},
                                    {"seed": True}, {"seed": 2**63}, {"missing": "pairwise"}])
def test_invalid_options(options):
    with pytest.raises(AnalysisError):
        s.pca_subspace_bootstrap(domain(n=70), NAMES, **({"components": 2} | options))


@pytest.mark.parametrize("options", [{"weights": "w"}, {"cluster": "g"}, {"device": "mps"},
                                    {"mineigen": 1}, {"rotate": "varimax"}, {"anchors": ["a", "d"]}])
def test_unsupported_weight_cluster_device_selected_count_and_axis_options(options):
    with pytest.raises(TypeError):
        s.pca_subspace_bootstrap(domain(), NAMES, components=2, **options)


@pytest.mark.parametrize("kind", ["boolean", "complex", "text", "infinity", "huge", "constant", "aliased"])
def test_nonnumeric_or_unresolved_sample_refused(kind):
    frame = domain(n=70)
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
        s.pca_subspace_bootstrap(frame, NAMES, components=2)


def test_dataset_summary_non_cpu_tiny_sample_and_duplicate_columns_refused():
    calls = []

    def batches():
        calls.append(True)
        yield domain()

    stream = Dataset.from_batches(batches, columns=NAMES, row_count=603)
    with pytest.raises(AnalysisError, match="resident"):
        s.pca_subspace_bootstrap(stream, NAMES, components=2)
    assert not calls
    summary = prepare([[1, .6], [.6, 1]], n=100, columns=["a", "b"], matrix="correlation")
    with pytest.raises(AnalysisError, match="resident"):
        s.pca_subspace_bootstrap(summary, ["a", "b"], components=1)
    with pytest.raises(AnalysisError, match="CPU"):
        s.pca_subspace_bootstrap({name: torch.empty(70, device="meta") for name in NAMES}, NAMES, components=2)
    with pytest.raises(AnalysisError, match="complete rows"):
        s.pca_subspace_bootstrap(domain(n=19), NAMES, components=2)
    frame = domain()
    frame.columns = ["a", "a", "c", "d", "e", "f"]
    with pytest.raises(AnalysisError, match="unique column"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2)


def test_row_width_work_export_and_workspace_admission_before_conversion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("rows materialized before admission")

    monkeypatch.setattr(u, "_selected", forbidden)
    with pytest.raises(AnalysisError, match="10000"):
        s.pca_subspace_bootstrap({name: range(10001) for name in NAMES}, NAMES, components=2)
    names = [f"x{i}" for i in range(16)]
    with pytest.raises(AnalysisError, match="15 variables"):
        s.pca_subspace_bootstrap({name: range(50) for name in names}, names, components=2)
    names = names[:15]
    with pytest.raises(AnalysisError, match="work"):
        s.pca_subspace_bootstrap({name: range(10000) for name in names}, names, components=2, replications=1999)
    with pytest.raises(AnalysisError, match="export"):
        s.pca_subspace_bootstrap({name: range(10000) for name in ["a", "b"]}, ["a", "b"],
                                 components=1, replications=1999)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        s.pca_subspace_bootstrap({name: range(603) for name in NAMES}, NAMES, components=2)
    label = "x"*4097
    with pytest.raises(AnalysisError, match="labels"):
        s.pca_subspace_bootstrap({label: range(100), "a": range(100)}, [label, "a"], components=1)


def test_source_identity_and_hierarchical_columns_refused_before_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fit happened before identity admission")

    monkeypatch.setattr(s, "_parameters", forbidden)
    frame = domain(n=70)
    frame.index = pd.Index([object() for _ in range(70)])
    with pytest.raises(AnalysisError, match="lossless"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2)
    frame = domain(n=310)
    frame.index = ["x"*5000+str(i) for i in range(310)]
    with pytest.raises(AnalysisError, match="one MiB"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=39)
    frame = domain(n=70)
    frame.columns = pd.MultiIndex.from_tuples([(name, "measurement") for name in NAMES])
    with pytest.raises(AnalysisError, match="one source column level"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2)
    frame = domain(n=70)
    frame.columns.name = object()
    with pytest.raises(AnalysisError, match="lossless"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2)


def test_index_text_and_depth_refuse_before_encoding(monkeypatch):
    frame = domain(n=70)
    frame.index = ["x"*64001, *range(69)]

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized label reached encoding")

    monkeypatch.setattr(u, "encode", forbidden)
    with pytest.raises(AnalysisError, match="64000"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=39)
    value = "x"
    for _ in range(34):
        value = (value,)
    frame.index = pd.Index([value, *range(69)], tupleize_cols=False)
    with pytest.raises(AnalysisError, match="depth"):
        s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=39)


@pytest.mark.parametrize("kind", ["integer", "decimal", "nested_decimal"])
def test_oversized_numeric_source_identity_refuses_before_scalar_encoding(monkeypatch, kind):
    value = 1 << 100000 if kind == "integer" else Decimal((0, (1,)*20000, 0))
    if kind == "nested_decimal":
        value = ("small", value)
    frame = domain(n=70)
    frame.index = pd.Index([value, *range(69)], dtype=object, tupleize_cols=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded numeric identity reached scalar encoding")

    monkeypatch.setattr(u, "encode", forbidden)
    with pytest.raises(AnalysisError, match="numeric source identity") as caught:
        s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=39)
    assert caught.value.code == "resource_limit"


@pytest.mark.parametrize("kind", ["autograd", "sparse"])
def test_unsupported_tensor_representation_refused_before_coercion(monkeypatch, kind):
    source = {name: torch.arange(70, dtype=torch.float64) for name in NAMES}
    if kind == "autograd":
        source["a"].requires_grad_(True)
    else:
        source["a"] = source["a"].to_sparse()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported tensor reached table coercion")

    monkeypatch.setattr(u, "_selected", forbidden)
    with pytest.raises(AnalysisError, match="strided CPU values") as caught:
        s.pca_subspace_bootstrap(source, NAMES, components=2, replications=39)
    assert caught.value.code == "invalid_data"


def test_default_device_and_dtype_do_not_change_native_cpu_float64_results():
    frame = domain(n=310)
    expected = s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=39, seed=541)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            actual = s.pca_subspace_bootstrap(frame, NAMES, components=2, replications=39, seed=541)
    finally:
        torch.set_default_dtype(previous)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])
    assert actual.attrs["device"] == "cpu" and actual.attrs["precision"] == "float64"
