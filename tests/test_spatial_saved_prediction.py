"""Independent keyed reduced-form means, joint delta covariance and restoration."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.index_codec import decode
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


MODELS = ["sar", "sem", "sac", "sdm"]


@lru_cache
def _fixture(model):
    n = 24
    keys = [f"unit-{i}" for i in range(n)]
    weights = oe.spatial_weights(
        keys,
        [
            (keys[i], keys[j], weight)
            for i in range(n)
            for j, weight in (((i - 1) % n, 1.0), ((i + 1) % n, 2.0), ((i + 5) % n, 0.3))
        ],
    )
    w = weights.dense().numpy()
    rng = np.random.default_rng(281)
    x, z = rng.normal(size=n), rng.normal(size=n)
    y = np.linalg.solve(
        np.eye(n) - 0.35 * w,
        1 + 0.8 * x - 0.4 * z + np.linalg.solve(np.eye(n) - 0.2 * w, rng.normal(size=n)),
    )
    data = pd.DataFrame({"id": keys, "y": y, "x": x, "z": z})
    options = {}
    if model == "sac":
        we = np.roll(w, 2, axis=1)
        np.fill_diagonal(we, 0)
        we /= we.sum(axis=1)[:, None]
        options["error_weights"] = oe.spatial_weights(
            keys, [(keys[i], keys[j], we[i, j]) for i, j in zip(*np.nonzero(we))]
        )
    result = getattr(oe, model)(data, "y", ["x", "z"], key="id", spatial_weights=weights, **options)
    return data, result, w


def _case(model):
    data, result, w = _fixture(model)
    return (
        data.copy(deep=True),
        ResultBundle.model_validate_json(result.model_dump_json()),
        w.copy(),
    )


def _oracle(result, query, w):
    """NumPy inverse/analytic derivatives, independent of Torch solve code."""
    terms = [c.term for c in result.coefficients]
    beta = np.array([c.estimate for c in result.coefficients])
    covariance = np.array(result.covariance_matrix)
    x = query[result.spec.predictors].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(query)), x)) if result.spec.intercept else x
    names = (["Intercept"] if result.spec.intercept else []) + result.spec.predictors
    if result.spec.estimator == "sdm":
        design = np.column_stack((design, w @ x))
        names += ["W:" + name for name in result.spec.predictors]
    indices = [terms.index(name) for name in names]
    inverse = (
        np.eye(len(query))
        if result.spec.estimator == "sem"
        else np.linalg.inv(np.eye(len(query)) - beta[terms.index("rho")] * w)
    )
    mean = inverse @ design @ beta[indices]
    gradient = np.zeros((len(query), len(beta)))
    gradient[:, indices] = inverse @ design
    if "rho" in terms:
        gradient[:, terms.index("rho")] = inverse @ w @ mean
    return mean, gradient, gradient @ covariance @ gradient.T


def _finite_difference(result, query, w):
    terms = [c.term for c in result.coefficients]
    beta = np.array([c.estimate for c in result.coefficients])
    x = query[result.spec.predictors].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(query)), x)) if result.spec.intercept else x
    names = (["Intercept"] if result.spec.intercept else []) + result.spec.predictors
    if result.spec.estimator == "sdm":
        design = np.column_stack((design, w @ x))
        names += ["W:" + name for name in result.spec.predictors]

    def mean(parameters):
        index = design @ parameters[[terms.index(name) for name in names]]
        if result.spec.estimator == "sem":
            return index
        return np.linalg.solve(np.eye(len(query)) - parameters[terms.index("rho")] * w, index)

    gradient = []
    for j, value in enumerate(beta):
        step = 2e-5 * (1 + abs(value))
        direction = np.zeros(len(beta))
        direction[j] = step
        gradient.append((mean(beta + direction) - mean(beta - direction)) / (2 * step))
    return np.array(gradient).T


@pytest.mark.parametrize("model", MODELS)
def test_restored_means_full_delta_covariance_normal_endpoints_and_independent_gradient(model):
    data, result, w = _case(model)
    query = data[["id", "x", "z"]].copy()
    query["x"] = 0.7 * query.x + 0.3
    query["z"] = -0.4 * query.z + 0.2
    before = result.model_dump_json()
    expected, gradient, covariance = _oracle(result, query, w)
    numeric_gradient = _finite_difference(result, query, w)
    out = oe.spatial_predict(result, data=query, alpha=0.1)
    assert_allclose(
        out["means"].mean.to_numpy() if not callable(out["means"].mean) else out["means"]["mean"],
        expected,
        rtol=2e-12,
        atol=3e-13,
    )
    assert_allclose(out["jacobian"], gradient, rtol=2e-12, atol=3e-13)
    assert_allclose(out["jacobian"], numeric_gradient, rtol=2e-8, atol=4e-9)
    assert_allclose(out["mean_covariance"], covariance, rtol=3e-12, atol=4e-13)
    assert_allclose(out["parameter_covariance"], result.covariance_matrix, rtol=0, atol=0)
    se = np.sqrt(covariance.diagonal())
    critical = norm.isf(0.05)
    assert_allclose(out["means"].std_error, se, rtol=3e-12, atol=3e-13)
    assert_allclose(out["means"].ci_low, expected - critical * se, atol=4e-12)
    assert_allclose(out["means"].ci_high, expected + critical * se, atol=4e-12)
    assert out.attrs["refitted"] is False
    assert out.attrs["critical_value"] == pytest.approx(critical, abs=2e-14)
    assert out.attrs["parameter_order"] == [c.term for c in result.coefficients]
    assert out.attrs["covariance_key_order"] == data.id.tolist()
    assert (
        out.attrs["source_result_sha256"]
        == hashlib.sha256(
            json.dumps(
                result.model_dump(mode="json"),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert out.attrs["backward_solve_relative_error"] < 1e-13
    assert result.model_dump_json() == before
    # Full cross blocks matter, unlike simply multiplying marginal variances.
    diagonal_approximation = gradient @ np.diag(np.diag(result.covariance_matrix)) @ gradient.T
    assert np.max(np.abs(diagonal_approximation - covariance)) > 1e-5


@pytest.mark.parametrize("model", MODELS)
def test_key_and_saved_parameter_permutations_preserve_complete_joint_geometry(model):
    data, result, w = _case(model)
    query = data[["id", "x", "z"]]
    reference = oe.spatial_predict(result, data=query)
    rows = np.random.default_rng(87).permutation(len(data))
    shuffled = query.iloc[rows].copy()
    shuffled.index = pd.Index([("row", int(i % 3)) for i in rows], tupleize_cols=False)
    changed = oe.spatial_predict(result, data=shuffled)
    assert changed["means"].key.tolist() == shuffled.id.tolist()
    assert_allclose(
        changed["means"][["mean", "std_error", "ci_low", "ci_high"]],
        reference["means"].iloc[rows][["mean", "std_error", "ci_low", "ci_high"]],
        atol=5e-12,
    )
    assert_allclose(
        changed["mean_covariance"],
        reference["mean_covariance"].to_numpy()[np.ix_(rows, rows)],
        atol=8e-12,
    )
    assert [decode(value) for value in changed["sample"].index_json] == shuffled.index.tolist()
    permutation = np.arange(len(result.coefficients))[::-1]
    result.coefficients = [result.coefficients[i] for i in permutation]
    result.covariance_matrix = np.array(result.covariance_matrix)[
        np.ix_(permutation, permutation)
    ].tolist()
    permuted = oe.spatial_predict(result, data=query)
    assert_allclose(
        permuted["means"][["mean", "std_error"]],
        reference["means"][["mean", "std_error"]],
        atol=5e-13,
    )
    assert_allclose(permuted["mean_covariance"], reference["mean_covariance"], atol=5e-13)
    assert_allclose(
        permuted["jacobian"], reference["jacobian"].to_numpy()[:, permutation], atol=5e-13
    )
    assert permuted.attrs["parameter_order"] == [c.term for c in result.coefficients]


@pytest.mark.parametrize("model", MODELS)
def test_nuisance_zero_gradients_constant_query_design_and_no_refitting(model, monkeypatch):
    data, result, w = _case(model)
    query = data[["id", "x", "z"]].copy()
    query["x"], query["z"] = 0.0, 1.0

    def fail(*_args, **_kwargs):
        raise AssertionError("A fit/optimizer was called for restored prediction")

    from openecon.econometrics.spatial import estimators, kernels

    monkeypatch.setattr(estimators, "fit_spatial", fail)
    monkeypatch.setattr(kernels, "fit_ml", fail)
    before = torch.random.get_rng_state().clone()
    out = oe.spatial_predict(result, data=query)
    assert torch.equal(before, torch.random.get_rng_state())
    for term in ("lambda", "ln_sigma2"):
        if term in out.attrs["parameter_order"]:
            assert (out["jacobian"]["parameter:" + term] == 0).all()
    expected, _, _ = _oracle(result, query, w)
    assert_allclose(out["means"]["mean"], expected, atol=3e-13)
    if model == "sem":
        assert out.attrs["backward_solve_relative_error"] == 0


@pytest.mark.parametrize("model", MODELS)
def test_full_summary_roundtrip_keeps_all_covariances_gradients_query_and_hashes(model):
    data, result, _ = _case(model)
    out = oe.spatial_predict(result, data=data[["id", "x", "z"]])
    encoded = oe.summary_state(out)
    replay = oe.restore_summary(encoded)
    assert oe.summary_state(replay) == encoded
    assert replay.attrs == out.attrs
    assert replay["mean_covariance"].shape == (24, 24)
    assert replay["jacobian"].shape == (24, len(result.coefficients))
    assert replay["sample"].key.tolist() == data.id.tolist()
    assert "tabular" in out["means"].to_latex()
    plan = out.attrs["resource_plan"]
    assert plan["estimated_workspace_bytes"] == sum(plan["buffers"].values())
    assert plan["estimated_workspace_bytes"] <= plan["budget_bytes"]


def test_actual_missing_fit_induces_and_restores_only_the_effective_closed_graph():
    data, original, _ = _case("sar")
    data.loc[3, "x"] = np.nan
    weights = oe.spatial_weights(
        original.extra["spatial_weights"]["keys"],
        [
            (
                original.extra["spatial_weights"]["keys"][i],
                original.extra["spatial_weights"]["keys"][j],
                v,
            )
            for i, j, v in zip(
                original.extra["spatial_weights"]["rows"],
                original.extra["spatial_weights"]["cols"],
                original.extra["spatial_weights"]["values"],
            )
        ],
    )
    fitted = oe.sar(data, "y", ["x", "z"], key="id", spatial_weights=weights, missing="drop")
    restored = ResultBundle.model_validate_json(fitted.model_dump_json())
    effective = data.drop(index=3)[["id", "x", "z"]]
    out = oe.spatial_predict(restored, data=effective)
    assert out.attrs["n"] == 23
    assert "unit-3" not in out.attrs["covariance_key_order"]
    assert out.attrs["saved_graph_sha256"] == restored.provenance["spatial_weight_hash"]
    with pytest.raises(AnalysisError, match="effective fitted-network"):
        oe.spatial_predict(restored, data=data[["id", "x", "z"]].fillna(0))


@pytest.mark.parametrize(
    "bad",
    [
        "partial",
        "extra",
        "duplicate",
        "unknown",
        "missing_x",
        "missing_id",
        "inf",
        "boolean",
        "strings",
        "unsafe_integer",
    ],
)
def test_closed_graph_query_refuses_incompatible_inputs(bad):
    data, result, _ = _case("sar")
    query = data[["id", "x", "z"]].copy()
    if bad == "partial":
        query = query.iloc[:-1]
    elif bad == "extra":
        query = pd.concat([query, query.iloc[:1]])
    elif bad == "duplicate":
        query.loc[1, "id"] = query.loc[0, "id"]
    elif bad == "unknown":
        query.loc[0, "id"] = "new-unit"
    elif bad == "missing_x":
        query.loc[0, "x"] = np.nan
    elif bad == "missing_id":
        query.loc[0, "id"] = None
    elif bad == "inf":
        query.loc[0, "x"] = np.inf
    elif bad == "boolean":
        query["x"] = True
    elif bad == "strings":
        query["x"] = "1"
    elif bad == "unsafe_integer":
        query["x"] = pd.Series([2**53 + 1] * 24, dtype="int64")
    with pytest.raises(AnalysisError):
        oe.spatial_predict(result, data=query)


@pytest.mark.parametrize(
    "bad",
    [
        "cov_shape",
        "cov_asymmetric",
        "cov_indefinite",
        "terms",
        "main_graph_hash",
        "rho",
        "innovation_variance",
        "unconverged",
        "distribution",
        "missing_effective_graph",
    ],
)
def test_saved_state_refuses_incompatible_law_covariance_and_integrity(bad):
    data, result, _ = _case("sar")
    if bad == "cov_shape":
        result.covariance_matrix = [[1.0]]
    elif bad == "cov_asymmetric":
        result.covariance_matrix[0][1] += 0.5
    elif bad == "cov_indefinite":
        result.covariance_matrix[0][0] = -1
    elif bad == "terms":
        result.coefficients[0].term = "wrong-role"
    elif bad == "main_graph_hash":
        result.provenance["spatial_weight_hash"] = "wrong"
    elif bad == "rho":
        next(c for c in result.coefficients if c.term == "rho").estimate = 1.0
    elif bad == "innovation_variance":
        result.extra["innovation_variance"] *= 2
    elif bad == "unconverged":
        result.provenance["optimizer"]["converged"] = False
    elif bad == "distribution":
        result.inference["distribution"] = "t"
    elif bad == "missing_effective_graph":
        result.extra.pop("spatial_weights")
    with pytest.raises(AnalysisError):
        oe.spatial_predict(result, data=data[["id", "x", "z"]])


def test_sac_separate_error_graph_integrity_and_lambda_domain_are_validated():
    data, result, _ = _case("sac")
    assert (
        result.extra["error_weight_summary"]["sha256"] != result.provenance["spatial_weight_hash"]
    )
    result.extra["error_weight_summary"]["sha256"] = "altered"
    with pytest.raises(AnalysisError, match="SAC error graph"):
        oe.spatial_predict(result, data=data[["id", "x", "z"]])
    _, result, _ = _case("sac")
    next(c for c in result.coefficients if c.term == "lambda").estimate = 1.0
    with pytest.raises(AnalysisError, match="lambda"):
        oe.spatial_predict(result, data=data[["id", "x", "z"]])


@pytest.mark.parametrize(
    "option,value",
    [
        ("alpha", True),
        ("alpha", 0),
        ("alpha", np.nan),
        ("max_n", True),
        ("max_n", 513),
        ("max_n", 4),
        ("max_work", True),
        ("max_work", 0),
        ("max_work", 1),
        ("max_work", 10**10),
    ],
)
def test_operation_ceiling_and_option_guards_before_solve(option, value, monkeypatch):
    data, result, _ = _case("sar")

    def fail(*_args, **_kwargs):
        raise AssertionError("Solve preceded admission")

    monkeypatch.setattr(torch.linalg, "solve", fail)
    with pytest.raises(AnalysisError):
        oe.spatial_predict(result, data=data[["id", "x", "z"]], **{option: value})


def test_configured_workspace_guard_and_numerical_default_isolation():
    data, result, _ = _case("sar")
    # Make the declared saved state large enough to breach one MiB without
    # changing any numerical inference or secretly allocating dense outputs.
    result.extra["audit_note"] = "a" * 100000
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.spatial_predict(result, data=data[["id", "x", "z"]])
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            out = oe.spatial_predict(result, data=data[["id", "x", "z"]])
        assert out.attrs["precision"] == "float64" and out.attrs["device"] == "cpu"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(before)


def test_refuses_other_results_streaming_and_unsupported_observation_band():
    data, result, _ = _case("sar")
    from openecon.dataset import Dataset

    with pytest.raises(AnalysisError):
        oe.spatial_predict(result, data=Dataset.from_frame(data))
    with pytest.raises(AnalysisError):
        oe.spatial_predict(result, data=data.to_dict("list"))
    with pytest.raises(AnalysisError):
        oe.spatial_predict(None, data=data)
    with pytest.raises(TypeError):
        oe.spatial_predict(result, data=data, interval="prediction")
    with pytest.raises(TypeError):
        oe.spatial_predict(result, data=data, spatial_weights=result.extra["spatial_weights"])
    other = oe.ols(data=data, y="y", x=["x", "z"])
    with pytest.raises(AnalysisError):
        oe.spatial_predict(other, data=data)


def test_full_source_identity_changes_for_metadata_but_mean_is_preserved():
    data, result, _ = _case("sem")
    original = oe.spatial_predict(result, data=data[["id", "x", "z"]])
    result.extra["audit_annotation"] = "second saved artifact"
    changed = oe.spatial_predict(result, data=data[["id", "x", "z"]])
    assert original.attrs["source_result_sha256"] != changed.attrs["source_result_sha256"]
    assert_allclose(
        original["means"][["mean", "std_error"]], changed["means"][["mean", "std_error"]], atol=0
    )


def test_sdm_no_intercept_unnormalized_asymmetric_graph_retains_isolate_and_recomputes_WX():
    data, original, _ = _case("sdm")
    payload = original.extra["spatial_weights"]
    keys = payload["keys"]
    weights = oe.spatial_weights(
        keys,
        [
            (keys[i], keys[j], 3.3 * v)
            for i, j, v in zip(payload["rows"], payload["cols"], payload["values"])
            if i != 0
        ],
        normalization="none",
        isolates="zero",
    )
    result = oe.sdm(data, "y", ["x", "z"], key="id", spatial_weights=weights, intercept=False)
    query = data[["id", "x", "z"]].copy()
    query["x"] *= 1.7
    query["z"] += 0.8
    out = oe.spatial_predict(result, data=query)
    assert "Intercept" not in out.attrs["parameter_order"]
    expected, gradient, covariance = _oracle(result, query, weights.dense().numpy())
    assert_allclose(out["means"]["mean"], expected, atol=8e-13)
    assert_allclose(out["jacobian"], gradient, atol=8e-13)
    assert_allclose(out["mean_covariance"], covariance, atol=8e-13)
    beta = {c.term: c.estimate for c in result.coefficients}
    assert out["means"]["mean"].iloc[0] == pytest.approx(
        beta["x"] * query.x.iloc[0] + beta["z"] * query.z.iloc[0], abs=3e-13
    )
    assert out["jacobian"]["parameter:rho"].iloc[0] == 0
    assert out.attrs["workspace_observed"]["full_mean_covariance_bytes"] == 24**2 * 8


@pytest.mark.parametrize(
    "change",
    [
        "short_positions",
        "duplicate_positions",
        "out_of_bounds",
        "original_count",
        "huge_variance",
        "nonfinite_metadata",
    ],
)
def test_restored_sample_geometry_and_json_identity_fail_with_analysis_error(change):
    data, result, _ = _case("sar")
    if change == "short_positions":
        result.sample_positions = result.sample_positions[:-1]
    elif change == "duplicate_positions":
        result.sample_positions[1] = result.sample_positions[0]
    elif change == "out_of_bounds":
        result.sample_positions[0] = result.nobs_original
    elif change == "original_count":
        result.nobs_original = 1
    elif change == "huge_variance":
        result.extra["innovation_variance"] = 10**1000
    else:
        result.extra["bad_number"] = float("inf")
    with pytest.raises(AnalysisError):
        oe.spatial_predict(result, data=data[["id", "x", "z"]])


@pytest.mark.parametrize(
    "change",
    [
        "provenance_model",
        "provenance_family",
        "design_terms",
        "parameter_count",
        "covariance_identity",
        "nested_position",
    ],
)
def test_saved_fitted_identity_and_nonscalar_positions_are_normalized_errors(change):
    data, result, _ = _case("sar")
    if change == "provenance_model":
        result.provenance["estimator"] = "sem"
    elif change == "provenance_family":
        result.provenance["family"] = "panel"
    elif change == "design_terms":
        result.provenance["design_terms"][0] = "fake"
    elif change == "parameter_count":
        result.inference["n_parameters"] = 999
    elif change == "covariance_identity":
        result.inference["covariance"] = "HC0"
    else:
        result.sample_positions[0] = [0]
    with pytest.raises(AnalysisError):
        oe.spatial_predict(result, data=data[["id", "x", "z"]])


def test_positive_tiny_query_uncertainty_is_refused_without_false_zero_SE():
    data, result, _ = _case("sem")
    # Keep the same legitimate Gaussian-ML fixed-network law, dropping only
    # the intercept role together with its fitted covariance/identity records.
    # The tiny nonzero query has a nonzero parameter gradient, hence positive
    # variance for this PD fitted covariance, although squaring underflows.
    result.spec.intercept = False
    retained = [i for i, c in enumerate(result.coefficients) if c.term != "Intercept"]
    result.coefficients = [result.coefficients[i] for i in retained]
    result.covariance_matrix = np.array(result.covariance_matrix)[
        np.ix_(retained, retained)
    ].tolist()
    result.provenance["design_terms"] = [c.term for c in result.coefficients]
    result.inference["n_parameters"] = len(result.coefficients)
    query = data[["id", "x", "z"]].copy()
    query["x"], query["z"] = 1e-180, 0.0
    assert float(np.array(result.covariance_matrix)[0, 0]) > 0
    assert query.x.iloc[0] != 0 and query.x.iloc[0] ** 2 == 0
    with pytest.raises(AnalysisError, match="Nonzero projected mean variance"):
        oe.spatial_predict(result, data=query)
    query["x"] = 0.0
    zero = oe.spatial_predict(result, data=query)
    assert (zero["means"].std_error == 0).all()
    assert (zero["mean_covariance"] == 0).all().all()


def test_psd_covariance_genuine_null_gradient_remains_exactly_zero():
    data, result, _ = _case("sem")
    # A PSD law can genuinely annihilate a nonzero gradient. Keep this
    # distinct from float64 underflow of a positive-definite law.
    p = len(result.coefficients)
    result.covariance_matrix = np.zeros((p, p)).tolist()
    query = data[["id", "x", "z"]]
    out = oe.spatial_predict(result, data=query)
    assert (out["means"].std_error == 0).all()
    assert (out["mean_covariance"] == 0).all().all()


def test_admitted_encoded_row_identity_sizes_are_reserved_before_output():
    data, result, _ = _case("sar")
    query = data[["id", "x", "z"]].copy()
    query.index = ["i" * 3000] * len(query)
    out = oe.spatial_predict(result, data=query)
    encoded_bytes = sum(len(value.encode()) for value in out["sample"].index_json)
    named = out.attrs["resource_plan"]["buffers"][
        "positions, keys and admitted encoded row identities"
    ]
    assert named > 2 * encoded_bytes
