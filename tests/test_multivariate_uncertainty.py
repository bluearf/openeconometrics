"""Independent principal-factor bootstrap numerical and refusal contracts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import uncertainty as u
from openecon.resources import use_workspace_budget


def data(distribution="normal", n=240, seed=163):
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        latent = rng.normal(size=n)
        errors = rng.normal(size=(n, 4))
    else:
        latent = (rng.lognormal(sigma=.5, size=n) - np.exp(.125)) / np.sqrt((np.exp(.25)-1)*np.exp(.25))
        errors = rng.standard_t(7, size=(n, 4)) / np.sqrt(7/5)
    loading = np.array([-.75, .7, .8, .65])
    values = latent[:, None] * loading + errors * np.sqrt(1 - loading**2)
    return pd.DataFrame(values * [2, 3, 1, 4] + [5, -2, 12, .5], columns=list("abcd"))


def independent_parameters(values, anchor):
    # This oracle never calls the production factor, moment or extraction code.
    shifted = values - values[0]
    centred = shifted - shifted.mean(0)
    covariance = centred.T @ centred / (len(values) - 1)
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    np.fill_diagonal(correlation, 1)
    smc = 1 - 1 / np.diag(np.linalg.inv(correlation))
    reduced = correlation.copy()
    np.fill_diagonal(reduced, smc)
    eigenvalues, vectors = np.linalg.eigh(reduced)
    loading = vectors[:, -1] * np.sqrt(eigenvalues[-1])
    loading *= 1 if loading[anchor] > 0 else -1
    return np.r_[loading, 1 - loading**2]


def independent_bootstrap(values, *, replications, seed, anchor):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Sharing the specified RNG stream isolates the independent fitting oracle.
    return np.array([independent_parameters(values[torch.randint(len(values), (len(values),), generator=generator).numpy()], anchor)
                     for _ in range(replications)])


@pytest.mark.parametrize("distribution", ["normal", "skewed_heavy_tail"])
def test_two_iid_dgps_full_independent_covariance_and_percentiles(distribution):
    frame = data(distribution)
    result = u.factor_bootstrap(frame, list("abcd"), replications=199, seed=701, anchor="a")
    reference = independent_bootstrap(frame.to_numpy(), replications=199, seed=701, anchor=0)
    point = independent_parameters(frame.to_numpy(), 0)
    np.testing.assert_allclose(result["replicates"], reference, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(result["estimates"].estimate, point, atol=2e-12, rtol=2e-12)
    expected_covariance = np.cov(reference, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], expected_covariance, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(expected_covariance.diagonal()), atol=2e-12)
    intervals = np.quantile(reference, [.025, .975], axis=0, method="linear")
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]], intervals.T, atol=2e-12)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias, reference.mean(0) - point, atol=2e-12)
    assert (result["replicates"]["loading:a"] > 0).all()
    assert result.attrs["successful_replications"] == 199
    assert not result.attrs["p_values_available"] and not result.attrs["inference_df_available"]
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert result.attrs["inference_target"] == "fixed one-factor principal-factor estimator functional"


def restore(result):
    payload = json.loads(json.dumps({"attrs": result.attrs, "tables": {
        name: json.loads(table.to_json(orient="split", double_precision=15))
        for name, table in result.items()}}, allow_nan=False))
    return TableSet({name: pd.DataFrame(**table) for name, table in payload["tables"].items()}, **payload["attrs"])


def test_seed_reproducibility_full_saved_result_and_raw_sample_replay():
    frame = data(n=120)
    one = u.factor_bootstrap(frame, list("abcd"), replications=59, confidence=.9, seed=43)
    two = u.factor_bootstrap(frame, list("abcd"), replications=59, confidence=.9, seed=43)
    other = u.factor_bootstrap(frame, list("abcd"), replications=59, confidence=.9, seed=44)
    for name in one:
        pd.testing.assert_frame_equal(one[name], two[name])
    assert not one["replicates"].equals(other["replicates"])
    saved = restore(one)
    assert saved.attrs == one.attrs
    for name in one:
        np.testing.assert_allclose(saved[name].to_numpy(dtype="float64"), one[name].to_numpy(dtype="float64"), atol=2e-14, rtol=2e-14, equal_nan=True)
        assert list(saved[name].columns) == list(one[name].columns)
        assert list(saved[name].index) == list(one[name].index)
    replay = u.factor_bootstrap(saved["sample"], saved.attrs["variables"],
                               replications=59, confidence=.9, seed=saved.attrs["seed"], anchor=saved.attrs["sign_anchor"])
    for name in ["estimates", "covariance", "replicates", "replicate_diagnostics", "point_loadings", "point_uniqueness"]:
        np.testing.assert_allclose(replay[name].to_numpy(dtype="float64"), one[name].to_numpy(dtype="float64"), atol=2e-12, rtol=2e-12, equal_nan=True)
    assert "df" in saved["estimates"] and saved["estimates"].df.isna().all()


def test_missing_rows_original_indices_and_input_preservation():
    frame = data(n=120)
    frame.index = ["same", "same", *[f"row-{i}" for i in range(118)]]
    frame.loc["row-7", "b"] = np.nan
    frame.loc["row-19", "d"] = np.nan
    frame["unused"] = "text"
    before = frame.copy(deep=True)
    result = u.factor_bootstrap(frame, list("abcd"), replications=39, seed=12)
    clean = frame[list("abcd")].dropna()
    expected = u.factor_bootstrap(clean, list("abcd"), replications=39, seed=12, anchor=result.attrs["sign_anchor"])
    for name in ["estimates", "covariance", "replicates", "replicate_diagnostics"]:
        pd.testing.assert_frame_equal(result[name], expected[name])
    pd.testing.assert_frame_equal(frame, before)
    assert result.attrs["n_missing"] == 2 and result.attrs["n"] == 118
    assert result.attrs["original_index"] == list(clean.index)
    assert result.attrs["sample_positions"] == np.flatnonzero(frame[list("abcd")].notna().all(axis=1)).tolist()
    assert list(result["sample"].index) == result.attrs["sample_positions"]
    with pytest.raises(AnalysisError, match="missing"):
        u.factor_bootstrap(frame, list("abcd"), missing="raise")


def test_mapping_and_records_match_without_materializing_unused_columns():
    frame = data(n=70)
    mapping = frame.to_dict(orient="list")
    mapping["unused"] = object()
    direct = u.factor_bootstrap(frame, list("abcd"), replications=39, seed=5)
    mapped = u.factor_bootstrap(mapping, list("abcd"), replications=39, seed=5)
    records = u.factor_bootstrap(frame.to_dict(orient="records"), list("abcd"), replications=39, seed=5)
    pd.testing.assert_frame_equal(mapped["replicates"], direct["replicates"])
    pd.testing.assert_frame_equal(records["replicates"], direct["replicates"])


def test_missing_index_labels_are_json_safe_and_positions_remain_exact():
    frame = data(n=70)
    frame.index = [np.nan, None, pd.NA, *range(67)]
    result = u.factor_bootstrap(frame, list("abcd"), replications=39, seed=13)
    assert result.attrs["original_index"][:3] == [None, None, None]
    assert result.attrs["sample_positions"] == list(range(70))
    json.dumps(result.attrs, allow_nan=False)


def test_every_failed_replication_is_reported_no_successful_subset(monkeypatch):
    original = u._parameters
    calls = []

    def fail_selected(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) in (3, 9):
            raise AnalysisError("singular_matrix", "declared test refit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(u, "_parameters", fail_selected)
    with pytest.raises(AnalysisError, match="2 of 39") as caught:
        u.factor_bootstrap(data(n=70), list("abcd"), replications=39)
    error = caught.value
    assert error.code == "bootstrap_failure"
    assert len(calls) == 40
    assert error.replications_attempted == 39 and error.successful_replications == 37
    assert [failure["replication"] for failure in error.failures] == [2, 8]
    assert all(failure["code"] == "singular_matrix" for failure in error.failures)


@pytest.mark.parametrize("options", [{"method": "ml"}, {"method": "pcf"}, {"replications": True},
                                    {"replications": 18}, {"replications": 2000}, {"confidence": 1},
                                    {"confidence": 0}, {"confidence": .999}, {"seed": -1}, {"seed": True},
                                    {"seed": 2**63}, {"anchor": "absent"}, {"missing": "pairwise"}])
def test_invalid_or_unsupported_method_options(options):
    with pytest.raises(AnalysisError):
        u.factor_bootstrap(data(), list("abcd"), **options)


@pytest.mark.parametrize("options", [{"weights": "w"}, {"cluster": "group"}, {"rotate": "varimax"}, {"factors": 2}, {"device": "mps"}])
def test_no_implicit_weight_cluster_rotation_device_or_factor_count(options):
    with pytest.raises(TypeError):
        u.factor_bootstrap(data(), list("abcd"), **options)


def test_dataset_refused_without_reading_source():
    calls = []

    def batches():
        calls.append(True)
        yield data()

    stream = Dataset.from_batches(batches, columns=list("abcd"), row_count=240)
    with pytest.raises(AnalysisError, match="resident"):
        u.factor_bootstrap(stream, list("abcd"))
    assert not calls


def test_geometry_and_workspace_admission_precedes_conversion(monkeypatch):
    def forbidden(*args):
        raise AssertionError("input was converted before admission")

    monkeypatch.setattr(u, "_selected_input", forbidden)
    with pytest.raises(AnalysisError, match="10000"):
        u.factor_bootstrap({name: range(10001) for name in "abcd"}, list("abcd"))
    names = [f"v{i}" for i in range(17)]
    with pytest.raises(AnalysisError, match="16 variables"):
        u.factor_bootstrap({name: range(50) for name in names}, names)
    names = names[:16]
    with pytest.raises(AnalysisError, match="work"):
        u.factor_bootstrap({name: range(10000) for name in names}, names, replications=1999)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        u.factor_bootstrap({name: range(4000) for name in "abcd"}, list("abcd"))


def test_summary_non_cpu_inputs_and_tiny_sample_refused():
    from openecon.econometrics.multivariate.summary import prepare

    summary = prepare([[1, .6, .5], [.6, 1, .5], [.5, .5, 1]], n=100,
                      columns=list("abc"), matrix="correlation")
    with pytest.raises(AnalysisError, match="summary"):
        u.factor_bootstrap(summary, list("abc"))
    with pytest.raises(AnalysisError, match="CPU"):
        u.factor_bootstrap({name: torch.empty(40, device="meta") for name in "abcd"}, list("abcd"))
    with pytest.raises(AnalysisError, match="complete iid rows"):
        u.factor_bootstrap(data(n=19), list("abcd"))


@pytest.mark.parametrize("bad", [np.inf, 1e151, 1j])
def test_nonfinite_extreme_complex_sample_refused(bad):
    frame = data(n=60)
    if isinstance(bad, complex):
        frame["a"] = frame["a"].astype(complex)
    frame.loc[0, "a"] = bad
    with pytest.raises(AnalysisError):
        u.factor_bootstrap(frame, list("abcd"))


def test_singular_and_zero_anchor_geometry_refused():
    frame = data(n=60)
    frame["a"] = frame.b
    with pytest.raises(AnalysisError, match="singular"):
        u.factor_bootstrap(frame, list("abcd"))
    # Orthogonal centred columns with a positively correlated first block leave
    # the d loading exactly zero in the leading principal factor.
    grid = np.arange(40.)
    basis, _ = np.linalg.qr(np.column_stack((np.ones(40), grid, grid**2, grid**3, grid**4)))
    frame = pd.DataFrame({"a": basis[:, 1] + .2*basis[:, 2], "b": basis[:, 1] + .3*basis[:, 3],
                          "c": basis[:, 1] + .4*basis[:, 2] + .1*basis[:, 3], "d": basis[:, 4]})
    with pytest.raises(AnalysisError, match="sign anchor"):
        u.factor_bootstrap(frame, list("abcd"), anchor="d")


def test_explicit_weak_nonseparated_orientation_and_heywood_gates(monkeypatch):
    original = u.factor

    def modified_fit(*args, **kwargs):
        fitted = original(*args, **kwargs)
        fitted["eigenvalues"].iloc[1, fitted["eigenvalues"].columns.get_loc("eigenvalue")] = fitted["eigenvalues"].iloc[0].eigenvalue
        return fitted

    monkeypatch.setattr(u, "factor", modified_fit)
    with pytest.raises(AnalysisError, match="not separated"):
        u.factor_bootstrap(data(), list("abcd"))

    def heywood_fit(*args, **kwargs):
        fitted = original(*args, **kwargs)
        fitted["uniqueness"].iloc[0, 0] = 0
        return fitted

    monkeypatch.setattr(u, "factor", heywood_fit)
    with pytest.raises(AnalysisError, match="nonpositive"):
        u.factor_bootstrap(data(), list("abcd"))
