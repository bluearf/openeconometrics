"""Independent finite-sample numerical oracles and domain/persistence checks."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest import finite_sample as native
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def frame(n=43, seed=207):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) @ np.array([[1, .6], [0, .8]])
    w = rng.uniform(.5, 2, n)
    y = .2 + x @ [1., -.3] + rng.normal(size=n) / np.sqrt(w)
    return pd.DataFrame({"y": y, "x": x[:, 0], "z": x[:, 1], "w": w})


def fit(weighted=False):
    data = frame()
    options = {"weights": "w", "weight_type": "aweight"} if weighted else {}
    return oe.ols(data=data, y="y", x=["x", "z"], covariance="nonrobust", **options)


def joint(estimates=(0., 0.), covariance=((1., .6), (.6, 1.)), **options):
    return oe.simultaneous_t_ci(estimates, covariance, df=options.pop("df", 5),
        family_description="Predeclared compatible fixed-design coefficient family",
        pivot_description="Known correlation, independent common chi-square residual scale",
        **options)


@pytest.mark.parametrize("df", [1., 2., 5., 137., 1_000_000.])
def test_local_gamma_sampler_against_independent_chi_square_distribution(df):
    generator = torch.Generator().manual_seed(208001)
    sample, attempts = native._chi_square(50_000, df, generator)
    # KS gate declared before observing samples; tests the full distribution.
    assert stats.kstest(sample.numpy(), stats.chi2(df).cdf).statistic < .012
    assert 1 <= attempts <= 128
    assert np.mean(sample.numpy()) == pytest.approx(df, abs=8*np.sqrt(2*df/50_000))


@pytest.mark.parametrize("df", [1, 2, 7, 137])
def test_single_t_interval_vs_independent_student_t(df):
    result = joint([3.], [[4.]], df=df, draws=100_000, seed=208002)
    expected = stats.t.isf(.025, df)
    # Use CDF-scale comparison because heavy tails amplify quantile uncertainty.
    assert abs(stats.t.sf(result.attrs["critical_value"], df)*2-.05) < .004
    assert result.attrs["degrees_of_freedom"] == df
    assert result.ci_low[0] == pytest.approx(3-2*result.attrs["critical_value"])
    assert result.ci_high[0] == pytest.approx(3+2*result.attrs["critical_value"])
    assert abs(stats.t.sf(expected, df)*2-.05) < 1e-12


def test_joint_t_common_denominator_and_independent_normal_reference():
    result = joint(draws=100_000, seed=208003, df=2)
    rng = np.random.default_rng(208004)
    noise = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], size=100_000)
    pivots = noise / np.sqrt(rng.chisquare(2, 100_000)/2)[:, None]
    critical = result.attrs["critical_value"]
    assert abs(np.mean(np.max(np.abs(pivots), axis=1)>critical)-.05)<.005
    # Independent marginal denominators generate the wrong joint law.
    wrong = noise / np.sqrt(rng.chisquare(2, (100_000, 2))/2)
    assert abs(np.mean(np.max(np.abs(wrong), axis=1)>critical)-.05)>.007


def test_t_label_permutation_psd_and_rng_isolation():
    state = torch.random.get_rng_state().clone()
    result = joint([2., 3.], [[4., 1.2], [1.2, 1.]], labels=["z", "a"], seed=208005)
    reversed_result = joint([3., 2.], [[1., 1.2], [1.2, 4.]], labels=["a", "z"], seed=208005)
    pd.testing.assert_frame_equal(result.iloc[::-1].reset_index(drop=True), reversed_result)
    assert torch.equal(state, torch.random.get_rng_state())
    singular = joint(covariance=[[1., 1.], [1., 1.]], df=5)
    assert singular.attrs["smallest_correlation_eigenvalue"] == 0
    assert singular.ci_high[0] == pytest.approx(singular.ci_high[1])
    json.loads(json.dumps(result.attrs, allow_nan=False))
    assert "\\begin{tabular}" in result.to_latex()


def test_joint_t_translation_and_positive_scale_equivariance():
    estimates = np.array([.3, -.2])
    covariance = np.array([[.04, .01], [.01, .09]])
    scale = np.array([3., .4])
    offset = np.array([5., -8.])
    original = joint(estimates, covariance, labels=["x", "z"], seed=208006)
    moved = joint(estimates*scale+offset, covariance*np.outer(scale, scale),
                  labels=["x", "z"], seed=208006)
    np.testing.assert_allclose(moved.ci_low, original.ci_low*scale+offset, atol=2e-14)
    np.testing.assert_allclose(moved.ci_high, original.ci_high*scale+offset, atol=2e-14)
    assert moved.attrs["critical_value"] == pytest.approx(original.attrs["critical_value"], abs=2e-14)


@pytest.mark.parametrize("df", [True, 0, -1, float("nan"), float("inf"), 1_000_001, [5, 7]])
def test_joint_t_refuses_wrong_df(df):
    with pytest.raises(AnalysisError, match="df"):
        joint(df=df)


@pytest.mark.parametrize("cov", [[[1., 2.], [2., 1.]], [[1., .2], [.3, 1.]], [[0., 0.], [0., 1.]]])
def test_joint_t_rejects_invalid_covariance(cov):
    with pytest.raises(AnalysisError) as error:
        joint(covariance=cov)
    assert error.value.code == "invalid_covariance"


def test_joint_t_preallocation_budget_and_missing_pivot(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("No numerical simulation should precede admission")
    monkeypatch.setattr(native, "_joint_t", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as e:
        joint(draws=200_000)
    assert e.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as e:
        joint([0.]*100, np.eye(100), draws=20_000)
    assert e.value.code == "work_budget"
    with pytest.raises(AnalysisError) as e:
        oe.simultaneous_t_ci([0.], [[1.]], df=5, pivot_description="", family_description="family")
    assert e.value.code == "missing_pivot_design"


@pytest.mark.parametrize("weighted", [False, True])
def test_ols_pivotal_stepdown_matches_independent_refit_and_persistence(weighted):
    data = frame()
    model = fit(weighted)
    restored = ResultBundle.model_validate_json(model.model_dump_json())
    before = restored.model_dump_json()
    result = oe.ols_stepdown(restored, error_model="known_precision_gaussian" if weighted else "iid_gaussian",
                             terms=["z", "x"], null_values=[-.3, 1.], draws=40_000, seed=207001)
    x = np.column_stack([np.ones(len(data)), data.x, data.z])
    y = data.y.to_numpy()
    weights = data.w.to_numpy() if weighted else np.ones(len(data))
    xw, yw = np.sqrt(weights)[:, None]*x, np.sqrt(weights)*y
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    residual = yw-xw@beta
    covariance = residual@residual/(len(data)-3)*np.linalg.inv(xw.T@xw)
    expected = (beta[[2, 1]]-[-.3, 1.])/np.sqrt(np.diag(covariance)[[2, 1]])
    np.testing.assert_allclose(result.statistic, expected, atol=2e-12)
    np.testing.assert_allclose(result.p_value, stats.t.sf(np.abs(expected), len(data)-3)*2, atol=2e-12)
    np.testing.assert_allclose(result.attrs["joint_covariance"], covariance[np.ix_([2, 1], [2, 1])], atol=2e-12)
    assert restored.model_dump_json() == before
    assert result.attrs["sample_hash"] == model.provenance["sample_hash"]
    assert result.attrs["null_design_verified"] is True
    assert result.attrs["failures"] == 0 and result.attrs["degrees_of_freedom"] == 40
    json.loads(json.dumps(result.attrs, allow_nan=False))


def test_ols_family_permutation_sample_drop_and_invalid_stored_state():
    data = frame()
    data.loc[4, "z"] = np.nan
    model = oe.ols(data=data, y="y", x=["x", "z"], missing="drop", covariance="nonrobust")
    a = oe.ols_stepdown(model, error_model="iid_gaussian", terms=["x", "z"], seed=207002)
    b = oe.ols_stepdown(model, error_model="iid_gaussian", terms=["z", "x"], seed=207002)
    np.testing.assert_array_equal(a.adjusted_p_value, b.adjusted_p_value.iloc[::-1])
    assert a.attrs["dropped_rows"] == 1 and a.attrs["nobs"] == 42
    corrupted = model.model_copy(deep=True)
    corrupted.sample_positions[1] = corrupted.sample_positions[0]
    with pytest.raises(AnalysisError) as e:
        oe.ols_stepdown(corrupted, error_model="iid_gaussian")
    assert e.value.code == "invalid_result_state"
    corrupted = model.model_copy(deep=True)
    corrupted.covariance_matrix[0][0] *= 2
    with pytest.raises(AnalysisError) as e:
        oe.ols_stepdown(corrupted, error_model="iid_gaussian")
    assert e.value.code == "invalid_result_state"


@pytest.mark.parametrize("tail", ["less", "greater", "two-sided"])
def test_ols_single_coefficient_tail_and_mc_count_identity(tail):
    model = fit()
    result = oe.ols_stepdown(model, terms=["x"], error_model="iid_gaussian", tail=tail, draws=100_000, seed=207003)
    t = float(result.statistic[0])
    expected = stats.t.cdf(t, 40) if tail == "less" else stats.t.sf(t, 40)
    if tail == "two-sided":
        expected = 2*stats.t.sf(abs(t), 40)
    assert result.p_value[0] == pytest.approx(expected, abs=2e-12)
    assert abs(result.adjusted_p_value[0]-expected)<.004
    assert result.adjusted_p_value[0] == pytest.approx((result.adjusted_exceedances[0]+1)/100_001)


@pytest.mark.parametrize("options", [{"covariance": "HC3"}, {"covariance": "cluster", "cluster": "g"},
                                      {"weights": "w", "weight_type": "pweight"}])
def test_ols_rejects_robust_cluster_and_wrong_weight_semantics(options):
    data = frame().assign(g=np.arange(43)//4)
    model = oe.ols(data=data, y="y", x=["x", "z"], **options)
    with pytest.raises(AnalysisError):
        oe.ols_stepdown(model, error_model="iid_gaussian")
    with pytest.raises(AnalysisError):
        oe.ols_stepdown(fit(True), error_model="iid_gaussian")


def test_hotelling_independent_sample_region_and_missing_alignment():
    data = frame()
    data.loc[3, "z"] = np.nan
    before = data.copy(deep=True)
    result = oe.hotelling_region(data, ["x", "z"], sampling_model="iid_multivariate_normal", missing="drop")
    array = data[["x", "z"]].dropna().to_numpy()
    n, p = array.shape
    covariance = np.cov(array, rowvar=False)/n
    radius = p*(n-1)/(n-p)*stats.f.isf(.05, p, n-p)
    np.testing.assert_allclose(result.estimate, array.mean(0), atol=1e-14)
    np.testing.assert_allclose(result.attrs["joint_covariance"], covariance, atol=1e-14)
    assert result.attrs["region_radius_squared"] == pytest.approx(radius, rel=1e-12)
    scale = np.array(result.attrs["region_scale"])
    np.testing.assert_allclose(np.array(result.attrs["normalized_region_precision"])/scale[:,None]/scale[None,:], np.linalg.inv(covariance), rtol=1e-12)
    np.testing.assert_allclose(result.projected_ci_low, array.mean(0)-np.sqrt(radius*np.diag(covariance)), rtol=1e-12)
    assert result.attrs["sample_positions"] == [i for i in range(43) if i!=3]
    assert result.attrs["dropped_rows"] == 1
    pd.testing.assert_frame_equal(data, before)
    json.loads(json.dumps(result.attrs, allow_nan=False))
    assert "\\begin{tabular}" in result.to_latex()


def test_hotelling_one_dimension_equals_student_t_and_affine_covariance():
    data = frame()
    result = oe.hotelling_region(data, ["x"], sampling_model="iid_multivariate_normal")
    assert result.attrs["region_radius_squared"] == pytest.approx(stats.t.isf(.025, 42)**2, rel=1e-12)
    original = oe.hotelling_region(data, ["x", "z"], sampling_model="iid_multivariate_normal")
    moved = oe.hotelling_region(data.assign(x=data.x*3+5,z=data.z*.4-8), ["x", "z"], sampling_model="iid_multivariate_normal")
    np.testing.assert_allclose(moved.estimate, original.estimate*[3,.4]+[5,-8], atol=2e-14)
    np.testing.assert_allclose(moved.projected_ci_high, original.projected_ci_high*[3,.4]+[5,-8], atol=2e-14)
    covariance = original.attrs["joint_covariance"]
    np.testing.assert_allclose(moved.attrs["joint_covariance"], np.array(covariance)*np.outer([3,.4],[3,.4]), atol=2e-14)


@pytest.mark.parametrize("change,code", [
    (lambda d:d.assign(z=d.x*2), "singular_covariance"),
    (lambda d:d.iloc[:2], "insufficient_observations"),
    (lambda d:d.assign(x=np.nan), "missing_values"),
    (lambda d:d.assign(x=np.inf), "nonfinite_values"),
    (lambda d:d.assign(x=True), "invalid_values"),
])
def test_hotelling_undefined_samples_fail(change, code):
    with pytest.raises(AnalysisError) as e:
        oe.hotelling_region(change(frame()), ["x", "z"], sampling_model="iid_multivariate_normal")
    assert e.value.code == code


def test_hotelling_resource_admission_and_unsupported_design():
    with use_workspace_budget(1), pytest.raises(AnalysisError) as e:
        oe.hotelling_region(frame(n=20_000), ["x", "z"], sampling_model="iid_multivariate_normal")
    assert e.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as e:
        oe.hotelling_region(frame(), ["x", "z"], sampling_model="cluster_robust")
    assert e.value.code == "unsupported_sampling_model"
