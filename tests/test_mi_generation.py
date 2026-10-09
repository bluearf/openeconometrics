"""Independent conjugacy/moment oracles, preservation, replay and admission failures."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import MIResult, admit, make_result
from openecon.econometrics.mi.generation import (
    _draw_niw, _niw_posterior, _prior, _regression_draw, _regression_posterior,
    mi_monotone, mi_mvn,
)
from openecon.resources import use_workspace_budget


@pytest.fixture(scope="module", autouse=True)
def small_factors():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prior(p=2):
    return {"mean": [0.] * p, "kappa": .2, "df": p + 3.,
            "scale": torch.eye(p, dtype=torch.float64).tolist()}


def panel():
    return pd.DataFrame({"x": [0., 1., 2., 3., 4., 5., 6., 7.],
                         "y": [1., 3., 2., 5., 4., 8., None, None]},
                        index=pd.Index(["a", "b", "c", "d", "e", "f", "f", "a"], name="unit"))


def test_admission_nullable_values_and_identity():
    frame = panel()
    frame["x"] = pd.array(frame["x"], dtype="Float64")
    before = frame.copy(deep=True)
    selected, values, missing, metadata = admit(frame, ["y", "x"], m=2)
    assert selected.index.equals(frame.index)
    assert values.dtype == torch.float64 and values.device.type == "cpu"
    assert missing.dtype == torch.bool and int(missing.sum()) == 2
    assert metadata["sample"]["positions"] == list(range(len(frame)))
    assert metadata["missing_counts"] == [2, 0]
    assert metadata["resource_plan"]["buffers"]["immutable result and JSON state"] > 0
    pd.testing.assert_frame_equal(before, frame)


@pytest.mark.parametrize("data", [panel(), {"x": [1., 2.], "y": [None, 4.]},
                                  [{"x": 1., "y": None}, {"x": 2., "y": 4.}]])
def test_resident_forms(data):
    frame, values, missing, metadata = admit(data, ["x", "y"])
    assert len(frame) == metadata["n"] and values.shape == missing.shape


def test_refuses_shape_work_and_workspace_before_numeric_allocation(monkeypatch):
    def allocation(*args, **kwargs):
        raise AssertionError("numeric allocation must not happen")
    monkeypatch.setattr(torch, "tensor", allocation)
    with pytest.raises(AnalysisError, match="10000"):
        admit(pd.DataFrame({"x": range(10001)}), ["x"])
    with pytest.raises(AnalysisError, match="m must"):
        admit(panel(), ["x", "y"], m=101)
    with pytest.raises(AnalysisError, match="max_work"):
        admit(panel(), ["x", "y"], max_work=1)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            admit(pd.DataFrame({"x": range(1000)}), ["x"])
        assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("frame,columns", [
    (pd.DataFrame({"x": [1., float("inf")]}), ["x"]),
    (pd.DataFrame({"x": [1., -float("inf")]}), ["x"]),
    (pd.DataFrame({"x": ["1", None]}), ["x"]),
    (pd.DataFrame({"x": [1 + 2j, None]}), ["x"]),
    (pd.DataFrame([[1., 2.]], columns=["x", "x"]), ["x"]),
    (panel(), ["x", "x"]),
    (panel(), "x"),
])
def test_invalid_panel_refusals(frame, columns):
    with pytest.raises(AnalysisError):
        admit(frame, columns)


def test_replay_source_and_weights_are_not_implicitly_materialized():
    class Replay:
        def collect(self):
            raise AssertionError("must not collect")
    with pytest.raises(AnalysisError, match="Dataset and replay"):
        admit(Replay(), ["x"])
    with pytest.raises(TypeError, match="weights"):
        mi_monotone(panel(), ["x", "y"], weights="x")


@pytest.mark.parametrize("invalid", [None, {}, {"mean": [0.], "kappa": 0., "df": 4., "scale": [[1.]]},
    {"mean": [0., 0.], "kappa": 1., "df": 1., "scale": [[1., 0.], [0., 1.]]},
    {"mean": [0., 0.], "kappa": 1., "df": 4., "scale": [[1., 2.], [2., 1.]]},
    {"mean": [0., 0.], "kappa": 1., "df": 4., "scale": [[1., .1], [0., 1.]]},
])
def test_explicit_proper_prior_required_before_admission(invalid, monkeypatch):
    monkeypatch.setattr("openecon.econometrics.mi.generation.admit", lambda *a, **k: pytest.fail("admitted invalid prior"))
    with pytest.raises(AnalysisError, match="NIW"):
        mi_mvn(panel(), ["x", "y"], prior=invalid)


@pytest.mark.parametrize("field,dtype", [("mean", torch.complex128), ("scale", torch.complex128),
                                        ("mean", torch.bool), ("scale", torch.bool)])
def test_prior_tensor_types_rejected_before_float_cast(field, dtype):
    options = prior()
    options[field] = torch.tensor(options[field], dtype=dtype)
    with pytest.raises(AnalysisError, match="real numeric"):
        mi_mvn(panel(), ["x", "y"], prior=options, m=1, burn=1, thin=1)


def test_samplers_ignore_ambient_torch_device_and_dtype():
    previous_device, previous_dtype = torch.get_default_device(), torch.get_default_dtype()
    p = prior()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        for result in (mi_mvn(panel(), ["x", "y"], prior=p, m=1, burn=1, thin=1),
                       mi_monotone(panel(), ["x", "y"], m=1)):
            assert result.metadata["engine"] == "torch_cpu_float64"
            assert result.dataset().iloc[-1, 1] != 0
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


def test_complete_data_conjugacy_scalar_exact_oracle():
    p = _prior({"mean": [0.], "kappa": 2., "df": 5., "scale": [[3.]]}, 1)
    posterior = _niw_posterior(torch.tensor([[1.], [2.], [4.]], dtype=torch.float64), p)
    # Hand-computed sums: n=3, sum=7, sumsq=21. Posterior mean=7/5,
    # scale=3+(21-49/3)+(2*3/5)*(7/3)^2=71/5.
    assert posterior["kappa"] == 5 and posterior["df"] == 8
    assert float(posterior["mean"][0]) == pytest.approx(7 / 5, abs=1e-14)
    assert float(posterior["scale"][0, 0]) == pytest.approx(71 / 5, abs=1e-14)


def test_niw_bartlett_sampling_moments_against_conjugate_theory():
    p = _prior({"mean": [1., -2.], "kappa": 3., "df": 12., "scale": [[3., 1.], [1., 2.]]}, 2)
    generator = torch.Generator().manual_seed(712)
    draws = [_draw_niw(p, generator) for _ in range(5000)]
    means = np.asarray([mu.tolist() for mu, _ in draws])
    covariances = np.asarray([sigma.tolist() for _, sigma in draws])
    # IW mean exists for df>p+1; marginal mu covariance is E[Sigma]/kappa.
    expected_sigma = np.array([[3., 1.], [1., 2.]]) / 9
    np.testing.assert_allclose(covariances.mean(0), expected_sigma, rtol=.055, atol=.007)
    np.testing.assert_allclose(means.mean(0), [1., -2.], atol=.018)
    np.testing.assert_allclose(np.cov(means, rowvar=False), expected_sigma / 3, rtol=.08, atol=.005)


def test_gaussian_regression_draw_moments_independent_linear_algebra_oracle():
    x = np.arange(10.) - 4.5
    X = np.column_stack((np.ones(10), x))
    y = 2 + .7 * x + np.array([1., -.8, .4, -.2, .7, -.6, .3, -.5, .9, -1.2])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    sse = float(((y - X @ beta)**2).sum())
    df = 8
    posterior = _regression_posterior(torch.tensor(X), torch.tensor(y))
    generator = torch.Generator().manual_seed(678)
    draws = [_regression_draw(posterior, generator) for _ in range(5000)]
    betas = np.asarray([b.tolist() for b, s in draws])
    variances = np.array([float(s)**2 for b, s in draws])
    expected_variance = sse / (df - 2)
    np.testing.assert_allclose(betas.mean(0), beta, atol=.015)
    assert variances.mean() == pytest.approx(expected_variance, rel=.045)
    np.testing.assert_allclose(np.cov(betas, rowvar=False), expected_variance * np.linalg.inv(X.T @ X), rtol=.08, atol=.003)


def test_monotone_public_predictive_moments_and_large_outcome_offset():
    x = np.arange(10.) - 4.5
    X = np.column_stack((np.ones(10), x))
    y = 2 + .7 * x + np.array([1., -.8, .4, -.2, .7, -.6, .3, -.5, .9, -1.2])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    variance = ((y - X @ beta)**2).sum() / 6
    prediction = np.array([1., 2.])
    expected_mean = prediction @ beta
    expected_variance = variance * (1 + prediction @ np.linalg.inv(X.T @ X) @ prediction)
    frame = pd.DataFrame({"x": [*x, 2.], "y": [*y, None]})
    values = []
    for seed in range(0, 2000, 100):
        result = mi_monotone(frame, ["x", "y"], m=100, seed=seed)
        values.extend(matrix[-1][1] for matrix in result.completed_matrices)
    assert np.mean(values) == pytest.approx(expected_mean, abs=.045)
    assert np.var(values, ddof=1) == pytest.approx(expected_variance, rel=.09)
    frame["y"] += 1e12
    shifted = mi_monotone(frame, ["x", "y"], m=2)
    assert shifted.metadata["diagnostics"][0]["regressions"][0]["sigma"] > 0


@pytest.mark.parametrize("method", ["mvn", "monotone"])
def test_seed_replay_observed_preservation_and_distinct_imputations(method):
    frame = panel()
    before = frame.copy(deep=True)
    options = {"prior": prior(), "burn": 25, "thin": 5} if method == "mvn" else {}
    sampler = mi_mvn if method == "mvn" else mi_monotone
    first = sampler(frame, ["x", "y"], seed=482, m=4, **options)
    second = sampler(frame, ["x", "y"], seed=482, m=4, **options)
    changed = sampler(frame, ["x", "y"], seed=483, m=4, **options)
    assert first.model_dump_json() == second.model_dump_json()
    assert first.completed_matrices != changed.completed_matrices
    assert len(set(first.completed_matrices)) == 4
    for i in range(1, 5):
        completed = first.dataset(i)
        assert completed.index.equals(frame.index)
        assert bool(torch.isfinite(torch.tensor(completed.to_numpy())).all())
        for column in frame:
            observed = frame[column].notna()
            pd.testing.assert_series_equal(completed.loc[observed, column], frame.loc[observed, column], check_names=False)
    pd.testing.assert_frame_equal(before, frame)
    with pytest.raises(AnalysisError):
        first.dataset(0)
    assert "missing" in first.latex


def test_mvn_arbitrary_missingness_all_missing_rows_and_columns():
    frame = pd.DataFrame({"x": [0., 1., None, 2., None], "y": [2., None, 3., None, None],
                          "z": [float("nan")] * 5}, index=[12, 10, 9, 9, 2])
    result = mi_mvn(frame, ["x", "y", "z"], prior=prior(3), m=3, burn=15, thin=4)
    assert result.metadata["missing_total"] == 10
    assert result.metadata["iterations"] == 27
    assert result.metadata["retained_iterations"] == (19, 23, 27)
    assert result.metadata["convergence"]["assessed"] is False
    assert result.metadata["prior"]["family"] == "normal_inverse_wishart"


def test_mvn_public_univariate_posterior_predictive_mean_and_variance():
    frame = pd.DataFrame({"x": [1., 2., 4., None]})
    p = {"mean": [0.], "kappa": 2., "df": 5., "scale": [[3.]]}
    values = []
    for seed in range(6):
        result = mi_mvn(frame, ["x"], m=100, seed=seed, burn=100, thin=8, prior=p)
        values.extend(matrix[-1][0] for matrix in result.completed_matrices)
    # Analytic predictive mean=7/5 and variance=(71/5)/(8-2)*(1+1/5)=71/25.
    assert np.mean(values) == pytest.approx(7 / 5, abs=.16)
    assert np.var(values, ddof=1) == pytest.approx(71 / 25, rel=.18)


def test_monotone_validates_nested_sets_and_declared_order():
    frame = pd.DataFrame({"x": [0., 1., 2., 3., 4., 5., 6., 7., None, None],
                          "y": [1., 3., 2., 5., 4., 8., None, None, None, None],
                          "z": [2., 1., 5., 2., 4., None, None, None, None, None]})
    result = mi_monotone(frame, ["z", "y", "x"], m=3)
    assert result.metadata["order"] == ("x", "y", "z")
    assert result.metadata["monotone_verified"] is True
    assert result.metadata["regression_draws"] == 9
    with pytest.raises(AnalysisError, match="nested"):
        mi_monotone(frame, ["x", "y", "z"], order=["z", "y", "x"])
    bad = panel()
    bad.iloc[0, 0] = float("nan")
    with pytest.raises(AnalysisError, match="nested"):
        mi_monotone(bad, ["x", "y"])
    with pytest.raises(AnalysisError, match="permutation"):
        mi_monotone(panel(), ["x", "y"], order=["x", "x"])


@pytest.mark.parametrize("frame,code", [
    (pd.DataFrame({"x": [1., None]}), "mi_residual_df"),
    (pd.DataFrame({"x": [1., 1., 1., None]}), "mi_zero_residual"),
    (pd.DataFrame({"x": [1., 1., 1., 1.], "y": [1., 3., 2., None]}), "mi_rank_deficient"),
    (pd.DataFrame({"x": [0., 1., 2., 3.], "y": [0., 2., 4., None]}), "mi_zero_residual"),
])
def test_monotone_improper_posterior_refusals(frame, code):
    with pytest.raises(AnalysisError) as error:
        mi_monotone(frame, list(frame.columns))
    assert error.value.code == code


@pytest.mark.parametrize("index", [pd.RangeIndex(3, 19, 2, name="id"),
    pd.Index([pd.NaT, pd.NA, None, "a", "b", "c", "d", "e"], dtype=object),
    pd.Index(["a", "a", "b", "c", "d", "e", "f", "g"], name="id"),
    pd.Index([1., float("nan"), 3., 4., 5., 6., 7., 8.]),
    pd.date_range("2024-01-01", periods=8, tz="Europe/Istanbul", name="date"),
    pd.timedelta_range("1d", periods=8, name="elapsed"),
    pd.CategoricalIndex(["b", "a", "b", "a", "b", "a", "b", "a"], categories=["a", "b", "unused"], ordered=True),
    pd.MultiIndex(levels=[["a", "b", "unused"], [10, 20, 30]],
                  codes=[[0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 2, 0, 1, 2, 2]], names=["firm", "time"]),
])
def test_saved_result_retains_declared_index_and_nested_immutable_state(index):
    frame = panel()
    frame.index = index
    result = mi_monotone(frame, ["x", "y"], m=2)
    restored = MIResult.model_validate_json(result.model_dump_json())
    pd.testing.assert_index_equal(restored.dataset().index, index, exact=True)
    assert restored.completed_matrices == result.completed_matrices
    assert restored.integrity_sha256 == result.integrity_sha256
    with pytest.raises(ValidationError):
        restored.seed = 5
    with pytest.raises(TypeError):
        restored.metadata["seed"] = 5
    with pytest.raises(TypeError):
        restored.metadata["sample"]["positions"][0] = 1


@pytest.mark.parametrize("change", ["observed", "imputed", "mask", "ids", "seed", "prior", "index", "nan"])
def test_full_state_replay_rejects_corruption(change):
    result = mi_mvn(panel(), ["x", "y"], prior=prior(), m=2, burn=3, thin=1)
    state = json.loads(result.model_dump_json())
    if change == "observed":
        state["completed_matrices"][0][0][0] = 100.
    elif change == "imputed":
        state["completed_matrices"][0][-1][1] += 1.
    elif change == "mask":
        state["metadata"]["missing_mask"][-1][1] = False
    elif change == "ids":
        state["metadata"]["imputation_ids"] = [2, 1]
    elif change == "seed":
        state["seed"] += 1
        state["metadata"]["seed"] += 1
    elif change == "prior":
        state["metadata"]["prior"]["kappa"] += 1
    elif change == "index":
        state["metadata"]["index"]["values"].append({"type": "scalar", "value": "extra"})
    elif change == "nan":
        state["completed_matrices"][0][-1][1] = float("nan")
    with pytest.raises((ValidationError, AnalysisError)):
        MIResult.model_validate(state)
    with pytest.raises(ValidationError):
        result.model_copy(update={"seed": result.seed + 1})


@pytest.mark.parametrize("sampler", [mi_mvn, mi_monotone])
def test_complete_panel_keeps_identical_datasets_with_zero_missing(sampler):
    frame = panel().fillna(9.)
    options = {"prior": prior(), "burn": 2, "thin": 1} if sampler is mi_mvn else {}
    result = sampler(frame, ["x", "y"], m=3, **options)
    assert len(set(result.completed_matrices)) == 1
    assert result.metadata["missing_total"] == 0 and result.metadata["iterations"] == 0
    pd.testing.assert_frame_equal(result.dataset(), frame)


def test_make_result_crosschecks_observed_cells_and_saved_geometry():
    frame, values, missing, metadata = admit(panel(), ["x", "y"], m=1)
    filled = torch.nan_to_num(values, nan=10.)
    bad = filled.clone()
    bad[0, 0] += 1
    with pytest.raises(ValidationError, match="observed"):
        make_result("mi_monotone", frame, values, missing, [bad], 0, metadata)
    metadata = copy.deepcopy(metadata)
    metadata["sample"]["positions"] = list(reversed(range(8)))
    with pytest.raises(ValidationError, match="sample"):
        make_result("mi_monotone", frame, values, missing, [filled], 0, metadata)


@pytest.mark.parametrize("change", ["converged", "parity", "work", "budget", "estimate", "assessed", "tiny_plan", "boolean_counts", "claim", "stationarity"])
def test_fresh_checksum_cannot_legitimize_false_compute_claims(change):
    frame, values, missing, metadata = admit(panel(), ["x", "y"], m=1)
    filled = torch.nan_to_num(values, nan=10.)
    if change == "converged":
        metadata["converged"] = True
    elif change == "parity":
        metadata["stata_parity_validated"] = True
    elif change == "work":
        metadata["work_estimate"] = -10
        metadata["max_work"] = 1
    elif change == "budget":
        metadata["resource_plan"]["budget_bytes"] = 1
    elif change == "estimate":
        metadata["resource_plan"]["estimated_workspace_bytes"] = 1
    elif change == "assessed":
        metadata["convergence"] = {"assessed": True}
    elif change == "tiny_plan":
        metadata["resource_plan"].update({"buffers": {"fake": 1}, "estimated_workspace_bytes": 1, "budget_bytes": 1})
    elif change == "boolean_counts":
        metadata["missing_counts"][0] = False
    elif change == "claim":
        metadata["convergence_claim"] = True
    elif change == "stationarity":
        metadata["logit_sampler"] = {"stationarity_claim": True}
    with pytest.raises(ValidationError):
        make_result("mi_mvn", frame, values, missing, [filled], 0, metadata)
