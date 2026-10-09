"""Independent analytic local-derivative and saved scalar RM-contrast references."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import t

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.regularized.derivatives import local_derivatives, local_average_derivatives
from openecon.econometrics.stats.rm_contrast import rm_contrast


def smoother(method="localreg", kernel="gaussian", linear=False):
    rng = np.random.default_rng(44173)
    x = rng.uniform(-1, 1, (72, 2))
    y = 2 + 4 * x[:, 0] - 2 * x[:, 1] if linear else np.sin(2 * x[:, 0]) + x[:, 1] ** 2 + rng.normal(0, .1, len(x))
    data = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    fit = getattr(oe, method)(data=data, y="y", x=["a", "b"], kernel=kernel,
                              selection="fixed", bandwidth=[.7, .55])
    return fit, data, x, y


def numpy_mean(x, y, q, h, degree):
    # Dense independent weighted normal equations, allowing complex-step q.
    delta = x - q
    weight = np.exp(-.5 * np.sum((delta / h) ** 2, axis=1))
    if degree == 0:
        return weight @ y / weight.sum()
    a = np.column_stack([np.ones(len(x)), delta])
    return np.linalg.solve(a.T @ (weight[:, None] * a), a.T @ (weight * y))[0]


@pytest.mark.parametrize("method,degree", [("kernelreg", 0), ("localreg", 1)])
def test_saved_gaussian_estimator_derivative_complex_step_and_mean_replay(method, degree):
    fit, data, x, y = smoother(method)
    saved = oe.ResultBundle.model_validate_json(fit.model_dump_json())
    before = saved.model_dump_json()
    query = pd.DataFrame({"a": [x[:, 0].min(), -.2, .45], "b": [.1, x[:, 1].max(), -.25], "y": [1e30] * 3},
                         index=pd.Index([9, 4, 9], name="query"))
    actual = local_derivatives(saved, data=query)
    expected = []
    for q in query[["a", "b"]].to_numpy():
        row = []
        for j in range(2):
            complex_query = q.astype(complex)
            complex_query[j] += 1e-20j
            row.append(numpy_mean(x, y, complex_query, np.array([.7, .55]), degree).imag / 1e-20)
        expected.append(row)
    np.testing.assert_allclose(actual[["d_mean_d_a", "d_mean_d_b"]], expected, rtol=3e-12, atol=2e-12)
    np.testing.assert_allclose(actual["mean"], oe.regularized_predict(saved, query), atol=2e-12)
    assert actual.index.tolist() == [9, 4, 9]
    assert saved.model_dump_json() == before
    assert actual.attrs["inference"].startswith("unavailable")
    average = local_average_derivatives(saved, data=query)
    np.testing.assert_allclose(average.average_derivative, np.mean(expected, axis=0), atol=2e-12)
    assert len(actual.attrs["state_sha256"]) == 64


def test_local_linear_exact_derivative_reproduction_at_boundaries_and_cpu_default():
    fit, data, _, _ = smoother(linear=True)
    with torch.device("meta"):
        out = local_derivatives(fit, data=data.iloc[[0, 31, 71]])
    np.testing.assert_allclose(out[["d_mean_d_a", "d_mean_d_b"]], [[4, -2]] * 3, atol=2e-12)
    assert out.attrs["device"] == "cpu"


def test_local_derivative_includes_moving_weights_not_just_local_slope():
    fit, _, x, y = smoother()
    q = np.array([.65, -.65])
    out = local_derivatives(fit, data={"a": [q[0]], "b": [q[1]]})
    a = np.column_stack([np.ones(len(x)), x - q])
    w = np.exp(-.5 * np.sum(((x - q) / [.7, .55]) ** 2, axis=1))
    local_slope = np.linalg.solve(a.T @ (w[:, None] * a), a.T @ (w * y))[1:]
    assert np.max(abs(out[["d_mean_d_a", "d_mean_d_b"]].to_numpy()[0] - local_slope)) > .01


@pytest.mark.parametrize("change,code", [
    (lambda q: q.assign(a=np.nan), "non_finite_values"),
    (lambda q: q.assign(a=2.), "outside_support"),
    (lambda q: q.iloc[:0], "invalid_predictors"),
    (lambda q: q.drop(columns="a"), "invalid_predictors"),
])
def test_derivative_query_refusals(change, code):
    fit, data, _, _ = smoother()
    with pytest.raises(AnalysisError) as caught:
        local_derivatives(fit, data=change(data.iloc[:3]))
    assert caught.value.code == code


def test_derivative_kernel_state_and_resource_refusals(monkeypatch):
    fit, data, _, _ = smoother(kernel="epanechnikov")
    with pytest.raises(AnalysisError, match="Gaussian"):
        local_derivatives(fit, data=data.iloc[:2])
    fit, data, _, _ = smoother()
    with pytest.raises(AnalysisError) as caught:
        local_derivatives(fit, data=data.iloc[:2], max_work=1)
    assert caught.value.code == "work_limit"
    with pytest.raises(AnalysisError):
        local_derivatives(fit, data=data.iloc[:2], max_work=True)
    corrupted = fit.model_copy(deep=True)
    corrupted.extra["smoother_state"]["training_x"][0].append(1.)
    with pytest.raises(AnalysisError) as caught:
        local_derivatives(corrupted, data=data.iloc[:2])
    assert caught.value.code == "invalid_state"
    with pytest.raises(AnalysisError) as caught:
        local_derivatives(fit, data=oe.Dataset.from_frame(data.iloc[:2]))
    assert caught.value.code == "streaming_unsupported"
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "1")
    with pytest.raises(AnalysisError) as caught:
        local_derivatives(fit, data=pd.concat([data.iloc[:2]] * 2048))
    assert caught.value.code == "workspace_limit"


def rm_data(crossed=False):
    rng = np.random.default_rng(7344)
    cells = 6 if crossed else 3
    n = 31
    group = np.array(["A"] * 13 + ["B"] * 18)
    a = rng.normal(size=(cells, cells))
    sigma = a @ a.T + np.diag(np.arange(1, cells + 1))
    means = np.linspace(.2, 1.7, cells)[None, :] + (group == "A")[:, None] * np.linspace(-.3, .8, cells)
    wide = means + rng.normal(size=(n, cells)) @ np.linalg.cholesky(sigma).T
    records = []
    for i in range(n):
        for j in range(cells):
            records.append([i, group[i], j // 3, j % 3, wide[i, j]])
    frame = pd.DataFrame(records, columns=["id", "g", "a", "time", "y"])
    if not crossed:
        frame = frame.drop(columns="a")
    return frame.sample(frac=1, random_state=17), wide, group


@pytest.mark.parametrize("crossed", [False, True])
@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("group_difference", [False, True])
def test_saved_rm_scalar_contrast_full_geometry_and_exact_t_oracle(crossed, dataset, group_difference):
    frame, y, groups = rm_data(crossed)
    within = ["a", "time"] if crossed else ["time"]
    x = np.column_stack([np.ones(len(y)), np.where(groups == "A", 1., -1.)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ beta
    covariance = e.T @ e / (len(y) - 2)
    bread = np.linalg.inv(x.T @ x)
    c_weights = np.linspace(-1, 1, y.shape[1])
    l_weights = np.array([0., 2.]) if group_difference else np.array([1., 0.])
    source = oe.Dataset.from_frame(frame) if dataset else frame
    result = oe.rm_anova(source, "y", "id", within, between=["g"])
    restored = oe.restore_summary(oe.summary_state(result))
    state = restored.attrs["rm_contrast_state"]
    np.testing.assert_allclose(state["coefficients"], beta, atol=2e-12)
    np.testing.assert_allclose(state["bread"], bread, atol=2e-13)
    np.testing.assert_allclose(state["residual_cell_covariance"], covariance, atol=2e-12)
    assert state["between_design_columns"] == ["Intercept", "g"]
    before = oe.summary_state(restored)
    with torch.device("meta"):
        output = rm_contrast(restored, c_weights, between_contrast=l_weights, null=.2, alpha=.1)
    actual = output["contrast"].iloc[0]
    estimate = l_weights @ beta @ c_weights
    se = np.sqrt((l_weights @ bread @ l_weights) * (c_weights @ covariance @ c_weights))
    statistic = (estimate - .2) / se
    assert actual.estimate == pytest.approx(estimate, abs=3e-12)
    assert actual.std_error == pytest.approx(se, rel=3e-12)
    assert actual.statistic == pytest.approx(statistic, abs=3e-12)
    assert actual.p_value == pytest.approx(2 * t.sf(abs(statistic), 29), rel=2e-7)
    assert actual.ci_low == pytest.approx(estimate - t.ppf(.95, 29) * se, rel=2e-7, abs=2e-9)
    assert actual.ci_high == pytest.approx(estimate + t.ppf(.95, 29) * se, rel=2e-7, abs=2e-9)
    assert output.attrs["sphericity_required"] is False
    assert oe.summary_state(restored) == before
    replay = oe.restore_summary(oe.summary_state(output))
    np.testing.assert_allclose(replay["contrast"], output["contrast"], atol=0)


def test_default_rm_contrast_is_equal_between_group_marginal_not_pooled_weighted():
    frame, y, groups = rm_data()
    result = oe.rm_anova(frame, "y", "id", ["time"], between=["g"])
    contrast = [-1., 0., 1.]
    output = rm_contrast(result, contrast)
    group_mean = .5 * (y[groups == "A"].mean(0) + y[groups == "B"].mean(0))
    expected = group_mean @ contrast
    assert output["contrast"].estimate.iloc[0] == pytest.approx(expected, abs=2e-12)
    assert abs(expected - y.mean(0) @ contrast) > .01


@pytest.mark.parametrize("weights", [[0, 0, 0], [1, -1], [[1], [0], [-1]], [1, np.nan, -1], [True, False, True], np.array([1, 0, -1], dtype=complex)])
def test_rm_contrast_refuses_invalid_vectors(weights):
    frame, _, _ = rm_data()
    fit = oe.rm_anova(frame, "y", "id", ["time"], between=["g"])
    with pytest.raises(AnalysisError) as caught:
        rm_contrast(fit, weights)
    assert caught.value.code == "invalid_contrast"


def test_rm_contrast_digest_df_missing_old_state_and_zero_variance_refusals():
    frame, _, _ = rm_data()
    fit = oe.rm_anova(frame, "y", "id", ["time"], between=["g"])
    old = copy.deepcopy(fit)
    old.attrs.pop("rm_contrast_state")
    with pytest.raises(AnalysisError) as caught:
        rm_contrast(old, [-1, 0, 1])
    assert caught.value.code == "unsupported_saved_geometry"
    changed = copy.deepcopy(fit)
    changed.attrs["rm_contrast_state"]["coefficients"][0][0] += 1
    with pytest.raises(AnalysisError) as caught:
        rm_contrast(changed, [-1, 0, 1])
    assert caught.value.code == "invalid_state"
    changed = copy.deepcopy(fit)
    state = changed.attrs["rm_contrast_state"]
    state["df_resid"] = 0
    state["sha256"] = hashlib.sha256(json.dumps({k:v for k,v in state.items() if k != "sha256"}, sort_keys=True,
                                               allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(AnalysisError) as caught:
        rm_contrast(changed, [-1, 0, 1])
    assert caught.value.code == "invalid_state"
    constant_effect = frame.copy()
    constant_effect["y"] = constant_effect["id"] + constant_effect["time"]
    constant = oe.rm_anova(constant_effect, "y", "id", ["time"])
    with pytest.raises(AnalysisError) as caught:
        rm_contrast(constant, [-1, 0, 1])
    assert caught.value.code == "zero_contrast_variance"


def test_large_existing_rm_design_remains_valid_with_saved_contrast_explicitly_unavailable():
    rng = np.random.default_rng(730128)
    wide = rng.normal(size=(8, 129))
    frame = pd.DataFrame([(i, j, wide[i, j]) for i in range(8) for j in range(129)],
                         columns=["id", "time", "y"])
    fit = oe.rm_anova(frame, "y", "id", ["time"])
    assert fit.attrs["within_cells"] == 129
    assert fit.attrs["rm_contrast_available"] is False
    assert fit.attrs["rm_contrast_state"] is None
    assert any("128 within cells" in note for note in fit.attrs["notes"])
    with pytest.raises(AnalysisError) as caught:
        rm_contrast(fit, [-1.] + [0.] * 127 + [1.])
    assert caught.value.code == "unsupported_saved_geometry"
