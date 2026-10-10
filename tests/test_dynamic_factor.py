"""Independent Gaussian conditioning, OIM, prediction and admission checks."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.tsworkflows import dfm, ssengine
from openecon.econometrics.tsworkflows.dfm import (
    FactorModel, dfactor, dfactor_forecast, dfactor_nowcast, dfactor_restore,
    fit_dynamic_factor,
)
from openecon.resources import use_workspace_budget


def tensor(value):
    return torch.tensor(value, dtype=torch.float64, device="cpu")


def fixed(r=2, p=5, n=11):
    rng = np.random.default_rng(32741)
    L = rng.normal(size=(p, r))
    L[:r] = np.eye(r)
    A = np.eye(r)*.45+np.triu(np.ones((r, r)), 1)*.1
    Q = np.eye(r)*.2+np.ones((r, r))*.04
    P0 = np.eye(r)*.6+np.ones((r, r))*.1
    y = rng.normal(size=(n, p))
    y[0, 0] = np.nan
    y[3] = np.nan
    y[-3:, -1] = np.nan
    values = dict(a0=np.arange(r)*.1, P0=P0, Z=L, T=A, Q=Q,
                  H=np.diag(np.linspace(.2, .6, p)), c=np.zeros(r), d=np.arange(p)*.03)
    return tensor(y), {k: tensor(v) for k, v in values.items()}


def gaussian_conditioning(y, values):
    """Primitive joint Gaussian prior, independently conditioned in one solve."""
    y = np.asarray(y)
    n, p = y.shape
    r = len(values["a0"])
    A, Q, L, H, P0 = [np.asarray(values[k]) for k in ("T", "Q", "Z", "H", "P0")]
    means = np.zeros((n+1, r))
    means[0] = values["a0"]
    full = np.zeros(((n+1)*r, (n+1)*r))
    full[:r, :r] = P0
    for t in range(n):
        means[t+1] = A @ means[t]+values["c"]
        for s in range(t+1):
            block = A @ full[t*r:(t+1)*r, s*r:(s+1)*r]
            full[(t+1)*r:(t+2)*r, s*r:(s+1)*r] = block
            full[s*r:(s+1)*r, (t+1)*r:(t+2)*r] = block.T
        full[(t+1)*r:(t+2)*r, (t+1)*r:(t+2)*r] = A @ full[t*r:(t+1)*r, t*r:(t+1)*r] @ A.T+Q
    G = np.zeros((n*p, (n+1)*r))
    for t in range(n):
        G[t*p:(t+1)*p, t*r:(t+1)*r] = L
    observation_mean = (G @ means.flatten()).reshape(n, p)+values["d"]
    S = G @ full @ G.T+np.kron(np.eye(n), H)
    present = np.isfinite(y).flatten()
    cross = full @ G.T[:, present]
    S = S[present][:, present]
    residual = (y-observation_mean).flatten()[present]
    posterior_mean = means.flatten()+cross @ np.linalg.solve(S, residual)
    posterior_covariance = full-cross @ np.linalg.solve(S, cross.T)
    likelihood = -.5*(len(residual)*np.log(2*np.pi)+np.linalg.slogdet(S)[1]+residual @ np.linalg.solve(S, residual))
    return posterior_mean.reshape(n+1, r), posterior_covariance, likelihood


@pytest.mark.parametrize("r,p", [(1, 3), (2, 5), (3, 7)])
def test_shared_factor_filter_rts_matches_full_gaussian_conditioning(r, p):
    y, values = fixed(r, p)
    output = ssengine.factor_smooth(y, values, profile=ssengine.FactorAdmission())
    mean, covariance, likelihood = gaussian_conditioning(y.numpy(), {k: v.numpy() for k, v in values.items()})
    np.testing.assert_allclose(output["smoothed"], mean[:-1], atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(float(output["log_likelihood"]), likelihood, atol=3e-13, rtol=3e-13)
    for t in range(len(y)):
        np.testing.assert_allclose(output["smoothed_covariance"][t], covariance[t*r:(t+1)*r, t*r:(t+1)*r], atol=3e-13, rtol=3e-13)
        if t < len(y)-1:
            np.testing.assert_allclose(output["lag_one_covariance"][t], covariance[(t+1)*r:(t+2)*r, t*r:(t+1)*r], atol=3e-13, rtol=3e-13)
    assert output["observed_count"][3] == 0
    assert len(output["smoothed"]) == len(y)


def test_static_factor_baseline_has_independent_dates():
    y, values = fixed(2, 5)
    values["T"] = torch.zeros((2, 2), dtype=torch.float64)
    output = ssengine.factor_smooth(y, values, profile=ssengine.FactorAdmission())
    assert torch.equal(output["lag_one_covariance"], torch.zeros_like(output["lag_one_covariance"]))
    assert torch.equal(output["smoothed"][3], torch.zeros(2, dtype=torch.float64))


def test_frozen_native_r_package_author_reference():
    """Actual native R outputs, source/version/license/hashes retained offline."""
    packet = json.loads((Path(__file__).parent/"fixtures"/"dynamic-factor-native-reference.json").read_text())
    source, reference = packet["source"], packet["reference"]
    output = ssengine.factor_smooth(tensor(np.asarray(source["y"], dtype=float)),
        {k: tensor(v) for k, v in source["system"].items()}, profile=ssengine.FactorAdmission())
    assert reference["source_identity_checked"] is True
    assert reference["KFAS"] == "1.6.0"
    assert reference["dfms"]["version"] == "1.0.1"
    assert packet["provenance"]["dfms_source"]["git_commit"] == "137a9a8d5a608a3e95f4015956be8742e2a789b1"
    for native in (reference["kfas"], reference["dfms"]):
        for key, expected in native.items():
            if key in output:
                actual = np.asarray(output[key])
                np.testing.assert_allclose(actual, np.asarray(expected).reshape(actual.shape), atol=2e-10, rtol=2e-10)


def simulated_frame(n=80):
    rng = np.random.default_rng(7191)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = .64*f[t-1]+rng.normal()*.65
    y = f[:, None]*np.array([1., .8, -.6])+rng.normal(size=(n, 3))*.5
    y[10:15, 1] = np.nan
    y[45] = np.nan
    y[-5:, 2] = np.nan
    frame = pd.DataFrame(y, columns=["anchor", "production", "prices"], index=pd.Index([f"id{i}" for i in range(n)], name="source"))
    frame["date"] = pd.date_range("2020-01-01", periods=n, freq="MS")
    return frame


@pytest.fixture(scope="module")
def fit():
    return dfactor(simulated_frame(), ["anchor", "production", "prices"], time="date",
                   restarts=2, max_em_iterations=12, max_ml_iterations=100)


def test_complete_fit_has_full_oim_and_monotone_em(fit):
    state = fit.attrs["dynamic_factor_result"]["state"]
    assert state["prior"]["target"] == "conditional-known-finite-prior"
    assert len(state["names"]) == 10
    assert np.asarray(state["inference"]["covariance"]).shape == (10, 10)
    assert np.linalg.eigvalsh(state["inference"]["information"]).min() > 0
    assert state["system"]["Z"][0] == [1.]
    assert 0 < abs(state["system"]["T"][0][0]) < 1
    for restart in state["optimization"]["restarts"]:
        values = [v["log_likelihood"] for v in restart["em_path"]]
        assert np.all(np.diff(values) >= 0)
        assert restart["converged"]
        assert restart["final_log_likelihood"] >= values[-1]-1e-8
    assert list(fit["smoothed"].index) == list(simulated_frame().index)
    assert state["output"]["observed_count"][45] == 0
    assert np.isfinite(fit["coefficients"][["estimate", "se", "z", "p", "lo", "hi"]].to_numpy()).all()


def test_parameter_hessian_matches_independent_finite_difference(fit):
    """Finite differences of a NumPy likelihood, not of production autodiff."""
    state = fit.attrs["dynamic_factor_result"]["state"]
    y = np.array(state["y"], dtype=float)
    point = np.asarray(state["theta"])

    def likelihood(theta):
        d, L = theta[:3], np.array([[1.], [theta[3]], [theta[4]]])
        A = np.array([[theta[5]]])
        Q, H = np.array([[np.exp(theta[6])**2]]), np.diag(np.exp(theta[7:]))
        mean, P, ll = np.asarray(state["prior"]["a0"]), np.asarray(state["prior"]["P0"]), 0.
        for row in y:
            present = np.isfinite(row)
            Z = L[present]
            if present.any():
                residual = row[present]-d[present]-Z @ mean
                noise = H[present][:, present]
                F = Z @ P @ Z.T+noise
                ll -= .5*(present.sum()*np.log(2*np.pi)+np.linalg.slogdet(F)[1]+residual @ np.linalg.solve(F, residual))
                gain = np.linalg.solve(F, Z @ P).T
                mean = mean+gain @ residual
                C = np.eye(1)-gain @ Z
                P = C @ P @ C.T+gain @ noise @ gain.T
            mean, P = A @ mean, A @ P @ A.T+Q
        return ll

    epsilon = 1e-4
    hessian = np.zeros((len(point), len(point)))
    for i in range(len(point)):
        a = np.eye(len(point))[i]*epsilon
        hessian[i, i] = (likelihood(point+a)-2*likelihood(point)+likelihood(point-a))/epsilon**2
        for j in range(i):
            b = np.eye(len(point))[j]*epsilon
            hessian[i, j] = hessian[j, i] = (likelihood(point+a+b)-likelihood(point+a-b)-likelihood(point-a+b)+likelihood(point-a-b))/(4*epsilon**2)
    np.testing.assert_allclose(-hessian, state["inference"]["information"], atol=2e-5, rtol=2e-5)


def test_restore_never_calls_optimizer(fit, monkeypatch):
    monkeypatch.setattr(dfm, "fit_dynamic_factor", lambda *a, **k: pytest.fail("refitting forbidden"))
    monkeypatch.setattr(dfm, "ml_optimize", lambda *a, **k: pytest.fail("optimization forbidden"))
    restored = dfactor_restore(fit.to_json())
    assert restored.attrs["dynamic_factor_result"]["sha256"] == fit.attrs["dynamic_factor_result"]["sha256"]
    assert restored["coefficients"].equals(fit["coefficients"])


@pytest.mark.parametrize("field", ["smoothed", "smoothed_covariance", "log_likelihood", "observed_mask"])
def test_rehashed_changed_posterior_refused(fit, field):
    record = copy.deepcopy(fit.attrs["dynamic_factor_result"])
    state = record["state"]
    if field == "log_likelihood":
        state["output"][field] += .1
    elif field == "observed_mask":
        state["output"][field][0][0] = 1
    elif field == "smoothed_covariance":
        state["output"][field][0][0][0] += .1
    else:
        state["output"][field][0][0] += .1
    state["sha256"] = dfm._digest({k: v for k, v in state.items() if k != "sha256"})
    record["sha256"] = dfm._digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(AnalysisError, match="replay"):
        dfactor_restore(record)


@pytest.mark.parametrize("field", ["ml_gradient_max", "ml_diagnostics", "em_max_iterations"])
def test_rehashed_optimizer_diagnostics_refused(fit, field):
    record = copy.deepcopy(fit.attrs["dynamic_factor_result"])
    state = record["state"]
    restart = state["optimization"]["restarts"][state["optimization"]["chosen_restart"]]
    restart[field] = 1e99 if field == "ml_gradient_max" else {"invented": "invalid"} if field == "ml_diagnostics" else True
    state["sha256"] = dfm._digest({k: v for k, v in state.items() if k != "sha256"})
    record["sha256"] = dfm._digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(AnalysisError, match="replay"):
        dfactor_restore(record)


def test_forecast_joint_covariance_and_delta(fit):
    out = dfactor_forecast(fit, 3)
    state = fit.attrs["dynamic_factor_result"]["state"]
    A, L, Q, H = [np.asarray(state["system"][k]) for k in ("T", "Z", "Q", "H")]
    P = np.asarray(state["output"]["next_covariance"])
    np.testing.assert_allclose(out["conditional_covariance"].to_numpy()[:3, :3], L @ P @ L.T+H, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out["conditional_covariance"].to_numpy()[:3, 3:6], L @ P @ A.T @ L.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out["conditional_covariance"].to_numpy()[3:6, 3:6], L @ (A @ P @ A.T+Q) @ L.T+H, rtol=1e-12, atol=1e-12)
    assert np.linalg.eigvalsh(out["parameter_covariance"].to_numpy()).min() > -1e-12
    assert np.trace(out["parameter_covariance"].to_numpy()) > 0
    assert out["predictions"].index[0] == pd.Timestamp("2026-09-01")
    without = dfactor_forecast(fit, 3, parameter_uncertainty=False)
    np.testing.assert_array_equal(without["parameter_covariance"], 0)
    np.testing.assert_allclose(without["predictions"]["mean"], out["predictions"]["mean"])


def test_missing_nowcast_joint_covariance_matches_gaussian_conditioning(fit):
    targets = [(45, "anchor"), (45, "prices"), (77, "prices")]
    out = dfactor_nowcast(fit, targets, parameter_uncertainty=False)
    state = fit.attrs["dynamic_factor_result"]["state"]
    y = np.asarray(state["y"], dtype=float)
    values = {k: np.asarray(v) for k, v in state["system"].items()}
    mean, covariance, _ = gaussian_conditioning(y, values)
    L, H = values["Z"], values["H"]
    cells = [(t, state_name) for t, state_name in ((45, 0), (45, 2), (77, 2))]
    expected_mean = [values["d"][j]+L[j] @ mean[t] for t, j in cells]
    expected = np.array([[L[j] @ covariance[t:t+1, s:s+1] @ L[k]+(H[j, k] if t == s else 0)
                          for s, k in cells] for t, j in cells])
    np.testing.assert_allclose(out["predictions"]["mean"], expected_mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out["conditional_covariance"].to_numpy(), expected, atol=1e-12, rtol=1e-12)


def test_nowcast_observed_targets_and_future_generators_refused(fit):
    with pytest.raises(AnalysisError, match="unobserved"):
        dfactor_nowcast(fit, [(0, "anchor")])
    with pytest.raises(AnalysisError, match="resident"):
        dfactor_forecast(fit, 2, future_time=(v for v in range(100000)))


def test_fit_budget_refusal_precedes_any_numeric_scan(monkeypatch):
    monkeypatch.setattr(torch, "isinf", lambda *a, **k: pytest.fail("numeric scan occurred"))
    monkeypatch.setattr(torch, "isfinite", lambda *a, **k: pytest.fail("numeric scan occurred"))
    with pytest.raises(AnalysisError, match="work"):
        fit_dynamic_factor(torch.empty((1000, 3), dtype=torch.float64), max_work=1)


def test_shared_factor_refusal_precedes_any_numeric_scan(monkeypatch):
    y, values = fixed()
    monkeypatch.setattr(torch, "isinf", lambda *a, **k: pytest.fail("numeric scan occurred"))
    monkeypatch.setattr(torch, "isfinite", lambda *a, **k: pytest.fail("numeric scan occurred"))
    with pytest.raises(AnalysisError, match="budget"):
        ssengine.factor_smooth(y, values, profile=ssengine.FactorAdmission(max_pass_work=1))


def test_resident_refusal_precedes_source_iteration():
    class Probe(list):
        def __iter__(self):
            pytest.fail("refused source was scanned")
    raw = {v: Probe([0.]*20001) for v in ("a", "b", "c")}
    with pytest.raises(AnalysisError, match="dates"):
        dfactor(raw, ["a", "b", "c"])


def test_nonexact_large_integer_source_refused():
    frame = pd.DataFrame({"a": [2**54+1]*10, "b": range(10), "c": range(10)})
    with pytest.raises(AnalysisError, match="rounded"):
        dfactor(frame, ["a", "b", "c"])


def test_unhashable_response_names_refused_as_input():
    with pytest.raises(AnalysisError, match="measurement"):
        dfactor({}, [[], [], []])


@pytest.mark.parametrize("budget", [True, 0, -1, 2_000_000_001, 1.5])
def test_factor_profile_budget_is_strict(budget):
    with pytest.raises(AnalysisError):
        ssengine.FactorAdmission(max_pass_work=budget)


def test_factor_profile_and_public_limits_are_distinct():
    y, values = fixed(2, 9)
    with pytest.raises(AnalysisError, match="1..8"):
        ssengine.kalman(y, values)
    output = ssengine.factor_kalman(y, values, profile=ssengine.FactorAdmission())
    assert torch.isfinite(output["log_likelihood"])
    with pytest.raises(AnalysisError):
        ssengine.factor_kalman(y, values, profile=None)


def test_input_prior_symmetry_is_checked_in_declared_coordinate_units():
    prior = tensor([[1., 1e-20], [0., 1e-30]])
    with pytest.raises(AnalysisError, match="coordinate units"):
        FactorModel(5, 2, [0, 1], tensor([0., 0.]), prior)
    valid = tensor([[1., 0.], [0., 1e-30]])
    model = FactorModel(5, 2, [0, 1], tensor([0., 0.]), valid)
    assert torch.equal(model.P0, valid)


def test_workspace_refusal_precedes_matrix_factorization(monkeypatch):
    monkeypatch.setattr(torch.linalg, "cholesky_ex", lambda *a, **k: pytest.fail("factorization before admission"))
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        fit_dynamic_factor(torch.empty((1000, 3), dtype=torch.float64))


def test_global_dtype_rng_and_default_device_are_preserved():
    before_dtype, before_device, rng = torch.get_default_dtype(), torch.get_default_device(), torch.get_rng_state().clone()
    y, values = fixed()
    model = FactorModel(5, 2, [0, 1], values["a0"], values["P0"])
    theta = model.encode(values)
    dfm._em_update(y, model, theta, ssengine.factor_smooth(y, values, profile=model.profile))
    assert torch.get_default_dtype() == before_dtype
    assert torch.get_default_device() == before_device
    assert torch.equal(torch.get_rng_state(), rng)


def test_bad_dtype_json_and_encoded_state_refused(fit):
    record = json.loads(fit.to_json())
    record["source"]["response_dtypes"][0] = "object"
    record["sha256"] = dfm._digest({k: v for k, v in record.items() if k != "sha256"})
    with pytest.raises(AnalysisError):
        dfactor_restore(record)
    with pytest.raises(AnalysisError):
        dfactor_restore(" "*(64*1024**2+1))
