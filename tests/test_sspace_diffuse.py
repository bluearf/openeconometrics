"""Independent flat-prior GLS conditioning, rank laws and resource refusal."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget
from openecon.econometrics.tsworkflows.ssengine import kalman, smooth
from openecon.econometrics.tsworkflows.ssdiffuse import (
    filter_exact_diffuse, forecast_exact_diffuse, restore_diffuse_state,
    save_diffuse_state, smooth_exact_diffuse, sspace_diffuse, diffuse_restore, diffuse_forecast, DiffuseResult,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64, device="cpu")


def local_level(n=6):
    y = tensor([[1.2], [math.nan], [1.8], [0.7], [2.1], [1.5]])[:n]
    values = {key: tensor(value) for key, value in dict(
        a0=[0.3], P_inf=[[1.0]], P_star=[[0.0]], Z=[[1.0]], T=[[1.0]],
        Q=[[0.25]], H=[[0.6]], c=[0.1], d=[-0.2]).items()}
    return y, values


def independent_joint(y, values):
    """NumPy primitive-noise expansion and GLS; no production state recursions."""
    y = y.detach().numpy()
    v = {key: value.detach().numpy() for key, value in values.items()}
    n, p = y.shape
    m = len(v["a0"])
    def at(key, t):
        return v[key][t] if v[key].ndim == (2 if key in {"c", "d"} else 3) else v[key]
    eig, basis = np.linalg.eigh(v["P_inf"])
    positive = eig > 1e-12
    initial_loading = basis[:, positive]*np.sqrt(eig[positive])
    width = (n+1)*m
    innovation = np.zeros((width, width))
    innovation[:m, :m] = v["P_star"]
    for t in range(n):
        innovation[(t+1)*m:(t+2)*m, (t+1)*m:(t+2)*m] = at("Q", t)
    expansion = np.zeros((width, width))
    expansion[:m, :m] = np.eye(m)
    means = [v["a0"]]
    loadings = [initial_loading]
    for t in range(n):
        sl = slice((t+1)*m, (t+2)*m)
        expansion[sl] = at("T", t)@expansion[t*m:(t+1)*m]
        expansion[sl, sl] = np.eye(m)
        means.append(at("T", t)@means[-1]+at("c", t))
        loadings.append(at("T", t)@loadings[-1])
    proper = expansion@innovation@expansion.T
    prior_mean, loading = np.concatenate(means), np.concatenate(loadings)
    positions = np.flatnonzero(~np.isnan(y.flatten()))
    observation = np.zeros((len(positions), width))
    H = np.zeros((n*p, n*p))
    offset = np.zeros(n*p)
    for t in range(n):
        H[t*p:(t+1)*p, t*p:(t+1)*p] = at("H", t)
        offset[t*p:(t+1)*p] = at("d", t)
    for i, physical in enumerate(positions):
        t, component = divmod(physical, p)
        observation[i, t*m:(t+1)*m] = at("Z", t)[component]
    omega = observation@proper@observation.T + H[positions][:, positions]
    design = observation@loading
    cross = proper@observation.T
    residual = y.flatten()[positions]-observation@prior_mean-offset[positions]
    inverse = np.linalg.inv(omega)
    if loading.shape[1]:
        information = design.T@inverse@design
        beta_cov = np.linalg.inv(information)
        beta = beta_cov@design.T@inverse@residual
        remainder = residual-design@beta
        bridge = loading-cross@inverse@design
        mean = prior_mean+loading@beta+cross@inverse@remainder
        covariance = proper-cross@inverse@cross.T+bridge@beta_cov@bridge.T
        logdiffuse = np.linalg.slogdet(information)[1]
    else:
        mean = prior_mean+cross@inverse@residual
        covariance = proper-cross@inverse@cross.T
        remainder, logdiffuse = residual, 0.0
    ll = -0.5*((len(positions)-loading.shape[1])*math.log(2*math.pi)
        + np.linalg.slogdet(omega)[1]+logdiffuse+remainder@inverse@remainder)
    # Universal predictor of primitive shocks, including every measurement.
    r = loading.shape[1]
    saddle = np.block([[omega, design], [design.T, np.zeros((r, r))]])
    solve = np.linalg.inv(saddle)
    cross_eta = H[:, positions]
    eta_bridge = np.concatenate((cross_eta, np.zeros((n*p, r))), axis=1)
    state_bridge = np.concatenate((cross, loading), axis=1)
    eta_mean = eta_bridge@solve@np.concatenate((residual, np.zeros(r)))
    eta_cov = H-eta_bridge@solve@eta_bridge.T
    eta_state = -eta_bridge@solve@state_bridge.T
    return dict(mean=mean.reshape(n+1, m), covariance=covariance.reshape(n+1, m, n+1, m),
                ll=ll, measurement=eta_mean.reshape(n, p), measurement_cov=eta_cov,
                measurement_state=eta_state)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("rank", [1, 2])
def test_correlated_scheduled_diffuse_full_outputs_against_independent_gls(seed, rank):
    rng = np.random.default_rng(seed)
    n, m, p = 5, 2, 2
    y = tensor(rng.normal(size=(n, p)))
    y[1] = math.nan
    y[3, seed % 2] = math.nan
    Z = np.array([[1., .35], [-.25, 1.]])
    values = {key: tensor(value) for key, value in dict(
        a0=[.3, -.4], P_inf=np.diag([1., float(rank == 2)]), P_star=[[.2, .07], [.07, .4]],
        Z=np.array([Z+0.03*t for t in range(n)]),
        T=np.array([[[.9, .1], [0., .75+.02*t]] for t in range(n)]),
        Q=np.array([[[.2+.01*t, .04], [.04, .12]] for t in range(n)]),
        H=np.array([[[.5, .18], [.18, .4+.03*t]] for t in range(n)]),
        c=np.array([[.02*t, -.01] for t in range(n)]),
        d=np.array([[.03, -.02*t] for t in range(n)])).items()}
    expected = independent_joint(y, values)
    actual = smooth_exact_diffuse(y, values)
    np.testing.assert_allclose(actual["smoothed"], expected["mean"][:-1], atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(actual["smoothed_joint_covariance"], expected["covariance"], atol=3e-12, rtol=3e-12)
    assert float(actual["log_likelihood"]) == pytest.approx(expected["ll"], abs=3e-12)
    np.testing.assert_allclose(actual["measurement_disturbance"], expected["measurement"], atol=3e-12)
    np.testing.assert_allclose(actual["next_mean"], expected["mean"][-1], atol=3e-12)
    np.testing.assert_allclose(actual["next_covariance"], expected["covariance"][-1, :, -1, :], atol=3e-12)
    assert actual["observed_count"].tolist() == [2, 0, 2, 1, 2]
    assert sum(actual["diffuse_observation_count"]) == rank
    for t in range(n):
        T = values["T"][t].numpy()
        covariance = expected["covariance"]
        lag = covariance[t+1, :, t, :]
        process_cov = covariance[t+1, :, t+1, :]+T@covariance[t, :, t, :]@T.T-lag@T.T-T@lag.T
        np.testing.assert_allclose(actual["process_disturbance_covariance"][t], process_cov, atol=3e-12)
        np.testing.assert_allclose(actual["measurement_disturbance_covariance"][t],
                                  expected["measurement_cov"][t*p:(t+1)*p, t*p:(t+1)*p], atol=3e-12)
        np.testing.assert_allclose(actual["measurement_state_covariance"][t],
                                  expected["measurement_state"][t*p:(t+1)*p, t*m:(t+1)*m], atol=3e-12)
        cross = (expected["measurement_state"][t*p:(t+1)*p, (t+1)*m:(t+2)*m]
                 -expected["measurement_state"][t*p:(t+1)*p, t*m:(t+1)*m]@T.T).T
        np.testing.assert_allclose(actual["process_measurement_covariance"][t], cross, atol=3e-12)


@pytest.mark.parametrize("prefix", [1, 3, 4, 5, 6])
def test_every_filtered_prefix_and_terminal_matches_gls(prefix):
    y, values = local_level(prefix)
    expected = independent_joint(y, values)
    actual = filter_exact_diffuse(y, values)
    np.testing.assert_allclose(actual["filtered"][-1], expected["mean"][-2], atol=2e-12)
    np.testing.assert_allclose(actual["filtered_covariance"][-1], expected["covariance"][-2, :, -2, :], atol=2e-12)
    assert float(actual["log_likelihood"]) == pytest.approx(expected["ll"], abs=2e-12)


def test_leading_all_missing_date_does_not_consume_diffuse_rank_or_calendar():
    y, values = local_level()
    y[0] = math.nan
    actual = smooth_exact_diffuse(y, values)
    expected = independent_joint(y, values)
    assert actual["prior_diffuse_rank"][:3] == [1, 1, 1]
    assert actual["diffuse_observation_count"][:3] == [0, 0, 1]
    assert actual["log_likelihood_contributions"][:2].tolist() == [0., 0.]
    np.testing.assert_allclose(actual["smoothed_joint_covariance"], expected["covariance"], atol=2e-12)


@pytest.mark.parametrize("scale", [.25, 4., 9.])
def test_diffuse_prior_scale_and_arbitrary_mean_normalization(scale):
    y, values = local_level()
    baseline = smooth_exact_diffuse(y, values)
    changed = dict(values, P_inf=values["P_inf"]*scale, a0=values["a0"]+13.)
    actual = smooth_exact_diffuse(y, changed)
    np.testing.assert_allclose(actual["smoothed"], baseline["smoothed"], atol=2e-12)
    np.testing.assert_allclose(actual["smoothed_joint_covariance"], baseline["smoothed_joint_covariance"], atol=2e-12)
    assert float(actual["log_likelihood"]-baseline["log_likelihood"]) == pytest.approx(-.5*math.log(scale), abs=2e-12)


def test_zero_diffuse_rank_recovers_existing_proper_prior_full_moments():
    y, values = local_level()
    values.update(P_inf=tensor([[0.]]), P_star=tensor([[.8]]))
    actual = smooth_exact_diffuse(y, values)
    proper_values = {key: value for key, value in values.items() if key not in {"P_inf", "P_star"}}
    proper_values["P0"] = values["P_star"]
    expected = smooth(y, proper_values)
    for key in ("smoothed", "smoothed_covariance", "lag_one_covariance", "process_disturbance",
                "process_disturbance_covariance", "measurement_disturbance", "joint_disturbance_covariance",
                "next_mean", "next_covariance", "log_likelihood"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-12, rtol=2e-12)


def test_no_large_prior_approximation_for_exact_deterministic_measurement():
    y = tensor([[4.], [math.nan], [math.nan]])
    _, values = local_level()
    values.update(H=tensor([[0.]]), Q=tensor([[0.]]), c=tensor([0.]), d=tensor([0.]))
    actual = smooth_exact_diffuse(y, values)
    assert actual["smoothed"].flatten().tolist() == [4., 4., 4.]
    assert torch.count_nonzero(actual["smoothed_joint_covariance"]) == 0
    assert float(actual["log_likelihood"]) == 0.


@pytest.mark.parametrize("kind", ["unobserved", "annihilated", "singular", "negative", "nonsymmetric"])
def test_invalid_or_unidentified_diffuse_geometry_is_explicit(kind):
    y, values = local_level()
    if kind == "unobserved":
        y[:] = math.nan
    elif kind == "annihilated":
        y[0] = math.nan
        values["T"] = tensor([[0.]])
    elif kind == "singular":
        values.update(H=tensor([[0.]]), Q=tensor([[0.]]))
    elif kind == "negative":
        values["P_inf"] = tensor([[-1.]])
    else:
        y = tensor([[1., 2.], [2., 3.]])
        values = {key: tensor(value) for key, value in dict(a0=[0., 0.], P_inf=[[1., .2], [0., 1.]],
                    P_star=np.eye(2), T=np.eye(2), Z=np.eye(2), Q=np.eye(2), H=np.eye(2), c=[0., 0.], d=[0., 0.]).items()}
    with pytest.raises(AnalysisError):
        filter_exact_diffuse(y, values)


def test_rank_diagnostic_mode_retains_improper_terminal_explicitly():
    y, values = local_level()
    y[:] = math.nan
    actual = filter_exact_diffuse(y, values, require_identified=False)
    assert actual["final_diffuse_rank"] == 1
    assert actual["next_diffuse_covariance"].tolist() == [[1.]]


def test_autograd_score_hessian_matches_independent_likelihood_differences():
    y, values = local_level()
    parameters = tensor([math.log(.25), math.log(.6)]).requires_grad_()
    def objective(theta):
        return filter_exact_diffuse(y, dict(values, Q=theta[0].exp().reshape(1, 1),
            H=theta[1].exp().reshape(1, 1)), retain=False)["log_likelihood"]
    score = torch.autograd.functional.jacobian(objective, parameters).numpy()
    hessian = torch.autograd.functional.hessian(objective, parameters).numpy()
    theta = parameters.detach().numpy()
    def independent(theta):
        return independent_joint(y, dict(values, Q=tensor([[np.exp(theta[0])]]),
                                         H=tensor([[np.exp(theta[1])]])))["ll"]
    h = 1e-4
    eye = np.eye(2)*h
    expected_score = np.array([(independent(theta+e)-independent(theta-e))/(2*h) for e in eye])
    expected_hessian = np.array([[(independent(theta+e+f)-independent(theta+e-f)
        -independent(theta-e+f)+independent(theta-e-f))/(4*h*h) for f in eye] for e in eye])
    np.testing.assert_allclose(score, expected_score, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(hessian, expected_hessian, atol=3e-7, rtol=3e-7)


def test_saved_full_state_and_conditional_future_replay_no_optimizer(monkeypatch):
    y, values = local_level()
    state = save_diffuse_state(y, values)
    assert restore_diffuse_state(json.dumps(state)) == state
    from openecon.engines import optimize
    monkeypatch.setattr(optimize, "maximize_bfgs", lambda *a, **kw: pytest.fail("Replay invoked estimation"))
    paths = {key: values[key].expand((3, *values[key].shape)).clone() for key in ("T", "Z", "Q", "H", "c", "d")}
    actual = forecast_exact_diffuse(state, paths)
    expected_mean = np.array(state["output"]["next_mean"])
    expected_cov = np.array(state["output"]["next_covariance"])
    for i in range(3):
        np.testing.assert_allclose(actual["mean"][i], values["Z"].numpy()@expected_mean+values["d"].numpy())
        np.testing.assert_allclose(actual["covariance"][i], values["Z"].numpy()@expected_cov@values["Z"].numpy().T+values["H"].numpy())
        expected_mean = values["T"].numpy()@expected_mean+values["c"].numpy()
        expected_cov = values["T"].numpy()@expected_cov@values["T"].numpy().T+values["Q"].numpy()
    assert actual["parameter_uncertainty"] is False


@pytest.mark.parametrize("field", ["schema", "checksum", "moment", "mask", "extra"])
def test_saved_state_tampering_is_rejected_even_with_rehashed_outputs(field):
    y, values = local_level()
    state = copy.deepcopy(save_diffuse_state(y, values))
    if field == "schema":
        state["schema"] += ".unknown"
    elif field == "checksum":
        state["sha256"] = "0"*64
    elif field == "moment":
        state["output"]["smoothed"][0][0] += .01
    elif field == "mask":
        state["output"]["observed_mask"][1][0] = True
    else:
        state["extra"] = True
    if field in {"moment", "mask"}:
        payload = {key: value for key, value in state.items() if key != "sha256"}
        state["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(AnalysisError):
        restore_diffuse_state(state)


def test_admission_before_factorization_and_ambient_torch_state_preserved(monkeypatch):
    y, values = local_level()
    default_dtype, rng = torch.get_default_dtype(), torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch.linalg, "eigvalsh", lambda *a, **kw: pytest.fail("Factorization preceded budget admission"))
    with pytest.raises(AnalysisError, match="work"):
        smooth_exact_diffuse(y, values, max_work=1)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        filter_exact_diffuse(y[:1].expand(6000, 1), values)
    assert torch.get_default_dtype() == default_dtype
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
def test_explicit_cpu_float64_only(dtype):
    y, values = local_level()
    with pytest.raises(AnalysisError):
        filter_exact_diffuse(torch.zeros_like(y, dtype=dtype), values)


def test_proper_likelihood_is_distinct_from_diffuse_normalization():
    y, values = local_level(1)
    actual = filter_exact_diffuse(y, values)
    assert float(actual["log_likelihood"]) == 0.
    assert actual["proper_observation_count"] == [0]
    assert actual["diffuse_observation_count"] == [1]
    proper_values = {key: value for key, value in values.items() if key not in {"P_inf", "P_star"}}
    proper_values["P0"] = tensor([[1e12]])
    assert float(kalman(y, proper_values)["log_likelihood"]) < -14.


def test_rotated_diffuse_direction_and_measurement_permutation_are_invariant():
    y = tensor([[1., -.2], [math.nan, .4], [.3, .8]])
    _, source = local_level()
    values = dict(source, a0=tensor([0., .4]), P_star=tensor([[.3, .1], [.1, .5]]),
                  P_inf=tensor([[1., 2.], [2., 4.]]), T=tensor([[1., .2], [0., .9]]),
                  Z=tensor([[.1, 1.], [1., -.3]]), Q=tensor([[.3, .03], [.03, .2]]),
                  H=tensor([[.7, .2], [.2, .4]]), c=tensor([.1, .2]), d=tensor([.3, -.1]))
    actual = smooth_exact_diffuse(y, values)
    oracle = independent_joint(y, values)
    np.testing.assert_allclose(actual["smoothed"], oracle["mean"][:-1], atol=3e-12)
    permuted = dict(values, Z=values["Z"].flip(0), H=values["H"].flip(0).flip(1), d=values["d"].flip(0))
    changed = smooth_exact_diffuse(y.flip(1), permuted)
    for key in ("smoothed", "smoothed_joint_covariance", "log_likelihood"):
        torch.testing.assert_close(actual[key], changed[key], atol=3e-12, rtol=3e-12)


def test_proper_all_missing_smoother_retains_unconditional_prior():
    y, values = local_level()
    y[:] = math.nan
    values.update(P_inf=tensor([[0.]]), P_star=tensor([[.4]]))
    actual = smooth_exact_diffuse(y, values)
    np.testing.assert_allclose(actual["smoothed"].flatten(), .3+np.arange(6)*.1, atol=2e-12)
    np.testing.assert_allclose(actual["smoothed_covariance"].flatten(), .4+np.arange(6)*.25, atol=2e-12)
    assert float(actual["log_likelihood"]) == 0.


@pytest.mark.parametrize("kind", ["array_boolean", "array_null", "oversized", "smoothing_boolean"])
def test_saved_geometry_is_checked_before_tensor_materialization(kind, monkeypatch):
    y, values = local_level()
    state = save_diffuse_state(y, values)
    if kind == "array_boolean":
        state["system"]["P_inf"][0][0] = True
    elif kind == "array_null":
        state["system"]["T"][0][0] = None
    elif kind == "oversized":
        state["system"]["P_star"] = [[0.]*17 for _ in range(17)]
    else:
        state["smoothing"] = 1
    payload = {key: value for key, value in state.items() if key != "sha256"}
    state["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Malformed state materialized before geometry check"))
    with pytest.raises(AnalysisError):
        restore_diffuse_state(state)


def test_saved_state_is_portable_across_sufficient_ambient_workspace_budgets():
    y, values = local_level()
    state = save_diffuse_state(y, values)
    with use_workspace_budget(32):
        actual = restore_diffuse_state(state)
    assert actual["output"]["smoothed"] == state["output"]["smoothed"]
    assert actual["output"]["workspace"]["budget_bytes"] == 32*1024**2


@pytest.mark.parametrize("kind", ["boolean", "float", "zero", "too_high"])
def test_invalid_work_budgets_fail_explicitly(kind):
    y, values = local_level()
    budget = {"boolean": True, "float": 1e6, "zero": 0, "too_high": 50_000_001}[kind]
    with pytest.raises(AnalysisError):
        filter_exact_diffuse(y, values, max_work=budget)


@pytest.mark.parametrize("index_kind", ["mixed", "range", "multi", "category", "timezone"])
def test_public_tables_saved_row_identity_calendar_and_latex(index_kind):
    import pandas as pd
    from openecon.econometrics.core import TableSet

    index = {"mixed": pd.Index([1, "1", None, ("x", 2), 3., "end"], name="case", tupleize_cols=False),
             "range": pd.RangeIndex(4, 16, 2, name="case"),
             "multi": pd.MultiIndex.from_arrays([["a", "a", "b", "b", "c", "c"], [1, 2, 1, 2, 1, 2]], names=["group", "case"]),
             "category": pd.CategoricalIndex(["a", "b", "b", "a", "c", "a"], categories=["c", "b", "a", "unused"], ordered=True, name="case"),
             "timezone": pd.date_range("2023-01-01", periods=6, tz="Europe/Istanbul", name="case")}[index_kind]
    y, values = local_level()
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=6, freq="MS"), "y": y.flatten().numpy()}, index=index)
    system = {key: value.tolist() for key, value in values.items()}
    result = sspace_diffuse(frame, "y", system=system, time="time", state_names=["level"])
    assert isinstance(result, DiffuseResult) and isinstance(result, TableSet)
    restored = diffuse_restore(result.to_json())
    for key in result:
        pd.testing.assert_frame_equal(result[key], restored[key])
        pd.testing.assert_index_equal(result[key].index, index)
    assert result["likelihood"]["observed_cells"].tolist() == [1, 0, 1, 1, 1, 1]
    assert "conditional" in result.to_latex()
    paths = {key: [system[key]]*2 for key in ("T", "Z", "Q", "H", "c", "d")}
    future = diffuse_forecast(restored, future_system=paths, future_time=pd.date_range("2024-07-01", periods=2, freq="MS"))
    assert isinstance(future, TableSet)
    assert future["forecast"]["horizon"].tolist() == [1, 2]
    assert future.attrs["parameter_uncertainty"] is False
    assert future.attrs["full_moments"]["covariance"]


@pytest.mark.parametrize("kind", ["unsorted", "gap", "missing", "duplicate", "response_bool", "raise_missing", "parameter"])
def test_public_adapter_refuses_unsupported_calendar_and_sample_domains(kind):
    import pandas as pd
    y, values = local_level()
    frame = pd.DataFrame({"time": range(6), "y": y.flatten().numpy()})
    system = {key: value.tolist() for key, value in values.items()}
    missing = "mask"
    if kind == "unsorted":
        frame["time"] = [2, 1, 0, 3, 4, 5]
    elif kind == "gap":
        frame["time"] = [0, 1, 2, 4, 5, 6]
    elif kind == "missing":
        frame.loc[1, "time"] = math.nan
    elif kind == "duplicate":
        frame.loc[1, "time"] = 0
    elif kind == "response_bool":
        frame["y"] = True
    elif kind == "raise_missing":
        missing = "raise"
    else:
        system["parameters"] = [{"name": "q"}]
    with pytest.raises(AnalysisError):
        sspace_diffuse(frame, "y", system=system, time="time", missing=missing)


def test_public_restore_rejects_rehashed_source_calendar_or_index_forgery():
    import pandas as pd
    y, values = local_level()
    result = sspace_diffuse(pd.DataFrame({"t": range(6), "y": y.flatten().numpy()}), "y",
                            system={k: v.tolist() for k, v in values.items()}, time="t")
    record = json.loads(result.to_json())
    record["source"]["calendar"]["step"] = 2.
    payload = {k: v for k, v in record.items() if k != "sha256"}
    record["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(AnalysisError):
        diffuse_restore(record)


@pytest.mark.parametrize("kind", ["oversized_mapping", "oversized_frame", "generator", "dataset", "misaligned_series", "work"])
def test_public_resident_admission_precedes_dataframe_coercion(kind, monkeypatch):
    import pandas as pd
    import openecon.analysis as analysis

    _, values = local_level()
    system = {k: v.tolist() for k, v in values.items()}
    class Dataset:
        def collect(self):
            pytest.fail("Unsupported Dataset was collected")
    data = {"y": [1.]*20001} if kind == "oversized_mapping" else pd.DataFrame({"y": [1.]*20001}) if kind == "oversized_frame" else (v for v in [{"y": 1.}]) if kind == "generator" else Dataset() if kind == "dataset" else {"y": pd.Series([1., 2.], index=[0, 1]), "t": pd.Series([0, 1], index=[1, 2])} if kind == "misaligned_series" else {"y": [1., 2.]}
    monkeypatch.setattr(analysis, "_coerce_frame", lambda *a: pytest.fail("DataFrame coercion preceded admission"))
    with pytest.raises(AnalysisError):
        sspace_diffuse(data, "y", system=system, time="t" if kind == "misaligned_series" else None,
                       max_work=1 if kind == "work" else 50_000_000)


def test_public_mapping_projects_declared_columns_without_consuming_unrelated_iterator():
    _, values = local_level()
    def unrelated():
        pytest.fail("Unrelated mapping column was consumed")
        yield 0
    result = sspace_diffuse({"y": [1., 2.], "unrelated": unrelated()}, "y",
                            system={k: v.tolist() for k, v in values.items()})
    assert result.attrs["n_periods"] == 2


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "UInt64"])
def test_nonexact_integer_measurements_are_refused_before_float_conversion(dtype, monkeypatch):
    import pandas as pd
    import openecon.analysis as analysis

    _, values = local_level()
    frame = pd.DataFrame({"y": pd.Series([2**53+1, 2**53+3], dtype=dtype)})
    monkeypatch.setattr(analysis, "_coerce_frame", lambda *a: pytest.fail("Lossy input reached conversion"))
    with pytest.raises(AnalysisError, match="exactly representable"):
        sspace_diffuse(frame, "y", system={k: v.tolist() for k, v in values.items()})


@pytest.mark.parametrize("dtype", ["UInt64", "Float32", "Float64", "float32"])
def test_numeric_source_dtype_replays_exactly_including_nullable_measurements(dtype):
    import pandas as pd
    _, values = local_level()
    raw = [2**53+2, None, 2**53+4] if dtype == "UInt64" else [1.125, None, 2.5]
    frame = pd.DataFrame({"y": pd.Series(raw, dtype=dtype)})
    result = sspace_diffuse(frame, "y", system={k: v.tolist() for k, v in values.items()})
    replay = diffuse_restore(result.to_json())
    pd.testing.assert_frame_equal(result["smoothed"], replay["smoothed"])
    assert replay.attrs["diffuse_result"]["source"]["response_dtypes"] == [dtype]


@pytest.mark.parametrize("dtype", ["object", "bool", "complex128", "not-a-dtype", "int8", "float16", "int64"])
def test_public_replay_refuses_forged_numeric_dtype_or_unrepresentable_values(dtype):
    import pandas as pd
    y, values = local_level()
    record = sspace_diffuse(pd.DataFrame({"y": y.flatten().numpy()}), "y",
                            system={k: v.tolist() for k, v in values.items()}).attrs["diffuse_result"]
    record = copy.deepcopy(record)
    record["source"]["response_dtypes"] = [dtype]
    payload = {k: v for k, v in record.items() if k != "sha256"}
    record["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(AnalysisError):
        diffuse_restore(record)


@pytest.mark.parametrize("kind", ["name", "label", "unused_category", "tuple_depth"])
def test_public_index_metadata_is_bounded_before_encoding_or_coercion(kind, monkeypatch):
    import pandas as pd
    import openecon.analysis as analysis
    import openecon.econometrics.postest.index_codec as codec
    _, values = local_level()
    item = "small"
    if kind == "tuple_depth":
        for _ in range(18):
            item = (item,)
    index = pd.Index([item], name="x"*1025 if kind == "name" else None, tupleize_cols=False)
    if kind == "label":
        index = pd.Index(["x"*1025])
    elif kind == "unused_category":
        index = pd.CategoricalIndex(["ok"], categories=["ok", "x"*1025])
    frame = pd.DataFrame({"y": [1.]}, index=index)
    monkeypatch.setattr(analysis, "_coerce_frame", lambda *a: pytest.fail("Unbounded metadata reached coercion"))
    monkeypatch.setattr(codec, "encode", lambda *a: pytest.fail("Unbounded metadata reached encoding"))
    with pytest.raises(AnalysisError):
        sspace_diffuse(frame, "y", system={k: v.tolist() for k, v in values.items()})


def test_replay_envelopes_precede_parser_digest_tensor_and_index_construction(monkeypatch):
    import openecon.econometrics.tsworkflows.ssdiffuse as module
    y, values = local_level()
    core = save_diffuse_state(y, values)
    forged = copy.deepcopy(core)
    forged["output"]["oversized"] = "x"*4097
    monkeypatch.setattr(module, "_digest", lambda *a: pytest.fail("Unbounded object reached digest"))
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Unbounded object reached tensor construction"))
    with pytest.raises(AnalysisError):
        restore_diffuse_state(forged)
    monkeypatch.setattr(module, "MAX_STATE_BYTES", 100)
    monkeypatch.setattr(json, "loads", lambda *a, **kw: pytest.fail("Unbounded text reached JSON parser"))
    with pytest.raises(AnalysisError):
        restore_diffuse_state(" "*101)


def test_improper_filtered_mean_and_intervals_are_not_presented_as_conditional_moments():
    import pandas as pd
    _, values = local_level()
    result = sspace_diffuse(pd.DataFrame({"y": [math.nan, 1., 2.]}), "y",
                            system={k: v.tolist() for k, v in values.items()})
    row = result["filtered"].iloc[0]
    assert all(pd.isna(row[k]) for k in ("mean", "conditional_sd", "conditional_variance", "conditional_lo", "conditional_hi"))
    assert result.attrs["diffuse_result"]["state"]["output"]["filtered"][0] == [.3]


@pytest.mark.parametrize("rescale", [1., 1e8])
def test_diffuse_rank_does_not_erase_a_small_unobserved_coordinate(rescale):
    values = {k: tensor(v) for k, v in dict(a0=[0., 0.], P_inf=[[1., 0.], [0., 1e-16*rescale**2]],
        P_star=[[0., 0.], [0., 0.]], T=[[1., 0.], [0., 1.]], Z=[[1., 0.]],
        Q=[[.1, 0.], [0., .1*rescale**2]], H=[[1.]], c=[0., 0.], d=[0.]).items()}
    y = tensor([[1.], [2.]])
    for routine in (filter_exact_diffuse, smooth_exact_diffuse):
        with pytest.raises(AnalysisError, match="identify"):
            routine(y, values)
    diagnostic = filter_exact_diffuse(y, values, require_identified=False)
    assert diagnostic["initial_diffuse_rank"] == 2
    assert diagnostic["final_diffuse_rank"] == 1
    assert float(diagnostic["next_diffuse_covariance"][1, 1]) == pytest.approx(1e-16*rescale**2, rel=2e-15, abs=0.)


def test_small_coordinate_identified_exact_diffuse_closed_form():
    values = {k: tensor(v) for k, v in dict(a0=[0., 0.], P_inf=[[1., 0.], [0., 1e-16]],
        P_star=[[0., 0.], [0., 0.]], T=[[1., 0.], [0., 1.]], Z=[[1., 0.], [0., 1.]],
        Q=[[.1, 0.], [0., .1]], H=[[1., 0.], [0., 1.]], c=[0., 0.], d=[0., 0.]).items()}
    actual = smooth_exact_diffuse(tensor([[1., 2.]]), values)
    assert actual["initial_diffuse_rank"] == 2 and actual["final_diffuse_rank"] == 0
    torch.testing.assert_close(actual["smoothed"], tensor([[1., 2.]]), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual["smoothed_covariance"], tensor([[[1., 0.], [0., 1.]]]), atol=1e-12, rtol=1e-12)
    assert float(actual["log_likelihood"]) == pytest.approx(-.5*math.log(1e-16))


def test_unresolved_near_collinear_initial_diffuse_rank_is_refused():
    _, values = local_level()
    values = dict(values, a0=tensor([0., 0.]), P_inf=tensor([[1., 1.-1e-16], [1.-1e-16, 1.]]),
                  P_star=tensor([[0., 0.], [0., 0.]]), T=torch.eye(2, dtype=torch.float64),
                  Z=torch.eye(2, dtype=torch.float64), H=torch.eye(2, dtype=torch.float64),
                  Q=torch.eye(2, dtype=torch.float64)*.1, c=tensor([0., 0.]), d=tensor([0., 0.]))
    with pytest.raises(AnalysisError, match="rank cannot be resolved"):
        smooth_exact_diffuse(tensor([[1., 2.]]), values)


@pytest.mark.parametrize("key", ["H", "Q", "P_star", "P_inf"])
def test_negative_small_coordinate_covariance_input_is_never_admitted(key):
    values = {k: tensor(v) for k, v in dict(a0=[0., 0.], P_inf=[[0., 0.], [0., 0.]],
        P_star=[[1., 0.], [0., 1.]], T=[[1., 0.], [0., 1.]], Z=[[1., 0.], [0., 1.]],
        Q=[[.1, 0.], [0., .1]], H=[[1., 0.], [0., 1.]], c=[0., 0.], d=[0., 0.]).items()}
    values[key] = tensor([[1., 0.], [0., -1e-16]])
    with pytest.raises(AnalysisError, match="negative diagonal"):
        smooth_exact_diffuse(tensor([[1., 2.]]), values)


@pytest.mark.parametrize("field", ["mask", "rank", "calendar"])
def test_rehashed_boolean_integer_cached_metadata_substitution_is_refused(field):
    import pandas as pd
    y, values = local_level()
    record = sspace_diffuse(pd.DataFrame({"y": y.flatten().numpy(), "t": range(len(y))}), "y",
                            system={k: v.tolist() for k, v in values.items()}, time="t").attrs["diffuse_result"]
    record = copy.deepcopy(record)
    if field == "mask":
        record["state"]["output"]["observed_mask"][0][0] = 1
    elif field == "rank":
        record["state"]["output"]["initial_diffuse_rank"] = True
    else:
        record["source"]["calendar"]["step"] = True
    if field != "calendar":
        core = record["state"]
        core["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in core.items() if k != "sha256"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    record["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != "sha256"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    with pytest.raises(AnalysisError):
        diffuse_restore(record)


def test_calendar_integer_spacing_is_checked_without_float_rounding():
    import pandas as pd
    _, values = local_level()
    with pytest.raises(AnalysisError, match="constant positive spacing"):
        sspace_diffuse(pd.DataFrame({"y": [1., 2., 3.], "t": [0, 2**53, 2**54+1]}), "y",
                        system={k: v.tolist() for k, v in values.items()}, time="t")


def test_datetime_calendar_with_undeclared_two_date_frequency_refuses_at_fit():
    import pandas as pd
    _, values = local_level()
    with pytest.raises(AnalysisError, match="at least three"):
        sspace_diffuse(pd.DataFrame({"y": [1., 2.], "t": pd.date_range("2024-01-01", periods=2, freq="MS")}), "y",
                        system={k: v.tolist() for k, v in values.items()}, time="t")


def test_future_label_generator_is_refused_without_consumption():
    import pandas as pd
    _, values = local_level()
    system = {k: v.tolist() for k, v in values.items()}
    result = sspace_diffuse(pd.DataFrame({"y": [1., 2., 3.]}), "y", system=system)
    def labels():
        pytest.fail("Unbounded future labels were consumed")
        yield 0
    with pytest.raises(AnalysisError, match="resident"):
        diffuse_forecast(result, future_system={k: [system[k]]*2 for k in ("T", "Z", "Q", "H", "c", "d")}, future_time=labels())


@pytest.mark.parametrize("budget", ["work", "workspace"])
def test_low_level_geometry_admission_precedes_all_materialized_numeric_scans(budget, monkeypatch):
    import openecon.econometrics.tsworkflows.ssdiffuse as module
    y, values = local_level()
    y = y[:1].expand(6000, 1)
    values = dict(values, T=values["T"].expand(6000, 1, 1))
    for name in ("isfinite", "isinf", "isnan"):
        monkeypatch.setattr(torch, name, lambda *a, **kw: pytest.fail("Numeric scan preceded tensor/work admission"))
    monkeypatch.setattr(module, "_covariance", lambda *a: pytest.fail("Covariance validation preceded admission"))
    if budget == "work":
        with pytest.raises(AnalysisError, match="work"):
            filter_exact_diffuse(y, values, max_work=1)
    else:
        with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
            filter_exact_diffuse(y, values)
