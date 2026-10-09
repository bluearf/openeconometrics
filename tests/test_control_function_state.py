"""Complete CF replay, physical alignment, hostile state and bounded prediction."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.control_function.state import digest, typed_cluster_codes, validate_state
from openecon.resources import use_workspace_budget


def source(n=140, *, scale=1.0):
    rng = np.random.default_rng(481)
    w, z, u, error = rng.normal(size=(4, n))
    d = 0.5*w + 0.8*z + u
    y = scale*(0.2 + 0.4*w + 0.7*d + 0.3*u + error)
    return pd.DataFrame(dict(y=y, d=d, w=w, z=z))


def fitted(data=None, **kwargs):
    return oe.cfregress(data=source() if data is None else data, y="y", endogenous="d",
                        x=["w"], instruments=["z"], **kwargs)


def rebound(result):
    state = result.extra["control_function_state"]
    state["integrity_sha256"] = digest(state)
    result.provenance["control_function_state_sha256"] = state["integrity_sha256"]
    return result


@pytest.mark.parametrize("representation", ["object", "mapping", "json"])
def test_complete_saved_replay_full_rows_and_owned_output(representation):
    data = source(510)
    data.index = pd.MultiIndex.from_arrays([["one", "two", "one"]*170, np.arange(510)//2], names=["group", "row"])
    result = fitted(data)
    before = result.model_dump_json()
    supplied = {"object": result, "mapping": result.model_dump(mode="json"), "json": before}[representation]
    table = oe.cf_predict(result=supplied)
    assert len(result.predictions) == 400 and len(table) == 510
    pd.testing.assert_index_equal(table.index, data.index)
    np.testing.assert_allclose(table["mean"], result.extra["control_function_state"]["fitted"], rtol=1e-13)
    table.attrs["joint_parameter_covariance"][0][0] = -1
    table["mean"] = 0
    assert result.model_dump_json() == before


def test_independent_full_covariance_query_delta_and_normal_ci():
    result = fitted()
    query = pd.DataFrame(dict(d=[0.4, 0.2, -0.8], w=[0.1, -0.2, 0.4], z=[0.8, -0.3, 0.5]), index=[4, 4, -1])
    before = query.copy(deep=True)
    output = oe.cf_predict(result=result, data=query, alpha=0.1)
    state = result.extra["control_function_state"]
    gamma, beta, covariance = (np.asarray(state[name]) for name in ["gamma", "beta", "joint_covariance"])
    z = np.column_stack([np.ones(3), query.w, query.z])
    x = np.column_stack([np.ones(3), query.w, query.d])
    residual = query.d.to_numpy()-z@gamma
    q = np.column_stack([x, residual])
    jacobian = np.column_stack([-beta[-1]*z, q])
    eta, variance = q@beta, np.einsum("ni,ij,nj->n", jacobian, covariance, jacobian)
    np.testing.assert_allclose(output["mean"], eta, atol=1e-13)
    np.testing.assert_allclose(output.std_error, np.sqrt(variance), rtol=1e-12)
    np.testing.assert_allclose(output.ci_low, eta-stats.norm.ppf(0.95)*np.sqrt(variance), atol=1e-13)
    np.testing.assert_allclose(output.attrs["mean_jacobian"], jacobian, rtol=1e-13)
    first = len(gamma)
    naive = np.einsum("ni,ij,nj->n", q, covariance[first:, first:], q)
    assert np.max(np.abs(variance-naive)) > 1e-4
    pd.testing.assert_frame_equal(query, before)
    pd.testing.assert_index_equal(output.index, query.index)


def test_missing_fitted_and_new_query_physical_positions_and_typed_index():
    data = source()
    data.loc[[3, 11], "z"] = np.nan
    data.index = pd.date_range("2023-01-01", periods=len(data), tz="UTC", name="day")
    result = fitted(data, missing="drop")
    output = oe.cf_predict(result=result)
    expected = [i for i in range(len(data)) if i not in [3, 11]]
    assert output.position.tolist() == expected
    pd.testing.assert_index_equal(output.index, data.index.take(expected))
    query = data.iloc[:8].drop(columns=["y"]).copy()
    query.iloc[0, query.columns.get_loc("w")] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        oe.cf_predict(result=result, data=query)
    dropped = oe.cf_predict(result=result, data=query, missing="drop")
    assert dropped.position.tolist() == [1, 2, 4, 5, 6, 7]
    pd.testing.assert_index_equal(dropped.index, query.index.take([1, 2, 4, 5, 6, 7]))


def test_single_cluster_list_and_typed_bool_integer_labels_are_distinct():
    data = source(160)
    labels = [True, 1, False, 0, *[f"c{i}" for i in range(36)]]*4
    data["cluster"] = pd.Series(labels, dtype=object)
    codes, count = typed_cluster_codes(data.cluster)
    assert codes[:4].tolist() == [0, 1, 2, 3] and count == 40
    result = fitted(data, covariance="cluster", cluster=["cluster"])
    assert result.extra["control_function_state"]["cluster_codes"] == codes.tolist()
    assert result.inference["cluster_count"] == count
    assert len(oe.cf_predict(result=result.model_dump_json())) == len(data)


@pytest.mark.parametrize("field", ["gamma", "beta", "residual", "design", "fitted", "row_scores",
                                  "bread", "meat", "joint_covariance", "scalar_score", "scalar_derivative"])
def test_coherently_rehashed_numerical_tampering_refuses(field):
    result = fitted().model_copy(deep=True)
    state = result.extra["control_function_state"]
    if isinstance(state[field][0], list):
        state[field][0][0] += 0.2
    else:
        state[field][0] += 0.2
    rebound(result)
    with pytest.raises(AnalysisError):
        oe.cf_predict(result=result)


@pytest.mark.parametrize("field", ["kind", "parameter_order", "source_columns", "sample_positions", "source_index",
                                  "working_dispersion", "optimizer", "schema"])
def test_rehashed_metadata_or_geometry_tampering_refuses(field):
    result = fitted().model_copy(deep=True)
    state = result.extra["control_function_state"]
    if field == "kind":
        state[field] = "logit"
    elif field == "parameter_order":
        state[field][0]["equation"] = "outcome"
    elif field == "source_columns":
        state[field][0]["values"][0]["value"] += 0.2
    elif field == "sample_positions":
        state[field][1] = 0
    elif field == "source_index":
        state[field]["start"] = 1
        state[field]["stop"] += 1
        # Changing the index to a new valid index changes state identity; the
        # unchanged separately retained source hash cannot authenticate labels.
        # The outer provenance checksum intentionally stays on the old identity.
        state["integrity_sha256"] = digest(state)
        with pytest.raises(AnalysisError, match="integrity"):
            oe.cf_predict(result=result)
        return
    elif field == "working_dispersion":
        state[field] = 2.0
    elif field == "optimizer":
        state[field]["tolerance"] = 1.0
    else:
        state[field] = "unsupported"
    rebound(result)
    with pytest.raises(AnalysisError):
        oe.cf_predict(result=result)


@pytest.mark.parametrize("corruption", ["zero_se", "double_se", "zero_covariance", "double_covariance"])
def test_tiny_outcome_units_do_not_hide_reported_inference_corruption(corruption):
    result = fitted(source(scale=1e-12))
    validate_state(result)
    bad = result.model_copy(deep=True)
    for i in range(len(bad.extra["control_function_state"]["gamma"]), len(bad.coefficients)):
        if corruption.endswith("se"):
            bad.coefficients[i].std_error *= 0 if corruption.startswith("zero") else 2
        else:
            for j in range(len(bad.coefficients)):
                factor = 0 if corruption.startswith("zero") else 2
                bad.covariance_matrix[i][j] *= factor
                bad.covariance_matrix[j][i] *= factor
    with pytest.raises(AnalysisError):
        validate_state(bad)


@pytest.mark.parametrize("options", [{"alpha": True}, {"alpha": float("nan")}, {"alpha": 0},
                                    {"missing": []}, {"missing": "ignore"}, {"max_work": True},
                                    {"max_work": 1}, {"max_work": 10000000001}])
def test_prediction_options_refuse(options):
    with pytest.raises(AnalysisError):
        oe.cf_predict(result=fitted(), **options)


@pytest.mark.parametrize("value", [True, complex(1, 2), float("inf"), 10**500])
def test_saved_raw_numeric_and_nonfinite_guards(value):
    result = fitted().model_copy(deep=True)
    result.extra["control_function_state"]["beta"][0] = value
    with pytest.raises(AnalysisError):
        oe.cf_predict(result=result)


def test_hash_corruption_preview_and_direct_mutated_spec_refuse():
    result = fitted()
    bad = result.model_copy(deep=True)
    bad.extra["control_function_state"]["beta"][0] += 0.01
    with pytest.raises(AnalysisError, match="checksum"):
        validate_state(bad)
    bad = result.model_copy(deep=True)
    bad.predictions = []
    with pytest.raises(AnalysisError, match="preview"):
        validate_state(bad)
    bad = result.model_copy(deep=True)
    bad.spec.options["tolerance"] = 1e10
    bad.extra["control_function_state"]["spec"] = bad.spec.model_dump(mode="json")
    rebound(bad)
    with pytest.raises(AnalysisError):
        validate_state(bad)


def test_admission_before_tensor_copies_current_budget_and_no_saved_budget_bypass(monkeypatch):
    result = fitted(source(300))
    def forbidden(*args, **kwargs):
        pytest.fail("Numerical allocation occurred before bounded refusal.")
    monkeypatch.setattr(torch, "tensor", forbidden)
    with pytest.raises(AnalysisError, match="max_work"):
        oe.cf_predict(result=result, max_work=1)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="budget"):
        oe.cf_predict(result=result)
    malformed = result.model_copy(deep=True)
    malformed.extra["control_function_state"]["z"] = [[[[1.0]]]]
    with pytest.raises(AnalysisError):
        oe.cf_predict(result=malformed)


def test_saved_replay_keeps_ambient_dtype_device_and_rng():
    result = fitted()
    before = result.model_dump_json()
    rng, dtype = torch.get_rng_state().clone(), torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            validate_state(result)
            output = oe.cf_predict(result=before)
            assert torch.empty(0).device.type == "meta"
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(rng, torch.get_rng_state())
        assert len(output) == result.nobs
    finally:
        torch.set_default_dtype(dtype)


def test_probability_ci_is_link_transformed_and_query_numerical_domain_refuses():
    data = source(220)
    rng = np.random.default_rng(913)
    data["y"] = rng.binomial(1, special.expit(-0.2+0.3*data.w+0.2*data.d))
    result = oe.cflogit(data=data, y="y", endogenous="d", x=["w"], instruments=["z"])
    output = oe.cf_predict(result=result, data=data.drop(columns="y"), alpha=0.1)
    expected = special.expit(output.eta-stats.norm.ppf(0.95)*output.eta_std_error)
    np.testing.assert_allclose(output.ci_low, expected, atol=1e-14)
    assert bool(((output.ci_low > 0) & (output.ci_high < 1)).all())
    query = data.drop(columns="y").iloc[:2].copy()
    query.loc[:, "d"] = 1e300
    with pytest.raises(AnalysisError):
        oe.cf_predict(result=result, data=query)


def test_prediction_inputs_refuse_missing_complex_duplicate_and_oversized_rows():
    result = fitted()
    inputs = [source().drop(columns="z"), source(5001), {"d": [1], "w": [1, 2], "z": [1]},
              pd.DataFrame([[1, 2, 3, 4]], columns=["d", "w", "z", "z"])]
    complex_data = source().drop(columns="y").astype(complex)
    inputs.append(complex_data)
    for query in inputs:
        with pytest.raises(AnalysisError):
            oe.cf_predict(result=result, data=query)


def test_saved_mapping_metadata_budget_before_model_copy(monkeypatch):
    saved = json.loads(fitted().model_dump_json())
    saved["provenance"]["oversized"] = "a" * 65537
    monkeypatch.setattr(oe, "cfregress", lambda **kwargs: pytest.fail("Replay cannot fit."))
    with pytest.raises(AnalysisError, match="metadata strings"):
        oe.cf_predict(result=saved)


def test_no_intercept_exact_zero_design_has_exact_zero_conditional_uncertainty():
    result = fitted(intercept=False)
    output = oe.cf_predict(result=result, data=dict(d=[0.0], w=[0.0], z=[0.0]))
    assert output["mean"].tolist() == output.std_error.tolist() == output.ci_low.tolist() == output.ci_high.tolist() == [0.0]


def test_coherent_tiny_gaussian_non_ols_outcome_parameters_refuse():
    from openecon.econometrics.control_function.kernels import evaluate_joint
    result = fitted(source(scale=1e-12)).model_copy(deep=True)
    state = result.extra["control_function_state"]
    arrays = [torch.tensor(state[name], dtype=torch.float64) for name in ("z", "x", "d", "y", "gamma", "beta")]
    arrays[-1].zero_()
    evaluated = evaluate_joint(*arrays[:4], "gaussian", *arrays[4:])
    for name, value in evaluated.items():
        if name in state and isinstance(value, torch.Tensor):
            state[name] = value.tolist()
    state["criterion"] = float(evaluated["criterion"])
    params = torch.cat(arrays[4:]).numpy()
    covariance = evaluated["joint_covariance"].numpy()
    se = np.sqrt(np.diag(covariance))
    result.covariance_matrix = covariance.tolist()
    for i, coefficient in enumerate(result.coefficients):
        coefficient.estimate, coefficient.std_error = params[i], se[i]
        coefficient.statistic = params[i]/se[i]
        coefficient.p_value = 2*stats.norm.sf(abs(coefficient.statistic))
        coefficient.ci_low = params[i]-stats.norm.ppf(0.975)*se[i]
        coefficient.ci_high = params[i]+stats.norm.ppf(0.975)*se[i]
    result.metrics["working_criterion"] = state["criterion"]
    for preview in result.predictions:
        i = result.sample_positions.index(preview["row"])
        preview["fitted"] = state["fitted"][i]
        preview["residual"] = preview["observed"]-preview["fitted"]
    rebound(result)
    with pytest.raises(AnalysisError, match="Gaussian outcome OLS Beta"):
        validate_state(result)


@pytest.mark.parametrize("api", ["cflogit", "cffraclogit"])
def test_complete_replay_certificate_work_admitted_before_buffers_and_forwarded(api, monkeypatch):
    from openecon.econometrics.control_function import kernels
    from openecon.econometrics.control_function.state import _admit
    data = source(120)
    rng = np.random.default_rng(711)
    probability = special.expit(-0.2+0.2*data.d+0.1*data.w)
    data["y"] = rng.binomial(1, probability) if api == "cflogit" else rng.beta(2, 3, len(data))
    result = getattr(oe, api)(data=data, y="y", endogenous="d", x=["w"], instruments=["z"])
    state = result.extra["control_function_state"]
    n, kz, kx, work, _ = _admit(state, 10000000000)
    analytic = 32*(2*n*(kz+kx+1)**2+(kz+kx+1)**3)
    certificate = kernels.certificate_work(n, kx+1, state["kind"])
    assert certificate > 0 and work == analytic+certificate
    allocate = torch.tensor
    def forbidden(*args, **kwargs):
        pytest.fail("Replay allocated a tensor before refusing the certificate work envelope.")
    monkeypatch.setattr(torch, "tensor", forbidden)
    with pytest.raises(AnalysisError, match="max_work"):
        validate_state(result, max_work=work-1)
    monkeypatch.setattr(torch, "tensor", allocate)
    evaluate, admitted = kernels.evaluate_joint, []
    def checked(*args, **kwargs):
        admitted.append(kwargs["max_work"])
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(kernels, "evaluate_joint", checked)
    validate_state(result, max_work=work+100)
    assert admitted == [work+100]
    query_work = 32*n*((kz+kx+1)**2+(kz+kx+1)+kz+kx)
    with pytest.raises(AnalysisError, match="max_work"):
        oe.cf_predict(result=result, max_work=work+query_work-1)
    output = oe.cf_predict(result=result, max_work=work+query_work)
    assert output.attrs["work_estimate"] == work+query_work
