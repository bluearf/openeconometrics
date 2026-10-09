"""Admission and scientific restoration contracts for dependent meta fits."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.meta import dependent as core
from openecon.econometrics.meta import dependent_kernels as kernels
from openecon.resources import use_workspace_budget


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def data_fixture(n=16):
    rng = np.random.default_rng(192329)
    groups = np.repeat(np.arange(n//2),2)
    x = rng.normal(size=n)
    y = .5-.8*x+rng.normal(size=n//2)[groups]+rng.normal(scale=.3,size=n)
    frame = pd.DataFrame({"id":[f"e{j}" for j in range(n)],"g":groups,"response":y,"x":x},
                         index=[f"physical{j//2}" for j in range(n)])
    s = np.zeros((n,n))
    for j in range(0,n,2):
        s[j:j+2,j:j+2] = [[.2,.08],[.08,.3]]
    covariance = pd.DataFrame(s,index=frame.id,columns=frame.id)
    return frame,covariance


def fitted(*, model="study", moderators=None):
    frame,s = data_fixture()
    return oe.meta_dependent(data=frame,covariance=s,study="g",effect="id",yi="response",
                             moderators=["x"] if moderators is None else moderators,model=model)


@pytest.mark.parametrize("method",["ML","REML","common"])
def test_common_normalizes_criterion_and_uses_full_known_covariance(method):
    frame,s = data_fixture()
    result = oe.meta_dependent(data=frame,covariance=s,study="g",effect="id",yi="response",method=method)
    inverse = np.linalg.inv(s.to_numpy())
    ones = np.ones(len(frame))
    variance = 1/(ones@inverse@ones)
    estimate = variance*(ones@inverse@frame.response.to_numpy())
    assert result["coefficients"].iloc[0].estimate == pytest.approx(estimate,rel=1e-12)
    assert result["covariance"].iloc[0,0] == pytest.approx(variance,rel=1e-12)
    state = result.attrs["prediction_state"]
    assert state["method"]=="common" and state["tau2"]==0
    assert state["roles"]=={"effect":"id","study":"g","yi":"response"}


@pytest.mark.parametrize("model",["common","effect","study"])
def test_complete_json_restoration_replays_without_optimizer_or_mutation(monkeypatch,model):
    result = fitted(model=model)
    state = json.loads(json.dumps(result.attrs["prediction_state"],allow_nan=False))
    before = copy.deepcopy(state)
    def forbidden(*args,**kwargs):
        raise AssertionError("saved state cannot rerun the variance optimizer")
    monkeypatch.setattr(kernels,"fit",forbidden)
    restored = core.make_result(state)
    for name in result:
        pd.testing.assert_frame_equal(result[name],restored[name])
    assert state==before


def test_coherent_gls_at_a_different_variance_still_fails_saved_fit_stationarity():
    result = fitted(model="effect")
    state,values = core.load_state(result)
    state["tau2"] = state["tau2"]*2+1
    point = kernels.at_variance(values["x"],values["y"],values["s"],values["cluster_index"],
                                state["model"],state["intercept"],state["tau2"])
    state.update(beta=point["beta"].tolist(),fit_covariance=point["covariance"].tolist(),
                 covariance=point["covariance"].tolist())
    for key in ("rss","loglik_ml","loglik_reml","variance_scale"):
        state["fit"][key] = point[key]
    state["solver"]["score_residual"] = point["score_reml"]*(1+state["tau2"]/point["variance_scale"])/state["n_effects"]
    state["solver"]["boundary"] = False
    with pytest.raises(AnalysisError,match="coherent"):
        core.load_state(core.seal_state(state))


def test_tiny_cross_study_covariance_is_never_silently_discarded():
    frame,s = data_fixture()
    s.iloc[0,2] = s.iloc[2,0] = 1e-20
    with pytest.raises(AnalysisError) as error:
        oe.meta_dependent(data=frame,covariance=s,study="g",effect="id",yi="response")
    assert error.value.code=="unsupported_dependence"


def test_tiny_within_study_asymmetry_is_rejected_without_symmetrization():
    frame,s = data_fixture()
    s.iloc[0,1] = np.nextafter(s.iloc[0,1],np.inf)
    with pytest.raises(AnalysisError) as error:
        oe.meta_dependent(data=frame,covariance=s,study="g",effect="id",yi="response")
    assert error.value.code=="invalid_covariance"


def test_sorted_json_state_restores_stable_table_column_order():
    result = fitted()
    state = json.loads(json.dumps(result.attrs["prediction_state"],sort_keys=True,allow_nan=False))
    restored = core.make_result(state)
    for name in result:
        pd.testing.assert_frame_equal(result[name],restored[name])
    assert result.to_latex()==restored.to_latex()


@pytest.mark.parametrize("collision",[(1,1.0),(0.0,-0.0)])
@pytest.mark.parametrize("role",["id","g"])
def test_equal_primitive_identifier_encodings_are_rejected_before_lookup(role,collision):
    frame,s = data_fixture()
    labels = frame[role].astype(object)
    labels.iloc[:2] = list(collision)
    frame[role] = labels
    with pytest.raises(AnalysisError) as error:
        oe.meta_dependent(data=frame,covariance=s,study="g",effect="id",yi="response")
    assert error.value.code=="ambiguous_identifier"


@pytest.mark.parametrize("collision",[(1,1.0),(0.0,-0.0)])
@pytest.mark.parametrize("axis",["index","columns"])
def test_covariance_axes_with_equal_primitive_labels_never_expand_selection(axis,collision):
    frame,s = data_fixture()
    labels = list(getattr(s,axis))
    labels[:2] = list(collision)
    setattr(s,axis,pd.Index(labels,dtype=object))
    assert getattr(s,axis).has_duplicates
    with pytest.raises(AnalysisError) as error:
        oe.meta_dependent(data=frame,covariance=s,study="g",effect="id",yi="response")
    assert error.value.code=="covariance_alignment"


@pytest.mark.parametrize("collision",[(1,1.0),(0.0,-0.0)])
def test_future_study_identifier_equality_collisions_are_rejected(collision):
    result = fitted()
    future = pd.DataFrame({"x":[-.5,.2,.8],"future":pd.Series([*collision,"new-study"],dtype=object)})
    with pytest.raises(AnalysisError) as error:
        oe.meta_dependent_predict_effect(result,data=future,study="future")
    assert error.value.code=="ambiguous_identifier"


@pytest.mark.parametrize("collision",[(1,1.0),(0.0,-0.0)])
def test_contrast_label_equality_collisions_are_rejected(collision):
    contrasts = pd.DataFrame([[1.,0.],[0.,1.]],columns=["Intercept","x"],
                             index=pd.Index(list(collision),dtype=object))
    with pytest.raises(AnalysisError) as error:
        oe.meta_dependent_contrast(fitted(),contrasts=contrasts)
    assert error.value.code=="ambiguous_identifier"


def test_identical_repeated_study_and_original_row_labels_still_round_trip():
    result = fitted()
    state,tensors = core.load_state(result)
    assert state["study_ids"]==[j//2 for j in range(16)]
    assert state["sample_labels"]==[f"physical{j//2}" for j in range(16)]
    assert tensors["cluster_index"].tolist()==state["study_ids"]


def test_numerical_sample_digest_rejects_relabelled_covariance_even_after_reseal():
    state = copy.deepcopy(fitted().attrs["prediction_state"])
    state["effect_ids"][0] = "different-effect"
    with pytest.raises(AnalysisError) as error:
        core.load_state(core.seal_state(state))
    assert error.value.code=="invalid_result_state"


def test_workspace_admission_precedes_numeric_materialization(monkeypatch):
    frame,s = data_fixture(64)
    def forbidden(*args,**kwargs):
        raise AssertionError("large geometry must be rejected before numerical allocation")
    monkeypatch.setattr(core,"_numeric",forbidden)
    with use_workspace_budget(1),pytest.raises(AnalysisError) as error:
        oe.meta_dependent(data=frame,covariance=s,study="g",effect="id",yi="response")
    assert error.value.code=="workspace_limit"


def test_result_table_drift_is_rejected_even_when_state_is_intact():
    result = fitted()
    result["coefficients"].loc[0,"estimate"] += .1
    with pytest.raises(AnalysisError) as error:
        core.load_state(result)
    assert error.value.code=="invalid_result_state"


@pytest.mark.parametrize("model",["effect","study"])
def test_random_fit_keeps_complete_boundary_profile_and_resolved_score_receipt(model):
    state = fitted(model=model).attrs["prediction_state"]
    solver = state["solver"]
    assert len(solver["profile_grid"])==82
    assert solver["profile_grid"][0]["tau2"]==0
    assert solver["profile_grid"][-1]["normalized_score"]<0
    assert solver["upper_bound"]>state["tau2"]
    assert abs(solver["score_residual"])<1e-7
    assert solver["candidates"][0]["tau2"]==0
    assert 82<=solver["evaluations"]<=1024
    assert state["sample_hash"]==core.sample_digest(state)
