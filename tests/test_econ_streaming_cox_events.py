"""Independent subset enumeration and event-time design/physical-score oracles."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.streaming_cox import fit_streaming_cox, supports_spec
from openecon.econometrics.survival.cox import fit_stcox
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget

from test_econ_streaming_cox import data, params, spec


def exact_moments(beta,frame):
    x = frame[["x","z"]].to_numpy()
    eta = x@beta+frame.off.to_numpy()
    time,entry,delta,strata = (frame[name].to_numpy() for name in ["t","t0","d","s"])
    ll,g,h = 0.,np.zeros(2),np.zeros((2,2))
    for stratum in pd.unique(strata):
        for event in np.unique(time[(strata == stratum)&(delta == 1)]):
            dead = (strata == stratum)&(time == event)&(delta == 1)
            risk = np.flatnonzero((strata == stratum)&(entry < event)&(time >= event))
            subsets = np.array(list(combinations(risk,int(dead.sum()))),dtype=int)
            indices = eta[subsets].sum(1)
            sufficient = x[subsets].sum(1)
            denominator = logsumexp(indices)
            probability = np.exp(indices-denominator)
            mean = probability@sufficient
            difference = sufficient-mean
            ll += eta[dead].sum()-denominator
            g += x[dead].sum(0)-mean
            h -= difference.T@(probability[:,None]*difference)
    return ll,g,h


def event_moments(beta,frame,ties,weights,texp):
    time,entry,delta,strata = (frame[name].to_numpy() for name in ["t","t0","d","s"])
    ll,g,h = 0.,np.zeros(2),np.zeros((2,2))
    physical_scores = np.zeros((len(frame),2))
    for stratum in pd.unique(strata):
        for event in np.unique(time[(strata == stratum)&(delta == 1)]):
            dead = (strata == stratum)&(time == event)&(delta == 1)
            risk = (strata == stratum)&(entry < event)&(time >= event)
            x = np.column_stack([frame.x.to_numpy(),frame.z.to_numpy()*(event if texp == "identity" else np.log(event))])
            eta = x@beta+frame.off.to_numpy()
            u = np.exp(eta)
            xr,xd = x[risk],x[dead]
            vr,vd = weights[risk]*u[risk],weights[dead]*u[dead]
            s0,s1,s2 = vr.sum(),vr@xr,xr.T@(vr[:,None]*xr)
            d0,d1,d2 = vd.sum(),vd@xd,xd.T@(vd[:,None]*xd)
            d,count = weights[dead].sum(),int(dead.sum())
            ll += weights[dead]@eta[dead]
            g += weights[dead]@xd
            for fraction in ([0.] if ties == "breslow" else np.arange(count)/count):
                multiplier = d if ties == "breslow" else 1.
                denominator = s0-fraction*d0
                mean = (s1-fraction*d1)/denominator
                ll -= multiplier*np.log(denominator)
                g -= multiplier*mean
                h -= multiplier*((s2-fraction*d2)/denominator-np.outer(mean,mean))
                physical_scores[risk] -= multiplier*u[risk,None]*(xr-mean)/denominator
                physical_scores[dead] += multiplier*fraction*u[dead,None]*(xd-mean)/denominator
            physical_scores[dead] += xd-(s1/s0 if ties == "breslow" else np.mean([(s1-a*d1)/(s0-a*d0) for a in np.arange(count)/count],axis=0))
    return ll,g,h,physical_scores


def solve(function):
    fit = minimize(lambda beta:-function(beta)[0],np.zeros(2),jac=lambda beta:-function(beta)[1],
                   method="BFGS",options={"gtol":1e-10})
    beta = fit.x
    for _ in range(4):
        _,g,h,*_ = function(beta)
        beta -= np.linalg.solve(h,g)
    return beta,function(beta)


@pytest.mark.parametrize("batch_rows",[3,11])
def test_exactp_against_independent_all_subset_enumeration(batch_rows,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(n=25)
    frame.t = frame.t0+1+np.minimum(frame.t-frame.t0-1,3)
    requested = spec("exactp")
    beta,(ll,g,h) = solve(lambda value:exact_moments(value,frame))
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=batch_rows)
    np.testing.assert_allclose(params(result),beta,rtol=2e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,np.linalg.inv(-h),rtol=2e-8,atol=3e-10)
    assert result.metrics["log_likelihood"] == pytest.approx(ll,rel=1e-12)
    assert np.linalg.norm(g) < 1e-9
    assert result.nobs == 25
    assert result.extra["replay_event_kernel"]["exact_method"].startswith("normalized elementary")
    assert result.provenance["data_hash"]
    assert supports_spec(requested)
    dense = fit_stcox(requested,frame)
    np.testing.assert_allclose(result.extra["baseline"]["survivor_kp"],dense.extra["baseline"]["survivor_kp"],rtol=1e-8,atol=1e-10)
    assert result.tests["ph_global"]["statistic"] == pytest.approx(dense.tests["ph_global"]["statistic"],rel=3e-7)
    assert ResultBundle.model_validate_json(result.model_dump_json()).metrics == result.metrics
    assert not list(tmp_path.iterdir())


def test_exact_reflection_all_failed_event_and_single_event_mix():
    frame = data(n=13,entry=False)
    frame.s = 0
    frame.t = np.repeat([1.,2.,3.],[8,3,2])
    frame.d = np.repeat([1.,0.,1.],[8,3,2])
    requested = spec("exactp")
    beta,(ll,_,h) = solve(lambda value:exact_moments(value,frame))
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=4)
    np.testing.assert_allclose(params(result),beta,rtol=3e-8,atol=1e-9)
    np.testing.assert_allclose(result.covariance_matrix,np.linalg.inv(-h),rtol=3e-8,atol=1e-9)
    assert result.metrics["log_likelihood"] == pytest.approx(ll,rel=3e-12)
    assert result.extra["replay_event_kernel"]["exact_recursion_max_states"] == 5


@pytest.mark.parametrize("ties",["breslow","efron"])
@pytest.mark.parametrize("texp",["identity","log"])
@pytest.mark.parametrize("kind",["nonrobust","robust","cluster"])
def test_tvc_full_likelihood_and_physical_lin_wei_scores_independent_oracle(ties,texp,kind,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(n=67)
    # One zero-exposure group must still enter the physical cluster count.
    frame.loc[0,["t0","t","d","g"]] = [0.,.2,0.,999.]
    requested = spec(ties,kind,tvc=["z"])
    requested.predictors = ["x"]
    requested.options["texp"] = texp
    beta,(ll,g,h,scores) = solve(lambda value:event_moments(value,frame,ties,np.ones(len(frame)),texp))
    bread = np.linalg.inv(-h)
    if kind == "nonrobust":
        covariance = bread
    elif kind == "robust":
        covariance = bread@(scores.T@scores*len(frame)/(len(frame)-1))@bread
    else:
        grouped = np.array([scores[frame.g.to_numpy() == group].sum(0) for group in pd.unique(frame.g)])
        covariance = bread@(grouped.T@grouped*len(grouped)/(len(grouped)-1))@bread
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=7)
    np.testing.assert_allclose(params(result),beta,rtol=3e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,covariance,rtol=3e-8,atol=3e-10)
    assert result.metrics["log_likelihood"] == pytest.approx(ll,rel=2e-12)
    np.testing.assert_allclose(scores.sum(0),g,atol=2e-12)
    if kind == "cluster":
        assert result.inference["cluster_count"] == frame.g.nunique()
    assert result.nobs == len(frame)
    assert [row.equation for row in result.coefficients] == ["main","tvc"]
    assert "baseline" not in result.extra and "ph_global" not in result.tests
    assert result.extra["replay_risk_sets"]["event_rows_materialized_on_disk"] is True
    assert result.provenance["tvc_rows"] > len(frame)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("weight_type",["fweight","iweight","pweight"])
@pytest.mark.parametrize("kind",["nonrobust","robust","cluster"])
def test_tvc_weights_original_counts_and_sandwich_units(weight_type,kind):
    frame = data(n=57)
    requested = spec("breslow",kind,weight_type,tvc=["z"])
    requested.predictors = ["x"]
    if weight_type == "pweight" and kind == "nonrobust":
        with pytest.raises(AnalysisError) as error:
            fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=9)
        assert error.value.code == "unsupported_covariance"
        return
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=9)
    dense = fit_stcox(requested,frame)
    np.testing.assert_allclose(params(result),params(dense),rtol=3e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=3e-8,atol=3e-10)
    assert result.metrics["n_failures"] == dense.metrics["n_failures"]
    assert result.metrics["time_at_risk"] == dense.metrics["time_at_risk"]
    assert result.nobs == dense.nobs


@pytest.mark.parametrize("ties",["breslow","efron"])
def test_tvc_multi_record_subject_robust_covariance_counts_physical_people(ties):
    frame = data(n=50,entry=False)
    first = frame.copy()
    first.t = frame.t/2
    first.d = 0.
    second = frame.copy()
    second.t0 = first.t
    frame = pd.concat([first,second]).sample(frac=1,random_state=7465).reset_index(drop=True)
    requested = spec(ties,"robust",tvc=["z"],id="id")
    requested.predictors = ["x"]
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=7)
    dense = fit_stcox(requested,frame)
    np.testing.assert_allclose(params(result),params(dense),rtol=3e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=3e-8,atol=3e-10)
    assert result.inference["cluster_count"] == result.metrics["n_subjects"] == 50
    assert result.nobs == 100


def test_tvc_only_when_main_is_absorbed_and_log_times_below_one():
    frame = data(n=58,entry=False)
    frame.t = frame.t/100
    frame.x = frame.s
    requested = spec("breslow",tvc=["z"])
    requested.predictors = ["x"]
    requested.options["texp"] = "log"
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=8)
    dense = fit_stcox(requested,frame)
    assert [row.term for row in result.coefficients] == ["tvc:z"]
    np.testing.assert_allclose(params(result),params(dense),rtol=3e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=3e-8,atol=3e-10)


def test_tvc_actual_parquet_missing_units_meta_rng_source_and_cleanup(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(n=80)
    requested = spec("breslow","robust",tvc=["z"])
    requested.predictors = ["x"]
    requested.missing = "drop"
    frame.loc[4,"z"] = np.nan
    path = tmp_path/"survival.parquet"
    frame.to_parquet(path,index=False)
    rng = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        result = fit_streaming_cox(requested,scan(path),batch_rows=11)
        assert torch.empty(0).device.type == "meta"
    assert torch.equal(rng,torch.random.get_rng_state())
    assert result.nobs == 79 and result.nobs_original == 80 and result.dropped_rows == 1
    altered = frame.copy()
    altered.x *= 1e80
    altered.z *= 1e-80
    scaled = fit_streaming_cox(requested,Dataset.from_frame(altered),batch_rows=11)
    unit = np.diag([1e80,1e-80])
    np.testing.assert_allclose(unit@params(scaled),params(result),rtol=5e-8,atol=5e-9)
    np.testing.assert_allclose(unit@scaled.covariance_matrix@unit,result.covariance_matrix,rtol=5e-8,atol=5e-10)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("mode",["exactp","tvc"])
def test_live_workspace_and_actual_work_budget_fail_without_model_fitting(mode,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(n=36)
    requested = spec("exactp") if mode == "exactp" else spec("breslow",tvc=["z"])
    with use_workspace_budget(8),pytest.raises(AnalysisError) as error:
        fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=6)
    assert error.value.code == "workspace_limit"
    from openecon.econometrics import streaming_cox_events
    monkeypatch.setattr(streaming_cox_events,"EXACT_WORK_LIMIT" if mode == "exactp" else "TVC_WORK_LIMIT",1)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=6)
    assert error.value.code == ("exact_too_large" if mode == "exactp" else "tvc_work_limit")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("defect,code",[("weight","unsupported_weights"),("robust","invalid_spec"),
                                       ("tvc","invalid_spec"),("collinear","singular_information")])
def test_exact_and_tvc_native_statistical_domains_cleanup(defect,code,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(n=56)
    requested = spec("exactp")
    if defect == "weight":
        requested.weights,requested.weight_type = "f","fweight"
    elif defect == "robust":
        requested.covariance = "robust"
    elif defect == "tvc":
        requested.columns["tvc"] = ["z"]
    else:
        requested.options["ties"] = "breslow"
        requested.columns["tvc"] = ["z","duplicate"]
        frame["duplicate"] = frame.z
    with pytest.raises(AnalysisError) as error:
        fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=7)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_tvc_source_change_on_final_pass_is_not_a_partial_success(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(n=41)
    requested = spec("breslow",tvc=["z"])
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        current = frame.copy()
        if calls >= 8:
            current.loc[2,"z"] += .3
        for start in range(0,len(frame),7):
            yield current.iloc[start:start+7]
    with pytest.raises(AnalysisError) as error:
        fit_streaming_cox(requested,Dataset.from_batches(factory,columns=list(frame)),batch_rows=7)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())
