"""Independent risk masks, likelihood/covariance and survivor oracles."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq, minimize
from scipy.stats import rankdata
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.streaming_cox import fit_streaming_cox, supports_spec
from openecon.econometrics.survival.cox import fit_stcox
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def data(n=110,entry=True):
    rng = np.random.default_rng(68201)
    x,z = rng.normal(size=(2,n))
    start = rng.integers(0,4,n).astype(float) if entry else np.zeros(n)
    exit = start+1+np.round(rng.exponential(np.exp(-.45*x+.25*z))*8)
    return pd.DataFrame({"t":exit,"t0":start,"d":rng.binomial(1,.65,n),"x":x,"z":z,
                         "s":np.arange(n)%3,"off":rng.normal(0,.15,n),"g":np.arange(n)%17,
                         "w":rng.uniform(.4,1.6,n),"f":rng.integers(1,4,n),"id":np.arange(n)})


def spec(ties="breslow",kind="nonrobust",weight_type=None,phtest="identity",**columns):
    return ModelSpec(estimator="stcox",outcome="t",predictors=["x","z"],intercept=False,
                     covariance=kind,cluster="g" if kind == "cluster" else None,
                     weights="f" if weight_type == "fweight" else "w" if weight_type else None,
                     weight_type=weight_type,columns={"failure":"d","entry":"t0","strata":"s","offset":"off",**columns},
                     options={"ties":ties,"phtest":phtest})


def params(result):
    return np.array([row.estimate for row in result.coefficients])


def moments(beta,frame,ties,w):
    """Direct independent full-array event risk masks, no ordered/native helpers."""
    x = frame[["x","z"]].to_numpy()
    eta = x@beta+frame.off.to_numpy()
    u = np.exp(eta)
    time,entry,delta,strata = (frame[name].to_numpy() for name in ["t","t0","d","s"])
    ll,g,h = 0.,np.zeros(2),np.zeros((2,2))
    scores = np.zeros_like(x)
    rows = []
    for stratum in pd.unique(strata):
        for event in np.unique(time[(strata == stratum)&(delta == 1)]):
            death = (strata == stratum)&(time == event)&(delta == 1)
            risk = (strata == stratum)&(entry < event)&(time >= event)
            xr,xd = x[risk],x[death]
            vr,vd = w[risk]*u[risk],w[death]*u[death]
            s0,s1,s2 = vr.sum(),vr@xr,xr.T@(vr[:,None]*xr)
            d0,d1,d2 = vd.sum(),vd@xd,xd.T@(vd[:,None]*xd)
            d,c = w[death].sum(),int(death.sum())
            ll += w[death]@eta[death]
            g += w[death]@xd
            means,pulled,a,b,b1 = np.zeros(2),np.zeros(2),0.,0.,np.zeros(2)
            fractions = [0.] if ties == "breslow" else np.arange(c)/c
            for fraction in fractions:
                mult = d if ties == "breslow" else 1.
                denominator = s0-fraction*d0
                mean = (s1-fraction*d1)/denominator
                ll -= mult*np.log(denominator)
                g -= mult*mean
                h -= mult*((s2-fraction*d2)/denominator-np.outer(mean,mean))
                a += mult/denominator
                b += mult*fraction/denominator
                pulled += mult*mean/denominator
                b1 += mult*fraction*mean/denominator
                means += mean if ties == "breslow" else mean/c
            scores[death] += xd-means+u[death,None]*(xd*b-b1)
            scores[risk] -= u[risk,None]*(xr*a-pulled)
            rows.append((stratum,event,death,risk,means,a,s0,u[death]/s0,w[death]))
    return ll,g,h,scores,rows


def oracle_fit(frame,ties,w):
    answer = minimize(lambda beta:-moments(beta,frame,ties,w)[0],np.zeros(2),
                      jac=lambda beta:-moments(beta,frame,ties,w)[1],method="BFGS",options={"gtol":1e-10})
    # A final independent Newton polish removes optimizer stopping noise.
    beta = answer.x
    for _ in range(3):
        _,g,h,_,_ = moments(beta,frame,ties,w)
        beta -= np.linalg.solve(h,g)
    return beta,moments(beta,frame,ties,w)


@pytest.mark.parametrize("ties",["breslow","efron"])
@pytest.mark.parametrize("kind",["nonrobust","robust","cluster"])
def test_global_delayed_entry_ties_strata_offset_scores_and_covariance(ties,kind,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data()
    requested = spec(ties,kind)
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=19)
    beta,(ll,g,h,scores,_) = oracle_fit(frame,ties,np.ones(len(frame)))
    np.testing.assert_allclose(params(result),beta,rtol=2e-8,atol=2e-9)
    assert result.metrics["log_likelihood"] == pytest.approx(ll,rel=2e-12)
    bread = np.linalg.inv(-h)
    if kind == "nonrobust":
        covariance = bread
    else:
        if kind == "cluster":
            sums = np.array([scores[frame.g.to_numpy() == group].sum(0) for group in np.unique(frame.g)])
            meat,factor = sums.T@sums,len(sums)/(len(sums)-1)
        else:
            meat,factor = scores.T@scores,len(frame)/(len(frame)-1)
        covariance = bread@(meat*factor)@bread
        np.testing.assert_allclose(result.inference["weighted_score_sum"],g,atol=1e-8)
    np.testing.assert_allclose(result.covariance_matrix,covariance,rtol=2e-8,atol=2e-10)
    assert result.metrics["n_failures"] == frame.d.sum()
    assert result.metrics["time_at_risk"] == (frame.t-frame.t0).sum()
    assert result.extra["replay_risk_sets"]["risk_interval"] == "entry < event_time <= exit"
    assert result.extra["replay_risk_sets"]["event_rows_materialized"] is False
    assert result.provenance["streaming"]["maximum_batch_rows"] <= 19
    assert result.sample_positions == []
    assert ResultBundle.model_validate_json(result.model_dump_json()).nobs == len(frame)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ties",["breslow","efron"])
@pytest.mark.parametrize("transform",["identity","log","km","rank"])
def test_baseline_kp_survival_and_all_ph_transforms_against_direct_oracle(ties,transform):
    frame = data(n=83)
    requested = spec(ties,phtest=transform)
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=13)
    beta = params(result)
    _,_,_,_,rows = moments(beta,frame,ties,np.ones(len(frame)))
    rows.sort(key=lambda row:(row[0],row[1]))
    baseline = result.extra["baseline"]
    hazards,kps,previous,H,KP = [],[],None,0.,1.
    for stratum,time,death,risk,mean,a,s0,r,w in rows:
        if stratum != previous:
            H,KP = 0.,1.
        previous = stratum
        H += a
        hazards.append(H)
        if w@r >= 1-1e-12:
            KP = 0.
        elif KP:
            def function(psi):
                return np.sum(w*r/(-np.expm1(r*psi)))-1
            right,left = -w.sum(),-w.sum()
            while function(left) > 0:
                left *= 2
            root = brentq(function,left,right,xtol=1e-12)
            KP *= np.exp(root/s0)
        kps.append(KP)
    np.testing.assert_allclose(baseline["cumulative_hazard"],hazards,rtol=2e-9,atol=2e-10)
    np.testing.assert_allclose(baseline["survivor_kp"],kps,rtol=2e-8,atol=2e-10)
    failing = frame.d.to_numpy() == 1
    times = frame.t.to_numpy()[failing]
    if transform == "identity":
        g = times
    elif transform == "log":
        g = np.log(times)
    elif transform == "rank":
        g = rankdata(times,method="average")
    else:
        survivor,values = 1.,{}
        for time in np.unique(times):
            risk = (frame.t0.to_numpy() < time)&(frame.t.to_numpy() >= time)
            d = ((frame.t.to_numpy() == time)&failing).sum()
            survivor *= 1-d/risk.sum()
            values[time] = 1-survivor
        g = np.array([values[time] for time in times])
    schoenfeld = np.empty((failing.sum(),2))
    indices = np.where(failing)[0]
    lookup = {index:j for j,index in enumerate(indices)}
    x = frame[["x","z"]].to_numpy()
    for _,_,death,_,mean,*_ in rows:
        for index in np.where(death)[0]:
            schoenfeld[lookup[index]] = x[index]-mean
    centred = g-g.mean()
    u = np.sum(schoenfeld*centred[:,None],axis=0)
    v = np.array(result.covariance_matrix)
    expected = len(g)*(u@v@u)/(centred@centred)
    assert result.tests["ph_global"]["statistic"] == pytest.approx(expected,rel=3e-8,abs=1e-10)
    dense = fit_stcox(requested,frame)
    for term in ["x","z"]:
        assert result.tests[f"ph_{term}"]["rho"] == pytest.approx(dense.tests[f"ph_{term}"]["rho"],rel=3e-8,abs=1e-9)


@pytest.mark.parametrize("weight_type",["fweight","iweight","pweight"])
@pytest.mark.parametrize("kind",["nonrobust","robust","cluster"])
def test_breslow_weight_domains_replication_and_pseudolikelihood(weight_type,kind):
    frame = data(n=91)
    requested = spec(kind=kind,weight_type=weight_type)
    if weight_type == "pweight" and kind == "nonrobust":
        with pytest.raises(AnalysisError) as error:
            fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=17)
        assert error.value.code == "unsupported_covariance"
        return
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=17)
    dense = fit_stcox(requested,frame)
    np.testing.assert_allclose(params(result),params(dense),rtol=3e-8,atol=2e-9)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=3e-8,atol=3e-10)
    assert result.nobs == dense.nobs
    assert result.metrics["log_likelihood"] == pytest.approx(dense.metrics["log_likelihood"],rel=3e-10)
    np.testing.assert_allclose(result.extra["baseline"]["survivor_kp"],dense.extra["baseline"]["survivor_kp"],rtol=5e-8,atol=3e-10)


@pytest.mark.parametrize("ties",["breslow","efron"])
def test_time_varying_multiple_records_and_robust_id_covariance(ties):
    frame = data(n=50)
    rows = []
    for index,row in frame.iterrows():
        midpoint = (row.t0+row.t)/2
        rows.extend([{**row.to_dict(),"t":midpoint,"d":0,"id":index},
                     {**row.to_dict(),"t0":midpoint,"x":row.x+.2,"id":index}])
    frame = pd.DataFrame(rows).sample(frac=1,random_state=4).reset_index(drop=True)
    requested = spec(ties,kind="robust",id="id")
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=11)
    dense = fit_stcox(requested,frame)
    np.testing.assert_allclose(params(result),params(dense),rtol=2e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=2e-8,atol=3e-10)
    assert result.inference["cluster_count"] == 50
    assert result.metrics["n_subjects"] == 50
    assert result.metrics["n_failures"] == frame.d.sum()


def test_risk_boundary_ties_and_exact_harrell_pairs():
    frame = data(n=65,entry=False)
    frame.t = (np.arange(len(frame))%7)+1.
    frame.x = np.round(frame.x,1)
    frame.z = 0.
    requested = spec()
    requested.predictors = ["x"]
    requested.columns.pop("entry")
    requested.columns.pop("offset")
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=7)
    eta = frame.x.to_numpy()*params(result)[0]
    pairs,agree,tied = 0,0,0
    for i in range(len(frame)):
        if not frame.d.iloc[i]:
            continue
        for j in range(len(frame)):
            if frame.s.iloc[j] == frame.s.iloc[i] and (frame.t.iloc[j] > frame.t.iloc[i] or (frame.t.iloc[j] == frame.t.iloc[i] and not frame.d.iloc[j])):
                pairs += 1
                agree += eta[i] > eta[j]
                tied += eta[i] == eta[j]
    assert result.extra["concordance"]["pairs"] == pairs
    assert result.extra["concordance"]["concordant"] == agree
    assert result.extra["concordance"]["tied_predictions"] == tied
    assert result.metrics["concordance"] == (agree+tied/2)/pairs


def test_actual_parquet_missing_nonrisk_categorical_strata_rank_and_source_counts(tmp_path):
    frame = data(n=210)
    frame["cat"] = pd.Categorical(np.array(["A","B","C"])[np.arange(len(frame))%3])
    frame["constant_within_strata"] = frame.s*2+1.
    frame.loc[5,"t"] = frame.loc[5,"t0"]
    frame.loc[7,"z"] = np.nan
    path = tmp_path/"survival.parquet"
    frame.to_parquet(path,index=False)
    requested = spec()
    requested.predictors += ["cat","constant_within_strata"]
    requested.categorical = ["cat"]
    requested.missing = "drop"
    result = fit_streaming_cox(requested,scan(path),batch_rows=23)
    dense = fit_stcox(requested,frame)
    assert result.nobs == 208 and result.nobs_original == 210 and result.dropped_rows == 2
    assert any("end on or before" in warning for warning in result.warnings)
    assert result.provenance["omitted_terms"]
    assert [row.term for row in result.coefficients] == [row.term for row in dense.coefficients]
    np.testing.assert_allclose(params(result),params(dense),rtol=3e-8,atol=3e-9)


@pytest.mark.parametrize("defect,code",[("no_failures","no_failures"),("overlap","overlapping_records"),
                                       ("negative_time","invalid_survival_time"),("invalid_failure","invalid_failure_indicator"),
                                       ("all_absorbed","no_covariates"),("efron_weights","unsupported_weights"),
                                       ("exactp","invalid_spec"),("tvc","invalid_spec")])
def test_explicit_statistical_and_unsupported_option_guards_cleanup(defect,code,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data()
    requested = spec()
    if defect == "no_failures":
        frame.d = 0
    elif defect == "overlap":
        requested.columns["id"] = "id"
        frame.loc[1,["id","t0","t"]] = [frame.id.iloc[0],frame.t0.iloc[0],frame.t.iloc[0]]
    elif defect == "negative_time":
        frame.loc[0,"t"] = -1
    elif defect == "invalid_failure":
        frame.loc[0,"d"] = 2
    elif defect == "all_absorbed":
        frame.x,frame.z = frame.s,frame.s*3
    elif defect == "efron_weights":
        requested = spec("efron",weight_type="iweight")
    elif defect == "exactp":
        requested.options["ties"] = "exactp"
        requested.covariance = "robust"
    else:
        requested.columns["tvc"] = ["z"]
        requested.options["ties"] = "exactp"
    with pytest.raises(AnalysisError) as error:
        fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=17)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())
    if defect in {"exactp","tvc"}:
        assert supports_spec(requested) is True


def test_cpu_scope_rng_workspace_and_changed_source_cleanup(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(n=65)
    rng = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        result = fit_streaming_cox(spec(),Dataset.from_frame(frame),batch_rows=13)
        assert torch.empty(0).device.type == "meta"
    assert torch.equal(rng,torch.random.get_rng_state())
    assert result.nobs == 65
    with use_workspace_budget(8),pytest.raises(AnalysisError) as error:
        fit_streaming_cox(spec(),Dataset.from_frame(frame),batch_rows=13)
    assert error.value.code == "workspace_limit"
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        copied = frame.copy()
        if calls >= 7:
            copied.loc[4,"x"] += .2
        for first in range(0,len(frame),13):
            yield copied.iloc[first:first+13]
    with pytest.raises(AnalysisError) as changed:
        fit_streaming_cox(spec(),Dataset.from_batches(factory,frame.columns,row_count=len(frame)),batch_rows=13)
    assert changed.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_many_failure_times_are_thinned_only_after_complete_baseline():
    frame = data(n=680,entry=False)
    frame.t = np.linspace(1,40,len(frame))
    frame.d = 1
    requested = spec()
    requested.columns.pop("entry")
    requested.options["concordance"] = False
    result = fit_streaming_cox(requested,Dataset.from_frame(frame),batch_rows=29)
    dense = fit_stcox(requested,frame)
    assert result.extra["baseline"]["thinned"] is True
    assert result.extra["baseline"]["n_times"] == 680
    assert len(result.extra["baseline"]["time"]) == 400
    np.testing.assert_allclose(result.extra["baseline"]["cumulative_hazard"],dense.extra["baseline"]["cumulative_hazard"],rtol=5e-8,atol=5e-9)
