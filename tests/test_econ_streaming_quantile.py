"""Independent LP/density/bootstrap oracles for exact disk-backed quantiles."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from scipy.special import ndtri
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.quantile.qreg import fit_qreg
from openecon.econometrics.quantile.sqreg import fit_bsqreg, fit_iqreg, fit_sqreg
from openecon.econometrics.replay_sample import ReplaySample
from openecon.econometrics.streaming_quantile import (
    _Problem, _Rows, _scratch, _spool, fit_streaming_quantile,
)
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def data(n=130):
    rng = np.random.default_rng(77135)
    x,z = rng.normal(size=(2,n))
    group = np.arange(n)%19
    category = np.array(["A","B","C"])[np.arange(n)%3]
    y = .7+1.2*x-.9*z+.3*(category == "B")-.4*(category == "C")+(1+.2*np.abs(x))*rng.normal(size=n)
    return pd.DataFrame({"y":y,"x":x,"z":z,"g":group,"cat":category,
                         "w":rng.uniform(.5,1.5,n),"f":rng.integers(1,4,n)})


def spec(name="qreg",covariance="nonrobust",density=None,weight_type=None,**options):
    if name != "qreg":
        covariance = "bootstrap"
        options = {"reps":5,"seed":412,**options}
    if density:
        options["density"] = density
    return ModelSpec(estimator=name,outcome="y",predictors=["x","z"],covariance=covariance,
                     weights="f" if weight_type == "fweight" else "w" if weight_type else None,
                     weight_type=weight_type,cluster="g" if covariance == "cluster" else None,options=options)


def coefficients(result):
    return np.array([row.estimate for row in result.coefficients])


def design(frame,categorical=False,intercept=True):
    return np.column_stack([*([np.ones(len(frame))] if intercept else []),frame.x,frame.z,
                            *([(frame.cat == label).astype(float) for label in ["B","C"]] if categorical else [])])


def solve_lp(x,y,tau,weights=None):
    """Independent check-loss primal LP, with nonnegative positive/negative residuals."""
    n,k = x.shape
    w = np.ones(n) if weights is None else np.asarray(weights)
    objective = np.r_[np.zeros(k),tau*w,(1-tau)*w]
    constraints = np.column_stack((x,np.eye(n),-np.eye(n)))
    solved = linprog(objective,A_eq=constraints,b_eq=y,bounds=[(None,None)]*k+[(0,None)]*(2*n),method="highs")
    assert solved.success, solved.message
    return solved.x[:k],solved.fun


def weighted_percentile(values,p,weights,raw=False):
    order = np.argsort(values,kind="stable")
    ordered, cumulative = values[order],np.cumsum(weights[order])
    total = cumulative[-1]
    slack = 8*np.finfo(float).eps*(total+1 if raw else total)
    target = p*(total+1)-1+slack if raw else p*total+slack
    j = min(np.searchsorted(cumulative,target,side="right"),len(ordered)-1)
    if not raw and j and abs(cumulative[j-1]-p*total) <= slack:
        return (ordered[j-1]+ordered[j])/2
    return ordered[j]


def density_oracle(x,y,beta,w,n,tau,kind,method,groups):
    """NumPy/SciPy primal LP/density sandwich, not native solver calls."""
    z = ndtri(tau)
    phi = np.exp(-z*z/2)/np.sqrt(2*np.pi)
    h = n**(-1/3)*ndtri(.975)**(2/3)*(1.5*phi**2/(2*z*z+1))**(1/3)
    residual = y-x@beta
    residual[np.abs(residual) < 1e-9] = 0.
    if method == "fitted":
        difference = solve_lp(x,y,tau+h,w)[0]-solve_lp(x,y,tau-h,w)[0]
        sparsity = np.average(x,axis=0,weights=w)@difference/(2*h)
        spacing = x@difference
        positive = spacing > 1e-10*np.max(np.abs(spacing))
        density = np.divide(2*h,spacing,out=np.zeros_like(spacing),where=positive)
    elif method == "residual":
        sparsity = (weighted_percentile(residual,tau+h,w)-weighted_percentile(residual,tau-h,w))/(2*h)
    else:
        centred = residual-np.average(residual,weights=w)
        sd = np.sqrt(w@centred**2/(n-1))
        iqr = weighted_percentile(residual,.75,w)-weighted_percentile(residual,.25,w)
        c = min(sd,iqr/1.34)*(ndtri(tau+h)-ndtri(tau-h))
        # All kernel-oracle cases below explicitly select Gaussian.
        density = np.exp(-(residual/c)**2/2)/(np.sqrt(2*np.pi)*c)
        sparsity = 1/(w@density/n)
    if kind == "nonrobust":
        covariance = np.linalg.inv(x.T@(w[:,None]*x))*(tau*(1-tau)*sparsity**2)
    else:
        bread = np.linalg.inv(x.T@((w*density)[:,None]*x))
        if kind == "robust":
            meat = x.T@((w*w*tau*(1-tau))[:,None]*x)
        else:
            psi = tau-(residual <= 0)
            scores = x*(w*psi)[:,None]
            sums = np.array([scores[groups == group].sum(0) for group in np.unique(groups)])
            meat = sums.T@sums
        covariance = bread@meat@bread
    return covariance,sparsity


@pytest.mark.parametrize("kind,method",[("nonrobust","fitted"),("robust","fitted"),("nonrobust","residual"),
                                       ("nonrobust","kernel"),("robust","kernel"),("cluster","kernel")])
@pytest.mark.parametrize("weighted",[False,True])
def test_full_sample_lp_and_density_inference_against_independent_oracle(kind,method,weighted,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data()
    options = {"quantile":.4}
    if method == "kernel":
        options["kernel"] = "gaussian"
    requested = spec(covariance=kind,density=method,weight_type="aweight" if weighted else None,**options)
    result = fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=23)
    x,y = design(frame),frame.y.to_numpy()
    w = frame.w.to_numpy()*len(frame)/frame.w.sum() if weighted else np.ones(len(frame))
    expected,objective = solve_lp(x,y,.4,w)
    np.testing.assert_allclose(coefficients(result),expected,rtol=2e-9,atol=2e-9)
    assert result.metrics["sum_adev"] == pytest.approx(objective,rel=3e-11,abs=3e-11)
    covariance,sparsity = density_oracle(x,y,expected,w,len(frame),.4,kind,method,frame.g.to_numpy())
    np.testing.assert_allclose(result.covariance_matrix,covariance,rtol=2e-8,atol=2e-10)
    assert result.metrics["sparsity"] == pytest.approx(sparsity,rel=2e-9)
    assert result.provenance["solver_diagnostics"]["exact_vertex"] is True
    assert result.extra["replay_solver"]["row_state_resident"] is False
    assert result.sample_positions == []
    assert result.provenance["streaming"]["maximum_batch_rows"] <= 23
    assert ResultBundle.model_validate_json(result.model_dump_json()).nobs == len(frame)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("weight_type",["fweight","pweight"])
@pytest.mark.parametrize("kind",["nonrobust","robust","cluster"])
def test_frequency_and_sampling_weight_contract_matches_existing_dense_estimator(weight_type,kind):
    frame = data()
    requested = spec(covariance=kind,density="kernel",weight_type=weight_type,kernel="gaussian")
    if weight_type == "pweight" and kind == "nonrobust":
        with pytest.raises(AnalysisError) as error:
            fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=17)
        assert error.value.code == "unsupported_covariance"
        return
    result = fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=17)
    dense = fit_qreg(requested,frame)
    np.testing.assert_allclose(coefficients(result),coefficients(dense),rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=2e-9,atol=2e-10)
    assert result.nobs == dense.nobs


@pytest.mark.parametrize("name",["bsqreg","sqreg","iqreg"])
@pytest.mark.parametrize("cluster",[False,True])
def test_exact_global_bootstrap_joint_blocks_and_differences(name,cluster):
    frame = data(n=63)
    requested = spec(name,reps=5,seed=451)
    if cluster:
        requested.cluster = "g"
    result = fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=11)
    x,y = design(frame),frame.y.to_numpy()
    taus = [.5] if name == "bsqreg" else [.25,.5,.75] if name == "sqreg" else [.25,.75]
    point = np.array([solve_lp(x,y,tau)[0] for tau in taus])
    expected = point.ravel() if name == "sqreg" else point[-1]-point[0] if name == "iqreg" else point[0]
    np.testing.assert_allclose(coefficients(result),expected,rtol=1e-9,atol=2e-9)
    generator = torch.Generator().manual_seed(451)
    codes,grouplabels = pd.factorize(frame.g,sort=False)
    population = len(grouplabels) if cluster else len(frame)
    draws = []
    for _ in range(5):
        sample = torch.randint(population,(population,),generator=generator).numpy()
        counts = np.bincount(sample,minlength=population)
        counts = counts[codes] if cluster else counts
        keep = counts > 0
        values = np.array([solve_lp(x[keep],y[keep],tau,counts[keep])[0] for tau in taus])
        draws.append(values.ravel() if name == "sqreg" else values[-1]-values[0] if name == "iqreg" else values[0])
    expected_covariance = np.cov(np.array(draws),rowvar=False,ddof=1)
    np.testing.assert_allclose(result.covariance_matrix,expected_covariance,rtol=2e-8,atol=2e-9)
    dense = {"bsqreg":fit_bsqreg,"sqreg":fit_sqreg,"iqreg":fit_iqreg}[name](requested,frame)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=2e-9,atol=2e-9)
    assert result.extra["bootstrap"]["joint_draws"] is True
    assert result.extra["bootstrap"]["reps_used"] == 5
    assert result.extra["bootstrap"]["resampling"] == ("clusters" if cluster else "observations")
    if name == "sqreg":
        assert np.max(np.abs(expected_covariance[:3,3:6])) > .001
        assert [row.term for row in result.coefficients][:3] == ["q25:Intercept","q25:x","q25:z"]
        assert [row.equation for row in result.coefficients][:6] == ["q25"]*3+["q50"]*3


@pytest.mark.parametrize("name",["qreg","bsqreg","sqreg","iqreg"])
def test_tied_discrete_designs_return_certified_lp_solutions(name):
    frame = data(n=72)
    frame.x = np.arange(len(frame))%3
    frame.z = np.arange(len(frame))%2
    frame.y = np.round(.4+frame.x-.7*frame.z+np.random.default_rng(441).normal(size=len(frame)))
    requested = spec(name,density="kernel" if name == "qreg" else None,**({"kernel":"gaussian"} if name == "qreg" else {}))
    result = fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=13)
    taus = [.5] if name in {"qreg","bsqreg"} else [.25,.5,.75] if name == "sqreg" else [.25,.75]
    x,y = design(frame),frame.y.to_numpy()
    if name == "iqreg":
        values = [np.array([result.extra["coefficients_low"][term] for term in ["Intercept","x","z"]]),
                  np.array([result.extra["coefficients_high"][term] for term in ["Intercept","x","z"]])]
    else:
        values = coefficients(result).reshape(-1,3)
    for tau,beta in zip(taus,values,strict=True):
        residual = y-x@beta
        objective = np.where(residual < 0,(tau-1)*residual,tau*residual).sum()
        assert objective == pytest.approx(solve_lp(x,y,tau)[1],abs=1e-8)
    assert result.extra["replay_solver"]["exact_lp_certificate"] is True


def test_simplex_finish_and_degenerate_pivots_are_exact_without_interior_iterations(tmp_path):
    frame = data(n=89)
    frame.y = np.round(frame.y)
    requested = spec()
    sample = ReplaySample(requested,Dataset.from_frame(frame),batch_rows=9)
    sample.add_design("x")
    sample.prepare()
    with _scratch() as (root,sql):
        base,_ = _spool(sample,root,sql,3)
        problem = _Problem(root,sql,base,3)
        try:
            problem.prepare()
            fit = problem.solve(.37,max_iterations=0)
            assert fit.pivots > 0
            beta = sample.designs["x"].transform@fit.beta
            expected,objective = solve_lp(design(frame),frame.y.to_numpy(),.37)
            np.testing.assert_allclose(beta,expected,rtol=2e-9,atol=2e-9)
            assert fit.objective == pytest.approx(objective,rel=1e-11)
        finally:
            problem.close()
            base.close()


@pytest.mark.parametrize("name",["qreg","bsqreg","sqreg","iqreg"])
def test_perfect_fit_is_refused_and_scratch_is_removed(name,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data()
    frame.y = 1+2*frame.x-3*frame.z
    with pytest.raises(AnalysisError) as error:
        fit_streaming_quantile(spec(name),Dataset.from_frame(frame),batch_rows=29)
    assert error.value.code == "perfect_fit"
    assert not list(tmp_path.iterdir())


def test_actual_parquet_missing_categories_and_cpu_scope_without_collect(tmp_path,monkeypatch):
    frame = data(n=480)
    frame.loc[5,"z"] = np.nan
    frame.cat = pd.Categorical(frame.cat,categories=["A","B","C"],ordered=True)
    path = tmp_path/"data.parquet"
    frame.to_parquet(path,index=False)
    requested = spec(density="kernel",kernel="gaussian")
    requested.predictors += ["cat"]
    requested.categorical = ["cat"]
    requested.missing = "drop"
    original_qr = torch.linalg.qr
    monkeypatch.setattr(torch.linalg,"qr",_bounded_qr(original_qr,31,6))
    before = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        result = fit_streaming_quantile(requested,scan(path),batch_rows=31)
        assert torch.empty(0).device.type == "meta"
    assert torch.equal(before,torch.random.get_rng_state())
    monkeypatch.setattr(torch.linalg,"qr",original_qr)
    dense = fit_qreg(requested,frame)
    np.testing.assert_allclose(coefficients(result),coefficients(dense),rtol=2e-10,atol=2e-10)
    assert result.nobs == 479 and result.dropped_rows == 1 and result.nobs_original == 480
    assert len(result.predictions) == 400
    assert [row["row"] for row in result.predictions[:7]] == [0,1,2,3,4,6,7]
    assert result.provenance["categorical_encoding"]["cat"]["levels"] == ["A","B","C"]


def _bounded_qr(original,rows,width):
    def call(matrix,*args,**kwargs):
        assert matrix.shape[0] <= max(rows,2*width) and matrix.shape[1] <= width
        return original(matrix,*args,**kwargs)
    return call


def test_no_intercept_and_outcome_units_preserve_exact_solution():
    frame = data()
    requested = spec(density="kernel",kernel="gaussian")
    requested.intercept = False
    result = fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=7)
    oracle,objective = solve_lp(design(frame,intercept=False),frame.y.to_numpy(),.5)
    np.testing.assert_allclose(coefficients(result),oracle,rtol=2e-10,atol=2e-10)
    assert result.metrics["sum_adev"] == pytest.approx(objective,rel=2e-11)


def test_source_change_on_final_replay_and_owned_scratch_cleanup(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data()
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls >= 5:
            changed.loc[17,"y"] += .1
        for first in range(0,len(frame),17):
            yield changed.iloc[first:first+17]
    source = Dataset.from_batches(factory,columns=frame.columns,row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_quantile(spec(),source,batch_rows=17)
    assert error.value.code == "source_changed"
    assert calls == 5
    assert not list(tmp_path.iterdir())


def test_real_joint_width_guard_precedes_disk_spool_and_scratch_corruption_guard(tmp_path,monkeypatch):
    frame = data()
    requested = spec("sqreg",quantiles=list(np.linspace(.05,.95,140)))
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    with use_workspace_budget(16),pytest.raises(AnalysisError) as error:
        fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=11)
    assert error.value.code == "workspace_limit"
    assert not list(tmp_path.iterdir())
    records = _Rows(Path(tmp_path)/"records.bin",2,3)
    try:
        records.append(torch.ones((3,2),dtype=torch.float64))
        records.file.truncate(3)
        with pytest.raises(AnalysisError) as truncated:
            list(records.blocks())
        assert truncated.value.code == "scratch_storage"
    finally:
        records.close()


@pytest.mark.parametrize("name",["bsqreg","sqreg","iqreg"])
def test_rank_deficient_bootstrap_draw_is_skipped_jointly(name):
    frame = data(n=38)
    frame["rare"] = 0.
    frame.loc[0,"rare"] = 1.
    requested = spec(name,reps=9,seed=96)
    requested.predictors += ["rare"]
    result = fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=7)
    generator = torch.Generator().manual_seed(96)
    accepted = sum(bool((torch.randint(len(frame),(len(frame),),generator=generator) == 0).any()) for _ in range(9))
    assert 2 <= accepted < 9
    assert result.extra["bootstrap"]["reps_used"] == accepted
    assert result.metrics["reps"] == accepted
    assert any("bootstrap samples" in warning for warning in result.warnings)


@pytest.mark.parametrize("options,kind,code",[({"density":"residual"},"robust","invalid_spec"),
                                             ({"quantile":.99},"nonrobust","bandwidth_out_of_range"),
                                             ({"density":"fitted","kernel":"gaussian"},"nonrobust","invalid_spec")])
def test_density_and_probability_window_fail_explicitly(options,kind,code):
    with pytest.raises(AnalysisError) as error:
        fit_streaming_quantile(spec(covariance=kind,**options),Dataset.from_frame(data()),batch_rows=19)
    assert error.value.code == code


def test_insufficient_disk_space_is_explicit_before_spooling(tmp_path,monkeypatch):
    from collections import namedtuple
    from openecon.econometrics import streaming_quantile as native
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    usage = namedtuple("usage","total used free")(100,99,1)
    monkeypatch.setattr(native.shutil,"disk_usage",lambda _:usage)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_quantile(spec(),Dataset.from_frame(data()))
    assert error.value.code == "scratch_space"
    assert not list(tmp_path.iterdir())


def test_predictor_affine_rescaling_preserves_global_optimum_and_covariance():
    frame = data()
    requested = spec(density="kernel",kernel="gaussian")
    baseline = fit_streaming_quantile(requested,Dataset.from_frame(frame),batch_rows=13)
    shifted = frame.copy()
    shifted.x = (frame.x+2)*1e50
    shifted.z = (frame.z-3)*1e-50
    result = fit_streaming_quantile(requested,Dataset.from_frame(shifted),batch_rows=13)
    transform = np.array([[1,2e50,-3e-50],[0,1e50,0],[0,0,1e-50]])
    np.testing.assert_allclose(transform@coefficients(result),coefficients(baseline),rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(transform@np.array(result.covariance_matrix)@transform.T,baseline.covariance_matrix,rtol=2e-9,atol=2e-10)
    assert result.metrics["sum_adev"] == pytest.approx(baseline.metrics["sum_adev"],rel=2e-12)
