"""Independent quadrature integrals, conditional enumeration and native contracts."""
from itertools import combinations

import numpy as np
from numpy.polynomial.hermite import hermgauss
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_ndtr, logsumexp
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.mixed.glmm import fit_melogit, fit_meprobit, fit_mepoisson
from openecon.econometrics.mixed.xt import fit_xtlogit, fit_xtprobit, fit_xtpoisson
from openecon.econometrics.streaming_glmm import fit_streaming_glmm, supports_spec
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


_DENSE = {"melogit":fit_melogit,"meprobit":fit_meprobit,"mepoisson":fit_mepoisson,
          "xtlogit":fit_xtlogit,"xtprobit":fit_xtprobit,"xtpoisson":fit_xtpoisson}


def data(family="logit",n_groups=12,size=8,random_slope=False):
    rng = np.random.default_rng(74753)
    n = n_groups*size
    group = np.repeat(np.arange(n_groups),size)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = np.linspace(-2.4,2.4,n_groups)[group]
    eta = .25+.6*x+u+(.7*np.sin(np.arange(n_groups))[group]*z if random_slope else 0)
    if family == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        probability = expit(eta) if family == "logit" else np.exp(log_ndtr(eta))
        y = rng.binomial(1,probability)
    return pd.DataFrame({"y":y,"x":x,"z":z,"g":group,"cluster":group//2,"t":np.tile(np.arange(size),n_groups),
                         "offset":rng.normal(0,.1,n),"exposure":rng.uniform(.8,1.2,n)})


def spec(estimator,kind="nonrobust",method="ghermite",points=7,model="re",random=None):
    columns = {"group":"g",**({"random":random} if random else {})} if estimator.startswith("me") else {}
    options = {"intpoints":points,"intmethod":method}
    if estimator.startswith("xt"):
        options = {"model":model,**(options if model == "re" else {})}
    return ModelSpec(estimator=estimator,outcome="y",predictors=["x"],columns=columns,
                     panel="g" if estimator.startswith("xt") else None,intercept=model != "fe",covariance=kind,
                     cluster="cluster" if kind == "cluster" else None,options=options)


def params(result):
    return np.array([row.estimate for row in result.coefficients])


def quadrature_value(theta,frame,family,points):
    """Independent NumPy Hermite sum of the integrated group probabilities."""
    grid,weight = hermgauss(points)
    v = np.sqrt(2)*grid
    weight = weight/np.sqrt(np.pi)
    ll = 0.
    for group in pd.unique(frame.g):
        part = frame[frame.g == group]
        index = theta[0]+theta[1]*part.x.to_numpy()[:,None]+np.exp(theta[2])*v
        y = part.y.to_numpy()[:,None]
        if family == "logit":
            conditional = -y*np.logaddexp(0,-index)-(1-y)*np.logaddexp(0,index)
        elif family == "probit":
            conditional = log_ndtr((2*y-1)*index)
        else:
            conditional = y*index-np.exp(index)-gammaln(y+1)
        ll += logsumexp(np.log(weight)+conditional.sum(0))
    return ll


def finite_gradient(function,theta,step=1e-5):
    eye = np.eye(len(theta))*step
    return np.array([(function(theta-2*row)-8*function(theta-row)+8*function(theta+row)-function(theta+2*row))/(12*step) for row in eye])


def finite_hessian(function,theta,step=2e-4):
    eye = np.eye(len(theta))*step
    base = function(theta)
    result = np.empty((len(theta),len(theta)))
    for i,left in enumerate(eye):
        result[i,i] = (-function(theta+2*left)+16*function(theta+left)-30*base+16*function(theta-left)-function(theta-2*left))/(12*step**2)
        for j,right in enumerate(eye[:i]):
            result[i,j] = result[j,i] = (function(theta+left+right)-function(theta+left-right)-function(theta-left+right)+function(theta-left-right))/(4*step**2)
    return result


@pytest.mark.parametrize("estimator,family",[("melogit","logit"),("meprobit","probit"),("mepoisson","poisson")])
def test_nonadaptive_full_integral_and_information_independent_numpy_hermite(estimator,family):
    frame = data(family)
    requested = spec(estimator,points=11)
    result = fit_streaming_glmm(requested,Dataset.from_frame(frame),batch_rows=5)
    values = params(result)
    theta = np.array([values[0],values[1],.5*np.log(values[2])])
    def function(value):
        return quadrature_value(value,frame,family,11)
    assert result.metrics["log_likelihood"] == pytest.approx(function(theta),rel=3e-12)
    np.testing.assert_allclose(finite_gradient(function,theta),np.zeros(3),atol=2e-6)
    expected = np.linalg.inv(-finite_hessian(function,theta))
    jacobian = np.diag([1,1,2*values[2]])
    np.testing.assert_allclose(result.covariance_matrix,jacobian@expected@jacobian,rtol=3e-4,atol=1e-5)
    assert result.nobs == len(frame)
    assert result.metrics["n_groups"] == 12
    assert result.provenance["data_hash"] and result.extra["replay_groups"]["panels_collected"] is False


@pytest.mark.parametrize("estimator,family",[("melogit","logit"),("meprobit","probit"),("mepoisson","poisson"),
                                             ("xtlogit","logit"),("xtprobit","probit"),("xtpoisson","poisson")])
@pytest.mark.parametrize("method",["mcaghermite","mvaghermite"])
def test_adaptive_all_six_native_point_and_full_covariance_contract(estimator,family,method,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data(family,n_groups=10,size=9)
    requested = spec(estimator,method=method,points=5)
    if estimator == "xtpoisson":
        requested.options["normal"] = True
    result = fit_streaming_glmm(requested,Dataset.from_frame(frame),batch_rows=4)
    dense = _DENSE[estimator](requested,frame)
    np.testing.assert_allclose(params(result),params(dense),rtol=2e-6,atol=3e-7)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=3e-6,atol=4e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(dense.metrics["log_likelihood"],rel=2e-9,abs=2e-8)
    assert [row.term for row in result.coefficients] == [row.term for row in dense.coefficients]
    assert supports_spec(requested)
    assert ResultBundle.model_validate_json(result.model_dump_json()).metrics == result.metrics
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind",["robust","cluster"])
def test_two_dimensional_random_slope_and_full_enclosing_group_sandwich(kind):
    frame = data("poisson",n_groups=16,size=7,random_slope=True)
    requested = spec("mepoisson",kind,method="mcaghermite",points=5,random=["z"])
    requested.columns.update({"offset":"offset","exposure":"exposure"})
    result = fit_streaming_glmm(requested,Dataset.from_frame(frame),batch_rows=5)
    dense = fit_mepoisson(requested,frame)
    np.testing.assert_allclose(params(result),params(dense),rtol=4e-6,atol=4e-7)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=5e-6,atol=5e-7)
    assert result.inference["cluster_count"] == (16 if kind == "robust" else 8)
    assert [row.equation for row in result.coefficients] == ["y","y","g","g"]


def conditional_value(beta,frame,logit):
    value = 0.
    for group in pd.unique(frame.g):
        part = frame[frame.g == group]
        y,x = part.y.to_numpy(),part.x.to_numpy()
        total = int(y.sum())
        if not total or (logit and total == len(part)):
            continue
        eta = x*beta[0]
        if logit:
            indices = np.array(list(combinations(range(len(part)),total)))
            value += y@eta-logsumexp(eta[indices].sum(1))
        else:
            value += gammaln(total+1)-gammaln(y+1).sum()+y@eta-total*logsumexp(eta)
    return value


@pytest.mark.parametrize("estimator,family",[("xtlogit","logit"),("xtpoisson","poisson")])
def test_conditional_independent_subset_or_multinomial_likelihood_and_dropped_sample(estimator,family):
    frame = data(family,n_groups=12,size=5)
    frame.loc[frame.g == 0,"y"] = 0
    if family == "logit":
        frame.loc[frame.g == 1,"y"] = 1
    requested = spec(estimator,model="fe")
    result = fit_streaming_glmm(requested,Dataset.from_frame(frame),batch_rows=4)
    def function(value):
        return conditional_value(value,frame,family == "logit")
    oracle = minimize(lambda value:-function(value),np.zeros(1),jac=lambda value:-finite_gradient(function,value),method="BFGS",options={"gtol":1e-8})
    np.testing.assert_allclose(params(result),oracle.x,rtol=2e-6,atol=2e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(function(params(result)),rel=3e-12)
    expected = np.linalg.inv(-finite_hessian(function,params(result)))
    np.testing.assert_allclose(result.covariance_matrix,expected,rtol=3e-5,atol=3e-7)
    dense = _DENSE[estimator](requested,frame)
    assert result.nobs == dense.nobs and result.nobs_original == len(frame)
    assert result.metrics["n_groups_dropped"] == dense.metrics["n_groups_dropped"]
    assert result.dropped_rows == len(frame)-result.nobs


@pytest.mark.parametrize("model,kind",[("re","nonrobust"),("re","robust"),("re","cluster"),("fe","robust")])
def test_gamma_and_conditional_poisson_closed_form_native_counts_and_covariance(model,kind):
    frame = data("poisson",n_groups=14,size=7)
    requested = spec("xtpoisson",kind,model=model)
    requested.columns.update({"offset":"offset","exposure":"exposure"})
    result = fit_streaming_glmm(requested,Dataset.from_frame(frame),batch_rows=4)
    dense = fit_xtpoisson(requested,frame)
    np.testing.assert_allclose(params(result),params(dense),rtol=3e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=3e-8,atol=3e-10)
    assert result.metrics["log_likelihood"] == pytest.approx(dense.metrics["log_likelihood"],rel=3e-12)


def test_actual_parquet_large_single_group_blocks_missing_categories_and_cpu_rng(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data("poisson",n_groups=8,size=18)
    frame["category"] = pd.Categorical(np.where(np.arange(len(frame))%2,"B","A"))
    frame.loc[7,"x"] = np.nan
    path = tmp_path/"panel.parquet"
    frame.to_parquet(path,index=False)
    requested = spec("xtpoisson")
    requested.predictors += ["category"]
    requested.categorical = ["category"]
    requested.missing = "drop"
    rng = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        result = fit_streaming_glmm(requested,scan(path),batch_rows=3)
        assert torch.empty(0).device.type == "meta"
    assert torch.equal(rng,torch.random.get_rng_state())
    assert result.nobs == len(frame)-1 and result.nobs_original == len(frame)
    assert result.provenance["streaming"]["maximum_batch_rows"] <= 3
    assert result.metrics["group_size_max"] == 18
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("defect,code",[("binary","invalid_binary_outcome"),("count","invalid_count_outcome"),
                                       ("constant","constant_outcome"),("group","insufficient_groups"),
                                       ("slope","collinear_random_effects"),("cluster","cluster_not_nested"),
                                       ("exposure","invalid_exposure"),("time","repeated_time_values"),
                                       ("probit_fe","invalid_spec"),("pa","unsupported_streaming_option")])
def test_explicit_domains_and_owned_scratch_cleanup(defect,code,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data("poisson" if defect in {"count","exposure","time"} else "logit")
    requested = spec("xtpoisson" if defect in {"count","exposure","time"} else "melogit")
    if defect == "binary":
        frame.loc[0,"y"] = 2
    elif defect == "count":
        frame.loc[0,"y"] = -1
    elif defect == "constant":
        frame.y = 0
    elif defect == "group":
        frame.g = 0
    elif defect == "slope":
        requested.columns["random"] = ["z"]
        frame.z = 1.
    elif defect == "cluster":
        requested.covariance,requested.cluster = "cluster","cluster"
        frame.loc[0,"cluster"] = 999
    elif defect == "exposure":
        requested.columns["exposure"] = "exposure"
        frame.loc[0,"exposure"] = 0.
    elif defect == "time":
        requested.time = "t"
        frame.loc[1,"t"] = frame.t.iloc[0]
    elif defect == "probit_fe":
        requested = spec("xtprobit",model="fe")
    else:
        requested = spec("xtlogit",model="pa")
    with pytest.raises(AnalysisError) as error:
        fit_streaming_glmm(requested,Dataset.from_frame(frame),batch_rows=4)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_workspace_source_replay_integrity_and_conditional_actual_work_guards(tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    frame = data("poisson",n_groups=6,size=8)
    with use_workspace_budget(8),pytest.raises(AnalysisError) as error:
        fit_streaming_glmm(spec("xtpoisson"),Dataset.from_frame(frame),batch_rows=4)
    assert error.value.code == "workspace_limit"
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        copied = frame.copy()
        if calls >= 5:
            copied.loc[2,"y"] += 1
        for start in range(0,len(frame),4):
            yield copied.iloc[start:start+4]
    with pytest.raises(AnalysisError) as error:
        fit_streaming_glmm(spec("xtpoisson"),Dataset.from_batches(factory,columns=list(frame)),batch_rows=4)
    assert error.value.code == "source_changed"
    from openecon.econometrics import streaming_glmm
    monkeypatch.setattr(streaming_glmm,"CONDITIONAL_WORK_LIMIT",1)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_glmm(spec("xtlogit",model="fe"),Dataset.from_frame(data()),batch_rows=4)
    assert error.value.code == "conditional_work_limit"
    assert not list(tmp_path.iterdir())


def test_conditional_single_informative_panel_remains_statistically_valid():
    frame = data("poisson",n_groups=3,size=12)
    frame.loc[frame.g != 1,"y"] = 0
    requested = spec("xtpoisson",model="fe")
    result = fit_streaming_glmm(requested,Dataset.from_frame(frame),batch_rows=3)
    dense = fit_xtpoisson(requested,frame)
    assert result.nobs == 12 and result.metrics["n_groups"] == 1
    np.testing.assert_allclose(params(result),params(dense),rtol=3e-8,atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix,dense.covariance_matrix,rtol=3e-8,atol=3e-10)


def test_independent_integrated_group_score_sandwich_matches_full_covariance():
    frame = data()
    result = fit_streaming_glmm(spec("melogit","robust",points=11),Dataset.from_frame(frame),batch_rows=4)
    values = params(result)
    theta = np.array([values[0],values[1],.5*np.log(values[2])])
    def function(value):
        return quadrature_value(value,frame,"logit",11)
    group_scores = []
    for group in pd.unique(frame.g):
        part = frame[frame.g == group]
        def group_value(value):
            return quadrature_value(value,part,"logit",11)
        group_scores.append(finite_gradient(group_value,theta))
    score = np.array(group_scores)
    bread = np.linalg.inv(-finite_hessian(function,theta))
    expected = bread@(score.T@score*len(score)/(len(score)-1))@bread
    jacobian = np.diag([1,1,2*values[2]])
    np.testing.assert_allclose(result.covariance_matrix,jacobian@expected@jacobian,rtol=3e-4,atol=1e-5)


@pytest.mark.parametrize("defect,code",[("method","invalid_spec"),("points","invalid_spec"),("covariance","unsupported_covariance")])
def test_malformed_copied_specs_never_select_an_implicit_integration_or_covariance(defect,code,tmp_path,monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY",str(tmp_path))
    requested = spec("melogit")
    if defect == "method":
        requested.options["intmethod"] = "approximate"
    elif defect == "points":
        requested.options["intpoints"] = True
    else:
        requested.covariance = "HC3"
    with pytest.raises(AnalysisError) as error:
        fit_streaming_glmm(requested,Dataset.from_frame(data()),batch_rows=4)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_random_slope_two_dimensional_independent_product_quadrature_derivatives():
    frame = data("poisson",n_groups=16,size=7,random_slope=True)
    result = fit_streaming_glmm(spec("mepoisson",random=["z"],points=5),Dataset.from_frame(frame),batch_rows=5)
    values = params(result)
    theta = np.r_[values[:2],.5*np.log(values[2:])]
    nodes,weights = hermgauss(5)
    grid = np.array(np.meshgrid(np.sqrt(2)*nodes,np.sqrt(2)*nodes,indexing="ij")).reshape(2,-1).T
    log_weight = np.log(np.outer(weights,weights).flatten()/np.pi)
    def function(value):
        answer = 0.
        for group in pd.unique(frame.g):
            part = frame[frame.g == group]
            random_design = np.column_stack([part.z.to_numpy(),np.ones(len(part))])
            index = (value[0]+value[1]*part.x.to_numpy())[:,None]+random_design@(grid*np.exp(value[2:])).T
            y = part.y.to_numpy()[:,None]
            answer += logsumexp(log_weight+(y*index-np.exp(index)-gammaln(y+1)).sum(0))
        return answer
    assert result.metrics["log_likelihood"] == pytest.approx(function(theta),rel=2e-12)
    np.testing.assert_allclose(finite_gradient(function,theta),np.zeros(4),atol=3e-6)
    bread = np.linalg.inv(-finite_hessian(function,theta))
    jacobian = np.diag(np.r_[1,1,2*values[2:]])
    np.testing.assert_allclose(result.covariance_matrix,jacobian@bread@jacobian,rtol=5e-4,atol=2e-5)
