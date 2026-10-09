"""Independent special-case laws and dense GLS/profile-information contrasts.

Closed-form df equalities hold only in the specified balanced orthogonal
Gaussian geometries. General Satterthwaite inference remains approximate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def forbid_fit(*args, **kwargs):
    raise AssertionError("Saved Satterthwaite contrast may not refit")


@pytest.fixture(scope="module")
def balanced():
    rng = np.random.default_rng(123)
    groups, size = 32, 6
    group = np.repeat(np.arange(groups), size)
    x = np.tile(np.linspace(-1, 1, size), groups)
    y0 = 1.4 + rng.normal(0, 1.7, groups)[group] + rng.normal(0, .8, groups * size)
    frame = pd.DataFrame({"g": group, "x": x, "y0": y0, "y": y0 + .6*x})
    mean = oe.mixed(data=frame, y="y0", group="g", method="reml")
    within = oe.mixed(data=frame, y="y", x=["x"], group="g", method="reml")
    return frame, mean, within, groups, size


def assert_inference(row, estimate, variance, df, null=.25, alpha=.1):
    se = math.sqrt(variance)
    statistic = (estimate-null)/se
    assert row.estimate == pytest.approx(estimate, rel=1e-7, abs=1e-9)
    assert row.std_error == pytest.approx(se, rel=1e-7, abs=1e-9)
    assert row.df == pytest.approx(df, rel=2e-6, abs=2e-6)
    assert row.t == pytest.approx(statistic, rel=1e-7, abs=1e-9)
    assert row.p_value == pytest.approx(2*stats.t.sf(abs(statistic),df), rel=2e-6, abs=1e-10)
    np.testing.assert_allclose([row.ci_low,row.ci_high], estimate+np.array([-1,1])*stats.t.ppf(1-alpha/2,df)*se,
                               rtol=2e-6, atol=1e-9)


def test_balanced_reml_intercept_df_from_independent_between_group_mean_law(balanced):
    frame, fitted, _, groups, _ = balanced
    means = frame.groupby("g").y0.mean().to_numpy()
    estimate = means.mean()
    variance = means.var(ddof=1)/groups
    result = oe.mixed_satterthwaite(fitted, data=frame, contrast={"Intercept":1}, null=.25, alpha=.1)
    assert_inference(result["contrast"].iloc[0],estimate,variance,groups-1)


def test_balanced_centered_within_slope_df_from_orthogonal_residual_law(balanced):
    frame, _, fitted, groups, size = balanced
    x = frame.x.to_numpy()
    centered_y = frame.y-frame.groupby("g").y.transform("mean")
    estimate = x @ centered_y.to_numpy()/(x@x)
    df = groups*(size-1)-1
    sigma2 = np.square(centered_y.to_numpy()-x*estimate).sum()/df
    variance = sigma2/(x@x)
    result = oe.mixed_satterthwaite(fitted, data=frame, contrast={"x":1}, null=.25, alpha=.1)
    assert_inference(result["contrast"].iloc[0],estimate,variance,df)


@pytest.fixture(scope="module")
def unbalanced():
    rng = np.random.default_rng(841)
    sizes = rng.integers(4,8,size=28)
    group = np.repeat(np.arange(len(sizes)),sizes)
    x = rng.normal(.2,1,size=len(group))
    effects = rng.multivariate_normal([0,0],[[.65,.2],[.2,1.5]],size=len(sizes))
    y = 1.2+.8*x+effects[group,0]*x+effects[group,1]+rng.normal(0,.7,len(group))
    frame = pd.DataFrame({"g":group,"x":x,"y":y}, index=[f"physical-{i}" for i in range(len(group))])
    return frame, {method:oe.mixed(data=frame,y="y",x=["x"],group="g",random=["x"],
                                  covstructure="unstructured",method=method) for method in ("ml","reml")}


def dense_gls(frame, theta, reml):
    """Explicit per-group V in NumPy, independently profiled fixed effects."""
    lower = np.array([[np.exp(theta[0]),0],[theta[2],np.exp(theta[1])]])
    random_cov = lower@lower.T
    sigma2 = np.exp(2*theta[-1])
    x = np.column_stack([np.ones(len(frame)),frame.x.to_numpy()])
    y = frame.y.to_numpy()
    information, xy, quadratic, logdet = np.zeros((2,2)),np.zeros(2),0.,0.
    for group in pd.unique(frame.g):
        positions = np.flatnonzero(frame.g.to_numpy()==group)
        z = np.column_stack([frame.x.to_numpy()[positions],np.ones(len(positions))])
        covariance = sigma2*np.eye(len(positions))+z@random_cov@z.T
        vx = np.linalg.solve(covariance,x[positions])
        vy = np.linalg.solve(covariance,y[positions])
        information += x[positions].T@vx
        xy += x[positions].T@vy
        quadratic += y[positions]@vy
        logdet += np.linalg.slogdet(covariance)[1]
    covariance_beta = np.linalg.inv(information)
    beta = covariance_beta@xy
    restricted = np.linalg.slogdet(information)[1] if reml else 0.
    log_likelihood = -.5*((len(frame)-(2 if reml else 0))*np.log(2*np.pi)+logdet+restricted+quadratic-xy@beta)
    return log_likelihood,beta,covariance_beta


def centered_hessian(function, x, step=2e-4):
    value = function(x)
    directions = np.eye(len(x))*step
    output = np.empty((len(x),len(x)))
    for i,di in enumerate(directions):
        output[i,i] = (function(x+di)-2*value+function(x-di))/step**2
        for j in range(i):
            dj = directions[j]
            output[i,j] = output[j,i] = (function(x+di+dj)-function(x+di-dj)-function(x-di+dj)+function(x-di-dj))/(4*step**2)
    return output


@pytest.mark.parametrize("method", ["ml","reml"])
def test_unbalanced_correlated_random_slope_df_full_profile_information_and_gls(method,unbalanced,monkeypatch):
    frame,fits = unbalanced
    fitted = fits[method]
    theta = np.asarray(fitted.extra["theta"])
    vector = np.array([.3,1.])
    reml = method=="reml"
    likelihood,beta,covariance_beta = dense_gls(frame,theta,reml)
    assert likelihood == pytest.approx(fitted.metrics["log_likelihood"],abs=1e-7)
    information = -centered_hessian(lambda point:dense_gls(frame,point,reml)[0],theta)
    covariance_theta = np.linalg.inv(information)
    variance = vector@covariance_beta@vector
    directions = np.eye(len(theta))*1e-5
    gradient = np.array([(vector@dense_gls(frame,theta+d,reml)[2]@vector-vector@dense_gls(frame,theta-d,reml)[2]@vector)/2e-5
                         for d in directions])
    df = 2*variance**2/(gradient@covariance_theta@gradient)
    from openecon.econometrics.mixed import lmm
    from openecon.engines import optimize
    monkeypatch.setattr(lmm,"_maximize",forbid_fit)
    monkeypatch.setattr(lmm,"fit_mixed",forbid_fit)
    monkeypatch.setattr(optimize,"maximize_bfgs",forbid_fit)
    result = oe.mixed_satterthwaite(fitted,data=frame,contrast={"Intercept":.3,"x":1},null=.25,alpha=.1)
    row = result["contrast"].iloc[0]
    assert_inference(row,vector@beta,variance,df)
    np.testing.assert_allclose(result["variance_covariance"].to_numpy(),covariance_theta,rtol=2e-5,atol=2e-7)
    np.testing.assert_allclose(result["variance_coordinates"].gradient,gradient,rtol=2e-6,atol=2e-9)
    assert result.attrs["refitted"] is False
    restored_fit = ResultBundle.model_validate_json(fitted.model_dump_json())
    second = oe.mixed_satterthwaite(restored_fit,data=frame,contrast={"Intercept":.3,"x":1},null=.25,alpha=.1)
    pd.testing.assert_frame_equal(second["contrast"],result["contrast"],check_exact=True)
    restored_summary = oe.restore_summary(oe.summary_state(result))
    for name in result:
        pd.testing.assert_frame_equal(restored_summary[name],result[name],check_dtype=False,check_exact=True)


@pytest.mark.parametrize("corruption", ["theta","theta_se","negative_theta_se","beta","fixed_se","tested_covariance","other_fixed_covariance","ragged_covariance","nonnumeric_theta","criterion"])
def test_saved_geometry_and_uncertainty_corruption_is_rejected(corruption,balanced):
    frame,_,baseline,_,_ = balanced
    fitted = baseline.model_copy(deep=True)
    if corruption=="theta":
        fitted.extra["theta"][0] += .1
    elif corruption=="theta_se":
        fitted.extra["theta_std_error"][0] *= 1.1
    elif corruption=="negative_theta_se":
        fitted.extra["theta_std_error"][0] *= -1
    elif corruption=="beta":
        fitted.coefficients[0].estimate += .1
    elif corruption=="fixed_se":
        fitted.coefficients[0].std_error *= 1.2
    elif corruption=="tested_covariance":
        fitted.covariance_matrix[1][1] *= 1.2
    elif corruption=="other_fixed_covariance":
        fitted.covariance_matrix[0][0] *= 1.2
    elif corruption=="ragged_covariance":
        fitted.covariance_matrix[0] = fitted.covariance_matrix[0][:-1]
    elif corruption=="nonnumeric_theta":
        fitted.extra["theta"][0] = "unavailable"
    else:
        fitted.extra["method"] = "ml"
    with pytest.raises(AnalysisError,match="Saved|saved|Variance coordinates"):
        oe.mixed_satterthwaite(fitted,data=frame,contrast={"x":1})


@pytest.mark.parametrize("change", ["row_order","outcome","index","missing"])
def test_exact_original_source_and_joint_missing_geometry(change,balanced):
    frame,_,fitted,_,_ = balanced
    changed = frame.copy()
    if change=="row_order":
        changed = changed.iloc[::-1]
    elif change=="outcome":
        changed.loc[0,"y"] += .1
    elif change=="index":
        changed.index = [f"other-{i}" for i in range(len(changed))]
        # Shared model source hashes encode physical row values/order, not
        # cosmetic index labels; source positions remain the same geometry.
        original = oe.mixed_satterthwaite(fitted,data=frame,contrast={"x":1})
        relabeled = oe.mixed_satterthwaite(fitted,data=changed,contrast={"x":1})
        pd.testing.assert_frame_equal(original["contrast"],relabeled["contrast"],check_exact=True)
        return
    else:
        changed.loc[0,"x"] = np.nan
    with pytest.raises(AnalysisError):
        oe.mixed_satterthwaite(fitted,data=changed,contrast={"x":1})


def test_joint_missing_positions_saved_replay_and_invalid_contrast_budget(balanced,monkeypatch):
    frame,_,_,_,_ = balanced
    missing = frame.copy()
    missing.loc[0,"x"] = np.nan
    missing.loc[8,"y"] = np.nan
    fitted = oe.mixed(data=missing,y="y",x=["x"],group="g",method="reml",missing="drop")
    restored = ResultBundle.model_validate_json(fitted.model_dump_json())
    result = oe.mixed_satterthwaite(restored,data=missing,contrast={"x":1})
    assert result.attrs["source_sample_positions"] == [i for i in range(len(frame)) if i not in (0,8)]
    for contrast in ({},{"/var(Residual)":1},{"x":0},{"x":True},{"x":np.inf}):
        with pytest.raises(AnalysisError):
            oe.mixed_satterthwaite(restored,data=missing,contrast=contrast)
    large = pd.concat([frame.assign(g=frame.g+32*offset) for offset in range(12)],ignore_index=True)
    large_fit = oe.mixed(data=large,y="y",x=["x"],group="g",method="reml")
    from openecon.econometrics.mixed import contrasts
    monkeypatch.setattr(contrasts,"_likelihood",forbid_fit)
    with pytest.raises(AnalysisError,match="max_work"):
        oe.mixed_satterthwaite(restored,data=missing,contrast={"x":1},max_work=1)
    with use_workspace_budget(1),pytest.raises(AnalysisError):
        oe.mixed_satterthwaite(large_fit,data=large,contrast={"x":1})
