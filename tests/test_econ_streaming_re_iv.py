"""G2SLS global replay, full current covariance and independent NumPy algebra."""
import numpy as np
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.econometrics.iv.xtivreg import fit_xtivreg
from openecon.econometrics.streaming_re_iv import fit_streaming_re_iv
from openecon.models import ModelSpec, ResultBundle


def data(unbalanced=False):
    rng = np.random.default_rng(151933)
    g, t = np.tile(np.arange(37), 19), np.repeat(np.arange(19), 37)
    x, z1, z2, v, noise = rng.normal(size=(5, len(g)))
    effects = rng.normal(size=37)*.8
    e = .4*x+.7*z1-.3*z2+v+.05*g
    frame = pd.DataFrame({"g": g, "t": t, "x": x, "z1": z1, "z2": z2, "endog": e,
        "y": .7+.5*x+1.2*e+effects[g]+noise*.6+v*.3,
        "cat": np.where(g%3==0, "A", "B"), "constant": g*.2, "cluster": g%9})
    if unbalanced:
        frame = frame[~((frame.g%4==0)&(frame.t>6))]
    return frame.sample(frac=1, random_state=7849).reset_index(drop=True)


def spec(*, small=False, covariance="nonrobust", categorical=False, time=True):
    return ModelSpec(estimator="xtivreg", outcome="y", predictors=["x", *(["cat", "constant"] if categorical else [])],
        categorical=["cat"] if categorical else [], panel="g", time="t" if time else None, covariance=covariance,
        cluster="cluster" if covariance=="cluster" else None,
        columns={"endogenous": ["endog"], "instruments": ["z1", "z2"]}, options={"model": "re", "small": small})


@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_global_g2sls_current_options_dense_parity(unbalanced, small, covariance):
    frame = data(unbalanced)
    s = spec(small=small, covariance=covariance, categorical=True)
    a, b = fit_xtivreg(s, frame), fit_streaming_re_iv(s, Dataset.from_frame(frame), batch_rows=17)
    assert [c.term for c in a.coefficients] == [c.term for c in b.coefficients]
    np.testing.assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=2e-9, atol=2e-10)
    for key, value in a.metrics.items():
        assert b.metrics[key] == (None if value is None else pytest.approx(value, rel=2e-10, abs=2e-10)), key
    for key in ["variance_components", "theta"]:
        for name, value in a.extra[key].items():
            assert b.extra[key][name] == (pytest.approx(value, rel=2e-10, abs=2e-10) if isinstance(value, (float, int)) else value), name
    for key in ["endogenous", "instruments", "omitted_instruments", "instrument_columns_used", "estimator"]:
        assert b.extra[key] == a.extra[key]
    assert b.nobs == a.nobs
    assert b.nobs_original == len(frame)
    assert b.dropped_rows == a.dropped_rows
    assert b.inference["df_inference"] == a.inference["df_inference"]
    assert b.provenance["streaming"]["maximum_batch_rows"]<=17
    ResultBundle.model_validate_json(b.model_dump_json())


def tsls(x, y, z, weights=None):
    if weights is not None:
        x, y, z = x*np.sqrt(weights)[:, None], y*np.sqrt(weights), z*np.sqrt(weights)[:, None]
    xhat = z@np.linalg.lstsq(z, x, rcond=None)[0]
    beta = np.linalg.lstsq(xhat, y, rcond=None)[0]
    residual = y-x@beta
    return beta, residual, np.linalg.inv(xhat.T@xhat)


def oracle(frame, covariance):
    x = np.column_stack([np.ones(len(frame)), frame[["x", "endog"]]])
    z = np.column_stack([np.ones(len(frame)), frame[["x", "z1", "z2"]]])
    y = frame.y.to_numpy()
    groups = np.unique(frame.g)
    xm = np.vstack([x[frame.g==g].mean(0) for g in groups])
    zm = np.vstack([z[frame.g==g].mean(0) for g in groups])
    ym = np.array([y[frame.g==g].mean() for g in groups])
    sizes = np.array([sum(frame.g==g) for g in groups])
    _, wr, _ = tsls(x[:, 1:]-xm[frame.g, 1:], y-ym[frame.g], z[:, 1:]-zm[frame.g, 1:])
    se2 = sum(wr**2)/(len(y)-len(groups)-2)
    _, br, _ = tsls(xm, ym, zm, sizes)
    ssrb = sum(br**2)
    trace = np.trace(np.linalg.solve((xm*sizes[:, None]).T@xm, (xm*sizes[:, None]).T@(xm*sizes[:, None])))
    su2 = max(0., (ssrb-(len(groups)-3)*se2)/(len(y)-trace))
    theta = 1-np.sqrt(se2/(sizes*su2+se2))
    xt, zt, yt = x-theta[frame.g, None]*xm[frame.g], z-theta[frame.g, None]*zm[frame.g], y-theta[frame.g]*ym[frame.g]
    beta, residual, bread = tsls(xt, yt, zt)
    if covariance=="nonrobust":
        cov = bread*sum(residual**2)/(len(y)-3)
    else:
        scores = (zt@np.linalg.lstsq(zt, xt, rcond=None)[0])*residual[:, None]
        labels = frame.g if covariance=="robust" else frame.cluster
        group_scores = np.vstack([scores[labels==g].sum(0) for g in np.unique(labels)])
        count = len(group_scores)
        cov = bread@(group_scores.T@group_scores)@bread*count/(count-1)*(len(y)-1)/(len(y)-3)
    return beta, cov, su2, se2


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("unbalanced", [False, True])
def test_independent_numpy_auxiliary_iv_and_final_actual_score_oracle(covariance, unbalanced):
    frame = data(unbalanced)
    expected = oracle(frame, covariance)
    result = fit_streaming_re_iv(spec(covariance=covariance), Dataset.from_frame(frame), batch_rows=11)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], expected[0], rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(result.covariance_matrix, expected[1], rtol=2e-11, atol=2e-11)
    assert result.extra["variance_components"]["sigma_u_squared"] == pytest.approx(expected[2], rel=2e-11)
    assert result.extra["variance_components"]["sigma_e_squared"] == pytest.approx(expected[3], rel=2e-11)
