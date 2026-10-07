"""Panel first-difference 2SLS dense parity and independent projection oracle."""
import numpy as np
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.econometrics.iv.xtivreg import fit_xtivreg
from openecon.econometrics.streaming_fd_iv import fit_streaming_fd_iv
from openecon.models import ModelSpec, ResultBundle


def data(*, gaps=False, dates=False):
    rng = np.random.default_rng(88132)
    g, t = np.tile(np.arange(31), 17), np.repeat(np.arange(17), 31)
    x, z1, z2, v, noise = rng.normal(size=(5, len(g)))
    e = .5*x+.8*z1-.4*z2+v+.1*g
    frame = pd.DataFrame({"g": g, "t": t, "x": x, "endog": e, "z1": z1, "z2": z2,
        "y": 1+.3*x+1.1*e+.04*g+noise+.3*v,
        "cat": np.where((t+g)%3==0, "A", "B"), "cluster": (g+t)%7})
    if gaps:
        frame = frame[~((frame.g%4==0)&(frame.t%5==0))]
    if dates:
        frame.t = pd.Timestamp("2020-01-01", tz="Europe/Istanbul")+pd.to_timedelta(frame.t*3, unit="D")
    return frame.sample(frac=1, random_state=753).reset_index(drop=True)


def spec(*, small=False, covariance="nonrobust", categorical=False):
    return ModelSpec(estimator="xtivreg", outcome="y", predictors=["x", *(["cat"] if categorical else [])],
        categorical=["cat"] if categorical else [], panel="g", time="t", covariance=covariance,
        cluster="cluster" if covariance=="cluster" else None,
        columns={"endogenous": ["endog"], "instruments": ["z1", "z2"]}, options={"model": "fd", "small": small})


@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("gaps,dates", [(False, False), (True, False), (True, True)])
def test_all_current_fd_iv_covariances_small_gaps_and_dates(small, covariance, gaps, dates):
    frame = data(gaps=gaps, dates=dates)
    s = spec(small=small, covariance=covariance, categorical=True)
    a, b = fit_xtivreg(s, frame), fit_streaming_fd_iv(s, Dataset.from_frame(frame), batch_rows=17)
    assert [c.term for c in a.coefficients] == [c.term for c in b.coefficients]
    np.testing.assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=2e-9, atol=2e-10)
    for key, value in a.metrics.items():
        assert b.metrics[key] == (None if value is None else pytest.approx(value, rel=2e-10, abs=2e-10)), key
    assert a.extra.keys() == b.extra.keys()
    for key, value in a.extra.items():
        assert b.extra[key] == (pytest.approx(value, rel=2e-10, abs=2e-10) if isinstance(value, (float, int)) else value), key
    assert b.nobs == a.nobs
    assert b.nobs_original == len(frame)
    assert b.dropped_rows == a.dropped_rows
    assert not b.sample_positions
    assert b.provenance["sample_position_count"] == b.nobs
    assert b.provenance["streaming"]["maximum_batch_rows"]<=17
    expected = {item["row"]: item for item in a.predictions}
    for item in b.predictions:
        if item["row"] in expected:
            for key in ["observed", "fitted", "residual"]:
                assert item[key] == pytest.approx(expected[item["row"]][key], rel=2e-10, abs=2e-10)
    ResultBundle.model_validate_json(b.model_dump_json())


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_independent_numpy_actual_difference_projection_and_sandwich(covariance):
    frame = data(gaps=True).sort_values(["g", "t"])
    valid = (frame.g.diff()==0)&(frame.t.diff()==1)
    differences = frame[["y", "x", "endog", "z1", "z2"]].diff().loc[valid]
    x = np.column_stack((np.ones(len(differences)), differences[["x", "endog"]]))
    z = np.column_stack((np.ones(len(differences)), differences[["x", "z1", "z2"]]))
    y = differences.y.to_numpy()
    xhat = z@np.linalg.lstsq(z, x, rcond=None)[0]
    beta = np.linalg.lstsq(xhat, y, rcond=None)[0]
    bread = np.linalg.inv(xhat.T@xhat)
    residual = y-x@beta
    n, k = x.shape
    if covariance=="nonrobust":
        cov = bread*sum(residual**2)/(n-k)
    else:
        groups = frame.loc[valid, "g" if covariance=="robust" else "cluster"]
        scores = xhat*residual[:, None]
        group_scores = np.vstack([scores[groups==group].sum(0) for group in np.unique(groups)])
        g = len(group_scores)
        cov = bread@(group_scores.T@group_scores)@bread*g/(g-1)*(n-1)/(n-k)
    result = fit_streaming_fd_iv(spec(covariance=covariance), Dataset.from_frame(frame.reset_index(drop=True)), batch_rows=11)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], beta, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(result.covariance_matrix, cov, rtol=3e-11, atol=3e-11)
