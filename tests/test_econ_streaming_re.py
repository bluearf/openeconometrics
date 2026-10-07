"""Random-effects global replay parity and independently assembled GLS oracle."""
import numpy as np
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.econometrics.core import make_spec
from openecon.econometrics.panel.xtreg import fit_xtreg
from openecon.econometrics.streaming_re import fit_streaming_re
from openecon.models import ResultBundle


def data(unbalanced=False, groups=31, periods=13):
    rng = np.random.default_rng(47381)
    g = np.tile(np.arange(groups), periods)
    t = np.repeat(np.arange(periods), groups)
    x, z = rng.normal(size=(2, len(g)))
    u = rng.normal(size=groups)*1.5
    frame = pd.DataFrame({"g": g, "t": t, "x": x, "z": z,
        "cat": np.where(g%3==0, "A", "B"), "constant": g*.1,
        "c1": g%7, "c2": g%9,
        "y": 1+.5*x-.3*z+u[g]+rng.normal(size=len(g))*.7})
    if unbalanced:
        frame = frame[~((frame.g%4==0)&(frame.t>4))]
    return frame.sample(frac=1, random_state=579).reset_index(drop=True)


def spec(*, sa=False, covariance="nonrobust", cluster=None, predictors=None, categorical=False, time=True):
    predictors = predictors or ["x", "z", *(["cat", "constant"] if categorical else [])]
    return make_spec("xtreg", outcome="y", predictors=predictors, panel="g", time="t" if time else None,
        covariance=covariance, cluster=cluster, categorical=["cat"] if categorical else [], options={"model": "re", "sa": sa})


@pytest.mark.parametrize("sa", [False, True])
@pytest.mark.parametrize("unbalanced", [False, True])
@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("robust", None), ("cluster", "c1"), ("cluster", ["c1", "c2"])])
def test_all_current_re_options_dense_parity(sa, unbalanced, covariance, cluster):
    frame = data(unbalanced)
    s = spec(sa=sa, covariance=covariance, cluster=cluster, categorical=True)
    a = fit_xtreg(s, frame)
    b = fit_streaming_re(s, Dataset.from_frame(frame), batch_rows=17)
    assert [c.term for c in a.coefficients] == [c.term for c in b.coefficients]
    np.testing.assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=2e-9, atol=2e-10)
    for key, value in a.metrics.items():
        assert b.metrics[key] == pytest.approx(value, rel=2e-10, abs=2e-11) if value is not None else b.metrics[key] is None
    for key in ["variance_components", "theta"]:
        for name, value in a.extra[key].items():
            assert b.extra[key][name] == (pytest.approx(value, rel=2e-10, abs=2e-11) if isinstance(value, (float, int)) else value)
    assert b.extra["ssr_transformed"] == pytest.approx(a.extra["ssr_transformed"], rel=2e-10)
    assert b.tests["breusch_pagan"]["statistic"] == pytest.approx(a.tests["breusch_pagan"]["statistic"], rel=2e-10)
    assert b.nobs == a.nobs
    assert b.inference["df_inference"] is None
    assert not b.sample_positions
    assert b.provenance["streaming"]["maximum_batch_rows"]<=17
    expected = {item["row"]: item["fitted"] for item in a.predictions}
    for item in b.predictions:
        if item["row"] in expected:
            assert item["fitted"] == pytest.approx(expected[item["row"]], rel=2e-10, abs=2e-10)
    ResultBundle.model_validate_json(b.model_dump_json())


def oracle(frame, sa):
    x = np.column_stack([np.ones(len(frame)), frame[["x", "z"]]])
    y = frame.y.to_numpy()
    groups = np.unique(frame.g)
    means = np.vstack([x[frame.g==g].mean(0) for g in groups])
    my = np.array([y[frame.g==g].mean() for g in groups])
    sizes = np.array([sum(frame.g==g) for g in groups])
    mx = means[frame.g]
    wy = y-my[frame.g]
    wx = x[:, 1:]-mx[:, 1:]
    bw = np.linalg.lstsq(wx, wy, rcond=None)[0]
    rssw = sum((wy-wx@bw)**2)
    se2 = rssw/(len(y)-len(groups)-2)
    bb = np.linalg.lstsq(means, my, rcond=None)[0]
    rssb = sum((my-means@bb)**2)
    dfb = len(groups)-3
    if sa:
        weighted = means*np.sqrt(sizes)[:, None]
        target = my*np.sqrt(sizes)
        b = np.linalg.lstsq(weighted, target, rcond=None)[0]
        numerator = sum((target-weighted@b)**2)-dfb*se2
        denominator = len(y)-np.trace(np.linalg.solve(weighted.T@weighted, (means*sizes[:, None]).T@(means*sizes[:, None])))
        su2 = max(0., numerator/denominator)
    else:
        su2 = max(0., rssb/dfb-se2*np.mean(1/sizes))
    theta = 1-np.sqrt(se2/(sizes*su2+se2))
    xs = x-theta[frame.g, None]*mx
    ys = y-theta[frame.g]*my[frame.g]
    beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
    residual = ys-xs@beta
    covariance = np.linalg.inv(xs.T@xs)*sum(residual**2)/(len(y)-3)
    return beta, covariance, su2, se2


@pytest.mark.parametrize("sa", [False, True])
@pytest.mark.parametrize("unbalanced", [False, True])
def test_independent_numpy_full_transformed_gls_oracle(sa, unbalanced):
    frame = data(unbalanced)
    expected = oracle(frame, sa)
    result = fit_streaming_re(spec(sa=sa), Dataset.from_frame(frame), batch_rows=11)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], expected[0], rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(result.covariance_matrix, expected[1], rtol=2e-11, atol=2e-11)
    assert result.extra["variance_components"]["sigma_u_squared"] == pytest.approx(expected[2], rel=2e-11)
    assert result.extra["variance_components"]["sigma_e_squared"] == pytest.approx(expected[3], rel=2e-11)


@pytest.mark.parametrize("sa", [False, True])
def test_no_time_and_within_collinear_regressors_retained_for_gls(sa):
    frame = data(True)
    frame["v"] = frame.x+frame.g*.4
    s = spec(sa=sa, predictors=["x", "z", "v", "constant"], time=False)
    a = fit_xtreg(s, frame)
    b = fit_streaming_re(s, Dataset.from_frame(frame), batch_rows=23)
    np.testing.assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=2e-9, atol=2e-10)
    assert b.extra["variance_components"]["within_regressors"] == 2
