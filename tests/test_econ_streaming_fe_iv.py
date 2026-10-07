"""Native retained-row FE IV parity and independent weighted dummy oracle."""
import numpy as np
import pandas as pd
import pytest
import openecon as oe

from openecon.dataset import Dataset
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.iv.xtivreg import fit_xtivreg
from openecon.econometrics.iv.ivreghdfe import fit_ivreghdfe
from openecon.econometrics.streaming_fe_iv import fit_streaming_fe_iv
from openecon.models import ModelSpec, ResultBundle


def data(seed=757319, n=311):
    rng = np.random.default_rng(seed)
    x, z1, z2, v, error = rng.normal(size=(5, n))
    f = pd.DataFrame({"x": x+.4, "z1": z1-.3, "z2": z2+.7, "f0": np.arange(n)%23,
                      "f1": rng.integers(0, 11, n), "cluster": np.arange(n)%17,
                      "parent": (np.arange(n)%23)//3, "weight": np.arange(n)%4+1,
                      "cat": pd.Categorical(np.arange(n)%3, categories=[0, 1, 2, 3])})
    f["endog"] = .4*x+.9*z1-.2*z2+v+.02*f.f0
    f["y"] = .7+.3*f.x+1.2*f.endog+.06*f.f0-.03*f.f1+error+.3*v
    return f


def spec(estimator="xtivreg", model="fe", covariance="nonrobust", cluster=None,
         weight_type=None, small=True, dimensions=2, categorical=False):
    return ModelSpec(estimator=estimator, outcome="y", predictors=["x", *(["cat"] if categorical else [])],
                     categorical=["cat"] if categorical else [], intercept=estimator == "xtivreg",
                     panel="f0" if estimator == "xtivreg" else None,
                     covariance=covariance, cluster=cluster,
                     weights="weight" if weight_type else None, weight_type=weight_type,
                     columns={"endogenous": ["endog"], "instruments": ["z1", "z2"],
                              **({"absorb": [f"f{i}" for i in range(dimensions)]} if estimator == "ivreghdfe" else {})},
                     options={"small": small, **({"model": model} if estimator == "xtivreg" else
                                              {"method": "2sls", "tolerance": 1e-10})})


def parity(dense, actual, tolerance=3e-8):
    assert [c.term for c in dense.coefficients] == [c.term for c in actual.coefficients]
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], [c.estimate for c in dense.coefficients], rtol=tolerance, atol=3e-10)
    np.testing.assert_allclose(actual.covariance_matrix, dense.covariance_matrix, rtol=tolerance, atol=3e-10)
    for key, value in dense.metrics.items():
        assert actual.metrics[key] == (None if value is None else pytest.approx(value, rel=tolerance, abs=3e-10)), key
    for name, expected in dense.tests.items():
        for key in ("statistic", "df", "df2", "p_value"):
            value = expected.get(key)
            if value is not None:
                assert actual.tests[name][key] == pytest.approx(value, rel=tolerance, abs=3e-9), (name, key)
    assert actual.nobs == dense.nobs
    assert actual.nobs_original == dense.nobs_original
    assert actual.dropped_rows == dense.dropped_rows
    assert actual.sample_positions == []
    assert not actual.provenance["streaming"]["dense_observation_matrix"]
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= 17
    assert ResultBundle.model_validate_json(actual.model_dump_json()).covariance_matrix == actual.covariance_matrix


@pytest.mark.parametrize("model", ["fe", "be"])
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("robust", None),
                                               ("cluster", "parent")])
def test_panel_fe_be_all_covariances_small_dense_parity(model, small, covariance, cluster):
    frame = data()
    s = spec(model=model, small=small, covariance=covariance, cluster=cluster, categorical=True)
    dense = fit_xtivreg(s, frame)
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    parity(dense, actual)


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("robust", None),
                                               ("cluster", "f0"), ("cluster", ["cluster", "f0"])])
@pytest.mark.parametrize("small", [False, True])
def test_hdfe_2sls_all_covariances_small_dense_parity(dimensions, covariance, cluster, small):
    frame = data()
    s = spec("ivreghdfe", covariance=covariance, cluster=cluster, small=small, dimensions=dimensions, categorical=True)
    dense = fit_ivreghdfe(s, frame)
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    parity(dense, actual)
    for row, expected in zip(actual.extra["first_stage"], dense.extra["first_stage"], strict=True):
        for key, value in expected.items():
            assert row[key] == (pytest.approx(value, rel=2e-7, abs=3e-9) if isinstance(value, (float, int)) else value), key


@pytest.mark.parametrize("weight_type", ["fweight", "aweight", "pweight"])
def test_hdfe_weighted_independent_full_dummy_projection_oracle(weight_type):
    frame = data()
    s = spec("ivreghdfe", covariance="robust", weight_type=weight_type)
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    raw_x, raw_z, y = frame[["x", "endog"]].to_numpy(), frame[["x", "z1", "z2"]].to_numpy(), frame.y.to_numpy()
    d = np.column_stack([pd.get_dummies(frame[name], dtype=float).to_numpy() for name in ("f0", "f1")])
    w = frame.weight.to_numpy(dtype=float)
    n = int(w.sum()) if weight_type == "fweight" else len(frame)
    if weight_type != "fweight":
        w *= len(frame)/w.sum()
    def residualize(values):
        return values-d@np.linalg.lstsq(d*w[:, None]**.5, values*w[:, None]**.5, rcond=None)[0]
    x, z, y = residualize(raw_x), residualize(raw_z), residualize(y[:, None])[:, 0]
    xhat = z@np.linalg.lstsq(z*w[:, None]**.5, x*w[:, None]**.5, rcond=None)[0]
    bread = np.linalg.inv(xhat.T@(xhat*w[:, None]))
    beta = bread@(xhat.T@(w*y))
    residual = y-x@beta
    scores = xhat*(residual*(w**.5 if weight_type == "fweight" else w))[:, None]
    df = n-np.linalg.matrix_rank(d)-len(beta)
    cov = bread@(scores.T@scores)@bread*n/df
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], beta, rtol=2e-9, atol=3e-11)
    np.testing.assert_allclose(actual.covariance_matrix, cov, rtol=3e-9, atol=3e-11)
    parity(fit_ivreghdfe(s, frame), actual)


@pytest.mark.parametrize("estimator,option", [("xtivreg", {"model": "re"}), ("xtivreg", {"model": "fd"})])
def test_unverified_methods_fail_explicitly_before_read(monkeypatch, estimator, option):
    s = spec(estimator).model_copy(update={"options": option})
    source = Dataset.from_frame(data())
    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified methods must fail before source iteration.")
    monkeypatch.setattr(source, "iter_batches", forbidden)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_fe_iv(s, source)
    assert error.value.code == "unsupported_streaming_method"


@pytest.mark.parametrize("method", ["liml", "gmm"])
def test_hdfe_verified_iv_methods_preserve_joint_covariance_and_saved_group_means(method):
    frame = data()
    specification = spec("ivreghdfe").model_copy(
        update={"options": {"method": method, "small": True, "tolerance": 1e-10}}
    )
    dense = fit_ivreghdfe(specification, frame)
    actual = fit_streaming_fe_iv(specification, Dataset.from_frame(frame), batch_rows=17)
    parity(dense, actual)
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    output = oe.predict(restored, Dataset.from_frame(frame), kind="response", batch_rows=17)
    predicted = pd.concat(list(output.iter_batches()))
    # An independent full dummy projection supplies the absorbed mean at the
    # fitted IV coefficients; this is not an xb-only prediction.
    slopes = np.array([coefficient.estimate for coefficient in actual.coefficients])
    xb = frame[["x", "endog"]].to_numpy() @ slopes
    dummies = np.column_stack(
        [pd.get_dummies(frame[name], dtype=float).to_numpy() for name in ("f0", "f1")]
    )
    expected = xb + dummies @ np.linalg.lstsq(dummies, frame.y.to_numpy() - xb, rcond=None)[0]
    np.testing.assert_allclose(predicted.response, expected, rtol=3e-8, atol=3e-9)
    assert predicted.index.equals(frame.index)
