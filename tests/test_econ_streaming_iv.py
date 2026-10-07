"""Native 2SLS replay parity including nonlinear score diagnostics."""
import numpy as np
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.econometrics.iv.ivregress import fit_ivregress
from openecon.econometrics.streaming_linear import fit_streaming_linear
from openecon.models import ModelSpec

from test_econ_streaming_linear import assert_parity


def iv_data():
    rng = np.random.default_rng(54671)
    n = 301
    x, z1, z2, e, v = rng.normal(size=(5, n))
    data = pd.DataFrame({"x": x, "z1": z1, "z2": z2, "endog": .4*x+.8*z1-.3*z2+v,
                         "cat": pd.Categorical(np.arange(n)%3, categories=[0, 1, 2, 3]),
                         "weight": np.arange(n)%4+1, "cluster": np.arange(n)%31,
                         "cluster2": np.arange(n)%19})
    data["y"] = .7+.3*x+1.4*data.endog+e+.4*v
    return data


def iv_spec(*, covariance="nonrobust", cluster=None, small=False, weight_type=None, intercept=True):
    return ModelSpec(estimator="ivregress", outcome="y", predictors=["x", "cat"],
                     categorical=["cat"], intercept=intercept, covariance=covariance, cluster=cluster,
                     weights="weight" if weight_type else None, weight_type=weight_type,
                     columns={"endogenous": ["endog"], "instruments": ["z1", "z2"]},
                     options={"method": "2sls", "small": small})


@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("robust", None),
                           ("cluster", "cluster"), ("cluster", ["cluster", "cluster2"])])
@pytest.mark.parametrize("weight_type", [None, "aweight", "fweight", "pweight"])
def test_all_2sls_covariances_weights_diagnostics(covariance, cluster, small, weight_type):
    if weight_type == "pweight" and covariance == "nonrobust":
        pytest.skip("Both paths reject conventional pweight covariance.")
    data = iv_data()
    spec = iv_spec(covariance=covariance, cluster=cluster, small=small, weight_type=weight_type)
    dense = fit_ivregress(spec, data)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, result, rtol=5e-8, atol=2e-9)
    assert result.inference["use_t"] == small
    assert set(result.tests) == set(dense.tests)
    for expected, actual in zip(dense.extra["first_stage"], result.extra["first_stage"], strict=True):
        for key, value in expected.items():
            assert actual[key] == (pytest.approx(value, rel=5e-8, abs=2e-9) if isinstance(value, (float, int)) else value)


def test_no_constant_iv_numpy_projection_oracle():
    data = iv_data()
    spec = ModelSpec(estimator="ivregress", outcome="y", predictors=["x"], intercept=False,
                     columns={"endogenous": ["endog"], "instruments": ["z1", "z2"]}, options={"method": "2sls"})
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    x, z = data[["x", "endog"]].to_numpy(), data[["x", "z1", "z2"]].to_numpy()
    projected = z@np.linalg.lstsq(z, x, rcond=None)[0]
    expected = np.linalg.solve(projected.T@projected, projected.T@data.y.to_numpy())
    np.testing.assert_allclose([c.estimate for c in result.coefficients], expected, atol=3e-12)


@pytest.mark.parametrize("method", ["liml", "gmm"])
def test_non_2sls_methods_preserve_native_full_inference(method):
    spec = iv_spec().model_copy(update={"options": {"method": method}})
    data = iv_data()
    assert_parity(fit_ivregress(spec, data),
                  fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17))


@pytest.mark.parametrize("method,kappa", [("fuller", None), ("kclass", 0.), ("kclass", .5)])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("exactly_identified", [False, True])
def test_extended_kclass_full_covariance_and_native_replay(method, kappa, covariance, exactly_identified):
    options = {"method": method, **({"kappa": kappa} if kappa is not None else {})}
    spec = iv_spec(covariance=covariance, cluster="cluster" if covariance == "cluster" else None)
    spec = spec.model_copy(update={"options": options, "predictors": ["x"], "categorical": [],
                                  "columns": {"endogenous": ["endog"],
                                              "instruments": ["z1"] if exactly_identified else ["z1", "z2"]}})
    data = iv_data()
    dense = fit_ivregress(spec, data)
    replay = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, replay, rtol=2e-8, atol=2e-10)
    assert dense.tests["overid_unavailable"]["p_value"] is None
    x = np.column_stack([np.ones(len(data)), data[["x", "endog"]]])
    z = np.column_stack([np.ones(len(data)), data[["x", *spec.columns["instruments"]]]])
    projected = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    effective_kappa = dense.metrics["kappa"]
    score_x = (1-effective_kappa)*x + effective_kappa*projected
    bread = np.linalg.inv(x.T @ score_x)
    beta = bread @ score_x.T @ data.y.to_numpy()
    residual = data.y.to_numpy() - x @ beta
    if covariance == "nonrobust":
        expected = (residual @ residual / len(data)) * bread
    else:
        scores = score_x * residual[:, None]
        if covariance == "cluster":
            sums = np.stack([scores[data.cluster == g].sum(0) for g in range(31)])
            meat = sums.T @ sums * 31/30
        else:
            meat = scores.T @ scores
        expected = bread @ meat @ bread.T
    np.testing.assert_allclose([row.estimate for row in dense.coefficients], beta, atol=1e-11)
    np.testing.assert_allclose(dense.covariance_matrix, expected, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("method,kappa", [("fuller", None), ("kclass", 0.), ("kclass", .5)])
def test_extended_iv_public_dataset_route_uses_replay_without_collection(monkeypatch, method, kappa):
    import openecon as oe

    data = iv_data()
    arguments = {"y": "y", "x": ["x"], "endog": ["endog"],
                 "instruments": ["z1", "z2"], "covariance": "robust", "method": method,
                 **({"kappa": kappa} if kappa is not None else {})}
    dense = oe.ivregress(data=data, **arguments)
    source = Dataset.from_batches(
        lambda: (data.iloc[start:start+17].copy() for start in range(0, len(data), 17)),
        list(data.columns), row_count=len(data),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("The public extended-IV route collected its Dataset")

    monkeypatch.setattr("openecon.econometrics.core._coerce_frame", forbidden)
    replay = oe.ivregress(data=source, **arguments)
    assert_parity(dense, replay, rtol=2e-8, atol=2e-10)
