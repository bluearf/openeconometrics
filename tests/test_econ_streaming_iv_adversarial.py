"""Independent projected-score oracle and adversarial native IV replay guards."""
import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.iv.ivregress import fit_ivregress
from openecon.econometrics.streaming_linear import fit_streaming_linear
from openecon.models import ModelSpec

from test_econ_streaming_linear import assert_parity


def multivariate_data():
    rng = np.random.default_rng(934651)
    n = 347
    x, z1, z2, z3, v1, v2, error = rng.normal(size=(7, n))
    data = pd.DataFrame({"x": x, "z1": z1, "z2": z2, "z3": z3,
                         "d1": .4*x+.9*z1-.3*z3+v1,
                         "d2": -.2*x+.7*z2+.4*z3+.2*v1+v2,
                         "weight": np.arange(n)%4+1,
                         "cluster": np.arange(n)%23, "cluster2": np.arange(n)%17})
    data["y"] = .8+.3*x+1.1*data.d1-.7*data.d2+error+.4*v1-.2*v2
    return data


def multivariate_spec(covariance="nonrobust", cluster=None, *, predictors=None,
                      intercept=True, weights=False, instruments=None):
    return ModelSpec(estimator="ivregress", outcome="y", predictors=["x"] if predictors is None else predictors,
                     intercept=intercept, covariance=covariance, cluster=cluster,
                     weights="weight" if weights else None, weight_type="aweight" if weights else None,
                     columns={"endogenous": ["d1", "d2"],
                              "instruments": instruments or ["z1", "z2", "z3"]},
                     options={"method": "2sls", "small": True})


@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("robust", None),
                           ("cluster", "cluster"), ("cluster", ["cluster", "cluster2"])])
@pytest.mark.parametrize("predictors,intercept", [(["x"], True), ([], False)])
def test_two_endogenous_full_diagnostics_dense_parity(covariance, cluster, predictors, intercept):
    data = multivariate_data()
    spec = multivariate_spec(covariance, cluster, predictors=predictors, intercept=intercept, weights=True)
    dense = fit_ivregress(spec, data)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, result, rtol=2e-7, atol=3e-9)
    assert result.extra["endogenous"] == ["d1", "d2"]
    assert set(result.tests) == set(dense.tests)
    for expected, actual in zip(dense.extra["first_stage"], result.extra["first_stage"], strict=True):
        for key, value in expected.items():
            assert actual[key] == (pytest.approx(value, rel=2e-7, abs=3e-9)
                                   if isinstance(value, (float, int)) else value)


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_observed_endogenous_residual_numpy_sandwich_oracle(covariance):
    data = multivariate_data()
    spec = multivariate_spec(covariance, "cluster" if covariance == "cluster" else None, weights=True)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    x = np.column_stack([np.ones(len(data)), data[["x", "d1", "d2"]].to_numpy()])
    z = np.column_stack([np.ones(len(data)), data[["x", "z1", "z2", "z3"]].to_numpy()])
    y = data.y.to_numpy()
    weight = data.weight.to_numpy(dtype=float)
    weight *= len(data)/weight.sum()
    zw = z*weight[:, None]**.5
    xhat = z@np.linalg.lstsq(zw, x*weight[:, None]**.5, rcond=None)[0]
    bread = np.linalg.inv(xhat.T@(weight[:, None]*xhat))
    beta = bread@(xhat.T@(weight*y))
    residual = y-x@beta
    n, k = x.shape
    if covariance == "nonrobust":
        covariance_expected = bread*(weight@residual**2)/(n-k)
    else:
        scores = xhat*(weight*residual)[:, None]
        if covariance == "robust":
            meat = scores.T@scores*n/(n-k)
        else:
            groups = np.unique(data.cluster)
            sums = np.stack([scores[data.cluster.to_numpy() == group].sum(0) for group in groups])
            meat = sums.T@sums*len(groups)/(len(groups)-1)*(n-1)/(n-k)
        covariance_expected = bread@meat@bread
    np.testing.assert_allclose([c.estimate for c in result.coefficients], beta, rtol=2e-11, atol=3e-12)
    np.testing.assert_allclose(result.covariance_matrix, covariance_expected, rtol=2e-10, atol=3e-12)
    # Fitted first-stage endogenous values give a different residual and cannot
    # accidentally certify this sandwich contract.
    assert np.linalg.norm((y-xhat@beta)-residual) > 1


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_exactly_identified_and_exact_first_stage(covariance):
    data = multivariate_data()
    data["d1"] = .4*data.x+.9*data.z1-.3*data.z2
    data["d2"] = -.2*data.x+.7*data.z2+.2*data.z1
    spec = multivariate_spec(covariance, "cluster" if covariance == "cluster" else None,
                             instruments=["z1", "z2"])
    dense = fit_ivregress(spec, data)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, result, rtol=1e-7, atol=2e-9)
    if covariance != "nonrobust":
        assert result.tests["overid_score"]["statistic"] is None
        assert result.tests["endog_robust_score"]["statistic"] is None


def test_collinear_instrument_global_omission_and_batch_invariance():
    data = multivariate_data()
    data["duplicate"] = 2*data.z1
    data["constant"] = 3.
    spec = multivariate_spec("robust", instruments=["z1", "duplicate", "z2", "z3", "constant"])
    dense = fit_ivregress(spec, data)
    first = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=1)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, result, rtol=1e-7, atol=2e-9)
    assert set(result.extra["omitted_instruments"]) == {"duplicate", "constant"}
    np.testing.assert_allclose(first.covariance_matrix, result.covariance_matrix, atol=4e-11)
    assert first.provenance["sample_positions_hash"] == result.provenance["sample_positions_hash"]


@pytest.mark.parametrize("role,columns", [
    ("exogenous", {"endogenous": ["x"], "instruments": ["z1"]}),
    ("self", {"endogenous": ["d1"], "instruments": ["d1"]}),
    ("included", {"endogenous": ["d1"], "instruments": ["x"]}),
    ("outcome", {"endogenous": ["d1"], "instruments": ["y"]}),
])
def test_invalid_roles_fail_before_source_iteration(monkeypatch, role, columns):
    spec = multivariate_spec().model_copy(update={"columns": columns})
    source = Dataset.from_frame(multivariate_data())

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid role must be checked before numerical source iteration.")

    monkeypatch.setattr(source, "iter_batches", forbidden)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(spec, source, batch_rows=17)
    assert error.value.code == "invalid_spec"


def test_no_usable_excluded_instruments_structured_underidentification():
    data = multivariate_data()
    data["z1"] = data.x
    data["z2"] = 2*data.x
    spec = multivariate_spec(instruments=["z1", "z2"])
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == "underidentified"
