"""Global profiled ML/REML parity and independent covariance oracles."""
import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import make_spec
from openecon.econometrics.mixed.lmm import fit_mixed
from openecon.econometrics.streaming_mixed import fit_streaming_mixed
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def data(n=300, seed=6144):
    rng = np.random.default_rng(seed)
    g = np.arange(n)%30
    frame = pd.DataFrame({"g": g, "cluster": g//3, "x": rng.normal(size=n), "category": np.where(g%2, "A", "B"), "w": rng.integers(1, 4, n)})
    frame["y"] = 1+.7*frame.x+rng.normal(size=30)[g]*1.2+rng.normal(size=n)*.6
    return frame


def spec(method="ml", covariance="nonrobust", *, weights=False, categorical=False, intercept=True, structure="independent"):
    return make_spec("mixed", outcome="y", predictors=["x", *(["category"] if categorical else [])],
                     intercept=intercept, covariance=covariance, cluster="cluster" if covariance=="cluster" else None,
                     weights="w" if weights else None, weight_type="fweight" if weights else None,
                     categorical=["category"] if categorical else [], columns={"group": ["g"]},
                     options={"method": method, "covstructure": structure})


def parity(a, b, rtol=3e-6, atol=3e-7):
    assert [c.term for c in a.coefficients] == [c.term for c in b.coefficients]
    assert [c.equation for c in a.coefficients] == [c.equation for c in b.coefficients]
    np.testing.assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients], rtol=rtol, atol=atol)
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=rtol, atol=atol)
    for key, value in a.metrics.items():
        if value is not None:
            assert b.metrics[key] == pytest.approx(value, rel=rtol, abs=atol)
    assert b.nobs == a.nobs
    assert not b.sample_positions
    assert b.provenance["streaming"]["maximum_batch_rows"]<=17
    ResultBundle.model_validate_json(b.model_dump_json())


@pytest.mark.parametrize("method,covariance", [("ml", "nonrobust"), ("ml", "robust"), ("ml", "cluster"), ("reml", "nonrobust")])
@pytest.mark.parametrize("weights,categorical,intercept", [(False, False, True), (True, False, True), (False, True, True), (False, False, False)])
def test_all_single_intercept_likelihood_and_covariance_options(method, covariance, weights, categorical, intercept):
    frame, s = data(), spec(method, covariance, weights=weights, categorical=categorical, intercept=intercept)
    parity(fit_mixed(s, frame), fit_streaming_mixed(s, Dataset.from_frame(frame), batch_rows=17))


@pytest.mark.parametrize("structure", ["independent", "unstructured", "exchangeable", "identity"])
def test_one_intercept_covariance_structures_reduce_to_identical_variance(structure):
    frame, s = data(), spec(structure=structure)
    parity(fit_mixed(s, frame), fit_streaming_mixed(s, Dataset.from_frame(frame), batch_rows=17))


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_independent_numpy_full_covariance_likelihood_and_gls(method):
    frame, s = data(), spec(method)
    result = fit_streaming_mixed(s, Dataset.from_frame(frame), batch_rows=17)
    g, s2 = [c.estimate for c in result.coefficients[-2:]]
    x, y = np.column_stack((np.ones(len(frame)), frame.x)), frame.y.to_numpy()
    v = s2*np.eye(len(frame))+g*(frame.g.to_numpy()[:, None]==frame.g.to_numpy()[None, :])
    precision_x, precision_y = np.linalg.solve(v, x), np.linalg.solve(v, y)
    gram = x.T@precision_x
    beta = np.linalg.solve(gram, x.T@precision_y)
    residual = y-x@beta
    pr = x.shape[1] if method=="reml" else 0
    ll = -.5*((len(frame)-pr)*np.log(2*np.pi)+np.linalg.slogdet(v)[1]+residual@np.linalg.solve(v, residual))
    if pr:
        ll -= .5*np.linalg.slogdet(gram)[1]
    np.testing.assert_allclose([c.estimate for c in result.coefficients[:2]], beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(result.covariance_matrix)[:2, :2], np.linalg.inv(gram), rtol=1e-12, atol=1e-12)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-12)


@pytest.mark.parametrize("option", ["three_levels", "group_is_outcome", "no_random_effect"])
def test_native_invalid_effects_fail_before_source_read(option):
    s = spec()
    if option=="three_levels":
        s = s.model_copy(update={"columns": {"group": ["cluster", "g", "category"]}})
    elif option=="group_is_outcome":
        s = s.model_copy(update={"columns": {"group": ["y"], "random": ["x"]}})
    else:
        s = s.model_copy(update={"options": {"random_intercept": False}})
    def forbidden():
        raise AssertionError("invalid mixed specification must refuse before reading")
        yield
    source = Dataset.from_batches(forbidden, data().columns.tolist(), row_count=300)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(s, source, batch_rows=17)
    assert error.value.code == "invalid_spec"


def test_early_workspace_refusal():
    with use_workspace_budget(8), pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(spec(), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "workspace_limit"
