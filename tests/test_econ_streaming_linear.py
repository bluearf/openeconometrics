"""Independent dense/native parity and bounded replay/spill contracts."""
import json

import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.linear.areg import fit_areg
from openecon.econometrics.linear.cnsreg import fit_cnsreg
from openecon.econometrics.streaming_linear import _GroupMeans, fit_streaming_linear
from openecon.models import ModelSpec, ResultBundle


def frame(seed=321, n=231):
    rng = np.random.default_rng(seed)
    result = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                           "group": np.arange(n)%31, "cluster": np.arange(n)%13,
                           "cat": pd.Categorical(np.arange(n)%3, categories=[0, 1, 2, 3]),
                           "weight": rng.uniform(.2, 3, n)})
    result["y"] = .8+1.1*result.x-.4*result.z+.07*result.group+rng.normal(size=n)
    return result


def spec_for(estimator, *, covariance="nonrobust", weight_type=None, categorical=False,
             intercept=True, cluster=None, missing="raise", constraints=None):
    predictors = ["x", "z", *(["cat"] if categorical else [])]
    options = ({"constraints": constraints or [{"terms": {"x": 1, "z": 1}, "value": .7}]}
               if estimator == "cnsreg" else {})
    return ModelSpec(estimator=estimator, outcome="y", predictors=predictors,
                     categorical=["cat"] if categorical else [], intercept=intercept,
                     covariance=covariance, cluster=cluster, missing=missing,
                     weights="weight" if weight_type else None, weight_type=weight_type,
                     columns={"absorb": "group"} if estimator == "areg" else {}, options=options)


def assert_parity(dense, replay, *, rtol=2e-9, atol=2e-10):
    assert [c.term for c in dense.coefficients] == [c.term for c in replay.coefficients]
    np.testing.assert_allclose([c.estimate for c in dense.coefficients],
                               [c.estimate for c in replay.coefficients], rtol=rtol, atol=atol)
    np.testing.assert_allclose(dense.covariance_matrix, replay.covariance_matrix, rtol=rtol, atol=atol)
    for key, value in dense.metrics.items():
        if value is None:
            assert replay.metrics[key] is None
        else:
            assert replay.metrics[key] == pytest.approx(value, rel=rtol, abs=atol)
    for key, test in dense.tests.items():
        for field in ["statistic", "df", "df2", "p_value"]:
            if test.get(field) is not None:
                assert replay.tests[key][field] == pytest.approx(test[field], rel=rtol, abs=atol)
    assert replay.nobs == dense.nobs
    assert replay.nobs_original == dense.nobs_original
    assert replay.dropped_rows == dense.dropped_rows
    assert replay.sample_positions == []
    assert replay.provenance["sample_positions_omitted"]
    assert replay.provenance["streaming"]["dense_observation_matrix"] is False
    assert replay.provenance["streaming"]["maximum_batch_rows"] <= 17
    restored = ResultBundle.model_validate_json(replay.model_dump_json())
    assert restored.covariance_matrix == replay.covariance_matrix
    assert "\\begin{tabular}" in restored.to_latex()


@pytest.mark.parametrize("estimator", ["areg", "cnsreg"])
@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("HC1", None),
                          ("cluster", "cluster"), ("cluster", ["cluster", "group"])])
@pytest.mark.parametrize("weight_type", [None, "aweight", "fweight", "pweight"])
def test_all_linear_covariance_weight_dense_parity(estimator, covariance, cluster, weight_type):
    if weight_type == "pweight" and covariance == "nonrobust":
        pytest.skip("Both dense and replay reject conventional pweight covariance.")
    data = frame()
    if weight_type == "fweight":
        data["weight"] = np.arange(len(data))%4+1
    spec = spec_for(estimator, covariance=covariance, cluster=cluster, weight_type=weight_type, categorical=True)
    dense = (fit_areg if estimator == "areg" else fit_cnsreg)(spec, data)
    replay = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, replay)
    if estimator == "areg":
        assert replay.extra == dense.extra


@pytest.mark.parametrize("intercept", [True, False])
def test_constrained_numpy_kkt_oracle(intercept):
    data = frame(n=160)
    constraints = [{"terms": {"x": 2, "z": -1}, "value": .3}]
    spec = spec_for("cnsreg", intercept=intercept, constraints=constraints)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    x = np.column_stack(([np.ones(len(data))] if intercept else [])+[data.x, data.z])
    r = np.array([[0, 2, -1]] if intercept else [[2, -1]], float)
    inverse = np.linalg.inv(x.T@x)
    ols = np.linalg.lstsq(x, data.y, rcond=None)[0]
    expected = ols-inverse@r.T@np.linalg.solve(r@inverse@r.T, r@ols-.3)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], expected, atol=2e-12)


def test_fixed_and_redundant_constraints_exact_records():
    data = frame()
    constraints = [{"terms": {"x": 1}, "value": 1.}, {"terms": {"x": 2}, "value": 2.}]
    spec = spec_for("cnsreg", constraints=constraints)
    dense = fit_cnsreg(spec, data)
    replay = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, replay)
    assert replay.extra["constrained_terms"]["x"] == pytest.approx(1., abs=1e-15)
    assert replay.metrics["df_constraints"] == 1
    assert any("redundant" in message for message in replay.warnings)


@pytest.mark.parametrize("estimator", ["areg", "cnsreg"])
def test_missing_zero_weight_physical_positions_and_batch_invariance(estimator):
    data = frame(n=703)
    data.loc[3, "z"] = np.nan
    data.loc[14, "weight"] = 0
    spec = spec_for(estimator, weight_type="aweight", missing="drop")
    first = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=1)
    other = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in first.coefficients], [c.estimate for c in other.coefficients], atol=2e-12)
    np.testing.assert_allclose(first.covariance_matrix, other.covariance_matrix, atol=2e-12)
    assert first.provenance["sample_positions_hash"] == other.provenance["sample_positions_hash"]
    assert first.dropped_rows == 2
    assert len(first.predictions) == 400
    assert [row["row"] for row in first.predictions][:15] == [i for i in range(17) if i not in {3, 14}]
    json.dumps(other.model_dump())


def test_absorbed_between_only_and_collinear_regressors():
    data = frame()
    data["z"] = data.group.astype(float)
    spec = spec_for("areg")
    dense = fit_areg(spec, data)
    replay = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, replay)
    assert "z" in replay.provenance["omitted_terms"]
    assert any("z" in message and "absorbed" in message for message in replay.warnings)


def test_group_spill_is_bounded_and_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(_GroupMeans, "cache_bytes", 1400)
    data = frame(n=701)
    data["group"] = np.arange(len(data))%173
    data["y"] += .1*data.group
    result = fit_streaming_linear(spec_for("areg", covariance="cluster", cluster="group"),
                                  Dataset.from_frame(data), batch_rows=17)
    diagnostics = result.provenance["solver_diagnostics"]
    assert diagnostics["fixed_effect_peak_cache_accounted_bytes"] <= 1400
    assert diagnostics["fixed_effect_spill_writes"] > 173
    assert diagnostics["fixed_effect_scratch_bytes"] > 0
    assert not list(tmp_path.iterdir())


def test_spill_cleanup_on_changed_replay_source(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    data = frame()
    calls = 0

    def batches():
        nonlocal calls
        calls += 1
        copy = data.copy()
        if calls >= 7:
            copy.loc[25, "y"] += 1
        yield copy

    source = Dataset.from_batches(batches, data.columns.tolist(), row_count=len(data))
    with pytest.raises(AnalysisError, match="changed"):
        fit_streaming_linear(spec_for("areg"), source, batch_rows=17)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("which,code", [("groups", "insufficient_groups"), ("exact", "perfect_fit"),
                                       ("constant", "constant_outcome")])
def test_absorbed_degenerate_samples(which, code):
    data = frame()
    if which == "groups":
        data["group"] = 0
    elif which == "exact":
        data["y"] = 2*data.x+.5*data.group
    else:
        data["y"] = 1
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(spec_for("areg"), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == code


def test_unsupported_family_is_not_collected(monkeypatch):
    data = frame()
    monkeypatch.setattr(Dataset, "collect", lambda *args, **kwargs: pytest.fail("Dataset collected"), raising=False)
    spec = ModelSpec(estimator="xtfmb", outcome="y", predictors=["x"], panel="group", time="cluster")
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(spec, Dataset.from_frame(data))
    assert error.value.code == "unsupported_streaming_estimator"


def test_affine_inconsistent_constraints():
    spec = spec_for("cnsreg", constraints=[{"terms": {"x": 1}, "value": 1},
                                         {"terms": {"x": 2}, "value": 3}])
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(spec, Dataset.from_frame(frame()), batch_rows=17)
    assert error.value.code == "inconsistent_constraints"


@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("HC1", None),
                                                ("cluster", "group"), ("cluster", ["group", "cluster"])])
def test_absorbed_singletons_kept_and_nested_dof(covariance, cluster):
    data = frame(n=211)
    data.loc[:29, "group"] = np.arange(1000, 1030)
    spec = spec_for("areg", covariance=covariance, cluster=cluster)
    dense = fit_areg(spec, data)
    replay = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, replay)
    assert replay.nobs == len(data)
    assert replay.metrics["n_groups"] == data.group.nunique()
    assert replay.metrics["df_resid"] == len(data)-data.group.nunique()-2


@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("HC1", None), ("cluster", "group")])
def test_absorbed_large_offsets_preserve_within_variation(covariance, cluster):
    data = frame()
    data["x"] += 2**30
    data["z"] -= 2**27
    data["y"] += 2**29
    spec = spec_for("areg", covariance=covariance, cluster=cluster)
    dense = fit_areg(spec, data)
    replay = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert [c.term for c in replay.coefficients] == ["Intercept", "x", "z"]
    np.testing.assert_allclose([c.estimate for c in replay.coefficients][1:],
                               [c.estimate for c in dense.coefficients][1:], rtol=2e-8, atol=2e-8)
    # Separate centered oracle, using represented input differences instead
    # of subtracting a large raw grand mean inside the normal equations.
    x = data[["x", "z"]].to_numpy()
    y = data.y.to_numpy()
    x, y = x-x[0], y-y[0]
    group = data.group.to_numpy()
    within_x, within_y = x.copy(), y.copy()
    for value in np.unique(group):
        selected = group == value
        within_x[selected] -= x[selected].mean(0)
        within_y[selected] -= y[selected].mean()
    beta = np.linalg.lstsq(within_x, within_y, rcond=None)[0]
    inverse = np.linalg.inv(within_x.T@within_x)
    resid = within_y-within_x@beta
    df = len(data)-len(np.unique(group))-2
    if covariance == "nonrobust":
        expected_covariance = inverse*(resid@resid/df)
    else:
        scores = within_x*resid[:, None]
        if covariance == "cluster":
            grouped = np.stack([scores[group == value].sum(0) for value in np.unique(group)])
            meat = grouped.T@grouped
            correction = len(grouped)/(len(grouped)-1)*(len(data)-1)/df
        else:
            meat = scores.T@scores
            correction = len(data)/df
        expected_covariance = inverse@meat@inverse*correction
    np.testing.assert_allclose([c.estimate for c in replay.coefficients][1:], beta, atol=2e-12)
    np.testing.assert_allclose(np.array(replay.covariance_matrix)[1:, 1:], expected_covariance, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(np.array(dense.covariance_matrix)[1:, 1:], expected_covariance, rtol=2e-7, atol=5e-10)
    assert replay.metrics["r_squared_within"] == pytest.approx(dense.metrics["r_squared_within"], abs=2e-8)


@pytest.mark.parametrize("intercept", [True, False])
def test_constraint_units_and_affine_nonzero_value(intercept):
    data = frame()
    data["x"] *= 1e-5
    data["z"] *= 1e4
    constraints = [{"terms": {"x": 1e-5, "z": 1e4}, "value": .7}]
    spec = spec_for("cnsreg", intercept=intercept, covariance="HC1", constraints=constraints)
    dense = fit_cnsreg(spec, data)
    replay = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    if intercept:
        assert_parity(dense, replay, rtol=2e-8, atol=2e-8)
    else:
        # Explicit one-dimensional affine substitution is independent of the
        # null-space QR's sensitivity to a nine-order coefficient-unit ratio.
        free_x = data.x.to_numpy()-1e-9*data.z.to_numpy()
        adjusted_y = data.y.to_numpy()-.7/1e4*data.z.to_numpy()
        beta = free_x@adjusted_y/(free_x@free_x)
        resid = adjusted_y-free_x*beta
        variance = ((free_x*resid)@(free_x*resid))/(free_x@free_x)**2*len(data)/(len(data)-1)
        assert replay.coefficients[0].estimate == pytest.approx(beta, rel=1e-13)
        assert replay.covariance_matrix[0][0] == pytest.approx(variance, rel=1e-13)
    estimates = {coefficient.term: coefficient.estimate for coefficient in replay.coefficients}
    assert 1e-5*estimates["x"]+1e4*estimates["z"] == pytest.approx(.7, abs=2e-12)


@pytest.mark.parametrize("estimator", ["areg", "cnsreg"])
def test_classical_pweights_guard_precedes_reader(estimator):
    data = frame()

    def reader():
        pytest.fail("Unsupported weight/covariance read data")
        yield data

    source = Dataset.from_batches(reader, data.columns.tolist())
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(spec_for(estimator, weight_type="pweight"), source)
    assert error.value.code == "unsupported_covariance"


def test_spill_storage_error_is_structured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path/"absent"))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(spec_for("areg"), Dataset.from_frame(frame()), batch_rows=17)
    assert error.value.code == "fixed_effect_spill_failed"
