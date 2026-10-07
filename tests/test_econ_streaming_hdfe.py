"""Global FE projection parity and independent dummy-space scientific oracle."""
import numpy as np
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.linear.reghdfe import fit_reghdfe
from openecon.econometrics.streaming_hdfe import _Vectors, fit_streaming_hdfe
from openecon.models import ModelSpec
from openecon.resources import use_workspace_budget


def data(seed=351937, dimensions=2, n=181):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                          "f0": np.arange(n)%13, "f1": rng.integers(0, 11, size=n),
                          "f2": rng.integers(0, 7, size=n), "weight": np.arange(n)%3+1,
                          "cluster": np.arange(n)%19, "cluster2": np.arange(n)%17,
                          "cat": pd.Categorical(np.arange(n)%3, categories=[0, 1, 2, 3])})
    frame["y"] = .4+1.1*frame.x-.3*frame.z+.04*frame.f0-.03*frame.f1+.07*frame.f2+rng.normal(size=n)
    for d in range(dimensions):
        frame.loc[0, f"f{d}"] = 1000+d
    return frame


def spec(dimensions=2, covariance="nonrobust", cluster=None, weight_type=None, *, categorical=False):
    return ModelSpec(estimator="reghdfe", outcome="y", predictors=["x", "z", *(["cat"] if categorical else [])],
                     intercept=False, columns={"absorb": [f"f{i}" for i in range(dimensions)]},
                     covariance=covariance, cluster=cluster, categorical=["cat"] if categorical else [],
                     weights="weight" if weight_type else None, weight_type=weight_type,
                     options={"tolerance": 1e-10})


def parity(dense, actual, tolerance=3e-8):
    assert [c.term for c in actual.coefficients] == [c.term for c in dense.coefficients]
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], [c.estimate for c in dense.coefficients], rtol=tolerance, atol=2e-10)
    np.testing.assert_allclose(actual.covariance_matrix, dense.covariance_matrix, rtol=tolerance, atol=2e-10)
    for name, value in dense.metrics.items():
        assert actual.metrics[name] == pytest.approx(value, rel=tolerance, abs=3e-10)
    assert actual.extra["absorbed"] == dense.extra["absorbed"]
    assert actual.nobs == dense.nobs
    assert actual.dropped_rows == dense.dropped_rows
    assert actual.sample_positions == []
    assert actual.provenance["streaming"]["dense_observation_matrix"] is False


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("HC1", None),
                                              ("cluster", ["cluster", "cluster2"])])
def test_multi_dimension_global_dense_parity(dimensions, covariance, cluster):
    frame = data(dimensions=dimensions)
    model = spec(dimensions, covariance, cluster, categorical=True)
    expected = fit_reghdfe(model, frame)
    actual = fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47)
    parity(expected, actual)


@pytest.mark.parametrize("weight_type", ["aweight", "fweight", "pweight"])
def test_weighted_global_projection_covariance_and_singletons(weight_type):
    frame = data()
    model = spec(2, "cluster", "cluster", weight_type)
    expected = fit_reghdfe(model, frame)
    actual = fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47)
    parity(expected, actual)


def test_independent_numpy_weighted_dummy_projection_and_hc1_oracle():
    frame = data(n=233)
    model = spec(2, "HC1", weight_type="aweight").model_copy(update={"options": {"tolerance": 1e-11, "drop_singletons": False}})
    actual = fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47)
    weight = frame.weight.to_numpy(dtype=float)
    weight *= len(frame)/weight.sum()
    x, y = frame[["x", "z"]].to_numpy(), frame.y.to_numpy()
    dummies = np.column_stack([pd.get_dummies(frame[name], dtype=float).to_numpy() for name in ["f0", "f1"]])
    square = weight**.5
    projected_x = x-dummies@np.linalg.lstsq(dummies*square[:, None], x*square[:, None], rcond=None)[0]
    projected_y = y-dummies@np.linalg.lstsq(dummies*square[:, None], y*square, rcond=None)[0]
    bread = np.linalg.inv(projected_x.T@(projected_x*weight[:, None]))
    beta = bread@(projected_x.T@(projected_y*weight))
    residual = projected_y-projected_x@beta
    df = len(frame)-np.linalg.matrix_rank(dummies)-2
    score = projected_x*(weight*residual)[:, None]
    covariance = bread@(score.T@score)@bread*len(frame)/df
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], beta, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(actual.covariance_matrix, covariance, rtol=5e-9, atol=3e-11)
    assert actual.metrics["df_resid"] == df


@pytest.mark.parametrize("drop", [False, True])
def test_singletons_categories_missing_and_batch_invariance(drop):
    frame = data(n=193)
    frame.loc[11, "x"] = np.nan
    model = spec(2, "HC1", categorical=True).model_copy(update={
        "missing": "drop", "options": {"drop_singletons": drop, "tolerance": 1e-11}})
    dense = fit_reghdfe(model, frame)
    one = fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=17)
    another = fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47)
    parity(dense, one)
    np.testing.assert_allclose(one.covariance_matrix, another.covariance_matrix, rtol=3e-10, atol=3e-12)
    assert one.provenance["sample_positions_hash"] == another.provenance["sample_positions_hash"]
    assert one.provenance["categorical_encoding"] == dense.provenance["categorical_encoding"]
    assert one.extra["singletons_dropped"] == int(drop)


@pytest.mark.parametrize("covariance,cluster", [("cluster", "f0"), ("cluster", ["f0", "f1"]),
                                               ("cluster", ["f1", "f0"])])
def test_nested_and_disconnected_dof_dense_parity(covariance, cluster):
    frame = data(n=223)
    model = spec(2, covariance, cluster)
    parity(fit_reghdfe(model, frame), fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47))


@pytest.mark.parametrize("units", ["offset", "small", "large"])
def test_reparameterized_units_centered_numpy_dummy_oracle(units):
    frame = data(n=193)
    if units == "offset":
        frame["x"] += 2**30
        frame["z"] -= 2**27
        frame["y"] += 2**29
    elif units == "small":
        frame["x"] *= 1e-70
        frame["z"] *= 1e-50
    else:
        frame["x"] *= 1e70
        frame["z"] *= 1e50
    model = spec(2, "HC1").model_copy(update={"options": {"drop_singletons": False, "tolerance": 1e-11}})
    actual = fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47)
    x, y = frame[["x", "z"]].to_numpy(), frame.y.to_numpy()
    x, y = x-x[0], y-y[0]
    scales = np.max(np.abs(x), axis=0)
    x /= scales
    dummy = np.column_stack([pd.get_dummies(frame[name], dtype=float).to_numpy() for name in ["f0", "f1"]])
    x -= dummy@np.linalg.lstsq(dummy, x, rcond=None)[0]
    y -= dummy@np.linalg.lstsq(dummy, y, rcond=None)[0]
    inverse = np.linalg.inv(x.T@x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y-x@beta
    df = len(frame)-np.linalg.matrix_rank(dummy)-2
    scores = x*residual[:, None]
    covariance = inverse@(scores.T@scores)@inverse*len(frame)/df
    np.testing.assert_allclose(np.array([c.estimate for c in actual.coefficients])*scales, beta, rtol=5e-10, atol=3e-11)
    np.testing.assert_allclose(np.array(actual.covariance_matrix)*scales[:, None]*scales[None, :], covariance, rtol=3e-9, atol=3e-11)


def test_poorly_connected_global_projection_independent_dummy_oracle():
    rng = np.random.default_rng(6387)
    edges = [(i, i) for i in range(31)]+[(i+1, i) for i in range(31)]
    frame = pd.DataFrame({"f0": [a for a, _ in edges]*3, "f1": [b for _, b in edges]*3})
    frame["x"], frame["z"] = rng.normal(size=(2, len(frame)))
    frame["y"] = .4+1.1*frame.x-.3*frame.z+.04*frame.f0-.03*frame.f1+rng.normal(size=len(frame))
    model = spec(2).model_copy(update={"options": {"tolerance": 1e-11}})
    actual = fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47)
    dummy = np.column_stack([pd.get_dummies(frame[name], dtype=float).to_numpy() for name in ["f0", "f1"]])
    x, y = frame[["x", "z"]].to_numpy(), frame.y.to_numpy()
    x -= dummy@np.linalg.lstsq(dummy, x, rcond=None)[0]
    y -= dummy@np.linalg.lstsq(dummy, y, rcond=None)[0]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], beta, rtol=3e-10, atol=3e-11)
    assert actual.extra["iterations"] < 100


@pytest.mark.parametrize("defect,code", [("nonconvergence", "absorption_nonconvergence"),
                                       ("all_singletons", "empty_sample"),
                                       ("all_absorbed", "empty_design"), ("exact", "perfect_fit")])
def test_degeneracy_nonconvergence_and_owned_storage_cleanup(defect, code, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data()
    model = spec(2)
    if defect == "nonconvergence":
        model = model.model_copy(update={"options": {"max_iterations": 1}})
    elif defect == "all_singletons":
        frame["f0"] = np.arange(len(frame))
    elif defect == "all_absorbed":
        frame["x"], frame["z"] = frame.f0, frame.f1
    else:
        frame["y"] = 1.1*frame.x-.3*frame.z+.1*frame.f0-.05*frame.f1
    with pytest.raises(AnalysisError) as error:
        fit_streaming_hdfe(model, Dataset.from_frame(frame), batch_rows=47)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_disk_and_shared_workspace_refused_before_state_write(tmp_path, monkeypatch):
    from collections import namedtuple
    import openecon.econometrics.streaming_hdfe as module
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))

    def forbidden(*args, **kwargs):
        pytest.fail("Insufficient resource budget must precede writing vector state.")

    monkeypatch.setattr(_Vectors, "writer", forbidden)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: usage(1_000_000, 0, 1_000_000))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_hdfe(spec(), Dataset.from_frame(data()), batch_rows=47)
    assert error.value.code == "fixed_effect_disk_limit"
    assert error.value.disk_plan["estimated_scratch_bytes"] > 0
    assert not list(tmp_path.iterdir())
    with use_workspace_budget(16):
        with pytest.raises(AnalysisError) as error:
            fit_streaming_hdfe(spec(2, "cluster", "cluster"), Dataset.from_frame(data()), batch_rows=47)
    assert error.value.code == "workspace_limit"
    assert not list(tmp_path.iterdir())


def test_changed_source_during_owned_projection_cleans_all_state(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data()
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        altered = frame.copy()
        if calls >= 10:
            altered.loc[5, "y"] += 1
        yield altered

    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError, match="changed"):
        fit_streaming_hdfe(spec(), source, batch_rows=47)
    assert calls >= 10
    assert not list(tmp_path.iterdir())
