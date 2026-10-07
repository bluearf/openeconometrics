"""Whole-sample gradient and indexed-output contracts on small raw batches."""
import gc
import math
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ResultBundle


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(64821)
    n = 360
    x, z = rng.normal(size=(2, n))
    g = pd.Categorical(np.resize(["B", "A", "C"], n), categories=["B", "A", "C"])
    eta = .2 + .7 * x - .2 * z + .3 * (g == "A") - .1 * (g == "C")
    frame = pd.DataFrame({"x": x, "z": z, "g": g, "w": rng.uniform(.4, 3, n),
                          "y": (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int),
                          "count": rng.poisson(np.exp(eta)), "normal": eta + rng.normal(size=n)})
    models = {}
    for name, y, options in [("logit", "y", {}), ("probit", "y", {}),
                             ("poisson", "count", {}), ("ologit", "count", {}),
                             ("hetprobit", "y", {"het": ["x", "z"]})]:
        weighting = {} if name in {"logit", "probit"} else {"weights": "w", "weight_type": "aweight"}
        model = getattr(oe, name)(data=frame, y=y, x=["x", "z", "g"], categorical=["g"],
                                   missing="drop", **weighting, **options)
        models[name] = ResultBundle.model_validate_json(model.model_dump_json())
    models["ols"] = ResultBundle.model_validate_json(oe.ols(
        data=frame, y="normal", x=["x", "z", "g"], categorical=["g"],
        weights="w", weight_type="aweight", missing="drop").model_dump_json())
    return frame, models


def source(frame, rows=13):
    return Dataset.from_batches(lambda: (frame.iloc[i:i + rows] for i in range(0, len(frame), rows)),
                                list(frame.columns), row_count=len(frame))


def collected(dataset):
    return pd.concat(list(dataset.iter_batches(batch_rows=11)))


@pytest.mark.parametrize("name", ["ols", "logit", "probit", "poisson", "ologit", "hetprobit"])
@pytest.mark.parametrize("method", ["ame", "mem"])
def test_global_effects_gradients_and_covariance_match_dense(fitted, name, method):
    frame, models = fitted
    evaluation = frame.iloc[:90].copy()
    evaluation.loc[:12, "x"] = np.nan  # entire first raw block is excluded
    evaluation.loc[13:25, "w"] = 0      # entire second block has zero mass
    evaluation.loc[13:25, "g"] = "A"
    # Scaling weights must never overflow the normalization accumulator.
    evaluation["w"] *= 1e280
    options = dict(variables=["x", "g"], method=method, at={"x": [-1., .5]})
    expected = oe.margins(models[name], data=evaluation, **options)
    actual = oe.margins(models[name], data=source(evaluation), batch_rows=13, **options)
    assert actual.columns.tolist() == expected.columns.tolist()
    assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"), rtol=5e-10, atol=3e-13)
    assert_allclose(actual.attrs["delta_gradients"], expected.attrs["delta_gradients"], rtol=5e-10, atol=3e-13)
    for key in ["evaluation_rows", "input_evaluation_rows", "complete_evaluation_rows", "zero_weight_rows_excluded"]:
        assert actual.attrs[key] == expected.attrs[key]
    assert actual.attrs["streaming"]["maximum_batch_rows"] <= 13
    assert actual.attrs["streaming"]["full_source_collected"] is False
    assert "\\toprule" in actual.to_latex()


@pytest.mark.parametrize("name", ["ols", "logit", "probit", "poisson", "ologit", "hetprobit"])
@pytest.mark.parametrize("kind", ["response", "xb", "stdp", "derivative"])
def test_predictions_preserve_duplicate_index_and_missing_positions(fitted, name, kind, tmp_path, monkeypatch):
    frame, models = fitted
    evaluation = frame.iloc[:61].copy()
    evaluation.index = pd.Index(np.resize(["same", "other", "same"], len(evaluation)), name="observation")
    evaluation.iloc[12:16, evaluation.columns.get_loc("x")] = np.nan
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    options = dict(kind=kind, term="x" if kind == "derivative" else None,
                   interval=None if kind == "stdp" else "mean")
    expected = oe.predict(models[name], evaluation, **options)
    output = oe.predict(models[name], source(evaluation), batch_rows=7, **options)
    assert isinstance(output, Dataset)
    actual = collected(output)
    assert actual.index.equals(expected.index)
    assert actual.columns.tolist() == expected.columns.tolist()
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-13, equal_nan=True)
    assert output.metadata["analysis"]["missing_prediction_rows"] == 4
    path = Path(output._owned_prediction_output.name)
    assert path.exists()
    del output
    gc.collect()
    assert not path.exists()


def test_logistic_ame_global_delta_gradient_independent_formula(fitted):
    frame, models = fitted
    model = models["logit"]
    frame = frame.iloc[:73]
    beta = np.array([c.estimate for c in model.coefficients])
    terms = [c.term for c in model.coefficients]
    columns = {"Intercept": np.ones(len(frame)), "x": frame.x, "z": frame.z,
               "g[A]": (frame.g == "A").astype(float), "g[C]": (frame.g == "C").astype(float)}
    x = np.column_stack([columns[term] for term in terms])
    p = 1 / (1 + np.exp(-(x @ beta)))
    w = np.ones(len(frame)) / len(frame)
    ix = terms.index("x")
    gradient = (w * beta[ix] * p * (1-p) * (1-2*p)) @ x
    gradient[ix] += np.dot(w, p * (1-p))
    expected = np.dot(w, beta[ix] * p * (1-p))
    se = math.sqrt(gradient @ np.array(model.covariance_matrix) @ gradient)
    actual = oe.margins(model, "x", data=source(frame), batch_rows=3)
    assert actual.estimate.iloc[0] == pytest.approx(expected, rel=1e-12)
    assert actual.std_error.iloc[0] == pytest.approx(se, rel=1e-12)
    assert_allclose(actual.attrs["delta_gradients"][0], gradient, rtol=1e-12, atol=1e-14)


def test_mem_verifies_source_values_between_intervention_passes(fitted):
    frame, models = fitted
    passes = 0
    def changing():
        nonlocal passes
        passes += 1
        data = frame.iloc[:35].copy()
        if passes > 1:
            data.loc[data.index[0], "z"] += .25
        yield data
    data = Dataset.from_batches(changing, list(frame.columns), row_count=35)
    with pytest.raises(AnalysisError, match="changed between") as error:
        oe.margins(models["logit"], "x", data=data, method="mem", at={"x": [0., 1.]}, batch_rows=7)
    assert error.value.code == "source_changed"


def test_output_failure_removes_owned_partial_file(fitted, tmp_path, monkeypatch):
    frame, models = fitted
    evaluation = frame.iloc[:30].copy()
    evaluation["g"] = evaluation.g.astype(object)
    evaluation.iloc[16, evaluation.columns.get_loc("g")] = "unfitted"
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError, match="absent from the fitted"):
        oe.predict(models["logit"], source(evaluation), batch_rows=7)
    assert list(tmp_path.iterdir()) == []


def test_margins_inference_context_and_resource_guard(fitted):
    frame, models = fitted
    expected = oe.margins(models["logit"], "x", data=source(frame), method="mem", batch_rows=13)
    with torch.inference_mode():
        actual = oe.margins(models["logit"], "x", data=source(frame), method="mem", batch_rows=13)
    assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"), rtol=1e-12)
    from openecon.resources import use_workspace_budget
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        oe.margins(models["logit"], "x", data=source(frame), batch_rows=65536)
    assert error.value.code == "workspace_limit"


def test_native_fitted_ols_predict_and_new_evaluation_margins(fitted):
    frame, _ = fitted
    model = oe.ols(data=frame, y="normal", x=["x", "z"])
    expected = oe.predict(model, frame.iloc[:29], interval="mean")
    output = oe.predict(model, source(frame.iloc[:29]), interval="mean", batch_rows=7)
    assert_allclose(collected(output), expected, rtol=1e-12)
    actual = oe.margins(model, "x", data=source(frame.iloc[:29]), batch_rows=7)
    assert actual.estimate.iloc[0] == pytest.approx(next(c.estimate for c in model.coefficients if c.term == "x"))


@pytest.mark.parametrize("index", [
    pd.Index([None, None, None, "same", 1, "1", pd.NA, "same", float("nan")], dtype=object, name="row"),
    pd.Index(pd.array([None, None, None, 4, 4, 5, 6, 7, 8], dtype="Int64"), name="row"),
    pd.Index(pd.array([None, None, None, "same", "other", "same", "last", "a", "b"], dtype="string"), name="row"),
    pd.MultiIndex.from_arrays([[None, None, None, "A", "A", "B", "B", "C", "C"],
                               [3, 3, 4, 5, 5, 6, 6, 7, 8]], names=["group", "row"]),
])
def test_owned_output_index_schema_survives_null_first_blocks(fitted, index):
    from openecon.econometrics.postest.index_codec import encode
    frame, models = fitted
    evaluation = frame.iloc[:len(index)].copy()
    evaluation.index = index
    expected = oe.predict(models["logit"], evaluation, interval="mean")
    output = oe.predict(models["logit"], source(evaluation, rows=3), interval="mean", batch_rows=3)
    actual = collected(output)
    assert actual.index.names == expected.index.names
    assert [encode(item) for item in actual.index] == [encode(item) for item in expected.index]
    assert_allclose(actual, expected, rtol=1e-12)


def test_source_cannot_mutate_saved_weight_roles_during_evaluation(fitted):
    frame, models = fitted
    result = models["ols"].model_copy(deep=True)
    expected = oe.margins(result, "g", data=frame.iloc[:37])
    def factory():
        result.spec.weights = None
        result.spec.weight_type = None
        yield frame.iloc[:37]
    data = Dataset.from_batches(factory, list(frame.columns), row_count=37)
    actual = oe.margins(result, "g", data=data, batch_rows=7)
    assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"), rtol=1e-12)


def test_parameter_workspace_guard_precedes_covariance_validation(fitted, monkeypatch):
    from openecon.econometrics.postest import prediction
    from openecon.resources import use_workspace_budget
    frame, models = fitted
    model = models["logit"].model_copy(update={"coefficients": models["logit"].coefficients * 200})
    monkeypatch.setattr(prediction, "_parameters", lambda *args: pytest.fail("allocated covariance before guard"))
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        oe.predict(model, source(frame.iloc[:7]), batch_rows=7)
    assert error.value.code == "workspace_limit"


def test_ame_applies_covariance_after_global_weighted_gradient(fitted):
    frame, models = fitted
    result = ResultBundle.model_validate_json(oe.poisson(data=frame, y="count", x=["x"],
        weights="w", weight_type="aweight").model_dump_json())
    for coefficient in result.coefficients:
        coefficient.estimate = 0. if coefficient.term == "Intercept" else 1.
        coefficient.std_error = 1e-10
    result.covariance_matrix = [[1e-20, 0.], [0., 1e-20]]
    evaluation = pd.DataFrame({"x": [0., 700.], "w": [1., 1e-300]})
    expected = oe.margins(result, "x", data=evaluation)
    actual = oe.margins(result, "x", data=source(evaluation, rows=1), batch_rows=1)
    assert math.isfinite(actual.std_error.iloc[0])
    assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"), rtol=1e-12)


def test_parameter_snapshot_does_not_copy_row_sized_fitted_state(fitted):
    class RowSizedState(list):
        def __deepcopy__(self, memo):
            pytest.fail("Evaluation copied an unrelated fitted sample or diagnostic")
    frame, models = fitted
    model = models["ols"].model_copy(deep=True)
    expected = oe.margins(model, "x", data=frame.iloc[:17])
    model.sample_positions = RowSizedState(range(100))
    model.predictions = RowSizedState()
    model.tests = {"unused": RowSizedState(range(100))}
    model.extra["random_effects"] = RowSizedState(range(100))
    model.provenance["sample_diagnostics"] = RowSizedState(range(100))
    actual = oe.margins(model, "x", data=source(frame.iloc[:17]), batch_rows=7)
    assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"), rtol=1e-12)
