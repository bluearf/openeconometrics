"""Public OLS regressions for device, bounded prediction and inference wiring."""
import json
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.dataset import Dataset
from openecon.linear_ols.estimation import _ExactWeightSum


@pytest.fixture
def frame():
    rng = np.random.default_rng(9841)
    data = pd.DataFrame({"x": rng.normal(size=720), "z": rng.normal(size=720),
                         "t": np.arange(720), "firm": np.repeat(np.arange(24), 30)})
    data["y"] = 2 + .8 * data.x - .2 * data.z + rng.normal(size=720) * (1 + .3 * data.x**2)
    data["w"] = rng.uniform(.2, 3, size=720)
    return data


def test_dataset_auto_records_unavailable_accelerator_fallback(frame, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = oe.ols(data=Dataset.from_frame(frame), y="y", x=["x", "z"], device="auto")
    assert result.provenance["device"] == "cpu"
    assert result.provenance["solver"] == "torch_tsqr"
    assert result.provenance["fallbacks"]


def test_dataset_explicit_cuda_falls_back_if_driver_cannot_execute(frame, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    result = oe.ols(data=Dataset.from_frame(frame), y="y", x=["x"], device="cuda")
    assert result.provenance["requested_device"] == "cuda"
    assert result.provenance["device"] == "cpu"
    assert result.provenance["fallbacks"]


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster"])
def test_dense_streamed_inference_choices_match_exactly(frame, covariance):
    arguments = dict(y="y", x=["x", "z"], covariance=covariance,
                     cluster="firm" if covariance == "cluster" else None)
    dense = oe.ols(data=frame, **arguments)
    streamed = oe.ols(data=Dataset.from_frame(frame), **arguments)
    assert dense.inference == streamed.inference
    assert_allclose(streamed.covariance_matrix, dense.covariance_matrix, rtol=2e-11, atol=2e-12)
    assert dense.provenance["solver"] == "torch_qr"
    assert streamed.provenance["solver"] == "torch_tsqr"
    for result in (dense, streamed):
        assert "standard_errors" not in result.inference
        assert "f_statistic" not in result.inference
        assert "inference_details" in result.provenance
        json.loads(result.model_dump_json())


def test_physical_positions_are_full_for_dense_and_bounded_for_streams(frame):
    frame.loc[[2, 33, 477], "z"] = np.nan
    frame.loc[[10, 530], "y"] = np.nan
    retained = np.flatnonzero(frame[["y", "x", "z"]].notna().all(axis=1)).tolist()
    dense = oe.ols(data=frame, y="y", x=["x", "z"])
    streamed = oe.ols(data=Dataset.from_frame(frame), y="y", x=["x", "z"])
    assert dense.sample_positions == retained
    assert len(dense.sample_positions) > 400
    assert streamed.sample_positions == []
    assert streamed.provenance["sample_position_count"] == len(retained)
    assert streamed.provenance["sample_positions_omitted"] is True
    for result in (dense, streamed):
        assert [row["row"] for row in result.predictions] == retained[:400]
        assert result.nobs == len(retained)
    assert len(streamed.model_dump_json()) < 100_000


def test_existing_analysis_memory_guard_routes_to_streaming(frame, monkeypatch):
    monkeypatch.setattr("openecon.analysis._MAX_DESIGN_BYTES", 1000)
    result = oe.ols(data=frame, y="y", x=["x", "z"])
    assert result.provenance["solver"] == "torch_tsqr"
    assert result.sample_positions == []


@pytest.mark.parametrize("terms", [["z"], ["y"], ["C(x)"]])
def test_direct_spec_cannot_override_declared_input_roles(terms):
    spec = oe.ModelSpec(outcome="y", predictors=["x"], options={"terms": terms})
    with pytest.raises(oe.AnalysisError) as error:
        oe.fit(spec, data=object())
    assert error.value.code == "invalid_spec"


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("covariance,hansen", [
    ("HC2", False), ("HC3", False), ("HC3", True),
    ("cluster_hc2", False), ("cluster_hc3", False), ("cluster_hc3", True),
])
def test_single_slope_model_f_uses_the_same_adjusted_contrast(frame, streamed, covariance, hansen):
    data = Dataset.from_frame(frame) if streamed else frame
    result = oe.ols(data=data, y="y", x=["x"], covariance=covariance,
                    cluster="firm" if covariance.startswith("cluster") else None,
                    dfadjust=True, hansen=hansen)
    coefficient = result.coefficients[1]
    contrast = result.lincom({"x": 1})
    test = result.tests["model"]
    assert test["df"] == 1
    assert test["df2"] == pytest.approx(contrast["df"], rel=1e-12)
    assert test["statistic"] == pytest.approx(coefficient.statistic**2, rel=1e-12)
    assert test["p_value"] == pytest.approx(coefficient.p_value, rel=2e-11, abs=1e-14)


@pytest.mark.parametrize("kind", ["residuals", "resid", "residual", "score", "rstandard",
                                  "rstudent", "cook", "cooksd", "dfits", "dffits",
                                  "covratio", "welsch", "dfbeta", "dfbetas"])
def test_iter_predict_projects_all_outcome_aliases_and_preserves_values(frame, kind):
    result = oe.ols(data=frame, y="y", x=["x", "z"])
    batches = list(result.iter_predict(batch_rows=31, kind=kind))
    actual = pd.concat(batches)
    expected = result.predict(kind=kind)
    assert max(map(len, batches)) <= 31
    assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=2e-11, atol=2e-12, equal_nan=True)
    assert [position for part in batches for position in part.attrs["physical_positions"]] == list(range(len(frame)))


@pytest.mark.parametrize("streamed", [False, True])
def test_iter_predict_projects_weights_for_aweight_leverage(frame, streamed):
    data = Dataset.from_frame(frame) if streamed else frame
    result = oe.ols(data=data, y="y", x=["x", "z"], weights="w", weight_type="aweight")
    actual = pd.concat(list(result.iter_predict(kind="hat", batch_rows=19)))
    reference = oe.ols(data=frame, y="y", x=["x", "z"], weights="w", weight_type="aweight")
    assert_allclose(actual.to_numpy(), reference.predict(kind="hat").to_numpy(), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_iter_predict_preserves_lag_and_difference_history_across_small_batches(frame, streamed, external):
    frame.loc[24, "y"] = np.nan
    frame.loc[37, "w"] = 0
    formula = "y ~ L(x, 2) + D(z)"
    dense = oe.ols(data=frame, formula=formula, time="t", weights="w", weight_type="aweight")
    result = (oe.ols(data=Dataset.from_frame(frame), formula=formula, time="t", weights="w", weight_type="aweight")
              if streamed else dense)
    data = Dataset.from_frame(frame) if external else None
    actual = pd.concat(list(result.iter_predict(data=data, batch_rows=7)))
    expected = dense.predict(data=frame)
    assert len(actual) == len(frame)
    assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=2e-11, atol=2e-12, equal_nan=True)


def test_iter_predict_preserves_unsorted_dense_lag_order(frame):
    shuffled = frame.sample(frac=1, random_state=48)
    result = oe.ols(data=shuffled, formula="y ~ L(x) + D(z)", time="t")
    actual = pd.concat(list(result.iter_predict(batch_rows=11)))
    expected = result.predict()
    assert actual.index.tolist() == shuffled.index.tolist()
    assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=2e-11, atol=2e-12, equal_nan=True)


def test_iter_predict_rechecks_factory_content_on_exhaustion(frame):
    source = Dataset.from_batches(lambda: iter([frame.copy()]), list(frame.columns))
    result = oe.ols(data=source, y="y", x=["x", "z"])
    frame.loc[21, "x"] += .125
    with pytest.raises(oe.AnalysisError) as error:
        list(result.iter_predict(batch_rows=23))
    assert error.value.code in {"dataset_changed", "source_changed"}


@pytest.mark.parametrize("n", [90, 99])
@pytest.mark.parametrize("chunk", [1, 3, 7, 4096])
def test_iweight_total_is_chunk_independent_before_integer_truncation(n, chunk):
    weights = torch.tensor([1.1, 1.2, .7] * (n // 3), dtype=torch.float64)
    accumulator = _ExactWeightSum()
    for batch in weights.split(chunk):
        accumulator.add(batch)
    expected = float(sum((Fraction(value) for value in weights.tolist()), Fraction()))
    assert accumulator.total() == expected == math.fsum(weights.tolist())
    assert int(accumulator.total()) == n


@pytest.mark.parametrize("n", [90, 99])
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_public_iweight_residual_df_is_independent_of_source_chunking(frame, n, chunk):
    data = frame.iloc[:n].copy()
    data["w"] = [1.1, 1.2, .7] * (n // 3)
    source = Dataset.from_batches(
        lambda: (data.iloc[start:start + chunk] for start in range(0, n, chunk)), list(data.columns))
    arguments = dict(y="y", x=["x", "z"], weights="w", weight_type="iweight", covariance="nonrobust")
    dense = oe.ols(data=data, **arguments)
    streamed = oe.ols(data=source, **arguments)
    assert dense.nobs == streamed.nobs == n
    assert dense.metrics["df_resid"] == streamed.metrics["df_resid"] == n - 3
    assert_allclose(streamed.covariance_matrix, dense.covariance_matrix, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("kind", ["residual", "score", "rstandard", "rstudent", "cooksd"])
def test_explicit_dataset_prediction_projects_outcome_for_aliases(frame, kind):
    result = oe.ols(data=frame, y="y", x=["x", "z"])
    actual = pd.concat(list(result.iter_predict(data=Dataset.from_frame(frame), kind=kind, batch_rows=17)))
    assert_allclose(actual.to_numpy(), result.predict(data=frame, kind=kind).to_numpy(), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("streamed", [False, True])
def test_normal_bootstrap_has_model_wald_test_and_correct_publication(frame, streamed):
    from scipy.stats import chi2
    data = Dataset.from_frame(frame) if streamed else frame
    result = oe.ols(data=data, y="y", x=["x", "z"], covariance="bootstrap", reps=19, seed=217)
    parameters = np.array([coefficient.estimate for coefficient in result.coefficients])[1:]
    covariance = np.array(result.covariance_matrix)[1:, 1:]
    expected = parameters @ np.linalg.solve(covariance, parameters)
    test = result.tests["model"]
    assert test["distribution"] == "chi2" and test["df"] == 2 and test["df2"] is None
    assert test["statistic"] == pytest.approx(expected, rel=2e-12)
    assert test["p_value"] == pytest.approx(chi2.sf(expected, 2), rel=2e-10, abs=1e-100)
    assert "Model Wald test: chi2(2)" in result.summary()
    assert "model Wald chi-square(2)" in result.to_latex()
