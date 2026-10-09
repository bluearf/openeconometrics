"""Full dense parity for native FE/BE replay, including panel-specific guards."""
import numpy as np
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.panel.xtreg import fit_xtreg
from openecon.econometrics.streaming_linear import fit_streaming_linear
from openecon.models import ModelSpec

from test_econ_streaming_linear import assert_parity, assert_group_state_parity, frame


def panel_spec(model, covariance="nonrobust", cluster=None, weight_type=None, wls=False):
    return ModelSpec(estimator="xtreg", outcome="y", predictors=["x", "z", "cat"],
                     categorical=["cat"], panel="group", time="time", covariance=covariance,
                     cluster=cluster, weights="weight" if weight_type else None,
                     weight_type=weight_type, options={"model": model, "wls": wls})


def panel_data(weight_type=None):
    data = frame()
    data["time"] = np.arange(len(data))//31
    data["cluster"] = data.group%7
    data["cluster2"] = data.group%11
    if weight_type:
        data["weight"] = data.group%4+1
    return data.sample(frac=1, random_state=85713).reset_index(drop=True)


@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("robust", None),
                      ("cluster", "cluster"), ("cluster", ["cluster", "cluster2"])])
@pytest.mark.parametrize("weight_type", [None, "aweight", "fweight", "pweight"])
def test_fixed_effects_all_supported_weights_covariances(covariance, cluster, weight_type):
    if weight_type == "pweight" and covariance == "nonrobust":
        pytest.skip("Both routes explicitly reject classical pweights.")
    data = panel_data(weight_type)
    spec = panel_spec("fe", covariance, cluster, weight_type)
    dense = fit_xtreg(spec, data)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, result)
    assert {k: v for k, v in result.extra.items() if k != "group_state"} == pytest.approx(
        {k: v for k, v in dense.extra.items() if k != "group_state"})
    assert_group_state_parity(dense.extra["group_state"], result.extra["group_state"])
    predictions = {item["row"]: item["fitted"] for item in dense.predictions}
    for item in result.predictions:
        assert item["fitted"] == pytest.approx(predictions[item["row"]], rel=1e-10, abs=1e-10)
    assert result.inference["small_sample_correction"] == dense.inference["small_sample_correction"]


@pytest.mark.parametrize("covariance,cluster", [("nonrobust", None), ("robust", None),
                      ("cluster", "cluster"), ("cluster", ["cluster", "cluster2"])])
@pytest.mark.parametrize("wls", [False, True])
def test_between_effects_all_supported_covariances_and_wls(covariance, cluster, wls):
    data = panel_data()
    spec = panel_spec("be", covariance, cluster, wls=wls)
    dense = fit_xtreg(spec, data)
    result = fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, result)
    assert result.extra == pytest.approx(dense.extra)


@pytest.mark.parametrize("model,code", [("fe", "cluster_not_nested"), ("be", "cluster_varies_within_panel")])
def test_non_nested_clusters_rejected(model, code):
    data = panel_data()
    data["cluster"] = np.arange(len(data))%9
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(panel_spec(model, "cluster", "cluster"), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == code


def test_fe_weight_must_be_exactly_constant_within_panel():
    data = panel_data("aweight")
    data["weight"] = data.weight.astype(float)
    data.loc[2, "weight"] += .001
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(panel_spec("fe", weight_type="aweight"), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == "weights_vary_within_panel"


@pytest.mark.parametrize("model", ["fe", "be"])
def test_repeated_panel_time_is_rejected_across_source_blocks(model):
    data = panel_data()
    group = data.loc[0, "group"]
    other = data.index[data.group == group][-1]
    data.loc[other, "time"] = data.loc[0, "time"]
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(panel_spec(model), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == "repeated_time_values"


@pytest.mark.parametrize("model", ["re", "fd", "mle", "pooled"])
def test_other_panel_models_preserve_native_contract(model):
    data, spec = panel_data(), panel_spec(model)
    try:
        reference = fit_xtreg(spec, data)
    except AnalysisError as native_error:
        with pytest.raises(AnalysisError) as replay_error:
            fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17)
        assert replay_error.value.code == native_error.code
    else:
        assert_parity(reference, fit_streaming_linear(spec, Dataset.from_frame(data), batch_rows=17))


def test_no_within_outcome_variation():
    data = panel_data()
    data["y"] = data.group*.4
    with pytest.raises(AnalysisError) as error:
        fit_streaming_linear(panel_spec("fe"), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == "no_within_variation"
