"""Independent FE instrument-space oracles and bounded failure contracts."""
import hashlib
import struct

import numpy as np
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.iv.ivreghdfe import fit_ivreghdfe
from openecon.econometrics.iv.xtivreg import fit_xtivreg
from openecon.econometrics.streaming_fe_iv import fit_streaming_fe_iv
from openecon.econometrics.streaming_hdfe import _Vectors
from openecon.resources import use_workspace_budget

from test_econ_streaming_fe_iv import data, parity, spec


@pytest.mark.parametrize("estimator,model", [("xtivreg", "fe"), ("xtivreg", "be"), ("ivreghdfe", "fe")])
def test_empty_exog_multiendogenous_exact_identification(estimator, model):
    frame = data()
    rng = np.random.default_rng(923115)
    frame["d2"] = .7*frame.z2+.3*frame.z1+rng.normal(size=len(frame))
    frame["y"] += .6*frame.d2
    s = spec(estimator, model=model, covariance="robust").model_copy(update={"predictors": [],
                 "columns": {**spec(estimator).columns, "endogenous": ["endog", "d2"]}})
    dense = (fit_xtivreg if estimator == "xtivreg" else fit_ivreghdfe)(s, frame)
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    parity(dense, actual, tolerance=2e-7)
    if estimator == "ivreghdfe":
        assert actual.tests["overid_score"]["statistic"] is None


@pytest.mark.parametrize("drop", [False, True])
def test_singletons_missing_frequency_weights_and_batch_alignment(drop):
    frame = data(n=503)
    frame.loc[0, ["f0", "f1", "weight"]] = [1000, 1001, 1]
    frame.loc[1, ["f0", "f1", "weight"]] = [1002, 1003, 2]
    frame.loc[15, "z2"] = np.nan
    frame.loc[16, "weight"] = 0
    s = spec("ivreghdfe", covariance="robust", weight_type="fweight").model_copy(update={
        "missing": "drop", "options": {"method": "2sls", "small": True, "tolerance": 1e-10, "drop_singletons": drop}})
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    parity(fit_ivreghdfe(s, frame), actual)
    second = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=7)
    np.testing.assert_allclose(second.covariance_matrix, actual.covariance_matrix, rtol=1e-8, atol=3e-10)
    positions = [i for i in range(len(frame)) if i not in {15, 16} and (not drop or i != 0)]
    assert actual.provenance["sample_positions_hash"] == hashlib.sha256(b"".join(struct.pack("<q", i) for i in positions)).hexdigest()
    assert actual.extra["singletons_dropped"] == int(drop)


@pytest.mark.parametrize("model", ["fe", "be"])
def test_panel_prediction_source_order_is_raw_observed_endogenous(model):
    frame = data().sample(frac=1, random_state=631).reset_index(drop=True)
    s = spec(model=model, covariance="robust")
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    b = np.array([c.estimate for c in actual.coefficients])
    expected = np.column_stack([np.ones(len(frame)), frame[["x", "endog"]]])@b
    assert [p["row"] for p in actual.predictions] == list(range(len(frame)))
    np.testing.assert_allclose([p["fitted"] for p in actual.predictions], expected, atol=3e-12)
    np.testing.assert_allclose([p["residual"] for p in actual.predictions], frame.y.to_numpy()-expected, atol=3e-12)


def test_large_offsets_centered_dummy_oracle_and_slope_scale_invariance():
    frame = data()
    s = spec("ivreghdfe", covariance="robust")
    first = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    frame.x += 2**30
    frame.y += 2**28
    frame.endog *= 1e-70
    frame.z1 *= 1e70
    second = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in second.coefficients],
                              np.array([c.estimate for c in first.coefficients])*[1, 1e70], rtol=4e-7)
    units = np.diag([1, 1e70])
    np.testing.assert_allclose(second.covariance_matrix, units@np.array(first.covariance_matrix)@units, rtol=8e-7)


def test_absorbed_and_duplicate_instrument_global_omissions():
    frame = data()
    frame["absorbed"] = frame.f0.astype(float)
    frame["duplicate"] = 3*frame.z1
    s = spec("ivreghdfe").model_copy(update={"predictors": ["x", "absorbed"],
          "columns": {"absorb": ["f0", "f1"], "endogenous": ["endog"], "instruments": ["z1", "duplicate", "z2", "absorbed"]}})
    # Absorbed cannot be assigned both exogenous and excluded roles.
    s = s.model_copy(update={"columns": {**s.columns, "instruments": ["z1", "duplicate", "z2"]}})
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    parity(fit_ivreghdfe(s, frame), actual)
    assert "absorbed" in actual.provenance["omitted_terms"]
    assert actual.extra["omitted_instruments"] == ["duplicate"]


def test_three_absorbed_dimensions_with_existing_conservative_dof_convention():
    frame = data()
    frame["f2"] = np.random.default_rng(3416).integers(0, 7, len(frame))
    frame.y += .03*frame.f2
    s = spec("ivreghdfe", covariance="cluster", cluster=["cluster", "f0"], dimensions=3)
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    parity(fit_ivreghdfe(s, frame), actual, tolerance=2e-7)
    assert len(actual.extra["absorbed"]) == 3


def test_nonnested_panel_cluster_uses_absorbed_correction_and_oracle():
    frame = data()
    s = spec(covariance="cluster", cluster="cluster")
    actual = fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    parity(fit_xtivreg(s, frame), actual)
    assert actual.inference["small_sample_correction"] == pytest.approx(17/16*(len(frame)-1)/(len(frame)-23-2))


def test_between_cluster_varying_within_panels_fails_and_cleans_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_fe_iv(spec(model="be", covariance="cluster", cluster="cluster"), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "cluster_varies_within_panel"
    assert not list(tmp_path.iterdir())


def test_workspace_refusal_before_owned_vector_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The workspace must reject before allocating state vectors.")
    monkeypatch.setattr(_Vectors, "__init__", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit_streaming_fe_iv(spec("ivreghdfe"), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "workspace_limit"


def test_changed_source_during_absorbed_iv_projection_cleans_owned_state(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, calls = data(), 0
    def reader():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls >= 10:
            changed.loc[3, "y"] += 1
        yield changed
    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_fe_iv(spec("ivreghdfe"), source, batch_rows=17)
    assert error.value.code == "source_changed"
    assert calls >= 10
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["all_absorbed", "underidentified", "perfect", "duplicate_time"])
def test_degenerate_models_structured_rejection(failure):
    frame = data()
    s = spec("ivreghdfe")
    if failure == "all_absorbed":
        frame.endog = frame.f0.astype(float)
        code = "no_endogenous_regressors"
    elif failure == "underidentified":
        frame.z1, frame.z2 = frame.f0.astype(float), 2*frame.f0.astype(float)
        code = "underidentified"
    elif failure == "perfect":
        frame.y = .3*frame.x+1.2*frame.endog+.07*frame.f0-.02*frame.f1
        code = "perfect_fit"
    else:
        frame["time"] = 1
        s = spec().model_copy(update={"time": "time"})
        code = "repeated_time_values"
    with pytest.raises(AnalysisError) as error:
        fit_streaming_fe_iv(s, Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == code
