"""Exact panel difference keys, spill cleanup, source identity and row guards."""
import hashlib
import struct

import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import streaming_fd_iv as native
from openecon.resources import use_workspace_budget

from test_econ_streaming_fd_iv import data, spec


def test_difference_position_hash_and_batch_invariance_against_pandas_oracle():
    frame = data(gaps=True)
    ordered = frame.sort_values(["g", "t"])
    alive = (ordered.g.diff()==0)&(ordered.t.diff()==1)
    # Native disk sorts encoded panel keys lexically; reproduce its declared
    # chronological second-row ordering rather than equating it with source order.
    from openecon.streaming_design import encode_cluster_labels
    ordered = frame.assign(key=encode_cluster_labels(frame.g)).sort_values(["key", "t"])
    alive = (ordered.key.shift()==ordered.key)&(ordered.t.diff()==1)
    positions = ordered.index[alive].tolist()
    expected_hash = hashlib.sha256(b"".join(struct.pack("<q", i) for i in positions)).hexdigest()
    a = native.fit_streaming_fd_iv(spec(covariance="robust"), Dataset.from_frame(frame), batch_rows=7)
    b = native.fit_streaming_fd_iv(spec(covariance="robust"), Dataset.from_frame(frame), batch_rows=31)
    assert a.provenance["sample_positions_hash"] == expected_hash == b.provenance["sample_positions_hash"]
    np.testing.assert_allclose(a.covariance_matrix, b.covariance_matrix, rtol=2e-11, atol=2e-11)
    assert [p["row"] for p in a.predictions] == positions[:400]


def test_exact_integer_periods_above_float_precision_and_wrap_guard():
    frame = data()
    baseline = native.fit_streaming_fd_iv(spec(), Dataset.from_frame(frame), batch_rows=17)
    frame.t = frame.t.astype("int64")+2**60
    shifted = native.fit_streaming_fd_iv(spec(), Dataset.from_frame(frame), batch_rows=17)
    np.testing.assert_allclose(shifted.covariance_matrix, baseline.covariance_matrix, rtol=2e-11, atol=2e-11)
    # max->min is never a true one-period adjacency despite int64 subtraction wrap.
    two = frame.loc[frame.g==0].iloc[:2].copy()
    two.t = np.array([-(1<<63), (1<<63)-1], dtype="int64")
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_fd_iv(spec(), Dataset.from_frame(two), batch_rows=1)
    assert error.value.code == "empty_sample"


@pytest.mark.parametrize("failure,code", [("duplicate", "repeated_time_values"), ("gaps", "empty_sample"),
    ("noninteger", "invalid_time"), ("bool", "invalid_time"), ("constant_difference", "perfect_fit"),
    ("underidentified", "underidentified")])
def test_early_scientific_guards_and_owned_cleanup(failure, code, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data()
    if failure=="duplicate":
        indexes = frame.index[frame.g==0]
        frame.loc[indexes[-1], "t"] = frame.loc[indexes[0], "t"]
    elif failure=="gaps":
        frame.t *= 2
    elif failure=="noninteger":
        frame.t = frame.t+.1
    elif failure=="bool":
        frame.t = frame.t.astype(bool)
    elif failure=="constant_difference":
        frame.y = 1.+frame.t*.2
    else:
        frame.z1 = frame.g*.3
        frame.z2 = frame.g*.4
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_fd_iv(spec(), Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_early_combined_workspace_refuses_owned_snapshot(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Budget refusal must precede chronological snapshot creation.")
    monkeypatch.setattr(native, "_Differences", forbidden)
    with use_workspace_budget(8), pytest.raises(AnalysisError) as error:
        native.fit_streaming_fd_iv(spec(covariance="robust"), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "workspace_limit"


def test_scratch_disk_refusal_precedes_creation_of_owned_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(native.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    def forbidden(*args, **kwargs):
        raise AssertionError("Disk planning must precede TemporaryDirectory creation.")
    monkeypatch.setattr(native.tempfile, "TemporaryDirectory", forbidden)
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_fd_iv(spec(covariance="robust"), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "replay_disk_limit"
    assert error.value.disk_plan["estimated_scratch_bytes"]>1024
    assert not list(tmp_path.iterdir())


def test_source_mutation_after_snapshot_detected_and_owned_spill_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data()
    source = Dataset.from_frame(frame)
    original = native._Differences.__init__
    def mutate_after_snapshot(self, *args, **kwargs):
        original(self, *args, **kwargs)
        frame.loc[0, "y"] += .1
    monkeypatch.setattr(native._Differences, "__init__", mutate_after_snapshot)
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_fd_iv(spec(), source, batch_rows=17)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_missing_predictors_create_real_gaps_and_categories_stay_global():
    frame = data()
    extra = frame.iloc[[0]].copy()
    extra.cat = "only_missing"
    extra.y = np.nan
    frame.loc[3, "z2"] = np.nan
    frame = pd.concat((extra, frame), ignore_index=True)
    result = native.fit_streaming_fd_iv(spec(categorical=True).model_copy(update={"missing": "drop"}),
                                       Dataset.from_frame(frame), batch_rows=17)
    complete = frame.dropna().sort_values(["g", "t"])
    used = sum((complete.g.diff()==0)&(complete.t.diff()==1))
    assert result.nobs == used
    assert result.dropped_rows == len(frame)-used
    assert "only_missing" in result.provenance["categorical_encoding"]["cat"]["levels"]


def test_multiple_endogenous_empty_exogenous_exact_identification():
    frame = data()
    frame["d2"] = .7*frame.z2+.3*frame.z1+np.random.default_rng(7235).normal(size=len(frame))
    frame.y += .5*frame.d2
    s = spec().model_copy(update={"predictors": [], "columns": {"endogenous": ["endog", "d2"], "instruments": ["z1", "z2"]}})
    from openecon.econometrics.iv.xtivreg import fit_xtivreg
    expected = fit_xtivreg(s, frame)
    result = native.fit_streaming_fd_iv(s, Dataset.from_frame(frame), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], [c.estimate for c in expected.coefficients], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(result.covariance_matrix, expected.covariance_matrix, rtol=2e-10, atol=2e-11)
