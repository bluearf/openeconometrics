"""RE replay scientific degeneracies, original units and bounded storage."""
import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import streaming_re as native
from openecon.resources import use_workspace_budget

from test_econ_streaming_re import data, spec


@pytest.mark.parametrize("sa", [False, True])
@pytest.mark.parametrize("response_scale", [1e-60, 1., 1e60])
def test_huge_offsets_and_units_obey_independent_affine_invariance(sa, response_scale):
    frame = data(True)
    baseline = native.fit_streaming_re(spec(sa=sa), Dataset.from_frame(frame), batch_rows=23)
    frame.x = (frame.x+2**27)*1e-45
    frame.z = frame.z*1e45
    frame.y = frame.y*response_scale
    answer = native.fit_streaming_re(spec(sa=sa), Dataset.from_frame(frame), batch_rows=31)
    transform = np.array([[1., -2**27, 0.], [0., 1e45, 0.], [0., 0., 1e-45]])*response_scale
    np.testing.assert_allclose([c.estimate for c in answer.coefficients], transform@np.array([c.estimate for c in baseline.coefficients]), rtol=2e-7, atol=2e-7*response_scale)
    expected = transform@np.array(baseline.covariance_matrix)@transform.T
    np.testing.assert_allclose(answer.covariance_matrix, expected, rtol=2e-7, atol=0.)
    assert answer.metrics["sigma_e"] == pytest.approx(baseline.metrics["sigma_e"]*response_scale, rel=2e-8)
    assert answer.metrics["sigma_u"] == pytest.approx(baseline.metrics["sigma_u"]*response_scale, rel=2e-8)
    assert answer.tests["breusch_pagan"]["statistic"] == pytest.approx(baseline.tests["breusch_pagan"]["statistic"], rel=2e-8)


@pytest.mark.parametrize("failure", ["constant", "within_constant", "within_perfect", "singletons", "few_groups", "cluster_nesting", "duplicate_time"])
def test_scientific_guard_and_owned_storage_cleanup(failure, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, s = data(), spec()
    if failure=="constant":
        frame.y = 2.
        code = "no_within_variation"
    elif failure=="within_constant":
        frame.y = frame.g*.4
        code = "no_within_variation"
    elif failure=="within_perfect":
        frame.y = 1+frame.x+frame.g*.3
        code = "no_within_variation"
    elif failure=="singletons":
        frame.g = np.arange(len(frame))
        code = "insufficient_observations"
    elif failure=="few_groups":
        frame.g = frame.g%2
        s = spec(time=False)
        code = "insufficient_observations"
    elif failure=="cluster_nesting":
        frame.c1 = np.arange(len(frame))%7
        s = spec(covariance="cluster", cluster="c1")
        code = "cluster_not_nested"
    else:
        row = frame.index[frame.g==frame.g.iloc[0]][-1]
        frame.loc[row, "t"] = frame.t.iloc[0]
        code = "repeated_time_values"
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re(s, Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_resource_refusal_precedes_group_storage_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Insufficient workspace must fail before SQLite caches are created.")
    monkeypatch.setattr(native, "_GroupMeans", forbidden)
    with use_workspace_budget(8), pytest.raises(AnalysisError) as error:
        native.fit_streaming_re(spec(covariance="robust"), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "workspace_limit"


def test_source_mutation_after_group_snapshot_is_detected_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data()
    source = Dataset.from_frame(frame)
    # Last raw replay currently precedes theta-summary. Mutate during the
    # within factor instead to exercise an actual subsequent source pass.
    factor = native._factor
    armed = False
    def mutate_after_within(*args, **kwargs):
        nonlocal armed
        answer = factor(*args, **kwargs)
        if not armed:
            frame.loc[0, "y"] += 1
            armed = True
        return answer
    monkeypatch.setattr(native, "_factor", mutate_after_within)
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re(spec(), source, batch_rows=17)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("sa", [False, True])
def test_many_dispersed_and_long_groups_never_collect_dataset(sa, monkeypatch):
    frame = data(True, groups=1003, periods=6)
    source = Dataset.from_frame(frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("Native RE must not collect its input.")
    monkeypatch.setattr(source, "collect", forbidden, raising=False)
    result = native.fit_streaming_re(spec(sa=sa), source, batch_rows=41)
    assert result.nobs == len(frame)
    assert result.metrics["n_groups"] == 1003
    assert result.provenance["streaming"]["maximum_batch_rows"]<=41
    assert result.provenance["solver_diagnostics"]["fixed_effect_peak_cache_accounted_bytes"]<=4*1024**2


def test_zero_between_component_matches_pooled_gls():
    frame = data()
    # Remove group means of every random regressor and noise, while keeping
    # the global model identified. The within residual variance stays positive.
    frame[["x", "z", "y"]] -= frame.groupby("g")[["x", "z", "y"]].transform("mean")
    result = native.fit_streaming_re(spec(), Dataset.from_frame(frame), batch_rows=17)
    x = np.column_stack((np.ones(len(frame)), frame[["x", "z"]]))
    beta = np.linalg.lstsq(x, frame.y, rcond=None)[0]
    np.testing.assert_allclose([c.estimate for c in result.coefficients], beta, rtol=2e-11, atol=2e-11)
    assert result.metrics["sigma_u"] == 0.
    assert all(value==0. for value in result.extra["theta"].values())
    assert any("reduces to pooled" in text for text in result.warnings)


def test_unsupported_weights_refused_before_read(monkeypatch):
    frame = data()
    frame["w"] = 1
    s = spec().model_copy(update={"weights": "w", "weight_type": "aweight"})
    source = Dataset.from_frame(frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("RE weight validation must precede reading.")
    monkeypatch.setattr(source, "iter_batches", forbidden)
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re(s, source, batch_rows=17)
    assert error.value.code == "unsupported_weights"


def test_exact_large_integer_period_keys_and_global_missing_categories():
    frame = data(True)
    frame.t = frame.t.astype("int64")+2**60
    missing = frame.iloc[[0]].copy()
    missing["cat"] = "only_missing"
    missing["y"] = np.nan
    frame = pd.concat([missing, frame], ignore_index=True)
    s = spec(categorical=True).model_copy(update={"missing": "drop"})
    result = native.fit_streaming_re(s, Dataset.from_frame(frame), batch_rows=17)
    assert result.dropped_rows == 1
    assert "only_missing" in result.provenance["categorical_encoding"]["cat"]["levels"]
    assert result.nobs == len(frame)-1
