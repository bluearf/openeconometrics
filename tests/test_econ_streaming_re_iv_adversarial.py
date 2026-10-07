"""RE-IV auxiliary identification, physical coordinates and honest scope."""
import numpy as np
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import streaming_re_iv as native
from openecon.econometrics.iv.xtivreg import fit_xtivreg
from openecon.resources import use_workspace_budget

from test_econ_streaming_re_iv import data, spec


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_native_g2sls_units_and_huge_offsets_match_affine_oracle(covariance):
    frame = data(True)
    baseline = native.fit_streaming_re_iv(spec(covariance=covariance), Dataset.from_frame(frame), batch_rows=17)
    frame.x = (frame.x+2**28)*1e-45
    frame.endog *= 1e45
    frame.z1 *= 1e50
    frame.z2 *= 1e-50
    frame.y = (frame.y+2**25)*1e-35
    result = native.fit_streaming_re_iv(spec(covariance=covariance), Dataset.from_frame(frame), batch_rows=23)
    transform = np.array([[1., -2**28, 0.], [0., 1e45, 0.], [0., 0., 1e-45]])*1e-35
    params = transform@np.array([c.estimate for c in baseline.coefficients])
    params[0] += 2**25*1e-35
    np.testing.assert_allclose([c.estimate for c in result.coefficients], params, rtol=3e-7, atol=1e-35*3e-7)
    np.testing.assert_allclose(result.covariance_matrix, transform@np.array(baseline.covariance_matrix)@transform.T, rtol=3e-7, atol=0.)
    assert result.metrics["sigma_e"] == pytest.approx(baseline.metrics["sigma_e"]*1e-35, rel=3e-8)
    assert result.metrics["sigma_u"] == pytest.approx(baseline.metrics["sigma_u"]*1e-35, rel=3e-8)


@pytest.mark.parametrize("which", ["endog", "between"])
def test_auxiliary_missing_endogenous_variation_uses_ols_and_keeps_global_term(which):
    frame = data()
    if which=="endog":
        frame.endog = .7*frame.g+np.random.default_rng(9307).normal(size=37)[frame.g]
    else:
        frame.endog -= frame.groupby("g").endog.transform("mean")
    frame.y = .7+.4*frame.x+1.2*frame.endog+np.random.default_rng(4911).normal(size=len(frame))+.03*frame.g
    s = spec(time=False)
    expected = fit_xtivreg(s, frame)
    result = native.fit_streaming_re_iv(s, Dataset.from_frame(frame), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], [c.estimate for c in expected.coefficients], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(result.covariance_matrix, expected.covariance_matrix, rtol=2e-9, atol=2e-10)
    assert result.extra["endogenous"] == ["endog"]


def test_two_endogenous_empty_exog_exact_identification():
    frame = data(True)
    frame["d2"] = .8*frame.z2+.2*frame.z1+np.random.default_rng(4879).normal(size=len(frame))+.03*frame.g
    frame.y += .7*frame.d2
    s = spec(covariance="robust").model_copy(update={"predictors": [], "columns": {"endogenous": ["endog", "d2"], "instruments": ["z1", "z2"]}})
    expected = fit_xtivreg(s, frame)
    result = native.fit_streaming_re_iv(s, Dataset.from_frame(frame), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], [c.estimate for c in expected.coefficients], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(result.covariance_matrix, expected.covariance_matrix, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("which", ["within", "between"])
def test_separate_auxiliary_identification_guards_not_pooled_proxy(which, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data()
    if which=="within":
        frame.z1 = frame.g*.2
        frame.z2 = (frame.g%7)*.3
    else:
        frame[["z1", "z2"]] -= frame.groupby("g")[["z1", "z2"]].transform("mean")
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re_iv(spec(), Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == "underidentified"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("which", ["singletons", "few_groups", "perfect", "duplicate"])
def test_component_and_source_key_degeneracy_guards(which, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, s = data(), spec()
    if which=="singletons":
        frame.g = np.arange(len(frame))
        code = "insufficient_observations"
    elif which=="few_groups":
        frame.g %= 2
        s = spec(time=False)
        code = "insufficient_observations"
    elif which=="perfect":
        frame.y = 1+.4*frame.x+1.2*frame.endog+frame.g*.2
        code = "perfect_fit"
    else:
        rows = frame.index[frame.g==0]
        frame.loc[rows[-1], "t"] = frame.loc[rows[0], "t"]
        code = "repeated_time_values"
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re_iv(s, Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_resource_refusal_precedes_owned_group_database(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Resource rejection must precede owned group moments.")
    monkeypatch.setattr(native, "_GroupMeans", forbidden)
    with use_workspace_budget(8), pytest.raises(AnalysisError) as error:
        native.fit_streaming_re_iv(spec(covariance="robust"), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "workspace_limit"


def test_native_re_weight_restriction_precedes_source_read(monkeypatch):
    source = Dataset.from_frame(data())
    def forbidden(*args, **kwargs):
        raise AssertionError("An unsupported native weighted RE IV fit must not read a source.")
    monkeypatch.setattr(source, "iter_batches", forbidden)
    s = spec().model_copy(update={"weights": "weight", "weight_type": "aweight",
                                 "options": {"model": "re", "ec2sls": True}})
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re_iv(s, source, batch_rows=17)
    assert error.value.code == "unsupported_weights"


def test_scratch_disk_refusal_precedes_creation_of_group_moments(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from openecon.econometrics import streaming_fd_iv
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(streaming_fd_iv.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    def forbidden(*args, **kwargs):
        raise AssertionError("Disk planning must precede group-moment SQLite creation.")
    monkeypatch.setattr(native, "_GroupMeans", forbidden)
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re_iv(spec(covariance="robust"), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "replay_disk_limit"
    assert error.value.disk_plan["estimated_scratch_bytes"]>1024
    assert not list(tmp_path.iterdir())


def test_mutated_source_during_global_re_fit_is_detected_and_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = data()
    source = Dataset.from_frame(frame)
    original = native._theta
    armed = False
    def mutate_after_components(*args):
        nonlocal armed
        answer = original(*args)
        if not armed:
            frame.loc[0, "y"] += .1
            armed = True
        return answer
    monkeypatch.setattr(native, "_theta", mutate_after_components)
    with pytest.raises(AnalysisError) as error:
        native.fit_streaming_re_iv(spec(), source, batch_rows=17)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_nonnested_cluster_is_explicit_warning_and_dense_contract():
    frame = data()
    frame.cluster = (frame.g+frame.t)%9
    s = spec(covariance="cluster")
    expected = fit_xtivreg(s, frame)
    result = native.fit_streaming_re_iv(s, Dataset.from_frame(frame), batch_rows=17)
    np.testing.assert_allclose(result.covariance_matrix, expected.covariance_matrix, rtol=2e-10, atol=2e-11)
    assert any("not nested" in text for text in result.warnings)
