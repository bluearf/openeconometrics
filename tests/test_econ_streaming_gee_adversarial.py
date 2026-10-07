"""Long panels, unusual links and strict GEE replay rejection oracles."""
import numpy as np
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.mixed.gee import fit_xtgee
from openecon.econometrics.streaming_gee import fit_streaming_gee

from test_econ_streaming_gee import data, parity, spec


@pytest.mark.parametrize("family,link", [
    ("gaussian", "log"), ("gaussian", "reciprocal"), ("gaussian", "inverse_squared"),
    ("gaussian", "logit"), ("gaussian", "probit"), ("gaussian", "cloglog"), ("gaussian", "loglog"),
    ("binomial", "probit"), ("binomial", "cloglog"), ("binomial", "loglog"),
    ("binomial", "log"), ("binomial", "identity"),
    ("poisson", "identity"), ("poisson", "reciprocal"), ("poisson", "inverse_squared"),
    ("gamma", "identity"), ("gamma", "reciprocal"), ("gamma", "inverse_squared"),
    ("igaussian", "identity"), ("igaussian", "reciprocal"), ("igaussian", "inverse_squared"),
    ("nbinomial", "identity"), ("nbinomial", "reciprocal"), ("nbinomial", "inverse_squared"),
])
def test_all_other_current_links_on_valid_domains(family, link):
    frame = data(family)
    if family == "gaussian":
        frame.y = .4+.05*np.tanh(frame.x)+np.random.default_rng(183).normal(0, .01, len(frame))
    # Modest bounded predictors keep noncanonical starting/fitted means inside
    # their actual domains; domains are still checked at every native step.
    frame.x *= .02
    s = spec(family, "independent", "robust", link=link)
    parity(fit_xtgee(s, frame), fit_streaming_gee(s, Dataset.from_frame(frame), batch_rows=17))


def test_long_panels_no_full_panel_or_correlation_allocation(monkeypatch):
    frame = data(n=5003)
    frame.panel = np.arange(len(frame))%3
    frame.y += .8*frame.panel
    source = Dataset.from_frame(frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("GEE must not collect any source or whole panel.")
    monkeypatch.setattr(source, "collect", forbidden, raising=False)
    result = fit_streaming_gee(spec(covariance="robust"), source, batch_rows=17)
    assert result.metrics["group_size_max"]>1600
    assert "working_correlation" not in result.extra
    assert result.provenance["streaming"]["maximum_batch_rows"]<=17
    parity(fit_xtgee(spec(covariance="robust"), frame), result)


def test_gaussian_large_shift_and_slope_scale_covariance_invariance():
    frame = data()
    first = fit_streaming_gee(spec(covariance="robust"), Dataset.from_frame(frame), batch_rows=17)
    frame.x = (frame.x+2**30)*1e-70
    frame.y += 2**35
    second = fit_streaming_gee(spec(covariance="robust"), Dataset.from_frame(frame), batch_rows=17)
    b = np.array([c.estimate for c in first.coefficients])
    t = np.array([[1, -2**30], [0, 1e70]])
    expected = t@b+np.array([2**35, 0])
    np.testing.assert_allclose([c.estimate for c in second.coefficients], expected, rtol=4e-5)
    np.testing.assert_allclose(second.covariance_matrix, t@np.array(first.covariance_matrix)@t.T, rtol=4e-5)


def test_changed_source_mid_group_moments_cleaned(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, calls = data(), 0
    def reader():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls>=10:
            changed.loc[4, "y"] += 1
        yield changed
    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_gee(spec(), source, batch_rows=17)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["constant", "perfect", "single_group", "all_short", "invalid_exposure", "bad_scale"])
def test_explicit_input_or_inference_rejections(failure):
    frame = data()
    s = spec()
    if failure == "constant":
        frame.y = 1.
        code = "constant_outcome"
    elif failure == "perfect":
        frame.y = 1+.3*frame.x
        code = "perfect_fit"
    elif failure == "single_group":
        frame.panel = 1
        code = "insufficient_groups"
    elif failure == "all_short":
        frame.panel = np.arange(len(frame))
        code = "insufficient_panel_length"
    elif failure == "invalid_exposure":
        s = s.model_copy(update={"columns": {"exposure": "exposure"}})
        frame.loc[4, "exposure"] = 0.
        code = "invalid_exposure"
    else:
        s = s.model_copy(update={"options": {"scale": []}})
        code = "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        fit_streaming_gee(s, Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == code
