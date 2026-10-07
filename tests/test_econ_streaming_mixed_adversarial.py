"""Independent scale, score/gradient and failure contracts for global mixed replay."""
import numpy as np
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.streaming_mixed import _Likelihood, fit_streaming_mixed

from test_econ_streaming_mixed import data, parity, spec


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_native_analytic_profile_gradient_against_independent_full_v(method):
    frame = data(n=90)
    x, y = np.column_stack((np.ones(len(frame)), frame.x)), frame.y.to_numpy()
    means, mass = [], []
    within = []
    for _, part in frame.groupby("g"):
        d = np.column_stack((np.ones(len(part)), part.x, part.y))
        means.append(d.mean(0))
        mass.append(len(part))
        within.extend(d-d.mean(0))
    class Groups:
        def group_batches(self, rows):
            # Nontrivial partitions check native group reduction invariance.
            for start in range(0, len(mass), 7):
                m = torch.tensor(np.array(means[start:start+7]), dtype=torch.float64)
                n = torch.tensor(mass[start:start+7], dtype=torch.float64)
                yield m, n, n, []
    sample = type("Sample", (), {"nobs": len(frame), "rows": 7})()
    like = _Likelihood(sample, Groups(), torch.tensor(np.array(within), dtype=torch.float64), method=="reml")
    theta = np.array([-.23, -.62])
    group_equal = frame.g.to_numpy()[:, None]==frame.g.to_numpy()[None, :]
    def oracle(t):
        v = np.exp(2*t[1])*np.eye(len(frame))+np.exp(2*t[0])*group_equal
        px, py = np.linalg.solve(v, x), np.linalg.solve(v, y)
        gram = x.T@px
        b = np.linalg.solve(gram, x.T@py)
        r = y-x@b
        pr = x.shape[1] if method=="reml" else 0
        ll = -.5*((len(frame)-pr)*np.log(2*np.pi)+np.linalg.slogdet(v)[1]+r@np.linalg.solve(v, r))
        if pr:
            ll -= .5*np.linalg.slogdet(gram)[1]
        return ll
    h = 2e-5
    gradient = np.array([(oracle(theta+h*np.eye(2)[i])-oracle(theta-h*np.eye(2)[i]))/(2*h) for i in range(2)])
    out = like.evaluate(torch.tensor(theta), gradient=True)
    assert float(out.value) == pytest.approx(oracle(theta), rel=1e-12)
    np.testing.assert_allclose(out.gradient, gradient, rtol=2e-9, atol=2e-8)


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_scale_and_large_offset_preserve_original_coordinate_inference(method):
    frame = data()
    first = fit_streaming_mixed(spec(method), Dataset.from_frame(frame), batch_rows=17)
    shift_x, shift_y, unit_x, unit_y = 2**25, 2**30, 1e-40, 1e40
    changed = frame.copy()
    changed.x = (frame.x+shift_x)*unit_x
    changed.y = (frame.y+shift_y)*unit_y
    second = fit_streaming_mixed(spec(method), Dataset.from_frame(changed), batch_rows=17)
    beta = np.array([c.estimate for c in first.coefficients])
    transform = np.array([[unit_y, -shift_x*unit_y, 0, 0], [0, unit_y/unit_x, 0, 0],
                          [0, 0, unit_y**2, 0], [0, 0, 0, unit_y**2]])
    expected = transform@beta
    expected[0] += shift_y*unit_y
    np.testing.assert_allclose([c.estimate for c in second.coefficients], expected, rtol=5e-6)
    np.testing.assert_allclose(second.covariance_matrix, transform@np.array(first.covariance_matrix)@transform.T, rtol=5e-6, atol=1e-10)
    # Arbitrary slope scaling changes the conventional REML determinant constant;
    # the likelihood reported in original coordinates must account for it.
    expected_ll = first.metrics["log_likelihood"]-len(frame)*np.log(unit_y)
    if method=="reml":
        expected_ll += 2*np.log(unit_y)-np.log(unit_x)
    assert second.metrics["log_likelihood"] == pytest.approx(expected_ll, rel=1e-10, abs=2e-5)


def test_many_long_panel_rows_replay_without_collection(monkeypatch):
    frame = data(n=5003)
    frame.g = np.arange(len(frame))%3
    frame.y = 1+.7*frame.x+np.array([-1.5, .2, 1.8])[frame.g]+np.random.default_rng(184).normal(0, .6, len(frame))
    source = Dataset.from_frame(frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("mixed must not collect or load one complete panel")
    monkeypatch.setattr(source, "collect", forbidden, raising=False)
    fit = fit_streaming_mixed(spec(), source, batch_rows=17)
    assert fit.metrics["group_size_max"]>1600
    assert fit.provenance["streaming"]["maximum_batch_rows"]<=17
    from openecon.econometrics.mixed.lmm import fit_mixed
    parity(fit_mixed(spec(), frame), fit)


def test_changed_source_during_actual_group_score_replay_removes_owned_spill(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, changed = data(), False
    import openecon.econometrics.streaming_mixed as native
    original_start = native._start
    def arm_after_global_moments(like):
        nonlocal changed
        result = original_start(like)
        changed = True
        return result
    monkeypatch.setattr(native, "_start", arm_after_global_moments)
    def reader():
        rows = frame.copy()
        if changed:
            rows.loc[4, "y"] += 1
        yield rows
    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(spec(covariance="robust"), source, batch_rows=17)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["constant", "perfect", "single_group", "single_rows", "cluster_nesting", "robust_reml"])
def test_scientific_degeneracy_and_covariance_guards(failure):
    frame, s = data(), spec()
    if failure=="constant":
        frame.y = 1.
        code = "constant_outcome"
    elif failure=="perfect":
        frame.y = 1+.7*frame.x
        code = "perfect_fit"
    elif failure=="single_group":
        frame.g = 1
        code = "insufficient_groups"
    elif failure=="single_rows":
        frame.g = np.arange(len(frame))
        code = "insufficient_group_size"
    elif failure=="cluster_nesting":
        frame.cluster = np.arange(len(frame))%7
        s = spec(covariance="cluster")
        code = "cluster_not_nested"
    else:
        s = spec("reml", "robust")
        code = "unsupported_covariance"
    with pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(s, Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == code


def test_fweights_equal_actual_row_replication_joint_covariance():
    frame = data()
    weighted = fit_streaming_mixed(spec(covariance="robust", weights=True), Dataset.from_frame(frame), batch_rows=17)
    replicated = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    reference = fit_streaming_mixed(spec(covariance="robust"), Dataset.from_frame(replicated), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in weighted.coefficients], [c.estimate for c in reference.coefficients], rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(weighted.covariance_matrix, reference.covariance_matrix, rtol=2e-8, atol=2e-8)
