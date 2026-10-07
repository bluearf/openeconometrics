"""Native IV/panel option parity across physical batch boundaries.

Resident references have fewer than the automatic replay threshold. Assertions
include full covariance, diagnostics and native weighting/sample conventions.
"""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.dataset import Dataset


@pytest.fixture
def frame():
    generator = torch.Generator().manual_seed(98751)
    rows = 360
    values = torch.randn(rows, 7, dtype=torch.float64, generator=generator)
    data = pd.DataFrame(values.numpy(), columns=["x", "z1", "z2", "z3", "u", "e", "v"])
    data["id"] = torch.arange(rows).div(9, rounding_mode="floor").numpy()
    data["time"] = torch.arange(rows).remainder(9).numpy()
    panel = torch.randn(40, dtype=torch.float64, generator=generator).numpy()
    data["endog"] = .9*data.z1+.3*data.z2+.2*data.z3+.6*data.u+.3*data.e+.6*panel[data.id]
    data["y"] = 1+.4*data.x+.7*data.endog+.9*data.u+data.v+panel[data.id]
    data["cluster"] = data.id//4
    data["cross_cluster"] = torch.arange(rows).remainder(13).numpy()
    data["aw"] = .2+data.x.abs()
    data["fw"] = [1, 2, 3]*120
    data["cat"] = ["a", "b", "c"]*120
    return data


def source(data, size):
    def batches():
        for start in range(0, len(data), size):
            yield data.iloc[start:start+size]
    return Dataset.from_batches(batches, list(data.columns), row_count=len(data))


def numeric_tree(reference, actual, *, tolerance=2e-8):
    if isinstance(reference, dict):
        assert set(actual) == set(reference)
        for key in reference:
            numeric_tree(reference[key], actual[key], tolerance=tolerance)
    elif isinstance(reference, list):
        assert len(actual) == len(reference)
        for left, right in zip(reference, actual, strict=True):
            numeric_tree(left, right, tolerance=tolerance)
    elif reference is None:
        assert actual is None
    elif isinstance(reference, (int, float)):
        assert actual == pytest.approx(reference, rel=tolerance, abs=tolerance)
    else:
        assert actual == reference


def compare(reference, actual, *, tolerance=2e-8):
    assert actual.nobs == reference.nobs
    assert actual.nobs_original == reference.nobs_original
    assert actual.dropped_rows == reference.dropped_rows
    assert [row.term for row in actual.coefficients] == [row.term for row in reference.coefficients]
    torch.testing.assert_close(torch.tensor([row.estimate for row in actual.coefficients], dtype=torch.float64),
                               torch.tensor([row.estimate for row in reference.coefficients], dtype=torch.float64),
                               rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(torch.tensor(actual.covariance_matrix, dtype=torch.float64),
                               torch.tensor(reference.covariance_matrix, dtype=torch.float64), rtol=tolerance, atol=tolerance)
    numeric_tree(reference.metrics, actual.metrics, tolerance=tolerance)
    numeric_tree(reference.tests, actual.tests, tolerance=tolerance)
    if "first_stage" in reference.extra:
        numeric_tree(reference.extra["first_stage"], actual.extra["first_stage"], tolerance=tolerance)
    assert actual.provenance["streaming"]["dense_observation_matrix"] is False
    assert actual.sample_positions == []
    assert len(actual.predictions) <= 400
    predictions = {row["row"]: row for row in reference.predictions}
    for row in actual.predictions:
        if row["row"] in predictions:
            for key in ("fitted", "observed", "residual"):
                assert row[key] == pytest.approx(predictions[row["row"]][key], rel=tolerance, abs=tolerance)


@pytest.mark.parametrize("size", [17, 193])
@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster", "hac"])
def test_all_iv_methods_covariances_and_diagnostics(frame, size, method, covariance):
    kwargs = dict(y="y", x=["x"], endog=["endog"], instruments=["z1", "z2", "z3"],
                  method=method, covariance=covariance, small=size == 17)
    if covariance == "cluster":
        kwargs["cluster"] = ["id", "cluster"] if method == "gmm" else ["cluster", "cross_cluster"]
    elif covariance == "hac":
        kwargs.update(time="time", panel="id", lags=3, kernel="parzen")
    compare(oe.ivregress(data=frame, **kwargs), oe.ivregress(data=source(frame, size), **kwargs))


@pytest.mark.parametrize("method", ["liml", "gmm"])
@pytest.mark.parametrize("weight_type", ["aweight", "fweight", "pweight"])
@pytest.mark.parametrize("intercept", [True, False])
def test_iv_weights_missing_categories_and_intercept(frame, method, weight_type, intercept):
    data = frame.copy()
    data.loc[[5, 217], "x"] = float("nan")
    kwargs = dict(y="y", x=["x", "cat"], categorical=["cat"], endog=["endog"],
                  instruments=["z1", "z2", "z3"], method=method, covariance="robust",
                  weights="fw" if weight_type == "fweight" else "aw", weight_type=weight_type,
                  intercept=intercept, missing="drop", small=True)
    if method == "gmm":
        kwargs.update(center=True, igmm=True)
    compare(oe.ivregress(data=data, **kwargs), oe.ivregress(data=source(data, 31), **kwargs))


@pytest.mark.parametrize("wmatrix,covariance", [("unadjusted", "nonrobust"), ("unadjusted", "robust"),
                 ("robust", "nonrobust"), ("robust", "hac"), ("hac", "robust"), ("cluster", "cluster")])
@pytest.mark.parametrize("center,igmm", [(False, False), (True, True)])
def test_gmm_weight_matrix_centering_iteration_and_reported_sandwich(frame, wmatrix, covariance, center, igmm):
    kwargs = dict(y="y", x=["x"], endog=["endog"], instruments=["z1", "z2", "z3"],
                  method="gmm", covariance=covariance, wmatrix=wmatrix, center=center, igmm=igmm)
    if covariance == "cluster":
        kwargs["cluster"] = "id"
    if covariance == "hac" or wmatrix == "hac":
        kwargs.update(time="time", panel="id", lags=2, kernel="bartlett")
    reference = oe.ivregress(data=frame, **kwargs)
    actual = oe.ivregress(data=source(frame, 43), **kwargs)
    compare(reference, actual)
    numeric_tree(reference.extra["gmm"], actual.extra["gmm"])


@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("size", [23, 197])
def test_absorbed_iv_methods_and_full_diagnostics(frame, method, covariance, size):
    kwargs = dict(y="y", x=["x"], endog=["endog"], instruments=["z1", "z2", "z3"],
                  absorb=["id", "time"], method=method, covariance=covariance, small=size == 23)
    if covariance == "cluster":
        kwargs["cluster"] = "id"
    compare(oe.ivreghdfe(data=frame, **kwargs), oe.ivreghdfe(data=source(frame, size), **kwargs))


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("size", [17, 197])
def test_ec2sls_unbalanced_panels_and_instrument_rank(frame, covariance, size):
    data = frame.drop(index=[1, 8, 59, 107, 108, 311]).copy()
    data["between_only"] = data.id//3
    kwargs = dict(y="y", x=["x", "between_only"], endog=["endog"], instruments=["z1", "z2", "z3"],
                  panel="id", time="time", model="re", ec2sls=True, covariance=covariance, small=size == 17)
    if covariance == "cluster":
        kwargs["cluster"] = "cluster"
    reference = oe.xtivreg(data=data, **kwargs)
    actual = oe.xtivreg(data=source(data, size), **kwargs)
    compare(reference, actual)
    assert actual.extra["estimator"] == "ec2sls"
    assert actual.extra["instrument_columns_used"] == reference.extra["instrument_columns_used"]
    numeric_tree(reference.extra["variance_components"], actual.extra["variance_components"])


@pytest.mark.parametrize("model,covariance", [("pooled", "nonrobust"), ("pooled", "robust"),
                ("pooled", "cluster"), ("pooled", "driscoll_kraay"), ("fd", "nonrobust"),
                ("fd", "robust"), ("fd", "cluster")])
@pytest.mark.parametrize("weight_type", [None, "aweight", "fweight", "pweight"])
def test_pooled_and_fd_all_weights_and_covariances(frame, model, covariance, weight_type):
    if weight_type == "pweight" and covariance == "nonrobust":
        return
    data = frame.drop(index=[1, 8, 59, 107, 108, 311]).copy()
    kwargs = dict(y="y", x=["x", "cat"], categorical=["cat"], panel="id", time="time", model=model,
                  covariance=covariance)
    if covariance == "cluster":
        kwargs["cluster"] = ["id", "cluster"]
    if covariance == "driscoll_kraay":
        kwargs.update(lags=3, kernel="parzen")
    if weight_type:
        kwargs.update(weights="fw" if weight_type == "fweight" else "aw", weight_type=weight_type)
    compare(oe.xtreg(data=data, **kwargs), oe.xtreg(data=source(data, 23), **kwargs))


@pytest.mark.parametrize("model", ["fe", "pooled"])
@pytest.mark.parametrize("kernel", ["bartlett", "truncated", "parzen", "quadratic_spectral"])
@pytest.mark.parametrize("lags", [0, 3, 17])
def test_dk_cross_section_sums_all_kernels_and_gaps(frame, model, kernel, lags):
    data = frame.drop(index=[1, 8, 59, 107, 108, 311]).copy()
    data["time"] *= 2
    if kernel == "truncated" and lags >= int(data.time.max()-data.time.min()):
        # Full rectangular bandwidth collapses the OLS score sum to zero;
        # the native dense fit itself has no positive inference covariance.
        for inputs in (data, source(data, 17)):
            with pytest.raises(oe.AnalysisError) as caught:
                oe.xtreg(data=inputs, y="y", x=["x"], panel="id", time="time", model=model,
                         covariance="driscoll_kraay", lags=lags, kernel=kernel)
            assert caught.value.code in {"invalid_covariance", "singular_covariance"}
        return
    kwargs = dict(y="y", x=["x"], panel="id", time="time", model=model,
                  covariance="driscoll_kraay", lags=lags, kernel=kernel)
    compare(oe.xtreg(data=data, **kwargs), oe.xtreg(data=source(data, 17), **kwargs))


@pytest.mark.parametrize("lags", [0, 3, 17])
@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
def test_qs_iv_datetime_panels_no_truncation(frame, lags, method):
    data = frame.drop(index=[1, 8, 59, 107, 108, 311]).copy()
    data["time"] = pd.to_datetime("2000-01-01")+pd.to_timedelta(data.time*2, unit="D")
    kwargs = dict(y="y", x=["x"], endog=["endog"], instruments=["z1", "z2", "z3"],
                  method=method, covariance="hac", panel="id", time="time",
                  lags=lags, kernel="quadratic_spectral")
    compare(oe.ivregress(data=data, **kwargs), oe.ivregress(data=source(data, 19), **kwargs))


@pytest.mark.parametrize("size", [17, 197])
def test_ml_random_effects_likelihood_information_and_ancillary_intervals(frame, size):
    data = frame.drop(index=[1, 8, 59, 107, 108, 311]).copy()
    kwargs = dict(y="y", x=["x", "cat"], categorical=["cat"], panel="id", time="time", model="mle")
    reference, actual = oe.xtreg(data=data, **kwargs), oe.xtreg(data=source(data, size), **kwargs)
    compare(reference, actual)
    numeric_tree(reference.extra["ln_sigma"], actual.extra["ln_sigma"])
    for left, right in zip(reference.coefficients[-2:], actual.coefficients[-2:], strict=True):
        assert right.ci_low == pytest.approx(left.ci_low, rel=2e-8, abs=2e-8)
        assert right.ci_high == pytest.approx(left.ci_high, rel=2e-8, abs=2e-8)


@pytest.mark.parametrize("estimator,kwargs", [("ivregress", dict(method="gmm", covariance="robust")),
                  ("xtreg", dict(model="mle", panel="id", time="time")),
                  ("xtreg", dict(model="fd", panel="id", time="time"))])
def test_model_option_replays_refuse_changed_sources(frame, estimator, kwargs):
    calls = 0
    def changing():
        nonlocal calls
        calls += 1
        yield frame.assign(x=frame.x+calls)
    options = dict(y="y", x=["x"], **kwargs)
    if estimator == "ivregress":
        options.update(endog=["endog"], instruments=["z1", "z2", "z3"])
    with pytest.raises(oe.AnalysisError) as caught:
        getattr(oe, estimator)(data=Dataset.from_batches(changing, list(frame.columns)), **options)
    assert caught.value.code == "source_changed"


def test_native_singular_two_way_gmm_moments_are_not_regularized(frame):
    kwargs = dict(y="y", x=["x"], endog=["endog"], instruments=["z1", "z2", "z3"],
                  method="gmm", covariance="cluster", cluster=["cluster", "cross_cluster"])
    for inputs in (frame, source(frame, 19)):
        with pytest.raises(oe.AnalysisError) as caught:
            oe.ivregress(data=inputs, **kwargs)
        assert caught.value.code == "singular_weight_matrix"


@pytest.mark.parametrize("model", ["mle", "fd"])
def test_panel_option_owned_scratch_is_cleaned_after_failure(frame, model, tmp_path, monkeypatch):
    import tempfile
    from openecon.econometrics import streaming_panel_options as native
    from openecon.econometrics.streaming_fd_iv import _Differences
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    if model == "mle":
        def fail_after_store(*args, **kwargs):
            raise RuntimeError("Injected evaluation failure after disk means exist.")
        monkeypatch.setattr(native._DiskRandomEffectsLikelihood, "__call__", fail_after_store)
        exception = oe.AnalysisError
    else:
        original, calls = _Differences.batches, 0
        def interrupt_after_snapshot(self):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("Injected interrupt after immutable FD snapshot.")
            yield from original(self)
        monkeypatch.setattr(_Differences, "batches", interrupt_after_snapshot)
        exception = KeyboardInterrupt
    with pytest.raises(exception):
        oe.xtreg(data=source(frame, 19), y="y", x=["x"], panel="id", time="time", model=model)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
@pytest.mark.parametrize("kernel", ["bartlett", "truncated", "parzen", "quadratic_spectral"])
def test_iv_white_hac_accepts_repeated_time_without_ordering(frame, method, kernel):
    data = frame.assign(time=0)
    kwargs = dict(y="y", x=["x"], endog=["endog"], instruments=["z1", "z2", "z3"],
                  method=method, covariance="hac", time="time", lags=0, kernel=kernel)
    compare(oe.ivregress(data=data, **kwargs), oe.ivregress(data=source(data, 19), **kwargs))


@pytest.mark.parametrize("estimator", ["ivregress", "xtreg"])
def test_ordered_covariance_huge_lags_refuse_before_engine_allocation(frame, estimator, monkeypatch):
    from openecon.engines.streaming_hac import HACAccumulator
    from openecon.econometrics.streaming_panel_options import DriscollKraayAccumulator
    from openecon.resources import use_workspace_budget
    def allocation_is_forbidden(*args, **kwargs):
        raise AssertionError("An ordered covariance engine was allocated before the combined budget check.")
    monkeypatch.setattr(HACAccumulator, "__init__", allocation_is_forbidden)
    monkeypatch.setattr(DriscollKraayAccumulator, "__init__", allocation_is_forbidden)
    kwargs = dict(y="y", x=["x"], time="time", lags=10**8, kernel="bartlett")
    if estimator == "ivregress":
        kwargs.update(endog=["endog"], instruments=["z1", "z2", "z3"], method="gmm", covariance="hac", panel="id")
    else:
        kwargs.update(model="pooled", panel="id", covariance="driscoll_kraay")
    with use_workspace_budget(64), pytest.raises(oe.AnalysisError) as caught:
        getattr(oe, estimator)(data=source(frame, 19), **kwargs)
    assert caught.value.code == "workspace_limit"
    assert "ordered_covariance_engine_buffers" in caught.value.resource_plan["buffers"]


@pytest.mark.parametrize("estimator", ["ivregress", "xtreg"])
def test_ordered_covariance_combines_reader_factors_and_engine_budget(frame, estimator, monkeypatch):
    from openecon.econometrics.replay_sample import ReplaySample
    from openecon.resources import use_workspace_budget
    changes, original = [], ReplaySample.plan_rows
    def track(self, operation, static_buffers, bytes_per_row):
        before = self.rows
        plan = original(self, operation, static_buffers, bytes_per_row)
        if operation == "combined ordered linear covariance and replay buffers":
            changes.append((before, self.rows, plan.record()))
        return plan
    monkeypatch.setattr(ReplaySample, "plan_rows", track)
    kwargs = dict(y="y", x=["x"], time="time", lags=3, kernel="quadratic_spectral", panel="id")
    if estimator == "ivregress":
        kwargs.update(endog=["endog"], instruments=["z1", "z2", "z3"], method="gmm", covariance="hac")
    else:
        kwargs.update(model="pooled", covariance="driscoll_kraay")
    reference = getattr(oe, estimator)(data=frame, **kwargs)
    with use_workspace_budget(64):
        actual = getattr(oe, estimator)(data=source(frame, len(frame)), **kwargs)
    compare(reference, actual)
    assert changes and any(after < before for before, after, _ in changes)
    for _, _, plan in changes:
        assert plan["estimated_workspace_bytes"] <= plan["budget_bytes"] == 64*1024**2
        buffers = plan["buffers"]
        assert buffers["ordered_covariance_engine_buffers"] > 16*1024**2
        assert buffers["ordered_covariance_global_IV_geometry"] > 0
        assert buffers["native_row_block"] > 0
        assert any("reader" in key for key in buffers)


@pytest.mark.parametrize("model", ["pooled", "fd"])
def test_panel_option_rounding_guard_uses_original_outcome_level(frame, model):
    data = frame.assign(y=frame.y+1e16)
    errors = []
    for inputs in (data, source(data, 17)):
        with pytest.raises(oe.AnalysisError) as caught:
            oe.xtreg(data=inputs, y="y", x=["x"], panel="id", time="time", model=model)
        errors.append(caught.value.code)
    assert errors == ["constant_outcome", "constant_outcome"]


def test_dk_single_period_has_same_native_refusal(frame):
    data = frame.loc[frame.time == 0]
    for inputs in (data, source(data, 17)):
        with pytest.raises(oe.AnalysisError) as caught:
            oe.xtreg(data=inputs, y="y", x=["x"], panel="id", time="time", model="pooled", covariance="driscoll_kraay")
        assert caught.value.code == "insufficient_periods"
