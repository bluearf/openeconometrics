"""General mixed replay: native parity and independent explicit-V oracles."""
import builtins
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import make_spec
from openecon.econometrics.mixed.lmm import fit_mixed
from openecon.econometrics.mixed.lmm_kernels import CovStructure
from openecon.econometrics.streaming_mixed import fit_streaming_mixed
from openecon.econometrics.streaming_mixed_general import _Factors, _GeneralLikelihood, _coordinate_scales
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def data(*, two=False, intercept_re=True, seed=7138, top=18, classes=2, size=9):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(top), (classes if two else 1) * size)
    g = np.tile(np.repeat(np.arange(classes if two else 1), size), top)
    low = s * (classes if two else 1) + g
    n = len(s)
    x, w = rng.normal(size=n), rng.normal(size=n)
    z = np.column_stack((x, np.ones(n) if intercept_re else w))
    effects = rng.multivariate_normal([0, 0], [[.9, .2], [.2, .9]], size=low.max() + 1)
    y = 1 + .6 * x - .4 * w + (effects[low] * z).sum(1) + rng.normal(0, .65, n)
    if two:
        y += rng.normal(0, 1.2, top)[s]
    return pd.DataFrame({"s": s, "g": g if two else s, "cluster": s // 2,
                         "x": x, "w": w, "y": y, "f": rng.integers(1, 4, n),
                         "category": pd.Categorical(np.where(np.arange(n) % 3, "b", "a"))})


def spec(*, two=False, intercept_re=True, method="ml", covariance="nonrobust", structure="independent", weights=False, categories=False, intercept=True):
    return make_spec("mixed", outcome="y", predictors=["x", "w", *(["category"] if categories else [])],
        columns={"group": ["s", "g"] if two else ["g"], "random": ["x"] if intercept_re else ["x", "w"]},
        intercept=intercept, covariance=covariance, cluster="cluster" if covariance == "cluster" else None,
        options={"random_intercept": intercept_re, "method": method, "covstructure": structure},
        categorical=["category"] if categories else [], weights="f" if weights else None, weight_type="fweight" if weights else None)


def source(frame, rows=29):
    def chunks():
        for start in range(0, len(frame), rows):
            yield frame.iloc[start:start + rows].copy()
    return Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))


def parity(expected, actual, *, tol=3e-5):
    assert [(c.term, c.equation) for c in actual.coefficients] == [(c.term, c.equation) for c in expected.coefficients]
    fields = ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]
    np.testing.assert_allclose([[getattr(c, field) for field in fields] for c in actual.coefficients],
        [[getattr(c, field) for field in fields] for c in expected.coefficients], rtol=tol, atol=tol * .1)
    np.testing.assert_allclose(actual.covariance_matrix, expected.covariance_matrix, rtol=tol, atol=tol * .1)
    for key, value in expected.metrics.items():
        assert actual.metrics[key] == pytest.approx(value, rel=tol, abs=tol * .1)
    for key, test in expected.tests.items():
        for field in ["statistic", "p_value", "df", "df2"]:
            if test.get(field) is not None:
                assert actual.tests[key][field] == pytest.approx(test[field], rel=tol, abs=tol * .1)
    for key in ["G", "residual_variance", "theta", "theta_std_error", "linear_log_likelihood"]:
        np.testing.assert_allclose(actual.extra[key], expected.extra[key], rtol=tol, atol=tol * .1)
    assert actual.extra["icc"] == pytest.approx(expected.extra["icc"], rel=tol, abs=tol * .1)
    for key in ["levels", "random_effects", "method", "covstructure"]:
        assert actual.extra[key] == expected.extra[key]
    assert (actual.nobs, actual.nobs_original, actual.dropped_rows) == (expected.nobs, expected.nobs_original, expected.dropped_rows)
    assert actual.sample_positions == [] and len(actual.predictions) <= 400
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= 17
    assert not actual.provenance["solver_diagnostics"]["group_factors"]["complete_group_observations_collected"]
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    assert restored.coefficients == actual.coefficients
    assert r"\begin{tabular}" in restored.to_latex()


@pytest.mark.parametrize("two", [False, True])
@pytest.mark.parametrize("intercept_re", [False, True])
@pytest.mark.parametrize("structure", ["independent", "identity", "exchangeable", "unstructured"])
@pytest.mark.parametrize("method,covariance", [("ml", "nonrobust"), ("ml", "robust"), ("ml", "cluster"), ("reml", "nonrobust")])
def test_all_native_structures_levels_and_random_intercept_options(two, intercept_re, structure, method, covariance):
    frame = data(two=two, intercept_re=intercept_re)
    current = spec(two=two, intercept_re=intercept_re, structure=structure, method=method, covariance=covariance)
    parity(fit_mixed(current, frame), fit_streaming_mixed(current, source(frame), batch_rows=17))


@pytest.mark.parametrize("method,covariance", [("ml", "nonrobust"), ("ml", "robust"), ("ml", "cluster"), ("reml", "nonrobust")])
@pytest.mark.parametrize("intercept", [False, True])
def test_weights_global_categories_missing_and_no_fixed_constant(method, covariance, intercept):
    frame = data(two=True)
    frame.loc[3, "x"] = np.nan
    frame.loc[27, "y"] = np.nan
    frame.loc[49, "f"] = 0
    current = spec(two=True, structure="unstructured", method=method, covariance=covariance,
                   weights=True, categories=True, intercept=intercept).model_copy(update={"missing": "drop"})
    parity(fit_mixed(current, frame), fit_streaming_mixed(current, source(frame), batch_rows=17))


@pytest.mark.parametrize("method,covariance", [("ml", "nonrobust"), ("ml", "robust"), ("ml", "cluster"), ("reml", "nonrobust")])
def test_two_level_random_intercepts_reduce_to_native_variance_and_icc(method, covariance):
    frame = data(two=True)
    frame["y"] = 1 + .6 * frame.x - .4 * frame.w + np.random.default_rng(72).normal(size=18)[frame.s] + np.random.default_rng(71).normal(size=36)[frame.s * 2 + frame.g] + np.random.default_rng(70).normal(0, .6, len(frame))
    current = spec(two=True, method=method, covariance=covariance).model_copy(update={"columns": {"group": ["s", "g"]}})
    parity(fit_mixed(current, frame), fit_streaming_mixed(current, source(frame), batch_rows=17))


@pytest.mark.parametrize("two,reml", [(False, False), (False, True), (True, False), (True, True)])
def test_independent_explicit_v_likelihood_and_analytic_gradient(two, reml):
    frame = data(two=two, top=5, classes=2, size=5)
    x = np.column_stack((np.ones(len(frame)), frame.x, frame.w))
    z = np.column_stack((frame.x, np.ones(len(frame))))
    y = frame.y.to_numpy()
    values = torch.tensor(np.column_stack((z, np.ones(len(frame)), x, y)), dtype=torch.float64)
    keys = [str((int(s), int(g))).encode() for s, g in zip(frame.s, frame.g, strict=True)]
    parents = [str(int(s if two else g)).encode() for s, g in zip(frame.s, frame.g, strict=True)]
    theta = np.array([-.3, -.1, .08, *([.2] if two else []), -.4])
    with torch.no_grad(), torch.device("cpu"):
        store = _Factors(values.shape[1], 3)
        try:
            for start in range(0, len(frame), 7):
                store.add(keys[start:start + 7], parents[start:start + 7], [], values[start:start + 7], torch.ones(min(7, len(frame) - start), dtype=torch.float64))
            like = _GeneralLikelihood(SimpleNamespace(nobs=len(frame), rows=3), store, CovStructure("unstructured", 2), reml, two)
            def oracle(t):
                lower = np.array([[np.exp(t[0]), 0], [t[2], np.exp(t[1])]])
                g = lower @ lower.T
                same_low = np.array(keys)[:, None] == np.array(keys)[None, :]
                v = np.exp(2 * t[-1]) * np.eye(len(frame)) + (z @ g @ z.T) * same_low
                if two:
                    v += np.exp(2 * t[-2]) * (frame.s.to_numpy()[:, None] == frame.s.to_numpy()[None, :])
                px, py = np.linalg.solve(v, x), np.linalg.solve(v, y)
                gram = x.T @ px
                b = np.linalg.solve(gram, x.T @ py)
                r = y - x @ b
                ll = -.5 * ((len(frame) - (3 if reml else 0)) * math.log(2 * math.pi) + np.linalg.slogdet(v)[1] + r @ np.linalg.solve(v, r))
                if reml:
                    ll -= .5 * np.linalg.slogdet(gram)[1]
                return ll, b, gram
            out = like.evaluate(torch.tensor(theta), gradient=True)
            ll, beta, gram = oracle(theta)
            assert float(out.value) == pytest.approx(ll, rel=1e-11, abs=1e-11)
            np.testing.assert_allclose(out.beta, beta, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(out.xvx / out.sigma2, gram, rtol=1e-11, atol=1e-11)
            step = 2e-5
            gradient = np.array([(oracle(theta + step * unit)[0] - oracle(theta - step * unit)[0]) / (2 * step) for unit in np.eye(len(theta))])
            np.testing.assert_allclose(out.gradient, gradient, rtol=2e-8, atol=2e-8)
        finally:
            store.close()


def test_batched_group_qr_includes_non_power_of_two_fragment_and_replay():
    rng = np.random.default_rng(394)
    values = torch.tensor(rng.normal(size=(37, 6)), dtype=torch.float64)
    keys = [b"A"] * 5 + [b"B"] * 17 + [b"C"] * 15
    weights = torch.tensor(rng.integers(1, 4, len(keys)), dtype=torch.float64)
    store = _Factors(6, 2)
    try:
        for start in range(0, len(keys), 11):
            store.add(keys[start:start + 11], keys[start:start + 11], [], values[start:start + 11], weights[start:start + 11])
        for records, factors in store.batches():
            for record, factor in zip(records, factors, strict=True):
                mask = np.array(keys) == record[0]
                block = values[mask] * weights[mask].sqrt()[:, None]
                torch.testing.assert_close(factor.T @ factor, block.T @ block, rtol=1e-13, atol=1e-13)
    finally:
        store.close()


def test_general_worker_has_no_external_estimator_or_collection_and_restores_meta(monkeypatch):
    frame = data(two=True, top=8, classes=2, size=41)
    current = spec(two=True, structure="unstructured")
    expected = fit_mixed(current, frame)
    src = source(frame)
    monkeypatch.setattr(src, "collect", lambda *args, **kwargs: pytest.fail("replay collected source"), raising=False)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels", "linearmodels", "sklearn"}:
            raise AssertionError("External numerical routine imported")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with torch.device("meta"):
        actual = fit_streaming_mixed(current, src, batch_rows=17)
        assert torch.empty(0).device.type == "meta"
    parity(expected, actual)


def test_joint_structural_buffers_refuse_before_any_group_qr_fetch(monkeypatch):
    frame, current = data(two=True), spec(two=True, structure="unstructured")
    monkeypatch.setattr(_Factors, "add", lambda *args, **kwargs: pytest.fail("Group factor allocated before plan"))
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(current, source(frame), batch_rows=17)
    assert error.value.code == "workspace_limit"


def test_random_only_geometry_refused_before_structure_metadata_or_source(monkeypatch):
    import openecon.econometrics.streaming_mixed_general as native
    current = spec().model_copy(update={"columns": {"group": ["g"], "random": [f"z{index}" for index in range(1000)]}})
    src = source(data())
    monkeypatch.setattr(native, "CovStructure", lambda *args: pytest.fail("Random-structure metadata allocated before plan"))
    monkeypatch.setattr(src, "iter_batches", lambda *args, **kwargs: pytest.fail("Source accessed before structural plan"))
    with use_workspace_budget(8), pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(current, src, batch_rows=17)
    assert error.value.code == "workspace_limit"


def test_minimum_random_geometry_does_not_treat_collinear_raw_fixed_width_as_retained():
    frame = data()
    duplicates = {f"copy{index}": frame.x * (index + 1) for index in range(270)}
    extended = pd.concat([frame, pd.DataFrame(duplicates)], axis=1)
    current = spec().model_copy(update={"predictors": ["x", "w", *duplicates]})
    expected = fit_mixed(spec(), frame)
    with use_workspace_budget(128):
        actual = fit_streaming_mixed(current, source(extended), batch_rows=17)
    parity(expected, actual)
    assert len(actual.provenance["omitted_terms"]) == 270


def test_factor_constructor_storage_failure_removes_owned_directory(tmp_path, monkeypatch):
    import openecon.econometrics.streaming_mixed_general as native
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    def fail(*args, **kwargs):
        raise OSError("Disposable storage fixture")
    monkeypatch.setattr(native.sqlite3, "connect", fail)
    with pytest.raises(OSError):
        _Factors(4, 3)
    assert not list(tmp_path.iterdir())


def test_original_unit_log_conversion_avoids_intermediate_square_overflow():
    large = math.log(1e200)
    torch.testing.assert_close(_coordinate_scales([2 * large - large - large]), torch.ones(1, dtype=torch.float64))
    for value in [-1000., 1000.]:
        with pytest.raises(AnalysisError, match="precision") as error:
            _coordinate_scales([value])
        assert error.value.code == "mixed_precision"


def test_invalid_trial_gls_is_rejected_without_aborting_native_line_search(monkeypatch):
    import openecon.econometrics.streaming_mixed_general as native
    like = _GeneralLikelihood(SimpleNamespace(nobs=10, rows=3), SimpleNamespace(width=4), CovStructure("independent", 1), False, False)
    monkeypatch.setattr(like, "_top_factors", lambda theta: (torch.eye(2, dtype=torch.float64), torch.zeros((), dtype=torch.float64)))
    def fail(*args):
        raise AnalysisError("singular_design", "Disposable invalid optimizer trial")
    monkeypatch.setattr(native, "_solve", fail)
    theta = torch.zeros(2, dtype=torch.float64)
    assert float(like.value(theta)) == -math.inf
    value, gradient = like(theta)
    assert float(value) == -math.inf
    torch.testing.assert_close(gradient, torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("failure", ["collinear", "cluster_crosses_top", "robust_reml", "no_random", "three_levels"])
def test_general_native_domains_and_cleanup(failure, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, current = data(two=True), spec(two=True)
    if failure == "collinear":
        frame.x = 1
        code = "collinear_random_effects"
    elif failure == "cluster_crosses_top":
        frame.cluster = np.arange(len(frame)) % 4
        current = spec(two=True, covariance="cluster")
        code = "cluster_not_nested"
    elif failure == "robust_reml":
        current = spec(two=True, method="reml", covariance="robust")
        code = "unsupported_covariance"
    elif failure == "no_random":
        current = current.model_copy(update={"columns": {"group": ["s", "g"]}, "options": {"random_intercept": False}})
        code = "invalid_spec"
    else:
        current = current.model_copy(update={"columns": {"group": ["cluster", "s", "g"]}})
        code = "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(current, source(frame), batch_rows=17)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_changed_source_after_variance_optimization_still_refused(tmp_path, monkeypatch):
    import openecon.econometrics.streaming_mixed_general as native
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, changed = data(two=True), False
    original = native._start
    def arm(like, z2):
        nonlocal changed
        theta = original(like, z2)
        changed = True
        return theta
    monkeypatch.setattr(native, "_start", arm)
    def chunks():
        copy = frame.copy()
        if changed:
            copy.loc[4, "y"] += 1
        for start in range(0, len(copy), 29):
            yield copy.iloc[start:start + 29].copy()
    src = Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_mixed(spec(two=True, structure="unstructured"), src, batch_rows=17)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("two", [False, True])
@pytest.mark.parametrize("method,covariance", [("ml", "robust"), ("reml", "nonrobust")])
def test_general_frequency_weight_matches_literal_row_replication(two, method, covariance):
    frame = data(two=two)
    weighted = fit_streaming_mixed(spec(two=two, method=method, covariance=covariance, structure="unstructured", weights=True), source(frame), batch_rows=17)
    replicated = frame.loc[frame.index.repeat(frame.f)].reset_index(drop=True)
    plain = fit_streaming_mixed(spec(two=two, method=method, covariance=covariance, structure="unstructured"), source(replicated), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in weighted.coefficients], [c.estimate for c in plain.coefficients], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(weighted.covariance_matrix, plain.covariance_matrix, rtol=2e-6, atol=2e-7)
    assert weighted.metrics["log_likelihood"] == pytest.approx(plain.metrics["log_likelihood"], rel=1e-11)


@pytest.mark.parametrize("method", ["ml", "reml"])
@pytest.mark.parametrize("structure", ["independent", "unstructured"])
def test_original_random_slope_units_and_saved_theta(method, structure):
    frame = data(intercept_re=False)
    current = spec(intercept_re=False, method=method, structure=structure)
    first = fit_streaming_mixed(current, source(frame), batch_rows=17)
    changed = frame.copy()
    unit_y, unit_x, unit_w = 1e20, 1e-12, 1e12
    changed.y *= unit_y
    changed.x *= unit_x
    changed.w *= unit_w
    second = fit_streaming_mixed(current, source(changed), batch_rows=17)
    row_scales = [unit_y, unit_y / unit_x, unit_y / unit_w,
                  unit_y**2 / unit_x**2, unit_y**2 / unit_w**2,
                  *([unit_y**2 / (unit_x * unit_w)] if structure == "unstructured" else []), unit_y**2]
    transform = np.diag(row_scales)
    np.testing.assert_allclose([c.estimate for c in second.coefficients], transform @ [c.estimate for c in first.coefficients], rtol=2e-6)
    np.testing.assert_allclose(second.covariance_matrix, transform @ np.array(first.covariance_matrix) @ transform.T, rtol=3e-6)
    expected = first.metrics["log_likelihood"] - len(frame) * math.log(unit_y)
    if method == "reml":
        expected += 3 * math.log(unit_y) - math.log(unit_x) - math.log(unit_w)
    assert second.metrics["log_likelihood"] == pytest.approx(expected, rel=1e-11, abs=1e-8)
    from openecon.econometrics.mixed.predict import mixed_predict
    restored = ResultBundle.model_validate_json(second.model_dump_json())
    actual_prediction = mixed_predict(restored, changed.assign(y=np.nan), kind="xb")
    expected_prediction = mixed_predict(second, changed.assign(y=np.nan), kind="xb")
    np.testing.assert_allclose(actual_prediction, expected_prediction, rtol=1e-12)
