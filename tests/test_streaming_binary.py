"""Bounded binary MLE parity, independent expectations, and failure contracts."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import expit, ndtr, ndtri
import torch

from openecon.engines.contracts import KernelError
from openecon.engines import separation, streaming_binary, torch_engine


@pytest.fixture(scope="module", autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(971)
    x = torch.cat((torch.ones((607, 1), dtype=torch.float64),
                   torch.randn((607, 3), generator=generator, dtype=torch.float64)), dim=1)
    p = torch.sigmoid(x @ torch.tensor([-0.2, 0.8, -0.5, 0.3], dtype=torch.float64))
    y = (torch.rand(607, generator=generator, dtype=torch.float64) < p).to(torch.float64)
    return x, y


def batches(x, y, size, keys=None):
    def factory():
        for start in range(0, len(x), size):
            pair = (x[start:start + size], y[start:start + size])
            yield (*pair, keys[start:start + size]) if keys is not None else pair
    return factory


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("batch_rows", [1, 7, 63, 999])
def test_matches_dense_mle_information_and_predictions(inputs, estimator, intercept, batch_rows):
    x, y = inputs
    if not intercept:
        x = x[:, 1:]
    dense = torch_engine.solve(estimator, x, y, "nonrobust", intercept=intercept)
    result = streaming_binary.solve_binary(estimator, batches(x, y, batch_rows), intercept=intercept)
    torch.testing.assert_close(result.parameters, dense.parameters, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(result.covariance, dense.covariance, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(result.fitted, dense.fitted[:400], rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(result.residuals, y[:400] - dense.fitted[:400], rtol=2e-9, atol=2e-10)
    assert result.residual_ss == pytest.approx(float((y - dense.fitted).square().sum()), rel=2e-10)
    assert result.log_likelihood == pytest.approx(dense.log_likelihood, rel=2e-12)
    prevalence = float(y.mean())
    null_ll = len(y) * (prevalence * math.log(prevalence) + (1 - prevalence) * math.log1p(-prevalence))
    assert result.null_log_likelihood == pytest.approx(null_ll, rel=2e-12)
    assert type(result.nobs) is int and result.nobs == 607
    assert result.nparams == x.shape[1]
    assert result.diagnostics["converged"] is True
    assert result.diagnostics["dtype"] == "float64"
    assert result.diagnostics["dense_observation_matrix"] is False
    assert result.diagnostics["retains_full_q"] is False


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("intercept", [True, False])
def test_independent_exact_binomial_mle_and_observed_information(estimator, intercept):
    values = np.repeat([-1., 0., 1.], 40)
    x = np.column_stack((np.ones(120), values)) if intercept else values[:, None]
    y = np.concatenate([np.r_[np.ones(count), np.zeros(40 - count)] for count in [10, 20, 30]])
    slope = math.log(3) if estimator == "logit" else ndtri(.75)
    expected = np.array([0., slope]) if intercept else np.array([slope])
    eta = x @ expected
    p = expit(eta) if estimator == "logit" else ndtr(eta)
    if estimator == "logit":
        weight = p * (1 - p)
    else:
        t = (2 * y - 1) * eta
        ratio = np.exp(-t * t / 2) / math.sqrt(2 * math.pi) / ndtr(t)
        weight = ratio * (ratio + t)
    covariance = np.linalg.inv(x.T @ (x * weight[:, None]))
    tx, ty = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    actual = streaming_binary.solve_binary(estimator, batches(tx, ty, 3), intercept=intercept)
    np.testing.assert_allclose(actual.parameters, expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(actual.fitted, p, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("intercept", [True, False])
def test_cluster_covariance_matches_dense_and_cleans_spill(inputs, estimator, intercept, tmp_path):
    x, y = inputs
    if not intercept:
        x = x[:, 1:]
    groups = torch.arange(len(x), dtype=torch.int64) % 41
    keys = [int(group).to_bytes(8, "little") for group in groups]
    dense = torch_engine.solve(estimator, x, y, "cluster", groups, intercept=intercept)
    actual = streaming_binary.solve_binary(estimator, batches(x, y, 17, keys), "cluster",
                                            intercept=intercept, scratch_directory=tmp_path)
    torch.testing.assert_close(actual.parameters, dense.parameters, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(actual.covariance, dense.covariance, rtol=2e-9, atol=2e-10)
    assert actual.group_count == 41
    assert actual.diagnostics["small_sample_correction"] == pytest.approx(41 / 40 * 606 / (607 - x.shape[1]))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_numeric_factory_handles_all_prior_passes_and_clusters_are_encoded_once(inputs, estimator, tmp_path):
    x, y = inputs
    groups = torch.arange(len(x), dtype=torch.int64) % 41
    keys = [int(group).to_bytes(8, "little") for group in groups]
    calls = []

    def numeric():
        calls.append("numeric")
        yield from batches(x, y, 7)()

    def clustered():
        calls.append("cluster")
        yield from batches(x, y, 23, keys)()

    result = streaming_binary.solve_binary(estimator, clustered, "cluster", numeric_batch_factory=numeric,
                                            scratch_directory=tmp_path)
    dense = torch_engine.solve(estimator, x, y, "cluster", groups)
    assert calls[-1] == "cluster" and calls.count("cluster") == 1
    assert all(kind == "numeric" for kind in calls[:-1])
    assert len(calls) == result.diagnostics["passes"]
    assert result.diagnostics["maximum_batch_rows"] == 23
    torch.testing.assert_close(result.parameters, dense.parameters, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(result.covariance, dense.covariance, rtol=2e-9, atol=2e-10)
    assert result.group_count == 41 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("callback", ["numeric", "cluster"])
def test_optional_factory_domain_errors_are_preserved_and_spill_is_cleaned(inputs, callback, tmp_path):
    class SourceFailure(ValueError):
        code = "source_changed"

    x, y = inputs
    expected = SourceFailure("Source changed in the selected callback")

    def numeric():
        if callback == "numeric":
            raise expected
        yield x, y

    def clustered():
        raise expected
        yield  # pragma: no cover - maintain a replayable generator callback

    with pytest.raises(SourceFailure) as caught:
        streaming_binary.solve_binary("logit", clustered, "cluster", numeric_batch_factory=numeric,
                                        scratch_directory=tmp_path)
    assert caught.value is expected and not list(tmp_path.iterdir())


def test_final_cluster_factory_count_is_checked_against_numeric_source(inputs, tmp_path):
    x, y = inputs
    keys = [str(index % 41).encode() for index in range(len(x)-1)]
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", batches(x[:-1], y[:-1], 13, keys), "cluster",
                                        numeric_batch_factory=batches(x, y, 7), scratch_directory=tmp_path)
    assert caught.value.code == "source_changed" and not list(tmp_path.iterdir())


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_extreme_units_and_high_offsets_match_original_design(inputs, estimator):
    x, y = inputs
    shifted = x.clone()
    shifted[:, 1] = 1e12 + shifted[:, 1] * 1e8
    shifted[:, 2] *= 1e-100
    baseline = torch_engine.solve(estimator, x, y, "nonrobust")
    actual = streaming_binary.solve_binary(estimator, batches(shifted, y, 23))
    mapping = torch.diag(torch.tensor([1., 1e-8, 1e100, 1.], dtype=torch.float64))
    mapping[0, 1] = -1e4
    torch.testing.assert_close(actual.parameters, mapping @ baseline.parameters, rtol=3e-9, atol=2e-8)
    torch.testing.assert_close(actual.covariance, mapping @ baseline.covariance @ mapping.T, rtol=3e-9, atol=2e-8)
    torch.testing.assert_close(actual.fitted, baseline.fitted[:400], rtol=3e-9, atol=2e-10)


@pytest.mark.parametrize("sample_limit", [0, 1, 400])
def test_all_observation_work_and_separation_constraints_are_bounded(inputs, sample_limit, monkeypatch):
    x, y = inputs
    native_terms = streaming_binary._binary_terms
    native_lp = separation.solve_box_lp
    term_shapes = []
    lp_shapes = []

    def terms(torch_module, estimator, z, outcome, theta):
        term_shapes.append(tuple(z.shape))
        assert len(z) <= 13
        return native_terms(torch_module, estimator, z, outcome, theta)

    def lp(objective, matrix, **kwargs):
        if len(matrix):
            lp_shapes.append(matrix.shape)
            assert matrix.shape[0] <= 4 * x.shape[1] and matrix.shape[1] == x.shape[1]
        return native_lp(objective, matrix, **kwargs)

    monkeypatch.setattr(streaming_binary, "_binary_terms", terms)
    monkeypatch.setattr(separation, "solve_box_lp", lp)
    result = streaming_binary.solve_binary("logit", batches(x, y, 13), sample_limit=sample_limit)
    assert term_shapes and lp_shapes
    for field in ("observed", "fitted", "residuals", "sample_positions"):
        assert getattr(result, field).shape == (sample_limit,)
    torch.testing.assert_close(result.sample_positions, torch.arange(sample_limit))
    assert result.diagnostics["maximum_batch_rows"] == 13
    assert result.diagnostics["maximum_input_batch_bytes"] == 13 * 5 * 8
    assert result.diagnostics["separation_peak_constraints"] <= 4 * x.shape[1]
    assert result.diagnostics["retained_factor_shape"] == [4, 4]


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("values,outcomes", [
    ([-2., -1., 1., 2.], [0., 0., 1., 1.]),
    ([-1., 0., 0., 1.], [0., 0., 1., 1.]),
    ([1., 2., 4., 5., 5., 6.], [0., 0., 0., 0., 1., 1.]),
])
def test_complete_and_quasi_separation_are_explicitly_rejected(estimator, values, outcomes):
    predictor = torch.tensor(values, dtype=torch.float64)
    x = torch.stack((torch.ones_like(predictor), predictor), dim=1)
    y = torch.tensor(outcomes, dtype=torch.float64)
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary(estimator, batches(x, y, 1))
    assert caught.value.code == "separation_detected"


def test_bounded_diagnostic_rejects_quasi_separation_not_seen_by_newton_direction():
    # Finite nuisance/intercept components make a Newton step fail the global
    # direction certificate; the constraint program still finds the recession.
    generator = torch.Generator().manual_seed(983)
    x = torch.cat((torch.ones((202, 1), dtype=torch.float64),
                   torch.randn((202, 2), generator=generator, dtype=torch.float64)), dim=1)
    x[:200, 1] = torch.arange(200, dtype=torch.float64) - 100
    x[200:, 1] = 0
    y = (x[:, 1] > 0).to(torch.float64)
    y[200:] = torch.tensor([0., 1.], dtype=torch.float64)
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", batches(x, y, 11), max_iter=1)
    assert caught.value.code == "separation_detected"


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("intercept", [True, False])
def test_multivariate_complete_separation_with_binding_lp_face(estimator, intercept):
    generator = torch.Generator().manual_seed(1001)
    predictors = torch.randn((409, 2), generator=generator, dtype=torch.float64)
    y = (predictors[:, 0] + 2 * predictors[:, 1] > (.2 if intercept else 0)).to(torch.float64)
    x = torch.cat((torch.ones((409, 1), dtype=torch.float64), predictors), dim=1) if intercept else predictors
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary(estimator, batches(x, y, 29), intercept=intercept, max_iter=1)
    assert caught.value.code == "separation_detected"


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_rare_classes_and_extreme_tail_rows_have_a_finite_fit(estimator):
    predictor = torch.cat((torch.zeros(2000, dtype=torch.float64),
                           torch.tensor([-1e5, -40., 40., 1e5], dtype=torch.float64)))
    x = torch.stack((torch.ones_like(predictor), predictor), dim=1)
    y = torch.zeros_like(predictor)
    y[:4] = 1
    y[-4:] = torch.tensor([0., 1., 0., 1.], dtype=torch.float64)
    dense = torch_engine.solve(estimator, x, y, "nonrobust")
    actual = streaming_binary.solve_binary(estimator, batches(x, y, 31))
    torch.testing.assert_close(actual.parameters, dense.parameters, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(actual.covariance, dense.covariance, rtol=2e-9, atol=2e-10)
    assert bool(torch.isfinite(actual.covariance).all())


@pytest.mark.parametrize("bad", ["all-zero", "all-one", "not-binary"])
def test_invalid_outcomes_rejected(inputs, bad):
    x, y = inputs
    y = torch.zeros_like(y) if bad == "all-zero" else torch.ones_like(y)
    if bad == "not-binary":
        y[12] = 2
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("probit", batches(x, y, 29))
    assert caught.value.code == "invalid_binary_outcome"


@pytest.mark.parametrize("bad,code", [("zero", "constant_predictor"), ("constant", "constant_predictor"),
                                      ("dependent", "singular_design"), ("nonfinite", "non_finite_values")])
def test_invalid_and_rank_deficient_design(inputs, bad, code):
    x, y = inputs
    x = x.clone()
    if bad == "zero":
        x[:, 2] = 0
    elif bad == "constant":
        x[:, 2] = 3
    elif bad == "dependent":
        x[:, 2] = x[:, 1] * 2
    else:
        x[0, 2] = math.inf
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", batches(x, y, 7))
    assert caught.value.code == code


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_no_intercept_rejects_constant_predictors_like_public_dense_api(estimator):
    x = torch.full((40, 1), 3., dtype=torch.float64)
    y = torch.cat((torch.ones(10, dtype=torch.float64), torch.zeros(30, dtype=torch.float64)))
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary(estimator, batches(x, y, 3), intercept=False)
    assert caught.value.code == "constant_predictor"


@pytest.mark.parametrize("changed_pass", [2, 3, 4])
def test_changed_source_count_is_rejected(inputs, changed_pass):
    x, y = inputs
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        stop = len(x) - int(calls == changed_pass)
        yield x[:stop], y[:stop]

    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", factory)
    assert caught.value.code == "source_changed"


def test_changed_source_width_and_one_shot_iterator_are_rejected(inputs):
    x, y = inputs

    def wrong_width():
        yield x[:20], y[:20]
        yield x[20:, :2], y[20:]

    iterator = iter([(x, y)])
    for factory in (wrong_width, lambda: iterator):
        with pytest.raises(KernelError) as caught:
            streaming_binary.solve_binary("logit", factory)
        assert caught.value.code == "source_changed"


@pytest.mark.parametrize("failed_pass", [1, 2, 3, 4])
def test_callback_domain_exception_keeps_its_identity_and_code(inputs, failed_pass):
    class SourceFailure(ValueError):
        code = "source_changed"

    x, y = inputs
    calls = 0
    expected = SourceFailure("Source changed while replaying the batch")

    def factory():
        nonlocal calls
        calls += 1
        if calls == failed_pass:
            raise expected
        yield x, y

    with pytest.raises(SourceFailure) as caught:
        streaming_binary.solve_binary("logit", factory)
    assert caught.value is expected and caught.value.code == "source_changed"


@pytest.mark.parametrize("options", [{"max_iter": 0}, {"max_iter": True}, {"tolerance": math.nan},
                                    {"tolerance": 0}, {"sample_limit": 401}, {"sample_limit": True}])
def test_invalid_options_fail_before_source_access(options):
    def inaccessible():
        raise AssertionError("Invalid options must not read the source")
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", inaccessible, **options)
    assert caught.value.code == "invalid_solver_options"


@pytest.mark.parametrize("estimator,covariance,code", [("ols", "nonrobust", "unsupported_estimator"),
                                                       ("logit", "HC3", "unsupported_covariance")])
def test_unsupported_options_fail_before_source_access(estimator, covariance, code):
    def inaccessible():
        raise AssertionError("Unsupported options must not read the source")
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary(estimator, inaccessible, covariance)
    assert caught.value.code == code


def test_nonconvergence_does_not_return_success(inputs):
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", batches(*inputs, 31), max_iter=1)
    assert caught.value.code == "nonconvergence"


@pytest.mark.parametrize("mode", ["missing", "wrong-length", "wrong-type", "one-group"])
def test_invalid_cluster_contracts_rejected(inputs, mode):
    x, y = inputs
    keys = None if mode == "missing" else [b"a"] * len(x)
    if mode == "wrong-length":
        keys = keys[:-1]
    elif mode == "wrong-type":
        keys[0] = "a"
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", batches(x, y, len(x), keys), "cluster")
    assert caught.value.code == ("insufficient_clusters" if mode == "one-group" else "invalid_clusters")


def test_inconclusive_separation_diagnostic_never_accepts_a_fit(inputs, monkeypatch):
    def failed(*args, **kwargs):
        raise KernelError("separation_check_failed", "Injected native LP precision failure")
    monkeypatch.setattr(separation, "solve_box_lp", failed)
    with pytest.raises(KernelError) as caught:
        streaming_binary.solve_binary("logit", batches(*inputs, 31))
    assert caught.value.code == "separation_check_failed"
