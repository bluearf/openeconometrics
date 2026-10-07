"""Independent OLS/covariance oracles and bounded-streaming regression checks."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import pytest
import statsmodels.api as sm
import torch

from openecon.engines.contracts import KernelError
from openecon.engines import streaming_ols, torch_engine


@pytest.fixture(scope="module", autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(931)
    x = torch.cat((torch.ones((241, 1), dtype=torch.float64),
                   torch.randn((241, 3), generator=generator, dtype=torch.float64)), dim=1)
    parameters = torch.tensor([0.8, 2.0, -0.7, 0.3], dtype=torch.float64)
    y = x @ parameters + torch.randn(241, generator=generator, dtype=torch.float64) * (0.2 + x[:, 1].square())
    return x, y


def batches(x, y, size) -> Callable:
    return lambda: ((x[start:start + size], y[start:start + size])
                    for start in range(0, len(x), size))


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3"])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("batch_rows", [1, 3, 47, 512])
def test_matches_independent_statsmodels_oracle(inputs, covariance, intercept, batch_rows):
    x, y = inputs
    if not intercept:
        x = x[:, 1:]
    reference = sm.OLS(y.numpy(), x.numpy(), hasconst=intercept).fit(cov_type=covariance)
    result = streaming_ols.solve_ols(batches(x, y, batch_rows), covariance, intercept=intercept)
    np.testing.assert_allclose(result.parameters, reference.params, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(result.covariance, reference.cov_params(), rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(result.fitted, reference.fittedvalues, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(result.residuals, reference.resid, rtol=2e-11, atol=2e-11)
    assert result.residual_ss == pytest.approx(reference.ssr, rel=3e-12)
    total_ss = reference.centered_tss if intercept else reference.uncentered_tss
    assert result.total_ss == pytest.approx(total_ss, rel=3e-12)
    assert result.log_likelihood == pytest.approx(reference.llf, rel=3e-12)
    assert type(result.nobs) is int and result.nobs == len(x)
    assert result.nparams == x.shape[1]
    assert result.solver == "torch_tsqr"
    assert result.diagnostics["dtype"] == "float64"
    assert result.diagnostics["retains_full_q"] is False


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3"])
def test_normalization_preserves_large_offsets_and_extreme_predictor_units(inputs, covariance):
    x, y = inputs
    transformed = x.clone()
    transformed[:, 1] = 1e12 + x[:, 1] * 1e8
    transformed[:, 2] *= 1e-12
    result = streaming_ols.solve_ols(batches(transformed, y, 7), covariance)
    reference = torch_engine.solve("ols", x, y, covariance)
    mapping = torch.diag(torch.tensor([1.0, 1e-8, 1e12, 1.0], dtype=torch.float64))
    mapping[0, 1] = -1e4
    torch.testing.assert_close(result.parameters, mapping @ reference.parameters, rtol=2e-10, atol=3e-8)
    torch.testing.assert_close(result.covariance, mapping @ reference.covariance @ mapping.T, rtol=3e-10, atol=3e-8)
    torch.testing.assert_close(result.fitted, reference.fitted, rtol=3e-10, atol=3e-10)
    assert result.condition_number < 2


def test_predictor_moments_do_not_square_extreme_magnitudes(inputs):
    x, y = inputs
    transformed = x.clone()
    transformed[:, 1] *= 1e100
    transformed[:, 2] *= 1e-100
    result = streaming_ols.solve_ols(batches(transformed, y, 23), "HC3")
    reference = torch_engine.solve("ols", x, y, "HC3")
    mapping = torch.diag(torch.tensor([1.0, 1e-100, 1e100, 1.0], dtype=torch.float64))
    torch.testing.assert_close(result.parameters, mapping @ reference.parameters, rtol=2e-12, atol=0)
    torch.testing.assert_close(result.covariance, mapping @ reference.covariance @ mapping.T, rtol=3e-12, atol=0)
    torch.testing.assert_close(result.fitted, reference.fitted, rtol=2e-12, atol=2e-12)


def test_nearly_collinear_scaled_design_matches_qr_without_normal_equations():
    generator = torch.Generator().manual_seed(943)
    first = torch.randn(4099, generator=generator, dtype=torch.float64)
    second = first + torch.randn(4099, generator=generator, dtype=torch.float64) * 1e-6
    x = torch.stack((torch.ones_like(first), first, second), dim=1)
    y = first * 1.4 + second * 0.4 + torch.randn(4099, generator=generator, dtype=torch.float64)
    result = streaming_ols.solve_ols(batches(x, y, 31), "HC3")
    reference = torch_engine.solve("ols", x, y, "HC3")
    torch.testing.assert_close(result.parameters, reference.parameters, rtol=3e-7, atol=1e-6)
    torch.testing.assert_close(result.covariance, reference.covariance, rtol=5e-7, atol=1e-5)
    torch.testing.assert_close(result.fitted, reference.fitted[:400], rtol=1e-6, atol=1e-7)
    assert result.condition_number > 1e6
    assert "design_columns" in result.diagnostics["rank_tolerance"]
    assert "observation" not in result.diagnostics["rank_tolerance"]


@pytest.mark.parametrize("sample_limit", [0, 1, 400])
def test_three_passes_keep_only_bounded_factors_and_prediction_samples(monkeypatch, sample_limit):
    factory_calls = 0
    qr_shapes = []
    native_qr = torch.linalg.qr

    def inspect_qr(matrix, *args, **kwargs):
        qr_shapes.append(tuple(matrix.shape))
        assert matrix.shape[0] <= 79
        assert matrix.shape[1] == 5
        return native_qr(matrix, *args, **kwargs)

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        generator = torch.Generator().manual_seed(949)
        for start in range(0, 2401, 79):
            size = min(79, 2401 - start)
            x = torch.cat((torch.ones((size, 1), dtype=torch.float64),
                           torch.randn((size, 3), generator=generator, dtype=torch.float64)), dim=1)
            y = x[:, 1] * 2 + torch.randn(size, generator=generator, dtype=torch.float64)
            yield x, y

    monkeypatch.setattr(torch.linalg, "qr", inspect_qr)
    result = streaming_ols.solve_ols(factory, sample_limit=sample_limit)
    assert factory_calls == 3
    assert qr_shapes
    assert result.nobs == 2401
    assert result.diagnostics["maximum_batch_rows"] == 79
    assert result.diagnostics["retained_factor_shape"] == [5, 5]
    assert result.diagnostics["tsqr_factor_capacity"] == 64
    assert result.diagnostics["tsqr_peak_retained_factors"] <= 5
    for field in ("observed", "fitted", "residuals", "sample_positions"):
        assert getattr(result, field).shape == (sample_limit,)
    torch.testing.assert_close(result.sample_positions, torch.arange(sample_limit))
    torch.testing.assert_close(result.observed - result.fitted, result.residuals)


def test_empty_blocks_are_allowed_without_losing_order(inputs):
    x, y = inputs

    def factory():
        yield x[:0], y[:0]
        yield x[:53], y[:53]
        yield x[:0], y[:0]
        yield x[53:], y[53:]

    result = streaming_ols.solve_ols(factory)
    reference = streaming_ols.solve_ols(batches(x, y, 53))
    torch.testing.assert_close(result.parameters, reference.parameters, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(result.observed, y)


@pytest.mark.parametrize("covariance", ["HC0", "unrecognized"])
def test_unsupported_covariance_rejects_before_reading_source(covariance):
    def factory():
        raise AssertionError("Unsupported covariance must not read the source")

    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(factory, covariance)
    assert error.value.code == "unsupported_covariance"


@pytest.mark.parametrize("sample_limit", [-1, 401, True, 1.5])
def test_sample_limit_cannot_grow_with_source_rows(inputs, sample_limit):
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(*inputs, 40), sample_limit=sample_limit)
    assert error.value.code == "invalid_solver_options"


@pytest.mark.parametrize("bad", ["zero", "constant", "dependent"])
@pytest.mark.parametrize("intercept", [True, False])
def test_rank_deficient_design_rejected(inputs, bad, intercept):
    x, y = inputs
    x = x.clone()
    if bad == "zero":
        x[:, 2] = 0
    elif bad == "constant":
        x[:, 2] = 3
    else:
        x[:, 2] = x[:, 1] * 2
    if not intercept:
        x = x[:, 1:]
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(x, y, 5), intercept=intercept)
    assert error.value.code == ("singular_design" if bad == "dependent" else "constant_predictor")


@pytest.mark.parametrize("pass_changed", [2, 3])
def test_changing_observation_count_is_rejected(inputs, pass_changed):
    x, y = inputs
    passes = 0

    def factory():
        nonlocal passes
        passes += 1
        stop = len(x) - int(passes == pass_changed)
        yield x[:stop], y[:stop]

    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(factory)
    assert error.value.code == "source_changed"


def test_changing_design_width_is_rejected(inputs):
    x, y = inputs

    def factory():
        yield x[:100], y[:100]
        yield x[100:, :2], y[100:]

    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(factory)
    assert error.value.code == "source_changed"


def test_unit_leverage_hc3_rejects_without_fallback():
    x = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.float64)
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(x, y, 1))
    assert error.value.code == "undefined_hc3"
    result = streaming_ols.solve_ols(batches(x, y, 1), "HC1")
    assert torch.isfinite(result.covariance).all()


@pytest.mark.parametrize("failure,code", [
    ("nonfinite", "non_finite_values"), ("float32", "invalid_precision"),
    ("constant_y", "constant_outcome"), ("intercept", "invalid_design"),
    ("shape", "invalid_design"), ("numpy", "invalid_design"),
])
def test_invalid_batches_rejected(inputs, failure, code):
    x, y = (value.clone() for value in inputs)
    if failure == "nonfinite":
        x[0, 1] = float("nan")
    elif failure == "float32":
        x = x.float()
    elif failure == "constant_y":
        y.fill_(3)
    elif failure == "intercept":
        x[:, 0] = 2
    elif failure == "shape":
        y = y[:, None]
    elif failure == "numpy":
        x = x.numpy()
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(x, y, 53))
    assert error.value.code == code


def test_too_few_observations_rejected(inputs):
    x, y = inputs
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(x[:4], y[:4], 3))
    assert error.value.code == "insufficient_observations"


def test_empty_stream_rejected():
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(lambda: iter(()))
    assert error.value.code == "insufficient_observations"


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3"])
@pytest.mark.parametrize("batch_rows", [1, 3, 4])
@pytest.mark.parametrize("outcome_scale", [1e-100, 1.0, 1e100])
@pytest.mark.parametrize("intercept", [True, False])
def test_zero_residual_variance_rejected(covariance, batch_rows, outcome_scale, intercept):
    # An exact fit need not produce literal zero: QR roundoff varies across
    # BLAS implementations and TSQR partitions. Batch size three also exposes
    # the defect on macOS, where the former single-block example was exact.
    x = torch.tensor([[1, -1], [1, -1], [1, 1], [1, 1]], dtype=torch.float64)
    y = x[:, 1] * outcome_scale
    if not intercept:
        x = x[:, 1:]
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(x, y, batch_rows), covariance, intercept=intercept)
    assert error.value.code == "numerical_failure"


@pytest.mark.parametrize("batch_rows", [1, 3, 4])
@pytest.mark.parametrize("outcome_scale", [1e-100, 1.0, 1e100])
def test_tiny_identifiable_residual_variance_is_not_rounded_to_zero(batch_rows, outcome_scale):
    x = torch.tensor([[1, -1], [1, -1], [1, 1], [1, 1]], dtype=torch.float64)
    # This alternating noise is orthogonal to both columns. Its exact SSR is
    # 4 * (scale * 1e-10)**2, despite a variance ratio of only 1e-20.
    noise = torch.tensor([1, -1, 1, -1], dtype=torch.float64) * 1e-10
    y = (x[:, 1] + noise) * outcome_scale
    result = streaming_ols.solve_ols(batches(x, y, batch_rows), "nonrobust")
    assert result.residual_ss == pytest.approx(4 * (outcome_scale * 1e-10) ** 2, rel=1e-5, abs=0)
    assert math.isfinite(result.log_likelihood)
    assert result.diagnostics["relative_residual_norm"] > result.diagnostics["residual_tolerance_value"]


def test_residual_resolution_uses_centered_outcome_with_intercept():
    x = torch.tensor([[1, -1], [1, -1], [1, 1], [1, 1]], dtype=torch.float64)
    noise = torch.tensor([1, -1, 1, -1], dtype=torch.float64) * 2**-12
    y = 2**40 + x[:, 1] * 2**-10 + noise
    result = streaming_ols.solve_ols(batches(x, y, 3), "nonrobust")
    # All values are exactly representable; scaling by the uncentered 2**40
    # level would misclassify genuine residual variance as numerical zero.
    assert result.residual_ss == pytest.approx(4 * 2**-24, rel=2e-12, abs=0)


def test_direct_kernel_width_bound_is_independent_of_row_count():
    x = torch.ones((2, 1025), dtype=torch.float64)
    y = torch.tensor([0, 1], dtype=torch.float64)
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(x, y, 2))
    assert error.value.code == "design_too_wide"


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("batch_rows", [1, 3, 29, 512])
def test_cluster_matches_independent_statsmodels_oracle(inputs, tmp_path, intercept, batch_rows):
    x, y = inputs
    if not intercept:
        x = x[:, 1:]
    groups = np.arange(len(x)) % 31
    keys = [b"group\0" + str(group).encode() for group in groups]
    def factory():
        return ((x[start:start + batch_rows], y[start:start + batch_rows], keys[start:start + batch_rows])
                for start in range(0, len(x), batch_rows))
    reference = sm.OLS(y.numpy(), x.numpy(), hasconst=intercept).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
    result = streaming_ols.solve_ols(factory, "cluster", intercept=intercept, scratch_directory=tmp_path)
    np.testing.assert_allclose(result.parameters, reference.params, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(result.covariance, reference.cov_params(), rtol=3e-11, atol=3e-11)
    assert result.group_count == 31
    assert result.diagnostics["cluster_count"] == 31
    assert result.diagnostics["small_sample_correction"] == pytest.approx(31 / 30 * (len(x) - 1) / (len(x) - x.shape[1]))
    assert result.diagnostics["cluster_aggregation"] == "sqlite_spill"
    assert not list(tmp_path.iterdir())


def test_cluster_missing_key_pairs_fail_clearly(inputs, tmp_path):
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(batches(*inputs, 10), "cluster", scratch_directory=tmp_path)
    assert error.value.code == "invalid_clusters"
    assert not list(tmp_path.iterdir())


def test_cluster_one_group_rejects_and_cleans(inputs, tmp_path):
    x, y = inputs
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(lambda: iter([(x, y, [b"one"] * len(x))]), "cluster", scratch_directory=tmp_path)
    assert error.value.code == "insufficient_clusters"
    assert not list(tmp_path.iterdir())


def test_cluster_count_changed_on_third_pass_cleans_spill(inputs, tmp_path):
    x, y = inputs
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        stop = len(x) - int(calls == 3)
        yield x[:stop], y[:stop], [f"group-{row % 9}".encode() for row in range(stop)]
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(factory, "cluster", scratch_directory=tmp_path)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_cluster_singleton_groups_equal_hc1(inputs, tmp_path):
    x, y = inputs
    def factory():
        return iter([(x, y, [str(row).encode() for row in range(len(x))])])
    clustered = streaming_ols.solve_ols(factory, "cluster", scratch_directory=tmp_path)
    hc1 = streaming_ols.solve_ols(batches(x, y, 51), "HC1")
    torch.testing.assert_close(clustered.parameters, hc1.parameters, rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(clustered.covariance, hc1.covariance, rtol=4e-12, atol=4e-12)
    assert clustered.group_count == len(x)
    assert not list(tmp_path.iterdir())


def test_cluster_numeric_factory_skips_key_encoding_on_first_two_passes(inputs, tmp_path):
    x, y = inputs
    calls = {"numeric": 0, "cluster": 0}
    keys = [f"group-{row % 17}".encode() for row in range(len(x))]

    def numeric_factory():
        calls["numeric"] += 1
        yield from batches(x, y, 23)()

    def cluster_factory():
        calls["cluster"] += 1
        for first in range(0, len(x), 23):
            yield x[first:first + 23], y[first:first + 23], keys[first:first + 23]

    result = streaming_ols.solve_ols(cluster_factory, "cluster", scratch_directory=tmp_path,
                                     numeric_batch_factory=numeric_factory)
    assert calls == {"numeric": 2, "cluster": 1}
    assert result.diagnostics["numeric_batch_factory_used"] is True
    assert result.group_count == 17
    reference = streaming_ols.solve_ols(cluster_factory, "cluster", scratch_directory=tmp_path)
    torch.testing.assert_close(result.parameters, reference.parameters, rtol=0, atol=0)
    torch.testing.assert_close(result.covariance, reference.covariance, rtol=0, atol=0)
    assert result.diagnostics["small_sample_correction"] == reference.diagnostics["small_sample_correction"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fail_pass", ["numeric_factor", "covariance"])
def test_cluster_numeric_factory_source_failure_cleans_spill(inputs, tmp_path, fail_pass):
    x, y = inputs
    calls = 0

    def numeric_factory():
        nonlocal calls
        calls += 1
        stop = len(x) - int(fail_pass == "numeric_factor" and calls == 2)
        yield x[:stop], y[:stop]

    def cluster_factory():
        yield x[:30], y[:30], [str(row % 7).encode() for row in range(30)]
        if fail_pass == "covariance":
            raise KernelError("source_changed", "The projected source changed during replay.")
        yield x[30:], y[30:], [str(row % 7).encode() for row in range(30, len(x))]

    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(cluster_factory, "cluster", scratch_directory=tmp_path,
                                 numeric_batch_factory=numeric_factory)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_invalid_numeric_factory_fails_before_source_access(inputs):
    def inaccessible():
        raise AssertionError("The source must not be accessed for invalid options.")
    with pytest.raises(KernelError) as error:
        streaming_ols.solve_ols(inaccessible, "cluster", numeric_batch_factory=False)
    assert error.value.code == "invalid_design"
