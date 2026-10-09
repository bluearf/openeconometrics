"""Independent algebra and numerical boundaries for fixed-X spatial diagnostics."""

import math

import numpy as np
import pytest
from scipy import stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.spatial.diagnostic_kernels import TESTS, diagnose, resource_estimates
from openecon.resources import use_workspace_budget


@pytest.fixture
def geometry():
    rng = np.random.default_rng(943)
    n = 48
    x = np.column_stack((np.ones(n), rng.normal(size=n), rng.uniform(-2, 2, size=n)))
    w = np.zeros((n, n))
    for i in range(n - 1):
        w[i, (i + 1) % n] = 0.7
        w[i, (i + 7) % n] = 0.3
    # Last unit is an explicit zero-row isolate. W remains asymmetric.
    innovation = rng.normal(size=n)
    y = 1.2 + x[:, 1] * 0.9 - x[:, 2] * 0.55 + innovation + 0.4 * (w @ innovation)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return x, y, beta, w


def run(geometry, **options):
    x, y, beta, w = geometry
    tensors = [torch.tensor(value, dtype=torch.float64, device="cpu") for value in (x, y, beta, w)]
    return diagnose(*tensors, wx_indices=[1, 2], draws=199, seed=43, **options)


def oracle(geometry):
    x, y, beta, w = geometry
    n, p = x.shape
    e = y - x @ beta
    mu = x @ beta
    q = np.linalg.qr(x, mode="complete")[0]
    residual_basis = q[:, p:]
    m = residual_basis @ residual_basis.T
    s = (w + w.T) / 2
    c = n / w.sum()
    reduced = residual_basis.T @ s @ residual_basis
    trace = np.trace(reduced)
    nu = n - p
    expected = c * trace / nu
    variance = c**2 * 2 * (np.sum(reduced**2) - trace**2 / nu) / (nu * (nu + 2))
    observed = c * (e @ w @ e) / (e @ e)
    s2 = e @ e / n
    a, b = e @ w @ e / s2, e @ w @ y / s2
    t = np.trace(w @ w + w.T @ w)
    h = (m @ w @ mu) @ (m @ w @ mu) / s2
    d = t + h
    lm = {
        "lm_error": a**2 / t,
        "lm_lag": b**2 / d,
        "robust_lm_error": (a - t * b / d) ** 2 / (t * h / d),
        "robust_lm_lag": (b - a) ** 2 / h,
        "lm_joint": np.array([b, a]) @ np.linalg.solve(np.array([[d, t], [t, t]]), [b, a]),
    }
    augmented = np.column_stack((x, w @ x[:, 1:]))
    augmented_beta = np.linalg.lstsq(augmented, y, rcond=None)[0]
    rss1 = np.sum((y - augmented @ augmented_beta) ** 2)
    df2 = n - augmented.shape[1]
    f = ((e @ e - rss1) / 2) / (rss1 / df2)
    return {
        "moran": observed,
        "expected": expected,
        "variance": variance,
        "scores": np.array([b, a]),
        "information": np.array([[d, t], [t, t]]),
        "lm": lm,
        "wx_f": f,
        "df2": df2,
        "residual": e,
        "m": m,
    }


def test_all_eight_targets_match_independent_dense_and_residual_basis_oracles(geometry):
    answer, reference = run(geometry), oracle(geometry)
    rows = {row["test"]: row for row in answer["tests"]}
    assert tuple(rows) == TESTS
    np.testing.assert_allclose(answer["scores"], reference["scores"], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        answer["information"], reference["information"], rtol=2e-13, atol=2e-13
    )
    for name in ("moran_normal", "moran_gaussian_mc"):
        assert rows[name]["statistic"] == pytest.approx(reference["moran"], rel=2e-13, abs=2e-13)
        assert rows[name]["expected"] == pytest.approx(reference["expected"], rel=2e-13, abs=2e-13)
        assert rows[name]["variance"] == pytest.approx(reference["variance"], rel=2e-13)
    for name, value in reference["lm"].items():
        assert rows[name]["statistic"] == pytest.approx(value, rel=5e-12)
        df = 2 if name == "lm_joint" else 1
        assert rows[name]["p_value"] == pytest.approx(stats.chi2.sf(value, df), abs=2e-13)
    assert rows["wx_f"]["statistic"] == pytest.approx(reference["wx_f"], rel=1e-12)
    assert rows["wx_f"]["df_denom"] == reference["df2"]
    assert rows["wx_f"]["p_value"] == pytest.approx(
        stats.f.sf(reference["wx_f"], 2, reference["df2"]), abs=2e-13
    )


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_moran_normal_tails_use_projected_expectation(geometry, alternative):
    row = run(geometry, tests=["moran_normal"], alternative=alternative)["tests"][0]
    ref = oracle(geometry)
    z = (ref["moran"] - ref["expected"]) / math.sqrt(ref["variance"])
    probability = {
        "two-sided": 2 * stats.norm.sf(abs(z)),
        "greater": stats.norm.sf(z),
        "less": stats.norm.cdf(z),
    }[alternative]
    assert row["z"] == pytest.approx(z, abs=2e-13)
    assert row["p_value"] == pytest.approx(probability, abs=2e-13)
    assert abs(ref["expected"] + 1 / (len(geometry[1]) - 1)) > 1e-4


def test_scores_match_two_full_sac_gaussian_likelihood_derivatives(geometry):
    x, y, beta, w = geometry
    s2 = np.sum((y - x @ beta) ** 2) / len(y)

    def loglik(rho, lam):
        a, b = np.eye(len(y)) - rho * w, np.eye(len(y)) - lam * w
        innovation = b @ (a @ y - x @ beta)
        return (
            np.linalg.slogdet(a)[1] + np.linalg.slogdet(b)[1] - innovation @ innovation / (2 * s2)
        )

    h = 1e-5
    derivative = np.array(
        [(loglik(h, 0) - loglik(-h, 0)) / (2 * h), (loglik(0, h) - loglik(0, -h)) / (2 * h)]
    )
    answer = run(geometry, tests=["lm_joint"])
    np.testing.assert_allclose(answer["scores"], derivative, rtol=2e-9, atol=2e-9)
    assert answer["geometry"]["score_variance_divisor"] == "SSE/n"
    assert "not heteroskedasticity robust" in answer["geometry"]["adjusted_semantics"]


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_complete_mc_draw_sequence_projection_tails_and_global_rng(geometry, alternative):
    before = torch.random.get_rng_state().clone()
    answer = run(geometry, tests=["moran_gaussian_mc"], alternative=alternative)
    assert torch.equal(before, torch.random.get_rng_state())
    x, y, beta, w = geometry
    q = answer["q"].numpy()
    generator = torch.Generator(device="cpu").manual_seed(43)
    expected_null = []
    for _ in range(199):
        z = torch.randn((len(y),), generator=generator, dtype=torch.float64).numpy()
        e = z - q @ (q.T @ z)
        expected_null.append(len(y) / w.sum() * (e @ w @ e) / (e @ e))
    np.testing.assert_allclose(answer["null_statistics"], expected_null, rtol=2e-13, atol=2e-13)
    row = answer["tests"][0]
    null = np.array(expected_null)
    tolerance = row["tie_tolerance"]
    if alternative == "greater":
        extreme = np.sum(null >= row["statistic"] - tolerance)
    elif alternative == "less":
        extreme = np.sum(null <= row["statistic"] + tolerance)
    else:
        extreme = np.sum(
            abs(null - row["expected"]) >= abs(row["statistic"] - row["expected"]) - tolerance
        )
    assert row["extreme_count"] == extreme
    assert row["p_value"] == (extreme + 1) / 200
    assert 0 < row["p_value"] <= 1
    assert "one independent torch.randn" in row["rng_policy"]
    assert torch.equal(
        answer["null_statistics"],
        run(geometry, tests=["moran_gaussian_mc"], alternative=alternative)["null_statistics"],
    )


def test_independent_gaussian_null_directions_recover_analytic_moments(geometry):
    ref = oracle(geometry)
    rng = np.random.default_rng(1893)
    n = len(geometry[1])
    residuals = rng.normal(size=(30_000, n)) @ ref["m"]
    w = geometry[3]
    values = (
        n
        / w.sum()
        * np.einsum("bi,ij,bj->b", residuals, w, residuals)
        / np.sum(residuals**2, axis=1)
    )
    assert np.mean(values) == pytest.approx(ref["expected"], abs=0.0025)
    assert np.var(values) == pytest.approx(ref["variance"], rel=0.025)


def test_same_column_space_permuted_terms_and_rows_preserve_algebra(geometry):
    x, y, beta, w = geometry
    baseline = run(geometry, tests=[name for name in TESTS if name != "moran_gaussian_mc"])
    term_order = [2, 0, 1]
    row_order = np.random.default_rng(38).permutation(len(y))
    transformed = (
        x[row_order][:, term_order],
        y[row_order],
        beta[term_order],
        w[np.ix_(row_order, row_order)],
    )
    tensors = [torch.tensor(v, dtype=torch.float64) for v in transformed]
    answer = diagnose(
        *tensors, tests=[name for name in TESTS if name != "moran_gaussian_mc"], wx_indices=[0, 2]
    )
    np.testing.assert_allclose(
        [r["statistic"] for r in answer["tests"]],
        [r["statistic"] for r in baseline["tests"]],
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(answer["information"], baseline["information"], rtol=2e-13)


@pytest.mark.parametrize("factor", [1e-120, -4.0, 1e120])
def test_multiplicative_outcome_units_leave_all_statistics_invariant(geometry, factor):
    x, y, beta, w = geometry
    first = run(geometry)
    second = run((x, factor * y, factor * beta, w))
    np.testing.assert_allclose(
        [r["statistic"] for r in second["tests"]],
        [r["statistic"] for r in first["tests"]],
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(second["information"], first["information"], rtol=3e-12)


def test_added_design_signal_changes_lag_tests_but_not_residual_targets(geometry):
    x, y, beta, w = geometry
    shift = np.array([3.0, 1.25, -0.75])
    transformed = (x, 4 * y + x @ shift, 4 * beta + shift, w)
    first, second = run(geometry), run(transformed)
    a = {r["test"]: r for r in first["tests"]}
    b = {r["test"]: r for r in second["tests"]}
    for name in ("moran_normal", "moran_gaussian_mc", "lm_error", "wx_f"):
        assert b[name]["statistic"] == pytest.approx(a[name]["statistic"], abs=3e-12)
    assert abs(a["lm_lag"]["statistic"] - b["lm_lag"]["statistic"]) > 0.01
    transformed_ref = oracle(transformed)
    for name, statistic in transformed_ref["lm"].items():
        assert b[name]["statistic"] == pytest.approx(statistic, rel=3e-12)


@pytest.mark.parametrize("factor", [1e-100, 2.0, 1e100])
def test_declared_graph_global_scale_preserves_tests_and_scales_full_scores(geometry, factor):
    x, y, beta, w = geometry
    first, second = run(geometry), run((x, y, beta, w * factor))
    np.testing.assert_allclose(
        [r["statistic"] for r in second["tests"]],
        [r["statistic"] for r in first["tests"]],
        rtol=3e-12,
        atol=3e-12,
    )
    np.testing.assert_allclose(second["scores"] / factor, first["scores"], rtol=3e-12)
    np.testing.assert_allclose(
        second["information"] / factor / factor, first["information"], rtol=3e-12
    )


def test_intercept_only_moran_recovers_raw_normality_moments_and_refuses_joint(geometry):
    _, y, _, w = geometry
    n = len(y)
    x, beta = np.ones((n, 1)), np.array([y.mean()])
    weights = w.copy()
    weights[-1, 0] = 1
    row = diagnose(
        *(torch.tensor(v, dtype=torch.float64) for v in (x, y, beta, weights)),
        tests=["moran_normal"],
    )["tests"][0]
    s0 = weights.sum()
    s1 = 0.5 * np.sum((weights + weights.T) ** 2)
    s2 = np.sum((weights.sum(0) + weights.sum(1)) ** 2)
    expected = -1 / (n - 1)
    variance = (n**2 * s1 - n * s2 + 3 * s0**2) / ((n**2 - 1) * s0**2) - expected**2
    assert row["expected"] == pytest.approx(expected, abs=2e-14)
    assert row["variance"] == pytest.approx(variance, rel=2e-13)
    tensors = [torch.tensor(v, dtype=torch.float64) for v in (x, y, beta, weights)]
    with pytest.raises(AnalysisError, match="cannot be separately resolved"):
        diagnose(*tensors, tests=["lm_joint"])
    assert len(diagnose(*tensors, tests=["lm_error", "lm_lag"])["tests"]) == 2


@pytest.mark.parametrize("target", ["moran_normal", "moran_gaussian_mc"])
def test_complete_graph_degenerate_projected_moran_refuses_requested_target(geometry, target):
    x, y, beta, _ = geometry
    w = np.ones((len(y), len(y))) - np.eye(len(y))
    with pytest.raises(AnalysisError) as exc:
        run((x, y, beta, w), tests=[target])
    assert exc.value.code == f"spatial_{target}_variance"


def test_wx_requested_directions_cannot_be_silently_reduced(geometry):
    x, y, beta, _ = geometry
    n = len(y)
    # W maps every vector to its mean minus self/n. With an intercept its
    # WX columns are already in X, so a two-direction added test is undefined.
    w = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    with pytest.raises(AnalysisError) as exc:
        run((x, y, beta, w), tests=["wx_f"])
    assert exc.value.code == "spatial_wx_f_rank"
    tensors = [torch.tensor(v, dtype=torch.float64) for v in geometry]
    with pytest.raises(AnalysisError, match="intercept"):
        diagnose(*tensors, tests=["wx_f"], wx_indices=[0])


def test_saved_coefficients_are_used_and_contradictory_fit_refused(geometry):
    x, y, beta, w = geometry
    changed = beta.copy()
    changed[1] += 0.05
    with pytest.raises(AnalysisError, match="saved coefficients"):
        run((x, y, changed, w), tests=["lm_error"])
    original_beta = beta.copy()
    answer = run(geometry, tests=["lm_error"])
    np.testing.assert_array_equal(beta, original_beta)
    np.testing.assert_allclose(answer["fitted"], x @ beta, rtol=1e-15)


@pytest.mark.parametrize("kind", ["duplicate", "zero", "near_rank"])
def test_design_rank_guard_preserves_every_requested_column(geometry, kind):
    x, y, beta, w = geometry
    altered = x.copy()
    altered[:, 2] = {
        "duplicate": x[:, 1],
        "zero": np.zeros(len(y)),
        "near_rank": x[:, 1] + 1e-14 * x[:, 2],
    }[kind]
    with pytest.raises(AnalysisError) as exc:
        run((altered, y, beta, w), tests=["lm_error"])
    assert exc.value.code == "spatial_diagnostics_rank"


def test_work_preflight_precedes_qr_and_rng_allocations(geometry, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical work allocated before preflight")

    monkeypatch.setattr(torch.linalg, "qr", forbidden)
    monkeypatch.setattr(torch, "randn", forbidden)
    with pytest.raises(AnalysisError) as exc:
        run(geometry, max_work=1)
    assert exc.value.code == "spatial_diagnostics_work_limit"


def test_workspace_preflight_precedes_projection(geometry, monkeypatch):
    x, y, beta, w = geometry
    n = 128
    expanded_x = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
    expanded_y = np.sin(np.arange(n)) + expanded_x[:, 1]
    expanded_beta = np.linalg.lstsq(expanded_x, expanded_y, rcond=None)[0]
    expanded_w = np.eye(n, k=1)
    tensors = [
        torch.tensor(v, dtype=torch.float64)
        for v in (expanded_x, expanded_y, expanded_beta, expanded_w)
    ]
    monkeypatch.setattr(
        torch.linalg,
        "qr",
        lambda *args, **kwargs: pytest.fail("QR executed before workspace check"),
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        diagnose(*tensors, tests=["moran_normal"], max_work=10**9)
    assert exc.value.code == "workspace_limit"


@pytest.mark.parametrize(
    "change",
    ["dtype", "shape", "negative", "diagonal", "nonfinite", "draws", "seed", "tests", "wx_index"],
)
def test_numeric_admission_failures_are_structured(geometry, change):
    tensors = [torch.tensor(v, dtype=torch.float64) for v in geometry]
    options = {"tests": ["moran_normal"], "wx_indices": [1, 2]}
    if change == "dtype":
        tensors[0] = tensors[0].float()
    elif change == "shape":
        tensors[1] = tensors[1][:3]
    elif change == "negative":
        tensors[3][0, 1] = -0.1
    elif change == "diagonal":
        tensors[3][0, 0] = 1
    elif change == "nonfinite":
        tensors[1][0] = float("inf")
    elif change == "draws":
        options["draws"] = True
    elif change == "seed":
        options["seed"] = 2**63
    elif change == "tests":
        options["tests"] = [["moran_normal"]]
    elif change == "wx_index":
        options["wx_indices"] = [[1]]
    with pytest.raises(AnalysisError):
        diagnose(*tensors, **options)


def test_actual_tensor_receipt_and_authoritative_work_proxy(geometry):
    answer = run(geometry)
    n, p = geometry[0].shape
    estimates = resource_estimates(n, p, draws=199, mc=True, wx=True)
    assert answer["geometry"]["work_estimate"] == estimates["work"]
    assert answer["resource_plan"]["buffers"] == estimates["buffers"]
    assert sum(v["bytes"] for v in answer["workspace_observed"].values()) < sum(
        estimates["buffers"].values()
    )
    assert answer["workspace_observed"]["null_statistics"]["shape"] == [199]
    assert all(
        v["device"] == "cpu" and v["dtype"] == "torch.float64"
        for v in answer["workspace_observed"].values()
    )


def test_kernel_does_not_inherit_meta_default_or_float32_dtype(geometry):
    tensors = [torch.tensor(v, dtype=torch.float64, device="cpu") for v in geometry]
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            answer = diagnose(*tensors, tests=["moran_gaussian_mc"], draws=5)
    finally:
        torch.set_default_dtype(old)
    assert answer["null_statistics"].device.type == "cpu"
    assert answer["null_statistics"].dtype == torch.float64
