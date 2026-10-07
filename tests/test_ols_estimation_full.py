"""Independent dense OLS/WLS, resampling and covariance formula references."""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hac, cov_cluster_2groups
import torch

from openecon.engines.contracts import KernelError
from openecon.linear_ols.estimation import estimate


TERMS = ["Intercept", "a", "b", "c"]


@pytest.fixture
def data():
    rng = np.random.default_rng(101)
    x = np.column_stack([np.ones(96), rng.normal(size=(96, 3))])
    y = x @ np.array([1., 2., -.5, .3]) + rng.normal(size=96) * (.2 + np.abs(x[:, 1]))
    weights = rng.uniform(.2, 4., size=96)
    return torch.tensor(x), torch.tensor(y), torch.tensor(weights)


def close(actual, expected, tolerance=2e-11):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
@pytest.mark.parametrize("weight_type", [None, "aw", "pw", "iw"])
def test_weighted_covariances_match_statsmodels_and_stata_degrees(data, intercept, covariance, weight_type):
    x, y, weights = data
    if not intercept:
        x = x[:, 1:]
    terms = TERMS if intercept else TERMS[1:]
    result = estimate(x, y, terms=terms, intercept=intercept, covariance=covariance,
                      weights=weights if weight_type else None, weight_type=weight_type)
    actual_covariance = "HC1" if weight_type == "pw" and covariance == "nonrobust" else covariance
    ref = (sm.WLS(y.numpy(), x.numpy(), weights=weights.numpy()) if weight_type else
           sm.OLS(y.numpy(), x.numpy())).fit(cov_type=actual_covariance)
    close(result["params"], ref.params)
    expected = ref.cov_params()
    nobs = len(y)
    if weight_type == "iw" and covariance == "nonrobust":
        nobs = int(weights.sum())
        expected = expected * (len(y) - x.shape[1]) / (nobs - x.shape[1])
        close(result["weights"], weights.numpy())
    elif weight_type:
        close(result["weights"].sum(), len(y))
    close(result["covariance"], expected)
    assert result["nobs"] == nobs
    assert result["df_resid"] == nobs - x.shape[1]
    assert result["df_inference"] == result["df_resid"]
    assert result["metrics"]["physical_nobs"] == len(y)
    assert result["inference"]["covariance"] == actual_covariance
    assert result["terms"] == terms
    close(result["x"], x.numpy())
    close(result["resid"], ref.resid)
    close(result["fitted"], ref.fittedvalues)
    close(result["leverage"].sum(), len(terms))
    close(result["sigma2"], np.dot(result["weights"].numpy(), ref.resid**2) / result["df_resid"])


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster"])
def test_frequency_weighted_results_equal_literal_expanded_ols(data, covariance):
    x, y, _ = data
    counts = torch.arange(len(y)) % 4 + 1
    expanded_x, expanded_y = torch.repeat_interleave(x, counts, dim=0), torch.repeat_interleave(y, counts)
    group = torch.arange(len(y)) % 13
    expanded_group = torch.repeat_interleave(group, counts)
    result = estimate(x, y, terms=TERMS, covariance=covariance, weights=counts, weight_type="fw",
                      clusters=[group] if covariance == "cluster" else None)
    model = sm.OLS(expanded_y.numpy(), expanded_x.numpy()).fit()
    reference = cov_cluster(model, expanded_group.numpy()) if covariance == "cluster" else model.get_robustcov_results(cov_type=covariance).cov_params() if covariance != "nonrobust" else model.cov_params()
    close(result["params"], model.params)
    close(result["covariance"], reference)
    assert result["nobs"] == int(counts.sum())
    assert result["df_resid"] == int(counts.sum()) - 4
    assert result["df_inference"] == (12 if covariance == "cluster" else result["df_resid"])


@pytest.mark.parametrize("weight_type", ["aw", "pw", "iw"])
@pytest.mark.parametrize("covariance", ["HC1", "HC3", "cluster"])
def test_robust_weights_invariant_to_scale(data, weight_type, covariance):
    x, y, w = data
    options = {"terms": TERMS, "covariance": covariance, "weight_type": weight_type}
    if covariance == "cluster":
        options["clusters"] = [torch.arange(len(y)) % 16]
    first = estimate(x, y, weights=w, **options)
    second = estimate(x, y, weights=w * 1e290, **options)
    close(first["params"], second["params"].numpy())
    close(first["covariance"], second["covariance"].numpy())
    assert first["nobs"] == second["nobs"] == len(y)


@pytest.mark.parametrize(("alias", "short"), [("aweight", "aw"), ("fweight", "fw"), ("pweight", "pw"), ("iweight", "iw")])
def test_public_weight_names_and_long_aliases(data, alias, short):
    x, y, w = data
    if short == "fw":
        w = w.ceil()
    first = estimate(x, y, terms=TERMS, weights=w, weight_type=alias)
    second = estimate(x, y, terms=TERMS, weights=w, weight_type=short)
    close(first["params"], second["params"].numpy())
    close(first["covariance"], second["covariance"].numpy())


def test_rank_omissions_are_deterministic_and_intercept_is_prioritized(data):
    x, y, _ = data
    extended = torch.column_stack([x[:, 1], torch.ones(len(y)), x[:, 2], x[:, 1] * 2, torch.ones(len(y)) * 3, x[:, 3], torch.zeros(len(y))])
    labels = ["a", "Intercept", "b", "twice_a", "constant_copy", "c", "zero"]
    result = estimate(extended, y, terms=labels, covariance="HC3")
    assert result["kept_indices"] == [0, 1, 2, 5]
    assert result["terms"] == ["a", "Intercept", "b", "c"]
    assert result["omitted_terms"] == ["twice_a", "constant_copy", "zero"]
    close(result["params"], sm.OLS(y.numpy(), extended[:, [0, 1, 2, 5]].numpy()).fit().params)
    assert any("Collinear" in warning for warning in result["warnings"])


def test_no_intercept_constant_predictor_is_estimable(data):
    x, y, _ = data
    result = estimate(x, y, terms=TERMS, intercept=False)
    close(result["params"], sm.OLS(y.numpy(), x.numpy()).fit().params)
    assert result["omitted_terms"] == []
    assert result["metrics"]["df_model"] == 4


def test_extreme_offset_and_differently_scaled_columns_keep_fitted_and_covariance(data):
    x, y, _ = data
    original = estimate(x, y, terms=TERMS, covariance="HC3")
    scaled = x.clone()
    scaled[:, 1] = 1e12 + x[:, 1]
    scaled[:, 2] *= 1e-100
    scaled[:, 3] *= 1e100
    result = estimate(scaled, y, terms=TERMS, covariance="HC3")
    # The offset input itself is rounded to ~1e-4; compare the equivalent
    # centered physical data rather than pretending lost bits are recoverable.
    centered = scaled.clone()
    centered[:, 1] -= 1e12
    centered[:, 2] *= 1e100
    centered[:, 3] *= 1e-100
    reference = estimate(centered, y, terms=TERMS, covariance="HC3")
    close(result["fitted"], reference["fitted"].numpy())
    close(result["resid"], reference["resid"].numpy())
    close(result["params"][1] , reference["params"][1].numpy())
    close(result["params"][2] * 1e-100, reference["params"][2].numpy())
    close(result["params"][3] * 1e100, reference["params"][3].numpy())
    assert result["omitted_terms"] == [] and original["omitted_terms"] == []


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4])
def test_arbitrary_multiway_cgm_matches_independent_inclusion_exclusion(data, dimensions):
    x, y, w = data
    labels = [np.arange(len(y)) % modulus for modulus in [12, 8, 7, 5]][:dimensions]
    result = estimate(x, y, terms=TERMS, weights=w, weight_type="aw", covariance="cluster", clusters=labels)
    model = sm.WLS(y.numpy(), x.numpy(), weights=w.numpy()).fit()
    raw = np.zeros((4, 4))
    for width in range(1, dimensions + 1):
        for selected in combinations(range(dimensions), width):
            _, inverse = np.unique(np.column_stack([labels[i] for i in selected]), axis=0, return_inverse=True)
            raw += (-1)**(width + 1) * cov_cluster(model, inverse)
    if dimensions > 1:
        values, vectors = np.linalg.eigh(raw)
        raw = (vectors * np.maximum(values, 0)) @ vectors.T
    close(result["covariance"], raw)
    assert result["df_inference"] == min(len(np.unique(label)) for label in labels) - 1
    assert len(result["inference"]["combination_counts"]) == 2**dimensions - 1
    assert result["inference"]["dfadjust"] is False
    assert result["inference"]["hansen"] is False


def test_two_way_cgm_matches_statsmodels_without_eigenvalue_clipping(data):
    x, y, _ = data
    first, second = np.arange(len(y)) % 12, np.arange(len(y)) // 8
    result = estimate(x, y, terms=TERMS, covariance="cluster", clusters=[first, second])
    ref = cov_cluster_2groups(sm.OLS(y.numpy(), x.numpy()).fit(), first, second)[0]
    values, vectors = np.linalg.eigh(ref)
    close(result["covariance"], (vectors * np.maximum(values, 0)) @ vectors.T)


@pytest.mark.parametrize("covariance", ["cluster_hc2", "cluster_hc3"])
def test_cluster_bias_corrections_match_full_annihilator_reference(data, covariance):
    x, y, w = data
    labels = np.arange(len(y)) // 12
    result = estimate(x, y, terms=TERMS, weights=w, weight_type="aw", covariance=covariance, clusters=[labels])
    wn = w.numpy() * len(y) / w.numpy().sum()
    xw = x.numpy() * np.sqrt(wn)[:, None]
    bread = np.linalg.inv(xw.T @ xw)
    model = sm.WLS(y.numpy(), x.numpy(), weights=wn).fit()
    residual = model.resid * np.sqrt(wn)
    meat = np.zeros((4, 4))
    for group in np.unique(labels):
        rows = xw[labels == group]
        matrix = np.eye(len(rows)) - rows @ bread @ rows.T
        eigenvalues, vectors = np.linalg.eigh(matrix)
        power = .5 if covariance == "cluster_hc2" else 1.
        adjustment = (vectors * eigenvalues**(-power)) @ vectors.T
        score = rows.T @ adjustment @ residual[labels == group]
        meat += np.outer(score, score)
    close(result["covariance"], bread @ meat @ bread)
    assert result["df_inference"] == 7
    assert result["inference"]["small_sample_correction"] == 1
    assert any("not enabled" in message for message in result["warnings"])


def kernel_reference(distance, lags, kernel):
    z = distance / (lags + 1)
    if kernel == "truncated":
        return float(z < 1)
    if kernel == "bartlett":
        return max(0, 1 - z)
    if kernel == "parzen":
        return 1 - 6*z*z + 6*z**3 if z <= .5 else 2 * max(0, 1-z)**3
    theta = 6 * math.pi * z / 5
    return 3 * (math.sin(theta)/theta - math.cos(theta)) / theta**2


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic_spectral", "truncated"])
@pytest.mark.parametrize("gaps", [False, True])
def test_hac_all_kernels_preserve_calendar_gaps_against_pair_formula(data, kernel, gaps):
    x, y, w = data
    x, y, w = x[:25], y[:25], w[:25]
    times = np.cumsum(np.arange(len(y)) % 3 + 1) if gaps else np.arange(len(y))
    # Scramble storage order to require sorting in the covariance only.
    order = np.random.default_rng(12).permutation(len(y))
    result = estimate(x[order], y[order], terms=TERMS, weights=w[order], weight_type="aw",
                      covariance="HAC", time=torch.tensor(times[order]), lags=3, kernel=kernel)
    wn = w.numpy() * len(y) / w.numpy().sum()
    ref = sm.WLS(y.numpy(), x.numpy(), weights=wn).fit()
    bread = np.linalg.inv(x.numpy().T @ (wn[:, None] * x.numpy()))
    scores = x.numpy() * (wn * ref.resid)[:, None]
    meat = scores.T @ scores
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            cross = np.outer(scores[i], scores[j])
            meat += kernel_reference(times[j] - times[i], 3, kernel) * (cross + cross.T)
    close(result["covariance"], bread @ meat @ bread * len(y) / (len(y) - 4))
    assert result["inference"]["time_gaps"] is True
    assert result["inference"]["bandwidth"] == 4
    if kernel == "quadratic_spectral":
        assert result["inference"]["noncompact_support"] is True
        assert any("noncompact" in message for message in result["warnings"])


def test_bartlett_hac_matches_statsmodels_newey_west(data):
    x, y, _ = data
    result = estimate(x, y, terms=TERMS, covariance="HAC", time=torch.arange(len(y)), lags=5)
    close(result["covariance"], cov_hac(sm.OLS(y.numpy(), x.numpy()).fit(), nlags=5))


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("weight_type", [None, "aw"])
def test_seeded_pairs_and_cluster_bootstrap_match_manual_refits(data, clustered, weight_type):
    x, y, w = data
    weights = w if weight_type else torch.ones_like(w)
    labels = torch.arange(len(y)) % 12
    result = estimate(x, y, terms=TERMS, weights=weights if weight_type else None, weight_type=weight_type,
                      covariance="bootstrap", clusters=[labels] if clustered else None, reps=31, seed=412)
    generator = torch.Generator().manual_seed(412)
    coefficients = []
    units = 12 if clustered else len(y)
    for _ in range(31):
        indices = torch.randint(units, (units,), generator=generator)
        count = torch.bincount(indices, minlength=units)
        draw = weights * (count[labels] if clustered else count)
        selected = draw > 0
        ref = sm.WLS(y[selected].numpy(), x[selected].numpy(), weights=draw[selected].numpy()).fit()
        coefficients.append(ref.params)
    close(result["covariance"], np.cov(np.array(coefficients), rowvar=False, ddof=1))
    close(result["inference"]["resampling_mean"], np.mean(coefficients, axis=0))
    assert result["inference"]["successful_reps"] == 31
    assert result["inference"]["seed"] == 412
    assert result["inference"]["distribution"] == "normal"
    assert result["df_inference"] == math.inf
    repeat = estimate(x, y, terms=TERMS, weights=weights if weight_type else None, weight_type=weight_type,
                      covariance="bootstrap", clusters=[labels] if clustered else None, reps=31, seed=412)
    assert torch.equal(result["covariance"], repeat["covariance"])


def test_frequency_bootstrap_uses_counts_without_expanded_rows(data, monkeypatch):
    x, y, _ = data
    x, y = x[:12], y[:12]
    w = torch.arange(12, dtype=torch.float64) + 1
    original = torch.randint

    def limited(high, size, **kwargs):
        assert math.prod(size) <= len(y)
        return original(high, size, **kwargs)

    monkeypatch.setattr(torch, "randint", limited)
    result = estimate(x, y, terms=TERMS, weights=w, weight_type="fw", covariance="bootstrap", reps=29, seed=32)
    generator = torch.Generator().manual_seed(32)
    coefficients = []
    for _ in range(29):
        remaining, mass = torch.tensor(float(w.sum()), dtype=w.dtype), w.sum().clone()
        count = torch.zeros_like(w)
        for i in range(len(w)-1):
            count[i] = torch.binomial(remaining, w[i]/mass, generator=generator)
            remaining -= count[i]
            mass -= w[i]
        count[-1] = remaining
        assert count.sum() == w.sum()
        selected = count > 0
        coefficients.append(sm.WLS(y[selected].numpy(), x[selected].numpy(), weights=count[selected].numpy()).fit().params)
    close(result["covariance"], np.cov(np.array(coefficients), rowvar=False))
    assert result["inference"]["frequency_count_resampling"] is True


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("weight_type", [None, "aw", "fw"])
def test_analytic_jackknife_matches_literal_delete_unit_refits(data, clustered, weight_type):
    x, y, w = data
    x, y, w = x[:20], y[:20], w[:20]
    labels = np.arange(len(y)) // 4
    if weight_type == "fw":
        counts = w.ceil().to(torch.int64)
        ref_x = torch.repeat_interleave(x, counts, dim=0).numpy()
        ref_y = torch.repeat_interleave(y, counts).numpy()
        ref_w = np.ones(len(ref_y))
        ref_group = np.repeat(labels, counts.numpy())
        weights = counts
    else:
        ref_x, ref_y = x.numpy(), y.numpy()
        ref_w = w.numpy() if weight_type else np.ones(len(y))
        ref_group = labels
        weights = w if weight_type else None
    result = estimate(x, y, terms=TERMS, weights=weights, weight_type=weight_type, covariance="jackknife",
                      clusters=[labels] if clustered else None)
    groups = np.unique(ref_group) if clustered else np.arange(len(ref_y))
    coefficients = []
    for group in groups:
        keep = ref_group != group if clustered else np.arange(len(ref_y)) != group
        coefficients.append(sm.WLS(ref_y[keep], ref_x[keep], weights=ref_w[keep]).fit().params)
    coefficients = np.array(coefficients)
    centered = coefficients - coefficients.mean(axis=0)
    expected = centered.T @ centered * (len(groups) - 1) / len(groups)
    close(result["covariance"], expected)
    assert result["df_inference"] == len(groups) - 1
    assert result["inference"]["reps"] == len(groups)


@pytest.mark.parametrize("covariance", ["HC2", "HC3", "cluster_hc2", "cluster_hc3"])
def test_unit_leverage_uses_explicit_moore_penrose_convention(covariance):
    x = torch.tensor([[1., 1.], [1., 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
    y = torch.tensor([2., 0., 1., 3.], dtype=torch.float64)
    result = estimate(x, y, terms=["Intercept", "singleton"], covariance=covariance,
                      clusters=[[0, 1, 1, 1]] if covariance.startswith("cluster") else None)
    assert torch.isfinite(result["covariance"]).all()
    assert any("Moore-Penrose" in message for message in result["warnings"])


def test_jackknife_rejects_delete_unit_loss_of_identification():
    x = torch.tensor([[1., 1.], [1., 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
    y = torch.tensor([2., 0., 1., 3.], dtype=torch.float64)
    for clusters in [None, [[0, 1, 1, 1]]]:
        with pytest.raises(KernelError) as exc:
            estimate(x, y, terms=["Intercept", "singleton"], covariance="jackknife", clusters=clusters)
        assert exc.value.code == "jackknife_rank_deficient"


@pytest.mark.parametrize(("options", "code"), [
    ({"covariance": "unknown"}, "unsupported_covariance"),
    ({"weight_type": "aw"}, "invalid_weights"),
    ({"weights": torch.ones(96), "weight_type": "unknown"}, "invalid_weights"),
    ({"weights": torch.zeros(96), "weight_type": "aw"}, "invalid_weights"),
    ({"weights": torch.ones(96)*1.5, "weight_type": "fw"}, "invalid_weights"),
    ({"covariance": "cluster", "clusters": [[1]*96]}, "insufficient_clusters"),
    ({"covariance": "cluster", "clusters": [[None]*96]}, "invalid_clusters"),
    ({"covariance": "HAC", "time": None}, "invalid_time"),
    ({"covariance": "HAC", "time": torch.zeros(96)}, "invalid_time"),
    ({"covariance": "HAC", "time": torch.arange(96)+.5}, "invalid_time"),
    ({"covariance": "HAC", "time": torch.arange(96), "lags": -1}, "invalid_covariance_options"),
    ({"covariance": "bootstrap", "reps": 1}, "invalid_resampling_options"),
    ({"covariance": "bootstrap", "seed": -1}, "invalid_resampling_options"),
    ({"covariance": "bootstrap", "clusters": [list(range(96)), list(range(96))]}, "unsupported_covariance"),
    ({"covariance": "cluster_hc2", "clusters": [list(range(96)), list(range(96))]}, "unsupported_covariance"),
])
def test_invalid_and_unimplemented_controls_have_truthful_errors(data, options, code):
    x, y, _ = data
    with pytest.raises(KernelError) as exc:
        estimate(x, y, terms=TERMS, **options)
    assert exc.value.code == code


def test_float64_autograd_free_result_and_input_unchanged(data):
    x, y, _ = data
    x, y = x.float().requires_grad_(True), y.float().requires_grad_(True)
    before_x, before_y = x.clone(), y.clone()
    result = estimate(x, y, terms=TERMS, covariance="HC3")
    for field in ["params", "covariance", "x", "y", "weights", "resid", "fitted", "bread", "leverage"]:
        assert result[field].dtype == torch.float64
        assert result[field].device == x.device
        assert result[field].requires_grad is False
    assert torch.equal(before_x, x) and torch.equal(before_y, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_results_stay_on_gpu_and_match_cpu(data):
    x, y, w = data
    cpu = estimate(x, y, terms=TERMS, covariance="HC3", weights=w, weight_type="aw")
    gpu = estimate(x.cuda(), y.cuda(), terms=TERMS, covariance="HC3", weights=w.cuda(), weight_type="aw")
    for field in ["params", "covariance", "x", "y", "weights", "resid", "fitted", "bread", "leverage"]:
        assert gpu[field].device.type == "cuda"
        close(gpu[field], cpu[field].numpy(), tolerance=3e-10)


def full_projection_adjustment(x, weights, gradient, groups, power):
    whitened = x * np.sqrt(weights)[:, None]
    bread = np.linalg.inv(whitened.T @ whitened)
    hat = whitened @ bread @ whitened.T
    residual_projection = np.eye(len(x)) - hat
    projection_columns = []
    for label in np.unique(groups):
        selected = np.flatnonzero(groups == label)
        partial = residual_projection[selected][:, selected]
        values, vectors = np.linalg.eigh(partial)
        inverse = np.where(values > 1e-12, np.maximum(values, 1e-300)**(-power), 0)
        adjustment = (vectors * inverse) @ vectors.T
        a = adjustment @ whitened[selected] @ bread @ gradient
        projection_columns.append(residual_projection[:, selected] @ a)
    matrix = np.column_stack(projection_columns)
    quadratic = matrix.T @ matrix
    trace = np.trace(quadratic)
    df = trace**2 / np.square(quadratic).sum()
    scale = np.sqrt(trace / (gradient @ bread @ gradient))
    return df, scale


@pytest.mark.parametrize("covariance", ["HC2", "HC3", "cluster_hc2", "cluster_hc3"])
@pytest.mark.parametrize("kind", [None, "aw", "fw"])
def test_dfadjust_and_hansen_match_full_projection_reference(data, covariance, kind):
    x, y, w = data
    x, y, w = x[:20], y[:20], w[:20]
    groups = np.arange(len(y)) // 4
    if kind == "fw":
        w = w.ceil()
        reference_x = np.repeat(x.numpy(), w.numpy().astype(int), axis=0)
        reference_weights = np.ones(len(reference_x))
        reference_groups = np.repeat(groups, w.numpy().astype(int))
    else:
        reference_x = x.numpy()
        reference_weights = (w.numpy() * len(y) / w.numpy().sum()) if kind else np.ones(len(y))
        reference_groups = groups
    clustered = covariance.startswith("cluster")
    if not clustered:
        reference_groups = np.arange(len(reference_x))
    result = estimate(x, y, terms=TERMS, covariance=covariance, weights=w if kind else None,
                      weight_type=kind, clusters=[groups] if clustered else None,
                      dfadjust=True, hansen=covariance.endswith("3"))
    for i in range(4):
        gradient = np.eye(4)[i]
        expected_df, expected_scale = full_projection_adjustment(
            reference_x, reference_weights, gradient, reference_groups, .5 if covariance.endswith("2") else 1.)
        close(result["inference"]["coefficient_df"][i], expected_df)
        close(result["inference"]["coefficient_scale"][i], expected_scale if covariance.endswith("3") else 1.)
    gradient = torch.tensor([0., .5, -.25, 0.], dtype=torch.float64)
    expected_df, expected_scale = full_projection_adjustment(
        reference_x, reference_weights, gradient.numpy(), reference_groups, .5 if covariance.endswith("2") else 1.)
    callback = result["contrast_inference"](gradient)
    close(callback["df"], expected_df)
    close(callback["scale"], expected_scale if covariance.endswith("3") else 1.)
    assert result["inference"]["dfadjust"] is True
    assert result["inference"]["hansen"] is covariance.endswith("3")


@pytest.mark.parametrize("covariance", ["HC3", "cluster_hc3"])
def test_hansen_scales_and_degrees_invariant_to_regressor_units(data, covariance):
    x, y, w = data
    options = {"terms": TERMS, "covariance": covariance, "weights": w, "weight_type": "aw", "hansen": True}
    if covariance.startswith("cluster"):
        options["clusters"] = [np.arange(len(y)) // 12]
    first = estimate(x, y, **options)
    scaled = x.clone()
    scaled[:, 1] *= 1e8
    second = estimate(scaled, y, **options)
    close(first["inference"]["coefficient_df"], second["inference"]["coefficient_df"])
    close(first["inference"]["coefficient_scale"], second["inference"]["coefficient_scale"])
    assert first["inference"]["coefficient_scale"][1] >= 1
    close(first["metrics"]["f_statistic"], second["metrics"]["f_statistic"])
    assert first["metrics"]["f_df_num"] == second["metrics"]["f_df_num"] == 3


@pytest.mark.parametrize("clustered", [False, True])
def test_singular_hansen_case_rejected_explicitly(clustered):
    x = torch.tensor([[1., 1.], [1., 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
    y = torch.tensor([2., 0., 1., 3.], dtype=torch.float64)
    with pytest.raises(KernelError) as exc:
        estimate(x, y, terms=["Intercept", "singleton"], covariance="cluster_hc3" if clustered else "HC3",
                 clusters=[[0, 1, 1, 1]] if clustered else None, hansen=True)
    assert exc.value.code == "unsupported_inference"


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic_spectral"])
def test_newey_west_1994_automatic_lags_match_manual_pilot(data, kernel):
    x, y, w = data
    times = np.cumsum(np.arange(len(y)) % 3 + 1)
    result = estimate(x, y, terms=TERMS, weights=w, weight_type="aw", covariance="HAC",
                      time=torch.tensor(times), lags="auto", kernel=kernel)
    parameters = {"bartlett": (1, 2/9, 1.1447), "parzen": (2, 4/25, 2.6614), "quadratic_spectral": (2, 2/25, 1.3221)}
    q, exponent, constant = parameters[kernel]
    pilot = int(20 * (len(y)/100)**exponent)
    weights = w.numpy() * len(y) / w.numpy().sum()
    residual = sm.WLS(y.numpy(), x.numpy(), weights=weights).fit().resid
    f = residual * weights * x.numpy()[:, 1:].sum(axis=1)
    sigma0 = f @ f / len(y)
    sums, orders = sigma0, 0.
    pairs = {value: i for i, value in enumerate(times)}
    for lag in range(1, pilot+1):
        sigma = sum(f[i]*f[pairs[value+lag]] for i, value in enumerate(times) if value+lag in pairs)/len(y)
        sums += 2*sigma
        orders += 2*sigma*lag**q
    expected = min(constant * abs(orders/sums)**(2/(2*q+1))*len(y)**(1/(2*q+1)), pilot)
    if kernel != "quadratic_spectral":
        expected = int(expected)
    close(result["inference"]["lags"], expected)
    assert result["inference"]["lag_selection"] == "Newey-West 1994 plug-in"
    assert result["inference"]["pilot_lags"] == pilot
    assert result["inference"]["automatic_lags"] is True
    assert result["inference"]["lag_selection_time_gaps"] is True


def test_all_covariances_avoid_observation_square_matrices(data, monkeypatch):
    x, y, w = data
    original = torch.eye

    def parameter_identity(size, *args, **kwargs):
        assert size <= x.shape[1]
        return original(size, *args, **kwargs)

    monkeypatch.setattr(torch, "eye", parameter_identity)
    for covariance in ["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "cluster_hc2", "cluster_hc3", "HAC", "bootstrap", "jackknife"]:
        options = {"terms": TERMS, "covariance": covariance, "weights": w, "weight_type": "aw"}
        if covariance.startswith("cluster"):
            options["clusters"] = [torch.arange(len(y)) % 12]
        if covariance == "HAC":
            options.update(time=torch.arange(len(y)), lags=4)
        if covariance == "bootstrap":
            options.update(reps=3, seed=3)
        if covariance.endswith("2") or covariance.endswith("3"):
            options["dfadjust"] = True
        result = estimate(x, y, **options)
        assert result["covariance"].shape == (4, 4)


def test_hac_sparse_calendar_and_default_lag_metadata(data):
    x, y, _ = data
    result = estimate(x[:8], y[:8], terms=TERMS, covariance="HAC",
                      time=torch.arange(8, dtype=torch.int64) * 1_000_000, lags=2_000_000)
    ref = sm.OLS(y[:8].numpy(), x[:8].numpy()).fit()
    scores = x[:8].numpy() * ref.resid[:, None]
    meat = scores.T @ scores
    for lag in [1, 2]:
        cross = scores[:-lag].T @ scores[lag:]
        meat += (1 - lag * 1_000_000 / 2_000_001) * (cross + cross.T)
    bread = np.linalg.inv(x[:8].numpy().T @ x[:8].numpy())
    close(result["covariance"], bread @ meat @ bread * 2)
    default = estimate(x[:8], y[:8], terms=TERMS, covariance="HAC", time=torch.arange(8))
    assert default["inference"]["lags"] == 6
    assert default["inference"]["lag_selection"] == "Stata N-2 default"


def test_bootstrap_failed_draws_are_counted_without_changed_terms():
    x = torch.tensor([[1., 1.], [1., 0.], [1., 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
    y = torch.tensor([2., 0., 1., 3., 4.], dtype=torch.float64)
    result = estimate(x, y, terms=["Intercept", "singleton"], covariance="bootstrap", reps=40, seed=41)
    assert result["terms"] == ["Intercept", "singleton"]
    assert result["inference"]["failed_reps"] > 0
    assert result["inference"]["successful_reps"] + result["inference"]["failed_reps"] == 40
    assert any("excluded" in message for message in result["warnings"])


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster",
                                       "cluster_hc2", "cluster_hc3", "HAC", "bootstrap", "jackknife"])
@pytest.mark.parametrize("weighted", [False, True])
def test_prediction_callbacks_are_translation_and_unit_invariant(data, covariance, weighted):
    x, y, weights = data
    shifted = x.clone()
    shifted[:, 1] += 1e12
    shifted[:, 2] *= 1e-80
    shifted[:, 3] *= 1e80
    # Center the rounded input itself: the lost 1e-4 source precision is real.
    reference_x = shifted.clone()
    reference_x[:, 1] -= 1e12
    reference_x[:, 2] *= 1e80
    reference_x[:, 3] *= 1e-80
    options = {"terms": TERMS, "covariance": covariance}
    if weighted:
        options.update(weights=weights, weight_type="aw")
    if covariance.startswith("cluster"):
        options["clusters"] = [torch.arange(len(y)) % 12]
    if covariance == "HAC":
        options.update(time=torch.arange(len(y)), lags=4)
    if covariance == "bootstrap":
        options.update(reps=19, seed=322)
    result = estimate(shifted, y, **options)
    reference = estimate(reference_x, y, **options)
    new_x, new_reference_x = shifted[:13].clone(), reference_x[:13].clone()
    new_x[:, 1] += .25
    new_reference_x[:, 1] += .25
    new_x[:, 2] -= .5e-80
    new_reference_x[:, 2] -= .5
    for actual, expected in [(shifted, reference_x), (new_x, new_reference_x)]:
        close(result["prediction_variance"](actual), reference["prediction_variance"](expected).numpy())
        close(result["prediction_leverage"](actual), reference["prediction_leverage"](expected).numpy())
        close(result["prediction_fitted"](actual), reference["prediction_fitted"](expected).numpy())
    close(result["prediction_leverage"](shifted) * result["weights"], result["leverage"].numpy())


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
def test_shifted_prediction_variance_matches_independent_centered_oracle(data, covariance):
    x, y, _ = data
    shifted = x.clone()
    shifted[:, 1] += 1e8
    centered = shifted.clone()
    centered[:, 1] -= 1e8
    labels = np.arange(len(y)) % 12
    result = estimate(shifted, y, terms=TERMS, covariance=covariance,
                      clusters=[labels] if covariance == "cluster" else None)
    oracle = sm.OLS(y.numpy(), centered.numpy()).fit(cov_type="nonrobust" if covariance == "cluster" else covariance)
    variance = cov_cluster(oracle, labels) if covariance == "cluster" else oracle.cov_params()
    expected = np.einsum("ij,jk,ik->i", centered.numpy(), variance, centered.numpy())
    close(result["prediction_variance"](shifted), expected)
    bread = np.linalg.inv(centered.numpy().T @ centered.numpy())
    close(result["prediction_leverage"](shifted), np.einsum("ij,jk,ik->i", centered.numpy(), bread, centered.numpy()))


def test_clipped_multiway_prediction_uses_factor_of_unchanged_public_covariance(data):
    x, y, w = data
    result = estimate(x, y, terms=TERMS, covariance="cluster", weights=w, weight_type="aw",
                      clusters=[np.arange(len(y)) % modulus for modulus in [12, 8, 7, 5]])
    factor = result["prediction_variance"].__self__.raw_factor
    assert factor is not None
    close(factor @ factor.T, result["covariance"].numpy())
    close(result["prediction_variance"](x), (x @ factor).square().sum(dim=1).numpy())
    assert result["inference"]["psd_projection"] == "raw coefficient coordinates; not basis invariant"


def test_prediction_callbacks_handle_omitted_terms_and_no_intercept(data):
    x, y, _ = data
    extended = torch.column_stack([x, x[:, 1] * 2])
    result = estimate(extended, y, terms=[*TERMS, "duplicate"])
    close(result["prediction_leverage"](result["x"]), result["leverage"].numpy())
    close(result["prediction_fitted"](result["x"]), result["fitted"].numpy())
    scaled = x[:, 1:].clone()
    scaled[:, 0] *= 1e-80
    scaled[:, 1] *= 1e80
    raw = estimate(scaled, y, terms=TERMS[1:], intercept=False, covariance="HC3")
    reference = estimate(x[:, 1:], y, terms=TERMS[1:], intercept=False, covariance="HC3")
    close(raw["prediction_variance"](scaled), reference["prediction_variance"](x[:, 1:]).numpy())


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster",
                                       "cluster_hc2", "cluster_hc3", "HAC", "bootstrap", "jackknife"])
@pytest.mark.parametrize("intercept", [False, True])
def test_arbitrary_functional_covariance_is_stable_with_high_offsets_and_units(data, covariance, intercept):
    x, y, _ = data
    units = torch.tensor([1., 1., 1e-80, 1e80], dtype=torch.float64)
    raw = x * units
    if intercept:
        raw[:, 1] += 1e8
    else:
        raw = raw[:, 1:]
    centered = raw.clone()
    if intercept:
        centered[:, 1] -= 1e8
        centered /= units
    else:
        centered /= units[1:]
    terms = TERMS if intercept else TERMS[1:]
    options = {"terms": terms, "intercept": intercept, "covariance": covariance}
    if covariance.startswith("cluster"):
        options["clusters"] = [torch.arange(len(y)) % 12]
    if covariance == "HAC":
        options.update(time=torch.arange(len(y)), lags=4)
    if covariance == "bootstrap":
        options.update(reps=19, seed=35)
    fit = estimate(raw, y, **options)
    reference = estimate(centered, y, **options)
    functionals = torch.tensor([[1., .25, -.75, 1.5], [2., -.5, .25, -.75],
                                [0., 1., .5, -1.], [-1., -.25, .75, -1.5]], dtype=torch.float64)
    if not intercept:
        functionals = functionals[:, 1:]
    raw_functionals = functionals * (units if intercept else units[1:])
    if intercept:
        raw_functionals[:, 1] += 1e8 * functionals[:, 0]
    expected = functionals @ reference["covariance"] @ functionals.T
    actual = fit["contrast_covariance"](raw_functionals)
    close(actual, expected.numpy())
    close(actual.diagonal(), fit["prediction_variance"](raw_functionals).numpy())
    close(fit["contrast_covariance"](raw_functionals[:1]), expected[:1, :1].numpy())
    assert actual.dtype == torch.float64 and actual.device == raw_functionals.device


def test_clipped_multiway_functional_covariance_retains_raw_projection(data):
    x, y, w = data
    fit = estimate(x, y, terms=TERMS, covariance="cluster", weights=w, weight_type="aw",
                   clusters=[np.arange(len(y)) % modulus for modulus in [12, 8, 7, 5]])
    rows = torch.tensor([[1., 2., -.5, .75], [0., 1., .5, -1.], [-2., .25, 3., .1]], dtype=torch.float64)
    factor = fit["prediction_variance"].__self__.raw_factor
    assert factor is not None
    expected_factor = rows @ factor
    close(fit["contrast_covariance"](rows), (expected_factor @ expected_factor.T).numpy())
    close(fit["contrast_covariance"](rows), (rows @ fit["covariance"] @ rows.T).numpy())
    assert fit["inference"]["psd_projection"] == "raw coefficient coordinates; not basis invariant"
