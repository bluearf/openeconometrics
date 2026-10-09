"""Independent cell-space likelihood, information and rank acceptance."""

from itertools import product
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2, norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.loglinear import (
    loglinear_ipf,
    loglinear_ml,
    loglinear_compare,
)
from openecon.econometrics.summary_state import summary_state, restore_summary
from openecon.resources import use_workspace_budget


def grid(shape=(2, 3), counts=None):
    dims = list("abc")[: len(shape)]
    coords = list(product(*(range(k) for k in shape)))
    frame = pd.DataFrame(coords, columns=dims)
    frame["count"] = (
        counts if counts is not None else [11 + 3 * i + (i % 3) * 2 for i in range(len(coords))]
    )
    return frame, dims, {d: list(range(k)) for d, k in zip(dims, shape)}


def dummy(frame, dims, levels):
    return np.column_stack(
        [
            np.ones(len(frame)),
            *[(frame[d].values == c).astype(float) for d in dims for c in levels[d][1:]],
        ]
    )


def oracle(x, y, offset=None):
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset)

    def obj(b):
        eta = offset + x @ b
        return np.sum(np.exp(eta) - y * eta + gammaln(y + 1))

    def jac(b):
        return x.T @ (np.exp(offset + x @ b) - y)

    b = np.zeros(x.shape[1])
    b[0] = np.log(sum(y) / sum(np.exp(offset)))
    fit = minimize(obj, b, jac=jac, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
    assert np.linalg.norm(jac(fit.x), np.inf) < 2e-5
    mu = np.exp(offset + x @ fit.x)
    info = x.T @ (mu[:, None] * x)
    return fit.x, mu, np.linalg.inv(info), -obj(fit.x)


def test_independence_closed_form_and_full_inference():
    data, dims, levels = grid(counts=[12, 24, 15, 20, 10, 32])
    fit = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]])
    observed = data["count"].values.reshape(2, 3)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    np.testing.assert_allclose(fit["cells"].fitted, expected.ravel(), rtol=1e-9)
    x = dummy(data, dims, levels)
    beta, mu, cov, ll = oracle(x, data["count"].values)
    np.testing.assert_allclose(fit["parameters"].estimate, beta, atol=2e-7)
    np.testing.assert_allclose(fit["covariance"], cov, rtol=2e-7, atol=1e-9)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    np.testing.assert_allclose(fit["parameters"].std_error, se, rtol=2e-7)
    np.testing.assert_allclose(fit["parameters"].p_value, 2 * norm.sf(abs(z)), atol=1e-7)
    np.testing.assert_allclose(fit["parameters"].ci_lower, beta - norm.ppf(0.975) * se, atol=2e-7)
    assert fit.attrs["df_resid"] == 2
    assert fit.attrs["log_likelihood"] == pytest.approx(ll, abs=1e-8)
    dev = 2 * np.sum(observed.ravel() * np.log(observed.ravel() / mu) - (observed.ravel() - mu))
    assert fit["goodness_of_fit"].loc["deviance", "statistic"] == pytest.approx(dev)
    assert fit["goodness_of_fit"].loc["deviance", "p_value"] == pytest.approx(chi2.sf(dev, 2))
    covariance_mean = mu**2 * np.einsum("ij,jk,ik->i", x, cov, x)
    np.testing.assert_allclose(fit["cells"].fitted_se, np.sqrt(covariance_mean), rtol=1e-7)


def test_conditional_independence_decomposable_formula():
    data, dims, levels = grid((2, 3, 2))
    fit = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a", "b"], ["b", "c"]])
    y = data["count"].values.reshape(2, 3, 2)
    expected = y.sum(2)[:, :, None] * y.sum(0)[None, :, :] / y.sum((0, 2))[None, :, None]
    np.testing.assert_allclose(fit["cells"].fitted, expected.ravel(), rtol=1e-9)
    assert fit.attrs["df_resid"] == 3
    assert fit.attrs["generating_margins"] == [["a", "b"], ["b", "c"]]


def test_structural_zero_support_rank_and_sampling_zero():
    data, dims, levels = grid((3, 3), [0, 12, 18, 21, 0, 17, 19, 13, 0])
    data["structural"] = data.a == data.b
    fit = loglinear_ipf(
        data, dims, "count", levels=levels, margins=[["a"], ["b"]], structural="structural"
    )
    active = ~data.structural
    x = dummy(data, dims, levels)[active]
    beta, mu, cov, ll = oracle(x, data.loc[active, "count"].values)
    assert fit.attrs["df_resid"] == int(active.sum() - np.linalg.matrix_rank(x)) == 1
    assert fit["cells"].loc[data.structural, "fitted"].eq(0).all()
    np.testing.assert_allclose(fit["cells"].loc[active, "fitted"], mu, rtol=1e-8)
    np.testing.assert_allclose(fit["covariance"], cov, rtol=1e-7)
    assert fit.attrs["log_likelihood"] == pytest.approx(ll, abs=1e-8)
    # A sampling zero has positive fitted mass, distinct from structural absence.
    data, dims, levels = grid((2, 3), [0, 15, 17, 22, 23, 20])
    fit = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]])
    assert fit["cells"].fitted.iloc[0] > 0
    assert fit.attrs["structural_zero_cells"] == 0


def test_support_aliases_and_zero_residual_df():
    data, dims, levels = grid((2, 2), [10, 0, 0, 20])
    data["s"] = [False, True, True, False]
    fit = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]], structural="s")
    assert fit.attrs["aliased_terms"] == ["b[1]"]
    assert fit.attrs["df_resid"] == 0
    assert fit["goodness_of_fit"].p_value.isna().all()
    np.testing.assert_allclose(fit["cells"].fitted, [10, 0, 0, 20])


def test_ml_offsets_permuted_rows_independent_oracle_and_full_hessian():
    data, dims, levels = grid((2, 3), [12, 24, 15, 20, 10, 32])
    data["off"] = [-0.4, 0.1, 0.3, 0.2, -0.2, 0.6]
    x = dummy(data, dims, levels)
    order = [5, 1, 3, 0, 4, 2]
    shuffled = data.iloc[order].copy()
    fit = loglinear_ml(
        shuffled,
        dims,
        "count",
        levels=levels,
        design=x[order],
        offset="off",
        terms=["i", "a1", "b1", "b2"],
    )
    beta, mu, cov, ll = oracle(x, data["count"].values, data.off.values)
    np.testing.assert_allclose(fit["parameters"].estimate, beta, atol=2e-7)
    np.testing.assert_allclose(fit["cells"].fitted, mu, rtol=2e-8)
    np.testing.assert_allclose(fit["covariance"], cov, rtol=3e-8)
    assert fit.attrs["sample_positions"] == [order.index(i) for i in range(6)]
    assert fit.attrs["log_likelihood"] == pytest.approx(ll, abs=1e-8)

    # Full finite-difference Hessian of independent explicit likelihood.
    def score(b):
        return x.T @ (np.exp(data.off.values + x @ b) - data["count"].values)

    step = 1e-5
    h = np.column_stack(
        [
            (score(beta + np.eye(4)[j] * step) - score(beta - np.eye(4)[j] * step)) / (2 * step)
            for j in range(4)
        ]
    )
    np.testing.assert_allclose(fit["information"], h, rtol=2e-8, atol=1e-8)


def test_ipf_known_offset_matches_ml():
    data, dims, levels = grid((2, 3))
    data["off"] = [-0.4, 0.1, 0.3, 0.2, -0.2, 0.6]
    x = dummy(data, dims, levels)
    ipf = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]], offset="off")
    ml = loglinear_ml(data, dims, "count", levels=levels, design=x, offset="off")
    np.testing.assert_allclose(ipf["cells"].fitted, ml["cells"].fitted, rtol=2e-8)


def test_nested_compare_restored_complete_states():
    data, dims, levels = grid((2, 3), [12, 24, 15, 20, 10, 32])
    small = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]])
    full = loglinear_ml(data, dims, "count", levels=levels, design=np.eye(6))
    for fitted in (small, full):
        state = summary_state(fitted)
        restored = restore_summary(state)
        assert summary_state(restored) == state
        assert restored.to_latex() == fitted.to_latex()
        for name in fitted:
            pd.testing.assert_frame_equal(fitted[name], restored[name], check_dtype=False)
    compared = loglinear_compare(
        restore_summary(summary_state(small)), restore_summary(summary_state(full))
    )
    assert compared.attrs["df"] == 2
    assert compared.attrs["statistic"] == pytest.approx(
        small["goodness_of_fit"].loc["deviance", "statistic"], abs=1e-8
    )
    assert compared.attrs["p_value"] == pytest.approx(chi2.sf(compared.attrs["statistic"], 2))


def test_nestedness_requires_design_span_and_same_grid():
    data, dims, levels = grid((2, 3))
    x = dummy(data, dims, levels)
    small = loglinear_ml(
        data, dims, "count", levels=levels, design=np.column_stack([np.ones(6), [0, 0, 0, 0, 0, 1]])
    )
    full = loglinear_ml(data, dims, "count", levels=levels, design=x)
    with pytest.raises(AnalysisError, match="not nested"):
        loglinear_compare(small, full)
    altered = data.copy()
    altered.loc[0, "count"] += 1
    other = loglinear_ml(altered, dims, "count", levels=levels, design=np.eye(6))
    with pytest.raises(AnalysisError, match="same cell"):
        loglinear_compare(full, other)
    with pytest.raises(AnalysisError, match="add identified"):
        loglinear_compare(full, full)


@pytest.mark.parametrize("mutation", ["counts", "means", "covariance", "coefficients", "offset"])
def test_tampered_restored_states_refused(mutation):
    data, dims, levels = grid()
    fit = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]])
    full = loglinear_ml(data, dims, "count", levels=levels, design=np.eye(6))
    altered = restore_summary(summary_state(fit))
    s = altered.attrs["state"]
    if mutation == "counts":
        s["grid"]["counts"][0] += 1
    elif mutation == "offset":
        s["grid"]["offset"][0] += 0.1
    elif mutation == "means":
        s["fitted_active"][0] *= 2
    elif mutation == "covariance":
        s["covariance"][0][0] *= 2
    else:
        s["coefficients"][0] += 0.1
    with pytest.raises(AnalysisError, match="checksum"):
        loglinear_compare(altered, full)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "duplicate",
        "count_negative",
        "count_fraction",
        "structural_count",
        "bad_levels",
        "undeclared",
        "incomplete",
        "boolean_count",
    ],
)
def test_invalid_grid_no_implicit_drop_or_pseudocount(bad):
    data, dims, levels = grid()
    kwargs = {}
    if bad == "missing":
        data.loc[0, "count"] = np.nan
    elif bad == "duplicate":
        data.loc[1, dims] = data.loc[0, dims].values
    elif bad == "count_negative":
        data.loc[0, "count"] = -1
    elif bad == "count_fraction":
        data["count"] = data["count"].astype(float)
        data.loc[0, "count"] = 1.5
    elif bad == "structural_count":
        data["s"] = False
        data.loc[0, "s"] = True
        kwargs["structural"] = "s"
    elif bad == "bad_levels":
        levels["a"] = [0, 0]
    elif bad == "undeclared":
        data.loc[0, "a"] = 7
    elif bad == "incomplete":
        data = data.iloc[:-1]
    elif bad == "boolean_count":
        data["count"] = True
    with pytest.raises(AnalysisError):
        loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]], **kwargs)


def test_nonconvergence_boundary_rank_resource_and_global_admission():
    data, dims, levels = grid((2, 2, 2), [0, 1, 2, 33, 10, 17, 5, 100])
    with pytest.raises(AnalysisError, match="did not attain"):
        loglinear_ipf(
            data,
            dims,
            "count",
            levels=levels,
            margins=[["a", "b"], ["a", "c"], ["b", "c"]],
            max_iter=1,
            tol=1e-12,
        )
    with pytest.raises(AnalysisError, match="rank-deficient"):
        loglinear_ml(data, dims, "count", levels=levels, design=np.ones((8, 2)))
    boundary, d, declared = grid((2, 2), [0, 0, 10, 12])
    with pytest.raises(AnalysisError, match="boundary"):
        loglinear_ipf(boundary, d, "count", levels=declared, margins=[["a"], ["b"]])
    with pytest.raises(AnalysisError, match="interior"):
        loglinear_ml(boundary, d, "count", levels=declared, design=np.eye(4))
    for kw in ({"max_work": 1}, {"max_bytes": 1}):
        with pytest.raises(AnalysisError):
            loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"], ["c"]], **kw)
    large, d, declared = grid((16, 16))
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        loglinear_ml(
            large,
            d,
            "count",
            levels=declared,
            design=np.eye(256)[:, :128],
            max_iter=20,
            max_work=2_000_000_000,
        )


def test_cpu_float64_ignores_caller_default_device_and_dtype():
    data, dims, levels = grid()
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            result = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]])
        assert result.attrs["device"] == "cpu" and result.attrs["precision"] == "float64"
        assert np.asarray(result["covariance"]).dtype == np.float64
    finally:
        torch.set_default_dtype(prior)


def test_rehashed_semantically_inconsistent_state_refused():
    data, dims, levels = grid()
    fit = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]])
    full = loglinear_ml(data, dims, "count", levels=levels, design=np.eye(6))
    bad = restore_summary(summary_state(fit))
    s = bad.attrs["state"]
    s["fitted_active"][0] *= 1.1
    s["sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in s.items() if k != "sha256"},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(AnalysisError, match="disagree"):
        loglinear_compare(bad, full)


def test_positive_margins_do_not_certify_interior_ipf_mle():
    data, dims, levels = grid((2, 2, 2), [0, 1, 1, 1, 1, 1, 1, 0])
    a, b, c = [data[d].to_numpy() for d in dims]
    separating = -1 + a + b + c - a * b - a * c - b * c
    assert np.all(separating <= 0)
    assert np.all(separating[data["count"].to_numpy() > 0] == 0)
    assert np.all(separating[data["count"].to_numpy() == 0] < 0)
    # This direction strictly increases likelihood for every finite beta.
    # All observed pairwise margins are positive nonetheless.
    for group in (["a", "b"], ["a", "c"], ["b", "c"]):
        assert (data.groupby(group)["count"].sum() > 0).all()
    with pytest.raises(AnalysisError, match="finite stationary interior"):
        loglinear_ipf(
            data,
            dims,
            "count",
            levels=levels,
            margins=[["a", "b"], ["a", "c"], ["b", "c"]],
            tol=1e-4,
            max_iter=2000,
        )


def test_scaled_design_large_count_does_not_mask_zero_cell_divergence():
    data, dims, levels = grid((2, 2), [0, 10_000_000, 2, 3])
    with pytest.raises(AnalysisError, match="interior"):
        loglinear_ml(data, dims, "count", levels=levels, design=np.diag([1.0, 1e-4, 1e-4, 1e-4]))


def test_coherently_rehashed_nonstationary_fit_is_not_comparison_evidence():
    data, dims, levels = grid((2, 3), [12, 24, 15, 20, 10, 32])
    small = loglinear_ipf(data, dims, "count", levels=levels, margins=[["a"], ["b"]])
    full = restore_summary(
        summary_state(loglinear_ml(data, dims, "count", levels=levels, design=np.eye(6)))
    )
    state = full.attrs["state"]
    state["coefficients"][0] += 0.1
    x = np.array(state["design"])
    b = np.array(state["coefficients"])
    y = data["count"].to_numpy()
    mu = np.exp(x @ b)
    information = x.T @ (mu[:, None] * x)
    state["fitted_active"] = mu.tolist()
    state["information"] = information.tolist()
    state["covariance"] = np.linalg.inv(information).tolist()
    state["log_likelihood"] = float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
    state["deviance"] = float(2 * np.sum(y * np.log(y / mu) - (y - mu)))
    state["sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in state.items() if k != "sha256"},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(AnalysisError, match="stationary Poisson MLE"):
        loglinear_compare(small, full)


def test_bad_raw_design_width_is_rejected_before_conversion(monkeypatch):
    data, dims, levels = grid()
    original = torch.as_tensor
    converted = []

    def spy(value, *args, **kwargs):
        converted.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", spy)
    with pytest.raises(AnalysisError, match="column width"):
        loglinear_ml(data, dims, "count", levels=levels, design=[[1], [1], [1], [1], [1], [1, 2]])
    assert converted == []


def test_nondecomposable_positive_ipf_independent_likelihood_and_covariance():
    data, dims, levels = grid((2, 2, 2), [7, 11, 19, 13, 31, 17, 23, 29])
    fit = loglinear_ipf(
        data, dims, "count", levels=levels, margins=[["a", "b"], ["a", "c"], ["b", "c"]]
    )
    a, b, c = [data[d].to_numpy() for d in dims]
    independent_design = np.column_stack([np.ones(8), a, b, c, a * b, a * c, b * c])
    beta, mu, covariance, likelihood = oracle(independent_design, data["count"].to_numpy())
    np.testing.assert_allclose(fit["cells"].fitted, mu, rtol=1e-8)
    np.testing.assert_allclose(fit["parameters"].estimate, beta, atol=2e-8)
    np.testing.assert_allclose(fit["covariance"], covariance, rtol=1e-8, atol=2e-10)
    assert fit.attrs["log_likelihood"] == pytest.approx(likelihood, abs=1e-8)
    assert fit.attrs["df_resid"] == 1
