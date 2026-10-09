"""Independent conditional likelihood/Hessian and hierarchy path acceptance."""

from itertools import product, combinations
import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2, norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical.sampling import (
    loglinear_multinomial,
    loglinear_product_multinomial,
    loglinear_sampling_compare,
)
from openecon.econometrics.categorical.selection import loglinear_select
from openecon.econometrics.categorical.loglinear import loglinear_ipf, _digest
from openecon.econometrics.summary_state import summary_state, restore_summary
from openecon.resources import use_workspace_budget


def cells(shape=(2, 3), counts=None):
    dims = list("abcd")[: len(shape)]
    data = pd.DataFrame(product(*(range(k) for k in shape)), columns=dims)
    data["n"] = counts if counts is not None else [10 + 3 * i + i % 3 for i in range(len(data))]
    levels = {d: list(range(k)) for d, k in zip(dims, shape)}
    x = np.column_stack(
        [(data[d].to_numpy() == k).astype(float) for d in dims for k in levels[d][1:]]
    )
    return data, dims, levels, x


def oracle(x, y, ids=None, off=None):
    ids = np.zeros(len(y), dtype=int) if ids is None else np.asarray(ids)
    off = np.zeros(len(y)) if off is None else np.asarray(off)
    groups = [np.flatnonzero(ids == k) for k in sorted(set(ids))]
    totals = np.array([y[i].sum() for i in groups])

    def evaluate(b):
        eta = off + x @ b
        lp = np.empty(len(y))
        mu = np.empty(len(y))
        info = np.zeros((x.shape[1], x.shape[1]))
        derivative = np.empty_like(x)
        for idx, N in zip(groups, totals):
            lp[idx] = eta[idx] - logsumexp(eta[idx])
            p = np.exp(lp[idx])
            mu[idx] = N * p
            z = x[idx] - p @ x[idx]
            info += z.T @ ((N * p)[:, None] * z)
            derivative[idx] = N * p[:, None] * z
        ll = np.sum(y * lp - gammaln(y + 1)) + gammaln(totals + 1).sum()
        return ll, mu, info, derivative, lp

    fit = minimize(
        lambda b: -evaluate(b)[0],
        np.zeros(x.shape[1]),
        jac=lambda b: x.T @ (evaluate(b)[1] - y),
        hess=lambda b: evaluate(b)[2],
        method="trust-exact",
        options={"gtol": 1e-9, "maxiter": 1000},
    )
    assert np.max(abs(x.T @ (evaluate(fit.x)[1] - y))) < 1e-5
    return fit.x, *evaluate(fit.x)


def test_multinomial_independence_full_joint_inference():
    data, dims, levels, x = cells(counts=[12, 24, 15, 20, 10, 32])
    fit = loglinear_multinomial(data, dims, "n", levels=levels, design=x)
    y = data.n.to_numpy()
    b, ll, mu, info, D, lp = oracle(x, y)
    expected = np.outer(y.reshape(2, 3).sum(1), y.reshape(2, 3).sum(0)) / sum(y)
    np.testing.assert_allclose(fit["cells"].fitted, expected.ravel(), rtol=1e-8)
    np.testing.assert_allclose(fit["parameters"].estimate, b, atol=2e-7)
    np.testing.assert_allclose(fit["information"], info, rtol=1e-7, atol=1e-10)
    cov = np.linalg.inv(info)
    np.testing.assert_allclose(fit["covariance"], cov, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(fit["parameters"].std_error, np.sqrt(cov.diagonal()), rtol=1e-7)
    np.testing.assert_allclose(
        fit["parameters"].p_value, 2 * norm.sf(abs(b / np.sqrt(cov.diagonal()))), atol=1e-7
    )
    np.testing.assert_allclose(
        fit["parameters"].ci_lower, b - norm.ppf(0.975) * np.sqrt(cov.diagonal()), atol=2e-7
    )
    np.testing.assert_allclose(
        fit["cells"].fitted_se, np.sqrt(np.einsum("ij,jk,ik->i", D, cov, D)), rtol=1e-7
    )
    assert fit.attrs["log_likelihood"] == pytest.approx(ll, abs=1e-8)
    assert fit.attrs["df_resid"] == 2
    assert sum(fit["cells"].fitted) == pytest.approx(sum(y))
    assert sum(fit["cells"].probability) == pytest.approx(1)


def test_saturated_multinomial_closed_form_and_nested_lr():
    data, dims, levels, x = cells(counts=[12, 24, 15, 20, 10, 32])
    small = loglinear_multinomial(data, dims, "n", levels=levels, design=x)
    saturated = np.eye(len(data))[:, 1:]
    full = loglinear_multinomial(data, dims, "n", levels=levels, design=saturated)
    np.testing.assert_allclose(full["cells"].fitted, data.n, atol=2e-7)
    np.testing.assert_allclose(
        full["parameters"].estimate, np.log(data.n.to_numpy()[1:] / data.n.iloc[0]), atol=2e-8
    )
    expected = np.diag(1 / data.n.to_numpy()[1:]) + np.ones((5, 5)) / data.n.iloc[0]
    np.testing.assert_allclose(full["covariance"], expected, rtol=1e-7)
    result = loglinear_sampling_compare(
        restore_summary(summary_state(small)), restore_summary(summary_state(full))
    )
    dev = 2 * np.sum(data.n * np.log(data.n / small["cells"].fitted))
    assert result.attrs["statistic"] == pytest.approx(dev, abs=1e-7)
    assert result.attrs["df"] == 2
    assert result.attrs["p_value"] == pytest.approx(chi2.sf(dev, 2))
    assert full.attrs["df_resid"] == 0
    assert full["goodness_of_fit"].p_value.isna().all()


def test_product_multinomial_common_binomial_odds_and_covariance():
    data, dims, levels, _ = cells((3, 2), [12, 25, 19, 33, 21, 14])
    x = (data.b.to_numpy() == 1).astype(float)[:, None]
    fit = loglinear_product_multinomial(
        data, dims, "n", levels=levels, conditioning=["a"], design=x
    )
    success = data.n[data.b == 1].sum()
    failure = data.n[data.b == 0].sum()
    odds = np.log(success / failure)
    assert fit["parameters"].estimate.iloc[0] == pytest.approx(odds, abs=1e-8)
    assert fit["covariance"].iloc[0, 0] == pytest.approx(1 / success + 1 / failure, abs=1e-9)
    np.testing.assert_allclose(fit["strata"].observed_total, fit["strata"].fitted_total, rtol=1e-12)
    b, ll, mu, info, D, lp = oracle(x, data.n.to_numpy(), data.a.to_numpy())
    np.testing.assert_allclose(fit["cells"].fitted, mu, rtol=1e-8)
    assert fit.attrs["log_likelihood"] == pytest.approx(ll, abs=1e-8)
    assert fit.attrs["df_resid"] == 2


def test_product_structural_support_offsets_row_reordering_and_lr():
    data, dims, levels, _ = cells((3, 3), [12, 14, 0, 18, 25, 13, 11, 9, 15])
    data["s"] = [False, False, True, False, False, False, False, False, False]
    data["off"] = np.linspace(-0.3, 0.4, len(data))
    x = np.column_stack([(data.b == j).astype(float) for j in [1, 2]])
    order = [8, 5, 3, 2, 0, 7, 6, 4, 1]
    fit = loglinear_product_multinomial(
        data.iloc[order],
        dims,
        "n",
        levels=levels,
        conditioning=["a"],
        design=x[order],
        structural="s",
        offset="off",
    )
    active = ~data.s
    b, ll, mu, info, D, lp = oracle(
        x[active], data.n.to_numpy()[active], data.a.to_numpy()[active], data.off.to_numpy()[active]
    )
    np.testing.assert_allclose(fit["parameters"].estimate, b, atol=2e-7)
    np.testing.assert_allclose(fit["covariance"], np.linalg.inv(info), rtol=1e-7)
    assert fit["cells"].fitted.iloc[2] == 0
    assert fit.attrs["df_resid"] == 3
    small = loglinear_product_multinomial(
        data,
        dims,
        "n",
        levels=levels,
        conditioning=["a"],
        design=x[:, :1],
        structural="s",
        offset="off",
    )
    assert loglinear_sampling_compare(small, fit).attrs["df"] == 1


@pytest.mark.parametrize("product_sampling", [False, True])
def test_full_summary_and_latex_roundtrip(product_sampling):
    data, dims, levels, x = cells()
    fit = (
        loglinear_product_multinomial(
            data, dims, "n", levels=levels, conditioning=["a"], design=x[:, 1:]
        )
        if product_sampling
        else loglinear_multinomial(data, dims, "n", levels=levels, design=x)
    )
    encoded = json.loads(json.dumps(summary_state(fit)))
    restored = restore_summary(encoded)
    assert summary_state(restored) == encoded
    assert list(restored) == list(fit)
    for name in fit:
        pd.testing.assert_frame_equal(fit[name], restored[name], check_dtype=False)
        assert fit[name].to_latex() == restored[name].to_latex()


@pytest.mark.parametrize(
    "problem",
    [
        "intercept",
        "stratum_constant",
        "empty",
        "wrong_conditioning",
        "design_rows",
        "levels",
        "missing",
        "fractional",
        "positive_structural",
        "zero_total",
        "rank",
    ],
)
def test_invalid_sampling_domains(problem):
    data, dims, levels, x = cells()
    kwargs = dict(levels=levels, conditioning=["a"], design=x[:, 1:])
    if problem == "intercept":
        kwargs["design"] = np.column_stack([np.ones(len(data)), x[:, 1:]])
    if problem == "stratum_constant":
        kwargs["design"] = x
    if problem == "empty":
        data.loc[data.a == 0, "n"] = 0
    if problem == "wrong_conditioning":
        kwargs["conditioning"] = ["a", "b"]
    if problem == "design_rows":
        kwargs["design"] = x[:-1, 1:]
    if problem == "levels":
        kwargs["levels"] = {"a": [0, 1]}
    if problem == "missing":
        data.loc[0, "n"] = np.nan
    if problem == "fractional":
        data["n"] = data.n.astype(float)
        data.loc[0, "n"] = 0.5
    if problem == "positive_structural":
        data["s"] = [True] + [False] * 5
        kwargs["structural"] = "s"
    if problem == "zero_total":
        data["n"] = 0
    if problem == "rank":
        kwargs["design"] = np.column_stack([x[:, 1], x[:, 1]])
    with pytest.raises(AnalysisError):
        loglinear_product_multinomial(data, dims, "n", **kwargs)


def test_nonconvergence_boundary_work_memory_and_default_device():
    data, dims, levels, x = cells()
    for kw in [dict(max_iter=1), dict(max_work=1), dict(max_bytes=1)]:
        with pytest.raises(AnalysisError):
            loglinear_multinomial(data, dims, "n", levels=levels, design=x, **kw)
    larger, dlarge, llarge, xlarge = cells((10, 10, 10))
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError, match="workspace"):
            loglinear_multinomial(
                larger,
                dlarge,
                "n",
                levels=llarge,
                design=xlarge,
                max_iter=10,
                max_work=2_000_000_000,
            )
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        assert (
            loglinear_multinomial(data, dims, "n", levels=levels, design=x).attrs["device"] == "cpu"
        )
    finally:
        torch.set_default_device(previous)
    data.n = [0, 10, 20, 30, 40, 50]
    with pytest.raises(AnalysisError):
        loglinear_multinomial(data, dims, "n", levels=levels, design=np.eye(6)[:, 1:])


def test_sampling_zero_keeps_positive_probability():
    data, dims, levels, x = cells(counts=[0, 20, 14, 16, 18, 9])
    fit = loglinear_multinomial(data, dims, "n", levels=levels, design=x)
    assert fit["cells"].fitted.iloc[0] > 0


def test_conditional_nestedness_quotients_out_different_stratum_shifts():
    data, dims, levels, _ = cells((3, 3))
    response = np.column_stack([(data.b == j).astype(float) for j in [1, 2]])
    shifted = response[:, :1] + 0.25 * data.a.to_numpy()[:, None]
    common = dict(levels=levels, conditioning=["a"])
    small = loglinear_product_multinomial(data, dims, "n", design=shifted, **common)
    reference = loglinear_product_multinomial(data, dims, "n", design=response[:, :1], **common)
    full = loglinear_product_multinomial(data, dims, "n", design=response, **common)
    np.testing.assert_allclose(small["cells"].fitted, reference["cells"].fitted, atol=1e-9)
    assert loglinear_sampling_compare(small, full).attrs["statistic"] == pytest.approx(
        loglinear_sampling_compare(reference, full).attrs["statistic"], abs=1e-9
    )


@pytest.mark.parametrize(
    "mutation", ["checksum", "coherent_beta", "covariance", "strata", "counts", "shape"]
)
def test_restored_model_semantic_corruption_refused(mutation):
    data, dims, levels, x = cells()
    small = loglinear_multinomial(data, dims, "n", levels=levels, design=x)
    full = loglinear_multinomial(data, dims, "n", levels=levels, design=np.eye(6)[:, 1:])
    bad = copy.deepcopy(full)
    s = bad.attrs["state"]
    if mutation == "checksum":
        s["coefficients"][0] += 0.1
    else:
        if mutation == "coherent_beta":
            s["coefficients"][0] += 0.1
            X = np.array(s["design"])
            b = np.array(s["coefficients"])
            y = np.array(s["grid"]["counts"])
            p = np.exp(X @ b - logsumexp(X @ b))
            mu = y.sum() * p
            z = X - p @ X
            info = z.T @ (mu[:, None] * z)
            s.update(
                probabilities=p.tolist(),
                fitted=mu.tolist(),
                information=info.tolist(),
                covariance=np.linalg.inv(info).tolist(),
                log_likelihood=float(np.sum(y * np.log(p) - gammaln(y + 1)) + gammaln(sum(y) + 1)),
            )
        if mutation == "covariance":
            s["covariance"][0][0] += 0.1
        if mutation == "strata":
            s["totals"][0] += 1
        if mutation == "counts":
            s["grid"]["counts"][0] += 1
        if mutation == "shape":
            s["design"] = [row + [1] for row in s["design"]]
        s["sha256"] = _digest({k: v for k, v in s.items() if k != "sha256"})
    with pytest.raises(AnalysisError):
        loglinear_sampling_compare(small, bad)


def test_lr_mismatched_sampling_and_non_nested_spaces_refused():
    data, dims, levels, x = cells()
    a = loglinear_multinomial(data, dims, "n", levels=levels, design=x[:, :1])
    b = loglinear_multinomial(data, dims, "n", levels=levels, design=x[:, 1:])
    with pytest.raises(AnalysisError):
        loglinear_sampling_compare(a, b)
    c = loglinear_product_multinomial(
        data, dims, "n", levels=levels, conditioning=["a"], design=x[:, 1:]
    )
    with pytest.raises(AnalysisError):
        loglinear_sampling_compare(a, c)


@pytest.mark.parametrize(
    "mutation",
    [
        "wide_rows",
        "oversized_vector",
        "extra_field",
        "terms_duplicate",
        "probability_text",
        "matrix_nested",
    ],
)
def test_saved_shape_and_scalar_admission_precedes_digest(monkeypatch, mutation):
    from openecon.econometrics.categorical import sampling

    data, dims, levels, x = cells()
    small = loglinear_multinomial(data, dims, "n", levels=levels, design=x)
    bad = copy.deepcopy(small)
    state = bad.attrs["state"]
    if mutation == "wide_rows":
        state["design"][0] *= 1000
    elif mutation == "oversized_vector":
        state["probabilities"] *= 1000
    elif mutation == "extra_field":
        state["unexpected"] = [0] * 10000
    elif mutation == "terms_duplicate":
        state["terms"][1] = state["terms"][0]
    elif mutation == "matrix_nested":
        state["covariance"][0][0] = [0.0] * 10000
    else:
        state["probabilities"][0] = "x" * 10000

    def no_digest(value):
        raise AssertionError("Unadmitted saved state was serialized")

    monkeypatch.setattr(sampling, "_digest", no_digest)
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Unadmitted nested scalar reached tensor allocation")
        ),
    )
    with pytest.raises(AnalysisError):
        sampling._checked(bad, 128 * 1024**2)


@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_selection_independent_hierarchy_path_and_exact_restore(criterion):
    data, dims, levels, _ = cells((2, 2, 2), [25, 32, 19, 31, 21, 22, 28, 29])
    result = loglinear_select(
        data,
        dims,
        "n",
        levels=levels,
        margins=[dims],
        criterion=criterion,
        max_iter=100,
        max_work=100_000_000,
    )
    terms = {s for k in range(1, 4) for s in combinations(dims, k)}

    def fit(ts):
        columns = [np.ones(len(data))]
        for t in sorted(ts, key=lambda t: (len(t), t)):
            columns.append(np.prod([(data[d] == 1).astype(float) for d in t], axis=0))
        X = np.column_stack(columns)
        y = data.n.to_numpy()
        opt = minimize(
            lambda b: np.sum(np.exp(X @ b) - y * (X @ b) + gammaln(y + 1)),
            np.zeros(X.shape[1]),
            jac=lambda b: X.T @ (np.exp(X @ b) - y),
            method="BFGS",
            options={"gtol": 1e-7},
        )
        assert np.max(abs(X.T @ (np.exp(X @ opt.x) - y))) < 2e-5
        ll = -opt.fun
        return -2 * ll + (2 if criterion == "aic" else np.log(sum(y))) * X.shape[1]

    current = fit(terms)
    while True:
        eligible = [t for t in terms if len(t) > 1 and not any(set(t) < set(o) for o in terms)]
        if not eligible:
            break
        candidates = sorted([(fit(terms - {t}), t) for t in eligible])
        value, term = candidates[0]
        if value >= current - 1e-8:
            break
        current = value
        terms.remove(term)
    expected = [list(t) for t in terms if not any(set(t) < set(o) for o in terms)]
    assert sorted(result.attrs["selected_margins"]) == sorted(expected)
    assert result.attrs["selected_criterion"] == pytest.approx(current, abs=2e-7)
    state = summary_state(result)
    restored = restore_summary(json.loads(json.dumps(state)))
    assert summary_state(restored) == state
    for name in result:
        pd.testing.assert_frame_equal(result[name], restored[name], check_dtype=False)
        assert result[name].to_latex() == restored[name].to_latex()
    selected = restore_summary(result.attrs["state"]["selected_summary"])
    np.testing.assert_allclose(result["selected_cells"].fitted, selected["cells"].fitted)
    for candidate in result.attrs["state"]["candidates"]:
        model = restore_summary(candidate)
        margins = model.attrs["generating_margins"]
        assert set().union(*map(set, margins)) == set(dims)


def test_selection_no_interactions_resource_and_candidate_failure():
    data, dims, levels, _ = cells()
    result = loglinear_select(data, dims, "n", levels=levels, margins=[["a"], ["b"]])
    assert result.attrs["termination"] == "main_effects_only"
    assert result.attrs["evaluated_models"] == 1
    for kw in [dict(max_models=2), dict(max_work=1), dict(max_bytes=1)]:
        if "max_models" in kw:
            triple, d3, l3, _ = cells((2, 2, 2))
            with pytest.raises(AnalysisError):
                loglinear_select(triple, d3, "n", levels=l3, margins=[d3], **kw)
        else:
            with pytest.raises(AnalysisError):
                loglinear_select(data, dims, "n", levels=levels, margins=[dims], **kw)
    data.loc[0, "n"] = 0
    with pytest.raises(AnalysisError):
        loglinear_select(data, dims, "n", levels=levels, margins=[dims])


def test_selection_structural_support_rank_matches_explicit_candidates():
    data, dims, levels, _ = cells((3, 3), [0, 12, 18, 21, 0, 17, 19, 13, 0])
    data["s"] = data.a == data.b
    selected = loglinear_select(
        data, dims, "n", levels=levels, margins=[dims], structural="s", max_iter=100
    )
    for state in selected.attrs["state"]["candidates"]:
        candidate = restore_summary(state)
        check = loglinear_ipf(
            data,
            dims,
            "n",
            levels=levels,
            margins=candidate.attrs["generating_margins"],
            structural="s",
            max_iter=100,
        )
        assert candidate.attrs["df_resid"] == check.attrs["df_resid"]
        np.testing.assert_allclose(candidate["cells"].fitted, check["cells"].fitted)


def test_selection_level_parameter_and_work_admission_precedes_fit(monkeypatch):
    from openecon.econometrics.categorical import selection

    def no_fit(*args, **kwargs):
        raise AssertionError("Unadmitted candidate reached numerical fit")

    monkeypatch.setattr(selection, "loglinear_ipf", no_fit)
    for levels in (
        {"a": list(range(1000)), "b": [0, 1]},
        {"a": list(range(16)), "b": list(range(16))},
    ):
        with pytest.raises(AnalysisError):
            loglinear_select([], ["a", "b"], "n", levels=levels, margins=[["a", "b"]])
    with pytest.raises(AnalysisError):
        loglinear_select(
            [], ["a", "b"], "n", levels={"a": [0, 1], "b": [0, 1]}, margins=[["a", "b"]], max_work=1
        )
