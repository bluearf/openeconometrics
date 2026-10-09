"""Independent analytic/SciPy/tail/convolution/exhaustive oracles and failure contracts."""

import itertools
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def group_data(a=0, n0=10, b=9, n1=10, shift=0.0, scale=1.0):
    return pd.DataFrame(
        {
            "y": [1] * a + [0] * (n0 - a) + [1] * b + [0] * (n1 - b),
            "x": [shift] * n0 + [shift + scale] * n1,
        },
        index=[f"row-{i}" for i in range(n0 + n1)],
    )


def raw_objective(beta, x, y):
    eta = x @ beta
    pr = special.expit(eta)
    info = x.T @ (pr[:, None] * (1 - pr[:, None]) * x)
    sign, logdet = np.linalg.slogdet(info)
    return -(np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * logdet) if sign > 0 else np.inf


@pytest.mark.parametrize(
    "a,b,n0,n1", [(0, 9, 10, 10), (0, 10, 10, 10), (3, 7, 12, 15), (10, 0, 10, 10)]
)
def test_firth_saturated_closed_form_all_covariance_wald_and_sample(a, b, n0, n1):
    frame = group_data(a, n0, b, n1)
    r = oe.firth_logit(frame, "y", ["x"], level=0.9)
    p0, p1 = (a + 0.5) / (n0 + 1), (b + 0.5) / (n1 + 1)
    beta = np.array([special.logit(p0), special.logit(p1) - special.logit(p0)])
    v0, v1 = 1 / (n0 * p0 * (1 - p0)), 1 / (n1 * p1 * (1 - p1))
    cov = np.array([[v0, -v0], [-v0, v0 + v1]])
    np.testing.assert_allclose(r["coefficients"].estimate, beta, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(r["covariance"], cov, rtol=1e-8)
    se = np.sqrt(cov.diagonal())
    np.testing.assert_allclose(r["coefficients"].std_error, se, rtol=1e-8)
    np.testing.assert_allclose(r["coefficients"].z, beta / se, rtol=1e-8)
    np.testing.assert_allclose(
        r["coefficients"].p_value, 2 * stats.norm.sf(np.abs(beta / se)), rtol=1e-8
    )
    np.testing.assert_allclose(
        r["coefficients"].ci_low, beta - stats.norm.ppf(0.95) * se, rtol=1e-8
    )
    s = r.attrs["state"]
    assert (
        s["sample_positions"] == list(range(len(frame)))
        and s["sample_labels"] == frame.index.tolist()
    )
    assert s["convergence"]["converged"] and s["convergence"]["free_score_max"] <= s["tol"]


@pytest.mark.parametrize("events", [0, 1, 5, 10])
def test_intercept_only_all_zero_all_one_honest_finite_fit(events):
    r = oe.firth_logit({"y": [1] * events + [0] * (10 - events)}, "y")
    assert r["coefficients"].estimate.iloc[0] == pytest.approx(
        special.logit((events + 0.5) / 11), abs=1e-8
    )


def test_firth_numeric_nonsaturated_matches_independent_scipy_and_full_raw_information():
    rng = np.random.default_rng(176)
    frame = pd.DataFrame({"x": rng.normal(5, 2, 40), "z": rng.uniform(-2, 3, 40)})
    frame["y"] = rng.binomial(1, special.expit(-2 + 0.3 * frame.x - 0.7 * frame.z))
    r = oe.firth_logit(frame, "y", ["x", "z"])
    x = np.column_stack([np.ones(len(frame)), frame[["x", "z"]]])
    oracle = optimize.minimize(
        raw_objective,
        np.zeros(3),
        args=(x, frame.y.to_numpy()),
        method="BFGS",
        options={"gtol": 1e-8},
    )
    np.testing.assert_allclose(r["coefficients"].estimate, oracle.x, atol=2e-6)
    prob = special.expit(x @ r["coefficients"].estimate.to_numpy())
    np.testing.assert_allclose(
        r["covariance"],
        np.linalg.inv(x.T @ (prob[:, None] * (1 - prob[:, None]) * x)),
        rtol=1e-9,
        atol=1e-9,
    )
    # Numeric finite differences check the penalized score independently of Torch.
    score = optimize._numdiff.approx_derivative(
        lambda v: raw_objective(v, x, frame.y.to_numpy()), r["coefficients"].estimate.to_numpy()
    ).ravel()
    assert max(abs(score)) < 2e-6


def test_firth_raw_shift_rescale_and_prediction_full_covariance():
    data = group_data(1, 9, 7, 10, shift=4.0, scale=3.0)
    fit = oe.firth_logit(data, "y", ["x"])
    new = pd.DataFrame({"x": [4.0, 5.0, 7.0]}, index=["first", "between", "last"])
    pred = oe.firth_predict(oe.finite_load(oe.finite_save(fit)), new, level=0.9)["predictions"]
    x = np.column_stack([np.ones(3), new.x])
    beta = fit["coefficients"].estimate.to_numpy()
    cov = fit["covariance"].to_numpy(float)
    link = x @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", x, cov, x))
    probability = special.expit(link)
    np.testing.assert_allclose(pred.link, link, atol=1e-10)
    np.testing.assert_allclose(pred.link_std_error, se, rtol=1e-10)
    np.testing.assert_allclose(pred.probability, probability, rtol=1e-10)
    np.testing.assert_allclose(
        pred.probability_std_error, se * probability * (1 - probability), rtol=1e-10
    )
    np.testing.assert_allclose(
        pred.ci_low, special.expit(link - stats.norm.ppf(0.95) * se), rtol=1e-10
    )
    assert pred.label.tolist() == new.index.tolist() and pred.position.tolist() == [0, 1, 2]


@pytest.mark.parametrize("term", ["_cons", "x"])
def test_profile_raw_intercept_and_slope_refits_match_independent_profile_roots(term):
    data = group_data(1, 9, 7, 10, shift=2.0, scale=1.5)
    fit = oe.firth_logit(data, "y", ["x"])
    beta = fit["coefficients"].estimate.to_numpy()
    x = np.column_stack([np.ones(len(data)), data.x])
    y = data.y.to_numpy()
    j = 0 if term == "_cons" else 1
    baseline = raw_objective(beta, x, y)
    cutoff = stats.chi2.ppf(0.9, 1)

    def profile(value):
        def f(nuisance):
            b = beta.copy()
            b[j] = value
            b[1 - j] = nuisance
            return raw_objective(b, x, y)

        solved = optimize.minimize_scalar(
            f, method="bounded", bounds=(-30, 30), options={"xatol": 1e-11}
        )
        return 2 * (solved.fun - baseline) - cutoff

    se = fit["coefficients"].std_error.iloc[j]
    expected = []
    for side in [-1, 1]:
        end = beta[j] + side * se
        while profile(end) < 0:
            end = beta[j] + 2 * (end - beta[j])
        expected.append(optimize.brentq(profile, min(end, beta[j]), max(end, beta[j]), xtol=1e-10))
    r = oe.firth_profile(fit, [term], level=0.9, max_work=300_000_000)
    np.testing.assert_allclose(r["intervals"][["ci_low", "ci_high"]].iloc[0], expected, atol=1e-7)
    for record in r.attrs["state"]["profile_evaluations"]:
        saved = record["convergence"]
        assert saved["free_score_max"] <= 1e-8 and saved["constraint_residual"] < 1e-9
    assert r.attrs["state"]["fit"] == fit.attrs["state"]


def test_joint_raw_constraint_lr_and_full_contrast_covariance():
    frame = group_data(1, 9, 7, 10, 2, 1.5)
    fit = oe.firth_logit(frame, "y", ["x"])
    rmat = np.array([[1.0, 0.4], [0.0, 1.0]])
    target = np.array([0.2, 0.3])
    r = oe.firth_test(fit, rmat.tolist(), target.tolist())
    constrained = np.linalg.solve(rmat, target)
    x = np.column_stack([np.ones(len(frame)), frame.x])
    y = frame.y.to_numpy()
    lr = 2 * (
        raw_objective(constrained, x, y)
        - raw_objective(fit["coefficients"].estimate.to_numpy(), x, y)
    )
    assert r["test"].penalized_lr.iloc[0] == pytest.approx(lr, abs=1e-8)
    assert r["test"].df.iloc[0] == 2 and r["test"].p_value.iloc[0] == pytest.approx(
        stats.chi2.sf(lr, 2), rel=1e-8
    )
    np.testing.assert_allclose(
        r["contrast_covariance"], rmat @ fit["covariance"].to_numpy() @ rmat.T, rtol=1e-10
    )
    np.testing.assert_allclose(r["constrained_coefficients"].estimate, constrained, atol=1e-10)


def test_single_restriction_refits_same_full_model_not_reduced_penalty():
    data = group_data(0, 10, 9, 10)
    fit = oe.firth_logit(data, "y", ["x"])
    r = oe.firth_test(fit, [[0, 1]])
    x = np.column_stack([np.ones(len(data)), data.x])
    y = data.y.to_numpy()
    beta = fit["coefficients"].estimate.to_numpy()
    null = optimize.minimize_scalar(
        lambda intercept: raw_objective(np.array([intercept, 0]), x, y),
        bounds=(-20, 20),
        method="bounded",
    )
    assert r["test"].penalized_lr.iloc[0] == pytest.approx(
        2 * (null.fun - raw_objective(beta, x, y)), abs=1e-8
    )
    assert r.attrs["state"]["constrained_convergence"]["free_dimensions"] == 1


@pytest.mark.parametrize(
    "cells", [[[7, 3], [2, 8]], [[0, 6], [5, 3]], [[6, 0], [2, 5]], [[2, 3], [4, 1]]]
)
def test_single_stratum_cmle_equal_tail_ci_matches_scipy_conditional_odds(cells):
    oracle = stats.contingency.odds_ratio(cells, kind="conditional")
    ci = oracle.confidence_interval(0.95)
    r = oe.exact_logistic([cells])
    row = r["inference"].iloc[0]
    if math.isinf(oracle.statistic):
        assert pd.isna(row.odds_cmle) and row.estimate_boundary == "positive_infinity"
    else:
        assert row.odds_cmle == pytest.approx(oracle.statistic, rel=1e-9, abs=1e-12)
    assert row.ci_low == pytest.approx(ci.low, rel=1e-9, abs=1e-12)
    if math.isinf(ci.high):
        assert pd.isna(row.ci_high) and row.ci_high_boundary == "positive_infinity"
    else:
        assert row.ci_high == pytest.approx(ci.high, rel=1e-9)
    a, b, c, d = np.asarray(cells).ravel()
    null = stats.hypergeom(a + b + c + d, a + c, a + b)
    assert row.p_value == pytest.approx(min(1.0, 2 * min(null.cdf(a), null.sf(a - 1))), abs=1e-12)
    assert r["support"].null_probability.sum() == pytest.approx(1.0)


def test_stratified_complete_distribution_matches_independent_polynomial_convolution_and_tails():
    tables = [[[3, 4], [1, 5]], [[5, 2], [3, 6]], [[0, 0], [1, 2]]]
    polynomial = np.array([1.0])
    offset = 0
    observed = 0
    for cells in tables:
        a, b, cc, d = np.asarray(cells).ravel()
        r1, r0, t = a + b, cc + d, a + cc
        low, high = max(0, t - r0), min(r1, t)
        observed += a
        offset += low
        polynomial = np.convolve(
            polynomial,
            np.array(
                [math.comb(r1, i) * math.comb(r0, t - i) for i in range(low, high + 1)], float
            ),
        )
    support = np.arange(offset, offset + len(polynomial))
    prob = polynomial * 2.0**support
    prob /= prob.sum()
    r = oe.exact_logistic(tables, null_odds=2, level=0.9)
    saved = r.attrs["state"]
    row = r["inference"].iloc[0]
    np.testing.assert_allclose(r["support"].null_probability, prob, atol=2e-15)
    assert r["support"].statistic.tolist() == support.tolist() and len(saved["strata"]) == 3
    assert row.p_value == pytest.approx(
        min(1.0, 2 * min(prob[support <= observed].sum(), prob[support >= observed].sum()))
    )
    for odds, mask in [(row.ci_low, support >= observed), (row.ci_high, support <= observed)]:
        p = polynomial * odds**support
        p /= p.sum()
        assert p[mask].sum() == pytest.approx(0.05, abs=1e-10)
    assert np.dot(r["support"].fit_probability, support) == pytest.approx(observed, abs=1e-9)


@pytest.mark.parametrize("k,exposure", [(0, 36), (27, 1), (84, 36), (1, 0.5), (10000, 20)])
def test_garwood_full_interval_and_inclusive_poisson_tails_against_scipy(k, exposure):
    r = oe.exact_poisson_rate(k, exposure, null_rate=0.8, level=0.95)["inference"].iloc[0]
    expected = [
        0 if k == 0 else stats.chi2.ppf(0.025, 2 * k) / 2 / exposure,
        stats.chi2.isf(0.025, 2 * (k + 1)) / 2 / exposure,
    ]
    np.testing.assert_allclose([r.ci_low, r.ci_high], expected, rtol=1e-10, atol=1e-13)
    assert r.rate == k / exposure and r.plugin_variance == k / exposure**2
    assert r.lower_tail == pytest.approx(stats.poisson.cdf(k, 0.8 * exposure), abs=1e-12)
    assert r.upper_tail == pytest.approx(stats.poisson.sf(k - 1, 0.8 * exposure), abs=1e-12)
    assert r.p_value == pytest.approx(
        min(
            1,
            2 * min(stats.poisson.cdf(k, 0.8 * exposure), stats.poisson.sf(k - 1, 0.8 * exposure)),
        ),
        abs=1e-12,
    )


def test_stata_published_poisson_ci_rounding_and_zero_endpoint():
    # Stata rci.pdf pages 7,9,8 respectively; no licensed Stata process was run.
    for k, t, lo, hi in [
        (84, 36, 1.861158, 2.888825),
        (27, 1, 17.79317, 39.28358),
        (0, 36, 0, 0.1024689),
    ]:
        row = oe.exact_poisson_rate(k, t)["inference"].iloc[0]
        assert row.ci_low == pytest.approx(lo, abs=5e-6) and row.ci_high == pytest.approx(
            hi, abs=5e-6
        )


def contaminated():
    x = np.arange(12, dtype=float)
    y = 1 + 2 * x + 0.05 * (-1.0) ** x + 0.009 * np.sin(1.7 * x)
    y[9:] += 30
    return pd.DataFrame({"x": x, "y": y}, index=[f"unit-{i}" for i in range(12)])


@pytest.mark.parametrize("h", [7, 8, 12])
def test_lts_every_candidate_global_computed_minimum_matches_independent_qr(h):
    frame = contaminated()
    x = np.column_stack([np.ones(len(frame)), frame.x])
    y = frame.y.to_numpy()
    candidates = []
    for idx in itertools.combinations(range(len(frame)), h):
        sub = x[list(idx)]
        q, r = np.linalg.qr(sub)
        beta = np.linalg.solve(r, q.T @ y[list(idx)])
        sse = np.sum((y[list(idx)] - sub @ beta) ** 2)
        candidates.append((sse, beta, idx))
    best = min(candidates, key=lambda v: v[0])
    result = oe.lts(frame, "y", ["x"], h=h)
    np.testing.assert_allclose(result["coefficients"].estimate, best[1], atol=1e-10)
    assert result["summary"].trimmed_sse.iloc[0] == pytest.approx(best[0], abs=1e-10)
    np.testing.assert_allclose(
        result["candidates"].subset_sse, [v[0] for v in candidates], atol=1e-10
    )
    assert result.attrs["state"]["subset_count"] == math.comb(12, h) == len(result["candidates"])
    assert result["residuals"].trimmed_inlier.sum() == h
    assert set(result["coefficients"].columns) == {"term", "estimate"}
    assert "covariance" not in result and "std_error" not in result["coefficients"]
    if h < 12:
        assert abs(best[1][1] - 2) < 0.02


def test_lts_default_trim_shift_scale_missing_identity_and_saved_predict():
    frame = contaminated()
    frame.x = 10 + 3 * frame.x
    frame.y = 100 + 4 * frame.y
    frame.loc["unit-1", "y"] = np.nan
    fit = oe.lts(frame, "y", ["x"], missing="drop")
    assert fit.attrs["state"]["h"] == (11 + 2 + 1) // 2
    assert fit.attrs["state"]["dropped_positions"] == [1] and fit.attrs["state"][
        "dropped_labels"
    ] == ["unit-1"]
    new = pd.DataFrame({"x": [10.0, 22.0, 43.0]}, index=["new-z", "new-a", "new-m"])
    pred = oe.lts_predict(oe.finite_load(oe.finite_save(fit)), new)
    np.testing.assert_allclose(
        pred["predictions"].predicted,
        fit["coefficients"].estimate.iloc[0] + new.x * fit["coefficients"].estimate.iloc[1],
        atol=1e-10,
    )
    assert (
        pred["predictions"].label.tolist() == new.index.tolist()
        and pred.attrs["state"]["fit"] == fit.attrs["state"]
    )
    assert pred.attrs["inference"] == "descriptive saved LTS mean; no SE/CI"


def all_results():
    data = group_data()
    fit = oe.firth_logit(data, "y", ["x"])
    trimmed = oe.lts(contaminated(), "y", ["x"], h=8)
    return [
        fit,
        oe.firth_predict(fit, data),
        oe.firth_profile(fit, ["x"], max_work=300_000_000),
        oe.firth_test(fit, [[0, 1]]),
        oe.exact_logistic([[[3, 2], [1, 4]]]),
        oe.exact_poisson_rate(0, 36),
        trimmed,
        oe.lts_predict(trimmed, contaminated()),
    ]


def test_all_eight_full_ordered_artifacts_roundtrip_and_latex(tmp_path):
    for result in all_results():
        path = tmp_path / (result.attrs["procedure"] + ".json")
        saved = oe.finite_save(result, path)
        loaded = oe.finite_load(path)
        assert list(loaded) == list(result) and loaded.attrs == result.attrs
        assert oe.finite_save(loaded) == saved and "\\begin{tabular}" in loaded.to_latex()
        for name in loaded:
            pd.testing.assert_frame_equal(loaded[name], result[name], check_dtype=False)


@pytest.mark.parametrize("mutation", ["table", "state", "order", "outer_checksum"])
def test_state_and_table_tampering_refused(mutation):
    fit = oe.firth_logit(group_data(), "y", ["x"])
    saved = oe.finite_save(fit)
    if mutation == "table":
        fit["covariance"].iloc[0, 0] += 1
    elif mutation == "state":
        fit.attrs["state"]["parameters"][0] += 1
    elif mutation == "order":
        fit["coefficients"] = fit.pop("coefficients")
    else:
        saved["sha256"] = "0" * 64
    with pytest.raises(AnalysisError):
        oe.finite_load(saved) if mutation == "outer_checksum" else oe.finite_save(fit)


@pytest.mark.parametrize("api", ["firth_logit", "lts"])
def test_numeric_design_input_failures_missing_options_weights_device(api):
    function = getattr(oe, api)
    for frame, kwargs in [
        (group_data().assign(x=True), {}),
        (group_data().assign(y=0.5), {} if api == "firth_logit" else {"h": 1}),
        (group_data().assign(x=np.inf), {}),
        (group_data(), {"device": "mps"}),
        (group_data(), {"weights": [1] * 20}),
        (group_data(), {"x": ["x", "x"]}),
        (group_data(), {"missing": "silent"}),
    ]:
        with pytest.raises(AnalysisError):
            function(frame, "y", **({"x": ["x"]} | kwargs))


def test_firth_nonconvergence_profile_work_bracket_and_rank_fail_closed():
    with pytest.raises(AnalysisError, match="convergence"):
        oe.firth_logit(group_data(), "y", ["x"], max_iter=1)
    fit = oe.firth_logit(group_data(), "y", ["x"])
    with pytest.raises(AnalysisError) as e:
        oe.firth_profile(fit, ["x"], max_work=1)
    assert e.value.code == "work_limit"
    with pytest.raises(AnalysisError) as e:
        oe.firth_profile(fit, ["x"], bracket_steps=1, max_work=300_000_000)
    assert e.value.code == "unbracketed_interval"
    for restrictions in [[[1, 0], [1, 0]], [[1, 2, 3]], [], [[np.nan, 0]]]:
        with pytest.raises(AnalysisError):
            oe.firth_test(fit, restrictions)


def test_lts_any_singular_candidate_subset_cap_and_workspace_guard_before_qr(monkeypatch):
    data = pd.DataFrame({"x": [0, 0, 0, 0, 1, 2], "y": [1, 2, 3, 4, 5, 6]})
    with pytest.raises(AnalysisError) as e:
        oe.lts(data, "y", ["x"], h=4)
    assert e.value.code == "unidentified_design"
    with pytest.raises(AnalysisError) as e:
        oe.lts(contaminated(), "y", ["x"], max_subsets=1)
    assert e.value.code == "resource_limit"

    def forbidden(*a, **k):
        raise AssertionError("QR allocated before workspace guard")

    monkeypatch.setattr(torch.linalg, "qr", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as e:
        oe.lts(contaminated(), "y", ["x"], h=7)
    assert e.value.code == "workspace_limit"


@pytest.mark.parametrize(
    "tables",
    [[], [[[True, 1], [2, 3]]], [[[1.2, 1], [2, 3]]], [[[0, 0], [1, 3]]], [[[3000, 3000], [1, 3]]]],
)
def test_exact_invalid_counts_margins_domain_refuse(tables):
    with pytest.raises(AnalysisError):
        oe.exact_logistic(tables)


@pytest.mark.parametrize(
    "k,t,q",
    [(True, 1, 1), (1.5, 1, 1), (-1, 1, 1), (10001, 1, 1), (1, 0, 1), (1, 1, 0), (1, 10001, 1)],
)
def test_poisson_invalid_counts_exposure_null_domain(k, t, q):
    with pytest.raises(AnalysisError):
        oe.exact_poisson_rate(k, t, null_rate=q)


def test_float32_and_meta_default_do_not_redirect_resident_geometry():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            fit = oe.firth_logit(group_data(), "y", ["x"])
            exact = oe.exact_logistic([[[3, 2], [1, 4]]])
            trimmed = oe.lts(contaminated(), "y", ["x"], h=8)
        assert (
            fit.attrs["precision"]
            == exact.attrs["precision"]
            == trimmed.attrs["precision"]
            == "float64"
        )
        assert fit.attrs["device"] == exact.attrs["device"] == trimmed.attrs["device"] == "cpu"
    finally:
        torch.set_default_dtype(previous)


def test_torch_free_manifest_and_native_procedures_without_scipy_statsmodels():
    code = """import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('scipy','statsmodels'): raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import openecon as oe
assert 'torch' not in sys.modules
assert set(oe.capabilities()['finite_regression']['procedures'])==set(%r)
assert 'torch' not in sys.modules
import torch; torch.set_num_threads(2)
fit=oe.firth_logit({'y':[0]*6+[1]*6,'x':[0]*6+[1]*6},'y',['x'])
oe.firth_test(fit,[[0,1]]); oe.firth_predict(fit,{'x':[0,1]}); oe.firth_profile(fit,['x'],max_work=300000000)
oe.exact_logistic([[[3,2],[1,4]]]); oe.exact_poisson_rate(84,36)
l=oe.lts({'x':list(range(8)),'y':[1,3,5,7,9,11,30,40]},'y',['x'],h=5); oe.lts_predict(l,{'x':[0,2]})
assert 'scipy' not in sys.modules and 'statsmodels' not in sys.modules
""" % (
        [
            "firth_logit",
            "firth_predict",
            "firth_profile",
            "firth_test",
            "exact_logistic",
            "exact_poisson_rate",
            "lts",
            "lts_predict",
        ],
    )
    run = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert run.returncode == 0, run.stderr


def test_firth_shuffled_joint_missing_selection_and_saved_state_do_not_refit(monkeypatch):
    frame = group_data(1, 9, 7, 10)
    frame.loc["row-2", "x"] = np.nan
    with pytest.raises(AnalysisError):
        oe.firth_logit(frame, "y", ["x"])
    shuffled = frame.sample(frac=1, random_state=385)
    fit = oe.firth_logit(shuffled, "y", ["x"], missing="drop")
    state = fit.attrs["state"]
    assert shuffled.iloc[state["sample_positions"]].index.tolist() == state["sample_labels"]
    assert state["dropped_labels"] == ["row-2"]
    complete = oe.firth_logit(frame.dropna(), "y", ["x"])
    np.testing.assert_allclose(
        fit["coefficients"].estimate, complete["coefficients"].estimate, atol=1e-9
    )
    from openecon.econometrics.finite import firth

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved prediction or load attempted a refit")

    monkeypatch.setattr(firth, "solve", forbidden)
    saved = oe.finite_load(oe.finite_save(fit))
    predicted = oe.firth_predict(
        saved, pd.DataFrame({"x": [0.0, np.nan, 1.0]}, index=["a", "b", "c"]), missing="drop"
    )
    assert predicted["predictions"].label.tolist() == ["a", "c"]
    assert predicted.attrs["state"]["prediction_sample"]["dropped_labels"] == ["b"]


def test_no_intercept_firth_scales_without_centring_and_lts_ols_domain():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [0, 0, 1, 0, 1, 1]})
    fit = oe.firth_logit(data, "y", ["x"], intercept=False)
    x = data.x.to_numpy()[:, None]
    oracle = optimize.minimize_scalar(
        lambda b: raw_objective(np.array([b]), x, data.y.to_numpy()),
        bounds=(-4, 4),
        method="bounded",
    )
    assert fit["coefficients"].estimate.iloc[0] == pytest.approx(oracle.x, abs=1e-6)
    assert fit.attrs["state"]["center"] == [0.0] and fit.attrs["state"]["terms"] == ["x"]
    trimmed = oe.lts(data, "y", ["x"], intercept=False, h=6)
    assert trimmed["coefficients"].estimate.iloc[0] == pytest.approx(
        np.dot(data.x, data.y) / np.dot(data.x, data.x)
    )


def test_lts_equal_objective_ties_are_descriptive_and_reproducible():
    data = pd.DataFrame({"x": list(range(8)), "y": [2.0] * 8})
    first = oe.lts(data, "y", ["x"], h=5)
    second = oe.lts(data, "y", ["x"], h=5)
    assert oe.finite_save(first) == oe.finite_save(second)
    assert first.attrs["state"]["best_subset"] == list(range(5))
    assert first.attrs["state"]["trimmed_sample_indices"] == list(range(5))
    assert first["candidates"].subset_sse.tolist() == [0.0] * math.comb(8, 5)
    np.testing.assert_allclose(first["coefficients"].estimate, [2.0, 0.0], atol=1e-12)


def test_exact_support_budget_refuses_before_convolution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Conditional distribution allocated before work budget")

    monkeypatch.setattr(torch, "logsumexp", forbidden)
    with pytest.raises(AnalysisError) as exc:
        oe.exact_logistic([[[20, 30], [25, 15]]], max_work=1)
    assert exc.value.code == "work_limit"


def test_firth_expected_information_covariance_differs_from_penalized_hessian():
    data = group_data(0, 10, 9, 10)
    fit = oe.firth_logit(data, "y", ["x"])
    state = fit.attrs["state"]
    from openecon.econometrics.finite.firth import objective

    x = torch.tensor(state["design_scaled"], dtype=torch.float64)
    y = torch.tensor(state["response"], dtype=torch.float64)
    theta = torch.tensor(state["parameters_scaled"], dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(lambda b: objective(b, x, y), theta)
    assert not np.allclose(np.linalg.inv(-hessian.numpy()), state["covariance_scaled"])
    assert "expected Fisher" in fit.attrs["inference"]
