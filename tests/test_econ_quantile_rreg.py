"""Independent oracles for Stata-style robust regression (rreg).

The whole procedure (Cook's distance screening, Huber then biweight IRLS with
the MAD scale, pseudovalue standard errors) is rewritten in NumPy in the form
that reproduces the output of Stata's manual example (see
tests/test_econ_quantile_oracle.py for the auto.dta check); the weight and psi
functions are checked against statsmodels' robust norms, Cook's distance
against statsmodels' influence measures and the pseudovalue covariance against
the M-estimator sandwich.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.robust import norms
from statsmodels.stats.outliers_influence import OLSInfluence

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.quantile import rreg as kernel
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec

X_COLS = ["x1", "x2"]


def lstsq(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def mad_scale(e):
    return np.median(np.abs(e - np.median(e))) / 0.6745


def rreg_oracle(x, y, tune=7.0, tol=0.01, huber_tol=None):
    huber_tol = max(0.05, tol) if huber_tol is None else huber_tol
    n, k = x.shape
    e = y - x @ lstsq(x, y)
    h = np.einsum("ij,jk,ik->i", x, np.linalg.inv(x.T @ x), x)
    cooks = e ** 2 * h / (k * (e @ e / (n - k)) * (1 - h) ** 2)
    keep = ~(cooks > 1)
    x, y = x[keep], y[keep]
    n = len(y)
    c = 4.685 * tune / 7
    e = y - x @ lstsq(x, y)
    w, log = np.ones(n), []
    for stage, threshold in (("huber", huber_tol), ("biweight", tol)):
        while True:
            mad = np.median(np.abs(e - np.median(e)))
            s = mad / 0.6745
            if stage == "huber":        # weights fall beyond 2 MAD
                new = np.minimum(1.0, 2 * mad / np.maximum(np.abs(e), 1e-300))
            else:
                u = e / s
                new = np.where(np.abs(u) < c, (1 - (u / c) ** 2) ** 2, 0.0)
            change = np.abs(new - w).max()
            w = new
            root = np.sqrt(w)
            beta = lstsq(x * root[:, None], y * root)
            e = y - x @ beta
            log.append((stage, change))
            if change < threshold:
                break
    # Pseudovalues: final weights w, the scale s that produced them, final residuals e.
    u = e / s
    slope = np.where(np.abs(u) < c, (1 - (u / c) ** 2) * (1 - 5 * (u / c) ** 2), 0.0)
    m = slope.mean()
    lam = 1 + k / (n - k) * (1 - m) / m
    pseudo = x @ beta + lam / m * w * e
    final = lstsq(x, pseudo)            # equals beta because X'We = 0
    rss = ((pseudo - x @ final) ** 2).sum()
    return {"beta": beta, "pseudo_beta": final, "cov": rss / (n - k) * np.linalg.inv(x.T @ x),
            "keep": keep, "log": log, "scale": s, "m": m, "lambda": lam, "x": x, "y": y,
            "weights": w, "resid": e, "rmse": np.sqrt(rss / (n - k))}


def make_data(seed=17, n=160, outliers=12, leverage=True):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.exponential(size=n)})
    frame["y"] = 1 + 0.5 * frame.x1 - 0.3 * frame.x2 + rng.normal(size=n)
    shocks = rng.choice([-1, 1], size=outliers) * rng.uniform(8, 20, outliers)
    frame.loc[np.arange(outliers), "y"] += shocks
    if leverage:
        frame.loc[n - 1, ["x1", "y"]] = [14.0, -45.0]      # a gross outlier with high leverage
    frame["sector"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def design(frame, cols=X_COLS):
    return np.column_stack([np.ones(len(frame)), frame[cols].to_numpy()])


def params(result):
    return np.array([c.estimate for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def test_weight_and_psi_functions_match_statsmodels_norms():
    u = torch.linspace(-12, 12, 481, dtype=torch.float64)
    # Stata's Huber weights fall beyond 2 MAD, i.e. beyond 2 * 0.6745 scaled residuals.
    assert_allclose(kernel.huber_weights(u).numpy(), norms.HuberT(1.349).weights(u.numpy()),
                    rtol=1e-14)
    for tune in (4.0, 7.0, 9.5):
        c = 4.685 * tune / 7
        assert_allclose(kernel.biweight_weights(u, c).numpy(),
                        norms.TukeyBiweight(c).weights(u.numpy()), rtol=1e-13, atol=1e-16)
    samples = (np.arange(7.0), np.array([3.0, 1.0, 2.0, 10.0]),
               np.random.default_rng(0).normal(size=51))
    for values in samples:
        tensor = torch.tensor(values)
        assert_allclose(float(kernel.median(tensor)), np.median(values), rtol=1e-15)
        assert_allclose(kernel.mad_scale(tensor), mad_scale(values), rtol=1e-14)
        # Stata divides by the rounded constant 0.6745 (statsmodels' default is Phi^-1(3/4)).
        assert_allclose(kernel.mad_scale(tensor), sm.robust.scale.mad(values, c=0.6745),
                        rtol=1e-12)


def test_cooks_distance_matches_statsmodels(data):
    x, y = design(data), data.y.to_numpy()
    reference = OLSInfluence(sm.OLS(y, x).fit()).cooks_distance[0]
    ours = kernel.cooks_distance(torch.tensor(x), torch.tensor(y)).numpy()
    assert_allclose(ours, reference, rtol=1e-9, atol=1e-14)
    assert (ours > 1).sum() == 1 and ours.argmax() == len(y) - 1


@pytest.mark.parametrize("tune, tolerance", [(7, 0.01), (5, 0.01), (9, 0.001), (7, 0.2)])
def test_rreg_matches_numpy_oracle(data, tune, tolerance):
    oracle = rreg_oracle(design(data), data.y.to_numpy(), tune=tune, tol=tolerance)
    result = oe.rreg(data=data, y="y", x=X_COLS, tune=tune, tolerance=tolerance)
    assert [c.term for c in result.coefficients] == ["Intercept", *X_COLS]
    assert_allclose(params(result), oracle["beta"], rtol=1e-9, atol=1e-11)
    assert_allclose(covariance(result), oracle["cov"], rtol=1e-8)
    n, k = int(oracle["keep"].sum()), 3
    assert result.nobs == n == len(data) - 1
    assert result.metrics["n_dropped_cooks"] == 1 and result.dropped_rows == 1
    assert result.sample_positions == np.flatnonzero(oracle["keep"]).tolist()
    assert any("Cook's distance" in warning for warning in result.warnings)
    assert_allclose(result.metrics["scale"], oracle["scale"], rtol=1e-9)
    assert_allclose(result.metrics["rmse"], oracle["rmse"], rtol=1e-9)
    stages = [stage for stage, _ in oracle["log"]]
    assert result.metrics["huber_iterations"] == stages.count("huber")
    assert result.metrics["biweight_iterations"] == stages.count("biweight")
    assert result.metrics["iterations"] == len(stages)
    log = result.extra["iteration_log"]
    assert [entry["stage"] for entry in log] == stages
    assert_allclose([entry["max_weight_change"] for entry in log],
                    [change for _, change in oracle["log"]], rtol=1e-7, atol=1e-10)
    assert log[-1]["max_weight_change"] < tolerance
    weights = result.extra["weights"]
    assert_allclose([weights["min"], weights["mean"], weights["max"]],
                    [oracle["weights"].min(), oracle["weights"].mean(), oracle["weights"].max()],
                    rtol=1e-8, atol=1e-12)
    assert weights["n_zero"] == int((oracle["weights"] == 0).sum())
    # The pseudovalue regression returns the IRLS coefficients themselves (X'We = 0).
    assert_allclose(oracle["pseudo_beta"], oracle["beta"], rtol=1e-8, atol=1e-10)
    assert_allclose(result.extra["pseudovalues"]["mean_psi_prime"], oracle["m"], rtol=1e-9)
    assert_allclose(result.extra["pseudovalues"]["lambda"], oracle["lambda"], rtol=1e-10)
    assert_allclose(result.extra["biweight_c"], 4.685 * tune / 7, rtol=1e-14)
    # t inference with N - K degrees of freedom and the F test of the pseudovalue regression.
    assert result.inference["use_t"] and result.inference["df_inference"] == n - k
    coefficient = result.coefficients[1]
    assert_allclose(coefficient.p_value, 2 * stats.t.sf(abs(coefficient.statistic), n - k),
                    rtol=1e-8)
    slopes = oracle["beta"][1:]
    f_stat = slopes @ np.linalg.solve(oracle["cov"][1:, 1:], slopes) / 2
    test = result.tests["model"]
    assert_allclose(test["statistic"], f_stat, rtol=1e-8)
    assert test["df"] == 2 and test["df2"] == n - k and test["distribution"] == "F"
    assert_allclose(test["p_value"], stats.f.sf(f_stat, 2, n - k), rtol=1e-7)
    assert result.metrics["df_model"] == 2 and result.metrics["df_resid"] == n - k


def test_rreg_resists_outliers(data):
    truth = np.array([1.0, 0.5, -0.3])
    robust = oe.rreg(data=data, y="y", x=X_COLS)
    ols = lstsq(design(data), data.y.to_numpy())
    assert np.abs(params(robust) - truth).max() < 0.25
    assert np.abs(ols - truth).max() > 1.0
    # The contaminated observations end with zero weight.
    assert robust.extra["weights"]["n_zero"] >= 10
    assert 0.9 < robust.metrics["scale"] < 1.2


def test_converged_fit_solves_the_m_estimating_equations():
    frame = make_data(seed=3, n=200, outliers=10, leverage=False)
    x, y = torch.tensor(design(frame)), torch.tensor(frame.y.to_numpy())
    fit = kernel.robust_regression(x, y, tolerance=1e-11, huber_tolerance=1e-11)
    resid = y - x @ fit.beta
    scale = kernel.mad_scale(resid)
    assert_allclose(fit.scale, scale, rtol=1e-8)
    weights = kernel.biweight_weights(resid / scale, 4.685)
    assert_allclose(fit.weights.numpy(), weights.numpy(), atol=1e-9)
    # Fixed point of IRLS: X' (w e) = 0, the biweight M-estimating equations.
    assert float((x.T @ (weights * resid)).abs().max()) < 1e-7
    # The pseudovalue covariance is the M-estimator sandwich with the degrees-of-freedom
    # correction lambda: lambda^2 s^2 [sum psi^2 / (n - k)] / m^2 (X'X)^-1.
    n, k = x.shape
    u = (resid / scale).numpy()
    psi = norms.TukeyBiweight(4.685).psi(u)
    m = norms.TukeyBiweight(4.685).psi_deriv(u).mean()
    lam = 1 + k / (n - k) * (1 - m) / m
    factor = (lam * scale / m) ** 2 * (psi ** 2).sum() / (n - k)
    expected = factor * np.linalg.inv((x.T @ x).numpy())
    assert_allclose(fit.covariance.numpy(), expected, rtol=1e-6)
    assert_allclose(fit.mean_psi_prime, m, rtol=1e-8)
    # statsmodels' RLM solves the same biweight equations with a MAD scale centred at zero;
    # with a symmetric bulk of residuals the estimates are close.
    reference = sm.RLM(frame.y.to_numpy(), design(frame), M=norms.TukeyBiweight(4.685)).fit()
    assert_allclose(fit.beta.numpy(), reference.params, atol=0.02)
    assert_allclose(np.sqrt(np.diag(fit.covariance.numpy())), reference.bse, rtol=0.1)


def test_rreg_without_screened_observations_and_design_options(data):
    clean = make_data(seed=5, n=120, outliers=5, leverage=False)
    result = oe.rreg(data=clean, y="y", x=X_COLS)
    oracle = rreg_oracle(design(clean), clean.y.to_numpy())
    assert oracle["keep"].all() and result.metrics["n_dropped_cooks"] == 0
    assert result.dropped_rows == 0 and result.nobs == 120
    assert_allclose(params(result), oracle["beta"], rtol=1e-9)
    no_constant = oe.rreg(data=clean, y="y", x=X_COLS, intercept=False)
    oracle = rreg_oracle(clean[X_COLS].to_numpy(), clean.y.to_numpy())
    assert [c.term for c in no_constant.coefficients] == X_COLS
    assert_allclose(params(no_constant), oracle["beta"], rtol=1e-9)
    assert_allclose(covariance(no_constant), oracle["cov"], rtol=1e-8)
    assert no_constant.tests["model"]["df"] == 2
    categorical = oe.rreg(data=clean, y="y", x=["x1", "sector"], categorical=["sector"])
    terms = [c.term for c in categorical.coefficients]
    assert terms == ["Intercept", "x1", "sector[b]", "sector[c]"]
    dummies = np.column_stack([np.ones(120), clean.x1, clean.sector == "b", clean.sector == "c"])
    oracle = rreg_oracle(dummies.astype(float), clean.y.to_numpy())
    assert_allclose(params(categorical), oracle["beta"], rtol=1e-9)


def test_rreg_collinearity_and_missing_data(data):
    frame = data.assign(twice=2 * data.x1)
    result = oe.rreg(data=frame, y="y", x=["x1", "twice", "x2"])
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert result.provenance["omitted_terms"] == ["twice"]
    assert_allclose(params(result), params(oe.rreg(data=data, y="y", x=X_COLS)), rtol=1e-10)
    frame = data.copy()
    frame.loc[[20, 31], "x2"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.rreg(data=frame, y="y", x=X_COLS)
    assert caught.value.code == "missing_values"
    dropped = oe.rreg(data=frame, y="y", x=X_COLS, missing="drop")
    complete = frame.dropna().reset_index(drop=True)
    oracle = rreg_oracle(design(complete), complete.y.to_numpy())
    assert_allclose(params(dropped), oracle["beta"], rtol=1e-9)
    assert dropped.nobs == int(oracle["keep"].sum()) and dropped.dropped_rows == 3
    assert len(dropped.warnings) == 2


def test_rreg_error_codes(data):
    cases = [
        ({"tune": 0}, "invalid_option"), ({"tune": -3}, "invalid_option"),
        ({"tolerance": 0}, "invalid_option"), ({"tolerance": 1.5}, "invalid_option"),
        ({"tune": "seven"}, "invalid_spec"), ({"covariance": "robust"}, "invalid_spec"),
        ({"covariance": "HC1"}, "invalid_spec"), ({"x": "x1"}, "invalid_spec"),
        ({"x": []}, "invalid_spec"), ({"x": ["x1", "absent"]}, "missing_columns"),
    ]
    for options, code in cases:
        arguments = {"data": data, "y": "y", "x": X_COLS, **options}
        with pytest.raises(AnalysisError) as caught:
            oe.rreg(**arguments)
        assert caught.value.code == code, options
    with pytest.raises(Exception) as caught:
        ModelSpec(estimator="rreg", outcome="y", predictors=X_COLS, weights="x2",
                  weight_type="aweight")
    assert "does not support aweights" in str(caught.value)
    with pytest.raises(Exception) as caught:
        ModelSpec(estimator="rreg", outcome="y", predictors=X_COLS, cluster="sector")
    assert "covariance" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.rreg(data=data.iloc[:3], y="y", x=X_COLS)
    assert caught.value.code == "insufficient_observations"
    exact = data.assign(y=1 + 2 * data.x1)
    with pytest.raises(AnalysisError) as caught:
        oe.rreg(data=exact, y="y", x=X_COLS)
    assert caught.value.code in {"perfect_fit", "zero_scale"}
    # More than half of the residuals exactly zero: the MAD scale collapses.
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x1": rng.normal(size=60)})
    frame["y"] = 2 * frame.x1
    frame.loc[:9, "y"] += rng.normal(size=10) * 0.01
    with pytest.raises(AnalysisError) as caught:
        oe.rreg(data=frame, y="y", x=["x1"], tolerance=1e-6)
    assert caught.value.code == "zero_scale"
    assert "median absolute deviation" in str(caught.value)
    with pytest.raises(KernelError) as caught:
        kernel.robust_regression(torch.ones((2, 2), dtype=torch.float64),
                                 torch.ones(2, dtype=torch.float64))
    assert caught.value.code == "insufficient_observations"
    x = torch.tensor(design(data))
    with pytest.raises(KernelError) as caught:
        kernel.robust_regression(x, torch.tensor(data.y.to_numpy()), tolerance=1e-14,
                                 huber_tolerance=1e-14, max_iterations=3)
    assert caught.value.code == "nonconvergence"


def test_rreg_round_trip_rendering_and_public_api(data):
    result = oe.rreg(data=data, y="y", x=X_COLS)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    json.dumps(result.model_dump())
    summary = result.summary()
    assert "Robust regression" in summary and "n_dropped_cooks" in summary and "F(2," in summary
    assert "tabular" in result.to_latex() and "x2" in result.to_latex()
    spec = ModelSpec(estimator="rreg", outcome="y", predictors=X_COLS)
    assert spec.covariance == "nonrobust"
    again = oe.fit(spec, data=data)
    assert_allclose(params(again), params(result), rtol=0, atol=0)
    listing = oe.capabilities()["estimators"]["rreg"]
    assert listing["family"] == "quantile" and listing["covariances"] == ["nonrobust"]
    assert listing["weights"] == [] and listing["stata"] == ["rreg"]
    assert result.provenance["solver"] == "huber_biweight_irls"
    assert result.provenance["stata_parity_validated"] is False
    assert len(result.predictions) == result.nobs
    assert callable(oe.rreg) and "Cook's distance" in oe.rreg.__doc__
