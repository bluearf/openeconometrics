"""Independent full mean/covariance ML, observed information and path algebra."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2

from openecon.econometrics.latent.sem import (
    latent_scores,
    latent_sem,
    latent_sem_covariance,
    sem_effects,
)

COLUMNS = ["x1", "x2", "x3", "y1", "y2", "y3"]
FACTORS = {"first": COLUMNS[:3], "second": COLUMNS[3:]}
NODES = COLUMNS + list(FACTORS)
PATHS = {"second": {"first": None}}
RESIDUALS = {("x2", "x3"): None}
INTERCEPTS = {"x1": 0.0, "y1": 0.0, "first": None, "second": None}


def named_moments(parameters):
    """Natural-unit primitive equations; no runtime decoding or Torch AD."""
    b, psi, intercept = np.zeros((8, 8)), np.zeros((8, 8)), np.zeros(8)
    b[0, 6] = b[3, 7] = 1
    for name, value in parameters.items():
        family, expression = name.split(":", 1)
        if family in ("loading", "path"):
            target, source = expression.split("<-")
            b[NODES.index(target), NODES.index(source)] = value
        elif family == "variance":
            psi[NODES.index(expression), NODES.index(expression)] = value
        elif family == "covariance":
            left, right = expression.split(",")
            i, j = NODES.index(left), NODES.index(right)
            psi[i, j] = psi[j, i] = value
        else:
            intercept[NODES.index(expression)] = value
    a = np.linalg.inv(np.eye(8) - b)
    return a @ psi @ a.T, a @ intercept, b, psi, intercept, a


TRUE = {
    "loading:x2<-first": 0.8,
    "loading:x3<-first": 0.7,
    "loading:y2<-second": 0.9,
    "loading:y3<-second": 0.6,
    "path:second<-first": 0.5,
    **{f"variance:{node}": v for node, v in zip(NODES, [0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 1.0, 0.8])},
    "covariance:x2,x3": 0.08,
    "intercept:x2": 0.3,
    "intercept:x3": -0.2,
    "intercept:y2": 0.1,
    "intercept:y3": 0.4,
    "intercept:first": 0.5,
    "intercept:second": -0.3,
}
OMEGA, MEAN, *_ = named_moments(TRUE)


def fit_summary(s, mean, n=701, **options):
    return latent_sem_covariance(
        s.tolist(),
        columns=COLUMNS,
        n=n,
        divisor="n",
        means=mean.tolist(),
        factors=FACTORS,
        paths=PATHS,
        residual_covariances=RESIDUALS,
        intercepts=INTERCEPTS,
        tolerance=1e-8,
        **options,
    )


@pytest.fixture(scope="module")
def exact():
    return fit_summary(OMEGA[:6, :6], MEAN[:6])


@pytest.fixture(scope="module")
def raw():
    y = np.random.default_rng(637).multivariate_normal(MEAN[:6], OMEGA[:6, :6], 807)
    frame = pd.DataFrame(
        y,
        columns=COLUMNS,
        index=pd.date_range("2001-01-01", periods=807, tz="Europe/Istanbul", name="date"),
    )
    result = latent_sem(
        frame,
        columns=COLUMNS,
        factors=FACTORS,
        paths=PATHS,
        residual_covariances=RESIDUALS,
        intercepts=INTERCEPTS,
        tolerance=1e-8,
    )
    return frame, result


def natural_objective(values, names, s, mean, n):
    omega, mu, b, psi, intercept, a = named_moments(dict(zip(names, values)))
    if np.linalg.eigvalsh(psi).min() <= 0:
        return 1e100, np.zeros(len(values))
    sigma, mu = omega[:6, :6], mu[:6]
    inv = np.linalg.inv(sigma)
    residual = mean - mu
    weight = inv - inv @ (s + np.outer(residual, residual)) @ inv
    gradient = []
    for name in names:
        db, dp, dc = np.zeros((8, 8)), np.zeros((8, 8)), np.zeros(8)
        family, expression = name.split(":", 1)
        if family in ("loading", "path"):
            target, source = expression.split("<-")
            db[NODES.index(target), NODES.index(source)] = 1
        elif family == "variance":
            dp[NODES.index(expression), NODES.index(expression)] = 1
        elif family == "covariance":
            left, right = expression.split(",")
            i, j = NODES.index(left), NODES.index(right)
            dp[i, j] = dp[j, i] = 1
        else:
            dc[NODES.index(expression)] = 1
        da = a @ db @ a
        ds = (da @ psi @ a.T + a @ dp @ a.T + a @ psi @ da.T)[:6, :6]
        dm = (da @ intercept + a @ dc)[:6]
        gradient.append(n / 2 * np.sum(weight * ds) - n * dm @ inv @ residual)
    value = (
        n
        / 2
        * (
            6 * math.log(2 * math.pi)
            + np.linalg.slogdet(sigma)[1]
            + np.trace(inv @ s)
            + residual @ inv @ residual
        )
    )
    return value, np.array(gradient)


def oracle(s, mean, names, n):
    initial = np.array([TRUE[name] for name in names])
    fit = minimize(
        lambda v: natural_objective(v, names, s, mean, n),
        initial,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": 1000},
    )
    assert np.max(np.abs(natural_objective(fit.x, names, s, mean, n)[1])) < 2e-5
    hess = np.zeros((len(names), len(names)))
    for j in range(len(names)):
        step = 1e-5 * max(1.0, abs(fit.x[j]))
        d = np.zeros(len(names))
        d[j] = step
        hess[:, j] = (
            natural_objective(fit.x + d, names, s, mean, n)[1]
            - natural_objective(fit.x - d, names, s, mean, n)[1]
        ) / (2 * step)
    return fit.x, -fit.fun, np.linalg.inv((hess + hess.T) / 2)


def test_independent_implied_joint_means_and_covariance(exact):
    r = exact.attrs["sem_state"]["results"]
    np.testing.assert_allclose(
        r["parameters"], [TRUE[n] for n in r["parameter_names"]], rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(r["node_covariance"], OMEGA, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(r["node_means"], MEAN, rtol=2e-6, atol=2e-7)
    expected_ll = -701 / 2 * (6 * (math.log(2 * math.pi) + 1) + np.linalg.slogdet(OMEGA[:6, :6])[1])
    assert r["log_likelihood"] == pytest.approx(expected_ll, abs=1e-9)
    assert r["diagnostics"]["df"] == 7


def test_independent_fitted_parameters_and_full_joint_oim(raw):
    frame, result = raw
    names = list(result["parameters"].index)
    s = np.cov(frame.to_numpy(), rowvar=False, bias=True)
    estimates, ll, covariance = oracle(s, frame.mean().to_numpy(), names, len(frame))
    np.testing.assert_allclose(result["parameters"]["estimate"], estimates, rtol=3e-6, atol=5e-7)
    assert result.attrs["log_likelihood"] == pytest.approx(ll, abs=2e-8)
    np.testing.assert_allclose(result["covariance"], covariance, rtol=5e-5, atol=2e-8)
    # Constrained marker intercepts with free latent means require the joint block.
    assert np.max(np.abs(covariance[:14, 14:])) > 1e-5


def test_raw_summary_n_and_n_minus_one_agree(raw):
    frame, result = raw
    other = fit_summary(
        np.cov(frame.to_numpy(), rowvar=False, bias=True), frame.mean().to_numpy(), len(frame)
    )
    n_minus = latent_sem_covariance(
        frame.cov(),
        columns=COLUMNS,
        n=len(frame),
        divisor="n-1",
        means=frame.mean(),
        factors=FACTORS,
        paths=PATHS,
        residual_covariances=RESIDUALS,
        intercepts=INTERCEPTS,
        tolerance=1e-8,
    )
    for candidate in (other, n_minus):
        np.testing.assert_allclose(
            candidate["parameters"], result["parameters"], rtol=2e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            candidate["covariance"], result["covariance"], rtol=2e-5, atol=1e-8
        )
        assert candidate.attrs["log_likelihood"] == pytest.approx(
            result.attrs["log_likelihood"], abs=1e-9
        )


def test_independent_fit_indices_and_baseline(raw):
    frame, result = raw
    fit = result.attrs["sem_state"]["results"]
    s, implied = (
        np.cov(frame.to_numpy(), rowvar=False, bias=True),
        np.array(fit["implied_covariance"]),
    )
    d = fit["diagnostics"]
    residual = frame.mean().to_numpy() - np.array(fit["node_means"])[:6]
    fml = (
        np.linalg.slogdet(implied)[1]
        - np.linalg.slogdet(s)[1]
        + np.trace(np.linalg.solve(implied, s))
        - 6
        + residual @ np.linalg.solve(implied, residual)
    )
    statistic = len(frame) * fml
    baseline = len(frame) * (np.log(s.diagonal()).sum() - np.linalg.slogdet(s)[1])
    df, base_df = 7, 15
    assert d["statistic"] == pytest.approx(statistic, abs=1e-9)
    assert d["p_value"] == pytest.approx(chi2.sf(statistic, df), rel=1e-10)
    assert d["baseline_statistic"] == pytest.approx(baseline, abs=1e-9)
    assert d["baseline_df"] == base_df
    excess = max(statistic - df, 0)
    assert d["CFI"] == pytest.approx(1 - excess / max(excess, baseline - base_df, 0), abs=1e-12)
    assert d["TLI"] == pytest.approx(
        (baseline / base_df - statistic / df) / (baseline / base_df - 1), abs=1e-12
    )
    assert d["RMSEA"] == pytest.approx(
        math.sqrt(max(statistic - df, 0) / (len(frame) * df)), abs=1e-12
    )
    scale = np.sqrt(s.diagonal())
    corr_resid = (s - implied) / np.outer(scale, scale)
    covariance_squares = corr_resid[np.tril_indices(6)] ** 2
    mean_squares = (residual / scale) ** 2
    assert d["SRMR_covariance"] == pytest.approx(np.sqrt(covariance_squares.mean()), abs=1e-12)
    assert d["standardized_mean_residual_rms"] == pytest.approx(
        np.sqrt(mean_squares.mean()), abs=1e-12
    )
    assert d["SRMR"] == pytest.approx(
        np.sqrt((covariance_squares.sum() + mean_squares.sum()) / 27), abs=1e-12
    )


@pytest.mark.parametrize("standardized", [False, True])
def test_independent_direct_indirect_total_delta(raw, standardized):
    _, result = raw
    r = result.attrs["sem_state"]["results"]
    names, values = r["parameter_names"], np.array(r["parameters"])

    def effects(v):
        omega, _, b, _, _, a = named_moments(dict(zip(names, v)))
        direct, total = b[4, 6], a[4, 6]
        factor = np.sqrt(omega[6, 6] / omega[4, 4]) if standardized else 1
        return np.array([direct, total - direct, total]) * factor

    jac = np.zeros((3, len(values)))
    for j in range(len(values)):
        delta = np.zeros(len(values))
        delta[j] = 1e-5
        jac[:, j] = (effects(values + delta) - effects(values - delta)) / (2e-5)
    actual = sem_effects(result, source="first", target="y2", standardized=standardized)
    np.testing.assert_allclose(
        actual["effects"]["estimate"], effects(values), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        actual["covariance"], jac @ np.array(r["covariance"]) @ jac.T, rtol=2e-7, atol=1e-10
    )
    assert actual["effects"].loc["direct", "std_error"] == 0
    assert pd.isna(actual["effects"].loc["direct", "p_value"])


def test_independent_regression_factor_scores_and_conditional_covariance(raw):
    frame, result = raw
    r = result.attrs["sem_state"]["results"]
    omega, mean = np.array(r["node_covariance"]), np.array(r["node_means"])
    weight = np.linalg.solve(omega[:6, :6], omega[:6, 6:]).T
    expected = mean[6:] + (frame.to_numpy() - mean[:6]) @ weight.T
    scores = latent_scores(result)
    assert scores.index.identical(frame.index)
    np.testing.assert_allclose(scores, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        scores.attrs["conditional_covariance"], omega[6:, 6:] - weight @ omega[:6, 6:], rtol=2e-12
    )
    subset = frame.iloc[[2, 5, 7]]
    np.testing.assert_allclose(
        latent_scores(result, subset), expected[[2, 5, 7]], rtol=2e-12, atol=2e-12
    )


def test_standardized_solution_joint_covariance_has_independent_delta(raw):
    _, result = raw
    r = result.attrs["sem_state"]["results"]
    values, names = np.array(r["parameters"]), r["parameter_names"]
    standardized_names = r["standardized_parameter_names"]

    def standard(v):
        omega, _, b, psi, intercept, _ = named_moments(dict(zip(names, v)))
        scale = np.sqrt(omega.diagonal())
        out = []
        for name in standardized_names:
            family, expression = name.split(":", 1)
            if family in ("loading", "path"):
                target, source = expression.split("<-")
                i, j = NODES.index(target), NODES.index(source)
                out.append(b[i, j] * scale[j] / scale[i])
            elif family == "variance":
                i = NODES.index(expression)
                out.append(psi[i, i] / scale[i] ** 2)
            elif family == "covariance":
                left, right = expression.split(",")
                i, j = NODES.index(left), NODES.index(right)
                out.append(psi[i, j] / scale[i] / scale[j])
            else:
                i = NODES.index(expression)
                out.append(intercept[i] / scale[i])
        return np.array(out)

    jac = np.zeros((len(standardized_names), len(values)))
    for j in range(len(values)):
        delta = np.zeros(len(values))
        delta[j] = 1e-5
        jac[:, j] = (standard(values + delta) - standard(values - delta)) / (2e-5)
    np.testing.assert_allclose(
        r["standardized_parameters"], standard(values), rtol=2e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        r["standardized_covariance"], jac @ np.array(r["covariance"]) @ jac.T, rtol=2e-7, atol=1e-10
    )
    assert result["standardized_parameters"].loc["loading:x1<-first", "std_error"] > 0
    assert result["standardized_parameters"].loc["variance:first", "std_error"] == 0


def test_observed_path_model_is_joint_random_exogenous_and_saturated():
    b = np.array([[0, 0, 0], [0.4, 0, 0], [0.2, 0.6, 0]])
    a = np.linalg.inv(np.eye(3) - b)
    sigma = a @ np.diag([1.0, 0.5, 0.7]) @ a.T
    result = latent_sem_covariance(
        sigma.tolist(),
        columns=["x", "m", "y"],
        n=500,
        divisor="n",
        means=[1.0, 2.0, 3.0],
        paths={"m": {"x": None}, "y": {"x": None, "m": None}},
        tolerance=1e-8,
    )
    np.testing.assert_allclose(result["paths"], b, atol=2e-7)
    diag = result.attrs["sem_state"]["results"]["diagnostics"]
    assert diag["df"] == 0 and diag["p_value"] is None
    assert diag["RMSEA"] == 0.0 and diag["TLI"] == 1.0
    assert result["parameters"].loc["variance:x", "std_error"] > 0
    assert result["parameters"].loc["intercept:x", "std_error"] > 0


def test_independence_baseline_zero_excess_uses_reference_boundary_rules():
    result = latent_sem_covariance(
        np.eye(3).tolist(),
        columns=["a", "b", "c"],
        n=300,
        divisor="n",
        means=[0.0, 0.0, 0.0],
        tolerance=1e-8,
    )
    diag = result.attrs["sem_state"]["results"]["diagnostics"]
    assert diag["df"] == 3
    assert diag["CFI"] == 1.0 and diag["TLI"] == pytest.approx(0, abs=1e-10)
    assert diag["RMSEA"] == pytest.approx(0, abs=1e-10)


def test_negative_baseline_excess_keeps_finite_ordinary_tli():
    result = latent_sem_covariance(
        [[1.0, 0.0], [0.0, 1.0]],
        columns=["x", "y"],
        n=303,
        divisor="n",
        meanstructure=False,
        residual_covariances={("x", "y"): 0.1},
        tolerance=1e-8,
    )
    d = result.attrs["sem_state"]["results"]["diagnostics"]
    assert d["df"] == 1 and d["baseline_df"] == 1
    assert d["baseline_statistic"] == 0
    assert d["TLI"] == pytest.approx(d["statistic"], abs=1e-12)
    assert d["TLI"] > 1 and d["CFI"] == 0


def test_constrained_means_contribute_to_joint_srmr_and_covariance_ml():
    sample = np.diag([1.0, 2.0, 3.0])
    means = np.array([0.3, -0.4, 0.5])
    result = latent_sem_covariance(
        sample.tolist(),
        columns=["a", "b", "c"],
        n=400,
        divisor="n",
        means=means.tolist(),
        intercepts={"a": 0.0, "b": 0.0, "c": 0.0},
        tolerance=1e-8,
    )
    fit = result.attrs["sem_state"]["results"]
    implied = np.diag(sample.diagonal() + means**2)
    np.testing.assert_allclose(fit["implied_covariance"], implied, atol=1e-7)
    covariance_squares = (
        (sample - implied) / np.sqrt(np.outer(sample.diagonal(), sample.diagonal()))
    )[np.tril_indices(3)] ** 2
    mean_squares = means**2 / sample.diagonal()
    d = fit["diagnostics"]
    assert d["df"] == 6
    assert d["SRMR"] == pytest.approx(
        np.sqrt((covariance_squares.sum() + mean_squares.sum()) / 9), abs=1e-8
    )
    assert d["SRMR_covariance"] == pytest.approx(np.sqrt(covariance_squares.mean()), abs=1e-8)
    assert d["standardized_mean_residual_rms"] == pytest.approx(
        np.sqrt(mean_squares.mean()), abs=1e-12
    )
    profiled = latent_sem_covariance(
        sample.tolist(),
        columns=["a", "b", "c"],
        n=400,
        divisor="n",
        meanstructure=False,
        tolerance=1e-8,
    ).attrs["sem_state"]["results"]["diagnostics"]
    assert profiled["SRMR"] == profiled["SRMR_covariance"]
    assert profiled["standardized_mean_residual_rms"] is None


def test_unit_disturbance_total_variance_and_marker_scaling_are_distinct(exact):
    marker = exact.attrs["sem_state"]["results"]
    unit = fit_summary(OMEGA[:6, :6], MEAN[:6], identification="unit_disturbance")
    u = unit.attrs["sem_state"]["results"]
    np.testing.assert_allclose(
        u["implied_covariance"], marker["implied_covariance"], rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        u["standardized_paths"], marker["standardized_paths"], rtol=2e-6, atol=1e-7
    )
    assert u["disturbance_covariance"][6][6] == 1.0 and u["disturbance_covariance"][7][7] == 1.0
    assert u["node_covariance"][7][7] > 1.1


def test_raw_unit_equality_labels_use_one_parameter():
    true = dict(TRUE)
    true["loading:x3<-first"] = 0.8
    omega, mean, *_ = named_moments(true)
    result = fit_summary(
        omega[:6, :6],
        mean[:6],
        equalities={"loading:x2<-first": "same", "loading:x3<-first": "same"},
    )
    assert result["parameters"].loc["equality:same", "estimate"] == pytest.approx(0.8, abs=1e-6)
    assert result.attrs["sem_state"]["results"]["diagnostics"]["df"] == 8
    assert result["paths"].loc["x2", "first"] == result["paths"].loc["x3", "first"]
