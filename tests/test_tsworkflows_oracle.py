"""Independent development-only density/NLS/recursion references for MARKET-129.

NumPy/SciPy/statsmodels in this file are test oracles, never runtime adapters.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.stats import norm
import openecon as oe
from openecon.models import ModelSpec, ResultBundle
from openecon.analysis_contracts import AnalysisError

torch.set_num_threads(1)


def density_reference(y, system):
    """Build the complete joint Gaussian distribution, not a Kalman recursion."""
    Z, T, Q, H = (np.asarray(system[k], dtype=float) for k in ("Z", "T", "Q", "H"))
    a, P = np.asarray(system["a0"], float), np.asarray(system["P0"], float)
    c, d = (
        np.asarray(system.get("c", np.zeros(len(a)))),
        np.asarray(system.get("d", np.zeros(len(Z)))),
    )
    n, p = y.shape
    means, powers, covs = [], [np.linalg.matrix_power(T, i) for i in range(n)], []
    for i in range(n):
        means.append(Z @ a + d)
        covs.append(P)
        a, P = T @ a + c, T @ P @ T.T + Q
    C = np.zeros((n * p, n * p))
    for i in range(n):
        for j in range(i + 1):
            block = Z @ powers[i - j] @ covs[j] @ Z.T
            if i == j:
                block += H
            C[i * p : (i + 1) * p, j * p : (j + 1) * p] = block
            C[j * p : (j + 1) * p, i * p : (i + 1) * p] = block.T
    error = y.reshape(-1) - np.asarray(means).reshape(-1)
    ll = -0.5 * (
        len(error) * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + error @ np.linalg.solve(C, error)
    )
    return ll, C, np.asarray(means)


def kalman_reference(y, s):
    Z, T, Q, H = (np.array(s[k], dtype=float) for k in ("Z", "T", "Q", "H"))
    a, P = np.array(s["a0"], float), np.array(s["P0"], float)
    c, d = np.array(s.get("c", np.zeros(len(a)))), np.array(s.get("d", np.zeros(len(Z))))
    filtered, covs, means, innovations = [], [], [], []
    for obs in y:
        mean = Z @ a + d
        F = Z @ P @ Z.T + H
        K = P @ Z.T @ np.linalg.inv(F)
        a = a + K @ (obs - mean)
        P = P - K @ Z @ P
        filtered.append(a.copy())
        covs.append(P.copy())
        means.append(mean)
        innovations.append(F)
        a, P = T @ a + c, T @ P @ T.T + Q
    return np.array(filtered), np.array(covs), np.array(means), a, P, np.array(innovations)


def hessian(fn, x, h=2e-4):
    H = np.empty((len(x), len(x)))
    e = np.eye(len(x)) * h
    for i in range(len(x)):
        for j in range(len(x)):
            H[i, j] = (
                fn(x + e[i] + e[j])
                - fn(x + e[i] - e[j])
                - fn(x - e[i] + e[j])
                + fn(x - e[i] - e[j])
            ) / (4 * h * h)
    return H


def fixed_system():
    return {
        "Z": [[1.0, 0.2], [-0.3, 1.0]],
        "T": [[0.65, 0.1], [0.0, 0.4]],
        "Q": [[0.2, 0.03], [0.03, 0.15]],
        "H": [[0.4, 0.05], [0.05, 0.3]],
        "a0": [0.3, -0.2],
        "P0": [[0.5, 0.1], [0.1, 0.4]],
        "c": [0.02, -0.01],
        "d": [0.1, 0.2],
    }


def test_general_multivariate_likelihood_state_and_forecast():
    rng = np.random.default_rng(129)
    y = rng.normal(size=(22, 2))
    s = fixed_system()
    result = oe.sspace(data={"a": y[:, 0], "b": y[:, 1]}, y="a", responses=["b"], system=s)
    ll, _, _ = density_reference(y, s)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, abs=1e-10)
    states, covs, means, a, P, F = kalman_reference(y, s)
    assert_allclose(result.extra["filtered"], states, atol=1e-12)
    assert_allclose(result.extra["filtered_covariance"], covs, atol=1e-12)
    assert_allclose(result.extra["predicted"], means, atol=1e-12)
    assert_allclose(result.extra["innovation_covariance"], F, atol=1e-12)
    future = oe.forecast(result, 6)
    Z, T, Q, H = (np.array(s[k]) for k in ("Z", "T", "Q", "H"))
    for i in range(6):
        C = Z @ P @ Z.T + H
        mu = Z @ a + np.array(s["d"])
        assert_allclose(future.attrs["measurement_covariance"][i], C, atol=1e-12)
        assert_allclose(future.attrs["state_covariance"][i], P, atol=1e-12)
        assert_allclose(future.attrs["measurement_mean"][i], mu, atol=1e-12)
        assert future.iloc[i].std_error == pytest.approx(np.sqrt(C[0, 0]), abs=1e-12)
        assert future.iloc[i].ci_high == pytest.approx(
            mu[0] + norm.ppf(0.975) * np.sqrt(C[0, 0]), abs=1e-10
        )
        a, P = T @ a + np.array(s["c"]), T @ P @ T.T + Q
    assert ResultBundle.model_validate_json(result.model_dump_json()).extra == result.extra
    assert result.coefficients == [] and result.inference["available"] is False


def test_sspace_full_parameter_covariance_against_joint_density():
    rng = np.random.default_rng(215)
    n = 42
    phi = 0.57
    q = 0.7
    H = 0.3
    state = rng.normal(scale=np.sqrt(q / (1 - phi**2)))
    y = []
    for _ in range(n):
        y.append(state + rng.normal(scale=np.sqrt(H)))
        state = phi * state + rng.normal(scale=np.sqrt(q))
    system = {
        "Z": [[1.0]],
        "T": [[0.4]],
        "Q": [[0.6]],
        "H": [[H]],
        "initialization": "stationary",
        "parameters": [
            {"name": "phi", "matrix": "T", "row": 0, "col": 0, "transform": "unit"},
            {"name": "q", "matrix": "Q", "row": 0, "col": 0, "transform": "positive"},
        ],
    }
    result = oe.sspace(data={"y": y}, y="y", system=system)

    def objective(z):
        phi, q = z
        if abs(phi) >= 0.99 or q <= 0:
            return 1e50
        C = (
            q / (1 - phi**2) * phi ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
            + np.eye(n) * H
        )
        return 0.5 * (
            n * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + np.asarray(y) @ np.linalg.solve(C, y)
        )

    oracle = minimize(
        objective, [0.4, 0.6], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-11}
    )
    actual = np.array([c.estimate for c in result.coefficients])
    assert_allclose(actual, oracle.x, atol=3e-6)
    assert_allclose(
        result.covariance_matrix, np.linalg.inv(hessian(objective, actual)), rtol=2e-5, atol=2e-6
    )
    assert result.provenance["optimizer"]["converged"] is True


@pytest.mark.parametrize("model", ["ANN", "AAN", "AdN", "ANA", "AAA", "AdA"])
def test_ets_all_additive_models_against_scalar_published_equations(model):
    rng = np.random.default_rng(84)
    n = 84
    period = 4
    fixed = {"alpha": 0.32}
    trend = model not in {"ANN", "ANA"}
    seasonal = model.endswith("A")
    damped = "d" in model
    if trend:
        fixed["beta"] = 0.08
    if seasonal:
        fixed["gamma"] = 0.12
    if damped:
        fixed["phi"] = 0.91
    init = [2.0] + ([0.1] if trend else []) + ([-0.3, 0.2, 0.4, -0.3] if seasonal else [])
    y = (
        3
        + np.arange(n) * 0.06
        + rng.normal(scale=0.5, size=n)
        + (np.resize([-0.3, 0.2, 0.4, -0.3], n) if seasonal else 0)
    )
    result = oe.ets(data={"y": y}, y="y", model=model, period=period, fixed=fixed, initial=init)
    level = init[0]
    b = init[1] if trend else 0
    seasons = list(init[-period:]) if seasonal else [0] * period
    phi = fixed.get("phi", 1.0)
    means = []
    errors = []
    for t, obs in enumerate(y):
        old = seasons[t % period]
        mean = level + phi * b + old
        error = obs - mean
        level = level + phi * b + fixed["alpha"] * error
        b = phi * b + fixed.get("beta", 0) * error
        if seasonal:
            seasons[t % period] = old + fixed["gamma"] * error
        means.append(mean)
        errors.append(error)
    sigma = np.sqrt(np.mean(np.square(errors)))
    assert result.coefficients[0].term == "sigma"
    assert result.coefficients[0].estimate == pytest.approx(sigma, rel=2e-6)
    assert result.covariance_matrix[0][0] == pytest.approx(sigma**2 / (2 * n), rel=3e-6)
    assert_allclose([r["fitted"] for r in result.predictions], means, atol=1e-12)
    assert result.metrics["log_likelihood"] == pytest.approx(
        -n / 2 * (np.log(2 * np.pi * sigma**2) + 1), abs=1e-8
    )

    # Forecast covariance independently obtained by perturbing each future shock.
    end_level, end_slope = level, b

    def paths(shocks):
        level, slope = end_level, end_slope
        season = seasons.copy()
        output = []
        for h, shock in enumerate(shocks):
            t = n + h
            mu = level + phi * slope + season[t % period]
            output.append(mu + shock)
            level = level + phi * slope + fixed["alpha"] * shock
            slope = phi * slope + fixed.get("beta", 0) * shock
            if seasonal:
                season[t % period] += fixed["gamma"] * shock
        return np.array(output)

    base = paths(np.zeros(10))
    impulse = np.column_stack([paths(np.eye(10)[j]) - base for j in range(10)])
    future = oe.forecast(result, 10)
    assert_allclose(future.forecast, base, atol=1e-12)
    assert_allclose(future.std_error, sigma * np.sqrt(np.sum(impulse**2, axis=1)), rtol=3e-6)
    assert ResultBundle.model_validate_json(result.model_dump_json()).spec == result.spec


def test_ets_estimated_initial_level_full_information_closed_form():
    rng = np.random.default_rng(85)
    y = rng.normal(size=80).cumsum() * 0.15 + 5
    alpha = 0.27
    # y_hat_t = (1-alpha)^t*l0 + alpha sum_(j<t) (1-alpha)^(t-1-j)*y_j
    v = (1 - alpha) ** np.arange(len(y))
    offset = np.zeros(len(y))
    for t in range(1, len(y)):
        offset[t] = (1 - alpha) * offset[t - 1] + alpha * y[t - 1]
    l0 = v @ (y - offset) / (v @ v)
    error = y - offset - v * l0
    sigma = np.sqrt(error @ error / len(y))
    result = oe.ets(data={"y": y}, y="y", fixed={"alpha": alpha})
    assert_allclose([c.estimate for c in result.coefficients], [l0, sigma], atol=2e-6)
    assert_allclose(
        result.covariance_matrix,
        np.diag([sigma**2 / (v @ v), sigma**2 / (2 * len(y))]),
        rtol=5e-6,
        atol=1e-8,
    )


def aligned_midas():
    rng = np.random.default_rng(93)
    high = pd.DataFrame(
        {"t": pd.date_range("1990-01-01", periods=360, freq="MS"), "z": rng.normal(size=360)}
    )
    high["released"] = high.t + pd.Timedelta(days=12)
    low = pd.DataFrame(
        {"t": pd.date_range("1991-01-01", periods=80, freq="QS"), "x": rng.normal(size=80)}
    )
    low["origin"] = low.t - pd.Timedelta(days=20)
    out = oe.midas_align(
        low=low,
        high=high,
        low_time="t",
        high_time="t",
        value="z",
        lags=6,
        frequency="MS",
        origin="origin",
        release_time="released",
    )
    j = np.linspace(0, 1, 6)
    w = np.exp(-3 * j + 1.1 * j * j)
    w /= w.sum()
    names = out.attrs["midas_alignment"]["lags"]
    out["y"] = 1.1 + 0.4 * out.x + 2 * (out[names].to_numpy() @ w) + rng.normal(scale=0.15, size=80)
    return out, high, low, names


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1"])
def test_midas_coefficients_full_nls_covariance_and_prediction(covariance):
    data, _, _, names = aligned_midas()
    X = np.column_stack([np.ones(len(data)), data.x])
    L = data[names].to_numpy()
    y = data.y.to_numpy()
    j = np.linspace(0, 1, 6)

    def means(z):
        w = np.exp(z[-2] * j + z[-1] * j * j)
        w /= w.sum()
        return X @ z[:2] + z[2] * (L @ w)

    def objective(z):
        return 0.5 * np.sum((y - means(z)) ** 2)

    oracle = minimize(objective, [1.0, 0.4, 2.0, -2.0, 1.0], method="BFGS", options={"gtol": 1e-7})
    result = oe.midas(data=data, y="y", x=["x"], covariance=covariance)
    theta = np.array([c.estimate for c in result.coefficients])
    assert_allclose(theta, oracle.x, atol=2e-4)
    eps = 1e-5
    J = np.column_stack(
        [(means(theta + e) - means(theta - e)) / (2 * eps) for e in np.eye(5) * eps]
    )
    r = y - means(theta)
    if covariance == "nonrobust":
        V = np.linalg.inv(J.T @ J) * (r @ r / (len(y) - 5))
    else:
        bread = np.linalg.inv(hessian(objective, theta))
        scores = J * r[:, None]
        V = bread @ (scores.T @ scores) @ bread
        if covariance == "HC1":
            V *= len(y) / (len(y) - 5)
    assert_allclose(result.covariance_matrix, V, rtol=2e-4, atol=1e-7)
    predicted = oe.midas_predict(result, data=data)
    assert_allclose(predicted.forecast, means(theta), atol=1e-10)
    assert_allclose(predicted.std_error, np.sqrt(np.sum(J @ V * J, axis=1)), rtol=2e-4, atol=1e-8)
    assert "parameter uncertainty only" in predicted.attrs["uncertainty"]
    assert "Exponential" in result.to_latex() or "Observations" in result.to_latex()


def test_midas_calendar_no_future_release_and_bound_metadata():
    aligned, high, low, names = aligned_midas()
    high.loc[high.t > low.origin.max(), "z"] = 1e9
    again = oe.midas_align(
        low=low,
        high=high,
        low_time="t",
        high_time="t",
        value="z",
        lags=6,
        frequency="MS",
        origin="origin",
        release_time="released",
    )
    assert_allclose(again[names], aligned[names])
    for i, row in enumerate(aligned.attrs["midas_alignment"]["rows"]):
        for date in row["release_times"]:
            assert pd.Timestamp(date) <= low.origin.iloc[i]
        expected = (
            high.loc[(high.t <= low.origin.iloc[i]) & (high.released <= low.origin.iloc[i])]
            .tail(6)
            .z.to_numpy()[::-1]
        )
        assert_allclose(aligned.loc[i, names].astype(float), expected)
    altered = aligned.copy()
    altered.loc[0, names[0]] += 1
    with pytest.raises(AnalysisError, match="changed after alignment"):
        oe.midas(data=altered, y="y", x=["x"])


def test_auto_arima_criterion_sample_and_existing_forecast_contract():
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rng = np.random.default_rng(12)
    y = rng.normal(size=120)
    for i in range(1, len(y)):
        y[i] += 0.7 * y[i - 1]
    data = pd.DataFrame({"y": y, "time": np.arange(120)})
    result = oe.auto_arima(
        data=data, y="y", time="time", d=0, max_p=1, max_q=0, constant=False, criterion="aicc"
    )
    records = result.extra["auto_selection"]["candidates"]
    for row in records:
        model = SARIMAX(y, order=tuple(row["order"]), initialization="stationary").fit(disp=False)
        assert row["status"] == "ok"
        assert row["log_likelihood"] == pytest.approx(model.llf, abs=2e-6)
        k = row["parameters"]
        n = row["nobs"]
        assert row["aicc"] == pytest.approx(
            -2 * model.llf + 2 * k + 2 * k * (k + 1) / (n - k - 1), abs=4e-6
        )
    assert result.spec.estimator == "arima" and result.spec.options["order"] == [1, 0, 0]
    assert result.sample_positions == list(range(120))
    direct = oe.arima(
        data=data, y="y", time="time", order=(1, 0, 0), constant=False, covariance="nonrobust"
    )
    assert_allclose(oe.forecast(result, 4).forecast, oe.forecast(direct, 4).forecast, atol=1e-12)


def test_rolling_and_recursive_use_only_origin_rows():
    rng = np.random.default_rng(21)
    data = pd.DataFrame({"y": rng.normal(size=30), "x": rng.normal(size=30), "t": np.arange(30)})
    spec = ModelSpec(outcome="y", predictors=["x"], time="t")
    result = oe.rolling(spec, data=data, window=12, step=6)
    for record in result.attrs["origins"]:
        start, stop = record["start"], record["stop_exclusive"]
        X = np.column_stack([np.ones(stop - start), data.x.iloc[start:stop]])
        y = data.y.iloc[start:stop]
        actual = result["coefficients"].loc[lambda d: d.origin == record["origin"]]
        estimates = dict(zip(actual.term, actual.estimate, strict=True))
        oracle = np.linalg.lstsq(X, y, rcond=None)[0]
        assert estimates["Intercept"] == pytest.approx(oracle[0], abs=1e-12)
        assert estimates["x"] == pytest.approx(oracle[1], abs=1e-12)
        assert record["input_positions"] == list(range(start, stop))
    changed = data.copy()
    changed.loc[24:, "y"] = 1e9
    modified = oe.rolling(spec, data=changed, window=12, step=6)
    assert_allclose(
        result["coefficients"].loc[lambda d: d.origin < 24].estimate,
        modified["coefficients"].loc[lambda d: d.origin < 24].estimate,
    )
    recursive = oe.recursive(spec, data=data, minimum=12, step=6)
    assert all(r["start"] == 0 for r in recursive.attrs["origins"])


@pytest.mark.parametrize(
    "change,code",
    [
        ({"Q": [[-1.0]]}, "invalid_covariance"),
        ({"H": [[0.0]]}, "invalid_covariance"),
        ({"T": [[1.2]]}, "unstable_system"),
        ({"initialization": "diffuse"}, "invalid_initialization"),
        ({"Z": [[0.0]]}, "unidentified_system"),
    ],
)
def test_system_constraints_raise_without_projection(change, code):
    s = {"Z": [[1.0]], "T": [[0.5]], "Q": [[0.2]], "H": [[0.3]], **change}
    with pytest.raises(AnalysisError) as exc:
        oe.sspace(data={"y": [1.0, 2.0, 3.0]}, y="y", system=s)
    assert exc.value.code == code


def test_explicit_search_and_geometry_guards():
    with pytest.raises(AnalysisError, match="Search needs"):
        oe.auto_arima(data={"y": list(range(100))}, y="y", d=0, max_p=8, max_q=8)
    with pytest.raises(AnalysisError, match="Rolling requires"):
        oe.rolling(
            ModelSpec(outcome="y", predictors=[]),
            data={"y": list(range(100))},
            window=2,
            max_fits=3,
        )
    with pytest.raises(AnalysisError):
        oe.sspace(data={"y": [1.0, 2.0]}, y="y", system={"T": np.eye(17).tolist()})
    with pytest.raises(AnalysisError):
        oe.ets(data={"y": list(range(60))}, y="y", model="AAA", fixed={"alpha": 0.9, "gamma": 0.4})


def test_full_secondary_measurement_forecast_overflow_is_an_error():
    system = {
        "Z": [[0.0], [1e308]],
        "T": [[0.5]],
        "Q": [[1.0]],
        "H": [[1.0, 0.0], [0.0, 1.0]],
        "a0": [0.0],
        "P0": [[0.0]],
    }
    result = oe.sspace(
        data={"first": [0.0], "second": [0.0]}, y="first", responses=["second"], system=system
    )
    with pytest.raises(AnalysisError) as exc:
        oe.forecast(result, 1)
    assert exc.value.code == "non_finite_forecast"


def test_stationary_system_preflights_lyapunov_before_kron(monkeypatch):
    from openecon.resources import use_workspace_budget

    system = {
        "Z": [[1.0] * 16],
        "T": np.diag(np.linspace(0.1, 0.8, 16)).tolist(),
        "Q": np.eye(16).tolist(),
        "H": [[1.0]],
        "initialization": "stationary",
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Lyapunov allocation occurred before its budget guard")

    monkeypatch.setattr(torch, "kron", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        oe.sspace(data={"y": [0.0, 1.0, 0.0]}, y="y", system=system)
    assert "budget" in str(exc.value).lower()


def test_free_ets_boundary_does_not_manufacture_interior_ci():
    with pytest.raises(AnalysisError) as exc:
        oe.ets(data={"y": np.arange(100, dtype=float)}, y="y", model="ANN", initial=[0.0])
    assert exc.value.code in {"boundary_solution", "singular_information", "nonconvergence"}


def test_conditional_forecast_uncertainty_survives_publication_table():
    result = oe.ets(data={"y": [1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.6, 2.8]}, y="y",
                    fixed={"alpha": 0.3}, initial=[1.0])
    future = oe.forecast(result, 3)
    assert future.attrs["uncertainty"] in future.attrs["publication_notes"]
    source = oe.to_latex(future)
    assert "parameter uncertainty excluded" in source
    assert "excluded from conditional forecast intervals" in source
