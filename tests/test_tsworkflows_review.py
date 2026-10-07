"""Independent review oracles for MARKET-129; development-only NumPy/SciPy."""

import copy
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ModelSpec, ResultBundle


def gaussian_system():
    return {
        "T": [[0.72, 0.11], [-0.04, 0.51]],
        "Z": [[1.0, 0.3], [-0.2, 1.0]],
        "Q": [[0.25, 0.04], [0.04, 0.15]],
        "H": [[0.18, 0.025], [0.025, 0.22]],
        "a0": [0.7, -0.3],
        "P0": [[0.5, 0.1], [0.1, 0.3]],
        "c": [0.03, -0.02],
        "d": [0.12, -0.08],
    }


def numpy_kalman(y, system):
    a, p = np.array(system["a0"]), np.array(system["P0"])
    t, z, q, h, c, d = (np.array(system[key]) for key in ["T", "Z", "Q", "H", "c", "d"])
    predicted, filtered, filtered_cov, ll = [], [], [], 0.0
    for observation in y:
        mean = z @ a + d
        error = observation - mean
        f = z @ p @ z.T + h
        ll += -0.5 * (
            len(observation) * np.log(2 * np.pi)
            + np.linalg.slogdet(f)[1]
            + error @ np.linalg.solve(f, error)
        )
        gain = p @ z.T @ np.linalg.inv(f)
        a = a + gain @ error
        p = p - gain @ z @ p
        p = (p + p.T) / 2
        predicted.append(mean)
        filtered.append(a.copy())
        filtered_cov.append(p.copy())
        a, p = t @ a + c, t @ p @ t.T + q
    return dict(
        ll=ll,
        predicted=np.array(predicted),
        filtered=np.array(filtered),
        covariance=np.array(filtered_cov),
        next_mean=a,
        next_covariance=p,
    )


def test_multivariate_known_state_filter_likelihood_and_all_forecast_uncertainties():
    system = gaussian_system()
    rng = np.random.default_rng(14)
    y = rng.normal(size=(30, 2))
    result = oe.sspace(
        data=pd.DataFrame({"y": y[:, 0], "other": y[:, 1]}),
        y="y",
        responses=["other"],
        system=system,
    )
    oracle = numpy_kalman(y, system)
    np.testing.assert_allclose(result.extra["predicted"], oracle["predicted"], atol=1e-12)
    np.testing.assert_allclose(result.extra["filtered"], oracle["filtered"], atol=1e-12)
    np.testing.assert_allclose(
        result.extra["filtered_covariance"], oracle["covariance"], atol=1e-12
    )
    assert result.metrics["log_likelihood"] == pytest.approx(oracle["ll"], abs=1e-11)
    assert not result.coefficients and result.inference["available"] is False
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    future = oe.forecast(saved, 8, alpha=0.1)
    a, p = oracle["next_mean"], oracle["next_covariance"]
    t, z, q, h, c, d = (np.array(system[key]) for key in ["T", "Z", "Q", "H", "c", "d"])
    expected_mean, expected_cov = [], []
    for _ in range(8):
        expected_mean.append(z @ a + d)
        expected_cov.append(z @ p @ z.T + h)
        a, p = t @ a + c, t @ p @ t.T + q
    np.testing.assert_allclose(future.attrs["measurement_mean"], expected_mean, atol=1e-12)
    np.testing.assert_allclose(future.attrs["measurement_covariance"], expected_cov, atol=1e-12)
    np.testing.assert_allclose(
        future.std_error, np.sqrt(np.array(expected_cov)[:, 0, 0]), atol=1e-12
    )
    critical = NormalDist().inv_cdf(0.95)
    np.testing.assert_allclose(
        future.ci_low, np.array(expected_mean)[:, 0] - critical * future.std_error, atol=1e-12
    )
    assert "parameter uncertainty excluded" in future.attrs["uncertainty"]


def test_known_stationary_prior_independently_solves_lyapunov():
    system = gaussian_system()
    system["initialization"] = "stationary"
    result = oe.sspace(
        data={"y": [1.0, -0.3, 0.8, 0.7], "other": [0.4, -0.1, 0.2, 0.5]},
        y="y",
        responses=["other"],
        system=system,
    )
    t, q, c = np.array(system["T"]), np.array(system["Q"]), np.array(system["c"])
    stationary_a = np.linalg.solve(np.eye(2) - t, c)
    stationary_p = np.linalg.solve(np.eye(4) - np.kron(t, t), q.reshape(-1)).reshape(2, 2)
    np.testing.assert_allclose(result.extra["system"]["a0"], stationary_a, atol=1e-12)
    np.testing.assert_allclose(result.extra["system"]["P0"], stationary_p, atol=1e-12)


def test_state_space_free_measurement_mean_variance_full_closed_form_covariance():
    y = np.random.default_rng(798).normal(2.0, 1.3, size=100)
    system = {
        "T": [[0.0]],
        "Z": [[1.0]],
        "Q": [[0.0]],
        "H": [[2.0]],
        "P0": [[0.0]],
        "a0": [0.0],
        "d": [1.0],
        "parameters": [
            {"name": "mean", "matrix": "d", "row": 0},
            {"name": "variance", "matrix": "H", "row": 0, "col": 0, "transform": "positive"},
        ],
    }
    result = oe.sspace(data={"y": y}, y="y", system=system, tolerance=1e-9)
    mean, variance = y.mean(), np.mean((y - y.mean()) ** 2)
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients], [mean, variance], rtol=1e-6
    )
    np.testing.assert_allclose(
        result.covariance_matrix, [[variance / len(y), 0], [0, 2 * variance**2 / len(y)]], atol=1e-7
    )
    ll = -0.5 * len(y) * (np.log(2 * np.pi * variance) + 1)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, abs=1e-6)


@pytest.mark.parametrize("model", ["ANN", "AAN", "AdN", "ANA", "AAA", "AdA"])
def test_all_additive_ets_known_smoothing_sigma_ml_and_forecast_covariance(model):
    trend, damped, seasonal = model not in {"ANN", "ANA"}, "d" in model, model.endswith("A")
    period, alpha, beta, gamma, phi = 3, 0.3, 0.07, 0.1, 0.87 if damped else 1.0
    m = 1 + int(trend) + (period if seasonal else 0)
    f, w, gain = np.zeros((m, m)), np.zeros(m), np.zeros(m)
    f[0, 0], w[0], gain[0] = 1.0, 1.0, alpha
    state = [1.0]
    fixed = {"alpha": alpha}
    if trend:
        f[0, 1], f[1, 1], w[1], gain[1] = phi, phi, phi, beta
        state.append(0.08)
        fixed["beta"] = beta
    if damped:
        fixed["phi"] = phi
    if seasonal:
        offset = 1 + int(trend)
        f[offset:-1, offset + 1 :] = np.eye(period - 1)
        f[-1, offset], w[offset], gain[-1] = 1.0, 1.0, gamma
        state.extend([0.2, -0.1, -0.1])
        fixed["gamma"] = gamma
    initial = np.array(state)
    errors = np.random.default_rng(891).normal(0, 0.4, size=90)
    observations, predictions = [], []
    state = initial.copy()
    for error in errors:
        predicted = w @ state
        observations.append(predicted + error)
        predictions.append(predicted)
        state = f @ state + gain * error
    result = oe.ets(
        data={"y": observations},
        y="y",
        model=model,
        period=period,
        fixed=fixed,
        initial=initial.tolist(),
        tolerance=1e-9,
    )
    sigma2 = np.mean(errors**2)
    assert [c.term for c in result.coefficients] == ["sigma"]
    assert result.coefficients[0].estimate == pytest.approx(np.sqrt(sigma2), rel=1e-6)
    assert result.covariance_matrix[0][0] == pytest.approx(sigma2 / (2 * len(errors)), rel=1e-5)
    np.testing.assert_allclose([p["fitted"] for p in result.predictions], predictions, atol=1e-12)
    forecast = oe.forecast(result, 7)
    covariance = np.zeros_like(f)
    means, variances = [], []
    for _ in range(7):
        means.append(w @ state)
        variances.append(w @ covariance @ w + sigma2)
        state = f @ state
        covariance = f @ covariance @ f.T + sigma2 * np.outer(gain, gain)
    np.testing.assert_allclose(forecast.forecast, means, atol=1e-12)
    np.testing.assert_allclose(forecast.std_error**2, variances, rtol=2e-6)


def midas_fixture():
    rng = np.random.default_rng(1618)
    high = pd.DataFrame(
        {
            "month": pd.date_range("2000-01-01", periods=280, freq="MS"),
            "value": rng.normal(size=280),
        }
    )
    low = pd.DataFrame({"quarter": pd.date_range("2001-03-31", periods=80, freq="QE")})
    aligned = oe.midas_align(
        low=low,
        high=high,
        low_time="quarter",
        high_time="month",
        value="value",
        lags=8,
        frequency="MS",
    )
    names = aligned.attrs["midas_alignment"]["lags"]
    lag = aligned[names].to_numpy()
    j = np.linspace(0, 1, len(names))
    raw = 1.2 * j - 0.6 * j**2
    weights = np.exp(raw - raw.max())
    weights /= weights.sum()
    aligned["y"] = 0.7 + 2.4 * (lag @ weights) + rng.normal(size=len(low)) * 0.18
    return aligned, names


def numpy_midas(z, lag):
    j = np.linspace(0, 1, lag.shape[1])
    basis = np.column_stack((j, j**2))
    raw = basis @ z[-2:]
    weights = np.exp(raw - raw.max())
    weights /= weights.sum()
    centered = basis - weights @ basis
    dweights = weights[:, None] * centered
    weighted_basis_cov = centered.T @ (weights[:, None] * centered)
    d2weights = weights[:, None, None] * (
        centered[:, :, None] * centered[:, None, :] - weighted_basis_cov
    )
    aggregate = lag @ weights
    jacobian = np.column_stack((np.ones(len(lag)), aggregate, z[1] * (lag @ dweights)))
    hessian = np.zeros((len(lag), 4, 4))
    hessian[:, 1, -2:] = lag @ dweights
    hessian[:, -2:, 1] = lag @ dweights
    hessian[:, -2:, -2:] = z[1] * np.einsum("nk,kab->nab", lag, d2weights)
    return z[0] + z[1] * aggregate, jacobian, hessian, weights


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1"])
def test_midas_coefficients_and_full_covariance_against_independent_scipy_nls(covariance):
    from scipy.optimize import least_squares

    aligned, names = midas_fixture()
    result = oe.midas(data=aligned, y="y", covariance=covariance, tolerance=1e-9)
    lag, y = aligned[names].to_numpy(), aligned.y.to_numpy()
    reference = least_squares(
        lambda z: y - numpy_midas(z, lag)[0],
        [0.7, 2.0, 0.0, 0.0],
        jac=lambda z: -numpy_midas(z, lag)[1],
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=5000,
    )
    expected, jacobian, hessian, weights = numpy_midas(reference.x, lag)
    residuals = y - expected
    if covariance == "nonrobust":
        cov = np.linalg.inv(jacobian.T @ jacobian) * (residuals @ residuals) / (len(y) - 4)
    else:
        bread = np.linalg.inv(jacobian.T @ jacobian - np.einsum("n,nab->ab", residuals, hessian))
        scores = jacobian * residuals[:, None]
        cov = bread @ (scores.T @ scores) @ bread
        if covariance == "HC1":
            cov *= len(y) / (len(y) - 4)
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients], reference.x, rtol=3e-5, atol=3e-5
    )
    np.testing.assert_allclose(result.covariance_matrix, cov, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(result.extra["weights"], weights, atol=1e-6)
    future = oe.midas_predict(
        ResultBundle.model_validate_json(result.model_dump_json()), data=aligned
    )
    np.testing.assert_allclose(future.forecast, expected, atol=1e-6)
    np.testing.assert_allclose(
        future.std_error**2, np.sum((jacobian @ cov) * jacobian, axis=1), rtol=3e-4, atol=1e-6
    )
    assert "parameter uncertainty only" in future.attrs["uncertainty"]


def test_release_aware_midas_actual_available_lags_and_future_perturbation():
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    high = pd.DataFrame(
        {
            "date": dates,
            "value": np.arange(10.0),
            "release": [pd.Timestamp(date.year, date.month, 15) for date in dates],
        }
    )
    low = pd.DataFrame(
        {"target": pd.to_datetime(["2020-08-31"]), "origin": pd.to_datetime(["2020-07-10"])}
    )
    kwargs = dict(
        low=low,
        high=high,
        low_time="target",
        high_time="date",
        value="value",
        release_time="release",
        origin="origin",
        lags=3,
        frequency="MS",
    )
    aligned = oe.midas_align(**kwargs)
    np.testing.assert_allclose(aligned[["midas_L0", "midas_L1", "midas_L2"]], [[5.0, 4.0, 3.0]])
    changed = high.copy()
    changed.loc[changed.release > low.origin.iloc[0], "value"] += 5000
    again = oe.midas_align(**{**kwargs, "high": changed})
    np.testing.assert_allclose(again[["midas_L0", "midas_L1", "midas_L2"]], [[5.0, 4.0, 3.0]])


@pytest.mark.parametrize("frequency", ["-1D", "0D"])
def test_midas_calendar_rejects_nonpositive_offsets_before_generating_lags(frequency):
    high = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=12, freq="D"), "value": np.arange(12.0)}
    )
    high["release"] = pd.Timestamp("2019-12-01")
    low = pd.DataFrame({"date": [pd.Timestamp("2020-01-05")]})
    with pytest.raises(AnalysisError):
        oe.midas_align(
            low=low,
            high=high,
            low_time="date",
            high_time="date",
            value="value",
            release_time="release",
            lags=3,
            frequency=frequency,
        )


def test_midas_mixed_naive_aware_calendar_timestamps_are_explicitly_rejected():
    high = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=12, freq="D"), "value": np.arange(12.0)}
    )
    high["release"] = pd.date_range("2020-01-02", periods=12, freq="D", tz="UTC")
    low = pd.DataFrame({"date": [pd.Timestamp("2020-01-10")]})
    with pytest.raises(AnalysisError):
        oe.midas_align(
            low=low,
            high=high,
            low_time="date",
            high_time="date",
            value="value",
            release_time="release",
            lags=3,
            frequency="D",
        )


def test_alignment_metadata_target_origin_are_bound_to_actual_rows():
    aligned, names = midas_fixture()
    record = copy.deepcopy(aligned.attrs["midas_alignment"])
    record["rows"][0]["origin"] = "2099-01-01T00:00:00"
    with pytest.raises(AnalysisError):
        oe.midas(data=aligned, y="y", alignment=record)


def test_string_datetime_order_matches_datetime_chronology_or_is_explicitly_rejected():
    values = np.random.default_rng(18).normal(size=20)
    frame = pd.DataFrame({"y": values, "date": pd.date_range("2020-01-01", periods=20, freq="D")})
    system = {"T": [[0.7]], "Z": [[1.0]], "Q": [[0.3]], "H": [[0.2]], "a0": [0.0], "P0": [[0.5]]}
    expected = oe.sspace(data=frame, y="y", time="date", system=system)
    descending = frame.iloc[::-1].copy()
    descending["date"] = descending.date.astype(str)
    try:
        actual = oe.sspace(data=descending, y="y", time="date", system=system)
    except AnalysisError as error:
        assert error.code in {"invalid_time", "time_gaps"}
    else:
        assert actual.metrics["log_likelihood"] == pytest.approx(
            expected.metrics["log_likelihood"], abs=1e-11
        )


def test_scalar_many_role_is_one_column_not_character_expansion():
    system = gaussian_system()
    spec = ModelSpec(
        estimator="sspace",
        outcome="y",
        predictors=[],
        intercept=False,
        columns={"responses": "other"},
        options={"system": system},
    )
    result = oe.fit(spec, data={"y": [1.0, 0.5, 2.0], "other": [0.3, 0.4, 0.2]})
    assert result.extra["responses"] == ["y", "other"]


def test_ets_invalid_fixed_container_rejected_before_fit():
    with pytest.raises(AnalysisError):
        oe.ets(data={"y": np.random.default_rng(10).normal(size=50)}, y="y", fixed=[])


def test_feasible_partial_fixed_ets_initialization_respects_coupled_bounds():
    from openecon.econometrics.tsworkflows.ets import ETS
    from openecon.econometrics.core import ModelFrame

    spec = ModelSpec(
        estimator="ets",
        outcome="y",
        predictors=[],
        intercept=False,
        options={"model": "AAN", "fixed": {"beta": 0.6}, "initial": [0.0, 0.0]},
    )
    frame = ModelFrame(spec, {"y": np.random.default_rng(18).normal(size=80)})
    engine = ETS(frame, frame.numeric("y"))
    decoded = engine.decode(engine.start)
    assert float(decoded[-1]["alpha"]) >= 0.6


def test_auto_arima_constant_numeric_bool_coercion_is_rejected():
    y = np.random.default_rng(56).normal(size=30)
    with pytest.raises(AnalysisError):
        oe.auto_arima(data={"y": y}, y="y", d=0, max_p=0, max_q=0, constant=1)


def test_rolling_recursive_fit_windows_and_future_changes_do_not_affect_earlier_estimates():
    rng = np.random.default_rng(981)
    df = pd.DataFrame({"x": np.arange(25.0), "y": rng.normal(size=25)})
    spec = ModelSpec(outcome="y", predictors=["x"], covariance="HC0")
    rolling = oe.rolling(spec, data=df, window=10, step=5)
    recursive = oe.recursive(spec, data=df, minimum=10, step=5)
    assert [record["input_positions"] for record in rolling.attrs["origins"]] == [
        list(range(i - 10, i)) for i in [10, 15, 20, 25]
    ]
    assert [record["input_positions"] for record in recursive.attrs["origins"]] == [
        list(range(i)) for i in [10, 15, 20, 25]
    ]
    changed = df.copy()
    changed.iloc[15:, changed.columns.get_loc("y")] += 100
    again = oe.rolling(spec, data=changed, window=10, step=5)
    for before, after in zip(
        rolling["coefficients"].to_dict("records"),
        again["coefficients"].to_dict("records"),
        strict=True,
    ):
        if before["origin"] < 15:
            assert before == after
    for table, origin_records in [
        (rolling["coefficients"], rolling.attrs["origins"]),
        (recursive["coefficients"], recursive.attrs["origins"]),
    ]:
        for record in origin_records:
            positions = record["input_positions"]
            design = np.column_stack((np.ones(len(positions)), df.x.iloc[positions]))
            expected = np.linalg.lstsq(design, df.y.iloc[positions], rcond=None)[0]
            rows = table.loc[table.origin == record["origin"]]
            np.testing.assert_allclose(rows.estimate, expected, atol=1e-11)


def test_workflow_dataset_errors_are_explicit_no_materialization():
    aligned, names = midas_fixture()
    source = Dataset.from_frame(aligned)
    with pytest.raises(AnalysisError) as caught:
        oe.midas(data=source, y="y", lags=names, alignment=aligned.attrs["midas_alignment"])
    assert caught.value.code == "streaming_unsupported"


@pytest.mark.parametrize("criterion", ["aic", "aicc", "bic"])
def test_auto_arima_exact_white_noise_likelihood_parameter_count_and_criteria(criterion):
    y = np.random.default_rng(818).normal(1.4, 0.7, size=70)
    result = oe.auto_arima(
        data={"y": y}, y="y", d=0, max_p=0, max_q=0, constant="auto", criterion=criterion
    )
    selection = result.extra["auto_selection"]
    for row in selection["candidates"]:
        k = 2 if row["constant"] else 1
        residuals = y - y.mean() if row["constant"] else y
        variance = np.mean(residuals**2)
        loglike = -0.5 * len(y) * (np.log(2 * np.pi * variance) + 1)
        expected_aic = -2 * loglike + 2 * k
        expected_aicc = expected_aic + 2 * k * (k + 1) / (len(y) - k - 1)
        expected_bic = -2 * loglike + np.log(len(y)) * k
        assert row["status"] == "ok" and row["parameters"] == k
        assert row["log_likelihood"] == pytest.approx(loglike, abs=1e-9)
        assert row["aic"] == pytest.approx(expected_aic, abs=1e-9)
        assert row["aicc"] == pytest.approx(expected_aicc, abs=1e-9)
        assert row["bic"] == pytest.approx(expected_bic, abs=1e-9)
    selected = min(selection["candidates"], key=lambda row: row["criterion"])
    assert result.spec.intercept == selected["constant"]
    assert selection["criterion_value"] == selected["criterion"]
    assert selection["candidate_count"] == 2 and selection["common_response_positions"] == list(
        range(len(y))
    )


def test_auto_arima_order_one_candidates_match_independent_stationary_gaussian_mle():
    from scipy.optimize import minimize

    rng = np.random.default_rng(441)
    y = np.empty(90)
    y[0] = 1.2 + rng.normal() / np.sqrt(1 - 0.67**2)
    for j in range(1, len(y)):
        y[j] = 1.2 + 0.67 * (y[j - 1] - 1.2) + rng.normal()
    result = oe.auto_arima(
        data={"y": y}, y="y", d=0, max_p=1, max_q=0, constant=True, criterion="aicc"
    )
    candidate = next(
        row for row in result.extra["auto_selection"]["candidates"] if row["order"] == [1, 0, 0]
    )

    def negative_ll(z):
        mean, phi, sigma = z[0], np.tanh(z[1]), np.exp(z[2])
        u = y - mean
        ss = (1 - phi**2) * u[0] ** 2 + np.sum((u[1:] - phi * u[:-1]) ** 2)
        return 0.5 * (
            len(y) * np.log(2 * np.pi)
            + 2 * len(y) * np.log(sigma)
            - np.log(1 - phi**2)
            + ss / sigma**2
        )

    solution = minimize(
        negative_ll,
        [y.mean(), np.arctanh(0.5), 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-11, "fatol": 1e-11, "maxiter": 10000},
    )
    assert solution.success and candidate["status"] == "ok"
    assert candidate["log_likelihood"] == pytest.approx(-solution.fun, abs=2e-7)
    assert candidate["parameters"] == 3
    assert result.extra["auto_selection"]["criterion_value"] == min(
        row["criterion"]
        for row in result.extra["auto_selection"]["candidates"]
        if row["status"] == "ok"
    )


def test_auto_arima_missing_edge_sorted_rows_and_persisted_sample_hashes():
    from openecon.econometrics.arima.estimators import prepare
    from openecon.analysis import _frame_hasher, _position_bytes

    y = np.random.default_rng(100).normal(size=36)
    y[[0, -1]] = np.nan
    frame = pd.DataFrame({"y": y, "time": np.arange(len(y))}).iloc[::-1].copy()
    result = oe.auto_arima(
        data=frame, y="y", time="time", d=1, max_p=0, max_q=0, constant=True, missing="drop"
    )
    lineage = prepare(result.spec, frame).frame
    expected = _frame_hasher(lineage.sample)
    expected.update(_position_bytes(lineage.positions))
    assert result.provenance["data_hash"] == _frame_hasher(lineage.original).hexdigest()
    assert result.provenance["sample_hash"] == expected.hexdigest()
    assert result.sample_positions == lineage.positions
    assert [row["row"] for row in result.predictions] == lineage.positions
    assert result.nobs_original == len(frame) and result.dropped_rows == 3


def test_rolling_forecasts_use_only_explicit_future_exogenous_values():
    rng = np.random.default_rng(213)
    frame = pd.DataFrame({"x": rng.normal(size=40)})
    frame["y"] = 1.4 + 0.8 * frame.x + rng.normal(size=40) * 0.3
    spec = ModelSpec(
        estimator="arima",
        outcome="y",
        predictors=["x"],
        covariance="nonrobust",
        options={"order": [0, 0, 0]},
    )
    with pytest.raises(AnalysisError) as caught:
        oe.rolling(spec, data=frame, window=20, step=10, forecast_steps=2)
    assert caught.value.code == "missing_exog"
    future_exog = {origin: {"x": [-2.0, 3.0]} for origin in [19, 29, 39]}
    result = oe.rolling(spec, data=frame, window=20, step=10, forecast_steps=2, exog=future_exog)
    for record in result.attrs["origins"]:
        positions = record["input_positions"]
        design = np.column_stack((np.ones(len(positions)), frame.x.iloc[positions]))
        beta = np.linalg.lstsq(design, frame.y.iloc[positions], rcond=None)[0]
        actual = result["forecasts"].loc[result["forecasts"].origin == record["origin"], "forecast"]
        np.testing.assert_allclose(actual, np.array([[1.0, -2.0], [1.0, 3.0]]) @ beta, atol=1e-9)
        assert record["covariance_matrix"] and record["coefficients"]


def test_rolling_midas_calendar_metadata_sliced_and_bound_per_origin():
    aligned, names = midas_fixture()
    spec = ModelSpec(
        estimator="midas",
        outcome="y",
        predictors=[],
        columns={"lags": names},
        options={"alignment": aligned.attrs["midas_alignment"], "tolerance": 1e-8},
    )
    result = oe.rolling(spec, data=aligned, window=40, step=40)
    assert len(result.attrs["origins"]) == 2
    for record in result.attrs["origins"]:
        assert record["status"] == "ok"
        metadata = record["spec"]["options"]["alignment"]
        assert len(metadata["rows"]) == 40
        selected = aligned.iloc[record["input_positions"]]
        assert (
            metadata["data_hash"]
            == __import__("hashlib")
            .sha256(
                pd.util.hash_pandas_object(
                    selected.loc[:, metadata["binding_columns"]], index=False
                ).values.tobytes()
            )
            .hexdigest()
        )
        assert metadata["rows"][0]["origin"] == selected.quarter.iloc[0].isoformat()
