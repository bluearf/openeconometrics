"""Independent native U-MIDAS, additive ETS selection and saved RTS acceptance."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec, ResultBundle


def aligned_fixture():
    rng = np.random.default_rng(7021)
    high = pd.DataFrame(
        {
            "month": pd.date_range("2000-01-01", periods=290, freq="MS"),
            "value": rng.normal(size=290),
        }
    )
    high["released"] = high["month"] + pd.Timedelta(days=12)
    low = pd.DataFrame(
        {
            "quarter": pd.date_range("2001-03-31", periods=84, freq="QE"),
            "control": rng.normal(size=84),
        }
    )
    low["origin"] = low["quarter"] - pd.Timedelta(days=20)
    aligned = oe.midas_align(
        low=low,
        high=high,
        low_time="quarter",
        high_time="month",
        value="value",
        lags=5,
        frequency="MS",
        origin="origin",
        release_time="released",
    )
    names = aligned.attrs["midas_alignment"]["lags"]
    design = np.column_stack([np.ones(len(aligned)), aligned["control"], aligned[names]])
    aligned["y"] = design @ np.array([0.7, -0.2, 0.8, -0.3, 0.2, 0.1, -0.1]) + rng.normal(
        0, 0.25, len(aligned)
    )
    return aligned, high, low, names, design


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def test_umidas_full_ols_law_inference_and_saved_prediction(covariance, tmp_path, monkeypatch):
    data, _, _, names, design = aligned_fixture()
    y = data.y.to_numpy()
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    bread = np.linalg.inv(design.T @ design)
    n, k = design.shape
    leverage = np.sum(design * (design @ bread), axis=1)
    if covariance == "nonrobust":
        expected = residual @ residual / (n - k) * bread
    else:
        squared = residual**2
        if covariance == "HC2":
            squared /= 1 - leverage
        if covariance == "HC3":
            squared /= (1 - leverage) ** 2
        expected = bread @ (design.T @ (design * squared[:, None])) @ bread
        if covariance == "HC1":
            expected *= n / (n - k)
    result = oe.umidas(data=data, y="y", x=["control"], covariance=covariance)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], beta, rtol=0, atol=2e-13)
    np.testing.assert_allclose(result.covariance_matrix, expected, rtol=1e-11, atol=1e-14)
    critical = student_t.ppf(0.975, n - k)
    se = np.sqrt(np.diag(expected))
    np.testing.assert_allclose(
        [c.ci_low for c in result.coefficients], beta - critical * se, atol=1e-12
    )
    np.testing.assert_allclose(
        [c.p_value for c in result.coefficients],
        2 * student_t.sf(np.abs(beta / se), n - k),
        atol=1e-11,
    )
    assert result.spec.estimator == "umidas"
    assert result.extra["umidas_state"]["lag_columns"] == names
    path = tmp_path / "umidas-model.json"
    path.write_text(result.model_dump_json())
    restored = ResultBundle.model_validate_json(path.read_text())
    monkeypatch.setattr(
        "openecon.linear_ols.fit_ols",
        lambda *args, **kwargs: pytest.fail("saved prediction refitted"),
    )
    prediction = oe.umidas_predict(restored, data=data)
    variance = np.sum((design @ expected) * design, axis=1)
    np.testing.assert_allclose(prediction["mean"], design @ beta, atol=2e-13)
    np.testing.assert_allclose(prediction["std_error"] ** 2, variance, atol=1e-14)
    np.testing.assert_allclose(
        prediction["ci_high"], design @ beta + critical * np.sqrt(variance), atol=1e-11
    )
    assert prediction.attrs["df"] == n - k and prediction.attrs["refitted"] is False
    if covariance == "nonrobust":
        outcome = oe.umidas_predict(restored, data=data, kind="outcome")
        np.testing.assert_allclose(
            outcome.std_error**2, variance + residual @ residual / (n - k), atol=1e-13
        )
    else:
        with pytest.raises(AnalysisError, match="Gaussian"):
            oe.umidas_predict(restored, data=data, kind="outcome")


def test_umidas_release_calendar_rolling_selection_and_later_data_isolation():
    data, high, low, names, _ = aligned_fixture()
    for i, row in enumerate(data.attrs["midas_alignment"]["rows"]):
        asof = pd.Timestamp(row["origin"])
        available = high.loc[(high.month <= asof) & (high.released <= asof)]
        newest = available.month.max()
        expected = [newest - j * pd.offsets.MonthBegin(1) for j in range(5)]
        assert list(pd.to_datetime(row["observation_times"])) == expected
        np.testing.assert_array_equal(
            data[names].iloc[i], high.set_index("month").loc[expected, "value"]
        )
    fit = oe.umidas(data=data, y="y", x=["control"])
    windows = oe.rolling(fit.spec, data=data, window=36, step=24)
    changed_high = high.copy()
    cutoff = pd.Timestamp(data.attrs["midas_alignment"]["rows"][35]["origin"])
    changed_high.loc[changed_high.released > cutoff, "value"] += 1e6
    changed = oe.midas_align(
        low=low,
        high=changed_high,
        low_time="quarter",
        high_time="month",
        value="value",
        lags=5,
        frequency="MS",
        origin="origin",
        release_time="released",
    )
    changed["y"] = data.y.copy()
    changed.loc[36:, "y"] += 1e5
    again = oe.rolling(
        fit.spec.model_copy(update={"options": {"alignment": changed.attrs["midas_alignment"]}}),
        data=changed,
        window=36,
        step=24,
    )
    first = windows.attrs["origins"][0]
    assert first["input_positions"] == list(range(36))
    for key in ["coefficients", "covariance_matrix", "inference", "metrics"]:
        assert first[key] == again.attrs["origins"][0][key]
    first_alignment = first["spec"]["options"]["alignment"]
    changed_alignment = again.attrs["origins"][0]["spec"]["options"]["alignment"]
    assert first_alignment["window_input_positions"] == list(range(36))
    # The retained training data and dates are identical. The transparent parent
    # provenance hash records the intentionally changed future source history.
    assert first_alignment["data_hash"] == changed_alignment["data_hash"]
    assert first_alignment["rows"] == changed_alignment["rows"]
    assert first_alignment["parent_data_hash"] != changed_alignment["parent_data_hash"]
    assert all(
        pd.Timestamp(release) <= pd.Timestamp(row["origin"])
        for row in first["spec"]["options"]["alignment"]["rows"]
        for release in row["release_times"]
    )


def test_umidas_original_missing_positions_rank_and_reserved_term():
    data, _, _, _, _ = aligned_fixture()
    data.loc[[0, 3], "y"] = np.nan
    result = oe.umidas(data=data, y="y", x=["control"], missing="drop")
    assert result.nobs_original == 84 and result.dropped_rows == 2
    assert result.sample_positions == [i for i in range(84) if i not in (0, 3)]
    assert result.extra["umidas_state"]["sample_alignment_positions"] == result.sample_positions
    duplicate = data.copy()
    duplicate["duplicate"] = duplicate["midas_L0"]
    with pytest.raises(AnalysisError, match="identified"):
        oe.umidas(data=duplicate, y="y", x=["duplicate"], missing="drop")
    reserved = data.copy()
    reserved["Intercept"] = reserved["control"]
    with pytest.raises(AnalysisError, match="Intercept"):
        oe.umidas(data=reserved, y="y", x=["Intercept"], missing="drop")
    no_constant = oe.umidas(data=reserved, y="y", x=["Intercept"], intercept=False, missing="drop")
    prediction = oe.umidas_predict(no_constant, data=reserved)
    X = reserved[["Intercept", *no_constant.spec.columns["lags"]]].to_numpy()
    np.testing.assert_allclose(
        prediction["mean"], X @ np.array([c.estimate for c in no_constant.coefficients]), atol=1e-13
    )


@pytest.mark.parametrize(
    "mutation", ["lag_values", "future_release", "broken_calendar", "saved_alignment"]
)
def test_umidas_alignment_tampering_is_explicitly_refused(mutation):
    data, _, _, _, _ = aligned_fixture()
    result = oe.umidas(data=data, y="y", x=["control"])
    evaluation = data.copy()
    evaluation.attrs = deepcopy(data.attrs)
    if mutation == "lag_values":
        evaluation.loc[0, "midas_L0"] += 1
    elif mutation == "future_release":
        evaluation.attrs["midas_alignment"]["rows"][0]["release_times"][0] = "2099-01-01"
    elif mutation == "broken_calendar":
        evaluation.attrs["midas_alignment"]["rows"][0]["observation_times"][1] = "2001-01-15"
    else:
        result.extra["umidas_state"]["alignment"] = None
    with pytest.raises(AnalysisError):
        oe.umidas_predict(result, data=evaluation)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "rows_type",
        "row_count",
        "duplicate_original",
        "boolean_original",
        "sample_positions",
        "lag_calendar",
        "missing_target",
        "future_release",
        "binding_columns",
        "malformed_hash",
        "retained_dropped_overlap",
    ],
)
def test_umidas_both_saved_alignment_copies_cannot_bypass_history_validation(mutation):
    data, _, _, _, _ = aligned_fixture()
    result = oe.umidas(data=data, y="y", x=["control"])
    record = deepcopy(result.extra["umidas_state"]["alignment"])
    if mutation == "schema":
        record["schema"] = True
    elif mutation == "rows_type":
        record["rows"] = None
    elif mutation == "row_count":
        record["rows"].pop()
    elif mutation == "duplicate_original":
        record["original_positions"][1] = record["original_positions"][0]
    elif mutation == "boolean_original":
        record["original_positions"][0] = False
    elif mutation == "sample_positions":
        result.extra["umidas_state"]["sample_alignment_positions"] = [0] * result.nobs
    elif mutation == "lag_calendar":
        record["rows"][0]["observation_times"][1] = record["rows"][0]["observation_times"][0]
    elif mutation == "missing_target":
        record["rows"][0].pop("target")
    elif mutation == "future_release":
        record["rows"][0]["release_times"][0] = "2099-01-01"
    elif mutation == "binding_columns":
        record["binding_columns"].pop(0)
    elif mutation == "malformed_hash":
        record["data_hash"] = "x" * 64
    else:
        record["dropped_positions"] = [record["original_positions"][0]]
    # Changing both copies defeats a simple equality-only guard. The actual
    # dated rows, hash shape and physical-position laws must still be checked.
    result.spec.options["alignment"] = record
    result.extra["umidas_state"]["alignment"] = deepcopy(record)
    with pytest.raises(AnalysisError) as error:
        oe.umidas_predict(result, data=data)
    assert error.value.code == "invalid_state"


def scalar_ets(y, model, initial, fixed, period=4):
    trend = model not in {"ANN", "ANA"}
    seasonal = model.endswith("A")
    level = float(initial[0])
    b = float(initial[1]) if trend else 0.0
    season = list(initial[-period:]) if seasonal else [0.0] * period
    phi = fixed.get("phi", 1.0)
    errors = []
    for date, value in enumerate(y):
        error = value - (level + phi * b + season[date % period])
        errors.append(error)
        level += phi * b + fixed["alpha"] * error
        if trend:
            b = phi * b + fixed["beta"] * error
        if seasonal:
            season[date % period] += fixed["gamma"] * error
    return np.asarray(errors)


def ets_options(models):
    output = {}
    for model in models:
        fixed = {"alpha": 0.32}
        if model not in {"ANN", "ANA"}:
            fixed["beta"] = 0.08
        if model.endswith("A"):
            fixed["gamma"] = 0.12
        if "d" in model:
            fixed["phi"] = 0.91
        output[model] = {"fixed": fixed}
    return output


def ets_linear_oracle(y, model, fixed, period=4):
    trend = model not in {"ANN", "ANA"}
    seasonal = model.endswith("A")
    leading = 1 + int(trend)
    free = leading + (period - 1 if seasonal else 0)

    def initial(coordinates):
        return (
            np.r_[coordinates[:leading], coordinates[leading:], -np.sum(coordinates[leading:])]
            if seasonal
            else coordinates
        )

    zero = scalar_ets(y, model, initial(np.zeros(free)), fixed, period)
    design = np.column_stack(
        [zero - scalar_ets(y, model, initial(np.eye(free)[j]), fixed, period) for j in range(free)]
    )
    coordinates = np.linalg.lstsq(design, zero, rcond=None)[0]
    residual = zero - design @ coordinates
    sigma = np.sqrt(residual @ residual / len(y))
    ll = (
        -len(y) / 2 * np.log(2 * np.pi)
        - len(y) * np.log(sigma)
        - residual @ residual / (2 * sigma * sigma)
    )
    covariance = np.zeros((free + 1, free + 1))
    covariance[:-1, :-1] = sigma * sigma * np.linalg.inv(design.T @ design)
    covariance[-1, -1] = sigma * sigma / (2 * len(y))
    return np.r_[coordinates, sigma], covariance, ll


@pytest.mark.parametrize("criterion", ["aic", "aicc", "bic"])
def test_auto_ets_complete_initial_sigma_ic_and_covariance_against_closed_form(criterion):
    models = ["ANN", "AAN", "AdN", "ANA", "AAA", "AdA"]
    y = (
        2
        + 0.012 * np.arange(100)
        + np.tile([-0.3, 0.1, 0.35, -0.15], 25)
        + np.random.default_rng(995).normal(0, 0.4, 100)
    )
    settings = ets_options(models)
    result = oe.auto_ets(
        data={"y": y},
        y="y",
        models=models,
        period=4,
        criterion=criterion,
        candidate_options=settings,
    )
    history = result.extra["auto_selection"]["candidates"]
    assert all(row["status"] == "ok" for row in history)
    scores = {}
    for row in history:
        beta, cov, ll = ets_linear_oracle(y, row["model"], settings[row["model"]]["fixed"])
        k = len(beta)
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(len(y))
        aicc = aic + 2 * k * (k + 1) / (len(y) - k - 1)
        assert (
            row["parameters"] == k
            and row["parameter_order"][-1] == "sigma"
            and row["initial_estimated"]
        )
        np.testing.assert_allclose([c["estimate"] for c in row["coefficients"]], beta, atol=2e-5)
        np.testing.assert_allclose(row["covariance_matrix"], cov, rtol=5e-5, atol=1e-6)
        for name, value in {"log_likelihood": ll, "aic": aic, "aicc": aicc, "bic": bic}.items():
            assert row[name] == pytest.approx(value, abs=3e-7)
        scores[row["model"]] = {"aic": aic, "aicc": aicc, "bic": bic}[criterion]
    assert result.extra["model"] == min(scores, key=scores.get)
    assert result.inference["model_selection_uncertainty"] is False


def test_auto_ets_candidate_failures_and_typed_resource_admission():
    y = np.random.default_rng(33).normal(size=40)
    result = oe.auto_ets(
        data={"y": y},
        y="y",
        models=["ANN", "AAN"],
        candidate_options={"ANN": {"fixed": []}, "AAN": {"fixed": {"alpha": 0.3, "beta": 0.1}}},
    )
    failed, valid = result.extra["auto_selection"]["candidates"]
    assert failed["status"] == "failed" and failed["error_code"] == "invalid_option"
    assert valid["status"] == "ok"
    with pytest.raises(AnalysisError) as info:
        oe.auto_ets(
            data={"y": y}, y="y", models=["ANN"], candidate_options={"ANN": {"initial": [None]}}
        )
    assert info.value.code == "no_valid_model" and len(info.value.candidates) == 1
    for options in (
        {"period": True},
        {"period": None},
        {"models": "ANN"},
        {"models": ["MNN"]},
        {"criterion": []},
        {"max_candidates": 1},
        {"max_work": 1},
    ):
        with pytest.raises(AnalysisError):
            oe.auto_ets(data={"y": y}, y="y", **options)


def test_auto_ets_rolling_refits_every_search_at_each_training_origin_and_full_restore(tmp_path):
    y = 3 + 0.01 * np.arange(120) + np.random.default_rng(778).normal(size=120) * 0.3
    data = pd.DataFrame({"y": y, "t": np.arange(120)})
    models = ["ANN", "AAN"]
    options = {"models": models, "candidate_options": ets_options(models), "criterion": "bic"}
    spec = ModelSpec(
        estimator="ets",
        outcome="y",
        intercept=False,
        time="t",
        options={"model": "ANN", "period": 4},
    )
    windows = oe.rolling(
        spec, data=data, window=60, step=30, forecast_steps=2, evaluate=True, selection=options
    )
    for origin in windows.attrs["origins"]:
        assert origin["auto_selection"]["candidate_count"] == 2
        assert origin["auto_selection"]["period"] == 4
        assert origin["auto_selection"]["common_response_positions"] == list(range(60))
        assert origin["nobs"] == 60
    changed = data.copy()
    changed.loc[60:, "y"] += 1e4
    again = oe.rolling(
        spec, data=changed, window=60, step=30, forecast_steps=2, evaluate=True, selection=options
    )
    for key in [
        "coefficients",
        "covariance_matrix",
        "inference",
        "metrics",
        "spec",
        "auto_selection",
    ]:
        assert windows.attrs["origins"][0][key] == again.attrs["origins"][0][key]
    first = oe.auto_ets(data=data.iloc[:60], y="y", time="t", period=4, **options)
    path = tmp_path / "selected-model.json"
    path.write_text(first.model_dump_json())
    restored = ResultBundle.model_validate_json(path.read_text())
    assert restored.extra["auto_selection"] == first.extra["auto_selection"]
    pd.testing.assert_frame_equal(oe.forecast(restored, 5), oe.forecast(first, 5))
    assert "excluded" in restored.extra["auto_selection"]["selection_uncertainty"]


def joint_state_oracle(y, system):
    """Dense complete Gaussian conditioning, independent of Kalman/RTS recursions."""
    n, p = y.shape
    m = len(system["a0"])

    def at(key, date):
        return np.asarray(system.get("schedules", {}).get(key, [system[key]] * n)[date])

    means = np.zeros(n * m)
    cov = np.zeros((n * m, n * m))
    means[:m] = system["a0"]
    cov[:m, :m] = system["P0"]
    for date in range(1, n):
        current = slice(date * m, (date + 1) * m)
        previous = slice((date - 1) * m, date * m)
        T = at("T", date - 1)
        means[current] = T @ means[previous] + at("c", date - 1)
        cov[current, : date * m] = T @ cov[previous, : date * m]
        cov[: date * m, current] = cov[current, : date * m].T
        cov[current, current] = T @ cov[previous, previous] @ T.T + at("Q", date - 1)
    Z = np.zeros((n * p, n * m))
    H = np.zeros((n * p, n * p))
    d = np.zeros(n * p)
    for date in range(n):
        Z[date * p : (date + 1) * p, date * m : (date + 1) * m] = at("Z", date)
        H[date * p : (date + 1) * p, date * p : (date + 1) * p] = at("H", date)
        d[date * p : (date + 1) * p] = at("d", date)
    observed = np.isfinite(y.ravel())
    Z = Z[observed]
    H = H[np.ix_(observed, observed)]
    d = d[observed]
    C = Z @ cov @ Z.T + H
    residual = y.ravel()[observed] - (Z @ means + d)
    posterior_mean = means + cov @ Z.T @ np.linalg.solve(C, residual)
    posterior_cov = cov - cov @ Z.T @ np.linalg.solve(C, Z @ cov)
    ll = -0.5 * (
        len(residual) * np.log(2 * np.pi)
        + np.linalg.slogdet(C)[1]
        + residual @ np.linalg.solve(C, residual)
    )
    return posterior_mean.reshape(n, m), posterior_cov, ll


@pytest.mark.parametrize("scheduled", [False, True])
def test_canonical_masked_scheduled_rts_full_covariance_against_joint_gaussian(
    scheduled, tmp_path, monkeypatch
):
    n = 9
    y = np.random.default_rng(456).normal(size=(n, 2))
    y[2, 0] = np.nan
    y[5] = np.nan
    system = {
        "T": [[0.6, 0.1], [0.0, 0.4]],
        "Z": [[1.0, 0.2], [-0.1, 1.0]],
        "Q": [[0.3, 0.04], [0.04, 0.2]],
        "H": [[0.25, 0.05], [0.05, 0.3]],
        "c": [0.02, -0.03],
        "d": [0.1, 0.2],
        "a0": [0.3, -0.2],
        "P0": [[0.7, 0.1], [0.1, 0.6]],
        "initialization": "known",
    }
    if scheduled:
        system["schedules"] = {
            "T": [[[0.5 + 0.02 * t, 0.1], [0.0, 0.4]] for t in range(n)],
            "d": [[0.1 + 0.03 * t, 0.2] for t in range(n)],
        }
    result = oe.sspace(
        data={"a": y[:, 0], "b": y[:, 1], "t": np.arange(n)},
        y="a",
        responses=["b"],
        time="t",
        system=system,
        missing="mask",
    )
    expected, cov, ll = joint_state_oracle(y, system)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, abs=1e-12)
    assert result.nobs == n and result.extra["observed_count"][5] == 0
    source = tmp_path / "sspace.json"
    source.write_text(result.model_dump_json())
    frozen = json.loads(source.read_text())
    digest = frozen["extra"].pop("state_sha256")
    assert (
        hashlib.sha256(
            json.dumps(frozen, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()
        == digest
    )
    restored = ResultBundle.model_validate_json(source.read_text())
    monkeypatch.setattr(
        "openecon.econometrics.tsworkflows.sspace.fit_sspace",
        lambda *a, **kw: pytest.fail("saved smoother refitted"),
    )
    smoothed = oe.sspace_smooth(restored)
    np.testing.assert_allclose(smoothed[["state_1", "state_2"]], expected, atol=2e-13)
    within = [cov[t * 2 : (t + 1) * 2, t * 2 : (t + 1) * 2] for t in range(n)]
    np.testing.assert_allclose(smoothed.attrs["covariance"], within, atol=2e-13)
    lag = oe.sspace_autocov(restored)
    cross = [cov[(t + 1) * 2 : (t + 2) * 2, t * 2 : (t + 1) * 2] for t in range(n - 1)]
    np.testing.assert_allclose(lag.attrs["covariance"], cross, atol=2e-13)
    assert "parameter uncertainty excluded" in smoothed.attrs["uncertainty"]
    corrupt = restored.model_copy(deep=True)
    corrupt.extra["observations"][0][0] += 1
    with pytest.raises(AnalysisError):
        oe.sspace_smooth(corrupt)
