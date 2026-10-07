"""Saved history targets: independent recursion/design, origin and replay checks."""

import gc
import numpy as np
import pandas as pd
import pytest
import openecon as oe
from openecon.models import ResultBundle
from openecon.analysis_contracts import AnalysisError


def collect(source):
    return pd.concat(list(source.iter_batches(batch_rows=11)))


def ar1(integrated=False):
    rng = np.random.default_rng(501)
    y = np.zeros(150)
    for t in range(1, len(y)):
        y[t] = 0.2 + 0.45 * y[t - 1] + rng.normal()
    if integrated:
        y = y.cumsum()
    return pd.DataFrame({"y": y, "t": np.arange(len(y))})


def test_saved_categorical_reference_is_reused_and_unknown_history_rejected():
    d = ar1()
    d["cat"] = pd.Categorical(["abc"[i % 3] for i in range(len(d))], categories=["a", "b", "c"])
    d["y"] += (d.cat == "b").astype(float) * 0.7
    r = oe.arima(
        data=oe.Dataset.from_frame(d),
        y="y",
        x=["cat"],
        categorical=["cat"],
        time="t",
        order=(1, 0, 0),
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    future = pd.DataFrame({"cat": ["a", "b"]})
    expected = collect(
        oe.dynamic_predict(
            r,
            oe.Dataset.from_frame(d),
            target="forecast_levels",
            origin="end",
            horizon=2,
            future=future,
        )
    )
    reordered = d.copy()
    reordered["cat"] = reordered.cat.cat.reorder_categories(["c", "b", "a"])
    actual = collect(
        oe.dynamic_predict(
            r,
            oe.Dataset.from_frame(reordered),
            target="forecast_levels",
            origin="end",
            horizon=2,
            future=future,
        )
    )
    np.testing.assert_allclose(actual.forecast, expected.forecast, atol=1e-10)
    hostile = d.copy()
    hostile["cat"] = hostile.cat.astype(object)
    hostile.loc[5, "cat"] = "unknown"
    with pytest.raises(AnalysisError, match="categorical level"):
        oe.dynamic_predict(
            r,
            oe.Dataset.from_frame(hostile),
            target="forecast_levels",
            origin="end",
            horizon=2,
            future=future,
        )


@pytest.mark.parametrize("integrated", [False, True])
@pytest.mark.parametrize("rows", [1, 19])
def test_arima_origin_levels_and_innovation_recursion(integrated, rows, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    d = ar1(integrated)
    r = oe.arima(data=oe.Dataset.from_frame(d), y="y", time="t", order=(1, int(integrated), 0))
    r = ResultBundle.model_validate_json(r.model_dump_json())
    shuffled = d.sample(frac=1, random_state=12)
    output = oe.dynamic_predict(
        r,
        oe.Dataset.from_frame(shuffled),
        target="forecast_levels",
        origin=108,
        horizon=5,
        batch_rows=rows,
    )
    actual = collect(output)
    beta = {c.term: c.estimate for c in r.coefficients}
    phi, constant, sigma = beta["ARMA:L1.ar"], beta["Intercept"], beta["/sigma"]
    last = d.y.iloc[108] - d.y.iloc[107] if integrated else d.y.iloc[108]
    level = d.y.iloc[108]
    want, var = [], []
    weights = []
    for h in range(5):
        last = constant + phi * (last - constant)
        level = level + last if integrated else last
        want.append(level)
        weights.append(sum(phi**j for j in range(h + 1)) if integrated else phi**h)
        var.append(sigma**2 * sum(w * w for w in weights))
    np.testing.assert_allclose(actual.forecast, want, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual.std_error, np.sqrt(var), rtol=1e-8)
    assert actual.period.tolist() == list(range(109, 114))
    assert output.metadata["analysis"]["ordered_history_rows"] == 109
    del output
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["ardl", "nardl"])
@pytest.mark.parametrize("rows", [1, 23])
def test_lag_partial_sum_fitted_level_and_full_delta(name, rows):
    from test_streaming_ardl import data

    d = data.__wrapped__().iloc[:130].copy()
    r = getattr(oe, name)(
        data=oe.Dataset.from_frame(d), y="y", x=["x"], time="t", lags=[2, 1], ec=True
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    out = collect(
        oe.dynamic_predict(
            r, oe.Dataset.from_frame(d), target="fitted_levels", origin="end", batch_rows=rows
        )
    )
    paths = {"x": d.x.to_numpy()}
    if name == "nardl":
        dx = np.diff(d.x, prepend=d.x.iloc[0])
        paths = {"x_positive": np.maximum(dx, 0).cumsum(), "x_negative": np.minimum(dx, 0).cumsum()}
    levels = r.extra["levels_coefficients"]
    wanted = []
    for t in range(2, len(d)):
        values = {"Intercept": 1.0, "L.y": d.y.iloc[t - 1], "L2.y": d.y.iloc[t - 2]}
        for variable, path in paths.items():
            values[variable] = path[t]
            values["L." + variable] = path[t - 1]
        wanted.append([values[key] for key in levels])
    x = np.array(wanted)
    np.testing.assert_allclose(
        out.fitted_level, x @ np.array(list(levels.values())), rtol=1e-10, atol=1e-9
    )
    error = np.sqrt(np.einsum("nk,kl,nl->n", x, np.array(r.extra["levels_covariance"]), x))
    np.testing.assert_allclose(out.std_error, error, rtol=1e-10, atol=1e-9)
    assert out.physical_row.tolist() == list(range(2, len(d)))


@pytest.mark.parametrize("name", ["ahreg", "xtdpd"])
def test_panel_first_difference_covariance_and_physical_rows(name):
    from test_econ_streaming_dpanel import fixture, spec, fit_streaming
    from test_econ_streaming_dynamic_gmm import spec as gspec, fit_streaming as gfit

    d = fixture(18, 7)
    r = (
        fit_streaming(spec(), oe.Dataset.from_frame(d), batch_rows=8)
        if name == "ahreg"
        else gfit(gspec(constant=False, collapse=True), oe.Dataset.from_frame(d), batch_rows=8)
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    out = collect(
        oe.dynamic_predict(
            r, oe.Dataset.from_frame(d), target="fitted_transformed", origin="end", batch_rows=5
        )
    )
    beta = np.array([c.estimate for c in r.coefficients])
    for _, row in out.iterrows():
        p = int(row.physical_row)
        values = {
            "L.y": d.y.iloc[p - 1] - d.y.iloc[p - 2],
            "L1.y": d.y.iloc[p - 1] - d.y.iloc[p - 2],
            "x": d.x.iloc[p] - d.x.iloc[p - 1],
            "z": d.z.iloc[p] - d.z.iloc[p - 1],
        }
        x = np.array([values[c.term] for c in r.coefficients])
        assert row.fitted_transformed == pytest.approx(x @ beta, rel=1e-9, abs=1e-10)
        assert row.std_error == pytest.approx(
            np.sqrt(x @ np.array(r.covariance_matrix) @ x), rel=1e-9
        )


@pytest.mark.parametrize("name", ["var", "vec"])
def test_var_representation_level_forecast_and_native_uncertainty(name):
    from test_streaming_var import data

    d = data.__wrapped__().iloc[:150].copy()
    r = getattr(oe, name)(
        data=oe.Dataset.from_frame(d),
        y=["a", "b"],
        time="t",
        lags=2,
        **({"rank": 1} if name == "vec" else {}),
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    out = collect(
        oe.dynamic_predict(
            r,
            oe.Dataset.from_frame(d),
            target="forecast_levels",
            origin=120,
            horizon=3,
            batch_rows=7,
        )
    )
    expected = oe.forecast(r, 3, data=d.loc[d.t <= 120])
    np.testing.assert_allclose(
        out.select_dtypes("number"), expected.select_dtypes("number"), rtol=1e-9, atol=1e-10
    )
    if name == "var":
        layout = r.extra["layout"]
        coef = np.array([c.estimate for c in r.coefficients]).reshape(2, layout["n_regressors"])
        last = d.loc[d.t <= 120, ["a", "b"]].tail(2).to_numpy()[::-1]
        path = []
        for h in range(3):
            point = coef @ np.r_[last.T.ravel(), 1.0]
            path.append(point)
            last = np.vstack((point, last[:-1]))
        for variable, column in [("a", 0), ("b", 1)]:
            np.testing.assert_allclose(
                out.loc[out.variable == variable, "forecast"], np.array(path)[:, column], rtol=1e-10
            )


def test_arch_conditional_variance_native_recursion():
    from test_econ_arch_oracle import simulate, oracle_of

    d = simulate(280, 11, model="garch")
    r = oe.arch(data=oe.Dataset.from_frame(d), y="y", x=["x"], time="t")
    out = collect(
        oe.dynamic_predict(
            r,
            oe.Dataset.from_frame(d),
            target="conditional_variance",
            origin="end",
            horizon=3,
            future={"x": [0.0, 0.0, 0.0]},
            batch_rows=13,
        )
    )
    theta, oracle = oracle_of(r, d)
    e, h, _ = oracle.path(theta)
    b = {c.term: c.estimate for c in r.coefficients}
    wanted = []
    value = b["ARCH:Intercept"] + b["ARCH:L1.arch"] * e[-1] ** 2 + b["ARCH:L1.garch"] * h[-1]
    for step in range(3):
        wanted.append(value)
        value = b["ARCH:Intercept"] + (b["ARCH:L1.arch"] + b["ARCH:L1.garch"]) * value
    np.testing.assert_allclose(out.variance_forecast, wanted, rtol=1e-7, atol=1e-8)


def test_ucm_filtered_history_forecast_components_and_markov_probabilities():
    from test_econ_streaming_ucm import data, spec, fit_streaming_ucm

    d = data(70)
    r = fit_streaming_ucm(spec(), oe.Dataset.from_frame(d))
    restored = ResultBundle.model_validate_json(r.model_dump_json())
    out = collect(
        oe.dynamic_predict(
            restored,
            oe.Dataset.from_frame(d),
            target="forecast_levels",
            origin="end",
            horizon=2,
            future={"x": [0.0, 0.0]},
        )
    )
    expected = oe.forecast(r, 2, exog={"x": [0.0, 0.0]})
    np.testing.assert_allclose(
        out.select_dtypes("number"), expected.select_dtypes("number"), rtol=1e-8, atol=1e-8
    )
    comp = collect(
        oe.dynamic_predict(
            restored,
            oe.Dataset.from_frame(d),
            target="components_smoothed",
            origin="end",
            batch_rows=9,
        )
    )
    native = collect(oe.ucm_components(restored, oe.Dataset.from_frame(d)))
    pd.testing.assert_frame_equal(
        comp, native, check_index_type=False, check_exact=False, rtol=1e-8
    )
    from test_econ_streaming_mswitch import data as md, specification, fit_streaming_mswitch

    d = md(80)
    r = fit_streaming_mswitch(specification(), oe.Dataset.from_frame(d))
    r = ResultBundle.model_validate_json(r.model_dump_json())
    native = collect(oe.mswitch_probabilities(r, oe.Dataset.from_frame(d)))
    for target in ["filtered", "predicted", "smoothed"]:
        out = collect(
            oe.dynamic_predict(
                r,
                oe.Dataset.from_frame(d),
                target="probabilities_" + target,
                origin="end",
                batch_rows=7,
            )
        )
        names = [name for name in native.columns if name.startswith(target + "_state")]
        np.testing.assert_allclose(
            out[names].to_numpy(dtype=float),
            native[names].to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-9,
        )
        np.testing.assert_allclose(out[names].sum(axis=1), 1.0, atol=1e-10)


def test_origin_missing_history_state_target_and_source_changes():
    d = ar1()
    r = oe.arima(data=oe.Dataset.from_frame(d), y="y", time="t", order=(1, 0, 0))
    for args in [
        {"history": None, "origin": "end"},
        {"history": oe.Dataset.from_frame(d), "origin": 999},
    ]:
        with pytest.raises(AnalysisError):
            oe.dynamic_predict(r, target="forecast_levels", horizon=2, **args)
    with pytest.raises(AnalysisError):
        oe.dynamic_predict(r, oe.Dataset.from_frame(d), target="response", origin="end")
    with pytest.raises(AnalysisError):
        oe.dynamic_predict(
            r,
            oe.Dataset.from_frame(d),
            target="forecast_levels",
            origin="end",
            horizon=2,
            uncertainty="parameter",
        )
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        current = d.copy()
        if calls > 1:
            current.loc[20, "y"] += 1
        yield current

    with pytest.raises(AnalysisError, match="changed"):
        oe.dynamic_predict(
            r,
            oe.Dataset.from_batches(factory, d.columns, row_count=len(d)),
            target="forecast_levels",
            origin="end",
            horizon=2,
        )
