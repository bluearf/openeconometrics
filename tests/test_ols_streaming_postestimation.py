import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.linear_ols import streaming_postestimation as post


@pytest.fixture
def frame():
    rng = np.random.default_rng(8393)
    data = pd.DataFrame({"x": rng.normal(size=180), "z": rng.normal(size=180),
                         "g": np.tile(["a", "b", "c"], 60), "firm": np.repeat(range(18), 10)})
    data["y"] = 2 + data.x + .4 * data.x**2 + rng.normal(size=180) * (1 + data.x**2)
    data["w"] = rng.integers(1, 5, len(data))
    return data


def source(frame, size=13):
    def batches():
        for start in range(0, len(frame), size):
            yield frame.iloc[start:start + size]
    return oe.Dataset.from_batches(batches, list(frame), row_count=len(frame))


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster"])
def test_public_streaming_result_and_iterator_match_dense(frame, covariance):
    opts = dict(formula="y ~ x * C(g) + z", covariance=covariance,
                cluster="firm" if covariance == "cluster" else None)
    dense = oe.ols(data=frame, **opts)
    stream = oe.ols(data=source(frame), **opts)
    assert_allclose([c.estimate for c in stream.coefficients], [c.estimate for c in dense.coefficients], rtol=1e-10, atol=1e-12)
    assert_allclose(stream.covariance_matrix, dense.covariance_matrix, rtol=1e-10, atol=1e-12)
    assert stream.provenance["streaming"]["full_sample_positions_retained"] is False
    batches = list(stream.iter_predict(batch_rows=17))
    assert max(len(part) for part in batches) <= 17
    assert_allclose(pd.concat(batches).to_numpy(), dense.predict().to_numpy(), rtol=1e-10, atol=1e-12)
    with pytest.raises(oe.AnalysisError, match="iter_predict"):
        stream.predict()
    assert_allclose(stream.predict(data=frame.iloc[:3]), dense.predict(data=frame.iloc[:3]), rtol=1e-10)
    assert "\\toprule" in stream.to_latex()
    assert oe.ResultBundle.model_validate(stream.model_dump()).nobs == 180


@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "pweight", "iweight"])
def test_vif_uses_whole_streamed_sample_not_chart_sample(frame, weight):
    kwargs = dict(formula="y ~ x * C(g) + z", covariance="HC3",
                  weights="w" if weight else None, weight_type=weight)
    dense = oe.ols(data=frame, **kwargs)
    stream = oe.ols(data=source(frame), **kwargs)
    assert_allclose(stream.vif().vif, dense.vif().vif, rtol=1e-10)


@pytest.mark.parametrize("method", ["ame", "mem"])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight"])
def test_margins_averages_contrasts_before_variance(frame, method, weight):
    kwargs = dict(formula="y ~ x * C(g) + I(z**2)", covariance="HC3",
                  weights="w" if weight else None, weight_type=weight)
    dense = oe.ols(data=frame, **kwargs)
    stream = oe.ols(data=source(frame, size=7), **kwargs)
    expected = dense.margins(method=method, at={"x": [-.5, 1.]})
    actual = stream.margins(method=method, at={"x": [-.5, 1.]})
    assert actual.variable.tolist() == expected.variable.tolist()
    assert_allclose(actual[["estimate", "std_error", "ci_low", "ci_high"]],
                    expected[["estimate", "std_error", "ci_low", "ci_high"]], rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("method", ["normal", "iid", "fstat"])
@pytest.mark.parametrize("rhs", [False, True])
def test_breusch_pagan_exact_auxiliary_replays(frame, method, rhs):
    dense = oe.ols(data=frame, formula="y ~ x + z")
    stream = oe.ols(data=source(frame), formula="y ~ x + z")
    expected, actual = dense.hettest(method=method, rhs=rhs), stream.hettest(method=method, rhs=rhs)
    assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-10)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-9)


def test_white_auxiliary_design_is_chunked_and_rank_aware(frame):
    dense = oe.ols(data=frame, formula="y ~ x + z + C(g)")
    stream = oe.ols(data=source(frame), formula="y ~ x + z + C(g)")
    assert stream.white_test()["statistic"] == pytest.approx(dense.white_test()["statistic"], rel=1e-9)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
def test_reset_reuses_fitted_covariance_in_streamed_augmented_model(frame, covariance):
    kwargs = dict(formula="y ~ x + z", covariance=covariance,
                  cluster="firm" if covariance == "cluster" else None)
    dense = oe.ols(data=frame, **kwargs)
    stream = oe.ols(data=source(frame), **kwargs)
    expected, actual = dense.reset_test(), stream.reset_test()
    assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-8)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-8)


def test_prediction_replay_rejects_mutated_values(frame):
    stream = oe.ols(data=source(frame), formula="y ~ x + z")
    frame.iloc[0, frame.columns.get_loc("x")] += 2
    with pytest.raises(oe.AnalysisError, match="changed"):
        list(stream.iter_predict(batch_rows=20))


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("powers", [2, (2, 3, 4)])
def test_reset_preserves_intercept_and_accepts_scalar_powers(frame, covariance, intercept, powers):
    kwargs = dict(y="y", x=["x", "z"], intercept=intercept, covariance=covariance,
                  cluster="firm" if covariance == "cluster" else None)
    dense = oe.ols(data=frame, **kwargs)
    stream = oe.ols(data=source(frame, size=7), **kwargs)
    expected = dense.reset_test(powers)
    actual = post.reset_test(stream, powers)
    assert actual["df_denom"] == expected["df_denom"]
    assert actual["df_num"] == expected["df_num"]
    assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-8)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-8)


@pytest.mark.parametrize("powers", [True, 1, [], [2, 2], None, [2.]])
def test_stream_reset_invalid_powers_are_actionable(frame, powers):
    stream = oe.ols(data=source(frame), y="y", x=["x"])
    with pytest.raises(oe.AnalysisError, match="powers"):
        post.reset_test(stream, powers)


@pytest.mark.parametrize("at", [3, "x", [("x", 1)], {"unfitted": 1}, {"x": []}])
def test_stream_margins_validates_mapping_before_replay(frame, at):
    stream = oe.ols(data=source(frame), y="y", x=["x"])
    passes = stream._state["replay"].passes
    with pytest.raises(oe.AnalysisError) as actual:
        post.margins(stream, at=at)
    assert actual.value.code == "invalid_margins"
    assert stream._state["replay"].passes == passes


def test_stream_mem_grid_limit_is_validated_without_scanning_full_source(frame):
    stream = oe.ols(data=source(frame), y="y", x=["x"])
    passes = stream._state["replay"].passes
    with pytest.raises(oe.AnalysisError) as failure:
        post.margins(stream, method="mem", at={"x": list(range(1001))})
    assert failure.value.code == "margins_grid_limit"
    assert stream._state["replay"].passes == passes


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("lags", [1, 4, 17])
@pytest.mark.parametrize("covariance", ["nonrobust", "HC3"])
def test_bg_keeps_history_across_unequal_batches_and_missing_rows(frame, intercept, lags, covariance):
    data = frame.copy()
    data["t"] = np.arange(len(data)) * 3  # Retained physical order, not interpolated time.
    data.loc[[2, 13, 57], "y"] = np.nan
    data.loc[29, "x"] = np.nan
    data.index = np.repeat(np.arange(len(data) // 2), 2)
    kwargs = dict(y="y", x=["x", "z"], time="t", intercept=intercept, covariance=covariance)
    dense = oe.ols(data=data, **kwargs)
    stream = oe.ols(data=source(data, size=7), **kwargs)
    expected = dense.breusch_godfrey(lags)
    actual = post.breusch_godfrey(stream, lags)
    for key in ("statistic", "p_value", "f_statistic", "f_p_value"):
        assert actual[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-12)
    assert actual["df"] == expected["df"]
    assert actual["df_denom"] == expected["df_denom"]
    assert actual["ordering"] == expected["ordering"]


@pytest.mark.parametrize("weight", [None, "fweight", "aweight", "pweight", "iweight"])
def test_distribution_diagnostics_use_all_weighted_residuals(frame, weight):
    from scipy.stats import kurtosis, skew
    from statsmodels.stats.stattools import durbin_watson
    data = pd.concat([frame] * 4, ignore_index=True)
    data.loc[[2, 191, 409, 711], "y"] = np.nan
    data.loc[401, "w"] = 0
    opts = dict(y="y", x=["x", "z"], covariance="HC3",
                weights="w" if weight else None, weight_type=weight)
    dense = oe.ols(data=data, **opts)
    stream = oe.ols(data=source(data, size=19), **opts)
    jb, dw = post._residual_distribution(stream)
    if weight in {None, "fweight"}:
        residual = dense._state["resid"].numpy()
        if weight:
            residual = np.repeat(residual, dense._state["weights"].numpy().astype(int))
        expected_skew, expected_kurtosis = skew(residual), kurtosis(residual, fisher=False)
        assert jb["skewness"] == pytest.approx(expected_skew, rel=1e-9, abs=1e-12)
        assert jb["kurtosis"] == pytest.approx(expected_kurtosis, rel=1e-9)
        assert jb["statistic"] == pytest.approx(len(residual) / 6 * (expected_skew**2 + (expected_kurtosis - 3)**2 / 4), rel=1e-9)
    else:
        assert jb["code"] == "unsupported_diagnostic_weight"
    if weight is None:
        assert dw["statistic"] == pytest.approx(durbin_watson(dense._state["resid"].numpy()), rel=1e-11)
    else:
        assert dw["code"] == "unsupported_diagnostic_weight"


@pytest.mark.parametrize("weight", [None, "fweight"])
def test_influence_iterator_preserves_full_sample_global_df_and_positions(frame, weight):
    data = pd.concat([frame] * 4, ignore_index=True)
    data.loc[[3, 189, 414], "y"] = np.nan
    data.loc[287, "x"] = np.nan
    data.loc[510, "w"] = 0
    data.index = np.repeat(np.arange(len(data) // 2), 2)
    opts = dict(y="y", x=["x", "z"], weights="w" if weight else None, weight_type=weight)
    dense, stream = oe.ols(data=data, **opts), oe.ols(data=source(data, size=19), **opts)
    batches = list(post.iter_influence(stream, batch_rows=7))
    assert max(map(len, batches)) <= 7
    positions = [position for table in batches for position in table.attrs["physical_positions"]]
    assert positions == dense._state["positions"].tolist()
    observed = pd.concat(batches)
    for name in post._INFLUENCE_COLUMNS:
        reference = dense.predict(kind=name).iloc[positions][name]
        assert_allclose(observed[name], reference, rtol=1e-9, atol=1e-11)
    assert observed.index.tolist() == data.index[positions].tolist()
    summary = post._influence_summary(stream)
    assert summary["rows"] == len(positions) > 400
    assert summary["mode"] == "summary"
    for name in post._INFLUENCE_COLUMNS:
        assert summary["columns"][name]["count"] == len(positions)
        assert summary["columns"][name]["mean"] == pytest.approx(float(observed[name].mean()), rel=1e-10, abs=1e-12)
        assert summary["columns"][name]["minimum"] == pytest.approx(float(observed[name].min()), rel=1e-10)
        assert summary["columns"][name]["maximum"] == pytest.approx(float(observed[name].max()), rel=1e-10)


@pytest.mark.parametrize("kind", ["xb", "residual", "score", "stdp", "stdf", "stdr", "rstandard", "rstudent", "leverage", "cook",
                                 "covratio", "dfits", "welsch", "dfbeta"])
def test_prediction_adapter_keeps_global_variance_and_degrees(frame, kind):
    dense = oe.ols(data=frame, y="y", x=["x", "z"])
    stream = oe.ols(data=source(frame, size=13), y="y", x=["x", "z"])
    batches = [post.predict_batch(stream, frame.iloc[start:start + 7], kind=kind, sample_only=True)
               for start in range(0, len(frame), 7)]
    assert_allclose(pd.concat(batches), dense.predict(kind=kind), rtol=1e-9, atol=1e-11)


def test_prediction_adapter_aweight_hat_uses_global_weight_scale(frame):
    kwargs = dict(y="y", x=["x", "z"], weights="w", weight_type="aweight")
    dense, stream = oe.ols(data=frame, **kwargs), oe.ols(data=source(frame), **kwargs)
    assert "weight_scale" in stream._state
    actual = pd.concat([post.predict_batch(stream, frame.iloc[start:start + 7], kind="hat", sample_only=True)
                        for start in range(0, len(frame), 7)])
    assert_allclose(actual, dense.predict(kind="hat"), rtol=1e-10)


@pytest.mark.parametrize("weight,covariance", [("fweight", "nonrobust"), ("aweight", "nonrobust"), (None, "HC3")])
def test_stream_diagnostic_restrictions_are_actionable(frame, weight, covariance):
    model = oe.ols(data=source(frame), y="y", x=["x"], covariance=covariance,
                   weights="w" if weight else None, weight_type=weight)
    if weight:
        with pytest.raises(oe.AnalysisError) as failure:
            post.breusch_godfrey(model)
        assert failure.value.code == "unsupported_diagnostic_weight"
    if weight == "aweight" or covariance == "HC3":
        with pytest.raises(oe.AnalysisError):
            list(post.iter_influence(model))


def test_stream_diagnostics_influence_is_summary_not_entire_output(frame):
    dense = oe.ols(data=frame, y="y", x=["x", "z"])
    stream = oe.ols(data=source(frame), y="y", x=["x", "z"])
    expected = dense.diagnostics(lags=3)
    actual = post.diagnostics(stream, lags=3)
    for name in ("breusch_pagan", "white", "reset", "breusch_godfrey", "jarque_bera", "durbin_watson"):
        assert actual[name]["statistic"] == pytest.approx(expected[name]["statistic"], rel=1e-8, abs=1e-12)
    assert isinstance(actual["influence"], dict)
    assert actual["influence"]["mode"] == "summary"


def test_stream_diagnostics_reject_mutated_source(frame):
    model = oe.ols(data=source(frame), y="y", x=["x"])
    frame.loc[4, "x"] += 3
    with pytest.raises(oe.AnalysisError, match="changed"):
        post._residual_distribution(model)
    with pytest.raises(oe.AnalysisError, match="changed"):
        list(post.iter_influence(model))


@pytest.mark.parametrize("adjustment", ["dfadjust", "hansen"])
@pytest.mark.parametrize("powers", [2, (2, 3, 4)])
def test_stream_reset_adjusted_contrast_matches_dense(frame, adjustment, powers):
    options = dict(y="y", x=["x", "z"], covariance="HC3", **{adjustment: True})
    dense = oe.ols(data=frame, **options)
    stream = oe.ols(data=source(frame), **options)
    expected, actual = dense.reset_test(powers), post.reset_test(stream, powers)
    assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-8)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-8)
    assert actual["df_denom"] == pytest.approx(expected["df_denom"], rel=1e-9)
    if not isinstance(powers, int):
        assert "df_adjustment_note" in actual


@pytest.mark.parametrize("weight", ["fweight", "aweight"])
@pytest.mark.parametrize("intercept", [False, True])
def test_stream_reset_weighted_no_intercept_matches_dense(frame, weight, intercept):
    options = dict(y="y", x=["x", "z"], intercept=intercept,
                   weights="w", weight_type=weight, covariance="HC3")
    dense, stream = oe.ols(data=frame, **options), oe.ols(data=source(frame), **options)
    expected, actual = dense.reset_test(), post.reset_test(stream)
    assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-8)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-8)


def test_batch_sample_only_diagnostics_scatter_missing_fit_rows(frame):
    data = frame.copy()
    data.loc[[2, 19], "y"] = np.nan
    data.loc[7, "x"] = np.nan
    data.loc[28, "w"] = 0
    data.index = np.repeat(np.arange(len(data) // 2), 2)
    options = dict(y="y", x=["x", "z"], weights="w", weight_type="fweight")
    dense, stream = oe.ols(data=data, **options), oe.ols(data=source(data), **options)
    for kind in ("dfbeta", "dfits", "covratio", "welsch"):
        output = pd.concat([post.predict_batch(stream, data.iloc[start:start + 13], kind=kind, sample_only=True)
                            for start in range(0, len(data), 13)])
        assert_allclose(output, dense.predict(kind=kind), rtol=1e-9, atol=1e-11, equal_nan=True)
        assert output.index.tolist() == data.index.tolist()
        with pytest.raises(oe.AnalysisError) as failure:
            post.predict_batch(stream, data.iloc[:4], kind=kind)
        assert failure.value.code == "prediction_sample_required"


def test_stream_bg_invalid_lags_and_width_fail_before_replaying(frame):
    stream = oe.ols(data=source(frame), y="y", x=["x"])
    passes = stream._state["replay"].passes
    for lags in (True, 0, -1, 1.5, 180):
        with pytest.raises(oe.AnalysisError) as failure:
            post.breusch_godfrey(stream, lags)
        assert failure.value.code == "invalid_diagnostic"
    assert stream._state["replay"].passes == passes
    # The auxiliary parameter budget is independent of dataset row count.
    stream._state["streaming"]["physical_nobs"] = 10**11
    with pytest.raises(oe.AnalysisError) as failure:
        post.breusch_godfrey(stream, 384)
    assert failure.value.code == "diagnostic_memory_limit"


def test_public_streaming_diagnostic_and_influence_dispatch(frame):
    stream = oe.ols(data=source(frame), y="y", x=["x", "z"])
    assert stream.breusch_godfrey(2)["statistic"] == pytest.approx(post.breusch_godfrey(stream, 2)["statistic"])
    assert stream.diagnostics()["influence"]["mode"] == "summary"
    assert len(pd.concat(list(stream.iter_influence(batch_rows=11)))) == len(frame)


@pytest.mark.parametrize("covariance", ["hac", "bootstrap", "jackknife", "cluster_hc2", "cluster_hc3"])
def test_reset_auxiliary_retains_covariance_options(frame, covariance):
    data = frame.copy()
    data["t"] = np.arange(len(data)) * 2
    options = dict(y="y", x=["x", "z"], covariance=covariance)
    if covariance.startswith("cluster_"):
        options["cluster"] = "firm"
    if covariance == "hac":
        options.update(time="t", lags=3, kernel="parzen")
    if covariance == "bootstrap":
        options.update(reps=19, seed=876)
    dense, stream = oe.ols(data=data, **options), oe.ols(data=source(data), **options)
    expected, actual = dense.reset_test(), post.reset_test(stream)
    assert actual["covariance"] == covariance
    if covariance == "bootstrap":
        # Exact multinomial laws, different draw algorithms. Reproducibility
        # is required across reader batch boundaries within the stream path.
        assert actual["distribution"] == expected["distribution"] == "chi2"
        alternate = oe.ols(data=source(data, size=29), **options)
        expected = post.reset_test(alternate)
    assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-8)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-8)


def test_public_streamed_newdata_influence_uses_global_state(frame):
    dense = oe.ols(data=frame, y="y", x=["x", "z"])
    stream = oe.ols(data=source(frame), y="y", x=["x", "z"])
    new = frame.iloc[[11, 21, 50]].copy()
    for kind in ("leverage", "rstandard", "rstudent", "cook", "stdr", "stdf"):
        assert_allclose(stream.predict(data=new, kind=kind), dense.predict(data=new, kind=kind), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("expression", ["L(x)", "D(x)", "L(x, 2)"])
@pytest.mark.parametrize("method", ["ame", "mem"])
def test_time_transform_margins_refuse_ambiguous_interventions(frame, expression, method):
    data = frame.copy()
    data["t"] = np.arange(len(data))
    data.loc[9, "y"] = np.nan
    kwargs = dict(formula=f"y ~ z + {expression}", time="t")
    dense, stream = oe.ols(data=data, **kwargs), oe.ols(data=source(data, size=7), **kwargs)
    for fitted in (dense, stream):
        with pytest.raises(oe.AnalysisError) as failure:
            fitted.margins(method=method)
        assert failure.value.code == "unsupported_margins_transform"
        assert "lincom" in str(failure.value)
        assert fitted.lincom({expression: 1})["std_error"] > 0


@pytest.mark.parametrize("weight", [None, "fweight"])
def test_dense_influence_iterator_slices_retained_state_without_whole_output(frame, weight):
    data = frame.copy()
    data.loc[9, "y"] = np.nan
    model = oe.ols(data=data, y="y", x=["x", "z"], weights="w" if weight else None, weight_type=weight)
    tables = list(model.iter_influence(batch_rows=11))
    positions = [position for table in tables for position in table.attrs["physical_positions"]]
    assert positions == model._state["positions"].tolist()
    assert max(map(len, tables)) <= 11
    actual = pd.concat(tables)
    for kind in post._INFLUENCE_COLUMNS:
        assert_allclose(actual[kind], model.predict(kind=kind).iloc[positions][kind], rtol=1e-10, atol=1e-12)


def test_auxiliary_design_expansion_is_bounded_before_reader_materialization(frame, monkeypatch):
    dense = oe.ols(data=frame, y="y", x=["x", "z"])
    stream = oe.ols(data=source(frame, size=180), y="y", x=["x", "z"])
    original = post.Dataset.from_batches
    sizes = []

    def wrap(cls, factory, columns, **kwargs):
        def audited():
            for batch in factory():
                sizes.append((len(batch), len(batch.columns)))
                yield batch
        return original(audited, columns, **kwargs)

    monkeypatch.setattr(post.Dataset, "from_batches", classmethod(wrap))
    monkeypatch.setattr(post, "_AUXILIARY_BYTES", 1024)
    white = post.white_test(stream)
    bg = post.breusch_godfrey(stream, lags=4)
    assert white["statistic"] == pytest.approx(dense.white_test()["statistic"], rel=1e-9)
    assert bg["statistic"] == pytest.approx(dense.breusch_godfrey(4)["statistic"], rel=1e-9)
    assert sizes and max(rows * columns * 8 * 8 for rows, columns in sizes) <= 1024


def test_lag_prediction_keeps_missing_current_input_and_excluded_row_history(frame):
    data = frame.copy()
    data["t"] = np.arange(len(data))
    data.loc[17, "x"] = np.nan
    data.loc[30, "z"] = np.nan
    options = dict(formula="y ~ L(x) + z", time="t")
    dense, stream = oe.ols(data=data, **options), oe.ols(data=source(data, size=7), **options)
    assert 17 in dense._state["positions"].tolist()  # x(16) is present.
    reference = pd.Series(np.nan, index=data.index)
    coefficient = {item.term: item.estimate for item in dense.coefficients}
    reference = coefficient["Intercept"] + coefficient["L(x)"] * data.x.shift(1) + coefficient["z"] * data.z
    assert np.isfinite(reference.iloc[17])
    assert np.isfinite(reference.iloc[31])  # z(30) is missing but x(30) supplies history.
    for fitted in (dense, stream):
        predicted = fitted.predict(data=data)["xb"]
        assert_allclose(predicted, reference, rtol=1e-10, atol=1e-12, equal_nan=True)
        streamed = pd.concat(list(fitted.iter_predict(batch_rows=7)))["xb"]
        assert_allclose(streamed, reference, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert_allclose(dense.predict()["xb"], reference, rtol=1e-10, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
@pytest.mark.parametrize("streamed", [False, True])
def test_prediction_variance_uses_centered_basis_on_shifted_regressors(frame, covariance, streamed):
    import statsmodels.api as sm
    from scipy.stats import t
    shifted = frame.copy()
    shifted.x += 1e8
    # Use the representable shifted inputs, centered before independent fitting.
    baseline = shifted.copy()
    baseline.x -= 1e8
    options = dict(y="y", x=["x", "z"], covariance=covariance,
                   cluster="firm" if covariance == "cluster" else None)
    fitted = oe.ols(data=source(shifted) if streamed else shifted, **options)
    design = sm.add_constant(baseline[["x", "z"]]).to_numpy()
    reference = sm.OLS(baseline.y, design).fit(cov_type=covariance,
                   cov_kwds={"groups": baseline.firm} if covariance == "cluster" else {}, use_t=True)
    new = shifted.iloc[[2, 11, 58]].copy()
    new.x += [.2, -.3, 1.]
    new_design = np.column_stack([np.ones(len(new)), new.x - 1e8, new.z])
    variance = (new_design @ reference.cov_params().to_numpy() * new_design).sum(axis=1)
    expected_fitted = reference.predict(new_design)
    predicted = fitted.predict(data=new, interval="mean")
    degrees = 17 if covariance == "cluster" else len(frame) - 3
    critical = t.ppf(.975, degrees)
    assert_allclose(predicted["stdp"], variance**.5, rtol=2e-7, atol=1e-10)
    assert_allclose(predicted["xb"], expected_fitted, rtol=2e-7, atol=1e-8)
    assert_allclose(predicted["ci_low"], expected_fitted - critical * variance**.5, rtol=2e-7, atol=1e-8)
    assert_allclose(predicted["ci_high"], expected_fitted + critical * variance**.5, rtol=2e-7, atol=1e-8)
    if covariance == "nonrobust":
        bread = np.linalg.inv(design.T @ design)
        leverage = (new_design @ bread * new_design).sum(axis=1)
        assert_allclose(fitted.predict(data=new, kind="leverage")["leverage"], leverage, rtol=2e-7, atol=1e-11)
        assert_allclose(fitted.predict(data=new, interval="obs")["stdf"], (variance + reference.scale)**.5, rtol=2e-7)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
@pytest.mark.parametrize("streamed", [False, True])
def test_shifted_linear_nonlinear_and_joint_contrasts_use_stable_basis(frame, covariance, streamed):
    import statsmodels.api as sm
    import torch
    from scipy.stats import t
    offsets = np.array([1e8, -2e8])
    shifted = frame.copy()
    shifted[["x", "z"]] += offsets
    shifted.y += 1e8
    baseline = shifted.copy()
    baseline[["x", "z"]] -= offsets
    baseline.y -= 1e8
    options = dict(y="y", x=["x", "z"], covariance=covariance,
                   cluster="firm" if covariance == "cluster" else None)
    fitted = oe.ols(data=source(shifted, size=7) if streamed else shifted, **options)
    reference = sm.OLS(baseline.y, sm.add_constant(baseline[["x", "z"]])).fit(
        cov_type=covariance, cov_kwds={"groups": baseline.firm} if covariance == "cluster" else {}, use_t=True)
    parameters = reference.params.to_numpy()
    covariance_matrix = reference.cov_params().to_numpy()
    degrees = 17 if covariance == "cluster" else len(frame) - 3
    bridge = np.array([[1., *offsets], [0, 1, 0], [0, 0, 1]])
    for row in np.array([[2., .3, -.2], [-1., -.1, .4], [0., 1., -.5]]):
        raw_row = row @ bridge
        mapping = dict(zip(["Intercept", "x", "z"], raw_row, strict=True))
        actual = fitted.lincom(mapping, constant=-row[0] * 1e8)
        expected = float(row @ parameters)
        error = float((row @ covariance_matrix @ row)**.5)
        assert actual["estimate"] == pytest.approx(expected, rel=2e-7, abs=2e-7)
        assert actual["std_error"] == pytest.approx(error, rel=2e-7)
        assert actual["ci_low"] == pytest.approx(expected - t.ppf(.975, degrees) * error, rel=2e-7, abs=2e-7)
        single = fitted.test(mapping, value=row[0] * 1e8 + .3)
        assert single["t_statistic"] == pytest.approx((expected - .3) / error, rel=2e-7, abs=2e-7)
    base_matrix = np.array([[1., 0, 0], [0, 1, -2]])
    raw_matrix = base_matrix @ bridge
    assert int(torch.linalg.matrix_rank(torch.tensor(raw_matrix))) == 1
    null = np.array([.7, 0.])
    actual = fitted.test(raw_matrix, value=null + np.array([1e8, 0]))
    oracle = reference.f_test((base_matrix, null))
    assert actual["distribution"] == "F"
    assert actual["statistic"] == pytest.approx(float(oracle.fvalue), rel=2e-7)
    assert actual["p_value"] == pytest.approx(float(oracle.pvalue), rel=2e-6)
    expression = fitted.nlcom(lambda b: (b["Intercept"] + 1e8 * b["x"] - 2e8 * b["z"] - 1e8)**2)
    gradient = np.array([2 * parameters[0], 0, 0])
    assert expression["estimate"] == pytest.approx(parameters[0]**2, rel=2e-7, abs=2e-7)
    assert expression["std_error"] == pytest.approx(float((gradient @ covariance_matrix @ gradient)**.5), rel=2e-7)
    margins = fitted.margins(variables=["x", "z"])
    assert_allclose(margins["estimate"], parameters[1:], rtol=2e-7, atol=1e-9)
    assert_allclose(margins["std_error"], np.diag(covariance_matrix)[1:]**.5, rtol=2e-7)
    with pytest.raises(oe.AnalysisError) as dependent:
        fitted.test(np.vstack([raw_matrix[0], 2 * raw_matrix[0]]))
    assert dependent.value.code == "invalid_restrictions"


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
@pytest.mark.parametrize("streamed", [False, True])
def test_no_intercept_contrast_rank_and_variance_are_invariant_to_units(frame, covariance, streamed):
    import statsmodels.api as sm
    import torch
    scaled = frame.copy()
    scaled.x *= 1e8
    scaled.z *= 1e-8
    options = dict(y="y", x=["x", "z"], intercept=False, covariance=covariance,
                   cluster="firm" if covariance == "cluster" else None)
    fitted = oe.ols(data=source(scaled, size=11) if streamed else scaled, **options)
    reference = sm.OLS(frame.y, frame[["x", "z"]]).fit(
        cov_type=covariance, cov_kwds={"groups": frame.firm} if covariance == "cluster" else {}, use_t=True)
    raw_matrix = np.diag([1e8, 1e-8])
    assert int(torch.linalg.matrix_rank(torch.tensor(raw_matrix))) == 1
    actual = fitted.test(raw_matrix, value=[.3, -.2])
    oracle = reference.f_test((np.eye(2), [.3, -.2]))
    assert actual["statistic"] == pytest.approx(float(oracle.fvalue), rel=1e-10)
    assert actual["p_value"] == pytest.approx(float(oracle.pvalue), rel=1e-9)
    row = np.array([1., -.3])
    contrast = fitted.lincom({"x": 1e8, "z": -.3e-8}, constant=.2)
    estimate = float(row @ reference.params + .2)
    standard_error = float((row @ reference.cov_params() @ row)**.5)
    assert contrast["estimate"] == pytest.approx(estimate, rel=1e-10)
    assert contrast["std_error"] == pytest.approx(standard_error, rel=1e-10)
