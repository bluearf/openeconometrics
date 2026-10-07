"""Meaningful dense/replay and independent weighted covariance oracles."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.linear_ols.design import OLSDesign
from openecon.linear_ols.estimation import estimate
from openecon.linear_ols.streaming import fit_streaming
from openecon.models import ModelSpec


def fixture(n=137):
    generator = torch.Generator().manual_seed(28)
    x = torch.randn((n, 2), generator=generator, dtype=torch.float64)
    y = 1.8 + x @ torch.tensor([2.1, -.6], dtype=torch.float64)
    y += torch.randn(n, generator=generator, dtype=torch.float64) * (.4 + x[:, 0].abs())
    return pd.DataFrame({"y": y.numpy(), "x": x[:, 0].numpy(), "z": x[:, 1].numpy(),
                         "w": torch.randint(1, 5, (n,), generator=generator).numpy(),
                         "g": [i % 11 for i in range(n)], "h": [(i // 7) % 9 for i in range(n)],
                         "time": [i + i // 8 for i in range(n)]})


def replay(frame, chunk=7):
    return Dataset.from_batches(lambda: (frame.iloc[i:i + chunk] for i in range(0, len(frame), chunk)),
                                columns=frame.columns.tolist(), row_count=len(frame))


def oracle(frame, spec):
    design = OLSDesign(spec.options.get("terms", spec.predictors), spec.categorical,
                       intercept=spec.intercept, time=spec.time).prepare(frame)
    columns = list(dict.fromkeys([spec.outcome, *design.required,
                                 *([spec.weights] if spec.weights else []),
                                 *([spec.cluster] if isinstance(spec.cluster, str) else spec.cluster or []),
                                 *([spec.time] if spec.time else [])]))
    retained = frame.dropna(subset=columns)
    if spec.weights:
        retained = retained.loc[retained[spec.weights] > 0]
    clusters = [spec.cluster] if isinstance(spec.cluster, str) else spec.cluster or []
    return estimate(design.encode(retained), torch.tensor(retained[spec.outcome].to_numpy(), dtype=torch.float64),
                    terms=design.terms, intercept=spec.intercept, covariance=spec.covariance,
                    weights=torch.tensor(retained[spec.weights].to_numpy(), dtype=torch.float64) if spec.weights else None,
                    weight_type=spec.weight_type, clusters=[retained[name].tolist() for name in clusters] or None,
                    time=torch.tensor(retained[spec.time].to_numpy()) if spec.time else None,
                    lags=spec.options.get("lags"), kernel=spec.options.get("kernel", "bartlett"))


@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "pweight", "iweight"])
@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def test_weighted_all_hc_dense_parity(weight, covariance):
    if weight == "pweight" and covariance == "nonrobust":
        return
    frame = fixture()
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance=covariance,
                     weights="w" if weight else None, weight_type=weight)
    actual, expected = fit_streaming(spec, replay(frame)), oracle(frame, spec)
    torch.testing.assert_close(actual["params"], expected["params"], rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-10, atol=1e-12)
    assert actual["nobs"] == expected["nobs"]
    assert actual["df_resid"] == expected["df_resid"]
    for metric in ["ss_resid", "ss_total", "ss_model", "rmse", "r_squared", "log_likelihood"]:
        assert actual["metrics"][metric] == pytest.approx(expected["metrics"][metric], rel=1e-11, abs=1e-11)
    assert actual["streaming"]["passes"] == 3


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def test_frequency_weights_equal_literal_expanded_rows(covariance):
    frame = fixture(39)
    weighted = ModelSpec(outcome="y", predictors=["x", "z"], weights="w", weight_type="fweight", covariance=covariance)
    actual = fit_streaming(weighted, replay(frame, 3))
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    expected = fit_streaming(ModelSpec(outcome="y", predictors=["x", "z"], covariance=covariance), replay(expanded, 11))
    torch.testing.assert_close(actual["params"], expected["params"], rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "pweight", "iweight"])
def test_cluster_cgm_dense_parity_and_cleanup(dimensions, weight, tmp_path, monkeypatch):
    frame = fixture(89)
    frame["third"] = [i % 5 for i in range(len(frame))]
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="cluster",
                     cluster=["g", "h", "third"][:dimensions], weights="w" if weight else None, weight_type=weight)
    actual, expected = fit_streaming(spec, replay(frame)), oracle(frame, spec)
    torch.testing.assert_close(actual["params"], expected["params"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    assert actual["df_inference"] == expected["df_inference"]
    assert actual["streaming"]["passes"] == 3 + (2**dimensions - 1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "truncated"])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "iweight"])
def test_finite_hac_actual_gaps_across_tiny_batches(kernel, weight):
    frame = fixture(79)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="hac", time="time",
                     weights="w" if weight else None, weight_type=weight,
                     options={"lags": 9, "kernel": kernel})
    actual, expected = fit_streaming(spec, replay(frame, 2)), oracle(frame, spec)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-10, atol=1e-12)
    assert actual["streaming"]["hac_lags"] == 9


def test_categories_interactions_declared_unused_and_missing_physical_positions():
    frame = fixture(601)
    frame["cat"] = pd.Categorical(["β" if i % 3 else "α" for i in range(len(frame))], categories=["β", "α", "unused"], ordered=True)
    frame.loc[[1, 5, 6], "y"] = np.nan
    frame.loc[[9, 10], "w"] = 0
    frame.loc[11, "g"] = np.nan
    spec = ModelSpec(outcome="y", predictors=["x", "cat"], categorical=["cat"], covariance="cluster", cluster="g",
                     weights="w", weight_type="aweight", missing="drop", options={"terms": ["x*C(cat)"]})
    actual, expected = fit_streaming(spec, replay(frame, 5)), oracle(frame, spec)
    assert actual["design"].categories["cat"] == ["β", "α", "unused"]
    assert actual["terms"] == expected["terms"]
    assert set(actual["omitted_terms"]) == {"cat[unused]", "x:cat[unused]"}
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    positions = actual["sample_positions"].tolist()
    assert len(positions) == 400 and positions[:9] == [0, 2, 3, 4, 7, 8, 12, 13, 14]
    assert actual["x"].shape == (400, len(actual["params"]))
    assert actual["samples_only"] is True
    assert actual["streaming"]["original"] == 601
    assert actual["streaming"]["used"] == 595


@pytest.mark.parametrize("covariance", ["nonrobust", "HC2", "HC3"])
def test_intercept_only_constant_outcome_allowed(covariance):
    frame = pd.DataFrame({"y": [12.] * 27})
    result = fit_streaming(ModelSpec(outcome="y", covariance=covariance), replay(frame, 1))
    torch.testing.assert_close(result["params"], torch.tensor([12.], dtype=torch.float64))
    assert result["sigma2"] < 1e-26


def test_high_offset_stability_and_deterministic_collinear_omission():
    frame = fixture(241)
    frame["x"] += 1e12
    frame["z_duplicate"] = frame.z * 3
    frame["constant"] = 12.
    spec = ModelSpec(outcome="y", predictors=["x", "z", "z_duplicate", "constant"], covariance="HC3")
    actual, expected = fit_streaming(spec, replay(frame, 4)), oracle(frame, spec)
    assert actual["terms"] == expected["terms"] == ["Intercept", "x", "z"]
    torch.testing.assert_close(actual["params"], expected["params"], rtol=1e-11, atol=1e-10)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-10, atol=1e-10)


def test_source_change_between_passes_caught():
    frame = fixture(70)
    passes = 0

    def factory():
        nonlocal passes
        passes += 1
        changed = frame.copy()
        if passes > 1:
            changed.loc[17, "y"] += .5
        yield changed

    source = Dataset.from_batches(factory, columns=frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError, match="changed") as error:
        fit_streaming(ModelSpec(outcome="y", predictors=["x", "z"]), source)
    assert error.value.code == "source_changed"


def test_invalid_weights_and_time_rejected():
    frame = fixture(35)
    frame.loc[3, "w"] = -1
    with pytest.raises(AnalysisError, match="nonnegative"):
        fit_streaming(ModelSpec(outcome="y", predictors=["x"], weights="w", weight_type="aweight"), replay(frame))
    frame.loc[3, "w"] = 1
    frame.loc[9, "time"] = frame.loc[8, "time"]
    with pytest.raises(AnalysisError, match="strictly increasing"):
        fit_streaming(ModelSpec(outcome="y", predictors=["x"], covariance="hac", time="time", options={"lags": 3}), replay(frame, 2))


def test_wide_interactions_rejected_before_cartesian_design_allocation():
    frame = fixture(81)
    frame["first"] = [f"a{i % 21}" for i in range(len(frame))]
    frame["second"] = [f"b{i % 21}" for i in range(len(frame))]
    spec = ModelSpec(outcome="y", predictors=["first", "second"], options={"terms": ["C(first):C(second)"]})
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, replay(frame))
    assert error.value.code == "model_too_wide"


@pytest.mark.parametrize("covariance", ["cluster_hc2", "cluster_hc3", "jackknife"])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "pweight", "iweight"])
def test_gram_cluster_adjustments_and_delete_cluster_match_dense(covariance, weight, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(139)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance=covariance, cluster="g",
                     weights="w" if weight else None, weight_type=weight)
    actual, expected = fit_streaming(spec, replay(frame, 3)), oracle(frame, spec)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    assert actual["df_inference"] == expected["df_inference"]
    assert actual["streaming"]["cluster_subsets"][0]["cluster_aggregation"] == "sqlite_gram_spill"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "pweight", "iweight"])
def test_analytic_streamed_delete_row_jackknife_matches_dense(weight):
    frame = fixture(83)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="jackknife",
                     weights="w" if weight else None, weight_type=weight)
    actual, expected = fit_streaming(spec, replay(frame, 2)), oracle(frame, spec)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual["inference"]["bias_estimate"], expected["inference"]["bias_estimate"], rtol=1e-8, atol=1e-11)
    assert actual["df_inference"] == expected["df_inference"]
    assert actual["streaming"]["passes"] == 3


def test_cluster_gram_spill_width_above_old_vector_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(90)
    generator = torch.Generator().manual_seed(813)
    names = [f"u{i}" for i in range(32)]
    values = torch.randn((len(frame), len(names)), generator=generator, dtype=torch.float64)
    for index, name in enumerate(names):
        frame[name] = values[:, index].numpy()
    spec = ModelSpec(outcome="y", predictors=names, covariance="cluster_hc3", cluster="g")
    actual, expected = fit_streaming(spec, replay(frame)), oracle(frame, spec)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    assert actual["streaming"]["cluster_subsets"][0]["cluster_gram_columns"] > 1024
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kernel", ["bartlett", "parzen"])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight"])
def test_automatic_hac_newey_west_selector_matches_dense(kernel, weight):
    frame = fixture(287)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="hac", time="time",
                     weights="w" if weight else None, weight_type=weight,
                     options={"lags": "auto", "kernel": kernel})
    actual, expected = fit_streaming(spec, replay(frame, 3)), oracle(frame, spec)
    assert actual["inference"]["lags"] == expected["inference"]["lags"]
    assert actual["inference"]["pilot_lags"] == expected["inference"]["pilot_lags"]
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    assert actual["streaming"]["passes"] == 4


@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "iweight"])
@pytest.mark.parametrize("cluster", [None, "g"])
def test_exact_bootstrap_is_repeatable_and_bounded_across_chunks(weight, cluster, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(41)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="bootstrap", cluster=cluster,
                     weights="w" if weight else None, weight_type=weight, options={"reps": 13, "seed": 181})
    first, second = fit_streaming(spec, replay(frame, 2)), fit_streaming(spec, replay(frame, 11))
    torch.testing.assert_close(first["covariance"], second["covariance"], rtol=1e-10, atol=1e-12)
    assert first["inference"]["random_count_algorithm"] == "exact conditional-binomial multinomial"
    assert first["inference"]["distribution"] == "normal"
    assert math.isinf(first["df_inference"])
    assert first["inference"]["successful_reps"] == 13
    assert first["streaming"]["passes"] == 16 + int(cluster is not None)
    assert not list(tmp_path.iterdir())


def test_frequency_bootstrap_matches_dense_same_exact_multinomial_seed():
    frame = fixture(37)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="bootstrap",
                     weights="w", weight_type="fweight", options={"reps": 17, "seed": 5})
    actual = fit_streaming(spec, replay(frame, 3))
    design = OLSDesign(["x", "z"], [], intercept=True).prepare(frame)
    expected = estimate(design.encode(frame), torch.tensor(frame.y.to_numpy(), dtype=torch.float64),
                        terms=design.terms, covariance="bootstrap", weights=torch.tensor(frame.w.to_numpy(), dtype=torch.float64),
                        weight_type="fweight", reps=17, seed=5)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("covariance", ["HC2", "HC3", "cluster_hc2", "cluster_hc3"])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight"])
def test_adjusted_coefficient_and_arbitrary_contrast_inference(covariance, weight, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(83)
    clustered = covariance.startswith("cluster")
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance=covariance, cluster="g" if clustered else None,
                     weights="w" if weight else None, weight_type=weight, options={"dfadjust": True})
    actual = fit_streaming(spec, replay(frame, 3))
    design = OLSDesign(["x", "z"], [], intercept=True).prepare(frame)
    expected = estimate(design.encode(frame), torch.tensor(frame.y.to_numpy(), dtype=torch.float64), terms=design.terms,
                        covariance=covariance, weights=torch.tensor(frame.w.to_numpy(), dtype=torch.float64) if weight else None,
                        weight_type=weight, clusters=[frame.g.tolist()] if clustered else None, dfadjust=True)
    assert actual["inference"]["dfadjust"] is True
    assert actual["inference"]["coefficient_df"] == pytest.approx(expected["inference"]["coefficient_df"], rel=1e-9)
    gradient = torch.tensor([.5, 2., -1.], dtype=torch.float64)
    first = actual["contrast_inference"](gradient)
    second = expected["contrast_inference"](gradient)
    assert first == pytest.approx(second, rel=1e-9)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("clustered", [False, True])
def test_hansen_scale_and_contrast_df_match_primary_formula(clustered, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(63)
    covariance = "cluster_hc3" if clustered else "HC3"
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance=covariance, cluster="g" if clustered else None,
                     options={"hansen": True})
    actual = fit_streaming(spec, replay(frame, 2))
    design = OLSDesign(["x", "z"], [], intercept=True).prepare(frame)
    expected = estimate(design.encode(frame), torch.tensor(frame.y.to_numpy(), dtype=torch.float64), terms=design.terms,
                        covariance=covariance, clusters=[frame.g.tolist()] if clustered else None, hansen=True)
    assert actual["inference"]["hansen"] is True
    assert actual["inference"]["coefficient_scale"] == pytest.approx(expected["inference"]["coefficient_scale"], rel=1e-9)
    assert actual["inference"]["coefficient_df"] == pytest.approx(expected["inference"]["coefficient_df"], rel=1e-9)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("lags", [0, 4, "auto"])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight"])
def test_exact_quadratic_spectral_hac_disk_all_pairs(lags, weight, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(77)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="hac", time="time",
                     weights="w" if weight else None, weight_type=weight,
                     options={"lags": lags, "kernel": "quadratic_spectral"})
    actual, expected = fit_streaming(spec, replay(frame, 3)), oracle(frame, spec)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-8, atol=1e-12)
    assert actual["inference"]["noncompact_support"] is True
    assert actual["inference"]["lags"] == pytest.approx(expected["inference"]["lags"], rel=1e-8)
    assert actual["streaming"]["computational_complexity"] == "quadratic time; bounded RAM"
    assert actual["streaming"]["hac_score_rows"] == len(frame)
    assert any("quadratic in rows" in warning for warning in actual["warnings"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("covariance", ["nonrobust", "HC2", "HC3", "cluster", "hac"])
def test_no_intercept_all_category_columns_and_weighted_covariance(covariance):
    frame = fixture(91)
    frame["cat"] = pd.Categorical(["β" if i % 3 else "α" for i in range(len(frame))], categories=["β", "α"])
    spec = ModelSpec(outcome="y", predictors=["x", "cat"], categorical=["cat"], intercept=False,
                     covariance=covariance, cluster="g" if covariance == "cluster" else None,
                     time="time" if covariance == "hac" else None,
                     weights="w", weight_type="aweight", options={"lags": 4} if covariance == "hac" else {})
    actual, expected = fit_streaming(spec, replay(frame, 3)), oracle(frame, spec)
    assert actual["terms"] == expected["terms"] == ["x", "cat[β]", "cat[α]"]
    torch.testing.assert_close(actual["params"], expected["params"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    assert actual["metrics"]["ss_total"] == pytest.approx(expected["metrics"]["ss_total"])


def test_formula_drop_hashes_only_final_retained_positions():
    frame = fixture(101)
    frame["cat"] = ["α" if i % 2 else "β" for i in range(len(frame))]
    spec = ModelSpec(outcome="y", predictors=["x", "cat"], covariance="HC3", missing="drop",
                     options={"terms": ["log(x)*C(cat)"]})
    actual = fit_streaming(spec, replay(frame, 2))
    kept = frame.loc[frame.x > 0]
    expected = fit_streaming(spec, replay(kept, 13))
    torch.testing.assert_close(actual["params"], expected["params"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    assert actual["sample_positions"].tolist() == kept.index.tolist()
    assert actual["streaming"]["used"] == len(kept)
    assert actual["dropped_rows"] == len(frame) - len(kept)


def test_qs_pair_tiles_cross_512_boundaries_match_dense(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(1037)
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance="hac", time="time",
                     options={"lags": 7, "kernel": "quadratic_spectral"})
    actual, expected = fit_streaming(spec, replay(frame, 13)), oracle(frame, spec)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-8, atol=1e-12)
    assert not list(tmp_path.iterdir())


def test_qs_spill_cleans_after_covariance_failure(tmp_path, monkeypatch):
    from openecon.linear_ols.streaming import _QuadraticSpectral
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))

    def fail(self):
        raise RuntimeError("owned covariance failure")

    monkeypatch.setattr(_QuadraticSpectral, "finish", fail)
    frame = fixture(41)
    spec = ModelSpec(outcome="y", predictors=["x"], covariance="hac", time="time",
                     options={"lags": 2, "kernel": "quadratic_spectral"})
    with pytest.raises(RuntimeError, match="owned covariance"):
        fit_streaming(spec, replay(frame))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("column", ["time", "w"])
def test_integer_precision_guard_precedes_float_conversion(column):
    frame = fixture(37)
    frame.loc[3, column] = 2**53 + 1
    spec = (ModelSpec(outcome="y", predictors=["x"], covariance="hac", time="time", options={"lags": 2})
            if column == "time" else ModelSpec(outcome="y", predictors=["x"], weights="w", weight_type="fweight"))
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, replay(frame))
    assert error.value.code in {"invalid_time", "invalid_weights"}


@pytest.mark.parametrize("n", [90, 99, 240])
def test_iweight_classical_near_integer_total_is_chunk_independent(n):
    frame = fixture(n)
    frame["w"] = ([1.1, 1.2, .7] * (n // 3 + 1))[:n]
    spec = ModelSpec(outcome="y", predictors=["x", "z"], weights="w", weight_type="iweight", covariance="nonrobust")
    expected = oracle(frame, spec)
    for chunk in [1, 3, 7, 1000]:
        actual = fit_streaming(spec, replay(frame, chunk))
        assert actual["nobs"] == expected["nobs"] == n
        torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)


def test_lags_and_differences_keep_raw_missing_zero_weight_history():
    frame = fixture(83)
    frame["time"] = list(range(len(frame)))
    frame.loc[[2, 7], "y"] = np.nan
    frame.loc[[3, 11], "w"] = 0
    # The observations after rows with missing y / zero w still use those
    # rows' x values as lag history, matching time-series operator semantics.
    spec = ModelSpec(outcome="y", predictors=["x", "z"], weights="w", weight_type="aweight", missing="drop",
                     covariance="HC3", time="time", options={"terms": ["L(x, 2)", "D(z)"]})
    actual = fit_streaming(spec, replay(frame, 3))
    augmented = frame.copy()
    augmented["lag_x"] = frame.x.shift(2)
    augmented["diff_z"] = frame.z.diff()
    expected = fit_streaming(ModelSpec(outcome="y", predictors=["lag_x", "diff_z"], weights="w", weight_type="aweight",
                                      missing="drop", covariance="HC3", time="time"), replay(augmented, 11))
    torch.testing.assert_close(actual["params"], expected["params"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=1e-9, atol=1e-12)
    assert actual["sample_positions"].tolist() == expected["sample_positions"].tolist()
    assert actual["dropped_rows"] == expected["dropped_rows"]
    assert callable(actual["state"]["residual_block"])


def test_lag_gaps_and_unsorted_time_are_explicit():
    frame = fixture(41)
    spec = ModelSpec(outcome="y", predictors=["x"], missing="drop", time="time", options={"terms": ["L(x)"]})
    actual = fit_streaming(spec, replay(frame, 2))
    assert 0 not in actual["sample_positions"].tolist()
    assert 8 not in actual["sample_positions"].tolist()
    shuffled = frame.iloc[[1, 0, *range(2, len(frame))]]
    with pytest.raises(AnalysisError, match="ascending"):
        fit_streaming(spec, replay(shuffled))


def test_new_object_category_after_discovery_is_source_changed():
    frame = fixture(31)
    frame["category"] = ["α" if i % 2 else "β" for i in range(len(frame))]
    passes = 0

    def factory():
        nonlocal passes
        passes += 1
        changed = frame.copy()
        if passes > 1:
            changed.loc[9, "category"] = "new"
        yield changed

    source = Dataset.from_batches(factory, columns=frame.columns.tolist(), row_count=len(frame))
    spec = ModelSpec(outcome="y", predictors=["x", "category"], categorical=["category"])
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec, source)
    assert error.value.code == "source_changed"
