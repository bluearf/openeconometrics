"""Independent full-design weighted-population SRSWR reference acceptance."""

from __future__ import annotations

import hashlib
from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def fixture():
    return pd.DataFrame({
        "w": [1., 2., 1., 3., 2., 4., 3., 2.],
        "psu": [1, 1, 2, 2, 1, 1, 2, 2],
        "stratum": ["a"] * 4 + ["b"] * 4,
        "y": [1., 2., 5., 9., 3., 8., 7., 11.],
        "z": [6., 4., 3., 8., 2., 5., 9., 1.],
        "x": [2., 3., 1., 4., 5., 2., 6., 3.],
        "q": [3., 1., 4., 2., 3., 5., 2., 6.],
        "category": ["red", "blue", "red", "blue", "blue", "red", "blue", "red"],
        "domain": [1, 1, 1, 0, 0, 0, 0, 0],
    })


def call(frame, kind, *, domain=None, missing="raise", deff=True, census=False):
    if census:
        frame = frame.assign(N=2)
    design = oe.survey_design(
        frame, weights="w", psu="psu", strata="stratum", fpc="N" if census else None
    )
    options = {"domain": domain, "missing": missing, "deff": deff}
    if kind == "proportion":
        return oe.survey_proportion(
            frame, design, "category", categories=["red", "blue", "absent"], **options
        )
    if kind == "ratio":
        return oe.survey_ratio(frame, design, ["y", "z"], ["x", "q"], **options)
    return getattr(oe, f"survey_{kind}")(frame, design, ["y", "z"], **options)


def independent(frame, kind, domain=None):
    """Direct equations and scalar moments without production helpers."""
    w = frame.w.to_numpy()
    use = np.ones(len(frame), dtype=bool)
    if domain:
        use &= frame[domain].to_numpy().astype(bool)
    if kind == "proportion":
        use &= frame.category.notna().to_numpy()
        values = np.column_stack(
            [frame.category.to_numpy() == c for c in ["red", "blue", "absent"]]
        ).astype(float)
        denominator = np.ones_like(values)
    else:
        roles = ["y", "z"] + (["x", "q"] if kind == "ratio" else [])
        use &= frame[roles].notna().all(axis=1).to_numpy()
        values = frame[["y", "z"]].fillna(0).to_numpy()
        denominator = (frame[["x", "q"]].fillna(0).to_numpy()
                       if kind == "ratio" else np.ones_like(values))
    numerator = (w * use) @ values
    if kind == "total":
        theta, influence = numerator, use[:, None] * values
    else:
        total_x = (w * use) @ denominator
        theta = numerator / total_x
        influence = use[:, None] * (values - denominator * theta) / total_x
    mean = sum(w[i] * influence[i] for i in range(len(w))) / sum(w)
    moments = sum(weight * np.outer(row - mean, row - mean)
                  for weight, row in zip(w, influence))
    reference = sum(w) / (len(w) - 1) * moments
    covariance = np.zeros_like(reference)
    for h in ["a", "b"]:
        rows = frame.stratum.to_numpy() == h
        totals = [(w[rows & (frame.psu.to_numpy() == p), None]
                   * influence[rows & (frame.psu.to_numpy() == p)]).sum(0)
                  for p in [1, 2]]
        covariance += np.outer(totals[0] - totals[1], totals[0] - totals[1])
    return theta, covariance, reference, mean, moments, influence


@pytest.mark.parametrize("kind", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("selection", ["all", "domain", "missing"])
def test_full_covariance_independent_weighted_moments(kind, selection):
    frame = fixture()
    domain = "domain" if selection != "all" else None
    if selection == "missing":
        frame.loc[1, "category" if kind == "proportion" else "y"] = np.nan
        frame.loc[4:, "category" if kind == "proportion" else "z"] = np.nan
    actual = call(frame, kind, domain=domain,
                  missing="drop" if selection == "missing" else "raise")
    theta, covariance, reference, mean, moments, _ = independent(frame, kind, domain)
    for observed, expected in [
        (actual.estimates, theta), (actual.covariance, covariance),
        (actual.metadata["srs_covariance"], reference),
        (actual.metadata["srs_reference_state"]["weighted_influence_mean"], mean),
        (actual.metadata["srs_reference_state"]["weighted_centered_crossproducts"], moments),
    ]:
        np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=1e-13)
    record = actual.metadata["srs_reference_state"]
    assert record["n_design"] == 8 and actual.df == 2
    assert record["sum_design_weights"] == sum(frame.w)
    assert record["fpc_applied"] is False
    assert record["eligibility"] == "fixed-domain-and-joint-complete-case-indicator"
    for i, effect in enumerate(actual.metadata["design_effect"]):
        if reference[i, i] > 0:
            assert effect == pytest.approx(covariance[i, i] / reference[i, i], rel=2e-13)
        else:
            assert effect is None
    restored = oe.SurveyResult.model_validate_json(actual.model_dump_json())
    assert restored.model_dump_json() == actual.model_dump_json()
    pd.testing.assert_frame_equal(restored.to_frame(), actual.to_frame())
    pd.testing.assert_frame_equal(restored.contrast([1.] * len(theta)),
                                  actual.contrast([1.] * len(theta)))
    assert oe.to_latex(restored.to_frame()) == oe.to_latex(actual.to_frame())


@pytest.mark.parametrize("kind", ["mean", "total", "ratio", "proportion"])
def test_weight_scaling_and_fpc_are_separate(kind):
    frame = fixture()
    original = call(frame, kind, domain="domain")
    scaled = call(frame.assign(w=frame.w * 7), kind, domain="domain")
    np.testing.assert_allclose(
        scaled.metadata["srs_covariance"],
        np.array(original.metadata["srs_covariance"]) * (49 if kind == "total" else 1),
        rtol=2e-13, atol=1e-13,
    )
    assert scaled.metadata["design_effect"] == pytest.approx(
        original.metadata["design_effect"], rel=2e-13
    )
    census = call(frame, kind, domain="domain", census=True)
    assert np.count_nonzero(census.covariance) == 0
    assert census.metadata["srs_covariance"] == original.metadata["srs_covariance"]
    for ref, effect in zip(np.diag(census.metadata["srs_covariance"]),
                           census.metadata["design_effect"]):
        assert effect == (0. if ref > 0 else None)


@pytest.mark.parametrize("kind", ["mean", "total", "ratio", "proportion"])
def test_complete_integer_weight_srswr_draw_enumeration(kind):
    frame = fixture().iloc[:4].copy()
    frame["w"] = [1., 2., 1., 2.]
    frame["stratum"], frame["psu"] = ["a", "a", "b", "b"], [1, 2, 1, 2]
    frame["domain"] = [1, 1, 0, 0]
    actual = call(frame, kind, domain="domain")
    _, _, reference, _, _, influence = independent(frame, kind, "domain")
    population = np.repeat(influence, frame.w.to_numpy().astype(int), axis=0)
    n, weight_sum = len(frame), sum(frame.w)
    # Exact linearized estimator draws; this does not claim exact nonlinear ratio variance.
    samples = np.array([weight_sum / n * population[list(draw)].sum(0)
                        for draw in product(range(len(population)), repeat=n)])
    exact = np.cov(samples, rowvar=False, bias=True) * n / (n - 1)
    np.testing.assert_allclose(reference, exact, rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(actual.metadata["srs_covariance"], exact, rtol=2e-13, atol=1e-13)


@pytest.mark.parametrize("kind", ["mean", "total", "ratio", "proportion"])
@pytest.mark.parametrize("domain", [None, "domain"])
def test_equal_weight_metadata_and_formula_are_unchanged(kind, domain):
    frame = fixture().assign(w=2.)
    actual = call(frame, kind, domain=domain)
    _, _, _, _, _, influence = independent(frame, kind, domain)
    weighted = frame.w.to_numpy()[:, None] * influence
    centered = weighted - weighted.mean(0)
    legacy = len(frame) / (len(frame) - 1) * centered.T @ centered
    np.testing.assert_allclose(actual.metadata["srs_covariance"], legacy, rtol=2e-13, atol=1e-13)
    assert actual.metadata["srs_reference"] == (
        "equal-weight independent row PSUs, full population/domain geometry, with replacement, no FPC"
    )
    assert "srs_reference_state" not in actual.metadata
    assert oe.SurveyResult.model_validate_json(actual.model_dump_json()) == actual


def rehash(state):
    state["integrity_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in state.items() if k != "integrity_sha256"},
        sort_keys=True, ensure_ascii=False, allow_nan=True, separators=(",", ":"),
    ).encode()).hexdigest()


@pytest.mark.parametrize("mutation", [
    "missing_state", "missing_label", "unknown_label", "unknown_field", "unknown_schema",
    "wor_law", "fpc", "eligibility", "n_bool", "n_wrong", "n_zero",
    "weight_bool", "weight_wrong", "weight_zero", "weight_nan", "weight_huge_int",
    "mean_shape", "mean_bool", "mean_nan", "mean_wrong", "mean_huge_int", "psu_shape", "psu_wrong",
    "moments_shape", "moments_bool", "moments_nan", "moments_asymmetric",
    "moments_non_psd", "moments_wrong", "reference_shape", "reference_bool",
    "reference_nan", "reference_asymmetric", "reference_non_psd", "reference_wrong",
    "effect_shape", "effect_bool", "effect_nan", "effect_negative", "effect_wrong", "effect_none",
    "effect_huge_int",
    "group_shape", "group_wrong", "correction_wrong", "covariance_wrong",
])
def test_rehashed_reference_state_mutations_are_refused(mutation):
    raw = call(fixture(), "mean").model_dump(mode="json")
    metadata, record = raw["metadata"], raw["metadata"]["srs_reference_state"]
    if mutation == "missing_state":
        metadata.pop("srs_reference_state")
    elif mutation in {"missing_label", "unknown_label"}:
        if mutation == "missing_label":
            metadata.pop("srs_reference")
        else:
            metadata["srs_reference"] = "hypothetical SRSWOR"
    elif mutation == "unknown_field":
        record["calibration"] = True
    elif mutation == "unknown_schema":
        record["schema_version"] = "unsupported-v2"
    elif mutation == "wor_law":
        record["law"] = "srswor"
    elif mutation == "fpc":
        record["fpc_applied"] = True
    elif mutation == "eligibility":
        record["eligibility"] = "drop-out-of-domain-rows"
    elif mutation.startswith("n_"):
        record["n_design"] = {"n_bool": True, "n_wrong": 7, "n_zero": 0}[mutation]
    elif mutation.startswith("weight_"):
        record["sum_design_weights"] = {
            "weight_bool": True, "weight_wrong": 9., "weight_zero": 0.,
            "weight_nan": float("nan"),
            "weight_huge_int": 10 ** 1000,
        }[mutation]
    elif mutation.startswith("mean_"):
        if mutation == "mean_shape":
            record["weighted_influence_mean"].pop()
        else:
            record["weighted_influence_mean"][0] = {
                "mean_bool": True, "mean_nan": float("nan"), "mean_wrong": 1.,
                "mean_huge_int": 10 ** 1000,
            }[mutation]
    elif mutation.startswith("psu_"):
        if mutation == "psu_shape":
            metadata["psu_influence_sums"].pop()
        else:
            metadata["psu_influence_sums"][0][0] += 1.
    elif mutation.startswith(("moments_", "reference_")):
        prefix, action = mutation.split("_", 1)
        matrix = (record["weighted_centered_crossproducts"] if prefix == "moments"
                  else metadata["srs_covariance"])
        if action == "shape":
            matrix.pop()
        elif action == "bool":
            matrix[0][0] = True
        elif action == "nan":
            matrix[0][0] = float("nan")
        elif action == "asymmetric":
            matrix[0][1] += 1.
        elif action == "non_psd":
            matrix[0][0] = -1.
        elif action == "wrong":
            matrix[0][0] += 1.
    elif mutation.startswith("effect_"):
        if mutation == "effect_shape":
            metadata["design_effect"].pop()
        else:
            metadata["design_effect"][0] = {
                "effect_bool": True, "effect_nan": float("nan"), "effect_negative": -1.,
                "effect_wrong": 1e6, "effect_none": None,
                "effect_huge_int": 10 ** 1000,
            }[mutation]
    elif mutation == "group_shape":
        metadata["stratum_psu_indices"].pop()
    elif mutation == "group_wrong":
        metadata["stratum_psu_indices"][0][0] = 3
    elif mutation == "correction_wrong":
        metadata["fpc_multipliers"][0] = .5
    elif mutation == "covariance_wrong":
        raw["covariance"][0][0] += 1.
    rehash(raw)
    with pytest.raises(ValueError, match="SRSWR"):
        oe.SurveyResult.model_validate(raw)


def test_zero_reference_has_undefined_design_effect():
    raw = call(fixture(), "proportion").model_dump(mode="json")
    assert raw["metadata"]["design_effect"][-1] is None
    raw["metadata"]["design_effect"][-1] = 0.
    rehash(raw)
    with pytest.raises(ValueError, match="undefined design effect"):
        oe.SurveyResult.model_validate(raw)


@pytest.mark.parametrize("deff", ["srswor", "calibrated", None, 1, []])
def test_no_implicit_reference_selector_or_fallback(deff):
    with pytest.raises(AnalysisError, match="deff must be Boolean"):
        call(fixture(), "mean", deff=deff)


def test_moment_work_guard_precedes_moment_allocation(monkeypatch):
    from openecon.econometrics.survey import targets

    monkeypatch.setattr(targets, "MAX_DEFF_WORK", 1)
    with pytest.raises(AnalysisError, match="full-design moment work"):
        call(fixture(), "mean")


def test_reference_overflow_is_an_explicit_numerical_error():
    frame = fixture().iloc[:4].copy()
    frame["w"], frame["stratum"] = [1., 2., 1., 2.], "a"
    frame["y"] = [2e200, -1e200, 2e200, -1e200]
    with pytest.raises(AnalysisError, match="SRSWR weighted centered crossproducts overflowed"):
        call(frame, "total")


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_finite_weight_scales_preserve_mean_reference_and_validation(scale):
    original = call(fixture(), "mean")
    actual = call(fixture().assign(w=lambda data: data.w * scale), "mean")
    np.testing.assert_allclose(actual.metadata["srs_covariance"],
                               original.metadata["srs_covariance"], rtol=2e-13, atol=1e-13)
    assert oe.SurveyResult.model_validate_json(actual.model_dump_json()) == actual
    raw = actual.model_dump(mode="json")
    # Mutate mean and corresponding PSU totals coherently: target residual identity must still hold.
    reference = raw["metadata"]["srs_reference_state"]
    reference["weighted_influence_mean"][0] = 1.0 / scale
    raw["metadata"]["psu_influence_sums"][0][0] = (
        raw["design"]["validation"]["sum_weights"] / scale
        - sum(row[0] for row in raw["metadata"]["psu_influence_sums"][1:])
    )
    rehash(raw)
    with pytest.raises(ValueError, match="target estimating equation"):
        oe.SurveyResult.model_validate(raw)
