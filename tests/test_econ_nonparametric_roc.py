"""Nonparametric family: roc and roccomp against brute-force pairwise comparisons."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.nonparametric.roc import placements


@pytest.fixture(scope="module")
def cases():
    rng = np.random.default_rng(51)
    n = 90
    y = rng.integers(0, 2, size=n)
    s1 = np.round(rng.normal(size=n) + y, 1)                        # rounded: tied scores
    s2 = np.round(0.5 * s1 + rng.normal(size=n) + 0.5 * y, 1)
    s3 = rng.normal(size=n)
    return pd.DataFrame({"y": y, "s1": s1, "s2": s2, "s3": s3})


def pairwise_kernel(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """psi[i, j] = 1 if positive i scores above negative j, 1/2 if tied (n1-by-n0, test only)."""
    positive, negative = score[y == 1][:, None], score[y == 0][None, :]
    return (positive > negative) + 0.5 * (positive == negative)


def test_placements_equal_the_pairwise_kernel_means(cases):
    y = cases["y"].to_numpy()
    psi = pairwise_kernel(cases["s1"].to_numpy(), y)
    v10, v01, _ = placements(torch.as_tensor(cases["s1"].to_numpy()), torch.as_tensor(y == 1))
    np.testing.assert_allclose(v10.numpy(), psi.mean(1), atol=1e-13)
    np.testing.assert_allclose(v01.numpy(), psi.mean(0), atol=1e-13)


def test_roc_area_standard_errors_and_test(cases):
    y, score = cases["y"].to_numpy(), cases["s1"].to_numpy()
    result = oe.roc(cases, "y", "s1", alpha=0.10)
    assert isinstance(result, TableSet) and list(result) == ["curve", "auc"]
    attrs = result.attrs
    psi = pairwise_kernel(score, y)
    n1, n0 = psi.shape
    area = psi.mean()
    assert attrs["auc"] == pytest.approx(area, rel=1e-13)
    assert (attrs["n_positive"], attrs["n_negative"]) == (n1, n0)
    delong = math.sqrt(psi.mean(1).var(ddof=1) / n1 + psi.mean(0).var(ddof=1) / n0)
    assert attrs["std_error"] == pytest.approx(delong, rel=1e-11)
    # Hanley and McNeil (1982, Table II): a tied pair contributes n=^2 / 3, not the
    # n=^2 / 4 of the squared placement, so each Q gains mean(n=^2) / 12.
    tied = psi == 0.5
    q1 = (psi.mean(0) ** 2).mean() + (tied.sum(0) ** 2).mean() / (12 * n1 ** 2)
    q2 = (psi.mean(1) ** 2).mean() + (tied.sum(1) ** 2).mean() / (12 * n0 ** 2)
    hanley = math.sqrt((area * (1 - area) + (n1 - 1) * (q1 - area ** 2)
                        + (n0 - 1) * (q2 - area ** 2)) / (n1 * n0))
    assert attrs["se_hanley_mcneil"] == pytest.approx(hanley, rel=1e-11)
    table = result["auc"]
    critical = stats.norm.isf(0.05)
    assert table.loc["delong", "ci_low"] == pytest.approx(area - critical * delong)
    assert table.loc["hanley_mcneil", "ci_high"] == pytest.approx(area + critical * hanley)
    # H0: area = 0.5 is the Mann-Whitney test.
    reference = stats.mannwhitneyu(score[y == 1], score[y == 0], use_continuity=False,
                                   method="asymptotic")
    assert attrs["auc"] == pytest.approx(reference.statistic / (n1 * n0))
    assert attrs["p_value"] == pytest.approx(reference.pvalue, rel=1e-9)
    ranksum = oe.ranksum(cases, "s1", "y").attrs
    assert attrs["z"] == pytest.approx(-ranksum["z"]) and attrs["auc"] == pytest.approx(
        1 - ranksum["porder"])


def test_roc_curve_coordinates_and_trapezoid(cases):
    y, score = cases["y"].to_numpy(), cases["s1"].to_numpy()
    result = oe.roc(cases, "y", "s1")
    curve = result["curve"]
    distinct = np.unique(score)
    assert len(curve) == len(distinct) + 1 == result.attrs["curve_points"]
    thresholds = curve["threshold"].to_numpy()
    assert thresholds[0] == distinct[0] - 1 and thresholds[-1] == distinct[-1] + 1
    np.testing.assert_allclose(thresholds[1:-1], (distinct[:-1] + distinct[1:]) / 2)
    for row in curve.iloc[::5].itertuples():
        predicted = score >= row.threshold
        assert row.sensitivity == pytest.approx(predicted[y == 1].mean())
        assert row.specificity == pytest.approx(1 - predicted[y == 0].mean())
        assert row.one_minus_specificity == pytest.approx(predicted[y == 0].mean())
    assert curve.iloc[0][["sensitivity", "one_minus_specificity"]].tolist() == [1.0, 1.0]
    assert curve.iloc[-1][["sensitivity", "one_minus_specificity"]].tolist() == [0.0, 0.0]
    trapezoid = -np.trapezoid(curve["sensitivity"], curve["one_minus_specificity"])
    assert trapezoid == pytest.approx(result.attrs["auc"], rel=1e-12)
    youden = curve["sensitivity"] + curve["specificity"] - 1
    assert result.attrs["youden_index"] == pytest.approx(youden.max())
    assert result.attrs["youden_threshold"] == pytest.approx(curve["threshold"][youden.idxmax()])


def test_roc_positive_level_perfect_and_constant_scores():
    data = {"state": ["ill", "ill", "ok", "ok", "ill", "ok"], "m": [5.0, 6.0, 1.0, 2.0, 7.0, 3.0]}
    default = oe.roc(data, "state", "m")                    # larger label ("ok") is positive
    assert default.attrs["positive"] == "ok" and default.attrs["auc"] == 0.0
    ill = oe.roc(data, "state", "m", positive="ill")
    assert ill.attrs["auc"] == 1.0 and ill.attrs["std_error"] == 0.0
    assert ill["auc"].loc["delong", ["ci_low", "ci_high"]].tolist() == [1.0, 1.0]
    constant = oe.roc({"y": [0, 1, 0, 1, 1], "m": [2.0] * 5}, "y", "m")
    assert constant.attrs["auc"] == 0.5 and "p_value" not in constant.attrs
    assert constant.attrs["notes"] and len(constant["curve"]) == 2
    tiny = oe.roc({"y": [0, 1, 0], "m": [1.0, 3.0, 2.0]}, "y", "m")
    assert tiny.attrs["std_error"] is None and tiny.attrs["se_hanley_mcneil"] == 0.0
    assert math.isnan(tiny["auc"].loc["delong", "ci_low"])


def test_roc_curve_is_thinned_to_400_points_but_area_uses_all():
    rng = np.random.default_rng(52)
    n = 20_000
    y = rng.integers(0, 2, size=n)
    score = rng.normal(size=n) + 0.8 * y
    result = oe.roc({"y": y, "m": score}, "y", "m")
    assert len(result["curve"]) == 400 and result.attrs["curve_points"] == n + 1
    assert result["curve"]["sensitivity"].iloc[0] == 1.0
    assert result["curve"]["sensitivity"].iloc[-1] == 0.0
    reference = stats.mannwhitneyu(score[y == 1], score[y == 0]).statistic
    assert result.attrs["auc"] == pytest.approx(reference / ((y == 1).sum() * (y == 0).sum()),
                                                rel=1e-12)


def test_roccomp_matches_the_delong_formulas(cases):
    y = cases["y"].to_numpy()
    names = ["s1", "s2", "s3"]
    kernels = [pairwise_kernel(cases[name].to_numpy(), y) for name in names]
    n1, n0 = kernels[0].shape
    theta = np.array([kernel.mean() for kernel in kernels])
    v10 = np.stack([kernel.mean(1) for kernel in kernels], axis=1)
    v01 = np.stack([kernel.mean(0) for kernel in kernels], axis=1)
    covariance = np.cov(v10.T) / n1 + np.cov(v01.T) / n0
    contrast = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    difference = contrast @ theta
    chi2 = difference @ np.linalg.solve(contrast @ covariance @ contrast.T, difference)
    result = oe.roccomp(cases, "y", names)
    assert list(result) == ["auc", "tests", "pairwise"]
    assert result.attrs["statistic"] == pytest.approx(chi2, rel=1e-10)
    assert result.attrs["df"] == 2
    assert result.attrs["p_value"] == pytest.approx(stats.chi2.sf(chi2, 2), rel=1e-8)
    np.testing.assert_allclose(result["auc"]["auc"], theta, rtol=1e-12)
    np.testing.assert_allclose(result["auc"]["std_error"], np.sqrt(np.diag(covariance)),
                               rtol=1e-10)
    np.testing.assert_allclose(np.array(result.attrs["covariance"]), covariance, rtol=1e-9,
                               atol=1e-15)
    pair = result["pairwise"].loc["s1 - s2"]
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    assert pair["z"] == pytest.approx((theta[0] - theta[1]) / math.sqrt(variance), rel=1e-10)
    # Two curves: the chi2 is the squared pairwise z; the single-curve SE equals roc's.
    two = oe.roccomp(cases, "y", ["s1", "s2"])
    assert two.attrs["statistic"] == pytest.approx(pair["z"] ** 2, rel=1e-10)
    assert two.attrs["df"] == 1
    assert two["auc"].loc["s1", "std_error"] == pytest.approx(
        oe.roc(cases, "y", "s1").attrs["std_error"], rel=1e-12)


def test_roccomp_listwise_deletion_and_redundant_scores(cases):
    frame = cases.copy()
    frame.loc[[0, 5], "s2"] = np.nan
    result = oe.roccomp(frame, "y", ["s1", "s2"])
    assert result.attrs["n"] == len(cases) - 2 and result.attrs["n_dropped"] == 2
    complete = frame.dropna()
    assert result["auc"].loc["s1", "auc"] == pytest.approx(
        oe.roc(complete, "y", "s1").attrs["auc"])
    # A monotone transform of a score has the same curve: its contrast is dropped (df 1).
    frame = cases.assign(s4=2 * cases["s1"] + 1)
    redundant = oe.roccomp(frame, "y", ["s1", "s4", "s2"])
    assert redundant.attrs["df"] == 1
    assert redundant.attrs["statistic"] == pytest.approx(
        oe.roccomp(cases, "y", ["s1", "s2"]).attrs["statistic"], rel=1e-9)
    with pytest.raises(AnalysisError) as error:
        oe.roccomp(frame, "y", ["s1", "s4"])
    assert error.value.code == "no_variation"


def test_error_codes(cases):
    calls = [
        (lambda: oe.roc(cases, "y", "nope"), "missing_columns"),
        (lambda: oe.roc(cases, "s1", "s2"), "not_binary"),
        (lambda: oe.roc({"y": [1, 1, 1], "m": [1.0, 2.0, 3.0]}, "y", "m"), "invalid_groups"),
        (lambda: oe.roc(cases, "y", "s1", positive=9), "invalid_option"),
        (lambda: oe.roc(cases, "y", "s1", alpha=2), "invalid_option"),
        (lambda: oe.roc(cases.assign(t="a"), "y", "t"), "non_numeric_column"),
        (lambda: oe.roc(cases, "y", ["s1"]), "invalid_spec"),
        (lambda: oe.roccomp(cases, "y", "s1"), "invalid_spec"),
        (lambda: oe.roccomp(cases, "y", ["s1"]), "invalid_spec"),
        (lambda: oe.roccomp(cases, "y", ["s1", "y"]), "invalid_spec"),
        (lambda: oe.roccomp({"y": [0, 1, 0], "a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]}, "y",
                            ["a", "b"]), "too_few_observations"),
    ]
    for call, code in calls:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, error.value.code, str(error.value))


def test_results_render_and_have_json_safe_attrs(cases):
    for result in (oe.roc(cases, "y", "s1"), oe.roccomp(cases, "y", ["s1", "s2", "s3"])):
        attrs = json.loads(json.dumps(result.attrs))
        assert attrs == result.attrs
        assert "[auc]" in str(result) and "\\begin{tabular}" in result.to_latex()
