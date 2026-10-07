"""Independent acceptance of MARKET-100; no source implementation is shared."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.linalg import helmert
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from test_econ_streaming_helpers import assert_tables, replay
from test_econ_stats_anova import _multivariate, _repeated


@pytest.mark.parametrize("method", ["spearman", "kendall"])
@pytest.mark.parametrize("pairwise", [True, False])
@pytest.mark.parametrize("rows", [1, 17, 1000])
def test_rank_ties_missing_oracle(method, pairwise, rows):
    rng = np.random.default_rng(1907)
    frame = pd.DataFrame(rng.integers(-4, 6, size=(137, 3)), columns=["x", "y", "z"], dtype=float)
    frame.loc[[0, 2, 9], "x"] = np.nan
    frame.loc[[1, 2, 7], "y"] = np.nan
    frame.loc[[0, 8], "z"] = np.nan
    result = oe.correlate(
        replay(frame, rows), list(frame), method=method, pairwise=pairwise, ci=True
    )
    assert_tables(
        result, oe.correlate(frame, list(frame), method=method, pairwise=pairwise, ci=True)
    )
    for a, b in itertools.combinations(frame, 2):
        selected = frame[[a, b]].dropna() if pairwise else frame.dropna()[[a, b]]
        oracle = (
            stats.spearmanr(selected[a], selected[b])
            if method == "spearman"
            else stats.kendalltau(selected[a], selected[b], method="asymptotic")
        )
        assert result["coefficients"].loc[a, b] == pytest.approx(oracle.statistic, abs=2e-13)
        assert result["p_values"].loc[a, b] == pytest.approx(oracle.pvalue, abs=2e-12)
        assert result["n"].loc[a, b] == len(selected)


def quantile(values, percent, method):
    ordered = sorted(values)
    n = len(ordered)
    if method == "stata":
        p = n * percent / 100
        k = round(p)
        if abs(p - k) <= 1e-9 * n:
            return (ordered[min(n - 1, max(0, k - 1))] + ordered[min(n - 1, max(0, k))]) / 2
        return ordered[min(n - 1, max(0, math.floor(p)))]
    p = (n + 1) * percent / 100
    k = math.floor(p)
    if k < 1:
        return ordered[0]
    if k >= n:
        return ordered[-1]
    return ordered[k - 1] + (p - k) * (ordered[k] - ordered[k - 1])


def test_requested_quantile_table_is_budgeted_before_result_allocation(tmp_path, monkeypatch):
    from openecon.resources import use_workspace_budget
    from openecon.econometrics.stats import streaming_describe

    frame = pd.DataFrame({"x": np.arange(64.), "z": np.arange(64.)**.5,
                          "g": np.arange(64) % 8})
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    def no_quantile(*args):
        pytest.fail("An oversized result must fail before evaluating its percentiles")
    monkeypatch.setattr(streaming_describe, "_quantile", no_quantile)
    with use_workspace_budget(32), pytest.raises(AnalysisError) as error:
        oe.describe(replay(frame, 7), ["x", "z"], by="g",
                    stats=[f"p{i/100}" for i in range(1, 2049)])
    assert error.value.code == "workspace_limit"
    assert error.value.resource_plan["buffers"]["requested_result_geometry"] > 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("method", ["stata", "haverage"])
@pytest.mark.parametrize("listwise", [True, False])
@pytest.mark.parametrize("rows", [1, 13])
def test_fractional_group_quantiles_and_moments(method, listwise, rows):
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(
        {"x": rng.normal(size=103), "z": rng.normal(size=103), "g": np.arange(103) % 3}
    )
    frame.loc[[1, 4, 18], "x"] = np.nan
    frame.loc[[2, 3, 18], "z"] = np.nan
    frame["g"] = pd.Categorical(frame.g, categories=[2, 0, 1, 7], ordered=True)
    options = dict(
        by="g",
        stats=[
            "n",
            "missing",
            "mean",
            "variance",
            "p0.5",
            "p12.5",
            "p50",
            "p87.25",
            "p99.5",
            "iqr",
            "skewness",
            "kurtosis",
            "ci",
        ],
        percentile_method=method,
        listwise=listwise,
    )
    result = oe.describe(replay(frame, rows), ["x", "z"], **options)
    pd.testing.assert_frame_equal(
        result, oe.describe(frame, ["x", "z"], **options), check_exact=False, rtol=2e-10, atol=2e-10
    )
    for _, row in result.iterrows():
        selected = frame.dropna(subset=["x", "z"]) if listwise else frame
        values = selected.loc[selected.g == row.g, row.variable].dropna().tolist()
        for percent in [0.5, 12.5, 50, 87.25, 99.5]:
            assert row[f"p{percent:g}"] == pytest.approx(
                quantile(values, percent, method), abs=1e-12
            )


@pytest.mark.parametrize("rows", [1, 7, 10000])
def test_repeated_anova_independent_f_and_sphericity(rows):
    frame = _repeated(subjects=18)
    result = oe.rm_anova(replay(frame, rows), "y", "id", ["A", "B"])
    assert_tables(result, oe.rm_anova(frame, "y", "id", ["A", "B"]), atol=1e-8)
    oracle = AnovaRM(frame, "y", "id", within=["A", "B"]).fit().anova_table
    for ours, theirs in [("A", "A"), ("B", "B"), ("A#B", "A:B")]:
        row = (
            result["within"]
            .loc[
                (result["within"].source == ours)
                & (result["within"].correction == "sphericity_assumed")
            ]
            .iloc[0]
        )
        assert row.statistic == pytest.approx(oracle.loc[theirs, "F Value"], rel=1e-10)
    # GG epsilon is independently computed from Helmert time contrasts.
    means = frame.groupby(["id", "A"]).y.mean().unstack().to_numpy()
    contrast = means @ helmert(means.shape[1], full=False).T
    covariance = np.cov(contrast, rowvar=False)
    epsilon = np.trace(covariance) ** 2 / ((means.shape[1] - 1) * np.trace(covariance @ covariance))
    row = result["sphericity"].loc["A"]
    assert row.epsilon_gg == pytest.approx(epsilon, abs=1e-10)


@pytest.mark.parametrize("rows", [1, 11, 10000])
def test_manova_independent_statsmodels_and_box_m(rows):
    frame = _multivariate(n=170)
    names = ["y1", "y2", "y3"]
    result = oe.manova(replay(frame, rows), names, ["a", "b"], covariates=["x"])
    assert_tables(result, oe.manova(frame, names, ["a", "b"], covariates=["x"]), atol=2e-8)
    oracle = MANOVA.from_formula("y1+y2+y3 ~ x + C(a, Sum)*C(b, Sum)", frame).mv_test()
    mapping = {"x": "x", "a": "C(a, Sum)", "b": "C(b, Sum)", "a#b": "C(a, Sum):C(b, Sum)"}
    for ours, theirs in mapping.items():
        rows_ = result["multivariate"].loc[result["multivariate"].effect == ours]
        reference = oracle.results[theirs]["stat"]
        for test, label in {
            "wilks": "Wilks' lambda",
            "pillai": "Pillai's trace",
            "hotelling": "Hotelling-Lawley trace",
            "roy": "Roy's greatest root",
        }.items():
            row = rows_.set_index("test").loc[test]
            assert row.value == pytest.approx(reference.loc[label, "Value"], rel=2e-9)
            if test != "hotelling":
                assert row.statistic == pytest.approx(reference.loc[label, "F Value"], rel=2e-9)
    blocks = [block[names].to_numpy() for _, block in frame.groupby(["a", "b"])]
    ns = np.array([len(block) for block in blocks])
    p = 3
    g = len(ns)
    df = ns - 1
    covs = [np.cov(block, rowvar=False) for block in blocks]
    pooled = sum(n * cov for n, cov in zip(df, covs)) / sum(df)
    m = sum(df) * np.linalg.slogdet(pooled)[1] - sum(
        n * np.linalg.slogdet(cov)[1] for n, cov in zip(df, covs)
    )
    correction = (2 * p * p + 3 * p - 1) / (6 * (p + 1) * (g - 1)) * (sum(1 / df) - 1 / sum(df))
    assert result["box_m"].iloc[0].chi2 == pytest.approx((1 - correction) * m, rel=1e-9)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "between"])
def test_subject_cells_fail_closed(mutation):
    frame = _repeated(subjects=18)
    frame["group"] = np.where(frame.id < 9, "a", "b")
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    if mutation == "missing":
        frame = frame.iloc[1:]
    if mutation == "between":
        frame.loc[0, "group"] = "b"
    with pytest.raises(AnalysisError):
        oe.rm_anova(replay(frame), "y", "id", ["A", "B"], between=["group"])


@pytest.mark.parametrize("procedure", ["rank", "describe", "repeated", "manova"])
def test_mutation_and_owned_cleanup(procedure, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = _repeated() if procedure == "repeated" else _multivariate()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls > 1:
            changed.iloc[0, changed.columns.get_loc("y" if procedure == "repeated" else "y1")] += (
                0.5
            )
        yield from (changed.iloc[i : i + 7] for i in range(0, len(changed), 7))

    source = Dataset.from_batches(factory, list(frame), row_count=len(frame))
    with pytest.raises(AnalysisError, match="changed"):
        if procedure == "rank":
            oe.correlate(source, ["y1", "y2"], method="kendall")
        elif procedure == "describe":
            oe.describe(source, ["y1", "y2"])
        elif procedure == "repeated":
            oe.rm_anova(source, "y", "id", ["A", "B"])
        else:
            oe.manova(source, ["y1", "y2"], ["a"])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("procedure", ["rank", "describe", "repeated", "manova"])
def test_geometry_budget_is_admitted_before_large_buffers(procedure):
    from openecon.resources import use_workspace_budget

    frame = _repeated() if procedure == "repeated" else _multivariate()
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        if procedure == "rank":
            oe.correlate(replay(frame), ["y1", "y2"], method="kendall")
        elif procedure == "describe":
            oe.describe(replay(frame), ["y1", "y2"])
        elif procedure == "repeated":
            oe.rm_anova(replay(frame), "y", "id", ["A", "B"])
        else:
            oe.manova(replay(frame), ["y1", "y2"], ["a"])


def test_stop_removes_owned_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = _multivariate()

    def factory():
        yield frame.iloc[:7]
        raise KeyboardInterrupt()

    source = Dataset.from_batches(factory, list(frame), row_count=len(frame))
    with pytest.raises(KeyboardInterrupt):
        oe.correlate(source, ["y1", "y2"], method="spearman")
    assert not list(tmp_path.iterdir())
