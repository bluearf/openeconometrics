"""External ordering/reduction oracles, subject boundaries and owned cleanup."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


def replay(frame, rows=7):
    return oe.Dataset.from_batches(
        lambda: (frame.iloc[i : i + rows] for i in range(0, len(frame), rows)),
        list(frame),
        row_count=len(frame),
    )


@pytest.mark.parametrize("method", ["spearman", "kendall"])
@pytest.mark.parametrize("pairwise", [True, False])
@pytest.mark.parametrize("rows", [1, 7, 1000])
def test_exact_ties_missing_and_per_pair_ranks_match_scipy(method, pairwise, rows):
    rng = np.random.default_rng(807)
    frame = pd.DataFrame(rng.integers(-3, 8, (97, 3)), columns=list("xyz"), dtype=float)
    frame.loc[[1, 4, 12], "x"] = np.nan
    frame.loc[[5, 8, 23], "z"] = np.nan
    actual = oe.correlate(
        replay(frame, rows), list(frame), method=method, pairwise=pairwise, ci=True
    )
    expected = oe.correlate(frame, list(frame), method=method, pairwise=pairwise, ci=True)
    for key in expected:
        pd.testing.assert_frame_equal(
            actual[key], expected[key], check_exact=False, rtol=1e-10, atol=1e-10
        )
    base = frame if pairwise else frame.dropna()
    for i, x in enumerate(frame):
        for y in list(frame)[i + 1 :]:
            pair = base[[x, y]].dropna()
            oracle = (
                stats.spearmanr(pair[x], pair[y])
                if method == "spearman"
                else stats.kendalltau(pair[x], pair[y], method="asymptotic")
            )
            assert actual["coefficients"].loc[x, y] == pytest.approx(oracle.statistic, abs=1e-12)
            assert actual["p_values"].loc[x, y] == pytest.approx(oracle.pvalue, abs=1e-10)
    assert actual.attrs["full_source_collected"] is False


@pytest.mark.parametrize("method", ["stata", "haverage"])
@pytest.mark.parametrize("listwise", [True, False])
def test_grouped_fractional_quantiles_moments_category_order(method, listwise):
    rng = np.random.default_rng(992)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=101) + 1e8,
            "z": rng.normal(size=101),
            "g": pd.Categorical(
                np.tile(["a", "b"], 51)[:101], categories=["b", "a", "unused"], ordered=True
            ),
        }
    )
    frame.loc[[3, 11], "x"] = np.nan
    frame.loc[[2, 24], "z"] = np.nan
    kwargs = dict(
        by="g",
        stats=[
            "n",
            "missing",
            "mean",
            "sd",
            "p2.5",
            "p50",
            "p97.5",
            "skewness",
            "kurtosis",
            "iqr",
            "ci",
        ],
        listwise=listwise,
        percentile_method=method,
    )
    actual = oe.describe(replay(frame, 3), ["x", "z"], **kwargs)
    expected = oe.describe(frame, ["x", "z"], **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-6, atol=1e-6)
    for _, row in actual.iterrows():
        sample = frame.dropna(subset=["x", "z"]) if listwise else frame
        values = sample.loc[sample.g == row.g, row.variable].dropna().to_numpy()
        # NumPy's documented definitions 5 (averaged inverted CDF) and 6 (Weibull).
        for percentile in [2.5, 50, 97.5]:
            assert row[f"p{percentile:g}"] == pytest.approx(
                np.percentile(
                    values,
                    percentile,
                    method="averaged_inverted_cdf" if method == "stata" else "weibull",
                ),
                abs=2e-8,
            )


def test_manova_independent_numpy_residual_sscp_and_multivariate_oracle():
    from statsmodels.multivariate.manova import MANOVA

    rng = np.random.default_rng(513)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=103),
            "b": rng.normal(size=103),
            "g": np.tile(["a", "b"], 52)[:103],
            "x": rng.normal(size=103),
        }
    )
    result = oe.manova(replay(frame, 5), ["a", "b"], ["g"], covariates=["x"])
    expected = oe.manova(frame, ["a", "b"], ["g"], covariates=["x"])
    for key in expected:
        pd.testing.assert_frame_equal(
            result[key], expected[key], check_exact=False, rtol=1e-9, atol=1e-9
        )
    design = np.column_stack([np.ones(len(frame)), np.where(frame.g == "a", 1.0, -1.0), frame.x])
    values = frame[["a", "b"]].to_numpy()
    resid = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    for i, name in enumerate(["a", "b"]):
        row = result["univariate"].query("outcome==@name and source=='error'").iloc[0]
        assert row.ss == pytest.approx(float(resid[:, i] @ resid[:, i]), rel=1e-12)
    oracle = (
        MANOVA.from_formula("a + b ~ C(g, Sum) + x", frame).mv_test().results["C(g, Sum)"]["stat"]
    )
    oracle = oracle.iloc[[1, 0, 2, 3]]
    rows = result["multivariate"].query("effect=='g'")
    assert rows.value.to_numpy() == pytest.approx(oracle["Value"].to_numpy(), rel=1e-10)
    assert rows.statistic.to_numpy() == pytest.approx(oracle["F Value"].to_numpy(), rel=1e-10)


@pytest.mark.parametrize("between", [None, ["g"]])
def test_repeated_complete_subjects_across_blocks_and_shuffled_order(between):
    rng = np.random.default_rng(611)
    frame = pd.DataFrame(
        {
            "id": np.repeat(np.arange(53), 6),
            "a": np.tile(np.repeat([1, 2], 3), 53),
            "b": np.tile([1, 2, 3], 106),
            "g": np.repeat(np.arange(53) % 2, 6),
            "y": rng.normal(size=318),
        }
    )
    expected = oe.rm_anova(frame, "y", "id", ["a", "b"], between=between)
    for rows in [1, 5, 1000]:
        result = oe.rm_anova(
            replay(frame.sample(frac=1, random_state=7), rows),
            "y",
            "id",
            ["a", "b"],
            between=between,
        )
        for key in expected:
            pd.testing.assert_frame_equal(
                result[key], expected[key], check_exact=False, rtol=1e-9, atol=1e-9
            )
    assert result.attrs["subject_matrix_collected"] is False
    if between is None:
        from statsmodels.stats.anova import AnovaRM

        oracle = AnovaRM(frame, "y", "id", within=["a", "b"]).fit().anova_table
        rows = result["within"].loc[
            (result["within"].correction == "sphericity_assumed")
            & result["within"].source.isin(["a", "b", "a#b"])
        ]
        assert len(rows) == len(oracle)
        for _, row in rows.iterrows():
            reference = oracle.loc[row.source.replace("#", ":")]
            assert row.statistic == pytest.approx(reference["F Value"], rel=1e-10)


@pytest.mark.parametrize("defect", ["duplicate", "incomplete", "between"])
def test_repeated_defects_checked_on_disk_and_cleanup(tmp_path, monkeypatch, defect):
    frame = pd.DataFrame(
        {
            "id": np.repeat(np.arange(10), 3),
            "t": np.tile([1, 2, 3], 10),
            "g": np.repeat(np.arange(10) % 2, 3),
            "y": np.arange(30, dtype=float) ** 0.3,
        }
    )
    if defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    if defect == "incomplete":
        frame = frame.iloc[1:]
    if defect == "between":
        frame.loc[0, "g"] = 1
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as error:
        oe.rm_anova(replay(frame, 2), "y", "id", ["t"], between=["g"])
    assert error.value.code == ("invalid_design" if defect == "between" else "unbalanced_design")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("procedure", ["rank", "describe", "rm", "manova"])
def test_source_changed_and_scratch_cleanup(procedure, tmp_path, monkeypatch):
    rng = np.random.default_rng(419)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=30),
            "y": rng.normal(size=30),
            "id": np.repeat(np.arange(10), 3),
            "t": np.tile([1, 2, 3], 10),
            "g": np.repeat(np.arange(10) % 2, 3),
        }
    )
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        part = frame.copy()
        if calls > 1:
            part.loc[0, "y"] += 1
        yield from (part.iloc[i : i + 3] for i in range(0, len(part), 3))

    source = oe.Dataset.from_batches(factory, list(frame))
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as error:
        if procedure == "rank":
            oe.correlate(source, ["x", "y"], method="kendall")
        elif procedure == "describe":
            oe.describe(source, ["x", "y"])
        elif procedure == "rm":
            oe.rm_anova(source, "y", "id", ["t"])
        else:
            oe.manova(source, ["x", "y"], ["g"])
    assert error.value.code == "source_changed"
    assert list(tmp_path.iterdir()) == []


def test_early_budget_guard():
    frame = pd.DataFrame(np.ones((3, 1000)), columns=[f"x{i}" for i in range(1000)])
    with use_workspace_budget(16), pytest.raises(AnalysisError, match="workspace"):
        oe.correlate(replay(frame), list(frame), method="spearman")
