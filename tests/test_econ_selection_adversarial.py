"""Edge cases of the selection family: awkward dtypes, long paths, extreme samples."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def _mixed(seed: int = 0, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.integers(0, 5, n), "flag": rng.random(n) > 0.5,
        "ni": pd.array(rng.integers(0, 9, n), dtype="Int64"),
        "nf": pd.array(rng.normal(size=n), dtype="Float64"),
        "g": pd.Categorical(rng.choice(["u", "v", "w"], n), categories=["w", "u", "v", "zz"]),
        "one": "same", "w": rng.uniform(1, 2, n)})
    frame["y"] = 0.8 * frame["a"] + 0.5 * frame["flag"] + 0.2 * frame["ni"].astype(float) \
        + rng.normal(size=n)
    frame.loc[3, "ni"] = pd.NA
    frame.loc[4, "nf"] = pd.NA
    frame.loc[5, "g"] = np.nan
    return frame


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    """The same data as ordinary float columns and hand-made indicators."""
    plain = frame.dropna().reset_index(drop=True)
    out = pd.DataFrame({name: plain[name].astype(float)
                        for name in ("y", "a", "b", "flag", "ni", "nf", "w")})
    out["g_u"] = (plain["g"] == "u").astype(float)
    out["g_v"] = (plain["g"] == "v").astype(float)
    return out


def test_stepwise_handles_nullable_boolean_and_categorical_columns():
    frame = _mixed()
    result = oe.stepwise(frame, "y", ["a", "b", "flag", "ni", "nf", "g", "one"],
                         categorical=["g", "one"], method="backward", p_remove=0.3)
    assert result.attrs["n"] == 77 and result.attrs["n_missing"] == 3
    assert any("do not vary" in note and "one" in note for note in result.attrs["notes"])
    plain = _plain(frame)
    selected = result.attrs["selected"]
    columns = [name for term in selected for name in (["g_u", "g_v"] if term == "g" else [term])]
    model = sm.OLS(plain["y"], sm.add_constant(plain[columns])).fit()
    np.testing.assert_allclose(result["coefficients"]["b"], model.params, rtol=1e-9)
    np.testing.assert_allclose(result["coefficients"]["std_error"], model.bse, rtol=1e-9)
    # The categorical reference is the first category; the unused category creates no column.
    full = oe.stepwise(frame, "y", ["a", "g"], categorical=["g"], method="backward",
                       p_remove=0.999999)
    assert list(full["coefficients"].index) == ["Intercept", "a", "g[u]", "g[v]"]
    assert math.isnan(result["excluded"].loc["one", "tolerance"]) \
        if "one" in result["excluded"].index else True


def test_long_paths_stay_accurate_after_many_sweeps():
    rng = np.random.default_rng(12)
    n, p = 400, 60
    x = rng.normal(size=(n, p))
    x[:, 1:] += 0.5 * x[:, :-1]                      # correlated neighbours
    y = x[:, :6] @ np.array([1.0, -0.8, 0.6, 0.5, -0.4, 0.3]) + rng.normal(size=n)
    names = [f"v{i}" for i in range(p)]
    frame = pd.DataFrame(x, columns=names).assign(y=y)
    result = oe.stepwise(frame, "y", names, method="backward", p_remove=0.01)
    steps = result["steps"]
    assert len(steps) > 40                           # more than one refresh of the swept matrix
    selected = result.attrs["selected"]
    model = sm.OLS(y, sm.add_constant(frame[selected])).fit()
    np.testing.assert_allclose(result["coefficients"]["b"], model.params, rtol=1e-9)
    assert steps.iloc[-1]["r_squared"] == pytest.approx(model.rsquared, rel=1e-10)
    # Replay the path: every removed term had the largest p-value in the model before it left.
    inside = list(names)
    for _, row in steps.iloc[1:].iterrows():
        current = sm.OLS(y, sm.add_constant(frame[inside])).fit()
        worst = current.pvalues.iloc[1:].idxmax()
        assert row["term"] == worst
        assert row["p_value"] == pytest.approx(current.pvalues[worst], rel=1e-6)
        inside.remove(worst)
    assert set(inside) == set(selected)
    remaining = sm.OLS(y, sm.add_constant(frame[inside])).fit()
    assert remaining.pvalues.iloc[1:].max() <= 0.01


def test_information_criteria_search_with_many_candidates():
    rng = np.random.default_rng(21)
    n, p = 250, 25
    x = rng.normal(size=(n, p))
    y = x[:, [0, 3, 7, 11]] @ np.array([0.6, -0.5, 0.4, 0.3]) + rng.normal(size=n)
    names = [f"v{i}" for i in range(p)]
    frame = pd.DataFrame(x, columns=names).assign(y=y)
    for criterion in ("aic", "bic"):
        result = oe.stepwise(frame, "y", names, criterion=criterion)
        selected = result.attrs["selected"]
        best = getattr(sm.OLS(y, sm.add_constant(frame[selected])).fit(), criterion)
        assert result.attrs[criterion] == pytest.approx(best, rel=1e-10)
        # No single entry or removal improves the criterion: a local optimum.
        for name in names:
            trial = [s for s in selected if s != name] if name in selected \
                else [*selected, name]
            design = sm.add_constant(frame[trial]) if trial else np.ones((n, 1))
            assert getattr(sm.OLS(y, design).fit(), criterion) >= best - 1e-9
        values = result["steps"][criterion].to_numpy()
        assert (np.diff(values) < 0).all()            # every step lowers the criterion
    assert len(oe.stepwise(frame, "y", names, criterion="bic").attrs["selected"]) <= \
        len(oe.stepwise(frame, "y", names, criterion="aic").attrs["selected"])


def test_underflowing_p_values_are_ranked_by_their_f_statistic():
    rng = np.random.default_rng(5)
    n = 20000
    strong, stronger = rng.normal(size=(2, n))
    y = 3 * strong + 6 * stronger + 0.01 * rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "strong": strong, "stronger": stronger})
    result = oe.stepwise(frame, "y", ["strong", "stronger"], method="forward")
    assert result.attrs["selected"] == ["stronger", "strong"]
    assert (result["steps"]["p_value"] == 0.0).all()
    assert result.attrs["r_squared"] < 1.0 and result.attrs["rmse"] == pytest.approx(0.01,
                                                                                    rel=0.05)


def test_more_candidates_than_observations():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.normal(size=(12, 30)), columns=[f"v{i}" for i in range(30)])
    frame["y"] = frame["v4"] * 2 + 0.1 * rng.normal(size=12)
    names = [f"v{i}" for i in range(30)]
    result = oe.stepwise(frame, "y", names, method="forward", p_enter=0.5)
    assert result.attrs["selected"][0] == "v4" and result.attrs["df2"] >= 1
    assert len(result["excluded"]) == 30 - len(result.attrs["selected"])
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(frame, "y", names, method="backward")
    assert error.value.code == "insufficient_observations"


def test_tabstat_many_groups_match_pandas():
    rng = np.random.default_rng(17)
    n = 6000
    frame = pd.DataFrame({"x": rng.normal(size=n).round(2), "g": rng.integers(0, 700, n),
                          "w": rng.uniform(0.2, 3.0, n)})
    table = oe.tabstat(frame, ["x"], by="g", total=False,
                       stats=["n", "mean", "sd", "min", "max", "median", "p25", "skewness"])
    reference = frame.groupby("g")["x"].agg(["count", "mean", "std", "min", "max"])
    assert list(table["g"]) == list(reference.index)
    np.testing.assert_allclose(table[["n", "mean", "min", "max"]].to_numpy(dtype=float),
                               reference[["count", "mean", "min", "max"]].to_numpy(), rtol=1e-12,
                               atol=1e-12)
    both = reference["count"].to_numpy() > 1
    np.testing.assert_allclose(table["std_dev"].to_numpy()[both],
                               reference["std"].to_numpy()[both], rtol=1e-10)
    assert table["std_dev"].isna().to_numpy()[~both].all()
    medians = frame.groupby("g")["x"].apply(
        lambda v: np.percentile(v, 50, method="averaged_inverted_cdf"))
    quartiles = frame.groupby("g")["x"].apply(
        lambda v: np.percentile(v, 25, method="averaged_inverted_cdf"))
    np.testing.assert_allclose(table["median"], medians, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(table["p25"], quartiles, rtol=1e-12, atol=1e-12)
    # Weighted group means and medians, group by group.
    weighted = oe.tabstat(frame, ["x"], by="g", total=False, weights="w",
                          stats=["mean", "median"]).set_index("g")
    for label in (0, 123, 699):
        group = frame[frame["g"] == label].sort_values("x", kind="stable")
        assert weighted.loc[label, "mean"] == pytest.approx(
            np.average(group["x"], weights=group["w"]), rel=1e-12, abs=1e-12)
        cumulative = group["w"].cumsum().to_numpy()
        first = int(np.argmax(cumulative > cumulative[-1] / 2))
        assert weighted.loc[label, "median"] == pytest.approx(group["x"].to_numpy()[first])


def test_collin_with_awkward_dtypes_and_scales():
    frame = _mixed()
    result = oe.collin(frame, ["a", "ni", "nf", "flag", "b"])
    plain = _plain(frame.drop(columns=["g"]).assign(g="u"))
    design = sm.add_constant(plain[["a", "ni", "nf", "flag", "b"]]).to_numpy()
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    expected = [variance_inflation_factor(design, i) for i in range(1, 6)]
    np.testing.assert_allclose(result["vif"]["vif"], expected, rtol=1e-9)
    assert result.attrs["n"] == len(plain)
    # Wildly different units do not matter: columns are equilibrated.
    scaled = frame.assign(a=frame["a"] * 1e12, b=frame["b"] * 1e-9)
    again = oe.collin(scaled, ["a", "ni", "nf", "flag", "b"])
    np.testing.assert_allclose(again["vif"]["vif"], result["vif"]["vif"], rtol=1e-8)
    np.testing.assert_allclose(again["condition"]["condition_index"],
                               result["condition"]["condition_index"], rtol=1e-8)


def test_curvefit_extreme_scales_and_integer_columns():
    x = np.arange(1, 41)
    frame = pd.DataFrame({"x": x, "y": 3e8 * x.astype(float) ** 1.7})
    frame["y"] *= np.exp(0.01 * np.sin(x))
    result = oe.curvefit(frame, "y", "x", models=["power", "linear", "cubic"])
    row = result["summary"].loc["power"]
    assert row["b1"] == pytest.approx(1.7, abs=0.01) and row["b0"] == pytest.approx(3e8, rel=0.05)
    cubic = np.polyfit(x, frame["y"], 3)[::-1]
    np.testing.assert_allclose(result["summary"].loc["cubic", ["b0", "b1", "b2", "b3"]]
                               .to_numpy(dtype=float), cubic, rtol=1e-6)
    # Results do not depend on the unit of measurement ...
    for factor in (1e-90, 1e90):
        moved = oe.curvefit(frame.assign(y=frame["y"] * factor), "y", "x",
                            models=["power", "linear"])
        assert moved["summary"].loc["power", "b1"] == pytest.approx(row["b1"], rel=1e-9)
        assert moved["summary"].loc["linear", "r_squared"] == pytest.approx(
            result["summary"].loc["linear", "r_squared"], rel=1e-9)
    # ... and a column whose squares would underflow is refused, never mis-reported.
    with pytest.raises(AnalysisError) as error:
        oe.curvefit(frame.assign(y=frame["y"] * 1e-300), "y", "x")
    assert error.value.code == "non_finite_values" and "rescale" in str(error.value)


def test_columns_too_small_for_sums_of_squares_are_refused():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(50, 3)), columns=["y", "a", "b"])
    tiny = frame.assign(a=frame["a"] * 1e-200)
    for call in (lambda: oe.stepwise(tiny, "y", ["a", "b"]),
                 lambda: oe.collin(tiny, ["a", "b"]),
                 lambda: oe.tabstat(tiny, ["a"], stats=["sd"])):
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == "non_finite_values" and "'a'" in str(error.value)
    # Small but representable units are fine and leave the tests unchanged.
    small = frame.assign(a=frame["a"] * 1e-60, y=frame["y"] * 1e40)
    base = oe.stepwise(frame, "y", ["a", "b"], method="backward", p_remove=0.999999)
    moved = oe.stepwise(small, "y", ["a", "b"], method="backward", p_remove=0.999999)
    np.testing.assert_allclose(moved["coefficients"]["t"], base["coefficients"]["t"], rtol=1e-9)
    np.testing.assert_allclose(oe.collin(small, ["a", "b"])["vif"]["vif"],
                               oe.collin(frame, ["a", "b"])["vif"]["vif"], rtol=1e-10)
    assert oe.tabstat(small, ["a"], stats=["sd"]).loc["a", "std_dev"] == pytest.approx(
        frame["a"].std() * 1e-60, rel=1e-12)


def test_block_tolerances_match_brute_force_vif():
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    rng = np.random.default_rng(31)
    n = 150
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    frame["g"] = np.where(frame["a"] + 0.5 * rng.normal(size=n) > 0.3, "hi",
                          np.where(frame["a"] < -0.6, "lo", "mid"))
    frame["y"] = frame["a"] + 0.5 * frame["b"] + rng.normal(size=n)
    result = oe.stepwise(frame, "y", ["g"], forced=["a", "b"], categorical=["g"],
                         method="forward", p_enter=1e-12)
    assert result.attrs["selected"] == ["a", "b"] and list(result["excluded"].index) == ["g"]
    dummies = pd.get_dummies(frame["g"], dtype=float).iloc[:, 1:]
    design = sm.add_constant(pd.concat([frame[["a", "b"]], dummies], axis=1)).to_numpy()
    tolerances = [1 / variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
    row = result["excluded"].loc["g"]
    assert row["tolerance"] == pytest.approx(min(tolerances[2:]), rel=1e-9)
    assert row["min_tolerance"] == pytest.approx(min(tolerances), rel=1e-9)
    larger = sm.OLS(frame["y"], design).fit()
    smaller = sm.OLS(frame["y"], design[:, :3]).fit()
    partial = (smaller.ssr - larger.ssr) / smaller.ssr
    assert row["partial_correlation"] == pytest.approx(math.sqrt(partial), rel=1e-9)
    # A stricter tolerance than the block can meet keeps it out of a backward start.
    strict = oe.stepwise(frame, "y", ["a", "b", "g"], categorical=["g"], method="backward",
                         tolerance=min(tolerances) * 1.01, p_remove=0.999999)
    assert "g" not in strict.attrs["selected"]
    assert any("'g' was not entered" in note for note in strict.attrs["notes"])


def test_curvefit_unrepresentable_back_transformations_are_missing_with_a_note():
    """Regression: exp(intercept) / exp(slope) that overflow or underflow used to be
    reported silently as missing (or as 0 after underflow)."""
    rng = np.random.default_rng(41)
    x = rng.uniform(1.0, 1.0001, 20) * 1e-6
    for sign in (1.0, -1.0):
        y = np.exp(sign * 3e9 * (x - 1e-6)) * rng.uniform(0.9, 1.1, 20)
        result = oe.curvefit(pd.DataFrame({"x": x, "y": y}), "y", "x",
                             models=["compound", "growth"])
        summary = result["summary"]
        assert np.isnan(summary.loc["compound", "b1"])           # exp(+-3e9) is not a number
        assert np.isfinite(summary.loc["growth", "b1"])          # the same fit, unexponentiated
        assert sign * summary.loc["growth", "b1"] > 1e9          # ln(b1) of compound
        assert summary.loc["compound", "r_squared"] == pytest.approx(
            summary.loc["growth", "r_squared"], rel=1e-12)
        assert any(note.startswith("compound: b0, b1 are not representable")
                   for note in result.attrs["notes"])


def test_stepwise_refuses_candidate_lists_too_large_for_memory():
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    names = [f"c{i}" for i in range(4000)]
    with pytest.raises(oe.AnalysisError) as error:
        oe.stepwise(frame, "y", names)
    assert error.value.code == "design_too_large"
