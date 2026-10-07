"""oe.stepwise against explicit selection loops built on statsmodels OLS."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.frame import DataFrame

X = [f"x{i}" for i in range(8)]


def _data(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    x[:, 3] += 0.6 * x[:, 0]
    y = 1 + 2 * x[:, 0] - x[:, 2] + 0.25 * x[:, 5] + rng.normal(size=n)
    frame = pd.DataFrame(x, columns=X)
    frame["y"] = y
    frame["g"] = rng.choice(list("abcd"), size=n)
    frame["y"] += frame["g"].map({"a": 0.0, "b": 0.5, "c": -0.4, "d": 0.1})
    frame["w"] = rng.uniform(0.5, 3.0, size=n)
    frame["f"] = rng.integers(1, 4, size=n)
    return frame


def _suppressor(seed: int = 1, n: int = 200) -> pd.DataFrame:
    """x1 is the best single predictor but redundant once x2 and x3 are in."""
    rng = np.random.default_rng(seed)
    x2, x3, x4 = rng.normal(size=(3, n))
    x1 = x2 + x3 + 0.35 * rng.normal(size=n)
    y = x2 + x3 + 0.5 * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4})


def _design(frame: pd.DataFrame, terms: list[str], blocks: dict[str, list[str]]) -> np.ndarray:
    columns = [np.ones(len(frame))]
    for term in terms:
        columns += [frame[name].to_numpy(dtype=float) for name in blocks[term]]
    return np.column_stack(columns)


def _fit(frame, y, terms, blocks, weights=None):
    design = _design(frame, terms, blocks)
    if weights is None:
        return sm.OLS(frame[y].to_numpy(), design).fit()
    return sm.WLS(frame[y].to_numpy(), design, weights=frame[weights].to_numpy()).fit()


def _oracle(frame, y, blocks, *, method="stepwise", criterion="pvalue", pe=0.05, pr=0.10,
            forced=(), weights=None):
    """Selection by refitting every candidate model with statsmodels."""
    terms = [*forced, *(name for name in blocks if name not in forced)]
    inside = list(forced) if method != "backward" else list(terms)
    steps = []

    def value(model):
        return model.aic if criterion == "aic" else model.bic

    for _ in range(4 * len(terms) + 4):
        current = _fit(frame, y, inside, blocks, weights)
        removals, entries = [], []
        if method != "forward":
            for term in inside:
                if term in forced:
                    continue
                smaller = _fit(frame, y, [t for t in inside if t != term], blocks, weights)
                f, p, _ = current.compare_f_test(smaller)
                removals.append((term, f, p, value(smaller) - value(current), smaller))
        if method != "backward":
            for term in terms:
                if term in inside:
                    continue
                larger = _fit(frame, y, [*inside, term], blocks, weights)
                f, p, _ = larger.compare_f_test(current)
                entries.append((term, f, p, value(larger) - value(current), larger))
        if criterion == "pvalue":
            if removals:
                term, f, p, _, model = max(removals, key=lambda item: item[2])
                if p > pr:
                    inside.remove(term)
                    steps.append(("removed", term, f, p, model.rsquared))
                    continue
            if entries:
                term, f, p, _, model = min(entries, key=lambda item: item[2])
                if p < pe:
                    inside.append(term)
                    steps.append(("entered", term, f, p, model.rsquared))
                    continue
            break
        moves = [("removed", *item) for item in removals] + [("entered", *item) for item in entries]
        if not moves:
            break
        action, term, f, p, delta, model = min(moves, key=lambda item: item[4])
        if not delta < 0:
            break
        inside.remove(term) if action == "removed" else inside.append(term)
        steps.append((action, term, f, p, model.rsquared))
    return inside, steps


def _numeric_blocks(names):
    return {name: [name] for name in names}


def _check_path(result, steps):
    table = result["steps"]
    table = table[table["action"] != "start"]
    assert list(table["action"]) == [step[0] for step in steps]
    assert list(table["term"]) == [step[1] for step in steps]
    np.testing.assert_allclose(table["statistic"], [step[2] for step in steps], rtol=1e-8)
    np.testing.assert_allclose(table["p_value"], [step[3] for step in steps], rtol=1e-7,
                               atol=1e-300)
    np.testing.assert_allclose(table["r_squared"], [step[4] for step in steps], rtol=1e-10)


# ---- public surface ---------------------------------------------------------------


def test_stepwise_is_exported_and_returns_tables():
    from openecon.econometrics import registry

    assert registry.public_exports()["stepwise"][0] == "openecon.econometrics.selection.stepwise"
    assert callable(oe.stepwise) and "REGRESSION" in oe.stepwise.__doc__
    result = oe.stepwise(_data(), "y", X)
    assert isinstance(result, TableSet)
    assert list(result) == ["steps", "coefficients", "excluded", "anova"]
    assert all(isinstance(table, DataFrame) for table in result.values())


def test_stepwise_renders_and_serializes():
    result = oe.stepwise(_data(), "y", [*X, "g"], categorical=["g"])
    text = str(result)
    assert "Stepwise regression of y" in text and "[excluded]" in text
    latex = result.to_latex()
    assert latex.count(r"\begin{tabular}") == 4
    assert json.loads(json.dumps(result.attrs)) == result.attrs
    for table in result.values():
        assert set(json.loads(table.to_json())) == set(table.columns)
        assert not np.isinf(table.select_dtypes("number").to_numpy(dtype=float)).any()


def test_stepwise_accepts_mappings_and_records():
    frame = _data(n=60)[["y", "x0", "x1", "x2"]]
    expected = oe.stepwise(frame, "y", ["x0", "x1", "x2"]).attrs["r_squared"]
    assert oe.stepwise(frame.to_dict("list"), "y", ["x0", "x1", "x2"]).attrs["r_squared"] \
        == pytest.approx(expected, rel=1e-13)
    assert oe.stepwise(frame.to_dict("records"), "y", ["x0", "x1", "x2"]).attrs["r_squared"] \
        == pytest.approx(expected, rel=1e-13)


# ---- selection paths --------------------------------------------------------------


@pytest.mark.parametrize("method", ["forward", "backward", "stepwise"])
@pytest.mark.parametrize("criterion", ["pvalue", "aic", "bic"])
def test_selection_path_matches_refitting_oracle(method, criterion):
    frame = _data()
    selected, steps = _oracle(frame, "y", _numeric_blocks(X), method=method, criterion=criterion)
    result = oe.stepwise(frame, "y", X, method=method, criterion=criterion)
    if method == "backward":
        assert set(result.attrs["selected"]) == set(selected)
        assert result["steps"].iloc[0]["action"] == "start"
    else:
        assert result.attrs["selected"] == selected
    _check_path(result, steps)


def test_stepwise_removes_a_redundant_term():
    frame = _suppressor()
    names = ["x1", "x2", "x3", "x4"]
    selected, steps = _oracle(frame, "y", _numeric_blocks(names))
    assert "removed" in [step[0] for step in steps]          # the design does its job
    result = oe.stepwise(frame, "y", names)
    assert result.attrs["selected"] == selected == ["x2", "x3"]
    _check_path(result, steps)
    table = result["steps"]
    removed = table[table["action"] == "removed"].iloc[0]
    assert removed["term"] == "x1" and removed["r_squared_change"] < 0
    assert removed["p_value"] > 0.10
    # Forward selection keeps it: no removal test.
    forward = oe.stepwise(frame, "y", names, method="forward")
    assert forward.attrs["selected"][0] == "x1" and "x1" in forward.attrs["selected"]


def test_thresholds_change_the_path():
    frame = _data(seed=5, n=120)
    for pe, pr in ((0.01, 0.02), (0.2, 0.3), (0.5, 0.6)):
        selected, steps = _oracle(frame, "y", _numeric_blocks(X), pe=pe, pr=pr)
        result = oe.stepwise(frame, "y", X, p_enter=pe, p_remove=pr)
        assert result.attrs["selected"] == selected
        _check_path(result, steps)
    backward, steps = _oracle(frame, "y", _numeric_blocks(X), method="backward", pr=0.01)
    result = oe.stepwise(frame, "y", X, method="backward", p_remove=0.01)
    assert set(result.attrs["selected"]) == set(backward)
    _check_path(result, steps)


def test_forced_terms_stay_and_start_the_model():
    frame = _data()
    blocks = _numeric_blocks(X)
    for method in ("forward", "stepwise", "backward"):
        selected, steps = _oracle(frame, "y", blocks, method=method, forced=("x7", "x1"))
        result = oe.stepwise(frame, "y", X, method=method, forced=["x7", "x1"])
        assert result.attrs["selected"][:2] == ["x7", "x1"]
        assert set(result.attrs["selected"]) == set(selected)
        _check_path(result, steps)
        start = result["steps"].iloc[0]
        assert start["action"] == "start" and start["step"] == 0
    # A forced term need not be among the candidates.
    result = oe.stepwise(frame, "y", ["x0", "x2"], forced=["x7"], method="forward")
    base = _fit(frame, "y", ["x7"], blocks)
    start = result["steps"].iloc[0]
    assert start["term"] == "x7"
    assert start["statistic"] == pytest.approx(base.fvalue, rel=1e-9)
    assert start["p_value"] == pytest.approx(base.f_pvalue, rel=1e-8)
    assert start["r_squared"] == pytest.approx(base.rsquared, rel=1e-10)
    assert "x7" not in result["excluded"].index


def test_categorical_terms_enter_as_blocks():
    frame = _data()
    dummies = pd.get_dummies(frame["g"], prefix="g", dtype=float).iloc[:, 1:]
    wide = pd.concat([frame, dummies], axis=1)
    blocks = {**_numeric_blocks(X), "g": list(dummies.columns)}
    for method in ("forward", "stepwise", "backward"):
        selected, steps = _oracle(wide, "y", blocks, method=method, pe=0.1, pr=0.15)
        result = oe.stepwise(frame, "y", [*X, "g"], categorical=["g"], method=method,
                             p_enter=0.1, p_remove=0.15)
        assert set(result.attrs["selected"]) == set(selected) and "g" in selected
        _check_path(result, steps)
    result = oe.stepwise(frame, "y", [*X, "g"], categorical=["g"], method="forward",
                         p_enter=0.1)
    entered = result["steps"].set_index("term").loc["g"]
    assert entered["df1"] == 3
    names = list(result["coefficients"].index)
    assert [name for name in names if name.startswith("g[")] == ["g[b]", "g[c]", "g[d]"]
    model = _fit(wide, "y", result.attrs["selected"], blocks)
    np.testing.assert_allclose(result["coefficients"]["b"], model.params, rtol=1e-9)
    np.testing.assert_allclose(result["coefficients"]["std_error"], model.bse, rtol=1e-9)
    # A categorical term that stays out is tested with all its degrees of freedom.
    strict = oe.stepwise(frame, "y", [*X, "g"], categorical=["g"], p_enter=1e-6,
                         p_remove=1e-5)
    assert "g" in strict["excluded"].index
    row = strict["excluded"].loc["g"]
    inside = strict.attrs["selected"]
    f, p, _ = _fit(wide, "y", [*inside, "g"], blocks).compare_f_test(
        _fit(wide, "y", inside, blocks))
    assert row["statistic"] == pytest.approx(f, rel=1e-9) and row["df1"] == 3
    assert row["p_value"] == pytest.approx(p, rel=1e-8)
    assert math.isnan(row["beta_in"]) and row["partial_correlation"] >= 0


# ---- final-model tables -----------------------------------------------------------


def test_final_model_tables_match_statsmodels():
    frame = _data()
    result = oe.stepwise(frame, "y", X, alpha=0.1)
    selected = result.attrs["selected"]
    model = _fit(frame, "y", selected, _numeric_blocks(X))
    table = result["coefficients"]
    assert list(table.index) == ["Intercept", *selected]
    np.testing.assert_allclose(table["b"], model.params, rtol=1e-10)
    np.testing.assert_allclose(table["std_error"], model.bse, rtol=1e-10)
    np.testing.assert_allclose(table["t"], model.tvalues, rtol=1e-9)
    np.testing.assert_allclose(table["p_value"], model.pvalues, rtol=1e-8, atol=1e-300)
    interval = model.conf_int(alpha=0.1)
    np.testing.assert_allclose(table["ci_low"], interval[:, 0], rtol=1e-9)
    np.testing.assert_allclose(table["ci_high"], interval[:, 1], rtol=1e-9)
    # Standardized coefficients, tolerances and variance inflation factors.
    sd = frame[selected].std().to_numpy() / frame["y"].std()
    np.testing.assert_allclose(table["beta"].iloc[1:], model.params[1:] * sd, rtol=1e-10)
    design = _design(frame, selected, _numeric_blocks(X))
    vif = [variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
    np.testing.assert_allclose(table["vif"].iloc[1:], vif, rtol=1e-9)
    np.testing.assert_allclose(table["tolerance"].iloc[1:], 1 / np.array(vif), rtol=1e-9)
    assert math.isnan(table.loc["Intercept", "beta"])
    anova = result["anova"]
    assert anova.loc["regression", "ss"] == pytest.approx(model.ess, rel=1e-10)
    assert anova.loc["residual", "ss"] == pytest.approx(model.ssr, rel=1e-10)
    assert anova.loc["total", "ss"] == pytest.approx(model.centered_tss, rel=1e-12)
    assert list(anova["df"]) == [model.df_model, model.df_resid, len(frame) - 1]
    assert anova.loc["regression", "statistic"] == pytest.approx(model.fvalue, rel=1e-9)
    assert anova.loc["regression", "p_value"] == pytest.approx(model.f_pvalue, rel=1e-8)
    attrs = result.attrs
    assert attrs["n"] == 300 and attrs["n_missing"] == 0
    assert attrs["r_squared"] == pytest.approx(model.rsquared, rel=1e-11)
    assert attrs["adjusted_r_squared"] == pytest.approx(model.rsquared_adj, rel=1e-11)
    assert attrs["rmse"] == pytest.approx(math.sqrt(model.mse_resid), rel=1e-10)
    assert attrs["aic"] == pytest.approx(model.aic, rel=1e-11)
    assert attrs["bic"] == pytest.approx(model.bic, rel=1e-11)
    last = result["steps"].iloc[-1]
    assert last["aic"] == pytest.approx(model.aic, rel=1e-11)
    assert last["adjusted_r_squared"] == pytest.approx(model.rsquared_adj, rel=1e-11)
    assert last["rmse"] == pytest.approx(math.sqrt(model.mse_resid), rel=1e-10)
    changes = result["steps"]["r_squared_change"].sum()
    assert changes == pytest.approx(model.rsquared, rel=1e-11)


def test_excluded_table_matches_augmented_regressions():
    frame = _data()
    result = oe.stepwise(frame, "y", X)
    selected = result.attrs["selected"]
    excluded = result["excluded"]
    assert set(excluded.index) == set(X) - set(selected)
    blocks = _numeric_blocks(X)
    base = _fit(frame, "y", selected, blocks)
    for name, row in excluded.iterrows():
        larger = _fit(frame, "y", [*selected, name], blocks)
        t = larger.tvalues[-1]
        assert row["t"] == pytest.approx(t, rel=1e-8)
        assert row["statistic"] == pytest.approx(t * t, rel=1e-8)
        assert row["p_value"] == pytest.approx(larger.pvalues[-1], rel=1e-7)
        assert row["df1"] == 1 and row["df2"] == larger.df_resid
        beta = larger.params[-1] * frame[name].std() / frame["y"].std()
        assert row["beta_in"] == pytest.approx(beta, rel=1e-9)
        partial = t / math.sqrt(t * t + larger.df_resid)
        assert row["partial_correlation"] == pytest.approx(partial, rel=1e-8)
        auxiliary = _fit(frame, name, selected, blocks)
        assert row["tolerance"] == pytest.approx(1 - auxiliary.rsquared, rel=1e-9)
        design = _design(frame, [*selected, name], blocks)
        lowest = min(1 / variance_inflation_factor(design, i)
                     for i in range(1, design.shape[1]))
        assert row["min_tolerance"] == pytest.approx(lowest, rel=1e-9)
    assert base.df_resid == result.attrs["df2"]


def test_empty_selection_reports_the_constant_only_model():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(80, 4)), columns=["y", "a", "b", "c"])
    result = oe.stepwise(frame, "y", ["a", "b", "c"], p_enter=1e-6, p_remove=1e-5)
    assert result.attrs["selected"] == [] and len(result["steps"]) == 0
    row = result["coefficients"].loc["Intercept"]
    assert row["b"] == pytest.approx(frame["y"].mean())
    assert row["std_error"] == pytest.approx(frame["y"].std() / math.sqrt(80))
    assert result.attrs["r_squared"] == 0.0 and result.attrs["statistic"] is None
    assert len(result["excluded"]) == 3
    assert any("only the constant" in note for note in result.attrs["notes"])


# ---- weights and missing data -----------------------------------------------------


def test_analytic_weights_match_wls():
    frame = _data()
    blocks = _numeric_blocks(X)
    for method in ("forward", "stepwise", "backward"):
        selected, steps = _oracle(frame, "y", blocks, method=method, weights="w")
        result = oe.stepwise(frame, "y", X, method=method, weights="w")
        assert set(result.attrs["selected"]) == set(selected)
        _check_path(result, steps)
    result = oe.stepwise(frame, "y", X, weights="w", weight_type="aweight")
    model = _fit(frame, "y", result.attrs["selected"], blocks, "w")
    np.testing.assert_allclose(result["coefficients"]["b"], model.params, rtol=1e-10)
    np.testing.assert_allclose(result["coefficients"]["std_error"], model.bse, rtol=1e-10)
    assert result.attrs["n"] == 300
    assert result.attrs["weights"] == {"column": "w", "type": "aweight"}
    # Stata scales analytic weights to sum to n: rmse uses the normalized weights.
    w = frame["w"].to_numpy() * len(frame) / frame["w"].sum()
    scaled = sm.WLS(frame["y"].to_numpy(), _design(frame, result.attrs["selected"], blocks),
                    weights=w).fit()
    assert result.attrs["rmse"] == pytest.approx(math.sqrt(scaled.mse_resid), rel=1e-10)
    # The result does not depend on the scale of analytic weights.
    frame["w10"] = 10 * frame["w"]
    again = oe.stepwise(frame, "y", X, weights="w10")
    np.testing.assert_allclose(again["coefficients"].to_numpy(dtype=float),
                               result["coefficients"].to_numpy(dtype=float), rtol=1e-10)


def test_frequency_weights_equal_replicated_rows():
    frame = _data(n=120)
    expanded = frame.loc[frame.index.repeat(frame["f"])].reset_index(drop=True)
    for method in ("forward", "backward", "stepwise"):
        weighted = oe.stepwise(frame, "y", [*X, "g"], categorical=["g"], method=method,
                               weights="f", weight_type="fweight")
        replicated = oe.stepwise(expanded, "y", [*X, "g"], categorical=["g"], method=method)
        assert weighted.attrs["selected"] == replicated.attrs["selected"]
        assert weighted.attrs["n"] == len(expanded)
        for name in weighted:
            left = weighted[name].select_dtypes("number").to_numpy(dtype=float)
            right = replicated[name].select_dtypes("number").to_numpy(dtype=float)
            np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-12)
    model = _fit(expanded, "y", weighted.attrs["selected"], {
        **_numeric_blocks(X),
        "g": list(pd.get_dummies(expanded["g"], prefix="g", dtype=float).columns[1:])},
    ) if "g" not in weighted.attrs["selected"] else None
    if model is not None:
        np.testing.assert_allclose(weighted["coefficients"]["std_error"], model.bse, rtol=1e-9)


def test_zero_weights_leave_the_sample():
    frame = _data(n=100)
    frame.loc[:9, "w"] = 0.0
    result = oe.stepwise(frame, "y", X, weights="w")
    reference = oe.stepwise(frame.iloc[10:], "y", X, weights="w")
    assert result.attrs["n"] == 90
    assert any("zero weight" in note for note in result.attrs["notes"])
    np.testing.assert_allclose(result["coefficients"].to_numpy(dtype=float),
                               reference["coefficients"].to_numpy(dtype=float), rtol=1e-9)


def test_missing_values_are_dropped_listwise_or_rejected():
    frame = _data()
    frame.loc[3, "y"] = np.nan
    frame.loc[7, "x6"] = np.nan          # a candidate that is never selected still counts
    frame.loc[9, "g"] = None
    result = oe.stepwise(frame, "y", [*X, "g"], categorical=["g"])
    assert result.attrs["n"] == 297 and result.attrs["n_missing"] == 3
    complete = frame.dropna().reset_index(drop=True)
    reference = oe.stepwise(complete, "y", [*X, "g"], categorical=["g"])
    assert result.attrs["selected"] == reference.attrs["selected"]
    np.testing.assert_allclose(result["coefficients"].to_numpy(dtype=float),
                               reference["coefficients"].to_numpy(dtype=float), rtol=1e-9)
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(frame, "y", X, missing="raise")
    assert error.value.code == "missing_values"


# ---- tolerance, collinearity and degenerate inputs --------------------------------


def test_tolerance_blocks_collinear_candidates():
    frame = _data()
    rng = np.random.default_rng(1)
    frame["copy"] = frame["x0"]
    frame["near"] = frame["x0"] + 1e-4 * rng.normal(size=len(frame))
    frame["sum"] = frame["x0"] + frame["x2"]
    frame["flat"] = 3.0
    names = [*X, "copy", "near", "sum", "flat"]
    result = oe.stepwise(frame, "y", names, method="forward", p_enter=0.99)
    selected = result.attrs["selected"]
    # Only one of the three near-identical columns can be in the model.
    assert len({"x0", "copy", "near"} & set(selected)) == 1
    assert not {"x0", "x2", "sum"} <= set(selected) and "flat" not in selected
    excluded = result["excluded"]
    assert (excluded.loc[list({"x0", "copy", "near"} - set(selected)), "tolerance"] < 1e-4).all()
    assert math.isnan(excluded.loc["flat", "statistic"])
    assert any("do not vary" in note and "flat" in note for note in result.attrs["notes"])
    assert (result["coefficients"]["tolerance"].iloc[1:] >= 1e-4).all()
    # Backward elimination cannot start from a singular model: the later copy is left out.
    backward = oe.stepwise(frame, "y", ["x0", "x2", "copy", "sum", "x5"], method="backward")
    assert "copy" not in backward.attrs["selected"] and "sum" not in backward.attrs["selected"]
    notes = " ".join(backward.attrs["notes"])
    assert "'copy' was not entered" in notes and "'sum' was not entered" in notes
    reference = oe.stepwise(frame, "y", ["x0", "x2", "x5"], method="backward")
    np.testing.assert_allclose(backward["coefficients"].to_numpy(dtype=float),
                               reference["coefficients"].to_numpy(dtype=float), rtol=1e-10)
    # A looser tolerance lets the nearly collinear column in.
    loose = oe.stepwise(frame, "y", ["x0", "near"], method="backward", p_remove=0.999999,
                        tolerance=1e-9)
    assert loose.attrs["selected"] == ["x0", "near"]
    tight = oe.stepwise(frame, "y", ["x0", "near"], method="backward", p_remove=0.999999)
    assert tight.attrs["selected"] == ["x0"]


def test_large_offsets_and_scales_do_not_change_the_fit():
    frame = _data()
    moved = frame.copy()
    moved["x0"] = 1e7 + frame["x0"]
    moved["x2"] = 1e-3 * frame["x2"] - 5e3
    moved["y"] = 1e5 * frame["y"] + 1e9
    base = oe.stepwise(frame, "y", X)
    result = oe.stepwise(moved, "y", X)
    assert result.attrs["selected"] == base.attrs["selected"] == ["x0", "x2", "x5"]
    np.testing.assert_allclose(result["steps"]["statistic"], base["steps"]["statistic"],
                               rtol=1e-6)
    np.testing.assert_allclose(result["coefficients"]["t"].iloc[1:],
                               base["coefficients"]["t"].iloc[1:], rtol=1e-6)
    # Slopes rescale exactly; a pseudo-inverse of the raw design would lose them here.
    expected = base["coefficients"]["b"].iloc[1:].to_numpy() * np.array([1e5, 1e8, 1e5])
    np.testing.assert_allclose(result["coefficients"]["b"].iloc[1:], expected, rtol=1e-6)
    np.testing.assert_allclose(result["coefficients"]["beta"].iloc[1:],
                               base["coefficients"]["beta"].iloc[1:], rtol=1e-6)
    centred = moved[["x0", "x2", "x5"]] - moved[["x0", "x2", "x5"]].mean()
    model = sm.OLS(moved["y"] - moved["y"].mean(), centred / centred.std()).fit()
    np.testing.assert_allclose(result["coefficients"]["b"].iloc[1:],
                               model.params.to_numpy() / centred.std().to_numpy(), rtol=1e-6)


def test_block_accumulation_equals_a_single_block(monkeypatch):
    from openecon.econometrics.selection import common as module

    frame = _data(n=3000)
    frame.loc[5, "x1"] = np.nan
    whole = oe.stepwise(frame, "y", [*X, "g"], categorical=["g"], weights="w")
    monkeypatch.setattr(module, "BLOCK_ELEMENTS", 64)        # forces several row blocks
    blocked = oe.stepwise(frame, "y", [*X, "g"], categorical=["g"], weights="w")
    assert blocked.attrs["selected"] == whole.attrs["selected"]
    for name in whole:
        np.testing.assert_allclose(blocked[name].select_dtypes("number").to_numpy(dtype=float),
                                   whole[name].select_dtypes("number").to_numpy(dtype=float),
                                   rtol=1e-10, atol=1e-12)


def test_sweep_kernels_invert_and_reverse():
    import torch

    from openecon.econometrics.selection import sweep

    rng = np.random.default_rng(4)
    z = rng.normal(size=(50, 5))
    gram = torch.as_tensor(z.T @ z)
    a = gram.clone()
    for k in (0, 2, 3):
        sweep.sweep(a, k)
    block = np.linalg.inv((z.T @ z)[np.ix_([0, 2, 3], [0, 2, 3])])
    np.testing.assert_allclose(-a[[0, 2, 3]][:, [0, 2, 3]].numpy(), block, rtol=1e-10)
    coefficients = np.linalg.lstsq(z[:, [0, 2, 3]], z[:, [1, 4]], rcond=None)[0]
    np.testing.assert_allclose(a[[0, 2, 3]][:, [1, 4]].numpy(), coefficients, rtol=1e-9)
    residual = z[:, [1, 4]] - z[:, [0, 2, 3]] @ coefficients
    np.testing.assert_allclose(a[[1, 4]][:, [1, 4]].numpy(), residual.T @ residual, rtol=1e-9)
    for k in (3, 0, 2):
        sweep.unsweep(a, k)
    np.testing.assert_allclose(a.numpy(), gram.numpy(), rtol=1e-10, atol=1e-10)
    # One-pass centred cross products equal the two-pass definition, with weights.
    w = rng.uniform(0.5, 2.0, size=50)
    moments = sweep.CrossProducts(torch.as_tensor(z[:7].mean(axis=0)))
    moments.add(torch.as_tensor(z[:20]), torch.as_tensor(w[:20]))
    moments.add(torch.as_tensor(z[20:]), torch.as_tensor(w[20:]))
    means, cross = moments.finish()
    mean = (w[:, None] * z).sum(axis=0) / w.sum()
    np.testing.assert_allclose(means.numpy(), mean, rtol=1e-12)
    np.testing.assert_allclose(cross.numpy(), (w[:, None] * (z - mean)).T @ (z - mean),
                               rtol=1e-11)


# ---- failures ---------------------------------------------------------------------


@pytest.mark.parametrize("kwargs, code", [
    ({"x": "x0"}, "invalid_spec"),
    ({"x": []}, "invalid_spec"),
    ({"x": ["x0", "x0"]}, "invalid_spec"),
    ({"x": ["x0", "y"]}, "invalid_spec"),
    ({"x": ["x0", "nope"]}, "missing_columns"),
    ({"x": ["x0", "g"]}, "non_numeric_column"),
    ({"x": ["x0"], "categorical": ["x1"]}, "invalid_spec"),
    ({"x": ["x0"], "method": "best"}, "invalid_option"),
    ({"x": ["x0"], "criterion": "cp"}, "invalid_option"),
    ({"x": ["x0"], "p_enter": 0.2, "p_remove": 0.1}, "invalid_option"),
    ({"x": ["x0"], "p_enter": 1.5}, "invalid_option"),
    ({"x": ["x0"], "tolerance": 0}, "invalid_option"),
    ({"x": ["x0"], "alpha": 1}, "invalid_option"),
    ({"x": ["x0"], "missing": "pairwise"}, "invalid_option"),
    ({"x": ["x0"], "weight_type": "aweight"}, "invalid_spec"),
    ({"x": ["x0"], "weights": "w", "weight_type": "pweight"}, "invalid_option"),
    ({"x": ["x0"], "weights": "w", "weight_type": "fweight"}, "noninteger_frequency_weights"),
])
def test_invalid_arguments_raise_analysis_errors(kwargs, code):
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(_data(n=40), "y", **kwargs)
    assert error.value.code == code
    assert str(error.value)


def test_degenerate_data_raise_analysis_errors():
    frame = _data(n=40)
    constant = frame.assign(y=2.0)
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(constant, "y", X)
    assert error.value.code == "zero_variance"
    exact = frame.assign(y=3 + 2 * frame["x0"] - frame["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(exact, "y", X)
    assert error.value.code == "perfect_fit"
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(exact, "y", X, method="backward")
    assert error.value.code == "perfect_fit"
    negative = frame.assign(w=-frame["w"])
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(negative, "y", X, weights="w")
    assert error.value.code == "negative_weights"
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(frame.assign(x0=np.inf), "y", X)
    assert error.value.code == "non_finite_values"
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(frame.iloc[:6], "y", X, method="backward")
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(frame.iloc[:0], "y", X)
    assert error.value.code == "empty_data"
    with pytest.raises(AnalysisError) as error:
        oe.stepwise(frame.assign(y=np.nan), "y", X)
    assert error.value.code == "empty_sample"
    # Forward selection never runs out of residual degrees of freedom: it stops entering.
    small = oe.stepwise(frame.iloc[:6], "y", X, method="forward", p_enter=0.9)
    assert small.attrs["df2"] >= 1
