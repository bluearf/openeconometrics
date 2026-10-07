"""oe.curvefit against transformed OLS (statsmodels) and numpy.polyfit."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.selection.curvefit import MODELS

U = 40.0


def _data(seed: int = 0, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 10, n)
    y = np.exp(0.3 + 0.25 * x + 0.1 * rng.normal(size=n))
    return pd.DataFrame({"x": x, "y": y})


# model -> (outcome transform, regressors, back-transform of (a0, a1))
_ORACLE = {
    "linear": (lambda y: y, lambda x: [x], None),
    "logarithmic": (lambda y: y, lambda x: [np.log(x)], None),
    "inverse": (lambda y: y, lambda x: [1 / x], None),
    "quadratic": (lambda y: y, lambda x: [x, x ** 2], None),
    "cubic": (lambda y: y, lambda x: [x, x ** 2, x ** 3], None),
    "compound": (np.log, lambda x: [x], (True, True)),
    "power": (np.log, lambda x: [np.log(x)], (True, False)),
    "s": (np.log, lambda x: [1 / x], (False, False)),
    "growth": (np.log, lambda x: [x], (False, False)),
    "exponential": (np.log, lambda x: [x], (True, False)),
    "logistic": (lambda y: np.log(1 / y - 1 / U), lambda x: [x], (True, True)),
}


def _reference(frame, name, intercept=True):
    transform, regressors, back = _ORACLE[name]
    design = np.column_stack(regressors(frame["x"].to_numpy()))
    if intercept:
        design = sm.add_constant(design)
    model = sm.OLS(transform(frame["y"].to_numpy()), design).fit()
    params = list(model.params)
    if not intercept:
        params = [np.nan, *params]
    if back is not None:
        if back[0]:
            params[0] = math.exp(params[0]) if intercept else np.nan
        if back[1]:
            params[1] = math.exp(params[1])
    return model, params + [np.nan] * (4 - len(params))


def test_curvefit_is_exported_and_renders():
    from openecon.econometrics import registry

    assert registry.public_exports()["curvefit"][0] == "openecon.econometrics.selection.curvefit"
    assert "CURVEFIT" in oe.curvefit.__doc__
    result = oe.curvefit(_data(), "y", "x", upper_bound=U)
    assert isinstance(result, TableSet) and list(result) == ["summary", "fitted"]
    assert list(result["summary"].index) == list(MODELS)
    text = str(result)
    assert "Curve estimation of y on x" in text and "of 400)" in text
    assert len(text.splitlines()) < 40                    # the plotting grid is abbreviated
    assert repr(result) == text and len(result["fitted"]) == 400
    latex = result.to_latex()
    assert latex.count(r"\begin{tabular}") + latex.count(r"\begin{longtable}") == 2
    assert json.loads(json.dumps(result.attrs)) == result.attrs
    for table in result.values():
        assert set(json.loads(table.to_json())) == set(table.columns)
        assert not np.isinf(table.select_dtypes("number").to_numpy(dtype=float)).any()


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("intercept", [True, False])
def test_each_model_matches_transformed_ols(name, intercept):
    frame = _data()
    result = oe.curvefit(frame, "y", "x", models=[name], upper_bound=U, intercept=intercept)
    model, params = _reference(frame, name, intercept)
    row = result["summary"].loc[name]
    assert row["r_squared"] == pytest.approx(model.rsquared, rel=1e-10)
    assert row["statistic"] == pytest.approx(model.fvalue, rel=1e-9)
    assert row["df1"] == model.df_model + (0 if intercept else 0) and row["df2"] == model.df_resid
    assert row["p_value"] == pytest.approx(model.f_pvalue, rel=1e-8, abs=1e-300)
    np.testing.assert_allclose(row[["b0", "b1", "b2", "b3"]].to_numpy(dtype=float), params,
                               rtol=1e-8, equal_nan=True)
    assert result.attrs["n"] == 80 and result.attrs["fitted_models"] == [name]


def test_polynomials_match_numpy_polyfit():
    frame = _data(seed=4)
    result = oe.curvefit(frame, "y", "x", models=["linear", "quadratic", "cubic"])
    for name, degree in (("linear", 1), ("quadratic", 2), ("cubic", 3)):
        expected = np.polyfit(frame["x"], frame["y"], degree)[::-1]
        observed = result["summary"].loc[name, ["b0", "b1", "b2", "b3"]].to_numpy(dtype=float)
        np.testing.assert_allclose(observed[:degree + 1], expected, rtol=1e-8)
        assert np.isnan(observed[degree + 1:]).all()


def test_fitted_curves_follow_the_reported_equations():
    frame = _data()
    result = oe.curvefit(frame, "y", "x", upper_bound=U)
    fitted, summary = result["fitted"], result["summary"]
    assert len(fitted) == 400 and list(fitted.columns) == ["x", *MODELS]
    grid = fitted["x"].to_numpy()
    assert grid[0] == pytest.approx(frame["x"].min()) and grid[-1] == pytest.approx(
        frame["x"].max())
    np.testing.assert_allclose(np.diff(grid), np.diff(grid)[0], rtol=1e-9)
    b = {name: summary.loc[name, ["b0", "b1", "b2", "b3"]].to_numpy(dtype=float)
         for name in MODELS}
    curves = {
        "linear": b["linear"][0] + b["linear"][1] * grid,
        "logarithmic": b["logarithmic"][0] + b["logarithmic"][1] * np.log(grid),
        "inverse": b["inverse"][0] + b["inverse"][1] / grid,
        "quadratic": b["quadratic"][0] + b["quadratic"][1] * grid + b["quadratic"][2] * grid ** 2,
        "cubic": b["cubic"][0] + b["cubic"][1] * grid + b["cubic"][2] * grid ** 2
        + b["cubic"][3] * grid ** 3,
        "compound": b["compound"][0] * b["compound"][1] ** grid,
        "power": b["power"][0] * grid ** b["power"][1],
        "s": np.exp(b["s"][0] + b["s"][1] / grid),
        "growth": np.exp(b["growth"][0] + b["growth"][1] * grid),
        "exponential": b["exponential"][0] * np.exp(b["exponential"][1] * grid),
        "logistic": 1 / (1 / U + b["logistic"][0] * b["logistic"][1] ** grid),
    }
    for name, expected in curves.items():
        np.testing.assert_allclose(fitted[name], expected, rtol=1e-8)
    # Compound, growth and exponential are the same regression of ln(y) on x.
    np.testing.assert_allclose(fitted["compound"], fitted["growth"], rtol=1e-12)
    assert summary.loc["compound", "r_squared"] == summary.loc["exponential", "r_squared"]
    assert math.log(summary.loc["compound", "b1"]) == pytest.approx(
        summary.loc["growth", "b1"], rel=1e-12)
    # Without a constant the multiplicative models have b0 = 1.
    origin = oe.curvefit(frame, "y", "x", models=["compound", "linear"], intercept=False)
    slope = origin["summary"].loc["compound", "b1"]
    np.testing.assert_allclose(origin["fitted"]["compound"], slope ** grid, rtol=1e-9)
    assert math.isnan(origin["summary"].loc["compound", "b0"])
    assert origin["summary"].loc["linear", "equation"] == "y = b1*x"


def test_logistic_default_upper_bound_is_infinite():
    frame = _data()
    result = oe.curvefit(frame, "y", "x", models=["logistic"])
    model = sm.OLS(np.log(1 / frame["y"]), sm.add_constant(frame["x"])).fit()
    row = result["summary"].loc["logistic"]
    assert row["b0"] == pytest.approx(math.exp(model.params.iloc[0]), rel=1e-9)
    assert row["b1"] == pytest.approx(math.exp(model.params.iloc[1]), rel=1e-9)
    assert row["r_squared"] == pytest.approx(model.rsquared, rel=1e-10)
    assert result.attrs["upper_bound"] is None
    grid = result["fitted"]["x"].to_numpy()
    np.testing.assert_allclose(result["fitted"]["logistic"], 1 / (row["b0"] * row["b1"] ** grid),
                               rtol=1e-9)


def test_domain_problems_skip_the_model_with_a_note():
    frame = _data()
    negative_y = frame.assign(y=frame["y"] - frame["y"].mean())
    result = oe.curvefit(negative_y, "y", "x")
    log_models = ["compound", "power", "s", "growth", "exponential", "logistic"]
    assert result.attrs["skipped"] == log_models
    assert result.attrs["fitted_models"] == ["linear", "logarithmic", "inverse", "quadratic",
                                             "cubic"]
    summary = result["summary"]
    assert summary.loc[log_models, "r_squared"].isna().all()
    assert summary.loc["compound", "equation"] == "y = b0 * b1^x"
    assert list(result["fitted"].columns) == ["x", *result.attrs["fitted_models"]]
    notes = result.attrs["notes"]
    assert len(notes) == 6 and all("non-positive" in note for note in notes)
    assert "Note: compound: not fitted" in str(result)
    # Predictor domain: ln(x) needs x > 0, 1/x needs x != 0.
    shifted = frame.assign(x=np.round(frame["x"]) - 5)
    result = oe.curvefit(shifted, "y", "x")
    assert result.attrs["skipped"] == ["logarithmic", "inverse", "power", "s"]
    nonzero = frame.assign(x=frame["x"] - 20)
    result = oe.curvefit(nonzero, "y", "x")
    assert result.attrs["skipped"] == ["logarithmic", "power"]
    assert not result["summary"].loc["inverse"].isna()["b1"]
    # An upper bound at or below the largest outcome disables only the logistic model.
    result = oe.curvefit(frame, "y", "x", upper_bound=float(frame["y"].max()))
    assert result.attrs["skipped"] == ["logistic"]
    assert "does not exceed" in result.attrs["notes"][0]
    # Too few distinct predictor values for a cubic.
    few = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 4, "y": np.arange(12.0) ** 1.5 + 1})
    result = oe.curvefit(few, "y", "x", models=["linear", "quadratic", "cubic"])
    assert result.attrs["skipped"] == ["cubic"]
    assert "too few distinct values" in result.attrs["notes"][0]
    tiny = pd.DataFrame({"x": [1.0, 2.0, 4.0], "y": [1.0, 3.0, 2.0]})
    result = oe.curvefit(tiny, "y", "x", models=["linear", "quadratic"])
    assert result.attrs["skipped"] == ["quadratic"]
    assert "at least 4 observations" in result.attrs["notes"][0]


def test_exact_fit_reports_no_f_statistic():
    x = np.linspace(1, 5, 12)
    frame = pd.DataFrame({"x": x, "y": 2.0 * np.exp(0.5 * x)})
    result = oe.curvefit(frame, "y", "x", models=["exponential", "linear"])
    row = result["summary"].loc["exponential"]
    assert row["r_squared"] == 1.0 and math.isnan(row["statistic"])
    assert row["b0"] == pytest.approx(2.0, rel=1e-10) and row["b1"] == pytest.approx(0.5)
    assert any("exact" in note for note in result.attrs["notes"])
    assert result["summary"].loc["linear", "r_squared"] < 1


def test_calendar_years_do_not_break_the_polynomials():
    rng = np.random.default_rng(3)
    year = np.arange(1990.0, 2024.0)
    t = year - year.mean()
    y = 5 + 0.8 * t - 0.05 * t ** 2 + 0.004 * t ** 3 + rng.normal(size=len(t))
    frame = pd.DataFrame({"year": year, "y": y})
    result = oe.curvefit(frame, "y", "year", models=["linear", "quadratic", "cubic"])
    assert result.attrs["skipped"] == []
    centred = sm.OLS(y, sm.add_constant(np.column_stack([t, t ** 2, t ** 3]))).fit()
    row = result["summary"].loc["cubic"]
    assert row["r_squared"] == pytest.approx(centred.rsquared, rel=1e-10)
    assert row["statistic"] == pytest.approx(centred.fvalue, rel=1e-8)
    assert row["b3"] == pytest.approx(centred.params[3], rel=1e-8)
    # The fitted curve is evaluated in the centred form and stays accurate.
    grid = result["fitted"]["year"].to_numpy() - year.mean()
    expected = centred.params @ np.vstack([np.ones_like(grid), grid, grid ** 2, grid ** 3])
    np.testing.assert_allclose(result["fitted"]["cubic"], expected, rtol=1e-9)


def test_missing_values_and_input_forms():
    frame = _data()
    frame.loc[[2, 9], "y"] = np.nan
    frame.loc[5, "x"] = np.nan
    result = oe.curvefit(frame, "y", "x", models=["linear"])
    assert result.attrs["n"] == 77 and result.attrs["n_missing"] == 3
    complete = frame.dropna()
    model = sm.OLS(complete["y"], sm.add_constant(complete["x"])).fit()
    assert result["summary"].loc["linear", "b1"] == pytest.approx(model.params["x"], rel=1e-10)
    with pytest.raises(AnalysisError) as error:
        oe.curvefit(frame, "y", "x", missing="raise")
    assert error.value.code == "missing_values"
    records = complete.to_dict("records")
    assert oe.curvefit(records, "y", "x", models=["linear"])["summary"].loc[
        "linear", "b1"] == pytest.approx(model.params["x"], rel=1e-10)
    # The order of the requested models is the order of the table.
    result = oe.curvefit(complete, "y", "x", models=["growth", "linear"])
    assert list(result["summary"].index) == ["growth", "linear"]
    assert list(result["fitted"].columns) == ["x", "growth", "linear"]


@pytest.mark.parametrize("kwargs, code", [
    ({"x": ["x"]}, "invalid_spec"),
    ({"x": "nope"}, "missing_columns"),
    ({"x": "y"}, "invalid_spec"),
    ({"x": "label"}, "non_numeric_column"),
    ({"x": "x", "models": "linear"}, "invalid_spec"),
    ({"x": "x", "models": []}, "invalid_spec"),
    ({"x": "x", "models": ["linear", "linear"]}, "invalid_spec"),
    ({"x": "x", "models": ["spline"]}, "invalid_option"),
    ({"x": "x", "upper_bound": -1}, "invalid_option"),
    ({"x": "x", "upper_bound": "big"}, "invalid_option"),
    ({"x": "x", "intercept": "yes"}, "invalid_option"),
    ({"x": "x", "missing": "keep"}, "invalid_option"),
])
def test_invalid_arguments_raise_analysis_errors(kwargs, code):
    frame = _data(n=20).assign(label="a")
    with pytest.raises(AnalysisError) as error:
        oe.curvefit(frame, "y", **kwargs)
    assert error.value.code == code


def test_degenerate_data_raise_analysis_errors():
    frame = _data(n=20)
    for column in ("x", "y"):
        with pytest.raises(AnalysisError) as error:
            oe.curvefit(frame.assign(**{column: 3.0}), "y", "x")
        assert error.value.code == "zero_variance"
    with pytest.raises(AnalysisError) as error:
        oe.curvefit(frame.iloc[:2], "y", "x")
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        oe.curvefit(frame.assign(y=np.inf), "y", "x")
    assert error.value.code == "non_finite_values"
    with pytest.raises(AnalysisError) as error:
        oe.curvefit(frame.assign(y=np.nan), "y", "x")
    assert error.value.code == "empty_sample"
    with pytest.raises(AnalysisError) as error:
        oe.curvefit(frame.rename(columns={"x": "linear"}), "y", "linear")
    assert error.value.code == "invalid_spec"
