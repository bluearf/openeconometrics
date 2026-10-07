"""Published textbook examples: Hald's cement data and the Longley data.

Hald (1952), analysed in Draper and Smith, "Applied Regression Analysis"
(stepwise chapter): with 0.15 to enter and to remove, x4 enters, then x1, then
x2, and x4 is removed; the final equation is y = 52.58 + 1.468 x1 + 0.6623 x2
with R-squared 97.87%. Longley (1967): the variance inflation factors reported
by every regression text and package (135.53, 1788.51, 33.62, 3.59, 399.15,
758.98) and the scaled condition number 43,275 of Belsley (1991).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import openecon as oe

HALD = pd.DataFrame({
    "x1": [7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10],
    "x2": [26, 29, 56, 31, 52, 55, 71, 31, 54, 47, 40, 66, 68],
    "x3": [6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8],
    "x4": [60, 52, 20, 47, 33, 22, 6, 44, 22, 26, 34, 12, 12],
    "y": [78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9, 83.8, 113.3, 109.4],
})
X = ["x1", "x2", "x3", "x4"]


def test_hald_stepwise_follows_draper_and_smith():
    result = oe.stepwise(HALD, "y", X, p_enter=0.15, p_remove=0.15)
    steps = result["steps"]
    assert list(steps["action"]) == ["entered", "entered", "entered", "removed"]
    assert list(steps["term"]) == ["x4", "x1", "x2", "x4"]
    np.testing.assert_allclose(steps["statistic"], [22.80, 108.22, 5.03, 1.86], atol=0.006)
    np.testing.assert_allclose(100 * steps["r_squared"], [67.45, 97.25, 98.23, 97.87],
                               atol=0.006)
    assert result.attrs["selected"] == ["x1", "x2"]
    table = result["coefficients"]
    assert table.loc["Intercept", "b"] == pytest.approx(52.58, abs=0.005)
    assert table.loc["x1", "b"] == pytest.approx(1.468, abs=0.0005)
    assert table.loc["x2", "b"] == pytest.approx(0.6623, abs=0.00005)
    assert result.attrs["statistic"] == pytest.approx(229.50, abs=0.005)
    assert result["anova"].loc["residual", "ss"] == pytest.approx(57.90, abs=0.005)


def test_hald_with_spss_default_thresholds_and_other_methods():
    # PIN(.05) POUT(.10): x2 (p = 0.052) is not entered after x4 and x1.
    result = oe.stepwise(HALD, "y", X)
    assert result.attrs["selected"] == ["x4", "x1"]
    table = result["coefficients"]
    assert table.loc["Intercept", "b"] == pytest.approx(103.10, abs=0.005)
    assert table.loc["x1", "b"] == pytest.approx(1.440, abs=0.0005)
    assert table.loc["x4", "b"] == pytest.approx(-0.614, abs=0.0005)
    assert result["excluded"].loc["x2", "p_value"] == pytest.approx(0.0517, abs=0.0001)
    # Backward elimination from the full model drops x3, then x4.
    backward = oe.stepwise(HALD, "y", X, method="backward")
    assert list(backward["steps"]["term"]) == ["(all terms)", "x3", "x4"]
    assert backward.attrs["selected"] == ["x1", "x2"]
    full = sm.OLS(HALD["y"], sm.add_constant(HALD[X])).fit()
    start = backward["steps"].iloc[0]
    assert start["r_squared"] == pytest.approx(full.rsquared, rel=1e-10)
    assert start["statistic"] == pytest.approx(full.fvalue, rel=1e-9)
    assert start["aic"] == pytest.approx(full.aic, rel=1e-10)


def test_hald_variance_inflation_factors():
    result = oe.collin(HALD, X)
    np.testing.assert_allclose(result["vif"]["vif"], [38.50, 254.42, 46.87, 282.51], atol=0.006)


def test_longley_collinearity_diagnostics():
    longley = sm.datasets.longley.load_pandas().data
    names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
    result = oe.collin(longley, names)
    np.testing.assert_allclose(
        result["vif"]["vif"], [135.53244, 1788.51348, 33.61889, 3.58893, 399.15102, 758.98060],
        rtol=2e-7)
    assert result.attrs["condition_number"] == pytest.approx(43275, abs=0.5)
    # The dominant dependency involves the constant and the year.
    last = result["condition"].iloc[-1]
    assert last["Intercept"] > 0.99 and last["YEAR"] > 0.99
    assert result["condition"]["eigenvalue"].sum() == pytest.approx(7.0, rel=1e-12)
