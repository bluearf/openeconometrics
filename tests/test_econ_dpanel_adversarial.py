"""Dynamic panel GMM: adversarial inputs, invariances and regressions of verified defects.

Regression tests for the defects found in verification (each names the defect):
the h(1) error variance, the small-sample factors and degrees of freedom, the
homoskedastic Arellano-Bond test, level-equation GMM groups, passthru dating,
difference-in-Sargan after one-step estimation (checked against the xtabond2
oracle in ``test_econ_dpanel_xtabond2``), and here the large-level collinearity
screen, precision with y around 1e8 and the too-few-panels paths.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from test_econ_dpanel_oracle import oracle, simulate
from test_econ_dpanel_xtabond2 import make_panel

import openecon as oe
from openecon.analysis_contracts import AnalysisError


@pytest.fixture(scope="module")
def panel():
    return make_panel(5, n=40)


def fit(data, **options):
    arguments = dict(data=data, y="y", x=["x"], panel="id", time="t")
    arguments.update(options)
    return oe.xtdpd(**arguments)


def code(data, **options):
    with pytest.raises(AnalysisError) as caught:
        fit(data, **options)
    return caught.value.code


def coefs(result):
    return np.array([c.estimate for c in result.coefficients])


def ses(result):
    return np.array([c.std_error for c in result.coefficients])


def test_regressor_with_huge_level_is_not_mistaken_for_the_constant(panel):
    """Defect: the uncentred screen of the stacked design dropped L1.y when y ~ 1e8."""
    shifted = panel.assign(y=panel.y + 1e8)
    for options in ({"system": True}, {"system": True, "twostep": True, "robust": True}):
        result = fit(shifted, **options)
        assert [c.term for c in result.coefficients] == ["Intercept", "L1.y", "x"]
        assert result.provenance["omitted_terms"] == []
        table = {c.term: c.estimate for c in result.coefficients}
        # The intercept absorbs (1 - rho) * 1e8: it is reported in the original units.
        assert_allclose(table["Intercept"], (1 - table["L1.y"]) * 1e8, rtol=1e-3)


def test_centring_is_an_exact_reparametrization(panel):
    """Shifting x by a constant in system GMM changes only the intercept (by -b_x c)."""
    base = fit(panel, system=True, twostep=True, robust=True)
    moved = fit(panel.assign(x=panel.x + 50.0), system=True, twostep=True, robust=True)
    b0, b1 = {c.term: c for c in base.coefficients}, {c.term: c for c in moved.coefficients}
    # x enters the level equation as a regressor and the difference equation through
    # differenced IV-style instruments, so the shift is absorbed by the constant only.
    assert_allclose(b1["L1.y"].estimate, b0["L1.y"].estimate, rtol=1e-8)
    assert_allclose(b1["x"].estimate, b0["x"].estimate, rtol=1e-8)
    assert_allclose(b1["Intercept"].estimate, b0["Intercept"].estimate - 50 * b0["x"].estimate,
                    rtol=1e-7)
    assert_allclose(b1["x"].std_error, b0["x"].std_error, rtol=1e-7)
    cov0, cov1 = np.array(base.covariance_matrix), np.array(moved.covariance_matrix)
    jac = np.eye(3)
    jac[0, 2] = -50.0
    assert_allclose(cov1, jac @ cov0 @ jac.T, rtol=1e-6, atol=1e-12)
    for name in ("hansen", "sargan", "ar1", "ar2"):
        assert_allclose(moved.tests[name]["statistic"], base.tests[name]["statistic"], rtol=1e-7)


def test_huge_level_matches_sixty_digit_arithmetic():
    """y around 1e8: OpenEcon agrees with mpmath; float64 explicit inverses do not."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    data = simulate(seed=5, n=60, periods=6)
    data = data.assign(y=data.y + 1e8)
    with np.errstate(invalid="ignore"):      # the float64 explicit-inverse oracle breaks down
        orc = oracle(data, system=True)
    blocks, n_inst, k = orc["blocks"], orc["n_instruments"], orc["k"]

    def matrix(values):
        return mp.matrix(np.atleast_2d(values).tolist())

    a = sum((matrix(b["z"]).T * matrix(b["x"]) for b in blocks), mp.zeros(n_inst, k))
    rhs = sum((matrix(b["z"]).T * matrix(b["y"][:, None]) for b in blocks), mp.zeros(n_inst, 1))
    zhz = sum((matrix(b["z"]).T * matrix(b["h"]) * matrix(b["z"]) for b in blocks),
              mp.zeros(n_inst, n_inst))
    w1 = zhz ** -1
    exact = np.array([float(v) for v in (a.T * w1 * a) ** -1 * (a.T * w1 * rhs)])
    result = oe.xtdpd(data=data, y="y", x=["x"], panel="id", time="year", system=True)
    assert_allclose(coefs(result), exact, rtol=1e-6)
    assert np.max(np.abs(orc["beta1"] - exact) / np.abs(exact)) > 1e-3


def test_too_few_panels_for_robust_or_two_step(panel):
    """Defect: one panel gave a misleading 'add instruments' error."""
    one = panel[panel.id == 0]
    assert fit(one).nobs > 0                                  # one-step nonrobust is defined
    for options in ({"robust": True}, {"twostep": True}):
        assert code(one, **options) == "insufficient_clusters"
    two = panel[panel.id < 2]
    with pytest.raises(AnalysisError) as caught:
        fit(two, system=True, twostep=True, robust=True)
    assert caught.value.code == "underidentified" and "panels" in str(caught.value)
    # One-step robust stays available; the two-step Hansen statistic is not.
    result = fit(two, system=True, robust=True)
    assert result.tests["hansen"]["statistic"] is None and "note" in result.tests["hansen"]
    assert any("Hansen statistic is not available" in w for w in result.warnings)


def test_invariances(panel):
    base = fit(panel, system=True, twostep=True, robust=True)
    # Duplicating every panel: same estimates, standard errors / sqrt(2), statistics doubled.
    dup = pd.concat([panel, panel.assign(id=panel.id + 1000)], ignore_index=True)
    doubled = fit(dup, system=True, twostep=True, robust=True)
    assert_allclose(coefs(doubled), coefs(base), rtol=1e-10)
    assert_allclose(ses(doubled), ses(base) / np.sqrt(2), rtol=1e-8)
    assert_allclose(doubled.tests["hansen"]["statistic"], 2 * base.tests["hansen"]["statistic"],
                    rtol=1e-8)
    # Rescaling y and x: L1.y unchanged, x scaled, every test statistic unchanged.
    scaled = fit(panel.assign(y=panel.y * 1e3, x=panel.x * 1e-2), system=True, twostep=True,
                 robust=True)
    table, ref = {c.term: c for c in scaled.coefficients}, {c.term: c for c in base.coefficients}
    assert_allclose(table["L1.y"].estimate, ref["L1.y"].estimate, rtol=1e-8)
    assert_allclose(table["x"].estimate, ref["x"].estimate * 1e5, rtol=1e-8)
    assert_allclose(table["Intercept"].estimate, ref["Intercept"].estimate * 1e3, rtol=1e-8)
    assert_allclose(table["x"].statistic, ref["x"].statistic, rtol=1e-7)
    for name in ("ar1", "ar2", "sargan", "hansen", "diff_hansen_level", "model"):
        assert_allclose(scaled.tests[name]["statistic"], base.tests[name]["statistic"],
                        rtol=1e-7)
    # Shifting time, relabelling panels and permuting rows change nothing.
    moved = fit(panel.assign(t=panel.t - 3000, id="p" + panel.id.astype(str)).iloc[::-1],
                system=True, twostep=True, robust=True)
    assert_allclose(coefs(moved), coefs(base), rtol=1e-11)
    assert_allclose(moved.tests["ar2"]["statistic"], base.tests["ar2"]["statistic"], rtol=1e-10)


def test_bad_inputs_raise_named_errors(panel):
    cases = [
        (panel.iloc[:0], {}, "empty_data"),
        (panel.iloc[:1], {}, "insufficient_observations"),
        (panel.assign(x=np.nan), {"missing": "drop"}, "empty_sample"),
        (panel.assign(x=np.nan), {}, "missing_values"),
        (panel.assign(y=1.0), {}, "perfect_fit"),
        (panel.assign(y=1.0), {"system": True}, "perfect_fit"),
        (panel.assign(y="a"), {}, "non_numeric_column"),
        (panel.assign(x=panel.x.astype(str)), {}, "non_numeric_column"),
        (panel.assign(y=np.where(panel.index == 3, np.inf, panel.y)), {}, "non_finite_values"),
        (pd.concat([panel, panel.iloc[:1]]), {}, "repeated_time_values"),
        (panel.assign(t=panel.t + 0.5), {}, "invalid_time"),
        (panel.assign(t=panel.t * 10 ** 9), {}, "time_span_too_large"),
        (panel[panel.t <= panel.t.min() + 1], {}, "insufficient_observations"),
        (panel, {"lags": 50}, "insufficient_observations"),
        (panel, {"lags": 6}, "insufficient_observations"),
        (panel, {"x": ["id"]}, "invalid_spec"),
        (panel, {"x": ["y"]}, "invalid_spec"),
        (panel, {"x": ["x", "x"]}, "invalid_spec"),
        (panel, {"gmm": [], "iv": [{"columns": ["x"]}]}, "underidentified"),
        (panel, {"gmm": [{"columns": ["y"], "lags": [100, 200]}]}, "underidentified"),
        (panel, {"lags": 0, "gmm": [], "iv": []}, "underidentified"),
        (panel, {"gmm": [{"columns": ["y"], "lags": "2 ."}]}, "invalid_spec"),
        (panel, {"gmm": [{"columns": ["y"], "lags": [0, 0]}], "system": True}, "invalid_spec"),
        (panel, {"iv": [{"columns": ["x"], "passthru": True}], "system": True}, "invalid_spec"),
        (panel, {"artests": 0}, "invalid_spec"),
        (panel, {"h": 0}, "invalid_spec"),
        (panel, {"lags": True}, "invalid_spec"),
        (panel, {"system": "yes"}, "invalid_spec"),
        (panel, {"covariance": "cluster"}, "invalid_spec"),
        (panel, {"robust": "yes"}, "invalid_spec"),
        (panel, {"missing": "zero"}, "invalid_spec"),
        (panel, {"alpha": 0.0}, "invalid_spec"),
        ([1, 2, 3], {}, "invalid_data"),
    ]
    for data, options, expected in cases:
        assert code(data, **options) == expected, (options, expected)


def test_options_at_their_bounds(panel):
    # Exactly identified (three periods): no overidentification test, no AR pairs.
    short = fit(panel[panel.t <= panel.t.min() + 2])
    assert short.tests["sargan"]["statistic"] is None and short.tests["sargan"]["df"] == 0
    assert short.tests["ar1"]["statistic"] is None and "note" in short.tests["ar1"]
    # artests beyond the panel length: the unavailable orders carry a note, not NaN.
    deep = fit(panel, artests=20)
    assert deep.tests["ar5"]["statistic"] is None and deep.tests["ar20"]["p_value"] is None
    assert all(np.isfinite(deep.tests[f"ar{m}"]["statistic"]) for m in (1, 2, 3))
    # A lag limit beyond the data is clamped (same instruments as 'all available').
    capped = fit(panel, gmm=[{"columns": ["y"], "lags": [2, 10 ** 6]}])
    assert_allclose(coefs(capped), coefs(fit(panel)), rtol=1e-12)
    # lo = 0 in a level-only group (the contemporaneous difference) is legal.
    level0 = fit(panel, system=True, x=["x", "w"],
                 gmm=[{"columns": ["y"], "lags": [2, None]},
                      {"columns": ["w"], "lags": [0, 0], "equation": "level"}])
    assert level0.extra["instrument_groups"]["gmm2"]["columns"] >= 1
    # Extreme magnitudes of x: coefficients scale exactly.
    tiny = fit(panel.assign(x=panel.x * 1e-8), twostep=True, robust=True)
    huge = fit(panel.assign(x=panel.x * 1e8), twostep=True, robust=True)
    ref = fit(panel, twostep=True, robust=True)
    assert_allclose(coefs(tiny) * [1, 1e-8], coefs(ref), rtol=1e-8)
    assert_allclose(coefs(huge) * [1, 1e8], coefs(ref), rtol=1e-8)
    # Collinear instruments are screened out without changing the estimator.
    dup_z = fit(panel.assign(z2=2 * panel.z), iv=[{"columns": ["x", "z", "z2"]}])
    one_z = fit(panel, iv=[{"columns": ["x", "z"]}])
    assert_allclose(coefs(dup_z), coefs(one_z), rtol=1e-10)
    assert dup_z.extra["instruments_collinear_dropped"] == 1
    # A constant regressor differences out (difference GMM) or duplicates the constant.
    for options in ({}, {"system": True}):
        steady = fit(panel.assign(x=3.0), **options)
        assert "x" in steady.provenance["omitted_terms"]
        assert any("x" in w for w in steady.warnings)
    # Singleton panels and categorical / string panel identifiers are handled.
    lone = pd.DataFrame({"id": [999], "t": [2001], "y": [0.1], "x": [0.2], "w": [0.0],
                         "z": [0.0]})
    assert_allclose(coefs(fit(pd.concat([panel, lone], ignore_index=True))), coefs(fit(panel)),
                    rtol=1e-12)
    assert_allclose(coefs(fit(panel.assign(id=pd.Categorical(panel.id)))), coefs(fit(panel)),
                    rtol=1e-12)


def test_results_never_contain_non_finite_numbers(panel):
    configurations = [{}, {"robust": True}, {"twostep": True}, {"system": True, "h": 1},
                      {"orthogonal": True, "small": True, "robust": True},
                      {"system": True, "twostep": True, "robust": True, "time_dummies": True,
                       "artests": 6}]
    for options in configurations:
        result = fit(panel, **options)
        payload = result.model_dump()
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, float):
                assert np.isfinite(item), options
