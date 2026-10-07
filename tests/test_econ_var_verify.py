"""Second verification pass of the var family: textbook output and structural identities.

Part 4 of the verification suite (published Stata output and the critical-value
sources are in ``test_econ_var_oracle.py``, algebraic oracles in
``test_econ_var_oracle_algebra.py``, hostile inputs in
``test_econ_var_adversarial.py``). Everything here is derived independently of
the implementation:

* the worked example of Lutkepohl (2005), sections 3.2.3 and 3.5.3 (West German
  investment / income / consumption): coefficients, the residual covariance,
  point forecasts, interval forecasts and the forecast MSE matrices WITH the
  estimation-uncertainty term -- the formula behind Stata's ``fcast compute``;
* the closed form of the one-step forecast MSE, ``Sigma (1 + m / T)``;
* the ``vecrank`` tables printed in [TS] vec intro (parameter counts, trace
  statistics from the printed log likelihoods, critical values);
* generalized impulse responses and their standard errors against the
  orthogonalized ones of the model refitted with the impulse variable ordered
  first (Pesaran and Shin 1998, Proposition 3.1);
* statsmodels ``VAR`` with a trend and without a constant;
* invariance of every reduced-form statistic to the order of the variables.
"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels
from numpy.testing import assert_allclose
from statsmodels.tsa.api import VAR

import openecon as oe
from openecon.econometrics.var import critical_values

NAMES = ["dln_inv", "dln_inc", "dln_consump"]


@pytest.fixture(scope="module")
def west_germany():
    """Lutkepohl's data up to 1978q4: 75 differences, 73 observations with two lags."""
    path = os.path.join(os.path.dirname(statsmodels.__file__), "tsa", "tests", "results",
                        "lutkepohl2.dta")
    if not os.path.exists(path):
        pytest.skip("statsmodels does not ship the Lutkepohl data")
    frame = pd.read_stata(path)
    return frame[frame.qtr <= "1978-12-31"].iloc[1:].reset_index(drop=True)


def simulated(seed=11, n=260):
    rng = np.random.default_rng(seed)
    a1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.3, 0.1], [0.0, 0.2, 0.4]])
    a2 = np.array([[-0.2, 0.0, 0.1], [0.0, 0.1, 0.0], [0.1, 0.0, -0.1]])
    chol = np.linalg.cholesky(np.array([[1.0, 0.3, 0.2], [0.3, 2.0, 0.1], [0.2, 0.1, 0.5]]))
    e = rng.normal(size=(n, 3)) @ chol.T
    y = np.zeros((n, 3))
    for t in range(2, n):
        y[t] = 1.0 + a1 @ y[t - 1] + a2 @ y[t - 2] + e[t]
    return pd.DataFrame(y, columns=["a", "b", "c"])


# ---- Lutkepohl (2005), New Introduction to Multiple Time Series Analysis ---------------------

def test_lutkepohl_textbook_estimates_forecasts_and_forecast_mse(west_germany):
    """Sections 3.2.3 and 3.5.3: VAR(2) with intercept, T = 73, Sigma with divisor T - Kp - 1.

    The printed numbers (three decimals for the coefficients and forecasts, the
    MSE matrices times 10^4) are reproduced to every printed digit. The
    forecast MSE is ``Sigma_y(h) + Omega(h) / T``: the estimation-uncertainty
    term matters (step 1: 23.34 against 21.30 without it).
    """
    result = oe.var(data=west_germany, y=NAMES, time="qtr", lags=2, dfk=True)
    assert result.nobs == 73
    estimate = {c.term: c.estimate for c in result.coefficients}
    a1 = [[-.320, .146, .961], [.044, -.153, .289], [-.002, .225, -.264]]
    a2 = [[-.161, .115, .934], [.050, .019, -.010], [.034, .355, -.022]]
    intercept = [-.017, .016, .013]
    for i, equation in enumerate(NAMES):
        assert estimate[f"{equation}:Intercept"] == pytest.approx(intercept[i], abs=6e-4)
        for v, variable in enumerate(NAMES):
            assert estimate[f"{equation}:L1.{variable}"] == pytest.approx(a1[i][v], abs=6e-4)
            assert estimate[f"{equation}:L2.{variable}"] == pytest.approx(a2[i][v], abs=6e-4)
    assert_allclose(np.array(result.extra["sigma"]) * 1e4,
                    [[21.30, .72, 1.23], [.72, 1.37, .61], [1.23, .61, .89]], atol=6e-3)

    table = oe.forecast(result, 2)
    assert "coefficient uncertainty" in table.attrs["mse_formula"]
    rows = {(row.variable, row.step): row for row in table.itertuples()}
    published = {   # forecast, half width of the 95% interval, MSE * 10^4
        ("dln_inv", 1): (-.011, .095, 23.34), ("dln_inv", 2): (.011, .098, 25.12),
        ("dln_inc", 1): (.020, .024, 1.505), ("dln_inc", 2): (.020, .025, 1.581),
        ("dln_consump", 1): (.022, .019, .978), ("dln_consump", 2): (.015, .020, 1.009)}
    for key, (forecast, half, mse) in published.items():
        row = rows[key]
        assert row.forecast == pytest.approx(forecast, abs=6e-4)
        assert row.std_error ** 2 * 1e4 == pytest.approx(mse, abs=6e-3 if mse > 2 else 6e-4)
        assert (row.ci_high - row.ci_low) / 2 == pytest.approx(half, abs=6e-4)
    # Without the estimation term the first MSE would be Sigma itself: 21.30, not 23.34.
    assert rows[("dln_inv", 1)].std_error ** 2 * 1e4 > 23.0


@pytest.mark.parametrize("options, innovation, estimation", [
    ({}, 1.0, 1.0), ({"dfk": True}, "dfk", "dfk"), ({"small": True}, 1.0, "dfk"),
    ({"small": True, "dfk": True}, "dfk", "dfk")])
def test_one_step_forecast_mse_has_a_closed_form(options, innovation, estimation):
    """h = 1: Omega(1) / T = (m / T) V-scale * Sigma, because sum_t z_t'(Z'Z)^-1 z_t = m.

    The innovation part uses the model's Sigma (divisor T, or T - m with dfk);
    the estimation part the Sigma of the coefficient covariance (T - m with
    small or dfk).
    """
    frame = simulated()
    result = oe.var(data=frame, y=["a", "b", "c"], lags=2, **options)
    t, m = result.metrics["T"], result.metrics["df_eq"]
    assert (t, m) == (258, 7)
    sigma_ml = np.diag(np.array(result.extra["sigma_ml"]))
    scale = {1.0: 1.0, "dfk": t / (t - m)}
    expected = sigma_ml * (scale[innovation] + scale[estimation] * m / t)
    table = oe.forecast(result, 1)
    assert_allclose(table.std_error.to_numpy() ** 2, expected, rtol=1e-10)


# ---- [TS] vec intro: vecrank dallas houston; vecrank austin dallas houston sa, lag(3) --------

def test_vec_intro_tables_follow_the_same_arithmetic_and_critical_values():
    """The printed tables (data: txhprice, not available offline) against OpenEcon's rules.

    Parameter counts ``K m2 + (K + m1 - r) r``, trace statistics ``2 {LL(K) - LL(r)}``
    from the printed log likelihoods, and the 5% critical values of
    Osterwald-Lenum's Table 1 for K - r = 4, 3, 2, 1 (47.21 is the one entry of
    dimension 4 that the Stata manuals print).
    """
    # vecrank dallas houston: K = 2, two lags, unrestricted constant: m1 = 2, m2 = 2 + 1.
    assert [2 * 3 + (2 + 2 - r) * r for r in range(3)] == [6, 9, 10]
    ll = [576.26444, 599.58781, 599.67706]
    assert_allclose([2 * (ll[2] - value) for value in ll[:2]], [46.8252, 0.1785], atol=6e-5)
    assert -166 * np.log(1 - np.array([0.24498, 0.00107])).sum() == pytest.approx(46.8252,
                                                                              abs=2e-3)
    # vecrank austin dallas houston sa, lag(3): K = 4, m1 = 4, m2 = 4 * 2 + 1.
    assert [4 * 9 + (4 + 4 - r) * r for r in range(5)] == [36, 43, 48, 51, 52]
    ll = [1107.7833, 1137.7484, 1153.6435, 1158.4191, 1158.5868]
    assert_allclose([2 * (ll[4] - value) for value in ll[:4]],
                    [101.6070, 41.6768, 9.8865, 0.3354], atol=2e-4)
    printed = [47.21, 29.68, 15.41, 3.76]
    assert [critical_values.lookup("constant", d, "trace", 5) for d in (4, 3, 2, 1)] == printed

    # The same layout from a fit: four variables, three lags.
    rng = np.random.default_rng(8)
    walk = rng.normal(size=(200, 4)).cumsum(axis=0)
    frame = pd.DataFrame(walk, columns=["austin", "dallas", "houston", "sa"])
    frame["dallas"] = frame["austin"] + rng.normal(size=200)
    table = oe.vecrank(data=frame, y=list(frame.columns), lags=3)
    assert list(table["parms"]) == [36, 43, 48, 51, 52]
    assert list(table["trace_cv5"][:4]) == printed and table.attrs["n_obs"] == 197
    assert_allclose(table["trace"][:4], 2 * (table["ll"][4] - table["ll"][:4]), rtol=1e-9)
    assert_allclose(table["max"][:4], 2 * np.diff(table["ll"]), rtol=1e-8, atol=1e-9)
    assert table.attrs["critical_values_flag"] is None


# ---- generalized responses: Pesaran and Shin (1998) ------------------------------------------

@pytest.mark.parametrize("options", [{}, {"dfk": True}, {"covariance": "robust"}])
def test_generalized_responses_are_orthogonalized_responses_with_the_impulse_first(options):
    """psi_j(n) = Phi_n Sigma e_j / sqrt(sigma_jj) is the Cholesky response to variable j when
    j is ordered first. Estimates AND delta-method standard errors must agree, which ties
    the generalized standard errors to the orthogonalized ones (those reproduce the output
    printed in [TS] irf table)."""
    frame = simulated(seed=21)
    names = ["a", "b", "c"]
    steps = 6
    base = oe.irf(oe.var(data=frame, y=names, lags=2, **options), steps=steps,
                  kind="generalized")
    for impulse in names:
        order = [impulse] + [name for name in names if name != impulse]
        other = oe.irf(oe.var(data=frame, y=order, lags=2, **options), steps=steps,
                       kind="orthogonalized")
        for response in names:
            ours = base[(base.impulse == impulse) & (base.response == response)]
            theirs = other[(other.impulse == impulse) & (other.response == response)]
            assert list(ours.step) == list(theirs.step) == list(range(steps + 1))
            assert_allclose(ours.irf, theirs.irf, rtol=1e-9, atol=1e-12)
            assert_allclose(ours.std_error, theirs.std_error, rtol=1e-8, atol=1e-12)


def test_response_identities():
    frame = simulated(seed=22)
    names = ["a", "b", "c"]
    result = oe.var(data=frame, y=names, lags=2, irf_steps=10)
    record = result.extra["irf"]
    sigma = np.array(result.extra["sigma"])
    simple, orth, fevd = (np.array(record[key]) for key in ("simple", "orthogonalized", "fevd"))
    assert_allclose(simple[0], np.eye(3), atol=1e-14)
    assert_allclose(orth[0], np.linalg.cholesky(sigma), atol=1e-12)
    # sum_i Theta_i Theta_i' = sum_i Phi_i Sigma Phi_i' and the shares add to one.
    assert_allclose(np.einsum("iab,icb->ac", orth, orth),
                    np.einsum("iab,bc,idc->ad", simple, sigma, simple), rtol=1e-10)
    assert_allclose(fevd[1:].sum(axis=2), 1.0, atol=1e-12)
    assert np.all(fevd[0] == 0.0) and np.all(fevd >= 0.0)
    # The h-step forecast-error variance shares come from the first h responses.
    cumulative = np.cumsum(orth ** 2, axis=0)
    assert_allclose(fevd[1:], (cumulative / cumulative.sum(axis=2, keepdims=True))[:-1],
                    rtol=1e-10)


# ---- statsmodels with other deterministic terms ---------------------------------------------

@pytest.mark.parametrize("deterministic, options", [
    ("ct", {"trend": True}), ("n", {"constant": False}), ("c", {})])
def test_deterministic_terms_match_statsmodels(deterministic, options):
    frame = simulated(seed=23)
    if deterministic == "ct":
        frame = frame + 0.01 * np.arange(len(frame))[:, None]
    names = ["a", "b", "c"]
    result = oe.var(data=frame, y=names, lags=2, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = VAR(frame[names]).fit(2, trend=deterministic)
    t, k, m = reference.nobs, 3, reference.params.shape[0]
    assert (result.nobs, result.metrics["df_eq"]) == (t, m)
    estimate = {c.term: c.estimate for c in result.coefficients}
    error = {c.term: c.std_error for c in result.coefficients}
    label = {"const": "Intercept", "trend": "trend"}
    for row in reference.params.index:
        term = label.get(row) or f"L{row[1]}.{row[3:]}"
        for equation in names:
            assert estimate[f"{equation}:{term}"] == pytest.approx(
                reference.params.loc[row, equation], rel=1e-8, abs=1e-10)
            # statsmodels divides by T - m; the default here is the ML divisor T.
            assert error[f"{equation}:{term}"] == pytest.approx(
                reference.stderr.loc[row, equation] * np.sqrt((t - m) / t), rel=1e-8)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-10)
    assert_allclose(np.array(result.extra["sigma"]), reference.sigma_u_mle, rtol=1e-9)
    assert result.metrics["det_sigma_ml"] == pytest.approx(
        np.linalg.det(reference.sigma_u_mle), rel=1e-9)
    # statsmodels' (Lutkepohl's) criteria drop the constant of the likelihood.
    constant = k * (1 + np.log(2 * np.pi))
    assert result.metrics["aic_per_obs"] - constant == pytest.approx(reference.aic, rel=1e-9)
    assert result.metrics["sbic_per_obs"] - constant == pytest.approx(reference.bic, rel=1e-9)
    assert result.metrics["hqic_per_obs"] - constant == pytest.approx(reference.hqic, rel=1e-9)
    assert result.extra["stability"]["stable"] == reference.is_stable()
    forecast = oe.forecast(result, 4)
    theirs = reference.forecast(frame[names].to_numpy()[-2:], 4)
    for v, name in enumerate(names):
        assert_allclose(forecast[forecast.variable == name].forecast, theirs[:, v], rtol=1e-8)


# ---- the order of the variables --------------------------------------------------------------

def test_reduced_form_statistics_do_not_depend_on_the_order_of_the_variables():
    frame = simulated(seed=24)
    names, other = ["a", "b", "c"], ["c", "a", "b"]
    first = oe.var(data=frame, y=names, lags=2, lm_lags=3)
    second = oe.var(data=frame, y=other, lags=2, lm_lags=3)
    for key in ("log_likelihood", "aic", "bic", "hqic", "fpe", "det_sigma_ml"):
        assert second.metrics[key] == pytest.approx(first.metrics[key], rel=1e-10)
    one = {c.term: (c.estimate, c.std_error) for c in first.coefficients}
    two = {c.term: (c.estimate, c.std_error) for c in second.coefficients}
    assert set(one) == set(two)
    for term, values in one.items():
        assert_allclose(two[term], values, rtol=1e-8, atol=1e-12)
    granger = {(row["equation"], row["excluded"]): row["statistic"]
               for row in first.extra["granger"]}
    for row in second.extra["granger"]:
        assert row["statistic"] == pytest.approx(granger[(row["equation"], row["excluded"])],
                                                 rel=1e-8)
    for key in ("lag_exclusion_L1", "lag_exclusion_L2", "lm_autocorrelation_L1",
                "lm_autocorrelation_L3"):
        assert second.tests[key]["statistic"] == pytest.approx(first.tests[key]["statistic"],
                                                               rel=1e-7)
    assert_allclose(sorted(r["modulus"] for r in second.extra["stability"]["eigenvalues"]),
                    sorted(r["modulus"] for r in first.extra["stability"]["eigenvalues"]),
                    rtol=1e-9)
    # The joint normality test orthogonalizes by a Cholesky factor: it DOES depend on the
    # order (as Stata's varnorm does); the generalized responses do not.
    assert second.tests["normality"]["statistic"] != pytest.approx(
        first.tests["normality"]["statistic"], rel=1e-6)
    a, b = (np.array(fit.extra["irf"]["generalized"]) for fit in (first, second))
    index = [other.index(name) for name in names]
    assert_allclose(b[:, index][:, :, index], a, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("trend", ["none", "rconstant", "constant", "rtrend", "trend"])
def test_johansen_statistics_do_not_depend_on_the_order_of_the_variables(trend):
    rng = np.random.default_rng(25)
    walk = rng.normal(size=(180, 2)).cumsum(axis=0)
    frame = pd.DataFrame({"a": walk[:, 0] + rng.normal(size=180), "b": walk[:, 1],
                          "c": 0.5 * walk[:, 0] - walk[:, 1] + rng.normal(size=180),
                          "d": walk[:, 0] + 0.1 * np.arange(180) + rng.normal(size=180)})
    names, other = ["a", "b", "c", "d"], ["d", "b", "a", "c"]
    first = oe.vecrank(data=frame, y=names, lags=3, trend=trend)
    second = oe.vecrank(data=frame, y=other, lags=3, trend=trend)
    for column in ("ll", "trace", "max", "eigenvalue", "aic"):
        assert_allclose(second[column].to_numpy(dtype=float), first[column].to_numpy(dtype=float),
                        rtol=1e-8, equal_nan=True)
    assert second.attrs["selected_rank"] == first.attrs["selected_rank"]
    # The fitted VEC model: the same likelihood and the same Pi = alpha beta', permuted.
    one = oe.vec(data=frame, y=names, lags=3, rank=2, trend=trend)
    two = oe.vec(data=frame, y=other, lags=3, rank=2, trend=trend)
    assert two.metrics["log_likelihood"] == pytest.approx(one.metrics["log_likelihood"],
                                                          rel=1e-10)
    index = [other.index(name) for name in names]
    assert_allclose(np.array(two.extra["pi"])[index][:, index], np.array(one.extra["pi"]),
                    rtol=1e-6, atol=1e-9)
    assert_allclose(np.array(two.extra["omega"])[index][:, index], np.array(one.extra["omega"]),
                    rtol=1e-7)
    forecast_one, forecast_two = oe.forecast(one, 5), oe.forecast(two, 5)
    for name in names:
        for column in ("forecast", "std_error"):
            assert_allclose(forecast_two[forecast_two.variable == name][column],
                            forecast_one[forecast_one.variable == name][column], rtol=1e-6)


# ---- sample bookkeeping ----------------------------------------------------------------------

def test_a_filtered_frame_with_its_own_index_and_dropped_ends():
    """Row labels are not positions: a filtered frame keeps its original index."""
    frame = simulated(seed=26, n=150)
    frame["t"] = np.arange(1000, 1150)
    frame.loc[[0, 1, 149], "a"] = np.nan
    reference = oe.var(data=frame.iloc[2:149].reset_index(drop=True), y=["a", "b"], time="t",
                       lags=2)
    scrambled = frame.sample(frac=1.0, random_state=3)
    scrambled.index = scrambled.index * 7 + 5
    result = oe.var(data=scrambled, y=["a", "b"], time="t", lags=2, missing="drop")
    assert result.nobs == reference.nobs == 145
    assert_allclose([c.estimate for c in result.coefficients],
                    [c.estimate for c in reference.coefficients], rtol=1e-12)
    assert list(oe.forecast(result, 2).period[:2]) == [1149, 1150]
    with pytest.raises(oe.analysis.AnalysisError) as error:
        oe.var(data=scrambled.assign(b=scrambled.b.where(scrambled.t != 1070)), y=["a", "b"],
               time="t", lags=2, missing="drop")
    assert error.value.code == "time_gaps"


# ---- regression: defects fixed in the second verification pass -------------------------------

def test_forecast_rejects_unknown_options_with_an_analysis_error():
    """oe.forecast forwards its keywords: an option the model's forecast function does not
    have (exog for a VEC model, a misspelt name) used to escape as a raw TypeError."""
    rng = np.random.default_rng(27)
    frame = pd.DataFrame(rng.normal(size=(120, 2)).cumsum(axis=0), columns=["a", "b"])
    var, vec = oe.var(data=frame, y=["a", "b"]), oe.vec(data=frame, y=["a", "b"])
    calls = [lambda: oe.forecast(var, 3, level=95), lambda: oe.var_forecast(var, 3, bogus=1),
             lambda: oe.forecast(vec, 3, exog=None), lambda: oe.vec_forecast(vec, 3, exog=frame)]
    for call in calls:
        with pytest.raises(oe.analysis.AnalysisError) as error:
            call()
        assert error.value.code == "invalid_option" and "accepts" in str(error.value)
    assert len(oe.forecast(var, 3, alpha=0.1)) == len(oe.forecast(vec, 3, data=frame)) == 6


def test_wrongly_typed_arguments_raise_analysis_errors_not_raw_exceptions():
    """irf_kinds=5 (TypeError), vecrank(lags=None) (TypeError) and vecrank(trend=None)
    (KeyError) used to escape; None now means the documented default."""
    rng = np.random.default_rng(28)
    frame = pd.DataFrame(rng.normal(size=(120, 3)).cumsum(axis=0), columns=["a", "b", "c"])
    names = ["a", "b", "c"]
    with pytest.raises(oe.analysis.AnalysisError) as error:
        oe.var(data=frame, y=names, irf_kinds=5)
    assert error.value.code == "invalid_spec" and "irf_kinds" in str(error.value)
    default = oe.vecrank(data=frame, y=names)
    for options in ({"lags": None}, {"trend": None}, {"lags": None, "trend": None}):
        table = oe.vecrank(data=frame, y=names, **options)
        assert (table.attrs["lags"], table.attrs["trend"]) == (2, "constant")
        assert_allclose(table["trace"][:3], default["trace"][:3], rtol=1e-12)
        assert list(table["trace_cv5"][:3]) == [29.68, 15.41, 3.76]


@pytest.mark.parametrize("function", ["var", "varsoc", "vecrank", "vec"])
def test_an_unordered_collection_of_variables_is_refused(function):
    """The order of y is part of the model (Cholesky order, Johansen normalization): a set,
    whose iteration order can change between runs, used to be accepted silently."""
    rng = np.random.default_rng(29)
    frame = pd.DataFrame(rng.normal(size=(120, 2)).cumsum(axis=0), columns=["a", "b"])
    for value in ({"a", "b"}, frozenset(["a", "b"]), {"a": 1, "b": 2}):
        with pytest.raises(oe.analysis.AnalysisError) as error:
            getattr(oe, function)(data=frame, y=value)
        assert error.value.code == "invalid_spec" and "ordered list" in str(error.value)
    getattr(oe, function)(data=frame, y=("a", "b"))          # a tuple is ordered: accepted


def test_variable_names_that_are_not_text_get_a_message_that_says_so():
    rng = np.random.default_rng(30)
    frame = pd.DataFrame(rng.normal(size=(60, 2)), columns=[0, 1])
    for value, fragment in (([0, 1], "non-empty text; got 0, 1"), (["a", " "], "got ' '"),
                            ([], "at least one column name")):
        with pytest.raises(oe.analysis.AnalysisError) as error:
            oe.var(data=frame, y=value)
        assert error.value.code == "invalid_spec" and fragment in str(error.value)


def test_vec_says_when_the_intercept_field_of_a_hand_built_spec_is_ignored():
    """The deterministic terms of vec come from trend=; intercept=False used to be dropped
    without a word although trend='constant' keeps the constant."""
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(rng.normal(size=(120, 2)).cumsum(axis=0), columns=["a", "b"])
    spec = dict(estimator="vec", outcome="a", columns={"system": ["a", "b"]})
    plain = oe.fit(oe.ModelSpec(**spec), data=frame)
    ignored = oe.fit(oe.ModelSpec(**spec, intercept=False), data=frame)
    assert not any("intercept=False" in text for text in plain.warnings)
    assert any("intercept=False is ignored" in text for text in ignored.warnings)
    assert [c.term for c in ignored.coefficients] == [c.term for c in plain.coefficients]
    assert "D_a:Intercept" in [c.term for c in ignored.coefficients]
    consistent = oe.fit(oe.ModelSpec(**spec, intercept=False, options={"trend": "rconstant"}),
                        data=frame)
    assert not any("intercept=False" in text for text in consistent.warnings)
