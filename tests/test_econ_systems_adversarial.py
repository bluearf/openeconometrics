"""Adversarial inputs for the systems family (verification suite).

Every case either produces the correct estimate (checked against a simpler
equivalent model or an exact invariance) or raises ``AnalysisError`` with the
documented code; no raw exception, NaN or silently changed model.
"""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError

EQS = [{"y": "y1", "x": ["a"]}, {"y": "y2", "x": ["b"]}]


@pytest.fixture(scope="module")
def d():
    rng = np.random.default_rng(101)
    n = 60
    frame = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "z"])
    frame["y1"] = 1 + frame.a + rng.normal(size=n)
    frame["y2"] = -1 + frame.b + 0.5 * frame.y1 + rng.normal(size=n)
    frame["const"] = 2.5
    frame["s"] = "text"
    frame["id"] = np.repeat(np.arange(6), 10)
    frame["t"] = np.tile(np.arange(10), 6) + 2000
    frame["yf"] = 1 + frame.a + rng.normal(0, 0.3, n) - np.abs(rng.normal(0, 0.8, n))
    frame["g"] = rng.integers(0, 12, n)
    frame["w"] = rng.uniform(0.5, 2.0, n)
    return frame


def code(call):
    with pytest.raises(AnalysisError) as error:
        call()
    message = str(error.value)
    assert message and "Traceback" not in message
    return error.value.code


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


# ---- regressions of fixed defects -----------------------------------------------------------


def test_constant_regressor_is_omitted_not_estimated(d):
    # Before the fix the compressed (centred) constant column was QR rounding noise that the
    # screen kept, giving coefficients of order 1e14.
    eqs = [{"y": "y1", "x": ["a", "const"]}, {"y": "y2", "x": ["b"]}]
    result = oe.sureg(data=d, equations=eqs)
    assert "y1:const" in result.provenance["omitted_terms"]
    reference = oe.sureg(data=d, equations=EQS)
    assert_allclose(estimates(result), estimates(reference), rtol=1e-10)
    mv = oe.mvreg(data=d, y=["y1", "y2"], x=["a", "const"])
    assert {"y1:const", "y2:const"} <= set(mv.provenance["omitted_terms"])
    # Without a constant the constant regressor is the intercept and is kept.
    kept = oe.sureg(data=d, equations=[{"y": "y1", "x": ["const", "a"], "constant": False},
                                       {"y": "y2", "x": ["b"]}])
    assert_allclose(estimates(kept)[0] * 2.5, estimates(reference)[0], rtol=1e-9)
    # A constant instrument of reg3 is dropped, the estimates are unchanged.
    r3 = [{"y": "y1", "x": ["y2", "a"]}, {"y": "y2", "x": ["y1", "b"]}]
    with_const = oe.reg3(data=d, equations=r3, exogenous=["const", "c"])
    plain = oe.reg3(data=d, equations=r3, exogenous=["c"])
    assert "const" in with_const.provenance["omitted_terms"]
    assert_allclose(estimates(with_const), estimates(plain), rtol=1e-10)


def test_constant_outcome_and_malformed_arguments(d):
    assert code(lambda: oe.sureg(data=d, equations=[{"y": "const", "x": ["a"]},
                                                    {"y": "y2", "x": ["b"]}])) == "constant_outcome"
    assert code(lambda: oe.sureg(data=d, equations=["y1 a", "y2 b"])) == "invalid_spec"
    assert code(lambda: oe.sureg(data=d, equations=None)) == "invalid_spec"
    assert code(lambda: oe.reg3(data=d, equations=[{"y": "y1"}, 5])) == "invalid_spec"
    assert code(lambda: oe.sureg(data=d, equations=EQS, constraints=["a = b"])) \
        == "invalid_constraint"
    assert code(lambda: oe.sureg(data=d, equations=EQS, constraints={"terms": {}})) \
        == "invalid_constraint"


def test_gmm_exact_fit_is_refused(d):
    moments = ["y1 - {b0} - {b1}*a"]
    assert code(lambda: oe.gmm(data=d, moments=["const - {b0} - {b1}*a"],
                               instruments=["a"])) == "perfect_fit"
    assert code(lambda: oe.gmm(data=d.assign(y1=1 + 2 * d.a), moments=moments,
                               instruments=["a", "b"])) == "perfect_fit"


def test_gmm_large_offsets_and_scales_are_exact_reparameterizations():
    # Normal equations squared the condition number and the iteration never settled with
    # an offset of 1e6; QR steps on orthonormal instrument bases recover the invariance.
    rng = np.random.default_rng(102)
    n = 300
    z1, z2, x1, v = rng.normal(size=(4, n))
    x2 = 0.5 * z1 + 0.5 * z2 + v
    frame = pd.DataFrame({"y": 1 + 0.5 * x1 - x2 + 0.8 * v + rng.normal(size=n),
                          "x1": x1, "x2": x2, "z1": z1, "z2": z2})
    moments = ["y - {b0} - {b1}*x1 - {b2}*x2"]
    kw = {"instruments": ["x1", "z1", "z2"], "winitial": "unadjusted"}
    base = oe.gmm(data=frame, moments=moments, **kw)
    shifted = oe.gmm(data=frame.assign(x1=frame.x1 + 1e6), moments=moments, **kw)
    assert_allclose(estimates(shifted)[1:], estimates(base)[1:], rtol=1e-8)
    assert_allclose(errors(shifted)[1:], errors(base)[1:], rtol=1e-7)
    assert_allclose(shifted.tests["hansen_j"]["statistic"], base.tests["hansen_j"]["statistic"],
                    rtol=1e-6)
    tiny = oe.gmm(data=frame.assign(y=frame.y * 1e-8), moments=moments, **kw)
    assert_allclose(estimates(tiny), estimates(base) * 1e-8, rtol=1e-8)
    assert_allclose(errors(tiny), errors(base) * 1e-8, rtol=1e-8)
    huge = oe.gmm(data=frame.assign(x2=frame.x2 * 1e8), moments=moments, **kw)
    assert_allclose(estimates(huge)[2], estimates(base)[2] * 1e-8, rtol=1e-8)


def test_frontier_degenerate_samples(d):
    assert code(lambda: oe.frontier(data=d, y="const", x=["a"])) == "constant_outcome"
    assert code(lambda: oe.frontier(data=d.assign(yf=1 + 2 * d.a), y="yf", x=["a"])) \
        == "perfect_fit"
    # Wrong-skewed residuals: the half-normal and exponential MLE of sigma_u is zero.
    for dist in ("hnormal", "exponential"):
        assert code(lambda: oe.frontier(data=d, y="yf", x=["a"], distribution=dist,
                                        cost=True)) == "boundary_solution"
    # A deterministic frontier (no noise) drives sigma_v to zero.
    exact = d.assign(yf=1 + d.a - np.abs(np.random.default_rng(3).normal(size=len(d))))
    assert code(lambda: oe.frontier(data=exact, y="yf", x=["a"])) == "boundary_solution"
    assert code(lambda: oe.frontier(data=exact, y="yf", x=["a"], distribution="tnormal")) \
        == "boundary_solution"


# ---- failure contract -----------------------------------------------------------------------


@pytest.mark.parametrize("label,call,expected", [
    ("empty", lambda d: oe.sureg(data=d.head(0), equations=EQS), "empty_data"),
    ("n<=k", lambda d: oe.sureg(data=d.head(2), equations=EQS), "insufficient_observations"),
    ("one row", lambda d: oe.sureg(data=d.head(1), equations=EQS), "insufficient_observations"),
    ("all missing", lambda d: oe.sureg(data=d.assign(a=np.nan), equations=EQS, missing="drop"),
     "empty_sample"),
    ("missing raise", lambda d: oe.sureg(data=d.assign(a=np.where(d.index == 0, np.nan, d.a)),
                                         equations=EQS), "missing_values"),
    ("infinite", lambda d: oe.sureg(data=d.assign(a=np.where(d.index == 0, np.inf, d.a)),
                                    equations=EQS), "non_finite_values"),
    ("string", lambda d: oe.sureg(data=d, equations=[{"y": "y1", "x": ["s"]}, EQS[1]]),
     "non_numeric_column"),
    ("one equation", lambda d: oe.sureg(data=d, equations=EQS[:1]), "invalid_spec"),
    ("duplicate labels", lambda d: oe.sureg(data=d, equations=[EQS[0], EQS[0]]), "invalid_spec"),
    ("identical equations", lambda d: oe.sureg(
        data=d, equations=[EQS[0], {**EQS[0], "name": "copy"}]), "singular_sigma"),
    ("outcome as regressor", lambda d: oe.sureg(data=d, equations=[{"y": "y1", "x": ["y1"]},
                                                                   EQS[1]]), "invalid_spec"),
    ("categorical outcome", lambda d: oe.sureg(data=d, equations=EQS, categorical=["y1"]),
     "invalid_spec"),
    ("negative weights", lambda d: oe.sureg(data=d.assign(w=-d.w), equations=EQS, weights="w",
                                            weight_type="aweight"), "negative_weights"),
    ("zero weights", lambda d: oe.sureg(data=d.assign(w=0.0), equations=EQS, weights="w",
                                        weight_type="fweight"), "empty_sample"),
    ("fractional fweights", lambda d: oe.sureg(data=d, equations=EQS, weights="w",
                                               weight_type="fweight"),
     "noninteger_frequency_weights"),
    ("pweights", lambda d: oe.sureg(data=d, equations=EQS, weights="w", weight_type="pweight"),
     "invalid_spec"),
    ("robust", lambda d: oe.sureg(data=d, equations=EQS, covariance="robust"), "invalid_spec"),
    ("iterate limit", lambda d: oe.sureg(data=d, equations=EQS, iterate=True, max_iterations=1),
     "nonconvergence"),
    ("unknown constraint term", lambda d: oe.sureg(
        data=d, equations=EQS, constraints=[{"terms": {"y1:zz": 1}, "value": 0}]),
     "invalid_constraint"),
    ("contradictory constraints", lambda d: oe.sureg(data=d, equations=EQS, constraints=[
        {"terms": {"y1:a": 1}, "value": 0}, {"terms": {"y1:a": 1}, "value": 1}]),
     "inconsistent_constraints"),
    ("mvreg one outcome", lambda d: oe.mvreg(data=d, y=["y1"], x=["a"]), "invalid_spec"),
    ("mvreg y string", lambda d: oe.mvreg(data=d, y="y1", x=["a"]), "invalid_spec"),
    ("mvreg constant outcome", lambda d: oe.mvreg(data=d, y=["y1", "const"], x=["a"]),
     "constant_outcome"),
    ("reg3 underidentified", lambda d: oe.reg3(data=d, equations=[
        {"y": "y1", "x": ["y2", "a"]}, {"y": "y2", "x": ["y1", "a"]}]), "underidentified"),
    ("reg3 no instruments", lambda d: oe.reg3(data=d, equations=[
        {"y": "y1", "x": ["y2"], "constant": False}, {"y": "y2", "x": ["y1"], "constant": False}]),
     "underidentified"),
    ("reg3 ireg3 with 2sls", lambda d: oe.reg3(data=d, equations=EQS, method="2sls", ireg3=True),
     "invalid_spec"),
    ("reg3 inst and endog", lambda d: oe.reg3(data=d, equations=EQS, instruments=["a"],
                                              endogenous=["b"]), "invalid_spec"),
    ("reg3 unused endog", lambda d: oe.reg3(data=d, equations=EQS, endogenous=["c"]),
     "invalid_spec"),
    ("reg3 outcome as instrument", lambda d: oe.reg3(data=d, equations=EQS, exogenous=["y1"]),
     "invalid_spec"),
    ("gmm injection", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - __import__('os')"]),
     "invalid_formula"),
    ("gmm unknown column", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*zz"]),
     "missing_columns"),
    ("gmm underidentified", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"]),
     "underidentified"),
    ("gmm not identified", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a - {b2}*a"],
                                            instruments=["a", "b"]), "not_identified"),
    ("gmm overflow at start", lambda d: oe.gmm(data=d.assign(a=d.a * 1000),
                                               moments=["y1 - exp({b0} + {b1}*a)"],
                                               instruments=["a"], start={"b1": 1}),
     "invalid_start"),
    ("gmm one cluster", lambda d: oe.gmm(data=d.assign(g=1), moments=["y1 - {b0} - {b1}*a"],
                                         instruments=["a"], cluster="g"), "insufficient_clusters"),
    ("gmm fewer clusters than moments", lambda d: oe.gmm(
        data=d.assign(g=np.arange(len(d)) % 2), moments=["y1 - {b0} - {b1}*a"],
        instruments=["a", "b"], cluster="g"), "singular_weight_matrix"),
    ("gmm hac without lags", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"],
                                              instruments=["a"], wmatrix="hac"), "invalid_spec"),
    ("gmm hac repeated time", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"],
                                               instruments=["a"], wmatrix="hac", lags=1,
                                               time="t"), "repeated_time_values"),
    ("gmm kernel without hac", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"],
                                                instruments=["a"], kernel="parzen"),
     "invalid_spec"),
    ("gmm unknown start", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"],
                                           instruments=["a"], start={"zz": 1}), "invalid_start"),
    ("gmm text start", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"],
                                        instruments=["a"], start={"b0": "x"}), "invalid_start"),
    ("gmm n<=p", lambda d: oe.gmm(data=d.head(2), moments=["y1 - {b0} - {b1}*a"],
                                  instruments=["a"]), "insufficient_observations"),
    ("gmm moments string", lambda d: oe.gmm(data=d, moments="y1 - {b0}"), "invalid_spec"),
    ("gmm pweights nonrobust", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"],
                                                instruments=["a"], weights="w",
                                                weight_type="pweight", covariance="nonrobust"),
     "unsupported_covariance"),
    ("gmm igmm limit", lambda d: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a - {b2}*b"],
                                        instruments=["a", "b", "c"], igmm=True,
                                        igmm_max_iterations=1), "nonconvergence"),
    ("frontier n<=k", lambda d: oe.frontier(data=d.head(3), y="yf", x=["a"]),
     "insufficient_observations"),
    ("frontier pweights nonrobust", lambda d: oe.frontier(data=d, y="yf", x=["a"], weights="w",
                                                          weight_type="pweight",
                                                          covariance="nonrobust"),
     "unsupported_covariance"),
    ("frontier aweights", lambda d: oe.frontier(data=d, y="yf", x=["a"], weights="w",
                                                weight_type="aweight"), "invalid_spec"),
    ("xtgls repeated time", lambda d: oe.xtgls(data=d.assign(t=0), y="y1", x=["a"], panel="id",
                                               time="t"), "repeated_time_values"),
    ("xtgls correlated T<m", lambda d: oe.xtgls(data=d[d.t < 2004], y="y1", x=["a"], panel="id",
                                                time="t", panels="correlated"),
     "insufficient_periods"),
    ("xtgls correlated unbalanced", lambda d: oe.xtgls(data=d.iloc[1:], y="y1", x=["a"],
                                                       panel="id", time="t",
                                                       panels="correlated"), "unbalanced_panel"),
    ("xtgls ar1 gaps", lambda d: oe.xtgls(data=d[d.t != 2003], y="y1", x=["a"], panel="id",
                                          time="t", corr="ar1"), "time_gaps"),
    ("xtgls fractional time", lambda d: oe.xtgls(data=d.assign(t=d.t + 0.5), y="y1", x=["a"],
                                                 panel="id", time="t"), "invalid_time"),
    ("xtgls psar1 singleton", lambda d: oe.xtgls(
        data=pd.concat([d, d.head(1).assign(id=99)]), y="y1", x=["a"], panel="id", time="t",
        corr="psar1"), "insufficient_observations"),
    ("xtgls nagar out of range", lambda d: oe.xtgls(
        data=d.assign(y1=d.groupby("id").c.cumsum()), y="y1", x=["a"], panel="id", time="t",
        corr="psar1", rhotype="nagar"), "invalid_rho"),
    ("xtgls perfect fit", lambda d: oe.xtgls(data=d.assign(y1=2 * d.a), y="y1", x=["a"],
                                             panel="id", time="t"), "perfect_fit"),
    ("xtgls igls limit", lambda d: oe.xtgls(data=d, y="y1", x=["a"], panel="id", time="t",
                                            panels="heteroskedastic", igls=True,
                                            max_iterations=1), "nonconvergence"),
    ("xtpcse het and independent", lambda d: oe.xtpcse(data=d, y="y1", x=["a"], panel="id",
                                                       time="t", hetonly=True, independent=True),
     "invalid_spec"),
    ("xtpcse no common period", lambda d: oe.xtpcse(
        data=d[~((d.id == 0) & (d.t > 2004)) & ~((d.id == 1) & (d.t <= 2004))], y="y1", x=["a"],
        panel="id", time="t"), "no_common_periods"),
    ("xtpcse perfect fit", lambda d: oe.xtpcse(data=d.assign(y1=2 * d.a), y="y1", x=["a"],
                                               panel="id", time="t"), "perfect_fit"),
    ("xtpcse covariance", lambda d: oe.xtpcse(data=d, y="y1", x=["a"], panel="id", time="t",
                                              covariance="nonrobust"), "invalid_spec"),
])
def test_failure_contract(d, label, call, expected):
    assert code(lambda: call(d)) == expected


def test_xtpcse_grid_too_large_is_refused():
    n = 40000
    frame = pd.DataFrame({"y": np.random.default_rng(0).normal(size=n),
                          "x": np.random.default_rng(1).normal(size=n),
                          "id": np.arange(n), "t": np.arange(n)})
    assert code(lambda: oe.xtpcse(data=frame, y="y", x=["x"], panel="id", time="t")) \
        == "workspace_limit"


def test_frontier_efficiency_needs_its_own_result_and_data(d):
    fit = oe.frontier(data=d, y="yf", x=["a"])
    assert code(lambda: oe.frontier_efficiency(oe.sureg(data=d, equations=EQS), d)) \
        == "invalid_result"
    assert code(lambda: oe.frontier_efficiency(fit, d.iloc[1:])) == "sample_mismatch"


# ---- extreme magnitudes and degenerate-but-valid samples -------------------------------------


def test_extreme_magnitudes_are_exact_rescalings(d):
    base = oe.sureg(data=d, equations=EQS)
    scaled = oe.sureg(data=d.assign(a=d.a * 1e8, y2=d.y2 * 1e-8), equations=EQS)
    factor = np.array([1, 1e-8, 1e-8, 1e-8])
    assert_allclose(estimates(scaled), estimates(base) * factor, rtol=1e-9)
    assert_allclose(errors(scaled), errors(base) * factor, rtol=1e-9)
    panel = oe.xtgls(data=d, y="y1", x=["a", "b"], panel="id", time="t",
                     panels="heteroskedastic", corr="ar1")
    offset = oe.xtgls(data=d.assign(a=d.a + 1e7), y="y1", x=["a", "b"], panel="id", time="t",
                      panels="heteroskedastic", corr="ar1")
    assert_allclose(estimates(offset)[1:], estimates(panel)[1:], rtol=1e-7)
    assert_allclose(errors(offset)[1:], errors(panel)[1:], rtol=1e-7)
    pcse = oe.xtpcse(data=d, y="y1", x=["a", "b"], panel="id", time="t", correlation="psar1")
    pcse_scaled = oe.xtpcse(data=d.assign(y1=d.y1 * 1e-8), y="y1", x=["a", "b"], panel="id",
                            time="t", correlation="psar1")
    assert_allclose(estimates(pcse_scaled), estimates(pcse) * 1e-8, rtol=1e-9)
    front = oe.frontier(data=d, y="yf", x=["a"])
    big = oe.frontier(data=d.assign(yf=d.yf * 1e8, a=d.a * 1e8), y="yf", x=["a"])
    assert_allclose(estimates(big)[:2], estimates(front)[:2] * np.array([1e8, 1]), rtol=1e-7)
    assert_allclose(estimates(big)[2:], estimates(front)[2:] + np.log(1e16), rtol=1e-8)
    shifted = oe.frontier(data=d.assign(a=d.a + 1e6), y="yf", x=["a"])
    assert_allclose(estimates(shifted)[1:], estimates(front)[1:], rtol=1e-7)


def test_single_panel_and_tiny_samples(d):
    one = d.assign(id=0, t=np.arange(len(d)))
    for result in (oe.xtgls(data=one, y="y1", x=["a"], panel="id", time="t",
                            panels="heteroskedastic"),
                   oe.xtgls(data=one, y="y1", x=["a"], panel="id", time="t", panels="correlated"),
                   oe.xtpcse(data=one, y="y1", x=["a"], panel="id", time="t")):
        ols = np.polyfit(one.a, one.y1, 1)[::-1]
        assert_allclose(estimates(result), ols, rtol=1e-9)
    # Fewer rows than distinct system columns (5 rows, 5 columns): every equation has
    # N > k_i, so SUR is defined (the compressed factor is then trapezoidal).
    tiny = oe.sureg(data=d.head(5), equations=EQS)
    assert np.all(np.isfinite(errors(tiny)))
    x = [np.column_stack([np.ones(5), d.head(5)[c]]) for c in ("a", "b")]
    ys = [d.head(5)[c].to_numpy() for c in ("y1", "y2")]
    e = np.column_stack([y - xi @ np.linalg.lstsq(xi, y, rcond=None)[0] for xi, y in zip(x, ys)])
    si = np.linalg.inv(e.T @ e / 5)
    a = np.block([[si[i, j] * x[i].T @ x[j] for j in range(2)] for i in range(2)])
    rhs = np.concatenate([sum(si[i, j] * x[i].T @ ys[j] for j in range(2)) for i in range(2)])
    assert_allclose(estimates(tiny), np.linalg.solve(a, rhs), rtol=1e-9)
    assert_allclose(errors(tiny), np.sqrt(np.diag(np.linalg.inv(a))), rtol=1e-9)
    six = oe.mvreg(data=d.head(4), y=["y1", "y2", "c"], x=["a", "b"])
    assert six.inference["df_inference"] == 1


def test_options_at_their_bounds(d):
    tight = oe.sureg(data=d, equations=EQS, iterate=True, tolerance=1e-15)
    assert tight.extra["converged"]
    ninety_nine = oe.sureg(data=d, equations=EQS, alpha=0.01)
    c = ninety_nine.coefficients[1]
    assert_allclose(c.ci_high - c.estimate, 2.5758293035489 * c.std_error, rtol=1e-9)
    assert code(lambda: oe.sureg(data=d, equations=EQS, tolerance=0.5)) == "invalid_spec"
    assert code(lambda: oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"], instruments=["a"],
                               lags=-1, wmatrix="hac")) == "invalid_spec"


def test_round_trip_of_every_estimator(d):
    results = [
        oe.sureg(data=d, equations=EQS), oe.mvreg(data=d, y=["y1", "y2"], x=["a"]),
        oe.reg3(data=d, equations=[{"y": "y1", "x": ["y2", "a"]}, {"y": "y2", "x": ["y1", "b"]}]),
        oe.gmm(data=d, moments=["y1 - {b0} - {b1}*a"], instruments=["a", "b"]),
        oe.frontier(data=d, y="yf", x=["a"]),
        oe.xtgls(data=d, y="y1", x=["a"], panel="id", time="t", corr="ar1"),
        oe.xtpcse(data=d, y="y1", x=["a"], panel="id", time="t"),
    ]
    for result in results:
        payload = json.loads(result.model_dump_json())
        assert payload["provenance"]["stata_parity_validated"] is False
        assert result.summary()
        refit = oe.fit(result.spec, data=d)
        assert_allclose(estimates(refit), estimates(result), rtol=1e-12)
