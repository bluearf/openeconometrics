"""Regression tests for defects found and fixed in the verification stage.

1. Regressors with a large level (a date, an identifier-like number) lost
   ``(mean / spread)^2`` digits: standard errors were silently wrong (35% for
   the Heckman two-step at a level of 1e6), ``ivtobit`` stopped converging at
   1e5 and the IV commands raised a raw ``IndexError`` or a false
   ``collinear_endogenous`` at 1e7. Every estimator now centres the regressors
   of equations with a constant and maps the constants back exactly.
2. ``opg`` with analytic / importance weights used ``sum w^2 s s'``; the
   information is linear in the weights (tested in the oracle files).
3. An exactly fitted outcome ended in an opaque ``nonconvergence`` (or, for
   the Heckman two-step, in a "result" with standard errors of 1e-17).
4. A tobit without a single uncensored outcome but censoring on both sides
   was sent to the optimizer; an interval regression with one common
   threshold likewise.
5. NumPy integers were rejected as limits (``ll=np.int64(0)``).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import special

import openecon as oe
from openecon.analysis import AnalysisError


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(424242)
    n = 900
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "z1": rng.normal(size=n), "z2": rng.normal(size=n),
                          "fw": rng.integers(1, 4, size=n).astype(float),
                          "off": rng.normal(scale=0.3, size=n)})
    u = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    frame["latent"] = 0.3 + 0.8 * frame.x1 - 0.5 * frame.x2 + u[:, 0]
    frame["y"] = frame.latent.clip(lower=0.0)
    frame["lo"] = np.floor(frame.latent)
    frame["hi"] = frame.lo + 1.0
    frame["s"] = (0.2 + 0.5 * frame.x1 + 0.8 * frame.z1 + u[:, 1] > 0) * 1.0
    frame["ys"] = np.where(frame.s == 1, frame.latent, np.nan)
    frame["d"] = np.where(frame.s == 1, (frame.latent > 0) * 1.0, np.nan)
    frame["e"] = 0.5 * frame.x1 + 0.7 * frame.z1 - 0.4 * frame.z2 + u[:, 1]
    frame["b"] = (0.2 + 0.5 * frame.x1 - 0.6 * frame.e + u[:, 0] > 0) * 1.0
    frame["t"] = (0.2 + 0.5 * frame.x1 - 0.6 * frame.e + u[:, 0]).clip(lower=0.0)
    return frame


SELECTION = {"x": ["x1", "x2"], "select": "s", "select_x": ["x1", "z1"]}
IV = {"x": ["x1"], "endog": ["e"], "instruments": ["z1", "z2"]}
CALLS = {
    "tobit": lambda d, **o: oe.tobit(data=d, y="y", x=["x1", "x2"], ll=0, **o),
    "truncreg": lambda d, **o: oe.truncreg(data=d, y="latent", x=["x1", "x2"], ll=-1.0, **o),
    "intreg": lambda d, **o: oe.intreg(data=d, y_low="lo", y_high="hi", x=["x1", "x2"], **o),
    "heckman": lambda d, **o: oe.heckman(data=d, y="ys", **SELECTION, **o),
    "heckman_twostep": lambda d, **o: oe.heckman(data=d, y="ys", method="twostep",
                                                 **SELECTION, **o),
    "heckprobit": lambda d, **o: oe.heckprobit(data=d, y="d", **SELECTION, **o),
    "ivprobit": lambda d, **o: oe.ivprobit(data=d, y="b", **IV, **o),
    "ivprobit_twostep": lambda d, **o: oe.ivprobit(data=d, y="b", method="twostep", **IV, **o),
    "ivtobit": lambda d, **o: oe.ivtobit(data=d, y="t", ll=0, **IV, **o),
    "ivtobit_twostep": lambda d, **o: oe.ivtobit(data=d, y="t", ll=0, method="twostep",
                                                 **IV, **o),
}


def frame_of(result):
    return pd.DataFrame({"estimate": [c.estimate for c in result.coefficients],
                         "std_error": [c.std_error for c in result.coefficients]},
                        index=[c.term for c in result.coefficients])


@pytest.fixture(scope="module")
def baseline(data):
    return {name: call(data) for name, call in CALLS.items()}


# ---- 1. regressors with a large level ------------------------------------------------


@pytest.mark.parametrize("name", list(CALLS))
@pytest.mark.parametrize("level", [1e4, 1e6, 1e8])
def test_a_regressor_with_a_large_level_changes_only_the_constants(data, baseline, name, level):
    moved = data.assign(x1=data.x1 + level)
    result, base = CALLS[name](moved), baseline[name]
    got, expected = frame_of(result), frame_of(base)
    slopes = [term for term in expected.index if not term.endswith("Intercept")]
    # The data themselves are only known to level * 2e-16, hence the tolerance at 1e8.
    rtol = 1e-7 if level < 1e7 else 1e-5
    assert_allclose(got.loc[slopes, "estimate"], expected.loc[slopes, "estimate"], rtol=rtol,
                    atol=1e-9)
    assert_allclose(got.loc[slopes, "std_error"], expected.loc[slopes, "std_error"], rtol=rtol)
    if "log_likelihood" in base.metrics:
        assert_allclose(result.metrics["log_likelihood"], base.metrics["log_likelihood"],
                        rtol=1e-9)
    for key in ("model", "rho", "exogeneity"):
        if key in base.tests:
            assert_allclose(result.tests[key]["statistic"], base.tests[key]["statistic"],
                            rtol=10 * rtol, atol=1e-8)
    # Every constant moves by minus level times the coefficient of x1 in its equation.
    for term in expected.index:
        if not term.endswith("Intercept"):
            continue
        partner = term.replace("Intercept", "x1")
        assert_allclose(got.loc[term, "estimate"],
                        expected.loc[term, "estimate"] - level * expected.loc[partner, "estimate"],
                        rtol=rtol)
    assert [w for w in result.warnings if "collinear" in w.lower()] == []


def test_constant_covariance_after_a_level_shift_is_the_exact_linear_map(data, baseline):
    level = 1e5
    result = CALLS["tobit"](data.assign(x1=data.x1 + level), covariance="robust")
    base = CALLS["tobit"](data, covariance="robust")
    jacobian = np.eye(4)
    jacobian[0, 1] = -level
    assert_allclose(np.array(result.covariance_matrix),
                    jacobian @ np.array(base.covariance_matrix) @ jacobian.T, rtol=1e-6,
                    atol=1e-12)


def test_iv_levels_of_instruments_and_endogenous_regressors(data, baseline):
    """Before the fix: IndexError (instrument) and a false collinear_endogenous (endogenous)."""
    level = 1e7
    for name in ("ivprobit", "ivtobit", "ivprobit_twostep", "ivtobit_twostep"):
        expected = frame_of(baseline[name])
        slopes = [term for term in expected.index if not term.endswith("Intercept")]
        for column in ("z1", "e"):
            moved = data.assign(**{column: data[column] + level})
            got = frame_of(CALLS[name](moved))
            assert_allclose(got.loc[slopes, "estimate"], expected.loc[slopes, "estimate"],
                            rtol=1e-6, atol=1e-9)
            assert_allclose(got.loc[slopes, "std_error"], expected.loc[slopes, "std_error"],
                            rtol=1e-6)
    # Shifting the endogenous regressor: outcome constant a - level * b, reduced form + level.
    moved = frame_of(CALLS["ivprobit"](data.assign(e=data.e + level)))
    expected = frame_of(baseline["ivprobit"])
    assert_allclose(moved.loc["Intercept", "estimate"],
                    expected.loc["Intercept", "estimate"] - level * expected.loc["e", "estimate"],
                    rtol=1e-7)
    assert_allclose(moved.loc["e:Intercept", "estimate"],
                    expected.loc["e:Intercept", "estimate"] + level, rtol=1e-12)
    assert_allclose(moved.loc["e:Intercept", "std_error"],
                    expected.loc["e:Intercept", "std_error"], rtol=1e-6)


def test_frequency_weights_and_no_constant_with_centring(data):
    """Centring uses the weighted means; without a constant nothing is centred."""
    replicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for name, call in CALLS.items():
        shifted = data.assign(x1=data.x1 + 1e5)
        weighted = call(shifted, weights="fw", weight_type="fweight")
        expanded = call(replicated.assign(x1=replicated.x1 + 1e5))
        assert_allclose(frame_of(weighted).to_numpy(), frame_of(expanded).to_numpy(), rtol=2e-6,
                        atol=1e-8), name
    plain = oe.tobit(data=data, y="y", x=["x1", "x2"], ll=0, intercept=False)
    assert [c.term for c in plain.coefficients] == ["x1", "x2", "/sigma"]


def test_prediction_sample_uses_the_reported_coefficients_on_the_raw_data(data):
    def rows(result):
        table = pd.DataFrame(result.predictions).set_index("row")
        return table.index.to_numpy(), table

    shifted = data.assign(x1=data.x1 + 50.0, e=data.e - 30.0)
    tobit = oe.tobit(data=shifted, y="y", x=["x1", "x2"], ll=0, offset="off")
    at, table = rows(tobit)
    b = frame_of(tobit).estimate
    index = b["Intercept"] + b["x1"] * shifted.x1 + b["x2"] * shifted.x2 + shifted.off
    assert_allclose(table.fitted, index.to_numpy()[at], rtol=1e-9)
    assert_allclose(table.observed, shifted.y.to_numpy()[at], rtol=1e-12)
    assert_allclose(table.residual, table.observed - table.fitted, atol=1e-12)
    heckman = oe.heckman(data=shifted, y="ys", **SELECTION)
    at, table = rows(heckman)
    g = frame_of(heckman).estimate
    index = g["select:Intercept"] + g["select:x1"] * shifted.x1 + g["select:z1"] * shifted.z1
    assert_allclose(table.fitted, special.ndtr(index.to_numpy()[at]), rtol=1e-8)
    for method in ("ml", "twostep"):
        probit = oe.ivprobit(data=shifted, y="b", method=method, **IV)
        at, table = rows(probit)
        d = frame_of(probit).estimate
        index = d["Intercept"] + d["x1"] * shifted.x1 + d["e"] * shifted.e
        assert_allclose(table.fitted, special.ndtr(index.to_numpy()[at]), rtol=1e-7, atol=1e-12)
        censored = oe.ivtobit(data=shifted, y="t", ll=0, method=method, **IV)
        at, table = rows(censored)
        d = frame_of(censored).estimate
        index = d["Intercept"] + d["x1"] * shifted.x1 + d["e"] * shifted.e
        assert_allclose(table.fitted, index.to_numpy()[at], rtol=1e-7, atol=1e-10)


# ---- 3. exactly fitted outcomes ------------------------------------------------------


def code_of(call):
    with pytest.raises(AnalysisError) as caught:
        call()
    assert len(str(caught.value)) > 20
    return caught.value.code


def test_an_exactly_fitted_outcome_is_reported_as_perfect_fit(data):
    exact = data.assign(y=1 + 2 * data.x1, latent=1 + 2 * data.x1, lo=1 + 2 * data.x1,
                        hi=1 + 2 * data.x1, t=3 + data.x1 - data.e,
                        ys=np.where(data.s == 1, 1 + 2 * data.x1 - data.x2, np.nan))
    constant = data.assign(y=2.0, latent=2.0, lo=2.0, hi=2.0)
    for frame in (exact, constant):
        assert code_of(lambda: oe.tobit(data=frame, y="y", x=["x1", "x2"], ll=-100)) \
            == "perfect_fit"
        assert code_of(lambda: oe.truncreg(data=frame, y="latent", x=["x1", "x2"], ll=-100)) \
            == "perfect_fit"
        assert code_of(lambda: oe.intreg(data=frame, y_low="lo", y_high="hi",
                                         x=["x1", "x2"])) == "perfect_fit"
    for method in ("ml", "twostep"):
        assert code_of(lambda: oe.heckman(data=exact, y="ys", method=method, **SELECTION)) \
            == "perfect_fit"
        assert code_of(lambda: oe.ivtobit(data=exact, y="t", ll=-100, method=method, **IV)) \
            == "perfect_fit"
    # A large level with real noise around it is not a perfect fit.
    noisy = data.assign(y=data.y + 1e8)
    result = oe.tobit(data=noisy, y="y", x=["x1", "x2"], ll=1e8)
    assert_allclose(frame_of(result).loc["/sigma", "estimate"],
                    frame_of(oe.tobit(data=data, y="y", x=["x1", "x2"], ll=0))
                    .loc["/sigma", "estimate"], rtol=1e-6)


# ---- 4. samples without identifying observations -------------------------------------


def test_censoring_without_identifying_observations(data):
    both = data.assign(y=np.where(data.latent > 0.3, 1.0, 0.0))
    assert code_of(lambda: oe.tobit(data=both, y="y", x=["x1"], ll=0, ul=1)) \
        == "no_uncensored_observations"
    assert code_of(lambda: oe.ivtobit(data=both.assign(t=both.y), y="t", ll=0, ul=1, **IV)) \
        == "no_uncensored_observations"
    # intreg: left- and right-censored observations at one common threshold are a probit.
    probit = data.assign(lo=np.where(data.latent > 0, 0.0, np.nan),
                         hi=np.where(data.latent > 0, np.nan, 0.0))
    assert code_of(lambda: oe.intreg(data=probit, y_low="lo", y_high="hi", x=["x1"])) \
        == "no_uncensored_observations"
    # ... but one interval observation identifies the scale and the fit goes through.
    mixed = data.assign(lo=np.where(data.latent > 1, 1.0, np.where(data.latent > 0, 0.0, np.nan)),
                        hi=np.where(data.latent > 1, np.nan, np.where(data.latent > 0, 1.0, 0.0)))
    result = oe.intreg(data=mixed, y_low="lo", y_high="hi", x=["x1", "x2"])
    assert result.metrics["n_interval"] > 0 and result.metrics["n_point"] == 0
    assert result.metrics["n_point"] == result.metrics["n_uncensored"]
    assert_allclose(frame_of(result).loc[["x1", "x2"], "estimate"], [0.8, -0.5], atol=0.15)


# ---- 5. NumPy scalars as limits ------------------------------------------------------


def test_numpy_scalars_are_accepted_as_limits(data):
    reference = frame_of(oe.tobit(data=data, y="y", x=["x1", "x2"], ll=0, ul=3))
    for low, high in ((np.int64(0), np.int32(3)), (np.float64(0.0), np.float32(3.0)),
                      (data.y.min(), 3)):
        result = oe.tobit(data=data, y="y", x=["x1", "x2"], ll=low, ul=high)
        assert_allclose(frame_of(result).to_numpy(), reference.to_numpy(), rtol=1e-12)
        assert result.spec.options == {"ll": 0.0, "ul": 3.0}
    assert oe.truncreg(data=data, y="latent", x=["x1"], ll=np.int64(-1)).nobs \
        == int((data.latent > -1).sum())
    assert oe.ivtobit(data=data, y="t", ll=np.int64(0), **IV).metrics["n_left_censored"] \
        == int((data.t <= 0).sum())
    for bad in (True, np.bool_(False), np.nan, np.inf, "0", [0]):
        assert code_of(lambda: oe.tobit(data=data, y="y", x=["x1"], ll=bad)) == "invalid_spec"


# ---- adversarial inputs: a helpful AnalysisError or a correct fit, never a traceback ----


def test_degenerate_samples_and_columns(data):
    tobit = {"y": "y", "x": ["x1", "x2"], "ll": 0}
    cases = [
        (lambda: oe.tobit(data=data.iloc[:0], **tobit), "empty_data"),
        (lambda: oe.tobit(data=data.iloc[:1], y="latent", x=["x1"]), "insufficient_observations"),
        (lambda: oe.tobit(data=data.iloc[:4], y="latent", x=["x1", "x2"], ul=50),
         "insufficient_observations"),
        (lambda: oe.truncreg(data=data.iloc[:3], y="latent", x=["x1", "x2"]),
         "insufficient_observations"),
        (lambda: oe.heckman(data=data.iloc[:6], y="ys", **SELECTION),
         "insufficient_observations"),
        (lambda: oe.ivprobit(data=data.iloc[:8], y="b", **IV), "insufficient_observations"),
        (lambda: oe.tobit(data=data.assign(x1=np.nan), missing="drop", **tobit), "empty_sample"),
        (lambda: oe.tobit(data=data.assign(y=data.y.astype(str)), **tobit),
         "non_numeric_column"),
        (lambda: oe.tobit(data=data.assign(x1="a"), **tobit), "non_numeric_column"),
        (lambda: oe.tobit(data=data.assign(y=np.where(data.index == 3, np.inf, data.y)),
                          **tobit), "non_finite_values"),
        (lambda: oe.tobit(data=data.assign(w=0.0), weights="w", weight_type="aweight", **tobit),
         "empty_sample"),
        (lambda: oe.tobit(data=data.assign(w=-1.0), weights="w", weight_type="iweight",
                          **tobit), "negative_weights"),
        (lambda: oe.tobit(data=data.assign(w=0.5), weights="w", weight_type="fweight", **tobit),
         "noninteger_frequency_weights"),
        (lambda: oe.tobit(data=data.assign(g=1), cluster="g", **tobit), "insufficient_clusters"),
        (lambda: oe.tobit(data=data.assign(g=np.where(data.index < 3, np.nan, 1.0)),
                          cluster="g", **tobit), "missing_values"),
        (lambda: oe.tobit(data=data, alpha=0.0, **tobit), "invalid_spec"),
        (lambda: oe.tobit(data=data, alpha=1.0, **tobit), "invalid_spec"),
        (lambda: oe.tobit(data=data, y="y", x=["x1"], ll=1, ul=1), "invalid_limits"),
        (lambda: oe.tobit(data=data, y="y", x=["x1"], ll=1e9), "no_uncensored_observations"),
        (lambda: oe.truncreg(data=data, y="latent", x=["x1"], ll=1e9), "empty_sample"),
        (lambda: oe.intreg(data=data.assign(lo=np.nan, hi=np.nan), y_low="lo", y_high="hi",
                           x=["x1"], missing="drop"), "empty_sample"),
        (lambda: oe.intreg(data=data.assign(lo=data.hi + 1), y_low="lo", y_high="hi", x=["x1"]),
         "invalid_interval"),
        (lambda: oe.heckman(data=data.assign(s=1.0, ys=data.latent), y="ys", **SELECTION),
         "constant_selection"),
        (lambda: oe.heckman(data=data.assign(s=data.s * 2), y="ys", **SELECTION),
         "invalid_binary_outcome"),
        (lambda: oe.heckman(data=data.assign(s=data.s.astype(str)), y="ys", **SELECTION),
         "non_numeric_column"),
        (lambda: oe.heckman(data=data.assign(k=data.s), y="ys", x=["x1"], select="s",
                            select_x=["x1", "k"]), "separation_detected"),
        (lambda: oe.heckprobit(data=data.assign(d=data.ys), y="d", **SELECTION),
         "invalid_binary_outcome"),
        (lambda: oe.heckprobit(data=data.assign(d=np.where(data.s == 1, 1.0, np.nan)), y="d",
                               **SELECTION), "constant_outcome"),
        (lambda: oe.ivprobit(data=data.assign(e=data.x1 + data.z1), y="b", **IV),
         "collinear_endogenous"),
        (lambda: oe.ivprobit(data=data.assign(c=2 * data.x1 + 1), y="b", x=["x1"], endog=["e"],
                             instruments=["c"]), "underidentified"),
        (lambda: oe.ivprobit(data=data.assign(e2=data.z2 + data.x2), y="b", x=["x1"],
                             endog=["e", "e2"], instruments=["z1"]), "underidentified"),
        (lambda: oe.ivprobit(data=data, y="b", x=["x1"], endog=["e"], instruments=["x1"]),
         "invalid_spec"),
        (lambda: oe.ivprobit(data=data.assign(b=1.0), y="b", **IV), "constant_outcome"),
        (lambda: oe.ivprobit(data=data.assign(b=(data.e > 0) * 1.0), y="b", **IV),
         "separation_detected"),
        (lambda: oe.ivtobit(data=data, y="t", ll=3, ul=1, **IV), "invalid_limits"),
    ]
    for call, expected in cases:
        assert code_of(call) == expected


def test_singleton_clusters_equal_the_robust_covariance(data):
    """G = N clusters: G/(G-1) times the outer product of single scores = N/(N-1) sandwich."""
    singles = data.assign(g=np.arange(len(data)))
    for name in ("tobit", "truncreg", "intreg", "heckman", "heckprobit", "ivprobit", "ivtobit"):
        robust = CALLS[name](singles, covariance="robust")
        cluster = CALLS[name](singles, cluster="g")
        assert_allclose(np.array(cluster.covariance_matrix), np.array(robust.covariance_matrix),
                        rtol=1e-9, atol=1e-14)
        assert cluster.inference["cluster_count"] == cluster.nobs == robust.nobs


def test_two_way_clustering_is_inclusion_exclusion_of_one_way_fits(data):
    """V(g, h) = V(g) + V(h) - V(g x h) once the common G/(G-1) factor is equalized."""
    frame = data.assign(g=np.arange(len(data)) % 45, h=np.arange(len(data)) % 37)
    frame["gh"] = frame.g * 1000 + frame.h
    for name in ("tobit", "heckman", "ivprobit"):
        def meat_free(result):
            count = result.inference["cluster_count"]
            return np.array(result.covariance_matrix) * (count - 1) / count

        both = CALLS[name](frame, cluster=["g", "h"])
        parts = [CALLS[name](frame, cluster=column) for column in ("g", "h", "gh")]
        expected = meat_free(parts[0]) + meat_free(parts[1]) - meat_free(parts[2])
        smallest = min(parts[0].inference["cluster_count"], parts[1].inference["cluster_count"])
        assert both.inference["cluster_count"] == smallest == 37
        if not both.inference.get("psd_adjusted"):
            assert_allclose(np.array(both.covariance_matrix),
                            expected * smallest / (smallest - 1), rtol=1e-8, atol=1e-13)


def test_tiny_samples_still_give_complete_finite_results(data):
    tiny = data.iloc[:14]
    for name in ("tobit", "truncreg", "intreg", "ivprobit_twostep", "ivtobit_twostep"):
        try:
            result = CALLS[name](tiny)
        except AnalysisError as error:          # a helpful refusal is acceptable, a crash is not
            assert error.code in {"nonconvergence", "separation_detected",
                                  "insufficient_observations", "boundary_solution"}
            continue
        values = frame_of(result).to_numpy()
        assert np.isfinite(values).all() and (values[:, 1] > 0).all()
        assert result.model_dump_json()
