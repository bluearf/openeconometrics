"""Adversarial inputs for the teffects family: every case must give a correct result or an
AnalysisError with a helpful snake_case code (never a raw traceback, NaN or inf)."""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def te_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * a))).astype(int)
    y = 1 + d + a + 0.5 * b + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "a": a, "b": b, "cl": rng.integers(0, 30, size=n)})


def code(call):
    with pytest.raises(AnalysisError) as error:
        call()
    assert error.value.code.replace("_", "").isalnum() and error.value.code.islower()
    assert len(str(error.value)) > 20
    return error.value.code


def finite(result):
    values = [v for c in result.coefficients
              for v in (c.estimate, c.std_error, c.statistic, c.p_value, c.ci_low, c.ci_high)]
    assert all(math.isfinite(v) for v in values)
    return result


TE = dict(y="y", treatment="d", x=["a", "b"])


# ---- teffects -------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw", "nnmatch", "psmatch"])
def test_teffects_degenerate_samples(method):
    df = te_data()
    assert code(lambda: oe.teffects(data=df.iloc[:0], method=method, **TE)) == "empty_data"
    assert code(lambda: oe.teffects(data=df.iloc[:1], method=method, **TE)) in {
        "invalid_treatment", "insufficient_observations"}
    tiny = df.iloc[:4].copy()
    tiny["d"] = [0, 1, 0, 1]
    assert code(lambda: oe.teffects(data=tiny, method=method, **TE)) in {
        "insufficient_observations", "overlap_violation", "separation_detected",
        "invalid_covariance", "singular_jacobian", "singular_covariance"}
    one_level = df.assign(d=1)
    assert code(lambda: oe.teffects(data=one_level, method=method, **TE)) == "invalid_treatment"
    holes = df.assign(a=np.nan)
    assert code(lambda: oe.teffects(data=holes, method=method, missing="drop", **TE)) \
        == "empty_sample"


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_teffects_constant_outcome_and_extreme_scales(method):
    df = te_data(1)
    flat = df.assign(y=3.0)
    try:
        result = oe.teffects(data=flat, method=method, **TE)
    except AnalysisError as error:
        assert error.code in {"invalid_covariance", "non_finite_result"}
    else:
        finite(result)
    base = finite(oe.teffects(data=df, method=method, **TE))
    scaled = df.assign(y=df.y * 1e8, a=df.a * 1e-8, b=df.b * 1e8 + 1e9)
    big = finite(oe.teffects(data=scaled, method=method, **TE))
    assert_allclose(big.coefficients[0].estimate, base.coefficients[0].estimate * 1e8,
                    rtol=1e-7)
    assert_allclose(big.coefficients[0].std_error, base.coefficients[0].std_error * 1e8,
                    rtol=1e-6)


def test_teffects_overlap_separation_and_collinearity():
    df = te_data(2)
    separated = df.assign(d=(df.a > 0).astype(int))
    for method in ("ipw", "ipwra", "aipw", "psmatch"):
        assert code(lambda: oe.teffects(data=separated, method=method, **TE)) \
            == "overlap_violation"
    near = df.assign(d=(df.a + 0.02 * np.random.default_rng(0).normal(size=len(df)) > 0)
                     .astype(int))
    assert code(lambda: oe.teffects(data=near, method="ipw", **TE)) == "overlap_violation"
    twin = df.assign(c=2 * df.a + 1)
    result = finite(oe.teffects(data=twin, y="y", treatment="d", x=["a", "c", "b"],
                                method="aipw"))
    assert any("Omitted" in w for w in result.warnings)
    # a covariate constant inside one treatment group is dropped from that equation only
    within = df.assign(c=np.where(df.d == 1, 1.0, df.b))
    result = finite(oe.teffects(data=within, y="y", treatment="d", x=["a", "c"], method="ra"))
    assert any("OME1:c" in w for w in result.warnings)


def test_teffects_invalid_inputs():
    df = te_data(3)
    assert code(lambda: oe.teffects(data=df.assign(a="text"), method="ra", **TE)) \
        in {"invalid_numeric", "non_numeric", "invalid_column", "invalid_values",
            "non_numeric_column"}
    assert code(lambda: oe.teffects(data=df.assign(y=np.inf), method="ra", **TE)) \
        in {"non_finite", "non_finite_values", "invalid_numeric", "missing_values"}
    assert code(lambda: oe.teffects(data=df, method="ra", metric="euclidean", **TE)) \
        == "invalid_spec"
    assert code(lambda: oe.teffects(data=df, y="y", treatment="d", x=["d", "a"])) \
        == "invalid_spec"
    assert code(lambda: oe.teffects(data=df, method="nnmatch", covariance="cluster",
                                    cluster="cl", **TE)) == "unsupported_covariance"
    assert code(lambda: oe.teffects(data=df.assign(cl=1), method="ra", covariance="cluster",
                                    cluster="cl", **TE)) == "insufficient_clusters"
    assert code(lambda: oe.teffects(data=df, method="ra", omodel="poisson", **TE)) \
        == "invalid_outcome"
    assert code(lambda: oe.teffects(data=df, method="aipw", omodel="logit", **TE)) \
        == "invalid_outcome"
    assert code(lambda: oe.teffects(data=df.assign(w=-1.0), method="ra", weights="w",
                                    weight_type="pweight", **TE)) == "negative_weights"
    assert code(lambda: oe.teffects(data=df.assign(w=1.5), method="ra", weights="w",
                                    weight_type="fweight", **TE)) \
        == "noninteger_frequency_weights"
    assert code(lambda: oe.teffects(data=df.assign(w=0.0), method="ra", weights="w",
                                    weight_type="pweight", **TE)) == "empty_sample"
    assert code(lambda: oe.teffects(data=df.assign(w=1.0), method="nnmatch", weights="w",
                                    weight_type="pweight", **TE)) == "unsupported_weights"
    three = df.assign(d=df.d + (df.a > 1))
    finite(oe.teffects(data=three, method="ra", estimand="atet", **TE))
    assert code(lambda: oe.teffects(data=three, method="psmatch", **TE)) == "invalid_treatment"
    finite(oe.teffects(data=df, method="aipw", estimand="atet", **TE))
    assert code(lambda: oe.teffects(data=df, method="ra", control=7, **TE)) \
        == "invalid_treatment"
    assert code(lambda: oe.teffects(data=df, method="ra", x="a", y="y", treatment="d")) \
        == "invalid_spec"


def test_teffects_labels_bool_and_string_treatments():
    df = te_data(4)
    base = finite(oe.teffects(data=df, method="aipw", **TE))
    text = finite(oe.teffects(data=df.assign(d=np.where(df.d == 1, "yes", "no")),
                              method="aipw", **TE))
    assert [c.term for c in text.coefficients] == ["ATE:ryesvsno.d", "POmean:no.d"]
    assert_allclose(text.coefficients[0].estimate, base.coefficients[0].estimate, rtol=1e-12)
    flag = finite(oe.teffects(data=df.assign(d=df.d.astype(bool)), method="aipw", **TE))
    assert_allclose(flag.coefficients[0].estimate, base.coefficients[0].estimate, rtol=1e-12)
    flipped = finite(oe.teffects(data=df, method="aipw", control=1, **TE))
    assert_allclose(flipped.coefficients[0].estimate, -base.coefficients[0].estimate,
                    rtol=1e-10)


def test_matching_bounds():
    df = te_data(5, n=120)
    groups = df.d.value_counts()
    assert code(lambda: oe.teffects(data=df, method="nnmatch",
                                    neighbors=int(groups.min()) + 1, **TE)) \
        == "insufficient_observations"
    assert code(lambda: oe.teffects(data=df, method="nnmatch", caliper=1e-9, **TE)) \
        == "caliper_violation"
    assert code(lambda: oe.teffects(data=df, method="psmatch", pstolerance=0.5, **TE)) \
        == "overlap_violation"
    flat = df.assign(a=1.0, b=2.0)
    assert code(lambda: oe.teffects(data=flat, method="nnmatch", **TE)) == "invalid_spec"
    one = finite(oe.teffects(data=df, y="y", treatment="d", x=["a"], method="nnmatch"))
    assert one.extra["matches"]["min"] >= 1


# ---- didregress / eventstudy -------------------------------------------------------


def panel(seed=0, groups=12, periods=5):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), periods)
    t = np.tile(np.arange(periods), groups)
    first = np.where(g < groups // 2, 2, np.nan)
    d = (t >= np.nan_to_num(first, nan=99)).astype(int)
    y = rng.normal(size=groups)[g] + 0.1 * t + d + rng.normal(size=len(g))
    return pd.DataFrame({"y": y, "d": d, "g": g, "t": t, "first": first,
                         "x": rng.normal(size=len(g))})


DID = dict(y="y", treatment="d", group="g", time="t")
ES = dict(y="y", group="g", time="t", treatment_time="first")


def test_didregress_degenerate_and_invalid():
    df = panel()
    assert code(lambda: oe.didregress(data=df.iloc[:0], **DID)) == "empty_data"
    assert code(lambda: oe.didregress(data=df.assign(d=2 * df.d), **DID)) == "invalid_treatment"
    assert code(lambda: oe.didregress(data=df.assign(d=1), **DID)) == "invalid_treatment"
    assert code(lambda: oe.didregress(data=df[df.g == 0], **DID)) in {
        "insufficient_observations", "invalid_treatment"}
    # every treated group starts together and there are no controls: not identified
    all_treated = df.assign(d=(df.t >= 2).astype(int))
    assert code(lambda: oe.didregress(data=all_treated, **DID)) == "no_within_variation"
    assert code(lambda: oe.didregress(data=df.iloc[:6], **DID)) in {
        "insufficient_observations", "invalid_treatment", "no_within_variation"}
    absorbed = df.assign(z=df.g * 2.0)
    result = finite(oe.didregress(data=absorbed, x=["x", "z"], **DID))
    assert any("absorbed" in w for w in result.warnings)
    assert code(lambda: oe.didregress(data=df.assign(w=-1.0), weights="w",
                                      weight_type="aweight", **DID)) == "negative_weights"
    assert code(lambda: oe.didregress(data=df, covariance="nonrobust", weights="x",
                                      weight_type="pweight", **DID.copy()))  in {
        "negative_weights", "unsupported_covariance"}


def test_didregress_labels_gaps_and_switching_treatment():
    df = panel(1)
    gaps = df.assign(t=df.t * 3 + 2000, g=df.g.map(lambda v: f"unit{v}"))
    a = finite(oe.didregress(data=df, **DID))
    b = finite(oe.didregress(data=gaps, **DID))
    assert_allclose(a.coefficients[0].estimate, b.coefficients[0].estimate, rtol=1e-10)
    assert b.extra["first_treated_periods"] == [2006]
    switching = df.copy()
    switching.loc[(switching.g == 0) & (switching.t == 4), "d"] = 0
    result = finite(oe.didregress(data=switching, **DID))
    assert result.tests == {} and "tests_note" in result.extra


def test_eventstudy_invalid_inputs():
    df = panel(2)
    bad = df.copy()
    bad.loc[0, "first"] = 3
    assert code(lambda: oe.eventstudy(data=bad, **ES)) == "invalid_treatment"
    assert code(lambda: oe.eventstudy(data=df.assign(first=np.nan), **ES)) == "invalid_treatment"
    assert code(lambda: oe.eventstudy(data=df.assign(first=df["first"] + 0.5), **ES)) \
        == "invalid_time"
    assert code(lambda: oe.eventstudy(data=df, reference=-9, **ES)) == "invalid_spec"
    assert code(lambda: oe.eventstudy(data=df, leads=0, **ES)) == "invalid_spec"
    result = finite(oe.eventstudy(data=df, lags=0, leads=1, **ES))
    assert [c.term for c in result.coefficients if c.term.startswith("l")] == ["lag0"]
    # several observations per group and period (repeated cross-sections) are legitimate
    finite(oe.eventstudy(data=pd.concat([df, df.iloc[:7]]), **ES))


# ---- csdid --------------------------------------------------------------------------


def test_csdid_invalid_inputs():
    df = panel(3, groups=20, periods=5)
    cs = dict(y="y", group="g", time="t", treatment_time="first")
    assert code(lambda: oe.csdid(data=df.iloc[1:], **cs)) == "unbalanced_panel"
    assert code(lambda: oe.csdid(data=pd.concat([df, df.iloc[:1]]), **cs)) == "unbalanced_panel"
    assert code(lambda: oe.csdid(data=df.assign(first=2.0), **cs)) == "invalid_treatment"
    assert code(lambda: oe.csdid(data=df.assign(first=df["first"] + 0.5), **cs)) in {
        "invalid_treatment", "invalid_time"}
    assert code(lambda: oe.csdid(data=df.assign(first=np.nan), **cs)) == "invalid_treatment"
    separated = df.assign(z=df["first"].notna() * 1.0)
    assert code(lambda: oe.csdid(data=separated, x=["z"], method="dr", **cs)) \
        in {"overlap_violation", "singular_jacobian", "separation_detected"}
    finite(oe.csdid(data=df, x=["x"], method="reg", **cs))


# ---- rdrobust ----------------------------------------------------------------------


def rd(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=n)
    y = x + 0.5 * (x >= 0) + rng.normal(size=n) * 0.3
    t = (rng.uniform(size=n) < np.where(x >= 0, 0.7, 0.3)).astype(float)
    return pd.DataFrame({"y": y, "x": x, "t": t, "cl": rng.integers(0, 40, size=n)})


def test_rdrobust_degenerate_and_invalid():
    df = rd()
    assert code(lambda: oe.rdrobust(data=df, y="y", running="x", cutoff=5.0)) == "invalid_cutoff"
    assert code(lambda: oe.rdrobust(data=df, y="y", running="x", h=1e-6)) \
        == "insufficient_observations"
    assert code(lambda: oe.rdrobust(data=df, y="y", running="x", h=-1.0)) == "invalid_spec"
    assert code(lambda: oe.rdrobust(data=df, y="y", running="x", h=[0.3])) == "invalid_spec"
    assert code(lambda: oe.rdrobust(data=df.assign(y=1.0), y="y", running="x")) in {
        "bandwidth_selection_failed", "invalid_covariance"}
    assert code(lambda: oe.rdrobust(data=df.assign(y=1.0), y="y", running="x", h=0.5)) \
        == "invalid_covariance"
    assert code(lambda: oe.rdrobust(data=df.assign(t=0.5), y="y", running="x", fuzzy="t",
                                    h=0.5)) == "weak_first_stage"
    assert code(lambda: oe.rdrobust(data=df.assign(cl=1), y="y", running="x", h=0.5,
                                    covariance="cluster", cluster="cl")) \
        == "insufficient_clusters"
    assert code(lambda: oe.rdrobust(data=df, y="y", running="x", p=3, q=2)) == "invalid_spec"
    coarse = df.assign(x=np.round(df.x, 1))
    assert code(lambda: oe.rdrobust(data=coarse, y="y", running="x", p=6, h=0.25)) \
        == "insufficient_observations"
    assert code(lambda: oe.rdrobust(data=df, y="y", running="x", covariates=["t"])) \
        == "not_implemented"


@pytest.mark.parametrize("scale,shift", [(1e-6, 0.0), (1e6, 0.0), (1.0, 1e6)])
def test_rdrobust_is_equivariant_to_running_variable_units(scale, shift):
    df = rd(1, n=800)
    moved = df.assign(x=df.x * scale + shift)
    base = finite(oe.rdrobust(data=df, y="y", running="x", h=0.4, b=0.7))
    other = finite(oe.rdrobust(data=moved, y="y", running="x", cutoff=shift, h=0.4 * scale,
                               b=0.7 * scale))
    for a, b in zip(base.coefficients, other.coefficients):
        assert_allclose(b.estimate, a.estimate, rtol=1e-8)
        assert_allclose(b.std_error, a.std_error, rtol=1e-8)
    # rdbwselect's first step uses the bias bandwidth range + 1e-8 in absolute units (as the
    # R/Stata package does), so selected bandwidths are equivariant only when that offset is
    # negligible relative to the range of the running variable
    base = finite(oe.rdrobust(data=df, y="y", running="x"))
    other = finite(oe.rdrobust(data=moved, y="y", running="x", cutoff=shift))
    tolerance = 1e-6 if scale >= 1 else 5e-3
    assert_allclose(other.metrics["h_left"], base.metrics["h_left"] * scale, rtol=tolerance)
    assert_allclose(other.coefficients[2].std_error, base.coefficients[2].std_error,
                    rtol=tolerance)


def test_rdplot_invalid_inputs():
    df = rd(2)
    assert code(lambda: oe.rdplot(data=df, y="y", running="x", binselect="xx")) == "invalid_spec"
    assert code(lambda: oe.rdplot(data=df, y="y", running="x", nbins=0)) == "invalid_spec"
    assert code(lambda: oe.rdplot(data=df, y="y", running="x", cutoff=3.0)) in {
        "insufficient_observations", "invalid_cutoff"}
    plot = oe.rdplot(data=df, y="y", running="x")
    assert plot.attrs["bins_left"] >= 1


def test_csdid_collinear_constant_covariates_and_label_time():
    """Regression: collinear or constant covariates raised a raw KernelError (reg) or a
    misleading nonconvergence (ipw/dr); they are now omitted per comparison, Stata/R style."""
    df = panel(5, groups=40, periods=5)
    cs = dict(y="y", group="g", time="t", treatment_time="first")
    for method in ("reg", "ipw", "dr"):
        only = oe.csdid(data=df, x=["x"], method=method, **cs)
        twin = oe.csdid(data=df.assign(x2=2 * df.x + 1), x=["x", "x2"], method=method, **cs)
        assert_allclose(np.array(twin.covariance_matrix), np.array(only.covariance_matrix),
                        rtol=1e-8)
        assert any("x2" in w and "collinearity" in w for w in twin.warnings)
        plain = oe.csdid(data=df, method=method, **cs)
        flat = oe.csdid(data=df.assign(k=1.0), x=["k"], method=method, **cs)
        assert_allclose([c.estimate for c in flat.coefficients],
                        [c.estimate for c in plain.coefficients], rtol=1e-9, atol=1e-12)
    labels = df.assign(t=df.t.map(lambda v: f"p{v}"))
    assert code(lambda: oe.csdid(data=labels, **cs)) == "invalid_time"
    assert code(lambda: oe.eventstudy(data=labels, **ES)) == "invalid_time"


def test_rdrobust_refuses_multiway_clusters():
    df = rd(3).assign(c2=lambda d: d.cl % 7)
    assert code(lambda: oe.rdrobust(data=df, y="y", running="x", h=0.5, covariance="cluster",
                                    cluster=["cl", "c2"])) in {"invalid_spec",
                                                               "cluster_dimensions"}
