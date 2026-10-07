"""Adversarial inputs for the survival family: every case either gives a valid, finite
result or raises AnalysisError with a helpful code (verify-and-repair pass).

Regression tests for defects fixed in the verification pass are marked "regression".
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival.special import log_gamma_tails


def make(seed=0, n=60):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"t": rng.exponential(1, n) + 0.01, "d": (rng.random(n) < .7) * 1.0,
                         "x": rng.normal(size=n), "z": rng.normal(size=n),
                         "g": rng.integers(0, 2, n), "w": rng.integers(1, 3, n) * 1.0,
                         "id": np.arange(n)})


BASE = make()


def cox(frame, **options):
    return oe.stcox(data=frame, time="t", failure="d", x=options.pop("x", ["x"]), **options)


def streg(frame, **options):
    return oe.streg(data=frame, time="t", failure="d", x=options.pop("x", ["x"]), **options)


def finite_result(result):
    values = [c.estimate for c in result.coefficients] + [c.std_error for c in result.coefficients]
    values += [v for v in result.metrics.values() if v is not None]
    assert all(math.isfinite(v) for v in values)
    json.loads(result.model_dump_json())          # JSON round trip (no NaN / inf)


ERRORS = [
    ("cox one row", lambda: cox(BASE.iloc[:1]), "no_covariates"),
    ("cox no failures", lambda: cox(BASE.assign(d=0.0)), "no_failures"),
    ("cox constant x", lambda: cox(BASE.assign(x=1.0)), "no_covariates"),
    ("cox x constant within strata", lambda: cox(BASE.assign(x=BASE.g * 1.0), strata="g"),
     "no_covariates"),
    ("cox separation", lambda: cox(BASE.assign(x=BASE.d)), "separation_detected"),
    ("cox group without failures",
     lambda: cox(BASE.assign(x=(BASE.index < 10) * 1.0, d=np.where(BASE.index < 10, 0, BASE.d))),
     "separation_detected"),
    ("cox two rows", lambda: cox(BASE.iloc[:2]), "separation_detected"),
    ("cox string x", lambda: cox(BASE.assign(x="a")), "non_numeric_column"),
    ("cox missing raise", lambda: cox(BASE.assign(x=np.where(BASE.index == 3, np.nan, BASE.x))),
     "missing_values"),
    ("cox negative time", lambda: cox(BASE.assign(t=BASE.t - 0.5)), "invalid_survival_time"),
    ("cox negative entry", lambda: cox(BASE.assign(t0=-1.0), entry="t0"),
     "invalid_survival_time"),
    ("cox failure coded 2", lambda: cox(BASE.assign(d=2.0)), "invalid_failure_indicator"),
    ("cox infinite time", lambda: cox(BASE.assign(t=np.where(BASE.index == 0, np.inf, BASE.t))),
     "non_finite_values"),
    ("cox negative iweights", lambda: cox(BASE.assign(w=-BASE.w), weights="w",
                                          weight_type="iweight"), "negative_weights"),
    ("cox all zero fweights", lambda: cox(BASE.assign(w=0.0), weights="w",
                                          weight_type="fweight"), "empty_sample"),
    ("cox fractional fweights", lambda: cox(BASE.assign(w=BASE.w + .5), weights="w",
                                            weight_type="fweight"),
     "noninteger_frequency_weights"),
    ("cox aweights", lambda: cox(BASE, weights="w", weight_type="aweight"), "invalid_spec"),
    ("cox efron with weights", lambda: cox(BASE, weights="w", weight_type="fweight",
                                           ties="efron"), "unsupported_weights"),
    ("cox exactp with weights", lambda: cox(BASE, weights="w", weight_type="fweight",
                                            ties="exactp"), "unsupported_weights"),
    ("cox exactp robust", lambda: cox(BASE, ties="exactp", covariance="robust"), "invalid_spec"),
    ("cox exactp tvc", lambda: cox(BASE, ties="exactp", tvc=["z"]), "invalid_spec"),
    ("cox pweights nonrobust", lambda: cox(BASE, weights="w", weight_type="pweight",
                                           covariance="nonrobust"), "unsupported_covariance"),
    ("cox single cluster", lambda: cox(BASE.assign(g=0), covariance="cluster", cluster="g"),
     "insufficient_clusters"),
    ("cox every record ends before entry", lambda: cox(BASE.assign(t0=BASE.t + 1), entry="t0"),
     "empty_sample"),
    ("cox overlapping records", lambda: cox(pd.concat([BASE.assign(t0=0.0)] * 2), entry="t0",
                                            id="id"), "overlapping_records"),
    ("cox bad ties", lambda: cox(BASE, ties="foo"), "invalid_spec"),
    ("cox bare string x", lambda: oe.stcox(data=BASE, time="t", failure="d", x="x"),
     "invalid_spec"),
    ("cox heavy exact ties",          # 8000 at risk with 4000 tied failures at t = 1
     lambda: cox(pd.DataFrame({"t": 1.0, "d": np.arange(8000) % 2 * 1.0,
                               "x": np.random.default_rng(0).normal(size=8000)}),
                 ties="exactp"), "exact_too_large"),
    ("streg one row", lambda: streg(BASE.iloc[:1]), "insufficient_observations"),
    ("streg n <= k", lambda: streg(BASE.iloc[:3], x=["x", "z"]), "insufficient_observations"),
    ("streg no failures", lambda: streg(BASE.assign(d=0.0)), "no_failures"),
    ("streg constant time", lambda: streg(BASE.assign(t=1.0, d=1.0), distribution="lognormal"),
     "boundary_solution"),
    ("streg exponential ancillary", lambda: streg(BASE, distribution="exponential",
                                                  ancillary=["z"]), "invalid_spec"),
    ("streg gompertz aft", lambda: streg(BASE, distribution="gompertz", metric="aft"),
     "invalid_spec"),
    ("streg lognormal ph", lambda: streg(BASE, distribution="lognormal", metric="ph"),
     "invalid_spec"),
    ("streg pweights opg", lambda: streg(BASE, weights="w", weight_type="pweight",
                                         covariance="opg"), "unsupported_covariance"),
    ("streg unknown distribution", lambda: streg(BASE, distribution="gamma"), "invalid_spec"),
    ("sts empty", lambda: oe.sts(BASE.iloc[:0], "t", failure="d"), "empty_data"),
    ("sts all censored with tests", lambda: oe.sts(BASE.assign(d=0.0), "t", failure="d", by="g"),
     "no_failures"),
    ("sts one group", lambda: oe.sts(BASE.assign(g=0), "t", failure="d", by="g"),
     "invalid_groups"),
    ("sts string time", lambda: oe.sts(BASE.assign(t="x"), "t", failure="d"),
     "non_numeric_column"),
    ("sts conftype", lambda: oe.sts(BASE, "t", failure="d", conftype="x"), "invalid_option"),
    ("sts negative fh", lambda: oe.sts(BASE, "t", failure="d", by="g", test="fh", fh_p=-1),
     "invalid_option"),
    ("sts trend with two groups", lambda: oe.sts(BASE, "t", failure="d", by="g", trend=True),
     "invalid_trend"),
    ("sts alpha", lambda: oe.sts(BASE, "t", failure="d", alpha=1.0), "invalid_option"),
    ("sts missing raise", lambda: oe.sts(BASE.assign(t=np.where(BASE.index == 0, np.nan,
                                                                BASE.t)), "t", failure="d",
                                         missing="raise"), "missing_values"),
    ("sts negative weights", lambda: oe.sts(BASE.assign(w=-BASE.w), "t", failure="d",
                                            weights="w"), "noninteger_frequency_weights"),
    ("sts strata without by", lambda: oe.sts(BASE, "t", failure="d", strata="g"),
     "invalid_spec"),
    ("sts continuous by", lambda: oe.sts(make(n=1500), "t", failure="d", by="x"),
     "too_many_groups"),
    ("ltable continuous by", lambda: oe.ltable(make(n=1500), "t", failure="d", by="x"),
     "too_many_groups"),
    ("ltable empty", lambda: oe.ltable(BASE.iloc[:0], "t", failure="d"), "empty_data"),
    ("ltable zero width", lambda: oe.ltable(BASE, "t", failure="d", intervals=0),
     "invalid_option"),
    ("ltable tiny width", lambda: oe.ltable(BASE, "t", failure="d", intervals=1e-9),
     "too_many_intervals"),
    ("ltable decreasing cutpoints", lambda: oe.ltable(BASE, "t", failure="d",
                                                      intervals=[1, 0.5]), "invalid_option"),
    ("ltable all censored tests", lambda: oe.ltable(BASE.assign(d=0.0), "t", failure="d",
                                                    by="g"), "no_failures"),
    ("stcurve after streg", lambda: oe.stcurve(streg(BASE)), "unsupported_result"),
    ("stcurve unknown at", lambda: oe.stcurve(cox(BASE), at={"q": 1}), "invalid_option"),
    ("stcurve other data", lambda: oe.stcurve(cox(BASE), data=BASE.iloc[:30]), "data_mismatch"),
]


@pytest.mark.parametrize("label,call,code", ERRORS, ids=[e[0] for e in ERRORS])
def test_invalid_inputs_raise_analysis_errors(label, call, code):
    with pytest.raises(AnalysisError) as caught:
        call()
    assert caught.value.code == code
    assert len(str(caught.value)) > 20


VALID = [
    ("cox collinear", lambda: cox(BASE.assign(x2=2 * BASE.x), x=["x", "x2"])),
    ("cox missing drop", lambda: cox(BASE.assign(x=np.where(BASE.index == 3, np.nan, BASE.x)),
                                     missing="drop")),
    ("cox zero times", lambda: cox(BASE.assign(t=np.where(BASE.index < 5, 0.0, BASE.t)))),
    ("cox tiny covariate", lambda: cox(BASE.assign(x=BASE.x * 1e-8))),
    ("cox huge covariate", lambda: cox(BASE.assign(x=BASE.x * 1e8))),
    ("cox huge times", lambda: cox(BASE.assign(t=BASE.t * 1e8))),
    ("cox single stratum", lambda: cox(BASE.assign(g=0), strata="g")),
    ("cox every record ties", lambda: cox(BASE.assign(t=np.ceil(BASE.t * 2)), ties="exactp")),
    ("cox tvc log with entry", lambda: cox(BASE.assign(t0=BASE.t / 3), entry="t0", tvc=["z"],
                                           texp="log")),
    ("cox categorical", lambda: cox(BASE.assign(c=np.array(list("abc"))[BASE.index % 3]),
                                    x=["x", "c"], categorical=["c"])),
    ("streg all fail", lambda: streg(BASE.assign(d=1.0))),
    ("streg huge times", lambda: streg(BASE.assign(t=BASE.t * 1e8))),
    ("streg tiny times", lambda: streg(BASE.assign(t=BASE.t * 1e-8))),
    ("streg year covariate", lambda: streg(BASE.assign(x=2000 + BASE.x),
                                           distribution="gompertz")),
    ("streg ggamma tiny times", lambda: streg(BASE.assign(t=BASE.t * 1e-6),
                                              distribution="ggamma")),
    ("streg collinear ancillary", lambda: streg(BASE.assign(x2=3 * BASE.x),
                                                ancillary=["x", "x2"])),
    ("streg strata exponential", lambda: streg(BASE, strata="g", distribution="exponential")),
]


@pytest.mark.parametrize("label,call", VALID, ids=[v[0] for v in VALID])
def test_hard_but_valid_inputs_give_finite_results(label, call):
    finite_result(call())


def test_time_and_covariate_rescaling():
    base = streg(BASE, x=["x", "z"])
    stretched = streg(BASE.assign(t=BASE.t * 1e8), x=["x", "z"])
    a, b = ({c.term: c.estimate for c in r.coefficients} for r in (base, stretched))
    p = math.exp(a["/ln_p"])
    assert b["x"] == pytest.approx(a["x"], rel=1e-7)
    assert b["/ln_p"] == pytest.approx(a["/ln_p"], rel=1e-8)
    assert b["Intercept"] == pytest.approx(a["Intercept"] - p * math.log(1e8), rel=1e-8)
    tiny = cox(BASE.assign(x=BASE.x * 1e-8))
    plain = cox(BASE)
    assert tiny.coefficients[0].estimate * 1e-8 == pytest.approx(plain.coefficients[0].estimate,
                                                                 rel=1e-7)


def test_regression_incomplete_gamma_handles_infinite_and_missing_arguments():
    """regression: x = inf made the continued fraction loop to its iteration limit."""
    s = torch.tensor([-math.inf, math.inf, float("nan"), 0.0], dtype=torch.float64)
    for shape in (0.03, 2.0, 2e4):
        log_p, log_q = log_gamma_tails(shape, s)
        assert log_p[0] == -math.inf and log_q[0] == 0
        assert log_p[1] == 0 and log_q[1] == -math.inf
        assert torch.isnan(log_p[2]) and torch.isnan(log_q[2])
        assert bool(torch.isfinite(log_p[3])) and bool(torch.isfinite(log_q[3]))


def test_regression_generalized_gamma_without_interior_maximum():
    """regression: a diverging kappa ended in numerical_failure / nonconvergence."""
    with pytest.raises(AnalysisError) as caught:
        streg(BASE.assign(t0=BASE.t / 2), entry="t0", distribution="ggamma")
    assert caught.value.code == "boundary_solution"
    assert "kappa" in str(caught.value)


def test_regression_streg_monotone_likelihood_is_reported_as_separation():
    """regression: a category without failures was reported as plain nonconvergence."""
    frame = BASE.assign(x=(BASE.index < 10) * 1.0, d=np.where(BASE.index < 10, 0.0, 1.0))
    with pytest.raises(AnalysisError) as caught:
        streg(frame)
    assert caught.value.code == "separation_detected"


def test_regression_constant_only_streg():
    """regression: Stata's `streg, distribution(...)` without covariates was refused."""
    for dist in ("exponential", "weibull", "ggamma"):
        result = oe.streg(data=BASE, time="t", failure="d", distribution=dist)
        finite_result(result)
        assert result.coefficients[0].term == "Intercept"
        assert result.tests["model"]["df"] == 0 and result.metrics["df_model"] == 0
    expo = oe.streg(data=BASE, time="t", failure="d", distribution="exponential")
    rate = BASE.d.sum() / BASE.t.sum()           # closed-form exponential MLE
    assert expo.coefficients[0].estimate == pytest.approx(math.log(rate), rel=1e-10)


def test_regression_hazard_ratio_overflow_is_missing_not_a_crash():
    """regression: math.exp overflowed (raw OverflowError) for a huge coefficient."""
    result = cox(BASE.assign(x=BASE.x * 1e-8), covariance="robust")
    assert result.extra["hazard_ratios"]["x"]["hazard_ratio"] is None
    payload = json.loads(result.model_dump_json())
    assert payload["extra"]["hazard_ratios"]["x"]["ci_high"] is None
    weibull = streg(BASE.assign(x=BASE.x * 1e-8))
    assert weibull.extra["hazard_ratios"]["x"]["ratio"] is None


def test_degenerate_tables_report_missing_values_not_errors():
    everyone = oe.sts(BASE.assign(t=1.0, d=1.0), "t", failure="d", by="g")
    table = everyone["survival"]
    assert (table["survivor"] == 0).all()
    assert table["std_error"].isna().all()          # undefined at S = 0 (Stata: missing)
    single = oe.sts(BASE.iloc[:1], "t", failure="d")
    assert single["survival"]["survivor"].iloc[0] == 0
    censored = oe.sts(BASE.assign(d=0.0), "t", failure="d")
    assert (censored["survival"]["survivor"] == 1).all()
    assert censored["summary"]["median"].isna().all()
    zero = oe.ltable(BASE.assign(t=np.where(BASE.index < 5, 0.0, BASE.t)), "t", failure="d")
    assert zero["table"]["entering"].iloc[0] == len(BASE)
