"""Moment existence, parameter permutation and incoming RD/ATET option guards."""

import math
import numpy as np
import pytest
from scipy import special
import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle
from test_econ_saved_survival import parametric as _parametric_fixture, collect

parametric = _parametric_fixture


def test_parametric_means_permuted_full_parameters_and_nonexistent_moments(parametric):
    original, data = parametric
    r = ResultBundle.model_validate_json(original.model_dump_json())
    r.coefficients.reverse()
    r.covariance_matrix = np.array(r.covariance_matrix)[::-1, ::-1].tolist()
    d = data.iloc[:4]
    dist, metric = r.extra["distribution"], r.extra["metric"]
    b = {c.term: c.estimate for c in r.coefficients}
    mu = b["Intercept"] + b["x"] * d.x.to_numpy()
    ancillary = {
        "weibull": "ln_p",
        "lognormal": "lnsigma",
        "loglogistic": "lngamma",
        "ggamma": "lnsigma",
    }.get(dist)
    a = (
        b.get("/" + str(ancillary), b.get(str(ancillary) + ":Intercept", 0.0))
        + b.get(str(ancillary) + ":z", 0.0) * d.z.to_numpy()
    )
    if dist == "gompertz":
        with pytest.raises(AnalysisError):
            oe.survival_predict(r, d, target="mean")
        return
    if dist == "exponential":
        wanted = np.exp(-mu if metric == "ph" else mu)
    elif dist == "weibull":
        shape = np.exp(a)
        wanted = np.exp(-mu / shape if metric == "ph" else mu) * special.gamma(1 + 1 / shape)
    elif dist == "lognormal":
        wanted = np.exp(mu + np.exp(2 * a) / 2)
    elif dist == "loglogistic":
        gamma = np.exp(a)
        if np.any(gamma >= 1):
            with pytest.raises(AnalysisError):
                oe.survival_predict(r, d, target="mean")
            return
        wanted = np.exp(mu) * math.pi * gamma / np.sin(math.pi * gamma)
    else:
        kappa = b["/kappa"]
        sigma = np.exp(a)
        if abs(kappa) < 0.02 and abs(kappa) > 1e-10 or np.any(kappa**-2 + sigma / kappa <= 0):
            with pytest.raises(AnalysisError):
                oe.survival_predict(r, d, target="mean")
            return
        g = kappa**-2
        wanted = np.exp(
            mu + special.gammaln(g + sigma / kappa) - special.gammaln(g) - sigma / kappa * np.log(g)
        )
    actual = oe.survival_predict(r, d, target="mean", interval="mean")
    np.testing.assert_allclose(actual.response, wanted, rtol=1e-8, atol=1e-9)
    assert np.isfinite(actual.std_error).all()


def test_incoming_covariate_rd_side_requires_nuisance_and_derivative_target_explicit():
    from test_econ_streaming_rd import fixture

    d = fixture(380)
    d["z"] = np.random.default_rng(22).normal(size=len(d))
    r = oe.rdrobust(data=d, y="y", running="x", covariates=["z"], h=1.0, b=1.3)
    effect = collect(oe.causal_evaluate(r, target="cutoff_effect", point=0.0)).iloc[0]
    assert effect.std_error == pytest.approx(r.coefficients[2].std_error)
    with pytest.raises(AnalysisError, match="nuisance covariance"):
        oe.causal_evaluate(r, target="cutoff_side", side="left", point=0.0)
    r = oe.rdrobust(data=d, y="y", running="x", deriv=1, h=1.0, b=1.3)
    with pytest.raises(AnalysisError, match="derivative target"):
        oe.causal_evaluate(r, target="cutoff_effect", point=0.0)


def test_multivalued_atet_keeps_its_conditioning_population_after_restore():
    from test_econ_teffects_matching import make_data

    d = make_data(n=180)
    d["d"] = np.random.default_rng(25).choice(3, len(d))
    r = oe.teffects(
        data=d, y="y", treatment="d", x=["x1", "x2"], method="aipw", estimand="atet", tlevel=2
    )
    r = ResultBundle.model_validate_json(r.model_dump_json())
    effect = collect(oe.causal_evaluate(r, target="population_effect", treatment="1")).iloc[0]
    assert effect.estimate == r.coefficients[0].estimate
    assert effect.conditioning_treatment == 2
