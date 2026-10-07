"""gnbreg, the family's likelihood kernels and its manifest.

Oracles: scipy.stats.nbinom with brute-force scipy.optimize maximization of an
independently written likelihood, numerical scores and Hessians of that
likelihood, statsmodels NegativeBinomial (constant alpha), duplicated rows for
frequency weights, ``optimize.check_derivatives`` for every likelihood of the
family and invariances that link the models to each other.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import optimize, stats

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.count import kernels
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec

X = ["x1", "x2"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(606)
    n = 2500
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "z1": rng.integers(0, 2, size=n).astype(float), "z2": rng.normal(size=n),
        "firm": np.repeat(np.arange(125), 20), "state": rng.integers(0, 12, size=n),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "expo": rng.uniform(0.5, 2.0, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
    })
    frame["lnexpo"] = np.log(frame.expo)
    mu = frame.expo * np.exp(0.5 + 0.4 * frame.x1 - 0.3 * frame.x2)
    alpha = np.exp(-1.0 + 1.0 * frame.z1 + 0.3 * frame.z2)
    frame["y"] = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(float)
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def design(frame, columns):
    if not columns:
        return np.ones((len(frame), 1))
    return sm.add_constant(frame[list(columns)]).to_numpy()


def gnb_loglik(theta, y, x, z, offset=0.0):
    """Per-observation generalized negative binomial log likelihood (scipy.stats)."""
    k = x.shape[1]
    mu = np.exp(x @ theta[:k] + offset)
    size = np.exp(-(z @ theta[k:]))
    return stats.nbinom.logpmf(y, size, size / (size + mu))


def numeric_scores(per_observation, theta, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = h * max(1.0, abs(theta[j]))
        columns.append((per_observation(theta + step) - per_observation(theta - step))
                       / (2 * step[j]))
    return np.column_stack(columns)


def numeric_hessian(total, theta, h=1e-4):
    theta = np.asarray(theta, dtype=float)
    k = len(theta)
    steps = np.diag([h * max(1.0, abs(value)) for value in theta])
    hessian = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            value = (total(theta + steps[i] + steps[j]) - total(theta + steps[i] - steps[j])
                     - total(theta - steps[i] + steps[j]) + total(theta - steps[i] - steps[j]))
            hessian[i, j] = hessian[j, i] = value / (4 * steps[i, i] * steps[j, j])
    return hessian


def close(actual, desired, rtol=5e-5):
    desired = np.asarray(desired)
    assert_allclose(actual, desired, rtol=rtol, atol=rtol * np.abs(desired).max())


def cluster_meat(scores, labels):
    sums = pd.DataFrame(scores).groupby(np.asarray(labels), sort=False).sum().to_numpy()
    return sums.T @ sums


def brute_force(objective, start):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = optimize.minimize(lambda t: -objective(t), start, method="BFGS",
                                 options={"gtol": 1e-9, "maxiter": 3000})
        best = optimize.minimize(lambda t: -objective(t), best.x, method="Nelder-Mead",
                                 options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 30000,
                                          "maxfev": 30000})
    return best.x, -best.fun


# ---- gnbreg --------------------------------------------------------------------------------


def test_gnbreg_matches_brute_force_and_information(data):
    result = oe.gnbreg(data=data, y="y", x=X, lnalpha=["z1", "z2"], exposure="expo")
    y, x, z = data.y.to_numpy(), design(data, X), design(data, ["z1", "z2"])
    offset = data.lnexpo.to_numpy()

    def total(theta):
        return gnb_loglik(theta, y, x, z, offset).sum()

    theta, value = brute_force(total, np.array([0.3, 0, 0, -0.5, 0, 0]))
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "x2", "lnalpha:Intercept", "lnalpha:z1", "lnalpha:z2"]
    assert [c.equation for c in result.coefficients] == ["y"] * 3 + ["lnalpha"] * 3
    assert_allclose(estimates(result), theta, rtol=5e-4, atol=5e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    assert result.metrics["log_likelihood"] == pytest.approx(total(estimates(result)), rel=1e-11)
    close(covariance(result), np.linalg.inv(-numeric_hessian(total, estimates(result))))
    assert result.metrics["aic"] == pytest.approx(-2 * value + 12, rel=1e-9)
    assert result.metrics["bic"] == pytest.approx(-2 * value + 6 * np.log(len(data)), rel=1e-9)
    # Stata's comparison model: constant-only mean equation, full lnalpha equation.
    _, null = brute_force(lambda t: gnb_loglik(t, y, x[:, :1], z, offset).sum(),
                          np.array([0.3, -0.5, 0, 0]))
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-8)
    assert result.tests["model"]["df"] == 2
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (value - null), rel=1e-5)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - value / null, rel=1e-6)
    # LR test of constant alpha against nbreg.
    plain = oe.nbreg(data=data, y="y", x=X, exposure="expo")
    assert result.extra["nbreg_log_likelihood"] == pytest.approx(
        plain.metrics["log_likelihood"], rel=1e-10)
    statistic = 2 * (value - plain.metrics["log_likelihood"])
    assert result.tests["lnalpha"]["df"] == 2
    assert result.tests["lnalpha"]["statistic"] == pytest.approx(statistic, rel=1e-6)
    assert result.tests["lnalpha"]["p_value"] == pytest.approx(stats.chi2.sf(statistic, 2),
                                                               rel=1e-6, abs=1e-300)
    fitted_alpha = np.exp(z @ estimates(result)[3:])
    assert result.extra["alpha"]["min"] == pytest.approx(fitted_alpha.min(), rel=1e-9)
    assert result.extra["alpha"]["max"] == pytest.approx(fitted_alpha.max(), rel=1e-9)
    assert result.extra["alpha"]["mean"] == pytest.approx(fitted_alpha.mean(), rel=1e-9)
    mean = np.exp(x @ estimates(result)[:3] + offset)
    for row in result.predictions[:20]:
        assert row["fitted"] == pytest.approx(mean[row["row"]], rel=1e-9)
    by_offset = oe.gnbreg(data=data, y="y", x=X, lnalpha=["z1", "z2"], offset="lnexpo")
    assert_allclose(estimates(by_offset), estimates(result), rtol=1e-9)


def test_gnbreg_without_lnalpha_regressors_is_nbreg(data):
    result = oe.gnbreg(data=data, y="y", x=X)
    plain = oe.nbreg(data=data, y="y", x=X)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2",
                                                     "lnalpha:Intercept"]
    assert_allclose(estimates(result), estimates(plain), rtol=1e-8)
    assert_allclose(errors(result), errors(plain), rtol=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(plain.metrics["log_likelihood"],
                                                             rel=1e-12)
    assert "lnalpha" not in result.tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = sm.NegativeBinomial(data.y, sm.add_constant(data[X])).fit(
            disp=0, method="newton", maxiter=200, tol=1e-12)
    assert_allclose(estimates(result)[:3], reference.params.to_numpy()[:3], rtol=1e-7)
    assert np.exp(estimates(result)[3]) == pytest.approx(reference.params.to_numpy()[3],
                                                         rel=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-10)


def test_gnbreg_covariances_and_weights(data):
    args = {"y": "y", "x": X, "lnalpha": ["z1"]}
    y, x, z = data.y.to_numpy(), design(data, X), design(data, ["z1"])
    base = oe.gnbreg(data=data, **args)
    theta = estimates(base)

    def per_observation(t):
        return gnb_loglik(t, y, x, z)

    scores = numeric_scores(per_observation, theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: per_observation(t).sum(), theta))
    n = len(data)
    close(covariance(oe.gnbreg(data=data, covariance="opg", **args)),
          np.linalg.inv(scores.T @ scores))
    robust = oe.gnbreg(data=data, covariance="robust", **args)
    close(covariance(robust), bread @ (scores.T @ scores) @ bread * n / (n - 1))
    assert robust.tests["model"]["label"].startswith("Wald chi2")
    assert robust.tests["lnalpha"]["label"].startswith("Wald chi2")
    wald = theta[4] ** 2 / covariance(robust)[4, 4]
    assert robust.tests["lnalpha"]["statistic"] == pytest.approx(wald, rel=1e-9)
    cluster = oe.gnbreg(data=data, cluster="firm", **args)
    groups = data.firm.nunique()
    close(covariance(cluster),
          bread @ cluster_meat(scores, data.firm) @ bread * groups / (groups - 1))
    two_way = oe.gnbreg(data=data, cluster=["firm", "state"], **args)
    pair = data.firm.astype(str) + "/" + data.state.astype(str)
    meat = (cluster_meat(scores, data.firm) + cluster_meat(scores, data.state)
            - cluster_meat(scores, pair))
    smallest = min(groups, data.state.nunique())
    # The inclusion-exclusion meat need not be positive semidefinite; its negative
    # eigenvalues are then set to zero (Cameron, Gelbach and Miller 2011) with a warning.
    # The projection is applied to the meat of the estimation coordinates (regressors
    # centered at their means, theta = T theta_c), like the glm family.
    adjusted = bool(np.linalg.eigvalsh(meat).min() < 0)
    assert two_way.inference["psd_adjusted"] is adjusted
    assert adjusted == any("positive semidefinite" in warning for warning in two_way.warnings)
    shift = np.eye(5)
    shift[0, 1:3] = -x[:, 1:].mean(axis=0)
    shift[3, 4] = -z[:, 1].mean()
    values, vectors = np.linalg.eigh(shift.T @ meat @ shift)
    back = np.linalg.inv(shift)
    meat = back.T @ ((vectors * np.clip(values, 0, None)) @ vectors.T) @ back
    close(covariance(two_way), bread @ meat @ bread * smallest / (smallest - 1))
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "opg", "robust"):
        weighted = oe.gnbreg(data=data, weights="fw", weight_type="fweight", covariance=kind,
                             **args)
        expanded = oe.gnbreg(data=repeated, covariance=kind, **args)
        assert weighted.nobs == len(repeated)
        assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-7, atol=1e-9)
        assert_allclose(covariance(weighted), covariance(expanded), rtol=1e-6, atol=1e-12)
    w = data.aw.to_numpy() * n / data.aw.sum()
    analytic = oe.gnbreg(data=data, weights="aw", weight_type="aweight", **args)
    best, value = brute_force(lambda t: (w * per_observation(t)).sum(),
                              np.array([0.3, 0, 0, -0.5, 0]))
    assert_allclose(estimates(analytic), best, rtol=5e-4, atol=5e-5)
    assert analytic.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    importance = oe.gnbreg(data=data, weights="aw", weight_type="iweight", **args)
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-7)
    sampling = oe.gnbreg(data=data, weights="aw", weight_type="pweight", **args)
    assert sampling.spec.covariance == "robust"
    raw = data.aw.to_numpy()
    weighted_scores = numeric_scores(per_observation, estimates(sampling)) * raw[:, None]
    weighted_bread = np.linalg.inv(-numeric_hessian(
        lambda t: (raw * per_observation(t)).sum(), estimates(sampling)))
    close(covariance(sampling),
          weighted_bread @ (weighted_scores.T @ weighted_scores) @ weighted_bread * n / (n - 1))
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=data, weights="aw", weight_type="pweight", covariance="nonrobust", **args)
    assert excinfo.value.code == "unsupported_covariance"


def test_gnbreg_failure_contract_collinearity_and_reporting(data):
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=data.assign(y=data.y - 3), y="y", x=X)
    assert excinfo.value.code == "invalid_count_outcome"
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=data.assign(y=0.0), y="y", x=X)
    assert excinfo.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=data, y="y", x=X, lnalpha="z1")
    assert excinfo.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=data, y="y", x=X, lnalpha=["absent"])
    assert excinfo.value.code == "missing_columns"
    # Underdispersed counts: the overdispersion is estimated at zero everywhere.
    rng = np.random.default_rng(2)
    under = rng.binomial(10, 0.3 + 0.05 * np.tanh(data.x1.to_numpy())).astype(float)
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=data.assign(y=under), y="y", x=X, lnalpha=["z1"])
    assert excinfo.value.code == "boundary_solution" and "oe.poisson" in str(excinfo.value)
    wide = oe.gnbreg(data=data.assign(twice=2 * data.x1, same=data.z1), y="y",
                     x=["x1", "twice", "sector"], lnalpha=["z1", "same", "sector"],
                     categorical=["sector"])
    assert wide.provenance["omitted_terms"] == ["twice", "lnalpha:same"]
    assert [c.term for c in wide.coefficients] == [
        "Intercept", "x1", "sector[b]", "sector[c]", "lnalpha:Intercept", "lnalpha:z1",
        "lnalpha:sector[b]", "lnalpha:sector[c]"]
    holes = data.copy()
    holes.loc[[0, 10], "z1"] = np.nan
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=holes, y="y", x=X, lnalpha=["z1"])
    assert excinfo.value.code == "missing_values"
    dropped = oe.gnbreg(data=holes, y="y", x=X, lnalpha=["z1"], missing="drop")
    assert dropped.nobs == len(data) - 2 and dropped.dropped_rows == 2
    no_constant = oe.gnbreg(data=data, y="y", x=X, lnalpha=["z1"], intercept=False)
    assert no_constant.metrics["pseudo_r_squared"] is None
    assert no_constant.tests["model"]["label"].startswith("Wald chi2")
    result = oe.gnbreg(data=data, y="y", x=X, lnalpha=["z1"])
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Generalized negative binomial regression" in text and "[lnalpha]" in text
    assert "lnalpha:z1" in text and "pseudo_r_squared" in text
    assert "lnalpha" in result.to_latex()
    refit = oe.fit(result.spec, data=data)
    assert_allclose(estimates(refit), estimates(result), rtol=1e-12)
    assert "gnbreg" in oe.gnbreg.__doc__ and "Example" in oe.gnbreg.__doc__


# ---- kernels: analytic derivatives of every likelihood ---------------------------------------


@pytest.fixture(scope="module")
def tensors():
    generator = torch.Generator().manual_seed(77)
    n = 400
    x = torch.cat([torch.ones((n, 1), dtype=torch.float64),
                   torch.randn((n, 2), dtype=torch.float64, generator=generator)], dim=1)
    z = torch.cat([torch.ones((n, 1), dtype=torch.float64),
                   torch.randn((n, 1), dtype=torch.float64, generator=generator)], dim=1)
    w = torch.rand(n, dtype=torch.float64, generator=generator) + 0.5
    offset = 0.2 * torch.randn(n, dtype=torch.float64, generator=generator)
    counts = torch.poisson(torch.exp(0.4 + 0.5 * x[:, 1]), generator=generator).to(torch.float64)
    keep = torch.rand(n, dtype=torch.float64, generator=generator) > 0.3
    limits = torch.randint(0, 4, (n,), generator=generator).to(torch.float64)
    normal = torch.randn(n, dtype=torch.float64, generator=generator)
    return {"x": x, "z": z, "w": w, "offset": offset, "counts": counts,
            "inflated": torch.where(keep, counts, torch.zeros_like(counts)),
            "limits": limits, "normal": normal}


def _check(objective, theta, gradient=1e-7, hessian=1e-6):
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < gradient, report
    assert report["hessian_max_rel_error"] < hessian, report
    assert report["hessian_asymmetry"] < 1e-9, report
    value, score, _ = objective(theta)
    assert float(objective.value(theta)) == pytest.approx(float(value), rel=1e-13)
    rows = objective.score_rows(theta)
    assert rows.shape == (len(objective.w), len(theta)) and bool(torch.isfinite(rows).all())
    assert_allclose((rows * objective.w[:, None]).sum(dim=0).numpy(), score.numpy(), rtol=1e-9,
                    atol=1e-9)
    assert float((objective.w * objective.observations(theta)).sum()) == pytest.approx(
        float(value), rel=1e-13)


BETA = [0.2, 0.3, -0.2]
GAMMA = [-0.5, 0.4]


def _theta(*parts):
    return torch.tensor([value for part in parts for value in part], dtype=torch.float64)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_zero_inflated_likelihood_derivatives(tensors, link):
    t = tensors
    for density, tail in ((kernels.PoissonDensity(), []), (kernels.NegBinDensity(), [-0.7])):
        designs = [t["x"], t["z"]] + ([None] if tail else [])
        offsets = [t["offset"], None] + ([None] if tail else [])
        pieces = kernels.ZeroInflatedPieces(density, t["inflated"], link)
        objective = kernels.IndexObjective(pieces, designs, t["w"], offsets)
        _check(objective, _theta(BETA, GAMMA, tail))
        # Far in both tails of the inflation index the pieces stay finite.
        extreme = _theta(BETA, [30.0, 0.0], tail)
        assert all(bool(torch.isfinite(part).all()) for part in objective(extreme))
        extreme = _theta(BETA, [-30.0, 0.0], tail)
        assert all(bool(torch.isfinite(part).all()) for part in objective(extreme))


@pytest.mark.parametrize("form", [None, "mean", "constant"])
def test_truncated_likelihood_derivatives(tensors, form):
    t = tensors
    density = kernels.PoissonDensity() if form is None else kernels.NegBinDensity(form)
    tail = [] if form is None else [-0.4]
    designs = [t["x"]] + ([None] if tail else [])
    offsets = [t["offset"]] + ([None] if tail else [])
    for limit, outcome in ((0, t["counts"] + 1), (3, t["counts"] + 4),
                           (t["limits"], t["counts"] + t["limits"] + 1)):
        pieces = kernels.TruncatedPieces(density, outcome, limit)
        objective = kernels.IndexObjective(pieces, designs, t["w"], offsets)
        _check(objective, _theta(BETA, tail), hessian=1e-6 if form is None else 5e-6)


def test_generalized_negative_binomial_binary_and_normal_derivatives(tensors):
    t = tensors
    pieces = kernels.CountPieces(kernels.NegBinDensity(), t["counts"])
    _check(kernels.IndexObjective(pieces, [t["x"], t["z"]], t["w"], [t["offset"], None]),
           _theta(BETA, GAMMA))
    _check(kernels.IndexObjective(pieces, [t["x"], None], t["w"], [t["offset"], None]),
           _theta(BETA, [-0.6]))
    _check(kernels.IndexObjective(kernels.CountPieces(kernels.PoissonDensity(), t["counts"]),
                                  [t["x"]], t["w"], [t["offset"]]), _theta(BETA))
    for link in ("logit", "probit", "cloglog"):
        binary = kernels.BinaryPieces(t["counts"] > 0, link)
        _check(kernels.IndexObjective(binary, [t["z"]], t["w"]), _theta(GAMMA))
    outcome = t["normal"].abs() + 0.5
    truncated = kernels.TruncatedNormalPieces(outcome, 0.5)
    _check(kernels.IndexObjective(truncated, [t["x"], None], t["w"]), _theta(BETA, [0.3]))
    lognormal = kernels.TruncatedNormalPieces(torch.log(outcome), None, -torch.log(outcome))
    _check(kernels.IndexObjective(lognormal, [t["x"], None], t["w"]), _theta(BETA, [0.3]))


def test_likelihood_invariances(tensors):
    t = tensors
    theta = _theta(BETA)
    plain = kernels.IndexObjective(kernels.CountPieces(kernels.PoissonDensity(), t["inflated"]),
                                   [t["x"]], t["w"], [t["offset"]])
    # Zero inflation with a vanishing inflation probability is the Poisson likelihood.
    inflated = kernels.IndexObjective(
        kernels.ZeroInflatedPieces(kernels.PoissonDensity(), t["inflated"], "logit"),
        [t["x"], t["z"]], t["w"], [t["offset"], None])
    value, gradient, hessian = inflated(_theta(BETA, [-45.0, 0.0]))
    reference = plain(theta)
    assert float(value) == pytest.approx(float(reference[0]), rel=1e-12)
    assert_allclose(gradient[:3].numpy(), reference[1].numpy(), rtol=1e-10)
    assert_allclose(hessian[:3, :3].numpy(), reference[2].numpy(), rtol=1e-10)
    # The truncated likelihood is the plain one minus ln Pr(Y > ll), observation by observation.
    outcome = t["counts"] + 3
    truncated = kernels.IndexObjective(
        kernels.TruncatedPieces(kernels.PoissonDensity(), outcome, 2), [t["x"]], t["w"],
        [t["offset"]])
    mu = np.exp((t["x"] @ theta + t["offset"]).numpy())
    expected = stats.poisson.logpmf(outcome.numpy(), mu) - stats.poisson.logsf(2, mu)
    assert_allclose(truncated.observations(theta).numpy(), expected, rtol=1e-11)
    # A negative binomial approaches Poisson within the validated gamma region.
    negbin = kernels.IndexObjective(
        kernels.CountPieces(kernels.NegBinDensity(), t["inflated"]), [t["x"], None], t["w"],
        [t["offset"], None])
    assert float(negbin.value(_theta(BETA, [math.log(2e-7)]))) == pytest.approx(
        float(reference[0]), rel=1e-6)
    from openecon.engines.contracts import KernelError
    with pytest.raises(KernelError, match="validated gamma-function region") as caught:
        negbin.value(_theta(BETA, [math.log(1e-7)]))
    assert caught.value.code == "precision_unsupported"
    # An overflowing predictor is rejected with a non-finite value.
    rejected = inflated(_theta([900.0, 0.0, 0.0], GAMMA))
    assert not math.isfinite(float(rejected[0]))


def test_estimator_invariances(data):
    # A zip fit on data whose zeros are all excess zeros separates into its two parts:
    # the count equation is the truncated Poisson fit of the positive counts.
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"x1": rng.normal(size=3000), "z1": rng.normal(size=3000)})
    positive = 1 + rng.poisson(np.exp(0.8 + 0.3 * frame.x1))
    frame["y"] = np.where(rng.uniform(size=3000) < 0.4, 0, positive).astype(float)
    hurdle = oe.hurdle(data=frame, y="y", x=["x1"], select_x=["z1"])
    truncated = oe.tpoisson(data=frame[frame.y > 0], y="y", x=["x1"])
    assert_allclose(estimates(hurdle)[:2], estimates(truncated), rtol=1e-9)
    assert_allclose(errors(hurdle)[:2], errors(truncated), rtol=1e-8)
    # Rescaling a regressor rescales its coefficient and nothing else.
    base = oe.gnbreg(data=data, y="y", x=X, lnalpha=["z2"])
    scaled = oe.gnbreg(data=data.assign(x1=data.x1 * 1000, z2=data.z2 / 1000 + 2000), y="y",
                       x=X, lnalpha=["z2"])
    assert estimates(scaled)[1] == pytest.approx(estimates(base)[1] / 1000, rel=1e-7)
    assert estimates(scaled)[4] == pytest.approx(estimates(base)[4] * 1000, rel=1e-6)
    assert scaled.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"],
                                                             rel=1e-10)
    assert errors(scaled)[4] == pytest.approx(errors(base)[4] * 1000, rel=1e-5)


# ---- manifest and public API ---------------------------------------------------------------


def test_manifest_and_specification_contract(data):
    names = [info.name for info in registry.all_estimators() if info.family == "count"]
    assert names == ["cpoisson", "cnbreg", "zip", "zinb", "tpoisson", "tnbreg", "churdle", "hurdle", "gnbreg"]
    exports = registry.public_exports()
    for name in names:
        info = registry.get(name)
        assert info.inference == "z" and info.default_covariance == "nonrobust"
        assert set(info.covariances) == {"nonrobust", "opg", "robust", "cluster"}
        assert set(info.weights) == {"fweight", "aweight", "pweight", "iweight"}
        assert name in exports and callable(getattr(oe, name))
        assert name in oe.capabilities()["estimators"]
    # Specification mistakes are rejected when the ModelSpec is built.
    with pytest.raises(ValidationError):
        ModelSpec(estimator="zip", outcome="y", predictors=X, columns={"select_x": ["z1"]})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="tpoisson", outcome="y", predictors=X, options={"ll": -1})
    with pytest.raises(ValidationError):
        ModelSpec(estimator="churdle", outcome="y", predictors=X)
    with pytest.raises(ValidationError):
        ModelSpec(estimator="gnbreg", outcome="y", predictors=X, covariance="HC1")
    with pytest.raises(ValidationError):
        ModelSpec(estimator="hurdle", outcome="y", predictors=X, panel="firm")
    # A spec built by hand fits through oe.fit like the convenience function.
    spec = ModelSpec(estimator="gnbreg", outcome="y", predictors=X, columns={"lnalpha": ["z1"]})
    assert spec.covariance == "nonrobust"
    direct = oe.fit(spec, data=data)
    assert_allclose(estimates(direct),
                    estimates(oe.gnbreg(data=data, y="y", x=X, lnalpha=["z1"])), rtol=1e-12)
    spec = ModelSpec(estimator="tpoisson", outcome="y", predictors=X,
                     columns={"truncation": "z1"}, options={"ll": 0})
    with pytest.raises(AnalysisError) as excinfo:
        oe.fit(spec, data=data[data.y > 1])
    assert excinfo.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as excinfo:
        oe.gnbreg(data=data.iloc[:0], y="y", x=X)
    assert excinfo.value.code == "empty_data"
