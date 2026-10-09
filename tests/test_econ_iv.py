"""ivregress (2SLS / LIML / GMM) against explicit NumPy algebra, SciPy and statsmodels oracles."""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import optimize, stats
from statsmodels.sandbox.regression.gmm import IV2SLS

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics import registry
from openecon.econometrics.iv import ESTIMATORS, EXPORTS
from openecon.models import ModelSpec, ResultBundle

X, ENDOG, INST = ["x1"], ["p", "p2"], ["z1", "z2", "z3"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261)
    n = 400
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "z1": rng.normal(size=n), "z2": rng.normal(size=n),
        "z3": rng.normal(size=n), "firm": np.repeat(np.arange(40), 10),
        "state": rng.integers(0, 8, size=n), "t": np.arange(n),
        "sector": rng.choice(["a", "b", "c"], size=n),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.0, size=n),
    })
    v, v2 = rng.normal(size=n), rng.normal(size=n)
    firm = np.repeat(rng.normal(size=40), 10)
    frame["p"] = frame.z1 + 0.5 * frame.z2 + 0.3 * frame.x1 + v
    frame["p2"] = -frame.z2 + 0.8 * frame.z3 + v2
    noise = rng.normal(size=n) * (1 + 0.6 * frame.z1.abs()) + 0.7 * firm
    frame["y"] = 1 + 2 * frame.p - frame.x1 + 0.5 * frame.p2 + 0.8 * v - 0.4 * v2 + noise
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def blocks(frame, x=X, endog=ENDOG, inst=INST, intercept=True):
    """(y, X1, X2, Z2) as NumPy arrays with the constant first in X1."""
    exog = [np.ones(len(frame))] if intercept else []
    x1 = np.column_stack([*exog, *(frame[c].to_numpy(float) for c in x)]) if exog or x \
        else np.empty((len(frame), 0))
    return (frame.y.to_numpy(float), x1, frame[list(endog)].to_numpy(float),
            frame[list(inst)].to_numpy(float))


def projector(a, w=None):
    """Weighted projection matrix A (A'WA)^-1 A'W (explicit n-by-n: small n only)."""
    w = np.ones(len(a)) if w is None else w
    return a @ np.linalg.solve(a.T @ (a * w[:, None]), (a * w[:, None]).T)


def k_class(y, x1, x2, z2, w=None, kappa=1.0):
    """b, residuals, bread and score regressors of the k-class estimator, by brute force."""
    n = len(y)
    w = np.ones(n) if w is None else w
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    xk = (np.eye(n) - kappa * (np.eye(n) - projector(z, w))) @ x
    bread = np.linalg.inv(xk.T @ (x * w[:, None]))
    b = bread @ (xk.T @ (w * y))
    return b, y - x @ b, bread, xk


def cluster_meat(scores, *labels):
    def one(codes):
        sums = pd.DataFrame(scores).groupby(np.asarray(codes), sort=False).sum().to_numpy()
        return sums.T @ sums
    if len(labels) == 1:
        return one(labels[0])
    both = pd.Series(labels[0]).astype(str) + "/" + pd.Series(labels[1]).astype(str)
    return one(labels[0]) + one(labels[1]) - one(both.to_numpy())


def bartlett_meat(scores, lags):
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    return meat


def iv(frame, **options):
    options = {"x": X, "endog": ENDOG, "instruments": INST, **options}
    return oe.ivregress(data=frame, y="y", **options)


# ---- manifest and public API ---------------------------------------------------------------


def test_manifest_registers_the_family_and_public_functions():
    assert [info.name for info in ESTIMATORS] == ["ivregress", "xtivreg", "ivreghdfe", "ivcue"]
    assert all(info.family == "iv" for info in ESTIMATORS)
    assert set(EXPORTS) == {"iv_weak_test", "iv_ar_confidence_set", "stock_yogo", "effective_f", "iv_saved_weak_test", "iv_saved_ar_confidence_set"}
    for name in ("ivregress", "xtivreg", "ivreghdfe", "ivcue"):
        assert callable(getattr(oe, name)) and name in dir(oe)
    info = registry.get("ivregress")
    assert info.covariances == ("nonrobust", "robust", "cluster", "hac")
    assert info.predictors == "optional" and info.cluster_dimensions == 2
    assert info.role("endogenous").required and info.role("instruments").many
    record = oe.capabilities()["estimators"]["ivregress"]
    assert record["options"]["method"]["choices"] == ["2sls", "liml", "gmm", "fuller", "kclass"]
    assert record["columns"]["instruments"]["required"] and record["weights"] == [
        "aweight", "fweight", "pweight"]
    assert "iv" in oe.capabilities()["families"]


# ---- 2SLS ------------------------------------------------------------------------------------


def test_2sls_matches_projection_algebra_and_statsmodels(data):
    result = iv(data)
    y, x1, x2, z2 = blocks(data)
    b, u, bread, _ = k_class(y, x1, x2, z2)
    n, k = len(y), 4
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "p", "p2"]
    assert_allclose(estimates(result), b, rtol=1e-11)
    reference = IV2SLS(y, np.column_stack([x1, x2]), np.column_stack([x1, z2])).fit()
    assert_allclose(estimates(result), reference.params, rtol=1e-10)
    # Stata's unadjusted VCE: s^2 = RSS/N, z statistics.
    assert_allclose(np.asarray(result.covariance_matrix), bread * (u @ u) / n, rtol=1e-10)
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.norm.sf(np.abs(b / np.sqrt(np.diag(bread) * (u @ u) / n))),
                    rtol=1e-8, atol=1e-300)
    assert result.inference["use_t"] is False and result.inference["df_inference"] is None
    assert result.spec.covariance == "nonrobust"
    tss = ((y - y.mean()) ** 2).sum()
    assert result.metrics["r_squared"] == pytest.approx(1 - u @ u / tss, rel=1e-12)
    assert result.metrics["adjusted_r_squared"] == pytest.approx(
        1 - (u @ u / tss) * (n - 1) / (n - k), rel=1e-12)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(u @ u / n), rel=1e-12)
    assert list(result.metrics) == ["r_squared", "adjusted_r_squared", "rmse", "df_model",
                                    "df_resid", "n_instruments", "n_endogenous"]
    assert result.metrics["df_model"] == 3 and result.metrics["df_resid"] == n - k
    assert result.metrics["n_instruments"] == 3 and result.metrics["n_endogenous"] == 2
    test = result.tests["model"]
    v = bread[1:, 1:] * (u @ u) / n
    wald = b[1:] @ np.linalg.solve(v, b[1:])
    assert test["distribution"] == "chi2" and test["df"] == 3
    assert test["statistic"] == pytest.approx(wald, rel=1e-9)
    assert test["p_value"] == pytest.approx(stats.chi2.sf(wald, 3), rel=1e-8, abs=1e-300)
    assert result.title == "Instrumental-variables (2SLS) regression"
    assert result.provenance["stata_parity_validated"] is False
    assert result.provenance["solver"] == "householder_qr_two_stage"
    assert result.nobs == n and len(result.predictions) == n


def test_small_gives_t_and_f_with_degrees_of_freedom_corrections(data):
    result = iv(data, small=True)
    y, x1, x2, z2 = blocks(data)
    b, u, bread, _ = k_class(y, x1, x2, z2)
    n, k = len(y), 4
    v = bread * (u @ u) / (n - k)
    assert_allclose(np.asarray(result.covariance_matrix), v, rtol=1e-10)
    # statsmodels' IV2SLS also divides by N - K.
    reference = IV2SLS(y, np.column_stack([x1, x2]), np.column_stack([x1, z2])).fit()
    assert_allclose(errors(result), reference.bse, rtol=1e-9)
    assert result.inference["use_t"] and result.inference["df_inference"] == n - k
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.t.sf(np.abs(b / np.sqrt(np.diag(v))), n - k), rtol=1e-8,
                    atol=1e-300)
    test = result.tests["model"]
    f = b[1:] @ np.linalg.solve(v[1:, 1:], b[1:]) / 3
    assert test["distribution"] == "F" and (test["df"], test["df2"]) == (3, n - k)
    assert test["statistic"] == pytest.approx(f, rel=1e-9)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(u @ u / (n - k)), rel=1e-12)
    assert result.inference["error_variance"] == "RSS/(N-K)"


@pytest.mark.parametrize("small", [False, True])
def test_robust_covariance_is_the_sandwich_on_projected_regressors(data, small):
    result = iv(data, covariance="robust", small=small)
    y, x1, x2, z2 = blocks(data)
    b, u, bread, xhat = k_class(y, x1, x2, z2)
    n, k = len(y), 4
    scores = xhat * u[:, None]
    factor = n / (n - k) if small else 1.0
    assert_allclose(np.asarray(result.covariance_matrix),
                    bread @ (scores.T @ scores) @ bread * factor, rtol=1e-10)
    assert result.inference["small_sample_correction"] == pytest.approx(factor)
    assert result.inference["use_t"] is small


@pytest.mark.parametrize("small", [False, True])
def test_cluster_covariance_applies_stata_factors(data, small):
    result = iv(data, cluster="firm", small=small)
    y, x1, x2, z2 = blocks(data)
    b, u, bread, xhat = k_class(y, x1, x2, z2)
    n, k, g = len(y), 4, 40
    factor = g / (g - 1) * ((n - 1) / (n - k) if small else 1.0)
    expected = bread @ cluster_meat(xhat * u[:, None], data.firm) @ bread * factor
    assert result.spec.covariance == "cluster"
    assert_allclose(np.asarray(result.covariance_matrix), expected, rtol=1e-10)
    assert result.inference["cluster_count"] == g
    assert result.inference["df_inference"] == (g - 1 if small else None)
    if small:
        assert_allclose([c.p_value for c in result.coefficients],
                        2 * stats.t.sf(np.abs(b / np.sqrt(np.diag(expected))), g - 1), rtol=1e-8,
                        atol=1e-300)
        assert result.tests["model"]["df2"] == g - 1


def test_two_way_cluster_uses_inclusion_exclusion_with_the_smaller_dimension(data):
    result = iv(data, cluster=["firm", "state"])
    y, x1, x2, z2 = blocks(data)
    _, u, bread, xhat = k_class(y, x1, x2, z2)
    meat = cluster_meat(xhat * u[:, None], data.firm, data.state)
    assert_allclose(np.asarray(result.covariance_matrix), bread @ meat @ bread * 8 / 7,
                    rtol=1e-9)
    assert result.inference["cluster_counts"] == [40, 8]
    assert any("Only 8 clusters" in warning for warning in result.warnings)


@pytest.mark.parametrize("small", [False, True])
def test_hac_covariance_is_newey_west_on_projected_scores(data, small):
    result = iv(data, covariance="hac", lags=3, time="t", small=small)
    y, x1, x2, z2 = blocks(data)
    _, u, bread, xhat = k_class(y, x1, x2, z2)
    n, k = len(y), 4
    expected = bread @ bartlett_meat(xhat * u[:, None], 3) @ bread
    assert_allclose(np.asarray(result.covariance_matrix),
                    expected * (n / (n - k) if small else 1.0), rtol=1e-10)
    assert result.inference["lags"] == 3 and result.inference["kernel"] == "bartlett"
    shuffled = data.sample(frac=1.0, random_state=3)
    again = iv(shuffled, covariance="hac", lags=3, time="t", small=small)
    assert_allclose(errors(again), errors(result), rtol=1e-10)
    parzen = iv(data, covariance="hac", lags=3, kernel="parzen", time="t")
    assert parzen.inference["kernel"] == "parzen"
    assert not np.allclose(errors(parzen), errors(result))


# ---- weights ---------------------------------------------------------------------------------


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_analytic_weights_are_normalized_to_the_sample_size(data, covariance):
    options = {"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance}
    result = iv(data, weights="aw", weight_type="aweight", **options)
    y, x1, x2, z2 = blocks(data)
    n = len(y)
    w = data.aw.to_numpy() * n / data.aw.sum()
    b, u, bread, xhat = k_class(y, x1, x2, z2, w)
    assert_allclose(estimates(result), b, rtol=1e-10)
    scores = xhat * (w * u)[:, None]
    expected = {"nonrobust": bread * (w * u * u).sum() / n,
                "robust": bread @ (scores.T @ scores) @ bread,
                "cluster": bread @ cluster_meat(scores, data.firm) @ bread * 40 / 39}[covariance]
    assert_allclose(np.asarray(result.covariance_matrix), expected, rtol=1e-10)
    ybar = (w * y).sum() / n
    assert result.metrics["r_squared"] == pytest.approx(
        1 - (w * u * u).sum() / (w * (y - ybar) ** 2).sum(), rel=1e-11)
    assert result.nobs == n


@pytest.mark.parametrize("method", ["2sls", "liml", "gmm"])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_frequency_weights_equal_the_duplicated_rows(data, method, covariance):
    options = {"method": method}
    options.update({"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance})
    weighted = iv(data, weights="fw", weight_type="fweight", **options)
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    duplicated = iv(expanded, **options)
    assert weighted.nobs == duplicated.nobs == int(data.fw.sum())
    assert_allclose(estimates(weighted), estimates(duplicated), rtol=1e-9)
    assert_allclose(np.asarray(weighted.covariance_matrix),
                    np.asarray(duplicated.covariance_matrix), rtol=1e-8)
    for name, value in duplicated.metrics.items():
        assert weighted.metrics[name] == pytest.approx(value, rel=1e-8), name
    for name, test in duplicated.tests.items():
        assert weighted.tests[name]["statistic"] == pytest.approx(test["statistic"], rel=1e-7), name
        assert weighted.tests[name]["df"] == test["df"]
    for mine, theirs in zip(weighted.extra["first_stage"], duplicated.extra["first_stage"]):
        for key in ("f_statistic", "r_squared", "partial_r_squared", "shea_partial_r_squared"):
            assert mine[key] == pytest.approx(theirs[key], rel=1e-8), key


def test_sampling_weights_need_and_default_to_a_robust_covariance(data):
    result = iv(data, weights="aw", weight_type="pweight")
    assert result.spec.covariance == "robust"
    same = iv(data, weights="aw", weight_type="aweight", covariance="robust")
    assert_allclose(errors(result), errors(same), rtol=1e-12)
    with pytest.raises(AnalysisError) as error:
        iv(data, weights="aw", weight_type="pweight", covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"
    with pytest.raises((ValidationError, AnalysisError)):
        iv(data, weights="aw", weight_type="iweight")
    bad = data.assign(aw=-data.aw)
    with pytest.raises(AnalysisError) as error:
        iv(bad, weights="aw", weight_type="aweight")
    assert error.value.code == "negative_weights"


# ---- LIML ------------------------------------------------------------------------------------


def liml_kappa(y, x1, x2, z2, w=None):
    n = len(y)
    w = np.ones(n) if w is None else w
    big = np.column_stack([y, x2])
    m_z = np.eye(n) - projector(np.column_stack([x1, z2]), w)
    m_1 = np.eye(n) - projector(x1, w) if x1.shape[1] else np.eye(n)
    inner, outer = big.T @ (w[:, None] * m_z) @ big, big.T @ (w[:, None] * m_1) @ big
    return np.linalg.eigvals(np.linalg.solve(inner, outer)).real.min()


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_liml_is_the_k_class_estimator_at_the_smallest_eigenvalue(data, covariance):
    options = {"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance}
    result = iv(data, method="liml", **options)
    y, x1, x2, z2 = blocks(data)
    n = len(y)
    kappa = liml_kappa(y, x1, x2, z2)
    b, u, bread, xk = k_class(y, x1, x2, z2, kappa=kappa)
    assert kappa > 1 and result.metrics["kappa"] == pytest.approx(kappa, rel=1e-11)
    assert_allclose(estimates(result), b, rtol=1e-9)
    scores = xk * u[:, None]
    expected = {"nonrobust": bread * (u @ u) / n, "robust": bread @ (scores.T @ scores) @ bread,
                "cluster": bread @ cluster_meat(scores, data.firm) @ bread * 40 / 39}[covariance]
    assert_allclose(np.asarray(result.covariance_matrix), expected, rtol=1e-9)
    assert list(result.metrics)[:4] == ["r_squared", "adjusted_r_squared", "rmse", "kappa"]
    assert result.title == "Instrumental-variables (LIML) regression"
    # estat overid after liml: Anderson-Rubin LR and Basmann F.
    ar, basmann = result.tests["anderson_rubin"], result.tests["basmann_f"]
    assert ar["statistic"] == pytest.approx(n * np.log(kappa), rel=1e-9) and ar["df"] == 1
    assert ar["p_value"] == pytest.approx(stats.chi2.sf(n * np.log(kappa), 1), rel=1e-8)
    assert basmann["statistic"] == pytest.approx((kappa - 1) * (n - 5) / 1, rel=1e-9)
    assert (basmann["df"], basmann["df2"]) == (1, n - 5)
    assert basmann["p_value"] == pytest.approx(stats.f.sf(basmann["statistic"], 1, n - 5),
                                               rel=1e-8)
    assert not any(name.startswith("endog") for name in result.tests)


def test_liml_with_weights_and_exact_identification(data):
    weighted = iv(data, method="liml", weights="aw", weight_type="aweight")
    y, x1, x2, z2 = blocks(data)
    w = data.aw.to_numpy() * len(y) / data.aw.sum()
    kappa = liml_kappa(y, x1, x2, z2, w)
    assert weighted.metrics["kappa"] == pytest.approx(kappa, rel=1e-10)
    assert_allclose(estimates(weighted), k_class(y, x1, x2, z2, w, kappa)[0], rtol=1e-9)
    # Exactly identified: kappa = 1 and LIML is 2SLS.
    exact = {"endog": ["p"], "instruments": ["z1"]}
    liml, tsls = iv(data, method="liml", **exact), iv(data, **exact)
    assert liml.metrics["kappa"] == 1.0
    assert_allclose(estimates(liml), estimates(tsls), rtol=1e-12)
    assert_allclose(errors(liml), errors(tsls), rtol=1e-12)
    assert liml.tests["anderson_rubin"]["statistic"] is None
    assert "exactly identified" in liml.tests["anderson_rubin"]["note"]


# ---- GMM -------------------------------------------------------------------------------------


def gmm(y, x1, x2, z2, meat, *, w=None, steps=1, center=False):
    """Linear GMM by brute force: 2SLS, then `steps` updates of S from the previous residuals.

    Returns (b, final residuals, S used, (A'S^-1 A)^-1, J, residuals S was built from).
    """
    n = len(y)
    w = np.ones(n) if w is None else w
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    a, c = z.T @ (x * w[:, None]), z.T @ (w * y)
    u = k_class(y, x1, x2, z2, w)[1]
    for _ in range(steps):
        rows = z * (w * u)[:, None]
        if center:
            rows = rows - rows.mean(axis=0)
        s, used = meat(rows), u
        bread = np.linalg.inv(a.T @ np.linalg.solve(s, a))
        b = bread @ (a.T @ np.linalg.solve(s, c))
        u = y - x @ b
    g = z.T @ (w * u)
    return b, u, s, bread, g @ np.linalg.solve(s, g), used


def white(rows):
    return rows.T @ rows


def test_two_step_gmm_with_a_robust_weight_matrix(data):
    result = iv(data, method="gmm")
    y, x1, x2, z2 = blocks(data)
    b, u, s, bread, j, _ = gmm(y, x1, x2, z2, white)
    n = len(y)
    assert result.spec.covariance == "robust" and result.extra["gmm"]["wmatrix"] == "robust"
    assert_allclose(estimates(result), b, rtol=1e-9)
    # Efficient GMM covariance (X'Z S^-1 Z'X)^-1, no small-sample factor.
    assert_allclose(np.asarray(result.covariance_matrix), bread, rtol=1e-8)
    test = result.tests["hansen_j"]
    assert test["statistic"] == pytest.approx(j, rel=1e-8) and test["df"] == 1
    assert test["p_value"] == pytest.approx(stats.chi2.sf(j, 1), rel=1e-7)
    assert result.metrics["j"] == pytest.approx(j, rel=1e-8)
    assert result.metrics["gmm_iterations"] == 1 and result.extra["gmm"]["converged"]
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(u @ u / n), rel=1e-10)
    assert result.title == "Instrumental-variables (GMM) regression"
    assert "efficient GMM" in result.inference["correction"]
    # The estimate minimizes the GMM criterion: brute-force minimization agrees.
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])

    def criterion(beta):
        g = z.T @ (y - x @ beta)
        return g @ np.linalg.solve(s, g)

    brute = optimize.minimize(criterion, np.zeros(4), method="BFGS", options={"gtol": 1e-9})
    assert_allclose(estimates(result), brute.x, rtol=1e-5, atol=1e-6)
    small = iv(data, method="gmm", small=True)
    assert_allclose(np.asarray(small.covariance_matrix), bread * n / (n - 4), rtol=1e-8)
    assert small.inference["use_t"] and small.tests["model"]["distribution"] == "F"


def test_gmm_matches_statsmodels_linear_gmm_and_supports_center_and_iteration(data):
    from statsmodels.sandbox.regression.gmm import LinearIVGMM

    y, x1, x2, z2 = blocks(data)
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    # statsmodels' default centering subtracts one overall mean; compare uncentered.
    options = {"wargs": {"centered": False}, "optim_args": {"disp": 0}}
    reference = LinearIVGMM(y, x, z).fit(maxiter=2, **options)
    two_step = iv(data, method="gmm")
    assert_allclose(estimates(two_step), reference.params, rtol=1e-8)
    assert_allclose(errors(two_step), reference.bse, rtol=2e-2)   # S at the final residuals
    centered = iv(data, method="gmm", center=True)
    b, _, _, bread, j, _ = gmm(y, x1, x2, z2, white, center=True)
    assert_allclose(estimates(centered), b, rtol=1e-9)
    assert centered.metrics["j"] == pytest.approx(j, rel=1e-8)
    assert_allclose(np.asarray(centered.covariance_matrix), bread, rtol=1e-8)
    assert centered.extra["gmm"]["centered"] is True
    assert not np.allclose(estimates(centered), estimates(two_step), rtol=1e-7)
    # Iterated GMM: a fixed point of the weight-matrix update.
    iterated = iv(data, method="gmm", igmm=True)
    fixed = gmm(y, x1, x2, z2, white, steps=60)
    assert_allclose(estimates(iterated), fixed[0], rtol=1e-8)
    assert_allclose(np.asarray(iterated.covariance_matrix), fixed[3], rtol=1e-7)
    assert_allclose(estimates(iterated),
                    LinearIVGMM(y, x, z).fit(maxiter=200, **options).params, rtol=1e-5)
    assert iterated.metrics["gmm_iterations"] > 2 and iterated.extra["gmm"]["igmm"]
    assert iterated.provenance["optimizer"]["method"] == "iterated GMM"
    assert iterated.provenance["optimizer"]["converged"] is True
    assert not np.allclose(estimates(iterated), estimates(two_step), rtol=1e-6)


def test_gmm_cluster_and_hac_weight_matrices(data):
    y, x1, x2, z2 = blocks(data)
    clustered = iv(data, method="gmm", cluster="firm")
    b, _, _, bread, j, _ = gmm(y, x1, x2, z2, lambda rows: cluster_meat(rows, data.firm))
    assert clustered.extra["gmm"]["wmatrix"] == "cluster"
    assert_allclose(estimates(clustered), b, rtol=1e-9)
    assert_allclose(np.asarray(clustered.covariance_matrix), bread * 40 / 39, rtol=1e-8)
    assert clustered.tests["hansen_j"]["statistic"] == pytest.approx(j, rel=1e-8)
    hac = iv(data, method="gmm", wmatrix="hac", lags=2, time="t")
    b, _, _, bread, j, _ = gmm(y, x1, x2, z2, lambda rows: bartlett_meat(rows, 2))
    assert hac.spec.covariance == "hac"
    assert_allclose(estimates(hac), b, rtol=1e-9)
    assert_allclose(np.asarray(hac.covariance_matrix), bread, rtol=1e-8)
    assert hac.tests["hansen_j"]["statistic"] == pytest.approx(j, rel=1e-8)
    weighted = iv(data, method="gmm", weights="aw", weight_type="aweight", covariance="robust")
    w = data.aw.to_numpy() * len(y) / data.aw.sum()
    b, _, _, bread, j, _ = gmm(y, x1, x2, z2, white, w=w)
    assert_allclose(estimates(weighted), b, rtol=1e-9)
    assert_allclose(np.asarray(weighted.covariance_matrix), bread, rtol=1e-8)
    assert weighted.metrics["j"] == pytest.approx(j, rel=1e-8)


def test_gmm_sandwich_when_the_covariance_differs_from_the_weight_matrix(data):
    y, x1, x2, z2 = blocks(data)
    n = len(y)
    b, u, s, bread, _, _ = gmm(y, x1, x2, z2, white)
    z = np.column_stack([x1, z2])
    a = z.T @ np.column_stack([x1, x2])
    loading = np.linalg.solve(s, a)                       # S^-1 A
    clustered = iv(data, method="gmm", wmatrix="robust", cluster="firm")
    meat = cluster_meat(z * u[:, None], data.firm)
    assert clustered.extra["gmm"]["wmatrix"] == "robust"
    assert_allclose(estimates(clustered), b, rtol=1e-9)
    assert_allclose(np.asarray(clustered.covariance_matrix),
                    bread @ loading.T @ meat @ loading @ bread * 40 / 39, rtol=1e-8)
    assert clustered.inference["correction"].startswith("GMM sandwich")
    unadjusted = iv(data, method="gmm", covariance="unadjusted")
    assert unadjusted.spec.covariance == "nonrobust"
    assert unadjusted.extra["gmm"]["wmatrix"] == "robust"
    assert_allclose(np.asarray(unadjusted.covariance_matrix),
                    bread @ loading.T @ (z.T @ z) @ loading @ bread * (u @ u) / n, rtol=1e-8)


def test_gmm_with_unadjusted_weights_and_exact_identification_is_2sls(data):
    tsls = iv(data)
    unadjusted = iv(data, method="gmm", wmatrix="unadjusted")
    assert unadjusted.spec.covariance == "nonrobust"
    assert_allclose(estimates(unadjusted), estimates(tsls), rtol=1e-12)
    assert_allclose(errors(unadjusted), errors(tsls), rtol=1e-12)
    assert unadjusted.tests["hansen_j"]["statistic"] == pytest.approx(
        tsls.tests["overid_sargan"]["statistic"], rel=1e-10)
    exact = {"endog": ["p"], "instruments": ["z1"]}
    for options in ({}, {"igmm": True}, {"cluster": "firm"}):
        result = iv(data, method="gmm", **exact, **options)
        assert_allclose(estimates(result), estimates(iv(data, **exact)), rtol=1e-9)
        assert result.tests["hansen_j"]["statistic"] is None
        assert abs(result.metrics["j"]) < 1e-12
    robust = iv(data, method="gmm", **exact)
    assert_allclose(errors(robust), errors(iv(data, covariance="robust", **exact)), rtol=1e-8)


def test_gmm_rejects_a_singular_weight_matrix_and_misplaced_options(data):
    few = data.assign(trio=np.arange(len(data)) % 3)
    with pytest.raises(AnalysisError) as error:
        iv(few, method="gmm", cluster="trio")
    assert error.value.code == "singular_weight_matrix"
    assert "fewer clusters than instruments" in str(error.value)
    for options in ({"wmatrix": "robust"}, {"igmm": True}, {"center": True}):
        with pytest.raises(AnalysisError) as error:
            iv(data, **options)
        assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        iv(data, method="gmm", wmatrix="cluster")
    assert error.value.code == "invalid_spec" and "cluster=" in str(error.value)
    with pytest.raises((ValidationError, AnalysisError)):
        iv(data, method="gmm", wmatrix="optimal")
    with pytest.raises(AnalysisError) as error:
        iv(data, covariance="hac")
    assert error.value.code == "invalid_spec" and "lags" in str(error.value)
    with pytest.raises(AnalysisError) as error:
        iv(data, lags=2)
    assert error.value.code == "invalid_spec"


# ---- first stage and weak identification -----------------------------------------------------


def residualize(a, b):
    return a - b @ np.linalg.lstsq(b, a, rcond=None)[0]


def test_first_stage_statistics_match_statsmodels_and_their_definitions(data):
    result = iv(data)
    y, x1, x2, z2 = blocks(data)
    n = len(y)
    z, x = np.column_stack([x1, z2]), np.column_stack([x1, x2])
    xhat = projector(z) @ x
    restriction = np.hstack([np.zeros((3, 2)), np.eye(3)])
    assert [row["term"] for row in result.extra["first_stage"]] == ["p", "p2"]
    for j, row in enumerate(result.extra["first_stage"]):
        full = sm.OLS(x2[:, j], z).fit()
        test = full.f_test(restriction)
        assert row["f_statistic"] == pytest.approx(float(test.fvalue), rel=1e-9)
        assert (row["df"], row["df2"]) == (3, n - 5) and row["covariance"] == "nonrobust"
        assert row["p_value"] == pytest.approx(float(test.pvalue), rel=1e-7)
        assert row["r_squared"] == pytest.approx(full.rsquared, rel=1e-10)
        assert row["adjusted_r_squared"] == pytest.approx(full.rsquared_adj, rel=1e-10)
        restricted = sm.OLS(x2[:, j], x1).fit()
        assert row["partial_r_squared"] == pytest.approx(1 - full.ssr / restricted.ssr, rel=1e-10)
        # Shea: squared correlation of x_j and xhat_j, each purged of the other regressors.
        others = [i for i in range(4) if i != 2 + j]
        a = residualize(x[:, 2 + j], x[:, others])
        b = residualize(xhat[:, 2 + j], xhat[:, others])
        assert row["shea_partial_r_squared"] == pytest.approx(
            (a @ b) ** 2 / ((a @ a) * (b @ b)), rel=1e-9)
    single = iv(data, endog=["p"], instruments=["z1", "z2"]).extra["first_stage"][0]
    assert single["shea_partial_r_squared"] == pytest.approx(single["partial_r_squared"], rel=1e-10)


@pytest.mark.parametrize("covariance", ["robust", "cluster", "hac"])
def test_first_stage_f_follows_the_model_covariance(data, covariance):
    options = {"robust": {"covariance": "robust"}, "cluster": {"cluster": "firm"},
               "hac": {"covariance": "hac", "lags": 2, "time": "t"}}[covariance]
    result = iv(data, **options)
    _, x1, x2, z2 = blocks(data)
    z = np.column_stack([x1, z2])
    restriction = np.hstack([np.zeros((3, 2)), np.eye(3)])
    fit_options = {"robust": {"cov_type": "HC1"},
                   "cluster": {"cov_type": "cluster", "cov_kwds": {"groups": data.firm}},
                   "hac": {"cov_type": "HAC", "cov_kwds": {"maxlags": 2, "use_correction": True}}}
    for j, row in enumerate(result.extra["first_stage"]):
        reference = sm.OLS(x2[:, j], z).fit(**fit_options[covariance])
        wald = reference.wald_test(restriction, use_f=True, scalar=True)
        assert row["f_statistic"] == pytest.approx(float(wald.statistic), rel=1e-8)
        assert row["df"] == 3 and row["covariance"] == covariance
        assert row["df2"] == (39 if covariance == "cluster" else len(data) - 5)
        assert row["p_value"] == pytest.approx(
            stats.f.sf(row["f_statistic"], 3, row["df2"]), rel=1e-7, abs=1e-300)


def rank_statistic(x2p, z2p, e, meat):
    """Kleibergen-Paap rk statistic for rank q - 1, in the full Kronecker form of the paper."""
    from scipy.linalg import inv, sqrtm

    n, q = x2p.shape
    width = z2p.shape[1]
    fy, fz = inv(sqrtm(x2p.T @ x2p)).real, inv(sqrtm(z2p.T @ z2p)).real
    theta = fy @ x2p.T @ z2p @ fz
    u, _, vt = np.linalg.svd(theta)
    v, r = vt.T, q - 1
    u12, u22, v12, v22 = u[:r, r:], u[r:, r:], v[:r, r:], v[r:, r:]
    a_perp = np.vstack([u12, u22]) @ inv(u22) @ sqrtm(u22 @ u22.T).real
    b_perp = sqrtm(v22 @ v22.T).real @ inv(v22.T) @ np.hstack([v12.T, v22.T])
    select = np.kron(b_perp, a_perp.T)
    lam = select @ theta.flatten(order="F")
    rows = np.einsum("il,ik->ilk", z2p @ fz, e @ fy).reshape(n, width * q)
    return lam @ np.linalg.solve(select @ meat(rows) @ select.T, lam)


def test_cragg_donald_and_kleibergen_paap_weak_identification_statistics(data):
    y, x1, x2, z2 = blocks(data)
    n, width = len(y), 3
    z = np.column_stack([x1, z2])
    m_z, m_1 = np.eye(n) - projector(z), np.eye(n) - projector(x1)
    e, x2p, z2p = m_z @ x2, m_1 @ x2, m_1 @ z2
    eigenvalue = np.linalg.eigvals(np.linalg.solve(x2.T @ m_z @ x2,
                                                   x2.T @ (m_1 - m_z) @ x2)).real.min()
    cragg_donald = eigenvalue * (n - 5) / width
    plain = iv(data)
    test = plain.tests["cragg_donald"]
    assert test["statistic"] == pytest.approx(cragg_donald, rel=1e-9)
    assert (test["df"], test["df2"], test["p_value"]) == (width, n - 5, None)
    assert "kleibergen_paap_rk_f" not in plain.tests
    # With the homoskedastic moment covariance the rk statistic is N times that eigenvalue.
    homoskedastic = rank_statistic(x2p, z2p, e, lambda rows: _homoskedastic(x2p, z2p, e))
    assert homoskedastic == pytest.approx(n * eigenvalue, rel=1e-8)
    robust = iv(data, covariance="robust").tests["kleibergen_paap_rk_f"]
    rk = rank_statistic(x2p, z2p, e, white)
    assert robust["rk_wald_chi2"] == pytest.approx(rk, rel=1e-8) and robust["rk_df"] == 2
    assert robust["statistic"] == pytest.approx(rk / n * (n - 5) / width, rel=1e-8)
    assert (robust["df"], robust["df2"]) == (width, n - 5)
    clustered = iv(data, cluster="firm").tests["kleibergen_paap_rk_f"]
    rk = rank_statistic(x2p, z2p, e, lambda rows: cluster_meat(rows, data.firm))
    assert clustered["statistic"] == pytest.approx(
        rk / (n - 1) * (n - 5) * 39 / 40 / width, rel=1e-8)
    # One endogenous regressor: Cragg-Donald is the first-stage F, Kleibergen-Paap the robust F.
    one = {"endog": ["p"], "instruments": ["z1", "z2"]}
    single = iv(data, **one)
    assert single.tests["cragg_donald"]["statistic"] == pytest.approx(
        single.extra["first_stage"][0]["f_statistic"], rel=1e-9)
    for options in ({"covariance": "robust"}, {"cluster": "firm"}):
        single = iv(data, **one, **options)
        assert single.tests["kleibergen_paap_rk_f"]["statistic"] == pytest.approx(
            single.extra["first_stage"][0]["f_statistic"], rel=1e-8)


def _homoskedastic(x2p, z2p, e):
    """Var(vec theta) under homoskedasticity: I (x) F_y' (E'E/n) F_y summed over observations."""
    from scipy.linalg import inv, sqrtm

    fy = inv(sqrtm(x2p.T @ x2p)).real
    return np.kron(np.eye(z2p.shape[1]), fy @ (e.T @ e) @ fy / len(e))


# ---- overidentification ----------------------------------------------------------------------


def test_sargan_and_basmann_overidentification_tests(data):
    result = iv(data)
    y, x1, x2, z2 = blocks(data)
    n = len(y)
    z = np.column_stack([x1, z2])
    u = k_class(y, x1, x2, z2)[1]
    auxiliary = sm.OLS(u, z).fit()
    sargan, basmann = result.tests["overid_sargan"], result.tests["overid_basmann"]
    assert sargan["statistic"] == pytest.approx(n * auxiliary.rsquared, rel=1e-8)
    explained = u @ projector(z) @ u
    assert basmann["statistic"] == pytest.approx(
        (n - 5) * explained / (u @ u - explained), rel=1e-8)
    for test in (sargan, basmann):
        assert test["df"] == 1 and test["distribution"] == "chi2"
        assert test["p_value"] == pytest.approx(stats.chi2.sf(test["statistic"], 1), rel=1e-7)
    assert "overid_score" not in result.tests
    exact = iv(data, endog=["p"], instruments=["z1"])
    assert exact.tests["overid_sargan"]["statistic"] is None
    assert "exactly identified" in exact.tests["overid_sargan"]["note"]


def test_wooldridge_robust_score_overidentification_test(data):
    y, x1, x2, z2 = blocks(data)
    n = len(y)
    z = np.column_stack([x1, z2])
    xhat = projector(z) @ np.column_stack([x1, x2])
    u = k_class(y, x1, x2, z2)[1]
    # Stata: regress q of the instruments on Xhat, then 1 on u * residual; N - RSS.
    for column in range(3):
        k_hat = residualize(z2[:, column], xhat)
        auxiliary = sm.OLS(np.ones(n), (u * k_hat)[:, None]).fit()
        robust = iv(data, covariance="robust").tests["overid_score"]
        assert robust["statistic"] == pytest.approx(n - auxiliary.ssr, rel=1e-7)
    assert robust["df"] == 1
    assert robust["p_value"] == pytest.approx(stats.chi2.sf(robust["statistic"], 1), rel=1e-7)
    clustered = iv(data, cluster="firm").tests["overid_score"]
    rows = (u * k_hat)[:, None]
    total = rows.sum(axis=0)
    assert clustered["statistic"] == pytest.approx(
        float(total @ np.linalg.solve(cluster_meat(rows, data.firm), total)), rel=1e-7)
    two = iv(data, endog=["p"], instruments=["z1", "z2", "z3"], covariance="robust")
    assert two.tests["overid_score"]["df"] == 2
    xhat = projector(z) @ np.column_stack([x1, x2[:, 0]])
    u = k_class(y, x1, x2[:, :1], z2)[1]
    k_hat = residualize(z2[:, :2], xhat)
    auxiliary = sm.OLS(np.ones(n), k_hat * u[:, None]).fit()
    assert two.tests["overid_score"]["statistic"] == pytest.approx(n - auxiliary.ssr, rel=1e-7)


# ---- endogeneity -----------------------------------------------------------------------------


def test_durbin_and_wu_hausman_endogeneity_tests(data):
    result = iv(data)
    y, x1, x2, z2 = blocks(data)
    n, k, q = len(y), 4, 2
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    u_e = residualize(y, x)                               # endogenous treated as exogenous
    u_c = k_class(y, x1, x2, z2)[1]
    # Stata's Methods and formulas: quadratic forms in the two residual vectors.
    gap = u_e @ projector(np.column_stack([z, x2])) @ u_e - u_c @ projector(z) @ u_c
    durbin, wu = result.tests["endog_durbin"], result.tests["endog_wu_hausman"]
    assert durbin["statistic"] == pytest.approx(gap / (u_e @ u_e / n), rel=1e-7)
    assert durbin["df"] == q
    assert durbin["p_value"] == pytest.approx(stats.chi2.sf(durbin["statistic"], q), rel=1e-7)
    expected = (gap / q) / ((u_e @ u_e - gap) / (n - k - q))
    assert wu["statistic"] == pytest.approx(expected, rel=1e-7)
    assert (wu["df"], wu["df2"]) == (q, n - k - q)
    # Equivalently the F test of the first-stage residuals in the augmented regression.
    v_hat = residualize(x2, z)
    augmented = sm.OLS(y, np.column_stack([x, v_hat])).fit()
    f = augmented.f_test(np.hstack([np.zeros((q, k)), np.eye(q)]))
    assert wu["statistic"] == pytest.approx(float(f.fvalue), rel=1e-8)
    assert wu["p_value"] == pytest.approx(float(f.pvalue), rel=1e-7)


@pytest.mark.parametrize("covariance", ["robust", "cluster"])
def test_robust_endogeneity_tests(data, covariance):
    options = {"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance}
    result = iv(data, **options)
    y, x1, x2, z2 = blocks(data)
    n, k, q = len(y), 4, 2
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    v_hat = residualize(x2, z)
    u_e, r = residualize(y, x), residualize(v_hat, x)
    rows = r * u_e[:, None]
    score = result.tests["endog_robust_score"]
    if covariance == "robust":
        # Wooldridge: N - RSS from regressing 1 on u * r.
        assert score["statistic"] == pytest.approx(n - sm.OLS(np.ones(n), rows).fit().ssr,
                                                   rel=1e-7)
        reference = sm.OLS(y, np.column_stack([x, v_hat])).fit(cov_type="HC1")
        df2 = n - k - q
    else:
        total = rows.sum(axis=0)
        assert score["statistic"] == pytest.approx(
            float(total @ np.linalg.solve(cluster_meat(rows, data.firm), total)), rel=1e-7)
        reference = sm.OLS(y, np.column_stack([x, v_hat])).fit(
            cov_type="cluster", cov_kwds={"groups": data.firm})
        df2 = 39
    assert score["df"] == q and score["distribution"] == "chi2"
    wald = reference.wald_test(np.hstack([np.zeros((q, k)), np.eye(q)]), use_f=True, scalar=True)
    regression = result.tests["endog_robust_regression"]
    assert regression["statistic"] == pytest.approx(float(wald.statistic), rel=1e-8)
    assert (regression["df"], regression["df2"]) == (q, df2)
    assert regression["p_value"] == pytest.approx(
        stats.f.sf(regression["statistic"], q, df2), rel=1e-7, abs=1e-300)
    assert "endog_durbin" not in result.tests


def test_gmm_c_statistic_is_the_difference_of_two_j_statistics(data):
    result = iv(data, method="gmm")
    y, x1, x2, z2 = blocks(data)
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    wide = np.column_stack([z, x2])
    u_ols = residualize(y, x)
    s = (wide * u_ols[:, None]).T @ (wide * u_ols[:, None])

    def j_statistic(instruments, cov):
        a, c = instruments.T @ x, instruments.T @ y
        b = np.linalg.solve(a.T @ np.linalg.solve(cov, a), a.T @ np.linalg.solve(cov, c))
        g = instruments.T @ (y - x @ b)
        return g @ np.linalg.solve(cov, g)

    c_statistic = j_statistic(wide, s) - j_statistic(z, s[:5, :5])
    test = result.tests["endog_c"]
    assert test["statistic"] == pytest.approx(c_statistic, rel=1e-7) and test["df"] == 2
    assert test["p_value"] == pytest.approx(stats.chi2.sf(c_statistic, 2), rel=1e-7, abs=1e-300)
    assert c_statistic > 0 and "endog_durbin" not in result.tests


# ---- model structure -------------------------------------------------------------------------


def test_instruments_equal_to_the_regressors_reproduce_ols(data):
    copy = data.assign(c1=data.p, c2=data.p2)
    result = iv(copy, instruments=["c1", "c2"], small=True)
    reference = sm.OLS(data.y, sm.add_constant(data[["x1", "p", "p2"]])).fit()
    assert_allclose(estimates(result), reference.params.values, rtol=1e-10)
    assert_allclose(errors(result), reference.bse.values, rtol=1e-10)
    assert result.metrics["r_squared"] == pytest.approx(reference.rsquared, rel=1e-10)
    # No first-stage residual: endogeneity and weak-identification statistics are undefined.
    for name in ("endog_durbin", "endog_wu_hausman", "cragg_donald"):
        assert result.tests[name]["statistic"] is None and "exactly" in result.tests[name]["note"]
    # Exactly identified LIML is 2SLS (kappa = 1) even when the first stage is exact ...
    liml = iv(copy, instruments=["c1", "c2"], method="liml", small=True)
    assert liml.metrics["kappa"] == 1.0
    assert_allclose(estimates(liml), reference.params.values, rtol=1e-10)
    assert_allclose(errors(liml), reference.bse.values, rtol=1e-10)
    # ... while an overidentified exact first stage leaves the LIML eigenvalue undefined.
    with pytest.raises(AnalysisError) as error:
        iv(copy, instruments=["c1", "c2", "z1"], method="liml")
    assert error.value.code == "perfect_first_stage"


def test_without_intercept_and_without_exogenous_regressors(data):
    result = iv(data, intercept=False)
    y, x1, x2, z2 = blocks(data, intercept=False)
    b, u, bread, _ = k_class(y, x1, x2, z2)
    n = len(y)
    assert [c.term for c in result.coefficients] == ["x1", "p", "p2"]
    assert_allclose(estimates(result), b, rtol=1e-10)
    assert_allclose(np.asarray(result.covariance_matrix), bread * (u @ u) / n, rtol=1e-10)
    assert result.metrics["r_squared"] == pytest.approx(1 - u @ u / (y @ y), rel=1e-12)
    assert result.inference["r_squared_definition"] == "uncentered"
    assert result.tests["model"]["df"] == 3 and result.metrics["df_model"] == 3
    for method in ("2sls", "liml", "gmm"):
        constant_only = oe.ivregress(data=data, y="y", endog=ENDOG, instruments=INST, method=method)
        assert [c.term for c in constant_only.coefficients] == ["Intercept", "p", "p2"]
        bare = oe.ivregress(data=data, y="y", endog=ENDOG, instruments=INST, method=method,
                            intercept=False)
        assert [c.term for c in bare.coefficients] == ["p", "p2"]
    y, x1, x2, z2 = blocks(data, x=[])
    constant_only = oe.ivregress(data=data, y="y", endog=ENDOG, instruments=INST)
    assert_allclose(estimates(constant_only), k_class(y, x1, x2, z2)[0], rtol=1e-10)
    y, x1, x2, z2 = blocks(data, x=[], intercept=False)
    bare = oe.ivregress(data=data, y="y", endog=ENDOG, instruments=INST, intercept=False,
                        method="liml")
    kappa = liml_kappa(y, x1, x2, z2)
    assert_allclose(estimates(bare), k_class(y, x1, x2, z2, kappa=kappa)[0], rtol=1e-9)
    assert bare.metrics["kappa"] == pytest.approx(kappa, rel=1e-10)


def test_categorical_exogenous_regressors_are_expanded(data):
    result = iv(data, x=["x1", "sector"], categorical=["sector"])
    dummies = pd.get_dummies(data.sector, drop_first=True, dtype=float).to_numpy()
    y, _, x2, z2 = blocks(data)
    x1 = np.column_stack([np.ones(len(y)), data.x1, dummies])
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "sector[b]", "sector[c]", "p", "p2"]
    assert_allclose(estimates(result), k_class(y, x1, x2, z2)[0], rtol=1e-10)
    assert result.provenance["categorical_encoding"]["sector"]["reference"] == "a"
    with pytest.raises(AnalysisError) as error:
        iv(data, categorical=["p"])
    assert error.value.code == "invalid_spec" and "exogenous" in str(error.value)


def test_collinear_regressors_and_instruments_are_omitted_and_recorded(data):
    frame = data.assign(x1_twice=2 * data.x1, z_sum=data.z1 + data.z2, z_exog=3 * data.x1 + 1,
                        p_exog=data.x1 - 2)
    result = iv(frame, x=["x1", "x1_twice"], instruments=[*INST, "z_sum", "z_exog"])
    assert_allclose(estimates(result), estimates(iv(data)), rtol=1e-9)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "p", "p2"]
    assert result.provenance["omitted_terms"] == ["x1_twice"]
    assert result.extra["omitted_instruments"] == ["z_sum", "z_exog"]
    assert result.extra["instruments"] == INST and result.metrics["n_instruments"] == 3
    assert any("Omitted because of collinearity: x1_twice" in w for w in result.warnings)
    assert any("Instrument(s) omitted" in w and "z_sum, z_exog" in w for w in result.warnings)
    # A redundant endogenous regressor is omitted like any other regressor.
    dropped = iv(frame, endog=["p", "p_exog"], instruments=INST)
    assert dropped.provenance["omitted_terms"] == ["p_exog"]
    assert dropped.extra["endogenous"] == ["p"] and dropped.metrics["n_endogenous"] == 1
    with pytest.raises(AnalysisError) as error:
        iv(frame, endog=["p_exog"], instruments=INST)
    assert error.value.code == "no_endogenous_regressors"
    # The order condition is checked after the screens.
    with pytest.raises(AnalysisError) as error:
        iv(frame, instruments=["z1", "z_exog"])
    assert error.value.code == "underidentified" and "order condition" in str(error.value)
    with pytest.raises(AnalysisError) as error:
        iv(data, instruments=["z1"])
    assert error.value.code == "underidentified"


def test_rank_condition_failure_is_reported(data):
    y, x1, x2, _ = blocks(data)
    # Instruments orthogonal to both endogenous regressors given X1: nothing is identified.
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(len(y), 3))
    orthogonal = residualize(noise, np.column_stack([x1, x2]))
    frame = data.assign(o1=orthogonal[:, 0], o2=orthogonal[:, 1], o3=orthogonal[:, 2])
    for method in ("2sls", "liml", "gmm"):
        with pytest.raises(AnalysisError) as error:
            iv(frame, instruments=["o1", "o2", "o3"], method=method)
        assert error.value.code == "underidentified" and "rank condition" in str(error.value)


def test_missing_values_policy_and_sample_bookkeeping(data):
    frame = data.copy()
    frame.loc[[3, 17], "z2"] = np.nan
    frame.loc[40, "p"] = np.nan
    with pytest.raises(AnalysisError) as error:
        iv(frame)
    assert error.value.code == "missing_values"
    result = iv(frame, missing="drop")
    kept = frame.dropna(subset=["y", "x1", "p", "p2", "z1", "z2", "z3"])
    assert result.nobs == len(kept) == 397 and result.dropped_rows == 3
    assert_allclose(estimates(result), estimates(iv(kept.reset_index(drop=True))), rtol=1e-11)
    assert 3 not in result.sample_positions and 40 not in result.sample_positions
    assert any("Excluded 3 observation(s)" in warning for warning in result.warnings)


def test_invalid_specifications_raise_actionable_errors(data):
    cases = [
        ({"endog": ["p", "x1"]}, "both as exogenous"),
        ({"instruments": ["z1", "z2", "x1"]}, "instrument for itself"),
        ({"instruments": ["z1", "z2", "p"]}, "its own instrument"),
        ({"endog": ["p", "y"]}, "outcome"),
    ]
    for options, fragment in cases:
        with pytest.raises(AnalysisError) as error:
            iv(data, **options)
        assert error.value.code == "invalid_spec" and fragment in str(error.value)
    with pytest.raises(AnalysisError) as error:
        iv(data, endog="p")
    assert error.value.code == "invalid_spec"
    with pytest.raises((ValidationError, AnalysisError)):
        iv(data, endog=[])
    with pytest.raises((ValidationError, AnalysisError)):
        iv(data, method="3sls")
    with pytest.raises((ValidationError, AnalysisError)):
        iv(data, covariance="HC3")
    with pytest.raises((ValidationError, AnalysisError)):
        iv(data, cluster=["firm", "state", "t"])
    with pytest.raises(AnalysisError) as error:
        iv(data, instruments=["z1", "z2", "nope"])
    assert error.value.code == "missing_columns"
    with pytest.raises(AnalysisError) as error:
        iv(data.assign(z1=data.sector))
    assert error.value.code == "non_numeric_column"
    with pytest.raises(AnalysisError) as error:
        iv(data.head(5))
    assert error.value.code == "insufficient_observations"
    tiny = iv(data.head(6))                    # estimable, but too small for the diagnostics
    assert list(tiny.tests) == ["model"] and tiny.extra["first_stage"] == []
    assert any("diagnostics were not computed" in warning for warning in tiny.warnings)
    with pytest.raises(AnalysisError) as error:
        iv(data.assign(y=3.0))
    assert error.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as error:
        iv(data.assign(y=1 + 2 * data.p - data.x1 + 0.5 * data.p2))
    assert error.value.code == "perfect_fit"
    with pytest.raises(AnalysisError) as error:
        iv(data.assign(firm=1), cluster="firm")
    assert error.value.code == "insufficient_clusters"


def test_results_round_trip_render_and_fit_from_a_spec(data):
    result = iv(data, method="liml", cluster="firm", small=True)
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    json.dumps(result.model_dump())
    text = result.summary()
    for fragment in ("Instrumental-variables (LIML) regression", "Intercept", "kappa",
                     "Anderson-Rubin", "Cragg-Donald", "Kleibergen-Paap", "P>|stat|"):
        assert fragment in text
    latex = result.to_latex()
    assert "\\begin{tabular}" in str(latex) and "p2" in str(latex)
    spec = ModelSpec(estimator="ivregress", outcome="y", predictors=["x1"],
                     columns={"endogenous": ENDOG, "instruments": INST},
                     options={"method": "gmm", "wmatrix": "robust"}, covariance="robust")
    assert ModelSpec.model_validate(json.loads(spec.model_dump_json())) == spec
    direct = fit(spec, data=data)
    assert_allclose(estimates(direct), estimates(iv(data, method="gmm")), rtol=1e-13)
    assert direct.provenance["estimator"] == "ivregress" and direct.provenance["family"] == "iv"
    assert direct.provenance["stata_equivalent"][0] == "ivregress 2sls"
    gmm_result = iv(data, method="gmm", igmm=True)
    assert ResultBundle.model_validate_json(gmm_result.model_dump_json()) == gmm_result
    records = oe.ivregress(data=data.to_dict("records"), y="y", x=X, endog=ENDOG,
                           instruments=INST)
    assert_allclose(estimates(records), estimates(iv(data)), rtol=1e-13)
    assert iv(data, alpha=0.1).coefficients[2].ci_low > iv(data).coefficients[2].ci_low
